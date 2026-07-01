module energy_util
  use chm_kinds
  use dimens_fcm
  implicit none
  private

#if KEY_DOMDEC
  integer numnod_save, mynod_save
#endif

  ! Public subroutines
  public set_mtsflags, calc_dmcons, write_dmcons, calc_umbrella_potential, calc_adumb_potential
  public calc_dihe_restraints, calc_noe_restraints, calc_redcns, calc_dbias, calc_hmcm
  public zero_energy_terms
#if KEY_DOMDEC==1
  public energy_recip, init_auxdata, write_to_auxdata, read_from_auxdata, get_nauxdata
#endif 

contains

#if KEY_DOMDEC==1
subroutine energy_recip(x, y, z, dx, dy, dz, qsecd)
  use chm_kinds
  use dimens_fcm
  use energym,only:eterm, qeterm, ewksum, ewself, ewqcor, ewutil, umbr, dmc, adumb, &
       nauxdata, auxdata, cdihe, noe, resd, charm, hmcm
  use ewald,only:kspace, ewvirial
  use memory
  use psf,only:natom, cg, cgtot
#if KEY_LJPME==1
  use psf, only: ljc6
  use energym, only: ljrec
  use domdec_common, only: q_split
  use pmeutil, only: qljpme
#endif
#if KEY_FLUCQ==1
  use flucq,only:qfluc   
#endif
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor 
#endif
#if KEY_BLOCK==1
  use block_ltm,only:qblock,qhybh 
#endif
  use new_timer 
  use pme_module,only:qpme
#if KEY_PARALLEL==1
  use machutil,only:eclock  
#endif
  use parallel,only:tmeri, timexte, timinte
  use domdec_common,only:q_domdec, q_cons_node, ncons, conslist
  use domdec_r2d_comm,only:recv_coord_from_direct, comm_coord_among_recip, zero_recip_force, &
       comm_force_among_recip, send_results_to_direct
  use enbxfast,only:calc_virial_noshift
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop
  use nbrecip_gpu_mod,only:read_recip_energy_gpu, read_recip_virial_gpu
  use domdec_common, only: gpu_code_version, q_gpu, q_split
#endif
  use dmcons,only:ndmc            
  use number
  implicit none
  ! Input / Output parameters
  real(chm_real), intent(inout) :: x(*), y(*), z(*)
  real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
  logical qsecd
  ! Variables
  real(chm_real) dd1_dummy(1)
  integer iupt_dummy(1)
  real(chm_real) TIMMER
  real(chm_real) dmc_ener(ndmc), dmc_rho(ndmc), vpress(9)
  logical q_stop_recip
  integer nloop

  if (qsecd) call wrndie(-5,'<energy_util>','Second derivatives not supported in DOMDEC')

  if (.not.qpme) then
     call wrndie(-5,'<energy>','energy_recip: must use PME, how did you end up here?')
  endif

  vpress = 0d0

  call timer_start(T_energy)

  call set_mtsflags()

  call init_auxdata(nauxdata, auxdata)

  nloop = 0

  do while (.true.)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('recv_coord_from_direct')
#endif

     call timer_start(T_crdcomm3)  
     call recv_coord_from_direct(natom, x, y, z, q_stop_recip)
     call timer_stop(T_crdcomm3)   

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     if (q_stop_recip) exit

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('comm_coord_among_recip')
#endif

     call comm_coord_among_recip(natom, x, y, z, cg)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('zero_recip_force')
#endif
     call zero_recip_force(dx, dy, dz)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif

     call zero_energy_terms()

     !-----------------------------------------------------------------------
     ! Calculate constraints etc., results are added to (dx, dy, dz) and vpress(1:9)
     ! ******* Only constraint node does these ************
     !-----------------------------------------------------------------------
     if (q_cons_node) then
#if KEY_ADUMB==1
        ! JMS 7/2012 -- adaptive umbrella sampler
        call calc_adumb_potential(eterm(adumb),x,y,z,dx,dy,dz,nloop)
#endif 
        ! CONS DIHE
        call calc_dihe_restraints(eterm(cdihe), x, y, z, dx, dy, dz, dd1_dummy, iupt_dummy, qsecd)
        ! NOE
        call calc_noe_restraints(eterm(noe), x, y, z, dx, dy, dz, dd1_dummy, iupt_dummy, qsecd)
        ! REDCNS
        call calc_redcns(eterm(resd), x, y, z, dx, dy, dz, dd1_dummy, iupt_dummy, qsecd)
        
        ! DENBIAS
        call calc_dbias(eterm(charm),.true.,.false.)
        
        ! Calculate virial for the above constraints / restraints
        call calc_virial_noshift(vpress, x, y, z, dx, dy, dz, ncons, conslist)
        
        ! Umbrella potential term
        call calc_umbrella_potential(eterm(umbr), vpress, .true.)
        ! Distance matrix constraint
        call calc_dmcons(dmc_ener, dmc_rho, x, y, z, dx, dy, dz, qsecd, vpress, .true.)
        eterm(dmc) = SUM(dmc_ener)

        !------------------------------------------------
        ! These don't contribute to virial in energy.src:
        !------------------------------------------------
        ! . Center of mass harmonic constraint
        call calc_hmcm(eterm(hmcm), x, y, z, dx, dy, dz)

     endif
     !-----------------------------------------------------------------------

     call timer_start(T_rec)
     if (.not.q_domdec) then
#if KEY_PARALLEL==1
        TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER 
#endif
#if KEY_PARALLEL==1
        TIMMER = ECLOCK()
#endif
     endif
     !.ab.HYBH supported. Special routine for forces (calculate dH/dlambda)...
     !+.ab.HYBH for the time being we assume that if QHYBH=.true. we don't use
     !+.ab. other block functionnalities -> block = false, restored after....
     !+.ab. to avoid interferrence with block introduced at the same time.
#if KEY_BLOCK==1
     IF (QHYBH) QBLOCK=.FALSE.                          
#endif
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('KSPACE')
#endif
     CALL KSPACE(ETERM(EWKSUM),ETERM(EWSELF),ETERM(EWQCOR),ETERM(EWUTIL),&
          QETERM(EWKSUM), .false., QETERM(EWQCOR),QETERM(EWUTIL), &
          X,Y,Z,DX,DY,DZ,NATOM,CG,CGTOT &
#if KEY_LJPME==1
          ,ljc6   &
#endif
#if KEY_FLUCQ==1
          ,QFLUC,FQCFOR     & 
#endif
          )
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
#if KEY_BLOCK==1
     IF (QHYBH) QBLOCK=.TRUE.                          
#endif
     
     call timer_stop(T_rec)     
     if (.not.q_domdec) then
#if KEY_PARALLEL==1
        TMERI(TIMEXTE) = TMERI(TIMEXTE) + ECLOCK()-TIMMER 
#endif
#if KEY_PARALLEL==1
        TIMMER = ECLOCK()                                 
#endif
     endif

     ! Communicate forces, virial, and energy terms among recip cores
     call timer_start(T_fcomm2)  

     call comm_force_among_recip(dx, dy, dz)

#if KEY_DOMDEC_GPU==1
     if (q_gpu .and. gpu_code_version==2) then
        call read_recip_virial_gpu(ewvirial)
        call read_recip_energy_gpu(eterm(ewksum), eterm(ewself))
     endif
#endif

     ! Default data:
     auxdata(1:9) = ewvirial(1:9)
     auxdata(10) = eterm(ewksum)
     auxdata(11) = eterm(ewqcor)
     auxdata(12) = eterm(ewutil)
     nauxdata = 12
#if KEY_DOMDEC_GPU==1
     if (q_split .and. q_gpu .and. gpu_code_version==2) then
       auxdata(13) = eterm(ewself)
       nauxdata = 13
     end if
#endif
#if KEY_LJPME==1
     if (q_split .and. qljpme) then
        auxdata(14) = eterm(ljrec)
        nauxdata = 14 
     end if
#endif

     ! Next, we are going to add some data to the auxdata -buffer and then send
     ! the buffer to the direct node
        call write_to_auxdata(nauxdata, auxdata, dmc_ener, dmc_rho, vpress)

#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('send_results_to_direct')
#endif
     ! Communicate results to direct nodes
     call send_results_to_direct(dx, dy, dz, nauxdata, auxdata)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()
#endif
     call timer_stop(T_fcomm2)   

     nloop = nloop + 1
  enddo

  call timer_stop(T_energy)             

  return
end subroutine energy_recip

#endif 

#if KEY_DOMDEC==1
  ! *
  ! * Pretend that we only have a single node in use
  ! *
  subroutine switch_to_single_node()
    use domdec_common,only:q_domdec,q_cons_node
    use parallel,only:numnod,mynod,mynodp
    implicit none

    if (.not.q_domdec .or. .not.q_cons_node) then
       call wrndie(-5,'<energy_util>','Invalid call to switch_to_single_node()')
    endif

    numnod_save = numnod
    mynod_save = mynod
    
    numnod = 1
    mynod = 0
    mynodp = 1
    
    return
  end subroutine switch_to_single_node

  ! *
  ! * Restore back to normal number of nodes
  ! *
  subroutine restore_node()
    use domdec_common,only:q_domdec,q_cons_node
    use parallel,only:numnod,mynod,mynodp
    implicit none

    if (.not.q_domdec .or. .not.q_cons_node) then
       call wrndie(-5,'<energy_util>','Invalid call to restore_node()')
    endif

    numnod = numnod_save
    mynod = mynod_save
    mynodp = mynod_save+1

    return
  end subroutine restore_node
#endif

#if KEY_DOMDEC==1 /*domdec*/
  
  ! *
  ! * Reads data from auxdata -array
  ! *
  subroutine read_from_auxdata(nauxdata, auxdata, dmc_ener, dmc_rho, vpress)
    use domdec_common,only:q_domdec
    use energym,only:eterm, qeterm, dmc, umbr, adumb, cdihe, noe, resd, charm
    use dmcons,only:qdmc,ndmc            
#if KEY_RXNCOR==1
    use rxncom,only:rxnind          
#endif
#if KEY_ADUMB==1
    use umb, only: numbr, coum
#endif 
#if KEY_MTS==1
    use tbmts,only:ene2, ene3
#endif
#if KEY_NOMISC==0
    use noem,only:noenum
    use resdist_ltm,only:rednum
#endif
#if KEY_DENBIAS==1
    use denbias,only:qdenbias
#endif
#if KEY_HMCOM==1
    use cstran_mod,only:qhmcm
    use energym,only:hmcm
#endif
    use cnst_fcm,only:ncsphi
    use parallel,only:mynod
    implicit none
    ! Input / Output
    integer, intent(inout) :: nauxdata
    real(chm_real), allocatable, dimension(:), intent(in) :: auxdata
    real(chm_real), intent(out) :: dmc_ener(ndmc), dmc_rho(ndmc), vpress(9)
    integer :: i
    
    if (q_domdec) then
#if KEY_DMCONS==1
       if (qdmc) then
          ! Assign ecdmd and dmc_rho
          if (mynod == 0) then
             dmc_ener(1:NDMC) = auxdata(nauxdata+1:nauxdata+NDMC)
             eterm(dmc) = SUM(dmc_ener)

             dmc_rho(1:NDMC) = auxdata(nauxdata+NDMC+1:nauxdata+2*NDMC)
          endif
          nauxdata = nauxdata + 2*NDMC
       endif
#endif 
#if KEY_RXNCOR==1
       if (rxnind /= 0 .and. qeterm(umbr)) then
          if (mynod == 0) then
             eterm(umbr) = auxdata(nauxdata+1)
          endif
          nauxdata = nauxdata + 1
       endif
#endif 
#if KEY_ADUMB==1
       !JMS 8/2012 -- Assign energy for ADUMB and reaction coordinates
       if (numbr > 0) then
          if (mynod == 0) eterm(adumb) = auxdata(nauxdata+1)
          nauxdata = nauxdata + 1
          if (mynod == 0) coum(1:numbr) = auxdata(nauxdata+1:nauxdata+numbr)
          nauxdata = nauxdata + numbr
       endif
#endif 
#if KEY_MTS==1
       if(ene2) then
#endif
          if((ncsphi.gt.0) .and. qeterm(cdihe)) then
             if (mynod == 0) eterm(cdihe) = auxdata(nauxdata+1)
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#if KEY_NOMISC==0
#if KEY_MTS==1
       if(ene2) then
#endif
          if((noenum.gt.0) .and. qeterm(noe)) then
             if (mynod == 0) eterm(noe) = auxdata(nauxdata+1)
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif
#if KEY_NOMISC==0
#if KEY_MTS==1
       if(ene2) then
#endif
          if((rednum.gt.0) .and. qeterm(resd)) then
             if (mynod == 0) eterm(resd) = auxdata(nauxdata+1)
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif

#if KEY_DENBIAS==1
#if KEY_MTS==1
       if(ene2) then
#endif
          if((qdenbias) .and. qeterm(charm)) then
             if (mynod == 0) eterm(charm) = auxdata(nauxdata+1)
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif

#if KEY_HMCOM==1
#if KEY_MTS==1
       if(ene3) then
#endif
          if (qhmcm.and.qeterm(hmcm)) then
             if (mynod == 0) eterm(hmcm) = auxdata(nauxdata+1)
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif
       ! Extract virial pressure contribution from constraints etc.
       vpress(1:9) = auxdata(nauxdata+1:nauxdata+9)
       nauxdata = nauxdata + 9
    endif
    
    return
  end subroutine read_from_auxdata

  ! *
  ! * Returns the size of auxdata
  ! *
  integer function get_nauxdata()
    use dmcons,only:ndmc            
#if KEY_DOMDEC_GPU==1
    use domdec_common, only: gpu_code_version, q_gpu, q_split
#endif
#if KEY_LJPME==1
    use domdec_common, only: q_split
    use pmeutil, only: qljpme
#endif
    implicit none

    real(chm_real) dummy1(ndmc), dummy2(ndmc), dummy3(9)

    ! Calculate the size of auxdata
    get_nauxdata = 12
#if KEY_DOMDEC_GPU==1
    if (q_split .and. q_gpu .and. gpu_code_version==2) get_nauxdata = 13
#endif
#if KEY_LJPME==1
     if (q_split .and. qljpme) get_nauxdata = 14 
#endif
    call auxdata_kernel(dummy1, dummy2, dummy3, get_nauxdata)

    return
  end function get_nauxdata

  ! *
  ! * Initializes nauxdata and auxdata
  ! *
  subroutine init_auxdata(nauxdata, auxdata)
    implicit none
    ! Input / Output
    integer, intent(out) :: nauxdata
    real(chm_real), allocatable, dimension(:), intent(inout) :: auxdata

    nauxdata = get_nauxdata()

    ! Allocate / Reallocate auxdata(1:nauxdata_new) as needed
    call alloc_realloc(nauxdata, auxdata)

    return
  contains
    ! *
    ! * Reallocates auxdata() -array as needed
    ! *
    subroutine alloc_realloc(new_size, auxdata)
      use memory,only:chmalloc, chmrealloc
      implicit none
      ! Input / Output
      integer, intent(in) :: new_size
      real(chm_real), allocatable, dimension(:), intent(inout) :: auxdata

      if (allocated(auxdata)) then
         if (size(auxdata) < new_size) then
            call chmrealloc('energy.src','init_auxdata','auxdata',new_size,crl=auxdata)
         endif
      endif

      if (.not.allocated(auxdata) .and. new_size > 0) then
         call chmalloc('energy.src','init_auxdata','auxdata',new_size,crl=auxdata)
      endif

      return
    end subroutine alloc_realloc
  end subroutine init_auxdata

  ! *
  ! * Writes (adds) data to auxdata -array
  ! *
  subroutine write_to_auxdata(nauxdata, auxdata, dmc_ener, dmc_rho, vpress)
    implicit none
    ! Input / Output
    integer, intent(inout) :: nauxdata
    real(chm_real), allocatable, dimension(:), intent(inout) :: auxdata
    real(chm_real), intent(in) :: dmc_ener(*), dmc_rho(*), vpress(9)

    call auxdata_kernel(dmc_ener, dmc_rho, vpress, nauxdata, auxdata)

    return
  end subroutine write_to_auxdata

  ! *
  ! * Kernel subroutine that does all the actual work
  ! *
  subroutine auxdata_kernel(dmc_ener, dmc_rho, vpress, nauxdata, auxdata)
    use domdec_common,only:q_domdec
    use energym,only:eterm, qeterm, dmc, umbr, adumb, cdihe, noe, resd, charm
    use dmcons,only:qdmc,ndmc            
#if KEY_RXNCOR==1
    use rxncom,only:rxnind          
#endif
#if KEY_ADUMB==1
    use umb, only: numbr, coum      
#endif
#if KEY_MTS==1
    use tbmts, only: ene2, ene3
#endif
#if KEY_NOMISC==0
    use noem,only:noenum
    use resdist_ltm,only:rednum
#endif
#if KEY_DENBIAS==1
    use denbias,only: qdenbias
#endif
#if KEY_HMCOM==1
    use cstran_mod,only:qhmcm
    use energym,only:hmcm
#endif
    use cnst_fcm,only:ncsphi
    implicit none
    ! Input
    real(chm_real), intent(in) :: dmc_ener(ndmc), dmc_rho(ndmc), vpress(9)
    integer, intent(inout) :: nauxdata
    real(chm_real), intent(out), optional :: auxdata(:)
    integer :: i

    if (q_domdec) then
#if KEY_DMCONS==1
       if (qdmc) then
          if (present(auxdata)) then
             auxdata(nauxdata+1:nauxdata+NDMC) = dmc_ener(1:NDMC)
             auxdata(nauxdata+NDMC+1:nauxdata+2*NDMC) = dmc_rho(1:NDMC)
          endif
          nauxdata = nauxdata + 2*NDMC
       endif
#endif 
#if KEY_RXNCOR==1
       if (rxnind /= 0 .and. qeterm(umbr)) then
          if (present(auxdata)) then
             auxdata(nauxdata+1) = eterm(umbr)
          endif
          nauxdata = nauxdata + 1
       endif
#endif 
#if KEY_ADUMB==1
       !JMS 7/2012 -- insert energy and reaction coordinates for adaptive umbrella
       if (numbr > 0 ) then
          if (present(auxdata)) then
             auxdata(nauxdata+1) = eterm(adumb)
          endif
          nauxdata = nauxdata + 1
          if (present(auxdata)) then
             auxdata(nauxdata+1:nauxdata+numbr) = coum(1:numbr)
          endif
          nauxdata = nauxdata + numbr
       endif
#endif 
#if KEY_MTS==1
       if(ene2) then                                          
#endif
          if((ncsphi.gt.0) .and. qeterm(cdihe)) then
             if (present(auxdata)) then
                auxdata(nauxdata+1) = eterm(cdihe)
             endif
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#if KEY_NOMISC==0
#if KEY_MTS==1
       if(ene2) then                                          
#endif
          if((noenum.gt.0) .and. qeterm(noe)) then
             if (present(auxdata)) then
                auxdata(nauxdata+1) = eterm(noe)
             endif
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif
#if KEY_NOMISC==0
#if KEY_MTS==1
       if(ene2) then                                          
#endif
          if((rednum.gt.0) .and. qeterm(resd)) then
             if (present(auxdata)) then
                auxdata(nauxdata+1) = eterm(resd)
             endif
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif

#if KEY_DENBIAS==1
#if KEY_MTS==1
       if(ene2) then                                          
#endif
          if((qdenbias) .and. qeterm(charm)) then
             if (present(auxdata)) then
                auxdata(nauxdata+1) = eterm(charm)
             endif
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif

#if KEY_HMCOM==1
#if KEY_MTS==1
       if(ene3) then
#endif
          if (qhmcm.and.qeterm(hmcm)) then
             if (present(auxdata)) then
                auxdata(nauxdata+1) = eterm(hmcm)
             endif
             nauxdata = nauxdata + 1
          endif
#if KEY_MTS==1
       endif
#endif
#endif
       ! Add vpress
       if (present(auxdata)) then
          auxdata(nauxdata+1:nauxdata+9) = vpress(1:9)
       endif
       nauxdata = nauxdata + 9
    endif

    return
  end subroutine auxdata_kernel

#endif /* (domdec)*/

  ! *
  ! * Sets mts flags ENE1, ENE2, ENE3
  ! *
  subroutine set_mtsflags()
#if KEY_MTS==1
    use tbmts,only:ene1,ene2,ene3        
    use tbmts_ltm, only: qtbmts
#endif
    implicit none

#if KEY_MTS==1 /*mts_setflags*/
    IF (.NOT. QTBMTS) THEN
       !        If MTS is not in use, then do all energy terms - BRB 01/06/98
       ENE1=.TRUE.  ! flag to do fast terms
       ENE2=.TRUE.  ! flag to do medium terms
       ENE3=.TRUE.  ! flag to do slow terms
    ENDIF
#endif /* (mts_setflags)*/

    return
  end subroutine set_mtsflags

  ! *
  ! * Calculate dmcons (distance matrix constraints)
  ! * NOTE: for domdec, all constraint nodes do the same calculation
  ! * NOTE: vpress should be optional argument, but optional arguments cannot be used
  ! * here because energy.src is not in a module :(
  ! *
  subroutine calc_dmcons(ecdmc, rho, x, y, z, dx, dy, dz, qsecd, vpress, q_vpress)
    use chm_kinds
    use psf,only:natom
#if KEY_MTS==1
    use tbmts,only:ene3                   
#endif
#if KEY_DMCONS==1
    use dmcons,only:qdmc,edmc,ndmc             
#endif
    use timerm,only:timer
    use machutil,only:wrttim
#if KEY_BLOCK==1
    use block_ltm,only:qhybh              
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod               
#endif
    implicit none
    ! Input / Output
#if KEY_DMCONS==1
    real(chm_real) :: ecdmc(ndmc), rho(ndmc)
#else
    real(chm_real) :: ecdmc(*), rho(*)
#endif
    real(chm_real) x(*), y(*), z(*), dx(*), dy(*), dz(*)
    logical qsecd
    real(chm_real) vpress(9)
    logical q_vpress

#if KEY_DMCONS==1
#if KEY_MTS==1
    IF(ENE3) THEN                                          
#endif
       !**clbiii change to make sure its mapped onto processor 0
       !**because dmcons and rgy write to units that have been
       !**only assigned on on processor 0.
#if KEY_DOMDEC==1
       if ((.not.q_domdec .and. mynod == 0) .or. q_cons_node) then 
#endif
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
          IF(MYNOD.EQ.0) THEN                      
#endif
             IF(QDMC.AND.(.NOT.QSECD)) THEN
                !.ab.
#if KEY_BLOCK==1
                IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
                     'HYBH and DMC incompatible.')
#endif 
                !.ab.
                if (q_vpress) then
                   CALL EDMC(ecdmc,rho,X,Y,Z,DX,DY,DZ,NATOM,vpress)
                else
                   CALL EDMC(ecdmc,rho,X,Y,Z,DX,DY,DZ,NATOM)
                endif
#if KEY_PARALLEL==1
                if (mynod == 0) then  
#endif
                   IF (TIMER.GT.1) CALL WRTTIM('DMC constraint energy times:')
#if KEY_PARALLEL==1
                endif  
#endif
             ENDIF
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
          ENDIF                        
#endif
#if KEY_DOMDEC==1
       endif                           
#endif
#if KEY_MTS==1
    ENDIF                                                  
#endif
#endif 

    return
  end subroutine calc_dmcons

  ! *
  ! * Writes dmcons to file. Only root node writes
  ! *
  subroutine write_dmcons(ecdmc, rho, qsecd)
    use chm_kinds
#if KEY_MTS==1
    use tbmts,only:ene3                   
#endif
#if KEY_DMCONS==1
    use dmcons,only:qdmc,write_edmc,ndmc
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod               
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh              
#endif
    implicit none
    ! Input
#if KEY_DMCONS==1
    real(chm_real) :: ecdmc(ndmc), rho(ndmc)
#else
    real(chm_real) :: ecdmc(*), rho(*)
#endif
    logical qsecd

#if KEY_DMCONS==1
#if KEY_MTS==1
    IF(ENE3) THEN                                          
#endif
       !**clbiii change to make sure its mapped onto processor 0
       !**because dmcons and rgy write to units that have been
       !**only assigned on on processor 0.
#if KEY_PARALLEL==1
       IF(MYNOD.EQ.0) THEN                      
#endif
          IF(QDMC.AND.(.NOT.QSECD)) THEN
             !.ab.
#if KEY_BLOCK==1
             IF (QHYBH) CALL WRNDIE(-5,'<ENERGY>', &
                  'HYBH and DMC incompatible.')
#endif 
             !.ab.
             call write_edmc(ecdmc, rho)
          ENDIF
#if KEY_PARALLEL==1
       else                          
#endif
#if KEY_PARALLEL==1
          call write_edmc(ecdmc,rho) 
#endif
#if KEY_PARALLEL==1
       ENDIF                         
#endif
#if KEY_MTS==1
    ENDIF                                                  
#endif
#endif 

    return
  end subroutine write_dmcons

  ! *
  ! * Calculate rxncor umbrella potential
  ! *
  subroutine calc_umbrella_potential(eumb, vpress, q_vpress)
    use chm_kinds
    use timerm,only:timer
#if KEY_MTS==1
    use tbmts,only:ene2
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod  
#endif
#if KEY_RXNCOR==1
    use rxenemod,only:rxnene 
    use rxncom,only:rxnind
    use energym,only:qeterm,umbr
#if KEY_DOMDEC==1
    use rxncom,only:q_comp_nod
#endif
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_ADUMBRXNCOR==1
    use umb, only: numbr 
#endif
    use machutil,only:wrttim
    implicit none
    ! Input / Output
    real(chm_real) eumb
    real(chm_real) vpress(9)
    logical q_vpress

#if KEY_RXNCOR==1
#if KEY_MTS==1
    IF(ENE2) THEN             
#endif
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
       IF(MYNOD.EQ.0) THEN    
#endif
#if KEY_DOMDEC==1
          if ((.not.q_domdec .and. mynod == 0) .or. q_cons_node) then 
#endif
             !JMS 8/2012 -- Do not go here if we are doing adaptive umbrella sampling.
             !This won't work if we are simultaneously imposing a harmonic umbrella potential
             ! with RXNCOR UMBRELLA and doing adaptive umbrella sampling e.g. on dihedrals.
             ! But this combination is sufficiently rare that we will ignore it.
             IF (RXNIND.NE.0 .AND. QETERM(UMBR) &
#if KEY_ADUMBRXNCOR==1
                  .and.(numbr<=0) &
#endif
                  ) THEN
#if KEY_DOMDEC==1
                ! Mark this node as computing rxnene
                q_comp_nod = .true.
#endif
                if (q_vpress) then
                   CALL RXNENE(EUMB,vpress)
                else
                   CALL RXNENE(EUMB)
                endif
                IF (TIMER.GT.1) CALL WRTTIM('Umbrella potential times:')
             ENDIF
#if KEY_DOMDEC==1
          endif
#endif
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
       ENDIF
#endif
#if KEY_MTS==1
    ENDIF                     
#endif
#endif 
    
    return
  end subroutine calc_umbrella_potential

  ! JMS 7/2012, DOMDEC/ADUMB integration -- calculate adaptive umbrella potential
  ! ADUMB ENER is untested and not guaranteed to work, esp. in domain decomposition.
  ! This has also been rewrapped onto node 0 from inode(12) and inode(5).
  subroutine calc_adumb_potential(eadumb,x,y,z,dx,dy,dz,callcount)
    use chm_kinds
    use timerm,only:timer
    use new_timer 
#if KEY_MTS==1
    use tbmts,only:ene2      
#endif
#if KEY_PARALLEL==1
    use parallel 
#endif
    use energym
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh                     
#endif
    use machutil,only:wrttim,eclock
    use econtmod, only: qecont, econt
    use psf, only: natom,natomt
    use number
#if KEY_ADUMB==1 /*adumb*/
    use umb
    use umbcor
#endif /*  (adumb)*/
    implicit none
    real(chm_real) eadumb,x(*),y(*),z(*),dx(*),dy(*),dz(*)
    integer callcount
    integer i
#if KEY_PARALLEL==1 /*pll*/
    INTEGER NALLWK
    !.ab.HybH.for//brodcasting+qtest.      PARAMETER (NALLWK=LENENT+LENENV+10)
    PARAMETER (NALLWK=3*LENENT+LENENV+10)
    real(chm_real) ALLWRK(NALLWK)
    real(chm_real) TIMMER, TIMME1
    INTEGER IPT
#endif /* (pll)*/

#if KEY_ADUMB==1
    !
    ! Get internal coordinate for dihedrals
    !
    !.ab.Note; not clear why use ADUM & TI... If it was contact A.Blondel.
    eadumb = zero

    !So that ADUMB ENER works in conventional parallel, allow all nodes through if that is
    !what we are doing
    !ADUMB ENER does not work with domain decomposition
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
    IF((MYNOD.EQ.0).or.(ieum.gt.0)) THEN    
#endif
#if KEY_DOMDEC==1
       if (q_domdec .and. (ieum>0)) call wrndie(-2,'<calc_adumb_potential>', &
            'ADUMB ENER may not work with domain decomposition. Use at your own risk.')

       if ((.not.q_domdec .and. (mynod == 0 .or. ieum > 0)) .or. q_cons_node) then 
#endif 

#if KEY_BLOCK==1
          IF ((QHYBH).AND. &
               ((NUMPHI.GT.0).OR.(NNUM.GT.0).OR.(IEUM.GT.0).OR.(NUMBR.GT.0))) &
               CALL WRNDIE(-5,'<ENERGY>', &
               'HYBH and ADUMB incompatible, see code.')
#endif 
          !.ab.
          IF(MOD(ECALLS,ADFREQ).EQ.0) THEN
             IF(NUMPHI.GT.0) THEN
                CALL UM1PHI(IUM,JUM,KUM,LUM,IPHIUM,COUM,NUMPHI,X,Y,Z)
                IF (TIMER.GT.1) &
                     CALL WRTTIM('Dihedral umbrella internal coord. times:')
             ENDIF
#if KEY_ADUMBRXNCOR==1
             IF(numbrxn.GT.0) THEN
                CALL UM1rxn(coum,x,y,z)
                IF (TIMER.GT.1) CALL WRTTIM('Dihedral umbrella internal coord. times:')
             ENDIF
#endif /*     */
             ! Get internal coordinate for NOE umbrella
             ! 
             IF(NNUM.GT.0) THEN
                CALL UM1NOE(NATOM,X,Y,Z)
                IF (TIMER.GT.1) &
                     CALL WRTTIM('NOE umbrella internal coord. times:')
             ENDIF
             !
             ! Get internal coordinate for energy
             ! 
             IF(IEUM.GT.0) THEN
#if KEY_PARALLEL==1
                call timer_start(T_ecomm) 
#if KEY_DOMDEC==1
                if (.not.q_domdec) then  
#endif
                   ! get energies
                   TMERI(TIMINTE) = TMERI(TIMINTE) + ECLOCK()-TIMMER
                   TIMMER = ECLOCK()
                   CALL PSYNC()
                   TMERI(TIMWAIT) = TMERI(TIMWAIT) + ECLOCK()-TIMMER
                   TIMMER = ECLOCK()
#if KEY_DOMDEC==1
                endif  
#endif
                DO I=1,LENENT
                   ALLWRK(I)=ETERM(I)
                ENDDO
                !
                CALL GCOMB(ALLWRK,LENENT)
                CALL UM1EN(ALLWRK,LENENT)
                call timer_stop( T_ecomm) 
#else /**/
                CALL UM1EN(ETERM,LENENT)
#endif 
                IF (TIMER.GT.1) &
                     CALL WRTTIM('Energy umbrella internal coord. times:')
             ENDIF
             !
             ! Update statistics and get forces
             ! JMS 7/2012 -- We will keep track of statistics and perform WHAM on both direct
             ! and reciprocal nodes 
             ! so they can be written to the output file. 
             IF(NUMBR.GT.0) THEN
                IF(STONUM ) THEN
                   call UMSTAT(STFPUM,STUPUM,STSFUM,STHIUM,STEMUM, &
                        HPCRRD,HPTOAV,HPCRMS1,HPCRMS2,HPTORM1,HPTORM2)

                ENDIF
                call UMDER(STFPUM)
                eadumb = -EUM
                IF (TIMER.GT.1) &
                     CALL WRTTIM('Adaptive Umbrella:')
             ENDIF
             !
             ! Galc forces from NOE umbrella
             !
             IF(NNUM.GT.0) THEN
                CALL UM2NOE(NATOM,DX,DY,DZ)
                IF (TIMER.GT.1) &
                     CALL WRTTIM('NOE umbrella calc. forces times:')
             ENDIF
             ! 
             ! Calc forces from energy umbrella
             !
             IF(IEUM.GT.0) THEN
                CALL UM2EN(NATOMT,DX,DY,DZ)
                IF (TIMER.GT.1) &
                     CALL WRTTIM('Energy umbrella calc. forces times:')
             ENDIF
             ! 
             ! Calc forces from dihedral umbrellas
             !
             !
             IF(NUMPHI.GT.0) THEN
                CALL UM2PHI(ETERM(CDIHE),IUM,JUM,KUM,LUM,IPHIUM,NUMPHI,DUM, &
                     DX,DY,DZ,X,Y,Z,QECONT,ECONT)
                IF (TIMER.GT.1) &
                     CALL WRTTIM('Dihedral umbrella calc. forces times:')
             ENDIF
#if KEY_ADUMBRXNCOR==1
             IF(NUMbrxn.GT.0) THEN
                CALL UM2rxn(DUM,DX,DY,DZ,X,Y,Z)
                IF (TIMER.GT.1) CALL WRTTIM('Dihedral umbrella calc. forces times:')            
             ENDIF
#endif /* */
             !
             !
          ENDIF
#if KEY_DOMDEC==1
       endif               
#endif
#if KEY_PARALLEL==1 && KEY_DOMDEC==0
    ENDIF                  
#endif
#endif 
    ! . End Adaptive Umbrella
    return
  end subroutine calc_adumb_potential

  ! *
  ! * Calculates dihedral restraints
  ! *
  subroutine calc_dihe_restraints(ecdihe, x, y, z, dx, dy, dz, dd1, iupt, qsecd)
#if KEY_MTS==1
    use tbmts,only:ene2
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh 
#endif
#if KEY_SCCDFTB==1
    use blockscc_fcm,only:qsccb, idxphi
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
    use domdec_cons,only:print_cons
#endif
    use timerm,only:timer
    use machutil,only:wrttim
    use cnst_fcm,only:ncsphi,ics,jcs,kcs,lcs,iccs,ncsphi,ccsc,ccsd,ccsb,ccscos,ccssin,ccsw
    use energym,only:qeterm, cdihe
    use econtmod,only:qecont, econt
    use eintern,only:ephi
    use parallel,only:numnod,mynod,mynodp
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: ecdihe
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
    real(chm_real), intent(inout) :: dd1(*)
    integer, intent(in) :: iupt(*)
    logical, intent(in) :: qsecd

    ! . Dihedral restraints.
#if KEY_MTS==1
    IF(ENE2) THEN                                          
#endif
       IF((NCSPHI.GT.0) .AND. QETERM(CDIHE)) THEN
          !.ab.Note: Nothing would prevent it, but I do not see why it should
          !.ab..be done. Contact A.Blondel.
#if KEY_BLOCK==1
          IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
               'HYBH and CDIHE incompatible, see code.')
#endif 
          !.ab.
#if KEY_SCCDFTB==1
          if(qsccb) idxphi=2                                  
#endif
#if KEY_DOMDEC==1
          if (.not.q_domdec .or. (q_domdec .and. q_cons_node)) then
             if (q_domdec) call switch_to_single_node()
#endif
             CALL EPHI(ecdihe,ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
                  CCSC,CCSD,CCSB,CCSCOS,CCSSIN,DX,DY,DZ,X,Y,Z, &
                  .TRUE.,CCSW,QECONT,ECONT,0,(/0/),DD1,IUPT,QSECD &
                  )
             IF (TIMER.GT.1) &
                  CALL WRTTIM('Dihedral constraint energy times:')
#if KEY_DOMDEC==1
             if (q_domdec) call restore_node()
          endif
#endif
       ENDIF
#if KEY_MTS==1
    ENDIF
#endif

    return
  end subroutine calc_dihe_restraints

  ! *
  ! * Calculates NOE restraints
  ! *
  subroutine calc_noe_restraints(enoe, x, y, z, dx, dy, dz, dd1, iupt, qsecd)
#if KEY_MTS==1
    use tbmts,only:ene2
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod, inode
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_NOMISC==0
    use noem,only:noenum,noesca,noeipt,noejpt,noeinm,noejnm, &
                  noelis,noeexp,noermn,noekmn,noermx,noekmx, &
#if KEY_PNOE==1
                  IsPNOE, C0X, C0Y, C0Z                 & 
                  , MVPNOE,OC0X,OC0Y,OC0Z                 & 
                  ,TC0X,TC0Y,TC0Z                 & 
                  , NMPNOE, IMPNOE,&
#endif
                  noefmx,noetcn,noeave,noemin,noersw,noesex,noeram
    use timerm,only:timer
    use machutil,only:wrttim
    use energym,only:qeterm, noe
    use econtmod,only:qecont, econt
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: enoe
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
    real(chm_real), intent(inout) :: dd1(*)
    integer, intent(in) :: iupt(*)
    logical, intent(in) :: qsecd
    ! Variables
    logical execute

#if KEY_NOMISC==0

    execute = .true.

#if KEY_MTS==1
    execute = execute .and. ene2
#endif

#if KEY_DOMDEC==1
    if (q_domdec) then
       execute = execute .and. q_cons_node
    else
#endif
#if KEY_PARALLEL==1
       execute = execute .and. (mynod == inode(5))
#endif
#if KEY_DOMDEC==1
    endif
#endif

    if (execute) then
       IF((NOENUM.GT.0) .AND. QETERM(NOE)) THEN
             !.ab.
#if KEY_BLOCK==1
          IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
               'HYBH and NOE incompatible.')
#endif 
          CALL NOECNS(enoe,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
               NOENUM,NOESCA,NOEIPT,NOEJPT,NOEINM,NOEJNM, &
               NOELIS,NOEEXP,NOERMN,NOEKMN,NOERMX,NOEKMX, &
               NOEFMX,NOETCN,NOEAVE,NOEMIN,DD1,IUPT,QSECD &
               ,NOERSW,NOESEX,NOERAM                        & !CJH, NOE_SOFT
#if KEY_PNOE==1
               , IsPNOE, C0X, C0Y, C0Z                 & 
               , MVPNOE,OC0X,OC0Y,OC0Z                 & 
               ,TC0X,TC0Y,TC0Z                 & 
               , NMPNOE, IMPNOE                        & 
#endif
               )
          IF (TIMER.GT.1) CALL WRTTIM('NOE constraint energy times:')
       ENDIF
    endif
#endif
    return
  end subroutine calc_noe_restraints

  ! . General distance restraints.
  subroutine calc_redcns(eresd, x, y, z, dx, dy, dz, dd1, iupt, qsecd)
#if KEY_MTS==1
    use tbmts,only:ene2
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_NOMISC==0
    use resdist_ltm,only:rednum,redsca,redipt,redilis,redklis, &
         redkval,redrval,redeval,redival,redmval
    use resdist,only:redcns
    use timerm,only:timer
    use machutil,only:wrttim
    use energym,only:qeterm, resd
    use econtmod,only:qecont, econt
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: eresd
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
    real(chm_real), intent(inout) :: dd1(*)
    integer, intent(in) :: iupt(*)
    logical, intent(in) :: qsecd

#if KEY_NOMISC==0
#if KEY_MTS==1
    IF(ENE2) THEN
#endif
       IF((REDNUM.GT.0) .AND. QETERM(RESD)) THEN
          !.ab.
#if KEY_BLOCK==1
          IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
               'HYBH and RESD incompatible.')
#endif 
#if KEY_DOMDEC==1
          if (.not.q_domdec .or. (q_domdec .and. q_cons_node)) then
             if (q_domdec) call switch_to_single_node()
#endif
             CALL REDCNS(eresd,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                  REDNUM,REDSCA,REDIPT,REDILIS,REDKLIS, &
                  REDKVAL,REDRVAL,REDEVAL,REDIVAL,REDMVAL, &
                  DD1,IUPT,QSECD)
             IF (TIMER.GT.1) &
                  CALL WRTTIM('Distance restraint energy times:')
#if KEY_DOMDEC==1
             if (q_domdec) call restore_node()
          endif
#endif
       ENDIF
#if KEY_MTS==1
    ENDIF
#endif
#endif
    return
  end subroutine calc_redcns

  ! . Density Bias.
  subroutine calc_dbias(ener_denbias,qforc,qprint)

#if KEY_MTS==1
    use tbmts,only:ene2
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_DENBIAS==1
    use denbias,only:qdenbias,denbias_vol_density
    use timerm,only:timer
    use machutil,only:wrttim
    use energym,only:qeterm, charm
#endif
    use deriv      !  DX,DY,DZ - Force components
    implicit none
    ! Input / Output
    logical, intent(in) :: qforc, qprint
    real(chm_real), intent(inout) :: ener_denbias

#if KEY_DENBIAS==1
#if KEY_MTS==1
    IF(ENE2) THEN
#endif
       IF((qdenbias) .AND. QETERM(charm)) THEN
          !.ab.
#if KEY_BLOCK==1
          IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
               'HYBH and DENBIAS incompatible.')
#endif 
#if KEY_DOMDEC==1
          if (.not.q_domdec .or. (q_domdec .and. q_cons_node)) then
             if (q_domdec) call switch_to_single_node()
#endif
             CALL denbias_vol_density(ener_denbias, .true., .false.)
             
             IF (TIMER.GT.1) &
                  CALL WRTTIM('Density bias energy times:')
#if KEY_DOMDEC==1
             if (q_domdec) call restore_node()
          endif
#endif
       ENDIF
#if KEY_MTS==1
    ENDIF
#endif
#endif

    return
  end subroutine calc_dbias

  ! . Center of mass harmonic constraint
  subroutine calc_hmcm(ehmcm, x, y, z, dx, dy, dz)
#if KEY_MTS==1
    use tbmts,only:ene3
#endif
#if KEY_BLOCK==1
    use block_ltm,only:qhybh
#endif
#if KEY_PARALLEL==1
    use parallel,only:mynod, inode
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
#endif
#if KEY_HMCOM==1
    use psf,only:natom, amass
    use cstran_mod,only:hmcmx,hmcmy,hmcmz,khmcm,rhmcm, &
         nhmcm,ihmcm,inhmcm,phmcm,lhmcmm,nhmcmr,khmcmr,hmcmr,ihmcm1,ihmcm2,qhmcm
    use timerm,only:timer
    use machutil,only:wrttim
    use energym,only:qeterm, hmcm
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(inout) :: ehmcm
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    real(chm_real), intent(inout) :: dx(*), dy(*), dz(*)
    ! Variables
    logical execute

#if KEY_HMCOM==1
    execute = .true.
#if KEY_MTS==1
    execute = execute .and. ene3
#endif

#if KEY_DOMDEC==1
    if (q_domdec) then
       execute = execute .and. q_cons_node
    else
#endif
#if KEY_PARALLEL==1
       execute = execute .and. (mynod == inode(8))
#endif
#if KEY_DOMDEC==1
    endif
#endif

    if (execute) then
       IF (QHMCM.AND.QETERM(HMCM)) THEN
          !.ab.
#if KEY_BLOCK==1
          IF (QHYBH) CALL WRNDIE(-5,'<ENERGY_UTIL>', &
               'HYBH and MassCenterHarmonic constraint incompatible.')
#endif 
          !.ab.
          CALL ECMCNS(ehmcm,HMCMX,HMCMY,HMCMZ,KHMCM,RHMCM, &
               NATOM,NHMCM,IHMCM,INHMCM,PHMCM, &
               X,Y,Z,DX,DY,DZ,LHMCMM,AMASS, &
               NHMCMR,KHMCMR,HMCMR,IHMCM1,IHMCM2)
          IF (TIMER.GT.1) &
               CALL WRTTIM('Center of mass harmonic constraint times:')
       ENDIF
    ENDIF
#endif

    return
  end subroutine calc_hmcm

  ! *
  ! * Zeros ETERM()
  ! *
  subroutine zero_energy_terms()
    use number,only:zero
    use energym,only:lenent, eterm
#if KEY_BLOCK==1
    use energym,only:etermr, etermp
#endif
    implicit none
    ! Variables
    integer i

    DO I = 1,LENENT
       ETERM(I) = ZERO
       !.ab.
#if KEY_BLOCK==1
       ETERMR(I) = ZERO
       ETERMP(I) = ZERO
#endif 
       !.ab.
    ENDDO
    
    return
  end subroutine zero_energy_terms

end module energy_util

