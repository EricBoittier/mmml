module blade_module
  use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
  use chm_kinds
  implicit none

  type(c_ptr) :: system = c_null_ptr

  logical :: blade_fn_use = .false.
  integer, parameter :: blade_fn_max = 128
  integer :: blade_fn_len = 0
  character(len=blade_fn_max) :: blade_fname = ' '

#if KEY_BLADE == 1
  interface
     function blade_init_system(ngpus, gpus) result(new_system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int, c_ptr
       implicit none
       type(c_ptr) :: new_system
       integer(c_int), value :: ngpus
       integer(c_int) :: gpus(*)
     end function blade_init_system

     subroutine blade_dest_system(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: system
     end subroutine blade_dest_system

     subroutine blade_set_device(system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr
       implicit none
       type(c_ptr), value :: system
     end subroutine blade_set_device

     subroutine blade_set_seed(system, seed, rank) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: seed, rank
     end subroutine blade_set_seed

     subroutine blade_set_verbose(system,v) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: v
     end subroutine blade_set_verbose

     subroutine blade_set_calcTermFlag(system, term, value) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_int
       implicit none
       type(c_ptr), value :: system
       integer(c_int), value :: term, value
     end subroutine blade_set_calcTermFlag

     subroutine blade_interpretter(file_name, system) bind(c)
       use, intrinsic :: iso_c_binding, only: c_ptr, c_char
       implicit none
       type(c_ptr), value :: system
       character(len=1, kind=c_char) :: file_name(*)
     end subroutine blade_interpretter

     ! Interrupt handling for Ctrl+C
     subroutine blade_set_interrupt(value) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
       integer(c_int), value :: value
     end subroutine blade_set_interrupt

     function blade_check_interrupt() result(flag) bind(c)
       use, intrinsic :: iso_c_binding, only: c_int
       implicit none
       integer(c_int) :: flag
     end function blade_check_interrupt

     subroutine blade_install_signal_handler() bind(c)
       implicit none
     end subroutine blade_install_signal_handler

     subroutine blade_restore_signal_handler() bind(c)
       implicit none
     end subroutine blade_restore_signal_handler
  end interface

contains

  subroutine export_psf_to_blade()
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_null_char, c_double
    use psf, only: natom, ibase, nictot, &
         nseg, segid, res, resid, &
         atype, iac, cg, amass, &
         nbond, ib, jb, &
         ntheta, it, jt, kt, &
         nphi, ip, jp, kp, lp, &
         nimphi, im, jm, km, lm &
#if KEY_CMAP==1
         , ncrterm, i1ct, j1ct, k1ct, l1ct, i2ct, j2ct, k2ct, l2ct &
#endif
    ;
#if KEY_LONEPAIR==1
    use lonepr, only: numlp, numlph, lpvalue, lpnhost, lphptr, lphost
#endif
    use blade_psf_module, only: &
         blade_init_structure, &
         blade_add_atom, blade_add_bond, &
         blade_add_angle, blade_add_dihe, &
         blade_add_impr, blade_add_cmap, &
         blade_add_virt2, blade_add_virt3, &
         blade_add_shake, blade_add_noe, &
         blade_add_harmonic, blade_add_boRest, &
         blade_add_anRest, blade_add_diRest, &
         blade_add_resd 
#if KEY_MLMM==1 && KEY_MLPTORCH==1
    use blade_psf_module, only:blade_add_tani ! eemlp added blade_add_resd and blade_add_tani eeresd
#endif
    use param, only: atc
    use shake, only: qshake
    ! use noem, only: noenum
    use noem
    use resdist ! resd  eeresd
    ! use cnst_fcm, only: qcnstr
    use cnst_fcm
    use consta, only: raddeg
    use mmfp
    use fstshk, only : lhonly
    use image, only: xtltyp, xucell, xtlabc, qorth, xtlrot
    use stream, only : outu, prnlev
    use, intrinsic :: iso_c_binding, only: c_bool
#if KEY_MLMM==1 && KEY_MLPTORCH==1
    use mlps_ini  ! eemlp
#endif    

    implicit none

    integer :: i, j, seg_i, res_i, atom_i
    logical :: found
    real(chm_real) :: x, y, z
    real(chm_real) :: convert
    logical(c_bool) :: is_pnoe_c  ! C-compatible boolean for BLaDE API

    call blade_init_structure(system)

    do seg_i = 1, nseg
        do res_i = nictot(seg_i) + 1, nictot(seg_i + 1)
           do atom_i = ibase(res_i) + 1, ibase(res_i + 1)
              ! Was using a numerical resid before: decodi(resid(res_i),8),
              ! requires use string, only: encodi, decodi
              call blade_add_atom(system, atom_i, &
                   trim(segid(seg_i)) // c_null_char, &
                   trim(resid(res_i)) // c_null_char, &
                   trim(res(res_i)) // c_null_char, &
                   trim(atype(atom_i)) // c_null_char, &
                   trim(atc(iac(atom_i))) // c_null_char, &
                   cg(atom_i), amass(atom_i))
           end do
        end do
     end do

     do i = 1, nbond
        call blade_add_bond(system, ib(i), jb(i))
     end do

     do i = 1, ntheta
        call blade_add_angle(system, it(i), jt(i), kt(i))
     end do

     do i = 1, nphi
        call blade_add_dihe(system, ip(i), jp(i), kp(i), lp(i))
     end do

     do i = 1, nimphi
        call blade_add_impr(system, im(i), jm(i), km(i), lm(i))
     end do

#if KEY_LONEPAIR==1
     do i = 1, numlp
        if (lpnhost(i)==2 .and. .not. (lpvalue(1,i)==0 .and. lpvalue(2,i)==0)) then
           j=lphptr(i)
           call blade_add_virt2(system, lphost(j), lphost(j+1), lphost(j+2), lpvalue(1,i), lpvalue(2,i))
        else if (lpnhost(i)==3 .and. .not. (lpvalue(1,i)==0)) then
           ! if (.not. (lpvalue(2,i)==0 .or. lpvalue(2,i)==180)) call wrndie(-3,'BLaDE', 'Lone pair source code lonepair.F90 has a bug for sin(theta) not equal to 0') ! added this functionality
           j=lphptr(i)
           call blade_add_virt3(system, lphost(j), lphost(j+1), lphost(j+2), lphost(j+3), lpvalue(1,i), lpvalue(2,i), lpvalue(3,i))
        else
           if (PRNLEV > 2) write (OUTU, '(a,i5,a,i5,a,d12.6,d12.6)') &
              "Lone pair ", i, "number of hosts ", lpnhost(i), &
              "parameters", lpvalue(1,i), lpvalue(2,i)
           do j = lphptr(i),lphptr(i)+lpnhost(i)
              if (PRNLEV > 2) write (OUTU, '(a,i5,a,i5,a,i5)') &
                 "Lone pair ", i, "host(", j, ") = ", lphost(j)
           end do
           call wrndie(-3, 'BLaDE', 'Lone pair type above is unsupported by BLaDE')
        end if
     end do
#endif

#if KEY_CMAP==1
     do i = 1, ncrterm
        call blade_add_cmap(system, i1ct(i), j1ct(i), k1ct(i), l1ct(i), &
                                    i2ct(i), j2ct(i), k2ct(i), l2ct(i))
     end do
#endif

     if(qshake .and. .not. lhonly) &
          call wrndie(-1, 'BLaDE', 'Only "shake off" or "shake fast bonh" options supported with BLaDE')

     call blade_add_shake(system,merge(1,0,qshake))

     ! if (noenum > 0) call wrndie(-3, 'BLaDE', 'NOE restraints not yet supported with BLaDE')
     ! Limited NOE support
     if (noenum > 0) then
        if (.not. noesca==1) call wrndie(-3, 'BLaDE', 'BLaDE NOE only supports noesca=1')
        do i = 1, noenum
           ! Check atom counts: PNOE only needs i-group, regular NOE needs both
           if (IsPNOE(i)) then
              ! PNOE: only check i-group (j-group is dummy for PNOE)
              if (noeinm(i)/=1) then
                 if (PRNLEV > 2) write (OUTU, '(a,i8)') &
                    "Number of atoms in PNOE i-group: ", noeinm(i)
                 call wrndie(-3, 'BLaDE', 'BLaDE PNOE only supports single atom selection')
              end if
           else
              ! Regular NOE: check both i and j groups
              if (noeinm(i)/=1 .or. noejnm(i)/=1) then
                 if (PRNLEV > 2) write (OUTU, '(a,i8,i8)') &
                    "Number of atoms in i and j groups: ", noeinm(i), noejnm(i)
                 call wrndie(-3, 'BLaDE', 'BLaDE NOE only supports groups of one atom')
              end if
           end if
           if (noetcn(i)>0) call wrndie(-3, 'BLaDE', 'BLaDE NOE does not support time averaging')

           is_pnoe_c = logical(IsPNOE(i), c_bool)

           call blade_add_noe(system, noelis(noeipt(i)), noelis(noejpt(i)), &
              noermn(i), noekmn(i), noermx(i), noekmx(i), noefmx(i), &
              noersw(i), noesex(i), c0x(i), c0y(i), c0z(i), is_pnoe_c)


           ! if (PRNLEV > 2) write (OUTU, '(a,i8,a,i8,a,i8,a,i8,a,i8,a,d12.6,a,d12.6,a,d12.6,a,d12.6,a,d12.6,a,d12.6,a,d12.6,a,d12.6,a,d12.6)') &
           !    " i = ", noelis(noeipt(i)), " j = ", noelis(noejpt(i)), &
           !    " noeinm = ", noeinm(i), " noejnm = ", noejnm(i), &
           !    " noeram = ", noeram(i), &
           !    " noermn = ", noermn(i), " noekmn = ", noekmn(i), &
           !    " noermx = ", noermx(i), " noekmx = ", noekmx(i), " noefmx = ", noefmx(i), &
           !    " noersw = ", noersw(i), " noesex = ", noesex(i), &
           !    " noetcn = ", noetcn(i), " noeave = ", noeave(i)
        end do
     end if

     ! if (qcnstr) call wrndie(-3, 'BLaDE', 'CONS HARM restraints not yet supported with BLaDE')
     ! Harmonic restraints implemented:
     call xtlaxsacc(qorth,xtlrot,xucell,xtlabc)
     if (QCNSTR) then
        do i = 1, NUMHSETS
           if (typhset(i)==0 .and. kcexpn(i)>0) then
              if (xhscale(i)==1 .and. yhscale(i)==1 .and. zhscale(i)==1) then
                 found=.false.
                 do atom_i = 1, natom
                    if (ihset(atom_i)==i .and. kcnstr(atom_i)/=0) then
                       if (qorth) then
                          x=refx(atom_i)
                          y=refy(atom_i)
                          z=refz(atom_i)
                       else
                          x=xtlrot(1,1)*refx(atom_i)+xtlrot(1,2)*refy(atom_i)+xtlrot(1,3)*refz(atom_i)
                          y=xtlrot(2,1)*refx(atom_i)+xtlrot(2,2)*refy(atom_i)+xtlrot(2,3)*refz(atom_i)
                          z=xtlrot(3,1)*refx(atom_i)+xtlrot(3,2)*refy(atom_i)+xtlrot(3,3)*refz(atom_i)
                       endif
                       call blade_add_harmonic(system, atom_i, kcnstr(atom_i), &
                            x, y, z, &
                            real(kcexpn(i),c_double))
                       found=.true.
                    endif
                 enddo
                 ! Make sure there are atoms of this type with non-zero force constants
                 if (found) then
                    if (PRNLEV > 2) then
                       write (OUTU, '(x,a,i4)') &
                         ": Setting up atom restraints for set ", i
                       write (OUTU, '(x,a,i4)') &
                         ": Using exponent of ", KCEXPN(i)
                    endif
                 endif
              else
                 if (PRNLEV > 2) write (OUTU, '(x,a,i2,a,/,10x,a)') &
                      ": Restraint set ", i, "w/ x(y,z)scale /= 1 or exponent <= 0 ", &
                      "not supported through BLaDE, no restraint of this type applied"
              endif
           else
              if (PRNLEV > 2) write (OUTU, '(x,a,i2,a,/,10x,a)') &
                   ": Restraint set ", i, " not applied, bestfit/relative or exponent <= 0", &
                   "not supported through BLaDE"
           endif
        enddo
     endif

     ! MMFP restraints
     ! subroutine void blade_add_boRest(system, i, j, kr, r0, lambdaBlock) bind(c)
     ! subroutine void blade_add_anRest(system, i, j, k, kt, t0, lambdaBlock) bind(c)
     ! subroutine void blade_add_diRest(system, i, j, k, l, kphi, nphi, phi0, width, lambdaBlock) bind(c)
     if (qgeo) then
        j = 1 ! j is the start of the next geo group
        do i=1, NTGEO
           if (ngeo(i+1)-ngeo(i) .ne. 1) call wrndie(-3, 'BLaDE', 'BLaDE MMFP only supports groups of one atom')
           if (jgeo(i) .ne. 3) call wrndie(-3, 'BLaDE', 'BLaDE MMFP only supports harmonic restraints')
           if (i==j) then ! skip this group if it's already included
              ! Set block on all to 1 to treat them as environment and not scale by lambda
              if (igeo(i) .eq. -1) then ! bonds
                 if (prnlev > 2) write (OUTU, '(a,i8,i8,a,d12.6,d12.6)') &
                    "mmfp bond",lstgeo(ngeo(i)),lstgeo(ngeo(i+1)),"parameters",fcgeo(i),drgeo(i)
                 call blade_add_boRest(system,lstgeo(ngeo(i)),lstgeo(ngeo(i+1)), &
                    fcgeo(i),drgeo(i),blgeo(i)) ! drgeo read in from droff
                 j=j+2
              else if (igeo(i) .eq. -4) then ! angles
                 if (prnlev > 2) write (OUTU, '(a,i8,i8,i8,a,d12.6,d12.6)') &
                    "mmfp angle",lstgeo(ngeo(i)),lstgeo(ngeo(i+1)),lstgeo(ngeo(i+2)),"parameters",fcgeo(i),trgeo(i)
                 ! trgeo is where potential is centered, dtgeo sometimes gives a flat bottom region but not for jgeo=3
                 call blade_add_anRest(system,lstgeo(ngeo(i)),lstgeo(ngeo(i+1)),lstgeo(ngeo(i+2)), &
                    fcgeo(i),trgeo(i),blgeo(i)) ! trgeo read in from tref
                 j=j+3
              else if (igeo(i) .eq. -5) then ! dihedrals
                 if (prnlev > 2) write (OUTU, '(a,i8,i8,i8,i8,a,d12.6,d12.6)') &
                    "mmfp dihedral",lstgeo(ngeo(i)),lstgeo(ngeo(i+1)),lstgeo(ngeo(i+2)),lstgeo(ngeo(i+3)),"parameters",fcgeo(i),trgeo(i)
                 ! trgeo is where potential is centered, dtgeo sometimes gives a flat bottom region but not for jgeo=3
                 ! nphi=0 indicates harmonic dihedral restraint, width=0 means no flat-bottom
                 call blade_add_diRest(system,lstgeo(ngeo(i)),lstgeo(ngeo(i+1)),lstgeo(ngeo(i+2)),lstgeo(ngeo(i+3)), &
                    fcgeo(i),0,trgeo(i),0.0d0,blgeo(i)); ! trgeo read in from tref
                 j=j+4
              else
                 call wrndie(-3, 'BLaDE', 'BLaDE MMFP only supports bonds, angles, and dihedrals')
                 j=j+1
              endif
           endif
        enddo
     endif

     ! CONS DIHE restraints
     ! Export dihedral restraints from CONS DIHE command
     if (ncsphi > 0) then
        do i=1, ncsphi
           if (prnlev > 2) write (OUTU, '(a,i8,i8,i8,i8,a,d12.6,d12.6,i4,d12.6)') &
              "cons dihe",ics(i),jcs(i),kcs(i),lcs(i),"parameters",ccsc(i),ccsb(i),ccsd(i),ccsw(i)
           ! ccsc=force, ccsb=min_angle (radians), ccsd=periodicity, ccsw=width (radians)
           ! lambdaBlock=1 means environment (not scaled by lambda)
           ! Convert ccsc from CHARMM convention (k*dphi^2) to BLaDE convention (0.5*k*dphi^2)
           ! Convert radians to degrees for BLaDE API (which expects degrees)
           convert=1
           if (ccsd(i)==0) convert=2
           call blade_add_diRest(system,ics(i),jcs(i),kcs(i),lcs(i), &
              convert*ccsc(i),ccsd(i),ccsb(i)*raddeg,ccsw(i)*raddeg,1)
        enddo
     endif

     ! CONS RESD restraints for a pair of distances
     if (rednum .gt. 0) then
      do i = 1, rednum
         if ((redipt(i+1)-redipt(i)) .ne. 2) call wrndie(-5, 'BLaDE', &
                   'BLaDE RESD only supports linear combination of two distances')

         call blade_add_resd(system, &
                          redilis(1,redipt(i)+1), redilis(2,redipt(i)+1), redilis(1,redipt(i)+2), redilis(2,redipt(i)+2), &
                          redklis(redipt(i)+1), redklis(redipt(i)+2), redrval(i), redkval(i))
      end do
     end if  ! eeresd
     
#if KEY_MLMM==1 && KEY_MLPTORCH==1
       ! BLaDE compatibility checks for MLMM
        if (with_pyth == 1) then
            call wrndie(-5,'', ' MLMM> PYTH Interface is not supported with BLADE.')
        end if
        if ( with_libtorch == 1 .and. with_dpmm==1 ) then
            call wrndie(-5,'', ' MLMM> DPMM is not supported with BLADE. Please disable DPMM to use BLADE.')
        end if
        if ( with_libtorch == 1 .and. with_tani==1 ) then
            if (mlps_scale_use) then
                call wrndie(-5,'', ' MLMM> The SCALe keyword is not supported with BLADE. Please disable SCALe to use BLADE.')
            end if
        end if
        !

     if ( mlnm2 > 0 .and. with_libtorch==1 .and. with_tani == 1) then ! eemlp
        if (mlps_ene_scale /= 1.0) call wrndie(-5, 'BLaDE', 'TorchANI energy scaling is currently not implemented for BLaDE')
        call blade_add_tani(system, with_tani_c, mlps_ptname_c, mlnm2_c, &
                               ml_idx_c, ml_qmZid_c, ml_imask_c, natom_cache_c)
     else if ( mlnm2 > 0 .and. (.not. with_tani == 1)) then
      call wrndie(-5, 'BLaDE', 'Only TorchANI model with LibTorch backend is currently supported for BLaDE')
     end if ! eemlp
#endif      

  end subroutine export_psf_to_blade

  subroutine export_param_to_blade()
    use, intrinsic :: iso_c_binding, only: c_associated, c_char, c_null_char
    use consta, only: raddeg
    use inbnd, only: e14fac, e14ff ! LUIS
    use psf, only: natom, iac ! LUIS
    use inbnd, only: e14fac, eps, nbxmod
    use defltsm, only: dfgeom
    use param, only: &
         natc, atc &  ! atoms
         , qflxparm &
         , ncb, cbb, cbc, kcb &  ! bonds
         , nct, ctb, ctc, ctub, ctuc, kct &  ! angles
         , ncp, cpb, cpc, cpd, kcp & ! dihedral angles
         , nci, cib, cic, cid, kci &  ! impropers
#if KEY_FLEXPARM == 1
         , cbai, cbaj &  ! bonds
         , ctai, ctaj, ctak &  ! angles
         , cpai, cpaj, cpak, cpal &  ! dihedral angles
         , ciai, ciaj, ciak, cial &  ! impropers
#endif /* KEY_FLEXPARM */
         , eff, vdwr, itc, maxatc &  ! nonbonded parameters
         , nbfixn, nbfixi, nbfixr    ! nbfix parameters

#if KEY_CMAP == 1
    use cmapm, only: nctp, gsctp, mctp, ctpa  ! source/energy/ecmaps.F90
#endif /* KEY_CMAP */

    use rtf, only: armass

    use blade_param_module, only: blade_init_parameters, &
         blade_add_parameter_atoms, &
         blade_add_parameter_bonds, &
         blade_add_parameter_angles, &
         blade_add_parameter_dihes, &
         blade_add_parameter_imprs, &
         blade_add_parameter_cmaps, &
         blade_add_parameter_cmaps_fill, &
         blade_add_parameter_nbonds, &
         blade_add_parameter_nbfixs

    implicit none

    integer :: i, atom_i, atom_j, atom_k, atom_l
    integer :: itemp
    integer(chm_int8) :: natc2, kcpi, kcii
    logical :: lflip
    integer :: i1, i2
    character(len=8) :: name_i, name_j, name_k, name_l
    real(chm_real),dimension(natc) :: e14fvec ! LUIS

    call blade_init_parameters(system)

    do i = 1, natc
       call blade_add_parameter_atoms(system, trim(atc(i)) // c_null_char, armass(i))
    end do

    do i = 1, ncb
#if KEY_FLEXPARM == 1
       if (qflxparm) then
          atom_i = cbai(i)
          atom_j = cbaj(i)
       else
#endif
          ! see io/parmio.F90
          atom_i=sqrt(2.0*kcb(i))+0.5
          atom_j=kcb(i)-atom_i*(atom_i-1)/2
#if KEY_FLEXPARM == 1
       endif
#endif
       name_i = trim(atc(atom_i)) // c_null_char
       name_j = trim(atc(atom_j)) // c_null_char

       call blade_add_parameter_bonds(system, name_i, name_j, cbc(i), cbb(i))
    end do

    do i = 1, nct
#if KEY_FLEXPARM == 1
       if (qflxparm) then
          atom_i = ctai(i)
          atom_j = ctaj(i)
          atom_k = ctak(i)
       else
#endif
          ! see io/parmio.F90
          itemp=(kct(i)-1)/natc
          atom_j=kct(i)-itemp*natc
          atom_i=sqrt(2.0*itemp)+0.5
          atom_k=itemp-atom_i*(atom_i-1)/2
#if KEY_FLEXPARM == 1
       endif
#endif
       name_i = trim(atc(atom_i)) // c_null_char
       name_j = trim(atc(atom_j)) // c_null_char
       name_k = trim(atc(atom_k)) // c_null_char

       call blade_add_parameter_angles(system, &
            name_i, name_j, name_k, &
            ctc(i), ctb(i) * raddeg, ctuc(i), ctub(i))
    end do

    do i = 1, ncp
#if KEY_FLEXPARM == 1
       if (qflxparm) then
          atom_i = cpai(i)
          atom_j = cpaj(i)
          atom_k = cpak(i)
          atom_l = cpal(i)
       else
#endif
          ! see io/parmio.F90, I1=atom_j, J1=atom_k, K1=atom_i, L1=atom_l
          atom_i=-1
          atom_l=-1
          kcpi=kcp(i)
          natc2=natc*(natc+1)/2
          if (.not.(kcpi <= natc2)) then
             lflip=.false.
             if (kcpi > natc2*natc2) then
                kcpi=kcpi-natc2*natc2
                lflip=.true.
             endif
             atom_j=kcpi/natc2
             kcpi=kcpi-atom_j*natc2
             atom_i=sqrt(2.0*atom_j)+0.5
             atom_l=atom_j-atom_i*(atom_i-1)/2
             if (lflip) then
                itemp=atom_i
                atom_i=atom_l
                atom_l=itemp
             endif
          endif
          atom_j=sqrt(2.0*kcpi)+0.5
          atom_k=kcpi-atom_j*(atom_j-1)/2
#if KEY_FLEXPARM == 1
       endif
#endif
       name_i = 'X' // c_null_char
       if (atom_i .ne. -1) then
          name_i = trim(atc(atom_i)) // c_null_char
       endif
       name_j = 'X' // c_null_char
       if (atom_j .ne. -1) then
          name_j = trim(atc(atom_j)) // c_null_char
       endif
       name_k = 'X' // c_null_char
       if (atom_k .ne. -1) then
          name_k = trim(atc(atom_k)) // c_null_char
       endif
       name_l = 'X' // c_null_char
       if (atom_l .ne. -1) then
          name_l = trim(atc(atom_l)) // c_null_char
       endif

       call blade_add_parameter_dihes(system, &
            name_i, name_j, name_k, name_l, &
            cpc(i), abs(cpd(i)), cpb(i) * raddeg)
       ! Sign of cpd is used to determine whether dihedral parameter has additional fourier components. BLaDE takes care of that automatically.
    end do

    do i = 1, nci
#if KEY_FLEXPARM == 1
       if (qflxparm) then
          atom_i = ciai(i)
          atom_j = ciaj(i)
          atom_k = ciak(i)
          atom_l = cial(i)
       else
#endif
          ! see io/parmio.F90, I1=atom_i, J1=atom_l, K1=atom_j, L1=atom_k
          atom_j=-1
          atom_k=-1
          kcii=kci(i)
          if (kcii > natc2) then
             !           no wildcards
             lflip=.false.
             if (kcii > natc2*natc2) then
                kcii=kcii-natc2*natc2
                lflip=.true.
             endif
             atom_i=kcii/natc2
             kcii=kcii-atom_i*natc2
             atom_j=sqrt(2.0*atom_i)+0.5
             atom_k=atom_i-atom_j*(atom_j-1)/2
             if (lflip) then
                itemp=atom_j
                atom_j=atom_k
                atom_k=itemp
             endif
             atom_i=sqrt(2.0*kcii)+0.5
             atom_l=kcii-atom_i*(atom_i-1)/2
             !
          elseif (kcii > 0) then
             !           wildcard type  A - X - X - B
             atom_i=sqrt(2.0*kcii)+0.5
             atom_l=kcii-atom_i*(atom_i-1)/2
             !
          elseif (kcii < -natc**3-natc**2) then
             !           wildcard type X - A - B - X
             kcii=-(natc**3+natc**2+kcii)
             atom_j=sqrt(2.0*kcii)+0.5
             atom_k=kcii-atom_j*(atom_j-1)/2
             atom_i=-1
             atom_l=-1
             !
          elseif (kcii < -natc*natc) then
             !           wildcard type  X - A - B - C
             atom_i=-1
             kcii=-kcii
             atom_l=(kcii/(natc*natc))
             kcii=kcii-natc*natc*(atom_l)
             atom_j=(kcii/natc)+1
             atom_k=kcii-natc*(atom_j-1)
             !
          else
             !           wildcard type X - X - A - B
             atom_i=-1
             atom_j=-1
             kcii=-kci(i)
             atom_k=(kcii/natc)+1
             atom_l=kcii-natc*(atom_k-1)
          endif
          ! see io/parmio.F90, I1=atom_i, J1=atom_l, K1=atom_j, L1=atom_k
#if KEY_FLEXPARM == 1
       endif
#endif
       name_i = 'X' // c_null_char
       if (atom_i .ne. -1) then
          name_i = trim(atc(atom_i)) // c_null_char
       endif
       name_j = 'X' // c_null_char
       if (atom_j .ne. -1) then
          name_j = trim(atc(atom_j)) // c_null_char
       endif
       name_k = 'X' // c_null_char
       if (atom_k .ne. -1) then
          name_k = trim(atc(atom_k)) // c_null_char
       endif
       name_l = 'X' // c_null_char
       if (atom_l .ne. -1) then
          name_l = trim(atc(atom_l)) // c_null_char
       endif

       call blade_add_parameter_imprs(system, &
            name_i, name_j, name_k, name_l, &
            cic(i), cid(i), cib(i) * raddeg)
    end do

#if KEY_CMAP == 1
    do i = 1, nctp
#if KEY_FLEXPARM == 1
       if (.not.qflxparm) then
#endif
          call wrndie(-2, 'BLaDE', 'BLaDE requires flex parameters for CMAP interactions. Please turn on flex parm')
#if KEY_FLEXPARM == 1
       endif
#endif

       call blade_add_parameter_cmaps(system, &
            trim(atc(ctpa(i, 3))) // c_null_char, &
            trim(atc(ctpa(i, 1))) // c_null_char, &
            trim(atc(ctpa(i, 2))) // c_null_char, &
            trim(atc(ctpa(i, 4))) // c_null_char, &
            trim(atc(ctpa(i, 7))) // c_null_char, &
            trim(atc(ctpa(i, 5))) // c_null_char, &
            trim(atc(ctpa(i, 6))) // c_null_char, &
            trim(atc(ctpa(i, 8))) // c_null_char, &
            gsctp(i))

       do i1 = 1, gsctp(i)
          do i2 = 1, gsctp(i)
             call blade_add_parameter_cmaps_fill(system, &
                  trim(atc(ctpa(i, 3))) // c_null_char, &
                  trim(atc(ctpa(i, 1))) // c_null_char, &
                  trim(atc(ctpa(i, 2))) // c_null_char, &
                  trim(atc(ctpa(i, 4))) // c_null_char, &
                  trim(atc(ctpa(i, 7))) // c_null_char, &
                  trim(atc(ctpa(i, 5))) // c_null_char, &
                  trim(atc(ctpa(i, 6))) // c_null_char, &
                  trim(atc(ctpa(i, 8))) // c_null_char, &
                  i1,i2,mctp(i)%grid(1)%A(i2,i1))
          enddo
       enddo
    end do
#endif /* KEY_CMAP */

    if (eps .ne. 1) call wrndie(-3, 'BLaDE', 'BLaDE only supports eps=1')
    if (nbxmod .ne. 5) call wrndie(-3, 'BLaDE', 'BLaDE only supports nonbonded exclusion mode nbxmod=5')

    do i = 1, natom ! LUIS
       e14fvec(itc(iac(i))) = e14ff(i) ! LUIS
    enddo ! LUIS

    do i = 1, natc
       name_i = trim(atc(i)) // c_null_char
       call blade_add_parameter_nbonds(system, name_i, &
            eff(itc(i)), vdwr(itc(i)), eff(itc(i)+maxatc), vdwr(itc(i)+maxatc), &
            !e14fac, & ! not LUIS
            e14fvec(itc(i)), & ! LUIS
            merge(1,0,dfgeom))
    end do

    do i = 1, nbfixn
       atom_i = nbfixi(1, i)
       name_i = trim(atc(atom_i)) // c_null_char

       atom_j = nbfixi(2, i)
       name_j = trim(atc(atom_j)) // c_null_char

       call blade_add_parameter_nbfixs(system, name_i, name_j, &
            nbfixr(1, i), nbfixr(2, i), nbfixr(3, i), nbfixr(4, i))
    end do
  end subroutine export_param_to_blade

  subroutine export_coords_to_blade()
    use blade_coords_module, only: &
         blade_init_coordinates, &
         blade_add_coordinates_position, &
         blade_add_coordinates_velocity, &
         blade_add_coordinates_box

    use psf, only: natom
    use coord, only: x, y, z
    use image, only: xtltyp, xucell, xtlabc, qorth, xtlrot
    use number, only: zero

    implicit none

    integer :: i, boxname

    call blade_init_coordinates(system, natom)

    ! if ((xtltyp .ne. 'ORTH') .and. (xtltyp .ne. 'TETR') .and. (xtltyp .ne. 'CUBI')) then
    !    call wrndie(-2, 'BLaDE', 'BLaDE requires an orthorhombic, tetragonal, or cubic box')
    ! end if
    if (xtltyp .eq. 'CUBI') then
       boxname=0
    elseif (xtltyp .eq. 'TETR') then
       boxname=1
    elseif (xtltyp .eq. 'ORTH') then
       boxname=2
    elseif (xtltyp .eq. 'MONO') then
       boxname=3
    elseif (xtltyp .eq. 'TRIC') then
       boxname=4
    elseif (xtltyp .eq. 'HEXA') then
       boxname=5
    elseif (xtltyp .eq. 'RHOM') then
       boxname=6
    elseif (xtltyp .eq. 'OCTA') then
       boxname=7
    elseif (xtltyp .eq. 'RHDO') then
       boxname=8
    else
       call wrndie(-5, 'BLaDE', 'What kind of crystal is this?')
    endif

    call blade_add_coordinates_box(system, boxname, &
         xucell(1), xucell(2), xucell(3), &
         xucell(4), xucell(5), xucell(6))

    call xtlaxsacc(qorth,xtlrot,xucell,xtlabc)

    do i = 1, natom
       if (qorth) then
          call blade_add_coordinates_position(system, i, x(i), y(i), z(i))
       else
          call blade_add_coordinates_position(system, i, &
             xtlrot(1,1)*x(i)+xtlrot(1,2)*y(i)+xtlrot(1,3)*z(i), &
             xtlrot(2,1)*x(i)+xtlrot(2,2)*y(i)+xtlrot(2,3)*z(i), &
             xtlrot(3,1)*x(i)+xtlrot(3,2)*y(i)+xtlrot(3,3)*z(i))
       endif
       call blade_add_coordinates_velocity(system, i, zero, zero, zero)
    end do
  end subroutine export_coords_to_blade

#if KEY_BLOCK == 1
  subroutine export_block_to_blade()
    use blade_block_module, only: &
         blade_init_msld, &
         blade_add_msld_atomassignment, &
         blade_add_msld_initialconditions, &
         blade_add_msld_termscaling, &
         blade_add_msld_flags, &
         blade_add_msld_charges, &
         blade_add_msld_bias, &
         blade_add_msld_thetacollbias, &
         blade_add_msld_thetaindebias, &
         blade_set_msld_thetaedgebias, &
         blade_set_msld_piecewise_constraint, &
         blade_add_msld_softbond, &
         blade_add_msld_atomrestraint, &
         blade_add_msld_atomrestraint_element, &
         blade_add_msld_block_friction, &
         blade_set_msld_block_fixed

    use block_ltm, only: nblock, iblckp
    use block_fcm, only: qnobo, qnoub, qnoan, qnoim, qnoph, qnoct
    use lambdam, only: qmld, nsitemld, nsubmld, blckmld, tbld, &
         isitemld, thetamld, thetavmld, thetam, bielam, bimlam, biblam, &
         iqldm_softcore, iqldm_pme, &
         sobob, sobon, soboa, &
         nsobo, soboi, soboj, &
         nbiasv, ibvidi, ibvidj, ibclas, irreup, irrlow, ikbias, ipbias, &
         qbiasthetacoll, qbiasthetainde, qbiasthetaedge, qbiasthetaflat, &
         ikbiasthetacoll, ipbiasthetacoll, ikbiasthetainde, &
         ikbiasthetaedge,inbiasthetaedge,iabiasthetaedge,itbiasthetaedge, &
         iwbiasthetaflat, ikbiasthetaflat, &
         ik1biascharge, ik2biascharge, ik3biascharge, iq03biascharge, iw3biascharge, &
         thetabib, fnexp_factor, fcnal_form, &
         qldm_scalecons, kscalecons, nires, iblcks_ires, iblcks_iat, &
         qlfix
    use psf, only: natom
    use number, only: zero, one
    use consta, only: timfac ! ps^-1
    use reawri, only: dyn_tbath

    implicit none

    integer :: i, j, k
    integer :: jmax

    if (qmld) then
       call blade_init_msld(system, nblock-1)

       do i = 1, natom
          call blade_add_msld_atomassignment(system, i, iblckp(i))
       enddo

       do i = 1, nsitemld
          jmax=nsubmld(i)
          if (i==1) then
             jmax=1
             k=1
          endif
          do j = 1, jmax
             if (i/=1) then
                k = blckmld(i,j)
             endif
             call blade_add_msld_initialconditions(system, k, isitemld(k), &
                  thetamld(i,j), thetavmld(i,j), bimlam(k), -bielam(k), zero)
             ! Last argument is total charge of block, currently unused.
          enddo
       enddo

       call blade_add_msld_termscaling(system, &
            1-merge(1,0,qnobo), 1-merge(1,0,qnoub), &
            1-merge(1,0,qnoan), 1-merge(1,0,qnoph), &
            1-merge(1,0,qnoim), 1-merge(1,0,qnoct))

       if (fcnal_form /= 'nexp' &
            .and. fcnal_form /= 'fixd' &
            .and. fcnal_form /= 'npwi') &
            call wrndie(-3, 'BLaDE', 'BLaDE only supports functional forms fnex, ffix, or fpwise')

       if ((fcnal_form == 'npwi') .xor. qbiasthetaflat) &
          call wrndie(-3, 'BLaDE', 'Error, BLaDE uses thbv flat to determine if fnpwise is set')
       ! Don't forget to change function declarations in source/blade_api/blade_block.F90
       call blade_add_msld_flags(system, &
            thetabib*timfac, fnexp_factor, &
            merge(tbld, dyn_tbath, tbld /= zero), &
            merge(1,0,iqldm_softcore>=1), merge(1,0,iqldm_softcore>=2), &
            iqldm_pme, &
            kscalecons, &
            1/sqrt(soboa),sobob,sobon, &
            merge(1,0,fcnal_form=='fixd'))

       call blade_add_msld_charges(system, ik1biascharge, ik2biascharge, ik3biascharge, iq03biascharge, iw3biascharge)

       ! Pass per-block friction values (scaled to internal units)
       do k = 1, nblock
          call blade_add_msld_block_friction(system, k, biblam(k)*timfac)
       enddo

       ! Pass per-block fixed flags to BLaDE
       do k = 1, nblock
          if (qlfix(k)) then
             call blade_set_msld_block_fixed(system, k, 1)
          endif
       enddo

       do i = 1, nbiasv
          if (ibclas(i)/=2) then
             call blade_add_msld_bias(system, ibvidi(i), ibvidj(i), &
                  ibclas(i), irreup(i), ikbias(i), ipbias(i))
          else
             call blade_add_msld_bias(system, ibvidi(i), ibvidj(i), &
                  ibclas(i), irrlow(i), ikbias(i), ipbias(i))
          endif
       enddo

       if (qbiasthetacoll) then
          do i = 2, nsitemld
             call blade_add_msld_thetacollbias(system,nsitemld,i,ikbiasthetacoll(i),ipbiasthetacoll(i))
          enddo
       endif
       if (qbiasthetainde) then
          do i = 2, nsitemld
             call blade_add_msld_thetaindebias(system,nsitemld,i,ikbiasthetainde(i))
          enddo
       endif
       if (qbiasthetaedge) then
          call blade_set_msld_thetaedgebias(system,ikbiasthetaedge,inbiasthetaedge,iabiasthetaedge,itbiasthetaedge)
       endif
       if (qbiasthetaflat) then
          do i = 2, nsitemld
             ! blade assumes qbiasthetaflat is set if and only if fcnal_form=='npwi'
             ! blade also assumes iwbiasthetaflat and ikbiasthetaflat are same for all sites
             if (i>2) then
                if (iwbiasthetaflat(i).ne.iwbiasthetaflat(i-1)) &
                   call wrndie(-3, 'BLaDE', 'Error, BLaDE does not support different thbv flat w values at different sites')
                if (ikbiasthetaflat(i).ne.ikbiasthetaflat(i-1)) &
                   call wrndie(-3, 'BLaDE', 'Error, BLaDE does not support different thbv flat k values at different sites')
             endif
             call blade_set_msld_piecewise_constraint(system, merge(1,0,qbiasthetaflat), iwbiasthetaflat(i), ikbiasthetaflat(i))
          enddo
       endif

       do i = 1, nsobo
          call blade_add_msld_softbond(system,soboi(i),soboj(i))
       enddo

       do i = 1, nires
          call blade_add_msld_atomrestraint(system)
          do j = iblcks_ires(i)+1, iblcks_ires(i+1)
             call blade_add_msld_atomrestraint_element(system,iblcks_iat(j))
          enddo
       enddo
    else
       call blade_init_msld(system, 0)

       do i = 1, natom
          call blade_add_msld_atomassignment(system, i, 1)
       enddo

       call blade_add_msld_termscaling(system, &
            0, 0, &
            0, 0, &
            0, 0) ! scaling irrelevant, set to false

       call blade_add_msld_flags(system, &
            1.0*timfac, 5.5*one, & ! gamma and fnex irrelevant
            -one, & ! lambda temperature follows the initial main temperature
            0, 0, & ! Softcores irrelevant
            1, & ! PME irrelevant
            zero, &
            one,one,one, & ! soft bonds irrelevant
            0) ! fix irrelevant

       call blade_add_msld_charges(system,zero,zero,zero,zero,zero)
    endif

  end subroutine export_block_to_blade
#endif /* KEY_BLOCK */

  subroutine export_options_to_blade
    use blade_options_module, only: blade_init_run, blade_add_run_flags
    use cnst_fcm, only: fbeta
    use inbnd, only: ctonnb, ctofnb, lvfswt, lvdw, lvgrom, lvshft, &
       lcons, lfswt, lshft, legrom
    use ewald, only: lewald
    use pme_module, only: qpme
    use ewald, only: kappa
    use pmeutil, only: nfft1, nfft2, nfft3, forder
    use image, only: xucell
    use shake, only: shktol
    use stream, only : outu, prnlev
    use consta, only: timfac
    use energym
    use psf, only: natom
    use number, only: one
    use lambdam, only: qmld, iqldm_pme

    implicit none

    ! real(chm_real) :: blade_gamma, blade_gridspace
    real(chm_real) :: blade_gamma
    logical :: l_elec_pme, l_elec_fswitch, l_elec_fshift
    integer :: i, lvfswt_i, lpme_i
    logical :: lvswt

    lvswt = lvdw .and. .not. ( lvfswt .or. lvshft .or. lvgrom )
    ! vdWfSwitch mapping: 0=VFSWITCH, 1=VSWITCH, 2=VSHIFT
    lvfswt_i = 0
    if (lvfswt) lvfswt_i = 0
    if (lvswt) lvfswt_i = 1
    if (lvshft) lvfswt_i = 2

    ! Didn't bother to check lelec or other flags in inbnd module
    l_elec_pme = lewald .and. qpme
    l_elec_fswitch = lcons .and. lfswt .and. &
       .not. (lshft .or. lewald .or. legrom)
    l_elec_fshift = lcons .and. lfswt .and. &
       lshft .and. .not. (lewald .or. legrom)
    lpme_i=1
    if (l_elec_fswitch) lpme_i=0
    if (l_elec_pme) lpme_i=1
    if (l_elec_fshift) lpme_i=2

    call blade_init_run(system)

    if (allocated(fbeta)) then
       do i=1,(natom-1)
          if (fbeta(i) /= fbeta(i+1)) call wrndie(-5, 'BLaDE', 'fbeta must be the same and equal to for all particles')
       enddo
       blade_gamma = fbeta(1) * timfac
       if(prnlev >= 2) write(outu,101) blade_gamma
       ! call wrndie(1, 'BLaDE', 'Just assumed fbeta was the same and equal to for all particles')
    else
       blade_gamma = 0.1 * timfac
       if(prnlev >= 2) write(outu,101) blade_gamma
       call wrndie(1, 'BLaDE', 'No fbeta value available, used default value of 0.1 ps^-1')
    endif
101 format('Assuming fbeta = ', F10.5)
    if (blade_gamma == 0) call wrndie(1, 'BLaDE', 'BLaDE langevin integrator using fbeta=0 for constant energy dynamics.')

    if (.not. (l_elec_pme .or. l_elec_fswitch .or. l_elec_fshift)) then
       call wrndie(-5, 'BLaDE', 'PME, FSWITCH, or FSHIFT is required for BLaDE')
    endif
    if (l_elec_pme) then
       if (qmld) then
          select case (iqldm_pme)
          case (1)
             call wrndie(1,'BLaDe','PME being used for electrostatics, MSLD PMEL ON mode')
          case (2)
             call wrndie(1,'BLaDe','PME being used for electrostatics, MSLD PMEL EX mode')
          case (3)
             call wrndie(1,'BLaDe','PME being used for electrostatics, MSLD PMEL NN mode')
          case default
             call wrndie(-5,'BLaDe','PMEL ON, EX, or NN must be set in BLOCK when using PME for electrostatics in BLaDE with BLOCK')
          end select
       else
          call wrndie(1,'BLaDe','PME being used for electrostatics in BLaDE')
       endif
    endif
    if (l_elec_fswitch) call wrndie(1,'BLaDe','FSWITCH being used for electrostatics in BLaDE')
    if (l_elec_fshift) call wrndie(1,'BLaDe','FSHIFT being used for electrostatics in BLaDE')

    if (.not.(lvfswt .or. lvswt .or. lvshft)) then
       call wrndie(-5, 'BLaDE', 'VFSWITCH, VSWITCH, or VSHIFT is required for BLaDE')
    endif
    if (lvfswt) call wrndie(1,'BLaDe','VFSWITCH being used for vdW switching in BLaDE')
    if (lvswt) call wrndie(1,'BLaDe','VSWITCH being used for vdW switching in BLaDE')
    if (lvshft) call wrndie(1,'BLaDe','VSHIFT being used for vdW switching in BLaDE')

    if(prnlev >= 2) write(outu,102) kappa
102    format('Found pme kappa = ', F10.5)

    if(prnlev >= 2) write(outu,103) ctofnb, ctonnb
103    format('Found ctofnb = ', F10.5, ' ctonnb = ', F10.5)

    ! call blade_add_run_flags(system, blade_gamma, kappa, ctofnb, ctonnb, blade_gridspace, forder, shktol)
    call blade_add_run_flags(system, blade_gamma, kappa, ctofnb, ctonnb, &
         lvfswt_i, lpme_i, &
         -one, nfft1, nfft2, nfft3, forder, shktol)

    ! Apply skipe commands
    call blade_set_calcTermFlag(system,1-1,merge(1,0,qeterm(bond))) ! eebond
    call blade_set_calcTermFlag(system,2-1,merge(1,0,qeterm(angle))) ! eeangle
    call blade_set_calcTermFlag(system,3-1,merge(1,0,qeterm(ureyb))) ! eeurey
    call blade_set_calcTermFlag(system,4-1,merge(1,0,qeterm(dihe))) ! eedihe
    call blade_set_calcTermFlag(system,5-1,merge(1,0,qeterm(imdihe))) ! eeimpr
    call blade_set_calcTermFlag(system,6-1,merge(1,0,qeterm(cmap))) ! eecmap
    if (qeterm(elec) .neqv. qeterm(vdw)) call wrndie(-3, 'BLaDE', 'SKIPE must be the same for vdw and elec terms. vdw and elec energy are both placed in vdw.')
    if (qeterm(elec) .neqv. qeterm(imelec)) call wrndie(-3, 'BLaDE', 'SKIPE must be the same for elec and imel terms. Apologies if you are using a module like domdec that does not even use imel.')
    if (qeterm(vdw) .neqv. qeterm(imvdw)) call wrndie(-3, 'BLaDE', 'SKIPE must be the same for vdw and imnb terms. Apologies if you are using a module like domdec that does not even use imnb.')
    if (prnlev >= 2 .and. qeterm(vdw)) write(outu,'(A)') 'Warning: Both elec and vdw energies placed in vdw'
    call blade_set_calcTermFlag(system,7-1,merge(1,0,qeterm(vdw))) ! eenb14
    call blade_set_calcTermFlag(system,8-1,merge(1,0,qeterm(vdw))) ! eenbdirect
    call blade_set_calcTermFlag(system,9-1,merge(1,0,qeterm(ewksum))) ! eenbrecip
    call blade_set_calcTermFlag(system,10-1,merge(1,0,qeterm(ewself))) ! eenbrecipself
    call blade_set_calcTermFlag(system,11-1,merge(1,0,qeterm(ewexcl))) ! eenbrecipexcl
    call blade_set_calcTermFlag(system,12-1,merge(1,0,qeterm(ldbv))) ! eelambda - msld biasing potentials
    call blade_set_calcTermFlag(system,13-1,merge(1,0,qeterm(thbv))) ! eetheta - msld theta biases
    call blade_set_calcTermFlag(system,14-1,merge(1,0,qeterm(cats))) ! eecats - BLOCK CATS restraints
    call blade_set_calcTermFlag(system,15-1,merge(1,0,qeterm(noe))) ! eenoe
    call blade_set_calcTermFlag(system,16-1,merge(1,0,qeterm(charm))) ! eeharmonic
    call blade_set_calcTermFlag(system,17-1,merge(1,0,qeterm(geo).or.qeterm(cdihe))) ! eemmfp and cons dihe
    call blade_set_calcTermFlag(system,18-1,merge(1,0,qeterm(resd))) ! eeresd
#if KEY_MLMM== 1 && KEY_MLPTORCH==1
    call blade_set_calcTermFlag(system,19-1,merge(1,0,qeterm(mlps))) ! eemlp
#endif
    ! 20 eebias - misc biases, and total charge restraints
    ! I don't think there's anything useful to do with qeprop
    ! eprop(epot)=blade_energy(21) ! eepotential
    ! eprop(totke)=blade_energy(22) ! eekinetic
    ! eprop(tote)=blade_energy(23) ! eetotal

  end subroutine export_options_to_blade
#endif /* KEY_BLADE */
end module blade_module
