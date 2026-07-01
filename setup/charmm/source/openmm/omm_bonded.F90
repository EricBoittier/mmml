!> Builds OpenMM Force objects to implement a CHARMM force field.
!> Mike Garrahan, 2011-12
module omm_bonded
#if KEY_OPENMM==1
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   use OpenMM
   implicit none

   private
   real(chm_real), allocatable, save, dimension(:) :: omm_index_torsion_k
   integer, save :: indexTorsion
   logical :: zero_bond, zero_angle, zero_urey, &
         zero_dihedral, zero_improper, zero_cmap

   public :: import_psf, reparameterize_torsions, transform_cmap

contains

   !> Converts PSF data into an OpenMM system.
   subroutine import_psf(system, nbopts)
      use omm_nbopts
      use omm_nonbond, only : setup_nonbonded
      use omm_gomodel, only : setup_gomodel
      use omm_switching, only : setup_switching
      use omm_gbsw, only : qgbsw_import_settings, setup_gbsw, qphmd_omm, qphmd_initialized
      use omm_gbsa, only : qgbsa, setup_gbsaobc2
      use omm_gbmv, only : setup_gbmv
      use omm_mses, only : setup_mses  ! AN EXAMPLE OF OPENMM PLUGIN
      use omm_ecomp, only : omm_init_eterms, omm_incr_eterms

      type(OpenMM_System), intent(inout) :: system
      type(omm_nbopts_t), intent(in) :: nbopts

      integer*4 group

      call mask_eterms()
      call omm_init_eterms
      call setup_atoms(system)
      group = omm_incr_eterms('bond')
      call setup_bonds(system, group)
      call setup_lonepairs(system)

      if (nbopts%use_etsr) then
         call wrndie(-1,'omm_bonded: import_psf', &
              'Problem with force field set-up: ETSR not supported with OpenMM')
         if(prnlev >= 2) write(outu,'(x,a)') &
              'CHARMM> CHARMM/OpenMM Interface Using ETEN Karanicolas/Brooks Go model'
         call setup_gomodel(system, nbopts)
      else if (nbopts%use_eten) then
         if(prnlev >= 2) write(outu,'(x,a)') &
              'CHARMM> CHARMM/OpenMM Interface Using ETEN Karanicolas/Brooks Go model'
         call setup_gomodel(system, nbopts)
      else 
         if ( (nbopts%use_omm_elec .and. nbopts%use_omm_vdw) ) then
            if(prnlev >=2) write(OUTU,'(a)') &
                 'CHARMM> Using OpenMM functionality for electrostatic and vdW interactions exclusively'
            if(prnlev >=2 .and. nbopts%use_vdw_ommswit) write(OUTU,'(a)') &
                 'CHARMM> with OpenMM vdW switching between ctonnb and ctofnb'
            call setup_nonbonded(system, nbopts)
         else if (nbopts%use_omm_elec) then
            if(prnlev >=2) write(OUTU,'(a)') &
                 'CHARMM> Using switching functions for vdW interactions and OpenMM PME/RxnFld for electrostatics'
            ! setup normal nonbonded functions for electrostatics
            call setup_nonbonded(system, nbopts)
            ! setup CHARMM specific switching
            call setup_switching(system, nbopts)
         else if ( .not. (nbopts%use_omm_elec .and. nbopts%use_omm_vdw) &
              .or. (.not. nbopts%use_omm_vdw .and. nbopts%use_omm_elec) ) then
            if(prnlev >=2) write(OUTU,'(a)') &
                 'CHARMM> Using switching functions for electrostatic and vdW interactions'
            call setup_switching(system, nbopts)
         else if (.not. nbopts%use_omm_elec .and. nbopts%use_omm_vdw) then
            call wrndie(-1,'omm_bonded: import_psf', &
                 'Problem with switching set-up: use_omm_vdw=true and use_omm_elec=false')
         endif
         if (PRNLEV >= 2) write (OUTU, '(a,/,a,l,a,l,a,l,a,l,a,l,/,a,l,a,l,a,l,a,l)') &
              'CHARMM> OpenMM switching function selection.', &
              'CHARMM> Electrostatic options: switch=',nbopts%use_elec_swit,' fswitch=', &
              nbopts%use_elec_fswit, ' fshift=',nbopts%use_elec_fshift,  &
              ' PME/Ewald=', nbopts%use_pme, ' RxnFld=', nbopts%use_elec_rxnfld, &
              'CHARMM> van der Waals options: vswitch=',nbopts%use_vdw_swit, &
              ' vfswitch=',nbopts%use_vdw_fswit, ' OpenMM vdW switch=',nbopts%use_vdw_ommswit, ' BLOCK vdW softcore:', nbopts%use_soft_omm

         if(nbopts%use_gbsaobc2) then
            group = omm_incr_eterms('gbenr')
            call setup_gbsaobc2(system, group, nbopts)
            if (PRNLEV >= 2) write (OUTU, '(a)') &
                 'CHARMM> OpenMM GBSA OBC2 method being used'
         endif
         
         if(nbopts%use_gbsw) then
            group = omm_incr_eterms('gbenr')
            call setup_gbsw(system, group, nbopts)
            if (PRNLEV >= 2) write (OUTU, '(a)') &
                 'CHARMM> OpenMM GBSW method being used'
         endif

         if(nbopts%use_gbmv) then
            group = omm_incr_eterms('gbenr')
            call setup_gbmv(system, group, nbopts)
            if (PRNLEV >= 2) write (OUTU, '(a)') &
                 'CHARMM> Only OpenMM CUDA platform available for GBMV method'
         endif

         if(nbopts%use_mses) then      ! AN EXAMPLE OF OPENMM PLUGIN
            group = omm_incr_eterms('gbenr') ! AN EXAMPLE OF OPENMM PLUGIN
            if (PRNLEV >= 2) write (OUTU, '(a)') &  ! AN EXAMPLE OF OPENMM PLUGIN
                 'CHARMM> OpenMM Reference, CPU and CUDA platform available for MSES method' ! AN EXAMPLE OF OPENMM PLUGIN
            call setup_mses(system, group, nbopts) ! AN EXAMPLE OF OPENMM PLUGIN
         endif ! AN EXAMPLE OF OPENMM PLUGIN

      endif
      group = omm_incr_eterms('angle')
      call setup_angles(system, group)
      group = omm_incr_eterms('dihe')
      call setup_torsions(system, group)
      group = omm_incr_eterms('imdihe')
      call setup_impropers(system,group)
      group = omm_incr_eterms('cmap')
      call setup_cmaps(system, group)
   end subroutine import_psf

   ! Sets flags according to any SKIPE commands in effect.
   subroutine mask_eterms()
      use energym

      zero_bond = .not. QETERM(BOND)
      zero_angle = .not. QETERM(ANGLE)
      zero_urey = .not. QETERM(UREYB)
      zero_dihedral = .not. QETERM(DIHE)
      zero_improper = .not. QETERM(IMDIHE)
      zero_cmap = .not. QETERM(CMAP)
   end subroutine mask_eterms

   subroutine setup_atoms(system)
      use psf, only: NATOM, AMASS, IMOVE
      use number, only: ZERO
      use omm_block, only : FindblockOneAtom, blockOneAtom
      type(OpenMM_System), intent(inout) :: system
      integer :: i, ijunk
      real*8 :: mass
      ! Setting the mass to zero for fixed atoms removes them from integration
      ! and should mimic CHARMM's CONS FIX function
      do i = 1, NATOM
         mass = AMASS(i)
         if(IMOVE(i) /=0) mass = zero
         ijunk = OpenMM_System_addParticle(system, mass)
      enddo
      if (PRNLEV >= 2 .and. any(IMOVE /=0)) write (OUTU, '(A)') &
           'CHARMM> OpenMM initiated with fixed atoms'
      ! Set-up block 1 atom reference if block is turned on
      blockOneAtom = FindblockOneAtom(natom)
   end subroutine setup_atoms

   ! Add lonepair interactions to the system.
   subroutine setup_lonepairs(system)
     use psf, only: NBOND, NTHETA, IB, JB
     use lonepr, only: NUMLP, NUMLPH, LPNHOST, LPVALUE, LPHPTR, LPWGHT, LPHOST
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_VirtualSite) this_Virtualsite
     type(OpenMM_LocalCoordinatesSite) this_site
     real*8 r,theta, phi, xweights(3), yweights(3), origin(3), local(3)
     integer*4 i, j
     integer*4 lpatom, atom1, atom2, atom3
     if(NUMLP == 0) return
     origin = [1.0d0,0.0d0,0.0d0]
     do i = 1, numlp
        !write(*,*) lpnhost(i),lphptr(i), &
        !     lpwght(i),lpvalue(1,i),lpvalue(2,i),lpvalue(3,i)
        r = lpvalue(1,i)/10.0
        lpatom = lphost(lphptr(i))
        atom1 = lphost(lphptr(i)+1)
        atom2 = lphost(lphptr(i)+2)
        if(lpnhost(i)>2) then
           atom3 = lphost(lphptr(i)+3)
           yweights = [0.0d0,-1.0d0,1.0d0]
           if(lpwght(i)) then
              xweights = [-1.0d0,0.0d0,1.0d0]
           else  
              r = -r
              xweights = [-1.0d0,0.5d0,0.5d0]
           endif
           theta = lpvalue(2,i) * acos(-1.0)/180.0d0
           phi = (180.0d0-lpvalue(3,i)) * acos(-1.0)/180.0d0
           local = [r*cos(theta), r*sin(theta)*cos(phi), r*sin(theta)*sin(phi)]
        else
           do j = 1, nbond
              if (ib(j) == atom2 .and. jb(j) /= atom1) then
                 atom3 = jb(j)
                 exit
              endif
              if (jb(j) == atom2 .and. ib(j) /= atom1) then
                 atom3 = ib(j)
                 exit
              endif
           enddo
           xweights = [1.0d0,-1.0d0,0.0d0]
           yweights = [0.0d0,-1.0d0,1.0d0]
           local = [r,lpvalue(2,i),lpvalue(3,i)]
        endif
        !write(*,*)'Lonepair atom',lpatom
        !write(*,*)'Atoms',atom1,atom2,atom3
        ! adjust to openmm atom numbers, shift by 1
        lpatom = lpatom - 1
        atom1 = atom1 - 1
        atom2 = atom2 - 1
        atom3 = atom3 - 1
        !write(*,*)'Origin: ',origin(1),origin(2),origin(3)
        !write(*,*)'Xweights: ',xweights(1),xweights(2),xweights(3)
        !write(*,*)'Yweights: ',yweights(1),yweights(2),yweights(3)
        !write(*,*)'Local: ',local(1),local(2),local(3)
        call OpenMM_LocalCoordinatesSite_create_2(this_site, atom1, atom2, atom3, &
             [origin(1),origin(2),origin(3)],[xweights(1),xweights(2),xweights(3)], &
             [yweights(1),yweights(2),yweights(3)], &
             [local(1),local(2),local(3)])
        this_Virtualsite = transfer(this_site,this_Virtualsite)
        call OpenMM_System_setVirtualSite(system,lpatom,this_Virtualsite)
     enddo
    return
   end subroutine setup_lonepairs

   ! Add bond interactions to the system.
   subroutine setup_bonds(system, group)
      use coord
      use psf, only: NBOND, NTHETA, IB, JB
      use code, only: ICB
      use param, only: CBB, CBC
      use omm_block, only : blockscale_bond
      use omm_ecomp, only : omm_incr_eterms

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(OpenMM_HarmonicBondForce) :: bonds
      integer*4 :: ii, iatom, jatom, katom, ic, ijunk
      real*8 :: r, k, r_init

      if(NBOND<=0 .and. NTHETA<=0) return
      call OpenMM_HarmonicBondForce_create(bonds)
      call OpenMM_Force_setForceGroup(   &
           transfer(bonds, OpenMM_Force(0)),group)
      ijunk = OpenMM_System_addForce(system, &
            transfer(bonds, OpenMM_Force(0)))

      do ii = 1, NBOND
         iatom = IB(ii)
         jatom = JB(ii)
         ic = ICB(ii)
         r = CBB(ic) / OpenMM_AngstromsPerNm
         if (zero_bond) then
            k = ZERO
         else
            k = blockscale_bond( iatom, jatom,  &
                 OpenMM_AngstromsPerNm**2 * OpenMM_KJPerKcal * CBC(ic) )
         endif
         if (PRNLEV >= 6) then
            r_init = sqrt((X(iatom)-X(jatom))**2 + &
                  (Y(iatom)-Y(jatom))**2 + &
                  (Z(iatom)-Z(jatom))**2) / OpenMM_AngstromsPerNm;
            if ((r_init / r) > 1.10) then
               write (OUTU, "(1x, a, i7, ', jatom = ', i7, ', r = ', f12.6, ' A')") &
                     'Long bond length: iatom = ', iatom, jatom, OpenMM_AngstromsPerNm * r_init
            endif
         endif
         ijunk = OpenMM_HarmonicBondForce_addBond(bonds, &
               iatom-1, jatom-1, r, 2*k)
      enddo

      ! combine Urey-Bradley with bonds because
      ! OpenMM Cuda platform allows only one HarmonicBondForce
      call setup_urey(bonds)
   end subroutine setup_bonds


   subroutine setup_urey(bonds)
      use psf, only: NTHETA, IT, KT
      use code, only: ICT
      use param, only: CTUB, CTUC
      use omm_block, only : blockscale_ub

      type(OpenMM_HarmonicBondForce), intent(inout) :: bonds
      integer :: ii, iatom, katom, ic, ijunk
      real(chm_real) :: r, k

      if(NTHETA<=0) return
      do ii = 1, NTHETA
         iatom = IT(ii)
         katom = KT(ii)
         ! do NOT append to bond_pairs
         ic = ICT(ii)
         if (CTUC(ic) == ZERO) cycle
         r = CTUB(ic) / OpenMM_AngstromsPerNm
         if (zero_urey) then
            k = ZERO
         else
            k = blockscale_ub( iatom, katom, &
                 OpenMM_AngstromsPerNm**2 * OpenMM_KJPerKcal * CTUC(ic) )
         endif
         ijunk = OpenMM_HarmonicBondForce_addBond(bonds, &
               iatom-1, katom-1, r, 2*k)
      enddo
   end subroutine setup_urey

   ! Add angles interactions to the system.
   subroutine setup_angles(system, group)
      use psf, only: NTHETA, IT, JT, KT
      use code, only: ICT
      use param, only: CTB, CTC
      use omm_block, only : blockscale_angle

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(OpenMM_HarmonicAngleForce) :: angles
      integer*4 :: ii, iatom, jatom, katom, ic, ijunk
      real*8 :: r, k

      if(NTHETA<=0) return
      call OpenMM_HarmonicAngleForce_create(angles)
      call OpenMM_Force_setForceGroup(   &
           transfer(angles, OpenMM_Force(0)),group)
      ijunk = OpenMM_System_addForce(system, &
                  transfer(angles, OpenMM_Force(0)))
      do ii = 1, NTHETA
         iatom = IT(ii)
         jatom = JT(ii)
         katom = KT(ii)
         ic = ICT(ii)
         r = CTB(ic)
         if (zero_angle) then
            k = ZERO
         else
            k = blockscale_angle( iatom, jatom, katom, OpenMM_KJPerKcal * CTC(ic) )
         endif
         ijunk = OpenMM_HarmonicAngleForce_addAngle(angles, &
               iatom-1, jatom-1, katom-1, r, 2*k)
      enddo
   end subroutine setup_angles

   ! Add dihedral angles interactions to the system.
   subroutine setup_torsions(system, group)
      use psf, only: NPHI, IP, JP, KP, LP
      use code, only: ICP
      use param, only: CPB, CPC, CPD
      use omm_block, only : blockscale_dihe
      use memory

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(OpenMM_PeriodicTorsionForce) :: torsion
      integer*4 :: ii, iatom, jatom, katom, latom, ic, ijunk, iper
      integer :: index_size
      real*8 :: k, phase

      if(NPHI<=0) return
      call OpenMM_PeriodicTorsionForce_create(torsion)
      call OpenMM_Force_setForceGroup(   &
           transfer(torsion, OpenMM_Force(0)),group)
      indexTorsion = OpenMM_System_addForce(system, &
                  transfer(torsion, OpenMM_Force(0)))

      do ii = 1, NPHI
         iatom = IP(ii)
         jatom = JP(ii)
         katom = abs(KP(ii))
         latom = abs(LP(ii))
         ic = ICP(ii)
         if (ic == 0) cycle
         do
            iper = CPD(ic)
            if (zero_dihedral) then
               k = ZERO
            else
               k = blockscale_dihe( iatom, jatom, katom, latom, &
                    CPD(ic), OpenMM_KJPerKcal * CPC(ic) )
            endif
            ijunk = OpenMM_PeriodicTorsionForce_addTorsion(torsion, &
                  iatom-1, jatom-1, katom-1, latom-1, &
                  abs(iper), CPB(ic), k)
            if (iper >= 0) exit
            ic = ic + 1 ! next part of compound period
         enddo
      enddo
      if(allocated(omm_index_torsion_k)) then
         index_size = size(omm_index_torsion_k)
         call chmdealloc('setup_torsions','omm_bonded', &
              'omm_index_torsion', index_size, crl=omm_index_torsion_k)
      endif

      index_size = OpenMM_PeriodicTorsionForce_getNumTorsions(torsion)
      call chmalloc('setup_torsions','omm_bonded', &
           'omm_index_torsion', index_size, crl=omm_index_torsion_k)

      do ii = 0, index_size-1
         call OpenMM_PeriodicTorsionForce_getTorsionParameters(torsion, &
              ii, iatom, jatom, katom, latom, &
              ic, phase, k)
         omm_index_torsion_k(ii+1) = k
      enddo

   end subroutine setup_torsions

   ! Add dihedral angles interactions to the system.
   subroutine reparameterize_torsions(system, context, lambda)
      use psf, only: NPHI, IP, JP, KP, LP
      use code, only: ICP
      use param, only: CPB, CPC, CPD
      use omm_block, only : blockscale_dihe

      type (OpenMM_System), intent(inout) :: system
      type (OpenMM_Context), intent(inout) :: context
      real(chm_real), intent(in) :: lambda

      type (OpenMM_Force) force
      type (OpenMM_PeriodicTorsionForce) :: torsion
      integer*4 :: ii, iatom, jatom, katom, latom, ic, ijunk, iper
      real*8 :: k, phase

      call OpenMM_System_getForce(system, indexTorsion, force)
      torsion = transfer(force,OpenMM_PeriodicTorsionForce(0))
      iper = OpenMM_PeriodicTorsionForce_getNumTorsions(torsion)
      do ii = 0, iper-1
         call OpenMM_PeriodicTorsionForce_getTorsionParameters(torsion, &
              ii, iatom, jatom, katom, latom, &
              ic, phase, k)
         k = omm_index_torsion_k(ii+1) * lambda
         call OpenMM_PeriodicTorsionForce_setTorsionParameters(torsion, &
              ii, iatom, jatom, katom, latom, &
              ic, phase, k)
      enddo
      call OpenMM_PeriodicTorsionForce_updateParametersInContext(torsion, &
           context)
    end subroutine reparameterize_torsions

   subroutine setup_impropers(system, group)
      use omm_util
      use psf, only: NIMPHI, IM, JM, KM, LM
      use code, only: ICI
      use param, only: CIB, CIC, CPD
      use omm_block, only : blockscale_dihe

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(OpenMM_CustomTorsionForce) :: improper
      type(OpenMM_DoubleArray) :: params
      integer :: ii, iatom, jatom, katom, latom, ic, ijunk
      real(chm_real) :: p1, p2

      if(NIMPHI<=0) return
      ! theta0 typically 0 or pi; theta in [-pi, pi]
      ! wrap is 2*pi if diff <= -pi, -2*pi if diff >= pi, else 0
      call OpenMM_CustomTorsionForce_create(improper, &
            'k*(diff+wrap)^2; wrap=2*pi*(step(-diff-pi)-step(diff-pi)); diff=theta-theta0; pi=3.141592653589793')
      ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(improper, 'k');
      ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(improper, 'theta0');
      call OpenMM_Force_setForceGroup(   &
           transfer(improper, OpenMM_Force(0)),group)

      ijunk = OpenMM_System_addForce(system, transfer(improper, OpenMM_Force(0)))

      call OpenMM_DoubleArray_create(params, 2)
      do ii = 1, NIMPHI
         iatom = IM(ii)
         jatom = JM(ii)
         katom = abs(KM(ii))
         latom = abs(LM(ii))
         ic = ICI(ii)
         if (ic == 0) cycle
         if (zero_improper) then
            p1 = ZERO
         else
            p1 = blockscale_dihe( iatom, jatom, katom, latom, &
                 CPD(ic), OpenMM_KJPerKcal * CIC(ic) )
         endif
         p2 = CIB(ic)
         call omm_param_set(params, [p1, p2])
         ijunk = OpenMM_CustomTorsionForce_addTorsion(improper, &
               iatom-1, jatom-1, katom-1, latom-1, params)
      enddo
      call OpenMM_DoubleArray_destroy(params)

   end subroutine setup_impropers

   subroutine setup_cmaps(system, group)
      use psf  ! I1CT..L2CT etc
      use code, only: ICCT
      use cmapm
      use omm_block, only : checkblock_cmap

      type(OpenMM_System), intent(inout) :: system
      integer*4, intent(in) :: group
      type(OpenMM_CMAPTorsionForce) :: cmaps
      type(OpenMM_DoubleArray) :: omm_map
      integer :: iat1, jat1, kat1, lat1, iat2, jat2, kat2, lat2
      integer :: imap, iterm, ijunk, msize

      if(NCTP<=0) return
      if(NCTP>0) call checkblock_cmap
      call OpenMM_CMAPTorsionForce_create(cmaps)
      call OpenMM_Force_setForceGroup(   &
           transfer(cmaps, OpenMM_Force(0)),group)
      ijunk = OpenMM_System_addForce(system, transfer(cmaps, OpenMM_Force(0)))

      do imap = 1, NCTP
         msize = MCTP(imap)%grid(1)%len1
         call OpenMM_DoubleArray_create(omm_map, msize**2)
         call transform_cmap(omm_map, msize, MCTP(imap)%grid(1)%a)
         ijunk = OpenMM_CMAPTorsionForce_addMap(cmaps, msize, omm_map)
         call OpenMM_DoubleArray_destroy(omm_map)
      enddo

      do iterm = 1, NCRTERM
         imap = ICCT(iterm)
         iat1 = I1CT(iterm)
         jat1 = J1CT(iterm)
         kat1 = K1CT(iterm)
         lat1 = L1CT(iterm)
         iat2 = I2CT(iterm)
         jat2 = J2CT(iterm)
         kat2 = K2CT(iterm)
         lat2 = L2CT(iterm)
         ijunk = OpenMM_CMAPTorsionForce_addTorsion(cmaps, &
               imap-1, iat1-1, jat1-1, kat1-1, lat1-1, iat2-1, jat2-1, kat2-1, lat2-1)
      enddo

   end subroutine setup_cmaps

   subroutine transform_cmap(omm_map, msize, charmm_map)
      use omm_util

      type(OpenMM_DoubleArray), intent(inout) :: omm_map
      integer, intent(in) :: msize
      real(chm_real), intent(in) :: charmm_map(:,:)
      real(chm_real) :: map2d(msize, msize)
      real(chm_real) :: mdata(msize**2)

      ! Converts -180..+179 in 2D to 0..359 in 1D
      if (zero_cmap) then
         mdata = ZERO
      else
         map2d = transpose(charmm_map)
         map2d = cshift(map2d, msize/2, 1)
         map2d = cshift(map2d, msize/2, 2)
         mdata = reshape(map2d, [msize**2])
         mdata = OpenMM_KJPerKcal * mdata
      endif
      call omm_param_set(omm_map, mdata)
   end subroutine transform_cmap

#endif 
end module omm_bonded

