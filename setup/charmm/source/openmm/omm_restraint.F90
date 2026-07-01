module omm_restraint
#if KEY_OPENMM==1
   use chm_kinds
   use number
   use stream, only: OUTU, PRNLEV
   use OpenMM

   implicit none

   private

   logical :: zero_consharm, using_box_params
   logical :: zero_pcharm ! principal component harmonic restraints
   logical :: zero_resd, zero_consdihe, zero_mmfpgeo
   character(len=2), parameter :: box_param(3) = ['bx', 'by', 'bz']

   public :: setup_restraints, update_restraint_box

contains

   ! Set-up any restraint terms available through OpenMM interface
   ! At present this is only harmonic restraints
  subroutine setup_restraints(system, periodic)
    type(OpenMM_System), intent(inout) :: system
    logical, intent(in) :: periodic
    
    ! Setup any harmonic restraints applied
    call mask_eterms()

    call setup_harm_restraints(system, periodic)
    call setup_pcharm_restraints(system, periodic)
    call setup_resd_restraints(system)
    call setup_cons_dihe(system)
    call setup_mmfpgeo_restraints(system)
  end subroutine setup_restraints

   !> Updates this Force's global parameters if in use.
   subroutine update_restraint_box(context, box)
      type(OpenMM_Context), intent(inout) :: context
      real*8, intent(in) :: box(3)
      integer :: i

      if (using_box_params) then
         do i = 1, 3
            call OpenMM_Context_setParameter(context, &
                  box_param(i), abs(box(i)))
         enddo
      endif
   end subroutine update_restraint_box

   subroutine mask_eterms()
      use energym
 
      zero_consharm = .not. QETERM(CHARM)
      zero_resd = .not. QETERM(RESD)
      zero_consdihe = .not. QETERM(CDIHE)
      zero_mmfpgeo = .not. QETERM(GEO)
      zero_pcharm = .not. qeterm(PCHARM)
   end subroutine mask_eterms

   subroutine setup_harm_restraints(system, periodic)
      use cnst_fcm
      use psf, only: NATOM
      use omm_ecomp, only : omm_incr_eterms

      type(OpenMM_System), intent(inout) :: system
      logical, intent(in) :: periodic
      character(len=*), parameter :: procname = ' OpenMM.setup_harm_restraints'
      logical :: restraint_ok(NUMHSETS), atom_restrained(NATOM)
      integer :: iset
      integer*4 :: group
      real(chm_real) :: hscale(3)

      using_box_params = .false.
      if (.not. QCNSTR) return
      restraint_ok = (TYPHSET == 0) .and. (KCEXPN > 0)
      if (any(restraint_ok)) then
         group = omm_incr_eterms('charm')    
         do iset = 1, NUMHSETS
            hscale = [XHSCALE(iset), YHSCALE(iset), ZHSCALE(iset)]
            if (restraint_ok(iset) .and. all(hscale == ONE)) then
               ! Make sure there are atoms of this type with non-zero force constants
               atom_restrained = (IHSET == iset) .and. (KCNSTR /= ZERO)
               if (any(atom_restrained)) then
                  if (PRNLEV > 2) then 
                     write (OUTU, '(x,2a,i4)') &
                       procname, ": Setting up atom restraints for set ", iset
                     write (OUTU, '(x,2a,i4)') &
                       procname, ": Using exponent of ", KCEXPN(iset)
                     if(periodic) write(OUTU,'(x,2a)') &
                          procname, ": Using periodic restraints"
                  endif
                  call add_harm_restraint_force(system, group, periodic, &
                       atom_restrained, &
                       KCNSTR, REFX, REFY, REFZ, KCEXPN(iset))
               endif
            else if (typhset(iset)/=3) then
               if (PRNLEV > 2) write (OUTU, '(x,2a,i2,a,/,10x,a)') &
                    procname, ": Restraint set ", iset, "w/ x(y,z)scale /= 1 or exponent <= 0 ", &
                    "not supported through OpenMM, no restraint of this type applied"
            endif
         enddo
      else if (any((typhset == 3) .or. (typhset == 0))) then
         return
      else
         if (PRNLEV > 2) write (OUTU, '(x,2a,/,10x,a)') &
              procname, ": No restraints applied, bestfit/relative or exponent <= 0", &
              "not supported through OpenMM"
      endif
   end subroutine setup_harm_restraints

   subroutine add_harm_restraint_force(system, group, periodic,  &
        atom_restrained, kcnstr, refx, refy, refz, nexp)
      use omm_util

      type(OpenMM_System), intent(inout) :: system
      logical, intent(in) :: periodic
      integer*4, intent(in) :: group
      logical, intent(in) :: atom_restrained(:)
      real(chm_real), intent(in) :: kcnstr(:), refx(:), refy(:), refz(:)
      integer, intent(in) :: nexp

      type(OpenMM_CustomExternalForce) :: consharm
      type(OpenMM_DoubleArray) :: params
      character(len=200) :: formula
      integer, parameter :: NP = 4
      character(len=2), parameter :: pname(NP) = ['k ', 'x0', 'y0', 'z0']
      real*8 :: box_a(3), box_b(3), box_c(3), box(3)
      real(chm_real) :: pval(NP), k_conversion
      integer :: i, ijunk, natom

      if (periodic) then
         write (formula, '(3(a,i0),3a)') &
               'k * (px^', nexp, &
                  ' + py^', nexp, &
                  ' + pz^', nexp, '); ', &
               'px=min(dx, bx-dx); py=min(dy, by-dy); pz=min(dz, bz-dz); ', &
               'dx=abs(x-x0); dy=abs(y-y0); dz=abs(z-z0)'
      else
         write (formula, '(3(a,i0),2a)') &
               'k * (dx^', nexp, &
                  ' + dy^', nexp, &
                  ' + dz^', nexp, '); ', &
               'dx=abs(x-x0); dy=abs(y-y0); dz=abs(z-z0)'
      endif

      call OpenMM_CustomExternalForce_create(consharm, formula)
      call OpenMM_Force_setForceGroup(   &
           transfer(consharm, OpenMM_Force(0)),group)
      do i = 1, NP
         ijunk = OpenMM_CustomExternalForce_addPerParticleParameter( &
               consharm, pname(i))
      enddo

      if (periodic) then
         ! assumes rectangular box aligned with coordinate axes
         call OpenMM_System_getDefaultPeriodicBoxVectors(system, &
               box_a, box_b, box_c)
         box = [box_a(1), box_b(2), box_c(3)]
         ! each Force must add its own references to global Context parameters
         do i = 1, 3
            ijunk = OpenMM_CustomExternalForce_addGlobalParameter( &
                  consharm, box_param(i), abs(box(i)))
         enddo
         using_box_params = .true.
      endif

      ijunk = OpenMM_System_addForce(system, transfer(consharm, OpenMM_Force(0)))

      if (zero_consharm) then
         k_conversion = ZERO
      else
         k_conversion = OpenMM_KJPerKcal * (OpenMM_AngstromsPerNm ** nexp)
      endif
      natom = size(atom_restrained)
      call OpenMM_DoubleArray_create(params, NP)
      do i = 1, natom
         if (atom_restrained(i)) then
            pval(1) = kcnstr(i) * k_conversion
            pval(2:4) = [refx(i), refy(i), refz(i)] / OpenMM_AngstromsPerNm
            call omm_param_set(params, pval)
            ijunk = OpenMM_CustomExternalForce_addParticle( &
                  consharm, i-1, params)
         endif
      enddo
      call OpenMM_DoubleArray_destroy(params)

    end subroutine add_harm_restraint_force
    
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!    
    subroutine setup_pcharm_restraints(system, periodic)
      use cnst_fcm
      use psf, only: NATOM
      use omm_ecomp, only : omm_incr_eterms

      type(OpenMM_System), intent(inout) :: system
      logical, intent(in) :: periodic
      character(len=*), parameter :: procname = ' OpenMM.setup_pcharm_restraints'
      logical :: restraint_ok(NUMHSETS), atom_restrained(NATOM)
      integer :: iset, ipca
      integer*4 :: group
      real(chm_real) :: hscale(3)

      using_box_params = .false.
      if (.not. QCNSTR) return
      restraint_ok = (TYPHSET == 3) .and. (KCEXPN > 0)
      if (any(restraint_ok)) then
         group = omm_incr_eterms('pcharm')    
         do iset = 1, NUMHSETS
            hscale = [XHSCALE(iset), YHSCALE(iset), ZHSCALE(iset)]
            if (restraint_ok(iset) .and. all(hscale == ONE)) then
               ! Make sure there are atoms of this type with non-zero force constants
               atom_restrained = (IHSET == iset) .and. (KCNSTR /= ZERO)
               if (any(atom_restrained)) then
                  if (PRNLEV > 2) then 
                     write (OUTU, '(x,2a,i4)') &
                          procname, ": Setting up principal component harmonic restraints for set ", iset
                     write (OUTU, '(x,2a,i4)') &
                          procname, ": Using exponent of ", KCEXPN(iset)
                     if(periodic) write(OUTU,'(x,2a)') &
                          procname, ": Using periodic restraints"
                  endif
                  do ipca = 1, numpca
                     call add_pcharm_restraint_force(system, group, periodic, &
                          atom_restrained, &
                          KCNSTR, REFX, REFY, REFZ, &
                          pcax(ipca,:), &
                          pcay(ipca,:), &
                          pcaz(ipca,:), &
                          KCEXPN(iset))
                  enddo
               endif
            else if (typhset(iset)/=0) then
               if (PRNLEV > 2) write (OUTU, '(x,2a,i2,a,/,10x,a)') &
                    procname, ": Restraint set ", iset, "w/ x(y,z)scale /= 1 or exponent <= 0 ", &
                    "not supported through OpenMM, no restraint of this type applied"
            endif
         enddo
      else if (any((typhset == 3) .or. (typhset == 0))) then
         return
      else
         if (PRNLEV > 2) write (OUTU, '(x,2a,/,10x,a)') &
              procname, ": No restraints applied, bestfit/relative or exponent <= 0", &
              "not supported through OpenMM"
      endif
    end subroutine setup_pcharm_restraints

   subroutine add_pcharm_restraint_force(system, group, periodic,  &
        atom_restrained, kcnstr, refx, refy, refz, &
        pcax, pcay, pcaz, &
        nexp)
      use omm_util

      type(OpenMM_System), intent(inout) :: system
      logical, intent(in) :: periodic
      integer*4, intent(in) :: group
      logical, intent(in) :: atom_restrained(:)
      real(chm_real), intent(in) :: kcnstr(:), refx(:), refy(:), refz(:), &
           pcax(:), pcay(:), pcaz(:)
      integer, intent(in) :: nexp

      type(OpenMM_CustomExternalForce) :: pcaharm
      type(OpenMM_DoubleArray) :: params
      character(len=200) :: formula
      integer, parameter :: NP = 7
      character(len=3), parameter :: pname(NP) = ['k  ', 'x0 ', 'y0 ', 'z0 ', 'pcx', 'pcy', 'pcz']
      real*8 :: box_a(3), box_b(3), box_c(3), box(3)
      real(chm_real) :: pval(NP), k_conversion
      integer :: i, ijunk, natom

      if (periodic) then
         write (formula, '(3(a,i0),3a)') &
               'k * (pmx^', nexp, &
                  ' + pmy^', nexp, &
                  ' + pmz^', nexp, '); ', &
               'pmx=abs(px*pcx); pmy=abs(py*pcy); pmz=abs(pz*pcz); ', &
               'px=min(dx, bx-dx); py=min(dy, by-dy); pz=min(dz, bz-dz); ', &
               'dx=(x-x0); dy=(y-y0); dz=(z-z0)'
      else
         write (formula, '(3(a,i0),2a)') &
               'k * (dx^', nexp, &
                  ' + dy^', nexp, &
                  ' + dz^', nexp, '); ', &
               'dx=abs((x-x0)*pcx); dy=abs((y-y0)*pcy); dz=abs((z-z0)*pcz)'
      endif

      call OpenMM_CustomExternalForce_create(pcaharm, formula)
      call OpenMM_Force_setForceGroup(   &
           transfer(pcaharm, OpenMM_Force(0)),group)
      do i = 1, NP
         ijunk = OpenMM_CustomExternalForce_addPerParticleParameter( &
               pcaharm, pname(i))
      enddo

      if (periodic) then
         ! assumes rectangular box aligned with coordinate axes
         call OpenMM_System_getDefaultPeriodicBoxVectors(system, &
               box_a, box_b, box_c)
         box = [box_a(1), box_b(2), box_c(3)]
         ! each Force must add its own references to global Context parameters
         do i = 1, 3
            ijunk = OpenMM_CustomExternalForce_addGlobalParameter( &
                  pcaharm, box_param(i), abs(box(i)))
         enddo
         using_box_params = .true.
      endif

      ijunk = OpenMM_System_addForce(system, transfer(pcaharm, OpenMM_Force(0)))

      if (zero_pcharm) then
         k_conversion = ZERO
      else
         k_conversion = OpenMM_KJPerKcal * (OpenMM_AngstromsPerNm ** nexp)
      endif
      natom = size(atom_restrained)
      call OpenMM_DoubleArray_create(params, NP)
      do i = 1, natom
         if (atom_restrained(i)) then
            pval(1) = kcnstr(i) * k_conversion
            pval(2:4) = [refx(i), refy(i), refz(i)] / OpenMM_AngstromsPerNm
            pval(5:7) = [pcax(i), pcay(i), pcaz(i)] 
            call omm_param_set(params, pval)
            ijunk = OpenMM_CustomExternalForce_addParticle( &
                  pcaharm, i-1, params)
         endif
      enddo
      call OpenMM_DoubleArray_destroy(params)

    end subroutine add_pcharm_restraint_force
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
   ! This routine sets up CHARMM restrained distance restraints for
   ! use through the OpenMM interface.
   subroutine setup_mmfpgeo_restraints(system)
     use mmfp, only : ntgeo, qgeo
     use omm_ecomp, only : omm_incr_eterms
     type(OpenMM_System), intent(inout) :: system
     integer :: i, nunique
     integer :: mmfpgeo_type(ntgeo)
     integer*4 :: group

     logical :: is_new_formula

     if (zero_mmfpgeo .or. .not. qgeo) return
     if (ntgeo > 0) group = omm_incr_eterms('geo')    
     

     ! Find the number of unique restraints
     nunique = 0
     mmfpgeo_type(1:ntgeo) = 1
     do i=1, ntgeo
        is_new_formula = newformula_mmfpgeo(i, mmfpgeo_type)
        if (is_new_formula) then
           nunique = nunique + 1
           mmfpgeo_type(i) = nunique
        end if
     end do

     call alloc_mmfpgeo_restraints(system, group, nunique, mmfpgeo_type)

   end subroutine setup_mmfpgeo_restraints

   subroutine alloc_mmfpgeo_restraints(system, group, nunique, mmfpgeo_type)    
     use mmfp, only : ntgeo, ngeo, igeo, jgeo, qprnt
     type(OpenMM_System), intent(inout) :: system
     integer*4, intent(in) :: group
     type(OpenMM_CustomCompoundBondForce) :: OMM_mmfpRestraintForce(1:nunique)
     integer, intent(in) :: nunique, mmfpgeo_type(ntgeo)
     character(len=2048) :: restraint_formula

     integer :: n, ndx 

     logical :: is_new_formula
      
     loop_alloc_restr: do n=1, ntgeo
        is_new_formula = newformula_mmfpgeo(n, mmfpgeo_type)

       if (is_new_formula) then
          if (igeo(n) == 0 ) cycle loop_alloc_restr

          call getmmfp_formula(n, restraint_formula, mmfpgeo_type(n))

          call add_mmfpgeo_restraint(system, group, &
               OMM_mmfpRestraintForce(mmfpgeo_type(n)), &
               restraint_formula, &
               nunique, n, &
               mmfpgeo_type(n))
       endif
       if (igeo(n) >= 1 ) then
           call add_mmfpgeo_addbond(system, &
                OMM_mmfpRestraintForce(mmfpgeo_type(n)), &
                ngeo(n+1)-ngeo(n), &
                nunique, n)
      elseif ( igeo(n) == -1 .or. igeo(n) == -2 .or. igeo(n) == -3 ) then
           call add_mmfpgeo_addbond(system, &
                OMM_mmfpRestraintForce(mmfpgeo_type(n)), &
                ngeo(n+2)-ngeo(n), &
                nunique, n)

      endif

     enddo loop_alloc_restr
     if(QPRNT) then
          do ndx = 1, ntgeo
             write(6,'(1X,2(A,I2))') 'Printing &
             RESTRAINT ', ndx, ' of ', ntgeo

             is_new_formula = newformula_mmfpgeo(ndx, mmfpgeo_type)
             if (is_new_formula) call print_mmfpgeo_restraints(system, &
                  OMM_mmfpRestraintForce(mmfpgeo_type(ndx)), ndx)
          enddo
     endif
   end subroutine alloc_mmfpgeo_restraints

   character(len=64) function setglobalname(name,i)
     character(len=32), intent(in) :: name
     character(len=64) :: globalname
     integer, intent(in) :: i
     write(globalname,'(a,a,i2)') name,'_',i
     setglobalname = trimblanks64(globalname)
   end function setglobalname

   subroutine getmmfp_formula(n, formula, itype)
     use mmfp, only : ngeo, igeo, jgeo, lstgeo
     integer, intent(in) :: n, itype
     integer :: ipt, ioffset
     character(len=2048), intent(inout) :: formula
     character(len=1), parameter :: mu(10) = ['a','b','c','d','e','f','g','h','i','j']
     character(len=64) :: xcm, ycm, zcm, xcm2, ycm2, zcm2
     character(len=1024) :: bx, by, bz, bx2, by2, bz2
     character(len=64) :: xdir, ydir, zdir, xref, yref, zref, droff, frc, pass
     pass = 'xdir'
     xdir = setglobalname(pass,itype)
     pass = 'ydir'
     ydir = setglobalname(pass,itype)
     pass = 'zdir'
     zdir = setglobalname(pass,itype)
     pass = 'xref'
     xref = setglobalname(pass,itype)
     pass = 'yref'
     yref = setglobalname(pass,itype)
     pass = 'zref'
     zref = setglobalname(pass,itype)
     pass = 'droff'
     droff = setglobalname(pass,itype)
     pass = 'frc'
     frc = setglobalname(pass,itype)
     bx = ''
     by = ''
     bz = ''
     xcm = ''
     ycm = ''
     zcm = ''
     bx2 = ''
     by2 = ''
     bz2 = ''
     xcm2 = ''
     ycm2 = ''
     zcm2 = ''
     formula = ''
     
     if ( igeo(n) >= 1 ) then
        do ipt = 1, ngeo(n+1)-ngeo(n)
           if (ipt == 1) then
              write(xcm,'(a,i2,a,i2)') 'm',ipt,'*x',ipt
              write(ycm,'(a,i2,a,i2)') 'm',ipt,'*y',ipt
              write(zcm,'(a,i2,a,i2)') 'm',ipt,'*z',ipt
           else
              write(xcm,'(a,i2,a,i2)') '+ m',ipt,'*x',ipt
              write(ycm,'(a,i2,a,i2)') '+ m',ipt,'*y',ipt
              write(zcm,'(a,i2,a,i2)') '+ m',ipt,'*z',ipt
           endif
           bx = trim(bx) // trim(xcm)
           by = trim(by) // trim(ycm)
           bz = trim(bz) // trim(zcm)
           xcm = ''
           ycm = ''
           zcm = ''
        enddo

        bx = 'delx = ' // trim(bx) // '- ' // trim(xref)
        by = 'dely = ' // trim(by) // '- ' // trim(yref)
        bz = 'delz = ' // trim(bz) // '- ' // trim(zref)
    endif

    if ( igeo(n) < 0 ) then
       do ipt = 1, ngeo(n+1)-ngeo(n)
           if (ipt == 1) then
              write(xcm,'(a,i2,a,i2)') 'm',ipt,'*x',ipt
              write(ycm,'(a,i2,a,i2)') 'm',ipt,'*y',ipt
              write(zcm,'(a,i2,a,i2)') 'm',ipt,'*z',ipt
           else
              write(xcm,'(a,i2,a,i2)') '+ m',ipt,'*x',ipt
              write(ycm,'(a,i2,a,i2)') '+ m',ipt,'*y',ipt
              write(zcm,'(a,i2,a,i2)') '+ m',ipt,'*z',ipt
           endif
           bx = trim(bx) // trim(xcm)
           by = trim(by) // trim(ycm)
           bz = trim(bz) // trim(zcm)
           xcm = ''
           ycm = ''
           zcm = ''
       enddo

       ioffset = ngeo(n+1)-ngeo(n)
       do ipt = 1, ngeo(n+2) - ngeo(n+1)
          if (ipt == 1) then
             write(xcm2,'(a,i2,a,i2)') &
             '(m',ipt+ioffset,'*x',ipt+ioffset
             write(ycm2,'(a,i2,a,i2)') &
             '(m',ipt+ioffset,'*y',ipt+ioffset
             write(zcm2,'(a,i2,a,i2)') &
             '(m',ipt+ioffset,'*z',ipt+ioffset
          else
             write(xcm2,'(a,i2,a,i2)') &
             '+ m',ipt+ioffset,'*x',ipt+ioffset
             write(ycm2,'(a,i2,a,i2)') &
             '+ m',ipt+ioffset,'*y',ipt+ioffset
             write(zcm2,'(a,i2,a,i2)') &
             '+ m',ipt+ioffset,'*z',ipt+ioffset
          endif
          bx2 = trim(bx2) // trim(xcm2)
          by2 = trim(by2) // trim(ycm2)
          bz2 = trim(bz2) // trim(zcm2)
          xcm2 = ''
          ycm2 = ''
          zcm2 = ''
       enddo
       bx = 'delx = ' // trim(bx) // '- ' // trim(bx2) // ')'
       by = 'dely = ' // trim(by) // '- ' // trim(by2) // ')'
       bz = 'delz = ' // trim(bz) // '- ' // trim(bz2) // ')'


   endif

   formula = trim(bz) // ';' // trim(by) // ';' // trim(bx) // ';'

   ! Geometry = sphere
   if(abs(igeo(n)) == 1) then
      formula = 'r=sqrt(q+small);q=delx*delx+dely*dely+delz*delz;' &
           // trim(formula)
   ! Geometry = cylinder
   elseif(abs(igeo(n))==2) then
      formula = 'r=sqrt(s+small);s=delax*delax+delay*delay+delaz*delaz;' // 'delaz=delz-q*' &
           // trim(zdir) // '; delay=dely-q*' // trim(ydir) // '; delax=delx-q*' // trim(xdir) // ';' &
           // 'q=delx*' // trim(xdir) // '+dely*' // trim(ydir) // '+delz*' // trim(zdir) // ';' &
           // trim(formula)
   ! Geometry = plane
   elseif(abs(igeo(n))==3 .and. jgeo(n) == 3) then
      formula = 'r=delx*' // trim(xdir) // '+dely*' // trim(ydir) // '+delz*' // trim(zdir) // ';' &
           // trim(formula)
   elseif(abs(igeo(n))==3 .and. jgeo(n) /= 3) then
      formula = 'r=abs(q);q=delx*' // trim(xdir) // '+dely*' // trim(ydir) // '+delz*' // trim(zdir) // ';' &
           // trim(formula)
   else
      call wrndie(-1,'<getmmfp_formula>', &
           'Unsupported GEO restraint geometry/shape in CHARMM/OpenMM')
   endif


   formula = 'dlta=r-' // trim(droff) // ';' // trim(formula)
   if(jgeo(n) == 1) then
      formula = 'step(dlta)*0.5*' // trim(frc) // '*dlta*dlta;' // trim(formula)
   elseif(jgeo(n) == 2) then 
      formula = '(1-step(dlta))*0.5*' // trim(frc) // '*dlta*dlta;' // trim(formula)
   elseif(jgeo(n) == 3) then 
      formula = '0.5*' // trim(frc) // '*dlta*dlta;' // trim(formula)
   else
      call wrndie(-1,'<getmmfp_formula>', &
           'Unsupported GEO restraint potential specification in CHARMM/OpenMM')
   endif

      formula = trimblanks(formula)
   end subroutine getmmfp_formula

   subroutine add_mmfpgeo_restraint(system, group, OMM_mmfpRestraintForce, &
        formula, nunique, n, resd_type)
     use mmfp, only : ngeo, igeo, jgeo, lstgeo, xrgeo, yrgeo, zrgeo, &
          xdgeo, ydgeo, zdgeo, drgeo, fcgeo
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_CustomCompoundBondForce), intent(inout) :: OMM_mmfpRestraintForce
     integer, intent(in) :: n, nunique, resd_type
     character(len=2048), intent(in) :: formula
     integer*4, intent(in) :: group
     character(len=1024) :: forceCnst 
     integer :: i, iparam, nparticles
     character(len=1), parameter :: mu(10) = ['a','b','c','d','e','f','g','h','i','j']
     real*8 :: norm, xdir, ydir, zdir
     real*8, parameter :: small = 1.0e-6
     character(len=64) :: axdir, aydir, azdir, axref, ayref, azref, adroff, afrc, pass 
     pass = 'xdir'
     axdir = setglobalname(pass,resd_type)
     pass = 'ydir'
     aydir = setglobalname(pass,resd_type)
     pass = 'zdir'
     azdir = setglobalname(pass,resd_type)
     pass = 'xref'
     axref = setglobalname(pass,resd_type)
     pass = 'yref'
     ayref = setglobalname(pass,resd_type)
     pass = 'zref'
     azref = setglobalname(pass,resd_type)
     pass = 'droff'
     adroff = setglobalname(pass,resd_type)
     pass = 'frc'
     afrc = setglobalname(pass,resd_type)

     nparticles = (ngeo(n+1)-ngeo(n))
     if (igeo(n) < 0 ) nparticles = nparticles + ngeo(n+2) - ngeo(n+1)

     ! Normalize xdir, ydir, zdir
     xdir = xdgeo(n)
     ydir = ydgeo(n)
     zdir = zdgeo(n)
     norm = xdgeo(n) * xdgeo(n) + ydgeo(n) * ydgeo(n) + zdgeo(n) * zdgeo(n) 
     if(norm > 0) then
        norm = sqrt(norm)
        xdir = xdgeo(n) / norm
        ydir = ydgeo(n) / norm
        zdir = zdgeo(n) / norm
     endif

     call OpenMM_CustomCompoundBondForce_create(OMM_mmfpRestraintForce, &
          nparticles,trim(formula))
      call OpenMM_Force_setForceGroup(   &
           transfer(OMM_mmfpRestraintForce, OpenMM_Force(0)),group)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,afrc, &
          fcgeo(n)*OpenMM_KJPerKcal*OpenMM_AngstromsPerNm*OpenMM_AngstromsPerNm)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(adroff),drgeo(n)/OpenMM_AngstromsPerNm)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(axref), xrgeo(n)/OpenMM_AngstromsPerNm)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(ayref), yrgeo(n)/OpenMM_AngstromsPerNm)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(azref), zrgeo(n)/OpenMM_AngstromsPerNm)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(axdir), xdir)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(aydir), ydir)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,trim(azdir), zdir)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_mmfpRestraintForce,'small', small/OpenMM_AngstromsPerNm)

     do i=1, ngeo(n+1)-ngeo(n)
        pass = ''
        write(pass,'(a,i2)') 'm',i
        pass = trimblanks64(pass)
        iparam = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
          OMM_mmfpRestraintForce,trim(pass))
     enddo
     
     if ( igeo(n) < 0 ) then
         do i=1, ngeo(n+2)-ngeo(n+1)
            pass = ''
            write(pass,'(a,i2)') 'm',i + ngeo(n+1)-ngeo(n)
            pass = trimblanks64(pass)
            iparam = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
              OMM_mmfpRestraintForce,trim(pass))
         enddo
     endif
     iparam = OpenMM_System_addForce(system, &
          transfer(OMM_mmfpRestraintForce, OpenMM_Force(0)))

   end subroutine add_mmfpgeo_restraint

   subroutine add_mmfpgeo_addbond(system, OMM_mmfpRestraintForce, &
        nparticles, nunique, n)
     use mmfp, only : ngeo, lstgeo, igeo
     use psf, only : amass
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_CustomCompoundBondForce), intent(inout) :: OMM_mmfpRestraintForce
     type(OpenMM_DoubleArray) :: params
     type(OpenMM_IntArray) :: particles 
     real*8 :: mcm, value, mcm2
     integer, intent(in) :: n, nunique, nparticles
     integer :: i, iparam

     call OpenMM_DoubleArray_create(params, nparticles)
     call OpenMM_IntArray_create(particles, nparticles)

     mcm = zero
     do i = ngeo(n), ngeo(n+1)-1
        mcm = mcm + amass(lstgeo(i))
     enddo
     if (igeo(n) < 0) then
        mcm2 = zero
        do i = ngeo(n+1), ngeo(n+2)-1
            mcm2 = mcm2 + amass(lstgeo(i))
        enddo
     endif

     iparam = 1
     do i = ngeo(n), ngeo(n+1)-1
        value = amass(lstgeo(i))/mcm
        call OpenMM_DoubleArray_set(params,iparam,value)
        call OpenMM_IntArray_set(particles,iparam,lstgeo(i)-1)
        iparam = iparam + 1
     enddo
     if (igeo(n) < 0) then
        do i = ngeo(n+1), ngeo(n+2)-1
            value = amass(lstgeo(i))/mcm2
            call OpenMM_DoubleArray_set(params,iparam,value)
            call OpenMM_IntArray_set(particles,iparam,lstgeo(i)-1)
            iparam = iparam + 1
        enddo
     endif

     iparam = OpenMM_CustomCompoundBondForce_addBond(OMM_mmfpRestraintForce, particles, params)
     call OpenMM_DoubleArray_destroy(params)
     call OpenMM_IntArray_destroy(particles)
   end subroutine add_mmfpgeo_addbond

   subroutine print_mmfpgeo_restraints(system, restraint, n)

     use mmfp, only : igeo
     type(OpenMM_System), intent(in) :: system
     type(OpenMM_CustomCompoundBondForce), intent(in) :: restraint
     integer, intent(in) :: n
     type(OpenMM_DoubleArray) :: params
     type(OpenMM_IntArray) :: particles
     integer :: nbonds, nparticles, nbondparams, nglobalparams
     integer :: iparam, ibond, i
     character(len=2048) :: restr_function
     character*16 :: globalname, bondparamname
     character*32 :: fmat
     real*8 :: pvalue
     integer, allocatable :: iparticles(:)
    
    
     if ( igeo(n) .eq. 0 ) return

     nbonds = OpenMM_CustomCompoundBondForce_getNumBonds(restraint)
     nparticles = OpenMM_CustomCompoundBondForce_getNumParticlesPerBond(restraint)
     allocate (iparticles(nparticles))
     nbondparams = OpenMM_CustomCompoundBondForce_getNumPerBondParameters(restraint)
     nglobalparams = OpenMM_CustomCompoundBondForce_getNumGlobalParameters(restraint)
     call OpenMM_DoubleArray_create(params, nbondparams)
     call OpenMM_IntArray_create(particles, nparticles)

     call OpenMM_CustomCompoundBondForce_getEnergyFunction(restraint,restr_function)
     write(outu,'(1x,a,/,9x,a)')' CHARMM> MMFP GEO Restraint Function:', &
     trim(restr_function)
     do iparam = 0, nglobalparams - 1
        if(iparam == 0) then
           pvalue = one / OpenMM_KJPerKcal/OpenMM_AngstromsPerNm/OpenMM_AngstromsPerNm
        elseif((iparam > 4) .and. (iparam < 8)) then
           pvalue = One
        else
           pvalue = OpenMM_AngstromsPerNm
        endif
        call OpenMM_CustomCompoundBondForce_getGlobalParameterName(restraint,iparam,globalname)
        write(outu,'(a,a,a,g10.4)') ' CHARMM> ',trim(globalname),' = ', &
             OpenMM_CustomCompoundBondForce_getGlobalParameterDefaultValue(restraint,iparam) &
             *pvalue
     enddo
     write(fmat,'(a,i3,a)')'(a,i3,a,',nparticles,'i3,a)'
     do ibond = 0, nbonds-1
        call OpenMM_CustomCompoundBondForce_getBondParameters(restraint, ibond, particles, params)
        do i = 1, nparticles
           Call OpenMM_IntArray_get(particles,i,iparticles(i))
           iparticles(i) = iparticles(i) + 1
        enddo
        
        if (igeo(n) .gt. 0 ) then
            write(outu,fmat)' CHARMM> Interaction between atom &
                 indices for bond ', &
                 ibond,':[',iparticles,']'
        endif

        do iparam = 1, nbondparams
           call OpenMM_CustomCompoundBondForce_getPerBondParameterName(restraint, iparam-1, &
                bondparamname)
           call OpenMM_DoubleArray_get(params,iparam, pvalue)
          write(outu,'(a,a,a,f9.2)') ' CHARMM> ',trim(bondparamname),'=',pvalue
        enddo
     enddo

     deallocate(iparticles)
     call OpenMM_DoubleArray_destroy(params)
     call OpenMM_IntArray_destroy(particles)

   end subroutine print_mmfpgeo_restraints
   
   ! June 2022
   ! AC expanding permitted igeo values from {1,2,3} to U {-1, -2, -3}
   ! to allow GEO DIST
   logical function newformula_mmfpgeo(n, rtype)
     use mmfp, only : ntgeo, ngeo, igeo, jgeo, &
          xrgeo, yrgeo, zrgeo, xdgeo, ydgeo, zdgeo, drgeo, fcgeo
     integer :: n
     integer :: i, natcm, rtype(1:ntgeo)
     integer, parameter :: nmaxcm = 100 !changed from 10
     logical :: newformula
     character(len=1024) :: wrnmsg 
     newformula = .true.

     if ( ( ngeo(n+1)-ngeo(n) > nmaxcm ) .or. &
          ( igeo(n) < -3 .or. igeo(n) > 3 ) ) then
          write(wrnmsg, '(1X,A,I2)') &
          'Number of atoms in restraint too large/geometric shape not supported for n = ',n 
          call wrndie(-1, '<setup_mmfpgeo_restraints>', wrnmsg)
     endif
     natcm = ( ngeo(n+1)-ngeo(n) )
     do i = 1, n-1
        newformula = newformula &
          .and.      ( ( ngeo(i+1)-ngeo(i) /= ngeo(n+1)-ngeo(n) ) &      ! compare natoms
          .or.       ( igeo(i) /= igeo(n) )                       &      ! compare geo type (sphere, cyclinder, plane)
          .or.       ( jgeo(i) /= jgeo(n) )                       &      ! compare sided-ness (inside, outside, symmetric)
          .or.       ( xrgeo(i) /= xrgeo(n) )                     &      ! compare xref
          .or.       ( yrgeo(i) /= yrgeo(n) )                     &      ! compare yref
          .or.       ( zrgeo(i) /= zrgeo(n) )                     &      ! compare zref
          .or.       ( xdgeo(i) /= xdgeo(n) )                     &      ! compare xdir
          .or.       ( ydgeo(i) /= ydgeo(n) )                     &      ! compare ydir
          .or.       ( drgeo(i) /= drgeo(n) )                     &      ! compare zdir
          .or.       ( fcgeo(i) /= fcgeo(n) ) )                          ! compare force constant
        if(.not. newformula) then
           rtype(n) = rtype(i)
           newformula_mmfpgeo = newformula
           return
        endif
     end do
     newformula_mmfpgeo = newformula
   end function newformula_mmfpgeo

  ! This routine sets up CHARMM restrained distance restraints for
   ! use through the OpenMM interface.
   subroutine setup_resd_restraints(system)
     use resdist_ltm, only : rednum
     use omm_ecomp, only : omm_incr_eterms
     type(OpenMM_System), intent(inout) :: system
     integer :: i, nunique
     integer :: resd_type(rednum)
     integer*4 :: group

     logical :: is_new_formula

     if(zero_resd) return
     if(rednum>0) group = omm_incr_eterms('resd')    
     
     ! Find the number of unique restraints
     nunique = 0
     resd_type(1:rednum) = 1
     do i=1, rednum
        is_new_formula = newformula(i,resd_type)
        if (is_new_formula) then
           nunique = nunique + 1
           resd_type(i) = nunique
        endif
     enddo
     
     call alloc_resd_restraints(system, group, nunique, resd_type)
   end subroutine setup_resd_restraints

   subroutine alloc_resd_restraints(system, group, nunique, resd_type)    
     use resdist_ltm, only : rednum, redipt, redeval, redival
     type(OpenMM_System), intent(inout) :: system
     integer*4, intent(in) :: group
     type(OpenMM_CustomCompoundBondForce) :: OMM_ResdRestraintForce(1:nunique)
     integer, intent(in) :: nunique, resd_type(:)
     character(len=1024) :: restraint_formula

     integer :: n

     logical :: is_new_formula
      
     do n=1, rednum
        is_new_formula = newformula(n, resd_type)
        if (is_new_formula) then
           call getccbf_formula(n, restraint_formula)
           call add_resd_restraint(system, group, OMM_ResdRestraintForce(resd_type(n)), &
                restraint_formula, nunique, n, resd_type(n))
        endif
        call add_resd_addbond(system, OMM_ResdRestraintForce(resd_type(n)), &
             redipt(n+1)-redipt(n), 2*(redipt(n+1)-redipt(n)), nunique, n, resd_type(n))
     enddo
     if(prnlev >=7) then
        do n = 1, rednum
           is_new_formula = newformula(n, resd_type)
           if (is_new_formula) call print_resd_restraints(system, &
                OMM_ResdRestraintForce(resd_type(n)),resd_type(n),redeval(n), redival(n))
        enddo
     endif
   end subroutine alloc_resd_restraints

   subroutine add_resd_restraint(system, group, OMM_ResdRestraintForce, &
        formula, nunique, n, resd_type)
     use resdist_ltm, only : rednum, redipt, redsca
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_CustomCompoundBondForce), intent(inout) :: OMM_ResdRestraintForce
     integer, intent(in) :: n, nunique, resd_type
     character(len=1024), intent(in) :: formula
     integer*4, intent(in) :: group
     character(len=1024) :: forceCnst 
     integer :: i, iparam, nparticles
     nparticles = 2 * (redipt(n+1)-redipt(n))
     call OpenMM_CustomCompoundBondForce_create(OMM_ResdRestraintForce, &
          nparticles,trim(formula))
      call OpenMM_Force_setForceGroup(   &
           transfer(OMM_ResdRestraintForce, OpenMM_Force(0)),group)
     iparam = OpenMM_CustomCompoundBondForce_addGlobalParameter( &
          OMM_ResdRestraintForce,'sval',redsca)
     iparam = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
          OMM_ResdRestraintForce,'kval')
     iparam = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
          OMM_ResdRestraintForce,'rval')

     do i=1, redipt(n+1)-redipt(n)
        write(forceCnst,'(a,i5)')'k',i
        forceCnst = trimblanks(forceCnst)
        iparam = OpenMM_CustomCompoundBondForce_addPerBondParameter( &
          OMM_ResdRestraintForce,trim(forceCnst))
        forceCnst = ''
     enddo

     iparam = OpenMM_System_addForce(system, &
          transfer(OMM_ResdRestraintForce, OpenMM_Force(0)))

   end subroutine add_resd_restraint

   subroutine add_resd_addbond(system, OMM_ResdRestraintForce, npairs, &
        nparticles, nunique, n, resd_type)
     use resdist_ltm, only : rednum, redipt, redklis, redilis, redeval, redkval, redrval, redival
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_CustomCompoundBondForce), intent(inout) :: OMM_ResdRestraintForce
     type(OpenMM_DoubleArray) :: params
     type(OpenMM_IntArray) :: particles
     integer, intent(in) :: n, nunique, resd_type, npairs, nparticles
     integer :: iparam, i, iparticles(nparticles), npair, npart
     real(chm_real) :: klis(npairs+2)
     real*8 :: redpwr, k_conversion
     k_conversion = OpenMM_KJPerKcal * (OpenMM_AngstromsPerNm ** (redeval(n)*redival(n)) )
     call OpenMM_DoubleArray_create(params, npairs+2)
     call OpenMM_IntArray_create(particles, nparticles)

     ! Add pair non-specific parameters
     klis(1) = redkval(n) * k_conversion
     klis(2) = redrval(n) / (OpenMM_AngstromsPerNm)**redival(n)
     npair = 2
     npart = 0
     do i = redipt(n)+1, redipt(n+1)
        npair = npair + 1
        npart = npart + 1
        klis(npair) = redklis(i)
        iparticles(npart) = redilis(1,i)-1
        npart = npart + 1
        iparticles(npart) = redilis(2,i)-1
     enddo
     do i=1, npairs+2
        call OpenMM_DoubleArray_set(params,i,klis(i))
     enddo
     do i=1, nparticles
        call OpenMM_IntArray_set(particles,i,iparticles(i))
     enddo
     iparam = OpenMM_CustomCompoundBondForce_addBond(OMM_ResdRestraintForce, particles, params)

     call OpenMM_DoubleArray_destroy(params)
     call OpenMM_IntArray_destroy(particles)
   end subroutine add_resd_addbond

   subroutine print_resd_restraints(system, restraint, resd_type, eval, ival)

     type(OpenMM_System), intent(in) :: system
     type(OpenMM_CustomCompoundBondForce), intent(in) :: restraint
     integer, intent(in) :: resd_type, eval, ival
     type(OpenMM_DoubleArray) :: params
     type(OpenMM_IntArray) :: particles
     integer :: nbonds, nparticles, nbondparams, nglobalparams
     integer :: iparam, ibond, i
     character(len=1024) :: restr_function
     character*16 :: globalname, bondparamname, units
     character*32 :: fmat
     real*8 :: pvalue, value,  k_conversion
     integer, allocatable :: iparticles(:)

     nbonds = OpenMM_CustomCompoundBondForce_getNumBonds(restraint)
     nparticles = OpenMM_CustomCompoundBondForce_getNumParticlesPerBond(restraint)
     allocate (iparticles(nparticles))
     nbondparams = OpenMM_CustomCompoundBondForce_getNumPerBondParameters(restraint)
     nglobalparams = OpenMM_CustomCompoundBondForce_getNumGlobalParameters(restraint)

     call OpenMM_DoubleArray_create(params, nbondparams)
     call OpenMM_IntArray_create(particles, nparticles)

     call OpenMM_CustomCompoundBondForce_getEnergyFunction(restraint,restr_function)
     write(outu,'(a,i3,a,/,9x,a)')' CHARMM> Resd Restraint Function for type ',resd_type,&
          ': ', trim(restr_function)
     do iparam = 0, nglobalparams - 1
        call OpenMM_CustomCompoundBondForce_getGlobalParameterName(restraint,iparam,globalname)
        write(outu,'(a,a,a,g10.4)') ' CHARMM> ',trim(globalname),' = ', &
             OpenMM_CustomCompoundBondForce_getGlobalParameterDefaultValue(restraint,iparam)
     enddo
     write(fmat,'(a,i3,a)')'(a,i3,a,',nparticles,'i3,a)'
     do ibond = 0, nbonds-1
        call OpenMM_CustomCompoundBondForce_getBondParameters(restraint, ibond, particles, params)
        do i = 1, nparticles
           Call OpenMM_IntArray_get(particles,i,iparticles(i))
           iparticles(i) = iparticles(i) + 1
        enddo
        write(outu,fmat)' CHARMM> Interaction between atom indices for bond ', &
             ibond,':[',iparticles,']'
        k_conversion =  OpenMM_KJPerKcal * (OpenMM_AngstromsPerNm ** (eval*ival))
        do iparam = 1, nbondparams
           call OpenMM_CustomCompoundBondForce_getPerBondParameterName(restraint, iparam-1, &
                bondparamname)
           call OpenMM_DoubleArray_get(params,iparam, pvalue)
           if(iparam == 1) then
              value = pvalue / k_conversion
              write(units,'(a,i2,a)') ' kcal/mol*A^(',eval*ival,')'
           elseif(iparam == 2) then
              value = pvalue * OpenMM_AngstromsPerNm**(ival)
              units = ' Angstrom'
              if(ival > 1 ) write(units,'(a,i2,a)') 'A^(',ival,')'
           else
              value = pvalue
              units = ' '
           endif
          write(outu,'(a,a,a,f9.2,1x,a)') ' CHARMM> ',trim(bondparamname),'=',value, units
        enddo
     enddo

     deallocate(iparticles)
     call OpenMM_DoubleArray_destroy(params)

     call OpenMM_IntArray_destroy(particles)
           
   end subroutine print_resd_restraints

   subroutine getccbf_formula(n, formula)
     use resdist_ltm, only : rednum, redrval, redipt, redival, redmval, redeval, redkval
     integer :: n, ipt, jpt
     character(len=1024) :: formula
     character*32 :: build
     formula = 'del = '
     build = ''
     jpt = 0
     if(redival(n) == 1) then
        do ipt = redipt(n)+1, redipt(n+1)
           jpt = jpt+1
           if (jpt == 1) then
              write(build,'(a,i2,a,i2,a,i2,a)') 'k',jpt,' * distance(p',2*jpt-1,', p', &
                   2*jpt,')'
           else
              write(build,'(a,i2,a,i2,a,i2,a)') '+ k',jpt,' * distance(p',2*jpt-1,', p', &
                   2*jpt,')'
           endif
           formula = trim(formula) // trim(build)
           build = ''
        enddo
     else
        do ipt = redipt(n)+1, redipt(n+1)
           jpt = jpt+1
           if (jpt == 1) then
              write(build,'(a,i2,a,i2,a,i2,a,i2)') 'k',jpt,' * distance(p',2*jpt-1,', p', &
                   2*jpt,')^',redival(n)
           else
              write(build,'(a,i2,a,i2,a,i2,a,i2)') '+ k',jpt,' * distance(p',2*jpt-1,', p', &
                   2*jpt,')^',redival(n)
           endif
           formula = trim(formula) // trim(build)
           build = ''
        enddo
     endif
     formula = trim(formula) // '-rval;'
     build = ''
     write(build,'(i2,a,i2,a)') redeval(n),'*(del)^(',redeval(n),');'
     formula = 'sval * kval /' // trim(build) // trim(formula)
     if(redmval(n)>0) then
        formula = 'step(del)*'//trim(formula)
     else if(redmval(n)<0) then
        formula = 'step(-del)*'//trim(formula)
     endif
     formula = trimblanks(formula)
   end subroutine getccbf_formula
   
    logical function newformula(n, rtype)
    use resdist_ltm, only : rednum, redrval, redipt, redival, redmval, redeval, redkval
     integer :: n
     integer :: i, rtype(1:rednum)
     newformula = .true.
     do i = 1, n-1
        newformula = newformula &
          .and.      ( ( redipt(i+1)-redipt(i) /= redipt(n+1)-redipt(n) ) &    ! compare npairs
          .or.       ( redmval(i) /= redmval(n) ) &                            ! compare imode
          .or.       ( redival(i) /= redival(n) ) &                            ! compare ival
          .or.       ( redeval(i) /= redeval(n) ) )                            ! compare eval
        if(.not. newformula) then
           rtype(n) = rtype(i)
           return
        endif
     end do
   end function newformula


   character(len=64) function trimblanks64(str1)
     character(len=64) :: str1
     integer :: i, ls1, ls2
     ls1 = len_trim(str1)
     ls2 = 0
     do i = 1, ls1
        if(str1(i:i) .ne. ' ') then
           ls2 = ls2 + 1
           trimblanks64(ls2:ls2) = str1(i:i)
        endif
     enddo
     do i = ls2+1,64
        trimblanks64(i:i)=' '
     enddo
   end function trimblanks64

   character(len=1024) function trimblanks(str1)
     character(len=1024) :: str1
     integer :: i, ls1, ls2
     ls1 = len_trim(str1)
     ls2 = 0
     do i = 1, ls1
        if(str1(i:i) .ne. ' ') then
           ls2 = ls2 + 1
           trimblanks(ls2:ls2) = str1(i:i)
        endif
     enddo
     do i = ls2+1,1024
        trimblanks(i:i)=' '
     enddo
   end function trimblanks

   ! Setup CHARMM's proper and improper dihedral restraint terms
   subroutine setup_cons_dihe(system)
     use cnst_fcm, only : ncsphi, ccsb, ccsc,ccsd, ccsw, &
          ics, jcs, kcs, lcs
     use omm_ecomp, only : omm_incr_eterms
     type(OpenMM_System), intent(inout) :: system
     type(OpenMM_CustomTorsionForce), allocatable :: cons_improper(:), cons_proper(:)
     logical :: firstimpr, firstdihe
     integer :: i, nunique_improper, nunique_proper, &
          unique_proper(ncsphi), unique_improper(ncsphi)
     integer*4 :: group

     if(zero_consdihe) return
     if(ncsphi>0) group = omm_incr_eterms('cdihe')    
     firstimpr = .true.
     firstdihe = .true.

     nunique_proper = calc_uniqueProper(unique_proper)
     nunique_improper = calc_uniqueImproper(unique_improper)
     if(nunique_improper>0) allocate ( cons_improper(nunique_improper) )
     if(nunique_proper>0) allocate ( cons_proper(nunique_proper) )

     do i = 1, ncsphi
         if(ccsd(i) == 0) then  ! Improper dihedral restraint
           firstimpr = newimproper(i, unique_improper)
           call set_improper_restraint(system, group, firstimpr, cons_improper(unique_improper(i)), &
                ccsb(i), ccsc(i), ccsw(i), &
                ics(i)-1, jcs(i)-1, kcs(i)-1, lcs(i)-1)
           firstimpr = .false.
        else                   ! Proper Dihedral restraint
           firstdihe = newproper(i,unique_proper)
           call set_proper_restraint(system, group, firstdihe, cons_proper(unique_proper(i)), &
                ccsb(i), ccsc(i), ccsd(i), &
                ics(i)-1, jcs(i)-1, kcs(i)-1, lcs(i)-1)
           firstdihe = .false.
        end if
     end do

     if(allocated(cons_proper)) deallocate(cons_proper)
     if(allocated(cons_improper)) deallocate(cons_improper)
   end subroutine setup_cons_dihe

   integer function calc_uniqueProper(unique)
      use cnst_fcm, only : ncsphi, ccsb, ccsc,ccsd, ccsw, &
           ics, jcs, kcs, lcs
      integer, intent(inout) :: unique(:)
      integer :: i, nunique

      logical :: is_new_proper

     ! Find the number of unique restraints
     nunique = 0
     do i=1, ncsphi
        if(ccsd(i) /= 0) then
           unique(i) = 1
           is_new_proper = newproper(i,unique)
           if (is_new_proper) then 
              nunique = nunique + 1
              unique(i) = nunique
           endif
        endif
     enddo
     calc_uniqueProper = nunique
   end function calc_uniqueProper

   logical function newproper(i,u)
      use cnst_fcm, only : ncsphi, ccsb, ccsc,ccsd, ccsw, &
           ics, jcs, kcs, lcs
      integer, intent(inout) :: u(:)
      integer, intent(in) :: i
      integer :: j

      newproper = .true.
      do j = 1, i-1
         if(ccsd(j)/=0) then
            newproper = newproper &
                 .and. ( (ics(j) == ics(i)) .and. (jcs(j) == jcs(i)) &
                 .and. (kcs(j) == kcs(i)) .and. (lcs(j) == lcs(i)) )  &
                 .or.  ( (lcs(j) == ics(i)) .and. (kcs(j) == jcs(i)) &
                 .and. (jcs(j) == kcs(i)) .and. (ics(j) == lcs(i)) )
            if(.not. newproper) then
               u(j) = u(i)
               return
            endif
         endif
      enddo
    end function newproper
 

   integer function calc_uniqueImproper(unique)
      use cnst_fcm, only : ncsphi, ccsb, ccsc,ccsd, ccsw, &
           ics, jcs, kcs, lcs
      integer, intent(inout) :: unique(:)
      integer :: i, nunique

      logical :: is_new_improper

     ! Find the number of unique restraints
     nunique = 0
     do i=1, ncsphi
        if(ccsd(i) == 0) then
           unique(i) = 1
           is_new_improper = newimproper(i,unique)
           if (is_new_improper) then 
              nunique = nunique + 1
              unique(i) = nunique
           endif
        endif
     enddo
     calc_uniqueImproper = nunique
   end function calc_uniqueImproper

   logical function newimproper(i,u)
      use cnst_fcm, only : ncsphi, ccsb, ccsc,ccsd, ccsw, &
           ics, jcs, kcs, lcs
      integer, intent(inout) :: u(:)
      integer, intent(in) :: i
      integer :: j

      newimproper = .true.
      do j = 1, i-1
         if(ccsd(j)==0) then
            newimproper = newimproper &
                 .and. ( (ics(j) == ics(i)) .and. (jcs(j) == jcs(i)) &
                 .and. (kcs(j) == kcs(i)) .and. (lcs(j) == lcs(i)) )  &
                 .or.  ( (lcs(j) == ics(i)) .and. (kcs(j) == jcs(i)) &
                 .and. (jcs(j) == kcs(i)) .and. (ics(j) == lcs(i)) )
            if(.not. newimproper) then
               u(j) = u(i)
               return
            endif
         endif
      enddo
    end function newimproper
 
   subroutine set_improper_restraint(system, group, first, cons_improper, cb, cc, cw, iat, jat, kat, lat)
      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_CustomTorsionForce), intent(inout) :: cons_improper
      integer*4, intent(in) :: group
      integer, intent(in) :: iat, jat, kat, lat
      logical, intent(in) :: first
      real(chm_real), intent(in) :: cb, cc, cw

      type(OpenMM_DoubleArray) :: params
      character(len=1024) :: formula
      integer :: ijunk
      real*8 width

      width = cw * acos(-1.0) / 180

      if(first) then     ! Setup functional form
         formula = 'diff=theta-theta0; pi=3.141592653589793'
         formula = 'wrap=2*pi*(step(-diff-pi)-step(diff-pi)); ' // formula
         formula = 'k * ( max(0, ( abs(diff+wrap)-width) ) )^2; ' // formula
         formula = trimblanks(formula)
         if(prnlev >=7) &
              write(outu,'(a,/,a)')' Setting up improper dihedral restraint with formula',trim(formula)
         call OpenMM_CustomTorsionForce_create(cons_improper, trim(formula))
         call OpenMM_Force_setForceGroup(   &
           transfer(cons_improper, OpenMM_Force(0)), group)
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_improper, 'k');
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_improper, 'theta0');
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_improper, 'width');
         ijunk = OpenMM_System_addForce(system, transfer(cons_improper, OpenMM_Force(0)))
      endif
      call OpenMM_DoubleArray_create(params, 3)
      call OpenMM_DoubleArray_set(params, 1, cc * OpenMM_KJPerKcal)    ! k
      call OpenMM_DoubleArray_set(params, 2, cb)                       ! theta0
      call OpenMM_DoubleArray_set(params, 3, width )                   ! width
      ijunk = OpenMM_CustomTorsionForce_addTorsion(cons_improper, &
           iat, jat, kat, lat, params)
      call OpenMM_DoubleArray_destroy(params)

    end subroutine set_improper_restraint


    subroutine set_proper_restraint(system, group, first, cons_proper, cb, cc, cd, iat, jat, kat, lat)
      type(OpenMM_System), intent(inout) :: system
      type(OpenMM_CustomTorsionForce), intent(inout) :: cons_proper
      integer*4, intent(in) :: group
      integer, intent(in) :: cd, iat, jat, kat, lat
      logical, intent(in) :: first
      real(chm_real), intent(in) :: cb, cc
      
      type(OpenMM_DoubleArray) :: params
      character(len=1024) :: formula
      integer :: ijunk
      real*8 :: period, theta0
      
      period = cd
      theta0 = cb
      
      if(first) then     ! Setup functional form
         formula = 'diff=theta-theta0; pi=3.141592653589793'
         formula = 'wrap=2*pi*(step(-diff-pi)-step(diff-pi)); ' // formula
         formula = 'k * ( 1 - cos(period*(wrap+diff)) ); ' // formula
         formula = trimblanks(formula)
         if(prnlev >=7) &
              write(outu,'(a,/,a)')' Setting up proper dihedral restraint with formula',trim(formula)
         call OpenMM_CustomTorsionForce_create(cons_proper, trim(formula))
         call OpenMM_Force_setForceGroup(   &
           transfer(cons_proper, OpenMM_Force(0)), group)
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_proper, 'k');
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_proper, 'period');
         ijunk = OpenMM_CustomTorsionForce_addPerTorsionParameter(cons_proper, 'theta0');
         ijunk = OpenMM_System_addForce(system, transfer(cons_proper, OpenMM_Force(0)))
      endif
      call OpenMM_DoubleArray_create(params, 3)
      call OpenMM_DoubleArray_set(params, 1, cc * OpenMM_KJPerKcal)    ! k
      call OpenMM_DoubleArray_set(params, 2, period)                   ! period
      call OpenMM_DoubleArray_set(params, 3, theta0)                   ! theta0
      ijunk = OpenMM_CustomTorsionForce_addTorsion(cons_proper, &
           iat, jat, kat, lat, params)
      call OpenMM_DoubleArray_destroy(params)

    end subroutine set_proper_restraint

#endif
end module omm_restraint
