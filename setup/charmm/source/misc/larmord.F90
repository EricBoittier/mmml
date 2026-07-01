module larmord
    use chm_kinds
    implicit none
#if KEY_LARMORD==1
    private

     character(len=*),private, parameter :: file_name="larmord.src"
    !
    !   - global variable for LARMORD 
    !
    !  csscale - force constant in chemical shift (cs) restraint potential
    !  csexp - experimental cs
    !  rcsexp - random coil cs
    !  csmae - accuracy of predictions: mean absolute error (mae)
    !  csr - accuracy of predictions: correlation coefficient 
    !  lcsalphas - larmor cofficient (alpha) for distances
    !  lcsbetas - larmor exponent (beta) for distances
    !
    !  iatom - atom number of cs nucleus
    !  nlcs - number of larmor distances parameters
    !  ncs - number of cs
    !  csresid - residue number of each cs nucleus
    !  lcsresid - residue number of neighbor atoms in parameter file
    !
    !  lcsatom - atom name for neighbor atoms used to calculate distances
    !  lcsresname - corresponding residue names
    !  lcsnucleus - cs prediction nucleus for larmor parameters
    !  csnucleus - cs prediction nucleus (experimental cs)
    !  csresname - corresponding residue name

    real(chm_real),save,allocatable,dimension(:) :: csexp, csmae, csr, rcsexp, &
         lcsalphas, lcsbetas
    real(chm_real),save :: csscale, cstemp
    integer,save,allocatable,dimension(:) :: csresid,iatom,lcsresid,lcsjatom
    integer, save :: cunit,lunit,ncs,nlcs, csnatom
    logical, save :: qlarmord, qlarmord_on, qdebug_larmord, qflat, &
         qweight_larmord1, qweight_larmord2, qharm, qlog
    character(len=6),save,allocatable,dimension(:) :: lcsresname,lcsnucleus, &
         lcsatom,csresname,csnucleus

    public :: setup_larmord, ener_larmord, qlarmord, qlarmord_on

    contains
      subroutine allocate_larmord()
        use memory
        character(len=*),parameter :: routine_name="allocate_larmord"
        integer :: len

        len = ncs * csnatom * nlcs
        call chmalloc(file_name,routine_name,'csexp ',ncs,crl=csexp)
        call chmalloc(file_name,routine_name,'csmae ',ncs,crl=csmae)
        call chmalloc(file_name,routine_name,'csr ',ncs,crl=csr)
        call chmalloc(file_name,routine_name,'rcsexp ',ncs,crl=rcsexp)
        call chmalloc(file_name,routine_name,'csresid ',ncs,intg=csresid)
        call chmalloc(file_name,routine_name,'iatom ',ncs,intg=iatom)
        call chmalloc(file_name,routine_name,'csresname ',ncs,ch6=csresname)
        call chmalloc(file_name,routine_name,'csnucleus ',ncs,ch6=csnucleus)
        call chmalloc(file_name,routine_name,'lcsalphas ',len,crl=lcsalphas)
        call chmalloc(file_name,routine_name,'lcsbetas ',len,crl=lcsbetas)
        call chmalloc(file_name,routine_name,'lcsresid ',len,intg=lcsresid)
        call chmalloc(file_name,routine_name,'lcsjatom ',len,intg=lcsjatom)
        call chmalloc(file_name,routine_name,'lcsresname ',len,ch6=lcsresname)
        call chmalloc(file_name,routine_name,'lcsnucleus ',len,ch6=lcsnucleus)
        call chmalloc(file_name,routine_name,'lcsatom ',len,ch6=lcsatom)

        return
      end subroutine allocate_larmord

      subroutine deallocate_larmord()
        use memory
        integer :: len0, len1
        character(len=*),parameter :: routine_name="deallocate_larmord"

        if(allocated(csexp)) then
           len0 = size(csexp)
           len1 = size(lcsalphas)
           call chmdealloc(file_name,routine_name,'csexp ',len0,crl=csexp)
           call chmdealloc(file_name,routine_name,'csmae ',len0,crl=csmae)
           call chmdealloc(file_name,routine_name,'csr ',len0,crl=csr)
           call chmdealloc(file_name,routine_name,'rcsexp ',len0,crl=rcsexp)
           call chmdealloc(file_name,routine_name,'csresid ',len0,intg=csresid)
           call chmdealloc(file_name,routine_name,'iatom ',len0,intg=iatom)
           call chmdealloc(file_name,routine_name,'csresname ',len0,ch6=csresname)
           call chmdealloc(file_name,routine_name,'csnucleus ',len0,ch6=csnucleus)
           call chmdealloc(file_name,routine_name,'lcsalphas ',len1,crl=lcsalphas)
           call chmdealloc(file_name,routine_name,'lcsbetas ',len1,crl=lcsbetas)
           call chmdealloc(file_name,routine_name,'lcsresid ',len1,intg=lcsresid)
           call chmdealloc(file_name,routine_name,'lcsjatom ',len1,intg=lcsjatom)
           call chmdealloc(file_name,routine_name,'lcsresname ',len1,ch6=lcsresname)
           call chmdealloc(file_name,routine_name,'lcsnucleus ',len1,ch6=lcsnucleus)
           call chmdealloc(file_name,routine_name,'lcsatom ',len1,ch6=lcsatom)
        endif

        return
      end subroutine deallocate_larmord

      subroutine reallocate_larmord()
        use memory
        integer :: len0, len1
        character(len=*),parameter :: routine_name="reallocate_larmord"

        if(allocated(csexp)) then
           len0 = size(lcsalphas)
           len1 = nlcs
           call chmrealloc(file_name,routine_name,'lcsalphas ',len1,crl=lcsalphas)
           call chmrealloc(file_name,routine_name,'lcsbetas ',len1,crl=lcsbetas)
           call chmrealloc(file_name,routine_name,'lcsresid ',len1,intg=lcsresid)
           call chmrealloc(file_name,routine_name,'lcsjatom ',len1,intg=lcsjatom)
           call chmrealloc(file_name,routine_name,'lcsresname ',len1,ch6=lcsresname)
           call chmrealloc(file_name,routine_name,'lcsnucleus ',len1,ch6=lcsnucleus)
           call chmrealloc(file_name,routine_name,'lcsatom ',len1,ch6=lcsatom)
        endif

        return
      end subroutine reallocate_larmord

      subroutine setup_larmord()
        use chm_kinds
        use stream
        use string
        use number
        use comand
        use consta, only : kboltz
        use chutil, only : matom,makind,getres
        use psf
        use coord
        use select
        implicit none

        integer, dimension(natom) :: islct
        
        if(.not.qlarmord) then
           qlarmord_on = .false.
           qdebug_larmord = .false.
           qflat = .false.
           qlog = .false.
           qharm = .false.
           qweight_larmord1 = .false.
           qweight_larmord2 = .false.
           ncs = 0
           nlcs = 0

           ! Parse LarmorD commands
           if(indxa(comlyn,comlen, 'OFF')>0 .or. indxa(comlyn,comlen, 'ON')>0) then
              call wrndie(-5,'<SETUP_LARMORD>', &
                   ' LarmorD not active, need to setup first')
              return
           endif
           if(.not.(indx(comlyn,comlen, 'CUNIT',5)>0 .and. &
                indx(comlyn,comlen, 'LUNIT',5)>0)) then
              call wrndie(-5,'<SETUP_LARMORD>', &
                   ' LarmorD requires LUNIT and CUNIT to be specified on first usage')
              return
           endif
           csscale = gtrmf(comlyn,comlen,'SCALE',one)
           cstemp = gtrmf(comlyn,comlen,'TEMP',300.0)
           cstemp = kboltz * cstemp
           qdebug_larmord = (indxa(comlyn,comlen,'DEBUG')>0)
           qflat = (indxa(comlyn,comlen,'FLAT')>0)
           qweight_larmord1 = (indxa(comlyn,comlen,'WT1')>0)
           qweight_larmord2 = (indxa(comlyn,comlen,'WT2')>0)
           qharm = (indxa(comlyn,comlen,'HARM')>0)
           qlog = (indxa(comlyn,comlen,'LOG')>0)

           cunit = gtrmi(comlyn,comlen,'CUNIT',-1)
           lunit = gtrmi(comlyn,comlen,'LUNIT',-1)
           ! Select atoms to be considered as part of CS calculations
           call selcta(comlyn,comlen,islct,x,y,z,wmain,.false.)
           csnatom = count(islct == 1, natom)

           call read_larmord
           qlarmord = .true.
           qlarmord_on = .true.
        else
           if(indxa(comlyn,comlen,'RESET')>0) then
              if(prnlev>=2) write(outu,'(a)')' LARMORD> Resetting all flags'
              qflat = .false.
              qweight_larmord1 = .false.
              qweight_larmord2 = .false.
              qharm = .false.
              qlog = .false.
              ! Parse LarmorD commands
              csscale = gtrmf(comlyn,comlen,'SCALE',one)
              cstemp = gtrmf(comlyn,comlen,'TEMP',300.0)
              cstemp = kboltz * cstemp
              qdebug_larmord = (indxa(comlyn,comlen,'DEBUG')>0)
              qflat = (indxa(comlyn,comlen,'FLAT')>0)
              qweight_larmord1 = (indxa(comlyn,comlen,'WT1')>0)
              qweight_larmord2 = (indxa(comlyn,comlen,'WT2')>0)
              qharm = (indxa(comlyn,comlen,'HARM')>0)
              qlog = (indxa(comlyn,comlen,'LOG')>0)

              if( (indx(comlyn,comlen,'CUNIT',5)>0) .and. &
                   (indx(comlyn,comlen,'LUNIT',5)>0) ) then
                 cunit = gtrmi(comlyn,comlen,'CUNIT',-1)
                 lunit = gtrmi(comlyn,comlen,'LUNIT',-1)
                 ! Select atoms to be considered as part of CS calculations
                 call selcta(comlyn,comlen,islct,x,y,z,wmain,.false.)
                 csnatom = count(islct == 1, natom)
                 call read_larmord()
              endif

           elseif(indxa(comlyn,comlen,'CLEA')>0) then
              call deallocate_larmord()
              qlarmord = .false.
              qlarmord_on = .false.
              return
           elseif(indxa(comlyn,comlen, 'CALC')>0) then
              if(qlarmord) then
                 call calculate_larmord()
              else
                 if(prnlev>2) write(outu,'(a)') ' LARMOR> No calculation done, LarmorD not setup'
              endif
           else !Look to see if keys are updated
              if(indx(comlyn,comlen,'SCALE',5)>0) csscale=gtrmf(comlyn,comlen,'SCALE',one)
              if(indx(comlyn,comlen,'TEMP',4)>0) then
                 cstemp = gtrmf(comlyn,comlen,'TEMP',300.0)
                 cstemp = kboltz * cstemp
              endif
              if(indx(comlyn,comlen,'DEBUG',5)>0) &
                   qdebug_larmord = (indxa(comlyn,comlen,'DEBUG')>0)
              if(indx(comlyn,comlen,'FLAT',4)>0) &
                   qflat = (indxa(comlyn,comlen,'FLAT')>0)
              if(indx(comlyn,comlen,'WT1',3)>0) &
                   qweight_larmord1 = (indxa(comlyn,comlen,'WT1')>0)
              if(indx(comlyn,comlen,'WT2',3)>0) &
                   qweight_larmord2 = (indxa(comlyn,comlen,'WT2')>0)
              if(indx(comlyn,comlen,'HARM',4)>0) &
                   qharm = (indxa(comlyn,comlen,'HARM')>0)
              if(indx(comlyn,comlen,'LOG',3)>0) &
                   qlog = (indxa(comlyn,comlen,'LOG')>0)
              if(indxa(comlyn,comlen, 'OFF')>0) then
                 if(qlarmord) then
                    qlarmord_on = .false.
                    if(prnlev>2) write(outu,'(a)') ' LARMOR> LarmorD restraints turned off'
                 endif
              else if(indxa(comlyn,comlen, 'ON')>0) then
                 if(qlarmord) then
                    qlarmord_on = .true.
                    if(prnlev>2) write(outu,'(a)') ' LARMOR> LarmorD restraints turned on'
                 else
                    if(prnlev>2) write(outu,'(a)') ' LARMOR> Nothing done, LarmorD not setup'
                 endif
              endif
           endif
        endif

        ! commandline error checking
        if ( (.not. qlog) .and. (.not. qharm)) then
           call wrndie(-5,'<SETUP_LARMORD>', &
                ' Need to specify whether to use a harmonic (HARM) or log-harmonic (LOG) restraining potenital')
        endif
        
        if (qlog .and. qharm) then
           call wrndie(-5,'<SETUP_LARMORD>', &
                ' LOG-HARMONIC and HARMONIC are mutually exclusive')
        endif
        
        if (qlog .and. qflat) then
           call wrndie(-5,'<SETUP_LARMORD>', &
                ' LOG and FLAT are mutually exclusive')
        endif
        if ( qweight_larmord1 .and. qweight_larmord2) then
           call wrndie(-5,'<SETUP_LARMORD>', &
                ' WT1 and WT2 are mutually exclusive')
        endif
        return
      end subroutine setup_larmord

      subroutine read_larmord()
        use chm_kinds
        use stream
        use string
        use number
        use comand
        use chutil, only : matom,makind,getres
        use psf
        use coord
        use select
        implicit none
        
        integer :: i, j, k, nsele, tmpcomlen, prnlev_old
        real(chm_real) :: tmpcsalpha, tmpcsbeta
        character(len=6) :: tmpnuc, tmpresn, tmpcsatom
        character(len=50) :: tmpcomlyn
        integer,dimension(natom) :: islct, jslct
      
        if(.not. (iolev>0) ) return
        
        read(cunit,*) ncs
        read(lunit,*) nlcs

        if (allocated(csexp)) then    ! assume we need to reload data files
           call deallocate_larmord()
        endif
        if (.not. allocated(csexp)) then 
           call allocate_larmord()
           
           ! Read Chemical Shift Data
           do i=1, ncs
              read(cunit,*) csresname(i),csresid(i),csnucleus(i), &
                   csexp(i), rcsexp(i), csmae(i), csr(i)
              ! get atom number
              iatom(i)=matom(csresid(i),csnucleus(i),atype,ibase,1,nres,.true.)
           end do
           ! Read Larmor Predictor Parameters
           islct = 0
           jslct = 0
           
           j = 0
           do i=1, nlcs
              read(lunit,*) tmpnuc, tmpresn, tmpcsatom, tmpcsalpha, tmpcsbeta
              tmpcomlyn = trim('SELE RESNAME '//tmpresn//' .AND. TYPE '//tmpcsatom//' END')
              tmpcomlen = len(tmpcomlyn)

              prnlev_old = prnlev
              prnlev = 0
              call selcta(tmpcomlyn,tmpcomlen,islct,x,y,z,wmain,.false.)
              prnlev = prnlev_old

              call makind(natom,islct,jslct,nsele)
              if (qdebug_larmord .and. prnlev >=5) then
                 write(outu,'(i5,2x,i5,2x,i5)') tmpcomlyn, nsele, count(islct==1, natom)
              end if
              do k=1, nsele
                 j = j + 1
                 lcsresid(j) = getres(jslct(k),ibase,nres)
                 lcsnucleus(j) = tmpnuc
                 lcsresname(j) = tmpresn
                 lcsatom(j) = tmpcsatom
                 lcsalphas(j) = tmpcsalpha
                 lcsbetas(j) = tmpcsbeta
                 lcsjatom(j) = matom(lcsresid(j),lcsatom(j),atype,ibase,1,nres,.false.)
                 if (qdebug_larmord .and. prnlev >= 5) then
                    write(outu,*) &
                         lcsresname(j),lcsresid(j),lcsnucleus(j),lcsatom(j), &
                         lcsalphas(j), lcsbetas(j), lcsjatom(j)
                 endif
              end do
           end do
           nlcs = j
           call reallocate_larmord()
        endif
        return

      end subroutine read_larmord

      subroutine calculate_larmord()
        use chm_kinds
        use psf
        use coord, only : x, y, z, wmain
        use chutil, only : matom
        use stream
        use number

        implicit none
        integer :: i, j, iatm, jatm
        integer, parameter :: mark = -99999999
        logical :: process
        real(chm_real) :: dist, larmor_alpha, larmor_beta, err, eu = zero, csweight = one
        real(chm_real) :: cspred, cserror

        eu = zero
        do i=1, ncs
           cspred = rcsexp(i)
           iatm = iatom(i)
           do j = 1, nlcs
              larmor_alpha = lcsalphas(j)
              larmor_beta = lcsbetas(j)
              jatm = lcsjatom(j)
              process = (lcsnucleus(j) == csnucleus(i)) &
                   .and. iatm /= jatm .and. jatm /= mark
              if ( process ) then
                 dist = (x(iatm)-x(jatm))*(x(iatm)-x(jatm)) &
                      + (y(iatm)-y(jatm))*(y(iatm)-y(jatm)) &
                      + (z(iatm)-z(jatm))*(z(iatm)-z(jatm))
                 dist=sqrt(dist)
                 cspred = cspred + larmor_alpha * (dist**larmor_beta)
              end if
           end do
           cserror = (cspred - csexp(i))
           if (abs(cserror) .lt. csmae(i) .and. qflat) then
              cserror = zero
           end if
           
           ! add differential weights to the error?
           if (qweight_larmord1) then
              csweight =  csr(i) / csmae(i)
           else if (qweight_larmord2) then
              csweight =  1 / csmae(i)
           end if 

           eu = eu + csweight * csweight * cserror * cserror 

           if (qdebug_larmord .and. prnlev >= 5) then
              write(outu,*) &
                   "larmord>>",csresid(i),csresname(i),csnucleus(i),cspred,csexp(i)
           end if
           wmain(iatm) = cspred
        end do
        ! report either a log-harmonic or harmonic restraint energy
        if (qlog) then
            if (csscale .eq. zero) then
               eu = zero
            else
              eu =  cstemp * ncs * log (eu) / 2 
            endif
        else 
            eu = csscale * eu / ncs
        endif
        if(prnlev>=5) write(outu,'(a,f12.5)') "LARMORD>> Error ", eu
       
        return
      end subroutine calculate_larmord

      subroutine ener_larmord(eu, x, y, z, dx, dy, dz)
        use chm_kinds
        use psf, only : nres, atype, ibase, natom
        use chutil, only : matom
        use number
        use stream

        implicit none

        real(chm_real), intent(in) :: x(*), y(*), z(*)
        real(chm_real), intent(inout) :: eu, dx(*), dy(*), dz(*)
        integer :: i, j, iatm, jatm
        integer, parameter :: mark = -99999999
        logical :: process
        real(chm_real) :: dist, larmor_alpha, larmor_beta, tcs
        real(chm_real) :: cspred, dd, cserror, csweight = one
        real(chm_real), dimension(natom) :: tdx, tdy , tdz, tdx1, tdy1, tdz1 
        
        tdx1 = zero
        tdy1 = zero
        tdz1 = zero
        eu = zero
        if(.not.qlarmord_on) return

        do i=1, ncs
           cspred = rcsexp(i)
           iatm = iatom(i)
           tdx = zero
           tdy = zero
           tdz = zero

           do j = 1, nlcs
              larmor_alpha = lcsalphas(j)
              larmor_beta = lcsbetas(j)
              jatm = lcsjatom(j)
              process = (lcsnucleus(j) == csnucleus(i)) &
                   .and. iatm /= jatm .and. jatm /= mark
              if ( process ) then
                 dist = (x(iatm)-x(jatm))*(x(iatm)-x(jatm)) &
                      + (y(iatm)-y(jatm))*(y(iatm)-y(jatm)) &
                      + (z(iatm)-z(jatm))*(z(iatm)-z(jatm))
                 dist=sqrt(dist)
                 cspred = cspred + larmor_alpha * (dist**larmor_beta)
                 dd = - two  * larmor_beta * larmor_alpha * (dist**(larmor_beta-2))
                 tdx(jatm) = tdx(jatm) + (x(iatm)-x(jatm)) * dd
                 tdy(jatm) = tdy(jatm) + (y(iatm)-y(jatm)) * dd
                 tdz(jatm) = tdz(jatm) + (z(iatm)-z(jatm)) * dd
                 tdx(iatm) = tdx(iatm) - (x(iatm)-x(jatm)) * dd
                 tdy(iatm) = tdy(iatm) - (y(iatm)-y(jatm)) * dd
                 tdz(iatm) = tdz(iatm) - (z(iatm)-z(jatm)) * dd
              end if
           end do
           cserror = (cspred - csexp(i))
           if (abs(cserror) .lt. csmae(i) .and. qflat) then
              cserror = zero
              dd = zero
           endif

           ! add differential weights to the error?
           if (qweight_larmord1) then
              csweight =  csr(i) / csmae(i)
           elseif (qweight_larmord2) then
              csweight =  1 / csmae(i)
           endif
           
           eu = eu + csweight * csweight * cserror * cserror 
           tdx1 = tdx1 + tdx * csweight * csweight * cserror 
           tdy1 = tdy1 + tdy * csweight * csweight * cserror 
           tdz1 = tdz1 + tdz * csweight * csweight * cserror 

        end do
        
        ! use either a log-harmonic or harmonic restraint potential
        if (qlog) then
            if (csscale == zero) then
               tcs =  zero
               eu = zero
            else
              tcs = cstemp * ncs / eu / 2
              eu =  cstemp * ncs * log (eu) / 2 
            endif
        else 
            tcs = csscale / ncs
            eu = csscale * eu / ncs
        endif
        
        tdx1 = tcs * tdx1
        tdy1 = tcs * tdy1
        tdz1 = tcs * tdz1
        
        
        do i=1, natom
           dx(i) = dx(i) + tdx1(i)
           dy(i) = dy(i) + tdy1(i)
           dz(i) = dz(i) + tdz1(i)
        end do
        return
      end subroutine ener_larmord
#endif
    end module larmord
