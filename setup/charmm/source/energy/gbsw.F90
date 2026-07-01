module gbsw
  use gb_common,only:alph_gb !we want to access Born radii via scalar command
  use chm_kinds
  use chm_types
  use bases_fcm
  implicit none

  real(chm_real),allocatable,dimension(:),save :: gbx,gby,gbz, &
       saverpb,radius,gbev,xgpleb,ygpleb,zgpleb, &
       xgp,ygp,zgp,wang, &
       rgp, wrad, dvsum, &
       gvdweps, gvdwr, gvdwaa, bb0, bb1, rborn, &
       gbdx,gbdy,gbdz, &
       dGdRGB, sfprod0gb,sfprod0vdw


  real(chm_real),allocatable,dimension(:,:),save :: gbpa
  real(chm_real4),allocatable,dimension(:),save ::  &
       xgbref,ygbref,zgbref, &
       tmpdist

  integer(chm_int2), allocatable :: natomingp(:,:,:)
  integer, allocatable :: gptoalst(:,:,:)
  integer, allocatable, dimension(:), save :: &
       startalst, gblookup, gbimattr, gbimatpt, gbitrn

  ! used as logical but want 1 byte per element
  integer(kind=int_byte),allocatable,dimension(:),save :: &
       chkgvdw, checkipgb, checkipvdw

  !------ Integer ---------------------------
  integer, save :: igbfrq,  nang, nrad, &
       IGBSWcall, &
       nxgp, nygp, nzgp, natlst, nbusygp, &
       natgb, &
       gbnatim, oldtotupd, &
       ipforce, &
       oldnatim, oldnatgb, &
       ngbsel ! hybrid solvent cphmd

  !------ Real ---------------------------
  real(chm_real),save :: epsp, epsw, kappa, sgamma,  &
       tmembgb, rcylngb, &
       sw, aa0, aa1, pbradsf, msw, &
       rminint, dgp, rbuffer, rsasa, &
       xgcen, ygcen, zgcen, &
       gbelec, gbsurf, gvdw, egouy, evolt ! added energy terms for GC and TM voltage
  real(chm_real), save :: acons, Volt, GKappa, OffSt, Psi0, GAlpha, TempFac ! global values for GC term

  !----- Logical ---------------------------
  LOGICAL, save :: QGBSW, QGvdW, Qrotinv, QGouyC, QVolt ! added terms for Gouy Chapman and TM voltage logicals (cb3)
  LOGICAL, private, save :: Qhybrid,Qradii

  !===============================================================
contains
  !---------------------------------------------------------------

  SUBROUTINE Gbsw_set(COMLYN, COMLEN)
    !------------------------------------------------------------------------
    ! Generalized Born model with a simple switching function
    !
    ! Aug, 2004 Solvent-solute nonpolar dispersion term added (Wonpil Im)
    ! Aug, 2002 Wonpil Im
    !

  use dimens_fcm
  use stream
  use string
  use psf
  use param
  use coord
  use number
  use image
  use deriv
  use memory
  use select

    character(len=*) :: comlyn
    integer ::      comlen

    !---- local -------------
    integer ::imode
    integer, allocatable,dimension(:) :: titres,titgrp,firgrp, &
         lasgrp,chkhis
    real(chm_real),allocatable,dimension(:) :: titQ
    integer ::ntitres
    integer ::nradtmp
    integer ::valence ! Valence of slat for GC term
    integer ::i
! cb3 made islct and gbsw_sel automatic arrays
    integer :: gbsw_sel(natom), islct(natom)
    real(chm_real)  conc,temp
    real(chm_real)  rmidint,rmaxint
    real(chm_real)  coefaa0(11),coefaa1(11),radsf(10)
    real(chm_real)  sgvdw,sgb
    real(chm_real)  h2oeps,h2or
    real(chm_real) anfr, area ! local values for GC term
    logical QGBener,QGBparm,QpKa,Qsrmodel
    logical Qmolsurf
    logical qhbok
    data coefaa0 / 0.0477104D0,-0.0810635D0,-0.14813D0,-0.1801D0, &
         -0.167984D0,-0.154184D0,-0.173099D0,-0.22786D0, &
         -0.306411D0,-0.394277D0,-0.482022D0 /
    data coefaa1 / 1.39997D0, 1.60004D0, 1.72925D0, 1.81745D0, &
         1.85601D0, 1.88639D0, 1.94527D0, 2.03593D0, &
         2.14722D0, 2.26453D0, 2.38013D0 /
    data radsf / 0.979D0, 0.965D0, 0.952D0, 0.939D0, 0.927D0, &
         0.914D0, 0.901D0, 0.888D0, 0.875D0, 0.861D0 /

    if(prnlev > 2) then
       write(outu,'(/)')
       write(outu,'(6x,3a)') &
            'Generalized Born model with a simple switching function (GBSW)'
       write(outu,'(/)')
    endif

    ! reset all GBSW parameters
    ! ------------------------------------------------------------------------

    if(INDXA(COMLYN,COMLEN,'RESE') > 0) then
       if(QGBSW) then
          QGBSW=.false.
          call chmdealloc('gbsw.src','Gbsw_set','saverpb',natom,crl=saverpb)
          call chmdealloc('gbsw.src','Gbsw_set','rborn',natom,crl=rborn)
          call chmdealloc('gbsw.src','Gbsw_set','rborn',natom,crl=alph_gb)
          call chmdealloc('gbsw.src','Gbsw_set','dvsum',natom,crl=dvsum)
          call chmdealloc('gbsw.src','Gbsw_set','gbdx',natom,crl=gbdx)
          call chmdealloc('gbsw.src','Gbsw_set','gbdy',natom,crl=gbdy)
          call chmdealloc('gbsw.src','Gbsw_set','gbdz',natom,crl=gbdz)
          call chmdealloc('gbsw.src','Gbsw_set','dgdrgb',natom,crl=dgdrgb)

          call chmdealloc('gbsw.src','Gbsw_set','xgbref',natom,cr4=xgbref)
          call chmdealloc('gbsw.src','Gbsw_set','ygbref',natom,cr4=ygbref)
          call chmdealloc('gbsw.src','Gbsw_set','zgbref',natom,cr4=zgbref)
          call chmdealloc('gbsw.src','Gbsw_set','xgpleb',nang,crl=xgpleb)
          call chmdealloc('gbsw.src','Gbsw_set','ygpleb',nang,crl=ygpleb)
          call chmdealloc('gbsw.src','Gbsw_set','zgpleb',nang,crl=zgpleb)
          call chmdealloc('gbsw.src','Gbsw_set','xgp',nang,crl=xgp)
          call chmdealloc('gbsw.src','Gbsw_set','ygp',nang,crl=ygp)
          call chmdealloc('gbsw.src','Gbsw_set','zgp',nang,crl=zgp)
          call chmdealloc('gbsw.src','Gbsw_set','wang',nang,crl=wang)
          call chmdealloc('gbsw.src','Gbsw_set','gbpa',3,3,crl=gbpa)
          call chmdealloc('gbsw.src','Gbsw_set','gbev',3,crl=gbev)
          call chmdealloc('gbsw.src','Gbsw_set','rgp',nrad,crl=rgp)
          call chmdealloc('gbsw.src','Gbsw_set','wrad',nrad,crl=wrad)
          call chmdealloc('gbsw.src','Gbsw_set','checkipgb',natom*nang*nrad,iby=checkipgb)
          call chmdealloc('gbsw.src','Gbsw_set','checkipvdw',natom*nang*nrad,iby=checkipvdw)
          if (allocated(sfprod0gb)) then
             call chmdealloc('gbsw.src','Gbsw_set','sfprod0gb',size(sfprod0gb),crl=sfprod0gb)
             call chmdealloc('gbsw.src','Gbsw_set','sfprod0vdw',size(sfprod0vdw),crl=sfprod0vdw)
          endif
          ! Gvdw part
          if(QGvdW) then
             QGvdW=.false.
             call chmdealloc('gbsw.src','Gbsw_set','gvdweps',natom,crl=gvdweps)
             call chmdealloc('gbsw.src','Gbsw_set','gvdwr',natom,crl=gvdwr)
             call chmdealloc('gbsw.src','Gbsw_set','gvdwaa',natom,crl=gvdwaa)
             call chmdealloc('gbsw.src','Gbsw_set','bb0',natom,crl=bb0)
             call chmdealloc('gbsw.src','Gbsw_set','bb1',natom,crl=bb1)
             call chmdealloc('gbsw.src','Gbsw_set','chkgvdw',natom,iby=chkgvdw)
          endif

          ! from GB_LOOKUP
          if (allocated(startalst)) call chmdealloc('gbsw.src','Gbsw_set','startalst',nbusygp,intg=startalst)
          if (allocated(gblookup)) call chmdealloc('gbsw.src','Gbsw_set','gblookup',natlst,intg=gblookup)
          if (allocated(tmpdist)) call chmdealloc('gbsw.src','Gbsw_set','tmpdist',natlst,cr4=tmpdist)
          call chmdealloc('gbsw.src','Gbsw_set','natomingp',nxgp,nygp,nzgp,ci2=natomingp)
          call chmdealloc('gbsw.src','Gbsw_set','gptoalst',nxgp,nygp,nzgp,intg=gptoalst)
          call chmdealloc('gbsw.src','Gbsw_set','gbx',natgb,crl=gbx)
          call chmdealloc('gbsw.src','Gbsw_set','gby',natgb,crl=gby)
          call chmdealloc('gbsw.src','Gbsw_set','gbz',natgb,crl=gbz)
          call chmdealloc('gbsw.src','Gbsw_set','radius',natgb,crl=radius)

          if(natgb > natom) then
             call chmdealloc('gbsw.src','Gbsw_set','gbimattr',(natim-natom)*8+1, &
                  intg=gbimattr)
             call chmdealloc('gbsw.src','Gbsw_set','gbimatpt',ntrans,intg=gbimatpt)
             call chmdealloc('gbsw.src','Gbsw_set','gbitrn',natgb,intg=gbitrn)
          endif
          ! cb3 added these, some may not be needed, but some casued problems
          ! with repeated gbsw reset and then calls
          ! Zero counters
          IGBSWcall = 0
          natlst = 0
          nbusygp = 0
          natgb = 0
          gbnatim = 0
          oldtotupd = 0
          ipforce = 0
          oldnatim = 0
          oldnatgb = 0
          if(QGOuyC) QGouyC = .false.
          if(QVolt) QVolt = .false.
          if(prnlev > 2) write(outu,'(a)')  &
               'All previous setup for GBSW is cleared now'
       else
          if(prnlev > 2) write(outu,'(a)') 'Nothing is setup for GBSW'
       endif
       return
    endif


    ! get some inputs
    ! ------------------------------------------------------------------------

    if(QGBSW) then
       ! GB energy and forces updating frequency
       igbfrq   = gtrmi(comlyn,comlen,'IGBF',1)
       if(prnlev>2) then
          write(outu,102) &
               'GB energy and forces will be updated every', &
               igbfrq,'step(s)'
          write(outu,103)
       endif
       return
    endif

    ! nang     : number of angular integration points
    ! nradtmp  : number of radial integration points
    ! epsp    : protein dielectric constant
    ! epsw    : solvent dielectric constatn
    ! conc     : salt concentration
    ! temp     : temperature
    ! sgamma   : nopolar surface tension coefficients
    ! rmaxint  : maximum distance for radial integration
    ! dgp      : grid spacing for lookup table
    ! rbuffer  : buffer length for lookup table
    ! tmembgb  : tickness of membrane centered at Z=0
    ! rcylngb  : radius of cylinder oriented along Z
    ! igbfrq   : GB energy and forces updating frequency
    ! QGBener  : GBEnergy?
    ! QGBparm  : GBparameterization?
    ! Qmolsurf : Molecular surface approximation?
    ! QpKa     : pKa calculations?
    ! Qsrmodel : use only single residue as a model compound?

    nang     = gtrmi(comlyn,comlen,'NANG',38)
    nradtmp  = gtrmi(comlyn,comlen,'NRAD',0)
    epsp    = gtrmf(comlyn,comlen,'EPSP',ONE)
    epsw    = gtrmf(comlyn,comlen,'EPSW',EIGHTY)
    conc     = gtrmf(comlyn,comlen,'CONC',ZERO)
    temp     = gtrmf(comlyn,comlen,'TEMP',THRHUN)
    sgamma   = gtrmf(comlyn,comlen,'SGAM',ZERO)
    rsasa    = gtrmf(comlyn,comlen,'RSAS',ZERO)
    rmaxint  = gtrmf(comlyn,comlen,'RMAX',TWENTY)
    dgp      = gtrmf(comlyn,comlen,'DGP',ONEPT5)
    rbuffer  = gtrmf(comlyn,comlen,'RBUF',ZERO)
    tmembgb  = gtrmf(comlyn,comlen,'TMEM',ZERO)
    rcylngb  = gtrmf(comlyn,comlen,'RCYL',ZERO)
    igbfrq   = gtrmi(comlyn,comlen,'IGBF',1)
    h2oeps   = gtrmf(comlyn,comlen,'H2OE',ZERO)
    h2or     = gtrmf(comlyn,comlen,'H2OR',ZERO)
    sgb      = gtrmf(comlyn,comlen,'SGB',ONE)
    sgvdw    = gtrmf(comlyn,comlen,'SGVD',ONE)
    QGBener  = ( INDXA(COMLYN,COMLEN,'GBEN') > 0 )
    QGBparm  = ( INDXA(COMLYN,COMLEN,'GBPA') > 0 )
    Qmolsurf = ( INDXA(COMLYN,COMLEN,'MOLS') > 0 )
    QGvdW    = ( INDXA(COMLYN,COMLEN,'GVDW') > 0 )
    QpKa     = ( INDXA(COMLYN,COMLEN,'PKA') > 0 )
    Qsrmodel = ( INDXA(COMLYN,COMLEN,'SRMO') > 0 )
    Qrotinv  = ( INDXA(COMLYN,COMLEN,'ROTI') > 0 )
!!!! Get parameters for Gouy Chepman and TM voltage model
    QGouyC = (indxa(comlyn,comlen, 'GOUY') > 0)
    if(conc > zero) kappa=sqrt(epsw/(2530.362733d0*conc/temp))

    Qradii = ( INDXA(COMLYN,COMLEN,'RADII') > 0 ) !flag to print GB radii

    !-Hybrid CPHMD: select atoms for GB radii calculation---------------
    ! Qhybrid: GB protonation free-energy and explicit solvent dynamics
    Qhybrid  = ( INDXA(COMLYN,COMLEN,'HYBRID') > 0 )
    if (Qhybrid) then
       QGBener = .false.
       QGvdw = .false.
       QpKa = .false.
       sgamma = zero
       CALL SELCTA(COMLYN,COMLEN,GBSW_SEL,X,Y,Z,WMAIN,.TRUE.)
       ngbsel = 0
       qhbok = .true.
       do i=1,natom
          if (gbsw_sel(i) .eq. 1) then
             ngbsel = ngbsel + 1
             qhbok = (ngbsel == i) .and. qhbok
          endif
       enddo
       if (ngbsel == natom .or. .not. qhbok) &
            call wrndie(-4,'<gbsw>',&
            'Problem with hybrid setup: not 1st segment or all atoms selected')
    else
       ngbsel = natom
    end if
   !-Hybrid CPHMD------------------------------------------------------

    volt = gtrmf(comlyn,comlen,'VOLT', zero)
    if (abs(volt) > zero) QVolt = .true.
    if(QGouyC .or. QVolt) then
       valence = gtrmi(comlyn,comlen,'VALE', 1)
       if(conc <= zero) then
          CALL WRNDIE(-4,'<CHARMM>', &
               'Need finite salt concentration for Gouy Chapman or TM Voltage potentials')
       else
          gkappa = 5.622667 * float(valence) * sqrt( conc / temp )
       endif
    endif
    if(QGouyC) then
       anfr = three * ptone
       anfr = gtrmf(comlyn,comlen,'ANFR',anfr)  !Fraction of anionic lipids
       area = Ninety - Twenty
       area = gtrmf(comlyn,comlen,'AREA',area)  !Surface area per lipid head group
       offst = gtrmf(comlyn,comlen,'OFFS',three) 
       Psi0 = -2334.2 * anfr/area/sqrt(temp*conc)
       GAlpha = log(Psi0 + sqrt( Psi0*Psi0 + 1))/float(valence)
       Psi0 = 1.7235D-4 * temp/float(valence) * log(Psi0 + sqrt( Psi0*Psi0 + 1))
       GAlpha = (exp(Galpha)-one)/(exp(GAlpha)+one)
       TempFac = 3.974d-3*temp
    endif
    if(QVolt) Acons = Volt / (Two + Four * Ten * GKappa * tmembgb)

    nrad=nradtmp
    if(nrad == 0) nrad=24
    !--- switching length
    if(Qmolsurf) then
       sw=TWO*PTONE
       sw=gtrmf(comlyn,comlen,'SW',sw)
       if(sw == PTONE) then
          aa0=1.26422D0
          aa1=0.0592646D0
       elseif(sw == TWO*PTONE) then
          aa0=1.20447D0
          aa1=0.186595D0
       elseif(sw == THREE*PTONE) then
          aa0=1.1177D0
          aa1=0.340755D0
       else
          CALL WRNDIE(-1,'<CHARMM>', &
               'GBSWms works only for sw=0.1, 0.2, and 0.3')
       endif
    else
       sw=THREE*PTONE
       sw=gtrmf(comlyn,comlen,'SW',sw)
       aa0=coefaa0(int(sw*10.0D0+rsmall)+1)
       aa1=coefaa1(int(sw*10.0D0+rsmall)+1)
    endif

    !--- aa0 : coefficient for CFA term
    !--- aa1 : coefficient for additonal term
    !--- msw : membrane switching length

    aa0 = gtrmf(comlyn,comlen,'AA0',aa0)
    aa1 = gtrmf(comlyn,comlen,'AA1',aa1)
    msw= gtrmf(comlyn,comlen,'MSW',SW)
    if((QGouyC .or. QVolt) .and. (two*msw /= offst) .and. prnlev > 2) then
       write(outu,102)' Lipid headgroup switching region for membrane and Gouy Chapman should match'
       write(outu,103)' Resetting MSW to be consistent with OFFS. MSM ->', offst / two
    endif

    if(sgb /= ONE) then
       aa0=aa0*sgb
       aa1=aa1*sgb
    endif


    !--- write some information
    !------------------------------------------------------------------------

    if(prnlev>2) then
       IF(Qrotinv) write(outu,102)  &
            'GBSW uses rotational invariance numerical quadrature procedure;'
        !-Hybrid-------------------------------------------------------------
       if(Qhybrid) write(outu,102) &
            'Number of atoms for hybrid GBSW    (ngbsel) =',ngbsel
       write(outu,102) &
            'Number of angular integration points (nang) =',nang
       write(outu,102) &
            'Number of radial  integration points (nrad) =',nrad
       write(outu,103) &
            'Maximum in radial integration        (rmax) =',rmaxint, &
            '[Angs]'
       write(outu,103)

       write(outu,103) &
            'Solvent dielectric constant          (epsw) =',epsw
       write(outu,103) &
            'Protein dielectric constant          (epsp) =',epsp
       if(conc > 0.0D0) then
          write(outu,103) &
               'Salt concentration                   (conc) =',conc,'[M]'
          write(outu,103) &
               'Temperature                          (temp) =',temp,'[K]'
          write(outu,103) &
               'Debye screening length              (kappa) =',kappa, &
               '[Angs]'
       endif
       if(sgamma > 0.0D0) then
          write(outu,103) &
               'Surface tension coefficient        (sgamma) =',sgamma, &
               '[kcal/mol/Angs**2]'
       endif
       if(tmembgb > 0.0D0) then
          write(outu,103) &
               'Membrane thickness centered at Z=0  (tmemb) =',tmembgb, &
               '[Angs]'
          write(outu,103) &
               'Membrane switching length           (msw*2) =',msw*2, &
               '[Angs]'
          write(outu,103)
          if(rcylngb > 0.0D0) then
             write(outu,103) &
                  'Radius of cylinder oriented along Z (rcyln) =',rcylngb, &
                  '[Angs]'
          endif
          if(QGouyC) then
             write(outu,102) 'Gouy Chapman theory used to represent anionic lipids'
             write(outu,103) &
                  'Membrane thickness               (tmemb) =', tmembgb, &
                  ' [Angs]'
             write(outu,103) 'Offset (switching area) offst=msw=',offst, &
                  ' [Angs]'
             write(outu,103) &
                  'Ionic strength           (valence*conc)=', valence*conc, &
                  ' [moles/liter]'
             write(outu,103) &
                  'Anionic lipid fraction           (anfr) =', anfr
             write(outu,103) &
                  'Area per lipid head group         (area) =', area, &
                  '[Angs^2]'
          endif
          if(QVolt) write(outu,103) &
               'Transmembrane potential              (volt)=', volt, &
               ' [volts] applied'
       endif
       write(outu,103)

       if(QGvdW) then
          write(outu,103) &
               'The attractive vdW nonpolar solvation is considered'
          write(outu,103) &
               'Epsilon of water oxygen            (h20eps) =',h2oeps, &
               '[kcal/mol]'
          write(outu,103) &
               'Radius (Rmin) of water oxygen        (h2or) =',h2or, &
               '[Angs]'
       endif
       write(outu,103)

       write(outu,103) &
            'Switching length                     (sw*2) =',sw*2,'[Angs]'
       write(outu,103)
       write(outu,103) &
            'Coefficient for the CFA term          (aa0) =',aa0
       write(outu,103) &
            'Coefficient for the correction term   (aa1) =',aa1
       if(Qmolsurf) then
          write(outu,103) &
               'GBSW approximation to molecular surface representation'
       endif
       write(outu,103)

       write(outu,103) &
            'Grid spacing for lookup table         (dgp) =',dgp,'[Angs]'
       write(outu,103) &
            'Buffer size for lookup table      (rbuffer) =',rbuffer, &
            '[Angs]'
       write(outu,103)
    endif

    pbradsf=1.0D0
    if(sw > 0.0D0) then
       pbradsf=radsf(int(sw*10.0D0+rsmall))
       if(prnlev>2) then
          write(outu,103) &
               'NOTE : PB radii are scaled by (PB radii+sw)*factor'
          write(outu,103) &
               '       where sw =',sw,'and factor =',pbradsf
          write(outu,103)
       endif
    endif

    if(prnlev>2) write(outu,102) &
         'GB energy and forces will be updated every',igbfrq,'step(s)'
    if(prnlev>2) write(outu,103)

102 format(6x,a,1x,i7,1x,a)
103 format(6x,a,1x,f7.3,1x,a,f7.3,1x,a)

    !--- atom selection for pKa calculations
    !------------------------------------------------------------------------

    if(QpKa) then
       !--- 100 = maxium number of titratable residues
       !--- titres : target titratable residue number
       !--- titgrp : target group whose atomic charges are changed upon titration
       !--- firgrp : first group number of target titratable residue number
       !--- lasgrp : last group number of target titratable residue number
       !--- titQ   : total net charge of target titratable residue number
       !--- chkhis : check histidine specifications

       call chmalloc('gbsw.src','Gbsw_set','titres',100,intg=titres)
       call chmalloc('gbsw.src','Gbsw_set','titgrp',100,intg=titgrp)
       call chmalloc('gbsw.src','Gbsw_set','firgrp',100,intg=firgrp)
       call chmalloc('gbsw.src','Gbsw_set','lasgrp',100,intg=lasgrp)
       call chmalloc('gbsw.src','Gbsw_set','titQ',100,crl=titQ)
       call chmalloc('gbsw.src','Gbsw_set','chkhis',100,intg=chkhis)
       imode=0                 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       call SELRPN(comlyn,comlen,islct,natomt,1,imode, &
            .false.,1,' ',0,resid,res,ibase,segid,nictot,nsegt, &
            .true.,x,y,z,.true.,1,wmain)

       if(imode /= 0)then
          call wrndie(0,'<SEL1AT>','ATOM SELECTION PARSING ERROR')
       endif

       call CHK_SELECT(natom,islct, &
            ibase,nres,resid,res,ngrp,igpbs,atype, &
            ntitres,titres,titgrp, &
            firgrp,lasgrp,titQ,chkhis, &
            comlyn,comlen)
    endif

    !--- allocate necessary storages
    !------------------------------------------------------------------------

    if(.not.QGBSW) then
       IGBSWcall =0
       !--- saved PB radii
       call chmalloc('gbsw.src','Gbsw_set','saverpb',natom,crl=saverpb)
       !--- calculated Born radii
       call chmalloc('gbsw.src','Gbsw_set','rborn',ngbsel,crl=rborn)
       !--- pointer to get Born radii from scalar command
       call chmalloc('gbsw.src','Gbsw_set','rborn',ngbsel,crl=alph_gb)
       !--- extra array for energy and force calculations
       call chmalloc('gbsw.src','Gbsw_set','dvsum',ngbsel,crl=dvsum)
       !--- GB forces
       call chmalloc('gbsw.src','Gbsw_set','gbdx',ngbsel,crl=gbdx)
       call chmalloc('gbsw.src','Gbsw_set','gbdy',ngbsel,crl=gbdy)
       call chmalloc('gbsw.src','Gbsw_set','gbdz',ngbsel,crl=gbdz)
       call chmalloc('gbsw.src','Gbsw_set','dgdrgb',ngbsel,crl=dgdrgb)
       !--- save coordinates for updating Lookup table
       call chmalloc('gbsw.src','Gbsw_set','xgbref',ngbsel,cr4=xgbref)
       call chmalloc('gbsw.src','Gbsw_set','ygbref',ngbsel,cr4=ygbref)
       call chmalloc('gbsw.src','Gbsw_set','zgbref',ngbsel,cr4=zgbref)

       !--- angular integrations points and weights (from Lebedev Quadrature)
       call chmalloc('gbsw.src','Gbsw_set','xgpleb',nang,crl=xgpleb)
       call chmalloc('gbsw.src','Gbsw_set','ygpleb',nang,crl=ygpleb)
       call chmalloc('gbsw.src','Gbsw_set','zgpleb',nang,crl=zgpleb)
       call chmalloc('gbsw.src','Gbsw_set','wang',nang,crl=wang)

       !--- rotation of xgpleb.. for rotational invariance
       call chmalloc('gbsw.src','Gbsw_set','xgp',nang,crl=xgp)
       call chmalloc('gbsw.src','Gbsw_set','ygp',nang,crl=ygp)
       call chmalloc('gbsw.src','Gbsw_set','zgp',nang,crl=zgp)

       !--- for rotational invariance
       call chmalloc('gbsw.src','Gbsw_set','gbpa',3,3,crl=gbpa)
       call chmalloc('gbsw.src','Gbsw_set','gbev',3,crl=gbev)
       !--- radial integrtions points and weights
       call chmalloc('gbsw.src','Gbsw_set','rgp',nrad,crl=rgp)
       call chmalloc('gbsw.src','Gbsw_set','wrad',nrad,crl=wrad)
       call chmalloc('gbsw.src','Gbsw_set','checkipgb',ngbsel*nang*nrad,iby=checkipgb)
       call chmalloc('gbsw.src','Gbsw_set','checkipvdw',ngbsel*nang*nrad,iby=checkipvdw)
       !--- save integration points which only constribute forces

       !--- Gvdw part
       if(QGvdw) then
          call chmalloc('gbsw.src','Gbsw_set','gvdweps',natom,crl=gvdweps)
          call chmalloc('gbsw.src','Gbsw_set','gvdwr',natom,crl=gvdwr)
          call chmalloc('gbsw.src','Gbsw_set','gvdwaa',natom,crl=gvdwaa)
          call chmalloc('gbsw.src','Gbsw_set','bb0',natom,crl=bb0)
          call chmalloc('gbsw.src','Gbsw_set','bb1',natom,crl=bb1)
          call chmalloc('gbsw.src','Gbsw_set','chkgvdw',natom,iby=chkgvdw)

       endif

    endif

    !--- modify and save PB raii
    !------------------------------------------------------------------------

    if(.not.QGBSW) then
       call PBRADIUS(ngbsel,wmain,sw,pbradsf,saveRPB,rminint)
    endif

    !--- get angular integration points and weights using Lebedev's quadrature
    !------------------------------------------------------------------------

    if(nang ==  26) &
         call LD0026C(xgpleb,ygpleb,zgpleb, &
         wang,nang)
    if(nang ==  38) &
         call LD0038C(xgpleb,ygpleb,zgpleb, &
         wang,nang)
    if(nang ==  50) &
         call LD0050C(xgpleb,ygpleb,zgpleb, &
         wang,nang)
    if(nang ==  110) &
         call LD0110C(xgpleb,ygpleb,zgpleb, &
         wang,nang)
    if(nang ==  2030) &
         call LD2030C(xgpleb,ygpleb,zgpleb, &
         wang,nang)

    if(nang /= 26.and.nang /= 38.and.nang /= 50.and. &
         nang /= 110.and.nang /= 2030) then
       if(prnlev>2) write(outu,102)  &
            'nang should be one of 26, 38, 50, or 110'
       call wrndie(-5,'<gbsw>','wrong nang selection')
    endif


    !--- get the radial integration points and weights
    !--- using Gauss-Legendre quadrature
    !------------------------------------------------------------------------

    if(nradtmp == 0) then
       if(rminint == ZERO) rminint=HALF
       if(rminint-sw > HALF) rminint=rminint-sw
       rmidint=rminint+HALF
       call gauleg(rminint,rmidint,rgp,wrad,5,0)
       nradtmp=nrad-5
       call gauleg(rmidint,rmaxint,rgp,wrad,nradtmp,5)
    else
       rminint=HALF
       call gauleg(rminint,rmaxint,rgp,wrad,nrad,0)
    endif

    !--- IF one wants to calculate solvation energy here,
    !------------------------------------------------------------------------

    if(.not.QpKa) then

       !---     build a list of GvdW contribution from isolated atom

       if(Qgvdw) then
          if(prnlev > 6 ) write(outu,'(a)') &
               "Gbsw_set  calling GB_GvdW0"
          call GB_GvdW0(natom,X,Y,Z,vdWR,eff,ITC,IAC,atype, &
               h2oeps,h2or,rmaxint,sw,sgvdw)
       endif

       !---     build a lookup table
       if(prnlev > 6 ) write(outu,'(a)') &
            "Gbsw_set  calling GB_LOOKUP"
       call GB_LOOKUP(X,Y,Z)

       !---     calculate the Born radii and the electrostatic solvation energy

       if(prnlev > 6 ) write(outu,'(a)') &
            "Gbsw_set  calling GB_SOLV0"
       call GB_SOLV0(ngbsel,GBx,GBy,GBz,cg, &
            sw,aa0,aa1, &
            xgp,ygp,zgp,wang,nang, &
            rgp,wrad,nrad,rminint, &
            rcylngb, &
            rBORN,gbelec, &
            nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
            startalst,gblookup, &
            QGBparm,QGBener,ipforce)

       if(QGBener) then
          if(prnlev>2) then
             write(outu,104) &
                  'Electrostatic solvation energy   =',gbelec,'[kcal/mol]'
             write(outu,104)

             if(QGvdW) write(outu,104) &
                  'vdW nonpolar solvation energy    =',gvdw,'[kcal/mol]'
             write(outu,104)
          endif

          !---     calculate the surface area (sw=0.1)

          if(sgamma > 0.0D0) then
             if(prnlev > 6 ) write(outu,'(a)') &
                  "Gbsw_set  calling GB_SURF0"
             call GB_SURF0(natom,atype,GBx,GBy,GBz, &
                  rsasa,ptone,tmembgb,msw, &
                  xgp,ygp,zgp,wang,nang,gbsurf, &
                  nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
                  startalst,gblookup,QGBparm)

             if(prnlev>2) write(outu,104) &
                  'Surface area                     =',gbsurf,'[Angs**2]'
             if(QGvdW) then
                if(prnlev>2) write(outu,104) &
                     'Cavity nonpolar solvation energy =',gbsurf*sgamma, &
                     '[kcal/mol]'
             else
                if(prnlev>2) write(outu,104) &
                     'Nonpolar solvation energy        =',gbsurf*sgamma, &
                     '[kcal/mol]'
             endif
             if(prnlev>2) write(outu,104)
          endif

          if(QGvdW) then
             if(prnlev>2) write(outu,104) &
                  'Total solvation energy           =', &
                  gbelec+gvdw+gbsurf*sgamma,'[kcal/mol]'
          else
             if(prnlev>2) write(outu,104) &
                  'Total solvation energy           =', &
                  gbelec+gbsurf*sgamma,'[kcal/mol]'
          endif
       endif
    endif

104 format(6x,a,1x,f15.5,1x,a)

    !--- IF one wants to calculate pKa,
    !------------------------------------------------------------------------

    if(QpKa) then
       !--- build a lookup table
       if(prnlev > 6 ) write(outu,'(a)') &
            "Gbsw_set  calling GB_LOOKUP"
       call GB_LOOKUP(X,Y,Z)

       if(prnlev > 6 ) write(outu,'(a)') &
            "Gbsw_set  calling GB_pKa"
       call GB_pKa(natom,GBx,GBy,GBz,cg, &
            sw,aa0,aa1, &
            xgp,ygp,zgp,wang,nang, &
            rgp,wrad,nrad,rminint, &
            epsp,epsw,rBORN, &
            nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
            startalst,gblookup,dvsum, &
            igpbs,nictot,nseg,segid,resid,res,atype, &
            ntitres,titres,titgrp, &
            firgrp,lasgrp,titQ,chkhis, &
            Qsrmodel)

       call chmdealloc('gbsw.src','Gbsw_set','titres',100,intg=titres)
       call chmdealloc('gbsw.src','Gbsw_set','titgrp',100,intg=titgrp)
       call chmdealloc('gbsw.src','Gbsw_set','firgrp',100,intg=firgrp)
       call chmdealloc('gbsw.src','Gbsw_set','lasgrp',100,intg=lasgrp)
       call chmdealloc('gbsw.src','Gbsw_set','titQ',100,crl=titQ)
       call chmdealloc('gbsw.src','Gbsw_set','chkhis',100,intg=chkhis)
    endif

    if(.not.QGBSW) QGBSW = .true.

    return
  end subroutine gbsw_set


  SUBROUTINE CHK_SELECT(natom,islct, &
       ibase,nres,resid,res,ngrp,igpbs,atype, &
       ntitres,titres,titgrp,firgrp,lasgrp,titQ,chkhis, &
       comlyn,comlen)
    !--------------------------------------------------------------------------
    !---  Make the list of the atoms that will be taken into account for
    !---  the pKa calculations.
    !---
    !---  first atom number of ith residue = ibase(i)+1
    !---  last  atom number of ith residue = ibase(i+1)
    !---
    !---  first atom number of ith group = igpbs(i)+1
    !---  last  atom number of ith group = igpbs(i+1)
    !---

  use stream
  use string

    integer :: natom, islct(*), ibase(*),  &
         nres, ngrp, igpbs(*)
    integer :: ntitres,titres(*),titgrp(*), &
         firgrp(*),lasgrp(*),chkhis(*)
    real(chm_real)  titQ(*)
    character(len=*) :: resid(*), res(*), atype(*)
    character(len=*) :: comlyn
    integer :: comlen

    !--- local variables
    integer :: i,j,k,kk,l,m
    integer :: firresat,lasresat,natinres,cnt
    integer :: firgrpat,lasgrpat
    integer :: cnthis,tmphis(50)

    ntitres=0
    do i=1,nres
       firresat=ibase(i)+1
       lasresat=ibase(i+1)
       natinres=lasresat-firresat+1
       cnt=0
       do j=firresat,lasresat
          if(islct(j) == 1) cnt=cnt+1
       enddo
       if(cnt > 0.and.cnt /= natinres) then
          if(prnlev>2) write(outu,'(a)') &
               'Warining: atom selection must be based on residue'
          call wrndie(0,'<gbsw>','wrong atom selection')
       endif
       if(cnt == natinres) then
          ntitres=ntitres+1
          titres(ntitres)=i
       endif
    enddo

    if(ntitres == 0) then
       if(prnlev>2) write(outu,'(a)')  &
            'Warining: there is no selected residue '
       call wrndie(-5,'<gbsw>','wrong atom selection')
    endif

    kk=1
    nitres_loop: do i=1,ntitres
       j=titres(i)
       firresat=ibase(j)+1
       lasresat=ibase(j+1)
       if(i /= 1) kk=lasgrp(i-1)

       if(res(j)(1:4) == 'ASP ') then
          titQ(i)=-1.0D0
          do k=kk,ngrp
             firgrpat=igpbs(k)+1
             lasgrpat=igpbs(k+1)
             if(firgrpat == firresat) firgrp(i)=k
             if(firgrpat.ge.firresat.and.lasgrpat.le.lasresat) then
                do l=firgrpat,lasgrpat
                   if (atype(l)(1:4) == 'OD1 ') then
                      titgrp(i)=k
                   endif
                enddo
             endif
             if(lasgrpat == lasresat) then
                lasgrp(i)=k
                cycle nitres_loop
             endif
          enddo
       elseif(res(j)(1:4) == 'GLU ') then
          titQ(i)=-1.0D0
          do k=kk,ngrp
             firgrpat=igpbs(k)+1
             lasgrpat=igpbs(k+1)
             if(firgrpat == firresat) firgrp(i)=k
             if(firgrpat.ge.firresat.and.lasgrpat.le.lasresat) then
                do l=firgrpat,lasgrpat
                   if (atype(l)(1:4) == 'OE1 ') then
                      titgrp(i)=k
                   endif
                enddo
             endif
             if(lasgrpat == lasresat) then
                lasgrp(i)=k
                cycle nitres_loop
             endif
          enddo
       elseif(res(j)(1:4) == 'LYS ') then
          titQ(i)=+1.0D0
          do k=kk,ngrp
             firgrpat=igpbs(k)+1
             lasgrpat=igpbs(k+1)
             if(firgrpat == firresat) firgrp(i)=k
             if(firgrpat.ge.firresat.and.lasgrpat.le.lasresat) then
                do l=firgrpat,lasgrpat
                   if (atype(l)(1:4) == 'NZ  ') then
                      titgrp(i)=k
                   endif
                enddo
             endif
             if(lasgrpat == lasresat) then
                lasgrp(i)=k
                cycle nitres_loop
             endif
          enddo
       elseif(res(j)(1:4) == 'ARG ') then
          titQ(i)=+1.0D0
          do k=kk,ngrp
             firgrpat=igpbs(k)+1
             lasgrpat=igpbs(k+1)
             if(firgrpat == firresat) firgrp(i)=k
             if(firgrpat.ge.firresat.and.lasgrpat.le.lasresat) then
                do l=firgrpat,lasgrpat
                   if (atype(l)(1:4) == 'NE  ') then
                      titgrp(i)=k
                   endif
                enddo
             endif
             if(lasgrpat == lasresat) then
                lasgrp(i)=k
                cycle nitres_loop
             endif
          enddo
       elseif(res(j)(1:4) == 'HSP ') then
          titQ(i)=+1.0D0
          do k=kk,ngrp
             firgrpat=igpbs(k)+1
             lasgrpat=igpbs(k+1)
             if(firgrpat == firresat) firgrp(i)=k
             if(firgrpat.ge.firresat.and.lasgrpat.le.lasresat) then
                do l=firgrpat,lasgrpat
                   if (atype(l)(1:4) == 'ND1 ') then
                      titgrp(i)=k
                   endif
                enddo
             endif
             if(lasgrpat == lasresat) then
                lasgrp(i)=k
                cycle nitres_loop
             endif
          enddo
       endif

       if(prnlev>2) write(outu,'(2a)') &
            'Warining: non-titratable residue is selected :', &
            res(j)(1:idleng)
       call wrndie(-5,'<gbsw>','wrong atom selection')
    enddo nitres_loop

    !--- HSP checking
    check_hsp_loop: do i=1,25
       !--- check HSP specification for ND1
       j = gtrmi(comlyn,comlen,'HSP1',0)
       if(j /= 0) then
          cnthis=cnthis+1
          tmphis(cnthis)=j
          cycle check_hsp_loop
       endif
       !--- check HSP specification for NE2
       k = gtrmi(comlyn,comlen,'HSP2',0)
       if(k /= 0) then
          cnthis=cnthis+1
          tmphis(cnthis)=k
          cycle check_hsp_loop
       endif
       exit check_hsp_loop
    enddo check_hsp_loop

    nitres_loop2: do i=1,ntitres
       j=titres(i)
       k=titgrp(i)
       chkhis(i)=-1
       firgrpat=igpbs(k)+1
       lasgrpat=igpbs(k+1)
       if(res(j)(1:4) == 'HSP ') then
          do l=firgrpat,lasgrpat
             do m=1,cnthis
                if(l == tmphis(m)) then
                   chkhis(i)=l
                   cycle nitres_loop2
                endif
             enddo
          enddo
       endif
    enddo nitres_loop2

    ! do i=1,ntitres
    !    j=titres(i)
    !    k=titgrp(i)
    !    firresat=ibase(j)+1
    !    lasresat=ibase(j+1)
    !    firgrpat=igpbs(k)+1
    !    lasgrpat=igpbs(k+1)
    !    write(*,*) firresat,lasresat,firgrpat,lasgrpat
    !    write(*,*) igpbs(firgrp(i))+1,igpbs(lasgrp(i)+1), &
    !         firgrpat,lasgrpat
    !    write(*,*) res(j),chkhis(i)
    !    firresat=igpbs(firgrp(i)-1)+1
    !    lasresat=igpbs(lasgrp(i)+2)
    !    do l=firresat,lasresat
    !       write(*,*) l,' ',atype(l)
    !    enddo
    !    write(*,*)
    ! enddo

    return
  end SUBROUTINE CHK_SELECT


  SUBROUTINE GB_SURF1(natom,atype,natim, &
       sw,AMASS, &
       dx,dy,dz, &
       imattr,imtrns,NOROT)
    !------------------------------------------------------------------------
    !     calculate van der Waals or Solven-accessible surface
    !     by taking a first derivative of the switching fucntion
    !
    !     NOTE:
    !     (1) sw = 0.1
    !     (2) non-hydrogen atoms only
    !
  use new_timer,only:timer_start,timer_stop,     & 
     t_3extra          

  use consta
  use number
#if KEY_PARALLEL==1
  use parallel
#endif 
    implicit none

    integer :: natom,natim
    integer :: imattr(*)
    real(chm_real)  dx(*),dy(*),dz(*)
    real(chm_real)  sw,AMASS(*)
    real(chm_real)  imtrns(*)
    logical NOROT
    character(len=*) :: atype(*)

    !--- local
    integer :: aa,bb,ii,jj,nsrad,cc
    parameter (nsrad=3)
    integer :: ix,iy,iz,ipi,in,nfil,nalst
    real(chm_real)  xa,ya,za,xb,yb,zb,xp,yp,zp
    real(chm_real)  drxgp,drygp,drzgp
    real(chm_real)  dxpb,dypb,dzpb
    real(chm_real)  radia,radib,dr,r,r2,rp,rsw1,rsw2
    real(chm_real)  surfsum
    real(chm_real)  sf,dsfra,dsfrb,dsfb,sfprod,sfprod0
    real(chm_real)  prefac1,prefac2,prefac3,prefac4,prefac5,prefac6
    real(chm_real)  prefac1m,prefac2m,prefac3m,prefac4m
    real(chm_real)  prefac0,prefac7
    real(chm_real)  xgp0,ygp0,zgp0,wang0
    real(chm_real)  rmin,rmax,srgp(nsrad),swrad(nsrad)
    real(chm_real)  fourpi
    real(chm_real)  tranx,trany,tranz,onedgp
    real(chm_real)  zmemb0,zmemb1,zmemb2,sfm,dsfm
    real(chm_real)  Sdxaa,Sdyaa,Sdzaa
    real(chm_real)  prefac6x,prefac6y,prefac6z
    real(chm_real)  GBdxaa2,GBdyaa2,GBdzaa2
    real(chm_real)  sflimit
    integer ::itrans,ipt

    !--- forces due to roation of integration points
    real(chm_real)  FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz

    !--- constants & pre-factors
    sflimit=1.0D-8

    fourpi=4.0D0*pi
    prefac0=fourpi*sgamma

    !--- prefac1 for r      term in switching function
    !--- prefac2 for r**3   term in switching function
    !--- prefac3 for const. term in first derivatives of switching function
    !--- prefac4 for r**2   term in first derivatives of switching function
    prefac1=3.0D0/4.0D0/sw
    prefac2=1.0D0/4.0D0/(sw*sw*sw)
    prefac3=3.0D0/4.0D0/sw
    prefac4=3.0D0/4.0D0/(sw*sw*sw)

    !--- for lookup grid table
    onedgp=1.0D0/dgp
    !--- we need last +0.5D0*dgp to  use sqrt(3.0)/2.0 in LOOKUP table building
    tranx=0.5D0*(nxgp-1)*dgp-xgcen+0.5D0*dgp
    trany=0.5D0*(nygp-1)*dgp-ygcen+0.5D0*dgp
    tranz=0.5D0*(nzgp-1)*dgp-zgcen+0.5D0*dgp

    !--- membrane
    zmemb0=0.0D0
    zmemb1=0.0D0
    zmemb2=0.0D0
    if(tmembgb > 0.0D0) then
       zmemb0=0.5D0*tmembgb
       zmemb1=zmemb0-msw
       zmemb2=zmemb0+msw
       !--- prefac1m for r      term in membrane switching function
       !--- prefac2m for r**3   term in membrane switching function
       !--- prefac3m for const. term in first derivatives of membrane switching function
       !--- prefac4m for r**2   term in first derivatives of membrane switching function
       prefac1m=3.0D0/4.0D0/msw
       prefac2m=1.0D0/4.0D0/(msw*msw*msw)
       prefac3m=3.0D0/4.0D0/msw
       prefac4m=3.0D0/4.0D0/(msw*msw*msw)
    endif

    !--- get the radial integration points and weights using Gauss-Legendre quadrature
    rmin=0.0
    rmax=2.0D0*sw

    call gauleg(rmin,rmax,srgp,swrad,nsrad,0)

    !--- calculate surface and its first derivatives
    gbsurf=zero
    FSxx=zero
    FSxy=zero
    FSxz=zero
    FSyx=zero
    FSyy=zero
    FSyz=zero
    FSzx=zero
    FSzy=zero
    FSzz=zero

#if KEY_PARALLEL==1
    natom_loop: do aa=mynodp,natom,numnod
#else /**/
    natom_loop: do aa=1,natom
#endif 
       radia=radius(aa)+rsasa
       if(atype(aa)(1:1) == 'H') cycle natom_loop
       if(radius(aa) == 0.0)    cycle natom_loop

       xa=gbx(aa)
       ya=gby(aa)
       za=gbz(aa)
       rmin=radia-sw
       !wi         rmax=radia+sw

       !wi         call gauleg(rmin,rmax,srgp,swrad,nsrad,0)

       surfsum=0.0D0
       Sdxaa=0.0D0
       Sdyaa=0.0D0
       Sdzaa=0.0D0

       radial_loop: do ii=1,nsrad      !! Radial Part
          dr=srgp(ii)+rmin
          r=dr-radia
          !--- switching function H'  with respect to vec r
          dsfra= prefac3-prefac4*r*r

          prefac5=swrad(ii)*dr*dr

          angular_loop: do jj=1,nang   !! Angular Part

             sfprod=1.0D0

             drzgp=dr*zgp(jj)
             zp=drzgp+za

             if(tmembgb > 0.0D0) then
                sfm=1.0D0
                if(abs(zp) < zmemb2) then
                   !--- inside impermeable membrane
                   if(abs(zp).le.zmemb1) then
                      sfm=0.0D0
                   else
                      if(zp > 0.0D0) then
                         r=zp-zmemb0
                         sfm=0.5D0+prefac1m*r-prefac2m*r*r*r
                      else
                         r=zp+zmemb0
                         sfm=0.5D0-prefac1m*r+prefac2m*r*r*r
                      endif
                   endif
                endif
                if(sfm < sflimit) then
                   sfprod=0.0D0
                   cycle angular_loop
                endif
                sfprod=sfm
             endif

             drxgp=dr*xgp(jj)
             drygp=dr*ygp(jj)
             xp=drxgp+xa
             yp=drygp+ya
             wang0=wang(jj)

             ix=int((xp+tranx)*onedgp)+1
             iy=int((yp+trany)*onedgp)+1
             iz=int((zp+tranz)*onedgp)+1

             nfil = natomingp(ix,iy,iz)
             if(nfil == 0) cycle angular_loop

             ipi = gptoalst(ix,iy,iz)
             nalst=startalst(ipi)

             nfil_loop: do in=1,nfil
                bb=gblookup(nalst+in)
                if(bb == aa) cycle nfil_loop
                ! if(atype(bb)(1:1) == 'H') cycle nfil_loop
                ! if(radius(bb) == 0.0) cycle nfil_loop

                xb=gbx(bb)
                yb=gby(bb)
                zb=gbz(bb)
                radib=radius(bb)+rsasa

                dxpb=xp-xb
                dypb=yp-yb
                dzpb=zp-zb
                r2=dxpb*dxpb+dypb*dypb+dzpb*dzpb

                rsw2=radib+sw
                rsw2=rsw2*rsw2
                !--- if the integration point is outside an atom
                if(r2.ge.rsw2) cycle nfil_loop

                rsw1=radib-sw
                rsw1=rsw1*rsw1
                !--- if the integration point is inside an atom
                if(r2.le.rsw1) then
                   sfprod=0.0D0
                   cycle angular_loop
                endif

                r=sqrt(r2)-radib
                sf=0.5D0+prefac1*r-prefac2*r*r*r ! switching function H
                sfprod=sfprod*sf
                if(sfprod < sflimit) then
                   sfprod=0.0D0
                   cycle angular_loop
                endif
             enddo nfil_loop

             sfprod0=sfprod*wang0*prefac5*prefac0

             nfil_loop2: do in=1,nfil
                bb=gblookup(nalst+in)
                if(bb == aa) cycle nfil_loop2
                ! if(atype(bb)(1:1) == 'H') cycle nfil_loop2
                ! if(radius(bb) == 0.0) cycle nfil_loop2
                xb=gbx(bb)
                yb=gby(bb)
                zb=gbz(bb)
                radib=radius(bb)+rsasa
                rsw1=radib-sw
                rsw1=rsw1*rsw1
                rsw2=radib+sw
                rsw2=rsw2*rsw2

                r2=(xp-xb)*(xp-xb)+(yp-yb)*(yp-yb)+(zp-zb)*(zp-zb)

                !--- if the integration point is outside an atom
                if(r2.ge.rsw2) cycle nfil_loop2
                !--- if the integration point is inside an atom
                if(r2.le.rsw1) cycle nfil_loop2

                rp=sqrt(r2)
                r=rp-radib
                sf=0.5D0+prefac1*r-prefac2*r*r*r  ! switching function H
                !--- switching function H'  with respect to vec r
                dsfrb=prefac3-prefac4*r*r

                prefac7=dsfra*sfprod0/(sf*rp)
                prefac6=dsfrb*prefac7

                !wi (xp-xb) is for x-axis unit vector to the integration point
                prefac6x=prefac6*(xp-xb)
                prefac6y=prefac6*(yp-yb)
                prefac6z=prefac6*(zp-zb)

                Sdxaa=Sdxaa+prefac6x
                Sdyaa=Sdyaa+prefac6y
                Sdzaa=Sdzaa+prefac6z

                !--- matrix of force * integration points for the
                !---     rotational invariance correction
                IF(Qrotinv) THEN
                   FSxx=FSxx+prefac6x*drxgp
                   FSxy=FSxy+prefac6x*drygp
                   FSxz=FSxz+prefac6x*drzgp
                   FSyx=FSyx+prefac6y*drxgp
                   FSyy=FSyy+prefac6y*drygp
                   FSyz=FSyz+prefac6y*drzgp
                   FSzx=FSzx+prefac6z*drxgp
                   FSzy=FSzy+prefac6z*drygp
                   FSzz=FSzz+prefac6z*drzgp
                ENDIF

                if(bb > natom) then   ! there is image atom
                   if(bb > natim) then
                      cc=gbimattr(bb-natim)
                   else
                      cc=imattr(bb) !get the corresponding primary atom index
                   endif

                   if(NOROT) then
                      dx(cc)=dx(cc)-prefac6x
                      dy(cc)=dy(cc)-prefac6y
                      dz(cc)=dz(cc)-prefac6z
                   else
                      itrans=gbitrn(bb)
                      ipt=(itrans-1)*12
                      GBdxaa2=prefac6x*imtrns(ipt+1)+ &
                           prefac6y*imtrns(ipt+4)+ &
                           prefac6z*imtrns(ipt+7)
                      GBdyaa2=prefac6x*imtrns(ipt+2)+ &
                           prefac6y*imtrns(ipt+5)+ &
                           prefac6z*imtrns(ipt+8)
                      GBdzaa2=prefac6x*imtrns(ipt+3)+ &
                           prefac6y*imtrns(ipt+6)+ &
                           prefac6z*imtrns(ipt+9)
                      dx(cc)=dx(cc)-GBdxaa2
                      dy(cc)=dy(cc)-GBdyaa2
                      dz(cc)=dz(cc)-GBdzaa2
                   endif
                else
                   dx(bb)=dx(bb)-prefac6x
                   dy(bb)=dy(bb)-prefac6y
                   dz(bb)=dz(bb)-prefac6z
                endif
             enddo nfil_loop2

             if(tmembgb > 0.0D0) then
                if(abs(zp) < zmemb2) then ! in membrane switching region
                   if(zp > 0.0D0) then
                      r=zp-zmemb0
                      !--- switching function H' with respect to vec r
                      dsfm= prefac3m-prefac4m*r*r
                   else
                      r=zp+zmemb0
                      !--- switching function H' with respect to vec r
                      dsfm=-prefac3m+prefac4m*r*r
                   endif
                   Sdzaa=Sdzaa+dsfra*dsfm*sfprod0/sfm
                endif
             endif

             surfsum=surfsum+dsfra*sfprod*wang0*prefac5

          enddo angular_loop
       enddo radial_loop

       surfsum=fourpi*surfsum

       ! write(6,'(i10,2f12.5)') aa,surfsum, &
       !      4.0*pi*radia*radia*(1+sw*sw/5.0D0/(radia*radia))

       gbsurf=gbsurf+surfsum

       dx(aa)=dx(aa)+Sdxaa
       dy(aa)=dy(aa)+Sdyaa
       dz(aa)=dz(aa)+Sdzaa

    enddo natom_loop

    IF(.not.Qrotinv) then
       ! call timer_stop(t_3extra)
       return
    endif

    !---  force contributions due to rotation of integration points
    CALL GB_ROTINVAR2(NATOM,gbX,gbY,gbZ,AMASS,DX,DY,DZ, &
         FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz)

    !
    return
  end SUBROUTINE GB_SURF1


  SUBROUTINE GB_SOLV1(natom,natim,cg,AMASS, &
       gbelec,dx,dy,dz, &
       JNBL,INBL,INBL14,INBLO14, &
       ntrans,imtrns,iminv,NOROT, &
       IMATPT,IMATTR,IMJNBL,IMINBL, &
       NIMNBSL,IMJNBSL,IMINBSL)
    !---------------------------------------------------------------------------
    !---     calculate Born radii and GB solvation energy and forces
    !---

  use new_timer,only:timer_start,timer_stop,     & 
     T_gb_radii,T_gb_energ,T_gb_force,t_1extra          

  use consta
  use number
  use inbnd
#if KEY_PARALLEL==1
  use parallel
#endif 

    integer ::natom,natim
    integer ::ntrans,iminv(*),imatpt(*), &
         imattr(*),ImJNBL(*),ImINBL(*)
    integer ::NImNBSL,ImJNBSL(*),ImINBSL(*)
    real(chm_real)  cg(*)
    real(chm_real)  AMASS(*)
    real(chm_real)  gbelec
    real(chm_real)  dx(*),dy(*),dz(*)
    real(chm_real)  imtrns(*)
    logical NOROT

    !--- local
    integer ::aa,bb,ii,jj
    integer ::ix,iy,iz,ipi,in,nfil,nalst
    integer ::cip,cip_aa,cip_rad,nradang,cmt
    integer(kind=int_byte) checkip0,chkgvdw0,checkip0gb
    real(chm_real)  xa,ya,za,xb,yb,zb,xp,yp,zp
    real(chm_real)  drxgp,drygp,drzgp
    real(chm_real)  dxab,dyab,dzab,dxpb,dypb,dzpb
    real(chm_real)  radib,dr,dr2,r,r2,rsw1,rsw2
    real(chm_real)  vsum,vsum2,angsum,angsum2
    real(chm_real)  sf,dsfrb,dsfb,sfprod,recipr,prefac0, &
         sfprodm,sfprodgb
    real(chm_real)  prefac1,prefac2,prefac3,prefac4
    real(chm_real)  prefac5,prefac6,prefac7,prefac8,prefac9,prefac10
    real(chm_real)  prefac1m,prefac2m,prefac3m,prefac4m
    real(chm_real)  prefac9x,prefac9y,prefac9z
    real(chm_real)  GBdxaa,GBdyaa,GBdzaa
    real(chm_real)  tau,Gab,scrfac,effsalt
    real(chm_real)  cga,cgb,cga2,cgab,rBorna,rBornb,rborna2,rbornb2
    real(chm_real)  r2ab,r2born,expfac,rGB,rGB2
    real(chm_real)  rp,dxang,dyang,dzang,wang0,wrad0
    real(chm_real)  dx_aa,dy_aa,dz_aa
    real(chm_real)  tranx,trany,tranz,onedgp
    real(chm_real)  r0,x0,y0,z0
    real(chm_real)  xgpmax,xgpmin,ygpmax,ygpmin,zgpmax,zgpmin
    real(chm_real)  dGdRGBaa,dRGBdrx,dRGBdry,dRGBdrz
    real(chm_real)  zmemb0,zmemb1,zmemb2,sfm,dsfm
    real(chm_real)  rcyln1,rcyln2
    real(chm_real)  sflimit,fourpi
    real(chm_real)  rmin2,vs2,vs3,dr3
    real(chm_real)  aavdwr,aavdweps,aasigma2,aarmin2,aaelj,dr4,dr10, &
         aabb0
    real(chm_real) tmp, tmpe, GAp, GAm, Psi, dPsi !temporary variables for GC/TM Voltage
    logical Qprint

    !--- switching function for electrostatic energy
    integer ::JNbl(*),INbl(*),Inbl14(*),INblo14(*)
    Integer ::k,ITemp,NPr,jpr
    real(chm_real)  C2OfNb,C2OnNb,Rul3,Rul12,RijL,RijU,FSw,DFSw
    logical Switch,LOuter

    !--- image atoms
    integer ::itrans,jtrans,istrt,iend,ipt,itemp0,cc
    real(chm_real)  GBdxaa2,GBdyaa2,GBdzaa2

    !--- forces due to roation of integration points
    real(chm_real)  FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz

    !--- constants & pre-factors

    sflimit = TENM8
    fourpi  = four*pi

    tau   = ( 1/epsp-1/epsw ) * ccelec
    if( kappa > ZERO ) scrfac = ONE / kappa

    !--- prefac1 for r      term in switching function
    !--- prefac2 for r**3   term in switching function
    !--- prefac3 for const. term in first derivatives of switching function
    !--- prefac4 for r**2   term in first derivatives of switching function
    prefac1 = THREE/FOUR /sw
    prefac2 = ONE  /FOUR /(sw*sw*sw)
    prefac3 = THREE/FOUR /sw
    prefac4 = THREE/FOUR /(sw*sw*sw)

    nradang = nrad * nang

    !--- for lookup grid table
    onedgp=ONE/dgp
    !--- we need last +0.5D0*dgp to use sqrt(3.0)/2.0 in LOOKUP table building
    tranx= HALF*(nxgp-1)*dgp-xgcen+ HALF*dgp
    trany= HALF*(nygp-1)*dgp-ygcen+ HALF*dgp
    tranz= HALF*(nzgp-1)*dgp-zgcen+ HALF*dgp

    !--- determine the dimension of the molecule
    xgpmax=-99999.0D0
    ygpmax=-99999.0D0
    zgpmax=-99999.0D0
    xgpmin= 99999.0D0
    ygpmin= 99999.0D0
    zgpmin= 99999.0D0
    !wi      do aa=1,natom
    do aa=1,natgb
       r0=radius(aa)+sw
       x0=gbx(aa)+r0
       if(x0 > xgpmax) xgpmax=x0
       x0=gbx(aa)-r0
       if(x0 < xgpmin) xgpmin=x0
       y0=gby(aa)+r0
       if(y0 > ygpmax) ygpmax=y0
       y0=gby(aa)-r0
       if(y0 < ygpmin) ygpmin=y0
       z0=gbz(aa)+r0
       if(z0 > zgpmax) zgpmax=z0
       z0=gbz(aa)-r0
       if(z0 < zgpmin) zgpmin=z0
    enddo

    !--- for membrane

    zmemb0 = ZERO
    zmemb1 = ZERO
    zmemb2 = ZERO
    if(tmembgb > ZERO) then
       zmemb0 = HALF*tmembgb
       zmemb1 = zmemb0-msw
       zmemb2 = zmemb0+msw
       !--- prefac1m for r      term in membrane switching function
       !--- prefac2m for r**3   term in membrane switching function
       !--- prefac3m for const. term in first derivatives of membrane switching function
       !--- prefac4m for r**2   term in first derivatives of membrane switching function
       prefac1m = THREE/FOUR /msw
       prefac2m = ONE  /FOUR /(msw*msw*msw)
       prefac3m = THREE/FOUR /msw
       prefac4m = THREE/FOUR /(msw*msw*msw)
       if(rcylngb  >  ZERO) then
          rcyln1=rcylngb-sw
          rcyln2=rcylngb+sw
       endif
    endif

    !--- initialize everything here

    gbelec=ZERO
    gvdw  =ZERO


    if( aa0  ==  ZERO .and. aa1  ==  ZERO ) return

    if(mod(IGBSWcall,igbfrq) == 0) then

       do aa=1,natom
          GBdx(aa)=ZERO
          GBdy(aa)=ZERO
          GBdz(aa)=ZERO
          dGdRGB(aa)=ZERO
          rborn(aa)=ZERO 
          alph_gb(aa)=ZERO
       enddo

       ipforce=0
       cmt=0

       !---
       !---  here, we calculate GB radii
       !---  ===========================
       !      call timer_stop(T_1extra)
       call timer_start(T_GB_RADII)                 

#if KEY_PARALLEL==1
       natom_loop: do aa=mynodp,natom,numnod
#else /**/
       natom_loop: do aa=1,natom
#endif 
          cga=cg(aa)
          xa=gbx(aa)
          ya=gby(aa)
          za=gbz(aa)
          r0=radius(aa)-sw

          vsum =ZERO
          vsum2=ZERO
          cip_aa=(aa-1)*nradang

          if(QGvdW) then
             aavdwr  =gvdwR(aa)
             aavdweps=gvdweps(aa)
             aarmin2 =aavdwr**(ONE/THREE)
             aasigma2=aarmin2/TWO**(ONE/THREE)
             aaelj   =ZERO
             chkgvdw(aa)=1
          endif

          Radial_loop: do ii=1,nrad       ! Radial Part
             dr=rgp(ii)
             cip_rad=(ii-1)*nang+cip_aa

             Radial_check: if(r0 > dr) then
                angsum =ONE
                angsum2=ONE
                do jj=1,nang
                   cip=cip_rad+jj
                   checkipvdw(cip)=0
                   checkipgb(cip)=0
                enddo
             else Radial_check

                angsum =ZERO
                angsum2=ZERO

                Angular_loop: do jj=1,nang ! Angular Part
                   sfprod=ONE
                   cip=cip_rad+jj
                   checkip0=1

                   zp=dr*zgp(jj)+za

                   if(tmembgb > ZERO.and..not.QGvdW) then
                      sfm=ONE
                      if(abs(zp) < zmemb2) then
                         if(abs(zp).le.zmemb1) then ! inside impermeable membrane
                            sfm=ZERO

                       !wic ----- cylindrical pore ----
                                              if(rcylngb > zero) then
                                                 xp=dr*xgp(jj)+xa
                                                 yp=dr*ygp(jj)+ya
                                                 r2=sqrt(xp*xp+yp*yp)
                                                 if(r2 < rcyln2) then
                      !wic inside aqueous cylindrical pore
                                                    if(r2.le.rcyln1) then
                                                       sfm=one
                                                    else
                                                       r=r2-rcylngb
                                                       sfm=0.5D0-prefac1*r+prefac2*r*r*r
                                                    endif
                                                 endif
                                              endif
                      !wic ----- cylindrical pore ----

                        else
                            if(zp > ZERO) then
                               r=zp-zmemb0
                               sfm=HALF+prefac1m*r-prefac2m*r*r*r
                            else
                               r=zp+zmemb0
                               sfm=HALF-prefac1m*r+prefac2m*r*r*r
                            endif
                       !wic ----- cylindrical pore ----
                                              if(rcylngb > zero) then
                                                 xp=dr*xgp(jj)+xa
                                                 yp=dr*ygp(jj)+ya
                                                 r2=sqrt(xp*xp+yp*yp)
                                                 if(r2 < rcyln2) then
                      !wic inside aqueous cylindrical pore
                                                    if(r2.le.rcyln1) then
                                                       sfm=one
                                                    else
                                                       r=r2-rcylngb
                                                       sf=half-prefac1*r+prefac2*r*r*r
                                                       sfm=(sf+sfm)
                                                       if(sfm > one) sfm=one
                                                    endif
                                                 endif
                                              endif
                      !wic ----- cylindrical pore ----

                        endif
                      endif
                      if(sfm < sflimit) then
                         sfprod=ZERO
                         checkip0=0
                         goto 89
                      endif
                      sfprod=sfm
                   endif

                   Zrange: if((zp >= zgpmin) .and. (zp <= zgpmax)) then

                      xp=dr*xgp(jj)+xa
                      Xrange: if((xp >= xgpmin) .and. (xp <= xgpmax)) then

                         yp=dr*ygp(jj)+ya
                         Yrange: if((yp >= ygpmin) .and. (yp <= ygpmax)) then

                            ix=int((xp+tranx)*onedgp)+1
                            iy=int((yp+trany)*onedgp)+1
                            iz=int((zp+tranz)*onedgp)+1

                            nfil = natomingp(ix,iy,iz)
                            if (nfil > 0) then
                               ipi = gptoalst(ix,iy,iz)
                               nalst=startalst(ipi)
                            endif

                            Nfil_loop: do in=1,nfil
                               bb=gblookup(nalst+in)
                               radib=radius(bb)
                               xb=gbx(bb)
                               yb=gby(bb)
                               zb=gbz(bb)

                               dxpb=xp-xb
                               dypb=yp-yb
                               dzpb=zp-zb
                               r2=dxpb*dxpb+dypb*dypb+dzpb*dzpb

                               rsw2=radib+sw
                               rsw2=rsw2*rsw2
                               !--- if the integration point is outside an atom
                               if(r2.ge.rsw2) cycle nfil_loop

                               rsw1=radib-sw
                               rsw1=rsw1*rsw1
                               !--- if the integration point is inside an atom
                               if(r2.le.rsw1) then
                                  sfprod=ZERO
                                  checkip0=0
                                  exit nfil_loop
                               endif

                               r=sqrt(r2)-radib
                               sf=HALF+prefac1*r-prefac2*r*r*r
                               sfprod=sfprod*sf
                               if(sfprod < sflimit) then
                                  sfprod=ZERO
                                  checkip0=0
                                  exit nfil_loop
                               endif
                            enddo nfil_loop
                         endif Yrange
                      endif Xrange
                   endif Zrange
                   if(sfprod == ONE) checkip0=0

89                 continue

                   checkip0gb=checkip0
                   sfprodgb=sfprod
                   if(tmembgb > ZERO.and.QGvdW) then
                      if(sfprod /= ZERO) then
                         sfm=ONE
                         if(abs(zp) < zmemb2) then
                            if(abs(zp).le.zmemb1) then !inside impermeable membrane
                               sfm=ZERO
                            else
                               if(zp > ZERO) then
                                  r=zp-zmemb0
                                  sfm=HALF+prefac1m*r-prefac2m*r*r*r
                               else
                                  r=zp+zmemb0
                                  sfm=HALF-prefac1m*r+prefac2m*r*r*r
                               endif
                            endif
                         endif
                         sfprodgb=sfm*sfprodgb
                         if(sfprodgb < sflimit) then
                            sfprodgb=ZERO
                            checkip0gb=0
                         elseif(sfprodgb < ONE) then
                            checkip0gb=1
                         endif
                      endif
                   endif

                   angsum=angsum+wang(jj)*(ONE-sfprod)
                   angsum2=angsum2+wang(jj)*(ONE-sfprodgb)

                   checkipvdw(cip)=checkip0
                   checkipgb(cip)=checkip0gb

                   if (checkip0 == 1 .or. checkip0gb == 1) then
                      ! save sfprod for force
                      ipforce = ipforce + 1
                      call ensure_sfprod_storage(ipforce)
                      sfprod0vdw(ipforce) = sfprod
                      sfprod0gb(ipforce) = sfprodgb
                   endif

                enddo Angular_loop
             endif Radial_check

             dr2=dr*dr
             dr3=dr2*dr
             wrad0=wrad(ii)
             vsum =vsum +angsum2*wrad0/dr2 ! vsum *4.0D0*pi
             vsum2=vsum2+angsum2*wrad0/(dr2*dr3) ! vsum2*4.0D0*pi

             if(QGvdW.and.dr2 > aasigma2) then
                if(dr2 > aarmin2) then
                   dr4  =dr2*dr2
                   dr10 =dr4*dr4*dr2
                   aaelj=aaelj+aavdweps*aavdwr*(aavdwr/dr10-TWO/dr4)* &
                        fourpi*angsum*wrad0
                else
                   aaelj=aaelj-aavdweps*dr2*fourpi*angsum*wrad0
                endif
             endif
          enddo Radial_loop

          rmin2=rminint*rminint
          vs2=(0.25D0/(rmin2*rmin2)-vsum2)**(0.25D0)
          recipr=aa0*(ONE/rminint-vsum)+aa1*vs2           ! vsum/4.0D0/pi
          dvsum(aa)=0.25D0/(vs2*vs2*vs2)                 ! here is 1/4
          rborn(aa)=ONE/recipr

          if(QGvdW) then
             aaelj=bb0(aa)*(gvdwaa(aa)-aaelj)+bb1(aa)
             if(aaelj > ZERO) then
                chkgvdw(aa)=0
                aaelj=ZERO
             endif
             gvdw=gvdw+aaelj
          endif

       enddo natom_loop

#if KEY_PARALLEL==1
       CALL GCOMB(rborn,natom)
#endif 

       call timer_stop(T_GB_RADII)                  

    endif

    if (Qradii) then
       do aa = 1,natom
          alph_gb(aa) = rborn(aa)
       enddo
    endif
 
    if (Qhybrid) return ! Hybrid CPHMD

    !---
    !--- here, we calculate GB energy and force components;
    !---      dG/d(r_ab)*d(r_ab)/dr_a and dG/dR_a
    !=============================================================
    !--- Note : This is done by three parts;
    !---        1. excluded atom lists
    !---        2. nonbonded atom lists (primary and image atoms)
    !---        3. self (aa=aa)

    call timer_start(T_GB_ENERG)                 

    Switch = LCons .and. .not. LShft .and. .not. LFSwt


    !--- 1. excluded atom lists
    !--- ======================

    ITemp = 0

#if KEY_PARALLEL==1
    do aa=mynodp,natom,numnod
#else /**/
    do aa=1,natom
#endif 

       if (aa  /=  1) ITemp = INblo14(aa - 1)
       NPr = INblo14(aa) - ITemp

       cga=cg(aa)
       if(cga /= ZERO) then
          xa=gbx(aa)
          ya=gby(aa)
          za=gbz(aa)
          rBorna=rBorn(aa)

          GBdxaa=ZERO
          GBdyaa=ZERO
          GBdzaa=ZERO
          dGdrGBaa=ZERO

          do jpr = 1, NPr
             k = Inbl14(Itemp+jpr)
             bb = Abs(k)

             if (k  >  0) then

                cgb= cg(bb)
                if(cgb /= ZERO) then
                   xb=gbx(bb)
                   yb=gby(bb)
                   zb=gbz(bb)
                   rBornb=rBorn(bb)

                   dxab=xb-xa
                   dyab=yb-ya
                   dzab=zb-za
                   r2ab=dxab*dxab+dyab*dyab+dzab*dzab
                   r2born=rBorna*rBornb

                   expfac=exp(-r2ab/(4.0D0*r2born))
                   rGB2=r2ab+r2born*expfac
                   rGB=sqrt(rGB2)

                   if(kappa > ZERO) then
                      effsalt=exp(-rGB*scrfac)/epsw
                      tau=( 1/epsp-effsalt ) * ccelec
                      cgab=cga*cgb
                      gbelec=gbelec-cgab*tau/rGB   ! cross term -> factor 1

                      prefac8=cgab/rGB2* &
                           (tau/rGB-scrfac*effsalt*ccelec)
                   else
                      cgab=cga*cg(bb)*tau
                      gbelec=gbelec-cgab/rGB        ! cross term -> factor 1

                      prefac8=cgab/(rGB2*rGB)
                   endif

                   prefac9=0.25D0*prefac8*(4.0D0-expfac)
                   prefac9x=prefac9*dxab
                   prefac9y=prefac9*dyab
                   prefac9z=prefac9*dzab
                   DX(bb)=DX(bb)+prefac9x
                   DY(bb)=DY(bb)+prefac9y
                   DZ(bb)=DZ(bb)+prefac9z
                   GBdxaa=GBdxaa+prefac9x    ! accumulate forces acting on aa
                   GBdyaa=GBdyaa+prefac9y
                   GBdzaa=GBdzaa+prefac9z

                   dGdRGBaa=dGdRGBaa+HALF*prefac8*expfac* &
                        (rbornb+r2ab/rborna*0.25D0)
                   dGdRGB(bb)=dGdRGB(bb)+HALF*prefac8*expfac* &
                        (rborna+r2ab/rbornb*0.25D0)
                endif
             endif
          enddo

          DX(aa)=DX(aa)-GBdxaa
          DY(aa)=DY(aa)-GBdyaa
          DZ(aa)=DZ(aa)-GBdzaa
          dGdRGB(aa)=dGdRGB(aa)+dGdrGBaa
       endif
    enddo

    !--- 2. nonbonded atom lists
    !--- =======================

    C2OfNB = CtOfNB * CtOfNB
    FSw    = ONE
    DFSw   = ZERO
    IF (Switch) THEN
       C2OnNb = CtOnNb * CtOnNb
       If (CtOfNb  >  CtOnNb) Then
          Rul3 = ONE / (C2OfNb - C2OnNb)**3
          Rul12= 12.0D0 * Rul3
       Endif
    Endif

    !--- 2.1 for primary atoms
    !--- ---------------------

    do aa=1,natom-1
       cga=cg(aa)

       if (cga /= ZERO) THEN
          xa=gbx(aa)
          ya=gby(aa)
          za=gbz(aa)
          rBorna=rBorn(aa)

          GBdxaa=ZERO
          GBdyaa=ZERO
          GBdzaa=ZERO
          dGdrGBaa=ZERO

#if KEY_IMCUBES==1
          if(lbycbim)then
             Itemp=INBL(aa+natom)
             Npr=INbl(aa)-ITEMP
          else
#endif 
             IF(aa > 1) THEN
                Itemp=INBL(aa-1)
                Npr=INBL(aa)-ITEMP
             ELSE
                Npr=INBL(aa)
                ITEMP=0
             ENDIF
#if KEY_IMCUBES==1
          endif
#endif 

          Do jpr = 1, NPr
             k  = JNbl(ITemp+jpr)
             bb = Abs(k)
             cgb= cg(bb)

             !wi               write(*,*) 'GB_SOLV1nb',aa,bb

             if (cgb /= ZERO) then
                xb=gbx(bb)
                yb=gby(bb)
                zb=gbz(bb)

                dxab=xb-xa
                dyab=yb-ya
                dzab=zb-za
                r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                If (r2ab  <  C2OfNb) Then
                   FSw = ONE
                   DFSw= ZERO
                   If (Switch) Then
                      LOuter = (r2ab  >  C2OnNb)
                      If (LOuter) Then
                         RijL = C2OnNb - r2ab
                         RijU = C2OfNb - r2ab
                         FSw  = RijU * RijU * (RijU-3.0D0*RijL) * Rul3
                         DfSw = RijL * RijU * Rul12 / FSw
                      Endif
                   Endif

                   rBornb=rBorn(bb)
                   r2born=rBorna*rBornb

                   expfac=exp(-r2ab/(4.0D0*r2born))
                   rGB2=r2ab+r2born*expfac
                   rGB=sqrt(rGB2)

                   if(kappa > ZERO) then
                      effsalt=exp(-rGB*scrfac)/epsw
                      tau=( 1/epsp-effsalt ) * ccelec
                      cgab=cga*cgb

                      Gab=-cgab*tau*FSw/rGB           ! cross term -> factor 1
                      gbelec=gbelec+Gab

                      prefac8=cgab*FSw/rGB2* &
                           (tau/rGB-scrfac*effsalt*ccelec)
                   else
                      cgab=cga*cgb*tau

                      Gab=-cgab*FSw/rGB               ! cross term -> factor 1
                      gbelec=gbelec+Gab

                      prefac8=cgab*FSw/(rGB2*rGB)    ! here we put FSw
                   endif

                   !---  here we put DFSw accumulate forces acting on aa
                   prefac9=0.25D0*prefac8*(4.0D0-expfac)+Gab*DFSw
                   prefac9x=prefac9*dxab
                   prefac9y=prefac9*dyab
                   prefac9z=prefac9*dzab
                   DX(bb)=DX(bb)+prefac9x
                   DY(bb)=DY(bb)+prefac9y
                   DZ(bb)=DZ(bb)+prefac9z
                   GBdxaa=GBdxaa+prefac9x
                   GBdyaa=GBdyaa+prefac9y
                   GBdzaa=GBdzaa+prefac9z

                   dGdRGBaa=dGdRGBaa+HALF*prefac8*expfac* &
                          ! it has FSw via prefac8
                        (rbornb+r2ab/rborna*0.25D0)
                   dGdRGB(bb)=dGdRGB(bb)+HALF*prefac8*expfac* &
                        (rborna+r2ab/rbornb*0.25D0)
                endif
             endif
          enddo

          DX(aa)=DX(aa)-GBdxaa
          DY(aa)=DY(aa)-GBdyaa
          DZ(aa)=DZ(aa)-GBdzaa
          dGdRGB(aa)=dGdRGB(aa)+dGdrGBaa
       endif

       !--- I don't know why we need this here, but
       !--- keep this here to be consistent with ENBFS8 subroutine
       !wi         Itemp=INBL(aa)
       !wi
    enddo

    !--- 2.2 for image atoms
    !--- -------------------

    if(natim > natom) then

       itemp0=natom+1
       do itrans=1,ntrans
          jtrans=iminv(itrans)
          istrt=itemp0
          iend=imatpt(itrans)
          ipt=(itrans-1)*12
          itemp0=iend+1

          !wi            write(*,*) 'solv1:',itrans,jtrans,istrt,iend

          do aa=istrt,iend
             cga=cg(aa)

             if (cga /= ZERO) THEN

                xa=gbx(aa)
                ya=gby(aa)
                za=gbz(aa)
                !--- get the corresponding primary atom index
                cc=imattr(aa)
                rBorna=rBorn(cc)

                GBdxaa=ZERO
                GBdyaa=ZERO
                GBdzaa=ZERO
                dGdrGBaa=ZERO

#if KEY_IMCUBES==1
                if(lbycbim)then
                   !--- natim for image nonbonding list
                   ITEMP=ImINBL(aa+natim)
                   Npr=ImINbl(aa)-ITEMP
                else
#endif 
                   IF(aa > 1) THEN
                      ITEMP=ImINBL(aa-1)
                      Npr=ImINBL(aa)-ITEMP
                   ELSE
                      Npr=ImINBL(aa)
                      ITEMP=0
                   ENDIF
#if KEY_IMCUBES==1
                endif
#endif 

                do jpr = 1, NPr
                   k  = ImJNBL(ITemp+jpr)
                   bb = Abs(k)
                   cgb= cg(bb)
                   ! write(*,*) 'GB_SOLV1im',aa,bb

                   if (cgb /= ZERO) then
                      xb=gbx(bb)
                      yb=gby(bb)
                      zb=gbz(bb)

                      dxab=xb-xa
                      dyab=yb-ya
                      dzab=zb-za
                      r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                      If (r2ab  <  C2OfNb) Then
                         FSw = ONE
                         DFSw= ZERO
                         If (Switch) Then
                            LOuter = (r2ab  >  C2OnNb)
                            If (LOuter) Then
                               RijL = C2OnNb - r2ab
                               RijU = C2OfNb - r2ab
                               FSw  = RijU * RijU * &
                                    (RijU-3.0D0*RijL) * Rul3
                               DfSw = RijL * RijU * Rul12 / FSw
                            Endif
                         Endif

                         rBornb=rBorn(bb)
                         r2born=rBorna*rBornb

                         expfac=exp(-r2ab/(4.0D0*r2born))
                         rGB2=r2ab+r2born*expfac
                         rGB=sqrt(rGB2)

                         if(kappa > ZERO) then
                            effsalt=exp(-rGB*scrfac)/epsw
                            tau=( 1/epsp-effsalt ) * ccelec
                            cgab=cga*cgb

                            Gab=-cgab*tau*FSw/rGB    ! cross term -> factor 1
                            gbelec=gbelec+Gab

                            prefac8=cgab*FSw/rGB2* &
                                 (tau/rGB-scrfac*effsalt*ccelec)
                         else
                            cgab=cga*cgb*tau

                            Gab=-cgab*FSw/rGB        ! cross term -> factor 1
                            gbelec=gbelec+Gab

                            prefac8=cgab*FSw/(rGB2*rGB)    ! here we put FSw
                         endif

                         !--- here we put DFSw to accumulate forces acting on aa
                         prefac9=0.25D0*prefac8*(4.0D0-expfac)+ &
                              Gab*DFSw
                         prefac9x=prefac9*dxab
                         prefac9y=prefac9*dyab
                         prefac9z=prefac9*dzab
                         DX(bb)=DX(bb)+prefac9x
                         DY(bb)=DY(bb)+prefac9y
                         DZ(bb)=DZ(bb)+prefac9z
                         if(NOROT) then
                            GBdxaa=GBdxaa+prefac9x
                            GBdyaa=GBdyaa+prefac9y
                            GBdzaa=GBdzaa+prefac9z
                         else
                            GBdxaa2=prefac9x*imtrns(ipt+1)+ &
                                 prefac9y*imtrns(ipt+4)+ &
                                 prefac9z*imtrns(ipt+7)
                            GBdyaa2=prefac9x*imtrns(ipt+2)+ &
                                 prefac9y*imtrns(ipt+5)+ &
                                 prefac9z*imtrns(ipt+8)
                            GBdzaa2=prefac9x*imtrns(ipt+3)+ &
                                 prefac9y*imtrns(ipt+6)+ &
                                 prefac9z*imtrns(ipt+9)
                            GBdxaa=GBdxaa+GBdxaa2
                            GBdyaa=GBdyaa+GBdyaa2
                            GBdzaa=GBdzaa+GBdzaa2
                         endif

                         dGdRGBaa=dGdRGBaa+HALF*prefac8*expfac*  &
                          ! it has FSw via prefac8
                              (rbornb+r2ab/rborna*0.25D0)
                         dGdRGB(bb)=dGdRGB(bb)+HALF*prefac8*expfac* &
                              (rborna+r2ab/rbornb*0.25D0)
                      endif
                   endif
                enddo

                DX(cc)=DX(cc)-GBdxaa
                DY(cc)=DY(cc)-GBdyaa
                DZ(cc)=DZ(cc)-GBdzaa
                dGdRGB(cc)=dGdRGB(cc)+dGdrGBaa
             endif
          enddo
       enddo

       !--- if self-energy terms exist,

       !wi       write(*,*) NImNBSL
       if(NImNBSL > 0) then

          itemp0=natom+1
          do itrans=1,ntrans
             jtrans=iminv(itrans)
             istrt=itemp0
             iend=imatpt(itrans)
             ipt=(itrans-1)*12
             itemp0=iend+1

             !wi            write(*,*) 'solv1:',itrans,jtrans,istrt,iend

             do aa=istrt,iend
                cga=cg(aa)

                if (cga /= ZERO) THEN

                   xa=gbx(aa)
                   ya=gby(aa)
                   za=gbz(aa)
                   !--- get the corresponding primary atom index
                   cc=imattr(aa)
                   rBorna=rBorn(cc)

                   GBdxaa=ZERO
                   GBdyaa=ZERO
                   GBdzaa=ZERO
                   dGdrGBaa=ZERO

#if KEY_IMCUBES==1
                   if(lbycbim)then
                      !--- natim for image nonbonding list
                      ITEMP=ImINBSL(aa+natim)
                      Npr=ImINBSL(aa)-ITEMP
                   else
#endif 
                      IF(aa > 1) THEN
                         ITEMP=ImINBSL(aa-1)
                         Npr=ImINBSL(aa)-ITEMP
                      ELSE
                         Npr=ImINBSL(aa)
                         ITEMP=0
                      ENDIF
#if KEY_IMCUBES==1
                   endif
#endif 

                   do jpr = 1, NPr
                      k  = ImJNBSL(ITemp+jpr)
                      bb = Abs(k)
                      cgb= cg(bb)

                      !wi               write(*,*) 'GB_SOLV1imS',aa,bb

                      if (cgb /= ZERO) then
                         xb=gbx(bb)
                         yb=gby(bb)
                         zb=gbz(bb)

                         dxab=xb-xa
                         dyab=yb-ya
                         dzab=zb-za
                         r2ab=dxab*dxab+dyab*dyab+dzab*dzab

                         If (r2ab  <  C2OfNb) Then
                            FSw = ONE
                            DFSw= ZERO
                            If (Switch) Then
                               LOuter = (r2ab  >  C2OnNb)
                               If (LOuter) Then
                                  RijL = C2OnNb - r2ab
                                  RijU = C2OfNb - r2ab
                                  FSw  = RijU * RijU * &
                                       (RijU-3.0D0*RijL) * Rul3
                                  DfSw = RijL * RijU * Rul12 / FSw
                               Endif
                            Endif

                            rBornb=rBorn(bb)
                            r2born=rBorna*rBornb

                            expfac=exp(-r2ab/(4.0D0*r2born))
                            rGB2=r2ab+r2born*expfac
                            rGB=sqrt(rGB2)

                            if(kappa > ZERO) then
                               effsalt=exp(-rGB*scrfac)/epsw
                               tau=( 1/epsp-effsalt ) * ccelec
!                               cgab=cga*cgb*tau*HALF    ! 0.5 for self-term
                               cgab=cga*cgb*HALF    ! 0.5 for self-term

!                               Gab=-cgab*FSw/rGB         ! cross term -> factor 1
                               Gab=-cgab*tau*FSw/rGB         ! cross term -> factor 1

                               gbelec=gbelec+Gab

                               prefac8=cgab*FSw/rGB2* &
                                    (tau/rGB-scrfac*effsalt*ccelec)
                            else
                               cgab=cga*cgb*tau*HALF    ! 0.5 for self-term

                               Gab=-cgab*FSw/rGB         ! cross term -> factor 1
                               gbelec=gbelec+Gab

                               prefac8=cgab*FSw/(rGB2*rGB)    ! here we put FSw
                            endif

                            !--- here we put DFSw to accumulate forces acting on aa
                            prefac9=0.25D0*prefac8*(4.0D0-expfac)+ &
                                 Gab*DFSw
                            prefac9x=prefac9*dxab
                            prefac9y=prefac9*dyab
                            prefac9z=prefac9*dzab
                            DX(bb)=DX(bb)+prefac9x
                            DY(bb)=DY(bb)+prefac9y
                            DZ(bb)=DZ(bb)+prefac9z
                            if(NOROT) then
                               GBdxaa=GBdxaa+prefac9x
                               GBdyaa=GBdyaa+prefac9y
                               GBdzaa=GBdzaa+prefac9z
                            else
                               GBdxaa2=prefac9x*imtrns(ipt+1)+ &
                                    prefac9y*imtrns(ipt+4)+ &
                                    prefac9z*imtrns(ipt+7)
                               GBdyaa2=prefac9x*imtrns(ipt+2)+ &
                                    prefac9y*imtrns(ipt+5)+ &
                                    prefac9z*imtrns(ipt+8)
                               GBdzaa2=prefac9x*imtrns(ipt+3)+ &
                                    prefac9y*imtrns(ipt+6)+ &
                                    prefac9z*imtrns(ipt+9)
                               GBdxaa=GBdxaa+GBdxaa2
                               GBdyaa=GBdyaa+GBdyaa2
                               GBdzaa=GBdzaa+GBdzaa2
                            endif

                            dGdRGBaa=dGdRGBaa+HALF*prefac8*expfac*   &
                          ! it has FSw via prefac8
                                 (rbornb+r2ab/rborna*0.25D0)
                            dGdRGB(bb)=dGdRGB(bb)+HALF*prefac8*expfac* &
                                 (rborna+r2ab/rbornb*0.25D0)
                         endif
                      endif
                   enddo

                   DX(cc)=DX(cc)-GBdxaa
                   DY(cc)=DY(cc)-GBdyaa
                   DZ(cc)=DZ(cc)-GBdzaa
                   dGdRGB(cc)=dGdRGB(cc)+dGdrGBaa
                endif
             enddo
          enddo
       endif

    endif

    !--- 3. self term
    !--- ============

#if KEY_PARALLEL==1
    do aa=mynodp,natom,numnod
#else /**/
    do aa=1,natom
#endif 
       cga=cg(aa)
       if(cga /= ZERO) then
          rBorna=rBorn(aa)
          if(kappa > ZERO) then
             effsalt=exp(-rBorna*scrfac)/epsw
             tau=( 1/epsp-effsalt ) * ccelec

             cga2=cga*cga

             gbelec = gbelec - HALF*cga2*tau/rBorna  !self term -> factor 1/2
             dGdRGB(aa)=dGdRGB(aa)+HALF*cga2/rborna* &
                  (tau/rborna-scrfac*effsalt*ccelec)
          else
             cga2=cga*cga*tau

             gbelec = gbelec - HALF*cga2/rBorna       !self term -> factor 1/2
             dGdRGB(aa)=dGdRGB(aa)+HALF*cga2/(rborna*rborna)
          endif
       endif
    enddo

    call timer_stop(T_GB_ENERG)                  
    call timer_start(T_GB_FORCE)                 

    if(mod(IGBSWcall,igbfrq) == 0) then

#if KEY_PARALLEL==1
       CALL GCOMB(dGdRGB,natom)
#endif 

       !---
       !--- here, we calculate forces by calculating dR_a/dr_a
       !--- ==================================================

       cmt=0
       FSxx=zero
       FSxy=zero
       FSxz=zero
       FSyx=zero
       FSyy=zero
       FSyz=zero
       FSzx=zero
       FSzy=zero
       FSzz=zero

#if KEY_PARALLEL==1
       Atom_loop2: do aa=mynodp,natom,numnod
#else /**/
       Atom_loop2: do aa=1,natom
#endif 
          cga=cg(aa)
          !wi         if(cga == 0.0D0.and..not.QGvdW) goto 96
          xa=gbx(aa)
          ya=gby(aa)
          za=gbz(aa)
          rBorna=rBorn(aa)
          rborna2=rborna*rborna
          dGdRGBaa=dGdRGB(aa)
          if(cga == ZERO) dGdRGBaa=ZERO
          GBdxaa=ZERO
          GBdyaa=ZERO
          GBdzaa=ZERO

          if(QGvdW) then
             chkgvdw0=chkgvdw(aa)
             if(chkgvdw0 == 1) then
                aavdwr  =gvdwR(aa)
                aavdweps=gvdweps(aa)
                aarmin2 =aavdwr**(1.0/3.0)
                aasigma2=aarmin2/2.0D0**(1.0/3.0)
                aabb0   =bb0(aa)
             endif
          endif

          cip_aa=(aa-1)*nradang

          Radial_loop2: do ii=1,nrad       ! Radial Part
             dr=rgp(ii)
             dr2=dr*dr
             dr3=dr2*dr
             wrad0=wrad(ii)
             prefac5=dGdRGBaa*rborna2*(aa0/dr2+dvsum(aa)*aa1/(dr2*dr3))* &
                  wrad0

             aaelj=0.0D0
             if(QGvdW) then
                if(chkgvdw0 == 1) then
                   if(dr2 > aasigma2) then
                      if(dr2 > aarmin2) then
                         dr4  =dr2*dr2
                         dr10 =dr4*dr4*dr2
                         aaelj=-aavdweps*aavdwr*(aavdwr/dr10-2.0D0/dr4)* &
                              fourpi*wrad0
                      else
                         aaelj=+aavdweps*dr2*fourpi*wrad0
                      endif
                      aaelj=aaelj*aabb0
                   endif
                endif
             endif

             cip_rad=(ii-1)*nang+cip_aa

             Angular_loop2: do jj=1,nang   ! Angular Part
                cip=cip_rad+jj
                if(checkipgb(cip) == 0.and.checkipvdw(cip) == 0)  &
                     cycle Angular_loop2

                drzgp=dr*zgp(jj)
                drxgp=dr*xgp(jj)
                drygp=dr*ygp(jj)
                xp=drxgp+xa
                yp=drygp+ya
                zp=drzgp+za

                cmt=cmt+1

                if(QGvdW) then
                   sfprodm=sfprod0gb(cmt)*prefac5*wang(jj)
                   sfprod =sfprodm+sfprod0vdw(cmt)*aaelj*wang(jj)
                else
                   sfprod =sfprod0gb(cmt)*prefac5*wang(jj)
                   sfprodm=sfprod
                endif

                if(sfprod == ZERO) cycle Angular_loop2

                if(tmembgb > ZERO) then
                   if(abs(zp) < zmemb2) then ! in membrane switching region
                      if(zp < zgpmin) goto 95
                      if(zp > zgpmax) goto 95
                      if(xp < xgpmin) goto 95
                      if(xp > xgpmax) goto 95
                      if(yp < ygpmin) goto 95
                      if(yp > ygpmax) goto 95
                   endif
                endif

                ix=int((xp+tranx)*onedgp)+1
                iy=int((yp+trany)*onedgp)+1
                iz=int((zp+tranz)*onedgp)+1

                nfil = natomingp(ix,iy,iz)
                if (nfil > 0) then
                   ipi = gptoalst(ix,iy,iz)
                   nalst=startalst(ipi)
                endif

                Nfil_loop2: do in=1,nfil
                   bb=gblookup(nalst+in)
                   if(bb == aa) cycle Nfil_loop2
                   radib=radius(bb)
                   xb=gbx(bb)
                   yb=gby(bb)
                   zb=gbz(bb)

                   dxpb=xp-xb
                   dypb=yp-yb
                   dzpb=zp-zb
                   r2=dxpb*dxpb+dypb*dypb+dzpb*dzpb

                   rsw2=radib+sw
                   rsw2=rsw2*rsw2
                   !--- if the integration point is outside an atom
                   if(r2.ge.rsw2) cycle Nfil_loop2

                   !wi  rsw1=radib-sw
                   !wi  rsw1=rsw1*rsw1
                   !wic if the integration point is inside an atom
                   !wi  if(r2.le.rsw1) cycle Nfil_loop2

                   rp=sqrt(r2)
                   r=rp-radib
                   r2=r*r
                   sf=HALF+prefac1*r-prefac2*r*r2
                   !wi  if(sf < sflimit) cycle Nfil_loop2

                   !--- switching function H'  with respect to vec r
                   dsfrb=prefac3-prefac4*r2

                   prefac6=sfprod/(sf*rp)
                   prefac7=-dsfrb*prefac6
                   dRGBdrx=prefac7*dxpb ! 4pi is cancelled
                   dRGBdry=prefac7*dypb
                   dRGBdrz=prefac7*dzpb
                   GBdxaa=GBdxaa+dRGBdrx
                   GBdyaa=GBdyaa+dRGBdry
                   GBdzaa=GBdzaa+dRGBdrz

                   !--- matrix of force * integration points
                   !--- for the rotational invariance correction
                   IF(Qrotinv) THEN
                      FSxx=FSxx+dRGBdrx*drxgp
                      FSxy=FSxy+dRGBdrx*drygp
                      FSxz=FSxz+dRGBdrx*drzgp
                      FSyx=FSyx+dRGBdry*drxgp
                      FSyy=FSyy+dRGBdry*drygp
                      FSyz=FSyz+dRGBdry*drzgp
                      FSzx=FSzx+dRGBdrz*drxgp
                      FSzy=FSzy+dRGBdrz*drygp
                      FSzz=FSzz+dRGBdrz*drzgp
                   ENDIF

                   if(bb > natom) then        ! there is image atom
                      if(bb > natim) then
                         cc=gbimattr(bb-natim)
                         !wi  write(*,*) bb,cc,natim
                      else
                         !--- get the corresponding primary atom index
                         cc=imattr(bb)
                      endif

                      if(NOROT) then
                         GBdx(cc)=GBdx(cc)-dRGBdrx
                         GBdy(cc)=GBdy(cc)-dRGBdry
                         GBdz(cc)=GBdz(cc)-dRGBdrz
                      else
                         itrans=gbitrn(bb)
                         ipt=(itrans-1)*12
                         GBdxaa2=dRGBdrx*imtrns(ipt+1)+ &
                              dRGBdry*imtrns(ipt+4)+ &
                              dRGBdrz*imtrns(ipt+7)
                         GBdyaa2=dRGBdrx*imtrns(ipt+2)+ &
                              dRGBdry*imtrns(ipt+5)+ &
                              dRGBdrz*imtrns(ipt+8)
                         GBdzaa2=dRGBdrx*imtrns(ipt+3)+ &
                              dRGBdry*imtrns(ipt+6)+ &
                              dRGBdrz*imtrns(ipt+9)
                         GBdx(cc)=GBdx(cc)-GBdxaa2
                         GBdy(cc)=GBdy(cc)-GBdyaa2
                         GBdz(cc)=GBdz(cc)-GBdzaa2
                      endif

                   else
                      GBdx(bb)=GBdx(bb)-dRGBdrx
                      GBdy(bb)=GBdy(bb)-dRGBdry
                      GBdz(bb)=GBdz(bb)-dRGBdrz
                   endif
                enddo Nfil_loop2

95              continue
                if(tmembgb > ZERO.and.checkipgb(cip) == 1) then
                   if(abs(zp) < zmemb2) then ! in membrane switching region
                      if(zp > ZERO) then
                         r=zp-zmemb0
                         !--- switching function SF
                         sfm=HALF+prefac1m*r-prefac2m*r*r*r
                         !--- switching function H' with respect to vec r
                         dsfm= prefac3m-prefac4m*r*r
                      else
                         r=zp+zmemb0
                         sfm=HALF-prefac1m*r+prefac2m*r*r*r
                         !--- switching function H' with respect to vec r
                         dsfm=-prefac3m+prefac4m*r*r
                      endif
                      prefac6=sfprodm/sfm
                      GBdzaa=GBdzaa-dsfm*prefac6
                   endif
                endif

             enddo Angular_loop2
          enddo Radial_loop2

          GBdx(aa)=GBdx(aa)+GBdxaa
          GBdy(aa)=GBdy(aa)+GBdyaa
          GBdz(aa)=GBdz(aa)+GBdzaa

96        continue
       enddo Atom_loop2

    endif

    !--- add (dG/dR_a)(dR_a/dr_a) forces to total forces

    do aa=1,natom
       dx(aa)=dx(aa)+GBdx(aa)
       dy(aa)=dy(aa)+GBdy(aa)
       dz(aa)=dz(aa)+GBdz(aa)
    enddo

!!!cb3 added Gouy Chapman and TM Voltage  - taken from imm1 model/code of Lazaridis
    if(QGouyC) then
       EGouy = zero
       do aa=1, natom
          cga = cg(aa)
          if(cga /= 0)  then
             tmp = abs(gbz(aa)) - tmembgb/two - offst
             if(tmp <= 0) then
                Psi = Psi0*23.06d0
             else
                tmpe = exp(-GKappa*tmp)
                GAp = (one + GAlpha*tmpe)
                GAm = (one - GAlpha*tmpe)
                Psi = TempFac*log((Gap/GAm))
                dz(aa) = dz(aa) + &
                     cga*TempFac*two*GAlpha/(GAm*GAp)* &
                     (-GKappa*sign(one,gbz(aa)))
             endif
             Egouy = Egouy + cga*Psi
          endif
       enddo
       GBElec = GBElec + EGouy
    endif
    if(QVolt) then
       Evolt = zero
       do aa=1, natom
          cga = cg(aa)
          if(cga /= 0)  then
             if(gbz(aa) <= -tmembgb/two) then
                tmp = gbz(aa) + tmembgb/two
                tmpe = exp(GKappa*tmp)
                Psi = Acons * tmpe * 23.06d0
                dPsi = Psi*GKappa
             elseif(gbz(aa) >= tmembgb/two) then
                tmp = gbz(aa) - tmembgb/two
                tmpe = exp(-GKappa*tmp)
                Psi = (Volt - Acons * tmpe) * 23.06d0
                dPsi = GKappa*Acons*tmpe*23.06d0
             else
                tmp = gbz(aa) + tmembgb/two
                Psi = Acons*(40.0d0*GKappa*tmp+One)*23.06d0
                dPsi = Psi*GKappa*40.0d0*23.06d0
             endif
             EVolt = EVolt + cga*Psi
             dz(aa) = dz(aa) + cga*dPsi
          endif
       enddo
       GBElec = GBElec + EVolt
    endif

    IF(.not.Qrotinv) then
       call timer_stop(T_GB_FORCE) 
       return
    endif

    !---  force contributions due to rotation of integration points
    CALL GB_ROTINVAR2(NATOM,gbX,gbY,gbZ,AMASS,DX,DY,DZ, &
         FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz)

    call timer_stop(T_GB_FORCE)                  

    return
  end SUBROUTINE GB_SOLV1


  subroutine ensure_sfprod_storage(reqsize)
    use memory
    implicit none
    integer, intent(in) :: reqsize
    integer :: padsize
    character(len=*), parameter :: PROC = 'ensure_sfprod_storage'

    if (allocated(sfprod0gb)) then
       if (reqsize > size(sfprod0gb)) then
          padsize = 2 * reqsize + 4
          call chmrealloc('gbsw.src',PROC,'sfprod0gb',padsize,crl=sfprod0gb)
          call chmrealloc('gbsw.src',PROC,'sfprod0vdw',padsize,crl=sfprod0vdw)
       ! else current size is OK
       endif
    else
       padsize = 2 * reqsize + 4
       call chmalloc('gbsw.src',PROC,'sfprod0gb',padsize,crl=sfprod0gb)
       call chmalloc('gbsw.src',PROC,'sfprod0vdw',padsize,crl=sfprod0vdw)
    endif
  end subroutine ensure_sfprod_storage


  SUBROUTINE GB_pKa(natom,x,y,z,cg,sw,aa0,aa1, &
       xgp,ygp,zgp,wang,nang, &
       rgp,wrad,nrad,rmin, &
       epsp,epsw,rBORN, &
       nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
       startalst,gblookup,cgmod, &
       igpbs,nictot,nseg,segid,resid,res,atype, &
       ntitres,titres,titgrp,firgrp,lasgrp,titQ,chkhis, &
       Qsrmodel)
    !---------------------------------------------------------------------------
    !---     calculate pK_intr and site-site interaction matrix
    !---

  use consta
  use stream

    integer ::iunit
    integer ::natom,nang,nrad
    integer ::nxgp,nygp,nzgp
    integer ::startalst(*), gblookup(*)
    integer ::igpbs(*),nictot(*),nseg
    integer ::ntitres,titres(*),titgrp(*), &
    firgrp(*),lasgrp(*),chkhis(*)
    real(chm_real)  x(*),y(*),z(*),cg(*)
    real(chm_real)  xgp(*),ygp(*),zgp(*),wang(*)
    real(chm_real)  rgp(*),wrad(*)
    real(chm_real)  sw,rmin,aa0,aa1
    real(chm_real)  epsp,epsw,gbelec
    real(chm_real)  rborn(*)
    real(chm_real)  dgp,xgcen,ygcen,zgcen
    real(chm_real)  titQ(*),cgmod(*)
    character(len=*) :: segid(*),resid(*),res(*),atype(*)
    logical Qsrmodel

    !--- local
    integer ::aa,bb,ii,jj,i,j,k
    integer ::ix,iy,iz,ipi,in,nfil,nalst
    integer ::natingrp,firresat,lasresat
    integer ::firgrpat,lasgrpat,firgrpat2,lasgrpat2
    real(chm_real)  xa,ya,za,xb,yb,zb,xp,yp,zp
    real(chm_real)  radib,dr,dr2,r,r2,rsw1,rsw2,vsum,vsum2,angsum
    real(chm_real)  sf,sfprod,prefac1,prefac2,recipr,prefac0
    real(chm_real)  tau
    real(chm_real)  cga,cgb,cgab,rBorna
    real(chm_real)  rab,r2ab,r2born,expfac,rGB,rGB2
    real(chm_real)  tranx,trany,tranz,onedgp
    real(chm_real)  r0,x0,y0,z0
    real(chm_real)  xgpmax,xgpmin,ygpmax,ygpmin,zgpmax,zgpmin
    real(chm_real)  Q,dQ,Q2,dQ2,cga_u,cga_p,cgb_u,cgb_p,elec_p,elec_u
    real(chm_real)  dGprot_born(100),dGprot_back(100),dG_born
    real(chm_real)  dGmod_born(100),dGmod_back(100),dG_back
    real(chm_real)  Ginter(100,100)
    real(chm_real)  pKa_shift,pKa_intr,pKa_mod,pKa_born,pKa_back
    real(chm_real)  temp,loge,kBT

    !--- constants & pre-factors

    tau   = ( 1/epsp-1/epsw )

    !--- prefac1 : for r    term in switching function
    !--- prefac2 : for r**3 term in switching function
    prefac1=3.0D0/4.0D0/sw
    prefac2=1.0D0/4.0D0/(sw*sw*sw)

    !--- for lookup grid table

    onedgp=1.0D0/dgp
    !--- we need last +0.5D0*dgp to use sqrt(3.0)/2.0 in LOOKUP table building
    tranx=0.5D0*(nxgp-1)*dgp-xgcen+0.5D0*dgp
    trany=0.5D0*(nygp-1)*dgp-ygcen+0.5D0*dgp
    tranz=0.5D0*(nzgp-1)*dgp-zgcen+0.5D0*dgp

    !--- determine the dimension of the molecule
    xgpmax=-99999.0D0
    ygpmax=-99999.0D0
    zgpmax=-99999.0D0
    xgpmin= 99999.0D0
    ygpmin= 99999.0D0
    zgpmin= 99999.0D0
    do aa=1,natom
       r0=radius(aa)+sw
       x0=x(aa)+r0
       if(x0 > xgpmax) xgpmax=x0
       x0=x(aa)-r0
       if(x0 < xgpmin) xgpmin=x0
       y0=y(aa)+r0
       if(y0 > ygpmax) ygpmax=y0
       y0=y(aa)-r0
       if(y0 < ygpmin) ygpmin=y0
       z0=z(aa)+r0
       if(z0 > zgpmax) zgpmax=z0
       z0=z(aa)-r0
       if(z0 < zgpmin) zgpmin=z0
    enddo

    !--- calculate GB radii in PROTEIN
    Atom_loop: do aa=1,natom
       xa=x(aa)
       ya=y(aa)
       za=z(aa)
       vsum =0.0D0
       vsum2=0.0D0

       Radial_loop: do ii=1,nrad       ! Radial Part
          dr=rgp(ii)
          angsum=0.0D0

          Angular_loop: do jj=1,nang   ! Angular Part
             sfprod=1.0D0

             xp=dr*xgp(jj)+xa
             Xrange: if((xp >= xgpmin) .and. (xp <= xgpmax))then
                yp=dr*ygp(jj)+ya
                Yrange: if((yp >= ygpmin) .and. (yp <= ygpmax))then
                   zp=dr*zgp(jj)+za
                   Zrange: if((zp >= zgpmin) .and. (zp <= zgpmax))then

                      ix=int((xp+tranx)*onedgp)+1
                      iy=int((yp+trany)*onedgp)+1
                      iz=int((zp+tranz)*onedgp)+1

                      nfil = natomingp(ix,iy,iz)
                      if (nfil > 0) then
                         ipi = gptoalst(ix,iy,iz)
                         nalst=startalst(ipi)
                      endif

                      Nfil_loop: do in=1,nfil
                         bb=gblookup(nalst+in)
                         radib=radius(bb)
                         xb=x(bb)
                         yb=y(bb)
                         zb=z(bb)
                         rsw1=radib-sw
                         rsw1=rsw1*rsw1
                         rsw2=radib+sw
                         rsw2=rsw2*rsw2

                         r2=(xp-xb)*(xp-xb)+(yp-yb)*(yp-yb)+(zp-zb)*(zp-zb)

                         !--- if the integration point is outside an atom
                         if(r2.ge.rsw2) cycle Nfil_loop
                         !--- if the integration point is inside an atom
                         if(r2.le.rsw1) then
                            sfprod=0.0D0
                            exit Nfil_loop
                         else
                            r=sqrt(r2)-radib
                            sf=0.5D0+prefac1*r-prefac2*r*r*r
                            sfprod=sfprod*sf
                         endif
                      enddo Nfil_loop

                   endif Zrange
                endif Yrange
             endif Xrange
             angsum=angsum+wang(jj)*(1.0D0-sfprod)
          enddo Angular_loop

          dr2=dr*dr
          vsum =vsum +angsum*wrad(ii)/dr2          ! vsum *4.0D0*pi
          vsum2=vsum2+angsum*wrad(ii)/(dr2*dr2*dr) ! vsum2*4.0D0*pi
       enddo Radial_loop

       recipr=aa0*(1.0D0/rmin-vsum)+           & ! vsum/4.0D0/pi
            aa1*(0.25D0/rmin**4-vsum2)**(0.25D0) ! vsum2/4.0D0/pi
       rborn(aa)=1.0D0/recipr
    enddo Atom_loop

    !--- copy cg to cgmod (see below)

    do aa=1,natom
       cgmod(aa)=cg(aa)
    enddo

    !---
    !--- dG_BORN and dG_BACK in protein
    !--- ==============================
    !---
    !--- calculate the electrostatic energy of protonated or
    !---                                            unprotonated states
    !--- for each titratable residue in PROTEIN, while other
    !--- titratable residues selected by user have neutral forms.
    !--- By definition, to get neutral forms we
    !---          add +1/#atomingrp to atoms in titrable groups for GLU and ASP
    !---          add -1/#atomingrp to atoms in titrable groups for LYS and ARG
    !---          add -1/#atomingrp to atoms in titrable groups for HSP
    !---
    !--- if titQ(I) = -1, resid I is GLU and ASP and
    !---                    they are in unprotonated state
    !--- if titQ(I) = +1, resid I is LYS, ARG, and HSP and
    !---                    they are in protonated state
    !---


    do i=1,ntitres

       !--- here modify the charges of other titratable groups to neutral forms
       do ii=1,ntitres
          if(ii /= i) then
             k=titgrp(ii)
             Q=titQ(ii)
             if(chkhis(ii) > 0) then
                dQ=-Q/2.0D0
                aa=chkhis(ii)
                cgmod(aa)=cg(aa)+dQ
                cgmod(aa+1)=cg(aa+1)+dQ
             else
                firgrpat=igpbs(k)+1
                lasgrpat=igpbs(k+1)
                natingrp=lasgrpat-firgrpat+1
                dQ=-Q/natingrp
                do aa=firgrpat,lasgrpat
                   cgmod(aa)=cg(aa)+dQ
                enddo
             endif
          endif
       enddo

       j=titres(i)
       k=titgrp(i)
       Q=titQ(i)

       if(Qsrmodel) then
          !--- first atom number from which a target titratable group
          !---                                     (model compound) is defined
          firresat=igpbs(firgrp(i))+1
          !--- last atom number to which a target titratable group is defined
          lasresat=igpbs(lasgrp(i)+1)
       else
          !--- first atom number from which a target titratable group
          !---                                    (model compound) is defined
          firresat=igpbs(firgrp(i)-1)+1
          !--- last atom number to which a target titratable group is defined
          lasresat=igpbs(lasgrp(i)+2)
       endif

       !--- histidine specifications for HSP1 or HSP2
       if(chkhis(i) > 0) then
          firgrpat=chkhis(i)
          lasgrpat=firgrpat+1
          natingrp=2
       else
          !--- first atom number from which atomic charges are changed upon titration
          firgrpat=igpbs(k)+1
          !--- last atom number to which atomic charges are changed upon titration
          lasgrpat=igpbs(k+1)
          natingrp=lasgrpat-firgrpat+1
       endif
       dQ=-Q/natingrp

       ! write(*,*) firresat,lasresat,firgrpat,lasgrpat,natingrp,Q
       ! do aa=firresat,lasresat
       !    write(*,*) aa, atype(aa), rborn(aa)
       ! enddo

       dGprot_born(i)=0.0D0   ! Born   electrostatic energy change
       dGprot_back(i)=0.0D0   ! static electrostatic energy change

       do aa=firgrpat,lasgrpat   ! loop over titrating group
          xa=x(aa)
          ya=y(aa)
          za=z(aa)
          rBorna=rBorn(aa)

          if(Q == -1.0D0) then
             cga_u=cg(aa)
             cga_p=cg(aa)+dQ
          else
             cga_p=cg(aa)
             cga_u=cg(aa)+dQ
          endif

          !--- self term -> factor 1/2
          dGprot_born(i)=dGprot_born(i) - &
               0.5D0*tau*(cga_p*cga_p-cga_u*cga_u)/rBorna

          !---   BORN contribution

          do bb=aa+1,lasgrpat
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)

             if(Q == -1.0D0) then
                cgb_u=cg(bb)
                cgb_p=cg(bb)+dQ
             else
                cgb_p=cg(bb)
                cgb_u=cg(bb)+dQ
             endif

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGprot_born(i)=dGprot_born(i) - &
                  tau*(cga_p*cgb_p-cga_u*cgb_u)/rGB
          enddo

          !---   BORN contribution

          !--- loop over atom whose charges are not changed upon titration
          do bb=firresat,firgrpat-1
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cg(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGprot_born(i)=dGprot_born(i) - &
                  tau*(cga_p-cga_u)*cgb/rGB
          enddo

          !--- loop over atom whose charges are not changed upon titration
          do bb=lasgrpat+1,lasresat
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cg(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGprot_born(i)=dGprot_born(i) - &
                  tau*(cga_p-cga_u)*cgb/rGB
          enddo

          !---   BACK contribution

          !--- loop over atoms for the rest (here we need Coulombic interactions)
          do bb=1,firresat-1
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cgmod(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             rab=sqrt(r2ab)
             dGprot_back(i)=dGprot_back(i)- &
                  tau*(cga_p-cga_u)*cgb/rGB +  & ! cross term -> factor 1
                  (cga_p-cga_u)*cgb/rab/epsp  ! Coulombic interaction
          enddo

          !--- loop over atoms for the rest (here we need Coulombic interactions)
          do bb=lasresat+1,natom
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cgmod(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             rab=sqrt(r2ab)
             dGprot_back(i)=dGprot_back(i)- &
                  tau*(cga_p-cga_u)*cgb/rGB +  & ! cross term -> factor 1
                  (cga_p-cga_u)*cgb/rab/epsp  ! Coulombic interaction
          enddo
       enddo
    enddo

    !---
    !--- SITE-SITE interaction
    !--- =====================
    !---
    !--- G12 = Gp1p2 - Gu1p2 - Gp1u2 + Gu1u2
    !---

    do i=1,ntitres
       k=titgrp(i)
       Q=titQ(i)

       if(chkhis(i) > 0) then
          firgrpat=chkhis(i)
          lasgrpat=firgrpat+1
          natingrp=2
       else
          !--- first atom number from which atomic charges are changed upon titration
          firgrpat=igpbs(k)+1
          !--- last atom number to which atomic charges are changed upon titration
          lasgrpat=igpbs(k+1)
          natingrp=lasgrpat-firgrpat+1
       endif
       dQ=-Q/natingrp

       Ginter(i,i)=0.0D0

       do j=i+1,ntitres
          k=titgrp(j)
          Q2=titQ(j)

          if(chkhis(j) > 0) then
             firgrpat2=chkhis(j)
             lasgrpat2=firgrpat2+1
             natingrp=2
          else
             !--- first atom number from which atomic charges are changed upon titration
             firgrpat2=igpbs(k)+1
             !--- last atom number to which atomic charges are changed upon titration
             lasgrpat2=igpbs(k+1)
             natingrp=lasgrpat2-firgrpat2+1
          endif
          dQ2=-Q2/natingrp

          Ginter(i,j)=0.0D0

          !--- loop over titrating group
          do aa=firgrpat,lasgrpat
             xa=x(aa)
             ya=y(aa)
             za=z(aa)
             rBorna=rBorn(aa)

             if(Q == -1.0D0) then
                cga_u=cg(aa)
                cga_p=cg(aa)+dQ
             else
                cga_p=cg(aa)
                cga_u=cg(aa)+dQ
             endif

             do bb=firgrpat2,lasgrpat2
                xb=x(bb)
                yb=y(bb)
                zb=z(bb)

                if(Q2 == -1.0D0) then
                   cgb_u=cg(bb)
                   cgb_p=cg(bb)+dQ2
                else
                   cgb_p=cg(bb)
                   cgb_u=cg(bb)+dQ2
                endif

                r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
                r2born=rBorna*rBorn(bb)

                expfac=exp(-r2ab/(4.0D0*r2born))
                rGB=sqrt(r2ab+r2born*expfac)

                rab=sqrt(r2ab)
                Ginter(i,j)=Ginter(i,j)- &
                     tau*(cga_p-cga_u)*(cgb_p-cgb_u)/rGB +  & ! cross term
                     (cga_p-cga_u)*(cgb_p-cgb_u)/rab/epsp  ! Coulombic interaction
             enddo
          enddo

          Ginter(j,i)=Ginter(i,j)
       enddo
    enddo

    !---
    !--- dG_BORN and dG_BACK in model
    !--- ============================
    !---
    !--- calculate BORN radii and the electrostatic energy of
    !---                      protonated or unprotonated states
    !--- for each titratable residue in MODEL which is a selected
    !---                                 titratable residue
    !---
    !--- if titQ(I) = -1, resid I is GLU and ASP and
    !---                              they are in unprotonated state
    !--- if titQ(I) = +1, resid I is LYS, ARG, and HSP and
    !---                              they are in protonated state
    !---
    !--- Since the model is small, we don't use the lookup table here.
    !---

    do i=1,ntitres
       j=titres(i)
       k=titgrp(i)
       Q=titQ(i)
       if(Qsrmodel) then
          firresat=igpbs(firgrp(i))+1
          lasresat=igpbs(lasgrp(i)+1)
       else
          firresat=igpbs(firgrp(i)-1)+1
          lasresat=igpbs(lasgrp(i)+2)
       endif

       if(chkhis(i) > 0) then
          firgrpat=chkhis(i)
          lasgrpat=firgrpat+1
          natingrp=2
       else
          !--- first atom number from which atomic charges are changed upon titration
          firgrpat=igpbs(k)+1
          !--- last atom number to which atomic charges are changed upon titration
          lasgrpat=igpbs(k+1)
          natingrp=lasgrpat-firgrpat+1
       endif
       dQ=-Q/natingrp

       !--- Born radii

       do aa=firresat,lasresat  ! loop over model
          xa=x(aa)
          ya=y(aa)
          za=z(aa)
          vsum =0.0D0
          vsum2=0.0D0

          do ii=1,nrad         ! Radial Part
             dr=rgp(ii)
             angsum=0.0D0

             do jj=1,nang      ! Angular Part
                sfprod=1.0D0

                xp=dr*xgp(jj)+xa
                yp=dr*ygp(jj)+ya
                zp=dr*zgp(jj)+za

                bb_loop: do bb=firresat,lasresat
                   radib=radius(bb)
                   xb=x(bb)
                   yb=y(bb)
                   zb=z(bb)
                   rsw1=radib-sw
                   rsw1=rsw1*rsw1
                   rsw2=radib+sw
                   rsw2=rsw2*rsw2

                   r2=(xp-xb)*(xp-xb)+(yp-yb)*(yp-yb)+(zp-zb)*(zp-zb)

                   !--- if the integration point is outside an atom
                   if(r2.ge.rsw2) cycle bb_loop
                   !--- if the integration point is inside an atom
                   if(r2.le.rsw1) then
                      sfprod=0.0D0
                      exit bb_loop
                   else
                      r=sqrt(r2)-radib
                      sf=0.5D0+prefac1*r-prefac2*r*r*r
                      sfprod=sfprod*sf
                   endif
                enddo bb_loop
                angsum=angsum+wang(jj)*(1.0D0-sfprod)
             enddo

             dr2=dr*dr
             vsum =vsum +angsum*wrad(ii)/dr2           ! vsum *4.0D0*pi
             vsum2=vsum2+angsum*wrad(ii)/(dr2*dr2*dr)  ! vsum2*4.0D0*pi
          enddo

          recipr=aa0*(1.0D0/rmin-vsum)+             & ! vsum/4.0D0/pi
               aa1*(0.25D0/rmin**4-vsum2)**(0.25D0)   ! vsum2/4.0D0/pi
          rborn(aa)=1.0D0/recipr

          ! write(*,*) aa, atype(aa), rborn(aa)
       enddo

       !--- Electrostatic energies

       dGmod_born(i)=0.0D0   ! Born   electrostatic energy change
       dGmod_back(i)=0.0D0   ! static electrostatic energy change

       do aa=firgrpat,lasgrpat   ! loop over titrating group
          xa=x(aa)
          ya=y(aa)
          za=z(aa)
          rBorna=rBorn(aa)

          if(Q == -1.0D0) then
             cga_u=cg(aa)
             cga_p=cg(aa)+dQ
          else
             cga_p=cg(aa)
             cga_u=cg(aa)+dQ
          endif

          !--- self term -> factor 1/2
          dGmod_born(i)=dGmod_born(i) - &
               0.5D0*tau*(cga_p*cga_p-cga_u*cga_u)/rBorna

          do bb=aa+1,lasgrpat
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)

             if(Q == -1.0D0) then
                cgb_u=cg(bb)
                cgb_p=cg(bb)+dQ
             else
                cgb_p=cg(bb)
                cgb_u=cg(bb)+dQ
             endif

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGmod_born(i)=dGmod_born(i) - &
                  tau*(cga_p*cgb_p-cga_u*cgb_u)/rGB
          enddo

          !--- loop over atom whose charges are not changed upon titration
          do bb=firresat,firgrpat-1
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cg(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGmod_born(i)=dGmod_born(i) - &
                  tau*(cga_p-cga_u)*cgb/rGB
          enddo

          !--- loop over atom whose charges are not changed upon titration
          do bb=lasgrpat+1,lasresat
             xb=x(bb)
             yb=y(bb)
             zb=z(bb)
             cgb=cg(bb)

             r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
             r2born=rBorna*rBorn(bb)

             expfac=exp(-r2ab/(4.0D0*r2born))
             rGB=sqrt(r2ab+r2born*expfac)

             !--- cross term -> factor 1
             dGmod_born(i)=dGmod_born(i) - &
                  tau*(cga_p-cga_u)*cgb/rGB
          enddo
       enddo
    enddo

    !--- calculate pKintrinsic (pKa shift) and write the result

    temp=300.0D0
    loge=log10(exp(1.0D0))
    kBT = kboltz*temp

    if(prnlev>2) write(outu,102) 'pKa calculations:'
    if(Qsrmodel) then
       if(prnlev>2) write(outu,102)  &
            'A model compound is an isolated ', &
            'single residue'
    else
       if(prnlev>2) write(outu,102)  &
            'A model compound is an extended ', &
            'isolated single residue'
    endif
    if(prnlev>2) write(outu,102) 'pKa_shift = pKa_intr - pKa_mod'
    if(prnlev>2) write(outu,102)  &
         '          = - log10(e)/kBT*(dG_born+dG_back)'
    if(prnlev>2) write(outu,102) 'dG_born and dG_back are in kcal/mol'
    if(prnlev>2) write(outu,102)
    if(prnlev>2) write(outu,'(6x,a,a)')  &
         '---------------------------------------', &
         '-------------------------------------------'
    if(prnlev>2) write(outu,'(19x,a,a)') &
         'pKa_shift  pKa_intr   pKa_mod  pKa_born  pKa_back   ', &
         'dG_born   dG_back'
    if(prnlev>2) write(outu,'(6x,a,a)')  &
         '---------------------------------------', &
         '-------------------------------------------'

    do i=1,ntitres
       j=titres(i)
       do ii=1,nseg
          aa=nictot(ii)
          bb=nictot(ii+1)
          if(j > aa.and.j.le.bb) k=ii
       enddo
       if(res(j)(1:4) == 'ASP ') pka_mod=4.0D0
       if(res(j)(1:4) == 'GLU ') pka_mod=4.4D0
       if(res(j)(1:4) == 'ARG ') pka_mod=0.0D0
       if(res(j)(1:4) == 'LYS ') pka_mod=10.4D0
       if(res(j)(1:4) == 'HSP ') pka_mod=0.0D0
       dG_born=(dGprot_born(i)-dGmod_born(i))*ccelec
       dG_back=(dGprot_back(i)-dGmod_back(i))*ccelec
       pKa_shift=-1.0D0*loge/kBT*(dG_born+dG_back)
       pKa_intr =pKa_shift+pKa_mod
       pKa_born =-1.0D0*loge/kBT*dG_born
       pKa_back =-1.0D0*loge/kBT*dG_back
       if(prnlev>2) write(outu,101) segid(k)(1:idleng), &
            res(j)(1:idleng), &
            resid(j)(1:idleng), &
            pKa_shift,pKa_intr,pKa_mod, &
            pKa_born,pKa_back,dG_born,dG_back
    enddo
    if(prnlev>2) write(outu,'(6x,a,a)')  &
         '---------------------------------------', &
         '-------------------------------------------'
    if(prnlev>2) write(outu,101)

    iunit=6
    if(prnlev>2) write(outu,102)  &
         'Site-site interaction energies are written in ', &
         'unit', iunit
    do i=1,ntitres
       ! write(iunit,'(i3)') i
       ! do j=1,ntitres
       write(iunit,'(5x,100f12.5)') (Ginter(i,j)*ccelec,j=1,ntitres)
       ! enddo
    enddo

    if(prnlev>2) write(outu,101)

101 format(6x,a,1x,a,a,7(f8.3,2x))
102 format(6x,a,a,i3)

    return
  end SUBROUTINE GB_pKa


  SUBROUTINE GB_SOLV0(natom,x,y,z,cg,sw,aa0,aa1, &
       xgp,ygp,zgp,wang,nang, &
       rgp,wrad,nrad,rmin, &
       rcylngb, &
       rBORN,gbelec, &
       nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
       startalst,gblookup, &
       QGBparm,QGBener,ipforce)
    !-----------------------------------------------------------------------
    !---     calculate Born radii and GB solvation energy
    !---

  use consta
  use number

    integer ::natom,nang,nrad
    integer ::nxgp,nygp,nzgp
    integer ::startalst(*), gblookup(*)
    integer ::ipforce
    real(chm_real)  x(*),y(*),z(*),cg(*)
    real(chm_real)  xgp(*),ygp(*),zgp(*),wang(*)
    real(chm_real)  rgp(*),wrad(*)
    real(chm_real)  sw,rmin,aa0,aa1
    ! real(chm_real)  tmembgb,msw
    real(chm_real)  rcylngb,gbelec
    real(chm_real)  rborn(*)
    real(chm_real)  dgp,xgcen,ygcen,zgcen
    logical QGBparm,QGBener

    !--- local
    integer ::aa,bb,ii,jj
    integer ::ix,iy,iz,ipi,in,nfil,nalst
    integer ::checkip
    real(chm_real)  xa,ya,za,xb,yb,zb,xp,yp,zp
    real(chm_real)  radib,dr,dr2,r,r2,rsw1,rsw2,vsum,vsum2, &
         angsum,angsum2
    real(chm_real)  sf,sfprod,prefac1,prefac2,recipr,prefac0
    real(chm_real)  prefac1m,prefac2m
    real(chm_real)  tau,scrfac,effsalt
    real(chm_real)  cga,cgab,rBorna
    real(chm_real)  r2ab,r2born,expfac,rGB,rGB2
    real(chm_real)  tranx,trany,tranz,onedgp
    real(chm_real)  r0,x0,y0,z0
    real(chm_real)  xgpmax,xgpmin,ygpmax,ygpmin,zgpmax,zgpmin
    real(chm_real)  zmemb0,zmemb1,zmemb2,sfm
    real(chm_real)  rcyln1,rcyln2
    real(chm_real)  sflimit
    real(chm_real)  rmin2
    real(chm_real)  aavdwr,aavdweps,aasigma2,aarmin2,aaelj,dr4,dr10

    !--- constants & pre-factors

    sflimit=1.0D-8

    tau   = ( 1/epsp-1/epsw ) * ccelec
    if(kappa > 0.0) scrfac = 1.0D0 / kappa

    !--- prefac1 :  for r    term in switching function
    !--- prefac2 : for r**3 term in switching function
    prefac1=3.0D0/4.0D0/sw
    prefac2=1.0D0/4.0D0/(sw*sw*sw)

    !--- for lookup grid table

    onedgp=1.0D0/dgp
    !--- we need last +0.5D0*dgp to use sqrt(3.0)/2.0 in LOOKUP table building
    tranx=0.5D0*(nxgp-1)*dgp-xgcen+0.5D0*dgp
    trany=0.5D0*(nygp-1)*dgp-ygcen+0.5D0*dgp
    tranz=0.5D0*(nzgp-1)*dgp-zgcen+0.5D0*dgp

    !--- determine the dimension of the molecule
    xgpmax=-99999.0D0
    ygpmax=-99999.0D0
    zgpmax=-99999.0D0
    xgpmin= 99999.0D0
    ygpmin= 99999.0D0
    zgpmin= 99999.0D0
    do aa=1,natom
       r0=radius(aa)+sw
       x0=x(aa)+r0
       if(x0 > xgpmax) xgpmax=x0
       x0=x(aa)-r0
       if(x0 < xgpmin) xgpmin=x0
       y0=y(aa)+r0
       if(y0 > ygpmax) ygpmax=y0
       y0=y(aa)-r0
       if(y0 < ygpmin) ygpmin=y0
       z0=z(aa)+r0
       if(z0 > zgpmax) zgpmax=z0
       z0=z(aa)-r0
       if(z0 < zgpmin) zgpmin=z0
    enddo

    zmemb0=0.0D0
    zmemb1=0.0D0
    zmemb2=0.0D0
    if(tmembgb > 0.0D0) then
       zmemb0=0.5D0*tmembgb
       zmemb1=zmemb0-msw
       zmemb2=zmemb0+msw
       !--- prefac1m for r    term in membrane switching function
       !--- prefac2m for r**3 term in membrane switching function
       prefac1m=3.0D0/4.0D0/msw
       prefac2m=1.0D0/4.0D0/(msw*msw*msw)
       if(rcylngb > 0.0D0) then
          rcyln1=rcylngb-sw
          rcyln2=rcylngb+sw
       endif
    endif

    !--- calculate GB radii

    gvdw=0.0D0
    ipforce=0

    do aa=1,natom
       xa=x(aa)
       ya=y(aa)
       za=z(aa)
       vsum =0.0D0
       vsum2=0.0D0
       if(QGvdW) then
          aavdwr  =gvdwR(aa)
          aavdweps=gvdweps(aa)
          aarmin2 =aavdwr**(1.0/3.0)
          aasigma2=aarmin2/2.0**(1.0/3.0)
          aaelj   =0.0D0
       endif

       do ii=1,nrad                      ! Radial Part
          dr=rgp(ii)
          angsum=0.0D0
          angsum2=0.0D0

          do jj=1,nang                   ! Angular Part
             checkip=1
             sfprod=1.0D0

             zp=dr*zgp(jj)+za

             if(tmembgb > 0.0D0.and..not.QGvdW) then
                sfm=1.0D0
                if(abs(zp) < zmemb2) then
                   if(abs(zp).le.zmemb1) then   ! inside impermeable membrane
                      sfm=0.0D0

                      !wic ----- cylindrical pore ----
                                              if(rcylngb > 0.0D0) then
                                                 xp=dr*xgp(jj)+xa
                                                 yp=dr*ygp(jj)+ya
                                                 r2=sqrt(xp*xp+yp*yp)
                                                 if(r2 < rcyln2) then
                      !wic inside aqueous cylindrical pore
                                                    if(r2.le.rcyln1) then
                                                       sfm=1.0D0
                                                    else
                                                       r=r2-rcylngb
                                                       sfm=0.5D0-prefac1*r+prefac2*r*r*r
                                                    endif
                                                 endif
                                              endif
                      !wic ----- cylindrical pore ----

                   else
                      if(zp > 0.0D0) then
                         r=zp-zmemb0
                         sfm=0.5D0+prefac1m*r-prefac2m*r*r*r
                      else
                         r=zp+zmemb0
                         sfm=0.5D0-prefac1m*r+prefac2m*r*r*r
                      endif

                      !wic ----- cylindrical pore ----
                                              if(rcylngb > 0.0D0) then
                                                 xp=dr*xgp(jj)+xa
                                                 yp=dr*ygp(jj)+ya
                                                 r2=sqrt(xp*xp+yp*yp)
                                                 if(r2 < rcyln2) then
                      !wic inside aqueous cylindrical pore
                                                    if(r2.le.rcyln1) then
                                                       sfm=1.0D0
                                                    else
                                                       r=r2-rcylngb
                                                       sf=0.5D0-prefac1*r+prefac2*r*r*r
                                                       sfm=(sf+sfm)
                                                       if(sfm > 1.0D0) sfm=1.0D0
                                                    endif
                                                 endif
                                              endif
                      !wic ----- cylindrical pore ----

                   endif
                endif
                if(sfm < sflimit) then
                   sfprod=0.0D0
                   checkip=0
                   goto 91
                endif
                sfprod=sfm
             endif

             Zrange: if((zp >= zgpmin) .and. (zp <= zgpmax)) then
                xp=dr*xgp(jj)+xa
                Xrange: if((xp >= xgpmin) .and. (xp <= xgpmax)) then
                   yp=dr*ygp(jj)+ya
                   Yrange: if((yp >= ygpmin) .and. (yp <= ygpmax)) then

                      ix=int((xp+tranx)*onedgp)+1
                      iy=int((yp+trany)*onedgp)+1
                      iz=int((zp+tranz)*onedgp)+1

                      nfil = natomingp(ix,iy,iz)
                      if (nfil > 0) then
                         ipi = gptoalst(ix,iy,iz)
                         nalst=startalst(ipi)
                      endif

                      Nfil_loop: do in=1,nfil
                         bb=gblookup(nalst+in)
                         radib=radius(bb)
                         xb=x(bb)
                         yb=y(bb)
                         zb=z(bb)

                         r2=(xp-xb)*(xp-xb)+(yp-yb)*(yp-yb)+(zp-zb)*(zp-zb)

                         rsw2=radib+sw
                         rsw2=rsw2*rsw2
                         !--- if the integration point is outside an atom
                         if(r2.ge.rsw2) cycle Nfil_loop

                         rsw1=radib-sw
                         rsw1=rsw1*rsw1
                         !--- if the integration point is inside an atom
                         if(r2.le.rsw1) then
                            sfprod=0.0D0
                            checkip=0
                            exit Nfil_loop
                         endif

                         r=sqrt(r2)-radib
                         sf=0.5D0+prefac1*r-prefac2*r*r*r
                         sfprod=sfprod*sf
                         if(sfprod < sflimit) then
                            sfprod=0.0D0
                            checkip=0
                            exit Nfil_loop
                         endif
                      enddo Nfil_loop
                   endif Yrange
                endif Xrange
             endif Zrange

91           continue
             if(sfprod == 1.0D0) checkip=0

             angsum=angsum+wang(jj)*(1.0D0-sfprod)

             if(tmembgb > 0.0D0.and.QGvdW) then
                sfm=1.0D0
                if(abs(zp) < zmemb2) then
                   if(abs(zp).le.zmemb1) then   ! inside impermeable membrane
                      sfm=0.0D0
                   else
                      if(zp > 0.0D0) then
                         r=zp-zmemb0
                         sfm=0.5D0+prefac1m*r-prefac2m*r*r*r
                      else
                         r=zp+zmemb0
                         sfm=0.5D0-prefac1m*r+prefac2m*r*r*r
                      endif
                   endif
                endif
                sfprod=sfm*sfprod
                if(sfprod < sflimit) then
                   sfprod=0.0D0
                endif
                if(sfprod > 0.0D0.and.sfprod < 1.0D0) checkip=1
             endif

             angsum2=angsum2+wang(jj)*(1.0D0-sfprod)
             if(checkip == 1) ipforce=ipforce+1

          enddo

          dr2=dr*dr
          vsum =vsum +angsum2*wrad(ii)/dr2           ! vsum *4.0D0*pi
          vsum2=vsum2+angsum2*wrad(ii)/(dr2*dr2*dr) ! vsum2*4.0D0*pi

          if(QGvdW.and.dr2 > aasigma2) then
             if(dr2 > aarmin2) then
                dr4  =dr2*dr2
                dr10 =dr4*dr4*dr2
                aaelj=aaelj+4.0D0*pi*angsum*wrad(ii)* &
                     aavdweps*aavdwr*(aavdwr/dr10-2.0/dr4)
             else
                aaelj=aaelj-4.0D0*pi*angsum*wrad(ii)*aavdweps*dr2
             endif
          endif
       enddo

       rmin2=rmin*rmin
       recipr=aa0*(1.0D0/rmin-vsum)+                       & ! vsum/4.0D0/pi
            aa1*(0.25D0/(rmin2*rmin2)-vsum2)**(0.25D0) ! vsum2/4.0D0/pi
       rborn(aa)=1.0D0/recipr

       !--- we need this to parameterize aa0 and aa1 against PB results
       if(QGBparm) then
          prefac0=-0.5D0*tau*CG(aa)*CG(aa)
          write(10,'(i10,2f15.7)') aa, &
               prefac0*(1.0D0/rmin-vsum), &
               prefac0*(0.25D0/rmin**4-vsum2)**(0.25D0)
          if(qgvdw)then
             write(11,'(i10,5f12.5)') aa,rborn(aa),prefac0/rborn(aa), &
                  -0.5D0*tau/rborn(aa), &
                  gvdwaa(aa)-aaelj,gvdwaa(aa)
          endif
          ! write(*,'(i10,6f12.5)') aa,rborn(aa),radius(aa),aavdwr, &
          !      aaelj,gvdwaa(aa),aaelj+gvdwaa(aa)
          ! write(*,*) 4*pi*aavdweps*aavdwr*( &
          !      -aavdwr/9.0/5.04560**9+2.0/3.0/5.04560**3+ &
          !      aavdwr/9.0/4.00000**9-2.0/3.0/4.00000**3 )
          ! write(*,*) 4*pi*aavdweps*sqrt(aavdwr)/3.0*(-1.0+1.0/sqrt(2.0))
       endif

       if(qgvdw)then
          aaelj=bb0(aa)*(gvdwaa(aa)-aaelj)+bb1(aa)
          if(aaelj > 0.0D0) aaelj=0.0D0
          gvdw=gvdw+aaelj

          if(QGBparm) then
             write(12,'(i10,5f12.5)') aa,rborn(aa),prefac0/rborn(aa), &
                  -0.5D0*tau/rborn(aa), &
                  aaelj,gvdwaa(aa)
          endif
       endif
    enddo

    IF(.not.QGBener) return

    !--- calculate the electrostatic solvation energy

    gbelec=0.0D0
    do aa=1,natom
       xa=x(aa)
       ya=y(aa)
       za=z(aa)
       cga=cg(aa)
       rBorna=rBorn(aa)

       if(kappa > 0.0D0) then
          effsalt=exp(-rBorna*scrfac)/epsw
          tau=( 1/epsp-effsalt ) * ccelec
       endif
       !--- self term -> factor 1/2
       gbelec = gbelec - 0.5D0*tau*cga*cga/rBorna

       do bb=aa+1,natom
          xb=x(bb)
          yb=y(bb)
          zb=z(bb)

          r2ab=(xb-xa)*(xb-xa)+(yb-ya)*(yb-ya)+(zb-za)*(zb-za)
          r2born=rBorna*rBorn(bb)

          expfac=exp(-r2ab/(4.0D0*r2born))
          rGB=sqrt(r2ab+r2born*expfac)

          if(kappa > 0.0D0) then
             effsalt=exp(-rGB*scrfac)/epsw
             tau=( 1/epsp-effsalt ) * ccelec
          endif
          !--- cross term -> factor 1
          gbelec = gbelec - tau*cga*cg(bb)/rGB

       enddo
    enddo

    return
  end SUBROUTINE GB_SOLV0


  SUBROUTINE GB_SURF0(natom,atype,x,y,z,rsasa,sw,tmembgb,msw, &
       xgp,ygp,zgp,wang,nang,gbsurf, &
       nxgp,nygp,nzgp,dgp,xgcen,ygcen,zgcen, &
       startalst,gblookup,QGBparm)
    !------------------------------------------------------------------------
    !---     calculate van der Waals or Solven-accessible surface
    !---     by taking a first derivative of the switching fucntion
    !---
    !---     NOTE:
    !---     (1) sw = 0.1
    !---     (2) non-hydrogen atoms only
    !---

  use consta
  use number

    integer ::natom,nang
    integer ::nxgp,nygp,nzgp
    integer ::startalst(*), gblookup(*)
    real(chm_real)  x(*),y(*),z(*),rsasa
    real(chm_real)  xgp(*),ygp(*),zgp(*),wang(*)
    real(chm_real)  dgp,xgcen,ygcen,zgcen
    real(chm_real)  sw,gbsurf,tmembgb,msw
    character(len=*) :: atype(*)
    logical QGBparm

    !--- local
    integer ::aa,bb,ii,jj,nsrad
    parameter (nsrad=3)
    integer ::ix,iy,iz,ipi,in,nfil,nalst
    real(chm_real)  xa,ya,za,xb,yb,zb,xp,yp,zp
    real(chm_real)  radia,radib,dr,r,r2,rp,rsw1,rsw2
    real(chm_real)  surfsum
    real(chm_real)  sf,dsfra,sfprod
    real(chm_real)  prefac1,prefac2,prefac3,prefac4,prefac5
    real(chm_real)  prefac1m,prefac2m,prefac3m,prefac4m
    real(chm_real)  xgp0,ygp0,zgp0,wang0
    real(chm_real)  rmin,rmax,srgp(nsrad),swrad(nsrad)
    real(chm_real)  fourpi
    real(chm_real)  tranx,trany,tranz,onedgp
    real(chm_real)  zmemb0,zmemb1,zmemb2,sfm
    real(chm_real)  sflimit

    !--- constants & pre-factors

    sflimit=1.0D-8

    fourpi=4.0D0*pi

    !--- prefac1 for r      term in switching function
    !--- prefac2 for r**3   term in switching function
    !--- prefac3 for const. term in first derivatives of switching function
    !--- prefac4 for r**2   term in first derivatives of switching function
    prefac1=3.0D0/4.0D0/sw
    prefac2=1.0D0/4.0D0/(sw*sw*sw)
    prefac3=3.0D0/4.0D0/sw
    prefac4=3.0D0/4.0D0/(sw*sw*sw)

    !--- for lookup grid table

    onedgp=1.0D0/dgp
    !--- we need last +0.5D0*dgp to use sqrt(3.0)/2.0 in LOOKUP table building
    tranx=0.5D0*(nxgp-1)*dgp-xgcen+0.5D0*dgp
    trany=0.5D0*(nygp-1)*dgp-ygcen+0.5D0*dgp
    tranz=0.5D0*(nzgp-1)*dgp-zgcen+0.5D0*dgp

    !--- for membrane

    zmemb0=0.0D0
    zmemb1=0.0D0
    zmemb2=0.0D0
    if(tmembgb > 0.0D0) then
       zmemb0=0.5D0*tmembgb
       zmemb1=zmemb0-msw
       zmemb2=zmemb0+msw
       !--- prefac1m for r      term in membrane switching function
       !--- prefac2m for r**3   term in membrane switching function
       !--- prefac3m for const. term in first derivatives of
       !----                                      membrane switching function
       !--- prefac4m for r**2   term in first derivatives of
       !---                                       membrane switching function
       prefac1m=3.0D0/4.0D0/msw
       prefac2m=1.0D0/4.0D0/(msw*msw*msw)
       prefac3m=3.0D0/4.0D0/msw
       prefac4m=3.0D0/4.0D0/(msw*msw*msw)
    endif

    !--- get the radial integration points and weights
    !---                 using Gauss-Legendre quadrature

    rmin=0.0
    rmax=2.0D0*sw

    call gauleg(rmin,rmax,srgp,swrad,nsrad,0)

    !--- calculate surface

    gbsurf=0.0D0
    Atom_loop: do aa=1,natom
       radia=radius(aa)+rsasa
       if(atype(aa)(1:1) == 'H') cycle atom_loop
       if(radia == 0.0)         cycle atom_loop

       xa=x(aa)
       ya=y(aa)
       za=z(aa)
       rmin=radia-sw
       !wi         rmax=radia+sw

       !wi         call gauleg(rmin,rmax,srgp,swrad,nsrad,0)

       surfsum=0.0D0

       Radial_loop: do ii=1,nsrad      !    !--- Radial Part
          dr=srgp(ii)+rmin
          r=dr-radia
          !--- switching function H'  with respect to vec r
          dsfra= prefac3-prefac4*r*r

          prefac5=swrad(ii)*dr*dr

          Angular_loop: do jj=1,nang   !! Angular Part

             sfprod=1.0D0

             zp=dr*zgp(jj)+za

             if(tmembgb > 0.0D0) then
                sfm=1.0D0
                if(abs(zp) < zmemb2) then
                   if(abs(zp).le.zmemb1) then  ! inside impermeable membrane
                      sfm=0.0D0
                   else
                      if(zp > 0.0D0) then
                         r=zp-zmemb0
                         sfm=0.5D0+prefac1m*r-prefac2m*r*r*r
                      else
                         r=zp+zmemb0
                         sfm=0.5D0-prefac1m*r+prefac2m*r*r*r
                      endif
                   endif
                endif
                if(sfm < sflimit) then
                   sfprod=0.0D0
                   cycle Angular_loop
                endif
                sfprod=sfm
             endif

             xp=dr*xgp(jj)+xa
             yp=dr*ygp(jj)+ya
             wang0=wang(jj)

             ix=int((xp+tranx)*onedgp)+1
             iy=int((yp+trany)*onedgp)+1
             iz=int((zp+tranz)*onedgp)+1

             nfil = natomingp(ix,iy,iz)
             if (nfil > 0) then
                ipi = gptoalst(ix,iy,iz)
                nalst=startalst(ipi)
             endif

             do in=1,nfil
                bb=gblookup(nalst+in)
                if(bb == aa) cycle
                xb=x(bb)
                yb=y(bb)
                zb=z(bb)
                radib=radius(bb)+rsasa

                r2=(xp-xb)*(xp-xb)+(yp-yb)*(yp-yb)+(zp-zb)*(zp-zb)

                rsw2=radib+sw
                rsw2=rsw2*rsw2
                !--- if the integration point is outside an atom
                if(r2.ge.rsw2) cycle

                rsw1=radib-sw
                rsw1=rsw1*rsw1
                !--- if the integration point is inside an atom
                if(r2.le.rsw1) then
                   sfprod=0.0D0
                   surfsum=surfsum+dsfra*sfprod*wang0*prefac5
                   exit
                endif

                r=sqrt(r2)-radib
                sf=0.5D0+prefac1*r-prefac2*r*r*r
                sfprod=sfprod*sf
                if(sfprod < sflimit) then
                   sfprod=0.0D0
                   surfsum=surfsum+dsfra*sfprod*wang0*prefac5
                   exit
                endif

             enddo

             surfsum=surfsum+dsfra*sfprod*wang0*prefac5
          enddo Angular_loop
       enddo Radial_loop

       surfsum=fourpi*surfsum

       if(QGBparm) then
          write(90,'(i10,2f12.5,a4)') aa,surfsum, &
               4.0*pi*radia*radia*(1+sw*sw/5.0D0/(radia*radia)), &
               atype(aa)
       endif

       gbsurf=gbsurf+surfsum
    enddo Atom_loop

    return
  end SUBROUTINE GB_SURF0


  SUBROUTINE GB_LOOKUP(X,Y,Z)
    !-----------------------------------------------------------------
    !---     (1) Build a lookup table
    !---     (2) Rotate integration points for roatational invariance
    !---

  use dimens_fcm
  use bases_fcm
  use psf
  use image
  use stream
  use contrl    ! to get TOTUPD
  use memory

    real(chm_real)  X(*),Y(*),Z(*)

    !--- local
    integer(chm_int8) :: nxyzgp
    integer :: i
    integer ::tmpnxgp,tmpnygp,tmpnzgp
    integer ::tmpnatlst,tmpnbusygp
    real(chm_real)  tranx,trany,tranz
    logical QLTupdate,QGBIM

    IGBSWcall=IGBSWcall+1
    if(IGBSWcall == 1) then
       nxgp=0
       nygp=0
       nzgp=0
       nbusygp=0
       natlst=0
       ipforce=0
       if(natim > 0 .and. .not. Qhybrid) then
          oldtotupd=totupd
          oldnatim =natim
          !--- number of atoms in inversion images
          call chmalloc('gbsw.src','GB_LOOKUP','gbimatpt',ntrans,intg=gbimatpt)
          call chmalloc('gbsw.src','GB_LOOKUP','gbimattr',(natim-natom)*8+1,intg=gbimattr)
       endif
    endif

    !--- check if there are image atoms and build all necessary image atoms
    !--- determine how many atoms (gbnatim) we need more to complete images
    !--- based on the inversion of the symmetry transformations.
    !--- NOTE: CHARMM uses only the images atoms of a lower inverse index, but
    !---       we need all image atoms to calculate Born Radii correctly.

    if(natim > 0 .and. .not. Qhybrid) then

       !--- Since we do not know how many atoms we need more, we guess here !!!

       if(natim > oldnatim) then
          if (allocated(gbimattr)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','gbimattr', &
               (oldnatim-natom)*8+1,intg=gbimattr)
          call chmalloc('gbsw.src','GB_LOOKUP','gbimattr',(natim-natom)*8+1,intg=gbimattr)
          oldnatim = natim
       endif

       CALL GB_IMAGE(natom,natim,X,Y,Z,cutim, &
            ntrans,BIMAG%imatpt,iminv, &
            gbnatim)

       natgb = natim + gbnatim

       !wi  QGBIM=(oldtotupd /= totupd).or.(natgb > oldnatgb)
       QGBIM=(natgb > oldnatgb)

       ! write(*,*) oldtotupd,totupd
       ! write(*,*) natom,natim,oldnatim,natgb,oldnatgb

       if(QGBIM) then
          if (allocated(gbx)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','gbx',natgb,crl=gbx)
          if (allocated(gby)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','gby',natgb,crl=gby)
          if (allocated(gbz)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','gbz',natgb,crl=gbz)
          if (allocated(radius)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','radius',natgb,crl=radius)
          if (allocated(gbitrn)) &
               call chmdealloc('gbsw.src','GB_LOOKUP','gbitrn',natgb,intg=gbitrn)

          call chmalloc('gbsw.src','GB_LOOKUP','gbx',natgb,crl=gbx)
          call chmalloc('gbsw.src','GB_LOOKUP','gby',natgb,crl=gby)
          call chmalloc('gbsw.src','GB_LOOKUP','gbz',natgb,crl=gbz)
          call chmalloc('gbsw.src','GB_LOOKUP','radius',natgb,crl=radius)
          call chmalloc('gbsw.src','GB_LOOKUP','gbitrn',natgb,intg=gbitrn)

          oldnatgb = natgb
          oldtotupd=totupd
       endif

    elseif(IGBSWcall == 1) then
       natgb = ngbsel
       call chmalloc('gbsw.src','GB_LOOKUP','gbx',natgb,crl=gbx)
       call chmalloc('gbsw.src','GB_LOOKUP','gby',natgb,crl=gby)
       call chmalloc('gbsw.src','GB_LOOKUP','gbz',natgb,crl=gbz)
       call chmalloc('gbsw.src','GB_LOOKUP','radius',natgb,crl=radius)
    endif

    !--- copy all X,Y,Z,saveRPB (in primary system) to GBx,GBy,GBz,radius
    ! write(*,*) "First", natim, natom, natgb, ntrans

    CALL GB_COOR0(ngbsel,X,Y,Z,GBx,GBy,GBz)

    !--- create all image atoms and their radii if requested
    ! write(*,*) natim, natom, natgb, ntrans
    if(natim > natom .and. .not. Qhybrid) then
       CALL GB_COOR1(natom,natim,natgb,X,Y,Z, &
            GBx,GBy,GBz, &
            gbitrn, &
            ntrans,imtrns,iminv, &
            BIMAG%imatpt,BIMAG%imattr,NOROT)
    endif

    !--- check if we need to save sfprod in integration points
    !---                              which contribute forces
    !--- it may cause segmentation fault if ipforce changes more than 5% -
    !--- in one-step (especially with membrane)

    if(mod(IGBSWcall,igbfrq) /= 0.and.IGBSWcall /= 1) return
    !--- check the displacements of (only primary) atoms
    !--- to decide if updating of lookup table is necessary

    if(rbuffer /= 0.0D0.and.IGBSWcall /= 1) then
       call GB_LTUPDATE(natom,X,Y,Z,rbuffer,QLTupdate)

       if(.not.QLTupdate) return ! do not update Lookup table
    endif

    !--- determine a grid size
    if(prnlev > 6) write(outu,'(a)') &
         "gb_lookup calling GB_LOOKUP0"
    CALL GB_LOOKUP0(natgb,GBx,GBy,GBz, &
         sw,dgp,rbuffer,rsasa,tmpnxgp,tmpnygp,tmpnzgp, &
         xgcen,ygcen,zgcen)

    !--- allocate HEAP storage and, if necessary, free previous HEAP storage
    if(tmpnxgp > nxgp.or.tmpnygp > nygp.or.tmpnzgp > nzgp) then
       if(IGBSWcall /= 1) then
          call chmdealloc('gbsw.src','GB_LOOKUP','natomingp',nxgp,nygp,nzgp,ci2=natomingp)
          call chmdealloc('gbsw.src','GB_LOOKUP','gptoalst',nxgp,nygp,nzgp,intg=gptoalst)
       endif
       nxgp=tmpnxgp
       nygp=tmpnygp
       nzgp=tmpnzgp

       call chmalloc('gbsw.src','GB_LOOKUP','natomingp',nxgp,nygp,nzgp,ci2=natomingp)
       call chmalloc('gbsw.src','GB_LOOKUP','gptoalst',nxgp,nygp,nzgp,intg=gptoalst)
    endif

    tranx=0.5D0*(nxgp-1)*dgp
    trany=0.5D0*(nygp-1)*dgp
    tranz=0.5D0*(nzgp-1)*dgp

    if(IGBSWcall == 1) then
       nxyzgp = nxgp * nygp  ! avoid int32 overflow
       nxyzgp = nxyzgp * nzgp
       if(prnlev>2) write(outu,102) 'GBSW lookup table information:'
       if(prnlev>2) write(outu,'(6x,a,i3,a,i3,a,i3,a,i10)') &
            'Number of grid points for lookup table : ', &
            nxgp,' x',nygp,' x',nzgp,' = ',nxyzgp
       if(prnlev>2) write(outu,102) 'Lookup table in x covers from', &
            xgcen-tranx,'to',xgcen+tranx
       if(prnlev>2) write(outu,102) 'Lookup table in y covers from', &
            ygcen-trany,'to',ygcen+trany
       if(prnlev>2) write(outu,102) 'Lookup table in z covers from', &
            zgcen-tranz,'to',zgcen+tranz
    endif

102 format(6x,a,1x,f8.3,1x,a,1x,f8.3)

    !--- determine how many atoms are in a grid point
    !--- the list does not include atoms whose radii are zero.

    CALL GB_LOOKUP1(natgb,GBx,GBy,GBz, &
         sw,dgp,rbuffer,rsasa,nxgp,nygp,nzgp,xgcen,ygcen,zgcen, &
         tmpnatlst,tmpnbusygp)

    !--- allocate storage and, if necessary, free the previous storage
    if(tmpnbusygp > nbusygp) then
       if(IGBSWcall /= 1)  &
            call chmdealloc('gbsw.src','GB_LOOKUP','startalst',nbusygp,intg=startalst)
       nbusygp=tmpnbusygp+tmpnbusygp/50
       !--- give the starting number of atom list to a grip point
       call chmalloc('gbsw.src','GB_LOOKUP','startalst',nbusygp,intg=startalst)
    endif

    if(tmpnatlst > natlst) then
       if(IGBSWcall /= 1) then
          call chmdealloc('gbsw.src','GB_LOOKUP','tmpdist',natlst,cr4=tmpdist)
          call chmdealloc('gbsw.src','GB_LOOKUP','gblookup',natlst,intg=gblookup)
       endif
       natlst=tmpnatlst+tmpnatlst/50
       call chmalloc('gbsw.src','GB_LOOKUP','gblookup',natlst,intg=gblookup)
       !--- save atom-to-grip point distance for sorting
       call chmalloc('gbsw.src','GB_LOOKUP','tmpdist',natlst,cr4=tmpdist)
    endif

    if(IGBSWcall == 1) then
       if(prnlev>2) write(outu,'(6x,a,i10)') &
            'Number of atoms listed in lookup table   = ', natlst
       if(prnlev>2) write(outu,'(6x,a,f10.3)') &
            'Percentage of occupied grid points       = ', &
            nbusygp*100.D0/(nxyzgp*1.0D0)
       if(prnlev>2) write(outu,102)
    endif

    !--- determine (a starting number - 1) for the atom lists in the table, and
    !--- build the lookup table

    CALL GB_LOOKUP2(natgb,GBx,GBy,GBz, &
         sw,dgp,rbuffer,rsasa,nxgp,nygp,nzgp,xgcen,ygcen,zgcen, &
         startalst,gblookup)

    !---
    !--- Since we are using a numerical quadrature for volume integration,
    !--- it is necessary to rotate the integration points onto molecular frame.
    !--- The following procedure is based on Johnson et al. CPL 220:377-384 (1994).
    !--- Instead of using the nuclear charge moment tensor,
    !--- we are using the inertia tensor (so I can use the CORINER subroutine).
    !
    IF(IGBSWcall == 1) THEN
       xgp(1:nang) = xgpleb(1:nang)
       ygp(1:nang) = ygpleb(1:nang)
       zgp(1:nang) = zgpleb(1:nang)
    ENDIF

    IF(Qrotinv) THEN
       CALL GB_ROTINVAR(ngbsel,X,Y,Z,AMASS, &
            GBPA,GBEV, &
            xgp,ygp,zgp,nang)
    ENDIF

    return
  end SUBROUTINE GB_LOOKUP


  SUBROUTINE GB_COOR0(natom,X,Y,Z,GBx,GBy,GBz)
    !---------------------------------------------------------------------------
    !--- copy all X,Y,Z,saveRPB to GBx,GBy,GBz,radius
    !---


    integer, intent(in) :: natom
    real(chm_real), intent(in) ::  x(natom),y(natom),z(natom)
    real(chm_real), intent(out) ::  GBx(natom),GBy(natom),GBz(natom)

    !--- local
    integer ::aa

    do aa=1,natom
       GBx(aa)=X(aa)
       GBy(aa)=Y(aa)
       GBz(aa)=Z(aa)
       radius(aa)=saveRPB(aa)
    enddo

    return
  end SUBROUTINE GB_COOR0


  SUBROUTINE GB_COOR1(natom,natim,natgb,X,Y,Z, &
       GBx,GBy,GBz,gbitrn, &
       ntrans,imtrns,iminv,imatpt,imattr,NOROT)
    !---------------------------------------------------------------------------
    !--- copy all X,Y,Z to GBx,GBy,GBz and create all image atoms

  use stream

    integer ::natom,natim,natgb,ntrans
    integer ::iminv(*),imatpt(*),imattr(*)
    integer ::gbitrn(*)
    real(chm_real)  x(*),y(*),z(*),GBx(*),GBy(*),GBz(*)
    real(chm_real)  imtrns(*)
    logical NOROT

    !--- local
    integer ::aa,bb,cc,ipt,pcc,nimat
    integer ::itemp,itemp0,itrans,jtrans,istrt,iend

    !--- generate full image atoms and copy PB radii from those in primary system

    aa=natim
    itemp=1
    itemp0=natom+1

    do bb=itemp0,natim
       GBx(bb)=X(bb)
       GBy(bb)=Y(bb)
       GBz(bb)=Z(bb)
    enddo

    Trans_loop: do itrans=1,ntrans
       jtrans=iminv(itrans)
       istrt=itemp0
       iend=imatpt(itrans)
       nimat=iend-istrt+1
       itemp0=iend+1

       do bb=istrt,iend
          cc=imattr(bb)
          !--- copy radius of atom in primary system to image atoms
          radius(bb)=radius(cc)
          gbitrn(bb)=itrans
       enddo

       if(jtrans == itrans) cycle Trans_loop ! inversion itself
       if(nimat.le.0) cycle Trans_loop       ! no image atoms in nonbond list

       !--- note that we create atoms in jth image using
       !--- nonbond atom lists in its inversion image (ith image)
       ipt=(jtrans-1)*12

       istrt=itemp
       iend=gbimatpt(itrans)
       nimat=iend-istrt+1
       itemp=iend+1

       if(nimat.le.0) cycle Trans_loop

       !wi         write(*,*) itrans,nimat,istrt,itemp
       !wi         if(nimat.le.0) then
       !wi            write(*,*) 'Something wrong in GB_IMAGE and GB_COOR1'
       !wi            stop
       !wi         endif

       if(NOROT) then
          do cc=istrt,iend
             aa=aa+1
             !--- primary atom corresponding to this image atom
             pcc=gbimattr(cc)
             GBx(aa)=X(pcc)+imtrns(ipt+10)
             GBy(aa)=Y(pcc)+imtrns(ipt+11)
             GBz(aa)=Z(pcc)+imtrns(ipt+12)
             radius(aa)=radius(pcc)   ! copy radius to image atoms
             gbitrn(aa)=jtrans
          enddo
       else
          do cc=istrt,iend
             aa=aa+1
             pcc=gbimattr(cc) !primary atom corresponding to this image atom
             GBx(aa)=X(pcc)*imtrns(ipt+1)+Y(pcc)*imtrns(ipt+2)+ &
                  Z(pcc)*imtrns(ipt+3)+imtrns(ipt+10)
             GBy(aa)=X(pcc)*imtrns(ipt+4)+Y(pcc)*imtrns(ipt+5)+ &
                  Z(pcc)*imtrns(ipt+6)+imtrns(ipt+11)
             GBz(aa)=X(pcc)*imtrns(ipt+7)+Y(pcc)*imtrns(ipt+8)+ &
                  Z(pcc)*imtrns(ipt+9)+imtrns(ipt+12)
             radius(aa)=radius(pcc)   ! copy radius to image atoms
             gbitrn(aa)=jtrans
          enddo
       endif
90  enddo Trans_loop

    if(prnlev > 6) then
       write(outu,*)
       do aa=1,natgb
          write(outu,'(i5,4f12.5)') aa,GBx(aa),GBy(aa),GBz(aa), &
               radius(aa)
       enddo
    endif

    !wi      do itrans=1,ntrans
    !wi         write(*,*) itrans,imatpt(itrans)
    !wi      enddo

    return
  end SUBROUTINE GB_COOR1


  SUBROUTINE GB_IMAGE(natom,natim,X,Y,Z,cutim, &
       ntrans,imatpt,iminv,gbnatim)
    !--------------------------------------------------------------------
    !--- determine how many atoms we need more to complete images based on
    !--- the inversion of the symmetry transformations
    !---

    integer ::natom,natim,ntrans,gbnatim
    integer ::imatpt(*),iminv(*)
    real(chm_real)  X(*),Y(*),Z(*),cutim

    !--- local
    integer ::aa,bb,cnt
    integer ::itemp0,itrans,jtrans,istrt,iend,nimat
    real(chm_real)  xa,ya,za,xb,yb,zb,dxab,dyab,dzab,r2ab
    real(chm_real)  cutim2

    cutim2=cutim*cutim
    gbnatim=0
    itemp0=natom+1

    Trans_loop: do itrans=1,ntrans
       jtrans=iminv(itrans)
       istrt=itemp0
       iend=imatpt(itrans)
       nimat=iend-istrt+1
       itemp0=iend+1

       if(jtrans == itrans) cycle Trans_loop ! inversion itself
       if(nimat.le.0) cycle Trans_loop        ! no image atoms in nonbond list

       cnt=0
       outer: do aa=1,natom
          xa=X(aa)
          ya=Y(aa)
          za=Z(aa)

          do bb=istrt,iend
             xb=X(bb)
             yb=Y(bb)
             zb=Z(bb)

             dxab=xb-xa
             dyab=yb-ya
             dzab=zb-za
             r2ab=dxab*dxab+dyab*dyab+dzab*dzab

             if(r2ab.le.cutim2) then

                cnt=cnt+1
                gbimattr(cnt+gbnatim)=aa
                cycle outer

             endif
          enddo
       enddo outer

       gbnatim=gbnatim+cnt
       gbimatpt(itrans)=gbnatim
    enddo Trans_loop

    return
  end SUBROUTINE GB_IMAGE


  SUBROUTINE GB_LTUPDATE(natom,x,y,z,rbuffer,QLTupdate)
    !------------------------------------------------------------------
    !---

  use number

    integer ::natom
    real(chm_real)  x(*),y(*),z(*)
    real(chm_real)  rbuffer
    logical QLTupdate

    !--- local
    integer ::aa,bb
    real(chm_real)  rbuffer2,r2,xad,yad,zad

    QLTupdate=.false.
    rbuffer2=(rbuffer-rsmall)*(rbuffer-rsmall)
    do aa=1,natom
       xad=xGBref(aa)-x(aa)
       yad=yGBref(aa)-y(aa)
       zad=zGBref(aa)-z(aa)
       r2=xad*xad+yad*yad+zad*zad
       if(r2 > rbuffer2) then
          do bb=1,natom
             xGBref(bb)=x(bb)
             yGBref(bb)=y(bb)
             zGBref(bb)=z(bb)
          enddo
          QLTupdate=.true.
          return
       endif
    enddo

    return
  end SUBROUTINE GB_LTUPDATE


  SUBROUTINE GB_LOOKUP0(natom,x,y,z,sw,dgp,rbuffer,rsasa, &
       nxgp,nygp,nzgp,xgcen,ygcen,zgcen)
    !----------------------------------------------------------------------
    !---

    integer ::natom
    integer ::nxgp,nygp,nzgp
    real(chm_real)  x(*),y(*),z(*)
    real(chm_real)  sw,dgp,rbuffer,rsasa
    real(chm_real)  xgcen,ygcen,zgcen

    !--- local

    integer ::aa
    real(chm_real)  xgpmax,xgpmin,ygpmax,ygpmin,zgpmax,zgpmin,prefac1
    real(chm_real)  r0,x0,y0,z0

    xgpmax=-99999.0D0
    ygpmax=-99999.0D0
    zgpmax=-99999.0D0
    xgpmin= 99999.0D0
    ygpmin= 99999.0D0
    zgpmin= 99999.0D0

    prefac1=sw+0.5D0*sqrt(3.0D0)*dgp+rbuffer+rsasa
    do aa=1,natom
       r0=radius(aa)+prefac1
       x0=x(aa)+r0
       if(x0 > xgpmax) xgpmax=x0
       x0=x(aa)-r0
       if(x0 < xgpmin) xgpmin=x0
       y0=y(aa)+r0
       if(y0 > ygpmax) ygpmax=y0
       y0=y(aa)-r0
       if(y0 < ygpmin) ygpmin=y0
       z0=z(aa)+r0
       if(z0 > zgpmax) zgpmax=z0
       z0=z(aa)-r0
       if(z0 < zgpmin) zgpmin=z0
    enddo

    nxgp=int( (xgpmax-xgpmin) / dgp ) + 2
    nygp=int( (ygpmax-ygpmin) / dgp ) + 2
    nzgp=int( (zgpmax-zgpmin) / dgp ) + 2

    xgcen=(xgpmax+xgpmin)/2.0D0
    ygcen=(ygpmax+ygpmin)/2.0D0
    zgcen=(zgpmax+zgpmin)/2.0D0

    return
  end SUBROUTINE GB_LOOKUP0


  SUBROUTINE GB_LOOKUP1(natom,x,y,z,sw,dgp,rbuffer,rsasa, &
       nxgp,nygp,nzgp,xgcen,ygcen,zgcen, &
       nalst,ipi)
    !----------------------------------------------------------------------
    !---

  use number
!---wi##IF PARALLEL
!---wi  use parallel
!---wi##ENDIF

    integer, intent(in) :: natom, nxgp, nygp, nzgp
    integer :: nalst, ipi
    real(chm_real)  x(*),y(*),z(*)
    real(chm_real)  sw,dgp,rbuffer,rsasa
    real(chm_real)  xgcen,ygcen,zgcen

    !--- local
    integer ::aa
    integer ::nfil
    integer ::ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer ::ig,jg,kg
    real(chm_real)  tranx,trany,tranz
    real(chm_real)  radia,xaa,yaa,zaa
    real(chm_real)  rsqaa,rsq,prefac0
    real(chm_real)  xaagp,yaagp,zaagp,yaagp2,zaagp2
    real(chm_real) :: gridpt(max(nxgp,nygp,nzgp))

    tranx=0.5D0*(nxgp-1)*dgp-xgcen
    trany=0.5D0*(nygp-1)*dgp-ygcen
    tranz=0.5D0*(nzgp-1)*dgp-zgcen

    !--- initialize lookup table
    natomingp = 0
    gptoalst = 0

    prefac0=sw+0.5D0*sqrt(3.0D0)*dgp+rbuffer+rsasa

!wi##IF PARALLEL
!wi      do 90 aa=mynodp,natom,numnod
!wi##ELSE
!wi      do 90 aa=1,natom
!wi##ENDIF

    do ig = 1, max(nxgp,nygp,nzgp)
       gridpt(ig) = (ig - 1) * dgp
    enddo
    do aa=1,natom
       radia=radius(aa)
       if(radia == 0.0D0) cycle
       xaa=x(aa)+tranx
       yaa=y(aa)+trany
       zaa=z(aa)+tranz
       ix=int(xaa/dgp)+1
       iy=int(yaa/dgp)+1
       iz=int(zaa/dgp)+1
       rsqaa=radia+prefac0
       nfil=int(rsqaa/dgp)+1
       rsqaa=rsqaa*rsqaa+rsmall

       jx1=ix-nfil
       if(jx1 < 1) jx1=1
       jx2=ix+nfil
       if(jx2 > nxgp) jx2=nxgp
       jy1=iy-nfil
       if(jy1 < 1) jy1=1
       jy2=iy+nfil
       if(jy2 > nygp) jy2=nygp
       jz1=iz-nfil
       if(jz1 < 1) jz1=1
       jz2=iz+nfil
       if(jz2 > nzgp) jz2=nzgp

       do kg = jz1, jz2
          zaagp = gridpt(kg) - zaa
          zaagp2 = zaagp*zaagp

          do jg = jy1, jy2
             yaagp = gridpt(jg) - yaa
             yaagp2 = yaagp*yaagp + zaagp2

             do ig = jx1, jx2
                xaagp = gridpt(ig) - xaa
                rsq = xaagp*xaagp + yaagp2

                if (rsq <= rsqaa) natomingp(ig,jg,kg) = natomingp(ig,jg,kg) + 1

             enddo
          enddo
       enddo
    enddo

!wi##IF PARALLEL
!wi      CALL IGCOMB(natomingp,nxyzgp)
!wi##ENDIF

    !--- determine how many grid points (ipi) are occupied
    !--- and how many atoms (nalst) should be listed

    ipi=0
    nalst=0
    do kg = 1, nzgp
       do jg = 1, nygp
          do ig = 1, nxgp
             nfil = natomingp(ig,jg,kg)
             if (nfil /= 0) then
                nalst=nalst+nfil
                ipi=ipi+1
                gptoalst(ig,jg,kg) = ipi
             endif
          enddo
       enddo
    enddo

    return
  end SUBROUTINE GB_LOOKUP1


  SUBROUTINE GB_LOOKUP2(natom,x,y,z,sw,dgp,rbuffer,rsasa, &
       nxgp,nygp,nzgp,xgcen,ygcen,zgcen, &
       startalst,gblookup)
    !------------------------------------------------------------------------

  use number
!wi##IF PARALLEL
!wi  use parallel
!wi##ENDIF

    integer, intent(in) :: natom, nxgp, nygp, nzgp
    integer ::startalst(*), gblookup(*)
    real(chm_real)  x(*),y(*),z(*)
    real(chm_real)  sw,dgp,rbuffer,rsasa
    real(chm_real)  xgcen,ygcen,zgcen

    !--- local
    integer ::aa,bb
    integer ::nfil,nalst,nalst0
    integer ::ix,iy,iz,jx1,jx2,jy1,jy2,jz1,jz2
    integer ::ig,jg,kg,ipi
    integer, parameter :: nfilmax = 250
    integer ::al(nfilmax)
    real(chm_real)  tranx,trany,tranz
    real(chm_real)  radia,xaa,yaa,zaa
    real(chm_real)  rsqaa,rsq,prefac0
    real(chm_real)  xaagp,yaagp,zaagp,yaagp2,zaagp2
    real(chm_real)  r,r2,segpar(nfilmax)
    real(chm_real) :: gridpt(max(nxgp,nygp,nzgp))

    tranx=0.5D0*(nxgp-1)*dgp-xgcen
    trany=0.5D0*(nygp-1)*dgp-ygcen
    tranz=0.5D0*(nzgp-1)*dgp-zgcen

    !--- determine ( a starting number - 1 ) for the atom lists in the table, and
    nalst=0
    do kg = 1, nzgp
       do jg = 1, nygp
          do ig = 1, nxgp
             nfil = natomingp(ig,jg,kg)
             if (nfil /= 0) then
                ipi = gptoalst(ig,jg,kg)
                startalst(ipi)=nalst
                nalst=nalst+nfil
             endif
          enddo
       enddo
    enddo
    nalst0=nalst

    !--- now, build the lookup table

    gblookup(1:nalst) = 0
    tmpdist(1:nalst) = ZERO

    prefac0=sw+0.5D0*sqrt(3.0D0)*dgp+rbuffer+rsasa

    do ig = 1, max(nxgp,nygp,nzgp)
       gridpt(ig) = (ig - 1) * dgp
    enddo
    Atom_loop: do aa=1,natom
       radia=radius(aa)
       if(radia == 0.0D0) cycle Atom_loop
       xaa=x(aa)+tranx
       yaa=y(aa)+trany
       zaa=z(aa)+tranz
       ix=int(xaa/dgp)+1
       iy=int(yaa/dgp)+1
       iz=int(zaa/dgp)+1
       rsqaa=radia+prefac0
       nfil=int(rsqaa/dgp)+1
       rsqaa=rsqaa*rsqaa+rsmall

       jx1=ix-nfil
       if(jx1 < 1) jx1=1
       jx2=ix+nfil
       if(jx2 > nxgp) jx2=nxgp
       jy1=iy-nfil
       if(jy1 < 1) jy1=1
       jy2=iy+nfil
       if(jy2 > nygp) jy2=nygp
       jz1=iz-nfil
       if(jz1 < 1) jz1=1
       jz2=iz+nfil
       if(jz2 > nzgp) jz2=nzgp

       do kg = jz1, jz2
          zaagp = gridpt(kg) - zaa
          zaagp2 = zaagp*zaagp

          do jg = jy1, jy2
             yaagp = gridpt(jg) - yaa
             yaagp2 = yaagp*yaagp + zaagp2

             ig_loop: do ig = jx1, jx2
                xaagp = gridpt(ig) - xaa
                rsq = xaagp*xaagp + yaagp2

                if (rsq <= rsqaa) then
                   nfil = natomingp(ig,jg,kg)
                   ipi = gptoalst(ig,jg,kg)
                   nalst=startalst(ipi)

                   do bb=1,nfil
                      if(gblookup(nalst+bb) == 0) then
                         gblookup(nalst+bb)=aa
                         tmpdist(nalst+bb)=rsq
                         cycle ig_loop
                      endif
                      if(bb == nfil) then
                         write(6,'(a,i3,a)') &
                              'something wrong in GB_LOOKUP'
                         stop
                      endif
                   enddo
                endif
             enddo ig_loop
          enddo
       enddo
    enddo Atom_loop

    !--- sort atom lists in an occpuided grid point according to
    !--- their screening ability of the grid point

    jx1=0
    jx2=0

    do kg = 1, nzgp
       do jg = 1, nygp
          do ig = 1, nxgp
             nfil = natomingp(ig,jg,kg)

             if (nfil < 1) cycle
             if(nfil > nfilmax) then
                call wrndie (0,'<GB_LOOKUP>','nfilmax must be made larger.')
             endif

             jx1=jx1+1

             ipi = gptoalst(ig,jg,kg)
             nalst=startalst(ipi)

             do bb=1,nfil
                aa=gblookup(nalst+bb)
                al(bb)=aa
                radia=radius(aa)+rsasa
                segpar(bb)=tmpdist(nalst+bb)/(radia*radia)
             enddo

             call qcksrt(nfil,segpar,al)

             do bb=1,nfil
                gblookup(nalst+bb)=al(bb)
             enddo

             ! write(50,*)
             ! do bb=1,nfil
             !    aa=gblookup(nalst+bb)
             !    write(50,'(i5,3f10.4)') &
             !         al(bb) !,segpar(bb),radius(al(bb))
             ! enddo

          enddo
       enddo
    enddo

!wi##IF PARALLEL
!wi      CALL IGCOMB(gblookup,nalst0)
!wi##ENDIF

    ! write(*,*) jx1,jx1*1.0/nxgp/nygp/nzgp
    ! write(*,*) jx2,jx2*1.0/nxgp/nygp/nzgp

    return
  end SUBROUTINE GB_LOOKUP2


  SUBROUTINE GB_ROTINVAR(NATOM,X,Y,Z,AMASS,B,EV, &
       XGP,YGP,ZGP,NANG)
    !----------------------------------------------------------------------
    !---  rotate Lebedev integration points along principal axes
    !---  for now, we do not inculde Hydrogen atoms for Inertial Tensor
    !---
    !---  B : principal axes                -> GBPA(3,3)
    !---  EV: eigenvalue of inertial tensor -> GBEV(3)
    !---

  use number
  use stream

    integer ::NATOM,NANG
    real(chm_real)  X(*),Y(*),Z(*),AMASS(*)
    real(chm_real)  XGP(*),YGP(*),ZGP(*)
    real(chm_real)  A(6), B(3,3), EV(3)

    !--- local
    integer ::I
    real(chm_real) XCM,YCM,ZCM, XX,XY,XZ,YY,YZ,ZZ,AMS,TMS
    real(chm_real) SCR(21)


    XCM = ZERO
    YCM = ZERO
    ZCM = ZERO
    TMS = ZERO
    !
    DO I=1,NATOM
       AMS = AMASS(I)
       TMS = TMS + AMS
       XCM = XCM + X(I)*AMS
       YCM = YCM + Y(I)*AMS
       ZCM = ZCM + Z(I)*AMS
    ENDDO

    IF (TMS > ZERO) THEN
       XCM = XCM / TMS
       YCM = YCM / TMS
       ZCM = ZCM / TMS
    ENDIF

    XX = ZERO
    XY = ZERO
    XZ = ZERO
    YY = ZERO
    YZ = ZERO
    ZZ = ZERO

    DO I=1,NATOM
       AMS = AMASS(I)
       XX = XX + AMS*(X(I)-XCM)*(X(I)-XCM)
       XY = XY + AMS*(X(I)-XCM)*(Y(I)-YCM)
       XZ = XZ + AMS*(X(I)-XCM)*(Z(I)-ZCM)
       YY = YY + AMS*(Y(I)-YCM)*(Y(I)-YCM)
       YZ = YZ + AMS*(Y(I)-YCM)*(Z(I)-ZCM)
       ZZ = ZZ + AMS*(Z(I)-ZCM)*(Z(I)-ZCM)
    ENDDO

    A(1) =  YY + ZZ
    A(2) = -XY
    A(3) = -XZ
    A(4) =  XX + ZZ
    A(5) = -YZ
    A(6) =  XX + YY

    ! WRITE (OUTU,105) (A(I),I=1,6)

    !---  diagonalize the matrix
    CALL DIAGQ(3,3,A,B,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
         SCR(16),SCR(19),SCR(1),0)
    !
    ! smallest eigenvalue, EV(1), is principal intertia vector,B(*,1)
    ! secondary vector is B(*,2), tertiary is B(*,3)
    !      WRITE(OUTU,'(/A)') ' Principal Moments of Inertia, amu*A^2'
    !      WRITE (OUTU,105) (EV(I),I=1,3)
    ! 105  FORMAT(' Sorted Eigenvalues: ',3G16.7)
    !      WRITE (OUTU,106) (B(I,1),I=1,3)
    ! 106  FORMAT(' Principal axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
    !      WRITE (OUTU,107) (B(I,2),I=1,3)
    ! 107  FORMAT(' Secondary axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
    !      WRITE (OUTU,108) (B(I,3),I=1,3)
    ! 108  FORMAT(' Tertiary axis,  X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
    !      write (outu,*)

    ! rotate Lebedev integration points
    do I=1,nang
       xgp(I)=B(1,1)*xgpleb(I)+B(1,2)*ygpleb(I)+B(1,3)*zgpleb(I)
       ygp(I)=B(2,1)*xgpleb(I)+B(2,2)*ygpleb(I)+B(2,3)*zgpleb(I)
       zgp(I)=B(3,1)*xgpleb(I)+B(3,2)*ygpleb(I)+B(3,3)*zgpleb(I)
       ! write(*,'(i5,6f10.5)')  &
       !      i,xgpleb(I),ygpleb(I),zgpleb(I),xgp(I),ygp(I),zgp(I)
    enddo

    return
  end SUBROUTINE GB_ROTINVAR


  SUBROUTINE GB_ROTINVAR2(NATOM,X,Y,Z,AMASS,DX,DY,DZ, &
       FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz)
    !------------------------------------------------------------------------
    !---  force contributions due to rotation of integration points
    !---

  use consta
  use number
#if KEY_PARALLEL==1
  use parallel
#endif 

    integer ::NATOM
    real(chm_real)  FSxx,FSxy,FSxz,FSyx,FSyy,FSyz,FSzx,FSzy,FSzz
    real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)  AMASS(*)

    !--- local
    integer ::I
    real(chm_real)  dxM(6),dyM(6),dzM(6),drP(3,3)
    real(chm_real)  OT(3,3),OMO(3,3), &
         dxO(3,3),dyO(3,3),dzO(3,3),OM(3,3)
    real(chm_real)  alpha,beta,gamma
    real(chm_real)  XCM,YCM,ZCM,TMS,XX,XY,XZ,YY,YZ,ZZ,DCM,AMS
    real(chm_real)  dxXX,dxXY,dxXZ,dxYY,dxYZ,dxZZ
    real(chm_real)  dyXX,dyXY,dyXZ,dyYY,dyYZ,dyZZ
    real(chm_real)  dzXX,dzXY,dzXZ,dzYY,dzYZ,dzZZ
    real(chm_real)  xa,ya,za

    !--- tranpose of principal axes matrix
    call TRANSPS(OT,gbpa,3,3)

    !--- center of mass for derivatives of inertial tensor
    XCM = ZERO
    YCM = ZERO
    ZCM = ZERO
    TMS = ZERO

    DO I=1,NATOM
       AMS = AMASS(I)
       TMS = TMS + AMS
       XCM = XCM + X(I)*AMS
       YCM = YCM + Y(I)*AMS
       ZCM = ZCM + Z(I)*AMS
    ENDDO

    IF (TMS > ZERO) THEN
       XCM = XCM / TMS
       YCM = YCM / TMS
       ZCM = ZCM / TMS
    ENDIF

#if KEY_PARALLEL==1
    do I=mynodp,natom,numnod
#else /**/
    do I=1,natom
#endif 
       xa=x(I)
       ya=y(I)
       za=z(I)

       !--- derivatives of inertia tensor M
       AMS = AMASS(I)
       ! XX  = AMS*(Xa-XCM)*(Xa-XCM)
       ! XY  = AMS*(Xa-XCM)*(Ya-YCM)
       ! XZ  = AMS*(Xa-XCM)*(Za-ZCM)
       ! YY  = AMS*(Ya-YCM)*(Ya-YCM)
       ! YZ  = AMS*(Ya-YCM)*(Za-ZCM)
       ! ZZ  = AMS*(Za-ZCM)*(Za-ZCM)

       dCM  = AMS/TMS

       dxXX = AMS*(ONE-dCM)*(Xa-XCM)*TWO
       dxXY = AMS*(ONE-dCM)*(Ya-YCM)
       dxXZ = AMS*(ONE-dCM)*(Za-ZCM)
       dxYY = ZERO
       dxYZ = ZERO
       dxZZ = ZERO

       dyXX = ZERO
       dyXY = AMS*(ONE-dCM)*(Xa-XCM)
       dyXZ = ZERO
       dyYY = AMS*(ONE-dCM)*(Ya-YCM)*TWO
       dyYZ = AMS*(ONE-dCM)*(Za-ZCM)
       dyZZ = ZERO

       dzXX = ZERO
       dzXY = ZERO
       dzXZ = AMS*(ONE-dCM)*(Xa-XCM)
       dzYY = ZERO
       dzYZ = AMS*(ONE-dCM)*(Ya-YCM)
       dzZZ = AMS*(ONE-dCM)*(Za-ZCM)*TWO

       dxM(1) =  dxYY + dxZZ
       dxM(2) = -dxXY
       dxM(3) = -dxXZ
       dxM(4) =  dxXX + dxZZ
       dxM(5) = -dxYZ
       dxM(6) =  dxXX + dxYY

       dyM(1) =  dyYY + dyZZ
       dyM(2) = -dyXY
       dyM(3) = -dyXZ
       dyM(4) =  dyXX + dyZZ
       dyM(5) = -dyYZ
       dyM(6) =  dyXX + dyYY

       dzM(1) =  dzYY + dzZZ
       dzM(2) = -dzXY
       dzM(3) = -dzXZ
       dzM(4) =  dzXX + dzZZ
       dzM(5) = -dzYZ
       dzM(6) =  dzXX + dzYY

       !--- calculate d(xyz)O from O(T) x d(xyz)M x O through Eqs.(16),(18),(19)
       call MULNXNFU(OM,OT,dxM,3)
       call MULNXN(OMO,OM,gbpa,3)
       alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
       beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
       gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
       drP(1,1)= zero
       drP(1,2)= gamma
       drP(1,3)=-beta
       drP(2,1)=-gamma
       drP(2,2)= zero
       drP(2,3)= alpha
       drP(3,1)= beta
       drP(3,2)=-alpha
       drP(3,3)= zero
       ! call MULNXN(dxO,O,drP,3)
       call MULNXN(OM,gbpa,drP,3)
       call MULNXN(dxO,OM,OT,3)

       call MULNXNFU(OM,OT,dyM,3)
       call MULNXN(OMO,OM,gbpa,3)
       alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
       beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
       gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
       drP(1,1)= zero
       drP(1,2)= gamma
       drP(1,3)=-beta
       drP(2,1)=-gamma
       drP(2,2)= zero
       drP(2,3)= alpha
       drP(3,1)= beta
       drP(3,2)=-alpha
       drP(3,3)= zero
       ! call MULNXN(dyO,O,drP,3)
       call MULNXN(OM,gbpa,drP,3)
       call MULNXN(dyO,OM,OT,3)

       call MULNXNFU(OM,OT,dzM,3)
       call MULNXN(OMO,OM,gbpa,3)
       alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
       beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
       gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
       drP(1,1)= zero
       drP(1,2)= gamma
       drP(1,3)=-beta
       drP(2,1)=-gamma
       drP(2,2)= zero
       drP(2,3)= alpha
       drP(3,1)= beta
       drP(3,2)=-alpha
       drP(3,3)= zero
       ! call MULNXN(dzO,O,drP,3)
       call MULNXN(OM,gbpa,drP,3)
       call MULNXN(dzO,OM,OT,3)

       dx(I)=dx(I)+ &
            dxO(1,1)*FSxx+dxO(1,2)*FSxy+dxO(1,3)*FSxz+ &
            dxO(2,1)*FSyx+dxO(2,2)*FSyy+dxO(2,3)*FSyz+ &
            dxO(3,1)*FSzx+dxO(3,2)*FSzy+dxO(3,3)*FSzz
       dy(I)=dy(I)+ &
            dyO(1,1)*FSxx+dyO(1,2)*FSxy+dyO(1,3)*FSxz+ &
            dyO(2,1)*FSyx+dyO(2,2)*FSyy+dyO(2,3)*FSyz+ &
            dyO(3,1)*FSzx+dyO(3,2)*FSzy+dyO(3,3)*FSzz
       dz(I)=dz(I)+ &
            dzO(1,1)*FSxx+dzO(1,2)*FSxy+dzO(1,3)*FSxz+ &
            dzO(2,1)*FSyx+dzO(2,2)*FSyy+dzO(2,3)*FSyz+ &
            dzO(3,1)*FSzx+dzO(3,2)*FSzy+dzO(3,3)*FSzz

       ! write(*,*) dxO(1,1)*FSxx+dxO(1,2)*FSxy+dxO(1,3)*FSxz+  &
       !      dxO(2,1)*FSyx+dxO(2,2)*FSyy+dxO(2,3)*FSyz+  &
       !      dxO(3,1)*FSzx+dxO(3,2)*FSzy+dxO(3,3)*FSzz,  &
       !      dyO(1,1)*FSxx+dyO(1,2)*FSxy+dyO(1,3)*FSxz+  &
       !      dyO(2,1)*FSyx+dyO(2,2)*FSyy+dyO(2,3)*FSyz+  &
       !      dyO(3,1)*FSzx+dyO(3,2)*FSzy+dyO(3,3)*FSzz,  &
       !      dzO(1,1)*FSxx+dzO(1,2)*FSxy+dzO(1,3)*FSxz+  &
       !      dzO(2,1)*FSyx+dzO(2,2)*FSyy+dzO(2,3)*FSyz+  &
       !      dzO(3,1)*FSzx+dzO(3,2)*FSzy+dzO(3,3)*FSzz
       ! WRITE (*,'(3f12.4)') FSxx,FSxy,FSxz
       ! WRITE (*,'(3f12.4)') FSyx,FSyy,FSyz
       ! WRITE (*,'(3f12.4)') FSzx,FSzy,FSzz
       ! WRITE (*,'(3f12.4)') alpha,beta,gamma
       ! WRITE (*,'(3f12.4)') (dzO(1,I),I=1,3)
       ! WRITE (*,'(3f12.4)') (dzO(2,I),I=1,3)
       ! WRITE (*,'(3f12.4)') (dzO(3,I),I=1,3)
       ! WRITE (*,*)

101 enddo

    return
  end SUBROUTINE GB_ROTINVAR2


  SUBROUTINE GB_GvdW0(natom,X,Y,Z,vdWR,epsilon,ITC,IAC,atype, &
       h2oeps,h2or,rmaxint,sw,sgvdw)
    !-----------------------------------------------------------------------
    !---     build a list of GvdW contribution from isolated atom
    !---

  use consta

    integer ::natom
    integer ::ITC(*),IAC(*)
    real(chm_real)  X(*),Y(*),Z(*),vdWR(*),epsilon(*)
    real(chm_real)  h2oeps,h2or,rmaxint,sw,sgvdw
    character(len=*) :: atype(*)

    !--- logcal
    integer :: aa
    real(chm_real)  rho_h2o,prefac0,e,r,r3,rmaxint3,correct

    rho_h2o=0.0336 ! 1/A^3
    prefac0=4.0*pi*(1.0/3.0/sqrt(2.0)+1.0/9.0-1.0)
    !wi      write(*,*) prefac0
    rmaxint3=rmaxint**3

    do aa=1,natom
       e=rho_h2o*sqrt(abs(epsilon(ITC(IAC(aa)))*h2oeps))
       r=vdwr(ITC(IAC(aa)))+h2or
       r3=r*r*r

       correct=4.0*pi*(-r3**4/9.0D0/rmaxint3**3 + &
            r3**2*2.0D0/3.0D0/rmaxint3)

       !wi         write(*,*) correct*e

       gvdwaa(aa)=prefac0*e*r3+correct*e
       gvdweps(aa)=e
       gvdwR(aa)=r3*r3

       bb0(aa)=1.0D0
       bb1(aa)=0.0D0

       if (atype(aa)(1:1) == 'O') then
          if(sw == 0.2D0) bb0(aa)=1.26606D0    !1.35421D0
          if(sw == 0.2D0) bb1(aa)=0.00714834D0 !0.198822D0
          if(sw == 0.3D0) bb0(aa)=1.27142D0    !1.35076D0
          if(sw == 0.3D0) bb1(aa)=-0.0240596D0 !0.150321D0
       endif

       if (atype(aa)(1:1) == 'N') then
          if(sw == 0.2D0) bb0(aa)=1.22666D0    !1.29385D0
          if(sw == 0.2D0) bb1(aa)=-0.0345273D0 !0.213333D0
          if(sw == 0.3D0) bb0(aa)=1.2368D0      !1.3707D0
          if(sw == 0.3D0) bb1(aa)=-0.0862215D0 !0.379616D0
       endif

       if (atype(aa)(1:1) == 'C') then
          if(sw == 0.2D0) bb0(aa)=1.26345D0    !1.24931D0
          if(sw == 0.2D0) bb1(aa)=0.252296D0   !0.153273D0
          if(sw == 0.3D0) bb0(aa)=1.20568D0    !1.26473D0
          if(sw == 0.3D0) bb1(aa)=0.17388D0    !0.128826D0
       endif

       if (atype(aa)(1:1) == 'S') then
          if(sw == 0.2D0) bb0(aa)=1.42532D0    !1.31778D0
          if(sw == 0.2D0) bb1(aa)=0.935019D0   !0.387406D0
          if(sw == 0.3D0) bb0(aa)=1.43119D0    !1.28979
          if(sw == 0.3D0) bb1(aa)=0.859714D0   !0.238872
       endif

       if (atype(aa)(1:1) == 'H') then
          if(sw == 0.2D0) bb0(aa)=1.17807D0    !1.19408D0
          if(sw == 0.2D0) bb1(aa)=0.0403896D0  !0.0288514D0
          if(sw == 0.3D0) bb0(aa)=1.14388D0    !1.20618D0
          if(sw == 0.3D0) bb1(aa)=0.02749D0    !0.0197701D0
       endif

       bb0(aa)=bb0(aa)*sgvdw
       bb1(aa)=bb1(aa)*sgvdw

       if (bb1(aa) == 0.0) write(*,*) atype(aa),bb0(aa),bb1(aa),sw

       !wi         write(*,*) aa,x(aa),y(aa),z(aa),
       !wi     $              vdwr(ITC(IAC(aa))),epsilon(ITC(IAC(aa))),
       !wi     $              gvdwaa(aa),gvdwR(aa)
    enddo

    return
  end SUBROUTINE GB_GvdW0


  SUBROUTINE PBRADIUS(natom,wmain,sw,pbradsf,radius,rminint)
    !-----------------------------------------------------------------------
    !     PB Radii are modified and stored.
    !

    integer ::natom
    real(chm_real)  wmain(*),radius(*)
    real(chm_real)  sw,pbradsf,rminint

    ! logcal
    integer :: aa

    rminint=9999.0D0
    do aa=1,natom
       radius(aa)=(wmain(aa)+sw)*pbradsf
       if(wmain(aa) == 0.0D0) radius(aa)=0.0D0
       if(radius(aa) < rminint) rminint=radius(aa)
    enddo

    return
  end SUBROUTINE PBRADIUS


  SUBROUTINE GAULEG(x1,x2,x,w,n,offset)
    !------------------------------------------------------------------------
    !
  use consta
  use number

    integer ::n,offset
    real(chm_real)  x1,x2,x(*),w(*)

    ! local
    real(chm_real)  eps
    parameter (eps=3.0D-14)

    integer ::i,j,m
    real(chm_real)  p1,p2,p3,pp,xl,xm,z,z1

    m=(n+1)/2
    xm=0.5D0*(x2+x1)
    xl=0.5D0*(x2-x1)
    do i=1,m
       z=cos(pi*(i-0.25D0)/(n+0.5D0))
       zloop: do
          p1=1.0D0
          p2=0.0D0
          do j=1,n
             p3=p2
             p2=p1
             p1=((2.0D0*j-1.0D0)*z*p2-(j-1.0D0)*p3)/j
          enddo
          pp=n*(z*p1-p2)/(z*z-1.0D0)
          z1=z
          z=z1-p1/pp
          if(abs(z-z1) <= EPS) exit zloop
       enddo zloop
       x(i+offset)=xm-xl*z
       x(n+1-i+offset)=xm+xl*z
       w(i+offset)=2.0d0*xl/((1.0D0-z*z)*pp*pp)
       w(n+1-i+offset)=w(i+offset)
    enddo

    return
  end SUBROUTINE GAULEG


  SUBROUTINE GEN_OHC(code, num, x, y, z, w, a, b, v)
    !------------------------------------------------------------------------
    !
    real(chm_real) ::  x(*),y(*),z(*),w(*)
    real(chm_real) ::  a,b,v
    integer ::code
    integer ::num
    real(chm_real) ::  c
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated from C to fortran77 by hand.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    !vw
    !vw    Given a point on a sphere (specified by a and b), generate all
    !vw    the equivalent points under Oh symmetry, making grid points with
    !vw    weight v.
    !vw    The variable num is increased by the number of different points
    !vw    generated.
    !vw
    !vw    Depending on code, there are 6...48 different but equivalent
    !vw    points.
    !vw
    !vw    code=1:   (0,0,1) etc                                (  6 points)
    !vw    code=2:   (0,a,a) etc, a=1/sqrt(2)                   ( 12 points)
    !vw    code=3:   (a,a,a) etc, a=1/sqrt(3)                   (  8 points)
    !vw    code=4:   (a,a,b) etc, b=sqrt(1-2 a^2)               ( 24 points)
    !vw    code=5:   (a,b,0) etc, b=sqrt(1-a^2), a input        ( 24 points)
    !vw    code=6:   (a,b,c) etc, c=sqrt(1-a^2-b^2), a/b input  ( 48 points)
    !vw

    select case(code)

    case(1)
       a=1.0d0
       x(1) =  a
       y(1) =  0.0d0
       z(1) =  0.0d0
       w(1) =  v
       x(2) = -a
       y(2) =  0.0d0
       z(2) =  0.0d0
       w(2) =  v
       x(3) =  0.0d0
       y(3) =  a
       z(3) =  0.0d0
       w(3) =  v
       x(4) =  0.0d0
       y(4) = -a
       z(4) =  0.0d0
       w(4) =  v
       x(5) =  0.0d0
       y(5) =  0.0d0
       z(5) =  a
       w(5) =  v
       x(6) =  0.0d0
       y(6) =  0.0d0
       z(6) = -a
       w(6) =  v
       num=num+6
       return
       !vw
    case(2)
       a=sqrt(0.5d0)
       x( 1) =  0d0
       y( 1) =  a
       z( 1) =  a
       w( 1) =  v
       x( 2) =  0d0
       y( 2) = -a
       z( 2) =  a
       w( 2) =  v
       x( 3) =  0d0
       y( 3) =  a
       z( 3) = -a
       w( 3) =  v
       x( 4) =  0d0
       y( 4) = -a
       z( 4) = -a
       w( 4) =  v
       x( 5) =  a
       y( 5) =  0d0
       z( 5) =  a
       w( 5) =  v
       x( 6) = -a
       y( 6) =  0d0
       z( 6) =  a
       w( 6) =  v
       x( 7) =  a
       y( 7) =  0d0
       z( 7) = -a
       w( 7) =  v
       x( 8) = -a
       y( 8) =  0d0
       z( 8) = -a
       w( 8) =  v
       x( 9) =  a
       y( 9) =  a
       z( 9) =  0d0
       w( 9) =  v
       x(10) = -a
       y(10) =  a
       z(10) =  0d0
       w(10) =  v
       x(11) =  a
       y(11) = -a
       z(11) =  0d0
       w(11) =  v
       x(12) = -a
       y(12) = -a
       z(12) =  0d0
       w(12) =  v
       num=num+12
       return
       !vw
    case (3)
       a = sqrt(1d0/3d0)
       x(1) =  a
       y(1) =  a
       z(1) =  a
       w(1) =  v
       x(2) = -a
       y(2) =  a
       z(2) =  a
       w(2) =  v
       x(3) =  a
       y(3) = -a
       z(3) =  a
       w(3) =  v
       x(4) = -a
       y(4) = -a
       z(4) =  a
       w(4) =  v
       x(5) =  a
       y(5) =  a
       z(5) = -a
       w(5) =  v
       x(6) = -a
       y(6) =  a
       z(6) = -a
       w(6) =  v
       x(7) =  a
       y(7) = -a
       z(7) = -a
       w(7) =  v
       x(8) = -a
       y(8) = -a
       z(8) = -a
       w(8) =  v
       num=num+8
       return
       !vw
    case(4)
       b = sqrt(1d0 - 2d0*a*a)
       x( 1) =  a
       y( 1) =  a
       z( 1) =  b
       w( 1) =  v
       x( 2) = -a
       y( 2) =  a
       z( 2) =  b
       w( 2) =  v
       x( 3) =  a
       y( 3) = -a
       z( 3) =  b
       w( 3) =  v
       x( 4) = -a
       y( 4) = -a
       z( 4) =  b
       w( 4) =  v
       x( 5) =  a
       y( 5) =  a
       z( 5) = -b
       w( 5) =  v
       x( 6) = -a
       y( 6) =  a
       z( 6) = -b
       w( 6) =  v
       x( 7) =  a
       y( 7) = -a
       z( 7) = -b
       w( 7) =  v
       x( 8) = -a
       y( 8) = -a
       z( 8) = -b
       w( 8) =  v
       x( 9) =  a
       y( 9) =  b
       z( 9) =  a
       w( 9) =  v
       x(10) = -a
       y(10) =  b
       z(10) =  a
       w(10) =  v
       x(11) =  a
       y(11) = -b
       z(11) =  a
       w(11) =  v
       x(12) = -a
       y(12) = -b
       z(12) =  a
       w(12) =  v
       x(13) =  a
       y(13) =  b
       z(13) = -a
       w(13) =  v
       x(14) = -a
       y(14) =  b
       z(14) = -a
       w(14) =  v
       x(15) =  a
       y(15) = -b
       z(15) = -a
       w(15) =  v
       x(16) = -a
       y(16) = -b
       z(16) = -a
       w(16) =  v
       x(17) =  b
       y(17) =  a
       z(17) =  a
       w(17) =  v
       x(18) = -b
       y(18) =  a
       z(18) =  a
       w(18) =  v
       x(19) =  b
       y(19) = -a
       z(19) =  a
       w(19) =  v
       x(20) = -b
       y(20) = -a
       z(20) =  a
       w(20) =  v
       x(21) =  b
       y(21) =  a
       z(21) = -a
       w(21) =  v
       x(22) = -b
       y(22) =  a
       z(22) = -a
       w(22) =  v
       x(23) =  b
       y(23) = -a
       z(23) = -a
       w(23) =  v
       x(24) = -b
       y(24) = -a
       z(24) = -a
       w(24) =  v
       num=num+24
       return
       !vw
    case(5)
       b=sqrt(1d0-a*a)
       x( 1) =  a
       y( 1) =  b
       z( 1) =  0d0
       w( 1) =  v
       x( 2) = -a
       y( 2) =  b
       z( 2) =  0d0
       w( 2) =  v
       x( 3) =  a
       y( 3) = -b
       z( 3) =  0d0
       w( 3) =  v
       x( 4) = -a
       y( 4) = -b
       z( 4) =  0d0
       w( 4) =  v
       x( 5) =  b
       y( 5) =  a
       z( 5) =  0d0
       w( 5) =  v
       x( 6) = -b
       y( 6) =  a
       z( 6) =  0d0
       w( 6) =  v
       x( 7) =  b
       y( 7) = -a
       z( 7) =  0d0
       w( 7) =  v
       x( 8) = -b
       y( 8) = -a
       z( 8) =  0d0
       w( 8) =  v
       x( 9) =  a
       y( 9) =  0d0
       z( 9) =  b
       w( 9) =  v
       x(10) = -a
       y(10) =  0d0
       z(10) =  b
       w(10) =  v
       x(11) =  a
       y(11) =  0d0
       z(11) = -b
       w(11) =  v
       x(12) = -a
       y(12) =  0d0
       z(12) = -b
       w(12) =  v
       x(13) =  b
       y(13) =  0d0
       z(13) =  a
       w(13) =  v
       x(14) = -b
       y(14) =  0d0
       z(14) =  a
       w(14) =  v
       x(15) =  b
       y(15) =  0d0
       z(15) = -a
       w(15) =  v
       x(16) = -b
       y(16) =  0d0
       z(16) = -a
       w(16) =  v
       x(17) =  0d0
       y(17) =  a
       z(17) =  b
       w(17) =  v
       x(18) =  0d0
       y(18) = -a
       z(18) =  b
       w(18) =  v
       x(19) =  0d0
       y(19) =  a
       z(19) = -b
       w(19) =  v
       x(20) =  0d0
       y(20) = -a
       z(20) = -b
       w(20) =  v
       x(21) =  0d0
       y(21) =  b
       z(21) =  a
       w(21) =  v
       x(22) =  0d0
       y(22) = -b
       z(22) =  a
       w(22) =  v
       x(23) =  0d0
       y(23) =  b
       z(23) = -a
       w(23) =  v
       x(24) =  0d0
       y(24) = -b
       z(24) = -a
       w(24) =  v
       num=num+24
       return
       !vw
    case (6)
       c=sqrt(1d0 - a*a - b*b)
       x( 1) =  a
       y( 1) =  b
       z( 1) =  c
       w( 1) =  v
       x( 2) = -a
       y( 2) =  b
       z( 2) =  c
       w( 2) =  v
       x( 3) =  a
       y( 3) = -b
       z( 3) =  c
       w( 3) =  v
       x( 4) = -a
       y( 4) = -b
       z( 4) =  c
       w( 4) =  v
       x( 5) =  a
       y( 5) =  b
       z( 5) = -c
       w( 5) =  v
       x( 6) = -a
       y( 6) =  b
       z( 6) = -c
       w( 6) =  v
       x( 7) =  a
       y( 7) = -b
       z( 7) = -c
       w( 7) =  v
       x( 8) = -a
       y( 8) = -b
       z( 8) = -c
       w( 8) =  v
       x( 9) =  a
       y( 9) =  c
       z( 9) =  b
       w( 9) =  v
       x(10) = -a
       y(10) =  c
       z(10) =  b
       w(10) =  v
       x(11) =  a
       y(11) = -c
       z(11) =  b
       w(11) =  v
       x(12) = -a
       y(12) = -c
       z(12) =  b
       w(12) =  v
       x(13) =  a
       y(13) =  c
       z(13) = -b
       w(13) =  v
       x(14) = -a
       y(14) =  c
       z(14) = -b
       w(14) =  v
       x(15) =  a
       y(15) = -c
       z(15) = -b
       w(15) =  v
       x(16) = -a
       y(16) = -c
       z(16) = -b
       w(16) =  v
       x(17) =  b
       y(17) =  a
       z(17) =  c
       w(17) =  v
       x(18) = -b
       y(18) =  a
       z(18) =  c
       w(18) =  v
       x(19) =  b
       y(19) = -a
       z(19) =  c
       w(19) =  v
       x(20) = -b
       y(20) = -a
       z(20) =  c
       w(20) =  v
       x(21) =  b
       y(21) =  a
       z(21) = -c
       w(21) =  v
       x(22) = -b
       y(22) =  a
       z(22) = -c
       w(22) =  v
       x(23) =  b
       y(23) = -a
       z(23) = -c
       w(23) =  v
       x(24) = -b
       y(24) = -a
       z(24) = -c
       w(24) =  v
       x(25) =  b
       y(25) =  c
       z(25) =  a
       w(25) =  v
       x(26) = -b
       y(26) =  c
       z(26) =  a
       w(26) =  v
       x(27) =  b
       y(27) = -c
       z(27) =  a
       w(27) =  v
       x(28) = -b
       y(28) = -c
       z(28) =  a
       w(28) =  v
       x(29) =  b
       y(29) =  c
       z(29) = -a
       w(29) =  v
       x(30) = -b
       y(30) =  c
       z(30) = -a
       w(30) =  v
       x(31) =  b
       y(31) = -c
       z(31) = -a
       w(31) =  v
       x(32) = -b
       y(32) = -c
       z(32) = -a
       w(32) =  v
       x(33) =  c
       y(33) =  a
       z(33) =  b
       w(33) =  v
       x(34) = -c
       y(34) =  a
       z(34) =  b
       w(34) =  v
       x(35) =  c
       y(35) = -a
       z(35) =  b
       w(35) =  v
       x(36) = -c
       y(36) = -a
       z(36) =  b
       w(36) =  v
       x(37) =  c
       y(37) =  a
       z(37) = -b
       w(37) =  v
       x(38) = -c
       y(38) =  a
       z(38) = -b
       w(38) =  v
       x(39) =  c
       y(39) = -a
       z(39) = -b
       w(39) =  v
       x(40) = -c
       y(40) = -a
       z(40) = -b
       w(40) =  v
       x(41) =  c
       y(41) =  b
       z(41) =  a
       w(41) =  v
       x(42) = -c
       y(42) =  b
       z(42) =  a
       w(42) =  v
       x(43) =  c
       y(43) = -b
       z(43) =  a
       w(43) =  v
       x(44) = -c
       y(44) = -b
       z(44) =  a
       w(44) =  v
       x(45) =  c
       y(45) =  b
       z(45) = -a
       w(45) =  v
       x(46) = -c
       y(46) =  b
       z(46) = -a
       w(46) =  v
       x(47) =  c
       y(47) = -b
       z(47) = -a
       w(47) =  v
       x(48) = -c
       y(48) = -b
       z(48) = -a
       w(48) =  v
       num=num+48
       return
    case default
       write (6,*) 'Gen_Oh: Invalid Code'
       call wrndie(-4,"<gbsw.src>GEN_OHC","Invalid code")
    end select

    return
  end SUBROUTINE GEN_OHC


  subroutine ld0026c(x,y,z,w,n)
    real(chm_real) ::  x(  26)
    real(chm_real) ::  y(  26)
    real(chm_real) ::  z(  26)
    real(chm_real) ::  w(  26)
    integer ::n
    real(chm_real) ::  a,b,v
    !VW
    !VW    LEBEDEV   26-POINT ANGULAR GRID
    !VW
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated using a C to fortran77 conversion
    !hvd   tool written by Dr. Christoph van Wuellen.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    n=1
    v=0.4761904761904762d-1
    call gen_ohc( 1, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.3809523809523810d-1
    call gen_ohc( 2, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.3214285714285714d-1
    call gen_ohc( 3, n, x(n), y(n), z(n), w(n), a, b, v)
    n=n-1
    return
  end subroutine ld0026c


  subroutine ld0038c(x,y,z,w,n)
    real(chm_real) ::  x(  38)
    real(chm_real) ::  y(  38)
    real(chm_real) ::  z(  38)
    real(chm_real) ::  w(  38)
    integer ::n
    real(chm_real) ::  a,b,v
    !VW
    !VW    LEBEDEV   38-POINT ANGULAR GRID
    !VW
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated using a C to fortran77 conversion
    !hvd   tool written by Dr. Christoph van Wuellen.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    n=1
    v=0.9523809523809524d-2
    call gen_ohc( 1, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.3214285714285714d-1
    call gen_ohc( 3, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4597008433809831d+0
    v=0.2857142857142857d-1
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    n=n-1
    return
  end subroutine ld0038c


  subroutine ld0050c(x,y,z,w,n)
    real(chm_real) ::  x(  50)
    real(chm_real) ::  y(  50)
    real(chm_real) ::  z(  50)
    real(chm_real) ::  w(  50)
    integer ::n
    real(chm_real) ::  a,b,v
    !VW
    !VW    LEBEDEV   50-POINT ANGULAR GRID
    !VW
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated using a C to fortran77 conversion
    !hvd   tool written by Dr. Christoph van Wuellen.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    n=1
    v=0.1269841269841270d-1
    call gen_ohc( 1, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.2257495590828924d-1
    call gen_ohc( 2, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.2109375000000000d-1
    call gen_ohc( 3, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3015113445777636d+0
    v=0.2017333553791887d-1
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    n=n-1
    return
  end subroutine ld0050c


  subroutine ld0110c(x,y,z,w,n)
    real(chm_real) ::  x( 110)
    real(chm_real) ::  y( 110)
    real(chm_real) ::  z( 110)
    real(chm_real) ::  w( 110)
    integer ::n
    real(chm_real) ::  a,b,v
    !VW
    !VW    LEBEDEV  110-POINT ANGULAR GRID
    !VW
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated using a C to fortran77 conversion
    !hvd   tool written by Dr. Christoph van Wuellen.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    n=1
    v=0.3828270494937162d-2
    call gen_ohc( 1, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.9793737512487512d-2
    call gen_ohc( 3, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1851156353447362d+0
    v=0.8211737283191111d-2
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6904210483822922d+0
    v=0.9942814891178103d-2
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3956894730559419d+0
    v=0.9595471336070963d-2
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4783690288121502d+0
    v=0.9694996361663028d-2
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    n=n-1
    return
  end subroutine ld0110c


  subroutine ld2030c(x,y,z,w,n)
    real(chm_real) ::  x(2030)
    real(chm_real) ::  y(2030)
    real(chm_real) ::  z(2030)
    real(chm_real) ::  w(2030)
    integer ::n
    real(chm_real) ::  a,b,v
    !VW
    !VW    LEBEDEV 2030-POINT ANGULAR GRID
    !VW
    !hvd
    !hvd   This subroutine is part of a set of subroutines that generate
    !hvd   Lebedev grids [1-6] for integration on a sphere. The original
    !hvd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
    !hvd   translated into fortran by Dr. Christoph van Wuellen.
    !hvd   This subroutine was translated using a C to fortran77 conversion
    !hvd   tool written by Dr. Christoph van Wuellen.
    !hvd
    !hvd   Users of this code are asked to include reference [1] in their
    !hvd   publications, and in the user- and programmers-manuals
    !hvd   describing their codes.
    !hvd
    !hvd   This code was distributed through CCL (http://www.ccl.net/).
    !hvd
    !hvd   [1] V.I. Lebedev, and D.N. Laikov
    !hvd       "A quadrature formula for the sphere of the 131st
    !hvd        algebraic order of accuracy"
    !hvd       Doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
    !hvd
    !hvd   [2] V.I. Lebedev
    !hvd       "A quadrature formula for the sphere of 59th algebraic
    !hvd        order of accuracy"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 50, 1995, pp. 283-286.
    !hvd
    !hvd   [3] V.I. Lebedev, and A.L. Skorokhodov
    !hvd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
    !hvd       Russian Acad. Sci. Dokl. Math., Vol. 45, 1992, pp. 587-592.
    !hvd
    !hvd   [4] V.I. Lebedev
    !hvd       "Spherical quadrature formulas exact to orders 25-29"
    !hvd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
    !hvd
    !hvd   [5] V.I. Lebedev
    !hvd       "Quadratures on a sphere"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 16,
    !hvd       1976, pp. 10-24.
    !hvd
    !hvd   [6] V.I. Lebedev
    !hvd       "Values of the nodes and weights of ninth to seventeenth
    !hvd        order Gauss-Markov quadrature formulae invariant under the
    !hvd        octahedron group with inversion"
    !hvd       Computational Mathematics and Mathematical Physics, Vol. 15,
    !hvd       1975, pp. 44-51.
    !hvd
    n=1
    v=0.4656031899197431d-4
    call gen_ohc( 1, n, x(n), y(n), z(n), w(n), a, b, v)
    v=0.5421549195295507d-3
    call gen_ohc( 3, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2540835336814348d-1
    v=0.1778522133346553d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6399322800504915d-1
    v=0.2811325405682796d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1088269469804125d+0
    v=0.3548896312631459d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1570670798818287d+0
    v=0.4090310897173364d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2071163932282514d+0
    v=0.4493286134169965d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2578914044450844d+0
    v=0.4793728447962723d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3085687558169623d+0
    v=0.5015415319164265d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3584719706267024d+0
    v=0.5175127372677937d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4070135594428709d+0
    v=0.5285522262081019d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4536618626222638d+0
    v=0.5356832703713962d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4979195686463577d+0
    v=0.5397914736175170d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5393075111126999d+0
    v=0.5416899441599930d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6115617676843916d+0
    v=0.5419308476889938d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6414308435160159d+0
    v=0.5416936902030596d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6664099412721607d+0
    v=0.5419544338703164d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6859161771214913d+0
    v=0.5428983656630975d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6993625593503890d+0
    v=0.5442286500098193d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.7062393387719380d+0
    v=0.5452250345057301d-3
    call gen_ohc( 4, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.7479028168349763d-1
    v=0.2568002497728530d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1848951153969366d+0
    v=0.3827211700292145d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3059529066581305d+0
    v=0.4579491561917824d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4285556101021362d+0
    v=0.5042003969083574d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5468758653496526d+0
    v=0.5312708889976025d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6565821978343439d+0
    v=0.5438401790747117d-3
    call gen_ohc( 5, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1253901572367117d+0
    b=0.3681917226439641d-1
    v=0.3316041873197344d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.1775721510383941d+0
    b=0.7982487607213301d-1
    v=0.3899113567153771d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2305693358216114d+0
    b=0.1264640966592335d+0
    v=0.4343343327201309d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2836502845992063d+0
    b=0.1751585683418957d+0
    v=0.4679415262318919d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3361794746232590d+0
    b=0.2247995907632670d+0
    v=0.4930847981631031d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3875979172264824d+0
    b=0.2745299257422246d+0
    v=0.5115031867540091d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4374019316999074d+0
    b=0.3236373482441118d+0
    v=0.5245217148457367d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4851275843340022d+0
    b=0.3714967859436741d+0
    v=0.5332041499895321d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5303391803806868d+0
    b=0.4175353646321745d+0
    v=0.5384583126021542d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5726197380596287d+0
    b=0.4612084406355461d+0
    v=0.5411067210798852d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.2431520732564863d+0
    b=0.4258040133043952d-1
    v=0.4259797391468714d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3002096800895869d+0
    b=0.8869424306722721d-1
    v=0.4604931368460021d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3558554457457432d+0
    b=0.1368811706510655d+0
    v=0.4871814878255202d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4097782537048887d+0
    b=0.1860739985015033d+0
    v=0.5072242910074885d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4616337666067458d+0
    b=0.2354235077395853d+0
    v=0.5217069845235350d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5110707008417874d+0
    b=0.2842074921347011d+0
    v=0.5315785966280310d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5577415286163795d+0
    b=0.3317784414984102d+0
    v=0.5376833708758905d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6013060431366950d+0
    b=0.3775299002040700d+0
    v=0.5408032092069521d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.3661596767261781d+0
    b=0.4599367887164592d-1
    v=0.4842744917904866d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4237633153506581d+0
    b=0.9404893773654421d-1
    v=0.5048926076188130d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4786328454658452d+0
    b=0.1431377109091971d+0
    v=0.5202607980478373d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5305702076789774d+0
    b=0.1924186388843570d+0
    v=0.5309932388325743d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5793436224231788d+0
    b=0.2411590944775190d+0
    v=0.5377419770895208d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6247069017094747d+0
    b=0.2886871491583605d+0
    v=0.5411696331677717d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.4874315552535204d+0
    b=0.4804978774953206d-1
    v=0.5197996293282420d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5427337322059053d+0
    b=0.9716857199366665d-1
    v=0.5311120836622945d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.5943493747246700d+0
    b=0.1465205839795055d+0
    v=0.5384309319956951d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6421314033564943d+0
    b=0.1953579449803574d+0
    v=0.5421859504051886d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6020628374713980d+0
    b=0.4916375015738108d-1
    v=0.5390948355046314d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    a=0.6529222529856881d+0
    b=0.9861621540127005d-1
    v=0.5433312705027845d-3
    call gen_ohc( 6, n, x(n), y(n), z(n), w(n), a, b, v)
    n=n-1
    return
  end subroutine ld2030c
end module gbsw

