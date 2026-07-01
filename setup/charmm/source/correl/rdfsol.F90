module rdfsol_mod
  use chm_kinds
  use dimens_fcm
  implicit none


contains
#if KEY_RDFSOL==0 /*rdfsol*/
  SUBROUTINE RDFSOL
    CALL WRNDIE(-1,'<RDFSOL>','RDFSOL is not currently compiled.')
    RETURN
  end SUBROUTINE RDFSOL

#else /*         (rdfsol)*/


  SUBROUTINE RDFSOL
    !
    !     this is the new ANACOR module to replace the old
    !     "organically grown" one with a new one for our analyses
    !     computes radial distribution functions for solvent molecules
    !
    !     TODOS:
    !     - if A or B is a set of 1 atom -> treat as point
    !     - if we want SAME but limit set A but not B -> do something
    !
    !     Tibor Rudas Nov 2002 - Jun 2003
    !
    !     modifications
    !     - change from XDIP and only calculating one property at a
    !       time to logical vars and the possibility to calculate
    !       multiple functions at a time
    !     - allow prototypes for subunits to be specified for both sets
    !
    use number
    use bases_fcm
    use comand
    use consta
    use energym
    use image
    use image_routines_module, only: transo
    use coord
    use psf
    use stream
    use string
    use cvio,only: trjspc
#if KEY_PROTO==1
    use proto_mod     
#endif
    use memory
    implicit none
    !
    INTEGER NSOLA
    integer,pointer,dimension(:) :: HPSOLA,hpsolb,hpsolu,hpsolv,hpsolx
    INTEGER NSOLB
    INTEGER NSOLU
    INTEGER NSOLV
    INTEGER NSOLX
    INTEGER NCORR,NBIN,IOUT
    INTEGER NUNIT,FIRSTU,BEGIN,SKIP,STOP
    real(chm_real) RMAX,VOL,XREF,YREF,ZREF,QREF,RARND
    real(chm_real) XAREF,YAREF,ZAREF
    INTEGER SPFAC,SELPOS
    INTEGER XASITE,XARND,NSP
    LOGICAL QBRUTE,QAWAT,QBWAT,QSAME,QLOC
    LOGICAL QARND,QASITE,QBSITE,QPREC,DOIMAGES,QIMMAX

    integer,allocatable,dimension(:),target :: space_st,space_hp
    integer,pointer,dimension(:) :: STISEL,STJSEL,STXSEL
    real(chm_real),allocatable,dimension(:),target :: space_hp2
    real(chm_real),pointer,dimension(:) :: hpax,hpay,hpaz

    !
    INTEGER IRDF,IDDIP,IQDIP,IHD,IUSER
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER
    LOGICAL QAPROT,QBPROT,QMASS
    INTEGER NAPROT,NBPROT
    !
    character(len=4) WRD
    real(chm_real),allocatable,dimension(:) :: hpg
    real(chm_real4),allocatable,dimension(:) :: hpmini
    INTEGER SPG
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    INTEGER NCALC
    !
    !
    !     are we doing images?
    DOIMAGES=.FALSE.
    QIMMAX=.FALSE.
    NSP=NATOM
    IF(NATIM > NATOM) THEN
       !        toggle IMMAx: if set means CUTIM is chosen sufficiently
       !               large (by the user!) to include all atoms in all
       !               transformations i.e. we can skip the "image-picking"
       !               for each frame and only generate coordinates
       IF(INDXA(COMLYN,COMLEN,'IMMA') > 0) THEN
          QIMMAX=.TRUE.
          WRITE(OUTU,'(A)') 'Using IMMAx -> not updating images'
       ENDIF
       DOIMAGES=.TRUE.
       NSP=NATIM
    ENDIF
    !
    !     do we want rdf analysis(=1) or dipoledipole(=2) or charge-dipole(=3)
    !     or h_D(=4)
    QRDF=.FALSE.
    QDDIP=.FALSE.
    QQDIP=.FALSE.
    QHD=.FALSE.
    QUSER=.FALSE.
    !     get output unit (write to current unit if none)
    IOUT=-1
    IOUT=GTRMI(COMLYN,COMLEN,'IOUT',IOUT)
    IF(IOUT > 0) THEN
       !        parse old style syntax
       IF(INDXA(COMLYN,COMLEN,'RDF ') > 0) THEN
          QRDF=.TRUE.
          IRDF=IOUT
       ELSEIF(INDXA(COMLYN,COMLEN,'DDIP') > 0) THEN
          QDDIP=.TRUE.
          IDDIP=IOUT
       ELSEIF(INDXA(COMLYN,COMLEN,'QDIP') > 0) THEN
          QQDIP=.TRUE.
          IQDIP=IOUT
       ELSEIF(INDXA(COMLYN,COMLEN,' HD') > 0) THEN
          QHD=.TRUE.
          IHD=IOUT
       ELSEIF(INDXA(COMLYN,COMLEN,'USER') > 0) THEN
          QUSER=.TRUE.
          IUSER=IOUT
       ENDIF
    ELSE
       !        parse new style syntax (analyze functions simultaneously)
       IRDF=0
       IRDF=GTRMI(COMLYN,COMLEN,'RDF ',IRDF)
       IF(IRDF > 0) QRDF=.TRUE.
       IDDIP=0
       IDDIP=GTRMI(COMLYN,COMLEN,'DDIP',IDDIP)
       IF(IDDIP > 0) QDDIP=.TRUE.
       IQDIP=0
       IQDIP=GTRMI(COMLYN,COMLEN,'QDIP',IQDIP)
       IF(IQDIP > 0) QQDIP=.TRUE.
       IHD=0
       IHD=GTRMI(COMLYN,COMLEN,' HD',IHD)
       IF(IHD > 0) QHD=.TRUE.
       IUSER=0
       IUSER=GTRMI(COMLYN,COMLEN,'USER',IUSER)
       IF(IUSER > 0) QUSER=.TRUE.
    ENDIF
    !
    !     do some parsing:
    !     get number of correlations (unused yet)
    NCORR=511
    NCORR=GTRMI(COMLYN,COMLEN,'NCOR',NCORR)
    !     get number of bins (spacing)
    NBIN=150
    NBIN=GTRMI(COMLYN,COMLEN,'NBIN',NBIN)
    !     get max R to evaluate
    RMAX=7.5
    RMAX=GTRMF(COMLYN,COMLEN,'RMAX',RMAX)
    !     get space factor (to multiply ntima by to allow for varying img nr)
    !     default to 3 since we might have to generate all images which are
    !     approximately twice as much as usually generated (and sometimes more...)
    SPFAC=3
    SPFAC=GTRMI(COMLYN,COMLEN,'SPFA',SPFAC)
    !     do we want explicit minimum image convention (useful if RMAX > L/2)
    QMINI=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'MINI') > 0) QMINI=.TRUE.
    !     are both sets the same?
    QSAME=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'SAME') > 0) QSAME=.TRUE.
    !     brute force or cubing(=default)
    QBRUTE=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'BRUT') > 0) QBRUTE=.TRUE.
    !     do we want precise or quick&d(=default) water-water function
    QPREC=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'PREC') > 0) QPREC=.TRUE.
    !     does the user supply us with a volume
    VOL=GTRMF(COMLYN,COMLEN,'VOLU',ZERO)
    !     mass weighting for charged moieties?
    QMASS=.FALSE.
    IF(INDXA(COMLYN,COMLEN,'MASS') > 0) QMASS=.TRUE.
    !
    !     check parameters (up to now) for sanity
    IF(NBIN <= 0) THEN
       WRITE(OUTU,'(A)') &
            ' *****  WARNING  ***** NBIN <= 0 --> resetting to 150'
       NBIN=150
    ENDIF
    IF(RMAX <= ZERO) THEN
       WRITE(OUTU,'(A)') &
            ' *****  WARNING  ***** RMAX <= 0.0 --> resetting to 7.5'
       RMAX=7.5D0
    ENDIF
    !
    NCALC=0
    IF(QRDF)  NCALC=NCALC+1
    IF(QDDIP) NCALC=NCALC+1
    IF(QQDIP) NCALC=NCALC+1
    IF(QHD)   NCALC=NCALC+1
    IF(QUSER) NCALC=NCALC+1
    IF(NCALC <= 0) CALL WRNDIE(-3,'<RDFSOL>', &
         'No function requested...')
    IF(QMINI.AND.(NCALC > 1)) CALL WRNDIE(-5,'<RDFSOL>', &
         'MINI can only be used for the calculation of one property.')
    !
    !     initialize traj specs
    CALL TRJSPC(COMLYN,COMLEN,NUNIT,FIRSTU,BEGIN,SKIP,STOP)
    !
    !----------------------------------------------------------------------
    !
    !     allocate space
    !     (allow for nr. of imgs. to change during run)
    IF(DOIMAGES) NSP=NSP*SPFAC
    call chmalloc('rdfsol.src','RDFSOL','space_ST',3*NSP,intg=space_ST)
    STISEL => space_st(0*nsp+1 : 1*nsp)
    STJSEL => space_st(1*nsp+1 : 2*nsp)
    STXSEL => space_st(2*nsp+1 : 3*nsp)
    call chmalloc('rdfsol.src','RDFSOL','HPSOLA',NSP,intgp=HPSOLA)
    call chmalloc('rdfsol.src','RDFSOL','HPSOLB',NSP,intgp=HPSOLB)
    call chmalloc('rdfsol.src','RDFSOL','HPSOLU',NSP,intgp=HPSOLU)
    call chmalloc('rdfsol.src','RDFSOL','HPSOLV',NSP,intgp=HPSOLV)
    call chmalloc('rdfsol.src','RDFSOL','HPSOLX',NSP,intgp=HPSOLX)
    !
    !     alternative "coordinates" and dipole moment
    call chmalloc('rdfsol.src','RDFSOL','space_hp2',3*Nsp,crl=space_hp2)
    HPAX  => space_hp2(0*nsp+1 : 1*nsp)
    HPAY  => space_hp2(1*nsp+1 : 2*nsp)
    HPAZ  => space_hp2(2*nsp+1 : 3*nsp)
    !
    !----- prepare data structures ----------------------------------------
    !
    !     G(i,r) holds the single functions
    !     a) for water-water rdfs: i=1->gOO, i=2->gOH and i=3->gHH
    !     b) for site/ensemble-water rdfs:
    !                              i=1->gXO, i=2->gXH and i=3->unused
    !     c) for site/ensemble-ensemble-rdfs:
    !                              i=1->gXY,  i=2,3->unused
    !     d) for water-water dipole-dipole fncs:
    !                              i=4->h_mu_mu, i=5->Gk(r)
    !     e) for dipole-water dipole-dipole fncs:
    !                              i=4->h_mu_mu, i=5->"Gk(r)?"
    !     f) for charge-water charge-dipole fncs:
    !                              i=6=h_q_mu
    !     g) for "h_\Delta" fnc:
    !                              i=7=h_\Delta
    !     h) for a user defined function:
    !                              i=8=USERRDF
    !
    SPG=8*(NBIN+1)
    call chmalloc('rdfsol.src','RDFSOL','HPG',SPG,crl=HPG)
    !
    !----------------------------------------------------------------------
    !     initialize
    stisel(1:natom)=0
    stjsel(1:natom)=0
    stxsel(1:natom)=0
    hpg(1:spg)=zero
    !
    !----------------------------------------------------------------------
    !
    LNMINI=1
    IF(QMINI) THEN
       LNMINI=NATOM
    ENDIF
    !     allocate anyway so that we have somthing to pass (just 1 real(chm_real) long
    !     if not used...)
    LNMIN2=LNMINI*LNMINI
    call chmalloc('rdfsol.src','RDFSOL','HPMINI',LNMIN2,cr4=HPMINI)
    !
    !----------------------------------------------------------------------
    !
    !     now comes the hard part:
    !     we need to decide:
    !     a) what are the two sets to iterate over:
    !        set A can be:
    !        ai)   water
    !        aii)  points in space
    !        aiii) points in space - c.o.m.        (1 atom is also a c.o.m. site!)
    !        aiv)  prototype
    !     b) set B can b:
    !        bi)   same as set A
    !        bii)  water
    !        biii) points in space
    !        biv)  points in space - c.o.m.
    !        bv)   prototype
    !     c) do we want properties to be calculated for all points or should
    !        one or both sets limited around:
    !        ci)   a point in space
    !        cii)  an atom
    !        ciii) a c.o.m.
    !
    NSOLA=0
    NSOLB=0
    NSOLU=0
    NSOLV=0
    NSOLX=0
    !
    QARND=.FALSE.
    QLOC=.FALSE.
    XARND=0
    !
    QASITE=.FALSE.
    XASITE=0
    QBSITE=.FALSE.
    !
    QAWAT=.FALSE.
    QBWAT=.FALSE.
    !
    QAPROT=.FALSE.
    QBPROT=.FALSE.
    NAPROT=0
    NBPROT=0
    !
100 CONTINUE
    !
    WRD=NEXTA4(COMLYN,COMLEN)
    !
    IF(WRD == 'AROU') THEN
       !        so we want to limit calculation of function around somewhere
       QARND=.TRUE.
       !        up to how far should we go
       RARND=7.5
       RARND=GTRMF(COMLYN,COMLEN,'RARO',RARND)
       !        does only the first set need to be around the site or both points
       !             (QLOC=.TRUE. means both)
       IF(INDXA(COMLYN,COMLEN,'LOCA') > 0) QLOC=.TRUE.
       !        next should be the selection of the site
       WRD=NEXTA4(COMLYN,COMLEN)
       !        put it back!
       CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
       IF((WRD == 'XREF').OR.(WRD == 'YREF').OR.(WRD == 'ZREF')) THEN
          XARND=1
          XAREF=GTRMF(COMLYN,COMLEN,'XREF',ZERO)
          YAREF=GTRMF(COMLYN,COMLEN,'YREF',ZERO)
          ZAREF=GTRMF(COMLYN,COMLEN,'ZREF',ZERO)
       ELSE IF(WRD == 'SELE') THEN
          !           i.e. c.o.m. site requested
          XARND=2
          CALL SHLSEL(COMLYN,COMLEN,HPSOLX,NSOLX,STXSEL)
       ELSE
          WRITE(OUTU,'(A,A)') &
               ' *** Around: No ref-site and no selection ', &
               '-> assuming (0 0 0)'
          XARND=1
          XAREF=ZERO
          YAREF=ZERO
          ZAREF=ZERO
       ENDIF
    ELSE IF(WRD == 'SETA') THEN
       !        i.e. we get to know what set A is
       WRD=NEXTA4(COMLYN,COMLEN)
       IF((WRD == 'XREF').OR.(WRD == 'YREF').OR.(WRD.EQ.'ZREF')) THEN
          !           i.e. fixed point in space requested
          !           put it back!
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
          QASITE=.TRUE.
          XASITE=1
          XREF=GTRMF(COMLYN,COMLEN,'XREF',ZERO)
          YREF=GTRMF(COMLYN,COMLEN,'YREF',ZERO)
          ZREF=GTRMF(COMLYN,COMLEN,'ZREF',ZERO)
          QREF=GTRMF(COMLYN,COMLEN,'QREF',ZERO)
       ELSE IF(WRD == 'SITE') THEN
          QASITE=.TRUE.
          XASITE=2
          WRD=NEXTA4(COMLYN,COMLEN)
          IF(WRD == 'SELE') THEN
             !              i.e. c.o.m. site requested
             !              put it back!
             CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
             XARND=2
             CALL SHLSEL(COMLYN,COMLEN,HPSOLA, &
                  NSOLA,STISEL)
          ELSE
             !              i.e. set A should be a c.o.m. site but no atom sel given
             !                   -> fall back to fixed point in space (0/0/0)
             WRITE(OUTU,'(A,A)') &
                  ' *** Site A: Site and no selection -> assuming (0 0 0)'
             XASITE=1
             XREF=ZERO
             YREF=ZERO
             ZREF=ZERO
             QREF=ZERO
          ENDIF
       ELSE IF(WRD == 'SELE') THEN
          !           i.e. multiple sites requested
          !           put it back!
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
          CALL SHLSEL(COMLYN,COMLEN,hpsola,NSOLA,STISEL)
       ELSE IF(WRD == 'WATE') THEN
          !           water
          QAWAT=.TRUE.
       ELSE IF(WRD == 'PROT') THEN
          !           a prototype
#if KEY_PROTO==1 /*proto*/
          QAPROT=.TRUE.
          NAPROT=NEXTI(COMLYN,COMLEN)
          IF((NAPROT < 0).OR.(NAPROT > MXPRTO)) THEN
             CALL WRNDIE(-3,'<RDFSOL>','Wrong prototype number')
             NAPROT=1
          ENDIF
          IF(.NOT.QPRTO(NAPROT)) THEN
             CALL WRNDIE(-3,'<RDFSOL>', &
                  'Prototype not in use - ignoring')
             QAPROT=.FALSE.
          ENDIF
          IF(QAPROT) CALL WSOLPRT(NAPROT,NSOLA,hpsola)
#else /* (proto)*/
          CALL WRNDIE(-1,'<RDFSOL>','PROTO is not currently compiled')
          RETURN
#endif /* (proto)*/
       ELSE
          !           i.e. we must have landed in the next section without anything
          !           put it back!
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
          WRITE(OUTU,'(A,A)') &
               ' *** Site A: No ref-site, no selection and not WATEr', &
               ' -> assuming WATEr'
          QAWAT=.TRUE.
       ENDIF
    ELSE IF(WRD == 'SETB') THEN
       !        i.e. we get to know what set B is
       WRD=NEXTA4(COMLYN,COMLEN)
       IF(WRD == 'SELE') THEN
          !           i.e. multiple site requested
          !           put it back!
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
          CALL SHLSEL(COMLYN,COMLEN,hpsolb,NSOLB,STJSEL)
       ELSE IF(WRD == 'SITE') THEN
          QBSITE=.TRUE.
          WRD=NEXTA4(COMLYN,COMLEN)
          IF(WRD == 'SELE') THEN
             !              i.e. c.o.m. site requested
             !              put it back!
             CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
             CALL SHLSEL(COMLYN,COMLEN,hpsolb, &
                  NSOLB,STJSEL)
          ELSE
             !              this must be wrong! B can only be a site of atoms!
             CALL WRNDIE(-5,'<RDFSOL>', &
                  'Site B cannot fixed in space (change the src).')
          ENDIF
       ELSE IF(WRD == 'WATE') THEN
          !           water
          QBWAT=.TRUE.
       ELSE IF(WRD == 'PROT') THEN
          !           a prototype
#if KEY_PROTO==1 /*proto*/
          QBPROT=.TRUE.
          NBPROT=NEXTI(COMLYN,COMLEN)
          IF((NBPROT < 0).OR.(NBPROT > MXPRTO)) THEN
             CALL WRNDIE(-3,'<RDFSOL>','Wrong prototype number')
             NBPROT=1
          ENDIF
          IF(.NOT.QPRTO(NBPROT)) THEN
             CALL WRNDIE(-3,'<RDFSOL>', &
                  'Prototype not in use - ignoring')
             QBPROT=.FALSE.
          ENDIF
          IF(QBPROT) CALL WSOLPRT(NBPROT,NSOLB,hpsolb)
#else /* (proto)*/
          CALL WRNDIE(-1,'<RDFSOL>','PROTO is not currently compiled')
          RETURN
#endif /* (proto)*/
       ELSE
          !           i.e. we must have landed in the next section without anything
          !           put it back!
          CALL JOINWD(COMLYN,MXCMSZ,COMLEN,WRD,4)
          WRITE(OUTU,'(A,A)') &
               ' *** Site B: No ref-site, no selection and not WATEr', &
               ' -> assuming WATEr'
          QBWAT=.TRUE.
       ENDIF
    ELSE IF(WRD == '    ') THEN
       !        nothing -> so do nothing
       CONTINUE
    ELSE
       WRITE(OUTU,'(3A)') ' Found >', WRD,'<'
       CALL WRNDIE(-5,'<RDFSOL>', &
            'Unknown keyword found.')
    ENDIF
    IF(COMLEN > 0) GOTO 100
    !
    IF((NSOLA == 0).AND.(.NOT.QAWAT).AND.(.NOT.QASITE) &
         .AND.(.NOT.QAPROT)) THEN
       WRITE(OUTU,'(A,A)') &
            ' *** Site A: No definition given -> assuming WATEr'
       QAWAT=.TRUE.
    ENDIF
    IF((.NOT.QSAME).AND.(NSOLB == 0).AND.(.NOT.QBWAT) &
         .AND.(.NOT.QBPROT)) THEN
       WRITE(OUTU,'(A,A)') &
            ' *** Site B: No definition given -> assuming WATEr'
       QBWAT=.TRUE.
    ENDIF
    !
    !     TODO: if we want SAME but limit set A but not B -> do something
    !
    !     now select water sets if requested
    IF(QAWAT) &
         CALL SELWAT(NSP,NSOLA,hpsola,STISEL)
    IF(QBWAT) &
         CALL SELWAT(NSP,NSOLB,hpsolb,STJSEL)
    !
    !     last try to get a volume: if both sets are limited take the volume
    !     of the limiting sphere (NsetB/Vol is the density used in the
    !     normalization)
    IF(VOL <= ZERO) THEN
       IF(QLOC) THEN
          !           both sets are limited -> use the volume of the limiting sphere
          VOL=(4.0/3.0)*PI*(RARND**3)
       ELSE
          !           try to get a volume from crystal/image setup
          VOL=EPROP(VOLUME)
       ENDIF
    ENDIF
    !
    !     if VOL is still 0 we can't proceed
    IF(VOL <= ZERO) CALL WRNDIE(-5,'<RDFSOL>', &
         'Cant get volume of the system -> specify VOLUme.')
    !
    !----------------------------------------------------------------------
    !
    !     now call the work routine
    CALL RDFSL2(nsp,NBIN,RMAX,VOL,IOUT, &
         NUNIT,FIRSTU,BEGIN,SKIP,STOP, &
         NSOLA,HPSOLA,STISEL, &
         NSOLB,HPSOLB,STJSEL, &
         NSOLU,HPSOLU, &
         NSOLV,HPSOLV, &
         NSOLX,HPSOLX, &
         QRDF,QDDIP,QQDIP,QHD,QUSER, &
         IRDF,IDDIP,IQDIP,IHD,IUSER, &
         HPAX,HPAY,HPAZ, &
         QAPROT,NAPROT,QBPROT,NBPROT, &
         HPG, &
         BIMAG%IMATTR,SPFAC, &
         DOIMAGES,QBRUTE,QPREC,QIMMAX,QMASS, &
         QAWAT,QBWAT,QASITE,QBSITE,XASITE,XREF,YREF,ZREF,QREF, &
         QSAME,QARND,QLOC,XARND,RARND,XAREF,YAREF,ZAREF, &
         QMINI,HPMINI,LNMINI,LNMIN2)
    !
    !     clean up
    !
    call chmdealloc('rdfsol.src','RDFSOL','space_ST',3*NSP,intg=space_ST)
    call chmdealloc('rdfsol.src','RDFSOL','HPG',SPG,crl=HPG)

    call chmdealloc('rdfsol.src','RDFSOL','space_hp2',3*Nsp,crl=space_hp2)

    call chmdealloc('rdfsol.src','RDFSOL','HPSOLA',NSP,intgp=HPSOLA)
    call chmdealloc('rdfsol.src','RDFSOL','HPSOLB',NSP,intgp=HPSOLB)
    call chmdealloc('rdfsol.src','RDFSOL','HPSOLU',NSP,intgp=HPSOLU)
    call chmdealloc('rdfsol.src','RDFSOL','HPSOLV',NSP,intgp=HPSOLV)
    call chmdealloc('rdfsol.src','RDFSOL','HPSOLX',NSP,intgp=HPSOLX)

    call chmdealloc('rdfsol.src','RDFSOL','HPMINI',LNMIN2,cr4=HPMINI)
    !
    RETURN
  END SUBROUTINE RDFSOL
  !
  !----------------------------------------------------------------------
  !

  SUBROUTINE RDFSL2(nsp,NBIN,RMAX,VOL,IOUT, &
       NUNIT,FIRSTU,BEGIN,SKIP,STOP, &
       NSOLA,HPSOLA,STISEL, &
       NSOLB,HPSOLB,STJSEL, &
       NSOLU,HPSOLU, &
       NSOLV,HPSOLV, &
       NSOLX,HPSOLX, &
       QRDF,QDDIP,QQDIP,QHD,QUSER, &
       IRDF,IDDIP,IQDIP,IHD,IUSER, &
       AX,AY,AZ, &
       QAPROT,NAPROT,QBPROT,NBPROT, &
       GXX, &
       IMATT,SPFAC, &
       DOIMAGES,QBRUTE,QPREC,QIMMAX,QMASS, &
       QAWAT,QBWAT,QASITE,QBSITE,XASITE,XREF,YREF,ZREF,QREF, &
       QSAME,QARND,QLOC,XARND,RARND,XAREF,YAREF,ZAREF, &
       QMINI,MINI,LNMINI,LNMIN2)
    !
    !     This is the actual work routine. It reads the frames from the
    !     trajectory, calls the appropriate subroutine to solve the
    !     distance problem and collects the requested function.
    !     Finally it does the output and returns.
    !
    !     Variables:
    !     NBIN - number of bins to sample into
    !     RMAX - maximum distance checked
    !     VOL  - volume of the system (for the normalization)
    !     IOUT - unit to write the output to
    !     NUNIT, FIRSTU, BEGIN, SKIP, STOP - trajectory specs
    !     NSOLA - number of atoms in set A
    !     HPSOLA - pointer for set A
    !     STISEL - pointer to 'flags'-like array for set A
    !     ----B  - values for set B
    !     ----U/V - sets to swap, reorder
    !     GXX    - arrays to sample values into (NBIN long)
    !     REVRES - reverse lookup table REVRES(ATOM) = resnum of ATOM
    !     IMATT  - IMATT(IMAGEATOM) = PRIMARY of IMAGEATOM
    !     SPFAC  - allocate SPFAC * actual images space for future images
    !     DIOMAGES - do we consider images?
    !     QBRUTE - use double-loop instead of cubing (where appropriate)
    !     QPREC - check for 2A more than RMAX in water-water to get all H-H pairs
    !     QIMMAX - can we skip the generation of "what to image" and just do coors?
    !     Q(A/B)WAT - is set A/B water
    !     Q(A/B)SITE - is set A/B a site (c.o.m. or fixed point in space)
    !     XASITE - what kind of site is set A
    !     X/Y/ZREF - the point in space
    !     QSAME - use set A for both sets
    !     QARND - limit set A around a fixed point
    !     QLOC  - limit set B also
    !     XARND - limit around what (c.o.m. or fixed point in space)
    !     RARND - limiting distance
    !     X/Y/ZAREF - the limiting point
    !     QMINI - use "real" minimum image conventions
    !     MINI - the minimum distance NxN matrix
    !     LNMINI - N (dimension of MINI)
    !     LNMINI2 - number of cells in MINI (LNMINI*LNMINI)
    !
    !     Tibor Rudas Nov 2002 - Jun 2003
    !
#if KEY_FLUCQ==1
    use flucqm,only: fqcfor       
#endif
    use number
    use bases_fcm
    use consta
    use coord
    use ctitla
    use cvio
    use image
    use image_routines_module, only: transo
    use imgup
    use psf
    use corsubs,only: cdipole,getcom
#if KEY_PROTO==1
    use proto_mod,only: adimprt,filprot    
#endif
    use stream
    ! needed for TRANSO
#if KEY_FLUCQ==1
    use flucq    
#endif
    use memory
    use rdfsol_subs,only: goodi2,goodis,gpydis,gxydi2,gxydis
    implicit none
    !
    INTEGER nsp,NBIN,IOUT,SPFAC
    INTEGER NUNIT,FIRSTU,BEGIN,SKIP,STOP
    INTEGER NSOLA
    integer,pointer,dimension(:) :: HPSOLA,hpsolb,hpsolu,hpsolv,hpsolx,seta,setb
    INTEGER NSOLB
    INTEGER NSOLU
    INTEGER NSOLV
    INTEGER NSOLX
    INTEGER IMATT(*),XASITE,XARND
    integer,allocatable,dimension(:) :: revres
    INTEGER IRDF,IDDIP,IQDIP,IHD,IUSER
    INTEGER NAPROT,NBPROT
    real(chm_real) RMAX,VOL
    real(chm_real) AX(*),AY(*),AZ(*),GXX(8,*)
    real(chm_real),allocatable,dimension(:,:) :: dip
    real(chm_real) XREF,YREF,ZREF,QREF
    real(chm_real) RARND,XAREF,YAREF,ZAREF
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER,QAPROT,QBPROT
    LOGICAL QBRUTE,QPREC,QAWAT,QBWAT,QASITE,QBSITE
    LOGICAL DOIMAGES,QSAME,QARND,QLOC,QIMMAX,QMASS
    !
    !     new variables for distance-matrix (minimum image approach)
    LOGICAL QMINI
    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(*)
    !
    INTEGER I,J,NSOLVR
    real(chm_real4),allocatable,dimension(:) :: TEMP
    integer,allocatable,dimension(:) :: ITEMP,JTEMP,freeat
    integer,allocatable,dimension(:) :: KTEMP
    INTEGER NLAST,NTIM,NTOT
    INTEGER NDEGF,IUNIT,NFREAT,NFILE,ISTEP,ISTATS,NSAVV
    INTEGER HPG1,HPG2,HPG3,HPG4,HPG5,HPG6
    INTEGER NSOLAI,NSOLBI,NSOLUI,NSOLVI
    INTEGER NSETA,NSETAI
    INTEGER NSETB,NSETBI
    integer,pointer,dimension(:) :: setasl,setbsl,stisel,stjsel,resst

    INTEGER XALGO,NN,NRESST
    INTEGER OLDPRN,OFFSET
    !
    real(chm_real) DELTA,DELTA2,DR,XNORM,ROODR,ROO,XTMP,DT
    real(chm_real) XBRED,YBRED,ZBRED,NORMA,NORMB,F1,F2,F3
    real(chm_real) XAD,YAD,ZAD,LDIP,QA,XBD,YBD,ZBD,LBDIP,QB
    real(chm_real) XP,YP,ZP,QP,XDP,YDP,ZDP,LDP,OCUTIM
    !     cube-vars:
    real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,X0,Y0,Z0,XD,YD,ZD, &
         RHI,RHINV
    INTEGER NCUBX,NCUBY,NCUBZ,NCUBE,NCUB,NTIMA
    integer,allocatable,dimension(:),target :: space_hpn0,space_hpm0,space_hpn, &
         space_hpn0m,space_hpm0m
    INTEGER,pointer,dimension(:) :: HPN0,HPN1,HPN2,HPN3, &
         HPM0,HPM1,HPM2,HPM3,HPM4,HPM5, &
         HPN0M,HPN1M,HPN2M,HPN3M, &
         HPM0M,HPM1M,HPM2M,HPM3M,HPM4M,HPM5M
    !
    REAL(chm_real) ::MARGIN=pt01
    CHARACTER(len=4) :: HDR1,HDR2,hdrc='COOR',hdrd='CORD',hdrv='VELD'
    !
    LOGICAL QWAT,QNORM
    LOGICAL QAPT,QBPT,QLIMAL
    !
    !TR      IF(QBRUTE) WRITE(OUTU,'(A)') ' Using brute-force algorithm'
    !
    !     build reverse residue lookup table and do water & dipole
    !     ensemble selection
    call chmalloc('rdfsol.src','RDFSL2','revRES',NATOM,intg=revRES)
    revres(1:natom)=0
    call chmalloc('rdfsol.src','RDFSL2','dip',4,Nsp,crl=dip)


    DO I=1,NRES
       DO J=(IBASE(I)+1),IBASE(I+1)
          REVRES(J)=I
       ENDDO
    ENDDO
    !
    !     spacing of bins
    DR=RMAX/NBIN
    !
    !----- decide some things helpful -------------------------------------
    !
    !     will set A be reduced to a point?
    QAPT=.FALSE.
    IF(QASITE) THEN
       !        i.e. it is requested
       QAPT=.TRUE.
    ELSE IF((.NOT.QAWAT).AND.(.NOT.QAPROT).AND.QDDIP) THEN
       !        i.e. we have a set which is NOT water or a prototype and want
       !             its dipole correlation function
       !             -> must reduce set to c.o.m & dipole
       QAPT=.TRUE.
    ENDIF
    !     will set B be reduced to a point?
    QBPT=.FALSE.
    IF(QBSITE) THEN
       !        i.e. it is requested
       QBPT=.TRUE.
    ELSE IF((.NOT.QSAME).AND.(.NOT.QBWAT).AND.(.NOT.QBPROT).AND. &
         (QDDIP.OR.QQDIP.OR.QHD.OR.QUSER)) THEN
       !        i.e. we have a set which is NOT water and want its dipole
       !             correlation function -> must reduce set to c.o.m & dipole
       QBPT=.TRUE.
    ENDIF
    !
    !     do we calculate water-water function without any com
    IF(QAWAT.AND.QBWAT) THEN
       IF((.NOT.QAPT).AND.(.NOT.QBPT)) THEN
          IF((.NOT.QARND).OR.(QLOC)) QSAME=.TRUE.
       ENDIF
    ENDIF
    !
    !     now check for combinations which are not useful / implemented
    IF(QARND.AND.QAPT) &
         WRITE(OUTU,'(A,A)') &
         ' Set A: reduction and c.o.m. not implemented', &
         ' -> ignoring reduction'
    IF(QLOC.AND.QBPT) &
         WRITE(OUTU,'(A,A)') &
         ' Set B: reduction and c.o.m. not implemented', &
         ' -> ignoring reduction'
    !
    IF((XASITE == 1).AND.(QDDIP.OR.QHD)) &
         CALL WRNDIE(-5,'<RDFSOL>', 'Set A: a point has no dipole.')
    IF(QAPT.AND.QSAME) &
         CALL WRNDIE(-5,'<RDFSOL>', &
         'Both sets reduce to the same point?')
    !
    !
    !----------------------------------------------------------------------
    !     F o r   t h e   c u b i n g   a l g o r i t h m
    !---- Find bounding box around molecule -------------------------------
    !
    !     provide for case without images (NATIM=0) or if we don't use them
    IF(DOIMAGES) THEN
       NTIM=NATIM-NATOM
       NLAST=NATIM
    ELSE
       NTIM=0
       NLAST=NATOM
    ENDIF
    XMIN = X(1)
    XMAX = XMIN
    YMIN = Y(1)
    YMAX = YMIN
    ZMIN = Z(1)
    ZMAX = ZMIN
    DO I = 2, NLAST
       XMIN = MIN(XMIN,X(I))
       YMIN = MIN(YMIN,Y(I))
       ZMIN = MIN(ZMIN,Z(I))
       XMAX = MAX(XMAX,X(I))
       YMAX = MAX(YMAX,Y(I))
       ZMAX = MAX(ZMAX,Z(I))
    ENDDO
    XD = XMAX - XMIN
    YD = YMAX - YMIN
    ZD = ZMAX - ZMIN
    !
    !---- Establish cube parameters ---------------------------------------
    !
    RHI = RMAX
    !
    !     if we are computing water-water or X-water or water-X rdfs and
    !     want to be precise we need to enlarge the minimum O-O distance
    !     sampled or we'll miss O-H and H-H pairs
    IF(QPREC.AND.QRDF.AND. &
         ((QAWAT.AND.(.NOT.QAPT)).OR. &
         (QBWAT.AND.(.NOT.QBPT)).OR. &
         (QAWAT.AND.QSAME)          ) )  THEN
       !        we do reset rmax
       QPREC=.TRUE.
       RHI=RHI+2
    ELSE
       !        we don't reset rmax (even if requested) record for
       !        normalizing issues in the end...
       QPREC=.FALSE.
    ENDIF
    !
    NN=2
    IF(DOIMAGES) THEN
       !        NEW: tr Aug 2005
       !             don't try to be smarter than the user (you'll mess things up)
       !             just warn
       !        store CUTIM and set to rhi since we need no more
       IF((.NOT.QIMMAX).AND.(CUTIM < RHI)) CALL WRNDIE(-3, &
            '<RDFSOL>','CUTIM < RMAX will give wrong results.')
       IF(QARND.OR.QLOC.OR.LIMALL.OR.QAPT.OR.QBPT) THEN
          !           i.e. all images will need to be generated and we'll find
          !                a-b' & a'-b so count them only once (NN=1)
          IF(.NOT.LIMALL) CALL WRNDIE(-3, &
               '<RDFSOL>','We need all images; please set IMALL.')
          NN=1
          !
          !           Call TRANSO/UPIMAG once, so that all image-info is up to date
          !           even if IMMAX is set (i.e. we will only call TRANSO
          !           and not UPIMAG when reading frames from traj)
          !
          !           Construct the coordinates.
          CALL TRANSO(X, Y, Z, &
               (/ zero /), (/ zero /), (/ zero /), &
               .FALSE.,.FALSE., &
               (/ zero /), &
               NATOM,NTRANS, &
               IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
               NOROT,NATIM &
#if KEY_FLUCQ==1
               ,QFLUC,CG,FQCFOR  & 
#endif
               )
          !           we don't want the output from UPIMAG -> reset PRNLEV
          OLDPRN=PRNLEV
          PRNLEV=1
          CALL UPIMAG0(X, Y, Z, WMAIN, 0)
          !           reset printlevel
          PRNLEV=OLDPRN
          NTIM=NATIM-NATOM
          NTIMA=NTIM*SPFAC
          IF(NTIM > NTIMA) THEN
             !              i.e. we generated more images than we allowed for...
             WRITE(OUTU,'(A,I10,A,I10,A,I10)') 'In step ', NTOT, &
                  ' there were ', NATIM, ' images and space for ', &
                  NTIMA
             CALL WRNDIE(-1,'<RDFSOL>', &
                  'Image overflow, please increase SPFAc manually.')
          ENDIF
       ENDIF
    ENDIF
    !
    RHINV = ONE/RHI
    !
    NCUBX = INT((XD/RHI) + MARGIN + 1)
    NCUBY = INT((YD/RHI) + MARGIN + 1)
    NCUBZ = INT((ZD/RHI) + MARGIN + 1)
    NCUBE = NCUBX*NCUBY*NCUBZ
    !
    X0 = XMIN - 0.5 * (NCUBX * RHI - XD)
    Y0 = YMIN - 0.5 * (NCUBY * RHI - YD)
    Z0 = ZMIN - 0.5 * (NCUBZ * RHI - ZD)
    !
    !---- Allocate work areas for XDIST -----------------------------------
    IF(NATIM > NATOM) THEN
       !        no of imgs changes from frame to frame -> allow for max.
       !        we must estimate!
       NTIMA=NTIM*SPFAC
       NCUB=NCUBE*SPFAC
    ELSE
       NTIMA=1
       NCUB=NCUBE
    ENDIF
    !     ........ primary ........
    call chmalloc('rdfsol.src','RDFSL2','space_HPN0',4*NATOM,intg=space_HPN0)
    HPN0   => space_hpn0(0*natom+1 : 1*natom)
    HPN1   => space_hpn0(1*natom+1 : 2*natom)
    HPN2   => space_hpn0(2*natom+1 : 3*natom)
    HPN3   => space_hpn0(3*natom+1 : 4*natom)

    call chmalloc('rdfsol.src','RDFSL2','space_HPM0',6*NCUB,intg=space_HPM0)
    HPM0   => space_hpm0(0*ncub+1 : 1*ncub)
    HPM1   => space_hpm0(1*ncub+1 : 2*ncub)
    HPM2   => space_hpm0(2*ncub+1 : 3*ncub)
    HPM3   => space_hpm0(3*ncub+1 : 4*ncub)
    HPM4   => space_hpm0(4*ncub+1 : 5*ncub)
    HPM5   => space_hpm0(5*ncub+1 : 6*ncub)
    !     ........ images ........
    !
    call chmalloc('rdfsol.src','RDFSL2','space_HPN0M',4*NTIMA,intg=space_HPN0M)
    HPN0M  => space_hpn0m(0*ntima+1 : 1*ntima)
    HPN1M  => space_hpn0m(1*ntima+1 : 2*ntima)
    HPN2M  => space_hpn0m(2*ntima+1 : 3*ntima)
    HPN3M  => space_hpn0m(3*ntima+1 : 4*ntima)
    !
    call chmalloc('rdfsol.src','RDFSL2','space_HPM0M',6*NCUB,intg=space_HPM0M)
    HPM0M  => space_hpm0m(0*ncub+1 : 1*ncub)
    HPM1M  => space_hpm0m(1*ncub+1 : 2*ncub)
    HPM2M  => space_hpm0m(2*ncub+1 : 3*ncub)
    HPM3M  => space_hpm0m(3*ncub+1 : 4*ncub)
    HPM4M  => space_hpm0m(4*ncub+1 : 5*ncub)
    HPM5M  => space_hpm0m(5*ncub+1 : 6*ncub)
    !
    !----------------------------------------------------------------------
    !     allocate space
    call chmalloc('rdfsol.src','RDFSL2','TEMP',NATOM,cr4=TEMP)
    call chmalloc('rdfsol.src','RDFSL2','ITEMP',NATOM,intg=ITEMP)
    call chmalloc('rdfsol.src','RDFSL2','JTEMP',NATOM,intg=JTEMP)
    call chmalloc('rdfsol.src','RDFSL2','KTEMP',NATOM,intg=KTEMP)
    call chmalloc('rdfsol.src','RDFSL2','FREEAT',NATOM,intg=FREEAT)
    !
    !----------------------------------------------------------------------
    !
    !     now decide which algorithm we use:
    XALGO=0
    IF(QSAME) THEN
       !        i.e. we use GOO and both sets are the same so we only need A
       XALGO=1
    ELSE IF((.NOT.QAPT).AND.(.NOT.QBPT)) THEN
       !        i.e. both sets are sets but not the same (at least not
       !             that we know)
       XALGO=2
    ELSE IF((QAPT.AND.(.NOT.QBPT)).OR.(QBPT.AND.(.NOT.QAPT))) THEN
       !        i.e. one set was reduced to a point and the other is a set
       XALGO=3
    ELSE IF(QAPT.AND.QBPT) THEN
       !        i.e. both sets are reduced to a point
       XALGO=4
    ELSE
       CALL WRNDIE(-5,'<RDFSOL>', &
            'How did you get here? Check Source!')
    ENDIF
    !----- do some output if desired --------------------------------------
    !
    IF (PRNLEV >= 2) THEN
       NSOLVR=-1
       IF(QAWAT) NSOLVR=NSOLA
       IF(QBWAT) NSOLVR=NSOLB
       CALL PRNPAR(OUTU,NATOM,NTIM,NCUBX,NCUBY,NCUBZ,NCUBE,RMAX, &
            RHI,QRDF,QDDIP,QQDIP,QHD,QUSER,QAWAT,QBWAT,XASITE, &
            NSOLVR,NSOLA,NSOLB,NSOLX,QAPT,QBPT, &
            QAPROT,NAPROT,QBPROT,NBPROT, &
            QSAME,QARND,QLOC,RARND,XARND,XREF,YREF,ZREF,QREF, &
            XAREF,YAREF,ZAREF,XALGO,QPREC,QBRUTE,DOIMAGES,LIMALL,NN, &
            QMINI,QIMMAX,QMASS)
    ENDIF
    !
    !----------------------------------------------------------------------
    !
    !     set accumulators = 0
    NORMA=0
    NORMB=0
    NSOLAI=0
    NSOLBI=0
    NSOLUI=0
    NSOLVI=0
    !
    !----------------------------------------------------------------------
    !  T h i s   i s   t h e   g r e a t   T r a j   r e a d i n g    L o o p
    !----------------------------------------------------------------------
    !     ... start reading
    HDR1=HDRC
    HDR2=HDRD
    NTOT=0
    IUNIT=FIRSTU
    ISTATS=1
200 CONTINUE
    CALL READCV(X,Y,Z, &
#if KEY_CHEQ==1
         (/ ZERO /), .FALSE., &  
#endif
         TEMP,NATOM,FREEAT,NFREAT, &
         FIRSTU,NUNIT,IUNIT,NFILE, &
         ISTEP,ISTATS,NDEGF,DELTA2, &
         BEGIN,STOP,SKIP,NSAVV,HDR1,HDR2, &
         TITLEA,NTITLA,.FALSE., (/ ZERO /), .true.)
    IF(NTOT == 0) THEN
       DELTA=DELTA2*TIMFAC
       !     Check to see if DELTA should be rounded to nearest integer femtosecond.
       IF(ABS(DELTA-INT(THOSND*DELTA+HALF)/THOSND) < DELTA/FTHSND) &
            DELTA=INT(THOSND*DELTA+HALF)/THOSND
    ENDIF
    !
    !     UPDATE IMAGES...
    IF(DOIMAGES) THEN
       !tr      28. 5. 2003: do we need to call TRANSO before UPIMAG
       !tr                   to get correct image centering ?
       !        Construct the coordinates.
       CALL TRANSO(X, Y, Z, &
            (/ zero /), (/ zero /), (/ zero /), &
            .FALSE., .FALSE., &
            (/ zero /), &
            NATOM,NTRANS, &
            IMTRNS,BIMAG%IMATPT,BIMAG%IMATTR, &
            NOROT,NATIM &
#if KEY_FLUCQ==1
            ,QFLUC,CG,FQCFOR    & 
#endif
            )
       IF(.NOT.QIMMAX) THEN
          !           we don't want the output from UPIMAG -> reset PRNLEV
          OLDPRN=PRNLEV
          PRNLEV=1
          CALL UPIMAG0(X, Y, Z, WMAIN, 0)
          !           reset printlevel
          PRNLEV=OLDPRN
          NTIM=NATIM-NATOM
          IF(NTIM > NTIMA) THEN
             !              i.e. we generated more images than we allowed for...
             WRITE(OUTU,'(A,I10,A,I10,A,I10)') 'In step ', NTOT, &
                  ' there were ', NATIM, &
                  ' images and space for ', NTIMA
             CALL WRNDIE(-1,'<RDFSOL>', &
                  'Image overflow, please increase SPFAc manually.')
          ENDIF
       ENDIF
       !        add images to set a & b
       IF((.NOT.QAPT).AND.(.NOT.QAPROT)) &
            CALL ADDIMG(NSOLA,hpsola,STISEL, &
            NSOLAI,NATOM,NATIM,IMATT)
       IF((.NOT.QBPT).AND.(.NOT.QBPROT)) &
            CALL ADDIMG(NSOLB,hpsolb,STJSEL, &
            NSOLBI,NATOM,NATIM,IMATT)
       !
#if KEY_PROTO==1 /*proto*/
       !        add images to prototypes, and check that they are complete
       IF(QAPROT) CALL ADIMPRT(NAPROT)
       IF(QBPROT) CALL ADIMPRT(NBPROT)
#endif /*    (proto)*/
    ENDIF
    !
    !---- Find bounding box around molecule -------------------------------
    !
    !     we need to redo this part for each frame since coors have changed
    IF(DOIMAGES) THEN
       NLAST=NATIM
    ELSE
       NLAST=NATOM
    ENDIF
    XMIN = X(1)
    XMAX = XMIN
    YMIN = Y(1)
    YMAX = YMIN
    ZMIN = Z(1)
    ZMAX = ZMIN
    DO I = 2, NLAST
       XMIN = MIN(XMIN,X(I))
       YMIN = MIN(YMIN,Y(I))
       ZMIN = MIN(ZMIN,Z(I))
       XMAX = MAX(XMAX,X(I))
       YMAX = MAX(YMAX,Y(I))
       ZMAX = MAX(ZMAX,Z(I))
    END DO
    XD = XMAX - XMIN
    YD = YMAX - YMIN
    ZD = ZMAX - ZMIN
    !
    !---- Establish cube parameters ---------------------------------------
    !
    NCUBX = INT((XD/RHI) + MARGIN + 1)
    NCUBY = INT((YD/RHI) + MARGIN + 1)
    NCUBZ = INT((ZD/RHI) + MARGIN + 1)
    NCUBE = NCUBX*NCUBY*NCUBZ
    !
    X0 = XMIN - 0.5 * (NCUBX * RHI - XD)
    Y0 = YMIN - 0.5 * (NCUBY * RHI - YD)
    Z0 = ZMIN - 0.5 * (NCUBZ * RHI - ZD)
    !
    !---- end of cube setup -----------------------------------------------
    !
    !     now reduce sets if requested, calculate c.o.m. of limiting
    !         site if needed (XARND=2) in any case the limiting point
    !         is stored in XAREF/YAREF/ZAREF
    !
    IF(QARND) THEN
       !        i.e. we want to limit set A or both
       IF(XARND == 2) &
            CALL GETCOM(HPSOLX,NSOLX,X,Y,Z,AMASS,CG, &
            XAREF,YAREF,ZAREF,XTMP)
       !        and now reduce set A to set U
       CALL LIMSET(hpsola,NSOLA,NSOLAI, &
            HPSOLU,NSOLU,NSOLUI, &
            X,Y,Z,XAREF,YAREF,ZAREF,RARND,DOIMAGES)
       !        and remap (images -> primary and then recreate images)
       IF(DOIMAGES) &
            CALL REMAP(HPSOLU,NSOLU,NSOLUI,KTEMP, &
            IMATT,NATOM,NATIM)
       !        if we want to be completely local-> reduce set B to set V
       IF(QLOC) THEN
          CALL LIMSET(hpsolb,NSOLB,NSOLBI, &
               HPSOLV,NSOLV,NSOLVI, &
               X,Y,Z,XAREF,YAREF,ZAREF,RARND,DOIMAGES)
          IF(DOIMAGES) &
               CALL REMAP(HPSOLV,NSOLV,NSOLVI,KTEMP, &
               IMATT,NATOM,NATIM)
       ENDIF
    ENDIF
    !
    !     get c.o.m. of set A and/or set B if needed
    IF(QAPT.AND.(XASITE /= 1)) THEN
       CALL GETCOM(hpsola,NSOLA,X,Y,Z,AMASS,CG, &
            XREF,YREF,ZREF,QREF)
       CALL CDIPOLE(NATOM,X,Y,Z,CG,NSOLA,hpsola, &
            XAD,YAD,ZAD,QA,.FALSE.,QMASS,AMASS)
       LDIP=DSQRT((XAD*XAD)+(YAD*YAD)+(ZAD*ZAD))
       XAD=XAD/LDIP
       YAD=YAD/LDIP
       ZAD=ZAD/LDIP
       !        TODO: we should do this once outside the reading loop
       !              but then we need yet another list...
       CALL RESLST(STISEL,HPSOLU,NSOLU)
    ENDIF
    !     fix: if both sets are the same (QSAME) we need not reduce setB
    !          although this may be assumed above (in these cases getcom
    !          will be called with an empty set and will then try to divide
    !          by sum_mass = 0 and abort)
    IF(QBPT.AND.(.NOT.QSAME)) THEN
       CALL GETCOM(hpsolb,NSOLB,X,Y,Z,AMASS,CG, &
            XBRED,YBRED,ZBRED,XTMP)
       CALL CDIPOLE(NATOM,X,Y,Z,CG,NSOLB,hpsolb, &
            XBD,YBD,ZBD,QB,.FALSE.,QMASS,AMASS)
       LBDIP=DSQRT((XAD*XAD)+(YAD*YAD)+(ZAD*ZAD))
       XBD=XAD/LBDIP
       YBD=YAD/LBDIP
       ZBD=ZAD/LBDIP
       CALL RESLST(STJSEL,HPSOLV,NSOLV)
    ENDIF
    !
    !     calculate water dipole moments (if needed)
    IF(QAWAT) THEN
       CALL WATDIP(NSOLA,NSOLAI,hpsola,DIP,X,Y,Z,CG)
    ENDIF
    IF(QBWAT) THEN
       CALL WATDIP(NSOLB,NSOLBI,hpsolb,DIP,X,Y,Z,CG)
    ENDIF
    !
    !     use prototypes:
    OFFSET=0
#if KEY_PROTO==1 /*proto*/
    IF(QAPROT) THEN
       CALL FILPROT(NAPROT,X,Y,Z,AX,AY,AZ,DIP, &
            NSOLA,NSOLAI,hpsola,QMASS)
    ELSE
       CALL CPCOR(X,Y,Z,AX,AY,AZ,MAX(NSOLA,NSOLAI),hpsola,QAWAT)
    ENDIF
    IF(.NOT.QSAME) THEN
       IF(QBPROT) THEN
          CALL FILPROT(NBPROT,X,Y,Z,AX,AY,AZ,DIP, &
               NSOLB,NSOLBI,hpsolb,QMASS)
       ELSE
          CALL CPCOR(X,Y,Z,AX,AY,AZ,MAX(NSOLB,NSOLBI), &
               hpsolb,QBWAT)
       ENDIF
    ENDIF
#else /* (proto)*/
    CALL CPCOR(X,Y,Z,AX,AY,AZ,MAX(NSOLA,NSOLAI),hpsola,QAWAT)
    IF(.NOT.QSAME) THEN
       CALL CPCOR(X,Y,Z,AX,AY,AZ,MAX(NSOLB,NSOLBI), &
            hpsolb,QBWAT)
    ENDIF
#endif /* (proto)*/
    !
    !
    !     First: assing the right data structure to SET(A/B) NSET(A/B)
    !            and SET(A/B)SL and to G1,G2,G3
    !
    !     Then: decide if we have:
    !          a) two identical sets (QSAME=.TRUE) -> call GOODIS
    !          b) two different sets ((.NOT.QAPT).AND.(.NOT.QBPT))
    !                                              -> call GXYDIS
    !          c) a point and a set  (QAPT.AND.(.NOT.QBPT))
    !                                              -> call GPYDIS
    !          d) two points (QAPT.AND.QBPT)       -> calc here
    !
    !
    !     the defaults:
    !
    SETA => HPSOLA
    SETASL => STISEL
    NSETA=NSOLA
    NSETAI=NSOLAI
    !
    SETB => HPSOLB
    SETBSL => STJSEL
    NSETB=NSOLB
    NSETBI=NSOLBI
    !
    IF(QARND) THEN
       IF(.NOT.QAPT) THEN
          SETA => HPSOLU
          NSETA=NSOLU
          NSETAI=NSOLUI
       ENDIF
       IF(QLOC.AND.(.NOT.QBPT)) THEN
          SETB => HPSOLV
          NSETB=NSOLV
          NSETBI=NSOLVI
       ENDIF
    ENDIF
    !
    !     now set the right pointers
    QWAT=.FALSE.
    IF(QSAME) THEN
       !        i.e. we use GOO and both sets are the same so we only need A
       !        XALGO=1, GOODIS
       NORMA=NORMA+NSETA
       NORMB=NORMB+NSETA
    ELSE IF((.NOT.QAPT).AND.(.NOT.QBPT)) THEN
       !        i.e. both sets are sets but not the same (at least not
       !             that we know)
       !        XALGO=2, GXYDIS
       NORMA=NORMA+NSETA
       NORMB=NORMB+NSETB
    ELSE IF(QAPT.AND.(.NOT.QBPT)) THEN
       !        i.e. set A was reduced to a point and B is a set
       !        XALGO=3, GPYDIS
       QWAT=QBWAT
       XP=XREF
       YP=YREF
       ZP=ZREF
       QP=QREF
       XDP=XAD
       YDP=YAD
       ZDP=ZAD
       LDP=LDIP
       !
       RESST=HPSOLU
       NRESST=NSOLU
       !
       NORMA=NORMA+1
       NORMB=NORMB+NSETB
    ELSE IF(QBPT.AND.(.NOT.QAPT)) THEN
       !        i.e. set B was reduced to a point and A is a set
       !        XALGO=3, GPYDIS
       QWAT=QAWAT
       XP=XBRED
       YP=YBRED
       ZP=ZBRED
       QP=QB
       XDP=XBD
       YDP=YBD
       ZDP=ZBD
       LDP=LBDIP
       !        swap sets since we always pass set B to GPYDIS
       NSETB=NSETA
       NSETBI=NSETAI
       SETB => SETA
       SETBSL => SETASL
       RESST=HPSOLV
       NRESST=NSOLV
       NORMA=NORMA+NSETA
       NORMB=NORMB+1
    ELSE IF(QAPT.AND.QBPT) THEN
       !        i.e. both sets are reduced to a point
       NORMA=NORMA+1
       NORMB=NORMB+1
    ELSE
       CALL WRNDIE(-5,'<RDFSOL>', &
            'How did you get here? Check Source(2)!')
    ENDIF
    !
    !---- zero out minimum image matrix -----------------------------------
    !
    IF(QMINI) MINI(1:LNMIN2) = ZERO
    !
    !----------------------------------------------------------------------
    !
    !     now call the function!
    IF(XALGO == 1) THEN
       IF(QBRUTE) THEN
          CALL GOODI2(NATOM,NATIM,AX,AY,AZ,RHI,NBIN,DR, &
               NSETA,NSETAI,SETA,SETASL,DIP, &
               REVRES,IMATT,QRDF,QDDIP,QQDIP,QHD,QUSER, &
               GXX,DOIMAGES,QAWAT,NN, &
               QMINI,MINI,LNMINI,LNMIN2)
       ELSE
          CALL GOODIS(NATOM,NATIM,AX,AY,AZ,RHI,RHINV,X0,Y0,Z0, &
               NCUBX,NCUBY,NCUBZ,NCUBE,IMATT,NBIN,DR, &
               NSETA,NSETAI,SETA,SETASL,DIP, &
               REVRES,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
               HPM0,HPM1,HPM2,HPN0, &
               HPN1,HPM0M,HPM1M,HPM2M, &
               HPN0M,HPN1M,ITEMP, &
               DOIMAGES,QAWAT,QPREC,NN, &
               QMINI,MINI,LNMINI,LNMIN2)
          !           collect _real_ minimum image result for this frame
          IF(QMINI) &
               CALL CLMNGO(QRDF,QDDIP,QQDIP,QHD,QUSER, &
               LNMINI,LNMIN2,MINI, &
               NSETA,NSETAI,SETA,SETASL, &
               DR,NBIN,GXX,QAWAT)
          !
       ENDIF
    ELSE IF(XALGO == 2) THEN
       IF(QBRUTE) THEN
          CALL GXYDI2(NATOM,NATIM,AX,AY,AZ,RMAX,NBIN,DR, &
               NSETA,NSETAI,SETA,SETASL, &
               NSETB,NSETBI,SETB,SETBSL,DIP, &
               REVRES,IMATT,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
               DOIMAGES,QAWAT,QBWAT,NN, &
               QMINI,MINI,LNMINI,LNMIN2)
       ELSE
          CALL GXYDIS(NATOM,NATIM,AX,AY,AZ,RHI,RHINV,X0,Y0,Z0, &
               NCUBX,NCUBY,NCUBZ,NCUBE,IMATT,NBIN,DR, &
               NSETA,NSETAI,SETA,SETASL, &
               NSETB,NSETBI,SETB,SETBSL,DIP, &
               REVRES,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
               HPM0,HPM1,HPM2,HPN0, &
               HPN1,HPM3,HPM4,HPM5, &
               HPN2,HPN3,HPM0M,HPM1M, &
               HPM2M,HPM3M,HPM4M,HPM5M, &
               HPN0M,HPN1M, &
               HPN2M,HPN3M, &
               DOIMAGES,QAWAT,QBWAT,QPREC,NN, &
               QMINI,MINI,LNMINI,LNMIN2)
          !           collect _real_ minimum image result for this frame
          IF(QMINI) &
               CALL CLMNG2(QRDF,QDDIP,QQDIP,QHD,QUSER, &
               LNMINI,LNMIN2,MINI, &
               NSETA,NSETAI,SETA,SETASL, &
               NSETB,NSETBI,SETB,SETBSL, &
               DR,NBIN,GXX,QAWAT,QBWAT)
       ENDIF
    ELSE IF(XALGO == 3) THEN
       CALL GPYDIS(NATOM,NATIM,AX,AY,AZ,RHI, &
            XP,YP,ZP,QP,XDP,YDP,ZDP,LDP, &
            IMATT,NBIN,DR, &
            NSETB,NSETBI,SETB,SETBSL,DIP, &
            REVRES,GXX,QRDF,QDDIP,QQDIP,QHD,QUSER, &
            RESST,NRESST, &
            DOIMAGES,QWAT,QPREC, &
            QMINI,MINI,LNMINI,LNMIN2)
    ELSE IF(XALGO == 4) THEN
       !         POINT-POINT - STILL TO COME
    ENDIF
    !
    NTOT=NTOT+1
    !
    IF(ISTATS >= 0) GOTO 200
    !
    !----------------------------------------------------------------------
    !  E n d   o f   t h e   g r e a t   T r a j   r e a d i n g    L o o p
    !----------------------------------------------------------------------
    !
    !     free space
    call chmdealloc('rdfsol.src','RDFSL2','space_HPN0',4*NATOM,intg=space_HPN0)

    call chmdealloc('rdfsol.src','RDFSL2','space_HPM0',6*NCUB,intg=space_HPM0)

    call chmdealloc('rdfsol.src','RDFSL2','space_HPN0M',4*NTIMA,intg=space_HPN0M)

    call chmdealloc('rdfsol.src','RDFSL2','space_HPM0M',6*NCUB,intg=space_HPM0M)
    !
    !     if we didn't reset rmax internally, the last bin is only
    !     half-filled -> count twice to avoid drop of last point
    IF(.NOT.QPREC) THEN
       GXX(1,NBIN)=GXX(1,NBIN)*2.0D0
       !tr  GXX(2,NBIN)=GXX(2,NBIN)*2.0D0
       !tr  GXX(3,NBIN)=GXX(3,NBIN)*2.0D0
       GXX(4,NBIN)=GXX(4,NBIN)*2.0D0
       GXX(5,NBIN)=GXX(5,NBIN)*2.0D0
       GXX(6,NBIN)=GXX(6,NBIN)*2.0D0
       GXX(7,NBIN)=GXX(7,NBIN)*2.0D0
       GXX(8,NBIN)=GXX(8,NBIN)*2.0D0
    ENDIF
    !
    !     normalize and do output
    NORMA=NORMA/NTOT
    NORMB=NORMB/NTOT
    IF(QRDF) THEN
       XNORM=(FOUR * PI * NORMA * NORMB)/(THREE * VOL)
       IF((QAWAT.AND.(.NOT.QAPT)).AND. &
            (QSAME.OR.(QBWAT.AND.(.NOT.QBPT)))) THEN
          !           i.e. we calculated water-water rdfs
          F1=1
          F2=4
          F3=4
       ELSE IF(QBWAT) THEN
          !           i.e. point/set to water
          F1=1
          F2=2
          F3=1
       ELSE
          !           i.e. set to set
          F1=1
          F2=1
          F3=1
       ENDIF
       DO I=1,NBIN
          !           for solana like binning
          ROO=(I-HALF)*DR
          ROODR=ROO+DR
          GXX(1,I)=GXX(1,I)/(F1*XNORM*NTOT*(ROODR**3 - ROO**3))
          GXX(2,I)=GXX(2,I)/(F2*XNORM*NTOT*(ROODR**3 - ROO**3))
          GXX(3,I)=GXX(3,I)/(F3*XNORM*NTOT*(ROODR**3 - ROO**3))
          WRITE(IRDF,802) (I*DR),GXX(1,I),GXX(2,I),GXX(3,I)
       ENDDO
    ENDIF
    IF(QDDIP) THEN
       GXX(4,1)=GXX(4,1)/NTOT
       GXX(5,1)=GXX(4,1)
       DO I=2,NBIN
          GXX(4,I)=GXX(4,I)/NTOT
          GXX(5,I)=GXX(4,I)+GXX(5,I-1)
       ENDDO
       XNORM=(FOUR * PI * NORMA * NORMB)/(THREE * VOL)
       F1=NORMA
       IF((QAWAT.AND.(.NOT.QAPT)) .OR. &
            (QBWAT.AND.(.NOT.QBPT))) F1=NSOLVR
       DO I=1,NBIN
          !           for solana like binning
          ROO=(I-HALF)*DR
          ROODR=ROO+DR
          !           avoid division by zero for I=1 (???)
          !            IF(I > 1) G1(I)=G1(I)/(XNORM*(ROODR**3 - ROO**3))
          GXX(4,I)=GXX(4,I)/(XNORM*(ROODR**3 - ROO**3))
          GXX(5,I)=ONE+(GXX(5,I)/F1)
          WRITE(IDDIP,802) (I*DR),GXX(4,I),GXX(5,I)
       ENDDO
    ENDIF
    IF(QQDIP) THEN
       XNORM=(FOUR * PI * NORMA * NORMB)/(THREE * VOL)
       DO I=1,NBIN
          !           for solana like binning
          ROO=(I-HALF)*DR
          ROODR=ROO+DR
          GXX(6,I)=GXX(6,I)/(XNORM*NTOT*(ROODR**3 - ROO**3))
          WRITE(IQDIP,802) (I*DR),GXX(6,I)
       ENDDO
    ENDIF
    IF(QHD) THEN
       XNORM=(FOUR * PI * NORMA * NORMB)/(THREE * VOL)
       DO I=1,NBIN
          !           for solana like binning
          ROO=(I-HALF)*DR
          ROODR=ROO+DR
          GXX(7,I)=GXX(7,I)/(XNORM*NTOT*(ROODR**3 - ROO**3))
          WRITE(IHD,802) (I*DR),GXX(7,I)
       ENDDO
    ENDIF
    IF(QUSER) THEN
       XNORM=(FOUR * PI * NORMA * NORMB)/(THREE * VOL)
       DO I=1,NBIN
          !           for solana like binning
          ROO=(I-HALF)*DR
          ROODR=ROO+DR
          GXX(8,I)=GXX(8,I)/(XNORM*NTOT*(ROODR**3 - ROO**3))
          WRITE(IUSER,802) (I*DR),GXX(8,I)
       ENDDO
    ENDIF
    !
802 FORMAT(4(1X,F10.6))
    !
    call chmdealloc('rdfsol.src','RDFSL2','TEMP',NATOM,cr4=TEMP)
    call chmdealloc('rdfsol.src','RDFSL2','ITEMP',NATOM,intg=ITEMP)
    call chmdealloc('rdfsol.src','RDFSL2','JTEMP',NATOM,intg=JTEMP)
    call chmdealloc('rdfsol.src','RDFSL2','KTEMP',NATOM,intg=KTEMP)
    call chmdealloc('rdfsol.src','RDFSL2','FREEAT',NATOM,intg=FREEAT)
    call chmdealloc('rdfsol.src','RDFSL2','revRES',NATOM,intg=revRES)
    call chmdealloc('rdfsol.src','RDFSL2','dip',4,Nsp,crl=dip)
    RETURN
  END SUBROUTINE RDFSL2
  !
  !-----------------------------------------------------------------------
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE LIMSET(SETA,NSETA,NSETAI,SETB,NSETB,NSETBI,X,Y,Z, &
       XREF,YREF,ZREF,RAD,DOIMG)
    !
    !     takes NSETA atoms whose atomnumbers are in SETA and
    !     transfers all atoms to SETB which are closer than RAD
    !     to a reference point XREF/ YREF / ZREF.
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !

    INTEGER SETA(*),NSETA,NSETAI,SETB(*),NSETB,NSETBI
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) XREF,YREF,ZREF,RAD
    LOGICAL DOIMG
    !
    INTEGER I,K
    real(chm_real) RAD2,XD,YD,ZD,XTMP
    !
    RAD2=RAD*RAD
    NSETB=0
    !
    DO I=1,NSETA
       K=SETA(I)
       XD=X(K)-XREF
       XD=XD*XD
       IF(XD <= RAD2) THEN
          YD=Y(K)-YREF
          YD=YD*YD
          IF(YD <= RAD2) THEN
             ZD=Z(K)-ZREF
             ZD=ZD*ZD
             IF(ZD <= RAD2) THEN
                XTMP=XD+YD+ZD
                IF(XTMP <= RAD2) THEN
                   NSETB=NSETB+1
                   SETB(NSETB)=K
                ENDIF
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    IF(DOIMG)THEN
       NSETBI=NSETB
       !
       DO I=(NSETA+1),NSETAI
          K=SETA(I)
          XD=X(K)-XREF
          XD=XD*XD
          IF(XD <= RAD2) THEN
             YD=Y(K)-YREF
             YD=YD*YD
             IF(YD <= RAD2) THEN
                ZD=Z(K)-ZREF
                ZD=ZD*ZD
                IF(ZD <= RAD2) THEN
                   XTMP=XD+YD+ZD
                   IF(XTMP <= RAD2) THEN
                      NSETBI=NSETBI+1
                      SETB(NSETBI)=K
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !
    ENDIF
    !
    RETURN
  END SUBROUTINE LIMSET
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE SELWAT(NMAX,NSET,SET,SETSL)
    !
    !     find all waters in the PSF (i.e. all residues with name 'TIP3')
    !     and put the atomnumbers of the oxygens in SET
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    use psf

    INTEGER NMAX,NSET,SET(*),SETSL(*)
    !
    !
    INTEGER I,J
    !
    DO I=1,NATOM
       SETSL(I)=0
    ENDDO
    !
    NSET=0
    DO I=1,NRES
       J=IBASE(I)+1
       IF(RES(I) == 'TIP3') THEN
          !           i.e. water found -> add the oxygen to the list and mark
          !                as selected
          NSET=NSET+1
          IF(NSET > NMAX) &
               CALL WRNDIE(-5,'<RDFSOL>', &
               'Too many waters.')
          SET(NSET)=J
          SETSL(J)=1
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE SELWAT
  !
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE RESLST(FLLST,SETA,NSETA)
    !
    !     make SETA a list of all residues who have an atom in FLLST
    !     selected
    !
    !     Tibor Rudas Feb 2003 - Jun 2003
    !
    use psf

    INTEGER FLLST(*),SETA(*),NSETA
    !
    !
    INTEGER I,J,K,L
    !
    NSETA=0
    DO I=1,NRES
       J=IBASE(I)+1
       K=IBASE(I+1)
       DO L=J,K
          IF(FLLST(L) > 0) THEN
             !              i.e. this atom of residue I is in the SET, so add this
             !                   residue and go straight to the next residue (avoids
             !                   doubles)
             NSETA=NSETA+1
             SETA(NSETA)=I
             GOTO 100
          ENDIF
       ENDDO
100    CONTINUE
    ENDDO
    !
    RETURN
  END SUBROUTINE RESLST
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE PRNPAR(OUTU,NATOM,NTIM,NCUBX,NCUBY,NCUBZ,NCUBE,RMAX, &
       RHI,QRDF,QDDIP,QQDIP,QHD,QUSER,QAWAT,QBWAT,XASITE, &
       NSOLVR,NSOLA,NSOLB,NSOLX,QAPT,QBPT, &
       QAPROT,NAPROT,QBPROT,NBPROT, &
       QSAME,QARND,QLOC,RARND,XARND,XREF,YREF,ZREF,QREF, &
       XAREF,YAREF,ZAREF,XALGO,QPREC,QBRUTE,DOIMAGES,LIMALL,NN, &
       QMINI,QIMMAX,QMASS)
    !
    !     Print the parameters we're working with
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !
#if KEY_PROTO==1
    use proto_mod     
#endif
    !
    INTEGER OUTU,NATOM,NTIM,NCUBX,NCUBY,NCUBZ,NCUBE
    real(chm_real) RMAX,RHI,RARND,XREF,YREF,ZREF,QREF
    real(chm_real) XAREF,YAREF,ZAREF
    INTEGER XASITE,NSOLVR,NSOLA,NSOLB,NSOLX,XARND,XALGO,NN
    INTEGER NAPROT,NBPROT
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER,QAPT,QBPT
    LOGICAL QAPROT,QBPROT
    LOGICAL QSAME,QAWAT,QBWAT,QARND,QLOC,QPREC,QBRUTE
    LOGICAL DOIMAGES,LIMALL,QMINI,QIMMAX,QMASS
    !
    INTEGER I
    !
    WRITE (OUTU,'(/,A)') &
         ' RDFSOL computing particle interactions using parameters:'
    WRITE (OUTU,801) '  Rmax checked                   =', RHI
    WRITE (OUTU,800) '  Number of primary particles    =', NATOM
    !
    IF(DOIMAGES) THEN
       WRITE (OUTU,800) '  Number of image particles      =', NTIM
       WRITE (OUTU,800) '  Counting prim-img interactions x', NN
       IF(LIMALL) THEN
          WRITE (OUTU,'(A)') '  Building all (redundand) images'
       ELSE
          WRITE (OUTU,'(A)') '  Building only non-redundand images'
       ENDIF
    ENDIF
    !
    IF(.NOT.QBRUTE) THEN
       WRITE (OUTU,'(/,A)') '  Using cube algorithm:'
       WRITE (OUTU,800) '  Number of cells in X dimension =', NCUBX
       WRITE (OUTU,800) '  Number of cells in Y dimension =', NCUBY
       WRITE (OUTU,800) '  Number of cells in Z dimension =', NCUBZ
       WRITE (OUTU,800) '  Number of cells, total         =', NCUBE
    ELSE
       WRITE(OUTU,'(/,A)') &
            '  Using simple algorithm instead of cubing'
    ENDIF
    !
    IF(QRDF)  WRITE (OUTU,810) &
         ' Calculating radial distribution.'
    IF(QDDIP) WRITE (OUTU,810) &
         ' Calculating dipole-dipole correlation.'
    IF(QQDIP) WRITE (OUTU,810) &
         ' Calculating charge-dipole correlation.'
    IF(QHD)   WRITE (OUTU,810) &
         ' Calculating dipole-dipole h_D function.'
    IF(QUSER) WRITE (OUTU,810) &
         ' Calculating (?) (User defined function USRRDF).'
    !
    WRITE (OUTU,'(A)') ''
    WRITE (OUTU,'(A)') ' Used sets:'
    WRITE (OUTU,'(A)') '  Set A:'
    IF(XASITE == 0) THEN
       IF(QAWAT) THEN
          WRITE(OUTU,811) &
               '   TIP3P water (', NSOLVR, ' residues)'
#if KEY_PROTO==1
       ELSE IF(QAPROT.AND. NAPROT>0) THEN
          IF(QPRTO(NAPROT)) then
             WRITE(OUTU,810) '   A prototype:'
             CALL PRNPRT(NAPROT)
          endif
#endif 
       ELSE
          WRITE (OUTU,812)   '   a set (', NSOLA,' atoms)'
          IF(QAPT) THEN
             WRITE (OUTU,'(A)') &
                  '     (c.o.m. and dipole of whole set)'
          ELSE
             WRITE (OUTU,'(A)') '     (average over all atoms)'
          ENDIF
       ENDIF
    ELSE IF(XASITE == 1) THEN
       WRITE (OUTU,813) '   A fixed point in space: X=', &
            XREF, ' Y=', YREF, ' Z=', ZREF, &
            ' (CHARGE=',QREF, ')'
    ELSE IF(XASITE == 2) THEN
       WRITE (OUTU,812) '   The center of mass of ', &
            NSOLA, ' atoms.'
    ENDIF
    !
    WRITE (OUTU,'(A)') '  Set B:'
    IF(QSAME) THEN
       WRITE (OUTU,'(A)') &
            '   same as set A'
    ELSE
       IF(QBWAT) THEN
          WRITE (OUTU,811) &
               '   TIP3P water (', NSOLVR, ' residues)'
#if KEY_PROTO==1
       ELSE IF(QBPROT.AND.QPRTO(NBPROT)) THEN
          WRITE(OUTU,810) '   A prototype:'
          CALL PRNPRT(NBPROT)
#endif 
       ELSE
          WRITE (OUTU,812)'   site(', NSOLB,' atoms)'
          IF(QBPT) WRITE (OUTU,'(A)') &
               '     (c.o.m. and dipole of whole set)'
       ENDIF
    ENDIF
    !
    IF(QARND)THEN
       IF(QLOC) THEN
          WRITE(OUTU,'(/,A,F6.2,A)') &
               '  Limiting both sets ',RARND,' A around:'
       ELSE
          WRITE(OUTU,'(/,A,F6.2,A)') &
               '  Limiting set A ',RARND,' A around:'
       ENDIF
       IF(XARND == 1) THEN
          WRITE (OUTU,813) '   A fixed point in space: X=', &
               XAREF, ' Y=', YAREF, ' Z=', ZAREF
       ELSE
          WRITE(OUTU,812) &
               '   The center of mass of ',NSOLX,' atoms.'
       ENDIF
    ENDIF
    !
    IF(QPREC.AND.QAWAT) WRITE(OUTU,'(/,A)') &
         '  Using the precise water RDF algorithm'
    !
    IF(QMINI) WRITE(OUTU,'(/,A)') &
         '  Using _REAL_ minimum image convention'
    !
    IF(QIMMAX) WRITE(OUTU,'(/,A)') &
         '  Only updating image coordinates.'
    !
    IF(QMASS) WRITE(OUTU,'(/,A)') &
         '  Using mass-weighting for charged molecules/sets.'
    !
    WRITE(OUTU,'(/)')
    !
800 FORMAT(A,I7)
801 FORMAT(A,F7.3)
810 FORMAT(/,A)
811 FORMAT(A,I10,A,I10,A)
812 FORMAT(A,I10,A)
813 FORMAT(A,4(F6.2,A))
    !
    RETURN
  END SUBROUTINE PRNPAR
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE ADDIMG(NSET,SET,SETSL,NNSET,NATOM,NATIM,IMATT)
    !
    !     routine that that checks all image atoms and adds them
    !     to SET if the primary atom an image was created from is part of
    !     the original set (SETSL-flag-list). NNSET then is the last atom
    !     in the set (i.e. 1-NSET are the primaries and (NSET+1)-NNSET the
    !     images).
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !

    INTEGER NSET,SET(*),SETSL(*),NNSET,NATOM,NATIM,IMATT(*)
    !
    INTEGER I,K
    !
    NNSET=NSET
    DO I=NATOM+1,NATIM
       K=IMATT(I)
       IF(SETSL(K) > 0) THEN
          NNSET=NNSET+1
          SET(NNSET)=I
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE ADDIMG
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE REMAP(SET,NSET,NSETI,FLSEL,IMATT,NATOM,NATIM)
    !
    !     Remap a set of atoms. This means: include all primary atoms
    !     of which images are included. Then add all images of all these.
    !     (Think of a sphere around a corner of a square(2D). First generate
    !     all quarter-spheres in primary space and then add all images of these
    !     quarters. This will avoid having to check img-img interactions later on)
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    !
    INTEGER SET(*),NSET,NSETI,FLSEL(*),IMATT(*),NATOM,NATIM
    !
    INTEGER I,K
    LOGICAL Q
    !
    !     1: find all primaries
    DO I=(NSET+1),NSETI
       K=IMATT(SET(I))
       Q=.FALSE.
       !        is the corresponding primary (K) of image I already in set
       CALL NINSET(K,NSET,SET,Q)
       IF(.NOT.Q) THEN
          !           so it's not there -> include it
          NSET=NSET+1
          SET(NSET)=K
       ENDIF
    ENDDO
    !
    !     2: build a selection like flag array in flsel
    DO I=1,NATOM
       FLSEL(I)=0
    ENDDO
    DO I=1,NSET
       FLSEL(SET(I))=1
    ENDDO
    !
    !     3: add the images
    CALL ADDIMG(NSET,SET,FLSEL,NSETI,NATOM,NATIM,IMATT)
    !
    RETURN
  END SUBROUTINE REMAP
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE NINSET(N,NSET,SET,Q)
    !
    !     checks if atom nr N is a part of SET
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !
    INTEGER N,NSET,SET(*)
    LOGICAL Q
    !
    INTEGER I
    !
    Q=.FALSE.
    DO I=1,NSET
       IF(SET(I) == N) THEN
          Q=.TRUE.
          RETURN
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE NINSET
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE CLMNGO(QRDF,QDDIP,QQDIP,QHD,QUSER,LNMINI,LNMIN2,MINI, &
       NSET,NSETI,SET,SETSL,DR,NBIN,GXX,QWAT)
    !
    !     Collects G1, G2 and G3 from the minimum distance matrix MINI
    !     which is of dimension LNMINI x LNMINI ( = LNMIN2)
    !
    !     assumes that FILGOO was used i.e. only one set (SET) was compared
    !     to itself
    !
    !     if N1 > N2 then MINI(N1,N2) contains the distance and MINI(N2,N1)
    !     the corresponding value
    !
    !     Sep. 2005: switch from XDIP to logicals to allow simultaneous comp.
    !
    use number

    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(LNMINI,LNMINI)
    INTEGER NSET,NSETI,SET(*),SETSL(*),NBIN
    real(chm_real) GXX(8,*),DR
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER,QWAT
    !
    INTEGER I,J,K,L,N1,N2,A,B,AA,BB,NB,FNC
    !
    !     now loop over all (primary) particles in the set
    DO I=1,(NSET-1)
       N1=SET(I)
       DO J=I+1,NSET
          N2=SET(J)
          !
          !           this should be unnecessary, as the list _should_ be
          !           ordered, but you never know... (comment these if they
          !           prove to be efficiency limiting)
          A=MAX(N1,N2)
          B=MIN(N1,N2)
          !
          !           p1-p2 (that is O-O if we calculate WATEr RDFs)
          IF(MINI(A,B) > ZERO) THEN
             !tr               NB=INT(MINI(A,B)/DR+HALF)+1
             NB=INT(MINI(A,B)/DR+HALF)
             IF((NB <= NBIN).AND.(NB > 0)) THEN
                IF(QRDF)  GXX(1,NB)=GXX(1,NB)+MINI(B,A)
                IF(QDDIP) GXX(4,NB)=GXX(4,NB)+MINI(B,A)
                IF(QQDIP) GXX(6,NB)=GXX(6,NB)+MINI(B,A)
                IF(QHD)   GXX(7,NB)=GXX(7,NB)+MINI(B,A)
                IF(QUSER) GXX(8,NB)=GXX(8,NB)+MINI(B,A)
             ENDIF
          ENDIF
          !
          IF(QRDF.AND.QWAT) THEN
             !              i.e. we calculate WATEr RDFs so we need GOH and GHH
             !
             !              O-H
             A=N1
             B=N2
             DO L=1,2
                DO K=1,2
                   !                    O-H
                   !                    CAUTION: here we DO need Min/Max
                   AA=MAX(A,B+K)
                   BB=MIN(A,B+K)
                   IF(MINI(AA,BB) > ZERO) THEN
                      !tr                        NB=INT(MINI(AA,BB)/DR)+1
                      NB=INT(MINI(AA,BB)/DR+HALF)
                      IF((NB <= NBIN).AND.(NB > 0)) &
                           GXX(2,NB)=GXX(2,NB)+MINI(BB,AA)
                   ENDIF
                ENDDO
                !
                !                 H1-H1 or H2-H2
                AA=MAX(A+L,B+L)
                BB=MIN(A+L,B+L)
                IF(MINI(AA,BB) > ZERO) THEN
                   !TR                     NB=INT(MINI(AA,BB)/DR)+1
                   NB=INT(MINI(AA,BB)/DR+HALF)
                   IF((NB <= NBIN).AND.(NB > 0)) &
                        GXX(3,NB)=GXX(3,NB)+MINI(BB,AA)
                ENDIF

                !
                !                 H1-H2
                AA=MAX(A+1,B+2)
                BB=MIN(A+1,B+2)
                IF(MINI(AA,BB) > ZERO) THEN
                   !TR                     NB=INT(MINI(AA,BB)/DR)+1
                   NB=INT(MINI(AA,BB)/DR+HALF)
                   IF((NB <= NBIN).AND.(NB > 0)) &
                        GXX(3,NB)=GXX(3,NB)+MINI(BB,AA)
                ENDIF
                !
                !                 and now swap A and B
                A=N2
                B=N1
                !
                !              end DO L=
             ENDDO
             !           end IF((XDIP == 2).AND.QWAT)
          ENDIF
          !
          !        end DO J=
       ENDDO
       !     end DO I=
    ENDDO
    !
    RETURN
  END SUBROUTINE CLMNGO
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE CLMNG2(QRDF,QDDIP,QQDIP,QHD,QUSER, &
       LNMINI,LNMIN2,MINI, &
       NSETA,NSETAI,SETA,SETASL, &
       NSETB,NSETBI,SETB,SETBSL, &
       DR,NBIN,GXX,QAWAT,QBWAT)
    !
    !     Collects GXX from the minimum distance matrix MINI
    !     which is of dimension LNMINI x LNMINI ( = LNMIN2)
    !
    !     assumes that FILGXY was used
    !
    !     if N1 > N2 then MINI(N1,N2) contains the distance and MINI(N2,N1)
    !     the corresponding value
    !
    !     Tibor Rudas Mar 2003 - Jun 2003
    !
    !     Sep. 2005: switch from XDIP to logicals to allow simultaneous comp.
    !
    use number

    INTEGER LNMINI,LNMIN2
    real(chm_real4) MINI(LNMINI,LNMINI)
    INTEGER NSETA,NSETAI,SETB(*),SETASL(*)
    INTEGER NSETB,NSETBI,SETA(*),SETBSL(*),NBIN
    real(chm_real) GXX(8,*),DR
    LOGICAL QRDF,QDDIP,QQDIP,QHD,QUSER,QAWAT,QBWAT
    !
    INTEGER I,J,K,L,N1,N2,A,B,AA,BB,NB,FNC
    !
    !     now loop over all (primary) particles in the set
    DO I=1,NSETA
       N1=SETA(I)
       DO J=1,NSETB
          N2=SETB(J)
          !
          A=MAX(N1,N2)
          B=MIN(N1,N2)
          !
          !           p1-p2 (that is O-O if we calculate WATEr RDFs)
          IF(MINI(A,B) > ZERO) THEN
             !TR               NB=INT(MINI(A,B)/DR+HALF)+1
             NB=INT(MINI(A,B)/DR+HALF)
             IF((NB <= NBIN).AND.(NB > 0)) THEN
                IF(QRDF)  GXX(1,NB)=GXX(1,NB)+MINI(B,A)
                IF(QDDIP) GXX(4,NB)=GXX(4,NB)+MINI(B,A)
                IF(QQDIP) GXX(6,NB)=GXX(6,NB)+MINI(B,A)
                IF(QHD)   GXX(7,NB)=GXX(7,NB)+MINI(B,A)
                IF(QUSER) GXX(8,NB)=GXX(8,NB)+MINI(B,A)
             ENDIF
          ENDIF
          !
          IF(QRDF.AND.(QAWAT.OR.QBWAT)) THEN
             !              i.e. we calculate WATEr RDFs so we need GOH and GHH
             !
             !              make B hold the water
             A=N1
             B=N2
             IF(QAWAT) THEN
                A=N2
                B=N1
             ENDIF
             DO L=1,2
                !                 X-H
                AA=MAX(A,B+L)
                BB=MIN(A,B+L)
                IF(MINI(AA,BB) > ZERO) THEN
                   !TR                     NB=INT(MINI(AA,BB)/DR)+1
                   NB=INT(MINI(AA,BB)/DR+HALF)
                   IF((NB <= NBIN).AND.(NB > 0)) &
                        GXX(2,NB)=GXX(2,NB)+MINI(BB,AA)
                ENDIF
             ENDDO
             !
             !           end IF(QRDF.AND.(QAWAT.OR.QBWAT))
          ENDIF
          !
          !        end DO J=
       ENDDO
       !     end DO I=
    ENDDO
    !
    RETURN
  END SUBROUTINE CLMNG2
  !
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  !
  SUBROUTINE CPCOR(X,Y,Z,AX,AY,AZ,NAT,ATM,QWAT)
    !     fills corrdinates x/y/z of atm(1..nat) into ax/ay/az
    !
    use number
    real(chm_real) X(*),Y(*),Z(*),AX(*),AY(*),AZ(*)
    INTEGER NAT,ATM(NAT)
    LOGICAL QWAT
    !
    INTEGER I,N
    !
    DO I=1,NAT
       N=ATM(I)
       AX(N)=X(N)
       AY(N)=Y(N)
       AZ(N)=Z(N)
    ENDDO
    IF(QWAT) THEN
       DO I=1,NAT
          N=ATM(I)+1
          AX(N)=X(N)
          AY(N)=Y(N)
          AZ(N)=Z(N)
          N=ATM(I)+2
          AX(N)=X(N)
          AY(N)=Y(N)
          AZ(N)=Z(N)
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE CPCOR
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE WATDIP(NST,NSTI,SET,DIP,X,Y,Z,CH)
    !     fills dipole array for waters in SET
    !
    use number

    INTEGER NST,NSTI,SET(*)
    real(chm_real) DIP(4,*),X(*),Y(*),Z(*),CH(*)
    !
    INTEGER PIVOT,I,J,K
    !
    DO I=1,NST
       PIVOT=SET(I)
       CALL GWDIP(SET(I),X,Y,Z,CH, &
            DIP(1,PIVOT),DIP(2,PIVOT),DIP(3,PIVOT),DIP(4,PIVOT), &
            .TRUE.)
    ENDDO
    !
    !     image waters: maybe put some check in here?
    DO I=NST+1,NSTI
       PIVOT=SET(I)
       CALL GWDIP(SET(I),X,Y,Z,CH, &
            DIP(1,PIVOT),DIP(2,PIVOT),DIP(3,PIVOT),DIP(4,PIVOT), &
            .TRUE.)
    ENDDO
    !
    RETURN
    !
  END SUBROUTINE WATDIP
#endif /*        (rdfsol)*/


end module rdfsol_mod

