SUBROUTINE PRINTE(UNIT, EPRPX, ETRMX, ENAME, ETYPE, QHEADR, &
     CYCLE, TIME, STEP, QGRMS)
  !-----------------------------------------------------------------------
  !     PRINTE prints out the energy and energy terms in standard format.
  !     The calling sequence is :
  !
  !      UNIT     The unit for the printing.
  !      EPRPX,   The energy property array to be printed.
  !      ETRMX    The energy term array to be printed.
  !      ENAME    Name of energy printout
  !      ETYPE    Type of energy print ('DYN','MIN','ENR')
  !      QHEADR   Print header before printing values?
  !      CYCLE    Cycle or iteration number.
  !      TIME     The dynamics time.
  !      STEP     The minimization step.
  !
  !     By Bernard R. Brooks    1983
  !     12-FEB-2003, Youngdo Won, Adjustable FORMAT implemented
  !

#if KEY_CHEQ==1
  use cheq,only:qcg                                   
#endif
#if KEY_TSALLIS==1
  use tsallis_module, only:ebias,etsbias,qttsall,tsq  
#endif
#if KEY_FACTS==1
  use facts_module,only:fctrun                        
#endif

  use chm_kinds
  use dimens_fcm
  use number
  !
  use deriv
  use energym
#if KEY_LRVDW==1
  use inbnd                        
#endif
  use fourdm
  use psf
  use stream
  use ffieldm
#if KEY_PATHINT==1
  use mpathint, only: qpint  
#endif
  use galgor
#if KEY_SASAE==1
  use sasa
#endif 
  use dcm_fcm, only: qdcm, dcmeprint
#if KEY_QCHEM==1
  use gamess_fcm, only : QQEWALD
#endif 
#if KEY_LARMORD==1
  use larmord, only : qlarmord, qlarmord_on
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:qfhdgb
  use derivdhdgb
#endif

#if KEY_RDC==1
  use rdc, only: qrdc
#endif

  implicit none

  !     . Passed variables.
  INTEGER CYCLE, UNIT
  CHARACTER(len=4) ENAME
  CHARACTER(len=3) ETYPE
  LOGICAL QHEADR, QGRMS
  real(chm_real)  EPRPX(LENENP), ETRMX(LENENT), STEP, TIME
  !     . Local variables.
  INTEGER I, NDIM
  LOGICAL QDYNAM, QMINMZ, QENER, QEWALD
  LOGICAL QCONST, QEXTER, QIMAGE, QINTER, QMISC, QPXTL, QRESTR
#if KEY_CONSHELIX==1
  LOGICAL QCONSH
#endif 
#if KEY_CHEQ==1
  LOGICAL QCGP                   
#endif
#if KEY_CMAP==1
  LOGICAL QCROSS                 
#endif
  !
  LOGICAL QMMFP, QMMFP2, QQUANT, QPBEQ
  LOGICAL QPOLAR
  !
  !epmf  SMG/MF 2008/2010
#if KEY_EPMF==1
  LOGICAL QEPMFPRN               
#endif
#if KEY_PRIMO==1
  LOGICAL QPRIMO                 
#endif
#if KEY_EDS==1
  LOGICAL QEDSENE                
#endif

#if KEY_ACE==1
  LOGICAL QACE
#endif 
#if KEY_ADUMB==1 || KEY_RXNCOR==1 || KEY_GAMUS==1
  LOGICAL QADUMB
#endif 
#if KEY_GRID==1
  LOGICAL QGRD  
#endif
#if KEY_MC==1
  LOGICAL QMC
#endif 
#if KEY_FACTS==1
  LOGICAL QFCTPOL
  LOGICAL QFCTNPL
#endif 
#if KEY_SASAE==1
  LOGICAL QPSASA
#endif 
  LOGICAL QRUSH  
#if KEY_FLUCQ==1
  LOGICAL QFLUCQ
#endif 
#if KEY_PHMD==1
  LOGICAL QPHMDQ
#endif 
#if KEY_OVERLAP==1
  LOGICAL QLAPQ  
#endif
  ! PJ 06/2005
#if KEY_PIPF==1
  LOGICAL QPIPF  
#endif
  !
#if KEY_EMAP==1
  LOGICAL QEMAP  
#endif
  !
#if KEY_PNM==1
  LOGICAL QPNM   
#endif
#if KEY_SSNMR==1
  LOGICAL QSSNMR
#endif
#if KEY_LJPME==1
  logical qljpme
#endif

  real(chm_real)  DRMS, EDIF, EOTHER
  SAVE    EDIF
  real(chm_real), parameter :: ACCU=0.49D-6 ! should be consistent with format 146
  !yw++
  CHARACTER(len=5) F13
  CHARACTER(len=60) F144
  !yw--
  !
  !=======================================================================
  QDYNAM=(ETYPE == 'DYN')
  QMINMZ=(ETYPE == 'MIN')
  QENER =(ETYPE == 'ENR')
  !=======================================================================
#if KEY_MC==1
  QMC=(ENAME == 'MC E')
#endif 
  !=======================================================================
  IF(PRNLEV < 2) GOTO 800
  !     . Find out which values to print.
#if KEY_CMAP==1
  QCROSS = .FALSE.  
#endif
#if KEY_EPMF==1
  QEPMFPRN = .FALSE.  
#endif
#if KEY_PRIMO==1
  QPRIMO = .FALSE.    
#endif
#if KEY_EDS==1
  QEDSENE = .FALSE.   
#endif

#if KEY_CFF==1 || KEY_MMFF==1 /*cff_mmff*/
  qinter = (abs(etrmx(bond))   > accu) .or. &
       (abs(etrmx(angle))  > accu) .or. &
       (abs(etrmx(dihe))   > accu)
  if(ffield == charmm .or. ffield == amberffn) then
     qinter = qinter .or. &
          (abs(etrmx(ureyb))  > accu) .or. &
          (abs(etrmx(imdihe)) > accu)
#if KEY_CMAP==1
     qcross = (abs(etrmx(cmap)) > accu)    
#endif
#if KEY_EPMF==1
     qepmfprn = (abs(etrmx(pmf1d)) > accu) .or.(abs(etrmx(pmf2d)).gt.accu) 
#endif
#if KEY_PRIMO==1
     qprimo = (abs(etrmx(primo)) > accu)    
#endif

#if KEY_EDS==1
     QEDSENE = (ABS(ETRMX(EDS)).GT.ACCU)
#endif 
#if KEY_CFF==1
  elseif(ffield == cff) then
     qinter = qinter .or. &
          (abs(etrmx(strb))   > accu) .or. &
          (abs(etrmx(strstr)) > accu) .or. &
          (abs(etrmx(bndbnd)) > accu) .or. &
          (abs(etrmx(bndtw))  > accu) .or. &
          (abs(etrmx(ebst))   > accu) .or. &
          (abs(etrmx(mbst))   > accu) .or. &
          (abs(etrmx(bbt))    > accu) .or. &
          (abs(etrmx(sst))    > accu)
#endif 
#if KEY_MMFF==1
  elseif(ffield == mmff) then
     qinter = qinter .or. &
          (abs(etrmx(strb))   > accu) .or. &
          (abs(etrmx(oopl))   > accu)
#endif 
  endif

#else /* (cff_mmff)*/
  qinter = (abs(etrmx(bond))   > accu) .or. &
       (abs(etrmx(angle))  > accu) .or. &
       (abs(etrmx(ureyb))  > accu) .or. &
       (abs(etrmx(dihe))   > accu) .or. &
       (abs(etrmx(imdihe)) > accu)
#if KEY_CMAP==1
  qcross= (abs(etrmx(cmap)) > accu)    
#endif
#endif /* (cff_mmff)*/

  qexter = (abs(etrmx(vdw))    > accu) .or. &
       (abs(etrmx(elec))   > accu) .or. &
       (abs(etrmx(hbond))  > accu) .or. &
       (abs(etrmx(user))   > accu) .or. &
       (abs(etrmx(asp))    > accu)
  qimage = (abs(etrmx(imvdw))  > accu) .or. &
       (abs(etrmx(imelec)) > accu) .or. &
       (abs(etrmx(imhbnd)) > accu) .or. &
       (abs(etrmx(rxnfld)) > accu) .or. &
       (abs(etrmx(extnde)) > accu)
  qewald = (abs(etrmx(ewksum)) > accu) .or. &
       (abs(etrmx(ewself)) > accu) .or. &
       (abs(etrmx(ewexcl)) > accu) .or. &
       (abs(etrmx(ewqcor)) > accu) .or. &
       (abs(etrmx(ewutil)) > accu)
  qconst = (abs(etrmx(charm))  > accu) .or. &
       (abs(etrmx(cdihe))  > accu) .or. &
       (abs(etrmx(cintcr)) > accu) .or. &
       (abs(etrmx(resd))   > accu) .or. &
       (abs(etrmx(noe))    > accu) .or. &
       (abs(etrmx(pcharm))   > accu)
#if KEY_CONSHELIX==1
  QCONSH = (ABS(ETRMX(ECHDL)) .GT.ACCU) .OR. &
       (ABS(ETRMX(ECHAN)) .GT.ACCU) .OR. &
       (ABS(ETRMX(ECHTN)) .GT.ACCU) .OR. &
       (ABS(ETRMX(ECHRN)) .GT.ACCU)
#endif 
  qrestr = (abs(etrmx(cqrt))   > accu) .or. &
#if KEY_HMCOM==1
       (abs(etrmx(hmcm))   > accu) .or. &    
#endif
#if KEY_CPATH==1
       (abs(etrmx(path))   > accu) .or. &    
#endif
       (abs(etrmx(eharm))  > accu) .or. &
       (abs(etrmx(shap))   > accu) .or. &
       (abs(etrmx(dmc))    > accu) .or. &
       (abs(etrmx(rgy))    > accu)
  qpolar = (abs(etrmx(polar))  > accu) .or. &
       (abs(etrmx(pull))   > accu)
  eother= etrmx(st2)+etrmx(imst2)
  qmisc  = (abs(etrmx(sbndry)) > accu) .or. &
       (abs(etrmx(tsm))    > accu) .or. &
#if KEY_RPATH==1
       (abs(etrmx(prms))   > accu) .or.   & 
#endif
#if KEY_RPATH==1
       (abs(etrmx(pang))   > accu) .or.   & 
#endif
       (abs(eother    )    > accu)
  qmmfp  = (abs(etrmx(geo))    > accu) .or. &
       (abs(etrmx(mdip))   > accu) .or. &
       (abs(etrmx(ssbp))   > accu) .or. &
       (abs(etrmx(shel))   > accu)

  ! CHR Extend MMFP printing to get room for new terms (here VMOD)
  qmmfp2 = (abs(etrmx(vmod))   > accu)
  qpbeq  = (abs(etrmx(pbelec)) > accu) .or. &
       (abs(etrmx(pbnp))   > accu) .or. &
       (abs(etrmx(gbenr))  > accu) .or. &
       (abs(etrmx(gsbp))   > accu) .or. &
#if KEY_DHDGB==1
!AP/MF
       (abs(etrmx(defe)) > accu) .or. &
#endif
       (abs(etrmx(smbp))   > accu) 
  qpxtl  =((abs(eprpx(xtlpe))  > accu) .or. &
       (abs(eprpx(xtlke))  > accu)).and. qdynam
  qquant = (abs(etrmx(qmel))   > accu) .or. &
       (abs(etrmx(qmvdw))  > accu)

#if KEY_ACE==1
  qace  =  (abs(etrmx(hydr))    > accu) .or. &
       (abs(eprpx(self))    > accu) .or. &
       (abs(eprpx(screen))  > accu) .or. &
       (abs(eprpx(coul))    > accu) .or. &
       (abs(eprpx(solv))    > accu) .or. &
       (abs(eprpx(inter))   > accu)
#endif 
#if KEY_ADUMB==1 || KEY_RXNCOR==1
  qadumb = (abs(etrmx(adumb))   > accu) .or. &
       (abs(etrmx(umbr))    > accu)
#if KEY_GAMUS==1
  qadumb = qadumb .or. (abs(etrmx(gamus)) > accu)
#endif 
#endif 
#if KEY_CHEQ==1
  qcgp =   (abs(eprpx(cgke))   > accu) .or. &
       (abs(eprpx(cgpot))  > accu)
  qcgp =   qcgp.and.qcg.and.qeterm(elec)
#endif
!! Yujin 
#if KEY_GRID==1 
  qgrd = (abs(etrmx(grvdw)+abs(etrmx(grelec))+abs(etrmx(grhbon)))   > accu) 
#endif
#if KEY_FACTS==1
  if (fctrun) then
     qfctpol = abs(etrmx(ifctpol)) > accu
     qfctnpl = abs(etrmx(ifctnpl)) > accu
  endif
#endif 
#if KEY_SASAE==1
  if (qsasa) then
     qpsasa = abs(etrmx(sastrm))  >  accu
  endif
#endif 
  qrush = ( abs(etrmx(rushrepu)+etrmx(rushphob)+etrmx(rushhbnd) &
       +etrmx(rushbdon)+etrmx(rushbacc)+etrmx(rusharom))  >  accu )
#if KEY_FLUCQ==1
  qflucq = (abs(etrmx(fqpol))   > accu) .or. &
       (abs(eprpx(fqkin))   > accu)
#endif 
#if KEY_PHMD==1
  qphmdq = (abs(etrmx(phenr))   > accu) .or. &
       (abs(eprpx(phkin))   > accu)
#endif 
#if KEY_OVERLAP==1
  qlapq  = (abs(etrmx(qovlap))  > accu)   
#endif

#if KEY_PIPF==1
  qpipf  = (abs(etrmx(pipf))  > accu) .or. &
       (abs(eprpx(dipk))  > accu) .or. &
       (abs(eprpx(dipt))  > accu)
#endif 
#if KEY_PNM==1
  qpnm = abs(etrmx(pnme))  >  accu         
#endif
#if KEY_EMAP==1
      QEMAP  = (ABS(ETRMX(EEMAP)) .GT.ACCU)
#endif 
#if KEY_SSNMR==1
      QSSNMR = (ABS(ETRMX(ECS)).GT.ACCU)
#endif
#if KEY_LJPME==1
    qljpme = abs(etrmx(ljrec)) > accu .or. abs(etrmx(ljsel)) > accu .or. abs(etrmx(ljexc)) > accu
#endif
  !
  !=======================================================================
  !     . Print out the headers if required.
  if (qheadr .and. prnlev >= 3) then
#if KEY_GENETIC==1
     if(qmonte) then
        IF (QENER) WRITE(UNIT,'(2A)') ENAME, &
             ' ENR:  Eval#     ENERgy      Delta-E       ACCEPT'
     elseif(qgalgor) then
        if (qener) write(unit,'(2a)') ename, &
             ' ENR:  Eval#     ENERgy      Delta-E         RANK'
     ELSE
        IF (QENER) WRITE(UNIT,'(2A)') ENAME, &
             ' ENR:  Eval#     ENERgy      Delta-E         GRMS'
     ENDIF
#else /**/
     IF (QENER) WRITE(UNIT,'(2A)') ENAME, &
          ' ENR:  Eval#     ENERgy      Delta-E         GRMS'
#endif 
     IF (QMINMZ) WRITE(UNIT,'(3A)') ENAME, &
          ' MIN: Cycle      ENERgy      Delta-E         GRMS    ', &
          'Step-size'
     IF (QDYNAM) THEN
        WRITE(UNIT,'(3A)') ENAME, &
             ' DYN: Step         Time      TOTEner  ', &
             '      TOTKe       ENERgy  TEMPerature'
        WRITE(UNIT,'(3A)') ENAME, &
             ' PROP:             GRMS      HFCTote  ', &
             '      HFCKe       EHFCor        VIRKe'
     ENDIF
     IF (QINTER) THEN
#if KEY_CFF==1 || KEY_MMFF==1
        if(ffield == charmm .or. ffield == amberffn) then
#endif 
           WRITE(UNIT,'(3A)') ENAME, &
                ' INTERN:          BONDs       ANGLes  ', &
                '     UREY-b    DIHEdrals    IMPRopers'
#if KEY_TSALLIS==1
           if (qttsall .and. &
                (ename /= 'AVER') .and. (ename.ne.'FLUC')) then
              write(unit,'(3A)')'DYNA', &
                   ' TSALLIS:         EBIAS      ETSBIAS', &
                   '       QTSALL'
           endif
#endif 
#if KEY_CFF==1
        ELSEIF(FFIELD == CFF) THEN
           WRITE(UNIT,'(3A)') ENAME, &
                ' INTERN:          BONDs       ANGLes  ', &
                '  DIHEdrals      OOPLane'
           WRITE(UNIT,'(3A)') ENAME, &
                '  CROSS:        STRBend   STRSTRetch  ', &
                '    BNDBend     BNDTwist'
           WRITE(UNIT,'(3A)') ENAME, &
                '  CROSS:       EBSTwist     MBSTwist  ', &
                '    BBTwist      SSTwist'
#endif 
#if KEY_MMFF==1
        ELSEIF(FFIELD == MMFF) THEN
           WRITE(UNIT,'(3A)') ENAME, &
                ' INTERN:          BONDs       ANGLes  ', &
                '    STRBend    DIHEdrals      OOPLane '
#endif 
#if KEY_CFF==1 || KEY_MMFF==1
        ENDIF
#endif 
     ENDIF

#if KEY_CMAP==1
#if KEY_EPMF==1 /*epmf*/
#if KEY_PRIMO==1
     IF (QCROSS .OR. QEPMFPRN .OR. QPRIMO) THEN
        WRITE(UNIT,'(4A)') ENAME, &
             ' CROSS:           CMAPs     ', &
             '   PMF1D        PMF2D', &
             '        PRIMO'
     ENDIF
#else /**/
     IF (QCROSS .OR. QEPMFPRN) THEN
        WRITE(UNIT,'(3A)') ENAME, &
             ' CROSS:           CMAPs     ', &
             '   PMF1D        PMF2D' 
     ENDIF
#endif 
#else /* (epmf)*/
     IF (QCROSS) THEN
        WRITE(UNIT,'(2A)') ENAME, &
             ' CROSS:           CMAPs     '
     ENDIF
#endif /* (epmf)*/
#endif 

     IF (QEXTER) WRITE(UNIT,'(3A)') ENAME, &
          ' EXTERN:        VDWaals         ELEC  ', &
          '     HBONds          ASP         USER'
     IF (QIMAGE) WRITE(UNIT,'(3A)') ENAME, &
          ' IMAGES:        IMNBvdw       IMELec  ', &
          '     IMHBnd       RXNField    EXTElec'
     IF (QEWALD) WRITE(UNIT,'(3A)') ENAME, &
          ' EWALD:          EWKSum       EWSElf  ', &
          '     EWEXcl       EWQCor       EWUTil'
     IF (QCONST) WRITE(UNIT,'(3A)') ENAME, &
          ' CONSTR:       HARMonic    CDIHedral  ', &
          '        CIC     RESDistance       NOE'
#if KEY_CONSHELIX==1
     IF (QCONSH) WRITE(UNIT,'(3A)') ENAME, &
          ' CONSTR:        HH_Dist       HH_Ang  ', &
          '     H_Tilt       H_Rota'
#endif 
     IF (QRESTR) WRITE(UNIT,'(3A)') ENAME, &
          ' RESTR:        CDROplet    EHARmonic  ', &
          '      SHAPe        DMC            RGY'
     IF (QPOLAR) WRITE(UNIT,'(2A)') ENAME, &
          ' POLAR:          EPOLAR         PULL'
     IF (QMISC) WRITE(UNIT,'(4A)') ENAME, &
          ' MISCEL:         SBOUnd          TSM  ', &

#if KEY_RPATH==1
          '      PRMS        PANGle        ', &
#else /**/
          '                                ', &
#endif 
          'other'


#if KEY_EDS==1
     IF (QEDSENE) WRITE(UNIT,'(2A)') ENAME, &
          ' EDS:               EDS'
#endif 

     IF (QQUANT) WRITE(UNIT,'(2A)') ENAME, &
          ' QUANTUM:        QMELec        QMVDw  '
     IF (QMMFP) WRITE(UNIT,'(3A)') ENAME, &
          ' MMFP:              GEO        MDIP    ', &
          '       SSBP        SHEL       DROFfa'

     ! CHR Additional MMFP lines provided (here for VMOD)
     IF (QMMFP2) WRITE(UNIT,'(3A)') ENAME, &
          ' MMFP2:            VMOD'

#if KEY_PATHINT==1
     IF (QPINT) WRITE(UNIT,'(3A)') ENAME, &
          ' PINT:'
#endif 

#if KEY_CHEQ==1
     IF (QCGP) WRITE(UNIT,'(3A)') ENAME, &
          ' CHEQ:           CGKE          CGPOT ', &
          '      ALLK         ALLP          ALLE'
#endif
#if KEY_DHDGB==1
!AP/MF
     IF (QPBEQ) THEN
        IF (QFHDGB) THEN
            WRITE(UNIT,'(3A)') ENAME, &
            ' PBEQ:             PBnp       PBelec        GBEnr',&
            '         DEFE         SMBP'
        ELSE
            WRITE(UNIT,'(3A)') ENAME, &
            ' PBEQ:             PBnp       PBelec        GBEnr',&
            '         GSBP         SMBP'
        ENDIF
     ENDIF
#else /**/
     IF (QPBEQ) WRITE(UNIT,'(3A)') ENAME, &
          ' PBEQ:             PBnp       PBelec        GBEnr', &
          '         GSBP         SMBP'
#endif 
#if KEY_GRID==1
     IF (QGRD) WRITE(UNIT,'(2A)') ENAME, &
          ' GRID:             GrvdW       GrElec       GrHBON'
#endif 
     IF (QDYNAM &
#if KEY_MC==1
     .OR.QMC &
#endif 
     ) WRITE(UNIT,'(3A)') ENAME, &
          ' PRESS:            VIRE         VIRI   ', &
          '    PRESSE       PRESSI       VOLUme'
     IF (QPXTL) WRITE(UNIT,'(3A)') ENAME, &
          ' XTLE:                       XTLTe     ', &
          '    SURFtension  XTLPe        XTLtemp'
#if KEY_LRVDW==1
     if(LLRVDW .OR. LLRVDW_MS)write(UNIT,'(3A)') ENAME, &    
#endif
#if KEY_LRVDW==1
          ' LRCor:            EVDW         VIRI   '          
#endif
#if KEY_LJPME==1
     IF (qljpme) WRITE(UNIT,'(3A)') ENAME, &
          ' LJPME:           LJREC       LJSElf       LJEXcl'
#endif

     !CC Note: SF has "borrowed" the XTLKe column to report surface tension
     !CC       This will be corrected in future versions.
     !CC     $      '    XTLKe        XTLPe        XTLtemp'
#if KEY_FOURD==1 /*4dheader*/
     IF (DIM4.AND.QDYNAM) WRITE(UNIT,'(3A)') ENAME, &
          '             ENER-EFOUR        TOTE4    ', &
          '    TOTK4        EFOUR        TEMP4      '
     IF (DIM4.AND.QMINMZ) WRITE(UNIT,'(3A)') ENAME, &
          '                 EFOUR    ENER-EFOUR     '
     IF (DIM4.AND.QENER) WRITE(UNIT,'(3A)') ENAME, &
          '                 EFOUR    ENER-EFOUR     '
#endif /* (4dheader)*/
#if KEY_ACE==1
     IF (QACE) WRITE(UNIT,'(3A)') ENAME, &
          ' ACE1:      HYDRophobic         SELF  ', &
          '  SCREENing      COULomb '
     IF (QACE) WRITE(UNIT,'(3A)') ENAME, &
          ' ACE2:        SOLVation  INTERaction '
#endif 
#if KEY_ADUMB==1 || KEY_RXNCOR==1 || KEY_GAMUS==1
     IF (QADUMB) WRITE(UNIT,'(2A)') ENAME, &
          ' UMBR:            ADUMB       RXNCor        GAMUS'
#endif 
#if KEY_FACTS==1
     IF (FCTRUN) THEN
        IF (QFCTPOL)  WRITE(UNIT,'(3A)') ENAME, &
             ' FCTPOL:         FCTPOL'
        IF (QFCTNPL)  WRITE(UNIT,'(3A)') ENAME, &
             ' FCTNPL:         FCTNPL'
     ENDIF
#endif 
#if KEY_SASAE==1
     IF (QSASA) THEN
        IF (QPSASA) WRITE(UNIT,'(3A)') ENAME, &
             ' SOLVAT:           SASL'
     ENDIF
#endif 
     IF (QRUSH) WRITE(UNIT,'(3A)') ENAME, &
          ' RUSH1:       REPUlsion  hydroPHOBic  ', &
          '      HBoND  BuriedDONOr BuriedACCept'
     IF (QRUSH) WRITE(UNIT,'(3A)') ENAME, &
          ' RUSH2:        AROMatic               ', &
          '                                     '
#if KEY_FLUCQ==1
     IF (QFLUCQ) WRITE(UNIT,'(3A)') ENAME, &
          ' FLUCQ:         FQPOlar    FQKInetic '
#endif 
#if KEY_PHMD==1
     IF (QPHMDQ) WRITE(UNIT,'(3A)') ENAME, &
          ' PHMD:            PHENr    PHKInetic '
#endif 
#if KEY_OVERLAP==1
     IF (QLAPQ) WRITE(UNIT,'(3A)') ENAME, &
          '   OLAP:           OLAP'
#endif 
     ! PJ 06/2005
#if KEY_PIPF==1
     IF (QPIPF) WRITE(UNIT,'(3A)') ENAME, &
          ' PIPF:             EPOL        DIPKe      DIPTemp'
#endif 
#if KEY_PNM==1
     IF (QPNM) WRITE(UNIT,'(3A)') ENAME, &
          '  PNM:              PNM'
#endif 
#if KEY_EMAP==1
      IF (QEMAP) WRITE(UNIT,'(3A)') ENAME, &
           ' EMAP:            EEMAP'
#endif 
#if KEY_SSNMR==1
      IF (QSSNMR) WRITE(UNIT,'(3A)') ENAME, &
           'SSNMR:             ECS'
#endif
#if KEY_RDC==1
        IF (QRDC) WRITE(UNIT,'(3A)') ENAME,           &
           '  RDC:             RDC'
#endif
     !
     WRITE(UNIT,'(2A)') ' ----------       ---------    ---------', &
          '    ---------    ---------    ---------'
  ENDIF
  !=======================================================================
  ! . Calculate some properties for the minimization option.
  IF (QMINMZ.OR.QENER) THEN
     ! . Calculate the energy difference.
#if KEY_MC==1 /*mc1*/
     !       ARD:  For MC force it to calculate the difference.
     !             Unclear to me why one would want to suppress this calculation.
     IF (QMC) THEN
        EDIF = EOLD - EPRPX(EPOT)
     ELSE
#endif /* (mc1)*/
        IF (EOLD /= EPRPX(EPOT)) EDIF = EOLD - EPRPX(EPOT)
#if KEY_MC==1 /*mc2*/
     ENDIF
#endif /* (mc2)*/
     IF (EOLD == 0.0) EDIF=ZERO
     ! . Set the old energy to the new value.
     EOLD = EPRPX(EPOT)
     ! . Calculate the RMS gradient.
     NDIM = 0
     DRMS = ZERO
     !
#if KEY_FOURD==1 /*4drmsf*/
     IF(DIM4) THEN
        DO I = 1,NATOM
           IF (IMOVE(I)  ==  0) THEN
              NDIM = NDIM + 4
              DRMS=DRMS+DX(I)*DX(I)+DY(I)*DY(I) &
                   +DZ(I)*DZ(I)+DFDIM(I)*DFDIM(I)
           ENDIF
        ENDDO
     ELSE
#endif /* (4drmsf)*/
        DO I = 1,NATOM
           IF (IMOVE(I)  ==  0) THEN
              NDIM = NDIM + 3
              DRMS = DRMS + DX(I)*DX(I) + DY(I)*DY(I) + DZ(I)*DZ(I)
           ENDIF
        ENDDO
#if KEY_DHDGB==1
!AP/MF
       IF(QFHDGB) THEN
          NDIM=NDIM+TOTALS2
           DO I=1,TOTALS2
              DRMS=DRMS+DS_DHDGB(I)**2
           ENDDO
       ENDIF
#endif
#if KEY_FOURD==1 /*4dendif*/
     ENDIF
#endif /* (4dendif)*/
     !
#if KEY_GENETIC==1
     if(.not.qgalgor)  then
#endif 
        !     Sometimes we get GRMS from other processes, dont overwrite!
        IF(QGRMS)THEN
           IF (NDIM  >  0) THEN
              EPRPX(GRMS) = SQRT(DRMS / NDIM)
           ELSE
              EPRPX(GRMS) = ZERO
           ENDIF
        ENDIF
#if KEY_GENETIC==1
     endif
#endif 
  ENDIF
  !=======================================================================
  ! . Print out the energy values.
  ! 144  FORMAT(A4,A1,I9,5F13.5)
  ! 146  FORMAT(A4,A8,2X,5F13.5)
  ! 149  FORMAT(A4,A8,15X,4F13.5)
  !
  call setfmt(eprpx(epot),eprpx(grms),edif,ZERO,ZERO,F13)
  F144='(A4,A1,I9,5'//F13//')'
#if KEY_GENETIC==1
  if(qgalgor.and.(.not.qmonte)) then
     IF (QENER) WRITE(UNIT,'(A4,A1,I9,2F13.5,i10)') ENAME,'>',CYCLE, &
          EPRPX(EPOT),EDIF,int(EPRPX(GRMS))
  else
     IF (QENER) WRITE(UNIT,F144) ENAME,'>',CYCLE, &
          EPRPX(EPOT),EDIF,EPRPX(GRMS)
  endif
#else /**/
  IF (QENER) WRITE(UNIT,F144) ENAME,'>',CYCLE, &
       EPRPX(EPOT),EDIF,EPRPX(GRMS)
#endif 
  IF (QMINMZ) WRITE(UNIT,F144) ENAME,'>',CYCLE, &
       EPRPX(EPOT),EDIF,EPRPX(GRMS),STEP
  IF (QDYNAM) THEN
     call setfmt(eprpx(tote),eprpx(totke),eprpx(epot), &
          eprpx(temps),ZERO,F13)
     F144='(A4,A1,I9,5'//F13//')'
     WRITE(UNIT,F144) ENAME,'>',CYCLE,TIME,EPRPX(TOTE),EPRPX(TOTKE), &
          EPRPX(EPOT),EPRPX(TEMPS)
     call setfmt(eprpx(grms),eprpx(hfcte),eprpx(hfcke),eprpx(ehfc), &
          eprpx(virke),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' PROP>  ', EPRPX(GRMS), EPRPX(HFCTE), &
          EPRPX(HFCKE), EPRPX(EHFC), EPRPX(VIRKE)
  ENDIF
  IF (QINTER) THEN
#if KEY_CFF==1 || KEY_MMFF==1
     if(ffield == charmm .or. ffield == amberffn) then
#endif 
        call setfmt(etrmx(bond),etrmx(angle),etrmx(ureyb), &
             etrmx(dihe),etrmx(imdihe),F13)
        F144='(A4,A8,2X,5'//F13//')'
        WRITE(UNIT,F144) ENAME,' INTERN>',ETRMX(BOND), &
             ETRMX(ANGLE),ETRMX(UREYB),ETRMX(DIHE),ETRMX(IMDIHE)
#if KEY_TSALLIS==1
        if (qttsall .and. &
             (ename /= 'AVER') .and. (ename.ne.'FLUC')) then
           write(unit,'(A4,A9,1X,3F13.5)')'DYNA',' TSALLIS>', &
                EBIAS,ETSBIAS,1.0-tsq
        endif
#endif 
#if KEY_CFF==1
     ELSEIF(FFIELD == CFF) THEN
        call setfmt(etrmx(bond),etrmx(angle),etrmx(dihe), &
             etrmx(imdihe),ZERO,F13)
        F144='(A4,A8,2X,4'//F13//')'
        WRITE(UNIT,F144) ENAME,' INTERN>',ETRMX(BOND), &
             ETRMX(ANGLE),ETRMX(DIHE),ETRMX(IMDIHE)
        call setfmt(etrmx(strb),etrmx(strstr),etrmx(bndbnd), &
             etrmx(bndtw),ZERO,F13)
        F144='(A4,A8,2X,4'//F13//')'
        WRITE(UNIT,F144) ENAME,'  CROSS>', &
             ETRMX(STRB),ETRMX(STRSTR),ETRMX(BNDBND),ETRMX(BNDTW)
        call setfmt(etrmx(ebst),etrmx(mbst),etrmx(bbt), &
             etrmx(sst),ZERO,F13)
        F144='(A4,A8,2X,4'//F13//')'
        WRITE(UNIT,F144) ENAME,'  CROSS>', &
             ETRMX(EBST),ETRMX(MBST),ETRMX(BBT),ETRMX(SST)
#endif 
#if KEY_MMFF==1
     ELSEIF(FFIELD == MMFF) THEN
        call setfmt(etrmx(bond),etrmx(angle),etrmx(strb), &
             etrmx(dihe),etrmx(oopl),F13)
        F144='(A4,A8,2X,5'//F13//')'
        WRITE(UNIT,F144) ENAME,' INTERN>',ETRMX(BOND), &
             ETRMX(ANGLE),ETRMX(STRB),ETRMX(DIHE),ETRMX(OOPL)
#endif 
#if KEY_CFF==1 || KEY_MMFF==1
     ENDIF
#endif 
  ENDIF

#if KEY_CMAP==1
#if KEY_EPMF==1 /*epmf*/
#if KEY_PRIMO==1
     IF(QCROSS .OR. QEPMFPRN .OR. QPRIMO) THEN
      call setfmt(etrmx(cmap),etrmx(pmf1d),etrmx(pmf2d),etrmx(primo),zero,f13)
      F144='(A4,A8,2X,4'//F13//')'
        WRITE(UNIT,F144) ENAME, ' CROSS> ',ETRMX(CMAP),ETRMX(PMF1D), &
              ETRMX(PMF2D),ETRMX(PRIMO)
     ENDIF
#else /**/
     IF(QCROSS .OR. QEPMFPRN) THEN
      call setfmt(etrmx(cmap),etrmx(pmf1d),etrmx(pmf2d),zero,zero,f13)
      F144='(A4,A8,2X,3'//F13//')'
        WRITE(UNIT,F144) ENAME, ' CROSS> ',ETRMX(CMAP),ETRMX(PMF1D), &
              ETRMX(PMF2D)
     ENDIF
#endif 
#else /* (epmf)*/
     IF (QCROSS) THEN
       call setfmt(etrmx(cmap),zero,zero,zero,zero,f13)
       F144='(A4,A8,2X,'//F13//')'
       WRITE(UNIT,F144) ENAME,' CROSS> ',ETRMX(CMAP)
     ENDIF
#endif /* (epmf)*/
#endif 

  IF (QEXTER) THEN
     call setfmt(etrmx(vdw),etrmx(elec),etrmx(hbond), &
          etrmx(asp),etrmx(user),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' EXTERN>',ETRMX(VDW), &
          ETRMX(ELEC),ETRMX(HBOND),ETRMX(ASP),ETRMX(USER)
  ENDIF

  IF (QIMAGE) THEN
     call setfmt(etrmx(imvdw),etrmx(imelec),etrmx(imhbnd), &
          etrmx(rxnfld),etrmx(extnde),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' IMAGES>',ETRMX(IMVDW), &
          ETRMX(IMELEC),ETRMX(IMHBND), &
          ETRMX(RXNFLD),ETRMX(EXTNDE)
  ENDIF

  IF (QEWALD) THEN
     call setfmt(etrmx(ewksum),etrmx(ewself),etrmx(ewexcl), &
          etrmx(ewqcor),etrmx(ewutil),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' EWALD> ',ETRMX(EWKSUM), &
          ETRMX(EWSELF),ETRMX(EWEXCL), &
          ETRMX(EWQCOR),ETRMX(EWUTIL)
  ENDIF

  IF (QCONST) THEN
     call setfmt(etrmx(charm),etrmx(cdihe),etrmx(cintcr), &
          etrmx(resd),etrmx(noe),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' CONSTR>',ETRMX(CHARM), &
          ETRMX(CDIHE),ETRMX(CINTCR),ETRMX(RESD),ETRMX(NOE)
     if(abs(etrmx(pcharm))>accu) then
        call setfmt(zero,zero,zero,zero,etrmx(pcharm),F13)
        F144='(A12,2X,'//F13//')'
        WRITE(UNIT,F144) ' CONSTR/PCA>',ETERM(PCHARM)
     endif
  ENDIF

#if KEY_CONSHELIX==1
  IF (QCONSH) THEN
     call setfmt(etrmx(echdl),etrmx(echan),etrmx(echtn), &
          etrmx(echrn),zero,F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' CONSTR>',ETRMX(ECHDL), &
          ETRMX(ECHAN),ETRMX(ECHTN),ETRMX(ECHRN)
  ENDIF
#endif 
  IF (QRESTR) THEN
     call setfmt(etrmx(cqrt),etrmx(eharm),etrmx(shap), &
          etrmx(dmc),etrmx(rgy),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' RESTR> ', &
#if KEY_CPATH==1
#if KEY_HMCOM==1
     ETRMX(CQRT)+ETRMX(PATH)+ETRMX(HMCM),  & 
#endif
#if KEY_HMCOM==0
          ETRMX(CQRT)+ETRMX(PATH),              & 
#endif
#else /**/
#if KEY_HMCOM==1
     ETRMX(CQRT)+ETRMX(HMCM),  & 
#endif
#if KEY_HMCOM==0
          ETRMX(CQRT),              & 
#endif
#endif 
     ETRMX(EHARM),ETRMX(SHAP),ETRMX(DMC),ETRMX(RGY)
  ENDIF

  IF (QPOLAR) THEN
     call setfmt(etrmx(polar),etrmx(pull),zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' POLAR> ',ETRMX(POLAR),ETRMX(PULL)
  ENDIF

  IF (QMISC)  THEN
#if KEY_RPATH==1
     call setfmt(etrmx(sbndry),etrmx(tsm),etrmx(prms), &
          etrmx(pang),eother,F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' MISCEL>',ETRMX(SBNDRY), &
          ETRMX(TSM),ETRMX(PRMS),ETRMX(PANG),EOTHER
#else /**/
     call setfmt(etrmx(sbndry),etrmx(tsm),zero, &
          zero,eother,F13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' MISCEL>',ETRMX(SBNDRY), &
          ETRMX(TSM),EOTHER
#endif 
  ENDIF

#if KEY_EDS==1
  IF (QEDSENE) THEN
     call setfmt(etrmx(eds),zero,zero,zero,zero,f13)
     f144='(a4,a8,2x,1'//f13//')'
     write(unit,f144) ename,' EDS>',etrmx(eds)
  ENDIF
#endif 

  IF (QQUANT) THEN
     call setfmt(etrmx(qmel),etrmx(qmvdw),zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' QUANTM>',ETRMX(QMEL),ETRMX(QMVDW)
#if KEY_QCHEM==1
     IF (QQEWALD) THEN 
        if (prnlev > 5) then 
          WRITE(UNIT, '(A,F20.7)') 'CHARMM> MM Ewald Energy:',  &
             ETRMX(EWKSUM)+ETRMX(EWSELF)+ETRMX(EWEXCL)+       &
             ETRMX(IMELEC)+ETRMX(ELEC)
        endif 
     ENDIF
#endif 
  ENDIF

  IF (QDCM) THEN
    CALL DCMEPRINT(UNIT)
  ENDIF

  IF (QMMFP)  THEN
     call setfmt(etrmx(geo),etrmx(mdip),etrmx(ssbp), &
          etrmx(shel),eprpx(droffa),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' MMFP>  ',ETRMX(GEO), &
          ETRMX(MDIP), ETRMX(SSBP), ETRMX(SHEL), EPRPX(DROFFA)
  ENDIF

  ! CHR Additional MMFP lines added for new terms (here VMOD)
  IF (QMMFP2)  THEN
     call setfmt(etrmx(vmod),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' MMFP2> ',ETRMX(VMOD)
  ENDIF

#if KEY_PATHINT==1
  IF (QPINT) THEN
     call setfmt(etrmx(pint),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' PINT>',ETRMX(PINT)
  ENDIF
#endif 

#if KEY_CHEQ==1
  IF (QCGP) THEN
     EPRPX(ALLP)=EPRPX(EPOT)
     EPRPX(ALLK)=EPRPX(TOTKE)+EPRPX(CGKE)
     EPRPX(ALLE)=EPRPX(ALLP)+EPRPX(ALLK)
     call setfmt(eprpx(cgke),eprpx(cgpot),eprpx(allk), &
          eprpx(allp),eprpx(alle),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' CHEQ>  ', &
          EPRPX(CGKE), EPRPX(CGPOT), EPRPX(ALLK), EPRPX(ALLP), &
          EPRPX(ALLE)
  ENDIF
#endif 

  IF (QPBEQ) THEN
#if KEY_DHDGB==1
!AP/MF
     IF (QFHDGB) THEN
        call setfmt(etrmx(pbnp),etrmx(pbelec),etrmx(gbenr), &
             etrmx(defe),etrmx(smbp),F13)
        F144='(A4,A8,2X,5'//F13//')'
        WRITE(UNIT,F144) ENAME,' PBEQ>  ', &
             ETRMX(PBNP),ETRMX(PBELEC),ETRMX(GBEnr),ETRMX(DEFE), &
             ETRMX(SMBP)
     ELSE
        call setfmt(etrmx(pbnp),etrmx(pbelec),etrmx(gbenr), &
             etrmx(gsbp),etrmx(smbp),F13)
        F144='(A4,A8,2X,5'//F13//')'
        WRITE(UNIT,F144) ENAME,' PBEQ>  ', &
             ETRMX(PBNP),ETRMX(PBELEC),ETRMX(GBEnr),ETRMX(GSBP), &
             ETRMX(SMBP)
     ENDIF
#else /**/
     call setfmt(etrmx(pbnp),etrmx(pbelec),etrmx(gbenr), &
          etrmx(gsbp),etrmx(smbp),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' PBEQ>  ', &
          ETRMX(PBNP),ETRMX(PBELEC),ETRMX(GBEnr),ETRMX(GSBP), &
          ETRMX(SMBP)
#endif
  ENDIF

!! Yujin 2-->3 extra etrmx 
#if KEY_GRID==1
  IF (QGRD) THEN
     call setfmt(etrmx(grvdw),etrmx(grelec),etrmx(grhbon), zero,zero,F13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' GRID>  ',ETRMX(GrvdW),ETRMX(GrElec),ETRMX(GrHBON)
  ENDIF
#endif 

  IF (QDYNAM &
#if KEY_MC==1
       .OR. QMC &  
#endif
       ) THEN
     call setfmt(eprpx(vire),eprpx(viri),eprpx(presse), &
          eprpx(pressi),eprpx(volume),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' PRESS> ',EPRPX(VIRE), &
          EPRPX(VIRI),EPRPX(PRESSE),EPRPX(PRESSI),EPRPX(VOLUME)
  ENDIF

  IF (QPXTL) THEN
     call setfmt(eprpx(xtlte),eprpx(xtlke),eprpx(xtlpe), &
          eprpx(xtltem),zero,F13)
     F144='(A4,A8,15X,4'//F13//')'
     WRITE(UNIT,F144) ENAME,' XTLE>  ',EPRPX(XTLTE), &
          EPRPX(XTLKE),EPRPX(XTLPE),EPRPX(XTLTEM)
  ENDIF

#if KEY_LRVDW==1
  IF (LLRVDW .OR. LLRVDW_MS) THEN
     call setfmt(etrmx(elrc),pvlrc,zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' LRCor> ',ETRMX(ELRC),PVLRC
  ENDIF
#endif 
#if KEY_LJPME==1
  if (qljpme) then
     call setfmt(etrmx(ljrec),etrmx(ljsel),etrmx(ljexc),zero,zero,f13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' LJPME> ',ETRMX(ljrec), ETRMX(ljsel), ETRMX(ljexc)
  endif
#endif

#if KEY_FOURD==1 /*4dprinte*/
  IF (DIM4.AND.QDYNAM) THEN
     call setfmt(eprpx(epot)-eprpx(epot4),eprpx(tot4), &
          eprpx(totk4),eprpx(epot4),eprpx(tem4),f13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' FOURTH>', &
          EPRPX(EPOT)-EPRPX(EPOT4),EPRPX(TOT4),EPRPX(TOTK4), &
          EPRPX(EPOT4),EPRPX(TEM4)
  ENDIF
  IF (DIM4.AND.QMINMZ) THEN
     call setfmt(etrmx(bk4d),eprpx(epot)-etrmx(bk4d), &
          zero,zero,zero,f13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' FOURTH>',ETRMX(BK4D), &
          EPRPX(EPOT)-ETRMX(BK4D)
  ENDIF
  IF (DIM4.AND.QENER) THEN
     call setfmt(etrmx(bk4d),eprpx(epot)-etrmx(bk4d), &
          zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' FOURTH>',ETRMX(BK4D), &
          EPRPX(EPOT)-ETRMX(BK4D)
  ENDIF
#endif /* (4dprinte)*/

#if KEY_ACE==1
  IF (QACE) THEN
     call setfmt(etrmx(hydr),eprpx(self),eprpx(screen), &
          eprpx(coul),zero,F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' ACE1>',ETRMX(HYDR), &
          EPRPX(SELF),EPRPX(SCREEN),EPRPX(COUL)
     call setfmt(eprpx(solv),eprpx(inter),zero,zero,zero,f13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' ACE2>',EPRPX(SOLV), &
          EPRPX(INTER)
  ENDIF
#endif 

#if KEY_ADUMB==1 || KEY_RXNCOR==1
  IF (QADUMB) THEN
#if KEY_GAMUS==1
     call setfmt(etrmx(adumb),etrmx(umbr),etrmx(gamus),zero,zero,F13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' UMBR>',ETRMX(ADUMB),ETRMX(UMBR),ETRMX(GAMUS)
#else /**/
     call setfmt(etrmx(adumb),etrmx(umbr),zero,zero,zero,F13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' UMBR>',ETRMX(ADUMB),ETRMX(UMBR),zero
#endif 
  ENDIF
#endif 

#if KEY_FACTS==1
  IF (FCTRUN .AND. (QFCTPOL .OR. QFCTNPL)) THEN
     call setfmt(etrmx(ifctpol),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' FCTPOL>',ETRMX(iFCTPOL)
     call setfmt(etrmx(ifctnpl),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' FCTNPL>',ETRMX(iFCTNPL)
  ENDIF
#endif 

#if KEY_SASAE==1
  IF (QSASA .AND. QPSASA) THEN
     call setfmt(etrmx(sastrm),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' SOLVAT>',ETRMX(SASTRM)
  ENDIF
#endif 

#if KEY_LARMORD==1
  IF (qlarmord .AND. qlarmord_on) THEN
     call setfmt(etrmx(cspres),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' CSRES>',ETRMX(cspres)
  ENDIF
#endif 

  IF (QRUSH) THEN
     call setfmt(etrmx(rushRepu),etrmx(rushPhob),etrmx(rushHbnd), &
          etrmx(rushBdon),etrmx(rushBacc),F13)
     F144='(A4,A8,2X,5'//F13//')'
     WRITE(UNIT,F144) ENAME,' RUSH1> ',etrmx(rushRepu), &
          etrmx(rushPhob),etrmx(rushHbnd),etrmx(rushBdon), &
          etrmx(rushBacc)
     call setfmt(etrmx(rushArom),zero,zero,zero,zero,F13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' RUSH2> ',etrmx(rushArom)
  ENDIF

#if KEY_FLUCQ==1
  IF (QFLUCQ) THEN
     call setfmt(etrmx(fqpol),eprpx(fqkin),zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,' FLUCQ> ',ETRMX(FQPOL),EPRPX(FQKIN)
  ENDIF
#endif 

#if KEY_PHMD==1
  IF (QPHMDQ) THEN
     call setfmt(etrmx(phenr),eprpx(phkin),zero,zero,zero,F13)
     F144='(A4,A8,2X,2'//F13//')'
     WRITE(UNIT,F144) ENAME,'PHMD>  ',ETRMX(PHEnr),EPRPX(PHKin)
  ENDIF
#endif 

#if KEY_OVERLAP==1
  IF (QLAPQ) THEN
     call setfmt(etrmx(qovlap),zero,zero,zero,zero,f13)
     F144='(A4,A8,2X,'//F13//')'
     WRITE(UNIT,F144) ENAME,' OLAP>',ETRMX(QOVLAP)
  ENDIF
#endif 

  ! PJ 06/2005
#if KEY_PIPF==1
  IF (QPIPF) THEN
     call setfmt(etrmx(pipf),eprpx(dipk),eprpx(dipt),zero,zero,f13)
     F144='(A4,A8,2X,3'//F13//')'
     WRITE(UNIT,F144) ENAME,' PIPF>  ',ETRMX(PIPF),EPRPX(DIPK), &
          EPRPX(DIPT)
  ENDIF
#endif 
#if KEY_PNM==1
  IF (QPNM) THEN
     call setfmt(etrmx(pnme),zero,zero,zero,zero,f13)
     F144='(A4,A8,2X,1'//F13//')'
     WRITE(UNIT,F144) ENAME,'  PNM>  ',ETRMX(PNME)
  ENDIF
#endif 
#if KEY_EMAP==1
      IF (QEMAP) THEN
         call setfmt(etrmx(eemap),zero,zero,zero,zero,f13)
         F144='(A4,A8,2X,3'//F13//')'
         WRITE(UNIT,F144) ENAME,' EMAP>  ',ETRMX(EEMAP)
      ENDIF
#endif 
#if KEY_SSNMR==1
      IF (QSSNMR) THEN
         call setfmt(etrmx(ecs),zero,zero,zero,zero,F13)
         F144='(A4,A8,2X,5'//F13//')'
         WRITE(UNIT,F144) ENAME,' SSNMR> ',ETRMX(ECS)
      ENDIF
#endif
#if KEY_RDC==1
     IF (QRDC) THEN
         call setfmt(etrmx(erdc),zero,zero,zero,zero,F13)
         F144='(A4,A8,2X,5'//F13//')'
         WRITE(UNIT,F144) ENAME,'   RDC>  ',ETRMX(ERDC)
      ENDIF
#endif

  !
  !
  !=======================================================================
  WRITE(UNIT,'(2A)') ' ----------       ---------    ---------', &
       '    ---------    ---------    ---------'
  !=======================================================================
800 CONTINUE
  RETURN
END SUBROUTINE PRINTE

subroutine setfmt(v1,v2,v3,v4,v5,fmt)
  !-----------------------------------------------------------------------
  !     Adjust format FMT depending on the largest absolute value of Vn
  use chm_kinds
  implicit none

  real(chm_real) v1,v2,v3,v4,v5
  character(len=*) fmt
  !     local
  real(chm_real), parameter :: tenp6=1.0d6,tenp7=1.0d7,tenp8=1.0d8,tenp9=1.0d9
  real(chm_real), parameter :: tenp10=1.0d10,tenp11=1.0d11
  real(chm_real) absval

  absval=abs(v1)
  if (abs(v2) > absval) absval=abs(v2)
  if (abs(v3) > absval) absval=abs(v3)
  if (abs(v4) > absval) absval=abs(v4)
  if (abs(v5) > absval) absval=abs(v5)

  fmt='F13.5'
  if (absval > tenp6) fmt='F13.4'
  if (absval > tenp7) fmt='F13.3'
  if (absval > tenp8) fmt='F13.2'
  if (absval > tenp9) fmt='F13.1'
  if (absval > tenp10) fmt='F13.0'
  if (absval > tenp11) fmt='E13.5'
  return
end subroutine setfmt

subroutine DumpEnFRC(unit)
  !-----------------------------------------------------------------------
  ! Prints energy terms and forces for testing purposes.
  !
  ! unit   The unit for the printing.
  !
  use chm_kinds
  use dimens_fcm
  use deriv
  use energym
  use psf
  use stream

  integer :: i, unit
  real(chm_real), parameter :: ACCU = 4.9D-7

  if (prnlev < 3) return
  if (unit <= 0) return
  write (unit,'("NATOM",i10/"ENERGY"/a6,es25.15)') &
       natom, ceprop(epot), eprop(epot)
  do i = 1, lenent
     if (abs(eterm(i)) > ACCU) then
        write (unit,'(a6,es25.15)') ceterm(i), eterm(i)
     endif
  enddo
  write (unit,'("FORCE")')
  do i = 1, natom
     write (unit,'(i10,3(es25.15))') i, dx(i), dy(i), dz(i)
  enddo
  if (unit /= outu) then
     write (outu, '(" Energies and Forces dumped to unit ", i6)') unit
  endif
  return
end subroutine DumpEnFRC

