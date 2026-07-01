module corman_mod
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  !-----------------------------------------------------------------------
  ! COOR SEARCH data (saved for multiple usage or for SHAPE code)
  !
  !    CSSAVE  -  Logical flag indicating a saved grid
  !    CSXMIN  -  maximum X-value
  !    CSXMAX  -  minimum X-value
  !    CSYMIN  -  maximum Y-value
  !    CSYMAX  -  minimum Y-value
  !    CSZMIN  -  maximum Z-value
  !    CSZMAX  -  minimum Z-value
  !    CSXGRID -  Grid count in X-direction
  !    CSYGRID -  Grid count in Y-direction
  !    CSZGRID -  Grid count in Z-direction
  !    CSjMAT  -  Pointer to saved grid on the heap
  !    CSSPACE -  Number of I*4 elements on the grid
  !    CSNSAVE -  Number of accumulated COOR SEARCH calcs
  !
  INTEGER,save :: CSXGRID,CSYGRID,CSZGRID,CSSPACE,CSNSAVE
  integer,allocatable,dimension(:) :: CSjMAT
  real(chm_real),save :: CSXMIN,CSXMAX,CSYMIN,CSYMAX,CSZMIN,CSZMAX
  LOGICAL,save :: CSSAVE
  !

contains

  subroutine corman_init
  use solana_mod,only:irlp,mrlp
  use corsubs,only:qaxisc,axisr
  use number

    QAXISC=.FALSE.
    CSSAVE=.FALSE.
    CSXMIN=ANUM
    CSXMAX=ANUM
    CSYMIN=ANUM
    CSYMAX=ANUM
    CSZMIN=ANUM
    CSZMAX=ANUM
    CSXGRID=0
    CSYGRID=0
    CSZGRID=0
    !
    QAXISC=.FALSE.
    AXISR=ZERO
    !
    IRLP=1
    MRLP=2
    return
  end subroutine corman_init

  SUBROUTINE CORCOM(COMLYN, COMLEN)
    !
    !     Process the COORDINATE commands.
    !
  use exfunc
    !
  use coord
  use coordc
  use image
  use psf
#if KEY_DIMS==1
  use dims           
#endif
  use stream
  use string
  use memory

    ! . Passed variables.
    CHARACTER COMLYN*(*)
    INTEGER   COMLEN
    ! . Local variables.
    INTEGER   NATIML, NSGIML
#if KEY_DIMS==1
    LOGICAL   LDIMS
#endif 
    LOGICAL   LCOMP
#if KEY_COMP2==1
    LOGICAL   LSECOND 
#endif
    INTEGER I
    !
    NATIML = NATOM
    NSGIML = NSEG
    IF(INDXA(COMLYN, COMLEN, 'IMAG')  >  0) THEN
       NATIML = NATOMT
       NSGIML = NSEGT
    ENDIF

    LCOMP  = INDXA(COMLYN, COMLEN, 'COMP')  >  0
#if KEY_COMP2==1
    LSECOND = INDXA(COMLYN, COMLEN, 'SECO')  >  0  
#endif
#if KEY_DIMS==1
    LDIMS = INDXA(COMLYN, COMLEN, 'DIMS')  >  0
#endif 
    ! ----------------------------------------------------------------
#if KEY_COMP2==1 /*comp2_1*/
    ! if requested, use the second comparison set of coordinates
    IF (LSECOND) THEN
       IF(LCOMP) THEN
#if KEY_DIMS==1
          IF(LDIMS) THEN
             CALL CORMAN('DIMS', DIMTARGXA, DIMTARGYA, DIMTARGZA, WDIMS, &
                  XCOMP, YCOMP, ZCOMP,WCOMP, NATIML, &
                  ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
                  AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, isdrude)
          ELSE
#endif 
             CALL CORMAN('COMPARISON', XCOMP2, YCOMP2, ZCOMP2, WCOMP2, &
                  X, Y, Z, WMAIN, NATIML, &
                  ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
                  AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, &
                  ISDRUDE)
#if KEY_DIMS==1
          ENDIF
       ELSE IF(LDIMS) THEN
          CALL CORMAN('DIMS', DIMTARGXA, DIMTARGYA, DIMTARGZA, WDIMS, &
               X, Y, Z, WMAIN, NATIML, &
               ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
               AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, isdrude)
#endif 
       ELSE
          CALL CORMAN('MAIN', X, Y, Z, WMAIN, XCOMP2, YCOMP2, ZCOMP2, &
               WCOMP2, NATIML, ATYPE, &
               RESID, RES, IBASE, SEGID, NICTOT, NSGIML, AMASS, &
               CG, NTRANS, IMNAME, IMTRNS, XUCELL, &
               ISDRUDE)
       ENDIF
    ELSE 
#endif /* (comp2_1)*/
       IF(LCOMP) THEN
#if KEY_DIMS==1
          IF(LDIMS) THEN
             CALL CORMAN('DIMS', DIMTARGXA, DIMTARGYA, DIMTARGZA, WDIMS, &
                  XCOMP, YCOMP, ZCOMP,WCOMP, NATIML, &
                  ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
                  AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, isdrude)
          ELSE
#endif 
             CALL CORMAN('COMPARISON', XCOMP, YCOMP, ZCOMP, WCOMP, &
                  X, Y, Z, WMAIN, NATIML, &
                  ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
                  AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, &
                  ISDRUDE)
#if KEY_DIMS==1
          ENDIF
       ELSE IF(LDIMS) THEN
          CALL CORMAN('DIMS', DIMTARGXA, DIMTARGYA, DIMTARGZA, WDIMS, &
               X, Y, Z, WMAIN, NATIML, &
               ATYPE, RESID, RES, IBASE, SEGID, NICTOT, NSGIML, &
               AMASS, CG, NTRANS, IMNAME, IMTRNS, XUCELL, isdrude)
#endif 
       ELSE
          CALL CORMAN('MAIN', X, Y, Z, WMAIN, XCOMP, YCOMP, ZCOMP, &
               WCOMP, NATIML, ATYPE, &
               RESID, RES, IBASE, SEGID, NICTOT, NSGIML, AMASS, &
               CG, NTRANS, IMNAME, IMTRNS, XUCELL, &
               ISDRUDE)
       ENDIF
#if KEY_COMP2==1
    ENDIF     
#endif
    !
    RETURN
  END SUBROUTINE CORCOM

  SUBROUTINE CORMAN(CNAME,X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,WCOMP, &
       NATOM,ATYPE,RESID,RES,IBASE,SEGID, &
       NICTOT,NSEG,AMASS,CG,NTRANS,IMNAME,IMTRNS,XUCELL, &
       ISDRUDE)
    !
    !     THIS ROUTINE HANDLES ALL OF THE COORDINATE MANIPULATION OPTIONS
    !
    !     By Bernard R. Brooks  (developed from 1981 to 1983)
    !     Victor Anisimov, 2004: entropy modifications
    !
    use vector
    use number
    use chutil
    use consta
    use comand
    use corman2
    use corman3,only:corhist,squareup
    use corsubs
    use cvio,only:trjspc
    use dynanal,only:avecor,paxanl
    use helix
    use pucker_mod,only:pucker
    use rgyr_mod,only:lsqp,rgyr,coriner
    use ctitla
#if KEY_NOGRAPHICS==0
    use graph                                               
#endif
    use hbanal_mod,only:hbanal
    use deriv
    use modpsf
    use scalar_module
    use select
    use shake
    use solvmap ! hwm
    use storage
    use stream
    use string
    use parallel
    use repdstr
    use secondary_structure,only: secstr
    use memory
    use solana_mod,only: solana
    use tmscore_m
    !
#if KEY_STRINGM==1
    ! Chirality checks
    use chirality
    ! Conformational consistency
    use confcons
#endif
    ! Entropy
    use entropy
    use holonom,only:holonoma
    use coorio_mod,only:coorio
    use surfacmod,only:surfac
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec 
#endif
#if KEY_DOMDEC==1
    use domdec_d2d_comm,only:copy_to_all 
#endif
    use param_store, only: set_param

    implicit none

    real(chm_real),allocatable,dimension(:,:) :: RPAX

    CHARACTER(len=*) CNAME
    real(chm_real) X(*),Y(*),Z(*),WMAIN(*), &
         XCOMP(*),YCOMP(*),ZCOMP(*),WCOMP(*)
    INTEGER NATOM

    real(chm_real),allocatable,dimension(:) :: RWORK
    real(chm_real),allocatable,dimension(:),target :: space_rl
    real(chm_real),pointer,dimension(:) :: xmat=>null(),ymat=>null(),zmat=>null(), &
         scr2=>null(),jmat2=>null()

    integer,allocatable,dimension(:),target :: space_int
    integer,pointer,dimension(:) :: islct=>null(),jslct=>null()

    CHARACTER(len=*) ATYPE(*),RESID(*),RES(*),SEGID(*)
    INTEGER IBASE(*)
    INTEGER NICTOT(*),NSEG
    real(chm_real)  AMASS(*),CG(*)
    INTEGER NTRANS
    CHARACTER(len=*) IMNAME(*)
    real(chm_real)  IMTRNS(12,*), XUCELL(6)
    LOGICAL ISDRUDE(*)
    !
    LOGICAL LENTRO
    !
    real(chm_real)  U(9),RN(3),PHI,RATE,AX(3),R0(3)
    LOGICAL LOK,EOF
    !br...stuff for reorienting structures in DYNA, 03-Feb-96
    LOGICAL ORIENT, LWMAIN
    INTEGER BMASS, ATOMPR
    !
    real(chm_real) ACONV,BCONV,CCONV,ALCONV,BECONV,GACONV
    !
    CHARACTER(len=4) WRD, WRD2
    CHARACTER(len=8) WRD8
    INTEGER ICNTRL(20)
    !
    real(chm_real) A,FACT,AMAX,AMASST,AMASSV,RMST,RMSV
    real(chm_real) XMIN,YMIN,ZMIN,WMIN,XMAX,YMAX,ZMAX,WMAX
    real(chm_real) XAVE,YAVE,ZAVE,WAVE
    real(chm_real) RCUT,RBUFF
    real(chm_real) XDIR,YDIR,ZDIR,XN,YN,ZN,XCEN,YCEN,ZCEN,XV,YV,ZV
    real(chm_real) PHASE,AMPLIT,ACCURACY,RPRO
    real(chm_real) QTOT,XDIP,YDIP,ZDIP,MDIP
    real(chm_real) RIJ,RCIJ,XIJ,YIJ,ZIJ,XCIJ,YCIJ,ZCIJ,DRMS
    INTEGER NRES,ITRANS,IUNIT,IMODE,NMISS,NSLCT,NSLCT2,NBPAIR
    INTEGER IXSUM,IYSUM,IZSUM,NSEL
    INTEGER IXREF,IYREF,IZREF,IXFLUC,IYFLUC,IZFLUC
    INTEGER ISPACE,NXREP
    INTEGER NUNIT,NBEGN,NSTOP,NSKIP
    INTEGER IPT,JPT,I,J,K,PSOPER
    integer iutil0
    integer,allocatable,dimension(:) :: jmat,iutil
    integer,allocatable,dimension(:,:) :: iscr2dim
    real(chm_real),allocatable,dimension(:) :: scr
    integer :: siz_scr
    INTEGER SET1,SET2,SET3,SET4,ID1,ID2,NDIG,AFLAG,DFLAG,HEAVY
    LOGICAL LRMS,LMASS,LNORO,LWEIG,LSEL2,LPRINT,LVACU,LHOLES,LSAVE
    LOGICAL LVERB,LAXIS,LVOLUM,QPAX,LPRNT,QOK,QCSND,QCRCV,QCIGN
    character(len=8) SID,RID1,RID2,PDEF

#if KEY_CONSHELIX==1 /*conshelix*/
    LOGICAL OFIRS,OLAST,TFIRS,TLAST
    LOGICAL LHMODC,LDNAM,LBHPM,LBHGM
#endif /* (conshelix)*/

    integer, parameter :: n_stats = 13
    character(len=4) :: stat_names(n_stats)
    real(chm_real) :: stat_vals(n_stats)
    
    LOGICAL OXYZ,QERR

    SAVE

    INTEGER ounit,ounit1,oslv,oslt,outmode ! hwm, for SMAP
    real(chm_real) rdum,dcut,hcut          ! hwm, for SMAP
    character(len=8) cdum                  ! hwm, for SMAP

    call chmalloc('corman.src','CORCOM','RWORK',NATom,crl=RWORK)
    call chmalloc('corman.src','corman','space_int',2*natom,intg=space_int)
    islct => space_int(1:natom)
    jslct => space_int(natom+1:2*natom)

    EOF=.FALSE.
    NRES=NICTOT(NSEG+1)
    IMODE=0
    NMISS=0
    LPRNT=(PRNLEV >= 3)
    LWEIG=(INDXA(COMLYN,COMLEN,'WEIG') > 0)
    WRD=NEXTA4(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !
    ! PROCESS COMMANDS THAT DO NOT REQUIRE AN ATOM SELECTION
    !
    !=======================================================================
    IF(WRD == 'READ') THEN
       IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       CALL COORIO(-1,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
            ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
            RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'WRIT') THEN
       IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
       CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
       CALL COORIO(0,IUNIT,COMLYN,COMLEN,TITLEA,NTITLA, &
            ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
            RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'PRIN') THEN
       IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       CALL COORIO(1,IUNIT,COMLYN,COMLEN,TITLEA,NTITLA, &
            ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
            RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'COVA') THEN
       ! process-covariance-command
       ! Allocate stack space for arrays

       CALL COVARI()

       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'DMAT') THEN
       ! process-distance_matrix-command
       CALL DISTMAT()
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'ANAL') THEN
       ! process-solvent-analysis
       ! Allocate space for solvent site selection
#if KEY_NOMISC==0
       CALL SOLANA(LWEIG)
#else /**/
       CALL WRNDIE(-1,'<CORMAN>','SOLANA code is not compiled')
#endif 
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'PUCK') THEN
       !
       ! Computes pucker of (deoxy)ribose rings
       !
       RID1=GTRMA(COMLYN,COMLEN,'RESI')
       IF(RID1  ==  ' ') THEN
          CALL WRNDIE(0,'<CORMAN>','NO RESIDUE(S) SELECTED FOR PUCKER')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       RID2=GTRMA(COMLYN,COMLEN,'TO')
       SID= GTRMA(COMLYN,COMLEN,'SEGI')
       ! Check what algorithm user wants (there should be nothing else to parse)
       ! Default to Altona&Sundarlingam
       PDEF='AS'
       CALL TRIMA(COMLYN,COMLEN)
       IF(COMLEN  >  0) PDEF=NEXTA4(COMLYN,COMLEN)
       IF(SID  ==  ' ') SID=SEGID(1)
       IF(RID2  ==  ' ') THEN
          CALL PUCKER(SID,RID1,PHASE,AMPLIT,PDEF,X,Y,Z)
          IF(PRNLEV >= 3) WRITE(OUTU,490) PDEF, &
               SID(1:idleng),RID1(1:idleng),PHASE,AMPLIT
490       FORMAT(/'   ',A,' Pucker parameters for ',A,2X,A, &
               ': Phase=',F6.2,' amplitude=',F7.4)
          ! Set command line substitution variables, adm jr.
          call set_param('PHASE',PHASE)
          call set_param('AMP',AMPLIT)
       ELSE
          ! So now we ASSUME that RID1&RID2 are numerical....
          ID1=GETRSN(SID,RID1,'    ', &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
          ID2=GETRSN(SID,RID2,'    ', &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
          IF(PRNLEV >= 3) WRITE(OUTU, 492) PDEF
492       FORMAT(/'                ',A,' PUCKER PARAMETERS', &
               /'     SEGID RESID    PHASE     AMPLITUDE')
          DO I=ID1,ID2
             CALL PUCKER(SID,RESID(I),PHASE,AMPLIT,PDEF,X,Y,Z)
             IF(PRNLEV >= 3) WRITE(OUTU,494) &
                  SID(1:idleng),RESID(I)(1:idleng),PHASE,AMPLIT
494          FORMAT(5X,A,2X,A,F10.2,5X,F7.4)
             ! Set command line substitution variables, adm jr.
             ! Note that values are set to those for the final sugar analyzed
             call set_param('PHASE',PHASE)
             call set_param('AMP',AMPLIT)
          ENDDO
       ENDIF
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    IF(WRD == 'SECS') THEN
       ! to process SECondaryStructure analysis
       !
       ! Can use two selections, but does not require the second one, so SELCTD
       ! is preferred to the selections done below.
       CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,QERR)
       IF(QERR) CALL WRNDIE(-2,'<CORMAN>','ATOM SELECTION ERROR') 
       CALL SECSTR(WMAIN,ISLCT,JSLCT,1)
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    !
    ! PROCESS COMMANDS THAT REQUIRE AN ATOM SELECTION
    !
    !=======================================================================
    ! to select atoms.
    IMODE=0
    CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE /= 0) THEN
       CALL WRNDIE(0,'<CORMAN>','ATOM SELECTION ERROR')
       call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
       nullify(islct,jslct)
       call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
       RETURN
    ENDIF
    !
    NSLCT=NSELCT(NATOM,ISLCT)
    IF(NSLCT == 0) THEN
       CALL WRNDIE(0,'<CORMAN>','ZERO ATOMS SELECTED')
    ENDIF
    !
    !=======================================================================
    ! to select-second-atoms
    IF(WRD == 'AXIS' .OR. WRD == 'DIST' .OR. WRD == 'MIND' .OR. &
      WRD == 'HBON' .OR. WRD == 'CONT' .OR. WRD == 'DUPL' .OR. WRD == 'DRMS' .OR. &
      WRD == 'HELI' .OR. WRD == 'DYNA' .OR. WRD == 'MAXD') THEN
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,JSLCT,NATOM,0,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE /= 0) THEN
          CALL WRNDIE(0,'<CORMAN>','ATOM SELECTION ERROR FOR SECOND')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       !
       NSLCT2=NSELCT(NATOM,JSLCT)
       LSEL2=(NSLCT2 > 0)
    ENDIF
    !=======================================================================
    ! to get-distance-spec
    IF(WRD == 'SET'  .OR. WRD == 'ROTA' .OR. WRD == 'TRAN' .OR. &
         WRD == 'DIST' .OR. WRD == 'TWIS') THEN
       LAXIS=(INDXA(COMLYN,COMLEN,'AXIS') > 0)
       IF(LAXIS) THEN
          IF(QAXISC) THEN
             RN(1)=AXISX
             RN(2)=AXISY
             RN(3)=AXISZ
             XCEN=AXISCX
             YCEN=AXISCY
             ZCEN=AXISCZ
          ELSE
             CALL WRNDIE(-1,'<CORMAN>', &
                  'AXIS requested, but no axis yet defined. - Ignored')
             call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
             nullify(islct,jslct)
             call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
             RETURN
          ENDIF
       ELSE
          FACT=GTRMF(COMLYN,COMLEN,'XDIR',ZERO)
          RN(1)=FACT
          FACT=GTRMF(COMLYN,COMLEN,'YDIR',ZERO)
          RN(2)=FACT
          FACT=GTRMF(COMLYN,COMLEN,'ZDIR',ZERO)
          RN(3)=FACT
          XCEN=GTRMF(COMLYN,COMLEN,'XCEN',ZERO)
          YCEN=GTRMF(COMLYN,COMLEN,'YCEN',ZERO)
          ZCEN=GTRMF(COMLYN,COMLEN,'ZCEN',ZERO)
       ENDIF
       !
       PHI = DOTVEC(RN,RN,3)
       FACT=GTRMF(COMLYN,COMLEN,'DIST',ANUM)
       IF(FACT /= ANUM) THEN
          IF(PHI < 0.00001) THEN
             CALL WRNDIE(0,'<CORMAN>','NO VECTOR SPECIFIED')
             call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
             nullify(islct,jslct)
             call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
             RETURN
          ENDIF
          CALL NORMALL(RN,3)
          DO I=1,3
             RN(I)=RN(I)*FACT
          ENDDO
       ENDIF
       !
       FACT=GTRMF(COMLYN,COMLEN,'FACT',ONE)
       XDIR=RN(1)*FACT
       YDIR=RN(2)*FACT
       ZDIR=RN(3)*FACT
    ENDIF
    !-----------------------------------------------------------------------
    !=======================================================================
    !-----------------------------------------------------------------------
    IF(WRD == 'SCAL') THEN
       ! to process-scale-command
       ! SCALES THE COORDINATE SET SPECIFIED
       !
       FACT=GTRMF(COMLYN,COMLEN,'FACT',ONE)
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=WMAIN(I)*FACT
             ELSE
                IF(X(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   X(I)=X(I)*FACT
                   Y(I)=Y(I)*FACT
                   Z(I)=Z(I)*FACT
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,38) CNAME
38     FORMAT(' SELECTED COORDINATES SCALED IN THE ',A,' SET.'/)
       !
#if KEY_STRINGM==1
       !------------------------------------ VO : conformational consistency check/fix
    else if(wrd  ==  'CONF') then
       call confcons_main(COMLYN,COMLEN,ISLCT)
       !------------------------------------ VO : chirality check/fix
    else if(wrd  ==  'CHIR') then
       call chirality_main(COMLYN,COMLEN,ISLCT)
       !-----------------------------------------------------------------------
#endif
    ELSE IF(WRD == 'ADD ') THEN
       ! to process-add-command
       !     ADDS THE COMPARISON COORDINATES WITH THE MAIN COORDINATES
       !
       DO I=1,NATOM
          IF(ISLCT(I) /= 1) THEN
             CONTINUE
          ELSE IF(LWEIG) THEN
             WMAIN(I)=WMAIN(I)+WCOMP(I)
          ELSE IF(XCOMP(I) == ANUM .OR. X(I) == ANUM) THEN
             NMISS=NMISS+1
             X(I)=ANUM
             Y(I)=ANUM
             Z(I)=ANUM
          ELSE
             X(I)=X(I)+XCOMP(I)
             Y(I)=Y(I)+YCOMP(I)
             Z(I)=Z(I)+ZCOMP(I)
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,34) CNAME
34     FORMAT(' SELECTED COORDINATES ADDED TO THE ',A,' SET.'/)
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'MASS') THEN
       ! to process-mass-command
       !     MASS WEIGHT THE COORDINATE SET SPECIFIED
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=WMAIN(I)*AMASS(I)
             ELSE
                IF(X(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   X(I)=X(I)*AMASS(I)
                   Y(I)=Y(I)*AMASS(I)
                   Z(I)=Z(I)*AMASS(I)
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,28) CNAME
28     FORMAT(' SELECTED COORDINATES MASS WEIGHTED IN THE ',A,' SET.'/)
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SET ') THEN
       ! to process-set-command
       !     SET THE COORDINATE SET SPECIFIED
       ! get-distance-spec
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=FACT
             ELSE
                X(I)=XDIR
                Y(I)=YDIR
                Z(I)=ZDIR
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,48) CNAME
48     FORMAT(' SELECTED COORDINATES SET IN THE ',A,' SET.'/)
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SDRU') THEN
       ! to process-drude-command
       !     SET THE DRUDE COORDINATES
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(ISDRUDE(I))then
                X(I)=X(I-1)
                Y(I)=Y(I-1)
                Z(I)=Z(I-1)
             ENDIF
          ENDIF
       ENDDO
#if KEY_PARALLEL==1
       CALL PSND8(X,NATOM)
       CALL PSND8(Y,NATOM)
       CALL PSND8(Z,NATOM)
#endif 
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SWAP') THEN
       ! to process-swap-command
       !     SWAPS THE COMPARISON COORDINATES WITH THE MAIN COORDINATES
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                A=WMAIN(I)
                WMAIN(I)=WCOMP(I)
                WCOMP(I)=A
             ELSE
                IF(X(I) == ANUM) NMISS=NMISS+1
                IF(XCOMP(I) == ANUM) NMISS=NMISS+1
                A=Y(I)
                Y(I)=YCOMP(I)
                YCOMP(I)=A
                A=Z(I)
                Z(I)=ZCOMP(I)
                ZCOMP(I)=A
                A=X(I)
                X(I)=XCOMP(I)
                XCOMP(I)=A
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,36)
36     FORMAT(' SELECTED COORDINATES SWAPPED.'/)
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'INIT') THEN
       ! to process-init-command
       !     INITIALIZES THE COORDINATE SET SPECIFIED
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=0.0
             ELSE
                X(I)=ANUM
                Y(I)=ANUM
                Z(I)=ANUM
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,42) CNAME
42     FORMAT(' SELECTED COORDINATES INITIALIZED IN THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'COPY') THEN
       ! to process-copy-command
       !     MODIFIES THE SPECIFIED COORDINATE SET
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=WCOMP(I)
             ELSE
                IF(XCOMP(I) == ANUM) NMISS=NMISS+1
                X(I)=XCOMP(I)
                Y(I)=YCOMP(I)
                Z(I)=ZCOMP(I)
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,45) CNAME
45     FORMAT(' SELECTED COORDINATES COPIED TO THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'ROTA') THEN
       ! to process-rota-command
       !
       ! get-distance-spec
       !
       !+ln - allow explicit input of rotation matrix
       IF(INDXA(COMLYN,COMLEN,'MATR')  >  0)THEN
          CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
               .TRUE.,'CORMAN> ')
          U(1)=NEXTF(COMLYN,COMLEN)
          U(2)=NEXTF(COMLYN,COMLEN)
          U(3)=NEXTF(COMLYN,COMLEN)
          CALL XTRANE(COMLYN,COMLEN,'CORMAN')
          CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
               .TRUE.,'CORMAN> ')
          U(4)=NEXTF(COMLYN,COMLEN)
          U(5)=NEXTF(COMLYN,COMLEN)
          U(6)=NEXTF(COMLYN,COMLEN)
          CALL XTRANE(COMLYN,COMLEN,'CORMAN')
          CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
               .TRUE.,'CORMAN> ')
          U(7)=NEXTF(COMLYN,COMLEN)
          U(8)=NEXTF(COMLYN,COMLEN)
          U(9)=NEXTF(COMLYN,COMLEN)
          CALL XTRANE(COMLYN,COMLEN,'CORMAN')
          !-ln
       ELSE
          PHI=GTRMF(COMLYN,COMLEN,'PHI',ZERO)
          CALL FNDU(U,RN,PHI,LOK)
          IF(.NOT.LOK) THEN
             CALL WRNDIE(0,'<CORMAN>','PARSING ERROR')
             call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
             nullify(islct,jslct)
             call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
             RETURN
          ENDIF
       ENDIF
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                XV=X(I)-XCEN
                YV=Y(I)-YCEN
                ZV=Z(I)-ZCEN
                XN=U(1)*XV+U(2)*YV+U(3)*ZV
                YN=U(4)*XV+U(5)*YV+U(6)*ZV
                ZN=U(7)*XV+U(8)*YV+U(9)*ZV
                X(I)=XN+XCEN
                Y(I)=YN+YCEN
                Z(I)=ZN+ZCEN
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,61) U
61     FORMAT(' ROTATION MATRIX'/,3(3F12.6/))
       CALL FNDROT(U,RN,PHI,LPRNT)
       IF(PRNLEV >= 3) WRITE(OUTU,62) CNAME
62     FORMAT(' SELECTED COORDINATES ROTATED IN THE ',A,' SET.'/)
       !
#if KEY_PARALLEL==1
       CALL PSND8(X,NATOM)
       CALL PSND8(Y,NATOM)
       CALL PSND8(Z,NATOM)
#endif 
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'TWIS') THEN
       ! to process-twist-command
       !
       ! get-distance-spec
       !
       !+ln - allow explicit input of rotation matrix
       RATE=GTRMF(COMLYN,COMLEN,'RATE',ZERO)
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                XV=X(I)-XCEN
                YV=Y(I)-YCEN
                ZV=Z(I)-ZCEN
                PHI=RATE*(XV*RN(1)+YV*RN(2)+ZV*RN(3))
                CALL FNDU(U,RN,PHI,LOK)
                XN=U(1)*XV+U(2)*YV+U(3)*ZV
                YN=U(4)*XV+U(5)*YV+U(6)*ZV
                ZN=U(7)*XV+U(8)*YV+U(9)*ZV
                X(I)=XN+XCEN
                Y(I)=YN+YCEN
                Z(I)=ZN+ZCEN
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,63) CNAME
63     FORMAT(' SELECTED COORDINATES TWISTED IN THE ',A,' SET.'/)
       !
#if KEY_PARALLEL==1
       CALL PSND8(X,NATOM)
       CALL PSND8(Y,NATOM)
       CALL PSND8(Z,NATOM)
#endif 
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'AXIS') THEN
       ! to process-axis-command
       !
       !      generates an axis
       ! select-second-atoms
       !
       ! save previous axis
       WMIN=ZERO
       IF(QAXISC) THEN
          WMIN=AXISR
          XMIN=AXISX/AXISR
          YMIN=AXISY/AXISR
          ZMIN=AXISZ/AXISR
       ENDIF
       !
       QAXISC=.TRUE.
       !
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       !
       AMASST=0.0
       XN=0.0
       YN=0.0
       ZN=0.0
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                IF(LMASS) THEN
                   AMASSV=AMASS(I)
                ELSE
                   AMASSV=1.0
                ENDIF
                AMASST=AMASST+AMASSV
                XN=XN+X(I)*AMASSV
                YN=YN+Y(I)*AMASSV
                ZN=ZN+Z(I)*AMASSV
             ENDIF
          ENDIF
       ENDDO
       !
       IF(AMASST /= ZERO) THEN
          AXISX=XN/AMASST
          AXISY=YN/AMASST
          AXISZ=ZN/AMASST
       ELSE
          AXISX=0.0
          AXISY=0.0
          AXISZ=0.0
          QAXISC=.FALSE.
          IF(PRNLEV >= 3) WRITE(OUTU,75) PHI
75        FORMAT('  Bad selection of atoms (none or zero weight).')
       ENDIF
       AXISCX=AXISX
       AXISCY=AXISY
       AXISCZ=AXISZ
       !
       IF(LSEL2) THEN
          AMASST=0.0
          XN=0.0
          YN=0.0
          ZN=0.0
          DO I=1,NATOM
             IF(JSLCT(I) == 1) THEN
                IF(X(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   IF(LMASS) THEN
                      AMASSV=AMASS(I)
                   ELSE
                      AMASSV=1.0
                   ENDIF
                   AMASST=AMASST+AMASSV
                   XN=XN+X(I)*AMASSV
                   YN=YN+Y(I)*AMASSV
                   ZN=ZN+Z(I)*AMASSV
                ENDIF
             ENDIF
          ENDDO
          !
          IF(AMASST /= ZERO) THEN
             AXISCX=(XN/AMASST+AXISX)*0.5
             AXISCY=(YN/AMASST+AXISY)*0.5
             AXISCZ=(ZN/AMASST+AXISZ)*0.5
             AXISX=XN/AMASST-AXISX
             AXISY=YN/AMASST-AXISY
             AXISZ=ZN/AMASST-AXISZ
          ELSE
             AXISX=-AXISX
             AXISY=-AXISY
             AXISZ=-AXISZ
             IF(PRNLEV >= 3) WRITE(OUTU,75) PHI
          ENDIF
          !
       ENDIF
       !
       AXISR=SQRT(AXISX*AXISX+AXISY*AXISY+AXISZ*AXISZ)
       IF(PRNLEV >= 3) WRITE(OUTU,73) CNAME,AXISX,AXISY,AXISZ,AXISR
73     FORMAT(' AXIS DEFINED FROM THE ',A,' COORDINATES.',/ &
            '  XAXIs=',F12.6,'  YAXIs=',F12.6, &
            '  ZAXIs=',F12.6,'  RAXIs=',F12.6)
       IF(AXISR < TENM5) THEN
          QAXISC=.FALSE.
          IF(PRNLEV >= 3) WRITE(OUTU,76)
76        FORMAT('  Norm of specified axis is too small.')
       ENDIF
       !
       IF(WMIN > ZERO .AND. AXISR > ZERO) THEN
          PHI=(XMIN*AXISX+YMIN*AXISY+ZMIN*AXISZ)/AXISR
          IF(PHI > ONE) PHI=ONE
          IF(PHI < MINONE) PHI=MINONE
          PHI=ACOS(PHI)*RADDEG
          IF(PRNLEV >= 3) WRITE(OUTU,74) PHI
74        FORMAT('  Angle with previous axis:',F12.6/)
       ENDIF
       !
       call set_param('XAXI',AXISX)
       call set_param('YAXI',AXISY)
       call set_param('ZAXI',AXISZ)
       call set_param('RAXI',AXISR)
       call set_param('XCEN',AXISCX)
       call set_param('YCEN',AXISCY)
       call set_param('ZCEN',AXISCZ)
       !
       IF(.NOT.QAXISC) THEN
          IF(PRNLEV >= 3) WRITE(OUTU,77) PHI
77        FORMAT('  No axis generated.')
       ENDIF
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'ORIE') THEN
       ! to process-orie-command
       IF(INDXA(COMLYN,COMLEN,'VIEW') > 0) THEN
#if KEY_NOGRAPHICS==1
          CALL WRNDIE(-1,'<CORMAN>','Graphics code not compiled')
#else /**/
          ! Orient selected atoms to the view matrix.
          IF(GRSCAL == 0.0) THEN
             CALL WRNDIE(0,'<CORMAN>','PARSING ERROR')
             call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
             nullify(islct,jslct)
             call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
             RETURN
          ENDIF
          !
          DO I=1,NATOM
             IF(ISLCT(I) == 1) THEN
                IF(X(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   XV=X(I)
                   YV=Y(I)
                   ZV=Z(I)
                   XN=ULAB(1,1)*XV+ULAB(1,2)*YV+ULAB(1,3)*ZV+ULAB(1,4)
                   YN=ULAB(2,1)*XV+ULAB(2,2)*YV+ULAB(2,3)*ZV+ULAB(2,4)
                   ZN=ULAB(3,1)*XV+ULAB(3,2)*YV+ULAB(3,3)*ZV+ULAB(3,4)
                   X(I)=XN/GRSCAL
                   Y(I)=YN/GRSCAL
                   Z(I)=ZN/GRSCAL
                ENDIF
             ENDIF
          ENDDO
          !
          ! Reset the view matrix (in case called from graphics)
          DO I=1,3
             DO J=1,3
                U(3*I+J-3)=ULAB(J,I)
             ENDDO
             DO J=1,4
                ULAB(J,I)=ZERO
             ENDDO
             ULAB(I,I)=ONE
          ENDDO
          DO J=1,3
             ULAB(J,4)=ZERO
          ENDDO
          ULAB(4,4)=ONE
          GRSCAL=ONE
          !
          IF(PRNLEV >= 3) WRITE(OUTU,61) U
          CALL FNDROT(U,RN,PHI,LPRNT)
          IF(PRNLEV >= 3) WRITE(OUTU,62) CNAME
#endif 
       ELSE

          LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
          LRMS=(INDXA(COMLYN,COMLEN,'RMS') > 0)
          LNORO=(INDXA(COMLYN,COMLEN,'NORO') > 0)
          !
          DO I=1,NATOM
             IF(ISLCT(I) == 1) THEN
                IF(X(I) == ANUM) THEN
                   ISLCT(I)=0
                   NMISS=NMISS+1
                ELSE
                   IF(LRMS.AND.XCOMP(I) == ANUM) THEN
                      ISLCT(I)=0
                      NMISS=NMISS+1
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
          !
          NSLCT=NSLCT-NMISS
          IF(NSLCT <= 0) THEN
             IF(PRNLEV >= 3) WRITE(OUTU,51)
51           FORMAT(' **WARNING** ALL SELECTED COORDINATES UNDEFINED')
          ELSE
             call chmalloc('corman.src','CORMAN','ISCR2dim',2,NATOM,intg=ISCR2dim)
             CALL ORINTC(NATOM,X,Y,Z,XCOMP,YCOMP,ZCOMP,AMASS,LMASS, &
                  LRMS,ISCR2dim,ISLCT,LWEIG,WMAIN, &
                  LNORO,.TRUE.)
             call chmdealloc('corman.src','CORMAN','ISCR2dim',2,NATOM,intg=ISCR2dim)
             !
             IF(PRNLEV >= 3) THEN
                IF(LNORO) THEN
                   WRITE(OUTU,72) CNAME
                ELSE
                   WRITE(OUTU,52) CNAME
                ENDIF
52              FORMAT(' ALL COORDINATES ORIENTED IN THE ',A, &
                     ' SET BASED ON SELECTED ATOMS.'/)
             ENDIF
          ENDIF
       ENDIF
#if KEY_PARALLEL==1
       CALL PSND8(X,NATOM)
       CALL PSND8(Y,NATOM)
       CALL PSND8(Z,NATOM)
#endif 
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'INER') THEN
       !
       ! Entropy keywords
       LENTRO=(INDXA(COMLYN,COMLEN,'ENTR') > 0)
       IF(LENTRO) THEN
          !          Termerature
          TK=GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
          !          Rotational symmetry number
          ! Explanation of sigma: C.J.Cramer, "Essentials of Comp.Chem.",2002,p.327
          SIGMA=GTRMF(COMLYN,COMLEN,'SIGM',ONE)
          ! Standard state: Tidor and Karplus, J Mol Biol (1994) vol. 238 (3) pp. 405-14
          ! Default is solution state of conentration 1M
          SSTAN=GTRMA(COMLYN,COMLEN,'STAN')
          !          Next is for a test purpose only (for debugging)
          UTEST=GTRMI(COMLYN,COMLEN,'TEST',0)
          IF(UTEST > 0) THEN
             !             Replace CHARMM coordinates by the ones read from external file
             DO I=1,NATOM
                READ(UTEST,*) X(I),Y(I),Z(I)
             ENDDO
             !             This file should be read no more than once
             UTEST=0
          ENDIF
       ENDIF
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                ISLCT(I)=0
                NMISS=NMISS+1
             ENDIF
          ENDIF
       ENDDO
       !
       NSLCT=NSLCT-NMISS
       IF(NSLCT <= 0) THEN
          IF(PRNLEV >= 3) WRITE(OUTU,51)
       ELSE
          CALL CORINER(NATOM,X,Y,Z,AMASS,ISLCT,LENTRO)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'AVER') THEN
       ! to process-aver-command
       FACT=GTRMF(COMLYN,COMLEN,'FACT',HALF)
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(LWEIG) THEN
                WMAIN(I)=WMAIN(I)+FACT*(WCOMP(I)-WMAIN(I))
             ELSE
                IF(XCOMP(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   IF(X(I) == ANUM) THEN
                      NMISS=NMISS+1
                      X(I)=XCOMP(I)
                      Y(I)=YCOMP(I)
                      Z(I)=ZCOMP(I)
                   ELSE
                      X(I)=X(I)+FACT*(XCOMP(I)-X(I))
                      Y(I)=Y(I)+FACT*(YCOMP(I)-Y(I))
                      Z(I)=Z(I)+FACT*(ZCOMP(I)-Z(I))
                   ENDIF
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !
       PHI=1.0-FACT
       IF(PRNLEV >= 3) WRITE(OUTU,81) PHI,FACT
81     FORMAT(' WEIGHTINGS (MODIFIED,ALTERNATE)',2F12.6)
       IF(PRNLEV >= 3) WRITE(OUTU,89) CNAME
89     FORMAT(' SELECTED COORDINATES WEIGHTED AVERAGED TO THE ',A, &
            ' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'TRAN') THEN
       ! to process-tran-command
       !
       ! get-distance-spec
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                X(I)=X(I)+XDIR
                Y(I)=Y(I)+YDIR
                Z(I)=Z(I)+ZDIR
             ENDIF
          ENDIF
       ENDDO
       !
       IF(PRNLEV >= 3) WRITE(OUTU,71) RN
71     FORMAT(' TRANSLATION VECTOR ',3F12.6)
       IF(PRNLEV >= 3) WRITE(OUTU,72) CNAME
72     FORMAT(' SELECTED COORDINATES TRANSLATED IN THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DIPO') THEN
       ! to process-dipole-command
       !
       ! convert ISLCT to index array
       NSEL=0
       DO I=1,NATOM
          IF(ISLCT(I) == 1)THEN
             IF(X(I) == ANUM)THEN
                NMISS=NMISS+1
             ELSE
                NSEL=NSEL+1
                ISLCT(NSEL)=I
             ENDIF
          ENDIF
       ENDDO
       !av
       !     OXYZ keyword forces dipole calculation to be performed in the main
       !     coordinate frame
       !       
       IF(INDXA(COMLYN,COMLEN,'OXYZ') > 0) THEN
          ! do not move dipole origin
          OXYZ=.FALSE.
       ELSE
          ! move dipole origin for ions
          OXYZ=.TRUE.
       ENDIF
       !tr--> 04142004
       !     use LMASS to indicate that the center of mass should be used
       !     as reference for carged units (ions) rather than the center of
       !     geometry
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       CALL CDIPOLE(NATOM,X,Y,Z,CG,NSEL,ISLCT,XDIP,YDIP,ZDIP,QTOT, &
            OXYZ,LMASS,AMASS)
       !tr     &                OXYZ)
       !tr<-- 04142004
       !av
       MDIP=SQRT(XDIP*XDIP+YDIP*YDIP+ZDIP*ZDIP)
       call set_param('RDIP',MDIP)
       !
       IF(PRNLEV >= 3) THEN
          IF(OXYZ.AND.LMASS) THEN
             WRITE(OUTU,82) QTOT,XDIP,YDIP,ZDIP,MDIP
82           FORMAT(' THE TOTAL CHARGE OF SELECTED ATOMS IS:',F12.6, &
                  /' DIPOLE MOMENT ABOUT CENTER OF MASS  (DEBYES) :', &
                  3F12.6,5X,F12.6)
          ELSEIF(OXYZ) THEN
             WRITE(OUTU,83) QTOT,XDIP,YDIP,ZDIP,MDIP
83           FORMAT(' THE TOTAL CHARGE OF SELECTED ATOMS IS:',F12.6, &
                  /' DIPOLE MOMENT ABOUT CENTER OF GEOMETRY  (DEBYES) :', &
                  3F12.6,5X,F12.6)
          ELSE
             WRITE(OUTU,84) QTOT,XDIP,YDIP,ZDIP,MDIP
84           FORMAT(' THE TOTAL CHARGE OF SELECTED ATOMS IS:',F12.6, &
                  /' DIPOLE MOMENT (DEBYES) :', &
                  3F12.6,5X,F12.6)
          ENDIF
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DRMS') THEN
       NBPAIR=0
       DRMS=ZERO
       XCIJ=ZERO
       YCIJ=ZERO
       ZCIJ=ZERO
       XIJ=ZERO
       YIJ=ZERO
       ZIJ=ZERO
       NMISS=0
! Default to all atoms if no selection given
       IF(NSLCT == 0) THEN
         NSLCT=NATOM
         ISLCT(1:NATOM)=1
       ENDIF
! and if no second selection, let it be identical to first 
       IF(NSLCT2 == 0) THEN
         NSLCT2=NSLCT
         JSLCT(1:NSLCT2)=ISLCT(1:NSLCT)
       ENDIF
! convert to atom pointers
       CALL MAKIND(NATOM,ISLCT,ISLCT,NSLCT)
       CALL MAKIND(NATOM,JSLCT,JSLCT,NSLCT2)
!
       DO I=1,NSLCT
          IF(X(ISLCT(I)) == ANUM .OR. XCOMP(ISLCT(I)) == ANUM) THEN
            NMISS=NMISS+1
          ELSE
            DO J=1,NSLCT2
              IF(X(JSLCT(J)) == ANUM .OR. XCOMP(JSLCT(J)) == ANUM) THEN
                NMISS=NMISS+1
              ELSE 
                IF(ISLCT(I) /= JSLCT(J) ) THEN ! Exclude trivial 0-distance to self
                  NBPAIR=NBPAIR+1
                  XIJ=(X(ISLCT(I))-X(JSLCT(J)))**2
                  YIJ=(Y(ISLCT(I))-Y(JSLCT(J)))**2
                  ZIJ=(Z(ISLCT(I))-Z(JSLCT(J)))**2
                  XCIJ=(XCOMP(ISLCT(I))-XCOMP(JSLCT(J)))**2
                  YCIJ=(YCOMP(ISLCT(I))-YCOMP(JSLCT(J)))**2
                  ZCIJ=(ZCOMP(ISLCT(I))-ZCOMP(JSLCT(J)))**2
                  RIJ=SQRT(XIJ+YIJ+ZIJ)
                  RCIJ=SQRT(XCIJ+YCIJ+ZCIJ)
                  DRMS=DRMS+(RIJ-RCIJ)**2
                ENDIF
              ENDIF
            ENDDO !JSLCT
          ENDIF 
        ENDDO !ISLCT
!
        IF(NBPAIR > 0)THEN
          DRMS=SQRT(DRMS/NBPAIR) !Distance based RMS
        ELSE
          DRMS=MINONE
        ENDIF
!
        CALL set_param('NBPAIR',NBPAIR)
        call set_param('DRMS',DRMS)
        IF(PRNLEV >= 3) WRITE(OUTU,191) NBPAIR,DRMS
 191    FORMAT('     Number of atom pairs: ',I12,/ &
               '     DISTANCE BASED RMS DIFF:',F10.3)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'RMS ') THEN
       ! to process-rms-command
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       !
       AMASST=0.0
       RMST=0.0
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                IF(XCOMP(I) == ANUM) THEN
                   NMISS=NMISS+1
                ELSE
                   AMASSV=1.0
                   IF(LWEIG) AMASSV=WMAIN(I)
                   IF(LMASS) AMASSV=AMASSV*AMASS(I)
                   AMASST=AMASST+AMASSV
                   RMST=RMST+(X(I)-XCOMP(I))**2*AMASSV
                   RMST=RMST+(Y(I)-YCOMP(I))**2*AMASSV
                   RMST=RMST+(Z(I)-ZCOMP(I))**2*AMASSV
                ENDIF
             ENDIF
          ENDIF
       ENDDO
       !
       IF(AMASST > 0.0) THEN
          RMSV=SQRT(RMST/AMASST)
       ELSE
          RMSV=0.0
       ENDIF
       call set_param('RMS ',RMSV)
       call set_param('MASS',AMASST)
       IF(PRNLEV >= 3) WRITE(OUTU,91) RMST,AMASST,RMSV
91     FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
            '       THUS RMS DIFF IS',F12.6)
       IF(PRNLEV >= 3) WRITE(OUTU,92)
92     FORMAT(' RMS FOUND FOR SELECTED COORDINATES WITHOUT ROTATION')
       !-----------------------------------------------------------------------
    ELSE IF (WRD == 'TMSC') THEN
       call tmscore(X, Y, Z, XCOMP, YCOMP, ZCOMP, NATOM, NRES, ISLCT)
    ELSE IF(WRD == 'DIFF') THEN
       ! to process-diff-command
       DO I=1,NATOM
          IF(LWEIG) THEN
             IF(ISLCT(I) == 1) WMAIN(I)=WMAIN(I)-WCOMP(I)
          ELSE IF(ISLCT(I) /= 1) THEN
             CONTINUE
          ELSE IF(XCOMP(I) == ANUM) THEN
             NMISS=NMISS+1
             X(I)=0.0
             Y(I)=0.0
             Z(I)=0.0
          ELSE IF(X(I) == ANUM) THEN
             NMISS=NMISS+1
             X(I)=0.0
             Y(I)=0.0
             Z(I)=0.0
          ELSE
             X(I)=X(I)-XCOMP(I)
             Y(I)=Y(I)-YCOMP(I)
             Z(I)=Z(I)-ZCOMP(I)
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,185) CNAME
185    FORMAT(' SELECTED COORDINATES DIFFERENCED TO THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'FORC') THEN
       ! to process-forc-command
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             X(I)=DX(I)
             Y(I)=DY(I)
             Z(I)=DZ(I)
             IF(LMASS) THEN
                IF(LONE(I)) THEN
                   X(I)=0.0
                   Y(I)=0.0
                   Z(I)=0.0
                ELSE
                   X(I)=X(I)/AMASS(I)
                   Y(I)=Y(I)/AMASS(I)
                   Z(I)=Z(I)/AMASS(I)
                ENDIF
             ENDIF
          ELSE
             X(I)=0.0
             Y(I)=0.0
             Z(I)=0.0
          ENDIF
       ENDDO

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
       if (q_domdec) then
          call copy_to_all(x, y, z)
       else
#endif 
#if KEY_SPACDEC==1
          CALL SPACBR(X,   NATOM,ICPUMAP)
          CALL SPACBR(Y,   NATOM,ICPUMAP)
          CALL SPACBR(Z,   NATOM,ICPUMAP)
#else /**/
          CALL VDGBR(X,Y,Z,1)
#endif 
#if KEY_DOMDEC==1
       endif  
#endif
#endif 

       IF(PRNLEV >= 3) WRITE(OUTU,285) CNAME
285    FORMAT(' SELECTED FORCES COPIED TO THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SHAK' .OR. WRD == 'HOLO') THEN
       ! to process-shak-command
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       DO I=1,NATOM
          ISLCT(I)=1-ISLCT(I)
       ENDDO
       LOK=.TRUE.
       DO I=1,NCONST
          IF(.NOT.INITIA(SHKAPR(1,I),X,Y,Z)) LOK=.FALSE.
          IF(.NOT.INITIA(SHKAPR(2,I),X,Y,Z)) LOK=.FALSE.
          IF(.NOT.INITIA(SHKAPR(1,I),XCOMP,YCOMP,ZCOMP)) LOK=.FALSE.
          IF(.NOT.INITIA(SHKAPR(2,I),XCOMP,YCOMP,ZCOMP)) LOK=.FALSE.
       ENDDO
       IF(LOK) THEN
          CALL HOLONOMA(X,Y,Z,XCOMP,YCOMP,ZCOMP,LMASS,.FALSE.,QOK)
          IF(PRNLEV >= 3) WRITE(OUTU,102) CNAME
102       FORMAT(' SELECTED COORDINATES CONSTRAINED IN THE ',A, &
               ' SET.'/)
       ELSE
          CALL WRNDIE(0,'<CORMAN>','Missing coordinates for holonom')
       ENDIF
#if KEY_PARALLEL==1
       ! Note: SHAKE contraints are local, so we need a VDGBR.
       ! replaced for full broadcast(not a problem for performance)
       ! but complete coordinates everywhere!
       !        CALL VDGBR(X,Y,Z,1)
       !
       CALL PSND8(X,NATOM)
       CALL PSND8(Y,NATOM)
       CALL PSND8(Z,NATOM)
#endif 
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DIST') THEN
       ! to process-dist-command
       !     CALCULATE THE DISTANCES BETWEEN ATOMS
       IF(LWEIG) THEN
          ! get-distance-spec
          DO I=1,NATOM
             IF(ISLCT(I) == 1) THEN
                IF(X(I) == ANUM) THEN
                   WMAIN(I)=ANUM
                ELSE
                   WMAIN(I)=SQRT((X(I)-XDIR)**2 + (Y(I)-YDIR)**2 + &
                        (Z(I)-ZDIR)**2)
                ENDIF
             ENDIF
          ENDDO
       ELSE
          !
          ! select-second-atoms
          CALL DISTAN(COMLYN,COMLEN,LSEL2,.FALSE.,.FALSE., &
               X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT,JSLCT)
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'MIND') THEN
       ! to process-mind-command
       ! select-second-atoms
       CALL DISTAN(COMLYN,COMLEN,LSEL2,.TRUE.,.FALSE., &
            X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT,JSLCT)
       !-----------------------------------------------------------------------
       !----------------------Denzil-------------------------------------------
    ELSE IF(WRD == 'MAXD') THEN
       ! to process-maxd-command
       ! select-second-atoms
       CALL DISTAN(COMLYN,COMLEN,LSEL2,.FALSE.,.TRUE., &
            X,Y,Z,XCOMP,YCOMP,ZCOMP,NATOM,ISLCT,JSLCT)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'HIST') THEN
       ! to process-HISTogram-command
       CALL CORHIST(COMLYN,COMLEN,X,Y,Z,WMAIN,LWEIG,ISLCT)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DUPL') THEN
       ! to process-dupl-command
       !
       !     DUPLICATE COORDINATES WITHIN A STRUCTURE
       !
       IF(INDXA(COMLYN,COMLEN,'PREV') == 0) THEN
          ! select-second-atoms
          IF(WRNLEV >= 2) THEN
             IF(NSLCT /= NSLCT2) THEN
                IF(WRNLEV >= 3) WRITE(OUTU,146) NSLCT,NSLCT2
146             FORMAT(' WARNING: NUMBER OF SELECTED ATOMS DOESNT', &
                     ' MATCH.',2I5, &
                     /' THE SMALLER NUMBER WILL BE USED')
             ENDIF
          ENDIF
          !
          !     Distributed duplication:
          !     Always send from process 0 to the one specified in NREP
          !     This has to be exclusive, ie only do the manipulations
          !     on the target replica, don't change anything on others!
          !
#if KEY_REPDSTR==1
          NXREP=GTRMI(COMLYN,COMLEN,'NREP',-1)
          QCSND=(NXREP > 0).AND.QREPDSTR.AND.(MYNODG == 0)
          QCRCV=(NXREP > 0).AND.QREPDSTR.AND.(MYNODG == NXREP)
          QCIGN=(NXREP > 0).AND.QREPDSTR.AND.(MYNODG /= NXREP)
          !            write(*,*)'CORMAN-0>meg,nxrep,snd,rcv,ign=',
          !     $           mynodg,nxrep,qcsnd,qcrcv,qcign
#endif 
          !
          NSLCT=MIN(NSLCT,NSLCT2)
          !
          IPT=0
          DO I=1,NSLCT
210          CONTINUE
             IPT=IPT+1
             IF (ISLCT(IPT) /= 1) GOTO 210
             RWORK(I)=X(IPT)
          ENDDO
#if KEY_REPDSTR==1
          IF(QCSND) CALL GSEN(NXREP,1,RWORK,8*NSLCT)
          IF(QCRCV) CALL GREC(0,1,RWORK,8*NSLCT)
          IF(QCIGN) GOTO 1220
#endif 
          JPT=0
          DO I=1,NSLCT
220          CONTINUE
             JPT=JPT+1
             IF (JSLCT(JPT) /= 1) GOTO 220
             X(JPT)=RWORK(I)
          ENDDO
1220      CONTINUE
          IPT=0
          DO I=1,NSLCT
211          CONTINUE
             IPT=IPT+1
             IF (ISLCT(IPT) /= 1) GOTO 211
             RWORK(I)=Y(IPT)
          ENDDO
#if KEY_REPDSTR==1
          IF(QCSND) CALL GSEN(NXREP,1,RWORK,8*NSLCT)
          IF(QCRCV) CALL GREC(0,1,RWORK,8*NSLCT)
          IF(QCIGN) GOTO 1221
#endif 
          JPT=0
          DO I=1,NSLCT
221          CONTINUE
             JPT=JPT+1
             IF (JSLCT(JPT) /= 1) GOTO 221
             Y(JPT)=RWORK(I)
          ENDDO
1221      CONTINUE
          IPT=0
          DO I=1,NSLCT
212          CONTINUE
             IPT=IPT+1
             IF (ISLCT(IPT) /= 1) GOTO 212
             RWORK(I)=Z(IPT)
          ENDDO
#if KEY_REPDSTR==1
          IF(QCSND) CALL GSEN(NXREP,1,RWORK,8*NSLCT)
          IF(QCRCV) CALL GREC(0,1,RWORK,8*NSLCT)
          IF(QCIGN) GOTO 1222
#endif 
          JPT=0
          DO I=1,NSLCT
222          CONTINUE
             JPT=JPT+1
             IF (JSLCT(JPT) /= 1) GOTO 222
             Z(JPT)=RWORK(I)
          ENDDO
1222      CONTINUE
       ELSE
          ! ONLY ONE SELECTION SPECIFIED. USE PREVIOUS ATOM INDICIES
          ! IF PROPER KEYWORD GIVEN.
          DO I=2,NATOM
             IF(ISLCT(I) == 1) THEN
                X(I)=X(I-1)
                Y(I)=Y(I-1)
                Z(I)=Z(I-1)
                WMAIN(I)=WMAIN(I-1)
             ENDIF
          ENDDO
          IF(PRNLEV >= 3) WRITE(OUTU,147)
147       FORMAT(' COORDINATES OF SELECTED ATOMS GENERATED FROM', &
               ' PREVIOUS ATOMS')
       ENDIF
       !
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'OPER') THEN
       ! to process-oper-command
       WRD8=NEXTA8(COMLYN,COMLEN)
       ITRANS=0
       DO I=1,NTRANS
          IF(IMNAME(I) == WRD8) ITRANS=I
       ENDDO
       IF(ITRANS == 0) THEN
          CALL WRNDIE(0,'<CORMAN>','NO SUCH TRANSFORMATION')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       !
       DO I=1,NATOM
          IF(ISLCT(I) == 1) THEN
             IF(X(I) == ANUM) THEN
                NMISS=NMISS+1
             ELSE
                XN=X(I)
                YN=Y(I)
                ZN=Z(I)
                X(I)=XN*IMTRNS(1,ITRANS)+YN*IMTRNS(2,ITRANS)+ &
                     IMTRNS(3,ITRANS)+IMTRNS(10,ITRANS)
                Y(I)=XN*IMTRNS(4,ITRANS)+YN*IMTRNS(5,ITRANS)+ &
                     ZN*IMTRNS(6,ITRANS)+IMTRNS(11,ITRANS)
                Z(I)=XN*IMTRNS(7,ITRANS)+YN*IMTRNS(8,ITRANS)+ &
                     ZN*IMTRNS(9,ITRANS)+IMTRNS(12,ITRANS)
             ENDIF
          ENDIF
       ENDDO
       IF(PRNLEV >= 3) WRITE(OUTU,161) (IMTRNS(I,ITRANS),I=1,12)
161    FORMAT(' TRANSFORMATION MATRIX ',4(/3F12.6))
       IF(PRNLEV >= 3) WRITE(OUTU,162) CNAME
162    FORMAT(' SELECTED COORDINATES TRANSFORMED IN THE ',A,' SET.'/)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'STAT') THEN
       ! to process-stat-command
       !     COMPUTE AND PRINT SOME SIMPLE STATISTICS
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)

       call calc_coor_stats(x, y, z, wmain, amass, natom, islct, lmass, &
            stat_names, stat_vals, nslct2, nmiss)

245    FORMAT(' STATISTICS FOR',I10,' SELECTED ATOMS:')
       IF (PRNLEV >= 3) WRITE(OUTU,245) NSLCT2

246    FORMAT('    XMIN =',F12.6,' XMAX =',F12.6,' XAVE =',F12.6/ &
            '    YMIN =',F12.6,' YMAX =',F12.6,' YAVE =',F12.6/ &
            '    ZMIN =',F12.6,' ZMAX =',F12.6,' ZAVE =',F12.6/ &
            '    WMIN =',F12.6,' WMAX =',F12.6,' WAVE =',F12.6)
       if (prnlev >= 3) then
          write(outu, 246) &
               stat_vals(1), stat_vals(5), stat_vals(9), &  ! x min, max, ave
               stat_vals(2), stat_vals(6), stat_vals(10), & ! y min, max, ave
               stat_vals(3), stat_vals(7), stat_vals(11), & ! z min, max, ave
               stat_vals(4), stat_vals(8), stat_vals(12)  ! w min, max, ave
       end if

       do i = 1, n_stats
          call set_param(stat_names(i), stat_vals(i))
       end do
       ! end coor stat command
    ELSE IF(WRD == 'SEAR') THEN
       IF(.NOT.CSSAVE) THEN
          IF(allocated(csjmat)) THEN
             call chmdealloc('corman.src','CORMAN','CSjMAT',CSSPACE,intg=CSjMAT)
             CSSPACE=0
          ENDIF
          CSNSAVE=0
          CSSPACE=0
       ENDIF
       ! to process-search-command
       !     parse print option
       LPRINT=.NOT.(INDXA(COMLYN,COMLEN,'NOPR') > 0)
       LPRINT=(INDXA(COMLYN,COMLEN,'PRIN') > 0)
       IF(LPRINT) IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
       !     parse save option
       IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) CSSAVE=.TRUE.
       IF(INDXA(COMLYN,COMLEN,'NOSA') > 0) CSSAVE=.FALSE.
       !     parse operation selection
       !CC         IF(INDXA(COMLYN,COMLEN,'INVE') > 0) THEN
       !CC
       !CC         ELSEIF(INDXA(COMLYN,COMLEN,'KEEP') > 0) THEN
       !CC
       !CC         ELSEIF(INDXA(COMLYN,COMLEN,'EXTE') > 0) THEN
       !CC
       !CC         ELSE
       !CC
       !CC         ENDIF
       !
       !     parse calculation mode
       LHOLES=(INDXA(COMLYN,COMLEN,'HOLE') > 0)
       LVACU=(INDXA(COMLYN,COMLEN,'VACU') > 0)
       LVACU=.NOT.(INDXA(COMLYN,COMLEN,'FILL') > 0)
       IF(LHOLES) LVACU=.TRUE.
       !     parse operation mode
       PSOPER=0
       IF(INDXA(COMLYN,COMLEN,'NAND') > 0) PSOPER=5
       IF(INDXA(COMLYN,COMLEN,'AND') > 0) PSOPER=1
       IF(INDXA(COMLYN,COMLEN,'XOR') > 0) PSOPER=3
       IF(INDXA(COMLYN,COMLEN,'OR') > 0) PSOPER=2
       IF(INDXA(COMLYN,COMLEN,'ADD') > 0) PSOPER=4
       IF(INDXA(COMLYN,COMLEN,'RESE') > 0) PSOPER=0
       !     parse buffer values
       RCUT=GTRMF(COMLYN,COMLEN,'RCUT',ZERO)
       RBUFF=GTRMF(COMLYN,COMLEN,'RBUF',ZERO)
       LVOLUM=(RCUT > ZERO .OR. RBUFF > ZERO)
       IF(LVOLUM) THEN
          siz_scr=natom
          call chmalloc('corman.src','CORMAN','SCR',NATOM,crl=SCR)
          IF(RCUT > ZERO) scr(1:natom) = rcut
          IF(RBUFF > ZERO) THEN
             scr(1:natom)=rbuff
             CALL ADDVEC(WMAIN,SCR,SCR,NATOM)
          ENDIF
       else
          siz_scr=1
          call chmalloc('corman.src','CORMAN','SCR',siz_scr,crl=SCR)        
       ENDIF
       !     parse grid boundaries and spacing
       CSXMIN=GTRMF(COMLYN,COMLEN,'XMIN',CSXMIN)
       CSXMAX=GTRMF(COMLYN,COMLEN,'XMAX',CSXMAX)
       CSYMIN=GTRMF(COMLYN,COMLEN,'YMIN',CSYMIN)
       CSYMAX=GTRMF(COMLYN,COMLEN,'YMAX',CSYMAX)
       CSZMIN=GTRMF(COMLYN,COMLEN,'ZMIN',CSZMIN)
       CSZMAX=GTRMF(COMLYN,COMLEN,'ZMAX',CSZMAX)
       AMAX=MAX(CSXMAX,CSXMIN,CSYMAX,CSYMIN,CSZMAX,CSZMIN)
       IF(AMAX >= ANUM) THEN
          CALL WRNDIE(0,'<CORMAN>','ALL LIMITS MUST BE SPECIFIED')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       CSXGRID=GTRMI(COMLYN,COMLEN,'XGRI',CSXGRID)
       CSYGRID=GTRMI(COMLYN,COMLEN,'YGRI',CSYGRID)
       CSZGRID=GTRMI(COMLYN,COMLEN,'ZGRI',CSZGRID)
       ISPACE=CSXGRID*CSYGRID*CSZGRID
       IF(ISPACE == 0) THEN
          CALL WRNDIE(0,'<CORMAN>','MUST SPECIFY ALL GRID INTEGERS')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       !
       IF(PSOPER > 0) THEN
          IF(ISPACE /= CSSPACE) THEN
             CALL WRNDIE(0,'<CORMAN>', &
                  'Must have same size grid as previous COOR SEARch')
             call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
             nullify(islct,jslct)
             call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
             RETURN
          ENDIF
       ELSE
          !           not using the old grid, free it..
          IF(allocated(csjmat)) THEN
             call chmdealloc('corman.src','CORMAN','CSjMAT',CSSPACE,intg=CSjMAT)
             CSSPACE=0
          ENDIF
          CSNSAVE=0
       ENDIF

       call chmalloc('corman.src','CORMAN','jmat',ISPACE+natom,intg=jmat)
       call chmalloc('corman.src','CORMAN','IUTIL',NATOM,intg=IUTIL)
       call chmalloc('corman.src','CORMAN','space_rl',CSXGRID+csygrid+cszgrid,crl=space_rl)
       xmat => space_rl(1:csxgrid)
       ymat => space_rl(csxgrid+1:csxgrid+csygrid)
       zmat => space_rl(csxgrid+csygrid+1:csxgrid+csygrid+cszgrid)

       CALL VSEARC(NATOM,X,Y,Z,WMAIN,CSXGRID,CSYGRID,CSZGRID, &
            XMAT,YMAT,ZMAT, &
            CSXMIN,CSXMAX,CSYMIN,CSYMAX,CSZMIN,CSZMAX, &
            ISLCT,LPRINT,IUNIT, &
            jMAT,CSjMAT,PSOPER,LVACU,LHOLES, &
            LVOLUM,CSNSAVE,WMAIN,SCR)
       !
       ! Create dummy atoms if requested
       SID=GTRMA(COMLYN,COMLEN,'CREA')
       IF(SID /= ' ') THEN
          PDEF=GTRMA(COMLYN,COMLEN,'CHEM')
          IF(PDEF == ' ') THEN
             CALL WRNDIE(0,'<CORMAN>', &
                  'No atom type specified with SEARch CREAte option')
          ELSE
             !            create new atoms...
             CALL ADDSRCHAT(SID,PDEF,CSXGRID,CSYGRID,CSZGRID, &
                  XMAT,YMAT,ZMAT,jMAT)
          ENDIF
       ENDIF
       !
       ! Save results if requested
       IF( allocated(CSjmat)) &
            call chmdealloc('corman.src','CORMAN','csspace',ISPACE,intg=csjmat)

       CSSPACE=0
       IF(CSSAVE) THEN
          CSSPACE=ISPACE
          call chmalloc('corman.src','CORMAN','csjmat',ISPACE,intg=csjmat)
          CSjMAT=jMAT
          call chmdealloc('corman.src','CORMAN','jmat',ISPACE,intg=jmat)
          CSNSAVE=CSNSAVE+1
       ELSE
          call chmdealloc('corman.src','CORMAN','jmat',ISPACE,intg=jmat)
          CSNSAVE=0
       ENDIF

       call chmdealloc('corman.src','CORMAN','IUTIL',NATOM,intg=IUTIL)
       call chmdealloc('corman.src','CORMAN','space_rl',CSXGRID+csygrid+cszgrid,crl=space_rl)
       nullify(xmat,ymat,zmat,scr2,jmat2)
       call chmdealloc('corman.src','CORMAN','SCR',NATOM,crl=SCR)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'VOLU') THEN
       ! to process-volume-command
       IF(ptrSTO(1)%len <= 0 .OR. ptrSTO(2)%len <= 0) THEN
          CALL WRNDIE(-1,'<CORMAN>','MUST FILL SCALAR ARRAYS TO USE')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       ISPACE=GTRMI(COMLYN,COMLEN,'SPAC',0)
       IF(ISPACE == 0) THEN
          CALL WRNDIE(0,'<CORMAN>','MUST SPECIFY THE SPACE PARAMETER')
          call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
          nullify(islct,jslct)
          call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
          RETURN
       ENDIF
       LHOLES=(INDXA(COMLYN,COMLEN,'HOLE') > 0)
       CALL VOLUME(NATOM,X,Y,Z,WMAIN,ISLCT,PTRSTO(1)%a,PTRSTO(2)%a,ISPACE,LHOLES)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SURF') THEN
       ! to process-surface-command
#if KEY_NOMISC==0
       call chmalloc('corman.src','CORMAN','space_rl',4*nslct+natom,crl=space_rl)
       xmat => space_rl(1:nslct)
       ymat => space_rl(nslct+1:2*nslct)
       zmat => space_rl(2*nslct+1:3*nslct)
       scr2 => space_rl(3*nslct+1:4*nslct)
       jmat2 => space_rl(4*nslct+1:4*nslct+natom)
       IMODE=-1
       CALL SURFAC(NATOM,X,Y,Z,WMAIN,LWEIG,ISLCT,NSLCT,XMAT, &
            YMAT,ZMAT,jMAT2,SCR2, &
            IMODE,RPRO,ACCURACY)
       call chmdealloc('corman.src','CORMAN','space_rl',4*nslct+natom,crl=space_rl)
       nullify(xmat,ymat,zmat,scr2,jmat2)
#else /**/
       CALL WRNDIE(-1,'<CORMAN>','SURFAC code is not compiled')
#endif 
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DRAW') THEN
       ! to process-draw-command
#if KEY_NOMISC==0
       CALL DRAWSP(COMLYN,COMLEN,X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,ISLCT)
#else /**/
       CALL WRNDIE(-1,'<CORMAN>','DRAW code is not compiled')
#endif 
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'CONV') THEN
       ! to process-conv-command
       WRD=NEXTA4(COMLYN,COMLEN)
       WRD2=NEXTA4(COMLYN,COMLEN)
       XUCELL(1)=NEXTF(COMLYN,COMLEN)
       XUCELL(2)=NEXTF(COMLYN,COMLEN)
       XUCELL(3)=NEXTF(COMLYN,COMLEN)
       XUCELL(4)=NEXTF(COMLYN,COMLEN)
       XUCELL(5)=NEXTF(COMLYN,COMLEN)
       XUCELL(6)=NEXTF(COMLYN,COMLEN)
       CALL XTRANE(COMLYN,COMLEN,'CORMAN')
       CALL CONCOR(WRD,WRD2,X,Y,Z,NATOM,ISLCT,XUCELL)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'RGYR') THEN
       ! to process-rgyr-command
       !
       !     If MASS keyword present, use AMASS as weighting
       !     IF WEIG keyword present, use WMAIN as weighting
       !     OTHERWISE use number of electrons per atom as weighting
       !
       !     The use of main or comparison coordinates is assumed to be
       !     taken care of by the CHARMM call to CORMAN.
       !
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       FACT=GTRMF(COMLYN,COMLEN,'FACT',ZERO)
       !
       CALL RGYR(NATOM,X,Y,Z,WMAIN,AMASS,ISLCT,FACT,LMASS,LWEIG)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'LSQP') THEN
       ! to process-lsqp-command
       !
       !     If MASS keyword present, use AMASS as weighting
       !     IF WEIG keyword present, use WMAIN as weighting
       !
       !     The use of main or comparison coordinates is assumed to be
       !     taken care of by the CORMAN call. LSQP code is in RGYR.FLX
       !
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       LVERB=(INDXA(COMLYN,COMLEN,'VERB') > 0)
       IUTIL0=0
       IF(INDXA(COMLYN,COMLEN,'MAJO') > 0) IUTIL0=1
       IF(INDXA(COMLYN,COMLEN,'MINO') > 0) IUTIL0=2
       IF(INDXA(COMLYN,COMLEN,'NORM') > 0) IUTIL0=3
       !
       CALL LSQP(NATOM,X,Y,Z,WMAIN,AMASS,ISLCT,LVERB,LMASS,LWEIG, &
            IUTIL0)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'DYNA') THEN
       ! to process-dynamics-averages
       !
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,IUNIT,NBEGN,NSKIP,NSTOP)
       !
       
       !
       LPRINT=.NOT.(INDXA(COMLYN,COMLEN,'NOPR') > 0)
       QPAX=(INDXA(COMLYN,COMLEN,'PAX') > 0)
       if(qpax)call chmalloc('corman.src','CORMAN','RPAX',15,NATOM,crl=rpax)
       !
       !br...04-Feb-96, MASS weighting of dynamics coordinate average
       ORIENT = INDXA(COMLYN, COMLEN, 'ORIE')  >  0
       IF(ORIENT)THEN
          IF(NSLCT2 == 0)THEN
             CALL WRNDIE(-2,'<CORMAN>','ZERO ATOM SELECTED ')
          ENDIF
          LNORO  = INDXA(COMLYN, COMLEN, 'NORO')  >  0
          LMASS  = INDXA(COMLYN, COMLEN, 'MASS')  >  0
          LWMAIN = INDXA(COMLYN, COMLEN, 'WMAI')  >  0
          WRITE(OUTU,'(6X,A,A)') &
               '* ALL COORDINATE FRAMES WILL BE REORIENTED WITH ', &
               ' RESPECT TO THE COMPARISON SET *'
          IF(LMASS)THEN
             WRITE(OUTU,'(6X,A)')'  MASS WEIGHTING WILL BE USED '
          ENDIF
          IF(LWMAIN)THEN
             WRITE(OUTU,'(6X,A)')'  WMAIN WEIGHTING WILL BE USED '
          ENDIF
          IF(LNORO)THEN
             WRITE(OUTU,'(6X,A)')'  NO ROTATIONS WILL BE USED '
          ENDIF
       ENDIF
       !
       CALL AVECOR(X,Y,Z,WMAIN,NATOM,NUNIT,IUNIT,NSKIP,NBEGN,NSTOP,ISLCT, &
            LPRINT,QPAX,rPAX, &
            ORIENT,LNORO,XCOMP,YCOMP,ZCOMP,WCOMP,JSLCT, &
            LMASS,LWMAIN,AMASS)

       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'HELI') THEN
#if KEY_CONSHELIX==1 /*conshelix*/
            OFIRS=(IndxA(Comlyn, Comlen,'1FIR') > 0)
            OLAST=(IndxA(Comlyn, Comlen,'1LAS') > 0)
            TFIRS=(IndxA(Comlyn, Comlen,'2FIR') > 0)
            TLAST=(IndxA(Comlyn, Comlen,'2LAS') > 0)
            LHMODC=.TRUE.
            IF((OLAST.AND.TFIRS).OR.(OFIRS.AND.TLAST)) THEN
               LHMODC=.FALSE.
            ENDIF
            IF(PRNLEV >= 6) WRITE(6,*) 'corman.src',LHMODC
            ! true
            IF(LHMODC) THEN
               WRITE(6,*) 'CHARMM selection followed (Default)'
            ELSE
               ! false
               WRITE(6,*) 'Hinge mode selection followed'
            ENDIF
            ! DNA mode for bhairpin or dna structure
            ! using helical moment of inertia
            LDNAM=(IndxA(Comlyn, Comlen,'DNA') > 0)
            ! BHP mode for bhairpin
            LBHPM=(IndxA(Comlyn, Comlen,'BHP') > 0)
            ! BHG mode for bhairpin using general moment of inertia
            LBHGM=(IndxA(Comlyn, Comlen,'BHG') > 0)
            REFAT=GTRMI(COMLYN,COMLEN,'REFA',1)
            ! Reference Axis for Rotation Angle
            LRAXM=(IndxA(Comlyn, Comlen,'ROTX') > 0)
            LRAYM=(IndxA(Comlyn, Comlen,'ROTY') > 0)
            LRAZM=(IndxA(Comlyn, Comlen,'ROTZ') > 0)
#endif /* (conshelix)*/
       ! to process-helix-command
       CALL TRIMA(COMLYN,COMLEN)
       IF(NSLCT2 > 0) THEN
          ! select-second-atoms
          CALL HELIX2(2,ISLCT,JSLCT,NATOM,X,Y,Z)
       ELSE
          CALL HLXSP1(COMLYN,COMLEN,NATOM,ISLCT)
          IF(NHLXAT  <=  0) THEN
             IF(WRNLEV >= 3) WRITE(OUTU,'(A)') &
                  '%CORMAN-WARNING: No atoms selected to define helix'
          ELSE
             ! analyze helix as in Chothia Levitt and Richardson
             CALL HELIX2(1,ISLCT,JSLCT,NATOM,X,Y,Z)
             ! analyze helix as in Aqvist
             DO I=1,3
                AX(I)=0.0
                R0(I)=0.0
             ENDDO
             CALL HELIX1(AX,R0,NDIG)
             IF(NDIG  >  0) THEN
                IF(PRNLEV >= 3) WRITE(OUTU,310) AX,R0,NDIG
310             FORMAT(/'     HELIX AXIS:',3F10.6/ &
                     '     Perp. vector to origin:',3F10.6/ &
                     '     Estimated number of significant digits:',I4/)
                !
                ! set the axis parameters for the next COOR ... AXIS command
                AXISX = AX(1)
                AXISY = AX(2)
                AXISZ = AX(3)
                AXISR = ONE
                AXISCX= R0(1)
                AXISCY= R0(2)
                AXISCZ= R0(3)
                !
                call set_param('XAXI',AXISX)
                call set_param('YAXI',AXISY)
                call set_param('ZAXI',AXISZ)
                call set_param('RAXI',AXISR)
                call set_param('XCEN',AXISCX)
                call set_param('YCEN',AXISCY)
                call set_param('ZCEN',AXISCZ)
                !
             ELSE
                IF(WRNLEV >= 3) WRITE(OUTU,320) -NDIG,AX,R0
320             FORMAT(/'     Something went wrong....IER=',I5,'???'/ &
                     '     HELIX AXIS (as found):',3F10.6/ &
                     '     Perp. vector to origin:',3F10.6/)
             ENDIF
          ENDIF
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'HBON' .OR. WRD == 'CONT') THEN
       ! to process-hbond-or-contact analysis
       ! allocate space for hb donor accpetor flags
       CALL HBANAL(COMLYN,COMLEN,ISLCT,JSLCT,WRD)
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'PAXA') THEN
       ! to process-pax-analysis
       !
       LSAVE =(INDXA(COMLYN,COMLEN,'SAVE') > 0)
       IF( .not. allocated(rpax)) THEN
          CALL WRNDIE(0,'<CORMAN>', &
               'No PAX present (run COOR DYNA PAX)')
       ELSE
          CALL TRJSPC(COMLYN,COMLEN,NUNIT,IUNIT,NBEGN,NSKIP,NSTOP)

          LPRINT=.NOT.(INDXA(COMLYN,COMLEN,'NOPR') > 0)

          CALL PAXANL(X,Y,Z,NATOM,NUNIT,IUNIT,NSKIP,NBEGN,NSTOP,ISLCT, &
               rpax,AMASS,LPRINT)

          IF(.NOT.LSAVE) THEN
             call chmdealloc('corman.src','CORMAN','rpax',15,NATOM,crl=rpax)
          ENDIF
       ENDIF
       !-----------------------------------------------------------------------
    ELSE IF(WRD == 'SMAP') THEN  ! hwm
       WRD2=NEXTA4(COMLYN,COMLEN)
       CALL TRJSPC(COMLYN,COMLEN,NUNIT,IUNIT,NBEGN,NSKIP,NSTOP)
       OUNIT=GTRMI(COMLYN,COMLEN,'IUNW',-1) ! output file unit 
       FACT=GTRMF(COMLYN,COMLEN,'RESO',1.4D0)
       RCUT=GTRMF(COMLYN,COMLEN,'RCUT',ANUM) ! default 9999
       CDUM=GTRMA(COMLYN,COMLEN,'SOLV') ! solvent atom
       if (CDUM == '') then
          CDUM='OH2' ! default solvent atom
       endif

!  write (outu,'(A,4I8)') 'iunit,nbegn,nskip,nstop ',iunit,nbegn,nskip,nstop

       OUTMODE=0
       IF(INDXA(COMLYN,COMLEN,'EMAP').NE.0) OUTMODE=1 
       IF(INDXA(COMLYN,COMLEN,'CARD').NE.0) OUTMODE=2
       if ((outmode .ne. 1 ) .and. (outmode .ne. 2)) then
          CALL WRNDIE(-2,'<CORMAN>', &
               'OUTMODE MISSING. NEED KEYWORD EMAP OR CARD')
       endif
       if (FACT <0.01) then
         CALL WRNDIE(-1,'<CORMAN>', 'SMAP: RESOlution too small.')
       endif

       ORIENT = INDXA(COMLYN, COMLEN, 'ORIE')  >  0  
       if (ORIENT) then
          LMASS= INDXA(COMLYN, COMLEN, 'MASS')>0
          LWMAIN = INDXA(COMLYN, COMLEN, 'WMAI')>0
          LNORO=INDXA(COMLYN,COMLEN,'NORO') > 0
!#if KEY_MPI==1
!          if (mynod .eq. 0 ) then
!#endif
          if (prnlev > 3) then
             write(OUTU,'(6X,A,A)') &
                  'Coordinate frames will be reoriented.'
             if (LMASS)  write(OUTU,'(6X,A)')'Mass weighting will be used.' 
             if (LWMAIN) write(OUTU,'(6X,A)')'Wmain weighting will be used.'
             if (LNORO)  write(OUTU,'(6X,A)')'No rotation will be done.'
          endif
       endif ! if (ORIENT) then
          
       if (WRD2 == 'SDEN') THEN
          if (ounit .eq. -1) then
             CALL WRNDIE(-2,'<CORMAN>','Output file missing. Set IUNW.')
          endif
          CALL slvden(X,Y,Z,WMAIN,NATOM,nunit,IUNIT,nskip,nbegn,nstop,ISLCT, &
               ATYPE,FACT,RCUT,OUNIT,outmode,ORIENT,XCOMP,YCOMP,ZCOMP,WCOMP, &
               LMASS,LWMAIN,LNORO,AMASS,cdum)
       else if (WRD2 == 'DIFT') THEN
          rdum=GTRMF(COMLYN,COMLEN,'WDIS',-1.0) ! warning distance 
          hcut=GTRMF(COMLYN,COMLEN,'HCUT',-1.0)
          dcut=GTRMF(COMLYN,COMLEN,'DCUT',-1.0)
          ounit1=GTRMI(COMLYN,COMLEN,'IDIF',-1) ! output unit for dift
          call slv_dift(x,y,z,wmain,natom,nunit,IUNIT,nskip,nbegn,nstop, &
               islct,atype,FACT,rcut,ounit,ounit1,outmode,orient,&
               xcomp,ycomp,zcomp,wcomp,lmass,lwmain,lnoro,amass,cdum,rdum, &
               dcut,hcut)
       else if (WRD2 == 'HBON') THEN
          ! output files
          oslv=GTRMI(COMLYN,COMLEN,'ISLV',-1) ! nHbond w/ solvent (water)
          oslt=GTRMI(COMLYN,COMLEN,'ISLT',-1) ! nHbond w/ solute
          ounit1=GTRMI(COMLYN,COMLEN,'ITOT',-1) ! nHbond w/ (solvent+solute)
          rdum=GTRMF(COMLYN,COMLEN,'CUTH',2.4)
          dcut=GTRMF(COMLYN,COMLEN,'DCUT',0.0)
          if (cdum(1:1).ne. 'O') then
             CALL WRNDIE(-2,'<CORMAN>','Select an oxygen for solvent atom&
                  & for H-bond analysis.')
          endif
          call slv_hbond(x,y,z,wmain,natom,nunit,iunit,nskip,nbegn,nstop, &
               islct, atype,fact,rcut,ounit,oslv,oslt,ounit1,outmode,orient,&
               xcomp,ycomp,zcomp,wcomp,lmass,lwmain,lnoro,amass,cg,rdum,&
               dcut,ibase,nres,cdum)
       endif
       !-----------------------------------------------------------------------
    ELSE
       CALL WRNDIE(0,'<CORMAN>','Unrecognized command')
       !-----------------------------------------------------------------------
    ENDIF
    !
    IF (NMISS > 0 .AND. WRNLEV >= 2) WRITE(OUTU,22) NMISS
22  FORMAT(/' **** WARNING **** FOR THIS OPERATION, THERE WERE',I5, &
         ' MISSING COORDINATES'/)
    !
    call chmdealloc('corman.src','corman','space_int',2*natom,intg=space_int)
    nullify(islct,jslct)
    call chmdealloc('corman.src','CORCOM','RWORK',natom,crl=RWORK)
    RETURN
  END SUBROUTINE CORMAN

  subroutine calc_coor_stats(x, y, z, w, amass, natom, islct, lmass, &
       out_names, out_stats, out_n_selected, out_n_misses)
    use chm_types, only: chm_real
    use number, only: anum

    implicit none

    ! input params
    real(chm_real), dimension(*) :: x, y, z, w, amass
    integer :: natom, islct(*)
    logical :: lmass

    ! output params
    character(len=4) :: out_names(*)
    real(chm_real), dimension(*) :: out_stats
    integer :: out_n_selected, out_n_misses

    ! locals
    integer :: i, n
    real(chm_real) :: amasst, amassv, &
         xmin, ymin, zmin, wmin, &
         xmax, ymax, zmax, wmax, &
         xave, yave, zave, wave
    character(len=4) :: names(13)

    out_n_selected = 0
    out_n_misses = 0

    amasst = 0.0

    xmin = anum
    ymin = anum
    zmin = anum
    wmin = anum

    xmax = -anum
    ymax = -anum
    zmax = -anum
    wmax = -anum

    xave = 0.0
    yave = 0.0
    zave = 0.0
    wave = 0.0

    do i = 1, natom
       if (islct(i) == 1) then
          if (x(i) == anum) then
             out_n_misses = out_n_misses + 1
          else
             out_n_selected = out_n_selected + 1
             if (x(i) > xmax) xmax = x(i)
             if (x(i) < xmin) xmin = x(i)
             if (y(i) > ymax) ymax = y(i)
             if (y(i) < ymin) ymin = y(i)
             if (z(i) > zmax) zmax = z(i)
             if (z(i) < zmin) zmin = z(i)
             if (w(i) > wmax) wmax = w(i)
             if (w(i) < wmin) wmin = w(i)
             if (lmass) then
                amassv = amass(i)
             else
                amassv = 1.0
             end if
             amasst = amasst + amassv
             xave = xave + x(i) * amassv
             yave = yave + y(i) * amassv
             zave = zave + z(i) * amassv
             wave = wave + w(i) * amassv
          end if
       end if
    end do

    if (amasst /= 0.0) then
       xave = xave / amasst
       yave = yave / amasst
       zave = zave / amasst
       wave = wave / amasst
    end if

    names = (/ 'XMIN', 'YMIN', 'ZMIN', 'WMIN', &
         'XMAX', 'YMAX', 'ZMAX', 'WMAX', &
         'XAVE', 'YAVE', 'ZAVE', 'WAVE', &
         'MASS' /)
    n = size(names)

    out_names(1:n) = names(1:n)

    out_stats(1) = xmin
    out_stats(2) = ymin
    out_stats(3) = zmin
    out_stats(4) = wmin

    out_stats(5) = xmax
    out_stats(6) = ymax
    out_stats(7) = zmax
    out_stats(8) = wmax

    out_stats(9) = xave
    out_stats(10) = yave
    out_stats(11) = zave
    out_stats(12) = wave

    out_stats(13) = amasst
  end subroutine calc_coor_stats

  SUBROUTINE DISTAN(COMLYN,COMLEN,LSEL2,LMIND,LMAXD,X,Y,Z, &
       XCOMP,YCOMP,ZCOMP,NATOMX,ISLCT,JSLCT)
    !
    !     This routine processes the COOR DISTance command.
    !
    !            By Bernard R. Brooks     3-DEC-1983
    !
  use number
  use exfunc
  use bases_fcm
  use inbnd
  use psf
  use memory
  use stream
  use string
  use corman3,only:distn2,distn3 ! hwm 
    !
    integer,save,allocatable,dimension(:) :: HPOINT
    character(len=*) COMLYN
    INTEGER COMLEN
    LOGICAL LSEL2,LMIND,LMAXD
    real(chm_real) X(*),Y(*),Z(*),XCOMP(*),YCOMP(*),ZCOMP(*)
    INTEGER NATOMX
    INTEGER ISLCT(*),JSLCT(*)
    !
    !
    INTEGER IUNIT
    real(chm_real) CUT
    LOGICAL LNEAR
    LOGICAL QENER,QCLOSE,QNONB,QNONO,Q14EX,QNO14,QEXCL,QNOEX
    LOGICAL WRONG,QEX(3), QTRI, QDIFF, QRESI ! hwm 
    LOGICAL QHIST,QHPRIN,QHSAVE
    INTEGER,save  :: HNUM,HLEN=0,hdum(1)
    real(chm_real) :: HMIN,HMAX,HNORM,HDENS
    integer, dimension(1), target :: pzer1 = [0]
    integer, dimension(:), pointer :: tinb14, tiblo14
    !
    CALL GETBND(BNBND,.TRUE.)

    IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
    CUT=8999.D0
    CUT=GTRMF(COMLYN,COMLEN,'CUT',CUT)
    QENER=(INDXA(COMLYN,COMLEN,'ENER') > 0)
    QCLOSE=(INDXA(COMLYN,COMLEN,'CLOS') > 0)
    QNONB=(INDXA(COMLYN,COMLEN,'NONB') > 0)
    QNONO=(INDXA(COMLYN,COMLEN,'NONO') > 0)
    Q14EX=(INDXA(COMLYN,COMLEN,'14EX') > 0)
    QNO14=(INDXA(COMLYN,COMLEN,'NO14') > 0)
    QEXCL=(INDXA(COMLYN,COMLEN,'EXCL') > 0)
    QNOEX=(INDXA(COMLYN,COMLEN,'NOEX') > 0)
    QTRI =(INDXA(COMLYN,COMLEN,'TRIA') > 0)
    QDIFF=(INDXA(COMLYN,COMLEN,'DIFF') > 0)
    LNEAR=(INDXA(COMLYN,COMLEN,'LNEAR') > 0)
    QRESI=(INDXA(COMLYN,COMLEN,'RESI') > 0) ! hwm  
    !
    WRONG = (QNONB .AND. QNONO) .OR. (Q14EX .AND. QNO14) .OR. &
         (QEXCL .AND. QNOEX)
    IF(WRONG) CALL WRNDIE(0,'<DISTAN>','Conflicting options.')
    !
    IF(.NOT.(QNONB.OR.QNONO)) QNONB=.TRUE.
    IF(.NOT.(QEXCL.OR.QNOEX)) QEXCL=.FALSE.
    IF(.NOT.(Q14EX.OR.QNO14)) Q14EX=.FALSE.
    QEX(1)=QEXCL
    QEX(2)=Q14EX
    QEX(3)=QNONB
    !
    QHIST=(INDXA(COMLYN,COMLEN,'HIST') > 0)
    IF(QHIST) THEN
       HMIN=GTRMF(COMLYN,COMLEN,'HMIN',ZERO)
       HMAX=GTRMF(COMLYN,COMLEN,'HMAX',ZERO)
       HNUM=GTRMI(COMLYN,COMLEN,'HNUM',0)
       IF(HMAX <= HMIN .OR. HNUM <= 0) THEN
          CALL WRNDIE(0,'<DISTAN>', &
               'Error in parsing histogram keywords.')
          QHIST=.FALSE.
       ENDIF
    ENDIF
    !
    IF(QHIST) THEN
       QHSAVE=(INDXA(COMLYN,COMLEN,'HSAV') > 0)
       QHPRIN=(INDXA(COMLYN,COMLEN,'HPRI') > 0).AND.(PRNLEV > 2)
       HNORM=GTRMF(COMLYN,COMLEN,'HNOR',ONE)
       HDENS=GTRMF(COMLYN,COMLEN,'HDEN',ZERO)
       IF(HLEN == 0) THEN
          call chmalloc('corman.src','DISTAN','HPOINT',HNUM,intg=HPOINT)
          hpoint=0
          HLEN=HNUM
       ELSE
          IF(HLEN /= HNUM) THEN
             CALL WRNDIE(-3,'<CORMAN>', &
                  'Histogram does not have the same length as saved')
             call chmdealloc('corman.src','DISTAN','HPOINT',HLEN,intg=HPOINT)
             call chmalloc('corman.src','DISTAN','HPOINT',HNUM,intg=HPOINT)
             hpoint=0
             HLEN=HNUM
          ENDIF
       ENDIF

       CALL DISTN2(IUNIT,LSEL2,LMIND,LMAXD,LNEAR,X,Y,Z, &
            XCOMP,YCOMP,ZCOMP, &
            QETEN,QETSR,                               & 
            NATOMX,ISLCT,JSLCT,CUT,QENER,QCLOSE,QEX,QTRI,QDIFF, &
            NATOM,NNB14,BNBND%INB14, &
            BNBND%IBLO14,EPS,E14FAC, &
            QHIST,HMIN,HMAX,HNUM,HPOINT, &
            QHPRIN,HNORM,HDENS)
       !
       IF(.NOT.QHSAVE) THEN
          call chmdealloc('corman3.src','DISTAN','HPOINT',HNUM,intg=HPOINT)
          HLEN=0
       ENDIF
    ELSE
       IF (QRESI) THEN  !hwm
          CALL DISTN3(IUNIT,LSEL2,X,Y,Z,ISLCT,JSLCT,CUT)
       ELSE
          if (associated(bnbnd%iblo14) ) then
             tiblo14 => bnbnd%iblo14
             tinb14 => bnbnd%inb14
          else
             tiblo14 => pzer1
             tinb14 => pzer1
          endif
          CALL DISTN2(IUNIT,LSEL2,LMIND,LMAXD,LNEAR,X,Y,Z, &
               XCOMP,YCOMP,ZCOMP, &
               QETEN,QETSR,                              & 
               NATOMX,ISLCT,JSLCT,CUT,QENER,QCLOSE,QEX,QTRI,QDIFF, &
               NATOM,NNB14,tINB14, &
               tIBLO14,EPS,E14FAC, &
               .FALSE.,0._chm_real,0._chm_real,0,hdum,.false.,0._chm_real,0._chm_real)
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE DISTAN
end module corman_mod

