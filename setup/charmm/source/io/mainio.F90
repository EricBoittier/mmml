SUBROUTINE MAINIO(MODE)
  !
  !     THIS IS THE MAIN I/O ROUTINE IN CHARMM
  !
  use chm_kinds
  use chm_types
#if KEY_CHEQ==1
  use cheq,only:qcg             
#endif
  use intcor_module           
  use intcor2,only:writic
  use dimens_fcm
  use exfunc, only: lunass
  use number
#if KEY_ACE==1
  use ace_module,only:qfiace   
#endif
  use bases_fcm
  use cnst_fcm
  use code
  use comand
  use contrl
  use coord
  use coordc
  use ctitla
#if KEY_DIMS==1
  use dims 
#endif
#if KEY_EMAP==1
  use emapmod       
#endif
  use energym
  use hbondm
  use image
  use image_routines_module, only: climag
  use inbnd
  use param
  use pert
  use psf
  use genpsf_m, only: atmini
  use rtf,only:atct, natct, nrtrs, nic, rtfrdr, rtwrit
  use machio,only:vopen
  use stream
  use string
  use memory
  use io
  use parmiom
  use parallel
  use tmd
  use surface
  use surfmemb
  use parallel  ! mh050712
  use repdstr   ! mh050712
  use consph    
  use univ
  use cstran_mod,only:wrcnst,rdcnst,prcnst
  use coorio_mod,only:coorio 
#if KEY_DIMS==1
  use dims, only: readnm   
#endif
#if KEY_QCHEM==1
  use gamess_fcm,only:qqchem
#endif 
  use resdist,only:redwri
  use bond_tables,only:read_bond_tables,read_angle_tables,&
       read_dihedral_tables
  !---   use nbutil_module,only:prnbnd,gtnbct
  use machio,only:vopen

  use param_store, only: set_param
  use cstuff, only: readnamd, writenamd
#ifdef KEY_RESIZE
  use resize
#endif  

  implicit none
  !
  CHARACTER(len=4) MODE
  !
  !
  !     local variables.
  CHARACTER(len=1) CHAIN
  CHARACTER(len=4)  WRD,WINIT
  character(len=8)  wrd8,SEGIDX 
  CHARACTER(len=12) FORM
  integer, parameter :: FNMAX = 128
  CHARACTER(len=FNMAX) :: FNAME, CMPD*20
  LOGICAL LAPPE,LCOMP,ERROR,QPRINT,LCARD,QERROR,QIMOPN,QUSED
  LOGICAL LATOM,LHETATM,LSEQRES
  INTEGER :: I,ICARD,IXX,ICYCLE=0,IPSF
  INTEGER   J,LEN,NINPUT,NCHAIN
  INTEGER   NSGIML,NRSIML,NDUMMY
  INTEGER   IPRINT,ISTART,IUNIT,IDUMMY,NATIML,FLEN,CLEN
  INTEGER IER,FLEN1,FLEN2
  INTEGER ::  ICNTRL(20)=(20*0)
  INTEGER ::  NSKIP,NALI,IFIRST
  INTEGER, parameter :: MAXALI=200,MAXSKIP=200
  CHARACTER(len=8) SKP,SKIP(MAXSKIP),ALI,ALIAS(2,MAXALI)
  !
  !     begin
  !=======================================================================
  IPRINT = 0
  QIMOPN = .FALSE.
  NSKIP = 0
  NALI =0

  io_mode: select case(mode)
  case('READ') io_mode
     !=======================================================================
     !       Begin Procedure PROCESS-READ-COMMAND
     !       . Check for a HELP request.
     IF (INDXA(COMLYN, COMLEN, 'HELP')  >  0) THEN
        CALL HELP('IO_READ')
        RETURN
     ENDIF
     !       READ COMMAND, BEFORE DISPATCHING, GET UNIT AND/OR NAME
     IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
     CALL GTRMWD(COMLYN,COMLEN,'NAME',4,FNAME,FNMAX,FLEN)

     ! Lennart start
     QIMOPN=(FLEN > 0)
     IF(QIMOPN .AND. (IOLEV  >  0) ) THEN
        !         A filename was specified. Now open it properly.
        IF(IUNIT  ==  -1)  IUNIT=LUNASS(90)
        ! Lennart end
        !         Parse basic I/O specifications
        IF (INDX(COMLYN,COMLEN,'FILE',4) > 0) THEN
           FORM='UNFORMATTED'
        ELSE
           FORM='FORMATTED'
        ENDIF
        IF(INDXA(COMLYN,COMLEN,'UNFO') > 0) FORM='UNFORMATTED'
        CALL VOPEN(IUNIT,FNAME,FORM,'READ',QERROR,0)
        IF(QERROR) CALL WRNDIE(0,'<MAINIO>', &
             '"READ...NAME...: OPEN" not possible.')
     ENDIF
     IF(IOLEV  <  0 .AND. IUNIT  ==  -1) IUNIT=ISTRM


     WRD=NEXTA4(COMLYN,COMLEN)
     read_mode: select case(wrd)
        !=======================================================================
     case('CONS') read_mode
        !=======================================================================
        !         Begin Procedure READ-CONSTRAINT-FILE
        !
        IF (IUNIT == -1) IUNIT=ISTRM
        IXX=INDXA(COMLYN,COMLEN,'CARD')
        IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
             ' MAINIO> Constraints being read from unit ', IUNIT, '.'
        IPSF=GTRMI(COMLYN,COMLEN,'PSF',1)
        IF(IPSF /= 1) CALL WRNDIE(0,'<MAINIO>', &
             'PSF option no longer supported on READ CONS')
        CALL RDCNST(IUNIT,TITLEB,NTITLB, &
             NCSPHI,ICS,JCS,KCS,LCS,ICCS, &
             CCSC,CCSB,CCSD,CCSW,CCSCOS,CCSSIN)
        !         End Procedure READ-CONSTRAINT-FILE
        !=======================================================================
     case('CPH') read_mode
        !=======================================================================
        !         Begin Procedure READ-CONSPH-FILE
        IF(INDXA(COMLYN,COMLEN,'CARD') < 0 .and. INDXA(COMLYN,COMLEN,'FORM') < 0) &
             CALL WRNDIE(-3,'<MAINIO>','CONST PH PARAM FILES MUST BE READ FORMATTED')
        CALL PARSEPHPRM(IUNIT)
     case('COOR') read_mode
        !=======================================================================
        !         Begin Procedure READ-COORDINATE-FILE
        !

        IF (INDXA(COMLYN,COMLEN,'COMP') > 0) THEN
#if KEY_COMP2==1 /*comp2_1*/
           IF(INDXA(COMLYN,COMLEN,'SECO') > 0) THEN
              CALL COORIO(-1,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
                   ICNTRL,NATOM,XCOMP2,YCOMP2,ZCOMP2,WCOMP2, &
                   ATYPE,RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
           ELSE 
#endif /* (comp2_1) */
              CALL COORIO(-1,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
                   ICNTRL,NATOM,XCOMP,YCOMP,ZCOMP,WCOMP,ATYPE, &
                   RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
#if KEY_COMP2==1
           ENDIF 
#endif
#if KEY_TMD==1 /*t1*/
        else if(indx(comlyn,comlen,'TARG',4) > 0) then
           if(qtmd)then
              call coorio(-1,iunit,comlyn,comlen,titleb,ntitlb, &
                   icntrl,natom,ixtar,iytar,iztar,itrcm,atype, &
                   resid,res,nres,ibase,segid,nictot,nseg,.false.)
           else
              call wrndie(-3,'<mainio>', &
                   ' use tmdi before reading target structure')
           endif
           ! msf tmd \./
        else if(indxa(comlyn,comlen,'tar2') > 0) then
           if(qtmd)then
              call coorio(-1,iunit,comlyn,comlen,titleb,ntitlb, &
                   icntrl,natom,ixtar2,iytar2,iztar2,itrcm,atype, &
                   resid,res,nres,ibase,segid,nictot,nseg,.false.)
           else
              call wrndie(-3,'<MAINIO>', &
                   ' USE TMDI BEFORE READING TARGET STRUCTURE')
           endif
           ! MSF End
#endif /* (t1)*/
#if KEY_DIMS==1 /*d1*/
        ELSE IF(INDXA(COMLYN,COMLEN,'DIMS') > 0) THEN
           CALL COORIO(-1,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
                ICNTRL,NATOM,DIMTARGXA,DIMTARGYA,DIMTARGZA, &
                WDIMS,ATYPE, &
                RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)

#endif /* (d1)*/
        ELSE
           CALL COORIO(-1,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
                ICNTRL,NATOM,X,Y,Z,WMAIN,ATYPE, &
                RESID,RES,NRES,IBASE,SEGID,NICTOT,NSEG,.FALSE.)
        ENDIF
        !         End Procedure READ-COORDINATE-FILE
        !=======================================================================
     case('HBON') read_mode
        !=======================================================================
        !         Begin Procedure READ-HBOND-FILE
        !
        IF (INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
           LCARD=.TRUE.
           IF (IUNIT == -1) IUNIT=ISTRM
        ELSE
           IXX=INDXA(COMLYN,COMLEN,'FILE')
           LCARD=.FALSE.
        ENDIF
        IF (IUNIT == -1)  GOTO 9000
        IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
             ' MAINIO> Hydrogen bonds being read from unit ', IUNIT, '.'
        CALL HBREAD(IUNIT,NATOM,TITLEB,NTITLB,ICNTRL, &
             NHB,MAXHB,IHB,JHB,KHB,LHB,LCARD, &
             SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
        CALL HCODES(ICH,NHB,IHB,JHB,IAC)
        !         End Procedure READ-HBOND-FILE

#if KEY_MMFF==1 /*m1*/
     case('MOL ', 'MERC', 'MOL2')   read_mode
        IF (IUNIT == -1) IUNIT=ISTRM
        CMPD = 'XXXX'
        call mmff_io(MODE,WRD,IUNIT,CMPD)

     case('DB  ') read_mode
        cmpd = next20(COMLYN,COMLEN)
        IF (IUNIT == -1) IUNIT=ISTRM
        call mmff_io(MODE,WRD,IUNIT,CMPD)

#endif /* (m1)*/
        !=======================================================================
     case('IC  ') read_mode
        !=======================================================================
        !         Begin Procedure READ-INTERNAL-COORDINATE-FILE
        !
        ICARD=1
        IF(INDXA(COMLYN,COMLEN,'FILE') > 0) ICARD=0
        IF(INDXA(COMLYN,COMLEN,'CARD') > 0) ICARD=1
        ISTART=1
        IF (IUNIT == -1) IUNIT=ISTRM
        IF(INDXA(COMLYN,COMLEN,'SAVE') > 0) THEN
           IF(INDXA(COMLYN,COMLEN,'APPE') > 0) ISTART=ics_struct%LENIC+1
           CALL READIC(ISTART,ICARD,IUNIT,ics_struct)
        ELSE
           IF(INDXA(COMLYN,COMLEN,'APPE') > 0) ISTART=icr_struct%LENIC+1
           CALL READIC(ISTART,ICARD,IUNIT,icr_struct)
        ENDIF
        CALL set_param('NIC',icr_struct%LENIC)
        !         End Procedure READ-INTERNAL-COORDINATE-FILE

        !=======================================================================
     case('IMAG') read_mode
        !=======================================================================
        !         Begin Procedure READ-IMAGES-TRANSFORMATION-FILE
        !
        IF (IUNIT == -1) IUNIT=ISTRM
        QPRINT=INDXA(COMLYN,COMLEN,'PRIN') > 0
        CALL IMREAD(COMLYN,COMLEN,IUNIT,QPRINT)
        !         setup default value for IMGFRQ if it has not been set
        IF(IMGFRQ <= 0) IMGFRQ=50
        CALL set_param('NTRA',NTRANS)
#ifdef KEY_RESIZE
        call set_rszf('mainio.F90','MAINIO',.true.)
#endif  
        !         End Procedure READ-IMAGES-TRANSFORMATION-FILE

#if KEY_SHAPES==1 /*s1*/
        !=======================================================================
     case('MDL ') read_mode
        !=======================================================================
        IF (IUNIT == -1)  GOTO 9000
        call rdmdl(iunit, comlyn, comlen)
#endif /* (s1)*/

        !=======================================================================
     case('NBON') read_mode
        !=======================================================================
        !         Begin Procedure READ-NONBONDED-FILE
        !
        IF (IUNIT == -1)  GOTO 9000
        IF(PRNLEV >= 2) WRITE(OUTU,550) IUNIT
550     FORMAT(/10X,'NON BONDED LIST BEING RED FROM UNIT',I3)
!!!        CALL READDT(BNBND,LNBND,SNBND,IUNIT,'NBND')
        IXX=INDXA(COMLYN,COMLEN,'FILE')
        !         End Procedure READ-NONBONDED-FILE

#if KEY_DIMS==1 /*d2*/
        !=======================================================================
     case('NM  ') read_mode
        !=======================================================================
        !     DIMS SELF AVOIDANCE NM
        IF(QDIMS) then
           CALL READNM(NMVEC,NMMATRIX,MTRAJ,IUNIT, &
                NTITLB,TITLEB,NMUNIT)
        ELSE
           CALL WRNDIE(3,'<MAINIO>', 'DIMS not initialized  &
                unable to read  NMs')
        ENDIF

#endif /* (d2)*/
        !=======================================================================
     case('PARA') read_mode
        !=======================================================================
        !         Begin Procedure READ-PARAMETER-FILE
        !
        if (indxa(comlyn,comlen,'CARD') > 0) then
           icard=1
           if (iunit == -1) iunit=istrm
           if (indxa(comlyn,comlen,'PRIN') > 0) then
              icard=2
              if (indxa(comlyn,comlen,'NBON') > 0) icard=3
           endif
        else
           if(indxa(comlyn,comlen,'FILE') > 0 )then
              call wrndie(-5,'<mainio.src>mainio', &
                   'Binary parameter files not supported in this or future CHARMMM versions.')
           endif
        endif
        lappe=(indxa(comlyn,comlen,'APPE') > 0)
        if(prnlev >= 2) write(outu,"(/10X,'PARAMETER FILE BEING READ FROM UNIT',I3)") iunit
        call parmio(iunit,ntitlb,titleb,icard,outu,natct,atct,lappe)

        !         now reset major data structures upon parameter file reading
        winit='INIT'
        j=4
        call gtnbct(winit,j,bnbnd)
        call climag(bimag)
        winit='INIT'
        j=4
        call gthbct(winit,j)
        mustup=.true.
        if(prnlev > 2) write (outu,'(a)') &
             ' PARMIO> NONBOND, HBOND lists and IMAGE atoms cleared.'

        !         End Procedure READ-PARAMETER-FILE
        !=======================================================================

#if KEY_ACE==1 /*a1*/
     case('ACEP') read_mode
        !=======================================================================
        !         Begin Procedure READ-ACE-PARAMETERS
        !
        IF (INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
           ICARD=1
           IF (IUNIT == -1) IUNIT=ISTRM
        ELSE
           IXX=INDXA(COMLYN,COMLEN,'FILE')
           IF (IUNIT == -1)  GOTO 9000
           ICARD=0
           CALL WRNDIE(-2,'<MAINIO>', &
                'Binary read of ACE parameters not yet implemented.')
        ENDIF
        IF(PRNLEV >= 2) WRITE(OUTU,201) IUNIT
201     FORMAT(/10X,'ACE PARAMETERS BEING READ FROM UNIT',I3)
        CALL ACEIO(IUNIT,NTITLB,TITLEB,ICARD,OUTU)
        !         make sure the routines ACEINI and IESFIX are called
        !         later on, after reading of new volumes:
        QFIACE=0
        !
        !         End Procedure READ-ACE-PARAMETERS
#endif /* (a1)*/

        !=======================================================================
     case('PSF ') read_mode
        !=======================================================================
        !         Begin Procedure READ-STRUCTURE-FILE
        !
        ! LNI mod August 2009, allow READ PSF CARD to read from current stream.
        !
        IF (IUNIT == -1) IUNIT=ISTRM

        LAPPE  = (INDXA(COMLYN,COMLEN,'APPE') > 0)
        IF(LAPPE) THEN
           J=NATOM+1
           IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
              ' MAINIO> Protein structure file being appended from unit ',IUNIT, '.'
        ELSE
           J=1
           IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
              ' MAINIO> Protein structure file being read from unit ',IUNIT, '.'
        ENDIF

        IF (INDXA(COMLYN,COMLEN,'FILE') > 0) THEN
          call wrndie(-5,'<mainio.src>','Only CARD format is now supported for PSF.')
        ENDIF
        ICARD=INDXA(COMLYN,COMLEN,'CARD')

        CALL PSF_read_formatted(IUNIT,TITLEB,NTITLB,LAPPE)
        
        CALL ATMINI(J,NATOM)
        !
        CALL PSFSUM(OUTU)
        !         End Procedure READ-STRUCTURE-FILE
        !=======================================================================

     case('RTF ') read_mode
        !=======================================================================
        !         Begin Procedure READ-TOPOLOGY-FILE
        !
        if (indxa(comlyn,comlen,'CARD') > 0) then
           icard=1
           if (iunit == -1) iunit=istrm
           qprint=indxa(comlyn,comlen,'PRIN') > 0
        else
           IXX=indxa(comlyn,comlen,'FILE')
           icard=0
        endif
        if(prnlev >= 2) write (outu, '(a,i3,a)') &
             ' MAINIO> Residue topology file being read from unit ', &
             iunit, '.'
        lappe=(indxa(comlyn,comlen,'APPE') > 0)
        call rtfrdr(iunit,titleb,ntitlb,icard,qprint,lappe)

        !         End Procedure READ-TOPOLOGY-FILE
        ! 
        ! ===========================================================
#if KEY_NOMISC==0 /*m2*/
        !=======================================================================
     case('SBOU') read_mode
        !=======================================================================
        CALL SBREAD
#endif /* (m2)*/

#if KEY_ASPENER==1 /*a3*/
        !=======================================================================
     case('SURF') read_mode
        ! check to se if surface arrays are allocated, if not alllocate.
        if( (.not.allocated(aspv)) .or. &
             (allocated(aspv) .and. size(aspv)/=natom) ) then
           call allocate_surface(natom)
        endif
        !=======================================================================
#if KEY_ASPMEMB==1 /*a4*/
        SOLVMEMB_ENTRD=.FALSE.        
#endif /* (a4)*/
        CALL SURFACE_IO(IUNIT)


#if KEY_ASPMEMB==1 /*a5*/
        !=======================================================================
     case('SAIM') read_mode
        ! check to se if surface arrays are allocated, if not alllocate.
        if( (.not.allocated(aspv)) .or. &
             (allocated(aspv) .and. size(aspv)/=natom) ) then
           call allocate_surface(natom)
        endif
        !=======================================================================
        SOLVENT_ENTERED=.FALSE.
        CALL SURFMEMB_IO(IUNIT)
#endif /* (a5)*/
#endif /*  (a3)*/
        !=======================================================================


     case('SEQU') read_mode
        !=======================================================================
        !         Begin Procedure READ-SEQUENCE-FILE
        !         FIRST, DEFINE TYPE OF SEGMENT
        !
#if KEY_MMFF==1 /*m3*/
        DATA_READ=SEQUENCE
#endif /* (m3)*/
        !
        ISTART=NICTOT(NSEG+1)
        WRD8=NEXTA8(COMLYN,COMLEN)
        IF (WRD8(1:4) == 'CARD') THEN
           NINPUT=0
           IF (IUNIT == -1) IUNIT=ISTRM
        ELSE IF (WRD8(1:4) == 'COOR') THEN
           NINPUT=2
           IF(INDXA(COMLYN,COMLEN,'RESI') > 0) NINPUT=3
           SEGIDX=GTRMA(COMLYN,COMLEN,'SEGI')
           IF(INDXA(COMLYN,COMLEN,'PDB') > 0) THEN
              NINPUT=4
              CHAIN=GTRMA(COMLYN,COMLEN,'CHAI')
              IF(CHAIN /= ' ' .AND. SEGIDX /= ' ') &
                 CALL WRNDIE(-1,'MAINIO','Specify at most one of CHAIN and SEGI')
           ENDIF
        ELSE IF (WRD8(1:4) == 'PDB') THEN
           NINPUT=4
           CHAIN=GTRMA(COMLYN,COMLEN,'CHAI')
           LATOM=(INDXA(COMLYN,COMLEN,'NOAT') == 0)
           LHETATM=(INDXA(COMLYN,COMLEN,'HETA') > 0)
           LSEQRES=(INDXA(COMLYN,COMLEN,'SEQR') > 0)
           IFIRST=GTRMI(COMLYN,COMLEN,'FIRS',1)
           SEGIDX=GTRMA(COMLYN,COMLEN,'SEGI')
           NCHAIN=GTRMI(COMLYN,COMLEN,'NCHA',0)
           IF(CHAIN(1:1) /= ' ' .AND. SEGIDX(1:1) /= ' ') &
              CALL WRNDIE(-1,'MAINIO','Specify at most one of CHAIN and SEGI')
           IF( NCHAIN > 0 .AND. (CHAIN(1:1) /= ' ' .OR. SEGIDX(1:1) /= ' ')) &
              CALL WRNDIE(-1,'MAINIO','Specify at most one of NCHAIN, CHAIN and SEGI') 
           ! Residue names to skip
           SKP=GTRMA(COMLYN,COMLEN,'SKIP')
           NSKIP=0
           DO WHILE (SKP/=' ')  
             IF(NSKIP == MAXSKIP) &
                CALL WRNDIE(-3,'<MAINIO>','SKIP table overflow')
             NSKIP=NSKIP+1
             SKIP(NSKIP)=SKP
             SKP=GTRMA(COMLYN,COMLEN,'SKIP')
           ENDDO
           IF(NSKIP>0 .AND. PRNLEV>2) WRITE(OUTU,'(A,I5)') 'Number of skipped resnames:',NSKIP

           ! Setup aliases
           ALI=GTRMA(COMLYN,COMLEN,'ALIA')
           NALI=0
           DO WHILE (ALI/=' ')  
             IF(NALI == MAXALI) &
                CALL WRNDIE(-3,'<MAINIO>','ALIAS table overflow')
              NALI=NALI+1
              ALIAS(1,NALI)=ALI
              ALIAS(2,NALI)=NEXTA8(COMLYN,COMLEN)
              IF(ALIAS(2,NALI)==' ') &
                 CALL WRNDIE(-2,'<MAINIO>','ALIAS need two residue names')
               ALI=GTRMA(COMLYN,COMLEN,'ALIA')
            ENDDO
            IF(NALI>0 .AND. PRNLEV>2) WRITE(OUTU,'(A,I5)') 'Number of alias pairs:',NALI
        ELSE
           NINPUT=-1
           CALL TRIMA(COMLYN,COMLEN)
           IF (COMLEN == 0) THEN
              CALL WRNDIE(0,'<MAINIO>', &
                   'Unrecognized read sequence mode:'//WRD)
           ELSE
              NDUMMY=NEXTI(COMLYN,COMLEN)
              !             For now, preserve old use of TIPS keyword for TIP3 residue name.
              IF(WRD8(1:4) == 'WATE') WRD8='OH2'
              IF(WRD8(1:4) == 'TIPS') WRD8='TIP3'
              !
#ifdef KEY_RESIZE
              call resize_psf('mainio.F90','MAINIO','NRES',NRES+NDUMMY, .true.)
#endif
              DO I=1,NDUMMY
                 NRES=NRES+1
                 CALL ENCODI(NRES-ISTART,RESID(NRES),8,IDUMMY)
                 RES(NRES)=WRD8
              ENDDO
           ENDIF
        ENDIF
        !
        IF(NINPUT >= 0) THEN
           CALL XTRANE(COMLYN,COMLEN,'CHARMM')
           IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
                ' MAINIO> Sequence information being read from unit ', &
                IUNIT, '.'
#ifdef KEY_RESIZE
           ! No global variable in arguments
           CALL SEQRDR(COMLYN,COMLEN,IUNIT,TITLEB,NTITLB,MAXTIT, &
                NINPUT,ISTART,CHAIN,SEGIDX,NCHAIN,NSKIP,SKIP,NALI,ALIAS, &
                LATOM,LHETATM,LSEQRES,IFIRST)
#else
           ! Global variables included as arguments (original)
           CALL SEQRDR(COMLYN,COMLEN,MXCMSZ,IUNIT,TITLEB,NTITLB,MAXTIT, &
                RES,NRES,RESID,NINPUT,ISTART,CHAIN,SEGIDX,NCHAIN,NSKIP,SKIP,NALI,ALIAS, &
                LATOM,LHETATM,LSEQRES,IFIRST)
#endif
           IF (reallow) THEN     
              IF(IOLEV > 0 .AND. NINPUT == 1) REWIND IUNIT
           ENDIF                 
        ENDIF
        !         End Procedure READ-SEQUENCE-FILE
        !=======================================================================

     case('TABL') read_mode
        !=======================================================================
        !         Begin Procedure READ-TABLE-FILE
        !
#if KEY_NOMISC==0
        IF (IUNIT == -1)  GOTO 9000
        IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
             ' MAINIO> Non-bond energy tables being read from unit ', &
             IUNIT, '.'
        CALL reatbl(IUNIT,NATC)
#else /**/
        CALL WRNDIE(-1,'<MAINIO>','nonbonded TABLE code is not compiled.')
#endif 

     case('BTAB') read_mode
        !=======================================================================
        !         Begin Procedure READ Bond TABLEs
        !
#if KEY_IF==1 || KEY_PARALLEL==1
        
#endif
        !        CALL WRNDIE(-1,'<MAINIO>','bond energy table code is not ready for parallel.')
#if KEY_ENDIF==1
        
#endif
#if KEY_NOMISC==0
        IF (IUNIT == -1)  GOTO 9000
        IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
             ' MAINIO> bond energy tables being read from unit ', &
             IUNIT, '.'
        CALL read_bond_tables(IUNIT,NATC)
        CALL read_angle_tables(IUNIT)
        CALL read_dihedral_tables(IUNIT)
        if (iolev > 0) IXX=INDXA(COMLYN,COMLEN,'FILE')
#else /**/
        CALL WRNDIE(-1,'<MAINIO>','ETABLE code is not compiled.')
#endif 
        !         End Procedure READ-TABLE-FILE
        !aag 06/07
        !=======================================================================
     case('NAMD') read_mode
        !=======================================================================
        !         Begin Procedure READ-NAMD binary coordinate file 
        !         extract the name of the file
        FLEN = 0
        CALL GTRMWD(COMLYN,COMLEN,'FILE',4,FNAME,FNMAX,FLEN)
        IF (FLEN  <=  0) THEN
           CALL WRNDIE(-5,'<MAINIO>','Specify file name')
        ENDIF
        IF (IPRINT > 0) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,'(A,A)') &
                'Reading as namd binary file',FNAME
        ENDIF
        !          file name should be in quotes to avoid case conversion
        IF(FNAME(1:1) /= '"' .or. FNAME(FLEN:FLEN) /= '"') &
             CALL WRNDIE(-5,'<MAINIO>', &
             'Enclose file name in double quotes "..." ')
        FLEN1 = FLEN-1
        FLEN2 = FLEN-2
        IF(IOLEV > 0 )THEN
           CALL READNAMD(X,Y,Z,NATOM,FNAME(2:FLEN1),FLEN2,IER)
           IF (IER  /=  0) &
                CALL WRNDIE(-5,'<MAINIO>','Error reading binary file')
        ENDIF
#if KEY_PARALLEL==1
        CALL PSND8(X, NATOM)
        CALL PSND8(Y, NATOM)
        CALL PSND8(Z, NATOM)
#endif /*    */
        RETURN
        !         End Procedure   READ-NAMD binary coordinate file 
        !!aag
        !=======================================================================

     case('UNIV') read_mode
        !=======================================================================
        !         Begin Procedure READ-UNIVERSAL-IO-FORMAT
        IF (IUNIT == -1) IUNIT=ISTRM
        CALL RUNIVF(IUNIT)
        !         End Procedure READ-UNIVERSAL-IO-FORMAT

#if KEY_NOMISC==0
     case('XRAY') read_mode
        !         Begin Procedure READ-XRAY-GRAPHICS-FILE
        CALL RDXRAY(IUNIT,COMLYN,COMLEN)
        CALL PSFSUM(OUTU)
        !         End Procedure READ-XRAY-GRAPHICS-FILE
#endif 

#if KEY_EMAP==1
        !WXW=======================================================================
     case('SEGI') read_mode
        !WXW=======================================================================
        !         Begin Procedure READ-PDB-CREAT-SEGMENT
        IF (IUNIT == -1) IUNIT=ISTRM
        CALL RDSEGID(IUNIT,COMLYN,COMLEN)
        !WXW       End Procedure READ-PDB-CREAT-SEGMENT
#endif 

     case default read_mode
        CALL WRNDIE(0,'<MAINIO>','Unrecognized command.:'//WRD)
     end select read_mode
     !       End Procedure PROCESS-READ-COMMAND
     !=======================================================================



  case('WRIT', 'PRIN') io_mode
     !=======================================================================
     !       Begin Procedure PROCESS-WRITE-OR-PRINT-COMMANDS
     !       . Check for a HELP request.
     if (indxa(comlyn, comlen, 'HELP') > 0) then
        if (mode  ==  'PRIN') then
           call help('IO_PRINT')
        else
           call help('IO_WRITE')
        endif
        return
     endif
     !
     !RCZ 92/08/17 - check if this is 'print crystal' or 'write crystal' comm
     !       if yes then call crystal module to process this command
     !       if not restore command line
     !
     wrd=nexta4(comlyn,comlen)
     out_mode: select case(wrd)
     case('CRYS') out_mode
        call joinwd(comlyn,mxcmsz,comlen,mode,4)
        call crystl
        return

     case('VIRI') out_mode
        if (iolev > 0) then
           write(OUTU,*) "Internal Virial:"
           write(OUTU,'(3F20.10)') EPRESS(VIXX), EPRESS(VIXY), EPRESS(VIXZ)
           write(OUTU,'(3F20.10)') EPRESS(VIYX), EPRESS(VIYY), EPRESS(VIYZ)
           write(OUTU,'(3F20.10)') EPRESS(VIZX), EPRESS(VIZY), EPRESS(VIZZ)
           write(OUTU,*)
           write(OUTU,*) "External Virial:"
           write(OUTU,'(3F20.10)') EPRESS(VEXX), EPRESS(VEXY), EPRESS(VEXZ)
           write(OUTU,'(3F20.10)') EPRESS(VEYX), EPRESS(VEYY), EPRESS(VEYZ)
           write(OUTU,'(3F20.10)') EPRESS(VEZX), EPRESS(VEZY), EPRESS(VEZZ)
        endif
        return

        !=======================================================================
     case('NAMD')  out_mode
        !=======================================================================
        !         Begin Procedure WRITE-NAMD binary coordinate file 
        !         extract the name of the file
        flen = 0
        call gtrmwd(comlyn,comlen,'FILE',4,fname,FNMAX,flen)
        if (flen  <=  0) then
           call wrndie(-5,'<mainio>','specify file name')
        endif
        if (iprint > 0) then
           if(prnlev >= 2) write(outu,'(a,a)') &
                'writing as namd binary file',fname
        endif
        !     file name should be in quotes to avoid case conversion
        if(fname(1:1) /= '"' .or. fname(flen:flen) /= '"') &
             call wrndie(-5,'<MAINIO>', &
             'enclose file name in double quotes "..." ')
        flen1 = flen-1
        flen2 = flen-2
        if(iolev > 0)then
           call writenamd(x,y,z,natom,fname(2:flen1),flen2,ier)
           if (ier /= 0) &
                call wrndie(-5,'<MAINIO>','error writing binary file')
        endif
        return
        !     End Procedure   WRITE-NAMD binary coordinate file 
     case default out_mode
        call joinwd(comlyn,mxcmsz,comlen,wrd,4)
     end select out_mode

     !=======================================================================
     write_mode: if (mode  ==  'WRIT') then
        !=======================================================================

        iunit=gtrmi(comlyn,comlen,'UNIT',-1)
        call gtrmwd(comlyn,comlen,'NAME',4,fname,FNMAX,flen)
        qimopn=(flen > 0)
        if(qimopn .and. iolev > 0) then
           !           A filename was specified. Now open it properly.
           if(iunit  ==  -1) iunit=lunass(90)
           !           Parse basic I/O specifications
           if (indx(comlyn,comlen,'FILE',4) > 0) then
              form='UNFORMATTED'
           else
              form='FORMATTED'
           endif
           if(indxa(comlyn,comlen,'UNFO') > 0) form='UNFORMATTED'
           if (indxa(comlyn,comlen,'APPE') > 0) then
              call vopen(iunit,fname,form,'APPEND',qerror,0)
           else
              call vopen(iunit,fname,form,'WRITE',qerror,0)
           endif
           if(qerror) call wrndie(0,'<MAINIO>', &
                '"write...name...: open" not possible.')
        endif
        if (iunit == -1 .and. .not. qimopn)  goto 9000
        call rdtitl(titlea,ntitla,istrm,0)
        iprint=0
#if KEY_STRINGM==0 /*  VO stringm : allow each string root to write file */
        if(iolev < 0) return
#endif
     else write_mode
        if(prnlev < 2) return
        iunit=outu
        iprint=1
     endif write_mode

     wrd=nexta4(comlyn,comlen)
     !=======================================================================
     !       SELECT for output keyword
     output_key: select case(wrd)
     case('CONS') output_key
        call output_constraints

     case('COOR')  output_key
        call output_coor

     case('ENER')  output_key
        !=======================================================================
        !         Begin Procedure WRITE-ENERGY-VALUES
        if(iunit /= outu) call wrtitl(titlea,ntitla,iunit,1)
        icycle=icycle+1
        call printe(iunit ,eprop, eterm, 'PRIN', 'ENR', .TRUE., &
             icycle, zero, zero, .true.)
        !         End Procedure WRITE-ENERGY-VALUES
        !=======================================================================

     case('HBON')  output_key
        call output_hbon

     case('IC  ') output_key
        call output_ic

     case('IMAG') output_key
        !=======================================================================
        !         Begin Procedure WRITE-IMAGES-TRANSFORMATION-FILE
        !
        CALL IMWRIT(IUNIT,COMLYN,COMLEN,BIMAG,IPRINT)
        !         End Procedure WRITE-IMAGES-TRANSFORMATION-FILE
        !=======================================================================
#if KEY_SHAPES==1
     case('MDL ')  output_key
        !=======================================================================
        call wrmdl(iunit, comlyn, comlen)
#endif 

#if KEY_MMFF==1
     case('MOL ', 'MERC', 'MOL2') output_key
        call mmff_io(MODE,WRD,IUNIT,CMPD)
#endif 

     case('NBON') output_key
        call output_nonbonded

     case('PARA') output_key
        !=======================================================================
        !         Begin Procedure WRITE-PARAMETER-FILE
        !
        if(indxa(comlyn,comlen,'FILE') > 0)then
           call wrndie(-5,'<MAINIO>', &
                'Binary parameter files not supported in this or future CHARMM versions')
        endif
        if(indxa(comlyn,comlen,'CARD') > 0) iprint=1
        qused=(indxa(comlyn,comlen,'USED') > 0)
#if KEY_QCHEM==1
        qqchem=(indxa(comlyn,comlen,'QCHEM') > 0)
#endif 
        call parwtr(iunit,ntitla,titlea,icntrl,iprint,qused)
        if(iprint == 0) call vclose(iunit,'KEEP',error)
        !         End Procedure WRITE-PARAMETER-FILE
        !=======================================================================

     case('PSF ') output_key
        call output_psf

     case('RTF ') output_key
        call output_rtf

     case('TABL') output_key
        call output_table_file

     case ('RESD') output_key
        call output_resd

     case('TITL') output_key
        !=======================================================================
        call wrtitl(titlea,ntitla,iunit,3)

#if KEY_NOMISC==0
     case('XRAY')  output_key
        call output_xray_graphics
#endif 

#if KEY_GAMESS==1
        !     Control GAMESS I/O
        !     Commands available: PUNCH, DICTNRY, DASORT
     case('PUNC') output_key
     case('DICT') output_key
     case('DASO') output_key
#endif 

     case default output_key
        call wrndie(0,'<MAINIO>','Unrecognized write command:'//WRD)
     end select output_key
     !       End Procedure PROCESS-WRITE-OR-PRINT-COMMANDS


  case default io_mode 
     CALL WRNDIE(-5,'<MAINIO>',' Unknown MODE')
  end select io_mode

  ! If a file was opened directly on the i/o commandline we also close it now
  if (qimopn) then
     if (iunit /= outu) call vclose(iunit,'KEEP',ERROR)
  endif
  return
9000 continue
  !     Begin Procedure NO-FILE-WAS-SPECIFIED
  CALL WRNDIE(0,'<MAINIO>','No unit specified. nothing done.')
  return

contains

  subroutine output_coor
    !=======================================================================
    !         Begin Procedure WRITE-COORDINATE-FILE
    !
    NATIML=NATOM
    NRSIML=NRES
    NSGIML=NSEG
    IF(INDXA(COMLYN,COMLEN,'IMAG') > 0) THEN
       NATIML=NATOMT
       NRSIML=NREST
       NSGIML=NSEGT
    ENDIF
    ! SAPATEL
#if KEY_CHEQ==1
    J=0
    IF (QCG) J=1
    QCG=(INDXA(COMLYN,COMLEN,'CHEQ') > 0)
#endif 
    ! SAPATEL
    IF (INDXA(COMLYN,COMLEN,'COMP') > 0) THEN
#if KEY_COMP2==1 /*comp2_2*/
       ! second comparison coordinate set
       IF(INDXA(COMLYN,COMLEN,'SECO') > 0) THEN
          CALL COORIO(IPRINT,IUNIT,COMLYN,COMLEN,TITLEA,NTITLA, &
               ICNTRL,NATIML,XCOMP2,YCOMP2,ZCOMP2,WCOMP2, &
               ATYPE,RESID,RES,NRSIML,IBASE,SEGID,NICTOT,NSGIML,.FALSE.)
       ELSE
#endif /* (comp2_2)*/
          CALL COORIO(IPRINT,IUNIT,COMLYN,COMLEN,TITLEA,NTITLA, &
               ICNTRL,NATIML,XCOMP,YCOMP,ZCOMP,WCOMP,ATYPE, &
               RESID,RES,NRSIML,IBASE,SEGID,NICTOT,NSGIML,.FALSE.)
#if KEY_COMP2==1
       ENDIF 
#endif
#if KEY_TMD==1
    ELSE IF(INDXA(COMLYN,COMLEN,'TARG') > 0) THEN
       if(qtmd)then
          CALL COORIO(IPRINT,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
               ICNTRL,NATIML,ixtar,iytar,iztar,itrcm,ATYPE, &
               RESID,RES,NRSIML,IBASE,SEGID,NICTOT,NSGIML,.FALSE.)
       else
          CALL WRNDIE(-3,'<MAINIO>', &
               ' USE TMDI BEFORE READING TARGET STRUCTURE')
       endif
       ! MSF tmd \./
    ELSE IF(INDXA(COMLYN,COMLEN,'TAR2') > 0) THEN
       if(qtmd)then
          CALL COORIO(IPRINT,IUNIT,COMLYN,COMLEN,TITLEB,NTITLB, &
               ICNTRL,NATIML,ixtar2,iytar2,iztar2,itrcm,ATYPE, &
               RESID,RES,NRSIML,IBASE,SEGID,NICTOT,NSGIML,.FALSE.)
       else
          CALL WRNDIE(-3,'<MAINIO>', &
               ' USE TMDI BEFORE READING TARGET STRUCTURE')
       endif
       ! MSF End
#endif 
    ELSE
       CALL COORIO(IPRINT,IUNIT,COMLYN,COMLEN,TITLEA,NTITLA, &
            ICNTRL,NATIML,X,Y,Z,WMAIN,ATYPE, &
            RESID,RES,NRSIML,IBASE,SEGID,NICTOT,NSGIML,.FALSE.)
    ENDIF
    !         End Procedure WRITE-COORDINATE-FILE
    return
  end subroutine output_coor

  subroutine output_hbon
    !=======================================================================
    !         Begin Procedure WRITE-HBOND-FILE
    !
    QPRINT=(INDXA(COMLYN,COMLEN,'ANAL') > 0) .OR. (IPRINT == 1)
    IF (INDXA(COMLYN,COMLEN,'CARD') > 0) THEN
       LCARD=.TRUE.
    ELSE
       IXX=INDXA(COMLYN,COMLEN,'FILE')
       LCARD=.FALSE.
    ENDIF
    CALL HBWRIT(IUNIT,QPRINT,LCARD,TITLEA,NTITLA,ICNTRL,IHB,JHB, &
         KHB,LHB,ICH,NHB,CUTHB,CUTHBA,CTONHB,CTOFHB, &
         CTONHA,CTOFHA,CHBA,CHBB,HBEXPN,X,Y,Z)
    !
    IF((LCARD.OR.QPRINT) .AND. NTRANS > 0) THEN
       IF(NIMHB > 0) &
            CALL HBWRIT(IUNIT,QPRINT,LCARD,TITLEA,0,ICNTRL, &
            IHB(NHB+1),JHB(NHB+1),KHB(NHB+1), &
            LHB(NHB+1),ICH(NHB+1),NIMHB,CUTHB,CUTHBA, &
            CTONHB,CTOFHB, &
            CTONHA,CTOFHA,CHBA,CHBB,HBEXPN,X,Y,Z)
    ENDIF
    !
    IF (IPRINT == 0) CALL VCLOSE(IUNIT,'KEEP',ERROR)
    !         End Procedure WRITE-HBOND-FILE
    return
  end subroutine output_hbon

  subroutine output_constraints
    !=======================================================================
    !         Begin Procedure WRITE-CONSTRAINT-FILE
    !
    IF (IPRINT > 0) THEN
#if KEY_PERT==1 /*pert_prcns*/
       IPSF=GTRMI(COMLYN,COMLEN,'PSF',1)
       IF(IPSF == 0) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
               '  MAINIO> PERT reference psf will be used.'
          CALL PRCNST(IUNIT,NCSPHP, &
               PRICS,PRJCS, &
               PRKCS,PRLCS, &
               PRCCSC,PRCCSB, &
               PRCCSD,PRCCSW, &
               NATOMP,QCNSRP,NUMHSETP, &
               PRIHSET,PRTYPEH, &
               PRKCNST,PRKCEXP, &
               LCIC,CCBIC,CCTIC,CCPIC,CCIIC, &
               QQCNST,LQMASS,KQCNST,KQEXPN)
       ELSE
#endif /* (pert_prcns)*/
          CALL PRCNST(IUNIT, &
               NCSPHI,ICS,JCS,KCS,LCS,CCSC,CCSB,CCSD,CCSW, &
               NATOM,QCNSTR,NUMHSETS,IHSET,TYPHSET,KCNSTR, &
               KCEXPN,LCIC,CCBIC,CCTIC,CCPIC,CCIIC, &
               QQCNST,LQMASS,KQCNST,KQEXPN)
#if KEY_PERT==1 /*pert_prcns2*/
       ENDIF
#endif /* (pert_prcns2)*/
    ELSE
       IXX=INDXA(COMLYN,COMLEN,'CARD')
       !
#if KEY_PERT==1 /*pert_wrcns*/
       IPSF=GTRMI(COMLYN,COMLEN,'PSF',1)
       IF(IPSF == 0) THEN
          IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
               '  MAINIO> PERT reference psf will be used.'
          CALL WRCNST(IUNIT,TITLEA,NTITLA,QCNSRP,LCIC,QQCNST, &
               NCSPHP,PRICS,PRJCS, &
               PRKCS,PRLCS, &
               PRICCS,PRCCSC, &
               PRCCSB,PRCCSD, &
               PRCCSW)
       ELSE
#endif /* (pert_wrcns)*/
          CALL WRCNST(IUNIT,TITLEA,NTITLA,QCNSTR,LCIC,QQCNST, &
               NCSPHI,ICS,JCS,KCS,LCS,ICCS,CCSC,CCSB,CCSD,CCSW)
#if KEY_PERT==1 /*pert_wrcns2*/
       ENDIF
#endif /* (pert_wrcns2)*/
       CALL VCLOSE(IUNIT,'KEEP',ERROR)
    ENDIF
    !         End Procedure WRITE-CONSTRAINT-FILE
    !=======================================================================
    return
  end subroutine output_constraints

  subroutine output_ic
    !=======================================================================
    !         begin procedure write-internal-coordinate-file
    !
    icard=1
    if(indxa(comlyn,comlen,'FILE') /= 0) icard=0
    if(indxa(comlyn,comlen,'CARD') > 0) icard=1
    if(indxa(comlyn,comlen,'RESI') > 0) icard=2
    if(indxa(comlyn,comlen,'RTF') > 0) icard=3
    if(indxa(comlyn,comlen,'SAVE') > 0) then
       call writic(1,ics_struct%lenic,iprint,icard,iunit, &
            ics_struct%b1ic,ics_struct%b2ic, &
            ics_struct%t1ic,ics_struct%t2ic, &
            ics_struct%pic, ics_struct%iar, &
            ics_struct%jar, ics_struct%kar, &
            ics_struct%lar, ics_struct%tar)
    else
       call writic(1,icr_struct%lenic,iprint,icard,iunit, &
            icr_struct%b1ic,icr_struct%b2ic, &
            icr_struct%t1ic,icr_struct%t2ic, &
            icr_struct%pic, icr_struct%iar, &
            icr_struct%jar, icr_struct%kar, &
            icr_struct%lar, icr_struct%tar)
    endif
    if(iprint == 0) call vclose(iunit,'KEEP',error)

    !         End Procedure WRITE-INTERNAL-COORDINATE-FILE
    !=======================================================================
    return
  end subroutine output_ic

  subroutine output_nonbonded
    !=======================================================================
    !         Begin Procedure WRITE-NONBONDED-FILE
    !
    IF (IPRINT > 0) THEN
       CALL PRNBND(IUNIT,BNBND,BIMAG)
    ELSE
       IF (INDXA(COMLYN,COMLEN,'FILE') > 0) THEN
          CALL WRNDIE(0,'<MAINIO>', 'write nbond disabled')
       ELSE
          CALL WRNDIE(0,'<MAINIO>', &
               'Invalid nonbond I/O option, use FILE.')
       ENDIF
    ENDIF
    !         End Procedure WRITE-NONBONDED-FILE
    !=======================================================================
    return
  end subroutine output_nonbonded

  subroutine output_psf
    !=======================================================================
    !         begin procedure write-structure-file
    !
    if(indxa(comlyn,comlen,'FILE') > 0) then
         call wrndie(-5,'<MAINIO>', &
         'Binary PSF files not supported in this CHARMM version')
    else
      ixx=indxa(comlyn,comlen,'CARD')
      ixx=iprint
      if(indxa(comlyn,comlen,'FULL') > 0) ixx=2
!!      if(indx(comlyn,comlen,'XPLO',4) > 0) ixx=3
      if(ixx==1) then
         call psfwrt(iunit,ixx,icntrl,titlea,ntitla)
      else
         call psfwr2(iunit,titlea,ntitla)
      endif
    endif
    if(iprint == 0) call vclose(iunit,'KEEP',error)
    !         End Procedure WRITE-STRUCTURE-FILE
    !=======================================================================
    return
  end subroutine output_psf

  subroutine output_rtf
    !=======================================================================
    !         Begin Procedure WRITE-TOPOLOGY-FILE
    !
    iprint=1  ! defualt rtf format
    if(indxa(comlyn,comlen,'FILE') > 0) iprint=0
    if(indxa(comlyn,comlen,'CARD') > 0) iprint=1
    len=nic(1,nrtrs)
    call rtwrit(titlea,ntitla,iunit,iprint,len)
    if(iprint == 0) call vclose(iunit,'KEEP',error)
    !         End Procedure WRITE-TOPOLOGY-FILE
    !=======================================================================
    return
  end subroutine output_rtf

  subroutine output_table_file
    !=======================================================================
    !         Begin Procedure WRITE-TABLE-FILE
    !
#if KEY_NOMISC==0
    IXX=INDXA(COMLYN,COMLEN,'FILE')
    CALL WRITBL(IUNIT,IPRINT,NATC)
    IF(IPRINT == 0) CALL VCLOSE(IUNIT,'KEEP',ERROR)
#else /**/
    CALL WRNDIE(-1,'<MAINIO>','ETABLE code is not compiled.')
#endif 
    !         End Procedure WRITE-TABLE-FILE
    !=======================================================================
    return
  end subroutine output_table_file

  subroutine output_resd
    !=======================================================================
    !         Begin Procedure: write restrained distance list
    !
#if KEY_NOMISC==0
    CALL REDWRI(IUNIT)
    IF(IPRINT == 0) CALL VCLOSE(IUNIT,'KEEP',ERROR)
#else /**/
    CALL WRNDIE(-1,'<MAINIO>', &
         'Restrained distance code is not compiled.')
#endif 
    !
    !=======================================================================
    return
  end subroutine output_resd

  subroutine output_xray_graphics
    !=======================================================================
    !         Begin Procedure WRITE-XRAY-GRAPHICS-FILE
    LCOMP=(INDXA(COMLYN,COMLEN,'COMP') > 0)
    IF (LCOMP) THEN
       CALL WRXRAY(IUNIT,COMLYN,COMLEN,XCOMP,YCOMP,ZCOMP,WCOMP)
    ELSE
       CALL WRXRAY(IUNIT,COMLYN,COMLEN,X,Y,Z,WMAIN)
    ENDIF
    !         End Procedure WRITE-XRAY-GRAPHICS-FILE
    return
  end subroutine output_xray_graphics


END SUBROUTINE MAINIO

  SUBROUTINE clear_something
  use chm_kinds
  use chm_types
  use comand
  use stream
  use string
  use rtf,only:clear_all_rtrs
  implicit none

  CHARACTER(len=4)  WRD

     WRD=NEXTA4(COMLYN,COMLEN)

     IF (WRD =='RTF') THEN
 !      Clear out all of the RTF arrays (except for NATCT related arrays)
        IF(PRNLEV >= 2) WRITE (OUTU, '(A,I3,A)') &
             ' CLEAR> All RTF residue based data is cleared and memory freed.'       
        call clear_all_rtrs
     ELSE
        CALL WRNDIE(0,'<MAINIO>','Unrecognized CLEAr command option.:'//WRD)
     ENDIF

END SUBROUTINE clear_something

