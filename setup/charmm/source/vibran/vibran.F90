module vibran_m
  implicit none
contains

SUBROUTINE VIBOPT(COMLYN,COMLEN)
  !
  ! VIBOPT is the front end routine for the VIBRAN options.
  !
  use chm_kinds
  use dimens_fcm
  use number,only:zero
  use exfunc
  use coord
  use coordc
  use memory
  use image
  use psf
  use string
  implicit none
  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  ! . Local variables.
  INTEGER   NATIML
  real(chm_real),allocatable,dimension(:) :: NEWX,NEWY,NEWZ
  real(chm_real),allocatable,dimension(:) :: NORMX,NORMY,NORMZ

  IF(INDXA(COMLYN,COMLEN,'COMP').GT.0) THEN
     CALL WRNDIE(1,'<VIBRAN>','COMP OPTION NOT SUPPORTED')
  ENDIF

  CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
       .TRUE.,.TRUE.,.TRUE.,.TRUE., &
       0,[zero],[zero],[zero],[zero],[zero],[zero])

  NATIML=MAX(NATOM,NATIM)

  call chmalloc('vibran.src','VIBOPT','NEWX',NATIML,crl=NEWX)
  call chmalloc('vibran.src','VIBOPT','NEWY',NATIML,crl=NEWY)
  call chmalloc('vibran.src','VIBOPT','NEWZ',NATIML,crl=NEWZ)
  call chmalloc('vibran.src','VIBOPT','NORMX',NATIML,crl=NORMX)
  call chmalloc('vibran.src','VIBOPT','NORMY',NATIML,crl=NORMY)
  call chmalloc('vibran.src','VIBOPT','NORMZ',NATIML,crl=NORMZ)

  call VIBRAN(X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP,NEWX,NEWY,NEWZ, &
       NORMX,NORMY,NORMZ)

  call chmdealloc('vibran.src','VIBOPT','NEWX',NATIML,crl=NEWX)
  call chmdealloc('vibran.src','VIBOPT','NEWY',NATIML,crl=NEWY)
  call chmdealloc('vibran.src','VIBOPT','NEWZ',NATIML,crl=NEWZ)
  call chmdealloc('vibran.src','VIBOPT','NORMX',NATIML,crl=NORMX)
  call chmdealloc('vibran.src','VIBOPT','NORMY',NATIML,crl=NORMY)
  call chmdealloc('vibran.src','VIBOPT','NORMZ',NATIML,crl=NORMZ)

  RETURN
END SUBROUTINE VIBOPT

SUBROUTINE VIBRAN(X,Y,Z,WMAIN,XCOMP,YCOMP,ZCOMP, &
     XNEW,YNEW,ZNEW,XNORM,YNORM,ZNORM)
  !
  ! VIBRATION ANALYSIS
  !
  !----------------------------------------------------------
  ! THIS ROUTINE AND ASSOCIATED ROUTINES WERE DEVELOPED
  ! BY BERNARD R BROOKS   FROM 1981 TO PRESENT (1984)
  !
  ! Mixed-basis diagonalization (DIMB), and CMPAct option
  ! for the reduced basis diagonalization added by:
  ! 01-Mar-1993 David Perahia
  ! 15-Dec-1994 Herman van Vlijmen
  ! 28-Dec-2004 Victor Anisimov: Entropy
  !----------------------------------------------------------
  !
  !      X,Y,Z,WMAIN  - main coordinates
  !      XCOMP,YCOMP,ZCOMP - comparison coordinates
  !      XNEW,YNEW,ZNEW - temporary coordinates
  !      XNORM,YNORM,ZNORM - temporary vector for displacement modes
  !
  !  SECOND DERIVATIVE DATA STRUCTURE
  !
  !      DD1  - second derivative matrix
  !
  !  NORMAL MODE DATA STRUCTURE
  !
  !      NNMDS         - maximum number of vectors allowed
  !      NFREQ         - number of vectors active
  !      DDV(3*NATOM,NNMDS) - vectors (normal modes)
  !      DDM(NATOM)    - root inverse mass array
  !      DDF(NNMDS)    - frequencies in cm-1
  !      DDEV(NNMDS)   - eigenvalues
  !      DDSCR(3*NATOM)- scratch working vector
  !
  !  DIAGONALIZATION DATA STRUCTURE
  !
  !      NDIM          - dimension of matrix to diagonalize
  !      DD5(NDIM*(NDIM+1)/2)- matrix to diagonalize (destroyed upon diag)
  !      DDS(8*(NDIM+1))   - scratch vectors for diagonalization
  !
  use chm_kinds
  use chm_types
  use intcor_module
  use dimens_fcm
  use exfunc
  use number
  use entropy
  use consta
  use bases_fcm
  use comand
  use corman_mod,only:corcom
  use ctitla
  use deriv
  use hbondm
  use mbh_m
  use memory
  use inbnd
  use psf
  use param
  use redbas_m
  use select
  use stream
  use string
  use parallel
  use rndnum     !yw 05-Aug-2008
  use cvio
  use vibcom
  use vibio
  use vibsub
#if KEY_DIMB==1
  use dimb  
#endif
  use clcg_mod,only:rngmodseeds
  use correl_mod,only:correl
  use chutil,only:atomid,makind
#if KEY_DIMB==1
  use nmdimb_module,only:nmdimb        
#endif
#if KEY_DIMB==1
  use dimbsubs,only:partit,nblist      
#endif

  implicit none

  real(chm_real) :: X(:), Y(:), Z(:), WMAIN(*), XNEW(:), YNEW(:), ZNEW(:)
  real(chm_real) XCOMP(:),YCOMP(:),ZCOMP(:), &
       XNORM(*),YNORM(*),ZNORM(*)

  LOGICAL EOF,LUSED,ERROR,OK


  INTEGER NAT3,NDIM,NFREQ,IUNIT,ISTRT,ISTOP,IXX
  integer,allocatable,dimension(:) :: ISLCT
  INTEGER NNMDS
  INTEGER NCYC,ITYPE,NTROT
  real(chm_real),allocatable,dimension(:) :: ITROT
  INTEGER IUNBAS,IUNTRN
  real(chm_real) STEP,TOL,PHAS,RTYPE,TFREQ,RTEMP,RKCAL, &
       RMRMS,RRMS,RFACT
  INTEGER NFRE,NADD,NUNIT,IFIRST,NSKIP,IDEST,ISOURC,I
  INTEGER NBEGN,NSTOP,NBEGN2,NCOORD,NDIM2,NSTOP2
  CHARACTER(len=4) :: HDR
  INTEGER ICNTRL(20)
  integer,allocatable,dimension(:) :: ISKP
  INTEGER NGRID,NGRID2
  real(chm_real),allocatable,dimension(:) :: IESTOR ,IRSTOR, IWSTOR
  integer,allocatable,dimension(:) :: IISTOR
  real(chm_real),allocatable,dimension(:) :: IXBOND,IXTHET,IXIMPH,IXPHI
  INTEGER IMOD,JSP,ISPACE,JSPACE,INEED
  ! Temporary memory allocation variables for AMASS backup
  real(chm_real),allocatable,dimension(:) :: BAM
  INTEGER BSPACE
#if KEY_CMAP==1
  real(chm_real),allocatable,dimension(:) :: IXCMAP
#endif 
  integer,allocatable,dimension(:) :: IND1
  INTEGER NSET1,NSMALL
  real(chm_real),allocatable,dimension(:) :: DD6
  LOGICAL LTHERMO,LRESI
  LOGICAL LAUTOD,LSSPAD,LCSPAD,LWRIT
  INTEGER ISEED
  real(chm_real) FACD,FACS,AFACT,RBOLT,RGAUS,FCUT
  LOGICAL LSUPE,LRAND
  !
  ! Variables used in OPTI and the CBON/CANG basis sets (G. Koenig 2014)
  real(chm_real),allocatable,dimension(:) :: FJAC !Array for Jacobian factors+angle radii
  integer,allocatable,dimension(:) :: ANGA ! Array for indices of atoms in angle CANG
  INTEGER NFJAC ! Number of entries in FJAC  (OPTI command)
  INTEGER IUNO  ! Output unit for OPTI
  !
  ! PARAMETER FITTING LOCAL VARIABLES
  !
  LOGICAL LDOTP,LVECT,LINTD,LAPPE,LNOMA,LNONO,LORTH,LSEQU,LCOMP
  LOGICAL LPAX,LNOMS,LCONT,LGROU,LNORM,LMEAN,LSPIR,LNOTP
  LOGICAL LDIPO,LRAISE,LADJU,LSHAK,LFINIT,LSTAT,LCARD,LQUAN
  LOGICAL LATOM,LIC,LUSER,LSAVE,LBIG,LFIX,LENER,LPARA
  LOGICAL LPURG,ERR,LNOTR,LNMASS,LDSCF,LENTRO

  CHARACTER(len=4) :: WRD
  real(chm_real),allocatable,dimension(:) :: DD1
  real(chm_real),allocatable,dimension(:) :: DDV,DDM,DDF
  real(chm_real),pointer,dimension(:) :: DD5
  real(chm_real),allocatable,target,dimension(:) :: DDS
  real(chm_real),allocatable,dimension(:) :: DDEV,DDSCR,DDV2
  real(chm_real),allocatable,dimension(:) :: DDFX,DDEVX
  real(chm_real),allocatable,dimension(:) :: XTEMP,YTEMP,ZTEMP

  ! QC: UW_06
  LOGICAL LNOVEC

#if KEY_DIMB==1
  !
  ! DIMB algorithm declarations
  ! DD1BLK      second derivatives of the potential energy of a submatrix
  !             upper diagonal form
  ! DD1BLL      second derivatives of the potential energy of a submatrix
  !             lower diagonal form
  ! QDISK       logical flag for disk storage of DD1CMP
  !

  real(chm_real),allocatable,dimension(:) :: DD1BLK,DD1BLL
  INTEGER NPAR,NFREG,NFRET
  INTEGER RESPAR(2,NPARMX),IUNPAR,ATMPAR(2,NPARMX),PARDIM,PARNMD
  INTEGER ATMPAS(2,NPARMX),ATMPAD(2,NPARMX),IUNMOD,IUNRMD
  real(chm_real),allocatable,dimension(:) :: PARDDV,PARDDF,PARDDE
  INTEGER PARTOT,VECSHR,ORGVEC
  real(chm_real),allocatable,dimension(:) :: PARDDM
  INTEGER ITMX,ISPBLK
  INTEGER IS1,IS2,BLATOM,PARD,SAVF
  INTEGER I900,I910,I920,I930
  real(chm_real) PARFRQ,CUTF1
  real(chm_real) TOLDIM,DDVALM
#endif 
  LOGICAL LSCI, qpresent
  integer,allocatable, dimension(:) :: lrngseeds

  ! Note: added by A.Ghysels - could be moved to better place
  INTEGER NBLK,NSUBS
  integer,allocatable,dimension(:) :: BLKINFO

  ! JZ_UW12
  INTEGER NSAVDD1
  LOGICAL QREST

  EOF=.FALSE.
  ERROR=.FALSE.
  TFREQ=5.0
  PHAS=30.0
  NCYC=1

  NAT3=NATOM*3
  NFREQ=0
  NNMDS=GTRMI(COMLYN,COMLEN,'NMOD',-99)
  IF(NNMDS.EQ.-99 .AND. NAT3.LE.150) NNMDS=NAT3
  IF(NNMDS.LE.0) NNMDS=1
  LNOMA=.FALSE.

  IF(PRNLEV.GE.2) WRITE(OUTU,722) NNMDS,NAT3
  ! QC:UW_06: Reduce space for REDU FIX
  LNOVEC=(INDXA(COMLYN,COMLEN,'NOVC').GT.0)
  IF (.NOT.LNOVEC) THEN
     call chmalloc('vibran.src','VIBRAN','DDV',NNMDS*NAT3,crl=DDV)
  ELSE
     call chmalloc('vibran.src','VIBRAN','DDV',2*NAT3,crl=DDV)
     if (prnlev.gt.2) WRITE(OUTU,'(a)')  &
          'VIBRAN> SUPRESS BACK TRANSFORMATION FOR REDU FIX'
  ENDIF
  ! QC:UW_06 DONE

  call chmalloc('vibran.src','VIBRAN','DDM',NATOM,crl=DDM)
  call FILDDM(DDM,AMASS,NATOM,LNOMA)
  call chmalloc('vibran.src','VIBRAN','DDF',NNMDS,crl=DDF)
  call chmalloc('vibran.src','VIBRAN','DDEV',NNMDS,crl=DDEV)
  DDEV(1:NNMDS)=zero
  call chmalloc('vibran.src','VIBRAN','DDSCR',MAX(NAT3,NNMDS),crl=DDSCR)
  call chmalloc('vibran.src','VIBRAN','ISLCT',NATOM,intg=ISLCT)

  ! Note: added by A.Ghysels - could be moved to better place
  call chmalloc('vibran.src','VIBRAN','BLKINFO',NATOM,intg=BLKINFO)

  ! Allocate arrays used in OPTI  (G. Koenig 2014)
  call chmalloc('vibran.src','VIBRAN','FJAC',NNMDS*3,crl=FJAC)
  call chmalloc('vibran.src','VIBRAN','ANGA',NNMDS*3,intg=ANGA)
  NFJAC = 0 
  FJAC(1:3*NNMDS) = 0.0E+0
  ANGA(1:3*NNMDS) = 0

722 FORMAT(' VIBRAN: Space allocated for',I6,' vectors of length',I6)
  nullify(DD5)

  parse: DO
     CALL XTRANE(COMLYN,COMLEN,'VIBRAN')
     LUSED=.TRUE.
     DO WHILE (LUSED.AND. .NOT.EOF)
        CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
             .TRUE.,'VIBRAN> ')
        CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     ENDDO
     IF (EOF) THEN
        IF (NSTRM.EQ.1) THEN
           ! Clean up space for normal modes.
           ! UW_06: Why allocate again? Is this a bug???
           !           QC: UW_06: suppressed eigenvectors
           IF (.NOT.LNOVEC) THEN
              call chmdealloc('vibran.src','VIBRAN','DDV',NNMDS*NAT3,crl=DDV)
           ELSE
              call chmdealloc('vibran.src','VIBRAN','DDV',2*NAT3,crl=DDV)
           ENDIF

           call chmdealloc('vibran.src','VIBRAN','DDM',NATOM,crl=DDM)
           call chmdealloc('vibran.src','VIBRAN','DDF',NNMDS,crl=DDF)
           call chmdealloc('vibran.src','VIBRAN','DDEV',NNMDS,crl=DDEV)
           call chmdealloc('vibran.src','VIBRAN','DDSCR',MAX(NAT3,NNMDS),crl=DDSCR)
           call chmdealloc('vibran.src','VIBRAN','ISLCT',NATOM,intg=ISLCT)

           ! Note: added by A.Ghysels - could be moved to better place
           call chmdealloc('vibran.src','VIBRAN','BLKINFO',NATOM,intg=BLKINFO)

           ! Deallocate arrays used in OPTI  (G. Koenig 2014)
           call chmdealloc('vibran.src','VIBRAN','FJAC',NNMDS*3,crl=FJAC)
           call chmdealloc('vibran.src','VIBRAN','ANGA',NNMDS*3,intg=ANGA)

           RETURN
        ENDIF

        !mbm..
        CALL PPSTRM(OK)
        EOF=.FALSE.
        CYCLE parse
     ENDIF
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD.EQ.'    ') CYCLE parse
     !
     ! Main parsing conditional
     !=======================================================================
     IF(WRD.EQ.'READ') THEN
        ! PROCESS-READ-COMMAND
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        LCARD=(INDXA(COMLYN,COMLEN,'CARD').GT.0)
        WRD=NEXTA4(COMLYN,COMLEN)
        IF (WRD.EQ.'NORM') THEN
           LAPPE=(INDXA(COMLYN,COMLEN,'APPE').GT.0)
           ISTRT=GTRMI(COMLYN,COMLEN,'MODE',1)
           ISTOP=GTRMI(COMLYN,COMLEN,'THRU',99999999)
           IXX=INDXA(COMLYN,COMLEN,'FILE')
           CALL RDNMD(LCARD,NFREQ,NNMDS,NAT3,NDIM, &
                DDV,DDSCR,DDF,DDEV, &
                IUNIT,LAPPE,ISTRT,ISTOP)
        ELSE
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !=======================================================================
     ELSE IF(WRD.EQ.'WRIT') THEN
        ! PROCESS-WRIT-COMMAND
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        IXX=INDXA(COMLYN,COMLEN,'FILE')
        WRD=NEXTA4(COMLYN,COMLEN)
        CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
        IF(WRD.EQ.'SECO') THEN
           LCARD=(INDXA(COMLYN,COMLEN,'CARD').GT.0)
           NDIM=0
           IF(LCARD) THEN
              NDIM=NAT3
              CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           ENDIF
           LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
           LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
           STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
           TOL=GTRMF(COMLYN,COMLEN,'TOL',PT0001)
           LNMASS=.NOT.(INDXA(COMLYN,COMLEN,'MASS').GT.0)

           call vib_checkspace()
           CALL WRTSCD(X,Y,Z,XNEW,YNEW,ZNEW,NAT3,DDM,LCARD, &
                ISLCT,BNBND,BIMAG,NFREQ,LNMASS,AMASS, &
                LFINIT,STEP,TOL,IUNIT,LRAISE)
        ELSE IF(WRD.EQ.'NORM') THEN
           LCARD=(INDXA(COMLYN,COMLEN,'CARD').GT.0)
           call vib_check_nfreq()

           call vib_get_mode_spec
              CALL WRTNMD(LCARD,ISTRT,ISTOP,NAT3, &
                   DDV,DDSCR,DDEV, &
                   IUNIT,AMASS)
        ELSE IF(WRD.EQ.'TRAJ') THEN
           call vib_check_nfreq()

           call vib_get_mode_spec()

           PHAS=GTRMF(COMLYN,COMLEN,'PHAS',PHAS)
           STEP=GTRMF(COMLYN,COMLEN,'STEP',ZERO)
           NCYC=GTRMI(COMLYN,COMLEN,'NCYC',NCYC)
           LSEQU=(INDXA(COMLYN,COMLEN,'SEQU').GT.0)
           LSHAK=(INDXA(COMLYN,COMLEN,'SHAK').GT.0)
           LSUPE=(INDXA(COMLYN,COMLEN,'SUPE').GT.0)
           LRAND=(INDXA(COMLYN,COMLEN,'RAND').GT.0)
           !             ISEED=GTRMI(COMLYN,COMLEN,'ISEE',314159)
           !             if (.not.qoldrng) then       !yw 05-Aug-2008
           !                CALL CLCGINIT(ISEED)
           !                ISEED=1
#if KEY_PARALLEL==1
           !                ISEED=MYNODP        
#endif
           !             endif
           !
           iseed=314159
           call chmalloc('vibran.src','VIBRAN','lrngseeds',Nrand,intg=lrngseeds)
           lrngseeds(1:nrand)=rngseeds(1:nrand)
           if(qoldrandom.or.qbrokenclcg)lrngseeds(1:nrand)=iseed
           call gtrmim(nrand,comlyn,comlen,'ISEE',lrngseeds,rngseeds,qpresent)
           call rngmodseeds(qpresent,iseed)
           call chmdealloc('vibran.src','VIBRAN','lrngseeds',Nrand,intg=lrngseeds)
           !
           call vib_get_magnitude_spec()

           call chmalloc('vibran.src','VIBRAN','ISKP',NATOM,intg=ISKP)
           IF(.NOT.LSUPE) THEN
                 CALL VIBTRJ(ISTRT,ISTOP,NAT3,DDM,IUNIT,ITYPE,RTYPE, &
                      LNOMA,LNONO,LSEQU,X,Y,Z,DDV,DDSCR, &
                      XCOMP,YCOMP,ZCOMP,XNORM,YNORM,ZNORM,PHAS,NCYC,STEP, &
                      DDEV,DDF,TFREQ, &
                      LSHAK,XNEW,YNEW,ZNEW,AMASS,IMOVE,ISKP)
           ELSE
                 CALL VIBSUP(ISTRT,ISTOP,NAT3,DDM,IUNIT,ITYPE,RTYPE, &
                      LNOMA,LNONO,LSEQU,X,Y,Z,DDV,DDSCR, &
                      XCOMP,YCOMP,ZCOMP,XNORM,YNORM,ZNORM,PHAS,NCYC,STEP, &
                      DDEV,DDF,TFREQ,LSHAK,XNEW,YNEW,ZNEW, &
                      AMASS,IMOVE,ISKP,LRAND,ISEED)

           ENDIF
           call chmdealloc('vibran.src','VIBRAN','ISKP',NATOM,intg=ISKP)
        ELSE
           ! Bad write option.
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        IF (IUNIT.NE.OUTU) CALL VCLOSE(IUNIT,'KEEP',ERROR)
        !=======================================================================
     ELSE IF(WRD.EQ.'PRIN') THEN
        ! PROCESS-PRIN-COMMAND
        IUNIT=OUTU
        WRD=NEXTA4(COMLYN,COMLEN)
        IF (WRD.EQ.'NORM') THEN
           call vib_check_nfreq()

           call vib_get_mode_spec()

           call vib_get_magnitude_spec()

           LINTD=(INDXA(COMLYN,COMLEN,'INTD').GT.0)
           LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
           LSTAT=(INDXA(COMLYN,COMLEN,'STAT').GT.0)
           LVECT=(INDXA(COMLYN,COMLEN,'VECT').GT.0)
           LDOTP=(INDXA(COMLYN,COMLEN,'DOTP').GT.0)
           LDIPO=(INDXA(COMLYN,COMLEN,'DIPO').GT.0)

              CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
              if (prnlev >= 2) CALL PRTNMD(ISTRT,ISTOP,IUNIT,NAT3,ISLCT, &
                   DDV,DDM,DDF,DDEV,DDSCR, &
                   LNOMA,LNONO,ITYPE,RTYPE,TFREQ, &
                   LINTD,LFINIT,LSTAT,LVECT,LDOTP,X,Y,Z,XNEW,YNEW,ZNEW, &
                   XNORM,YNORM,ZNORM,CG,LDIPO, &
                   AMASS,IBASE,ATYPE,RES,NRES &
                   )
        ELSE
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !=======================================================================
     ELSE IF(WRD.EQ.'IC  ') THEN
        CALL INTCOR(COMLYN,COMLEN)
        !=======================================================================
     ELSE IF(WRD.EQ.'COOR') THEN
        CALL CORCOM(COMLYN,COMLEN)
        !=======================================================================
     ELSE IF(WRD.EQ.'CORR') THEN
        CALL CORREL(NFREQ,DDV)
        !=======================================================================
     ELSE IF(WRD.EQ.'DIAG') THEN
        ! PROCESS-DIAG-COMMAND
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NAT3)
        NADD=GTRMI(COMLYN,COMLEN,'NADD',0)
        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        IF(NFREQ.GT.0 .AND. PRNLEV.GE.2) WRITE(OUTU,247) NFREQ
247     FORMAT(I5,'  MODES ALREADY EXIST. THEY WILL BE SUPERCEDED')
        IF(NFRE.GT.NAT3) NFRE=NAT3
        IF(NFRE.GT.NNMDS) NFRE=NNMDS
        IF((NFRE+NADD).GT.NAT3 .OR. NADD.LT.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,632) NFRE,NADD,NAT3
632        FORMAT(' **** ERROR **** BAD NADD VALUE.',3I6)
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF

        ! Finite derivatives for normal modes
        LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
        STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
        ! JZ_UW12: For saving DD1 in intervals
        NSAVDD1=GTRMI(COMLYN,COMLEN,'NSDD',3*NATOM)
        QREST=(INDXA(COMLYN,COMLEN,'REST').GT.0)

        ! Flag for excluding Drudes from numerical differenciation
        LDSCF=(INDXA(COMLYN,COMLEN,'DSCF').GT.0.AND.QDRUDE.AND.LFINIT)
        ! Backup atomic masses
        ! Move Drude masses to corresponding real atoms and recalculate DDM
        IF(LDSCF) THEN
           BSPACE=NATOM*8
           call chmalloc('vibran.src','VIBRAN','BAM',BSPACE,crl=BAM)
           BAM(1:NATOM)=AMASS(1:NATOM)
           DO I=2,NATOM
              IF(ISDRUDE(I)) THEN
                 AMASS(I-1)=AMASS(I-1)+AMASS(I)
                 AMASS(I)=ZERO
              ENDIF
           ENDDO
           CALL FILDDM(DDM,AMASS,NATOM,LNOMA)
        ENDIF

        ! Entropy keywords
        LENTRO=(INDXA(COMLYN,COMLEN,'ENTR').GT.0)
        IF(LENTRO) THEN
           ! Termerature
           TK=GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
           ! Rotational symmetry number
           ! Explanation of sigma: C.J.Cramer, "Essentials of Comp.Chem.",2002,p.327
           SIGMA=GTRMF(COMLYN,COMLEN,'SIGM',ONE)
           ! Standard state: Tidor and Karplus, J Mol Biol (1994) vol. 238 (3) pp. 405-14
           ! Default is solution state of conentration 1M
           SSTAN=GTRMA(COMLYN,COMLEN,'STAN')
           ! Next is for a test purpose only (for developers)
           UTEST=GTRMI(COMLYN,COMLEN,'TEST',0)
        ENDIF

        NFREQ=NFRE
        IF(NFRE.LE.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,249)
249        FORMAT(/'**** ERROR **** NUMBER OF SPECIFIED FREQUENCIES', &
                ' IS LESS THAN ONE')
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF

        call vib_checkspace()

        CALL NORMDS(X,Y,Z,NAT3,BNBND,BIMAG,NFREQ,LNOMA,AMASS, &
             DDV,DDM,DDF,DDEV, &
             NADD,LRAISE, &
             LFINIT,STEP,LDSCF,LENTRO, &
             NSAVDD1,QREST) ! JZ_UW12
        ! Restore atomic masses (AMASS) to their original value
        IF(LDSCF) THEN
           AMASS(1:NATOM)=BAM(1:NATOM)
           call chmdealloc('vibran.src','VIBRAN','BAM',BSPACE,crl=BAM)
        ENDIF
        !=======================================================================
        !QQQQ
     ELSE IF(WRD.EQ.'VSA') THEN
        ! Do a subsystem normal mode calculation where all non-selected degrees
        ! of freedom are optimized. - BRB
        !
        CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NAT3)
        NADD=GTRMI(COMLYN,COMLEN,'NADD',0)
        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        IF(NFRE.GT.NAT3) NFRE=NAT3
        IF(NFRE.GT.NNMDS) NFRE=NNMDS
        IF((NFRE+NADD).GT.NAT3 .OR. NADD.LT.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,632) NFRE,NADD,NAT3
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF

        ! Finite derivatives for normal modes
        LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
        STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)

        ! Entropy keywords
        LENTRO=(INDXA(COMLYN,COMLEN,'ENTR').GT.0)
        IF(LENTRO) THEN
           ! Termerature
           TK=GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
           ! Rotational symmetry number
           ! Explanation of sigma: C.J.Cramer, "Essentials of Comp.Chem.",2002,p.327
           SIGMA=GTRMF(COMLYN,COMLEN,'SIGM',ONE)
           ! Standard state: Tidor and Karplus, J Mol Biol (1994) vol. 238 (3) pp. 405-14
           ! Default is solution state of conentration 1M
           SSTAN=GTRMA(COMLYN,COMLEN,'STAN')
           ! Next is for a test purpose only (for developers)
           UTEST=GTRMI(COMLYN,COMLEN,'TEST',0)
        ENDIF

        NFREQ=NFRE
        IF(NFRE.LE.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,249)
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF

        call vib_checkspace()

        CALL SUBSDS(X,Y,Z,NAT3,BNBND,BIMAG,NFREQ,LNOMA,AMASS, &
             DDV,DDM,DDF,DDEV, &
             ISLCT,NADD,LRAISE, &
             LFINIT,STEP,LDSCF,LENTRO)

        !=======================================================================
#if KEY_DIMB==1
     ELSE IF(WRD.EQ.'DIMB') THEN
        ! PROCESS-DIMB-COMMAND
        ! LSCI=.T. -> use eispack routines for diagonalization
        ! LSCI=.F. -> use local routines for diagonalization
        !
        QCMPCT=.TRUE.
        LSCI=.FALSE.
        LSCI=(INDXA(COMLYN,COMLEN,'SCIL').GT.0)
        LBIG=(INDXA(COMLYN,COMLEN,'BIG').GT.0)
        QDISK=(INDXA(COMLYN,COMLEN,'DISK').GT.0)
        QDW=(INDXA(COMLYN,COMLEN,'DWIN').GT.0)
        PARD=GTRMI(COMLYN,COMLEN,'PARD',200)
        PARDIM=PARD*3
        ! IUNPAR=GTRMI(COMLYN,COMLEN,'UNPA',-1)
        ! IUNBAS=GTRMI(COMLYN,COMLEN,'IUNB',-1)
        ! IUNTRN=GTRMI(COMLYN,COMLEN,'IUNT',-1)
        !
        ! Unit for restart modes
        !
        IUNRMD=GTRMI(COMLYN,COMLEN,'IUNR',-1)
        !
        ! Unit where to save the modes after each iteration
        !
        IUNMOD=GTRMI(COMLYN,COMLEN,'IUNM',-1)
        IF(IUNMOD.EQ.-1) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,413)
413        FORMAT(' Missing unit for saving modes')
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !
        ! unit where to read the compact second derivatives for restart
        ! if ISDR=-1 no restart, all the second derivatives are recalculated
        ! ISDR=GTRMI(COMLYN,COMLEN,'ISDR',-1)
        ! unit where to save the second derivatives
        ! ISDW=GTRMI(COMLYN,COMLEN,'ISDW',-1)
        !
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NAT3)
        NADD=GTRMI(COMLYN,COMLEN,'NADD',0)
        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        IF(NFREQ.GT.0) WRITE(OUTU,247) NFREQ
        IF(NFRE.GT.NAT3) NFRE=NAT3
        IF(NFRE.GT.NNMDS) NFRE=NNMDS
        IF((NFRE+NADD).GT.NAT3 .OR. NADD.LT.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,632) NFRE,NADD,NAT3
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !
        ! NFREG : maximum allowed number of frequencies
        !
        NFREG=NFRE
        IF(NFRE.LE.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,249)
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        NDIM=NAT3

        CALL ALLOCATE_C_PAIR_LIST()

        CALL NBLIST(X,Y,Z,NATOM,CTOFNB,PINBCM,PJNBCM,MNBCMP, &
             LENCMP)

        call ALLOCATE__C_2ND_DERIVATIVES()

        !
        ! Read the residue numbering for the partitioning of the
        ! molecule
        !
        CALL PARTIT(NAT3,NFREG,NPARMX,NPAR,ATMPAR,PARDIM)
        !
        ! maximum number of modes for a block
        !
        PARNMD=PARDIM+3
        PARFRQ=GTRMF(COMLYN,COMLEN,'CUTF',HUNDRD)
        CUTF1=GTRMF(COMLYN,COMLEN,'CTF1',FIFTY)
        ITMX=GTRMI(COMLYN,COMLEN,'ITER',10)
        IF(LBIG.AND.(ITMX.NE.0)) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,415)
415        FORMAT(' BIG keyword can only be used with ITER 0')
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        IF(LBIG) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,417) IUNMOD
417        FORMAT(/' NMDIMB: Basis vectors will not be ', &
                'kept in memory, but saved on unit',I5/)
        ENDIF
        SAVF=GTRMI(COMLYN,COMLEN,'SAVF',-1)
        TOLDIM=GTRMF(COMLYN,COMLEN,'TOLE',PT05)
        DDVALM=GTRMF(COMLYN,COMLEN,'STRE',ZERO)

        CALL ALLOCATE_NORMAL_MODES()

        call FILDDM(PARDDM,AMASS,NATOM,LNOMA)
        PARDDE(1:PARNMD)=zero

        call ALLOCATE_2ND_DERIVATIVES()

        JSPACE=(PARDIM+1)*8
        call chmalloc('vibran.src','VIBRAN','DDS',JSPACE,crl=DDS)

        CALL NMDIMB(X,Y,Z,NAT3, &
             BNBND,BIMAG,LNOMA,AMASS,DDS,DDSCR, &
             PARDDV,DDV,DDM,PARDDF,DDF, &
             PARDDE,DDEV,DD1BLK,DD1BLL,NADD, &
             LRAISE,PDD1CM,PINBCM,PJNBCM,NPAR, &
             ATMPAR,ATMPAS,BLATOM,PARDIM,NFREG,NFRET,PARFRQ,CUTF1, &
             ITMX,TOLDIM,IUNMOD,IUNRMD,LBIG,LSCI,ATMPAD,SAVF, &
             NBOND,IB,JB,DDVALM)
        NFREQ=NFRET
        QCMPCT=.FALSE.

        IF(QDISK) THEN
           call chmdealloc('vibran.src','VIBRAN','PDD1CM', &
                6+LENDSK*9,crl=PDD1CM)
        ELSE
           call chmdealloc('vibran.src','VIBRAN','PDD1CM', &
                NATOM*6+LENCMP*9,crl=PDD1CM)
        ENDIF

        call chmdealloc('vibran.src','VIBRAN','PINBCM',NATOM,intg=PINBCM)
        call chmdealloc('vibran.src','VIBRAN','PJNBCM',NATOM*MNBCMP,intg=PJNBCM)

        call chmdealloc('vibran.src','VIBRAN','DD1BLK',ISPBLK,crl=DD1BLK)
        call chmdealloc('vibran.src','VIBRAN','DD1BLL',ISPBLK,crl=DD1BLL)

        call chmdealloc('vibran.src','VIBRAN','DDS',JSPACE,crl=DDS)

        call chmdealloc('vibran.src','VIBRAN','PARDDV',PARNMD*PARDIM,crl=PARDDV)
        call chmdealloc('vibran.src','VIBRAN','PARDDM',NATOM,crl=PARDDM)
        call chmdealloc('vibran.src','VIBRAN','PARDDF',PARNMD,crl=PARDDF)
        call chmdealloc('vibran.src','VIBRAN','PARDDE',PARNMD,crl=PARDDE)

#endif /*  DIMB*/
        !=======================================================================
     ELSE IF(WRD.EQ.'QUAS') THEN
        ! PROCESS-QUASI-COMMAND
        !
        ! modified by Nathan Desdouits and Arnaud Blondel, 4/2012
        !
        ! parsed arguments :
        !
        !    NOTP : do not use temperature. Best used for PCA
        !           computations.
        !
        !    MEAN : use the mean structure of the trajectory as
        !           the reference structure instead of COMP
        !
        !    NORM : normalize the modes after diagonalization
        !           (useful when projecting afterward)
        !
        !    AUTO : this is the switch to specify in which space
        ! or CORD*  to compute the covariance matrix. CORD selects
        ! or STPD   coordinate space (secular matrix), STPD selects
        !           snapshot space, AUTO selects the space for
        !           which the covariance matrix is smallest. In
        !           general, you'll want to avoid STPD, as it
        !           will always have equal or worse performance than
        !           the AUTOmatic choice.
        !           This switch has been added as the results of
        !           the diagonalization is indepedent of the choosen
        !           space (except for numerical imprecisions), but
        !           diagonalization is faster in the smallest space.
        !           Note that the use of the dynamic snapshot space
        !           (STPD) disables the THERmo analysis, as the
        !           required secular matrix isn't computed.
        !
        !    Note : you may want to use VIBRAN's NOMAss option to
        !           perform a PCA. e.g:
        !               > NOMAss
        !               > OPEN READ UNIT 1 NAME ....
        !               > QUAS FIRS 1 NOTP MEAN NORM AUTO
        !               > OPEN READ UNIT 1 NAME ....
        !               > PROJ TRAJ FIRS 1 NONO
        !
        !--------------------------------------------------------------
        !
        ! variables: | signification:
        !            |
        !   LNOTP    | logicals, corresponding to the
        !   LNORM    | switches explained above.
        !   LMEAN    |
        !            |
        !   LAUTOD   | logicals, corresponding respectively
        !   LSSPAD   | to the AUTO, STPD and CORD switches
        !   LCSPAD   |
        !            |
        !            |
        !   NDIM     | integer, corresponds to the length of the edge of
        !            | the correlation matrix that will be computed.
        !            | Can be equal to NNMDS, 3*NSET1 or NCOORD (the
        !            | number of dynamic steps in the given trajectory).
        !            | If LCSPAD is true, equal to min(NNMDS,3*NSET1).
        !            | If LSSPAD is true, equal to min(NNMDS,NCOORD).
        !            | If LAUTOD is true, equal to
        !            |  min(NNMDS,NCOORD,3*NSET1).
        !            |
        !   NDIM2    | integer, length of the other dimension.
        !            | NCOORD if in CORD, 3*NSET1 if in STPD.
        !            |
        !   NBEGN(2) | integers, used to estimate the number of dynamic
        !   NSTOP(2) | steps that will be read, in order to infer the
        !   NSKIP    | correct dimensions when LAUTOD is true.
        !            |
        !
        !       other variables were already there in the original code.
        !

        !
        ! parse keywords
        !
        LNOTP=(INDXA(COMLYN,COMLEN,'NOTP').GT.0)
        RTEMP=GTRMF(COMLYN,COMLEN,'TEMP',FMARK)
        IF(.NOT.LNOTP)THEN
           IF(RTEMP.LE.0.0) CALL WRNDIE(-2,'<VIBRAN>','BAD TEMPERATURE')
        ENDIF
        IF(RTEMP.LE.0.0) RTEMP=-1.
        LMEAN=(INDXA(COMLYN,COMLEN,'MEAN').GT.0)
        LNORM=(INDXA(COMLYN,COMLEN,'NORM').GT.0)
        LAUTOD=INDXA(COMLYN,COMLEN,'AUTO').GT.0
        LSSPAD=INDXA(COMLYN,COMLEN,'STPD').GT.0
        LCSPAD=INDXA(COMLYN,COMLEN,'CORD').GT.0
        IF(.NOT.((LAUTOD.OR.LSSPAD).OR.LCSPAD)) THEN
           LCSPAD=.TRUE.
        ELSE IF((LCSPAD.AND.LSSPAD).OR. &
                (LAUTOD.AND.LSSPAD).OR. &
                (LCSPAD.AND.LAUTOD)) THEN
           CALL WRNDIE(-2,'<VIBRAN>', &
            'AUTO, STPD AND CORD ARE MUTUALLY EXCLUSIVE')
        ENDIF

        !
        ! parse trajectory specs
        !
        if (allocated(IND1)) then
           call chmrealloc('vibran.src','VIBRAN','IND1',NATOM,intg=IND1)
        else
           call chmalloc('vibran.src','VIBRAN','IND1',NATOM,intg=IND1)
        endif
        CALL SELCTA(COMLYN,COMLEN,IND1,X,Y,Z,WMAIN,.TRUE.)
        CALL MAKIND(NATOM,IND1,IND1,NSET1)
        CALL TRJSPC(COMLYN,COMLEN,NUNIT,IFIRST,NBEGN,NSKIP,NSTOP)
        NDIM=3*NSET1

        !
        ! estimate the number of dynamic steps that will be read
        !
        NSTOP2=NSTOP
        NBEGN2=NBEGN
        IF(NSTOP2.EQ.0) THEN
           DO I=IFIRST,IFIRST+NUNIT-1
              CALL GTICNT(I,HDR,ICNTRL,.FALSE.,.TRUE.,.FALSE.)
              NSTOP2=NSTOP2+ICNTRL(1)
           ENDDO
        ENDIF
        IF(NBEGN2.EQ.0) THEN
           NBEGN2=1
        ENDIF
        NCOORD=(NSTOP2+1-NBEGN2)/NSKIP
        IF(NCOORD.LE.1)THEN
           NCOORD=NNMDS
        ENDIF

        NDIM2=NDIM

        !
        ! choose the space in which to perform the calculations.
        !
        LSSPAD=LSSPAD.OR.(LAUTOD.AND.NDIM.GT.NCOORD)
        IF(LSSPAD) THEN
           NDIM=NCOORD
           IF(LAUTOD.AND.PRNLEV.GE.2) THEN
              WRITE(OUTU,559)
559     FORMAT('SNAPSHOT SPACE HAS BEEN AUTOMATICALLY SELECTED')
           ENDIF
        ELSE
           IF(LAUTOD.AND.PRNLEV.GE.2) THEN
              WRITE(OUTU,560)
560     FORMAT('COORDINATE SPACE HAS BEEN AUTOMATICALLY SELECTED')
           ENDIF
        ENDIF

        !
        ! sanity checks for the matrix size
        !
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NDIM)
        NADD=GTRMI(COMLYN,COMLEN,'NADD',0)
        FCUT=GTRMF(COMLYN,COMLEN,'FCUT',PT0001)
        IF(NFREQ.GT.0 .AND. PRNLEV.GE.2) WRITE(OUTU,247) NFREQ
        IF(NFRE.GT.NDIM) NFRE=NDIM
        IF(NFRE.GT.NNMDS) NFRE=NNMDS
        IF((NFRE+NADD).GT.NDIM .OR. NADD.LT.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,632) NFRE,NADD,NDIM
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF

        NFREQ=NFRE
        IF(NFRE.LE.0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,249)
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        LTHERMO=(INDXA(COMLYN,COMLEN,'THER').GT.0)
        LRESI = .FALSE.
        IF(LTHERMO)THEN

           call vib_get_mode_spec()

           LRESI=(INDXA(COMLYN,COMLEN,'RESI').GT.0)
        ENDIF

        !
        ! allocate and process the quasi, using QUASI if CORD is selected,
        ! and SQUAS if STPD is selected.
        !
        call vib_allocate_diag()
        IF(.NOT.LSSPAD)THEN
           CALL QUASI(X,Y,Z,NATOM,NDIM,NFREQ,RTEMP, &
                DDV,DDM,DDF,DDEV,DD5,DDS,DDSCR,NADD, &
                XNEW,YNEW,ZNEW,NUNIT,IFIRST,NSKIP,NBEGN,NSTOP,LTHERMO, &
                ISTRT,ISTOP,NSET1,IND1,LRESI,FCUT, &
                LNOTP,LNOMA,LNORM,LMEAN,NCOORD)
        ELSE
           CALL SQUAS(X,Y,Z,NATOM,NDIM,NDIM2,NFREQ,RTEMP, &
                DDV,DDM,DDF,DDEV,DD5,DDS,DDSCR,NADD, &
                XNEW,YNEW,ZNEW,NUNIT,IFIRST,NSKIP,NBEGN,NSTOP, &
                ISTRT,ISTOP,NSET1,IND1,FCUT, &
                LNOTP,LNOMA,NCOORD,LNORM,LMEAN)
        ENDIF
        call chmdealloc('vibran.src','VIBRAN','DDS',JSPACE,crl=DDS)

        ! mkg 2009: leave IND1 allocated in case next cmd is PRINt
        call FILDDM(DDM,AMASS,NATOM,LNOMA)
        !=======================================================================
     ELSE IF(WRD.EQ.'RBQU') THEN
        ! PROCESS-REDUCE-QUASI-COMMAND
        !
        RTEMP=GTRMF(COMLYN,COMLEN,'TEMP',FMARK)
        IF(RTEMP.LE.0.0) CALL WRNDIE(-2,'<VIBRAN>','BAD TEMPERATURE')
        CALL TRJSPC(COMLYN,COMLEN,NUNIT,IFIRST,NBEGN,NSKIP,NSTOP)
        IF (NFREQ.GT.0 .AND. PRNLEV.GE.2) WRITE(OUTU,247) NFREQ

        IUNBAS=GTRMI(COMLYN,COMLEN,'IUNB',-1)
        IUNTRN=GTRMI(COMLYN,COMLEN,'IUNT',-1)
        ISTRT=1
        ISTOP=99999999
        CALL RDNMD(.FALSE.,NFREQ,NNMDS,NAT3,NDIM, &
             DDV,DDSCR,DDF,DDEV, &
             IUNBAS,.FALSE.,ISTRT,ISTOP)
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NDIM)
        IF(NFRE.GT.NDIM) NFRE=NDIM
        IF(NDIM.GT.NNMDS) THEN
           CALL WRNDIE(-1,'<VIBRAN>', &
                'Not enough space to hold the basis. Allocate more space.')
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        IF(NFRE.GT.NNMDS) NFRE=NNMDS
        NFREQ=NFRE
        call chmalloc('vibran.src','VIBRAN','DDV2',NFREQ*NDIM,crl=DDV2)

        call vib_allocate_diag()

        ! get extra space if needed to hold reduce basis eigenvectors
        NADD=0

        CALL RBQUAS(XNEW,YNEW,ZNEW,NAT3,X,Y,Z,NUNIT,IFIRST,NSKIP, &
             IUNBAS,IUNTRN,NDIM,NFREQ,AMASS,DDV,DDM, &
             DDF,DDEV,DDSCR,DD5, &
             DDS,DDV2,NADD,RTEMP,NBEGN,NSTOP)
        call chmdealloc('vibran.src','VIBRAN','DDS',JSPACE,crl=DDS)
        call FILDDM(DDM,AMASS,NATOM,LNOMA)
        call chmdealloc('vibran.src','VIBRAN','DDV2',NFREQ*NDIM,crl=DDV2)
        !=======================================================================
     ELSE IF(WRD.EQ.'FLUC') THEN
        ! PROCESS-FLUCTUATION-COMMAND
        call vib_check_nfreq()

        call vib_get_mode_spec()

        call vib_get_magnitude_spec()

        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
        LQUAN=(INDXA(COMLYN,COMLEN,'QUAN').GT.0)
        LVECT=(INDXA(COMLYN,COMLEN,'VERB').GT.0)

        LATOM=(INDXA(COMLYN,COMLEN,'ATOM').GT.0)
        LIC=(INDXA(COMLYN,COMLEN,'IC').GT.0)
        LUSER=(INDXA(COMLYN,COMLEN,'USER').GT.0)
        CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
        CALL VIBFLU(ISTRT,ISTOP,IUNIT,NAT3,DDV,DDM, &
             DDF,DDEV,DDSCR,LNOMA,LNONO, &
             ITYPE,RTYPE,TFREQ,LQUAN,LVECT,ISLCT, &
             X,Y,Z,XCOMP,YCOMP,ZCOMP,XNORM,YNORM,ZNORM,AMASS, &
             LATOM,LIC,LUSER,NRES,IBASE,ATYPE, &
             icr_struct%lenic, &
             icr_struct%B1ic,icr_struct%B2ic, &
             icr_struct%T1ic,icr_struct%T2ic, &
             icr_struct%PIC, icr_struct%IAR, &
             icr_struct%JAR, icr_struct%KAR, &
             icr_struct%LAR, icr_struct%TAR)

        !=======================================================================
     ELSE IF(WRD.EQ.'PAFL') THEN
        ! PROCESS-PAFLU-COMMAND
        !
        ! Added October 1985 by B. Tidor.  Calculates and optionally
        ! diagonalizes positional fluctuation matrix for centers
        ! (atoms or groups of atoms).  Prints cartesian or principal
        ! axis fluctuations and unit vectors prepresenting principal
        ! axes for each center.
        !
        call vib_check_nfreq()

        call vib_get_mode_spec()

        call vib_get_magnitude_spec()

        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
        LQUAN=(INDXA(COMLYN,COMLEN,'QUAN').GT.0)
        LVECT=(INDXA(COMLYN,COMLEN,'VERB').GT.0)
        LATOM=(INDXA(COMLYN,COMLEN,'ATOM').GT.0)
        LUSER=(INDXA(COMLYN,COMLEN,'USER').GT.0)
        LNOMS=(INDXA(COMLYN,COMLEN,'MASS').LE.0)
        LPAX=(INDXA(COMLYN,COMLEN,'COOR').LE.0)
        LGROU=(INDXA(COMLYN,COMLEN,'GROU').GT.0)
        LSAVE=(INDXA(COMLYN,COMLEN,'SAVE').GT.0)
        LCONT=(INDXA(COMLYN,COMLEN,'CONT').GT.0)
        CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

        CALL VPAFL0(ISTRT,ISTOP,IUNIT,NAT3,DDV,DDM, &
             DDF,DDEV,DDSCR,LNOMA,LNONO, &
             ITYPE,RTYPE,TFREQ,LQUAN,LVECT,ISLCT, &
             X,Y,Z,XCOMP,YCOMP,ZCOMP,XNORM,YNORM,ZNORM,AMASS, &
             LATOM,LUSER,LPAX,LGROU,LNOMS, &
             LSAVE,LCONT,NRES,IBASE,ATYPE)

        !=======================================================================
     ELSE IF(WRD.EQ.'EDIT') THEN
        ! PROCESS-EDIT-COMMAND

        !! BTM -- don't prematurely process the atom selection
        !!CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
        WRD=NEXTA4(COMLYN,COMLEN)
        LNONO=(INDXA(COMLYN,COMLEN,'NONO').GT.0)
        IF(WRD.EQ.'INCL') THEN

           call vib_get_magnitude_spec()

           LORTH=(INDXA(COMLYN,COMLEN,'ORTH').GT.0)

           call vib_get_new_mode()

           IDEST=GTRMI(COMLYN,COMLEN,'TO',NFREQ+1)
           IF(IDEST.GT.NNMDS) THEN
              CALL WRNDIE(2,'<VIBRAN>','NO SPACE IN WHICH TO APPEND')
           ENDIF
           NFREQ=MAX(NFREQ,IDEST)
           CALL APPENM(IDEST,NFREQ,NAT3,XNEW,YNEW,ZNEW,DDV, &
                DDM,DDF,DDSCR,DDEV, &
                ITYPE,RTYPE,LNOMA,LNONO,LORTH,TFREQ)
        ELSE IF(WRD.EQ.'ADD ') THEN

           call vib_get_destination()

           call vib_get_source()

           FACD=1.0
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'ADD ',0,0)
        ELSE IF(WRD.EQ.'MOVE') THEN

           call vib_get_destination()

           call vib_get_source()

           FACD=0.0
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'ADD ',0,0)
        ELSE IF(WRD.EQ.'MULT') THEN

           call vib_get_source()

           IDEST=ISOURC
           FACD=0.0
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'ADD ',0,0)
        ELSE IF(WRD.EQ.'SET ') THEN

           call vib_get_source()

           IDEST=ISOURC
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'SET ',0,0)
        ELSE IF(WRD.EQ.'NORM') THEN

           call vib_get_source()

           IDEST=ISOURC
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'NORM',0,0)
        ELSE IF(WRD.EQ.'ZERO') THEN

           call vib_get_source()

           ISTRT=ISOURC
           ISTOP=GTRMI(COMLYN,COMLEN,'THRU',ISTRT)
           IF(ISTOP.GT.NNMDS) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,235) ISTOP,NNMDS
              ISTOP=NNMDS
           ENDIF
           NFREQ=MAX(NFREQ,ISTOP)
           ISOURC=0
           FACS=0.0
           FACD=0.0
           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           DO I=ISTRT,ISTOP
              CALL MANMOD(NAT3,DDV,ISLCT,I,ISOURC, &
                   FACD,FACS,'ZERO',DDEV,DDF)
           ENDDO
        ELSE IF(WRD.EQ.'REMO') THEN

           call vib_get_mode_spec()

           call vib_get_magnitude_spec()

           call vib_get_new_mode()

           CALL REMONM(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW,DDM, &
                DDSCR,DDV,LNONO)
        ELSE IF(WRD.EQ.'SHAK') THEN

           call vib_get_mode_spec()

           CALL SHAKNM(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW,X,Y,Z,DDM, &
                DDV,LNONO,AMASS,.TRUE.,IMOVE)
        ELSE IF(WRD.EQ.'DELE') THEN

           call vib_get_mode_spec()

           CALL DELNRM(ISTRT,ISTOP,NFREQ,DDV,DDEV, &
                DDF,NAT3)
        ELSE IF(WRD.EQ.'ORTH') THEN
           IF(LNONO) CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
           LPURG=(INDXA(COMLYN,COMLEN,'PURG').GT.0)

           call vib_get_mode_spec()

           TOL=GTRMF(COMLYN,COMLEN,'TOL',TENM5)
           CALL ORTHNM(ISTRT,ISTOP,NFREQ,DDV,NAT3,LPURG,TOL)
        ELSE IF(WRD.EQ.'COPY') THEN

           call vib_get_destination()

           call vib_get_source()

           CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
           CALL MANMOD(NAT3,DDV,ISLCT, &
                IDEST,ISOURC,FACD,FACS,'COPY',DDEV,DDF)
        ELSE
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !=======================================================================
     ELSE IF(WRD.EQ.'FILL') THEN
        ! PROCESS-FILL-COMMAND

        call vib_get_mode_spec()

        call vib_get_magnitude_spec()

        LAPPE=(INDXA(COMLYN,COMLEN,'APPE').GT.0)
        WRD=NEXTA4(COMLYN,COMLEN)
        CALL SETCMP(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW, &
             DDV,DDM,DDSCR,DDEV, &
             ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT)
        IF(WRD.EQ.'DIFF') THEN
           IF(LAPPE) THEN
              DO I=1,NATOM
                 XCOMP(I)=XCOMP(I)+XNEW(I)
                 YCOMP(I)=YCOMP(I)+YNEW(I)
                 ZCOMP(I)=ZCOMP(I)+ZNEW(I)
              ENDDO
           ELSE
              DO I=1,NATOM
                 XCOMP(I)=X(I)+XNEW(I)
                 YCOMP(I)=Y(I)+YNEW(I)
                 ZCOMP(I)=Z(I)+ZNEW(I)
              ENDDO
           ENDIF
        ELSE IF(WRD.EQ.'COMP') THEN
           IF(LAPPE) THEN
              DO I=1,NATOM
                 XCOMP(I)=XCOMP(I)+XNEW(I)
                 YCOMP(I)=YCOMP(I)+YNEW(I)
                 ZCOMP(I)=ZCOMP(I)+ZNEW(I)
              ENDDO
           ELSE
              DO I=1,NATOM
                 XCOMP(I)=XNEW(I)
                 YCOMP(I)=YNEW(I)
                 ZCOMP(I)=ZNEW(I)
              ENDDO
           ENDIF
        ELSE
           CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
        ENDIF
        !=======================================================================
     ELSE IF(WRD.EQ.'PROJ') THEN
        ! PROCESS-PROJ-COMMAND
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        WRD=CURRA4(COMLYN,COMLEN)
        !
        ! parse arguments:
        !
        !   TRAJ : project each frame of a trajectory on the
        !          specified modes.
        !
        !      WRITe  : write the projection on the last 4 modes
        !               as atomic coordinates in a coordinate file
        !               (pdb, card, file...). For each structure of
        !               the trajectory corresponds a different atom.
        !               It is useful to project a trajectory onto
        !               its 4 principal modes and view it using a
        !               molecular visualization program. 4th mode
        !               values are stored in the b-factor column.
        !
        !      SPIRal : if WRITe is specified, prepend atomic
        !               coordinates ressembling a spiral before
        !               the actual projections. This is useful to
        !               trick some molecular visualization programs
        !               (e.g: VMD) into linking consecutive atoms
        !               with bonds, so that the 3D projections can
        !               appear as a continuous line.
        !
        !   other first keyword : perform original PROJ command
        !
        IF(WRD.EQ.'TRAJ') THEN
           WRD=NEXTA4(COMLYN,COMLEN)
           CALL TRJSPC(COMLYN,COMLEN,NUNIT,IFIRST,NBEGN,NSKIP,NSTOP)
           LWRIT=(INDXA(COMLYN,COMLEN,'WRIT').GT.0)
           LSPIR=(INDXA(COMLYN,COMLEN,'SPIR').GT.0)
           call vib_get_mode_spec()
           call vib_get_magnitude_spec()
           CALL TRJDOT(ISTRT,ISTOP,NAT3, &
                XCOMP,YCOMP,ZCOMP,XNEW,YNEW,ZNEW, &
                DDV,DDM,DDSCR,DDF,LNONO,.false.,LNOMA, &
                NUNIT,IFIRST,NBEGN,NSKIP,NSTOP, &
                LWRIT,COMLYN,COMLEN,LSPIR)
        ELSE

           call vib_get_new_mode()

           call vib_get_mode_spec()

           call vib_get_magnitude_spec()

           CALL COMDOT(ISTRT,ISTOP,NAT3,XNEW,YNEW,ZNEW, &
                DDV,DDM,DDSCR,DDF, &
                LNONO,.FALSE.)
        ENDIF

        !=======================================================================
     ELSE IF(WRD.EQ.'OPTI') THEN
        ! PROCESS-MODE OPTIMIZATION COMMAND
        !
        ! parse arguments:
        !
        !   no argument: Does a Newton Raphson step along the modes 
        !
        !   ENER Calculates dU to next minimum + vibrational entropy + Jacobian 
        !
        !   PARA Uses force constants from parameter file to calculate 
        !   dU to next minimum + vibrational entropy + Jacobian for CBON and CANG modes
        !
        !   CMPN add the atom displacements to the comparison coordinate set
        !
        !   TMPR sets the temperature for the Boltzmann constant
        !
        !   IUNO saves the total dG_cons, energy change, vibrational entropy 
        !   and Jacobian entropy to the specified unit   
        !

        call vib_get_new_mode() ! GK Why is it necessary to add a new mode here?
        call vib_get_mode_spec()
        call vib_get_magnitude_spec()

        IUNO = GTRMI(COMLYN,COMLEN,'IUNO',OUTU)
        LENER=(INDXA(COMLYN,COMLEN,'ENER').GT.0)
        LPARA=(INDXA(COMLYN,COMLEN,'PARA').GT.0)
        LCOMP=(INDXA(COMLYN,COMLEN,'CMPN').GT.0)
        RTEMP=GTRMF(COMLYN,COMLEN,'TMPR',3.0E+2)
        
        IF(LCOMP) THEN
           CALL MODEOPTI(ISTRT,ISTOP,IUNO,NAT3,NATOM,NNMDS,RTEMP,XNEW,YNEW,ZNEW, &
                XCOMP,YCOMP,ZCOMP,DDV,DDM,DDSCR,DDEV,DDF,NFJAC, &
                FJAC,ANGA,LENER,LPARA)
        ELSE
           CALL MODEOPTI(ISTRT,ISTOP,IUNO,NAT3,NATOM,NNMDS,RTEMP,XNEW,YNEW,ZNEW, &
                X,Y,Z,DDV,DDM,DDSCR,DDEV,DDF,NFJAC,FJAC,ANGA, &
                LENER,LPARA)           
        ENDIF

         !=======================================================================
     ELSE IF(WRD.EQ.'RAYL') THEN
        ! PROCESS-RALEIG-COMMAND

        call vib_get_mode_spec()

        LSAVE=(INDXA(COMLYN,COMLEN,'SAVE').GT.0)

        call vib_checkspace()

        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        CALL RALEIG(ISTRT,ISTOP,X,Y,Z,NAT3, &
             BNBND,BIMAG,DDV,DDEV,DDF, &
             DDSCR,LSAVE,DDM)
        !=======================================================================
     ELSE IF(WRD.EQ.'BASI') THEN
        ! PROCESS-BASIS-COMMAND
        CALL REDBAS(COMLYN,COMLEN,NFREQ,NNMDS,NAT3,DDV, &
             DDSCR,DDF,DDEV,DDM, &
             XNEW,YNEW,ZNEW, &
             icr_struct%lenic, &
             icr_struct%IAR, icr_struct%JAR, &
             icr_struct%KAR, icr_struct%LAR, &
             icr_struct%TAR)
        !=======================================================================
     ELSE IF(WRD.EQ.'BLKS') THEN
        ! Added by An Ghysels dec 2007
        ! PROCESS-BLKS-COMMAND
        CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
        WRD=NEXTA4(COMLYN,COMLEN)

        ! initialisation of block information
        IF(WRD.EQ.'INIT') THEN

           CALL MBH_INITBLK(NATOM,NSUBS,NBLK,BLKINFO)

        ELSE IF(WRD.EQ.'ADDB') THEN

           ! add a block
           CALL MBH_ADDBLK(NATOM,NSUBS,NBLK,BLKINFO,ISLCT)

        ENDIF

        !======================================================================
     ELSE IF(WRD.EQ.'MBH') THEN
        ! Added by An Ghysels dec 2007
        ! PROCESS-MBH-COMMAND
        ! Do a subsystem normal mode analysis with MOBILE BLOCK HESSIAN (MBH).
        !
        ! defaults:
        ! nfre   = nat3 = nfreq
        ! nadd   = 0
        ! lentro = false
        ! lfinit = false
        ! lentro = false
        ! lraise = false (no raise of trans/rot)
        ! lnoma  = false (masses not set to 1)
        ! bnbnd  =
        ! bimag  =
        ! ldscf  =
        ! step   =
        !
        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        LNOMA=.FALSE.
        NFRE=NAT3
        NFREQ=NFRE
        NADD=0
        ! NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NAT3)
        ! NADD=GTRMI(COMLYN,COMLEN,'NADD',0)
        ! TODO  Extra checking on number of freqs
        !
        ! Finite derivatives for normal modes
        LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
        STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
        ! Entropy keywords
        LENTRO=(INDXA(COMLYN,COMLEN,'ENTR').GT.0)
        IF(LENTRO) THEN
           ! Temperature
           TK=GTRMF(COMLYN,COMLEN,'TEMP',ROOMT)
           ! Rotational symmetry number, Explanation of sigma:
           ! C.J.Cramer, "Essentials of Comp.Chem.",2002,p.327
           SIGMA=GTRMF(COMLYN,COMLEN,'SIGM',ONE)
           ! Standard state: Tidor and Karplus, J Mol Biol (1994) vol. 238 (3) pp. 405-14
           ! Default is solution state of conentration 1M
           SSTAN=GTRMA(COMLYN,COMLEN,'STAN')
           ! Next is for a test purpose only (for developers)
           UTEST=GTRMI(COMLYN,COMLEN,'TEST',0)
        ENDIF

        CALL MBH(NATOM,NSUBS,NBLK,BLKINFO, &
             X,Y,Z,NAT3,BNBND,BIMAG, &
             NFREQ,AMASS, &
             DDV,DDM,DDF,DDEV, &
             DDSCR,NADD,LRAISE, &
             LFINIT,STEP,LDSCF,LENTRO)

        !======================================================================
     ELSE IF(WRD.EQ.'REDU') THEN

        ! PROCESS-REDUCE-DIAG-COMMAND
        IF (NFREQ.GT.0 .AND. PRNLEV.GE.2) WRITE(OUTU,247) NFREQ
        IUNBAS=GTRMI(COMLYN,COMLEN,'IUNB',-1)
        IUNTRN=GTRMI(COMLYN,COMLEN,'IUNT',-1)
        LRAISE=(INDXA(COMLYN,COMLEN,'RAIS').GT.0)
        LBIG=(INDXA(COMLYN,COMLEN,'BIG').GT.0)
        LFIX=(INDXA(COMLYN,COMLEN,'FIX').GT.0)
        ! QC: UW_06 ADD THE OPTIONS TO DO FINITE DIFFERENCE.
        LFINIT=(INDXA(COMLYN,COMLEN,'FINI').GT.0)
        STEP=GTRMF(COMLYN,COMLEN,'STEP',PT005)
        ! JZ_UW12: For saving DD1 in intervals
        NSAVDD1=GTRMI(COMLYN,COMLEN,'NSDD',3*NATOM)
        QREST=(INDXA(COMLYN,COMLEN,'REST').GT.0)


        IF(.NOT.LFIX) THEN
           ISTRT=1
           ISTOP=99999999
           IF(LBIG) ISTOP=1
           CALL RDNMD(.FALSE.,NFREQ,NNMDS,NAT3,NDIM, &
                DDV,DDSCR,DDF,DDEV, &
                IUNBAS,.FALSE.,ISTRT,ISTOP)
        ELSE
           LBIG=.FALSE.
           NDIM=0
           DO I=1,NATOM
              IF(IMOVE(I).EQ.0) NDIM=NDIM+3
           ENDDO
           IF(NDIM.EQ.0) THEN
              CALL WRNDIE(-2,'<VIBRAN>', &
                   'All atoms are fixed. No normal mode calculation possible')
           ENDIF
        ENDIF
        NFRE=GTRMI(COMLYN,COMLEN,'NFRE',NDIM)
        IF(NFRE.GT.NDIM) NFRE=NDIM
        IF(.NOT.LBIG) THEN
           IF(NDIM.GT.NNMDS) THEN
              CALL WRNDIE(-1,'<VIBRAN>', &
                   'Not enough space to hold the basis. Use BIG or allocate more')
           ENDIF
           IF(NFRE.GT.NNMDS) NFRE=NNMDS
        ENDIF
        NFREQ=NFRE

#if KEY_DIMB==1
        QCMPCT=(INDXA(COMLYN,COMLEN,'CMPA').GT.0)
        IF(QCMPCT) THEN
           call ALLOCATE_C_PAIR_LIST()

           CALL NBLIST(X,Y,Z,NATOM,CTOFNB,PINBCM,PJNBCM, &
                MNBCMP,LENCMP)

           ISPACE=NATOM*6+LENCMP*9
        ELSE
#endif /*  DIMB*/

           ISPACE=(NAT3*(NAT3+1))/2

#if KEY_DIMB==1
        ENDIF  
#endif
        call chmalloc('vibran.src','VIBRAN','DD1',ISPACE,crl=DD1)

        call vib_allocate_diag()

        call chmalloc('vibran.src','VIBRAN','DDV2',NFREQ*NDIM,crl=DDV2)
        ! get extra space if needed to hold reduce basis eigenvectors
        NADD=0
        IF(LBIG) THEN
           call chmalloc('vibran.src','VIBRAN','DDFX',NFREQ,crl=DDFX)
           call chmalloc('vibran.src','VIBRAN','DDEVX',NFREQ,crl=DDEVX)
           CALL RBDIAG(X,Y,Z,NAT3, &
                BNBND,BIMAG,IUNBAS,IUNTRN,NDIM,NFREQ,AMASS, &
                DDV,DDM,DDFX,DDEVX,DDSCR, &
                DD1,DD5,DDS,DDV2,LBIG,NADD, &
                LFIX,IMOVE, &
                LFINIT,STEP,LNOVEC,LDSCF, &
                NSAVDD1,QREST)          ! JZ_UW12
           call chmdealloc('vibran.src','VIBRAN','DDEVX',NFREQ,crl=DDEVX)
           call chmdealloc('vibran.src','VIBRAN','DDFX',NFREQ,crl=DDFX)
        ELSE
           CALL RBDIAG(X,Y,Z,NAT3, &
                BNBND,BIMAG,IUNBAS,IUNTRN,NDIM,NFREQ,AMASS, &
                DDV,DDM,DDF,DDEV,DDSCR, &
                DD1,DD5,DDS,DDV2,LBIG,NADD, &
                LFIX,IMOVE, &
                LFINIT,STEP,LNOVEC,LDSCF, &
                NSAVDD1,QREST)          ! JZ_UW12
        ENDIF
        call chmdealloc('vibran.src','VIBRAN','DDV2',NFREQ*NDIM,crl=DDV2)
        call chmdealloc('vibran.src','VIBRAN','DDS',JSPACE,crl=DDS)
        call chmdealloc('vibran.src','VIBRAN','DD1',ISPACE,crl=DD1)
#if KEY_DIMB==1
        IF(QCMPCT) THEN
           call chmdealloc('vibran.src','VIBRAN','PINBCM',NATOM,intg=PINBCM)
           call chmdealloc('vibran.src','VIBRAN','PJNBCM',NATOM*MNBCMP,intg=PJNBCM)
        ENDIF
#endif /*  DIMB*/
        !=======================================================================
     ELSE IF(WRD.EQ.'EXPL') THEN
        ! PROCESS-EXPL-COMMAND

        call vib_get_mode_spec()

        call vib_get_magnitude_spec()

        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        LCOMP=(INDXA(COMLYN,COMLEN,'COMP').GT.0)
        LADJU=(INDXA(COMLYN,COMLEN,'ADJU').GT.0)
        RBOLT=GTRMF(COMLYN,COMLEN,'BOLT',ZERO)
        RGAUS=GTRMF(COMLYN,COMLEN,'GAUS',ZERO)
        LSHAK=(INDXA(COMLYN,COMLEN,'SHAK').GT.0)
        NGRID=GTRMI(COMLYN,COMLEN,'GRID',3)
        CALL SETCMP(ISTRT,ISTOP,NAT3,XNORM,YNORM,ZNORM, &
             DDV,DDM,DDSCR,DDEV, &
             ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT)

        call chmalloc('vibran.src','VIBRAN','ISKP',NATOM,intg=ISKP)
        NGRID2=NGRID
        call chmalloc('vibran.src','VIBRAN','IESTOR',NGRID2,crl=IESTOR)
        call chmalloc('vibran.src','VIBRAN','IRSTOR',NGRID2,crl=IRSTOR)
        call chmalloc('vibran.src','VIBRAN','IWSTOR',NGRID2,crl=IWSTOR)
        call chmalloc('vibran.src','VIBRAN','IISTOR',NGRID,intg=IISTOR)
        call chmalloc('vibran.src','VIBRAN','XTEMP',NATOM,crl=XTEMP)
        call chmalloc('vibran.src','VIBRAN','YTEMP',NATOM,crl=YTEMP)
        call chmalloc('vibran.src','VIBRAN','ZTEMP',NATOM,crl=ZTEMP)
        IF(LCOMP) THEN
           CALL EXPLNM(XCOMP,YCOMP,ZCOMP,XNEW,YNEW,ZNEW, &
                XNORM,YNORM,ZNORM,XTEMP,YTEMP, &
                ZTEMP,NGRID,BNBND,BIMAG,IESTOR, &
                IRSTOR,IWSTOR,IISTOR,IUNIT, &
                AFACT,LADJU,RBOLT,RGAUS,DDEV,DDF, &
                ISTRT,LSHAK,NATOM,AMASS,IMOVE,ISKP)
        ELSE
           CALL EXPLNM(X,Y,Z,XNEW,YNEW,ZNEW, &
                XNORM,YNORM,ZNORM,XTEMP,YTEMP, &
                ZTEMP,NGRID,BNBND,BIMAG,IESTOR, &
                IRSTOR,IWSTOR,IISTOR,IUNIT, &
                AFACT,LADJU,RBOLT,RGAUS,DDEV,DDF, &
                ISTRT,LSHAK,NATOM,AMASS,IMOVE,ISKP)
        ENDIF
        call chmdealloc('vibran.src','VIBRAN','ZTEMP',NATOM,crl=ZTEMP)
        call chmdealloc('vibran.src','VIBRAN','YTEMP',NATOM,crl=YTEMP)
        call chmdealloc('vibran.src','VIBRAN','XTEMP',NATOM,crl=XTEMP)
        call chmdealloc('vibran.src','VIBRAN','IISTOR',NGRID,intg=IISTOR)
        call chmdealloc('vibran.src','VIBRAN','IWSTOR',NGRID2,crl=IWSTOR)
        call chmdealloc('vibran.src','VIBRAN','IRSTOR',NGRID2,crl=IRSTOR)
        call chmdealloc('vibran.src','VIBRAN','IESTOR',NGRID2,crl=IESTOR)
        call chmdealloc('vibran.src','VIBRAN','ISKP',NATOM,intg=ISKP)
        !=======================================================================
     ELSE IF(WRD.EQ.'PED ') THEN
        ! process potential energy distribution command

        call vib_get_mode_spec()

        call vib_get_magnitude_spec()

        TOL=GTRMF(COMLYN,COMLEN,'TOL',PT0001)
        call chmalloc('vibran.src','VIBRAN','IXBOND',NBOND,crl=IXBOND)
        call chmalloc('vibran.src','VIBRAN','IXTHET',NTHETA,crl=IXTHET)
        call chmalloc('vibran.src','VIBRAN','IXIMPH',NIMPHI,crl=IXIMPH)
#if KEY_CMAP==1
        call chmalloc('vibran.src','VIBRAN','IXCMAP',NCRTERM,crl=IXCMAP)
#endif 
        call chmalloc('vibran.src','VIBRAN','IXPHI',NPHI,crl=IXPHI)
        DO IMOD=ISTRT,ISTOP
           IF(PRNLEV.GE.2) WRITE(OUTU,83) IMOD
83         FORMAT(/' POTENTIAL ENERGY DISTRIBUTION FOR MODE NUMBER',I5)
           CALL SETCMP(IMOD,IMOD,NAT3,XNORM,YNORM,ZNORM,DDV, &
                DDM,DDSCR,DDEV, &
                ITYPE,RTYPE,LNOMA,LNONO,TFREQ,AFACT)

           CALL PED(X,Y,Z,XNORM,YNORM,ZNORM,TOL,IXBOND, &
                IXTHET,IXPHI,IXIMPH &
#if KEY_CMAP==1
                ,IXCMAP &  
#endif
                )
        ENDDO
        call chmdealloc('vibran.src','VIBRAN','IXBOND',NBOND,crl=IXBOND)
        call chmdealloc('vibran.src','VIBRAN','IXTHET',NTHETA,crl=IXTHET)
        call chmdealloc('vibran.src','VIBRAN','IXPHI',NPHI,crl=IXPHI)
        call chmdealloc('vibran.src','VIBRAN','IXIMPH',NIMPHI,crl=IXIMPH)
#if KEY_CMAP==1
        call chmdealloc('vibran.src','VIBRAN','IXCMAP',NCRTERM,crl=IXCMAP)
#endif 
        !=======================================================================
     ELSE IF(WRD.EQ.'THER') THEN
        ! PROCESS-THERMO-COMMAND
        ! Compute Thermodynamic functions

        call vib_get_mode_spec()

        RTEMP=GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
        FCUT=GTRMF(COMLYN,COMLEN,'FCUT',PT0001)
        STEP=30.0D0
        STEP=GTRMF(COMLYN,COMLEN,'STEP',STEP)
        CALL THERMV(ISTRT,ISTOP,DDF,RTEMP,STEP,.TRUE., &
             FCUT,NSMALL)
        !=======================================================================
     ELSE IF(WRD.EQ.'REMA') THEN
        ! PROCESS-REMOVE-ATOMS-FROM-NORMAL-MODES
        CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        IXX=INDXA(COMLYN,COMLEN,'FILE')
        CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
        call chmalloc('vibran.src','VIBRAN','ITROT',NATOM,crl=ITROT)
        CALL REMATM(NFREQ,NAT3,INEED,ISLCT, &
             DDV,DDSCR,AMASS,ITROT)
        LCARD=(INDXA(COMLYN,COMLEN,'CARD').GT.0)
        call vib_check_nfreq()

        call vib_get_mode_spec()

        CALL WRTNMD(LCARD,ISTRT,ISTOP,INEED,DDV,DDSCR, &
             DDEV,IUNIT,ITROT)
        call chmdealloc('vibran.src','VIBRAN','ITROT',NATOM,crl=ITROT)
        !=======================================================================
     ELSE IF(WRD.EQ.'MASS') THEN
        LNOMA=.FALSE.
        CALL FILDDM(DDM,AMASS,NATOM,LNOMA)
        !=======================================================================
     ELSE IF(WRD.EQ.'NOMA') THEN
        LNOMA=.TRUE.
        CALL FILDDM(DDM,AMASS,NATOM,LNOMA)
        !=======================================================================
     ELSE IF(WRD.EQ.'END ') THEN
        ! PROCESS-END-COMMAND
        ! Clean up space for normal modes.
        ! QC:UW_06
        IF (.NOT.LNOVEC) THEN
           call chmdealloc('vibran.src','VIBRAN','DDV',NNMDS*NAT3,crl=DDV)
        ELSE
           call chmdealloc('vibran.src','VIBRAN','DDV',2*NAT3,crl=DDV)
        ENDIF
        call chmdealloc('vibran.src','VIBRAN','DDM',NATOM,crl=DDM)
        call chmdealloc('vibran.src','VIBRAN','DDF',NNMDS,crl=DDF)
        call chmdealloc('vibran.src','VIBRAN','DDEV',NNMDS,crl=DDEV)
        call chmdealloc('vibran.src','VIBRAN','DDSCR',MAX(NAT3,NNMDS),crl=DDSCR)
        call chmdealloc('vibran.src','VIBRAN','ISLCT',NATOM,intg=ISLCT)

        RETURN
        !=======================================================================
     ELSE
        CALL WRNDIE(0,'<VIBRAN>','UNRECOGNIZED COMMAND')
     ENDIF
  ENDDO parse
  !yw
230 FORMAT(/' **** NO SOURCE-MODE SPECIFIED FOR EDIT COMMAND ***'/)
235 FORMAT(/' **** CANNOT ACCESS MODE',I4,'; CURRENT MAXIMUM IS',I4/)
241 FORMAT(/' **** THERE ARE NOT THAT MANY MODES ****',2I6)
243 FORMAT(/' **** ERROR **** IMPROPER MODE SPECIFICATION',2I6)

  RETURN

  !-------- End of subroutine VIBRAN code ------------------------------
  !=====================================================================

  !=================================================================
  !------------ Contained recursive subroutines --------------------

contains

#if KEY_DIMB==1 /*dimb_subs*/

  !-----------------------------------------------------------------------
  ! ALLOCATE-SPACE-FOR-COMPACT-PAIR-LIST
  SUBROUTINE ALLOCATE_C_PAIR_LIST()
    use memory
    call chmalloc('vibran.src','ALLOCATE_C_PAIR_LIST','PINBCM', &
         NATOM,intg=PINBCM)
    call chmalloc('vibran.src','ALLOCATE_C_PAIR_LIST','PJNBCM', &
         NATOM*MNBCMP,intg=PJNBCM)

    return
  END SUBROUTINE ALLOCATE_C_PAIR_LIST

  !--------------------------------------------------------------------
  ! ALLOCATE-SPACE-FOR-COMPACT-SECOND-DERIVATIVES
  SUBROUTINE ALLOCATE__C_2ND_DERIVATIVES()
    use memory
    IF(QDISK) THEN
       call chmalloc('vibran.src','ALLOCATE__C_2ND_DERIVATIVES','PDD1CM', &
            6+LENDSK*9,crl=PDD1CM)
    ELSE
       call chmalloc('vibran.src','ALLOCATE__C_2ND_DERIVATIVES','PDD1CM', &
            NATOM*6+LENCMP*9,crl=PDD1CM)
    ENDIF
    return
  END SUBROUTINE ALLOCATE__C_2ND_DERIVATIVES

  !-----------------------------------------------------------------------
  ! ALLOCATE-SPACE-FOR-BLOCK-NORMAL-MODES
  subroutine ALLOCATE_NORMAL_MODES()
    use memory
    IF(PRNLEV.GE.2) WRITE(OUTU,723) PARNMD,PARDIM
723 FORMAT(/' VIBRAN: Space allocated for',I5,' vectors of length',I6, &
         /'         for diagonalization by partitioning method')
    call chmalloc('vibran.src','ALLOCATE_NORMAL_MODES', &
         'PARDDV',PARNMD*PARDIM,crl=PARDDV)
    call chmalloc('vibran.src','ALLOCATE_NORMAL_MODES', &
         'PARDDM',NATOM,crl=PARDDM)
    call chmalloc('vibran.src','ALLOCATE_NORMAL_MODES', &
         'PARDDF',PARNMD,crl=PARDDF)
    call chmalloc('vibran.src','ALLOCATE_NORMAL_MODES', &
         'PARDDE',PARNMD,crl=PARDDE)
    return
  END subroutine ALLOCATE_NORMAL_MODES

  !-----------------------------------------------------------------------
  ! ALLOCATE-SPACE-FOR-BLOCK-SECOND-DERIVATIVES
  SUBROUTINE ALLOCATE_2ND_DERIVATIVES()
    use memory
    ISPBLK=(PARDIM*(PARDIM+1))/2
    call chmalloc('vibran.src','ALLOCATE_2ND_DERIVATIVES', &
         'DD1BLK',ISPBLK,crl=DD1BLK)
    call chmalloc('vibran.src','ALLOCATE_2ND_DERIVATIVES', &
         'DD1BLL',ISPBLK,crl=DD1BLL)
  END SUBROUTINE ALLOCATE_2ND_DERIVATIVES

#endif /* DIMB (dimb_subs)*/

  subroutine vib_check_nfreq()
    IF (NFREQ.LE.0) THEN
       IF (WRNLEV.GE.2) WRITE(OUTU,382) NFREQ
382    FORMAT(/' **** ERROR **** THERE ARE NO MODES YET',I5/)
       CALL WRNDIE(0,'<VIBRAN>','ILLEGAL INPUT')
    ENDIF
  end subroutine vib_check_nfreq

  subroutine vib_get_new_mode()

    LNOTR=(INDXA(COMLYN,COMLEN,'NOTR').GT.0)
    call GETMOD(X,Y,Z,XNEW,YNEW,ZNEW,XCOMP,YCOMP,ZCOMP,DDSCR,DDM, &
         LNOMA,LNOTR,NNMDS,NFREQ,DDEV,NFJAC,FJAC,ANGA,ISLCT,ERR)

    return
  end subroutine vib_get_new_mode

  !-----------------------------------------------------------------------
  ! TO GET-SOURCE
  subroutine vib_get_source()

    ISOURC=GTRMI(COMLYN,COMLEN,'MODE',-99)
    IF(ISOURC.EQ.-99) THEN
       IF(WRNLEV.GE.2) WRITE(OUTU,230)
230    FORMAT(/' **** NO SOURCE-MODE SPECIFIED FOR EDIT COMMAND ***'/)
       CALL WRNDIE(0,'<VIBRAN>','vib_get_source: assertion failure')
       return
    ENDIF
    FACS=GTRMF(COMLYN,COMLEN,'SCAL',ONE)
  end subroutine vib_get_source

  !-----------------------------------------------------------------------
  ! TO GET-DESTINATION
  subroutine vib_get_destination()

    IDEST=GTRMI(COMLYN,COMLEN,'TO',NFREQ+1)
    IF(IDEST.GT.NNMDS) THEN
       IF(WRNLEV.GE.2) WRITE(OUTU,235) IDEST,NNMDS
235    FORMAT(/' **** CANNOT ACCESS MODE',I4,'; CURRENT MAXIMUM IS',I4/)
       CALL WRNDIE(0,'<VIBRAN>','vib_get_destination: IDEST > NNMDS')
       return
    ENDIF
    NFREQ=MAX(NFREQ,IDEST)
  end subroutine vib_get_destination

  !-----------------------------------------------------------------------
  ! TO GET-MODE-SPEC
  subroutine vib_get_mode_spec()

    ISTRT=GTRMI(COMLYN,COMLEN,'MODE',-99)
    IF(ISTRT.EQ.-99) THEN
       ISTOP=NFREQ
    ELSE
       ISTOP=ISTRT
    ENDIF
    IF(ISTRT.LT.1) ISTRT=1
    ISTOP=GTRMI(COMLYN,COMLEN,'THRU',ISTOP)
    IF(ISTOP.GT.NFREQ) ISTOP=NFREQ
    IF(NFREQ.LT.ISTRT) THEN
       IF(WRNLEV.GE.2) WRITE(OUTU,241) ISTRT,NFREQ
241    FORMAT(/' **** THERE ARE NOT THAT MANY MODES ****',2I6)
       CALL WRNDIE(0,'<VIBRAN>','vib_get_mode_spec: NFREQ < ISTRT')
       return
    ENDIF
    IF(ISTOP.LT.ISTRT) THEN
       IF(WRNLEV.GE.2) WRITE(OUTU,243) ISTRT,ISTOP
243    FORMAT(/' **** ERROR **** IMPROPER MODE SPECIFICATION',2I6)
       CALL WRNDIE(0,'<VIBRAN>','vib_get_mode_spec: ISTOP < ISTRT')
       return
    ENDIF
  end subroutine vib_get_mode_spec

  !-----------------------------------------------------------------------
  ! TO GET-MAGNITUDE-SPEC
  subroutine vib_get_magnitude_spec()

    LNONO=(INDXA(COMLYN,COMLEN,'NONO').GT.0)
    RTEMP=GTRMF(COMLYN,COMLEN,'TEMP',FMARK)
    RKCAL=GTRMF(COMLYN,COMLEN,'KCAL',FMARK)
    RMRMS=GTRMF(COMLYN,COMLEN,'MRMS',FMARK)
    RRMS=GTRMF(COMLYN,COMLEN,'RMS',FMARK)
    RFACT=GTRMF(COMLYN,COMLEN,'FACT',FMARK)
    IF(RTEMP.LE.FMARK.AND.RKCAL.LE.FMARK) THEN
       IF(RRMS.LE.FMARK) THEN
          IF(RMRMS.LE.FMARK) THEN
             IF (RFACT.LE.FMARK) RFACT=1.0
             RTYPE=RFACT
             ITYPE=4
          ELSE
             RTYPE=RMRMS
             ITYPE=5
          ENDIF
       ELSE
          IF(RFACT.GT.FMARK) then
             CALL WRNDIE(0,'<VIBRAN>','vib_get_magnitude_spec: RRMS > FMARK and RFACT > FMARK')
             return
          endif
          RTYPE=RRMS
          ITYPE=3
       ENDIF
    ELSE
       TFREQ=GTRMF(COMLYN,COMLEN,'TFRE',TFREQ)
       IF(RTEMP.LT.0.0) THEN
          IF(RKCAL.LT.0.0) then
             CALL WRNDIE(0,'<VIBRAN>','vib_get_magnitude_spec: RTEMP < 0 and RKCAL < 0')
             return
          endif

          RTYPE=RKCAL
          ITYPE=2
       ELSE
          IF(RKCAL.GE.0.0) then
             CALL WRNDIE(0,'<VIBRAN>','vib_get_magnitude_spec: RTEMP >= 0 and RKCAL >= 0')
             return
          endif
          RTYPE=RTEMP
          ITYPE=1
       ENDIF
    ENDIF
  end subroutine vib_get_magnitude_spec

  !-----------------------------------------------------------------------

  subroutine vib_checkspace()
    IF (NAT3.GT.3600) CALL WRNDIE(-1,'<VIBRAN>', &
         'NAT3 for second derivatives is >3600')
  end subroutine vib_checkspace

  !-----------------------------------------------------------------------
  ! TO ALLOCATE-SPACE-FOR-DIAGONALIZATION
  subroutine vib_allocate_diag()

    IF (NDIM.GT.3600) CALL WRNDIE(-1,'<VIBRAN>', &
         'NDIM for diagonalization is >3600')
    JSPACE=(NDIM+1)*8
    JSP=(NDIM*(NDIM+1))/2
    JSPACE=JSPACE+JSP
    call chmalloc('vibran.src','vib_allocate_diag','DDS',JSPACE,crl=DDS)
    DD5 => DDS(JSPACE-JSP+1:JSPACE)
  end subroutine vib_allocate_diag


END SUBROUTINE VIBRAN

end module vibran_m

