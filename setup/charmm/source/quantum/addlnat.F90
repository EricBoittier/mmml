module linkatom
  use chm_kinds
  implicit none

contains

SUBROUTINE ADDLNAT(OUTUX)
  !
  ! This routine adds a link atom to the PSF.
  !
  !     Command line sytax:
  !       ADDLinkatom  link-atom-name  qm-atom-spec  mm-atom-spec [ CHARge real ]
  !
  !     link-atom-name ::= a four character descriptor starting with QQ.
  !
  !     atom-spec::= {residue-number atom-name}
  !                  { segid  resid atom-name }
  !                  { BYNUm  atom-number     }
  !
  ! The first atom is the QM atom it is bonded to, the second atom is the
  ! classical atom it is mimicking.
  !
  !           By David Chatfield - NIH - February, 1993
  !
  use dimens_fcm
  use exfunc, only: nindx
  use number
  use memory
  use psf
  use rtf,only:atct,natct,armass
  use comand
  use coord
  use param
  use select
  use stream
  use string
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_CADPAC==1 || KEY_GAUSSIAN==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  use gamess_fcm
#if KEY_SCCDFTB==1
  use sccdftb  
#endif
#if KEY_MNDO97==1
  use mndo97   
#endif
#if KEY_SQUANTM==1
  use squantm  
#endif
#endif 
  use chutil
  !
  INTEGER OUTUX
  !
  ! This routine adds a link atom to the PSF.
  !
  !     IQAT   = link atom antecedent
  !     ICAT   = other atom adjacent to link atom
  !     IQGRP  = number of antecedent group to link atom group
  !     IQGRP  = resid of residue containing link atom
  !
  CHARACTER(len=8) QQAT
  INTEGER IPA(3),NIPA
  INTEGER IQGRP,IQADD,IQAT,IQRES,ICAT
  INTEGER I,IAT
  INTEGER I1,J1,NUMBERL,ICBX
  real(chm_real) DELTX,DELTY,DELTZ,COEFF,QQCH
  INTEGER DIM, idum(1)
  integer, allocatable, dimension(:) :: SEGLST,RESLST,GRPLST,MAP,INVMAP,LINB,LIBLO
  !
#if KEY_QUANTUM==0 && KEY_GAMESS==0 && KEY_GAMESSUK==0 && KEY_CADPAC==0 && KEY_GAUSSIAN==0 && KEY_SCCDFTB==0 && KEY_QCHEM==0 && KEY_MNDO97==0 && KEY_SQUANTM==0 && KEY_QTURBO==0 && KEY_G09==0
  CALL WRNDIE(0,'<ADDLNAT>','(S)QUANTUM, CADPAC, MNDO97 or GAMESS code not compiled.') 
#else /**/
  !
  QQAT=NEXTA8(COMLYN,COMLEN)
  !
  IF(QQAT(1:2).NE.'QQ') CALL WRNDIE(-1,'<ADDLNAT>','Please start link atoms with "QQ"')
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_CADPAC==1 || KEY_GAUSSIAN==1 || KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  NQQCHG=NQQCHG+1
  QQCHG(NQQCHG)=GTRMF(COMLYN,COMLEN,'CHAR',-THOSND)
#endif 
  !
  CALL NXTATM(IPA,NIPA,3,COMLYN,COMLEN,idum,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
  !
  IF(NIPA.NE.2) THEN
     CALL WRNDIE(-2,'<ADDLNAT>','Must specify exactly 2 atoms.')
     RETURN
  ENDIF
  !CCC
  !CC      WRITE(OUTUX,'(A,A,F7.3)')'Charge on link atom ',' = ',QQCHG(NQQCHG)
  !CC
  !
  CALL XTRANE(COMLYN,COMLEN,'ADDLNAT')
  !
  IQAT=IPA(1)
  ICAT=IPA(2)
  !
  IF(IQAT.LE.0 .OR. IQAT.GT.NATOM) THEN
     CALL WRNDIE(-2,'<ADDLNAT>','Invalid quantum atom.')
     RETURN
  ENDIF
  IF(ICAT.LE.0 .OR. ICAT.GT.NATOM) THEN
     CALL WRNDIE(-2,'<ADDLNAT>','Invalid classical atom.')
     RETURN
  ENDIF
  !
  IQGRP=GETRES(IQAT,IGPBS,NGRP)
  IQRES=GETRES(IQAT,IBASE,NRES)
  IQADD=IGPBS(IQGRP+1)+1
  !
  ! insert an atom at position IQADD

  DIM=NATOM+2
  call chmalloc('addlnat.src','ADDLNAT','SEGLST',DIM,intg=SEGLST)
  call chmalloc('addlnat.src','ADDLNAT','RESLST',DIM,intg=RESLST)
  call chmalloc('addlnat.src','ADDLNAT','GRPLST',DIM,intg=GRPLST)
  call chmalloc('addlnat.src','ADDLNAT','MAP',DIM,intg=MAP)
  call chmalloc('addlnat.src','ADDLNAT','INVMAP',DIM,intg=INVMAP)
  call chmalloc('addlnat.src','ADDLNAT','LINB',NNB,intg=LINB)
  call chmalloc('addlnat.src','ADDLNAT','LIBLO',DIM,intg=LIBLO)

  CALL INSERTATOM(IQADD,SEGLST,RESLST,GRPLST,MAP,INVMAP,LINB,LIBLO)

  call chmdealloc('addlnat.src','ADDLNAT','SEGLST',size(SEGLST),intg=SEGLST)
  call chmdealloc('addlnat.src','ADDLNAT','RESLST',size(RESLST),intg=RESLST)
  call chmdealloc('addlnat.src','ADDLNAT','GRPLST',size(GRPLST),intg=GRPLST)
  call chmdealloc('addlnat.src','ADDLNAT','MAP',size(MAP),intg=MAP)
  call chmdealloc('addlnat.src','ADDLNAT','INVMAP',size(INVMAP),intg=INVMAP)
  call chmdealloc('addlnat.src','ADDLNAT','LINB',size(LINB),intg=LINB)
  call chmdealloc('addlnat.src','ADDLNAT','LIBLO',size(LIBLO),intg=LIBLO)

  !
  IF(ICAT.GE.IQADD) ICAT=ICAT+1
  !
  CG(IQADD)    = zero
  IBLO(IQADD)  = IBLO(IQADD-1)
  RSCLF(IQADD) = one
#if KEY_WCA==1
  WCA(IQADD)   = one                   
#endif
  IMOVE(IQADD) = 0
  ATYPE(IQADD)  = QQAT
  !
  DO I = 1,NATCT
     IF(ATCT(I)(1:2).EQ.'QQ') THEN
        AMASS(IQADD)=ARMASS(I)
        IF(AMASS(ICAT)-ARMASS(I).GT.2.0) THEN
           AMASS(ICAT)=AMASS(ICAT)-ARMASS(I)
        ELSE
           CALL WRNDIE(-2,'<ADDLNAT>','Classical atom is too small')
        ENDIF
        IAC(IQADD)=I
        GOTO 200
     ENDIF
  ENDDO
  CALL WRNDIE(-3,'<ADDLNAT>','No link atoms in the RTF')
200 CONTINUE
  !
  DO I = NGRP,IQGRP,-1
     IMOVEG(I+1) = IMOVEG(I)
     IGPBS(I+1)  = IGPBS(I)
     IGPTYP(I+1) = IGPTYP(I)
  ENDDO
  IMOVEG(IQGRP+1) = 0
  IGPBS(IQGRP+1)  = IQADD-1
  IGPTYP(IQGRP+1) = 0
  NGRP            = NGRP+1
  !
  IGPBS(NGRP+1) = NATOM
  !
  ! position the new link atom one angstrom from the quantum atom
  ! along the vector from the quantum to the classical atom.
  DELTX = X(ICAT)-X(IQAT)
  DELTY = Y(ICAT)-Y(IQAT)
  DELTZ = Z(ICAT)-Z(IQAT)
  !
  ! Place the link atom based on the distance in the parameter file
  ! (If the parameter is not found, put it at 1A from the QM atom).
  I1=IAC(IQAT)
  J1=IAC(IQADD)
  IF(I1.LE.0.OR.J1.LE.0) THEN
     COEFF=ONE
     CALL WRNDIE(-2,'<ADDLNAT>','No atom type for link atom')
  ELSE
     IF(I1.GT.J1) THEN
        NUMBERL=I1*(I1-1)/2+J1
     ELSE
        NUMBERL=J1*(J1-1)/2+I1
     ENDIF
     ICBX=NINDX(NUMBERL,KCB,NCB)
#if KEY_FLEXPARM==1
     ! there might be better way for this,
     ! but for now we check all bond parameters
     ! in case of flex parameters icbx is 0 from above
     if (QFLXPARM) then
        do i=1,ncb
           if(((cbai(i)==i1).and.(cbaj(i)==j1)).or. &
                ((cbai(i)==j1).and.(cbaj(i)==i1))) icbx=i
        enddo
     endif
#endif
     IF(ICBX.EQ.0) THEN
        COEFF=ONE
        CALL WRNDIE(0,'<ADDLNAT>','No parameter for QM-link atom')
     ELSE
        COEFF=CBB(ICBX)
     ENDIF
     IF(PRNLEV.GE.3) WRITE(OUTUX,93) COEFF
93   FORMAT('ADDLNAT: Link atom placed',F10.5,' A from QM atom.')
  ENDIF
  !
  COEFF = COEFF / SQRT(DELTX*DELTX + DELTY*DELTY + DELTZ*DELTZ)
  X(IQADD) = X(IQAT) + COEFF*DELTX
  Y(IQADD) = Y(IQAT) + COEFF*DELTY
  Z(IQADD) = Z(IQAT) + COEFF*DELTZ
  !
  ! Add bonds connecting the new link atom to both the
  ! QM and MM atom (needed for gamess)
  NBOND    =NBOND+1
  IB(NBOND)=IQAT
  JB(NBOND)=IQADD
  NBOND    =NBOND+1
  IB(NBOND)=ICAT
  JB(NBOND)=IQADD
  !
  ! Add angle connecting the new link atom to QM and MM atoms
  ! The QM atom is the center.  (needed for gamess).  
  NTHETA    =NTHETA+1
  IT(NTHETA)=IQADD
  JT(NTHETA)=IQAT
  KT(NTHETA)=ICAT
  !
  ! Reset the nonbond and image lists.  Print new totals
  CALL PSFSUM(OUTUX)
#endif 
  RETURN
END SUBROUTINE ADDLNAT

SUBROUTINE INSERTATOM(IQADD,SEGLST,RESLST,GRPLST,MAP,INVMAP,LINB,LIBLO)
  !-----------------------------------------------------------------------
  use dimens_fcm
  use number
  use psf
  use genpsf_m, only: atmini
  use modpsf
  use stream
  use mmffm

  INTEGER   IQADD, SEGLST(:), RESLST(:), GRPLST(:)
  INTEGER   MAP(:), INVMAP(:)
  INTEGER   LINB(:), LIBLO(:)
  !
  EXTERNAL EXCH
  INTEGER   INDEX, OFFSET, PATGRP
  INTEGER GROUP,I,II,IPATC,IPT,IDELTA,K,ERRCNT
  INTEGER   J, JJ, KK, LL, NL, PATRES, OLDRES,OLDATM
  LOGICAL   FOUND, QDELET, ERROR, BYPASS, QNEWRS, TT
  INTEGER IRSZ
  real(chm_real)  RX
  !     Temporary mark for deleted atoms, bonds,... in PSF:
  INTEGER, parameter :: MARK=-99999999
  !
  !     Mark for unknown atom coordinates for added atoms:
  !
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ATDN

  if (natomt .gt. natom) then
      call wrndie(-5, '<PATIC>', &
          'ADDLnat command not supported with images.')
  end if

  !
  !     Fill segment list SEGLST with values corresonding to current
  !     PSF.
  !
  DO I=1,NSEG
     DO J=NICTOT(I)+1,NICTOT(I+1)
        SEGLST(IBASE(J)+1:IBASE(J+1))=I
     ENDDO
  ENDDO
  !
  ! Fill residue list RESLST with values corresponding to current PSF.
  !
  DO I=1,NRES
     RESLST(IBASE(I)+1:IBASE(I+1))=I
  ENDDO
  !
  ! Fill group list GRPLST with values corresponding to current PSF
  !
  DO I=1,NGRP
     GRPLST(IGPBS(I)+1:IGPBS(I+1))=I
  ENDDO
  !
  ! This atom should be added.
  !
  IF(NATOM+1.GT.MAXA) THEN
     CALL WRNDIE(-4,'<PATIC>','Max numb. of atoms MAXA exceeded')
     RETURN
  ENDIF
  !
  BYPASS=.FALSE.
  NATOM=NATOM+1
  IAC(NATOM)=0
  AMASS(NATOM)=ZERO
#if KEY_MMFF==1
  if (allocated(AtNum)) AtNum(NATOM) = 0
#endif 
  CG(NATOM)    =zero
  ATYPE(NATOM) =' '
  RSCLF(NATOM) =ONE
#if KEY_WCA==1
  WCA(NATOM)   =ONE                
#endif
  IMOVE(NATOM) =0
  RESLST(NATOM)=RESLST(IQADD-1)
  SEGLST(NATOM)=SEGLST(IQADD-1)
  GRPLST(NATOM)=GRPLST(IQADD-1)
  !
  ! here no explicit nonbonded exclusions are allowed for this atom
  !
  IBLO(NATOM)=NNB
  !
  CALL ATMINI(NATOM,NATOM)
  !
  DO I=1,IQADD
     INVMAP(I)=I
  ENDDO
  DO I=IQADD,NATOM
     INVMAP(I)=I-1
  ENDDO
  INVMAP(IQADD)=NATOM
  DO I=1,NATOM
     MAP(INVMAP(I))=I
  ENDDO
  !
  CALL MAPIC(MAP,INVMAP,SEGLST,RESLST,GRPLST,LINB,LIBLO,MARK,BYPASS)
  !
  RETURN
END SUBROUTINE INSERTATOM

SUBROUTINE RELLNAT(OUTU)
  !
  ! This routine relocates a link atom, placing it 1 Angstrom
  ! away from the QM atom to which it is bonded, along the
  ! vector joining the QM and MM atoms specified.
  !
  !     Command line sytax:
  !       RELLinkatom  link-atom-name  atom-spec  atom-spec
  !
  !     atom-spec::= {residue-number atom-name}
  !                  { segid  resid atom-name }
  !                  { BYNUm  atom-number     }
  !
  ! The first atom is the QM atom to which the link atom is bonded,
  ! the second atom is the classical atom the link atom is mimicking.
  !
  !           By David Chatfield - NIH - June, 1993
  !
  use dimens_fcm
  use number
  use psf
  use comand
  use coord
  use chutil
  use select
  use string
  !
  INTEGER OUTU
  !
  !
  !     IQADD  = link atom
  !     IQAT   = link atom antecedent
  !     ICAT   = other atom adjacent to link atom
  !
  CHARACTER(len=8) QQAT,SID,RID,REN,AC
  INTEGER IPA(2),NIPA
  INTEGER IQADD,IQAT,ICAT,idum(1)
  real(chm_real) DELTX,DELTY,DELTZ,COEFF,DIST
  !
#if KEY_QUANTUM==0 && KEY_GAMESS==0 && KEY_GAMESSUK==0 && KEY_CADPAC==0 && KEY_GAUSSIAN==0 && KEY_SCCDFTB==0 && KEY_QCHEM==0 && KEY_MNDO97==0 && KEY_SQUANTM==0 && KEY_QTURBO==0 && KEY_G09==0
  CALL WRNDIE(0,'<RELLNAT>','(S)QUANTUM, CADPAC, MNDO97 or GAMESS code not compiled.')
#else /**/
  !
  QQAT=NEXTA8(COMLYN,COMLEN)
  !
  CALL NXTATM(IPA,NIPA,2,COMLYN,COMLEN,idum,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
  !
  IF(NIPA.NE.2) THEN
     CALL WRNDIE(-2,'<RELLNAT>','Must specify exactly 2 atoms.')
     RETURN
  ENDIF
  !
  CALL XTRANE(COMLYN,COMLEN,'RELLNAT')
  !
  IQAT=IPA(1)
  ICAT=IPA(2)
  !
  IF(IQAT.LE.0 .OR. IQAT.GT.NATOM) THEN
     CALL WRNDIE(-2,'<RELLNAT>','Invalid quantum atom.')
     RETURN
  ENDIF
  IF(ICAT.LE.0 .OR. ICAT.GT.NATOM) THEN
     CALL WRNDIE(-2,'<RELLNAT>','Invalid classical atom.')
     RETURN
  ENDIF
  !
  CALL ATOMID(IQAT,SID,RID,REN,AC)
  IQADD=GETATN(SID,RID,QQAT,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
  !
  DELTX = X(ICAT)-X(IQAT)
  DELTY = Y(ICAT)-Y(IQAT)
  DELTZ = Z(ICAT)-Z(IQAT)
  DIST = SQRT(DELTX*DELTX + DELTY*DELTY + DELTZ*DELTZ)
  IF(DIST.LT.1.D-9) THEN
     CALL WRNDIE(-2,'<RELLNAT>','QM and MM atoms too close.')
     RETURN
  ENDIF
  COEFF = one / SQRT(DELTX*DELTX + DELTY*DELTY + DELTZ*DELTZ)
  X(IQADD) = X(IQAT) + COEFF*DELTX
  Y(IQADD) = Y(IQAT) + COEFF*DELTY
  Z(IQADD) = Z(IQAT) + COEFF*DELTZ
  !
  ! Reset the nonbond and image lists.  Print new totals
  !      CALL PSFSUM(OUTU)
  !
  !MH07: Moved from the end of this file, because other than QM
  !      may use FINDEL. MSCALE for example.
#endif 
  RETURN
END SUBROUTINE RELLNAT
!
SUBROUTINE FINDEL(NAME,MASS,NUM,ELE,ZNUM,QINIGM)
  !
  ! This routine finds the standard symbol for an atom's element type
  ! on the basis of the atomic mass.  Checking is done to ensure
  ! that the element's standard symbol and the atom's chemical
  ! type begin with the same letter.  DCC 9.12.92
  !
  !   NAME  - Name (type or chemical type) of atom to be typed
  !   MASS  - Mass of atom to be typed
  !   NUM   - Atom number (for error messages)
  !   ELE   - Element name returned   (C*4)
  !   ZNUM  - Element number returned (R*8)
  !   QINIGM- Printout only the first time
  !
  use stream
  use psf,only:ATYPE
  use chutil,only:atomid

  CHARACTER(len=*) NAME,ELE
  real(chm_real) MASS
  INTEGER NUM
  real(chm_real) ZNUM
  LOGICAL QINIGM
  !
  INTEGER IZ
  CHARACTER(len=8) SIDDN,RIDDN,RESDN,ACDN
  CHARACTER(len=1) W
  !
  W=NAME(1:1)
  ELE=NAME
  IZ=0
  !
#if KEY_QCHEM==1 || KEY_QTURBO==1 || KEY_G09==1
  IF(NAME.EQ.'XX')THEN
     ELE=' X'
     IZ=0
     GOTO 500
  ENDIF
#endif 
  !
  IF(MASS .LT.  0.0 ) THEN
     ! bad value   
     CALL WRNDIE(-1,'<FINDEL>','Atomic mass is negative') 
  ! Guanhua: for MM link host
#if KEY_G09==1
      else if ((mass>11.002) .and. (mass<11.004)) then
         ele='C'
         iz=6
         if (w.ne.'C') goto 900
#endif 
  ELSE IF(MASS.LT.  0.8 ) THEN
     ! lone pair 
     IF(NAME.EQ. 'LP') THEN
        CALL WRNDIE(-1,'<FINDEL>','Type lone pair not a QM atom type') 
        IZ=0
     ELSE                      
        IF (NAME(1:2) .EQ. 'QQ') THEN
           ELE=' H'
           IZ=1
           IF(W.NE.'H' .AND. W.NE.'Q') GOTO 900
        ELSE 
           CALL WRNDIE(-1,'<FINDEL>','Atomic mass less than 0.8 but type not lone pair')
           IZ=0
        ENDIF
     ENDIF
  ELSE IF(MASS.LT.  3.5 ) THEN
     ! link atom or some type of hydrogen 
     IF (NAME(1:2) .EQ. 'QQ') THEN
        IZ=1
     ELSE
        ELE=' H'
        IZ=1
     ENDIF
     IF(W.NE.'H' .AND. W.NE.'Q') GOTO 900 
  ELSE IF(MASS.LT.  5.5 ) THEN
     ! helium               
     ELE='HE'
     IZ=2
     IF(W.NE.'H') GOTO 900
  ELSE IF(MASS.LT.  8.5 ) THEN
     ! lithium         
     ELE='LI'
     IZ=3
     IF(W.NE.'L') GOTO 900
  ELSE IF(MASS.LT. 9.8 ) THEN
     ! beryllium
     ELE='BE'
     IZ=4
     IF(W.NE.'B') GOTO 900
  ELSE IF(MASS.LT. 11.5 ) THEN
     ! boron                        
     ELE=' B'
     IZ=5
     IF(W.NE.'B') GOTO 900
  ELSE IF(MASS.LT. 13.5 ) THEN
     ! carbon
     ELE=' C'             
     IZ=6
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT. 15.5 ) THEN
     ! nitrogen
     ELE=' N'
     IZ=7
     IF(W.NE.'N') GOTO 900
  ELSE IF(MASS.LT. 18.5 ) THEN    
     ! oxygen
     ELE=' O'
     IZ=8
     IF(W.NE.'O') GOTO 900  
  ELSE IF(MASS.LT. 19.5 ) THEN    
     ! fluorine
     ELE=' F'
     IZ=9
     IF(W.NE.'F') GOTO 900
  ELSE IF(MASS.LT. 22.4 ) THEN    
     ! neon    
     ELE='NE'
     IZ=10
     IF(W.NE.'N') GOTO 900
  ELSE IF(MASS.LT. 23.5 ) THEN    
     ! sodium
     ELE='NA'
     IZ=11
     !   charmm name is usually SOD
     IF(W.NE.'N' .AND. W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 26.5 ) THEN    
     ! magnesium
     ELE='MG'
     IZ=12
     IF(W.NE.'M') GOTO 900
  ELSE IF(MASS.LT. 27.5 ) THEN    
     ! aluminum
     ELE='AL'
     IZ=13
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT. 30.5 ) THEN    
     ! silicon
     ELE='SI'
     IZ=14
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 31.5 ) THEN    
     ! phosphorus
     ELE=' P'
     IZ=15
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT. 34.5 ) THEN    
     ! sulfur
     ELE=' S'
     IZ=16
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 38.5 ) THEN    
     ! chlorine
     ELE='CL'
     IZ=17
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT. 43.5 ) THEN    
     !        difficult to assign by mass. Look at atom name.
     IZ=19
     IF(W.EQ.'K') THEN
        ! potassium
        ELE=' K'
        IZ=19
     ELSE IF(W.EQ.'A') THEN
        ! argon 
        ELE='AR'
        IZ=18
     ELSE IF(W.EQ.'C') THEN
        ! calcium
        ELE='CA'
        IZ=20
     ELSE
        GOTO 900
     ENDIF
     !
  ELSE IF(MASS.LT. 45.5 ) THEN    
     ! scandium
     ELE='SC'
     IZ=21
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 49.5 ) THEN    
     ! titanium
     ELE='TI'
     IZ=22
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT. 51.5 ) THEN    
     ! vanadium
     ELE=' V'
     IZ=23
     IF(W.NE.'V') GOTO 900
  ELSE IF(MASS.LT. 53.5 ) THEN    
     ! chromium
     ELE='CR'
     IZ=24
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT. 55.5 ) THEN    
     ! manganese
     ELE='MN'
     IZ=25
     IF(W.NE.'M') GOTO 900
  ELSE IF(MASS.LT. 57.5 ) THEN    
     ! iron  
     ELE='FE'
     IZ=26
     IF(W.NE.'F') GOTO 900
  ELSE IF(MASS.LT. 62.5 ) THEN    
     IZ=28
     IF(W.EQ.'N') THEN
        ! nickel
        ELE='NI'
        IZ=28
        ! cobalt
     ELSE IF(W.EQ.'C') THEN
        ELE='CO'
        IZ=27
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT. 64.5 ) THEN    
     ! copper
     ELE='CU'
     IZ=29
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT. 68.5 ) THEN    
     ! zinc  
     ELE='ZN'
     IZ=30
     IF(W.NE.'Z') GOTO 900
  ELSE IF(MASS.LT. 71.5 ) THEN    
     ! gallium
     ELE='GA'
     IZ=31
     IF(W.NE.'G') GOTO 900
  ELSE IF(MASS.LT. 73.5 ) THEN    
     ! germanium
     ELE='GE'
     IZ=32
     IF(W.NE.'G') GOTO 900
  ELSE IF(MASS.LT. 77.5 ) THEN    
     ! arsenic
     ELE='AS'
     IZ=33
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT. 79.5 ) THEN    
     ! selenium
     ELE='SE'
     IZ=34
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 82.5 ) THEN    
     ! bromine
     ELE='BR'
     IZ=35
     IF(W.NE.'B') GOTO 900
  ELSE IF(MASS.LT. 84.5 ) THEN    
     ! krypton
     ELE='KR'
     IZ=36
     IF(W.NE.'K') GOTO 900
  ELSE IF(MASS.LT. 86.5 ) THEN    
     ! rubidium
     ELE='RB'
     IZ=37
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT. 88.5 ) THEN    
     ! strontium
     ELE='SR'
     IZ=38
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT. 90.5 ) THEN    
     ! yttrium
     ELE=' Y'
     IZ=39
     IF(W.NE.'Y') GOTO 900
  ELSE IF(MASS.LT. 92.0 ) THEN    
     ! zirconium 
     ELE='ZR'
     IZ=40
     IF(W.NE.'Z') GOTO 900
  ELSE IF(MASS.LT. 94.5 ) THEN    
     ! niobium
     ELE='NB'
     IZ=41
     IF(W.NE.'N') GOTO 900
  ELSE IF(MASS.LT. 97.5 ) THEN    
     ! molybdenum
     ELE='MO'
     IZ=42
     IF(W.NE.'M') GOTO 900
  ELSE IF(MASS.LT.100.5 ) THEN    
     ! technetium
     ELE='TC'  
     IZ=43
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT.102.0 ) THEN    
     ! ruthenium
     ELE='RU'
     IZ=44
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT.104.5 ) THEN    
     ! rhodium
     ELE='RH'
     IZ=45
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT.107.0 ) THEN    
     ! palladium
     ELE='PD'
     IZ=46
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.110.5 ) THEN    
     ! silver
     ELE='AG'
     IZ=47
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT.113.5 ) THEN    
     ! cadmium
     ELE='CD'
     IZ=48
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT.116.5 ) THEN    
     ! indium
     ELE='IN'
     IZ=49
     IF(W.NE.'I') GOTO 900
  ELSE IF(MASS.LT.120.5 ) THEN    
     ! tin   
     ELE='SN'
     IZ=50
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT.125.5 ) THEN    
     ! antimony
     ELE='SB'
     IZ=51
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT.129.5 ) THEN    
     IZ=53
     IF(W.EQ.'I') THEN
        ! iodine       
        ELE=' I'
        IZ=53
     ELSE IF(W.EQ.'T') THEN
        ! tellurium
        ELE='TE'
        IZ=52
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT.132.0 ) THEN    
     ! xenon 
     ELE='XE'
     IZ=54
     IF(W.NE.'X') GOTO 900
  ELSE IF(MASS.LT.135.5 ) THEN    
     ! cesium
     ELE='CS'
     IZ=55
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT.138.0 ) THEN    
     ! barium
     ELE='BA'
     IZ=56
     IF(W.NE.'B') GOTO 900
  ELSE IF(MASS.LT.139.5 ) THEN    
     ! lanthanum
     ELE='LA'
     IZ=57
     IF(W.NE.'L') GOTO 900
  ELSE IF(MASS.LT.140.5 ) THEN    
     ! cerium
     ELE='CE'
     IZ=58
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT.142.5 ) THEN    
     ! praseodymium
     ELE='PR'
     IZ=59
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.144.9 ) THEN    
     ! neodymium
     ELE='ND'
     IZ=60
     IF(W.NE.'N') GOTO 900
  ELSE IF(MASS.LT.148.5 ) THEN    
     ! promethium
     ELE='PM'
     IZ=61
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.151.5 ) THEN    
     ! samarium
     ELE='SM'
     IZ=62
     IF(W.NE.'S') GOTO 900
  ELSE IF(MASS.LT.155.5 ) THEN    
     ! europium
     ELE='EU'
     IZ=63
     IF(W.NE.'E') GOTO 900
  ELSE IF(MASS.LT.158.0 ) THEN    
     ! gadolinium
     ELE='GD'
     IZ=64
     IF(W.NE.'G') GOTO 900
  ELSE IF(MASS.LT.161.5 ) THEN    
     ! terbium
     ELE='TB'
     IZ=65
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT.163.5 ) THEN    
     ! dysprosium
     ELE='DY'
     IZ=66
     IF(W.NE.'D') GOTO 900
  ELSE IF(MASS.LT.166.5 ) THEN    
     ! holmium
     ELE='HO'
     IZ=67
     IF(W.NE.'H') GOTO 900
  ELSE IF(MASS.LT.168.0 ) THEN    
     ! erbium
     ELE='ER'
     IZ=68
     IF(W.NE.'E') GOTO 900
  ELSE IF(MASS.LT.171.5 ) THEN    
     ! thulium
     ELE='TM'
     IZ=69
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT.174.0 ) THEN    
     ! ytterbium
     ELE='YB'
     IZ=70
     IF(W.NE.'Y') GOTO 900
  ELSE IF(MASS.LT.176.5 ) THEN    
     ! lutetium
     ELE='LU'
     IZ=71
     IF(W.NE.'L') GOTO 900
  ELSE IF(MASS.LT.180.0 ) THEN    
     ! hafnium
     ELE='HF'
     IZ=72
     IF(W.NE.'H') GOTO 900
  ELSE IF(MASS.LT.182.5 ) THEN    
     ! tantalum
     ELE='TA'
     IZ=73
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT.185.5 ) THEN    
     ! tungsten
     ELE=' W'
     IZ=74
     IF(W.NE.'W') GOTO 900
  ELSE IF(MASS.LT.188.5 ) THEN    
     ! rhenium
     ELE='RE'
     IZ=75
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT.191.5 ) THEN    
     ! osmium
     ELE='OS'
     IZ=76
     IF(W.NE.'O') GOTO 900
  ELSE IF(MASS.LT.193.5 ) THEN    
     ! iridium
     ELE='IR'
     IZ=77
     IF(W.NE.'I') GOTO 900
  ELSE IF(MASS.LT.195.5 ) THEN    
     ! platinum
     ELE='PT'
     IZ=78
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.198.5 ) THEN    
     ! gold  
     ELE='AU'
     IZ=79
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT.202.5 ) THEN    
     ! mercury
     ELE='HG'
     IZ=80
     IF(W.NE.'H') GOTO 900
  ELSE IF(MASS.LT.206.5 ) THEN    
     ! thallium
     ELE='Tl'
     IZ=81
     IF(W.NE.'T') GOTO 900
  ELSE IF(MASS.LT.207.5 ) THEN    
     ! lead  
     ELE='PB'
     IZ=82
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.209.5 ) THEN    
     IZ=83
     IF(W.EQ.'B') THEN
        ! bismuth
        ELE='BI'
        IZ=83
     ELSE IF(W.EQ.'P') THEN
        ! polonium
        ELE='PO'
        IZ=84
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT.216.0 ) THEN    
     ! astatine
     ELE='AT'
     IZ=85
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT.222.5 ) THEN    
     ! radon 
     ELE='RN'
     IZ=86
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT.224.5 ) THEN    
     ! francium
     ELE='FR'
     IZ=87
     IF(W.NE.'F') GOTO 900
  ELSE IF(MASS.LT.226.5 ) THEN    
     ! radium
     ELE='RA'
     IZ=88
     IF(W.NE.'R') GOTO 900
  ELSE IF(MASS.LT.228.5 ) THEN    
     ! actinium
     ELE='AC'
     IZ=89
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT.236.5 ) THEN    
     IZ=91
     IF(W.EQ.'P') THEN
        ! protoactinium
        ELE='PA'
        IZ=91
     ELSE IF(W.EQ.'T') THEN
        ! thorium      
        ELE='TH'
        IZ=90
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT.242.5 ) THEN    
     IZ=92
     IF(W.EQ.'N') THEN
        ! neptunium
        ELE='NP'
        IZ=93
     ELSE IF(W.EQ.'U') THEN
        ! uranium   
        ELE=' U'
        IZ=92
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT.243.5 ) THEN    
     ! americium
     ELE='AM'
     IZ=95
     IF(W.NE.'A') GOTO 900
  ELSE IF(MASS.LT.246.5 ) THEN    
     ! plutonium
     ELE='PU'
     IZ=94
     IF(W.NE.'P') GOTO 900
  ELSE IF(MASS.LT.248.5 ) THEN
     ! curium
     ELE='CM'
     IZ=96
     IF(W.NE.'C') GOTO 900
  ELSE IF(MASS.LT.251.5 ) THEN    
     IZ=97
     IF(W.EQ.'B') THEN
        ! berkelium
        ELE = 'BK'
        IZ=97
     ELSE IF(W.EQ.'C') THEN
        ! californium
        ELE = 'CF'
        IZ=98
     ELSE
        GOTO 900
     ENDIF
  ELSE IF(MASS.LT.255.5 ) THEN    
     ! einsteinium
     ELE='ES'
     IZ=99
     IF(W.NE.'E') GOTO 900
  ELSE IF(MASS.LT.257.5 ) THEN    
     ! fermium    
     ELE='FM'
     IZ=100
     IF(W.NE.'F') GOTO 900
  ELSE
     ! mass out of range; unknown atom type 
     CALL WRNDIE(-1,'<FINDEL>','Mass out of range; Unknown atom type')
     IZ=100.0
  ENDIF
  !

500 CONTINUE

#if KEY_QCHEM==1
  IF ( ATYPE(NUM) .EQ. 'GHH' ) THEN
     ELE='@H'
  ELSE IF ( ATYPE(NUM) .EQ. 'GHC' ) THEN
     ELE='@C'
  ELSE IF ( ATYPE(NUM) .EQ. 'GHN' ) THEN
     ELE='@N'
  ELSE IF ( ATYPE(NUM) .EQ. 'GHO' ) THEN
     ELE='@O'
  ELSE IF ( ATYPE(NUM) .EQ. 'GHS' ) THEN
     ELE='@S'
  ELSE IF ( ATYPE(NUM) .EQ. 'GHP' ) THEN
     ELE='@P'
  ENDIF
#endif


  IF((PRNLEV.GE.4).AND.QINIGM) THEN
     CALL ATOMID(NUM,SIDDN,RIDDN,RESDN,ACDN)
     WRITE(OUTU,35)  NUM,SIDDN(1:idleng),RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng),ELE,IZ
35   FORMAT(' FINDEL: Quantum atom',I6,4(1X,A),'  assigned to element: ',A,I4)
  ENDIF
  !
  ZNUM=IZ
  RETURN
  !
900 CONTINUE
  IF(WRNLEV.GT.3) WRITE(OUTU,45) NUM,NAME,MASS,ELE
45 FORMAT(' FINDEL>  Cannot find element type for number',I5,' Chemical type: ',A6,'  Mass:',F10.4,/, &
       '          Its element type is set to: ',A6)
  if(wrnlev > 3) write(outu,'(a)')' FINDEL> Check atom order in ADDLink command; or:'
  if(wrnlev > 3) write(outu,'(a)') &
       ' FINDEL> The MM atom from ADDLink command is in th QM selection!'
  CALL WRNDIE(-1,'<FINDEL>', 'Atom type name does not match mass')
  IF(PRNLEV.GE.4) THEN
     CALL ATOMID(NUM,SIDDN,RIDDN,RESDN,ACDN)
     WRITE(OUTU,46)  NUM,SIDDN(1:idleng),RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng),ELE,IZ
46   FORMAT(' FINDEL>  Cannot find element type for number',I5,' SEGID: ',A,' RESID: ',A,' RESN: ',A,' IUPAC: ',A,/, &
          ' Chemical type: ',A,'  Atomic Number:',I4)
  ENDIF
  ZNUM=IZ
  RETURN
END SUBROUTINE FINDEL

end module linkatom

