module proto_mod
  use chm_kinds
  use chm_types
  use dimens_fcm
    implicit none

  !     proto.fcm - Contains global variables pertaining to prototype
  !                 definitions
  !
  !     TR 16. Jan 2006
  !
  !     MXPRTO - Maximum number of prototype definitions
  !     QPRTO  - is prototype x in use?
  !     NPRTO  - number of atoms in prototype
  !     HPPRTO - relative distances from pivot atom (if nonconsecutive)
  !     NSPRT  - number of pivot atoms
  !     HPSPRT - the pivot atoms for the respective prototype
  !     NSIPRT - number of image pivot atoms
  !     SPPROT - gives the dimension of the arrays allocated for this prot.
  !
  integer,parameter :: mxprto=10
  !
  logical qprto(mxprto)
  integer,dimension(mxprto) :: nprto,nsprt,nsiprt,spprot
  type(chm_iarray),dimension(mxprto),save :: hpprto,hpsprt

contains


#if KEY_PROTO==1 /*proto*/
  subroutine proto_init
    QPRTO(1:mxprto)=.FALSE.
    return
  end subroutine proto_init

  SUBROUTINE PROTO
    !
    !     parse prototype commands
    !
    use exfunc
    use number
    !
    use bases_fcm
    use comand
    use image
    use psf
    use stream
    use memory
    use string
    !
    !
    integer,allocatable,dimension(:) :: STISEL
    character(len=4) WRD
    INTEGER I
    INTEGER NPRT
    !
    call chmalloc('proto.src','PROTO','STISEL',NATOM,intg=STISEL)

    WRD=NEXTA4(COMLYN,COMLEN)
    IF((WRD.EQ.'DEFI').OR.(WRD.EQ.'REMO') &
         .OR. (WRD.EQ.'IMAG')) THEN
       !        Parse protoype number for these commands
       NPRT=-1
       NPRT=NEXTI(COMLYN,COMLEN)
       IF((NPRT.LT.0).OR.(NPRT.GT.MXPRTO)) THEN
          CALL WRNDIE(-2,'<PROTO>','Wrong prototype number')
          NPRT=1
       ENDIF
    ENDIF
    !
    IF(WRD.EQ.'DEFI') THEN
       IF(QPRTO(NPRT)) THEN
          if (prnlev >= 2) then
             WRITE(OUTU,800) 'WARNING: Prototype ', NPRT, &
                  ' already exists - overwriting!'
          endif
800       FORMAT(A,I2,A)
          call chmdealloc('proto.src','PROTO','HPPRTO(NPRT)%a', &
               NATOM,intg=HPPRTO(NPRT)%a)
          call chmdealloc('proto.src','PROTO','HPSPRT(NPRT)%a', &
               SPPROT(NPRT),intg=HPSPRT(NPRT)%a)
       ENDIF
       call chmalloc('proto.src','PROTO','HPPRTO(NPRT)%a',NATOM,intg=HPPRTO(NPRT)%a)
       SPPROT(NPRT)=NATOM
       IF(NATIM.GT.NATOM) SPPROT(NPRT)=NATIM
       call chmalloc('proto.src','PROTO','HPSPRT(NPRT)%a',SPPROT(NPRT),intg=HPSPRT(NPRT)%a)
       NPRTO(NPRT)=0
       NPRTO(NPRT)=GTRMI(COMLYN,COMLEN,'PNUM',NPRTO(NPRT))
       NPRTO(NPRT)=NPRTO(NPRT)*(-1)
       IF(NPRTO(NPRT).EQ.0) THEN
          !           i.e. we need an example - the prototype
          CALL SHLSEL(COMLYN,COMLEN,HPPRTO(NPRT)%a, &
               NPRTO(NPRT),stisel)
       ENDIF
       !        now get the selection containing the protos to use here
       CALL SHLSEL(COMLYN,COMLEN,HPSPRT(NPRT)%a, &
            NSPRT(NPRT),stisel)
       !        pick out the pivot atoms
       CALL GETPRT(NPRTO(NPRT),HPPRTO(NPRT)%a, &
            NSPRT(NPRT),HPSPRT(NPRT)%a, &
            NATOM,stisel)
       !        finally: flag the prototype as used
       QPRTO(NPRT)=.TRUE.

    ELSEIF(WRD.EQ.'REMO') THEN
       IF(.NOT.QPRTO(NPRT)) THEN
          if (prnlev >= 2) then
             WRITE(OUTU,800) 'WARNING: Prototype ',NPRT, &
                  ' is not in use - ignoring command'
          endif
       ELSE
          call chmdealloc('proto.src','PROTO','HPPRTO(NPRT)%a',NATOM,&
               intg=HPPRTO(NPRT)%a)
          SPPROT(NPRT)=NATOM
          IF(NATIM.GT.NATOM) SPPROT(NPRT)=NATIM
          call chmdealloc('proto.src','PROTO','HPSPRT(NPRT)%a',SPPROT(NPRT),&
               intg=HPSPRT(NPRT)%a)
          NPRTO(NPRT)=0
          QPRTO(NPRT)=.FALSE.
          if (prnlev >= 2) WRITE(OUTU,800) ' Prototype ',NPRT,' has been removed.'
       ENDIF
    ELSEIF(WRD.EQ.'IMAG') THEN
       !        check if there are any images present at all!
       IF(NATIM.GT.NATOM) THEN
          !           fill the selection array
          CALL ADIMPRT(NPRT)
       ELSE
          CALL WRNDIE(-1, '<PROTO>', &
               'No images present (not set up/filled?)')
       ENDIF
    ELSEIF(WRD.EQ.'INFO') THEN
       DO I=1,MXPRTO
          IF(QPRTO(I)) THEN
             CALL PRNPRT(I)
          ELSE
             IF(PRNLEV.GT.5) WRITE(OUTU,800)  &
                  '     Prototype ',I,' is not in use.'
          ENDIF
       ENDDO
    ENDIF
    !
    !
    call chmdealloc('proto.src','PROTO','STISEL',NATOM,intg=STISEL)
    !
    RETURN
  END SUBROUTINE PROTO
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE GETPRT(NPRT,PROTO,NLST,LIST,NATOM,FLAGS)
    !     generates all PROTotype relevant data from the raw input
    !
    use exfunc
    use number
    use chutil,only:atomid
    !
    INTEGER NPRT,PROTO(*),NLST,LIST(*),NATOM,FLAGS(*)
    !
    INTEGER PIVOT,I,J
    character(len=4) SEGID,RESID,RESNAM,ATNAM,PRESNA,PATNAM
    !
    IF(NPRT.LT.0) THEN
       !        we just got the number of prototype atoms and assume them
       !        to be consecutive
       NPRT=NPRT*(-1)
       PIVOT=LIST(1)
       DO I=1,NPRT
          PROTO(I)=I-1
       ENDDO
    ELSE
       !        extract the (relative) atom numbers from the protype list
       PIVOT=PROTO(1)
       DO I=1,NPRT
          PROTO(I)=PROTO(I)-PIVOT
       ENDDO
    ENDIF
    !
    !     zero out flag list
    DO I=1,NATOM
       FLAGS(I)=ZERO
    ENDDO
    !     now get the identity (PRESNA and PATNAM) of the pivot atom
    CALL ATOMID(PIVOT,SEGID,RESID,PRESNA,PATNAM)
    !
    !     and now extract all atoms that have the same RESNAM and ATNAM
    !     than the pivot from the LIST and store them in LIST
    J=1
    DO I=1,NLST
       CALL ATOMID(LIST(I),SEGID,RESID,RESNAM,ATNAM)
       IF((PRESNA.EQ.RESNAM).AND.(PATNAM.EQ.ATNAM)) THEN
          FLAGS(LIST(I))=1
          LIST(J)=LIST(I)
          J=J+1
       ENDIF
    ENDDO
    NLST=J-1
    !
    RETURN
  END SUBROUTINE GETPRT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE PRNPRT(NPRT)
    !
    !     wrapper that can be called with only the PROT-number
    !
    use exfunc
    use number
    !
    !
    INTEGER NPRT
    !
    !
    CALL PRNPR2(NPRT,NPRTO(NPRT),HPPRTO(NPRT)%a, &
         NSPRT(NPRT),NSIPRT(NPRT),HPSPRT(NPRT)%a )
    !
    RETURN
  END SUBROUTINE PRNPRT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE PRNPR2(N,NPROTO,PROTO,NSPRTO,NSIPRT,SPROTO)
    !
    !     print info about prototype N
    !
    use exfunc
    use number
    !
    use stream
    use chutil,only:atomid

    !
    INTEGER N,NPROTO,PROTO(*),NSPRTO,NSIPRT,SPROTO(*)
    !
    INTEGER I
    character(len=4) SEGID,RESID,RESNAM,ATNAM,PRESNA,PATNAM
    !
    if (prnlev >= 2) then
       WRITE(OUTU,801) ''
       WRITE(OUTU,800) '     Information about prototype ',N
       WRITE(OUTU,801) ''
       WRITE(OUTU,802) '      prototype contains ',NPROTO,' atoms:'
       WRITE(OUTU,801) '       relative atom numbers:'
    endif
    DO I=1,NPROTO
       !        now get the identity (PRESNA and PATNAM) of the pivot atom
       CALL ATOMID(SPROTO(1)+PROTO(I),SEGID,RESID,PRESNA,PATNAM)
       if (prnlev >= 2) WRITE(OUTU,803) PROTO(I),RESID,PRESNA,PATNAM
    ENDDO
    if (prnlev >= 2) then
       WRITE(OUTU,802) '       there are ',NSPRTO, &
            ' pivot atoms for this prototype'
    endif
    IF(PRNLEV.GT.5) THEN
       DO I=1,NSPRTO
          CALL ATOMID(SPROTO(I),SEGID,RESID,PRESNA,PATNAM)
          WRITE(OUTU,804) SPROTO(I),PRESNA,PATNAM
       ENDDO
       IF(NSIPRT.GT.NSPRTO) THEN
          WRITE(OUTU,802) '       and ',NSIPRT-NSPRTO, &
               ' image pivot atoms:'
          DO I=NSPRTO+1,NSIPRT
             CALL ATOMID(SPROTO(I),SEGID,RESID,PRESNA,PATNAM)
             WRITE(OUTU,804) SPROTO(I),PRESNA,PATNAM
          ENDDO
       ENDIF
    ENDIF
    if (prnlev >= 2) WRITE(OUTU,801) ''
    !
800 FORMAT(A,I3)
801 FORMAT(A)
802 FORMAT(A,I8,A)
803 FORMAT(10X,I10,10X,A,2X,A,2X,A)
804 FORMAT(10X,I10,10X,A,2X,A)
    !
    RETURN
  END SUBROUTINE PRNPR2
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE PRTFLG(NAT,NPIV,PIVOTS,FLAG)
    !
    !     fill the FLAG(*) array with 0/1 for the NPIV atoms in PIVOTS(*)
    !

    INTEGER NAT,NPIV,PIVOTS(*),FLAG(*)
    !
    INTEGER I
    !
    DO I=1,NAT
       FLAG(I)=0
    ENDDO
    DO I=1,NPIV
       FLAG(PIVOTS(I))=1
    ENDDO
    !
    RETURN
  END SUBROUTINE PRTFLG
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE ADIMPRT(NPRT)
    !
    !     wrapper that can be called with only the PROT-number
    !
    use exfunc
    use number
    use bases_fcm
    use image
    use psf
    use memory
    !
    integer,allocatable,dimension(:) :: STISEL
    INTEGER NPRT

    call chmalloc('proto.src','ADIMPRT','STISEL',NATOM,intg=STISEL)

    CALL PRTFLG(NATOM,NSPRT(NPRT),HPSPRT(NPRT)%a, stisel)
    CALL ADIMPR2(NPRTO(NPRT),HPPRTO(NPRT)%a, &
         NSPRT(NPRT),HPSPRT(NPRT)%a,NSIPRT(NPRT),SPPROT(NPRT), &
         stisel,NATOM,NATIM,BIMAG%IMATTR)
    call chmdealloc('proto.src','ADIMPRT','STISEL',NATOM,intg=STISEL)

    RETURN
  END SUBROUTINE ADIMPRT
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE ADIMPR2(NPRT,PROTO,NPIV,PIV,NIPIV,NSTMAX,SETSL, &
       NATOM,NATIM,IMATT)
    !
    !     routine that that checks all image atoms and adds them
    !     to SET if the primary atom an image was created from is part of
    !     the original set (SETSL-flag-list). NNSET then is the last atom
    !     in the set (i.e. 1-NSET are the primaries and (NSET+1)-NNSET the
    !     images).
    !
    !     Tibor Rudas Jan 2003 - Jun 2003
    !     TR 3/2006: add check for maximum size of SET to avoid overflows
    !
    use number
    use stream

    !
    INTEGER NPRT,PROTO(*),NPIV,PIV(*),NIPIV,SETSL(*)
    INTEGER NSTMAX,NATOM,NATIM,IMATT(*)
    !
    !
    INTEGER I,J,K,N2,TMP
    INTEGER CHKED,ADDED
    LOGICAL QGO
    !
    !     add image (pivot) atoms - but only if corresponding prototype is
    !                               complete (ie all image atoms of this proto
    !                               are present)
    CHKED=0
    ADDED=0
    NIPIV=NPIV
    DO I=NATOM+1,NATIM
       K=IMATT(I)
       IF(SETSL(K).GT.0) THEN
          CHKED=CHKED+1
          QGO=.TRUE.
          DO J=1,NPRT
             TMP=I+PROTO(J)
             IF(TMP.GT.NATIM) THEN
                QGO=.FALSE.
             ELSE
                N2=IMATT(TMP)-K-PROTO(J)
                IF(N2.NE.ZERO) QGO=.FALSE.
             ENDIF
          ENDDO
          IF(QGO) THEN
             !              i.e. we found a image pivot atom and the image proto is complete
             ADDED=ADDED+1
             NIPIV=NIPIV+1
             IF(NIPIV.GT.NSTMAX) THEN
                CALL WRNDIE(-3, '<ADIMPRT>', &
                     'Trying to add images beyond requested set-size')
                NIPIV=NSTMAX
             ELSE
                PIV(NIPIV)=I
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    !     do statistics output if requested
    IF(PRNLEV.GT.5) WRITE(OUTU,800) CHKED,ADDED
    !
800 FORMAT(' ADIMPR2 has checked ',I5, &
         ' image prototypes and added ',I5)
    RETURN
  END SUBROUTINE ADIMPR2
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE CALPRDI(NP,QMASS)
    !
    !     calculate the sum of (possibly charged) prototype dipole moments
    !
    use exfunc
    use coord
    use psf
    use image,only:natim
    use memory
    use param_store, only: set_param

    integer,allocatable,dimension(:) :: ISpace
    INTEGER NP
    LOGICAL QMASS
    !
    real(chm_real) XD,YD,ZD,QTOT

    call chmalloc('proto.src','CALPRDI','Ispace',NPRTO(NP),intg=Ispace)

    IF(.NOT.QPRTO(NP)) RETURN
    CALL CALPRD2(max(NATOM,natim),X,Y,Z,CG,AMASS,XD,YD,ZD,QTOT, &
         NPRTO(NP),HPPRTO(NP)%a, &
         NSPRT(NP),HPSPRT(NP)%a,Ispace, &
         QMASS)
    !
    call set_param('XDIP',XD)
    call set_param('YDIP',YD)
    call set_param('ZDIP',ZD)
    call set_param('CHARGE',QTOT)
    call chmdealloc('proto.src','CALPRDI','Ispace',NPRTO(NP),intg=Ispace)
    RETURN
  END SUBROUTINE CALPRDI
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE CALPRD2(NAT,X,Y,Z,CG,AMASS, &
       XD,YD,ZD,QTOT,NPRT,PROTO,NPIV,PIVOT,TEMP,QMASS)
    !
    !     this is the work part: generate temp. atom list and
    !     calculate prototype dipole moment, sum up and return.
    !     the call of CDIPOLE means that the mass of a set is recalculated
    !     for each set but it is more generic since the actual calculation
    !     of the dipole moment is deferred to one central routine
    !
    use corsubs,only:cdipole
    use exfunc
    use number

    real(chm_real) X(*),Y(*),Z(*),CG(*),AMASS(*),XD,YD,ZD,QTOT
    INTEGER NAT,NPRT,PROTO(*),NPIV,PIVOT(*),TEMP(*)
    LOGICAL QMASS
    !
    INTEGER I,J,PIV
    real(chm_real) XDT,YDT,ZDT,QT
    !
    XD=ZERO
    YD=ZERO
    ZD=ZERO
    QTOT=ZERO
    !
    DO I=1,NPIV
       PIV=PIVOT(I)
       DO J=1,NPRT
          TEMP(J)=PIV+PROTO(J)
       ENDDO
       !
       CALL CDIPOLE(NAT,X,Y,Z,CG,NPRT,TEMP,XDT,YDT,ZDT,QT, &
            .TRUE.,QMASS,AMASS)
       XD=XD+XDT
       YD=YD+YDT
       ZD=ZD+ZDT
       QTOT=QTOT+QT
    ENDDO
    !
    RETURN
  END SUBROUTINE CALPRD2
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE CALPRVE(NP,QMASS)
    !
    !     calculate the sum of (possibly charged) prototype c.o.m. velocities
    !
    use exfunc
    use coord
    use psf
    use param_store, only: set_param

    INTEGER NP
    LOGICAL QMASS
    real(chm_real) XVCM,YVCM,ZVCM
    !
    !
    IF(.NOT.QPRTO(NP)) RETURN
    CALL CALPRV2(NATOM,X,Y,Z,CG,AMASS,XVCM,YVCM,ZVCM, &
         NPRTO(NP),HPPRTO(NP)%a, &
         NSPRT(NP),HPSPRT(NP)%a, &
         QMASS)
    !
    call set_param('XVCM',XVCM)
    call set_param('YVCM',YVCM)
    call set_param('ZVCM',ZVCM)
    !
    RETURN
  END SUBROUTINE CALPRVE
  !
  !----------------------------------------------------------------------
  !
  SUBROUTINE CALPRV2(NAT,X,Y,Z,CG,AMASS, &
       XVCM,YVCM,ZVCM,NPRT,PROTO,NPIV,PIVOT,QMASS)
    !
    !     this is the work part: generate temp. atom list and
    !     calculate prototype c.o.m. velocity, sum up and return
    !
    !     we could reuse routine GETCOM in rdfsol.src but this
    !     would mean to re-calculate the total mass for each prototype
    !     (which should be the same for all prototypes)
    !
    use exfunc
    use number

    real(chm_real) X(*),Y(*),Z(*),CG(*),AMASS(*),XVCM,YVCM,ZVCM
    INTEGER NAT,NPRT,PROTO(*),NPIV,PIVOT(*)
    LOGICAL QMASS
    !
    INTEGER I,J,PIV,NN
    real(chm_real) XVT,YVT,ZVT,MASS
    !
    XVCM=ZERO
    YVCM=ZERO
    ZVCM=ZERO
    MASS=ZERO
    !
    !     calc total mass of one prototype only once
    PIV=PIVOT(1)
    MASS=ONE
    IF(QMASS) THEN
       DO J=1,NPRT
          MASS=MASS+AMASS(PIV+PROTO(J))
       ENDDO
    ENDIF
    !
    DO I=1,NPIV
       PIV=PIVOT(I)
       XVT=ZERO
       YVT=ZERO
       ZVT=ZERO
       IF(QMASS) THEN
          DO J=1,NPRT
             NN=PIV+PROTO(J)
             XVT=XVT+(X(NN)*AMASS(NN))
             YVT=YVT+(Y(NN)*AMASS(NN))
             ZVT=ZVT+(Z(NN)*AMASS(NN))
          ENDDO
       ELSE
          DO J=1,NPRT
             NN=PIV+PROTO(J)
             XVT=XVT+X(NN)
             YVT=YVT+Y(NN)
             ZVT=ZVT+Z(NN)
          ENDDO
       ENDIF
       XVCM=XVCM+(XVT/MASS)
       YVCM=YVCM+(YVT/MASS)
       ZVCM=ZVCM+(ZVT/MASS)
    ENDDO
    !
    !
    RETURN
  END SUBROUTINE CALPRV2
  !
  SUBROUTINE FILPROT(NPRT,X,Y,Z,AX,AY,AZ,DIP,NSOL,NSOLI,SOL, &
       QMASS)
    !
    !     wrapper callable with little args
    !
    use exfunc
    use number
    !
    use psf
    use image,only:natim

    INTEGER NPRT,NSOL,NSOLI,SOL(*)
    real(chm_real) DIP(4,*),X(*),Y(*),Z(*),AX(*),AY(*),AZ(*)
    LOGICAL QMASS

    !
    CALL FILPRO2(max(NATOM,natim),NPRTO(NPRT),HPPRTO(NPRT)%a, &
         NSPRT(NPRT),NSIPRT(NPRT),HPSPRT(NPRT)%a, &
         NSOL,NSOLI,SOL, &
         X,Y,Z,AX,AY,AZ,DIP,QMASS,CG,AMASS)
    !
    RETURN
  END SUBROUTINE FILPROT
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE FILPRO2(NATOM,NPRT,PROTO,NLST,NLSTI,LIST, &
       NSOL,NSOLI,SOL, &
       X,Y,Z,AX,AY,AZ,DIP,QMASS,CG,AMASS)
    !
    !     calculates the centers of mass and dipole moments
    !     for a given prototype set
    !
    use exfunc
    use number
    use stream
    use corsubs,only:cdipole,getcom

    INTEGER NATOM,NPRT,PROTO(*),NLST,NLSTI,LIST(*)
    INTEGER NSOL,NSOLI,SOL(*)
    real(chm_real) DIP(4,*),X(*),Y(*),Z(*),AX(*),AY(*),AZ(*)
    real(chm_real) CG(*),AMASS(*)
    LOGICAL QMASS
    !
    INTEGER PIVOT,I,J,NLAST
    real(chm_real) CHRG,XD,YD,ZD
    integer,dimension(NPRT) :: TMP
    !
    !     1: transfer the pivot image atom-numbers from PROTO to SETA/B
    NSOLI=NSOL
    IF(NLSTI.GT.NLST) THEN
       DO I=NLST+1,NLSTI
          NSOLI=NSOLI+1
          SOL(NSOLI)=LIST(I)
       ENDDO
    ENDIF
    !
    !     2: fill coors with c.o.geom. or c.o.m. and dip with the dipole moment
    !     try to do real & images in one...
    NLAST=NLST
    IF(NLSTI.GT.NLST) NLAST=NLSTI
    DO I=1,NLAST
       PIVOT=LIST(I)
       DO J=1,NPRT
          TMP(J)=PIVOT+PROTO(J)
       ENDDO
       !        calc center of mass (CHRG is just a dummy)
       CALL GETCOM(TMP,NPRT,X,Y,Z,AMASS,CG, &
            AX(PIVOT),AY(PIVOT),AZ(PIVOT),CHRG)
       !        calc dipole moment (OXYZ not supported (yet))
       CALL CDIPOLE(NATOM,X,Y,Z,CG,NPRT,TMP, &
            XD,YD,ZD,CHRG,.FALSE.,QMASS,AMASS)
       DIP(4,PIVOT)=DSQRT((XD*XD)+(YD*YD)+(ZD*ZD))
       IF(DIP(4,PIVOT).GT.ZERO) THEN
          DIP(1,PIVOT)=XD/DIP(4,PIVOT)
          DIP(2,PIVOT)=YD/DIP(4,PIVOT)
          DIP(3,PIVOT)=ZD/DIP(4,PIVOT)
       ELSE
          DIP(1,PIVOT)=ZERO
          DIP(2,PIVOT)=ZERO
          DIP(3,PIVOT)=ZERO
       ENDIF
    ENDDO
    !
    RETURN
  END SUBROUTINE FILPRO2
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE WSOLPRT(NPRT,NSOL,SOL)
    !
    !     wrapper callable with little args
    !
    use exfunc
    use number
    use psf

    INTEGER NPRT,NSOL,SOL(*)
    !
    !
    CALL SOLPR2(NSPRT(NPRT),HPSPRT(NPRT)%a,NSOL,SOL)
    !
    RETURN
  END SUBROUTINE WSOLPRT
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE SOLPR2(NPIV,PIV,NSOL,SOL)
    !
    !     just fill the atomnumbers from the PROTO array to SOLA or SOLB
    !     for the proper handling/accounting of the prototypes as RDFSOL units
    !
    use exfunc
    use number
    use stream

    INTEGER NPIV,PIV(NPIV),NSOL,SOL(NPIV)
    !
    INTEGER I
    !
    NSOL=NPIV
    DO I=1,NPIV
       SOL(I)=PIV(I)
    ENDDO
    !
    RETURN
  END SUBROUTINE SOLPR2
  !

#else /*        (proto)*/

  SUBROUTINE PROTO
    CALL WRNDIE(-1,'<PROTO>','PROTO is not currently compiled')
    RETURN
  end SUBROUTINE PROTO

#endif /*       (proto)*/


end module proto_mod


