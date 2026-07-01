SUBROUTINE HBUILD(COMLYN, COMLEN)
  !-----------------------------------------------------------------------
  !     Process the HBUILD command.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm

  use bases_fcm
  use coord
  use hbondm
  use inbnd
  use psf
  use stream
  use memory
  use datstr,only:dupldt_nbond,freedt_nbond
!---   use nbutil_module,only:gtnbct
  implicit none

  integer,allocatable,dimension(:) :: Iarr,jarr
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN

  INTEGER   NWATAC
  LOGICAL   LWARN, QPRINT, QZERO
  real(chm_real)    ANGLON, CUTWAT, DISTOF, PHISTP

#if KEY_NOMISC==1
  CALL WRNDIE(-1,'<HBUILD>','HBUILD code is not compiled.')
  RETURN
#else /**/

  call chmalloc('hbuild.src','HBUILD','Iarr',NATOM,intg=Iarr)
  call chmalloc('hbuild.src','HBUILD','Jarr',NATOM,intg=Jarr)
  CALL GTHBUI(COMLYN,COMLEN,Iarr,Jarr,NWATAC,X,Y,Z,WMAIN, &
       PHISTP,QPRINT,LWARN,DISTOF,ANGLON,QZERO,CUTWAT)
  IF (QZERO) THEN
     CALL WRNDIE(4,'<GTHBUI>','Zero selection specified')
     COMLEN=0
  ELSE
     CALL GTHBCT(COMLYN,COMLEN)
     CALL FREEDT_nbond(BNBNDC)
     CALL DUPLDT_nbond(BNBNDC,BNBND)
     CALL GTNBCT(COMLYN,COMLEN,BNBNDC)
     CALL HBUIL1(OUTU,iarr,jarr,NWATAC,X,Y,Z,BNBNDC, &
          PHISTP,QPRINT,CUTWAT)
#if KEY_NOST2==0
     IF (LWARN.AND.NST2 > 0) CALL ST2WRN(DISTOF,ANGLON,X,Y,Z,Iarr)
#endif 

     CALL FREEDT_nbond(BNBNDC)
  ENDIF
  call chmdealloc('hbuild.src','HBUILD','Iarr',NATOM,intg=Iarr)
  call chmdealloc('hbuild.src','HBUILD','Jarr',NATOM,intg=Jarr)

  RETURN
END SUBROUTINE HBUILD

SUBROUTINE HBUIL1(UNIT,HFLAGS,WATACC,NWATAC,X,Y,Z, &
     BNBND,PHISTP,QPRINT,CUTWAT)
  !-----------------------------------------------------------------------
  !     by Axel Brunger, DEC-82
  !
  !     HBUILD constructs unknown hydrogen positions, based on a minimum
  !     energy configuration search involving the environment of a given
  !     donor atom of the protein or the water.
  !     (This is a dummy subroutine to allocate STACK space for HBUIL2.)
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc, only: order5

  use psf
  use param
  use inbnd
  use hbondm
  use stream
  implicit none

  INTEGER   UNIT
  INTEGER HFLAGS(*), WATACC(*)
  INTEGER   NWATAC
  real(chm_real) X(*),Y(*),Z(*)
  TYPE(nonbonddatastructure) BNBND
  real(chm_real)      PHISTP
  LOGICAL   QPRINT
  real(chm_real)      CUTWAT

  EXTERNAL  EXCH5

  integer,parameter :: MAXDIH = 15
  !
  !     First print some information,
  !
  IF(PRNLEV >= 2) THEN
     WRITE(UNIT,1000) CUTHB,CUTHBA
     WRITE(UNIT,2000) CTONHB,CTOFHB,CTONHA,CTOFHA
     IF (LHBFG) THEN
        WRITE(UNIT,3000)
     ELSE
        WRITE(UNIT,3500)
     ENDIF
     WRITE(UNIT,5000) PHISTP
  ENDIF
1000 FORMAT(/' CUToff Hydrogen Bond  distance =',F10.4, &
       10X,' Angle =',F10.4)
2000 FORMAT(' CuT switching ON HB dist. = ',F10.4,' OFf HB dist. =', &
       F10.4,/ &
       ' CuT switching ON Hb Angle = ',F10.4,' OFf Hb Angle =', &
       F10.4)
3000 FORMAT(' ACCEptor antecedents included')
3500 FORMAT(' NO ACceptor antecedents included')
5000 FORMAT(' dihedral PHI STePs for spin =',F10.4)

  IF (CUTHB <= 1.0.AND.CUTWAT.LE.0.0) THEN
     CUTWAT=7.5
  ELSE IF (CUTWAT <= 0.0) THEN
     CUTWAT=CUTHB
  ENDIF
  IF(PRNLEV >= 2) WRITE(UNIT,5100) CUTWAT
5100 FORMAT(' cutoff for water acceptor search CUTWat=',F10.4,/)
  !
  !     the PSF donor and acceptor lists are sorted
  !
  CALL SORT(NDON,EXCH5,ORDER5,IDON,IHD1,0,0,0,0,0,2)
  CALL SORT(NACC,EXCH5,ORDER5,IACC,IAC1,0,0,0,0,0,2)

  CALL HBUIL2(UNIT,HFLAGS,WATACC,NWATAC, &
       X,Y,Z,BNBND,PHISTP,CUTWAT,QPRINT, &
       MAXDIH)

  RETURN
END SUBROUTINE HBUIL1

SUBROUTINE HBUIL2(UNIT,HFLAGS,WATACC,NWATAC, &
     X,Y,Z,BNBND,PHISTP,CUTWAT,QPRINT, &
     MAXDIH)
  !-----------------------------------------------------------------------
  !     Axel Brunger, DEC-82
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc, only: orderr
  use number

  use consta
  use psf
  use param
  use code
  use deriv
  use inbnd
  use hbondm
  use stream
  use chutil,only:hydrog,lone,initia,atomid
!---   use nbutil_module,only:nbondx
  implicit none

  INTEGER   UNIT
  INTEGER   HFLAGS(*), WATACC(*)
  INTEGER   NWATAC
  real(chm_real)    X(*),Y(*),Z(*)
  TYPE(nonbonddatastructure) BNBND
  real(chm_real)    PHISTP, CUTWAT
  LOGICAL   QPRINT
  INTEGER   MAXDIH

  integer :: HHCODE(NATOM), NATBON(NATOM), IATBON(IATBMX,NATOM)
  real(chm_real) :: WATPRT(NATOM)
  integer :: WATPER(NATOM)
  integer :: FREEAT(IATBMX), FIXDAT(IATBMX)
  real(chm_real) :: FREEBD(IATBMX), FRFXAN(IATBMX,IATBMX)
  real(chm_real) :: FRFRAN(IATBMX,IATBMX)
  integer :: IDIHDL(MAXDIH)
  integer :: IWORK(NACC+NST2)
  real(chm_real) :: RWORK(NACC+NST2)
  integer :: IOFF(NATC)

  EXTERNAL    EXCH
  INTEGER     I, II, J
  INTEGER     BESHB, BES2HB, DONOFS, DONOLS
  INTEGER     ATOM, NFIXAT, NFREAT, DONOR
  INTEGER     IBT, JBT, NDIHDL
  INTEGER     WATER, NWATER

  CHARACTER(len=8) SIDDN, RIDDN, RESDN, ACDN, ACHY(4)

  real(chm_real)      CHIRAL
  real(chm_real)      LOWERG, LOWDIS, DIST2, XWAT, YWAT
  real(chm_real)      ZWAT, XD, YD, ZD, CUTHB2
  real(chm_real)      PHI, PHIMAX, RAD, BESPHI, ENERGY
  real(chm_real)      DDX(3), DDY(3), DDZ(3), CSTE, SNTE
  LOGICAL     SPARTD, ERROR, PLANE, CONDIT
  !
  !     Use of HFLAGS:
  !
  !     On input: (HFLAGS(ATOM) == 1) : hydrogen ATOM should be
  !     constructed
  !     In addition during execution: (HFLAGS(ATOM) == 2) :
  !     ATOM is a heavy atom bonded to a hydrogen I with
  !     HFLAGS(I) == 1.
  !
  !     The HHCODE array defines a code for each ATOM with HFLAGS(ATOM)
  !      /= 0
  !
  !     HHCODE(ATOM) =  1       atoms form a group with fixed geometry,
  !     10      are atoms of a water,
  !     100     atoms form a group with a rotational
  !     freedom,
  !     10000   are atoms of an isolated water,
  !     100000  are atoms of a water with only
  !     one HB acceptor,
  !
  !     WATACC are used only during construction of water hydrogens and
  !     lone pairs. WATACC contains a list of all atoms which are
  !     considered as an acceptor when constructing a plane for (H-O-H)
  !
  !     Note: DONOR is a heavy atom bonded to selected (HFLAGS) hydrogens.
  !
  !     Note: the IDON, IHD1 and IACC, IAC1 listings should be sorted.
  !
  RAD=PI/ONE8TY
  !
  !     MAKE BOND LIST
  !     First make a list of all bonds
  !     for every atom.
  !
  DO I=1,NATOM
     NATBON(I)=0
  ENDDO
  DO I=1,NBOND
     IBT=IB(I)
     JBT=JB(I)
      if(isdrude(ibt).or.isdrude(jbt)) goto 20
     !
     !     If a bond links two unknown atoms, ignore it. -BRB
     IF (.NOT.(HFLAGS(IBT) == 1 .AND. HFLAGS(JBT).EQ.1)) THEN

        NATBON(IBT)=NATBON(IBT)+1
        IF(NATBON(IBT) > IATBMX) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,25) IBT
           CALL DIEWRN(-4)
        ENDIF
        NATBON(JBT)=NATBON(JBT)+1
        IF(NATBON(JBT) > IATBMX) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,25) JBT
           CALL DIEWRN(-4)
        ENDIF
        IATBON(NATBON(JBT),JBT)=IBT
        IATBON(NATBON(IBT),IBT)=JBT
     ENDIF
 20 continue
  enddo
25 FORMAT(' ERROR FROM HBUILD:',/,' Too many bonds for atom',I5)
  !
  !---- Begin Procedure DETERMINE-HEAVY-ATOMS
  !
  !     Determines heavy atoms bonded to those hydrogens which should
  !     be constructed.
  !     They are indicated by a "2" in the HFLAGS array
  !     whereas hydrogens are indicated by a "1" in HFLAGS.
  !
  DO I=1,NATOM
     IF (HFLAGS(I) == 1) THEN
        !
        !         check that this is actually a hydrogen or a lone pair
        !
        IF (.NOT.(HYDROG(I).OR.LONE(I)) .OR. NATBON(I) /= 1) THEN
           IF(WRNLEV >= 2) WRITE(OUTU,*) HYDROG(I),LONE(I),NATBON(I)
           CALL ATOMID(I,SIDDN,RIDDN,RESDN,ACDN)
           IF(WRNLEV >= 2) WRITE(UNIT,45) SIDDN(1:idleng), &
                RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
           CALL DIEWRN(0)
           RETURN
        ELSE
           DONOR=IATBON(1,I)
           IF (.NOT. INITIA(DONOR,X,Y,Z)) THEN
              CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
              IF(WRNLEV >= 2) WRITE(UNIT,55) SIDDN(1:idleng), &
                   RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
              CALL DIEWRN(0)
              RETURN
           ELSE
              HFLAGS(DONOR)=2
           ENDIF
        ENDIF
     ENDIF
  enddo
45 FORMAT(' WARNING FROM HBUILD:',/, &
       ' Attempt to construct the atom',4(1X,A),', which is ', &
       'not a hydrogen according to its mass and its bonds.')
55 FORMAT(' WARNING FROM HBUILD:',/, &
       ' Attempt to construct a hydrogen bonded to the atom', &
       4(1X,A), ' with unknown coordinates.')
  !
  !     Finally, initialize the HHCODE array.
  !
  DO I=1,NATOM
     IF (HFLAGS(I) /= 0) THEN
        HHCODE(I)=1
     ELSE
        HHCODE(I)=0
     ENDIF
  ENDDO
  !---- End Procedure DETERMINE-HEAVY-ATOMS
  !
  !---- Begin Procedure FILL-CODE-ARRAYS-FOR-DIHEDRALS-AND-IMPROPERS
  !
  !     Fills code arrays ICP and ICI for dihedrals and impropers.
  !     Needs common blocks PARAM, PSF, RTF. The IOFF array is filled
  !     for use in GET-EQUILIBRIUM-GEOMETRY.
  !
  DO I=1,NATC
     IOFF(I)=(I*(I-1))/2
  ENDDO
  CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC,NBOND,IB,JB, &
       NTHETA,IT,JT,KT,NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
       QDRUDE,NBDRUDE,                                 & ! DRUDE
#if KEY_CMAP==1
       [0],0,[0],[0],[0],[0],[0],[0],[0],[0],                            & 
#endif
       .FALSE.,.FALSE.)
  !---- End Procedure FILL-CODE-ARRAYS-FOR-DIHEDRALS-AND-IMPROPERS
  !
  !     now, loop over all heavy donor atoms,
  !
  DONOR=0
  !  IF (DONOR < NATOM) then    !GOTO 200
  FREEAT(1:IATBMX) = 0
  FIXDAT(1:IATBMX) = 0
  loop100: do while( donor < natom)
     DONOR=DONOR+1
     IF (HFLAGS(DONOR) == 2) THEN

        CALL FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
             FREEAT,FIXDAT,X,Y,Z)

        !
        !---- Begin Procedure SEPARATE-WATERS
        IF (NFIXAT == 0) THEN
           SPARTD=.TRUE.
           HHCODE(DONOR)=10
           DO J=1,NFREAT
              HHCODE(FREEAT(J))=10
           ENDDO
        ELSE
           SPARTD=.FALSE.
        ENDIF
        !---- End Procedure SEPARATE-WATERS
        !
        IF (.NOT. SPARTD) THEN
           !
           !---- Begin Procedure SEPARATE-DONORS-TO-SPIN
           IF (NFIXAT < 2) THEN
              SPARTD=.TRUE.
              HHCODE(DONOR)=100
              DO J=1,NFREAT
                 HHCODE(FREEAT(J))=100
              ENDDO
           ELSE
              SPARTD=.FALSE.
           ENDIF
           !---- End Procedure SEPARATE-DONORS-TO-SPIN
           !
           IF (.NOT. SPARTD) THEN
              CALL GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
                   FREEBD,FRFRAN,FRFXAN,IOFF)
              IF (NFREAT == 0) THEN
                 !
                 !     Do nothing.
                 !
              ELSE IF (NFREAT == 1 .AND. NFIXAT.EQ.2) THEN
                 !
                 !     Planar geometry, simply place free hydrogen.
                 !
                 CALL HPLACE(DONOR,FIXDAT(1),FIXDAT(2),FREEAT(1),0, &
                      FREEBD(1),ZERO,FRFXAN(1,1),FRFXAN(1,2),ZERO,ZERO, &
                      ONE8TY,ZERO,X,Y,Z)
                 !
                 !---- WRITE-INFO-PLACE
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 DO J=1,NFREAT
                    CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                 ENDDO
                 DO J=NFREAT+1,3
                    ACHY(J)='    '
                 ENDDO
                 IF(PRNLEV >= 2) WRITE(UNIT,960) &
                      (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                      RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

              ELSE IF (NFREAT == 1 .AND. NFIXAT.EQ.3) THEN
                 !
                 !     Tetrahedral geometry with three known atoms, place free atom.
                 !
                 !---- Begin Procedure DETERMINE-CHIRALITY-OF-FIXED-ATOMS
                 !
                 DDY(1)=X(FIXDAT(1))-X(DONOR)
                 DDY(2)=Y(FIXDAT(1))-Y(DONOR)
                 DDY(3)=Z(FIXDAT(1))-Z(DONOR)
                 DDZ(1)=X(FIXDAT(2))-X(DONOR)
                 DDZ(2)=Y(FIXDAT(2))-Y(DONOR)
                 DDZ(3)=Z(FIXDAT(2))-Z(DONOR)
                 CALL FINDBS(DDX,DDY,DDZ,2,ERROR)
                 IF (ERROR) THEN
                    CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                    IF(WRNLEV >= 2) WRITE(UNIT,155) SIDDN(1:idleng), &
                         RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
                 ENDIF
155              FORMAT(' ERROR FROM HBUILD:',/, &
                      ' Distance zero for atoms bonded to DONOR',4(1X,A))
                 CHIRAL=(X(FIXDAT(3))-X(DONOR))*DDX(1)+ &
                      (Y(FIXDAT(3))-Y(DONOR))*DDX(2)+ &
                      (Z(FIXDAT(3))-Z(DONOR))*DDX(3)
                 IF (CHIRAL < 0.0) THEN
                    CHIRAL=1.0
                 ELSE
                    CHIRAL=-1.0
                 ENDIF
                 !---- End Procedure DETERMINE-CHIRALITY-OF-FIXED-ATOMS
                 !
                 CALL HPLACE(DONOR,FIXDAT(1),FIXDAT(2),FREEAT(1),0, &
                      FREEBD(1),ZERO,FRFXAN(1,1),FRFXAN(1,2), &
                      ZERO,ZERO,CHIRAL*ONE2TY,ZERO,X,Y,Z)
                 !
                 !---- WRITE-INFO-PLACE
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 DO J=1,NFREAT
                    CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                 ENDDO
                 DO J=NFREAT+1,3
                    ACHY(J)='    '
                 ENDDO
                 IF(PRNLEV >= 2) WRITE(UNIT,960) &
                      (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                      RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

              ELSE IF (NFREAT == 2 .AND. NFIXAT.EQ.2) THEN
                 !
                 !     Tetrahedral geometry with two known atoms, simply place two free
                 !     hydrogens.
                 !
                 CALL HPLACE(DONOR,FIXDAT(1),FIXDAT(2),FREEAT(1), &
                      FREEAT(2),FREEBD(1),FREEBD(2), &
                      FRFXAN(1,1),FRFXAN(1,2),FRFXAN(2,1), &
                      FRFXAN(2,2),ONE2TY,-ONE2TY,X,Y,Z)
                 !
                 !---- WRITE-INFO-PLACE
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 DO J=1,NFREAT
                    CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                 ENDDO
                 DO J=NFREAT+1,3
                    ACHY(J)='    '
                 ENDDO
                 IF(PRNLEV >= 2) WRITE(UNIT,960) &
                      (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                      RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

              ELSE
                 !---- WRITE-NO-GEOMETRY
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 IF(WRNLEV >= 2) WRITE(UNIT,950) SIDDN(1:idleng), &
                      RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
              ENDIF
           ENDIF
        ENDIF
     ENDIF
  enddo loop100     !IF (.NOT.(DONOR >= NATOM)) GOTO 100
  !
  !     Begin section for spinning donor groups
  !
  DONOR=0
  do while(donor < natom)
     DONOR=DONOR+1
     IF (HFLAGS(DONOR) == 2) THEN
        IF (HHCODE(DONOR) == 100) THEN
           CALL FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
                FREEAT,FIXDAT,X,Y,Z)
           CALL GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
                FREEBD,FRFRAN,FRFXAN,IOFF)
           !
           !---- Begin Procedure SEARCH-PSF-DONOR-LIST
           !
           !     determines for atom DONOR a first and a last pointer into
           !     the PSF donor list (IDON, IHD1).
           !
           DONOFS=0
310        CONTINUE
           DONOFS=DONOFS+1
           IF (.NOT.(DONOR == IDON(DONOFS).OR.DONOFS >= NDON))  &
                GOTO 310
           DONOLS=DONOFS-1
320        CONTINUE
           DONOLS=DONOLS+1
           IF (.NOT.(DONOR /= IDON(DONOLS).OR.DONOLS >= NDON)) &
                GOTO 320
           IF (.NOT. (IDON(DONOLS) == DONOR.AND.DONOLS.EQ.NDON)) &
                DONOLS=DONOLS-1
           !---- End Procedure SEARCH-PSF-DONOR-LIST
           !
           !---- Begin Procedure SEARCH-DONOR-DIHEDRALS
           !
           !     Searches for all dihedrals with the dihedral axis (J,K)
           !     where either J or K is the DONOR atom.
           !
           NDIHDL=0
           DO J=1,NPHI
              IF (JP(J) == DONOR .OR. KP(J).EQ.DONOR) THEN
                 NDIHDL=NDIHDL+1
                 IF (NDIHDL <= MAXDIH) THEN
                    IDIHDL(NDIHDL)=J
                 ELSE
                    CALL WRNDIE(-4,'<HBUILD>', &
                         ' Maximum number of dihedrals MAXDIH exceeded. ' &
                         //'Check source.')
                 ENDIF
              ENDIF
           ENDDO
           !---- End Procedure SEARCH-DONOR-DIHEDRALS
           !
           !---- CONSTRUCT-NONBONDED-LIST-FOR-DONOR
           CALL NBONDX(X,Y,Z,BNBND, &
                NFREAT,FREEAT,X(DONOR),Y(DONOR),Z(DONOR))

           IF (NFREAT == 0) THEN
              !
              !     Do nothing.
              !
           ELSE IF (NFREAT == 1 .AND. NFIXAT.EQ.1) THEN
              !
              !     Spin donor group around bond (DONOR---FIXDAT(1))
              !     to find lowest energy configuration.
              !
              CALL ROTBASE(UNIT,DONOR,NDIHDL,IDIHDL,FIXDAT, &
                   DDX,DDY,DDZ,X,Y,Z,FRFXAN,CSTE,SNTE,ERROR)
              IF (.NOT. ERROR) THEN
                 PHI=0.0
                 IF (ABS(FRFXAN(1,1)-180.0) > 1.0E-04) THEN
                    !
                    !     If (fixed atom)-donor-(free atom) linear,
                    !     there is no need for a spin.
                    !
                    LOWERG=RBIG
                    DO WHILE (PHI < 359.0)
                       ATOM=1

                       CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD, &
                             CSTE,SNTE,DONOR,ATOM,FREEAT)
                       CALL CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL, &
                             BNBND,IOFF,X,Y,Z,PHI,ENERGY,QPRINT)

                       IF (ENERGY  <=  LOWERG) THEN
                          BESPHI=PHI
                          LOWERG=ENERGY
                       ENDIF
                       PHI=PHI+PHISTP
                    ENDDO
                    PHI=BESPHI
                 ENDIF

                 ATOM=1
                 CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                      DONOR,ATOM,FREEAT)
                 !
                 !---- WRITE-INFO-SPIN
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 DO J=1,NFREAT
                    CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                 ENDDO
                 DO J=NFREAT+1,3
                    ACHY(J)='    '
                 ENDDO
                 IF(PRNLEV >= 2) WRITE(UNIT,970) &
                      (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                      RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
                 !               End Procedure WRITE-INFO-SPIN
              ENDIF

           ELSE IF (NFREAT == 2 .AND. NFIXAT.EQ.1) THEN
              !
              !     Spin donor group around bond (DONOR---FIXDAT(1))
              !     to find lowest energy configuration with two "free" hydrogens.
              !
              CALL ROTBASE(UNIT,DONOR,NDIHDL,IDIHDL,FIXDAT, &
                   DDX,DDY,DDZ,X,Y,Z,FRFXAN,CSTE,SNTE,ERROR)
              IF (.NOT. ERROR) THEN
                 IF (ABS(FRFXAN(1,1)-180.0) < 1.0E-04) THEN
                    !---- WRITE-NO-GEOMETRY
                    CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                    IF(WRNLEV >= 2) WRITE(UNIT,950) SIDDN(1:idleng), &
                         RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

                 ELSE
                    !
                    !---- Begin Procedure ADJUST-PHIMAX-ACCORDING-TO-SYMMETRY
                    CONDIT=.TRUE.
                    DO J=1,NFREAT-1
                       CONDIT=CONDIT.AND. &
                            ABS(CG(FREEAT(J))-CG(FREEAT(J+1))) < 1.0E-4 &
                            .AND.IAC(FREEAT(J)) == IAC(FREEAT(J+1))
                    ENDDO
                    IF (CONDIT) THEN
                       PHIMAX=360.0/NFREAT - 1.0
                    ELSE
                       PHIMAX=359.0
                    ENDIF
                    !---- End Procedure ADJUST-PHIMAX-ACCORDING-TO-SYMMETRY
                    !
                    PHI=0.0
                    LOWERG=RBIG
                    DO WHILE (PHI < PHIMAX)
                       ATOM=1
                       CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD, &
                             CSTE,SNTE,DONOR,ATOM,FREEAT)

                       ! now the second free atom
                       CALL HPLACE(DONOR,FIXDAT(1),FREEAT(1),FREEAT(2),0, &
                             FREEBD(2),ZERO, &
                             FRFXAN(2,1),FRFRAN(1,2),ZERO,ZERO, &
                             ONE8TY,ZERO,X,Y,Z)

                       CALL CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL, &
                             BNBND,IOFF,X,Y,Z,PHI,ENERGY,QPRINT)

                       IF (ENERGY  <=  LOWERG) THEN
                          BESPHI=PHI
                          LOWERG=ENERGY
                       ENDIF
                       PHI=PHI+PHISTP
                    ENDDO
                    PHI=BESPHI

                    ATOM=1
                    CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD, &
                         CSTE,SNTE,DONOR,ATOM,FREEAT)

                    !                 now the second free atom
                    CALL HPLACE(DONOR,FIXDAT(1),FREEAT(1),FREEAT(2),0, &
                         FREEBD(2),ZERO, &
                         FRFXAN(2,1),FRFRAN(1,2),ZERO,ZERO, &
                         ONE8TY,ZERO,X,Y,Z)
                    !---- WRITE-INFO-SPIN
                    CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                    DO J=1,NFREAT
                       CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                    ENDDO
                    DO J=NFREAT+1,3
                       ACHY(J)='    '
                    ENDDO
                    IF(PRNLEV >= 2) WRITE(UNIT,970) &
                         (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                         RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

                 ENDIF
              ENDIF

           ELSE IF (NFREAT == 3 .AND. NFIXAT.EQ.1) THEN
              !
              !     Spin donor group around bond (DONOR---FIXDAT(1))
              !     to find lowest energy configuration with three "free" hydrogens.
              !
              CALL ROTBASE(UNIT,DONOR,NDIHDL,IDIHDL,FIXDAT, &
                   DDX,DDY,DDZ,X,Y,Z,FRFXAN,CSTE,SNTE,ERROR)
              IF (.NOT. ERROR) THEN
                 IF (ABS(FRFXAN(1,1)-180.0) < 1.0E-04) THEN
                    !---- WRITE-NO-GEOMETRY
                    CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                    IF(WRNLEV >= 2) WRITE(UNIT,950) SIDDN(1:idleng), &
                         RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

                 ELSE
                    !
                    !---- Begin Procedure ADJUST-PHIMAX-ACCORDING-TO-SYMMETRY
                    CONDIT=.TRUE.
                    DO J=1,NFREAT-1
                       CONDIT=CONDIT.AND.ABS(CG(FREEAT(J))- &
                            CG(FREEAT(J+1))) < 1.0E-4 &
                            .AND.IAC(FREEAT(J)) == IAC(FREEAT(J+1))
                    ENDDO
                    IF (CONDIT) THEN
                       PHIMAX=360.0/NFREAT - 1.0
                    ELSE
                       PHIMAX=359.0
                    ENDIF
                    !---- End Procedure ADJUST-PHIMAX-ACCORDING-TO-SYMMETRY
                    !
                    PHI=0.0
                    LOWERG=RBIG
                    DO WHILE (PHI < PHIMAX)
                       ATOM=1
                       CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD, &
                             CSTE,SNTE,DONOR,ATOM,FREEAT)

                       ! Now the second and third free atom,
                       CALL HPLACE(DONOR,FIXDAT(1),FREEAT(1),FREEAT(2), &
                             FREEAT(3),FREEBD(2),FREEBD(3), &
                             FRFXAN(2,1),FRFRAN(2,1),FRFXAN(3,1), &
                             FRFRAN(3,1),ONE2TY,-ONE2TY,X,Y,Z)
                       CALL CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL, &
                             BNBND,IOFF,X,Y,Z,PHI,ENERGY,QPRINT)

                       IF (ENERGY  <=  LOWERG) THEN
                          BESPHI=PHI
                          LOWERG=ENERGY
                       ENDIF
                       PHI=PHI+PHISTP
                    ENDDO
                    PHI=BESPHI

                    ATOM=1
                    CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD, &
                         CSTE,SNTE,DONOR,ATOM,FREEAT)
                    !
                    !                 Now the second and third free atom,
                    CALL HPLACE(DONOR,FIXDAT(1),FREEAT(1),FREEAT(2), &
                         FREEAT(3), &
                         FREEBD(2),FREEBD(3), &
                         FRFXAN(2,1),FRFRAN(2,1),FRFXAN(3,1), &
                         FRFRAN(3,1),ONE2TY,-ONE2TY,X,Y,Z)
                    !
                    !---- WRITE-INFO-SPIN
                    CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                    DO J=1,NFREAT
                       CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
                    ENDDO
                    DO J=NFREAT+1,3
                       ACHY(J)='    '
                    ENDDO
                    IF(PRNLEV >= 2) WRITE(UNIT,970) &
                         (ACHY(J)(1:idleng),J=1,3),SIDDN(1:idleng), &
                         RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

                 ENDIF
              ENDIF

           ELSE
              !
              !---- WRITE-NO-GEOMETRY
              CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
              IF(WRNLEV >= 2) WRITE(UNIT,950) SIDDN(1:idleng), &
                   RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)

           ENDIF
        ENDIF
     ENDIF
  enddo      ! IF (.NOT.(DONOR >= NATOM)) GOTO 300
  !
  !     Begin section for water, first determine water sequence
  !
  !---- Begin Procedure DETERMINE-WATER-SEQUENCE
  !
  !     this section determines a water sequence which is based on the
  !     distance from the protein. The distance of a water from the
  !     protein is defined by the minimum distance between the water
  !     oxygen and all other atoms not part of a water molecule.
  !     A permutation WATPER is evaluated, waters near the protein
  !     are at the top of the list.
  !
  NWATER=0
  loop550: DO WATER=1,NATOM
     IF (HFLAGS(WATER) == 2.AND.HHCODE(WATER).EQ.10) THEN
        !
        !         the condition above selects oxygens of waters
        !
        NWATER=NWATER+1
        LOWDIS=RBIG/2.0
        XWAT=X(WATER)
        YWAT=Y(WATER)
        ZWAT=Z(WATER)
        DO J=1,NATOM
           IF (HHCODE(J) /= 10) THEN
              XD=X(J)-XWAT
              YD=Y(J)-YWAT
              ZD=Z(J)-ZWAT
              DIST2=XD*XD+YD*YD+ZD*ZD
              IF (DIST2 <= LOWDIS) LOWDIS=DIST2
           ENDIF
        enddo
        WATPRT(WATER)=LOWDIS
     ELSE
        WATPRT(WATER)=RBIG
     ENDIF
  enddo loop550     !550     CONTINUE
  CALL SORTP(NATOM,WATPER,ORDERR,WATPRT,1,0,0,0,0,0,0)
  !
  !---- End Procedure DETERMINE-WATER-SEQUENCE
  !
  !     now the waters with acceptors defining a plane,
  !
  II=0
  loop600: do while(ii < nwater)
     II=II+1
     DONOR=WATPER(II)
     CALL FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
          FREEAT,FIXDAT,X,Y,Z)
     CALL GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
          FREEBD,FRFRAN,FRFXAN,IOFF)
     !
     !---- Begin Procedure SEARCH-PSF-DONOR-LIST
     !
     !     determines for atom DONOR a first and a last pointer into
     !     the PSF donor list (IDON, IHD1).
     !
     DONOFS=0
610  CONTINUE
     DONOFS=DONOFS+1
     IF (.NOT.(DONOR == IDON(DONOFS).OR.DONOFS >= NDON)) GOTO 610
     DONOLS=DONOFS-1
620  CONTINUE
     DONOLS=DONOLS+1
     IF (.NOT.(DONOR /= IDON(DONOLS).OR.DONOLS >= NDON)) GOTO 620
     IF (.NOT. (IDON(DONOLS) == DONOR.AND.DONOLS.EQ.NDON)) THEN
        DONOLS=DONOLS-1
     ENDIF
     !---- End ProcedureSEARCH-PSF-DONOR-LIST
     !
     !---- Begin Procedure FIND-HBONDS-TO-WATER-WITHIN-CUTHB
     !
     !     This section is used for waters.
     !     Find all hbonds to DONOR (without referencing hydrogens) within
     !     donor-acceptor distance cutoff CUTWAT.
     !
     CUTHB2=CUTWAT*CUTWAT
     NHB=0
     DO J=1,NWATAC
        IF (WATACC(J) /= DONOR) THEN
           XD=X(DONOR)-X(WATACC(J))
           YD=Y(DONOR)-Y(WATACC(J))
           ZD=Z(DONOR)-Z(WATACC(J))
           IF (XD*XD+YD*YD+ZD*ZD < CUTHB2) THEN
              IF (NHB >= MAXHB) THEN
                 IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
                      ' %HBUILD-ERR: MAXHB (HBOND) exceeded'
              ELSE
                 NHB=NHB+1
                 IHB(NHB)=DONOR
                 JHB(NHB)=WATACC(J)
                 KHB(NHB)=0
                 LHB(NHB)=0
              ENDIF
           ENDIF
        ENDIF
     ENDDO
     !---- End Procedure FIND-HBONDS-TO-WATER-WITHIN-CUTHB
     !
     !---- Begin Procedure SEPARATE-ISOLATED-WATERS
     !
     !     Check number of hydrogen bonds formed with DONOR.
     !
     IF (NHB == 0) THEN
        SPARTD=.TRUE.
        HHCODE(DONOR)=10000
     ELSE
        SPARTD=.FALSE.
     ENDIF
     !---- End Procedure SEPARATE-ISOLATED-WATERS
     !
     IF (.NOT. SPARTD) THEN
        !
        !---- Begin Procedure CONSTRUCT-H-O-H-PLANE
        !
        !     Using the hbond list generated by the procedure
        !     FIND-HBONDS-TO-WATER-WITHIN-CUTHB, this section determines
        !     the two nearest acceptors of the water.
        !     These two acceptors and the water oxygen define a plane.
        !     If only one acceptor is present PLANE=.FALSE. is returned
        !
        IF (NHB > NACC+NST2) THEN
           CALL WRNDIE(-4,'<HBUILD>', &
                'NHB greater max. dimension of RWORK or IWORK. Check ' &
                //'source.')
        ENDIF
        DO J=1,NHB
           DDX(1)=X(IHB(J))-X(JHB(J))
           DDX(2)=Y(IHB(J))-Y(JHB(J))
           DDX(3)=Z(IHB(J))-Z(JHB(J))
           RWORK(J)=DDX(1)*DDX(1)+DDX(2)*DDX(2)+DDX(3)*DDX(3)
        ENDDO
        !
        !         sort the quadratic distances stored in RWORK (get perm. IWORK)
        !
        CALL SORTP(NHB,IWORK,ORDERR,RWORK,1,0,0,0,0,0,0)
        !
        !         the plane is defined by DONOR, JHB(BESHB) and JHB(BES2HB)
        !
        BESHB=IWORK(1)
        PLANE=(NHB > 1)
        IF (PLANE) BES2HB=IWORK(2)
        !
        !---- End Procedure CONSTRUCT-H-O-H-PLANE
        !
        IF (.NOT. PLANE) THEN
           !
           !     here the waters with only one acceptor,
           !
           !---- Begin Procedure PLACE-ONE-HYDROGEN-ON-BEST-HBOND
           !
           !     Places one hydrogen on "best" hydrogen bond.
           !
           DDX(1)=X(JHB(BESHB))-X(DONOR)
           DDX(2)=Y(JHB(BESHB))-Y(DONOR)
           DDX(3)=Z(JHB(BESHB))-Z(DONOR)
           DDY(1)=0.0
           DDY(2)=DDX(3)
           DDY(3)=-DDX(2)
           CALL FINDBS(DDX,DDY,DDZ,1,ERROR)
           IF (ERROR) THEN
              IF(WRNLEV >= 2) WRITE(UNIT,625) JHB(BESHB),DONOR
              CALL DIEWRN(-4)
              RETURN
           ENDIF
625        FORMAT(' ERROR FROM HBUILD:',/,  &
                ' distance zero between atoms',2I5)

           X(FREEAT(1))=FREEBD(1)*DDX(1)+X(DONOR)
           Y(FREEAT(1))=FREEBD(1)*DDX(2)+Y(DONOR)
           Z(FREEAT(1))=FREEBD(1)*DDX(3)+Z(DONOR)
           HFLAGS(FREEAT(1))=0
           !
           !     the statement above defines FREEAT(1) as known.
           !
           !---- End Procedure PLACE-ONE-HYDROGEN-ON-BEST-HBOND
           !
           CALL FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
                FREEAT,FIXDAT,X,Y,Z)
           CALL GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
                FREEBD,FRFRAN,FRFXAN,IOFF)
           NDIHDL=0
           !
           !---- CONSTRUCT-NONBONDED-LIST-FOR-DONOR
           CALL NBONDX(X,Y,Z,BNBND, &
                NFREAT,FREEAT,X(DONOR),Y(DONOR),Z(DONOR))

           CALL ROTBASE(UNIT,DONOR,NDIHDL,IDIHDL,FIXDAT, &
                DDX,DDY,DDZ,X,Y,Z,FRFXAN,CSTE,SNTE,ERROR)

           IF (.NOT. ERROR) THEN
              PHI=0.0
              LOWERG=RBIG
              DO WHILE (PHI < 359)
                 ATOM=1
                 CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                       DONOR,ATOM,FREEAT)

                 CALL PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
                       FREEBD,FRFXAN,FRFRAN,X,Y,Z)

                 CALL CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL, &
                       BNBND,IOFF,X,Y,Z,PHI,ENERGY,QPRINT)

                 IF (ENERGY  <=  LOWERG) THEN
                    BESPHI=PHI
                    LOWERG=ENERGY
                 ENDIF
                 PHI=PHI+PHISTP
              ENDDO
              PHI=BESPHI

              ATOM=1
              CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                   DONOR,ATOM,FREEAT)

              CALL PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
                   FREEBD,FRFXAN,FRFRAN,X,Y,Z)
              !
              !---- Begin Procedure WRITE-WATER-INFO-SPIN1
              !
              CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
              CALL ATOMID(FIXDAT(1),SIDDN,RIDDN,RESDN,ACHY(1))
              DO J=1,NFREAT
                 CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J+1))
              ENDDO
              DO J=NFREAT+2,4
                 ACHY(J)='    '
              ENDDO

              IF(PRNLEV >= 2) WRITE(UNIT,655)  &
                   ACHY(1)(1:idleng),ACHY(2)(1:idleng), &
                   ACHY(3)(1:idleng),ACHY(4)(1:idleng), &
                   SIDDN(1:idleng),RIDDN(1:idleng), &
                   RESDN(1:idleng),ACDN(1:idleng)
              IF(WATPRT(DONOR) < RBIG/2.1) THEN
                 IF(PRNLEV >= 2) WRITE(UNIT,665) SQRT(WATPRT(DONOR))
              ENDIF

655           FORMAT(' Place:',A,', Spin:',A,',',A,',',A, &
                   ' constructed for water',4(1X,A),'.')
665           FORMAT(' Minimum distance between this water', &
                   ' and protein is ',F8.5,' A.')
              !
              !---- End Procedure WRITE-WATER-INFO-SPIN1
              !
           ENDIF
        ELSE
           !
           !     and here the waters with at least two acceptors,
           !
           NDIHDL=0
           !---- CONSTRUCT-NONBONDED-LIST-FOR-DONOR
           CALL NBONDX(X,Y,Z,BNBND, &
                NFREAT,FREEAT,X(DONOR),Y(DONOR),Z(DONOR))
           !
           !---- Begin Procedure DEFINE-WATER-ORTHO-BASE
           !
           !     Given the the position of two best acceptors and donor
           !     an orthonormal base DDX, DDY, DDZ is determined with DDX
           !     perpendicular to the plane defined by the acceptors and the donor.
           !
           DDY(1)=X(JHB(BESHB))-X(DONOR)
           DDY(2)=Y(JHB(BESHB))-Y(DONOR)
           DDY(3)=Z(JHB(BESHB))-Z(DONOR)
           DDZ(1)=X(JHB(BES2HB))-X(DONOR)
           DDZ(2)=Y(JHB(BES2HB))-Y(DONOR)
           DDZ(3)=Z(JHB(BES2HB))-Z(DONOR)
           CALL FINDBS(DDX,DDY,DDZ,2,ERROR)
           IF (ERROR) THEN
              CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
              IF(WRNLEV >= 2) WRITE(UNIT,675) SIDDN(1:idleng), &
                   RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
              CALL DIEWRN(-4)
              RETURN
           ENDIF
           CSTE=0.0
           SNTE=1.0

675        FORMAT(' ERROR FROM HBUILD:',/, &
                ' Cant define a plane for the water',4(1X,A))
           !
           !---- End Procedure DEFINE-WATER-ORTHO-BASE
           !
           !     now rotate the water in the plane defined by the best two
           !     acceptors to find a low energy configuration.
           !
           PHI=0.0
           LOWERG=RBIG
           DO WHILE (PHI < 359)
              ATOM=1
              CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                    DONOR,ATOM,FREEAT)
              PHI=PHI+FRFRAN(1,2)

              ATOM=2
              CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                    DONOR,ATOM,FREEAT)

              CALL PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
                    FREEBD,FRFXAN,FRFRAN,X,Y,Z)

              PHI=PHI-FRFRAN(1,2)
              CALL CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL, &
                    BNBND,IOFF,X,Y,Z,PHI,ENERGY,QPRINT)

              IF (ENERGY  <=  LOWERG) THEN
                 BESPHI=PHI
                 LOWERG=ENERGY
              ENDIF
              PHI=PHI+PHISTP
           ENDDO
           PHI=BESPHI

           ATOM=1
           CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                DONOR,ATOM,FREEAT)
           PHI=BESPHI+FRFRAN(1,2)

           ATOM=2
           CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
                DONOR,ATOM,FREEAT)

           CALL PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
                FREEBD,FRFXAN,FRFRAN,X,Y,Z)
           !
           !---- Begin Procedure WRITE-WATER-INFO-SPIN2
           CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
           DO J=1,NFREAT
              CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
           ENDDO
           DO J=NFREAT+1,4
              ACHY(J)='    '
           ENDDO
           IF(PRNLEV >= 2) WRITE(UNIT,715)  &
                ACHY(1)(1:idleng),ACHY(2)(1:idleng), &
                ACHY(3)(1:idleng),ACHY(4)(1:idleng), &
                SIDDN(1:idleng),RIDDN(1:idleng), &
                RESDN(1:idleng),ACDN(1:idleng)
           IF(WATPRT(DONOR) < RBIG/2.1) THEN
              IF(PRNLEV >= 2) WRITE(UNIT,725) SQRT(WATPRT(DONOR))
           ENDIF
715        FORMAT(' Spin:',A,',',A,',',A,',',A, &
                ' constructed for water',4(1X,A),'.')
725        FORMAT(' Minimum distance between this water', &
                ' and protein is ',F8.5,' A.')
           !---- End Procedure WRITE-WATER-INFO-SPIN2
           !
        ENDIF
     ENDIF
  enddo loop600    !IF (.NOT.(II >= NWATER)) GOTO 600
  !
  !     now, place waters with no acceptors in a standard way,
  !
  II=0
  loop900: do while(ii < nwater)   !IF (II >= NWATER) GOTO 900
     II=II+1
     DONOR=WATPER(II)
     IF (HHCODE(DONOR) == 10000) THEN
        CALL FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
             FREEAT,FIXDAT,X,Y,Z)
        CALL GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
             FREEBD,FRFRAN,FRFXAN,IOFF)
        !
        !---- Begin Procedure DEFINE-STANDARD-ORTHO-BASE
        DDX(1)=1.0
        DDX(2)=0.0
        DDX(3)=0.0
        DDY(1)=0.0
        DDY(2)=1.0
        DDY(3)=0.0
        DDZ(1)=0.0
        DDZ(2)=0.0
        DDZ(3)=1.0
        CSTE=0.0
        SNTE=1.0
        !---- End Procedure DEFINE-STANDARD-ORTHO-BASE
        !
        PHI=0.0
        ATOM=1
        CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
             DONOR,ATOM,FREEAT)
        PHI=FRFRAN(1,2)

        ATOM=2
        CALL PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
             DONOR,ATOM,FREEAT)

        CALL PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
             FREEBD,FRFXAN,FRFRAN,X,Y,Z)
        !
        !---- Begin Procedure WRITE-WATER-INFO-PLACE
        CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
        DO J=1,NFREAT
           CALL ATOMID(FREEAT(J),SIDDN,RIDDN,RESDN,ACHY(J))
        ENDDO
        DO J=NFREAT+1,4
           ACHY(J)='    '
        ENDDO
        IF(PRNLEV >= 2) WRITE(UNIT,855)  &
             ACHY(1)(1:idleng),ACHY(2)(1:idleng), &
             ACHY(3)(1:idleng),ACHY(4)(1:idleng), &
             SIDDN(1:idleng),RIDDN(1:idleng), &
             RESDN(1:idleng),ACDN(1:idleng)
        IF(WATPRT(DONOR) < RBIG/2.1) THEN
           IF(PRNLEV >= 2) WRITE(UNIT,865) SQRT(WATPRT(DONOR))
        ENDIF
855     FORMAT(' Place:',A,',',A,',',A,',',A, &
             ' constructed for water',4(1X,A),'.')
865     FORMAT(' Minimum distance between this water', &
             ' and protein is ',F8.5,' A.')
        !---- End Procedure WRITE-WATER-INFO-PLACE
        !
     ENDIF
  enddo loop900
  !
  !     Finally, make shure that hydrogen bond list is reset.
  !
  NHB=0

950 FORMAT(' WARNING FORM HBUILD.'/, &
       ' Geometry not programmed for donor ',4(1X,A),/, &
       '. No attempt was made to build hydrogens for ', &
       'this donor.')
960 FORMAT(' Place: ',3(A,','),' constructed for donor', &
       4(1X,A),'.')
970 FORMAT(' Spin: ',3(A,','),' constructed for donor', &
       4(1X,A),'.')

  RETURN
END SUBROUTINE HBUIL2

SUBROUTINE FXFRATM(UNIT,NFREAT,NFIXAT,DONOR,NATBON,IATBON,HFLAGS, &
     FREEAT,FIXDAT,X,Y,Z)
  !------------------------------------------------------------------------
  !     Procedure GET-FIXED-AND-FREE-ATOMS-BONDED-TO-DONOR
  !
  use chm_kinds
  use dimens_fcm
  use exfunc, only: order
  use stream
  use chutil,only:initia,atomid
  implicit none

  INTEGER UNIT,NFREAT,NFIXAT,DONOR
  INTEGER NATBON(*), IATBON(IATBMX,*), HFLAGS(*)
  INTEGER FREEAT(*), FIXDAT(*)
  real(chm_real)  X(*),Y(*),Z(*)

  EXTERNAL EXCH
  INTEGER J,BONDAT
  CHARACTER(len=8) SIDDN, RIDDN, RESDN, ACDN

  NFREAT=0
  NFIXAT=0
  DO J=1,NATBON(DONOR)
     BONDAT=IATBON(J,DONOR)
     IF (HFLAGS(BONDAT) == 1) THEN
        !
        !     add this hydrogen to the FREEAT list (if its coordinates
        !     are already known it is reconstructed.)
        !
        NFREAT=NFREAT+1
        FREEAT(NFREAT)=BONDAT
     ELSE
        !
        !     check that the other atoms bonded to DONOR are actually known
        !     and add them to the FIXDAT list.
        !
        IF (INITIA(BONDAT,X,Y,Z)) THEN
           NFIXAT=NFIXAT+1
           FIXDAT(NFIXAT)=BONDAT
        ELSE
           CALL ATOMID(BONDAT,SIDDN,RIDDN,RESDN,ACDN)
           IF(WRNLEV >= 2) WRITE(UNIT,115) SIDDN(1:idleng), &
                RIDDN(1:idleng),RESDN(1:idleng),ACDN(1:idleng)
           CALL DIEWRN(0)
           RETURN
        ENDIF
     ENDIF
  enddo
115 FORMAT(' WARNING FROM HBUILD:',/, &
       ' Attempt to construct a hydrogen forming an angle ', &
       ' with the unknown atom',4(1X,A))

  !
  !     finally, sort the fixed and the free listing.
  !
  CALL SORT(NFREAT,EXCH,ORDER,FREEAT,1,0,0,0,0,0,0)
  CALL SORT(NFIXAT,EXCH,ORDER,FIXDAT,1,0,0,0,0,0,0)

  RETURN
END SUBROUTINE FXFRATM

SUBROUTINE GETEQGM(UNIT,DONOR,NFREAT,NFIXAT,FREEAT,FIXDAT, &
     FREEBD,FRFRAN,FRFXAN,IOFF)
  !-----------------------------------------------------------------------
  !     Procedure GET-EQUILIBRIUM-GEOMETRY
  !
  use chm_kinds
  use dimens_fcm
  use number

  use consta
  use psf
  use param
  use stream
  implicit none

  INTEGER UNIT,DONOR,NFREAT,NFIXAT
  INTEGER FREEAT(*),FIXDAT(*),IOFF(*)
  real(chm_real)  FREEBD(*), FRFRAN(IATBMX,IATBMX),  &
       FRFXAN(IATBMX,IATBMX)

  INTEGER J,K,I1,J1,K1,CODE,KEY
  INTEGER QAT(5)
  !
  !     First the "free" equilibrium bond values,
  !
  J1=IAC(DONOR)
  DO J=1,NFREAT
     I1=IAC(FREEAT(J))
     QAT(1)=DONOR
     QAT(2)=FREEAT(J)
     CALL CODES(QAT(3),[0],[0],[0], &
          NATOM,IMOVE,IAC,1,QAT(1),QAT(2), &
          0,[0],[0],[0],0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
          .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
          [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          & 
#endif
          .TRUE.,.TRUE.)
     KEY=QAT(3)
     IF (KEY == 0) THEN
        IF(WRNLEV >= 2) WRITE(UNIT,185) ATC(I1),ATC(J1)
        CALL DIEWRN(-4)
        RETURN
     ELSE
        FREEBD(J)=CBB(KEY)
     ENDIF
  ENDDO
  !
  !     now the "free-fixed" equilibrium angles,
  !
  DO J=1,NFREAT
     DO K=1,NFIXAT
        I1=IAC(FREEAT(J))
        K1=IAC(FIXDAT(K))
        QAT(1)=FREEAT(J)
        QAT(2)=DONOR
        QAT(3)=FIXDAT(K)
        CALL CODES([0],QAT(4),[0],[0], &
             NATOM,IMOVE,IAC,0,[0],[0], &
             1,QAT(1),QAT(2),QAT(3), &
             0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
             .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
             [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          & 
#endif
             .TRUE.,.TRUE.)
        KEY=QAT(4)
        IF (KEY == 0) THEN
           IF(WRNLEV >= 2) WRITE(UNIT,195) ATC(I1),ATC(J1),ATC(K1)
           CALL DIEWRN(-4)
           RETURN
        ELSE
           FRFXAN(J,K)=CTB(KEY)*RADDEG
        ENDIF
     ENDDO
  ENDDO
  !
  !     now the "free-free" equilibrium angles.
  !
  DO J=1,NFREAT
     DO K=J+1,NFREAT
        I1=IAC(FREEAT(J))
        K1=IAC(FREEAT(K))
        QAT(1)=FREEAT(J)
        QAT(2)=DONOR
        QAT(3)=FREEAT(K)
        CALL CODES([0],QAT(4),[0],[0], &
             NATOM,IMOVE,IAC,0,[0],[0], &
             1,QAT(1),QAT(2),QAT(3), &
             0,[0],[0],[0],[0],0,[0],[0],[0],[0], &
             .FALSE.,0,                    & ! DRUDE
#if KEY_CMAP==1
             [0],0,[0],[0],[0],[0],[0],[0],[0],[0],          & 
#endif
             .TRUE.,.TRUE.)
        KEY=QAT(4)
        IF (KEY == 0) THEN
           IF(WRNLEV >= 2) WRITE(UNIT,195) ATC(I1),ATC(J1),ATC(K1)
        ELSE
           FRFRAN(J,K)=CTB(KEY)*RADDEG
           FRFRAN(K,J)=CTB(KEY)*RADDEG
        ENDIF
     ENDDO
  ENDDO

185 FORMAT(' ERROR FROM HBUILD:',/, &
       ' could not find bond parameters between atom types',2A7)
195 FORMAT(' ERROR FROM HBUILD:',/, &
       ' could not find angle parameters between atom types',3A7)

  RETURN
END SUBROUTINE GETEQGM

SUBROUTINE ROTBASE(UNIT,DONOR,NDIHDL,IDIHDL,FIXDAT, &
     DDX,DDY,DDZ,X,Y,Z,FRFXAN,CSTE,SNTE,ERROR)
  !-----------------------------------------------------------------------
  !     Procedure DEFINE-ROTATION-ORTHO-BASE
  !
  use chm_kinds
  use dimens_fcm
  use number

  use consta
  use psf
  use stream
  implicit none

  INTEGER UNIT,DONOR,NDIHDL
  INTEGER FIXDAT(*),IDIHDL(*)
  real(chm_real)  DDX(3),DDY(3),DDZ(3),X(*),Y(*),Z(*)
  real(chm_real)  FRFXAN(IATBMX,IATBMX),CSTE,SNTE
  LOGICAL ERROR

  real(chm_real)  THETA

  DDX(1)=X(FIXDAT(1))-X(DONOR)
  DDX(2)=Y(FIXDAT(1))-Y(DONOR)
  DDX(3)=Z(FIXDAT(1))-Z(DONOR)
  !
  !     the following definition of DDY identifies dihedral PHI with
  !     dihedral IP, JP, KP, LP (if present).
  !
  IF (NDIHDL > 0)then
     if(JP(IDIHDL(1)) == DONOR) THEN
        DDY(1)=X(LP(IDIHDL(1)))-X(KP(IDIHDL(1)))
        DDY(2)=Y(LP(IDIHDL(1)))-Y(KP(IDIHDL(1)))
        DDY(3)=Z(LP(IDIHDL(1)))-Z(KP(IDIHDL(1)))
     ELSE IF (KP(IDIHDL(1)) == DONOR) THEN
        DDY(1)=X(IP(IDIHDL(1)))-X(JP(IDIHDL(1)))
        DDY(2)=Y(IP(IDIHDL(1)))-Y(JP(IDIHDL(1)))
        DDY(3)=Z(IP(IDIHDL(1)))-Z(JP(IDIHDL(1)))
     ELSE
        DDY(1)=ZERO
        DDY(2)=DDX(3)
        DDY(3)=-DDX(2)
     ENDIF
  ELSE
     DDY(1)=ZERO
     DDY(2)=DDX(3)
     DDY(3)=-DDX(2)
  ENDIF
  CALL FINDBS(DDX,DDY,DDZ,1,ERROR)
  IF (ERROR) THEN
     IF(WRNLEV >= 2) WRITE(UNIT,45) FIXDAT(1),DONOR
     CALL DIEWRN(-4)
     RETURN
  ENDIF
  THETA=DEGRAD*FRFXAN(1,1)
  CSTE=COS(THETA)
  SNTE=SIN(THETA)

45 FORMAT(' ERROR FROM HBUILD:',/, &
       ' distance zero between atoms',2I5)

  RETURN
END SUBROUTINE ROTBASE

SUBROUTINE PLACEPHI(PHI,X,Y,Z,DDX,DDY,DDZ,FREEBD,CSTE,SNTE, &
     DONOR,ATOM,FREEAT)
  !-----------------------------------------------------------------------
  !     Procedure PLACE-ATOM-WITH-DIHEDRAL-PHI
  !
  !     places free ATOM with dihedral PHI and distance FREEBD(ATOM)
  !     with respect to the center DONOR and the defined base set
  !     DDX, DDY, DDZ.
  !
  use chm_kinds
  use dimens_fcm
  use consta
  implicit none

  real(chm_real)  PHI,CSTE,SNTE
  real(chm_real)  X(*),Y(*),Z(*),DDX(3),DDY(3),DDZ(3),FREEBD(*)
  INTEGER ATOM,DONOR,FREEAT(*)

  real(chm_real) CSPI,SNPI

  CSPI=COS(PHI*DEGRAD)
  SNPI=SIN(PHI*DEGRAD)
  X(FREEAT(ATOM))=FREEBD(ATOM)*(CSTE*DDX(1)+ &
       SNTE*(CSPI*DDY(1)-SNPI*DDZ(1)))+X(DONOR)
  Y(FREEAT(ATOM))=FREEBD(ATOM)*(CSTE*DDX(2)+ &
       SNTE*(CSPI*DDY(2)-SNPI*DDZ(2)))+Y(DONOR)
  Z(FREEAT(ATOM))=FREEBD(ATOM)*(CSTE*DDX(3)+ &
       SNTE*(CSPI*DDY(3)-SNPI*DDZ(3)))+Z(DONOR)

  RETURN
END SUBROUTINE PLACEPHI

SUBROUTINE PLCLONE(UNIT,DONOR,NFREAT,FREEAT,FIXDAT, &
     FREEBD,FRFXAN,FRFRAN,X,Y,Z)
  !-----------------------------------------------------------------------
  !     Procedure PLACE-LONE-PAIRS
  !     This routine places lone pairs based on the positions of
  !     the hydrogens.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use stream
  use chutil,only:lone,atomid
  implicit none

  INTEGER UNIT,DONOR,NFREAT
  INTEGER FREEAT(*), FIXDAT(*)
  real(chm_real)  X(*), Y(*), Z(*)
  real(chm_real)  FREEBD(*), FRFXAN(IATBMX,IATBMX),  &
       FRFRAN(IATBMX,IATBMX)

  CHARACTER(len=8) SIDDN, RIDDN, RESDN, ACDN

  IF (NFREAT == 4) THEN
     !
     !     this is true for the part of the algorithm using a rotation
     !     of the water in a plane
     !
     IF (.NOT. (LONE(FREEAT(3)).AND.LONE(FREEAT(4))) ) THEN
        CALL ATOMID(FREEAT(3),SIDDN,RIDDN,RESDN,ACDN)
        IF(WRNLEV >= 2) WRITE(UNIT,65) SIDDN(1:idleng), &
             RIDDN(1:idleng),RESDN(1:idleng)
        CALL DIEWRN(0)
        RETURN
     ENDIF
     CALL HPLACE(DONOR,FREEAT(1),FREEAT(2),FREEAT(3),FREEAT(4), &
          FREEBD(3),FREEBD(4), &
          FRFRAN(3,1),FRFRAN(3,2),FRFRAN(4,1), &
          FRFRAN(4,2),ONE2TY,-ONE2TY,X,Y,Z)

  ELSE IF (NFREAT == 3) THEN
     !
     !     this is true for water performing a rotation around the
     !     acceptor-hydrogen-oxygen axis.
     !
     IF (.NOT. (LONE(FREEAT(2)).AND.LONE(FREEAT(3))) ) THEN
        CALL ATOMID(FREEAT(3),SIDDN,RIDDN,RESDN,ACDN)
        IF(WRNLEV >= 2) WRITE(UNIT,65) SIDDN(1:idleng), &
             RIDDN(1:idleng),RESDN(1:idleng)
        CALL DIEWRN(0)
        RETURN
     ENDIF
     CALL HPLACE(DONOR,FIXDAT(1),FREEAT(1),FREEAT(2),FREEAT(3), &
          FREEBD(2),FREEBD(3), &
          FRFXAN(2,1),FRFRAN(2,1),FRFXAN(3,1), &
          FRFRAN(3,1),ONE2TY,-ONE2TY,X,Y,Z)
  ENDIF

65 FORMAT(' ERROR FROM HBUILD: missing lone pairs for ',3(1X,A))

  RETURN
END SUBROUTINE PLCLONE

SUBROUTINE CONFIEN(UNIT,DONOFS,DONOLS,NDIHDL,IDIHDL,BNBND,IOFF, &
     X,Y,Z,PHI,ENERGY,QPRINT)
  !-----------------------------------------------------------------------
  !     Procedure EVALUATE-PARTIAL-ENERGY-OF-THIS-CONFIGURATION
  !
#if KEY_MMFF==1
  !     Jay L. Banks, add the MMFF code 19-Oct-95
#endif 

  use ewald,only:
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor           
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number

  use consta
  use psf
  use param
  use code
  use deriv
  use eintern
  use enbonda
  use inbnd
  use hbondm
  use parallel,only: gcomb
  use stream
#if KEY_FLUCQ==1
  use flucq   
#endif
#if KEY_MMFF==1
  use ffieldm
  use mmffm
#endif 
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
  implicit none

  INTEGER UNIT,DONOFS,DONOLS,NDIHDL
  INTEGER IDIHDL(*),IOFF(*)
  type(nonbondDataStructure) BNBND
  real(chm_real)  X(*),Y(*),Z(*)
  real(chm_real)  PHI,ENERGY
  LOGICAL QPRINT

  INTEGER J
  real(chm_real)  LOCERG,EHBERG,DIHERG,VDWERG,ELCERG,ST2ERG

  EHBERG=0.0
  DIHERG=0.0
  VDWERG=0.0
  ELCERG=0.0
  ST2ERG=0.0
  !
  !     First, find all hydrogen bonds within CUTHB and CUTHBA
  !
  CALL HBFIND(UNIT,.FALSE.,IDON,IHD1,DONOFS,DONOLS, &
       IACC,IAC1,1,NACC,X,Y,Z,IHB,JHB,KHB,LHB,1, &
       NHB,MAXHB,LHBFG,CUTHB,CUTHBA)
  !
  !     now determine code array ICH for HBONDS,
  !
  CALL HCODES(ICH,NHB,IHB,JHB,IAC)
  !
  !     now evaluate the hyrogen bond energies,
  !
  CALL EHBOND(EHBERG,IHB,JHB,KHB,LHB,ICH,NHB, &
       CHBA,CHBB,DX,DY,DZ,X,Y,Z,.FALSE.,0, &
       0,0,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
       0,0,.FALSE.)
  ENERGY=EHBERG
  !
  !     now, evaluate the dihedral energy,
  !
  DO J=1,NDIHDL
#if KEY_DOMDEC==1
     if (q_domdec) then
        CALL WRNDIE(-5,'<CONFIEN>','NOT READY FOR DOMDEC')
     else
#endif 
        CALL EPHI(LOCERG,IP(IDIHDL(J)),JP(IDIHDL(J)),KP(IDIHDL(J)), &
             LP(IDIHDL(J)),ICP(IDIHDL(J)),1,CPC,CPD,CPB,CPCOS, &
             CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
             .FALSE.,(/ZERO/),0,(/0/),(/ZERO/),(/0/),.FALSE. &
             )
#if KEY_DOMDEC==1
     endif  
#endif

     DIHERG=DIHERG+LOCERG
  ENDDO
#if KEY_PARALLEL==1
  CALL GCOMB(DIHERG,1)          
#endif
  ENERGY=ENERGY+DIHERG
  !
  !     now, evaluate the van der Waals and electrostatic energy.
  !
#if KEY_MMFF==1
  if (ffield == charmm .or. ffield == amberffn) then
#endif 
     !
     !     QC: Bug fix 06/27/07
     CALL EVDW(VDWERG,ELCERG,1,NATOM,BNBND%JNB, &
          BNBND%INBLO,CG,RSCLF,CNBA,CNBB,MAXCN, &
          IAC,ITC,NATC,IOFF, &
          QETEN,QETSR,                                 & 
          LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT,LDBEXP, &
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,CGONNB,CGOFNB,EPS,E14FAC, &
#if KEY_NBIPS==1
          LVIPS,LEIPS,                                  & 
#endif
          LEGROM,LVGROM,                                &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                                 & 
#endif
#if KEY_WCA==1
          .FALSE., ZERO, WCA,                            & 
#endif
          .FALSE.,(/ZERO/),(/ZERO/),(/0/),.FALSE. &
#if KEY_IMCUBES==1
          ,lbycbim                              & 
#endif
          ,.FALSE.)  ! Wei Chen 2015
#if KEY_MMFF==1
  ELSE
     !yw...02-Feb-96 EVDW_MM call is moved here from EVDW
     CALL EVDW_MM(VDWERG,ELCERG,NATOM, &
          BNBND%JNB,BNBND%INBLO,CG,CNBA,CNBB,MAXCN, &
          IAC,ITC,NATC,IOFF, &
          LELEC,LVDW,LCONS,LSHFT,LVSHFT,LFSWT,LVFSWT, &
          DX,DY,DZ,X,Y,Z,CTONNB,CTOFNB,EPS,E14FAC, &
          .FALSE.,0,0,0,.FALSE.,LVTRUNC,CTVTRN,LMSHFT, &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,                        & 
#endif
          DELBUF,GAMBUF,DELQ &
#if KEY_IMCUBES==1
          ,lbycbim                              & 
#endif
          )
  ENDIF
#endif 

#if KEY_NOST2==0
  !     now, evaluate the ST2 energy (call ENST2 with mode .TRUE.)
  !
  IF (NST2 > 0) THEN
     CALL ENST2(ST2ERG,X,Y,Z,DX,DY,DZ,BNBND%JNBG, &
          BNBND%INBLOG,NGRP,CTONNB,CTOFNB,.FALSE.)
  ELSE
     ST2ERG=0.0
  ENDIF
#endif 

  ENERGY=ENERGY+VDWERG+ELCERG+ST2ERG
  IF (QPRINT .AND. PRNLEV >= 2) THEN
     IF (PHI < 1.0E-04) WRITE(OUTU,171)
     WRITE(OUTU,172) PHI,EHBERG,DIHERG,VDWERG,ELCERG, &
          ST2ERG,ENERGY
  ENDIF
171 FORMAT(1X,'rel. PHI',2X,' HBONDS',3X,'DIHEDRALS',2X,'  VDW  ', &
       3X,'  ELEC ',3X,'  ST2  ',3X,' ENERGY')
172 FORMAT(1X,F5.1,5X,6(F7.3,3X))

  RETURN
END SUBROUTINE CONFIEN

SUBROUTINE FINDBS(X,Y,Z,MODE,ERROR)
  !-----------------------------------------------------------------------
  !     MODE=1:
  !     Routine finds orthonormal axis set using X,Y.
  !     X-axis is parallel to x on input.
  !
  !     MODE=2:
  !     Routine finds orthonormal axis set with x-axis perpendicular to
  !     the plane defined by input vectors Y,Z.
  !
  !     Returns orthonormal axis-set in X,Y,Z.
  !
  !     DEC-82 Axel Brunger
  !
  use chm_kinds
  implicit none
  real(chm_real)    X(*), Y(*), Z(*)
  INTEGER   MODE
  LOGICAL   ERROR

  real(chm_real)  NORM

  ERROR=.FALSE.
  IF (MODE == 1) THEN
     NORM=SQRT(X(1)*X(1)+X(2)*X(2)+X(3)*X(3))
     IF (NORM < 1.0D-04) THEN
        ERROR=.TRUE.
     ELSE
        X(1)=X(1)/NORM
        X(2)=X(2)/NORM
        X(3)=X(3)/NORM
        !
        !         now subtract the projection of X to Y from Y,
        !
        NORM=(Y(1)*X(1)+Y(2)*X(2)+Y(3)*X(3))
        Y(1)=Y(1)-NORM*X(1)
        Y(2)=Y(2)-NORM*X(2)
        Y(3)=Y(3)-NORM*X(3)
        !
        !         now normalize Y,
        !
        NORM=SQRT(Y(1)*Y(1)+Y(2)*Y(2)+Y(3)*Y(3))
        IF (NORM < 1.0D-05) THEN
           !
           !           in case X and Y collinear:
           !
           IF (ABS(X(2)) > 1.0D-05) THEN
              Y(1)=0.0
              Y(2)=X(3)
              Y(3)=-X(2)
           ELSE
              Y(1)=-X(3)
              Y(2)=0.0
              Y(3)=X(1)
           ENDIF
           NORM=SQRT(Y(1)*Y(1)+Y(2)*Y(2)+Y(3)*Y(3))
        ENDIF
        Y(1)=Y(1)/NORM
        Y(2)=Y(2)/NORM
        Y(3)=Y(3)/NORM
        !
        !         now compute the cross product between X and Y
        !
        Z(1)=X(2)*Y(3)-Y(2)*X(3)
        Z(2)=X(3)*Y(1)-Y(3)*X(1)
        Z(3)=X(1)*Y(2)-Y(1)*X(2)
        !
        !         Z is already normalized.
        !
     ENDIF
  ELSE IF (MODE == 2) THEN
     !
     !       First, normalize Y
     !
     NORM=SQRT(Y(1)*Y(1)+Y(2)*Y(2)+Y(3)*Y(3))
     IF (NORM < 1.0D-05) THEN
        ERROR=.TRUE.
     ELSE
        Y(1)=Y(1)/NORM
        Y(2)=Y(2)/NORM
        Y(3)=Y(3)/NORM
        !
        !         now substract the projection of Y to Z from Z,
        !
        NORM=(Z(1)*Y(1)+Z(2)*Y(2)+Z(3)*Y(3))
        Z(1)=Z(1)-NORM*Y(1)
        Z(2)=Z(2)-NORM*Y(2)
        Z(3)=Z(3)-NORM*Y(3)
        !
        !         now normalize Z,
        !
        NORM=SQRT(Z(1)*Z(1)+Z(2)*Z(2)+Z(3)*Z(3))
        IF (NORM < 1.0D-05) THEN
           ERROR=.TRUE.
        ELSE
           Z(1)=Z(1)/NORM
           Z(2)=Z(2)/NORM
           Z(3)=Z(3)/NORM
           !
           !           now compute the cross product between Z and Y,
           !
           X(1)=Y(2)*Z(3)-Z(2)*Y(3)
           X(2)=Y(3)*Z(1)-Z(3)*Y(1)
           X(3)=Y(1)*Z(2)-Z(1)*Y(2)
        ENDIF
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE FINDBS

SUBROUTINE HPLACE(DONOR,FIXDA1,FIXDA2,FREEA1,FREEA2, &
     FREEB1,FREEB2, &
     FRFX11,FRFX12,FRFX21,FRFX22, &
     PHI1,PHI2,X,Y,Z)
  !-----------------------------------------------------------------------
  !     Routine places "free" atom FREEA1 with a dihedral PHI1,
  !     optional places  "free" atom FREEA2 with a dihedral PHI2;
  !     two fixed atoms, the center atom "DONOR" and the bondlengths
  !     FREEB* and the angles FRFX** are required (angles are in deg.).
  !     The coordinate arrays X,Y,Z being modified. An average over
  !     two ways to place the free atom is being performed.
  !
  !     DEC-82 Axel Brunger
  !
  use chm_kinds
  use intcor2,only:cartcv
  implicit none
  INTEGER   DONOR, FIXDA1, FIXDA2, FREEA1, FREEA2
  real(chm_real)    FREEB1, FREEB2
  real(chm_real)    FRFX11, FRFX12, FRFX21, FRFX22
  real(chm_real)      PHI1, PHI2
  real(chm_real) X(*),Y(*),Z(*)

  real(chm_real)      XAVE,YAVE,ZAVE
  LOGICAL OK

  CALL CARTCV(X,Y,Z,FIXDA1,FIXDA2,DONOR,FREEA1,FREEB1, &
       FRFX12,PHI1,OK)
  XAVE=X(FREEA1)
  YAVE=Y(FREEA1)
  ZAVE=Z(FREEA1)
  CALL CARTCV(X,Y,Z,FIXDA2,FIXDA1,DONOR,FREEA1,FREEB1, &
       FRFX11,-PHI1,OK)
  X(FREEA1)=0.5*(XAVE+X(FREEA1))
  Y(FREEA1)=0.5*(YAVE+Y(FREEA1))
  Z(FREEA1)=0.5*(ZAVE+Z(FREEA1))
  !
  !     now place, if necessary, place the second free atom
  !
  IF (FREEA2 > 0) THEN
     CALL CARTCV(X,Y,Z,FIXDA1,FIXDA2,DONOR,FREEA2,FREEB2, &
          FRFX22,PHI2,OK)
     XAVE=X(FREEA2)
     YAVE=Y(FREEA2)
     ZAVE=Z(FREEA2)
     CALL CARTCV(X,Y,Z,FIXDA2,FIXDA1,DONOR,FREEA2,FREEB2, &
          FRFX21,-PHI2,OK)
     X(FREEA2)=0.5*(XAVE+X(FREEA2))
     Y(FREEA2)=0.5*(YAVE+Y(FREEA2))
     Z(FREEA2)=0.5*(ZAVE+Z(FREEA2))
  ENDIF
  RETURN
END SUBROUTINE HPLACE

SUBROUTINE GTHBUI(COMLYN,COMLEN,HFLAGS,WATACC,NWATAC,X,Y,Z,WMAIN, &
     PHISTP,QPRINT,QWARN,DISTOF,ANGLON,QZERO,CUTWAT)
  !-----------------------------------------------------------------------
  !     This is the HBUILD command interpreter.
  !     This call should be accompanied by calls to GTHBCT for the HBOND
  !     lists and by GTNBCT for the NBOND lists.
  !
  !     Syntax:
  !
  !     HBUILD  [<atom-selection>] hbond-spec nbond-spec [PHIStp real]
  !     [PRINt]  [WARN] [DISTof <real>] [ANGLon <real>]
  !     [CUTWat <real>]
  !
  !     <atom-selection> gives the atoms to be (re-) constructed.
  !     If not specified all unkown hydrogens and lone pairs are
  !     constructed by default. This is equivalent to
  !     "SELEction (LONE .OR. HYDRogen) .AND..NOT. INITial END".
  !
  !     PHIStp is the stepsize in degrees for the performance of the spin
  !     algorithm in HBUIL2.
  !     If PRINt is specified information about the energies during the
  !     spin is printed.
  !
  !     CUTWater=<real>  {* cutoff for water search *}
  !
  !     DEC-83 Axel Brunger
  !
  use chm_kinds
  use dimens_fcm
  use select
  use number
  use psf
  use chutil,only:hydrog,initia,lone
  use string

  implicit none

  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  INTEGER HFLAGS(*), WATACC(*)
  INTEGER   NWATAC
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  real(chm_real)      PHISTP
  LOGICAL   QPRINT, QWARN
  real(chm_real)      DISTOF, ANGLON
  LOGICAL   QZERO
  real(chm_real)      CUTWAT

  INTEGER   I
  !
  !     the following section fills the WATACC list containing all
  !     protein acceptors. This list is used
  !     during the water hydrogen assignment procedure.
  !
  DO I=1,NATOM
     HFLAGS(I)=0
  ENDDO
  DO I=1,NACC
     HFLAGS(IACC(I))=1
  ENDDO
  DO I=1,NGRP
     IF (IGPTYP(I) == 3) HFLAGS(IGPBS(I)+1)=1
  ENDDO
  NWATAC=0
  DO I=1,NATOM
     IF (HFLAGS(I) == 1) THEN
        NWATAC=NWATAC+1
        WATACC(NWATAC)=I
     ENDIF
  ENDDO
  !
  !     call selection procedure
  CALL SELCTA(COMLYN,COMLEN,HFLAGS,X,Y,Z,WMAIN,.TRUE.)
  !
  !     make default selection
  IF(NSELCT(NATOM,HFLAGS) == NATOM) THEN
     DO I=1,NATOM
        IF ((LONE(I).OR.HYDROG(I)).AND..NOT.INITIA(I,X,Y,Z)) THEN
           HFLAGS(I)=1
        ELSE
           HFLAGS(I)=0
        ENDIF
     ENDDO
  ENDIF
  !
  !     check if zero selection specified
  QZERO=(NSELCT(NATOM,HFLAGS) == 0)

  QPRINT=(INDXA(COMLYN,COMLEN,'PRIN') > 0)
  QWARN=(INDXA(COMLYN,COMLEN,'WARN') > 0)
  DISTOF=5.2
  DISTOF=GTRMF(COMLYN,COMLEN,'DIST',DISTOF)
  ANGLON=140.D0
  ANGLON=GTRMF(COMLYN,COMLEN,'ANGL',ANGLON)
  PHISTP=GTRMF(COMLYN,COMLEN,'PHIS',TEN)
  CUTWAT=GTRMF(COMLYN,COMLEN,'CUTW',MINONE)

  RETURN
#endif 
END SUBROUTINE GTHBUI

