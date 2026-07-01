module struc
  use chm_kinds
  use dimens_fcm

#if KEY_RISM==1 /*struc_fcm*/
  use rism
!-----------------------------------------------------------------------
!       Energy parameters, coordinates variables and atom names
!-----------------------------------------------------------------------
!     X(DSITE),Y(DSITE),Z(DSITE)        Atomic coordinates
!     X2(DSITE),Y2(DSITE),Z2(DSITE)     Atomic coordinates for the
!                          changed structure (used in subroutine fluc)
!     ICHEM(DSITE)         index for the location in the array BASE 
!                          of the various atoms in the structure 
!     A(DPAIR), B(DPAIR)   van der waals coefficients
!     C(DPAIR)             Coulombic coefficient
!     INBASE(DBASE,DBASE)  Index for the  atom pairs formed from
!                          atoms in the parameter list
!     NBASE                # of atoms in the parameter list
!     NPBASE               # of atom pairs in the parameter
!                          list
!     EPSIL(DPBASE)        well depth of the vdw interaction
!     SIG(DPBASE)          sigma value of the vdw interaction
!     CHARGE(DBASE)        Coulombic charge
!     NSEGV                # of solvent segments
!     ISEGM(DSITV)         Index for the various solvent segments
!     IZMAT(4,DSITE,DU+1)  Index for the location in the array BASE
!                          of the atoms mentioned in the Z matrix
!     ZMAT(3,DSITE,DU+1)   Stores the Z-matrix specifications
!
!                 ZMATRIX(1,I,J) :  bond between atoms I-1, I
!                 ZMATRIX(2,I,J) :  angle theta between I-2,I-1,I
!                 ZMATRIX(3,I,J) :  angle phi between I-3,I-2,I-1,I
!                 J=1 for solvent, J=2 for solute 1, J=3 for solute 2
!     ATNAM(DSITE)         Atom name
!     RESNAM(DSITE)        Residue name
!     SEGMID(DSITE)        Segment name
!     BASE(DBASE)          Array that lists the various atoms
!                          found in the parameter list
!                          

      real(chm_real) X(DSITE),Y(DSITE),Z(DSITE)
      real(chm_real) X2(DSITE),Y2(DSITE),Z2(DSITE)
      INTEGER ICHEM(DSITE)
      real(chm_real)  A(DPAIR),B(DPAIR),C(DPAIR)
      INTEGER INBASE(DBASE,DBASE),NBASE
      real(chm_real) EPSIL(DPBASE),SIG(DPBASE),CHARG(DBASE)
      INTEGER NSEGV,ISEGM(DSITV)
      INTEGER IZMAT(4,DSITE,DU+1)
      real(chm_real)  ZMAT(3,DSITE,DU+1)

      CHARACTER(len=6) ATNAM(DSITE),RESNAM(DSITE),SEGMID(DSITE)
      CHARACTER(len=6) BASE(DBASE)
      CHARACTER(len=6) ASEGM(DSITV)

contains
  subroutine struc_iniall()
    use number,only:zero
    x(1:dsite)=zero
    y(1:dsite)=zero
    z(1:dsite)=zero
    x2(1:dsite)=zero
    y2(1:dsite)=zero
    z2(1:dsite)=zero
    atnam(1:dsite)=' '
    resnam(1:dsite)=' '
    segmid(1:dsite)=' '
    ichem(1:dsite)=0

    base(1:dbase)=' '
    charg(1:dbase)=zero

    asegm(1:dsitv)=' '

    epsil(1:dpbase)=zero
    sig(1:dpbase)=zero

    a(1:dpair)=zero
    b(1:dpair)=zero
    c(1:dpair)=zero
    return
  end subroutine struc_iniall

#endif /* (struc_fcm)*/
!
end module struc

