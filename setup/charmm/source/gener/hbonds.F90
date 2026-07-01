SUBROUTINE HBONDS(UNIT,IDON,IHD1,NDON,IACC,IAC1,NACC, &
     IHB,JHB,KHB,LHB,ICH,NHB,MAXHBL,CHBA,CHBB,HBEXPN, &
     CTONHB,CTOFHB,CTONHA,CTOFHA,CUTHB,CUTHBA,LHBFG, &
     NCH,KCH,IAC,ATC,NATC, &
     IMOVE,BEST,HBEXCL,IBLO,INB,NNB,X,Y,Z,NATOM)
  !
  !     by Axel Brunger, NOV-82.
  !
  !     HBONDS generates a hydrogen bond listing and edits the listing.
  !
  use chm_kinds
  use exfunc
  use memory
  use stream
  implicit none
  !
  real(chm_real),allocatable,dimension(:) :: ENERGY
  INTEGER   UNIT,NDON,NACC,NHB,MAXHBL,NATC,NNB,NATOM
  INTEGER   IDON(*),IHD1(*),IACC(*),IAC1(*)
  INTEGER   IHB(*),JHB(*),KHB(*),LHB(*),ICH(*)
  real(chm_real)    CHBA(*), CHBB(*)
  INTEGER   HBEXPN(*)
  real(chm_real)    CTONHB, CTOFHB, CTONHA, CTOFHA, CUTHB, CUTHBA
  LOGICAL   LHBFG
  INTEGER   KCH(*), NCH
  INTEGER   IAC(*)
  CHARACTER(len=4) ATC(*)
  INTEGER   IMOVE(*)
  LOGICAL   BEST,HBEXCL
  INTEGER   IBLO(*),INB(*)
  real(chm_real)    X(*),Y(*),Z(*)
  !
  !
  LOGICAL QPRINT
  !
  IF(CUTHB.LT.1.01) THEN
     ! No hydrogen bond list because cutoff is too small.
     NHB=0
     RETURN
  ENDIF
  !
  QPRINT=(PRNLEV.GE.5)
  !
  !     make a list of hydrogen bonds within the cutoff's,
  !
  CALL HBFIND(UNIT,QPRINT,IDON,IHD1,1,NDON,IACC,IAC1,1,NACC, &
       X,Y,Z,IHB,JHB,KHB,LHB,1, &
       NHB,MAXHBL,LHBFG,CUTHB,CUTHBA)
  !
  !     now get the parameters for the hydrogen bonds,
  !
  CALL HCODES(ICH,NHB,IHB,JHB,IAC)
  !
  !     now edit the hydrogen bond list (sorting, delete 'fixed' bonds,
  !     if required get 'best' hydrogen bond
  !     for each hydrogen, compress list).
  !
  !     Remark: HBEDIT compresses (IHB, JHB, KHB, LHB, ICH)
  !     simultaneously.
  !
  call chmalloc('hbonds.src','HBONDS','ENERGY',NHB,crl=ENERGY)
  CALL HBEDIT(UNIT,X,Y,Z,IMOVE,NATOM,IHB,JHB,KHB,LHB,NHB,MAXHBL, &
       ICH,CHBA,CHBB,HBEXPN,BEST,HBEXCL,IBLO,INB,NNB, &
       CTONHB,CTOFHB,CTONHA,CTOFHA,ENERGY)
  !
  call chmdealloc('hbonds.src','HBONDS','ENERGY',NHB,crl=ENERGY)
  RETURN
END SUBROUTINE HBONDS

SUBROUTINE PRNHBD(IUNIT)
  !
  !  Print the Hbond options and cutoff values.
  !
  use chm_kinds
  use dimens_fcm
  !
  use stream
  use hbondm
  implicit none
  !
  INTEGER IUNIT
  !
  IF(PRNLEV.GE.2) THEN
     WRITE(IUNIT,1000) CUTHB,CUTHBA
1000 FORMAT(' PRNHBD: CUToff Hydrogen Bond  distance =',F10.4, &
          '   Angle =',F10.4)
     WRITE(IUNIT,2000) CTONHB,CTOFHB,CTONHA,CTOFHA
2000 FORMAT('         CuT switching ON HB dist. = ',F10.4, &
          '  OFf HB dist. =',F10.4,/ &
          '         CuT switching ON Hb Angle = ',F10.4, &
          '  OFf Hb Angle =',F10.4)
     IF(LHBFG) THEN
        WRITE(IUNIT,3000)
     ELSE
        WRITE(IUNIT,3500)
     ENDIF
3000 FORMAT('         ACCEptor antecedents included')
3500 FORMAT('         NO ACceptor antecedents included')
     IF(BEST) THEN
        WRITE(IUNIT,4000)
     ELSE
        WRITE(IUNIT,4500)
     ENDIF
4000 FORMAT( &
          '         Best hydrogen bond for each hydrogen will be found')
4500 FORMAT( &
          '         All hydrogen bonds for each hydrogen will be found')
     IF(HBEXCL) THEN
        WRITE(IUNIT,5000)
     ELSE
        WRITE(IUNIT,5500)
     ENDIF
5000 FORMAT( &
          '         Hydrogen bonds between excluded atoms will be deleted'/)
5500 FORMAT( &
          '         Hydrogen bonds between excluded atoms will be kept'/)
  ENDIF
  RETURN
END SUBROUTINE PRNHBD

SUBROUTINE HBFIND(UNIT,QPRINT,IDON,IHD1,MDON,NDON,IACC,IAC1, &
     MACC,NACC,X,Y,Z,IHB,JHB,KHB,LHB,MHB, &
     NHB,MAXHBL,LHBFG,CUTHB,CUTHBA)
  !
  !
  !     Selects all hydrogen bonds within a donor-acceptor-distance
  !     cutoff CUTHB and a donor-hydrogen-acceptor-angle cutoff CUTHBA.
  !       Flags:
  !              LHBFG =.TRUE. generate list with acceptor antecedents,
  !              QPRINT=.TRUE. print message information on UNIT,
  !
  !
  !       Indices:
  !                 MDON  is the first index of DONOrs,
  !                 NDON   -"-   last   -"-      -"-
  !
  !                 MACC   -"-   first  -"-     ACCEptors,
  !                 NACC   -"-   last   -"-      -"-
  !
  !                 MHB    -"-   first  -"-  of the hbond list,
  !       (on exit) NHB    -"-   last   -"-      -"-          .
  !
  !     Axel Brunger, NOV-82
  !
  !
  use chm_kinds
  use stream
  use chutil,only:atomid,initia

  implicit none
  INTEGER   UNIT
  LOGICAL   QPRINT
  INTEGER   IDON(*), IHD1(*), IACC(*), IAC1(*)
  INTEGER   MDON, NDON, MACC, NACC
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER   IHB(*), JHB(*), KHB(*), LHB(*)
  INTEGER   MHB, NHB, MAXHBL
  LOGICAL   LHBFG
  real(chm_real)    CUTHB, CUTHBA
  !
  !
  INTEGER   I, J, DONOR, HYDRO, ACCE, ANTE
  INTEGER   DISTEX, ANGLEX
  real(chm_real)    X1D, Y1D, Z1D, X2D, Y2D, Z2D
  real(chm_real)    DIST2, CUTHB2, CSANGL, CSCHBA
  CHARACTER(len=8) SG1, SG2, RS1, RS2, RN1, RN2, AT1, AT2
  real(chm_real), parameter :: RAD = 0.174532925D-01
  !
  CUTHB2=CUTHB*CUTHB
  CSCHBA=-COS(CUTHBA*RAD)
  !lb..01-AUG-98
  CSANGL=0.0
  !lb..
  DISTEX=0
  ANGLEX=0
  NHB=MHB-1
  I=MDON

1000 FORMAT(' ERROR FROM HBFIND:',/, &
       ' Coordinates not initiated for atom',4(1X,A))
1010 FORMAT(' ERROR FROM HBFIND: ',/, &
       ' Zero distance between atoms',4(1X,A),' and',4(1X,A))

  DO WHILE (NHB.LE.MAXHBL .AND. I.LE.NDON)
     DONOR=IDON(I)
     HYDRO=IHD1(I)
     IF (HYDRO.GT.0) THEN
        IF (.NOT. INITIA(HYDRO,X,Y,Z)) THEN
           CALL ATOMID(HYDRO,SG1,RS1,RN1,AT1)
           IF(WRNLEV.GE.2) WRITE(UNIT,1000) SG1(1:idleng), &
                RS1(1:idleng),RN1(1:idleng),AT1(1:idleng)
           CALL DIEWRN(0)
        ENDIF
     ENDIF
     DO J=MACC,NACC
        ACCE=IACC(J)
        IF (LHBFG) ANTE=IAC1(J)
        IF (DONOR .NE. ACCE) THEN
           !              evaluate-dist2-between-donor-and-acceptor
           X1D=X(DONOR)-X(ACCE)
           Y1D=Y(DONOR)-Y(ACCE)
           Z1D=Z(DONOR)-Z(ACCE)
           DIST2=(X1D*X1D+Y1D*Y1D+Z1D*Z1D)
           !
           IF(DIST2 .LE. 1.0E-05) THEN
              CALL ATOMID(DONOR,SG1,RS1,RN1,AT1)
              CALL ATOMID(ACCE,SG2,RS2,RN2,AT2)
              IF(WRNLEV.GE.2) WRITE(UNIT,1010)  &
                   SG1(1:idleng),RS1(1:idleng),RN1(1:idleng), &
                   AT1(1:idleng), &
                   SG2(1:idleng),RS2(1:idleng),RN2(1:idleng), &
                   AT2(1:idleng)
              CALL DIEWRN(-4)
           ELSE
              IF(DIST2 .GE. CUTHB2) THEN
                 DISTEX=DISTEX+1
              ELSE
                 IF (HYDRO.GT.0) THEN
                    !                       get-cos-angle-dono-hydro-acce
                    X1D=X(ACCE)-X(HYDRO)
                    Y1D=Y(ACCE)-Y(HYDRO)
                    Z1D=Z(ACCE)-Z(HYDRO)
                    X2D=X(DONOR)-X(HYDRO)
                    Y2D=Y(DONOR)-Y(HYDRO)
                    Z2D=Z(DONOR)-Z(HYDRO)
                    CSANGL=(X1D*X2D+Y1D*Y2D+Z1D*Z2D)/SQRT( &
                         (X1D*X1D+Y1D*Y1D+Z1D*Z1D)* &
                         (X2D*X2D+Y2D*Y2D+Z2D*Z2D))
                 ENDIF
                 IF(HYDRO.GT.0 .AND. CSANGL.GE.CSCHBA) THEN
                    ANGLEX=ANGLEX+1
                 ELSE
                    NHB=NHB+1
                    IF(NHB.GT.MAXHBL) THEN
                       CALL WRNDIE(-4,'<HBFIND>', &
                            'Maximum number of hydrogen bonds MAXHB exceeded')
                    ELSE
                       IHB(NHB)=DONOR
                       KHB(NHB)=HYDRO
                       JHB(NHB)=ACCE
                       IF(LHBFG) THEN
                          LHB(NHB)=ANTE
                       ELSE
                          LHB(NHB)=0
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
     ENDDO
     I=I+1
  ENDDO
  IF (QPRINT) WRITE(UNIT,1030)  DISTEX, ANGLEX
1030 FORMAT(' HBFIND-exclusions:',I7,' due to distance cutoff, ',I7, &
       ' due to angle cutoff')
  RETURN
  !
END SUBROUTINE HBFIND

SUBROUTINE HBEDIT(UNIT,X,Y,Z,IMOVE,NATOM,IHB,JHB,KHB,LHB, &
     NHB,MAXHBL,ICH,CHBA,CHBB,HBEXPN,BEST,HBEXCL,IBLO,INB,NNB, &
     CTONHB,CTOFHB,CTONHA,CTOFHA,ENERGY)
  !
  !
  !   1.  Sorts hbond list stored in (IHB,KHB,JHB,LHB,ICH, NHB)
  !       first priority donor, second priority hydrogen,...
  !
  !   1a. If HBEXCL is set, all hydrogen bonds containing pairs of atoms
  !       on the excluded list (or as 1-4 interactions) are removed.
  !
  !   2.  If the BEST option is true for each hydrogen
  !       the lowest energy hydrogen bond is taken, the remaining are
  !       excluded.
  !
  !   3.  Hbonds with all partizipating atoms fixed (IMOVE) are excluded.
  !
  !   4.  A search is made for double hydrogen bonds, i.e. two hbonds
  !       ATOM1-hydrogen1...ATOM2, ATOM1...hydrogen2-ATOM2; a message
  !       is printed.
  !
  !     The filtering process is performed in the above order.
  !     Finally, the list is compressed.
  !
  !     Axel Brunger, NOV-82
  !
  use chm_kinds
  use exfunc
  use stream
  use chutil,only:atomid
  implicit none
  EXTERNAL EXCH5
  INTEGER   UNIT
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER   IMOVE(*)
  INTEGER   NATOM
  INTEGER   IHB(*), JHB(*), KHB(*), LHB(*)
  INTEGER   NHB, MAXHBL
  LOGICAL   BEST,HBEXCL
  INTEGER   IBLO(*),INB(*)
  INTEGER   NNB
  INTEGER   ICH(*)
  INTEGER   HBEXPN(*)
  real(chm_real)    CHBA(*), CHBB(*), CTONHB, CTOFHB, CTONHA, CTOFHA
  real(chm_real)    ENERGY(*)
  !
  INTEGER   I,J,BESTI,II,INBX,IS,IQ
  INTEGER   FIXDEX, BESTEX, DUPLEX, DELETI,HBEXEX
  INTEGER   DONOR, HYDRO, ACCE, ANTE
  LOGICAL   FIXDHB, DUPLCT, DONE
  real(chm_real)    LOWERG
  real(chm_real)    TOTERG
  CHARACTER (len=8) SIDDN,RIDDN,RESDN,ACDN,SIDDN2,RIDDN2,RESDN2,ACDN2
  !
  !     temporary mark used for hbonds to be deleted
  INTEGER, parameter :: MARK = -99999999
  !
  !     sort-hbond-list
  CALL SORT(NHB,EXCH5,ORDER5,IHB,KHB,JHB,LHB,ICH,0,0,5)
  !
  DUPLEX=0
  FIXDEX=0
  BESTEX=0
  HBEXEX=0
  !
  ! Duplications in the hydrogen bond list are deleted.
  !
  DONOR=IHB(1)
  HYDRO=KHB(1)
  ACCE=JHB(1)
  ANTE=LHB(1)
  DO I=2,NHB
     DUPLCT=((DONOR.EQ.IHB(I)).AND.(HYDRO.EQ.KHB(I)).AND. &
          (ACCE.EQ.JHB(I)).AND.(ANTE.EQ.LHB(I)))
     IF(DUPLCT) THEN
        DELETI=I
        DONE=(IHB(DELETI).NE.MARK)
        IHB(DELETI)=MARK
        IF (DONE) DUPLEX=DUPLEX+1
     ENDIF
     DONOR=IHB(I)
     HYDRO=KHB(I)
     ACCE=JHB(I)
     ANTE=LHB(I)
  ENDDO
  !
  !     If HBEXCL is set, check each hbond and remove those that have an
  !     exclusion between the acceptor and donor atoms.
  !
  IF(HBEXCL) THEN
     IF(NNB.LE.0) THEN
        CALL WRNDIE(1,'<HBEDIT>', &
             'NO EXCLUSIONS YET FOR HBEXCL OPTION')
     ELSE
        DO I=1,NHB
           !              since the exclusion list has on occasion been inverted,
           !              we check both directions.
           IF(IHB(I).EQ.1) THEN
              IS=1
              !..B980623.rn
           ELSEIF (IHB(I) .EQ. MARK) THEN
              GOTO 3333
              !..
           ELSE
              IS=IBLO(IHB(I)-1)+1
           ENDIF
           IQ=IBLO(IHB(I))
           DO J=IS,IQ
              INBX=INB(J)
              IF(INBX.LT.0) INBX=-INBX
              IF(INBX.EQ.JHB(I)) THEN
                 DELETI=I
                 DONE=(IHB(DELETI).NE.MARK)
                 IHB(DELETI)=MARK
                 IF(DONE) HBEXEX=HBEXEX+1
              ENDIF
           ENDDO
           IF(JHB(I).EQ.1) THEN
              IS=1
           ELSE
              IS=IBLO(JHB(I)-1)+1
           ENDIF
           IQ=IBLO(JHB(I))
           DO J=IS,IQ
              INBX=INB(J)
              IF(INBX.LT.0) INBX=-INBX
              IF(INBX.EQ.IHB(I)) THEN
                 DELETI=I
                 DONE=(IHB(DELETI).NE.MARK)
                 IHB(DELETI)=MARK
                 IF(DONE) HBEXEX=HBEXEX+1
              ENDIF
           ENDDO
3333       CONTINUE
        ENDDO
     ENDIF
  ENDIF
  !
  !
  !     If BEST specified, determine hydrogen bond with lowest energy
  !     for each hydrogen and delete others.
  !
  IF (BEST) THEN
     !        evaluate-energy-for-hbond-list
     CALL EHBOND(TOTERG,IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
          0,0,0,X,Y,Z,.FALSE.,0,3,ENERGY,0,0, &
          CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
          0,0,.FALSE.)
     !
     I=1
     DO WHILE (I.LE.NHB)
        DONOR=IHB(I)
        IF(DONOR.GT.0) THEN
           HYDRO=KHB(I)
           BESTI=I
           LOWERG=ENERGY(I)
           I=I+1
           DO WHILE (IHB(I).EQ.DONOR .AND. KHB(I).EQ.HYDRO .AND. &
                I.LE.NHB)
              IF (IHB(I).GT.0) THEN
                 IF(ENERGY(I) .GT. LOWERG) THEN
                    DELETI=I
                    DONE=(IHB(DELETI).NE.MARK)
                    IHB(DELETI)=MARK
                    IF (DONE) BESTEX=BESTEX+1
                 ELSE
                    DELETI=BESTI
                    DONE=(IHB(DELETI).NE.MARK)
                    IHB(DELETI)=MARK
                    IF (DONE) BESTEX=BESTEX+1
                    BESTI=I
                    LOWERG=ENERGY(I)
                 ENDIF
              ENDIF
              I=I+1
           ENDDO
        ELSE
           I=I+1
        ENDIF
     ENDDO
  ENDIF
  !
  !     If all participating atoms are fixed the hbond is deleted.
  !
  DO I=1,NHB
     IF (IHB(I).GT.0) THEN
        DONOR=IHB(I)
        HYDRO=KHB(I)
        ACCE=JHB(I)
        ANTE=LHB(I)
        FIXDHB=(IMOVE(DONOR).GT.0).AND.(IMOVE(ACCE).GT.0)
        IF (HYDRO.NE.0) FIXDHB=FIXDHB.AND.(IMOVE(HYDRO).GT.0)
        IF (ANTE.NE.0) FIXDHB=FIXDHB.AND.(IMOVE(ANTE).GT.0)
        IF (FIXDHB) THEN
           DELETI=I
           DONE=(IHB(DELETI).NE.MARK)
           IHB(DELETI)=MARK
           IF (DONE) FIXDEX=FIXDEX+1
        ENDIF
     ENDIF
  ENDDO
  !
  !     compress-list
  II=0
  DO I=1,NHB
     IF (IHB(I).NE.MARK) THEN
        II=II+1
        IHB(II)=IHB(I)
        JHB(II)=JHB(I)
        KHB(II)=KHB(I)
        LHB(II)=LHB(I)
        ICH(II)=ICH(I)
     ENDIF
  ENDDO
  NHB=II
  !
  IF(PRNLEV.GE.5) WRITE(UNIT,1000) DUPLEX,BESTEX,FIXDEX,HBEXEX
1000 FORMAT(' HBEDIT-deletions: ',I7,' due to duplications,    ',I7, &
       ' due to best-option,',/, &
       '                   ',I7,' due to fixed atoms,     ',I7, &
       ' due to exclusions')
  IF (WRNLEV.GE.5) THEN
     !     determine-double-hydrogen-bonds
     DO I=1,NHB
        DONOR=IHB(I)
        IF (DONOR.GT.0) THEN
           ACCE=JHB(I)
           DO J=I+1,NHB
              IF (DONOR.EQ.JHB(J) .AND. ACCE.EQ.IHB(J)) THEN
                 CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,ACDN)
                 CALL ATOMID(ACCE,SIDDN2,RIDDN2,RESDN2,ACDN2)
                 IF(PRNLEV.GE.5) THEN
                    WRITE(UNIT,1100) &
                         SIDDN(1:idleng),RIDDN(1:idleng), &
                         RESDN(1:idleng),ACDN(1:idleng), &
                         SIDDN2(1:idleng),RIDDN2(1:idleng), &
                         RESDN2(1:idleng),ACDN2(1:idleng)
                 ENDIF
              ENDIF
           ENDDO
        ENDIF
     ENDDO
  ENDIF
1100 FORMAT(' Info: double HBOND between atoms ', &
       4(1X,A),' and ',4(1X,A))
  !
  IF(PRNLEV.GE.5) WRITE(UNIT,1110) NHB
1110 FORMAT(' HBEDIT: currently ',I5,' hydrogen bonds present')
  !
  RETURN
END SUBROUTINE HBEDIT

SUBROUTINE HBREAD(IUNIT,NATOM,TITLE,NTITL,ICNTRL, &
     NHB,MAXHBL,IHB,JHB,KHB,LHB,LCARD, &
     SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
  !
  !     Reads in a hydrogen bond list;
  !     in case LCARD=.TRUE. the file on IUNIT should be formatted,
  !     in case LCARD=.FALSE. a binary version is assumed.
  !
  !     NOV-82 AB
  !
  use chm_kinds
  use exfunc
  use stream
  use string
  use chutil,only:getatn
  use parallel,only:psnd4

  implicit none
  INTEGER     IUNIT, NATOM
  CHARACTER(len=*)  TITLE(*)
  INTEGER     NTITL, ICNTRL(20), NHB, MAXHBL
  INTEGER     IHB(*), JHB(*), KHB(*), LHB(*)
  LOGICAL     LCARD
  CHARACTER(len=*)     SEGID(*), RESID(*),ATYPE(*)
  INTEGER     IBASE(*)
  INTEGER     NICTOT(*), NSEG
  !
  INTEGER     I
  LOGICAL     OK
  !
  character(len=80) fline
  integer      ilen
  logical      lextfmt

  CHARACTER(len=8) AC,DN,KN,LN,RIDAC,RIDDN,SIDAC,SIDDN,BLANKS
  CHARACTER(len=4) RHDR,HBHDR
  DATA HBHDR,BLANKS/'HBON','        '/
  !
  IF(IOLEV.GT.0) THEN
     !
     IF(.NOT. LCARD) THEN
        !
        ! read hbonds from binary file
        !
        CALL TRYORO(IUNIT,'UNFORMATTED')
        !
        IF (reallow) THEN    
           REWIND (UNIT=IUNIT)
        ENDIF                
        READ(IUNIT) RHDR,ICNTRL
        IF(HBHDR.NE.RHDR) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,201) HBHDR,RHDR
201        FORMAT(' EXPECTED = "',A4,'" FOUND = "',A4,'"')
           CALL WRNDIE(-1,'<HBREAD>','HEADERS DONT MATCH')
        ENDIF
        CALL RDTITL(TITLE,NTITL,IUNIT,-1)
        IF(PRNLEV.GE.2) CALL WRTITL(TITLE,NTITL,OUTU,1)
        !
        READ(IUNIT) NHB
        IF(NHB.GT.MAXHBL) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,1020) MAXHBL
1020       FORMAT(' ERROR FROM HBREAD:'/' The hydrogen bond limit ', &
                'of ',I5,' has been exceeded. No list was read.')
           CALL DIEWRN(0)
        ELSE
           READ(IUNIT) (IHB(I),JHB(I),KHB(I),I=1,NHB)
           IF(ICNTRL(1).GT.16) THEN
              READ(IUNIT) (LHB(I),I=1,NHB)
           ELSE
              DO I=1,NHB
                 LHB(I)=0
              ENDDO
           ENDIF
        ENDIF
        DO I=1,NHB
           IF (IHB(I).GT.0) THEN
              OK=IHB(I).LE.NATOM .AND. &
                   JHB(I).GE.1 .AND. JHB(I).LE.NATOM .AND. &
                   KHB(I).GE.0 .AND. KHB(I).LE.NATOM .AND. &
                   LHB(I).GE.0 .AND. LHB(I).LE.NATOM
              IF (.NOT.OK) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,1030) I
1030             FORMAT(' **** ERROR FROM HBREAD:'/ &
                      ' Hydrogen bond ',I5, &
                      ' which was just read'/ &
                      ' refers to either negative atom', &
                      ' numbers or atoms larger than NATOM.')
                 CALL DIEWRN(-4)
              ENDIF
           ENDIF
        ENDDO
        !
        !
     ELSE
        CALL TRYORO(IUNIT,'FORMATTED')
        !
        ! Read hbonds from cards.
        !
        CALL RDTITL(TITLE,NTITL,IUNIT,0)
        !
        READ(IUNIT,'(a)') fline
        ilen=len(fline)
        call cnvtuc(fline,ilen)
        NHB=nexti(fline,ilen)
        lextfmt=(indxa(fline,ilen,'EXT').gt.0)
        if (lextfmt) then
           ilen=8
           fline='(8(2X,A8))'
        else
           ilen=4
           fline='(8(2X,A4))'
        endif
        IF(NHB.GT.MAXHBL) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,1020) MAXHBL
           CALL DIEWRN(0)
        ELSE
           DO I=1,NHB
              READ(IUNIT,fline) SIDDN,RIDDN,DN,KN,AC,LN,RIDAC,SIDAC
              !
              ! try to identify all atoms within the context of the current PSF
              !
              IHB(I)=GETATN(SIDDN,RIDDN,DN,SEGID,RESID,ATYPE,IBASE, &
                   NICTOT,NSEG)
              JHB(I)=GETATN(SIDAC,RIDAC,AC,SEGID,RESID,ATYPE,IBASE, &
                   NICTOT,NSEG)
              !
              ! a skipping of hydrogens is allowed, e.g. reading a explicit
              ! hydrogen bond listing into a "similar" PSF without hydrogens:
              !
              IF(KN .NE. BLANKS) THEN
                 KHB(I)=GETATN(SIDDN,RIDDN,KN,SEGID,RESID,ATYPE, &
                      IBASE,NICTOT,NSEG)
                 IF (KHB(I).EQ.-1) KHB(I)=0
              ELSE
                 KHB(I)=0
              ENDIF
              IF(LN .NE. BLANKS) THEN
                 LHB(I)=GETATN(SIDAC,RIDAC,LN,SEGID,RESID,ATYPE, &
                      IBASE,NICTOT,NSEG)
              ELSE
                 LHB(I)=0
              ENDIF
              !
              ! now check, if all atoms are identified
              !
              IF((IHB(I).EQ.-1).OR.(JHB(I).EQ.-1).OR. &
                   (LHB(I).EQ.-1)) THEN
                 IF(WRNLEV.GE.2) WRITE(OUTU,1700) &
                      SIDDN(1:ilen),RIDDN(1:ilen), &
                      DN(1:ilen),KN(1:ilen), &
                      AC(1:ilen),LN(1:ilen), &
                      SIDAC(1:ilen),RIDAC(1:ilen)
                 CALL DIEWRN(-4)
              ENDIF
           ENDDO
        ENDIF
     ENDIF
1700 FORMAT(' HBONDS:: In PSF no atoms for HBOND ',/,8(2X,A))
     IF(PRNLEV.GE.2) WRITE(OUTU,1010) NHB
1010 FORMAT(/10X,I5,' hydrogen bonds were read.')
     !
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(NHB,1)
  CALL PSND4(IHB,NHB)
  CALL PSND4(JHB,NHB)
  CALL PSND4(KHB,NHB)
  CALL PSND4(LHB,NHB)
#endif 
  !
  RETURN
END SUBROUTINE HBREAD

SUBROUTINE HBWRIT(IUNIT,LPRINT,LCARD,TITLE,NTITL,ICNTRL,IHB, &
     JHB,KHB,LHB,ICH,NHB,CUTHB,CUTHBA,CTONHB, &
     CTOFHB,CTONHA,CTOFHA,CHBA,CHBB, &
     HBEXPN,X,Y,Z)
  !
  !     Writes out a hydrogen bond list;
  !     in case LCARD and .NOT. LPRINT the list is written to
  !             formatted IUNIT,
  !     in case .NOT. LCARD and .NOT. LPRINT the list is written to
  !             binary IUNIT,
  !     in case  LPRINT a list including distances, energies, etc. is
  !             written to formatted IUNIT.
  !
  !     Remark: the information about the current PSF in the ATOMID call
  !             is passed via the /PSF/ common block!
  !
  !     NOV-82 AB
  !
  use chm_kinds
  use stream
  use chutil,only:atomid
  implicit none
  INTEGER   IUNIT
  LOGICAL   LPRINT, LCARD
  CHARACTER(len=*) TITLE(*)
  INTEGER   NTITL, ICNTRL(20)
  INTEGER   IHB(*), JHB(*), KHB(*), LHB(*), ICH(*)
  INTEGER   NHB
  real(chm_real)    CUTHB, CUTHBA, CTONHB, CTOFHB, CTONHA, CTOFHA
  real(chm_real)    CHBA(*), CHBB(*)
  INTEGER   HBEXPN(*)
  real(chm_real) X(*),Y(*),Z(*)
  !
  real(chm_real)      DIST, ANGLE, ENERGY, ANTEAN
  real(chm_real)      EHB
  INTEGER     I, NSLCT
  INTEGER     ANTE, DONOR, HYDRO, ACCE
  !
  CHARACTER(len=8) DN, LN, KN, AC, RESDN, RIDDN, SIDDN
  CHARACTER(len=8) RESAC, RIDAC, SIDAC, BLANKS
  CHARACTER(len=4) HDR
  DATA        HDR/'HBON'/,BLANKS/'        '/
  !
  IF(OUTU.EQ.IUNIT .AND. PRNLEV.LT.2) RETURN
  IF(OUTU.NE.IUNIT .AND. IOLEV.LT.0) RETURN
  !
  !     nslct describes the number of "active" hydrogen bonds:
  !
  NSLCT=0
  DO I=1,NHB
     IF (IHB(I).GT.0) NSLCT=NSLCT+1
  ENDDO
  !
  IF(.NOT. LPRINT .AND. .NOT. LCARD) THEN
     !
     ! a binary file on unit IUNIT is written
     !
     ICNTRL(1)=18
     DO I=2,20
        ICNTRL(I)=0
     ENDDO
     !
     WRITE(IUNIT) HDR,ICNTRL
     CALL WRTITL(TITLE,NTITL,IUNIT,-1)
     WRITE(IUNIT) NHB
     IF (NHB.GT.0) THEN
        WRITE(IUNIT) (IHB(I),JHB(I),KHB(I),I=1,NHB)
        WRITE(IUNIT) (LHB(I),I=1,NHB)
     ENDIF
     IF(WRNLEV.GE.2) WRITE(OUTU,1000) NHB,IUNIT,HDR,ICNTRL
1000 FORMAT(/,2X,I6,' H-BOND PAIRS WRITTEN ON UNIT',I4, &
          ' WITH A HEADER ',A4,/,5X,20I2,' AS ICNTRL ARRAY',/)
     !
     !
  ELSE IF(LPRINT) THEN
     !
     ! a human comprehensable information is printed on IUNIT
     !
     IF(LCARD) THEN
        IF(NTITL.GT.0) CALL WRTITL(TITLE,NTITL,IUNIT,0)
        WRITE(IUNIT,1009) NSLCT,CUTHB,CUTHBA
1009    FORMAT(I5,2F10.4)
     ELSE
        WRITE(IUNIT,1010) NSLCT,CUTHB,CUTHBA
1010    FORMAT(/,5X,'LIST OF',I8,4X,'H-BOND PAIRS',/, &
             5X,'SELECTED WITH CUTHB =',F6.2,', CUTHBA =',F6.2/)
        WRITE(IUNIT,1020)
1020    FORMAT(/,17X,'HBOND ',26X,'ENERGY',1X,'DISTANCE',1X, &
             'A-H-D',2X,'H-A-AA')
     ENDIF
     !
     DO I=1,NHB
        IF (IHB(I).GT.0) THEN
           DONOR=IHB(I)
           ACCE=JHB(I)
           HYDRO=KHB(I)
           ANTE=LHB(I)
           CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,DN)
           CALL ATOMID(ACCE,SIDAC,RIDAC,RESAC,AC)
           IF(HYDRO.GT.0) THEN
              CALL ATOMID(HYDRO,SIDDN,RIDDN,RESDN,KN)
           ELSE
              KN=BLANKS
           ENDIF
           IF(ANTE.GT.0) THEN
              CALL ATOMID(ANTE,SIDAC,RIDAC,RESAC,LN)
           ELSE
              LN=BLANKS
           ENDIF
           !
           ! now get the data energy, A-D distance, A-H-D angle, H-A-AA angle
           !
           CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,3,ENERGY,0,0, &
                CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
                0,0,.FALSE.)
           CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,1,DIST,0,0, &
                CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
                0,0,.FALSE.)
           IF (KHB(I).GT.0) THEN
              CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                   CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,2, &
                   ANGLE,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA, &
                   HBEXPN,0,0,.FALSE.)
              CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                   CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,4, &
                   ANTEAN,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA, &
                   HBEXPN,0,0,.FALSE.)
           ENDIF
           !
           !     finally write everthing on IUNIT
           !
           IF(KHB(I).GT.0) THEN
              WRITE(IUNIT,1030) I,SIDDN(1:idleng),RIDDN(1:idleng), &
                   DN(1:idleng),KN(1:idleng), &
                   AC(1:idleng),LN(1:idleng),RIDAC(1:idleng), &
                   SIDAC(1:idleng),ENERGY,DIST,ANGLE,ANTEAN
           ELSE
              WRITE(IUNIT,1030) I,SIDDN(1:idleng),RIDDN(1:idleng), &
                   DN(1:idleng),KN(1:idleng), &
                   AC(1:idleng),LN(1:idleng),RIDAC(1:idleng), &
                   SIDAC(1:idleng),ENERGY,DIST
           ENDIF
        ENDIF
     ENDDO
1030 FORMAT(1X,I4,'(',A,' ',A,' ',A,'-',A,')(', &
          A,'-',A,' ',A,' ',A,')',4(F7.3,1X))
     !
     !
  ELSE IF(.NOT. LPRINT .AND. LCARD) THEN
     !
     ! a card file is written on IUNIT (can be edited!)
     !
     IF(NTITL.GT.0) CALL WRTITL(TITLE,NTITL,IUNIT,0)
     AC=blanks
     if (qextfmt) AC='  EXT   '
     WRITE(IUNIT,1050) NSLCT,AC
1050 FORMAT(I5,A8)
     DO I=1,NHB
        IF (IHB(I).GT.0) THEN
           !
           !     get atom id`s (SEGID, RESID, RESIDUE, NAME)
           !
           DONOR=IHB(I)
           ACCE=JHB(I)
           HYDRO=KHB(I)
           ANTE=LHB(I)
           CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,DN)
           CALL ATOMID(ACCE,SIDAC,RIDAC,RESAC,AC)
           IF(HYDRO.GT.0) THEN
              CALL ATOMID(HYDRO,SIDDN,RIDDN,RESDN,KN)
           ELSE
              KN=BLANKS
           ENDIF
           IF(ANTE.GT.0) THEN
              CALL ATOMID(ANTE,SIDAC,RIDAC,RESAC,LN)
           ELSE
              LN=BLANKS
           ENDIF
           !
           !     write out everything
           !
           WRITE(IUNIT,1060) SIDDN(1:idleng),RIDDN(1:idleng), &
                DN(1:idleng),KN(1:idleng),AC(1:idleng), &
                LN(1:idleng),RIDAC(1:idleng),SIDAC(1:idleng)
1060       FORMAT(8(2X,A))
        ENDIF
     ENDDO
     !
  ENDIF
  RETURN
END SUBROUTINE HBWRIT

SUBROUTINE GTHBCT(COMLYN,COMLEN)
  !
  !
  !     This is the HBOND clause interpreter. The syntax is
  !
  !     HBONd      hbond-spec
  !
  ! hbond-spec ::= [BEST] [DUMMy] [CUTHB real] [CUTHA real] [ACCE] [INIT]
  !                [ALL ]                                 [NOAC]
  !
  !      [HBEX]  [CTONHB real] [CTOFHB real] [CTONHA real] [CTOFHA real]
  !      [HBNOex]
  !
  !     INIT sets the cutoff's and switching function numbers to initial
  !     values: (CUTHB=4.5, CTONHB=3.5, CTOFHB=4.0, CUTHBA=90.0,
  !              CTONHA=50.0, CTOFHA=70.0).
  !     After the initialisation of CHARMM a call to this routine with
  !     INIT is automatically performed.
  !     If the cutoff's are not specified in a call to this routine, they
  !     default to old values. If the switching function values are not
  !     specified, old values are taken only if the corresponding cutoff
  !     has not been changed. If the corresponding cutoff has been changed
  !     the switching functions will be derived from the cutoff.
  !
  !     The keyword ACCE (default) includes acceptor antecedents.
  !     The keyword ALL allows several hydrogen bonds to each hydrogen.
  !     BEST takes the hydrogen bond with the lowest energy using EHBOND.
  !     The HBEX keyword supresses ony acceptor-donor pair that have a
  !     nonbond exclusion (or a 1-4 interaction if NBXMd=+-5)
  !
  !     If DUMMY is specified CUTHB and CUTHBA are set to zero selecting
  !     no hydrogen bonds at all.
  !
  !     The integer CHBINI keeps track of which variables have been
  !     initialized by bit mapping. A value of -1 means that all variables
  !     are presumed to have been properly initialized. The map is as
  !     follows:
  !             1       CTOFHB
  !             2       CTONHB
  !             4       CTOFHA
  !             8       CTONHA
  !
  !     Author: Robert Bruccoleri
  !     Overhauled 13-FEB-1984 : BRB
  !
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use defltsm
  use hbondm
  use stream
  use string

  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  !
  LOGICAL QDUMMY
  real(chm_real)  C
  LOGICAL QSET,LINIT
  !
  LINIT=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'INIT').GT.0) THEN
     CHBINI=0
     CUTHB=DFCTHB
     CUTHBA=DFCTHA
     BEST=DFBEST
     HBEXCL=DFHBEX
     LHBFG=DFLHBF
     CALL TRIME(COMLYN,COMLEN)
     LINIT=(COMLEN.EQ.0)
  ENDIF
  IF(INDXA(COMLYN,COMLEN,'BEST').GT.0) BEST=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'ALL ').GT.0) BEST=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'HBEX').GT.0) HBEXCL=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'HBNO').GT.0) HBEXCL=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'ACCE').GT.0) LHBFG=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'NOAC').GT.0) LHBFG=.FALSE.
  QDUMMY=INDXA(COMLYN,COMLEN,'DUMM').GT.0
  IF (QDUMMY) THEN
     CUTHBA=0.0
     CUTHB=0.0
  ENDIF
  !
  ! Once a value is set, it will only be changed if it is reset
  ! this differs from the way it was done before version 16
  ! the angle default value differences have been changed as well.
  !
  !
  CUTHBA=GTRMF(COMLYN,COMLEN,'CUTHA',CUTHBA)
  CUTHBA=GTRMF(COMLYN,COMLEN,'CUTHBA',CUTHBA)
  CUTHB=GTRMF(COMLYN,COMLEN,'CUTHB',CUTHB)
  !
  QSET=CHBINI.EQ.-1.OR.MOD(CHBINI,2).GE.1
  IF(QSET) THEN
     C=CTOFHB
  ELSE
     C=-1.0
  ENDIF
  C=GTRMF(COMLYN,COMLEN,'CTOFHB',C)
  IF(C.EQ.-1.0) THEN
     CTOFHB=CUTHB-(DFCTHB-DFCFHB)
  ELSE
     CTOFHB=C
  ENDIF
  IF (CTOFHB.LT.0.0) CTOFHB=0.0
  IF (C.GE.0.0.AND..NOT.QSET) CHBINI=CHBINI+1
  !
  QSET=CHBINI.EQ.-1.OR.MOD(CHBINI,4).GE.2
  IF(QSET) THEN
     C=CTONHB
  ELSE
     C=-1.0
  ENDIF
  C=GTRMF(COMLYN,COMLEN,'CTONHB',C)
  IF(C.EQ.-1.0) THEN
     CTONHB=CTOFHB-(DFCFHB-DFCOHB)
  ELSE
     CTONHB=C
  ENDIF
  IF (CTONHB.LT.0.0) CTONHB=0.0
  IF (C.GE.0.0 .AND. .NOT.QSET) CHBINI=CHBINI+2
  !
  !
  QSET=CHBINI.EQ.-1 .OR. MOD(CHBINI,8).GE.4
  IF(QSET) THEN
     C=CTOFHA
  ELSE
     C=-1.0
  ENDIF
  C=GTRMF(COMLYN,COMLEN,'CTOFHA',C)
  IF(C.EQ.-1.0) THEN
     CTOFHA=CUTHBA-(DFCTHA-DFCFHA)
  ELSE
     CTOFHA=C
  ENDIF
  IF (CTOFHA.LT.0.0) CTOFHA=0.0
  IF (C.GE.0.0 .AND. .NOT.QSET) CHBINI=CHBINI+4
  !
  QSET=CHBINI.EQ.-1 .OR. MOD(CHBINI,16).GE.8
  IF(QSET) THEN
     C=CTONHA
  ELSE
     C=-1.0
  ENDIF
  C=GTRMF(COMLYN,COMLEN,'CTONHA',C)
  IF(C.EQ.-1.0) THEN
     CTONHA=CTOFHA-(DFCFHA-DFCOHA)
  ELSE
     CTONHA=C
  ENDIF
  IF (CTONHA.LT.0.0) CTONHA=0.0
  IF (C.GE.0.0 .AND. .NOT.QSET) CHBINI=CHBINI+8
  !
  ! finally print all otions
  IF (.NOT. LINIT) CALL PRNHBD(OUTU)
  RETURN
END SUBROUTINE GTHBCT

SUBROUTINE HBTRIM
  !
  ! TRIMS THE HBOND LIST TO INCLUDE ONLY HBONDS WITH LOW ENERGIES
  !
  !     Remark: the information about the current PSF in the ATOMID call
  !             is passed via the /PSF/ common block
  !
  !     NOV-82 AB, JUN-88 BRB
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use comand
  use exfunc
  use hbondm
  use param
  use coord
  use chutil,only:atomid
  use string
  implicit none
  !
  !
  real(chm_real)      DIST, ANGLE, ENERGY, ANTEAN, ECUT
  real(chm_real)      EHB
  INTEGER     I, II, NSLCT
  INTEGER     ANTE, DONOR, HYDRO, ACCE
  !
  CHARACTER(len=8) DN, LN, KN, AC, RESDN, RIDDN, SIDDN
  CHARACTER(len=8) RESAC, RIDAC, SIDAC, BLANKS
  !
  !     temporary mark used for hbonds to be deleted
  INTEGER, parameter :: MARK = -99999999
  DATA BLANKS/'        '/
  !
  !     nslct describes the number of "active" hydrogen bonds:
  !
  ECUT=NEXTF(COMLYN,COMLEN)
  !
  NSLCT=0
  DO I=1,NHB
     IF (IHB(I).GT.0) NSLCT=NSLCT+1
  ENDDO
  !
  ! a human comprehensable information is printed on OUTU
  !
  IF(PRNLEV.GE.2) WRITE(OUTU,1010) NSLCT,ECUT
1010 FORMAT(/,5X,'TRIM OF',I8,4X,'H-BOND PAIRS',/, &
       5X,'SELECTED WITH ENERGY CUTOFF=',F6.2/)
  IF(PRNLEV.GE.2) WRITE(OUTU,1020)
1020 FORMAT(/,17X,'HBOND ',26X,'ENERGY',1X,'DISTANCE',1X, &
       'A-H-D',2X,'H-A-AA')
  !
  DO I=1,NHB
     IF(IHB(I).GT.0) THEN
        DONOR=IHB(I)
        ACCE=JHB(I)
        HYDRO=KHB(I)
        ANTE=LHB(I)
        CALL ATOMID(DONOR,SIDDN,RIDDN,RESDN,DN)
        CALL ATOMID(ACCE,SIDAC,RIDAC,RESAC,AC)
        IF(HYDRO.GT.0) THEN
           CALL ATOMID(HYDRO,SIDDN,RIDDN,RESDN,KN)
        ELSE
           KN=BLANKS
        ENDIF
        IF(ANTE.GT.0) THEN
           CALL ATOMID(ANTE,SIDAC,RIDAC,RESAC,LN)
        ELSE
           LN=BLANKS
        ENDIF
        !
        !     now get the data energy, A-D distance, A-H-D angle, H-A-AA angle
        !
        CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1,CHBA, &
             CHBB,0,0,0,X,Y,Z,.FALSE.,0,3,ENERGY,0,0, &
             CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
             0,0,.FALSE.)
        !        
        IF(EHB.LE.ECUT) THEN
           CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,1,DIST,0,0, &
                CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN, &
                0,0,.FALSE.)
           IF(KHB(I).GT.0) THEN
              CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                   CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,2, &
                   ANGLE,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA, &
                   HBEXPN,0,0,.FALSE.)
              CALL EHBOND(EHB,IHB(I),JHB(I),KHB(I),LHB(I),ICH(I),1, &
                   CHBA,CHBB,0,0,0,X,Y,Z,.FALSE.,0,4, &
                   ANTEAN,0,0,CTONHB,CTOFHB,CTONHA,CTOFHA, &
                   HBEXPN,0,0,.FALSE.)
           ENDIF
           !
           ! finally write everthing on OUTU
           !
           IF(PRNLEV.GE.2) THEN
              IF(KHB(I).GT.0) THEN
                 WRITE(OUTU,1030) I,SIDDN(1:idleng), &
                      RIDDN(1:idleng),DN(1:idleng),KN(1:idleng), &
                      AC(1:idleng),LN(1:idleng),RIDAC(1:idleng), &
                      SIDAC(1:idleng),ENERGY,DIST,ANGLE,ANTEAN
              ELSE
                 WRITE(OUTU,1030) I,SIDDN(1:idleng), &
                      RIDDN(1:idleng),DN(1:idleng),KN(1:idleng), &
                      AC(1:idleng),LN(1:idleng),RIDAC(1:idleng), &
                      SIDAC(1:idleng),ENERGY,DIST
              ENDIF
           ENDIF
           !             
        ELSE
           IHB(I)=MARK
        ENDIF
     ENDIF
  ENDDO
1030 FORMAT(1X,I4,'(',A,' ',A,' ',A,'-',A,')(', &
       A,'-',A,' ',A,' ',A,')',4(F7.3,1X))
  !
  II=0
  DO I=1,NHB
     IF (IHB(I).GT.0) THEN
        II=II+1
        IHB(II)=IHB(I)
        JHB(II)=JHB(I)
        KHB(II)=KHB(I)
        LHB(II)=LHB(I)
        ICH(II)=ICH(I)
     ENDIF
  ENDDO
  NHB=II
  !
  RETURN
END SUBROUTINE HBTRIM
!
SUBROUTINE EDHBATOMS(COMLYN,COMLEN,OPER)
  !
  ! This routine modified the list of donors and acceptors
  ! that are in the PSF.
  !
  !           By Bernard R. Brooks - NIH - April 28, 1999
  !    Overhauled by L. Nilsson, KI, December 2005
  !
  !  { DONOr    } { SET    } [NOANtecedents] [SHOW] atom-selection
  !  { ACCEptor } { ADD    }
  !               { REMOve }
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use exfunc

  use hbondm
  use psf
  use memory
  use coord
  use select
  use string
  use param_store, only: set_param

  implicit none

  integer,allocatable,dimension(:) :: ISLCT
  INTEGER COMLEN
  CHARACTER(len=*) COMLYN
  CHARACTER(len=4) OPER

  CHARACTER(len=4) WRD
  LOGICAL QANTI,QDONO
  INTEGER I

  QDONO=(OPER.EQ.'DONO')
  WRD=NEXTA4(COMLYN,COMLEN)

  call chmalloc('hbonds.src','EDHBATOMS','ISLCT',NATOM,intg=ISLCT)
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

  QANTI = INDXA(COMLYN, COMLEN, 'NOAN') .EQ. 0

  IF(.NOT.QANTI.AND.QDONO) THEN
     ! Process donors without anticedents
     IF(PRNLEV.GE.2) WRITE(OUTU,35) 'donor','without'
     CALL EDHBATOM2(ISLCT,QANTI,WRD,IDON,IHD1,NDON)
     CALL set_param('NDON',NDON)
  ELSE IF(QDONO) THEN
     ! Process donors with anticedents
     IF(PRNLEV.GE.2) WRITE(OUTU,35) 'donor','with'
     CALL EDHBATOM2(ISLCT,QANTI,WRD,IHD1,IDON,NDON)
     CALL set_param('NDON',NDON)
  ELSE
     ! Process acceptors with or without anticedents
     IF(.NOT.QANTI.AND.PRNLEV.GE.2) WRITE(OUTU,35)'donor','without'
     IF(QANTI.AND.PRNLEV.GE.2) WRITE(OUTU,35) 'donor','with'
     CALL EDHBATOM2(ISLCT,QANTI,WRD,IACC,IAC1,NACC)
     CALL set_param('NACC',NACC)
  ENDIF
  !
35 FORMAT(' EDHBATOMS: Modifying ',A,' lists ',A,' antecedents')
  !
  call chmdealloc('hbonds.src','EDHBATOMS','ISLCT',NATOM,intg=ISLCT)
  !    
  IF(INDXA(COMLYN, COMLEN, 'SHOW') .GT. 0 .AND.PRNLEV.GE.2) THEN
     IF(QDONO) THEN
        WRITE(OUTU,'(/I8,A)') NDON,' !NDON: donors'
        WRITE(OUTU,'(8I8)') (IDON(I),IHD1(I),I=1,NDON)
     ELSE
        WRITE(OUTU,'(/I8,A)') NACC,' !NACC: acceptors'
        WRITE(OUTU,'(8I8)') (IACC(I),IAC1(I),I=1,NACC)
     ENDIF
  ENDIF
  !
  RETURN
END SUBROUTINE EDHBATOMS
!
SUBROUTINE EDHBATOM2(ISLCT,QANTI,WRD,IARR,JARR,NARR)
  !
  ! This routine modifies the list of donors and acceptors.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use stream
  use psf
  implicit none
  !
  INTEGER ISLCT(*)
  LOGICAL QANTI
  CHARACTER(len=4) WRD
  INTEGER IARR(*),JARR(*),NARR
  !
  INTEGER I,J,K,COUNT,NOLD
  EXTERNAL  EXCH5
  !
  NOLD=NARR
  !
  IF(WRD.EQ.'REMO') THEN
     J=0
     DO I=1,NARR
        IF(IARR(I).GT.0) THEN
           IF(ISLCT(IARR(I)).EQ.0) THEN
              J=J+1
              IARR(J)=IARR(I)
              JARR(J)=JARR(I)
           ENDIF
        ENDIF
     ENDDO
     NARR=J
     IF(PRNLEV.GE.2) WRITE(OUTU,45) NOLD-NARR
45   FORMAT(' EDHBATOMS:  A total of',I8,' entries were removed')
     RETURN
  ENDIF
  !
  IF(WRD.EQ.'ADD') THEN
     !        first, remove any from selection that already exist
     DO I=1,NARR
        IF(IARR(I).GT.0) ISLCT(IARR(I))=0
     ENDDO
  ENDIF
  !
  IF(WRD.EQ.'SET' .OR. WRD.EQ.'ADD') THEN
     IF(WRD.EQ.'SET') NARR=0
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) THEN
           NARR=NARR+1
           IF(NARR.GT.MAXPAD)  &
                CALL WRNDIE(-4,'<EDHBATOM>', &
                'DONOR/ACCEPTOR ARRAY OVERFLOW')
           IARR(NARR)=I
           JARR(NARR)=0
           IF(QANTI) THEN
              COUNT=0
              DO J=1,NBOND
                 IF(IB(J).EQ.I) THEN
                    K=JB(J)
                    COUNT=COUNT+1
                 ENDIF
                 IF(JB(J).EQ.I) THEN
                    K=IB(J)
                    COUNT=COUNT+1
                 ENDIF
              ENDDO
              IF(COUNT.EQ.1) JARR(NARR)=K
           ENDIF
        ENDIF
     ENDDO
  ELSE
     CALL WRNDIE(-1,'<EDHBATOM>','Illegal edit operation')
     RETURN
  ENDIF
  !
  ! Now sort the final list
  IF(WRD.EQ.'ADD') THEN
     CALL SORT(NARR,EXCH5,ORDER5,IARR,JARR,0,0,0,0,0,2)
     IF(PRNLEV.GE.2) WRITE(OUTU,65) NARR-NOLD
65   FORMAT(' EDHBATOMS:  A total of',I8,' entries were added')
  ELSE
     IF(PRNLEV.GE.2) WRITE(OUTU,75) NARR
75   FORMAT(' EDHBATOMS:  A total of',I8,' entries were set')
  ENDIF
  !          
  RETURN
END SUBROUTINE EDHBATOM2

