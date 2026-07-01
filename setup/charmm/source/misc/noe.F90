SUBROUTINE NOESET
  !
  ! Sets up NOE constraint force field.
  !
  ! Author: Axel Brunger
#if KEY_PNOE==1
  ! Overhauled PNOE part DEC-2004 MS  
#endif
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use memory
  !
  use psf
  implicit none
  ! local
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: JSLCT
  ! begin
#if KEY_NOMISC==1
  CALL WRNDIE(-1,'<NOESET>','NOE code is not compiled.')
#else
  call chmalloc('noe.src','NOESET','ISLCT',NATOM,intg=ISLCT)
  call chmalloc('noe.src','NOESET','JSLCT',NATOM,intg=JSLCT)
  CALL NOESE2(ISLCT,JSLCT)
  call chmdealloc('noe.src','NOESET','ISLCT',NATOM,intg=ISLCT)
  call chmdealloc('noe.src','NOESET','JSLCT',NATOM,intg=JSLCT)
#endif
  RETURN
END SUBROUTINE NOESET

#if KEY_NOMISC == 0
SUBROUTINE NOESE2(ISLCT,JSLCT)
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use psf
  use noem
  use comand
  use coord
  use select
  use string
  use stream
  use chutil,only:atomid
  use parallel, only:psnd4,psnd8
#if KEY_DOMDEC==1
  use domdec_common,only:domdec_system_changed
#endif
  implicit none
  !
  integer, dimension(natom) :: islct, jslct
  !
  ! local
  INTEGER I,J,N,II
  INTEGER IUNIT
  CHARACTER(len=4) WD
  CHARACTER(len=8) SIDI,RIDI,RENI,ACI
  CHARACTER(len=8) SIDJ,RIDJ,RENJ,ACJ
  real(chm_real) DCUT, TEMPR
  real(chm_real) RNKMN,RNRMN,RNKMX,RNRMX,RNFMX,RNTCN,RNEXP
#if KEY_PNOE==1
  real(chm_real) cc0x,cc0y,cc0z  
#endif
  !JH
  real(chm_real) RNSEX,RNRSW,RNRAM
  LOGICAL QERR,EOF,DONE,QANAL,OLD,QMINDIST
#if KEY_PNOE==1
  logical lpnoe  
#endif
  !
  ! begin
  !
  DONE=.FALSE.
  EOF=.FALSE.
  !
  DO WHILE(.NOT.DONE)
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
          'NOE> ')
     !
     ! This should not happen. for now there is no stream handling inside
     ! the noe module.
     IF(EOF) GOTO 220
     !
     WD=NEXTA4(COMLYN,COMLEN)
     !
     IF(WD.EQ.' ') THEN
        CONTINUE
     ELSE IF(WD.EQ.'TEMP') THEN    ! to restore old syntax
        TEMPR=NEXTF(COMLYN,COMLEN) ! bug#93091009801
     ELSE IF(WD.EQ.'RESE') THEN
        NOENUM=0
        NOENM2=0
        NOESCA=1.0
#if KEY_DOMDEC==1
        call domdec_system_changed()
#endif
     ELSE IF(WD.EQ.'ASSI') THEN
        CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE., &
             QERR)
        IF(QERR) THEN
           CALL WRNDIE(0,'<NOESET>','SELECTION ERROR')
           GOTO 120
        ENDIF
#if KEY_DOMDEC==1
        call domdec_system_changed()
#endif
        !
        OLD=INDEX(COMLYN,'M').EQ.0 .AND. INDEX(COMLYN,'T').EQ.0
        IF(OLD) THEN ! bug#93091009801
           RNRMN=NEXTF(COMLYN,COMLEN)
           RNRMX=RNRMN
           RNKMN=NEXTF(COMLYN,COMLEN)
           RNKMX=NEXTF(COMLYN,COMLEN)
           RNKMN=0.5*KBOLTZ*TEMPR/RNKMN**2
           RNKMX=0.5*KBOLTZ*TEMPR/RNKMX**2
           RNEXP=THREE
           RNTCN=ZERO
#if KEY_PNOE==1
           lpnoe=.FALSE.
           cc0x=ANUM
           cc0y=ANUM
           cc0z=ANUM
#endif 
        ELSE
           RNKMN=GTRMF(COMLYN,COMLEN,'KMIN',ZERO)
           RNRMN=GTRMF(COMLYN,COMLEN,'RMIN',ZERO)
           RNKMX=GTRMF(COMLYN,COMLEN,'KMAX',ZERO)
           RNRMX=GTRMF(COMLYN,COMLEN,'RMAX',ANUM)
           !              RNFMX=GTRMF(COMLYN,COMLEN,'FMAX',ANUM)
           RNFMX=GTRMF(COMLYN,COMLEN,'FMAX',ONE)
           RNTCN=GTRMF(COMLYN,COMLEN,'TCON',ZERO)
           RNEXP=GTRMF(COMLYN,COMLEN,'REXP',ONE)
           !JH
           RNSEX=GTRMF(COMLYN,COMLEN,'SEXP',ONE)
           RNRSW=GTRMF(COMLYN,COMLEN,'RSWI',MINONE)
           RNRAM=0
           if(INDXA(COMLYN,COMLEN,'SUMR').GT.0) RNRAM=1
           !
           QMINDIST=(INDXA(COMLYN,COMLEN,'MIND').GT.0)
           IF(QMINDIST) RNEXP=ONE
#if KEY_PNOE==1
           cc0x=GTRMF(COMLYN,COMLEN,'CNOX',ANUM)
           cc0y=GTRMF(COMLYN,COMLEN,'CNOY',ANUM)
           cc0z=GTRMF(COMLYN,COMLEN,'CNOZ',ANUM)
           lpnoe = ( (cc0x.ne.ANUM) .and. (cc0y.ne.ANUM) &
                .and. (cc0z.ne.ANUM) )
           !JH
           if(lpnoe) RNRSW=MINONE
#endif 
        ENDIF
        !
        if (noenum >= noemax) call noe_add_storage()
        NOENUM=NOENUM+1
        !
        NOEIPT(NOENUM)=NOENM2+1
        NOEINM(NOENUM)=0
        N=NOENM2
        DO I=1,NATOM
           IF(ISLCT(I).EQ.1) THEN
              if (noenm2 >= noemax) call noe_add_storage()
              NOENM2=NOENM2+1
              NOELIS(NOENM2)=I
              NOEINM(NOENUM)=NOEINM(NOENUM)+1
           ENDIF
        ENDDO
        NOEJPT(NOENUM)=NOENM2+1
        NOEJNM(NOENUM)=0

        !           IF(lpnoe) then we should skip the J-list because  
        !           it is never used; for now it is kept (MS, 2003):  

        DO J=1,NATOM
           IF(JSLCT(J).EQ.1) THEN
              if (noenm2 >= noemax) call noe_add_storage()
              NOENM2=NOENM2+1
              NOELIS(NOENM2)=J
              NOEJNM(NOENUM)=NOEJNM(NOENUM)+1
           ENDIF
        ENDDO
        !
        IF(NOEJNM(NOENUM).EQ.0 .OR. NOEINM(NOENUM).EQ.0) THEN
           CALL WRNDIE(0,'<NOESET>', &
                'Zero atom selected for this restraint. Ignored.')
           NOENUM=NOENUM-1
           NOENM2=N
           GOTO 120
        ENDIF
        !
#if KEY_PNOE==1
        IsPNOE(NOENUM)=lpnoe
        C0X(NOENUM)=cc0x
        C0Y(NOENUM)=cc0y
        C0Z(NOENUM)=cc0z
        MVPNOE(NOENUM)=.FALSE.
        !
#endif 
        NOEKMN(NOENUM)=RNKMN
        NOERMN(NOENUM)=RNRMN
        NOEKMX(NOENUM)=RNKMX
        NOERMX(NOENUM)=RNRMX
        NOEFMX(NOENUM)=RNFMX
        NOETCN(NOENUM)=RNTCN
        NOEEXP(NOENUM)=RNEXP
        NOEMIN(NOENUM)=QMINDIST
        !JH
        NOERSW(NOENUM)=RNRSW
        NOESEX(NOENUM)=RNSEX
        NOERAM(NOENUM)=RNRAM
        !
        IF(NOERMX(NOENUM).GT.ZERO) THEN
           NOEAVE(NOENUM)=ONE/NOERMX(NOENUM)**3
        ELSE
           NOEAVE(NOENUM)=ZERO
        ENDIF
        !
        IF(PRNLEV.GE.2) THEN
           IF(NOEINM(NOENUM).LE.1 .AND. NOEJNM(NOENUM).LE.1) THEN
              I=NOELIS(NOEIPT(NOENUM))
              J=NOELIS(NOEJPT(NOENUM))
              CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
              CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
#if KEY_PNOE==1
              if(IsPNOE(NOENUM)) then
                 WRITE(OUTU,509) NOENUM, &
                      I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
                      c0x(NOENUM),c0y(NOENUM),c0z(NOENUM), &
                      NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM)
509              FORMAT(' NOE: ADDING RESTRAINT BETWEEN ATOM ', &
                      I5,(I5,1X, A,1X,A,1X,A),/, &
                      '      AND POINT ', 3(f10.3,1x),/, &
                      ' RMIN=',F10.3, &
                      ' KMIN=',F10.3,' RMAX=',F10.3, &
                      ' KMAX=',F10.3,' FMAX=',F10.3, &
                      ' TCON=',F10.3)
                 !JH
              else if(RNRSW.GT.0) then
                 WRITE(OUTU,516) NOENUM, &
                      I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
                      J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
                      NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM), &
                      NOEEXP(NOENUM), &
                      NOERSW(NOENUM),NOESEX(NOENUM), &
                      NOERAM(NOENUM)
              else
                 WRITE(OUTU,510) NOENUM, &
                      I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
                      J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
                      NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM), &
                      NOEEXP(NOENUM)
              endif
#else /**/
              !JH
              if(RNRSW.GT.0) then
                 WRITE(OUTU,516) NOENUM, &
                      I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
                      J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
                      NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM), &
                      NOEEXP(NOENUM), &
                      NOERSW(NOENUM),NOESEX(NOENUM), &
                      NOERAM(NOENUM)
              else
                 WRITE(OUTU,510) NOENUM, &
                      I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
                      J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
                      NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM), &
                      NOEEXP(NOENUM)
              endif
#endif 
510           FORMAT(' NOE: ADDING RESTRAINT ',I5,2(I5,1X, &
                   A,1X,A,1X,A),/'   RMIN=',F10.3, &
                   ' KMIN=',F10.3,' RMAX=',F10.3, &
                   ' KMAX=',F10.3,' FMAX=',F10.3, &
                   ' TCON=',F10.3,' REXP=',F10.3)
              !JH
516           FORMAT(' NOE: ADDING RESTRAINT ',I5,2(I5,1X, &
                   A,1X,A,1X,A),/'   RMIN=',F8.3, &
                   ' KMIN=',F8.3,' RMAX=',F8.3, &
                   ' KMAX=',F8.3,' FMAX=',F8.3,/, &
                   '   TCON=',F8.3,' REXP=',F8.3, &
                   ' RSWI=',F8.3,' SEXP=',F8.3,' RAM=',I2)
           ELSE
              !
#if KEY_PNOE==1
              IF(IsPNOE(NOENUM)) THEN
                 WRITE(OUTU,'(A,2(A,1X,I5))') ' NOE: ADDING', &
                      ' PNOE RESTRAINT #',NOENUM, &
                      ', # atoms 1st set',NOEINM(NOENUM)
              ELSE
#endif 
                 WRITE(OUTU,'(3(A,1X,I5))') &
                      '  NOE: ADDING RESTRAINT #',NOENUM, &
                      ', # atoms 1st set',NOEINM(NOENUM), &
                      ', # atoms 2nd set',NOEJNM(NOENUM)
#if KEY_PNOE==1
              ENDIF  
#endif
              !
              DO I=1,NOEINM(NOENUM)
                 II=NOELIS(NOEIPT(NOENUM)+I-1)
                 CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
                 WRITE(OUTU,511) II,SIDI(1:idleng), &
                      RIDI(1:idleng),ACI(1:idleng)
511              FORMAT('        FIRST SET ATOM:',I5,3(1X,A))
              ENDDO
              !
#if KEY_PNOE==1
              IF(IsPNOE(NOENUM)) THEN
                 WRITE(OUTU,'(A,3(1X,F10.3))') &
                      '       WITH PNOE POINT:', &
                      C0X(NOENUM),C0Y(NOENUM),C0Z(NOENUM)
              ELSE
#endif 
                 DO I=1,NOEJNM(NOENUM)
                    II=NOELIS(NOEJPT(NOENUM)+I-1)
                    CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
                    WRITE(OUTU,512) II,SIDI(1:idleng), &
                         RIDI(1:idleng),ACI(1:idleng)
512                 FORMAT('       SECOND SET ATOM:',I5,3(1X,A))
                 ENDDO
#if KEY_PNOE==1
              ENDIF  
#endif
              !JH
              if(RNRSW.GT.0) then
                 WRITE(OUTU,517) NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM), &
                      NOEEXP(NOENUM), &
                      NOERSW(NOENUM),NOESEX(NOENUM), &
                      NOERAM(NOENUM)
              else
                 WRITE(OUTU,515) NOERMN(NOENUM),NOEKMN(NOENUM), &
                      NOERMX(NOENUM),NOEKMX(NOENUM), &
                      NOEFMX(NOENUM),NOETCN(NOENUM)
515              FORMAT('   RMIN=',F10.3,' KMIN=',F10.3,' RMAX=',F10.3, &
                      ' KMAX=',F10.3,' FMAX=',F10.3,' TCON=',F10.3)
517              FORMAT('   RMIN=',F8.3,' KMIN=',F8.3,' RMAX=',F8.3, &
                      ' KMAX=',F8.3,' FMAX=',F8.3,/, &
                      '   TCON=',F8.3,' REXP=',F8.3, &
                      ' RSWI=',F8.3,' SEXP=',F8.3, ' RAM=',I2)
                 !
                 IF(NOEMIN(NOENUM))THEN
                    WRITE(OUTU,'(1x,a)') &
                         'Minimum distance restraint for the group of atom'
                 ENDIF
              endif
              !
           ENDIF
        ENDIF
120     CONTINUE
        !
#if KEY_PNOE==1
     ELSE IF(WD.EQ.'MPNO') THEN
        !           moving point-NOE:
        I=GTRMI(COMLYN,COMLEN,'INOE',0)
        cc0x=GTRMF(COMLYN,COMLEN,'TNOX',ANUM)
        cc0y=GTRMF(COMLYN,COMLEN,'TNOY',ANUM)
        cc0z=GTRMF(COMLYN,COMLEN,'TNOZ',ANUM)
        IF((I.GT.0).AND.(I.LE.NOENUM).AND. &
             (cc0x.ne.ANUM) .and. (cc0y.ne.ANUM) &
             .and. (cc0z.ne.ANUM)) THEN
           IF(.NOT.IsPNOE(I)) &
                CALL WRNDIE(-2,'<NOESET>','ERROR -- ONLY PNOEs CAN MOVE')
           MVPNOE(I)=.TRUE.
           IMPNOE=0
           TC0X(I)=cc0x
           TC0Y(I)=cc0y
           TC0Z(I)=cc0z
           OC0X(I)=C0X(I)
           OC0Y(I)=C0Y(I)
           OC0Z(I)=C0Z(I)
        ELSE
           CALL WRNDIE(-2,'<NOESET>','ERROR READING MOVING PNOE')
        ENDIF
     ELSE IF(WD.EQ.'NMPN') THEN
        !           specify No of steps for motion:
        IMPNOE=0
        NMPNOE=NEXTF(COMLYN,COMLEN)
        IF(PRNLEV.GE.2) WRITE(OUTU,610) NMPNOE
610     FORMAT('  NOE: MOVING PNOEs OVER',I10,' STEPS')
#endif 
     ELSE IF(WD.EQ.'WRIT' .OR. WD.EQ.'PRIN') THEN
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
        DCUT=GTRMF(COMLYN,COMLEN,'CUT',ZERO)
        QANAL=(INDXA(COMLYN,COMLEN,'ANAL').GT.0)
        CALL NOEWRI(IUNIT,QANAL,DCUT)
        !
     ELSE IF(WD.EQ.'READ') THEN
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
        CALL NOEREA(IUNIT)
        !
     ELSE IF(WD.EQ.'SCAL') THEN
        NOESCA=NEXTF(COMLYN,COMLEN)
        IF(PRNLEV.GE.2) WRITE(OUTU,620) NOESCA
620     FORMAT('  NOE: SCALE FACTOR SET TO',F10.3)
     ELSE IF(WD.EQ.'END ') THEN
        DONE=.TRUE.
     ELSE
        CALL WRNDIE(0,'<NOESET>','UNKNOWN OPTION')
     ENDIF
  ENDDO
  !
220 CONTINUE
  IF(PRNLEV.GE.2) WRITE(OUTU,230) NOENUM
230 FORMAT(' NOE:  CURRENT NUMBER OF CONSTRAINTS=',I4)
#if KEY_PARALLEL==1
  !     added broadcast of NOE parameters (up to c32a1 only hidden
  !     in routines NOERA and NOEREA_OLD, why not here???),
  !     PSND's should be removed in NOEREA and NOEREA_OLD, MS 2004.
  CALL PSND4(NOENUM,1)
  CALL PSND4(NOENM2,1)
#if KEY_PNOE==1
  CALL PSND4(NMPNOE,1)  
#endif
#if KEY_PNOE==1
  CALL PSND4(IMPNOE,1)  
#endif
  CALL PSND8(NOESCA,1)
  IF(NOENUM.GT.0) THEN
     CALL PSND4(NOEINM,NOENUM)
     CALL PSND4(NOEIPT,NOENUM)
     CALL PSND4(NOEJNM,NOENUM)
     CALL PSND4(NOEJPT,NOENUM)
     CALL PSND4(NOELIS,NOENM2)
#if KEY_PNOE==1
     CALL PSND4(IsPNOE,NOENUM)  
#endif
#if KEY_PNOE==1
     CALL PSND4(MVPNOE,NOENUM)  
#endif
     CALL PSND8(NOERMN,NOENUM)
     CALL PSND8(NOEKMN,NOENUM)
     CALL PSND8(NOERMX,NOENUM)
     CALL PSND8(NOEKMX,NOENUM)
     CALL PSND8(NOEFMX,NOENUM)
     CALL PSND8(NOETCN,NOENUM)
     CALL PSND8(NOEEXP,NOENUM)
     CALL PSND8(NOEAVE,NOENUM)
     !     and for soft soft asymptote NOE:
     CALL PSND8(NOERSW,NOENUM)
     CALL PSND8(NOESEX,NOENUM)
     CALL PSND4(NOERAM,NOENUM)

#if KEY_PNOE==1
     CALL PSND8(C0X,NOENUM)
     CALL PSND8(C0Y,NOENUM)
     CALL PSND8(C0Z,NOENUM)
     CALL PSND8(OC0X,NOENUM)
     CALL PSND8(OC0Y,NOENUM)
     CALL PSND8(OC0Z,NOENUM)
     CALL PSND8(TC0X,NOENUM)
     CALL PSND8(TC0Y,NOENUM)
     CALL PSND8(TC0Z,NOENUM)
#endif 
  ENDIF
#endif 
  RETURN
END SUBROUTINE NOESE2

SUBROUTINE NOEWRI(IUNIT,QANAL,DCUT)
  !
  ! print a list of current NOE constraints
  ! ANALys option output format changed FEB-85 LN
  ! Overhauled SEP-87 BRB
  !
  use chm_kinds
  use dimens_fcm
  use number
  use version
  use noem
  use ctitla
  use coord
  use stream
  use chutil,only:atomid
  use param_store, only: set_param

  implicit none
  !
  INTEGER IUNIT
  LOGICAL QANAL
  real(chm_real) DCUT
  !
  ! local
  CHARACTER(len=8) SIDI,RIDI,RENI,ACI
  INTEGER N,II,JJ,I,J,NIJ
  INTEGER VIOLNU
  real(chm_real) SIJ,RIJ,DR,DEL,DF,EIJ,XIJ,YIJ,ZIJ,RLIM
  real(chm_real) ATOT,AEXP ! ATEMP
  real(chm_real) VIOL
  LOGICAL QLIST
  real(chm_real) MINIJ
  INTEGER IIMIN,JJMIN
  !JH paramters A,B in soft asymptote NOE potential
  real(chm_real) SOFTA,SOFTB
  LOGICAL QSOFT

  ! begin
  !
  QLIST=(OUTU.EQ.IUNIT)
  DR=ZERO
  VIOL=ZERO
  VIOLNU=0
  !
  IF(QLIST) THEN
     IF(PRNLEV.LT.2) RETURN
     IF(QANAL .AND. DCUT.GT.0.0) THEN
        WRITE(IUNIT,490) DCUT
     ELSE
        WRITE(IUNIT,495)
     ENDIF
490  FORMAT(' NOEPRI: LISTING FOR CONSTRAINTS WITH RDELTA >',F7.3)
495  FORMAT(' NOEPRI: LISTING OF ALL CONSTRAINTS')
     WRITE(IUNIT,27) NOENUM,NOESCA
27   FORMAT('  NUMBER OF CONSTRAINTS=',I4,'  SCALE FACTOR=',F10.3/)
  ELSE
     CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
     IF(IOLEV.LT.0) RETURN
     CALL WRTITL(TITLEA,NTITLA,IUNIT,0)
     WRITE(IUNIT,28) NOENUM,NOENM2,VERNUM,NOESCA
28   FORMAT(3I5,F10.4)
  ENDIF
  !
  !     Write header
  !
  WRITE (IUNIT,'(A)') &
       '  NOE             ATOMS'
  WRITE (IUNIT,'(A)') &
       '             RMIN        KMIN        RMAX' // &
       '        KMAX        FMAX        TCON        REXP'
  IF(QANAL) THEN
     WRITE (IUNIT,'(A)') &
          '                 R       RDELTA          ENERGY' // &
          '        FORCE         RAVE'
  ELSE
     WRITE (IUNIT,*)
  ENDIF
  !JH
  N=1
  QSOFT=.false.
  do while(.not.QSOFT.and.N.le.NOENUM) ! if any NOE has soft asymptote
     QSOFT=NOERSW(N).GT.0
     N=N+1
  enddo
  if(QSOFT) then
     WRITE (IUNIT,*)
     WRITE (IUNIT,'(A)') '  NOEs with soft asymptote are in format:'
     WRITE (IUNIT,'(A)') &
          '     RMIN     KMIN     RMAX     KMAX     FMAX' // &
          '     TCON     REXP     RSWI     SEXP RAM'
     IF(QANAL) THEN
        WRITE (IUNIT,'(A)') &
             '    R       RDELTA    ENERGY   FORCE    RAVE'
     ELSE
        WRITE (IUNIT,*)
     ENDIF
     WRITE (IUNIT,*)
  endif
  !
  !     Now write the noe restraint parameters
  !
  DO N=1,NOENUM
     IF(QANAL) THEN
        ATOT=ZERO
        AEXP=HALF/NOEEXP(N)
        NIJ=0
        DO I=1,NOEINM(N)
           II=NOELIS(NOEIPT(N)+I-1)
#if KEY_PNOE==1
           IF(IsPNOE(N)) THEN
              XIJ=X(II)-C0X(N)
              YIJ=Y(II)-C0Y(N)
              ZIJ=Z(II)-C0Z(N)
              SIJ=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
              ATOT=ATOT+SIJ**AEXP
              NIJ=NIJ+1
              IF(NOEMIN(N))THEN
                 IF(NIJ.EQ.1)THEN
                    MINIJ=SIJ
                    IIMIN=II
                    JJMIN=0
                 ELSE
                    IF(SIJ.LT.MINIJ)THEN
                       MINIJ=SIJ
                       IIMIN=II
                       JJMIN=0
                    ENDIF
                 ENDIF
              ENDIF
           ELSE
#endif 
              DO J=1,NOEJNM(N)
                 JJ=NOELIS(NOEJPT(N)+J-1)
                 XIJ=X(II)-X(JJ)
                 YIJ=Y(II)-Y(JJ)
                 ZIJ=Z(II)-Z(JJ)
                 SIJ=XIJ*XIJ+YIJ*YIJ+ZIJ*ZIJ
                 ATOT=ATOT+SIJ**AEXP
                 NIJ=NIJ+1
                 IF(NOEMIN(N))THEN
                    IF(NIJ.EQ.1)THEN
                       MINIJ=SIJ
                       IIMIN=II
                       JJMIN=JJ
                    ELSE
                       IF(SIJ.LT.MINIJ)THEN
                          MINIJ=SIJ
                          IIMIN=II
                          JJMIN=JJ
                       ENDIF
                    ENDIF
                 ENDIF
              ENDDO
#if KEY_PNOE==1
           ENDIF  
#endif
        ENDDO
        !
        IF(NOEMIN(N))THEN
           RIJ=SQRT(MINIJ)
        ELSE
           !JH
           if(NOERAM(N).EQ.0) ATOT=ATOT/NIJ ! average instead of summation
           RIJ=ATOT**NOEEXP(N)
        ENDIF
        !
        DF=0.0
        EIJ=0.0
        DR=0.0

        IF(RIJ.LT.NOERMN(N)) THEN
           DEL=RIJ-NOERMN(N)
           DF=NOESCA*NOEKMN(N)*DEL
           EIJ=0.5*NOESCA*NOEKMN(N)*DEL*DEL
           DR=-DEL
        ENDIF
        !JH
        if(NOERSW(N).GT.0) then  ! new NOE with soft asymptote

           IF(RIJ.GT.NOERMX(N).AND.RIJ.LE.NOERMX(N)+NOERSW(N)) THEN
              DEL=RIJ-NOERMX(N)
              DF=DF+NOESCA*NOEKMX(N)*DEL
              EIJ=EIJ+0.5*NOESCA*NOEKMX(N)*DEL*DEL
              DR=DEL
           ENDIF
           IF(RIJ.GT.NOERMX(N)+NOERSW(N)) THEN
              SOFTB=(NOEFMX(N)-NOERSW(N))/NOESEX(N)* &
                   NOERSW(N)**(NOESEX(N)+1)
              SOFTA=0.5*NOERSW(N)**2-NOEFMX(N)*NOERSW(N)- &
                   NOERSW(N)*(NOEFMX(N)-NOERSW(N))/NOESEX(N)
              DEL=RIJ-NOERMX(N)
              DF=NOESCA*NOEKMX(N)* &
                   (NOEFMX(N)-SOFTB/DEL**(NOESEX(N)+1))
              EIJ=NOESCA*NOEKMX(N)* &
                   (SOFTA+SOFTB/DEL**NOESEX(N)+NOEFMX(N)*DEL)
              DR=DEL
           ENDIF

        else                ! old NOE

           IF(RIJ.GT.NOERMX(N)) THEN
              DEL=RIJ-NOERMX(N)
              DF=DF+NOESCA*NOEKMX(N)*DEL
              EIJ=EIJ+0.5*NOESCA*NOEKMX(N)*DEL*DEL
              IF(DEL.GT.DR) DR=DEL
           ENDIF
           !
           IF(NOEKMX(N).GT.0.0) THEN
              RLIM=NOERMX(N)+NOEFMX(N)/NOEKMX(N)
           ELSE
              RLIM=9999.0
           ENDIF
           IF(RIJ.GT.RLIM) THEN
              DEL=RIJ-RLIM
              DF=DF-NOESCA*NOEKMX(N)*DEL
              EIJ=EIJ-0.5*NOESCA*NOEKMX(N)*DEL*DEL
           ENDIF

        endif
        VIOL=VIOL+DR
        IF(DR.GT.DCUT)VIOLNU=VIOLNU+1
        !
     ENDIF
     !
     IF((.NOT.QANAL).OR.(.NOT.QLIST).OR.(DR.GE.DCUT)) THEN
        !
        ! Write the list of participating atoms
#if KEY_PNOE==1
        IF(IsPNOE(N)) THEN
           WRITE(IUNIT,510) '      PNOE:',N,NOEINM(N),0
        ELSE
#endif 
           WRITE(IUNIT,510) ' RESTRAINT:',N,NOEINM(N),NOEJNM(N)
#if KEY_PNOE==1
        ENDIF  
#endif
510     FORMAT(A11,3I5)
        !
        DO I=1,NOEINM(N)
           II=NOELIS(NOEIPT(N)+I-1)
           CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
           WRITE(IUNIT,511) II,SIDI(1:idleng), &
                RIDI(1:idleng),ACI(1:idleng)
511        FORMAT('        FIRST SET ATOM:',I5,3(1X,A))
        ENDDO
        !
#if KEY_PNOE==1
        IF(IsPNOE(N)) THEN
           WRITE(IUNIT,513) C0X(N),C0Y(N),C0Z(N)
        ELSE
#endif 
           DO I=1,NOEJNM(N)
              II=NOELIS(NOEJPT(N)+I-1)
              CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
              WRITE(IUNIT,512) II,SIDI(1:idleng), &
                   RIDI(1:idleng),ACI(1:idleng)
512           FORMAT('       SECOND SET ATOM:',I5,3(1X,A))
           ENDDO
#if KEY_PNOE==1
        ENDIF  
#endif
#if KEY_PNOE==1
513     FORMAT('       WITH PNOE POINT:',3(1X,F10.3))  
#endif
        !JH
        if(NOERSW(N).GT.0) then
           WRITE (IUNIT,127) NOERMN(N),NOEKMN(N),NOERMX(N), &
                NOEKMX(N),NOEFMX(N),NOETCN(N),NOEEXP(N), &
                NOERSW(N),NOESEX(N),NOERAM(N)
127        FORMAT(9F9.3,1X,I2)
        else
           WRITE (IUNIT,128) NOERMN(N),NOEKMN(N),NOERMX(N), &
                NOEKMX(N),NOEFMX(N),NOETCN(N),NOEEXP(N)
128        FORMAT(5X,7F12.3)
        endif
        IF(NOEMIN(N))THEN
           WRITE(OUTU,'(a)') &
                'Minimum distance restraint for the group of atom'
#if KEY_PNOE==1
           IF(IsPNOE(N)) THEN
              WRITE(OUTU,'(a,i6)') 'nearest PNOE atom found: ',IIMIN
           ELSE
#endif 
              WRITE(OUTU,'(a,2i6)') 'nearest pair found: ',IIMIN,JJMIN
#if KEY_PNOE==1
           ENDIF  
#endif
        ENDIF
        IF(QANAL) THEN
           WRITE(IUNIT,128) RIJ,DR,EIJ,DF,NOEAVE(N)
        ELSE
           WRITE (IUNIT,*)
        ENDIF
     ENDIF
     !
  ENDDO

  call set_param('VIOL',VIOL)
  !JH
  CALL set_param('VIOLNU',VIOLNU)
  WRITE (IUNIT,129) VIOL
129 FORMAT('TOTAL Violations are',7F12.3)
  WRITE (IUNIT,130) VIOLNU
130 FORMAT('TOTAL # of Violations is',I6)
  !
  RETURN
END SUBROUTINE NOEWRI

SUBROUTINE NOEREA(IUNIT)
  !
  !     THIS ROUTINE READS AN NOE CARD FILE
  !
  !     By Bernard R. Brooks    1987
  !(soft asymptote is missing here)
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use ctitla
  use noem
  use psf
  use string
  use stream
  use version
  use chutil,only:getatn
  use parallel,only: psnd4,psnd8

  implicit none
  !
  INTEGER IUNIT
  !
  !
  INTEGER I,N,IPT,IVERS,II
  CHARACTER(len=8) AII,AIS,AIR
  CHARACTER(len=12) KEYW
  integer      llen
  character(len=80) line
  !
  IF(IUNIT.NE.OUTU .AND. PRNLEV.GE.2) WRITE(OUTU,822) IUNIT
822 FORMAT(' NOE RESTRAINTS READ FROM UNIT',I3)
  !
  IF(IOLEV.GT.0) THEN
     CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
     READ(IUNIT,28,ERR=900) NOENUM,NOENM2,IVERS,NOESCA
28   FORMAT(3I5,F10.4)
     IF(IVERS.NE.VERNUM) GOTO 900
     !
     READ(IUNIT,*)
     READ(IUNIT,*)
     READ(IUNIT,*)
     !
     IPT=1
     DO N=1,NOENUM
        !
        READ(IUNIT,510,END=520) KEYW,I,NOEINM(N),NOEJNM(N)
510     FORMAT(A11,3I5)
        GOTO 530
520     CONTINUE
        !              end of file encountered...
        NOENUM=N-1
        NOENM2=IPT-1
        GOTO 540
530     CONTINUE
        !
        IF(I.NE.N) GOTO 924
#if KEY_PNOE==1
        IsPNOE(N)=(INDEX(KEYW,'PNOE').GT.0)  
#endif
#if KEY_PNOE==1
        MVPNOE(N)=.FALSE.                    
#endif
        NOEIPT(N)=IPT
        IPT=IPT+NOEINM(N)
        ! 511       FORMAT(23X,I5,1X,A4,1X,A4,1X,A4) for II,AIS,AIR,AII
        DO I=1,NOEINM(N)
           !              READ(IUNIT,511,ERR=924,END=924) II,AIS,AIR,AII
           READ(IUNIT,'(23x,a)',ERR=924,END=924) line
           llen=len(line)
           ii=nexti(line,llen)
           ais=nexta8(line,llen)
           air=nexta8(line,llen)
           aii=nexta8(line,llen)
           II=GETATN(AIS,AIR,AII,SEGID,RESID,ATYPE,IBASE, &
                NICTOT,NSEGT)
           IF(II.LE.0) GOTO 924
           NOELIS(NOEIPT(N)+I-1)=II
        ENDDO
        !
        NOEJPT(N)=IPT
        IPT=IPT+NOEJNM(N)
#if KEY_PNOE==1
        IF(IsPNOE(N)) THEN
           !              there is just one line to read with x,y,z of point:
           READ(IUNIT,513,ERR=924,END=924) C0X(N),C0Y(N),C0Z(N)
           !              for back-compatibility, copy I-list over to J-list
           DO I=1,NOEJNM(N)
              NOELIS(NOEJPT(N)+I-1)=NOELIS(NOEIPT(N)+I-1)
           ENDDO
        ELSE
           !              make sure an inadvertent access to C0X,Y,Z shows:
           C0X(N)=ANUM
           C0Y(N)=ANUM
           C0Z(N)=ANUM
#endif 
           DO I=1,NOEJNM(N)
              !              READ(IUNIT,511,ERR=924,END=924) II,AIS,AIR,AII
              READ(IUNIT,'(23x,a)',ERR=924,END=924) line
              llen=len(line)
              ii=nexti(line,llen)
              ais=nexta8(line,llen)
              air=nexta8(line,llen)
              aii=nexta8(line,llen)
              II=GETATN(AIS,AIR,AII,SEGID,RESID,ATYPE,IBASE, &
                   NICTOT,NSEGT)
              IF(II.LE.0) GOTO 924
              NOELIS(NOEJPT(N)+I-1)=II
           ENDDO
#if KEY_PNOE==1
        ENDIF  
#endif
#if KEY_PNOE==1
513     FORMAT(23X,3(1X,F10.3))  
#endif

        READ(IUNIT,128) NOERMN(N),NOEKMN(N),NOERMX(N),NOEKMX(N), &
             NOEFMX(N),NOETCN(N),NOEEXP(N)
128     FORMAT(5X,7F12.3)
        !
        NOEAVE(N)=NOERMX(N)
        READ(IUNIT,*)
     ENDDO
540  CONTINUE
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(NOENUM,1)
  CALL PSND4(NOENM2,1)
#if KEY_PNOE==1
  CALL PSND4(NMPNOE,1)  
#endif
#if KEY_PNOE==1
  CALL PSND4(IMPNOE,1)  
#endif
  CALL PSND8(NOESCA,1)
  IF(NOENUM.GT.0) THEN
     CALL PSND4(NOEINM,NOENUM)
     CALL PSND4(NOEIPT,NOENUM)
     CALL PSND4(NOEJNM,NOENUM)
     CALL PSND4(NOEJPT,NOENUM)
     CALL PSND4(NOELIS,NOENM2)
#if KEY_PNOE==1
     CALL PSND4(IsPNOE,NOENUM)  
#endif
#if KEY_PNOE==1
     CALL PSND4(MVPNOE,NOENUM)  
#endif
     CALL PSND8(NOERMN,NOENUM)
     CALL PSND8(NOEKMN,NOENUM)
     CALL PSND8(NOERMX,NOENUM)
     CALL PSND8(NOEKMX,NOENUM)
     CALL PSND8(NOEFMX,NOENUM)
     CALL PSND8(NOETCN,NOENUM)
     CALL PSND8(NOEEXP,NOENUM)
     CALL PSND8(NOEAVE,NOENUM)
#if KEY_PNOE==1
     CALL PSND8(C0X,NOENUM)
     CALL PSND8(C0Y,NOENUM)
     CALL PSND8(C0Z,NOENUM)
     CALL PSND8(OC0X,NOENUM)
     CALL PSND8(OC0Y,NOENUM)
     CALL PSND8(OC0Z,NOENUM)
     CALL PSND8(TC0X,NOENUM)
     CALL PSND8(TC0Y,NOENUM)
     CALL PSND8(TC0Z,NOENUM)
#endif 
  ENDIF
#endif 
  !
  RETURN
  !
900 CONTINUE
  CALL WRNDIE(-1,'<NOEREA>', &
       'OLD OR BAD VERSION OF NOE FILE.  CANNOT READ')
  IF(PRNLEV.GE.2) &
       WRITE(OUTU,'(A)') ' NOE> Will try to read old format'
  REWIND IUNIT
  CALL NOEREA_OLD(IUNIT)
  RETURN
924 CONTINUE
  CALL WRNDIE(-1,'<NOEREA>','INPUT ERROR OR EOF ENCOUNTERED')
  NOENUM=0
  NOENM2=0
  NOESCA=1.0
#if KEY_PARALLEL==1
  CALL PSND4(NOENUM,1)
  CALL PSND4(NOENM2,1)
  CALL PSND8(NOESCA,1)
#endif 
  RETURN
END SUBROUTINE NOEREA

SUBROUTINE NOECNS(EN,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
     NOENUM,NOESCA,NOEIPT,NOEJPT,NOEINM, &
     NOEJNM,NOELIS,NOEEXP,NOERMN, &
     NOEKMN,NOERMX,NOEKMX,NOEFMX, &
     NOETCN,NOEAVE,NOEMIN,DD1,IUPT,QSECD &
     !JH
     ,NOERSW,NOESEX,NOERAM &
#if KEY_PNOE==1
     , IsPNOE, C0X, C0Y, C0Z   & 
#endif
#if KEY_PNOE==1
     , MVPNOE,OC0X,OC0Y,OC0Z   & 
#endif
#if KEY_PNOE==1
     ,TC0X,TC0Y,TC0Z   & 
#endif
#if KEY_PNOE==1
     , NMPNOE, IMPNOE          & 
#endif
     )
  !
  ! Routine computes force field for NOE constraints
  !
  ! C   in [KCAL/(MOL*A**2)],  d in [A] units.
  !  IJ
  !
  ! original author: Axel Brunger
  ! Overhauled by B. Brooks  Sept-1987
  !
  use chm_kinds
  use number
  use dimens_fcm
  use reawri
#if KEY_DIMB==1
  use dimb  
#endif
#if KEY_GENETIC==1
  use galgor        
#endif
#if KEY_FOURD==1
  use fourdm         
#endif
  implicit none
  !
  ! I/O
  real(chm_real) EN
  real(chm_real) DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER NOENUM
  real(chm_real) NOESCA
  INTEGER NOEIPT(*), NOEJPT(*), NOEINM(*), NOEJNM(*), NOELIS(*)
  real(chm_real) NOEEXP(*)
  real(chm_real) NOERMN(*), NOEKMN(*), NOERMX(*), NOEKMX(*),  &
       NOEFMX(*)
  real(chm_real) NOETCN(*), NOEAVE(*)
  !JH
  real(chm_real) NOERSW(*), NOESEX(*)
  INTEGER NOERAM(*)
  LOGICAL NOEMIN(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
#if KEY_PNOE==1
  LOGICAL IsPNOE(*), MVPNOE(*)
  real(chm_real)  C0X(*), C0Y(*), C0Z(*)
  real(chm_real)  OC0X(*),OC0Y(*),OC0Z(*)
  real(chm_real)  TC0X(*),TC0Y(*),TC0Z(*)
  INTEGER NMPNOE, IMPNOE
#endif 
  !
  !
#if KEY_GENETIC==1
  integer istart
#endif 
#if KEY_FOURD==1 /*4ddecl*/
  ! 4D varibles:
  real(chm_real) FDIJ,RFD
#endif /* (4ddecl)*/
  !
  ! local
  real(chm_real) XIJ,YIJ,ZIJ,SIJ,RIJ,EIJ,DF,DEL,RLIM,DDF,DFA
  real(chm_real) RX,RY,RZ,A,RIJU,AEXP,ATOT,SOFTA,SOFTB
  INTEGER I,J,N,II,JJ,IADD,NIJ
  INTEGER NWARN
#if KEY_PNOE==1
  INTEGER NWARN2  
#endif
#if KEY_PNOE==1
  LOGICAL MOVED   
#endif
  real(chm_real) MINIJ
  INTEGER IMIN,JMIN

  ! begin
  NWARN=0
#if KEY_PNOE==1
  NWARN2=0       
#endif
#if KEY_PNOE==1
  MOVED=.FALSE.  
#endif
  !
#if KEY_GENETIC==1
  istart = 1
  if(qgalgor) then
     istart=int(EN)
  endif
  EN=0.
  DO N=istart, NOENUM
#else /**/
  EN=0.
  DO N=1,NOENUM
#endif 
     !
#if KEY_PNOE==1
     ! check whether it is a moving PNOE; if yes, shift it:
     IF((IsPNOE(N)).AND.(MVPNOE(N))) THEN
        IF((IMPNOE.GE.0).AND.(IMPNOE.LE.NMPNOE)) THEN
           MOVED=.TRUE.
           XIJ=TC0X(N)-OC0X(N)
           YIJ=TC0Y(N)-OC0Y(N)
           ZIJ=TC0Z(N)-OC0Z(N)
           DEL=DBLE(IMPNOE)/DBLE(NMPNOE)
           C0X(N)=OC0X(N)+XIJ*DEL
           C0Y(N)=OC0Y(N)+YIJ*DEL
           C0Z(N)=OC0Z(N)+ZIJ*DEL
        ENDIF
     ENDIF
#endif 
     ! loop over all pairs of atoms belonging to constraint N
     ATOT=ZERO
     AEXP=HALF/NOEEXP(N)
     NIJ=0
     DO II=1,NOEINM(N)
        I=NOELIS(NOEIPT(N)+II-1)
#if KEY_PNOE==1
        if(IsPNOE(N)) then
           rx=x(i)-c0x(N)
           ry=y(i)-c0y(N)
           rz=z(i)-c0z(N)
           SIJ=RX*RX+RY*RY+RZ*RZ
           ATOT=ATOT+SIJ**AEXP
           NIJ=NIJ+1
           IF(NOEMIN(N))THEN
              IF(NIJ.EQ.1)THEN
                 MINIJ=SIJ
                 IMIN=I
                 JMIN=0
              ELSE
                 IF(SIJ.LT.MINIJ)THEN
                    MINIJ=SIJ
                    IMIN=I
                    JMIN=0
                 ENDIF
              ENDIF
           ENDIF
        else
#endif 
           DO JJ=1,NOEJNM(N)
              J=NOELIS(NOEJPT(N)+JJ-1)
              RX=X(I)-X(J)
              RY=Y(I)-Y(J)
              RZ=Z(I)-Z(J)
              SIJ=RX*RX+RY*RY+RZ*RZ
#if KEY_FOURD==1
              IF(DIM4) THEN
                 RFD=FDIM(I)-FDIM(J)
                 SIJ=SIJ+RFD*RFD
              ENDIF
#endif 
              ATOT=ATOT+SIJ**AEXP
              NIJ=NIJ+1
              IF(NOEMIN(N))THEN
                 IF(NIJ.EQ.1)THEN
                    MINIJ=SIJ
                    IMIN=I
                    JMIN=J
                 ELSE
                    IF(SIJ.LT.MINIJ)THEN
                       MINIJ=SIJ
                       IMIN=I
                       JMIN=J
                    ENDIF
                 ENDIF
              ENDIF
           ENDDO
#if KEY_PNOE==1
        endif
#endif 
     ENDDO
     !
     IF(NOEMIN(N))THEN
        RIJ=SQRT(MINIJ)
     ELSE
        !JH
        if(NOERAM(N).EQ.0) ATOT=ATOT/NIJ
        RIJ=ATOT**NOEEXP(N)
     ENDIF
     !
     ! Compute energies
     !
     DF=0.0
     EIJ=0.0
     DDF=0.0
     !
     ! compute contributions for distances too close.
     IF(RIJ.LT.NOERMN(N)) THEN
        DEL=RIJ-NOERMN(N)
        DF=NOESCA*NOEKMN(N)*DEL
        EIJ=0.5*NOESCA*NOEKMN(N)*DEL*DEL
        DDF=NOESCA*NOEKMN(N)
     ENDIF
     !
     ! Average RIJ if requested
     RIJU=RIJ
     IF(NOETCN(N).GT.ZERO) THEN
        DEL=DELTA
        IF(DEL.LE.ZERO) DEL=PT001
        NOEAVE(N)=(NOEAVE(N)*(NOETCN(N)-DEL)+DEL/RIJ**3)/NOETCN(N)
        RIJU=NOEAVE(N)**(-THIRD)
     ENDIF
     !
     ! compute contributions for interactions that are too far
     if(NOERSW(N).GT.0) then     ! new NOE potential with soft asymptote

        IF(RIJU.GE.NOERMX(N).AND.RIJU.LE.NOERMX(N)+NOERSW(N)) THEN
           DEL=RIJU-NOERMX(N)
           DF=NOESCA*NOEKMX(N)*DEL
           EIJ=0.5*NOESCA*NOEKMX(N)*DEL*DEL
           DDF=NOESCA*NOEKMX(N)
        ENDIF
        IF(RIJU.GT.NOERMX(N)+NOERSW(N)) THEN
           SOFTB=(NOEFMX(N)-NOERSW(N))/NOESEX(N)* &
                NOERSW(N)**(NOESEX(N)+1)
           SOFTA=0.5*NOERSW(N)**2-NOEFMX(N)*NOERSW(N)- &
                NOERSW(N)*(NOEFMX(N)-NOERSW(N))/NOESEX(N)
           DEL=RIJU-NOERMX(N)
           DF=NOESCA*NOEKMX(N)* &
                (NOEFMX(N)-SOFTB/DEL**(NOESEX(N)+1))
           EIJ=NOESCA*NOEKMX(N)* &
                (SOFTA+SOFTB/DEL**NOESEX(N)+NOEFMX(N)*DEL)
           DDF=NOESCA*NOEKMX(N)* &
                (NOESEX(N)+1)*SOFTB/DEL**(NOESEX(N)+2)
        ENDIF

     else                   ! old NOE

        IF(RIJU.GE.NOERMX(N)) THEN
           DEL=RIJU-NOERMX(N)
           DF=DF+NOESCA*NOEKMX(N)*DEL
           EIJ=EIJ+0.5*NOESCA*NOEKMX(N)*DEL*DEL
           DDF=DDF+NOESCA*NOEKMX(N)
        ENDIF
        !
        ! correct far interactions for maximum force allowed
        IF(NOEKMX(N).GT.0.0) THEN
           RLIM=NOERMX(N)+NOEFMX(N)/NOEKMX(N)
        ELSE
           RLIM=9999.0
        ENDIF
        IF(RIJU.GT.RLIM) THEN
           DEL=RIJU-RLIM
           DF=DF-NOESCA*NOEKMX(N)*DEL
           EIJ=EIJ-0.5*NOESCA*NOEKMX(N)*DEL*DEL
           DDF=DDF-NOESCA*NOEKMX(N)
        ENDIF

     endif                  ! if(NOERSW(N).GT.0)
     !
     IF(NOEMIN(N))THEN
        DFA=DF
     ELSE
        !JH         DFA=DF*RIJ/(ATOT*NIJ)
        DFA=DF*RIJ/ATOT
        IF(NOERAM(N).EQ.0) DFA=DFA/NIJ !!! otherwise summation
     ENDIF
     !
     AEXP=HALF/NOEEXP(N)-ONE
     DO II=1,NOEINM(N)
        I=NOELIS(NOEIPT(N)+II-1)
        IF(.NOT.NOEMIN(N).OR.(NOEMIN(N).AND.(I.EQ.IMIN)))THEN
#if KEY_PNOE==1
           if(IsPNOE(N)) then
              rx=x(i)-c0x(N)
              ry=y(i)-c0y(N)
              rz=z(i)-c0z(N)
              SIJ=RX*RX+RY*RY+RZ*RZ
              A=SIJ**AEXP
              !
              XIJ=A*RX*DFA
              YIJ=A*RY*DFA
              ZIJ=A*RZ*DFA
              ! accumulate total derivatives
              DX(I)=DX(I)+XIJ
              DY(I)=DY(I)+YIJ
              DZ(I)=DZ(I)+ZIJ
           else
#endif 
              DO JJ=1,NOEJNM(N)
                 J=NOELIS(NOEJPT(N)+JJ-1)
                 IF(.NOT.NOEMIN(N).OR.(NOEMIN(N).AND.(J.EQ.JMIN)))THEN
                    RX=X(I)-X(J)
                    RY=Y(I)-Y(J)
                    RZ=Z(I)-Z(J)
                    SIJ=RX*RX+RY*RY+RZ*RZ
#if KEY_FOURD==1
                    IF(DIM4) THEN
                       RFD=FDIM(I)-FDIM(J)
                       SIJ=SIJ+RFD*RFD
                    ENDIF
#endif 
                    A=SIJ**AEXP
                    !
                    XIJ=A*RX*DFA
                    YIJ=A*RY*DFA
                    ZIJ=A*RZ*DFA
                    ! accumulate total derivatives
                    DX(I)=DX(I)+XIJ
                    DX(J)=DX(J)-XIJ
                    DY(I)=DY(I)+YIJ
                    DY(J)=DY(J)-YIJ
                    DZ(I)=DZ(I)+ZIJ
                    DZ(J)=DZ(J)-ZIJ
#if KEY_FOURD==1
                    IF(DIM4) THEN
                       FDIJ=A*RFD*DFA
                       DFDIM(I)=DFDIM(I)+FDIJ
                       DFDIM(J)=DFDIM(J)-FDIJ
                    ENDIF
#endif 
                 ENDIF !NOEMIN
              ENDDO
#if KEY_PNOE==1
           endif
#endif 
        ENDIF !NOEMIN
     ENDDO
     !
     EN=EN+EIJ
     !
     IF(QECONT) THEN
#if KEY_PNOE==1
        !           use the fact that I-list and J-list are the same
        !           if IsPNOE(N) --> nothing needs changed, as the same
        !           atoms will be assigned 0.5*EIJ twice:
#endif 
        EIJ=EIJ*HALF
        DO II=1,NOEINM(N)
           I=NOELIS(NOEIPT(N)+II-1)
           ECONT(I)=ECONT(I)+EIJ/NOEINM(N)
        ENDDO
        DO JJ=1,NOEJNM(N)
           J=NOELIS(NOEJPT(N)+JJ-1)
           ECONT(J)=ECONT(J)+EIJ/NOEJNM(N)
        ENDDO
     ENDIF
     !
     ! second derivative part
     IF(QSECD) THEN
        IF(NIJ.NE.1) THEN
           NWARN=NWARN+1
#if KEY_PNOE==1
        ELSEIF(IsPNOE(N)) THEN
           NWARN2=NWARN2+1
#endif 
        ELSE
           !
           DF=DF/RIJ
           DDF=(DDF-DF)/SIJ
           !
#if KEY_DIMB==1 /*dimbcmp*/
           IF(QCMPCT) THEN
              CALL NOECMP(I,J,RX,RY,RZ,DF,DDF,DD1, &
                   PINBCM,PJNBCM)
           ELSE
#endif /* (dimbcmp)*/
              !
              IF(J.LT.I) THEN
                 JJ=3*I-2
                 II=3*J-2
              ELSE
                 JJ=3*J-2
                 II=3*I-2
              ENDIF
              !
              A=RX*RX*DDF+DF
              IADD=IUPT(II)+JJ
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II)+II
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ)+JJ
              DD1(IADD)=DD1(IADD)+A
              !
              A=RY*RY*DDF+DF
              IADD=IUPT(II+1)+JJ+1
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+1)+II+1
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ+1)+JJ+1
              DD1(IADD)=DD1(IADD)+A
              !
              A=RZ*RZ*DDF+DF
              IADD=IUPT(II+2)+JJ+2
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+2)+II+2
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ+2)+JJ+2
              DD1(IADD)=DD1(IADD)+A
              !
              A=RX*RY*DDF
              IADD=IUPT(II)+JJ+1
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+1)+JJ
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II)+II+1
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ)+JJ+1
              DD1(IADD)=DD1(IADD)+A
              !
              A=RX*RZ*DDF
              IADD=IUPT(II)+JJ+2
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+2)+JJ
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II)+II+2
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ)+JJ+2
              DD1(IADD)=DD1(IADD)+A
              !
              A=RY*RZ*DDF
              IADD=IUPT(II+1)+JJ+2
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+2)+JJ+1
              DD1(IADD)=DD1(IADD)-A
              IADD=IUPT(II+1)+II+2
              DD1(IADD)=DD1(IADD)+A
              IADD=IUPT(JJ+1)+JJ+2
              DD1(IADD)=DD1(IADD)+A
#if KEY_DIMB==1 /*dimbendif*/
           ENDIF
#endif /* (dimbendif)*/
        ENDIF
     ENDIF
     !
  ENDDO
  !
#if KEY_PNOE==1
  IF(MOVED) IMPNOE=IMPNOE+1
  IF(NWARN2.GT.0) CALL WRNDIE(-2,'<NOECNS>', &
       'There is no Hessian code for PNOE restraints.')
#endif 
  IF(NWARN.GT.0) CALL WRNDIE(-2,'<NOECNS>', &
       'There is no Hessian code for multiple atom NOE restraints.')
  !
  RETURN
END SUBROUTINE NOECNS

SUBROUTINE NOEREA_OLD(IUNIT)
  !
  !     THIS ROUTINE READS AN NOE CARD FILE
  !
  !     By Bernard R. Brooks    1987
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
#if KEY_PNOE==1
  use number     
#endif
  use ctitla
  use noem
  use psf
  use stream
  use string
  use chutil,only:getatn
  use parallel,only: psnd4,psnd8

  implicit none
  !
  INTEGER IUNIT
  !
  !
  INTEGER I,J,N,IPT
  CHARACTER(len=8) AII,AJJ,AIS,AIR,AJS,AJR
  integer llen
  character(len=80) line
  !
  IF(IUNIT.NE.OUTU .AND. PRNLEV.GE.2) WRITE(OUTU,822) IUNIT
822 FORMAT(' NOE RESTRAINTS READ FROM UNIT',I3)
  !
  IF(IOLEV.GT.0) THEN
     CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
     READ(IUNIT,28) NOENUM,NOESCA
28   FORMAT(I5,F10.4)
     READ(IUNIT,29) AII
29   FORMAT(A4)
     IF(PRNLEV.GE.2) WRITE(OUTU,31) AII
31   FORMAT('  LINE LABEL=',A4,'"')
     READ(IUNIT,*)
     READ(IUNIT,*)
     READ(IUNIT,*)
     !
     IPT=1
     DO N=1,NOENUM
        !
        !  Modified by SRD 1/27/91.
        !
        NOEIPT(N)=IPT
        IPT=IPT+1
        NOEJPT(N)=IPT
        IPT=IPT+1
        !           READ(IUNIT,
        !     &           '(5X,2X,I5,1X,A4,3X,A4,1X,A4,5X,I5,1X,A4,3X,A4,1X,A4)',
        !     &           ERR=24,END=24) I,AIS,AIR,AII,J,AJS,AJR,AJJ
        read(iunit,'(7x,a)',err=24,end=24) line
        llen=len(line)
        i=nexti(line,llen)
        ais=nexta8(line,llen)
        air=nexta8(line,llen)
        aii=nexta8(line,llen)
        j=nexti(line,llen)
        ajs=nexta8(line,llen)
        ajr=nexta8(line,llen)
        ajj=nexta8(line,llen)
        READ(IUNIT, &
             '(7X,F10.0,2X,F10.0,5X,F10.0,2X,F10.0,2X,F10.0,2X,F10.0)', &
             ERR=24,END=24) NOERMN(N),NOEKMN(N), &
             NOERMX(N),NOEKMX(N),NOEFMX(N),NOETCN(N)
        NOEEXP(N)=1 ! substitute default value
        NOEINM(N)=1
        NOEJNM(N)=1
        NOEAVE(N)=NOERMX(N)
        READ(IUNIT,*)
        READ(IUNIT,*)
        NOELIS(NOEIPT(N))=GETATN(AIS,AIR,AII,SEGID,RESID,ATYPE,IBASE, &
             NICTOT,NSEGT)
        NOELIS(NOEJPT(N))=GETATN(AJS,AJR,AJJ,SEGID,RESID,ATYPE,IBASE, &
             NICTOT,NSEGT)
#if KEY_PNOE==1
        IsPNOE(N)=.FALSE.
        MVPNOE(N)=.FALSE.
        C0X(N)=ANUM
        C0Y(N)=ANUM
        C0Z(N)=ANUM
#endif 
     ENDDO
  ENDIF
  !
#if KEY_PARALLEL==1
  CALL PSND4(NOENUM,1)
  CALL PSND4(NOENM2,1)
#if KEY_PNOE==1
  CALL PSND4(NMPNOE,1)  
  CALL PSND4(IMPNOE,1)  
#endif
  CALL PSND8(NOESCA,1)
  IF(NOENUM.GT.0) THEN
     CALL PSND4(NOEINM,NOENUM)
     CALL PSND4(NOEIPT,NOENUM)
     CALL PSND4(NOEJNM,NOENUM)
     CALL PSND4(NOEJPT,NOENUM)
     CALL PSND4(NOELIS,NOENM2)
#if KEY_PNOE==1
     CALL PSND4(IsPNOE,NOENUM)  
     CALL PSND4(MVPNOE,NOENUM)  
#endif
     CALL PSND8(NOERMN,NOENUM)
     CALL PSND8(NOEKMN,NOENUM)
     CALL PSND8(NOERMX,NOENUM)
     CALL PSND8(NOEKMX,NOENUM)
     CALL PSND8(NOEFMX,NOENUM)
     CALL PSND8(NOETCN,NOENUM)
     CALL PSND8(NOEEXP,NOENUM)
     CALL PSND8(NOEAVE,NOENUM)
  ENDIF
#endif 
  !
  RETURN
  !
24 CONTINUE
  CALL WRNDIE(0,'<NOEREA>','INPUT ERROR OR EOF ENCOUNTERED')
  NOENUM=0
  NOENM2=0
  NOESCA=1.0
#if KEY_PARALLEL==1
  CALL PSND4(NOENUM,1)
  CALL PSND4(NOENM2,1)
  CALL PSND8(NOESCA,1)
#endif 
  RETURN
END SUBROUTINE NOEREA_OLD

#endif /* KEY_NOMISC */
