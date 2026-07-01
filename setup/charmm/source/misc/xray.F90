#if KEY_NOMISC==0
SUBROUTINE RDXRAY(IUNIT,COMLYN,COMLEN)
  !
  !   This routine reads a card file compatable with
  !   Richard Feldmann's XRAY display program
  !
  !   Note:: This routine replaces the PSF and COORD common blocks
  !
  ! SYNTAX:
  !   READ XRAY [UNIT int]
  !
  !      By Bernard R. Brooks    March-1988
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use string
  use psf
  use coord
  use ctitla
  use parallel,only: psnd4
#ifdef KEY_RESIZE
  use resize
#endif  
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  INTEGER IUNIT
  !
  INTEGER IMOST,BOXSIZ,DUMMY,I,J,ILAST,NAT,IX,IAT,NB
  CHARACTER(len=8) RESTYP,ATTYPE
  LOGICAL QPSF
  !
  INTEGER LEN
  INTEGER, PARAMETER :: MAXLEN=100
  CHARACTER(len=MAXLEN) LINE
#ifdef KEY_RESIZE
  integer nr0,ng0,nb0,ns0
  nr0=nres; ng0=ngrp; nb0=nbond; ns0=nseg
#endif  
  !
  !C      QPSF=(INDXA(COMLYN,COMLEN,'PSF').GT.0)
  QPSF=.TRUE.
  !
  NTITLB=2
  IF(IOLEV.GT.0) THEN
     READ(IUNIT,35) TITLEB(1)
     READ(IUNIT,35) TITLEB(2)
35   FORMAT(A)
     CALL WRTITL(TITLEB,NTITLB,OUTU,1)
  ENDIF
  !
  IF(QPSF) THEN
     IF(IOLEV.GT.0) THEN
        READ(IUNIT,45) NATOM,IMOST
        READ(IUNIT,45) BOXSIZ
45      FORMAT(4I5)
     ENDIF
#if KEY_PARALLEL==1
     CALL PSND4(NATOM,1)
#endif 
     !
     NRES=0
     NSEG=0
     NBOND=0
     NTHETA=0
     NPHI=0
     NIMPHI=0
#if KEY_CMAP==1
     NCRTERM=0
#endif 
     NNB=0
     NDON=0
     NACC=0
     NGRP=0
     NST2=0
     ILAST=0
     !
     DO IAT=1,NATOM
        IF(IOLEV.GT.0) READ(IUNIT,55) ATYPE(IAT),RESTYP,LINE
55      FORMAT(3A)
#if KEY_PARALLEL==1
        CALL PSNDC(ATYPE(IAT),1)
        CALL PSNDC(RESTYP,1)
        CALL PSNDC(LINE,1)
#endif 
        AMASS(IAT)=1.0
        CG(IAT)=0.0
        IAC(IAT)=1
        IBLO(IAT)=0
        IMOVE(I)=0
        RSCLF(I)=1.0
#if KEY_WCA==1
        WCA(I)=1.0           
#endif
        !
        LEN=MAXLEN
        CALL TRIME(LINE,LEN)
        DO I=1,LEN
           IF(LINE(I:I).EQ.',') LINE(I:I)=' '
        ENDDO
        I=NEXTI(LINE,LEN)
        !
        I=NEXTI(LINE,LEN)
        IF(I.GT.NRES) THEN
           NRES=I
           NGRP=I
#ifdef KEY_RESIZE
           if (nres .gt. nr0) then
              nr0=nres+drsz0
              call resize_psf('xray.F90','RDXRAY','NRES',nr0, .true.)
           endif
           if (ngrp .gt. ng0) then
              ng0=ngrp+drsz0
              call resize_psf('xray.F90','RDXRAY','NGRP',ng0, .true.)
           endif
#endif    
           IMOVEG(I)=0
           IGPBS(I)=IAT-1
           IGPTYP(I)=0
           IBASE(I)=IAT-1
           RES(I)=RESTYP
           CALL ENCODI(I-ILAST,RESID(I),8,DUMMY)
        ENDIF
        !
        X(IAT)=NEXTF(LINE,LEN)
        Y(IAT)=NEXTF(LINE,LEN)
        Z(IAT)=NEXTF(LINE,LEN)
        !
        NB=NEXTI(LINE,LEN)
        DO IX=1,NB
           J=NEXTI(LINE,LEN)
           IF(J.GT.I) THEN
              NBOND=NBOND+1
#ifdef KEY_RESIZE
              if (nbond .gt. nb0) then
                 nb0=nbond+drsz0
                 call resize_psf('xray.F90','RDXRAY','NBOND',nb0, .true.)
              endif
#endif              
              IB(NBOND)=IAT
              JB(NBOND)=J
           ENDIF
        ENDDO
        !
        NB=NEXTI(LINE,LEN)
        DO IX=1,NB
           J=NEXTI(LINE,LEN)
        ENDDO
        !
        I=NEXTI(LINE,LEN)
        IF(I.GT.NSEG) THEN
           NSEG=I
#ifdef KEY_RESIZE
           if (nseg .gt. ns0) then
              ns0=nseg+drsz0
              call resize_psf('xray.F90','RDXRAY','NSEG',ns0, .true.)
           endif
#endif    
           NICTOT(I)=NRES-1
           CALL ENCODI(I,SEGID(I),8,DUMMY)
        ENDIF
        !
        WMAIN(IAT)=NEXTF(LINE,LEN)
        !
     ENDDO
#ifdef KEY_RESIZE
     if (nres .lt. nr0) &
          call resize_psf('xray.F90','RDXRAY','NRES',nres, .true.)
     if (ngrp .lt. ng0) &
          call resize_psf('xray.F90','RDXRAY','NGRP',ngrp, .true.)
     if (nbond .lt. nb0) &
          call resize_psf('xray.F90','RDXRAY','NBOND',nbond, .true.)
     if (nseg .lt. ns0) &
          call resize_psf('xray.F90','RDXRAY','NSEG',nseg, .true.)
#endif
     IBASE(NRES+1)=NATOM
     IGPBS(NGRP+1)=NATOM
     NICTOT(NSEG+1)=NRES
     !
     NREST=NRES
     NATOMT=NATOM
     NBONDT=NBOND
     NBONDT2=NBOND ! hwm: per BRB's suggestion on 220722
     NSEGT=NSEG
     NTHETT=NTHETA
     NPHIT=NPHI
     NIMPHT=NIMPHI
#if KEY_CMAP==1
     NCRTT=NCRTERM
#endif 
     NGRPT=NGRP
     !===========================================================
  ELSE
     IF(IOLEV.GT.0) THEN
        READ(IUNIT,45) NAT,IMOST
        READ(IUNIT,45) BOXSIZ
        !
        IF(NAT.NE.NATOM) THEN
           CALL WRNDIE(0,'<RDXRAY>','NUMBER OF ATOMS DONT MATCH')
           RETURN
        ENDIF
     ENDIF
     !
     DO IAT=1,NAT
        IF(IOLEV.GT.0) READ(IUNIT,55) ATTYPE,RESTYP,LINE
#if KEY_PARALLEL==1
        CALL PSNDC(ATTYPE,1)
        CALL PSNDC(LINE,1)
#endif 
        IF (ATTYPE /= ATYPE(IAT)) THEN
           CALL WRNDIE(0,'<RDXRAY>','ATOMS DONT MATCH')
           RETURN
        ENDIF
        !
        LEN=MAXLEN
        CALL TRIME(LINE,LEN)
        DO I=1,LEN
           IF(LINE(I:I).EQ.',') LINE(I:I)=' '
        ENDDO
        I=NEXTI(LINE,LEN)
        I=NEXTI(LINE,LEN)
        !
        X(IAT)=NEXTF(LINE,LEN)
        Y(IAT)=NEXTF(LINE,LEN)
        Z(IAT)=NEXTF(LINE,LEN)
        !
        NB=NEXTI(LINE,LEN)
        DO IX=1,NB
           J=NEXTI(LINE,LEN)
        ENDDO
        NB=NEXTI(LINE,LEN)
        DO IX=1,NB
           J=NEXTI(LINE,LEN)
        ENDDO
        !
        I=NEXTI(LINE,LEN)
        WMAIN(IAT)=NEXTF(LINE,LEN)
        !
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE RDXRAY

SUBROUTINE WRXRAY(IUNIT,COMLYN,COMLEN,X,Y,Z,WMAIN)
  !
  !   This routine generates a card file compatable with
  !   Richard Feldmann's XRAY display program
  !
  ! SYNTAX:
  !   WRITe XRAY [UNIT int] [COMP] [BOXSise integer] [atom-selection]
  !
  !      By Bernard R. Brooks    30-APR-1985
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use ctitla
  use select
  use stream
  use string
  use psf
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  INTEGER IUNIT
  !
  real(chm_real),allocatable,dimension(:) :: ATSIZE
  integer,allocatable,dimension(:) :: ISLCT
  integer,allocatable,dimension(:) :: NATBON
  integer,allocatable,dimension(:) :: IATBON
  !
  INTEGER BOXSIZ
  INTEGER IATMXB,NATIML
  LOGICAL LRESID,ERROR
  !
  IF(NATOM.LE.0) RETURN
  CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
  CALL WRTITL(TITLEA,NTITLA,0,2)
  !
  IF(OUTU.EQ.IUNIT .AND. IOLEV.LT.0) RETURN
  IF(OUTU.NE.IUNIT .AND. PRNLEV.LE.2) RETURN
  !
  NATIML=NATOM
  call chmalloc('xray.src','WRXRAY','ATSIZE',NATIML,crl=ATSIZE)
  call chmalloc('xray.src','WRXRAY','ISLCT',NATIML,intg=ISLCT)
  call chmalloc('xray.src','WRXRAY','NATBON',NATIML,intg=NATBON)
  IATMXB=8
  IF(NATIML.LT.200) IATMXB=20
  call chmalloc('xray.src','WRXRAY','IATBON',IATMXB*NATIML,intg=IATBON)
  call SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  BOXSIZ=GTRMI(COMLYN,COMLEN,'BOXS',32)
  LRESID=(INDXA(COMLYN,COMLEN,'RESI').GT.0)
  !
  call XRAY2(IUNIT,NATBON,IATBON,IATMXB,X,Y,Z,ISLCT,ATSIZE, &
       BOXSIZ,LRESID)
  !
  call chmdealloc('xray.src','WRXRAY','ATSIZE',NATIML,crl=ATSIZE)
  call chmdealloc('xray.src','WRXRAY','ISLCT',NATIML,intg=ISLCT)
  call chmdealloc('xray.src','WRXRAY','NATBON',NATIML,intg=NATBON)
  call chmdealloc('xray.src','WRXRAY','IATBON',IATMXB*NATIML,intg=IATBON)
  !
  CALL VCLOSE(IUNIT,'KEEP',ERROR)
  !
  RETURN
END SUBROUTINE WRXRAY

SUBROUTINE XRAY2(IUNIT,NATBON,IATBON,IATMXB, &
     X,Y,Z,ISLCT,ATSIZE,BOXSIZ,LRESID)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use psf
  use param
  use ctitla
  use stream
  use string
  use chutil,only:getres,getseg
  !
  implicit none
  !
  INTEGER IUNIT,IATMXB,BOXSIZ
  INTEGER NATBON(*),IATBON(IATMXB,*)
  real(chm_real) X(*),Y(*),Z(*)
  INTEGER ISLCT(*)
  real(chm_real) ATSIZE(*)
  LOGICAL LRESID
  !
  CHARACTER(len=128) LINE
  INTEGER LINELN
  CHARACTER(len=30) FMT
  real(chm_real) XR,YR,ZR
  INTEGER NSLCT,ISEG,IRES
  INTEGER IBT,JBT,I,J,IMOST,IPT,ILEN
  INTEGER II
  LOGICAL LASTB
  CHARACTER(len=8) RESO,RESX
  CHARACTER(len=1) TEMP
  !
  !     Next make a list of all the possible 1-3 and 1-4 interactions.
  !     This is based on the bond list.  First make a list of all bonds
  !     for every atom.
  !
  NSLCT=0
  DO I=1,NATOM
     NATBON(I)=0
     IF(ISLCT(I).GT.0) THEN
        NSLCT=NSLCT+1
        ISLCT(I)=NSLCT
        !C            ATSIZE(I)=VDWR(ITC(IAC(I)))
        TEMP=ATYPE(I)
        IF(TEMP.EQ.'H') THEN
           ATSIZE(I)=1.0
        ELSE IF(TEMP.EQ.'C') THEN
           ATSIZE(I)=1.6
        ELSE IF(TEMP.EQ.'N') THEN
           ATSIZE(I)=1.7
        ELSE IF(TEMP.EQ.'O') THEN
           ATSIZE(I)=1.35
        ELSE IF(TEMP.EQ.'F') THEN
           ATSIZE(I)=1.05
        ELSE
           ATSIZE(I)=1.9
        ENDIF
     ENDIF
  ENDDO
  !
  IMOST=4
  DO I=1,NBOND
     IBT=IB(I)
     JBT=JB(I)
     IF(IBT.LE.0 .OR. JBT.LE.0) THEN
        CONTINUE
     ELSE IF(ISLCT(IBT).LE.0 .OR. ISLCT(JBT).LE.0) THEN
        CONTINUE
     ELSE
        NATBON(IBT)=NATBON(IBT)+1
        IF(NATBON(IBT).GT.IMOST) IMOST=NATBON(IBT)
        IF(NATBON(IBT).GT.IATMXB) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,335) IBT
335        FORMAT(' <XRAY2>: Too many bonds for atom',I5, &
                ' Check code')
           CALL DIEWRN(-4)
        ENDIF
        IATBON(NATBON(IBT),IBT)=ISLCT(JBT)
        NATBON(JBT)=NATBON(JBT)+1
        IF(NATBON(JBT).GT.IMOST) IMOST=NATBON(JBT)
        IF(NATBON(JBT).GT.IATMXB) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,335) JBT
           CALL DIEWRN(-4)
        ENDIF
        IATBON(NATBON(JBT),JBT)=ISLCT(IBT)
     ENDIF
  ENDDO
  !
  WRITE(IUNIT,35) TITLEA(1)
  WRITE(IUNIT,35) TITLEA(NTITLA)
35 FORMAT(A)
  WRITE(IUNIT,45) NSLCT,IMOST,1
  WRITE(IUNIT,45) BOXSIZ
45 FORMAT(4I5)
  !
  !          123456789+123456789+123456789+
  FMT='(2A,2I5,3F12.6,QQI5,F12.6)'
  !
  DO I=1,NATOM
     IF(ISLCT(I).GT.0) THEN
        II=I
        IRES=GETRES(II,IBASE,NRES)
        ISEG=GETSEG(IRES,NICTOT,NSEG)
        XR=X(I)
        YR=Y(I)
        ZR=Z(I)
        WRITE(FMT(16:17),'(I2)') 3+NATBON(I)
        RESO=RES(IRES)
        !
        IF(LRESID) THEN
           ILEN=8
           CALL SPLITI(RESID(IRES),IRES,RESX,ILEN)
           IF(ILEN.GT.0) RESO(idleng:idleng)=RESX(1:1)
        ENDIF
        !
        WRITE(LINE,FMT) ATYPE(I)(1:idleng),RESO(1:idleng), &
             ISLCT(I),IRES,XR,YR,ZR, &
             NATBON(I),(IATBON(J,I),J=1,NATBON(I)),0, &
             ISEG,ATSIZE(I)
        !
        LINELN=LEN(LINE)
        DO WHILE(LINE(LINELN:LINELN).EQ.' ')
           LINELN=LINELN-1
        ENDDO
        LASTB=.FALSE.
        IPT=8
        DO J=9,LINELN
           IF(LINE(J:J).EQ.' ') THEN
              IF(LASTB) THEN
                 IPT=IPT+1
                 LINE(IPT:IPT)=','
                 LASTB=.FALSE.
              ENDIF
           ELSE
              IPT=IPT+1
              TEMP=LINE(J:J)
              LINE(IPT:IPT)=TEMP
              LASTB=.TRUE.
           ENDIF
        ENDDO
        WRITE(IUNIT,65) LINE(1:IPT)
65      FORMAT(A)
        !
     ENDIF
  ENDDO
  !
END SUBROUTINE XRAY2
#else /**/
SUBROUTINE NULL_XR
  RETURN
END SUBROUTINE NULL_XR
#endif 

