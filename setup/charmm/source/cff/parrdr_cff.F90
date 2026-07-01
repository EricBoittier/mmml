#if KEY_CFF==1
SUBROUTINE PARRDR_CFF(IUNIT,ICARD,MLEN)
  !
  use chm_kinds
  !
  use consta
  use dimens_fcm
  use energym
  use exfunc
  use number
  use param
  use stream
  use psf
  use rtf,only:armass, reallocate_natct
  use cff_fcm
  !
  implicit none
  INTEGER IUNIT,ICARD,MLEN
  !     *** local variables ***
  INTEGER ::NBC(MLEN),IOFF(MLEN),IDX(MLEN),WORK(MLEN*2),ITMP(MLEN)
  INTEGER I,IAT,NBCX,J,NBEXP,I1,J1,K1,L1,NNDX
  INTEGER NOIJ,NOKL,PASS
  character(len=4) ACPTYP(10),HYDTYP(10),ch1,ch2,ch3,ch4
  character(len=12) COMBIN,NBTYPE
  integer(chm_int8) :: natc2,j8

  logical orderi8
  external orderi8
  !
  J=1
  DO I=1,MAXATC
     NBC(I)=0
     IOFF(I)=J
     J=J+I+1
  ENDDO

  INNB = 0 ! initialize cff_fcm%innb before filling and using

  IF (ICARD == 0) THEN
     READ(IUNIT)NATC,NCB,NCBO,NCT,NCTO,NCP,NCPO,NCI,NCAA,NTTP,NNONBP, &
          NNBDP,FORCE,FRCVER
     call reallocate_natct(natc)
! BIOVIA Code Start
#if KEY_LICENSE==1
        IF (NATC.LT.0) THEN
          NATC=-NATC
          CALL checkout_license_cff()
          DECODE=.TRUE.
        ELSE
          DECODE=.FALSE.
        ENDIF
#endif
! BIOVIA Code End
     READ(IUNIT)(ATC(IAT),IAT=1,NATC),(INNB(IAT),IAT=1,NATC)
     READ(IUNIT)(IBE(IAT),IAT=1,NATC),(ITE(IAT),IAT=1,NATC)
     READ(IUNIT)(IPE(IAT),IAT=1,NATC),(IOE(IAT),IAT=1,NATC)
     READ(IUNIT)(IBAE(IAT),IAT=1,NATC),(INBAE(IAT),IAT=1,NATC)
     READ(IUNIT)(ITAAE(IAT),IAT=1,NATC),(ITEAE(IAT),IAT=1,NATC)
     READ(IUNIT)(IPCAE(IAT),IAT=1,NATC),(IPEAE(IAT),IAT=1,NATC)
     READ(IUNIT)(IOCAE(IAT),IAT=1,NATC),(IOEAE(IAT),IAT=1,NATC)
     READ(IUNIT)(ARMASS(IAT),IAT=1,NATC),(ITC(I),I=1,NATC)
     IF (NCBO > 0) READ(IUNIT)(BID1(J),J=1,NCBO), &
          (BID2(J),J=1,NCBO),(CBOND1(I),I=1,NCBO), &
          (CBOND2(I),I=1,NCBO),(CBOND3(I),I=1,NCBO), &
          (CBOND4(I),I=1,NCBO),(KCB(I),I=1,NCB)
     IF (NCTO > 0) READ(IUNIT)(TID1(J),J=1,NCTO), &
          (TID2(J),J=1,NCTO),(TID3(J),J=1,NCTO), &
          (CTHET1(I),I=1,NCTO),(CTHET2(I),I=1,NCTO), &
          (CTHET3(I),I=1,NCTO),(CTHET4(I),I=1,NCTO), &
          (CTHET5(I),I=1,NCTO),(CTHET6(I),I=1,NCTO), &
          (CTHET7(I),I=1,NCTO),(KCT(I),I=1,NCT)
     IF (NCPO > 0) THEN
        READ(IUNIT)(PID1(J),J=1,NCPO),(PID2(J),J=1,NCPO), &
             (PID3(J),J=1,NCPO),(PID4(J),J=1,NCPO),(CPD(J),J=1,NCPO)
        READ(IUNIT)(CPHI1(I),I=1,NCPO), &
             (CPHI2(I),I=1,NCPO),(CPHI3(I),I=1,NCPO),(CPHI4(I),I=1,NCPO), &
             (CBP11(I),I=1,NCPO),(CBP12(I),I=1,NCPO), &
             (CBP13(I),I=1,NCPO),(CBP21(I),I=1,NCPO),(CBP22(I),I=1,NCPO), &
             (CBP23(I),I=1,NCPO),(CBP31(I),I=1,NCPO),(CBP32(I),I=1,NCPO), &
             (CBP33(I),I=1,NCPO),(CTP11(I),I=1,NCPO),(CTP12(I),I=1,NCPO), &
             (CTP13(I),I=1,NCPO),(CTP21(I),I=1,NCPO),(CTP22(I),I=1,NCPO), &
             (CTP23(I),I=1,NCPO),(CSGN1(I),I=1,NCPO),(CSGN2(I),I=1,NCPO), &
             (CSGN3(I),I=1,NCPO),(CBB2(I),I=1,NCPO),(KCP(I),I=1,NCP)
     ENDIF
     IF (NCI > 0) THEN
        READ(IUNIT)(OPID1(J),J=1,NCI),(OPID2(J),J=1,NCI), &
             (OPID3(J),J=1,NCI),(OPID4(J),J=1,NCI), &
             (COPLN1(I),I=1,NCI),(KCI(I),I=1,NCI)
     ENDIF
     IF (NTTP > 0) READ(IUNIT)(TTID1(J),J=1,NTTP), &
          (TTID2(J),J=1,NTTP),(TTID3(J),J=1,NTTP),(TTID4(J),J=1,NTTP), &
          (CTT(I),I=1,NTTP)
     IF (NNBDP > 0) THEN
        READ(IUNIT)(PNBID1(J),J=1,NNBDP),(PNBID2(J),J=1,NNBDP)
        READ(IUNIT)(CNB1(I),I=1,NNBDP),(CNB2(I),I=1,NNBDP), &
             (CNB1(I+MAXCN),I=1,NNBDP),(CNB2(I+MAXCN),I=1,NNBDP),(CNB5(I),I=1,NNBDP)
     ENDIF
     READ(IUNIT)((MNO(I,J),I=1,NATC),J=1,NATC)
  ELSE
     FORCE='CFF89'
     NBEXP=12
     COMBIN = 'geometric'
     NBTYPE = 'A-B'
     PASS = 1
     NBCX = 1
     IHYDNB=0
     CALL NEWPRM(IUNIT,ACPTYP,HYDTYP,COMBIN,NBTYPE,NBEXP,PASS,NNDX, &
          ITMP)
     !
     !     On the first pass through the file only the regular parameters
     !     were stored.  Now save the number of parameters then get the
     !     bond-ordered parameters.
     !
     NCB = NCBO
     NCT = NCTO
     NCP = NCPO
     NCAA = NTTP
     REWIND IUNIT
     PASS = 2
     CALL NEWPRM(IUNIT,ACPTYP,HYDTYP,COMBIN,NBTYPE,NBEXP,PASS,NNDX, &
          ITMP)
     IF (ATC(NATC) == '*   ') NATC = NATC - 1
     !
     !-----reorder the names in the potential parameter list for
     !     the bonds, angles, torsions and theta*theta such that
     !     outer atoms are in numerical kind order, i.e. order
     !     the atoms on their kinds in knd. also order the middle
     !     2 atoms in the torsions. the order of the middle 2
     !     atoms in the theta*theta interaction is important -
     !     i.e. if they are swapped this is a different parameter.
     CALL PARODR
     !
     !-----assign non-bond parameters
     CALL NBNDAS(NBCX,COMBIN,NBTYPE,NBEXP,ITMP)
     !
     CALL HBNDAS(6,ACPTYP,HYDTYP)
     !
     NATC2=(NATC*(NATC+1))/2
     !
     IF ( NCB > 0 ) THEN
        DO I=1,NCB
           I1=BID1(I)
           J1=BID2(I)
           IF (I1 > J1) THEN
              KCB(I)=IOFF(I1)+J1
           ELSE
              KCB(I)=IOFF(J1)+I1
           ENDIF
        ENDDO
     ENDIF
     IF ( NCT > 0 ) THEN
        DO I=1,NCT
           I1=TID1(I)
           J1=TID2(I)
           K1=TID3(I)
           IF (I1 < 0) I1=0
           IF (K1 < 0) K1=0
           IF (I1 > K1) THEN
              J=IOFF(I1)+K1
           ELSE IF (K1 > 0) THEN
              J=IOFF(K1)+I1
           ELSE
              J = 0
           ENDIF
           KCT(I)=NATC*J+J1
        ENDDO
     ENDIF
     IF ( NCP > 0 ) THEN
        DO I=1,NCP
           I1=PID2(I)
           J1=PID3(I)
           K1=PID1(I)
           L1=PID4(I)
           IF (K1 == 0)K1=-1
           IF (L1 == 0)L1=-1
           IF (K1 > 0.AND.L1.GT.0) THEN
              IF (I1 > J1) THEN
                 NOIJ=IOFF(I1)+J1
              ELSE
                 NOIJ=IOFF(J1)+I1
              ENDIF
              IF (K1 > L1) THEN
                 NOKL=IOFF(K1)+L1
              ELSE
                 NOKL=IOFF(L1)+K1
              ENDIF
              J8=NOIJ+NOKL*NATC2
              IF((I1-J1)*(K1-L1) < 0) J8=J8+NATC2*NATC2
           ELSEIF (K1 <= 0.AND.L1.LE.0) THEN
              IF (I1 > J1) THEN
                 J8=IOFF(I1)+J1
              ELSE
                 J8=IOFF(J1)+I1
              ENDIF
           ENDIF
           KCP(I)=J8
        ENDDO
     ENDIF
     DO I=1,NCPO
        CPD(I)=1
     ENDDO
     IF ( NCI > 0 ) THEN
        DO I=1,NCI
           I1=OPID1(I)
           J1=OPID2(I)
           K1=OPID3(I)
           L1=OPID4(I)
           IF (I1 == 0)I1=-1
           IF (K1 == 0)K1=-1
           IF (L1 == 0)L1=-1
           IF (I1 > 0.AND.J1.GT.0.AND.K1.GT.0.AND.L1.GT.0) THEN
              !           NO WILDCARDS
              IF(I1 > L1) THEN
                 NOIJ=IOFF(I1)+L1
              ELSE
                 NOIJ=IOFF(L1)+I1
              ENDIF
              IF(J1 > K1) THEN
                 NOKL=IOFF(J1)+K1
              ELSE
                 NOKL=IOFF(K1)+J1
              ENDIF
              J8=NOIJ+NOKL*NATC2
              IF((J1-K1)*(I1-L1) < 0) J8=J8+NATC2*NATC2
           ELSEIF (I1 < 0.AND.J1 > 0.AND.K1.LT.0.AND.L1.LT.0) THEN
              !             WILDCARD TYPE X - A - X - X
              J8=-(IOFF(J1)+NATC**3+NATC**2)
           ENDIF
           KCI(I)=J8
        ENDDO
     ENDIF
     DO I=1,NCTO
        CTHET2(I)=CTHET2(I)*DEGRAD
     ENDDO
     !
     !     Done reading data, now sort the code arrays by key number.
     !
     CALL SORTP(NCB,IDX,ORDER,KCB,1,0,0,0,0,0,0)
     CALL AINDX4(IDX,KCB,NCB,WORK)
     CALL AINDEX(IDX,CBOND1,NCB,WORK)
     CALL AINDEX(IDX,CBOND2,NCB,WORK)
     CALL AINDEX(IDX,CBOND3,NCB,WORK)
     CALL AINDEX(IDX,CBOND4,NCB,WORK)
     CALL SORTP(NCT,IDX,ORDER,KCT,1,0,0,0,0,0,0)
     CALL AINDX4(IDX,KCT,NCT,WORK)
     CALL AINDX4(IDX,TID1,NCT,WORK)
     CALL AINDX4(IDX,TID2,NCT,WORK)
     CALL AINDX4(IDX,TID3,NCT,WORK)
     CALL AINDEX(IDX,CTHET1,NCT,WORK)
     CALL AINDEX(IDX,CTHET2,NCT,WORK)
     CALL AINDEX(IDX,CTHET3,NCT,WORK)
     CALL AINDEX(IDX,CTHET4,NCT,WORK)
     CALL AINDEX(IDX,CTHET5,NCT,WORK)
     CALL AINDEX(IDX,CTHET6,NCT,WORK)
     CALL AINDEX(IDX,CTHET7,NCT,WORK)
     CALL SRT8P(NCP,IDX,ORDERi8,KCP,1,0,0,0,0,0,0)
     CALL AINDX8(IDX,KCP,NCP,WORK)
     CALL AINDEX(IDX,CPHI1,NCP,WORK)
     CALL AINDEX(IDX,CPHI2,NCP,WORK)
     CALL AINDEX(IDX,CPHI3,NCP,WORK)
     CALL AINDEX(IDX,CPHI4,NCP,WORK)
     CALL AINDEX(IDX,CBP11,NCP,WORK)
     CALL AINDEX(IDX,CBP12,NCP,WORK)
     CALL AINDEX(IDX,CBP13,NCP,WORK)
     CALL AINDEX(IDX,CBP21,NCP,WORK)
     CALL AINDEX(IDX,CBP22,NCP,WORK)
     CALL AINDEX(IDX,CBP23,NCP,WORK)
     CALL AINDEX(IDX,CBP31,NCP,WORK)
     CALL AINDEX(IDX,CBP32,NCP,WORK)
     CALL AINDEX(IDX,CBP33,NCP,WORK)
     CALL AINDEX(IDX,CTP11,NCP,WORK)
     CALL AINDEX(IDX,CTP12,NCP,WORK)
     CALL AINDEX(IDX,CTP13,NCP,WORK)
     CALL AINDEX(IDX,CTP21,NCP,WORK)
     CALL AINDEX(IDX,CTP22,NCP,WORK)
     CALL AINDEX(IDX,CTP23,NCP,WORK)
     CALL AINDEX(IDX,CBB2,NCP,WORK)
     CALL AINDEX(IDX,CSGN1,NCP,WORK)
     CALL AINDEX(IDX,CSGN2,NCP,WORK)
     CALL AINDEX(IDX,CSGN3,NCP,WORK)
     CALL AINDX4(IDX,CPD,NCP,WORK)
     CALL AINDX4(IDX,PID1,NCP,WORK)
     CALL AINDX4(IDX,PID2,NCP,WORK)
     CALL AINDX4(IDX,PID3,NCP,WORK)
     CALL AINDX4(IDX,PID4,NCP,WORK)
     CALL SRT8P(NCI,IDX,ORDERi8,KCI,1,0,0,0,0,0,0)
     CALL AINDX8(IDX,KCI,NCI,WORK)
     CALL AINDEX(IDX,COPLN1,NCI,WORK)
     CALL AINDX4(IDX,OPID1,NCI,WORK)
     CALL AINDX4(IDX,OPID2,NCI,WORK)
     CALL AINDX4(IDX,OPID3,NCI,WORK)
     CALL AINDX4(IDX,OPID4,NCI,WORK)
     DO I=1,NATC
        ITC(I)=I
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE PARRDR_CFF


SUBROUTINE NEWPRM(IUNIT,ACPTYP,HYDTYP,COMBIN,NBTYPE,NBEXP,PASS, &
     NNDX,ITMP)
  !
  use chm_kinds
  !
  use consta
  use chm_kinds
  use dimens_fcm
  use memory
  use number
  use param
  use psf
  use rtf,only:armass, reallocate_natct
  use cff_fcm
  use stream
  implicit none
  !
  INTEGER IAT1,IAT2,IAT3,IAT4,IAT5,IAT6,IAT7,IAT8,IAT9,IAT10,LLEN, &
       ILOC,LASTCH,INDX,I,J,IUNIT,IA,MXORDR,LAST,N,NBEXP,PASS,NNDX, &
       ITMP(*)
  real(chm_real) A,BX,C,D,E,F
  real(chm_real) :: MAXVER,VER,DISVER,ANGVER,DONVER,ACPVER
  ! clb3 made alllocatable
  !real(chm_real) :: ATVER(MAXATC),EQVER(MAXATC),QBVER(MAXCB),QAVER(MAXCT), &
  !     QTVER(MAXCP),QOVER(MAXCI),QNBVER(MXNBDP), QAAVER(MXTTP),QBAVER(MAXCT),QBBVER(MAXCT),QAATVER(MAXCP), &
  !     QEBTVER(MAXCP),QMBTVER(MAXCP),QATVER(MAXCP),QBBPVER(MAXCP),AEQVER(MAXATC)
  real(chm_real), allocatable, dimension(:), save :: ATVER,EQVER,QBVER,QAVER, &
       QTVER,QOVER,QNBVER, QAAVER,QBAVER,QBBVER,QAATVER, &
       QEBTVER,QMBTVER,QATVER,QBBPVER,AEQVER
  character(len=1),parameter :: TAB = char(9)
  character(len=1) IORDER
  character(len=4) AT1,AT2,AT3,AT4,AT5,AT6,AT7,AT8,AT9,AT10,ACPTYP(10), &
       HYDTYP(10),UPCASE
  character(len=8) CFFVER
  character(len=30) TTYPE,WORD
  character(len=12) COMBIN,NBTYPE
  character(len=256) INLINE, TEMP
  LOGICAL FOUND,GOON,ERROR
  character(len=*), parameter :: file_name="parrdr_cff.src", routine_name="newprm"
 DATA MXORDR/5/

  integer readstat

  ! Allocate needed memory
  if(.not.allocated(atver)) then
     call chmalloc(file_name,routine_name,'atver',maxatc,crl=atver)
     call chmalloc(file_name,routine_name,'eqver',maxatc,crl=eqver)
     call chmalloc(file_name,routine_name,'qbver',maxcb,crl=qbver)
     call chmalloc(file_name,routine_name,'qaver',maxct,crl=qaver)
     call chmalloc(file_name,routine_name,'qtver',maxcp,crl=qtver)
     call chmalloc(file_name,routine_name,'qover',maxci,crl=qover)
     call chmalloc(file_name,routine_name,'qnbver',mxnbdp,crl=qnbver)
     call chmalloc(file_name,routine_name,'qaaver',mxttp,crl=qaaver)
     call chmalloc(file_name,routine_name,'qbaver',maxct,crl=qbaver)
     call chmalloc(file_name,routine_name,'qbbver',maxct,crl=qbbver)
     call chmalloc(file_name,routine_name,'qaatver',maxcp,crl=qaatver)
     call chmalloc(file_name,routine_name,'qebtver',maxcp,crl=qebtver)
     call chmalloc(file_name,routine_name,'qmbtver',maxcp,crl=qmbtver)
     call chmalloc(file_name,routine_name,'qatver',maxcp,crl=qatver)
     call chmalloc(file_name,routine_name,'qbbpver',maxcp,crl=qbbpver)
     call chmalloc(file_name,routine_name,'aeqver',maxatc,crl=aeqver)
  endif

  ERROR = .FALSE.
  IF (PASS == 1) THEN
     DO I = 1,MAXATC
        ATVER(I) = 0.0
        EQVER(I) = 0.0
        AEQVER(I) = 0.0
     ENDDO
     DO I = 1,MAXCB
        QBVER(I) = 0.0
        CBOND1(I) = ZERO
        CBOND2(I) = ZERO
        CBOND3(I) = ZERO
        CBOND4(I) = ZERO
     ENDDO
     DO I = 1,MAXCT
        QAVER(I) = 0.0
        QBAVER(I) = 0.0
        QBBVER(I) = 0.0
        CTHET1(I) = ZERO
        CTHET2(I) = ZERO
        CTHET3(I) = ZERO
        CTHET4(I) = ZERO
        CTHET5(I) = ZERO
        CTHET6(I) = ZERO
        CTHET7(I) = ZERO
     ENDDO
     DO I = 1,MAXCP
        QTVER(I) = 0.0
        QATVER(I) = 0.0
        QAATVER(I) = 0.0
        QEBTVER(I) = 0.0
        QMBTVER(I) = 0.0
        QBBPVER(I) = 0.0
        CPHI1(I) = ZERO
        CPHI2(I) = ZERO
        CPHI3(I) = ZERO
        CPHI4(I) = ZERO
        CBP11(I) = ZERO
        CBP12(I) = ZERO
        CBP13(I) = ZERO
        CBP21(I) = ZERO
        CBP22(I) = ZERO
        CBP23(I) = ZERO
        CBP31(I) = ZERO
        CBP32(I) = ZERO
        CBP33(I) = ZERO
        CTP11(I) = ZERO
        CTP12(I) = ZERO
        CTP13(I) = ZERO
        CTP21(I) = ZERO
        CTP22(I) = ZERO
        CTP23(I) = ZERO
        CSGN1(I) = ZERO
        CSGN2(I) = ZERO
        CSGN3(I) = ZERO
        CBB2(I) = ZERO
     ENDDO
     DO I = 1,MAXCI
        QOVER(I) = 0.0
        COPLN1(I) = ZERO
     ENDDO
     DO I = 1,MXTTP
        QAAVER(I) = 0.0
        CTT(I) = ZERO
     ENDDO
     DO I = 1,MXNBDP
        QNBVER(I) = 0.0
        CNB1(I) = ZERO
        CNB2(I) = ZERO
        CNB1(I+MAXCN) = ZERO
        CNB2(I+MAXCN) = ZERO
        CNB5(I) = ZERO
     ENDDO
     NATC=0
     NCBO=0
     NCTO=0
     NCPO=0
     NCI=0
     NNONBP=0
     NTTP=0
     NNDX=0
     IAT1=0
! BIOVIA Code Start
#if KEY_LICENSE==1
        CALL GETFIRST(IUNIT,DECODE)
#endif
! BIOVIA Code End
  ENDIF
  MAXVER = 10.0
  error = .false.
  TTYPE = ''
  loop180: do while(.true.)
     READ(IUNIT,700,iostat=readstat)inline    ! END=600)INLINE
     if(readstat /= 0) then
        error = error .or. (readstat > 0)
        exit loop180
     endif
! BIOVIA Code Start
#if KEY_LICENSE==1
     IF (DECODE) THEN
       LLEN = LASTCH(INLINE)+1
       CALL DECODE_FF(INLINE,TEMP,LLEN)
       IF (LLEN > 1) THEN
          INLINE=TEMP(1:LLEN-1)
       ELSE
          INLINE=''
       ENDIF
     ENDIF
#endif
! BIOVIA Code End
     LLEN=LASTCH(INLINE)
     IF (LLEN > 1) THEN
        IF (INLINE(1:1) /= '!') THEN
           IF (INLINE(1:1) /= '>') THEN
              DO I=1,LLEN
                 IF (INLINE(I:I) == TAB) INLINE(I:I)=' '
              ENDDO
              IF (INLINE(1:1) == '#') THEN
                 IF (TTYPE == 'equivalence') THEN
                    IF (PASS == 1) THEN
                       IAT1 = 0
                       DO I = 1,NATC
                          IF (ATC(I) == '$$') IAT1 = I
                       ENDDO
                       IF (IAT1 == 0) THEN
                          NATC=NATC + 1
                          AT1 = '$$'
                          ATC(NATC) = AT1
                          INNB(NATC) = NATC
                          IBE(NATC) = NATC
                          ITE(NATC) = NATC
                          IPE(NATC) = NATC
                          IOE(NATC) = NATC
                          DO I = 1,MXORDR
                             IORDER=CHAR(I+ICHAR('0'))
                             NATC=NATC+1
                             AT1 = '$$'//IORDER
                             ATC(NATC)=AT1
                             INNB(NATC) = NATC
                             IBE(NATC) = NATC
                             ITE(NATC) = NATC
                             IPE(NATC) = NATC
                             IOE(NATC) = NATC
                          ENDDO
                       ENDIF
                    ENDIF
                 ENDIF
                 ILOC=2
                 CALL FNDNXT(INLINE,TTYPE,ILOC,' ',FOUND)
                 IF (TTYPE == 'version') THEN
                    CALL FNDNXT(INLINE,FRCVER,ILOC,'#',FOUND)
                 ELSEIF (TTYPE == 'define') THEN
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    IF (WORD(1:3) == 'cff') THEN
                       FORCE = 'CFF89'
                    ELSEIF (WORD(1:4) == 'cvff') THEN
                       FORCE = 'CVFF'
                    ELSEIF (WORD(1:5) == 'amber') THEN
                       FORCE = 'AMBER'
                    ENDIF
                    IF(WORD(4:5) == '91') THEN
                       CFFVER = 'cff91'
                    ELSE
                       CFFVER = 'cff'
                    ENDIF
                 ENDIF
              ELSEIF (PASS == 1) THEN
                 IF (TTYPE == 'atom_types') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    IF (FORCE == 'CHARM') AT1=UPCASE(AT1)
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    IF(CFFVER == 'cff91') THEN
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,711)IAT1
                    ELSE
                       iat1 = iat1 + 1
                    ENDIF
                    call reallocate_natct(iat1)
                    IF (VER >= ATVER(IAT1) .AND. VER <= MAXVER) THEN
                       ATVER(IAT1)=VER
                       ARMASS(IAT1)=A
                       ATC(IAT1)=AT1
                       IF(IAT1 > NATC)NATC=IAT1
                    ENDIF
                 ELSEIF (TTYPE == 'equivalence') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    IF (INDEX(AT1,'$') == 0) THEN
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                       IF (VER >= EQVER(IAT1) .AND. VER <= MAXVER) THEN
                          EQVER(IAT1)=VER
                          CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                          CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                          CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                          CALL FNDNXT(INLINE,AT5,ILOC,' ',FOUND)
                          CALL FNDNXT(INLINE,AT6,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) &
                               CALL FNDNXT(INLINE,AT6,ILOC,'#',FOUND)
                          CALL FNDATM(ATC,NATC,AT2,IAT2)
                          CALL FNDATM(ATC,NATC,AT3,IAT3)
                          CALL FNDATM(ATC,NATC,AT4,IAT4)
                          CALL FNDATM(ATC,NATC,AT5,IAT5)
                          CALL FNDATM(ATC,NATC,AT6,IAT6)
                          INNB(IAT1)=IAT2
                          IBE(IAT1)=IAT3
                          ITE(IAT1)=IAT4
                          IPE(IAT1)=IAT5
                          IOE(IAT1)=IAT6
                       ENDIF
                    ELSE
                       AT1 = UPCASE(AT1)
                       LAST = LASTCH(AT1)
                       AT1 = AT1(1:LAST)//'$'
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                       IF (VER >= EQVER(IAT1) .AND. VER <= MAXVER) THEN
                          EQVER(IAT1)=VER
                          INNB(IAT1) = IAT1
                          IBE(IAT1) = IAT1
                          ITE(IAT1) = IAT1
                          IPE(IAT1) = IAT1
                          IOE(IAT1) = IAT1
                          DO I = 1, MXORDR
                             IORDER=CHAR(I+ICHAR('0'))
                             AT1 = AT1(1:LAST)//IORDER
                             CALL FNDATM(ATC,NATC,AT1,IAT1)
                             INNB(IAT1) = IAT1
                             IBE(IAT1) = IAT1
                             ITE(IAT1) = IAT1
                             IPE(IAT1) = IAT1
                             IOE(IAT1) = IAT1
                          ENDDO
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'auto_equivalence') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    IF (VER >= AEQVER(IAT1) .AND. VER <= MAXVER) THEN
                       AEQVER(IAT1)=VER
                       CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT5,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT6,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT7,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT8,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT9,ILOC,' ',FOUND)
                       CALL FNDNXT(INLINE,AT10,ILOC,' ',FOUND)
                       IF (.NOT.FOUND) &
                            CALL FNDNXT(INLINE,AT10,ILOC,'#',FOUND)
                       CALL FNDATM(ATC,NATC,AT2,IAT2)
                       CALL FNDATM(ATC,NATC,AT3,IAT3)
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                       CALL FNDATM(ATC,NATC,AT5,IAT5)
                       CALL FNDATM(ATC,NATC,AT6,IAT6)
                       CALL FNDATM(ATC,NATC,AT7,IAT7)
                       CALL FNDATM(ATC,NATC,AT8,IAT8)
                       CALL FNDATM(ATC,NATC,AT9,IAT9)
                       CALL FNDATM(ATC,NATC,AT10,IAT10)
                       INBAE(IAT1)=IAT2
                       IBAE(IAT1)=IAT4
                       ITEAE(IAT1)=IAT5
                       ITAAE(IAT1)=IAT6
                       IPEAE(IAT1)=IAT7
                       IPCAE(IAT1)=IAT8
                       IOEAE(IAT1)=IAT9
                       IOCAE(IAT1)=IAT10
                    ENDIF
                 ELSEIF (TTYPE == 'hbond_definition') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    IF (WORD == 'distance') THEN
                       IF (VER >= DISVER .AND. VER <= MAXVER) THEN
                          DISVER=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)HBDIST
                       ENDIF
                    ELSEIF (WORD == 'angle') THEN
                       IF (VER >= ANGVER .AND. VER <= MAXVER) THEN
                          ANGVER=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)HBANGL
                       ENDIF
                    ELSEIF (WORD == 'donors') THEN
                       IF (VER >= DONVER .AND. VER <= MAXVER) THEN
                          DONVER=VER
                          GOON = .TRUE.
240                       CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) THEN
                             CALL FNDNXT(INLINE,AT1,ILOC,'#',FOUND)
                             GOON = .FALSE.
                          ENDIF
                          NHTYP = NHTYP + 1
                          HYDTYP(NHTYP) = AT1
                          CALL FNDATM(ATC,NATC,AT1,IAT1)
                          HTYP(NHTYP) = IAT1
                          IF (GOON) GOTO 240
                       ENDIF
                    ELSEIF (WORD == 'acceptors') THEN
                       IF (VER >= ACPVER .AND. VER <= MAXVER) THEN
                          ACPVER=VER
                          GOON = .TRUE.
250                       CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) THEN
                             CALL FNDNXT(INLINE,AT1,ILOC,'#',FOUND)
                             GOON = .FALSE.
                          ENDIF
                          NACTYP = NACTYP + 1
                          ACPTYP(NACTYP) = AT1
                          CALL FNDATM(ATC,NATC,AT1,IAT1)
                          ATYP(NACTYP) = IAT1
                          IF (GOON) GOTO 250
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'quadratic_bond') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    INDX = 0
                    IF (NCBO > 1) THEN
                       DO J = 1,NCBO
                          IF (AT1 == ATC(BID1(J)).AND.AT2.EQ.ATC(BID2(J)).OR. &
                               AT2 == ATC(BID1(J)).AND.AT1.EQ.ATC(BID2(J)))INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCBO=NCBO+1
                       IF (NCBO > MAXCB) THEN
                          INDX = MAXCB
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCBO
                       ENDIF
                       CALL FNDATM(ATC,NATC,AT1,BID1(INDX))
                       CALL FNDATM(ATC,NATC,AT2,BID2(INDX))
                    ENDIF
                    IF (VER >= QBVER(INDX) .AND. VER <= MAXVER) THEN
                       QBVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       IF (FORCE  == 'CFF89') THEN
                          READ(WORD,710) CBOND1(INDX)
                       ELSE
                          READ(WORD,710) CBOND4(INDX)
                       ENDIF
                    ENDIF
                    BORD(INDX)=1.0
                 ELSEIF (TTYPE == 'quartic_bond') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    INDX = 0
                    IF (NCBO > 1) THEN
                       DO J = 1,NCBO
                          IF (AT1 == ATC(BID1(J)).AND.AT2.EQ.ATC(BID2(J)).OR. &
                               AT2 == ATC(BID1(J)).AND.AT1.EQ.ATC(BID2(J)))INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCBO=NCBO+1
                       IF (NCBO > MAXCB) THEN
                          INDX = MAXCB
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCBO
                       ENDIF
                    ENDIF
                    IF (VER >= QBVER(INDX) .AND. VER <= MAXVER) THEN
                       QBVER(INDX)=VER
                       CALL FNDATM(ATC,NATC,AT1,BID1(INDX))
                       CALL FNDATM(ATC,NATC,AT2,BID2(INDX))
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND1(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND3(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CBOND4(INDX)
                    ENDIF
                    BORD(INDX)=1.0
                 ELSEIF (TTYPE == 'quadratic_angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    IF (AT3(1:1)  ==  '*') THEN
                       AT10 = AT3(2:2)
                       IF (AT10 == ' ') THEN
                          IAT3 = 0
                       ELSE
                          READ(AT10,720)IAT3
                          IAT3 = -IAT3
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT3,IAT3)
                    ENDIF
                    INDX = 0
                    IF (NCTO > 1) THEN
                       DO J = 1,NCTO
                          IF (IAT2 == TID2(J)) THEN
                             IF (IAT1 == TID1(J) .AND. IAT3.EQ.TID3(J) .OR. &
                                  IAT3 == TID1(J).AND.IAT1.EQ.TID3(J)) INDX=J
                          ENDIF
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCTO=NCTO+1
                       IF (NCTO > MAXCT) THEN
                          INDX = MAXCT
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCTO
                       ENDIF
                    ENDIF
                    IF (VER >= QAVER(INDX) .AND. VER <= MAXVER) THEN
                       QAVER(INDX)=VER
                       TID1(INDX) = IAT1
                       TID2(INDX) = IAT2
                       TID3(INDX) = IAT3
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CTHET1(INDX)
                    ENDIF
                    TORD1(INDX)=1.0
                    TORD2(INDX)=1.0
                 ELSEIF (TTYPE == 'quartic_angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    IF (AT3(1:1)  ==  '*') THEN
                       AT10 = AT3(2:2)
                       IF (AT10 == ' ') THEN
                          IAT3 = 0
                       ELSE
                          READ(AT10,720)IAT3
                          IAT3 = -IAT3
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT3,IAT3)
                    ENDIF
                    INDX = 0
                    IF (NCTO > 1) THEN
                       DO J = 1,NCTO
                          IF (IAT2 == TID2(J)) THEN
                             IF (IAT1 == TID1(J) .AND. IAT3.EQ.TID3(J) .OR. &
                                  IAT3 == TID1(J).AND.IAT1.EQ.TID3(J)) INDX=J
                          ENDIF
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCTO=NCTO+1
                       IF (NCTO > MAXCT) THEN
                          INDX = MAXCT
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCTO
                       ENDIF
                    ENDIF
                    IF (VER >= QAVER(INDX) .AND. VER <= MAXVER) THEN
                       QAVER(INDX)=VER
                       TID1(INDX) = IAT1
                       TID2(INDX) = IAT2
                       TID3(INDX) = IAT3
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET1(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET6(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CTHET7(INDX)
                    ENDIF
                    TORD1(INDX)=1.0
                    TORD2(INDX)=1.0
                 ELSEIF (TTYPE == 'torsion_1') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    IF (AT4(1:1)  ==  '*') THEN
                       AT10 = AT4(2:2)
                       IF (AT10 == ' ') THEN
                          IAT4 = 0
                       ELSE
                          READ(AT10,720)IAT4
                          IAT4 = -IAT4
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                    ENDIF
                    INDX = 0
                    IF (NCPO > 1) THEN
                       DO J = 1,NCPO
                          IF ((IAT1 == PID1(J).AND.IAT2.EQ.PID2(J).AND. &
                               IAT3 == PID3(J).AND.IAT4.EQ.PID4(J)) .OR. &
                               (IAT1 == PID4(J) .AND. IAT2.EQ.PID3(J) .AND. &
                               IAT3 == PID2(J).AND.IAT4.EQ.PID1(J))) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCPO=NCPO+1
                       IF (NCPO > MAXCP) THEN
                          INDX = MAXCP
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCPO
                       ENDIF
                       PID1(INDX) = IAT1
                       PID2(INDX) = IAT2
                       PID3(INDX) = IAT3
                       PID4(INDX) = IAT4
                    ENDIF
                    IF (VER >= QTVER(INDX) .AND. VER <= MAXVER) THEN
                       QTVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) A
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,720) IA
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       IF (FORCE == 'CFF89') THEN
                          IF (IA  ==  1) THEN
                             CPHI1(INDX)=A
                             READ(WORD,710) A
                             CSGN1(INDX)=-SIGN(ONE,COS(A*DEGRAD))
                          ELSEIF (IA  ==  2) THEN
                             CPHI2(INDX)=A
                             READ(WORD,710) A
                             CSGN2(INDX)=-SIGN(ONE,COS(A*DEGRAD))
                          ELSEIF (IA  ==  3) THEN
                             CPHI3(INDX)=A
                             READ(WORD,710) A
                             CSGN3(INDX)=-SIGN(ONE,COS(A*DEGRAD))
                          ENDIF
                       ELSE
                          CPHI1(INDX)=A
                          CPHI2(INDX)=IA
                          READ(WORD,710) CPHI3(INDX)
                       ENDIF
                    ENDIF
                    PORD1(INDX)=1.0
                    PORD2(INDX)=1.0
                    PORD3(INDX)=1.0
                 ELSEIF (TTYPE == 'torsion_3') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    IF (AT4(1:1)  ==  '*') THEN
                       AT10 = AT4(2:2)
                       IF (AT10 == ' ') THEN
                          IAT4 = 0
                       ELSE
                          READ(AT10,720)IAT4
                          IAT4 = -IAT4
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                    ENDIF
                    INDX = 0
                    IF (NCPO > 1) THEN
                       DO J = 1,NCPO
                          IF ((IAT1 == PID1(J).AND.IAT2.EQ.PID2(J).AND. &
                               IAT3 == PID3(J).AND.IAT4.EQ.PID4(J)) .OR. &
                               (IAT1 == PID4(J) .AND. IAT2.EQ.PID3(J) .AND. &
                               IAT3 == PID2(J).AND.IAT4.EQ.PID1(J))) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCPO=NCPO+1
                       IF (NCPO > MAXCP) THEN
                          INDX = MAXCP
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCPO
                       ENDIF
                       PID1(INDX) = IAT1
                       PID2(INDX) = IAT2
                       PID3(INDX) = IAT3
                       PID4(INDX) = IAT4
                    ENDIF
                    IF (VER >= QTVER(INDX) .AND. VER <= MAXVER) THEN
                       QTVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)A
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)D
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)BX
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)E
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)C
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710)F
                       CPHI1(INDX)=A
                       CPHI2(INDX)=BX
                       CPHI3(INDX)=C
                       CSGN1(INDX)=SIGN(ONE,COS(D*DEGRAD))
                       CSGN2(INDX)=SIGN(ONE,COS(E*DEGRAD))
                       CSGN3(INDX)=SIGN(ONE,COS(F*DEGRAD))
                    ENDIF
                    PORD1(INDX)=1.0
                    PORD2(INDX)=1.0
                    PORD3(INDX)=1.0
                 ELSEIF (TTYPE == 'wilson_out_of_plane') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    IF (AT1 == '*   ') THEN
                       IAT1=0
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    IF (AT3 == '*   ') THEN
                       IAT3=0
                    ELSE
                       CALL FNDATM(ATC,NATC,AT3,IAT3)
                    ENDIF
                    IF (AT4 == '*   ') THEN
                       IAT4=0
                    ELSE
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                    ENDIF
                    INDX = 0
                    IF (NCI > 1) THEN
                       DO J = 1,NCI
                          IF (IAT1 == OPID1(J).AND.IAT2.EQ.OPID2(J).AND. &
                               IAT3 == OPID3(J).AND.IAT4.EQ.OPID4(J))INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCI=NCI+1
                       IF (NCI > MAXCI) THEN
                          INDX = MAXCI
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCI
                       ENDIF
                    ENDIF
                    IF (VER >= QOVER(INDX) .AND. VER <= MAXVER) THEN
                       QOVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)A
                       OPID1(INDX) = IAT1
                       OPID2(INDX) = IAT2
                       OPID3(INDX) = IAT3
                       OPID4(INDX) = IAT4
                       COPLN1(INDX)=A
                    ENDIF
                 ELSEIF (TTYPE == 'nonbond(9-6)') THEN
                    NBEXP=9
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    IF (WORD == '@type') THEN
                       CALL FNDNXT(INLINE,NBTYPE,ILOC,'#',FOUND)
                    ELSEIF (WORD == '@combination') THEN
                       CALL FNDNXT(INLINE,COMBIN,ILOC,'#',FOUND)
                    ELSE
                       READ(WORD,710)VER
                       ILOC = 10
                       CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                       INDX = 0
                       IF (NNONBP > 1) THEN
                          DO J = 1,NNONBP-1
                             IF (AT1 == ATC(PNBID1(J)))INDX=J
                          ENDDO
                       ENDIF
                       IF (INDX == 0) THEN
                          NNONBP=NNONBP+1
                          IF (NNONBP > MXNBDP) THEN
                             INDX = MXNBDP
                             ERROR = .TRUE.
                          ELSE
                             INDX = NNONBP
                          ENDIF
                          CALL FNDATM(ATC,NATC,AT1,PNBID1(INDX))
                       ENDIF
                       IF (VER >= QNBVER(INDX) .AND. VER <= MAXVER) THEN
                          QNBVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)CNB1(INDX)
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)CNB2(INDX)
                          CNB1(INDX+MAXCN)=9.0
                          NNDX = NNDX + 1
                          ITMP(NNDX)=PNBID1(INDX)
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'bond-bond') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    INDX = 0
                    IF (NCTO > 1) THEN
                       DO J = 1,NCTO
                          IF (AT1 == ATC(TID1(J)).AND.AT2.EQ.ATC(TID2(J)).AND. &
                               AT3 == ATC(TID3(J))) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX /= 0) THEN
                       IF (VER >= QBBVER(INDX) .AND. VER <= MAXVER) THEN
                          QBBVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710) CTHET3(INDX)
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'bond-angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    INDX=0
                    DO I=1,NCTO
                       IF (IAT1 == TID1(I) .AND. IAT2.EQ.TID2(I) .AND. &
                            IAT3 == TID3(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QBAVER(INDX) .AND. VER <= MAXVER) THEN
                          QBAVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) &
                               CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710) CTHET4(INDX)
                          IF (IAT1 == IAT3) THEN
                             CTHET5(INDX)=CTHET4(INDX)
                          ELSE
                             CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                             READ(WORD,710) CTHET5(INDX)
                          ENDIF
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'angle-angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    INDX = 0
                    IF (NTTP > 1) THEN
                       DO J = 1,NTTP-1
                          IF (AT1 == ATC(TTID1(J)) .AND. AT2.EQ.ATC(TTID2(J)) &
                               .AND.AT3 == ATC(TTID3(J)).AND.AT4.EQ.ATC(TTID4(J))) &
                               INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NTTP=NTTP+1
                       IF (NTTP > MXTTP) THEN
                          INDX = MXTTP
                          ERROR = .TRUE.
                       ELSE
                          INDX = NTTP
                       ENDIF
                       CALL FNDATM(ATC,NATC,AT1,TTID1(NTTP))
                       CALL FNDATM(ATC,NATC,AT2,TTID2(NTTP))
                       CALL FNDATM(ATC,NATC,AT3,TTID3(NTTP))
                       CALL FNDATM(ATC,NATC,AT4,TTID4(NTTP))
                    ENDIF
                    IF (VER >= QAAVER(INDX) .AND. VER <= MAXVER) THEN
                       QAAVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CTT(NTTP)
                    ENDIF
                    TTORD1(INDX)=1.0
                    TTORD2(INDX)=1.0
                    TTORD3(INDX)=1.0
                 ELSEIF (TTYPE == 'angle-angle-torsion_1') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)A
                    IF (AT1 == '*   ') THEN
                       IAT1=0
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    IF (AT4 == '*   ') THEN
                       IAT4=0
                    ELSE
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                    ENDIF
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QAATVER(INDX) .AND. VER <= MAXVER) THEN
                          QAATVER(INDX)=VER
                          CPHI4(INDX)=A
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'end_bond-torsion_3') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    IF (.NOT.FOUND) &
                         CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)C
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)D
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)E
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)F
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QEBTVER(INDX) .AND. VER <= MAXVER) THEN
                          QEBTVER(INDX)=VER
                          CBP11(INDX)=A
                          CBP12(INDX)=BX
                          CBP13(INDX)=C
                          IF (IAT1 == IAT4 .AND. IAT2.EQ.IAT3) THEN
                             CBP31(INDX)=A
                             CBP32(INDX)=BX
                             CBP33(INDX)=C
                          ELSE
                             CBP31(INDX)=D
                             CBP32(INDX)=E
                             CBP33(INDX)=F
                          ENDIF
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'middle_bond-torsion_3') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)C
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QMBTVER(INDX) .AND. VER <= MAXVER) THEN
                          QMBTVER(INDX)=VER
                          CBP21(INDX)=A
                          CBP22(INDX)=BX
                          CBP23(INDX)=C
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'angle-torsion_3') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    IF (.NOT.FOUND) &
                         CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)C
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)D
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)E
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)F
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QATVER(INDX) .AND. VER <= MAXVER) THEN
                          QATVER(INDX)=VER
                          CTP11(INDX)=A
                          CTP12(INDX)=BX
                          CTP13(INDX)=C
                          IF (IAT1 == IAT4 .AND. IAT2.EQ.IAT3) THEN
                             CTP21(INDX)=A
                             CTP22(INDX)=BX
                             CTP23(INDX)=C
                          ELSE
                             CTP21(INDX)=D
                             CTP22(INDX)=E
                             CTP23(INDX)=F
                          ENDIF
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'bond-bond_1_3') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                    READ(WORD,710)A
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QBBPVER(INDX) .AND. VER <= MAXVER) THEN
                          QBBPVER(INDX)=VER
                          CBB2(INDX)=A
                       ENDIF
                    ENDIF
                 ENDIF
              ELSEIF (PASS == 2) THEN
                 IF (TTYPE == 'order_bond') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    INDX = 0
                    IF (NCBO > 1) THEN
                       DO J = 1,NCBO
                          IF (AT1 == ATC(BID1(J)).AND.AT2.EQ.ATC(BID2(J)).AND. &
                               A == BORD(J).OR. &
                               AT2 == ATC(BID1(J)).AND.AT1.EQ.ATC(BID2(J)).AND. &
                               A == BORD(J))INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCBO=NCBO+1
                       IF (NCBO > MAXCB) THEN
                          INDX = MAXCB
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCBO
                       ENDIF
                    ENDIF
                    IF (VER >= QBVER(INDX) .AND. VER <= MAXVER) THEN
                       QBVER(INDX)=VER
                       CALL FNDATM(ATC,NATC,AT1,BID1(INDX))
                       CALL FNDATM(ATC,NATC,AT2,BID2(INDX))
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND1(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CBOND3(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CBOND4(INDX)
                    ENDIF
                    BORD(INDX)=A
                 ELSEIF (TTYPE == 'order_angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    IF (AT3(1:1)  ==  '*') THEN
                       AT10 = AT3(2:2)
                       IF (AT10 == ' ') THEN
                          IAT3 = 0
                       ELSE
                          READ(AT10,720)IAT3
                          IAT3 = -IAT3
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT3,IAT3)
                    ENDIF
                    INDX = 0
                    IF (NCTO > 1) THEN
                       DO J = 1,NCTO
                          IF (IAT2 == TID2(J)) THEN
                             IF ((IAT1 == TID1(J) .AND. IAT3.EQ.TID3(J) .AND. &
                                  A == TORD1(J) .AND. BX.EQ.TORD2(J)) .OR. &
                                  (IAT3 == TID1(J).AND.IAT1.EQ.TID3(J) .AND. &
                                  A == TORD2(J) .AND. BX.EQ.TORD1(J))) INDX=J
                          ENDIF
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCTO=NCTO+1
                       IF (NCTO > MAXCT) THEN
                          INDX = MAXCT
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCTO
                       ENDIF
                    ENDIF
                    IF (VER >= QAVER(INDX) .AND. VER <= MAXVER) THEN
                       QAVER(INDX)=VER
                       TID1(INDX) = IAT1
                       TID2(INDX) = IAT2
                       TID3(INDX) = IAT3
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET2(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET1(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710) CTHET6(INDX)
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CTHET7(INDX)
                       TORD1(INDX)=A
                       TORD2(INDX)=BX
                    ENDIF
                 ELSEIF (TTYPE == 'order_torsion') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    IF (AT1(1:1)  ==  '*') THEN
                       AT10 = AT1(2:2)
                       IF (AT10 == ' ') THEN
                          IAT1 = 0
                       ELSE
                          READ(AT10,720)IAT1
                          IAT1 = -IAT1
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT1,IAT1)
                    ENDIF
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    IF (AT4(1:1)  ==  '*') THEN
                       AT10 = AT4(2:2)
                       IF (AT10 == ' ') THEN
                          IAT4 = 0
                       ELSE
                          READ(AT10,720)IAT4
                          IAT4 = -IAT4
                       ENDIF
                    ELSE
                       CALL FNDATM(ATC,NATC,AT4,IAT4)
                    ENDIF
                    INDX = 0
                    IF (NCPO > 1) THEN
                       DO J = 1,NCPO
                          IF ((IAT1 == PID1(J).AND.IAT2.EQ.PID2(J).AND. &
                               IAT3 == PID3(J).AND.IAT4.EQ.PID4(J).AND. &
                               A == PORD1(J).AND.BX.EQ.PORD2(J).AND. &
                               C == PORD3(J)) .OR. &
                               (IAT1 == PID4(J) .AND. IAT2.EQ.PID3(J) .AND. &
                               IAT3 == PID2(J).AND.IAT4.EQ.PID1(J).AND. &
                               A == PORD3(J).AND.BX.EQ.PORD2(J).AND. &
                               C == PORD1(J))) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NCPO=NCPO+1
                       IF (NCPO > MAXCP) THEN
                          INDX = MAXCP
                          ERROR = .TRUE.
                       ELSE
                          INDX = NCPO
                       ENDIF
                       PID1(INDX) = IAT1
                       PID2(INDX) = IAT2
                       PID3(INDX) = IAT3
                       PID4(INDX) = IAT4
                       PORD1(INDX) = A
                       PORD2(INDX) = BX
                       PORD3(INDX) = C
                    ENDIF
                    IF (VER >= QTVER(INDX) .AND. VER <= MAXVER) THEN
                       QTVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)A
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)D
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)BX
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)E
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)C
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710)F
                       CPHI1(INDX)=A
                       CPHI2(INDX)=BX
                       CPHI3(INDX)=C
                       CSGN1(INDX)=SIGN(ONE,COS(D*DEGRAD))
                       CSGN2(INDX)=SIGN(ONE,COS(E*DEGRAD))
                       CSGN3(INDX)=SIGN(ONE,COS(F*DEGRAD))
                    ENDIF
                 ELSEIF (TTYPE == 'order-bond-bond') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    INDX = 0
                    IF (NCTO > 1) THEN
                       DO J = 1,NCTO
                          IF (AT1 == ATC(TID1(J)).AND.AT2.EQ.ATC(TID2(J)).AND. &
                               AT3 == ATC(TID3(J)).AND.A.EQ.TORD1(J).AND. &
                               BX == TORD2(J)) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX /= 0) THEN
                       IF (VER >= QBBVER(INDX) .AND. VER <= MAXVER) THEN
                          QBBVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710) CTHET3(INDX)
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'order-bond-angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    INDX=0
                    DO I=1,NCTO
                       IF (IAT1 == TID1(I) .AND. IAT2.EQ.TID2(I) .AND. &
                            IAT3 == TID3(I).AND.A.EQ.TORD1(I).AND. &
                            BX == TORD2(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QBAVER(INDX) .AND. VER <= MAXVER) THEN
                          QBAVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) THEN
                             CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                             READ(WORD,710) CTHET4(INDX)
                             CTHET5(INDX)=CTHET4(INDX)
                          ELSE
                             READ(WORD,710) CTHET4(INDX)
                             CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                             IF (FOUND) THEN
                                READ(WORD,710) CTHET5(INDX)
                             ELSE
                                CTHET5(INDX)=CTHET4(INDX)
                             ENDIF
                          ENDIF
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'order-angle-angle') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    INDX = 0
                    IF (NTTP > 1) THEN
                       DO J = 1,NTTP
                          IF (AT1 == ATC(TTID1(J)) .AND. AT2.EQ.ATC(TTID2(J)) &
                               .AND.AT3 == ATC(TTID3(J)).AND.AT4.EQ.ATC(TTID4(J)) &
                               .AND.A == TTORD1(J).AND.BX.EQ.TTORD2(J).AND. &
                               C == TTORD3(J)) INDX=J
                       ENDDO
                    ENDIF
                    IF (INDX == 0) THEN
                       NTTP=NTTP+1
                       IF (NTTP > MXTTP) THEN
                          INDX = MXTTP
                          ERROR = .TRUE.
                       ELSE
                          INDX = NTTP
                       ENDIF
                       CALL FNDATM(ATC,NATC,AT1,TTID1(NTTP))
                       CALL FNDATM(ATC,NATC,AT2,TTID2(NTTP))
                       CALL FNDATM(ATC,NATC,AT3,TTID3(NTTP))
                       CALL FNDATM(ATC,NATC,AT4,TTID4(NTTP))
                       TTORD1(INDX)=A
                       TTORD2(INDX)=BX
                       TTORD3(INDX)=C
                    ENDIF
                    IF (VER >= QAAVER(INDX) .AND. VER <= MAXVER) THEN
                       QAAVER(INDX)=VER
                       CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                       READ(WORD,710) CTT(NTTP)
                    ENDIF
                 ELSEIF (TTYPE == 'order-angle-angle-torsion') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO I=1,NCPO
                       IF (IAT1 == PID1(I) .AND. IAT2.EQ.PID2(I) .AND. &
                            IAT3 == PID3(I) .AND. IAT4.EQ.PID4(I) .AND. &
                            A == PORD1(I) .AND. BX.EQ.PORD2(I) .AND. &
                            C == PORD3(I)) INDX=I
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QAATVER(INDX) .AND. VER <= MAXVER) THEN
                          QAATVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)CPHI4(INDX)
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'order-end_bond-torsion') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO J=1,NCPO
                       IF (IAT1 == PID1(J) .AND. IAT2.EQ.PID2(J) .AND. &
                            IAT3 == PID3(J) .AND. IAT4.EQ.PID4(J) .AND. &
                            A == PORD1(J) .AND. BX.EQ.PORD2(J) .AND. &
                            C == PORD3(J)) INDX=J
                    ENDDO
                    IF (INDX /= 0) THEN
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)A
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       READ(WORD,710)BX
                       CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                       IF (.NOT.FOUND) THEN
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)C
                          D=A
                          E=BX
                          F=C
                       ELSE
                          READ(WORD,710)C
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)D
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)E
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)F
                       ENDIF
                       IF (VER >= QEBTVER(INDX) .AND. VER <= MAXVER) THEN
                          QEBTVER(INDX)=VER
                          CBP11(INDX)=A
                          CBP12(INDX)=BX
                          CBP13(INDX)=C
                          CBP31(INDX)=D
                          CBP32(INDX)=E
                          CBP33(INDX)=F
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'order-middle_bond-torsion') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO J=1,NCPO
                       IF (IAT1 == PID1(J) .AND. IAT2.EQ.PID2(J) .AND. &
                            IAT3 == PID3(J) .AND. IAT4.EQ.PID4(J) .AND. &
                            A == PORD1(J) .AND. BX.EQ.PORD2(J) .AND. &
                            C == PORD3(J)) INDX=J
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QMBTVER(INDX) .AND. VER <= MAXVER) THEN
                          QMBTVER(INDX)=VER
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)A
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)BX
                          CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                          READ(WORD,710)C
                          CBP21(INDX)=A
                          CBP22(INDX)=BX
                          CBP23(INDX)=C
                       ENDIF
                    ENDIF
                 ELSEIF (TTYPE == 'order-angle-torsion') THEN
                    ILOC = 1
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)VER
                    ILOC = 10
                    CALL FNDNXT(INLINE,AT1,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT2,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT3,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,AT4,ILOC,' ',FOUND)
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)A
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)BX
                    CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                    READ(WORD,710)C
                    CALL FNDATM(ATC,NATC,AT1,IAT1)
                    CALL FNDATM(ATC,NATC,AT2,IAT2)
                    CALL FNDATM(ATC,NATC,AT3,IAT3)
                    CALL FNDATM(ATC,NATC,AT4,IAT4)
                    INDX=0
                    DO J=1,NCPO
                       IF (IAT1 == PID1(J) .AND. IAT2.EQ.PID2(J) .AND. &
                            IAT3 == PID3(J) .AND. IAT4.EQ.PID4(J) .AND. &
                            A == PORD1(J) .AND. BX.EQ.PORD2(J) .AND. &
                            C == PORD3(J)) INDX=J
                    ENDDO
                    IF (INDX /= 0) THEN
                       IF (VER >= QATVER(INDX) .AND. VER <= MAXVER) THEN
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)A
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          READ(WORD,710)BX
                          CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                          IF (.NOT.FOUND) THEN
                             CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                             READ(WORD,710)C
                             D=A
                             E=BX
                             F=C
                          ELSE
                             READ(WORD,710)C
                             CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                             READ(WORD,710)D
                             CALL FNDNXT(INLINE,WORD,ILOC,' ',FOUND)
                             READ(WORD,710)E
                             CALL FNDNXT(INLINE,WORD,ILOC,'#',FOUND)
                             READ(WORD,710)F
                          ENDIF
                          CALL FNDATM(ATC,NATC,AT1,IAT1)
                          QATVER(INDX)=VER
                          CTP11(INDX)=A
                          CTP12(INDX)=BX
                          CTP13(INDX)=C
                          CTP21(INDX)=D
                          CTP22(INDX)=E
                          CTP23(INDX)=F
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
           ENDIF
        ENDIF
     ENDIF
  enddo loop180

  IF (ERROR) THEN
     IF(WRNLEV > -3) WRITE(OUTU,730)NCBO,MAXCB,NCTO,MAXCT, &
          NCPO,MAXCP,NCI,MAXCI,NNONBP,MXNBDP
     CALL WRNDIE(-3,"NEWPRM<parrdr_cff.src>","Error in reading or parsing")
     RETURN
  ENDIF

  ! addedd by clb3
  if(pass==2 .and. allocated(atver)) then
     call chmdealloc(file_name,routine_name,'atver',maxatc,crl=atver)
     call chmdealloc(file_name,routine_name,'eqver',maxatc,crl=eqver)
     call chmdealloc(file_name,routine_name,'qbver',maxcb,crl=qbver)
     call chmdealloc(file_name,routine_name,'qaver',maxct,crl=qaver)
     call chmdealloc(file_name,routine_name,'qtver',maxcp,crl=qtver)
     call chmdealloc(file_name,routine_name,'qover',maxci,crl=qover)
     call chmdealloc(file_name,routine_name,'qnbver',mxnbdp,crl=qnbver)
     call chmdealloc(file_name,routine_name,'qaaver',mxttp,crl=qaaver)
     call chmdealloc(file_name,routine_name,'qbaver',maxct,crl=qbaver)
     call chmdealloc(file_name,routine_name,'qbbver',maxct,crl=qbbver)
     call chmdealloc(file_name,routine_name,'qaatver',maxcp,crl=qaatver)
     call chmdealloc(file_name,routine_name,'qebtver',maxcp,crl=qebtver)
     call chmdealloc(file_name,routine_name,'qmbtver',maxcp,crl=qmbtver)
     call chmdealloc(file_name,routine_name,'qatver',maxcp,crl=qatver)
     call chmdealloc(file_name,routine_name,'qbbpver',maxcp,crl=qbbpver)
     call chmdealloc(file_name,routine_name,'aeqver',maxatc,crl=aeqver)
  endif
! BIOVIA Code Start
#if KEY_LICENSE==1
     IF (DECODE) THEN
!    This call resets the counter in the decode routine so that the
!    file can be read again.
        LLEN = 5
        CALL DECODE_FF("#end",TEMP,LLEN)
     ENDIF
#endif
! BIOVIA Code End
  RETURN
700 FORMAT(256A)
710 FORMAT(F12.6)
711 FORMAT(I3)
720 FORMAT(I1)
730 FORMAT(' ERROR: Read error or TOO MANY PARAMETERS FOR CURRENT DIMENSIONS.'/ &
       ' ONE OR MORE OF THE FOLLOWING DIMENSIONS MUST BE INCREASED'/ &
       ' REQUIRED LIMIT   CURRENT LIMIT'/4(2I15/))
END SUBROUTINE NEWPRM


SUBROUTINE PARODR
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use param
  use psf
  use energym
  use cff_fcm
  implicit none
  !
  INTEGER I,KND1,KND2,KND3,KND4
  real(chm_real) CTMP
  !
  !-----order parameter types.
  !     note. the non-bond parameters are not ordered as the search
  !     mechanism for these parameters in nbpk does not need the
  !     parameters to be ordered - neither of course are the charge
  !     parameters ordered.
  !
  !-----bond parms. reorder 2 names only.
  IF (NCB  >  0) THEN
     DO I=1,NCB
        KND1 = BID1(I)
        KND2 = BID2(I)
        BID1(I) = MIN0(KND1,KND2)
        BID2(I) = MAX0(KND1,KND2)
     ENDDO
  ENDIF
  !
  !-----angle parameters. reorder first and last names - 1 and 3
  IF (NCT  >  0) THEN
     DO I=1,NCT
        KND1 = TID1(I)
        KND3 = TID3(I)
        TID1(I) = MIN0(KND1,KND3)
        TID3(I) = MAX0(KND1,KND3)
        !-----if the 1st and 3rd atoms were switched then we must also switch
        !     the bond*theta interaction constants.
        IF (TID1(I) /= KND1) THEN
           CTMP=CTHET4(I)
           CTHET4(I)=CTHET5(I)
           CTHET5(I)=CTMP
        ENDIF
     ENDDO
  ENDIF
  !
  !-----torsion parameters have 4 names - reorder the atoms as follows:
  !     if kndpn(2)=kndpn(3) then reorder names for atoms 1,4 to be in
  !     numerical kind order.
  !     otherwise reorder 2,3 into numerical kind order - and reorder
  !     atoms 1,4 if atoms 2,3 are reordered - so that atom 1 is still
  !     next to the original atom 2.
  IF (NCP  >  0) THEN
     DO I=1,NCP
        KND1 = PID1(I)
        KND2 = PID2(I)
        KND3 = PID3(I)
        KND4 = PID4(I)
        IF (KND2  ==  KND3) THEN
           !
           !-----reorder atoms 1,4 only - 2,3 are identical kind type
           PID1(I) = MIN0(KND1,KND4)
           PID4(I) = MAX0(KND1,KND4)
           IF (FORCE == 'CFF89') THEN
              IF (PID1(I) /= KND1) THEN
                 CTMP=CBP11(I)
                 CBP11(I)=CBP31(I)
                 CBP31(I)=CTMP
                 CTMP=CBP12(I)
                 CBP12(I)=CBP32(I)
                 CBP32(I)=CTMP
                 CTMP=CBP13(I)
                 CBP13(I)=CBP33(I)
                 CBP33(I)=CTMP
                 CTMP=CTP11(I)
                 CTP11(I)=CTP21(I)
                 CTP21(I)=CTMP
                 CTMP=CTP12(I)
                 CTP12(I)=CTP22(I)
                 CTP22(I)=CTMP
                 CTMP=CTP13(I)
                 CTP13(I)=CTP23(I)
                 CTP23(I)=CTMP
              ENDIF
           ENDIF
           !
           !-----check if atoms 2,3 are already in the right order - if so
           !     no change to order
        ELSEIF (KND2  >  KND3) THEN
           !
           !-----reorder 2,3 and 1,4
           PID1(I) = KND4
           PID2(I) = KND3
           PID3(I) = KND2
           PID4(I) = KND1
           IF (FORCE == 'CFF89') THEN
              CTMP=CBP11(I)
              CBP11(I)=CBP31(I)
              CBP31(I)=CTMP
              CTMP=CBP12(I)
              CBP12(I)=CBP32(I)
              CBP32(I)=CTMP
              CTMP=CBP13(I)
              CBP13(I)=CBP33(I)
              CBP33(I)=CTMP
              CTMP=CTP11(I)
              CTP11(I)=CTP21(I)
              CTP21(I)=CTMP
              CTMP=CTP12(I)
              CTP12(I)=CTP22(I)
              CTP22(I)=CTMP
              CTMP=CTP13(I)
              CTP13(I)=CTP23(I)
              CTP23(I)=CTMP
           ENDIF
        ENDIF
     ENDDO
  ENDIF
  !
  !-----out of plane parameter have 4 names - only the second name is
  !     significant - reorder the 1,3 and 4 names to be the kind
  !     ordering of the atoms.
  IF (NCI  >  0) THEN
     DO I=1,NCI
        KND1 = OPID1(I)
        KND3 = OPID3(I)
        KND4 = OPID4(I)
        OPID1(I) = MIN0(KND1,KND3,KND4)
        IF (OPID1(I)  ==  KND1) OPID3(I) = MIN0(KND3,KND4)
        IF (OPID1(I)  ==  KND3) OPID3(I) = MIN0(KND1,KND4)
        IF (OPID1(I)  ==  KND4) OPID3(I) = MIN0(KND1,KND3)
        OPID4(I) = MAX0(KND1,KND3,KND4)
     ENDDO
  ENDIF
  !
  !-----theta*theta parameters - 4 names as in torsions - only
  !     outer 2 are ordered as an c' (ca n) h theta*theta is
  !     different from c' (n ca) hn.
  IF (NTTP  >  0) THEN
     DO I=1,NTTP
        KND1 = TTID1(I)
        KND4 = TTID4(I)
        TTID1(I) = MIN0(KND1,KND4)
        TTID4(I) = MAX0(KND1,KND4)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE PARODR


SUBROUTINE NBNDAS(NBC,COMBIN,NBTYPE,NBEXP,ITMP)
  !
  use chm_kinds
  !
  use dimens_fcm
  use number
  use param
  use cff_fcm
  use stream
  implicit none
  !
  !     Function
  !     Assign non-bond parameters - use geometric   mean for all
  !     single atoms.
  !     Also set up no array which for a given pair of kinds of atoms
  !     points to the non-bond potential parameters in cnb1 and cnb2.
  !     this overwrites the input single atom parameters with the new
  !     pair parameters.
  !     note. although more parameters are generated than used this
  !     works as the single atom non-bond parameters are put at the end
  !     of cpv set up special values in no to indicate if the parameters
  !     are zero. this allos nonbp or nbsums to skip them.
  !     a value of -3 indicates both parameters are zero.
  !     nb*   = 1 means that single atom parameters were input,
  !     nb*   = 2 pair parameters - in which case only set up no array.
  !
  !     6/86 -- by David A. Pearlman
  !     modified to 1) allow epsilon/r* type input when non-bond
  !     *pairs* are specified; and 2) set up a default value for the
  !     replusive parameter, n, in A/r**n - B/r**6. This value
  !     can be used if either no value for n is specified by the
  !     user, or if the value of n specified by the user is not a
  !     valid option. At present, correct behaviour by all Discover
  !     routines will only occur if n=12. After some modifications
  !     (not yet implemented), n=9 will also be a valid option.
  !     How discover assigns a value to n is controlled by the
  !     variable ihdwir (i.e. i-"hardwire"--get it?), which has been added
  !     in this revision of the code. If ihdwir=1, then a hardwired
  !     value, rhdwir, is used for the repulsive exponent n if, and
  !     only if, the value of n is not set by the user to a valid value.
  !     There are 'ivalid' valid values for n, and these are stored
  !     in array 'rvalid'. If ihdwir=2, then the hardwired value, rhdwir,
  !     is used for the repulsive exponent, n, in all cases, regardless
  !     of the value specified by the user. Values of ihdwir, rhdwir,
  !     ivalid, and rvalid are all specified in data statements below.
  !     In the current implementation, the program is set with
  !     ihdwir=2, i.e. a value of 12 is *always* used for n. When the code
  !     for n=9 is fully implemented, ihdwir should be changed to 1.
  !
  INTEGER I,ICH,IHDWIR,IP,IT,IVALID,J,JP,JT,K, &
       KND1,KND2,NBC,NPAR,PX(2,MAXATC),ITMP(*)
  INTEGER NBEXP
  real(chm_real) EXPR,RHDWIR,E(MAXATC),EX(MAXATC), &
       R(MAXATC),RVALID(5),R1,E1,EX1,R13,R16,R23,R26
  LOGICAL ERROR
  character(len=12) COMBIN,NBTYPE
  real(chm_real) ACOEF(MAXATC),BCOEF(MAXATC)
  !
  !-----thefollowing data statements contain ihdwir and rhdwir. ihdwir
  !     controls the behavior of the program if the value of the repulsive
  !     exponent specified by the user is invalid (see revisions
  !     section above). This behavior is effected by setting the value of
  !     ihdwir to 1. It can also be used to force the program to use
  !     the repulsive coefficient contained in rhdwir. This behavior
  !     is effected by setting the value of ihdwir to 2. If ihdwir=0,
  !     then the program makes no checks for the validities of user
  !     specified   values of the repulsive exponents.
  !
  DATA IHDWIR/1/, RHDWIR/12.0D0/
  !
  !-----thefollowing data statements contain ivalid and rvalid. ivalid
  !     gives the number of possible values for the repulsive exponent, n.
  !     rvalid contains these values, along with 0's for the remaining
  !     elements of the rvalid array. If ihdwir is not equal to 1, these
  !     values are not used.
  !
  DATA IVALID/2/, RVALID/9.0D0, 12.0D0, 0.0D0, 0.0D0, 0.0D0/
  !
  ERROR=.FALSE.
  NNBDP = NNONBP
  NPAR = NNONBP
  !
  !-----first,check if a valid value of the exponent has been specified
  !     (do check if ihdwir /= 0)
  !
  IF (FORCE == 'CVFF'.OR.FORCE.EQ.'AMBER') THEN
     IF (IHDWIR > 0) THEN
        loop110: DO I=1, NPAR
           EX(I) = CNB1(I+MAXCN)
           IF (IHDWIR == 1) THEN
              DO ICH=1, IVALID
                 IF (EX(I) == RVALID(ICH)) cycle loop110
              ENDDO
              CNB1(I+MAXCN) = RHDWIR
           ELSE IF (IHDWIR == 2) THEN
              CNB1(I+MAXCN) = RHDWIR
           ENDIF
        enddo loop110
     ENDIF
  ENDIF
  !
  !-----only need to assign non-bonds if single atom parameters
  !     input
  !
  IF (NBC == 1) THEN
     !
     !-----loop over non-bond single atom parms.
     !
     IF (NPAR-IHYDNB  >  0) THEN
        !
        !-----save single atom non-bond parms (elements 1 to npar-ihydnb)
        !-----and hydrogen bond parameters (elements npar-ihydnb+1 to npar)
        !-----in temporary arrays e,r and ex
        !
        !-----save single atom non-bond parms (which must be less than no.
        !     of atom types allowed - currently MAXATC) in temporary arrays e,r
        !     and ex.
        do i=1,npar-ihydnb
           ex(i) = cnb1(i+MAXCN)
           px(1,i)=pnbid1(i)
           px(2,i)=pnbid2(i)
           !
           if (nbtype == 'r-eps') then
              r(i) = cnb1(i)
              e(i) = cnb2(i)
              if (nbexp == 9) then
                 acoef(i)=2*e(i)*r(i)**9
                 bcoef(i)=3*e(i)*r(i)**6
              else
                 acoef(i) = (six/(ex(i)-six))*e(i)*r(i)**ex(i)
                 bcoef(i) = (ex(i)/(ex(i)-six))*e(i)*r(i)**6
              end if
           else
              acoef(i)=cnb1(i)
              bcoef(i)=cnb2(i)
              if (cnb2(i) > zero) then
                 if (nbexp == 9) then
                    r(i)=((3*cnb1(i))/(2*cnb2(i)))**(1.0/3.0)
                    e(i)=4*cnb2(i)**3/(27*cnb1(i)**2)
                 else
                    r(i)=(two*cnb1(i)/cnb2(i))**sixth
                    e(i)=cnb2(i)*cnb2(i)/(four*cnb1(i))
                 end if
              else
                 r(i) = zero
                 e(i) = zero
              endif
           end if
           vdwr(itmp(i)) = r(i)
        enddo
        !
        DO I=NPAR-ihydnb+1, npar
           R(I)=CNB1(I)
           E(I)=CNB2(I)
           EX(I)=CNB1(I+MAXCN)
           PX(1,I)=PNBID1(I)
           PX(2,I)=PNBID2(I)
           VDWR(ITMP(I)) = R(I)
        ENDDO
        !
        IF (COMBIN /= 'geometric') THEN
           IF (COMBIN == 'arithmetic') THEN
              !
              K = 0
              DO I=1,NPAR-IHYDNB
                 DO J=1,I
                    K = K + 1
                    IF (K > MXNBDP) THEN
                       IF(PRNLEV > 3) WRITE(OUTU,6040)MXNBDP,MXNBDP
                       STOP
                    ENDIF
                    CNB2(K+MAXCN) = HALF*(R(I)+R(J))
                    CNB5(K) = SQRT(E(I)*E(J))
                    CNB1(K+MAXCN) = HALF*(EX(I)+EX(J))
                    PNBID1(K) = PX(1,I)
                    PNBID2(K) = PX(1,J)
                 ENDDO
              ENDDO
              !
           ELSEIF (COMBIN == 'sixth-power') THEN
              !
              K = 0
              DO I=1,NPAR-IHYDNB
                 R13 = R(I)*R(I)*R(I)
                 R16 = R13*R13
                 DO J=1,I
                    K = K + 1
                    IF (K > MXNBDP) THEN
                       IF(PRNLEV > 3) WRITE(OUTU,6040)MXNBDP,MXNBDP
                       STOP
                    ENDIF
                    !
                    R23 = R(J)*R(J)*R(J)
                    R26 = (R23*R23 + R16)*HALF
                    IF (R26  >  ZERO ) THEN
                       CNB2(K+MAXCN) = R26**SIXTH
                       CNB5(K) = SQRT(E(I)*E(J))*TWO*R13*R23 /  &
                            (r(i)**6+r(j)**6)
                    ELSE
                       CNB2(K+MAXCN) = ZERO
                       CNB5(K) = ZERO
                    ENDIF
                    CNB1(K+MAXCN)=EX(I)
                    PNBID1(K) = PX(1,I)
                    PNBID2(K) = PX(1,J)
                 ENDDO
              ENDDO
           ENDIF
           !
           !-----convert from r* end epsilon to a's and b's and save them
           !
           DO I=1,K
              R1 = CNB2(I+MAXCN)
              E1 = CNB5(I)
              EX1 = CNB1(I+MAXCN)
              IF (NBEXP == 9) THEN
                 CNB1(I)=2*E1*R1**9
                 CNB2(I)=3*E1*R1**6
              ELSE
                 CNB1(I) = (SIX/(EX1-SIX))*E1*R1**EX1
                 CNB2(I) = (EX1/(EX1-SIX))*E1*R1**6
              END IF
           ENDDO
           !
           !-----h-bond params; pack as is:
           !
           DO I=NPAR-IHYDNB+1,NPAR
              K = K + 1
              IF (K > MXNBDP) THEN
                 IF(PRNLEV > 3) WRITE(OUTU,6040)MXNBDP,MXNBDP
                 STOP
              ENDIF
              CNB1(K)=R(I)
              CNB2(K)=E(I)
              CNB1(K+MAXCN)=EX(I)
              IF (R(I) == 0 .OR. E(I).EQ.0) THEN
                 CNB2(I+MAXCN) = ZERO
                 CNB5(I) = ZERO
              ELSE
                 CNB2(I+MAXCN)=(TWO*R(I)/E(I))**SIXTH
                 CNB5(I)=E(I)*E(I)/(FOUR*R(I))
              ENDIF
              PNBID1(K) = PX(1,I)
              PNBID2(K) = PX(2,I)
           ENDDO
           !
        ELSE
           !
           !-----a'sand b's - find geometric mean of all pairs from single
           !     atoms in input.
           !
           K = 0
           DO I=1,NPAR-IHYDNB
              DO J=1,I
                 K = K + 1
                 IF (K > MXNBDP) THEN
                    IF(PRNLEV > 3) WRITE(OUTU,6040)MXNBDP,MXNBDP
                    STOP
                 ENDIF
                 CNB1(K)=SQRT(ACOEF(I)*ACOEF(J))
                 CNB2(K)=SQRT(BCOEF(I)*BCOEF(J))
                 CNB1(K+MAXCN)=HALF*(EX(I)+EX(J))
                 PNBID1(K) = PX(1,I)
                 PNBID2(K) = PX(1,J)
              ENDDO
           ENDDO
           !
           !-----h-bond params; pack as is:
           !
           DO I=NPAR-IHYDNB+1,NPAR
              K = K + 1
              IF (K > MXNBDP) THEN
                 IF(PRNLEV > 3) WRITE(OUTU,6040)MXNBDP,MXNBDP
                 STOP
              ENDIF
              CNB1(K)=R(I)
              CNB2(K)=E(I)
              CNB1(K+MAXCN)=EX(I)
              PNBID1(K) = PX(1,I)
              PNBID2(K) = PX(2,I)
           ENDDO
           !
           !-----convert A and B to r* and epsilon
           !
           DO I=1,K
              if (cnb2(i) > zero) then
                 if (nbexp == 9) then
                    cnb2(i+MAXCN)=((3*cnb1(i))/(2*cnb2(i)))**(1.0/3.0)
                    cnb5(i)=4*cnb2(i)**3/(27*cnb1(i)**2)
                 else
                    cnb2(i+MAXCN)=(two*cnb1(i)/cnb2(i))**sixth
                    cnb5(i)=cnb2(i)*cnb2(i)/(four*cnb1(i))
                 end if
              else
                 cnb2(i+MAXCN) = zero
                 cnb5(i) = zero
              endif
           ENDDO
        ENDIF
        !
        NBC=2
        NNBDP=K
     ENDIF
  ELSE
     !
     !-----start here if non-bonds were put in as pairs; first
     !     check to see if an epsilon/r* to A/B conversion is necessary.
     !     this fixes a bug in previous versions of the program.
     !     -- D. Pearlman
     !
     R(1) = CNB1(1)
     E(1) = CNB2(1)
     IF (NBTYPE == 'r-eps') THEN
        !
        DO I=1,NNBDP
           R1 = CNB1(I)
           E1 = CNB2(I)
           EX1 = CNB1(I+MAXCN)
           CNB1(I) = (SIX/(EX1-SIX))*E1*R1**EX1
           CNB2(I) = (EX1/(EX1-SIX))*E1*R1**6
           CNB2(I+MAXCN) = R1
           CNB5(I) = E1
        ENDDO
     ELSE
        !
        !-----convert A and B to r* and epsilon
        !
        DO I=1,K
           IF (CNB1(I) == 0 .OR. CNB2(I).EQ.0) THEN
              CNB2(I+MAXCN) = ZERO
              CNB5(I) = ZERO
           ELSE
              CNB2(I+MAXCN)=(TWO*CNB1(I)/CNB2(I))**SIXTH
              CNB5(I)=CNB2(I)*CNB2(I)/(FOUR*CNB1(I))
           ENDIF
        ENDDO
     ENDIF
  ENDIF
  !
  !     add one more set of parameters with values of zero
  !
  NNBDP = NNBDP+1
  CNB1(NNBDP) = ZERO
  CNB2(NNBDP) = ZERO
  CNB1(NNBDP+MAXCN) = CNB1(1+MAXCN)
  CNB2(NNBDP+MAXCN) = ZERO
  CNB5(NNBDP) = ZERO
  NATC=NATC+1
  ATC(NATC)='xx  '
  PNBID1(NNBDP) = NATC
  PNBID2(NNBDP) = NATC
  !
  !-----setup no array. this points to parameters in cpv for each
  !     possible pair of types of atom.  All undefined parameters will
  !     be set to zero by pointing to the special parameters just defined.
  !
  DO IP=1,NATC
     DO JP=1,NATC
        MNO(IP,JP) = NNBDP
     ENDDO
  ENDDO
  !
  NNONBP=NNBDP
  NNBTYP = NNBDP
  IF (NNBDP  <=  0) RETURN
  !
  !----for regular non-bonds, multiply repulsion by repulsion exponent,
  !----and dispersion by dispersion exponent(6). this is needed for
  !     the calculation of 1-st derivatives in nbsums.
  EXPR = CNB1(1+MAXCN)
  !
  DO IP=1,NNBDP-IHYDNB-1
     CNBD1(IP) = CNB1(IP)*EXPR
     CNBD2(IP) = CNB2(IP)*SIX
  ENDDO
  !
  !-----now account for the repulsive and dispersion of the hydrogen-bond
  !-----10-12 terms
  !
  EXPR=TWELVE
  DO IP=NNBDP-IHYDNB,NNBDP
     CNBD1(IP)=CNB1(IP)*EXPR
     CNBD2(IP)=CNB2(IP)*TEN
  ENDDO
  !
  !
  !-----loop over all pair potential parameters - find kinds of
  !     atoms involved and update pointer in no.
  IF (NNBDP  <=  0) RETURN
  !
  loop370: DO IT=1,NATC
     KND1 = INNB(IT)
     if (KND1 < 1) then
        VDWR(IT) = ONE
     else
        VDWR(IT) = VDWR(KND1)
     endif
     loop360: DO JT=1,IT
        KND2 = INNB(JT)
        !
        loop400: DO IP=1,NNBDP-IHYDNB-1
           !
           IF ((PNBID1(IP) == KND1 .AND. PNBID2(IP).EQ.KND2) .OR. &
                (PNBID2(IP) == KND1 .AND. PNBID1(IP).EQ.KND2)) THEN
              MNO(IT,JT) = IP
              MNO(JT,IT) = IP
              cycle loop360   !GOTO 360
           ENDIF
        enddo loop400
     enddo loop360
  enddo loop370
  RETURN
6040 FORMAT(/1X,'Error in NBNDAS: the non-bond parameter', &
       ' id array PNBID2(,',I6,') in PARID.INS ', &
       'has overflowed.'/ &
       ' MXNBDP = ',I6,' must be increased in', &
       ' PARAM.INS'/ &
       '       This run is being terminated.')
END SUBROUTINE NBNDAS


SUBROUTINE HBNDAS(IPRINT,ACPTYP,HYDTYP)
  use chm_kinds
  use dimens_fcm
  use number
  use param
  use cff_fcm
  implicit none
  ! Function
  !     This routine:
  !
  !     1) assigns special values to the pointer array no
  !       which points to the non-bond potential parameters for
  !       hydrogen bonds.
  !
  !     2) it assigns a special value for "Watt's potential" water
  !       hydrogen bonds. All hydrogen bonds between water and any other
  !       molecules (including another water) will be handled by the
  !       "Watt's potential" if *AND ONLY IF*:
  !          a) 'hw' and 'ow' are defined as parent atom types in the
  !              potential file;
  !          b) 'hw' is the *first* atom type defined in the hydrogen
  !              bond donor list; and
  !          c) 'ow' is the *first* atom type defined in the hydrogen
  !              bond acceptor list.
  !
  !     If hydrogen bond 10-12 parameters have been included in the
  !     potential file, then explicit A/r**12-B/r**10 interactions are
  !     calculated for hydrogen bonds. Otherwise, only coulombic
  !     interactions between hydrogen bonded atoms are calculated.
  !
  !     The values assigned to the no(iat,jat) array are:
  !
  !              -5       use Watt's potential; only occurs if the
  !                       equivalence atom types for the two atoms are
  !                       'hw' and 'ow'
  !
  !      -2       no 10-12 parameters were included in the potential
  !                       file; calculate only coulombic interactions for
  !                       hydrogen bond.
  !
  !    <-10       10-12 parameters are to be used. abs(no(iat,jat)+10)
  !               is a pointer to the appropriate parameters in the
  !                       cpv array.
  !
  INTEGER IACTYP,IATYP,IHTYP,IP,IPRINT,IWHBK, &
       JATYP,JWHBK,KWHB,KNDPAR(8)
  character(len=4) IWHB,JWHB,ACPTYP(1),HYDTYP(1)
  LOGICAL FOUND
  !
  DATA IWHB/'hw  '/,JWHB/'ow  '/
  !
  DATA KWHB/-5/
  !
  !      WRITE (OUTU,6000)
  !
  !-----loop over all pairs of atom types and check if the non-bond
  !     equivalent kind of the 2 atoms are in the hydrogen bond donor
  !     and acceptor arrays. if they are, then:
  !     a) if ihydnb=0, i.e. no explicit 10-12 parameters were input,
  !        then set no(iat,jat)=-2
  !     b) if ihydnb>0, i.e. explicit 10-12 parameters were read in
  !        potential file, then set no(iat,jat)=(-ip-10), where ip
  !        is a pointer into the non-bond parameter arrays.
  loop200: DO IATYP=1,NATC
     !
     DO IHTYP=1,NHTYP
        IF (IATYP  ==  HTYP(IHTYP)) GO TO 50
     enddo
     cycle loop200
     !
     !-----now loop over the atom types and find all hydrogen bond acceptor
     !     atoms.
50   loop100: DO JATYP=1,NATC
        !
        DO IACTYP=1,NACTYP
           IF (JATYP  ==  ATYP(IACTYP)) GO TO 70
        enddo
        cycle loop100
        !
        !-----set the equivalent parameter pointer for this particular pair
        !     of hydrogen bonding atoms to the hydrogen bond type value.
        !      no(inb(iatyp),inb(jatyp)) = khb
        !      no(inb(jatyp),inb(iatyp)) = khb
70      IF (IHYDNB == 0) GO TO 95
        !
        !-----search for non-bond parameters for this hydrogen bond
        !
        loop82: DO IP=NNBDP-IHYDNB, NNBDP-1
           !
           !-----unpack kinds of atoms involved for this parameter
           !
           KNDPAR(1) = PNBID1(IP)
           KNDPAR(2) = PNBID2(IP)
           IF (KNDPAR(1) == IATYP.AND.KNDPAR(2).EQ.JATYP) GO TO 84
           IF (KNDPAR(1) == JATYP.AND.KNDPAR(2).EQ.IATYP) GO TO 84
        enddo loop82
        GO TO 95
84      CONTINUE
        MNO(IATYP,JATYP)=-IP-10
        MNO(JATYP,IATYP)=-IP-10
95      CONTINUE
        !
     enddo loop100
     !
  enddo loop200
  !
6020 FORMAT (1X,5X,A2,I5,' -- ',A2,I5)
END SUBROUTINE HBNDAS

SUBROUTINE FNDNXT(STRING,WORD,IPOS,LTERM,FOUND)
  ! Function
  !     This subroutine finds the next occurence of the terminator
  !     character lterm in the character variable string.
  !     If it is found then the logical variable found is set to .true.
  !     The first 10 non-blank characters are returned in the variable
  !     word, and the pointer ipos is moved to the string position
  !     immediately following the terminator.  The end of the character
  !     string can be specified with a terminator of lterm='#'.
  !
  use chm_kinds
  implicit none
  !
  !   Arguments
  INTEGER IPOS,J,LPOS,LASTCH
  INTEGER I,ITAB,JTAB
  CHARACTER(len=*) STRING,WORD
  CHARACTER(len=1) LTERM
  LOGICAL FOUND
  FOUND=.FALSE.
  !
  !   Fill the variable word with blanks so trailing characters will
  !   always be blanks if there are less than 10 characters.
  J=LEN(WORD)
  DO I=1,J
     WORD(I:I)=' '
  enddo
  !
  !   set lpos to the position of the last non-blank character in this
  !   line.
  !
  LPOS=LASTCH(STRING)
  !
  !   use itab to find the first non-blank character in the string
  !
  ITAB=IPOS
10 IF(STRING(ITAB:ITAB)  /=  ' ')GOTO 20
  ITAB=ITAB+1
  IF(ITAB > LPOS) RETURN
  GOTO 10
  !
  !   We now have the first non-blank character following ipos.
  !   Next start to look for the terminator.
  !
20 JTAB=ITAB
  IF(LTERM == '#') GOTO 35
30 IF(STRING(JTAB:JTAB) == LTERM) GOTO 40
  JTAB=JTAB+1
  IF(JTAB > LPOS) RETURN
  GOTO 30
35 JTAB=LPOS+1
40 IF(ITAB == JTAB) GOTO 45
  WORD=STRING(ITAB:JTAB-1)
45 IPOS=JTAB+1
  FOUND=.TRUE.
  RETURN
END SUBROUTINE FNDNXT


SUBROUTINE FNDATM(ATC,NATC,ANAME2,IPTR)
  !
  use chm_kinds
  implicit none
  INTEGER I,NATC,IPTR
  character(len=8) ATC(*)
  character(len=4) ANAME2

  character(len=8) ANAME

  ANAME = ANAME2 // "    "

  IF (NATC == 0) THEN
     NATC=1
     IPTR=1
     ATC(NATC)=ANAME
  ELSE
     IPTR=0
     DO I=1,NATC
        IF (ATC(I) == ANAME) IPTR=I
     enddo
     IF (IPTR == 0) THEN
        NATC=NATC+1
        ATC(NATC)=ANAME
        IPTR=NATC
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE FNDATM

FUNCTION LASTCH(STRING)
  !
  use chm_kinds
  implicit none
  INTEGER LASTCH,I,J,K
  !
  CHARACTER(len=*) STRING
  J=LEN(STRING)
  loop10: DO I=1,J
     K=J-I+1
     IF(STRING(K:K)  /=  ' ') GOTO 20
  enddo loop10
  K=0
20 LASTCH=K
  RETURN
END FUNCTION LASTCH



!================================================================
character(len=*) FUNCTION UPCASE(LINE)
  !
  use chm_kinds
  implicit none
  !
  SAVE
  character(len=*) LINE
  character(len=1) TABLE(0:255),LOCAS(26),UPCAS(26)
  LOGICAL NFIRST
  INTEGER I,LEN1,LEN2
  DATA NFIRST/.FALSE./
  DATA LOCAS/'a','b','c','d','e','f','g','h','i','j','k','l','m', &
       'n','o','p','q','r','s','t','u','v','w','x','y','z'/
  DATA UPCAS/'A','B','C','D','E','F','G','H','I','J','K','L','M', &
       'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'/
  IF(NFIRST)GOTO 100
  !================================================================
  !==   First time through this subroutine.                      ==
  !==   Set up the character translation table to translate all  ==
  !==   lower case ASCII characters to upper case.               ==
  !================================================================
  NFIRST=.TRUE.
  DO I=0,255
     TABLE(I)=CHAR(I)
  enddo
  DO I=1,26
     TABLE(ICHAR(LOCAS(I)))=UPCAS(I)
  enddo
100 LEN1=LEN(LINE)
  LEN2=LEN(UPCASE)
  DO I=1,MIN(LEN1,LEN2)
     UPCASE(I:I)=TABLE(ICHAR(LINE(I:I)))
  enddo
  IF(LEN2 > LEN1)UPCASE(LEN1+1:LEN2)=' '
  RETURN
END FUNCTION UPCASE

SUBROUTINE PARWTR_CFF(IUNIT)
  !
  use chm_kinds
  use consta
  use dimens_fcm
  use energym
  use exfunc
  use number
  use param
  use psf
  use rtf,only:armass
  use cff_fcm
  use stream
  implicit none
  !
  INTEGER IUNIT
  INTEGER I,IAT,NBCX,J,NBEXP,I1,J1,K1,L1
  !
  IF(IOLEV < 0) RETURN
  !
! BIOVIA Code Start
#if KEY_LICENSE==1
      IF (DECODE) NATC=-NATC
#endif
! BIOVIA Code End

  WRITE(IUNIT)NATC,NCB,NCBO,NCT,NCTO,NCP,NCPO,NCI,NCAA,NTTP,NNONBP, &
       NNBDP,FORCE,FRCVER
! BIOVIA Code Start
#if KEY_LICENSE==1
      IF (DECODE) NATC=-NATC
#endif
! BIOVIA Code End


  WRITE(IUNIT)(ATC(IAT),IAT=1,NATC),(INNB(IAT),IAT=1,NATC)
  WRITE(IUNIT)(IBE(IAT),IAT=1,NATC),(ITE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(IPE(IAT),IAT=1,NATC),(IOE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(IBAE(IAT),IAT=1,NATC),(INBAE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(ITAAE(IAT),IAT=1,NATC),(ITEAE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(IPCAE(IAT),IAT=1,NATC),(IPEAE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(IOCAE(IAT),IAT=1,NATC),(IOEAE(IAT),IAT=1,NATC)
  WRITE(IUNIT)(ARMASS(IAT),IAT=1,NATC),(ITC(I),I=1,NATC)
  IF (NCBO > 0) WRITE(IUNIT)(BID1(J),J=1,NCBO), &
       (BID2(J),J=1,NCBO),(CBOND1(I),I=1,NCBO), &
       (CBOND2(I),I=1,NCBO),(CBOND3(I),I=1,NCBO), &
       (CBOND4(I),I=1,NCBO),(KCB(I),I=1,NCB)
  IF (NCTO > 0) WRITE(IUNIT)(TID1(J),J=1,NCTO), &
       (TID2(J),J=1,NCTO),(TID3(J),J=1,NCTO), &
       (CTHET1(I),I=1,NCTO),(CTHET2(I),I=1,NCTO), &
       (CTHET3(I),I=1,NCTO),(CTHET4(I),I=1,NCTO), &
       (CTHET5(I),I=1,NCTO),(CTHET6(I),I=1,NCTO), &
       (CTHET7(I),I=1,NCTO),(KCT(I),I=1,NCT)
  IF (NCPO > 0) THEN
     WRITE(IUNIT)(PID1(J),J=1,NCPO),(PID2(J),J=1,NCPO), &
          (PID3(J),J=1,NCPO),(PID4(J),J=1,NCPO),(CPD(J),J=1,NCPO)
     WRITE(IUNIT)(CPHI1(I),I=1,NCPO), &
          (CPHI2(I),I=1,NCPO),(CPHI3(I),I=1,NCPO),(CPHI4(I),I=1,NCPO), &
          (CBP11(I),I=1,NCPO),(CBP12(I),I=1,NCPO), &
          (CBP13(I),I=1,NCPO),(CBP21(I),I=1,NCPO),(CBP22(I),I=1,NCPO), &
          (CBP23(I),I=1,NCPO),(CBP31(I),I=1,NCPO),(CBP32(I),I=1,NCPO), &
          (CBP33(I),I=1,NCPO),(CTP11(I),I=1,NCPO),(CTP12(I),I=1,NCPO), &
          (CTP13(I),I=1,NCPO),(CTP21(I),I=1,NCPO),(CTP22(I),I=1,NCPO), &
          (CTP23(I),I=1,NCPO),(CSGN1(I),I=1,NCPO),(CSGN2(I),I=1,NCPO), &
          (CSGN3(I),I=1,NCPO),(CBB2(I),I=1,NCPO),(KCP(I),I=1,NCP)
  ENDIF
  IF (NCI > 0) THEN
     WRITE(IUNIT)(OPID1(J),J=1,NCI),(OPID2(J),J=1,NCI), &
          (OPID3(J),J=1,NCI),(OPID4(J),J=1,NCI), &
          (COPLN1(I),I=1,NCI),(KCI(I),I=1,NCI)
  ENDIF
  IF (NTTP > 0) WRITE(IUNIT)(TTID1(J),J=1,NTTP), &
       (TTID2(J),J=1,NTTP),(TTID3(J),J=1,NTTP),(TTID4(J),J=1,NTTP), &
       (CTT(I),I=1,NTTP)
  IF (NNBDP > 0) THEN
     WRITE(IUNIT)(PNBID1(J),J=1,NNBDP),(PNBID2(J),J=1,NNBDP)
     WRITE(IUNIT)(CNB1(I),I=1,NNBDP),(CNB2(I),I=1,NNBDP), &
          (CNB1(I+MAXCN),I=1,NNBDP),(CNB2(I+MAXCN),I=1,NNBDP),(CNB5(I),I=1,NNBDP)
  ENDIF
  WRITE(IUNIT)((MNO(I,J),I=1,NATC),J=1,NATC)
  return
end SUBROUTINE PARWTR_CFF

! BIOVIA Code Start
#if KEY_LICENSE==1
SUBROUTINE GETFIRST(IUNIT,DECODE)

  use stream

  IMPLICIT NONE
  CHARACTER*256 LINE
  CHARACTER*256 CODED
  CHARACTER*120 MESSAGE
  INTEGER LENGTH,LASTCH,IUNIT
  LOGICAL DECODE
  DECODE = .FALSE.
 !
 ! Read the first line of the forcefield file.  If the line starts
 ! with !BIOSYM then the file is not encoded.  Otherwise, try
 ! decoding it.  If it still doesn't start with !BIOSYM then the
 ! file is not a valid cff forcefield file.
 !
  READ(IUNIT,'(A256)') CODED
  IF (CODED(1:7).NE.'!BIOSYM') THEN
     LENGTH = LASTCH(CODED)+1
     CALL DECODE_FF(CODED,LINE,LENGTH)
     IF (LINE(1:7).NE.'!BIOSYM') THEN
        WRITE(MESSAGE,'(A)')'Problem reading forcefield file'
        CALL WRNDIE(-5,'<GETFIRST>', MESSAGE )
        STOP
     ENDIF
 !
 ! The forcefield file is encoded, check for a valid license.
 !
     CALL checkout_license_cff()
     DECODE=.TRUE.
  ENDIF
  RETURN
END SUBROUTINE GETFIRST
#endif  /* LICENSE */

#endif
! BIOVIA Code End
SUBROUTINE NULL_parrdr_CFF
 RETURN
END SUBROUTINE NULL_parrdr_CFF

