module parmiom
  private  parrdr,parrdr2,parrdr3,addnbf,chckrep
contains
  SUBROUTINE PARMIO(IUNIT,NTITL,TITLE,ICARD,JUNIT,NATCT,ATCT,QAPPE &
#if KEY_LIBRARY == 1
       , flex &
#endif /* LIBRARY */
       )
    !
    !     THIS ROUTINE ALLOCATES SPACE AND CALLS PARRD3, PARRDR_MMFF or
    !     PARRDR_CFF
    !
    use chm_kinds
    use dimens_fcm
    use memory
    use comand
    use inbnd
    use param
    use param_store, only: set_param
    use stream
    use string
    use ffieldm
#if KEY_CMAP==1
    use cmapm, only: maxctp, nctp
#endif
#if KEY_CFF==1
    use energym
#endif
#if KEY_MMFF==1
    use mmffm
#endif
    implicit none
#if KEY_CFF==1
    INTEGER,allocatable,dimension(:) :: work2
#endif
    !
    CHARACTER(len=*) TITLE(*)
    INTEGER IUNIT,NTITL,ICARD,JUNIT
    INTEGER NATCT
    CHARACTER(len=*) ATCT(*)
    LOGICAL QAPPE
    ! local
    INTEGER I,MLEN
    LOGICAL QFLEX,BAD

#if KEY_LIBRARY == 1
    logical, optional :: flex
#endif /* LIBRARY */
 
    !yw...11-Aug-2003 initialize the local QFLEX
    QFLEX = .FALSE.

#if KEY_LIBRARY == 1
    if (present(flex)) qflex = flex
#endif /* LIBRARY */

#if KEY_CFF==1 || KEY_MMFF==1
    if(ffield == charmm .or. ffield == amberffn) then
#endif
       !-----------------------------------------------------------------------
       !
       ! Section for CHARMM type force field
       !
#if KEY_LIBRARY == 1
      if (.not. present(flex)) &
#endif /* LIBRARY */
          QFLEX=(INDXA(COMLYN,COMLEN,'FLEX') /= 0)

       IF(ICARD /= 0) THEN
          IF(QAPPE) THEN
             IF(.NOT.QNBFIX) THEN
                CALL WRNDIE(-2,'<PARMIO>', &
                     'Cannot use APPEnd option after reading an old binary file')
                RETURN
             ENDIF
          ENDIF
          IF(QFLEX) THEN
#if KEY_FLEXPARM==1
             IF(QAPPE) THEN
                IF(.NOT.QFLXPARM) THEN
                   CALL WRNDIE(-2,'<PARMIO>', &
                        'Cannot use FLEX option with APPEND if prior not flexible')
                   RETURN
                ENDIF
             ELSE
                NATC=0
                NACTEQV=0
                QFLXPARM=.TRUE.
                DO I=1,MAXATC
                   ATC(I)=' '
                   ATCCNT(I)=0
                ENDDO
             ENDIF
#else /**/
             CALL WRNDIE(-4,'<PARMIO>','Flexible option is not compiled')
             RETURN
#endif
          ELSE
             IF(NATCT <= 0) THEN
                CALL WRNDIE(-2,'<PARMIO>', &
                     'Cannot read a card parameter file without a topology file')
                RETURN
             ENDIF
             ! check to see if the append option is allowed
             IF(QAPPE) THEN
                IF(NATC == 0) THEN
                   CALL WRNDIE(-2,'<PARMIO>', &
                        'Cannot use APPEnd option for initial read')
                   RETURN
                ENDIF
#if KEY_FLEXPARM==1
                IF(QFLXPARM) THEN
                   CALL WRNDIE(-2,'<PARMIO>', &
                        'Must use FLEX option with APPEND if prior was flexible')
                   RETURN
                ENDIF
#endif
                BAD=(NATC /= NATCT)
                DO I=1,NATC
                   IF(ATC(I) /= ATCT(I)) BAD=.TRUE.
                ENDDO
                IF(BAD) THEN
                   CALL WRNDIE(-2,'<PARMIO>', &
                        'Cannot use APPEnd option when RTF MASSES list is modified')
                   RETURN
                ENDIF
             ELSE
                !          copy NATC and ATC from the RTF
                NATC=NATCT
                DO I=1,NATC
                   ATC(I)=ATCT(I)
                   ATCCNT(I)=0
                ENDDO
                DO I=NATC+1,MAXATC
                   ATC(I)=' '
                   ATCCNT(I)=0
                ENDDO
             ENDIF
             QFLXPARM=.FALSE.
          ENDIF ! (QFLEX)
       ELSE
          IF(QAPPE) THEN
             CALL WRNDIE(-2,'<PARMIO>', &
                  'Cannot read a binary parameter file with APPEnd option')
             QAPPE=.FALSE.
          ENDIF
          IF(QFLEX) THEN
             CALL WRNDIE(-2,'<PARMIO>', &
                  'Cannot read a binary parameter file with flexible option')
             RETURN
          ENDIF
          QFLXPARM=.FALSE.
       ENDIF
       !
       MLEN=MAX(MAXATC,MAXCB,MAXCT,MAXCP,MAXCI,MAXCH)
#if KEY_CMAP==1
       IF(MLEN < MAXCTP) MLEN=MAXCTP
#endif
       CALL PARRDR(IUNIT,NTITL,TITLE,ICARD,JUNIT,NATCT,ATCT, &
            MLEN,COMLYN,COMLEN,QAPPE)
       MLEN=MAX(NATC,NCB,NCT,NCP,NCI,NCH)
#if KEY_CMAP==1
       IF(MLEN < NCTP) MLEN=NCTP
#endif
       CALL PARRDR2(ICARD,MLEN)
       CALL PARRDR3(ICARD,JUNIT)
       !
       !-----------------------------------------------------------------------
#if KEY_CFF==1
       !
       ! Section for CFF type force field
       !
    ELSEIF(FFIELD == CFF) THEN
       IF(QAPPE) THEN
          CALL WRNDIE(-4,'<PARMIO>', &
               'Cannot use APPEND with CFF force field')
          RETURN
       ENDIF
#if KEY_FLEXPARM==1
       IF(QFLEX) THEN
          CALL WRNDIE(-4,'<PARMIO>', &
               'FLEX option not compatible with CFF')
          RETURN
       ENDIF
#endif
       !
       MLEN=MAX(MAXATC,MAXCB,MAXCT,MAXCP,MAXCI,MAXCH)
       CALL PARRDR_CFF(IUNIT,ICARD,MLEN)

#endif
       !-----------------------------------------------------------------------
       !
       ! Section for MMFF type force field
       !
#if KEY_MMFF==1
    elseif(FFIELD == MMFF) then
       IF(QAPPE) THEN
          CALL WRNDIE(-4,'<PARMIO>', &
               'Cannot use APPEND with MMFF force field')
          RETURN
       ENDIF
#if KEY_FLEXPARM==1
       IF(QFLEX) THEN
          CALL WRNDIE(-4,'<PARMIO>', &
               'FLEX option not compatible with MMFF')
          RETURN
       ENDIF
#endif
       !brb..07-FEB-99 Protect MMFF specific parameter reading options.
       if(INDXA(COMLYN,COMLEN,'MMFF') /= 0) then
          CALL PARRDR_MMFF(IUNIT,COMLYN,COMLEN)
          CALL set_param('NCSB',NCSB)   ! strech-bend
          CALL set_param('NCOOP',NCOOP) ! out-of-plane
          CALL set_param('NCQ',NCQ)     ! bond charge increments
       else
          call wrndie(-5,'<PARMIO>','MMFF file not specified')
       endif
#endif
       !-----------------------------------------------------------------------
#if KEY_CFF==1 || KEY_MMFF==1
    ENDIF
#endif
    !
    CALL set_param('NATC',NATC)
#if KEY_FLEXPARM==1
    CALL set_param('NACTEQV',NACTEQV)
#endif
    CALL set_param('NCB',NCB)
    CALL set_param('NCT',NCT)
    CALL set_param('NCP',NCP)
    CALL set_param('NCI',NCI)
    CALL set_param('NCH',NCH)
    CALL set_param('NCN',NCN)
    CALL set_param('NATVDW',NATVDW)
#if KEY_CMAP==1
    CALL set_param('NCTP',NCTP)
#endif
    !
    RETURN
  END subroutine parmio

  SUBROUTINE PARWTR(IUNIT,NTITL,TITLE,ICNTRL,IPRINT,QUSED)
    !
    !     This routine writes a pararmeter file module or prints the
    !     parameters.
    !
    !       IUNIT  <- Unit to print/write to
    !       NTITL  <- Number of title lines
    !       TITLE  <- Title lines
    !       ICNTRL <- Control array (integer)
    !       IPRINT <- 0=binary, 1=print
    !       QUSED  <- Only print parameters actually used in the most
    !                 recent energy evaluation (PSF dependent)
    !
    use chm_kinds
    use dimens_fcm
    use param
    use version
    use number
    use consta
    use defltsm
#if KEY_CMAP==1
    use cmapm
#endif
#if KEY_CFF==1 || KEY_MMFF==1
    use ffieldm
#endif
    use stream
#if KEY_QCHEM==1
    use psf
    use gamess_fcm,only:QQEWALD,qcffatmtyp,QQCHEM
    use rtf, only:mac
    use memory
#endif
    implicit none
    CHARACTER(len=*) TITLE(*)
    INTEGER NTITL,IUNIT,IPRINT
    INTEGER ICNTRL(20)
    LOGICAL QUSED
    !
    INTEGER  :: NBC(MAXATC),NBCCNT(MAXATC),NBCWC(MAXATC)
    !
    !
    LOGICAL LFLIP,QALL
    INTEGER I1,J1,K1,L1,ITEMP,I,J,K,L,M,N,tmpI1,tmpJ1,tmpK1,tmpL1,numqcatmtyp
    real(chm_real)  A14,B14,VDW14,EMIN14,A,B,VDW,EMIN,R6,R614
    integer(chm_int8) :: KCPI,KCII,NATC2,intg
    !
    CHARACTER(len=4) :: HDR='PARA'
    CHARACTER(len=8) :: AX='X   ',AI,AJ,AK,AL,AZ(8)
    CHARACTER(len=1) BI,BJ
    CHARACTER(len=120) LINE
#if KEY_MMFF==1
    INTEGER MCLASS
    CHARACTER(len=80) SCRTCH
#endif
    !
#if KEY_QCHEM==1
!   integer,allocatable,dimension(:) :: qcffatmtyp
    integer,allocatable,dimension(:) :: mapqcffatm
#endif
    !
#if KEY_MMFF==1
    IF (FFIELD == MMFF) THEN
       MCLASS = 0
    ENDIF
#endif
#if KEY_QCHEM==1
    IF(QQCHEM) THEN
       open(92,FILE='MyForceField.prm',status='unknown')
       write(92,'(A)')'$force_field_params'
       J=0
       do I=1,NATC
         if(ATCCNT(I).gt.0) then
           J=J+1
         endif
       enddo
       allocate(qcffatmtyp(0:J-1))
       allocate(mapqcffatm(0:J-1))
       J=0
       K=-1
       do I=1,NATC
         if(ATCCNT(I).gt.0) then
           qcffatmtyp(J)=I
           mapqcffatm(J)=K
           J=J+1
           K=K-1
         endif
       enddo
       write(92,'(A,I8)')'NumAtomTypes ',J
       numqcatmtyp=J
    ENDIF
#endif
    !
    IF(IPRINT /= 0) GOTO 500
    !
    !-----------------------------------------------------------------------
    !     write parameter file module (binary)
    !
    IF(IOLEV <= 0) RETURN
    !
#if KEY_FLEXPARM==1
    IF (QFLXPARM) THEN
       CALL WRNDIE(0,'<PARWTR>', &
            'Cannot write flexible paramters (yet).')
       RETURN
    ENDIF
#endif
#if KEY_CFF==1
    IF (FFIELD == CFF) THEN
       CALL PARWTR_CFF(IUNIT)
       RETURN
    ENDIF
#endif
    ICNTRL(1)=VERNUM
    DO I=2,20
       ICNTRL(I)=0
    ENDDO
    !
    WRITE(IUNIT) HDR,ICNTRL
    CALL WRTITL(TITLE,NTITL,IUNIT,-1)
    WRITE(IUNIT) NATC,NCB,NCT,NCP,NCI,NCN,NCH
    WRITE(IUNIT) (ATC(I),I=1,NATC)
    WRITE(IUNIT) (ITC(I),ALP(I),EFF(I),VDWR(I),I=1,NATC), &
         (ALP(I),EFF(I),VDWR(I),I=MAXATC+1,MAXATC+NATC)
    WRITE(IUNIT) (KCB(I),I=1,NCB),(CBC(I),I=1,NCB),(CBB(I),I=1,NCB)
    WRITE(IUNIT) (KCT(I),I=1,NCT),(CTC(I),I=1,NCT),(CTB(I),I=1,NCT)
    WRITE(IUNIT) (CTUB(I),I=1,NCT),(CTUC(I),I=1,NCT)
    WRITE(IUNIT) (KCP(I),I=1,NCP),(CPC(I),I=1,NCP), &
         (CPD(I),I=1,NCP),(CPB(I),I=1,NCP)
    WRITE(IUNIT) (KCI(I),I=1,NCI),(CIC(I),I=1,NCI), &
         (CID(I),I=1,NCI),(CIB(I),I=1,NCI)
    WRITE(IUNIT) (I,I=1,NCN),(CNBA(I),I=1,NCN),(CNBB(I),I=1,NCN), &
         (CNBA(I+MAXCN),I=1,NCN), &
         (CNBB(I+MAXCN),I=1,NCN)
    WRITE(IUNIT)  NBFIXN,(NBFIXI(1,I),NBFIXI(2,I),I=1,NBFIXN), &
         (NBFIXR(1,I),NBFIXR(2,I),NBFIXR(3,I),NBFIXR(4,I),I=1,NBFIXN)
    WRITE(IUNIT) (HBEXPN(I),I=1,4),(KCH(I),I=1,NCH), &
         (CHBA(I),I=1,NCH),(CHBB(I),I=1,NCH)
    WRITE(IUNIT) DEFLTS,DFUSED
    RETURN
    !
    !-----------------------------------------------------------------------
    !     print parameters to output
    !
500 CONTINUE
    !
    IF(IOLEV <= 0  .AND. IUNIT /= OUTU) RETURN
    IF(PRNLEV <= 1 .AND. IUNIT == OUTU) RETURN
    !
    QALL=.NOT.QUSED
    !
    !------
    ! atoms
    WRITE(IUNIT,10)
10  FORMAT(/9X,'ATOM PROPERTIES'// &
         '  #  CODE ATOM',3X,'POLARIZABILITY', &
         '  N(EFF)     RADIUS    C6(L-R)    COUNT'/)
    !
    NATC2=NATC*(NATC+1)/2
    !
#if KEY_QCHEM==1
    IF(QQCHEM) THEN
    M=-1
    N=-1
    K=numqcatmtyp
    typloop: do J=0,K-1
      vdwloop: do L=1,NATC
        if(ATCCNT(L) > 0) then
          if(L .gt. M) then
            natloop: do I=1,NATOM
              if(IAC(I) == qcffatmtyp(J)) then
                  write(92,'(A,I8,1X,3F10.4)')'AtomType ',mapqcffatm(J),CG(I),VDWR(ITC(L)),eff(itc(L))
                  M=L
                  exit vdwloop
              endif
            enddo natloop
          endif
        endif
      enddo vdwloop
    enddo typloop
    ENDIF
#endif

    DO I=1,NATC
       IF(ATC(I) /= ' ') THEN
          IF(ITC(I) /= 0) THEN
             IF(QALL.OR.ATCCNT(I) > 0) THEN
                WRITE(IUNIT,15) I,ITC(I),ATC(I)(1:idleng),ALP(ITC(I)), &
                     EFF(ITC(I)),VDWR(ITC(I)),C6DEF(ITC(I)),ATCCNT(I)
15              FORMAT(I4,1X,I3,2X,A,6X,4F11.4,I6)
             ENDIF
          ELSE
             IF(QALL.OR.ATCCNT(I) > 0) THEN
                WRITE(IUNIT,15) I,0,ATC(I)(1:idleng), &
                     ZERO,ZERO,ZERO,ZERO,ATCCNT(I)
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    !------
    ! equiv groups
#if KEY_FLEXPARM==1 /*flex_group*/
    IF(QFLXPARM) THEN
       WRITE(IUNIT,11)
11     FORMAT(/9X,'EQUIVALENCE GROUP PROPERTIES'// &
            '   #   GROUP   COUNT'/)
       !
       DO I=1,NACTEQV
          L1=0
          DO J=1,NATC
             IF(ACTEQVM(J,I) /= 0) THEN
                L1=L1+ATCCNT(J)
             ENDIF
          ENDDO

          IF(QALL.OR.L1 > 0) THEN
             WRITE(IUNIT,16) I,ACTEQV(I)(1:idleng),L1
16           FORMAT(I4,3X,A,I8)
             !
             LINE=' '
             K1=1
             DO J=1,NATC
                IF(ACTEQVM(J,I) /= 0 .AND. ATC(J)/=' ') THEN
                   LINE(K1:K1+idleng-1)=ATC(J)(1:idleng)
                   K1=K1+idleng+1
                   IF(K1 > 80) THEN
                      WRITE(IUNIT,38) LINE
38                    FORMAT(10X,' - ',A)
                      LINE=' '
                      K1=1
                   ENDIF
                ENDIF
             ENDDO
             IF(K1 > 1) WRITE(IUNIT,38) LINE(1:K1)
          ENDIF
       ENDDO
    ENDIF
#endif /* (flex_group)*/
    !
    !------
    ! bonds
    !
#if KEY_MMFF==1
    if(ffield == charmm .or. ffield == amberffn) then
#endif
       WRITE(IUNIT,65)
65     FORMAT(/9X,'BOND PARAMETERS', &
            //8X,'#',3X,'B O N D',10X,'CODE',6X,'KB',7X,'BO      COUNT'/)
#if KEY_MMFF==1
    elseif(FFIELD == MMFF) then
       WRITE(IUNIT,66)
66     FORMAT(/9X,'BOND PARAMETERS', &
            //8X,'#',3X,'B O N D',10X,'CODE',6X,'KB',7X,'BO',5X,'CLASS'/)
    else
       write(SCRTCH,'(a,i5)') 'Unknown FFIELD = ',FFIELD
       CALL WRNDIE(-2,'<PARWTR>',SCRTCH(:22))
       RETURN
    endif
#endif
    !
    DO I=1,NCB
#if KEY_MMFF==1
       if(ffield == charmm .or. ffield == amberffn) then
#endif
#if KEY_FLEXPARM==1 /*flex_bond*/
          IF(QFLXPARM) THEN
             IF(CBAI(I) > 0) THEN
                AI=ATC(CBAI(I))
             ELSE
                AI=ACTEQV(-CBAI(I))
             ENDIF
             IF(CBAJ(I) > 0) THEN
                AJ=ATC(CBAJ(I))
             ELSE
                AJ=ACTEQV(-CBAJ(I))
             ENDIF
          ELSE
#endif /* (flex_bond)*/
             I1=SQRT(TWO*KCB(I))+HALF
             J1=KCB(I)-I1*(I1-1)/2
             AI=ATC(I1)
             AJ=ATC(J1)
#if KEY_FLEXPARM==1
          ENDIF
#endif
          IF(QALL.OR.ICBCNT(I) > 0) THEN
             WRITE(IUNIT,70) I,AI(1:idleng),AJ(1:idleng), &
                  KCB(I),CBC(I),CBB(I),ICBCNT(I)
70           FORMAT(6X,I4,2X,A,' - ',A,3X,I7,F10.1,F10.3,I8)

#if KEY_QCHEM==1
             IF(QQCHEM) THEN
                n=0
                K=numqcatmtyp
                do j=1,natc
                  if(atccnt(j).gt.0) then
                    if(ATC(j) .eq. AI(1:idleng)) then
                       if((ai(1:1) .eq. 'H') .and. (aj(1:1) .eq. 'H')) then
                          !don't write shake bonds
                       else
                          do n=0,K-1
                             if (qcffatmtyp(n) .eq. I1) then
                               tmpI1 = mapqcffatm(n)
                             endif
                             if(qcffatmtyp(n) .eq. J1) then
                               tmpJ1 = mapqcffatm(n)
                             endif
                          enddo
                          write(92,'(A,2I8,2F10.4)')'Bond',tmpI1,tmpJ1,CBC(I),CBB(I)
                       endif
                    endif
                  endif
                enddo
             ENDIF
#endif
          ENDIF
#if KEY_MMFF==1
       elseif(FFIELD == MMFF) then
          call BONDCON('GET',I1,J1,MCLASS,KCB(I))
          WRITE(IUNIT,71) I,ATC(I1)(1:idleng),ATC(J1)(1:idleng), &
               KCB(I),CBC(I),CBB(I),MCLASS
71        FORMAT(6X,I4,2X,A,' - ',A,3X,I7,F10.1,F10.3,I5)
       endif
#endif
    ENDDO
    !
    !------
    ! angles
#if KEY_MMFF==1
    if(ffield == charmm .or. ffield == amberffn) then
#endif
       WRITE(IUNIT,75)
75     FORMAT(/9X,'BOND ANGLE PARAMETERS', &
            //8X,'#   A N G L E',14X,'CODE',6X,'KT',7X,'TO',7X, &
            'UK',8X,'U0      COUNT'/)
#if KEY_MMFF==1
    elseif(FFIELD == MMFF) then
       WRITE(IUNIT,76)
76     FORMAT(/9X,'BOND ANGLE PARAMETERS', &
            //8X,'#   A N G L E',14X,'CODE',6X,'KT',7X,'TO',7X, &
            'UK',8X,'U0',5X,'CLASS'/)
    endif
#endif
    DO I=1,NCT
#if KEY_MMFF==1
       if(ffield == charmm .or. ffield == amberffn) then
#endif
#if KEY_FLEXPARM==1 /*flex_angle*/
          IF(QFLXPARM) THEN
             IF(CTAI(I) > 0) THEN
                AI=ATC(CTAI(I))
             ELSE
                AI=ACTEQV(-CTAI(I))
             ENDIF
             IF(CTAJ(I) > 0) THEN
                AJ=ATC(CTAJ(I))
             ELSE
                AJ=ACTEQV(-CTAJ(I))
             ENDIF
             IF(CTAK(I) > 0) THEN
                AK=ATC(CTAK(I))
             ELSE
                AK=ACTEQV(-CTAK(I))
             ENDIF
          ELSE
#endif /* (flex_angle)*/
             ITEMP=(KCT(I)-1)/NATC
             J1=KCT(I)-ITEMP*NATC
             I1=SQRT(TWO*ITEMP)+HALF
             K1=ITEMP-I1*(I1-1)/2
             AI=ATC(I1)
             AJ=ATC(J1)
             AK=ATC(K1)
#if KEY_FLEXPARM==1
          ENDIF
#endif
          !
          IF(QALL.OR.ICTCNT(I) > 0) THEN
             WRITE(IUNIT,80) I,AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
                  KCT(I),CTC(I),CTB(I)*RADDEG, &
                  CTUC(I),CTUB(I),ICTCNT(I)
80           FORMAT(6X,I4,2X,A,' - ',A,' - ',A,I9,F9.2, &
                  F10.4,F10.4,F10.4,4I8)
#if KEY_QCHEM==1
             IF(QQCHEM) THEN
                do j=1,NATC
                  if(ATCCNT(j).gt.0) then
                    if(ATC(j) .eq. AI(1:idleng)) then
                       do n=0,K-1
                          if (qcffatmtyp(n) .eq. I1) then
                            tmpI1 = mapqcffatm(n)
                          endif
                          if(qcffatmtyp(n) .eq. J1) then
                            tmpJ1 = mapqcffatm(n)
                          endif
                          if(qcffatmtyp(n) .eq. K1) then
                            tmpK1 = mapqcffatm(n)
                          endif
                       enddo
                       write(92,'(A,3I8,2F10.4)')'Angle',tmpI1,tmpJ1,tmpK1,CTC(I),CTB(I)*RADDEG
                    endif
                  endif
                enddo
             ENDIF
#endif
          ENDIF
#if KEY_MMFF==1
       elseif(FFIELD == MMFF) then
          CALL THETCON('GET',I1,J1,K1,MCLASS,KCT(I))
          WRITE(IUNIT,81) I,AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
               KCT(I),CTC(I),CTB(I)*RADDEG,CTUC(I),CTUB(I),MCLASS
81        FORMAT(6X,I4,2X,A,' - ',A,' - ',A,I9,F9.2, &
               F10.4,F10.4,F10.4,I5)
       endif
#endif
    ENDDO
    !
    !------
    ! dihedral
    !
    WRITE(IUNIT,85)
85  FORMAT(/9X,'DIHEDRAL ANGLE PARAMETERS', &
         //8X,'#',3X,'T O R S I O N',17X,'CODE',6X,'KP', &
         9X,'N',7X,'DELTA   COUNT'/)
    !
    DO I=1,NCP
#if KEY_FLEXPARM==1 /*flex_angle*/
       IF(QFLXPARM) THEN
          IF(CPAI(I) > 0) THEN
             AK=ATC(CPAI(I))
          ELSE
             AK=ACTEQV(-CPAI(I))
          ENDIF
          IF(CPAJ(I) > 0) THEN
             AI=ATC(CPAJ(I))
          ELSE
             AI=ACTEQV(-CPAJ(I))
          ENDIF
          IF(CPAK(I) > 0) THEN
             AJ=ATC(CPAK(I))
          ELSE
             AJ=ACTEQV(-CPAK(I))
          ENDIF
          IF(CPAL(I) > 0) THEN
             AL=ATC(CPAL(I))
          ELSE
             AL=ACTEQV(-CPAL(I))
          ENDIF
       ELSE
#endif /* (flex_angle)*/
          AK=AX
          AL=AX
          KCPI=KCP(I)
          IF(KCPI <= NATC2) GOTO 35
          LFLIP=.FALSE.
          IF(KCPI > NATC2*NATC2) THEN
             KCPI=KCPI-NATC2*NATC2
             LFLIP=.TRUE.
          ENDIF
          I1=KCPI/NATC2
          KCPI=KCPI-I1*NATC2
          K1=SQRT(TWO*I1)+HALF
          L1=I1-K1*(K1-1)/2
          AK=ATC(K1)
          AL=ATC(L1)
          IF(LFLIP) THEN
             AL=ATC(K1)
             AK=ATC(L1)
          ENDIF
35        I1=SQRT(TWO*KCPI)+HALF
          J1=KCPI-I1*(I1-1)/2
          AI=ATC(I1)
          AJ=ATC(J1)
#if KEY_FLEXPARM==1
       ENDIF
#endif
       !
       IF(QALL.OR.ICPCNT(I) > 0) THEN
          WRITE(IUNIT,90) I,AK(1:idleng),AI(1:idleng),AJ(1:idleng), &
               AL(1:idleng),KCP(I),CPC(I),CPD(I),CPB(I)*RADDEG, &
               ICPCNT(I)
90        FORMAT(6X,I4,2X,A,' - ',A,' - ',A,' - ', &
               A,I9,F10.2,I9,F11.4,I8)
#if KEY_QCHEM==1
          IF(QQCHEM) THEN
             do j=1,NATC
               if(ATCCNT(j).gt.0) then
                 if(ATC(j) .eq. AI(1:idleng)) then
                    do n=0,K-1
                       if (qcffatmtyp(n) .eq. I1) then
                         tmpI1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. J1) then
                         tmpJ1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. K1) then
                         tmpK1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. L1) then
                         tmpL1 = mapqcffatm(n)
                       endif
                    enddo
                    write(92,'(A,4I8,2F10.4,I4)')'Torsion', &
                    tmpK1,tmpI1,tmpJ1,tmpL1,CPC(I),CPB(I)*RADDEG,CPD(I)
                 endif
               endif
             enddo
          ENDIF
#endif
       ENDIF
    ENDDO
    !
    !------
    ! impropers
    !
    WRITE(IUNIT,95)
95  FORMAT(/9X,'IMPROPER TORSION PARAMETERS', &
         //8X,'#',3X,'T O R S I O N',17X,'CODE',6X,'KP', &
         9X,'N',7X,'DELTA   COUNT'/)
    DO I=1,NCI
#if KEY_FLEXPARM==1 /*flex_angle*/
       IF(QFLXPARM) THEN
          IF(CIAI(I) > 0) THEN
             AI=ATC(CIAI(I))
          ELSE
             AI=ACTEQV(-CIAI(I))
          ENDIF
          IF(CIAJ(I) > 0) THEN
             AK=ATC(CIAJ(I))
          ELSE
             AK=ACTEQV(-CIAJ(I))
          ENDIF
          IF(CIAK(I) > 0) THEN
             AL=ATC(CIAK(I))
          ELSE
             AL=ACTEQV(-CIAK(I))
          ENDIF
          IF(CIAL(I) > 0) THEN
             AJ=ATC(CIAL(I))
          ELSE
             AJ=ACTEQV(-CIAL(I))
          ENDIF
       ELSE
#endif /* (flex_angle)*/
          AK=AX
          AL=AX
          KCII=KCI(I)
          IF (KCII > NATC2) THEN
             !           no wildcards
             LFLIP=.FALSE.
             IF(KCII > NATC2*NATC2) THEN
                KCII=KCII-NATC2*NATC2
                LFLIP=.TRUE.
             ENDIF
             I1=KCII/NATC2
             KCII=KCII-I1*NATC2
             K1=SQRT(TWO*I1)+HALF
             L1=I1-K1*(K1-1)/2
             AK=ATC(K1)
             AL=ATC(L1)
             IF(LFLIP) THEN
                AL=ATC(K1)
                AK=ATC(L1)
             ENDIF
             I1=SQRT(TWO*KCII)+HALF
             J1=KCII-I1*(I1-1)/2
             AI=ATC(I1)
             AJ=ATC(J1)
             !
          ELSEIF (KCII > 0) THEN
             !           wildcard type  A - X - X - B
             I1=SQRT(TWO*KCII)+HALF
             J1=KCII-I1*(I1-1)/2
             AI=ATC(I1)
             AJ=ATC(J1)
             !
          ELSEIF (KCII < -NATC**3-NATC**2) THEN
             !           wildcard type X - A - B - X
             KCII=-(NATC**3+NATC**2+KCII)
             K1=SQRT(TWO*KCII)+HALF
             L1=KCII-K1*(K1-1)/2
             AK=ATC(K1)
             AL=ATC(L1)
             AI=AX
             AJ=AX
             !
          ELSEIF (KCII < -NATC*NATC) THEN
             !           wildcard type  X - A - B - C
             AI=AX
             KCII=-KCII
             J1=(KCII/(NATC*NATC))
             KCII=KCII-NATC*NATC*(J1)
             K1=(KCII/NATC)+1
             L1=KCII-NATC*(K1-1)
             AK=ATC(K1)
             AL=ATC(L1)
             AJ=ATC(J1)
             !
          ELSE
             !           wildcard type X - X - A - B
             AI=AX
             AK=AX
             I1=-KCI(I)
             L1=(I1/NATC)+1
             J1=I1-NATC*(L1-1)
             AL=ATC(L1)
             AJ=ATC(J1)
          ENDIF
#if KEY_FLEXPARM==1
       ENDIF
#endif
       !
       IF(QALL.OR.ICICNT(I) > 0) THEN
          WRITE(IUNIT,90) I,AI(1:idleng),AK(1:idleng),AL(1:idleng), &
               AJ(1:idleng),KCI(I),CIC(I),CID(I), &
               CIB(I)*RADDEG,ICICNT(I)
#if KEY_QCHEM==1
          IF(QQCHEM) THEN
             do j=1,NATC
               if(ATCCNT(j).gt.0) then
                 if(ATC(j) .eq. AI(1:idleng)) then
                    do n=0,K-1
                       if (qcffatmtyp(n) .eq. I1) then
                         tmpI1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. J1) then
                         tmpJ1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. K1) then
                         tmpK1 = mapqcffatm(n)
                       endif
                       if(qcffatmtyp(n) .eq. L1) then
                         tmpL1 = mapqcffatm(n)
                       endif
                    enddo
                    write(92,'(A,4I8,2F10.4,I4)')'Improper', &
                    tmpI1,tmpK1,tmpL1,tmpJ1,CIC(I),CIB(I)*RADDEG,CID(I)
                 endif
               endif
             enddo
          ENDIF
#endif
       ENDIF
    ENDDO

#if KEY_QCHEM==1
    IF(QQCHEM) then
      write(92,'(A)')'$end'
      flush(92)
      close(92)
    endif
#endif

    !------
    ! cmap
    !
#if KEY_CMAP==1
    IF (NCTP > 0) THEN
       WRITE(IUNIT,97)
97     FORMAT(/9X,'CROSSTERM MAPS',//8X,'#',4X,'CODE',6X, &
            'N    COUNT')
       DO I=1,NCTP
          IF(QALL.OR.ICTPCNT(I) > 0) THEN
             WRITE(IUNIT,96) I,KCTP(I),GSCTP(I),GSCTP(I),ICTPCNT(I)
96           FORMAT(5X,I4,2X,I9,1X,I4,' x',I4,I8)
#if KEY_FLEXPARM==1
             IF(QFLXPARM) THEN
                DO J=1,8
                   IF(CTPA(I,J) > 0) THEN
                      AZ(J)=ATC(CTPA(I,J))
                   ELSE
                      AZ(J)=ACTEQV(-CTPA(I,J))
                   ENDIF
                ENDDO
                WRITE(IUNIT,98) (AZ(J)(1:idleng),J=1,8)
98              FORMAT(10X,A,'-',A,'-',A,'-',A,'==', &
                     A,'-',A,'-',A,'-',A)
             ENDIF
#endif
          ENDIF
       ENDDO
    ENDIF
#endif
    !
    !------
    ! vdw
    !
    WRITE(IUNIT,100)
100 FORMAT(/12X,'NONBOND PARAMETERS'// &
         8X,'#',4X,'PAIR',10X,'CODE', &
         9X,'A',10X,'B',8X,'RMIN',9X,'EMIN',8X, &
         '1-4: A',8X,'B',10X,'RMIN',11X,'EMIN    COUNT')

    DO I=1,NATVDW
       NBCCNT(I)=0
       NBC(I)=-1
       NBCWC(I)=0
       DO J=NATC,1,-1
          IF(ITC(J) == I) THEN
             NBCWC(I)=NBCWC(I)+1
             IF(ATCCNT(J) > 0) THEN
                NBC(I)=J
                NBCCNT(I)=NBCCNT(I)+1
             ELSE
                IF(NBC(I) < 0) NBC(I)=J
             ENDIF
          ENDIF
       ENDDO
    ENDDO

    I=0
    DO I1=1,NATVDW
       DO J1=1,I1
          I=I+1
          IF(NBC(I1) > 0 .AND. NBC(J1)>0) THEN
             AI=ATC(NBC(I1))
             AJ=ATC(NBC(J1))
             BI=' '
             IF(NBCWC(I1) > 1) BI='*'
             BJ=' '
             IF(NBCWC(J1) > 1) BJ='*'
             K1=NBCCNT(I1)*NBCCNT(J1)
             !
             EMIN=-CNBB(I)
             VDW=SQRT(CNBA(I))
             R6=VDW**6
             B=-2.0*EMIN*R6
             A=0.5*B*R6
             EMIN14=-CNBB(I+MAXCN)
             VDW14=SQRT(CNBA(I+MAXCN))
             R614=VDW14**6
             B14=-2.0*EMIN14*R614
             A14=0.5*B14*R614
             IF(QALL.OR. K1 > 0) THEN
                WRITE(IUNIT,105) I,AI(1:idleng),BI,AJ(1:idleng),BJ, &
                     A,B,VDW,EMIN,A14,B14,VDW14,EMIN14,K1
105             FORMAT(I9,2X,A,A1,'---',A,A1, &
                     2(F13.2,F9.2,F12.3,F15.4),I8)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !
    !------
    ! hbond
    !
    WRITE(IUNIT,110)
110 FORMAT(/9X,'HYDROGEN BOND PARAMETERS', &
         //7X,'#    PAIR         CODE',6X,'A',10X,'B    COUNT'/)
    DO I=1,NCH
#if KEY_FLEXPARM==1 /*flex_bond*/
       IF(QFLXPARM) THEN
          IF(CHAD(I) > 0) THEN
             AI=ATC(CHAD(I))
          ELSE
             AI=ACTEQV(-CHAD(I))
          ENDIF
          IF(CHAA(I) > 0) THEN
             AJ=ATC(CHAA(I))
          ELSE
             AJ=ACTEQV(-CHAA(I))
          ENDIF
       ELSE
#endif /* (flex_bond)*/
          I1=((KCH(I)-1)/NATC)+1
          J1=KCH(I)-NATC*(I1-1)
          AI=ATC(I1)
          AJ=ATC(J1)
#if KEY_FLEXPARM==1
       ENDIF
#endif
       IF(QALL.OR.ICHCNT(I) > 0) THEN
          WRITE(IUNIT,115) I,AI(1:idleng),AJ(1:idleng), &
               KCH(I),CHBA(I),CHBB(I),ICHCNT(I)
115       FORMAT(5X,I4,2X,A,'...',A,I7,2F11.1,I8)
       ENDIF
    ENDDO
    !
!   flush(IUNIT)
    !------
    RETURN
  END SUBROUTINE PARWTR

  !
  !private routines below
  !
  SUBROUTINE PARRDR(IUNIT,NTITL,TITLE,ICARD,JUNIT,NATCT,ATCT,MLEN &
       ,COMLYN,COMLEN,QAPPE)
    !
    !     THIS ROUTINE READS INTERNAL COORDINATE PARAMETERS FROM
    !     A FILE MODULE OR CARDS.  It has been updated to read card
    !     format input free field with a title.
    !
    !     ICARD controls the mode and print out level.
    !      0=binary
    !      1=card - no print out
    !      2=card - print out of most terms but not full nonbond matrix
    !      3=card - with full print out.
    !
    !     ICNTRL(1) = 15 for version 15. This marks the fact the
    !     coefficients for the hydrogen bond potential were changed.
    !               = 17 indicates that separate 1-4 nonbond interactions
    !               are supported.  Also has CNBA and CNBB converted to
    !               sigma**2 and EMIN
    !
    !     SYNTAX:  After the title, card file data is divided into sections
    !      beginning with a keyword line and followed by data lines read
    !      free field:
#if KEY_FLEXPARM==1 /*flexparm_comments*/
    !     ATOM
    !         MASS   code   type   mass
    !     EQUIvalence
    !         group  atom [ repeat(atom) ]
#endif /* (flexparm_comments)*/
    !     BOND
    !         atom atom force_constant distance
    !     ANGLe or THETA
    !         atom atom atom force_constant theta_min
    !     DIHE or PHI
    !         atom atom atom atom force_constant periodicity phi_max
#if KEY_CMAP==1
    !     CMAP
    !         atom atom atom atom atom atom atom atom resolution
    !         ... map data ...
#endif
    !     IMPRoper or IMPHI
    !         atom atom atom atom force_constant i_phi_min
    !     NBONd or NONB
    !         atom polarizability  e  vdW_radius -
    !           [1-4 polarizability  e  vdW_radius]
    !     NBFIX
    !         atom atom emin rmin [ emin14 rmin14 ]
    !     HBOND [AEXP ia] [REXP ir] [HAEX ih] [AAEX iaa]
    !         atom atom well_depth distance
    !
    !     KAPPA atom atom atom atom atom force_const
    !
    !     14TG atom atom atom atom Trans_const Gauche_const
    !
    !     LCH2 atom atom atom atom atom force_const
    !
    use chm_kinds
    use number
    use dimens_fcm
    use param
    use consta
    use defltsm
    use stream
    use string
    use nbthole
    use memory
#if KEY_CMAP==1
    use cmapm
#endif
    use parallel
    use ensemble
#if KEY_STRINGM==1 /*  VO stringm */
    use machio, only: ifreeu
    use multicom_aux
#endif
    use exfunc,only:srchws
    implicit none
    INTEGER IUNIT,NTITL
    CHARACTER(len=*) TITLE(*)
    INTEGER ICARD,JUNIT,NATCT,MLEN
    CHARACTER(len=*) ATCT(*)
    real(chm_real)  TEMP(MLEN)
    INTEGER IOFF(MAXATC),IDX(MLEN)
    CHARACTER(len=*) COMLYN
    INTEGER COMLEN,NNEGANG
    LOGICAL QAPPE
    !
    ! local
    INTEGER ICNTRL(20)
    INTEGER PRLEV
    LOGICAL DATLIN,EOF,QPRINT
    !
    real(chm_real) :: CONST=362.3461_chm_real,RAD
    INTEGER I,J,K,I1,J1,K1,L1,M1,II,IID,IPT,ITEMP,NFND
    INTEGER ICB,ICT,ICP,ICI,ICH,ICTP
    INTEGER IJ(4)
    INTEGER N,NOIJ,NOKL
    real(chm_real)  E,R,ALPHA,A,B,HBRAT,E14,R14,ALPH14,C
    real(chm_real)  C6VAL,FDENOM,FNUM,R6,VDW
    real(chm_real)  RMASS
    CHARACTER(len=4) WRD,CURRNT
    CHARACTER(len=8) :: WORD,AI,AJ,AK,AL,AM,AX='X   ',BI,BJ,BK,BL
#if KEY_CMAP==1
    CHARACTER(len=8) AZ(8)
    INTEGER I2,J2,K2,L2
    INTEGER GSIZE
    INTEGER IN
    INTEGER XM
#endif
#if KEY_STRINGM==1 /*  VO stringm v */
    integer :: oldiol
    logical :: qstr
    common /replicaio/ qstr ! need global variable
#endif
    !
    CHARACTER(len=4) :: PARHDR='PARA',HDR
    CHARACTER(len=80) LINE

    integer(chm_int8) :: no,natc2
#if KEY_CMAP==1
    integer(chm_int8) :: no1,no2
#endif
    !
    !     Keywords for command line reading.
    !
    RAD=DEGRAD
    QPRINT=.FALSE.
    !
    !     Set up a temporary code numbers array.
    !
    K=0
    DO I=1,MAXATC
       IOFF(I)=K
       K=K+I
    ENDDO
    !
#if KEY_STRINGM==1 /*  VO stringm v */
    qstr=iunit.le.size(ifreeu) ! protect against OOB on ifreeu
    if (qstr) qstr=((MOD(IFREEU(IUNIT),8).eq.0).and.(IFREEU(IUNIT).ne.0).and.(ME_LOCAL.eq.0))
    if (qstr) then
     oldiol=iolev
     iolev=1
    endif
#endif
    !
    IF(IOLEV > 0) THEN
       !
       !-----------------------------------------------------------------------
       ! Read binary file format
       IF (ICARD == 0) THEN
          !         Begin Procedure READ-BINARY-FILE
          !         READ PARAMETERS FROM PARAMETER MODULE
          !
#if KEY_CMAP==1
          IF(WRNLEV >= 6) WRITE(OUTU,215)
215       FORMAT(' ***** Warning from PARRDR *****',/, &
               ' Cross-terms are not available with binary parameter file')
#endif
          !
          CALL TRYORO(IUNIT,'UNFORMATTED')
          IF (reallow) THEN
             REWIND(UNIT=IUNIT)
          ENDIF
          READ(IUNIT) HDR,ICNTRL
          IF(HDR /= PARHDR) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,201) PARHDR,HDR
201          FORMAT(' EXPECTED = "',A4,'" FOUND = "',A4,'"')
             CALL WRNDIE(-1,'<PARMIO>','HEADERS DONT MATCH')
          ENDIF
          IF(ICNTRL(1) == 1) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,213)
213          FORMAT(' ***** Warning from PARRDR *****',/, &
                  ' You are using a parameter file which has slightly',/, &
                  ' erroneous values (30 ppm) for hydrogen bond coefficients.', &
                  /,' To correct the error, regenerate the binary file',/, &
                  ' using version 15 or later of CHARMM.')
          ENDIF
          IF(ICNTRL(1) <= 16) THEN
             IF(WRNLEV >= 2) WRITE(OUTU,127)
127          FORMAT(' ***** WARNING ***** Old parameter file.',/, &
                  ' 1-4 nonbond interactions set to main values.')
          ENDIF
          CALL RDTITL(TITLE,NTITL,IUNIT,-1)
          CALL WRTITL(TITLE,NTITL,OUTU,1)
          READ(IUNIT) NATC,NCB,NCT,NCP,NCI,NCN,NCH
          IF (ICNTRL(1) <= 16) THEN
             READ(IUNIT) (ITC(I),ALP(I),EFF(I),VDWR(I),I=1,NATC)
             DO I=1,NATC
                CALL ENCODI(I,ATC(I),8,J)
                ALP(I+MAXATC)=ALP(I)
                EFF(I+MAXATC)=EFF(I)
                VDWR(I+MAXATC)=VDWR(I)
             ENDDO
          ELSE
             READ(IUNIT) (ATC(I),I=1,NATC)
             READ(IUNIT) (ITC(I),ALP(I),EFF(I),VDWR(I),I=1,NATC), &
                  (ALP(I),EFF(I),VDWR(I),I=1+MAXATC,NATC+MAXATC)
          ENDIF
          NATVDW=0
          DO J=1,NATC
             IF(ITC(J) > NATVDW) NATVDW=ITC(J)
          ENDDO
          READ(IUNIT) (KCB(I),I=1,NCB),(CBC(I),I=1,NCB), &
               (CBB(I),I=1,NCB)
          READ(IUNIT) (KCT(I),I=1,NCT),(CTC(I),I=1,NCT), &
               (CTB(I),I=1,NCT)
          NNEGANG = 0
          DO I=1,MAXCT
             IF(CTB(I) < 0) THEN
                NNEGANG = NNEGANG + 1
                CTB(I)=COS(CTB(I))-ONE
             ENDIF
          ENDDO
          IF(NNEGANG > 0) THEN
             IF(PRNLEV > 3) THEN
                WRITE(OUTU,'(A,I4,A)') 'PARRDR> ', NNEGANG, ' NEGATIVE ANGLE MINIMA FOUND!'
                WRITE(OUTU,'(A)') 'GROMACS style angle energy function used for angles with negative minima.'
             ENDIF
          ELSE IF(PRNLEV > 6) THEN
             WRITE(OUTU,'(A)') 'PARRDR> ALL ANGLES HAVE POSITIVE MINIMA'
          ENDIF

          IF (ICNTRL(1) >= 20) THEN
             READ(IUNIT) (CTUB(I),I=1,NCT),(CTUC(I),I=1,NCT)
          ELSE
             CTUB(1:NCT)=ZERO
             CTUC(1:NCT)=ZERO
          ENDIF
          IF (ICNTRL(1) <= 17) THEN
             READ(IUNIT) (KCP(I),I=1,NCP),(CPC(I),I=1,NCP), &
                  (TEMP(I),I=1,NCP),(CPB(I),I=1,NCP)
             READ(IUNIT) (KCI(I),I=1,NCI),(CIC(I),I=1,NCI), &
                  (CIB(I),I=1,NCI)
             DO I=1,NCP
                CPD(I)=INT(TEMP(I)+0.001)
                IF(CPD(I) == INT(TEMP(I)-0.001)) CALL DIEWRN(-3)
             ENDDO
             CID(1:NCI) = 0
          ELSE
             READ(IUNIT) (KCP(I),I=1,NCP),(CPC(I),I=1,NCP), &
                  (CPD(I),I=1,NCP),(CPB(I),I=1,NCP)
             READ(IUNIT) (KCI(I),I=1,NCI),(CIC(I),I=1,NCI), &
                  (CID(I),I=1,NCI),(CIB(I),I=1,NCI)
          ENDIF
          !
          ! read vdw parameters
          IF (ICNTRL(1) <= 16) THEN
             READ(IUNIT) (j,I=1,NCN),(CNBA(I),I=1,NCN), &
                  (CNBB(I),I=1,NCN)
             DO I=1,NCN
                A=CNBA(I)
                B=CNBB(I)
                IF (A == 0.0 .OR. B==0.0) THEN
                   CNBA(I)=1.0
                   CNBB(I)=0.0
                ELSE
                   CNBA(I)=((A+A)/B)**(1.0/3.0)
                   CNBB(I)=B*B/(4.0*A)
                ENDIF
                CNBA(I+MAXCN)=CNBA(I)
                CNBB(I+MAXCN)=CNBB(I)
             ENDDO
          ELSE
             READ(IUNIT) (J,I=1,NCN), &
             (CNBA(I),I=1,NCN),(CNBB(I),I=1,NCN), &
             (CNBA(I+MAXCN),I=1,NCN), &
             (CNBB(I+MAXCN),I=1,NCN)
          ENDIF
          IF(ICNTRL(1) <= 23) THEN
             QNBFIX=.FALSE.
          ELSE
             QNBFIX=.TRUE.
             READ(IUNIT) NBFIXN,(NBFIXI(1,I),NBFIXI(2,I),I=1,NBFIXN), &
                  (NBFIXR(1,I),NBFIXR(2,I),NBFIXR(3,I),NBFIXR(4,I),I=1,NBFIXN)
          ENDIF
!
! read hbond parameters
          IF (ICNTRL(1) <= 16) THEN
             READ(IUNIT) (KCH(I),I=1,NCH),(CHBA(I),I=1,NCH), &
                  (CHBB(I),I=1,NCH)
             HBEXPN(1)=12
             HBEXPN(2)=10
             HBEXPN(3)=4
             HBEXPN(4)=2
             !           NOW TRANSFORM CODES VALUE TO NEW STRUCTURE. NATC*(IHB-1)+JHB
             !
             K1=1
             DO I=1,NCH
                IF (KCH(I) == KCH(K1)) THEN
                   IF(CHBA(I) /= CHBA(K1) .OR. CHBB(I)/=CHBB(K1)) THEN
                      IF(WRNLEV >= 2) WRITE(OUTU,216)
216                   FORMAT(' ***** WARNING ***** ', &
                           'CONFLICING HBOND PARAMETERS IN THE OLD FILE.',/, &
                           ' THE SECOND WILL BE IGNORED. THIS MAY LEAD', &
                           ' TO INCORRECT RESULTS.')
                      CALL DIEWRN(0)
                      CHBA(I)=CHBA(K1)
                      CHBB(I)=CHBB(K1)
                   ENDIF
                ELSE
                   K1=I
                ENDIF
                I1=(SQRT(TWO*KCH(I))+HALF)
                J1=KCH(I)-IOFF(I1)
                KCH(I)=NATC*(I1-1)+J1
                KCH(I+NCH)=NATC*(J1-1)+I1
                CHBA(I+NCH)=CHBA(I)
                CHBB(I+NCH)=CHBB(I)
             ENDDO
             NCH=NCH+NCH
             !
          ELSE
             READ(IUNIT) (HBEXPN(I),I=1,4),(KCH(I),I=1,NCH), &
                  (CHBA(I),I=1,NCH),(CHBB(I),I=1,NCH)
          ENDIF
          !
          !         read nonbond and hbond defaults
          !
          IF (ICNTRL(1) <= 18) THEN
             DO I=1,MAXDEF
                DFUSED(I)=.FALSE.
             ENDDO
             !           set up the old defaults for an old file
             !           nonbond defaults if not specified in the parameter files
             DFNBXM = 5
             DFCTNB = 10.0
             DFCFNB = 9.0
             DFCONB = 6.0
             DFGONB = 0.0
             DFGFNB = 10.0
             DFWMIN = 1.4
             DFEPS  = 1.0
             DFE14F = 1.0
             DFLGRP =.FALSE.
             DFLCNS =.TRUE.
             DFLSHF =.TRUE.
             DFLGES =.FALSE.
             DFLGVS =.FALSE.
             DFLFSW =.TRUE.
             DFLVSH =.TRUE.
             DFLVFS =.FALSE.
             DFLBYC =.FALSE.
#if KEY_IMCUBES==1
             DFLBYCI =.FALSE.
#endif
#if KEY_LRVDW==1
             DFLLRV  =.FALSE.
#endif
             DFGEOM  =.FALSE.
             !rjp..02-FEB-99 BYCC
             DFLBCC =.FALSE.
             !           hbond defaults if not specified in the parameter sets
             DFCTHB = 0.5
             DFCFHB = 5.0
             DFCOHB = 4.0
             DFCTHA = 110.0
             DFCFHA = 90.0
             DFCOHA = 90.0
             DFLHBF =.TRUE.
             DFHBEX =.FALSE.
             DFBEST =.FALSE.
          ELSE
             READ(IUNIT) (IDX(I),I=1,MAXDEF),DFUSED
             !rsz  bugfix on DEC ALPHA (problem w/ equivalences data and byte order?)
             DO I=1,MAXDEF
                !rcz           IF(DFUSED(I)) DEFLTS(I)=IDX(I)
                DEFLTS(I)=IDX(I)
             ENDDO
          ENDIF
          !
          IF(ICNTRL(1) <= 16) THEN
             NATC2=(NATC*(NATC+1))/2
             N=NCP
             DO I=1,N
                IF(KCP(I) > NATC2) THEN
                   NCP=NCP+1
                   KCP(NCP)=KCP(I)+NATC2*NATC2
                   CPC(NCP)=CPC(I)
                   CPD(NCP)=CPD(I)
                   CPB(NCP)=CPB(I)
                ENDIF
             ENDDO
             N=NCI
             DO I=1,N
                IF(KCI(I) > NATC2) THEN
                   NCI=NCI+1
                   KCI(NCI)=KCI(I)+NATC2*NATC2
                   CIC(NCI)=CIC(I)
                   CIB(NCI)=CIB(I)
                   CID(NCI)=CID(I)
                ENDIF
             ENDDO
          ENDIF
          !
#if KEY_CMAP==1
          NCTP=0
#endif
#if KEY_FLEXPARM==1
          NACTEQV=0
#endif
          !
          !         End Procedure READ-BINARY-FILE
          !-----------------------------------------------------------------------
          ! Read card file format
       ELSE
          !         Begin Procedure READ-CARD-FILE
          !         read parameters from cards
          !
          CURRNT='JUNK'
          PRLEV=ICARD-1
          IF(PRNLEV < 2) PRLEV=0
          CALL TRYORO(IUNIT,'FORMATTED')
          CALL RDTITL(TITLE,NTITL,IUNIT,0)
          !
          NATC2=(NATC*(NATC+1))/2
          IF(QAPPE) THEN
             !           calculate the number of van der waal types
             NATVDW=0
             DO J=1,NATC
                IF(ITC(J) > NATVDW) NATVDW=ITC(J)
             ENDDO
             IF(NATVDW > 0) THEN
                IF(ALP(NATVDW) == 0.0 .AND. EFF(NATVDW)==0.0 .AND. &
                     VDWR(NATVDW) == 0.0) THEN
                   !           reset ITC values for atom types with no nonbond interactions
                   DO J=1,NATC
                      IF(ITC(J) == NATVDW) ITC(J)=0
                   ENDDO
                   NATVDW=NATVDW-1
                ENDIF
             ENDIF
          ELSE
             ITC(1:MAXATC) = 0
             DO I=1,MAXDEF
                DFUSED(I)=.FALSE.
             ENDDO
             NCB=0
             NCT=0
             NCP=0
             NCI=0
             NCN=0
             NATVDW=0
             NCH=0
             NBFIXN=0
             QNBFIX=.TRUE.
#if KEY_CMAP==1
             NCTP=0
#endif
#if KEY_FLEXPARM==1
             NACTEQV=0
#endif
          ENDIF
          !
#if KEY_FLEXPARM==1 /*init_acteqv*/
          IF(QFLXPARM) THEN
             IF(NACTEQV == 0) THEN
                NACTEQV=1
                ACTEQV(1)=AX
                IACTEQV(1)=100
                DO J=1,MAXATC
                   ACTEQVM(J,1)=1
                ENDDO
             ENDIF
          ENDIF
#endif /* (init_acteqv)*/
          !
          !-----------------------------------------------------------------------
8900      CONTINUE
          !         Begin Procedure GET-AN-INPUT-LINE
          EOF=.FALSE.
          !
9000      CONTINUE
          CALL XTRANE(COMLYN,COMLEN,'PARRDR')
9100      CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.FALSE., &
               QPRINT,'PARRDR> ')
          CALL TRIME(COMLYN,COMLEN)
          IF (.NOT.(EOF.OR.COMLEN > 0)) GOTO 9100
          DATLIN=.FALSE.
          WORD=NEXTA8(COMLYN,COMLEN)
          WRD=WORD(1:4)
          !
          !         Note the order of picking keys here is a bit perverse because
          !         the old keys were overlapping.
          !
          !------
          ! Angles
          IF (WRD == 'THET' .OR. WRD=='ANGL' .OR. WRD=='UREY') THEN
             CURRNT='THET'
             IF(PRLEV >= 1) WRITE(JUNIT,315)
315          FORMAT(/9X,'BOND ANGLE PARAMETERS',/, &
                  8X,'#',6X,'A N G L E',11X,'CODE',8X,'KT',8X,'TO', &
                  8X,'UK',8X,'U0'/)
             !------
             ! Impropers
          ELSEIF (WRD == 'IMPR' .OR. WRD=='IMPH') THEN
             CURRNT='IMPH'
             IF(PRLEV >= 1) WRITE(JUNIT,335)
335          FORMAT(/9X,'IMPROPER TORSION PARAMETERS',/, &
                  9X,'#',8X,'T O R S I O N',11X,'CODE',8X,'KI',8X,'IO'/)
             !------
             ! Fluctuating charges
#if KEY_FLUCQ==1
          ELSEIF (WRD == 'FLUC') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,328)
328          FORMAT(/9X,'FLUCTUATING CHARGE PARAMETERS',/, &
                  10X,'#  ATOM    CHI       ZETA      P.Q.N.   MASS')
#endif
             !------
             ! Dihedrals
          ELSEIF (WRD == 'PHI ' .OR. WRD=='DIHE') THEN
             CURRNT='DIHE'
             IF(PRLEV >= 1) WRITE(JUNIT,325)
325          FORMAT(/9X,'DIHEDRAL ANGLE PARAMETERS',/, &
                  8X,'#',9X,'T O R S I O N',11X,'CODE',8X,'KP', &
                  9X,'N',5X,'DELTA'/)
             !------
             ! CMAP
#if KEY_CMAP==1
          ELSEIF (WRD == 'CMAP') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,329)
329          FORMAT(/9X,'DIHEDRAL CROSS-TERM MAP',/, &
                  9X,'#',9X,'TORSION 1',18X, &
                  'TORSION 2',17X,'CODE',3X,'SIZE'/)
#endif
             !------
             ! OPLS Combination rules
          ELSEIF (WRD == 'COMB') THEN
             CALL WRNDIE(-4,'<PARRDR>', &
                  'Obsolete command, Use: NBON ... GEOMetric ...')
             !           this is now a feature supported by READ PARAM APPEND.
             !           its parsing is now an option of the NONB command - BRB
             !------
             ! Van der Waal
          ELSEIF (WRD == 'NONB' .OR. WRD=='NBON') THEN
             CURRNT='NONB'
             !
             I=GTRMI(COMLYN,COMLEN,'NBXM',-999)
             IF(I /= -999) THEN
                DFNBXM=I
                DUNBXM=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CUTN',FMARK)
             IF(A >= 0.0) THEN
                DFCTNB=A
                DUCTNB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTONNB',FMARK)
             IF(A >= 0.0) THEN
                DFCONB=A
                DUCONB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTOFNB',FMARK)
             IF(A >= 0.0) THEN
                DFCFNB=A
                DUCFNB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CGONNB',FMARK)
             IF(A >= 0.0) THEN
                DFGONB=A
                DUGONB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CGOFNB',FMARK)
             IF(A >= 0.0) THEN
                DFGFNB=A
                DUGFNB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'WMIN',FMARK)
             IF(A >= 0.0) THEN
                DFWMIN=A
                DUWMIN=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'E14F',FMARK)
             IF(A >= 0.0) THEN
                DFE14F=A
                DUE14F=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'EPS',FMARK)
             IF(A >= 0.0) THEN
                DFEPS=A
                DUEPS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'GROU') > 0) THEN
                DFLGRP=.TRUE.
                DULGRP=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'ATOM') > 0) THEN
                DFLGRP=.FALSE.
                DULGRP=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'CDIE') > 0) THEN
                DFLCNS=.TRUE.
                DULCNS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'RDIE') > 0) THEN
                DFLCNS=.FALSE.
                DULCNS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'SHIF') > 0) THEN
                DFLSHF=.TRUE.
                DULSHF=.TRUE.
                DFLFSW=.FALSE.
                DULFSW=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'GSHI') > 0) THEN
                DFLGES=.TRUE.
                DULGES=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'VGSH') > 0) THEN
                DFLGVS=.TRUE.
                DULGVS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'SWIT') > 0) THEN
                DFLSHF=.FALSE.
                DULSHF=.TRUE.
                DFLFSW=.FALSE.
                DULFSW=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'FSWI') > 0) THEN
                DFLSHF=.FALSE.
                DULSHF=.TRUE.
                DFLFSW=.TRUE.
                DULFSW=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'FSHI') > 0) THEN
                DFLSHF=.TRUE.
                DULSHF=.TRUE.
                DFLFSW=.TRUE.
                DULFSW=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'VSHI') > 0) THEN
                DFLVSH=.TRUE.
                DULVSH=.TRUE.
                DFLVFS=.FALSE.
                DULVFS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'VSWI') > 0) THEN
                DFLVSH=.FALSE.
                DULVSH=.TRUE.
                DFLVFS=.FALSE.
                DULVFS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'VFSW') > 0) THEN
                DFLVSH=.FALSE.
                DULVSH=.TRUE.
                DFLVFS=.TRUE.
                DULVFS=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'BYGR') > 0) THEN
                DFLBYC=.FALSE.
                DULBYC=.TRUE.
#if KEY_IMCUBES==1
                DFLBYCI=.FALSE.
                DULBYCI=.TRUE.
#endif
#if KEY_LRVDW==1
                DFLLRV=.FALSE.
                DULLRV=.TRUE.
#endif
                DFLBCC =.FALSE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'BYCU') > 0) THEN
                DFLBYC=.TRUE.
                DULBYC=.TRUE.
                DFLBCC =.FALSE.
             ENDIF
#if KEY_IMCUBES==1
             IF(INDXA(COMLYN,COMLEN,'GEOM') > 0) THEN
                DFGEOM=.TRUE.
                DUGEOM=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'ARIT') > 0) THEN
                DFGEOM=.FALSE.
                DUGEOM=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'BYCB') > 0) THEN
                DFLBYCI=.TRUE.
                DULBYCI=.TRUE.
             ENDIF
#endif
#if KEY_LRVDW==1
             IF(INDXA(COMLYN,COMLEN,'LRC') > 0) THEN
                DFLLRV=.TRUE.
                DULLRV=.TRUE.
             ENDIF
#endif
             !rjp..02-FEB-99 BYCC
             IF(INDXA(COMLYN,COMLEN,'BYCC') > 0) THEN
                DFLBYC=.FALSE.
                DFLBCC=.TRUE.
                DULBCC=.TRUE.
             ENDIF
             !
             IF(INDXA(COMLYN,COMLEN,'VDIS') > 0) CONTINUE
             IF(INDXA(COMLYN,COMLEN,'VSIG') > 0) THEN
                CALL WRNDIE(-1,'<PARRDR>','Vsigma no longer supported')
                IF(WRNLEV >= 2) WRITE(OUTU,*) &
                     'Proceeding with Vdistance option'
             ENDIF
             !
             IF(INDXA(COMLYN,COMLEN,'VATO') > 0) THEN
                IF(DFLGRP.AND.DULGRP) THEN
                   CALL WRNDIE(-1,'<PARRDR>', &
                        'VATOM no longer supported with GROUp')
                   IF(WRNLEV >= 2) &
                        WRITE(OUTU,*) 'Proceeding with VGROUP option'
                ENDIF
             ENDIF
             !
             IF(INDXA(COMLYN,COMLEN,'VGRO') > 0) THEN
                IF(.NOT.DFLGRP.AND.DULGRP) THEN
                   CALL WRNDIE(-1,'<PARRDR>', &
                        'VGROup no longer supported with ATOM')
                   IF(WRNLEV >= 2) &
                        WRITE(OUTU,*) 'Proceeding with VATOM option'
                ENDIF
             ENDIF
             !
             IF(PRLEV >= 1) WRITE(JUNIT,345)
345          FORMAT(/9X,'NONBONDED PARAMETERS',/, &
                  10X,'#  ATOM    CODE   POLARIZABILITY   N (EFF)  ', &
                  'RADIUS    (1-4:)  ...   C6(long-range)'/)
             !------
             ! VDW fixes
          ELSEIF (WRD == 'NBFI') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,375)
375          FORMAT(/9X,'NONBOND FIXES',/, &
                  7X,'#    PAIR',12X,'EMIN',8X,'RMIN', &
                  6X,'EMIN14',6X,'RMIN14'/)
             !------
             ! Pair-specific thole
          ELSEIF (WRD == 'THOL') THEN
            CURRNT=WRD
            THOLCUT=GTRMF(COMLYN,COMLEN,'TCUT',FIVE)
            MAXNBTHOLE=-GTRMI(COMLYN,COMLEN,'MAXN',0)
            if(prnlev>=6) write(OUTU,'(6x,a,f8.3)') 'THOLE CUTOFF ',THOLCUT
            if(prnlev>=6) write(OUTU,'(6x,a,i6)')   'MAXNBTHOLE ',MAXNBTHOLE
            THOLCUT=THOLCUT**2
             !------
             ! Hydrogen bonds
          ELSEIF (WRD == 'HBON') THEN
             CURRNT=WRD
             HBEXPN(1)=GTRMI(COMLYN,COMLEN,'REXP',12)
             HBEXPN(2)=GTRMI(COMLYN,COMLEN,'AEXP',10)
             HBEXPN(3)=GTRMI(COMLYN,COMLEN,'HAEX',4)
             HBEXPN(4)=GTRMI(COMLYN,COMLEN,'AAEX',2)
             IF(HBEXPN(1) <= HBEXPN(2)) CALL DIEWRN(-1)
             DO II=1,4
                IF(HBEXPN(II) < 0) CALL DIEWRN(-1)
             ENDDO
             !
             A=GTRMF(COMLYN,COMLEN,'CUTHB',FMARK)
             IF(A >= 0.0) THEN
                DFCTHB=A
                DUCTHB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTONHB',FMARK)
             IF(A >= 0.0) THEN
                DFCOHB=A
                DUCOHB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTOFHB',FMARK)
             IF(A >= 0.0) THEN
                DFCFHB=A
                DUCFHB=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CUTHA',FMARK)
             IF(A >= 0.0) THEN
                DFCTHA=A
                DUCTHA=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTONHA',FMARK)
             IF(A >= 0.0) THEN
                DFCOHA=A
                DUCOHA=.TRUE.
             ENDIF
             A=GTRMF(COMLYN,COMLEN,'CTOFHA',FMARK)
             IF(A >= 0.0) THEN
                DFCFHA=A
                DUCFHA=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'ACCE') > 0) THEN
                DFLHBF=.TRUE.
                DULHBF=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'NOAC') > 0) THEN
                DFLHBF=.FALSE.
                DULHBF=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'HBEX') > 0) THEN
                DFHBEX=.TRUE.
                DUHBEX=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'HBNO') > 0) THEN
                DFHBEX=.FALSE.
                DUHBEX=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'BEST') > 0) THEN
                DFBEST=.TRUE.
                DUBEST=.TRUE.
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'ALL') > 0) THEN
                DFBEST=.FALSE.
                DUBEST=.TRUE.
             ENDIF
             !
             IF(PRLEV >= 1) WRITE(JUNIT,365)
365          FORMAT(/9X,'HYDROGEN BOND PARAMETERS',/, &
                  7X,'#    PAIR',9X,'CODE   EMIN  RMIN',6X,'A',10X,'B'/)
             !------
             ! Bonds
          ELSEIF (WRD == 'BOND') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,305)
305          FORMAT(10X,'BOND PARAMETERS',/, &
                  8X,'#',3X,'B O N D',9X,'CODE',8X,'KB',8X,'BO'/)
             !
             !------
             ! primitive atoms
          ELSEIF (WRD == 'ATOM') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,317)
317          FORMAT(10X,'ATOM DEFINITIONS',/, &
                  8X,'#',3X,'A T O M '/)
             !
             !------
             ! equivalence groups
          ELSEIF (WRD == 'EQUI') THEN
             CURRNT=WRD
             IF(PRLEV >= 1) WRITE(JUNIT,318)
318          FORMAT(10X,'EQUIVALENCE GROUPS',/, &
                  8X,'#',3X,'E Q U I V A L E N C E S'/)
             !
#if KEY_FLEXPARM==1
             IF(.NOT.QFLXPARM) THEN
#endif
                CALL WRNDIE(-3,'<PARRDR>', &
                     'EQUIvalences found for non-flexible option')
#if KEY_FLEXPARM==1
             ENDIF
#endif
             !
             !------
             ! Print command
          ELSEIF (WRD == 'PRIN') THEN
             IF(INDXA(COMLYN,COMLEN,'ON') > 0 .AND. PRNLEV>1) THEN
                QPRINT=.TRUE.
                PRLEV=2
             ENDIF
             IF(INDXA(COMLYN,COMLEN,'OFF') > 0) THEN
                QPRINT=.FALSE.
                PRLEV=0
             ENDIF
             !------
             ! End command
          ELSEIF (WRD == 'END ') THEN
             EOF=.TRUE.
          ELSE
             DATLIN=.TRUE.
          ENDIF
          IF (.NOT.(DATLIN.OR.EOF)) GOTO 9000
          !-----------------------------------------------------------------------
          IF(CURRNT == 'JUNK') THEN
             CALL WRNDIE(-1,'<PARRDR>','HEADER CARD NOT FOUND')
             RETURN
          ENDIF
          !
          IF(.NOT.EOF) THEN
             !------
             ! Atoms
             IF (CURRNT == 'ATOM') THEN
                !
                !             This section reads atoms
                !
                IF(WORD /= 'MASS') THEN
                   CALL WRNDIE(0,'<PARRDR>', &
                        'Bad atom specification syntax. Expected MASS.')
                   GOTO 9000
                ENDIF
                !
                II=NEXTI(COMLYN,COMLEN)
                AI=NEXTA8(COMLYN,COMLEN)
                IF(II <= 0 .OR. II > MAXATC) THEN
                   if(.not.ii==-1) then
                   CALL WRNDIE(0,'<PARRDR>', &
                        'ATOM has an invalid code. Ignored.')
                   GOTO 9000
                   else
                      ii=srchws(atct,natct,ai)
                      if(ii==0) ii = natc + 1
                   endif
                ENDIF
                !
                IF(AI /= ATCT(II)) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,35) &
                        II,AI(1:idleng),ATCT(II)(1:idleng)
35                 FORMAT(' PARRDR> WARNING: Atom:',I5,' "', &
                        A,'" does not match RTF name "',A,'"')
                ENDIF
#if KEY_FLEXPARM==1 /*flex_atoms*/
                IF(QFLXPARM) THEN
                   IF(ATC(II) == AI) THEN
                      CALL WRNDIE(-1,'<PARRDR>', &
                           'ATOM already exists with the same name')
                   ENDIF
                   IF(ATC(II) /= AI .AND. ATC(II)/=' ') THEN
                      CALL WRNDIE(-1,'<PARRDR>', &
                           'ATOM already exists with a different name')
                   ENDIF
                ENDIF
#endif /* (flex_atoms)*/
                IF(II > NATC) NATC=II
                atc(ii) = ai
                i1 = 0
                do j = 1, maxatc
                   if(atc(j) ==ai) i1 = i1 + 1
                enddo
               IF(i1>1) THEN
                   CALL WRNDIE(-1,'<PARRDR>', &
                        'ATOM has the same name as another atom.')
                ENDIF
                !
                RMASS=NEXTF(COMLYN,COMLEN)
                !
#if KEY_FLEXPARM==1 /*flex_atoms*/
                if( qflxparm .and. (srchws(acteqv,nacteqv,ai) > 0) ) then
                   i1 = -1
                   IF( I1 > 0 ) THEN
                      CALL WRNDIE(-1,'<PARRDR>', &
                           'ATOM has the same name as an equivalence group')
                   ENDIF
                endif
#endif /* (flex_atoms)*/
                IF(PRLEV >= 1) WRITE(JUNIT,36) II,AI(1:idleng),RMASS
36              FORMAT(6X,'MASS',I5,2X,A,F12.5)
                !------
                ! Equivalences
             ELSEIF (CURRNT == 'EQUI') THEN
                !
                !             This section reads equivalences
                !
#if KEY_FLEXPARM==1 /*flex_groups*/
                AI=WORD
                i1=srchws(atc,natc,ai)
                IF(I1 > 0) THEN
                   CALL WRNDIE(-1,'<PARRDR>', &
                        'Equivalence group has the same name as an atom')
                ENDIF
                i1=srchws(acteqv,nacteqv,ai)
                IF(I1 > 0) THEN
                   CALL WRNDIE(-1,'<PARRDR>', &
                        'Equivalence group already exists. It is redefined.')
                ELSE
                   !                add new equivalence group
                   IF(NACTEQV+1 >= MAXACTEQV) CALL WRNDIE(-3,'<PARRDR>', &
                        'Maximum no. of equivalence groups reached')

                   NACTEQV=NACTEQV+1
                   ACTEQV(NACTEQV)=AI
                   I1=NACTEQV
                   DO J=1,MAXATC
                      ACTEQVM(J,I1)=0
                   ENDDO
                ENDIF
                !             redefine the group based on current list
                IACTEQV(I1)=NEXTI(COMLYN,COMLEN)
                IF(IACTEQV(I1) <= 0 .OR. IACTEQV(I1) > 100) THEN
                   CALL WRNDIE(-4,'<PARRDR>', &
                        'Equivalence group specification has bad value')
                ENDIF
                K1=NACTEQV+1
                DO J=1,MAXATC
                   ACTEQVM(J,K1)=0
                ENDDO
                !
                AJ=NEXTA8(COMLYN,COMLEN)
                DO WHILE (AJ /= '   ')
                   J1=0
                   DO J=1,NATC
                      IF(EQWDWC(ATC(J),AJ)) THEN
                         J1=J
                         ACTEQVM(J,K1)=1
                      ENDIF
                   ENDDO
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AJ) THEN
                         J1=-J
                         DO K=1,NATC
                            IF(ACTEQVM(K,J) /= 0) ACTEQVM(K,K1)=1
                         ENDDO
                      ENDIF
                   ENDDO
                   IF(J1 == 0) THEN
                      IF(WRNLEV >= 2) WRITE(OUTU,37)  &
                           AI(1:idleng),AJ(1:idleng)
37                    FORMAT(' PARRDR> WARNING: ATOM IN GROUP ', &
                           2(A,1X),' DOES NOT EXIST')
                   ENDIF
                   AJ=NEXTA8(COMLYN,COMLEN)
                ENDDO
                !
                LINE=' '
                I=1
                AJ=AI
                DO J=1,MAXATC
                   ACTEQVM(J,I1)=ACTEQVM(J,K1)
                   IF(ACTEQVM(J,I1) /= 0 .AND. ATC(J)/=' ') THEN
                      LINE(I:I+idleng-1)=ATC(J)(1:idleng)
                      I=I+idleng+1
                      IF(I > 80) THEN
                         IF(PRLEV >= 1) WRITE(JUNIT,38) AJ(1:idleng),LINE
38                       FORMAT(6X,A,' - ',A)
                         LINE=' '
                         I=1
                         AJ=' '
                      ENDIF
                   ENDIF
                   ACTEQVM(J,K1)=0
                ENDDO
                IF(PRLEV >= 1 .AND. I > 1) WRITE(JUNIT,38) &
                     AJ(1:idleng),LINE(1:I)
                !
#endif /* (flex_groups)*/
                !------
                ! Bonds
             ELSEIF (CURRNT == 'BOND') THEN
                !
                !             This section reads bonds
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                C=NEXTF(COMLYN,COMLEN)
                B=NEXTF(COMLYN,COMLEN)
                !
                IF(NCB  >=  MAXCB) CALL WRNDIE(-4,'<PARRDR>', &
                     'Maximum no. of bonds reached')
                ICB=NCB+1
                !
                i1 = srchws(atc,natc,ai)
                j1 = srchws(atc,natc,aj)
                !
#if KEY_FLEXPARM==1 /*flex_bonds*/
                ! Check for duplicate bond parameters
                IF(QFLXPARM) THEN
                   DO I=1,NCB
                      IF(CBAI(I) > 0) THEN
                         BI=ATC(CBAI(I))
                      ELSE
                         BI=ACTEQV(-CBAI(I))
                      ENDIF
                      IF(CBAJ(I) > 0) THEN
                         BJ=ATC(CBAJ(I))
                      ELSE
                         BJ=ACTEQV(-CBAJ(I))
                      ENDIF
                      IF( (BI == AI .AND. BJ==AJ) .OR. &
                           (BJ == AJ .AND. BI==AI) ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,822) &
                              I,AI(1:idleng),AJ(1:idleng)
822                      FORMAT(' PARRDR> Error: Repeated BOND parameter (', &
                              I6,'): ',A,2X,A,' is replaced')
                         ICB=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) I1=-J
                      IF(ACTEQV(J) == AJ) J1=-J
                   ENDDO
                ENDIF
#endif /* (flex_bonds)*/
                !
                IF((AI(1:3) == 'DRU').AND.(AJ=='X   '))THEN
                   if(i1 > 0)then
                      if (prnlev > 6) write(outu,'(/,2a)')  &
                           ' PARRDR> WARNING: wild card for drude ',AJ
                      do j=1,NATC
                         if((atc(j)(1:4) /= '    ') &
                              .and.(atc(j)(1:1) /= 'H'))then
                            IF (I1 > J) THEN
                               KCB(ICB)=IOFF(I1)+J
                            ELSE
                               KCB(ICB)=IOFF(J)+I1
                            ENDIF
                            if (prnlev > 6) write(outu,'(3i5,2x,a,2x,a,2f10.4,i10)') &
                                 ICB,i1,j,atc(i1),atc(j),C,B,KCB(ICB)
                            CBC(ICB)=C
                            CBB(ICB)=B
                            ICBCNT(ICB)=0
                            ICB=ICB+1
                            NCB=NCB+1
                         endif
                      enddo
                      write(outu,*)
                   endif
                   !GOTO 8900

                ELSE IF((AI == 'X   ').AND.(AJ(1:3)=='DRU'))THEN
                   if(j1 > 0)then
                      if (prnlev > 6) write(outu,'(/,2a)')  &
                           ' PARRDR> WARNING: wild card for drude ',AI
                      do i=1,NATC
                         if((atc(i)(1:4) /= '    ') &
                              .and.(atc(j1)(1:1) /= 'H'))then
                            IF (i > J1) THEN
                               KCB(ICB)=IOFF(i)+J1
                            ELSE
                               KCB(ICB)=IOFF(J1)+i
                            ENDIF
                            if (prnlev > 6) write(outu,'(3i5,2x,a,2x,a,2f10.4,i10)') &
                                 ICB,i,j1,atc(i),atc(j1),C,B,KCB(ICB)
                            CBC(ICB)=C
                            CBB(ICB)=B
                            ICBCNT(ICB)=0
                            ICB=ICB+1
                            NCB=NCB+1
                         endif
                      enddo
                      write(outu,*)
                   endif
                   !GOTO 8900
                ENDIF

                IF (I1 == 0.OR.J1==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,30)  &
                        AI(1:idleng),AJ(1:idleng),C,B
30                 FORMAT(' PARRDR> WARNING: ATOMS IN BOND ', &
                        2(A,1X),2F10.5,' DONT EXIST')
                   GOTO 8900
                ENDIF
                !
#if KEY_FLEXPARM==1 /*flex_bonds2*/
                IF(QFLXPARM) THEN
                   CBAI(ICB)=I1
                   CBAJ(ICB)=J1
                   KCB(ICB)=0
                   IF(I1 < 0) KCB(ICB)=-IACTEQV(-I1)
                   IF(J1 < 0) KCB(ICB)=KCB(ICB)-IACTEQV(-J1)
                ELSE
#endif /* (flex_bonds2)*/
                   IF (I1 > J1) THEN
                      KCB(ICB)=IOFF(I1)+J1
                   ELSE
                      KCB(ICB)=IOFF(J1)+I1
                   ENDIF
#if KEY_FLEXPARM==1
                ENDIF
#endif
                CBC(ICB)=C
                CBB(ICB)=B
                ICBCNT(ICB)=0
                !
                IF(ICB > NCB) NCB=ICB
                IF(PRLEV >= 1) WRITE(JUNIT,310)  &
                     ICB,AI(1:idleng),AJ(1:idleng),KCB(ICB),C,B
310             FORMAT(6X,I4,2X,A,' - ',A,I9,F10.1,F10.3)
                !------
                ! Angles
             ELSEIF (CURRNT == 'THET') THEN
                !
                !             This section reads angles.
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                AK=NEXTA8(COMLYN,COMLEN)
                C=NEXTF(COMLYN,COMLEN)
                B=NEXTF(COMLYN,COMLEN)
                !
                IF(NCT  >=  MAXCT) CALL WRNDIE(-3,'<PARRDR>', &
                     'Maximum no. of angles reached')
                ICT=NCT+1
                !
                i1 = srchws(atc,natc,ai)
                j1 = srchws(atc,natc,aj)
                k1 = srchws(atc,natc,ak)
                !
#if KEY_FLEXPARM==1 /*flex_angles*/
                IF(QFLXPARM) THEN
                   ! Check for duplicate angle parameters
                   DO I=1,NCT
                      IF(CTAI(I) > 0) THEN
                         BI=ATC(CTAI(I))
                      ELSE
                         BI=ACTEQV(-CTAI(I))
                      ENDIF
                      IF(CTAJ(I) > 0) THEN
                         BJ=ATC(CTAJ(I))
                      ELSE
                         BJ=ACTEQV(-CTAJ(I))
                      ENDIF
                      IF(CTAK(I) > 0) THEN
                         BK=ATC(CTAK(I))
                      ELSE
                         BK=ACTEQV(-CTAK(I))
                      ENDIF
                      IF( (BI == AI .AND. BJ==AJ .AND. BK==AK) .OR. &
                           (BI == AK .AND. BJ==AJ .AND. BK==AI) ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,823)  &
                              I,AI(1:idleng),AJ(1:idleng),AK(1:idleng)
823                      FORMAT(' PARRDR> Error: Repeated ANGLE parameter (', &
                              I6,'): ',A,2X,A,2X,A,' is replaced')
                         ICT=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) I1=-J
                      IF(ACTEQV(J) == AJ) J1=-J
                      IF(ACTEQV(J) == AK) K1=-J
                   ENDDO
                ENDIF
#endif /* (flex_angles)*/
                !
                IF (I1 == 0.OR.J1==0.OR.K1==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,55) &
                        AI(1:idleng),AJ(1:idleng),AK(1:idleng),C,B
55                 FORMAT(' PARRDR> WARNING: ATOMS IN ANGLE ', &
                        3(A,1X),2F10.5,' DONT EXIST')
                   GOTO 8900
                ENDIF
                !
#if KEY_FLEXPARM==1 /*flex_angles2*/
                IF(QFLXPARM) THEN
                   CTAI(ICT)=I1
                   CTAJ(ICT)=J1
                   CTAK(ICT)=K1
                   KCT(ICT)=0
                   IF(I1 < 0) KCT(ICT)=-IACTEQV(-I1)
                   IF(J1 < 0) KCT(ICT)=KCT(ICT)-IACTEQV(-J1)
                   IF(K1 < 0) KCT(ICT)=KCT(ICT)-IACTEQV(-K1)
                ELSE
#endif /* (flex_angles2)*/
                   IF (I1 > K1) THEN
                      NO=IOFF(I1)+K1
                   ELSE
                      NO=IOFF(K1)+I1
                   ENDIF
                   KCT(ICT)=NATC*NO+J1
#if KEY_FLEXPARM==1
                ENDIF
#endif
                !
                CTC(ICT)=C
                CTB(ICT)=B*RAD
                ICTCNT(ICT)=0
                !
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   CTUC(ICT)=NEXTF(COMLYN,COMLEN)
                ELSE
                   CTUC(ICT)=0.0
                ENDIF
                CTUB(ICT)=0.0
                IF (COMLEN > 0) CTUB(ICT)=NEXTF(COMLYN,COMLEN)
                !
                IF(ICT > NCT) NCT=ICT
                !
                IF(PRLEV >= 1) THEN
                   WRITE(JUNIT,320) ICT, &
                        AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
                        KCT(ICT),C,B,CTUC(ICT),CTUB(ICT)
320                FORMAT(6X,I4,2X,A,' - ',A,' - ',A,I9,4F10.2)
                ENDIF
                !------
                ! Dihedrals
             ELSEIF (CURRNT == 'DIHE') THEN
                !
                !             This section reads dihedrals.
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                AK=NEXTA8(COMLYN,COMLEN)
                AL=NEXTA8(COMLYN,COMLEN)
                C=NEXTF(COMLYN,COMLEN)
                IID=NEXTI(COMLYN,COMLEN)
                B=NEXTF(COMLYN,COMLEN)
                !
                IF(NCP  >=  MAXCP) CALL WRNDIE(-3,'<PARRDR>', &
                     'Maximum no. of dihedrals reached')
                ICP=NCP+1
                !
                ! Note order k1, i1, j1, l1
                k1 = srchws(atc,natc,ai)
                i1 = srchws(atc,natc,aj)
                j1 = srchws(atc,natc,ak)
                l1 = srchws(atc,natc,al)
                !
                IF(AI == AX) K1=-1
                IF(AL == AX) L1=-1
                !
#if KEY_FLEXPARM==1 /*flex_dihedrals*/
                IF(QFLXPARM) THEN
                   ! Check for duplicate dihedral parameters
                   M1=0
                   IPT=NCP
                   !               first check for repeated periodicity within current set.
                   !               also find boundary of current set.
                   DO I=NCP,1,-1
                      IF(CPAI(I) > 0) THEN
                         BI=ATC(CPAI(I))
                      ELSE
                         BI=ACTEQV(-CPAI(I))
                      ENDIF
                      IF(CPAJ(I) > 0) THEN
                         BJ=ATC(CPAJ(I))
                      ELSE
                         BJ=ACTEQV(-CPAJ(I))
                      ENDIF
                      IF(CPAK(I) > 0) THEN
                         BK=ATC(CPAK(I))
                      ELSE
                         BK=ACTEQV(-CPAK(I))
                      ENDIF
                      IF(CPAL(I) > 0) THEN
                         BL=ATC(CPAL(I))
                      ELSE
                         BL=ACTEQV(-CPAL(I))
                      ENDIF
                      IF(M1 == 0) THEN
                         IF( BI == AI .AND. BJ==AJ .AND. &
                              BK == AK .AND. BL==AL) THEN
                            IPT=I-1
                            IF(WRNLEV >= 2 .AND. IID == CPD(I)) THEN
                               WRITE(OUTU,824) I,AI(1:idleng),AJ(1:idleng), &
                                    AK(1:idleng),AL(1:idleng)
                               ICP=I
                               M1=1
                            ENDIF
                         ELSE
                            M1=1
                         ENDIF
                      ENDIF
                   ENDDO
                   !               now check for repeats
                   DO I=1,IPT
                      IF(CPAI(I) > 0) THEN
                         BI=ATC(CPAI(I))
                      ELSE
                         BI=ACTEQV(-CPAI(I))
                      ENDIF
                      IF(CPAJ(I) > 0) THEN
                         BJ=ATC(CPAJ(I))
                      ELSE
                         BJ=ACTEQV(-CPAJ(I))
                      ENDIF
                      IF(CPAK(I) > 0) THEN
                         BK=ATC(CPAK(I))
                      ELSE
                         BK=ACTEQV(-CPAK(I))
                      ENDIF
                      IF(CPAL(I) > 0) THEN
                         BL=ATC(CPAL(I))
                      ELSE
                         BL=ACTEQV(-CPAL(I))
                      ENDIF
                      IF( (BI == AI .AND. BJ==AJ .AND. &
                           BK == AK .AND. BL==AL) .OR. &
                           (BI == AL .AND. BJ==AK .AND. &
                           BK == AJ .AND. BL==AI) ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,824) I,AI(1:idleng), &
                              AJ(1:idleng),AK(1:idleng),AL(1:idleng)
824                      FORMAT(' PARRDR> Error: Repeated DIHE parameter (', &
                              I6,'): ',A,2X,A,2X,A,2X,A,' is replaced')
                         ICP=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) K1=-J
                      IF(ACTEQV(J) == AJ) I1=-J
                      IF(ACTEQV(J) == AK) J1=-J
                      IF(ACTEQV(J) == AL) L1=-J
                   ENDDO
                ENDIF
#endif /* (flex_dihedrals)*/
                !
                IF (I1 == 0.OR.J1==0.OR.K1==0.OR.L1==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,80) AI(1:idleng), &
                        AJ(1:idleng),AK(1:idleng),AL(1:idleng),C,IID,B
80                 FORMAT(' PARRDR> WARNING: ATOMS IN DIHEDRAL ', &
                        4(A,1X),F10.5,I5,F10.5,' DONT EXIST')
                   GOTO 8900
                ENDIF
                !
#if KEY_FLEXPARM==1 /*flex_dihedrals2*/
                IF(QFLXPARM) THEN
                   CPAI(ICP)=K1
                   CPAJ(ICP)=I1
                   CPAK(ICP)=J1
                   CPAL(ICP)=L1
                   KCP(ICP)=0
                   IF(I1 < 0) KCP(ICP)=-IACTEQV(-I1)
                   IF(J1 < 0) KCP(ICP)=KCP(ICP)-IACTEQV(-J1)
                   IF(K1 < 0) KCP(ICP)=KCP(ICP)-IACTEQV(-K1)
                   IF(L1 < 0) KCP(ICP)=KCP(ICP)-IACTEQV(-L1)
                ELSE
#endif /* (flex_dihedrals2)*/
                   !
                   IF (K1 > 0.AND.L1>0) THEN
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
                      NO=NOIJ+NOKL*NATC2
                      IF((I1-J1)*(K1-L1) < 0) NO=NO+NATC2*NATC2
                   ELSEIF (K1 <= 0.AND.L1<=0) THEN
                      IF (I1 > J1) THEN
                         NO=IOFF(I1)+J1
                      ELSE
                         NO=IOFF(J1)+I1
                      ENDIF
                   ELSE
                      IF(WRNLEV >= 2) WRITE(OUTU,100) &
                           AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
                           AL(1:idleng),C,IID,B
100                   FORMAT(' PARRDR> WARNING: WILDCARDS IN PHI ', &
                           4(A,1X),F10.5,I5,F10.5,' ILLEGAL')
                      GOTO 8900
                   ENDIF
                   KCP(ICP)=NO
#if KEY_FLEXPARM==1
                ENDIF
#endif
                !
                CPC(ICP)=C
                CPD(ICP)=IID
                IF(B == 180.0) THEN
                   CPB(ICP)=PI
                ELSE
                   CPB(ICP)=B*RAD
                ENDIF
                ICPCNT(ICP)=0
                IF(ICP > NCP) NCP=ICP
                !
                IF(PRLEV >= 1) THEN
                   WRITE(JUNIT,330) ICP,AI(1:idleng),AJ(1:idleng), &
                        AK(1:idleng),AL(1:idleng),KCP(ICP),C,IID,B
330                FORMAT(6X,I4,2X,A,' - ',A,' - ',A,' - ',A, &
                        I12,F10.2,I5,F10.2)
                ENDIF

                !
#if KEY_CMAP==1 /*cmap_main*/
                !------
                ! CMAP
             ELSEIF (CURRNT == 'CMAP') THEN
                !
                !             This section reads dihedrals.  Note the permutation between
                !             AI AJ AK AL and K I J L.
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                AK=NEXTA8(COMLYN,COMLEN)
                AL=NEXTA8(COMLYN,COMLEN)
                !
                BI=NEXTA8(COMLYN,COMLEN)
                BJ=NEXTA8(COMLYN,COMLEN)
                BK=NEXTA8(COMLYN,COMLEN)
                BL=NEXTA8(COMLYN,COMLEN)
                !
                GSIZE=NEXTI(COMLYN,COMLEN)
                !
                IF(NCTP  >=  MAXCTP) CALL WRNDIE(-3,'<PARRDR>', &
                     'Maximum no. of cross-term maps reached')
                ICTP=NCTP+1
                !
                ! Note order k, i, j, l
                k1 = srchws(atc,natc,ai)
                i1 = srchws(atc,natc,aj)
                j1 = srchws(atc,natc,ak)
                l1 = srchws(atc,natc,al)
                k2 = srchws(atc,natc,bi)
                i2 = srchws(atc,natc,bj)
                j2 = srchws(atc,natc,bk)
                l2 = srchws(atc,natc,bl)
                !
#if KEY_FLEXPARM==1 /*flex_cmap*/
                ! Check for duplicate cmap parameters
                IF(QFLXPARM) THEN
                   DO I=1,NCTP
                      DO J=1,8
                         IF(CTPA(I,J) > 0) THEN
                            AZ(J)=ATC(CTPA(I,J))
                         ELSE
                            AZ(J)=ACTEQV(-CTPA(I,J))
                         ENDIF
                      ENDDO
                      IF( (AZ(1) == AI .AND. AZ(2)==AJ .AND. &
                           AZ(3) == AK .AND. AZ(4)==AL .AND. &
                           AZ(5) == BI .AND. AZ(6)==BJ .AND. &
                           AZ(7) == BK .AND. AZ(8)==BL) ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,825) I, &
                              AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
                              AL(1:idleng),BI(1:idleng),BJ(1:idleng), &
                              BK(1:idleng),BL(1:idleng)
825                      FORMAT(' PARRDR> Error: Repeated CMAP parameter (', &
                              I6,'): ',A,2X,A,2X,A,2X,A,2X, &
                              A,2X,A,2X,A,2X,A,' is replaced')
                         ICTP=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) I1=-J
                      IF(ACTEQV(J) == AJ) J1=-J
                      IF(ACTEQV(J) == AK) K1=-J
                      IF(ACTEQV(J) == AL) L1=-J
                      IF(ACTEQV(J) == BI) I2=-J
                      IF(ACTEQV(J) == BJ) J2=-J
                      IF(ACTEQV(J) == BK) K2=-J
                      IF(ACTEQV(J) == BL) L2=-J
                   ENDDO
                ENDIF
#endif /* (flex_cmap)*/
                !
                IF (I1 == 0.OR.J1==0.OR.K1==0.OR.L1==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,85) 'PHI1', &
                        AI(1:idleng),AJ(1:idleng), &
                        AK(1:idleng),AL(1:idleng)
85                 FORMAT(' PARRDR> WARNING: ATOMS IN ',A4,' (CMAP) ', &
                        4(A,1X),' DONT EXIST')
                   GOTO 8900
                ELSEIF (I2 == 0.OR.J2==0.OR.K2==0.OR.L2==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,85) 'PHI2', &
                        BI(1:idleng),BJ(1:idleng), &
                        BK(1:idleng),BL(1:idleng)
                   GOTO 8900
                ENDIF

#if KEY_FLEXPARM==1 /*flex_cmap2*/
                IF(QFLXPARM) THEN
                   CTPA(ICTP,1)=I1
                   CTPA(ICTP,2)=J1
                   CTPA(ICTP,3)=K1
                   CTPA(ICTP,4)=L1
                   CTPA(ICTP,5)=I2
                   CTPA(ICTP,6)=J2
                   CTPA(ICTP,7)=K2
                   CTPA(ICTP,8)=L2
                   KCTP(ICTP)=0
                   IF(I1 < 0) KCTP(ICTP)=-IACTEQV(-I1)
                   IF(J1 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-J1)
                   IF(K1 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-K1)
                   IF(L1 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-L1)
                   IF(I2 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-I2)
                   IF(J2 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-J2)
                   IF(K2 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-K2)
                   IF(L2 < 0) KCTP(ICTP)=KCTP(ICTP)-IACTEQV(-L2)
                ELSE
#endif /* (flex_cmap2)*/
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
                   NO1=NOIJ+NOKL*NATC2
                   IF((I1-J1)*(K1-L1) < 0) NO1=NO1+NATC2*NATC2

                   IF (I2 > J2) THEN
                      NOIJ=IOFF(I2)+J2
                   ELSE
                      NOIJ=IOFF(J2)+I2
                   ENDIF
                   IF (K2 > L2) THEN
                      NOKL=IOFF(K2)+L2
                   ELSE
                      NOKL=IOFF(L2)+K2
                   ENDIF
                   NO2=NOIJ+NOKL*NATC2
                   IF((I2-J2)*(K2-L2) < 0) NO2=NO2+NATC2*NATC2
                   !
                   KCTP(ICTP)=NO1+NO2
#if KEY_FLEXPARM==1
                ENDIF
#endif

                GSCTP(ICTP)=GSIZE
                IF(ICTP > NCTP) THEN
                   NCTP=ICTP
                   call cmap_allocate_table(mctp(nctp),gsize)
                ENDIF
                CALL RDCMAP(COMLYN,MXCMSZ,COMLEN,QPRINT, &
                     IUNIT,GSIZE,MCTP(ICTP)%grid(1)%A)
                !
                ICTPCNT(ICTP)=0
                XM=GSIZE/2
                CALL SETCMAP(GSIZE,XM,MCTP(NCTP),THR6TY/GSIZE)
                IF(PRLEV >= 1) THEN
                   WRITE(JUNIT,336) ICTP,AI(1:idleng),AJ(1:idleng), &
                        AK(1:idleng),AL(1:idleng), &
                        BI(1:idleng),BJ(1:idleng), &
                        BK(1:idleng),BL(1:idleng), &
                        KCTP(ICTP),GSCTP(ICTP)
                ENDIF
336             FORMAT(6X,I4,2X,A,' - ',A,' - ',A,' - ',A, &
                     2X,A,' - ',A,' - ',A,' - ',A,I12,I5)

#endif /* (cmap_main)*/
                !------
                ! Impropers
             ELSEIF (CURRNT == 'IMPH') THEN
                !
                !             This section reads improper dihedrals.
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                AK=NEXTA8(COMLYN,COMLEN)
                AL=NEXTA8(COMLYN,COMLEN)
                C=NEXTF(COMLYN,COMLEN)
                IID=NEXTI(COMLYN,COMLEN)
                B=NEXTF(COMLYN,COMLEN)
                !
                IF(NCI  >=  MAXCI) CALL WRNDIE(-3,'<PARRDR>', &
                     'Maximum no. of impropers reached')
                ICI=NCI+1
                !
                ! Note order i, k, l, j
                i1 = srchws(atc,natc,ai)
                k1 = srchws(atc,natc,aj)
                l1 = srchws(atc,natc,ak)
                j1 = srchws(atc,natc,al)
                !
                IF(AI == AX) I1=-1
                IF(AJ == AX) K1=-1
                IF(AK == AX) L1=-1
                IF(AL == AX) J1=-1

#if KEY_FLEXPARM==1 /*flex_imphi*/
                IF(QFLXPARM) THEN
                   ! Check for duplicate improper dihedral parameters
                   DO I=1,NCI
                      IF(CIAI(I) > 0) THEN
                         BI=ATC(CIAI(I))
                      ELSE
                         BI=ACTEQV(-CIAI(I))
                      ENDIF
                      IF(CIAJ(I) > 0) THEN
                         BJ=ATC(CIAJ(I))
                      ELSE
                         BJ=ACTEQV(-CIAJ(I))
                      ENDIF
                      IF(CIAK(I) > 0) THEN
                         BK=ATC(CIAK(I))
                      ELSE
                         BK=ACTEQV(-CIAK(I))
                      ENDIF
                      IF(CIAL(I) > 0) THEN
                         BL=ATC(CIAL(I))
                      ELSE
                         BL=ACTEQV(-CIAL(I))
                      ENDIF
                      IF( (BI == AI .AND. BJ==AJ .AND. &
                           BK == AK .AND. BL==AL) .OR. &
                           (BI == AL .AND. BJ==AK .AND. &
                           BK == AJ .AND. BL==AI) ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,826) I,AI(1:idleng), &
                              AJ(1:idleng),AK(1:idleng),AL(1:idleng)
826                      FORMAT(' PARRDR> Error: Repeated IMPH parameter (', &
                              I6,'): ',A,2X,A,2X,A,2X,A,' is replaced')
                         ICI=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) I1=-J
                      IF(ACTEQV(J) == AJ) K1=-J
                      IF(ACTEQV(J) == AK) L1=-J
                      IF(ACTEQV(J) == AL) J1=-J
                   ENDDO
                ENDIF
#endif /* (flex_imphi)*/

                !
                IF (I1 == 0.OR.J1==0.OR.K1==0.OR.L1==0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,115) &
                        AI(1:idleng),AK(1:idleng),AL(1:idleng), &
                        AJ(1:idleng),C,IID,B
115                FORMAT(' PARRDR> WARNING: ATOMS IN IMPHI ', &
                        4(A,1X),F10.5,I5,F10.5,' DONT EXIST')
                   GOTO 8900
                ENDIF
                !
#if KEY_FLEXPARM==1 /*flex_imphi2*/
                IF(QFLXPARM) THEN
                   CIAI(ICI)=I1
                   CIAJ(ICI)=K1
                   CIAK(ICI)=L1
                   CIAL(ICI)=J1
                   KCI(ICI)=0
                   IF(I1 < 0) KCI(ICI)=-IACTEQV(-I1)
                   IF(J1 < 0) KCI(ICI)=KCI(ICI)-IACTEQV(-J1)
                   IF(K1 < 0) KCI(ICI)=KCI(ICI)-IACTEQV(-K1)
                   IF(L1 < 0) KCI(ICI)=KCI(ICI)-IACTEQV(-L1)
                ELSE
#endif /* (flex_imphi2)*/
                   !
                   IF (I1 > 0.AND.J1>0.AND.K1>0.AND.L1>0) THEN
                      !                   no wildcards
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
                      NO=NOIJ+NOKL*NATC2
                      IF((I1-J1)*(K1-L1) < 0) NO=NO+NATC2*NATC2
                      !
                   ELSEIF (K1 < 0.AND.L1<0.AND.I1 > 0.AND.J1>0) &
                        THEN
                      !                       wildcard type   A - X - X - B
                      IF (I1 > J1) THEN
                         NO=IOFF(I1)+J1
                      ELSE
                         NO=IOFF(J1)+I1
                      ENDIF
                      !
                   ELSEIF (K1 > 0.AND.L1>0.AND.I1 < 0.AND.J1<0) &
                        THEN
                      !                       wildcard type X - A - B - X
                      IF (K1 > L1) THEN
                         NO=-(IOFF(K1)+L1+NATC**3+NATC**2)
                      ELSE
                         NO=-(IOFF(L1)+K1+NATC**3+NATC**2)
                      ENDIF
                      !
                   ELSEIF (K1 > 0.AND.L1>0.AND.I1 < 0.AND.J1>0) &
                        THEN
                      !                       wildcard type X - A - B - C
                      NO=-(J1+NATC*(L1-1)+NATC*NATC*K1)
                      !
                   ELSEIF (K1 < 0.AND.L1 > 0.AND.I1<0.AND.J1>0) &
                        THEN
                      !                       wildcard type X - X - A - B
                      NO=-(J1+NATC*(L1-1))
                      !
                   ELSE
                      IF(WRNLEV >= 2) WRITE(OUTU,135)  &
                           AI(1:idleng),AJ(1:idleng),AK(1:idleng), &
                           AL(1:idleng),C,IID,B
135                   FORMAT(' PARRDR> WARNING: WILDCARDS IN IMPHI ', &
                           4(A,1X),F10.5,I5,F10.5,' ILLEGAL')
                      GOTO 8900
                   ENDIF
                   KCI(ICI)=NO
#if KEY_FLEXPARM==1
                ENDIF
#endif
                !
                CIC(ICI)=C
                CID(ICI)=IID
                IF(B == 180.0) THEN
                   CIB(ICI)=PI
                ELSE
                   CIB(ICI)=B*RAD
                ENDIF
                ICICNT(ICI)=0
                IF(ICI > NCI) NCI=ICI
                !
                IF(PRLEV >= 1) &
                     WRITE(JUNIT,330) ICI,AI(1:idleng),AJ(1:idleng), &
                     AK(1:idleng),AL(1:idleng),KCI(ICI),C,IID,B
                !
                !------
                ! van der Waals
             ELSEIF (CURRNT == 'NONB') THEN
                !
                !             This section reads non-bonds.
                !
                AI=WORD
                ALPHA=NEXTF(COMLYN,COMLEN)
                E=NEXTF(COMLYN,COMLEN)
                R=NEXTF(COMLYN,COMLEN)
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   ALPH14=NEXTF(COMLYN,COMLEN)
                ELSE
                   ALPH14=ALPHA
                ENDIF
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   E14=NEXTF(COMLYN,COMLEN)
                ELSE
                   E14=E
                ENDIF
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   R14=NEXTF(COMLYN,COMLEN)
                ELSE
                   R14=R
                ENDIF
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   ! Read the value for long-range C6 self-coeff
                   C6VAL=NEXTF(COMLYN,COMLEN)
                   ! the value should be negative or zero.
                   IF(C6VAL > ZERO) THEN
                      IF(WRNLEV >= 2) WRITE(OUTU,349)
349                   FORMAT(' ERROR: Specified C6-long-range value', & 
                               ' must not be positive.',/ &
                               ' The opposite value is assumed.')
                      C6VAL=-C6VAL
                   ENDIF
                ELSE
                   ! Calculate default value for long-range C6 self-coefficient
                   ! save as a positive value.
                   VDW=R*TWO
                   R6=VDW**6
                   IF(R6 <= 0.000001) R6=0.000001
                   FNUM=CONST*ALPHA**2
                   IF (E >= 0.5 .AND. FNUM /= 0.0) THEN
                      ! derive value from Slater-Kirkwood self interaction
                      FDENOM=SQRT(ABS(ALPHA/E))*TWO
                      C6VAL=FNUM/FDENOM
                   ELSEIF (E < 0.0) THEN
                      ! derive value from E-min and R-min self-interaction
                      C6VAL=-TWO*E*R6
                   ELSE
                      C6VAL=0.0
                   ENDIF
                ENDIF
                NATVDW=NATVDW+1
                I=0
                DO J=1,NATC
                   IF(EQWDWC(ATC(J),AI)) THEN
                      I=J
                      IF(ITC(I) > 0 .AND. WRNLEV >= 2) &
                           WRITE(JUNIT,351) ATC(I)(1:idleng)
351                   FORMAT(' PARRDR> NOTE: atom type "',A, &
                           '" is removed from previous group')
                      ITC(I)=NATVDW
                      ALP(NATVDW)=ALPHA
                      EFF(NATVDW)=E
                      VDWR(NATVDW)=R
                      ALP(NATVDW+MAXATC)=ALPH14
                      EFF(NATVDW+MAXATC)=E14
                      VDWR(NATVDW+MAXATC)=R14
                      C6DEF(NATVDW)=C6VAL
                      IF(PRLEV >= 1) THEN
                         WRITE(JUNIT,350) NATVDW,ATC(I)(1:idleng), &
                              I,ALP(NATVDW),EFF(NATVDW),VDWR(NATVDW), &
                              ALP(NATVDW+MAXATC),EFF(NATVDW+MAXATC), &
                              VDWR(NATVDW+MAXATC),C6DEF(NATVDW)
350                      FORMAT(6X,I5,3X,A,I7,2(F13.3,F14.3,F9.3),F12.3)
                      ENDIF
                   ENDIF
                ENDDO
                IF(I == 0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,150) AI(1:idleng)
150                FORMAT(' PARRDR> WARNING: ATOM FOR NBOND ', &
                        A,' DOESNT EXIST')
                   NATVDW=NATVDW-1
                ENDIF
#if KEY_FLUCQ==1
                !------
                ! Fluctuating charges
             ELSEIF (CURRNT == 'FLUC') THEN
                !
                !             This section reads fluctuating charge parameters
                !
                AI=WORD
                B=NEXTF(COMLYN,COMLEN)
                C=NEXTF(COMLYN,COMLEN)
                I=NEXTI(COMLYN,COMLEN)
                E=NEXTF(COMLYN,COMLEN)
                i1 = srchws(atc,natc,ai)
                IF(I1 == 0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,153) AI(1:idleng)
153                FORMAT(' PARRDR> WARNING: ATOM TYPE ', &
                        A,' DOESNT EXIST')
                ELSE
                   FQCHI(I1)=B
                   FQZETA(I1)=C
                   FQPRIN(I1)=I
                   FQCHMA(I1)=E
                ENDIF
#endif
                !------
                ! VDW fixes
             ELSEIF (CURRNT == 'NBFI') THEN
                !
                !             This section reads nonbond fixes
                !
                AI=WORD
                IF(AI == '*') AI=AX       ! allow both wildcard formats
                !             AJ=NEXTA4(COMLYN,COMLEN)
                AJ=NEXTA8(COMLYN,COMLEN)
                IF(AJ == '*') AJ=AX       ! allow both wildcard formats
                E=NEXTF(COMLYN,COMLEN)
                R=NEXTF(COMLYN,COMLEN)
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   E14=NEXTF(COMLYN,COMLEN)
                ELSE
                   E14=E
                ENDIF
                CALL TRIME(COMLYN,COMLEN)
                IF (COMLEN > 0) THEN
                   R14=NEXTF(COMLYN,COMLEN)
                ELSE
                   R14=R
                ENDIF
                I1=0
                J1=0
                !
                IF(AI == AX .AND. AJ /= AX) THEN
                   AK=AI
                   AI=AJ
                   AJ=AK
                ENDIF
                IF (AI == AX .AND. AJ.EQ.AX) THEN
                   I1=1000
                   J1=1000
                   !               ADD-NBFIX-TO-TABLE
                   !..B980112.ep
                   !..B980610.mc                NBFIXN=NBFIXN+1
                   CALL ADDNBF(I1,J1,E,R,E14,R14,AI,AJ,AK,AL,AX, &
                        JUNIT,PRLEV,NBFIXN)
                ELSEIF (AJ == AX) THEN
                   J1=1000
                   J=0
9200               IF (J < NATC) THEN
                      J=J+1
                      IF(EQWDWC(ATC(J),AI)) THEN
                         I1=J
                         !                   ADD-NBFIX-TO-TABLE
                         !..B980112.ep
                         !..B980610.mc                    NBFIXN=NBFIXN+1
                         CALL ADDNBF(I1,J1,E,R,E14,R14,AI,AJ,AK,AL,AX, &
                              JUNIT,PRLEV,NBFIXN)
                      ENDIF
                      GOTO 9200
                   ENDIF
                ELSE
                   NFND=0
                   J=0
9300               IF (J < NATC) THEN
                      J=J+1
                      IF(EQWDWC(ATC(J),AI)) THEN
                         I1=J
                         K=0
9400                     IF (K < NATC) THEN
                            K=K+1
                            IF(EQWDWC(ATC(K),AJ)) THEN
                               J1=K
                               NFND=NFND+1
                               !                       ADD-NBFIX-TO-TABLE
                               !..B980112.ep
                               !..B980610.mc                        NBFIXN=NBFIXN+1
                               CALL ADDNBF(I1,J1,E,R,E14,R14,AI,AJ,AK,AL,AX, &
                                    JUNIT,PRLEV,NBFIXN)
                            ENDIF
                            GOTO 9400
                         ENDIF
                      ENDIF
                      GOTO 9300
                   ENDIF
                   IF(NFND == 0) CALL WRNDIE(0,'<PARRDR>', &
                        'NO MATCH FOR NBFIX')
                   IF(NFND > 1) CALL WRNDIE(-1,'<PARRDR>', &
                        'NBFIX involves an atom in a vdw group; Whole VDW group modified')
                ENDIF
                !
                IF(I1 == 0.OR.J1.EQ.0) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,39)  &
                        AI(1:idleng),AJ(1:idleng),E,R
39                 FORMAT(' PARRDR> WARNING: ATOMS IN NBFIX ', &
                        2(A,1X),2F10.5,' DONT EXIST')
                ENDIF
                !------
                ! Pair-specific Thole parameters
            ELSEIF (CURRNT == 'THOL') THEN
              AI=WORD
              AJ=NEXTA8(COMLYN,COMLEN)
              R= NEXTF(COMLYN,COMLEN)
                i1 = srchws(atc,natc,ai)
                j1 = srchws(atc,natc,aj)
              call ADDTHOLE(I1,J1,R)
                if(prnlev>=6) write(outu,*) 'Pair-specific Thole: ',NBTHOL,AI,AJ,R,I1,J1
                !-------
                ! Hydrogen bonds
             ELSEIF (CURRNT == 'HBON') THEN
                !
                !             This section reads h-bonds.
                !
                AI=WORD
                AJ=NEXTA8(COMLYN,COMLEN)
                E=NEXTF(COMLYN,COMLEN)
                R=NEXTF(COMLYN,COMLEN)
                HBRAT=HBEXPN(2)
                HBRAT=HBRAT/HBEXPN(1)
                B=E*R**HBEXPN(2)/(HBRAT-1.0)
                A=B*R**(HBEXPN(1)-HBEXPN(2))*HBRAT
                !
#if KEY_FLEXPARM==1 /*flex_hbonds*/
                IF(QFLXPARM) THEN
                   !
                   IF(NCH  >=  MAXCH) CALL WRNDIE(-3,'<PARRDR>', &
                        'Maximum no. of H-bonds reached')
                   ICH=NCH+1
                   !
                   i1 = srchws(atc,natc,ai)
                   j1 = srchws(atc,natc,aj)
                   !
                   DO I=1,NCH
                      IF(CHAD(I) > 0) THEN
                         BI=ATC(CHAD(I))
                      ELSE
                         BI=ACTEQV(-CHAD(I))
                      ENDIF
                      IF(CHAA(I) > 0) THEN
                         BJ=ATC(CHAA(I))
                      ELSE
                         BJ=ACTEQV(-CHAA(I))
                      ENDIF
                      IF(BI == AI .AND. BJ==AJ) THEN
                         IF(WRNLEV >= 2) WRITE(OUTU,827) I, &
                              AI(1:idleng),AJ(1:idleng)
827                      FORMAT(' PARRDR> Error: Repeated HBOND parameter (', &
                              I6,'): ',A,2X,A,' is replaced')
                         ICH=I
                      ENDIF
                   ENDDO
                   !
                   DO J=1,NACTEQV
                      IF(ACTEQV(J) == AI) I1=-J
                      IF(ACTEQV(J) == AJ) J1=-J
                   ENDDO

                   IF(ICH > NCH) NCH=ICH
                   CHAD(ICH)=I1
                   CHAA(ICH)=J1
                   KCH(NCH)=0
                   IF(I1 < 0) KCH(ICH)=-IACTEQV(-I1)
                   IF(J1 < 0) KCH(ICH)=KCH(ICH)-IACTEQV(-J1)
                   !
                   CHBA(ICH)=A
                   CHBB(ICH)=B
                   IF(PRLEV >= 1) WRITE(JUNIT,370) ICH,AI(1:idleng), &
                        AJ(1:idleng),KCH(ICH),E,R,A,B
                   ICHCNT(ICH)=0
                ELSE
#endif /* FLEXPARM (flex_hbonds)*/
                   !
                   I1=0
                   J1=0
                   DO J=1,NATC
                      IF(EQWDWC(ATC(J),AI)) THEN
                         I1=J
                         DO K=1,NATC
                            IF(EQWDWC(ATC(K),AJ)) THEN
                               J1=K
                               IF(NCH  >=  MAXCH) CALL WRNDIE(-3,'<PARRDR>', &
                                    'Maximum no. of H-bonds reached')
                               NCH=NCH+1
                               NO=NATC*(I1-1)+J1
                               KCH(NCH)=NO
                               CHBA(NCH)=A
                               CHBB(NCH)=B
                               ICHCNT(NCH)=0
                               IF(PRLEV >= 1) THEN
                                  WRITE(JUNIT,370) NCH, &
                                       ATC(I1)(1:idleng),ATC(J1)(1:idleng), &
                                       KCH(NCH),E,R,A,B
370                               FORMAT(5X,I4,2X,A,' - ',A,I7,F6.1,F6.2, &
                                       2F11.1)
                               ENDIF
                            ENDIF
                         ENDDO
                      ENDIF
                   ENDDO
                   IF(I1 == 0.OR.J1==0) THEN
                      IF(WRNLEV >= 2) WRITE(OUTU,195) &
                           AI(1:idleng),AJ(1:idleng),E,R
195                   FORMAT(' PARRDR> WARNING: ATOMS IN HBOND ', &
                           2(A,1X),2F10.5,' DONT EXIST')
                   ENDIF
#if KEY_FLEXPARM==1
                ENDIF
#endif
                !------
             ELSE
                CALL WRNDIE(-5,'<PARRDR>','Bad CURRNT - coding error')
             ENDIF
             !-----------------------------------------------------------------------
          ENDIF
          IF (.NOT.EOF) GOTO 8900
          !
          !         End Procedure READ-CARD-FILE

          NNEGANG = 0
          DO I=1,MAXCT
             IF(CTB(I) < 0) THEN
                NNEGANG = NNEGANG + 1
                CTB(I)=COS(CTB(I))-ONE
             ENDIF
          ENDDO
          IF(NNEGANG > 0) THEN
             IF(PRNLEV > 3) THEN
                WRITE(OUTU,'(A,I4,A)') 'PARRDR> ', NNEGANG, ' NEGATIVE ANGLE MINIMA FOUND!'
                WRITE(OUTU,'(A)') 'GROMACS style angle energy function used for angles with negative minima.'
             ENDIF
          ELSE IF(PRNLEV > 6) THEN
             WRITE(OUTU,'(A)') 'PARRDR> ALL ANGLES HAVE POSITIVE MINIMA'
          ENDIF

       ENDIF    ! (ICARD == 0)
    ENDIF      ! (IOLEV > 0)
    !
#if KEY_STRINGM==1 /*  VO : restore iolev */
    if (qstr) iolev=oldiol
#endif
    !
    RETURN
  END subroutine PARRDR

  SUBROUTINE PARRDR2(ICARD,MLEN)
    !
    ! This routine finishes the process of processing the parameters after
    ! all I/O is complete (reading parameters). - BRB
    !
    ! It performs the following tasks
    !   1. sorts/resorts the parameters
    !   2. checks for duplicated parameters
    !   3. Does parallel communication
    !   4. Sets up extra dihedral/improper arrays
    !
    use chm_kinds
    use number
    use exfunc
    use dimens_fcm
    use param
    use consta
    use defltsm
    use stream
    use memory
    use parallel
    use ensemble
#if KEY_STRINGM==1 /*  VO */
    use multicom_aux
#endif
#if KEY_CMAP==1
    use cmapm
#endif
    implicit none
    !
    INTEGER ICARD,MLEN
    real(chm_real)  TEMP(MLEN)
    INTEGER IOFF(MAXATC),IDX(MLEN),TMPI(MLEN)
    integer(chm_int8) :: tmpi8(MLEN)
    !
#if KEY_STRINGM==1 /*  VO stringm v */
    logical :: qstr
    common /replicaio/ qstr ! need global variable
#endif
    !
    ! local
    INTEGER I,J,K
    !     real(chm_real) :: CONST=362.3461_chm_real,RAD
    !     INTEGER I,J,K,I1,J1,IPT,NO
    !     real(chm_real)  E,R,EMIN,FNUM,VDW,R6,A,B
    !     real(chm_real)  E14,R14,EMIN14,FNUM14,VDW14,R614,A14,B14
    !     real(chm_real)  FDENOM,R6I,R6J
    !
    !      DATA CONST
    !
    !      RAD=DEGRAD
    !
    logical orderi8
    external orderi8
    !
    !     Set up a temporary code numbers array.
    !
    K=0
    DO I=1,MAXATC
       IOFF(I)=K
       K=K+I
    ENDDO
    !
    IF(IOLEV>0 &
#if KEY_STRINGM==1 /*  VO */
    &          .or.qstr &
#endif
    &                     ) THEN
       !
       !-----------------------------------------------------------------------
       !         Begin Procedure SORT-PARAMETERS
       !
       !         Flag multiple-term dihedrals
       !
#if KEY_FLEXPARM==1
       IF(QFLXPARM) THEN
          DO I=1,NCP-1
             IF(CPAI(I) == CPAI(I+1)) THEN
                IF(CPAJ(I) == CPAJ(I+1)) THEN
                   IF(CPAK(I) == CPAK(I+1)) THEN
                      IF(CPAL(I) == CPAL(I+1)) CPD(I) = - ABS(CPD(I))
                   ENDIF
                ENDIF
             ENDIF
          ENDDO
       ELSE
#endif
          DO I=1,NCP-1
             IF (KCP(I) == KCP(I+1)) CPD(I) = - ABS(CPD(I))
          ENDDO
#if KEY_FLEXPARM==1
       ENDIF
#endif
       !
       !         Done reading data, now sort the code arrays by key number.
       !
       ! bonds
       CALL SORTP(NCB,IDX,ORDER,KCB,1,0,0,0,0,0,0)
       CALL AINDX4(IDX,KCB,NCB,TMPI)
       CALL AINDEX(IDX,CBB,NCB,TEMP)
       CALL AINDEX(IDX,CBC,NCB,TEMP)
#if KEY_FLEXPARM==1 /*flex_bond*/
       IF(QFLXPARM) THEN
          CALL AINDX4(IDX,CBAI,NCB,TEMP)
          CALL AINDX4(IDX,CBAJ,NCB,TEMP)
       ENDIF
#endif /* (flex_bond)*/
       ! angles
       CALL SORTP(NCT,IDX,ORDER,KCT,1,0,0,0,0,0,0)
       CALL AINDX4(IDX,KCT,NCT,TMPI)
       CALL AINDEX(IDX,CTB,NCT,TEMP)
       CALL AINDEX(IDX,CTC,NCT,TEMP)
       CALL AINDEX(IDX,CTUB,NCT,TEMP)
       CALL AINDEX(IDX,CTUC,NCT,TEMP)
#if KEY_FLEXPARM==1 /*flex_angle*/
       IF(QFLXPARM) THEN
          CALL AINDX4(IDX,CTAI,NCT,TMPI)
          CALL AINDX4(IDX,CTAJ,NCT,TMPI)
          CALL AINDX4(IDX,CTAK,NCT,TMPI)
       ENDIF
#endif /* (flex_angle)*/
       ! dihedrals
       CALL SRT8P(NCP,IDX,ORDERi8,KCP,1,0,0,0,0,0,0)
       CALL AINDX8(IDX,KCP,NCP,tmpi8)
       CALL AINDEX(IDX,CPB,NCP,TEMP)
       CALL AINDEX(IDX,CPC,NCP,TEMP)
       CALL AINDX4(IDX,CPD,NCP,TMPI)
#if KEY_FLEXPARM==1 /*flex_dihedral*/
       IF(QFLXPARM) THEN
          CALL AINDX4(IDX,CPAI,NCP,TMPI)
          CALL AINDX4(IDX,CPAJ,NCP,TMPI)
          CALL AINDX4(IDX,CPAK,NCP,TMPI)
          CALL AINDX4(IDX,CPAL,NCP,TMPI)
       ENDIF
#endif /* (flex_dihedral)*/
       ! cmap
#if KEY_CMAP==1
       CALL SRT8P(NCTP,IDX,ORDERi8,KCTP,1,0,0,0,0,0,0)
       CALL AINDX8(IDX,KCTP,NCTP,tmpi8)
       CALL AINDX4(IDX,GSCTP,NCTP,TMPI)
       CALL AINDX_CMAP(IDX,MCTP,NCTP)
#if KEY_FLEXPARM==1 /*flex_cmap*/
       IF(QFLXPARM) THEN
          DO I=1,8
             CALL AINDX4(IDX,CTPA(1,I),NCTP,TMPI)
          ENDDO
       ENDIF
#endif /* (flex_cmap)*/
#endif
       ! imph
       CALL SRT8P(NCI,IDX,ORDERi8,KCI,1,0,0,0,0,0,0)
       CALL AINDX8(IDX,KCI,NCI,tmpi8)
       CALL AINDEX(IDX,CIB,NCI,TEMP)
       CALL AINDEX(IDX,CIC,NCI,TEMP)
       CALL AINDX4(IDX,CID,NCI,TMPI)
#if KEY_FLEXPARM==1 /*flex_imphi*/
       IF(QFLXPARM) THEN
          CALL AINDX4(IDX,CIAI,NCI,TMPI)
          CALL AINDX4(IDX,CIAJ,NCI,TMPI)
          CALL AINDX4(IDX,CIAK,NCI,TMPI)
          CALL AINDX4(IDX,CIAL,NCI,TMPI)
       ENDIF
#endif /* (flex_imphi)*/
       ! hbond
       CALL SORTP(NCH,IDX,ORDER,KCH,1,0,0,0,0,0,0)
       CALL AINDX4(IDX,KCH,NCH,TMPI)
       CALL AINDEX(IDX,CHBA,NCH,TEMP)
       CALL AINDEX(IDX,CHBB,NCH,TEMP)
#if KEY_FLEXPARM==1 /*flex_hbond*/
       IF(QFLXPARM) THEN
          CALL AINDX4(IDX,CHAD,NCH,TMPI)
          CALL AINDX4(IDX,CHAA,NCH,TMPI)
       ENDIF
#endif /* (flex_hbond)*/
       !
       ! End of paramter sorting
       !
       ! check-for-repetitions
#if KEY_FLEXPARM==1
       IF(.NOT.QFLXPARM) THEN
#endif
          CALL CHCKREP(ICARD,IOFF)
#if KEY_FLEXPARM==1
       ENDIF
#endif
       !
    ENDIF
    !


#if KEY_PARALLEL==1 /*parallel_a*/
       CALL PSND4(NATC, 1)
       CALL PSND4(NCB, 1)
       CALL PSND4(NCT, 1)
       CALL PSND4(NCP, 1)
       CALL PSND4(NCI, 1)
       CALL PSND4(NCH, 1)
       CALL PSND4(KCB, NCB)
       CALL PSND4(KCT, NCT)
       CALL PSND8(KCP, NCP)
       CALL PSND8(KCI, NCI)
       CALL PSND4(KCH, NCH)
       CALL PSND4(HBEXPN, 4)
       CALL PSND8(CBC, NCB)
       CALL PSND8(CBB, NCB)
       CALL PSND8(CTC, NCT)
       CALL PSND8(CTB, NCT)
       CALL PSND8(CPC, NCP)
       CALL PSND4(CPD, NCP)
       CALL PSND8(CPB, NCP)
       CALL PSND8(CIC, NCI)
       CALL PSND4(CID, NCI)
       CALL PSND8(CIB, NCI)
       CALL PSND8(CHBA, NCH)
       CALL PSND8(CHBB, NCH)
       CALL PSND8(CTUB, NCT)
       CALL PSND8(CTUC, NCT)
       CALL PSNDC(ATC, NATC)

#if KEY_FLUCQ==1
       CALL PSND8(FQCHI, NATC)
       CALL PSND8(FQZETA, NATC)
       CALL PSND4(FQPRIN, NATC)
#endif

#if KEY_CMAP==1 /*cmap*/
       CALL PSND4(NCTP, 1)
       CALL PSND8(KCTP, NCTP)
       CALL PSND4(GSCTP, NCTP)
       DO I=1,NCTP
#if KEY_PARALLEL==1
          if (mynod /= 0) then
#endif
             call cmap_allocate_table(MCTP(I), GSCTP(I))
          ENDIF
          DO J = 1, 4
             CALL PSND8m(MCTP(I)%grid(J)%A, GSCTP(I)*GSCTP(I))
          ENDDO
       ENDDO
#endif /* (cmap)*/

#if KEY_FLEXPARM==1 /*send_flexparm*/
       IF(QFLXPARM) THEN
          CALL PSND4(NACTEQV,1)
          CALL PSND4(IACTEQV, NACTEQV)
          CALL PSNDC(ACTEQV, NACTEQV)
          DO I=1,NACTEQV
             CALL PSND4(ACTEQVM(1:NATC,I),NATC)
          ENDDO
          CALL PSND4(CBAI,NCB)
          CALL PSND4(CBAJ,NCB)
          CALL PSND4(CTAI,NCT)
          CALL PSND4(CTAJ,NCT)
          CALL PSND4(CTAK,NCT)
          CALL PSND4(CPAI,NCP)
          CALL PSND4(CPAJ,NCP)
          CALL PSND4(CPAK,NCP)
          CALL PSND4(CPAL,NCP)
          CALL PSND4(CIAI,NCI)
          CALL PSND4(CIAJ,NCI)
          CALL PSND4(CIAK,NCI)
          CALL PSND4(CIAL,NCI)
          CALL PSND4(CHAD,NCH)
          CALL PSND4(CHAA,NCH)
#if KEY_CMAP==1
          DO I=1,8
#endif
#if KEY_CMAP==1
             CALL PSND4(CTPA(1:NCTP,I),NCTP)
#endif
#if KEY_CMAP==1
          ENDDO
#endif
       ENDIF
#endif /* (send_flexparm)*/
       !
       CALL PSND4(DEFLTS, MAXDEF)
       CALL PSND4(DFUSED, MAXDEF)
       CALL PSND4(QNBFIX,1)
#endif /* (parallel_a)*/
       !
    DO I=1,NCP
       IF(CPB(I) == 0.0) THEN
          CPCOS(I)=ONE
          CPSIN(I)=ZERO
       ELSE IF(CPB(I) == PI) THEN
          CPCOS(I)=MINONE
          CPSIN(I)=ZERO
       ELSE
          CPCOS(I)=COS(CPB(I))
          CPSIN(I)=SIN(CPB(I))
       ENDIF
    ENDDO
    DO I=1,NCI
       IF(CIB(I) == 0.0) THEN
          CICOS(I)=ONE
          CISIN(I)=ZERO
       ELSE IF(CIB(I) == PI) THEN
          CICOS(I)=MINONE
          CISIN(I)=ZERO
       ELSE
          CICOS(I)=COS(CIB(I))
          CISIN(I)=SIN(CIB(I))
       ENDIF
    ENDDO
    ! AB.
    !
    RETURN
  END subroutine PARRDR2

  SUBROUTINE PARRDR3(ICARD,JUNIT)
    !
    ! This routine finishes the process of processing the parameters after
    ! all I/O is complete (reading parameters). - BRB
    !
    ! It performs the following tasks
    !   1. sets up vdw types
    !   2. sets up vdw tables
    !   3. Processes NBFIX items
    !
    use chm_kinds
    use number
    use dimens_fcm
    use param
    use consta
    use defltsm
    use stream
    use nbthole
#if KEY_PARALLEL==1
    use parallel
#endif
#if KEY_ENSEMBLE==1
    use ensemble
#endif
#if KEY_STRINGM==1 /*  VO */
    use multicom_aux
#endif
    !
    implicit none
    !
    INTEGER ICARD,JUNIT
    !
    ! local
    INTEGER NBC(MAXATC),IOFF(MAXATC)
    INTEGER PRLEV
    real(chm_real) :: CONST=362.3461_chm_real,RAD
    INTEGER I,J,K,I1,J1,IPT,NO
    real(chm_real)  E,R,EMIN,FNUM,VDW,R6,A,B
    real(chm_real)  E14,R14,EMIN14,FNUM14,VDW14,R614,A14,B14
    real(chm_real)  FDENOM,R6I,R6J
    !
#if KEY_STRINGM==1 /*  VO stringm */
    logical :: qstr
    common /replicaio/ qstr
#endif
    !
    RAD=DEGRAD
    !
    !     Set up a temporary code numbers array.
    !
    K=0
    DO I=1,MAXATC
       NBC(I)=0
       IOFF(I)=K
       K=K+I
    ENDDO
    !
    IF(IOLEV>0 &
#if KEY_STRINGM==1 /*  VO */
    &          .or.qstr &
#endif
    &                     ) THEN
       !-----------------------------------------------------------------------
       !
       PRLEV=ICARD-1
       IF(PRNLEV < 2) PRLEV=0
       !
       !         Begin Procedure SORT-NONBOND-TABLES
       K=NATVDW
       DO I=1,NATC
          IF(ATC(I) /= '    ') THEN
             IF(ITC(I) == 0) THEN
                K=NATVDW+1
                ITC(I)=K
                ALP(K)=0.0
                EFF(K)=0.0
                VDWR(K)=0.0
                ALP(K+MAXATC)=0.0
                EFF(K+MAXATC)=0.0
                VDWR(K+MAXATC)=0.0
                C6DEF(K)=0.0
                IF(WRNLEV >= 2) WRITE(OUTU,2000) I,ATC(I)(1:idleng)
2000            FORMAT('*****  WARNING  ***** PARRDR ', &
                     'no nonbond parameters for atom type:',I4,2X,A,/, &
                     ' NO nonbond interactions will be computed for', &
                     ' this atom type.')
             ENDIF
          ENDIF
       ENDDO
       NATVDW=K
       !
       DO I=1,NATVDW
          NBC(I)=-1
          DO J=NATC,1,-1
             IF(ITC(J) == I) NBC(I)=J
          ENDDO
          IF(NBC(I) <= 0) THEN
             IF(WRNLEV >= 4) THEN
                WRITE(OUTU,353) 'IATVDW',I
                WRITE(OUTU,353) 'NATVDW',NATVDW
                WRITE(OUTU,353) 'NBC',(NBC(J),J=1,NATVDW)
                WRITE(OUTU,353) 'NATC',NATC
                WRITE(OUTU,353) 'ITC',(ITC(J),J=1,NATC)
             ENDIF
             CALL WRNDIE(-3,'<PARRDR>', &
                  'Null nonbond group found. Redo.')
          ENDIF
353       FORMAT('PARRDR: ',A8,10I5,(/16X,10I5))
       ENDDO
       !
       NCN=0
       DO I=1,NATVDW
          DO J=1,I
             NCN=NCN+1
             IF(NBC(I) > 0 .AND. NBC(J)>0) THEN
                IF(MOD(NCN,50) == 1 .AND. PRLEV >= 2) WRITE(JUNIT,355)
355             FORMAT(/7X,'#',4X,'PAIR',5X,'CODE',9X,'A',9X,'B',5X, &
                     'SUM OF RADII',3X,'DEPTH OF WELL'/)
                IF (NBC(I) > NBC(J)) THEN
                   NO=IOFF(NBC(I))+NBC(J)
                ELSE
                   NO=IOFF(NBC(J))+NBC(I)
                ENDIF
                !
                VDW=VDWR(I)+VDWR(J)
                IF(DFGEOM) VDW=SQRT(VDWR(I)*VDWR(J))*TWO
                R6=VDW**6
                VDW14=VDWR(I+MAXATC)+VDWR(J+MAXATC)
                IF(DFGEOM) VDW14=SQRT(VDWR(I+MAXATC)*VDWR(J+MAXATC))*TWO
                R614=VDW14**6
                IF(R6 <= 0.000001) R6=0.000001
                IF(R614 <= 0.000001) R614=0.000001
                FNUM=CONST*ALP(I)*ALP(J)
                FNUM14=CONST*ALP(I+MAXATC)*ALP(J+MAXATC)
                IF (EFF(I) >= 0.5 .AND. EFF(J)>=0.5 .AND. FNUM /= 0.0) &
                     THEN
                   FDENOM=SQRT(ABS(ALP(I)/EFF(I)))+SQRT(ABS(ALP(J)/EFF(J) &
                        ))
                   B=FNUM/FDENOM
                   EMIN=-0.5*B/R6
                   FDENOM=SQRT(ABS(ALP(I+MAXATC)/EFF(I+MAXATC)))+ &
                        SQRT(ABS(ALP(J+MAXATC)/EFF(J+MAXATC)))
                   B14=FNUM14/FDENOM
                   EMIN14=-0.5*B14/R614
                ELSEIF (EFF(I) < 0.0 .AND. EFF(J)<0.0) THEN
                   EMIN=-SQRT(ABS(EFF(I)*EFF(J)))
                   B=-2.0*EMIN*R6
                   EMIN14=-SQRT(ABS(EFF(I+MAXATC)*EFF(J+MAXATC)))
                   B14=-2.0*EMIN14*R614
                ELSEIF (EFF(I) < 0.0 .AND. EFF(J) >= 0.5) THEN
                   R6J=(VDWR(J)+VDWR(J))**6
                   FNUM=0.25*CONST*ALP(J)*SQRT(ABS(ALP(J)*EFF(J)))/R6J
                   EMIN=-SQRT(ABS(EFF(I)*FNUM))
                   B=-2.0*EMIN*R6
                   R6J=(VDWR(J+MAXATC)+VDWR(J+MAXATC))**6
                   FNUM=0.25*CONST*ALP(J+MAXATC)* &
                        SQRT(ABS(ALP(J+MAXATC)*EFF(J+MAXATC)))/R6J
                   EMIN14=-SQRT(ABS(EFF(I+MAXATC)*FNUM))
                   B14=-2.0*EMIN14*R614
                ELSEIF (EFF(J) < 0.0 .AND. EFF(I) >= 0.5) THEN
                   R6I=(VDWR(I)+VDWR(I))**6
                   FNUM=0.25*CONST*ALP(I)*SQRT(ABS(ALP(I)*EFF(I)))/R6I
                   EMIN=-SQRT(ABS(EFF(J)*FNUM))
                   B=-2.0*EMIN*R6
                   R6I=(VDWR(I+MAXATC)+VDWR(I+MAXATC))**6
                   FNUM=0.25*CONST*ALP(I+MAXATC)* &
                        SQRT(ABS(ALP(I+MAXATC)*EFF(I+MAXATC)))/R6I
                   EMIN14=-SQRT(ABS(EFF(J+MAXATC)*FNUM))
                   B14=-2.0*EMIN14*R614
                ELSE
                   B=0.0
                   EMIN=0.0
                   B14=0.0
                   EMIN14=0.0
                ENDIF
                IF(NCN > MAXCN) THEN
                   IF(WRNLEV >= 2) WRITE(OUTU,129)
129                FORMAT(' PARRDR> EXCEEDED NON-BOND CODE TABLE SPACE')
                   CALL DIEWRN(-3)
                ENDIF
                !
                !             remove terms with a zero atomic radii
                IF(VDW < TENM5) THEN
                   VDW=ONE
                   EMIN=ZERO
                ENDIF
                IF(VDW14 < TENM5) THEN
                   VDW14=ONE
                   EMIN14=ZERO
                ENDIF
                CNBA(NCN)=VDW*VDW
                CNBB(NCN)=-EMIN
                !
                CNBA(NCN+MAXCN)=VDW14*VDW14
                CNBB(NCN+MAXCN)=-EMIN14
             ENDIF
          ENDDO
       ENDDO
       !
       !         PROCESS NBOND FIXES
       !
       DO K=1,NBFIXN
          I=NBFIXI(1,K)
          J=NBFIXI(2,K)
          E=NBFIXR(1,K)
          R=NBFIXR(2,K)
          E14=NBFIXR(3,K)
          R14=NBFIXR(4,K)
          E=-E
          E14=-E14
          R=R*R
          R14=R14*R14
          IF (I == 1000 .AND. J==1000) THEN
             IPT=0
             DO I=1,NATVDW
                DO J=1,I
                   IPT=IPT+1
                   CNBA(IPT)=R
                   CNBB(IPT)=E
                   CNBA(IPT+MAXCN)=R14
                   CNBB(IPT+MAXCN)=E14
                ENDDO
             ENDDO
          ELSEIF (I == 1000) THEN
             J1=ITC(J)
             DO I1=1,NATVDW
                IF (I1 > J1) THEN
                   IPT=IOFF(I1)+J1
                ELSE
                   IPT=IOFF(J1)+I1
                ENDIF
                CNBA(IPT)=R
                CNBB(IPT)=E
                CNBA(IPT+MAXCN)=R14
                CNBB(IPT+MAXCN)=E14
             ENDDO
          ELSE
             J1=ITC(J)
             I1=ITC(I)
             IF (I1 > J1) THEN
                IPT=IOFF(I1)+J1
             ELSE
                IPT=IOFF(J1)+I1
             ENDIF
             CNBA(IPT)=R
             CNBB(IPT)=E
             CNBA(IPT+MAXCN)=R14
             CNBB(IPT+MAXCN)=E14
          ENDIF
       ENDDO
       !
       !
       IF(PRLEV >= 2) THEN
          IPT=0
          DO I=1,NATVDW
             DO J=1,I
                IPT=IPT+1
                EMIN=-CNBB(IPT)
                VDW=SQRT(CNBA(IPT))
                R6=VDW**6
                B=-2.0*EMIN*R6
                A=0.5*B*R6
                EMIN14=-CNBB(IPT+MAXCN)
                VDW14=SQRT(CNBA(IPT+MAXCN))
                R614=VDW14**6
                B14=-2.0*EMIN14*R614
                A14=0.5*B14*R614
                WRITE(JUNIT,360) IPT, &
                     ATC(NBC(I))(1:idleng),ATC(NBC(J))(1:idleng), &
                     A,B,VDW,EMIN,A14,B14,VDW14,EMIN14
             ENDDO
          ENDDO
       ENDIF
360    FORMAT(5X,I4,2X,A,' : ',A,2(F13.2,F9.2,F12.3,F15.4))
       !         End Procedure SORT-NONBOND-TABLES
    ENDIF
    !

#if KEY_PARALLEL==1
    call psync()
    CALL PSND4(NATVDW,1)
    CALL PSND4(NCN, 1)
    CALL PSND4(ITC, MAXATC)
    CALL PSND8(ALP, NATVDW+MAXATC)
    CALL PSND8(EFF, NATVDW+MAXATC)
    CALL PSND8(VDWR,NATVDW+MAXATC)
    CALL PSND8(C6DEF,NATVDW)    
    CALL PSND8(CNBA,NCN+MAXCN)
    CALL PSND8(CNBB,NCN+MAXCN)
    CALL PSND4(NBTHOL,1)
    CALL PSND8(NBTHOLXIJ,NBTHOL)
    DO I=1,NBTHOL
    CALL PSND4(NBTHOLIJ(1,I),1)
    CALL PSND4(NBTHOLIJ(2,I),1)
    ENDDO
    CALL PSND8(THOLCUT,1)
#endif
    !
    RETURN
  END subroutine PARRDR3


  SUBROUTINE ADDNBF(I1,J1,E,R,E14,R14,AI,AJ,AK,AL,AX,JUNIT,PRLEV, &
       NBFIXNT)
    !-----------------------------------------------------------------------
    !     ADD-NBFIX-TO-TABLE
    !
    use chm_kinds
    use dimens_fcm
    use param
    use stream
    implicit none
    !
    CHARACTER(len=8) AI,AJ,AK,AL,AX
    INTEGER     I1,J1,JUNIT,PRLEV,NBFIXNT
    real(chm_real)      E,R,E14,R14
    !
    !..B980112.ep      NBFIXN=NBFIXN+1
    !..B980610.mc
    NBFIXN=NBFIXNT
    NBFIXN=NBFIXN+1
    !..
    IF(NBFIXN > MAXNBF) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,41) MAXNBF
41     FORMAT(' PARRDR> Error: NBFIX space exceeded.',1X,I7)
       CALL DIEWRN(-4)
    ENDIF
    IF (I1 > J1) THEN
       NBFIXI(1,NBFIXN)=I1
       NBFIXI(2,NBFIXN)=J1
    ELSE
       NBFIXI(2,NBFIXN)=I1
       NBFIXI(1,NBFIXN)=J1
    ENDIF
    NBFIXR(1,NBFIXN)=E
    NBFIXR(2,NBFIXN)=R
    NBFIXR(3,NBFIXN)=E14
    NBFIXR(4,NBFIXN)=R14
    AK=AX
    AL=AX
    !..B980610.mc
    NBFIXNT=NBFIXN
    !..
    IF(I1 <= NATC) AK=ATC(I1)
    IF(J1 <= NATC) AL=ATC(J1)
    IF(PRLEV >= 1) WRITE(JUNIT,380) NBFIXN,AK(1:idleng),AL(1:idleng), &
         E,R,E14,R14
380 FORMAT(3X,I4,2X,A,' - ',A,4F12.4)
    IF(E > 0.0 .OR. E14>0.0) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,318) AI(1:idleng),AJ(1:idleng),E,E14
318    FORMAT(' PARRDR> Warning (NBFIX): NONBOND FIX', &
            ' has a positive EMIN.',2(1X,A),2F12.5)
       CALL DIEWRN(-2)
    ENDIF
    RETURN
  END subroutine ADDNBF

  SUBROUTINE ADDTHOLE(I1,J1,R)
!-----------------------------------------------------------------------
!     ADD-NBTHOLE-TABLE
  use chm_kinds
  use number
  use dimens_fcm
  use stream
  use param
  use nbthole
  implicit none
!
      INTEGER     I1, J1
      real(chm_real)      R
!
      NBTHOL=NBTHOL+1

      if(NBTHOL > MAXTHOLE) then
         WRITE(OUTU,'(a,1x,i5)') &
        ' PARRDR> Error: NBTHOLE space exceeded.',MAXTHOLE
         CALL DIEWRN(-4)
      endif

      if(i1 > j1)then
         NBTHOLIJ(1,NBTHOL)=i1
         NBTHOLIJ(2,NBTHOL)=j1
      else
         NBTHOLIJ(2,NBTHOL)=i1
         NBTHOLIJ(1,NBTHOL)=j1
      endif
         NBTHOLXIJ(NBTHOL)=R

      RETURN
  END SUBROUTINE ADDTHOLE

  SUBROUTINE CHCKREP(ICARD,IOFF)
    !
    !     Check for repetitions in paramters (old method)
    !
    use chm_kinds
    use number
    use stream
    !
    use dimens_fcm
    use param
    implicit none
    !
    INTEGER       ICARD,IOFF(*)
    !
    !     local
    !
    CHARACTER(len=8) AI,AJ,AK,AL
    INTEGER     I,ITEMP,I1,J,J1,K1,L1,NOIJ,NOKL,T1
    LOGICAL     LREPET,OK,BAD,LMULT
    integer(chm_int8) :: natc2,no
    !
    NATC2=NATC*(NATC+1)/2
    LREPET=.FALSE.

    DO I=NCB,2,-1
       IF(KCB(I) == KCB(I-1)) THEN
          LREPET=.TRUE.
          I1=SQRT(TWO*KCB(I))+HALF
          J1=KCB(I)-I1*(I1-1)/2
          AI=ATC(I1)
          AJ=ATC(J1)
          IF(WRNLEV >= 2) WRITE(OUTU,822) KCB(I), &
               AI(1:idleng),AJ(1:idleng)
822       FORMAT(' PARRDR> Error: Repeated BOND parameter (',I6,'): ', &
               A,2X,A)

          ! copy values from the second to the first
          CBB(I-1)=CBB(I)
          CBC(I-1)=CBC(I)
       ENDIF
5555   CONTINUE
    ENDDO
    IF(LREPET) CALL DIEWRN(-1)
    !
    LREPET=.FALSE.
    DO I=NCT,2,-1
       IF(KCT(I) == KCT(I-1)) THEN
          LREPET=.TRUE.
          ITEMP=(KCT(I)-1)/NATC
          J1=KCT(I)-ITEMP*NATC
          I1=SQRT(TWO*ITEMP)+HALF
          K1=ITEMP-I1*(I1-1)/2
          AI=ATC(I1)
          AJ=ATC(J1)
          AK=ATC(K1)
          IF(WRNLEV >= 2) WRITE(OUTU,823) KCT(I), &
               AI(1:idleng),AJ(1:idleng),AK(1:idleng)
823       FORMAT(' PARRDR> Error: Repeated ANGLE parameter (',I6,'): ', &
               A,2X,A,2X,A)
          ! copy values from the second to the first
          CTB(I-1)=CTB(I)
          CTC(I-1)=CTC(I)
          CTUB(I-1)=CTUB(I)
          CTUC(I-1)=CTUC(I)
       ENDIF
    ENDDO
    IF(LREPET) CALL DIEWRN(-1)
    !
    !     Check for multiple dihedral entries and inform the user
    !     Kottalam,  June 10, 1990
    !     Check if multiple dihedrals have the same periodicity
    !     Remove older parameters if there are duplications.
    !
    LREPET=.FALSE.
    DO I=NCP,2,-1
       BAD=.FALSE.
       LMULT=.FALSE.
       IF(KCP(I) == KCP(I-1)) THEN
          ! Generate printing strings
          NO = KCP(I)
          OK = .TRUE.
          IF (NO > NATC2*NATC2) THEN
             NO = NO - NATC2*NATC2
             OK = .FALSE.
          ENDIF
          NOIJ = MOD(NO,NATC2)
          NOKL = (NO-NOIJ)/NATC2

          if (noij <= 0) then
             AJ = 'X'
             AK = 'X'
          else
             do j1 = 1, (maxatc - 1)
               if (ioff(j1) < noij .and. noij <= ioff(j1 + 1)) exit
             end do

             I1 = NOIJ - IOFF(J1)

             IF(.NOT.OK) THEN
                T1 = I1
                I1 = J1
                J1 = T1
             ENDIF

             AJ = ATC(I1)
             AK = ATC(J1)
          end if

          IF (NOKL <= 0) THEN
             AI = 'X'
             AL = 'X'
          ELSE
             do l1 = 1, (maxatc - 1)
               if (ioff(l1) < nokl .and. nokl <= ioff(l1 + 1)) exit
             end do

             K1 = NOKL-IOFF(L1)
             AI = ATC(K1)
             AL = ATC(L1)
          ENDIF
       ENDIF
       ! Loop over all subordinate pairs.
       DO J=I-1,1,-1
          IF(KCP(I) /= KCP(J)) GOTO 9900
          IF(CPD(J) >= 0) BAD=.TRUE.
          IF(BAD) THEN
             CPB(J)=0.0
             CPC(J)=0.0
             CPD(J)=-1
          ENDIF
          ! Check if multiple dihedrals have the same periodicity
          IF(ABS(CPD(I)) == ABS(CPD(J))) THEN
             IF(CPC(J) /= 0.0) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,824) I,KCP(I),ABS(CPD(I)), &
                     AI(1:idleng),AJ(1:idleng),AK(1:idleng),AL(1:idleng)
824             FORMAT(' PARRDR> ERROR: Repeated torsion periodicity:', &
                     ' INDEX',I5,' CODE',I10,' PERIODICITY',I3,5X, &
                     A,'-',A,'-',A,'-',A)
             ENDIF
          ENDIF
          ! flag normal multiple termn dihedrals
          IF(CPD(J) < 0) LMULT=.TRUE.
       ENDDO
9900   CONTINUE
       IF(BAD) THEN
          LREPET=.TRUE.
          IF(WRNLEV >= 2) WRITE(OUTU,827) I,KCP(I),AI(1:idleng), &
               AJ(1:idleng),AK(1:idleng),AL(1:idleng)
827       FORMAT(' PARRDR> ERROR: Repeated dihedral term:', &
               ' INDEX',I5,' CODE',I10,5X,A,'-',A,'-',A,'-',A)
       ENDIF
       IF(LMULT) THEN
          IF(WRNLEV >= 6) WRITE(OUTU,101) I,KCP(I),AI(1:idleng), &
               AJ(1:idleng),AK(1:idleng),AL(1:idleng)
101       FORMAT(' PARRDR> Multiple terms for dihedral type:', &
               ' INDEX',I5,' CODE',I10,5X,A,'-',A,'-',A,'-',A)
       ENDIF
    ENDDO
    !
7777 CONTINUE
    IF(LREPET) CALL DIEWRN(0)
    !
    ! Check improper dihedrals
    LREPET=.FALSE.
    DO I=NCI,2,-1
       IF(KCI(I) == KCI(I-1)) THEN
          LREPET=.TRUE.
          IF(WRNLEV >= 2) WRITE(OUTU,825) I,KCI(I)
825       FORMAT(' PARRDR> Error: Repeated IMPROPER parameter:', &
               ' INDEX',I5,' CODE',I10)
          ! copy values from the second to the first
          CIB(I-1)=CIB(I)
          CIC(I-1)=CIC(I)
          CID(I-1)=CID(I)
       ENDIF
    ENDDO
8888 CONTINUE
    IF(LREPET) CALL DIEWRN(0)
    !
    IF(ICARD > 0) THEN
       LREPET=.FALSE.
       DO I=NCH,2,-1
          IF(KCH(I) == KCH(I-1)) THEN
             LREPET=.TRUE.
             IF(WRNLEV >= 2) WRITE(OUTU,826) I,KCH(I)
826          FORMAT(' PARRDR> Error: Repeated H-BOND parameter:', &
                  ' INDEX',I5,' CODE',I10)
             ! copy values from the second to the first
             CHBA(I-1)=CHBA(I)
             CHBB(I-1)=CHBB(I)
          ENDIF
       ENDDO
       IF(LREPET) CALL DIEWRN(0)
    ENDIF
    !
    RETURN
  END subroutine CHCKREP
end module parmiom
