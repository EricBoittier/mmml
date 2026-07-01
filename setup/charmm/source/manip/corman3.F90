module corman3
  use chm_kinds
#if KEY_CHEQ==1
  use cheq,only:qcg,   &
     DCH,SUMDCH,DDCH
#endif
  implicit none

contains


  SUBROUTINE DISTN2(IUNIT,LSEL2,LMIND,LMAXD,LNEAR,X,Y,Z, &
       XCOMP,YCOMP,ZCOMP, &
       QETEN,QETSR,                                      &
       NATOM,ISEL,JSEL,CUT,QENER,QCLOSE,QEXCL,QTRI,QDIFF, &
       NATOMR,NNB14,INB14,IBLO14,EPS,E14FAC, &
       QHIST,HMIN,HMAX,HNUM,HPOINT,QHPRIN,HNORM,HDENS)
    !
    !     THIS ROUTINE CALCULATES THE DISTANCES BETWEEN ALL THE
    !     SELECTED ATOMS IN THE FIRST SET WITH THE SELECTED ATOMS OF THE
    !     SECOND LESS THAN CUT APART
    !
    !     By Bernard R. Brooks     3-DEC-1983
    !

  use chutil, only: initia
  use consta
  use exfunc
  use number
  use param_store, only: set_param
  use stream

  implicit none

    INTEGER IUNIT
    LOGICAL LSEL2,LMIND,LMAXD,LNEAR
    real(chm_real) X(*),Y(*),Z(*),XCOMP(*),YCOMP(*),ZCOMP(*)
    INTEGER NATOM
    INTEGER ISEL(*),JSEL(*)
    real(chm_real) CUT
    LOGICAL QENER,QCLOSE,QEXCL(3),QTRI,QDIFF
    INTEGER NATOMR,NNB14
    INTEGER INB14(*),IBLO14(*)
    real(chm_real)  EPS,E14FAC
    LOGICAL QHIST,QHPRIN
    INTEGER HNUM,HPOINT(*)
    real(chm_real)  HMIN,HMAX,HNORM,HDENS
    !
    INTEGER IMIN,JMIN,I,J,K,IEXCL,NUMEXL(3)
    INTEGER IMAX,JMAX
    real(chm_real) CUTSQ,S2MIN,DX,DY,DZ,S2,S,S2MAX
    real(chm_real) XMIN,XMAX,XAVE,YNORM,GOFR,BINW
    real(chm_real) EVDWR,EVDWT,ELECT
    LOGICAL LPRNT
    LOGICAL QETEN, QETSR
    INTEGER ICOUNT ! added by B. Roux
    !
    LPRNT=(IUNIT == OUTU .AND. PRNLEV >= 5)
    IF(IUNIT /= OUTU .AND. IOLEV > 0) LPRNT=.TRUE.
    !
    IF(.NOT.LSEL2 .AND. QENER) THEN
       CALL WRNDIE(0,'<DISTAN>', &
            'Double selection needed for ENERgy option')
       RETURN
    ENDIF
    !
    CUTSQ=CUT*CUT
    EVDWR=ZERO
    EVDWT=ZERO
    ELECT=ZERO
    !
    S2MIN=HALF*ANUM**2
    IMIN=0
    JMIN=0
    ICOUNT=0
    S2MAX=0
    IMAX=0
    JMAX=0
    !
    IF(LPRNT) THEN
       IF(LMIND) THEN
          WRITE(IUNIT,24)
24        FORMAT(/12X,'MINIMUM DISTANCE FOR ALL SELECTED ATOMS')
       ELSE IF(LMAXD) THEN
          WRITE(IUNIT,224)
224       FORMAT(/12X,'MAXIMUM DISTANCE FOR ALL SELECTED ATOMS')
       ELSE IF(LNEAR) THEN
          WRITE(IUNIT,23)
23        FORMAT(/12X,'MINIMUM DISTANCE FOR EACH SELECTED ATOMS')
       ELSE
          WRITE(IUNIT,22)
22        FORMAT(/12X,'DISTANCES FOR SELECTED ATOMS')
          !        ENDIF
          !        ENDIF
       ENDIF
    ENDIF
    !
    IF(.NOT.LSEL2) THEN
       ! only one selection specified, use both coordinate sets
       IF(LPRNT) WRITE(IUNIT,26)
26     FORMAT(12X,'ONLY ONE SELECTION SET SPECIFIED, A COMPARISON ', &
            'WILL BE DONE')
    ENDIF
    !
    NUMEXL(1)=0
    NUMEXL(2)=0
    NUMEXL(3)=0
    !
    DO I=1,NATOM
       IF(ISEL(I) == 1 .AND. INITIA(I,X,Y,Z)) THEN
          IF(LNEAR) THEN
             S2MIN=ANUM
             IMIN=0
             JMIN=0
          ENDIF
          DO J=1,NATOM
             IF(LSEL2) THEN
                IF(JSEL(J) /= 1) GOTO 100
                IF(.NOT.INITIA(J,X,Y,Z)) GOTO 100
                IF(QTRI .AND. ISEL(J) == 1 .AND. I >= J) GOTO 100
                IF(QDIFF) THEN
                   DX=X(I)-XCOMP(J)
                   DY=Y(I)-YCOMP(J)
                   DZ=Z(I)-ZCOMP(J)
                ELSE
                   DX=X(I)-X(J)
                   DY=Y(I)-Y(J)
                   DZ=Z(I)-Z(J)
                ENDIF
                !
             ELSE
                IF(.NOT.(ISEL(J) == 1 .AND. &
                     INITIA(J,XCOMP,YCOMP,ZCOMP))) GOTO 100
                DX=X(I)-XCOMP(J)
                DY=Y(I)-YCOMP(J)
                DZ=Z(I)-ZCOMP(J)
             ENDIF
             S2=DX*DX+DY*DY+DZ*DZ
             IF(LMIND .OR. LNEAR) THEN
                IF(S2 < S2MIN) THEN
                   IEXCL=QEXCLT(I,J,NATOMR,NNB14,INB14,IBLO14)
                   IF(.NOT.QEXCL(IEXCL)) GOTO 100
                   S2MIN=S2
                   IMIN=I
                   JMIN=J
                ENDIF
             ELSE
                !     Denzil----------------------------------------------------
                !                  IF(S2 <= CUTSQ) THEN
                IF((S2 <= CUTSQ) .AND. .NOT. LMAXD) THEN
                   !     Denzil----------------------------------------------------
                   IEXCL=QEXCLT(I,J,NATOMR,NNB14,INB14,IBLO14)
                   NUMEXL(IEXCL)=NUMEXL(IEXCL)+1
                   IF(.NOT.QEXCL(IEXCL)) GOTO 100
                   S=SQRT(S2)
                   !
                   ICOUNT=ICOUNT+1
                   CALL PRNDIST(LPRNT,IUNIT,QENER,QCLOSE, &
                        QHIST,HMIN,HMAX,HNUM,HPOINT,I,J, &
                        QETEN,QETSR,                    &
                        IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
                ENDIF
             ENDIF
             ! Denzil----------------------------------------------------
             !
             IF(LMAXD) THEN
                IF(S2 > S2MAX) THEN
                   IEXCL=QEXCLT(I,J,NATOMR,NNB14,INB14,IBLO14)
                   IF(.NOT.QEXCL(IEXCL)) GOTO 100
                   S2MAX=S2
                   IMAX=I
                   JMAX=J
                ENDIF
                !               ELSE
                !     Denzil----------------------------------------------------
                !                  IF(S2 <= CUTSQ) THEN
                !                  IF((S2 <= CUTSQ) .AND. .NOT. LMIND) THEN
                !     Denzil----------------------------------------------------
                !                     IEXCL=QEXCLT(I,J,NATOMR,NNB14,INB14,IBLO14)
                !                     NUMEXL(IEXCL)=NUMEXL(IEXCL)+1
                !                     IF(.NOT.QEXCL(IEXCL)) GOTO 100
                !                     S=SQRT(S2)
                !
                !                     ICOUNT=ICOUNT+1
                !                     CALL PRNDIST(LPRNT,IUNIT,QENER,QCLOSE,
                !     &                            QHIST,HMIN,HMAX,HNUM,HPOINT,I,J,
                !     &                            QETEN,QETSR,
                !     &                            IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
                !                  ENDIF
             ENDIF
             !
             ! Denzil----------------------------------------------------

100          CONTINUE
          ENDDO
          IF(LNEAR) THEN
             IF(IMIN > 0) THEN
                S=SQRT(S2MIN)
                CALL PRNDIST(LPRNT,IUNIT,QENER,QCLOSE, &
                     QHIST,HMIN,HMAX,HNUM,HPOINT,IMIN,JMIN, &
                     QETEN,QETSR,                      &
                     IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    IF(LMIND) THEN
       IF(IMIN > 0) THEN
          S=SQRT(S2MIN)
          call set_param('MIND',S)
          CALL PRNDIST(LPRNT,IUNIT,QENER,QCLOSE, &
               QHIST,HMIN,HMAX,HNUM,HPOINT,IMIN,JMIN, &
               QETEN,QETSR,                        &
               IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
          CALL set_param('MINDA1',IMIN)
          CALL set_param('MINDA2',JMIN)
       ELSE
          CALL WRNDIE(1,'<DISTAN>','NO DEFINABLE DISTANCE')
       ENDIF

       !  Denzil----------------------------------------------------
       !
       !      ELSE IF(.NOT.LNEAR) THEN
    ELSE IF(.NOT.LNEAR .AND. .NOT. LMAXD) THEN
       !
       !  Denzil----------------------------------------------------

       IF(LPRNT) WRITE(IUNIT,143) NUMEXL
143    FORMAT(' TOTAL EXCLUSION COUNT =',I10/ &
            ' TOTAL 1-4 EXCLUSIONS  =',I10/ &
            ' TOTAL NON-EXCLUSIONS  =',I10)
    ENDIF

    ! Denzil----------------------------------------------------
    !
    IF(LMAXD) THEN
       IF(IMAX > 0) THEN
          S=SQRT(S2MAX)
          call set_param('MAXD',S)
          CALL PRNDIST(LPRNT,IUNIT,QENER,QCLOSE, &
               QHIST,HMIN,HMAX,HNUM,HPOINT,IMAX,JMAX, &
               QETEN,QETSR,                        &
               IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
          CALL set_param('MAXDA1',IMAX)
          CALL set_param('MAXDA2',JMAX)
       ELSE
          CALL WRNDIE(1,'<DISTAN>','NO DEFINABLE DISTANCE')
       ENDIF
       !      ELSE IF(.NOT.LNEAR) THEN
       !         IF(LPRNT) WRITE(IUNIT,243) NUMEXL
       ! 243     FORMAT(' TOTAL EXCLUSION COUNT =',I10/
       !     1          ' TOTAL 1-4 EXCLUSIONS  =',I10/
       !     2          ' TOTAL NON-EXCLUSIONS  =',I10)
    ENDIF
    !
    ! Denzil----------------------------------------------------

    !
    IF(QENER .AND. LPRNT) WRITE(IUNIT,144) EVDWR,EVDWT,ELECT
144 FORMAT(' TOTAL VDW REPULSIVE ENERGY =',F13.4/ &
         ' TOTAL VDW           ENERGY =',F13.4/ &
         ' TOTAL ELECTROSTATIC ENERGY =',F13.4)
    !
    GOFR=ZERO
    IF(QHIST) THEN
       IF(QHPRIN) THEN
          WRITE(OUTU,245) ' Final normalized histogram of distances:'
          BINW=(HMAX-HMIN)/HNUM
          DO I=1,HNUM
             XMIN=HMIN+(I-1)*BINW
             XMAX=XMIN+BINW
             S=HPOINT(I)/HNORM
             IF(HDENS > 0.0) GOFR=S*3/4/PI/HDENS/(XMAX**3-XMIN**3)
             WRITE(OUTU,244) (XMIN+XMAX)/2,GOFR,S
244          FORMAT(3F15.5)
          ENDDO
          WRITE(OUTU,245) ' '
245       FORMAT(A)
       ENDIF
    ENDIF
    CALL set_param('NPAIR',ICOUNT)
    !
    RETURN
  END SUBROUTINE DISTN2

  SUBROUTINE DISTN3(IUNIT,LSEL2,X,Y,Z,ISEL,JSEL,CUT)
    !
    ! Minimum distance between each pair of residues from two atom
    ! selections is calculated and printed. Only pairs that are within
    ! distance CUT are considered.  This subroutine assumes that residue
    ! are listed consecutively in the selection, which should be the case.
    !
    !  2014 By Wonmuk Hwang (hwm)
    !
  use exfunc
  use number
  use consta
  use memory
    !
  use stream
  use chutil, only: initia,atomid,getres
  use psf
  use param_store, only: set_param

    INTEGER IUNIT
    LOGICAL LSEL2,LMIND
    real(chm_real) X(*),Y(*),Z(*)
    INTEGER ISEL(*),JSEL(*)
    real(chm_real) CUT
    integer i,j,k
    LOGICAL LPRNT
    INTEGER ICOUNT
    !
    character(len=8) isid,irid,iren,iat,jsid,jrid,jren,jat;
    character(len=8) icurr_resi,jcurr_resi,icurr_segi,jcurr_segi
    integer,allocatable, dimension(:):: iatom,jatom
    character(len=8),allocatable,dimension(:):: iresi,jresi,isegi,jsegi
    integer insel,jnsel,inatom,jnatom,i1,j1,i2,j2
    integer imin,jmin
    real(chm_real) cutsq,s2min,dx,dy,dz,s2,s
    !
    LPRNT=(IUNIT == OUTU .AND. PRNLEV >= 4)
    IF(IUNIT /= OUTU .AND. IOLEV > 0) LPRNT=.TRUE.
    !
    CUTSQ=CUT*CUT
    ICOUNT=0
    !
    IF(.NOT.LSEL2) THEN
       ! only one selection specified, use both coordinate sets
       IF(LPRNT) WRITE(IUNIT,26)
26     FORMAT(12X,'CALCULATING DISTANCES WITHIN SINGLE SELECTION.')
    ENDIF
    !
    IF(LPRNT) THEN
       WRITE(IUNIT,24) cut
24     FORMAT(/12X,'MIN DISTANCE BETWEEN RESIDUE PAIRS',&
            ' WITHIN ', F7.3,' A.')
    ENDIF
    !
    insel=0; jnsel=0 ! count number of selected atoms
    do i=1,natom  ! extract selected atoms and their resids
       if (isel(i)==1 .and. initia(i,x,y,z)) then
          insel=insel+1
          if (.not.lsel2) jnsel=jnsel+1
       endif
       if (lsel2) then
          if (jsel(i)==1 .and. initia(i,x,y,z)) jnsel=jnsel+1
       endif
    enddo

    ! allocate memory
    call chmalloc('corman3.src','DISTN3','iatom',insel,intg=iatom)
    call chmalloc('corman3.src','DISTN3','jatom',jnsel,intg=jatom)
    call chmalloc('corman3.src','DISTN3','iresi',insel,ch8=iresi)
    call chmalloc('corman3.src','DISTN3','jresi',jnsel,ch8=jresi)
    call chmalloc('corman3.src','DISTN3','isegi',insel,ch8=isegi)
    call chmalloc('corman3.src','DISTN3','jsegi',jnsel,ch8=jsegi)

    i1=0; j1=0;
    do i=1,natom  ! extract selected atoms and their resids
       if (isel(i)==1 .and. initia(i,x,y,z)) then
          i1=i1+1
          call atomid(i,isid,irid,iren,iat)
          iatom(i1)=i; iresi(i1)=irid; isegi(i1)=isid
          if (.not.lsel2) then
             j1=j1+1
             jatom(j1)=i; jresi(j1)=irid; jsegi(j1)=isid
          endif
       endif
       if (lsel2) then
          if (jsel(i)==1 .and. initia(i,x,y,z)) then
             j1=j1+1
             call atomid(i,isid,irid,iren,iat)
             jatom(j1)=i; jresi(j1)=irid; jsegi(j1)=isid
          endif
       endif
    enddo

    i1=0
    do i=1,insel ! go over 1st selection
      if (i .lt. i1) cycle ! skip already considered residue
      icurr_resi=iresi(i); icurr_segi=isegi(i)
      inatom=1 ! number of selected atoms in a resi
      do i1=i,insel ! find range for a given resid, which is (i,i1-1)
         if ((iresi(i1) .ne. icurr_resi) .or. (isegi(i1) .ne. icurr_segi)) then
            exit
         else
            inatom=inatom+1
         endif
      enddo
      j1=0;
      do j=1,jnsel ! go over 2nd selection
         if (j .lt. j1) cycle ! skip already considered residue
         jcurr_resi=jresi(j); jcurr_segi=jsegi(j)
         jnatom=1 ! number of selected atoms in a resi
         do j1=j,jnsel ! find range for a given resid, which is (j,j1-1)
            if ((jresi(j1) .ne. jcurr_resi) .or. &
                 (jsegi(j1) .ne. jcurr_segi)) then
               exit
            else
               jnatom=jnatom+1
            endif
         enddo
         if (isegi(i) .eq. jsegi(j)) then ! avoid the same residue
            i2=getres(iatom(i),ibase,nres)
            j2=getres(jatom(j),ibase,nres)
            if (i2 .eq. j2) cycle
         endif
         ! With a single selection, for the same segi, only consider
         ! the case when the first IRES is less than the second
         if (.not.lsel2) then
            if (isegi(i) .eq. jsegi(j)) then ! avoid the same residue
               i2=getres(iatom(i),ibase,nres)
               j2=getres(jatom(j),ibase,nres)
               if (i2 .ge. j2 ) cycle
            endif
         endif

         ! now take distance between icurr_resi and jcurr_resi
         imin=0; jmin=0; s2min=anum;
         do i2=i,(i1-1)
           do j2=j,(j1-1)
              dx=x(iatom(i2))-x(jatom(j2)); if (abs(dx)>cut) cycle
              dy=y(iatom(i2))-y(jatom(j2)); if (abs(dy)>cut) cycle
              dz=z(iatom(i2))-z(jatom(j2)); if (abs(dz)>cut) cycle
              s2=dx*dx+dy*dy+dz*dz
              if(s2 < s2min) then
                 s2min=s2; imin=i2; jmin=j2
              endif
           enddo ! do j2=j,j1
         enddo ! do i2=i,i1
         !
         if (s2min <= cutsq) then
            s=sqrt(s2min)
            icount=icount+1
            call atomid(iatom(imin),isid,irid,iren,iat)
            call atomid(jatom(jmin),jsid,jrid,jren,jat)
            if(lprnt) write(iunit,333) iatom(imin),isid(1:idleng), &
                 iren(1:idleng),irid(1:idleng),iat(1:idleng), &
                 jatom(jmin),jsid(1:idleng),jren(1:idleng),jrid(1:idleng), &
                 jat(1:idleng),s
333         format(i6,4(1x,a),' - ',i6,4(1x,a),3f12.4)
         endif ! i1(f (s2min <= cutsq) then
      enddo ! do j=1,jnsel ! go over 2nd selection
    enddo ! do i=1,insel ! go over 1st selection
    if(lprnt) then
       write(iunit,334) icount
334     format(12x,'Total number of distance pairs counted= ',I8)
    endif

    call set_param('NPAIR',icount)
    call chmdealloc('corman3.src','DISTN3','iatom',insel,intg=iatom)
    call chmdealloc('corman3.src','DISTN3','jatom',jnsel,intg=jatom)
    call chmdealloc('corman3.src','DISTN3','iresi',insel,ch8=iresi)
    call chmdealloc('corman3.src','DISTN3','jresi',jnsel,ch8=jresi)
    call chmdealloc('corman3.src','DISTN3','isegi',insel,ch8=isegi)
    call chmdealloc('corman3.src','DISTN3','jsegi',jnsel,ch8=jsegi)
    RETURN
  END SUBROUTINE DISTN3

  SUBROUTINE EDIST(I,J,R,EVDW,ELEC,EPS,E14FAC, &
       QETEN,QETSR,                          &
       IEXCL)
    !
    !     THIS ROUTINE COMPUTES THE PAIR ENERGY BETWEEN TWO ATOMS
    !     NO CUTOFF IS ATTEMPTED. NO CHECK OF THE EXCLUSION LIST IS DONE
    !
  use dimens_fcm
  use number
  use psf
  use param
  use consta

    INTEGER I,J
    real(chm_real) R,EVDW,ELEC,EPS,E14FAC
    INTEGER IEXCL
    !
    INTEGER I1,J1,IC
    real(chm_real) SIG6, FAC, FAC2
    real(chm_real) SIG2,SIG10
    LOGICAL QETEN, QETSR
    !
    !C  use stream
    !
    !     assume no energy or force if atoms very close (i.e. same atom)
    !     or if there is an exclusion between them.
    IF((IEXCL == 1) .OR. (R  <  0.001D0)) THEN
       ELEC=ZERO
       EVDW=ZERO
       RETURN
    ENDIF
    !
    !
    I1=ITC(IAC(I))
    J1=ITC(IAC(J))
    IF(I1 > J1) THEN
       IC=(I1*(I1-1))/2 +J1
    ELSE
       IC=(J1*(J1-1))/2 +I1
    ENDIF
    IF(IEXCL == 2) IC=IC+MAXCN
    IF (.NOT.QETEN .AND. .NOT. QETSR) THEN
       SIG6=(CNBA(IC)/(R*R))**3
       EVDW=CNBB(IC)*(SIG6*(SIG6-2.0))
    ELSE IF (QETSR) THEN
       SIG2=CNBA(IC)/(R*R)
       SIG6=SIG2*SIG2*SIG2
       SIG10=SIG2*SIG2*SIG2*SIG2*SIG2
       FAC=FOUR/(NINE*SIG2)
       FAC2=FAC*FAC
       EVDW=CNBB(IC)*(SIG6*(SIG2*SIG2*(SIG2*THIRTN-(NINE+NINE))+FOUR)) &
            / ( ONE + FAC2*FAC2*FAC2)
    ELSE ! QETEN
       SIG2=CNBA(IC)/(R*R)
       SIG6=SIG2*SIG2*SIG2
       SIG10=SIG2*SIG2*SIG2*SIG2*SIG2
       EVDW=CNBB(IC)*(SIG6*(SIG2*SIG2*(SIG2*THIRTN-(NINE+NINE))+FOUR))
    ENDIF
    !
    ELEC=CCELEC*CG(I)*CG(J)/(EPS*R)
    IF(IEXCL == 2) ELEC=ELEC*E14FAC
    RETURN
  END SUBROUTINE EDIST

  INTEGER FUNCTION QEXCLT(II,JJ,NATOMR,NNB14,INB14,IBLO14)
    !
    !     THIS ROUTINE DOES THE WORK OF QEXCLT.
    !
    INTEGER II,JJ,NATOMR,NNB14
    INTEGER INB14(*),IBLO14(*)
    !
    INTEGER IS,IQ,IPT,I,J
    !
    IF(II == JJ) THEN
       ! an atom is always excluded to itself
       QEXCLT=1
       RETURN
    ENDIF
    QEXCLT=3
    !     if exclusion list is not present, ignore exclusions
    IF(NNB14 <= 0) RETURN
    !     if images are used, don't worry about image exclusions...
    IF(II > NATOMR .OR. JJ > NATOMR) RETURN
    !
    !
    IF(II > JJ) THEN
       I=JJ
       J=II
    ELSE
       I=II
       J=JJ
    ENDIF
    IF(I == 1) THEN
       IS=1
    ELSE
       IS=IBLO14(I-1)+1
    ENDIF
    IQ=IBLO14(I)
    DO IPT=IS,IQ
       IF(INB14(IPT) == J) THEN
          QEXCLT=1
          RETURN
       ENDIF
       IF(INB14(IPT) == -J) THEN
          QEXCLT=2
          RETURN
       ENDIF
    ENDDO
    RETURN
  END FUNCTION QEXCLT

  SUBROUTINE PRNDIST(LPRNT,IUNIT, &
       QENER,QCLOSE,QHIST,HMIN,HMAX,HNUM,HPOINT,I,J, &
       QETEN,QETSR,                                  &
       IEXCL,S,EPS,E14FAC,EVDWR,EVDWT,ELECT)
    !
    !     This routine prints the distances between all the
    !     selected atoms. It also fils the histogram tables and
    !     accumulates pair total energies.
    !
    !     By Bernard R. Brooks     28-OCT-1992
    !
  use exfunc
  use number
    !
  use stream
  use chutil,only:atomid

    LOGICAL LPRNT
    LOGICAL QETEN,QETSR
    INTEGER IUNIT
    LOGICAL QENER,QCLOSE,QHIST
    INTEGER HNUM,HPOINT(*)
    real(chm_real)  HMIN,HMAX
    INTEGER I,J,IEXCL
    real(chm_real) S,EPS,E14FAC,EVDWR,EVDWT,ELECT
    !
    INTEGER K
    real(chm_real) S2,EVDW,ELEC
    character(len=8) ISID,IRID,IREN,IAT,JSID,JRID,JREN,JAT
    !
    IF(QENER) THEN
       CALL EDIST(I,J,S,EVDW,ELEC,EPS,E14FAC, &
            QETEN,QETSR,                                  &
            IEXCL)
       IF(QCLOSE.AND.(EVDW < ZERO)) RETURN
       CALL ATOMID(I,ISID,IRID,IREN,IAT)
       CALL ATOMID(J,JSID,JRID,JREN,JAT)
       IF(LPRNT) WRITE(IUNIT,33) I,ISID(1:idleng), &
            IREN(1:idleng),IRID(1:idleng),IAT(1:idleng), &
            J,JSID(1:idleng),JREN(1:idleng),JRID(1:idleng), &
            JAT(1:idleng),S,EVDW,ELEC
       EVDWT=EVDWT+EVDW
       IF(EVDW > ZERO) EVDWR=EVDWR+EVDW
       ELECT=ELECT+ELEC
       !
    ELSE
       CALL ATOMID(I,ISID,IRID,IREN,IAT)
       CALL ATOMID(J,JSID,JRID,JREN,JAT)
       IF(LPRNT) WRITE(IUNIT,33) I,ISID(1:idleng), &
            IREN(1:idleng),IRID(1:idleng),IAT(1:idleng), &
            J,JSID(1:idleng),JREN(1:idleng),JRID(1:idleng), &
            JAT(1:idleng),S
33     FORMAT(I5,4(1X,A),'-',I4,4(1X,A),3F12.4)
    ENDIF
    !
    IF(QHIST) THEN
       S2=(S-HMIN)/(HMAX-HMIN)
       IF(S2 >= 0.0 .AND. S2 < 1.0) THEN
          K=S2*HNUM+1
          HPOINT(K)=HPOINT(K)+1
       ENDIF
    ENDIF
    !
    RETURN
  END SUBROUTINE PRNDIST

  SUBROUTINE CORHIST(COMLYN,COMLEN,X,Y,Z,W,QWEIGH,ISLCT)
    !
    ! Computes histogram of particle density of selected atoms
    ! For now no attempt to handle images.
    ! LNI March 1998
    !
  use dimens_fcm
  use number
  use exfunc
    !
  use psf
  use stream
  use string
  use memory
  use param_store, only: set_param

    real(chm_real),allocatable,dimension(:),save :: HPOINT
    character(len=*) COMLYN
    INTEGER COMLEN, ISLCT(*)
    real(chm_real) X(*),Y(*),Z(*),W(*)
    LOGICAL QWEIGH
    !
    LOGICAL QHSAVE,QHPRIN
    real(chm_real) HMIN, HMAX, HNORM, HDENS
    INTEGER :: HNUM, IUNIT
    character(len=4) :: HMODE,WD
    character(len=16) :: HMODES='X   Y   Z   R   '
    integer,SAVE :: HLEN=0, HCOUNT
    character(len=4),save :: hmod1='    '

    ! At this point the HIST keyword has been parsed off COMLYN
    ! Now find out what type of histogram /X,Y,Z,R/ user wants
    !
    HMODE=NEXTA4(COMLYN,COMLEN)
    IF(INDEX(HMODES,HMODE) ==  0)THEN
       CALL WRNDIE(0,'<CORHIS>','Unknown histogram type: '//HMODE)
       RETURN
    ENDIF
    HMIN=GTRMF(COMLYN,COMLEN,'HMIN',ZERO)
    HMAX=GTRMF(COMLYN,COMLEN,'HMAX',ZERO)
    HNUM=GTRMI(COMLYN,COMLEN,'HNUM',0)
    IF(HMAX <= HMIN .OR. HNUM <= 0) THEN
       CALL WRNDIE(0,'<CORHIS>', &
            'Error in parsing histogram keywords.')
       RETURN
    ENDIF
    !
    QHSAVE=(INDXA(COMLYN,COMLEN,'HSAV') > 0)
    QHPRIN=(INDXA(COMLYN,COMLEN,'HPRI') > 0).AND.(PRNLEV > 2)
    HNORM=GTRMF(COMLYN,COMLEN,'HNOR',ONE)
    HDENS=GTRMF(COMLYN,COMLEN,'HDEN',ZERO)
    IUNIT=GTRMI(COMLYN,COMLEN,'IUNI',OUTU)
    !
    IF(HLEN == 0) THEN
       HCOUNT=0
       call chmalloc('corman3.src','CORHIST','HPOINT',HNUM,crl=HPOINT)
       HPOINT=ZERO
       HLEN=HNUM
       HMOD1=HMODE
    ELSE
       IF(HLEN /= HNUM) THEN
          call chmdealloc('corman3.src','CORHIST','HPOINT',HNUM,crl=HPOINT)
          CALL WRNDIE(-3,'<CORHIS>', &
               'Histogram does not have the same length as saved')
          call chmalloc('corman3.src','CORHIST','HPOINT',HNUM,crl=HPOINT)
          hpoint=zero
          HLEN=HNUM
          HCOUNT=0
       ENDIF
       IF(HMODE  /=  HMOD1)THEN
          CALL WRNDIE(-2,'<CORHIS>', &
               'Histogram "'//HMODE//'" different than saved: '//HMOD1)
          !
          ! We don't really allow this; if user wants to mix X,Y,Z histograms
          ! this can be done w/ appropriate coor commands
          RETURN
       ENDIF
    ENDIF
    CALL set_param('NCONFIG',HCOUNT)
    !
    CALL CORHIS2(IUNIT,X,Y,Z,W,NATOM,ISLCT,HMODE, &
         HMIN,HMAX,HNUM,HPOINT, &
         QHPRIN,QWEIGH,HNORM,HDENS)
    HCOUNT=HCOUNT+1
    !
    IF(.NOT.QHSAVE) THEN
       call chmdealloc('corman3.src','CORHIST','HPOINT',HNUM,crl=HPOINT)
       HLEN=0
       HMOD1='    '
    ENDIF
    CALL set_param('NCONFIG',HCOUNT)
    RETURN
  END SUBROUTINE CORHIST

  SUBROUTINE CORHIS2(IUNIT,X,Y,Z,W,NATOMX,ISLCT,HMODE, &
       HMIN,HMAX,HNUM,HPOINT, &
       QHPRIN,QWEIGH,HNORM,HDENS)
  use exfunc
  use number
  use consta
    !
  use stream
  use chutil,only:initia

    INTEGER IUNIT,NATOMX,HNUM,ISLCT(*)
    real(chm_real) X(*),Y(*),Z(*),W(*),HMIN,HMAX,HNORM,HDENS,HPOINT(*)
    LOGICAL QHPRIN,QWEIGH
    character(len=4) HMODE
    !
    character(len=4) GRHEAD
    LOGICAL QIUN
    INTEGER I,J
    real(chm_real) R,S,DH,BINW,XMAX,XMIN,GOFR
    !
    QIUN= (IUNIT /= OUTU) .AND. (IUNIT > 0)
    DH=HMAX-HMIN
    DO I=1,NATOMX
       IF(ISLCT(I) == 1 .AND. INITIA(I,X,Y,Z))THEN
          IF(HMODE == 'X   ')THEN
             S=(X(I)-HMIN)/DH
          ELSEIF(HMODE == 'Y   ')THEN
             S=(Y(I)-HMIN)/DH
          ELSEIF(HMODE == 'Z   ')THEN
             S=(Z(I)-HMIN)/DH
          ELSEIF(HMODE == 'R   ')THEN
             R=SQRT( X(I)*X(I)+Y(I)*Y(I)+Z(I)*Z(I) )
             S=(R-HMIN)/DH
          ENDIF
          IF(S > ZERO .AND. S < ONE)THEN
             J=S*HNUM+1
             IF(QWEIGH)THEN
                HPOINT(J)=HPOINT(J)+W(I)
             ELSE
                HPOINT(J)=HPOINT(J)+1
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    IF(QHPRIN) THEN
       GRHEAD='    '
       IF(HMODE == 'R    ') GRHEAD='G(R)'
       WRITE(OUTU,'(A)') &
            ' Final (normalized) atom histogram:'
       WRITE(OUTU,245) HMODE,GRHEAD
       WRITE(OUTU,'(A)')'-------------------------------------------'
       IF(QIUN) WRITE(IUNIT,245) HMODE,GRHEAD
245    FORMAT(8X,A,11X,'Density',10X,A)
       BINW=DH/HNUM
       DO I=1,HNUM
          XMIN=HMIN+(I-1)*BINW
          XMAX=XMIN+BINW
          S=HPOINT(I)/HNORM
          IF(HMODE == 'R   ')THEN
             GOFR=ZERO
             IF(HDENS > ZERO) GOFR=S*3/4/PI/HDENS/(XMAX**3-XMIN**3)
             WRITE(OUTU,244) (XMIN+XMAX)/2,S,GOFR
             IF(QIUN) WRITE(IUNIT,244) (XMIN+XMAX)/2,S,GOFR
          ELSE
             WRITE(OUTU,244) (XMIN+XMAX)/2,S
             IF(QIUN) WRITE(IUNIT,244) (XMIN+XMAX)/2,S
          ENDIF
244       FORMAT(3F15.5)
       ENDDO
       WRITE(OUTU,*) ' '
    ENDIF
    RETURN
  END SUBROUTINE CORHIS2

  SUBROUTINE SQUAREUP(XMAX,XMIN,YMAX,YMIN,ZMAX,ZMIN,XAVG, &
       YAVG,ZAVG,XX,YY,ZZ,NATOMX,ACTV,NACT)
    ! -----------------------------------------------------
    ! Does a quick alignment of the molecule with the nearest axis.
    !  --RJ Petrella 5.12.98
    ! Molecular coordinates first copied and zeroed to (xavg,yavg,zavg).
    ! The coordinates are then projected onto the xy plane. The molecular
    ! projection is checked to see whether it is "diagonal" : i.e.
    ! if its "corners" are in diagonally opposite quadrants of the x-y plane.
    ! This means that the points on the molecule defining xmax, xmin,ymax, and
    ! ymin must be in an arrangement where one of the x-extremes is
    ! paired with one of the y-extremes in a single quadrant, and the other
    ! x-extreme is paired with the other y-extreme in the opposite
    ! quadrant.  Otherwise, the molecule is not touched in this
    ! projection.  If it meets the diagonal criteria, however, it is
    ! rotated in the x-y plane, such that the diagonal of the rectangle
    ! circumscribing the molecular projection is aligned with the nearest
    ! axis (x or y).  After rotation, the above process of projection,
    ! testing for diagonality,  and rotation (if indicated) are repeated
    ! for the ZX and YZ projections.  The reoriented coordinates are then
    ! returned. The advantage of this algorithm is that it does not
    ! necessitate finding the long axis of the molecule.  It works best for
    ! linear or oblong shapes . The disadvantage is that it is probably
    ! less precise than reorientation via long-axis and it may not
    ! optimally orient highly asymmetric molecules.
    !
  use dimens_fcm
  use memory

    INTEGER ACTV(*)
    !
    INTEGER NACT,NATOMX
    !
    real(chm_real) AVAL,XR,YR,ZR,VOLPTS
    real(chm_real) XX(*),YY(*),ZZ(*),XAVG,YAVG,ZAVG
    real(chm_real) XMAX(3),XMIN(3),YMAX(3),YMIN(3),ZMAX(3),ZMIN(3)
    real(chm_real) XMAXL(3),XMINL(3),YMAXL(3),YMINL(3), &
         ZMAXL(3),ZMINL(3)
    INTEGER ACT,I,K,NUMBX,NUMBY,NUMBZ
    real(chm_real) XMAXY,XMAXZ,YMAXX,YMAXZ,ZMAXX,ZMAXY
    real(chm_real) XMINY,XMINZ,YMINX,YMINZ,ZMINX,ZMINY,LOWVOL
    real(chm_real),allocatable,dimension(:),target :: space_rl
    real(chm_real),pointer,dimension(:) :: XXX,YYY,ZZZ,XLO,YLO,ZLO
    !
    !      WRITE(6,*) 'SQUARING THINGS UP'
    ! Store original coordinates and center molecule at origin
    call chmalloc('corman3.src','squareup','space_rl',6*natomx,crl=space_rl)

    xxx => space_rl(1:natomx)
    yyy => space_rl(natomx+1:2*natomx)
    zzz => space_rl(2*natomx+1:3*natomx)
    xlo => space_rl(3*natomx+1:4*natomx)
    ylo => space_rl(4*natomx+1:5*natomx)
    zlo => space_rl(5*natomx+1:6*natomx)

    DO I = 1,NACT
       ACT = ACTV(I)
       XX(ACT) = XX(ACT) - XAVG
       YY(ACT) = YY(ACT) - YAVG
       ZZ(ACT) = ZZ(ACT) - ZAVG
       XLO(ACT) = XX(ACT)
       YLO(ACT) = YY(ACT)
       ZLO(ACT) = ZZ(ACT)
    ENDDO
    !
    XMIN(1) = XMIN(1) - XAVG
    XMAX(1) = XMAX(1) - XAVG
    XMIN(2) = XMIN(2) - YAVG
    XMAX(2) = XMAX(2) - YAVG
    XMIN(3) = XMIN(3) - ZAVG
    XMAX(3) = XMAX(3) - ZAVG
    !
    YMIN(1) = YMIN(1) - XAVG
    YMAX(1) = YMAX(1) - XAVG
    YMIN(2) = YMIN(2) - YAVG
    YMAX(2) = YMAX(2) - YAVG
    YMIN(3) = YMIN(3) - ZAVG
    YMAX(3) = YMAX(3) - ZAVG
    !
    ZMIN(1) = ZMIN(1) - XAVG
    ZMAX(1) = ZMAX(1) - XAVG
    ZMIN(2) = ZMIN(2) - YAVG
    ZMAX(2) = ZMAX(2) - YAVG
    ZMIN(3) = ZMIN(3) - ZAVG
    ZMAX(3) = ZMAX(3) - ZAVG
    DO K = 1,3
       XMINL(K) = XMIN(K)
       XMAXL(K) = XMAX(K)
       YMINL(K) = YMIN(K)
       YMAXL(K) = YMAX(K)
       ZMINL(K) = ZMIN(K)
       ZMAXL(K) = ZMAX(K)
    ENDDO
10  FORMAT(A,I4,A,F8.2,A,F8.2,A,F8.2)
    !
    XR = XMAX(1) - XMIN(1)
    YR = YMAX(2) - YMIN(2)
    ZR = ZMAX(3) - ZMIN(3)
    !  Store original volume of box circumscribing the points
    !  (NB this is the volume circumscribing the CENTERS
    !  of the atoms, not the whole atoms s)
    LOWVOL = XR*YR*ZR
    WRITE(6,11) 'VOL OF PTS BEFORE REORIENT= ',LOWVOL
11  FORMAT(A,F14.3)
    !
    !  TEST XY projection:
    IF ((XMAX(2) > 0).AND.(XMAX(1) > 0).AND.(YMAX(2) > 0) &
         .AND.(YMAX(1) > 0).AND.(XMIN(1) < 0).AND.(XMIN(2) < 0) &
         .AND.(YMIN(1) < 0).AND.(YMIN(2) < 0)) THEN
       !
       AVAL = (ATAN2(YMAX(2),XMAX(1)) + ATAN2(-YMIN(2),-XMIN(1)))/2
20     FORMAT(A,F8.5)
       DO K = 1,3
          XMIN(K) = 99999
          XMAX(K) = -99999
          YMIN(K) = 99999
          YMAX(K) = -99999
       ENDDO
       IF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             XXX(ACT) = XX(ACT)*COS(AVAL) - YY(ACT)*SIN(AVAL)
             YYY(ACT) = XX(ACT)*SIN(AVAL) + YY(ACT)*COS(AVAL)
             XX(ACT) = XXX(ACT)
             YY(ACT) = YYY(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          ZMAXX = ZMAX(1)*COS(AVAL) - ZMAX(2)*SIN(AVAL)
          ZMAXY = ZMAX(1)*SIN(AVAL) + ZMAX(2)*COS(AVAL)
          ZMINX = ZMIN(1)*COS(AVAL) - ZMIN(2)*SIN(AVAL)
          ZMINY = ZMIN(1)*SIN(AVAL) + ZMIN(2)*COS(AVAL)
          ZMAX(1) = ZMAXX
          ZMAX(2) = ZMAXY
          ZMIN(1) = ZMINX
          ZMIN(2) = ZMINY
          !
       ELSEIF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             XXX(ACT) = XX(ACT)*COS(AVAL) + YY(ACT)*SIN(AVAL)
             YYY(ACT) = -XX(ACT)*SIN(AVAL) + YY(ACT)*COS(AVAL)
             XX(ACT) = XXX(ACT)
             YY(ACT) = YYY(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          ZMAXX = ZMAX(1)*COS(AVAL) + ZMAX(2)*SIN(AVAL)
          ZMAXY = -ZMAX(1)*SIN(AVAL) + ZMAX(2)*COS(AVAL)
          ZMINX = ZMIN(1)*COS(AVAL) + ZMIN(2)*SIN(AVAL)
          ZMINY = -ZMIN(1)*SIN(AVAL) + ZMIN(2)*COS(AVAL)
          ZMAX(1) = ZMAXX
          ZMAX(2) = ZMAXY
          ZMIN(1) = ZMINX
          ZMIN(2) = ZMINY
       ENDIF
25     FORMAT(A,I4,A,F8.2)
    ELSEIF ((XMAX(1) > 0).AND.(XMAX(2) < 0).AND.(YMIN(1) &
         > 0).AND.(YMIN(2) < 0).AND.(XMIN(1) < 0).AND. &
         (XMIN(2) > 0).AND.(YMAX(2) > 0).AND.(YMAX(1) < 0)) &
         THEN
       AVAL = (ATAN2(YMAX(2),-XMIN(1)) + ATAN2(-YMIN(2),XMAX(1))) &
            /2
       DO K = 1,3
          XMIN(K) = 99999
          XMAX(K) = -99999
          YMIN(K) = 99999
          YMAX(K) = -99999
       ENDDO
       !
       IF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             XXX(ACT) = XX(ACT)*COS(AVAL) - YY(ACT)*SIN(AVAL)
             YYY(ACT) = XX(ACT)*SIN(AVAL) + YY(ACT)*COS(AVAL)
             XX(ACT) = XXX(ACT)
             YY(ACT) = YYY(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          ZMAXX = ZMAX(1)*COS(AVAL) - ZMAX(2)*SIN(AVAL)
          ZMAXY = ZMAX(1)*SIN(AVAL) + ZMAX(2)*COS(AVAL)
          ZMINX = ZMIN(1)*COS(AVAL) - ZMIN(2)*SIN(AVAL)
          ZMINY = ZMIN(1)*SIN(AVAL) + ZMIN(2)*COS(AVAL)
          ZMAX(1) = ZMAXX
          ZMAX(2) = ZMAXY
          ZMIN(1) = ZMINX
          ZMIN(2) = ZMINY
          !
       ELSEIF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             XXX(ACT) = XX(ACT)*COS(AVAL) + YY(ACT)*SIN(AVAL)
             YYY(ACT) = -XX(ACT)*SIN(AVAL) + YY(ACT)*COS(AVAL)
             XX(ACT) = XXX(ACT)
             YY(ACT) = YYY(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          ZMAXX = ZMAX(1)*COS(AVAL) + ZMAX(2)*SIN(AVAL)
          ZMAXY = -ZMAX(1)*SIN(AVAL) + ZMAX(2)*COS(AVAL)
          ZMINX = ZMIN(1)*COS(AVAL) + ZMIN(2)*SIN(AVAL)
          ZMINY = -ZMIN(1)*SIN(AVAL) + ZMIN(2)*COS(AVAL)
          ZMAX(1) = ZMAXX
          ZMAX(2) = ZMAXY
          ZMIN(1) = ZMINX
          ZMIN(2) = ZMINY
       ENDIF
    ENDIF
    !
    XR = XMAX(1) - XMIN(1)
    YR = YMAX(2) - YMIN(2)
    ZR = ZMAX(3) - ZMIN(3)
    VOLPTS = XR*YR*ZR
    IF (VOLPTS < LOWVOL) THEN
       LOWVOL = VOLPTS
       DO I = 1,NACT
          ACT = ACTV(I)
          XLO(ACT) = XX(ACT)
          YLO(ACT) = YY(ACT)
          ZLO(ACT) = ZZ(ACT)
       ENDDO
       DO K = 1,3
          XMINL(K) = XMIN(K)
          XMAXL(K) = XMAX(K)
          YMINL(K) = YMIN(K)
          YMAXL(K) = YMAX(K)
          ZMINL(K) = ZMIN(K)
          ZMAXL(K) = ZMAX(K)
       ENDDO
    ENDIF
    DO I = 1,NACT
       ACT = ACTV(I)
       XX(ACT) = XLO(ACT)
       YY(ACT) = YLO(ACT)
       ZZ(ACT) = ZLO(ACT)
    ENDDO
    DO K = 1,3
       XMIN(K) = XMINL(K)
       XMAX(K) = XMAXL(K)
       YMIN(K) = YMINL(K)
       YMAX(K) = YMAXL(K)
       ZMIN(K) = ZMINL(K)
       ZMAX(K) = ZMAXL(K)
    ENDDO
    WRITE(6,11) 'VOL AFTER XY PROJ= ',LOWVOL
    !
    !  TEST ZX projection:
    IF ((ZMAX(3) > 0).AND.(ZMAX(1) > 0).AND.(XMAX(3) > 0) &
         .AND.(XMAX(1) > 0).AND.(ZMIN(1) < 0).AND.(ZMIN(3) < 0) &
         .AND.(XMIN(1) < 0).AND.(XMIN(3) < 0)) THEN
       AVAL = (ATAN2(XMAX(1),ZMAX(3)) + ATAN2(-XMIN(1),-ZMIN(3)))/2
       DO K = 1,3
          XMIN(K) = 99999
          XMAX(K) = -99999
          ZMIN(K) = 99999
          ZMAX(K) = -99999
       ENDDO
       IF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             ZZZ(ACT) = ZZ(ACT)*COS(AVAL) - XX(ACT)*SIN(AVAL)
             XXX(ACT) = ZZ(ACT)*SIN(AVAL) + XX(ACT)*COS(AVAL)
             ZZ(ACT) = ZZZ(ACT)
             XX(ACT) = XXX(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          YMAXZ = YMAX(3)*COS(AVAL) - YMAX(1)*SIN(AVAL)
          YMAXX = YMAX(3)*SIN(AVAL) + YMAX(1)*COS(AVAL)
          YMINZ = YMIN(3)*COS(AVAL) - YMIN(1)*SIN(AVAL)
          YMINX = YMIN(3)*SIN(AVAL) + YMIN(1)*COS(AVAL)
          YMAX(1) = YMAXX
          YMAX(3) = YMAXZ
          YMIN(1) = YMINX
          YMIN(3) = YMINZ
          !
       ELSEIF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             ZZZ(ACT) = ZZ(ACT)*COS(AVAL) + XX(ACT)*SIN(AVAL)
             XXX(ACT) = -ZZ(ACT)*SIN(AVAL) + XX(ACT)*COS(AVAL)
             ZZ(ACT) = ZZZ(ACT)
             XX(ACT) = XXX(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          YMAXZ = YMAX(3)*COS(AVAL) + YMAX(1)*SIN(AVAL)
          YMAXX = -YMAX(3)*SIN(AVAL) + YMAX(1)*COS(AVAL)
          YMINZ = YMIN(3)*COS(AVAL) + YMIN(1)*SIN(AVAL)
          YMINX = -YMIN(3)*SIN(AVAL) + YMIN(1)*COS(AVAL)
          YMAX(1) = YMAXX
          YMAX(3) = YMAXZ
          YMIN(1) = YMINX
          YMIN(3) = YMINZ
       ENDIF
    ELSEIF ((ZMAX(3) > 0).AND.(ZMAX(1) < 0).AND.(XMIN(3) > 0) &
         .AND.(XMIN(1) < 0).AND.(ZMIN(3) < 0).AND.(ZMIN(1) > 0) &
         .AND.(XMAX(1) > 0).AND.(XMAX(3) < 0)) THEN
       AVAL = (ATAN2(XMAX(1),-ZMIN(3)) + ATAN2(-XMIN(1),ZMAX(3))) &
            /2
       DO K = 1,3
          XMIN(K) = 99999
          XMAX(K) = -99999
          ZMIN(K) = 99999
          ZMAX(K) = -99999
       ENDDO
       IF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             ZZZ(ACT) = ZZ(ACT)*COS(AVAL) - XX(ACT)*SIN(AVAL)
             XXX(ACT) = ZZ(ACT)*SIN(AVAL) + XX(ACT)*COS(AVAL)
             ZZ(ACT) = ZZZ(ACT)
             XX(ACT) = XXX(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          YMAXZ = YMAX(3)*COS(AVAL) - YMAX(1)*SIN(AVAL)
          YMAXX = YMAX(3)*SIN(AVAL) + YMAX(1)*COS(AVAL)
          YMINZ = YMIN(3)*COS(AVAL) - YMIN(1)*SIN(AVAL)
          YMINX = YMIN(3)*SIN(AVAL) + YMIN(1)*COS(AVAL)
          YMAX(1) = YMAXX
          YMAX(3) = YMAXZ
          YMIN(1) = YMINX
          YMIN(3) = YMINZ
          !
       ELSEIF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             ZZZ(ACT) = ZZ(ACT)*COS(AVAL) + XX(ACT)*SIN(AVAL)
             XXX(ACT) = -ZZ(ACT)*SIN(AVAL) + XX(ACT)*COS(AVAL)
             ZZ(ACT) = ZZZ(ACT)
             XX(ACT) = XXX(ACT)
             IF (XX(ACT) > XMAX(1)) THEN
                XMAX(1) = XX(ACT)
                XMAX(2) = YY(ACT)
                XMAX(3) = ZZ(ACT)
             ENDIF
             IF (XX(ACT) < XMIN(1)) THEN
                XMIN(1) = XX(ACT)
                XMIN(2) = YY(ACT)
                XMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          YMAXZ = YMAX(3)*COS(AVAL) + YMAX(1)*SIN(AVAL)
          YMAXX = -YMAX(3)*SIN(AVAL) + YMAX(1)*COS(AVAL)
          YMINZ = YMIN(3)*COS(AVAL) + YMIN(1)*SIN(AVAL)
          YMINX = -YMIN(3)*SIN(AVAL) + YMIN(1)*COS(AVAL)
          YMAX(1) = YMAXX
          YMAX(3) = YMAXZ
          YMIN(1) = YMINX
          YMIN(3) = YMINZ
          !
       ENDIF
    ENDIF
    XR = XMAX(1) - XMIN(1)
    YR = YMAX(2) - YMIN(2)
    ZR = ZMAX(3) - ZMIN(3)
    VOLPTS = XR*YR*ZR
    IF (VOLPTS < LOWVOL) THEN
       LOWVOL = VOLPTS
       DO I = 1,NACT
          ACT = ACTV(I)
          XLO(ACT) = XX(ACT)
          YLO(ACT) = YY(ACT)
          ZLO(ACT) = ZZ(ACT)
       ENDDO
       DO K = 1,3
          XMINL(K) = XMIN(K)
          XMAXL(K) = XMAX(K)
          YMINL(K) = YMIN(K)
          YMAXL(K) = YMAX(K)
          ZMINL(K) = ZMIN(K)
          ZMAXL(K) = ZMAX(K)
       ENDDO
    ENDIF
    DO I = 1,NACT
       ACT = ACTV(I)
       XX(ACT) = XLO(ACT)
       YY(ACT) = YLO(ACT)
       ZZ(ACT) = ZLO(ACT)
    ENDDO
    DO K = 1,3
       XMIN(K) = XMINL(K)
       XMAX(K) = XMAXL(K)
       YMIN(K) = YMINL(K)
       YMAX(K) = YMAXL(K)
       ZMIN(K) = ZMINL(K)
       ZMAX(K) = ZMAXL(K)
    ENDDO
    WRITE(6,11) 'VOL AFTER ZX PROJ= ',LOWVOL
    !
    !  TEST YZ projection:
    IF ((YMAX(2) > 0).AND.(YMAX(3) > 0).AND.(ZMAX(3) > 0) &
         .AND.(ZMAX(2) > 0).AND.(YMIN(2) < 0).AND.(YMIN(3) < 0) &
         .AND.(ZMIN(2) < 0).AND.(ZMIN(3) < 0)) THEN
       AVAL = (ATAN2(ZMAX(3),YMAX(2)) + ATAN2(-ZMIN(3),-YMIN(2)))/2
       DO K = 1,3
          YMIN(K) = 99999
          YMAX(K) = -99999
          ZMIN(K) = 99999
          ZMAX(K) = -99999
       ENDDO
       IF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             YYY(ACT) = YY(ACT)*COS(AVAL) - ZZ(ACT)*SIN(AVAL)
             ZZZ(ACT) = YY(ACT)*SIN(AVAL) + ZZ(ACT)*COS(AVAL)
             YY(ACT) = YYY(ACT)
             ZZ(ACT) = ZZZ(ACT)
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          XMAXY = XMAX(2)*COS(AVAL) - XMAX(3)*SIN(AVAL)
          XMAXZ = XMAX(2)*SIN(AVAL) + XMAX(3)*COS(AVAL)
          XMINY = XMIN(2)*COS(AVAL) - XMIN(3)*SIN(AVAL)
          XMINZ = XMIN(2)*SIN(AVAL) + XMIN(3)*COS(AVAL)
          XMAX(2) = XMAXY
          XMAX(3) = XMAXZ
          XMIN(2) = XMINY
          XMIN(3) = XMINZ
          !
       ELSEIF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             YYY(ACT) = YY(ACT)*COS(AVAL) + ZZ(ACT)*SIN(AVAL)
             ZZZ(ACT) = -YY(ACT)*SIN(AVAL) + ZZ(ACT)*COS(AVAL)
             YY(ACT) = YYY(ACT)
             ZZ(ACT) = ZZZ(ACT)
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          XMAXY = XMAX(2)*COS(AVAL) + XMAX(3)*SIN(AVAL)
          XMAXZ = -XMAX(2)*SIN(AVAL) + XMAX(3)*COS(AVAL)
          XMINY = XMIN(2)*COS(AVAL) + XMIN(3)*SIN(AVAL)
          XMINZ = -XMIN(2)*SIN(AVAL) + XMIN(3)*COS(AVAL)
          XMAX(2) = XMAXY
          XMAX(3) = XMAXZ
          XMIN(2) = XMINY
          XMIN(3) = XMINZ
       ENDIF
    ELSEIF ((YMAX(2) > 0).AND.(YMAX(3) < 0).AND.(ZMIN(2) > 0) &
         .AND.(ZMIN(3) < 0).AND.(YMIN(2) < 0).AND.(YMIN(3) > 0) &
         .AND.(ZMAX(3) > 0).AND.(ZMAX(2) < 0)) THEN
       AVAL = (ATAN2(ZMAX(3),-YMIN(2)) + ATAN2(-ZMIN(3),YMAX(2))) &
            /2
       DO K = 1,3
          YMIN(K) = 99999
          YMAX(K) = -99999
          ZMIN(K) = 99999
          ZMAX(K) = -99999
       ENDDO
       IF (AVAL <= (0.7853981)) THEN
          DO I = 1,NACT
             ACT = ACTV(I)
             YYY(ACT) = YY(ACT)*COS(AVAL) - ZZ(ACT)*SIN(AVAL)
             ZZZ(ACT) = YY(ACT)*SIN(AVAL) + ZZ(ACT)*COS(AVAL)
             YY(ACT) = YYY(ACT)
             ZZ(ACT) = ZZZ(ACT)
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          XMAXY = XMAX(2)*COS(AVAL) - XMAX(3)*SIN(AVAL)
          XMAXZ = XMAX(2)*SIN(AVAL) + XMAX(3)*COS(AVAL)
          XMINY = XMIN(2)*COS(AVAL) - XMIN(3)*SIN(AVAL)
          XMINZ = XMIN(2)*SIN(AVAL) + XMIN(3)*COS(AVAL)
          XMAX(2) = XMAXY
          XMAX(3) = XMAXZ
          XMIN(2) = XMINY
          XMIN(3) = XMINZ
       ELSEIF (AVAL > (0.7853981)) THEN
          AVAL = 1.5707963 - AVAL
          DO I = 1,NACT
             ACT = ACTV(I)
             YYY(ACT) = YY(ACT)*COS(AVAL) + ZZ(ACT)*SIN(AVAL)
             ZZZ(ACT) = -YY(ACT)*SIN(AVAL) + ZZ(ACT)*COS(AVAL)
             YY(ACT) = YYY(ACT)
             ZZ(ACT) = ZZZ(ACT)
             IF (YY(ACT) > YMAX(2)) THEN
                YMAX(1) = XX(ACT)
                YMAX(2) = YY(ACT)
                YMAX(3) = ZZ(ACT)
             ENDIF
             IF (YY(ACT) < YMIN(2)) THEN
                YMIN(1) = XX(ACT)
                YMIN(2) = YY(ACT)
                YMIN(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) > ZMAX(3)) THEN
                ZMAX(1) = XX(ACT)
                ZMAX(2) = YY(ACT)
                ZMAX(3) = ZZ(ACT)
             ENDIF
             IF (ZZ(ACT) < ZMIN(3)) THEN
                ZMIN(1) = XX(ACT)
                ZMIN(2) = YY(ACT)
                ZMIN(3) = ZZ(ACT)
             ENDIF
          ENDDO
          XMAXY = XMAX(2)*COS(AVAL) + XMAX(3)*SIN(AVAL)
          XMAXZ = -XMAX(2)*SIN(AVAL) + XMAX(3)*COS(AVAL)
          XMINY = XMIN(2)*COS(AVAL) + XMIN(3)*SIN(AVAL)
          XMINZ = -XMIN(2)*SIN(AVAL) + XMIN(3)*COS(AVAL)
          XMAX(2) = XMAXY
          XMAX(3) = XMAXZ
          XMIN(2) = XMINY
          XMIN(3) = XMINZ
       ENDIF
    ENDIF
    XR = XMAX(1) - XMIN(1)
    YR = YMAX(2) - YMIN(2)
    ZR = ZMAX(3) - ZMIN(3)
    VOLPTS = XR*YR*ZR
    IF (VOLPTS < LOWVOL) THEN
       DO I = 1,NACT
          ACT = ACTV(I)
          XLO(ACT) = XX(ACT)
          YLO(ACT) = YY(ACT)
          ZLO(ACT) = ZZ(ACT)
       ENDDO
       DO K = 1,3
          XMINL(K) = XMIN(K)
          XMAXL(K) = XMAX(K)
          YMINL(K) = YMIN(K)
          YMAXL(K) = YMAX(K)
          ZMINL(K) = ZMIN(K)
          ZMAXL(K) = ZMAX(K)
       ENDDO
    ENDIF
    DO I = 1,NACT
       ACT = ACTV(I)
       XX(ACT) = XLO(ACT) + XAVG
       YY(ACT) = YLO(ACT) + YAVG
       ZZ(ACT) = ZLO(ACT) + ZAVG
    ENDDO
    DO K = 1,3
       XMIN(K) = XMINL(K) + XAVG
       XMAX(K) = XMAXL(K) + XAVG
       YMIN(K) = YMINL(K) + YAVG
       YMAX(K) = YMAXL(K) + YAVG
       ZMIN(K) = ZMINL(K) + ZAVG
       ZMAX(K) = ZMAXL(K) + ZAVG
    ENDDO
    WRITE(6,11) 'VOL AFTER YZ PROJ= ',LOWVOL
    !
    call chmdealloc('corman3.src','squareup','space_rl',6*natomx,crl=space_rl)
    nullify(xxx,yyy,zzz,xlo,ylo,zlo)
    RETURN
  END SUBROUTINE SQUAREUP

#if KEY_CHEQ==1 /*cheq_main*/
  SUBROUTINE HISTOGRAM(VECT,COMLYN,COMLEN)
  use dimens_fcm
  use number
  use exfunc
  use bases_fcm
  use inbnd
  use psf
  use stream
  use string
  use memory

    integer,allocatable,dimension(:),save :: IPOINT
    character(len=*) COMLYN
    INTEGER COMLEN
    real(chm_real) VECT(*)
    !
    INTEGER,save :: IOPT=0,HLEN,HNUM
    integer :: i
    real(chm_real) WEIG
    real(chm_real),save :: HMIN,HMAX

    IF (INDXA(COMLYN,COMLEN,'SETU') > 0) THEN
       IOPT=1
       HMIN=GTRMF(COMLYN,COMLEN,'HMIN',ZERO)
       HMAX=GTRMF(COMLYN,COMLEN,'HMAX',ZERO)
       HNUM=GTRMI(COMLYN,COMLEN,'HNUM',0)
       IF(HMAX <= HMIN .OR. HNUM <= 0) THEN
          CALL WRNDIE(0,'<HIST>', &
               'Error in parsing histogram keywords.')
       ENDIF
       IF(HLEN == 0) THEN
          call chmalloc('corman3.src','HISTOGRAM','IPOINT',HNUM,intg=IPOINT)
          ipoint=0
          HLEN=HNUM
       ENDIF
       RETURN
    ENDIF
    IF (HLEN <= 0) CALL WRNDIE(0,'<HIST>','HIST vector not setup.')
    IF (INDXA(COMLYN,COMLEN,'FINI') > 0) THEN
       WEIG=GTRMI(COMLYN,COMLEN,'WEIG',1)  ! SAPATEL HACK
       !         WEIG=GTRMF(COMLYN,COMLEN,'WEIG',1.0)
       IF (WEIG == 0.0)THEN
          WEIG=1.0
       ELSE
          WEIG=1.0/WEIG
       ENDIF
       CALL PRINTHIST(iPOINT,HMIN,HMAX,HNUM,WEIG)
       call chmdealloc('corman3.src','HISTOGRAM','iPOINT',HNUM,intg=iPOINT)
       IOPT=0
       RETURN
    ENDIF
    CALL HISTCUMU(VECT,NATOM,iPOINT,HMIN,HMAX,HNUM)
    RETURN
  END SUBROUTINE HISTOGRAM

  SUBROUTINE HISTCUMU(VECT,NATOM,HIS,HMIN,HMAX,HNUM)

    real(chm_real) VECT(*),HMIN,HMAX,HI
    INTEGER HIS(*),HNUM,KI,I,NATOM

    DO I=1,NATOM
       HI=(VECT(I)-HMIN)/(HMAX-HMIN)
       !         write(6,*) 'HISTCUMU: hi',hi
       IF ((HI >= 0.0) .AND. (HI < 1.0)) THEN
          KI=HI*HNUM+1
          !            write(6,*) 'HISTCUMU: KI',KI
          HIS(KI)=HIS(KI)+1
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE HISTCUMU
  SUBROUTINE PRINTHIST(HIS,HMIN,HMAX,HNUM,WEIG)
  use stream

    real(chm_real) HMIN,HMAX,HI,DH,WEIG,HIST
    INTEGER HIS(*),HNUM,KI,I
    DH=(HMAX-HMIN)/HNUM
    DO I=1,HNUM
       HI=HMIN+(I-1)*DH
       HIST=HIS(I)*WEIG
       WRITE(6,*) HI, HIST
       WRITE(OUTU,'(2f20.5)')HI,HIST
    ENDDO
    RETURN
  END SUBROUTINE PRINTHIST
  !
  SUBROUTINE HISTOGRAM2(ISLCT,COMLYN,COMLEN)
#if KEY_CHEQ==1
  use cheq,only:qcg,   &
     DCH,SUMDCH,DDCH
#endif
  use dimens_fcm
  use number
  use exfunc
     !
     !
  use bases_fcm
  use inbnd
  use psf
  use stream
  use string
  use coord
  use consta
  use memory

    integer,allocatable,dimension(:),save :: HPOINT
    INTEGER I,J,K,ISLCT(*)
    character(len=*) COMLYN
    INTEGER COMLEN
    INTEGER,save :: IOPT=0,HLEN,HNUM
    integer :: KI
    real(chm_real),save :: HMIN,HMAX,MYHIST(10000)
    real(chm_real) WEIG,HI,DH,HIST

    IF (INDXA(COMLYN,COMLEN,'STUP') > 0) THEN
       HMIN=GTRMF(COMLYN,COMLEN,'HMIN',ZERO)
       HMAX=GTRMF(COMLYN,COMLEN,'HMAX',ZERO)
       HNUM=GTRMI(COMLYN,COMLEN,'HNUM',0)
       IF(HLEN == 0) THEN
          IOPT=1
          call chmalloc('corman3.src','HISTOGRAM2','HPOINT',HNUM,intg=HPOINT)
          hpoint=0
          HLEN=HNUM
       ENDIF
    ENDIF

    IF (INDXA(COMLYN,COMLEN,'FINI') > 0) THEN
       write(6,*)"DONE"
       WEIG=GTRMI(COMLYN,COMLEN,'WEIG',1)  ! SAPATEL HACK
       !         WEIG=GTRMF(COMLYN,COMLEN,'WEIG',1.0)
       write(6,*) " WEIGHT = ", WEIG
       IF (WEIG == 0.0)THEN
          WEIG=1.0
       ELSE
          WEIG=1.0/WEIG
       ENDIF
       DH=(HMAX-HMIN)/HNUM
       DO I=1,HNUM
          HI=HMIN+(I-1)*DH
          HIST=MYHIST(I)*WEIG
          WRITE(6,*) HI, HIST
       ENDDO

       call chmdealloc('corman3.src','HISTOGRAM2','HPOINT',HNUM,intg=HPOINT)
       IOPT=0
       RETURN
    ENDIF

    DO I = 1,NATOM
       IF (ISLCT(I) /= 0) THEN
          HI = (CG(I)-HMIN)/(HMAX-HMIN)
          IF ((HI >= 0.0) .AND. (HI < 1.0)) THEN
             KI=HI*HNUM+1
             MYHIST(KI) = MYHIST(KI) + 1
          ENDIF
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE HISTOGRAM2
#endif /* (cheq_main)*/


end module corman3
