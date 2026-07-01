module rxcons4
  use chm_kinds
  use dimens_fcm
  implicit none

contains

#if KEY_RXNCONS==1 /*rxcons_4*/
  SUBROUTINE RX_4_SETUP(COMLYN,COMLEN)
    use number
    use coord
    use memory
    use shake
    use psf
    use replica_mod
    use parallel
    use repdstr
    use stream
    use rxncons
    use rxnconswt
    use rxncons4
#if KEY_REPLICA==1
    use EPATHMOD                            
#endif
    !
    real(chm_real),allocatable,dimension(:,:) :: XF,YF,ZF
    real(chm_real),allocatable,dimension(:,:) :: XTAN,YTAN,ZTAN
    real(chm_real),allocatable,dimension(:,:) :: XM1,YM1,ZM1
    real(chm_real),allocatable,dimension(:,:) :: XP1,YP1,ZP1
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA,RB,ARMS
    real(chm_real),allocatable,dimension(:) :: FRMS,DERR,D_TAN
    real(chm_real),allocatable,dimension(:) :: DTLV,DTLVC,DVDRL
    real(chm_real),allocatable,dimension(:) :: DVDR,DVDRR
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN

    ! . Local variables.
    INTEGER ISLCT, JSLCT
    LOGICAL ERR,QPRINT
    real(chm_real) WTOT

    INTEGER NATREP,NREP
    INTEGER NALLOC

#if KEY_REPLICA==1 || KEY_REPDSTR==1
    INTEGER,ALLOCATABLE,DIMENSION(:) :: ISTREP,ICNREP
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: WEIGHT
    logical :: qret

    IF((.NOT.LCLEAN).AND.(.NOT.QPFORCE)) THEN
       qret = .not.qrep
#if KEY_REPDSTR==1
       qret = .not.qrepdstr                       
#endif
#if KEY_REPDSTR==1 && KEY_REPLICA==1
       qret = (.not.qrepdstr) .and. (.not.qrep)   
#endif
       IF(QRET) RETURN

#if KEY_REPDSTR==1
       if (qrepdstr) then
          ! since we delete the originals in the standard replica code
          ! we need +1 below:
          nrepl = nrepdstr + 1
          ! we need icnrep !?? Do we??? where do we get it ???
          !           write(outu,*)'RX_4_SETUP>nrepl=', nrepl
       endif
#endif 

       QPRINT=((PRNLEV.GE.6).OR.LRXPRN)
       !
       ! rx_4_setup chmalloc
       call chmalloc('rxcons_4.src','RX_4_SETUP','ISTREP',NREPL,intg=istrep)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ICNREP',NREPL,intg=icnrep)
       call chmalloc('rxcons_4.src','RX_4_SETUP','WEIGHT',Natom,crl=weight)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ARMS',Nrepl,crl=arms)
       call chmalloc('rxcons_4.src','RX_4_SETUP','FRMS',Nrepl,crl=frms)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DERR',Nrepl,crl=derr)
       !
       frms(1:nrepl) = zero
       derr(1:nrepl) = zero
       !
       ! ISTREP: the starting atom of each replica
       !
       CALL EPATHC(ISTREP,ICNREP,WEIGHT, &
            NATREP,WTOT,QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
            QCWCOMP2,  & 
#endif
            QCMASS,QPRINT,ERR)

#if KEY_REPDSTR==1
       !       again after epathc add +1
       if(qrepdstr)nrepl=nrepdstr+1
       !        write(outu,*)'RX_4_SETUP>icnrep1=', icnrep(1)
       !        write(outu,*)'RX_4_SETUP-2>nrepl=', nrepl
#endif 
       !
       CALL RX_4_SETUP2(ISTREP,ICNREP,WEIGHT, &
            NATREP,WTOT,NREPL,QPTCSCL,NRXNCS,NRXATM, &
            QFXREP,IFXREP)

       NALLOC=NRXATM*(NREPL-1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','IRXATM',NALLOC,intg=IRXATM)
       call chmalloc('rxcons_4.src','RX_4_SETUP','XCORD',NALLOC,crl=XCORD)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YCORD',NALLOC,crl=YCORD)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZCORD',NALLOC,crl=ZCORD)
       call chmalloc('rxcons_4.src','RX_4_SETUP','XREFC',NALLOC,crl=XREFC)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YREFC',NALLOC,crl=YREFC)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZREFC',NALLOC,crl=ZREFC)

       call chmalloc('rxcons_4.src','RX_4_SETUP','WMASS',NRXATM,crl=WMASS)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ATMASS',NRXATM,crl=ATMASS)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DLTAN',NREPL,crl=DLTAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DLCRV',NREPL,crl=DLCRV)
       call chmalloc('rxcons_4.src','RX_4_SETUP','FDOTTAN',NREPL,crl=FDOTTAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','FOFFP',NREPL,crl=FOFFP)

       PDLENG=ZERO

       CALL RX_4_SETUP3(NRXATM,NRXNCS,NATREP,NREPL, &
            IRXATM,WMASS,ATMASS,WTOT, &
            ISTREP,ICNREP,WEIGHT, &
            QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
            QCWCOMP2,  & 
#endif
            QCMASS,LPRID)

       NALLOC=NRXATM*(NREPL-1)

       call chmalloc('rxcons_4.src','RX_4_SETUP','XF',NREPL-1,NRXATM,crl=XF)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YF',NREPL-1,NRXATM,crl=YF)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZF',NREPL-1,NRXATM,crl=ZF)

       call chmalloc('rxcons_4.src','RX_4_SETUP','XTAN',NREPL-1,NRXATM,crl=XTAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YTAN',NREPL-1,NRXATM,crl=YTAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZTAN',NREPL-1,NRXATM,crl=ZTAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','XM1',NREPL-1,NRXATM,crl=XM1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YM1',NREPL-1,NRXATM,crl=YM1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZM1',NREPL-1,NRXATM,crl=ZM1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','XP1',NREPL-1,NRXATM,crl=XP1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','YP1',NREPL-1,NRXATM,crl=YP1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ZP1',NREPL-1,NRXATM,crl=ZP1)
       call chmalloc('rxcons_4.src','RX_4_SETUP','ATMPR',2*NRXATM,intg=ATMPR)
       call chmalloc('rxcons_4.src','RX_4_SETUP','RA',3*NRXATM,crl=RA)
       call chmalloc('rxcons_4.src','RX_4_SETUP','RB',3*NRXATM,crl=RB)

       call chmalloc('rxcons_4.src','RX_4_SETUP','D_TAN',NREPL,crl=D_TAN)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DTLV',NREPL,crl=DTLV)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DTLVC',NREPL,crl=DTLVC)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DVDRL',NREPL,crl=DVDRL)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DVDR',NREPL,crl=DVDR)
       call chmalloc('rxcons_4.src','RX_4_SETUP','DVDRR',NREPL,crl=DVDRR)

       CALL RX_4_SETUP4(NRXATM,NRXNCS,NATREP,NREPL,X,Y,Z, &
            XCORD,YCORD,ZCORD, &
            XM1,YM1,ZM1, &
            XP1,YP1,ZP1, &
            IRXATM,WMASS,ATMASS, &
            WTOT,QPTCSCL,QCNOTRN,QCNOROT,QPRINT,MAXITR, &
            ARMS,FRMS,DERR, &
            D_TAN,DTLV,DTLVC, &
            DVDRL,DVDR,DVDRR, &
            ATMPR,RA,RB,LPRID, &
            QPTAN,IUTAN)

       NALLOC=NRXATM*(NREPL-1)

       !
       ! rx_4_setup chmdealloc
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XF',NREPL-1,NRXATM,crl=XF)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YF',NREPL-1,NRXATM,crl=YF)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZF',NREPL-1,NRXATM,crl=ZF)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XTAN',NREPL-1,NRXATM,crl=XTAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YTAN',NREPL-1,NRXATM,crl=YTAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZTAN',NREPL-1,NRXATM,crl=ZTAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XM1',NREPL-1,NRXATM,crl=XM1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YM1',NREPL-1,NRXATM,crl=YM1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZM1',NREPL-1,NRXATM,crl=ZM1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XP1',NREPL-1,NRXATM,crl=XP1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YP1',NREPL-1,NRXATM,crl=YP1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZP1',NREPL-1,NRXATM,crl=ZP1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ATMPR',2*NRXATM,intg=ATMPR)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','RA',3*NRXATM,crl=RA)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','RB',3*NRXATM,crl=RB)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','D_TAN',NREPL,crl=D_TAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DTLV',NREPL,crl=DTLV)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DTLVC',NREPL,crl=DTLVC)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DVDRL',NREPL,crl=DVDRL)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DVDR',NREPL,crl=DVDR)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DVDRR',NREPL,crl=DVDRR)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ISTREP',NREPL,intg=istrep)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ICNREP',NREPL,intg=icnrep)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','WEIGHT',Natom,crl=weight)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DERR',Nrepl,crl=derr)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','FRMS',Nrepl,crl=frms)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ARMS',Nrepl,crl=arms)
       !
       LRXCNS=.TRUE.
       QHOLO =.TRUE.
    ELSEIF(QPFORCE)THEN

       IF(.NOT.LRXCNS) CALL WRNDIE(-1,'<RX_4_SETUP>', &
            'path not set up yet')

       CALL PPFORCE(NREPL,PDLENG,FDOTTAN,FOFFP,DLTAN,DLCRV)

    ELSE
       NALLOC=NRXATM*(NREPL-1)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','IRXATM',NALLOC,intg=IRXATM)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XCORD',NALLOC,crl=XCORD)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YCORD',NALLOC,crl=YCORD)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZCORD',NALLOC,crl=ZCORD)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','XREFC',NALLOC,crl=XREFC)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','YREFC',NALLOC,crl=YREFC)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ZREFC',NALLOC,crl=ZREFC)

       call chmdealloc('rxcons_4.src','RX_4_SETUP','WMASS',NRXATM,crl=WMASS)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','ATMASS',NRXATM,crl=ATMASS)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DLTAN',NREPL,crl=DLTAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','DLCRV',NREPL,crl=DLCRV)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','FDOTTAN',NREPL,crl=FDOTTAN)
       call chmdealloc('rxcons_4.src','RX_4_SETUP','FOFFP',NREPL,crl=FOFFP)
    ENDIF
#endif 
    RETURN
  END SUBROUTINE RX_4_SETUP

  SUBROUTINE RX_4_SETUP2(ISTREP,ICNREP,WEIGHT,NATREP,WTOT, &
       NREPL,QCYCL,NRXNCS,NRXATM, &
       QFXREP,IFXREP)
    use number
    use stream
#if KEY_REPDSTR==1
    use repdstr,only: irepdstr,nrepdstr,qrepdstr   
#endif

    INTEGER NREPL
    INTEGER I,IA

    INTEGER ISTREP(*),ICNREP(*),NATREP,NRXNCS,NRXATM
    real(chm_real) WEIGHT(*),WTOT
    INTEGER IFXREP

    LOGICAL QCYCL,QFXREP

    if(prnlev.gt.2)WRITE(OUTU,100)'number of replica: ',NREPL-1

    NATREP=ICNREP(1)

    NRXATM=NATREP

    NRXNCS=NREPL-3
    IF(QCYCL)NRXNCS=NREPL-1

    ! We need to subtract one D.O.F. for REPDSTR
    ! Two at the end points
#if KEY_REPDSTR==1
    if(qrepdstr)then
       nrxncs=1
       !for non-cyclic increase the endpoints by 1
       if((.not.qcycl).and.(irepdstr == 0)) nrxncs=2
       if((.not.qcycl).and.(irepdstr == (nrepdstr-1))) nrxncs=2
    endif
#endif 

    if(prnlev.gt.2)WRITE(OUTU,100)'number of constraints: ', NRXNCS

    IF(QFXREP)THEN
       if(prnlev.gt.2)WRITE(OUTU,*)'replica ',IFXREP,' will be fixed'
    ENDIF

100 FORMAT('RX_4_SETUP2> ',A,I5)
200 FORMAT('RX_4_SETUP2> ',A,F16.6)
    RETURN
  END SUBROUTINE RX_4_SETUP2

  SUBROUTINE RX_4_SETUP3(NRXATM,NRXNCS,NATREP,NREPL, &
       IRXATM,WMASS,ATMASS,WTOT, &
       ISTREP,ICNREP,WEIGHT, &
       QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
       QCWCOMP2,  & 
#endif
       QCMASS,LPRID)
    use number
    use consta
    use psf
    use coord
    use coordc
    use stream
    use lonepr

    INTEGER NRXATM,NRXNCS,NATREP,IRXATM(*),NREPL
    real(chm_real) WMASS(*),ATMASS(*)
    INTEGER ISTREP(*),ICNREP(*),LPRID(*)
    real(chm_real) WEIGHT(*),WTOT,FACT,SUM

    INTEGER I,IA,NT,ILP,N,IPT,J,K

    LOGICAL QCWEIG,QCWCOMP,QCMASS

#if KEY_COMP2==1
    LOGICAL QCWCOMP2
#endif 
    real(chm_real) TEMP

    NT=0
    WTOT=ZERO

    IA=ISTREP(1)

    !      write(outu,*)'RX_4_SETUP3>ia/istrep1=',ia

    TEMP=ONE/DBLE(NATREP)

    DO I=1,NATOM
       LPRID(I)=0
    ENDDO

    DO I=1,NATREP
       IA=IA+1
       IRXATM(I)=IA
       WMASS(I)=ONE
       IF(QCWEIG.AND.(.NOT.QCMASS))THEN
          IF(QCWCOMP)THEN
             WMASS(I)=WCOMP(IA)
#if KEY_COMP2==1
          ELSEIF(QCWCOMP2)THEN
             WMASS(I)=WCOMP2(IA)
#endif 
          ELSE
             WMASS(I)=WMAIN(IA)
          ENDIF
       ELSEIF(QCMASS.AND.(.NOT.QCWEIG))THEN
          WMASS(I)=AMASS(IA)
       ELSEIF(QCMASS.AND.QCWEIG)THEN
          IF(QCWCOMP)THEN
             WMASS(I)=AMASS(IA)*WCOMP(IA)
#if KEY_COMP2==1
          ELSEIF(QCWCOMP2)THEN
             WMASS(I)=AMASS(IA)*WCOMP2(IA)
#endif 
          ELSE
             WMASS(I)=AMASS(IA)*WMAIN(IA)
          ENDIF
       ENDIF

       WTOT=WTOT+WMASS(I)

       ! for a lonepair site, use the inverse of wmass instead,
       ! since its amass has been set to zero
       IF(IMOVE(IA).NE.-1)THEN
          ATMASS(I)=ONE/AMASS(IA)
       ELSE
          ATMASS(I)=ONE/WMASS(I)
       ENDIF

       !        IF(WEIGHT(IA).GT.ZERO)THEN
       !          NT=NT+1
       !          WMASS(NT)=WEIGHT(IA)
       !          WTOT=WTOT+WMASS(NT)
       !          ATMASS(NT)=ONE/AMASS(IA)
       !        ENDIF
    ENDDO

    if(prnlev.gt.2)WRITE(OUTU,200)'total weight     : ',WTOT

    IF(NUMLP.GT.0)THEN
       DO ILP=NUMLP,1,-1
          N=LPNHOST(ILP)
          IF(N.LE.3)CALL WRNDIE(-1,'<RX_4_SETUP3>', &
               '  PCONS does not allow none cent type of lonepairs')

          IPT=LPHPTR(ILP)
          I=LPHOST(IPT)

          LPRID(I)=ILP
       ENDDO

       IA=ISTREP(1)
       DO I=1,NATREP
          IA=IA+1
          IF(IMOVE(IA).EQ.-1)THEN
             ILP=LPRID(IA)
             N=LPNHOST(ILP)
             IPT=LPHPTR(ILP)
             IF(WMASS(IA).GT.ZERO)THEN
                DO K=1,N
                   J=LPHOST(IPT+K)
                   IF(WMASS(J).GT.ZERO) CALL WRNDIE(-1,'<RX_4_SETUP3>', &
                        'lonepair and hosts cannot both have non-zero weight')
                ENDDO
             ENDIF
          ENDIF
       ENDDO
    ENDIF

    !      DO I=1,NRXATM
    !        WRITE(OUTU,100)IRXATM(I),WMASS(I),ATMASS(I)
    !      ENDDO

100 FORMAT('RX_4_SETUP3> ',I5,2F16.6)
200 FORMAT('RX_4_SETUP3> ',A,F16.6)
300 FORMAT('RX_4_SETUP3> ',A,A)

    RETURN
  END SUBROUTINE RX_4_SETUP3


  SUBROUTINE RX_4_SETUP4(NRXATM,NRXNCS,NATREP,NREPL,X,Y,Z, &
       XCORD,YCORD,ZCORD, &
       XM1,YM1,ZM1, &
       XP1,YP1,ZP1, &
       IRXATM,WMASS,ATMASS, &
       WTOT,QCYCL,QCNOTRN,QCNOROT,LPRINT,MXITER, &
       ARMS,FRMS,DERR,D_TAN,DTLV,DTLVC, &
       DVDRL,DVDR,DVDRR, &
       ATMPR,RA,RB,LPRID,QPTAN,IUTAN)
    use number
    use stream
    use parallel
    use repdstr
    use psf  ! for writing NATOM
    use lonepr
    use epathmod

    INTEGER NRXATM,NRXNCS,NATREP,NREPL,MXITER
    LOGICAL QCYCL,QCNOTRN,QCNOROT,LPRINT,LRXPRN,QPTAN

    real(chm_real) X(*),Y(*),Z(*),XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XM1(NREPL-1,*),YM1(NREPL-1,*),ZM1(NREPL-1,*)
    real(chm_real) XP1(NREPL-1,*),YP1(NREPL-1,*),ZP1(NREPL-1,*)
    INTEGER IRXATM(*)
    real(chm_real) WMASS(*),ATMASS(*),WTOT
    real(chm_real) D_TAN(*),DTLV(*),DTLVC(*)
    real(chm_real) DVDRL(*),DVDR(*),DVDRR(*)
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM)
    real(chm_real) ARMS(*),FRMS(*),DERR(*),DT

    INTEGER NREP,NPSG,IREP,JREP,KREP,IPT,JPT
    INTEGER I,J,IA,NALL,ITER,IBEGIN,ISTOP
    INTEGER LPRID(*),K,N,ILP,IUTAN

    real(chm_real) RMST,RMST0,ATMP
    real(chm_real) DEVA(3,3),EVWID
    real(chm_real) FACT,SUM,SUM1,TEMP,AERR,DERRO
    real(chm_real) XD,YD,ZD
    real(chm_real) TANP1,TANP2,RIJ,WL,WR

    NREP=NREPL-1
    NPSG=NREP-1
    IF(QCYCL)NPSG=NREP

    NALL=NREP*NRXATM

    TEMP=ONE/DBLE(NPSG)

    ! copy ccordinates

    !      write(mynodg+30,*) &
    !           'rx_4_setup4>before psnd():irepdstr,mynod,mynodg,numnod,numnodg=', &
    !           irepdstr,mynod,mynodg,numnod,numnodg

#if KEY_REPDSTR==1
    if(.not.qrepdstr)then         
#endif
       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NATREP*(JREP-1)+IRXATM(J)
             I=I+1
             XCORD(I)=X(IA)
             YCORD(I)=Y(IA)
             ZCORD(I)=Z(IA)
          ENDDO
       ENDDO
#if KEY_REPDSTR==1 /*set4a*/
    else
       !
       !        this will change once we have selection on
       !        repd command
       xcord(1:nall)=zero
       ycord(1:nall)=zero
       zcord(1:nall)=zero
       !!         write(outu,*)'rx_4_setup4>natrep,nrxatm=',natrep,nrxatm
       ia=natrep*irepdstr
       do j = 1, nrxatm
          xcord(ia+j)=x(j)
          ycord(ia+j)=y(j)
          zcord(ia+j)=z(j)
       enddo
       !
       ! there are more than one way to deal with this setup:
       ! 1. broadcast my coordinates to everyone and use
       !    existing code (no loop modification needed)
       ! 2. do a similar loop modifications that we did
       !    for a restraint replica path code and then
       !    communicate as needed
       ! 3....
       !
       ! we choose 1. here: need to broadcast to everyone
       !
       !     prepare communication arrays: arepdmap,prepdmap
       allocate(arepdmap(0:nrepdstr),prepdmap(0:nrepdstr))

       do i = 0, nrepdstr
          arepdmap(i)=i*nrxatm
          prepdmap(i)=i*numnod
       enddo
       !         write(mynodg+30,'(a,10(1x,i0))') &
       !              'rx_4_setup4>before if:irepdstr,mynod,mynodg,numnod,numnodg=', &
       !              irepdstr,mynod,mynodg,numnod,numnodg
       if ( mynodg .eq. (numnod*irepdstr)) then
          !           this has to be protected! Not everyone can call this!!!
          !
          call repdbr(xcord)
          call repdbr(ycord)
          call repdbr(zcord)
       endif
       !
       !        We also need to psnd the coordinates to local processes
       !        (everybody does the same?? - do we really need this ???
       !        it would be more efficient to skip this code on processes
       !        other then local zeros and psnd only local coordinates at the end!
       call psnd8(xcord,nall)
       call psnd8(ycord,nall)
       call psnd8(zcord,nall)
    endif

    !      write(outu,*)'RX_4_SETUP4>after broadcast'
    !      do i=1,nall
    !         write(outu,'(i5,3f15.5)')i,xcord(i),ycord(i),zcord(i)
    !      enddo


#endif /* (set4a)*/
    !
    ! define the range of active replicas
    !
    IF(QCYCL)THEN
       IBEGIN=2
       ISTOP=NREP
    ELSE
       IBEGIN=2
       ISTOP=NREP-1
    ENDIF

    FRMS (1:NREPL)=zero
    ARMS (1:NREPL)=zero
    DERR (1:NREPL)=zero
    DTLV (1:NREPL)=zero
    DTLVC(1:NREPL)=zero
    DVDRL(1:NREPL)=zero
    DVDR (1:NREPL)=zero
    DVDRR(1:NREPL)=zero

    ! initial call

    CALL GETVECTS(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XM1,YM1,ZM1,XP1,YP1,ZP1,ARMS,RMST, &
         QCNOTRN,QCNOROT,QCYCL)

    RMST0=RMST

    SUM=ZERO
    DO I=1,NPSG
       DERR(I)=-(ARMS(I)-RMST0)*HALF
       SUM=SUM+DERR(I)*DERR(I)
    ENDDO
    AERR=SQRT(SUM*TEMP)
    !
    !      write(outu,*)'RX_4_SETUP4>aerr=',aerr
    !

    ! iteration

    ITER=0
1111 CONTINUE

    CALL GETDTLVS(NPSG,NREP,NRXATM, &
         XM1,YM1,ZM1,XP1,YP1,ZP1, &
         DTLV,DTLVC,DVDRL,DVDR,DVDRR)

    CALL TRIDIAG(DVDRL,DVDR,DVDRR,DERR,FRMS,NPSG)

    !      DO I=1,NPSG
    !        WRITE(OUTU,'(I5,7F16.6)')I,ARMS(I),RMST0,
    !     $                     DVDRL(I),DVDR(I),DVDRR(I),-DERR(I),FRMS(I)
    !      ENDDO

    DO JREP=IBEGIN,ISTOP
       IREP=JREP-1

       WL=FRMS(IREP)
       WR=FRMS(JREP)

       IA=(JREP-1)*NRXATM
       DO I=1,NRXATM
          IA=IA+1
          XCORD(IA)=XCORD(IA)+ &
               XM1(JREP,I)*WL-XP1(JREP,I)*WR
          YCORD(IA)=YCORD(IA)+ &
               YM1(JREP,I)*WL-YP1(JREP,I)*WR
          ZCORD(IA)=ZCORD(IA)+ &
               ZM1(JREP,I)*WL-ZP1(JREP,I)*WR
       ENDDO
    ENDDO

    CALL GETVECTS(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XM1,YM1,ZM1,XP1,YP1,ZP1,ARMS,RMST, &
         QCNOTRN,QCNOROT,QCYCL)

    SUM=ZERO
    DO I=1,NPSG
       DERR(I)=-(ARMS(I)-RMST0)*HALF
       !        WRITE(OUTU,200)I,ARMS(I),DERR(I),FRMS(I)
       SUM=SUM+DERR(I)*DERR(I)
    ENDDO
    AERR=SQRT(SUM*TEMP)

    ITER=ITER+1
    IF(LPRINT)WRITE(OUTU,'(A,I5,E16.6)')'ITER ',ITER,AERR

    IF(AERR.GT.TENM8)THEN
       IF(ITER.LE.MXITER)THEN
          GOTO 1111
       ELSE
          WRITE(OUTU,'(A,I5,E16.6)')'RX_4_SETUP4> ',ITER,AERR
          CALL WRNDIE(-1,'<RX_4_SETUP4>', &
               'Path distance constraints do not converge')
       ENDIF
    ENDIF

    !      WRITE(OUTU,'(A,I5,E16.6)')'CONVERGED',ITER,AERR

    IF(QPTAN)THEN
       IF(QCYCL)THEN
          IBEGIN=1
          ISTOP=NREP
       ELSE
          IBEGIN=2
          ISTOP=NREP-1
       ENDIF

       DO JREP=IBEGIN,ISTOP
          SUM=ZERO
          DO I=1,NRXATM
             XM1(JREP,I)=XM1(JREP,I)+XP1(JREP,I)
             YM1(JREP,I)=YM1(JREP,I)+YP1(JREP,I)
             ZM1(JREP,I)=ZM1(JREP,I)+ZP1(JREP,I)
             SUM=SUM+XM1(JREP,I)*XM1(JREP,I)+ &
                  YM1(JREP,I)*YM1(JREP,I)+ &
                  ZM1(JREP,I)*ZM1(JREP,I)
          ENDDO

          TEMP=ONE/SQRT(SUM)

          DO I=1,NRXATM
             XM1(JREP,I)=XM1(JREP,I)*TEMP
             YM1(JREP,I)=YM1(JREP,I)*TEMP
             ZM1(JREP,I)=ZM1(JREP,I)*TEMP
          ENDDO

          TANP1=ZERO
          DO I=1,NRXATM
             SUM=XM1(JREP,I)*XM1(JREP,I)+ &
                  YM1(JREP,I)*YM1(JREP,I)+ &
                  ZM1(JREP,I)*ZM1(JREP,I)

             XP1(JREP,I)=SUM

             WRITE(IUTAN,33)JREP,I,XM1(JREP,I),YM1(JREP,I),ZM1(JREP,I), &
                  WMASS(I),SUM
             IF(SUM.GT.RSMALL)TANP1=TANP1-SUM*DLOG(SUM)
          ENDDO

          TANP2=ZERO
          SUM=ZERO
          IA=(JREP-1)*NRXATM
          DO I=1,NRXATM-1
             DO J=I+1,NRXATM
                RIJ=(XCORD(IA+I)-XCORD(IA+J))*(XCORD(IA+I)-XCORD(IA+J))+ &
                     (YCORD(IA+I)-YCORD(IA+J))*(YCORD(IA+I)-YCORD(IA+J))+ &
                     (ZCORD(IA+I)-ZCORD(IA+J))*(ZCORD(IA+I)-ZCORD(IA+J))
                RIJ=SQRT(RIJ)
                SUM1=XP1(JREP,I)*XP1(JREP,J)+ &
                     YP1(JREP,I)*YP1(JREP,J)+ &
                     ZP1(JREP,I)*ZP1(JREP,J)
                TANP2=TANP2+SUM1*RIJ
                SUM=SUM+SUM1
             ENDDO
          ENDDO
          TANP2=TANP2/SUM
          WRITE(IUTAN,34)"Tangent",JREP,TANP1,TANP2
       ENDDO

33     FORMAT(2I5,4F12.4,F16.6)
34     FORMAT(A,I5,2F16.6)
    ENDIF

#if KEY_REPDSTR==1
    if(.not.qrepdstr)then         
#endif

       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NATREP*(JREP-1)+IRXATM(J)
             I=I+1
             X(IA)=XCORD(I)
             Y(IA)=YCORD(I)
             Z(IA)=ZCORD(I)
          ENDDO
       ENDDO
#if KEY_REPDSTR==1
    else
       !
       !        this will change once we have selection on
       !        repd command
       ia=natrep*irepdstr
       do j = 1, nrxatm
          x(j)=xcord(ia+j)
          y(j)=ycord(ia+j)
          z(j)=zcord(ia+j)
       enddo

    endif
#endif 
    !
200 FORMAT('RX_4_SETUP4> ',I10,3F16.8)
300 FORMAT('RX_4_SETUP4> ',I5,4F16.8)

    RETURN
  END SUBROUTINE RX_4_SETUP4

  SUBROUTINE TRIDIAG(A,B,C,R,U,N)
    use number
    use stream
    use memory

    INTEGER N
    real(chm_real) A(N),B(N),C(N),R(N),U(N)
    INTEGER J
    real(chm_real) BET
    real(chm_real),allocatable,dimension(:) :: GAM

    call chmalloc('rxcons_4.src','TRIDIAG','GAM',N,crl=gam)
    IF(B(1).EQ.0.D0)CALL WRNDIE(-1,'<TRIDIAG>', &
         'TRIDIAG failed')
    BET=B(1)
    U(1)=R(1)/BET
    DO J=2,N
       GAM(J)=C(J-1)/BET
       BET=B(J)-A(J)*GAM(J)
       IF(BET.EQ.0.D0)CALL WRNDIE(-1,'<TRIDIAG>', &
            'TRIDIAG failed')
       U(J)=(R(J)-A(J)*U(J-1))/BET
    ENDDO
    DO J=N-1,1,-1
       U(J)=U(J)-GAM(J+1)*U(J+1)
    ENDDO

    call chmdealloc('rxcons_4.src','TRIDIAG','GAM',N,crl=gam)

    RETURN
  END SUBROUTINE TRIDIAG

  SUBROUTINE GETDTLVS(NPSG,NREP,NRXATM, &
       XM1,YM1,ZM1,XP1,YP1,ZP1, &
       DTLV,DTLVC,DVDRL,DVDR,DVDRR)
    use number
    use stream

    INTEGER NPSG,NRXATM,NREP
    real(chm_real) XM1(NREP,*),YM1(NREP,*),ZM1(NREP,*)
    real(chm_real) XP1(NREP,*),YP1(NREP,*),ZP1(NREP,*)
    real(chm_real) DTLV(*),DTLVC(*),DVDRL(*),DVDR(*),DVDRR(*)

    INTEGER I,J,IA,JREP,IREP

    real(chm_real) SUM,SUM1

    DO IREP=1,NPSG
       JREP = IREP + 1
       IF (JREP > NREP) JREP = JREP - NREP

       SUM=ZERO
       SUM1=ZERO

       DO I=1,NRXATM
          SUM=SUM+XM1(JREP,I)*XM1(JREP,I)+ &
               YM1(JREP,I)*YM1(JREP,I)+ &
               ZM1(JREP,I)*ZM1(JREP,I)

          SUM1=SUM1+XM1(JREP,I)*XP1(JREP,I)+ &
               YM1(JREP,I)*YP1(JREP,I)+ &
               ZM1(JREP,I)*ZP1(JREP,I)
       ENDDO

       DTLV(IREP) =SUM
       DTLVC(IREP)=SUM1

       IF(IREP.EQ.1)THEN
          DVDR(IREP) = DTLV(IREP)
          DVDRR(IREP)=-DTLVC(IREP)
       ELSEIF(IREP.EQ.NPSG)THEN
          DVDRL(IREP)=-DTLVC(IREP-1)
          DVDR(IREP) = DTLV(IREP)
       ELSE
          DVDRL(IREP)=-DTLVC(IREP-1)
          DVDR(IREP) = DTLV(IREP)*TWO
          DVDRR(IREP)=-DTLVC(IREP)
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE GETDTLVS

  SUBROUTINE GETVECTS(NREP,NPSG,NRXNCS,NRXATM, &
       XCORD,YCORD,ZCORD, &
       WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
       XM1,YM1,ZM1,XP1,YP1,ZP1,ARMS,RMST, &
       QCNOTRN,QCNOROT,QCYCL)
    !
    ! xtan, ytan, ztan are arrays
    ! each corresponds to the derivative of di,i+1
    ! wrt ri (xtan(1,*,*)) and ri+1 (xtan(2,*,*))
    !
    use number
    use stream

    implicit none

    INTEGER NREP,NPSG,NRXNCS,NRXATM
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) WMASS(*),ATMASS(*)
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),DEVA(3,3)
    real(chm_real) XM1(NREP,*),YM1(NREP,*),ZM1(NREP,*)
    real(chm_real) XP1(NREP,*),YP1(NREP,*),ZP1(NREP,*)
    real(chm_real) RMST,ARMS(*)

    real(chm_real) ROTRB,ROTRA,TEMP,FACT,EVWID
    INTEGER I,J,K
    INTEGER HPT,IPT,JPT,KPT,LPT,JREP,KREP,NALL
    LOGICAL QCNOTRN,QCNOROT,QCYCL
    real(chm_real) ATMP,WTOT,SUM,F1,F2
    real(chm_real) TEMPX,TEMPY,TEMPZ,TEMPL

    INTEGER IBEGIN,ISTOP

    evwid = zero ! not set & local, not sure of intent here

    NALL=NREP*NRXATM
    XM1(1:nrep,1:nrxatm)=zero
    YM1(1:nrep,1:nrxatm)=zero
    ZM1(1:nrep,1:nrxatm)=zero
    XP1(1:nrep,1:nrxatm)=zero
    YP1(1:nrep,1:nrxatm)=zero
    ZP1(1:nrep,1:nrxatm)=zero

    TEMP=ONE/DBLE(NPSG)

    IBEGIN=1
    ISTOP=NPSG

    RMST=ZERO
    DO JREP=IBEGIN,ISTOP
       KREP=JREP+1
       IF(JREP.EQ.ISTOP)THEN
          IF(QCYCL)THEN
             KREP=1
          ENDIF
       ENDIF

       IPT=(JREP-1)*NRXATM
       JPT=(KREP-1)*NRXATM

       DO I=1,NRXATM
          IPT=IPT+1
          JPT=JPT+1
          ATMPR(1,I)=IPT
          ATMPR(2,I)=JPT
       ENDDO
       !
       CALL ECBSTF(XCORD,YCORD,ZCORD,XCORD,YCORD,ZCORD, &
            ATMPR,NRXATM, &
            QCNOTRN,QCNOROT,.FALSE., (/ ZERO /), .FALSE.,WMASS, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       !        write(outu,*)'GETVECTS>jrep,wtot=',jrep, wtot

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(JREP)=ATMP
       RMST=RMST+ATMP

       TEMPL=ONE/ARMS(JREP)/WTOT

       DO I=1,NRXATM
          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(1,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,1)*RA(J,I)
          ENDDO
          XP1(JREP,I)=(ROTRB-RA(1,I))*TEMPL
          XM1(KREP,I)=(RB(1,I)-ROTRA)*TEMPL

          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(2,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,2)*RA(J,I)
          ENDDO
          YP1(JREP,I)=(ROTRB-RA(2,I))*TEMPL
          YM1(KREP,I)=(RB(2,I)-ROTRA)*TEMPL

          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(3,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,3)*RA(J,I)
          ENDDO
          ZP1(JREP,I)=(ROTRB-RA(3,I))*TEMPL
          ZM1(KREP,I)=(RB(3,I)-ROTRA)*TEMPL
       ENDDO
    ENDDO

    RMST=RMST*TEMP

    DO JREP=1,NREP
       DO I=1,NRXATM
          FACT=WMASS(I)
          XM1(JREP,I)=XM1(JREP,I)*FACT
          YM1(JREP,I)=YM1(JREP,I)*FACT
          ZM1(JREP,I)=ZM1(JREP,I)*FACT
          XP1(JREP,I)=XP1(JREP,I)*FACT
          YP1(JREP,I)=YP1(JREP,I)*FACT
          ZP1(JREP,I)=ZP1(JREP,I)*FACT
       ENDDO
    ENDDO

    RETURN
  END SUBROUTINE GETVECTS

  SUBROUTINE GETTAN(NREP,NPSG,NRXNCS,NRXATM, &
       XCORD,YCORD,ZCORD, &
       WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
       XTAN,YTAN,ZTAN, &
       XM1,YM1,ZM1,XP1,YP1,ZP1, &
       ARMS,RMST,QCNOTRN,QCNOROT,QCYCL, &
       DLTAN,DLCRV,D_TAN)

    use number
    use stream
    use epathmod

    implicit none

    INTEGER NREP,NPSG,NRXNCS,NRXATM
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) WMASS(*),ATMASS(*)
    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM),DEVA(3,3)
    real(chm_real) XTAN(NREP,*),YTAN(NREP,*),ZTAN(NREP,*)
    real(chm_real) XM1(NREP,*),YM1(NREP,*),ZM1(NREP,*)
    real(chm_real) XP1(NREP,*),YP1(NREP,*),ZP1(NREP,*)
    real(chm_real) RMST,ARMS(*),DLTAN(*),DLCRV(*),D_TAN(*)

    real(chm_real) ROTRB,ROTRA,TEMP,FACT,EVWID
    INTEGER I,J,K
    INTEGER IPT,JPT,KPT,JREP,KREP,NALL,IREP
    LOGICAL QCNOTRN,QCNOROT,QCYCL
    real(chm_real) ATMP,WTOT,SUM,AVG,SUM1
    real(chm_real) TEMPX,TEMPY,TEMPZ,A,B

    INTEGER IBEGIN,ISTOP

    evwid = zero ! not set & local, not sure of intent here
    NALL=NREP*NRXATM
    XTAN (1:nrep,1:nrxatm) =zero
    YTAN (1:nrep,1:nrxatm) =zero
    ZTAN (1:nrep,1:nrxatm) =zero
    XM1  (1:nrep,1:nrxatm) =zero
    YM1  (1:nrep,1:nrxatm) =zero
    ZM1  (1:nrep,1:nrxatm) =zero
    XP1  (1:nrep,1:nrxatm) =zero
    YP1  (1:nrep,1:nrxatm) =zero
    ZP1  (1:nrep,1:nrxatm) =zero
    DLTAN(1:NREP+1)=zero
    TEMP=ONE/DBLE(NPSG)

    !      write(*,*)' hi rcons '
    !      do i=1,nall
    !        write(*,*)i,xcord(i),ycord(i),zcord(i)
    !      enddo
    !      write(*,*)' hi rcons '

    RMST=ZERO

    DO JREP=1,NPSG
       KREP=JREP+1
       IF(JREP.EQ.NREP)THEN
          IF(QCYCL)THEN
             KREP=1
          ENDIF
       ENDIF

       IPT=(JREP-1)*NRXATM
       JPT=(KREP-1)*NRXATM

       DO I=1,NRXATM
          IPT=IPT+1
          JPT=JPT+1
          ATMPR(1,I)=IPT
          ATMPR(2,I)=JPT
       ENDDO

       CALL ECBSTF(XCORD,YCORD,ZCORD,XCORD,YCORD,ZCORD, &
            ATMPR,NRXATM, &
            QCNOTRN,QCNOROT,.FALSE., (/ ZERO /), .FALSE.,WMASS, &
            WTOT,ATMP,RA,RB,DEVA,EVWID)

       ATMP=ATMP/WTOT
       IF(ATMP.LT.ZERO) ATMP=ZERO
       ATMP=SQRT(ATMP)
       ARMS(JREP)=ATMP

       RMST=RMST+ATMP

       DO I=1,NRXATM
          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(1,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,1)*RA(J,I)
          ENDDO
          XP1(JREP,I)=(ROTRB-RA(1,I))
          XM1(KREP,I)=(RB(1,I)-ROTRA)

          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(2,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,2)*RA(J,I)
          ENDDO
          YP1(JREP,I)=(ROTRB-RA(2,I))
          YM1(KREP,I)=(RB(2,I)-ROTRA)

          ROTRB=ZERO
          ROTRA=ZERO
          DO J=1,3
             ROTRB=ROTRB+DEVA(3,J)*RB(J,I)
             ROTRA=ROTRA+DEVA(J,3)*RA(J,I)
          ENDDO
          ZP1(JREP,I)=(ROTRB-RA(3,I))
          ZM1(KREP,I)=(RB(3,I)-ROTRA)
       ENDDO
    ENDDO

    RMST=RMST*TEMP

    TEMP=ONE/WTOT

    !      write(*,*)' rxcons '
    !      do jrep=1,nrep
    !        do i=1,nrxatm
    !          write(*,*)jrep,i,xtan(jrep,i),ytan(jrep,i),ztan(jrep,i)
    !        enddo
    !      enddo
    !      write(*,*)' rxcons '

    IBEGIN=2
    ISTOP=NPSG
    IF(QCYCL)IBEGIN=1

    DO JREP=IBEGIN,ISTOP
       DO I=1,NRXATM
          XTAN(JREP,I)=XM1(JREP,I)+XP1(JREP,I)
          YTAN(JREP,I)=YM1(JREP,I)+YP1(JREP,I)
          ZTAN(JREP,I)=ZM1(JREP,I)+ZP1(JREP,I)
          XP1(JREP,I) =XP1(JREP,I)-XM1(JREP,I)
          YP1(JREP,I) =YP1(JREP,I)-YM1(JREP,I)
          ZP1(JREP,I) =ZP1(JREP,I)-ZM1(JREP,I)
       ENDDO
    ENDDO

    DO JREP=IBEGIN,ISTOP
       SUM=ZERO
       SUM1=ZERO
       DO I=1,NRXATM
          FACT=WMASS(I)*TEMP
          TEMPX=XTAN(JREP,I)
          TEMPY=YTAN(JREP,I)
          TEMPZ=ZTAN(JREP,I)
          SUM=SUM+(TEMPX*TEMPX+TEMPY*TEMPY+TEMPZ*TEMPZ) &
               *FACT**2
          SUM1=SUM1+(TEMPX*TEMPX+TEMPY*TEMPY+TEMPZ*TEMPZ) &
               *FACT
          XTAN(JREP,I)=XTAN(JREP,I)*FACT
          YTAN(JREP,I)=YTAN(JREP,I)*FACT
          ZTAN(JREP,I)=ZTAN(JREP,I)*FACT
       ENDDO
       SUM=SQRT(SUM)*HALF
       SUM1=SQRT(SUM1)*HALF
       DLTAN(JREP)=SUM1*SUM1/SUM
    ENDDO

    CALL NORM_PATH(NREP,NRXATM,XTAN,YTAN,ZTAN,D_TAN,qcycl)

    DO JREP=IBEGIN,ISTOP
       SUM=ZERO
       DO I=1,NRXATM
          FACT=WMASS(I)*TEMP
          TEMPX=XP1(JREP,I)
          TEMPY=YP1(JREP,I)
          TEMPZ=ZP1(JREP,I)
          SUM=SUM+(TEMPX*TEMPX+TEMPY*TEMPY+TEMPZ*TEMPZ)*FACT
          XP1(JREP,I)=XP1(JREP,I)*FACT
          YP1(JREP,I)=YP1(JREP,I)*FACT
          ZP1(JREP,I)=ZP1(JREP,I)*FACT
       ENDDO
       SUM=SQRT(SUM)
       DLCRV(JREP)=SUM*HALF
    ENDDO

    CALL NORM_PATH(NREP,NRXATM,XP1,YP1,ZP1,D_TAN,qcycl)

    !      open(66,file="tan1.dat",status="unknown")
    !      do jrep=1,nrep
    !        do i=1,nrxatm
    !          write(66,'(2I5,3F16.6)')Jrep,i,
    !     $          xtan(jrep,i),ytan(jrep,i),ztan(jrep,i)
    !        enddo
    !      enddo
    !      close(66)
    !      open(66,file="crv1.dat",status="unknown")
    !      do jrep=1,nrep
    !        do i=1,nrxatm
    !          write(66,'(2I5,4F16.6)')Jrep,i,
    !     $          xp1(jrep,i),yp1(jrep,i),zp1(jrep,i),DLCRV(JREP)
    !        enddo
    !      enddo
    !      close(66)

    RETURN
  END SUBROUTINE GETTAN


  SUBROUTINE RXNCONS_4(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
       NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MXITER,PLAGRANM, &
       LRXPRN,IRXATM,XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
       DTSQ,ZFAC,GFAC,NRXNCS,IMOVE, &
       NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)
    !
#if KEY_REPLICA==1
    use EPATHMOD                            
#endif
    use number
    use stream
    use replica_mod
    use memory
    use rxnconswt
    use rxncons4
    use parallel
    use repdstr
    !
    ! . Parsed variables
    real(chm_real),allocatable,dimension(:) :: XF
    real(chm_real),allocatable,dimension(:) :: YF
    real(chm_real),allocatable,dimension(:) :: ZF
    real(chm_real),allocatable,dimension(:) :: XTAN
    real(chm_real),allocatable,dimension(:) :: YTAN
    real(chm_real),allocatable,dimension(:) :: ZTAN
    real(chm_real),allocatable,dimension(:) :: XM1
    real(chm_real),allocatable,dimension(:) :: YM1
    real(chm_real),allocatable,dimension(:) :: ZM1
    real(chm_real),allocatable,dimension(:) :: XP1
    real(chm_real),allocatable,dimension(:) :: YP1
    real(chm_real),allocatable,dimension(:) :: ZP1
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA
    real(chm_real),allocatable,dimension(:) :: RB
    real(chm_real),allocatable,dimension(:) :: ARMS
    real(chm_real),allocatable,dimension(:) :: FRMS
    real(chm_real),allocatable,dimension(:) :: DERR
    real(chm_real),allocatable,dimension(:) :: D_TAN
    real(chm_real),allocatable,dimension(:) :: DTLV
    real(chm_real),allocatable,dimension(:) :: DTLVC
    real(chm_real),allocatable,dimension(:) :: DVDRL
    real(chm_real),allocatable,dimension(:) :: DVDR
    real(chm_real),allocatable,dimension(:) :: DVDRR
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
    real(chm_real) AMASS(*)
    INTEGER NATOM,IRXCNS,NRXATM,IRXATM(*),MXITER,NRXNCS
    LOGICAL LRXCNS,LDYNA,PLAGRANM,LRXPRN
    real(chm_real) VRXNCS,LAGRANM,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),DTSQ,ZFAC,GFAC
    INTEGER LPRID(*)
    INTEGER IMOVE(*),NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    real(chm_real) LPVALUE(3,*)
    LOGICAL LPWGHT(*)
    !
#if KEY_PARALLEL==1
    real(chm_real) GCARR(10),TIMMER
#endif 
    !
    ! . Local variables
    !
    INTEGER I, IA
    INTEGER ATFRST,ATLAST

    INTEGER NALLOC
    INTEGER NATREP
    INTEGER,ALLOCATABLE,DIMENSION(:) :: ISTREP,ICNREP
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: WEIGHT
    INTEGER XO,YO,ZO

    INTEGER J,IREP,JREP,NREP,NPSG
    real(chm_real) WTOT
    LOGICAL ERR,QPRINT,QCYCL
    INTEGER ILP,IPT,JPT,K,N
    real(chm_real) XD,YD,ZD
    !
    ! . Just testing
    !
    !      ATFRST=1
    !      ATLAST=NATOM
    !...##IF PARALLEL
    !...##IF PARAFULL
    !C     Define the atom bounds for this processor.
    !      IF(LDYNA) THEN
    !        ATFRST=1+IPARPT(MYNOD)
    !        ATLAST=IPARPT(MYNODP)
    !        WRITE(OUTU,*)' Hi ', MYNODP, ATFRST, ATLAST
    !      ENDIF
    !...##ENDIF
    !...##ENDIF
    !
    !
    !
    !. Copy coordinates
    !
    !...##IF PARALLEL
    !            CALL PSYNC()
    !            TIMMER=ECLOCK()
    !c            CALL VDGBR(X,Y,Z,1)
    !c            CALL VDGBR(XREF,YREF,ZREF,1)
    !            TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
    !            CALL PSYNC()
    !            TIMMER=ECLOCK()
    !...##ENDIF

    ! no need to add one here !
    !!      if(qrepdstr)

    ! Make sure we have correct system sizes: some routines may play with them
    ! the next line turns out to be the fix when running dynamics. Mini was OK
#if KEY_REPDSTR==1
    if(qrepdstr)nrepl=nrepdstr+1     
#endif

    NREP=NREPL-1
    NPSG=NREP-1
    QCYCL=QPTCSCL
    IF(QCYCL)NPSG=NREP

    NALLOC=NRXATM*NREP

#if KEY_REPDSTR==1
    if(.not.qrepdstr) then            
#endif
       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NRXATM*(JREP-1)+IRXATM(J)
             I=I+1
             XCORD(I)=X(IA)
             YCORD(I)=Y(IA)
             ZCORD(I)=Z(IA)
             !         IF(MYNOD.EQ.0) THEN
             !           WRITE(OUTU,*)' X    ',IA,XCORD(I),YCORD(I),ZCORD(I)
             !         ENDIF
          ENDDO
       ENDDO
#if KEY_REPDSTR==1
    else
       !
       !        this will change once we have selection on
       !        repd command
       xcord(1:nalloc)=zero
       ycord(1:nalloc)=zero
       zcord(1:nalloc)=zero
       ia=nrxatm*irepdstr
       do j = 1, nrxatm
          xcord(ia+j)=x(j)
          ycord(ia+j)=y(j)
          zcord(ia+j)=z(j)
       enddo
       if ( mynodg .eq. (numnod*irepdstr)) then
          !           this has to be protected! Not everyone can call this!!!
          call repdbr(xcord)
          call repdbr(ycord)
          call repdbr(zcord)
       endif
       !  see the comments about the efficiency before the psnd8 calls above
       call psnd8(xcord,nalloc)
       call psnd8(ycord,nalloc)
       call psnd8(zcord,nalloc)
    endif
#endif 

    QPRINT=((PRNLEV.GE.6).OR.LRXPRN)
#if KEY_REPDSTR==1
    if(qrepdstr)nrepl=nrepdstr+1           
#endif
    ! rxncons_4 chmalloc
    call chmalloc('rxcons_4.src','RXNCONS_4','WEIGHT',NATOM,crl=weight)
    call chmalloc('rxcons_4.src','RXNCONS_4','ISTREP',NREPL,intg=istrep)
    call chmalloc('rxcons_4.src','RXNCONS_4','ICNREP',NREPL,intg=icnrep)
    !
    CALL EPATHC(ISTREP,ICNREP,WEIGHT, &
         NATREP,WTOT,QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
         QCWCOMP2,  & 
#endif
         QCMASS,QPRINT,ERR)
#if KEY_REPDSTR==1
    !       again after epathc add +1
    if(qrepdstr)nrepl=nrepdstr+1
    !        write(outu,*)'RXNCONS_4>icnrep1=', icnrep(1)
    !        write(outu,*)'RXNCONS_4>nrepl=', nrepl
#endif 
    !
    !      write(outu,*)'RXNCONS_4:after epathc>nrepl=', nrepl

    NALLOC=NRXATM*NREP

    ! nrepl may change in epathc, or in other places
    call chmalloc('rxcons_4.src','RXNCONS_4','ARMS',NREPL,crl=ARMS)
    call chmalloc('rxcons_4.src','RXNCONS_4','FRMS',NREPL,crl=FRMS)
    call chmalloc('rxcons_4.src','RXNCONS_4','DERR',NREPL,crl=DERR)


    ARMS(1:NREPL)=zero
    FRMS(1:NREPL)=zero
    DERR(1:NREPL)=zero

    !      write(outu,*)'RXNCONS_4:after vzero>nrepl=', nrepl

    call chmalloc('rxcons_4.src','RXNCONS_4','XF',NALLOC,crl=XF)
    call chmalloc('rxcons_4.src','RXNCONS_4','YF',NALLOC,crl=YF)
    call chmalloc('rxcons_4.src','RXNCONS_4','ZF',NALLOC,crl=ZF)
    call chmalloc('rxcons_4.src','RXNCONS_4','XTAN',NALLOC,crl=XTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4','YTAN',NALLOC,crl=YTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4','ZTAN',NALLOC,crl=ZTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4','XM1',NALLOC,crl=XM1)
    call chmalloc('rxcons_4.src','RXNCONS_4','YM1',NALLOC,crl=YM1)
    call chmalloc('rxcons_4.src','RXNCONS_4','ZM1',NALLOC,crl=ZM1)
    call chmalloc('rxcons_4.src','RXNCONS_4','XP1',NALLOC,crl=XP1)
    call chmalloc('rxcons_4.src','RXNCONS_4','YP1',NALLOC,crl=YP1)
    call chmalloc('rxcons_4.src','RXNCONS_4','ZP1',NALLOC,crl=ZP1)
    call chmalloc('rxcons_4.src','RXNCONS_4','ATMPR',2*NRXATM,intg=ATMPR)
    call chmalloc('rxcons_4.src','RXNCONS_4','RA',3*NRXATM,crl=RA)
    call chmalloc('rxcons_4.src','RXNCONS_4','RB',3*NRXATM,crl=RB)
    call chmalloc('rxcons_4.src','RXNCONS_4','D_TAN',NREPL,crl=D_TAN)
    call chmalloc('rxcons_4.src','RXNCONS_4','DTLV',NREPL,crl=DTLV)
    call chmalloc('rxcons_4.src','RXNCONS_4','DTLVC',NREPL,crl=DTLVC)
    call chmalloc('rxcons_4.src','RXNCONS_4','DVDRL',NREPL,crl=DVDRL)
    call chmalloc('rxcons_4.src','RXNCONS_4','DVDR',NREPL,crl=DVDR)
    call chmalloc('rxcons_4.src','RXNCONS_4','DVDRR',NREPL,crl=DVDRR)

    CALL RXNCONS_42(NRXATM,VRXNCS,LAGRANM,MXITER,IRXATM, &
         NRXNCS,NREPL,NPSG,QPRINT, &
         XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC,ZFAC,GFAC, &
         XTAN,YTAN,ZTAN, &
         XM1,YM1,ZM1, &
         XP1,YP1,ZP1, &
         XF,YF,ZF, &
         ATMPR,RA,RB,WMASS, &
         ATMASS,WTOT,QCNOTRN,QCNOROT,QPTCSCL, &
         ISTREP,ICNREP, &
         ARMS,FRMS,DERR,D_TAN, &
         DTLV,DTLVC, &
         DVDRL,DVDR,DVDRR, &
         QFXREP,IFXREP,DLTAN,DLCRV)
    !
    !    repdstr needs this to distribute
    !
    !!      write(outu,*)'RXNCONS_4>after RXNCONS_42'

    !!      write(outu,*)'rxncons_4>qrepdstr,natrep,nrxatm,irepdstr=', &
    !!           qrepdstr,natrep,nrxatm,irepdstr

#if KEY_REPDSTR==1
    if(.not.qrepdstr)then                     
#endif
       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NATREP*(JREP-1)+IRXATM(J)
             I=I+1
             IF(IMOVE(IA).NE.-1)THEN
                X(IA)=XCORD(I)
                Y(IA)=YCORD(I)
                Z(IA)=ZCORD(I)
             ELSE
                ILP=LPRID(IA)

                N=LPNHOST(ILP)
                IPT=LPHPTR(ILP)

                XD=XCORD(I)-X(IA)
                YD=YCORD(I)-Y(IA)
                ZD=ZCORD(I)-Z(IA)

                X(IA)=XCORD(I)
                Y(IA)=YCORD(I)
                Z(IA)=ZCORD(I)

                DO K=1,N
                   JPT=LPHOST(IPT+K)
                   X(JPT)=X(JPT)+XD
                   Y(JPT)=Y(JPT)+YD
                   Z(JPT)=Z(JPT)+ZD
                ENDDO
             ENDIF
          ENDDO
       ENDDO
#if KEY_REPDSTR==1
    else
       ! to preserve the same code ia is a local index and I
       ! runs over all replicas
       I=natrep*irepdstr
       DO IA=1,NRXATM
          !          IA=NATREP*(JREP-1)+IRXATM(J)
          I=I+1
          IF(IMOVE(IA).NE.-1)THEN
             X(IA)=XCORD(I)
             Y(IA)=YCORD(I)
             Z(IA)=ZCORD(I)
          ELSE
             ILP=LPRID(IA)

             N=LPNHOST(ILP)
             IPT=LPHPTR(ILP)

             XD=XCORD(I)-X(IA)
             YD=YCORD(I)-Y(IA)
             ZD=ZCORD(I)-Z(IA)

             X(IA)=XCORD(I)
             Y(IA)=YCORD(I)
             Z(IA)=ZCORD(I)

             DO K=1,N
                JPT=LPHOST(IPT+K)
                X(JPT)=X(JPT)+XD
                Y(JPT)=Y(JPT)+YD
                Z(JPT)=Z(JPT)+ZD
             ENDDO
          ENDIF
       ENDDO

    endif
#endif 
#if KEY_REPDSTR==1
!!!          if(qrepdstr)ia=j              
#endif

    ! rxncons_4 chmdealloc

#if KEY_REPDSTR==1
    if(qrepdstr)nrepl=nrepdstr+1           
#endif

    call chmdealloc('rxcons_4.src','RXNCONS_4','ATMPR',2*NRXATM,intg=ATMPR)
    call chmdealloc('rxcons_4.src','RXNCONS_4','RA',3*NRXATM,crl=RA)
    call chmdealloc('rxcons_4.src','RXNCONS_4','RB',3*NRXATM,crl=RB)

    NALLOC=NRXATM*NREP

    call chmdealloc('rxcons_4.src','RXNCONS_4','XF',NALLOC,crl=XF)
    call chmdealloc('rxcons_4.src','RXNCONS_4','YF',NALLOC,crl=YF)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ZF',NALLOC,crl=ZF)
    call chmdealloc('rxcons_4.src','RXNCONS_4','XTAN',NALLOC,crl=XTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4','YTAN',NALLOC,crl=YTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ZTAN',NALLOC,crl=ZTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4','XM1',NALLOC,crl=XM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4','YM1',NALLOC,crl=YM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ZM1',NALLOC,crl=ZM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4','XP1',NALLOC,crl=XP1)
    call chmdealloc('rxcons_4.src','RXNCONS_4','YP1',NALLOC,crl=YP1)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ZP1',NALLOC,crl=ZP1)

    call chmdealloc('rxcons_4.src','RXNCONS_4','ARMS',NREPL,crl=ARMS)
    call chmdealloc('rxcons_4.src','RXNCONS_4','FRMS',NREPL,crl=FRMS)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DERR',NREPL,crl=DERR)
    call chmdealloc('rxcons_4.src','RXNCONS_4','WEIGHT',NATOM,crl=weight)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ISTREP',NREPL,intg=istrep)
    call chmdealloc('rxcons_4.src','RXNCONS_4','ICNREP',NREPL,intg=icnrep)
    call chmdealloc('rxcons_4.src','RXNCONS_4','D_TAN',NREPL,crl=D_TAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DTLV',NREPL,crl=DTLV)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DTLVC',NREPL,crl=DTLVC)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DVDRL',NREPL,crl=DVDRL)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DVDR',NREPL,crl=DVDR)
    call chmdealloc('rxcons_4.src','RXNCONS_4','DVDRR',NREPL,crl=DVDRR)

    !      write(outu,*)'RXNCONS_4>end of routine'

    RETURN
  END SUBROUTINE RXNCONS_4

  SUBROUTINE RXNCONS_42(NRXATM,VRXNCS,LAGRANM,MXITER,IRXATM, &
       NRXNCS,NREPL,NPSG,LPRINT, &
       XCORD,YCORD,ZCORD,XREFC,YREFC,ZREFC, &
       ZFAC,GFAC,XTAN,YTAN,ZTAN, &
       XM1,YM1,ZM1,XP1,YP1,ZP1, &
       XF,YF,ZF,ATMPR,RA,RB, &
       WMASS,ATMASS,WTOT,QCNOTRN,QCNOROT, &
       QCYCL,ISTREP,ICNREP,ARMS,FRMS,DERR,D_TAN, &
       DTLV,DTLVC,DVDRL,DVDR,DVDRR, &
       QFXREP,IFXREP,DLTAN,DLCRV)

    use stream
    use number

    ! . Parsed variables
    !
    INTEGER NATOM,NRXATM,IRXATM(*),MXITER,NRXNCS,NPSG
    real(chm_real) VRXNCS,LAGRANM,XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),ZFAC,GFAC
    real(chm_real) XF(*),YF(*),ZF(*)
    INTEGER NREPL,ISTREP(*),ICNREP(*)

    INTEGER ATMPR(2,NRXATM),IFXREP
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM)
    real(chm_real) XTAN(NREPL-1,*),YTAN(NREPL-1,*),ZTAN(NREPL-1,*)
    real(chm_real) XM1(NREPL-1,*),YM1(NREPL-1,*),ZM1(NREPL-1,*)
    real(chm_real) XP1(NREPL-1,*),YP1(NREPL-1,*),ZP1(NREPL-1,*)
    real(chm_real) WMASS(*),ATMASS(*),WTOT,ARMS(*),FRMS(*),DERR(*)
    real(chm_real) DLTAN(*),D_TAN(*),DLCRV(*)
    real(chm_real) DTLV(*),DTLVC(*),DVDRL(*),DVDR(*),DVDRR(*)

    LOGICAL QCNOTRN,QCNOROT,QCYCL,QFXREP,LPRINT
    !

    !
    ! . Local variables
    !
    INTEGER ITER
    INTEGER NREP,IREP,JREP,KREP,NALL,IBEGIN,ISTOP
    INTEGER IPT,JPT,I,J,IA,IB,IC
    real(chm_real) RMST,RMST0,TEMP,SUM,FACT
    real(chm_real) DEVA(3,3),EVWID
    real(chm_real) DT,AERR,ATMP

    NREP=NREPL-1

    NALL=NREP*NRXATM

    TEMP=ONE/DBLE(NPSG)

    !      write(outu,*)'RXNCONS_42>begin of routine'
    !      write(outu,*)'RXNCONS_42>nrep,nall,temp=',nrep,nall,temp

    ! Backup original coordinates
    I=0
    DO JREP=1,NREP
       DO J=1,NRXATM
          I=I+1
          XF(I)=XCORD(I)
          YF(I)=YCORD(I)
          ZF(I)=ZCORD(I)
       ENDDO
    ENDDO

    !      write(outu,*)'RXNCONS_42>after XF()'
    !
    ! define the range of active replicas
    !
    IF(QCYCL)THEN
       IBEGIN=2
       ISTOP=NREP
    ELSE
       IBEGIN=2
       ISTOP=NREP-1
    ENDIF

    FRMS (1:NREPL)=zero
    ARMS (1:NREPL)=zero
    DERR (1:NREPL)=zero
    DTLV (1:NREPL)=zero
    DTLVC(1:NREPL)=zero
    DVDRL(1:NREPL)=zero
    DVDR (1:NREPL)=zero
    DVDRR(1:NREPL)=zero

    ! initial call

    !      write(outu,*)'RXNCONS_42>before getvects...'

    CALL GETVECTS(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XM1,YM1,ZM1,XP1,YP1,ZP1,ARMS,RMST, &
         QCNOTRN,QCNOROT,QCYCL)

    !      write(outu,*)'RXNCONS_42>after getvects...'

    RMST0=RMST

    SUM=ZERO
    DO I=1,NPSG
       DERR(I)=-(ARMS(I)-RMST0)*HALF
       SUM=SUM+DERR(I)*DERR(I)
    ENDDO
    AERR=SQRT(SUM*TEMP)

    ! iteration

    ITER=0
1111 CONTINUE

    CALL GETDTLVS(NPSG,NREP,NRXATM, &
         XM1,YM1,ZM1,XP1,YP1,ZP1, &
         DTLV,DTLVC,DVDRL,DVDR,DVDRR)

    CALL TRIDIAG(DVDRL,DVDR,DVDRR,DERR,FRMS,NPSG)

    !      DO I=1,NPSG
    !        WRITE(OUTU,'(I5,7F16.6)')I,ARMS(I),RMST0,
    !     $                     DVDRL(I),DVDR(I),DVDRR(I),DERR(I),FRMS(I)
    !      ENDDO

    DO JREP=IBEGIN,ISTOP
       IREP=JREP-1
       !        write(50+irep-1,'(a,8i5)')
       !     $    'RXNCONS4>irep,nrep,nrxatm,ibegin,istop,iter=',
       !     $    irep,nrep,nrxatm,ibegin,istop,iter

       IA=(JREP-1)*NRXATM
       DO I=1,NRXATM
          IA=IA+1
          XCORD(IA)=XCORD(IA)+ &
               XM1(JREP,I)*FRMS(IREP)-XP1(JREP,I)*FRMS(JREP)
          YCORD(IA)=YCORD(IA)+ &
               YM1(JREP,I)*FRMS(IREP)-YP1(JREP,I)*FRMS(JREP)
          ZCORD(IA)=ZCORD(IA)+ &
               ZM1(JREP,I)*FRMS(IREP)-ZP1(JREP,I)*FRMS(JREP)
       ENDDO

    ENDDO

    CALL GETVECTS(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XM1,YM1,ZM1,XP1,YP1,ZP1,ARMS,RMST, &
         QCNOTRN,QCNOROT,QCYCL)

    SUM=ZERO
    DO I=1,NPSG
       DERR(I)=-(ARMS(I)-RMST0)*HALF
       !        WRITE(OUTU,200)I,ARMS(I),DERR(I),FRMS(I)
       SUM=SUM+DERR(I)*DERR(I)
    ENDDO
    AERR=SQRT(SUM*TEMP)

    ITER=ITER+1
    IF(LPRINT)WRITE(OUTU,'(A,I5,E16.6)')'ITER ',ITER,AERR

    IF(AERR.GT.TENM8)THEN
       IF(ITER.LE.MXITER)THEN
          GOTO 1111
       ELSE
          WRITE(OUTU,'(A,I5,E16.6)')'RXNCONS_42> ',ITER,AERR
          CALL WRNDIE(-1,'<RXNCONS_42>', &
               'Path distance constraints do not converge')
       ENDIF
    ENDIF

    !      WRITE(OUTU,'(A,I5,E16.6)')'CONVERGED!',ITER,AERR

    CALL GETTAN(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XTAN,YTAN,ZTAN,XM1,YM1,ZM1,XP1,YP1,ZP1, &
         ARMS,RMST,QCNOTRN,QCNOROT,QCYCL,DLTAN,DLCRV,D_TAN)


201 FORMAT('RXNCONS_42> ', &
         '   REPLICA', &
         '        DISTANCE', &
         '     DIFF TO AVG', &
         '    DIST TO PREV')

200 FORMAT('RXNCONS_42> ',I10,3F16.8)
300 FORMAT('RXNCONS_42> ',I5,4F16.8)

    RETURN
  END SUBROUTINE RXNCONS_42

  SUBROUTINE RXNCONS_4F(XREF,YREF,ZREF,DX,DY,DZ,NATOM,IRXCNS, &
       NRXATM,NRXNCS,LRXPRN,IRXATM,XCORD,YCORD,ZCORD, &
       XREFC,YREFC,ZREFC,IMOVE,NUMLP, &
       LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,AMASS,LPRID)
    !
#if KEY_REPLICA==1
    use EPATHMOD                            
#endif
    use number
    use stream
    use replica_mod
    use rxnconswt
    use rxncons4
    use memory
#if KEY_PARALLEL==1
    use parallel      
#endif
    use repdstr
    use machutil,only:eclock
    !
    ! . Parsed variables
    !
    real(chm_real),allocatable,dimension(:,:) :: XTAN
    real(chm_real),allocatable,dimension(:,:) :: YTAN
    real(chm_real),allocatable,dimension(:,:) :: ZTAN
    real(chm_real),allocatable,dimension(:,:) :: XM1
    real(chm_real),allocatable,dimension(:,:) :: YM1
    real(chm_real),allocatable,dimension(:,:) :: ZM1
    real(chm_real),allocatable,dimension(:,:) :: XP1
    real(chm_real),allocatable,dimension(:,:) :: YP1
    real(chm_real),allocatable,dimension(:,:) :: ZP1
    integer,allocatable,dimension(:) :: ATMPR
    real(chm_real),allocatable,dimension(:) :: RA
    real(chm_real),allocatable,dimension(:) :: RB
    real(chm_real),allocatable,dimension(:) :: ARMS
    real(chm_real),allocatable,dimension(:) :: D_TAN
    real(chm_real) XREF(*),YREF(*),ZREF(*),DX(*),DY(*),DZ(*)
    INTEGER NATOM,IRXCNS,NRXATM,NRXNCS,IRXATM(*)
    INTEGER LPRID(*)
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*),AMASS(*)
    LOGICAL LRXPRN

#if KEY_PARALLEL==1
    real(chm_real) GCARR(10),TIMMER
#endif 
    INTEGER NREP,IREP,JREP,KREP,NPSG,NALLOC
    INTEGER I,J,IA

    INTEGER,ALLOCATABLE,DIMENSION(:) :: ISTREP,ICNREP
    REAL(CHM_REAL),ALLOCATABLE,DIMENSION(:) :: WEIGHT

    INTEGER NATREP,ICS
    real(chm_real) WTOT
    LOGICAL QPRINT,ERR,QCYCL
    INTEGER IMOVE(*),NUMLP
    INTEGER LPNHOST(*),LPHPTR(*),LPHOST(*)
    INTEGER ILP,IPT,JPT,N,K
    real(chm_real) LPVALUE(3,*)
    LOGICAL LPWGHT(*)

#if KEY_PARALLEL==1
!    CALL PSYNC()
    TIMMER=ECLOCK()
    !            CALL VDGBR(XREF,YREF,ZREF,1)
    CALL VDGBR(DX,DY,DZ,1)
    TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
!    CALL PSYNC()
    TIMMER=ECLOCK()
#endif 

    QCYCL=QPTCSCL

#if KEY_REPDSTR==1
    IF(QREPDSTR) NREPL=NREPDSTR+1       
#endif
    NREP=NREPL-1
    NPSG=NREP-1
    IF(QCYCL)NPSG=NREP
    NALLOC=NRXATM*(NREPL-1)

#if KEY_REPDSTR==1
    if(.not.qrepdstr) then              
#endif

       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NRXATM*(JREP-1)+IRXATM(J)
             I=I+1
             XCORD(I)=XREF(IA)
             YCORD(I)=YREF(IA)
             ZCORD(I)=ZREF(IA)

             IF(IMOVE(IA).NE.-1)THEN
                XREFC(I)=DX(IA)
                YREFC(I)=DY(IA)
                ZREFC(I)=DZ(IA)
             ELSE
                !            write(*,*)' oo ',jrep,j,dx(ia),dy(ia),dz(ia)
                XREFC(I)=ZERO
                YREFC(I)=ZERO
                ZREFC(I)=ZERO

                ILP=LPRID(IA)

                N=LPNHOST(ILP)
                IPT=LPHPTR(ILP)

                DO K=1,N
                   JPT=LPHOST(IPT+K)
                   XREFC(I)=XREFC(I)+DX(JPT)
                   YREFC(I)=YREFC(I)+DY(JPT)
                   ZREFC(I)=ZREFC(I)+DZ(JPT)
                ENDDO

                DX(IA)=XREFC(I)
                DY(IA)=YREFC(I)
                DZ(IA)=ZREFC(I)
                !            write(*,*)'ooo ',jrep,j,dx(ia),dy(ia),dz(ia)
             ENDIF

             !         IF(MYNOD.EQ.0) THEN
             !           WRITE(OUTU,*)' X    ',IA,XCORD(I),YCORD(I),ZCORD(I)
             !           WRITE(OUTU,*)' XREF ',IA,XREFC(I),YREFC(I),ZREFC(I)
             !         ENDIF
          ENDDO
       ENDDO
#if KEY_REPDSTR==1
    else
       !fill-up xcord & xrefc
       xcord(1:nalloc)=zero
       ycord(1:nalloc)=zero
       zcord(1:nalloc)=zero
       xrefc(1:nalloc)=zero
       yrefc(1:nalloc)=zero
       zrefc(1:nalloc)=zero
       ia=nrxatm*irepdstr
       do j = 1, nrxatm
          xcord(ia+j)=xref(j)
          ycord(ia+j)=yref(j)
          zcord(ia+j)=zref(j)
          !  NEED the code for lonepairs
          IF(IMOVE(j).NE.-1)THEN
             xrefc(ia+j)=dx(j)
             yrefc(ia+j)=dy(j)
             zrefc(ia+j)=dz(j)
          ELSE
             !            write(*,*)' oo ',jrep,j,dx(ia),dy(ia),dz(ia)
             XREFC(ia+j)=ZERO
             YREFC(ia+j)=ZERO
             ZREFC(ia+j)=ZERO

             ILP=LPRID(j)

             N=LPNHOST(ILP)
             IPT=LPHPTR(ILP)

             DO K=1,N
                JPT=LPHOST(IPT+K)
                XREFC(ia+j)=XREFC(ia+j)+DX(JPT)
                YREFC(ia+j)=YREFC(ia+j)+DY(JPT)
                ZREFC(ia+j)=ZREFC(ia+j)+DZ(JPT)
             ENDDO

             DX(j)=XREFC(Ia+j)
             DY(j)=YREFC(Ia+j)
             DZ(j)=ZREFC(Ia+j)
             !            write(*,*)'ooo ',jrep,j,dx(ia),dy(ia),dz(ia)
          ENDIF
          ! end lonepair
       enddo
       if ( mynodg .eq. (numnod*irepdstr)) then
          ! this has to be protected! Not everyone can call this!!!
          call repdbr(xcord)
          call repdbr(ycord)
          call repdbr(zcord)
          call repdbr(xrefc)
          call repdbr(yrefc)
          call repdbr(zrefc)
       endif
       !        see the comments about the efficiency before the psnd8 calls above
       call psnd8(xcord,nalloc)
       call psnd8(ycord,nalloc)
       call psnd8(zcord,nalloc)
       call psnd8(xrefc,nalloc)
       call psnd8(yrefc,nalloc)
       call psnd8(zrefc,nalloc)
    endif
#endif 

    QPRINT=(PRNLEV.GE.6)
    ! rxncons_4f chmalloc
    ! bug??? this maybe is nrepl???      
    !              call chmalloc('rxcons_4.src','RXNCONS_4F','ARMS',NRXNCS,crl=ARMS)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ARMS',NREPL,crl=ARMS)
    !
    call chmalloc('rxcons_4.src','RXNCONS_4F','WEIGHT',NATOM,crl=weight)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ISTREP',NREPL,intg=istrep)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ICNREP',NREPL,intg=icnrep)
    !
    CALL EPATHC(ISTREP,ICNREP,WEIGHT, &
         NATREP,WTOT,QCWEIG,QCWCOMP, &
#if KEY_COMP2==1
         QCWCOMP2,  &                   
#endif
         QCMASS,QPRINT,ERR)

#if KEY_REPDSTR==1
    if(qrepdstr) nrepl=nrepdstr+1              
#endif

    ARMS(1:NREPL)=zero
    !
    NALLOC=NRXATM*(NREPL-1)

    call chmalloc('rxcons_4.src','RXNCONS_4F','XTAN',NREPL-1,NRXATM,crl=XTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4F','YTAN',NREPL-1,NRXATM,crl=YTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ZTAN',NREPL-1,NRXATM,crl=ZTAN)
    call chmalloc('rxcons_4.src','RXNCONS_4F','XM1',NREPL-1,NRXATM,crl=XM1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','YM1',NREPL-1,NRXATM,crl=YM1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ZM1',NREPL-1,NRXATM,crl=ZM1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','XP1',NREPL-1,NRXATM,crl=XP1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','YP1',NREPL-1,NRXATM,crl=YP1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ZP1',NREPL-1,NRXATM,crl=ZP1)
    call chmalloc('rxcons_4.src','RXNCONS_4F','ATMPR',2*NRXATM,intg=ATMPR)
    call chmalloc('rxcons_4.src','RXNCONS_4F','RA',3*NRXATM,crl=RA)
    call chmalloc('rxcons_4.src','RXNCONS_4F','RB',3*NRXATM,crl=RB)
    call chmalloc('rxcons_4.src','RXNCONS_4F','D_TAN',NREPL,crl=D_TAN)

    CALL RXNCONS_4F2(NRXATM,NRXNCS,NREPL,NATREP,NPSG, &
         XCORD,YCORD,ZCORD, &
         XREFC,YREFC,ZREFC, &
         WMASS,ATMASS, &
         XTAN,YTAN,ZTAN, &
         XM1,YM1,ZM1, &
         XP1,YP1,ZP1, &
         ATMPR,RA,RB, &
         QCNOTRN,QCNOROT,QPTCSCL, &
         ISTREP,ICNREP,ARMS, &
         DLTAN,DLCRV,D_TAN,WTOT, &
         FDOTTAN,FOFFP,PDLENG)

    !     repdstr needs to distribute this one
    !
#if KEY_REPDSTR==1
    if(.not.qrepdstr)then                
#endif
       I=0
       DO JREP=1,NREP
          DO J=1,NRXATM
             IA=NRXATM*(JREP-1)+IRXATM(J)
             I=I+1
             IF(IMOVE(IA).NE.-1)THEN
                DX(IA)=XREFC(I)
                DY(IA)=YREFC(I)
                DZ(IA)=ZREFC(I)
             ELSE
                ILP=LPRID(IA)
                N=LPNHOST(ILP)
                IPT=LPHPTR(ILP)

                XREFC(I)=XREFC(I)-DX(IA)
                YREFC(I)=YREFC(I)-DY(IA)
                ZREFC(I)=ZREFC(I)-DZ(IA)

                DO K=1,N
                   JPT=LPHOST(IPT+K)
                   XREFC(I)=XREFC(I)+DX(JPT)
                   YREFC(I)=YREFC(I)+DY(JPT)
                   ZREFC(I)=ZREFC(I)+DZ(JPT)
                ENDDO

                DO K=1,N
                   JPT=LPHOST(IPT+K)
                   DX(JPT)=DX(JPT)+XREFC(I)
                   DY(JPT)=DY(JPT)+YREFC(I)
                   DZ(JPT)=DZ(JPT)+ZREFC(I)
                ENDDO

                DX(IA)=ZERO
                DY(IA)=ZERO
                DZ(IA)=ZERO
             ENDIF
          ENDDO
       ENDDO
#if KEY_REPDSTR==1
    ELSE
       I=natrep*irepdstr
       DO IA=1,NRXATM
          I=I+1
          IF(IMOVE(IA).NE.-1)THEN
             DX(IA)=XREFC(I)
             DY(IA)=YREFC(I)
             DZ(IA)=ZREFC(I)
          ELSE
             ILP=LPRID(IA)
             N=LPNHOST(ILP)
             IPT=LPHPTR(ILP)
             XREFC(I)=XREFC(I)-DX(IA)
             YREFC(I)=YREFC(I)-DY(IA)
             ZREFC(I)=ZREFC(I)-DZ(IA)
             DO K=1,N
                JPT=LPHOST(IPT+K)
                XREFC(I)=XREFC(I)+DX(JPT)
                YREFC(I)=YREFC(I)+DY(JPT)
                ZREFC(I)=ZREFC(I)+DZ(JPT)
             ENDDO
             DO K=1,N
                JPT=LPHOST(IPT+K)
                DX(JPT)=DX(JPT)+XREFC(I)
                DY(JPT)=DY(JPT)+YREFC(I)
                DZ(JPT)=DZ(JPT)+ZREFC(I)
             ENDDO
             DX(IA)=ZERO
             DY(IA)=ZERO
             DZ(IA)=ZERO
          ENDIF
       ENDDO
    ENDIF
#endif 

    ! rxncons_4f chmdealloc
    call chmdealloc('rxcons_4.src','RXNCONS_4F','XTAN',NREPL-1,NRXATM,crl=XTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','YTAN',NREPL-1,NRXATM,crl=YTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ZTAN',NREPL-1,NRXATM,crl=ZTAN)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','XM1',NREPL-1,NRXATM,crl=XM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','YM1',NREPL-1,NRXATM,crl=YM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ZM1',NREPL-1,NRXATM,crl=ZM1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','XP1',NREPL-1,NRXATM,crl=XP1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','YP1',NREPL-1,NRXATM,crl=YP1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ZP1',NREPL-1,NRXATM,crl=ZP1)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ATMPR',2*NRXATM,intg=ATMPR)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','RA',3*NRXATM,crl=RA)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','RB',3*NRXATM,crl=RB)
    !BUG??? nrepl?      call chmdealloc('rxcons_4.src','RXNCONS_4F','ARMS',NRXNCS,crl=ARMS)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ARMS',NREPL,crl=ARMS)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','WEIGHT',NATOM,crl=weight)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ISTREP',NREPL,intg=istrep)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','ICNREP',NREPL,intg=icnrep)
    call chmdealloc('rxcons_4.src','RXNCONS_4F','D_TAN',NREPL,crl=D_TAN)

    RETURN
  END SUBROUTINE RXNCONS_4F

  SUBROUTINE RXNCONS_4F2(NRXATM,NRXNCS,NREPL,NATREP,NPSG, &
       XCORD,YCORD,ZCORD, &
       XREFC,YREFC,ZREFC, &
       WMASS,ATMASS, &
       XTAN,YTAN,ZTAN, &
       XM1,YM1,ZM1, &
       XP1,YP1,ZP1, &
       ATMPR,RA,RB, &
       QCNOTRN,QCNOROT,QCYCL, &
       ISTREP,ICNREP,ARMS,DLTAN,DLCRV,D_TAN,WTOT, &
       FDOTTAN,FOFFP,PDLENG)
    use stream
    use number
    ! . Parsed variables
    !
    implicit none
    INTEGER NRXATM,NRXNCS,NATREP,NREPL,NPSG
    real(chm_real) XCORD(*),YCORD(*),ZCORD(*)
    real(chm_real) XREFC(*),YREFC(*),ZREFC(*)
    real(chm_real) XTAN(NREPL-1,*),YTAN(NREPL-1,*),ZTAN(NREPL-1,*)
    real(chm_real) XM1(NREPL-1,*),YM1(NREPL-1,*),ZM1(NREPL-1,*)
    real(chm_real) XP1(NREPL-1,*),YP1(NREPL-1,*),ZP1(NREPL-1,*)
    INTEGER ISTREP(*),ICNREP(*)

    INTEGER ATMPR(2,NRXATM)
    real(chm_real) RA(3,NRXATM),RB(3,NRXATM)
    real(chm_real) WMASS(*),ATMASS(*),ARMS(*),WTOT
    real(chm_real) FDOTTAN(*),FOFFP(*),PDLENG
    LOGICAL QCNOTRN,QCNOROT,QCYCL

    !

    ! . Local variables
    !
    INTEGER NREP,IREP,JREP,NALL,KREP,ITER
    INTEGER IPT,JPT,I,J,IA,IBEGIN,ISTOP
    real(chm_real) RMST,TEMP,SUM,SUM1,SUM2,ATMP,FRMS,AERR
    real(chm_real) ERR,DEVA(3,3),EVWID,FACT,FACT1
    real(chm_real) TEMPX,TEMPY,TEMPZ,DLTAN(*),DLCRV(*),D_TAN(*)

    temp = zero ! local & not set, not sure of intent here

    FDOTTAN(1:NREPL)=zero
    FOFFP(1:NREPL)=zero
    NREP=NREPL-1

    CALL GETTAN(NREP,NPSG,NRXNCS,NRXATM, &
         XCORD,YCORD,ZCORD, &
         WMASS,ATMASS,ATMPR,RA,RB,DEVA, &
         XTAN,YTAN,ZTAN,XM1,YM1,ZM1,XP1,YP1,ZP1, &
         ARMS,RMST,QCNOTRN,QCNOROT,QCYCL,DLTAN,DLCRV,D_TAN)


    PDLENG=RMST

    SUM=ZERO
    DO I=1,NPSG
       ARMS(I)=ARMS(I)-RMST
       SUM=SUM+ARMS(I)*ARMS(I)
    ENDDO
    AERR=SQRT(SUM*TEMP)
    !      write(*,*)' hi force ',rmst,aerr,WTOT

    DO JREP=1,NREP
       !         write(50+jrep-1,'(a,5i5)')'RXNCONS4F>jrep,nrep,nrxatm=',
       !     $   jrep,nrep,nrxatm
       SUM=ZERO
       SUM1=ZERO
       SUM2=ZERO
       IA=(JREP-1)*NRXATM
       DO I=1,NRXATM
          IA=IA+1
          SUM=SUM+XTAN(JREP,I)*XREFC(IA)+ &
               YTAN(JREP,I)*YREFC(IA)+ &
               ZTAN(JREP,I)*ZREFC(IA)

          SUM1=SUM1+XREFC(IA)*XREFC(IA)+ &
               YREFC(IA)*YREFC(IA)+ &
               ZREFC(IA)*ZREFC(IA)

       ENDDO

       FDOTTAN(JREP)=SUM
       FOFFP(JREP)=SQRT(SUM1-SUM*SUM)

       IA=(JREP-1)*NRXATM
       DO I=1,NRXATM
          IA=IA+1
          XREFC(IA)=XREFC(IA)-SUM*XTAN(JREP,I)
          YREFC(IA)=YREFC(IA)-SUM*YTAN(JREP,I)
          ZREFC(IA)=ZREFC(IA)-SUM*ZTAN(JREP,I)
       ENDDO
    ENDDO

    !      DO JREP=1,NREP
    !        SUM=ZERO
    !        IA=(JREP-1)*NRXATM
    !        DO I=1,NRXATM
    !          IA=IA+1
    !          SUM=SUM+XM1(JREP,I)*XREFC(IA)+
    !     $            YM1(JREP,I)*YREFC(IA)+
    !     $            ZM1(JREP,I)*ZREFC(IA)
    !        ENDDO
    !
    !        IA=(JREP-1)*NRXATM
    !        DO I=1,NRXATM
    !          IA=IA+1
    !          XREFC(IA)=XREFC(IA)-SUM*XM1(JREP,I)
    !          YREFC(IA)=YREFC(IA)-SUM*YM1(JREP,I)
    !          ZREFC(IA)=ZREFC(IA)-SUM*ZM1(JREP,I)
    !        ENDDO
    !      ENDDO

100 FORMAT('RX_4_SETUP4> ','  ITER  ','   AVG RMSD   ', &
         '   AVG ERROR   ')
200 FORMAT('RX_4_SETUP4> ',I5,3F16.8)
300 FORMAT('RX_4_SETUP4> ',I5,4F16.8)
    RETURN
  END SUBROUTINE RXNCONS_4F2

  SUBROUTINE PPFORCE(NREPL,PDLENG,FDOTTAN,FOFFP,DLTAN,DLCRV)
    use stream
    use number
    use memory
    use parallel
    use repdstr
#if KEY_REPDSTR==1
    use repdstr 
#endif

    integer :: ierr

    real(chm_real) PDLENG,FDOTTAN(*),FOFFP(*),DLTAN(*),DLCRV(*),SUM
    real(chm_real) FDOTTMP1,FDOTTMP2

    INTEGER NREPL

    INTEGER I,NREP
    real(chm_real),allocatable,dimension(:) :: repdft

    NREP=NREPL-1

#if KEY_REPDSTR==1
    IF(QREPDSTR) THEN
       CALL CHMALLOC('rxcons_4.src','PPFORCE','REPDFT',NREPDSTR,CRL=REPDFT)
       repdft = zero
       if(mynod==0)repdft(irepdstr+1)=fdottan(irepdstr+1)
       call psetglob
       call gcomb(repdft,nrepdstr)
       call psetloc
    ENDIF
#endif 
#if KEY_PARALLEL==1
    IF(MYNOD.EQ.0)THEN
#endif 
       WRITE(OUTU,100)PDLENG
       WRITE(OUTU,199)

       SUM=ZERO

       FDOTTMP1=FDOTTAN(1)
       IF(QREPDSTR)FDOTTMP1=REPDFT(1)
       WRITE(OUTU,200)1,FDOTTMP1,FOFFP(1),SUM,DLTAN(1),DLCRV(1)

       DO I=2,NREP
          FDOTTMP1=FDOTTAN(I-1)
          FDOTTMP2=FDOTTAN(I)
#if KEY_REPDSTR==1
          IF(QREPDSTR)FDOTTMP1=REPDFT(I-1)                      
#endif
#if KEY_REPDSTR==1
          IF(QREPDSTR)FDOTTMP2=REPDFT(I)                        
#endif
          SUM=SUM+HALF*(FDOTTMP1*DLTAN(I-1)+FDOTTMP2*DLTAN(I))
          WRITE(OUTU,200)I,FDOTTMP2,FOFFP(I),SUM,DLTAN(I),DLCRV(I)
       ENDDO
#if KEY_PARALLEL==1
    ENDIF
#endif 

#if KEY_REPDSTR==1
    IF(QREPDSTR)CALL CHMDEALLOC('rxcons_4.src','PPFORCE','REPDFT',NREPDSTR,CRL=REPDFT)  
#endif

100 FORMAT(' PPFORCE> AVERAGE PATH LENG: ',F16.6)
199 FORMAT(' PPFORCE> REPLICA    F DOT TANGENT ', &
         ' ACC F DOT DL        DLTAN ')
200 FORMAT(' PPFORCE> ',I5, 5F12.4)
    RETURN
  END SUBROUTINE PPFORCE


#else /* (rxcons_4)*/

  SUBROUTINE RX_4_SETUP(COMLYN,COMLEN)
    INTEGER COMLEN
    CHARACTER(len=*) COMLYN
    return
  end SUBROUTINE RX_4_SETUP

#endif /* (rxcons_4)*/

end module rxcons4

