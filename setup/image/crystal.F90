SUBROUTINE CRYSTL
  !-----------------------------------------------------------------------
  !     Here the CRYSTAL options available in CHARMM are processed.
  !
  !     Author : Martin J. Field.
  !     Date   : April 1986.
  !              Fully revised December 1986.
  !              Updated for CHARMM 22 October 1990.
  !
  !     Input variables:
  !
  !     COMAND           - The command to be processed.
  !     COMLYN,COMLEN    - The command line and its length.
  !     IUNIT            - The unit for Crystal file I/O operations.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use comand
  use coord
  use ctitla
  use energym
  use memory
  use image
  use image_routines_module, only: inimag, reimag
  use psf
  use select
  use stream
  use string
  use prssre
  use xtlfrq_m
#if KEY_STRINGM==1 /*   VO string method */
  use machio, only: ifreeu
  use multicom_aux
#endif
#ifdef KEY_RESIZE
  use resize, only: set_rszf
#endif  
 
  implicit none
  !
  !     Local variables.
  !
  integer,allocatable,dimension(:) :: ISLCT
  real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
  real(chm_real),allocatable,dimension(:) :: DDFTMP
  CHARACTER(len=4) WORD
  INTEGER I,IUNIT
  LOGICAL QERROR,QXGRP
  !
#if KEY_STRINGM==1 /*  VO stringm v */
  integer :: oldiol
  logical :: qstr
  common /replicaio/ qstr
#endif
  !
  !
  !     Determine the command requested.
  !
  WORD = NEXTA4(COMLYN,COMLEN)
  !
  !     Get the unit number of a file.
  !
  IUNIT = GTRMI(COMLYN,COMLEN,'UNIT',-1)
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
  !=======================================================================
  IF (WORD .EQ. 'BUIL') THEN
     call chmalloc('crystal.src','CRYSTL','ISLCT',NATOM,intg=ISLCT)
     CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
     IF(INDXA(COMLYN,COMLEN,'NAMI').GT.0) QXTLNAMETYPE=.true.
     IF(INDXA(COMLYN,COMLEN,'COO1').GT.0) QXTLNAMETYPE=.false.
     CALL INIMAG(BIMAG,.TRUE.)
     CALL REIMAG(BIMAG,0,0)
     CALL XBUILD
     call chmalloc('crystal.src','CRYSTL','TRANSF',3,4,XNSYMM,crl=TRANSF)
     CALL XBUILD2(NATOM,X,Y,Z,ISLCT,TRANSF)
     call chmdealloc('crystal.src','CRYSTL','ISLCT',NATOM,intg=ISLCT)
     call chmdealloc('crystal.src','CRYSTL','TRANSF',3,4,XNSYMM,crl=TRANSF)
#ifdef KEY_RESIZE
     call set_rszf('crystal.F90','CRYSTL',.true.)
#endif     
     CALL GETVOL(EPROP(VOLUME))
     CALL XTLMSR(XUCELL)
     !=======================================================================
  ELSE IF (WORD .EQ. 'DEFI') THEN
     XTLTYP = NEXTA4(COMLYN,COMLEN)
     DO I = 1,6
        XUCELL(I) = NEXTF(COMLYN,COMLEN)
        XTLREF(I) = XUCELL(I)
     ENDDO
     IF(CUTXTL.EQ.FMARK) THEN
        CUTXTL=MAX(XUCELL(1),XUCELL(2),XUCELL(3)) ! bug#92268cjrs04
        IF(CUTXTL.GT.THIRTY) CUTXTL=THIRTY
        IF(CUTXTL.LT.TEN) CUTXTL=TEN
     ENDIF
     CALL XTLAXS(XTLABC,XUCELL)
     CALL XTLSYM(XTLABC,XUCELL,XTLTYP,XDIM,XTLREF)
     CALL XTLMSR(XUCELL)
     IF(PRNLEV.GE.2) CALL PRNXTLD(OUTU,'    ',XTLTYP,XUCELL, &
          .FALSE.,ZERO,.FALSE.,[ZERO])
     !       setup default value for IMGFRQ if it has not been set
     IF(IMGFRQ.LE.0) IMGFRQ=50
     !=======================================================================
     !       clear the crystal and image facility.
  ELSE IF (WORD .EQ. 'FREE') THEN
     CALL INCRYS
     CALL INIMAG(BIMAG,.TRUE.)
     CALL REIMAG(BIMAG,0,0)
     !=======================================================================
  ELSE IF (WORD .EQ. 'PHON') THEN
     IF(XNSYMM.NE.1) THEN
        CALL WRNDIE(-5,'<CRYSTL>','Must have XNSYMM=1 for phonons.')
     ENDIF
     IF(NPHONS.GT.0.AND.NKPTS.GT.0) THEN
        call chmdealloc('crystal.src','CRYSTL','XDDFP',(NPHONS*(NPHONS+1))/2,cmpx=XDDFP)
        call chmdealloc('crystal.src','CRYSTL','XFREQP',NKPTS*NPHONS,crl=XFREQP)
        call chmdealloc('crystal.src','CRYSTL','XEVALP',NKPTS*NPHONS,crl=XEVALP)
     call chmdealloc('crystal.src','CRYSTL','XEVECP',NKPTS*NPHONS*NPHONS,cmpx=XEVECP)
     ENDIF
     NKPTS=GTRMI(COMLYN,COMLEN,'NKPO',1)
     IF(NKPTS.LE.0) THEN
        CALL WRNDIE(-1,'<CRYSTL>','NKPOINTS invalid. Reset to 1.')
        NKPTS=1
     ENDIF
     KSTART(1:3) = ZERO
     KSTEP (1:3) = ZERO
     IF (INDXA(COMLYN,COMLEN,'KVEC').GT.0) THEN
        DO I=1,3
           KSTART(I)=NEXTF(COMLYN,COMLEN)
        ENDDO
        IF(INDXA(COMLYN,COMLEN,'TO').GT.0) THEN
           DO I=1,3
              KSTEP(I)=NEXTF(COMLYN,COMLEN)
           ENDDO
        ENDIF
        IF(NKPTS.GT.1) THEN
           DO I=1,3
              KSTEP(I)=(KSTEP(I)-KSTART(I))/(NKPTS-1)
           ENDDO
        ENDIF
     ELSE
        CALL WRNDIE(-1,'<CRYSTL>','No k-vectors given. Set to zero.')
        NKPTS=1
     ENDIF
     NPHONS=3*NATOM
     call chmalloc('crystal.src','CRYSTL','DDFTMP',(NPHONS*(NPHONS+1))/2,crl=DDFTMP)
     call chmalloc('crystal.src','CRYSTL','XDDFP',(NPHONS*(NPHONS+1))/2,cmpx=XDDFP)
     call chmalloc('crystal.src','CRYSTL','XFREQP',NKPTS*NPHONS,crl=XFREQP)
     call chmalloc('crystal.src','CRYSTL','XEVALP',NKPTS*NPHONS,crl=XEVALP)

     call chmalloc('crystal.src','CRYSTL','XEVECP',NKPTS*NPHONS*NPHONS,cmpx=XEVECP)

     ddftmp = zero
     CALL PHONON(X,Y,Z,NKPTS,KSTART,KSTEP,NPHONS,XDDFP, &
          DDFTMP,XFREQP,XEVALP, &
          XEVECP,.TRUE.)
     call chmdealloc('crystal.src','CRYSTL','DDFTMP',(NPHONS*(NPHONS+1))/2,crl=DDFTMP)
     !=======================================================================
  ELSE IF (WORD .EQ. 'PRIN') THEN
     !       Dispersion curves.
     IF (INDXA(COMLYN,COMLEN,'PHON').GT.0) THEN
        IF(PRNLEV.GE.2) CALL PRNTDC(COMLYN,COMLEN,OUTU)
        !         Crystal file.
     ELSE
        IF(PRNLEV.GE.2) &
             CALL XFWRIT(OUTU,TITLEA,NTITLA,MAXSYM,NTRANS, &
             XNSYMM,XSYMOP,XNOP,XNA,XNB,XNC,.TRUE.)
     ENDIF
     !=======================================================================
  ELSE IF (WORD .EQ. 'READ') THEN
     IF (INDXA(COMLYN,COMLEN,'CARD') .LE. 0) CALL WRNDIE &
          (-5,'<CRYSTL>','Crystal file card read only allowed.')
     IF (IUNIT .EQ. -1 .OR. IUNIT .EQ. ISTRM) THEN
        IUNIT = ISTRM
        IF(PRNLEV.GE.2) WRITE (OUTU, &
             '('' CRYSTAL FILE BEING READ FROM INPUT STREAM'')')
     ELSE
        IF(PRNLEV.GE.2) WRITE (OUTU, &
             '('' CRYSTAL FILE BEING READ FROM UNIT '',i5)') IUNIT
     ENDIF
     CALL INIMAG(BIMAG,.TRUE.)
     CALL REIMAG(BIMAG,0,0)
     !RCZ 92/06/18 (?) CALL XTLSYM(XTLABC,XUCELL,XTLTYP,XDIM,XTLREF)
     CALL XFREAD(IUNIT,OUTU,NTRANS,XNSYMM,XSYMOP,XNOP,XNA,XNB,XNC)
#ifdef KEY_RESIZE
     call set_rszf('crystal.F90','CRYSTL',.true.)
#endif     
     call chmalloc('crystal.src','CRYSTL','TRANSF',3,4,XNSYMM,crl=TRANSF)
     CALL IMFILL(TRANSF,.TRUE.)
     call chmdealloc('crystal.src','CRYSTL','TRANSF',3,4,XNSYMM,crl=TRANSF)
     !=======================================================================
  ELSE IF (WORD .EQ. 'VIBR') THEN
     QXGRP = INDXA(COMLYN,COMLEN,'GROUP').NE.0
     IF(XNSYMM.NE.1) &
          CALL WRNDIE(-5,'<CRYSTL>','Must have XNSYMM=1 for vibrations.')
     IF(NFREQX.GT.0) THEN
        call chmdealloc('crystal.src','CRYSTL','XDDF',(NFREQX*(NFREQX+1))/2,crl=XDDF)
        call chmdealloc('crystal.src','CRYSTL','XFREQ',NFREQX,crl=XFREQ)
        call chmdealloc('crystal.src','CRYSTL','XEVAL',NFREQX,crl=XEVAL)
        call chmdealloc('crystal.src','CRYSTL','XEVEC',NFREQX*NFREQX,crl=XEVEC)
     ENDIF
     NFREQX=3*NATOM
     call chmalloc('crystal.src','CRYSTL','XDDF',(NFREQX*(NFREQX+1))/2,crl=XDDF)
     call chmalloc('crystal.src','CRYSTL','XFREQ',NFREQX,crl=XFREQ)
     call chmalloc('crystal.src','CRYSTL','XEVAL',NFREQX,crl=XEVAL)
     call chmalloc('crystal.src','CRYSTL','XEVEC',NFREQX*NFREQX,crl=XEVEC)
     xddf = zero
     CALL XTLFRQ(X,Y,Z,NFREQX,XDDF,XFREQ,XEVAL, &
          XEVEC,.TRUE.,QXGRP)
     !=======================================================================
  ELSE IF (WORD .EQ. 'WRIT') THEN
     IF (IUNIT .EQ. -1) THEN
        CALL WRNDIE(-5,'<CRYSTL>','No I/O unit for write specified.')
     ENDIF
     CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
     !       Dispersion curves.
     IF (INDXA(COMLYN,COMLEN,'PHON').GT.0) THEN
        IF(NKPTS.LE.0.OR.NPHONS.LE.0) THEN
           CALL WRNDIE(-1,'<CRYSTL>','No phonons to write out.')
        ENDIF
        IF(IOLEV.GT.0) CALL WRITDC(IUNIT,NATOM,NKPTS,NPHONS,AMASS, &
             XEVALP,XEVECP)
        !         Vibrations.
     ELSE IF (INDXA(COMLYN,COMLEN,'VIBR').GT.0) THEN
        IF(IOLEV.GT.0) CALL XFRQWR(COMLYN,COMLEN,IUNIT)
        !         Crystal file.
     ELSE
        IF (INDXA(COMLYN,COMLEN,'CARD') .LE. 0) CALL WRNDIE &
             (-5,'<CRYSTL>','Crystal file card read only allowed.')
        IF(IOLEV.GT.0) THEN
           CALL XFWRIT(IUNIT,TITLEA,NTITLA,MAXSYM,NTRANS, &
                XNSYMM,XSYMOP,XNOP,XNA,XNB,XNC,.FALSE.)
           CALL VCLOSE(IUNIT,'KEEP',QERROR)
        ENDIF
     ENDIF
     !=======================================================================
  ELSE
     CALL WRNDIE(-1,'<CRYSTL>','Unrecognised CRYSTAL option.')
  ENDIF
  !
#if KEY_STRINGM==1 /*  VO : restore iolev */
  if (qstr) iolev=oldiol 
#endif
  !
  RETURN
END SUBROUTINE CRYSTL

SUBROUTINE IMFILL(TRANSF,QFIRST)
  !-----------------------------------------------------------------------
  !     This subroutine takes the information in the CRYSTAL data
  !     structure and fills the IMAGE common blocks with it.
  !
  use chm_kinds
  use dimens_fcm
  use vector
  use image
  use stream
  implicit none
  !
  LOGICAL   QFIRST
  real(chm_real)    TRANSF(3,4,*)
  !
  CHARACTER(len=4) WORD4
  CHARACTER(len=8) WORD8
  LOGICAL ERROR
  INTEGER   I,IPT,ITRAN,J,JPT,JTRAN,K
  real(chm_real)    DET,T(3),U(3,3),XABC(3), XTLINV(6)
  !
  real(chm_real), PARAMETER :: ZERO = 0.D0, TOL = 1.D-4, ONE = 1.D0
  !
  !     Perform some checks on the input symmetry operations.
  !
  IF (QFIRST) THEN
     DO ITRAN = 1,(XNSYMM-1)
        loop100: DO JTRAN = (ITRAN+1),XNSYMM
           DO I = 1,4
              DO J = 1,3
                 IF (XSYMOP(J,I,JTRAN) .NE. XSYMOP(J,I,ITRAN)) cycle loop100
              ENDDO
           ENDDO
           IF(WRNLEV.GE.2) WRITE (OUTU,'(A,2I5)') &
                ' IMFILL> WARNING: Two symmetry operations are ' &
                //'identical : ', &
                ITRAN,JTRAN
           CALL WRNDIE(-3,'<IMFILL>','Two symmetry operations are ' &
                //'identical')
        ENDDO loop100
     ENDDO
  ENDIF
  !
  !     Construct the transformation matrices from the symmetry operations
  !
  CALL MBUILD(XNSYMM,MAXSYM,XSYMOP,XTLABC,TRANSF,XTLINV)
  !
  !     Now build the image transformations.
  !
  IPT = 0
  DO I = 1,NTRANS
     ITRAN = XNOP(I)
     XABC(1) = XNA(I)
     XABC(2) = XNB(I)
     XABC(3) = XNC(I)
     DO J = 1,3
        DO K = 1,3
           IPT = IPT + 1
           IMTRNS(IPT) = TRANSF(J,K,ITRAN)
        ENDDO
     ENDDO
     !RCZ  DO (J = 1,3) T(J) = TRANSF(J,4,ITRAN)
     !RCZ  DO (J = 1,3)
     !RCZ  DO (K = 1,3) T(K) = T(K) + XABC(J) * XTLABC(K,J)
     !RCZ  FIN
     T(1)=TRANSF(1,4,ITRAN) &
          + XABC(1)*XTLABC(1)+XABC(2)*XTLABC(2)+XABC(3)*XTLABC(4)
     T(2)=TRANSF(2,4,ITRAN) &
          + XABC(1)*XTLABC(2)+XABC(2)*XTLABC(3)+XABC(3)*XTLABC(5)
     T(3)=TRANSF(3,4,ITRAN) &
          + XABC(1)*XTLABC(4)+XABC(2)*XTLABC(5)+XABC(3)*XTLABC(6)
     DO J = 1,3
        IPT = IPT + 1
        IMTRNS(IPT) = T(J)
     ENDDO
  enddo
  !
  IF (QFIRST) THEN
     !
     !       Determine the image names.
     !
     ERROR=.false.
     IF(QXTLNAMETYPE) THEN

 !  Crystal code requires extended format for 8 character image transformation names.
        if(idleng<8) then
           IF(PRNLEV.GE.2) WRITE (OUTU,'(20X,A)') &
             ' CRYSTAL: CHANGING TO EXTENDED FORMAT TO HANDLE LONGER IMAGE TRANSFORMATION NAMES.'
           idleng=8 
        endif

        DO I = 1,NTRANS
         ! new format
           WORD8=' '
           J=XNA(I)
           IF(J.EQ.0) THEN
              WORD8(1:1)='Z'
           ELSEIF(J.GT.0) THEN
              WORD8(1:1)='P'
           ELSE
              WORD8(1:1)='N'
           ENDIF
           J=IABS(XNA(I))
           IF(J.GT.9) ERROR=.true.
           WRITE(WORD4,'(I4)') J
           WORD8(2:2)=WORD4(4:4)
           
           J=XNB(I)
           IF(J.EQ.0) THEN
              WORD8(3:3)='Z'
           ELSEIF(J.GT.0) THEN
              WORD8(3:3)='P'
           ELSE
              WORD8(3:3)='N'
           ENDIF
           J=IABS(XNB(I))
           IF(J.GT.9) ERROR=.true.
           WRITE(WORD4,'(I4)') J
           WORD8(4:4)=WORD4(4:4)
           
           J=XNC(I)
           IF(J.EQ.0) THEN
              WORD8(5:5)='Z'
           ELSEIF(J.GT.0) THEN
              WORD8(5:5)='P'
           ELSE
              WORD8(5:5)='N'
           ENDIF
           J=IABS(XNC(I))
           IF(J.GT.9) ERROR=.true.
           WRITE(WORD4,'(I4)') J
           WORD8(6:6)=WORD4(4:4)

           WORD8(7:7)='R'
           J=XNOP(I)
           IF(J.GT.9) ERROR=.true.
           WRITE(WORD4,'(I4)') J
           WORD8(8:8)=WORD4(4:4)
 
           IMNAME(I) = WORD8          
        ENDDO 
     ENDIF      

     IF(ERROR .AND. WRNLEV.GE.2) WRITE (OUTU,'(A)') &
        'Cannot use new crystal names. Too many transformations.'

     IF(ERROR .OR. .NOT.QXTLNAMETYPE) THEN
       ! old naming for images
        DO I = 1,NTRANS
           WRITE(WORD4,'(I4)') I
           IF (I .LT.10) WORD4(1:3) = 'C00'
           IF (I .LE.99) WORD4(1:2) = 'C0'
           IF (I .GT.99) WORD4(1:1) = 'C'
           IMNAME(I) = WORD4
        ENDDO 
     ENDIF      

     !
     !       Check the operations' determinants and for NOROT.
     !
     NOROT = .TRUE.
     DO ITRAN = 1,XNSYMM
        CALL DETM33(TRANSF(1,1,ITRAN),DET)
        IF (ABS(ABS(DET) - ONE) .GT. TOL) THEN
           IF(WRNLEV.GE.2) WRITE (OUTU,'(/,A,I4,A,F14.8)') &
                ' IMFILL> WARNING: Non-unitary transformation matrix ', &
                ITRAN, &
                ' with determinant = ',DET
           CALL WRNDIE(-3,'<IMFILL>','Non-unitary transformation ' &
                //'matrix')
        ENDIF
        IF (DET .LT. ZERO) THEN
           IF(WRNLEV.GE.2) WRITE (OUTU,'(A,I4,2X,A)') &
                ' Symmetry operation ',ITRAN, &
                ' is a mirror reflection or inversion.'
        ENDIF
        !
        DO I = 1,3
           DO J = 1,3
              U(J,I) = TRANSF(J,I,ITRAN)
           ENDDO
           U(I,I) = U(I,I) - ONE
        ENDDO
        IF (DOTVEC(U,U,9) .GT. TOL) NOROT=.FALSE.
     enddo
     !
     IF (NOROT) THEN
        IF(PRNLEV.GE.2) WRITE (OUTU,'(20X,A)') &
             ' THERE ARE NO ROTATIONS FOR THIS TRANSFORMATION SET'
     ELSE
        IF(PRNLEV.GE.2) WRITE(OUTU,'(20X,A)') &
             ' THERE ARE ROTATIONS FOR THIS TRANSFORMATION SET'
     ENDIF
     !
     !       Determine the inverses of each transformation.
     !
     loop400: DO ITRAN = 1,NTRANS
        IPT = (ITRAN-1)*12
        DO JTRAN = 1,NTRANS
           JPT = (JTRAN-1)*12
           CALL MULMXN(T,IMTRNS(IPT+10),1,3,IMTRNS(JPT+1),3,3)
           CALL ADDVEC(T,IMTRNS(JPT+10),T,3)
           IF (DOTVEC(T,T,3) .LE. TOL) THEN
              CALL MULNXN(U,IMTRNS(IPT+1),IMTRNS(JPT+1),3)
              DO I = 1,3
                 U(I,I) = U(I,I) - ONE
              ENDDO
              IF (DOTVEC(U,U,9) .LE. TOL) THEN
                 IMINV(ITRAN) = JTRAN
                 cycle loop400
              ENDIF
           ENDIF
        ENDDO
        IF(WRNLEV.GE.2) WRITE (OUTU,'(/,A,I4,A)') &
             ' IMFILL> WARNING: Transformation ',ITRAN, &
             ' has no inverse.  Check Crystal Operations.'
        IMINV(ITRAN) = 0
        CALL WRNDIE(-3,'<IMFILL>','Transformation with no inverse')
     enddo loop400
     !
     IF(PRNLEV.GT.2) WRITE(OUTU,'(1X,I5,A,/)') NTRANS, &
          ' Transformations have been processed.'
  ENDIF
  !
  RETURN
END SUBROUTINE IMFILL

SUBROUTINE MBUILD(XNSYMM,MAXSYM,XSYMOP,XTLABC,TRANSF,XTLINV)
  !-----------------------------------------------------------------------
  !     The transformation matrices corresponding to the input crystal
  !     symmetry operations are created here.
  !
  use chm_kinds
  use stream
  implicit none
  !
  INTEGER   XSYMOP(3,4,*)
  INTEGER   MAXSYM,XNSYMM
  real(chm_real)    TRANSF(3,4,XNSYMM),WRK1(3,4),WRK2(3,3), &
       XTLABC(6),XTLINV(6)
  !
  INTEGER I,ITRANS,J
  LOGICAL OK
  real(chm_real)  RFACT
  !
  real(chm_real), PARAMETER :: ZERO = 0.D0, ONE = 1.D0
  !
  RFACT = ONE / MAXSYM
  CALL INVT33S(XTLINV,XTLABC,OK)
  IF (.NOT.OK) &
       CALL WRNDIE(-5,'<MBUILD>','Crystal metric matrix is singular.')
  !
  DO ITRANS = 1,XNSYMM
     DO I = 1,3
        DO J = 1,3
           WRK1(J,I) = XSYMOP(J,I,ITRANS)
        ENDDO
     ENDDO
     DO I = 1,3
        WRK1(I,4) = RFACT * XSYMOP(I,4,ITRANS)
     ENDDO
     !
     CALL MULNXNFL(WRK2,WRK1,XTLINV,3)
     CALL MULNXNLF(TRANSF(1,1,ITRANS),XTLABC,WRK2,3)
     !
     TRANSF(1,4,ITRANS) = &
          XTLABC(1)*WRK1(1,4)+XTLABC(2)*WRK1(2,4)+XTLABC(4)*WRK1(3,4)
     TRANSF(2,4,ITRANS) = &
          XTLABC(2)*WRK1(1,4)+XTLABC(3)*WRK1(2,4)+XTLABC(5)*WRK1(3,4)
     TRANSF(3,4,ITRANS) = &
          XTLABC(4)*WRK1(1,4)+XTLABC(5)*WRK1(2,4)+XTLABC(6)*WRK1(3,4)
  ENDDO
  IF (PRNLEV.GE.7) THEN
     DO ITRANS = 1,XNSYMM
        WRITE(OUTU, '(A,I6)' ) ' Transformation ', ITRANS
        WRITE(OUTU, '(3F16.5)' ) &
             (( TRANSF(J,I,ITRANS), J = 1,3 ), I = 1,4 )
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE MBUILD

SUBROUTINE PRNXTLD(OUTUX,PRNID,XTLTYP,XUCELL,LGRAD,GRAD, &
    QPRESX,EPRESX)
  !-----------------------------------------------------------------------
  !     Print out the crystal parameters during dynamics
  !
  use chm_kinds
  use stream
  use energym
  use prssre,only : QP21XCEN ! dynamic IMXCEN for P21 tetragonal lattice
  use image,only: IMXCEN
 implicit none
 !
 INTEGER     OUTUX
 real(chm_real)      XUCELL(6),GRAD
 CHARACTER(len=4) PRNID,XTLTYP
 LOGICAL     LGRAD,QPRESX
 real(chm_real) :: EPRESX(*)
 !
 IF(PRNLEV.LE.2) RETURN
 WRITE (OUTUX, '(2A)' ) &
      ' Crystal Parameters : Crystal Type = ', XTLTYP
 IF (QP21XCEN.and.(PRNID=='DYNA')) THEN
  WRITE (OUTUX,10) PRNID, &
      ' A     = ',XUCELL(1),' B    = ',XUCELL(2),' C     = ',XUCELL(3),' XCEN = ',IMXCEN
10 FORMAT(6X,A4,4(A,F10.5))
 ELSE
  WRITE (OUTUX,11) PRNID, &
      ' A     = ',XUCELL(1),' B    = ',XUCELL(2),' C     = ',XUCELL(3)
 ENDIF
 WRITE (OUTUX,11) PRNID, &
      ' Alpha = ',XUCELL(4),' Beta = ',XUCELL(5),' Gamma = ',XUCELL(6)
11 FORMAT(6X,A4,3(A,F10.5))
 !
 IF(QPRESX) THEN
    WRITE(OUTU,12) PRNID, &
         ' PIXX =  ',EPRESX(PIXX),' PIYY = ',EPRESX(PIYY), &
         ' PIZZ = ',EPRESX(PIZZ)
    WRITE(OUTU,12) PRNID, &
         ' PIXY =  ',EPRESX(PIXY),' PIXZ = ',EPRESX(PIXZ), &
         ' PIYZ = ',EPRESX(PIYZ)
 ENDIF
12 FORMAT(6X,A4,3(A,F10.2))
 !
 IF(LGRAD) WRITE (OUTUX,22) PRNID,GRAD
22 FORMAT(6X,A4,' Gradient Norm = ',F10.5,/)
 !
 !
 RETURN
END SUBROUTINE PRNXTLD

SUBROUTINE XBUILD
  !-----------------------------------------------------------------------
  !     This subroutine is parsing 'crystal build' command
  !     and reading symmetry cards.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use image
  use number
  use stream
  use string
#if KEY_DOMDEC==1
  use inbnd,only:cutnb  
#endif

 implicit none
 !
 !
 INTEGER     I,J
 LOGICAL     EOF,QERROR
 !
 !     Initialize Local Logicals
 !
 EOF = .FALSE.
 IF(XTLTYP.EQ.'    ') THEN
    CALL WRNDIE(-1,'<XBUILD>', &
         'No crystal type defined. Cannot build.')
    RETURN
 ENDIF
 !
 !     Process input options.
 !
 XNSYMM = GTRMI(COMLYN,COMLEN,'NOPE',0)
 CUTXTL = GTRMF(COMLYN,COMLEN,'CUTO',CUTXTL)
#if KEY_DOMDEC==1
 cutnb = cutxtl                               
#endif
 IF(PRNLEV.GE.5) WRITE(OUTU,45) CUTXTL
45 FORMAT( &
      ' XBUILD> Building all transformations with a minimum atom-atom', &
      /,'         contact distance of less than',F8.2,' Angstroms.')
 IF (XNSYMM .GE. MAXSYM) CALL WRNDIE(-5,'<XBUILD>', &
      'Too many symmetry operations specified.')
 CALL XTRANE(COMLYN,COMLEN,'XBUILD')
 !
 !     Read symmetry operations into XSYMOP. The first is the identity.
 !
 DO I = 1,4
    DO J = 1,3
       XSYMOP(J,I,1) = 0
    ENDDO
 ENDDO
 DO I = 1,3
    XSYMOP(I,I,1) = 1
 ENDDO
 !
 XNSYMM = XNSYMM + 1
 DO I = 2,XNSYMM
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE., &
         .TRUE.,'XBUILD> ')
    IF(EOF) THEN
       IF(WRNLEV.GE.2) WRITE (OUTU,'(A)') &
            ' XBUILD> WARNING : Exiting as EOF reached.'
       RETURN
    ENDIF
    CALL XSYMPA(COMLYN,COMLEN,MAXSYM,XSYMOP(1,1,I),QERROR)
    DO J = 1,COMLEN
       COMLYN(J:J) = ' '
    ENDDO
    IF (QERROR) &
         CALL WRNDIE(-5,'<XBUILD>','Symmetry operation parsing error.')
 ENDDO
 !
 RETURN
END SUBROUTINE XBUILD

SUBROUTINE XBUILD2(NATOM,X,Y,Z,ISLCT,TRANSF)
  !-----------------------------------------------------------------------
  !     This subroutine generates the CRYSTAL image file that is
  !     required to perform minimizations or vibrational analyses.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use image
  use number
  use memory
  use stream
 implicit none
 !
 real(chm_real),allocatable,dimension(:) :: XDIST
 real(chm_real),allocatable,dimension(:) :: x0, y0, z0
 real(chm_real),allocatable,dimension(:) :: x1, y1, z1
 INTEGER     ISLCT(*)
 INTEGER     NATOM
 real(chm_real)      X(*),Y(*),Z(*)
 !
 INTEGER     I,NUMSEL
 LOGICAL     OK
 !
 real(chm_real)      TRANSF(3,4,*),XTLINV(6)
 !
 !     Check the coordinates and allocate space for the crystal generatio
 !
 NUMSEL = 0
 OK=.TRUE.
 DO I = 1,NATOM
    IF (ISLCT(I) .EQ. 1) THEN
       IF (X(I) .GE. ANUM) THEN
          OK=.FALSE.
       ELSE
          NUMSEL = NUMSEL + 1
       ENDIF
    ENDIF
 ENDDO
 IF (NUMSEL .LE. 0) THEN
    CALL WRNDIE(0,'<XBUILD2>','No coordinates selected.')
    RETURN
 ENDIF
 IF(.NOT.OK) THEN
    CALL WRNDIE(0,'<XBUILD2>', &
         'Undefined coordinate in selected set.')
 ENDIF
#if KEY_DEBUG==1
 IF(PRNLEV.GE.2) WRITE(OUTU,'(I10,A,I10,A)') NUMSEL, &
      ' atoms out of ', NATOM,' selected.'
#endif 
 !
 call chmalloc('crystal.src','XBUILD2','XDIST',MAXTRN,crl=XDIST)
 call chmalloc('crystal.src','XBUILD2','X0',NUMSEL,crl=X0)
 call chmalloc('crystal.src','XBUILD2','Y0',NUMSEL,crl=Y0)
 call chmalloc('crystal.src','XBUILD2','Z0',NUMSEL,crl=Z0)
 call chmalloc('crystal.src','XBUILD2','X1',NUMSEL,crl=X1)
 call chmalloc('crystal.src','XBUILD2','Y1',NUMSEL,crl=Y1)
 call chmalloc('crystal.src','XBUILD2','Z1',NUMSEL,crl=Z1)
 !
 !     Compute the transformation matrices from the symmetry operations.
 !
#if KEY_DEBUG==1
 IF(PRNLEV.GE.2) THEN
    WRITE (OUTU,'(A)') ' Lattice Vectors :'
    WRITE (OUTU,'(A,3F16.10)') ' A = ',XTLABC(1),XTLABC(2),XTLABC(4)
    WRITE (OUTU,'(A,3F16.10)') ' B = ',XTLABC(2),XTLABC(3),XTLABC(5)
    WRITE (OUTU,'(A,3F16.10)') ' C = ',XTLABC(4),XTLABC(5),XTLABC(6)
 ENDIF
#endif 
 CALL MBUILD(XNSYMM,MAXSYM,XSYMOP,XTLABC,TRANSF,XTLINV)
 !
 !     Generate the crystal.
 !
 CALL XSCAN(CUTXTL,MAXTRN,NATOM,XNSYMM,ISLCT,X,Y,Z,TRANSF, &
      X0,Y0,Z0,XTLABC, &
      X1,Y1,Z1, &
      XDIST,XNOP,XNA,XNB,XNC,NTRANS)
 !
 call chmdealloc('crystal.src','XBUILD2','XDIST',MAXTRN,crl=XDIST)
 call chmdealloc('crystal.src','XBUILD2','X0',NUMSEL,crl=X0)
 call chmdealloc('crystal.src','XBUILD2','Y0',NUMSEL,crl=Y0)
 call chmdealloc('crystal.src','XBUILD2','Z0',NUMSEL,crl=Z0)
 call chmdealloc('crystal.src','XBUILD2','X1',NUMSEL,crl=X1)
 call chmdealloc('crystal.src','XBUILD2','Y1',NUMSEL,crl=Y1)
 call chmdealloc('crystal.src','XBUILD2','Z1',NUMSEL,crl=Z1)
 CALL IMFILL(TRANSF,.TRUE.)
 !
 RETURN
END SUBROUTINE XBUILD2

SUBROUTINE XFREAD(IUNIT,OUTU,NTRANS,XNSYMM,XSYMOP, &
    XNOP,XNA,XNB,XNC)
  !-----------------------------------------------------------------------
  !     This subroutine reads in the CRYSTAL image file. Only
  !     CARD format is allowed at present.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use comand
  use ctitla
  use string

  implicit none
  !
  INTEGER     XNOP(*),XNA(*),XNB(*),XNC(*),XSYMOP(3,4,*)
  INTEGER     IUNIT,NTRANS,OUTU,XNSYMM
  !
  CHARACTER(len=4) WORD
  INTEGER     I,ISYMM,ITRAN
  LOGICAL     DONE,EOF,QERROR,QIMAG,QSYMM
  !
  CALL TRYORO(IUNIT,'FORMATTED')
  CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
  CALL WRTITL(TITLEB,NTITLB,OUTU,1)
  !
  EOF   = .FALSE.
  QIMAG = .FALSE.
  QSYMM = .FALSE.
  !
1000 CONTINUE
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE., &
       .FALSE.,'XFREAD> ')
  IF (EOF) THEN
     CALL WRNDIE(0,'<XFREAD>','End-of-file in input.')
     RETURN
  ENDIF
  WORD = NEXTA4(COMLYN,COMLEN)
  CALL XTRANE(COMLYN,COMLEN,'XFREAD')
  !=======================================================================
  IF (WORD .EQ. 'IMAG') THEN
     DONE  = .FALSE.
     QIMAG = .TRUE.
     ITRAN = 0
1100 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE., &
          .FALSE.,'XFREAD> ')
     IF (EOF) THEN
        CALL WRNDIE(0,'<XFREAD>','End-of-file in input.')
        DONE = .TRUE.
     ENDIF
     WORD = NEXTA4(COMLYN,COMLEN)
     IF (WORD .EQ. '    ') THEN
     ELSE IF (WORD .EQ. 'END ') THEN
        DONE = .TRUE.
     ELSE
        ITRAN = ITRAN + 1
        I = 4
        XNOP(ITRAN) = NEXTI(WORD,I)
        XNA(ITRAN)  = NEXTI(COMLYN,COMLEN)
        XNB(ITRAN)  = NEXTI(COMLYN,COMLEN)
        XNC(ITRAN)  = NEXTI(COMLYN,COMLEN)
     ENDIF
     CALL XTRANE(COMLYN,COMLEN,'XFREAD')
     IF (.NOT.DONE) GOTO 1100
     NTRANS = ITRAN
     !=======================================================================
  ELSE IF (WORD .EQ. 'SYMM') THEN
     DONE  = .FALSE.
     QSYMM = .TRUE.
     ISYMM = 0
1200 CONTINUE
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE., &
          .FALSE.,'XFREAD> ')
     IF (EOF) THEN
        CALL WRNDIE(0,'<XFREAD>','End-of-file in input.')
        DONE = .TRUE.
     ENDIF
     I = INDEX(COMLYN,'(')
     IF (I .EQ. 0) THEN
        WORD = NEXTA4(COMLYN,COMLEN)
        IF (WORD .EQ. '    ') THEN
        ELSE IF (WORD .EQ. 'END ') THEN
           DONE = .TRUE.
        ELSE
           CALL WRNDIE(0,'<XFREAD>','Symmetry operation input error.')
        ENDIF
     ELSE
        ISYMM  = ISYMM + 1
        IF (ISYMM .GT. MAXSYM) &
             CALL WRNDIE(-5,'<XFREAD>', &
             'Too many symmetry operations input.')
        CALL XSYMPA(COMLYN,COMLEN,MAXSYM,XSYMOP(1,1,ISYMM),QERROR)
        DO I = 1,COMLEN
           COMLYN(I:I) = ' '
        ENDDO
        IF (QERROR) &
             CALL WRNDIE(-5,'<XFREAD>', &
             'Symmetry operation parsing error.')
     ENDIF
     IF (.NOT.DONE) GOTO 1200
     XNSYMM = ISYMM
     !=======================================================================
  ELSE IF (WORD .EQ. '    ') THEN
     !=======================================================================
  ELSE
     CALL WRNDIE(0,'<XFREAD>','Unrecognised command.')
  ENDIF
  IF (.NOT.(QIMAG .AND. QSYMM)) GOTO 1000
  !
  RETURN
  !
END SUBROUTINE XFREAD

SUBROUTINE XFWRIT(IUNIT,TITLE,NTITLE,MAXSYM,NTRANS,XNSYMM, &
     XSYMOP,XNOP,XNA,XNB,XNC,QPRINT)
  !-----------------------------------------------------------------------
  !     This subroutine writes the CRYSTAL image file in CARD format
  !     to a data file or prints it on the current output stream.
  !
  use chm_kinds
  use stream
  implicit none
  !
  CHARACTER(len=*) TITLE(*)
  INTEGER   XNOP(*),XNA(*),XNB(*),XNC(*),XSYMOP(3,4,*)
  INTEGER   IUNIT,MAXSYM,NTITLE,NTRANS,XNSYMM
  LOGICAL   QPRINT
  !
  !
  CHARACTER(len=40) CSYMOP
  INTEGER   I
  !
  IF(IOLEV.LT.0) RETURN
  !
  !     Write out title and symmetry operations.
  !
  IF (.NOT.QPRINT) CALL WRTITL(TITLE,NTITLE,IUNIT,0)
  WRITE (IUNIT,'(A)') ' SYMMETRY'
  DO I = 1,XNSYMM
     CALL XSYMWR(CSYMOP,XSYMOP(1,1,I),MAXSYM)
     WRITE (IUNIT,'(1X,A)') CSYMOP
  ENDDO
  WRITE (IUNIT,'(A,/)') ' END'
  !
  !     Write out images.
  !
  WRITE (IUNIT,'(A)') ' IMAGES'
  WRITE (IUNIT,'(A)') ' ! Operation     A     B     C'
  DO I = 1,NTRANS
     WRITE (IUNIT,'(6X,4(I6))') XNOP(I),XNA(I),XNB(I),XNC(I)
  ENDDO
  WRITE (IUNIT,'(A)') ' END'
  !
  RETURN
END SUBROUTINE XFWRIT

SUBROUTINE XSCAN(CUTOFF,MAXTRN,NATOM,XNSYMM,ISLCT,X,Y,Z, &
     TRANSF,X0,Y0,Z0,XTLABC,X1,Y1,Z1, &
     XDIST,XNOP,XNA,XNB,XNC,NTRANS)
  !-----------------------------------------------------------------------
  !     XSCAN searches the lattice for points within CUTOFF Angstroms
  !     of the primary atoms. The primary atoms are transformed to each
  !     equivalent position in turn and then a loop over the allowed
  !     lattice translations is performed. Only those lying within the
  !     cutoff are kept.
  !
  use chm_kinds
  use number
  use stream
  use bases_fcm
  !
  implicit none
  !
  INTEGER   MAXTRN,NATOM,NTRANS,XNSYMM
  INTEGER   ISLCT(*),XNA(*),XNB(*),XNC(*),XNOP(*)
  !RCZ 92/06/11 - get rid of XORDER
  !     INTEGER   ISLCT(*),XNA(*),XNB(*),XNC(*),XNOP(*),XORDER(MAXTRN)
  real(chm_real)    CUTOFF,TRANSF(3,4,*),XDIST(*),XTLABC(6), &
       X(*),Y(*),Z(*),X0(*),Y0(*),Z0(*),X1(*),Y1(*),Z1(*)
  !
  INTEGER   NUMSEL
  INTEGER   NTRANS0
  INTEGER   IA,IB,IC,ITRAN,XNA1,XNA2,XNB1,XNB2,XNC1,XNC2
  INTEGER   I,J,K,II,JJ,KK,IAT,JAT,I2
  LOGICAL   OK,QSKIP
  real(chm_real)    CUTOF2,R(3,3),S2,S2MIN,T(3), &
       ALOW,AHIGH,BLOW,BHIGH,CLOW,CHIGH, &
       AMIN,AMAX,BMIN,BMAX,CMIN,CMAX,AINV,BINV,CINV, &
       AMINC,AMAXC,BMINC,BMAXC,CMINC,CMAXC,XTLINV(6)
  real(chm_real)    XA,YA,ZA,XB,YB,ZB,XC,YC,ZC,XD,YD,ZD,XT,YT,ZT
  real(chm_real)    XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX
  real(chm_real)    XMINC,XMAXC,YMINC,YMAXC,ZMINC,ZMAXC
  real(chm_real)    XCENTC,YCENTC,ZCENTC,XCENT,YCENT,ZCENT
  real(chm_real)    SVAL,SMIN,RVAL
  !
  real(chm_real) RLIMC(-1:1,-1:1,-1:1),RLIM(-1:1,-1:1,-1:1)
  real(chm_real) ILIMC(-1:1,-1:1,-1:1),ILIM(-1:1,-1:1,-1:1)
  !
  !MFC ERROR XMINC,XMAXC,YMINC,... used before defined
  !    setting to zero to avoid compiler warnings
  xminc=0.d0
  yminc=0.d0
  zminc=0.d0
  xmaxc=0.d0
  ymaxc=0.d0
  zmaxc=0.d0
  !MFC
  NTRANS0 = NTRANS
  J=0
  DO I=1,NATOM
     IF(ISLCT(I).EQ.1.AND.X(I).LT.ANUM) THEN
        J = J+1
        X0(J) = X(I)
        Y0(J) = Y(I)
        Z0(J) = Z(I)
     ENDIF
  ENDDO
  NUMSEL=J
  !
  !     Calculate reciprocal vectors (XTLINV) and
  !     inverses of interplanar spacings (AINV,BINV,CINV)
  !
  CALL INVT33S(XTLINV, XTLABC, OK)
  IF(.NOT.OK) CALL WRNDIE(-5,'<XSCAN>',' XTLABC is singular.')
  AINV=ONE/SQRT(XTLINV(1)**2+XTLINV(2)**2+XTLINV(4)**2)
  BINV=ONE/SQRT(XTLINV(2)**2+XTLINV(3)**2+XTLINV(5)**2)
  CINV=ONE/SQRT(XTLINV(4)**2+XTLINV(5)**2+XTLINV(6)**2)
  AMIN =  RBIG
  AMAX = -RBIG
  BMIN =  RBIG
  BMAX = -RBIG
  CMIN =  RBIG
  CMAX = -RBIG
  !
  DO I=1,NUMSEL
     XA=(X0(I)*XTLINV(1)+Y0(I)*XTLINV(2)+Z0(I)*XTLINV(4))*AINV
     XB=(X0(I)*XTLINV(2)+Y0(I)*XTLINV(3)+Z0(I)*XTLINV(5))*BINV
     XC=(X0(I)*XTLINV(4)+Y0(I)*XTLINV(5)+Z0(I)*XTLINV(6))*CINV
     AMIN=MIN(AMIN,XA)
     AMAX=MAX(AMAX,XA)
     BMIN=MIN(BMIN,XB)
     BMAX=MAX(BMAX,XB)
     CMIN=MIN(CMIN,XC)
     CMAX=MAX(CMAX,XC)
  ENDDO

  AMINC = AMIN-CUTOFF
  AMAXC = AMAX+CUTOFF
  BMINC = BMIN-CUTOFF
  BMAXC = BMAX+CUTOFF
  CMINC = CMIN-CUTOFF
  CMAXC = CMAX+CUTOFF
  !
#if KEY_DEBUG==1
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,'(A)') ' Search Bounds :'
     WRITE(OUTU,'(A,2F16.5)') ' A Range ', AMINC,AMAXC
     WRITE(OUTU,'(A,2F16.5)') ' B Range ', BMINC,BMAXC
     WRITE(OUTU,'(A,2F16.5)') ' C Range ', CMINC,CMAXC
  ENDIF
#endif 
  !
  ! Calculate bounds on atoms (in 26 directions) from box center
  !
  CALL SCANBOX(NUMSEL,X0,Y0,Z0,RLIMC,ILIMC,XCENTC,YCENTC,ZCENTC)
  !
  CUTOF2 = CUTOFF*CUTOFF
  !
  NTRANS = 0
  DO ITRAN=1,XNSYMM
     IF (ITRAN.EQ.1) THEN   ! just translation (no rotation)
        XMIN=XMINC
        XMAX=XMAXC
        YMIN=YMINC
        YMAX=YMAXC
        ZMIN=ZMINC
        ZMAX=ZMAXC
        DO I=1,NUMSEL
           X1(I) = X0(I)
           Y1(I) = Y0(I)
           Z1(I) = Z0(I)
        ENDDO
        DO I=-1,1
           DO J=-1,1
              DO K=-1,1
                 RLIM(I,J,K)=RLIMC(I,J,K)
                 ILIM(I,J,K)=ILIMC(I,J,K)
              ENDDO
           ENDDO
        ENDDO
        XCENT=XCENTC
        YCENT=YCENTC
        ZCENT=ZCENTC
     ELSE
        DO I=1,3
           DO J=1,3
              R(J,I) = TRANSF(J,I,ITRAN)
           ENDDO
           T(I) = TRANSF(I,4,ITRAN)
        ENDDO
        DO I=1,NUMSEL
           X1(I) = T(1) + R(1,1)*X0(I) + R(1,2)*Y0(I) + R(1,3)*Z0(I)
           Y1(I) = T(2) + R(2,1)*X0(I) + R(2,2)*Y0(I) + R(2,3)*Z0(I)
           Z1(I) = T(3) + R(3,1)*X0(I) + R(3,2)*Y0(I) + R(3,3)*Z0(I)
        ENDDO
        !
        CALL SCANBOX(NUMSEL,X1,Y1,Z1,RLIM,ILIM,XCENT,YCENT,ZCENT)
        !
        AMIN =  RBIG
        AMAX = -RBIG
        BMIN =  RBIG
        BMAX = -RBIG
        CMIN =  RBIG
        CMAX = -RBIG
        DO I=1,NUMSEL
           XA = (X1(I)*XTLINV(1)+Y1(I)*XTLINV(2)+Z1(I)*XTLINV(4))*AINV
           XB = (X1(I)*XTLINV(2)+Y1(I)*XTLINV(3)+Z1(I)*XTLINV(5))*BINV
           XC = (X1(I)*XTLINV(4)+Y1(I)*XTLINV(5)+Z1(I)*XTLINV(6))*CINV
           AMIN=MIN(AMIN,XA)
           AMAX=MAX(AMAX,XA)
           BMIN=MIN(BMIN,XB)
           BMAX=MAX(BMAX,XB)
           CMIN=MIN(CMIN,XC)
           CMAX=MAX(CMAX,XC)
        ENDDO
     ENDIF
     !
     ALOW  = (AMINC-AMAX)/AINV
     AHIGH = (AMAXC-AMIN)/AINV
     BLOW  = (BMINC-BMAX)/BINV
     BHIGH = (BMAXC-BMIN)/BINV
     CLOW  = (CMINC-CMAX)/CINV
     CHIGH = (CMAXC-CMIN)/CINV
     XNA1 = INT(ALOW)
     XNA2 = INT(AHIGH)
     XNB1 = INT(BLOW)
     XNB2 = INT(BHIGH)
     XNC1 = INT(CLOW)
     XNC2 = INT(CHIGH)
     IF (ALOW  .LT. ZERO) XNA1 = XNA1-1
     IF (BLOW  .LT. ZERO) XNB1 = XNB1-1
     IF (CLOW  .LT. ZERO) XNC1 = XNC1-1
     IF (AHIGH .GT. ZERO) XNA2 = XNA2+1
     IF (BHIGH .GT. ZERO) XNB2 = XNB2+1
     IF (CHIGH .GT. ZERO) XNC2 = XNC2+1
     !
     IF(PRNLEV.GE.2) WRITE (OUTU,2001) &
          ITRAN,XNA1,XNA2,XNB1,XNB2,XNC1,XNC2
2001 FORMAT(/' Range of Grid Search for Transformation ',I5, &
          ' :',/, &
          ' Lattice Vector A ',I5,' TO ',I5,/, &
          ' Lattice Vector B ',I5,' TO ',I5,/, &
          ' Lattice Vector C ',I5,' TO ',I5,/)
     !
     !       Use lattice symmetry to reduce loop count for operation 1.
     !
     IF (ITRAN .EQ. 1) THEN
        I = MAX(XNA2,XNB2,XNC2)
        IF(XNA2 .EQ. I) THEN
           XNA2 = 0
        ELSEIF(XNB2 .EQ. I) THEN
           XNB2 = 0
        ELSEIF(XNC2 .EQ. I) THEN
           XNC2 = 0
        ENDIF
     ENDIF
     !
     !       Loop over the lattice translations.
     !
     DO IC=XNC1,XNC2
        XC = XTLABC(4) * IC
        YC = XTLABC(5) * IC
        ZC = XTLABC(6) * IC
        !
        DO IA=XNA1,XNA2
           XA = XTLABC(1) * IA
           YA = XTLABC(2) * IA
           ZA = XTLABC(4) * IA
           !
           DO IB=XNB1,XNB2
              !
              IF(ITRAN+IC**2+IA**2+IB**2.EQ.1) GOTO 400 ! ignore primary
              !
              XB = XTLABC(2) * IB
              YB = XTLABC(3) * IB
              ZB = XTLABC(5) * IB
              !
              XD = XA+XB+XC
              YD = YA+YB+YC
              ZD = ZA+ZB+ZC
              !
              SMIN=-RBIG
              DO I=-1,1
                 DO J=-1,1
                    DO K=-1,1
                       I2=I*I+J*J+K*K
                       IF(I2.GT.0) THEN
                          RVAL=I*XD+J*YD+K*ZD
                          S2=I2
                          SVAL=(RVAL-RLIMC(I,J,K)-RLIM(-I,-J,-K))/SQRT(S2)
                          IF(SMIN.LT.SVAL) THEN
                             SMIN=SVAL
                             II=I
                             JJ=J
                             KK=K
                          ENDIF
                       ENDIF
                    ENDDO
                 ENDDO
              ENDDO
              !
              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,666) IA,IB,IC,SMIN,XD,YD,ZD,II,JJ,KK
666              FORMAT(' TRYING:',3I5,4F10.4,3I5)
              ENDIF
              !
              !             the actual distance has SMIN as an lower bound
              IF(SMIN.GT.CUTOFF) GOTO 400  ! can't be close enough
              !
              !             check minimum distances from edge atoms (27x27).
              !
              S2MIN = RBIG
              DO I=-1,1
                 DO J=-1,1
                    DO K=-1,1
                       IAT=ILIMC(I,J,K)
                       DO II=-1,1
                          DO JJ=-1,1
                             DO KK=-1,1
                                JAT=ILIM(II,JJ,KK)
                                XT = X1(JAT) + XD - X0(IAT)
                                YT = Y1(JAT) + YD - Y0(IAT)
                                ZT = Z1(JAT) + ZD - Z0(IAT)
                                S2=XT**2+YT**2+ZT**2
                                IF(S2MIN.GT.S2) S2MIN=S2
                             ENDDO
                          ENDDO
                       ENDDO
                    ENDDO
                 ENDDO
              ENDDO
              !
              !
              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,677) SQRT(S2MIN),SMIN
677              FORMAT(20X,'Minimum edge atom distance.',2F10.4)
                 IF(SQRT(S2MIN).LT.SMIN) WRITE(OUTU,697) SMIN-SQRT(S2MIN)
697              FORMAT(40X,'XXXXXX Bad box edge difference.',2F10.4)
                 IF(SQRT(S2MIN).GE.SMIN) WRITE(OUTU,698) SMIN-SQRT(S2MIN)
698              FORMAT(40X,'           Box edge difference.',2F10.4)
              ENDIF
              !
              !
              !  The actual distance has S2MIN as an upper bound
              IF(S2MIN.LT.CUTOF2) GOTO 300 ! some distance is close
              !
              !  Now, no edge atom distance is close enough, but the box plane
              !  separation is less than the cutoff.
              !
              !  Since the edge atom method is inaccurate for small cufoffs
              !  Keep any box with a distance less than 5A.
              !
              IF(SMIN.LT.FIVE) GOTO 300
              !
              !  Keep any box with an edge atom distance less than 6A.
              !
              IF(S2MIN.LT.SIX*SIX) GOTO 300
              !
              !  Get rid of other unlikely transformations...
              !  When boxes are far apart, the edge atom method should be accurate.
              !  The cutoff scale factor is a bit less than COS(ATAN(SQRT(TWO))*HALF)
              !
              IF(SMIN.GT.CUTOFF*0.85) GOTO 400
              !
              !  Keep other likely transformations...
              !  If the box distance is less than half of the cutoff, and a large
              !  cutoff is used (>10A), then the box will likely be needed.
              !  This may save time for large boxes where atoms overflow the
              !  boundary in a complex manner.
              !
              IF(SMIN.LT.CUTOFF*HALF) GOTO 300
              !
              !   We don't expect to get here very often.  If we do, then do a
              !   double atom search (slow but effective).
              !
              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,668) SQRT(S2MIN),SMIN,CUTOFF
668              FORMAT(' Double loop needed:',5F10.4)
              ENDIF
              !
              !   Do double loop only for non overlapping boxes where none of the edge
              !   distances is less than the cutoff but yet the minimum box-box
              !   distance is less than the cutoff distance.  Exit early if an
              !   acceptable distance is found. - BRB
              !
              S2MIN = RBIG
              DO J=1,NUMSEL
                 XT = X1(J) + XD
                 YT = Y1(J) + YD
                 ZT = Z1(J) + ZD
                 DO I=1,NUMSEL
                    S2=(XT-X0(I))**2+(YT-Y0(I))**2+(ZT-Z0(I))**2
                    IF(S2MIN.GT.S2) THEN
                       S2MIN=S2
                       IF(S2.LT.CUTOF2) GOTO 300
                    ENDIF
                 ENDDO
              ENDDO
              !             ... no pair distance close enough...
              GOTO 400
              !
300           CONTINUE  ! accept this image (if jumped here)

              IF(PRNLEV.GT.8) THEN
                 WRITE(OUTU,667) SQRT(S2MIN),SMIN
667              FORMAT(20X,'Accepted based on distances.',2F10.4)
              ENDIF

              NTRANS = NTRANS + 1
              IF(NTRANS .GT. MAXTRN) THEN
                 CALL WRNDIE(0,'<XSCAN>', &
                      'Too many transformations generated.')
                 NTRANS=NTRANS-1
              ENDIF
              XDIST(NTRANS)  = SQRT(S2MIN)
              XNOP(NTRANS) = ITRAN
              XNA(NTRANS) = IA
              XNB(NTRANS) = IB
              XNC(NTRANS) = IC
              !
400           CONTINUE ! reject this image (if jumped here)
              !
           ENDDO
        ENDDO
     ENDDO
     !
     !       Add in the inverse transformations for operation 1.
     !
     IF(ITRAN .EQ. 1) THEN
        J = NTRANS
        DO I = 1,J
           IA = XNA(I)
           IB = XNB(I)
           IC = XNC(I)
           QSKIP = (XNA2.EQ.0 .AND. IA.EQ.0) .OR. &
                (XNB2.EQ.0 .AND. IB.EQ.0) .OR. &
                (XNC2.EQ.0 .AND. IC.EQ.0)
           IF (.NOT.QSKIP) THEN
              NTRANS = NTRANS + 1
              IF (NTRANS .GT. MAXTRN) THEN
                 CALL WRNDIE(0,'<XSCAN>', &
                      'Too many transformations generated.')
                 NTRANS = NTRANS-1
              ENDIF
              XDIST(NTRANS) = XDIST(I)
              XNOP(NTRANS)  = 1
              XNA(NTRANS)   = -IA
              XNB(NTRANS)   = -IB
              XNC(NTRANS)   = -IC
           ENDIF
        ENDDO
     ENDIF
     ! initialize new BIMAG%IMATPT elements
     DO I=NTRANS0+1,NTRANS
       BIMAG%IMATPT(I)=0
     ENDDO
  ENDDO
  !
  IF(PRNLEV.GE.2) THEN
     WRITE (OUTU, &
          '(/,'' The number of transformations generated = '',I5)') &
          NTRANS
     WRITE (OUTU, &
          '(//,'' Number  Symop   A   B   C   Distance'',/)')
     !RCZ *  '(//,'' Number  Symop   A   B   C   Distance  Order'',/)')
     DO I = 1,NTRANS
        WRITE (OUTU,'(2X,I5,2X,I5,3I4,1X,F10.4)') I,XNOP(I), &
             XNA(I),XNB(I),XNC(I),XDIST(I)
        !RCZ 92/06/11 - get rid of XORDER
        !         WRITE (OUTU,'(2X,I5,2X,I5,3I4,1X,F10.4,2X,I5)') I,XNOP(I),
        !         *      XNA(I),XNB(I),XNC(I),XDIST(I),XORDER(I)
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE XSCAN

SUBROUTINE SCANBOX(NAT,X,Y,Z,RLIM,ILIM,XCENT,YCENT,ZCENT)
  !
  ! This routine scans a group of atoms and returns limits in 26 directions.
  !
  use chm_kinds
  use number
  use stream
  implicit none
  !
  INTEGER NAT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) RLIM(-1:1,-1:1,-1:1)
  real(chm_real) ILIM(-1:1,-1:1,-1:1)
  real(chm_real) XCENT,YCENT,ZCENT
  !
  INTEGER I,J,K,IAT,I2
  real(chm_real)  RMAX,RVAL,S2
  real(chm_real)  RMX(-1:1,-1:1,-1:1)
  LOGICAL OK
  real(chm_real)  RX,RY,RZ,R
  !
  !------------------------------------------------------
  ! Calculate boundary values in 26 directions (RLIM)

  DO I=-1,1
     DO J=-1,1
        DO K=-1,1
           RMAX=-RBIG
           DO IAT=1,NAT
              RVAL=I*X(IAT)+J*Y(IAT)+K*Z(IAT)
              IF(RVAL.GT.RMAX) RMAX=RVAL
           ENDDO
           RLIM(I,J,K)=RMAX
        ENDDO
     ENDDO
  ENDDO
  !
  !------------------------------------------------------
  ! Calculate center of box (linear least squares method)
  !
  XCENT=0.0
  YCENT=0.0
  ZCENT=0.0
  DO I=0,1
     DO J=-1,1
        DO K=-1,1
           IF(4*I+2*J+K.GT.0) THEN
              S2=HALF*(RLIM(I,J,K)-RLIM(-I,-J,-K))/NINE
              XCENT=XCENT+S2*I
              YCENT=YCENT+S2*J
              ZCENT=ZCENT+S2*K
           ENDIF
        ENDDO
     ENDDO
  ENDDO
  !
  IF(PRNLEV.GT.8) THEN
     WRITE(OUTU,667) XCENT,YCENT,ZCENT
667  FORMAT(20X,'Transformation centered at:',3F10.4)
     IF(PRNLEV.GT.10) THEN
        DO I=0,1
           DO J=-1,1
              DO K=-1,1
                 IF(4*I+2*J+K.GT.0) THEN
                    R=XCENT*I+YCENT*J+ZCENT*K
                    R=R-HALF*(RLIM(I,J,K)-RLIM(-I,-J,-K))
                    S2=I*I+J*J+K*K
                    R=R/SQRT(S2)
                    WRITE(OUTU,669) I,J,K,R
669                 FORMAT(10X,'Planar distance to center:',3I3,F10.4)
                 ENDIF
              ENDDO
           ENDDO
        ENDDO
     ENDIF
  ENDIF
  !
  !---------------------------------------
  ! Calculate boundary atom pointers (ILIM)
  DO I=-1,1
     DO J=-1,1
        DO K=-1,1
           RMX(I,J,K)=-RBIG
        ENDDO
     ENDDO
  ENDDO
  !
  DO IAT=1,NAT
     RX=X(IAT)-XCENT
     RY=Y(IAT)-YCENT
     RZ=Z(IAT)-ZCENT
     R=SQRT(RX*RX+RY*RY+RZ*RZ)
     DO I=-1,1
        DO J=-1,1
           DO K=-1,1
              RVAL=I*RX+J*RY+K*RZ
              I2=I*I+J*J+K*K
              IF(I2.GT.0) THEN
                 S2=I2
                 RVAL=RVAL/SQRT(S2)
              ENDIF
              RVAL=TWO*RVAL-R
              IF(RVAL.GT.RMX(I,J,K)) THEN
                 RMX(I,J,K)=RVAL
                 ILIM(I,J,K)=IAT
              ENDIF
           ENDDO
        ENDDO
     ENDDO
  ENDDO
  !
  IF(PRNLEV.GT.8) THEN
     DO I=-1,1
        DO J=-1,1
           DO K=-1,1
              IAT=ILIM(I,J,K)
              WRITE(OUTU,698) I,J,K,RLIM(I,J,K), &
                   RMX(I,J,K),IAT,X(IAT),Y(IAT),Z(IAT)
698           FORMAT(20X,'Box dir:',3I4,2F10.4,I5,3F10.4)
           ENDDO
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE SCANBOX

SUBROUTINE XSYMPA(COMLYN,COMLEN,MAXSYM,XSYMOP,QERROR)
  !-----------------------------------------------------------------------
  !     Parses and interprets a symmetry operator, such as (-x,y+1/2,-z).
  !
  !     Author: Axel T. Brunger, 16-AUG-86
  !     Modified Martin J. Field for CHARMM, December 1986.
  !
  use chm_kinds
  use exfunc
  use stream
  use string, only:decodi,cnvtuc

  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER       XSYMOP(3,4)
  INTEGER       COMLEN,MAXSYM
  LOGICAL       QERROR
  !
  !
  CHARACTER(len=1) XYZOP(3)*1,TP*40
  INTEGER       DET, FIELD, FIRST(3), I, IC1, IC2, IFF, II, III, &
       IL, LAST(3), TPLEN, XYZ
  !
  QERROR = .FALSE.
  !
  !     Remove blanks from input string and convert characters to upper case
  !
  TPLEN = 0
  DO I = 1,COMLEN
     IF(COMLYN(I:I) .NE. ' ') THEN
        TPLEN = TPLEN + 1
        TP(TPLEN:TPLEN) = COMLYN(I:I)
     ENDIF
  ENDDO
  CALL CNVTUC(TP,TPLEN)
  !
  !     Get comma separators.
  !
  IC1 = INDEX(TP(1:TPLEN),',')
  IC2 = INDEX(TP(IC1+1:TPLEN),',')
  IF (IC1 .EQ. 0 .OR. IC2 .EQ. 0) THEN
     IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
          ' XSYMPA> WARNING: missing comma(s).'
     QERROR = .TRUE.
  ENDIF
  IC2 = IC2 + IC1
  IF (TP(1:1) .NE. '(' .OR. TP(TPLEN:TPLEN) /= ')') THEN
     IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
          ' XSYMPA> WARNING: missing parenthesis.'
     QERROR=.TRUE.
  ENDIF
  !
  !     Prepare string boundary lists for outer loop.
  !
  FIRST(1) = 2
  FIRST(2) = IC1+1
  FIRST(3) = IC2+1
  LAST(1)  = IC1-1
  LAST(2)  = IC2-1
  LAST(3)  = TPLEN-1
  !
  !     Prepare X-Y-Z operator list for inner loop.
  !
  XYZOP(1) = 'X'
  XYZOP(2) = 'Y'
  XYZOP(3) = 'Z'
  !
  !     Outer loop over sring fields.
  !
  DO FIELD = 1,3
     !
     IFF = FIRST(FIELD)
     IL  = LAST(FIELD)
     !
     !       Inner loop over basic operators X, Y, Z
     !
     DO XYZ=1,3
        I = INDEX(TP(IFF:IL),XYZOP(XYZ))
        IF (I .GT. 0) THEN
           IF (TP(I+IFF-2:I+IFF-2).EQ.'-') THEN
              XSYMOP(FIELD,XYZ) = -1
           ELSE
              XSYMOP(FIELD,XYZ) = 1
           ENDIF
        ELSE
           XSYMOP(FIELD,XYZ) = 0
        ENDIF
     ENDDO
     !
     !       Check if any translations are present.
     !
     I = INDEX(TP(IFF:IL),'/')
     IF (I .GT. 1) THEN
        II  = DECODI(TP(I+IFF-2:I+IFF-2),1)
        III = DECODI(TP(I+IFF:I+IFF),1)
        IF (II .EQ. 0 .OR. III .EQ. 0) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                ' XSYMPA> WARNING: Error in interpreting fraction.'
           QERROR = .TRUE.
        ELSE
           IF (TP(I+IFF-3:I+IFF-3).EQ.'-') THEN
              XSYMOP(FIELD,4) = -(II * MAXSYM) / III
           ELSE
              XSYMOP(FIELD,4) = (II * MAXSYM) / III
           ENDIF
        ENDIF
     ELSE
        XSYMOP(FIELD,4) = 0
     ENDIF
     !
  enddo
  !
  !     Check the determinant.
  !
  DET = XSYMOP(1,1) * &
       (XSYMOP(2,2) * XSYMOP(3,3) - XSYMOP(2,3) * XSYMOP(3,2)) - &
       XSYMOP(1,2) * &
       (XSYMOP(2,1) * XSYMOP(3,3) - XSYMOP(3,1) * XSYMOP(2,3)) + &
       XSYMOP(1,3) * &
       (XSYMOP(2,1) * XSYMOP(3,2) - XSYMOP(3,1) * XSYMOP(2,2))
  IF (ABS(DET).NE.1) THEN
     IF(WRNLEV.GE.2) WRITE (OUTU,'(A,I3)') &
          ' XSYMPA> WARNING: Determinant invalid = ',DET
     QERROR = .TRUE.
  ENDIF
  !
  RETURN
END SUBROUTINE XSYMPA

SUBROUTINE XSYMWR(CSYMOP,XSYMOP,MAXSYM)
  !-----------------------------------------------------------------------
  !     The input symmetry operation, XSYMOP, is converted to CHARACTER fo
  !
  use chm_kinds
  implicit none
  !
  CHARACTER(len=*) CSYMOP
  INTEGER     XSYMOP(3,4)
  INTEGER     MAXSYM
  !
  CHARACTER(len=4) XYZ(3),BOTTOM,TOP
  INTEGER     BIG,FIELD,HCF,I,IP,IT,IXYZ
  !
  !     Parameters needed because of a problem with the FLECS compiler.
  !
  character(len=1), PARAMETER :: BRA = '(', KET = ')'
  !
  IT = LEN(CSYMOP)
  DO I = 1,IT
     CSYMOP(I:I) = ' '
  ENDDO
  !
  CSYMOP(1:1) = '('
  IP = 2
  !
  XYZ(1) = 'X'
  XYZ(2) = 'Y'
  XYZ(3) = 'Z'
  !
  DO FIELD = 1,3
     !
     !       Loop over the X-Y-Z part of the operation.
     !
     DO I = 1,3
        IXYZ = XSYMOP(FIELD,I)
        IF (IXYZ .EQ. 1) THEN
           IF (CSYMOP(IP-1:IP-1) .EQ. BRA .OR. CSYMOP(IP-1:IP-1) &
                .EQ. ',') THEN
              CSYMOP(IP:IP) = XYZ(I)
              IP = IP + 1
           ELSE
              CSYMOP(IP:IP+1) = '+'//XYZ(I)
              IP = IP + 2
           ENDIF
        ELSE IF (IXYZ .EQ. -1) THEN
           CSYMOP(IP:IP+1) = '-'//XYZ(I)
           IP = IP + 2
        ENDIF
     ENDDO
     !
     !       Write out the translation part.
     !
     IT = XSYMOP(FIELD,4)
     IF (IT .NE. 0) THEN
        HCF = 1
        BIG = MAX(IT,MAXSYM)
        DO I = 2,BIG
           IF (MAXSYM .EQ. ((MAXSYM/I) * I)) THEN
              IF (IT .EQ. ((IT/I) * I)) HCF = I
           ENDIF
        ENDDO
        WRITE (BOTTOM,'(I4)') MAXSYM/HCF
        WRITE (TOP,'(I4)') (ABS (IT) / HCF)
        IF (IT .LT. 0) THEN
           CSYMOP(IP:IP+4) = '-'//TOP
           IP = IP + 5
        ELSE
           IF (CSYMOP(IP-1:IP-1) .EQ. BRA .OR. CSYMOP(IP-1:IP-1) &
                .EQ. ',') THEN
              CSYMOP(IP:IP+3) = TOP
              IP = IP + 4
           ELSE
              CSYMOP(IP:IP+4) = '+'//TOP
              IP = IP + 5
           ENDIF
        ENDIF
        CSYMOP(IP:IP) = '/'
        IP = IP + 1
        CSYMOP(IP:IP+3) = BOTTOM
        IP = IP + 4
     ENDIF
     !
     CSYMOP(IP:IP) = ','
     IP = IP + 1
  enddo
  !
  IP = IP - 1
  CSYMOP(IP:IP) = ')'
  !
  !     Take out spaces left in by translational output.
  !
  IT = 0
  DO I = 1,IP
     IF (CSYMOP(I:I) .NE. ' ') THEN
        IT = IT + 1
        CSYMOP(IT:IT) = CSYMOP(I:I)
     ENDIF
  ENDDO
  DO I = (IT+1),IP
     CSYMOP(I:I) = ' '
  ENDDO
  !
  RETURN
END SUBROUTINE XSYMWR

SUBROUTINE XTLAXS(XTLABC,XUCELL)
  !-----------------------------------------------------------------------
  !     THIS SUBROUTINE FINDS THE UNIT CELL TRANSLATIONS.
  !     unit cell is rotated to symmetric representation
  !     (S. Nose, M.L. Klein Mol. Phys. 1983, 50(5) 1055-1076)
  !
  !--------- INPUT:
  !
  !     XUCELL - unit cell parameters (a,b,c,alpha,beta,gamma)
  !
  !--------- OUTPUT:
  !
  !     XTLABC - symmetric shape matrix
  !
  !--------- other variables
  !
  !     DIFF   - if test=.true. it will contain difference between initial
  !     unit cell parameters and parameters calculated after
  !     symmetrization
  !     ERROR  - flag indicating if there were errors
  !
  !     H      - shape matrix; it contains calculated unit cell vectors (by co
  !
  !     TEST   - logical flag; if .true. comparison of symmetrized unit cell
  !     parameters with original parameters will be performed
  !     RSMALL - small number for tolerance testing
  !
  use chm_kinds
  use consta
  use number
  use stream
  implicit none
  !
  LOGICAL :: TEST=.true., ERROR
  real(chm_real) XTLABC(6),XUCELL(6),DIFF
  !
  INTEGER I,IER,J
  real(chm_real) A,B,C
  real(chm_real) EVAL(3),EV(3,3),HTH(6)

  ERROR = .FALSE.

  !     CALCULATE H(TRANSPOSE)H : HTH(I,J)=SUM(K) [H(K,I)*H(K,J)]
  !     HTH(I,J) = DOT_PRODUCT(A(I),A(J))

  CALL GenTen(XUCELL,HTH)

  !     DIAGONALIZE H(TRANSPOSE)H

  !     JOBN=1 LOWER TRIANGLE;  JOBN=11 FULL MATRIX
  !     CALL EIGRS(MATRIX,NDIM,JOBN,EVAL,EVEC,NDIM,IER)
  CALL EIGRS(HTH,3,1,EVAL,EV,3,IER)
  IF (IER .GT. 128) THEN
     IER = IER - 128
     WRITE (OUTU,'(A,I6)') &
          ' XTLAXS> EIGRS: FAILED TO CONVERGE ON ROOT NUMBER ', IER
     ERROR=.TRUE.
     CALL WRNDIE(-5,'<XTLAXS>','Diagonalization failed.')
     RETURN
  ENDIF

  ERROR=EVAL(1).LT.RSMALL .OR. EVAL(2).LT.RSMALL .OR. &
       EVAL(3).LT.RSMALL

  IF(ERROR) THEN
     IF(PRNLEV.GE.2) THEN
        WRITE(OUTU,'(A)') ' XTLAXS> ERROR: Negative eigenvalues'
        WRITE(OUTU,'(A/A)') &
             ' XTLAXS> ALPHA, BETA, GAMMA PARAMETERS ARE INCONSISTENT', &
             ' XTLAXS> UNIT CELL VECTORS CANNOT BE GENERATED'
        WRITE(OUTU,'(A,I2,1X,E10.2,5X,3F10.6)') &
             (' XTLAXS> EVAL & EV: ',J,EVAL(J),(EV(I,J),I=1,3),J=1,3)
        WRITE(OUTU,'(A,E10.4)') ' XTLAXS> RSMALL=',RSMALL
        WRITE(OUTU,'(/A,6F8.2)') &
             ' XTLAXS> XUCELL=',(XUCELL(I),I=1,6)
     ENDIF
     CALL WRNDIE(-5,'<XTLAXS>','Inconsistent angles.')
     RETURN
  ENDIF

  !     H(SYMMETRIC) = V D V (-1) = V U (TR) H = RH

  A=SQRT(EVAL(1))
  B=SQRT(EVAL(2))
  C=SQRT(EVAL(3))
  XTLABC(1)=A*EV(1,1)*EV(1,1)+B*EV(1,2)*EV(1,2)+C*EV(1,3)*EV(1,3)
  XTLABC(3)=A*EV(2,1)*EV(2,1)+B*EV(2,2)*EV(2,2)+C*EV(2,3)*EV(2,3)
  XTLABC(6)=A*EV(3,1)*EV(3,1)+B*EV(3,2)*EV(3,2)+C*EV(3,3)*EV(3,3)
  XTLABC(2)=A*EV(1,1)*EV(2,1)+B*EV(1,2)*EV(2,2)+C*EV(1,3)*EV(2,3)
  XTLABC(4)=A*EV(1,1)*EV(3,1)+B*EV(1,2)*EV(3,2)+C*EV(1,3)*EV(3,3)
  XTLABC(5)=A*EV(2,1)*EV(3,1)+B*EV(2,2)*EV(3,2)+C*EV(2,3)*EV(3,3)

  IF(TEST) THEN
     !
     !       CALCULATE A,B,C,ALPHA,BETA,GAMMA AND COMPARE WITH OLD VALUES
     !
     CALL XTLLAT(HTH,XTLABC)
     DIFF=ABS(XUCELL(1)-HTH(1))+ABS(XUCELL(2)-HTH(2))+ &
          ABS(XUCELL(3)-HTH(3))+ABS(XUCELL(4)-HTH(4))+ &
          ABS(XUCELL(5)-HTH(5))+ABS(XUCELL(6)-HTH(6))
     IF(DIFF.GT.RSMALL) THEN
        ERROR=.TRUE.
        IF(PRNLEV.GT.2) THEN
           WRITE(OUTU,'(A,2E12.4)') &
                ' XTLAXS> DIFF.GT.RSMALL: DIFF,RSMALL=',DIFF,RSMALL
           WRITE(OUTU,'(A,6F8.2)') &
                ' OLD: A,B,C,ALPHA,BETA,GAMMA=',(XUCELL(I),I=1,6)
           WRITE(OUTU,'(A,6F8.2)') &
                ' NEW: A,B,C,ALPHA,BETA,GAMMA=',(HTH(I),I=1,6)
           WRITE(OUTU,'(A,6E10.2)') &
                ' OLD-NEW: =',(XUCELL(I)-HTH(I),I=1,6)
           CALL WRNDIE(-5,'<XTLAXS>','Tolerance exceded.')
        ENDIF
     ENDIF
  ENDIF

  RETURN
END SUBROUTINE XTLAXS

subroutine xtlaxsacc(qorth,xtlrot,xucell,xtlabc)
  use chm_kinds
  use consta
  use number
  use stream
  use image, only: xtltyp
  implicit none
  !
  real(chm_real), intent(in) :: XUCELL(6),XTLABC(6)
  real(chm_real), intent(out) :: xtlrot(3,3)
  logical, intent(out) :: qorth
  !
  real(chm_real) :: v,xtlacc(3,3),kxtlabc(3,3)

  qorth=(xucell(4)==90 .and. xucell(5)==90 .and. xucell(6)==90)

  if (qorth) then
     xtlrot=0
     xtlrot(1,1)=1
     xtlrot(2,2)=1
     xtlrot(3,3)=1
  else
     ! Box vectors of accelerator orientation
     xtlacc=0
     xtlacc(1,1)=xucell(1)
     xtlacc(2,1)=xucell(2)*cos(DEGRAD*xucell(6))
     xtlacc(2,2)=xucell(2)*sin(DEGRAD*xucell(6))
     xtlacc(3,1)=xucell(3)*cos(DEGRAD*xucell(5))
     xtlacc(3,2)=xucell(3)*(cos(DEGRAD*xucell(4))-cos(DEGRAD*xucell(5))*cos(DEGRAD*xucell(6)))/sin(DEGRAD*XUCELL(6))
     xtlacc(3,3)=sqrt(xucell(3)*xucell(3)-xtlacc(3,1)*xtlacc(3,1)-xtlacc(3,2)*xtlacc(3,2))

     ! Reciprocal box vectors of symmetric orientation
     v=xtlacc(1,1)*xtlacc(2,2)*xtlacc(3,3)
     kxtlabc(1,1)=(xtlabc(3)*xtlabc(6)-xtlabc(5)*xtlabc(5))/v ! 22*33-23*32
     kxtlabc(1,2)=(xtlabc(5)*xtlabc(4)-xtlabc(2)*xtlabc(6))/v ! 23*31-21*33
     kxtlabc(1,3)=(xtlabc(2)*xtlabc(5)-xtlabc(3)*xtlabc(4))/v ! 21*32-22*31
     kxtlabc(2,1)=kxtlabc(1,2)
     kxtlabc(2,2)=(xtlabc(6)*xtlabc(1)-xtlabc(4)*xtlabc(4))/v ! 33*11-31*13
     kxtlabc(2,3)=(xtlabc(4)*xtlabc(2)-xtlabc(5)*xtlabc(1))/v ! 31*12-32*11
     kxtlabc(3,1)=kxtlabc(1,3)
     kxtlabc(3,2)=kxtlabc(2,3)
     kxtlabc(3,3)=(xtlabc(1)*xtlabc(3)-xtlabc(2)*xtlabc(2))/v ! 11*22-12*21

     ! Rotation matrix
     xtlrot=matmul(transpose(xtlacc),kxtlabc)
  endif

  return
end subroutine xtlaxsacc

SUBROUTINE XTLLAT(XUCELL,XTLABC)
  !-----------------------------------------------------------------------
  !     The lattice parameters are calculated here from the lattice
  !     translations.
  !
  !RCZ 92/01/24
  !
  use chm_kinds
  use consta
  use number
  use stream
  implicit none
  !
  real(chm_real)      XTLABC(6),XUCELL(6)
  !
  !     local
  !
  real(chm_real)      A,B,C,AB,BC,CA
  !
  !     Calculate A,B,C,ALPHA,BETA,GAMMA from shape matrix XTLABC
  !
  A  = SQRT(XTLABC(1)**2 + XTLABC(2)**2 + XTLABC(4)**2)
  B  = SQRT(XTLABC(2)**2 + XTLABC(3)**2 + XTLABC(5)**2)
  C  = SQRT(XTLABC(4)**2 + XTLABC(5)**2 + XTLABC(6)**2)
  AB = XTLABC(2)*(XTLABC(1) + XTLABC(3)) + XTLABC(4)*XTLABC(5)
  BC = XTLABC(5)*(XTLABC(3) + XTLABC(6)) + XTLABC(2)*XTLABC(4)
  CA = XTLABC(4)*(XTLABC(1) + XTLABC(6)) + XTLABC(2)*XTLABC(5)
  XUCELL(1) = A
  XUCELL(2) = B
  XUCELL(3) = C
  XUCELL(4) = ACOS(BC/(B*C))*RADDEG
  XUCELL(5) = ACOS(CA/(C*A))*RADDEG
  XUCELL(6) = ACOS(AB/(A*B))*RADDEG
  RETURN
END SUBROUTINE XTLLAT

SUBROUTINE XTLSYM(XTLABC,XUCELL,XTLTYP,XDIM,XTLREF)
  !-----------------------------------------------------------------------
  !     CHECK IF SYMMETRY IS CONSERVED
  !
  !                  a 0 0
  !  cubic(1)        0 a 0    a = b = c ;  alpha = beta = gamma = 90.0
  !                  0 0 a
  !
  !                  a 0 0
  !  tetragonal(2)   0 a 0    a = b ; alpha = beta = gamma = 90.0
  !                  0 0 c
  !
  !                  a 0 0
  !  rectangular(1)  0 b 0    a/A=b/B=c/C ; alpha = beta = gamma = 90.0
  !                  0 0 c
  !
  !                  R U 0
  !  hexagonal(2)    U R 0    a = b ; alpha = beta = 90.0 and gamma = 120.0
  !                  0 0 c    R = a*cos(15), U = -a*sin(15)
  !
  !                  R S S
  !  rhombohedral(2) S R S    a = b = c ; alpha=beta=gamma<120 (!=90)
  !   (trigonal)     S S R
  !
  !                  a 0 0
  !  orthorhombic(3) 0 b 0    alpha = beta = gamma = 90.0
  !                  0 0 c
  !
  !                  R 0 W
  !  monoclinic(4)   0 b 0    alpha = gamma = 90.0 ( beta > 90 ??
  !                  W 0 T
  !
  !                  R U W
  !  triclinic(6)    U S V    a != b != c ; alpha != beta != gamma
  !                  W V T
  !
  !-----------------------------------------------------------------------
  use chm_kinds
  use consta
  use number
  use stream
  implicit none
  !
  !     input
  !
  CHARACTER(len=4) XTLTYP
  INTEGER     XDIM
  real(chm_real)      XTLABC(6),XUCELL(6),XTLREF(6)
  !
  !     local
  !
  INTEGER     I
  LOGICAL     ERROR
  real(chm_real)      A,B,C,ALPHA,BETA,GAMMA,DIFF,D
  real(chm_real)      COS15,SIN15,TAN15,TANPI16
  real(chm_real)      XTLPRV(6)
  !
  DO I=1,6
     XTLPRV(I)=XTLABC(I)
  ENDDO
  ERROR=.FALSE.

  IF(XTLTYP.EQ.'CUBI' .OR. XTLTYP.EQ.'ORTH' .OR. &
       XTLTYP.EQ.'TETR' .OR. XTLTYP.EQ.'RECT' ) THEN
     DIFF=ABS(XTLABC(2))+ABS(XTLABC(4))+ABS(XTLABC(5))
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     XTLABC(2)=ZERO
     XTLABC(4)=ZERO
     XTLABC(5)=ZERO
     !
     IF(XTLTYP.EQ.'CUBI') THEN
        XDIM=1
        A=(XTLABC(1)+XTLABC(3)+XTLABC(6))*THIRD
        DIFF=ABS(XTLABC(1)-A)+ABS(XTLABC(3)-A)+ABS(XTLABC(6)-A)
        IF(DIFF.GT.RSMALL) ERROR=.TRUE.
        XTLABC(1)=A
        XTLABC(3)=A
        XTLABC(6)=A
        !
     ELSEIF(XTLTYP.EQ.'TETR') THEN
        XDIM=2
        A=(XTLABC(1)+XTLABC(3))*HALF
        DIFF=ABS(XTLABC(1)-A)+ABS(XTLABC(3)-A)
        IF(DIFF.GT.RSMALL) ERROR=.TRUE.
        XTLABC(1)=A
        XTLABC(3)=A
        !
     ELSEIF(XTLTYP.EQ.'RECT') THEN
        XDIM=1         
        A=(XTLABC(1)/XTLREF(1)+XTLABC(3)/XTLREF(2)+ &
             XTLABC(6)/XTLREF(3))*THIRD
        DIFF=ABS(XTLABC(1)-A*XTLREF(1))+ABS(XTLABC(6)-A*XTLREF(3))+ &
             ABS(XTLABC(3)-A*XTLREF(2))
        IF(DIFF.GT.RSMALL) ERROR=.TRUE.
        XTLABC(1)=A*XTLREF(1)
        XTLABC(3)=A*XTLREF(2)
        XTLABC(6)=A*XTLREF(3)
        !
     ELSEIF(XTLTYP.EQ.'ORTH') THEN
        XDIM=3
     ENDIF
     !
  ELSEIF(XTLTYP.EQ.'MONO') THEN
     XDIM=4
     DIFF=ABS(XTLABC(2))+ABS(XTLABC(5))
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     XTLABC(2)=ZERO
     XTLABC(5)=ZERO
     !
  ELSEIF(XTLTYP.EQ.'HEXA') THEN
     COS15=COS(PI/TWELVE)
     SIN15=SIN(PI/TWELVE)
     TAN15=SIN15/COS15
     XDIM=2
     DIFF=ABS(XTLABC(4))+ABS(XTLABC(5))+ &
          ABS(XTLABC(2)+TAN15*XTLABC(3))+ &
          ABS(XTLABC(2)+TAN15*XTLABC(1))
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     A=SQRT(XTLABC(1)**2+XTLABC(2)**2)
     XTLABC(1)= A*COS15
     XTLABC(2)=-A*SIN15
     XTLABC(3)= A*COS15
     XTLABC(4)=ZERO
     XTLABC(5)=ZERO
     !
  ELSEIF(XTLTYP.EQ.'RHOM') THEN
     XDIM=2
     A=(XTLABC(1)+XTLABC(3)+XTLABC(6))*THIRD
     B=(XTLABC(2)+XTLABC(4)+XTLABC(5))*THIRD
     DIFF=ABS(XTLABC(1)-A)+ABS(XTLABC(3)-A)+ABS(XTLABC(6)-A)+ &
          ABS(XTLABC(2)-B)+ABS(XTLABC(4)-B)+ABS(XTLABC(5)-B)
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     XTLABC(1)=A
     XTLABC(3)=A
     XTLABC(6)=A
     XTLABC(2)=B
     XTLABC(4)=B
     XTLABC(5)=B
     !
  ELSEIF(XTLTYP.EQ.'OCTA') THEN
     XDIM=1
     A=(XTLABC(1)+XTLABC(3)+XTLABC(6))*THIRD
     B=-A/5.0
     DIFF=ABS(XTLABC(1)-A)+ABS(XTLABC(3)-A)+ABS(XTLABC(6)-A)+ &
          ABS(XTLABC(2)-B)+ABS(XTLABC(4)-B)+ABS(XTLABC(5)-B)
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     XTLABC(1)=A
     XTLABC(3)=A
     XTLABC(6)=A
     XTLABC(2)=B
     XTLABC(4)=B
     XTLABC(5)=B
     !
  ELSEIF(XTLTYP.EQ.'RHDO') THEN
     XDIM=1
     A=(XTLABC(1)+XTLABC(6))*HALF
     TANPI16=PI/16.0
     TANPI16=TAN(TANPI16)
     B=A*(ONE-TANPI16**2)
     C=A*TANPI16*SQRT(TWO)
     D=-A*TANPI16**2
     DIFF=ABS(XTLABC(1)-A)+ABS(XTLABC(3)-B)+ABS(XTLABC(6)-A)+ &
          ABS(XTLABC(2)-C)+ABS(XTLABC(4)-D)+ABS(XTLABC(5)-C)
     IF(DIFF.GT.RSMALL) ERROR=.TRUE.
     XTLABC(1)=A
     XTLABC(3)=B
     XTLABC(6)=A
     XTLABC(2)=C
     XTLABC(4)=D
     XTLABC(5)=C
     !
  ELSEIF(XTLTYP.EQ.'TRIC') THEN
     XDIM=6
  ELSE
     CALL WRNDIE(-5,'<XTLSYM>','Unknown crystal type.')
  ENDIF
  !
  IF(ERROR) THEN
     IF(PRNLEV.GT.2) THEN
        WRITE(OUTU,'(A,2E12.4)')' XTLSYM> DIFF,RSMALL',DIFF,RSMALL
        WRITE(OUTU,'(A,A4)')    ' XTLSYM> XTLTYP=',XTLTYP
        WRITE(OUTU,'(1X,A,6F12.6)') ' XTLABC:',XTLPRV
        CALL WRNDIE(-4,'<XTLSYM>', &
             'Shape matrix symmetry is not conserved')
     ENDIF
  ENDIF
  !
  A     = XUCELL(1)
  B     = XUCELL(2)
  C     = XUCELL(3)
  ALPHA = XUCELL(4)
  BETA  = XUCELL(5)
  GAMMA = XUCELL(6)

  IF(A.LT.PTONE .OR. B.LT.PTONE .OR. C.LT.PTONE  .OR. &
       ALPHA.LT.PTONE .OR. BETA.LT.PTONE .OR. GAMMA.LT.PTONE ) THEN
     WRITE(OUTU,'(A,3F10.3)')'A,B,C            = ',A,B,C
     WRITE(OUTU,'(A,3F10.3)')'ALPHA,BETA,GAMMA = ',ALPHA,BETA,GAMMA
     CALL WRNDIE(-5,'<XTLSYM>','Illegal XUCELL')
  ENDIF

  IF (XTLTYP.EQ.'MONO' .OR. XTLTYP.EQ.'ORTH') THEN
     ALPHA = NINETY
     GAMMA = NINETY
     IF (XTLTYP.EQ.'ORTH') BETA  = NINETY
  ELSEIF(XTLTYP.EQ.'TETR' .OR. XTLTYP.EQ.'HEXA') THEN
     A = (A+B)*HALF
     B = A
     ALPHA = NINETY
     BETA  = NINETY
     GAMMA = NINETY
     IF(XTLTYP.EQ.'HEXA') GAMMA = ONE2TY
  ELSEIF(XTLTYP .EQ. 'CUBI' .OR. XTLTYP.EQ.'RHOM' .OR. &
       XTLTYP .EQ. 'OCTA' .OR. XTLTYP.EQ.'RHDO' ) THEN
     A = (A+B+C)*THIRD
     B = A
     C = A
     IF(XTLTYP .EQ. 'CUBI') THEN
        ALPHA = NINETY
        BETA  = NINETY
        GAMMA = NINETY
     ELSEIF(XTLTYP .EQ. 'RHOM') THEN
        ALPHA = (ALPHA+BETA+GAMMA)*THIRD
        BETA  = ALPHA
        GAMMA = ALPHA
     ELSEIF(XTLTYP .EQ. 'OCTA') THEN
        !            ALPHA = 109.4712206344907D0
        ALPHA = DACOS(-THIRD)*RADDEG
        BETA  = ALPHA
        GAMMA = ALPHA
     ELSEIF(XTLTYP .EQ. 'RHDO') THEN
        ALPHA = SIXTY
        BETA  = NINETY
        GAMMA = SIXTY
     ENDIF
  ELSEIF(XTLTYP.EQ.'RECT') THEN
     D = (A/XTLREF(1)+B/XTLREF(2)+C/XTLREF(3))*THIRD
     A = D*XTLREF(1)
     B = D*XTLREF(2)
     C = D*XTLREF(3)
     ALPHA = NINETY
     BETA  = NINETY
     GAMMA = NINETY
  ENDIF
  !
  !     Calculate difference before and after symmetrization
  !
  DIFF=ABS(XUCELL(1)-A)+ABS(XUCELL(2)-B)+ABS(XUCELL(3)-C)+ &
       ABS(XUCELL(4)-ALPHA)+ABS(XUCELL(5)-BETA)+ABS(XUCELL(6)-GAMMA)

  IF(DIFF.GT.RSMALL) THEN
     IF(PRNLEV.GT.2) THEN
        WRITE(OUTU,'(A,A4)')    ' XTLSYM> XTLTYP=',XTLTYP
        WRITE(OUTU,'(A,6F8.2)') ' XTLSYM> XUCELL BEFORE=', &
             (XUCELL(I),I=1,6)
        WRITE(OUTU,'(A,6F8.2)') ' XTLSYM> XUCELL AFTER =', &
             A,B,C,ALPHA,BETA,GAMMA
     ENDIF
     CALL WRNDIE(-4,'<XTLSYM>','XUCELL Symmetry is not conserved')
  ENDIF
  !
  !     Fill XUCELL
  !
  XUCELL(1) = A
  XUCELL(2) = B
  XUCELL(3) = C
  XUCELL(4) = ALPHA
  XUCELL(5) = BETA
  XUCELL(6) = GAMMA
  RETURN
END SUBROUTINE XTLSYM

!==============================================================================
SUBROUTINE ROTXTL(NATOM,X,Y,Z,ISLCT,XTLABC,XUCELL)
  !-----------------------------------------------------------------------
  !     This subroutine rotates the selected primary atoms according to
  !     following formula.
  !
  !     Generally,
  !                            G
  !     fractional coor.   --------->   cartesian coor.
  !                            F
  !        G : Axis transformation matrix
  !        F : Coordinate transformation matrix        G(T)F = 1
  !
  !     1) Generate the axis transfromation matrix and
  !                 the coordinate transfromation matrix
  !
  !         x axis // a axis
  !         y axis is in a plane by a1-a2
  !         z axis // a^3 ( ^ means reciprocal vector)
  !
  !     2) Calculate rotation matrix R
  !         R = XTLABC x F(-1) = XTLABC x G(T)
  !
  !     August 1, 1995  Wonpil Im (Hanyang University, Seoul, Korea)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use stream
  implicit none
  !
  INTEGER     ISLCT(*)
  INTEGER     NATOM
  real(chm_real)      X(*),Y(*),Z(*)
  !
  INTEGER     I,NUMSEL,J
  real(chm_real)      XUCELL(6),XTLABC(6),HTH(6),IHTH(6),RUCELL(6)
  real(chm_real)      G(3,3),GT(3,3),R(3,3)
  real(chm_real)      X0,Y0,Z0,ANGSUM
  LOGICAL     OK
  !
  !rcz...test if rotation is necessary, Ryszard Czerminski 02-Aug-95
  !
  ANGSUM = ABS(XUCELL(4)-NINETY) &
       + ABS(XUCELL(5)-NINETY) &
       + ABS(XUCELL(6)-NINETY)
  IF (ANGSUM.LT.PT001) RETURN
  !rcz..
  !
  !     Check the coordinates and allocate space for the crystal generatio
  !
  NUMSEL = 0
  DO I = 1,NATOM
     IF (ISLCT(I) .EQ. 1) THEN
        IF (X(I) .GE. ANUM) THEN
           CALL WRNDIE(0,'<ROTXTL>', &
                'Undefined coordinate in selected set.')
        ELSE
           NUMSEL = NUMSEL + 1
        ENDIF
     ENDIF
  ENDDO
  IF (NUMSEL .LE. 0) THEN
     CALL WRNDIE(0,'<ROTXTL>','No coordinates selected.')
     RETURN
  ENDIF
  !
  !     CALCULATE METRIC TENSOR HTH
  !
  CALL GENTEN(XUCELL,HTH)
  !
  !     CALCULATE RECIPROCAL BASIS VECTOR BY CALLING COMRBV
  !
  CALL INVT33S(IHTH,HTH,OK)
  CALL COMRBV(XUCELL,IHTH,RUCELL)
  !
  !     CALCULATE MATRIX G AND MATRIX A
  !
  G(1,1) =  1.0 / XUCELL(1)
  G(1,2) =  0.0
  G(1,3) =  0.0
  G(2,1) = -1.0 / (XUCELL(1)*tan(XUCELL(6)*degrad))
  G(2,2) =  1.0 / (XUCELL(2)*sin(XUCELL(6)*degrad))
  G(2,3) =  0.0
  G(3,1) =  RUCELL(1)*cos(RUCELL(5)*degrad)
  G(3,2) =  RUCELL(2)*cos(RUCELL(4)*degrad)
  G(3,3) =  RUCELL(3)
  !
  !     CALCULATE ROTATION MATRIX R
  !
  CALL TRANSPS(GT,G,3,3)
  CALL MULNXNLF(R,XTLABC,GT,3)
  !
  !     ROTATE THE PRIMARY ATOMS
  !
  write(OUTU,'(a)') ' ROTXTL> Rotation Matrix '
  write(OUTU,'(12X,3f10.5)') ((R(i,j),j=1,3),i=1,3)
  !RCZ950801 DO I = 1, NUMSEL
  DO I = 1,NATOM
     IF (ISLCT(I).ne.0) THEN
        X0 = X(I)
        Y0 = Y(I)
        Z0 = Z(I)
        X(I) = R(1,1)*X0 + R(1,2)*Y0 + R(1,3)*Z0
        Y(I) = R(2,1)*X0 + R(2,2)*Y0 + R(2,3)*Z0
        Z(I) = R(3,1)*X0 + R(3,2)*Y0 + R(3,3)*Z0
     ENDIF
  ENDDO
  write(OUTU,'(a)') ' ROTXTL> Rotation is completed '
  !
  RETURN
END SUBROUTINE ROTXTL


Subroutine COMRBV(XUCELL,IHTH,RUCELL)
  !------------------------------------------------------------------------------
  !
  !     COMputer Reciprocal Basis Vectors
  !
  !     a^(i) = (sum j) IHTH(i,j) a(j)  !! CAUTION IHTH(6)
  !
  !
  use chm_kinds
  implicit none
  integer   i
  real(chm_real)    XUCELL(6),HTH(6),IHTH(6),RUCELL(6)
  real(chm_real)    U(3),V(3),W(3),TEMU(6),RBV(3,3)
  real(chm_real)    a,b,c,alpha,beta,gamma
  !
  !     CALCULATE RECIPROCAL BASIS VECTOR
  !
  RBV(1,1) = IHTH(1) * XUCELL(1)
  RBV(1,2) = IHTH(2) * XUCELL(2)
  RBV(1,3) = IHTH(4) * XUCELL(3)
  RBV(2,1) = IHTH(2) * XUCELL(1)
  RBV(2,2) = IHTH(3) * XUCELL(2)
  RBV(2,3) = IHTH(5) * XUCELL(3)
  RBV(3,1) = IHTH(4) * XUCELL(1)
  RBV(3,2) = IHTH(5) * XUCELL(2)
  RBV(3,3) = IHTH(6) * XUCELL(3)
  !
  !     To calculate lengths and angles of reciprocal basis vectors
  !
  !     reset a = 1 , b = 1, c = 1
  !     because the basis vector of the reciprocal vector has
  !     the following unit cell parameter : (1, 1, 1, alpha, beta, gamma)
  !
  !     reset rbv(3,3) ----> U(3),V(3),W(3)
  !
  DO I = 1, 6
     TEMU(I) = XUCELL(I)
  ENDDO
  TEMU(1) = 1.0
  TEMU(2) = 1.0
  TEMU(3) = 1.0
  !
  do  i = 1,3
     U(i) = RBV(1,i)
     V(i) = RBV(2,i)
     W(i) = RBV(3,i)
  enddo
  !
  CALL GenTen(TEMU,HTH)
  !
  CALL Callen(U,V,HTH,a,b,gamma)
  CALL Callen(V,W,HTH,b,c,alpha)
  CALL Callen(W,U,HTH,c,a,beta)
  !
  RUCELL(1) = A
  RUCELL(2) = B
  RUCELL(3) = C
  RUCELL(4) = ALPHA
  RUCELL(5) = BETA
  RUCELL(6) = GAMMA
  !
  return
end Subroutine COMRBV


Subroutine GenTen(XUCELL,HTH)
  !------------------------------------------------------------------------------
  !     Compute Metric Tensor HTH (symmetric metrix)
  !     HTH = a(i):a(j), where a(i) is basis vectors.
  !
  use chm_kinds
  use number
  use consta
  implicit none
  real(chm_real)    XUCELL(6),HTH(6)
  !
  HTH(1) = XUCELL(1)**2
  HTH(3) = XUCELL(2)**2
  HTH(6) = XUCELL(3)**2
  !
  !rcz..Adds test for angles close to 90, Ryszard Czerminki 02-Aug-95
  !      HTH(2)=XUCELL(1)*XUCELL(2)*COS(DEGRAD*XUCELL(6))
  !      HTH(4)=XUCELL(1)*XUCELL(3)*COS(DEGRAD*XUCELL(5))
  !      HTH(5)=XUCELL(2)*XUCELL(3)*COS(DEGRAD*XUCELL(4))
  IF(ABS(XUCELL(4)-NINETY).LT.RSMALL) THEN
     HTH(5)=ZERO
  ELSE
     HTH(5)=XUCELL(2)*XUCELL(3)*COS(DEGRAD*XUCELL(4))
  ENDIF
  IF(ABS(XUCELL(5)-NINETY).LT.RSMALL) THEN
     HTH(4)=ZERO
  ELSE
     HTH(4)=XUCELL(1)*XUCELL(3)*COS(DEGRAD*XUCELL(5))
  ENDIF
  IF(ABS(XUCELL(6)-NINETY).LT.RSMALL) THEN
     HTH(2)=ZERO
  ELSE
     HTH(2)=XUCELL(1)*XUCELL(2)*COS(DEGRAD*XUCELL(6))
  ENDIF
  !rcz..
  return
end Subroutine GenTen


Subroutine CalLen(A,B,HTH,magA,magB,angle)
  !------------------------------------------------------------------------------
  !     compute the length of each vectors and angle between them
  !
  !
  use chm_kinds
  use consta
  implicit none
  integer   i,j
  real(chm_real)    A(3),B(3),HTH(6),G(3,3)
  real(chm_real)    magA,magB,ANGLE
  real(chm_real)    w,dotval
  !
  G(1,1) = HTH(1)
  G(1,2) = HTH(2)
  G(1,3) = HTH(4)
  G(2,2) = HTH(3)
  G(2,3) = HTH(5)
  G(3,3) = HTH(6)
  G(2,1) = G(1,2)
  G(3,1) = G(1,3)
  G(3,2) = G(2,3)
  !
  !     CALCULATE DOT PRODUCT
  !
  DOTVAL = 0.0
  DO I = 1, 3
     DO J = 1, 3
        dotval = dotval + A(i) * B(j) * G(i,j)
     enddo
  enddo
  !
  !     CALCULATE THE LENGTHS OF EACH VECTORS
  !
  magA = 0.0
  magB = 0.0
  do I = 1, 3
     do J = 1, 3
        magA =   magA + A(i) * A(j) * g(i,j)
        magB =   magB + B(i) * B(j) * g(i,j)
     enddo
  enddo
  magA = sqrt(magA)
  magB = sqrt(magB)
  !
  !     CALCULATE THE ANGELE BETWEEN THE VECTORS
  !
  W =  dotval/magA/magB
  angle = acos(W) * raddeg
  !
  RETURN
END Subroutine CalLen

SUBROUTINE GETXTL(VARB,XTLABC,XTLTYP,XTLREF)
  !-----------------------------------------------------------------------
  !     Put the crystal variables into the variable array.
  !
  use chm_kinds
  use consta
  use number
  implicit none
  !
  CHARACTER(len=4) XTLTYP
  real(chm_real)      VARB(*), XTLABC(6), XTLREF(6)
  !
  real(chm_real) TANPI16, TANP2
  !
  IF (XTLTYP.EQ.'CUBI') THEN
     VARB(1) = XTLABC(1)
  ELSEIF (XTLTYP.EQ.'RECT') THEN
     VARB(1) = XTLABC(1)/XTLREF(1)
  ELSEIF (XTLTYP.EQ.'TETR') THEN
     VARB(1) = XTLABC(1)
     VARB(2) = XTLABC(6)
  ELSEIF (XTLTYP.EQ.'HEXA') THEN
     VARB(1) = SQRT(XTLABC(1)**2+XTLABC(2)**2)
     VARB(2) = XTLABC(6)
  ELSEIF (XTLTYP .EQ. 'RHOM') THEN
     VARB(1) = XTLABC(1)
     VARB(2) = XTLABC(2)
  ELSEIF (XTLTYP .EQ. 'OCTA') THEN
     VARB(1) = XTLABC(1)
  ELSEIF (XTLTYP .EQ. 'RHDO') THEN
     TANPI16=PI/16.0
     TANPI16=TAN(TANPI16)
     TANP2=TANPI16**2
     VARB(1) = XTLABC(1)*(ONE+TANP2)
  ELSEIF (XTLTYP .EQ. 'ORTH') THEN
     VARB(1) = XTLABC(1)
     VARB(2) = XTLABC(3)
     VARB(3) = XTLABC(6)
  ELSEIF (XTLTYP .EQ. 'MONO') THEN
     VARB(1) = XTLABC(1)
     VARB(2) = XTLABC(3)
     VARB(3) = XTLABC(6)
     VARB(4) = XTLABC(4)
  ELSEIF (XTLTYP .EQ. 'TRIC') THEN
     VARB(1) = XTLABC(1)
     VARB(2) = XTLABC(3)
     VARB(3) = XTLABC(6)
     VARB(4) = XTLABC(2)
     VARB(5) = XTLABC(4)
     VARB(6) = XTLABC(5)
  ELSE
     CALL WRNDIE(-5,'<GETXTL>',' Unknown crystal type.')
  ENDIF
  RETURN
END SUBROUTINE GETXTL

SUBROUTINE PUTXTL(VARB,XTLABCX,XTLTYPX,XTLREFX)
  !-----------------------------------------------------------------------
  !     Fill crystal arrays with the variables.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use image
  use memory
  implicit none
  !
  real(chm_real),allocatable,dimension(:,:,:) :: TRANSF
  real(chm_real)  VARB(*),XTLABCX(6),XTLREFX(6)
  CHARACTER(len=4) XTLTYPX
  !
  CALL SETXTL(VARB,XTLABCX,XTLTYPX,XTLREFX)
  !
  xucold(1:6) = xucell(1:6)
  CALL XTLLAT(XUCELL,XTLABCX)
  call chmalloc('crystal.src','CRYSTL','TRANSF',3,4,XNSYMM,crl=TRANSF)
  CALL IMFILL(TRANSF,.FALSE.)
  RETURN
END SUBROUTINE PUTXTL

SUBROUTINE SETXTL(VARB,XTLABC,XTLTYP,XTLREF)
  !-----------------------------------------------------------------------
  !     Fill crystal arrays with the variables.
  !
  use chm_kinds
  use consta
  use number
  implicit none
  !
  real(chm_real)  VARB(*),XTLABC(6),XTLREF(6)
  CHARACTER(len=*) XTLTYP
  !
  real(chm_real) COS15,SIN15,TANPI16,TANP2,V1
  !
  XTLABC(1:6)=zero
  IF (XTLTYP.EQ.'CUBI') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(1)
     XTLABC(6)=VARB(1)
  ELSEIF (XTLTYP.EQ.'RECT') THEN
     XTLABC(1)=VARB(1)*XTLREF(1)
     XTLABC(3)=VARB(1)*XTLREF(2)
     XTLABC(6)=VARB(1)*XTLREF(3)
  ELSEIF (XTLTYP.EQ.'TETR') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(1)
     XTLABC(6)=VARB(2)
  ELSEIF (XTLTYP.EQ.'HEXA') THEN
     COS15=COS(PI/TWELVE)
     SIN15=SIN(PI/TWELVE)
     XTLABC(1)=VARB(1)*COS15
     XTLABC(3)=XTLABC(1)
     XTLABC(2)=-VARB(1)*SIN15
     XTLABC(6)=VARB(2)
  ELSEIF (XTLTYP.EQ.'RHOM') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(1)
     XTLABC(6)=VARB(1)
     XTLABC(2)=VARB(2)
     XTLABC(4)=VARB(2)
     XTLABC(5)=VARB(2)
  ELSEIF (XTLTYP.EQ.'OCTA') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(1)
     XTLABC(6)=VARB(1)
     XTLABC(2)=-VARB(1)/5.0
     XTLABC(4)=-VARB(1)/5.0
     XTLABC(5)=-VARB(1)/5.0
  ELSEIF (XTLTYP.EQ.'RHDO') THEN
     TANPI16=PI/16.0
     TANPI16=TAN(TANPI16)
     TANP2=TANPI16**2
     V1=VARB(1)/(ONE+TANP2)
     XTLABC(1)=V1
     XTLABC(3)=V1*(ONE-TANP2)
     XTLABC(6)=V1
     XTLABC(2)= V1*TANPI16*SQRT(TWO)
     XTLABC(4)=-V1*TANP2
     XTLABC(5)= XTLABC(2)
  ELSEIF (XTLTYP.EQ.'ORTH') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(2)
     XTLABC(6)=VARB(3)
  ELSEIF (XTLTYP.EQ.'MONO') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(2)
     XTLABC(6)=VARB(3)
     XTLABC(4)=VARB(4)
  ELSEIF (XTLTYP.EQ.'TRIC') THEN
     XTLABC(1)=VARB(1)
     XTLABC(3)=VARB(2)
     XTLABC(6)=VARB(3)
     XTLABC(2)=VARB(4)
     XTLABC(4)=VARB(5)
     XTLABC(5)=VARB(6)
  ELSE
     CALL WRNDIE(-5,'<SETXTL>',' Unknown crystal type.')
  ENDIF
  RETURN
END SUBROUTINE SETXTL

SUBROUTINE XTLMSR(XUCELL)
  !
  ! This routine sets the command line substitution variables
  !  ?XTLA, ?XTLB, ?XTLC, ?XTLALPHA, ?XTLBETA, ?XTLGAMMA
  ! from the contents of XUCELL
  !
  use chm_kinds
  use param_store, only: set_param
  implicit none
  real(chm_real)      XUCELL(6)
  !
  call set_param('XTLA',XUCELL(1))
  call set_param('XTLB',XUCELL(2))
  call set_param('XTLC',XUCELL(3))
  call set_param('XTLALPHA',XUCELL(4))
  call set_param('XTLBETA',XUCELL(5))
  call set_param('XTLGAMMA',XUCELL(6))
  !
  !==============================================================================
  RETURN
END SUBROUTINE XTLMSR

subroutine incrys
  !-----------------------------------------------------------------------
  !     Initialise some crystal variables.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use image
  implicit none
  !
  xdim   = 0
  xnsymm = 0
  xtltyp = '    '
  xucell(1:6) = zero
  xtlabc(1:6) = zero
  !
  nfreqx = 0
  nkpts  = 0
  nphons = 0
  !
  xnnnb  = 0
  RETURN
END SUBROUTINE INCRYS

