module image_routines_module
 use rtf, only: nrtrs
 implicit none
contains
  
  SUBROUTINE INIMAG(BIMAG,QRESET)
  !
  !     THIS ROUTINE INTIIALIZES THE IMAGE DATA STRUCTURE
  !
  !     Author: Bernie Brooks
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  !
  use image
  use psf
  use datstr,only:freedt_image
  implicit none
  type(imageDataStructure) BIMAG
  logical qreset

  call freedt_image(bimag)
  !
  bimag%nimnbs=0
  bimag%nimnbg=0
  bimag%nimnbx=0
  bimag%nimnb=0
  bimag%niminb=0
  !
  if(.not.qreset) return
  !
  !     define non data structure elements
  !
  ntrans=0
  norot=.false.
  natim=0
  nimgrp=0
  nimhb=0
  nimbon=0
  nimang=0
  nimdih=0
  nimauto=0
  nimimp=0
#if KEY_CMAP==1
  nimcrt=0
#endif 
  limall=.false.
  liminv=.true.
  limcen=.false.
  imxcen=0.0
  imycen=0.0
  imzcen=0.0
  !
  !     Update psf numbers
  natomt=natom
  nrest=nres
  nsegt=nseg
  ngrpt=ngrp
  nbondt=nbond
  nbondt2=nbond
  nthett=ntheta
  nphit=nphi
  npautot=npauto
  nimpht=nimphi
#if KEY_CMAP==1
  ncrtt=ncrterm 
#endif
  MAXIMEX=0
  NIMEXCL=0
  return
end subroutine inimag

SUBROUTINE TRANSO(X,Y,Z,DX,DY,DZ,LFORCE,QECONT,ECONT, &
     NATOM,NTRANS,IMTRNS,IMATPT,IMATTR,NOROT,NATIM &
#if KEY_FLUCQ==1
     ,QFLUC,CG,FQCFOR    & 
#endif
     )
  !-----------------------------------------------------------------------
  !     THIS ROUTINES COMPUTES THE COORDINATES OF ALL IMAGE ATOMS
  !     FROM THE TRANSFORMATION MATRICIES.
  !
  !     By Bernard R. Brooks    9/83
  !
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL LFORCE,QECONT
  real(chm_real) ECONT(*)
  INTEGER NATOM,NTRANS
  real(chm_real)  IMTRNS(*)
  INTEGER IMATPT(*),IMATTR(*)
  LOGICAL NOROT
  INTEGER NATIM
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) CG(*),FQCFOR(*)
#endif 
  !
  INTEGER ITEMP,I,ITRANS,IPT,ISTRT,IEND,J
  !
  ITEMP=NATOM+1
  !
  IF(LFORCE) THEN
     DO I=ITEMP,NATIM
        DX(I)=0.0
        DY(I)=0.0
        DZ(I)=0.0
     ENDDO
#if KEY_FLUCQ==1
     IF (QFLUC) THEN
        DO I=ITEMP,NATIM
           FQCFOR(I)=0.0
        ENDDO
     ENDIF
#endif 
  ENDIF
  !
  IF(QECONT) THEN
     DO I=ITEMP,NATIM
        ECONT(I)=0.0
     ENDDO
  ENDIF
  !
  IF (NOROT) THEN
     DO ITRANS=1,NTRANS
        IPT=(ITRANS-1)*12
        ISTRT=ITEMP
        IEND=IMATPT(ITRANS)
        ITEMP=IEND+1
        DO I=ISTRT,IEND
           J=IMATTR(I)
           X(I)=X(J)+IMTRNS(IPT+10)
           Y(I)=Y(J)+IMTRNS(IPT+11)
           Z(I)=Z(J)+IMTRNS(IPT+12)
#if KEY_FLUCQ==1
           ! Update charges when using FLUCQ
           CG(I)=CG(J)
#endif 
        ENDDO
     ENDDO
  ELSE
     DO ITRANS=1,NTRANS
        IPT=(ITRANS-1)*12
        ISTRT=ITEMP
        IEND=IMATPT(ITRANS)
        ITEMP=IEND+1
        DO I=ISTRT,IEND
           J=IMATTR(I)
           X(I)=X(J)*IMTRNS(IPT+1)+Y(J)*IMTRNS(IPT+2)+ &
                Z(J)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
           Y(I)=X(J)*IMTRNS(IPT+4)+Y(J)*IMTRNS(IPT+5)+ &
                Z(J)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
           Z(I)=X(J)*IMTRNS(IPT+7)+Y(J)*IMTRNS(IPT+8)+ &
                Z(J)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
#if KEY_FLUCQ==1
           ! Update charges when using FLUCQ
           CG(I)=CG(J)
#endif 
        ENDDO
     ENDDO
  ENDIF
  !
  RETURN
END SUBROUTINE TRANSO

#if KEY_BLOCK==1 /*ldm_image*/

SUBROUTINE TRANSRSTP(X,Y,Z,ENVDX,ENVDY,ENVDZ,NATOM,NTRANS,IMTRNS, &
     IMATPT,IMATTR,NOROT,NATIM,IMINV)
  !
  !     THIS ROUTINE COPIES THE FORCES OF IMAGE ATOMS COMING FROM THE RESTRAINING
  !     POTENTIAL INTO FORCES ON PRIMARY ATOMS AFTER ENERGY DETERMINATION.
  !
  !
  use chm_kinds
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),ENVDX(*),ENVDY(*),ENVDZ(*)
  INTEGER NATOM,NTRANS
  real(chm_real) IMTRNS(*)
  INTEGER IMATPT(*),IMATTR(*)
  LOGICAL NOROT
  INTEGER NATIM,IMINV(*)
  !
  real(chm_real) FA1,FA2,FA3,FB1,FB2,FB3
  INTEGER IPT,ITRANS,J,ITEMP,JTRANS,ISTRT,IEND,I
  !
  IPT=0

  ITEMP=NATOM+1
  DO ITRANS=1,NTRANS
     JTRANS=IMINV(ITRANS)
     IPT=(ITRANS-1)*12
     ISTRT=ITEMP
     IEND=IMATPT(ITRANS)
     ITEMP=IEND+1
     !
     DO I=ISTRT,IEND
        !
        FA1=ENVDX(I)
        FA2=ENVDY(I)
        FA3=ENVDZ(I)
        IF(FA1.EQ.0.0) THEN
           IF(FA2.EQ.0.0) THEN
              IF(FA3.EQ.0.0) GOTO 100
           ENDIF
        ENDIF
        IF (NOROT) THEN
           FB1=FA1
           FB2=FA2
           FB3=FA3
        ELSE
           FB1=FA1*IMTRNS(IPT+1)+FA2*IMTRNS(IPT+4)+FA3*IMTRNS(IPT+7)
           FB2=FA1*IMTRNS(IPT+2)+FA2*IMTRNS(IPT+5)+FA3*IMTRNS(IPT+8)
           FB3=FA1*IMTRNS(IPT+3)+FA2*IMTRNS(IPT+6)+FA3*IMTRNS(IPT+9)
        ENDIF
        !
        J=IMATTR(I)
        ENVDX(J)=ENVDX(J)+FB1
        ENVDY(J)=ENVDY(J)+FB2
        ENVDZ(J)=ENVDZ(J)+FB3
        !
100     CONTINUE

     ENDDO
  ENDDO
  !
  ITEMP=NATOM+1
  DO I=ITEMP,NATIM
     ENVDX(I)=0.0
     ENVDY(I)=0.0
     ENVDZ(I)=0.0
  ENDDO
  !
  RETURN
END SUBROUTINE TRANSRSTP
#endif /* (ldm_image)  LDM*/

SUBROUTINE TRANSI(X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM,NTRANS,IMTRNS, &
     IMATPT,IMATTR,NOROT,NATIM,IMINV,IMFORC,IMTORQ &
#if KEY_FLUCQ==1
     ,QFLUC,FQCFOR    & 
#endif
     )
  !
  !     THIS ROUTINE COPIES THE FORCES OF IMAGE ATOMS INTO FORCES ON
  !     PRIMARY ATOMS AFTER ENERGY DETERMINATION.
  !
  !     By Bernard R. Brooks    9/83
  !
  !
  use chm_kinds
  use parallel,only:gcomb
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QECONT
  real(chm_real)  ECONT(*)
  INTEGER NATOM,NTRANS
  real(chm_real) IMTRNS(*)
  INTEGER IMATPT(*),IMATTR(*)
  LOGICAL NOROT
  INTEGER NATIM,IMINV(*)
  real(chm_real)  IMFORC(*),IMTORQ(*)
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 
  !
  real(chm_real) FA1,FA2,FA3,FB1,FB2,FB3
  INTEGER IPT,ITRANS,J,ITEMP,JTRANS,IIPT,JJPT,ISTRT,IEND,I
  !
  IPT=0
  DO ITRANS=1,NTRANS
     DO J=1,3
        IPT=IPT+1
        IMFORC(IPT)=0.0
        IMTORQ(IPT)=0.0
     ENDDO
  ENDDO
  !
  ITEMP=NATOM+1
  DO ITRANS=1,NTRANS
     JTRANS=IMINV(ITRANS)
     IPT=(ITRANS-1)*12
     IIPT=(ITRANS-1)*3
     JJPT=(JTRANS-1)*3
     ISTRT=ITEMP
     IEND=IMATPT(ITRANS)
     ITEMP=IEND+1
     DO I=ISTRT,IEND
        !
        FA1=DX(I)
        FA2=DY(I)
        FA3=DZ(I)
        IF(FA1.EQ.0.0) THEN
           IF(FA2.EQ.0.0) THEN
              IF(FA3.EQ.0.0) GOTO 100
           ENDIF
        ENDIF
        IF (NOROT) THEN
           FB1=FA1
           FB2=FA2
           FB3=FA3
        ELSE
           FB1=FA1*IMTRNS(IPT+1)+FA2*IMTRNS(IPT+4)+FA3*IMTRNS(IPT+7)
           FB2=FA1*IMTRNS(IPT+2)+FA2*IMTRNS(IPT+5)+FA3*IMTRNS(IPT+8)
           FB3=FA1*IMTRNS(IPT+3)+FA2*IMTRNS(IPT+6)+FA3*IMTRNS(IPT+9)
        ENDIF
        !
        IMTORQ(IIPT+1)=IMTORQ(IIPT+1)-Y(I)*FA3+Z(I)*FA2
        IMTORQ(IIPT+2)=IMTORQ(IIPT+2)-Z(I)*FA1+X(I)*FA3
        IMTORQ(IIPT+3)=IMTORQ(IIPT+3)-X(I)*FA2+Y(I)*FA1
        IMFORC(IIPT+1)=IMFORC(IIPT+1)-FA1
        IMFORC(IIPT+2)=IMFORC(IIPT+2)-FA2
        IMFORC(IIPT+3)=IMFORC(IIPT+3)-FA3
        !
        J=IMATTR(I)
        IMTORQ(JJPT+1)=IMTORQ(JJPT+1)+Y(J)*FB3-Z(J)*FB2
        IMTORQ(JJPT+2)=IMTORQ(JJPT+2)+Z(J)*FB1-X(J)*FB3
        IMTORQ(JJPT+3)=IMTORQ(JJPT+3)+X(J)*FB2-Y(J)*FB1
        IMFORC(JJPT+1)=IMFORC(JJPT+1)+FB1
        IMFORC(JJPT+2)=IMFORC(JJPT+2)+FB2
        IMFORC(JJPT+3)=IMFORC(JJPT+3)+FB3
        DX(J)=DX(J)+FB1
        DY(J)=DY(J)+FB2
        DZ(J)=DZ(J)+FB3
        !
        IF(QECONT) ECONT(J)=ECONT(J)+ECONT(I)
100     CONTINUE
     ENDDO
#if KEY_FLUCQ==1
     IF (QFLUC) THEN
        DO I=ISTRT,IEND
           J=IMATTR(I)
           FQCFOR(J)=FQCFOR(J)+FQCFOR(I)
        ENDDO
     ENDIF
#endif 
  ENDDO
  !
#if KEY_PARALLEL==1
  !     Sum IMFORC and IMTORQ on parallel machines
  CALL GCOMB(IMFORC,3*NTRANS)
  CALL GCOMB(IMTORQ,3*NTRANS)
#endif 
  ITEMP=NATOM+1
  DO I=ITEMP,NATIM
     DX(I)=0.0
     DY(I)=0.0
     DZ(I)=0.0
  ENDDO
#if KEY_FLUCQ==1
  IF (QFLUC) THEN
     DO I=ITEMP,NATIM
        FQCFOR(I)=0.0
     ENDDO
  ENDIF
#endif 
  !
  RETURN
END SUBROUTINE TRANSI

SUBROUTINE REIMAG(BIMAG,MAXJMB,MXJMBG)
  !
  !     THIS ROUTINE RESIZES THE IMAGE DATA STRUCTURE
  !
  !     By Bernard R. Brooks    9/83
  !

  use chm_kinds
  use chm_types
  use dimens_fcm
  use image
  use psf
  use datstr, only: useddt_image, imgrow
#ifdef KEY_RESIZE    
  use resize,only: resize_bimag
#endif  
  implicit none
  !
   type(imageDataStructure) BIMAG
   INTEGER MAXJMB,MXJMBG
  !
  INTEGER L
#ifdef KEY_RESIZE    
    integer:: i0
    i0=2
#endif
  !
  IF (.not.allocated(BIMAG%IMATPT)) CALL INIMAG(BIMAG,.FALSE.)
  !
  IF (MAXJMB.EQ.0 .AND. MXJMBG.EQ.0) THEN
     !
     allocate(bimag%IMATPT(MAXTRN))
     bimag%IMATPT(1:MAXTRN) = 0
#ifdef KEY_RESIZE
     if (.not.(allocated(bimag%IMATTR))) then
        allocate(bimag%IMATTR(i0))
        bimag%IMATTR = 0
     endif
     if (.not.(allocated(bimag%IMINB))) then
        allocate(bimag%IMINB(i0))
        bimag%IMINB = 0
     endif
#if KEY_IMCUBES==1
     if (.not.(allocated(bimag%IMIBLO))) then
        allocate(bimag%IMIBLO(i0*2))
        bimag%IMIBLO = 0
     endif
#else /**/
     if (.not.(allocated(bimag%IMIBLO))) then
        allocate(bimag%IMIBLO(i0))
        bimag%IMIBLO = 0
     endif
#endif 
     if (.not.(allocated(bimag%IMING))) then
        allocate(bimag%IMING(I0))
        bimag%IMING = 0
     endif
     if (.not.(allocated(bimag%IMIGLO))) then
        allocate(bimag%IMIGLO(I0))
        bimag%IMIGLO = 0
     endif
     call resize_bimag(bimag)
#else /* KEY_RESIZE */
     allocate(bimag%IMATTR(MAXAIM))
     bimag%IMATTR(1:MAXAIM) = 0
     allocate(bimag%IMINB(MAXAIM))
     bimag%IMINB(1:MAXAIM) = 0
#if KEY_IMCUBES==1
     allocate(bimag%IMIBLO(MAXAIM*2))
     bimag%IMIBLO(1:MAXAIM*2) = 0
#else /**/
     allocate(bimag%IMIBLO(MAXAIM))
     bimag%IMIBLO(1:MAXAIM) = 0
#endif 
     allocate(bimag%IMING(MAXGRP))
     bimag%IMING(1:MAXGRP) = 0
     allocate(bimag%IMIGLO(MAXGRP))
     bimag%IMIGLO(1:MAXGRP) = 0
#endif /* KEY_RESIZE */
     allocate(bimag%IMCENF(NATOM))
     bimag%IMCENF(1:NATOM) = 0
     !
     !
  ELSE
     !
     ! Note: allocate more space as needed, but do not reduce
     ! space. This limits the number of times that this data
     ! structure is copied (as NATIM fluctuates). - BRB - July,1995
     ! When initially allocating space, get 10% more than needed
     ! for image atom and image group based arrays.
     !
     L = 0
     if (allocated(bimag%IMBLO)) L = size(bimag%IMBLO)
     IF(L.EQ.0) THEN
#if KEY_IMCUBES==1
        L=NATIM*2.2+10
#else /**/
        L=NATIM*1.1+10
#endif 
     ELSE
#if KEY_IMCUBES==1
        L=NATIM*2
#else /**/
        L=NATIM
#endif 
     ENDIF
     call IMGROW(bimag%IMBLO, L)

     L=MAXJMB
     call IMGROW(bimag%IMJNB, L)

     L = 0
     if (allocated(bimag%IMBLOS)) L = size(bimag%IMBLOS)
     IF(L.EQ.0) THEN
#if KEY_IMCUBES==1
        L=NATIM*2.2+10
#else /**/
        L=NATIM*1.1+10
#endif 
     ELSE
#if KEY_IMCUBES==1
        L=NATIM*2
#else /**/
        L=NATIM
#endif 
     ENDIF
     call IMGROW(bimag%IMBLOS, L)

     L = 0
     if (allocated(bimag%IMJNBS)) L = size(bimag%IMJNBS)
     IF(L.EQ.0) THEN
#if KEY_IMCUBES==1
        L=NATIM*2.2+10
#else /**/
        L=NATIM*1.1+10
#endif 
     ELSE
        L=NATIM
     ENDIF
     call IMGROW(bimag%IMJNBS, L)

     L = 0
     if (allocated(bimag%IMBLOG)) L = size(bimag%IMBLOG)
     IF(L.EQ.0) THEN
        L=NIMGRP*1.1+10
     ELSE
        L=NIMGRP
     ENDIF
     call IMGROW(bimag%IMBLOG, L)

     L=MXJMBG
     call IMGROW(bimag%IMJNBG, L)

     L = 0
     if (allocated(bimag%IMBLOX)) L = size(bimag%IMBLOX)
     IF(L.EQ.0) THEN
        L=NIMGRP*1.1+10
     ELSE
        L=NIMGRP
     ENDIF
     call IMGROW(bimag%IMBLOX, L)

     L = 0
     if (allocated(bimag%IMJNBX)) L = size(bimag%IMJNBX)
     IF(L.EQ.0) THEN
        L=NIMGRP*1.1+10
     ELSE
        L=NIMGRP
     ENDIF
     call IMGROW(bimag%IMJNBX, L)

  ENDIF
  RETURN
END SUBROUTINE REIMAG

SUBROUTINE CLIMAG(BIMAGX)
  !
  !     THIS ROUTINE CLEARS THE ATOMS FROM THE IMAGE DATA STRUCTURE
  !
  !     By Bernard R. Brooks    2/92
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  use image
  use psf
  implicit none
  !
   type(imageDataStructure) bimagx
  !
  !     Update psf numbers
  NATOMT=NATOM
  NREST=NRES
  NSEGT=NSEG
  NGRPT=NGRP
  NBONDT=NBOND
  NBONDT2=NBOND ! hwm 220722
  NTHETT=NTHETA
  NPHIT=NPHI
  NIMPHT=NIMPHI
#if KEY_CMAP==1
  !sbbug      NCRTT=NIMCRT
  NCRTT=NCRTERM
#endif 
  NATIM=0
  NIMGRP=0
  NIMHB=0
  NIMBON=0
  NIMANG=0
  NIMDIH=0
  NIMAUTO=0
  NIMIMP=0
#if KEY_CMAP==1
  NIMCRT=0
#endif 
  !
  !.....25-Jan-2007 fix to clear image centering upon PSF modification
  IF(LIMCEN) CALL WRNDIE(0,'<CLIMAG>', &
       'PSF modified with image centering active: Now off.')
  LIMCEN=.FALSE.
  !
  IF(NTRANS.EQ.0) RETURN
  !
  !     Clear the image PSF elements when the PSF changes.
  !
  if(allocated(bimagx%IMBLO)) deallocate(bimagx%IMBLO)
  allocate(bimagx%IMBLO(NATOM))
  if(allocated(bimagx%IMJNB)) deallocate(bimagx%IMJNB)
  if(allocated(bimagx%IMBLOS)) deallocate(bimagx%IMBLOS)
  allocate(bimagx%IMBLOS(NATOM))
  if(allocated(bimagx%IMJNBS)) deallocate(bimagx%IMJNBS)
  allocate(bimagx%IMJNBS(NATOM))
  if(allocated(bimagx%IMBLOG)) deallocate(bimagx%IMBLOG)
  allocate(bimagx%IMBLOG(NGRP))
  if(allocated(bimagx%IMBLOG)) deallocate(bimagx%IMBLOG)
  if(allocated(bimagx%IMBLOX)) deallocate(bimagx%IMBLOX)
  allocate(bimagx%IMBLOX(NGRP))
  if(allocated(bimagx%IMJNBX)) deallocate(bimagx%IMJNBX)
  allocate(bimagx%IMJNBX(NGRP))
  if(allocated(bimagx%IMCENF)) deallocate(bimagx%IMCENF)
  allocate(bimagx%IMCENF(NATOM))

  !
  BIMAGX%NIMNBS=0
  BIMAGX%NIMNBG=0
  BIMAGX%NIMNBX=0
  BIMAGX%NIMNB=0
  BIMAGX%NIMINB=0
  BIMAGX%NIMING=0
  !
  bimagx%IMATPT(1:NTRANS) = 0
  bimagx%IMIBLO(1:NATOM ) = 0
  !
  NATIM=NATOM
  NIMGRP=NGRP
  NIMHB=0
  NIMBON=0
  NIMANG=0
  NIMDIH=0
  NIMAUTO=0
  NIMIMP=0
#if KEY_CMAP==1
  NIMCRT=0
#endif 
  !
  RETURN
END SUBROUTINE CLIMAG

SUBROUTINE IMPATC(COMLYN,COMLEN)
  !
  !     Process the image patch command.
  !
  !     patch image internal coordinate list for specific interactions
  !     handles patches through the rtf patch residues
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use memory
  use image
  use psf
  use stream
  use chutil,only:getrsn
  use code,only:mustup
  use upimag_util,only: IMAUTGEN
  use string
  implicit none

  ! . Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN

  ! . Local variables.
  integer,allocatable,dimension(:,:) :: IMATST

  !  MXNPA = maximum number of residues allowed in a IMPAtch command
  INTEGER MXNPA
  PARAMETER(MXNPA=99)
  !RCZ

  character(len=8) pres, PPIMA, PPRES, PPSEG
  character(len=4) WINIT
  INTEGER     I, IPRES(MXNPA), IMPRES(MXNPA), J, K
  LOGICAL     CMPLTD, LSETIC, LWARN, QAUTOG, QREMDUP, QSORT
  !
  LSETIC=(INDXA(COMLYN,COMLEN,'SETU').GT.0)
  LWARN=(INDXA(COMLYN,COMLEN,'WARN').GT.0)

  QAUTOG=.NOT.(INDXA(COMLYN,COMLEN,'NOAU').GT.0)
  IF(INDXA(COMLYN,COMLEN,'AUTO').GT.0) QAUTOG=.TRUE.
  QREMDUP=.NOT.(INDXA(COMLYN,COMLEN,'NORE').GT.0)
  IF(INDXA(COMLYN,COMLEN,'REMD').GT.0) QREMDUP=.TRUE.
  QSORT=(INDXA(COMLYN,COMLEN,'SORT').GT.0)
  IF(QREMDUP) QSORT=.TRUE.

  PRES=NEXTA8(COMLYN,COMLEN)
  K=0
  CMPLTD=.TRUE.
100 IF (COMLEN.GT.0 .AND. K.LT.MXNPA) THEN
     PPIMA=NEXTA8(COMLYN,COMLEN)
     PPSEG=NEXTA8(COMLYN,COMLEN)
     PPRES=NEXTA8(COMLYN,COMLEN)
     CALL TRIME(COMLYN,COMLEN)
     K=K+1
     IPRES(K)=GETRSN(PPSEG,PPRES,' ',SEGID,RESID, &
          ATYPE,IBASE,NICTOT,NSEG)
     IF (PPIMA(1:4).EQ.'PRIM') THEN
        IMPRES(K)=0
     ELSE
        IMPRES(K)=-1
        DO I=1,NTRANS
           IF(IMNAME(I).EQ.PPIMA) IMPRES(K)=I
        ENDDO
     ENDIF
     CMPLTD=CMPLTD .AND. (IPRES(K).NE.-1) .AND. (IMPRES(K).NE.-1)
     GOTO 100
  ENDIF
  IF (CMPLTD) THEN
     call chmalloc('images.src','IMPATC','IMATST',NATOM,NTRANS,intg=IMATST)

     CALL IMPAT2(PRES,IPRES,IMPRES,K,LWARN,LSETIC, &
          NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP, &
#if KEY_CMAP==1
          NIMCRT, &           
#endif
          NATOM,NTRANS,BIMAG%IMATPT,BIMAG%IMATTR,IMATST)

     call chmdealloc('images.src','IMPATC','IMATST',NATOM,NTRANS,intg=IMATST)
     WINIT='RESE'
     J=4
     CALL GTNBCT(WINIT,J,BNBND)
     MUSTUP = .TRUE.

     CALL IMAUTGEN(QAUTOG,QREMDUP,QSORT)


     IF(PRNLEV.GE.2) WRITE(OUTU,701) NATOMT-NATOM,NIMBON,NIMANG, &
                                     NIMDIH,NIMAUTO,NIMIMP, &
#if KEY_CMAP==1
                                     NIMCRT, &
#else /**/
                                     0, &
#endif 
                                     NIMEXCL

701  FORMAT(10X,'CURRENT IMAGE INTERNAL COORDINATE COUNTS:'/ &
     10X,'NATOM   NBOND   NTHETA  NPHI    NIMAUTO NIMPHI  NIMCRT  NIMEXCL'/7X,10I8)
  ELSE
     CALL WRNDIE(-3,'<CHARMM>','BAD RESIDUE SPECIFIED FOR PATCHING')
  ENDIF
  !
  RETURN
END SUBROUTINE IMPATC

INTEGER FUNCTION IMATOM(IPRES,IMPRES,ATOM,ATYPE,IBASE, &
     NPRES,LWARN,NATOM,IMATST)
  !
  !     THIS ROUTINE RETURNS THE ATOM NUMBER GIVEN A RESIDUE NUMBER
  !     PRECEDING THE ATOM NAME. THE RESIDUE NUMBER IS FOUND BY A LOOKUP
  !     INTO IPRES.
  !
  !     By Bernard R. Brooks    9/83
  !
  use stream, only: wrnlev, outu
  use string, only: spliti

  implicit none
  !
  INTEGER MXNPA
  PARAMETER(MXNPA=99)
  INTEGER IPRES(MXNPA),IMPRES(MXNPA)
  character(len=*) ATOM,ATYPE(*)
  INTEGER IBASE(*)
  INTEGER NPRES,NATOM
  LOGICAL LWARN
  INTEGER IMATST(NATOM,*)
  !
  INTEGER IRS,J,ITRANS,ISTA,ISTO,IFND,I
  INTEGER MARK
  character(len=8) WORD,WORDX
  character(len=1) CHAR
  !     character(len=1 CHAR,IN(9)
  !     DATA IN/'1','2','3','4','5','6','7','8','9'/
  DATA    MARK/-99999999/
  !
  WORD=ATOM
  CHAR=WORD(1:1)
  !
  IF(CHAR.EQ.'*') THEN
     WORDX=WORD(2:)
     WORD=WORDX
     CHAR=WORD(1:1)
  ENDIF
  !
  IRS=0
  I=LEN(WORD)
  ! SPLITI (from util/string.src)
  ! splits string WORD='12ABC' into integer J=12 and string WORDX='ABC'
  ! I is the lenght of resulting string WORDX
  CALL SPLITI(WORD,J,WORDX,I)
  IF(J.NE.0) THEN
     IRS=IPRES(J)
     ITRANS=IMPRES(J)
     WORD=WORDX
  ELSE
     IRS=IPRES(1)
     ITRANS=IMPRES(1)
  ENDIF
  !C      DO J=1,NPRES
  !C        IF(CHAR.EQ.IN(J)) THEN
  !C          IRS=IPRES(J)
  !C          ITRANS=IMPRES(J)
  !C        ENDIF
  !C      ENDDO
  !C      IF (IRS.EQ.0) THEN
  !C        IRS=IPRES(1)
  !C        ITRANS=IMPRES(1)
  !C      ELSE
  !C        WORDX=WORD(2:)
  !C        WORD=WORDX
  !C        CHAR=WORD(1:1)
  !C      ENDIF
  !
  ISTA=IBASE(IRS)+1
  ISTO=IBASE(IRS+1)
  IFND=0
  DO I=ISTA,ISTO
     IF (WORD == ATYPE(I)) THEN
        IFND=I
        GOTO 100
     ENDIF
  ENDDO
  !     Begin Procedure ATOM-NOT-FOUND
  IF(LWARN .AND. WRNLEV.GE.2) WRITE(OUTU,13) WORD,IRS
13 FORMAT(' *** IN IMATOM *** ''',A4,''' ATOM TYPE NOT FOUND FOR ', &
       'RESIDUE',I5)
  IMATOM=MARK
  RETURN
  !     End Procedure ATOM-NOT-FOUND
  !
100 CONTINUE
  IF (ITRANS.EQ.0) THEN
     IMATOM=IFND
  ELSE
     !       THIS IS AN IMAGE ATOM
     I=IMATST(IFND,ITRANS)
     IF(I.LE.0) THEN
        IF(LWARN .AND. WRNLEV.GE.2) WRITE(OUTU,105) WORD,IRS,ITRANS
105     FORMAT(' *** IN IMATOM *** ''',A4,''' ATOM FOUND FOR ' &
             //'RESIDUE',I5,/ &
             15X,' BUT HAS NO IMAGE FOR TRANSFORMATION',I4)
        I=MARK
     ENDIF
     IMATOM=I
  ENDIF
  !
  RETURN
END FUNCTION IMATOM

SUBROUTINE IMPAT2(PRES,IPRES,IMPRES,NPRES,LWARN,LSETIC, &
     NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP, &
#if KEY_CMAP==1
  NIMCRT, &      
#endif
  NATOMX,NTRANS,IMATPT,IMATTR,IMATST)
  !
  !     GENERAL PATCHING ROUTINE FOR IMAGES.
  !     PATCHING DATA READ FROM RTF.
  !     ANY ITEM NOT PRESENT WILL BE ADDED. ERRORS ARE FATAL UNLESS LWARN SET.
  !
  !     By Bernard R. Brooks    9/83
  !
  use intcor_module,only: reintc_new,icr_struct
  use intcor2,only: puticel
  use dimens_fcm
  use exfunc
  use consta
  use psf
  use bases_fcm
  use rtf,only:nrtrs, aa, rtrtyp, nic, mib, mjb, mit, mjt, mkt, mip, mjp, &
    mkp, mlp, mim, mjm, mkm, mlm, &
#if KEY_CMAP==1
    mi1ct, mj1ct, mk1ct, ml1ct, mi2ct, mj2ct, mk2ct, ml2ct, & 
#endif
    bari, barj, bark, barl, bart, icb1, icb2, icth1, icth2, icphi, &
    mautob, mautot, mautop
  use stream
  use string
  use machutil,only:die
#ifdef KEY_RESIZE
  use resize,only: resize_psf
#endif

  implicit none
  !
  INTEGER MXNPA
  PARAMETER(MXNPA=99)
  character(len=*) PRES
  INTEGER IPRES(MXNPA),IMPRES(MXNPA),NPRES
  LOGICAL LWARN,LSETIC
  INTEGER NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP,NATOMX,NTRANS
#if KEY_CMAP==1
  INTEGER NIMCRT
#endif 
  INTEGER IMATPT(*),IMATTR(*),IMATST(NATOMX,NTRANS)
  !
  !
  INTEGER ITEMP,ITRANS,IR,I,J,IPATC,IPT,K,N,II,JJ,KK,LL
#if KEY_CMAP==1
  INTEGER II1,JJ1,KK1,LL1,II2,JJ2,KK2,LL2
#endif 
  LOGICAL FOUND,OK
  integer wtlen
  !
  !
  !     CHECK NUMBERS FOR CONSISTANCY
  OK=(NBOND+NIMBON.EQ.NBONDT).AND.(NTHETA+NIMANG.EQ.NTHETT).AND. &
       (NPHI+NIMDIH.EQ.NPHIT) .AND. (NPAUTO+NIMAUTO.EQ.NPAUTOT) &
       .AND. (NIMPHI+NIMIMP.EQ.NIMPHT) &
#if KEY_CMAP==1
       .AND.(NCRTERM+NIMCRT.EQ.NCRTT)    
#endif
#if KEY_CMAP==0
  ;                                 
#endif
  IF (.NOT.OK) CALL WRNDIE(-3,'<IMPATC>', &
       'IMAGE INTERNAL COORDINATES DONT MATCH TOTALS')
  !
  !     SET UP IMAGE ATOM POINTER ARRAY (IMATST)
  !
#ifdef KEY_RESIZE
  if (ntrans .gt. 1) then ! add space for additional variables
     ii1=ntrans*npres
  else
     ii1=2*npres
  endif
  call resize_psf('images.F90','IMPAT2','NBOND',nbondt+ii1, .true.)
  call resize_psf('images.F90','IMPAT2','NTHETA',nthett+ii1, .true.)
  call resize_psf('images.F90','IMPAT2','NPHI',nphit+ii1, .true.)
  call resize_psf('images.F90','IMPAT2','NIMPHI',nimpht+ii1, .true.)
#endif

  ITEMP=NATOM
  DO ITRANS=1,NTRANS
     DO IR=1,NATOM
        IF (ITEMP.LT.IMATPT(ITRANS)) THEN
           I=IMATTR(ITEMP+1)
        ELSE
           I=0
        ENDIF
        IF (IR.EQ.I) THEN
           ITEMP=ITEMP+1
           IMATST(IR,ITRANS)=ITEMP
        ELSE
           IMATST(IR,ITRANS)=0
        ENDIF
     ENDDO
  ENDDO
  !
  !     FIND PATCH RESIDUE NUMBER
  !
  FOUND=.FALSE.
  DO J=1,NRTRS
     IF(PRES.EQ.AA(J)) THEN
        IF (RTRTYP(J).NE.1) THEN
           if(WRNLEV.GE.2) then
              wtlen=len(pres)
              call trime(pres,wtlen)
              WRITE(OUTU,231) PRES(1:wtlen)
           endif
           CALL DIE
        ELSE
           IPATC=J
           FOUND=.TRUE.
           exit ! hwm2206: Don't go further           
        ENDIF
     ENDIF
  ENDDO
  IF (.NOT.FOUND) THEN
     if(WRNLEV.GE.2) then
        wtlen=len(pres)
        call trime(pres,wtlen)
        WRITE(OUTU,233) PRES(1:wtlen)
     endif
     CALL DIE
  ENDIF
231 FORMAT(/' ***** ERROR in IMPATC ***** Residue ''',A, &
       ''' has the wrong patch type')
233 FORMAT(/' ***** ERROR in IMPATC ***** Residue ''',A,''' was', &
       ' not found.')
  !
  K=IPATC
  !
  !     PATCH ATOM SEQUENCE.
  !     NOTE: NO ATOMS MAY BE DELETED OR ADDED. NO ATOM CHARACTERISTIC
  !     MAY BE MODIFIED. USE THE MAIN PATCH ROUTINE FOR SUCH THINGS.
  !     ALL ATOM PATCH SPECS WILL BE IGNORED.
  !
  IPT=0
  IF(K.GT.1) IPT=NIC(1,K-1)
  IF(NIC(1,K).GT.IPT) CALL WRNDIE(-3,'<IMPATC>', &
       'ATOMS MAY NOT BE MODIFIED BY IMAGE PATCHES')
  !
  !     ADD IN INTERNAL COORDINATES USING EXISTING ATOM SEQUENCE
  !
  !     PUT IN BONDS
  IPT=0
  IF(K.GT.1) IPT=NIC(2,K-1)
  N=NIC(2,K)-IPT
  DO I=1,N
     IPT=IPT+1
     II=IMATOM(IPRES,IMPRES,MIB(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ=IMATOM(IPRES,IMPRES,MJB(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     IF (II.GT.0.AND.JJ.GT.0) THEN
        NIMBON=NIMBON+1
        NBONDT=NBONDT+1
        NBONDT2=NBONDT
#ifndef KEY_RESIZE        
        IF(NBONDT.GT.MAXB)THEN
           CALL WRNDIE(-3,'<IMPAT2>', &
                'Maximum number of bonds exceeded')
           return
        ENDIF
#endif        
        IB(NBONDT)=MIN(II,JJ)
        JB(NBONDT)=MAX(II,JJ)
        ICBAUTO(NBOND)=MAUTOB(IPT) ! save autogen flag
     ELSE
        IF(WRNLEV.GE.2) WRITE(OUTU,140) AA(K),MIB(IPT),MJB(IPT)
140     FORMAT(' ** WARNING ** BOND NOT FOUND FOR RESIDUE ',A4, &
             '.'/,' ATOMS',2(1X,'"',A4,'"'),' WERE REQUESTED.')
        IF (.NOT.LWARN) CALL DIE
     ENDIF
  ENDDO
  !
  !     PUT IN ANGLES
  IPT=0
  IF(K.GT.1) IPT=NIC(3,K-1)
  N=NIC(3,K)-IPT
  !
  DO I=1,N
     IPT=IPT+1
     II=IMATOM(IPRES,IMPRES,MIT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ=IMATOM(IPRES,IMPRES,MJT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     KK=IMATOM(IPRES,IMPRES,MKT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     IF (II.GT.0.AND.JJ.GT.0.AND.KK.GT.0) THEN
        ! how this residue should do autogenerate
        IF(iand(MAUTOT(IPT),1).gt.0) THEN
           ! save any specified angles for this or future autogens
           NIMAUTO=NIMAUTO+1
           NPAUTOT=NPAUTOT+1
           IPAUTO(NPAUTOT)=II
           JPAUTO(NPAUTOT)=JJ
           KPAUTO(NPAUTOT)=KK
           LPAUTO(NPAUTOT)=0
           ICPAUTO(NPAUTOT)=MAUTOT(IPT)
        ELSE
           NIMANG=NIMANG+1
           NTHETT=NTHETT+1
           IT(NTHETT)=MIN(II,KK)
           JT(NTHETT)=JJ
           KT(NTHETT)=MAX(II,KK)
!  write(outu,*) ' Adding image angle: NTHETT,II,JJ,KK:',NTHETT,II,JJ,KK

        ENDIF
     ELSE
        IF(WRNLEV.GE.2) WRITE(OUTU,150) AA(K),MIT(IPT),MJT(IPT), &
             MKT(IPT)
150     FORMAT(' ** WARNING ** ANGLE NOT FOUND FOR RESIDUE ',A4, &
             '.'/,' ATOMS',3(1X,'"',A4,'"'),' WERE REQUESTED.')
        IF (.NOT.LWARN) CALL DIE
     ENDIF
  ENDDO
  !
  !     PUT IN DIHEDRALS
  IPT=0
  IF(K.GT.1) IPT=NIC(4,K-1)
  N=NIC(4,K)-IPT
  !
  DO I=1,N
     IPT=IPT+1
     II=IMATOM(IPRES,IMPRES,MIP(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ=IMATOM(IPRES,IMPRES,MJP(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     KK=IMATOM(IPRES,IMPRES,MKP(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     LL=IMATOM(IPRES,IMPRES,MLP(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     IF (II.GT.0.AND.JJ.GT.0.AND.KK.GT.0.AND.LL.GT.0) THEN
        ! how this residue should do autogenerate
        IF(iand(MAUTOP(IPT),1).gt.0) THEN  ! dihedral autogenerate
           ! save any specified dihedrals for this or future autogens
           NIMAUTO=NIMAUTO+1
           NPAUTOT=NPAUTOT+1
           IPAUTO(NPAUTOT)=II
           JPAUTO(NPAUTOT)=JJ
           KPAUTO(NPAUTOT)=KK
           LPAUTO(NPAUTOT)=LL
           ICPAUTO(NPAUTOT)=MAUTOP(IPT)
        ELSE
           NIMDIH=NIMDIH+1
           NPHIT=NPHIT+1
           IP(NPHIT)=II
           JP(NPHIT)=JJ
           KP(NPHIT)=KK
           LP(NPHIT)=LL
        ENDIF
     ELSE
        IF(WRNLEV.GE.2) WRITE(OUTU,160) AA(K),MIP(IPT),MJP(IPT), &
             MKP(IPT),MLP(IPT)
160     FORMAT(' ** WARNING ** DIHEDRAL NOT FOUND FOR RESIDUE ',A4, &
             '.'/,' ATOMS',4(1X,'"',A4,'"'),' WERE REQUESTED.')
        IF (.NOT.LWARN) CALL DIE
     ENDIF
  ENDDO
  !
  !     PUT IN IMPROPER DIHEDRALS
  IPT=0
  IF(K.GT.1) IPT=NIC(5,K-1)
  N=NIC(5,K)-IPT
  DO I=1,N
     IPT=IPT+1
     II=IMATOM(IPRES,IMPRES,MIM(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ=IMATOM(IPRES,IMPRES,MJM(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     KK=IMATOM(IPRES,IMPRES,MKM(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     LL=IMATOM(IPRES,IMPRES,MLM(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     IF (II.GT.0.AND.JJ.GT.0.AND.KK.GT.0.AND.LL.GT.0) THEN
        NIMIMP=NIMIMP+1
        NIMPHT=NIMPHT+1
        IM(NIMPHT)=II
        JM(NIMPHT)=JJ
        KM(NIMPHT)=KK
        LM(NIMPHT)=LL
     ELSE
        IF(WRNLEV.GE.2) WRITE(OUTU,170) AA(K),MIM(IPT),MJM(IPT), &
             MKM(IPT),MLM(IPT)
170     FORMAT(' ** WARNING ** IMPROPER NOT FOUND FOR RESIDUE ',A4, &
             '.'/,' ATOMS',4(1X,'"',A4,'"'),' WERE REQUESTED.')
        IF (.NOT.LWARN) CALL DIE
     ENDIF
  ENDDO

#if KEY_CMAP==1
  !     PUT IN CROSS TERMS
  IPT=0
  IF(K.GT.1) IPT=NIC(11,K-1)
  N=NIC(11,K)-IPT
  DO I=1,N
     IPT=IPT+1
     II1=IMATOM(IPRES,IMPRES,MI1CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ1=IMATOM(IPRES,IMPRES,MJ1CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     KK1=IMATOM(IPRES,IMPRES,MK1CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     LL1=IMATOM(IPRES,IMPRES,ML1CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     II2=IMATOM(IPRES,IMPRES,MI2CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     JJ2=IMATOM(IPRES,IMPRES,MJ2CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     KK2=IMATOM(IPRES,IMPRES,MK2CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     LL2=IMATOM(IPRES,IMPRES,ML2CT(IPT),ATYPE,IBASE, &
          NPRES,LWARN,NATOM,IMATST)
     IF (II1.GT.0.AND.JJ1.GT.0.AND.KK1.GT.0.AND.LL1.GT.0 &
          .AND.II2.GT.0.AND.JJ2.GT.0.AND.KK2.GT.0.AND.LL2.GT.0) &
          THEN
        NIMCRT=NIMCRT+1
        NCRTT=NCRTT+1
        I1CT(NCRTT)=II1
        J1CT(NCRTT)=JJ1
        K1CT(NCRTT)=KK1
        L1CT(NCRTT)=LL1
        I2CT(NCRTT)=II2
        J2CT(NCRTT)=JJ2
        K2CT(NCRTT)=KK2
        L2CT(NCRTT)=LL2
     ELSE
        IF(WRNLEV.GE.2) WRITE(OUTU,175) AA(K), &
             MI1CT(IPT),MJ1CT(IPT),MK1CT(IPT),ML1CT(IPT), &
             MI2CT(IPT),MJ2CT(IPT),MK2CT(IPT),ML2CT(IPT)
175     FORMAT(' ** WARNING ** CROSSTERM NOT FOUND FOR RESIDUE ',A4, &
             '.'/,' ATOMS',4(1X,'"',A4,'"'),/,4(1X,'"',A4,'"'), &
             ' WERE REQUESTED.')
        IF (.NOT.LWARN) CALL DIE
     ENDIF
  ENDDO
#endif 

  !
  !     PUT IN DONORS
  !     DONORS MAY NOT BE ADDED.
  IPT=0
  IF(K.GT.1) IPT=NIC(7,K-1)
  IF(NIC(7,K).GT.IPT) &
       CALL WRNDIE(-1,'<IMPATC>','DONORS MAY NOT BE ADDED FOR IMAGES')
  !
  !     PUT IN ACCEPTORS
  !     ACCEPTORS MAY NOT BE ADDED
  IPT=0
  IF(K.GT.1) IPT=NIC(8,K-1)
  IF(NIC(8,K).GT.IPT) CALL WRNDIE(-1,'<IMPATC>', &
       'ACCEPTORS MAY NOT BE ADDED FOR IMAGES')
  !
  !     PUT IN BUILD/IC ELEMENTS
  IF(LSETIC) THEN
     IPT=0
     IF(K.GT.1) IPT=NIC(9,K-1)
     N=NIC(9,K)-IPT
     DO I=1,N
        IPT=IPT+1
        II=IMATOM(IPRES,IMPRES,BARI(IPT),ATYPE,IBASE, &
             NPRES,LWARN,NATOM,IMATST)
        JJ=IMATOM(IPRES,IMPRES,BARJ(IPT),ATYPE,IBASE, &
             NPRES,LWARN,NATOM,IMATST)
        KK=IMATOM(IPRES,IMPRES,BARK(IPT),ATYPE,IBASE, &
             NPRES,LWARN,NATOM,IMATST)
        LL=IMATOM(IPRES,IMPRES,BARL(IPT),ATYPE,IBASE, &
             NPRES,LWARN,NATOM,IMATST)
        IF (II.GT.0.AND.JJ.GT.0.AND.KK.GT.0.AND.LL.GT.0) THEN
           !
           IF(icr_struct%INTLEN.LE.icr_struct%LENIC) THEN
              IR=icr_struct%LENIC+200
              CALL reintc_new(IR,icr_struct)
           endif
           icr_struct%LENIC=icr_struct%LENIC+1
           CALL PUTICEL(II,JJ,KK,LL,BART(IPT),ICB1(IPT),ICB2(IPT), &
                ICTH1(IPT),ICTH2(IPT),ICPHI(IPT),icr_struct%LENIC, &
                icr_struct%B1ic,icr_struct%B2ic, &
                icr_struct%T1ic,icr_struct%T2ic, &
                icr_struct%PIC, icr_struct%IAR, &
                icr_struct%JAR, icr_struct%KAR, &
                icr_struct%LAR, icr_struct%TAR)
        ELSE
           IF(WRNLEV.GE.2) WRITE(OUTU,210) AA(K),BARI(IPT),BARJ(IPT), &
                BARK(IPT),BARL(IPT)
210        FORMAT(' ** WARNING ** IC NOT FOUND FOR RESIDUE ',A4, &
                '.'/,' ATOMS',4(1X,'"',A4,'"'),' WERE REQUESTED.')
           IF (.NOT.LWARN) CALL DIE
        ENDIF
     ENDDO
     !
  ENDIF
  !
  RETURN
END SUBROUTINE IMPAT2



SUBROUTINE IMSPEC(COMLYN,COMLEN,IMCENF, &
     LIMCEN,IMXCEN,IMYCEN,IMZCEN,NATOM,X,Y,Z,WMAIN)
  !
  !     THIS ROUTINE PROCESSES SPECIFICATIONS TO MODIFY THE CENTERING
  !     PROCESS IN THE IMAGE UPDATE
  !
  use chm_kinds
  use exfunc
  use stream
  use memory
  use select
  use string

  implicit none
  !
  character(len=*) COMLYN
  INTEGER COMLEN
  INTEGER,allocatable,dimension(:) :: ISLCT
  integer :: IMCENF(*)
  LOGICAL LIMCEN
  real(chm_real) IMXCEN,IMYCEN,IMZCEN
  INTEGER NATOM
  real(chm_real)  X(*),Y(*),Z(*),WMAIN(*)
  !
  INTEGER I,IMODE
  character(len=4) WRD
  !
  call chmalloc('images.src','IMSPEC','ISLCT',natom,intg=ISLCT)

  IF (.NOT.LIMCEN) IMCENF(1:NATOM) = 0
  !
  WRD=NEXTA4(COMLYN,COMLEN)
  IMXCEN=GTRMF(COMLYN,COMLEN,'XCEN',IMXCEN)
  IMYCEN=GTRMF(COMLYN,COMLEN,'YCEN',IMYCEN)
  IMZCEN=GTRMF(COMLYN,COMLEN,'ZCEN',IMZCEN)
  !
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
  !
  IMODE=-1
  IF (WRD.EQ.'FIXE') THEN
     IMODE=0
  ELSEIF (WRD.EQ.'BYSE') THEN
     IMODE=1
  ELSEIF (WRD.EQ.'BYRE') THEN
     IMODE=2
  ELSEIF (WRD.EQ.'BYGR') THEN
     IMODE=3
  ELSEIF (WRD.EQ.'BYAT') THEN
     IMODE=4
  ELSE
     CALL WRNDIE(-1,'<IMSPEC>','UNRECOGNIZED COMMAND')
  ENDIF
  !
  IF(IMODE.GE.0) THEN
     DO I=1,NATOM
        IF(ISLCT(I).EQ.1) IMCENF(I)=IMODE
     ENDDO
  ENDIF
  !
  LIMCEN=.FALSE.
  DO I=1,NATOM
     IF(IMCENF(I).GT.0) LIMCEN=.TRUE.
  ENDDO
  IF(PRNLEV.GE.2) THEN
     IF (LIMCEN) THEN
        WRITE(OUTU,'(A)') ' IMAGE CENTERING ON FOR SOME ATOMS'
     ELSE
        WRITE(OUTU,'(A)') ' IMAGE CENTERING TURNED OFF'
     ENDIF
  ENDIF
  !
  call chmdealloc('images.src','IMSPEC','ISLCT',natom,intg=ISLCT)

  RETURN
END SUBROUTINE IMSPEC

SUBROUTINE IMCENT(IMXCEN,IMYCEN,IMZCEN,IMCENF,NTRANS,IMTRNS, &
     IMNAME,X,Y,Z,LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ,check_groups,&
     NumLattice,ILATT,lattice_vector)
  !
  !     THIS ROUTINE PROCESSES IMAGE CENTERING FOR SELECTED ATOMS
  !     DURING AN IMAGE UPDATE (IF REQUESTED)
  !
  use chm_kinds
  use dimens_fcm
#if KEY_TSM==1
  use tsms_mod
#endif 
  use psf
  use stream
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
#if KEY_DOMDEC==1
  use groupxfast,only:invgroup,group,group_out  
#endif
  use number,only:zero

  use image_util, only: procatoms

  implicit none
  !
  real(chm_real)  IMXCEN,IMYCEN,IMZCEN
  INTEGER IMCENF(*)
  INTEGER NTRANS,NumLattice,ILATT
  character(len=*) IMNAME(*)
  real(chm_real) IMTRNS(12,*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) XOLD(*),YOLD(*),ZOLD(*),VX(*),VY(*),VZ(*)
  real(chm_real) lattice_vector(3,3)
  INTEGER LDYNA
  logical check_groups   ! check_groups = .true. for DOMDEC:
                         ! Only single group molecules are re-centered
  !
  INTEGER MODE
  INTEGER ISEG,IMIN,IS,IQ,I,ITRAN,IRES,IAT,icnt,j,k
  !     real(chm_real)  XMAX,XMIN,YMAX,YMIN,ZMAX,ZMIN,XCEN,YCEN,ZCEN
  !     real(chm_real)  R2,RMIN,XN,YN,ZN
  !
#if KEY_DOMDEC==1
  integer is_t, iq_t  
#endif
  IF(PRNLEV.GE.4) WRITE(OUTU,22) IMXCEN,IMYCEN,IMZCEN
22 FORMAT(/' SELECTED IMAGES ATOMS BEING CENTERED ABOUT',3F10.6)
#if KEY_DOMDEC==1
  if (q_domdec) then
     if (imxcen /= zero .or. imycen /= zero .or. imzcen /= zero) then
        call wrndie(-5,'<imcent>','For DOMDEC, images must be centered at (0, 0, 0)')
     endif
  endif
#endif 

#if KEY_DOMDEC==0
  ! for lattice transformation for NAMD trajectory.
  if(NumLattice.gt.0) then
     icnt=0
     do i=-ILATT,ILATT
        do j=-ILATT,ILATT
           do k=-ILATT,ILATT
              icnt=icnt+1
              IMTRNS(1,icnt) = 1.0d0
              IMTRNS(2:4,icnt) =0.0d0
              IMTRNS(5,icnt) = 1.0d0
              IMTRNS(6:8,icnt) = 0.0d0
              IMTRNS(9,icnt) = 1.0d0
              IMTRNS(10:12,icnt)=real(i)*lattice_vector(1:3,1)+  &
                                 real(j)*lattice_vector(1:3,2)+  &
                                 real(k)*lattice_vector(1:3,3)
           end do
        end do
     end do
  end if
#endif

  !
  !     FIRST SEE IF ANY SEGMENTS NEED UPDATING (IMIN=1)
  !
  MODE=1
  ISEG=0
1000 IF (ISEG.LT.NSEG) THEN
     ISEG=ISEG+1
     IMIN=5
     IS=IBASE(NICTOT(ISEG)+1)+1
     IQ=IBASE(NICTOT(ISEG+1)+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
        if (check_groups) then
#if KEY_DOMDEC==1
           if (invgroup(is) /= invgroup(iq)) then
              ! Atoms is and iq belong to different DOMDEC groups
              if (prnlev >= 7) then
                 write (outu,'(a)') &
                      'Image centering NOT done for atom selections that cross DOMDEC groups'
                 write (outu,'(a)') &
                      'Only small molecules (e.g. water) with 1 to 4 atoms are within single group'
                 write (outu,'(a)') 'Remove IMAGe settings for all other molecules'
              endif
              goto 1000
           endif
#endif 
        endif
        !         PROCESS-THESE-ATOMS
        CALL PROCATOMS(IMXCEN,IMYCEN,IMZCEN,NTRANS,IMTRNS,X,Y,Z, &
             LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ, IMIN,ITRAN,IS,IQ)
        !
        IF(ITRAN.GT.0 .AND. PRNLEV.GE.5) &
             WRITE(OUTU,15) SEGID(ISEG)(1:idleng),IMNAME(ITRAN)
15      FORMAT(' SEGMENT ',A,' OPERATED ON BY TRANSFORMATION ',A)
     ENDIF
     GOTO 1000
  ENDIF
  !
  !     NEXT SEE IF ANY RESIDUES NEED UPDATING (IMIN=2)
  !
  MODE=2
  IRES=0
2000 IF (IRES.LT.NRES) THEN
     IRES=IRES+1
     IMIN=5
     IS=IBASE(IRES)+1
     IQ=IBASE(IRES+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
        if (check_groups) then
#if KEY_DOMDEC==1
           if (invgroup(is) /= invgroup(iq)) then
              ! Atoms is and iq belong to different DOMDEC groups
              if (prnlev >= 7) then
                 write (outu,'(a)') &
                      'Image centering NOT done for atom selections that cross DOMDEC groups'
                 write (outu,'(a)') &
                      'Only small molecules (e.g. water) with 1 to 4 atoms are within single group'
                 write (outu,'(a)') 'Remove IMAGe settings for all other molecules'
              endif
              goto 2000
           endif
#endif 
        endif
        !         PROCESS-THESE-ATOMS
        CALL PROCATOMS(IMXCEN,IMYCEN,IMZCEN,NTRANS,IMTRNS,X,Y,Z, &
             LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ, IMIN,ITRAN,IS,IQ)
        IF(ITRAN.GT.0) THEN
           IF(PRNLEV.GE.5) WRITE(OUTU,25) IRES,IMNAME(ITRAN)
25         FORMAT(' RESIDUE',I5,' OPERATED ON BY TRANSFORMATION ',A)
        ENDIF
     ENDIF
     GOTO 2000
  ENDIF
  !
  !     NEXT SEE IF ANY GROUPS NEED UPDATING (IMIN=3)
  !
  MODE=3
  IRES=0
3000 IF (IRES.LT.NGRP) THEN
     IRES=IRES+1
     IMIN=5
     IS=IGPBS(IRES)+1
     IQ=IGPBS(IRES+1)
     DO I=IS,IQ
        IF(IMCENF(I).LT.IMIN) IMIN=IMCENF(I)
     ENDDO
     IF(IMIN.EQ.MODE) THEN
        if (check_groups) then
#if KEY_DOMDEC==1
           if (invgroup(is) /= invgroup(iq)) then
              ! Atoms is and iq belong to different DOMDEC groups
              if (prnlev >= 7) then
                 write (outu,'(a)') &
                      'Image centering NOT done for atom selections that cross DOMDEC groups'
                 write (outu,'(a)') &
                      'Only small molecules (e.g. water) with 1 to 4 atoms are within single group'
                 write (outu,'(a)') 'Remove IMAGe settings for all other molecules'
              endif
              goto 3000
           endif
#endif 
        endif
        !         PROCESS-THESE-ATOMS
        CALL PROCATOMS(IMXCEN,IMYCEN,IMZCEN,NTRANS,IMTRNS,X,Y,Z, &
             LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ, IMIN,ITRAN,IS,IQ)
        IF(ITRAN.GT.0) THEN
           IF(PRNLEV.GE.5) WRITE(OUTU,35) IRES,IMNAME(ITRAN)
35         FORMAT(' GROUP',I5,' OPERATED ON BY TRANSFORMATION ',A)
        ENDIF
     ENDIF
     GOTO 3000
  ENDIF
  !
  !
  !     LASTLY, SEE IF ANY ATOMS NEED UPDATING (IMIN=4)
  !
  MODE=4
  IAT=0
4000 IF (IAT.LT.NATOM) THEN
     IAT=IAT+1
     IS=IAT
     IQ=IAT
     IF(IMCENF(IAT).EQ.MODE) THEN
        if (check_groups) then
#if KEY_DOMDEC==1
           call group_out(group(invgroup(is)), is_t, iq_t)
           if ( iq_t - is_t >= 1) then
              if (prnlev >= 7) write (outu,'(a)') &
                   'DOMDEC: Unable to center atoms in groups with more than one atom'
              goto 4000
           endif
#endif 
        endif
        !         PROCESS-THESE-ATOMS
        CALL PROCATOMS(IMXCEN,IMYCEN,IMZCEN,NTRANS,IMTRNS,X,Y,Z, &
             LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ, IMIN,ITRAN,IS,IQ)
        IF(ITRAN.GT.0) THEN
           IF(PRNLEV.GE.5) WRITE(OUTU,45) IAT,IMNAME(ITRAN)
45         FORMAT(' ATOM',I5,' OPERATED ON BY TRANSFORMATION ',A)
        ENDIF
     ENDIF
     GOTO 4000
  ENDIF
  !
#if KEY_TSM==1
  !---  Make sure that piggyback for xold etc is set.
  IF(QTSM.AND.PIGSET) THEN
     IF(LDYNA.NE.0) CALL PIGCVSET(XOLD,YOLD,ZOLD)
     IF(ABS(LDYNA).EQ.1) CALL BACK0(VX,VY,VZ)
  ENDIF
#endif 
  RETURN
END SUBROUTINE IMCENT

end module image_routines_module
