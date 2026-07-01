subroutine ecnstr(ec,qcnstr,refx,refy,refz,kcnstr,natom, &
     kcexpn,xhscale,yhscale,zhscale,qtype, &
     numhsets,typhset,ihset,qhnort,qhnotr, &
     x,y,z,dx,dy,dz, &
     qecont,econt,dd1,iupt,qsecd &
     ,numpca,pcax,pcay,pcaz &
     )

  !-----------------------------------------------------------------------
  ! CALCULATES THE ENERGY OF A HARMONIC POSITIONAL CONSTRAINT.
  ! REFX, REFY, AND REFZ GIVES THE CONSTRAINT POSITION. KCNSTR GIVES
  ! THE FORCE CONSTANT, I.E. U=K/2*DELX**2. IF ALL THE FORCE CONSTANTS
  ! ARE ZERO, QCNSTR WILL BE FALSE WHICH WILL ALLOW A QUICK TEST TO
  ! BYPASS THIS CALCULATION.
  !
  ! Overhauled by Bernard R. Brooks   1983
  ! BESTFIT and RELATIVE restraint types added July 1997 - BRB
  ! SPASIBA Force Field added 11/01 P. Lagant and R.Stote
  ! SPASIBA Force Field removed for c34b1 and c35a1 releases
  !

  use chm_kinds
  use cons_rmsd, only: qrmsd,maxalloc,ecnst3
  use number
  use stream
  use memory
  use parallel
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
  implicit none

  real(chm_real) ec
  logical qcnstr
  real(chm_real) refx(*),refy(*),refz(*)
  integer numpca
  real(chm_real)  pcax(numpca,natom),pcay(numpca,natom),pcaz(numpca,natom), epca
  real(chm_real) kcnstr(*)
  integer natom
  integer kcexpn(*)
  real(chm_real) xhscale(*),yhscale(*),zhscale(*)
  integer qtype,numhsets,typhset(*),ihset(*)
  logical qhnort(*), qhnotr(*)
  real(chm_real) x(*),y(*),z(*),dx(*),dy(*),dz(*),dd1(*)
  logical qecont
  real(chm_real) econt(*)
  integer iupt(*)
  logical qsecd

  integer i,j,iset,jset,ipcset
  integer ninset(numhsets),npair
  integer alloc
  logical lprint
  integer,allocatable,dimension(:) :: atompr
  real(chm_real),allocatable,dimension(:) :: &
       bmass,ra,rb,rb2,dra,drb,drb2,drtemp
  integer ierr_allocate

  ec=zero
  if(.not.qcnstr) return
  lprint=(prnlev.gt.8)

  if(qrmsd.and.qtype.lt.0)then

#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<ecnstr>','RMSD constraint not implemented on domdec')
     endif
#endif 

     if(maxalloc.gt.0) then           ! allocate all temporary space
        alloc=maxalloc
        call chmalloc('ecnstr.src','ECNSTR','atompr',2*alloc,intg=atompr)
        call chmalloc('ecnstr.src','ECNSTR','bmass',alloc,crl=bmass)
        call chmalloc('ecnstr.src','ECNSTR','ra',3*alloc,crl=ra)
        call chmalloc('ecnstr.src','ECNSTR','rb',3*alloc,crl=rb)
        call chmalloc('ecnstr.src','ECNSTR','rb2',3*alloc,crl=rb2)
        call chmalloc('ecnstr.src','ECNSTR','dra',3*alloc,crl=dra)
        call chmalloc('ecnstr.src','ECNSTR','drb',3*alloc,crl=drb)
        call chmalloc('ecnstr.src','ECNSTR','drb2',3*alloc,crl=drb2)
        call chmalloc('ecnstr.src','ECNSTR','drtemp',3*alloc,crl=drtemp)
     endif

#if KEY_PARALLEL==1
     if(mynod.eq.0) then  
#endif
        call ecnst3(x,y,z,atompr,lprint,ec,dx,dy,dz, &
             bmass,ra,rb,rb2, &
             dra,drb,drb2,drtemp)

#if KEY_PARALLEL==1
     endif                
#endif

  else

     DO ISET=1,NUMHSETS
        NINSET(ISET)=0
     ENDDO
     ! Determine which sets are active and check for errors.
     DO I=1,NATOM
        J=IHSET(I)
        IF(J.GT.NUMHSETS .OR. J.LT.0) THEN
           CALL WRNDIE(-4,'<ECNSTR>','Bad atom set value - IGNORE')
           RETURN
        ENDIF
        IF(J.GT.0) NINSET(J)=NINSET(J)+1
     ENDDO
     ALLOC=0
     DO ISET=1,NUMHSETS
        ! do some simple checks
        IF(TYPHSET(ISET).GT.0 .and. typhset(iset) .ne. 2) THEN
           IF(QSECD) CALL WRNDIE(-2,'<ECNSTR>', &
                'Second derivates not supported for best fit restraints')
           IF(QECONT) CALL WRNDIE(-2,'<ECNSTR>', &
                'Energy partitioning not supported for best fit restraints')
           IF(ALLOC.LT.NINSET(ISET)) ALLOC=NINSET(ISET)
        ENDIF
     ENDDO

     ! Allocate all temporary space
     IF(ALLOC.GT.0) THEN
        call chmalloc('ecnstr.src','ECNSTR','atompr',2*alloc,intg=atompr)
        call chmalloc('ecnstr.src','ECNSTR','bmass',alloc,crl=bmass)
        call chmalloc('ecnstr.src','ECNSTR','ra',3*alloc,crl=ra)
        call chmalloc('ecnstr.src','ECNSTR','rb',3*alloc,crl=rb)
        call chmalloc('ecnstr.src','ECNSTR','rb2',3*alloc,crl=rb2)
        call chmalloc('ecnstr.src','ECNSTR','dra',3*alloc,crl=dra)
        call chmalloc('ecnstr.src','ECNSTR','drb',3*alloc,crl=drb)
        call chmalloc('ecnstr.src','ECNSTR','drb2',3*alloc,crl=drb2)
        call chmalloc('ecnstr.src','ECNSTR','drtemp',3*alloc,crl=drtemp)
     ENDIF

     ! Process each restraint set
     DO ISET=1,NUMHSETS
        IF(NINSET(ISET)>0) THEN  ! only process non-null sets

           IF( (TYPHSET(ISET).EQ.0 .or. typhset(iset).eq.3) .AND. QTYPE.GE.0) THEN
              ! Process standard restraint energy term (old).
              if ( typhset(iset) == 0 .and. qtype == 1) then
                 CALL ECNST1(EC,REFX,REFY,REFZ,KCNSTR,NATOM, &
                      KCEXPN(ISET),XHSCALE(ISET), &
                      YHSCALE(ISET),ZHSCALE(ISET), &
                      ISET,IHSET,X,Y,Z,DX,DY,DZ, &
                      QECONT,ECONT,DD1,IUPT,QSECD &
                      ,typhset(iset),(/0/),(/0/),(/0/) &
                      )
              endif
              if ( typhset(iset) == 3 .and. qtype == 2) then
                 do ipcset = 1, numpca
                    epca = zero
                    CALL ECNST1(EPCA,REFX,REFY,REFZ,KCNSTR,NATOM, &
                         KCEXPN(ISET),XHSCALE(ISET), &
                         YHSCALE(ISET),ZHSCALE(ISET), &
                         ISET,IHSET,X,Y,Z,DX,DY,DZ, &
                         QECONT,ECONT,DD1,IUPT,QSECD &
                         ,typhset(iset),pcax(ipcset,1:natom), &
                         pcay(ipcset,1:natom), &
                         pcaz(ipcset,1:natom) &
                         )
                    ec = ec + epca
                    write(*,*)'ipcset:',ipcset,epca,ec
                 enddo
              endif

#if KEY_PARALLEL==1
           ELSE IF(MYNOD.NE.INODE(ISET)) THEN
              CONTINUE
#endif 
           ELSE IF(TYPHSET(ISET).EQ.1 .AND. QTYPE.LE.0) THEN
              ! Process best-fit restraint energy term.
#if KEY_DOMDEC==1
              if (q_domdec) then
                 call wrndie(-5,'<ecnstr>','Bestfit restraint not implemented on domdec')
              endif
#endif 
              NPAIR=NINSET(ISET)
              j=1
              DO I=1,NATOM
                 IF(IHSET(I).EQ.ISET) THEN
                    atompr(J)=I
                    atompr(J+1)=I
                    J=J+2
                 ENDIF
              ENDDO

              CALL ECNST2(X,Y,Z,REFX,REFY,REFZ,ATOMPR,NPAIR, &
                   QHNORT(ISET),QHNOTR(ISET),LPRINT,EC, &
                   .TRUE.,DX,DY,DZ,.FALSE., (/ ZERO /), (/ ZERO /), (/ ZERO /), KCNSTR, &
                   BMASS,RA,RB, &
                   DRA,DRB,DRTEMP)

           ELSE IF(TYPHSET(ISET) == 2 .AND. QTYPE.LE.0) THEN
              ! Process best-fit restraint energy for two PSF subsets
#if KEY_DOMDEC==1
              if (q_domdec) then
                 call wrndie(-5,'<ecnstr>','Bestfit restraint not implemented on domdec')
              endif
#endif 
              JSET=TYPHSET(ISET)
              IF(TYPHSET(JSET).NE.ISET) CALL WRNDIE(-4,'<ECNSTR>', &
                   'Error in relative set types - coding error')
              IF(ISET.LT.JSET) THEN  ! only process once (ISET<JSET)
                 NPAIR=NINSET(ISET)
                 IF(NPAIR.NE.NINSET(JSET)) CALL WRNDIE(-2,'<ECNSTR>', &
                      'Number of selected atoms in two sets do not match')
                 IF(NPAIR.GT.NINSET(JSET)) NPAIR=NINSET(JSET)

                 j=1
                 DO I=1,NATOM
                    IF(IHSET(I).EQ.ISET) THEN
                       atompr(J)=I
                       J=J+2
                    ENDIF
                 ENDDO
                 j=2
                 DO I=1,NATOM
                    IF(IHSET(I).EQ.JSET) THEN
                       atompr(J)=I
                       J=J+2
                    ENDIF
                 ENDDO

                 CALL ECNST2(X,Y,Z,X,Y,Z,ATOMPR,NPAIR, &
                      QHNORT(ISET),QHNOTR(ISET),LPRINT,EC, &
                      .TRUE.,DX,DY,DZ,.TRUE.,DX,DY,DZ,KCNSTR, &
                      BMASS,RA,RB, &
                      DRA,DRB,DRTEMP)
              ENDIF

           ELSE IF((TYPHSET(ISET).EQ.0 .or. TYPHSET(ISET).EQ.3) .AND. QTYPE.LT.0) THEN
              CONTINUE
           ELSE IF(TYPHSET(ISET).GT.0 .AND. QTYPE.GT.0) THEN
              CONTINUE
           ELSE
              CALL WRNDIE(-4,'<ECNSTR>','Bad set type')
           ENDIF
        ENDIF
     ENDDO
  ENDIF  !  IF(QRMSD.AND.QTYPE.LT.0)THEN

  ! Free all temporary space
  IF(ALLOC.GT.0) THEN
     call chmdealloc('ecnstr.src','ECNSTR','atompr',2*alloc,intg=atompr)
     call chmdealloc('ecnstr.src','ECNSTR','bmass',alloc,crl=bmass)
     call chmdealloc('ecnstr.src','ECNSTR','ra',3*alloc,crl=ra)
     call chmdealloc('ecnstr.src','ECNSTR','rb',3*alloc,crl=rb)
     call chmdealloc('ecnstr.src','ECNSTR','rb2',3*alloc,crl=rb2)
     call chmdealloc('ecnstr.src','ECNSTR','dra',3*alloc,crl=dra)
     call chmdealloc('ecnstr.src','ECNSTR','drb',3*alloc,crl=drb)
     call chmdealloc('ecnstr.src','ECNSTR','drb2',3*alloc,crl=drb2)
     call chmdealloc('ecnstr.src','ECNSTR','drtemp',3*alloc,crl=drtemp)
  ENDIF

  RETURN
END SUBROUTINE ECNSTR

SUBROUTINE ECNST1(EC,REFX,REFY,REFZ,KCNSTR,NATOM, &
     KCEXPN,XHSCALE,YHSCALE,ZHSCALE, &
     ISET,IHSET,X,Y,Z,DX,DY,DZ, &
     QECONT,ECONT,DD1,IUPT,QSECD, &
     hrtyp, PCAX, PCAY, PCAZ &
     )

  !-----------------------------------------------------------------------
  ! CALCULATES THE ENERGY OF A HARMONIC POSITIONAL CONSTRAINT.
  ! REFX, REFY, AND REFZ GIVES THE CONSTRAINT POSITION. KCNSTR GIVES
  ! THE FORCE CONSTANT, I.E. U=K/2*DELX**2.
  !
  ! Overhauled by Bernard R. Brooks   1983
  !
  use chm_kinds
  use number
  use dimens_fcm
  use parallel
#if KEY_DIMB==1
  use dimb       
#endif
#if KEY_GENETIC==1
  use galgor     
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec,natoml,atoml  
#endif

  implicit none
  integer,intent(in) :: hrtyp
  real(chm_real) pcax(natom), pcay(natom), pcaz(natom)
  real(chm_real) EC
  real(chm_real) REFX(*),REFY(*),REFZ(*)
  real(chm_real) KCNSTR(*)
  INTEGER NATOM
  INTEGER KCEXPN
  real(chm_real) XHSCALE,YHSCALE,ZHSCALE
  INTEGER ISET,IHSET(*)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),DD1(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  INTEGER IUPT(*)
  LOGICAL QSECD

  INTEGER ATFRST,ATLAST, ia
  real(chm_real) E1,CEXPN,C,S,R,DF,DDF
  real(chm_real) AX,AY,AZ
  INTEGER I,II,IADD,KEXPN
  real(chm_real), parameter :: TOL = 1.D-12
#if KEY_GENETIC==1
  INTEGER First
  First = 1
  if(qGA_Ener) First = Int(EC)
#endif 

  CEXPN=KCEXPN
  KEXPN=KCEXPN
  IF (NATOM.EQ.0) RETURN

  ! Set loop limits
#if KEY_DOMDEC==1 /*domdec*/
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif /* (domdec)*/
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile option'
#endif /* (parstest)*/
  ! Define the atom bounds for this processor.
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
!  iloop: DO I=ATFRST,ATLAST
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
#if KEY_GENETIC==1
  atfrst = first
  atlast = natom
!  iloop: DO I=First,NATOM
#else /**/
  atfrst = 1
  atlast = natom
!  iloop: DO I=1,NATOM
#endif 
#if KEY_SPACDEC==0
!     IF(JPBLOCK(I).NE.MYNOD) CYCLE iloop  
#endif
#else /* (parfmain)*/
#error  'Illegal parallel compile option'
#endif /* (parfmain)*/
#else /* (paramain)*/
#if KEY_GENETIC==1
  atfrst = first
  atlast = natom
!  iloop: DO I=First,NATOM
#else /**/
  atfrst = 1
  atlast = natom
!  iloop: DO I=1,NATOM
#endif 
#endif /* (paramain)  PARALLEL*/
#if KEY_DOMDEC==1
  endif      
#endif
  iloop: do ia=atfrst, atlast
#if KEY_DOMDEC==1 /*domdec*/
     if (q_domdec) then
        i = atoml(ia)
     else
#endif /* (domdec)*/
        i = ia
#if KEY_DOMDEC==1
     endif   
#endif

#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile option'
#endif /* (parstest)*/
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
#if KEY_SPACDEC==0
     IF(JPBLOCK(I).NE.MYNOD) CYCLE iloop  
#endif
#else /* (parfmain)*/
#error  'Illegal parallel compile option'
#endif /* (parfmain)*/
#else /* (paramain)*/
#endif /* (paramain)  PARALLEL*/

     IF(ISET.EQ.IHSET(I)) THEN
        C=KCNSTR(I)
     ELSE
        C=0.0
     ENDIF
     IF(C.NE.0.0) THEN
        AX=X(I)-REFX(I)
        AY=Y(I)-REFY(I)
        AZ=Z(I)-REFZ(I)
        if(hrtyp == 3) then
           ax = ax * pcax(i)
           ay = ay * pcay(i)
           az = az * pcaz(i)
        endif
        S=XHSCALE*AX*AX+YHSCALE*AY*AY+ZHSCALE*AZ*AZ

        IF (KEXPN.EQ.2) THEN
           E1=C*S
           DF=TWO*C
           DDF=ZERO
        ELSE IF (KEXPN.EQ.4) THEN
           E1=C*S*S
           DF=C*FOUR*S
           DDF=EIGHT*C
        ELSE IF (KEXPN.EQ.3) THEN
           R=SQRT(S)
           E1=C*S*R
           DF=C*THREE*R
           IF (S.GT.TOL) THEN
              DDF=THREE*C/R
           ELSE
              DDF=ZERO
           ENDIF
        ELSE IF (KEXPN.EQ.1) THEN
           R=SQRT(S)
           E1=C*R
           IF (S.GT.TOL) THEN
              DF=C/R
              DDF=-C/(R*S)
           ELSE
              DF=0.0
              DDF=0.0
           ENDIF
        ELSE IF (KEXPN.LE.0) THEN
           E1=0.0
           DF=0.0
           DDF=0.0
        ELSE
           DF=C*S**((CEXPN-FOUR)*HALF)
           E1=DF*S*S
           DDF=(CEXPN-2.0)*DF*CEXPN
           DF=DF*CEXPN*S
        ENDIF
        EC=EC+E1
        IF(QECONT) ECONT(I)=ECONT(I)+E1

        DX(I)=DX(I)+AX*DF*XHSCALE
        DY(I)=DY(I)+AY*DF*YHSCALE
        DZ(I)=DZ(I)+AZ*DF*ZHSCALE

#if KEY_IPRESS==1
        IF (QIPRSS) PVIR(I)=PVIR(I)+DF*S
#endif 

        IF (QSECD) THEN

#if KEY_DIMB==1
           IF(QCMPCT) THEN
              CALL ECNCMP(I,AX,AY,AZ,DF,DDF,XHSCALE,YHSCALE,ZHSCALE, &
                   DD1,PINBCM,PJNBCM)
           ELSE
#endif /*  DIMB*/

              II=3*I-2
              IADD=IUPT(II)+II
              DD1(IADD)=DD1(IADD)+(AX*AX*DDF*XHSCALE+DF)*XHSCALE
              IADD=IUPT(II+1)+II+1
              DD1(IADD)=DD1(IADD)+(AY*AY*DDF*YHSCALE+DF)*YHSCALE
              IADD=IUPT(II+2)+II+2
              DD1(IADD)=DD1(IADD)+(AZ*AZ*DDF*ZHSCALE+DF)*ZHSCALE
              IADD=IUPT(II)+II+1
              DD1(IADD)=DD1(IADD)+AX*AY*DDF*XHSCALE*YHSCALE
              IADD=IUPT(II)+II+2
              DD1(IADD)=DD1(IADD)+AX*AZ*DDF*XHSCALE*ZHSCALE
              IADD=IUPT(II+1)+II+2
              DD1(IADD)=DD1(IADD)+AY*AZ*DDF*YHSCALE*ZHSCALE

#if KEY_DIMB==1
           ENDIF  
#endif
        ENDIF

     ENDIF
  ENDDO iloop

  RETURN
END SUBROUTINE ECNST1

SUBROUTINE ECNST2(XA,YA,ZA,XB,YB,ZB,ATOMPR,NPAIR, &
     LNOROT,LNOTRN,LPRINT,ERMS, &
     QFORCA,DXA,DYA,DZA,QFORCB,DXB,DYB,DZB, &
     KCNSTR,BMASS,RA,RB,DRA,DRB,DRTEMP)
  !-----------------------------------------------------------------------
  ! Harmonic restraints with best fit.  Main energy calculation routine.
  !
  !  XA,YA,ZA ->  Coordinates for atom set A
  !  XB,YB,ZB ->  Coordinates for atom set B (may be the same as A)
  !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
  !  NPAIR    ->  The number of atoms pairs in the best fit analysis
  !  LNOROT   ->  flag to indicate no rotation in the best fit.
  !  LNOTRN   ->  flag to indicate no translation in the best fit.
  !  LPRINT   ->  Print flag (detailed results)
  !  ERMS     <-  Returned energy value
  !  QFORCA   ->  Compute forces for set A?
  ! DXA,DYA,DZA <->  Set A force vectors
  !  QFORCB   ->  Compute forces for set B?
  ! DXB,DYB,DZB <->  Set B force vectors
  !  KCNSTR   ->  Weighting array (set A atom based)
  !  BMASS    ->  Final weighting array
  !  RA       -  VA - <VA>cm
  !  RB       -  VB - <VB>cm
  !  D        -  RA - U*RB
  !  DRA      -  (dE/dRA)
  !  DRB      -  (-dE/dRB)
  !
  !    By Bernard R. Brooks - NIH - July 1997
  !
  use chm_kinds
  use number
  use stream
  implicit none

  real(chm_real)  XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
  INTEGER NPAIR
  INTEGER ATOMPR(2,NPAIR)
  LOGICAL LNOROT,LNOTRN,LPRINT
  real(chm_real)  ERMS
  LOGICAL QFORCA,QFORCB
  real(chm_real)  DXA(*),DYA(*),DZA(*),DXB(*),DYB(*),DZB(*)
  real(chm_real)  KCNSTR(*)

  ! Temporary space arrays
  real(chm_real) BMASS(NPAIR)
  real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
  real(chm_real) DRA(3,NPAIR),DRB(3,NPAIR),DRTEMP(3,NPAIR)

  ! Local variables and arrays
  real(chm_real) DEVA(3,3)
  real(chm_real) DERMS,TMASS,RMST
  INTEGER K,I
  real(chm_real) U(3,3),DU(3,3,3,3)

  CALL ECBSTF(XA,YA,ZA,XB,YB,ZB,ATOMPR,NPAIR, &
       LNOROT,LNOTRN,LPRINT,KCNSTR,.TRUE.,BMASS, &
       TMASS,RMST,RA,RB,DEVA,ZERO)

  IF(TMASS.LT.RSMALL) RETURN ! don't process nonpositive total weight.
  !
  ! Calculate energy and force mods on all paired atoms
  !
  ERMS=ERMS+RMST
  ! compute dE/d(rmsv)/rmsv
  DERMS=TWO*TMASS

56 format(A/,(10X,3F12.5))
  IF(LPRINT) write(6,56) ' The E value:',RMST

  CALL ECFORC(ATOMPR,NPAIR,LNOROT,LNOTRN,LPRINT, &
       QFORCA,DXA,DYA,DZA,QFORCB,DXB,DYB,DZB, &
       DERMS,BMASS,TMASS,RA,RB,DRA,DRB,DRTEMP,DEVA)

  RETURN
END SUBROUTINE ECNST2

SUBROUTINE ECBSTF(XA,YA,ZA,XB,YB,ZB,ATOMPR,NPAIR, &
     LNOROT,LNOTRN,LPRINTP,KCNSTR,QKCOPY,BMASS, &
     TMASS,RMST,RA,RB,DEVA,EVWID)
  !-----------------------------------------------------------------------
  ! Harmonic restraints with best fit.  Bestfit calculation routine.
  !
  !  XA,YA,ZA ->  Coordinates for atom set A
  !  XB,YB,ZB ->  Coordinates for atom set B (may be the same as A)
  !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
  !  NPAIR    ->  The number of atoms pairs in the best fit analysis
  !  LNOROT   ->  flag to indicate no rotation in the best fit.
  !  LNOTRN   ->  flag to indicate no translation in the best fit.
  !  LPRINT   ->  Print flag (detailed results)
  !  KCNSTR   ->  Weighting array (set A atom based)
  !  QKCOPY   ->  Copy KCNSTR data to BMASS?
  !  BMASS    <-> Final weighting array
  !  TMASS    <-  Total weight
  !  RA       <-  VA - <<VA>>
  !  RB       <-  VB - <<VB>>
  !
  use chm_kinds
  use number
  use stream
  use corsubs,only: frotu
  implicit none

  INTEGER NPAIR
  real(chm_real)  XA(*),YA(*),ZA(*),XB(*),YB(*),ZB(*)
  INTEGER ATOMPR(2,NPAIR)
  LOGICAL LNOROT,LNOTRN,LPRINTP,QKCOPY
  real(chm_real)  KCNSTR(*),TMASS,RMST

  ! Temporary space arrays
  real(chm_real) BMASS(NPAIR)
  real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
  real(chm_real) DEVA(3,3)

  ! Local variables and arrays
  real(chm_real) EVA(3),U(3,3),ATOT,BTOT,EVWID
  real(chm_real) R(3,3)
  real(chm_real) CMA(3),CMB(3),CMC(3)
  real(chm_real) RMSV,ATMP,BTMP
  INTEGER K,KA,KB,I,J
  LOGICAL QEVW,LPRINT

  LPRINT=LPRINTP
  IF(PRNLEV.LE.2)LPRINT=.FALSE.

  TMASS=ZERO
  DO K=1,NPAIR
     KA=ATOMPR(1,K)
     KB=ATOMPR(2,K)
     RA(1,K)=XA(KA)
     RA(2,K)=YA(KA)
     RA(3,K)=ZA(KA)
     RB(1,K)=XB(KB)
     RB(2,K)=YB(KB)
     RB(3,K)=ZB(KB)
     IF(QKCOPY) BMASS(K)=KCNSTR(KA)
     TMASS=TMASS+BMASS(K)
  ENDDO

  IF(TMASS.LT.RSMALL) RETURN ! don't process nonpositive total weight.

  IF(LPRINT) write(OUTU,56) ' The BMASS value:',BMASS
56 format(A/,(10X,3F12.5))

  ! Compute centers of mass (or weight)
  DO I=1,3
     CMA(I)=ZERO
     CMB(I)=ZERO
     CMC(I)=ZERO
  ENDDO

  IF(.NOT.LNOTRN) THEN
     DO K=1,NPAIR
        DO I=1,3
           CMA(I)=CMA(I)+RA(I,K)*BMASS(K)
           CMB(I)=CMB(I)+RB(I,K)*BMASS(K)
        ENDDO
     ENDDO

     DO I=1,3
        CMA(I)=CMA(I)/TMASS
        CMB(I)=CMB(I)/TMASS
        CMC(I)=CMA(I)-CMB(I)
     ENDDO
  ENDIF

  DO K=1,NPAIR
     DO I=1,3
        RA(I,K)=RA(I,K)-CMA(I)
        RB(I,K)=RB(I,K)-CMB(I)
     ENDDO
  ENDDO

  IF(LPRINT) write(OUTU,56) ' The RA value:',RA
  IF(LPRINT) write(OUTU,56) ' The RB value:',RB

  IF (LPRINT) THEN
     WRITE(OUTU,44) CMB
     WRITE(OUTU,45) CMA
     WRITE(OUTU,46) CMC
  ENDIF
44 FORMAT(' CENTER OF ATOMS BEFORE TRANSLATION',3F12.5)
45 FORMAT(' CENTER OF REFERENCE COORDINATE SET',3F12.5)
46 FORMAT(' NET TRANSLATION OF ROTATED ATOMS  ',3F12.5)

  IF (LNOROT) THEN

     RMST=ZERO
     DO K=1,NPAIR
        DO I=1,3
           RMST=RMST+(RA(I,K)-RB(I,K))**2*BMASS(K)
        ENDDO
     ENDDO

  ELSE
     !
     ! compute rotation matrix from lagrangian
     !
     ATOT=ZERO
     BTOT=ZERO
     DO I=1,3
        DO J=1,3
           R(I,J)=ZERO
        ENDDO
     ENDDO
     DO K=1,NPAIR
        DO I=1,3
           ATOT=ATOT + RA(I,K)*RA(I,K)*BMASS(K)
           BTOT=BTOT + RB(I,K)*RB(I,K)*BMASS(K)
           DO J=1,3
              R(I,J)=R(I,J) + RA(I,K)*RB(J,K)*BMASS(K)
           ENDDO
        ENDDO
     ENDDO
     CALL FROTU(R,EVA,DEVA,U,EVWID,QEVW,LPRINT)

     IF(LPRINT) write(6,56) ' The ATOT,BTOT and EV(*) values:', &
          ATOT,BTOT,EVA(1),EVA(2),EVA(3)

     IF(QEVW) THEN
        ! Note: Do not use conventional (higher accuracy) method in case
        ! the width parameter (EVWID) is employed.
        RMST=ATOT+BTOT-2.0*(EVA(1)+EVA(2)+EVA(3))
     ELSE
        RMST=ZERO
        DO K=1,NPAIR
           ATMP=ZERO
           DO I=1,3
              BTMP=RA(I,K)
              DO J=1,3
                 BTMP=BTMP-U(I,J)*RB(J,K)
              ENDDO
              ATMP=ATMP+BTMP**2
           ENDDO
           RMST=RMST+ATMP*BMASS(K)
        ENDDO
     ENDIF

  ENDIF

  IF(RMST .lt. RSMALL) RMST=ZERO
  RMSV=SQRT(RMST/TMASS)
  IF(LPRINT) WRITE(OUTU,14) RMST,TMASS,RMSV
14 FORMAT(' TOTAL SQUARE DIFF IS',F12.4,'  DENOMINATOR IS',F12.4,/ &
       '       THUS RMS DIFF IS',F12.6)

  RETURN
END SUBROUTINE ECBSTF


SUBROUTINE ECFORC(ATOMPR,NPAIR,LNOROT,LNOTRN,LPRINT, &
     QFORCA,DXA,DYA,DZA,QFORCB,DXB,DYB,DZB, &
     DERMS,BMASS,TMASS,RA,RB,DRA,DRB,DRTEMP,DEVA)
  !-----------------------------------------------------------------------
  ! Harmonic restraints with best fit.  Compute and apply forces.
  !
  !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
  !  NPAIR    ->  The number of atoms pairs in the best fit analysis
  !  LNOROT   ->  flag to indicate no rotation in the best fit.
  !  LNOTRN   ->  flag to indicate no translation in the best fit.
  !  LPRINT   ->  Print flag (detailed results)
  !  QFORCA   ->  Compute forces for set A?
  ! DXA,DYA,DZA <->  Set A force vectors
  !  QFORCB   ->  Compute forces for set B?
  ! DXB,DYB,DZB <->  Set B force vectors
  !  DERMS    ->  dE/dRMS
  !  BMASS    ->  Final weighting array
  !  TMASS    ->  total weight
  !  RA       ->  VA - <VA>cm
  !  RB       ->  VB - <VB>cm
  !  DRA      -  (dE/dRA)
  !  DRB      -  (-dE/dRB)
  !
  use chm_kinds
  use number
  use stream
  implicit none

  INTEGER NPAIR
  INTEGER ATOMPR(2,NPAIR)
  LOGICAL LNOROT,LNOTRN,LPRINT
  LOGICAL QFORCA,QFORCB
  real(chm_real)  DXA(*),DYA(*),DZA(*),DXB(*),DYB(*),DZB(*)
  real(chm_real)  DERMS

  ! Temporary space arrays with data
  real(chm_real) BMASS(NPAIR),TMASS
  real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
  ! Temporary space arrays without data
  real(chm_real) DRA(3,NPAIR),DRB(3,NPAIR),DRTEMP(3,NPAIR)
  real(chm_real) DEVA(3,3)

  ! Local variables and arrays
  INTEGER K,KA,KB,I,J,II,JJ
  real(chm_real) TA,TB,KRMS
  real(chm_real) R(3,3)
  real(chm_real) DEU(3,3),DER(3,3),UT(3,3)

  ! don't do anything if neither force flag is set.
  IF(.NOT.(QFORCA.OR.QFORCB)) RETURN

  IF (LNOROT) THEN

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=(RA(I,K)-RB(I,K))*DERMS*BMASS(K)/TMASS
           DRB(I,K)=-DRA(I,K)
        ENDDO
     ENDDO

  ELSE

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=RA(I,K)
           DO J=1,3
              DRA(I,K)=DRA(I,K)-DEVA(I,J)*RB(J,K)
           ENDDO
           DRA(I,K)=DRA(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO

     IF(LPRINT) write(6,56) ' The RAx value:',RA
     IF(LPRINT) write(6,56) ' The RBx value:',RB
     IF(LPRINT) write(6,56) ' The DEVA value:',DEVA
     IF(LPRINT) write(6,56) ' The DRA value:',DRA
56   format(A/,(10X,3F12.5))

     ! compute dE/dRB(1) (negative)
     DO K=1,NPAIR
        DO I=1,3
           DRB(I,K)=RB(I,K)
           DO J=1,3
              DRB(I,K)=DRB(I,K)-DEVA(J,I)*RA(J,K)
           ENDDO
           DRB(I,K)=DRB(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO
     IF(LPRINT) write(6,56) ' The DRB value:',DRB

     ! compute dE/dVA
     ! compute dE/dVB
     !CC        IF(.NOT.LNOTRN) THEN
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              TA=0.0
     !CC              DO J=1,NPAIR
     !CC                TA=TA-DRA(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRA(I,K)-TA/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRA(I,K)=DRTEMP(I,K)
     !CC              TB=0.0
     !CC              DO J=1,NPAIR
     !CC                TB=TB-DRB(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRB(I,K)-TB/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRB(I,K)=DRTEMP(I,K)
     !CC            ENDDO
     !CC          ENDDO
     !CC        ENDIF
     IF(LPRINT) write(6,56) ' The DRA value:',DRA
     IF(LPRINT) write(6,56) ' The DRB value:',DRB

  ENDIF

  ! compute forces (as required)
  IF(QFORCA) THEN
     DO K=1,NPAIR
        KA=ATOMPR(1,K)
        DXA(KA)=DXA(KA)+DRA(1,K)
        DYA(KA)=DYA(KA)+DRA(2,K)
        DZA(KA)=DZA(KA)+DRA(3,K)
     ENDDO
  ENDIF

  IF(QFORCB) THEN
     DO K=1,NPAIR
        KB=ATOMPR(2,K)
        DXB(KB)=DXB(KB)+DRB(1,K)
        DYB(KB)=DYB(KB)+DRB(2,K)
        DZB(KB)=DZB(KB)+DRB(3,K)
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE ECFORC

SUBROUTINE EICCON(ECIC,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
     CCBIC,CCTIC,CCPIC,CCIIC,DD1,IUPT,QSECD,KBEXPN,LUPPER)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE COMPUTES THE IC CONSTRAINTS BY MAKING SEVERAL CALLS
  ! TO EBOND, ETHETA, AND EPHI.
  !
  ! By Bernard R. Brooks   March 1983
  !
  use chm_kinds
  use intcor_module
  use dimens_fcm
  use bases_fcm
  implicit none
  real(chm_real) ECIC,DX(*),DY(*),DZ(*),X(*),Y(*),Z(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  real(chm_real) CCBIC,CCTIC,CCPIC,CCIIC
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER KBEXPN
  LOGICAL LUPPER

  IF (icr_struct%lenic <= 0) RETURN
  CALL EICCN2(ECIC,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
       CCBIC,CCTIC,CCPIC,CCIIC,DD1,IUPT,QSECD,KBEXPN,LUPPER, &
       icr_struct%lenic, icr_struct%B1ic,icr_struct%B2ic, &
       icr_struct%T1ic,icr_struct%T2ic, &
       icr_struct%PIC, icr_struct%IAR, &
       icr_struct%JAR, icr_struct%KAR, &
       icr_struct%LAR, icr_struct%TAR)

  RETURN
END SUBROUTINE EICCON

SUBROUTINE EICCN2(ECIC,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
     CCBIC,CCTIC,CCPIC,CCIIC,DD1,IUPT,QSECD,KBEXPN,LUPPER, &
     LENIC,B1IC,B2IC,T1IC,T2IC,PIC,IAR,JAR,KAR,LAR,TAR)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE COMPUTES THE IC CONSTRAINTS BY MAKING SEVERAL CALLS
  ! TO EBOND, ETHETA, AND EPHI.
  !
  ! By Bernard R. Brooks   March 1983
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use chutil,only:initia
  use eintern
  implicit none
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) CCBIC,CCTIC,CCPIC,CCIIC
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER KBEXPN
  LOGICAL LUPPER
  INTEGER LENIC
  real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
  INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
  LOGICAL  TAR(*)

  real(chm_real) EBIJ,EBIK,EBKL,ETIJK,ETIKJ,ETJKL,EPIJKL,EIIJKL, &
       ECIC,PHI
  INTEGER IC,I,J,K,L
  integer,dimension(LENIC) :: ICINDX,ISKIP,JSKIP
  real(chm_real),dimension(LENIC) :: CCONST,CVALUE,CCOSVA,CSINVA

  EBIJ = 0.0
  EBIK = 0.0
  EBKL = 0.0
  ETIJK=0.0
  ETIKJ=0.0
  ETJKL=0.0
  EPIJKL=0.0
  EIIJKL=0.0
  !
  ! BOND I-J
  !
  IF(CCBIC.GT.0.0) THEN
     DO IC=1,LENIC
        CCONST(IC)=CCBIC
        ICINDX(IC)=IC
        CVALUE(IC)=B2IC(IC)
        I=IAR(IC)
        J=JAR(IC)
        ISKIP(IC)=1
        IF (TAR(IC)) THEN
        ELSE IF (B2IC(IC).LE.0.0001) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EBOND(EBIJ,IAR,JAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP,DD1,IUPT, &
          QSECD,KBEXPN,LUPPER)
     !
     ! BOND I-K
     !
     DO IC=1,LENIC
        I=IAR(IC)
        K=KAR(IC)
        ISKIP(IC)=1
        IF (.NOT.TAR(IC)) THEN
        ELSE IF (B2IC(IC).LE.0.0001) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EBOND(EBIK,IAR,KAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP, &
          DD1,IUPT,QSECD,KBEXPN,LUPPER)

     !
     ! BOND K-L
     !
     DO IC=1,LENIC
        CVALUE(IC)=B1IC(IC)
        K=KAR(IC)
        L=LAR(IC)
        ISKIP(IC)=1
        IF (B1IC(IC).LE.0.0001) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (L.LE.0) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(L,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EBOND(EBKL,KAR,LAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP, &
          DD1,IUPT,QSECD,KBEXPN,LUPPER)

  ENDIF
  !
  ! ANGLE I-J-K
  !
  IF(CCTIC.GT.0.0) THEN
     DO IC=1,LENIC
        CCONST(IC)=CCTIC
        ICINDX(IC)=IC
        CVALUE(IC)=T2IC(IC)*DEGRAD
        I=IAR(IC)
        J=JAR(IC)
        K=KAR(IC)
        ISKIP(IC)=1
        IF (TAR(IC)) THEN
        ELSE IF (T2IC(IC).LE.0.0001) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EANGLE(ETIJK,IAR,JAR,KAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP,DD1,IUPT,QSECD &
          )

     !
     ! ANGLE I-K-J
     !
     DO IC=1,LENIC
        I=IAR(IC)
        J=JAR(IC)
        K=KAR(IC)
        ISKIP(IC)=1
        IF (.NOT.TAR(IC)) THEN
        ELSE IF (T2IC(IC).LE.0.0001) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EANGLE(ETIKJ,IAR,KAR,JAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP,DD1,IUPT,QSECD &
          )

     !
     ! ANGLE I-J-K
     !
     DO IC=1,LENIC
        CVALUE(IC)=T1IC(IC)*DEGRAD
        L=LAR(IC)
        J=JAR(IC)
        K=KAR(IC)
        ISKIP(IC)=1
        IF (T1IC(IC).LE.0.0001) THEN
        ELSE IF (L.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (.NOT.INITIA(L,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EANGLE(ETJKL,JAR,KAR,LAR,ICINDX,LENIC,CCONST,CVALUE, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,1,ISKIP,DD1,IUPT,QSECD &
          )

  ENDIF
  !
  ! DIHEDRAL I-J-K-L
  !
  IF(CCPIC.GT.0.0) THEN
     DO IC=1,LENIC
        CCONST(IC)=CCPIC
        ICINDX(IC)=IC
        PHI=PIC(IC)*DEGRAD
        CVALUE(IC)=PHI
        CCOSVA(IC)=COS(PHI)
        CSINVA(IC)=SIN(PHI)
        I=IAR(IC)
        J=JAR(IC)
        K=KAR(IC)
        L=LAR(IC)
        ISKIP(IC)=1
        JSKIP(IC)=0
        IF (TAR(IC)) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (L.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(L,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EPHI(EPIJKL,IAR,JAR,KAR,LAR,ICINDX,LENIC,CCONST, &
          JSKIP,CVALUE,CCOSVA,CSINVA,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),QECONT,ECONT,1,ISKIP,DD1,IUPT,QSECD &
          )

  ENDIF

  IF(CCIIC.GT.0.0) THEN
     DO IC=1,LENIC
        CCONST(IC)=CCIIC
        ICINDX(IC)=IC
        PHI=PIC(IC)*DEGRAD
        CVALUE(IC)=PHI
        CCOSVA(IC)=COS(PHI)
        CSINVA(IC)=SIN(PHI)
        I=IAR(IC)
        J=JAR(IC)
        K=KAR(IC)
        L=LAR(IC)
        ISKIP(IC)=1
        JSKIP(IC)=0
        IF(.NOT.TAR(IC)) THEN
        ELSE IF (I.LE.0) THEN
        ELSE IF (J.LE.0) THEN
        ELSE IF (K.LE.0) THEN
        ELSE IF (L.LE.0) THEN
        ELSE IF (.NOT.INITIA(I,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(J,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(K,X,Y,Z)) THEN
        ELSE IF (.NOT.INITIA(L,X,Y,Z)) THEN
        ELSE
           ISKIP(IC)=0
        ENDIF
     ENDDO

     CALL EPHI(EIIJKL,IAR,JAR,KAR,LAR,ICINDX,LENIC,CCONST, &
          JSKIP,CVALUE,CCOSVA,CSINVA,DX,DY,DZ,X,Y,Z, &
          .FALSE.,(/ZERO/),QECONT,ECONT,1,ISKIP,DD1,IUPT,QSECD &
          )

  ENDIF

  ECIC=EBIJ+EBIK+EBKL+ETIJK+ETIKJ+ETJKL+EPIJKL+EIIJKL

  RETURN
END SUBROUTINE EICCN2

SUBROUTINE EQUART(ECQRT,X,Y,Z,NATOM,LMASS,AMASS,DX,DY,DZ, &
     QECONT,ECONT,CONST,IEXP,QSECD)
  !-----------------------------------------------------------------------
  ! THIS ROUTINE ADDS A QUARTIC POTENTIAL (IF IEXP=4) ABOUT THE
  ! CENTER OF MASS. ADJUSTMENT IS DONE TO AVOID ANY NET
  ! TRANSLATE/ROTATE FORCES
  !
  !                                  IEXP
  ! E= CONST* SUM ( MASS * ( R - R  )    )
  !              I      I     I   CM
  !
  !
  ! By Bernard R. Brooks   July 1983
  !
  use chm_kinds
  use machutil,only:die
  implicit none
  real(chm_real) ECQRT
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) AMASS(*),CONST
  real(chm_real) DX(*),DY(*),DZ(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  LOGICAL LMASS,QSECD
  INTEGER NATOM,IEXP

  INTEGER JEXP,N,I
  real(chm_real) AMASST,AM,EQ,S,R,EN,DF,CON
  real(chm_real) XCM,YCM,ZCM,DXT,DYT,DZT,RX,RY,RZ

  IF (CONST.EQ.0.0) RETURN
  IF (QSECD) THEN
     CALL WRNDIE(-2,'<EQUART>','NO SECOND DERIVATIVES FROM EQUART')
     RETURN
  ENDIF
#if KEY_IPRESS==1
  IF (QIPRSS) CALL WRNDIE(-2,'<EQUART>', &
       'NO INTERNAL PRESSURE FROM EQUART')
#endif 
  JEXP=IEXP
  CON=CONST
  IF(JEXP.LE.0) CALL DIE
  IF(CON.LT.0.0) CALL DIE
  N=NATOM
  !
  ! FIND THE CENTER OF MASS
  !
  AMASST=0.0
  XCM=0.0
  YCM=0.0
  ZCM=0.0
  DXT=0.0
  DYT=0.0
  DZT=0.0
  DO I=1,N
     IF (LMASS) THEN
        AM=AMASS(I)
     ELSE
        AM=1.0
     ENDIF
     AMASST=AMASST+AM
     XCM=XCM+AM*X(I)
     YCM=YCM+AM*Y(I)
     ZCM=ZCM+AM*Z(I)
     DXT=DXT+DX(I)
     DYT=DYT+DY(I)
     DZT=DZT+DZ(I)
  ENDDO
  XCM=XCM/AMASST
  YCM=YCM/AMASST
  ZCM=ZCM/AMASST
  !
  ! COMPUTE ENERGY AND FORCES FOR EACH ATOM NEGLECTING FOR NOW
  ! THE MOTION CONTRIBUTIONS TO THE CENTER OF MASS.
  !
  EQ=0.0
  DO I=1,N
     RX=X(I)-XCM
     RY=Y(I)-YCM
     RZ=Z(I)-ZCM
     S=RX*RX+RY*RY+RZ*RZ
     R=SQRT(S)
     IF(R.GT.0.000001) THEN
        IF (LMASS) THEN
           AM=AMASS(I)
        ELSE
           AM=1.0
        ENDIF
        EN=CON*AM*R**JEXP
        EQ=EQ+EN
        DF=JEXP*EN/S
        DX(I)=DX(I)+DF*RX
        DY(I)=DY(I)+DF*RY
        DZ(I)=DZ(I)+DF*RZ
        IF(QECONT) ECONT(I)=ECONT(I)+EN
     ENDIF
  ENDDO
  ECQRT=EQ
  !
  ! FIND THE NET FORCES OF THE SYSTEM
  !
  DXT=-DXT
  DYT=-DYT
  DZT=-DZT
  DO I=1,N
     DXT=DXT+DX(I)
     DYT=DYT+DY(I)
     DZT=DZT+DZ(I)
  ENDDO
  !
  ! NOW PROJECT OUT TRANSLATIONAL FORCES TO CONSERVE ENERGY
  ! AND TO PREVENT THE MOLECULE FROM RUNNING OFF.
  ! (THEY WERE NEGLECTED IN THE DERIVATIVE CALCULATION ABOVE)
  !
  DO I=1,N
     IF (LMASS) THEN
        AM=AMASS(I)
     ELSE
        AM=1.0
     ENDIF
     DX(I)=DX(I)-AM*DXT/AMASST
     DY(I)=DY(I)-AM*DYT/AMASST
     DZ(I)=DZ(I)-AM*DZT/AMASST
  ENDDO

  RETURN
END SUBROUTINE EQUART

#if KEY_HMCOM==1 /*hmcm*/
subroutine ecmcns(ecm,refx,refy,refz,kcnstr,rdist,natom, &
     ncsets,icset,incset,cset, &
     x,y,z,dx,dy,dz, &
     lmass,amass, &
     ncrsts,krcnst,refr,cmr1,cmr2)
  !-----------------------------------------------------------------------
  ! Calculates the energy of harmonic restraint for the
  ! centers of mass of subsets of atoms.
  ! This is used for minimization of lattice-based structures
  ! that have been rebuilt from sidechain centers of mass.
  !
  ! REFX, REFY, and REFZ give the restraint position. KCNSTR gives
  ! the force constant, i.e. U=K*DELTA**2.
  !
  ! NCSETS is the number of centers of mass to be restrained,
  ! For each subset of atoms ICSET points to the beginning of
  ! INCSET indices in CSET.
  !
  ! In addition relative distances between centers of mass can be
  ! restrained. NCRSTS is the number of such restraints. KRCNST
  ! contains the force constants, REFR the restraint distances,
  ! and CMR1 and CMR2 the indices for which centers of mass the
  ! relative restraint applies.
  !
  ! by Michael Feig 1999
  !
  use chm_kinds
  use number
  use cstran_mod, only: maxhcm,hmcmtype
  use stream
  !use parallel
  implicit none

  real(chm_real) ecm
  real(chm_real) refx(*),refy(*),refz(*)
  real(chm_real) kcnstr(*),rdist(*)

  integer natom
  integer ncsets
  integer icset(*),incset(*),cset(*)
  real(chm_real) x(*),y(*),z(*),dx(*),dy(*),dz(*)
  logical lmass(*)
  real(chm_real) amass(*)

  integer ncrsts
  real(chm_real) krcnst(*)
  real(chm_real) refr(*)
  integer cmr1(*)
  integer cmr2(*)

  real(chm_real) en,am,rx,ry,rz,s
  real(chm_real) xcm(maxhcm),ycm(maxhcm),zcm(maxhcm),amasst(maxhcm)
  real(chm_real) rs,ars

  integer iset,in,inx,istart,irset,i1,i2
  logical :: qxref,qyref,qzref

  if(ncsets.gt.maxhcm) then
     CALL WRNDIE(-4,'<ECMCNS>','NCSETS exceeds MAXHCM')
     return
  endif

  en=0.0
  do iset=1,ncsets
     istart=icset(iset)

     qxref = btest(hmcmtype(iset),0)
     qyref = btest(hmcmtype(iset),1)
     qzref = btest(hmcmtype(iset),2)
     amasst(iset)=0.0
     xcm(iset)=0.0
     ycm(iset)=0.0
     zcm(iset)=0.0

     !        WRITE(OUTU,728)
     ! 728    FORMAT('HMCM: -----------------------------------')
     !        WRITE(OUTU,729) ISET
     ! 729    FORMAT('HMCM: set  ',I4)

     do in=0,incset(iset)-1
        inx=cset(istart+in)
        !og/ivk     if (lmass(inx)) then
        if (lmass(iset)) then
           am=amass(inx)
        else
           am=1.0
        endif
        amasst(iset)=amasst(iset)+am
        xcm(iset)=xcm(iset)+am*x(inx)
        ycm(iset)=ycm(iset)+am*y(inx)
        zcm(iset)=zcm(iset)+am*z(inx)
     enddo

     xcm(iset)=xcm(iset)/amasst(iset)
     ycm(iset)=ycm(iset)/amasst(iset)
     zcm(iset)=zcm(iset)/amasst(iset)

     rx=zero
     ry=zero
     rz=zero
     if(qxref) rx=xcm(iset)-refx(iset)
     if(qyref) ry=ycm(iset)-refy(iset)
     if(qzref) rz=zcm(iset)-refz(iset)

     s=sqrt(rx*rx+ry*ry+rz*rz)
     rs=s-rdist(iset)

     en=en+kcnstr(iset)*rs*rs
! LNI. Bugfix 2015-05-12. rs and s can both be zero... If only s is zero we have a problem
     if(rs == zero .and. s == zero) s=one
     if(qxref) rx=rx*2.0*kcnstr(iset)/amasst(iset)*rs/s
     if(qyref) ry=ry*2.0*kcnstr(iset)/amasst(iset)*rs/s
     if(qzref) rz=rz*2.0*kcnstr(iset)/amasst(iset)*rs/s

     do in=0,incset(iset)-1
        inx=cset(istart+in)
        !ivk        if (lmass(inx)) then
        if (lmass(iset)) then
           am=amass(inx)
        else
           am=1.0
        endif
        if(qxref) dx(inx)=dx(inx)+rx*am
        if(qyref) dy(inx)=dy(inx)+ry*am
        if(qzref) dz(inx)=dz(inx)+rz*am
     enddo
  enddo

  if (ncrsts.gt.0) then
     do irset=1,ncrsts
        i1=cmr1(irset)
        i2=cmr2(irset)

        rx=xcm(i1)-xcm(i2)
        ry=ycm(i1)-ycm(i2)
        rz=zcm(i1)-zcm(i2)

        s=sqrt(rx*rx+ry*ry+rz*rz)

        rs=s-refr(irset)

        en=en+krcnst(irset)*rs*rs
! LNI: Bugfix. if s=rs=0 we can avoid a 0/0 NaN. If only s is zero we are in trouble
        if(rs == zero .and. s == zero) s=one         
        rs=rs*2.0*krcnst(irset)/s

        ars=rs/amasst(i1)
        istart=icset(i1)
        do in=0,incset(i1)-1
           inx=cset(istart+in)
           !og/ivk        if (lmass(inx)) then
           if (lmass(i1)) then
              am=amass(inx)
           else
              am=1.0
           endif
           !og/ivk        dx(inx)=dx(inx)+rs*rx*am
           !og/ivk        dy(inx)=dy(inx)+rs*ry*am
           !og/ivk        dz(inx)=dz(inx)+rs*rz*am
           dx(inx)=dx(inx)+ars*rx*am
           dy(inx)=dy(inx)+ars*ry*am
           dz(inx)=dz(inx)+ars*rz*am
        enddo

        ars=rs/amasst(i2)
        istart=icset(i2)
        do in=0,incset(i2)-1
           inx=cset(istart+in)
           !og/ivk        if (lmass(inx)) then
           if (lmass(i2)) then
              am=amass(inx)
           else
              am=1.0
           endif
           !og/ivk        dx(inx)=dx(inx)-rs*rx*am
           !og/ivk        dy(inx)=dy(inx)-rs*ry*am
           !og/ivk        dz(inx)=dz(inx)-rs*rz*am
           dx(inx)=dx(inx)-ars*rx*am
           dy(inx)=dy(inx)-ars*ry*am
           dz(inx)=dz(inx)-ars*rz*am
        enddo
     enddo
  endif

  ecm=en

  return
end subroutine ecmcns
#endif /* (hmcm)*/

#if KEY_CPATH==1
!---------------------------------------------------------------
! Sean Law/Michael Feig, MSU, 2007
! GIVEN A SET OF POINTS, PERFORM CUBIC SPLINE INTERPOLATION

SUBROUTINE SPLINT (SPLINEN,NPTS,COEFFA,COEFFB,COEFFC,COEFFD, &
     MAXSPLINEPTS,MAXNSPLINES)

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm

  ! For At^3+Bt^2+Ct+D, provide the arrays to store
  ! the cubic spline interpolation values.  The arrays
  ! provided should correspond to the coefficients
  ! (COEFFicient) of A, B, C, and D, where A, B, and C
  ! are empty and D contains the X, Y, Z coordinates
  ! for the initial spline points.  Once interpolated,
  ! the cubic spline coefficients will be filled into
  ! COEFFA, COEFFB, and COEFFC.  Here, the COEFF arrays
  ! should be a 3-D array where the first dimension
  ! corresponds to the X, Y, or Z coordinates (X=1, Y=2,
  ! and Z=3), the second dimension corresponds to
  ! the number of spline points, and the third dimension
  ! corresponds to the spline piece (if there is only
  ! one piece then it would be 1 in your array).

  implicit none
  INTEGER NPTS !Total number of spline points
  INTEGER SPLINEN !The spline piece number
  INTEGER MAXSPLINEPTS !Maximum number of spline points
  INTEGER MAXNSPLINES !Maximum number of splines
  real(chm_real) COEFFA(3,MAXSPLINEPTS,MAXNSPLINES)
  real(chm_real) COEFFB(3,MAXSPLINEPTS,MAXNSPLINES)
  real(chm_real) COEFFC(3,MAXSPLINEPTS,MAXNSPLINES)
  real(chm_real) COEFFD(3,MAXSPLINEPTS,MAXNSPLINES)

  ! Data dictionary: Declare LOCAL variable types & definitions
  INTEGER I,J                 !Loop index
  real(chm_real) T(NPTS)          !Independent array
  real(chm_real) LAMBDA(NPTS),ZETA(NPTS),MU(NPTS)
  real(chm_real) H(NPTS),ALPHA(NPTS)!Temporary arrays

  !...###############################################################
  !                                                              #
  ! Natural Cubic Spline Algorithm                               #
  !                                                              #
  ! S = A*T^3 + B*T^2 + C*T + D                                  #
  !                                                              #
  !...###############################################################


  !...###############################################################
  !                                                              #
  ! Forming a tridiagonal matrix and solving the linear system   #
  !                                                              #
  !...###############################################################

  DO J=1,3
     !     !STEP0 - Define all values of T
     DO I=1,NPTS,1
        T(I)=I
     ENDDO
     !     !STEP1 - Establish step size (for Tridiagonal Matrix)
     DO I=1,NPTS-1,1
        H(I)=T(I+1)-T(I)
     ENDDO
     !     !STEP2 - Establish the right side of [Tridiagonal]*[C]=[RIGHT]
     DO I=2,NPTS-1,1
        ALPHA(I)=3*(COEFFD(J,I+1,SPLINEN)*H(I-1)-COEFFD(J,I,SPLINEN) &
             *(T(I+1)-T(I-1))+COEFFD(J,I-1,SPLINEN)*H(I)) &
             /(H(I-1)*H(I))
     ENDDO
     !     !###############################################################
     !     ! STEP 3,4,5 and part of 6 solve the tridiagonal linear system #
     !     !###############################################################

     !     !STEP3
     LAMBDA(1)=1
     MU(1)=0
     ZETA(1)=0
     !     !STEP4
     DO I=2,NPTS-1,1
        LAMBDA(I)=2*(T(I+1)-T(I-1))-H(I-1)*MU(I-1)
        MU(I)=H(I)/LAMBDA(I)
        ZETA(I)=(ALPHA(I)-H(I-1)*ZETA(I-1))/LAMBDA(I)
     ENDDO
     !     !STEP5
     LAMBDA(NPTS)=1
     ZETA(NPTS)=0
     COEFFB(J,NPTS,SPLINEN)=0
     !     !STEP6
     DO I=NPTS-1,1,-1
        COEFFB(J,I,SPLINEN)=ZETA(I)-MU(I)*COEFFB(J,I+1,SPLINEN)
        COEFFC(J,I,SPLINEN)=(COEFFD(J,I+1,SPLINEN) &
             -COEFFD(J,I,SPLINEN))/H(I)-H(I)*(COEFFB(J,I+1,SPLINEN) &
             +2*COEFFB(J,I,SPLINEN))/3
        COEFFA(J,I,SPLINEN)=(COEFFB(J,I+1,SPLINEN) &
             -COEFFB(J,I,SPLINEN))/(3*H(I))
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE SPLINT
!-----------------------------------------------------------------------
SUBROUTINE PATHINIPROJ (PATHWRD)

  ! Projection of point on PATH
  ! Start with finding the COM and then projecting it onto the path
  ! Use an initial approximation (TINIAPPROX) by projecting each
  ! COM onto each spline piece while using the endpoints from each
  ! spline piece as a tangent vector.  Once TINIPPROX is obtained,
  ! further determine PATHTINI by using an approximated tangent at
  ! at TINIAPPROX.  Note, for this initial projection, all spline
  ! points for a given spline are tested for against its
  ! corresponding COM selection points.  Furthermore, PATHTINI is
  ! determined for the initial COMParison coordinate set while
  ! PATHTPRIME is determined the same way for the MAIN coordinate
  ! set.  For the first window (from Umbrella Sampling), PATHTINI
  ! should be equal to PATHTPRIME.  For subsequent windows, PATHTINI
  ! should not change while PATHTPRIME will have moved and will
  ! not be equal to PATHTINI.When the COMP key is not referenced,
  ! both PATHTINI and PATHTPRIME will be taken from the MAIN set
  ! and it is assumed (since COMP key is not used) that it is the
  ! first window from Umbrella Sampling.

  ! PATHWRD     - DETERMINES WHERE COORDINATES SHOULD BE READ FROM
  ! J,K,I       - LOOP COUNTERS
  ! COMX        - CENTER OF MASS FOR X,Y,Z
  ! COMY
  ! COMZ
  ! GX,GY,GZ    - TANGENT VECTOR
  ! A,B,C,D     - INPUT FOR CUBIC SPLINE INTERPOLATION
  ! TVAL        - VALUE OF T AFTER PROJECTION OF COM ONTO CUBIC SPLINE
  ! DIST        - ARRAY FOR STORING THE DISTANCE FOR EACH VALID
  !               VALUE OF TVAL
  ! COMPDIST    - INITIAL DISTANCE COMPARISON VALUE
  ! RX,RY,RZ    - THE X,Y,Z COORDINATES ASSOCIATED WITH A GIVEN TVAL

  use chm_kinds
  use dimens_fcm
  use cnst_fcm
  use stream
  use coord
  use coordc
  implicit none

  CHARACTER(len=4) PATHWRD
  INTEGER J,K,I
  real(chm_real)  TINIAPPROX(MAXNPATHS,MAXNCOM)
  real(chm_real)  COMX,COMY,COMZ,GX,GY,GZ,A,B,C,D,TVAL
  real(chm_real)  DIST(PATHNATMS(PATHN)-1),COMPDIST,RX,RY,RZ

  IF (PATHWRD.EQ.'MAIN') THEN
     CALL WRNDIE (1,'<CSTRAN>','COMP NOT REFERENCED IN CONS PATH.  ' &
          //'DEFAULTING TO MAIN COOR SET.')
  ENDIF

  DO J=1,PATHNCOM(PATHN)
     COMX=0
     COMY=0
     COMZ=0
     COMPDIST=1000
     DO K=PATHINDX(PATHCOMI(PATHN,J)), &
          PATHINDX(PATHCOMI(PATHN,J))+PATHCOMILEN(PATHN,J)-1
        IF (PATHWRD.EQ.'COMP') THEN
           ! MAKE SURE THAT COMP COOR SET IS FILLED
           ! OTHERWISE, USE MAIN COOR SET
           IF ((XCOMP(K).NE.9999.0).AND.(YCOMP(K).NE.9999.0) &
                .AND.(XCOMP(K).NE.9999.0)  ) THEN
              COMX=COMX+XCOMP(K)
              COMY=COMY+YCOMP(K)
              COMZ=COMZ+ZCOMP(K)
           ELSE
              IF (PRNLEV.GT.2) &
                   WRITE(OUTU,2008) PATHN,J,K
2008          FORMAT (' CSTRAN>      SPLINE=',I3,' COM=',I3, &
                   ' ATOM NUMBER=',I3)
              CALL WRNDIE (1,'<CSTRAN>', &
                   'COMP REFERENCED BUT COOR NOT LOADED.  ' &
                   //'DEFAULTING TO MAIN COOR SET.')
              IF (PRNLEV.GT.2) &
                   WRITE(OUTU,*) " CSTRAN>      COMP coor set" &
                   //" may be incomplete"
              COMX=COMX+X(K)
              COMY=COMY+Y(K)
              COMZ=COMZ+Z(K)
              PATHWRD='MAIN'
           ENDIF
        ELSE
           COMX=COMX+X(K)
           COMY=COMY+Y(K)
           COMZ=COMZ+Z(K)
        ENDIF
     ENDDO
     COMX=COMX/PATHCOMILEN(PATHN,J)
     COMY=COMY/PATHCOMILEN(PATHN,J)
     COMZ=COMZ/PATHCOMILEN(PATHN,J)
     !        WRITE(*,*) COMX,COMY,COMZ
     DO I=1,PATHNATMS(PATHN)-1
        ! Perform initial projection of COM here
        GX=PATHD(1,I,PATHN)-PATHD(1,I+1,PATHN)
        GY=PATHD(2,I,PATHN)-PATHD(2,I+1,PATHN)
        GZ=PATHD(3,I,PATHN)-PATHD(3,I+1,PATHN)
        A=GX*PATHA(1,I,PATHN)+GY*PATHA(2,I,PATHN)+GZ*PATHA(3,I,PATHN)
        B=GX*PATHB(1,I,PATHN)+GY*PATHB(2,I,PATHN)+GZ*PATHB(3,I,PATHN)
        C=GX*PATHC(1,I,PATHN)+GY*PATHC(2,I,PATHN)+GZ*PATHC(3,I,PATHN)
        D=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)-COMY) &
             +GZ*(PATHD(3,I,PATHN)-COMZ)
        CALL SOLVECUBIC (A,B,C,D,TVAL)
        IF((TVAL.GT.0.0).AND.(TVAL.LE.1.0+PATHTOL)) THEN
           CALL CUBICEVAL (TVAL,I,RX,RY,RZ)
           DIST(I)=SQRT((COMX-RX)*(COMX-RX)+(COMY-RY)*(COMY-RY) &
                +(COMZ-RZ)*(COMZ-RZ))
        ELSE
           DIST(I)=1000
        ENDIF
        IF(COMPDIST.GT.DIST(I)) THEN
           IF (PATHWRD.NE."PRYM") THEN
              TINIAPPROX(PATHN,J)=(I)+TVAL
              COMPDIST=DIST(I)
           ENDIF
        ENDIF
     ENDDO
     IF((COMPDIST.EQ.1000).AND.(PATHWRD.EQ."PRYM")) THEN
        TINIAPPROX(PATHN,J)=PATHTINI(PATHN,J)+PATHTZERO(PATHN,J)
     ENDIF
     !        WRITE(OUTU,*) PATHN,J,COMX,COMY,COMZ,TINIAPPROX(PATHN,J)
     IF((TINIAPPROX(PATHN,J).GE.(1.0-PATHTOL)).AND. &
          ((TINIAPPROX(PATHN,J).LE.PATHNATMS(PATHN)+PATHTOL))) THEN
        !           WRITE(*,*) 'TINI First Approximation'
        !           WRITE(*,*) TINIAPPROX(PATHN,J)
        ! End of first approximation using spline endpoints
        ! as tangent approximation.  Below, is the beginning
        ! of determining PATHTINI (the "TRUE" initial value of T for COMP set)
        ! or determining PATHTPRIME (the initial value of T for the MAIN set).
        DO I=1,PATHNATMS(PATHN)-1
           IF ( (TINIAPPROX(PATHN,J)-I).LE.1.0+PATHTOL) THEN
              COMPDIST=1000
              IF(I.EQ.1.0) THEN
                 DO K=I,2
                    CALL TINIFINAL (I,TINIAPPROX,J,K &
                         ,COMX,COMY,COMZ,TVAL)
                    IF((TVAL.GT.0.0).AND.(TVAL.LE.1.0+PATHTOL)) THEN
                       CALL CUBICEVAL (TVAL,K,RX,RY,RZ)
                       DIST(K)=SQRT &
                            ((COMX-RX)*(COMX-RX) &
                            +(COMY-RY)*(COMY-RY) &
                            +(COMZ-RZ)*(COMZ-RZ))
                    ELSE
                       DIST(K)=1000
                    ENDIF
                    IF(COMPDIST.GT.DIST(K)) THEN
                       IF (PATHWRD.NE.'PRYM') THEN
                          PATHTINI(PATHN,J)=(K)+TVAL
                       ELSE
                          PATHTPRIME(PATHN,J)=(K)+TVAL
                       ENDIF
                       COMPDIST=DIST(K)
                    ENDIF
                 ENDDO
              ELSE IF (I.EQ.PATHNATMS(PATHN)-1) THEN
                 IF (I.EQ.1.0) THEN
                    ! There is only one spline piece!
                    K=1
                    CALL TINIFINAL (I,TINIAPPROX,J,K &
                         ,COMX,COMY,COMZ,TVAL)
                    IF((TVAL.GT.0.0).AND.(TVAL.LE.1.0+PATHTOL)) THEN
                       CALL CUBICEVAL (TVAL,K,RX,RY,RZ)
                       DIST(K)=SQRT &
                            ((COMX-RX)*(COMX-RX)+(COMY-RY)*(COMY-RY) &
                            +(COMZ-RZ)*(COMZ-RZ))
                    ELSE
                       DIST(K)=1000
                    ENDIF
                    IF(COMPDIST.GT.DIST(K)) THEN
                       IF (PATHWRD.NE.'PRYM') THEN
                          PATHTINI(PATHN,J)=(K)+TVAL
                       ELSE
                          PATHTPRIME(PATHN,J)=(K)+TVAL
                       ENDIF
                       COMPDIST=DIST(K)
                    ENDIF
                 ELSE
                    DO K=I-1,I
                       CALL TINIFINAL (I,TINIAPPROX,J,K &
                            ,COMX,COMY,COMZ,TVAL)
                       IF((TVAL.GT.0.0).AND. &
                            (TVAL.LE.1.0+PATHTOL)) THEN
                          CALL CUBICEVAL (TVAL,K,RX,RY,RZ)
                          DIST(K)=SQRT &
                               ((COMX-RX)*(COMX-RX) &
                               +(COMY-RY)*(COMY-RY) &
                               +(COMZ-RZ)*(COMZ-RZ))
                       ELSE
                          DIST(K)=1000
                       ENDIF
                       IF(COMPDIST.GT.DIST(K)) THEN
                          IF (PATHWRD.NE.'PRYM') THEN
                             PATHTINI(PATHN,J)=(K)+TVAL
                          ELSE
                             PATHTPRIME(PATHN,J)=(K)+TVAL
                          ENDIF
                          COMPDIST=DIST(K)
                       ENDIF
                    ENDDO
                 END IF
              ELSE
                 DO K=I-1,I+1
                    CALL TINIFINAL (I,TINIAPPROX,J,K &
                         ,COMX,COMY,COMZ,TVAL)
                    IF((TVAL.GT.0.0).AND.(TVAL.LE.1.0+PATHTOL)) THEN
                       CALL CUBICEVAL (TVAL,K,RX,RY,RZ)
                       DIST(K)=SQRT &
                            ((COMX-RX)*(COMX-RX) &
                            +(COMY-RY)*(COMY-RY) &
                            +(COMZ-RZ)*(COMZ-RZ))
                    ELSE
                       DIST(K)=1000
                    ENDIF
                    IF(COMPDIST.GT.DIST(K)) THEN
                       IF (PATHWRD.NE.'PRYM') THEN
                          PATHTINI(PATHN,J)=(K)+TVAL
                       ELSE
                          PATHTPRIME(PATHN,J)=(K)+TVAL
                       ENDIF
                       COMPDIST=DIST(K)
                    ENDIF
                 ENDDO
              ENDIF
              EXIT
           ENDIF
        ENDDO
        !           WRITE(*,*) 'TINIT Final'
        !           IF (PATHWRD.EQ.'COMP') THEN
        !              WRITE(*,*) PATHTINI(PATHN,J),PATHWRD
        !           ELSE
        !              WRITE(*,*) PATHTPRIME(PATHN,J),PATHWRD
        !           ENDIF
     ELSE
        ! Warning called because TINIAPPROX for COM cannot be found
        IF (PRNLEV.GT.2) &
             WRITE(OUTU,1475) PATHN,J
1475    FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2)
        CALL WRNDIE (-20,'<CSTRAN>', &
             'COM outside of projectable range for TINI. ' &
             //' Increase spline points or remove COM.')
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE PATHINIPROJ
!-----------------------------------------------------------------------
SUBROUTINE PATHTEST (TESTUNIT)

  ! This subroutine has two functions.  The original function was to
  ! allow the user to visualize the path and COM projections prior to
  ! the start of the simulation.  The later function was added to allow
  ! post-production analysis of a trajectory.
  !
  ! TESTUNIT    - The user specified unit number to write to
  ! ATMNUM      - Keeps track of the atom number
  ! ATMNAMES    - An array of atom names that represent a typical
  !               DNA backbone (used for visualization purposes)
  ! ATMTRACK    - Keeps track of which atom name to use
  ! SPLINEDIST  - The square of the End-to-end, straight line spline
  !               piece distance.  The square has been used to avoid SQRT
  ! SPLITSPLINE - Determines if distance between spline points is greater
  !               than 7 angstroms.
  ! CHAINID     - CHAINID='S' is for "Spline" and CHAINID='C' is for "COM"
  ! TVAL        - T-value on a given spline piece
  ! XINI        - Return values from calling CUBICEVAL.
  ! YINI
  ! ZINI
  ! XFINAL
  ! YFINAL
  ! ZFINAL
  !
  ! EXAMPLE OUTPUT DATA STRUCTURE
  ! ATOM      1  P     1 S   1      20.356  13.969  16.245     1 0.000
  ! ATOM      2  O1P   1 S   1      18.599  13.199  16.040     1 0.271
  ! ATOM      3  O2P   1 S   1      17.720  12.850  15.920     1 0.407
  ! ATOM      4  O5*   1 S   1      16.840  12.543  15.780     1 0.543
  ! ATOM      5  C5*   1 S   1      15.959  12.292  15.612     1 0.679
  ! ATOM      6  C4*   1 S   1      15.076  12.112  15.409     1 0.814
  ! ATOM      7  C3*   1 S   1      14.192  12.017  15.165     1 0.950
  ! ATOM      8  P     2 S   2      13.866  12.006  15.063     2 0.000
  ! ATOM      9  O1P   2 S   2      12.107  12.179  14.397     2 0.271
  ! ATOM     10  O2P   2 S   2      11.254  12.395  14.000     2 0.407
  ! ATOM     11  O5*   2 S   2      10.435  12.684  13.567     2 0.543
  ! ATOM     12  C5*   2 S   2       9.661  13.034  13.104     2 0.679
  ! ATOM     13  C4*   2 S   2       8.946  13.436  12.615     2 0.814
  ! ATOM     14  C3*   2 S   2       8.299  13.879  12.107     2 0.950
  ! ATOM     15  P     3 S   3       8.081  14.050  11.915     3 0.000
  ! ATOM     16  O1P   3 S   3       7.084  15.038  10.846     3 0.271
  ! ATOM     17  O2P   3 S   3       6.684  15.562  10.296     3 0.407
  ! ATOM     18  O5*   3 S   3       6.336  16.103   9.738     3 0.543
  ! ATOM     19  C5*   3 S   3       6.027  16.655   9.174     3 0.679
  ! ATOM     20  C4*   3 S   3       5.746  17.216   8.605     3 0.814
  ! ATOM     21  C3*   3 S   3       5.480  17.781   8.035     3 0.950
  ! ATOM     22  P     4 S   4       5.384  17.990   7.824     3 0.000
  ! ATOM     23  COM   1 C   1       6.747  15.474  10.388     3 0.385
  ! ATOM     24  COM   2 C   2      18.152  13.017  15.981     1 0.340
  ! ATOM     25  COM   3 C   3      11.047  12.460  13.896     2 0.441
  !
  ! Column 1 = RECORD NAME
  ! Column 2 = ATOM NUMBER
  ! Column 3 = ATOM TYPE (Points along a spline path have spline points
  !            represented as a DNA backbone while COM points are
  !            represented as type "COM").
  ! Column 4 = RESIDUE NAME (Numbered according to either spline piece
  !            number for splines or COM numbers for COM).
  ! Column 5 = CHAINID (S = Spline and C = COM)
  ! Column 6 = RESIDUE NUMBER (Same as Column 4)
  ! Column 7 = PATH NUMBER
  ! Column 8 = X-COOR     These X, Y, Z-COOR are interpolated points
  ! Column 9 = Y-COOR     for splines and projection points for COM
  ! Column 10= Z-COOR
  ! Column 11= Spline Piece (Represents the spline piece that the point
  !            was taken from).
  ! Column 12= TVAL (Represents the T-value from a specific spline piece
  !            identified in Column 10).  If this TVAL is substituted
  !            into its corresponding spline piece (from Column 10 which
  !            will look like At^3+Bt^2+Ct+D), this will give the
  !            corresponding X-COOR, Y-COOR, and Z-COOR.
  !
  ! The output has only been tested using PyMOL

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm

  implicit none
  INTEGER          TESTUNIT,I,J,ATMNUM,ATMTRACK
  INTEGER          SPLINEDIST,SPLITSPLINE,M,LOOPI,LOOPF
  CHARACTER(LEN=4) ATMNAMES(8)
  CHARACTER(LEN=1) CHAINID
  real(chm_real)           TVAL
  real(chm_real)           XINI,YINI,ZINI,XFINAL,YFINAL,ZFINAL

  ATMNAMES=(/' P  ',' O1P',' O2P',' O5*',' C5*' &
       ,' C4*',' C3*',' 03*'/)

  ATMNUM=1
  ATMTRACK=1
  CHAINID='S'

  ! Check output unit.  If TESTUNI.EQ.OUTU then write
  IF (TESTUNIT.EQ.OUTU) THEN
     LOOPI=1
     LOOPF=PATHN
  ELSE
     LOOPI=PATHN
     LOOPF=PATHN
  ENDIF
  DO M=LOOPI,LOOPF
     PATHN=M
     ! Post-MD Trajectory Analysis
     IF (TESTUNIT.EQ.OUTU) THEN
        DO I=1,PATHNCOM(PATHN)
           WRITE(TESTUNIT,2503) PATHN,I,PATHTINI(PATHN,I) &
                ,PATHTZERO(PATHN,I) &
                ,PATHTZERO(PATHN,I)+PATHTINI(PATHN,I) &
                ,PATHTPRIME(PATHN,I)
2503       FORMAT('PATH = ',I2,' COM = ',I4,' TINIT = ',F13.8 &
                ,' TZERO = ',F13.8,' ABSTZERO = ',F13.8, &
                ' TPRIME = ',F13.8)
        ENDDO
     ELSE
        ! Visualizing setup before MD
        ! Generating coordinates for the spline
        DO I=1,PATHNATMS(PATHN)
           IF (I.NE.PATHNATMS(PATHN)) THEN
              TVAL=0
              CALL CUBICEVAL(TVAL,I,XINI,YINI,ZINI)
              TVAL=1
              CALL CUBICEVAL(TVAL,I,XFINAL,YFINAL,ZFINAL)
              SPLINEDIST=(XINI-XFINAL)*(XINI-XFINAL)+ &
                   (YINI-YFINAL)*(YINI-YFINAL)+ &
                   (ZINI-ZFINAL)*(ZINI-ZFINAL)
              ! Check to see if spline piece is more than 6 Angstroms or less
              SPLITSPLINE=1+SPLINEDIST/(7*7)
              DO J=1,SPLITSPLINE*7
                 IF (J.EQ.1) THEN
                    TVAL=0
                 ELSE
                    TVAL=(0.95/(SPLITSPLINE*7.0))*J
                 ENDIF
                 CALL CUBICEVAL(TVAL,I,XINI,YINI,ZINI)
                 IF (PRNLEV.GT.2) &
                      WRITE(TESTUNIT,2107) ATMNUM,ATMNAMES(ATMTRACK) &
                      ,I,CHAINID,I,PATHN,XINI,YINI,ZINI,I,TVAL
2107             FORMAT('ATOM  ',I5,' ',A4,' ',I3,' ', &
                      A1,I4,' ',I3,3F8.3,'  ',I4,' ',F5.3)
                 ATMNUM=ATMNUM+1
                 IF (ATMTRACK.NE.7.0) THEN
                    ATMTRACK=ATMTRACK+1
                 ELSE
                    ATMTRACK=1
                 ENDIF
              ENDDO
           ELSE
              TVAL=0
              CALL CUBICEVAL(TVAL,I,XINI,YINI,ZINI)
              IF (PRNLEV.GT.2) &
                   WRITE(TESTUNIT,2108) ATMNUM,I,CHAINID &
                   ,I,PATHN,XINI,YINI,ZINI,I,TVAL
2108          FORMAT('ATOM  ',I5,' ',' P  ',' ',I3,' ',A1,I4,' ' &
                   ,I3,3F8.3,'  ',I4,' ',F5.3)
              ATMNUM=ATMNUM+1
           ENDIF
        ENDDO

        ! Generating coordinates for the COM projections (TPRIME) from
        ! the MAIN set.  For COM projections from the COMP set, see below.

        CHAINID='C'

        DO I=1,PATHNCOM(PATHN)
           DO J=1,PATHNATMS(PATHN)-1
              IF((PATHTPRIME(PATHN,I)-J).LT.1.0) THEN
                 TVAL=PATHTPRIME(PATHN,I)-J
                 EXIT
              ENDIF
           ENDDO
           CALL CUBICEVAL(TVAL,J,XINI,YINI,ZINI)
           IF (PRNLEV.GT.2) &
                WRITE(TESTUNIT,2109) ATMNUM,I,CHAINID &
                ,I,PATHN,XINI,YINI,ZINI,J,TVAL
2109       FORMAT('ATOM  ',I5,' ',' COM',' ',I3,' ',A1,I4,' ' &
                ,I3,3F8.3,'  ',I4,' ',F5.3)
           ATMNUM=ATMNUM+1
        ENDDO
        ! Generating coordinates for TINI.

        CHAINID='I'

        DO I=1,PATHNCOM(PATHN)
           IF ((PATHTINI(PATHN,I).NE.PATHTPRIME(PATHN,I))) THEN
              DO J=1,PATHNATMS(PATHN)-1
                 IF((PATHTINI(PATHN,I)-J).LT.1.0) THEN
                    TVAL=PATHTINI(PATHN,I)-J
                    EXIT
                 ENDIF
              ENDDO
              CALL CUBICEVAL(TVAL,J,XINI,YINI,ZINI)
              IF (PRNLEV.GT.2) &
                   WRITE(TESTUNIT,2019) ATMNUM,I,CHAINID &
                   ,I,PATHN,XINI,YINI,ZINI,J,TVAL
2019          FORMAT('ATOM  ',I5,' ',' TIN',' ',I3,' ',A1,I4,' ' &
                   ,I3,3F8.3,'  ',I4,' ',F5.3)
              ATMNUM=ATMNUM+1
           ENDIF
        ENDDO

        ! Generating coordinates for TZERO.

        CHAINID='Z'

        DO I=1,PATHNCOM(PATHN)
           DO J=1,PATHNATMS(PATHN)-1
              IF((PATHTINI(PATHN,I)+PATHTZERO(PATHN,I)-J) &
                   .LT.1.0) THEN
                 TVAL=PATHTINI(PATHN,I)+PATHTZERO(PATHN,I)-J
                 EXIT
              ENDIF
           ENDDO
           CALL CUBICEVAL(TVAL,J,XINI,YINI,ZINI)
           IF (PRNLEV.GT.2) &
                WRITE(TESTUNIT,2190) ATMNUM,I,CHAINID &
                ,I,PATHN,XINI,YINI,ZINI,J,TVAL
2190       FORMAT('ATOM  ',I5,' ',' TZE',' ',I3,' ',A1,I4,' ' &
                ,I3,3F8.3,'  ',I4,' ',F5.3)
           ATMNUM=ATMNUM+1
        ENDDO
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE PATHTEST
!-----------------------------------------------------------------------

SUBROUTINE PATHEPROJ (PATHE,X,Y,Z,DX,DY,DZ)

  ! H,I,J,K     - LOOP COUNTERS
  ! COMTVAL     - VALUE OF T FOR A GIVEN COM
  ! ROOTTYPE    - AFTER THE CUBIC ROOT HAS BEEN DETERMINED
  !               (0.0 <= COMTVAL <= 1.0 + PATHTOL), THE TYPE
  !               OF ROOT IS RECORDED FOR DETERMINATION OF
  !               PARTIAL DERIVATIVES.  A CUBIC ROOT WILL
  !               HAVE SEVEN POSSIBLE DIFFERENT TYPES OF ROOTS
  !               AND, THUS, SEVEN DIFFERENT POSSIBLE PARTIAL
  !               DERIVATIVES.
  ! COMX        - CENTER OF MASS VALUES
  ! COMY
  ! COMZ
  ! PATHTANG    - CENTER OF TANGENT = PATHTINI+PATHTZERO
  !               MORE PRECISELY, PATHTANG IS THE GLOBAL VALUE
  !               OF TZERO
  ! GX,GY,GZ    - TANGENT VECTOR
  ! GXI,GYI,GZI - INITIAL POINT OF TANGENT VECTOR
  ! GXF,GYF,GZF - FINAL POINT OF TANGENT VECTOR
  ! A,B,C,D     - INPUT FOR CUBIC SPLINE INTERPOLATION
  ! TVAL        - VALUE OF T AFTER PROJECTION OF COM ONTO CUBIC SPLINE
  ! DIST        - ARRAY FOR STORING THE DISTANCE FOR EACH VALID
  !               VALUE OF TVAL
  ! COMPDIST    - INITIAL DISTANCE COMPARISON VALUE
  ! RX,RY,RZ    - THE X,Y,Z COORDINATES ASSOCIATED WITH A GIVEN TVAL
  ! PATHE       - ENERGY CALCULATION FOR SPLINE TO BE RETURNED BACK
  !               TO ENERGY TERM

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm

  implicit none
  real(chm_real)  X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  INTEGER H,I,J,K,ROOTTYPE
  real(chm_real)  COMX,COMY,COMZ
  real(chm_real)  PATHTANG
  real(chm_real)  GXI,GXF,GYI,GYF,GZI,GZF,GX,GY,GZ
  real(chm_real)  A,B,C,D,TVAL,RX,RY,RZ
  real(chm_real)  DIST(MAXNATMS-1),COMPDIST
  real(chm_real)  COMTVAL,PATHE

  IF (PATHNCOUNT.NE.0) THEN
     DO PATHN=1,PATHNCOUNT
        DO J=1,PATHNCOM(PATHN)
           COMTVAL=-7
           IF(PATHCOMI(PATHN,J).GT.0) THEN
              COMX=0
              COMY=0
              COMZ=0
              COMPDIST=1000
              DO K=PATHINDX(PATHCOMI(PATHN,J)), &
                   (PATHINDX(PATHCOMI(PATHN,J))) &
                   +PATHCOMILEN(PATHN,J)-1
                 COMX=COMX+X(K)
                 COMY=COMY+Y(K)
                 COMZ=COMZ+Z(K)
                 !                    WRITE(*,*) "ATOM =",K,"XCOOR =",X(K)
              ENDDO
              COMX=COMX/PATHCOMILEN(PATHN,J)
              COMY=COMY/PATHCOMILEN(PATHN,J)
              COMZ=COMZ/PATHCOMILEN(PATHN,J)
              !                 WRITE(*,*) COMX,COMY,COMZ
              ! Begin projection of COM onto PATHTINI+PATHTZERO
              PATHTANG=PATHTINI(PATHN,J)+PATHTZERO(PATHN,J)
              IF (PATHTANG.GT.PATHNATMS(PATHN)) THEN
                 IF (PRNLEV.GT.2)  &
                      WRITE(OUTU,1477) PATHN,J,PATHTANG, &
                      PATHNATMS(PATHN)
1477             FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2, &
                      ", PATHTINI+PATHTANG=PATHTANG=",F8.3, &
                      ", RANGE=",I2)
                 CALL WRNDIE (-20,'<CSTRAN>', &
                      'TZERO is outside spline range. ' &
                      //' Increase spline points, decrease TZERO,' &
                      //'or remove COM.')
              ELSE IF (((PATHTANG-SQRT(1.0/PATHK(PATHN,J))).LT.1.0) &
                   .OR.((PATHTANG+SQRT(1.0/PATHK(PATHN,J))).GT. &
                   PATHNATMS(PATHN))) THEN
                 IF (PRNLEV.GT.2) &
                      WRITE(OUTU,1476) PATHN,J,PATHTANG, &
                      PATHNATMS(PATHN)
1476             FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2, &
                      ", PATHTINI+PATHTZERO=PATHTANG=",F8.3, &
                      ", RANGE=",I2)
                 CALL WRNDIE (-20,'<CSTRAN>', &
                      'TANGENT is outside spline range. ' &
                      //' Increase spline points, decrease TZERO,'  &
                      //'or remove COM.')
              ELSE
                 CALL GLOBALCUBICEVAL (PATHTANG &
                      -SQRT(1.0/PATHK(PATHN,J)),GXI,GYI,GZI)
                 CALL GLOBALCUBICEVAL (PATHTANG &
                      +SQRT(1.0/PATHK(PATHN,J)),GXF,GYF,GZF)
                 GX=GXF-GXI
                 GY=GYF-GYI
                 GZ=GZF-GZI
                 DO I=1,PATHNATMS(PATHN)
                    IF(PATHTANG-I.LE.1.0) EXIT
                 ENDDO
                 ! Determine where center of Tangent is located
                 IF (I.EQ.1.0) THEN
                    ! Tangent located on first spline
                    DO H=I,2
                       CALL ENERTVAL (GX,GY,GZ,H,COMX,COMY,COMZ,TVAL &
                            ,DIST,COMPDIST,COMTVAL,ROOTTYPE)
                    ENDDO
                 ELSE IF (I.EQ.PATHNATMS(PATHN)-1) THEN
                    IF (I.EQ.1.0) THEN
                       ! There is only one spline piece!
                       H=1
                       CALL ENERTVAL (GX,GY,GZ,H,COMX,COMY,COMZ,TVAL &
                            ,DIST,COMPDIST,COMTVAL,ROOTTYPE)
                    ELSE
                       DO H=I-1,I
                          CALL ENERTVAL (GX,GY,GZ,H,COMX,COMY,COMZ &
                               ,TVAL,DIST,COMPDIST,COMTVAL,ROOTTYPE)
                       ENDDO
                    ENDIF
                 ELSE
                    DO H=I-1,I+1
                       CALL ENERTVAL (GX,GY,GZ,H,COMX,COMY,COMZ,TVAL &
                            ,DIST,COMPDIST,COMTVAL,ROOTTYPE)
                    ENDDO
                 ENDIF
              ENDIF
              PATHTPRIME(PATHN,J)=COMTVAL
              IF (COMTVAL.EQ.-7) THEN
                 IF (PRNLEV.GT.2) &
                      WRITE(OUTU,1599) PATHN,J,PATHTPRIME(PATHN,J) &
                      ,PATHTANG,PATHTINI(PATHN,J)
1599             FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2, &
                      ", PATHTPRIME=",F8.3,", PATHTANG=",F8.3, &
                      ", PATHTINI=",F8.3)
                 CALL WRNDIE (-20,'<CSTRAN>', &
                      'COM is within projectable range.  ' &
                      //'Roots not found.' &
                      //'  Try using a smaller TZERO closer' &
                      //' to TPRIME.')
              ELSE IF ((COMTVAL.LT.1.0).OR. &
                   (COMTVAL.GT.PATHNATMS(PATHN))) THEN
                 IF (PRNLEV.GT.2) &
                      WRITE(OUTU,1598) PATHN,J
1598             FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2)
                 CALL WRNDIE (-20,'<CSTRAN>', &
                      'COM is outside of projectable range. ' &
                      //' Increase spline points or remove COM.')
              ELSE
                 !                    WRITE(*,*) COMTVAL,ROOTTYPE,PATHCOMILEN(PATHN,J)
                 ! ADD POTENTIAL ENERGY CALCULATION AND FORCE CALCULATION HERE!
                 ! NOW THAT YOU KNOW WHAT THE ROOTTYPE IS, FIND DERIVATIVE!
                 ! CALL PATHDERIVATIVES - USING ROOTTYPE - DONE
                 PATHTPRIME(PATHN,J)=COMTVAL
                 PATHE=PATHE+PATHK(PATHN,J)* &
                      (COMTVAL-(PATHTZERO(PATHN,J) &
                      +PATHTINI(PATHN,J)))* &
                      (COMTVAL-(PATHTZERO(PATHN,J) &
                      +PATHTINI(PATHN,J)))
                 DO K=PATHINDX(PATHCOMI(PATHN,J)), &
                      (PATHINDX(PATHCOMI(PATHN,J))) &
                      +PATHCOMILEN(PATHN,J)-1
                    CALL PATHPD(ROOTTYPE,COMTVAL &
                         ,PATHCOMILEN(PATHN,J),COMX,COMY,COMZ &
                         ,GX,GY,GZ)
                    !                       WRITE(*,1982) ROOTTYPE,PATHPDX,PATHPDY,PATHPDZ
                    !     $                       ,COMTVAL
                    ! 1982                  FORMAT (I3,4F8.3)

                    DX(K)=DX(K)+2*(PATHK(PATHN,J)*(COMTVAL &
                         -(PATHTZERO(PATHN,J)+PATHTINI(PATHN,J))) &
                         *(PATHPDX))
                    DY(K)=DY(K)+2*(PATHK(PATHN,J)*(COMTVAL &
                         -(PATHTZERO(PATHN,J)+PATHTINI(PATHN,J))) &
                         *(PATHPDY))
                    DZ(K)=DZ(K)+2*(PATHK(PATHN,J)*(COMTVAL &
                         -(PATHTZERO(PATHN,J)+PATHTINI(PATHN,J))) &
                         *(PATHPDZ))
                 ENDDO
              ENDIF
           ENDIF
        ENDDO
     ENDDO
  ENDIF
  RETURN
END SUBROUTINE PATHEPROJ
!----------------------------------------------------------------------

SUBROUTINE ENERTVAL (GX,GY,GZ,H,COMX,COMY,COMZ,TVAL,DIST,COMPDIST &
     ,COMTVAL,ROOTTYPE)

  ! NOTE: ONE ASSUMPTION IS MADE HERE THAT THERE IS ONLY ONE
  ! SOLUTION (COMTVAL) AS LONG AS TZERO IS CLOSE TO TPRIME

  use chm_kinds
  use dimens_fcm
  use cnst_fcm

  implicit none
  real(chm_real)  GX,GY,GZ,COMX,COMY,COMZ,TVAL,DIST(MAXNPATHS)
  real(chm_real)  COMPDIST,COMTVAL
  INTEGER H,ROOTTYPE
  ! LOCAL VARIABLES
  real(chm_real)  A,B,C,D
  real(chm_real)  RX,RY,RZ

  A=GX*PATHA(1,H,PATHN)+GY*PATHA(2,H,PATHN) &
       +GZ*PATHA(3,H,PATHN)
  B=GX*PATHB(1,H,PATHN)+GY*PATHB(2,H,PATHN) &
       +GZ*PATHB(3,H,PATHN)
  C=GX*PATHC(1,H,PATHN)+GY*PATHC(2,H,PATHN) &
       +GZ*PATHC(3,H,PATHN)
  D=GX*(PATHD(1,H,PATHN)-COMX) &
       +GY*(PATHD(2,H,PATHN)-COMY) &
       +GZ*(PATHD(3,H,PATHN)-COMZ)
  !     WRITE(*,*) A,B,C,D,H
  CALL SOLVECUBIC (A,B,C,D,TVAL)
  !     WRITE(*,1991) A,B,C,D
  ! 1991   FORMAT (4F8.3)
  IF((TVAL.GT.0.0).AND.(TVAL.LE.1.0+PATHTOL)) THEN
     CALL CUBICEVAL (TVAL,H,RX,RY,RZ)
     DIST(H)=SQRT( &
          (COMX-RX)*(COMX-RX) &
          +(COMY-RY)*(COMY-RY) &
          +(COMZ-RZ)*(COMZ-RZ) &
          )
  ELSE
     DIST(H)=1000
  ENDIF
  IF (COMPDIST.GT.DIST(H)) THEN
     COMTVAL=H+TVAL
     ROOTTYPE=PATHROOTTYPE
  ENDIF
  RETURN
END SUBROUTINE ENERTVAL
!----------------------------------------------------------------------

FUNCTION PATHCBRT(INPUT) result(pathcbrt_1)

  ! FUNCTION FOR CALCULATING THE CUBIC ROOT OF A REAL NUMBER
  use chm_kinds
  implicit none
  real(chm_real) INPUT,pathcbrt_1

  IF(INPUT.LT.0.0) THEN
     PATHCBRT_1=(-1.0)*(-INPUT)**(1.0/3.0)
  ELSE
     PATHCBRT_1=INPUT**(1.0/3.0)
  ENDIF
  RETURN
END FUNCTION PATHCBRT
!----------------------------------------------------------------------

SUBROUTINE PATHPDNEW (ROOTTYPE,COMTVAL,N,COMX,COMY,COMZ,GX,GY,GZ)

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm
  implicit none

  INTEGER ROOTTYPE,I,J,N,SIGN
  real(chm_real)  COMTVAL,COMX,COMY,COMZ,GX,GY,GZ
  real(chm_real)  EH,BE,SEE,DEE,A,B,C,D,VX,VY,VZ,Q,R
  real(chm_real)  DELTA_SQ,H_SQ,DELTA,H,PI,THETA
  real(chm_real)  ROOT(3)

  PI=4.0*ATAN(1.0)

  DO I=1,MAXNATMS
     IF(COMTVAL-I.LT.1.0+PATHTOL) THEN
        EXIT
     ENDIF
  ENDDO
  EH=GX*PATHA(1,I,PATHN)+GY*PATHA(2,I,PATHN) &
       +GZ*PATHA(3,I,PATHN)
  BE=GX*PATHB(1,I,PATHN)+GY*PATHB(2,I,PATHN) &
       +GZ*PATHB(3,I,PATHN)
  SEE=GX*PATHC(1,I,PATHN)+GY*PATHC(2,I,PATHN) &
       +GZ*PATHC(3,I,PATHN)
  DEE=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  VX=GX*(PATHD(1,I,PATHN))+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  VY=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  VZ=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN))
  A=BE/EH
  B=SEE/EH
  C=DEE/EH
  Q=(A**2-3*B)/9
  R=(2*A**3-9*A*B+27*C)/54

  !     WRITE(*,*) A,B,C,Q,R,VX,COMX,ROOTTYPE
  !     WRITE(*,*) ROOTTYPE
  IF (Q**3.GT.R**2) THEN !THERE ARE THREE REAL ROOTS
     THETA=ACOS(R/SQRT(Q**3))
     !        WRITE(*,*) "THETA", THETA,"ACOS",(R/SQRT(Q**3))
  ENDIF
  !     WRITE(*,*) ROOTTYPE
  IF (ROOTTYPE.EQ.8) THEN
     PATHPDX = SQRT(Q) * SIN(THETA / 3) * GX / N  &
          / EH /(SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3))  &
          / 3
     PATHPDY = SQRT(Q) * SIN(THETA / 3) * GY / N  &
          / EH /(SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3))  &
          / 3
     PATHPDZ = SQRT(Q) * SIN(THETA / 3) * GZ / N  &
          / EH /(SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3))  &
          / 3
  ELSE IF (ROOTTYPE.EQ.9) THEN
     PATHPDX = SQRT(Q) * COS(THETA / 3 + PI / 6) * GX / N / EH  &
          / (SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3)) / 3
     PATHPDY = SQRT(Q) * COS(THETA / 3 + PI / 6) * GY / N / EH  &
          / (SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3)) / 3
     PATHPDZ = SQRT(Q) * COS(THETA / 3 + PI / 6) * GZ / N / EH  &
          / (SQRT(Q ** 3)) / (SQRT(1 - R ** 2 / Q ** 3)) / 3
  ELSE IF (ROOTTYPE.EQ.10) THEN
     PATHPDX = -SQRT(Q) * SIN(THETA / 3 + PI / 3) &
          * GX / N / EH /(SQRT(Q ** 3))/ (SQRT(1 - R**2 / Q ** 3))  &
          / 3
     PATHPDY = -SQRT(Q) * SIN(THETA / 3 + PI / 3) &
          * GY / N / EH /(SQRT(Q ** 3))/ (SQRT(1 - R**2 / Q ** 3))  &
          / 3
     PATHPDZ = -SQRT(Q) * SIN(THETA / 3 + PI / 3) &
          * GZ / N / EH /(SQRT(Q ** 3))/ (SQRT(1 - R**2 / Q ** 3))  &
          / 3
  ELSE IF (ROOTTYPE.EQ.11) THEN
     PATHPDX=0
     PATHPDY=0
     PATHPDZ=0
  ELSE IF (ROOTTYPE.EQ.12) THEN
     IF (R.LT.0) THEN
        SIGN=-1
     ELSE IF (R.GT.0) THEN
        SIGN=1
     ELSE
        SIGN=0
     ENDIF
     PATHPDX = -SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3)))  &
          ** 2)**(1.0/3.0)) * (ABS(-R) / (-R) * GX / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GX / N / EH / 2) /  &
          3 + Q / SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3))) &
          **4 )**(1.0/3.0)) * (ABS(-R) / (-R) * GX / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GX / N / EH / 2) / 3
     PATHPDY = -SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3)))  &
          ** 2)**(1.0/3.0)) * (ABS(-R) / (-R) * GY / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GY / N / EH / 2) /  &
          3 + Q / SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3))) &
          **4 )**(1.0/3.0)) * (ABS(-R) / (-R) * GY / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GY / N / EH / 2) / 3
     PATHPDZ = -SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3)))  &
          ** 2)**(1.0/3.0)) * (ABS(-R) / (-R) * GZ / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GZ / N / EH / 2) /  &
          3 + Q / SIGN * (((1/(ABS(-R) + SQRT((R) ** 2 - Q ** 3))) &
          **4 )**(1.0/3.0)) * (ABS(-R) / (-R) * GZ / N / EH / 2 -  &
          (1/SQRT((R) ** 2 - Q ** 3)) * (R) * GZ / N / EH / 2) / 3
  ENDIF
  RETURN
END SUBROUTINE PATHPDNEW
!----------------------------------------------------------------------

SUBROUTINE PATHPD (ROOTTYPE,COMTVAL,N,COMX,COMY,COMZ,GX,GY,GZ)

  ! PATH RESTRAINT - PARTIAL DERIVATIVES
  ! DETERMINE THE PARTIAL DERIVATIVES DEPENDING ON THE TYPE OF
  ! ROOT THAT WAS FOUND

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm
  implicit none

  INTEGER ROOTTYPE,I,J,N
  real(chm_real)  COMTVAL,COMX,COMY,COMZ,GX,GY,GZ
  real(chm_real)  A,B,C,D,RX,RY,RZ,Q,TN,FTNX,FTNY,FTNZ,FTN
  real(chm_real)  DELTA_SQ,H_SQ,DELTA,H,PI,THETA
  real(chm_real)  ROOT(3),PATHCBRT


  DO I=1,MAXNATMS
     IF(COMTVAL-I.LT.1.0+PATHTOL) THEN
        EXIT
     ENDIF
  ENDDO
  A=GX*PATHA(1,I,PATHN)+GY*PATHA(2,I,PATHN) &
       +GZ*PATHA(3,I,PATHN)
  B=GX*PATHB(1,I,PATHN)+GY*PATHB(2,I,PATHN) &
       +GZ*PATHB(3,I,PATHN)
  C=GX*PATHC(1,I,PATHN)+GY*PATHC(2,I,PATHN) &
       +GZ*PATHC(3,I,PATHN)
  D=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  RX=GX*(PATHD(1,I,PATHN))+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  RY=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)) &
       +GZ*(PATHD(3,I,PATHN)-COMZ)
  RZ=GX*(PATHD(1,I,PATHN)-COMX)+GY*(PATHD(2,I,PATHN)-COMY) &
       +GZ*(PATHD(3,I,PATHN))
  TN=(-B/(3.0*A))
  Q=A*TN*TN*TN+B*TN*TN+C*TN
  FTN=A*TN*TN*TN+B*TN*TN+C*TN+D
  DELTA_SQ=(B*B-3.0*A*C)/(9.0*A*A)
  H_SQ=4.0*(A*A)*(DELTA_SQ*DELTA_SQ*DELTA_SQ)
  DELTA=SQRT(DELTA_SQ)
  H=2.0*A*(DELTA_SQ)**(3.0/2.0)
  PI=4.0*ATAN(1.0)

  !     IF (((DELTA_SQ.LT.0.0).OR.(H_SQ.LT.0.0)).AND.(ROOTTYPE.GT.7)
  !    $     .AND.(FTN*FTN.LT.H_SQ)) THEN
  !     IF (ROOTTYPE.GT.7) THEN
  IF(.TRUE.) THEN
     CALL PATHPDNEW(ROOTTYPE,COMTVAL,N,COMX,COMY,COMZ,GX,GY,GZ)
     COMTVAL=COMTVAL*1
  ELSE
     IF (ROOTTYPE.EQ.1.0) THEN
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=(2.0**(2.0/3.0))*(PATHCBRT((1.0/A*(-Q-RX+GX*COMX &
                   +SQRT((Q+RX-GX*COMX)**2.0-H_SQ)))**(-2.0)))/A*(GX/N &
                   -((Q+RX-GX*COMX)**2.0-H_SQ)**(-1.0/2.0)*(Q+RX-GX* &
                   COMX)*GX/N)/6.0+(2.0**(2.0/3.0))*(PATHCBRT((1.0/A &
                   *(-Q-RX+GX*COMX-SQRT((Q+RX-GX*COMX)**2.0-H_SQ)))** &
                   (-2.0)))/A*(GX/N+((Q+RX-GX*COMX)**2.0-H_SQ)**(-1.0 &
                   /2.0)*(Q+RX-GX*COMX)*GX/N)/6.0
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=(2.0**(2.0/3.0))*(PATHCBRT((1.0/A*(-Q-RY+GY*COMY &
                   +SQRT((Q+RY-GY*COMY)**2.0-H_SQ)))**(-2.0)))/A*(GY/N &
                   -((Q+RY-GY*COMY)**2.0-H_SQ)**(-1.0/2.0)*(Q+RY-GY* &
                   COMY)*GY/N)/6.0+(2.0**(2.0/3.0))*(PATHCBRT((1.0/A &
                   *(-Q-RY+GY*COMY-SQRT((Q+RY-GY*COMY)**2.0-H_SQ)))** &
                   (-2.0)))/A*(GY/N+((Q+RY-GY*COMY)**2.0-H_SQ)**(-1.0 &
                   /2.0)*(Q+RY-GY*COMY)*GY/N)/6.0
           ELSE
              PATHPDZ=(2.0**(2.0/3.0))*(PATHCBRT((1.0/A*(-Q-RZ+GZ*COMZ &
                   +SQRT((Q+RZ-GZ*COMZ)**2.0-H_SQ)))**(-2.0)))/A*(GZ/N &
                   -((Q+RZ-GZ*COMZ)**2.0-H_SQ)**(-1.0/2.0)*(Q+RZ-GZ* &
                   COMZ)*GZ/N)/6.0+(2.0**(2.0/3.0))*(PATHCBRT((1.0/A &
                   *(-Q-RZ+GZ*COMZ-SQRT((Q+RZ-GZ*COMZ)**2.0-H_SQ)))** &
                   (-2.0)))/A*(GZ/N+((Q+RZ-GZ*COMZ)**2.0-H_SQ)**(-1.0 &
                   /2.0)*(Q+RZ-GZ*COMZ)*GZ/N)/6.0
           ENDIF
        ENDDO
     ELSE IF (ROOTTYPE.EQ.2.0) THEN
        PATHPDX=0
        PATHPDY=0
        PATHPDZ=0
     ELSE IF (ROOTTYPE.EQ.3.0) THEN
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=-(2.0**(2.0/3.0)*((Q+RX-GX*COMX)/A)  &
                   **(-2.0/3.0)*GX/N/A)/6.0
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=-(2.0**(2.0/3.0)*((Q+RY-GY*COMY)/A)  &
                   **(-2.0/3.0)*GY/N/A)/6.0
           ELSE
              PATHPDZ=-(2.0**(2.0/3.0)*((Q+RZ-GZ*COMZ)/A)  &
                   **(-2.0/3.0)*GZ/N/A)/6.0
           ENDIF
        ENDDO
     ELSE IF (ROOTTYPE.EQ.4.0) THEN
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=(2.0**(2.0/3.0)*(PATHCBRT((Q+RX-GX*COMX)/A)**(-2.0))*GX/N/A)/3.0
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=(2.0**(2.0/3.0)*(PATHCBRT((Q+RY-GY*COMY)/A)**(-2.0))*GY/N/A)/3.0
           ELSE
              PATHPDZ=(2.0**(2.0/3.0)*(PATHCBRT((Q+RZ-GZ*COMZ)/A)**(-2.0))*GZ/N/A)/3.0
           ENDIF
        ENDDO
     ELSE IF (ROOTTYPE.EQ.5.0) THEN
        IF ((DELTA_SQ.LT.0.0)) THEN
           CALL WRNDIE (-20,'<CSTRAN>', &
                'IMAGINARY NUMBER IN ROOT (TYPE=5) CALCULATION.')
        ENDIF
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=2.0/3.0*DELTA*COS(PI/6.0+ACOS((Q+RX-GX*COMX)/H) &
                   /3.0)*GX/N/H*(1.0-(Q+RX-GX*COMX)**2.0/(H*H))**(-1.0 &
                   /2.0)
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=2.0/3.0*DELTA*COS(PI/6.0+ACOS((Q+RY-GY*COMY)/H) &
                   /3.0)*GY/N/H*(1.0-(Q+RY-GY*COMY)**2.0/(H*H))**(-1.0 &
                   /2.0)
           ELSE
              PATHPDZ=2.0/3.0*DELTA*COS(PI/6.0+ACOS((Q+RZ-GZ*COMZ)/H) &
                   /3.0)*GZ/N/H*(1.0-(Q+RZ-GZ*COMZ)**2.0/(H*H))**(-1.0 &
                   /2.0)
           ENDIF
        ENDDO
     ELSE IF (ROOTTYPE.EQ.6.0) THEN
        IF ((DELTA_SQ.LT.0.0).OR.(H_SQ.LT.0.0)) THEN
           CALL WRNDIE (-20,'<CSTRAN>', &
                'IMAGINARY NUMBER FOUND IN ROOT (TYPE=6) CALCULATION.')
        ENDIF
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=2.0/3.0*DELTA*SIN(ACOS((Q+RX-GX*COMX)/ &
                   H)/3.0)*GX/N/H*(1.0-(Q+RX-GX*COMX)**2  &
                   /(H*H))**(-1.0/2.0)
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=2.0/3.0*DELTA*SIN(ACOS((Q+RY-GY*COMY)/ &
                   H)/3.0)*GY/N/H*(1.0-(Q+RY-GY*COMY)**2  &
                   /(H*H))**(-1.0/2.0)
           ELSE
              PATHPDZ=2.0/3.0*DELTA*SIN(ACOS((Q+RZ-GZ*COMZ)/ &
                   H)/3.0)*GZ/N/H*(1.0-(Q+RZ-GZ*COMZ)**2  &
                   /(H*H))**(-1.0/2.0)
           ENDIF
        ENDDO
     ELSE IF (ROOTTYPE.EQ.7.0) THEN
        ! ROOTYPE IS 7.0
        IF ((DELTA_SQ.LT.0.0).OR.(H_SQ.LT.0.0)) THEN
           IF (PRNLEV.GT.2) &
                WRITE(OUTU,159) A,B,C,D
159        FORMAT ("CSTRAN>    A=",F12.8,", B=",F12.8, &
                ", C=",F12.8,", D=",F12.8)
           CALL WRNDIE (-20,'<CSTRAN>', &
                'IMAGINARY NUMBER FOUND IN ROOT (TYPE=7) CALCULATION.')
        ENDIF
        DO J=1,3
           IF (J.EQ.1.0) THEN
              PATHPDX=-2.0/3.0*DELTA*SIN(PI/3.0+ACOS((Q+RX &
                   -GX*COMX)/H)/3.0)*GX/N/H*(1.0-(Q+RX &
                   -GX*COMX)**2.0/(H*H))**(-1.0/2.0)
           ELSE IF (J.EQ.2.0) THEN
              PATHPDY=-2.0/3.0*DELTA*SIN(PI/3.0+ACOS((Q+RY &
                   -GY*COMY)/H)/3.0)*GY/N/H*(1.0-(Q+RY &
                   -GY*COMY)**2.0/(H*H))**(-1.0/2.0)

           ELSE
              PATHPDZ=-2.0/3.0*DELTA*SIN(PI/3.0+ACOS((Q+RZ &
                   -GZ*COMZ)/H)/3.0)*GZ/N/H*(1.0-(Q+RZ &
                   -GZ*COMZ)**2.0/(H*H))**(-1.0/2.0)
           ENDIF
        ENDDO
     ENDIF

  ENDIF !GOES TO PATHPDNEW
  RETURN
END SUBROUTINE PATHPD

!----------------------------------------------------------------------

SUBROUTINE SOLVECUBICNEW (EH,BE,SEE,DEE,TVAL)

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm
  use number
  implicit none

  real(chm_real)  EH,BE,SEE,DEE,TVAL
  INTEGER I
  ! Local variables
  real(chm_real)  Q,R,THETA
  real(chm_real)  ROOT(3)
  real(chm_real)  PI,PATHCBRT
  REAL(chm_real) SIGN
  real(chm_real)  A,B,C,ALPHA,BETA,AY,BY,CY,DY
  LOGICAL ROOTL

  ROOTL=.TRUE.
  PATHROOTTYPE=0
  PI=4.0*ATAN(1.0)

  A=BE/EH
  B=SEE/EH
  C=DEE/EH
  Q=(A**2-3*B)/9
  R=(2*A**3-9*A*B+27*C)/54
  IF (R**2.LT.Q**3) THEN
     THETA=ACOS(R/SQRT(Q**3))
     ROOT(1)=(-2)*SQRT(Q)*COS(THETA/3)-A/3
     ROOT(2)=(-2)*SQRT(Q)*COS((THETA+2*PI)/3)-A/3
     ROOT(3)=(-2)*SQRT(Q)*COS((THETA-2*PI)/3)-A/3
     !        WRITE(OUTU,*) ROOT(1), ROOT(2), ROOT(3)
     DO I=1,3
        IF ((ROOT(I).GE.0.0).AND.(ROOT(I).LE.1.0+PATHTOL)) THEN
           IF (ROOTL) THEN
              PATHROOTTYPE=I+7
              ROOTL=.FALSE.
              TVAL=ROOT(I)
           ELSE
              CALL WRNDIE (-20,'<CSTRAN>','Multiple roots found.')
           ENDIF
        ELSE
           IF (ROOTL) THEN
              TVAL=7
           ENDIF
        ENDIF
     ENDDO
  ELSE
     IF (R.LT.0) THEN
        SIGN=minone
     ELSE IF (R.GT.0) THEN
        SIGN=one
     ELSE
        SIGN=zero
     ENDIF
     !        ALPHA=(-1)*SIGN*PATHCBRT(ABS(R)+SQRT(R**2-Q**3))
     ALPHA=minone*SIGN*(ABS(R)+SQRT(R*R-Q*Q*Q))**(third)
     IF (ALPHA.EQ.0) THEN
        BETA=0
        ROOT(1)=ALPHA+BETA-A/3
        IF ((ROOT(1).GE.0.0).AND.(ROOT(1).LE.1.0+PATHTOL)) THEN
           PATHROOTTYPE=11
           TVAL=ALPHA+BETA-A/3
        ELSE
           TVAL=7
        ENDIF
     ELSE
        BETA=Q/ALPHA
        ROOT(1)=ALPHA+BETA-A/3
        IF ((ROOT(1).GE.0.0).AND.(ROOT(1).LE.1.0+PATHTOL)) THEN
           PATHROOTTYPE=12
           TVAL=ALPHA+BETA-A/3
        ELSE
           TVAL=7
        ENDIF
     ENDIF
  ENDIF
  RETURN
END SUBROUTINE SOLVECUBICNEW

!----------------------------------------------------------------------
SUBROUTINE SOLVECUBIC (A,B,C,D,TVAL)

  ! SOLVE FOR THE REAL ROOTS OF A CUBIC POLYNOMIAL (GENERATED FROM
  ! CUBIC SPLINE INTERPOLATION!!)
  !
  ! PROVIDE THE COEFFICIENTS TO A CUBIC FUNCTION IN THE FORM
  ! At^3+Bt^2+Ct+D
  ! AND RETURN REAL ROOT, TVAL THAT SATISFIES THE FOLLOWING
  ! 0.0 <= TVAL <= 1.0+(SOME SMALL TOLERANCE)
  ! SINCE TVAL IS TAKEN FROM A SPLINE PIECE AND EACH SPLINE
  ! PIECE ONLY HAS VALUES BETWEEN ZERO AND ONE.

  use chm_kinds
  use dimens_fcm
  use cnst_fcm
  implicit none

  real(chm_real)  A,B,C,D,TVAL
  INTEGER I
  ! Local variables
  real(chm_real)  FTN,TN,DELTA,DELTA_SQ,H,H_SQ,THETA,THETA_DEG
  real(chm_real)  ROOT(3)
  real(chm_real)  PI,CBRT1,CBRT2,EVAL1,EVAL2,EVAL3,PATHCBRT
  LOGICAL ROOTL

  PI=4.0*ATAN(1.0)
  ROOTL=.TRUE.
  PATHROOTTYPE=0

  TN=(-B)/(3.0*A)
  FTN=A*TN*TN*TN+B*TN*TN+C*TN+D
  DELTA_SQ=(B*B-3.0*A*C)/(9.0*A*A)
  H_SQ=4.0*(A*A)*(DELTA_SQ*DELTA_SQ*DELTA_SQ)

  !     IF (((H_SQ.LE.0).OR.(DELTA_SQ.LE.0)).AND.(FTN*FTN.LT.H_SQ)) THEN
  IF (.TRUE.) THEN
     CALL SOLVECUBICNEW(A,B,C,D,TVAL)
  ELSE
     !        WRITE(*,*) "SOLVECUBIC", A,B,C,D,FTN*FTN,H_SQ
     IF ((FTN*FTN.GT.H_SQ).AND.(ABS(FTN*FTN-H_SQ).GT.1E-10)) THEN
        !        WRITE(*,*) 'ONLY ONE REAL ROOT'
        CBRT1=((1.0/(2.0*A))*(-FTN+SQRT(FTN*FTN-H_SQ)))
        CBRT1=PATHCBRT(CBRT1)
        CBRT2=((1.0/(2.0*A))*(-FTN-SQRT(FTN*FTN-H_SQ)))
        CBRT2=PATHCBRT(CBRT2)
        ROOT(1)=TN+CBRT1+CBRT2
        !        WRITE(*,*) "ONE REAL ROOT",ROOT(1)
        IF ((ROOT(1).GE.0.0).AND.(ROOT(1).LE.1.0+PATHTOL)) THEN
           TVAL=ROOT(1)
           PATHROOTTYPE=1
        ELSE
           TVAL=7
        ENDIF
     ELSE IF ((FTN*FTN.EQ.H_SQ).OR.(ABS(FTN*FTN-H_SQ).LT.1E-10)) THEN
        H=SQRT(H_SQ)
        IF ((H.EQ.0.0).OR.(H.LT.1E-10)) THEN
           !           WRITE(*,*) 'The Three Real Roots are Equal'
           DO I=1,3
              ROOT(I)=TN
           ENDDO
           !           WRITE(*,*) ROOT(1),ROOT(2),ROOT(3)
           IF ((ROOT(1).GE.0.0).AND.(ROOT(1).LE.1.0+PATHTOL)) THEN
              TVAL=ROOT(1)
              PATHROOTTYPE=2
           ELSE
              TVAL=7
           ENDIF
        ELSE
           IF(FTN/(2*A).LT.0.0) THEN
              DELTA=(-1.0)*((-1.0)*FTN/(2*A))**(1.0/3.0)
           ELSE
              DELTA=(FTN/(2*A))**(1.0/3.0)
           ENDIF
           !           WRITE(*,*) 'There are Three Real Roots (Two Equal)'
           DO I=1,3
              IF (I.NE.3.0) THEN
                 ROOT(I)=TN+DELTA
              ELSE
                 ROOT(I)=TN-2*DELTA
              ENDIF
           ENDDO
           !           WRITE(*,*) ROOT(1),ROOT(2),ROOT(3)
           IF (((ROOT(1).GE.0.0).AND.(ROOT(1).LE.1.0+PATHTOL)).AND. &
                ((ROOT(3).GE.0.0).AND.(ROOT(3).LE.1.0+PATHTOL))) THEN
              CALL WRNDIE (-20,'<CSTRAN>','Multiple roots found.')
           ELSE IF (((ROOT(1).GE.0.0).AND. &
                (ROOT(1).LE.1.0+PATHTOL))) THEN
              TVAL=ROOT(1)
              PATHROOTTYPE=3
           ELSE IF (((ROOT(3).GE.0.0).AND. &
                (ROOT(3).LE.1.0+PATHTOL))) THEN
              TVAL=ROOT(3)
              PATHROOTTYPE=4
           ELSE
              TVAL=7
           ENDIF
        ENDIF
     ELSE IF (FTN*FTN.LT.H_SQ) THEN
        !        WRITE(*,*) 'THERE ARE THREE DISTINCT ROOTS'
        DELTA=SQRT((B*B-3.0*A*C)/(9.0*A*A))
        H=2.0*A*DELTA*DELTA*DELTA
        !        WRITE(*,*) "PROJECTION"
        !        WRITE(*,5432) A,B,C,D,DELTA,H
        ! 5432   FORMAT (6F8.1)
        THETA=(ACOS(-FTN/H))/3.0
        ROOT(1)=TN+2.0*DELTA*COS(THETA)
        ROOT(2)=TN+2.0*DELTA*COS(2.0*PI/3.0+THETA)
        ROOT(3)=TN+2.0*DELTA*COS(4.0*PI/3.0+THETA)
        DO I=1,3
           !           WRITE(*,*) ROOT(I)
           IF ((ROOT(I).GE.0.0).AND.(ROOT(I).LE.1.0+PATHTOL)) THEN
              IF (ROOTL) THEN
                 TVAL=ROOT(I)
                 PATHROOTTYPE=I+4
                 ROOTL=.FALSE.
              ELSE
                 CALL WRNDIE (-20,'<CSTRAN>','Multiple roots found.')
              ENDIF
           ELSE
              IF (ROOTL) THEN
                 TVAL=7
              ENDIF
           ENDIF
        ENDDO
     ENDIF

  ENDIF !GOES TO SOLVECUBICNEW
  RETURN
END SUBROUTINE SOLVECUBIC
!-----------------------------------------------------------------------

SUBROUTINE CUBICEVAL (TVAL,I,EVALX,EVALY,EVALZ)

  ! GIVEN THE SPLINE AND VALUE OF T, FIND THE X,Y,Z COORDINATES
  ! THIS ASSUMES THAT:
  ! 0.0 <= TVAL <= 1.0+PATHTOL

  use chm_kinds
  use dimens_fcm
  use cnst_fcm
  implicit none

  real(chm_real)  TVAL,EVALX,EVALY,EVALZ
  INTEGER I

  EVALX=PATHA(1,I,PATHN)*TVAL*TVAL*TVAL+PATHB(1,I,PATHN)*TVAL*TVAL &
       +PATHC(1,I,PATHN)*TVAL+PATHD(1,I,PATHN)
  EVALY=PATHA(2,I,PATHN)*TVAL*TVAL*TVAL+PATHB(2,I,PATHN)*TVAL*TVAL &
       +PATHC(2,I,PATHN)*TVAL+PATHD(2,I,PATHN)
  EVALZ=PATHA(3,I,PATHN)*TVAL*TVAL*TVAL+PATHB(3,I,PATHN)*TVAL*TVAL &
       +PATHC(3,I,PATHN)*TVAL+PATHD(3,I,PATHN)

  RETURN
END SUBROUTINE CUBICEVAL
!-----------------------------------------------------------------------

SUBROUTINE GLOBALCUBICEVAL (TVAL,EVALX,EVALY,EVALZ)

  ! GIVEN A GLOBAL TVAL, DETERMINE THE SPLINE PIECE AND
  ! X,Y,Z COORDINATES

  use chm_kinds
  use dimens_fcm
  use cnst_fcm
  implicit none

  real(chm_real)  TVAL,EVALX,EVALY,EVALZ,EVALT
  INTEGER I

  DO I=1,MAXNATMS-1
     IF(TVAL-I.LE.1.0) THEN
        EVALT=TVAL-I
        EXIT
     ENDIF
  ENDDO

  EVALX=PATHA(1,I,PATHN)*EVALT*EVALT*EVALT &
       +PATHB(1,I,PATHN)*EVALT*EVALT &
       +PATHC(1,I,PATHN)*EVALT+PATHD(1,I,PATHN)
  EVALY=PATHA(2,I,PATHN)*EVALT*EVALT*EVALT &
       +PATHB(2,I,PATHN)*EVALT*EVALT &
       +PATHC(2,I,PATHN)*EVALT+PATHD(2,I,PATHN)
  EVALZ=PATHA(3,I,PATHN)*EVALT*EVALT*EVALT &
       +PATHB(3,I,PATHN)*EVALT*EVALT &
       +PATHC(3,I,PATHN)*EVALT+PATHD(3,I,PATHN)

  RETURN
END SUBROUTINE GLOBALCUBICEVAL
!-----------------------------------------------------------------------
SUBROUTINE TINIFINAL (I,TINIAPPROX,J,K,COMX,COMY,COMZ,TVAL)

  ! ONCE TINIAPPROX HAS BEEN DETERMINED, PATHTINI IS FURTHER
  ! COMPUTED HERE SINCE TINIAPPROX WAS A CRUDE APPROXIMATION.
  ! TINIAPPROX USES THE ENDPOINTS OF A SPLINE PIECE AS THE
  ! TANGENT WHEREAS TINIFINAL USES THE POINT FOUND IN TINIAPPROX
  ! AS THE CENTER OF THE NEW TANGENT (THE NEW TANGENT IS
  ! DETERMINED BY ADDING AND SUBTRACTING SOME SMALL DELTA T FROM
  ! THE CENTER OF THE NEW TANGENT).  FROM THIS NEW TANGENT,
  ! PATHTINI IS FOUND AND SHOULD BE A BETTER AND MORE ACCURATE
  ! APPROXIMATION OF THE TRUE PROJECTION.

  use chm_kinds
  use dimens_fcm
  use stream
  use cnst_fcm
  implicit none

  INTEGER J,K,I

  real(chm_real)  TINIAPPROX(MAXNPATHS,MAXNCOM)
  real(chm_real)  COMX,COMY,COMZ,GX,GY,GZ
  real(chm_real)  TVAL,GXI,GXF,GYI,GYF,GZI,GZF
  real(chm_real)  A,B,C,D,EVALX,EVALY,EVALZ

  IF (((TINIAPPROX(PATHN,J)+SQRT(1.0/PATHK(PATHN,J)).GE. &
       PATHNATMS(PATHN))).OR. &
       (((TINIAPPROX(PATHN,J)-SQRT(1.0/PATHK(PATHN,J)) &
       .LT.1.0)))) THEN
     IF (PRNLEV.GT.2) &
          WRITE(OUTU,4598) PATHN,J
4598 FORMAT ("CSTRAN>    SPLINE=",I3,", COM=",I2)
     CALL WRNDIE (-20,'<CSTRAN>','Secant outside of spline range.' &
          //'  Use more atoms to define spline or remove COM.')
  ENDIF
  CALL GLOBALCUBICEVAL (TINIAPPROX(PATHN,J)-SQRT(1.0/PATHK(PATHN,J)) &
       ,GXI,GYI,GZI)
  CALL GLOBALCUBICEVAL (TINIAPPROX(PATHN,J)+SQRT(1.0/PATHK(PATHN,J)) &
       ,GXF,GYF,GZF)
  GX=GXF-GXI
  GY=GYF-GYI
  GZ=GZF-GZI
  A=GX*PATHA(1,K,PATHN)+GY*PATHA(2,K,PATHN)+GZ*PATHA(3,K,PATHN)
  B=GX*PATHB(1,K,PATHN)+GY*PATHB(2,K,PATHN)+GZ*PATHB(3,K,PATHN)
  C=GX*PATHC(1,K,PATHN)+GY*PATHC(2,K,PATHN)+GZ*PATHC(3,K,PATHN)
  D=GX*(PATHD(1,K,PATHN)-COMX)+GY*(PATHD(2,K,PATHN)-COMY) &
       +GZ*(PATHD(3,K,PATHN)-COMZ)
  CALL SOLVECUBIC (A,B,C,D,TVAL)

  RETURN
END SUBROUTINE TINIFINAL

#endif 
!-----------------------------------------------------------------------

SUBROUTINE ECFORCNEB(ATOMPR,JREP,KREP,NPAIR, &
     LNOROT,LNOTRN,LPRINT, &
     QFORCA,DXA,DYA,DZA,QFORCB,DXB,DYB,DZB, &
     DERMS,BMASS,TMASS,RA,RB,DRA,DRB, &
     PATANG,DRTEMP,DEVA,NREPL)   ! jwchu
  !-----------------------------------------------------------------------
  ! Harmonic restraints with best fit.  Compute and apply forces.
  !
  !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
  !  NPAIR    ->  The number of atoms pairs in the best fit analysis
  !  LNOROT   ->  flag to indicate no rotation in the best fit.
  !  LNOTRN   ->  flag to indicate no translation in the best fit.
  !  LPRINT   ->  Print flag (detailed results)
  !  QFORCA   ->  Compute forces for set A?
  ! DXA,DYA,DZA <->  Set A force vectors
  !  QFORCB   ->  Compute forces for set B?
  ! DXB,DYB,DZB <->  Set B force vectors
  !  DERMS    ->  dE/dRMS
  !  BMASS    ->  Final weighting array
  !  TMASS    ->  total weight
  !  RA       ->  VA - <VA>cm
  !  RB       ->  VB - <VB>cm
  !  DRA      -  (dE/dRA)
  !  DRB      -  (-dE/dRB)
  !
  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use psf
  use parallel
  implicit none

  INTEGER NPAIR,JREP,KREP,NREPL
  INTEGER ATOMPR(2,NPAIR)
  LOGICAL LNOROT,LNOTRN,LPRINT
  LOGICAL QFORCA,QFORCB
  real(chm_real)  DXA(*),DYA(*),DZA(*),DXB(*),DYB(*),DZB(*)
  real(chm_real)  DERMS,PATANG(3,NREPL,NPAIR)

  ! Temporary space arrays with data
  real(chm_real) BMASS(NPAIR),TMASS
  real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
  ! Temporary space arrays without data
  real(chm_real) DRA(3,NPAIR),DRB(3,NPAIR),DRTEMP(3,NPAIR)
  real(chm_real) DEVA(3,3)

  ! Local variables and arrays
  INTEGER ATFRST,ATLAST
  INTEGER K,KA,KB,I,J,II,JJ
  real(chm_real) TA,TB,KRMS
  real(chm_real) R(3,3)
  real(chm_real) DEU(3,3),DER(3,3),UT(3,3)
  real(chm_real) FA,FB,SUM
  FA=ZERO
  FB=ZERO

  ! don't do anything if neither force flag is set.
  IF(.NOT.(QFORCA.OR.QFORCB)) RETURN

  ! Define the atom bounds for this processor.
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /**/
  ATFRST=1
  ATLAST=NATOM
#endif 

  IF (LNOROT) THEN

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=(RA(I,K)-RB(I,K))*DERMS*BMASS(K)/TMASS
           DRB(I,K)=-DRA(I,K)
        ENDDO
     ENDDO

  ELSE

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=RA(I,K)
           DO J=1,3
              DRA(I,K)=DRA(I,K)-DEVA(I,J)*RB(J,K)
           ENDDO
           DRA(I,K)=DRA(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO

     IF(LPRINT) write(6,56) ' The RAx value:',RA
     IF(LPRINT) write(6,56) ' The RBx value:',RB
     IF(LPRINT) write(6,56) ' The DEVA value:',DEVA
     IF(LPRINT) write(6,56) ' The DRA value:',DRA
56   format(A/,(10X,3F12.5))

     ! compute dE/dRB(1) (negative)
     DO K=1,NPAIR
        DO I=1,3
           DRB(I,K)=RB(I,K)
           DO J=1,3
              DRB(I,K)=DRB(I,K)-DEVA(J,I)*RA(J,K)
           ENDDO
           DRB(I,K)=DRB(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO
     IF(LPRINT) write(6,56) ' The DRB value:',DRB

     ! compute dE/dVA
     ! compute dE/dVB
     !CC        IF(.NOT.LNOTRN) THEN
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              TA=0.0
     !CC              DO J=1,NPAIR
     !CC                TA=TA-DRA(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRA(I,K)-TA/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRA(I,K)=DRTEMP(I,K)
     !CC              TB=0.0
     !CC              DO J=1,NPAIR
     !CC                TB=TB-DRB(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRB(I,K)-TB/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRB(I,K)=DRTEMP(I,K)
     !CC            ENDDO
     !CC          ENDDO
     !CC        ENDIF
     IF(LPRINT) write(6,56) ' The DRA value:',DRA
     IF(LPRINT) write(6,56) ' The DRB value:',DRB

  ENDIF

  ! compute forces (as required)
  FA=ZERO
  IF(QFORCA) THEN
     DO K=1,NPAIR
        DO I=1,3
           FA=FA+DRA(I,K)*PATANG(I,JREP,K)
        ENDDO
     ENDDO
     DO K=1,NPAIR
        KA=ATOMPR(1,K)
        IF(KA.GE.ATFRST.AND.KA.LE.ATLAST) THEN
           DXA(KA)=DXA(KA)+FA*PATANG(1,JREP,K)
           DYA(KA)=DYA(KA)+FA*PATANG(2,JREP,K)
           DZA(KA)=DZA(KA)+FA*PATANG(3,JREP,K)
        ENDIF
     ENDDO
  ENDIF

  FB=ZERO
  IF(QFORCB) THEN
     DO K=1,NPAIR
        DO I=1,3
           FB=FB+DRB(I,K)*PATANG(I,KREP,K)
        ENDDO
     ENDDO
     DO K=1,NPAIR
        KB=ATOMPR(2,K)
        IF(KB.GE.ATFRST.AND.KB.LE.ATLAST) THEN
           DXB(KB)=DXB(KB)+FB*PATANG(1,KREP,K)
           DYB(KB)=DYB(KB)+FB*PATANG(2,KREP,K)
           DZB(KB)=DZB(KB)+FB*PATANG(3,KREP,K)
        ENDIF
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE ECFORCNEB

SUBROUTINE ECFORCA(ATOMPR,NPAIR,LNOROT,LNOTRN,LPRINT, &
     QFORCA,DXA,DYA,DZA,QFORCB,DXB,DYB,DZB, &
     DERMS,BMASS,TMASS,RA,RB,DRA,DRB,DRTEMP,DEVA, &
     QRPATH)
  !-----------------------------------------------------------------------
  ! it is the same as ECFORC but in parallel mode  ! jwchu
  ! Harmonic restraints with best fit.  Compute and apply forces.
  !
  !  ATOMPR(2,NPAIR) ->  The pointers into set A and B
  !  NPAIR    ->  The number of atoms pairs in the best fit analysis
  !  LNOROT   ->  flag to indicate no rotation in the best fit.
  !  LNOTRN   ->  flag to indicate no translation in the best fit.
  !  LPRINT   ->  Print flag (detailed results)
  !  QFORCA   ->  Compute forces for set A?
  ! DXA,DYA,DZA <->  Set A force vectors
  !  QFORCB   ->  Compute forces for set B?
  ! DXB,DYB,DZB <->  Set B force vectors
  !  DERMS    ->  dE/dRMS
  !  BMASS    ->  Final weighting array
  !  TMASS    ->  total weight
  !  RA       ->  VA - <VA>cm
  !  RB       ->  VB - <VB>cm
  !  DRA      -  (dE/dRA)
  !  DRB      -  (-dE/dRB)
  !
  use chm_kinds
  use number
  use stream
  use dimens_fcm
  use psf
  use parallel

  implicit none
  INTEGER NPAIR
  INTEGER ATOMPR(2,NPAIR)
  LOGICAL LNOROT,LNOTRN,LPRINT
  LOGICAL QFORCA,QFORCB,QRPATH
  real(chm_real)  DXA(*),DYA(*),DZA(*),DXB(*),DYB(*),DZB(*)
  real(chm_real)  DERMS

  ! Temporary space arrays with data
  real(chm_real) BMASS(NPAIR),TMASS
  real(chm_real) RA(3,NPAIR),RB(3,NPAIR)
  ! Temporary space arrays without data
  real(chm_real) DRA(3,NPAIR),DRB(3,NPAIR),DRTEMP(3,NPAIR)
  real(chm_real) DEVA(3,3)

  ! Local variables and arrays
  INTEGER K,KA,KB,I,J,II,JJ,ATFRST,ATLAST ! jwchu
  real(chm_real) TA,TB,KRMS
  real(chm_real) R(3,3)
  real(chm_real) DEU(3,3),DER(3,3),UT(3,3)

  ! don't do anything if neither force flag is set.
  IF(.NOT.(QFORCA.OR.QFORCB)) RETURN
  !
  ! NOTE from MH(Nov 04)
  !
  ! This code has many problems. So far it is fixed
  ! for parallel/parallel Replica Path with the QM
  ! codes. NEB still needs some work, I believe.
  ! Also: Check that all the data are available
  !       in this processor!!
  !
  ! Define the atom bounds for this processor.
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
  ATFRST=1+IPARPT(MYNOD)
  ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
  ATFRST=1
  ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /**/
  ATFRST=1
  ATLAST=NATOM
#endif 

#if KEY_REPLICA==1
#if KEY_PARALLEL==1
  ! QQMPAR not needed, just QRPATH!!!
  !     IF(QQMPAR.AND.QRPATH)THEN
  ! This is needed because RPATH is executed in one process only
  IF(QRPATH)THEN
     ATFRST=1
     ATLAST=NATOM
  ENDIF
#endif 
#endif 

  IF (LNOROT) THEN

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=(RA(I,K)-RB(I,K))*DERMS*BMASS(K)/TMASS
           DRB(I,K)=-DRA(I,K)
        ENDDO
     ENDDO

  ELSE

     ! compute dE/dD
     ! compute dE/dRA(1)
     DO K=1,NPAIR
        DO I=1,3
           DRA(I,K)=RA(I,K)
           DO J=1,3
              DRA(I,K)=DRA(I,K)-DEVA(I,J)*RB(J,K)
           ENDDO
           DRA(I,K)=DRA(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO

     IF(LPRINT) write(OUTU,56) ' The RAx value:',RA
     IF(LPRINT) write(OUTU,56) ' The RBx value:',RB
     IF(LPRINT) write(OUTU,56) ' The DEVA value:',DEVA
     IF(LPRINT) write(OUTU,56) ' The DRA value:',DRA
56   format(A/,(10X,3F12.5))

     ! compute dE/dRB(1) (negative)
     DO K=1,NPAIR
        DO I=1,3
           DRB(I,K)=RB(I,K)
           DO J=1,3
              DRB(I,K)=DRB(I,K)-DEVA(J,I)*RA(J,K)
           ENDDO
           DRB(I,K)=DRB(I,K)*DERMS*BMASS(K)/TMASS
        ENDDO
     ENDDO
     IF(LPRINT) write(OUTU,56) ' The DRB value:',DRB

     ! compute dE/dVA
     ! compute dE/dVB
     !CC        IF(.NOT.LNOTRN) THEN
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              TA=0.0
     !CC              DO J=1,NPAIR
     !CC                TA=TA-DRA(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRA(I,K)-TA/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRA(I,K)=DRTEMP(I,K)
     !CC              TB=0.0
     !CC              DO J=1,NPAIR
     !CC                TB=TB-DRB(I,J)*BMASS(J)
     !CC              ENDDO
     !CC              DRTEMP(I,K)=DRB(I,K)-TB/TMASS
     !CC            ENDDO
     !CC          ENDDO
     !CC          DO K=1,NPAIR
     !CC            DO I=1,3
     !CC              DRB(I,K)=DRTEMP(I,K)
     !CC            ENDDO
     !CC          ENDDO
     !CC        ENDIF
     IF(LPRINT) write(OUTU,56) ' The DRA value:',DRA
     IF(LPRINT) write(OUTU,56) ' The DRB value:',DRB

  ENDIF

  ! compute forces (as required)
  IF(QFORCA) THEN
     DO K=1,NPAIR
        KA=ATOMPR(1,K)
        IF(KA.GE.ATFRST.AND.KA.LE.ATLAST) THEN
           DXA(KA)=DXA(KA)+DRA(1,K)
           DYA(KA)=DYA(KA)+DRA(2,K)
           DZA(KA)=DZA(KA)+DRA(3,K)
        ENDIF
     ENDDO
  ENDIF

  IF(QFORCB) THEN
     DO K=1,NPAIR
        KB=ATOMPR(2,K)
        IF(KB.GE.ATFRST.AND.KB.LE.ATLAST) THEN
           DXB(KB)=DXB(KB)+DRB(1,K)
           DYB(KB)=DYB(KB)+DRB(2,K)
           DZB(KB)=DZB(KB)+DRB(3,K)
        ENDIF
     ENDDO
  ENDIF

  RETURN
END SUBROUTINE ECFORCA
