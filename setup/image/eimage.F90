module eimg
contains

SUBROUTINE EIMAGE(X,Y,Z,DX,DY,DZ,QECONT,ECONT,QSECD)
  !
  !     THIS ROUTINE HANDLES THE SETUP AND CALLING OF ENERGY ROUTINES
  !     TO DETERMINE THE IMAGE ENERGY CONTRIBUTION.
  !
  !     By Bernard R. Brooks    9/83
  !
  !     Jay L. Banks 19 October 1995: added calls to EOOPL and ESTRBND.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use eintern
  use energym
  use image
  use hbondm
  use param
  use psf
  use code
  use timerm
  use stream
  use parallel
  use tbmts
  use cmapm
#if KEY_MMFF==1
  use ffieldm
  use mmffm
  use escalar_mm
#endif 
#if KEY_BLOCK==1
  use block_fcm, only : setterm     
#endif
  use machutil,only:timre,timrb,wrttim
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec          
#endif
  implicit none
  !
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QECONT
  real(chm_real) ECONT(*)
  LOGICAL QSECD
  !
  real(chm_real) EBIM,ETIM,EPIM,EIIM,ETIMU
#if KEY_CMAP==1
  real(chm_real) EICT
#endif 
  LOGICAL TBM22,TBM44
  INTEGER J1,J3
  !
  !
#if KEY_MMFF==1
  real(chm_real) TMP ! temp for OOPL and STRBND energies
  INTEGER DERIVS
  !
  DERIVS=1
#endif 

#if KEY_DOMDEC==1
  if (q_domdec) then
     CALL WRNDIE(-5,'<EIMAGE>','NOT READY FOR DOMDEC')
  else
#endif 

  IF(NTRANS.EQ.0) RETURN
  IF(QSECD) CALL WRNDIE(-4,'<EIMAGE>','NO HESSIAN WITH IMAGES')
  !
  !--------------------------------------------------------------
  TBM22=.TRUE.
  TBM44=.TRUE.
  NBOMM=0
  NTUBMM=0
  NTHEMM=0
  NPHMM=0
  NIMMHM=0
#if KEY_CMAP==1
  NIMCTM=0
#endif 

  !
#if KEY_MTS==1
  IF (QTBMTS) THEN
     CALL SELMTS(TBM22,TBM44,J1,J3,.TRUE.)
  ELSE
#endif 
     NBOMM=NIMBON
     NTHEMM=NIMANG
     NPHMM=NIMDIH
     NIMMHM=NIMIMP
     J1=NBOND
     J3=NTHETA
     !
     NPHM=NPHI
     NIMPHM=NIMPHI
#if KEY_CMAP==1
     NIMCTM=NIMCRT
     NPCTM=NCRTERM
#endif 

#if KEY_MTS==1
  ENDIF
#endif 
  !
  !--------------------------------------------------------------
  !
  ! Do internal energy terms if present between primary and image atoms
  !
  EBIM=0.0
  IF(NBOMM.GT.0.AND.QETERM(BOND)) THEN
     CALL EBOND(EBIM,IB(J1+1),JB(J1+1),ICB(J1+1),NBOMM, &
          CBC,CBB,DX,DY,DZ,X,Y,Z, &
          QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),QSECD,2,.FALSE.)
     IF(TIMER.GT.1) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,2000) 'IMAGE BOND ENERGY TIMES:'
        CALL TIMRE
        CALL TIMRB
     ENDIF
     ETERM(BOND)=ETERM(BOND)+EBIM
  ENDIF
2000 FORMAT(1X,A)
  !
  ETIM=0.0
  ETIMU=0.0
  IF(NTHEMM.GT.0.AND.QETERM(ANGLE)) THEN
     IF(TBM22) THEN
        CALL EANGLE(ETIM,IT(J3+1),JT(J3+1),KT(J3+1), &
             ICT(J3+1),NTHEMM,CTC,CTB,DX,DY,DZ,X,Y,Z, &
             QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),QSECD &
             )
     ENDIF

     CALL EBOND(ETIMU,IT(J3+1),KT(J3+1),ICT(J3+1),NTHEMM, &
          CTUC,CTUB,DX,DY,DZ,X,Y,Z, &
          QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),QSECD,2,.FALSE.)

     ETIMU=ETIMU+ETIM
     IF(TIMER.GT.1) THEN
        IF(PRNLEV.GE.2) WRITE(OUTU,2000) 'IMAGE ANGLE ENERGY TIMES:'
        CALL TIMRE
        CALL TIMRB
     ENDIF
     ETERM(ANGLE)=ETERM(ANGLE)+ETIMU
  ENDIF
  !
  EPIM=0.0
  IF(TBM44) THEN
     IF(NIMDIH.GT.0.AND.QETERM(DIHE)) THEN
        !.ab.HYBH Assign terms. No block.f90, but energy.fcm, use SETTERM routine.
        !.ab. Most wrndie already in energy, should be Ok.
#if KEY_BLOCK==1
        CALL SETTERM(DIHE)    
#endif
        !.ab.
        CALL EPHI(EPIM,IP(NPHM+1),JP(NPHM+1),KP(NPHM+1),LP(NPHM+1), &
             ICP(NPHM+1),NPHMM,CPC,CPD,CPB,CPCOS,CPSIN,DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),QSECD &
             )

        IF(TIMER.GT.1) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,2000) 'IMAGE DIHEDRAL ENERGY TIMES:'
           CALL TIMRE
           CALL TIMRB
        ENDIF
        ETERM(DIHE)=ETERM(DIHE)+EPIM
     ENDIF
     !
     EIIM=0.0
     IF(NIMIMP.GT.0.AND.QETERM(IMDIHE)) THEN
        !.ab.
#if KEY_BLOCK==1
        CALL SETTERM(IMDIHE)  
#endif
        !.ab.
        CALL EPHI(EIIM,IM(NIMPHM+1),JM(NIMPHM+1),KM(NIMPHM+1), &
             LM(NIMPHM+1),ICI(NIMPHM+1),NIMMHM, &
             CIC,CID,CIB,CICOS,CISIN,DX,DY,DZ,X,Y,Z, &
             .FALSE.,(/ZERO/),QECONT,ECONT,0,(/0/),(/ZERO/),(/0/),QSECD &
             )

        IF(TIMER.GT.1) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,2000) &
                'IMAGE IMPROPER DIHEDERAL ENERGY TIMES:'
           CALL TIMRE
           CALL TIMRB
        ENDIF
        ETERM(IMDIHE)=ETERM(IMDIHE)+EIIM
     ENDIF

#if KEY_CMAP==1
     EICT=0.0
     IF(NIMCRT.GT.0.AND.QETERM(CMAP)) THEN
        !.ab.
#if KEY_BLOCK==1
        CALL SETTERM(CMAP)    
#endif
        !.ab.         what to do with cmap.... Nothing done so far.
        !.ab.         continue...
        !.ab.

        CALL ECMAP(EICT,I1CT(NPCTM+1),J1CT(NPCTM+1), &
             K1CT(NPCTM+1),L1CT(NPCTM+1), &
             I2CT(NPCTM+1),J2CT(NPCTM+1), &
             K2CT(NPCTM+1),L2CT(NPCTM+1), &
             ICCT(NPCTM+1),NIMCTM, &
             DX,DY,DZ,X,Y,Z, &
             QECONT,ECONT,0, (/0/), (/ZERO/), (/0/), QSECD)
        IF(TIMER.GT.1) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,2000) &
                'IMAGE CROSS TERM ENERGY TIMES:'
           CALL TIMRE
           CALL TIMRB
        ENDIF
        ETERM(CMAP)=ETERM(CMAP)+EICT
     ENDIF
#endif 

  ENDIF
  !
  ! Compute image hbond energies
  !
#if KEY_MTS==1
  IF ((ENE2 .AND. QTBMTS) .OR. (.NOT. QTBMTS)) THEN
#endif 
     ETERM(IMHBND)=0.0
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(19)) THEN
#endif 
        IF(NIMHB.GT.0.AND.QETERM(IMHBND)) THEN
           CALL EHBOND(ETERM(IMHBND),IHB(NHB+1),JHB(NHB+1),KHB(NHB+1), &
                LHB(NHB+1),ICH(NHB+1),NIMHB,CHBA,CHBB, &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT,0,0,0,0, &
                CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,QSECD)
           IF(TIMER.GT.1) THEN
              IF(PRNLEV.GE.2) WRITE(OUTU,2000) &
                   'IMAGE HYDROGEN BOND ENERGY TIMES:'
              CALL TIMRE
              CALL TIMRB
           ENDIF
        ENDIF
#if KEY_PARALLEL==1
     ENDIF
#endif 
#if KEY_MTS==1
  ENDIF
#endif 
  !
  !
#if KEY_MMFF==1
  if(FFIELD.eq.MMFF) then
     IF(NIMANG.GT.0) THEN
        IF(QETERM(OOPL)) THEN
           !     Out-off-plane energy (MMFF)
           TMP=0.0
           CALL EOOPL(TMP,IT(NTHETA+1),JT(NTHETA+1), &
                KT(NTHETA+1),LTHETA(NTHETA+1),icoop(NTHETA+1), &
                NIMANG,OoplFC,X,Y,Z,DX,DY,DZ, (/ ZERO /), DERIVS, &
                QECONT,ECONT,0, (/ 0 /))
           IF(TIMER.GT.1) CALL WRTTIM( &
                'IMAGE OUT-OFF-PLANE ENERGY TIMES:')
           ETERM(OOPL)=ETERM(OOPL)+TMP
        ENDIF
        !
        IF(NIMBON.GT.0.AND.QETERM(STRB)) THEN
           !     Strech-Bend coupling energy (MMFF)
           TMP=0.0
           CALL ESTRBND(TMP,IT(NTHETA+1),JT(NTHETA+1), &
                KT(NTHETA+1),ICT,NIMANG,CTB, &
                IB(NBOND+1),JB(NBOND+1),ICB(NBOND+1), &
                CBB,StrbList,ICSTBN,STBNP, &
                X,Y,Z,DX,DY,DZ, (/ ZERO /), DERIVS, &
                QECONT,ECONT,0, (/ 0 /))
           IF(TIMER.GT.1) CALL WRTTIM( &
                'IMAGE STRETCH-BEND ENERGY TIMES:')
           ETERM(STRB)=ETERM(STRB)+TMP
        ENDIF
     ENDIF
  endif
#endif /*  MMFF*/
#if KEY_DOMDEC==1
  endif  
#endif
  !
  RETURN
END SUBROUTINE EIMAGE

SUBROUTINE EIMNBD(EIMVDW,EIMEL,BNBND,BIMAG,BIMAGX, &
     IFRSTA,ILASTA,CG,RSCLF,ILASTG,IGPBS,IGPTYP, &
     IAC,IACNBX,DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
     QEWEX,EIMELX,QIMVDW,QIMEL, &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,    & 
#endif
     QIMST2,NST2,EIMST2 &
#if KEY_WCA==1
     ,WCA              & 
#endif
#if KEY_LJPME==1
     ,QLJEXC, ELJEXC  & 
#endif
     )
  !
  !     THIS ROUTINE HANDLES THE SETUP AND CALLING OF ENERGY ROUTINES
  !     TO DETERMINE THE IMAGE ENERGY CONTRIBUTION.
  !
  !     By Bernard R. Brooks    9/83
  !
#if KEY_LOOKUP==1
  use LOOKUP,only:elookup,qvv,nnbbycb,nnbibycb,qvv,iwwflg,qlookup,quu,nwwoim,nwwo, & 
     iwwo,iwoonbli,jwoonbli,ivunbli,jvunbli,iuunbli,juunbli,iwwenr,ewweeli,ewwenbi  
  use fast         
#endif
  use nb_module

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use enbond_mod
  use inbnd
  use image
  use timerm
  use stream

  use dcm_fcm,only:QDCM,DCMSETIM !MD 

#if KEY_GRAPE==1
  use grape,only:lgrape,igrape         
  use grapemod,only:qgpuene,gpuinblo   
#endif
  use datstr
  use machutil,only:timrb,timre
  use machutil,only:wrttim
     !---   use nbutil_module,only:setbnd,getbnd
     !
  implicit none
  !
  real(chm_real) EIMVDW,EIMEL,EIMST2
  type(nonbondDataStructure) BNBND
  type(imageDataStructure) BIMAG
  type(imageDataStructure) BIMAGX

  INTEGER IFRSTA, ILASTA
  INTEGER ILASTG,IGPBS(*),IGPTYP(*),IAC(*),IACNBX(*),NST2
  real(chm_real) CG(*), RSCLF(*)
  real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  LOGICAL QECONT,QEWEX,QIMVDW,QIMEL,QIMST2
  real(chm_real) EIMELX
  real(chm_real) ECONT(*)
  !
#if KEY_FLUCQ==1
  LOGICAL QFLUC              
#endif
#if KEY_FLUCQ==1
  real(chm_real) FQCFOR(*)           
#endif
#if KEY_WCA==1
  real(chm_real) WCA(*)              
#endif
  !
  real(chm_real) ENBX,EELX,EST2X
  INTEGER I,IMX
  real(chm_real),parameter :: DD0(1) = (/ ZERO /)
  integer,parameter :: IUPT0(1) = (/ 0 /)
  !
#if KEY_LOOKUP==1
  real(chm_real) EELW,ENBW
#endif 
#if KEY_LJPME==1
  LOGICAL QLJEXC
  real(chm_real) ELJEXC
#endif

  !
  !!      INTEGER BDUMMY(SNBNDT+1)
  !     Set up a dummy data structure for enbond.
  type(nonbondDataStructure) BDUMMY
  !
  IF(NTRANS.EQ.0) RETURN

  !!      CALL PRINTDT_nbond('EIMNBD_entry','BNBND',BNBND)
  !!      CALL PRINTDT_image('EIMNBD_entry','BIMAG',BIMAG)

  CALL ALIASDT_nbond(BDUMMY, BNBND)
  !     BDUMMY%NBDIST=BNBND%NBDIST
  !     BDUMMY%NBINTS=BNBND%NBINTS
  !     BDUMMY%LNBOPT=BNBND%LNBOPT
  !
  !     Make sure counters, flags and cuttoffs, etc are correct.
  CALL GETBND(BNBND,.TRUE.)
  !
  EIMVDW=ZERO
  EIMEL=ZERO
  EIMST2=ZERO
  ENBX=ZERO
  EELX=ZERO
  EST2X=ZERO

  CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAG)
  CALL NBSET_G14_FROM_IMG(BDUMMY, BIMAG)
  NNG14=BIMAG%NIMING
  !
  !     Check if any self-energy terms are present
  !
  IF (BIMAGX%NIMNBS.GT.0 .OR. BIMAGX%NIMNBX.GT.0) THEN
     !       Self terms are present
     IMX=NATIM
     DO I=1,IMX
        DX(I)=DX(I)*TWO
        DY(I)=DY(I)*TWO
        DZ(I)=DZ(I)*TWO
     ENDDO

     CALL NBSET_FROM_IMG_SX(BDUMMY, BIMAGX)
     NNNB =BIMAGX%NIMNBS
     NNNBG=BIMAGX%NIMNBX
     !
     !       We will not do the Ewald image exclusions here.
     !       Note: we need to create a special image self exclusion list to
     !       do this properly.  For now, EWALD must not be used if atoms
     !       are excluded from their own images!
     NNB14= 0
     !
     CALL SETBND(BDUMMY)
     IF(QDCM) CALL DCMSETIM(.TRUE.,NATIM) !WARN DCM ROUTINE THIS IS AN IMAGE RUN
     !
     CALL ENBOND(ENBX,EELX,BDUMMY,IFRSTA,ILASTA,CG,RSCLF, &
          ILASTG,IGPBS,IGPTYP,IAC,IACNBX,DX,DY,DZ,X,Y,Z, &
          QECONT,ECONT,.FALSE.,ZERO,DD0,IUPT0,.FALSE.,QIMVDW,QIMEL, &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR      ,   & 
#endif
          QIMST2,NST2,EST2X,.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA       & 
#endif
#if KEY_LJPME==1
          ,QLJEXC, ELJEXC  & 
#endif

          )

     !
     ENBX=ENBX*HALF
     EELX=EELX*HALF
     EST2X=EST2X*HALF
     DO I=1,IMX
        DX(I)=DX(I)*HALF
        DY(I)=DY(I)*HALF
        DZ(I)=DZ(I)*HALF
     ENDDO
     IF(QDCM) CALL DCMSETIM(.FALSE.,NATIM) !WARN DCM ROUTINE THIS IS AN IMAGE RUN
  ENDIF
  !
  !     compute image nonbonded energies
  !

  IF (BIMAGX%NIMNB.GT.0 .OR. BIMAGX%NIMNBG.GT.0) THEN
     !
     !       We do the Ewald image exclusions here.
     !       Note: we need to create a special image self exclusion list to
     !       do this properly.  Here we assume that none of the image exclusions
     !       are of the self type.
     NNB14=BIMAGX%NIMINB

     CALL NBSET_FROM_IMG_G(BDUMMY, BIMAGX)
     NNNB =BIMAGX%NIMNB
     NNNBG=BIMAGX%NIMNBG

     CALL SETBND(BDUMMY)
     !
     IF(QDCM) CALL DCMSETIM(.TRUE.,NATIM) !WARN DCM ROUTINE THIS IS AN IMAGE RUN
#if KEY_LOOKUP==1
     IF(.NOT.QUU)THEN                 
#endif
#if KEY_GRAPE==1
        qgpuene=.false.                                                             
#endif
#if KEY_GRAPE==1
        if(lgrape.and.(mod(igrape,10)==3))gpuinblo(1:ilasta)=bdummy%inblo(1:ilasta) 
#endif
        CALL ENBOND(EIMVDW,EIMEL,BDUMMY,IFRSTA,ILASTA,CG,RSCLF,ILASTG, &
             IGPBS,IGPTYP,IAC,IACNBX,DX,DY,DZ,X,Y,Z, &
             QECONT,ECONT,QEWEX,EIMELX,DD0,IUPT0,.FALSE.,QIMVDW,QIMEL, &
#if KEY_FLUCQ==1
             QFLUC,FQCFOR,         & 
#endif
             QIMST2,NST2,EIMST2,.FALSE. &
#if KEY_WCA==1
             ,.FALSE.,ONE,WCA       & 
#endif
#if KEY_LJPME==1
             ,QLJEXC, ELJEXC  & 
#endif

             )
        !
        IF (TIMER.GT.1) THEN
           IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,A)') &
                'IMAGE NONBOND AND ELECTROSTATIC ENERGY TIMES:'
           CALL TIMRE
           CALL TIMRB
        ENDIF
#if KEY_LOOKUP==1
     ENDIF                            
#endif
    IF(QDCM) CALL DCMSETIM(.FALSE.,NATIM) !WARN DCM ROUTINE THIS IS AN IMAGE RUN

  ENDIF


  !
  EIMVDW=EIMVDW+ENBX
  EIMEL =EIMEL +EELX
  EIMST2=EIMST2+EST2X
#if KEY_LOOKUP==1
  IF(QLOOKUP)THEN
     CALL ELOOKUP(ENBW,EELW,(NWWOIM-NWWO),IWWO(NWWO+1), &
          IWOONBLI,JWOONBLI,NATIM,IVUNBLI,JVUNBLI, &
          IUUNBLI,JUUNBLI)
     IF(IWWENR.GT.0)THEN
        EWWEELI=EELW
        EWWENBI=ENBW
     ELSE
        EELW=EWWEELI
        ENBW=EWWENBI
     ENDIF
     EIMEL=EIMEL+EELW
     EIMVDW=EIMVDW+ENBW
     IF(TIMER.GT.1) CALL WRTTIM('Image lookup energy times:')
  ENDIF
#endif 

  !
  call FREEDT_nbond(BDUMMY)
  !
  RETURN
END SUBROUTINE EIMNBD

SUBROUTINE LATTFD(XTLTYP,XTLABC,VIXX,DXTL,XTLREF)
  !
  !     Calculate the derivatives with respect to lattice parameters.
  !
  use chm_kinds
  use number
  use stream
  use timerm
  implicit none
  !
  CHARACTER(len=4) XTLTYP
  real(chm_real)      XTLABC(6),VIXX(9),DXTL(6),XTLREF(6)
  real(chm_real)      WRK1(9),XTLINV(6)
  LOGICAL     OK
  !
  CALL INVT33S(XTLINV,XTLABC,OK)
  CALL MULNXNFL(WRK1,VIXX,XTLINV,3)
  CALL LATTRN(XTLTYP,WRK1,DXTL,XTLREF)
  !
  RETURN
END SUBROUTINE LATTFD

SUBROUTINE LATTRN(XTLTYP,DA,DXTL,XTLREF)
  !
  !  Transform of force tensor from cartesian to lattice parameters.
  !
  use chm_kinds
  use number
  use stream
  use timerm
  use consta
  implicit none
  !
  CHARACTER(len=4) XTLTYP
  real(chm_real)      DA(3,3),DXTL(6),XTLREF(6)
  !
  INTEGER     I,J
  real(chm_real) COS15, SIN15, TANPI16, TANP2
  !
  ! Scale factors for averaging values removed 6/18/94 by BRB
  !
  DXTL(1:6)=zero
  IF (XTLTYP.EQ.'CUBI') THEN
     DXTL(1) = (DA(1,1)+DA(2,2)+DA(3,3))
  ELSEIF (XTLTYP.EQ.'RECT') THEN
     DXTL(1) = DA(1,1)*XTLREF(1)+DA(2,2)*XTLREF(2)+DA(3,3)*XTLREF(3)
  ELSEIF (XTLTYP.EQ.'TETR') THEN
     DXTL(1) = (DA(1,1)+DA(2,2))
     DXTL(2) = DA(3,3)
  ELSEIF (XTLTYP.EQ.'HEXA') THEN
     COS15=COS(PI/TWELVE)
     SIN15=SIN(PI/TWELVE)
     DXTL(1) = ((DA(1,1)+DA(2,2))*COS15- &
          (DA(1,2)+DA(2,1))*SIN15)
     DXTL(2) = DA(3,3)
  ELSEIF (XTLTYP .EQ. 'RHOM') THEN
     DXTL(1) = (DA(1,1)+DA(2,2)+DA(3,3))
     DXTL(2) = (DA(1,2)+DA(2,1)+DA(2,3)+ &
          DA(3,2)+DA(1,3)+DA(3,1))
  ELSEIF (XTLTYP .EQ. 'OCTA') THEN
     DXTL(1) = (DA(1,1)+DA(2,2)+DA(3,3)) -0.2* &
          (DA(1,2)+DA(2,1)+DA(2,3)+ &
          DA(3,2)+DA(1,3)+DA(3,1))
  ELSEIF (XTLTYP .EQ. 'RHDO') THEN
     TANPI16=PI/16.0
     TANPI16=TAN(TANPI16)
     TANP2=TANPI16**2
     DXTL(1) = (DA(1,1)+DA(2,2)*(ONE-TANP2)+DA(3,3) &
          + TANPI16*SQRT(TWO)*(DA(1,2)+DA(2,1)+DA(2,3)+DA(3,2)) &
          - TANP2*(DA(1,3)+DA(3,1)))/(ONE+TANP2)
  ELSEIF (XTLTYP .EQ. 'ORTH') THEN
     DXTL(1) = DA(1,1)
     DXTL(2) = DA(2,2)
     DXTL(3) = DA(3,3)
  ELSEIF (XTLTYP .EQ. 'MONO') THEN
     DXTL(1) = DA(1,1)
     DXTL(2) = DA(2,2)
     DXTL(3) = DA(3,3)
     DXTL(4) = (DA(1,3)+DA(3,1))
  ELSEIF (XTLTYP .EQ. 'TRIC') THEN
     DXTL(1) = DA(1,1)
     DXTL(2) = DA(2,2)
     DXTL(3) = DA(3,3)
     DXTL(4) = (DA(1,2)+DA(2,1))
     DXTL(5) = (DA(1,3)+DA(3,1))
     DXTL(6) = (DA(2,3)+DA(3,2))
  ELSE
     WRITE(OUTU,'(A,A)') ' LATTRN> XTLTYP=',XTLTYP
     CALL WRNDIE(-5,'<LATTRN>',' Unknown crystal type.')
  ENDIF
  IF (PRNLEV.GE.9) THEN
     WRITE(OUTU,'(/A)') ' LATTRN> Lattice parameter derivatives :'
     WRITE(OUTU,'(3F16.5)') (DXTL(I),I=1,6)
     WRITE(OUTU,'(/A,A4)') ' LATTRN> DA matrix; XTLTYP=',XTLTYP
     WRITE(OUTU,'(1X,A,I2,3F16.5)') &
          (' DA ROW:',I,(DA(I,J),J=1,3),I=1,3)
  ENDIF
  !
  return
end SUBROUTINE LATTRN
end module eimg

