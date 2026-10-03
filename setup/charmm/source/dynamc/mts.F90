module tbmts
  use chm_kinds
  use chm_types
  use tbmts_ltm
  implicit none
!
! Multiple Time Scale MD
! by Masa Watanabe
!
! logical
!  SLFG     - Flag indicating long/short range nonbond separation
!  SLFG1    -  use short distances for this energy call
!  SLFG2    -  use long  distances for this energy call
!  ENE1     - Flag to invoke calculations on the fastest bin.
!             (Shotest time-step)
!  ENE2     - Flag to invoke calculations on the middle bin.
!             (Medium time-step)
!  ENE3     - Flag to invoke calculations on the slowest bin
!             (Longest time-step)
! integer
!  NMTS1    - Factor of middle bin time-step. dt-middle = nmts1*dt
!  NMTS2    - Factor of slowest bin time-step. dt-slowest = nmts2*dt-middle
!  NMTS     - Factor of slowest bin time-step. dt-slowest = nmts*dt
!  IB1      - Number of bonds in the fastest bin.
!  IB2      - Number of bonds in the middle bin.
!  IB3      - Number of bonds in the slowest bin.
!  ITH1     - Number of angles in the fastest bin.
!  ITH2     - Number of angles in the middle bin.
!  TBM(5)   - Integer flag for separations of bonds.
!  TBM1(5)  - Integer flag for separations of angles.
!
! The following parameters are used for force separations of
!      long/short  nonbonded (VDW and electrostatics) interactions.
! Real
!  RSCUT    - Short range poteintial cutoff distance
!  RHEAL    - Switching function healing length
!  Useful parameters
!  RSCUT2   =  RSCUT * RSCUT
!  RSHL     =  RSCUT - RHEAL
!  RSHL2    =  RSHL * RSHL
!  RSCUT2T  =  (RSCUT - RHEAL - Buff)^2
!  RSHL2T   =  (RSCUT + Buff)^2
!         where Buff is a Buffer healing length to extend the nonbond
!               neighbor list for short range contribution.
! A. Sandu: QLNX If true, use LN
!           QIMPULSE: Use LN + impulse (default = extrapolation)
!
      INTEGER NBOMM, NBONM, NTUBMM, NPHMM, NPHM, &
              NIMPHM, NTHEMM, NIMMHM, NTHETM, NTHUBM
#if KEY_CMAP==1
      INTEGER NPCTM, NIMCTM  
#endif

#if KEY_MTS==1 /*mts_fcm*/
      LOGICAL SLFG,SLFG1,SLFG2
      LOGICAL ENE1,ENE2,ENE3
      INTEGER NMTS1,NMTS2,NMTS
      INTEGER IB1,IB2,IB3
      INTEGER ITH1,ITH2
      INTEGER TBM(5), TBM1(5)
      real(chm_real)  RSCUT,RHEAL,RSCUT2,RSHL,RSHL2,RSCUT2T,RSHL2T
      LOGICAL QLNX, QIMPULSE

      INTEGER NBONF, NBONS, NTHETF, NTHETS, NPHF, NPHS, &
              NIMPHF, NIMPHS, NBONQ, NBOMF, NBOMS, &
              NTHEMF, NTHEMS, NPHMF, NPHMS, NIMMHF, NIMMHS, &
              NBOMQ
#if KEY_CMAP==1
      INTEGER NPCTF, NPCTS, NIMCTF, NIMCTS  
#endif

!-------------------------------------------------------------
      integer, allocatable, dimension(:) :: &
              IBMS1, JBMS1, IBMS2, JBMS2, ICBM1, ICBM2, &
              ITMS1, JTMS1, KTMS1, ITMS2, JTMS2, KTMS2,  &
              ICTM1, ICTM2, JBMS3, ICBM3, IBMS3, &
              IBMS4, JBMS4, ICBM4, &
              ITMS4, JTMS4, KTMS4, ICTM4

!
!     This common block contains the base arrays of dynamically
!     allocated data structures which refer to the main coordinate set.
!     It is up to future programmers to see that they are maintained
!     with approriate dimensions for MTS.
!
! In CHARMM,
!  BNBND, LNBND   - Base arrays for nonbond interactions.
!  BIMAG, LIMAG   - Base arrays for images.
!
! For MTS
!  Fastest or Medium Time Step Bin:
!  BNBNM1, LNBNM1 - Base arrays for nonbond interactions in this bin.
!  BIMTS1, LIMTS1 - Base arrays for images in this bin.
!
!  Slowest Time Step Bin:
!  BNBNM2, LNBNM2 - Base arrays for nonbond interactions in this bin
!  BIMTS2, LIMTS2 - Base arrays for images in this bin.
!

  type(nonbondDataStructure),save :: BNBNM1, BNBNM2
  type(imageDataStructure),save :: BIMTS1, BIMTS2

#endif /* (mts_fcm)*/

contains

#if KEY_MTS==1 /*mts_main*/
  subroutine tbmts_iniall()
    qtbmts = .false.
    tbhy=.false.
    tbhy1=.false.
    tbhy2=.false.
    nmts=1
    nmts1=1
    nmts2=1
    ene1=.false.
    ene2=.false.
    ene3=.false.
    ib1=0
    ib2=0
    ib3=0
    ith1=0
    ith2=0
    return
  end subroutine tbmts_iniall

  subroutine allocate_tbmts()
    use dimens_fcm
    use memory
    character(len=*), parameter :: fname = 'mts.src', pname = 'allocate_tbmts'

    call chmalloc(fname,pname,'IMTS',MAXAIM,intg=IMTS)
    call chmalloc(fname,pname,'IMTF',MAXAIM,intg=IMTF)

    call chmalloc(fname,pname,'IBMS1',MAXB,intg=IBMS1)
    call chmalloc(fname,pname,'JBMS1',MAXB,intg=JBMS1)
    call chmalloc(fname,pname,'IBMS2',MAXB,intg=IBMS2)
    call chmalloc(fname,pname,'JBMS2',MAXB,intg=JBMS2)
    call chmalloc(fname,pname,'IBMS3',MAXB,intg=IBMS3)
    call chmalloc(fname,pname,'JBMS3',MAXB,intg=JBMS3)
    call chmalloc(fname,pname,'IBMS4',MAXB,intg=IBMS4)
    call chmalloc(fname,pname,'JBMS4',MAXB,intg=JBMS4)
    call chmalloc(fname,pname,'ICBM1',MAXB,intg=ICBM1)
    call chmalloc(fname,pname,'ICBM2',MAXB,intg=ICBM2)
    call chmalloc(fname,pname,'ICBM3',MAXB,intg=ICBM3)
    call chmalloc(fname,pname,'ICBM4',MAXB,intg=ICBM4)

    call chmalloc(fname,pname,'ICTM1',MAXT,intg=ICTM1)
    call chmalloc(fname,pname,'ICTM2',MAXT,intg=ICTM2)
    call chmalloc(fname,pname,'ICTM4',MAXT,intg=ICTM4)
    call chmalloc(fname,pname,'ITMS1',MAXT,intg=ITMS1)
    call chmalloc(fname,pname,'JTMS1',MAXT,intg=JTMS1)
    call chmalloc(fname,pname,'KTMS1',MAXT,intg=KTMS1)
    call chmalloc(fname,pname,'ITMS2',MAXT,intg=ITMS2)
    call chmalloc(fname,pname,'JTMS2',MAXT,intg=JTMS2)
    call chmalloc(fname,pname,'KTMS2',MAXT,intg=KTMS2)
    call chmalloc(fname,pname,'ITMS4',MAXT,intg=ITMS4)
    call chmalloc(fname,pname,'JTMS4',MAXT,intg=JTMS4)
    call chmalloc(fname,pname,'KTMS4',MAXT,intg=KTMS4)
  end subroutine allocate_tbmts

SUBROUTINE MTS(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This routine interprets commands dealing with 
  !     Multiple time scaled method
  !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  use dimens_fcm
  use chutil
  use number
  use energym
  use bases_fcm
  use inbnd
  use datstr
  use psf
  use shake
  use stream
  use string
  use image

  implicit none

  character(len=*) COMLYN
  INTEGER COMLEN
  !     local
  LOGICAL     QEND,QCHECK,EOF,LUSED
  LOGICAL     BINN
  INTEGER     NINT,JINT,I,J,K,ITEMP,JTEMP,NPM
  INTEGER     NPM1
  real(chm_real)      TLM,RR1,RR2,RR3
  character(len=4) WRD
  !
  !     begin
  BINN=.TRUE.
  TBHY=.FALSE.
  TBHY1=.FALSE.
  TBHY2=.FALSE.
  !
  SLFG=.FALSE.
  SLFG1=.FALSE.
  SLFG2=.FALSE.
  !
  EOF=.FALSE.
  QCHECK=.FALSE.
  QEND=.FALSE.
  QTBMTS = .TRUE.
  NMTS1=1
  NMTS2=1
  CALL TRIMA(COMLYN,COMLEN)
  IF (COMLEN > 0) NMTS1=NEXTI(COMLYN,COMLEN)
  IF (COMLEN > 0) THEN
     NMTS2=NEXTI(COMLYN,COMLEN)
  ELSE
     BINN = .FALSE.
  ENDIF
  !
  NMTS=NMTS1*NMTS2
  !
  IF(.NOT.BINN) NMTS2 = 0
  IF(PRNLEV >= 2) THEN
     IF(NMTS2 >= 1) THEN      
        WRITE(OUTU,988) NMTS,NMTS1,NMTS2
988     FORMAT(/,' MTS> Total cycle =',I3,'; first cycle =',I3, &
             '; Second cycle =',I3)
     ELSE
        WRITE(OUTU,9988) NMTS
9988    FORMAT(/,' MTS> Total cycle = ',I3)
     ENDIF
  ENDIF
  !
  IMTS(1:natom)=-1
  IMTF(1:natom)= 0
  !
  TBM(1:5)=-1
  TBM1(1:5)=-1   
  !
  lused=.true.
  do while(lused)
     CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
          EOF,.TRUE.,.TRUE.,'MTS> ')
     CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD  ==  '    ') LUSED=.TRUE.
  enddo
  !
  loop20: do while(.true.)
     !     Selection of component       
     IF (WRD  ==  'BOND') THEN
        I=NEXTI(COMLYN,COMLEN) 
        TBM(1)=1 
        TBHY=.FALSE.
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> Bond selection'
        ENDIF
     ELSE IF (WRD  ==  'BONH') THEN
        I=NEXTI(COMLYN,COMLEN) 
        TBM(1)=-1
        TBHY=.TRUE.
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> Bond involving hydrogen selection'
        ENDIF
     ELSE IF (WRD  ==  'ANGL') THEN 
        I=NEXTI(COMLYN,COMLEN)
        IF(BINN.AND.(I == 2)) THEN
           TBM1(2)=1
        ELSE IF(I == 1) THEN
           TBM(2)=1
        ENDIF
        IF(PRNLEV >= 2)  &
             WRITE(OUTU,*) ' MTS> Angle selection'      
     ELSE IF (WRD  ==  'ANGH') THEN
        IF(NMTS2 >= 1) THEN
           CALL WRNDIE(-2, 'MTS', 'Selection Error')
        ENDIF
        TBM(3)=1
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> Angle involving hydrogen selection'
        ENDIF
     ELSE IF (WRD  ==  'MASS') THEN
        I=NEXTI(COMLYN,COMLEN)
        TLM=NEXTF(COMLYN,COMLEN)
        TBHY1=.TRUE.
        IF(BINN.AND.(I == 2)) TBHY2=.TRUE. 
        IF(TLM <= 0) THEN
           TBHY1=.FALSE.
           TBHY2=.FALSE.
        ENDIF
        IF(PRNLEV >= 2) WRITE(OUTU,310) TLM
310     FORMAT(1X,'MTS> Nonbonded forces on mass under',F14.5,/, &
             '     for fast scaled motions')
     ELSE IF(WRD  ==  'SLFG') THEN
        RSCUT = GTRMF(COMLYN,COMLEN,'RSCU',SIX)
        RHEAL = GTRMF(COMLYN,COMLEN,'RHEA',ONE)
        RR3   = GTRMF(COMLYN,COMLEN,'BUFF',ONE)
        SLFG = .TRUE.
        !
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,319) RSCUT,RHEAL,RR3
319        FORMAT(1X,'MTS> Short-long range contribution separations',/, &
                '     Short range potential cutoff distance = ',F14.5,/, &
                '     Switching function healing length     = ',F14.5,/, &
                '     Buffer healing length                 = ',F14.5)
        ENDIF
        !
        RSHL = RSCUT - RHEAL
        !
        ! Addin buffer region
        !
        RR1 = RSHL - RR3
        RR2 = RSCUT + RR3
        !
        IMTS(1:natom)=1
        !
        IF(RSCUT <= 0.0) THEN
           RHEAL = 1.0
           RSHL = 0.0
           RSCUT = 0.0
           RR1 = 0.0
           RR2 = 0.0
        ENDIF
        !
        RSHL2 = RSHL * RSHL
        RSCUT2 = RSCUT * RSCUT
        RSHL2T = RR1 * RR1
        RSCUT2T = RR2 * RR2
        !
     ELSE IF(WRD  ==  'DIHE') THEN
        I=NEXTI(COMLYN,COMLEN)
        IF(BINN.AND.(I == 2)) THEN
           TBM1(4)=1
           TBM1(5)=1
        ELSE IF(I == 1) THEN
           TBM(4)=1
           TBM(5)=1
        ENDIF
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> Dihedral and Improper Selection'
        ENDIF
     ELSE IF(WRD  ==  'IMDI') THEN
        I=NEXTI(COMLYN,COMLEN)
        IF(BINN.AND.(I == 2)) THEN
           TBM1(4)=1
           TBM1(5)=1
        ELSE IF(I == 1) THEN
           TBM(4)=1
           TBM(5)=1
        ENDIF
        !
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> Dihedral and Improper Selection'
        ENDIF
     ELSE IF(WRD  ==  'ALL ') THEN
        I=NEXTI(COMLYN,COMLEN)
        IF(I <= 1) THEN 
           DO  J=1,5
              TBM(J)=1
           ENDDO
           TBM(3)=-1
        ELSE
           DO  J=1,5
              TBM1(J)=1
           ENDDO
           TBM1(3)=-1
        ENDIF
        IF(PRNLEV >= 2) THEN
           WRITE(OUTU,*) ' MTS> All internal-motion selection'
        ENDIF
     ELSE IF (WRD  ==  'CLEA') THEN
        QTBMTS = .FALSE.
        TBHY=.FALSE.       
        TBHY1=.FALSE.       
        TBHY2=.FALSE.
        SLFG=.FALSE.
        SLFG1=.FALSE.
        SLFG2=.FALSE.
        IF(PRNLEV >= 2) WRITE(OUTU,918)
        TBM(1:5)=-1
        TBM1(1:5)=-1
918     FORMAT(/,' MTS> Multi-time step method is clear now!! ')
     ELSE IF (WRD  ==  'END ') THEN
        !     Procedure FINISH-UP-AND-END
        CALL XTRANE(COMLYN,COMLEN,'MTS ')
        exit loop20
        !
     ENDIF
     !
     CALL XTRANE(COMLYN,COMLEN,'MTS ')
     IF (QEND) THEN
        WRD='END '
     ELSE
        lused=.true.
        do while(lused)
           CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM, &
                EOF,.TRUE.,.TRUE.,'MTS> ')
           CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
           WRD=NEXTA4(COMLYN,COMLEN)
           IF (WRD  ==  '    ') LUSED=.TRUE.
        enddo
     ENDIF
  enddo loop20
  !
  !------------------------------------------------------ 
  DO I=1,NBOND
     IF(TBM(1) > 0) THEN
        J=IB(I)
        K=JB(I)
        IMTS(J)=1
        IMTS(K)=1
     ELSE IF(TBHY) THEN
        J=IB(I)
        K=JB(I)
        IF(HYDROG(J).OR.HYDROG(K)) THEN
           IMTS(J)=1
           IMTS(K)=1
        ENDIF
     ENDIF
  enddo
  !-------------------------------------------------------
  DO I=1,NTHETA
     IF((TBM1(2) > 0).OR.(TBM(2).GT.0)) THEN
        J=IT(I)
        K=KT(I)
        IMTS(J)=1
        IMTS(K)=1
     ELSE IF(TBM(3) > 0) THEN
        J=IT(I)
        K=KT(I)
        IF(HYDROG(J).OR.HYDROG(K)) THEN
           IMTS(J)=1
           IMTS(K)=1
        ENDIF
     ENDIF
  enddo
  !------------------------------------------------------- 
  NPM=0
  NPM1=NATOM
  IF(TBHY1) THEN
     DO I=1,NATOM
        IMTS(I)=1
        IF(AMASS(I) < TLM .AND. .NOT.LONE(I)) THEN
           NPM=NPM+1
           IMTF(I)=1
        ENDIF
        !       IF(HYDROG(I)) IMTF(I)=1
     enddo
     NPM1=NPM1-NPM
  ENDIF
  !
  IF(SLFG.OR.TBHY1) THEN
     ! remove current nonbond list 
     CALL RENBND(BNBND,0,NATOM,NGRP,0)      
     ! duplicate nonbond list structure       
     CALL FREEDT_nbond(BNBNM1)
     CALL FREEDT_nbond(BNBNM2)
     CALL DUPLDT_nbond(BNBNM1,BNBND)
     CALL DUPLDT_nbond(BNBNM2,BNBND)
  ENDIF
  !mw...B980917.mw w/o MTS image generation fixes by separating the
  !mw...           following IF-block
  ! image
  IF(NTRANS.GT.0) THEN
     CALL FREEDT_image(BIMTS1)
     CALL FREEDT_image(BIMTS2)
     CALL DUPLDT_image(BIMTS1,BIMAG)
     CALL DUPLDT_image(BIMTS2,BIMAG)
  ENDIF
  !
  !--------------------------------------------------------
  !
  ! Teatment of shake
  !
  IF(QSHAKE) THEN
     IF(.NOT.SLFG) THEN
        DO I=1,NCONST
           J=SHKAPR(1,I)
           K=SHKAPR(2,I)
           IMTS(J)=-1
           IMTS(K)=-1
           IF(TBHY1) THEN
              IF(HYDROG(J)) IMTF(J)=0
              IF(HYDROG(K)) IMTF(K)=0      
           ENDIF
        ENDDO
     ELSE
        DO I=1,NCONST
           J=SHKAPR(1,I)
           K=SHKAPR(2,I)
           !             IMTM(J)=2
           !             IMTM(K)=2
        ENDDO
     ENDIF
  ENDIF
  !
  IF (NMTS1 == 1 .AND. NMTS2 <= 1) QTBMTS = .FALSE.
  !
  IF(TBHY1.AND.SLFG) THEN
     CALL WRNDIE(-2, 'MTS', 'Selection Error')
  ENDIF
  !
  IF (QTBMTS) CALL DFININ(NPM,NPM1)
  !
  RETURN
END SUBROUTINE MTS

SUBROUTINE DFININ(NP1,NP2) 
  use chm_kinds
  use dimens_fcm
  use chutil
  use psf
  use code
  use param
  use stream
  use image
  !
  implicit none
  INTEGER NP1,NP2
  !
  INTEGER I,MM,J,K,L,IC
  INTEGER JJ,NN
  INTEGER AA,BB,CC,DD,EE,FF,QK
#if KEY_CMAP==1
  INTEGER GG,HH
#endif 
  character(len=12) A1,A2,A3,B1,B2,B3
  !
  NBONF=0
  NBONS=0
  NBONQ=0      
  nose_ne_0: IF (NBOND /= 0) then
     loop10: do MM=1,NBOND 
        I=IB(MM)
        J=JB(MM)
        IC=ICB(MM)
        !
        IF(TBHY) THEN
           IF(HYDROG(I).OR.HYDROG(J)) THEN
              NBONF=NBONF+1 
              IBMS1(NBONF)=I
              JBMS1(NBONF)=J
              ICBM1(NBONF)=IC
           ELSE
              IF(NMTS2 < 1) THEN 
                 NBONS=NBONS+1 
                 IBMS2(NBONS)=I
                 JBMS2(NBONS)=J
                 ICBM2(NBONS)=IC
              ELSE
                 NBONQ=NBONQ+1     
                 IBMS3(NBONQ)=I
                 JBMS3(NBONQ)=J
                 ICBM3(NBONQ)=IC
              ENDIF
           ENDIF
        ELSE
           NBONF=NBONF+1 
           IBMS1(NBONF)=I
           JBMS1(NBONF)=J
           ICBM1(NBONF)=IC
        ENDIF
     enddo loop10
     !
  endif nose_ne_0
  !      
  NTHETF=0
  NTHETS=0
  nth_ne_0: IF (NTHETA /= 0) then
     IF((TBM(2) < 0).AND.(TBM(3).LT.0) &
          .AND.(TBM1(2) < 0)) THEN      
        DO MM=1,NTHETA
           I=IT(MM)
           J=JT(MM)
           K=KT(MM)
           IC=ICT(MM)       
           ITMS2(MM)=I
           JTMS2(MM)=J      
           KTMS2(MM)=K      
           ICTM2(MM)=IC
        enddo
        NTHETS=NTHETA
     ELSE
        DO MM=1,NTHETA
           I=IT(MM)
           J=JT(MM)
           K=KT(MM)
           IC=ICT(MM)      
           IF(TBM(3) > 0) THEN      
              IF(HYDROG(I).OR.HYDROG(J).OR.HYDROG(K)) THEN 
                 NTHETF=NTHETF+1       
                 ITMS1(NTHETF)=I
                 JTMS1(NTHETF)=J
                 KTMS1(NTHETF)=K       
                 ICTM1(NTHETF)=IC
              ELSE
                 NTHETS=NTHETS+1 
                 ITMS2(NTHETS)=I
                 JTMS2(NTHETS)=J
                 KTMS2(NTHETS)=K       
                 ICTM2(NTHETS)=IC
              ENDIF
           ELSE
              NTHETF=NTHETF+1    
              ITMS1(NTHETF)=I
              JTMS1(NTHETF)=J
              KTMS1(NTHETF)=K       
              ICTM1(NTHETF)=IC
           ENDIF
        enddo
     ENDIF
     !
  endif nth_ne_0
  !

  NPHF=0
  NPHS=0
  NIMPHF=0
  NIMPHS=0
#if KEY_CMAP==1
  NPCTF=0
  NPCTS=0
#endif 
  nphi_ne_0: IF(NPHI /= 0) then
     IF((TBM(4) > 0).OR.(TBM1(4).GT.0)) THEN
        NPHF=NPHI
     ELSE
        NPHS=NPHI
     ENDIF
  endif nphi_ne_0
  !
  nimp_ne_0: IF(NIMPHI /= 0) then
     IF((TBM(4) > 0).OR.(TBM1(4).GT.0)) THEN
        NIMPHF=NIMPHI
     ELSE
        NIMPHS=NIMPHI
     ENDIF
     !      DO 12 IPHI=1,NPHI
     !      I=IP(IPHI)
     !      J=JP(IPHI) 
     !      K=KP(IPHI)
     !      L=LP(IPHI)
     !      IPMS(IPHI)=1 
     !      IF(HYDROG(I).OR.HYDROG(J).OR.HYDROG(K).OR.
     !     &     HYDROG(L)) THEN
     !      IPMS(IPHI)=-1
     !      ENDIF
     !12    CONTINUE
  endif nimp_ne_0

#if KEY_CMAP==1
  IF(NCRTERM /= 0) then
     IF((TBM(4) > 0).OR.(TBM1(4).GT.0)) THEN
        NPCTF=NCRTERM
     ELSE
        NPCTS=NCRTERM
     ENDIF
  endif
#endif 
  !
  !---- Initialize ------------------------------
  !
  NBOMF=0
  NBOMS=0
  NBOMQ=0
  NTHEMF=0
  NTHEMS=0
  NPHMF=0
  NPHMS=0
  NIMMHF=0
  NIMMHS=0
  !
  IB1=NBONF
  IB2=NBONS
  IB3=NBONQ
  ITH1=NTHETF
  ITH2=NTHETS
  !----------------------------------------
  !
  ! Image pointers
  !
  ntrans_gt_0: IF(NTRANS > 0) then
     JJ=NIMBON+NIMANG+NIMDIH+NIMIMP
     !
     nimb_ne_0: IF (NIMBON /= 0) then
        loop100: DO NN=1,NIMBON 
           MM=NBOND+NN
           I=IB(MM)
           J=JB(MM)
           IC=ICB(MM)
           !
           IF(TBHY) THEN
              IF(HYDROG(I).OR.HYDROG(J)) THEN
                 NBOMF=NBOMF+1
                 IB1=IB1+1
                 IBMS1(IB1)=I
                 JBMS1(IB1)=J
                 ICBM1(IB1)=IC
              ELSE
                 IF(NMTS2 < 1) THEN
                    IB2=IB2+1
                    NBOMS=NBOMS+1
                    IBMS2(IB2)=I
                    JBMS2(IB2)=J
                    ICBM2(IB2)=IC
                 ELSE
                    IB3=IB3+1
                    NBOMQ=NBOMQ+1
                    IBMS3(IB3)=I
                    JBMS3(IB3)=J
                    ICBM3(IB3)=IC
                 ENDIF
              ENDIF
           ELSE
              IB1=IB1+1
              NBOMF=NBOMF+1
              IBMS1(IB1)=I
              JBMS1(IB1)=J
              ICBM1(IB1)=IC
           ENDIF
        enddo loop100
        !
     endif nimb_ne_0
     !      
     nimang_ne_0: IF (NIMANG /= 0) then
        IF((TBM(2) < 0) .AND. (TBM(3).LT.0)) THEN
           DO  NN=1,NIMANG
              MM=NTHETA+NN
              I=IT(MM)
              J=JT(MM)
              K=KT(MM)
              IC=ICT(MM)       
              ITMS2(MM)=I
              JTMS2(MM)=J
              KTMS2(MM)=K
              ICTM2(MM)=IC
           enddo
           NTHEMS=NIMANG
        ELSE
           DO  NN=1,NIMANG
              MM=NTHETA+NN
              I=IT(MM)
              J=JT(MM)
              K=KT(MM)
              IC=ICT(MM) 
              IF(TBM(3) > 0) THEN      
                 IF(HYDROG(I).OR.HYDROG(J).OR.HYDROG(K)) THEN 
                    NTHEMF=NTHEMF+1       
                    ITH1=ITH1+1
                    ITMS1(ITH1)=I
                    JTMS1(ITH1)=J
                    KTMS1(ITH1)=K       
                    ICTM1(ITH1)=IC
                 ELSE
                    NTHEMS=NTHEMS+1 
                    ITH2=ITH2+1
                    ITMS2(ITH2)=I
                    JTMS2(ITH2)=J
                    KTMS2(ITH2)=K       
                    ICTM2(ITH2)=IC
                 ENDIF
              ELSE
                 NTHEMF=NTHEMF+1    
                 ITH1=ITH1+1
                 ITMS1(ITH1)=I
                 JTMS1(ITH1)=J
                 KTMS1(ITH1)=K       
                 ICTM1(ITH1)=IC
              ENDIF
           enddo
        ENDIF
        !
     endif nimang_ne_0
     !
     nimdih_ne_0: IF(NIMDIH /= 0) then
        IF(TBM(4) > 0) THEN
           NPHMF=NIMDIH 
        ELSE
           NPHMS=NIMDIH
        ENDIF
     endif nimdih_ne_0
     !
     IF(NIMIMP /= 0) then
        IF(TBM(4) > 0) THEN
           NIMMHF=NIMIMP
        ELSE
           NIMMHS=NIMIMP
        ENDIF
     endif
  endif ntrans_gt_0
  !
  CALL DFININ2
  !
  IF(TBM1(2) > 0) THEN
     AA=0
     BB=NTHETF
  ELSE
     AA=NTHETF
     BB=0
  ENDIF
  !
  QK=0
  !
  IF(TBM1(4) > 0) THEN
     CC=0
     DD=NPHF
     EE=NIMPHF
     FF=0
#if KEY_CMAP==1
     GG=0
     HH=NPCTF
#endif 
  ELSE
     CC=NPHF
     DD=0
     FF=NIMPHF      
     EE=0
#if KEY_CMAP==1
     GG=NPCTF
     HH=0
#endif 
  ENDIF
  !
  IF(PRNLEV >= 2) THEN
     WRITE(OUTU,1003)
1003 FORMAT(//,'** Primary atom internal districution **') 
     !
     WRITE(OUTU,1001)      
     WRITE(OUTU,9810) '    BOND ', NBOND,NBONF,NBONQ,NBONS
     WRITE(OUTU,9810) '    ANGLE', NTHETA,AA,BB,NTHETS
     WRITE(OUTU,9810) '    U-B  ', NTHETA,AA,BB,NTHETS
     WRITE(OUTU,9810) '    DIHE ', NPHI,CC,DD,NPHS
     WRITE(OUTU,9810) '    IMPRO',  NIMPHI,FF,EE,NIMPHS
#if KEY_CMAP==1
     WRITE(OUTU,9810) '    CMAP ',  NCRTERM,GG,HH,NPCTS
#endif 
     !
     IF((NTRANS > 0).AND.(JJ.GT.0)) THEN 
        WRITE(OUTU,1002)       
        WRITE(OUTU,9811) '    BOND ', NIMBON,NBOMF,NBOMS
        WRITE(OUTU,9811) '    ANGLE', NIMANG,NTHEMF,NTHEMS 
        WRITE(OUTU,9811) '    U-B  ', NIMANG,NTHEMF,NTHEMS
        WRITE(OUTU,9811) '    DIHE ', NIMDIH,NPHMF,NPHMS 
        WRITE(OUTU,9811) '    IMPRO', NIMIMP,NIMMHF,NIMMHS 
     ENDIF
1002 FORMAT(/,'** Image atom internal motion distribution **')
     !
  ENDIF
  IF(.NOT.SLFG) THEN
     A1='            '
     A2='            '
     A3='            '
     B1='            '
     B2='            '     
     B3='            '
     IF(TBHY1) THEN
        IF(TBHY2) THEN
           A2='          XX'
           B3='          XX'
        ELSE
           A1='          XX'
           B3='          XX' 
        ENDIF
     ELSE
        A3='          XX'         
        B3='          XX' 
     ENDIF
     !
     ! Print out
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,1004)
        WRITE(OUTU,*) 'Light ATM', B2,A1,A2,A3
        WRITE(OUTU,*) 'Others   ', B2,B1,B2,B3
        WRITE(OUTU,321) NP1
        WRITE(OUTU,322) NP2
     ENDIF
  ELSE
     IF(PRNLEV >= 2) WRITE(OUTU,323)
  ENDIF
  !
323 FORMAT(//,2X,'SHORT-LONG Range force separations are applied ')
  !
1001 FORMAT(/,2X,'INT. FORC.',2X,'TOTAL DEG.',2X,'FAST-1 DEG.', &
       2X,'FAST-2 DEG.',2X,'SLOW DEG.')
9810 FORMAT(A,4(2X,I10))
9811 FORMAT(A,3(2X,I10))
1004 FORMAT(/,' *** Nonbond force distribution ***',/)
321 FORMAT(' # OF ATOM for fast scaled motions',I10)
322 FORMAT(' # of ATOM for slow scaled motions',I10,/)
  !
  RETURN
END SUBROUTINE DFININ
#else /* (mts_main)*/

SUBROUTINE MTS(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This routine interprets commands dealing with 
  !     Multiple time scaled method
  !-----------------------------------------------------------------------
  use chm_kinds
  use chm_types
  !
  character(len=*) COMLYN
  INTEGER COMLEN
  !
  CALL WRNDIE(0,'<MTS>','MTS code not compiled')
  return
end SUBROUTINE MTS
#endif /* (mts_main)*/






SUBROUTINE GRAM( &
#if KEY_MTS==1
     XMI,YMI,ZMI,XMM,YMM,ZMM, &    
#endif
     DX, DY, DZ &
#if KEY_DHDGB==1
!AP/MF
     ,DS_DHDGB &
#endif
    )
  !
  ! .calculate EPROP(GRMS) for velocity 
  !  Verlet method
  !  by Masa watanabe
  !
  use chm_kinds
  use dimens_fcm
  use number
  use energym
  use psf
#if KEY_PARALLEL==1
  use parallel   
#endif
#if KEY_DHDGB==1
!AP/MF
  use dhdgb,only:qfhdgb,totals
#endif

  implicit none
  !
#if KEY_DHDGB==1
!AP/MF
  real(chm_real),optional :: DS_DHDGB(*)
#endif
  real(chm_real) DX(*),DY(*),DZ(*)
  INTEGER I
  real(chm_real) S,NDIM
#if KEY_MTS==1
  real(chm_real) XMI(*),YMI(*),ZMI(*)
  real(chm_real) XMM(*),YMM(*),ZMM(*)       
  real(chm_real) DXD,DYD,DZD
#endif 
  !
  !mw parallel implementation
#if KEY_PARALLEL==1
  real(chm_real) GCARR(11),TIMMER
#endif 
  !
  INTEGER ATFRST,ATLAST
  !
#if KEY_PARALLEL==1 /*paramain*/
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
#else /* (paramain)*/
  ATFRST=1
  ATLAST=NATOM
#endif /* (paramain)*/
  !
  NDIM=0
  S=ZERO
  EPROP(GRMS)=ZERO
#if KEY_MTS==1 /*mts_a*/
  IF (QTBMTS) THEN
     !        DO I=1,NATOM
     DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              IF(NMTS2 >= 1) THEN      
                 DXD=DX(I)+XMI(I)+XMM(I)
                 DYD=DY(I)+YMI(I)+YMM(I)
                 DZD=DZ(I)+ZMI(I)+ZMM(I)
              ELSE
                 DXD=DX(I)+XMI(I)
                 DYD=DY(I)+YMI(I)
                 DZD=DZ(I)+ZMI(I)
              ENDIF
              !
              NDIM=NDIM+3.0
              S=S+DXD*DXD+DYD*DYD+DZD*DZD
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
  ELSE
     !        DO I = 1, NATOM
     DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
        IF(JPBLOCK(I) == MYNOD) THEN
#endif 
           IF(IMOVE(I) == 0) THEN
              NDIM=NDIM+3.0
              S=S+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)
           ENDIF
#if KEY_PARASCAL==1
        ENDIF
#endif 
     ENDDO
  ENDIF
#else /* (mts_a)*/
  !        DO I = 1, NATOM
  !         IF(IMOVE(I) == 0) THEN
  !           NDIM=NDIM+3
  !           S=S+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)
  !         ENDIF 
  !        ENDDO
  DO I=ATFRST,ATLAST
#if KEY_PARASCAL==1
     IF(JPBLOCK(I) == MYNOD) THEN
#endif 
        IF(IMOVE(I) == 0) THEN
           NDIM=NDIM+3.0
           S=S+DX(I)*DX(I)+DY(I)*DY(I)+DZ(I)*DZ(I)
        ENDIF
#if KEY_PARASCAL==1
     ENDIF
#endif 
  ENDDO
#endif /* (mts_a)*/
  !
#if KEY_DHDGB==1
!AP/MF
          IF (QFHDGB) THEN
              DO I=1,TOTALS
                 NDIM=NDIM+1
                 S=S+DS_DHDGB(I)**2
              ENDDO
          ENDIF
#endif
#if KEY_PARALLEL==1
  GCARR(1) = S
  GCARR(2) = NDIM
  CALL GCOMB(GCARR,2)
  S = GCARR(1)
  NDIM = GCARR(2)
#endif 
  IF(NDIM > 0.0) EPROP(GRMS)=SQRT(S/NDIM)
  !
  RETURN
END SUBROUTINE GRAM

SUBROUTINE DYNASHK(VX,VY,VZ,XNEW,YNEW,ZNEW, &
     AMASS,IMOVE,ISKP,NATOM,DELTA)
  ! 
  ! .calculation of SHAKE and forces from SHAKE application
  !  for Velocity Verlet Method
  !
  use chm_kinds
  use dimens_fcm
  use cnst_fcm
  use contrl
  use coord
  use number
  use shake
  use stream
  use parallel
  use holonom,only:holonoma
#if KEY_DOMDEC==1
  use domdec_common,only:natoml,atoml,q_domdec  
#endif
  implicit none
  real(chm_real) VX(*),VY(*),VZ(*),XNEW(*),YNEW(*),ZNEW(*)
  real(chm_real) VX1,VY1,VZ1
  real(chm_real) AMASS(*)
  INTEGER IMOVE(*),ISKP(*)
  real(chm_real)  DELTA 
  INTEGER NATOM
  !
  real(chm_real) DELTA2,DELTAS,FACT,FACT1
  INTEGER I,J,K
  
  real(chm_real),allocatable,dimension(:) :: xxi,yyi,zzi
  integer :: alloc_err
  INTEGER ATFRST,ATLAST, ia
  LOGICAL QOK
  !--------------------------------------------------
  allocate(xxi(natom),yyi(natom),zzi(natom),stat=alloc_err)
  if(alloc_err /= 0)then
     write(outu,'(a)')"cannot allocate tmp crd arrays in mts.src"
     CALL WRNDIE(0,'<CVELOCI>','ATOM SELECTION ERROR')
  endif

#if KEY_DOMDEC==1
  if (q_domdec) then
     atfrst = 1
     atlast = natoml
  else
#endif 
#if KEY_PARALLEL==1 /*paramain*/
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
#else /* (paramain)*/
     ATFRST=1
     ATLAST=NATOM
#endif /* (paramain)*/
#if KEY_DOMDEC==1
  endif  
#endif
  !---------------------------------------------------
  DELTAS=HALF*DELTA
  DELTA2=DELTA*DELTAS
  FACT1=1.0/DELTA
  ! 
  !          DO  I=1,NATOM

  do ia=atfrst,atlast
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif 
        i = ia
#if KEY_DOMDEC==1
     endif  
#endif
#if KEY_PARASCAL==1
     IF(JPBLOCK(I) == MYNOD) THEN  
#endif
        XXI(I) = XNEW(I)
        YYI(I) = YNEW(I)
        ZZI(I) = ZNEW(I)
#if KEY_PARASCAL==1
     ENDIF  
#endif
  ENDDO
  !==============================================================      
  ! process holonomic constraints
  CALL HOLONOMA(XNEW,YNEW,ZNEW,X,Y,Z,.TRUE.,.TRUE.,QOK)
  !==============================================================
  !           DO  I=1,NATOM
  do ia=atfrst,atlast
#if KEY_DOMDEC==1
     if (q_domdec) then
        i = atoml(ia)
     else
#endif 
        i = ia
#if KEY_DOMDEC==1
     endif  
#endif
#if KEY_PARASCAL==1
     IF(JPBLOCK(I) == MYNOD) THEN  
#endif
        !           FACT= - AMASS(I)/DELTA2
        !           VX1=(XNEW(I)-XXI(I))*FACT
        !           VY1=(YNEW(I)-YYI(I))*FACT
        !           VZ1=(ZNEW(I)-ZZI(I))*FACT
        !
        IF(IMOVE(I) == 0) THEN
           !           FACT1=DELTAS/AMASS(I) 
           !           VX(I)=VX(I)-FACT1*VX1
           !           VY(I)=VY(I)-FACT1*VY1
           !           VZ(I)=VZ(I)-FACT1*VZ1
           VX(I)=VX(I)+FACT1*(XNEW(I)-XXI(I))
           VY(I)=VY(I)+FACT1*(YNEW(I)-YYI(I))
           VZ(I)=VZ(I)+FACT1*(ZNEW(I)-ZZI(I))
        ENDIF
        !           VX1 = 0.0
        !           VY1 = 0.0
        !           VZ1 = 0.0
#if KEY_PARASCAL==1
     ENDIF  
#endif
  ENDDO
  RETURN
END SUBROUTINE DYNASHK

#if KEY_MTS==1 /*mts_b*/
SUBROUTINE DFININ2
  !-----------------------------------------------------------------------
  !    
 use chm_kinds
 use dimens_fcm
 use number
 use cnst_fcm
 use code
 use image
 use param
 use psf
 use stream
 implicit none
 !
 INTEGER  I,J
 !--------------------------------------------
 J=0    
 I=0   
 !-------------------------------------------
 !--- Store original data ----------------
 IF(NBOND == 0) GOTO 20
 DO J=1,NBOND
    I=I+1
    IBMS4(I)=IB(I)
    JBMS4(I)=JB(I)
    ICBM4(I)=ICB(I)
 ENDDO
 IF(NIMBON == 0) GOTO 20
 DO J=1,NIMBON
    I=I+1
    IBMS4(I)=IB(I)
    JBMS4(I)=JB(I)
    ICBM4(I)=ICB(I)
 ENDDO
20 CONTINUE
 !------- Angle -----------------------------
 I=0
 IF(NTHETA == 0) GOTO 30
 DO J=1,NTHETA
    I=I+1
    ITMS4(I)=IT(I)
    JTMS4(I)=JT(I)
    KTMS4(I)=KT(I)
    ICTM4(I)=ICT(I)
 ENDDO
 IF(NIMANG == 0) GOTO 30
 DO J=1,NIMANG
    I=I+1
    ITMS4(I)=IT(I)
    JTMS4(I)=JT(I)
    KTMS4(I)=KT(I)
    ICTM4(I)=ICT(I)
 ENDDO
30 CONTINUE
 !------- End of restore --------------------------
 !
 I=0
 J=0
 !
 !------- Bond -------------------------------
 !
 IF(IB1 > 0) THEN
    DO J=1,IB1
       I=I+1
       IB(I)=IBMS1(I)
       JB(I)=JBMS1(I)
       ICB(I)=ICBM1(I)
    ENDDO
 ENDIF
 !
 IF(IB2 > 0) THEN
    DO J=1,IB2
       I=I+1
       IB(I)=IBMS2(I)
       JB(I)=JBMS2(I)
       ICB(I)=ICBM2(I)
    ENDDO
 ENDIF
 !
 IF(IB3 > 0) THEN
    DO J=1,IB3
       I=I+1
       IB(I)=IBMS3(I)
       JB(I)=JBMS3(I)
       ICB(I)=ICBM3(I)
    ENDDO
 ENDIF
 !
 !-------- Angle terms -----------------------
 !
 I=0
 IF(ITH1 > 0) THEN
    DO J=1,ITH1
       I=I+1
       IT(I)=ITMS1(I)
       JT(I)=JTMS1(I)
       KT(I)=KTMS1(I)
       ICT(I)=ICTM1(I)
    ENDDO
 ENDIF
 IF(ITH2 > 0) THEN
    DO J=1,ITH2
       I=I+1
       IT(I)=ITMS2(I)
       JT(I)=JTMS2(I)
       KT(I)=KTMS2(I)
       ICT(I)=ICTM2(I)
    ENDDO
 ENDIF
 !
 !---------------------------------------------------------------------
 !-------------------------------------------------------------------- 
 !
 RETURN
END SUBROUTINE DFININ2

SUBROUTINE DFININ3
  !-----------------------------------------------------------------------
  !    
 use chm_kinds
 use chm_types
 use dimens_fcm
 use datstr
 use number
 use cnst_fcm
 use code
 use image
 use param
 use psf
 use bases_fcm
 use stream
 use inbnd
 implicit none
 !
 INTEGER  I,J
 !--------------------------------------------
 J=0    
 I=0   
 !-------------------------------------------
 !--- Restore original data ----------------
 IF(NBOND == 0) GOTO 20
 DO J=1,NBOND
    I=I+1
    IB(I)=IBMS4(I)
    JB(I)=JBMS4(I)
    ICB(I)=ICBM4(I)
 ENDDO
 IF(NIMBON == 0) GOTO 20
 DO J=1,NIMBON
    I=I+1
    IB(I)=IBMS4(I)
    JB(I)=JBMS4(I)
    ICB(I)=ICBM4(I)
 ENDDO
20 CONTINUE
 !------- Angle -----------------------------
 I=0
 IF(NTHETA == 0) GOTO 30
 DO J=1,NTHETA
    I=I+1
    IT(I)=ITMS4(I)
    JT(I)=JTMS4(I)
    KT(I)=KTMS4(I)
    ICT(I)=ICTM4(I)
 ENDDO
 IF(NIMANG == 0) GOTO 30
 DO J=1,NIMANG
    I=I+1
    IT(I)=ITMS4(I)
    JT(I)=JTMS4(I)
    KT(I)=KTMS4(I)
    ICT(I)=ICTM4(I)
 ENDDO
30 CONTINUE
 !------- End of restore --------------------------
 !
 !------ Free the spaces for non-bond list --------
 CALL FREEDT_nbond(BNBNM1)
 CALL FREEDT_nbond(BNBNM2)
 CALL FREEDT_image(BIMTS1)
 CALL FREEDT_image(BIMTS2)
 RETURN
END SUBROUTINE DFININ3

SUBROUTINE SELMTS(TBM2,TBM4,J1,J3,QIMM)
  !
  ! This routine selects internal energy contributions for each energy
  ! call when MTS is in use.
  !
  !   TBM2   - Flag indicating that bonds and angles will be computed
  !   TBM4   - Flag indicating that dihedral and impropers will be computed
  !   J1     - First bond to calculate
  !   J3     - First angle to callculate
  !   QIMM   - Flag to indicate that only images terms should be selected.
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  implicit none
  INTEGER J1,J3
  LOGICAL TBM2,TBM4,QIMM
  !
  TBM2=.FALSE.
  TBM4=.FALSE.
  J1=1
  J3=1

  IF(ENE3) THEN
     IF((TBM(2) < 0).AND.(TBM1(2).LT.0)) TBM2=.TRUE.
     IF((TBM(4) < 0).AND.(TBM1(4).LT.0)) TBM4=.TRUE.
  ELSE
     IF(ENE1.AND.(TBM(2) > 0))  TBM2=.TRUE.
     IF(ENE2.AND.(TBM1(2) > 0)) TBM2=.TRUE.
     IF(ENE1.AND.(TBM(4) > 0))  TBM4=.TRUE.
     IF(ENE2.AND.(TBM1(4) > 0)) TBM4=.TRUE.
  ENDIF
  !
  IF(.NOT.QIMM) THEN
     NBONM=0
     NTHUBM=0
     NTHETM=0
     !
     IF(ENE3) THEN
        NBONM=NBONS
        J1=IB1+1
     ENDIF
     !
     IF(ENE1) THEN
        NBONM=NBONF
        J1=1
     ENDIF
     !
     IF(ENE2) THEN
        NBONM=NBONQ
        J1=IB1+IB2+1
     ENDIF
     !
     IF(TBM2) THEN
        IF(ENE3) THEN
           NTHETM=NTHETS
           NTHUBM=NTHETS
           J3=ITH1+1
        ELSE
           NTHETM=NTHETF
           NTHUBM=NTHETF
           J3=1
        ENDIF
     ENDIF
     !
  ELSE
     NBOMM=0
     NTUBMM=0
     NTHEMM=0
     NPHMM=0
     NIMMHM=0
#if KEY_CMAP==1
     NIMCTM=0
#endif 
     !
     IF(ENE3) THEN
        NBOMM=NBOMS
        J1=IB1+NBONS
        IF(TBM2) THEN
           NTHEMM=NTHEMS
           J3=ITH1+NTHETS
        ENDIF
        IF(TBM4) THEN
           NIMMHM=NIMMHS
#if KEY_CMAP==1
           NIMCTM=NIMCTS
#endif 
           NPHMM=NPHMS
        ENDIF
     ENDIF
     ! 
     IF(ENE1) THEN 
        NBOMM=NBOMF
        J1=NBONF
        IF(TBM2) THEN
           NTHEMM=NTHEMF
           J3=NTHETF
        ENDIF
        IF(TBM4) THEN 
           NIMMHM=NIMMHF
#if KEY_CMAP==1
           NIMCTM=NIMCTF
#endif 
           NPHMM=NPHMF
        ENDIF
     ENDIF
     !
     IF(ENE2) THEN
        NBOMM=NBOMQ
        J1=IB1+IB2+NBONQ
        IF(TBM2) THEN
           NTHEMM=NTHEMF
           J3=NTHETF
        ENDIF
        IF(TBM4) THEN
           NIMMHM=NIMMHF
#if KEY_CMAP==1
           NIMCTM=NIMCTF 
#endif
           NPHMM=NPHMF
        ENDIF
     ENDIF
  ENDIF
  !
  return
end SUBROUTINE SELMTS
#endif /* (mts_b)*/

  ! middle-scheme SIN(R) related
#if KEY_MIDSINR == 1 /*mid-sinr*/
  subroutine midsinr_init_holo(natom,atfrst,atlast,l_val,qmass_1,KBT,deltasL, &
                               imove,dx,dy,dz,amass,vx,vy,vz,  &
                               v_1ij)
     !
     ! Equilibrate thermostat variable v_1ij around the current velocity
     !
     ! ioption = 1 : b_val calc. only using dx/dy/dz
     !         = 2 :                  using both dx/dy/dz & xmi/ymi/zmi
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number

     implicit none
     integer :: natom,l_val,atfrst,atlast
     integer :: imove(*)
     real(chm_real) :: dx(*),dy(*),dz(*),amass(*),vx(*),vy(*),vz(*)
     real(chm_real) :: v_1ij(3,L_val,natom)
     real(chm_real) :: qmass_1,KBT,deltasL

     ! local variables
     integer        :: i,j,k,k_val
     real(chm_real) :: ll_val,ll_val2,ll_val3,b_tmp,c_val,b_val(3),e_pbdt(3),e_mbdt(3), &
                       cosh_val(3),sinh_val(3)
     real(chm_real) :: vxyz_i(3),s_val(3)

     ! local varaibles set
     ll_val = float(L_val)
     ll_val2= qmass_1*float(L_val)/float(L_val+1)
     ll_val3= float(L_val)*KBT

     ! Equilibrate v_1ij around the current velocity
     do k=1,5 ! equilibrate by running 5 times.
        do i=atfrst,atlast
           if(imove(i) == 0) then
              b_tmp    = DELTASL/SQRT(ll_val*amass(i)*KBT)   ! KBT=TBATH*KBOLTZ
              b_val(1) =-dx(i)*b_tmp
              b_val(2) =-dy(i)*b_tmp
              b_val(3) =-dz(i)*b_tmp

              ! cosh (b dt) = 1/2*(e^(b dt) + e^(-b dt)) &
              ! sinh (b dt) = 1/2*(e^(b dt) - e^(-b dt))
              c_val         = sqrt(ll_val*KBT/amass(i))
              e_pbdt(1:3)   = EXP( b_val(1:3))
              e_mbdt(1:3)   = EXP(-b_val(1:3))
              cosh_val(1:3) = half*(e_pbdt(1:3)+e_mbdt(1:3))
              sinh_val(1:3) = half*(e_pbdt(1:3)-e_mbdt(1:3))
              vxyz_i(1)= vx(i)*cosh_val(1) + c_val*sinh_val(1)
              vxyz_i(2)= vy(i)*cosh_val(2) + c_val*sinh_val(2)
              vxyz_i(3)= vz(i)*cosh_val(3) + c_val*sinh_val(3)

              s_val(1:3) = zero
              do j=1,L_val
                 s_val(1:3) = s_val(1:3)+v_1ij(1:3,j,i)*v_1ij(1:3,j,i)
              end do
              s_val(1:3)=amass(i)*vxyz_i(1:3)*vxyz_i(1:3)+ll_val2*s_val(1:3)
              s_val(1:3)=sqrt( ll_val3/s_val(1:3) )

              !! update velocity
              !vx(i) = s_val(1)*vxyz_i(1)
              !vy(i) = s_val(2)*vxyz_i(2)
              !vz(i) = s_val(3)*vxyz_i(3)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i) = s_val(1:3)*v_1ij(1:3,j,i)
              end do
           end if
        end do
     end do
     return
  end subroutine midsinr_init_holo

  subroutine midsinr_propagate_dt(natom,atfrst,atlast,l_val,qmass_1,KBT,deltasL, &
                                  imove,dx,dy,dz,amass,vx,vy,vz,  &
                                  v_1ij,ioption)
     !
     ! middle-scheme SIN(R) thermostat, propogate by dt/2
     !
     ! deltas can be deltas        (dt/2)
     !               SS2X          (DELTA*NMTS1/2) for MTS; middle-step case
     !               SA2X          (DELTA*NMTS0/2)
     !
     ! dx/dy/dz can also be 
     !               dx/dy/dz      (call with DELTAS or SA2X); init & outer-step propagation
     !               XMI,YMI,ZMI   (call with DELTAS); inner-step  propagation
     !               XMM,YMM,ZMM   (call with SS2X)  ; middle-step propagation
     !
     ! ioption = 1: normal (all moving particles)
     !           2: do the cycle over (imts(i) >=0) case
     !
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number
     !!use tbmts   ! for mts case imts array

     implicit none
     integer :: natom,l_val,atfrst,atlast,ioption
     integer :: imove(*)
     real(chm_real) :: dx(*),dy(*),dz(*),amass(*),vx(*),vy(*),vz(*)
     real(chm_real) :: v_1ij(3,L_val,natom)
     real(chm_real) :: qmass_1,KBT,deltasL

     ! local variables
     integer        :: i,j,k,k_val
     real(chm_real) :: ll_val,ll_val2,ll_val3,b_tmp,c_val,b_val(3),e_pbdt(3),e_mbdt(3), &
                       cosh_val(3),sinh_val(3)
     real(chm_real) :: vxyz_i(3),s_val(3)

     ! local varaibles set
     ll_val = float(L_val)
     ll_val2= qmass_1*float(L_val)/float(L_val+1)
     ll_val3= float(L_val)*KBT

     ! propogate by dt/2
#if KEY_MTS==1
     if(ioption == 1) then
#endif
        do i=atfrst,atlast
           if(imove(i) == 0) then
              b_tmp    = DELTASL/SQRT(ll_val*amass(i)*KBT)   ! KBT=TBATH*KBOLTZ
              b_val(1) =-dx(i)*b_tmp
              b_val(2) =-dy(i)*b_tmp
              b_val(3) =-dz(i)*b_tmp

              ! cosh (b dt) = 1/2*(e^(b dt) + e^(-b dt)) &
              ! sinh (b dt) = 1/2*(e^(b dt) - e^(-b dt))
              c_val         = sqrt(ll_val*KBT/amass(i))
              e_pbdt(1:3)   = EXP( b_val(1:3))
              e_mbdt(1:3)   = EXP(-b_val(1:3))
              cosh_val(1:3) = half*(e_pbdt(1:3)+e_mbdt(1:3))
              sinh_val(1:3) = half*(e_pbdt(1:3)-e_mbdt(1:3))
              vxyz_i(1)= vx(i)*cosh_val(1) + c_val*sinh_val(1)
              vxyz_i(2)= vy(i)*cosh_val(2) + c_val*sinh_val(2)
              vxyz_i(3)= vz(i)*cosh_val(3) + c_val*sinh_val(3)

              s_val(1:3) = zero
              do j=1,L_val
                 s_val(1:3) = s_val(1:3)+v_1ij(1:3,j,i)*v_1ij(1:3,j,i)
              end do
              s_val(1:3)=amass(i)*vxyz_i(1:3)*vxyz_i(1:3)+ll_val2*s_val(1:3)
              s_val(1:3)=sqrt( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vxyz_i(1)
              vy(i) = s_val(2)*vxyz_i(2)
              vz(i) = s_val(3)*vxyz_i(3)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i) = s_val(1:3)*v_1ij(1:3,j,i)
              end do
           end if
        end do
#if KEY_MTS==1
     else if(ioption == 2) then
        do i=atfrst,atlast
           if(imts(i) >= 0 .and. imove(i) == 0) then
              b_tmp    = DELTASL/SQRT(ll_val*amass(i)*KBT)   ! KBT=TBATH*KBOLTZ
              b_val(1) =-dx(i)*b_tmp
              b_val(2) =-dy(i)*b_tmp
              b_val(3) =-dz(i)*b_tmp

              ! cosh (b dt) = 1/2*(e^(b dt) + e^(-b dt)) &
              ! sinh (b dt) = 1/2*(e^(b dt) - e^(-b dt))
              c_val         = sqrt(ll_val*KBT/amass(i))
              e_pbdt(1:3)   = EXP( b_val(1:3))
              e_mbdt(1:3)   = EXP(-b_val(1:3))
              cosh_val(1:3) = half*(e_pbdt(1:3)+e_mbdt(1:3))
              sinh_val(1:3) = half*(e_pbdt(1:3)-e_mbdt(1:3))
              vxyz_i(1)= vx(i)*cosh_val(1) + c_val*sinh_val(1)
              vxyz_i(2)= vy(i)*cosh_val(2) + c_val*sinh_val(2)
              vxyz_i(3)= vz(i)*cosh_val(3) + c_val*sinh_val(3)

              s_val(1:3) = zero
              do j=1,L_val
                 s_val(1:3) = s_val(1:3)+v_1ij(1:3,j,i)*v_1ij(1:3,j,i)
              end do
              s_val(1:3)=amass(i)*vxyz_i(1:3)*vxyz_i(1:3)+ll_val2*s_val(1:3)
              s_val(1:3)=sqrt( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vxyz_i(1)
              vy(i) = s_val(2)*vxyz_i(2)
              vz(i) = s_val(3)*vxyz_i(3)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i) = s_val(1:3)*v_1ij(1:3,j,i)
              end do
           end if
        end do
     end if
#endif
     return
  end subroutine midsinr_propagate_dt

  subroutine midsinr_thermostat_init1(natom,l_val,qmass_1,qmass_2,gamma_val,KBT,deltaL, &
                                      v_1ij,v_2ij,ig)
     ! init of middle sin(r) thermostat variables.
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number
     use consta, only: TIMFAC,PI
     use parallel
     use clcg_mod,only : random

     implicit none
     integer :: natom,l_val,atfrst,atlast,ig
     real(chm_real) :: v_1ij(3,L_val,natom),v_2ij(3,L_val,natom)
     real(chm_real) :: qmass_1,qmass_2,gamma_val,KBT,deltaL

     ! local variables
     integer :: itm,j
     real(chm_real) :: rand_val(3),z_val,r1_tmp,r2_tmp,aa_val,bb_val,cc_val,dd_val
     real(chm_real),parameter :: PIS=PI

     !
     ! do itm = 1,natom
     !   v_1ij(1:3,1:L_val,itm) = zero  ! zeroing values...
     !   v_2ij(1:3,1:L_val,itm) = zero
     !end do
     z_val  = EXP(-TIMFAC*gamma_val*DELTAL)
     r1_tmp = sqrt(KBT*(one-z_val**2)/qmass_1)  ! sqrt(kT/q1 * (1-z^2)
     r2_tmp = sqrt(KBT*(one-z_val**2)/qmass_2)
#if KEY_PARALLEL==1
     if(mynod==0) then
#endif
        do itm=1,natom
           do j=1,L_val
              aa_val = sqrt(mintwo*log(random(ig)))
              bb_val = two*PIS*random(ig)
              cc_val = sqrt(mintwo*log(random(ig)))
              dd_val = two*PIS*random(ig)
              rand_val(1) = aa_val*COS(bb_val)
              rand_val(2) = aa_val*SIN(bb_val)
              rand_val(3) = cc_val*COS(dd_val)
              v_1ij(1:3,j,itm) = r1_tmp*rand_val(1:3)

              aa_val = sqrt(mintwo*log(random(ig)))
              bb_val = two*PIS*random(ig)
              rand_val(1) = cc_val*SIN(dd_val)
              rand_val(2) = aa_val*COS(bb_val)
              rand_val(3) = aa_val*SIN(bb_val)
              v_2ij(1:3,j,itm) = r2_tmp*rand_val(1:3)
           end do
        end do
#if KEY_PARALLEL==1
     end if
     if(numnod>0) then
        call PSND8m(v_1ij,3*L_val*natom)
        call PSND8m(v_2ij,3*L_val*natom)
     end if
#endif
     return
  end subroutine midsinr_thermostat_init1

  subroutine midsinr_thermostat_update(natom,atfrst,atlast,l_val,qmass_1,qmass_2,gamma_val, &
                                       KBT,deltasL,deltaL,imove,vx,vy,vz,amass,                 &
                                       v_1ij,v_2ij,ig,ioption)
     !
     ! update iso-Kinetic thermostat variables (by propagating it by dt)
     !
     ! deltas can be deltas        (dt/2)          ; for inner loop update (where skipping not moving particles
     !                                                                      during the inner loop)
     !               SA2X          (DELTA*NMTS0/2) ; for outer loop update (where applied to all particles)
     !
     ! deltasL = deltaL/2; i.e., deltaL = dt (or DT) ; deltasL = dt/2 (or DT/2)
     !
     ! ioption==1: it is applied to all imove(i)==0 particles
     !
     !          2: it is applied to imts(i) >= 0 and imove(i)==0 particles
     !             for imts(i)<0 particles, which do not move during the inner time step, 
     !             updated in the outer loop (ioption==3)
     ! 
     !          3: it is applied to imts(i) <  0 and imove(i)==0 particles
     !             for their updates at the outer loop.
     !
     ! How is L_iso-K updated?
     ! 1) L_iso-K by dt/2  (for outer step, Dt/2)
     ! 2) L_DOU by dt      (for outer step, Dt)
     ! 3) L_iso-K by dt/2  (for outer step, Dt/2)
     !
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number
     !!use tbmts   ! for mts case imts array
     use consta, only: TIMFAC,PI
     use clcg_mod,only : random

     implicit none
     integer :: natom,l_val,atfrst,atlast,ioption,ig
     integer :: imove(*)
     real(chm_real) :: vx(*),vy(*),vz(*),amass(*)
     real(chm_real) :: v_1ij(3,L_val,natom),v_2ij(3,L_val,natom)
     real(chm_real) :: qmass_1,qmass_2,gamma_val,KBT,deltasL,deltaL

     ! local variables
     integer        :: i,j,k,k_val
     real(chm_real) :: ll_val,ll_val2,ll_val3
     real(chm_real) :: s_val(3),z_val,r_qmg,r_tmp,aa_val,bb_val,cc_val,dd_val,rval
     real(chm_real) :: vs_1ij(3,L_val),rand_val(3),u_val(3)
     real(chm_real),parameter :: PIS=PI

     ! local varaibles set
     ll_val = float(L_val)
     ll_val2= qmass_1*float(L_val)/float(L_val+1)
     ll_val3= float(L_val)*KBT
     rval   = TIMFAC*DELTASL

     z_val  = EXP(-TIMFAC*gamma_val*DELTAL)
     r_qmg  = one/(gamma_val*qmass_2)
     r_tmp  = sqrt(KBT*(one-z_val*z_val)/qmass_2) ! sqrt(kT/q2 * (1-z^2)

     ! 1. apply L_iso-K by dt/2
#if KEY_MTS==1
     if(ioption == 1) then
#endif
        do i=atfrst,atlast
           if(imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                   vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                   s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
#if KEY_MTS==1
     else if(ioption == 2) then
        do i=atfrst,atlast
           if(imts(i) >= 0 .and. imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                   vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                   s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
     else if(ioption == 3) then
        do i=atfrst,atlast
           if(imts(i) <  0 .and. imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                   vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                   s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
     end if
#endif

     ! 2. apply L_DOU by dt
     k_val  = 0
#if KEY_MTS==1
     if(ioption == 1) then
#endif
        do i=atfrst,atlast
           if(imove(i) == 0) then
              do j=1,L_val
                 ! based on the subroutine DLNGV ... ig should be passed in.
                 if(k_val == 0) then
                    k_val = 1
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    cc_val = sqrt(mintwo*log(random(ig)))
                    dd_val = two*PIS*random(ig)
                    rand_val(1) = aa_val*COS(bb_val)
                    rand_val(2) = aa_val*SIN(bb_val)
                    rand_val(3) = cc_val*COS(dd_val)
                 else
                    k_val = 0
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    rand_val(1) = cc_val*SIN(dd_val)
                    rand_val(2) = aa_val*COS(bb_val)
                    rand_val(3) = aa_val*SIN(bb_val)
                 end if

                 u_val(1:3)    =r_qmg*(qmass_1*v_1ij(1:3,j,i)*v_1ij(1:3,j,i)-KBT)
                 v_2ij(1:3,j,i)=z_val*v_2ij(1:3,j,i) + u_val(1:3)*(one-z_val) + r_tmp*rand_val(1:3)
              end do
           end if
        end do
#if KEY_MTS==1
     else if(ioption == 2) then
        do i=atfrst,atlast
           if(imts(i) >= 0 .and. imove(i) == 0) then
              do j=1,L_val
                 ! based on the subroutine DLNGV ... ig should be passed in.
                 if(k_val == 0) then
                    k_val = 1
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    cc_val = sqrt(mintwo*log(random(ig)))
                    dd_val = two*PIS*random(ig)
                    rand_val(1) = aa_val*COS(bb_val)
                    rand_val(2) = aa_val*SIN(bb_val)
                    rand_val(3) = cc_val*COS(dd_val)
                 else
                    k_val = 0
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    rand_val(1) = cc_val*SIN(dd_val)
                    rand_val(2) = aa_val*COS(bb_val)
                    rand_val(3) = aa_val*SIN(bb_val)
                 end if

                 u_val(1:3)    =r_qmg*(qmass_1*v_1ij(1:3,j,i)*v_1ij(1:3,j,i)-KBT)
                 v_2ij(1:3,j,i)=z_val*v_2ij(1:3,j,i) + u_val(1:3)*(one-z_val) + r_tmp*rand_val(1:3)
              end do
           end if
        end do
     else if(ioption == 3) then
        do i=atfrst,atlast
           if(imts(i) <  0 .and. imove(i) == 0) then
              do j=1,L_val
                 ! based on the subroutine DLNGV ... ig should be passed in.
                 if(k_val == 0) then
                    k_val = 1
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    cc_val = sqrt(mintwo*log(random(ig)))
                    dd_val = two*PIS*random(ig)
                    rand_val(1) = aa_val*COS(bb_val)
                    rand_val(2) = aa_val*SIN(bb_val)
                    rand_val(3) = cc_val*COS(dd_val)
                 else
                    k_val = 0
                    aa_val = sqrt(mintwo*log(random(ig)))
                    bb_val = two*PIS*random(ig)
                    rand_val(1) = cc_val*SIN(dd_val)
                    rand_val(2) = aa_val*COS(bb_val)
                    rand_val(3) = aa_val*SIN(bb_val)
                 end if

                 u_val(1:3)    =r_qmg*(qmass_1*v_1ij(1:3,j,i)*v_1ij(1:3,j,i)-KBT)
                 v_2ij(1:3,j,i)=z_val*v_2ij(1:3,j,i) + u_val(1:3)*(one-z_val) + r_tmp*rand_val(1:3)
              end do
           end if
        end do
     end if
#endif

     ! 3. apply L_iso-K by dt/2
#if KEY_MTS==1
     if(ioption == 1) then
#endif
        do i=atfrst,atlast
           if(imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                    vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                    s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
#if KEY_MTS==1
     else if(ioption == 2) then
        do i=atfrst,atlast
           if(imts(i) >= 0 .and. imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                    vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                    s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
     else if(ioption == 3) then
        do i=atfrst,atlast
           if(imts(i) <  0 .and. imove(i) == 0) then
              s_val(1:3) = zero
              do j=1,L_val
                 do k=1,3
                    vs_1ij(k,j)=v_1ij(k,j,i)*EXP(-v_2ij(k,j,i)*rval)
                    s_val(k)   =s_val(k)+vs_1ij(k,j)*vs_1ij(k,j)
                 end do
              end do
              s_val(1) = amass(i)*vx(i)*vx(i) + ll_val2*s_val(1)
              s_val(2) = amass(i)*vy(i)*vy(i) + ll_val2*s_val(2)
              s_val(3) = amass(i)*vz(i)*vz(i) + ll_val2*s_val(3)
              s_val(1:3) = sqrt ( ll_val3/s_val(1:3) )

              ! update velocity
              vx(i) = s_val(1)*vx(i)
              vy(i) = s_val(2)*vy(i)
              vz(i) = s_val(3)*vz(i)

              ! update v_1,i,j
              do j=1,L_val
                 v_1ij(1:3,j,i)=s_val(1:3)*vs_1ij(1:3,j)
              end do
           end if
        end do
     end if
#endif 
     return
  end subroutine midsinr_thermostat_update

  subroutine midsinr_temperature_check(natom,atfrst,atlast,l_val,qmass_1,qmass_2,ndegf, &
                                       imove,amass,vx,vy,vz,                            &
                                       v_1ij,v_2ij,istep)
     !
     ! calculate temperature and check
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number
     !!use tbmts   ! for mts case imts array
     use consta, only: KBOLTZ
     use parallel

     implicit none
     integer :: natom,l_val,atfrst,atlast,istep,ndegf
     integer :: imove(*)
     real(chm_real) :: amass(*),vx(*),vy(*),vz(*)
     real(chm_real) :: v_1ij(3,L_val,natom),v_2ij(3,L_val,natom)
     real(chm_real) :: qmass_1,qmass_2

     ! local variables
     integer :: i,j
     real(chm_real) :: ll_val2,tempi_local(3),rval,ravl

     ! reference taget temperature = (L+1)/L*system_temp
     ! since <1/2 m v_i^2> = L/L+1 kT/2
     !ll_val = float(L_val+1)/float(L_val)
     !EPROP(TEMPS)=ll_val*TEMPI/(KBOLTZ*NDEGF)

     ! LkT = sum m v_i^2 + q1*L/(L+1) sum (sum v_1ij^2)
     tempi_local(1:3) = zero
     ll_val2          = qmass_1 * float(L_val)/float(L_val + 1)
     do i=atfrst,atlast
        if(imove(i)==0) then
           tempi_local(1) = tempi_local(1) + amass(i)*(vx(i)**2 + vy(i)**2 + vz(i)**2)

           rval = zero
           ravl = zero
           do j=1,L_val
              rval=rval+dot_product(v_1ij(1:3,j,i),v_1ij(1:3,j,i))
              ravl=ravl+dot_product(v_2ij(1:3,j,i),v_2ij(1:3,j,i))
           end do
           tempi_local(2) = tempi_local(2) + rval
           tempi_local(3) = tempi_local(3) + ravl
        end if
     end do
     tempi_local(2) = tempi_local(1) + ll_val2*tempi_local(2)
     tempi_local(3) = qmass_2*tempi_local(3)
#if KEY_PARALLEL==1
     call gcomb(tempi_local,3)
     if(mynod==0) then
#endif
        tempi_local(1) = tempi_local(1)/(KBOLTZ*NDEGF)               ! based on vx,vy,vz
        tempi_local(2) = tempi_local(2)/(float(L_val)*KBOLTZ*NDEGF)  ! for isokinetic constraints (v_1)
        tempi_local(3) = tempi_local(3)/(float(L_val)*KBOLTZ*NDEGF)  ! for v_2
        write(6,'(A,I7,3F12.5)') 'TEMP:',istep,tempi_local(1),tempi_local(2),tempi_local(3)
#if KEY_PARALLEL==1
     end if   ! (mynod==0)
#endif
     return
  end subroutine midsinr_temperature_check
#endif    /*mid-sinr*/

end module tbmts

