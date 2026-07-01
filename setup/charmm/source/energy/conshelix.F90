module conshelix_m
  use chm_kinds
  implicit none

#if KEY_CONSHELIX==1 /*conshelix*/
!***********************************************************************
!   Set up helix-helix interaction
!***********************************************************************
!   Authors: Jinhyuk Lee and Wonpil Im (2006)
!            jinhyuk@ku.edu  wonpil@ku.edu
!   @ Center for Bioinformatics in Univ. of Kansas
!
!   Reference: 1. Jinhyuk Lee and Wonpil Im, J. Comput. Chem., 28, 669(2007)
!              2. Jinhyuk Lee and Wonpil Im, Chem. Phys. Lett., 441, 132(2007)
!              3. C.Chothia, J.Mol.Biol. 145, 215(1981)
!              4. etc
!   Bug, Error, Problems, Suggestion etc reports welcome
!
! [SYNTAX: CONS HELIx commands]
! 
! CONS { HELIx } DISTance <real> FORCe <real> -
!      { BHG   } [SELE 1st-atom-selection END] [SELE 2nd-atom-selection END] -
!                UDISt <int> STEP <int>
! 
! CONS { HELIx } { [CROSs] [HINGe] } ANGLe <real> FORCe <real> -
!      { BHG   } [SELE  1st-atom-selection END] [SELE 2nd-atom-selection END] -
!                UANG <int> STEP <int>
! 
! CONS { HELIx } { TilX } ANGLe <real> FORCe <real> -
!      { BHG   } { TilY } [SELE atom-selection END] -
!                { TilZ } UANG <int> STEP <int>
! 
! CONS { HELIx } ROTH { RotX } ANGLe <real> FORCe <real> REFA <int> -
!      { BHG   }      { RotY } [SELE atom-selection END] -
!                     { RotZ } UANG <int> STEP <int>
! 
!    Single selection - Tilt angle constraint or Rotation angle constraint
!    Double selection - Cross angle constraint or Hinge angle constraint
! 
!    CROSs : default option - Crossing angle constraint
!    HINGe : Hinge angle constraint
! 
!    HELIx : for a-helix
!    BHG   : for beta-hairpin
! 
! CONS HELIx RESEt     ! reset CONS HELIx
!
!***********************************************************************
! Set up COM-COM constraints
!***********************************************************************
! Author: Jinhyuk Lee and Wonpil IM (2006)
!
! SYNTAX: CONS COM DISTance real FORCe real -
!              SELE { atom selection } END SELE { atom selection } END -
!              UDISt int STEP int
! 
! Reset of the constraints after the calculation
!         CONS COM RESEt ( or CONS HELIx RESEt )
!
!***********************************************************************
contains

SUBROUTINE CONSHELIX_INIALL()
  use conshelix_fcm
  ! rem redundant variables
  !      QSAVRG=.FALSE.
  QCONSH=.FALSE.
  CHNUM=0
  OCHNUM=1
END SUBROUTINE CONSHELIX_INIALL

SUBROUTINE CONSHELIX(LCOM)
  use conshelix_fcm
  use stream
  use string
  use dimens_fcm
  use psf
  use select
  use number
  use coord
  use comand
  use coordc
  use memory

  INTEGER I
  INTEGER NNSEL(2),K
  LOGICAL LRESET,LCOM,ERR
  LOGICAL OFIRS,OLAST,TFIRS,TLAST
  !-----------------------------------------------------------------------
  ! Reset option to terminate constraints
  LRESET=(INDXA(COMLYN, COMLEN, 'RESE') > 0)
  IF(QCONSH .AND. LRESET) THEN
     QCONSH = .FALSE.
     WRITE(OUTU,*) 'TOTAL ', CHNUM,' CONSTRAINTS RESETED.'
     !
     ! Free HEAP Size
     DO I=1,CHNUM
        call chmdealloc('conshelix.src','conshelix','ASLCT',nsel(1,i),intgp=ASLCT(i)%a) 
        call chmdealloc('conshelix.src','conshelix','BSLCT',nsel(2,i),intgp=BSLCT(i)%a) 
        NSEL(1,I)=0
        NSEL(2,I)=0
     ENDDO
     !    
     call chmdealloc('conshelix.src','conshelix', 'SSLCT',mxconsh,NATOM,intg=SSLCT)
     call chmdealloc('conshelix.src','conshelix','JJSLCT',mxconsh,NATOM,intg=JJSLCT)
     !
     CHNUM=0
  ELSE
     IF(.NOT.QCONSH) THEN
        call chmalloc('conshelix.src','conshelix','SSLCT',mxconsh,NATOM,intg=SSLCT)
        call chmalloc('conshelix.src','conshelix','JJSLCT',mxconsh,NATOM,intg=JJSLCT)
        CHNUM=0
     ENDIF

     CHNUM=CHNUM+1

     IF(CHNUM > MXCONSH) THEN
        CALL WRNDIE(-3,'HLXSPC','MAXIMUM NUMBER OF CONSTRAINTS IS EXCEEDED.')
        STOP
     ENDIF
     ! for COM - COM constraints 
     LCOMM(CHNUM)=LCOM
     ! initialize
     LDNA(CHNUM)=.FALSE.
     LBHG(CHNUM)=.FALSE.
     LBHP(CHNUM)=.FALSE.
     !
     ! DNA mode
     !
     LDNA(CHNUM)=(IndxA(Comlyn, Comlen,'DNA') > 0)
     ! B-hairpin mode (general moment of inertia)
     LBHG(CHNUM)=(IndxA(Comlyn, Comlen,'BHG') > 0)
     ! B-hairpin mode (general moment of inertia) + sele (helix type-dna)
     LBHP(CHNUM)=(IndxA(Comlyn, Comlen,'BHP') > 0)
     !
     HDIST(CHNUM)=GTRMF(COMLYN,COMLEN,'DIST',-1.d0)
     !
     IF(HDIST(CHNUM) == -1.D0) THEN
        !
        ! CONS HELIX ANGLe
        ! cross angle and tilt angle
        ANGL(CHNUM)=GTRMF(COMLYN,COMLEN,'ANGL',-9999.d0)
        IF(ANGL(CHNUM) == -9999.D0)THEN
           CALL WRNDIE(-3,'HLXSPC','CONSTRAINT TYPE IS NOT DEFINED.')
           STOP
        ENDIF
        AFOR(CHNUM)   =GTRMF(COMLYN,COMLEN,'FORC',ZERO)
        CHUNITA(CHNUM)=GTRMI(COMLYN,COMLEN,'UANG',6)
        CHSTEPA(CHNUM)=GTRMI(COMLYN,COMLEN,'STEP',0)
        !
        ! Cross and Hinge angle
        LCROS(CHNUM)=(IndxA(Comlyn, Comlen,'CROS') > 0)
        LHING(CHNUM)=(IndxA(Comlyn, Comlen,'HING') > 0)
        !
        ! Tilt angle: Tilt direction, tilz (default)
        LTAX(CHNUM)=(IndxA(Comlyn, Comlen,'TILX') > 0)
        LTAY(CHNUM)=(IndxA(Comlyn, Comlen,'TILY') > 0)
        LTAZ(CHNUM)=(IndxA(Comlyn, Comlen,'TILZ') > 0)
        !
        ! Rotation angle
        LROTH(CHNUM)=(IndxA(Comlyn, Comlen,'ROTH') > 0)
        ! Reference atom
        REFA(CHNUM) = GTRMI(COMLYN,COMLEN,'REFA',-1)
        ! Reference Axis: rotz (default)
        LRAX(CHNUM)=(IndxA(Comlyn, Comlen,'ROTX') > 0)
        LRAY(CHNUM)=(IndxA(Comlyn, Comlen,'ROTY') > 0)
        LRAZ(CHNUM)=(IndxA(Comlyn, Comlen,'ROTZ') > 0)
        !
        ! use harmonic potential for rotation angle
        ! default - periodic function
        !
        LHARM(CHNUM)=(IndxA(Comlyn, Comlen,'HARM')>0)
        !
        IF(.NOT.LCROS(CHNUM).AND..NOT.LHING(CHNUM)) LCROS(CHNUM)=.TRUE.
        IF(LCROS(CHNUM).AND.LHING(CHNUM)) THEN
           CALL WRNDIE(-3,'HLXSPC','Do not use two opitions together, CROSs and HINGe')
           STOP
        ENDIF
        IF(LROTH(CHNUM).AND.REFA(CHNUM).EQ.-1) THEN
           CALL WRNDIE(-3,'HLXSPC','REFERENCE ATOM (REFA) FOR ROTATION ANGLE IS REQUIRED.')
           STOP
        ENDIF
        IF(PRNLEV >= 6) WRITE(6,*) 'Constraint: ','Cross:',LCROS(CHNUM),'Hing:', &
             LHING(CHNUM),'Rota:',LROTH(CHNUM)
        !
        OFIRS=(IndxA(Comlyn, Comlen,'1FIR')>0)
        OLAST=(IndxA(Comlyn, Comlen,'1LAS')>0)
        TFIRS=(IndxA(Comlyn, Comlen,'2FIR')>0)
        TLAST=(IndxA(Comlyn, Comlen,'2LAS')>0)
        LHMOD(CHNUM)=.TRUE.
        IF((OLAST.AND.TFIRS).OR.(OFIRS.AND.TLAST)) THEN
           LHMOD(CHNUM)=.FALSE.
        ENDIF
        IF(PRNLEV >= 6) WRITE(6,*) 'LHMOD',lhmod(chnum)
        ! true
        IF(LHMOD(CHNUM)) THEN
           IF(PRNLEV >= 5) WRITE(6,*) 'CHARMM selection followed (Default)'
        ELSE
           ! false
           IF(PRNLEV >= 5) WRITE(6,*) 'Hinge mode selection followed'
        ENDIF
        !
        XANG(CHNUM) = .TRUE.
     ELSE
        !
        ! CONS HELIx DISTance & CONS COM DIST
        !
        DFOR(CHNUM)  =GTRMF(COMLYN,COMLEN,'FORC',ZERO)
        CHUNIT(CHNUM)=GTRMI(COMLYN,COMLEN,'UDIS',6)
        CHSTEP(CHNUM)=GTRMI(COMLYN,COMLEN,'STEP',0)
     ENDIF
     !  Select atoms 
     !-----------------------------------------------------------------------
     ! Allocate Heap Arrays for CONSHELIX
     !-----------------------------------------------------------------------
     CALL SELCTD(COMLYN,COMLEN,SSLCT(CHNUM,:),JJSLCT(CHNUM,:),X,Y,Z,WMAIN,.TRUE.,ERR)
     !
     ! count number of selected atoms, subroutine
     NSEL(1,CHNUM)=NSELCT(NATOM,SSLCT(CHNUM,:))
     NSEL(2,CHNUM)=NSELCT(NATOM,JJSLCT(CHNUM,:))
     !
     CALL chmalloc('conshelix.src','conshelix','ASLCT(chnum)',nsel(1,chnum),intgp=ASLCT(chnum)%A) 
     CALL chmalloc('conshelix.src','conshelix','BSLCT(chnum)',nsel(2,chnum),intgp=BSLCT(chnum)%A) 
     !
     ! define the atom number of selected atoms
     IF(LDNA(CHNUM).OR.LBHP(CHNUM)) THEN
        CALL DANSAD(SSLCT(CHNUM,:),ASLCT(CHNUM)%A,NATOM)
        CALL DANSAD(JJSLCT(CHNUM,:),BSLCT(CHNUM)%A,NATOM)
     ELSEIF (LBHG(CHNUM)) THEN
        CALL DANSAHP(SSLCT(CHNUM,:),ASLCT(CHNUM)%A,NATOM)
        CALL DANSAHP(JJSLCT(CHNUM,:),BSLCT(CHNUM)%A,NATOM)
     ELSE
        CALL DANSA(SSLCT(CHNUM,:),ASLCT(CHNUM)%A,NATOM)
        CALL DANSA(JJSLCT(CHNUM,:),BSLCT(CHNUM)%A,NATOM)
     ENDIF
     !
     IF(LDNA(CHNUM).OR.LBHP(CHNUM)) THEN
        NSEL(1,CHNUM)=NSEL(1,CHNUM)/2
        NSEL(2,CHNUM)=NSEL(2,CHNUM)/2
     ENDIF
     ! check whether two selected helixes are same or not
     !----------------------------------------------------------------------
     NNSEL(1)=NSEL(1,CHNUM)
     NNSEL(2)=NSEL(2,CHNUM)
     !
     CALL CKTHEL(ASLCT(CHNUM)%a,BSLCT(CHNUM)%a,NNSEL,LOHEL(CHNUM))
     !
     IF(PRNLEV >= 6) THEN
        IF(LOHEL(CHNUM)) THEN
           WRITE(OUTU,*) 'Two selected domains are identical'
           WRITE(OUTU,*) 'In the case of doing constraints about angle: ',&
                'Do constraints of Tilt angle'
        ENDIF
     ENDIF
     ! debug
     WRITE(OUTU,*) 'Constraints : ',CHNUM
     WRITE(OUTU,*) 'Number of selected atoms'
     IF(LOHEL(CHNUM)) THEN
        WRITE(OUTU,*) '1st Helix :',NSEL(1,CHNUM),'2nd Helix :          None'
     ELSE
        WRITE(OUTU,*) '1st Helix :',NSEL(1,CHNUM),'2nd Helix :',NSEL(2,CHNUM)
     ENDIF
     ! set by consh 
     QCONSH = .TRUE.
  ENDIF
  RETURN
END SUBROUTINE CONSHELIX

SUBROUTINE ECHE(ECHEL,X,Y,Z,DX,DY,DZ,HDIST,DFOR,NSEL,ASLCT,BSLCT, &
     AMASS,CHUNIT,CHSTEP,ENEMIND,LQSTPRT,MCOUNT)
  !----------------------------------------------------------------------
  !      THIS ROUTINE ADDS A Quadratic POTENTIAL to restrain between
  !      COM and COM
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use conshelix_fcm, only: MXHEL, NCOOR

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)

  real(chm_real) CRVEC(NCOOR,MXHEL)
  real(chm_real) HDIST,DFOR
  real(chm_real) AMASS(*)
  real(chm_real) TMSS(MXHEL)
  real(chm_real) MASSS,FMASSS
  real(chm_real) TMS,AMS
  real(chm_real) ATOT,DIS(NCOOR),RIJ,DEL,DF,EIJ,DFA,DDIS(NCOOR)
  real(chm_real) ENEMIND

  INTEGER NATOMM
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,J,K,L
  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT

  LOGICAL LQSTPRT
  !
  DO K=1,MXHEL
     ! center of mass for derivatives of inertial tensor
     DO L=1,NCOOR
        CRVEC(L,K) = ZERO
     ENDDO
     TMS = ZERO

     DO i=1,nsel(k)
        IF(K == 1) THEN
           NATOMM=ASLCT(I)
        ELSE
           NATOMM=BSLCT(I)            
        ENDIF

        AMS = AMASS(NATOMM)
        TMS = TMS + AMS

        CRVEC(1,K)=CRVEC(1,K)+X(NATOMM)*AMS
        CRVEC(2,K)=CRVEC(2,K)+Y(NATOMM)*AMS
        CRVEC(3,K)=CRVEC(3,K)+Z(NATOMM)*AMS
     ENDDO
     IF(TMS > ZERO) THEN
        DO L=1,NCOOR
           CRVEC(L,K)=CRVEC(L,K)/TMS
        ENDDO
     ENDIF
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'COM1',(CRVEC(L,1),L=1,3)
     WRITE(OUTU,*) 'COM2',(CRVEC(L,2),L=1,3)
  ENDIF
  !
  RIJ=ZERO
  ATOT=ZERO
  ! distance from 2 to 1 ; 1 <--- 2
  DO I=1,3
     ! The force direction is from Center Of Geometry (COG)
     DIS(I)=CRVEC(I,1)-CRVEC(I,2)
     RIJ=RIJ+DIS(I)*DIS(I)
  ENDDO
  ATOT=SQRT(RIJ)
  DEL=ATOT-HDIST

  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1

  IF(PRNLEV > 6.AND.DYNAMQ) WRITE(OUTU,*) 'mdstep,lrmdstep',mdstep,atot,lqstprt,mcount

  IF(CHSTEP > 0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') ATOT
     ENDIF
  ENDIF
  ! scalar force
  DF=DFOR*DEL
  ! energy - charmm def. k(d-d_0)^2
  EIJ=DFOR*DEL*DEL
  ! define orientation of force
  ! f=k(rij-delta)/|rij|  
  ! charmm def 2k(d-d_0)
  DFA=DF/ATOT*2.D0
  ! f=k(rij-delta)/|rij|  *rij  <--- unit vector
  ! applyed force made from constraint
  ! dfa : scalar force / |rij|
  ! dis(i) : direction vector 
  DO I=1,3
     DDIS(I)=DIS(I)*DFA
  ENDDO
  ! measure total mass of selected atoms (M)
  TMSS(1)=ZERO
  TMSS(2)=ZERO
  ! first selected
  K=1
  DO I=1,NSEL(K)
     TMSS(K)=TMSS(K)+AMASS(ASLCT(I))
  ENDDO
  ! second selected
  K=2
  DO I=1,NSEL(K)
     TMSS(K)=TMSS(K)+AMASS(BSLCT(I))
  ENDDO
  !first selected
  K=1
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     MASSS=AMASS(NATOMM)
     ! fraction of mass
     FMASSS=MASSS/TMSS(K)
     DX(NATOMM)=DX(NATOMM)+FMASSS*DDIS(1)
     DY(NATOMM)=DY(NATOMM)+FMASSS*DDIS(2)
     DZ(NATOMM)=DZ(NATOMM)+FMASSS*DDIS(3)
  ENDDO
  !second selected
  K=2
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     MASSS=AMASS(NATOMM)
     ! fraction of mass
     FMASSS=MASSS/TMSS(K)
     DX(NATOMM)=DX(NATOMM)-FMASSS*DDIS(1)
     DY(NATOMM)=DY(NATOMM)-FMASSS*DDIS(2)
     DZ(NATOMM)=DZ(NATOMM)-FMASSS*DDIS(3)
  ENDDO
  ! energy
  ECHEL=ENEMIND+EIJ
  ENEMIND=ECHEL
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(COM) E. of #', ' : ',eij
     WRITE(OUTU,*) 'enemind',enemind
  ENDIF
  RETURN
END SUBROUTINE ECHE

SUBROUTINE ECHEMD(ECHEL,X,Y,Z,DX,DY,DZ,CRVEC,HDIST,DFOR,NSEL, &
     ASLCT,BSLCT,AMASS,CTVEC,CPRVEC,CAVEC,CU,CEV,CBVEC,CEVEC, &
     CSVAL,CPVEC,LLIMIT,CHUNIT,CHSTEP,ENEMIND,LQSTPRT,MCOUNT, &
     LBHG,LBHP)
  !----------------------------------------------------------------------
  !      THIS ROUTINE ADDS A Quadratic POTENTIAL to restrain between
  !      helix and helix minimum distance restranint
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) CRVEC(NCOOR,MXHEL),CAVEC(NCOOR,MXHEL)
  real(chm_real) CTVEC(NCOOR,MXHEL),CBVEC(NCOOR,MXHEL)
  real(chm_real) CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) HDIST,DFOR
  real(chm_real) AMASS(*)
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) CPRVEC(NCOOR,MXSLT,MXHEL)
  !
  real(chm_real) ATOT,DIS(NCOOR),RIJ,DEL,DF,EIJ,DFA,DDIS(NCOOR)
  real(chm_real) TMSS(MXHEL)!,LACC(MXHEL)
  real(chm_real) DOTK, DOTKM
  real(chm_real) XCG,YCG,ZCG
  real(chm_real) DBKX(MXSLT),DBKY(MXSLT),DBKZ(MXSLT)
  real(chm_real) DEKX(MXSLT),DEKY(MXSLT),DEKZ(MXSLT)
  real(chm_real) DSKX(MXSLT),DSKY(MXSLT),DSKZ(MXSLT)
  real(chm_real) DSJX(MXSLT),DSJY(MXSLT),DSJZ(MXSLT)
  real(chm_real) R1K,RMK,R1J,RMJ
  real(chm_real) DU11,DU22,DU12,PW1,PW2,DET,PRESK,PRESJ
  real(chm_real) DDETXK(MXSLT),DDETYK(MXSLT),DDETZK(MXSLT)
  real(chm_real) DTKX(MXSLT),DTKY(MXSLT),DTKZ(MXSLT)
  real(chm_real) DTJX(MXSLT),DTJY(MXSLT),DTJZ(MXSLT)
  real(chm_real) DDKX(MXSLT),DDKY(MXSLT),DDKZ(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) DVAL,PREU,ONUM,ENEMIND
  real(chm_real) CSVAL(2)

  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER NATOMM,FSATOM,LSATOM
  INTEGER I,J,K,KK,JK,JJ

  real(chm_real) DBKINTX(MXSLT),DBKINTY(MXSLT),DBKINTZ(MXSLT)
  real(chm_real) DEKINTX(MXSLT),DEKINTY(MXSLT),DEKINTZ(MXSLT)
  real(chm_real) DBKYX(MXSLT),DBKYY(MXSLT),DBKYZ(MXSLT)
  real(chm_real) DBKZX(MXSLT),DBKZY(MXSLT),DBKZZ(MXSLT)
  real(chm_real) DEKYX(MXSLT),DEKYY(MXSLT),DEKYZ(MXSLT)
  real(chm_real) DEKZX(MXSLT),DEKZY(MXSLT),DEKZZ(MXSLT)
  real(chm_real) DPW1X(MXSLT),DPW1Y(MXSLT),DPW1Z(MXSLT)
  real(chm_real) DPW2X(MXSLT),DPW2Y(MXSLT),DPW2Z(MXSLT)
  real(chm_real) DDU12X(MXSLT),DDU12Y(MXSLT),DDU12Z(MXSLT)
  real(chm_real) DDU11X(MXSLT),DDU11Y(MXSLT),DDU11Z(MXSLT)
  !
  real(chm_real) DTKYX(MXSLT),DTKYY(MXSLT),DTKYZ(MXSLT)
  real(chm_real) DTKZX(MXSLT),DTKZY(MXSLT),DTKZZ(MXSLT)
  real(chm_real) DTJYX(MXSLT),DTJYY(MXSLT),DTJYZ(MXSLT)
  real(chm_real) DTJZX(MXSLT),DTJZY(MXSLT),DTJZZ(MXSLT)
  ! 2ND HELIX
  real(chm_real) DBJINTX(MXSLT),DBJINTY(MXSLT),DBJINTZ(MXSLT)
  real(chm_real) DEJINTX(MXSLT),DEJINTY(MXSLT),DEJINTZ(MXSLT)
  real(chm_real) DBJX(MXSLT),DBJY(MXSLT),DBJZ(MXSLT)
  real(chm_real) DBJYX(MXSLT),DBJYY(MXSLT),DBJYZ(MXSLT)
  real(chm_real) DBJZX(MXSLT),DBJZY(MXSLT),DBJZZ(MXSLT)
  real(chm_real) DEJX(MXSLT),DEJY(MXSLT),DEJZ(MXSLT)
  real(chm_real) DEJYX(MXSLT),DEJYY(MXSLT),DEJYZ(MXSLT)
  real(chm_real) DEJZX(MXSLT),DEJZY(MXSLT),DEJZZ(MXSLT)
  real(chm_real) DPW1XJ(MXSLT),DPW1YJ(MXSLT),DPW1ZJ(MXSLT)
  real(chm_real) DPW2XJ(MXSLT),DPW2YJ(MXSLT),DPW2ZJ(MXSLT)
  real(chm_real) DDU22XJ(MXSLT),DDU22YJ(MXSLT),DDU22ZJ(MXSLT)
  real(chm_real) DDU12XJ(MXSLT),DDU12YJ(MXSLT),DDU12ZJ(MXSLT)
  real(chm_real) DDETXJ(MXSLT),DDETYJ(MXSLT),DDETZJ(MXSLT)
  real(chm_real) DSKXJ(MXSLT),DSKYJ(MXSLT),DSKZJ(MXSLT)
  real(chm_real) DSJXJ(MXSLT),DSJYJ(MXSLT),DSJZJ(MXSLT)
  real(chm_real) DTJXJ(MXSLT),DTJYJ(MXSLT),DTJZJ(MXSLT)
  real(chm_real) DTJYJX(MXSLT),DTJYJY(MXSLT),DTJYJZ(MXSLT)
  real(chm_real) DTJZJX(MXSLT),DTJZJY(MXSLT),DTJZJZ(MXSLT)
  real(chm_real) DTKXJ(MXSLT),DTKYJ(MXSLT),DTKZJ(MXSLT)
  real(chm_real) DTKYJX(MXSLT),DTKYJY(MXSLT),DTKYJZ(MXSLT)
  real(chm_real) DTKZJX(MXSLT),DTKZJY(MXSLT),DTKZJZ(MXSLT)
  real(chm_real) DDJX(MXSLT),DDJY(MXSLT),DDJZ(MXSLT)
  real(chm_real) DUJX(MXSLT),DUJY(MXSLT),DUJZ(MXSLT)
  real(chm_real) PA,PB
  real(chm_real) LSK0SJ,LSK1SJ,LSJ0SK,LSJ1SK
  real(chm_real) DAX(MXSLT),DAY(MXSLT),DAZ(MXSLT)
  real(chm_real) DBX(MXSLT),DBY(MXSLT),DBZ(MXSLT)
  real(chm_real) DAXJ(MXSLT),DAYJ(MXSLT),DAZJ(MXSLT)
  real(chm_real) DBXJ(MXSLT),DBYJ(MXSLT),DBZJ(MXSLT)
  !
  LOGICAL LLIMIT,LNLIM,LQSTPRT
  LOGICAL LSK0,LSK1,LSJ0,LSJ1
  LOGICAL LSK0SJ0,LSK0SJ1,LSK1SJ0,LSK1SJ1
  !
  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT
  !
  ! CHECK ROUTINE VARIABLES
  !
  real(chm_real) W1,W2,U11,U12,U22
  real(chm_real) SK,SJ,DE,DI
  real(chm_real) BKINT,EKINT,BJINT,EJINT
  real(chm_real) BKX,BKY,BKZ,BJX,BJY,BJZ
  real(chm_real) EKX,EKY,EKZ,EJX,EJY,EJZ
  real(chm_real) TKX,TKY,TKZ,TJX,TJY,TJZ

  LOGICAL LBHG,LBHP
  !
  LSK0=(CSVAL(1) == 0.D0)
  LSK1=(CSVAL(1) == 1.D0)
  LSJ0=(CSVAL(2) == 0.D0)
  LSJ1=(CSVAL(2) == 1.D0)

  IF((.NOT.LSK0).AND.(.NOT.LSK1).AND.(.NOT.LSJ0).AND.(.NOT.LSJ1)) THEN 
     LNLIM=.TRUE.
  ELSE
     LNLIM=.FALSE.
  ENDIF

  LSK0SJ0=((CSVAL(1) == 0.D0).AND.(CSVAL(2) == 0.D0))
  LSK0SJ1=((CSVAL(1) == 0.D0).AND.(CSVAL(2) == 1.D0))
  LSK1SJ0=((CSVAL(1) == 1.D0).AND.(CSVAL(2) == 0.D0))
  LSK1SJ1=((CSVAL(1) == 1.D0).AND.(CSVAL(2) == 1.D0))
  IF(LSK0SJ0.OR.LSK0SJ1.OR.LSK1SJ0.OR.LSK1SJ1)THEN
     LSK0=.FALSE.
     LSK1=.FALSE.
     LSJ0=.FALSE.
     LSJ1=.FALSE.
  ENDIF
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'lsk0',lsk0,lsk1,lsj0,lsj1,lsk0sj0,lsk0sj1,lsk1sj0,lsk1sj1,lnlim
  ENDIF
  !
  ! U = k ( D - D_0 )^2
  ! D = |t_k - t_j|
  K=1
  J=K+1
  Dval=0.d0
  DO I=1,3
     DVAL=DVAL+(CTVEC(I,K)-CTVEC(I,J))*(CTVEC(I,K)-CTVEC(I,J))
  ENDDO
  DVAL=SQRT(DVAL)

  IF((PRNLEV >= 6).AND.DYNAMQ.AND.(CHSTEP > 0)) &
       WRITE(OUTU,*) 'HAHA',CHUNIT,CHSTEP,MDSTEP,MOD(MDSTEP,CHSTEP),DVAL

  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1

  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,DVAL,LQSTPRT,MCOUNT
     WRITE(CHUNIT,*) MDSTEP+1,DVAL,LSK0,LSK1,LSJ0,LSJ1,LSK0SJ0,LSK0SJ1,LSK1SJ0,LSK1SJ1,LNLIM
  ENDIF
  !
  IF(CHSTEP > 0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') Dval
     ENDIF
  ENDIF
  !
  ! (U_kx)' = 2 k ( D - D_0 ) (D_kx)'
  ! preU = 2 k ( D - D_0 )
  PREU=2.D0*DFOR*(Dval -hdist)
  !----------------------------------------------------------------------
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO

  IF(LBHG.OR.LBHP) THEN
     ! beta-hairpin mode and general moment of inertia
     CALL ROTINVARG(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO)
  ELSE
     ! default (helix) moment of inertia
     CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ENDIF
  !
  ! k = number of helix ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  !
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,'(A11,3F15.10)') 'eigenvector',o(1,1),o(2,1),o(3,1)         
     WRITE(OUTU,*) '    first atoms'
     WRITE(OUTU,*) '    cdxo      cdyo      cdzo'
     DO J=1,3
        WRITE(OUTU,'(3F10.5)') CDXO(K,1,J),CDYO(K,1,J),CDZO(K,1,J)
     ENDDO
     ! 1st helix
     DO J=1,3
        WRITE(OUTU,'(I2,1X,3F10.5)') J,O(1,J),O(2,J),O(3,J)
     ENDDO
  ENDIF
  !
  ! common value
  J=K+1
  !
  ! first and last atom
  FSATOM=ASLCT(1)
  LSATOM=ASLCT(NSEL(K))
  ONUM=1.D0/NSEL(K)
  !
  ! center of geometry: bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)
  !
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'COG',XCG,YCG,ZCG
  !
  !----------------------------------------------------------------------
  ! Constant value
  !
  ! dot product
  ! a_k (dot) ( r_1k - bar(r_k) )
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  DOTK=O(1,1)*(X(FSATOM)-XCG)+O(2,1)*(Y(FSATOM)-YCG)+O(3,1)*(Z(FSATOM)-ZCG)
  !
  ! a_k (dot) ( r_mk - bar(r_k) )
  DOTKM=O(1,1)*(X(LSATOM)-XCG)+O(2,1)*(Y(LSATOM)-YCG)+O(3,1)*(Z(LSATOM)-ZCG)
  !
  ! Det = U_11 * U_22 - U_12 ^2
  DU11=0.d0
  DU22=0.d0
  DU12=0.d0
  DO I=1,3
     ! absolute value |e_k - b_k|^2 and |e_j - b_j|^2
     DU11=DU11+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,K)-CBVEC(I,K))
     DU22=DU22+(CEVEC(I,J)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
     ! inner product (e_k - b_k) (dot) (e_j - b_j)
     DU12=DU12+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,J)-CBVEC(I,J))
  ENDDO
  DET =DU11*DU22-DU12*DU12
  !
  ! PreSk = - W_1 * U_22 + W_2 * U_12
  PW1=0.d0
  PW2=0.d0
  ! inner product PW1 = (b_k - b_j) (dot) (e_k - b_k)
  !               PW2 = (b_k - b_j) (dot) (e_j - b_j)
  DO I=1,3
     PW1=PW1+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,K)-CBVEC(I,K))
     PW2=PW2+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
  ENDDO
  PRESK=-PW1*DU22+PW2*DU12
  !
  ! PreSj = W_2 * U_11 - W_1 * U_12
  PRESJ=PW2*DU11-PW1*DU12
  !
  ! limit S_k = 0
  IF(LLIMIT.AND.LSK0) LSK0SJ=PW2/DU22
  !
  ! limit S_k = 1
  IF(LLIMIT.AND.LSK1) THEN
     PB=0.D0
     DO I=1,3
        PB=PB+(CEVEC(I,K)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
     ENDDO
     LSK1SJ=PB/DU22
  ENDIF
  ! limit S_j = 0
  IF(LLIMIT.AND.LSJ0) LSJ0SK=-PW1/DU11
  !
  ! limit S_j = 1
  IF(LLIMIT.AND.LSJ1) THEN
     PA=0.D0
     DO I=1,3
        PA=PA+(CEVEC(I,J)-CBVEC(I,K))*(CEVEC(I,K)-CBVEC(I,K))
     ENDDO
     LSJ1SK=PA/DU11
  ENDIF
  !-----------------------------------------------------------------------
  ! Main loop 
  DO I=1,NSEL(K)
     !
     ! (r_1kx)'
     ! i=1 -> 1, otherwise -> 0
     IF(I == 1) THEN 
        R1K=1.D0
     ELSE 
        R1K=0.D0
     ENDIF
     !
     ! (r_mkx)'
     ! i=1 -> 1, otherwise -> 0
     IF(I == NSEL(K)) THEN 
        RMK=1.D0
     ELSE 
        RMK=0.D0
     ENDIF
     !
     NATOMM=ASLCT(I)
     !
     ! (a_k (dot) (r_1k - bar(r_k)))'
     ! dbkintx by x
     ! dbkintx = (a_kx)' (r_1kx - xcg) + a_kx ( (r_1kx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_1ky - ycg) + (a_kz)' ( r_1kz - zcg)
     DBKINTX(I)=CDXO(K,I,1)*(X(FSATOM)-XCG)+O(1,1)*(R1K-ONUM) &
          +CDXO(K,I,2)*(Y(FSATOM)-YCG)+CDXO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTY(I)=CDYO(K,I,1)*(X(FSATOM)-XCG)+CDYO(K,I,2)*(Y(FSATOM)-YCG) &
          +O(2,1)*(R1K-ONUM)+CDYO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTZ(I)=CDZO(K,I,1)*(X(FSATOM)-XCG)+CDZO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(FSATOM)-ZCG)+O(3,1)*(R1K-ONUM)
     ! (a_k (dot) (r_mk - bar(r_k)))'
     ! dekintx
     ! dekintx = (a_kx)' (r_mkx - xcg) + a_kx ( (r_mkx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_mky - ycg) + (a_kz)' ( r_mkz - zcg)
     DEKINTX(I)=CDXO(K,I,1)*(X(LSATOM)-XCG)+O(1,1)*(RMK-ONUM) &
          +CDXO(K,I,2)*(Y(LSATOM)-YCG)+CDXO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTY(I)=CDYO(K,I,1)*(X(LSATOM)-XCG)+CDYO(K,I,2)*(Y(LSATOM)-YCG) &
          +O(2,1)*(RMK-ONUM)+CDYO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTZ(I)=CDZO(K,I,1)*(X(LSATOM)-XCG)+CDZO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(LSATOM)-ZCG)+O(3,1)*(RMK-ONUM)
     ! dbk 1 
     ! (b_kxx)' == (b_kx)'
     ! (b_kx)' = 1 / nsel(k) + ( a_k (dot) (r_1k - bar(r_k)))' a_kx
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kx)'
     DBKX(I)=DBKINTX(I)*O(1,1)+DOTK*CDXO(K,I,1)+ONUM
     DBKY(I)=DBKINTY(I)*O(1,1)+DOTK*CDYO(K,I,1)
     DBKZ(I)=DBKINTZ(I)*O(1,1)+DOTK*CDZO(K,I,1)
     ! b_ky by x
     ! (b_kyx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_ky
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kyx)'
     DBKYX(I)=DBKINTX(I)*O(2,1)+DOTK*CDXO(K,I,2)
     DBKYY(I)=DBKINTY(I)*O(2,1)+DOTK*CDYO(K,I,2)+ONUM
     DBKYZ(I)=DBKINTZ(I)*O(2,1)+DOTK*CDZO(K,I,2)
     ! b_kz by x 
     ! (b_kzx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_kz
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kzx)'
     DBKZX(I)=DBKINTX(I)*O(3,1)+DOTK*CDXO(K,I,3)
     DBKZY(I)=DBKINTY(I)*O(3,1)+DOTK*CDYO(K,I,3)
     DBKZZ(I)=DBKINTZ(I)*O(3,1)+DOTK*CDZO(K,I,3)+ONUM
     ! dek
     ! (e_kxx)' == (e_kx)'
     ! (e_kx) ' = 1 / nsel(k) + (a_k (dot) ( r_mk - bar(r_k)))' a_kx
     !                        + (a_k (dot) ( r_mk - bar(r_k))) (a_kx)'
     DEKX(I)=DEKINTX(I)*O(1,1)+DOTKM*CDXO(K,I,1)+ONUM
     DEKY(I)=DEKINTY(I)*O(1,1)+DOTKM*CDYO(K,I,1)
     DEKZ(I)=DEKINTZ(I)*O(1,1)+DOTKM*CDZO(K,I,1)
     ! e_ky by x
     ! (e_kyx)' = ( a_k (dot) (r_mk - bar(r_k)))' a_ky
     !          + ( a_k (dot) (r_mk - bar(r_k))) (a_kyx)'
     DEKYX(I)=DEKINTX(I)*O(2,1)+DOTKM*CDXO(K,I,2)
     DEKYY(I)=DEKINTY(I)*O(2,1)+DOTKM*CDYO(K,I,2)+ONUM
     DEKYZ(I)=DEKINTZ(I)*O(2,1)+DOTKM*CDZO(K,I,2)
     ! e_kz by x
     ! (e_kzx)' = ( a_k (dot) (r_mk - bar(r_k)))' a_kz
     !          + ( a_k (dot) (r_mk -bar(r_k))) (a_kzx)'
     DEKZX(I)=DEKINTX(I)*O(3,1)+DOTKM*CDXO(K,I,3)
     DEKZY(I)=DEKINTY(I)*O(3,1)+DOTKM*CDYO(K,I,3)
     DEKZZ(I)=DEKINTZ(I)*O(3,1)+DOTKM*CDZO(K,I,3)+ONUM
     ! (PW1)' 
     ! (b_k - b_j) (dot) (e_k - b_k)
     !  = (b_kx)'  (e_kx - b_kx) + (b_kx - b_jx) ((e_kx)' - (b_kx)')
     !  + (b_kyx)' (e_ky - b_ky) + (b_ky - b_jy) ((e_kyx)'-(b_kyx)')
     !  + (bkzx)'  (e_kz - b_kz) + (b_kz - b_jz) ((e_kzx)'-(b_kzx)')
     !----------------------------------------------------------------------
     DPW1X(I)=DBKX(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKX(I)-DBKX(I)) &
          +DBKYX(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYX(I)-DBKYX(I)) &
          +DBKZX(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZX(I)-DBKZX(I))
     DPW1Y(I)=DBKY(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKY(I)-DBKY(I)) &
          +DBKYY(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYY(I)-DBKYY(I)) &
          +DBKZY(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZY(I)-DBKZY(I))
     DPW1Z(I)=DBKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKZ(I)-DBKZ(I)) &
          +DBKYZ(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYZ(I)-DBKYZ(I)) &
          +DBKZZ(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZZ(I)-DBKZZ(I))
     ! (DU22)' by k (x,y,z) = 0
     ! (PW2)'
     ! (b_k-b_j) (dot) (e_j-b_j)
     ! (b_kx)' (e_jx - b_jx) + (b_kyx)' (e_jy - b_jy) 
     !       + (b_kzx)' (e_jz - b_jz)
     DPW2X(I)=DBKX(I)*(CEVEC(1,J)-CBVEC(1,J))+DBKYX(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZX(I)*(CEVEC(3,J)-CBVEC(3,J))
     DPW2Y(I)=DBKY(I)*(CEVEC(1,J)-CBVEC(1,J))+DBKYY(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZY(I)*(CEVEC(3,J)-CBVEC(3,J))
     DPW2Z(I)=DBKZ(I)*(CEVEC(1,J)-CBVEC(1,J))+DBKYZ(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ! (DU12)'
     ! (e_k - b_k) (dot) (e_j - b_j)
     ! ((e_kx)' -(b_kx)')(e_jx - b_jx) + ((e_kyx)' - (b_kyx)')(e_jy-b_jy)
     ! + ((e_kzx)'-(b_kzx)') (e_jz-b_jz)
     DDU12X(I)=(DEKX(I)-DBKX(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYX(I)-DBKYX(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZX(I)-DBKZX(I))*(CEVEC(3,J)-CBVEC(3,J))
     DDU12Y(I)=(DEKY(I)-DBKY(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYY(I)-DBKYY(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZY(I)-DBKZY(I))*(CEVEC(3,J)-CBVEC(3,J))
     DDU12Z(I)=(DEKZ(I)-DBKZ(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYZ(I)-DBKYZ(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZZ(I)-DBKZZ(I))*(CEVEC(3,J)-CBVEC(3,J))
     ! (DU11)'
     ! |e_k-b_k|^2
     ! 2 (e_kx - b_kx) ((e_kx)' - (b_kx)') + 2 (e_ky - b_ky) ((e_kyx)'-(b_kyx)')
     ! + 2 (e_kz - b_kz) ( (e_kzx)' - (b_kzx)' )
     DDU11X(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKX(I)-DBKX(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYX(I)-DBKYX(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZX(I)-DBKZX(I))
     DDU11Y(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKY(I)-DBKY(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYY(I)-DBKYY(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZY(I)-DBKZY(I))
     DDU11Z(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKZ(I)-DBKZ(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYZ(I)-DBKYZ(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZZ(I)-DBKZZ(I))
     ! (DET)' by k
     ! (Det)' = (DU11)' DU22 + (DU22)' DU11 - 2 DU12 (DU12)'
     ! (DU22)' = 0
     DDETXK(I)=DDU11X(I)*DU22-2.D0*DU12*DDU12X(I)
     DDETYK(I)=DDU11Y(I)*DU22-2.D0*DU12*DDU12Y(I)
     DDETZK(I)=DDU11Z(I)*DU22-2.D0*DU12*DDU12Z(I)
     !
     ! limit S_k = 1
     ! B=(e_k-b_j) (dot) (e_j-b_j)
     IF(LLIMIT.AND.LSK1) THEN
        DBX(I)=DEKX(I)*(CEVEC(1,J)-CBVEC(1,J))+DEKYX(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZX(I)*(CEVEC(3,J)-CBVEC(3,J))
        DBY(I)=DEKY(I)*(CEVEC(1,J)-CBVEC(1,J))+DEKYY(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZY(I)*(CEVEC(3,J)-CBVEC(3,J))
        DBZ(I)=DEKZ(I)*(CEVEC(1,J)-CBVEC(1,J))+DEKYZ(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ENDIF
     ! limit S_j = 1
     ! A=(b_k-e_j) (dot) (e_k-b_k)
     ! (A)' 
     IF(LLIMIT.AND.LSJ1) THEN
        DAX(I)=-DBKX(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKX(I)-DBKX(I)) &
             -DBKYX(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYX(I)-DBKYX(I)) &
             -DBKZX(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZX(I)-DBKZX(I))
        DAY(I)=-DBKY(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKY(I)-DBKY(I)) &
             -DBKYY(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYY(I)-DBKYY(I)) &
             -DBKZY(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZY(I)-DBKZY(I))
        DAZ(I)=-DBKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKZ(I)-DBKZ(I)) &
             -DBKYZ(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYZ(I)-DBKYZ(I)) &
             -DBKZZ(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! (S_k)' by k
     ! ( - (pw1)' du22 - pw1 (du22)' + (pw2)' du12 + pw2 (du12)' ) / Det
     !              + ( - pw1 du22 + pw2 du12 ) ( -1 * Det ^ -2 (Det)')
     ! (du22)' = 0 
     DSKX(I)=(-DPW1X(I)*DU22+DPW2X(I)*DU12+PW2*DDU12X(I))/DET  &
          +PRESK*(-1.D0/(DET*DET)*DDETXK(I))
     DSKY(I)=(-DPW1Y(I)*DU22+DPW2Y(I)*DU12+PW2*DDU12Y(I))/DET  &
          +PRESK*(-1.D0/(DET*DET)*DDETYK(I))
     DSKZ(I)=(-DPW1Z(I)*DU22+DPW2Z(I)*DU12+PW2*DDU12Z(I))/DET  &
          +PRESK*(-1.D0/(DET*DET)*DDETZK(I))
     ! LIMIT S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DSKX(I)=-DPW1X(I)/DU11+PW1/(DU11*DU11)*DDU11X(I)
        DSKY(I)=-DPW1Y(I)/DU11+PW1/(DU11*DU11)*DDU11Y(I)
        DSKZ(I)=-DPW1Z(I)/DU11+PW1/(DU11*DU11)*DDU11Z(I)
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DSKX(I)=DAX(I)/DU11-PA/(DU11*DU11)*DDU11X(I)
        DSKY(I)=DAY(I)/DU11-PA/(DU11*DU11)*DDU11Y(I)
        DSKZ(I)=DAZ(I)/DU11-PA/(DU11*DU11)*DDU11Z(I)
     ENDIF
     ! (S_j)' by k
     ! ( (pw2)' du11 + (du11)' pw2 - (pw1)' du12 - (du12)' pw1 ) / det
     !               + (pw2 du11 - pw1 du12) ( -1 Det^-2 (Det)')
     DSJX(I)=(DPW2X(I)*DU11+DDU11X(I)*PW2-DPW1X(I)*DU12-DDU12X(I)*PW1) &
          /DET+PRESJ*(-1.D0/(DET*DET)*DDETXK(I))
     DSJY(I)=(DPW2Y(I)*DU11+DDU11Y(I)*PW2-DPW1Y(I)*DU12-DDU12Y(I)*PW1) &
          /DET+PRESJ*(-1.D0/(DET*DET)*DDETYK(I))
     DSJZ(I)=(DPW2Z(I)*DU11+DDU11Z(I)*PW2-DPW1Z(I)*DU12-DDU12Z(I)*PW1) &
          /DET+PRESJ*(-1.D0/(DET*DET)*DDETZK(I))
     ! LIMIT S_K = 0
     IF(LLIMIT.AND.LSK0) THEN
        DSJX(I)=DPW2X(I)/DU22
        DSJY(I)=DPW2Y(I)/DU22
        DSJZ(I)=DPW2Z(I)/DU22
     ENDIF
     !
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DSJX(I)=DBX(I)/DU22
        DSJY(I)=DBY(I)/DU22
        DSJZ(I)=DBZ(I)/DU22
     ENDIF
     !(t_kxx)'=(t_kx)'
     !(t_kx)' = (b_kx)' + (S_k)' ( e_kx - b_kx ) + S_k ( (e_kx)' - (b_kx)' )
     DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKX(I)-DBKX(I))
     DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKY(I)-DBKY(I))
     DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKZ(I)-DBKZ(I))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKX(I)=DBKX(I)
        DTKY(I)=DBKY(I)
        DTKZ(I)=DBKZ(I)
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKX(I)=DEKX(I)
        DTKY(I)=DEKY(I)
        DTKZ(I)=DEKZ(I)
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKX(I)-DBKX(I))
        DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKY(I)-DBKY(I))
        DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKZ(I)-DBKZ(I))
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKX(I)-DBKX(I))
        DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKY(I)-DBKY(I))
        DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKZ(I)-DBKZ(I))
     ENDIF
     ! (t_kyx)' = (b_ky)' + (S_k)' (e_ky-b_ky) + S_k ( (e_ky)' - (b_ky)' )
     DTKYX(I)=DBKYX(I)+DSKX(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYX(I)-DBKYX(I))      
     DTKYY(I)=DBKYY(I)+DSKY(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYY(I)-DBKYY(I))      
     DTKYZ(I)=DBKYZ(I)+DSKZ(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYZ(I)-DBKYZ(I))      
     ! LIMIT S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKYX(I)=DBKYX(I)
        DTKYY(I)=DBKYY(I)
        DTKYZ(I)=DBKYZ(I)
     ENDIF
     ! LIMIT S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKYX(I)=DEKYX(I)
        DTKYY(I)=DEKYY(I)
        DTKYZ(I)=DEKYZ(I)
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKYX(I)=DBKYX(I)+DSKX(I)*(CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYX(I)-DBKYX(I))      
        DTKYY(I)=DBKYY(I)+DSKY(I)*(CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYY(I)-DBKYY(I))      
        DTKYZ(I)=DBKYZ(I)+DSKZ(I)*(CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYZ(I)-DBKYZ(I))      
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKYX(I)=DBKYX(I)+DSKX(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYX(I)-DBKYX(I))      
        DTKYY(I)=DBKYY(I)+DSKY(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYY(I)-DBKYY(I))      
        DTKYZ(I)=DBKYZ(I)+DSKZ(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYZ(I)-DBKYZ(I))      
     ENDIF
     ! (t_kzx)' = (b_kz)' + (S_k)' (e_kz-b_kz) 
     !                    + S_k ( (e_kz)' - (b_kz)' )
     DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZX(I)-DBKZX(I))
     DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZY(I)-DBKZY(I))
     DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZZ(I)-DBKZZ(I))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKZX(I)=DBKZX(I)
        DTKZY(I)=DBKZY(I)
        DTKZZ(I)=DBKZZ(I)         
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 )
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKZX(I)=DEKZX(I)
        DTKZY(I)=DEKZY(I)
        DTKZZ(I)=DEKZZ(I)         
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZX(I)-DBKZX(I))
        DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZY(I)-DBKZY(I))
        DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZX(I)-DBKZX(I))
        DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZY(I)-DBKZY(I))
        DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! (t_jx)' by k
     ! (t_jx)' = (S_j)' (e_jx - b_jx)
     DTJX(I)=DSJX(I)*(CEVEC(1,J)-CBVEC(1,J))
     DTJY(I)=DSJY(I)*(CEVEC(1,J)-CBVEC(1,J))
     DTJZ(I)=DSJZ(I)*(CEVEC(1,J)-CBVEC(1,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJX(I)=DSJX(I)*(CEVEC(1,J)-CBVEC(1,J))
        DTJY(I)=DSJY(I)*(CEVEC(1,J)-CBVEC(1,J))
        DTJZ(I)=DSJZ(I)*(CEVEC(1,J)-CBVEC(1,J))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJX(I)=0.D0
        DTJY(I)=0.D0
        DTJZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJX(I)=0.D0
        DTJY(I)=0.D0
        DTJZ(I)=0.D0
     ENDIF
     ! (t_jy)' by k
     ! (t_jy)' = (S_j)' (e_jy -b_jy)
     DTJYX(I)=DSJX(I)*(CEVEC(2,J)-CBVEC(2,J))
     DTJYY(I)=DSJY(I)*(CEVEC(2,J)-CBVEC(2,J))
     DTJYZ(I)=DSJZ(I)*(CEVEC(2,J)-CBVEC(2,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJYX(I)=DSJX(I)*(CEVEC(2,J)-CBVEC(2,J))
        DTJYY(I)=DSJY(I)*(CEVEC(2,J)-CBVEC(2,J))
        DTJYZ(I)=DSJZ(I)*(CEVEC(2,J)-CBVEC(2,J))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJYX(I)=0.D0
        DTJYY(I)=0.D0
        DTJYZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJYX(I)=0.D0
        DTJYY(I)=0.D0
        DTJYZ(I)=0.D0
     ENDIF
     ! (t_jz)' by k
     ! (t_jz)' = (S_j)' (e_jz - b_jz)
     DTJZX(I)=DSJX(I)*(CEVEC(3,J)-CBVEC(3,J))
     DTJZY(I)=DSJY(I)*(CEVEC(3,J)-CBVEC(3,J))
     DTJZZ(I)=DSJZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJZX(I)=DSJX(I)*(CEVEC(3,J)-CBVEC(3,J))
        DTJZY(I)=DSJY(I)*(CEVEC(3,J)-CBVEC(3,J))
        DTJZZ(I)=DSJZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJZX(I)=0.D0
        DTJZY(I)=0.D0
        DTJZZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJZX(I)=0.D0
        DTJZY(I)=0.D0
        DTJZZ(I)=0.D0
     ENDIF
     ! (D_kx)' = 1 / (2 D) [  2 (t_kx-t_jx) ( (t_kx)' - (t_jx)' )
     !                      + 2 (t_ky-t_jy) ( (t_ky)' - (t_jy)' )
     !                      + 2 (t_kz-t_jz) ( (t_kz)' - (t_jz)' ) ]
     DDKX(I)=(2.D0*(CTVEC(1,K)-CTVEC(1,J))*(DTKX(I)-DTJX(I)) &
          +2.D0*(CTVEC(2,K)-CTVEC(2,J))*(DTKYX(I)-DTJYX(I)) &
          +2.D0*(CTVEC(3,K)-CTVEC(3,J))*(DTKZX(I)-DTJZX(I)))/(2.D0*DVAL)
     DDKY(I)=(2.D0*(CTVEC(1,K)-CTVEC(1,J))*(DTKY(I)-DTJY(I)) &
          +2.D0*(CTVEC(2,K)-CTVEC(2,J))*(DTKYY(I)-DTJYY(I)) &
          +2.D0*(CTVEC(3,K)-CTVEC(3,J))*(DTKZY(I)-DTJZY(I)))/(2.D0*DVAL)
     DDKZ(I)=(2.D0*(CTVEC(1,K)-CTVEC(1,J))*(DTKZ(I)-DTJZ(I)) &
          +2.D0*(CTVEC(2,K)-CTVEC(2,J))*(DTKYZ(I)-DTJYZ(I)) &
          +2.D0*(CTVEC(3,K)-CTVEC(3,J))*(DTKZZ(I)-DTJZZ(I)))/(2.D0*DVAL)
     ! (U_kx)' = 2 k (D - D_0) (D_kx)'
     DUKX(I)=PREU*DDKX(I)
     DUKY(I)=PREU*DDKY(I)
     DUKZ(I)=PREU*DDKZ(I)
  ENDDO
  !
  IF(PRNLEV >= 6) THEN
     ! check routine
     K=1
     J=K+1
     W1=0.D0
     W2=0.D0
     U11=0.D0
     U12=0.D0
     U22=0.D0
     DO I=1,3
        W1=W1+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,K)-CBVEC(I,K))
        W2=W2+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
        U11=U11+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,K)-CBVEC(I,K))
        U12=U12+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,J)-CBVEC(I,J))
        U22=U22+(CEVEC(I,J)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
     ENDDO

     DE=U11*U22-U12*U12
     SK=(-W1*U22+W2*U12)/DE
     SJ=(W2*U11-W1*U12)/DE

     WRITE(OUTU,*) 'DE,SK,SJ',DE,SK,SJ
     WRITE(OUTU,*) DET,PRESK/DET,PRESJ/DET

     BKINT=CAVEC(1,K)*(X(ASLCT(1))-XCG)+CAVEC(2,K)*(Y(ASLCT(1))-YCG) &
          +CAVEC(3,K)*(Z(ASLCT(1))-ZCG)
     BKX=XCG+BKINT*CAVEC(1,K)
     BKY=YCG+BKINT*CAVEC(2,K)
     BKZ=ZCG+BKINT*CAVEC(3,K)

     WRITE(OUTU,*) 'BKINT',BKINT,DOTK
     WRITE(OUTU,*) 'XCG,YCG,ZCG',XCG,YCG,ZCG
     WRITE(OUTU,*) CRVEC(1,1),CRVEC(2,1),CRVEC(3,1)
     WRITE(OUTU,*) 'CAVEC',CAVEC(1,1),CAVEC(2,1),CAVEC(3,1)
     WRITE(OUTU,*) O(1,1),O(2,1),O(3,1)
     WRITE(OUTU,*) X(ASLCT(1)),Y(ASLCT(1)),Z(ASLCT(1))
     WRITE(OUTU,*) 'BKX,BKY,BKZ',BKX,BKY,BKZ
     WRITE(OUTU,*) 'BKX,BKY,BKZ',CBVEC(1,1),CBVEC(2,1),CBVEC(3,1)
     !
     EKINT=CAVEC(1,K)*(X(ASLCT(NSEL(K)))-XCG)+CAVEC(2,K)*(Y(ASLCT(NSEL(K)))-YCG) &
          +CAVEC(3,K)*(Z(ASLCT(NSEL(K)))-ZCG)
     EKX=XCG+EKINT*CAVEC(1,K)
     EKY=YCG+EKINT*CAVEC(2,K)
     EKZ=ZCG+EKINT*CAVEC(3,K)
     !
     K=2
     XCG=ZERO
     YCG=ZERO
     ZCG=ZERO
     !
     DO I=1,NSEL(K)
        NATOMM=BSLCT(I)
        XCG=XCG+X(NATOMM)
        YCG=YCG+Y(NATOMM)
        ZCG=ZCG+Z(NATOMM)
     ENDDO
     XCG=XCG/NSEL(K)
     YCG=YCG/NSEL(K)
     ZCG=ZCG/NSEL(K)

     BJINT=CAVEC(1,J)*(X(BSLCT(1))-XCG) &
          +CAVEC(2,J)*(Y(BSLCT(1))-YCG) &
          +CAVEC(3,J)*(Z(BSLCT(1))-ZCG)
     BJX=XCG+BJINT*CAVEC(1,J)
     BJY=YCG+BJINT*CAVEC(2,J)
     BJZ=ZCG+BJINT*CAVEC(3,J)
     !
     EJINT=CAVEC(1,J)*(X(BSLCT(NSEL(K)))-XCG) &
          +CAVEC(2,J)*(Y(BSLCT(NSEL(K)))-YCG) &
          +CAVEC(3,J)*(Z(BSLCT(NSEL(K)))-ZCG)
     EJX=XCG+EJINT*CAVEC(1,J)
     EJY=YCG+EJINT*CAVEC(2,J)
     EJZ=ZCG+EJINT*CAVEC(3,J)
     !
     TKX=BKX+SK*(EKX-BKX)
     TKY=BKY+SK*(EKY-BKY)
     TKZ=BKZ+SK*(EKZ-BKZ)
     !
     TJX=BJX+SJ*(EJX-BJX)
     TJY=BJY+SJ*(EJY-BJY)
     TJZ=BJZ+SJ*(EJZ-BJZ)
     !
     DI=SQRT((TKX-TJX)**2+(TKY-TJY)**2+(TKZ-TJZ)**2)
     !
     WRITE(OUTU,'(6(1X,F10.5))') TKX,TKY,TKZ,TJX,TJY,TJZ
     WRITE(OUTU,'(6(1X,F10.5))') CTVEC(1,1),CTVEC(2,1),CTVEC(3,1),CTVEC(1,2),CTVEC(2,2),CTVEC(3,2)
     WRITE(OUTU,*) 'CHECK ROUTINE-DI,DVAL', DI,DVAL
  ENDIF
  !-----------------------------------------------------------------------
  ! SECOND HELIX
  !-----------------------------------------------------------------------
  K=2
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO

  IF(LBHG.OR.LBHP) THEN
     ! beta-hairpin mode and general moment of inertia
     !      write(6,*) 'hairpin: general moment of inertia'
     CALL ROTINVARG(K,X,Y,Z,NSEL,BSLCT,AMASS,O,EV,CDXO,CDYO,CDZO)
  ELSE
     ! default (helix) moment of inertia
     CALL ROTINVAR2(K,X,Y,Z,NSEL,BSLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ENDIF
  !
  J=K-1
  JK=1 
  JJ=2
  FSATOM=BSLCT(1)
  LSATOM=BSLCT(NSEL(K))
  ONUM=1.D0/NSEL(K)
  !
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)
  ! dot product
  ! a_j (dot) ( r_1j - bar(r_j) )
  ! a_jx -> o(1,1), a_jy -> o(2,1), a_jz -> o(3,1)
  DOTK=O(1,1)*(X(FSATOM)-XCG)+O(2,1)*(Y(FSATOM)-YCG)+O(3,1)*(Z(FSATOM)-ZCG)
  ! a_j (dot) ( r_mj - bar(r_j) )
  DOTKM=O(1,1)*(X(LSATOM)-XCG)+O(2,1)*(Y(LSATOM)-YCG)+O(3,1)*(Z(LSATOM)-ZCG)
  !
  DO I=1,NSEL(K)
     ! (r_1jx)'
     ! i=1       --> 1
     ! otherwise --> 0
     IF(I == 1) THEN 
        R1J=1.D0
     ELSE 
        R1J=0.D0
     ENDIF
     ! (r_mjx)'
     ! i=m       --> 1
     ! otherwise --> 0
     IF(I == NSEL(K)) THEN 
        RMJ=1.D0
     ELSE 
        RMJ=0.D0
     ENDIF
     !
     NATOMM=BSLCT(I)
     ! (a_j (dot) (r_1j - bar(r_j)))'
     DBJINTX(I)=CDXO(K,I,1)*(X(FSATOM)-XCG)+O(1,1)*(R1J-ONUM) &
          +CDXO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(FSATOM)-ZCG)
     DBJINTY(I)=CDYO(K,I,1)*(X(FSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(FSATOM)-YCG)+O(2,1)*(R1J-ONUM) &
          +CDYO(K,I,3)*(Z(FSATOM)-ZCG)
     DBJINTZ(I)=CDZO(K,I,1)*(X(FSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(FSATOM)-ZCG)+O(3,1)*(R1J-ONUM)
     ! (a_j (dot) (r_mj - bar(r_j)))'
     DEJINTX(I)=CDXO(K,I,1)*(X(LSATOM)-XCG)+O(1,1)*(RMJ-ONUM ) &
          +CDXO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(LSATOM)-ZCG)
     DEJINTY(I)=CDYO(K,I,1)*(X(LSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(LSATOM)-YCG)+O(2,1)*(RMJ-ONUM) &
          +CDYO(K,I,3)*(Z(LSATOM)-ZCG)
     DEJINTZ(I)=CDZO(K,I,1)*(X(LSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(LSATOM)-ZCG)+O(3,1)*(RMJ-ONUM)
     ! (b_jx)' == (b_jxx)'
     DBJX(I)=DBJINTX(I)*O(1,1)+DOTK*CDXO(K,I,1)+ONUM
     DBJY(I)=DBJINTY(I)*O(1,1)+DOTK*CDYO(K,I,1)
     DBJZ(I)=DBJINTZ(I)*O(1,1)+DOTK*CDZO(K,I,1)
     ! (b_jyx)'
     DBJYX(I)=DBJINTX(I)*O(2,1)+DOTK*CDXO(K,I,2)
     DBJYY(I)=DBJINTY(I)*O(2,1)+DOTK*CDYO(K,I,2)+ONUM
     DBJYZ(I)=DBJINTZ(I)*O(2,1)+DOTK*CDZO(K,I,2)
     ! (b_jzx)'
     DBJZX(I)=DBJINTX(I)*O(3,1)+DOTK*CDXO(K,I,3)
     DBJZY(I)=DBJINTY(I)*O(3,1)+DOTK*CDYO(K,I,3)
     DBJZZ(I)=DBJINTZ(I)*O(3,1)+DOTK*CDZO(K,I,3)+ONUM
     ! (e_jx)' = 1 / nsel(k) + [ (a_jx)' ( r_mjx - 1 / nsel(k) * Si(x_i) ) 
     !                       - a_jx ( (r_mjx)' - 1 / nsel(k) ) ] * a_jx
     !                       + [ a_j (dot) ( r_mj - bar(r_j) ) ] (a_jx)'
     DEJX(I)=DEJINTX(I)*O(1,1)+DOTKM*CDXO(K,I,1)+ONUM
     DEJY(I)=DEJINTY(I)*O(1,1)+DOTKM*CDYO(K,I,1)
     DEJZ(I)=DEJINTZ(I)*O(1,1)+DOTKM*CDZO(K,I,1)
     ! (e_jyx)'
     DEJYX(I)=DEJINTX(I)*O(2,1)+DOTKM*CDXO(K,I,2)
     DEJYY(I)=DEJINTY(I)*O(2,1)+DOTKM*CDYO(K,I,2)+ONUM
     DEJYZ(I)=DEJINTZ(I)*O(2,1)+DOTKM*CDZO(K,I,2)
     ! (e_jzx)'
     DEJZX(I)=DEJINTX(I)*O(3,1)+DOTKM*CDXO(K,I,3)
     DEJZY(I)=DEJINTY(I)*O(3,1)+DOTKM*CDYO(K,I,3)
     DEJZZ(I)=DEJINTZ(I)*O(3,1)+DOTKM*CDZO(K,I,3)+ONUM
     ! (pw1)'
     DPW1XJ(I)=-DBJX(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYX(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZX(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DPW1YJ(I)=-DBJY(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYY(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZY(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DPW1ZJ(I)=-DBJZ(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYZ(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZZ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ! (pw2)'
     DPW2XJ(I)=-DBJX(I)*(CEVEC(1,JJ)-CBVEC(1,JJ)) &
          -DBJYX(I)*(CEVEC(2,JJ)-CBVEC(2,JJ)) &
          -DBJZX(I)*(CEVEC(3,JJ)-CBVEC(3,JJ)) &
          +(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
          +(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
          +(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))  
     DPW2YJ(I)=-DBJY(I)*(CEVEC(1,JJ)-CBVEC(1,JJ)) &
          -DBJYY(I)*(CEVEC(2,JJ)-CBVEC(2,JJ)) &
          -DBJZY(I)*(CEVEC(3,JJ)-CBVEC(3,JJ)) &
          +(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &  
          +(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
          +(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZY(I)-DBJZY(I))  
     DPW2ZJ(I)=-DBJZ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ)) &
          -DBJYZ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ)) &
          -DBJZZ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ)) &
          +(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &  
          +(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
          +(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))  
     ! (du22)'
     DDU22XJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))
     DDU22YJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     DDU22ZJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     ! (du12)'
     DDU12XJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJX(I)-DBJX(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYX(I)-DBJYX(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZX(I)-DBJZX(I))
     DDU12YJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJY(I)-DBJY(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYY(I)-DBJYY(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZY(I)-DBJZY(I))
     DDU12ZJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJZ(I)-DBJZ(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYZ(I)-DBJYZ(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZZ(I)-DBJZZ(I))
     ! (DU11)' = 0
     ! (det)' 
     DDETXJ(I)=DU11*DDU22XJ(I)-TWO*DU12*DDU12XJ(I)
     DDETYJ(I)=DU11*DDU22YJ(I)-TWO*DU12*DDU12YJ(I)
     DDETZJ(I)=DU11*DDU22ZJ(I)-TWO*DU12*DDU12ZJ(I)
     ! limit S_K = 1
     ! B=(e_k-b_j) (dot) (e_j-b_j)
     IF(LLIMIT.AND.LSK1) THEN
        DBXJ(I)=-DBJX(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
             -DBJYX(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
             -DBJZX(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))
        DBYJ(I)=-DBJY(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &
             -DBJYY(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
             -DBJZY(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZY(I)-DBJZY(I))
        DBZJ(I)=-DBJZ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &
             -DBJYZ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
             -DBJZZ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! limit S_j = 1
     ! A=(b_k-e_j) (dot) (e_k-b_k)
     ! (A)'
     IF(LLIMIT.AND.LSJ1) THEN
        DAXJ(I)=-DEJX(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYX(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZX(I)*(CEVEC(3,JK)-CBVEC(3,JK))
        DAYJ(I)=-DEJY(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYY(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZY(I)*(CEVEC(3,JK)-CBVEC(3,JK))
        DAZJ(I)=-DEJZ(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYZ(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZZ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ENDIF
     ! (S_k)'
     DSKXJ(I)=(-DPW1XJ(I)*DU22-PW1*DDU22XJ(I) &
          +DPW2XJ(I)*DU12+PW2*DDU12XJ(I))/DET &
          +PRESK*(-1.D0/(DET*DET)*DDETXJ(I))
     DSKYJ(I)=(-DPW1YJ(I)*DU22-PW1*DDU22YJ(I) &
          +DPW2YJ(I)*DU12+PW2*DDU12YJ(I))/DET &
          +PRESK*(-1.D0/(DET*DET)*DDETYJ(I))
     DSKZJ(I)=(-DPW1ZJ(I)*DU22-PW1*DDU22ZJ(I) &
          +DPW2ZJ(I)*DU12+PW2*DDU12ZJ(I))/DET &
          +PRESK*(-1.D0/(DET*DET)*DDETZJ(I))
     ! LIMIT S_J = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DSKXJ(I)=-DAXJ(I)/DU11
        DSKYJ(I)=-DAYJ(I)/DU11
        DSKZJ(I)=-DAZJ(I)/DU11
     ENDIF
     ! Bugs in crossing angle Sj = 0
     ! Limit S_J = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DSKXJ(I)=-DPW1XJ(I)/DU11
        DSKYJ(I)=-DPW1YJ(I)/DU11
        DSKZJ(I)=-DPW1ZJ(I)/DU11
     ENDIF
     ! (S_J)'
     DSJXJ(I)=(DPW2XJ(I)*DU11-DPW1XJ(I)*DU12-PW1*DDU12XJ(I))/DET+PRESJ*(-1.D0/(DET*DET)*DDETXJ(I))
     DSJYJ(I)=(DPW2YJ(I)*DU11-DPW1YJ(I)*DU12-PW1*DDU12YJ(I))/DET+PRESJ*(-1.D0/(DET*DET)*DDETYJ(I))
     DSJZJ(I)=(DPW2ZJ(I)*DU11-DPW1ZJ(I)*DU12-PW1*DDU12ZJ(I))/DET+PRESJ*(-1.D0/(DET*DET)*DDETZJ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DSJXJ(I)=DPW2XJ(I)/DU22-PW2/(DU22*DU22)*DDU22XJ(I)
        DSJYJ(I)=DPW2YJ(I)/DU22-PW2/(DU22*DU22)*DDU22YJ(I)
        DSJZJ(I)=DPW2ZJ(I)/DU22-PW2/(DU22*DU22)*DDU22ZJ(I)
     ENDIF
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DSJXJ(I)=DBXJ(I)/DU22-PB/(DU22*DU22)*DDU22XJ(I)
        DSJYJ(I)=DBYJ(I)/DU22-PB/(DU22*DU22)*DDU22YJ(I)
        DSJZJ(I)=DBZJ(I)/DU22-PB/(DU22*DU22)*DDU22ZJ(I)
     ENDIF
     ! (t_jx)' = (t_jxx)'
     DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJX(I)-DBJX(I))
     DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJY(I)-DBJY(I))
     DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJZ(I)-DBJZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJX(I)-DBJX(I))
        DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJY(I)-DBJY(I))
        DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJZ(I)-DBJZ(I))
     ENDIF
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJX(I)-DBJX(I))
        DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJY(I)-DBJY(I))
        DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJZ(I)-DBJZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJXJ(I)=DBJX(I)
        DTJYJ(I)=DBJY(I)
        DTJZJ(I)=DBJZ(I)
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJXJ(I)=DEJX(I)
        DTJYJ(I)=DEJY(I)
        DTJZJ(I)=DEJZ(I)
     ENDIF
     ! (t_jy)'
     DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYX(I)-DBJYX(I))
     DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYY(I)-DBJYY(I))
     DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYZ(I)-DBJYZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYX(I)-DBJYX(I))
        DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYY(I)-DBJYY(I))
        DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYZ(I)-DBJYZ(I))
     ENDIF
     ! limit S_k = 1 
     IF(LLIMIT.AND.LSK1) THEN
        DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYX(I)-DBJYX(I))
        DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYY(I)-DBJYY(I))
        DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYZ(I)-DBJYZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJYJX(I)=DBJYX(I)
        DTJYJY(I)=DBJYY(I)
        DTJYJZ(I)=DBJYZ(I)
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJYJX(I)=DEJYX(I)
        DTJYJY(I)=DEJYY(I)
        DTJYJZ(I)=DEJYZ(I)
     ENDIF
     ! (t_jz)'
     DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZX(I)-DBJZX(I))
     DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZY(I)-DBJZY(I))
     DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZZ(I)-DBJZZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZX(I)-DBJZX(I))
        DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZY(I)-DBJZY(I))
        DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZX(I)-DBJZX(I))
        DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZY(I)-DBJZY(I))
        DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJZJX(I)=DBJZX(I)
        DTJZJY(I)=DBJZY(I)
        DTJZJZ(I)=DBJZZ(I)
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJZJX(I)=DEJZX(I)
        DTJZJY(I)=DEJZY(I)
        DTJZJZ(I)=DEJZZ(I)
     ENDIF
     ! (t_kx)'
     DTKXJ(I)=DSKXJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     DTKYJ(I)=DSKYJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     DTKZJ(I)=DSKZJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKXJ(I)=0.D0
        DTKYJ(I)=0.D0
        DTKZJ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKXJ(I)=0.D0
        DTKYJ(I)=0.D0
        DTKZJ(I)=0.D0
     ENDIF
     ! (t_ky)'
     DTKYJX(I)=DSKXJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     DTKYJY(I)=DSKYJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     DTKYJZ(I)=DSKZJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKYJX(I)=0.D0
        DTKYJY(I)=0.D0
        DTKYJZ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKYJX(I)=0.D0
        DTKYJY(I)=0.D0
        DTKYJZ(I)=0.D0
     ENDIF
     ! (t_kz)'
     DTKZJX(I)=DSKXJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DTKZJY(I)=DSKYJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DTKZJZ(I)=DSKZJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 )
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKZJX(I)=0.D0
        DTKZJY(I)=0.D0
        DTKZJZ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKZJX(I)=0.D0
        DTKZJY(I)=0.D0
        DTKZJZ(I)=0.D0
     ENDIF
     ! (D_jx)'
     DDJX(I)=(TWO*(CTVEC(1,JK)-CTVEC(1,JJ))*(DTKXJ(I)-DTJXJ(I)) &
          +TWO*(CTVEC(2,JK)-CTVEC(2,JJ))*(DTKYJX(I)-DTJYJX(I)) &
          +TWO*(CTVEC(3,JK)-CTVEC(3,JJ))*(DTKZJX(I)-DTJZJX(I)))/(TWO*DVAL)
     DDJY(I)=(TWO*(CTVEC(1,JK)-CTVEC(1,JJ))*(DTKYJ(I)-DTJYJ(I)) &
          +TWO*(CTVEC(2,JK)-CTVEC(2,JJ))*(DTKYJY(I)-DTJYJY(I)) &
          +TWO*(CTVEC(3,JK)-CTVEC(3,JJ))*(DTKZJY(I)-DTJZJY(I)))/(TWO*DVAL)
     DDJZ(I)=(TWO*(CTVEC(1,JK)-CTVEC(1,JJ))*(DTKZJ(I)-DTJZJ(I)) &
          +TWO*(CTVEC(2,JK)-CTVEC(2,JJ))*(DTKYJZ(I)-DTJYJZ(I)) &
          +TWO*(CTVEC(3,JK)-CTVEC(3,JJ))*(DTKZJZ(I)-DTJZJZ(I)))/(TWO*DVAL)
     ! (U_jx)'
     DUJX(I)=PREU*DDJX(I)
     DUJY(I)=PREU*DDJY(I)
     DUJZ(I)=PREU*DDJZ(I)
  ENDDO
  ! add to DX, DY, DZ
  ! U = k ( D - D_0 )^2
  EIJ=DFOR*(DVAL-HDIST)*(DVAL-HDIST)
  ! first selected
  K=1
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO
  !
  IF(PRNLEV >= 6) THEN
     ! k=1
     WRITE(6,*) 'CPRVEC'
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     !
     WRITE(6,*) 'FORCE'
     !
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I),DUKY(I),DUKZ(I)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUJX(I),DUJY(I),DUJZ(I)
     ENDDO
  ENDIF
  !
  ! SECOND SELECTED
  !
  K=2
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUJX(I)
     DY(NATOMM)=DY(NATOMM)+DUJY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUJZ(I)
  ENDDO
  !
  ECHEL=ENEMIND+EIJ
  ENEMIND=ECHEL
  !
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'CONST(MIND) E. OF #', ' : ',EIJ
     WRITE(OUTU,*) 'ENEMIND',ENEMIND
  ENDIF
  RETURN
END SUBROUTINE ECHEMD

SUBROUTINE ECHXA(ECHEL,X,Y,Z,DX,DY,DZ,ANGL,AFOR,NSEL,ASLCT,BSLCT, &
     AMASS,CTVEC,CPRVEC,CAVEC,CU,CEV,CBVEC,CEVEC,CSVAL,CPVEC,LLIMIT, &
     CHUNIT,CHSTEP,ENEANG,LQSTPRT,MCOUNT)
  !----------------------------------------------------------------------
  !     FORCE CALCULATION ABOUT CROSS ANGLE(OMEGA) BETWEEN TWO AXIS
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT
  use vector

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) CAVEC(NCOOR,MXHEL),CTVEC(NCOOR,MXHEL)
  real(chm_real) CBVEC(NCOOR,MXHEL),CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) ANGL,AFOR
  real(chm_real) AMASS(*)
  real(chm_real) DOT,THETA,TMP,TMP2,TMP3
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) C1(3),C2(3),C3(3),C1XC2(3),C2XC3(3),ORIE(3)
  real(chm_real) COST,DOTP,DOTPORIE,AKXB,BXC,TJMTK
  real(chm_real) DTJMTKX(MXSLT),DTJMTKY(MXSLT),DTJMTKZ(MXSLT)
  real(chm_real) DTBX(MXSLT),DTBY(MXSLT),DTBZ(MXSLT)
  real(chm_real) DTBYX(MXSLT),DTBYY(MXSLT),DTBYZ(MXSLT)
  real(chm_real) DTBZX(MXSLT),DTBZY(MXSLT),DTBZZ(MXSLT)
  real(chm_real) DAKXBKX(MXSLT),DAKXBKY(MXSLT),DAKXBKZ(MXSLT)
  real(chm_real) DBXCX(MXSLT),DBXCY(MXSLT),DBXCZ(MXSLT)
  real(chm_real) DCOSTX(MXSLT),DCOSTY(MXSLT),DCOSTZ(MXSLT)
  real(chm_real) DTJMTKXJ(MXSLT),DTJMTKYJ(MXSLT),DTJMTKZJ(MXSLT)
  real(chm_real) DTBXJ(MXSLT),DTBYJ(MXSLT),DTBZJ(MXSLT)
  real(chm_real) DTBYXJ(MXSLT),DTBYYJ(MXSLT),DTBYZJ(MXSLT)
  real(chm_real) DTBZXJ(MXSLT),DTBZYJ(MXSLT),DTBZZJ(MXSLT)
  real(chm_real) DAKXBJX(MXSLT),DAKXBJY(MXSLT),DAKXBJZ(MXSLT)
  real(chm_real) DBXCXJ(MXSLT),DBXCYJ(MXSLT),DBXCZJ(MXSLT)
  real(chm_real) DCOSTXJ(MXSLT),DCOSTYJ(MXSLT),DCOSTZJ(MXSLT)
  real(chm_real) EIJ

  INTEGER NATOMM,FSATOM,LSATOM
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,J,K,KK,JK,JJ

  real(chm_real) XCG,YCG,ZCG,DOTK,DOTKM
  real(chm_real) DBKX(MXSLT),DBKY(MXSLT),DBKZ(MXSLT)
  real(chm_real) DEKX(MXSLT),DEKY(MXSLT),DEKZ(MXSLT)
  real(chm_real) DSKX(MXSLT),DSKY(MXSLT),DSKZ(MXSLT)
  real(chm_real) DSJX(MXSLT),DSJY(MXSLT),DSJZ(MXSLT)
  real(chm_real) R1K,RMK,R1J,RMJ
  real(chm_real) EJMBJSQ,EKMBKDEJMBJ,BKMBJDEJMBJ
  real(chm_real) DU11,DU22,DU12,PW1,PW2,DET,PRESK,PRESJ
  real(chm_real) DDETXK(MXSLT),DDETYK(MXSLT),DDETZK(MXSLT)
  real(chm_real) DTKX(MXSLT),DTKY(MXSLT),DTKZ(MXSLT)
  real(chm_real) DTJX(MXSLT),DTJY(MXSLT),DTJZ(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) PREU,ONUM,ENEANG
  real(chm_real) CSVAL(2)
  real(chm_real) DBKINTX(MXSLT),DBKINTY(MXSLT),DBKINTZ(MXSLT)
  real(chm_real) DEKINTX(MXSLT),DEKINTY(MXSLT),DEKINTZ(MXSLT)
  real(chm_real) DBKYX(MXSLT),DBKYY(MXSLT),DBKYZ(MXSLT)
  real(chm_real) DBKZX(MXSLT),DBKZY(MXSLT),DBKZZ(MXSLT)
  real(chm_real) DEKYX(MXSLT),DEKYY(MXSLT),DEKYZ(MXSLT)
  real(chm_real) DEKZX(MXSLT),DEKZY(MXSLT),DEKZZ(MXSLT)
  real(chm_real) DPW1X(MXSLT),DPW1Y(MXSLT),DPW1Z(MXSLT)
  real(chm_real) DPW2X(MXSLT),DPW2Y(MXSLT),DPW2Z(MXSLT)
  real(chm_real) DDU12X(MXSLT),DDU12Y(MXSLT),DDU12Z(MXSLT)
  real(chm_real) DDU11X(MXSLT),DDU11Y(MXSLT),DDU11Z(MXSLT)
  real(chm_real) DTKYX(MXSLT),DTKYY(MXSLT),DTKYZ(MXSLT)
  real(chm_real) DTKZX(MXSLT),DTKZY(MXSLT),DTKZZ(MXSLT)
  real(chm_real) DTJYX(MXSLT),DTJYY(MXSLT),DTJYZ(MXSLT)
  real(chm_real) DTJZX(MXSLT),DTJZY(MXSLT),DTJZZ(MXSLT)
  ! 2ND HELIX
  real(chm_real) DBJINTX(MXSLT),DBJINTY(MXSLT),DBJINTZ(MXSLT)
  real(chm_real) DEJINTX(MXSLT),DEJINTY(MXSLT),DEJINTZ(MXSLT)
  real(chm_real) DBJX(MXSLT),DBJY(MXSLT),DBJZ(MXSLT)
  real(chm_real) DBJYX(MXSLT),DBJYY(MXSLT),DBJYZ(MXSLT)
  real(chm_real) DBJZX(MXSLT),DBJZY(MXSLT),DBJZZ(MXSLT)
  real(chm_real) DEJX(MXSLT),DEJY(MXSLT),DEJZ(MXSLT)
  real(chm_real) DEJYX(MXSLT),DEJYY(MXSLT),DEJYZ(MXSLT)
  real(chm_real) DEJZX(MXSLT),DEJZY(MXSLT),DEJZZ(MXSLT)
  real(chm_real) DPW1XJ(MXSLT),DPW1YJ(MXSLT),DPW1ZJ(MXSLT)
  real(chm_real) DPW2XJ(MXSLT),DPW2YJ(MXSLT),DPW2ZJ(MXSLT)
  real(chm_real) DDU22XJ(MXSLT),DDU22YJ(MXSLT),DDU22ZJ(MXSLT)
  real(chm_real) DDU12XJ(MXSLT),DDU12YJ(MXSLT),DDU12ZJ(MXSLT)
  real(chm_real) DDETXJ(MXSLT),DDETYJ(MXSLT),DDETZJ(MXSLT)
  real(chm_real) DSKXJ(MXSLT),DSKYJ(MXSLT),DSKZJ(MXSLT)
  real(chm_real) DSJXJ(MXSLT),DSJYJ(MXSLT),DSJZJ(MXSLT)
  real(chm_real) DTJXJ(MXSLT),DTJYJ(MXSLT),DTJZJ(MXSLT)
  real(chm_real) DTJYJX(MXSLT),DTJYJY(MXSLT),DTJYJZ(MXSLT)
  real(chm_real) DTJZJX(MXSLT),DTJZJY(MXSLT),DTJZJZ(MXSLT)
  real(chm_real) DTKXJ(MXSLT),DTKYJ(MXSLT),DTKZJ(MXSLT)
  real(chm_real) DTKYJX(MXSLT),DTKYJY(MXSLT),DTKYJZ(MXSLT)
  real(chm_real) DTKZJX(MXSLT),DTKZJY(MXSLT),DTKZJZ(MXSLT)
  real(chm_real) DUJX(MXSLT),DUJY(MXSLT),DUJZ(MXSLT)
  real(chm_real) PA,PB
  real(chm_real) LSK0SJ,LSK1SJ,LSJ0SK,LSJ1SK
  real(chm_real) DAX(MXSLT),DAY(MXSLT),DAZ(MXSLT)
  real(chm_real) DBX(MXSLT),DBY(MXSLT),DBZ(MXSLT)
  real(chm_real) DAXJ(MXSLT),DAYJ(MXSLT),DAZJ(MXSLT)
  real(chm_real) DBXJ(MXSLT),DBYJ(MXSLT),DBZJ(MXSLT)

  LOGICAL LLIMIT,LNLIM,LQSTPRT
  LOGICAL LSK0,LSK1,LSJ0,LSJ1
  LOGICAL LSK0SJ0,LSK0SJ1,LSK1SJ0,LSK1SJ1

  INTEGER MCOUNT
  INTEGER CHUNIT,CHSTEP

  real(chm_real) CPRVEC(NCOOR,MXSLT,MXHEL)
  real(chm_real) SUMX,SUMY,SUMZ,SUM

  LSK0=(CSVAL(1) == 0.D0)
  LSK1=(CSVAL(1) == 1.D0)
  LSJ0=(CSVAL(2) == 0.D0)
  LSJ1=(CSVAL(2) == 1.D0)

  IF((.NOT.LSK0).AND.(.NOT.LSK1).AND.(.NOT.LSJ0).AND.(.NOT.LSJ1)) THEN 
     LNLIM=.TRUE.
  ELSE
     LNLIM=.FALSE.
  ENDIF

  LSK0SJ0=((CSVAL(1) == 0.D0).AND.(CSVAL(2) == 0.D0))
  LSK0SJ1=((CSVAL(1) == 0.D0).AND.(CSVAL(2) == 1.D0))
  LSK1SJ0=((CSVAL(1) == 1.D0).AND.(CSVAL(2) == 0.D0))
  LSK1SJ1=((CSVAL(1) == 1.D0).AND.(CSVAL(2) == 1.D0))
  IF(LSK0SJ0.OR.LSK0SJ1.OR.LSK1SJ0.OR.LSK1SJ1)THEN
     LSK0=.FALSE.
     LSK1=.FALSE.
     LSJ0=.FALSE.
     LSJ1=.FALSE.
  ENDIF
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'LSK0',LSK0,LSK1,LSJ0,LSJ1,LSK0SJ0,LSK0SJ1,LSK1SJ0,LSK1SJ1,LNLIM
  ENDIF
  ! u = k ( t - t_0 )^2
  ! cos(t) = (c1 (x) c2) (dot) (c2 (x) c3) / (|c1(x)c2| |c2(x)c3|)
  !        = c1xc2 (dot) c2xc3 / ( |c1xc2| |c2xc3| )
  K=1
  J=K+1
  TMP=0.D0
  TMP2=0.D0
  TMP3=0.D0
  DO I=1,3
     ! c1 = a_k
     C1(I)=CAVEC(I,K)
     ! c2 = ( t_j - t_k ) / | t_j - t_k | = b
     C2(I)=CTVEC(I,J)-CTVEC(I,K)
     TMP2=TMP2+C2(I)*C2(I)
     ! c3 = - a_j = c
     C3(I)=-CAVEC(I,J)
  ENDDO
  ! |t_j - t_k|=tjmtk
  TJMTK=SQRT(TMP2)
  DO I=1,3
     C2(I)=C2(I)/SQRT(TMP2)
  ENDDO
  ! cross prod
  ! c1xc2 = ( a_k (x) b ) = ( c1 (x) c2 ) 
  CALL CROSS3(C1,C2,C1XC2)
  ! c2xc3 = ( b (x) c ) = ( c2 (x) c3 )
  CALL CROSS3(C2,C3,C2XC3)
  ! to get the symbol
  ! orie =
  ! ( a_k (X) b ) (X) ( b (X) c )
  ! ( c1 (X) c2 ) (X) ( c2 (X) c3 )
  ! c1xc2 (X) c2xc3
  CALL CROSS3(C1XC2,C2XC3,ORIE)
  ! dotporie = ( orie (dot) c2 )
  CALL DOTPR(C2,ORIE,3,DOTPORIE)
  ! dotp
  ! =       c1xc2 (dot) c2xc3
  ! = (a_k (X) b) (dot) (b (X) c)
  CALL DOTPR(C1XC2,C2XC3,3,DOTP)
  ! akxb
  ! = |a_k (X) b| = |c1xc2|
  ! bxc
  ! = |b (X) c| = |c2xc3|
  AKXB=0.D0
  BXC=0.D0
  DO I=1,3
     AKXB=AKXB+C1XC2(I)*C1XC2(I)
     BXC=BXC+C2XC3(I)*C2XC3(I)
  ENDDO
  AKXB=SQRT(AKXB)
  BXC=SQRT(BXC)

  COST=DOTP/(AKXB*BXC)
  THETA=ACOS(COST)*RADDEG
  IF(DOTPORIE.LT.ZERO) THETA=-THETA
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'COST',COST,THETA
  ! WRITE DOWN TO FILE
  IF((PRNLEV >= 6).AND.DYNAMQ.AND.(CHSTEP > 0)) THEN
     WRITE(OUTU,*) 'HAHA',CHUNIT,CHSTEP,MDSTEP,MOD(MDSTEP,CHSTEP)
  ENDIF

  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1
  IF(PRNLEV>6) THEN
     WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,THETA,LQSTPRT,MCOUNT
     WRITE(CHUNIT,*) MDSTEP+1,THETA, &
          LSK0,LSK1,LSJ0,LSJ1,LSK0SJ0,LSK0SJ1,LSK1SJ0,LSK1SJ1,LNLIM
  ENDIF

  IF(CHSTEP>0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') THETA
     ENDIF
  ENDIF
  ! (u_kx)' = 2 k ( t - t_0 ) / - sin(t) (cos(t))'
  ! preu = - 2 k ( t - t_0 ) / sin(t)
  ! due to singlar point near 0.d0 and pi, ignore force near that
  ! 2011/11/04
  IF((ABS(THETA) <= 1.0D-4).OR.(ABS(THETA) >= (180.D0-1.0D-4))) THEN
     PREU=0.D0
  ELSE
     PREU=-2.D0*AFOR*(THETA-ANGL)*DEGRAD/SIN(THETA*DEGRAD)
  ENDIF
  !
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'CSVAL',CSVAL(1),CSVAL(2)
  !----------------------------------------------------------------------
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'EIGENVECTOR_PRE',O(1,1),O(2,1),O(3,1)
     WRITE(OUTU,*) 'EIGENVECTOR_PRE2',CAVEC(1,1),CAVEC(2,1),CAVEC(3,1)
     WRITE(OUTU,*) 'EIGENVALUE',EV(1),EV(2),EV(3)
  ENDIF
  !     
  CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ! k = number of helix ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'EIGENVECTOR',O(1,1),O(2,1),O(3,1)
     WRITE(OUTU,*) '    FIRST ATOMS'
     WRITE(OUTU,*) '    CDXO      CDYO      CDZO'
     DO J=1,3
        WRITE(OUTU,'(3F10.5)') CDXO(K,1,J),CDYO(K,1,J),CDZO(K,1,J)
     ENDDO
  ENDIF
  ! common value
  J=k+1
  ! first and last atom
  FSATOM=ASLCT(1)
  LSATOM=ASLCT(NSEL(K))
  ONUM=1.D0/NSEL(K)
  ! center of geometry: bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)

  IF(PRNLEV >= 6) WRITE(OUTU,*) 'COG',XCG,YCG,ZCG
  !----------------------------------------------------------------------
  ! Constant value
  ! dot product
  ! a_k (dot) ( r_1k - bar(r_k) )
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  DOTK=O(1,1)*(X(FSATOM)-XCG)+O(2,1)*(Y(FSATOM)-YCG)+O(3,1)*(Z(FSATOM)-ZCG)
  ! a_k (dot) ( r_mk - bar(r_k) )
  DOTKM=O(1,1)*(X(LSATOM)-XCG)+O(2,1)*(Y(LSATOM)-YCG) + O(3,1)*(Z(LSATOM)-ZCG)
  ! Det = U_11 * U_22 - U_12 ^2
  DU11=0.D0
  DU22=0.D0
  DU12=0.D0
  DO I=1,3
     ! absolute value |e_k - b_k|^2 and |e_j - b_j|^2
     DU11=DU11+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,K)-CBVEC(I,K))
     DU22=DU22+(CEVEC(I,J)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
     ! inner product (e_k - b_k) (dot) (e_j - b_j)
     DU12=DU12+(CEVEC(I,K)-CBVEC(I,K))*(CEVEC(I,J)-CBVEC(I,J))
  ENDDO
  Det=DU11*DU22-DU12*DU12
  ! PreSk = - W_1 * U_22 + W_2 * U_12
  PW1=0.d0
  PW2=0.d0
  ! inner product PW1 = (b_k - b_j) (dot) (e_k - b_k)
  !               PW2 = (b_k - b_j) (dot) (e_j - b_j)
  DO I=1,3
     PW1=PW1+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,K)-CBVEC(I,K))
     PW2=PW2+(CBVEC(I,K)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
  ENDDO
  PRESK=-PW1*DU22+PW2*DU12
  ! PreSj = W_2 * U_11 - W_1 * U_12
  PRESJ=PW2*DU11-PW1*DU12
  ! limit S_k = 0
  IF(LLIMIT.AND.LSK0) LSK0SJ=PW2/DU22
  ! limit S_k = 1
  IF(LLIMIT.AND.LSK1) THEN
     PB=0.D0
     DO I=1,3
        PB=PB+(CEVEC(I,K)-CBVEC(I,J))*(CEVEC(I,J)-CBVEC(I,J))
     ENDDO
     LSK1SJ=PB/DU22
  ENDIF
  ! limit S_j = 0
  IF(LLIMIT.AND.LSJ0) THEN
     LSJ0SK=-PW1/DU11
  ENDIF
  ! limit S_j = 1
  IF(LLIMIT.AND.LSJ1) THEN
     PA=0.D0
     DO I=1,3
        PA=PA+(CEVEC(I,J)-CBVEC(I,K))*(CEVEC(I,K)-CBVEC(I,K))
     ENDDO
     LSJ1SK=PA/DU11
  ENDIF
  !----------------------------------------------------------------------
  ! Main loop 
  DO I=1,NSEL(K)
     ! (r_1kx)'
     ! i=1       --> 1
     ! otherwise --> 0
     IF(I == 1) THEN 
        R1K=1.D0
     ELSE 
        R1K=0.D0
     ENDIF
     ! (r_mkx)'
     ! i=m       --> 1
     ! otherwise --> 0
     IF(I == NSEL(K)) THEN 
        RMK=1.D0
     ELSE 
        RMK=0.D0
     ENDIF

     NATOMM=ASLCT(I)
     ! (a_k (dot) (r_1k - bar(r_k)))'
     ! dbkintx by x
     ! dbkintx = (a_kx)' (r_1kx - xcg) + a_kx ( (r_1kx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_1ky - ycg) + (a_kz)' ( r_1kz - zcg)
     DBKINTX(I)=CDXO(K,I,1)*(X(FSATOM)-XCG)+O(1,1)*(R1K-ONUM) &
          +CDXO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTY(I)=CDYO(K,I,1)*(X(FSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(FSATOM)-YCG)+O(2,1)*(R1K-ONUM) &
          +CDYO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTZ(I)=CDZO(K,I,1)*(X(FSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(FSATOM)-ZCG)+O(3,1)*(R1K-ONUM)
     ! (a_k (dot) (r_mk - bar(r_k)))'
     ! dekintx
     ! dekintx = (a_kx)' (r_mkx - xcg) + a_kx ( (r_mkx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_mky - ycg) + (a_kz)' ( r_mkz - zcg)
     DEKINTX(I)=CDXO(K,I,1)*(X(LSATOM)-XCG)+O(1,1)*(RMK-ONUM ) &
          +CDXO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTY(I)=CDYO(K,I,1)*(X(LSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(LSATOM)-YCG)+O(2,1)*(RMK-ONUM) &
          +CDYO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTZ(I)=CDZO(K,I,1)*(X(LSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(LSATOM)-ZCG)+O(3,1)*(RMK-ONUM)
     ! dbk 1 
     ! (b_kxx)' == (b_kx)'
     ! (b_kx)' = 1 / nsel(k) + ( a_k (dot) (r_1k - bar(r_k)))' a_kx
     !         + ( a_k (dot) (r_1k - bar(r_k))) (a_kx)'
     DBKX(I)=DBKINTX(I)*O(1,1)+DOTK*CDXO(K,I,1)+ONUM
     DBKY(I)=DBKINTY(I)*O(1,1)+DOTK*CDYO(K,I,1)
     DBKZ(I)=DBKINTZ(I)*O(1,1)+DOTK*CDZO(K,I,1)
     ! b_ky by x
     ! (b_kyx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_ky
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kyx)'
     DBKYX(I)=DBKINTX(I)*O(2,1)+DOTK*CDXO(K,I,2)
     DBKYY(I)=DBKINTY(I)*O(2,1)+DOTK*CDYO(K,I,2)+ONUM
     DBKYZ(I)=DBKINTZ(I)*O(2,1)+DOTK*CDZO(K,I,2)
     ! b_kz by x 
     ! (b_kzx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_kz
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kzx)'
     DBKZX(I)=DBKINTX(I)*O(3,1)+DOTK*CDXO(K,I,3)
     DBKZY(I)=DBKINTY(I)*O(3,1)+DOTK*CDYO(K,I,3)
     DBKZZ(I)=DBKINTZ(I)*O(3,1)+DOTK*CDZO(K,I,3)+ONUM
     ! dek
     ! (e_kxx)' == (e_kx)'
     ! (e_kx)' = 1 / nsel(k) + (a_k (dot) ( r_mk - bar(r_k)))' a_kx
     !                       + (a_k (dot) ( r_mk - bar(r_k))) (a_kx)'
     DEKX(I)=DEKINTX(I)*O(1,1)+DOTKM*CDXO(K,I,1)+ONUM
     DEKY(I)=DEKINTY(I)*O(1,1)+DOTKM*CDYO(K,I,1)
     DEKZ(I)=DEKINTZ(I)*O(1,1)+DOTKM*CDZO(K,I,1)
     ! e_ky by x
     ! (e_kyx)' = ( a_k (dot) (r_mk - bar(r_k)))' a_ky
     !          + ( a_k (dot) (r_mk - bar(r_k))) (a_kyx)'
     DEKYX(I)=DEKINTX(I)*O(2,1)+DOTKM*CDXO(K,I,2)
     DEKYY(I)=DEKINTY(I)*O(2,1)+DOTKM*CDYO(K,I,2)+ONUM
     DEKYZ(I)=DEKINTZ(I)*O(2,1)+DOTKM*CDZO(K,I,2)
     ! e_kz by x
     ! (e_kzx)' = ( a_k (dot) (r_mk - bar(r_k)))' a_kz
     !          + ( a_k (dot) (r_mk - bar(r_k))) (a_kzx)'
     DEKZX(I)=DEKINTX(I)*O(3,1)+DOTKM*CDXO(K,I,3)
     DEKZY(I)=DEKINTY(I)*O(3,1)+DOTKM*CDYO(K,I,3)
     DEKZZ(I)=DEKINTZ(I)*O(3,1)+DOTKM*CDZO(K,I,3)+ONUM
     ! (PW1)' 
     ! (b_k - b_j) (dot) (e_k - b_k)
     !  = (b_kx)'  (e_kx - b_kx) + (b_kx - b_jx) ((e_kx)' -  (b_kx)')
     !  + (b_kyx)' (e_ky - b_ky) + (b_ky - b_jy) ((e_kyx)'- (b_kyx)')
     !  + (bkzx)'  (e_kz - b_kz) + (b_kz - b_jz) ((e_kzx)'- (b_kzx)')
     !----------------------------------------------------------------------
     DPW1X(I)=DBKX(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKX(I)-DBKX(I)) &
          +DBKYX(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYX(I)-DBKYX(I)) &
          +DBKZX(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZX(I)-DBKZX(I))
     DPW1Y(I)=DBKY(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKY(I)-DBKY(I)) &
          +DBKYY(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYY(I)-DBKYY(I)) &
          +DBKZY(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZY(I)-DBKZY(I))
     DPW1Z(I)=DBKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+(CBVEC(1,K)-CBVEC(1,J))*(DEKZ(I)-DBKZ(I)) &
          +DBKYZ(I)*(CEVEC(2,K)-CBVEC(2,K))+(CBVEC(2,K)-CBVEC(2,J))*(DEKYZ(I)-DBKYZ(I)) &
          +DBKZZ(I)*(CEVEC(3,K)-CBVEC(3,K))+(CBVEC(3,K)-CBVEC(3,J))*(DEKZZ(I)-DBKZZ(I))
     ! (du22)' by k (x,y,z) = 0
     ! (pw2)'
     ! (b_k-b_j) (dot) (e_j-b_j)
     ! (b_kx)' (e_jx - b_jx) + (b_kyx)' (e_jy - b_jy) 
     !                       + (b_kzx)' (e_jz - b_jz)
     DPW2X(I)=DBKX(I)*(CEVEC(1,J)-CBVEC(1,J)) &
          +DBKYX(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZX(I)*(CEVEC(3,J)-CBVEC(3,J))
     DPW2Y(I)=DBKY(I)*(CEVEC(1,J)-CBVEC(1,J)) &
          +DBKYY(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZY(I)*(CEVEC(3,J)-CBVEC(3,J))
     DPW2Z(I)=DBKZ(I)*(CEVEC(1,J)-CBVEC(1,J)) &
          +DBKYZ(I)*(CEVEC(2,J)-CBVEC(2,J)) &
          +DBKZZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ! (du12)'
     ! (e_k - b_k) (dot) (e_j - b_j)
     ! ((e_kx)' -(b_kx)')(e_jx - b_jx) + ((e_kyx)' - (b_kyx)')(e_jy-b_jy)
     !                                 + ((e_kzx)' - (b_kzx)')(e_jz-b_jz)
     DDU12X(I)=(DEKX(I)-DBKX(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYX(I)-DBKYX(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZX(I)-DBKZX(I))*(CEVEC(3,J)-CBVEC(3,J))
     DDU12Y(I)=(DEKY(I)-DBKY(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYY(I)-DBKYY(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZY(I)-DBKZY(I))*(CEVEC(3,J)-CBVEC(3,J))
     DDU12Z(I)=(DEKZ(I)-DBKZ(I))*(CEVEC(1,J)-CBVEC(1,J)) &
          +(DEKYZ(I)-DBKYZ(I))*(CEVEC(2,J)-CBVEC(2,J)) &
          +(DEKZZ(I)-DBKZZ(I))*(CEVEC(3,J)-CBVEC(3,J))
     ! (du11)'
     ! |e_k-b_k|^2
     ! 2 (e_kx - b_kx) ((e_kx)' - (b_kx)') + 2 (e_ky - b_ky) ((e_kyx)'-(b_kyx)')
     ! + 2 (e_kz - b_kz) ( (e_kzx)' - (b_kzx)' )
     DDU11X(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKX(I)-DBKX(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYX(I)-DBKYX(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZX(I)-DBKZX(I))
     DDU11Y(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKY(I)-DBKY(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYY(I)-DBKYY(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZY(I)-DBKZY(I))
     DDU11Z(I)=2.D0*(CEVEC(1,K)-CBVEC(1,K))*(DEKZ(I)-DBKZ(I)) &
          +2.D0*(CEVEC(2,K)-CBVEC(2,K))*(DEKYZ(I)-DBKYZ(I)) &
          +2.D0*(CEVEC(3,K)-CBVEC(3,K))*(DEKZZ(I)-DBKZZ(I))
     ! (Det)' by k
     ! (Det)' = (DU11)' DU22 + (DU22)' DU11 - 2 DU12 (DU12)'
     ! (DU22)' = 0
     DDETXK(I)=DDU11X(I)*DU22-2.D0*DU12*DDU12X(I)
     DDETYK(I)=DDU11Y(I)*DU22-2.D0*DU12*DDU12Y(I)
     DDETZK(I)=DDU11Z(I)*DU22-2.D0*DU12*DDU12Z(I)
     ! limit S_k = 1
     ! B=(e_k-b_j) (dot) (e_j-b_j)
     IF(LLIMIT.AND.LSK1) THEN
        DBX(I)=DEKX(I)*(CEVEC(1,J)-CBVEC(1,J)) &
             +DEKYX(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZX(I)*(CEVEC(3,J)-CBVEC(3,J))
        DBY(I)=DEKY(I)*(CEVEC(1,J)-CBVEC(1,J)) &
             +DEKYY(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZY(I)*(CEVEC(3,J)-CBVEC(3,J))
        DBZ(I)=DEKZ(I)*(CEVEC(1,J)-CBVEC(1,J)) &
             +DEKYZ(I)*(CEVEC(2,J)-CBVEC(2,J)) &
             +DEKZZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ENDIF
     ! limit s_j = 1
     ! a=(b_k-e_j) (dot) (e_k-b_k)
     ! (a)' 
     IF(LLIMIT.AND.LSJ1) THEN
        DAX(I)=-DBKX(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKX(I)-DBKX(I)) &
             -DBKYX(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYX(I)-DBKYX(I)) &
             -DBKZX(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZX(I)-DBKZX(I))
        DAY(I)=-DBKY(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKY(I)-DBKY(I)) &
             -DBKYY(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYY(I)-DBKYY(I)) &
             -DBKZY(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZY(I)-DBKZY(I))
        DAZ(I)=-DBKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+(CEVEC(1,J)-CBVEC(1,K))*(DEKZ(I)-DBKZ(I)) &
             -DBKYZ(I)*(CEVEC(2,K)-CBVEC(2,K))+(CEVEC(2,J)-CBVEC(2,K))*(DEKYZ(I)-DBKYZ(I)) &
             -DBKZZ(I)*(CEVEC(3,K)-CBVEC(3,K))+(CEVEC(3,J)-CBVEC(3,K))*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! (S_k)' by k
     ! ( - (pw1)' du22 - pw1 (du22)' + (pw2)' du12 + pw2 (du12)' ) / Det
     ! + ( - pw1 du22 + pw2 du12 ) ( -1 * Det ^ -2 (Det)')
     ! (du22)' = 0 
     DSKX(I)=(-DPW1X(I)*DU22+DPW2X(I)*DU12+PW2*DDU12X(I))/DET+PRESK*(-1.D0/(DET*DET)*DDETXK(I))
     DSKY(I)=(-DPW1Y(I)*DU22+DPW2Y(I)*DU12+PW2*DDU12Y(I))/DET+PRESK*(-1.D0/(DET*DET)*DDETYK(I))
     DSKZ(I)=(-DPW1Z(I)*DU22+DPW2Z(I)*DU12+PW2*DDU12Z(I))/DET+PRESK*(-1.D0/(DET*DET)*DDETZK(I))
     ! limit s_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DSKX(I)=-DPW1X(I)/DU11+PW1/(DU11*DU11)*DDU11X(I)
        DSKY(I)=-DPW1Y(I)/DU11+PW1/(DU11*DU11)*DDU11Y(I)
        DSKZ(I)=-DPW1Z(I)/DU11+PW1/(DU11*DU11)*DDU11Z(I)
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DSKX(I)=DAX(I)/DU11-PA/(DU11*DU11)*DDU11X(I)
        DSKY(I)=DAY(I)/DU11-PA/(DU11*DU11)*DDU11Y(I)
        DSKZ(I)=DAZ(I)/DU11-PA/(DU11*DU11)*DDU11Z(I)
     ENDIF
     ! (S_j)' by k
     ! ( (pw2)' du11 + (du11)' pw2 - (pw1)' du12 - (du12)' pw1 ) / det
     !               + (pw2 du11 - pw1 du12) ( -1 Det^-2 (Det)')
     DSJX(I)=(DPW2X(I)*DU11+DDU11X(I)*PW2-DPW1X(I)*DU12 &
          -DDU12X(I)*PW1)/DET+PRESJ*(-1.D0/(DET*DET)*DDETXK(I))
     DSJY(I)=(DPW2Y(I)*DU11+DDU11Y(I)*PW2-DPW1Y(I)*DU12 &
          -DDU12Y(I)*PW1)/DET+PRESJ*(-1.D0/(DET*DET)*DDETYK(I))
     DSJZ(I)=(DPW2Z(I)*DU11+DDU11Z(I)*PW2-DPW1Z(I)*DU12 &
          -DDU12Z(I)*PW1)/DET+PRESJ*(-1.D0/(DET*DET)*DDETZK(I))
     ! limit s_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DSJX(I)=DPW2X(I)/DU22
        DSJY(I)=DPW2Y(I)/DU22
        DSJZ(I)=DPW2Z(I)/DU22
     ENDIF
     ! limit s_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DSJX(I)=DBX(I)/DU22
        DSJY(I)=DBY(I)/DU22
        DSJZ(I)=DBZ(I)/DU22
     ENDIF
     ! (t_kxx)'=(t_kx)'
     ! (t_kx)' = (b_kx)' + (S_k)' ( e_kx - b_kx ) 
     !                   + S_k ( (e_kx)' - (b_kx)' )
     DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKX(I)-DBKX(I))
     DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKY(I)-DBKY(I))
     DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+PRESK/DET*(DEKZ(I)-DBKZ(I))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKX(I) = DBKX(I)
        DTKY(I) = DBKY(I)
        DTKZ(I) = DBKZ(I)
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKX(I) = DEKX(I)
        DTKY(I) = DEKY(I)
        DTKZ(I) = DEKZ(I)
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKX(I)-DBKX(I))
        DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKY(I)-DBKY(I))
        DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))-PW1/DU11*(DEKZ(I)-DBKZ(I))
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKX(I)=DBKX(I)+DSKX(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKX(I)-DBKX(I))
        DTKY(I)=DBKY(I)+DSKY(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKY(I)-DBKY(I))
        DTKZ(I)=DBKZ(I)+DSKZ(I)*(CEVEC(1,K)-CBVEC(1,K))+PA/DU11*(DEKZ(I)-DBKZ(I))
     ENDIF
     ! (t_kyx)' = (b_ky)' + (S_k)' (e_ky-b_ky) 
     !              + S_k ( (e_ky)' - (b_ky)' )
     DTKYX(I)=DBKYX(I)+DSKX(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYX(I)-DBKYX(I))      
     DTKYY(I)=DBKYY(I)+DSKY(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYY(I)-DBKYY(I))      
     DTKYZ(I)=DBKYZ(I)+DSKZ(I)*(CEVEC(2,K)-CBVEC(2,K))+PRESK/DET*(DEKYZ(I)-DBKYZ(I))      
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 )
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKYX(I)=DBKYX(I)
        DTKYY(I)=DBKYY(I)
        DTKYZ(I)=DBKYZ(I)
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 )
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKYX(I)=DEKYX(I)
        DTKYY(I)=DEKYY(I)
        DTKYZ(I)=DEKYZ(I)
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKYX(I)=DBKYX(I)+DSKX(I)* (CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYX(I)-DBKYX(I))      
        DTKYY(I)=DBKYY(I)+DSKY(I)* (CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYY(I)-DBKYY(I))      
        DTKYZ(I)=DBKYZ(I)+DSKZ(I)* (CEVEC(2,K)-CBVEC(2,K))-PW1/DU11*(DEKYZ(I)-DBKYZ(I))      
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKYX(I)=DBKYX(I)+DSKX(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYX(I)-DBKYX(I))      
        DTKYY(I)=DBKYY(I)+DSKY(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYY(I)-DBKYY(I))      
        DTKYZ(I)=DBKYZ(I)+DSKZ(I)*(CEVEC(2,K)-CBVEC(2,K))+PA/DU11*(DEKYZ(I)-DBKYZ(I))      
     ENDIF
     ! (t_kzx)' = (b_kz)' + (S_k)' (e_kz-b_kz) + S_k ( (e_kz)' - (b_kz)' )
     DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZX(I)-DBKZX(I))
     DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZY(I)-DBKZY(I))
     DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))+PRESK/DET*(DEKZZ(I)-DBKZZ(I))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 )
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKZX(I)=DBKZX(I)
        DTKZY(I)=DBKZY(I)
        DTKZZ(I)=DBKZZ(I)         
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKZX(I)=DEKZX(I)
        DTKZY(I)=DEKZY(I)
        DTKZZ(I)=DEKZZ(I)         
     ENDIF
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZX(I)-DBKZX(I))
        DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZY(I)-DBKZY(I))
        DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))-PW1/DU11*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! limit S_j = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DTKZX(I)=DBKZX(I)+DSKX(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZX(I)-DBKZX(I))
        DTKZY(I)=DBKZY(I)+DSKY(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZY(I)-DBKZY(I))
        DTKZZ(I)=DBKZZ(I)+DSKZ(I)*(CEVEC(3,K)-CBVEC(3,K))+PA/DU11*(DEKZZ(I)-DBKZZ(I))
     ENDIF
     ! (t_jx)' by k
     ! (t_jx)' = (S_j)' (e_jx - b_jx)
     DTJX(I)=DSJX(I)*(CEVEC(1,J)-CBVEC(1,J))
     DTJY(I)=DSJY(I)*(CEVEC(1,J)-CBVEC(1,J))
     DTJZ(I)=DSJZ(I)*(CEVEC(1,J)-CBVEC(1,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJX(I)=DSJX(I)*(CEVEC(1,J)-CBVEC(1,J))
        DTJY(I)=DSJY(I)*(CEVEC(1,J)-CBVEC(1,J))
        DTJZ(I)=DSJZ(I)*(CEVEC(1,J)-CBVEC(1,J))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 )
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJX(I)=0.D0
        DTJY(I)=0.D0
        DTJZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJX(I)=0.D0
        DTJY(I)=0.D0
        DTJZ(I)=0.D0
     ENDIF
     ! (t_jy)' by k
     ! (t_jy)' = (S_j)' (e_jy -b_jy)
     DTJYX(I)=DSJX(I)*(CEVEC(2,J)-CBVEC(2,J))
     DTJYY(I)=DSJY(I)*(CEVEC(2,J)-CBVEC(2,J))
     DTJYZ(I)=DSJZ(I)*(CEVEC(2,J)-CBVEC(2,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJYX(I)=DSJX(I)*(CEVEC(2,J)-CBVEC(2,J))
        DTJYY(I)=DSJY(I)*(CEVEC(2,J)-CBVEC(2,J))
        DTJYZ(I)=DSJZ(I)*(CEVEC(2,J)-CBVEC(2,J))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJYX(I)=0.D0
        DTJYY(I)=0.D0
        DTJYZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 )
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJYX(I)=0.D0
        DTJYY(I)=0.D0
        DTJYZ(I)=0.D0
     ENDIF
     ! (t_jz)' by k
     ! (t_jz)' = (S_j)' (e_jz - b_jz)
     DTJZX(I)=DSJX(I)*(CEVEC(3,J)-CBVEC(3,J))
     DTJZY(I)=DSJY(I)*(CEVEC(3,J)-CBVEC(3,J))
     DTJZZ(I)=DSJZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJZX(I)=DSJX(I)*(CEVEC(3,J)-CBVEC(3,J))
        DTJZY(I)=DSJY(I)*(CEVEC(3,J)-CBVEC(3,J))
        DTJZZ(I)=DSJZ(I)*(CEVEC(3,J)-CBVEC(3,J))
     endif
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 )
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJZX(I)=0.D0
        DTJZY(I)=0.D0
        DTJZZ(I)=0.D0
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJZX(I)=0.D0
        DTJZY(I)=0.D0
        DTJZZ(I)=0.D0
     ENDIF
     ! |t_j - t_k|
     ! = sqrt( (t_jx-t_kx)^2 + (t_jy-t_ky)^2 + (t_jz-t_kz)^2 )
     ! = tjmtk
     ! (tjmtk)'= dtjmtkx by x
     DTJMTKX(I)=1.D0/(TWO*TJMTK) &
          *(TWO*(CTVEC(1,J)-CTVEC(1,K))*(DTJX(I)-DTKX(I)) &
          +TWO*(CTVEC(2,J)-CTVEC(2,K))*(DTJYX(I)-DTKYX(I)) &
          +TWO*(CTVEC(3,J)-CTVEC(3,K))*(DTJZX(I)-DTKZX(I)))
     DTJMTKY(I)=1.D0/(TWO*TJMTK) &
          *(TWO*(CTVEC(1,J)-CTVEC(1,K))*(DTJY(I)-DTKY(I)) &
          +TWO*(CTVEC(2,J)-CTVEC(2,K))*(DTJYY(I)-DTKYY(I)) &
          +TWO*(CTVEC(3,J)-CTVEC(3,K))*(DTJZY(I)-DTKZY(I)))
     DTJMTKZ(I)=1.D0/(TWO*TJMTK) &
          *(TWO*(CTVEC(1,J)-CTVEC(1,K))*(DTJZ(I)-DTKZ(I)) &
          +TWO*(CTVEC(2,J)-CTVEC(2,K))*(DTJYZ(I)-DTKYZ(I)) &
          +TWO*(CTVEC(3,J)-CTVEC(3,K))*(DTJZZ(I)-DTKZZ(I)))
     ! B_x by x = (dtbx)'
     ! b_x = (t_jx - t_kx) / |t_j - t_k|
     ! (b_x)'= ((t_jx)' - (t_kx)') / |t_j - t_k| - (t_jx - t_kx) / (|t_j-t_k|^2) (|t_j-t_k|)'
     DTBX(I)=(DTJX(I)-DTKX(I))/TJMTK-(CTVEC(1,J)-CTVEC(1,K))*DTJMTKX(I)/(TJMTK*TJMTK)
     DTBY(I)=(DTJY(I)-DTKY(I))/TJMTK-(CTVEC(1,J)-CTVEC(1,K))*DTJMTKY(I)/(TJMTK*TJMTK)
     DTBZ(I)=(DTJZ(I)-DTKZ(I))/TJMTK-(CTVEC(1,J)-CTVEC(1,K))*DTJMTKZ(I)/(TJMTK*TJMTK)
     ! b_y by x = (dtbyx)'
     DTBYX(I)=(DTJYX(I)-DTKYX(I))/TJMTK-(CTVEC(2,J)-CTVEC(2,K))*DTJMTKX(I)/(TJMTK*TJMTK)
     DTBYY(I)=(DTJYY(I)-DTKYY(I))/TJMTK-(CTVEC(2,J)-CTVEC(2,K))*DTJMTKY(I)/(TJMTK*TJMTK)
     DTBYZ(I)=(DTJYZ(I)-DTKYZ(I))/TJMTK-(CTVEC(2,J)-CTVEC(2,K))*DTJMTKZ(I)/(TJMTK*TJMTK)
     ! b_z by x = (dtbzx)'
     DTBZX(I)=(DTJZX(I)-DTKZX(I))/TJMTK-(CTVEC(3,J)-CTVEC(3,K))*DTJMTKX(I)/(TJMTK*TJMTK)
     DTBZY(I)=(DTJZY(I)-DTKZY(I))/TJMTK-(CTVEC(3,J)-CTVEC(3,K))*DTJMTKY(I)/(TJMTK*TJMTK)
     DTBZZ(I)=(DTJZZ(I)-DTKZZ(I))/TJMTK-(CTVEC(3,J)-CTVEC(3,K))*DTJMTKZ(I)/(TJMTK*TJMTK)
     ! (|a_k (X) b|)' by x = dakxbkx
     !----------------------------------------------------------------------
     DAKXBKX(I)=1.D0/(TWO*AKXB)*(TWO*(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(CDXO(K,I,2)*C2(3)+C1(2)*DTBZX(I)-CDXO(K,I,3)*C2(2)-C1(3)*DTBYX(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(CDXO(K,I,3)*C2(1)+C1(3)*DTBX(I) &
          -CDXO(K,I,1)*C2(3)-C1(1)*DTBZX(I))+TWO*(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(CDXO(K,I,1)*C2(2)+C1(1)*DTBYX(I)-CDXO(K,I,2)*C2(1)-C1(2)*DTBX(I)))
     DAKXBKY(I)=1.D0/(TWO*AKXB)*(TWO*(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(CDYO(K,I,2)*C2(3)+C1(2)*DTBZY(I)-CDYO(K,I,3)*C2(2)-C1(3)*DTBYY(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(CDYO(K,I,3)*C2(1)+C1(3)*DTBY(I) &
          -CDYO(K,I,1)*C2(3)-C1(1)*DTBZY(I))+TWO*(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(CDYO(K,I,1)*C2(2)+C1(1)*DTBYY(I)-CDYO(K,I,2)*C2(1)-C1(2)*DTBY(I)))
     DAKXBKZ(I)=1.D0/(TWO*AKXB)*(TWO*(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(CDZO(K,I,2)*C2(3)+C1(2)*DTBZZ(I)-CDZO(K,I,3)*C2(2)-C1(3)*DTBYZ(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(CDZO(K,I,3)*C2(1)+C1(3)*DTBZ(I) &
          -CDZO(K,I,1)*C2(3)-C1(1)*DTBZZ(I))+TWO*(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(CDZO(K,I,1)*C2(2)+C1(1)*DTBYZ(I)-CDZO(K,I,2)*C2(1)-C1(2)*DTBZ(I)))
     ! (|b (X) c|)' by x = dbxcx
     DBXCX(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(C3(3)*DTBYX(I)-C3(2)*DTBZX(I))+TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(C3(1)*DTBZX(I)-C3(3)*DTBX(I))+TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(C3(2)*DTBX(I)-C3(1)*DTBYX(I)))
     DBXCY(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(C3(3)*DTBYY(I)-C3(2)*DTBZY(I))+TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(C3(1)*DTBZY(I)-C3(3)*DTBY(I))+TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(C3(2)*DTBY(I)-C3(1)*DTBYY(I)))
     DBXCZ(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(C3(3)*DTBYZ(I)-C3(2)*DTBZZ(I))+TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(C3(1)*DTBZZ(I)-C3(3)*DTBZ(I))+TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(C3(2)*DTBZ(I)-C3(1)*DTBYZ(I)))
     ! (cos(theta))' by k = dcostx
     DCOSTX(I)=((CDXO(K,I,2)*C2(3)+C1(2)*DTBZX(I)-CDXO(K,I,3)*C2(2)-C1(3)*DTBYX(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(C3(3)*DTBYX(I)-C3(2)*DTBZX(I))+(CDXO(K,I,3)*C2(1)+C1(3)*DTBX(I) &
          -CDXO(K,I,1)*C2(3)-C1(1)*DTBZX(I))*(C3(1)*C2(3)-C3(3)*C2(1)) &
          +(C1(3)*C2(1)-C1(1)*C2(3))*(C3(1)*DTBZX(I)-C3(3)*DTBX(I)) &
          +(CDXO(K,I,1)*C2(2)+C1(1)*DTBYX(I)-CDXO(K,I,2)*C2(1)-C1(2)*DTBX(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(C3(2)*DTBX(I)-C3(1)*DTBYX(I)))/(AKXB*BXC) &
          -DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBKX(I)*BXC+AKXB*DBXCX(I))
     DCOSTY(I)=((CDYO(K,I,2)*C2(3)+C1(2)*DTBZY(I)-CDYO(K,I,3)*C2(2)-C1(3)*DTBYY(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(C3(3)*DTBYY(I)-C3(2)*DTBZY(I))+(CDYO(K,I,3)*C2(1)+C1(3)*DTBY(I) &
          -CDYO(K,I,1)*C2(3)-C1(1)*DTBZY(I))*(C3(1)*C2(3)-C3(3)*C2(1)) &
          +(C1(3)*C2(1)-C1(1)*C2(3))*(C3(1)*DTBZY(I)-C3(3)*DTBY(I)) &
          +(CDYO(K,I,1)*C2(2)+C1(1)*DTBYY(I)-CDYO(K,I,2)*C2(1)-C1(2)*DTBY(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(C3(2)*DTBY(I)-C3(1)*DTBYY(I)))/(AKXB*BXC) &
          -DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBKY(I)*BXC+AKXB*DBXCY(I))
     DCOSTZ(I)=((CDZO(K,I,2)*C2(3)+C1(2)*DTBZZ(I)-CDZO(K,I,3)*C2(2)-C1(3)*DTBYZ(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(C3(3)*DTBYZ(I)-C3(2)*DTBZZ(I))+(CDZO(K,I,3)*C2(1)+C1(3)*DTBZ(I) &
          -CDZO(K,I,1)*C2(3)-C1(1)*DTBZZ(I))*(C3(1)*C2(3)-C3(3)*C2(1)) &
          +(C1(3)*C2(1)-C1(1)*C2(3))*(C3(1)*DTBZZ(I)-C3(3)*DTBZ(I)) &
          +(CDZO(K,I,1)*C2(2)+C1(1)*DTBYZ(I)-CDZO(K,I,2)*C2(1)-C1(2)*DTBZ(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(C3(2)*DTBZ(I)-C3(1)*DTBYZ(I)))/(AKXB*BXC) &
          -DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBKZ(I)*BXC+AKXB*DBXCZ(I))
     ! (U_kx)' = 2 k (T - T_0) / - SIN(T) (COS(T))'
     ! = 2 k (T - T_0) / - SIN(T) dcostx(i)
     DUKX(I)=PREU*DCOSTX(I)
     DUKY(I)=PREU*DCOSTY(I)
     DUKZ(I)=PREU*DCOSTZ(I)
  ENDDO
  !----------------------------------------------------------------------
  ! SECOND HELIX
  K=2
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO

  CALL ROTINVAR2(K,X,Y,Z,NSEL,BSLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)

  J=K-1
  JK=1 
  JJ=2
  FSATOM=BSLCT(1)
  LSATOM=BSLCT(NSEL(K))
  ONUM=1.D0/NSEL(K)

  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)
  ! dot product
  ! a_j (dot) ( r_1j - bar(r_j) )
  ! a_jx -> o(1,1), a_jy -> o(2,1), a_jz -> o(3,1)
  DOTK=O(1,1)*(X(FSATOM)-XCG)+O(2,1)*(Y(FSATOM)-YCG)+O(3,1)*(Z(FSATOM)-ZCG)
  ! a_j (dot) ( r_mj - bar(r_j) )
  DOTKM=O(1,1)*(X(LSATOM)-XCG)+O(2,1)*(Y(LSATOM)-YCG)+O(3,1)*(Z(LSATOM)-ZCG)
  DO I=1,NSEL(K)
     ! (r_1jx)'
     ! i=1       --> 1
     ! otherwise --> 0
     IF(I == 1) THEN 
        R1J=1.D0
     ELSE 
        R1J=0.D0
     ENDIF
     ! (r_mjx)'
     ! i=m       --> 1
     ! otherwise --> 0
     IF(I == NSEL(K)) THEN 
        RMJ=1.D0
     ELSE 
        RMJ=0.D0
     ENDIF

     NATOMM=BSLCT(I)
     ! (a_j (dot) (r_1j - bar(r_j)))'
     DBJINTX(I)=CDXO(K,I,1)*(X(FSATOM)-XCG)+O(1,1)*(R1J-ONUM) &
          +CDXO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(FSATOM)-ZCG)
     DBJINTY(I)=CDYO(K,I,1)*(X(FSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(FSATOM)-YCG)+O(2,1)*(R1J-ONUM) &
          +CDYO(K,I,3)*(Z(FSATOM)-ZCG)
     DBJINTZ(I)=CDZO(K,I,1)*(X(FSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(FSATOM)-ZCG)+O(3,1)*(R1J-ONUM)
     ! (A_j (dot) (r_mj - bar(r_j)))'
     DEJINTX(I)=CDXO(K,I,1)*(X(LSATOM)-XCG)+O(1,1)*(RMJ-ONUM) &
          +CDXO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(LSATOM)-ZCG)
     DEJINTY(I)=CDYO(K,I,1)*(X(LSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(LSATOM)-YCG)+O(2,1)*(RMJ-ONUM) &
          +CDYO(K,I,3)*(Z(LSATOM)-ZCG)
     DEJINTZ(I)=CDZO(K,I,1)*(X(LSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(LSATOM)-ZCG)+O(3,1)*(RMJ-ONUM)
     ! (b_jx)' == (b_jxx)'
     DBJX(I)=DBJINTX(I)*O(1,1)+DOTK*CDXO(K,I,1)+ONUM
     DBJY(I)=DBJINTY(I)*O(1,1)+DOTK*CDYO(K,I,1)
     DBJZ(I)=DBJINTZ(I)*O(1,1)+DOTK*CDZO(K,I,1)
     ! (b_jyx)'
     DBJYX(I)=DBJINTX(I)*O(2,1)+DOTK*CDXO(K,I,2)
     DBJYY(I)=DBJINTY(I)*O(2,1)+DOTK*CDYO(K,I,2)+ONUM
     DBJYZ(I)=DBJINTZ(I)*O(2,1)+DOTK*CDZO(K,I,2)
     ! (b_jzx)'
     DBJZX(I)=DBJINTX(I)*O(3,1)+DOTK*CDXO(K,I,3)
     DBJZY(I)=DBJINTY(I)*O(3,1)+DOTK*CDYO(K,I,3)
     DBJZZ(I)=DBJINTZ(I)*O(3,1)+DOTK*CDZO(K,I,3)+ONUM
     ! (e_jx)' = 1 / nsel(k) + [ (a_jx)' ( r_mjx - 1 / nsel(k) * Si(x_i) ) 
     !                       - a_jx ( (r_mjx)' - 1 / nsel(k) ) ] * a_jx
     !                       + [ a_j (dot) ( r_mj - bar(r_j) ) ] (a_jx)'
     DEJX(I)=DEJINTX(I)*O(1,1)+DOTKM*CDXO(K,I,1)+ONUM
     DEJY(I)=DEJINTY(I)*O(1,1)+DOTKM*CDYO(K,I,1)
     DEJZ(I)=DEJINTZ(I)*O(1,1)+DOTKM*CDZO(K,I,1)
     ! (e_jyx)'
     DEJYX(I)=DEJINTX(I)*O(2,1)+DOTKM*CDXO(K,I,2)
     DEJYY(I)=DEJINTY(I)*O(2,1)+DOTKM*CDYO(K,I,2)+ONUM
     DEJYZ(I)=DEJINTZ(I)*O(2,1)+DOTKM*CDZO(K,I,2)
     ! (e_jzx)'
     DEJZX(I)=DEJINTX(I)*O(3,1)+DOTKM*CDXO(K,I,3)
     DEJZY(I)=DEJINTY(I)*O(3,1)+DOTKM*CDYO(K,I,3)
     DEJZZ(I)=DEJINTZ(I)*O(3,1)+DOTKM*CDZO(K,I,3)+ONUM
     ! (pw1)'
     DPW1XJ(I)=-DBJX(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYX(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZX(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DPW1YJ(I)=-DBJY(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYY(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZY(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DPW1ZJ(I)=-DBJZ(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
          -DBJYZ(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
          -DBJZZ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ! (pw2)'
     DPW2XJ(I)=-DBJX(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
          -DBJYX(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
          -DBJZX(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))
     DPW2YJ(I)=-DBJY(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &
          -DBJYY(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
          -DBJZY(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZY(I)-DBJZY(I))
     DPW2ZJ(I)=-DBJZ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CBVEC(1,JK)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &
          -DBJYZ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CBVEC(2,JK)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
          -DBJZZ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CBVEC(3,JK)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     ! (du22)'
     DDU22XJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))
     DDU22YJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     DDU22ZJ(I)=TWO*(CEVEC(1,JJ)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &
          +TWO*(CEVEC(2,JJ)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
          +TWO*(CEVEC(3,JJ)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     ! (du12)'
     DDU12XJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJX(I)-DBJX(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYX(I)-DBJYX(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZX(I)-DBJZX(I))
     DDU12YJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJY(I)-DBJY(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYY(I)-DBJYY(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZY(I)-DBJZY(I))
     DDU12ZJ(I)=(CEVEC(1,JK)-CBVEC(1,JK))*(DEJZ(I)-DBJZ(I)) &
          +(CEVEC(2,JK)-CBVEC(2,JK))*(DEJYZ(I)-DBJYZ(I)) &
          +(CEVEC(3,JK)-CBVEC(3,JK))*(DEJZZ(I)-DBJZZ(I))
     ! (du11)' = 0
     ! (det)' 
     DDETXJ(I)=DU11*DDU22XJ(I)-TWO*DU12*DDU12XJ(I)
     DDETYJ(I)=DU11*DDU22YJ(I)-TWO*DU12*DDU12YJ(I)
     DDETZJ(I)=DU11*DDU22ZJ(I)-TWO*DU12*DDU12ZJ(I)
     ! LIMIT S_K = 1
     ! B=(e_k-b_j) (dot) (e_j-b_j)
     IF(LLIMIT.AND.LSK1) THEN
        DBXJ(I)=-DBJX(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJX(I)-DBJX(I)) &
             -DBJYX(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYX(I)-DBJYX(I)) &
             -DBJZX(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZX(I)-DBJZX(I))
        DBYJ(I)=-DBJY(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJY(I)-DBJY(I)) &
             -DBJYY(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYY(I)-DBJYY(I)) &
             -DBJZY(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZY(I)-DBJZY(I))
        DBZJ(I)=-DBJZ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+(CEVEC(1,JK)-CBVEC(1,JJ))*(DEJZ(I)-DBJZ(I)) &
             -DBJYZ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+(CEVEC(2,JK)-CBVEC(2,JJ))*(DEJYZ(I)-DBJYZ(I)) &
             -DBJZZ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+(CEVEC(3,JK)-CBVEC(3,JJ))*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! limit S_j = 1
     ! A=(b_k-e_j) (dot) (e_k-b_k)
     ! (A)'
     IF(LLIMIT.AND.LSJ1) THEN
        DAXJ(I)=-DEJX(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYX(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZX(I)*(CEVEC(3,JK)-CBVEC(3,JK))
        DAYJ(I)=-DEJY(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYY(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZY(I)*(CEVEC(3,JK)-CBVEC(3,JK))
        DAZJ(I)=-DEJZ(I)*(CEVEC(1,JK)-CBVEC(1,JK)) &
             -DEJYZ(I)*(CEVEC(2,JK)-CBVEC(2,JK)) &
             -DEJZZ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ENDIF
     ! (S_K)'
     DSKXJ(I)=(-DPW1XJ(I)*DU22-PW1*DDU22XJ(I)+DPW2XJ(I)*DU12+PW2*DDU12XJ(I)) &
          /DET+PRESK*(-1.D0/(DET*DET)*DDETXJ(I))
     DSKYJ(I)=(-DPW1YJ(I)*DU22-PW1*DDU22YJ(I)+DPW2YJ(I)*DU12+PW2*DDU12YJ(I)) &
          /DET+PRESK*(-1.D0/(DET*DET)*DDETYJ(I))
     DSKZJ(I)=(-DPW1ZJ(I)*DU22-PW1*DDU22ZJ(I)+DPW2ZJ(I)*DU12+PW2*DDU12ZJ(I)) &
          /DET+PRESK*(-1.D0/(DET*DET)*DDETZJ(I))
     ! LIMIT S_J = 1
     IF(LLIMIT.AND.LSJ1) THEN
        DSKXJ(I)=-DAXJ(I)/DU11
        DSKYJ(I)=-DAYJ(I)/DU11
        DSKZJ(I)=-DAZJ(I)/DU11
     ENDIF
     ! Bugs in crossing angle Sj = 0
     ! limit S_j = 0
     IF(LLIMIT.AND.LSJ0) THEN
        DSKXJ(I)=-DPW1XJ(I)/DU11
        DSKYJ(I)=-DPW1YJ(I)/DU11
        DSKZJ(I)=-DPW1ZJ(I)/DU11
     ENDIF
     ! (S_j)'
     DSJXJ(I)=(DPW2XJ(I)*DU11-DPW1XJ(I)*DU12-PW1*DDU12XJ(I))/DET+PRESJ &
          *(-1.D0/(DET*DET)*DDETXJ(I))
     DSJYJ(I)=(DPW2YJ(I)*DU11-DPW1YJ(I)*DU12-PW1*DDU12YJ(I))/DET+PRESJ &
          *(-1.D0/(DET*DET)*DDETYJ(I))
     DSJZJ(I)=(DPW2ZJ(I)*DU11-DPW1ZJ(I)*DU12-PW1*DDU12ZJ(I))/DET+PRESJ &
          *(-1.D0/(DET*DET)*DDETZJ(I))
     ! LIMIT S_K = 0
     IF(LLIMIT.AND.LSK0) THEN
        DSJXJ(I)=DPW2XJ(I)/DU22-PW2/(DU22*DU22)*DDU22XJ(I)
        DSJYJ(I)=DPW2YJ(I)/DU22-PW2/(DU22*DU22)*DDU22YJ(I)
        DSJZJ(I)=DPW2ZJ(I)/DU22-PW2/(DU22*DU22)*DDU22ZJ(I)
     ENDIF
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DSJXJ(I)=DBXJ(I)/DU22-PB/(DU22*DU22)*DDU22XJ(I)
        DSJYJ(I)=DBYJ(I)/DU22-PB/(DU22*DU22)*DDU22YJ(I)
        DSJZJ(I)=DBZJ(I)/DU22-PB/(DU22*DU22)*DDU22ZJ(I)
     ENDIF
     ! (T_JX)' = (T_JXX)'
     DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJX(I)-DBJX(I))
     DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJY(I)-DBJY(I))
     DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PRESJ/DET*(DEJZ(I)-DBJZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJX(I)-DBJX(I))
        DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJY(I)-DBJY(I))
        DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+LSK0SJ*(DEJZ(I)-DBJZ(I))
     ENDIF
     ! limit S_k = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJXJ(I)=DBJX(I)+DSJXJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJX(I)-DBJX(I))
        DTJYJ(I)=DBJY(I)+DSJYJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJY(I)-DBJY(I))
        DTJZJ(I)=DBJZ(I)+DSJZJ(I)*(CEVEC(1,JJ)-CBVEC(1,JJ))+PB/DU22*(DEJZ(I)-DBJZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJXJ(I)=DBJX(I)
        DTJYJ(I)=DBJY(I)
        DTJZJ(I)=DBJZ(I)
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJXJ(I)=DEJX(I)
        DTJYJ(I)=DEJY(I)
        DTJZJ(I)=DEJZ(I)
     ENDIF
     ! (t_jy)'
     DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYX(I)-DBJYX(I))
     DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYY(I)-DBJYY(I))
     DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PRESJ/DET*(DEJYZ(I)-DBJYZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYX(I)-DBJYX(I))
        DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYY(I)-DBJYY(I))
        DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+LSK0SJ*(DEJYZ(I)-DBJYZ(I))
     ENDIF
     ! limit S_k = 1 
     IF(LLIMIT.AND.LSK1) THEN
        DTJYJX(I)=DBJYX(I)+DSJXJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYX(I)-DBJYX(I))
        DTJYJY(I)=DBJYY(I)+DSJYJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYY(I)-DBJYY(I))
        DTJYJZ(I)=DBJYZ(I)+DSJZJ(I)*(CEVEC(2,JJ)-CBVEC(2,JJ))+PB/DU22*(DEJYZ(I)-DBJYZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJYJX(I)=DBJYX(I)
        DTJYJY(I)=DBJYY(I)
        DTJYJZ(I)=DBJYZ(I)
     ENDIF
     ! LIMIT S_J = 1 OR ( S_K = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJYJX(I)=DEJYX(I)
        DTJYJY(I)=DEJYY(I)
        DTJYJZ(I)=DEJYZ(I)
     ENDIF
     ! (t_jz)'
     DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZX(I)-DBJZX(I))
     DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZY(I)-DBJZY(I))
     DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PRESJ/DET*(DEJZZ(I)-DBJZZ(I))
     ! limit S_k = 0
     IF(LLIMIT.AND.LSK0) THEN
        DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZX(I)-DBJZX(I))
        DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZY(I)-DBJZY(I))
        DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+LSK0SJ*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! LIMIT S_K = 1
     IF(LLIMIT.AND.LSK1) THEN
        DTJZJX(I)=DBJZX(I)+DSJXJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZX(I)-DBJZX(I))
        DTJZJY(I)=DBJZY(I)+DSJYJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZY(I)-DBJZY(I))
        DTJZJZ(I)=DBJZZ(I)+DSJZJ(I)*(CEVEC(3,JJ)-CBVEC(3,JJ))+PB/DU22*(DEJZZ(I)-DBJZZ(I))
     ENDIF
     ! limit S_j = 0 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSJ0).OR.LSK1SJ0.OR.LSK0SJ0) THEN
        DTJZJX(I)=DBJZX(I)
        DTJZJY(I)=DBJZY(I)
        DTJZJZ(I)=DBJZZ(I)
     ENDIF
     ! limit S_j = 1 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSJ1).OR.LSK0SJ1.OR.LSK1SJ1) THEN
        DTJZJX(I)=DEJZX(I)
        DTJZJY(I)=DEJZY(I)
        DTJZJZ(I)=DEJZZ(I)
     ENDIF
     ! (t_kx)'
     DTKXJ(I)=DSKXJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     DTKYJ(I)=DSKYJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     DTKZJ(I)=DSKZJ(I)*(CEVEC(1,JK)-CBVEC(1,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) 
     !               or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKXJ(I)=0.D0
        DTKYJ(I)=0.D0
        DTKZJ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) 
     !               or ( S_k = 1 and S_j = 1 )
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKXJ(I)=0.D0
        DTKYJ(I)=0.D0
        DTKZJ(I)=0.D0
     ENDIF
     ! (t_ky)'
     DTKYJX(I)=DSKXJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     DTKYJY(I)=DSKYJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     DTKYJZ(I)=DSKZJ(I)*(CEVEC(2,JK)-CBVEC(2,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKYJX(I)=0.D0
        DTKYJY(I)=0.D0
        DTKYJZ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) or ( S_k = 1 and S_j = 1)
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKYJX(I)=0.D0
        DTKYJY(I)=0.D0
        DTKYJZ(I)=0.D0
     ENDIF
     ! (t_kz)'
     DTKZJX(I)=DSKXJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DTKZJY(I)=DSKYJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     DTKZJZ(I)=DSKZJ(I)*(CEVEC(3,JK)-CBVEC(3,JK))
     ! limit S_k = 0 or ( S_k = 0 and S_j = 1 ) or ( S_k = 0 and S_j = 0 )
     IF((LLIMIT.AND.LSK0).OR.LSK0SJ1.OR.LSK0SJ0) THEN
        DTKZJX(I)=0.D0
        DTKZJY(I)=0.D0
        DTKZJZ(I)=0.D0
     ENDIF
     ! limit S_k = 1 or ( S_k = 1 and S_j = 0 ) or ( S_k = 1 and S_j = 1)
     IF((LLIMIT.AND.LSK1).OR.LSK1SJ0.OR.LSK1SJ1) THEN
        DTKZJX(I)=0.D0
        DTKZJY(I)=0.D0
        DTKZJZ(I)=0.D0
     ENDIF
     ! |t_j - t_k|
     DTJMTKXJ(I)=1.D0/(TWO*TJMTK)* &
          (TWO*(CTVEC(1,JJ)-CTVEC(1,JK))*(DTJXJ(I)-DTKXJ(I)) &
          +TWO*(CTVEC(2,JJ)-CTVEC(2,JK))*(DTJYJX(I)-DTKYJX(I)) &
          +TWO*(CTVEC(3,JJ)-CTVEC(3,JK))*(DTJZJX(I)-DTKZJX(I)))
     DTJMTKYJ(I)=1.D0/(TWO*TJMTK)* &
          (TWO*(CTVEC(1,JJ)-CTVEC(1,JK))*(DTJYJ(I)-DTKYJ(I)) &
          +TWO*(CTVEC(2,JJ)-CTVEC(2,JK))*(DTJYJY(I)-DTKYJY(I)) &
          +TWO*(CTVEC(3,JJ)-CTVEC(3,JK))*(DTJZJY(I)-DTKZJY(I)))
     DTJMTKZJ(I)=1.D0/(TWO*TJMTK)* &
          (TWO*(CTVEC(1,JJ)-CTVEC(1,JK))*(DTJZJ(I)-DTKZJ(I)) &
          +TWO*(CTVEC(2,JJ)-CTVEC(2,JK))*(DTJYJZ(I)-DTKYJZ(I)) &
          +TWO*(CTVEC(3,JJ)-CTVEC(3,JK))*(DTJZJZ(I)-DTKZJZ(I)))
     ! B_x by x_j
     DTBXJ(I)=(DTJXJ(I)-DTKXJ(I))/TJMTK-(CTVEC(1,JJ)-CTVEC(1,JK))*DTJMTKXJ(I)/(TJMTK*TJMTK)
     DTBYJ(I)=(DTJYJ(I)-DTKYJ(I))/TJMTK-(CTVEC(1,JJ)-CTVEC(1,JK))*DTJMTKYJ(I)/(TJMTK*TJMTK)
     DTBZJ(I)=(DTJZJ(I)-DTKZJ(I))/TJMTK-(CTVEC(1,JJ)-CTVEC(1,JK))*DTJMTKZJ(I)/(TJMTK*TJMTK)
     ! b_y by x_j
     DTBYXJ(I)=(DTJYJX(I)-DTKYJX(I))/TJMTK-(CTVEC(2,JJ)-CTVEC(2,JK))*DTJMTKXJ(I)/(TJMTK*TJMTK)
     DTBYYJ(I)=(DTJYJY(I)-DTKYJY(I))/TJMTK-(CTVEC(2,JJ)-CTVEC(2,JK))*DTJMTKYJ(I)/(TJMTK*TJMTK)
     DTBYZJ(I)=(DTJYJZ(I)-DTKYJZ(I))/TJMTK-(CTVEC(2,JJ)-CTVEC(2,JK))*DTJMTKZJ(I)/(TJMTK*TJMTK)
     ! b_z by x_j
     DTBZXJ(I)=(DTJZJX(I)-DTKZJX(I))/TJMTK-(CTVEC(3,JJ)-CTVEC(3,JK))*DTJMTKXJ(I)/(TJMTK*TJMTK)
     DTBZYJ(I)=(DTJZJY(I)-DTKZJY(I))/TJMTK-(CTVEC(3,JJ)-CTVEC(3,JK))*DTJMTKYJ(I)/(TJMTK*TJMTK)
     DTBZZJ(I)=(DTJZJZ(I)-DTKZJZ(I))/TJMTK-(CTVEC(3,JJ)-CTVEC(3,JK))*DTJMTKZJ(I)/(TJMTK*TJMTK)
     ! (|a_k (X) b|)' by x_j
     DAKXBJX(I)=1.D0/(TWO*AKXB)*( &
          TWO*(C1(2)*C2(3)-C1(3)*C2(2))*(C1(2)*DTBZXJ(I)-C1(3)*DTBYXJ(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(C1(3)*DTBXJ(I)-C1(1)*DTBZXJ(I)) &
          +TWO*(C1(1)*C2(2)-C1(2)*C2(1))*(C1(1)*DTBYXJ(I)-C1(2)*DTBXJ(I)))
     DAKXBJY(I)=1.D0/(TWO*AKXB)*( &
          TWO*(C1(2)*C2(3)-C1(3)*C2(2))*(C1(2)*DTBZYJ(I)-C1(3)*DTBYYJ(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(C1(3)*DTBYJ(I)-C1(1)*DTBZYJ(I)) &
          +TWO*(C1(1)*C2(2)-C1(2)*C2(1))*(C1(1)*DTBYYJ(I)-C1(2)*DTBYJ(I)))
     DAKXBJZ(I)=1.D0/(TWO*AKXB)*( &
          TWO*(C1(2)*C2(3)-C1(3)*C2(2))*(C1(2)*DTBZZJ(I)-C1(3)*DTBYZJ(I)) &
          +TWO*(C1(3)*C2(1)-C1(1)*C2(3))*(C1(3)*DTBZJ(I)-C1(1)*DTBZZJ(I)) &
          +TWO*(C1(1)*C2(2)-C1(2)*C2(1))*(C1(1)*DTBYZJ(I)-C1(2)*DTBZJ(I)))
     ! (|b (X) c|)' by x_j
     ! = dbxcxj
     DBXCXJ(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(-CDXO(K,I,3)*C2(2)+C3(3)*DTBYXJ(I)+CDXO(K,I,2)*C2(3)-C3(2)*DTBZXJ(I)) &
          +TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(-CDXO(K,I,1)*C2(3)+C3(1)*DTBZXJ(I)+CDXO(K,I,3)*C2(1)-C3(3)*DTBXJ(I)) &
          +TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(-CDXO(K,I,2)*C2(1)+C3(2)*DTBXJ(I)+CDXO(K,I,1)*C2(2)-C3(1)*DTBYXJ(I)))
     DBXCYJ(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(-CDYO(K,I,3)*C2(2)+C3(3)*DTBYYJ(I)+CDYO(K,I,2)*C2(3)-C3(2)*DTBZYJ(I)) &
          +TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(-CDYO(K,I,1)*C2(3)+C3(1)*DTBZYJ(I)+CDYO(K,I,3)*C2(1)-C3(3)*DTBYJ(I)) &
          +TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(-CDYO(K,I,2)*C2(1)+C3(2)*DTBYJ(I)+CDYO(K,I,1)*C2(2)-C3(1)*DTBYYJ(I)))
     DBXCZJ(I)=1.D0/(TWO*BXC)*(TWO*(C3(3)*C2(2)-C3(2)*C2(3)) &
          *(-CDZO(K,I,3)*C2(2)+C3(3)*DTBYZJ(I)+CDZO(K,I,2)*C2(3)-C3(2)*DTBZZJ(I)) &
          +TWO*(C3(1)*C2(3)-C3(3)*C2(1)) &
          *(-CDZO(K,I,1)*C2(3)+C3(1)*DTBZZJ(I)+CDZO(K,I,3)*C2(1)-C3(3)*DTBZJ(I)) &
          +TWO*(C3(2)*C2(1)-C3(1)*C2(2)) &
          *(-CDZO(K,I,2)*C2(1)+C3(2)*DTBZJ(I)+CDZO(K,I,1)*C2(2)-C3(1)*DTBYZJ(I)))
     ! (cos(theta))' by x_j = dcostxj
     DCOSTXJ(I)=((C1(2)*DTBZXJ(I)-C1(3)*DTBYXJ(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(-CDXO(K,I,3)*C2(2)+C3(3)*DTBYXJ(I)+CDXO(K,I,2)*C2(3)-C3(2)*DTBZXJ(I)) &
          +(C1(3)*DTBXJ(I)-C1(1)*DTBZXJ(I)) &
          *(C3(1)*C2(3)-C3(3)*C2(1))+(C1(3)*C2(1)-C1(1)*C2(3)) &
          *(-CDXO(K,I,1)*C2(3)+C3(1)*DTBZXJ(I)+CDXO(K,I,3)*C2(1)-C3(3)*DTBXJ(I)) &
          +(C1(1)*DTBYXJ(I)-C1(2)*DTBXJ(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(-CDXO(K,I,2)*C2(1)+C3(2)*DTBXJ(I)+CDXO(K,I,1)*C2(2)-C3(1)*DTBYXJ(I))) &
          /(AKXB*BXC)-DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBJX(I)*BXC+AKXB*DBXCXJ(I))
     DCOSTYJ(I)=((C1(2)*DTBZYJ(I)-C1(3)*DTBYYJ(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(-CDYO(K,I,3)*C2(2)+C3(3)*DTBYYJ(I)+CDYO(K,I,2)*C2(3)-C3(2)*DTBZYJ(I)) &
          +(C1(3)*DTBYJ(I)-C1(1)*DTBZYJ(I)) &
          *(C3(1)*C2(3)-C3(3)*C2(1))+(C1(3)*C2(1)-C1(1)*C2(3)) &
          *(-CDYO(K,I,1)*C2(3)+C3(1)*DTBZYJ(I)+CDYO(K,I,3)*C2(1)-C3(3)*DTBYJ(I)) &
          +(C1(1)*DTBYYJ(I)-C1(2)*DTBYJ(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(-CDYO(K,I,2)*C2(1)+C3(2)*DTBYJ(I)+CDYO(K,I,1)*C2(2)-C3(1)*DTBYYJ(I))) &
          /(AKXB*BXC)-DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBJY(I)*BXC+AKXB*DBXCYJ(I))
     DCOSTZJ(I)=((C1(2)*DTBZZJ(I)-C1(3)*DTBYZJ(I)) &
          *(C3(3)*C2(2)-C3(2)*C2(3))+(C1(2)*C2(3)-C1(3)*C2(2)) &
          *(-CDZO(K,I,3)*C2(2)+C3(3)*DTBYZJ(I)+CDZO(K,I,2)*C2(3)-C3(2)*DTBZZJ(I)) &
          +(C1(3)*DTBZJ(I)-C1(1)*DTBZZJ(I)) &
          *(C3(1)*C2(3)-C3(3)*C2(1))+(C1(3)*C2(1)-C1(1)*C2(3)) &
          *(-CDZO(K,I,1)*C2(3)+C3(1)*DTBZZJ(I)+CDZO(K,I,3)*C2(1)-C3(3)*DTBZJ(I)) &
          +(C1(1)*DTBYZJ(I)-C1(2)*DTBZJ(I)) &
          *(C3(2)*C2(1)-C3(1)*C2(2))+(C1(1)*C2(2)-C1(2)*C2(1)) &
          *(-CDZO(K,I,2)*C2(1)+C3(2)*DTBZJ(I)+CDZO(K,I,1)*C2(2)-C3(1)*DTBYZJ(I))) &
          /(AKXB*BXC)-DOTP/(AKXB*BXC*AKXB*BXC)*(DAKXBJZ(I)*BXC+AKXB*DBXCZJ(I))
     ! (U_jx)'
     DUJX(I)=PREU*DCOSTXJ(I)
     DUJY(I)=PREU*DCOSTYJ(I)
     DUJZ(I)=PREU*DCOSTZJ(I)
  ENDDO
  ! U = k ( D - D_0 )^2
  EIJ=AFOR*(THETA-ANGL)*(THETA-ANGL)*DEGRAD*DEGRAD
  ! first selected
  K=1
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO

  IF(PRNLEV >= 6) THEN
     ! K=1
     WRITE(6,*) 'CPRVEC'
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO

     WRITE(6,*) 'FORCE'
     ! force = - (u)'
     SUMX=0.D0
     SUMY=0.D0
     SUMZ=0.D0
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I),DUKY(I),DUKZ(I)
        SUMX=SUMX+DUKX(I)
        SUMY=SUMY+DUKY(I)
        SUMZ=SUMZ+DUKZ(I)
     ENDDO

     SUM=SQRT(SUMX*SUMX+SUMY*SUMY+SUMZ*SUMZ)
     WRITE(6,*) 'SUM,K',SUMX,SUMY,SUMZ,SUM

     SUMX=0.D0
     SUMY=0.D0
     SUMZ=0.D0
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUJX(I),DUJY(I),DUJZ(I)
        SUMX=SUMX+DUJX(I)
        SUMY=SUMY+DUJY(I)
        SUMZ=SUMZ+DUJZ(I)
     ENDDO

     SUM=SQRT(SUMX*SUMX+SUMY*SUMY+SUMZ*SUMZ)
     WRITE(6,*) 'SUM,J',SUMX,SUMY,SUMZ,SUM
  ENDIF
  ! second selected
  K=2
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUJX(I)
     DY(NATOMM)=DY(NATOMM)+DUJY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUJZ(I)
  ENDDO
  ! energy
  ECHEL=ENEANG+EIJ
  ENEANG=ECHEL
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(X Ang) E. of #', ' : ',eij
     WRITE(OUTU,*) 'ENEANG',ENEANG
  ENDIF
  RETURN
END SUBROUTINE ECHXA

SUBROUTINE ECHTA(ECHEL,X,Y,Z,DX,DY,DZ,ANGL,AFOR,NSEL,ASLCT,BSLCT,AMASS, &
     CAVEC,CU,CEV,CBVEC,CEVEC,CPVEC,CHUNIT,CHSTEP,ENEANG,LQSTPRT,MCOUNT, &
     LTAX,LTAY,LTAZ,LBHG,LBHP)
  !----------------------------------------------------------------------
  !    Helix tilt angle restraint routine
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT
  use vector

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)

  real(chm_real) ANGL,AFOR,ENEANG,AKDVZ,THETA,PREU
  real(chm_real) AMASS(*)
  real(chm_real) CAVEC(NCOOR,MXHEL),CBVEC(NCOOR,MXHEL),CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) AK(NCOOR),VZ(NCOOR)
  real(chm_real) DAKX(MXSLT),DAKY(MXSLT),DAKZ(MXSLT)
  real(chm_real) DCOSTX(MXSLT),DCOSTY(MXSLT),DCOSTZ(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) EIJ

  real(chm_real) DUK,DUKOAK,ANG
  real(chm_real) PAVEX,PAVEY,PAVEZ
  real(chm_real) NAVEX,NAVEY,NAVEZ
  real(chm_real) PHI,TMP,TMP2
  real(chm_real) DIREX,DIREY,DIREZ
  real(chm_real) CX,CY,CZ
  real(chm_real) ADDA,ACDDIR,MAGAC,PPP
  real(chm_real) BEGX,BEGY,BEGZ

  INTEGER NUMP,NUMN
  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,J,K,KK
  INTEGER NATOMM

  LOGICAL LMETH,LQSTPRT
  LOGICAL LTAX,LTAY,LTAZ,LBHG,LBHP

  ! first helix
  K=1
  ! U = k ( T - T_0 )^2
  ! cos(T) = a_k (dot) v_z / (|a_k| |v_z|)
  IF(PRNLEV >= 6) WRITE(6,*) 'Ltax, Ltay, Ltaz :',ltax,ltay,ltaz
  ! tilx 
  IF(LTAX) THEN
     DO I=1,3
        AK(I)=CAVEC(I,K)
        IF(I == 1) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
     ! TILY
  ELSEIF(LTAY) THEN
     DO I=1,3
        AK(I)=CAVEC(I,K)
        IF(I == 2) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
     ! tilz or default
  ELSE
     DO I=1,3
        AK(I)=CAVEC(I,K)
        IF(I == 3) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
  ENDIF
  ! akdvz
  ! = a_k (dot) v_z
  CALL DOTPR(AK,VZ,3,AKDVZ)
  ! unit vector a_k and v_z
  THETA=ACOS(AKDVZ)*RADDEG
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'AKDVZ',AKDVZ,THETA
  ! (U_kx)' = 2 k ( T - T_0 ) / - sin(T) (cos(T))'
  ! preU = - 2 k (T - T_0 ) / sin(t)
  PREU=-TWO*AFOR*(THETA-ANGL)*DEGRAD/SIN(THETA*DEGRAD)
  ! write down to file
  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP < MCOUNT) MCOUNT=MCOUNT-1
  IF(PRNLEV > 6.AND.DYNAMQ) WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,THETA,LQSTPRT,MCOUNT
  IF(CHSTEP > 0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') THETA
     ENDIF
  ENDIF
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'eigenvector_pre',o(1,1),o(2,1),o(3,1)
     WRITE(OUTU,*) 'eigenvector_pre2',cavec(1,1),cavec(2,1),cavec(3,1)
     WRITE(OUTU,*) 'eigenvector_pre3',ak(1),ak(2),ak(3)
     WRITE(OUTU,*) 'eigenvalue',ev(1),ev(2),ev(3)
  ENDIF
  IF(LBHG.OR.LBHP) THEN
     ! beta-hairpin mode and general moment of inertia
     CALL ROTINVARG(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO)
  ELSE
     ! DEFAULT (HELIX) MOMENT OF INERTIA
     CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ENDIF
  ! K = NUMBER OF HELIX ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  !-----------------------------------------------------------------------
  ! Main loop
  DO I=1,NSEL(K)
     ! (|a_k|)'
     ! dakx(i)
     ! = 1 / ( 2 |a_k| ) ( 2 a_kx (a_kx)' + 2 a_ky (a_ky)' + 2 a_kz (a_kz)' )
     DAKX(I)=1.D0/TWO*(TWO*AK(1)*CDXO(K,I,1)+TWO*AK(2)*CDXO(K,I,2)+TWO*AK(3)*CDXO(K,I,3))
     DAKY(I)=1.D0/TWO*(TWO*AK(1)*CDYO(K,I,1)+TWO*AK(2)*CDYO(K,I,2)+TWO*AK(3)*CDYO(K,I,3))
     DAKZ(I)=1.D0/TWO*(TWO*AK(1)*CDZO(K,I,1)+TWO*AK(2)*CDZO(K,I,2)+TWO*AK(3)*CDZO(K,I,3))
     ! (cos(T))'
     ! = (a_kz)' + (a_k (dot) v_z) (-1) (|a_k| |v_z|)^-2
     !            * ( (|a_k|)' |v_z| )
     ! Tilt_X
     IF(LTAX) THEN
        DCOSTX(I)=CDXO(K,I,1)-AKDVZ*DAKX(I)
        DCOSTY(I)=CDYO(K,I,1)-AKDVZ*DAKY(I)
        DCOSTZ(I)=CDZO(K,I,1)-AKDVZ*DAKZ(I)
        ! Tilt_Y
     ELSEIF(LTAY) THEN
        DCOSTX(I)=CDXO(K,I,2)-AKDVZ*DAKX(I)
        DCOSTY(I)=CDYO(K,I,2)-AKDVZ*DAKY(I)
        DCOSTZ(I)=CDZO(K,I,2)-AKDVZ*DAKZ(I)
        ! Tilt_Z
     ELSE
        DCOSTX(I)=CDXO(K,I,3)-AKDVZ*DAKX(I)
        DCOSTY(I)=CDYO(K,I,3)-AKDVZ*DAKY(I)
        DCOSTZ(I)=CDZO(K,I,3)-AKDVZ*DAKZ(I)
     ENDIF
     ! (U_kx) = 2 k (T-T_0) / - sin(T) (cos(T))'
     ! = preU * dcostx(i)
     DUKX(I)=PREU*DCOSTX(I)
     DUKY(I)=PREU*DCOSTY(I)
     DUKZ(I)=PREU*DCOSTZ(I)
  ENDDO
  ! U = K (T-T_0)^2
  EIJ=AFOR*(THETA-ANGL)*(THETA-ANGL)*DEGRAD*DEGRAD
  ! The calculated forces add original ones
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO

  IF(PRNLEV >= 6) THEN
     !---------------------------------------------------------------------
     ! routine to get the forces which are parallel or anti-parallel
     ! to principal axis
     ! dukx duky dukz : individual tilting forces
     ! ak(1) ak(2) ak(3) : principal axis 

     ! get the theta(tilt) and phi(rotation angle) angle
     ! for the unit theta vector (direction vector)
     ! theta : theta (deg)
     ! direc = (cos(phi)*cos(theta),sin(phi)*cos(theta),-sin(theta)
     ! phi definition (rad)
     ! s=sqrt(x*x+y*y)
     ! x >= 0  phi = asin(y/s) 
     ! x <  0  phi = pi-asin(y/s) 
     TMP=SQRT(AK(1)*AK(1)+AK(2)*AK(2))
     IF(AK(1) >= 0.D0) PHI=ASIN(AK(2)/TMP)
     IF(AK(1) <  0.D0) PHI=PI-ASIN(AK(2)/TMP)
     DIREX=COS(PHI)*COS(THETA*DEGRAD)
     DIREY=SIN(PHI)*COS(THETA*DEGRAD)
     DIREZ=-SIN(THETA*DEGRAD)
     WRITE(6,*) 'direction vector', direx,direy,direz
     ! initization
     NUMP=0
     NUMN=0
     PAVEX=0.D0
     PAVEY=0.D0
     PAVEZ=0.D0
     NAVEX=0.D0
     NAVEY=0.D0
     NAVEZ=0.D0
     TMP2=0.D0
     ! A = (begin of principal axis) : cbvec : begx
     BEGX=CBVEC(1,1)
     BEGY=CBVEC(2,1)
     BEGZ=CBVEC(3,1)
     ! select method1(.false.) and method2(.true.)
     LMETH=.FALSE.

     DO I=1,NSEL(K)
        NATOMM=ASLCT(I)
        ! using first
        IF(.NOT.LMETH) THEN
           ! get the rotation angle
           ! see my note
           ! c = d - ( Ad (dot) a ) a
           !
           ! inner product for helix axis to check same orientation or not
           ! tilt <= 90 same direction ; 90 < tilt <= 180 opposite direction
           ! inner product for individual force (needed - ) to get the orientation
           !
           ! get the selected c_alpha coordinates (D) and projected coordinate (C)
           !
           ! inner product (Ad (dot) a)
           ADDA=(X(NATOMM)-BEGX)*AK(1)+(Y(NATOMM)-BEGY)*AK(2)+(Z(NATOMM)-BEGZ)*AK(3)
           ! c = d - ( ad (dot) a ) a
           CX=X(NATOMM)-ADDA*AK(1)
           CY=Y(NATOMM)-ADDA*AK(2)
           CZ=Z(NATOMM)-ADDA*AK(3)
           WRITE(6,*) X(NATOMM),Y(NATOMM),Z(NATOMM)
           WRITE(6,*) ADDA
           WRITE(6,*) AK(1),AK(2),AK(3)
           WRITE(6,*) 'BEGX',BEGX,BEGY,BEGZ
           WRITE(6,*) 'CX,CY,CZ',CX,CY,CZ
           ! define the individual force information (pos and neg)
           ! ac (dot) direc = |ac| cos (ppp)
           ! ppp = acos ( acddir / |ac| )
           ! ppp <= 90 degs ; positive
           ! ppp > 90 degs ; negative
           ACDDIR=(CX-BEGX)*DIREX+(CY-BEGY)*DIREY+(CZ-BEGZ)*DIREZ
           MAGAC=SQRT((CX-BEGX)*(CX-BEGX)+(CY-BEGY)*(CY-BEGY)+(CZ-BEGZ)*(CZ-BEGZ))
           PPP=ACOS(ACDDIR/MAGAC)*RADDEG
           ! FRONT slant
           IF(PPP <= 90.D0) THEN
              ! real force
              PAVEX=PAVEX-DUKX(I)
              PAVEY=PAVEY-DUKY(I)
              PAVEZ=PAVEZ-DUKZ(I)
              WRITE(6,*) 'PAVEX',PAVEX,PAVEY,PAVEZ
              NUMP=NUMP+1
           ENDIF
           ! back slant
           IF(PPP > 90.D0) THEN
              NAVEX=NAVEX-DUKX(I)
              NAVEY=NAVEY-DUKY(I)
              NAVEZ=NAVEZ-DUKZ(I)
              NUMN=NUMN+1
           ENDIF
        ENDIF
        ! using second
        IF(LMETH) THEN
           ! cos(ang)= duk (o) ak / |duk| |ak| = duk (o) ak / |duk|
           ! |duk|
           DUK=SQRT(DUKX(I)*DUKX(I)+DUKY(I)*DUKY(I)+DUKZ(I)*DUKZ(I))
           ! dukoak = duk (o) ak
           DUKOAK=DUKX(I)*AK(1)+DUKY(I)*AK(2)+DUKZ(I)*AK(3)
           ANG=ACOS(DUKOAK/DUK)*RADDEG 

           IF(ANG <= 90.D0) THEN
              PAVEX=PAVEX-DUKX(I)
              PAVEY=PAVEY-DUKY(I)
              PAVEZ=PAVEZ-DUKZ(I)
              NUMP=NUMP+1
           ENDIF
           IF(ANG > 90.D0) THEN
              NAVEX=NAVEX-DUKX(I)
              NAVEY=NAVEY-DUKY(I)
              NAVEZ=NAVEZ-DUKZ(I)
              NUMN=NUMN+1
           ENDIF
        ENDIF
     ENDDO
     ! projection into the axis
     TMP=-PAVEX*AK(1)-PAVEY*AK(2)-PAVEZ*AK(3)
     TMP2=NAVEX*AK(1)+NAVEY*AK(2)+NAVEZ*AK(3)
     WRITE(6,*) 'Projection into the Axis',2*(TMP+TMP2)
     !---------------------------------------------------------------------
  ENDIF
  IF(PRNLEV >= 6) THEN
     ! k=1
     WRITE(6,*) 'CPRVEC'
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     WRITE(6,*) 'FORCE'
     ! force = - (U)' WRONG
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I),DUKY(I),DUKZ(I)
     ENDDO
  ENDIF
  ! energy
  ECHEL=ENEANG+EIJ
  ENEANG=ECHEL
  ! DEBUG
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(T Ang) E. of #', ' : ',eij
     WRITE(OUTU,*) 'eneang',eneang
     WRITE(OUTU,*) 'force',-two*afor*(theta-angl)*degrad 
  ENDIF
  RETURN
END SUBROUTINE ECHTA

SUBROUTINE ECHro(ECHEL,X,Y,Z,DX,DY,DZ,ANGL,AFOR,NSEL,ASLCT,BSLCT, &
     AMASS,CPRVEC,CAVEC,CU,CEV,CBVEC,CEVEC,CPVEC,CHUNIT,CHSTEP,REFA, &
     ENEANG,LQSTPRT,MCOUNT,LHARM,LBHG,LBHP,LRAX,LRAY,LRAZ)
  !----------------------------------------------------------------------
  !    Helix rotation angle restraint routine along helical axis
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT
  use vector
  use param_store, only: set_param

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)

  real(chm_real) ANGL,AFOR
  real(chm_real) AMASS(*)
  real(chm_real) CAVEC(NCOOR,MXHEL),CBVEC(NCOOR,MXHEL),CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) ENEANG,AKDVZ,THETA,PREU,EIJ
  real(chm_real) AK(NCOOR),VZ(NCOOR)
  real(chm_real) DAKX(MXSLT),DAKY(MXSLT),DAKZ(MXSLT)
  real(chm_real) DCOSTX(MXSLT),DCOSTY(MXSLT),DCOSTZ(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) CPRVEC(NCOOR,MXSLT,MXHEL)
  real(chm_real) duk,dukoak,ang
  real(chm_real) pavex,pavey,pavez
  real(chm_real) navex,navey,navez
  real(chm_real) phi,tmp,tmp2
  real(chm_real) phia,thetaa
  real(chm_real) direx,direy,direz,cx,cy,cz,begx,begy,begz
  real(chm_real) adda,acddir,magac,ppp
  real(chm_real) orie(3),dotporie,czz(3),makp
  real(chm_real) rcprvec(ncoor),ralpha(ncoor),calpha(ncoor)
  real(chm_real) dvec(ncoor),dvxca(ncoor),mcalpha,ddvxcaak
  real(chm_real) rhoang,prhoang
  real(chm_real) xcg,ycg,zcg,onum,r1k,rmk,dotk,dotkm
  real(chm_real) DBKINTX(MXSLT),DBKINTY(MXSLT),DBKINTZ(MXSLT)
  real(chm_real) DBKX(MXSLT),DBKY(MXSLT),DBKZ(MXSLT)
  real(chm_real) DBKYX(MXSLT),DBKYY(MXSLT),DBKYZ(MXSLT)
  real(chm_real) DBKZX(MXSLT),DBKZY(MXSLT),DBKZZ(MXSLT)
  real(chm_real) oaalpalx(mxslt),oaalpaly(mxslt),oaalpalz(mxslt)
  real(chm_real) dcax(mxslt),dcay(mxslt),dcaz(mxslt)
  real(chm_real) dcayx(mxslt),dcayy(mxslt),dcayz(mxslt)
  real(chm_real) dcazx(mxslt),dcazy(mxslt),dcazz(mxslt)
  real(chm_real) dphix(mxslt),dphiy(mxslt),dphiz(mxslt)
  real(chm_real) dthetax(mxslt),dthetay(mxslt),dthetaz(mxslt)
  real(chm_real) ddvxx(mxslt),ddvxy(mxslt),ddvxz(mxslt)
  real(chm_real) ddvyx(mxslt),ddvyy(mxslt),ddvyz(mxslt)
  real(chm_real) ddvzx(mxslt),ddvzy(mxslt),ddvzz(mxslt)
  real(chm_real) dvdcalx(mxslt),dvdcaly(mxslt),dvdcalz(mxslt)
  real(chm_real) dvdotcal
  real(chm_real) dabscax(mxslt),dabscay(mxslt),dabscaz(mxslt)
  real(chm_real) dabsdvx(mxslt),dabsdvy(mxslt),dabsdvz(mxslt)
  real(chm_real) drhox(mxslt),drhoy(mxslt),drhoz(mxslt)
  real(chm_real) domcalx(mxslt),domcaly(mxslt),domcalz(mxslt)
  real(chm_real) idvec(ncoor),midvec
  real(chm_real) prang

  INTEGER NUMP,NUMN
  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,II,J,K,KK
  INTEGER NATOMM
  INTEGER REFA

  LOGICAL LQSTPRT,LBHG,LBHP
  LOGICAL LMETH,LHARM
  LOGICAL LRAX,LRAY,LRAZ

  ! first helix
  K=1
  ! get eigen vector 1 
  DO I=1,3
     AK(I)=CAVEC(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(6,*) 'EV',AK(1),AK(2),AK(3)
     DO I=1,NSEL(K)
        WRITE(6,'(A6,I3,3F8.3)') 'CPRVEC',I,CPRVEC(1,I,K),CPRVEC(2,I,K),CPRVEC(3,I,K)
     ENDDO
  ENDIF
  ! get the projected point of reference atom
  !     and the reference point
  DO I=1,3
     RCPRVEC(I)=CPRVEC(I,REFA,K)
  ENDDO

  NATOMM=ASLCT(REFA)
  RALPHA(1)=X(NATOMM)
  RALPHA(2)=Y(NATOMM)
  RALPHA(3)=Z(NATOMM)

  IF(PRNLEV >= 6) THEN
     WRITE(6,*) 'PREFA',(RCPRVEC(I),I=1,3)
     WRITE(6,*) 'RALPH',(RALPHA(I),I=1,3)
  ENDIF
  ! another method to get inverse dvector
  ! idvec=(-azax,-azay,1-azaz)
  IF(PRNLEV >= 6) WRITE(6,*) 'LRAX, LRAY, LRAZ :',LRAX,LRAY,LRAZ
  ! Rotation Reference Axis
  IF(LRAX) THEN
     DO I=1,3
        IF(I == 1) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
     ! RotY
  ELSEIF(LRAY) THEN
     DO I=1,3
        IF(I == 2) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
     ! RotZ or default
  ELSE
     DO I=1,3
        IF(I == 3) THEN
           VZ(I)=1.D0
        ELSE
           VZ(I)=0.D0
        ENDIF
     ENDDO
  ENDIF
  !
  IDVEC(1)=(AK(1)*AK(1)-1.D0)*VZ(1)+AK(1)*AK(2)*VZ(2)+AK(1)*AK(3)*VZ(3)
  IDVEC(2)=AK(2)*AK(1)*VZ(1)+(AK(2)*AK(2)-1.D0)*VZ(2)+AK(2)*AK(3)*VZ(3)
  IDVEC(3)=AK(3)*AK(1)*VZ(1)+AK(3)*AK(2)*VZ(2)+(AK(3)*AK(3)-1.D0)*VZ(3)
  ! RotX
  !   IDVEC(1)=AK(1)*AK(1)-1.D0
  !   IDVEC(2)=AK(1)*AK(2)
  !   IDVEC(3)=AK(1)*AK(3)
  ! RotY
  !   IDVEC(1)=AK(2)*AK(1)
  !   IDVEC(2)=AK(2)*AK(2)-1.D0
  !   IDVEC(3)=AK(2)*AK(3)
  ! RotZ
  !   IDVEC(1)=AK(3)*AK(1)
  !   IDVEC(2)=AK(3)*AK(2)
  !   IDVEC(3)=AK(3)*AK(3)-1.D0
  !
  TMP=0.D0
  DO I=1,3
     TMP=TMP+IDVEC(I)*IDVEC(I)
  ENDDO
  MIDVEC=SQRT(TMP)
  DO I=1,3
     DVEC(I)=IDVEC(I)/MIDVEC
  ENDDO
  !
  IF(PRNLEV >= 6) WRITE(6,'(A8,3F8.3)') 'DVECTOR', (DVEC(I),I=1,3)
  ! rotation angle along helical axis
  ! calpha = ralpha - rcprvec 
  TMP=0.D0
  DO I=1,3
     CALPHA(I)=RALPHA(I)-RCPRVEC(I)
     TMP=TMP+CALPHA(I)*CALPHA(I)
  ENDDO
  MCALPHA=SQRT(TMP)
  ! unit vector of calpha
  DO I=1,3
     CALPHA(I)=CALPHA(I)/MCALPHA
  ENDDO
  !
  IF(PRNLEV >= 6) THEN
     CALL DOTPR(CALPHA,AK,3,TMP)
     WRITE(6,*) 'HERE',ACOS(TMP)*RADDEG
     TMP=0.D0
     DO I=1,3
        TMP=TMP+CALPHA(I)*AK(I)
     ENDDO
     WRITE(6,*) 'HERE2',ACOS(TMP)*RADDEG
  ENDIF
  ! sign of rho angle 
  !  0 <= dvec(x)calpha (dot) ak <= 1 ( <= 90 degs) (-) sign
  ! -1 <= dvec(x)calpha (dot) ak <  0 ( >  90 degs) (+) sign
  !       ddvxcaak 
  CALL CROSS3(DVEC,CALPHA,DVXCA)
  CALL DOTPR(DVXCA,AK,3,DDVXCAAK)
  !
  ! rho angle
  ! cos(rho)= dvec (dot) calpha
  CALL DOTPR(DVEC,CALPHA,3,TMP)
  DVDOTCAL=TMP
  RHOANG=ACOS(TMP)*RADDEG
  PRHOANG=RHOANG

  IF(DDVXCAAK > ZERO) RHOANG=-RHOANG

  call set_param('ROTT',RHOANG)

  IF(PRNLEV >= 6) THEN
     WRITE(6,*) 'RhoAng:',PRHOANG,RHOANG,'RefAng:',ANGL
     WRITE(6,*) 'dvec(dot)ca',DVDOTCAL,MCALPHA*COS(RHOANG*DEGRAD)
  ENDIF
  IF(LHARM) THEN 
     ! harmonic function
     ! U = k ( r - r_0 )^2
     ! (U_kx)' = 2 k ( r - r_0 ) / - sin(r) ( dvec (dot) calpha )' 
     ! preU = - 2 k (r - r_0 ) / sin(r)
     TMP=ABS(RHOANG-ANGL)
     ! due to singlar point near 0.d0 and pi, ignore force near that
     IF((ABS(RHOANG) <= 1.0D-4).OR.(ABS(RHOANG) >= (180.D0-1.0D-4))) THEN
        PREU=0.D0
     ELSE
        IF(TMP >= 180.D0) THEN
           TMP=(180.D0-ABS(RHOANG))+(180.D0-ABS(ANGL))
           IF(ANGL < 0.D0) THEN
              PREU=TWO*AFOR*TMP*DEGRAD/SIN(RHOANG*DEGRAD)
           ELSE
              PREU=-TWO*AFOR*TMP*DEGRAD/SIN(RHOANG*DEGRAD)
           ENDIF
        ELSE
           PREU=-TWO*AFOR*(RHOANG-ANGL)*DEGRAD/SIN(RHOANG*DEGRAD)
        ENDIF
     ENDIF
  ELSE
     ! periodic function
     ! U = -k ( 1+ cos(n r - r_0) ) n = 1
     !   (U)' by x
     ! = k sin(n r -r_0) n r'
     ! = n k sin(nr - r_0) / -sin(r) [ ]
     IF((ABS(RHOANG) <= 1.0D-2).OR.(ABS(RHOANG) >= (180.D0-1.0D-2))) THEN
        PREU=0.D0
     ELSE
        PREU=-AFOR*SIN((RHOANG-ANGL)*DEGRAD)/SIN(RHOANG*DEGRAD)
     ENDIF
  ENDIF
  ! write down to file
  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1
  IF(PRNLEV > 6.AND.DYNAMQ) WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,RHOANG,LQSTPRT,MCOUNT
  IF(CHSTEP > 0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') RHOANG
     ENDIF
  ENDIF
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'eigenvector_pre',o(1,1),o(2,1),o(3,1)
     WRITE(OUTU,*) 'eigenvector_pre2',cavec(1,1),cavec(2,1),cavec(3,1)
     WRITE(OUTU,*) 'eigenvector_pre3',ak(1),ak(2),ak(3)
     WRITE(OUTU,*) 'eigenvalue',ev(1),ev(2),ev(3)
  ENDIF
  !
  IF(LBHG.OR.LBHP) THEN
     ! beta-hairpin mode and general moment of inertia
     CALL ROTINVARG(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO)
  ELSE
     ! default (helix) moment of inertia
     CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ENDIF
  ! k = number of helix ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  !----------------------------------------------------------------------
  ! constant value
  ! first and last atom
  ONUM=1.D0/NSEL(K)
  ! center of geometry: bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)
  !
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'cog',xcg,ycg,zcg
  ! dot product
  ! [] = ak (dot) (alpha-<r>)
  DOTK=AK(1)*(RALPHA(1)-XCG)+AK(2)*(RALPHA(2)-YCG)+AK(3)*(RALPHA(3)-ZCG)
  !-----------------------------------------------------------------------
  ! Main loop
  DO I=1,NSEL(K)
     ! (alpha)'
     ! i=refa    --> 1
     ! otherwise --> 0
     IF(I == REFA) THEN 
        RMK=1.D0
     ELSE 
        RMK=0.D0
     ENDIF
     !  p_a = <r> + [ak(dot)(alpha-<r>)] ak
     ! ([])' by x
     ! = (akx)' (alphax-xcg) + akx ((alphax)' - onum)
     ! + (aky)' (alphay-ycg) 
     ! + (akz)' (alphaz-zcg) 
     DBKINTX(I)=CDXO(K,I,1)*(RALPHA(1)-XCG)+AK(1)*(RMK-ONUM) &
          +CDXO(K,I,2)*(RALPHA(2)-YCG) &
          +CDXO(K,I,3)*(RALPHA(3)-ZCG)
     DBKINTY(I)=CDYO(K,I,1)*(RALPHA(1)-XCG) &
          +CDYO(K,I,2)*(RALPHA(2)-YCG)+AK(2)*(RMK-ONUM) &
          +CDYO(K,I,3)*(RALPHA(3)-ZCG)
     DBKINTZ(I)=CDZO(K,I,1)*(RALPHA(1)-XCG) &
          +CDZO(K,I,2)*(RALPHA(2)-YCG) &
          +CDZO(K,I,3)*(RALPHA(3)-ZCG)+AK(3)*(RMK-ONUM)
     ! (palx)' by x
     ! = 1/nsel(k) + ([])' akx +[] (akx)'
     DBKX(I)=DBKINTX(I)*AK(1)+DOTK*CDXO(K,I,1)+ONUM
     DBKY(I)=DBKINTY(I)*AK(1)+DOTK*CDYO(K,I,1)
     DBKZ(I)=DBKINTZ(I)*AK(1)+DOTK*CDZO(K,I,1)
     ! (paly)' by x
     DBKYX(I)=DBKINTX(I)*AK(2)+DOTK*CDXO(K,I,2)
     DBKYY(I)=DBKINTY(I)*AK(2)+DOTK*CDYO(K,I,2)+ONUM
     DBKYZ(I)=DBKINTZ(I)*AK(2)+DOTK*CDZO(K,I,2)
     ! (palz)' by x
     DBKZX(I)=DBKINTX(I)*AK(3)+DOTK*CDXO(K,I,3)
     DBKZY(I)=DBKINTY(I)*AK(3)+DOTK*CDYO(K,I,3)
     DBKZZ(I)=DBKINTZ(I)*AK(3)+DOTK*CDZO(K,I,3)+ONUM
     ! (1/|alpha-palpha|)' by x 
     ! = - (|alp-palp|)**-3 { ( alx - palx ) ( (alx)' - (palx)' )
     !     + ( aly - paly ) ( - (paly)' )
     !     + ( alz - palz ) ( - (palz)' ) }
     OAALPALX(I)=-((RALPHA(1)-RCPRVEC(1))*(RMK-DBKX(I)) &
          +(RALPHA(2)-RCPRVEC(2))*(-DBKYX(I)) &
          +(RALPHA(3)-RCPRVEC(3))*(-DBKZX(I)))/MCALPHA/MCALPHA/MCALPHA
     OAALPALY(I)=-((RALPHA(1)-RCPRVEC(1))*(-DBKY(I)) &
          +(RALPHA(2)-RCPRVEC(2))*(RMK-DBKYY(I)) &
          +(RALPHA(3)-RCPRVEC(3))*(-DBKZY(I)))/MCALPHA/MCALPHA/MCALPHA
     OAALPALZ(I)=-((RALPHA(1)-RCPRVEC(1))*(-DBKZ(I)) &
          +(RALPHA(2)-RCPRVEC(2))*(-DBKYZ(I)) &
          +(RALPHA(3)-RCPRVEC(3))*(RMK-DBKZZ(I)))/MCALPHA/MCALPHA/MCALPHA
     ! (calx)' by x
     !  = ((alphax)'-(palphax)')/mcalpha + (alphax-palphax)(1/|mcalpha|)'
     DCAX(I)=(RMK-DBKX(I))/MCALPHA+(RALPHA(1)-RCPRVEC(1))*OAALPALX(I)
     DCAY(I)=-DBKY(I)/MCALPHA+(RALPHA(1)-RCPRVEC(1))*OAALPALY(I)
     DCAZ(I)=-DBKZ(I)/MCALPHA+(RALPHA(1)-RCPRVEC(1))*OAALPALZ(I)
     ! (caly)' by x
     !  = -(palphay)' 
     DCAYX(I)=-DBKYX(I)/MCALPHA+(RALPHA(2)-RCPRVEC(2))*OAALPALX(I)
     DCAYY(I)=(RMK-DBKYY(I))/MCALPHA+(RALPHA(2)-RCPRVEC(2))*OAALPALY(I)
     DCAYZ(I)=-DBKYZ(I)/MCALPHA+(RALPHA(2)-RCPRVEC(2))*OAALPALZ(I)
     ! (calz)' by x
     DCAZX(I)=-DBKZX(I)/MCALPHA+(RALPHA(3)-RCPRVEC(3))*OAALPALX(I)
     DCAZY(I)=-DBKZY(I)/MCALPHA+(RALPHA(3)-RCPRVEC(3))*OAALPALY(I)
     DCAZZ(I)=(RMK-DBKZZ(I))/MCALPHA+(RALPHA(3)-RCPRVEC(3))*OAALPALZ(I)
     ! s = sqrt (akx**2+aky**2)
     ! (1/s)' by x
     ! = - (akx*(akx)'+aky*(aky)')/s**3

     ! (1 / |idvec|)' by x
     ! = - 1/midvec**3
     !  * [ az ax {(az)' ax + az (ax)'}
     !     +az ay {(az)' ay + az (ay)'}
     !     +(1-az az)(-2 az (az)') ]

     ! direction vector derivative
     ! dvx = az ax / | idvec |
     ! (dvx)' by x
     ! =  ( (az)' ax + az (ax)' )/ midvec + az ax (1/|idvec|)'
     DDVXX(I)=(2*AK(1)*CDXO(K,I,1)*VZ(1) &
          +(CDXO(K,I,1)*AK(2)+AK(1)*CDXO(K,I,2))*VZ(2) &
          +(CDXO(K,I,1)*AK(3)+AK(1)*CDXO(K,I,3))*VZ(3))/MIDVEC
     DDVXY(I)=(2*AK(1)*CDYO(K,I,1)*VZ(1) &
          +(CDYO(K,I,1)*AK(2)+AK(1)*CDYO(K,I,2))*VZ(2) &
          +(CDYO(K,I,1)*AK(3)+AK(1)*CDYO(K,I,3))*VZ(3))/MIDVEC
     DDVXZ(I)=(2*AK(1)*CDZO(K,I,1)*VZ(1) &
          +(CDZO(K,I,1)*AK(2)+AK(1)*CDZO(K,I,2))*VZ(2) &
          +(CDZO(K,I,1)*AK(3)+AK(1)*CDZO(K,I,3))*VZ(3))/MIDVEC
     ! dvy = az ay / |idvec|
     ! (dvy)' by x
     ! =  ( (az)' ay + az (ay)' )/ midvec + az ay (1/|idvec|)'
     DDVYX(I)=((CDXO(K,I,1)*AK(2)+AK(1)*CDXO(K,I,2))*VZ(1) &
          +2*AK(2)*CDXO(K,I,2)*VZ(2) &
          +(CDXO(K,I,2)*AK(3)+AK(2)*CDXO(K,I,3))*VZ(3))/MIDVEC
     DDVYY(I)=((CDYO(K,I,1)*AK(2)+AK(1)*CDYO(K,I,2))*VZ(1) &
          +2*AK(2)*CDYO(K,I,2)*VZ(2) &
          +(CDYO(K,I,2)*AK(3)+AK(2)*CDYO(K,I,3))*VZ(3))/MIDVEC
     DDVYZ(I)=((CDZO(K,I,1)*AK(2)+AK(1)*CDZO(K,I,2))*VZ(1) &
          +2*AK(2)*CDZO(K,I,2)*VZ(2) &
          +(CDZO(K,I,2)*AK(3)+AK(2)*CDZO(K,I,3))*VZ(3))/MIDVEC
     ! dvz =  (-1+az*az)/|idvec|
     ! (dvz)' by x
     ! = ( 2 az (az)')/midvec + (-1+az*az) (1/|idvec|)'
     DDVZX(I)=((CDXO(K,I,1)*AK(3)+AK(1)*CDXO(K,I,3))*VZ(1) &
          +(CDXO(K,I,2)*AK(3)+AK(2)*CDXO(K,I,3))*VZ(2) &
          +2*AK(3)*CDXO(K,I,3)*VZ(3))/MIDVEC
     DDVZY(I)=((CDYO(K,I,1)*AK(3)+AK(1)*CDYO(K,I,3))*VZ(1) &
          +(CDYO(K,I,2)*AK(3)+AK(2)*CDYO(K,I,3))*VZ(2) &
          +2*AK(3)*CDYO(K,I,3)*VZ(3))/MIDVEC
     DDVZZ(I)=((CDZO(K,I,1)*AK(3)+AK(1)*CDZO(K,I,3))*VZ(1) &
          +(CDZO(K,I,2)*AK(3)+AK(2)*CDZO(K,I,3))*VZ(2) &
          +2*AK(3)*CDZO(K,I,3)*VZ(3))/MIDVEC

     ! dvec (dot) cal = dvx calx + dvy caly + dvz calz
     ! ( )' by x
     ! = (dvx)' calx + dvx (calx)' 
     ! + (dvy)' caly + dvy (caly)' 
     ! + (dvz)' calz + dvz (calz)' 
     DVDCALX(I)=DDVXX(I)*CALPHA(1)+DVEC(1)*DCAX(I) &
          +DDVYX(I)*CALPHA(2)+DVEC(2)*DCAYX(I) &
          +DDVZX(I)*CALPHA(3)+DVEC(3)*DCAZX(I)
     DVDCALY(I)=DDVXY(I)*CALPHA(1)+DVEC(1)*DCAY(I) &
          +DDVYY(I)*CALPHA(2)+DVEC(2)*DCAYY(I) &
          +DDVZY(I)*CALPHA(3)+DVEC(3)*DCAZY(I)
     DVDCALZ(I)=DDVXZ(I)*CALPHA(1)+DVEC(1)*DCAZ(I) &
          +DDVYZ(I)*CALPHA(2)+DVEC(2)*DCAYZ(I) &
          +DDVZZ(I)*CALPHA(3)+DVEC(3)*DCAZZ(I)
     ! (1/|dvec|)' by x
     ! - 1/ |dvec|**3 * 
     ! azax/midvec{((az)'ax+az(ax)')/midvec+azax(1/|idvec|)'}
     ! +azay/midvec{((az)'ay+az(ay)')/midvec+azay(1/|idvec|)'}
     ! +(-1+az*az)/midvec{(2az(az)')/midvec+(-1+azaz)(1/|idvec|)'}
     DABSDVX(I)=-(DVEC(1)*DDVXX(I)+DVEC(2)*DDVYX(I)+DVEC(3)*DDVZX(I))
     DABSDVY(I)=-(DVEC(1)*DDVXY(I)+DVEC(2)*DDVYY(I)+DVEC(3)*DDVZY(I))
     DABSDVZ(I)=-(DVEC(1)*DDVXZ(I)+DVEC(2)*DDVYZ(I)+DVEC(3)*DDVZZ(I))
     ! (1/|cal|)' by x
     ! = - [ (alx-palx)/mcalpha {((alx)'-(palx)')/mcalpha+(alx-palx)(1/|al-pal|)'}
     !     + (aly-paly)/mcalpha {((aly)'-(paly)')/mcalpha+(aly-paly)(1/|al-pal|)'}
     !     + (alz-palz)/mcalpha {((alz)'-(palz)')/mcalpha+(alz-palz)(1/|al-pal|)'} ]
     DOMCALX(I)=-(CALPHA(1)*DCAX(I)+CALPHA(2)*DCAYX(I)+CALPHA(3)*DCAZX(I))
     DOMCALY(I)=-(CALPHA(1)*DCAY(I)+CALPHA(2)*DCAYY(I)+CALPHA(3)*DCAZY(I))
     DOMCALZ(I)=-(CALPHA(1)*DCAZ(I)+CALPHA(2)*DCAYZ(I)+CALPHA(3)*DCAZZ(I))
     ! (rho)' by x    omit  -1/sin(rho)
     ! = (dvec(dot)cal)'/(|dvec||cal|)+(dvec(dot)cal){(1/|dvec|)'/|cal|+(1/|cal|)'/|dvec|}
     DRHOX(I)=DVDCALX(I)+DVDOTCAL*(DABSDVX(I)+DOMCALX(I))
     DRHOY(I)=DVDCALY(I)+DVDOTCAL*(DABSDVY(I)+DOMCALY(I))
     DRHOZ(I)=DVDCALZ(I)+DVDOTCAL*(DABSDVZ(I)+DOMCALZ(I))
     ! (U_kx) = 2 k (r-r_0) * (r)'
     ! = preU * dvdcalx(i)
     DUKX(I)=PREU*DRHOX(I)
     DUKY(I)=PREU*DRHOY(I)
     DUKZ(I)=PREU*DRHOZ(I)
  ENDDO
  IF(LHARM) THEN 
     ! harmonic function
     ! U = k (r-r_0)^2
     TMP=ABS(RHOANG-ANGL)
     IF(TMP >= 300.D0) THEN
        TMP=(180.D0-ABS(RHOANG))+(180.D0-ABS(ANGL))
        EIJ=AFOR*TMP*TMP*DEGRAD*DEGRAD
     ELSE
        EIJ=AFOR*(RHOANG-ANGL)*(RHOANG-ANGL)*DEGRAD*DEGRAD
     ENDIF
  ELSE
     ! periodic function
     ! U = k ( 1- cos(n r - r_0) ) n = 1
     !   (U)' by x
     ! = k sin(n r -r_0) n r'
     ! = n k sin(nr - r_0) / -sin(r) [ ]
     EIJ=AFOR*(ONE-COS((RHOANG-ANGL)*DEGRAD))
  ENDIF
!!!!!Sunhwan
!!!!!
!!!!!
  ! The calculated forces add original ones
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO
  !
  IF(PRNLEV >= 6) THEN
     ! K=1
     WRITE(6,*) 'CPRVEC'
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     !
     WRITE(6,*) 'FORCE'
     ! force = - (U)' wrong
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I),DUKY(I),DUKZ(I)
     ENDDO
  ENDIF
  ! energy
  ECHEL=ENEANG+EIJ
  ENEANG=ECHEL
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(P Ang) E. of #', ' : ',eij
     WRITE(OUTU,*) 'eneang',eneang
     WRITE(OUTU,*) 'Force',-two*afor*(rhoang-angl)*degrad 
  ENDIF
  RETURN
END SUBROUTINE ECHro

SUBROUTINE ECHHA(ECHEL,X,Y,Z,DX,DY,DZ,ANGL,AFOR,NSEL,ASLCT,BSLCT,AMASS, &
     CAVEC,CU,CEV,CBVEC,CEVEC,CPVEC,CHUNIT,CHSTEP,ENEANG,LQSTPRT,MCOUNT, &
     LHMOD)
  !----------------------------------------------------------------------
  !    Helix-helix hinge bending angle restraint routine 
  !    Original Version
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT
  use vector

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)

  real(chm_real) ANGL,AFOR
  real(chm_real) AMASS(*)
  real(chm_real) CAVEC(NCOOR,MXHEL),CBVEC(NCOOR,MXHEL),CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) ENEANG,AKDAJ,THETA,PREU,EIJ
  real(chm_real) AK(NCOOR),AJ(NCOOR)
  real(chm_real) DAKX(MXSLT),DAKY(MXSLT),DAKZ(MXSLT)
  real(chm_real) DAJX(MXSLT),DAJY(MXSLT),DAJZ(MXSLT)
  real(chm_real) dakajx(mxslt),dakajy(mxslt),dakajz(mxslt)
  real(chm_real) dakajxj(mxslt),dakajyj(mxslt),dakajzj(mxslt)
  real(chm_real) DCOSTX(MXSLT),DCOSTY(MXSLT),DCOSTZ(MXSLT)
  real(chm_real) DCOSTXj(MXSLT),DCOSTYj(MXSLT),DCOSTZj(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) DUjX(MXSLT),DUjY(MXSLT),DUjZ(MXSLT)

  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,J,K,KK
  INTEGER NATOMM

  LOGICAL LQSTPRT
  LOGICAL LHMOD
  ! temporal variables
  real(chm_real) SUMX,SUMY,SUMZ,SUM
  real(chm_real) ANL

  ! define hinge angle according to selection mode(CHARMM selection or Hinge mode selection)
  ! first helix
  K=1
  ! second helix
  J=K+1
  ! U = k ( H - H_0 )^2
  ! cos(H) = a_k (dot) a_j / (|a_k| |a_j|)
  DO I=1,3
     AK(I)=CAVEC(I,K)
     AJ(I)=CAVEC(I,J)
  ENDDO
  ! akdaj
  ! = a_k (dot) a_j
  CALL DOTPR(AK,AJ,3,AKDAJ)
  ! unit vector a_k and a_j
  THETA=ACOS(AKDAJ)*RADDEG
  !   
  IF(.NOT.LHMOD) THEN
     IF(PRNLEV >= 6) WRITE(OUTU,*) 'AKDAJ',AKDAJ,180.D0-THETA
  ELSE
     IF(PRNLEV >= 6) WRITE(OUTU,*) 'AKDAJ',AKDAJ,THETA
  ENDIF
  ! (U_kx)' = 2 k ( H - H_0 ) / - sin(H) (cos(H))'
  ! preU = - 2 k ( H - H_0 ) / sin(H)
  IF(.NOT.LHMOD) THEN
     ANL=180.D0-ANGL
     PREU=-TWO*AFOR*(THETA-ANL)*DEGRAD/SIN(THETA*DEGRAD)
  ELSE
     PREU=-TWO*AFOR*(THETA-ANGL)*DEGRAD/SIN(THETA*DEGRAD)
  ENDIF
  ! write down to file
  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1
  IF(.NOT.LHMOD) THEN
     IF(PRNLEV>6.AND.DYNAMQ) WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,180.D0-THETA, &
          LQSTPRT,MCOUNT
  ELSE
     IF(PRNLEV>6.AND.DYNAMQ) WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,THETA,LQSTPRT,MCOUNT
  ENDIF
  IF(CHSTEP > 0) THEN
     IF(.NOT.LHMOD) THEN
        IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
           WRITE(CHUNIT,'(1X,F12.5)') 180.D0-THETA
        ENDIF
     ELSE
        IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
           WRITE(CHUNIT,'(1X,F12.5)') THETA
        ENDIF
     ENDIF
  ENDIF
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'eigenvector_pre',o(1,1),o(2,1),o(3,1)
     WRITE(OUTU,*) 'eigenvector_pre2',cavec(1,1),cavec(2,1),cavec(3,1)
     WRITE(OUTU,*) 'eigenvector_pre3',ak(1),ak(2),ak(3)
     WRITE(OUTU,*) 'eigenvalue',ev(1),ev(2),ev(3)
  ENDIF
  !     
  CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  ! k = number of helix ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  !-----------------------------------------------------------------------
  ! Main loop
  DO I=1,NSEL(K)
     ! debug
     IF(PRNLEV >= 6) WRITE(6,*) 'CDYO',CDYO(K,I,1),CDYO(K,I,2),CDYO(K,I,3)
     ! (|a_k|)'
     ! dakx(i)
     ! = 1 / ( 2 |a_k| ) ( 2 a_kx (a_kx)' + 2 a_ky (a_ky)' + 2 a_kz (a_kz)' )
     DAKX(I)=1.D0*(AK(1)*CDXO(K,I,1)+AK(2)*CDXO(K,I,2)+AK(3)*CDXO(K,I,3))
     DAKY(I)=1.D0*(AK(1)*CDYO(K,I,1)+AK(2)*CDYO(K,I,2)+AK(3)*CDYO(K,I,3))
     DAKZ(I)=1.D0*(AK(1)*CDZO(K,I,1)+AK(2)*CDZO(K,I,2)+AK(3)*CDZO(K,I,3))
     ! (akdaj)'=(a_k (dot) a_j)'
     ! = (a_kx)' a_jx +  (a_ky)' a_jy + (a_kz)' a_jz 
     ! by x in k
     DAKAJX(I)=CDXO(K,I,1)*AJ(1)+CDXO(K,I,2)*AJ(2)+CDXO(K,I,3)*AJ(3)
     DAKAJY(I)=CDYO(K,I,1)*AJ(1)+CDYO(K,I,2)*AJ(2)+CDYO(K,I,3)*AJ(3)
     DAKAJZ(I)=CDZO(K,I,1)*AJ(1)+CDZO(K,I,2)*AJ(2)+CDZO(K,I,3)*AJ(3)
     ! (cos(H))'
     ! = (akdaj)' + (a_k (dot) a_j) (-1) (|a_k| |a_j|)^-2
     !            * ( (|a_k|)' |a_j| + |a_k| (|a_j|)' )
     ! (|a_k|)' = 0 by x_j
     ! (|a_j|)' = 0 by x_k
     DCOSTX(I)=DAKAJX(I)-AKDAJ*DAKX(I)
     DCOSTY(I)=DAKAJY(I)-AKDAJ*DAKY(I)
     DCOSTZ(I)=DAKAJZ(I)-AKDAJ*DAKZ(I)
     ! (U_kx) = 2 k (T-T_0) / - sin(T) (cos(T))'
     ! = preU * dcostx(i)
     DUKX(I)=PREU*DCOSTX(I)
     DUKY(I)=PREU*DCOSTY(I)
     DUKZ(I)=PREU*DCOSTZ(I)
  ENDDO
  !----------------------------------------------------------------------
  ! SECOND HELIX
  K=2
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  !
  CALL ROTINVAR2(K,X,Y,Z,NSEL,BSLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  !
  J=k-1
  !-----------------------------------------------------------------------
  ! Main loop for second helix
  DO I=1,NSEL(K)
     ! (|a_j|)'
     ! dajx(i)
     !    = 1 / ( 2 |a_j| ) ( 2 a_jx (a_jx)' + 2 a_jy (a_jy)' + 2 a_jz (a_jz)' )
     DAJX(I)=1.D0*(AJ(1)*CDXO(K,I,1)+AJ(2)*CDXO(K,I,2)+AJ(3)*CDXO(K,I,3))
     DAJY(I)=1.D0*(AJ(1)*CDYO(K,I,1)+AJ(2)*CDYO(K,I,2)+AJ(3)*CDYO(K,I,3))
     DAJZ(I)=1.D0*(AJ(1)*CDZO(K,I,1)+AJ(2)*CDZO(K,I,2)+AJ(3)*CDZO(K,I,3))
     ! (akdaj)'=(a_k (dot) a_j)'
     !    = (a_jx)' a_kx +  (a_jy)' a_ky + (a_jz)' a_kz 
     ! by x in j
     DAKAJXJ(I)=CDXO(K,I,1)*AK(1)+CDXO(K,I,2)*AK(2)+CDXO(K,I,3)*AK(3)
     DAKAJYJ(I)=CDYO(K,I,1)*AK(1)+CDYO(K,I,2)*AK(2)+CDYO(K,I,3)*AK(3)
     DAKAJZJ(I)=CDZO(K,I,1)*AK(1)+CDZO(K,I,2)*AK(2)+CDZO(K,I,3)*AK(3)
     ! (cos(H))'
     !    = (akdaj)' + (a_k (dot) a_j) (-1) (|a_k| |a_j|)^-2
     !               * ( (|a_k|)' |a_j| + |a_k| (|a_j|)' )
     ! (|a_k|)' = 0 by x_j
     ! (|a_j|)' = 0 by x_k
     DCOSTXJ(I)=DAKAJXJ(I)-AKDAJ*DAJX(I)
     DCOSTYJ(I)=DAKAJYJ(I)-AKDAJ*DAJY(I)
     DCOSTZJ(I)=DAKAJZJ(I)-AKDAJ*DAJZ(I)
     ! (U_kx)' = 2 k (T-T_0) / - sin(T) (cos(T))'
     !         = preU * dcostx(i)
     DUJX(I)=PREU*DCOSTXJ(I)
     DUJY(I)=PREU*DCOSTYJ(I)
     DUJZ(I)=PREU*DCOSTZJ(I)
  ENDDO
  ! U = K (T-T_0)^2
  IF(.NOT.LHMOD) THEN
     ANL=180.D0-ANGL
     EIJ=AFOR*(THETA-ANL)*(THETA-ANL)*DEGRAD*DEGRAD
  ELSE
     EIJ=AFOR*(THETA-ANGL)*(THETA-ANGL)*DEGRAD*DEGRAD
  ENDIF
  ! THE calculated forces add original ones
  K=1
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO
  ! SECOND thing
  K=2
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUJX(I)
     DY(NATOMM)=DY(NATOMM)+DUJY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUJZ(I)
  ENDDO
  ! CHECK routine force direction
  IF(PRNLEV >= 6) THEN
     WRITE(6,*) 'position'
     ! k=1
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO

     WRITE(6,*) 'FORCE'

     SUMX=0.D0
     SUMY=0.D0
     SUMZ=0.D0
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I),DUKY(I),DUKZ(I)
        SUMX=SUMX+DUKX(I)
        SUMY=SUMY+DUKY(I)
        SUMZ=SUMZ+DUKZ(I)
     ENDDO
     SUM=SQRT(SUMX*SUMX+SUMY*SUMY+SUMZ*SUMZ)
     WRITE(6,*) 'SUM,K',SUMX,SUMY,SUMZ,SUM
     !
     SUMX=0.D0
     SUMY=0.D0
     SUMZ=0.D0
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUJX(I),DUJY(I),DUJZ(I)
        SUMX=SUMX+DUJX(I)
        SUMY=SUMY+DUJY(I)
        SUMZ=SUMZ+DUJZ(I)
     ENDDO

     SUM=SQRT(SUMX*SUMX+SUMY*SUMY+SUMZ*SUMZ)
     WRITE(6,*) 'SUM,J',SUMX,SUMY,SUMZ,SUM
  ENDIF
  ! energy
  ECHEL=ENEANG+EIJ
  ENEANG=echel
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(H Ang) E. of #', ' : ',eij
     WRITE(OUTU,*) 'ENEANG',ENEANG
  ENDIF
  RETURN
END SUBROUTINE ECHHA

SUBROUTINE ECHHA2(ECHEL,X,Y,Z,DX,DY,DZ,ANGL,AFOR,NSEL,ASLCT,BSLCT,AMASS, &
     CAVEC,CU,CEV,CBVEC,CEVEC,CPVEC,CHUNIT,CHSTEP,ENEANG,LQSTPRT,MCOUNT)
  !----------------------------------------------------------------------
  !    Helix-helix hinge bending angle restraint routine
  !    same to original one.. partial done (k helix)
  !----------------------------------------------------------------------
  use stream
  use number
  use contrl
  use consta
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT
  use vector

  real(chm_real) ECHEL
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DX(*),DY(*),DZ(*)

  real(chm_real) ANGL,AFOR
  real(chm_real) AMASS(*)
  real(chm_real) CAVEC(NCOOR,MXHEL),CBVEC(NCOOR,MXHEL),CEVEC(NCOOR,MXHEL)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)
  real(chm_real) CU(9,MXHEL),CEV(3,MXHEL)
  real(chm_real) O(3,3),EV(3)
  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) ENEANG,CKDCJ,THETA,PREU,EIJ
  real(chm_real) CK(NCOOR),CJ(NCOOR)
  real(chm_real) DmcKX(MXSLT),DmcKY(MXSLT),DmcKZ(MXSLT)
  real(chm_real) DAJX(MXSLT),DAJY(MXSLT),DAJZ(MXSLT)
  real(chm_real) dckcjx(mxslt),dckcjy(mxslt),dckcjz(mxslt)
  real(chm_real) dakajxj(mxslt),dakajyj(mxslt),dakajzj(mxslt)
  real(chm_real) DCOSTX(MXSLT),DCOSTY(MXSLT),DCOSTZ(MXSLT)
  real(chm_real) DCOSTXj(MXSLT),DCOSTYj(MXSLT),DCOSTZj(MXSLT)
  real(chm_real) DUKX(MXSLT),DUKY(MXSLT),DUKZ(MXSLT)
  real(chm_real) DUjX(MXSLT),DUjY(MXSLT),DUjZ(MXSLT)

  INTEGER CHUNIT,CHSTEP
  INTEGER MCOUNT
  INTEGER NSEL(MXHEL)      
  INTEGER ASLCT(*),BSLCT(*)
  INTEGER I,J,K,KK
  INTEGER NATOMM

  LOGICAL LQSTPRT

  real(chm_real) XCG,YCG,ZCG,ONUM,R1K,RMK,DOTK,DOTKM
  real(chm_real) MCK,MCJ
  INTEGER FSATOM,LSATOM

  real(chm_real) dbkintx(mxslt),dbkinty(mxslt),dbkintz(mxslt)
  real(chm_real) dekintx(mxslt),dekinty(mxslt),dekintz(mxslt)
  real(chm_real) dbkx(mxslt),dbky(mxslt),dbkz(mxslt)
  real(chm_real) dbkyx(mxslt),dbkyy(mxslt),dbkyz(mxslt)
  real(chm_real) dbkzx(mxslt),dbkzy(mxslt),dbkzz(mxslt)
  real(chm_real) dekx(mxslt),deky(mxslt),dekz(mxslt)
  real(chm_real) dekyx(mxslt),dekyy(mxslt),dekyz(mxslt)
  real(chm_real) dekzx(mxslt),dekzy(mxslt),dekzz(mxslt)
  real(chm_real) dckx(mxslt),dcky(mxslt),dckz(mxslt)
  real(chm_real) dckyx(mxslt),dckyy(mxslt),dckyz(mxslt)
  real(chm_real) dckzx(mxslt),dckzy(mxslt),dckzz(mxslt)
  ! tmp
  real(chm_real) ak(ncoor),aj(ncoor)
  real(chm_real) akdaj
  ! first helix
  K=1
  ! second helix
  J=K+1
  ! U = k ( H - H_0 )^2
  ! cos(H) = c_k (dot) c_j / (|c_k| |c_j|)
  DO I=1,3
     CK(I)=CEVEC(I,1)-CBVEC(I,1)
     CJ(I)=CEVEC(I,2)-CBVEC(I,2)
  ENDDO
  ! ckdcj
  ! = c_k (dot) c_j
  CALL DOTPR(CK,CJ,3,CKDCJ)
  ! |c_k|
  MCK=0.D0
  DO I=1,3
     MCK=MCK+CK(I)*CK(I)
  ENDDO
  MCK=SQRT(MCK)
  ! |c_j|
  MCJ=0.D0
  DO I=1,3
     MCJ=MCJ+CJ(I)*CJ(I)
  ENDDO
  MCJ=SQRT(MCJ)
  ! debug
  WRITE(6,*) CEVEC(1,1),CEVEC(2,1),CEVEC(3,1)
  WRITE(6,*) CBVEC(1,1),CBVEC(2,1),CBVEC(3,1)
  WRITE(6,*) CK(1)/MCK,CK(2)/MCK,CK(3)/MCK
  WRITE(6,*) CJ(1)/MCJ,CJ(2)/MCJ,CJ(3)/MCJ
  !
  THETA=ACOS(CKDCJ/MCK/MCJ)*RADDEG
  !   
  IF(PRNLEV >= 6) WRITE(OUTU,*) 'CKDCJ',CKDCJ,THETA
  ! (U_kx)' = 2 k ( H - H_0 ) / - sin(H) (cos(H))'
  ! preU = - 2 k ( H - H_0 ) / sin(H)
  PREU=-TWO*AFOR*(THETA-ANGL)*DEGRAD/SIN(THETA*DEGRAD)
  ! write down to file
  IF(DYNAMQ) MCOUNT=MCOUNT+1
  IF(MDSTEP.LT.MCOUNT) MCOUNT=MCOUNT-1
  IF(PRNLEV > 6.AND.DYNAMQ) WRITE(OUTU,*) 'MDSTEP,LRMDSTEP',MDSTEP,THETA,LQSTPRT,MCOUNT
  IF(CHSTEP > 0) THEN
     IF(DYNAMQ.AND.(MOD(MDSTEP+1,CHSTEP) == 0).AND.(.NOT.LQSTPRT)) THEN
        WRITE(CHUNIT,'(1X,F12.5)') THETA
     ENDIF
  ENDIF
  ! first helix Force (x)
  K=1
  ! get force of principal axis
  ! first helix
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'eigenvector_pre',o(1,1),o(2,1),o(3,1)
     WRITE(OUTU,*) 'eigenvector_pre2',cavec(1,1),cavec(2,1),cavec(3,1)
     WRITE(OUTU,*) 'eigenvector_pre3',ak(1),ak(2),ak(3)
     WRITE(OUTU,*) 'eigenvalue',ev(1),ev(2),ev(3)
  ENDIF
  !
  CALL ROTINVAR2(K,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  !
  ! k = number of helix ; i = selected atom 
  ! cdxo(k,i,1) = (a_kx)' by x ; cdyo(k,i,1) = (a_kxy)' by y ; cdzo(k,i,1) = (a_kxz)' by z
  ! cdxo(k,i,2) = (a_ky)' by x ; cdyo(k,i,2) = (a_kyy)' by y ; cdzo(k,i,2) = (a_kyz)' by z
  ! cdxo(k,i,3) = (a_kz)' by x ; cdyo(k,i,3) = (a_kzy)' by y ; cdzo(k,i,3) = (a_kzz)' by z
  ! eigen vectors
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  !
  ! first and last atom
  FSATOM=ASLCT(1)
  LSATOM=ASLCT(NSEL(K))
  ONUM=1.D0/NSEL(K)
  ! center of geometry: bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(K)
  YCG=YCG/NSEL(K)
  ZCG=ZCG/NSEL(K)
  ! dot product
  ! a_k (dot) ( r_1k - bar(r_k) )
  ! a_kx -> o(1,1), a_ky -> o(2,1), a_kz -> o(3,1)
  DOTK=O(1,1)*(X(FSATOM)-XCG)+O(2,1)*(Y(FSATOM)-YCG)+O(3,1)*(Z(FSATOM)-ZCG)
  ! a_k (dot) ( r_mk - bar(r_k) )
  DOTKM=O(1,1)*(X(LSATOM)-XCG)+O(2,1)*(Y(LSATOM)-YCG)+O(3,1)*(Z(LSATOM)-ZCG)
  ! debug
  WRITE(6,*) 'DOTK',DOTK,DOTKM
  !-----------------------------------------------------------------------
  ! Main loop
  DO I=1,NSEL(K)
     ! (r_1kx)'
     ! i=1       --> 1
     ! otherwise --> 0
     IF(I == 1) THEN 
        R1K=1.D0
     ELSE 
        R1K=0.D0
     ENDIF
     ! (r_mkx)'
     ! i=m       --> 1
     ! otherwise --> 0
     IF(I == NSEL(K)) THEN 
        RMK=1.D0
     ELSE 
        RMK=0.D0
     ENDIF
     !
     natomm=aslct(i)
     ! (a_k (dot) (r_1k - bar(r_k)))'
     ! dbkintx by x
     ! dbkintx = (a_kx)' (r_1kx - xcg) + a_kx ( (r_1kx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_1ky - ycg) + (a_kz)' ( r_1kz - zcg)
     DBKINTX(I)=CDXO(K,I,1)*(X(FSATOM)-XCG)+O(1,1)*(R1K-ONUM ) &
          +CDXO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTY(I)=CDYO(K,I,1)*(X(FSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(FSATOM)-YCG)+O(2,1)*(R1K-ONUM) &
          +CDYO(K,I,3)*(Z(FSATOM)-ZCG)
     DBKINTZ(I)=CDZO(K,I,1)*(X(FSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(FSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(FSATOM)-ZCG)+O(3,1)*(R1K-ONUM)
     ! (a_k (dot) (r_mk - bar(r_k)))'
     ! dekintx
     ! dekintx = (a_kx)' (r_mkx - xcg) + a_kx ( (r_mkx)' - 1 / nsel(k)) 
     !         + (a_ky)' (r_mky - ycg) + (a_kz)' ( r_mkz - zcg)
     DEKINTX(I)=CDXO(K,I,1)*(X(LSATOM)-XCG)+O(1,1)*(RMK-ONUM) &
          +CDXO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDXO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTY(I)=CDYO(K,I,1)*(X(LSATOM)-XCG) &
          +CDYO(K,I,2)*(Y(LSATOM)-YCG)+O(2,1)*(RMK-ONUM) &
          +CDYO(K,I,3)*(Z(LSATOM)-ZCG)
     DEKINTZ(I)=CDZO(K,I,1)*(X(LSATOM)-XCG) &
          +CDZO(K,I,2)*(Y(LSATOM)-YCG) &
          +CDZO(K,I,3)*(Z(LSATOM)-ZCG)+O(3,1)*(RMK-ONUM)
     ! dbk 1 
     ! (b_kxx)' == (b_kx)'
     ! (b_kx)' = 1 / nsel(k) + ( a_k (dot) (r_1k - bar(r_k)))' a_kx
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kx)'
     DBKX(I)=DBKINTX(I)*O(1,1)+DOTK*CDXO(K,I,1)+ONUM
     DBKY(I)=DBKINTY(I)*O(1,1)+DOTK*CDYO(K,I,1)
     DBKZ(I)=DBKINTZ(I)*O(1,1)+DOTK*CDZO(K,I,1)
     ! b_ky by x
     ! (b_kyx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_ky
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kyx)'
     DBKYX(I)=DBKINTX(I)*O(2,1)+DOTK*CDXO(K,I,2)
     DBKYY(I)=DBKINTY(I)*O(2,1)+DOTK*CDYO(K,I,2)+ONUM
     DBKYZ(I)=DBKINTZ(I)*O(2,1)+DOTK*CDZO(K,I,2)
     ! b_kz by x 
     ! (b_kzx)' = ( a_k (dot) (r_1k - bar(r_k)))' a_kz
     !          + ( a_k (dot) (r_1k - bar(r_k))) (a_kzx)'
     DBKZX(I)=DBKINTX(I)*O(3,1)+DOTK*CDXO(K,I,3)
     DBKZY(I)=DBKINTY(I)*O(3,1)+DOTK*CDYO(K,I,3)
     DBKZZ(I)=DBKINTZ(I)*O(3,1)+DOTK*CDZO(K,I,3)+ONUM
     ! dek
     ! (e_kxx)' == (e_kx)'
     ! (e_kx)' = 1 / nsel(k) + (a_k (dot) ( r_mk - bar(r_k)))' a_kx
     !                      + (a_k (dot) ( r_mk - bar(r_k))) (a_kx)'
     DEKX(I)=DEKINTX(I)*O(1,1)+DOTKM*CDXO(K,I,1)+ONUM
     DEKY(I)=DEKINTY(I)*O(1,1)+DOTKM*CDYO(K,I,1)
     DEKZ(I)=DEKINTZ(I)*O(1,1)+DOTKM*CDZO(K,I,1)
     ! e_ky by x
     ! (e_kyx)' = (a_k (dot) (r_mk - bar(r_k)))' a_ky
     !           + ( a_k (dot) (r_mk -bar(r_k))) (a_kyx)'
     DEKYX(I)=DEKINTX(I)*O(2,1)+DOTKM*CDXO(K,I,2)
     DEKYY(I)=DEKINTY(I)*O(2,1)+DOTKM*CDYO(K,I,2)+ONUM
     DEKYZ(I)=DEKINTZ(I)*O(2,1)+DOTKM*CDZO(K,I,2)
     ! e_kz by x
     ! (e_kzx)' = (a_k (dot) (r_mk - bar(r_k)))' a_kz
     !           + ( a_k (dot) (r_mk -bar(r_k))) (a_kzx)'
     DEKZX(I)=DEKINTX(I)*O(3,1)+DOTKM*CDXO(K,I,3)
     DEKZY(I)=DEKINTY(I)*O(3,1)+DOTKM*CDYO(K,I,3)
     DEKZZ(I)=DEKINTZ(I)*O(3,1)+DOTKM*CDZO(K,I,3)+ONUM
     ! ckx by x = (ckx)'
     ! = (ekx-bkx) by x
     ! = (ekx)' - (bkx)'
     DCKX(I)=DEKX(I)-DBKX(I)
     DCKY(I)=DEKY(I)-DBKY(I)
     DCKZ(I)=DEKZ(I)-DBKZ(I)
     ! cky by x = (cky)'
     DCKYX(I)=DEKYX(I)-DBKYX(I)
     DCKYY(I)=DEKYY(I)-DBKYY(I)
     DCKYZ(I)=DEKYZ(I)-DBKYZ(I)
     ! ckz by x = (ckz)'
     DCKZX(I)=DEKZX(I)-DBKZX(I)
     DCKZY(I)=DEKZY(I)-DBKZY(I)
     DCKZZ(I)=DEKZZ(I)-DBKZZ(I)
     ! (|c_k|)' by x in k
     ! dmckx(i)
     ! = 1 / ( 2 |c_k| ) ( 2 c_kx (c_kx)' + 2 c_ky (c_ky)' + 2 c_kz (c_kz)' )
     DMCKX(I)=1.D0/MCK*(CK(1)*DCKX(I)+CK(2)*DCKYX(I)+CK(3)*DCKZX(I))
     DMCKY(I)=1.D0/MCK*(CK(1)*DCKY(I)+CK(2)*DCKYY(I)+CK(3)*DCKZY(I))
     DMCKZ(I)=1.D0/MCK*(CK(1)*DCKZ(I)+CK(2)*DCKYZ(I)+CK(3)*DCKZZ(I))
     ! (ckdcj)'=(c_k (dot) c_j)'
     ! = (c_kx)' c_jx +  (c_ky)' c_jy + (c_kz)' c_jz 
     ! by x in k
     DCKCJX(I)=DCKX(I)*CJ(1)+DCKYX(I)*CJ(2)+DCKZX(I)*CJ(3)
     DCKCJY(I)=DCKY(I)*CJ(1)+DCKYY(I)*CJ(2)+DCKZY(I)*CJ(3)
     DCKCJZ(I)=DCKZ(I)*CJ(1)+DCKYZ(I)*CJ(2)+DCKZZ(I)*CJ(3)
     ! (cos(H))'
     ! = (ckdcj)'/(|c_k| |c_j|) + (c_k (dot) c_j) (-1) (|c_k| |c_j|)^-2
     !            * ( (|c_k|)' |c_j| + |c_k| (|c_j|)' )
     ! (|c_k|)' = 0 by x_j
     ! (|c_j|)' = 0 by x_k
     DCOSTX(I)=DCKCJX(I)/(MCK*MCJ)-CKDCJ/(MCK*MCJ*MCK*MCJ)*DMCKX(I)*MCJ
     DCOSTY(I)=DCKCJY(I)/(MCK*MCJ)-CKDCJ/(MCK*MCJ*MCK*MCJ)*DMCKY(I)*MCJ
     DCOSTZ(I)=DCKCJZ(I)/(MCK*MCJ)-CKDCJ/(MCK*MCJ*MCK*MCJ)*DMCKZ(I)*MCJ
     ! (U_kx) = 2 k (T-T_0) / - sin(T) (cos(T))'
     ! = preU * dcostx(i)
     DUKX(I)=PREU*DCOSTX(I)
     DUKY(I)=PREU*DCOSTY(I)
     DUKZ(I)=PREU*DCOSTZ(I)
  ENDDO
  !----------------------------------------------------------------------
  ! second helix
  K=2
  DO I=1,3
     DO J=1,3
        KK=J+3*(I-1)
        O(J,I)=CU(KK,K)
     ENDDO
     EV(I)=CEV(I,K)
  ENDDO
  !
  CALL ROTINVAR2(K,X,Y,Z,NSEL,BSLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  !
  J=K-1
  !-----------------------------------------------------------------------
  ! Main loop for second helix
  DO I=1,NSEL(K)
     ! (|a_j|)'
     ! dajx(i)
     ! = 1 / ( 2 |a_j| ) ( 2 a_jx (a_jx)' + 2 a_jy (a_jy)' + 2 a_jz (a_jz)' )
     DAJX(I)=1.D0*(AJ(1)*CDXO(K,I,1)+AJ(2)*CDXO(K,I,2)+AJ(3)*CDXO(K,I,3))
     DAJY(I)=1.D0*(AJ(1)*CDYO(K,I,1)+AJ(2)*CDYO(K,I,2)+AJ(3)*CDYO(K,I,3))
     DAJZ(I)=1.D0*(AJ(1)*CDZO(K,I,1)+AJ(2)*CDZO(K,I,2)+AJ(3)*CDZO(K,I,3))
     ! (akdaj)'=(a_k (dot) a_j)'
     ! = (a_jx)' a_kx +  (a_jy)' a_ky + (a_jz)' a_kz 
     ! by x in j
     DAKAJXJ(I)=CDXO(K,I,1)*AK(1)+CDXO(K,I,2)*AK(2)+CDXO(K,I,3)*AK(3)
     DAKAJYJ(I)=CDYO(K,I,1)*AK(1)+CDYO(K,I,2)*AK(2)+CDYO(K,I,3)*AK(3)
     DAKAJZJ(I)=CDZO(K,I,1)*AK(1)+CDZO(K,I,2)*AK(2)+CDZO(K,I,3)*AK(3)
     ! (cos(H))'
     ! = (akdaj)' + (a_k (dot) a_j) (-1) (|a_k| |a_j|)^-2
     !            * ( (|a_k|)' |a_j| + |a_k| (|a_j|)' )
     ! (|a_k|)' = 0 by x_j
     ! (|a_j|)' = 0 by x_k
     DCOSTXJ(I)=DAKAJXJ(I)-AKDAJ*DAJX(I)
     DCOSTYJ(I)=DAKAJYJ(I)-AKDAJ*DAJY(I)
     DCOSTZJ(I)=DAKAJZJ(I)-AKDAJ*DAJZ(I)
     ! (U_kx)' = 2 k (T-T_0) / - sin(T) (cos(T))'
     !         = preU * dcostx(i)
     DUJX(I)=PREU*DCOSTXJ(I)
     DUJY(I)=PREU*DCOSTYJ(I)
     DUJZ(I)=PREU*DCOSTZJ(I)
  ENDDO
  ! U = k (T-T_0)^2
  EIJ=AFOR*(THETA-ANGL)*(THETA-ANGL)*DEGRAD*DEGRAD
  ! The calculated forces add original ones
  K=1
  DO I=1,NSEL(K)
     NATOMM=ASLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUKX(I)
     DY(NATOMM)=DY(NATOMM)+DUKY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUKZ(I)
  ENDDO
  ! SECOND THING
  K=2
  DO I=1,NSEL(K)
     NATOMM=BSLCT(I)
     DX(NATOMM)=DX(NATOMM)+DUJX(I)
     DY(NATOMM)=DY(NATOMM)+DUJY(I)
     DZ(NATOMM)=DZ(NATOMM)+DUJZ(I)
  ENDDO
  ! check routine force direction
  IF(PRNLEV >= 6) THEN
     WRITE(6,*) 'POSITION'
     ! K=1
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') X(NATOMM),Y(NATOMM),Z(NATOMM)
     ENDDO
     WRITE(6,*) 'FORCE'
     DO I=1,NSEL(1)
        NATOMM=ASLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUKX(I)+X(NATOMM),DUKY(I)+Y(NATOMM),DUKZ(I)+Z(NATOMM)
     ENDDO
     DO I=1,NSEL(2)
        NATOMM=BSLCT(I)
        WRITE(6,'(3(1X,F10.5))') DUJX(I)+X(NATOMM),DUJY(I)+Y(NATOMM),DUJZ(I)+Z(NATOMM)
     ENDDO
  ENDIF
  ! energy
  ECHEL=ENEANG+EIJ
  ENEANG=ECHEL
  ! debug
  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Const(H Ang) E. of #', ' : ',eij
     WRITE(OUTU,*) 'eneang',eneang
  ENDIF
  RETURN
END SUBROUTINE ECHHA2

SUBROUTINE CALHELC(X,Y,Z,NUM)
  !----------------------------------------------------------------------
  !     select and calc helix properties
  !
  !----------------------------------------------------------------------
  use dimens_fcm
  use stream
  use psf
  use conshelix_fcm
  use helix,only:helix2

  real(chm_real) X(*),Y(*),Z(*)
  INTEGER NUM
  INTEGER OPRNLEV
  ! define printlev to 0 in ordet to prevent printout of helix
  OPRNLEV=PRNLEV
  PRNLEV=0

  IF(PRNLEV >= 6) THEN
     WRITE(OUTU,*) 'Helix information: ',num, ' of total: ',chnum
  ENDIF
  IF(LOHEL(NUM)) THEN
     ! (single helix selection)
     CALL HELIX2(1,SSLCT(num,:),JJSLCT(num,:),NATOM,X,Y,Z)
  ELSE
     ! get helix information (Double selection)
     CALL HELIX2(2,SSLCT(num,:),JJSLCT(num,:),NATOM,X,Y,Z)
  ENDIF
  ! return prnlev to original
  PRNLEV=OPRNLEV
  RETURN
END SUBROUTINE calhelc

SUBROUTINE DANSA(SLCT,ASLCT,NATOM)
  !----------------------------------------------------------------------
  !     return the atom numbers of the selected atoms
  !
  !----------------------------------------------------------------------
  INTEGER SLCT(*),ASLCT(*),NATOM
  INTEGER I,NSEL

  NSEL=0
  DO I=1,NATOM
     IF(SLCT(I) == 1) THEN
        NSEL=NSEL+1
        ASLCT(NSEL)=I
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE DANSA

SUBROUTINE DANSAD(SLCT,ASLCT,NATOM)
  !----------------------------------------------------------------------
  !     return the atom numbers of the selected atoms
  !     for DNA selection
  !----------------------------------------------------------------------
  INTEGER SLCT(*),ASLCT(*),NATOM
  INTEGER I,K,NSEL

  NSEL=0
  DO I=1,NATOM
     IF(SLCT(I) == 1) THEN
        NSEL=NSEL+1
        ASLCT(NSEL)=I
     ENDIF
  ENDDO

  IF(MOD(NSEL,2) == 1) THEN
     CALL WRNDIE(-3,'HLXSPC','Number of selected atoms for DNA should be even Num')
     STOP
  ENDIF

  NSEL=NSEL/2

  DO I=1,NSEL
     IF(MOD(I,2) == 0) THEN
        K=(NSEL*2-1)-(I-2)
        ASLCT(I)=ASLCT(K)
     ELSE
        ASLCT(I)=ASLCT(I)
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE DANSAD

SUBROUTINE DANSAHP(SLCT,ASLCT,NATOM)
  !----------------------------------------------------------------------
  !     return the atom numbers of the selected atoms
  !     for beta-hairpin
  !----------------------------------------------------------------------
  INTEGER SLCT(*),ASLCT(*),NATOM
  INTEGER I,NSEL
  INTEGER,PARAMETER :: MXHEL=2,NCOOR=3,MXSLT=100
  INTEGER K,TMP(MXSLT)

  NSEL=0
  DO I=1,NATOM
     IF(SLCT(I) == 1) THEN
        NSEL=NSEL+1
        ASLCT(NSEL)=I
     ENDIF
  ENDDO
  IF(MOD(NSEL,2) == 1) THEN
     CALL WRNDIE(-3,'HLXSPC','Number of selected atoms for hairpin should be even Number')
     STOP
  ENDIF
  DO I=1,NSEL
     TMP(I)=ASLCT(I)
  ENDDO
  DO I=1,NSEL
     IF(MOD(I,2) == 0) THEN
        K=NSEL-(I-2)/2
        ASLCT(I)=TMP(K)
     ELSE
        K=I-(I-1)/2
        ASLCT(I)=TMP(K)
     ENDIF
  ENDDO
  RETURN
END SUBROUTINE DANSAHP

SUBROUTINE CKTHEL(ASLCT,BSLCT,NSEL,LOHEL)
  !----------------------------------------------------------------------
  !
  !----------------------------------------------------------------------
  use conshelix_fcm, only: MXHEL

  INTEGER ASLCT(*),BSLCT(*)
  INTEGER NSEL(MXHEL)
  INTEGER I,J,K
  LOGICAL LOHEL

  LOHEL=.TRUE.
  IF(NSEL(1) == NSEL(2)) THEN
     DO I=1,NSEL(1)
        IF(LOHEL.AND.(ASLCT(I) == BSLCT(I))) THEN
           LOHEL=.TRUE.
        ELSE
           LOHEL=.FALSE.
        ENDIF
     ENDDO
  ELSE
     LOHEL=.FALSE.
  ENDIF
  RETURN
END SUBROUTINE CKTHEL

SUBROUTINE ROTINVARg(NUMK,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO)
  !----------------------------------------------------------------------
  !    derivative of general moment of inertia
  !----------------------------------------------------------------------
  use consta
  use number

  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) O(3,3),EV(3),AMASS(*) 
  ! increase mxslt from 50 to 100
  INTEGER,PARAMETER :: MXHEL=2,NCOOR=3,MXSLT=100

  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)

  INTEGER NSEL(MXHEL)
  INTEGER ASLCT(*)
  INTEGER NUMK
  ! LOCAL
  real(chm_real) DXM(6),DYM(6),DZM(6),DRP(3,3)
  real(chm_real) OT(3,3),OMO(3,3),DXO(3,3),DYO(3,3),DZO(3,3),OM(3,3)
  real(chm_real) ALPHA,BETA,GAMMA
  real(chm_real) XCM,YCM,ZCM,TMS,XX,XY,XZ,YY,YZ,ZZ,DCM,AMS
  real(chm_real) DXXX,DXXY,DXXZ,DXYY,DXYZ,DXZZ
  real(chm_real) DYXX,DYXY,DYXZ,DYYY,DYYZ,DYZZ
  real(chm_real) DZXX,DZXY,DZXZ,DZYY,DZYZ,DZZZ
  real(chm_real) XA,YA,ZA,XCG,YCG,ZCG
  real(chm_real) ONUM,DSP
  real(chm_real) DXXXC,DXXYC,DXXZC,DXYYC,DXYZC,DXZZC
  real(chm_real) DYXXC,DYXYC,DYXZC,DYYYC,DYYZC,DYZZC
  real(chm_real) DZXXC,DZXYC,DZXZC,DZYYC,DZYZC,DZZZC
  real(chm_real) DP(MXSLT-2)

  INTEGER I,II,J,JJ,K,L
  INTEGER NATOMM
  !
  ! tranpose of principal axes matrix
  CALL TRANSPS(OT,O,3,3)
  ! center of geometry = bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(NUMK)
     NATOMM=ASLCT(I)
     XCG=XCG+X(NATOMM)
     YCG=YCG+Y(NATOMM)
     ZCG=ZCG+Z(NATOMM)
  ENDDO
  XCG=XCG/NSEL(NUMK)
  YCG=YCG/NSEL(NUMK)
  ZCG=ZCG/NSEL(NUMK)
  !      do I=mynodp,nsel(numk),numnod
  DO I=1,NSEL(NUMK)
     NATOMM=ASLCT(I)
     XA=X(NATOMM)
     YA=Y(NATOMM)
     ZA=Z(NATOMM)
     ! M by x
     dxXXc=ZERO 
     DXXYC=-(YA-YCG)
     DXXZC=-(ZA-ZCG)
     DXYYC=TWO*(XA-XCG)
     DXYZC=ZERO
     DXZZC=TWO*(XA-XCG)
     ! M by y
     DYXXC=TWO*(YA-YCG)
     DYXYC=-(XA-XCG) 
     DYXZC=ZERO
     DYYYC=ZERO
     DYYZC=-(ZA-ZCG)
     DYZZC=TWO*(YA-YCG)
     ! M by z
     DZXXC=TWO*(ZA-ZCG)
     DZXYC=ZERO
     DZXZC=-(XA-XCG) 
     DZYYC=TWO*(ZA-ZCG)
     DZYZC=-(YA-YCG) 
     DZZZC=ZERO

     DXM(1)=DXXXC
     DXM(2)=DXXYC
     DXM(3)=DXXZC
     DXM(4)=DXYYC
     DXM(5)=DXYZC
     DXM(6)=DXZZC

     DYM(1)=DYXXC
     DYM(2)=DYXYC
     DYM(3)=DYXZC
     DYM(4)=DYYYC
     DYM(5)=DYYZC
     DYM(6)=DYZZC

     DZM(1)=DZXXC
     DZM(2)=DZXYC
     DZM(3)=DZXZC
     DZM(4)=DZYYC
     DZM(5)=DZYZC
     DZM(6)=DZZZC
     ! calculate d(xyz)O from O(T) x d(xyz)M x O through Eqs. (16), (18), and (19)
     ! dxO
     CALL MULNXNFU(OM,OT,dxM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)= ZERO
     DRP(1,2)= GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)= ZERO
     DRP(2,3)= ALPHA
     DRP(3,1)= BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)= ZERO
     CALL MULNXN(DXO,O,DRP,3)
     ! dyO
     CALL MULNXNFU(OM,OT,DYM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)=ZERO
     DRP(1,2)=GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)=ZERO
     DRP(2,3)=ALPHA
     DRP(3,1)=BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)=ZERO
     CALL MULNXN(DYO,O,DRP,3)
     ! dzO
     CALL MULNXNFU(OM,OT,DZM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)=ZERO
     DRP(1,2)=GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)=ZERO
     DRP(2,3)=ALPHA
     DRP(3,1)=BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)=ZERO
     CALL MULNXN(DZO,O,DRP,3)
     ! designate number of selected helix
     K=NUMK
     DO L=1,3
        CDXO(K,I,L)=DXO(L,1)
        CDYO(K,I,L)=DYO(L,1)
        CDZO(K,I,L)=DZO(L,1)
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE ROTINVARg

SUBROUTINE ROTINVAR2(NUMK,X,Y,Z,NSEL,ASLCT,AMASS,O,EV,CDXO,CDYO,CDZO,CPVEC)
  !----------------------------------------------------------------------
  !  force contributions due to rotation of integration points
  !  first developed by Wonpil Im ( GB_ROTINVAR2 in gbsw.src )
  !  modified for diagonalization differential
  !----------------------------------------------------------------------
  use consta
  use number
  use conshelix_fcm, only: MXHEL, NCOOR, MXSLT

  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) O(3,3),EV(3),AMASS(*) 

  real(chm_real) CDXO(MXHEL,MXSLT,3),CDYO(MXHEL,MXSLT,3),CDZO(MXHEL,MXSLT,3)
  real(chm_real) CPVEC(NCOOR,MXSLT-2,MXHEL)

  INTEGER NSEL(MXHEL)
  INTEGER ASLCT(*)
  INTEGER NUMK
  ! LOCAL
  real(chm_real) DXM(6),DYM(6),DZM(6),DRP(3,3)
  real(chm_real) OT(3,3),OMO(3,3),DXO(3,3),DYO(3,3),DZO(3,3),OM(3,3)
  real(chm_real) ALPHA,BETA,GAMMA,ONUM,DSP
  real(chm_real) XCM,YCM,ZCM,TMS,XX,XY,XZ,YY,YZ,ZZ,DCM,AMS
  real(chm_real) DXXX,DXXY,DXXZ,DXYY,DXYZ,DXZZ
  real(chm_real) DYXX,DYXY,DYXZ,DYYY,DYYZ,DYZZ
  real(chm_real) DZXX,DZXY,DZXZ,DZYY,DZYZ,DZZZ
  real(chm_real) XA,YA,ZA,XCG,YCG,ZCG
  real(chm_real) DXXXC,DXXYC,DXXZC,DXYYC,DXYZC,DXZZC
  real(chm_real) DYXXC,DYXYC,DYXZC,DYYYC,DYYZC,DYZZC
  real(chm_real) DZXXC,DZXYC,DZXZC,DZYYC,DZYZC,DZZZC
  real(chm_real) DP(MXSLT-2)

  INTEGER I,II,J,JJ,K,L
  INTEGER NATOMM
  ! tranpose of principal axes matrix
  call TRANSPS(OT,O,3,3)
  ! center of geometry = bar(r_k)
  XCG=ZERO
  YCG=ZERO
  ZCG=ZERO
  DO I=1,NSEL(NUMK)-2
     XCG=XCG+CPVEC(1,I,NUMK)
     YCG=YCG+CPVEC(2,I,NUMK)
     ZCG=ZCG+CPVEC(3,I,NUMK)
  ENDDO
  XCG=XCG/(NSEL(NUMK)-TWO)
  YCG=YCG/(NSEL(NUMK)-TWO)
  ZCG=ZCG/(NSEL(NUMK)-TWO)
  !      do I=mynodp,nsel(numk),numnod
  DO I=1,NSEL(NUMK)
     NATOMM=ASLCT(I)
     XA=X(NATOMM)
     YA=Y(NATOMM)
     ZA=Z(NATOMM)

     DO II=1,NSEL(NUMK)-2
        DP(II)=0.D0
     ENDDO
     DSP=0.D0
     ! dp
     ! i=1 by x_1
     IF(I == 1) THEN
        K=I
        DP(K)=1.D0
     ENDIF
     ! i=2 by x_2
     IF(I == 2) THEN
        K=I-1
        DP(K)=-2.D0
        K=I
        DP(K)=1.D0
     ENDIF
     ! i=n-1 by x_n-1
     IF(I == (NSEL(NUMK)-1)) THEN
        K=I-2
        DP(K)=1.D0
        K=I-1
        DP(K)=-2.D0
     ENDIF
     ! i=n
     IF(I == (NSEL(NUMK))) THEN
        K=I-2
        DP(K)=1.D0
     ENDIF
     ! 3 <= i <= n-2
     IF((I >= 3).AND.(I <= (NSEL(NUMK)-2))) THEN
        K=I-2
        DP(K)=1.D0
        K=I-1
        DP(K)=-2.D0
        K=I
        DP(K)=1.D0
     ENDIF
     ! dsp
     IF((I == 1).OR.(I == NSEL(NUMK))) THEN
        DSP=1.D0
     ENDIF
     IF((I == 2).OR.(I == (NSEL(NUMK)-1))) THEN
        DSP=-1.D0
     ENDIF
     ! M by x
     DXXXC=ZERO
     DXXYC=ZERO
     DXXZC=ZERO
     DXYYC=ZERO
     DXYZC=ZERO
     DXZZC=ZERO

     DO II=1,NSEL(NUMK)-2
        DXXXC=DXXXC+TWO*(CPVEC(1,II,NUMK)-XCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
        DXXYC=DXXYC+(DP(II)-DSP/(NSEL(NUMK)-TWO))*(CPVEC(2,II,NUMK)-YCG)
        DXXZC=DXXZC+(DP(II)-DSP/(NSEL(NUMK)-TWO))*(CPVEC(3,II,NUMK)-ZCG)
     ENDDO
     ! M by y
     DYXXC=ZERO
     DYXYC=ZERO
     DYXZC=ZERO
     DYYYC=ZERO
     DYYZC=ZERO
     DYZZC=ZERO

     DO II=1,NSEL(NUMK)-2
        DYXYC=DYXYC+(CPVEC(1,II,NUMK)-XCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
        DYYYC=DYYYC+TWO*(CPVEC(2,II,NUMK)-YCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
        DYYZC=DYYZC+(CPVEC(3,II,NUMK)-ZCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
     ENDDO
     ! M by z
     DZXXC=ZERO
     DZXYC=ZERO
     DZXZC=ZERO
     DZYYC=ZERO
     DZYZC=ZERO
     DZZZC=ZERO

     DO II=1,NSEL(NUMK)-2
        DZXZC=DZXZC+(CPVEC(1,II,NUMK)-XCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
        DZYZC=DZYZC+(CPVEC(2,II,NUMK)-YCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
        DZZZC=DZZZC+TWO*(CPVEC(3,II,NUMK)-ZCG)*(DP(II)-DSP/(NSEL(NUMK)-TWO))
     ENDDO

     DXM(1)=DXXXC
     DXM(2)=DXXYC
     DXM(3)=DXXZC
     DXM(4)=DXYYC
     DXM(5)=DXYZC
     DXM(6)=DXZZC

     DYM(1)=DYXXC
     DYM(2)=DYXYC
     DYM(3)=DYXZC
     DYM(4)=DYYYC
     DYM(5)=DYYZC
     DYM(6)=DYZZC

     DZM(1)=DZXXC
     DZM(2)=DZXYC
     DZM(3)=DZXZC
     DZM(4)=DZYYC
     DZM(5)=DZYZC
     DZM(6)=DZZZC
     ! calculate d(xyz)O from O(T) x d(xyz)M x O through Eqs. (16), (18), and (19)
     ! dxO
     CALL MULNXNFU(OM,OT,DXM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)= ZERO
     DRP(1,2)= GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)= ZERO
     DRP(2,3)= ALPHA
     DRP(3,1)= BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)= ZERO

     CALL MULNXN(DXO,O,DRP,3)
     ! dyO
     CALL MULNXNFU(OM,OT,DYM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)= ZERO
     DRP(1,2)= GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)= ZERO
     DRP(2,3)= ALPHA
     DRP(3,1)= BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)= ZERO
     CALL MULNXN(DYO,O,DRP,3)
     ! dzO
     CALL MULNXNFU(OM,OT,DZM,3)
     CALL MULNXN(OMO,OM,O,3)
     ALPHA=-OMO(2,3)/(EV(2)-EV(3))
     BETA =-OMO(1,3)/(EV(3)-EV(1))
     GAMMA=-OMO(1,2)/(EV(1)-EV(2))
     DRP(1,1)= ZERO
     DRP(1,2)= GAMMA
     DRP(1,3)=-BETA
     DRP(2,1)=-GAMMA
     DRP(2,2)= ZERO
     DRP(2,3)= ALPHA
     DRP(3,1)= BETA
     DRP(3,2)=-ALPHA
     DRP(3,3)= ZERO
     CALL MULNXN(DZO,O,DRP,3)
     !designate number of selected helix
     K=NUMK
     DO L=1,3
        CDXO(K,I,L)=DXO(L,1)
        CDYO(K,I,L)=DYO(L,1)
        CDZO(K,I,L)=DZO(L,1)
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE ROTINVAR2

#endif /* (conshelix)*/
end module conshelix_m

