!CHARMM Element source/csa/csacntrl.src $Revision: 1.2 $
!
SUBROUTINE CSACNTRL(COMLYN,COMLEN)
  !----------------------------------------------------------------------
  !
  ! Main control routine for CSA commands
  !
  ! Written in October 2007 by Jinwoo Lee
  ! Code design by Bernie Brooks
  !
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use number
  use psf
  use coord
  use stream
  use string
  use parallel
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
#if KEY_CSA==1 /*csa_cntrl*/
  !

  INTEGER NCONF,NBANKM,JSTART,JEND,MOV1,MOV2,MOV3,MOV4,IS1,IS2
  INTEGER RAN0,RAN1,IRR,NSEED,NTOTAL
  real(chm_real)  CUT1,CUT2,ESTOp
  INTEGER ICMAX
  LOGICAL QDISTM,QCOMMC
  !
  CHARACTER(len=4) WRD
  !
  ! Parse all command line options      
  NCONF =GTRMI(COMLYN,COMLEN,'NCON',50)
  NBANKM=GTRMI(COMLYN,COMLEN,'NBAN',0)
  JSTART=GTRMI(COMLYN,COMLEN,'JSTA',0)
  JEND=  GTRMI(COMLYN,COMLEN,'JEND',0)
  MOV1=  GTRMI(COMLYN,COMLEN,'MOV1',0)
  MOV2=  GTRMI(COMLYN,COMLEN,'MOV2',0)
  MOV3=  GTRMI(COMLYN,COMLEN,'MOV3',0)
  MOV4=  GTRMI(COMLYN,COMLEN,'MOV4',0)
  IS1=   GTRMI(COMLYN,COMLEN,'IS1',0)
  IS2=   GTRMI(COMLYN,COMLEN,'IS2',0)
  RAN0=  GTRMI(COMLYN,COMLEN,'RAN0',0)
  RAN1=  GTRMI(COMLYN,COMLEN,'RAN1',0)
  IRR=   GTRMI(COMLYN,COMLEN,'IRR',0)
  NSEED= GTRMI(COMLYN,COMLEN,'NSEE',0)
  NTOTAL=GTRMI(COMLYN,COMLEN,'NTOT',0)
  CUT1=  GTRMF(COMLYN,COMLEN,'CUT1',ZERO)
  CUT2=  GTRMF(COMLYN,COMLEN,'CUT2',ZERO)
  ESTOP= GTRMF(COMLYN,COMLEN,'NTOT',ZERO)
  ICMAX= GTRMI(COMLYN,COMLEN,'ICMA',0)
  WRD=GTRMA(COMLYN,COMLEN,'DIST')
  IF(WRD.EQ.'ANGL') THEN
     QDISTM=.FALSE.
  ELSEIF (WRD.EQ.'RMDS') THEN
     QDISTM=.TRUE.
  ELSE 
     CALL WRNDIE(-1,'<CSACNTRL>','Unrecognized DISTance option')
  ENDIF
  WRD=GTRMA(COMLYN,COMLEN,'COMM')
  IF(WRD.EQ.'IC') THEN
     QCOMMC=.FALSE.
  ELSEIF (WRD.EQ.'COOR') THEN
     QCOMMC=.TRUE.
  ELSE
     CALL WRNDIE(-1,'<CSACNTRL>','Unrecognized COMMunication option')
  ENDIF
  ! End of keyword parsing.
  !
#endif /* (csa_cntrl)*/
  !
  RETURN
END SUBROUTINE CSACNTRL

