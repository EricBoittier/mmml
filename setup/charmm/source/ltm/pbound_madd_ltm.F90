module pbound_madd
  use chm_kinds
  use dimens_fcm
      integer madd_pb(27,27)
! Interior, no images needed, translate vecotr is 14, (0,0,0) PB_INTERIOR=14,
      data &
           madd_pb(1,14)/14/,madd_pb(2,14)/14/,madd_pb(3,14)/14/, &
           madd_pb(4,14)/14/,madd_pb(5,14)/14/,madd_pb(6,14)/14/, &
           madd_pb(7,14)/14/,madd_pb(8,14)/14/,madd_pb(9,14)/14/, &
           madd_pb(10,14)/14/,madd_pb(11,14)/14/,madd_pb(12,14)/14/, &
           madd_pb(13,14)/14/,madd_pb(14,14)/14/,madd_pb(15,14)/14/, &
           madd_pb(16,14)/14/,madd_pb(17,14)/14/,madd_pb(18,14)/14/, &
           madd_pb(19,14)/14/,madd_pb(20,14)/14/,madd_pb(21,14)/14/, &
           madd_pb(22,14)/14/,madd_pb(23,14)/14/,madd_pb(24,14)/14/, &
           madd_pb(25,14)/14/,madd_pb(26,14)/14/,madd_pb(27,14)/14/
! Face xmin PB_FACEXMN =13
      data &
           madd_pb(1,13)/13/,madd_pb(2,13)/14/,madd_pb(3,13)/14/, &
           madd_pb(4,13)/13/,madd_pb(5,13)/14/,madd_pb(6,13)/14/, &
           madd_pb(7,13)/13/,madd_pb(8,13)/14/,madd_pb(9,13)/14/, &
           madd_pb(10,13)/13/,madd_pb(11,13)/14/,madd_pb(12,13)/14/, &
           madd_pb(13,13)/13/,madd_pb(14,13)/14/,madd_pb(15,13)/14/, &
           madd_pb(16,13)/13/,madd_pb(17,13)/14/,madd_pb(18,13)/14/, &
           madd_pb(19,13)/13/,madd_pb(20,13)/14/,madd_pb(21,13)/14/, &
           madd_pb(22,13)/13/,madd_pb(23,13)/14/,madd_pb(24,13)/14/, &
           madd_pb(25,13)/13/,madd_pb(26,13)/14/,madd_pb(27,13)/14/
!  Face xmax PB_FACEXMX =15
      data &
           madd_pb(1,15)/14/,madd_pb(2,15)/14/,madd_pb(3,15)/15/, &
           madd_pb(4,15)/14/,madd_pb(5,15)/14/,madd_pb(6,15)/15/, &
           madd_pb(7,15)/14/,madd_pb(8,15)/14/,madd_pb(9,15)/15/, &
           madd_pb(10,15)/14/,madd_pb(11,15)/14/,madd_pb(12,15)/15/, &
           madd_pb(13,15)/14/,madd_pb(14,15)/14/,madd_pb(15,15)/15/, &
           madd_pb(16,15)/14/,madd_pb(17,15)/14/,madd_pb(18,15)/15/, &
           madd_pb(19,15)/14/,madd_pb(20,15)/14/,madd_pb(21,15)/15/, &
           madd_pb(22,15)/14/,madd_pb(23,15)/14/,madd_pb(24,15)/15/, &
           madd_pb(25,15)/14/,madd_pb(26,15)/14/,madd_pb(27,15)/15/
!  Face ymin PB_FACEYMN =11,
      data     &
           madd_pb(1,11)/11/,madd_pb(2,11)/11/,madd_pb(3,11)/11/, &
           madd_pb(4,11)/14/,madd_pb(5,11)/14/,madd_pb(6,11)/14/, &
           madd_pb(7,11)/14/,madd_pb(8,11)/14/,madd_pb(9,11)/14/, &
           madd_pb(10,11)/11/,madd_pb(11,11)/11/,madd_pb(12,11)/11/, &
           madd_pb(13,11)/14/,madd_pb(14,11)/14/,madd_pb(15,11)/14/, &
           madd_pb(16,11)/14/,madd_pb(17,11)/14/,madd_pb(18,11)/14/, &
           madd_pb(19,11)/11/,madd_pb(20,11)/11/,madd_pb(21,11)/11/, &
           madd_pb(22,11)/14/,madd_pb(23,11)/14/,madd_pb(24,11)/14/, &
           madd_pb(25,11)/14/,madd_pb(26,11)/14/,madd_pb(27,11)/14/
!  Face ymax  PB_FACEYMX=17
      data  &
           madd_pb(1,17)/14/,madd_pb(2,17)/14/,madd_pb(3,17)/14/, &
           madd_pb(4,17)/14/,madd_pb(5,17)/14/,madd_pb(6,17)/14/, &
           madd_pb(7,17)/17/,madd_pb(8,17)/17/,madd_pb(9,17)/17/, &
           madd_pb(10,17)/14/,madd_pb(11,17)/14/,madd_pb(12,17)/14/, &
           madd_pb(13,17)/14/,madd_pb(14,17)/14/,madd_pb(15,17)/14/, &
           madd_pb(16,17)/17/,madd_pb(17,17)/17/,madd_pb(18,17)/17/, &
           madd_pb(19,17)/14/,madd_pb(20,17)/14/,madd_pb(21,17)/14/, &
           madd_pb(22,17)/14/,madd_pb(23,17)/14/,madd_pb(24,17)/14/, &
           madd_pb(25,17)/17/,madd_pb(26,17)/17/,madd_pb(27,17)/17/
! face zmin PB_FACEZMN=5,
      data  &
           madd_pb(1,5)/5/,madd_pb(2,5)/5/,madd_pb(3,5)/5/, &
           madd_pb(4,5)/5/,madd_pb(5,5)/5/,madd_pb(6,5)/5/, &
           madd_pb(7,5)/5/,madd_pb(8,5)/5/,madd_pb(9,5)/5/, &
           madd_pb(10,5)/14/,madd_pb(11,5)/14/,madd_pb(12,5)/14/, &
           madd_pb(13,5)/14/,madd_pb(14,5)/14/,madd_pb(15,5)/14/, &
           madd_pb(16,5)/14/,madd_pb(17,5)/14/,madd_pb(18,5)/14/, &
           madd_pb(19,5)/14/,madd_pb(20,5)/14/,madd_pb(21,5)/14/, &
           madd_pb(22,5)/14/,madd_pb(23,5)/14/,madd_pb(24,5)/14/, &
           madd_pb(25,5)/14/,madd_pb(26,5)/14/,madd_pb(27,5)/14/
!  top face  images from bottom  PB_FACEZMX=23,
      data  &
           madd_pb(1,23)/14/,madd_pb(2,23)/14/,madd_pb(3,23)/14/, &
           madd_pb(4,23)/14/,madd_pb(5,23)/14/,madd_pb(6,23)/14/, &
           madd_pb(7,23)/14/,madd_pb(8,23)/14/,madd_pb(9,23)/14/, &
           madd_pb(10,23)/14/,madd_pb(11,23)/14/,madd_pb(12,23)/14/, &
           madd_pb(13,23)/14/,madd_pb(14,23)/14/,madd_pb(15,23)/14/, &
           madd_pb(16,23)/14/,madd_pb(17,23)/14/,madd_pb(18,23)/14/, &
           madd_pb(19,23)/23/,madd_pb(20,23)/23/,madd_pb(21,23)/23/, &
           madd_pb(22,23)/23/,madd_pb(23,23)/23/,madd_pb(24,23)/23/, &
           madd_pb(25,23)/23/,madd_pb(26,23)/23/,madd_pb(27,23)/23/
! corner 1 x=0,y=0,z=0 PB_CORNER000=1
      data  &
           madd_pb(1,1)/1/,madd_pb(2,1)/2/,madd_pb(3,1)/2/, &
           madd_pb(4,1)/4/,madd_pb(5,1)/5/,madd_pb(6,1)/5/, &
           madd_pb(7,1)/4/,madd_pb(8,1)/5/,madd_pb(9,1)/5/, &
           madd_pb(10,1)/10/,madd_pb(11,1)/11/,madd_pb(12,1)/11/, &
           madd_pb(13,1)/13/,madd_pb(14,1)/14/,madd_pb(15,1)/14/, &
           madd_pb(16,1)/13/,madd_pb(17,1)/14/,madd_pb(18,1)/14/, &
           madd_pb(19,1)/10/,madd_pb(20,1)/11/,madd_pb(21,1)/11/, &
           madd_pb(22,1)/13/,madd_pb(23,1)/14/,madd_pb(24,1)/14/, &
           madd_pb(25,1)/13/,madd_pb(26,1)/14/,madd_pb(27,1)/14/
!  corner2 x=max, y=0 z=0  PB_CORNER100=3,
      data  &
           madd_pb(1,3)/2/,madd_pb(2,3)/2/,madd_pb(3,3)/3/, &
           madd_pb(4,3)/5/,madd_pb(5,3)/5/,madd_pb(6,3)/6/, &
           madd_pb(7,3)/5/,madd_pb(8,3)/5/,madd_pb(9,3)/6/, &
           madd_pb(10,3)/11/,madd_pb(11,3)/11/,madd_pb(12,3)/12/, &
           madd_pb(13,3)/14/,madd_pb(14,3)/14/,madd_pb(15,3)/15/, &
           madd_pb(16,3)/14/,madd_pb(17,3)/14/,madd_pb(18,3)/15/, &
           madd_pb(19,3)/11/,madd_pb(20,3)/11/,madd_pb(21,3)/12/, &
           madd_pb(22,3)/14/,madd_pb(23,3)/14/,madd_pb(24,3)/15/, &
           madd_pb(25,3)/14/,madd_pb(26,3)/14/,madd_pb(27,3)/15/
!  corner3 x=0, y=max z=0  PB_CORNER010=7
      data   &
           madd_pb(1,7)/4/,madd_pb(2,7)/5/,madd_pb(3,7)/5/, &
           madd_pb(4,7)/4/,madd_pb(5,7)/5/,madd_pb(6,7)/5/, &
           madd_pb(7,7)/7/,madd_pb(8,7)/8/,madd_pb(9,7)/8/, &
           madd_pb(10,7)/13/,madd_pb(11,7)/14/,madd_pb(12,7)/14/, &
           madd_pb(13,7)/13/,madd_pb(14,7)/14/,madd_pb(15,7)/14/, &
           madd_pb(16,7)/16/,madd_pb(17,7)/17/,madd_pb(18,7)/17/, &
           madd_pb(19,7)/13/,madd_pb(20,7)/14/,madd_pb(21,7)/14/, &
           madd_pb(22,7)/13/,madd_pb(23,7)/14/,madd_pb(24,7)/14/, &
           madd_pb(25,7)/16/,madd_pb(26,7)/17/,madd_pb(27,7)/17/
!  corner4 x=max, y=max z=0 PB_CORNER110=9
      data  &
           madd_pb(1,9)/5/,madd_pb(2,9)/5/,madd_pb(3,9)/6/, &
           madd_pb(4,9)/5/,madd_pb(5,9)/5/,madd_pb(6,9)/6/, &
           madd_pb(7,9)/8/,madd_pb(8,9)/8/,madd_pb(9,9)/9/, &
           madd_pb(10,9)/14/,madd_pb(11,9)/14/,madd_pb(12,9)/15/, &
           madd_pb(13,9)/14/,madd_pb(14,9)/14/,madd_pb(15,9)/15/, &
           madd_pb(16,9)/17/,madd_pb(17,9)/17/,madd_pb(18,9)/18/, &
           madd_pb(19,9)/14/,madd_pb(20,9)/14/,madd_pb(21,9)/15/, &
           madd_pb(22,9)/14/,madd_pb(23,9)/14/,madd_pb(24,9)/15/, &
           madd_pb(25,9)/17/,madd_pb(26,9)/17/,madd_pb(27,9)/18/
!  corner5 x=0 y=0 z=max   PB_CORNER001=19
      data  &
           madd_pb(1,19)/10/,madd_pb(2,19)/11/,madd_pb(3,19)/11/, &
           madd_pb(4,19)/13/,madd_pb(5,19)/14/,madd_pb(6,19)/14/, &
           madd_pb(7,19)/13/,madd_pb(8,19)/14/,madd_pb(9,19)/14/, &
           madd_pb(10,19)/10/,madd_pb(11,19)/11/,madd_pb(12,19)/11/, &
           madd_pb(13,19)/13/,madd_pb(14,19)/14/,madd_pb(15,19)/14/, &
           madd_pb(16,19)/13/,madd_pb(17,19)/14/,madd_pb(18,19)/14/ , &
           madd_pb(19,19)/19/,madd_pb(20,19)/20/,madd_pb(21,19)/20/, &
           madd_pb(22,19)/22/,madd_pb(23,19)/23/,madd_pb(24,19)/23/, &
           madd_pb(25,19)/22/,madd_pb(26,19)/23/,madd_pb(27,19)/23/
!  corner6 x=max y=0 z=max PB_CORNER101=21
      data  &
           madd_pb(1,21)/11/,madd_pb(2,21)/11/,madd_pb(3,21)/12/, &
           madd_pb(4,21)/14/,madd_pb(5,21)/14/,madd_pb(6,21)/15/, &
           madd_pb(7,21)/14/,madd_pb(8,21)/14/,madd_pb(9,21)/15/, &
           madd_pb(10,21)/11/,madd_pb(11,21)/11/,madd_pb(12,21)/12/, &
           madd_pb(13,21)/14/,madd_pb(14,21)/14/,madd_pb(15,21)/15/, &
           madd_pb(16,21)/14/,madd_pb(17,21)/14/,madd_pb(18,21)/15/, &
           madd_pb(19,21)/20/,madd_pb(20,21)/20/,madd_pb(21,21)/21/, &
           madd_pb(22,21)/23/,madd_pb(23,21)/23/,madd_pb(24,21)/24/, &
           madd_pb(25,21)/23/,madd_pb(26,21)/23/,madd_pb(27,21)/24/
!  corner7 x=0 y=max z=max   PB_CORNER011=25
      data  &
           madd_pb(1,25)/13/,madd_pb(2,25)/14/,madd_pb(3,25)/14/, &
           madd_pb(4,25)/13/,madd_pb(5,25)/14/,madd_pb(6,25)/14/, &
           madd_pb(7,25)/16/,madd_pb(8,25)/17/,madd_pb(9,25)/17/, &
           madd_pb(10,25)/13/,madd_pb(11,25)/14/,madd_pb(12,25)/14/, &
           madd_pb(13,25)/13/,madd_pb(14,25)/14/,madd_pb(15,25)/14/, &
           madd_pb(16,25)/16/,madd_pb(17,25)/17/,madd_pb(18,25)/17/, &
           madd_pb(19,25)/22/,madd_pb(20,25)/23/,madd_pb(21,25)/23/, &
           madd_pb(22,25)/22/,madd_pb(23,25)/23/,madd_pb(24,25)/23/, &
           madd_pb(25,25)/25/,madd_pb(26,25)/26/,madd_pb(27,25)/26/
!  corner8 x=max y=max z=max PB_CORNER111=27,
      data  &
           madd_pb(1,27)/14/,madd_pb(2,27)/14/,madd_pb(3,27)/15/, &
           madd_pb(4,27)/14/,madd_pb(5,27)/14/,madd_pb(6,27)/15/, &
           madd_pb(7,27)/17/,madd_pb(8,27)/17/,madd_pb(9,27)/18/, &
           madd_pb(10,27)/14/,madd_pb(11,27)/14/,madd_pb(12,27)/15/, &
           madd_pb(13,27)/14/,madd_pb(14,27)/14/,madd_pb(15,27)/15/, &
           madd_pb(16,27)/17/,madd_pb(17,27)/17/,madd_pb(18,27)/18/, &
           madd_pb(19,27)/23/,madd_pb(20,27)/23/,madd_pb(21,27)/24/, &
           madd_pb(22,27)/23/,madd_pb(23,27)/23/,madd_pb(24,27)/24/, &
           madd_pb(25,27)/26/,madd_pb(26,27)/26/,madd_pb(27,27)/27/
! Edge 00X x=0 y=0   PB_EDGE00X=10
      data     &
           madd_pb(1,10)/10/,madd_pb(2,10)/11/,madd_pb(3,10)/11/, &
           madd_pb(4,10)/13/,madd_pb(5,10)/14/,madd_pb(6,10)/14/, &
           madd_pb(7,10)/13/,madd_pb(8,10)/14/,madd_pb(9,10)/14/, &
           madd_pb(10,10)/10/,madd_pb(11,10)/11/,madd_pb(12,10)/11/, &
           madd_pb(13,10)/13/,madd_pb(14,10)/14/,madd_pb(15,10)/14/, &
           madd_pb(16,10)/13/,madd_pb(17,10)/14/,madd_pb(18,10)/14/, &
           madd_pb(19,10)/10/,madd_pb(20,10)/11/,madd_pb(21,10)/11/, &
           madd_pb(22,10)/13/,madd_pb(23,10)/14/,madd_pb(24,10)/14/, &
           madd_pb(25,10)/13/,madd_pb(26,10)/14/,madd_pb(27,10)/14/
! Edge 01X x=0 y=1  PB_EDGE01X=16
      data  &
           madd_pb(1,16)/13/,madd_pb(2,16)/14/,madd_pb(3,16)/14/, &
           madd_pb(4,16)/13/,madd_pb(5,16)/14/,madd_pb(6,16)/14/, &
           madd_pb(7,16)/16/,madd_pb(8,16)/17/,madd_pb(9,16)/17/, &
           madd_pb(10,16)/13/,madd_pb(11,16)/14/,madd_pb(12,16)/14/, &
           madd_pb(13,16)/13/,madd_pb(14,16)/14/,madd_pb(15,16)/14/, &
           madd_pb(16,16)/16/,madd_pb(17,16)/17/,madd_pb(18,16)/17/, &
           madd_pb(19,16)/13/,madd_pb(20,16)/14/,madd_pb(21,16)/14/, &
           madd_pb(22,16)/13/,madd_pb(23,16)/14/,madd_pb(24,16)/14/, &
           madd_pb(25,16)/16/,madd_pb(26,16)/17/,madd_pb(27,16)/17/
! Edge 10X x=1 y=0   PB_EDGE10X=12,
      data  &
           madd_pb(1,12)/11/,madd_pb(2,12)/11/,madd_pb(3,12)/12/, &
           madd_pb(4,12)/14/,madd_pb(5,12)/14/,madd_pb(6,12)/15/, &
           madd_pb(7,12)/14/,madd_pb(8,12)/14/,madd_pb(9,12)/15/, &
           madd_pb(10,12)/11/,madd_pb(11,12)/11/,madd_pb(12,12)/12/, &
           madd_pb(13,12)/14/,madd_pb(14,12)/14/,madd_pb(15,12)/15/, &
           madd_pb(16,12)/14/,madd_pb(17,12)/14/,madd_pb(18,12)/15/, &
           madd_pb(19,12)/11/,madd_pb(20,12)/11/,madd_pb(21,12)/12/, &
           madd_pb(22,12)/14/,madd_pb(23,12)/14/,madd_pb(24,12)/15/, &
           madd_pb(25,12)/14/,madd_pb(26,12)/14/,madd_pb(27,12)/15/
! Edge 11X x=1 y=1    PB_EDGE11X=18
      data     &
           madd_pb(1,18)/14/,madd_pb(2,18)/14/,madd_pb(3,18)/15/, &
           madd_pb(4,18)/14/,madd_pb(5,18)/14/,madd_pb(6,18)/15/, &
           madd_pb(7,18)/17/,madd_pb(8,18)/17/,madd_pb(9,18)/18/, &
           madd_pb(10,18)/14/,madd_pb(11,18)/14/,madd_pb(12,18)/15/, &
           madd_pb(13,18)/14/,madd_pb(14,18)/14/,madd_pb(15,18)/15/, &
           madd_pb(16,18)/17/,madd_pb(17,18)/17/,madd_pb(18,18)/18/, &
           madd_pb(19,18)/14/,madd_pb(20,18)/14/,madd_pb(21,18)/15/, &
           madd_pb(22,18)/14/,madd_pb(23,18)/14/,madd_pb(24,18)/15/, &
           madd_pb(25,18)/17/,madd_pb(26,18)/17/,madd_pb(27,18)/18/
! Edge X00 y=0 z=0  PB_EDGEX00=2
      data &
           madd_pb(1,2)/2/,madd_pb(2,2)/2/,madd_pb(3,2)/2/, &
           madd_pb(4,2)/5/,madd_pb(5,2)/5/,madd_pb(6,2)/5/, &
           madd_pb(7,2)/5/,madd_pb(8,2)/5/,madd_pb(9,2)/5/, &
           madd_pb(10,2)/11/,madd_pb(11,2)/11/,madd_pb(12,2)/11/, &
           madd_pb(13,2)/14/,madd_pb(14,2)/14/,madd_pb(15,2)/14/, &
           madd_pb(16,2)/14/,madd_pb(17,2)/14/,madd_pb(18,2)/14/, &
           madd_pb(19,2)/11/,madd_pb(20,2)/11/,madd_pb(21,2)/11/, &
           madd_pb(22,2)/14/,madd_pb(23,2)/14/,madd_pb(24,2)/14/, &
           madd_pb(25,2)/14/,madd_pb(26,2)/14/,madd_pb(27,2)/14/
! Edge X10 y=1 z=0 PB_EDGEX10=8,
      data     &
           madd_pb(1,8)/5/,madd_pb(2,8)/5/,madd_pb(3,8)/5/, &
           madd_pb(4,8)/5/,madd_pb(5,8)/5/,madd_pb(6,8)/5/, &
           madd_pb(7,8)/8/,madd_pb(8,8)/8/,madd_pb(9,8)/8/, &
           madd_pb(10,8)/14/,madd_pb(11,8)/14/,madd_pb(12,8)/14/, &
           madd_pb(13,8)/14/,madd_pb(14,8)/14/,madd_pb(15,8)/14/, &
           madd_pb(16,8)/17/,madd_pb(17,8)/17/,madd_pb(18,8)/17/, &
           madd_pb(19,8)/14/,madd_pb(20,8)/14/,madd_pb(21,8)/14/, &
           madd_pb(22,8)/14/,madd_pb(23,8)/14/,madd_pb(24,8)/14/, &
           madd_pb(25,8)/17/,madd_pb(26,8)/17/,madd_pb(27,8)/17/
! Edge X01 y=0 z=1   PB_EDGEX01=20
      data     &
           madd_pb(1,20)/11/,madd_pb(2,20)/11/,madd_pb(3,20)/11/, &
           madd_pb(4,20)/14/,madd_pb(5,20)/14/,madd_pb(6,20)/14/, &
           madd_pb(7,20)/14/,madd_pb(8,20)/14/,madd_pb(9,20)/14/, &
           madd_pb(10,20)/11/,madd_pb(11,20)/11/,madd_pb(12,20)/11/, &
           madd_pb(13,20)/14/,madd_pb(14,20)/14/,madd_pb(15,20)/14/, &
           madd_pb(16,20)/14/,madd_pb(17,20)/14/,madd_pb(18,20)/14/, &
           madd_pb(19,20)/20/,madd_pb(20,20)/20/,madd_pb(21,20)/20/, &
           madd_pb(22,20)/23/,madd_pb(23,20)/23/,madd_pb(24,20)/23/, &
           madd_pb(25,20)/23/,madd_pb(26,20)/23/,madd_pb(27,20)/23/
! Edge X11 y=1 z=1  PB_EDGEX11=26
      data     &
           madd_pb(1,26)/14/,madd_pb(2,26)/14/,madd_pb(3,26)/14/, &
           madd_pb(4,26)/14/,madd_pb(5,26)/14/,madd_pb(6,26)/14/, &
           madd_pb(7,26)/17/,madd_pb(8,26)/17/,madd_pb(9,26)/17/, &
           madd_pb(10,26)/14/,madd_pb(11,26)/14/,madd_pb(12,26)/14/, &
           madd_pb(13,26)/14/,madd_pb(14,26)/14/,madd_pb(15,26)/14/, &
           madd_pb(16,26)/17/,madd_pb(17,26)/17/,madd_pb(18,26)/17/, &
           madd_pb(19,26)/23/,madd_pb(20,26)/23/,madd_pb(21,26)/23/, &
           madd_pb(22,26)/23/,madd_pb(23,26)/23/,madd_pb(24,26)/23/, &
           madd_pb(25,26)/26/,madd_pb(26,26)/26/,madd_pb(27,26)/26/
! Edge 0X0 x=0 z=0  PB_EDGE0X0=4
      data     &
           madd_pb(1,4)/4/,madd_pb(2,4)/5/,madd_pb(3,4)/5/, &
           madd_pb(4,4)/4/,madd_pb(5,4)/5/,madd_pb(6,4)/5/, &
           madd_pb(7,4)/4/,madd_pb(8,4)/5/,madd_pb(9,4)/5/, &
           madd_pb(10,4)/13/,madd_pb(11,4)/14/,madd_pb(12,4)/14/, &
           madd_pb(13,4)/13/,madd_pb(14,4)/14/,madd_pb(15,4)/14/, &
           madd_pb(16,4)/13/,madd_pb(17,4)/14/,madd_pb(18,4)/14/, &
           madd_pb(19,4)/13/,madd_pb(20,4)/14/,madd_pb(21,4)/14/, &
           madd_pb(22,4)/13/,madd_pb(23,4)/14/,madd_pb(24,4)/14/, &
           madd_pb(25,4)/13/,madd_pb(26,4)/14/,madd_pb(27,4)/14/
! Edge 0X1 x=0 z=1   PB_EDGE0X1=22
      data     &
           madd_pb(1,22)/13/,madd_pb(2,22)/14/,madd_pb(3,22)/14/, &
           madd_pb(4,22)/13/,madd_pb(5,22)/14/,madd_pb(6,22)/14/, &
           madd_pb(7,22)/13/,madd_pb(8,22)/14/,madd_pb(9,22)/14/, &
           madd_pb(10,22)/13/,madd_pb(11,22)/14/,madd_pb(12,22)/14/, &
           madd_pb(13,22)/13/,madd_pb(14,22)/14/,madd_pb(15,22)/14/, &
           madd_pb(16,22)/13/,madd_pb(17,22)/14/,madd_pb(18,22)/14/, &
           madd_pb(19,22)/22/,madd_pb(20,22)/23/,madd_pb(21,22)/23/, &
           madd_pb(22,22)/22/,madd_pb(23,22)/23/,madd_pb(24,22)/23/, &
           madd_pb(25,22)/22/,madd_pb(26,22)/23/,madd_pb(27,22)/23/
! Edge 1X0 x=1 z=0  PB_EDGE1X0=6,
      data     &
           madd_pb(1,6)/5/,madd_pb(2,6)/5/,madd_pb(3,6)/6/, &
           madd_pb(4,6)/5/,madd_pb(5,6)/5/,madd_pb(6,6)/6/, &
           madd_pb(7,6)/5/,madd_pb(8,6)/5/,madd_pb(9,6)/6/, &
           madd_pb(10,6)/14/,madd_pb(11,6)/14/,madd_pb(12,6)/15/, &
           madd_pb(13,6)/14/,madd_pb(14,6)/14/,madd_pb(15,6)/15/, &
           madd_pb(16,6)/14/,madd_pb(17,6)/14/,madd_pb(18,6)/15/, &
           madd_pb(19,6)/14/,madd_pb(20,6)/14/,madd_pb(21,6)/15/, &
           madd_pb(22,6)/14/,madd_pb(23,6)/14/,madd_pb(24,6)/15/, &
           madd_pb(25,6)/14/,madd_pb(26,6)/14/,madd_pb(27,6)/15/
! Edge 1X1 x=1 z=1  PB_EDGE1X1=24
      data     &
           madd_pb(1,24)/14/,madd_pb(2,24)/14/,madd_pb(3,24)/15/, &
           madd_pb(4,24)/14/,madd_pb(5,24)/14/,madd_pb(6,24)/15/, &
           madd_pb(7,24)/14/,madd_pb(8,24)/14/,madd_pb(9,24)/15/, &
           madd_pb(10,24)/14/,madd_pb(11,24)/14/,madd_pb(12,24)/15/, &
           madd_pb(13,24)/14/,madd_pb(14,24)/14/,madd_pb(15,24)/15/, &
           madd_pb(16,24)/14/,madd_pb(17,24)/14/,madd_pb(18,24)/15/, &
           madd_pb(19,24)/23/,madd_pb(20,24)/23/,madd_pb(21,24)/24/, &
           madd_pb(22,24)/23/,madd_pb(23,24)/23/,madd_pb(24,24)/24/, &
           madd_pb(25,24)/23/,madd_pb(26,24)/23/,madd_pb(27,24)/24/
end module pbound_madd

