#
# Calculate the inverse matrix (xinvbox) of the basis (boxsiz)
#
subroutine inversebox
  implicit REAL*8 (A-H,O-Z)
  real*8 xhlp
  logical period
  integer nlat
  common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(3)
   
  xhlp=-boxsiz(1,3)*boxsiz(2,2)*boxsiz(3,1)+boxsiz(1,2)*boxsiz(2,3)*boxsiz(3,1)
  xhlp=xhlp+boxsiz(1,3)*boxsiz(2,1)*boxsiz(3,2)
  xhlp=xhlp-boxsiz(1,1)*boxsiz(2,3)*boxsiz(3,2)
  xhlp=xhlp-boxsiz(1,2)*boxsiz(2,1)*boxsiz(3,3)
  xhlp=xhlp+boxsiz(1,1)*boxsiz(2,2)*boxsiz(3,3)

  xinvbox(1,1)=(-boxsiz(2,3)*boxsiz(3,2)+boxsiz(2,2)*boxsiz(3,3))/xhlp
  xinvbox(2,1)=(boxsiz(2,3)*boxsiz(3,1)-boxsiz(2,1)*boxsiz(3,3))/xhlp
  xinvbox(3,1)=(-boxsiz(2,2)*boxsiz(3,1)+boxsiz(2,1)*boxsiz(3,2))/xhlp

  xinvbox(1,2)=(boxsiz(1,3)*boxsiz(3,2)-boxsiz(1,2)*boxsiz(3,3))/xhlp
  xinvbox(2,2)=(-boxsiz(1,3)*boxsiz(3,1)+boxsiz(1,1)*boxsiz(3,3))/xhlp
  xinvbox(3,2)=(boxsiz(1,2)*boxsiz(3,1)-boxsiz(1,1)*boxsiz(3,2))/xhlp

  xinvbox(1,3)=(-boxsiz(1,3)*boxsiz(2,2)+boxsiz(1,2)*boxsiz(2,3))/xhlp
  xinvbox(2,3)=(boxsiz(1,3)*boxsiz(2,1)-boxsiz(1,1)*boxsiz(2,3))/xhlp
  xinvbox(3,3)=(-boxsiz(1,2)*boxsiz(2,1)+boxsiz(1,1)*boxsiz(2,2))/xhlp
end

#
# find the lenght of the shortest vertex in your supercell 
#
real*8 function shortvertice(boxsiz)
  real*8 yhlp,boxsiz(3,3),testbox(6)
  
  testbox(1)=boxsiz(1,1)**2+boxsiz(1,2)**2+boxsiz(1,3)**2
  testbox(2)=boxsiz(2,1)**2+boxsiz(2,2)**2+boxsiz(2,3)**2
  testbox(3)=boxsiz(3,1)**2+boxsiz(3,2)**2+boxsiz(3,3)**2

  testbox(4)=(boxsiz(1,1)-boxsiz(2,1))**2+(boxsiz(1,2)-boxsiz(2,2))**2+ _ 
             (boxsiz(1,3)-boxsiz(2,3))**2
  testbox(5)=(boxsiz(1,1)-boxsiz(3,1))**2+(boxsiz(1,2)-boxsiz(3,2))**2+ _
             (boxsiz(1,3)-boxsiz(3,3))**2
  testbox(6)=(boxsiz(3,1)-boxsiz(2,1))**2+(boxsiz(3,2)-boxsiz(2,2))**2+ _ 
             (boxsiz(3,3)-boxsiz(2,3))**2

  yhlp=MIN(testbox(1),testbox(2),testbox(3),testbox(4),testbox(5),testbox(6))
  shortvertice=sqrt(yhlp)
end

#
# Put a difference vector back into supercell
#
subroutine difback(dif,boxsiz,xinvbox)
     implicit REAL*8 (A-H,O-Z)
     real*8 dif(3),boxsiz(3,3),xinvbox(3,3)
     real*8 xx1,xy1,xz1
     
     xx1=dif(1)*xinvbox(1,1)+dif(2)*xinvbox(2,1)+dif(3)*xinvbox(3,1)
     xy1=dif(1)*xinvbox(1,2)+dif(2)*xinvbox(2,2)+dif(3)*xinvbox(3,2)
     xz1=dif(1)*xinvbox(1,3)+dif(2)*xinvbox(2,3)+dif(3)*xinvbox(3,3)

     if(xx1>0.5) xx1=xx1-1.0
     if(xx1<-0.5) xx1=xx1+1.0
     if(xy1>0.5) xy1=xy1-1.0
     if(xy1<-0.5) xy1=xy1+1.0
     if(xz1>0.5) xz1=xz1-1.0
     if(xz1<-0.5) xz1=xz1+1.0    
       
     dif(1)=xx1*boxsiz(1,1)+xy1*boxsiz(2,1)+xz1*boxsiz(3,1)
     dif(2)=xx1*boxsiz(1,2)+xy1*boxsiz(2,2)+xz1*boxsiz(3,2)
     dif(3)=xx1*boxsiz(1,3)+xy1*boxsiz(2,3)+xz1*boxsiz(3,3)
end

#
# Put Atom at (x,z,z) back into supercell
#
subroutine coordback(x,y,z)
  implicit REAL*8 (A-H,O-Z)
  real*8 x,y,z
  real*8 xx1,xy1,xz1
  logical period
  integer nlat
  common /box/ boxsiz(3,3), xinvbox(3,3), xnullvec(3), period, nlat(3)

  xx1=(x-xnullvec(1))*xinvbox(1,1)+(y-xnullvec(2))*xinvbox(2,1)+(z-xnullvec(3))*xinvbox(3,1)
  xy1=(x-xnullvec(1))*xinvbox(1,2)+(y-xnullvec(2))*xinvbox(2,2)+(z-xnullvec(3))*xinvbox(3,2)
  xz1=(x-xnullvec(1))*xinvbox(1,3)+(y-xnullvec(2))*xinvbox(2,3)+(z-xnullvec(3))*xinvbox(3,3)

  if(xx1>0.5) xx1=xx1-1.0
  if(xx1<-0.5) xx1=xx1+1.0
  if(xy1>0.5) xy1=xy1-1.0
  if(xy1<-0.5) xy1=xy1+1.0
  if(xz1>0.5) xz1=xz1-1.0
  if(xz1<-0.5) xz1=xz1+1.0
        
  x=xx1*boxsiz(1,1)+xy1*boxsiz(2,1)+xz1*boxsiz(3,1)+xnullvec(1)
  y=xx1*boxsiz(1,2)+xy1*boxsiz(2,2)+xz1*boxsiz(3,2)+xnullvec(2)
  z=xx1*boxsiz(1,3)+xy1*boxsiz(2,3)+xz1*boxsiz(3,3)+xnullvec(3)
end

#
#get the three summation limits for the matrix construction in
#the gamma point approximation
#output: nlat(3)
#
subroutine gamma_summind(slkcutoff,boxsiz,nlat)
  implicit none
  real*8 slkcutoff, boxsiz(3,3)
  integer nlat(3)
  real*8 u(3),v(3),w(3),helpv(3),l,lu,lv,lw 

#get vectors and lengths
  u(1) = boxsiz(1,1); u(2) = boxsiz(1,2); u(3) = boxsiz(1,3)
  v(1) = boxsiz(2,1); v(2) = boxsiz(2,2); v(3) = boxsiz(2,3)
  w(1) = boxsiz(3,1); w(2) = boxsiz(3,2); w(3) = boxsiz(3,3)
  lu = sqrt(u(1)**2 + u(2)**2 + u(3)**2)
  lv = sqrt(v(1)**2 + v(2)**2 + v(3)**2)
  lw = sqrt(w(1)**2 + w(2)**2 + w(3)**2)
   
   
#see, whether length of u is shorter than projected side of v
   CALL CROSSSCC(u,v,helpv)
   l = min(lu,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lv)
   
#see, whether length of u is shorter than projected side of w
   CALL CROSSSCC(u,w,helpv)
   l = min(lu,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lw)
  
#set nlat(1) according to slkcutoff. At least 1.
   nlat(1) = max(int(2*slkcutoff/l),1)

#for v,w do the same as for u...
   CALL CROSSSCC(v,u,helpv)
   l = min(lv,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lu)
   
   CALL CROSSSCC(v,w,helpv)
   l = min(lv,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lw)
  
   nlat(2) = max(int(2*slkcutoff/l),1)


   CALL CROSSSCC(w,u,helpv)
   l = min(lw,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lu)
   
   CALL CROSSSCC(w,v,helpv)
   l = min(lw,sqrt(helpv(1)**2 + helpv(2)**2 + helpv(3)**2)/lv)
  
   nlat(3) = max(int(2*slkcutoff/l),1)

end

