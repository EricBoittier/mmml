module dcor_network
  use chm_kinds
  use number
  use consta

  implicit none

contains

subroutine distcor(tq1,d1,tq2,d2,n,dvar1,dvar2,dcov,dcor)
  !
  !     THIS ROUTINE TAKES THE TIME SERIES AND CALCULATES
  !     THE DISTANCE COVARIANCE.
  !     TQ1 and TQ2 are two time series with N entries
  !     They are of D1 and D2 dimension respectively
  !     DVAR1 - Distance variance of TQ1
  !     DVAR2 - Distance variance of TQ2
  !     DCOV  - Distance covariance between TQ1 and TQ2
  !     DCOR  - DCOV/SQRT(DVAR1*DVAR2)
  !           - Distance correlation coefficient
  !     Author: Amitava Roy 03/27/2016
  !     Reference : Detection of Long-Range Concerted Motions in Protein by a Distance Covariance
  !                 Amitava Roy and Carol Beth Post
  !                 J. Chem. Theory Comput., 2012, 8 (9), pp 3009–3014
  !               
  !                 Measuring and testing dependence by correlation of distances
  !                 GJ Székely, ML Rizzo, NK Bakirov
  !                 The Annals of Statistics 35.6 (2007): 2769-2794.
  !                 The calculation follows the algebraic identity 2.15-2.18 described in
  !                 page 2776 of the paper. Variables S1(3), S2(3) and S3(3) have the same meaning
  !                 as in the paper. S1(1), S2(1), S3(1) are to calculate distance variance
  !                 of the first series. S1(2), S2(2), S3(2) are to calculate distance variance
  !                 of the second series.
  !
  implicit none
  !
  integer, intent(in)           :: n,d1,d2
  real(chm_real), intent(in)    :: tq1(n,d1),tq2(n,d2)
  real(chm_real), intent(out)   :: dvar1,dvar2,dcov,dcor
  !
  !Local variables
  integer                       :: i,j,k,l,n2,n4
  real(chm_real)                :: s2a,s2b
  real(chm_real)                :: s1(3),s2(3),s3(3),bdcv(3)
  !Local intermediate variables
  real(chm_real)                :: aj(d1),bj(d2),ak(d1),bk(d2),ad(d1),bd(d2)
  real(chm_real)                :: dak,dbk,das,dbs

! Initialize
  s1=zero
  s2a=zero
  s2b=zero
  s3=zero

  do j=1,n
    aj=tq1(j,:)
    bj=tq2(j,:)
    das=zero 
    dbs=zero
    do k=1,n
      ak=tq1(k,:)
      bk=tq2(k,:)
      ! Calculate Eucledian distances between snap shots |ai-aj|
      ! The definition of distance is valid for any dimension
      ad=aj-ak
      bd=bj-bk
      ad=ad*ad
      bd=bd*bd
      dak=sum(ad)
      dbk=sum(bd)
      s1(1)=s1(1)+dak
      s1(2)=s1(2)+dbk
      dak=sqrt(dak)
      dbk=sqrt(dbk)
      s1(3)=s1(3)+dak*dbk   !Eqn 2.15 in the 2nd reference
      s2a=s2a+dak           !First sum of eqn 2.16 in the 2nd reference
      s2b=s2b+dbk           !Second sum of eqn 2.16 in the 2nd reference
      das=das+dak
      dbs=dbs+dbk
    enddo
    s3(1)=s3(1)+das*das
    s3(2)=s3(2)+dbs*dbs
    s3(3)=s3(3)+das*dbs     !An algebraic identity of eqn 2.17 in the 2nd reference
                            ! |ak-a1||bk-b1|+|ak-a1||bk-b2|+
                            ! |ak-a2||bk-b1|+|ak-a2||bk-b2|=
                            ! {|ak-a1|+|ak-a2|}{|bk-b1|+|bk-b2|}
                            ! das={|ak-a1|+|ak-a2|}, dbs={|bk-b1|+|bk-b2|}
  enddo
  s1=s1/n/n
  s2(1)=(s2a*s2a)/n/n/n/n
  s2(2)=(s2b*s2b)/n/n/n/n
  s2(3)=(s2a*s2b)/n/n/n/n
  s3=s3/n/n/n
  do j=1,3
    bdcv(j)=sqrt(s1(j)+s2(j)-two*s3(j))   !Eqn 2.18 in the 2nd reference
  enddo
  dvar1=bdcv(1)
  dvar2=bdcv(2)
  dcov=bdcv(3)
  dcor=bdcv(3)/sqrt(bdcv(1)*bdcv(2))
  return
  !
end subroutine distcor


end module dcor_network
