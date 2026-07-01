! CHARMM Element source/misc/fctblockl.src $Revision: 2.0 $
#if KEY_BLOCK==1 /*block_main*/
#if KEY_FACTS==1 /*facts_block*/
   SUBROUTINE fctblk(fctpol             ,&
                     fctnpl             ,&
                     reanat             ,&
                     reablo,reanbo      ,&
                     fct1ll,fct1lb      ,&
                     fct2ll,fct2lb      ,&
                     imanat             ,&
                     imablo,imanbo      ,&
                     imattr             ,&
                     fct3ll,fct3lb      ,&
                     fct4ll,fct4lb      ,&
                     iblock,blcoe,blcov)

! This routine calculates the FACTS solvation free energy and its
! derivative with respect to Cartesian coordinates.
!
! Author: Francois Marchand, Urs Haberthuer.
! Inclusion of RSMALL in distance calculation 
! to avoid division by 0
! 3.4 - inclusion of imove
!
! Sun Jan 31 15:55:12 CET 2010
!     Modified: Francois Marchand
!     Converted to f95 standart
!
   use chm_kinds
   use dimens_fcm

   ! use block_fcm
   use coord
   use deriv
   use facts_module

   ! use inbnd
   use number
   use psf

   use stream
#if KEY_PARALLEL==1
   use parallel
#endif 
   use memory
       implicit none

      real(kind=chm_real), intent(out)  :: fctpol   , fctnpl
      integer, intent(in)               :: reanat   , imanat
      integer, intent(in)               :: imattr(*)
      integer, intent(in)               :: reablo(*), reanbo(*)
      integer, intent(in)               :: fct1ll(*), fct1lb(*)
      integer, intent(in)               :: fct2ll(*), fct2lb(*)
      integer, intent(in)               :: imablo(*), imanbo(*)
      integer, intent(in)               :: fct3ll(*), fct3lb(*)
      integer, intent(in)               :: fct4ll(*), fct4lb(*)
      integer, intent(in)               :: iblock(*)
      real(kind=chm_real), intent(in)   :: blcoe(*) , blcov(*)

   integer             :: a,b,u,v,i,j,k
   logical             :: qmove
   integer             :: Ubl,Vbl,KK
   integer             :: reanat2

   ! new
   integer :: iacmax,ierr,ierr2

   real(kind=chm_real) :: vux,vuy,vuz
   real(kind=chm_real) :: fctdvu,fctivu,fctsvu

   real(kind=chm_real) :: fct01vuss,fct02vuss,fct03vuss,fct04vuss
   real(kind=chm_real) :: fct05vuss,fct06vuss,fct07vuss
   real(kind=chm_real) :: fct01uvss,fct02uvss,fct03uvss,fct04uvss
   real(kind=chm_real) :: fct05uvss,fct06uvss,fct07uvss

   real(kind=chm_real) :: tmp01xs,tmp01ys,tmp01zs
   real(kind=chm_real) :: tmp02xs,tmp02ys,tmp02zs
   real(kind=chm_real) :: tmp03xs,tmp03ys,tmp03zs
   real(kind=chm_real) :: tmp01xx,tmp01xy,tmp01xz
   real(kind=chm_real) :: tmp01yy,tmp01yz,tmp01zz

   real(kind=chm_real) :: fct04xs,fct04ys,fct04zs

   real(kind=chm_real),allocatable,dimension(:,:) :: fct01ss, fct02ss, fct03ss
   real(kind=chm_real),allocatable,dimension(:,:) :: fct01xs, fct01ys, fct01zs
   real(kind=chm_real),allocatable,dimension(:,:) :: fct02xs, fct02ys, fct02zs
   real(kind=chm_real),allocatable,dimension(:,:) :: fct03xs, fct03ys, fct03zs
   real(kind=chm_real),allocatable,dimension(:,:) :: fct01xx, fct01xy, fct01xz
   real(kind=chm_real),allocatable,dimension(:,:) :: fct01yy, fct01yz
   real(kind=chm_real),allocatable,dimension(:,:) :: fct01zz

   real(kind=chm_real) :: fcttp1,fcttp2,fcttp3,fcttp4
   real(kind=chm_real) :: fcttp5,fcttp6,fcttp7,fcttp8

   real(kind=chm_real),allocatable, dimension(:,:) :: fctis1, fctis2
   real(kind=chm_real),allocatable, dimension(:,:) :: fctmov, fctmos

   real(kind=chm_real),allocatable, dimension(:,:) :: fctqsg
   real(kind=chm_real),allocatable, dimension(:,:) :: fctisg
   real(kind=chm_real),allocatable, dimension(:,:) :: fctqdv, fctqds
   real(kind=chm_real),allocatable, dimension(:,:) :: fctusg
   real(kind=chm_real),allocatable, dimension(:,:) :: fctudv, fctuds
   real(kind=chm_real),allocatable, dimension(:,:) :: fctvsg
   real(kind=chm_real),allocatable, dimension(:,:) :: fctvdv, fctvds
   real(kind=chm_real),allocatable, dimension(:,:) :: fctwsg
   real(kind=chm_real),allocatable, dimension(:,:) :: fctwdv, fctwds

   real(kind=chm_real) :: fctpl1, fctpl2
   real(kind=chm_real) :: fctnp1, fctnp2, fctnp3

   real(kind=chm_real) :: blkpl,blknp,blkplssx,blkplssy,blkplssz

   real(kind=chm_real),allocatable, dimension(:,:) :: tmppl1, tmpnp1
   real(kind=chm_real),allocatable, dimension(:,:) :: tmpplssx, tmpplssy, tmpplssz ,&
                                                      tmpnpssx, tmpnpssy, tmpnpssz

   real(kind=chm_real),allocatable, dimension(:,:) :: fctqsx, fctqsy, fctqsz
   real(kind=chm_real),allocatable, dimension(:,:) :: fctusx, fctusy, fctusz
   real(kind=chm_real),allocatable, dimension(:,:) :: fctvsx, fctvsy, fctvsz
   real(kind=chm_real),allocatable, dimension(:,:) :: fctwsx, fctwsy, fctwsz


   real(kind=chm_real) :: fct01vu, fct02vu, fct03vu, fct04vu
   real(kind=chm_real) :: fct05vu, fct06vu, fct07vu
   real(kind=chm_real) :: fct08vu, fct09vu, fct10vu, fct11vu
   real(kind=chm_real) :: fct12vu, fct13vu, fct14vu

   real(kind=chm_real) :: fct03vu1, fct03vu2, fct04vu1, fct04vu2
   real(kind=chm_real) :: fct05vu1, fct05vu2, fct06vu1, fct06vu2
   real(kind=chm_real) :: fct07vu1, fct07vu2, fct08vu1, fct08vu2
   real(kind=chm_real) :: fct09vu1, fct09vu2, fct10vu1, fct10vu2
   real(kind=chm_real) :: fct11vu1, fct11vu2, fct12vu1, fct12vu2
   real(kind=chm_real) :: fct13vu1, fct13vu2

   ! Salt Stuff
   real(kind=chm_real) :: fctslt01, fctslt02, fctslt03
   real(kind=chm_real) :: fctqsgslt
   real(kind=chm_real) :: fctcgslt(2,natom)

   real(kind=chm_real) :: fctggg(2,natom), fcthhh(2,natom), fctkkk(2,natom)

   real(kind=chm_real),allocatable, dimension(:) :: fctssx, fctssy, fctssz
   real(kind=chm_real),allocatable, dimension(:) :: fctsix, fctsiy, fctsiz
   real(kind=chm_real),allocatable, dimension(:) :: fctiix, fctiiy, fctiiz

   real(kind=chm_real) :: tmpiix1  , tmpiiy1  , tmpiiz1
   real(kind=chm_real) :: tmpiix2  , tmpiiy2  , tmpiiz2
   real(kind=chm_real) :: tmpsix(2), tmpsiy(2), tmpsiz(2)

   real(kind=chm_real) :: fct02vuxs, fct02vuys, fct02vuzs
   real(kind=chm_real) :: fct02uvxs, fct02uvys, fct02uvzs
   real(kind=chm_real) :: fct03vuxs, fct03vuys, fct03vuzs
   real(kind=chm_real) :: fct03uvxs, fct03uvys, fct03uvzs
   real(kind=chm_real) :: fct04vuxs, fct04vuys, fct04vuzs
   real(kind=chm_real) :: fct04uvxs, fct04uvys, fct04uvzs
   real(kind=chm_real) :: fct01vuxx, fct01vuxy, fct01vuxz
   real(kind=chm_real) ::            fct01vuyy, fct01vuyz
   real(kind=chm_real) ::                       fct01vuzz
   real(kind=chm_real) :: fct01uvxx, fct01uvxy, fct01uvxz
   real(kind=chm_real) ::            fct01uvyy, fct01uvyz
   real(kind=chm_real) ::                       fct01uvzz

   real(kind=chm_real) :: fctqsvux, fctqsvuy, fctqsvuz
   real(kind=chm_real) :: fctqsuvx, fctqsuvy, fctqsuvz
   real(kind=chm_real) :: fctusvux, fctusvuy, fctusvuz
   real(kind=chm_real) :: fctusuvx, fctusuvy, fctusuvz
   real(kind=chm_real) :: fctvsvux, fctvsvuy, fctvsvuz
   real(kind=chm_real) :: fctvsuvx, fctvsuvy, fctvsuvz
   real(kind=chm_real) :: fctwsvux, fctwsvuy, fctwsvuz
   real(kind=chm_real) :: fctwsuvx, fctwsuvy, fctwsuvz

   real(kind=chm_real) :: fctvdw01, fctvdw02
   real(kind=chm_real),allocatable,dimension(:,:)::fctvdwen, fctvdwdf
! ===================================================================
! ---------------------------------------------------------------------
! Array Allocation
!
   ! Real 1D
   call chmalloc('fctblock.src','FCTBLK','fctssx', natom, crl=fctssx, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctssy', natom, crl=fctssy, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctssz', natom, crl=fctssz, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctsix', natom, crl=fctsix, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctsiy', natom, crl=fctsiy, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctsiz', natom, crl=fctsiz, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctiix', natom, crl=fctiix, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctiiy', natom, crl=fctiiy, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctiiz', natom, crl=fctiiz, ierr=ierr2, qdie=.true.)

   ! Real 2D
   call chmalloc('fctblock.src','FCTBLK','fct01ss ', 2, natom, crl=fct01ss , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct02ss ', 2, natom, crl=fct02ss , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct03ss ', 2, natom, crl=fct03ss , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01xs ', 2, natom, crl=fct01xs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01ys ', 2, natom, crl=fct01ys , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01zs ', 2, natom, crl=fct01zs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct02xs ', 2, natom, crl=fct02xs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct02ys ', 2, natom, crl=fct02ys , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct02zs ', 2, natom, crl=fct02zs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct03xs ', 2, natom, crl=fct03xs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct03ys ', 2, natom, crl=fct03ys , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct03zs ', 2, natom, crl=fct03zs , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01xx ', 2, natom, crl=fct01xx , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01xy ', 2, natom, crl=fct01xy , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01xz ', 2, natom, crl=fct01xz , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01yy ', 2, natom, crl=fct01yy , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01yz ', 2, natom, crl=fct01yz , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fct01zz ', 2, natom, crl=fct01zz , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctis1'  , 2, natom, crl=fctis1  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctis2'  , 2, natom, crl=fctis2  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctmov'  , 2, natom, crl=fctmov  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctmos'  , 2, natom, crl=fctmos  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqsg'  , 2, natom, crl=fctqsg  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctisg'  , 2, natom, crl=fctisg  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqdv'  , 2, natom, crl=fctqdv  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqds'  , 2, natom, crl=fctqds  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctusg'  , 2, natom, crl=fctusg  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctudv'  , 2, natom, crl=fctudv  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctuds'  , 2, natom, crl=fctuds  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvsg'  , 2, natom, crl=fctvsg  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvdv'  , 2, natom, crl=fctvdv  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvds'  , 2, natom, crl=fctvds  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwsg'  , 2, natom, crl=fctwsg  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwdv'  , 2, natom, crl=fctwdv  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwds'  , 2, natom, crl=fctwds  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqsx'  , 2, natom, crl=fctqsx  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqsy'  , 2, natom, crl=fctqsy  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctqsz'  , 2, natom, crl=fctqsz  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctusx'  , 2, natom, crl=fctusx  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctusy'  , 2, natom, crl=fctusy  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctusz'  , 2, natom, crl=fctusz  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvsx'  , 2, natom, crl=fctvsx  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvsy'  , 2, natom, crl=fctvsy  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvsz'  , 2, natom, crl=fctvsz  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwsx'  , 2, natom, crl=fctwsx  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwsy'  , 2, natom, crl=fctwsy  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctwsz'  , 2, natom, crl=fctwsz  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmppl1'  , 2, natom, crl=tmppl1  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpnp1'  , 2, natom, crl=tmpnp1  , ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpplssx', 2, natom, crl=tmpplssx, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpnpssx', 2, natom, crl=tmpnpssx, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpplssy', 2, natom, crl=tmpplssy, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpnpssy', 2, natom, crl=tmpnpssy, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpplssz', 2, natom, crl=tmpplssz, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','tmpnpssz', 2, natom, crl=tmpnpssz, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvdwen', 2, natom, crl=fctvdwen, ierr=ierr2, qdie=.true.)
   call chmalloc('fctblock.src','FCTBLK','fctvdwdf', 2, natom, crl=fctvdwdf, ierr=ierr2, qdie=.true.)
! ===================================================================

   if (fctsfe) then

      fctpl1=zero
      fctpl2=zero
      fctnp1=zero
      fctnp2=zero
      fctnp3=zero

      qmove=.false.

      fct01ss=zero

      fct03ss=zero
      fct01xs=zero
      fct01ys=zero
      fct01zs=zero
      fct02xs=zero
      fct02ys=zero
      fct02zs=zero
      fct03xs=zero
      fct03ys=zero
      fct03zs=zero
      fct01xx=zero
      fct01xy=zero
      fct01xz=zero
      fct01yy=zero
      fct01yz=zero
      fct01zz=zero

      fctggg =zero
      fctkkk =zero

      fctssx =zero
      fctssy =zero
      fctssz =zero
      fctsix =zero
      fctsiy =zero
      fctsiz =zero
      fctiix =zero
      fctiiy =zero
      fctiiz =zero
! ==========================================

#if KEY_PARALLEL==1
      reanat2 = 2*reanat
      fctqsg =zero
      if(mynod==0)then
         fct02ss=fctbin
      else
         fct02ss=zero
      endif
#else /**/
      fct02ss=fctbin
#endif 
      do i=1,reanat
         if(imove(i)>0) qmove=.true.
      enddo
! ==========================================

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            U=i
            V=fct1lb(j)
            vux=x(V)-x(U)
            vuy=y(V)-y(U)
            vuz=z(V)-z(U)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            if ((fctdvu < fct1cn(fctidx(U)))  .or. &
                (fctdvu < fct1cn(fctidx(V)))) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(U))) then
               fct01vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U)))
               fct02vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))
               fct06vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))              &
                          *fctivu
               ! ------------
               tmp01xs = fct03vuss*vux
               tmp01ys = fct03vuss*vuy
               tmp01zs = fct03vuss*vuz
               tmp02xs = fct05vuss*vux
               tmp02ys = fct05vuss*vuy
               tmp02zs = fct05vuss*vuz
               tmp03xs = (-fct04vuss+fct06vuss)*vux
               tmp03ys = (-fct04vuss+fct06vuss)*vuy
               tmp03zs = (-fct04vuss+fct06vuss)*vuz
               tmp01xx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vux
               tmp01xy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vuy
               tmp01xz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vuz
               tmp01yy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuy*vuy
               tmp01yz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuy*vuz
               tmp01zz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuz*vuz
               ! ------------
               if(Ubl==Vbl) then
                  fct01ss(1,U) = fct01ss(1,U) + fct01vuss
                  fct02ss(1,U) = fct02ss(1,U) + fct02vuss
                  fct03ss(1,U) = fct03ss(1,U) + fct03vuss
                  fct01xs(1,U) = fct01xs(1,U) + tmp01xs
                  fct01ys(1,U) = fct01ys(1,U) + tmp01ys
                  fct01zs(1,U) = fct01zs(1,U) + tmp01zs
                  fct02xs(1,U) = fct02xs(1,U) + tmp02xs
                  fct02ys(1,U) = fct02ys(1,U) + tmp02ys
                  fct02zs(1,U) = fct02zs(1,U) + tmp02zs
                  fct03xs(1,U) = fct03xs(1,U) + tmp03xs
                  fct03ys(1,U) = fct03ys(1,U) + tmp03ys
                  fct03zs(1,U) = fct03zs(1,U) + tmp03zs
                  fct01xx(1,U) = fct01xx(1,U) + tmp01xx
                  fct01xy(1,U) = fct01xy(1,U) + tmp01xy
                  fct01xz(1,U) = fct01xz(1,U) + tmp01xz
                  fct01yy(1,U) = fct01yy(1,U) + tmp01yy
                  fct01yz(1,U) = fct01yz(1,U) + tmp01yz
                  fct01zz(1,U) = fct01zz(1,U) + tmp01zz
                  ! -----------------------------------------
                  fct01ss(2,U) = fct01ss(2,U) + fct01vuss
                  fct02ss(2,U) = fct02ss(2,U) + fct02vuss
                  fct03ss(2,U) = fct03ss(2,U) + fct03vuss
                  fct01xs(2,U) = fct01xs(2,U) + tmp01xs
                  fct01ys(2,U) = fct01ys(2,U) + tmp01ys
                  fct01zs(2,U) = fct01zs(2,U) + tmp01zs
                  fct02xs(2,U) = fct02xs(2,U) + tmp02xs
                  fct02ys(2,U) = fct02ys(2,U) + tmp02ys
                  fct02zs(2,U) = fct02zs(2,U) + tmp02zs
                  fct03xs(2,U) = fct03xs(2,U) + tmp03xs
                  fct03ys(2,U) = fct03ys(2,U) + tmp03ys
                  fct03zs(2,U) = fct03zs(2,U) + tmp03zs
                  fct01xx(2,U) = fct01xx(2,U) + tmp01xx
                  fct01xy(2,U) = fct01xy(2,U) + tmp01xy
                  fct01xz(2,U) = fct01xz(2,U) + tmp01xz
                  fct01yy(2,U) = fct01yy(2,U) + tmp01yy
                  fct01yz(2,U) = fct01yz(2,U) + tmp01yz
                  fct01zz(2,U) = fct01zz(2,U) + tmp01zz
               else
                  fct01ss(1,U) = fct01ss(1,U) + fct01vuss*blcoe(kk)
                  fct02ss(1,U) = fct02ss(1,U) + fct02vuss*blcoe(kk)
                  fct03ss(1,U) = fct03ss(1,U) + fct03vuss*blcoe(kk)
                  fct01xs(1,U) = fct01xs(1,U) + tmp01xs*blcoe(kk)
                  fct01ys(1,U) = fct01ys(1,U) + tmp01ys*blcoe(kk)
                  fct01zs(1,U) = fct01zs(1,U) + tmp01zs*blcoe(kk)
                  fct02xs(1,U) = fct02xs(1,U) + tmp02xs*blcoe(kk)
                  fct02ys(1,U) = fct02ys(1,U) + tmp02ys*blcoe(kk)
                  fct02zs(1,U) = fct02zs(1,U) + tmp02zs*blcoe(kk)
                  fct03xs(1,U) = fct03xs(1,U) + tmp03xs*blcoe(kk)
                  fct03ys(1,U) = fct03ys(1,U) + tmp03ys*blcoe(kk)
                  fct03zs(1,U) = fct03zs(1,U) + tmp03zs*blcoe(kk)
                  fct01xx(1,U) = fct01xx(1,U) + tmp01xx*blcoe(kk)
                  fct01xy(1,U) = fct01xy(1,U) + tmp01xy*blcoe(kk)
                  fct01xz(1,U) = fct01xz(1,U) + tmp01xz*blcoe(kk)
                  fct01yy(1,U) = fct01yy(1,U) + tmp01yy*blcoe(kk)
                  fct01yz(1,U) = fct01yz(1,U) + tmp01yz*blcoe(kk)
                  fct01zz(1,U) = fct01zz(1,U) + tmp01zz*blcoe(kk)
               endif
            endif
            if (fctdvu < fct1cn(fctidx(V))) then
               fct01uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V)))
               fct02uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))
               fct06uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))              &
                          *fctivu
               ! ------------
               tmp01xs = -fct03uvss*vux
               tmp01ys = -fct03uvss*vuy
               tmp01zs = -fct03uvss*vuz
               tmp02xs = -fct05uvss*vux
               tmp02ys = -fct05uvss*vuy
               tmp02zs = -fct05uvss*vuz
               tmp03xs = -(-fct04uvss+fct06uvss)*vux
               tmp03ys = -(-fct04uvss+fct06uvss)*vuy
               tmp03zs = -(-fct04uvss+fct06uvss)*vuz
               tmp01xx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vux
               tmp01xy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vuy
               tmp01xz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vuz
               tmp01yy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuy*vuy
               tmp01yz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuy*vuz
               tmp01zz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuz*vuz
               ! ------------
               if(Ubl==Vbl) then
                  fct01ss(1,V) = fct01ss(1,V) + fct01uvss
                  fct02ss(1,V) = fct02ss(1,V) + fct02uvss
                  fct03ss(1,V) = fct03ss(1,V) + fct03uvss
                  fct01xs(1,V) = fct01xs(1,V) + tmp01xs
                  fct01ys(1,V) = fct01ys(1,V) + tmp01ys
                  fct01zs(1,V) = fct01zs(1,V) + tmp01zs
                  fct02xs(1,V) = fct02xs(1,V) + tmp02xs
                  fct02ys(1,V) = fct02ys(1,V) + tmp02ys
                  fct02zs(1,V) = fct02zs(1,V) + tmp02zs
                  fct03xs(1,V) = fct03xs(1,V) + tmp03xs
                  fct03ys(1,V) = fct03ys(1,V) + tmp03ys
                  fct03zs(1,V) = fct03zs(1,V) + tmp03zs
                  fct01xx(1,V) = fct01xx(1,V) + tmp01xx
                  fct01xy(1,V) = fct01xy(1,V) + tmp01xy
                  fct01xz(1,V) = fct01xz(1,V) + tmp01xz
                  fct01yy(1,V) = fct01yy(1,V) + tmp01yy
                  fct01yz(1,V) = fct01yz(1,V) + tmp01yz
                  fct01zz(1,V) = fct01zz(1,V) + tmp01zz
                  ! -----------------------------------------
                  fct01ss(2,V) = fct01ss(2,V) + fct01uvss
                  fct02ss(2,V) = fct02ss(2,V) + fct02uvss
                  fct03ss(2,V) = fct03ss(2,V) + fct03uvss
                  fct01xs(2,V) = fct01xs(2,V) + tmp01xs
                  fct01ys(2,V) = fct01ys(2,V) + tmp01ys
                  fct01zs(2,V) = fct01zs(2,V) + tmp01zs
                  fct02xs(2,V) = fct02xs(2,V) + tmp02xs
                  fct02ys(2,V) = fct02ys(2,V) + tmp02ys
                  fct02zs(2,V) = fct02zs(2,V) + tmp02zs
                  fct03xs(2,V) = fct03xs(2,V) + tmp03xs
                  fct03ys(2,V) = fct03ys(2,V) + tmp03ys
                  fct03zs(2,V) = fct03zs(2,V) + tmp03zs
                  fct01xx(2,V) = fct01xx(2,V) + tmp01xx
                  fct01xy(2,V) = fct01xy(2,V) + tmp01xy
                  fct01xz(2,V) = fct01xz(2,V) + tmp01xz
                  fct01yy(2,V) = fct01yy(2,V) + tmp01yy
                  fct01yz(2,V) = fct01yz(2,V) + tmp01yz
                  fct01zz(2,V) = fct01zz(2,V) + tmp01zz
               else
                  fct01ss(1,V) = fct01ss(1,V) + fct01uvss*blcoe(kk)
                  fct02ss(1,V) = fct02ss(1,V) + fct02uvss*blcoe(kk)
                  fct03ss(1,V) = fct03ss(1,V) + fct03uvss*blcoe(kk)
                  fct01xs(1,V) = fct01xs(1,V) + tmp01xs*blcoe(kk)
                  fct01ys(1,V) = fct01ys(1,V) + tmp01ys*blcoe(kk)
                  fct01zs(1,V) = fct01zs(1,V) + tmp01zs*blcoe(kk)
                  fct02xs(1,V) = fct02xs(1,V) + tmp02xs*blcoe(kk)
                  fct02ys(1,V) = fct02ys(1,V) + tmp02ys*blcoe(kk)
                  fct02zs(1,V) = fct02zs(1,V) + tmp02zs*blcoe(kk)
                  fct03xs(1,V) = fct03xs(1,V) + tmp03xs*blcoe(kk)
                  fct03ys(1,V) = fct03ys(1,V) + tmp03ys*blcoe(kk)
                  fct03zs(1,V) = fct03zs(1,V) + tmp03zs*blcoe(kk)
                  fct01xx(1,V) = fct01xx(1,V) + tmp01xx*blcoe(kk)
                  fct01xy(1,V) = fct01xy(1,V) + tmp01xy*blcoe(kk)
                  fct01xz(1,V) = fct01xz(1,V) + tmp01xz*blcoe(kk)
                  fct01yy(1,V) = fct01yy(1,V) + tmp01yy*blcoe(kk)
                  fct01yz(1,V) = fct01yz(1,V) + tmp01yz*blcoe(kk)
                  fct01zz(1,V) = fct01zz(1,V) + tmp01zz*blcoe(kk)
               endif
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(V)-x(U)
            vuy=y(V)-y(U)
            vuz=z(V)-z(U)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(U)))  .or. &
                (fctdvu < fct1cn(fctidx(V)))) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(U))) then
               fct01vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U)))
               fct02vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctsvu
               fct03vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctivu
               fct04vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fctsvu*fctivu
               fct05vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))
               fct06vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))              &
                          *fctsvu
               fct07vuss = fctvol(fctidx(V))              &
                          *(one-fctdvu*fct1ci(fctidx(U))) &
                          *fct1fn(fctidx(U))              &
                          *fctivu
               ! ------------
               tmp01xs = fct03vuss*vux
               tmp01ys = fct03vuss*vuy
               tmp01zs = fct03vuss*vuz
               tmp02xs = fct05vuss*vux
               tmp02ys = fct05vuss*vuy
               tmp02zs = fct05vuss*vuz
               tmp03xs = (-fct04vuss+fct06vuss)*vux
               tmp03ys = (-fct04vuss+fct06vuss)*vuy
               tmp03zs = (-fct04vuss+fct06vuss)*vuz
               tmp01xx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vux
               tmp01xy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vuy
               tmp01xz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vux*vuz
               tmp01yy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuy*vuy
               tmp01yz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuy*vuz
               tmp01zz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                        *fctivu)*vuz*vuz
               ! ------------
               if(Ubl==Vbl) then
                  fct01ss(1,U) = fct01ss(1,U) + fct01vuss
                  fct02ss(1,U) = fct02ss(1,U) + fct02vuss
                  fct03ss(1,U) = fct03ss(1,U) + fct03vuss
                  fct01xs(1,U) = fct01xs(1,U) + tmp01xs
                  fct01ys(1,U) = fct01ys(1,U) + tmp01ys
                  fct01zs(1,U) = fct01zs(1,U) + tmp01zs
                  fct02xs(1,U) = fct02xs(1,U) + tmp02xs
                  fct02ys(1,U) = fct02ys(1,U) + tmp02ys
                  fct02zs(1,U) = fct02zs(1,U) + tmp02zs
                  fct03xs(1,U) = fct03xs(1,U) + tmp03xs
                  fct03ys(1,U) = fct03ys(1,U) + tmp03ys
                  fct03zs(1,U) = fct03zs(1,U) + tmp03zs
                  fct01xx(1,U) = fct01xx(1,U) + tmp01xx
                  fct01xy(1,U) = fct01xy(1,U) + tmp01xy
                  fct01xz(1,U) = fct01xz(1,U) + tmp01xz
                  fct01yy(1,U) = fct01yy(1,U) + tmp01yy
                  fct01yz(1,U) = fct01yz(1,U) + tmp01yz
                  fct01zz(1,U) = fct01zz(1,U) + tmp01zz
                  ! -----------------------------------------
                  fct01ss(2,U) = fct01ss(2,U) + fct01vuss
                  fct02ss(2,U) = fct02ss(2,U) + fct02vuss
                  fct03ss(2,U) = fct03ss(2,U) + fct03vuss
                  fct01xs(2,U) = fct01xs(2,U) + tmp01xs
                  fct01ys(2,U) = fct01ys(2,U) + tmp01ys
                  fct01zs(2,U) = fct01zs(2,U) + tmp01zs
                  fct02xs(2,U) = fct02xs(2,U) + tmp02xs
                  fct02ys(2,U) = fct02ys(2,U) + tmp02ys
                  fct02zs(2,U) = fct02zs(2,U) + tmp02zs
                  fct03xs(2,U) = fct03xs(2,U) + tmp03xs
                  fct03ys(2,U) = fct03ys(2,U) + tmp03ys
                  fct03zs(2,U) = fct03zs(2,U) + tmp03zs
                  fct01xx(2,U) = fct01xx(2,U) + tmp01xx
                  fct01xy(2,U) = fct01xy(2,U) + tmp01xy
                  fct01xz(2,U) = fct01xz(2,U) + tmp01xz
                  fct01yy(2,U) = fct01yy(2,U) + tmp01yy
                  fct01yz(2,U) = fct01yz(2,U) + tmp01yz
                  fct01zz(2,U) = fct01zz(2,U) + tmp01zz
               else
                  fct01ss(1,U) = fct01ss(1,U) + fct01vuss*blcoe(kk)
                  fct02ss(1,U) = fct02ss(1,U) + fct02vuss*blcoe(kk)
                  fct03ss(1,U) = fct03ss(1,U) + fct03vuss*blcoe(kk)
                  fct01xs(1,U) = fct01xs(1,U) + tmp01xs*blcoe(kk)
                  fct01ys(1,U) = fct01ys(1,U) + tmp01ys*blcoe(kk)
                  fct01zs(1,U) = fct01zs(1,U) + tmp01zs*blcoe(kk)
                  fct02xs(1,U) = fct02xs(1,U) + tmp02xs*blcoe(kk)
                  fct02ys(1,U) = fct02ys(1,U) + tmp02ys*blcoe(kk)
                  fct02zs(1,U) = fct02zs(1,U) + tmp02zs*blcoe(kk)
                  fct03xs(1,U) = fct03xs(1,U) + tmp03xs*blcoe(kk)
                  fct03ys(1,U) = fct03ys(1,U) + tmp03ys*blcoe(kk)
                  fct03zs(1,U) = fct03zs(1,U) + tmp03zs*blcoe(kk)
                  fct01xx(1,U) = fct01xx(1,U) + tmp01xx*blcoe(kk)
                  fct01xy(1,U) = fct01xy(1,U) + tmp01xy*blcoe(kk)
                  fct01xz(1,U) = fct01xz(1,U) + tmp01xz*blcoe(kk)
                  fct01yy(1,U) = fct01yy(1,U) + tmp01yy*blcoe(kk)
                  fct01yz(1,U) = fct01yz(1,U) + tmp01yz*blcoe(kk)
                  fct01zz(1,U) = fct01zz(1,U) + tmp01zz*blcoe(kk)
               endif
            endif
            if (fctdvu < fct1cn(fctidx(V))) then
               fct01uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V)))
               fct02uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctsvu
               fct03uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctivu
               fct04uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fctsvu*fctivu
               fct05uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))
               fct06uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))              &
                          *fctsvu
               fct07uvss = fctvol(fctidx(U))              &
                          *(one-fctdvu*fct1ci(fctidx(V))) &
                          *fct1fn(fctidx(V))              &
                          *fctivu
               ! ------------
               tmp01xs = -fct03uvss*vux
               tmp01ys = -fct03uvss*vuy
               tmp01zs = -fct03uvss*vuz
               tmp02xs = -fct05uvss*vux
               tmp02ys = -fct05uvss*vuy
               tmp02zs = -fct05uvss*vuz
               tmp03xs = -(-fct04uvss+fct06uvss)*vux
               tmp03ys = -(-fct04uvss+fct06uvss)*vuy
               tmp03zs = -(-fct04uvss+fct06uvss)*vuz
               tmp01xx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vux
               tmp01xy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vuy
               tmp01xz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vux*vuz
               tmp01yy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuy*vuy
               tmp01yz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuy*vuz
               tmp01zz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                        *fctivu)*vuz*vuz
               ! ------------
               if(Ubl==Vbl) then
                  fct01ss(1,V) = fct01ss(1,V) + fct01uvss
                  fct02ss(1,V) = fct02ss(1,V) + fct02uvss
                  fct03ss(1,V) = fct03ss(1,V) + fct03uvss
                  fct01xs(1,V) = fct01xs(1,V) + tmp01xs
                  fct01ys(1,V) = fct01ys(1,V) + tmp01ys
                  fct01zs(1,V) = fct01zs(1,V) + tmp01zs
                  fct02xs(1,V) = fct02xs(1,V) + tmp02xs
                  fct02ys(1,V) = fct02ys(1,V) + tmp02ys
                  fct02zs(1,V) = fct02zs(1,V) + tmp02zs
                  fct03xs(1,V) = fct03xs(1,V) + tmp03xs
                  fct03ys(1,V) = fct03ys(1,V) + tmp03ys
                  fct03zs(1,V) = fct03zs(1,V) + tmp03zs
                  fct01xx(1,V) = fct01xx(1,V) + tmp01xx
                  fct01xy(1,V) = fct01xy(1,V) + tmp01xy
                  fct01xz(1,V) = fct01xz(1,V) + tmp01xz
                  fct01yy(1,V) = fct01yy(1,V) + tmp01yy
                  fct01yz(1,V) = fct01yz(1,V) + tmp01yz
                  fct01zz(1,V) = fct01zz(1,V) + tmp01zz
                  ! -----------------------------------------
                  fct01ss(2,V) = fct01ss(2,V) + fct01uvss
                  fct02ss(2,V) = fct02ss(2,V) + fct02uvss
                  fct03ss(2,V) = fct03ss(2,V) + fct03uvss
                  fct01xs(2,V) = fct01xs(2,V) + tmp01xs
                  fct01ys(2,V) = fct01ys(2,V) + tmp01ys
                  fct01zs(2,V) = fct01zs(2,V) + tmp01zs
                  fct02xs(2,V) = fct02xs(2,V) + tmp02xs
                  fct02ys(2,V) = fct02ys(2,V) + tmp02ys
                  fct02zs(2,V) = fct02zs(2,V) + tmp02zs
                  fct03xs(2,V) = fct03xs(2,V) + tmp03xs
                  fct03ys(2,V) = fct03ys(2,V) + tmp03ys
                  fct03zs(2,V) = fct03zs(2,V) + tmp03zs
                  fct01xx(2,V) = fct01xx(2,V) + tmp01xx
                  fct01xy(2,V) = fct01xy(2,V) + tmp01xy
                  fct01xz(2,V) = fct01xz(2,V) + tmp01xz
                  fct01yy(2,V) = fct01yy(2,V) + tmp01yy
                  fct01yz(2,V) = fct01yz(2,V) + tmp01yz
                  fct01zz(2,V) = fct01zz(2,V) + tmp01zz
               else
                  fct01ss(1,V) = fct01ss(1,V) + fct01uvss*blcoe(kk)
                  fct02ss(1,V) = fct02ss(1,V) + fct02uvss*blcoe(kk)
                  fct03ss(1,V) = fct03ss(1,V) + fct03uvss*blcoe(kk)
                  fct01xs(1,V) = fct01xs(1,V) + tmp01xs*blcoe(kk)
                  fct01ys(1,V) = fct01ys(1,V) + tmp01ys*blcoe(kk)
                  fct01zs(1,V) = fct01zs(1,V) + tmp01zs*blcoe(kk)
                  fct02xs(1,V) = fct02xs(1,V) + tmp02xs*blcoe(kk)
                  fct02ys(1,V) = fct02ys(1,V) + tmp02ys*blcoe(kk)
                  fct02zs(1,V) = fct02zs(1,V) + tmp02zs*blcoe(kk)
                  fct03xs(1,V) = fct03xs(1,V) + tmp03xs*blcoe(kk)
                  fct03ys(1,V) = fct03ys(1,V) + tmp03ys*blcoe(kk)
                  fct03zs(1,V) = fct03zs(1,V) + tmp03zs*blcoe(kk)
                  fct01xx(1,V) = fct01xx(1,V) + tmp01xx*blcoe(kk)
                  fct01xy(1,V) = fct01xy(1,V) + tmp01xy*blcoe(kk)
                  fct01xz(1,V) = fct01xz(1,V) + tmp01xz*blcoe(kk)
                  fct01yy(1,V) = fct01yy(1,V) + tmp01yy*blcoe(kk)
                  fct01yz(1,V) = fct01yz(1,V) + tmp01yz*blcoe(kk)
                  fct01zz(1,V) = fct01zz(1,V) + tmp01zz*blcoe(kk)
               endif
            endif
         enddo
      enddo

#if KEY_PARALLEL==1
      call gcombr(fct01ss,reanat2) ! MH2023: extend the gcomb interface for fix
      call gcombr(fct02ss,reanat2)
      call gcombr(fct03ss,reanat2)
      call gcombr(fct01xs,reanat2)
      call gcombr(fct01ys,reanat2)
      call gcombr(fct01zs,reanat2)
      call gcombr(fct02xs,reanat2)
      call gcombr(fct02ys,reanat2)
      call gcombr(fct02zs,reanat2)
      call gcombr(fct03xs,reanat2)
      call gcombr(fct03ys,reanat2)
      call gcombr(fct03zs,reanat2)
      call gcombr(fct01xx,reanat2)
      call gcombr(fct01xy,reanat2)
      call gcombr(fct01xz,reanat2)
      call gcombr(fct01yy,reanat2)
      call gcombr(fct01yz,reanat2)
      call gcombr(fct01zz,reanat2)
#endif 
      
#if KEY_DEBUG != 1
      
! Do IMOVE expansion of code
! disable expand when debug is active
   IF(QMOVE) THEN
#define FACTS_IMOVE 1
#include "fctblock.inc"
#undef FACTS_IMOVE
   ELSE
#include "fctblock.inc"
   ENDIF
   
#else  /* KEY_DEBUG */
   
#define FACTS_IMOVE 1
#include "fctblock.inc"
#undef FACTS_IMOVE
   
#endif  /* KEY_DEBUG */ 
   
      do i=1,reanat
         dx(i)=dx(i)+fctssx(i)+fctsix(i)+fctiix(i)
         dy(i)=dy(i)+fctssy(i)+fctsiy(i)+fctiiy(i)
         dz(i)=dz(i)+fctssz(i)+fctsiz(i)+fctiiz(i)
      enddo

      fctpol=fctpl1+fctpl2
      fctnpl=fctnp1

      ! write(outu,*) 'fctblock.src: FCTSFE - fctpol: ', fctpol
      ! write(outu,*) 'fctblock.src: FCTSFE - fctnpl: ', fctnpl
   endif

   ! ---------------------------------------------------------------------
   ! Array Allocation

   ! Real 1D
   call chmdealloc('fctblock.src','FCTBLK','fctssx', natom, crl=fctssx)
   call chmdealloc('fctblock.src','FCTBLK','fctssy', natom, crl=fctssy)
   call chmdealloc('fctblock.src','FCTBLK','fctssz', natom, crl=fctssz)
   call chmdealloc('fctblock.src','FCTBLK','fctsix', natom, crl=fctsix)
   call chmdealloc('fctblock.src','FCTBLK','fctsiy', natom, crl=fctsiy)
   call chmdealloc('fctblock.src','FCTBLK','fctsiz', natom, crl=fctsiz)
   call chmdealloc('fctblock.src','FCTBLK','fctiix', natom, crl=fctiix)
   call chmdealloc('fctblock.src','FCTBLK','fctiiy', natom, crl=fctiiy)
   call chmdealloc('fctblock.src','FCTBLK','fctiiz', natom, crl=fctiiz)

   ! Real 2D
   call chmdealloc('fctblock.src','FCTBLK','fct01ss' , 2, natom, crl=fct01ss )
   call chmdealloc('fctblock.src','FCTBLK','fct02ss' , 2, natom, crl=fct02ss )
   call chmdealloc('fctblock.src','FCTBLK','fct03ss' , 2, natom, crl=fct03ss )
   call chmdealloc('fctblock.src','FCTBLK','fct01xs' , 2, natom, crl=fct01xs )
   call chmdealloc('fctblock.src','FCTBLK','fct01ys' , 2, natom, crl=fct01ys )
   call chmdealloc('fctblock.src','FCTBLK','fct01zs' , 2, natom, crl=fct01zs )
   call chmdealloc('fctblock.src','FCTBLK','fct02xs' , 2, natom, crl=fct02xs )
   call chmdealloc('fctblock.src','FCTBLK','fct02ys' , 2, natom, crl=fct02ys )
   call chmdealloc('fctblock.src','FCTBLK','fct02zs' , 2, natom, crl=fct02zs )
   call chmdealloc('fctblock.src','FCTBLK','fct03xs' , 2, natom, crl=fct03xs )
   call chmdealloc('fctblock.src','FCTBLK','fct03ys' , 2, natom, crl=fct03ys )
   call chmdealloc('fctblock.src','FCTBLK','fct03zs' , 2, natom, crl=fct03zs )
   call chmdealloc('fctblock.src','FCTBLK','fct01xx ', 2, natom, crl=fct01xx )
   call chmdealloc('fctblock.src','FCTBLK','fct01xy' , 2, natom, crl=fct01xy )
   call chmdealloc('fctblock.src','FCTBLK','fct01xz' , 2, natom, crl=fct01xz )
   call chmdealloc('fctblock.src','FCTBLK','fct01yy' , 2, natom, crl=fct01yy )
   call chmdealloc('fctblock.src','FCTBLK','fct01yz' , 2, natom, crl=fct01yz )
   call chmdealloc('fctblock.src','FCTBLK','fct01zz' , 2, natom, crl=fct01zz )
   call chmdealloc('fctblock.src','FCTBLK','fctis1'  , 2, natom, crl=fctis1  )
   call chmdealloc('fctblock.src','FCTBLK','fctis2'  , 2, natom, crl=fctis2  )
   call chmdealloc('fctblock.src','FCTBLK','fctmov'  , 2, natom, crl=fctmov  )
   call chmdealloc('fctblock.src','FCTBLK','fctmos'  , 2, natom, crl=fctmos  )
   call chmdealloc('fctblock.src','FCTBLK','fctqsg'  , 2, natom, crl=fctqsg  )
   call chmdealloc('fctblock.src','FCTBLK','fctisg'  , 2, natom, crl=fctisg  )
   call chmdealloc('fctblock.src','FCTBLK','fctqdv'  , 2, natom, crl=fctqdv  )
   call chmdealloc('fctblock.src','FCTBLK','fctqds'  , 2, natom, crl=fctqds  )
   call chmdealloc('fctblock.src','FCTBLK','fctusg'  , 2, natom, crl=fctusg  )
   call chmdealloc('fctblock.src','FCTBLK','fctudv'  , 2, natom, crl=fctudv  )
   call chmdealloc('fctblock.src','FCTBLK','fctuds'  , 2, natom, crl=fctuds  )
   call chmdealloc('fctblock.src','FCTBLK','fctvsg'  , 2, natom, crl=fctvsg  )
   call chmdealloc('fctblock.src','FCTBLK','fctvdv'  , 2, natom, crl=fctvdv  )
   call chmdealloc('fctblock.src','FCTBLK','fctvds'  , 2, natom, crl=fctvds  )
   call chmdealloc('fctblock.src','FCTBLK','fctwsg'  , 2, natom, crl=fctwsg  )
   call chmdealloc('fctblock.src','FCTBLK','fctwdv'  , 2, natom, crl=fctwdv  )
   call chmdealloc('fctblock.src','FCTBLK','fctwds'  , 2, natom, crl=fctwds  )
   call chmdealloc('fctblock.src','FCTBLK','fctqsx'  , 2, natom, crl=fctqsx  )
   call chmdealloc('fctblock.src','FCTBLK','fctqsy'  , 2, natom, crl=fctqsy  )
   call chmdealloc('fctblock.src','FCTBLK','fctqsz'  , 2, natom, crl=fctqsz  )
   call chmdealloc('fctblock.src','FCTBLK','fctusx'  , 2, natom, crl=fctusx  )
   call chmdealloc('fctblock.src','FCTBLK','fctusy'  , 2, natom, crl=fctusy  )
   call chmdealloc('fctblock.src','FCTBLK','fctusz'  , 2, natom, crl=fctusz  )
   call chmdealloc('fctblock.src','FCTBLK','fctvsx'  , 2, natom, crl=fctvsx  )
   call chmdealloc('fctblock.src','FCTBLK','fctvsy'  , 2, natom, crl=fctvsy  )
   call chmdealloc('fctblock.src','FCTBLK','fctvsz'  , 2, natom, crl=fctvsz  )
   call chmdealloc('fctblock.src','FCTBLK','fctwsx'  , 2, natom, crl=fctwsx  )
   call chmdealloc('fctblock.src','FCTBLK','fctwsy'  , 2, natom, crl=fctwsy  )
   call chmdealloc('fctblock.src','FCTBLK','fctwsz'  , 2, natom, crl=fctwsz  )
   call chmdealloc('fctblock.src','FCTBLK','tmppl1'  , 2, natom, crl=tmppl1  )
   call chmdealloc('fctblock.src','FCTBLK','tmpnp1'  , 2, natom, crl=tmpnp1  )
   call chmdealloc('fctblock.src','FCTBLK','tmpplssx', 2, natom, crl=tmpplssx)
   call chmdealloc('fctblock.src','FCTBLK','tmpnpssx', 2, natom, crl=tmpnpssx)
   call chmdealloc('fctblock.src','FCTBLK','tmpplssy', 2, natom, crl=tmpplssy)
   call chmdealloc('fctblock.src','FCTBLK','tmpnpssy', 2, natom, crl=tmpnpssy)
   call chmdealloc('fctblock.src','FCTBLK','tmpplssz', 2, natom, crl=tmpplssz)
   call chmdealloc('fctblock.src','FCTBLK','tmpnpssz', 2, natom, crl=tmpnpssz)
   call chmdealloc('fctblock.src','FCTBLK','fctvdwen', 2, natom, crl=fctvdwen)
   call chmdealloc('fctblock.src','FCTBLK','fctvdwdf', 2, natom, crl=fctvdwdf)

! ---------------------------------------------------------------------
   RETURN
   END Subroutine fctblk
#endif /* (facts_block)*/
#endif /* (block_main)*/
