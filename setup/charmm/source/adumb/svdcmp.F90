#if KEY_ADUMB==1 /*adumb*/
!
!     C. Bartels, 1996, 
!     Adapted from source code of PROSA of P. Guentert (with permission)
!
subroutine svdcmp (a,m,n,mp,np,w,v,work)
  use chm_kinds
  use number
  use stream
  implicit none
  !
  integer m,n,mp,np,jj
  integer i,j,k,l,nm,its
  real(chm_real) :: s,f,g,h,tmp,c,anorm,scale,x,y,z
  integer,parameter :: nmax=1000,maxits=50
  real(chm_real),parameter :: eps=1.0E-7
  real(chm_real)    a(mp,n),w(n),v(np,n),rv1(nmax),work(m),rnorm
  !
  if (m.lt.n) then
     call wrndie(-5,'svdcmp<svdcmp.src>','*** Fatal error: More columns than rows in SVD.')
     return
  else if (n.gt.nmax) then
     call wrndie(-5,'svdcmp<svdcmp.src>','*** Fatal error: Too many columns in SVD.')
     return
  end if
  g=zero
  scale=zero
  anorm=zero
  do i=1,n
     l=i+1
     rv1(i)=scale*g
     g=zero
     s=zero
     scale=zero
     if (i.le.m) then
        do k=i,m
           scale=scale+abs(a(k,i))
        enddo
        if (scale.ne.zero) then
           tmp=1.0/scale
           do k=i,m
              a(k,i)=a(k,i)*tmp
              s=s+a(k,i)*a(k,i)
           enddo
           f=a(i,i)
           g=-sign(sqrt(s),f)
           h=f*g-s
           a(i,i)=f-g
           if (i.ne.n) then
              do j=l,n
                 s=zero
                 do k=i,m
                    s=s+a(k,i)*a(k,j)
                 enddo
                 f=s/h
                 do k=i,m
                    a(k,j)=a(k,j)+f*a(k,i)
                 enddo
              enddo
           end if
           do k=i,m
              a(k,i)=scale*a(k,i)
           enddo
        end if
     end if
     w(i)=scale*g
     g=zero
     s=zero
     scale=zero
     if ((i.le.m) .and. (i.ne.n)) then
        do k=l,n
           scale=scale+abs(a(i,k))
        enddo
        if (scale.ne.zero) then
           tmp=1.0/scale
           do k=l,n
              a(i,k)=a(i,k)*tmp
              s=s+a(i,k)*a(i,k)
           enddo
           f=a(i,l)
           g=-sign(sqrt(s),f)
           h=f*g-s
           a(i,l)=f-g
           tmp=1.0/h
           do k=l,n
              rv1(k)=a(i,k)*tmp
           enddo
           if (i.ne.m) then
              do j=l,m
                 work(j)=zero
              enddo
              do k=l,n
                 do j=l,m
                    work(j)=work(j)+a(j,k)*a(i,k)
                 enddo
              enddo
              do k=l,n
                 do j=l,m
                    a(j,k)=a(j,k)+work(j)*rv1(k)
                 enddo
              enddo
           end if
           do k=l,n
              a(i,k)=scale*a(i,k)
           enddo
        end if
     end if
     rnorm=anorm
     anorm=max(rnorm,(abs(w(i))+abs(rv1(i))))
  enddo
  !pg
  anorm=anorm*eps
  !pg
  do i=n,1,-1
     if (i.lt.n) then
        if (g.ne.zero) then
           do j=l,n
              v(j,i)=(a(i,j)/a(i,l))/g
           enddo
           do j=l,n
              s=zero
              do k=l,n
                 s=s+a(i,k)*v(k,j)
              enddo
              do k=l,n
                 v(k,j)=v(k,j)+s*v(k,i)
              enddo
           enddo
        end if
        do j=l,n
           v(i,j)=zero
        enddo
        do j=l,n
           v(j,i)=zero
        enddo
     end if
     v(i,i)=1.0
     g=rv1(i)
     l=i
  enddo
  loop39: do i=n,1,-1
     l=i+1
     g=w(i)
     if (i.lt.n) then
        a(i,l:n)=zero
     end if
     if (g.ne.zero) then
        g=1.0/g
        if (i.ne.n) then
           do j=l,n
              s=zero
              do k=l,m
                 s=s+a(k,i)*a(k,j)
              enddo
              f=(s/a(i,i))*g
              do k=i,m
                 a(k,j)=a(k,j)+f*a(k,i)
              enddo
           enddo
        end if
        a(i:m,i)=a(i:m,i)*g
     else
        a(i:m,i)=zero
     end if
     a(i,i)=a(i,i)+1.0
  enddo loop39
  loop49: do k=n,1,-1
     do its=1,maxits
        loop41: do l=k,1,-1
           nm=l-1
           if (abs(rv1(l)).le.anorm) go to 2
           if (abs(w(nm)).le.anorm) exit loop41
        enddo loop41
        c=zero
        s=1.0
        loop43: do i=l,k
           f=s*rv1(i)
           if (abs(f).le.anorm) then
              g=w(i)
              h=sqrt(f*f+g*g)
              w(i)=h
              h=1.0/h
              c=(g*h)
              s=-(f*h)
              loop42: do j=1,m
                 y=a(j,nm)
                 z=a(j,i)
                 a(j,nm)=(y*c)+(z*s)
                 a(j,i)=-(y*s)+(z*c)
              enddo loop42
           end if
        enddo loop43
2       z=w(k)
        if (l.eq.k) then
           if (z.lt.zero) then
              w(k)=-z
              v(1:n,k)=-v(1:n,k)
           end if
           go to 3
        end if
        if (its.eq.maxits) then
           call wrndie(-5,'svdcmp<svdcmp.src>', &
                '*** Fatal error: Too many iterations in SVD.')
           return
        end if
        x=w(l)
        nm=k-1
        y=w(nm)
        g=rv1(nm)
        h=rv1(k)
        f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y)
        g=sqrt(f*f+1.0)
        f=((x-z)*(x+z)+h*((y/(f+sign(g,f)))-h))/x
        c=1.0
        s=1.0
        do j=l,nm
           i=j+1
           g=rv1(i)
           y=w(i)
           h=s*g
           g=c*g
           z=sqrt(f*f+h*h)
           rv1(j)=z
           c=f/z
           s=h/z
           f=(x*c)+(g*s)
           g=-(x*s)+(g*c)
           h=y*s
           y=y*c
           do jj=1,n
              x=v(jj,j)
              z=v(jj,i)
              v(jj,j)=(x*c)+(z*s)
              v(jj,i)=-(x*s)+(z*c)
           enddo
           z=sqrt(f*f+h*h)
           w(j)=z
           if (z.ne.zero) then
              z=1.0/z
              c=f*z
              s=h*z
           end if
           f=(c*g)+(s*y)
           x=-(s*g)+(c*y)
           do jj=1,m
              y=a(jj,j)
              z=a(jj,i)
              a(jj,j)=(y*c)+(z*s)
              a(jj,i)=-(y*s)+(z*c)
           enddo
        enddo
        rv1(l)=zero
        rv1(k)=f
        w(k)=x
     enddo
3    continue
  enddo loop49
  !
  return
end subroutine svdcmp

#endif /* (adumb)*/

subroutine null_svd
 return
END subroutine null_svd

