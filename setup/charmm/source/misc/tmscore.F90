!
! Adapted in 2012 by Mike Garrahan and Alex Dickson
! from http://zhanglab.ccmb.med.umich.edu/TM-score/
!************************************************************************
!     This module compares two protein structures and identifies the
!     best superposition that has the highest TM-score. By default,
!     TM-score is normalized by the second protein.
!     For comments/suggestions, please contact email: zhng@umich.edu.
!
!     Reference:
!     Yang Zhang, Jeffrey Skolnick, Proteins, 2004 57:702-10,
!     DOI 10.1002/prot.20264
!
!     Permission to use, copy, modify, and distribute this module for
!     any purpose, with or without fee, is hereby granted, provided that
!     the notices on the head, the reference information, and this
!     copyright notice appear in all copies or substantial portions of
!     the Software. It is provided "as is" without express or implied
!     warranty.
!****************** Updating history ************************************
!     2012/05/07: Improved RMSD calculation subroutine which speeds up
!                 TM-score program by 30%.
!     2012/06/05: Added an option '-l L' which calculates TM-score normalized
!                 by a specific length 'L'.
!************************************************************************

module tmscore_m
   use chm_kinds
   use number
   implicit none

   private
   integer, parameter :: nmax = 5000

   ! These were COMMON blocks in zhanglab's code
   real :: d, d0, d0_fix
   integer :: nseqA, nseqB
   integer, allocatable, dimension(:) :: iA, iB, i_ali
   integer :: n_ali, n_cut ![1,n_ali],align residues for the score
   real(chm_real) :: score, score_maxsub, score_fix, score10
   character(len=*),parameter :: file_name="tmscore"

   public :: tmscore

contains

  subroutine allocate_tmscore(n)
    use chm_kinds
    use memory

    integer, intent(in) :: n
    character(len=*),parameter :: routine_name="allocate_tmcore"

    call chmalloc(file_name,routine_name,'iA',n,intg=iA)
    call chmalloc(file_name,routine_name,'iB',n,intg=iB)
    call chmalloc(file_name,routine_name,'i_ali',n,intg=i_ali)

  end subroutine allocate_tmscore

  subroutine deallocate_tmscore(n)
    use chm_kinds
    use memory

    integer, intent(in) :: n
    character(len=*),parameter :: routine_name="deallocate_tmcore"

    call chmdealloc(file_name,routine_name,'iA',n,intg=iA)
    call chmdealloc(file_name,routine_name,'iB',n,intg=iB)
    call chmdealloc(file_name,routine_name,'i_ali',n,intg=i_ali)

  end subroutine deallocate_tmscore

   subroutine tmscore(x, y, z, xcomp, ycomp, zcomp, natom, nres, islct)

     use memory
      real(chm_real), intent(in) :: x(natom), y(natom), z(natom)
      real(chm_real), intent(in) :: xcomp(natom), ycomp(natom), zcomp(natom)
      integer, intent(in) :: natom, nres
      integer, intent(in) :: islct(:)

      logical :: selection(size(islct))
      integer :: num_selected, i
      ! TODO allocatable
      ! Make them dynamic arrays
      real(chm_real), dimension(natom) :: xa, ya, za, xb, yb, zb
!      character(len=*),parameter :: routine_name="tmscore"

      call allocate_tmscore(natom)

!      call chmdealloc(file_name,routine_name,'xa',natom,crl=xa)
!      call chmdealloc(file_name,routine_name,'ya',natom,crl=ya)
!      call chmdealloc(file_name,routine_name,'za',natom,crl=za)
!      call chmdealloc(file_name,routine_name,'xb',natom,crl=xb)
!      call chmdealloc(file_name,routine_name,'yb',natom,crl=yb)
!      call chmdealloc(file_name,routine_name,'zb',natom,crl=zb)

      selection = islct == 1
      num_selected = count(selection)
      nseqA = num_selected
      nseqB = num_selected
      n_ali = num_selected
      do i = 1, num_selected
         iA(i) = i
         iB(i) = i
         i_ali(i) = i
      enddo

      xb = pack(x, selection)
      yb = pack(y, selection)
      zb = pack(z, selection)
      xa = pack(xcomp, selection)
      ya = pack(ycomp, selection)
      za = pack(zcomp, selection)

      call tmscore_main(xa, ya, za, xb, yb, zb, nres, natom)

      call deallocate_tmscore(natom)
   end subroutine tmscore

   subroutine tmscore_main(xa, ya, za, xb, yb, zb, nres, natom)
      use stream, only: PRNLEV, OUTU
      use param_store, only: set_param

      implicit none

      real(chm_real), intent(in), dimension(:) :: xa, ya, za, xb, yb, zb
      integer, intent(in) :: nres, natom

      ! Dynamic arrays
      integer, dimension(natom) :: L_ini
      real(chm_real), dimension(natom) :: xt, yt, zt, w

      ! RMSD
      real(chm_real), dimension(3,natom) :: r_1,r_2,r_3
      real(chm_real) :: u(3,3),t(3),rms,drms !armsd is real
      character(len=3), parameter :: aa(-1:20) = [ &
            'BCK','GLY','ALA','SER','CYS', &
            'VAL','THR','ILE','PRO','MET', &
            'ASP','ASN','LEU','LYS','GLU', &
            'GLN','ARG','HIS','PHE','TYR', &
            'TRP','CYX' ]
      character(len=1), parameter :: slc(-1:20) = [ &
            'X','G','A','S','C', &
            'V','T','I','P','M', &
            'D','N','L','K','E', &
            'Q','R','H','F','Y', &
            'W','C' ]

      ! formerly implicit
      integer :: i, j, k
      integer, dimension(natom) :: k_ali, k_ali0
      integer :: m, m_out, m_fix, m_len
      integer :: l0_fix, l_ini_min
      integer :: i_init, l_init, il_max, il, ll, ka, ka0, ier, it
      integer :: narg, n_it, n_init_max, n_init, neq
      real :: d0_search, d_output, armsd, dis, d_tmp
      real :: rmsd, rmsd_ali
      real(chm_real) :: score10_max
      real(chm_real) :: score_max, score_fix_max

      w = ONE

      ! options
      m_out = -1
      m_fix = -1
      m_len = -1
      i = 0
      j = 0

      ! d0
      if (nres > 15) then
         d0 = 1.24*(nres-15)**(ONE/THREE)-1.8
      else
         d0 = HALF
      endif
      if (m_len == 1) then
         d0 = 1.24*(l0_fix-15)**(ONE/THREE)-1.8
      endif
      if (d0 < HALF) d0 = HALF
      if (m_fix == 1) d0 = d0_fix
      ! d0_search
      d0_search = d0
      if (d0_search > 8) d0_search = 8
      if (d0_search < 4.5) d0_search = 4.5
      ! iterative parameters
      n_it = 20                   !maximum number of iterations
      d_output = 5                !for output alignment
      if (m_fix == 1) d_output = d0_fix
      n_init_max = 6              !maximum number of L_init
      n_init = 0
      L_ini_min = 4
      if (n_ali < 4) L_ini_min = n_ali
      do i = 1,n_init_max-1
         n_init = n_init+1
         L_ini(n_init) = n_ali/2**(n_init-1)
         if (L_ini(n_init) <= L_ini_min) then
            L_ini(n_init) = L_ini_min
            goto 402
         endif
      enddo
      n_init = n_init+1
      L_ini(n_init) = L_ini_min
 402  continue

      !
      ! find the maximum score starting from local structures superposition
      !
      score_max = -1              !TM-score
      score10_max = -1            !TM-score10
      init_len: do i_init = 1, n_init
        L_init = L_ini(i_init)
        iL_max = n_ali-L_init+1
        shift: do iL = 1, iL_max  !on aligned residues, [1,nseqA]
           LL = 0
           ka = 0
           do i = 1,L_init
              k = iL+i-1          ![1,n_ali] common aligned
              r_1(1,i) = xa(iA(k))
              r_1(2,i) = ya(iA(k))
              r_1(3,i) = za(iA(k))
              r_2(1,i) = xb(iB(k))
              r_2(2,i) = yb(iB(k))
              r_2(3,i) = zb(iB(k))
              ka = ka+1
              k_ali(ka) = k
              LL = LL+1
           enddo
           if (i_init == 1) then  !global superposition
              call u3b(w,r_1,r_2,LL,2,rms,u,t,ier) !0:rmsd; 1:u,t; 2:rmsd,u,t
              armsd = sqrt(rms/LL)
              rmsd_ali = armsd
           else
              call u3b(w,r_1,r_2,LL,1,rms,u,t,ier) !u rotate r_1 to r_2
           endif
           do j = 1,nseqA
              xt(j) = t(1) + u(1,1)*xa(j) + u(1,2)*ya(j) + u(1,3)*za(j)
              yt(j) = t(2) + u(2,1)*xa(j) + u(2,2)*ya(j) + u(2,3)*za(j)
              zt(j) = t(3) + u(3,1)*xa(j) + u(3,2)*ya(j) + u(3,3)*za(j)
           enddo
           d = d0_search-1
           ! init, get scores, n_cut+i_ali(i) for iteration
           call score_fun(xt, yt, zt, xb, yb, zb)
           if (score_max < score) then
              score_max = score
              ka0 = ka
              do i = 1,ka0
                 k_ali0(i) = k_ali(i)
              enddo
           endif
           if (score10_max < score10) score10_max = score10
           ! iteration for extending
           d = d0_search+1
           iter: do it = 1, n_it
              LL = 0
              ka = 0
              do i = 1,n_cut
                 m = i_ali(i)     ![1,n_ali]
                 r_1(1,i) = xa(iA(m))
                 r_1(2,i) = ya(iA(m))
                 r_1(3,i) = za(iA(m))
                 r_2(1,i) = xb(iB(m))
                 r_2(2,i) = yb(iB(m))
                 r_2(3,i) = zb(iB(m))
                 ka = ka+1
                 k_ali(ka) = m
                 LL = LL+1
              enddo
              call u3b(w,r_1,r_2,LL,1,rms,u,t,ier) !u rotate r_1 to r_2
              do j = 1,nseqA
                 xt(j) = t(1) + u(1,1)*xa(j) + u(1,2)*ya(j) + u(1,3)*za(j)
                 yt(j) = t(2) + u(2,1)*xa(j) + u(2,2)*ya(j) + u(2,3)*za(j)
                 zt(j) = t(3) + u(3,1)*xa(j) + u(3,2)*ya(j) + u(3,3)*za(j)
              enddo
              ! init, get scores, n_cut+i_ali(i) for iteration
              call score_fun(xt, yt, zt, xb, yb, zb)
              if (score_max < score) then
                 score_max = score
                 ka0 = ka
                 do i = 1,ka
                    k_ali0(i) = k_ali(i)
                 enddo
              endif
              if (score10_max < score10) score10_max = score10
              if (it == n_it) exit iter
              if (n_cut == ka) then
                 neq = 0
                 do i = 1,n_cut
                    if (i_ali(i) == k_ali(i)) neq = neq+1
                 enddo
                 if (n_cut == neq) exit iter
              endif
           enddo iter
         enddo shift
      enddo init_len

      !
      ! Output
      !
      if (PRNLEV >= 2) write (OUTU,513) rmsd_ali
 513  format ('RMSD of  the common residues = ',F8.3)
      if (PRNLEV >= 2) write (OUTU,*)
      if (m_len == 1) then
         score_max = score_max*real(nseqB,chm_real)/real(l0_fix,chm_real)
      endif
      if (PRNLEV >= 2) write (OUTU,504) score_max, d0, score10_max
 504  format ('TM-score = ',f6.4,'  (d0 = ',f5.2,',',' TM10 = ',f6.4,')')

      call set_param('TMSCORE', score_max)
      call set_param('TMD0', real(d0,chm_real))
      call set_param('TM10', score10_max)

   end subroutine tmscore_main

   !
   ! 1, collect those residues with dis<d;
   ! 2, calculate score_TM
   !
   subroutine score_fun(xt, yt, zt, xb, yb, zb)
      real(chm_real), intent(in), dimension(:) :: xt, yt, zt, xb, yb, zb
      real(chm_real) :: score_maxsub_sum, score_sum, score_sum10
      real :: d_tmp, dissq, d_tmpsq
      integer :: i, j, k

      d_tmp = d
      d_tmpsq = d**2
      do
         n_cut = 0                   !number of residue-pairs dis<d, for iteration
         score_sum = 0               !TMscore
         score_sum10 = 0             !TMscore10
         do k = 1,n_ali
            i = iA(k)                ![1,nseqA] reoder number of structureA
            j = iB(k)                ![1,nseqB]
            dissq = (xt(i)-xb(j))**2 + (yt(i)-yb(j))**2 + (zt(i)-zb(j))**2
            ! for iteration
            if (dissq < d_tmpsq) then
               n_cut = n_cut+1
               i_ali(n_cut) = k      ![1,n_ali], mark the residue-pairs in dis<d
            endif
            ! for TM-score
            score_sum = score_sum + 1/(1 + dissq/(d0**2))
            ! for TM-score10
            if (dissq < 100) then
               score_sum10 = score_sum10 + 1/(1 + dissq/(d0**2))
            endif
         enddo
         if (n_cut >= 3 .or. n_ali <= 3) exit
         d_tmp = d_tmp + HALF
         d_tmpsq = d_tmp**2
      enddo
      score = score_sum / nseqB      !TM-score
      score10 = score_sum10 / nseqB  !TM-score10

      return
   end subroutine score_fun

   ! Calculate sum of (r_d-r_m)^2
   !  w    - w(m) is weight for atom pair  c m                    (given)
   !  x    - x(i,m) are coordinates of atom c m in set x          (given)
   !  y    - y(i,m) are coordinates of atom c m in set y          (given)
   !  n    - n is number of atom pairs                            (given)
   !  mode  - 0:calculate rms     only                            (given,short)
   !          1:calculate     u,t only                            (given,medium)
   !          2:calculate rms,u,t                                 (given,longer)
   !  rms   - sum of w*(ux+t-y)**2 over all atom pairs            (result)
   !  u    - u(i,j) is   rotation  matrix for best superposition  (result)
   !  t    - t(i)   is translation vector for best superposition  (result)
   !  ier  - 0: a unique optimal superposition has been determined(result)
   !       -1: superposition is not unique but optimal
   !       -2: no result obtained because of negative weights w
   !           or all weights equal to ZERO.
   !
   subroutine u3b(w, x, y, n, mode, rms, u, t, ier)
      real(chm_real), intent(in) :: w(:), x(:,:), y(:,:)
      integer, intent(in) :: n, mode
      real(chm_real), intent(out) :: rms, u(3,3), t(3)
      integer, intent(out) :: ier

      integer i, j, k, l, m1, m
      real(chm_real) :: r(3,3), xc(3), yc(3), wc
      real(chm_real) :: a(3,3), b(3,3), e(3), rr(6), ss(6)
      real(chm_real) :: e0, d, spur, det, cof, h, g
      real(chm_real) :: cth, sth, sqrth, p, sigma
      real(chm_real) :: c1x, c1y, c1z, c2x, c2y, c2z
      real(chm_real) :: s1x, s1y, s1z, s2x, s2y, s2z
      real(chm_real) :: sxx, sxy, sxz, syx, syy, syz, szx, szy, szz

      real(chm_real), parameter :: sqrt3 = 1.73205080756888d+00
      real(chm_real), parameter :: tol = 1.0d-2
      integer, parameter :: ip(9) = [1, 2, 4, 2, 3, 5, 4, 5, 6]
      integer, parameter :: ip2312(4) = [2, 3, 1, 2]

      wc = ZERO
      rms = ZERO
      e0 = ZERO
      s1x = ZERO
      s1y = ZERO
      s1z = ZERO
      s2x = ZERO
      s2y = ZERO
      s2z = ZERO
      sxx = ZERO
      sxy = ZERO
      sxz = ZERO
      syx = ZERO
      syy = ZERO
      syz = ZERO
      szx = ZERO
      szy = ZERO
      szz = ZERO

      do i = 1, 3
         xc(i) = ZERO
         yc(i) = ZERO
         t(i) = ZERO
         do j = 1, 3
            r(i,j) = ZERO
            u(i,j) = ZERO
            a(i,j) = ZERO
            if (i == j) then
               u(i,j) = ONE
               a(i,j) = ONE
            endif
         enddo
      enddo

      ier = -1
      if (n < 1) return
      ier = -2

      do m = 1, n
         c1x = x(1, m)
         c1y = x(2, m)
         c1z = x(3, m)

         c2x = y(1, m)
         c2y = y(2, m)
         c2z = y(3, m)

         s1x = s1x + c1x
         s1y = s1y + c1y;
         s1z = s1z + c1z;

         s2x = s2x + c2x;
         s2y = s2y + c2y;
         s2z = s2z + c2z;

         sxx = sxx + c1x*c2x;
         sxy = sxy + c1x*c2y;
         sxz = sxz + c1x*c2z;

         syx = syx + c1y*c2x;
         syy = syy + c1y*c2y;
         syz = syz + c1y*c2z;

         szx = szx + c1z*c2x;
         szy = szy + c1z*c2y;
         szz = szz + c1z*c2z;
      enddo

      xc(1) = s1x/n;
      xc(2) = s1y/n;
      xc(3) = s1z/n;

      yc(1) = s2x/n;
      yc(2) = s2y/n;
      yc(3) = s2z/n;
      if (mode == 2 .or. mode == 0) then ! need rmsd
         do m = 1, n
            do i = 1, 3
               e0 = e0+ (x(i, m)-xc(i))**2 + (y(i, m)-yc(i))**2
            enddo
         enddo
      endif

      r(1, 1) = sxx - s1x*s2x/n;
      r(2, 1) = sxy - s1x*s2y/n;
      r(3, 1) = sxz - s1x*s2z/n;
      r(1, 2) = syx - s1y*s2x/n;
      r(2, 2) = syy - s1y*s2y/n;
      r(3, 2) = syz - s1y*s2z/n;
      r(1, 3) = szx - s1z*s2x/n;
      r(2, 3) = szy - s1z*s2y/n;
      r(3, 3) = szz - s1z*s2z/n;

      det = r(1,1) * ((r(2,2)*r(3,3)) - (r(2,3)*r(3,2))) &
           - r(1,2) * ((r(2,1)*r(3,3)) - (r(2,3)*r(3,1))) &
           + r(1,3) * ((r(2,1)*r(3,2)) - (r(2,2)*r(3,1)))

      sigma = det

      m = 0
      do j = 1, 3
         do i = 1, j
            m = m+1
            rr(m) = r(1,i)*r(1,j) + r(2,i)*r(2,j) + r(3,i)*r(3,j)
         enddo
      enddo

      spur = (rr(1)+rr(3)+rr(6)) / THREE
      cof = (((((rr(3)*rr(6) - rr(5)*rr(5)) + rr(1)*rr(6)) &
           - rr(4)*rr(4)) + rr(1)*rr(3)) - rr(2)*rr(2)) / THREE
      det = det*det

      do i = 1, 3
         e(i) = spur
      enddo
      if (spur <= ZERO) goto 40
      d = spur*spur
      h = d - cof
      g = (spur*cof - det)/TWO - spur*h
      if (h <= ZERO) then
         if (mode == 0) then
            goto 50
         else
            goto 30
         endif
      endif
      sqrth = sqrt(h)
      d = h*h*h - g*g
      if (d < ZERO) d = ZERO
      d = atan2(sqrt(d), -g) / THREE
      cth = sqrth * cos(d)
      sth = sqrth*sqrt3*sin(d)
      e(1) = (spur + cth) + cth
      e(2) = (spur - cth) + sth
      e(3) = (spur - cth) - sth

      if (mode == 0) then
         goto 50
      endif

      do l = 1, 3, 2
         d = e(l)
         ss(1) = (d-rr(3)) * (d-rr(6))  - rr(5)*rr(5)
         ss(2) = (d-rr(6)) * rr(2)      + rr(4)*rr(5)
         ss(3) = (d-rr(1)) * (d-rr(6))  - rr(4)*rr(4)
         ss(4) = (d-rr(3)) * rr(4)      + rr(2)*rr(5)
         ss(5) = (d-rr(1)) * rr(5)      + rr(2)*rr(4)
         ss(6) = (d-rr(1)) * (d-rr(3))  - rr(2)*rr(2)

         if (abs(ss(1)) >= abs(ss(3))) then
            j = 1
            if (abs(ss(1)) < abs(ss(6))) j = 3
         else if (abs(ss(3)) >= abs(ss(6))) then
            j = 2
         else
            j = 3
         endif

         d = ZERO
         j = 3 * (j - 1)

         do i = 1, 3
            k = ip(i+j)
            a(i,l) = ss(k)
            d = d + ss(k)*ss(k)
         enddo
         if (d > ZERO) d = ONE / sqrt(d)
         do i = 1, 3
            a(i,l) = a(i,l) * d
         enddo
      enddo

      d = a(1,1)*a(1,3) + a(2,1)*a(2,3) + a(3,1)*a(3,3)
      if ((e(1) - e(2)) > (e(2) - e(3))) then
         m1 = 3
         m = 1
      else
         m1 = 1
         m = 3
      endif

      p = ZERO
      do i = 1, 3
         a(i,m1) = a(i,m1) - d*a(i,m)
         p = p + a(i,m1)**2
      enddo
      if (p <= tol) then
         p = ONE
         do i = 1, 3
            if (p < abs(a(i,m))) cycle
            p = abs(a(i,m))
            j = i
         enddo
         k = ip2312(j)
         l = ip2312(j+1)
         p = sqrt(a(k,m)**2 + a(l,m)**2)
         if (p <= tol) goto 40
         a(j,m1) = ZERO
         a(k,m1) = -a(l,m)/p
         a(l,m1) = a(k,m)/p
      else
         p = ONE / sqrt(p)
         do i = 1, 3
            a(i,m1) = a(i,m1)*p
         enddo
      endif

      a(1,2) = a(2,3)*a(3,1) - a(2,1)*a(3,3)
      a(2,2) = a(3,3)*a(1,1) - a(3,1)*a(1,3)
      a(3,2) = a(1,3)*a(2,1) - a(1,1)*a(2,3)

 30   do l = 1, 2
         d = ZERO
         do i = 1, 3
            b(i,l) = r(i,1)*a(1,l) + r(i,2)*a(2,l) + r(i,3)*a(3,l)
            d = d + b(i,l)**2
         enddo
         if (d > ZERO) d = ONE / sqrt(d)
         do i = 1, 3
            b(i,l) = b(i,l)*d
         enddo
      enddo
      d = b(1,1)*b(1,2) + b(2,1)*b(2,2) + b(3,1)*b(3,2)
      p = ZERO

      do i = 1, 3
         b(i,2) = b(i,2) - d*b(i,1)
         p = p + b(i,2)**2
      enddo
      if (p <= tol) then
         p = ONE
         do i = 1, 3
            if (p < abs(b(i,1))) cycle
            p = abs(b(i,1))
            j = i
         enddo
         k = ip2312(j)
         l = ip2312(j+1)
         p = sqrt(b(k,1)**2 + b(l,1)**2)
         if (p <= tol) goto 40
         b(j,2) = ZERO
         b(k,2) = -b(l,1)/p
         b(l,2) = b(k,1)/p
      else
         p = ONE / sqrt(p)
         do i = 1, 3
            b(i,2) = b(i,2)*p
         enddo
      endif

      b(1,3) = b(2,1)*b(3,2) - b(2,2)*b(3,1)
      b(2,3) = b(3,1)*b(1,2) - b(3,2)*b(1,1)
      b(3,3) = b(1,1)*b(2,2) - b(1,2)*b(2,1)

      do i = 1, 3
         do j = 1, 3
            u(i,j) = b(i,1)*a(j,1) + b(i,2)*a(j,2) + b(i,3)*a(j,3)
         enddo
      enddo

 40   do i = 1, 3
         t(i) = ((yc(i) - u(i,1)*xc(1)) - u(i,2)*xc(2)) - u(i,3)*xc(3)
      enddo
 50   do i = 1, 3
         if (e(i) < ZERO) e(i) = ZERO
         e(i) = sqrt(e(i))
      enddo

      ier = 0
      if (e(2) <= (e(1) * 1.0d-05)) ier = -1

      d = e(3)
      if (sigma < ZERO) then
         d = - d
         if ((e(2) - e(3)) <= (e(1) * 1.0d-05)) ier = -1
      endif
      d = (d + e(2)) + e(1)

      if (mode == 2 .or. mode == 0) then ! need rmsd
         rms = (e0 - d) - d
         if (rms < ZERO) rms = ZERO
      endif

      return
   end subroutine u3b

end module tmscore_m

