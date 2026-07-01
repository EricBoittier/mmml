module abpo_collvar
! Module for collective variables
!  cv_base is an abstract derived type with two data components
!  and four deferred type-bound procedures: cv_init, cv_eval, cv_ener,
!  and cv_final. The explicit interface of these procedures are specified
!  below
!
!  To define a new type of collective variable: 
!     * Inherit cv_base to define a new derived type. 
!     * Implement the three required procedures and any other procedure(s) 
!        that may be needed by the specific type. 
!     * Update the selct-case construct of the parse_cv subroutine.
!
!  cv_init should parse the command line and set the correct value
!     for the period data component. (period < 0 indicates non-periodic).
!  cv_eval should calculate the value of the specific cv given X, Y, Z.
!  cv_ener should evaluate the gradient of the specific cv given X, Y, Z,
!     and add the gradient (scaled by a factor s) to DX, DY, DZ. 
!  cv_final should deallocate any allocated storage and reset any counter
!
   use chm_kinds
   use dimens_fcm

   implicit none
#if KEY_ABPO==1
   type, abstract :: cv_base
      real(kind=chm_real) :: val = 0.0
      real(kind=chm_real) :: period = -1.0
   contains
      procedure(cv_init), deferred :: cv_init
      procedure(cv_eval), deferred :: cv_eval
      procedure(cv_ener), deferred :: cv_ener
      procedure(cv_final), deferred :: cv_final
   end type cv_base
   abstract interface
      subroutine cv_init(this, comlyn, comlen)
      ! Parse the command line to initialize the object
         import :: cv_base
         class(cv_base) :: this
         character(len=*) :: comlyn
         integer :: comlen
      end subroutine cv_init
      subroutine cv_eval(this, val, x, y, z)
      ! Calculate the cv value and assign it to val
         use chm_kinds
         import :: cv_base
         class(cv_base) :: this
         real(kind=chm_real) :: val
         real(kind=chm_real) :: x(*), y(*), z(*)
      end subroutine cv_eval
      subroutine cv_ener(this, x, y, z, dx, dy, dz, s)
      ! Add s * (d_cv/dx) to dx
         use chm_kinds
         import :: cv_base
         class(cv_base) :: this
         real(kind=chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
         real(kind=chm_real) :: s
      end subroutine cv_ener
      subroutine cv_final(this)
      ! Deallocate storage
         import :: cv_base
         class(cv_base) :: this
      end subroutine cv_final
   end interface

   type cv_ptr_wrapper
       class(cv_base), pointer :: p
   end type cv_ptr_wrapper

   type, extends(cv_base) :: cv1
      integer :: n_pair
      integer, allocatable, dimension(:, :) :: redilis 
      real(kind=chm_real), allocatable, dimension(:) :: redklis 
   contains
      procedure :: cv_init => cv1_init
      procedure :: cv_eval => cv1_eval
      procedure :: cv_ener => cv1_ener
      procedure :: cv_final => cv1_final
   end type cv1

   type, extends(cv_base) :: cv2
      integer :: n1, n2
      integer, allocatable, dimension(:) :: ilis1, ilis2 
   contains
      procedure :: cv_init => cv2_init
      procedure :: cv_eval => cv2_eval
      procedure :: cv_ener => cv2_ener
      procedure :: cv_final => cv2_final
   end type cv2

   type, extends(cv_base) :: cv3
      integer :: n1, n2, n3
      integer, allocatable, dimension(:) :: ilis1, ilis2, ilis3 
   contains
      procedure :: cv_init => cv3_init
      procedure :: cv_eval => cv3_eval
      procedure :: cv_ener => cv3_ener
      procedure :: cv_final => cv3_final
   end type cv3

   type, extends(cv_base) :: cv4
      integer :: n1, n2, n3, n4
      integer, allocatable, dimension(:) :: ilis1, ilis2, ilis3, ilis4 
   contains
      procedure :: cv_init => cv4_init
      procedure :: cv_eval => cv4_eval
      procedure :: cv_ener => cv4_ener
      procedure :: cv_final => cv4_final
   end type cv4

   contains
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Procedures for generic CVs                     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine parse_cv(comlyn, comlen, cv_array, n_cv)
      ! Parse the command line for cv definitions
      ! Update cv_array with the newly defined cvs 
      ! Update n_cv
         use string
         use stream
         use chutil
         !
         character(len=*) :: comlyn
         integer :: comlen
         type(cv_ptr_wrapper), allocatable, dimension(:) :: cv_array
         integer :: n_cv
         !
         character(len=4) :: cv_type
         class(cv_base), pointer :: ptr
         !
         if (n_cv /= 0) deallocate(cv_array)
         n_cv = 0
         do 
            call trime(comlyn, comlen)
            if (comlen <= 0) exit
            cv_type = nexta4(comlyn, comlen)
            select case (cv_type)
            case ('DIST')
               allocate(cv1 :: ptr)
            case ('DSEL')
               allocate(cv2 :: ptr)
            case ('ASEL')
               allocate(cv3 :: ptr)
            case ('TSEL')
               allocate(cv4 :: ptr)
            case default
               write(outu, *) 'Cannot recognize the CV type ', cv_type
               call wrndie(-1, '<parse_cv>', &
                  'Invalid CV type. Check syntax.')
               exit
            end select
            call ptr%cv_init(comlyn, comlen)
            call add_cv(ptr, cv_array, n_cv)
         end do
         write(outu,*) "parse_cv: A total of ", n_cv, " CVs are defined"
      end subroutine parse_cv

      subroutine add_cv(cv_ptr, cv_array, n_cv)
      ! Add a cv object associated to cv_ptr to cv_array
      !    and increment n_cv
         class(cv_base), pointer :: cv_ptr
         type(cv_ptr_wrapper), allocatable, dimension(:) :: cv_array
         integer :: n_cv

         type(cv_ptr_wrapper), allocatable, dimension(:) :: temp_array
         integer :: i
         if (n_cv == 0) then
            allocate(cv_array(1))
         else
            allocate(temp_array(n_cv))
            do i = 1,n_cv
               temp_array(i) = cv_array(i)
            end do 
            deallocate(cv_array)
            allocate(cv_array(n_cv + 1))
            do i = 1,n_cv
               cv_array(i) = temp_array(i)
            end do
            deallocate(temp_array)
         end if 
         n_cv = n_cv + 1
         cv_array(n_cv)%p => cv_ptr
      end subroutine add_cv

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Procedures for CV1                     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine cv1_init(this, comlyn, comlen)
         ! Initialize a Type 1 CV 
         !
         ! syntax:
         !
         ! DISTance NP number-of-pairs -
         !    repeat( real first-atom second-atom )    
         !
         !  CV = c1*R1 + c2*R2 + ... + cn*Rn
         !
         use number
         use psf
         use select
         use stream
         use string
         use chutil
         use memory
         !
         class(cv1) :: this
         character(len=*) :: comlyn
         integer :: comlen
         ! local variables
         integer, allocatable, dimension(:) :: islct
         integer i, j, n, ni
         character(len=8) sidi, ridi, reni, aci
         character(len=8) sidj, ridj, renj, acj
         !---------------------------------------------------------------
         ! Specify periodicity
         this%period = -1.0
         !---------------------------------------------------------------
         ! Get the number of pairs
         this%n_pair = gtrmi(comlyn, comlen, 'NP', 0)
         if (this%n_pair <= 0) then
            call wrndie(0, '<cv1_init>', &
               'Wrong number of atom pairs specified. Check syntax.')
            call this%cv_final()
            return
         end if
         !---------------------------------------------------------------
         ! Allocate spaces
         call chmalloc('abpo.src', 'cv1_init', 'cv1%redilis', &
            2, this%n_pair, intg=this%redilis)
         call chmalloc('abpo.src', 'cv1_init', 'cv1%redklis', &
            this%n_pair, crl=this%redklis)
         call chmalloc('abpo.src', 'cv1_init', 'islct', &
            natom, intg=islct)
         !---------------------------------------------------------------
         ! Parse loop over n_pair
         do n = 1, this%n_pair 
            if (comlen <= 0) then
               call wrndie(0, '<cv1_init>', &
                  'Wrong number of atom pairs specified. Check syntax.')
               call this%cv_final()
               call chmdealloc('abpo.src', 'cv1_init', 'islct', &
                  natom, intg=islct)
               return
            end if
!c            write(6,888) n,comlyn(1:40)
!c888         format(' cv1_init parsing:::',i5,' comlyn:"',A)
            this%redklis(n)=nextf(comlyn,comlen)
            if(this%redklis(n) == zero) then
               call wrndie(0,'<cv1_init>', &
                  'No distance (or zero) scale factor specified. Check syntax.')
               call this%cv_final()
               call chmdealloc('abpo.src', 'cv1_init', 'islct', &
                  natom, intg=islct)
               return
            end if
            call nxtatm(this%redilis(1,n),ni,2,comlyn,comlen,islct, &
               segid,resid,atype,ibase,nictot,nseg,res,natom)
            if (ni /= 2) then
               call wrndie(0, '<cv1_init>', &
                  'Wrong number of atoms specified. Check syntax.')
               ! deallocation?
               return
            end if
            call trime(comlyn,comlen)
         end do
         !---------------------------------------------------------------
         ! Clean up Temporary space and print information
         call chmdealloc('abpo.src', 'cv1_init', 'islct', &
            natom, intg=islct)
         if (prnlev >= 2) then
            write(outu,510) this%n_pair
510         format(' cv1_init: Adding a DIST cv, Number of atom pairs=',i4)
            do n = 1, this%n_pair
               i = this%redilis(1, n)
               j = this%redilis(2, n)
               call atomid(i, sidi, ridi, reni, aci)
               call atomid(j, sidj, ridj, renj, acj)
               write(outu, 512) &
                  i,sidi(1:idleng),ridi(1:idleng),aci(1:idleng), &
                  j,sidj(1:idleng),ridj(1:idleng),acj(1:idleng), &
                  this%redklis(n)
512            format('      atom pair: ',2(i5,1x,a,1x,a,1x,a), &
               '   distance factor=',f12.4)
            end do
         end if
      end subroutine cv1_init

      subroutine cv1_final(this)
         use memory 
         class(cv1) :: this
         call chmdealloc('abpo.src', 'cv1_final', 'cv1%redilis', &
            2, this%n_pair, intg=this%redilis)
         call chmdealloc('abpo.src', 'cv1_final', 'cv1%redklis', &
            this%n_pair, crl=this%redklis)
         this%n_pair = 0
      end subroutine cv1_final

      subroutine cv1_eval(this, val, x, y, z)
      ! Calc the value of cv1
         use number
         class(cv1) :: this
         real(kind=chm_real) :: val
         real(kind=chm_real) :: x(*), y(*), z(*)
         !
         integer :: n, i, j
         real(kind=chm_real) :: xij, yij, zij, sij, tmp
         !
            tmp = ZERO
            do n = 1, this%n_pair
               i = this%redilis(1, n)
               j = this%redilis(2, n)
               xij = x(i) - x(j)
               yij = y(i) - y(j)
               zij = z(i) - z(j)
               sij = xij*xij + yij*yij + zij*zij
               tmp = tmp + sqrt(sij) * this%redklis(n)
            end do
         !print *, "Evaluating value of CV1", tmp
         this%val = tmp
         val = tmp
      end subroutine cv1_eval

      subroutine cv1_ener(this, x, y, z, dx, dy, dz, s)
      ! Add s * (d_cv1/d_x) to dx 
      !! remove the repeated calculation
         class(cv1) :: this
         real(kind=chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
         real(kind=chm_real) :: s
         !
         integer :: n, i, j
         real(kind=chm_real) :: xij, yij, zij, sij, val
         !
            do n = 1, this%n_pair
               i = this%redilis(1, n)
               j = this%redilis(2, n)
               xij = x(i) - x(j)
               yij = y(i) - y(j)
               zij = z(i) - z(j)
               sij = sqrt(xij*xij + yij*yij + zij*zij)
               sij = s * this%redklis(n) / sij
               dx(i) = dx(i) + sij * xij 
               dx(j) = dx(j) - sij * xij 
               dy(i) = dy(i) + sij * yij 
               dy(j) = dy(j) - sij * yij 
               dz(i) = dz(i) + sij * zij 
               dz(j) = dz(j) - sij * zij 
               !print *, "real foce", sij * xij, sij * yij, sij * zij
            end do
         !print *, "Calculating forces on CV1"
      end subroutine cv1_ener

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Procedures for cv2                     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine cv2_init(this, comlyn, comlen)
         ! Initialize a Type 3 CV 
         !
         ! syntax:
         !
         ! DSEL first-atom-selection second-atom-selection
         !
         !  CV = distance between two centers of mass
         !
         use psf, only: natom
         use select
         use stream
         use memory
         use coord, only: x, y, z, wmain
         !
         class(cv2) :: this
         character(len=*) :: comlyn
         integer :: comlen
         ! local variables
         integer, dimension(natom) :: islct1, islct2
         integer i, j1, j2
         !---------------------------------------------------------------
         ! Specify periodicity
         this%period = -1.0
         !---------------------------------------------------------------
         ! Parse atom selections
         call selcta(comlyn, comlen, islct1, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct2, x, y, z, wmain, .true.)
         if (count(islct1/=islct2, natom) == 0) &
            call wrndie(-1, '<cv2_init>', 'Identical atom selections.')
         this%n1 = count(islct1==1, natom)
         this%n2 = count(islct2==1, natom)
         if (this%n1 == 0 .or. this%n2 == 0) call wrndie(-1, '<cv2_init>', &
            'At least one atom selection is empty.')
         ! Allocate spaces for atom lists
         call chmalloc('abpo.src', 'cv2_init', 'cv2%ilis1', &
            this%n1, intg=this%ilis1)
         call chmalloc('abpo.src', 'cv2_init', 'cv2%ilis2', &
            this%n2, intg=this%ilis2)
         j1 = 1
         j2 = 1
         do i = 1, natom
            if (islct1(i) == 1) then
               this%ilis1(j1) = i
               j1 = j1 + 1
            end if
            if (islct2(i) == 1) then
               this%ilis2(j2) = i
               j2 = j2 + 1
            end if
         end do
         !---------------------------------------------------------------
         if (prnlev >= 2) then
            write(outu,*) ' cv2_init: Adding a DSEL cv'
         end if
      end subroutine cv2_init

      subroutine cv2_final(this)
         use memory 
         !
         class(cv2) :: this
         !
         call chmdealloc('abpo.src', 'cv2_final', 'cv2%ilis1', &
            this%n1, intg=this%ilis1)
         call chmdealloc('abpo.src', 'cv2_final', 'cv2%ilis2', &
            this%n2, intg=this%ilis2)
         this%n1 = 0
         this%n2 = 0
      end subroutine cv2_final

      subroutine cv2_eval(this, val, x, y, z)
      ! Calc the value of cv2
         use number
         use psf, only: amass
         !
         class(cv2) :: this
         real(kind=chm_real) :: val
         real(kind=chm_real) :: x(*), y(*), z(*)
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2, tm
         !
         call com(x1, y1, z1, tm, this%ilis1, this%n1)
         call com(x2, y2, z2, tm, this%ilis2, this%n2)
         val = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
         this%val = val
         !print *, "Evaluating value of cv2", val
      end subroutine cv2_eval

      subroutine cv2_ener(this, x, y, z, dx, dy, dz, s)
      ! Add s * (d_cv2/d_x) to dx 
      !! remove the repeated calculation
         use number
         use psf, only: amass
         !
         class(cv2) :: this
         real(kind=chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
         real(kind=chm_real) :: s
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2, tm1, tm2
         real(kind=chm_real) :: wx, wy, wz
         !
         !!  do we wanna update cv%val when we do cv%ener?
         call com(x1, y1, z1, tm1, this%ilis1, this%n1)
         call com(x2, y2, z2, tm2, this%ilis2, this%n2)
         this%val = sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
         wx = (x1 - x2) / this%val / tm1 * s
         wy = (y1 - y2) / this%val / tm1 * s
         wz = (z1 - z2) / this%val / tm1 * s
         do i = 1, this%n1
            j = this%ilis1(i)
            dx(j) = dx(j) + amass(j) * wx 
            dy(j) = dy(j) + amass(j) * wy 
            dz(j) = dz(j) + amass(j) * wz 
         end do
         wx = (x2 - x1) / this%val / tm2 * s
         wy = (y2 - y1) / this%val / tm2 * s
         wz = (z2 - z1) / this%val / tm2 * s
         do i = 1, this%n2
            j = this%ilis2(i)
            dx(j) = dx(j) + amass(j) * wx 
            dy(j) = dy(j) + amass(j) * wy 
            dz(j) = dz(j) + amass(j) * wz 
         end do
         !print *, "Calculating forces on cv2"
      end subroutine cv2_ener

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Procedures for cv3                     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine cv3_init(this, comlyn, comlen)
         ! Initialize a Type 4 CV 
         !
         ! syntax:
         !
         ! DSEL first-atom-selection second-atom-selection -
         !     third-atom-selection
         !
         !  CV = angle between three centers of mass
         !
         use consta
         use psf, only: natom
         use select
         use stream
         use memory
         use coord, only: x, y, z, wmain
         !
         class(cv3) :: this
         character(len=*) :: comlyn
         integer :: comlen
         ! local variables
         integer, dimension(natom) :: islct1, islct2, islct3
         integer i, j1, j2, j3
         !---------------------------------------------------------------
         ! Specify periodicity
         this%period = -1.0
         !---------------------------------------------------------------
         ! Parse atom selections
         call selcta(comlyn, comlen, islct1, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct2, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct3, x, y, z, wmain, .true.)
         if (count(islct1/=islct2, natom) == 0 .or. &
               count(islct1/=islct3, natom) == 0 .or. &
               count(islct2/=islct3, natom) == 0) &
            call wrndie(-1, '<cv3_init>', 'Identical atom selections.')
         this%n1 = count(islct1==1, natom)
         this%n2 = count(islct2==1, natom)
         this%n3 = count(islct3==1, natom)
         if (this%n1 == 0 .or. this%n2 == 0 .or. this%n3 == 0) &
            call wrndie(-1, '<cv3_init>', &
            'At least one atom selection is empty.')
         ! Allocate spaces for atom lists
         call chmalloc('abpo.src', 'cv3_init', 'cv3%ilis1', &
            this%n1, intg=this%ilis1)
         call chmalloc('abpo.src', 'cv3_init', 'cv3%ilis2', &
            this%n2, intg=this%ilis2)
         call chmalloc('abpo.src', 'cv3_init', 'cv3%ilis3', &
            this%n3, intg=this%ilis3)
         j1 = 1
         j2 = 1
         j3 = 1
         do i = 1, natom
            if (islct1(i) == 1) then
               this%ilis1(j1) = i
               j1 = j1 + 1
            end if
            if (islct2(i) == 1) then
               this%ilis2(j2) = i
               j2 = j2 + 1
            end if
            if (islct3(i) == 1) then
               this%ilis3(j3) = i
               j3 = j3 + 1
            end if
         end do
         !---------------------------------------------------------------
         if (prnlev >= 2) then
            write(outu,*) ' cv3_init: Adding a ASEL cv'
         end if
      end subroutine cv3_init

      subroutine cv3_final(this)
         use memory 
         !
         class(cv3) :: this
         !
         call chmdealloc('abpo.src', 'cv3_final', 'cv3%ilis1', &
            this%n1, intg=this%ilis1)
         call chmdealloc('abpo.src', 'cv3_final', 'cv3%ilis2', &
            this%n2, intg=this%ilis2)
         call chmdealloc('abpo.src', 'cv3_final', 'cv3%ilis3', &
            this%n3, intg=this%ilis3)
         this%n1 = 0
         this%n2 = 0
         this%n3 = 0
      end subroutine cv3_final

      subroutine cv3_eval(this, val, x, y, z)
      ! Calc the value of cv3
         use number
         use psf, only: amass
         !
         class(cv3) :: this
         real(kind=chm_real) :: val
         real(kind=chm_real) :: x(*), y(*), z(*)
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2, x3, y3, z3
         real(kind=chm_real) :: ax, ay, az, bx, by, bz
         real(kind=chm_real) :: ra, rb, ab, cost, tm
         !
         call com(x1, y1, z1, tm, this%ilis1, this%n1)
         call com(x2, y2, z2, tm, this%ilis2, this%n2)
         call com(x3, y3, z3, tm, this%ilis3, this%n3)
         ! f=r1-r2, g=r3-r2.
         ax = x1 - x2
         ay = y1 - y2
         az = z1 - z2
         bx = x3 - x2
         by = y3 - y2
         bz = z3 - z2
         ra = sqrt(ax**2 + ay**2 + az**2)
         rb = sqrt(bx**2 + by**2 + bz**2)
         ab = ax*bx + ay*by + az*bz
         cost = ab / ra / rb
         val = acos(cost)
         this%val = val
         !print *, "Evaluating value of cv3", val
      end subroutine cv3_eval

      subroutine cv3_ener(this, x, y, z, dx, dy, dz, s)
      ! Add s * (d_cv3/d_x) to dx 
      !! remove the repeated calculation
         use number
         use psf, only: amass
         !
         class(cv3) :: this
         real(kind=chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
         real(kind=chm_real) :: s
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2, x3, y3, z3
         real(kind=chm_real) :: tm1, tm2, tm3
         real(kind=chm_real) :: ax, ay, az, bx, by, bz
         real(kind=chm_real) :: ra2, ra, rb2, rb, ab
         real(kind=chm_real) :: gx, gy, gz, rg 
         real(kind=chm_real) :: axgx, axgy, axgz, gxbx, gxby, gxbz
         real(kind=chm_real) :: dtax, dtay, dtaz, dtbx, dtby, dtbz
         real(kind=chm_real) :: wx1, wy1, wz1, wx2, wy2, wz2, wx3, wy3, wz3
         !
         !! do we wanna update cv%val when we do cv%ener?
         call com(x1, y1, z1, tm1, this%ilis1, this%n1)
         call com(x2, y2, z2, tm2, this%ilis2, this%n2)
         call com(x3, y3, z3, tm3, this%ilis3, this%n3)
         ax = x1 - x2
         ay = y1 - y2
         az = z1 - z2
         bx = x3 - x2
         by = y3 - y2
         bz = z3 - z2
         ra2 = ax**2 + ay**2 + az**2
         ra = sqrt(ra2)
         rb2 = bx**2 + by**2 + bz**2
         rb = sqrt(rb2)
         ab = ax*bx + ay*by + az*bz
         ! g=axb/|axb|
         gx = ay*bz - az*by 
         gy = az*bx - ax*bz 
         gz = ax*by - ay*bx 
         rg = sqrt(gx**2 + gy**2 + gz**2)
         ! check rg to avoid zero division
         if (rg < RPRECI) then
            gx = ONE
            gy = ZERO
            gz = ZERO
         else
            gx = gx / rg
            gy = gy / rg
            gz = gz / rg
         end if
         axgx = ay*gz - az*gy 
         axgy = az*gx - ax*gz 
         axgz = ax*gy - ay*gx 
         gxbx = gy*bz - gz*by 
         gxby = gz*bx - gx*bz 
         gxbz = gx*by - gy*bx 
         ! dt/da = axg/a^2
         dtax = axgx / ra2
         dtay = axgy / ra2
         dtaz = axgz / ra2
         ! dt/db = gxb/b^2
         dtbx = gxbx / rb2
         dtby = gxby / rb2
         dtbz = gxbz / rb2
         wx1 = dtax * s / tm1
         wy1 = dtay * s / tm1
         wz1 = dtaz * s / tm1
         wx2 = - (dtax + dtbx) * s / tm2
         wy2 = - (dtay + dtby) * s / tm2
         wz2 = - (dtaz + dtbz) * s / tm2
         wx3 = dtbx * s / tm3
         wy3 = dtby * s / tm3
         wz3 = dtbz * s / tm3
         do i = 1, this%n1
            j = this%ilis1(i)
            dx(j) = dx(j) + amass(j) * wx1 
            dy(j) = dy(j) + amass(j) * wy1
            dz(j) = dz(j) + amass(j) * wz1 
         end do
         do i = 1, this%n2
            j = this%ilis2(i)
            dx(j) = dx(j) + amass(j) * wx2 
            dy(j) = dy(j) + amass(j) * wy2 
            dz(j) = dz(j) + amass(j) * wz2 
         end do
         do i = 1, this%n3
            j = this%ilis3(i)
            dx(j) = dx(j) + amass(j) * wx3 
            dy(j) = dy(j) + amass(j) * wy3 
            dz(j) = dz(j) + amass(j) * wz3 
         end do
         !print *, "Calculating forces on cv3"
      end subroutine cv3_ener

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Procedures for cv4                     
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      subroutine cv4_init(this, comlyn, comlen)
         ! Initialize a Type 5 CV 
         !
         ! syntax:
         !
         ! TSEL first-atom-selection second-atom-selection -
         !     third-atom-selection fourth-atom-selection
         !
         !  CV = torsion angle between four centers of mass
         !
         use number
         use consta
         use psf, only: natom
         use select
         use stream
         use memory
         use coord, only: x, y, z, wmain
         !
         class(cv4) :: this
         character(len=*) :: comlyn
         integer :: comlen
         ! local variables
         integer, dimension(natom) :: islct1, islct2, islct3, islct4
         integer i, j1, j2, j3, j4
         !---------------------------------------------------------------
         ! Specify periodicity
         this%period = PI * 2
         !---------------------------------------------------------------
         ! Parse atom selections
         call selcta(comlyn, comlen, islct1, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct2, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct3, x, y, z, wmain, .true.)
         call selcta(comlyn, comlen, islct4, x, y, z, wmain, .true.)
         if (count(islct1/=islct2, natom) == 0 .or. &
               count(islct1/=islct3, natom) == 0 .or. &
               count(islct1/=islct4, natom) == 0 .or. &
               count(islct2/=islct3, natom) == 0 .or. &
               count(islct2/=islct4, natom) == 0 .or. &
               count(islct3/=islct4, natom) == 0) &
            call wrndie(-1, '<cv4_init>', 'Identical atom selections.')
         this%n1 = count(islct1==1, natom)
         this%n2 = count(islct2==1, natom)
         this%n3 = count(islct3==1, natom)
         this%n4 = count(islct4==1, natom)
         if (this%n1 == 0 .or. this%n2 == 0 .or. this%n3 == 0 .or. &
               this%n4 == 0) &
            call wrndie(-1, '<cv4_init>', &
            'At least one atom selection is empty.')
         ! Allocate spaces for atom lists
         call chmalloc('abpo.src', 'cv4_init', 'cv4%ilis1', &
            this%n1, intg=this%ilis1)
         call chmalloc('abpo.src', 'cv4_init', 'cv4%ilis2', &
            this%n2, intg=this%ilis2)
         call chmalloc('abpo.src', 'cv4_init', 'cv4%ilis3', &
            this%n3, intg=this%ilis3)
         call chmalloc('abpo.src', 'cv4_init', 'cv4%ilis4', &
            this%n4, intg=this%ilis4)
         j1 = 1
         j2 = 1
         j3 = 1
         j4 = 1
         do i = 1, natom
            if (islct1(i) == 1) then
               this%ilis1(j1) = i
               j1 = j1 + 1
            end if
            if (islct2(i) == 1) then
               this%ilis2(j2) = i
               j2 = j2 + 1
            end if
            if (islct3(i) == 1) then
               this%ilis3(j3) = i
               j3 = j3 + 1
            end if
            if (islct4(i) == 1) then
               this%ilis4(j4) = i
               j4 = j4 + 1
            end if
         end do
         !---------------------------------------------------------------
         if (prnlev >= 2) then
            write(outu,*) ' cv4_init: Adding a TSEL cv'
         end if
      end subroutine cv4_init

      subroutine cv4_final(this)
         use memory 
         !
         class(cv4) :: this
         !
         call chmdealloc('abpo.src', 'cv4_final', 'cv4%ilis1', &
            this%n1, intg=this%ilis1)
         call chmdealloc('abpo.src', 'cv4_final', 'cv4%ilis2', &
            this%n2, intg=this%ilis2)
         call chmdealloc('abpo.src', 'cv4_final', 'cv4%ilis3', &
            this%n3, intg=this%ilis3)
         call chmdealloc('abpo.src', 'cv4_final', 'cv4%ilis4', &
            this%n4, intg=this%ilis4)
         this%n1 = 0
         this%n2 = 0
         this%n3 = 0
         this%n4 = 0
      end subroutine cv4_final

      subroutine cv4_eval(this, val, x, y, z)
      ! Calc the value of cv4
         use number
         use psf, only: amass
         !
         class(cv4) :: this
         real(kind=chm_real) :: val
         real(kind=chm_real) :: x(*), y(*), z(*)
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2 
         real(kind=chm_real) :: x3, y3, z3, x4, y4, z4, tm
         real(kind=chm_real) :: fx, fy, fz, gx, gy, gz, hx, hy, hz
         real(kind=chm_real) :: ax, ay, az, bx, by, bz 
         real(kind=chm_real) :: ra2, rb2, rg2, rg 
         real(kind=chm_real) :: ra2r, rb2r, rabr, rgr
         real(kind=chm_real) :: cp, sp
         !
         call com(x1, y1, z1, tm, this%ilis1, this%n1)
         call com(x2, y2, z2, tm, this%ilis2, this%n2)
         call com(x3, y3, z3, tm, this%ilis3, this%n3)
         call com(x4, y4, z4, tm, this%ilis4, this%n4)
         ! f=r1-r2, g=r2-r3, h=r4-rk.
         fx=x1-x2
         fy=y1-y2
         fz=z1-z2
         gx=x2-x3
         gy=y2-y3
         gz=z2-z3
         hx=x4-x3
         hy=y4-y3
         hz=z4-z3
         ! a=f^g, b=h^g
         ax=fy*gz-fz*gy
         ay=fz*gx-fx*gz
         az=fx*gy-fy*gx
         bx=hy*gz-hz*gy
         by=hz*gx-hx*gz
         bz=hx*gy-hy*gx
         ! rg=|g|, rgr=1/|g|
         ra2=ax*ax+ay*ay+az*az
         rb2=bx*bx+by*by+bz*bz
         rg2=gx*gx+gy*gy+gz*gz
         rg=sqrt(rg2)
         rgr=one/rg
         ra2r=one/ra2
         rb2r=one/rb2
         rabr=sqrt(ra2r*rb2r)
         ! cp=cos(phi)
         cp=(ax*bx+ay*by+az*bz)*rabr
         ! sp=sin(phi), note that sin(phi).g/|g|=b^a/(|a|.|b|)
         ! which can be simplified to sin(phi)=|g|h.a/(|a|.|b|)
         sp=rg*rabr*(ax*hx+ay*hy+az*hz)
         if (cp.gt.ptone ) then
            val=asin(sp)
         else
            val=sign(acos(max(cp,minone)),sp)
         end if
         !print *, "Evaluating value of cv4", val
         this%val = val
      end subroutine cv4_eval

      subroutine cv4_ener(this, x, y, z, dx, dy, dz, s)
      ! Add s * (d_cv4/d_x) to dx 
      !! remove the repeated calculation
         use number
         use psf, only: amass
         !
         class(cv4) :: this
         real(kind=chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
         real(kind=chm_real) :: s
         !
         integer :: i, j
         real(kind=chm_real) :: x1, y1, z1, x2, y2, z2
         real(kind=chm_real) :: x3, y3, z3, x4, y4, z4
         real(kind=chm_real) :: tm1, tm2, tm3, tm4
         real(kind=chm_real) :: fx, fy, fz, gx, gy, gz, hx, hy, hz
         real(kind=chm_real) :: ax, ay, az, bx, by, bz 
         real(kind=chm_real) :: ra2, rb2, rg2, rg 
         real(kind=chm_real) :: ra2r, rb2r, rgr
         real(kind=chm_real) :: fg, hg, fga, hgb, gaa, gbb
         real(kind=chm_real) :: dtfx, dtfy, dtfz, dtgx, dtgy, dtgz, dthx, dthy, dthz
         real(kind=chm_real) :: wx1, wy1, wz1, wx2, wy2, wz2
         real(kind=chm_real) :: wx3, wy3, wz3, wx4, wy4, wz4
         !
         !! do we wanna update cv%val when we do cv%ener?
         call com(x1, y1, z1, tm1, this%ilis1, this%n1)
         call com(x2, y2, z2, tm2, this%ilis2, this%n2)
         call com(x3, y3, z3, tm3, this%ilis3, this%n3)
         call com(x4, y4, z4, tm4, this%ilis4, this%n4)
         ! f=ri-rj, g=rj-rk, h=rl-rk.
         fx=x1-x2
         fy=y1-y2
         fz=z1-z2
         gx=x2-x3
         gy=y2-y3
         gz=z2-z3
         hx=x4-x3
         hy=y4-y3
         hz=z4-z3
         ! a=f^g, b=h^g
         ax=fy*gz-fz*gy
         ay=fz*gx-fx*gz
         az=fx*gy-fy*gx
         bx=hy*gz-hz*gy
         by=hz*gx-hx*gz
         bz=hx*gy-hy*gx
         ! rg=|g|, rgr=1/|g|
         ra2=ax*ax+ay*ay+az*az
         rb2=bx*bx+by*by+bz*bz
         rg2=gx*gx+gy*gy+gz*gz
         rg=sqrt(rg2)
         rgr=one/rg
         ra2r=one/ra2
         rb2r=one/rb2
         ! Compute derivatives wrt catesian coordinates.
           ! gaa=-|g|/a^2, gbb=|g|/b^2, fg=f.g, hg=h.g
           ! fga=f.g/(|g|a^2), hgb=h.g/(|g|b^2)
         fg=fx*gx+fy*gy+fz*gz
         hg=hx*gx+hy*gy+hz*gz
         fga=fg*ra2r*rgr
         hgb=hg*rb2r*rgr
         gaa=-ra2r*rg
         gbb=rb2r*rg
         ! dtfi=d(phi)/dfi, dtgi=d(phi)/dgi, dthi=d(phi)/dhi. 
         dtfx=gaa*ax
         dtfy=gaa*ay
         dtfz=gaa*az
         dtgx=fga*ax-hgb*bx
         dtgy=fga*ay-hgb*by
         dtgz=fga*az-hgb*bz
         dthx=gbb*bx
         dthy=gbb*by
         dthz=gbb*bz
         ! dfi=de/dfi, dgi=de/dgi, dhi=de/dhi.
         wx1=s*dtfx/tm1
         wy1=s*dtfy/tm1
         wz1=s*dtfz/tm1
         wx2=s*(-dtfx+dtgx)/tm2
         wy2=s*(-dtfy+dtgy)/tm2
         wz2=s*(-dtfz+dtgz)/tm2
         wx3=s*(-dthx-dtgx)/tm3
         wy3=s*(-dthy-dtgy)/tm3
         wz3=s*(-dthz-dtgz)/tm3
         wx4=s*dthx/tm4
         wy4=s*dthy/tm4
         wz4=s*dthz/tm4
         ! distribute over ri.
         do i = 1, this%n1
            j = this%ilis1(i)
            dx(j) = dx(j) + amass(j) * wx1 
            dy(j) = dy(j) + amass(j) * wy1
            dz(j) = dz(j) + amass(j) * wz1 
         end do
         do i = 1, this%n2
            j = this%ilis2(i)
            dx(j) = dx(j) + amass(j) * wx2 
            dy(j) = dy(j) + amass(j) * wy2 
            dz(j) = dz(j) + amass(j) * wz2 
         end do
         do i = 1, this%n3
            j = this%ilis3(i)
            dx(j) = dx(j) + amass(j) * wx3 
            dy(j) = dy(j) + amass(j) * wy3 
            dz(j) = dz(j) + amass(j) * wz3 
         end do
         do i = 1, this%n4
            j = this%ilis4(i)
            dx(j) = dx(j) + amass(j) * wx4 
            dy(j) = dy(j) + amass(j) * wy4 
            dz(j) = dz(j) + amass(j) * wz4 
         end do
         !print *, "Calculating forces on cv4"
      end subroutine cv4_ener

      subroutine com(comx, comy, comz, tm, ilis, n)
      ! Calc the center of mass of atoms in ilis
         use number
         use psf, only: amass
         use coord, only: x, y, z
         !
         integer :: n
         real(kind=chm_real) :: comx, comy, comz, tm
         integer, dimension(n) :: ilis
         !
         integer :: i, j
         !
         comx = ZERO
         comy = ZERO
         comz = ZERO
         tm = ZERO
         do i = 1, n
            j = ilis(i)
            comx = comx + x(j) * amass(j)
            comy = comy + y(j) * amass(j)
            comz = comz + z(j) * amass(j)
            tm = tm + amass(j)
         end do
         comx = comx / tm
         comy = comy / tm
         comz = comz / tm
      end subroutine com

#endif /* ABPO*/
end module abpo_collvar



