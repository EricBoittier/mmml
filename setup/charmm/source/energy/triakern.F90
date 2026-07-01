module triakern

   use RKHS
   use chm_kinds

   implicit none

   logical, save :: qkern, qfkern
   integer, save :: nkern, nmodl, nckb, nckt
   integer, dimension(:), allocatable, save :: ickb, ickt
   logical, dimension(:), allocatable, save :: lkerna, lkernb
   integer, dimension(:,:), allocatable, save :: kernt
   integer, dimension(:), allocatable, save :: ikern, jkern, kkern
   character(len=100), dimension(:), allocatable, save :: kernk
   character(len=4), dimension(:), allocatable, save :: kernc
   integer, dimension(:), allocatable, save :: kernm
   real(chm_real), dimension(:), allocatable, save :: kecnv, kpcnv
   integer, parameter :: kernt_size = 3
   character(len=4), parameter :: csv=".csv"
   character(len=7), parameter :: kern=".kernel"
   
   type(kernel), dimension(:), allocatable, save :: pes
   logical, dimension(:), allocatable, save :: kread, kload
      
   ! A bit of documentation can be useful.


   ! Variables:
   ! lkerna :: Vector containing the angle term treated by kernels.
   ! lkernb :: Vector containing the bonding term treated by kernels.
   ! qkern, qfkern :: Logical variables for triakern
   ! nkern :: Number of kernels
   ! nmodl :: Number of different kernels models
   ! kernt :: (3, N) type of kernel for each variable.
   ! ikern, jkern, kkern :: Vectors containing the ith,jth and kth atoms used 
   !                        in RKHS.
   ! kernk :: Name of the file containing the kernel
   ! kernc :: Type of coordinate needed for the transformation.
   ! kernm :: Kernel model type
   ! kecnv, kpcnv :: Energy (RKHS -> kcal/mol) and position (Angstrom -> RKHS)
   !                 conversion factors
   ! csv, kern :: Filename extensions
   ! pes :: RKHS object for each kernel model (kernm(nmodl))
   ! kread, kload :: RKHS object initialization flags

   contains

   subroutine tria_kern_init()
   ! This subroutine simply initialize the logical variables needed in CHARMM

      implicit none

      ! Initialize flags
      qkern = .false.
      qfkern = .false.
      
      ! Initialize number of kernel representations¨
      nkern = 0
      nmodl = 0
      nckb = 0
      nckt = 0
      
   end subroutine tria_kern_init

   subroutine tria_kern_set(comlyn,comlen)
   ! This subroutine reads the input command

      use string

      implicit none

      character(len=4) :: wrd, word
      integer :: inuni, idx, ic
      character(len=*), intent(in) :: comlyn
      integer, intent(in) :: comlen
      ! Variables
      ! wrd, word :: Character, defines if we are modifying or adding bonds.
      ! ic :: Integer number corresponding to capital or small letter

      ! Read and decide if it is an new structure or need to be added.
      if(.not.qkern) qkern = .true.

      wrd = nexta4(comlyn, comlen)

      ! Convert the character of word in small letters
      word = wrd
      do idx = 1, 4
         ic = ichar(word(idx:idx))
         if (ic >= 65 .and. ic < 91) word(idx:idx) = char(ic + 32)
      end do
      wrd = word

      select case(wrd)
         case('acti') ! Modifying existing bond/angle
            call tria_kern_def(comlyn,comlen)
            if(.not.qfkern) qfkern = .true.
         case ('clea')
            call tria_kern_clear()
            qkern = .false.
            qfkern = .false.
         case default  ! in case of unrecognized variable
            call wrndie(-3,'<triakern>','Unrecognized option.')
      end select
      
      return

   end subroutine tria_kern_set

   subroutine tria_kern_update()
   ! This subroutine removes the bond and angular terms from CHARMM FF.
   ! For each Kernel, this subroutine remove the ith bend 
   ! interaction from ict and the corresponding bond terms from icb.
   ! The function is called by ... module, as the bond and angle contribution
   ! gets reactivated at the start of new computation (eg. energy or dynamics).
   
      use code
      use psf
      
      implicit none

      integer :: i
      
!       ! Set angle status icbt
!       do i = 1, ntheta
!          if (lkerna(i)) ict(i) = 0
!          ! PER ANGOLI FUNZIONA non so lati
!          ! English, MF, do you speak it! - Jules Winfield, Pulp Fiction (1994)
!       enddo
! 
!       ! Set bond status icbb
!       do i = 1, nbond
!          if (lkernb(i)) icb(i) = 0
!       enddo
      
      do i = 1, nckt
         ict(ickt(i)) = 0
      enddo
      do i = 1, nckb
         icb(ickb(i)) = 0
      enddo
      
      return

   end subroutine tria_kern_update

   subroutine tria_kern_def(comlyn,comlen)
   ! Definition of the triatomic kernels
   
      use dimens_fcm
      use code
      use number
      use param
      use psf
      use string
      use memory
      use coord
      use select
      use modpsf ! CMPRIC(MARK,LSORT) to delete marked bonds
      use stream, only: outu, prnlev
#if KEY_PARALLEL==1
      use parallel, only: comm_charmm, mynod
      use mpi, only: mpi_integer, mpi_real8, mpi_logical, mpi_character
#endif
 
      implicit none

      character(len=*), intent(in) :: comlyn
      integer, intent(in) :: comlen
      integer,parameter :: mark = -99999999
      character(len=4) :: wrd
      character(len=100) :: word
      
      logical :: flag
      integer :: i, j, k, ii, jj, kk, nlen, idx, ic, ierror
      integer :: nkrn, max_kern, new_kern
      integer :: tmp_nckt, tmp_nckb, new_nckt, new_nckb
      
      character(len=100), dimension(:), allocatable :: tmp_kernel
      character(len=4), dimension(:), allocatable :: tmp_coor
      integer, dimension(:), allocatable :: tmp_it, tmp_jt, tmp_kt
      integer, dimension(:), allocatable :: itemp, jtemp, ktemp
      integer, dimension(:), allocatable :: tmp_modl
      integer, dimension(:,:), allocatable :: tmp_type
      real(chm_real), dimension(:), allocatable :: tmp_ecnv, tmp_pcnv
      integer, dimension(:), allocatable :: tmp_ickt, tmp_ickb

      ! Variables
      ! comlyn and comlen :: Variables used in CHARMM for reading arguments.
      ! mark :: Constant number used for removing bonds.
      ! wrd and word :: Character variables needed for storing information.
      ! itemp, jtemp, ktemp :: Temporary vectors storing the atom number 
      !                        treated by kernels.
      ! max_kern :: Max number of kernels
      ! nkern :: number of kernels
      ! tmp_it, tmp_jt, tmp_kt :: Temporary vectors storing the the atom 
      !                           number. Needed for interaction table.
      ! tmp_type :: Temporary matrix (3,nkern) stores the Kernel function used 
      !             for each variable.
      ! tmp_kernel :: Temporary vector with name of the external file.
      ! tmp_coor :: Temporary vector storing information over the coordinate 
      !             set of the kernel.
      ! tmp_modl :: Temporary vector storing kernel model type.
      ! new_kern :: Number of new kernel sites

      ! Initialization of the logical vectors lkerna and lkernb.
      if (.not.qfkern) then
         call chmalloc('triakern.src','EkernDEF','LKERNA',ntheta,log=lkerna) 
         call chmalloc('triakern.src','EkernDEF','LKERNB',nbond,log=lkernb)
         lkerna = .false.
         lkernb = .false.
      endif

      max_kern = nkern + ntheta

      ! Allocation/deallocation/reallocation of the variables (described below)
      call chmalloc('triakern.src','EkernDEF','ITEMP',natom,intg=itemp)
      call chmalloc('triakern.src','EkernDEF','JTEMP',natom,intg=jtemp)
      call chmalloc('triakern.src','EkernDEF','KTEMP',natom,intg=ktemp)
      call chmalloc('triakern.src','EkernDEF','TMP_IT',max_kern,intg=tmp_it)
      call chmalloc('triakern.src','EkernDEF','TMP_JT',max_kern,intg=tmp_jt)
      call chmalloc('triakern.src','EkernDEF','TMP_KT',max_kern,intg=tmp_kt)
      call chmalloc('triakern.src','EkernDEF','TMP_TYPE',&
                    kernt_size,max_kern,intg=tmp_type)
      call chmalloc('triakern.src','EkernDEF','TMP_COOR',&
                    max_kern,ch4=tmp_coor)
      call chmalloc('triakern.src','EkernDEF','TMP_MODL',&
                    max_kern,intg=tmp_modl)
      call chmalloc('triakern.src','EkernDEF','TMP_ECNV',&
                    max_kern,crl=tmp_ecnv)
      call chmalloc('triakern.src','EkernDEF','TMP_PCNV',&
                    max_kern,crl=tmp_pcnv)
      call chmalloc('triakern.src','EkernDEF','TMP_ICKT',&
                    ntheta,intg=tmp_ickt)
      call chmalloc('triakern.src','EkernDEF','TMP_ICKB',&
                    nbond,intg=tmp_ickb)
      allocate (tmp_kernel(max_kern)) ! CHARMM does not handle this allocation

      ! Initialize temporary variables
      tmp_it = 0
      tmp_jt = 0
      tmp_kt = 0
      tmp_kernel = '             '
      tmp_type = 0
      tmp_coor = '    '
      tmp_modl = 0
      tmp_ecnv = 1.0
      tmp_pcnv = 1.0
      tmp_ickt = 0
      tmp_ickb = 0

      ! Assign values to temporary variables
      if (qfkern) then
         if (nkern>0) then
            tmp_it(1:nkern) = ikern(1:nkern)  ! atom 1
            tmp_jt(1:nkern) = jkern(1:nkern)  ! atom 2
            tmp_kt(1:nkern) = kkern(1:nkern)  ! atom 3
            tmp_kernel(1:nkern) = kernk(1:nkern)
            tmp_type(:,1:nkern) = kernt(:,1:nkern)
            tmp_coor(1:nkern) = kernc(1:nkern)
            tmp_modl(1:nkern) = kernm(1:nkern)
            tmp_ecnv(1:nkern) = kecnv(1:nkern)
            tmp_pcnv(1:nkern) = kpcnv(1:nkern)
            tmp_ickt(1:nckt) = ickt(1:nckt)
            tmp_ickb(1:nckb) = ickb(1:nckb)
         end if
      end if

      ! Temporary storage
      tmp_kernel(nkern+1) = next100(comlyn,comlen)
      tmp_type(1,nkern+1) = nexti(comlyn,comlen)
      tmp_type(2,nkern+1) = nexti(comlyn,comlen)
      tmp_type(3,nkern+1) = nexti(comlyn,comlen)
      tmp_coor(nkern+1) = nexta4(comlyn,comlen)
      
      ! Switch to lower cases
      word = tmp_kernel(nkern+1)
      nlen = len(word)
      do idx = 1, nlen
         ic = ichar(word(idx:idx))
         if (ic >= 65 .and. ic < 91) word(idx:idx) = char(ic + 32)
      end do
      tmp_kernel(nkern+1) = trim(word)
      
      word = tmp_coor(nkern+1)
      do idx = 1, 4
         ic = ichar(word(idx:idx))
         if (ic >= 65 .and. ic < 91) word(idx:idx) = char(ic + 32)
      end do
      tmp_coor(nkern+1) = trim(word)

      ! Select the three atoms
      call selcta(comlyn,comlen,itemp,x,y,z,wmain,.true.)
      call selcta(comlyn,comlen,jtemp,x,y,z,wmain,.true.)
      call selcta(comlyn,comlen,ktemp,x,y,z,wmain,.true.)

      ! Look for unit conversion
      flag = .true.
      do while (flag)
         wrd = nexta4(comlyn, comlen)
         word = wrd
         ! Switch to lower cases
         do idx = 1, 4
            ic = ichar(word(idx:idx))
            if (ic >= 65 .and. ic < 91) word(idx:idx) = char(ic + 32)
         end do
         wrd = word
         select case(wrd)
            case('hart') ! Energy conversion factor RKHS(Hartree) -> kcal/mol
               tmp_ecnv(nkern+1) = 627.5094738898777
            case('kjmo') ! Energy conversion factor RKHS(kJ/mol) -> kcal/mol
               tmp_ecnv(nkern+1) = 0.2390057361376673
            case('evol') ! Energy conversion factor RKHS(eV) -> kcal/mol
               tmp_ecnv(nkern+1) = 23.060548012069496
            case('ecnv') ! Energy conversion factor RKHS -> kcal/mol
               tmp_ecnv(nkern+1) = nextf(comlyn,comlen)
            case('bohr') ! Position conversion factor Angstrom -> Bohr
               tmp_pcnv(nkern+1) = 1.8897261258369282
            case('pcnv') ! Position conversion factor Angstrom -> RKHS
               tmp_pcnv(nkern+1) = nextf(comlyn,comlen)
            case ('')
               exit
            case default
               ! Skip unrecognized options
               word = next100(comlyn, comlen)
         end select
      end do

      ! Check kernel model type.
      ! Kernel model are considered the same, if the name of the external file,
      ! the kernel coordinate type and the kernel function types match.
      ! If a match is found, flag becomes true and kernel model type was 
      ! assigned in form of a running index, else a new kernel model type index
      ! is assigned.
      flag = .false.
      do i = 1, nkern
         if (tmp_kernel(nkern+1) == tmp_kernel(i)) then
            if (tmp_coor(nkern+1) == tmp_coor(i)) then
               if (tmp_type(1,nkern+1) == tmp_type(1,i) .and. &
                   tmp_type(2,nkern+1) == tmp_type(2,i) .and. &
                   tmp_type(3,nkern+1) == tmp_type(3,i)) then
                  ! Matching kernel model type is found
                  flag = .true.
                  tmp_modl(nkern+1) = tmp_modl(i)
               end if
            end if
         end if
         if (flag) exit
      end do
      ! No matching kernel model type found
      if (.not. flag) then
         if (nkern>0) then
            tmp_modl(nkern+1) = maxval(tmp_modl(:nkern)) + 1
         else
            tmp_modl(1) = 1
         end if
      end if
      
      ! New kernel counter
      nkrn = nkern
      
      ! Temporary found angle and bond counters
      tmp_nckt = nckt
      tmp_nckb = nckb
      
      ! Detect angle
      do k = 1, ntheta
         ii = it(k)
         jj = jt(k)
         kk = kt(k)
         if ( (ii>0) .and. (jj>0) .and. (kk>0) ) then
            ! Center atom 2
            if ( (jtemp(jj)==1) ) then
               ! Angle 1-2-3
               if ( (itemp(ii)==1) .and. (ktemp(kk)==1) )&
                  then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = ii
                  tmp_jt(nkrn) = jj
                  tmp_kt(nkrn) = kk
               ! Angle 3-2-1
               elseif( (itemp(kk)==1) .and. (ktemp(ii)==1) ) then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = kk
                  tmp_jt(nkrn) = jj
                  tmp_kt(nkrn) = ii
               endif
            ! Center atom 1
            elseif ( (itemp(jj)==1) ) then
               ! Angle 2-1-3
               if( (jtemp(ii)==1) .and. (ktemp(kk)==1) ) then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = jj
                  tmp_jt(nkrn) = ii
                  tmp_kt(nkrn) = kk
               ! Angle 3-1-2
               elseif( (jtemp(kk)==1) .and. (ktemp(ii)==1) ) then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = jj
                  tmp_jt(nkrn) = kk
                  tmp_kt(nkrn) = ii
               endif
            ! Center atom 3
            elseif ( (ktemp(jj)==1) ) then
               ! Angle 1-3-2
               if( (itemp(ii)==1) .and. (jtemp(kk)==1) ) then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = ii
                  tmp_jt(nkrn) = kk
                  tmp_kt(nkrn) = jj
               ! Angle 2-3-1
               elseif( (itemp(kk)==1) .and. (jtemp(ii)==1) ) then
                  nkrn = nkrn + 1
                  tmp_nckt = tmp_nckt + 1
                  tmp_ickt(tmp_nckt) = k
                  lkerna(k) = .true.
                  tmp_it(nkrn) = kk
                  tmp_jt(nkrn) = ii
                  tmp_kt(nkrn) = jj
               endif
            endif
         endif
      enddo
      
      ! Detect bonds
      do k = 1, nbond
         ii = ib(k)
         jj = jb(k)
         if( (ii>0) .and. (jj>0)) then
            ! Check kernel atoms (old kernel number (nkern) to new one (nkrn) 
            ! without regarding redundancy)
            flag = .false.
            do kk = nkern+1, nkrn
               ! Bond 1-2
               if( ( (tmp_it(kk)==ii) .and. (tmp_jt(kk)==jj) ) .or. &
                   ( (tmp_it(kk)==jj) .and. (tmp_jt(kk)==ii) ) ) then
                  flag = .true.
                  lkernb(k) = .true.
               ! Bond 2-3
               elseif( ( (tmp_jt(kk)==ii) .and. (tmp_kt(kk)==jj) ) .or. &
                   ( (tmp_jt(kk)==jj) .and. (tmp_kt(kk)==ii) ) ) then
                  flag = .true.
                  lkernb(k) = .true.
               ! Bond 3-1
               elseif( ( (tmp_kt(kk)==ii) .and. (tmp_it(kk)==jj) ) .or. &
                   ( (tmp_kt(kk)==jj) .and. (tmp_it(kk)==ii) ) ) then
                  flag = .true.
                  lkernb(k) = .true.
               endif
            enddo
            if (flag) then
               tmp_nckb = tmp_nckb + 1
               tmp_ickb(tmp_nckb) = k
            endif
         endif
      enddo
      
      ! Deactivate angle and bond potential contributions
      call tria_kern_update()
      
      ! Mark redundant definitions
      new_kern = nkrn
      ! Iterate over all kernels
      do i = 1, nkrn
         ii = tmp_it(i)
         jj = tmp_jt(i)
         kk = tmp_kt(i)
         ! Iterate over new kernels in reverse (to avoid marking older kernels
         ! as redundant)
         do j = nkrn, i+1, -1
            ! Mark redundant, if atom indices matches
            if (ii==tmp_it(j)) then
               if ( (jj==tmp_jt(j)) .and. (kk==tmp_kt(j)) ) then
                  if(tmp_it(j).ne.mark) new_kern = new_kern - 1
                  tmp_it(j) = mark
               endif
            endif
         enddo
      enddo
      
      ! Mark redundant angles and bonds
      new_nckt = tmp_nckt
      do i = 1, tmp_nckt
         ii = tmp_ickt(i)
         do j = tmp_nckt, i+1, -1
            if (ii==tmp_ickt(j)) then
               if(tmp_ickt(j).ne.mark) new_nckt = new_nckt - 1
               tmp_ickt(j) = mark
            endif
         enddo
      enddo
      
      new_nckb = tmp_nckb
      do i = 1, tmp_nckb
         ii = tmp_ickb(i)
         do j = tmp_nckb, i+1, -1
            if (ii==tmp_ickb(j)) then
               if(tmp_ickb(j).ne.mark) new_nckb = new_nckb - 1
               tmp_ickb(j) = mark
            endif
         enddo
      enddo
      
      ! Deallocate previous module arrays
      if (qfkern)then
         call chmdealloc('triakern.src','EKERNDEF','KERNC',&
                         nkern,ch4=kernc)   ! tmp_coor
         call chmdealloc('triakern.src','EKERNDEF','KERNT',&
                         kernt_size,nkern,intg=kernt) ! tmp_type
         call chmdealloc('triakern.src','EKERNDEF','KERNM',&
                         nkern,intg=kernm)   ! tmp_modl
         call chmdealloc('triakern.src','EKERNDEF','IKERN',&
                         nkern,intg=ikern)  ! tmp_it
         call chmdealloc('triakern.src','EKERNDEF','JKERN',&
                         nkern,intg=jkern)  ! tmp_jt
         call chmdealloc('triakern.src','EKERNDEF','KKERN',&
                         nkern,intg=kkern)  ! tmp_kt
         call chmdealloc('triakern.src','EKERNDEF','KECNV',&
                         nkern,crl=kecnv)  ! tmp_ecnv
         call chmdealloc('triakern.src','EKERNDEF','KPCNV',&
                         nkern,crl=kpcnv)  ! tmp_pcnv
         call chmdealloc('triakern.src','EKERNDEF','ICKT',&
                         nckt,intg=ickt)   ! tmp_ickt
         call chmdealloc('triakern.src','EKERNDEF','ICKB',&
                         nckb,intg=ickb)   ! tmp_ickb
         deallocate (kernk) ! tmp_kernel
         
         deallocate (pes)
         deallocate (kread, kload)
         
      endif

      ! (Re)allocate model arrays
      call chmalloc('triakern.src','EKERNDEF','KERNC',&
                    new_kern,ch4=kernc)   ! tmp_coor
      call chmalloc('triakern.src','EKERNDEF','KERNT',&
                    kernt_size,new_kern,intg=kernt)  ! tmp_type
      call chmalloc('triakern.src','EKERNDEF','KERNM',&
                    new_kern,intg=kernm)   ! tmp_modl
      call chmalloc('triakern.src','EKERNDEF','IKERN',&
                    new_kern,intg=ikern)
      call chmalloc('triakern.src','EkernDEF','JKERN',&
                    new_kern,intg=jkern)
      call chmalloc('triakern.src','EkernDEF','KKERN',&
                    new_kern,intg=kkern)
      call chmalloc('triakern.src','EKERNDEF','KECNV',&
                    new_kern,crl=kecnv)
      call chmalloc('triakern.src','EKERNDEF','KPCNV',&
                    new_kern,crl=kpcnv)
      call chmalloc('triakern.src','EKERNDEF','ICKT',&
                    new_nckt,intg=ickt)
      call chmalloc('triakern.src','EKERNDEF','ICKB',&
                    new_nckb,intg=ickb)      
      allocate (kernk(new_kern))
      
      
      ! Reassign kernels:
      ! Previous kernels
      ikern(:nkern) = tmp_it(:nkern)
      jkern(:nkern) = tmp_jt(:nkern)
      kkern(:nkern) = tmp_kt(:nkern)
      kernc(:nkern) = tmp_coor(:nkern)
      kernt(:,:nkern) = tmp_type(:,:nkern)
      kernk(:nkern) = tmp_kernel(:nkern)
      kernm(:nkern) = tmp_modl(:nkern)
      kecnv(:nkern) = tmp_ecnv(:nkern)
      kpcnv(:nkern) = tmp_pcnv(:nkern)
      ickt(:nckt) = tmp_ickt(:nckt)
      ickb(:nckb) = tmp_ickb(:nckb)
      ! New kernels
      j = nkern
      do i = nkern+1, nkrn
         if ((tmp_it(i)==mark)) cycle
         j = j + 1
         if (j>new_kern) then
            call wrndie(-3,'<triakern>','Kernel assignment went wrong.')
         endif
         ikern(j) = tmp_it(i)
         jkern(j) = tmp_jt(i)
         kkern(j) = tmp_kt(i)
         kernc(j) = tmp_coor(nkern+1)
         kernt(:,j) = tmp_type(:,nkern+1)
         kernk(j) = tmp_kernel(nkern+1)
         kernm(j) = tmp_modl(nkern+1)
         kecnv(j) = tmp_ecnv(nkern+1)
         kpcnv(j) = tmp_pcnv(nkern+1)
      enddo
      j = nckt
      do i = nckt+1, tmp_nckt
         if ((tmp_ickt(i)==mark)) cycle
         j = j + 1
         if (j > new_nckt) then
            call wrndie(-3,'<triakern>','Angle assignment went wrong.')
         endif
         ickt(j) = tmp_ickt(i)
      enddo
      j = nckb
      do i = nckb+1, tmp_nckb
         if ((tmp_ickb(i)==mark)) cycle
         j = j + 1
         if (j > new_nckb) then
            call wrndie(-3,'<triakern>','Bond assignment went wrong.')
         endif
         ickb(j) = tmp_ickb(i)
      enddo
      
      ! Number of different kernel models
      nmodl = maxval(kernm)
      
      ! Number of affected angle and bonds
      nckt = new_nckt
      nckb = new_nckb
      
      ! Allocate RKHS object array and flags
      allocate (pes(nmodl))
      allocate (kread(nmodl), kload(nmodl))
      kread = .false.
      kload = .false.

      ! Deallocate temporary variables
      call chmdealloc('triakern.src','EkernDEF','ITEMP',natom,intg=itemp)
      call chmdealloc('triakern.src','EkernDEF','JTEMP',natom,intg=jtemp)
      call chmdealloc('triakern.src','EkernDEF','KTEMP',nkern,intg=ktemp)
      call chmdealloc('triakern.src','EkernDEF','TMP_IT',nkern,intg=tmp_it)
      call chmdealloc('triakern.src','EkernDEF','TMP_JT',nkern,intg=tmp_jt)
      call chmdealloc('triakern.src','EkernDEF','TMP_KT',nkern,intg=tmp_kt)
      call chmdealloc('triakern.src','EkernDEF','TMP_COOR',nkern,ch4=tmp_coor)
      call chmdealloc('triakern.src','EkernDEF','TMP_TYPE',&
                      kernt_size,nkern,intg=tmp_type)
      call chmdealloc('triakern.src','EkernDEF','TMP_MODL',nkern,intg=tmp_modl)
      call chmdealloc('triakern.src','EkernDEF','TMP_ECNV',nkern,crl=tmp_ecnv)
      call chmdealloc('triakern.src','EkernDEF','TMP_PCNV',nkern,crl=tmp_pcnv)
      call chmdealloc('triakern.src','EkernDEF','TMP_ICKT',ntheta,intg=tmp_ickt)
      call chmdealloc('triakern.src','EkernDEF','TMP_ICKB',nbond,intg=tmp_ickb)
      deallocate(tmp_kernel)
      
      ! Set global parameter
      nkern = new_kern

#if KEY_PARALLEL==1
      call mpi_barrier(comm_charmm, ierror)
      ! Broadcast data to other nodes
      call mpi_bcast(kernc, 4, mpi_character, 0, comm_charmm, ierror)
      call mpi_bcast(kernt, nkern, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(kernm, nkern, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(ikern, nkern, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(jkern, nkern, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(kkern, nkern, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(kecnv, nkern, mpi_real8, 0, comm_charmm, ierror)
      call mpi_bcast(kpcnv, nkern, mpi_real8, 0, comm_charmm, ierror)
      call mpi_bcast(ickt, nckt, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(ickb, nckb, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(kernk, 100, mpi_character, 0, comm_charmm, ierror)
      call mpi_bcast(nmodl, 1, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(nckb, 1, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(nckt, 1, mpi_integer, 0, comm_charmm, ierror)
      call mpi_bcast(lkerna, ntheta, mpi_logical, 0, comm_charmm, ierror)
      call mpi_bcast(lkernb, nbond, mpi_logical, 0, comm_charmm, ierror)
#endif
      
      ! Print setup information
#if KEY_PARALLEL==1
     if(mynod.eq.0) then ! if in node 0
#endif
         write(outu,'(A,I4)') ' TRIAKERN> Number of kernels :: ', nkern
         write(outu,'(A,I4)') ' TRIAKERN> Number of different kernel models :: ',& 
            nmodl
         if(prnlev.ge.5) then
            do idx = 1, nkern
               write(outu,'(A,I4)') ' TRIAKERN> Kernel Model :', idx
               write(outu,'(5A,3I4)') ' TRIAKERN> filename, model and types: ',&
                  trim(kernk(idx)), ", ", kernc(idx), ", ", kernt(:,idx)
               write(outu,'(A,3I4)') ' TRIAKERN>   atom indices: ',&
                  ikern(idx), jkern(idx), kkern(idx)
            enddo
         endif
#if KEY_PARALLEL==1
     endif
#endif

      return

   end subroutine tria_kern_def

   subroutine tria_kern_clear()
   ! Deallocation of the vectors used in this module

      use memory
      use psf

      implicit none
      !!! VEDI COSA METTERE COME DEALLOCAZIONE
      
      if (qfkern) then
         call chmdealloc('triakern.src','EKERNDEF','LKERNA',ntheta,log=lkerna)
         call chmdealloc('triakern.src','EKERNDEF','LKERNB',nbond,log=lkernb)
         call chmdealloc('triakern.src','EKERNDEF','KERNC',nkern,ch4=kernc)
         call chmdealloc('triakern.src','EKERNDEF','KERNT',&
                         kernt_size,nkern,intg=kernt)
         call chmdealloc('triakern.src','EKERNDEF','KERNM',nkern,intg=kernm)
         call chmdealloc('triakern.src','EKERNDEF','IKERN',nkern,intg=ikern)
         call chmdealloc('triakern.src','EKERNDEF','JKERN',nkern,intg=jkern)
         call chmdealloc('triakern.src','EKERNDEF','KKERN',nkern,intg=kkern)
         call chmdealloc('triakern.src','EKERNDEF','KECNV',nkern,crl=kecnv)
         call chmdealloc('triakern.src','EKERNDEF','KPCNV',nkern,crl=kpcnv)
         call chmdealloc('triakern.src','EKERNDEF','ICKT',nkern,intg=ickt)
         call chmdealloc('triakern.src','EKERNDEF','ICKB',nkern,intg=ickb)

         deallocate (kernk)
         deallocate (pes)
         deallocate (kread, kload)
         nkern = 0
         nmodl = 0
         nckt = 0
         nckb = 0
      endif
      
      return

   end subroutine tria_kern_clear

   subroutine tria_kern_ener(ebond,x,y,z,dx,dy,dz)
   ! Energy and forces evaluation.

      use code
      use number
#if KEY_PARALLEL==1
      use parallel
#endif
      use psf
      use string
      
      implicit none

      real(chm_real) :: ebond
      real(chm_real) :: x(*), y(*), z(*), dx(*), dy(*), dz(*)
      
      integer :: i, j, k, m, ii, jj
      real(chm_real) :: erkhs, ekern
      real(chm_real), dimension(3) :: dkern, mass
      real(chm_real), dimension(3) :: rtrkn, dedtrkn
      real(chm_real), dimension(3) :: rrkhs, dedrkhs
      real(chm_real), dimension(3) :: rintr, dedintr
      real(chm_real), dimension(3,3) :: rcart, dedcart
      real(chm_real), dimension(3,3) :: dtrkndintr
      real(chm_real), dimension(3,3,3) :: dtrkndcart
      ! Variables
      ! ebond     :: CHARMM bond energy
      ! x,y,z     :: Atomic coordinates
      ! dx,dy,dz  :: Atomic potential Derivatives
      ! ekern     :: Energy contribution of the Kernels
      ! dkern     :: Kernel energy derivatives
      ! mass      :: Atomic masses of the triatomic.
      ! rcart     :: Cartesian coordinates of the three atoms
      ! rintr     :: Internal distance coordinates between the three atoms
      ! rtrkn     :: Reduced coordinates of the three atoms
      !              (e.g. Jacobi, internal, Radau)
      ! rrkhs     :: Ordered and manipulated reduced coordinates as supposed
      !              by the RKHS model (e.g, angle theta to z(theta)).
      ! dedcart   :: Energy derivative for system atoms along Cartesian
      !              coordinates rcart
      ! dedintr   :: Energy derivative for system atoms along internal distance
      !              coordinates rintr
      ! dedtrkn   :: Energy derivative for system atoms along reduced
      !              coordinates rtrkn
      ! dedrkhs   :: Energy derivative for system atoms along ordered and 
      !              manipulated reduced coordinates rrkhs
      ! dtrkndintr:: Transformation matrix for derivatives with respect to
      !              reduced coordinates to internal coordinates
      ! dtrkndcart:: Transformation matrix for derivatives with respect to
      !              reduced coordinates to Cartesian coordinates
      
      ! Initialize variables
      ekern = 0.d0 ; dkern = 0.d0 ; mass = 0.d0

#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
      do m = mynodp, nkern, numnod
#else
#error 'Illegal parallel compile option'
#endif
#else
      do m = 1 , nkern ! loop over the number of bonds
#endif
         ! Assign atoms from charmm to the module
         i=ikern(m)
         j=jkern(m)
         k=kkern(m)
!          write(outu,*) mynod, kernc(m), m, nkern
         ! Based on the topology of kernel calculate energy and derivatives
         select case(kernc(m))

            case("rddz")
               ! Radau coordinates ordered by bond distance d1(1-2), d2(2-3)
               ! and angle theta(1-2-3) in z (z = (1 - cos(theta))/2)
               !        1       !
               !       / \      !
               !      /   \     !
               !   d1/  th \d2  !
               !    /       \   !
               !   /         \  !
               !  2           3 !
               ! Order: d1, d2, z
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)
               
               ! Compute Cartesian to Radau with derivative conversion factors
               ! Radau: 1 - d1 ; 2 - d2 ; 3 - theta
               call cartesian_to_radau(rcart, rtrkn, dtrkndcart)
               
               ! Order and transform reduced coordinates
               ! No change in order
               rrkhs = rtrkn(:)
               ! Convert theta to z(theta)
               rrkhs(3) = (1.0d0 - cos(rtrkn(3)))/2.0d0
               
               ! Compute potential and derivatives in Radau
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)
               
               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn = dedrkhs(:)
               ! Convert de/dz to de/dtheta
               dedtrkn(3) = dedrkhs(3)*sin(rtrkn(3))/2.0d0
               
               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedtrkn(1)*dtrkndcart(1,jj,ii)    &
                                      + dedtrkn(2)*dtrkndcart(2,jj,ii)  &
                                      + dedtrkn(3)*dtrkndcart(3,jj,ii)
                  end do
               end do

               ! Numerical derivation
!                call get_potential_radau_numerical(   &
!                   kernk(m), kernt(:,m), kernm(m),    &
!                   rcart, mass, erkhs, dtrkndintr)
                  
!                if (sqrt(sum((dtrkndintr - dedcart)**2))/9.0d0 .ge. 1.0d-8) then
!                   write(*,*) 'Kernel ', m
!                   write(*,*) 'Cartesian ', rcart
!                   write(*,*) 'Radau ', rtrkn
!                   write(*,*) 'Analytical', dedcart
!                   write(*,*) 'Numerical ', dtrkndintr
!                   write(*,*) 'Deviation ',            &
!                      sqrt(sum((dtrkndintr - dedcart)**2))/9.0d0
!                end if

               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)
               
               ! Use numerical derivative 
!                dedcart = dtrkndintr*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)

            case("rzdd", "int1")
               ! Radau coordinates ordered by
               ! angle theta(1-2-3) in z conversion (z = (1 - cos(theta))/2),
               ! bond distance d1(1-2) and d2(2-3).
               !        1       !
               !       / \      !
               !      /   \     !
               !   d1/  th \d2  !
               !    /       \   !
               !   /         \  !
               !  2           3 !
               ! Order: z, d1, d2
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to Radau with derivative conversion factors
               ! Radau: 1 - d1 ; 2 - d2 ; 3 - theta
               call cartesian_to_radau(rcart, rtrkn, dtrkndcart)
               
               ! Order and transform reduced coordinates
               ! Switch d1 with theta and convert theta to z(theta)
               rrkhs(1) = (1.0d0 - cos(rtrkn(3)))/2.0d0
               rrkhs(2) = rtrkn(2)
               rrkhs(3) = rtrkn(1)
               
               ! Compute potential and derivatives in Radau
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)
               
               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn(1) = dedrkhs(3)
               dedtrkn(2) = dedrkhs(2)
               dedtrkn(3) = dedrkhs(1)*sin(rtrkn(3))/2.0d0
               
               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedtrkn(1)*dtrkndcart(1,jj,ii)    &
                                      + dedtrkn(2)*dtrkndcart(2,jj,ii)  &
                                      + dedtrkn(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)

            case("rddt")
               ! Radau coordinates ordered by bond distance d1(1-2), d2(2-3)
               ! and angle theta(1-2-3) in radians(!)
               !        1       !
               !       / \      !
               !      /   \     !
               !   d1/  th \d2  !
               !    /       \   !
               !   /         \  !
               !  2           3 !
               ! Order: d1, d2, theta
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)
               
               ! Compute Cartesian to Radau with derivative conversion factors
               ! Radau: 1 - d1 ; 2 - d2 ; 3 - theta
               call cartesian_to_radau(rcart, rtrkn, dtrkndcart)
               
               ! Order and transform reduced coordinates
               ! No change in order
               rrkhs = rtrkn(:)
               
               ! Compute potential and derivatives in Radau
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)
               
               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn = dedrkhs(:)
               
               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedtrkn(1)*dtrkndcart(1,jj,ii)    &
                                      + dedtrkn(2)*dtrkndcart(2,jj,ii)  &
                                      + dedtrkn(3)*dtrkndcart(3,jj,ii)
                  end do
               end do

               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("rtdd")
               ! Radau coordinates ordered by
               ! angle theta(1-2-3)  in radians(!),
               ! bond distance d1(1-2) and d2(2-3).
               !        1       !
               !       / \      !
               !      /   \     !
               !   d1/  th \d2  !
               !    /       \   !
               !   /         \  !
               !  2           3 !
               ! Order: theta, d1, d2
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to Radau with derivative conversion factors
               ! Radau: 1 - d1 ; 2 - d2 ; 3 - theta
               call cartesian_to_radau(rcart, rtrkn, dtrkndcart)
               
               ! Order and transform reduced coordinates
               ! Switch d1 with theta and convert theta to z(theta)
               rrkhs(1) = rtrkn(3)
               rrkhs(2) = rtrkn(2)
               rrkhs(3) = rtrkn(1)
               
               ! Compute potential and derivatives in Radau
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)
               
               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn(1) = dedrkhs(3)
               dedtrkn(2) = dedrkhs(2)
               dedtrkn(3) = dedrkhs(1)
               
               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedtrkn(1)*dtrkndcart(1,jj,ii)    &
                                      + dedtrkn(2)*dtrkndcart(2,jj,ii)  &
                                      + dedtrkn(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("jdrz")
               ! Jacobi coordinates ordered by bond distance d1(1-2),
               ! distance CoM(1,2)-3 as R and angle theta(d1,R) in 
               ! z conversion (z = (1 - cos(theta))/2)
               !        3         theta = 180°  !
               !       /|\            1--2--3   !
               !      / | \                     !
               !   d3/ R|  \d2    theta = 0°    !
               !    /   |th \         3--1--2   !
               !   /____|____\                  !
               !  1    d1    2                  !
               ! Order: d1, R, z
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to internal coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - d2 ; 3 - d3
               call cartesian_to_internal(rcart, rintr, dtrkndcart)

               ! Compute internal to Jacobi coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - R ; 3 - theta
               mass = (/ amass(i), amass(j), amass(k)/)
               call internal_to_jacobi(rintr, mass, rtrkn, dtrkndintr)

               ! Order and transform reduced coordinates
               ! No change in order
               rrkhs = rtrkn(:)
               ! Convert theta to z(theta)
               rrkhs(3) = (1.0d0 - cos(rtrkn(3)))/2.0d0

               ! Compute potential and derivatives in Jacobi coordinates
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)

               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn = dedrkhs(:)
               ! Convert de/dz to de/dtheta
               dedtrkn(3) = dedrkhs(3)*sin(rtrkn(3))/2.0d0

               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates:
               
               ! Jacobi to internal coordinates
               do ii = 1, 3
                    dedintr(ii) = dedtrkn(1)*dtrkndintr(ii,1)     &
                                  + dedtrkn(2)*dtrkndintr(ii,2)   &
                                  + dedtrkn(3)*dtrkndintr(ii,3)
               end do

               ! Internal to Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedintr(1)*dtrkndcart(1,jj,ii)    &
                                      + dedintr(2)*dtrkndcart(2,jj,ii)  &
                                      + dedintr(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Numerical derivation
!                call get_potential_jacobi_numerical(   &
!                   kernk(m), kernt(:,m), kernm(m),     &
!                   rcart, mass, erkhs, dtrkndintr)
!                call debug_get_derivatives_jacobi_numerical(   &
!                   kernk(m), kernt(:,m), kernm(m),     &
!                   rcart, mass, erkhs, dtrkndintr)

!                if (sqrt(sum((dtrkndintr - dedcart)**2))/9.0d0 .ge. 1.0d-8) then
!                   write(*,*) 'Kernel ', m
!                   write(*,*) 'Cartesian ', rcart
!                   write(*,*) 'Jacobi ', rtrkn
!                   write(*,*) 'Analytical', dedcart
!                   write(*,*) 'Numerical ', dtrkndintr
!                   write(*,*) 'Deviation ',            &
!                      sqrt(sum((dtrkndintr - dedcart)**2))/9.0d0
!                end if

               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)
               
               ! Use numerical derivative 
!                dedcart = dtrkndintr*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("jzrd", "jaco")
               ! Jacobi coordinates ordered by angle theta(d1,R) in 
               ! z conversion (z = (1 - cos(theta))/2),
               ! distance CoM(1,2)-3 as R and bond distance d1(1-2).
               !        3         theta = 180°  !
               !       /|\            1--2--3   !
               !      / | \                     !
               !   d3/ R|  \d2    theta = 0°    !
               !    /   |th \         3--1--2   !
               !   /____|____\                  !
               !  1    d1    2                  !
               ! Order: z, R, d1
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to internal coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - d2 ; 3 - d3
               call cartesian_to_internal(rcart, rintr, dtrkndcart)

               ! Compute internal to Jacobi coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - R ; 3 - theta
               mass = (/ amass(i), amass(j), amass(k)/)
               call internal_to_jacobi(rintr, mass, rtrkn, dtrkndintr)

               ! Order and transform reduced coordinates
               ! Switch d1 with theta and convert theta to z(theta)
               rrkhs(1) = (1.0d0 - cos(rtrkn(3)))/2.0d0
               rrkhs(2) = rtrkn(2)
               rrkhs(3) = rtrkn(1)

               ! Compute potential and derivatives in Jacobi coordinates
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)

               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn(1) = dedrkhs(3)
               dedtrkn(2) = dedrkhs(2)
               dedtrkn(3) = dedrkhs(1)*sin(rtrkn(3))/2.0d0

               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates:
               
               ! Jacobi to internal coordinates
               do ii = 1, 3
                    dedintr(ii) = dedtrkn(1)*dtrkndintr(ii,1)     &
                                  + dedtrkn(2)*dtrkndintr(ii,2)   &
                                  + dedtrkn(3)*dtrkndintr(ii,3)
               end do

               ! Internal to Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedintr(1)*dtrkndcart(1,jj,ii)    &
                                      + dedintr(2)*dtrkndcart(2,jj,ii)  &
                                      + dedintr(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("jdrt")
               ! Jacobi coordinates ordered by bond distance d1(1-2),
               ! distance CoM(1,2)-3 as R and angle theta(d1,R) in 
               ! radians(!)
               !        3         theta = 180°  !
               !       /|\            1--2--3   !
               !      / | \                     !
               !   d3/ R|  \d2    theta = 0°    !
               !    /   |th \         3--1--2   !
               !   /____|____\                  !
               !  1    d1    2                  !
               ! Order: d1, R, theta
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to internal coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - d2 ; 3 - d3
               call cartesian_to_internal(rcart, rintr, dtrkndcart)

               ! Compute internal to Jacobi coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - R ; 3 - theta
               mass = (/ amass(i), amass(j), amass(k)/)
               call internal_to_jacobi(rintr, mass, rtrkn, dtrkndintr)

               ! Order and transform reduced coordinates
               ! No change in order
               rrkhs = rtrkn(:)
               
               ! Compute potential and derivatives in Jacobi coordinates
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)

               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn = dedrkhs(:)
               
               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates:
               
               ! Jacobi to internal coordinates
               do ii = 1, 3
                    dedintr(ii) = dedtrkn(1)*dtrkndintr(ii,1)     &
                                  + dedtrkn(2)*dtrkndintr(ii,2)   &
                                  + dedtrkn(3)*dtrkndintr(ii,3)
               end do

               ! Internal to Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedintr(1)*dtrkndcart(1,jj,ii)    &
                                      + dedintr(2)*dtrkndcart(2,jj,ii)  &
                                      + dedintr(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)
               
               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("jtrd")
               ! Jacobi coordinates ordered by angle theta(d1,R) in radians(!),
               ! distance CoM(1,2)-3 as R and bond distance d1(1-2).
               !        3         theta = 180°  !
               !       /|\            1--2--3   !
               !      / | \                     !
               !   d3/ R|  \d2    theta = 0°    !
               !    /   |th \         3--1--2   !
               !   /____|____\                  !
               !  1    d1    2                  !
               ! Order: theta, R, d1
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to internal coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - d2 ; 3 - d3
               call cartesian_to_internal(rcart, rintr, dtrkndcart)

               ! Compute internal to Jacobi coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - R ; 3 - theta
               mass = (/ amass(i), amass(j), amass(k)/)
               call internal_to_jacobi(rintr, mass, rtrkn, dtrkndintr)

               ! Order and transform reduced coordinates
               ! Switch d1 with theta
               rrkhs(1) = rtrkn(3)
               rrkhs(2) = rtrkn(2)
               rrkhs(3) = rtrkn(1)

               ! Compute potential and derivatives in Jacobi coordinates
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)

               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedtrkn(1) = dedrkhs(3)
               dedtrkn(2) = dedrkhs(2)
               dedtrkn(3) = dedrkhs(1)

               ! Convert derivative with respect to reduced coordinates to
               ! Cartesian coordinates:
               
               ! Jacobi to internal coordinates
               do ii = 1, 3
                    dedintr(ii) = dedtrkn(1)*dtrkndintr(ii,1)     &
                                  + dedtrkn(2)*dtrkndintr(ii,2)   &
                                  + dedtrkn(3)*dtrkndintr(ii,3)
               end do

               ! Internal to Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedintr(1)*dtrkndcart(1,jj,ii)    &
                                      + dedintr(2)*dtrkndcart(2,jj,ii)  &
                                      + dedintr(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)
            
            case("iddd", "i123", "int2")
               ! Internal coordinates ordered by bond distance d1(1-2),
               ! bond distance d2(2-3) and bond distance d3(2-3).
               !        3                       !
               !       / \                      !
               !      /   \                     !
               !   d3/     \d2                  !
               !    /       \                   !
               !   /____ ____\                  !
               !  1    d1    2                  !
               ! Order: d1, d2, d3
               
               ! Assign Cartesian coordinates and convert
               rcart(1, :) = (/ x(i), x(j), x(k) /)*kpcnv(m)
               rcart(2, :) = (/ y(i), y(j), y(k) /)*kpcnv(m)
               rcart(3, :) = (/ z(i), z(j), z(k) /)*kpcnv(m)

               ! Compute Cartesian to internal coordinates with derivative 
               ! conversion factors internal to Cartesian
               ! internal: 1 - d1 ; 2 - d2 ; 3 - d3
               call cartesian_to_internal(rcart, rintr, dtrkndcart)
               
               ! Order and transform reduced coordinates
               ! No change in order or conversion
               rrkhs = rintr(:)
               
               ! Compute potential and derivatives in Jacobi coordinates
               call get_energy( &
                  rrkhs, kernk(m), kernt(:,m), kernm(m), erkhs, dedrkhs)
               
               ! Back-Order and back-transform derivative of reduced and 
               ! manipulated coordinates
               dedintr = dedrkhs(:)
               
               ! Internal to Cartesian coordinates
               ! Iterate over atoms
               do ii = 1, 3
                  ! Iterate over Cartesians
                  do jj = 1, 3
                     dedcart(jj,ii) = dedintr(1)*dtrkndcart(1,jj,ii)    &
                                      + dedintr(2)*dtrkndcart(2,jj,ii)  &
                                      + dedintr(3)*dtrkndcart(3,jj,ii)
                  end do
               end do
               
               ! Convert energy and derivative units
               erkhs = erkhs*kecnv(m)
               dedcart = dedcart*kecnv(m)*kpcnv(m)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)

            case("hype")
               
               ! Hyperspherical coordinates, see how
               call calcener_hype(erkhs,dedcart)

               ! Add potential energy contribution
               ekern = ekern + erkhs

               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)


               case("cust")

               call calcener_cust(erkhs,dedcart)

               ! Add potential energy contribution
               ekern = ekern + erkhs
               
               ! Add derivative contribution
               dx(i) = dx(i) + dedcart(1, 1)
               dx(j) = dx(j) + dedcart(1, 2)
               dx(k) = dx(k) + dedcart(1, 3)
               dy(i) = dy(i) + dedcart(2, 1)
               dy(j) = dy(j) + dedcart(2, 2)
               dy(k) = dy(k) + dedcart(2, 3)
               dz(i) = dz(i) + dedcart(3, 1)
               dz(j) = dz(j) + dedcart(3, 2)
               dz(k) = dz(k) + dedcart(3, 3)


            case default
               call wrndie(-3,'<extbond>','Unknown corrdynate type. Implement your own coordype in calc_ener_custom')

         end select
!          write(outu,*) mynod, m, ekern, erkhs
!          flush(outu)
      enddo
      
      ! Update the CHARMM energy term. Due to the nature of kernels, it is 
      ! unfeasible to separate the contributions.
      ! For this reason this term is directly added to the bond energy term
      ebond = ebond + ekern
!       write(outu,*) mynod, ekern, ebond
      return

   end subroutine tria_kern_ener
   
   subroutine get_energy(rrkhs, filename, ktype, mtype, e, der)
      ! Get RKHS potential energy and derivates
      
!       use RKHS
      
      implicit none
      real(chm_real), dimension(:), intent(in) :: rrkhs(3)
      character(len=100), intent(in) :: filename
      integer, dimension(3), intent(in) :: ktype
      integer, intent(in) :: mtype
      real(chm_real), intent(out) :: e
      real(chm_real), dimension(3), intent(out) :: der
      !real(chm_real), parameter :: lambda = 1.0d-16

      if (.not. kread(mtype)) then
         
         if (.not. kload(mtype)) then
            ! 'kload' will be true if the file exists and false otherwise
            inquire(file=(trim(filename) // trim(kern)), exist=kload(mtype))
            if (kload(mtype)) call wrndie(3,'<extbond>',&
               'Kernel file already exists. No new kernel will be written.')
         end if
      
         if (kload(mtype)) then
            
            call pes(mtype)%load_from_file(trim(filename) // trim(kern))
            kread(mtype) = .true.
            
         else
            
            ! Read data from file
            call pes(mtype)%read_grid(trim(filename) // trim(csv))
            
            ! Choose one-dimensional kernel for dimension 1
            call pes(mtype)%k1d(1)%init(ktype(1)) 
            ! Choose one-dimensional kernel for dimension 2
            call pes(mtype)%k1d(2)%init(ktype(2))
            ! Choose one-dimensional kernel for dimension 3
            call pes(mtype)%k1d(3)%init(ktype(3))
            
            ! Determine kernel coefficients
            !call pes(mtype)%calculate_coefficients_slow(lambda)
            call pes(mtype)%calculate_coefficients_fast()
            call pes(mtype)%calculate_sums()
            
            ! Save kernels to file
            call pes(mtype)%save_to_file(trim(filename) // trim(kern))
            
            kread(mtype) = .true.
            
         end if
         
      end if
      
      ! Compute potential V and derivatives in kernel coordinates
      call pes(mtype)%evaluate_fast(rrkhs, e, der)
      
      return

   end subroutine get_energy

   subroutine cartesian_to_radau(rcart, rradau, dradaudcart)

      implicit none
      
      real(chm_real), dimension(:,:), intent(in) :: rcart(3,3)
      real(chm_real), dimension(:), intent(out) :: rradau(3)
      real(chm_real), dimension(:,:,:), intent(out) :: dradaudcart(3,3,3)
      
      real(chm_real) :: dot12, costheta, dacosx_dx, r12, r12sqr
      real(chm_real), dimension(:) :: d1(3), d2(3)
      
      ! Compute r1
      d1 = rcart(:, 1) - rcart(:, 2)
      rradau(1) = dsqrt(sum(d1*d1))
      
      ! Compute r2
      d2 = rcart(:, 3) - rcart(:, 2)
      rradau(2) = dsqrt(sum(d2*d2))
      
      ! Compute theta
      dot12 = sum(d1*d2)
      r12 = rradau(1)*rradau(2)
      costheta = dot12/r12
      rradau(3) = acos(costheta)
      
      ! dradaudcart : (radau, cart, atom)
      
      ! Compute dr1 / d cart_atom
      dradaudcart(1,1,1) =  d1(1)/rradau(1)
      dradaudcart(1,2,1) =  d1(2)/rradau(1)
      dradaudcart(1,3,1) =  d1(3)/rradau(1)
      dradaudcart(1,1,2) = -d1(1)/rradau(1)
      dradaudcart(1,2,2) = -d1(2)/rradau(1)
      dradaudcart(1,3,2) = -d1(3)/rradau(1)
      dradaudcart(1,1,3) =  0.0d0
      dradaudcart(1,2,3) =  0.0d0
      dradaudcart(1,3,3) =  0.0d0
      
      ! Compute dr2 / d cart_atom
      dradaudcart(2,1,1) =  0.0d0
      dradaudcart(2,2,1) =  0.0d0
      dradaudcart(2,3,1) =  0.0d0
      dradaudcart(2,1,2) = -d2(1)/rradau(2)
      dradaudcart(2,2,2) = -d2(2)/rradau(2)
      dradaudcart(2,3,2) = -d2(3)/rradau(2)
      dradaudcart(2,1,3) =  d2(1)/rradau(2)
      dradaudcart(2,2,3) =  d2(2)/rradau(2)
      dradaudcart(2,3,3) =  d2(3)/rradau(2)
      
      ! Compute dtheta / d cart_atom
      dacosx_dx = -1.0d0/sqrt(1.0d0 - costheta**2)
      dradaudcart(3,1,1) = dacosx_dx/r12                           &
                              *(d2(1)                                 &
                              - dot12/r12*(                           &                              
                                 dradaudcart(1,1,1)*rradau(2)))
      dradaudcart(3,2,1) = dacosx_dx/r12                           &
                              *(d2(2)                                 &                              
                              - dot12/r12*(                           &
                                 dradaudcart(1,2,1)*rradau(2)))
      dradaudcart(3,3,1) = dacosx_dx/r12                           &
                              *(d2(3)                                 &                              
                              - dot12/r12*(                           &
                                 dradaudcart(1,3,1)*rradau(2)))
      dradaudcart(3,1,2) = dacosx_dx/r12                           &
                              *(-(d1(1) + d2(1))                      &
                              - dot12/r12*(                           &
                                 dradaudcart(1,1,2)*rradau(2)   &
                              + rradau(1)*dradaudcart(2,1,2)))
      dradaudcart(3,2,2) = dacosx_dx/r12                           &
                              *(-(d1(2) + d2(2))                      &
                              - dot12/r12*(                           &
                                 dradaudcart(1,2,2)*rradau(2)   &
                                 + rradau(1)*dradaudcart(2,2,2)))
      dradaudcart(3,3,2) = dacosx_dx/r12                           &
                              *(-(d1(3) + d2(3))                      &
                              - dot12/r12*(                           &
                                 dradaudcart(1,3,2)*rradau(2)   &
                                 + rradau(1)*dradaudcart(2,3,2)))
      dradaudcart(3,1,3) = dacosx_dx/r12                           &
                              *(d1(1)                                 &
                              - dot12/r12*(                           &
                                 rradau(1)*dradaudcart(2,1,3)) )
      dradaudcart(3,2,3) = dacosx_dx/r12                           &
                              *(d1(2)                                 &
                              - dot12/r12*(                           &
                                 rradau(1)*dradaudcart(2,2,3)))
      dradaudcart(3,3,3) = dacosx_dx/r12                           &
                              *(d1(3)                                 &
                              - dot12/r12*(                           &
                                 rradau(1)*dradaudcart(2,3,3)))
      
      return
   
   end subroutine cartesian_to_radau

   subroutine cartesian_to_internal(rcart, rintr, dintrdcart)

      implicit none
      
      real(chm_real), dimension(:,:), intent(in) :: rcart(3,3)
      real(chm_real), dimension(:), intent(out) :: rintr(3)
      real(chm_real), dimension(:,:,:), intent(out) :: dintrdcart(3,3,3)
      
      real(chm_real), dimension(:,:) :: drcrt(3,3)
      
      ! Internal vectors (cart, atom)
      drcrt(1,1) = rcart(1,2) - rcart(1,1)
      drcrt(2,1) = rcart(2,2) - rcart(2,1)
      drcrt(3,1) = rcart(3,2) - rcart(3,1)   ! 1->2
      drcrt(1,2) = rcart(1,3) - rcart(1,2)
      drcrt(2,2) = rcart(2,3) - rcart(2,2)
      drcrt(3,2) = rcart(3,3) - rcart(3,2)   ! 2->3
      drcrt(1,3) = rcart(1,1) - rcart(1,3)
      drcrt(2,3) = rcart(2,1) - rcart(2,3)
      drcrt(3,3) = rcart(3,1) - rcart(3,3)   ! 3->1
      
      ! Internal distances 
      rintr(1) = sqrt(                 &
            (drcrt(1,1)*drcrt(1,1))    &
            + (drcrt(2,1)*drcrt(2,1))  &
            + (drcrt(3,1)*drcrt(3,1)))
      rintr(2) = sqrt(                 &
         (drcrt(1,2)*drcrt(1,2))       &
         + (drcrt(2,2)*drcrt(2,2))     &
         + (drcrt(3,2)*drcrt(3,2)))
      rintr(3) = sqrt(                 &
         (drcrt(1,3)*drcrt(1,3))       &
         + (drcrt(2,3)*drcrt(2,3))     &
         + (drcrt(3,3)*drcrt(3,3)))
      
      ! Derivatives: internal to cartesian (cart, intr, atom)
      dintrdcart(1,1,1) = -drcrt(1,1)/rintr(1)
      dintrdcart(1,1,2) =  drcrt(1,1)/rintr(1)
      dintrdcart(1,1,3) =  0.0d0
      dintrdcart(1,2,1) = -drcrt(2,1)/rintr(1)
      dintrdcart(1,2,2) =  drcrt(2,1)/rintr(1)
      dintrdcart(1,2,3) = 0.0d0
      dintrdcart(1,3,1) = -drcrt(3,1)/rintr(1)
      dintrdcart(1,3,2) =  drcrt(3,1)/rintr(1)
      dintrdcart(1,3,3) = 0.0d0

      dintrdcart(2,1,1) = 0.0d0
      dintrdcart(2,1,2) = -drcrt(1,2)/rintr(2)
      dintrdcart(2,1,3) =  drcrt(1,2)/rintr(2)
      dintrdcart(2,2,1) = 0.0d0
      dintrdcart(2,2,2) = -drcrt(2,2)/rintr(2)
      dintrdcart(2,2,3) =  drcrt(2,2)/rintr(2)
      dintrdcart(2,3,1) = 0.0d0
      dintrdcart(2,3,2) = -drcrt(3,2)/rintr(2)
      dintrdcart(2,3,3) =  drcrt(3,2)/rintr(2)

      dintrdcart(3,1,1) =  drcrt(1,3)/rintr(3)
      dintrdcart(3,1,2) = 0.0d0
      dintrdcart(3,1,3) = -drcrt(1,3)/rintr(3)
      dintrdcart(3,2,1) =  drcrt(2,3)/rintr(3)
      dintrdcart(3,2,2) = 0.0d0
      dintrdcart(3,2,3) = -drcrt(2,3)/rintr(3)
      dintrdcart(3,3,1) =  drcrt(3,3)/rintr(3)
      dintrdcart(3,3,2) = 0.0d0
      dintrdcart(3,3,3) = -drcrt(3,3)/rintr(3)
      
      return
   
   end subroutine cartesian_to_internal
   
   subroutine internal_to_jacobi(rintr, mass, rjaco, djacodintr)

      implicit none
      
      real(chm_real), dimension(:), intent(in) :: rintr(3), mass(3)
      real(chm_real), dimension(:), intent(out) :: rjaco(3)
      real(chm_real), dimension(:,:), intent(out) :: djacodintr(3,3)
      
      real(chm_real), parameter :: pc = sqrt(epsilon(1.0d0))
      real(chm_real) :: cmu1, cmu2, rcm1, rcm2
      
      ! Mass parameter
      cmu1 = mass(2)/(mass(1) + mass(2))
      cmu2 = mass(1)/(mass(1) + mass(2))
      
      ! Atom 1 and 2 to CoM distances
      rcm1 = rintr(1)*cmu1
      rcm2 = rintr(1)*cmu2
      
      ! Distance 1-2 : d (or r)
      rjaco(1) = rintr(1)
      
      ! Distance CoM(1-2)-3 : r (or R)
      rjaco(2) = sqrt(  &
         rintr(2)**2/rintr(1)*rcm1 + rintr(3)**2/rintr(1)*rcm2 - rcm1*rcm2)
      if (abs(rjaco(2)) < pc) rjaco(2) = pc
      
      ! Theta
      rjaco(3) = (rcm2**2 + rjaco(2)**2 - rintr(2)**2)/2.0d0/rcm2/rjaco(2)
      rjaco(3) = min(1.0d0, max(-1.0d0, rjaco(3)))
      rjaco(3) = acos(rjaco(3))
      
      ! djacodintr : (jaco, atom)
      ! r
      djacodintr(:,1) = (/ 1.0d0, 0.0d0, 0.0d0 /)
      ! R
      djacodintr(1,2) = -cmu1*cmu2*rintr(1)/rjaco(2)
      djacodintr(2,2) = rintr(2)*cmu1/rjaco(2)
      djacodintr(3,2) = rintr(3)*cmu2/rjaco(2)
      ! Theta
      djacodintr(1,3) = (                                               &
         djacodintr(1,2)/rjaco(2)*cos(rjaco(3)) + cos(rjaco(3))/rintr(1)&
         - (rjaco(2)*djacodintr(1,2)+rcm2*cmu2)/rcm2/rjaco(2))          &
         /sqrt(1.0d0-cos(rjaco(3))**2)
      djacodintr(2,3) = (                                               &
         rintr(2)/rcm2/rjaco(2) - djacodintr(2,2)/rcm2                  &
         + cos(rjaco(3))/rjaco(2)*djacodintr(2,2))                      &
         /sqrt(1.0d0 - cos(rjaco(3))**2)
      djacodintr(3,3) = (                                               &
         cos(rjaco(3))/rjaco(2)*djacodintr(3,2) - djacodintr(3,2)/rcm2) &
         /sqrt(1.0d0 - cos(rjaco(3))**2)
      
      return
      
   end subroutine internal_to_jacobi
   
   subroutine get_potential_radau_numerical(  &
      filename, ktype, mtype, pos_cart, V, der_cart)
      ! Get Cartesian derivates numerically for Radau d1,d2,z
      
      use chm_kinds
      
      implicit none
      
      real(chm_real), dimension(:,:), intent(in) :: pos_cart(3,3)
      character(len=100), intent(in) :: filename
      integer, dimension(3), intent(in) :: ktype
      integer, intent(in) :: mtype
      real(chm_real), intent(out) :: V
      real(chm_real), dimension(:,:), intent(out) :: der_cart(3,3)
      
      integer :: ii, jj, kk
      real(chm_real) :: d0h, d02h
      real(chm_real), dimension(:) :: pos_radau(3), der_radau(3), V_temp(4)
      real(chm_real), dimension(3) :: ddz
      real(chm_real), dimension(:,:) :: pos_temp(3,3)
      real(chm_real), dimension(:,:,:) :: der(3,3,3)
      
      real(chm_real), parameter :: h=0.0001d0
      
      ! Compute potential energy
      call cartesian_to_radau(pos_cart, pos_radau, der)
      ddz = pos_radau(:)
      ddz(3) = (1.0d0 - cos(pos_radau(3)))/2.0d0
      call get_energy(ddz, filename, ktype, mtype, V, der_radau)
      
      ! Iterate over atoms
      do ii = 1, 3
         
         ! Iterate over Cartesians
         do jj = 1, 3
            
            ! Refresh auxiliary coordinate array
            pos_temp = pos_cart
            
            ! -2h
            pos_temp(jj, ii) = pos_cart(jj, ii) - 2.0d0*h
            call cartesian_to_radau(pos_temp, pos_radau, der)
            ddz = pos_radau(:)
            ddz(3) = (1.0d0 - cos(pos_radau(3)))/2.0d0
            call get_energy( &
               ddz, filename, ktype, mtype, V_temp(1), der_radau)
            
            ! -h
            pos_temp(jj, ii) = pos_cart(jj, ii) - h
            call cartesian_to_radau(pos_temp, pos_radau, der)
            ddz = pos_radau(:)
            ddz(3) = (1.0d0 - cos(pos_radau(3)))/2.0d0
            call get_energy( &
               ddz, filename, ktype, mtype, V_temp(2), der_radau)
            
            ! +h
            pos_temp(jj, ii) = pos_cart(jj, ii) + h
            call cartesian_to_radau(pos_temp, pos_radau, der)
            ddz = pos_radau(:)
            ddz(3) = (1.0d0 - cos(pos_radau(3)))/2.0d0
            call get_energy( &
               ddz, filename, ktype, mtype, V_temp(3), der_radau)
            
            ! +2h
            pos_temp(jj, ii) = pos_cart(jj, ii) + 2.0d0*h
            call cartesian_to_radau(pos_temp, pos_radau, der)
            ddz = pos_radau(:)
            ddz(3) = (1.0d0 - cos(pos_radau(3)))/2.0d0
            call get_energy( &
               ddz, filename, ktype, mtype, V_temp(4), der_radau)
            
            ! Compute numerical derivative
            d0h = (V_temp(3) - V_temp(2))/2.0d0/h
            d02h = (V_temp(4) - V_temp(1))/4.0d0/h
            der_cart(jj, ii) = (4.0d0*d0h - d02h)/3.0d0
            
         end do
         
      end do

      return

   end subroutine get_potential_radau_numerical
   
   subroutine get_potential_jacobi_numerical(  &
      filename, ktype, mtype, pos_cart, mass, V, der_cart)
      ! Get Cartesian derivates numerically for Jacobi d1,R,z
      
      use chm_kinds
      
      implicit none
      
      real(chm_real), dimension(:), intent(in) :: mass(3)
      real(chm_real), dimension(:,:), intent(in) :: pos_cart(3,3)
      character(len=100), intent(in) :: filename
      integer, dimension(3), intent(in) :: ktype
      integer, intent(in) :: mtype
      real(chm_real), intent(out) :: V
      real(chm_real), dimension(:,:), intent(out) :: der_cart(3,3)
      
      integer :: ii, jj, kk
      real(chm_real) :: d0h, d02h
      real(chm_real), dimension(:) :: pos_internal(3)
      real(chm_real), dimension(:) :: pos_jacobi(3)
      real(chm_real), dimension(:) :: V_temp(4)
      real(chm_real), dimension(:) :: pos_rkhs(3), der1(3)
      real(chm_real), dimension(:,:) :: pos_temp(3,3), der2(3,3)
      real(chm_real), dimension(:,:,:) :: der3(3,3,3)
      
      real(chm_real), parameter :: h=0.0001d0
      
      ! Compute potential energy
      call cartesian_to_internal(pos_cart, pos_internal, der3)
      call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
      pos_rkhs = pos_jacobi(:)
      pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
      call get_energy(pos_rkhs, filename, ktype, mtype, V, der1)

      ! Iterate over atoms
      do ii = 1, 3
         
         ! Iterate over Cartesians
         do jj = 1, 3
            
            ! Refresh auxiliary coordinate array
            pos_temp = pos_cart
            
            ! -2h
            pos_temp(jj, ii) = pos_cart(jj, ii) - 2.0d0*h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(1), der1)
            
            ! -h
            pos_temp(jj, ii) = pos_cart(jj, ii) - h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(2), der1)
            
            ! +h
            pos_temp(jj, ii) = pos_cart(jj, ii) + h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(3), der1)
            
            ! +2h
            pos_temp(jj, ii) = pos_cart(jj, ii) + 2.0d0*h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(4), der1)
            
            ! Compute numerical derivative
            d0h = (V_temp(3) - V_temp(2))/2.0d0/h
            d02h = (V_temp(4) - V_temp(1))/4.0d0/h
            der_cart(jj, ii) = (4.0d0*d0h - d02h)/3.0d0
            
         end do
         
      end do

      return

   end subroutine get_potential_jacobi_numerical

   subroutine debug_get_derivatives_jacobi_numerical(  &
      filename, ktype, mtype, pos_cart, mass, V, der_cart)
      ! Debugging function to compute derivatives numerically
      
      use chm_kinds
      
      implicit none
      
      real(chm_real), dimension(:), intent(in) :: mass(3)
      real(chm_real), dimension(:,:), intent(in) :: pos_cart(3,3)
      character(len=100), intent(in) :: filename
      integer, dimension(3), intent(in) :: ktype
      integer, intent(in) :: mtype
      real(chm_real), intent(out) :: V
      real(chm_real), dimension(:,:), intent(out) :: der_cart(3,3)
      
      integer :: ii, jj, kk
      real(chm_real) :: d0h, d02h
      real(chm_real), dimension(:) :: pos_internal(3)
      real(chm_real), dimension(:) :: pos_jacobi(3)
      real(chm_real), dimension(:) :: V_temp(4)
      real(chm_real), dimension(:) :: pos_rkhs(3), der0(3), der1(3)
      real(chm_real), dimension(:,:) :: pos_temp(3,3), der2(3,3)
      real(chm_real), dimension(:,:,:) :: der3(3,3,3)
      
      real(chm_real), parameter :: h=0.0001d0
      
      ! Compute potential energy
      call cartesian_to_internal(pos_cart, pos_internal, der3)
      call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
      pos_rkhs = pos_jacobi(:)
      pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
      call get_energy(pos_rkhs, filename, ktype, mtype, V, der1)
!       write(*,*) 'Numerical ekern:', V
      
      ! RKHS derivatives
      
      ! Iterate over RKHS coordinates
      do ii = 1, 3
         
         ! Refresh auxiliary coordinate array
         pos_temp(:,1) = pos_rkhs
         
         ! -2h
         pos_temp(ii, 1) = pos_rkhs(ii) - 2.0d0*h
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(1), der0)
         
         ! -h
         pos_temp(ii, 1) = pos_rkhs(ii) - h
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(2), der0)
         
         ! +h
         pos_temp(ii, 1) = pos_rkhs(ii) + h
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(3), der0)
         
         ! +2h
         pos_temp(ii, 1) = pos_rkhs(ii) + 2.0d0*h
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(4), der0)
         
         ! Compute numerical derivative
         d0h = (V_temp(3) - V_temp(2))/2.0d0/h
         d02h = (V_temp(4) - V_temp(1))/4.0d0/h
         der1(ii) = (4.0d0*d0h - d02h)/3.0d0
         
      end do
!       write(*,*) 'Numerical derivatives RKHS:', der1
      
      ! Jacobi derivatives
      
      ! Iterate over Jacobi coordinates
      do ii = 1, 3
         
         ! Refresh auxiliary coordinate array
         pos_temp(:, 2) = pos_jacobi
         
         ! -2h
         pos_temp(:, 1) = pos_jacobi
         pos_temp(ii, 1) = pos_jacobi(ii) - 2.0d0*h
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(1), der0)
         
         ! -h
         pos_temp(:, 1) = pos_jacobi
         pos_temp(ii, 1) = pos_jacobi(ii) - h
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(2), der0)
         
         ! +h
         pos_temp(:, 1) = pos_jacobi
         pos_temp(ii, 1) = pos_jacobi(ii) + h
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(3), der0)
         
         ! +2h
         pos_temp(:, 1) = pos_jacobi
         pos_temp(ii, 1) = pos_jacobi(ii) + 2.0d0*h
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(4), der0)
         
         ! Compute numerical derivative
         d0h = (V_temp(3) - V_temp(2))/2.0d0/h
         d02h = (V_temp(4) - V_temp(1))/4.0d0/h
         der1(ii) = (4.0d0*d0h - d02h)/3.0d0
         
      end do
!       write(*,*) 'Numerical derivatives Jacobi:', der1
      
      ! Internal derivatives
      
      ! Iterate over internal coordinates
      do ii = 1, 3
         
         ! Refresh auxiliary coordinate array
         pos_temp(:, 2) = pos_internal
         
         ! -2h
         pos_temp(ii, 2) = pos_internal(ii) - 2.0d0*h
         call internal_to_jacobi(pos_temp(:,2), mass, pos_temp(:,1), der2)
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(1), der0)
         
         ! -h
         pos_temp(ii, 2) = pos_internal(ii) - h
         call internal_to_jacobi(pos_temp(:,2), mass, pos_temp(:,1), der2)
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(2), der0)
         
         ! +h
         pos_temp(ii, 2) = pos_internal(ii) + h
         call internal_to_jacobi(pos_temp(:,2), mass, pos_temp(:,1), der2)
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(3), der0)
         
         ! +2h
         pos_temp(ii, 2) = pos_internal(ii) + 2.0d0*h
         call internal_to_jacobi(pos_temp(:,2), mass, pos_temp(:,1), der2)
         pos_temp(3, 1) = (1.0d0 - cos(pos_temp(3, 1)))/2.0d0
         call get_energy( &
            pos_temp(:, 1), filename, ktype, mtype, V_temp(4), der0)

         ! Compute numerical derivative
         d0h = (V_temp(3) - V_temp(2))/2.0d0/h
         d02h = (V_temp(4) - V_temp(1))/4.0d0/h
         der1(ii) = (4.0d0*d0h - d02h)/3.0d0
         
      end do
!       write(*,*) 'Numerical derivatives Internal:', der1
      
      ! Cartesian derivatives
      
      ! Iterate over atoms
      do ii = 1, 3
         
         ! Iterate over Cartesians
         do jj = 1, 3
            
            ! Refresh auxiliary coordinate array
            pos_temp = pos_cart
            
            ! -2h
            pos_temp(jj, ii) = pos_cart(jj, ii) - 2.0d0*h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(1), der1)
            
            ! -h
            pos_temp(jj, ii) = pos_cart(jj, ii) - h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(2), der1)
            
            ! +h
            pos_temp(jj, ii) = pos_cart(jj, ii) + h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(3), der1)
            
            ! +2h
            pos_temp(jj, ii) = pos_cart(jj, ii) + 2.0d0*h
            call cartesian_to_internal(pos_temp, pos_internal, der3)
            call internal_to_jacobi(pos_internal, mass, pos_jacobi, der2)
            pos_rkhs = pos_jacobi(:)
            pos_rkhs(3) = (1.0d0 - cos(pos_jacobi(3)))/2.0d0
            call get_energy( &
               pos_rkhs, filename, ktype, mtype, V_temp(4), der1)
            
            ! Compute numerical derivative
            d0h = (V_temp(3) - V_temp(2))/2.0d0/h
            d02h = (V_temp(4) - V_temp(1))/4.0d0/h
            der_cart(jj, ii) = (4.0d0*d0h - d02h)/3.0d0
            
         end do
         
      end do
!       write(*,*) 'Numerical dedcart:', der_cart

      return

   end subroutine debug_get_derivatives_jacobi_numerical


subroutine calcener_hype(ekern,dedcart)
 use RKHS
 implicit none

 real(chm_real),intent(out) :: ekern
 real(chm_real),intent(out),dimension(3,3) :: dedcart

 return
end subroutine calcener_hype

subroutine calcener_cust(ekern,dedcart)
 use RKHS
 implicit none

 real(chm_real), intent(out) :: ekern, dedcart(3,3)
 write(*,'(A)') "TRIAKERN> CUSTOM ENERGY subroutine: This a sandbox subroutine that can be used to implement your kernels. "
 write(*,'(A)') "          You need to implement the gradients in the 'cust' section in  the 'tria_kern_ener' subroutine. "

 return
end subroutine calcener_cust


real function taylor(x) result(y)
! function that convert an angle (in radiants) to a suitable version used in the kernels
 real(chm_real),intent(in) :: x
 y= (1.000-x)/2.0
 return
end function

end module triakern
