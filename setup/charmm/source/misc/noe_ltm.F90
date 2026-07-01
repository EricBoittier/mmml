module noem
  use chm_kinds, only: chm_real

  implicit none

#if KEY_NOMISC==0 /*noe_fcm*/
  !
  !  data structure (COMMON block) for NOE constraints
  !
  ! actual number of NOE constraints:
  INTEGER NOENUM
  ! actual number of NOE atoms:
  INTEGER NOENM2
  ! scale factor for NOE energies and forces
  real(chm_real) NOESCA
  ! Atom lists pointer for NOE atom indicies
  INTEGER, allocatable, dimension(:) :: NOEIPT, NOEJPT
  ! Atom lists counts
  INTEGER, allocatable, dimension(:) ::  NOEINM, NOEJNM
  ! Atom lists for NOE indicies
  INTEGER, allocatable, dimension(:) ::  NOELIS
#if KEY_PNOE==1
  ! Logical switch to PNOE (atom(s)<-->point NOE) for EACH noe
  LOGICAL, allocatable, dimension(:) ::  IsPNOE
  ! Point coordinates for EACH pnoe
  real(chm_real), allocatable, dimension(:) :: C0X,C0Y,C0Z
  ! Logical switch from fixed to moving PNOE
  LOGICAL, allocatable, dimension(:) ::  MVPNOE
  ! Original and target point coordinates for EACH moving pnoe
  real(chm_real), allocatable, dimension(:) :: OC0X,OC0Y,OC0Z,TC0X,TC0Y,TC0Z
  ! Actual step No and total No of steps for moving PNOE (same for all)
  INTEGER NMPNOE,IMPNOE
#endif
  ! list for averaging method for computing (interproton) distance RIJ
  INTEGER, allocatable, dimension(:) ::  NOERAM
  ! list for minimum distances and force constants
  real(chm_real), allocatable, dimension(:) :: NOERMN, NOEKMN
  ! list for maximum distances and force constants
  real(chm_real), allocatable, dimension(:) :: NOERMX, NOEKMX, NOEFMX
  ! list for switch distance and soft-exponent
  real(chm_real), allocatable, dimension(:) :: NOERSW, NOESEX
  ! list of time constant and average 1/R**3 distance value
  real(chm_real), allocatable, dimension(:) :: NOETCN, NOEAVE
  ! Exponent reciprocal exponent for multiple atom averaging.
  real(chm_real), allocatable, dimension(:) :: NOEEXP
  ! Logical for mindist
  LOGICAL, allocatable, dimension(:) :: NOEMIN

contains

  subroutine noe_init
    use memory, only: chmalloc
    use chm_kinds, only: chm_real
    use dimens_fcm, only: noemax
    
    implicit none
    
    noenum=0
    noenm2=0
    noesca=1.0_chm_real

    if (allocated(noelis)) call noe_uninit() ! deallocate in case of reset
    
    call chmalloc('noe_ltm.src', 'noe_init', 'noeipt', noemax, intg=noeipt)
    call chmalloc('noe_ltm.src', 'noe_init', 'noejpt', noemax, intg=noejpt)
    call chmalloc('noe_ltm.src', 'noe_init', 'noeinm', noemax, intg=noeinm)
    call chmalloc('noe_ltm.src', 'noe_init', 'noejnm', noemax, intg=noejnm)
    call chmalloc('noe_ltm.src', 'noe_init', 'noelis', noemax, intg=noelis)
    call chmalloc('noe_ltm.src', 'noe_init', 'noeram', noemax, intg=noeram)

    noeipt = 0
    noejpt = 0
    noeinm = 0
    noejnm = 0
    noelis = 0
    noeram = 0
    
    call chmalloc('noe_ltm.src', 'noe_init', 'noemin', noemax, log=noemin)

    noemin = .false.
    
    call chmalloc('noe_ltm.src', 'noe_init', 'noermn', noemax, crl=noermn)
    call chmalloc('noe_ltm.src', 'noe_init', 'noekmn', noemax, crl=noekmn)
    call chmalloc('noe_ltm.src', 'noe_init', 'noermx', noemax, crl=noermx)
    call chmalloc('noe_ltm.src', 'noe_init', 'noekmx', noemax, crl=noekmx)
    call chmalloc('noe_ltm.src', 'noe_init', 'noefmx', noemax, crl=noefmx)
    call chmalloc('noe_ltm.src', 'noe_init', 'noersw', noemax, crl=noersw)
    call chmalloc('noe_ltm.src', 'noe_init', 'noesex', noemax, crl=noesex)
    call chmalloc('noe_ltm.src', 'noe_init', 'noetcn', noemax, crl=noetcn)
    call chmalloc('noe_ltm.src', 'noe_init', 'noeave', noemax, crl=noeave)
    call chmalloc('noe_ltm.src', 'noe_init', 'noeexp', noemax, crl=noeexp)

    noermn = 0.0
    noekmn = 0.0
    noermx = 0.0
    noekmx = 0.0
    noefmx = 0.0
    noersw = 0.0
    noesex = 0.0
    noetcn = 0.0
    noeave = 0.0
    noeexp = 0.0
    
#if KEY_PNOE==1
    call chmalloc('noe_ltm.src', 'noe_init', 'ispnoe', noemax, log=ispnoe)
    call chmalloc('noe_ltm.src', 'noe_init', 'mvpnoe', noemax, log=mvpnoe)

    ispnoe = .false.
    mvpnoe = .false.
    
    call chmalloc('noe_ltm.src', 'noe_init', 'c0x', noemax, crl=c0x)
    call chmalloc('noe_ltm.src', 'noe_init', 'c0y', noemax, crl=c0y)
    call chmalloc('noe_ltm.src', 'noe_init', 'c0z', noemax, crl=c0z)

    c0x = 0.0
    c0y = 0.0
    c0z = 0.0
    
    call chmalloc('noe_ltm.src', 'noe_init', 'oc0x', noemax, crl=oc0x)
    call chmalloc('noe_ltm.src', 'noe_init', 'oc0y', noemax, crl=oc0y)
    call chmalloc('noe_ltm.src', 'noe_init', 'oc0z', noemax, crl=oc0z)

    oc0x = 0.0
    oc0y = 0.0
    oc0z = 0.0
        
    call chmalloc('noe_ltm.src', 'noe_init', 'tc0x', noemax, crl=tc0x)
    call chmalloc('noe_ltm.src', 'noe_init', 'tc0y', noemax, crl=tc0y)
    call chmalloc('noe_ltm.src', 'noe_init', 'tc0z', noemax, crl=tc0z)

    tc0x = 0.0
    tc0y = 0.0
    tc0z = 0.0
#endif

    return
  end subroutine noe_init

  subroutine noe_uninit()
    use memory, only: chmdealloc
    use dimens_fcm, only: noemax
    
    implicit none
    
    call chmdealloc('noe_ltm.src', 'noe_init', 'noeipt', noemax, intg=noeipt)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noejpt', noemax, intg=noejpt)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noeinm', noemax, intg=noeinm)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noejnm', noemax, intg=noejnm)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noelis', noemax, intg=noelis)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noeram', noemax, intg=noeram)
    
    call chmdealloc('noe_ltm.src', 'noe_init', 'noemin', noemax, log=noemin)

    call chmdealloc('noe_ltm.src', 'noe_init', 'noermn', noemax, crl=noermn)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noekmn', noemax, crl=noekmn)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noermx', noemax, crl=noermx)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noekmx', noemax, crl=noekmx)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noefmx', noemax, crl=noefmx)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noersw', noemax, crl=noersw)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noesex', noemax, crl=noesex)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noetcn', noemax, crl=noetcn)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noeave', noemax, crl=noeave)
    call chmdealloc('noe_ltm.src', 'noe_init', 'noeexp', noemax, crl=noeexp)
    
#if KEY_PNOE==1
    call chmdealloc('noe_ltm.src', 'noe_init', 'ispnoe', noemax, log=ispnoe)
    call chmdealloc('noe_ltm.src', 'noe_init', 'mvpnoe', noemax, log=mvpnoe)

    call chmdealloc('noe_ltm.src', 'noe_init', 'c0x', noemax, crl=c0x)
    call chmdealloc('noe_ltm.src', 'noe_init', 'c0y', noemax, crl=c0y)
    call chmdealloc('noe_ltm.src', 'noe_init', 'c0z', noemax, crl=c0z)

    call chmdealloc('noe_ltm.src', 'noe_init', 'oc0x', noemax, crl=oc0x)
    call chmdealloc('noe_ltm.src', 'noe_init', 'oc0y', noemax, crl=oc0y)
    call chmdealloc('noe_ltm.src', 'noe_init', 'oc0z', noemax, crl=oc0z)
        
    call chmdealloc('noe_ltm.src', 'noe_init', 'tc0x', noemax, crl=tc0x)
    call chmdealloc('noe_ltm.src', 'noe_init', 'tc0y', noemax, crl=tc0y)
    call chmdealloc('noe_ltm.src', 'noe_init', 'tc0z', noemax, crl=tc0z)
#endif

    return
  end subroutine noe_uninit

  subroutine noe_add_storage()
    use memory, only: chmrealloc
    use dimens_fcm, only: noemax
    
    implicit none

    noemax = 2 * noemax

    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noeipt', noemax, intg=noeipt)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noejpt', noemax, intg=noejpt)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noeinm', noemax, intg=noeinm)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noejnm', noemax, intg=noejnm)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noelis', noemax, intg=noelis)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noeram', noemax, intg=noeram)

    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noemin', noemax, log=noemin)
    
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noermn', noemax, crl=noermn)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noekmn', noemax, crl=noekmn)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noermx', noemax, crl=noermx)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noekmx', noemax, crl=noekmx)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noefmx', noemax, crl=noefmx)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noersw', noemax, crl=noersw)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noesex', noemax, crl=noesex)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noetcn', noemax, crl=noetcn)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noeave', noemax, crl=noeave)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'noeexp', noemax, crl=noeexp)
    
#if KEY_PNOE==1
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'ispnoe', noemax, log=ispnoe)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'mvpnoe', noemax, log=mvpnoe)
    
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'c0x', noemax, crl=c0x)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'c0y', noemax, crl=c0y)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'c0z', noemax, crl=c0z)
    
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'oc0x', noemax, crl=oc0x)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'oc0y', noemax, crl=oc0y)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'oc0z', noemax, crl=oc0z)
        
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'tc0x', noemax, crl=tc0x)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'tc0y', noemax, crl=tc0y)
    call chmrealloc('noe_ltm.src', 'noe_add_storage', 'tc0z', noemax, crl=tc0z)    
#endif
  end subroutine noe_add_storage

#endif /* (noe_fcm)*/

end module noem
