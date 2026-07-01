module hash
  ! *
  ! * Hash table for integers
  ! *
  ! * August 2014, Antti-Pekka Hynninen, antti.pekka.hynninen@nrel.gov
  ! *
  use chm_kinds
  implicit none
  private

  ! Hash structure with linked list
  type hash_t
     sequence
     integer key, val, next
  end type hash_t

  type hashtable_t
     ! Number of keys
     integer nkey
     ! Number of buckets
     integer n
     ! Size of the allocated space for keys
     integer size
     ! Starting position for storing keys in linked list
     integer start
     ! Hash table entries hash(0:size-1)
     type(hash_t), allocatable, dimension(:) :: hash
  end type hashtable_t

  public hashtable_t

  public init_hash, uninit_hash, clear_hash, set_hash, get_hash, change_hash, reinit_hash
  public and_hash, or_hash, set_or_hash

contains

  ! *
  ! * Initialize hash table. If hash table is already allocated, deallocates it and restarts
  ! *
  subroutine init_hash(hashtable, nkey)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: nkey
    ! Variables

    if (allocated(hashtable%hash)) then
       call uninit_hash(hashtable)
       !call wrndie(-5,'<hash>','init_hash, hash table is already allocated')
    endif

    hashtable%n = max(4, 2**int(ceiling(log(real(2*nkey))/log(2.0))) )
    hashtable%size = int(hashtable%n*1.5)
    hashtable%start = hashtable%n   ! Start storing at the end of hash
    allocate(hashtable%hash(0:hashtable%size-1))

    call clear_hash(hashtable)

    return
  end subroutine init_hash

  ! *
  ! * Uninitialize hash table
  ! *
  subroutine uninit_hash(hashtable)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable

    if (allocated(hashtable%hash)) deallocate(hashtable%hash)
    hashtable%n = 0
    hashtable%size = 0
    hashtable%nkey = 0

    return
  end subroutine uninit_hash

  ! *
  ! * Clears hash table
  ! *
  subroutine clear_hash(hashtable)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    ! Variables
    integer i

    do i=0,hashtable%size-1
       hashtable%hash(i)%key = -1
       hashtable%hash(i)%next = -1
    enddo
    hashtable%nkey = 0

    return
  end subroutine clear_hash

  ! *
  ! * Re-initializes (optimizes) hash table
  ! *
  subroutine reinit_hash(hashtable)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    ! Variables
    integer nkey
    real(chm_real) loadfac

    ! load factor = #keys / #buckets
    loadfac = hashtable%nkey/hashtable%n
    if (hashtable%nkey > 0 .and. (loadfac > 0.67 .or. loadfac < 0.2)) then
       nkey = hashtable%nkey
       call uninit_hash(hashtable)
       call init_hash(hashtable, nkey)
    else
       call clear_hash(hashtable)
    endif

    return
  end subroutine reinit_hash

  ! *
  ! * Set a value in hash table
  ! *
  subroutine set_hash(hashtable, key, val)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: key, val
    ! Variables
    integer i, j, k
    type(hash_t), allocatable, dimension(:) :: hash_tmp
    integer new_size

    i = mod(key, hashtable%n)
    if (hashtable%hash(i)%key >= 0) then
       ! This key already has a value, add to the linked list
       ! j = last item in the linked list
       j = i
       do while (hashtable%hash(j)%next >= 0)
          j = hashtable%hash(j)%next
       enddo
       ! Look for an empty spot in the storage
       i = hashtable%start
       if (i < hashtable%size) then
          do while (i < hashtable%size .and. hashtable%hash(i)%key >= 0)
             i = i + 1
          enddo
       endif
       ! Re-allocate if needed
       if (i == hashtable%size) then
          new_size = int(i*1.5)
          ! Make a copy of the hash table
          allocate(hash_tmp(0:hashtable%size-1))
          do k=0,hashtable%size-1
             hash_tmp(k) = hashtable%hash(k)
          enddo
          ! Allocate hash table with new size
          deallocate(hashtable%hash)
          allocate(hashtable%hash(0:new_size-1))
          ! Copy back
          do k=0,hashtable%size-1
             hashtable%hash(k) = hash_tmp(k)
          enddo
          do k=hashtable%size,new_size-1
             hashtable%hash(k)%key = -1
             hashtable%hash(k)%next = -1
          enddo
          hashtable%size = new_size
       endif
       ! Add to the end of the linked list
       hashtable%hash(j)%next = i
       hashtable%start = i + 1
    endif
    ! Store key and value
    hashtable%hash(i)%key = key
    hashtable%hash(i)%val = val
    hashtable%nkey = hashtable%nkey + 1

    return
  end subroutine set_hash

  ! *
  ! * Change a value in hash table (does nothing if the key is not found)
  ! *
  subroutine change_hash(hashtable, key, val)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: key, val
    ! Variables
    integer i

    i = mod(key, hashtable%n)
    do while (i >= 0)
       if (hashtable%hash(i)%key == key) then
          hashtable%hash(i)%val = val
          return
       endif
       i = hashtable%hash(i)%next
    enddo
    
    return
  end subroutine change_hash

  ! *
  ! * Change a value in hash table by binary AND opetarion with mask
  ! * (does nothing if the key is not found)
  ! *
  subroutine and_hash(hashtable, key, mask)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: key, mask
    ! Variables
    integer i

    i = mod(key, hashtable%n)
    do while (i >= 0)
       if (hashtable%hash(i)%key == key) then
          hashtable%hash(i)%val = iand(hashtable%hash(i)%val, mask)
          return
       endif
       i = hashtable%hash(i)%next
    enddo
    
    return
  end subroutine and_hash

  ! *
  ! * Change a value in hash table by binary OR opetarion with mask
  ! * (does nothing if the key is not found)
  ! *
  subroutine or_hash(hashtable, key, mask)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: key, mask
    ! Variables
    integer i

    i = mod(key, hashtable%n)
    do while (i >= 0)
       if (hashtable%hash(i)%key == key) then
          hashtable%hash(i)%val = ior(hashtable%hash(i)%val, mask)
          return
       endif
       i = hashtable%hash(i)%next
    enddo
    
    return
  end subroutine or_hash

  ! *
  ! * Change a value in hash table by binary OR opetarion with mask,
  ! * Sets the value if the key is not found
  ! *
  subroutine set_or_hash(hashtable, key, mask)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(inout) :: hashtable
    integer, intent(in) :: key, mask
    ! Variables
    integer i

    i = mod(key, hashtable%n)
    do while (i >= 0)
       if (hashtable%hash(i)%key == key) then
          hashtable%hash(i)%val = ior(hashtable%hash(i)%val, mask)
          return
       endif
       i = hashtable%hash(i)%next
    enddo

    ! Key not found => set it
    call set_hash(hashtable, key, mask)

    return
  end subroutine set_or_hash

  ! *
  ! * Get a value from hash table, returns .false. if the value is not found
  ! * val is only set if it was found
  ! *
  logical function get_hash(hashtable, key, val)
    implicit none
    ! Input / Output
    type(hashtable_t), intent(in) :: hashtable
    integer, intent(in) :: key
    integer, intent(inout) :: val
    ! Variables
    integer i

    i = mod(key, hashtable%n)
    do while (i >= 0)
       if (hashtable%hash(i)%key == key) then
          val = hashtable%hash(i)%val
          get_hash = .true.
          return
       endif
       i = hashtable%hash(i)%next
    enddo

    get_hash = .false.
    return
  end function get_hash

end module hash
