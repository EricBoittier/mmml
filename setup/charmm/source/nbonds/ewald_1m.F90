module ewald_1m
  !-- contains global variables which can cause circular dependencies
  !
  use chm_kinds
  use chm_types
  implicit none
  !   EWVIRIAL(9)  - reciprocal space contribution to pressure tensor
  !   QSETUPK      - Flag indicating that the k-space needs to be set up

  !   PKXV         - pointer to integer array KXV(MAXKV) 
  !   PKYV         - pointer to integer array KYV(MAXKV) 
  !   PKZV         - pointer to integer array KZV(MAXKV) 
  !   PKVEC        - pointer to KVEC vectors for the particular geometry
  !   MAXKV        - Allocated dimension of KVEC, KXV, KYV, KZV

  logical lewald,qsetupk
  real(chm_real) :: kappa, EWVIRIAL(9),ewvirial2(9)
  integer maxkv,erfmod,ewnpts

  real(chm_real),allocatable,dimension(:) :: PKVEC
  integer,       allocatable,dimension(:) :: PKXV,PKYV,PKZV


contains
  subroutine ewald_init
    LEWALD=.FALSE.
    QSETUPK=.FALSE.
    MAXKV=0
    ERFMOD=1
    EWNPTS=0
    return
  end subroutine ewald_init

  subroutine ewald_kv_clear
    use memory
    character(len=12) :: file="ewald_1m.src",routine="ewald_kv_clear"
    IF(MAXKV > 0) THEN
       call chmdealloc(file,routine,"pkvec",maxkv,crl=pkvec)
       call chmdealloc(file,routine,"pkxv", maxkv,intg=pkxv)
       call chmdealloc(file,routine,"pkyv", maxkv,intg=pkyv)
       call chmdealloc(file,routine,"pkzv", maxkv,intg=pkzv)
    ENDIF
    return
  end subroutine ewald_kv_clear

  subroutine ewald_kv_allocate
    use memory
    character(len=12) :: file="ewald_1m.src",routine="ewald_kv_clear"
    call chmalloc(file,routine,"pkvec",maxkv,crl=pkvec)
    call chmalloc(file,routine,"pkxv", maxkv,intg=pkxv)
    call chmalloc(file,routine,"pkyv", maxkv,intg=pkyv)
    call chmalloc(file,routine,"pkzv", maxkv,intg=pkzv)
    return
  end subroutine ewald_kv_allocate

end module ewald_1m

