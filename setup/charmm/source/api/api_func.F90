module api_func
  implicit none

  private
  
  ! MLpot atom indices list
  integer, parameter :: max_Nml = 50000
  integer, parameter :: max_Npr = 3998000
  logical :: mlpot_is_init
  integer :: Nml, Nmlp, Nmlmmp, Nmlpcc
  integer, dimension(*) :: mlidx(max_Nml)
  integer, dimension(*) :: mlidxi(max_Npr), mlidxj(max_Npr)
  integer, dimension(*) :: idxp(max_Npr)
  integer, dimension(*) :: idxi(max_Npr), idxj(max_Npr), idxjp(max_Npr)
  integer, dimension(*) :: idxu(max_Npr), idxv(max_Npr)
  integer, dimension(*) :: idxup(max_Npr), idxvp(max_Npr)
  integer, dimension(*) :: idxk(max_Npr), idxkp(max_Npr)
  
  interface
     function callback(natom, &
          x_pos, y_pos, z_pos, &
          dx, dy, dz) bind(c)
       use, intrinsic :: iso_c_binding, only: c_double, c_int
       implicit none
       real(c_double) :: callback
       integer(c_int), value :: natom
       real(c_double), dimension(*) :: &
            x_pos, y_pos, z_pos, &
            dx, dy, dz
     end function callback
  
  function callback_mlpot(    &
          natom, ntrans, natim, &
          idxp,                 &
          x_pos, y_pos, z_pos,  &
          dx, dy, dz,           &
          Nmlp, Nmlmmp,         &
          idxi, idxj,           &
          idxjp,                &
          idxu, idxv,           &
          idxup, idxvp) bind(c)
       use, intrinsic :: iso_c_binding, only: c_double, c_int
       implicit none
       real(c_double) :: callback_mlpot
       integer(c_int), value :: natom, ntrans, natim, Nmlp, Nmlmmp
       integer(c_int), dimension(*) :: idxp
       real(c_double), dimension(*) :: &
            x_pos, y_pos, z_pos, &
            dx, dy, dz
        integer(c_int), dimension(*) :: idxi, idxj, idxjp
        integer(c_int), dimension(*) :: idxu, idxv, idxup, idxvp
     end function callback_mlpot
  end interface
  
  procedure(callback), pointer :: user_func
  procedure(callback_mlpot), pointer :: user_mlpot

  public :: func_set, func_call, func_is_set, func_unset,       &
            mlpot_set_func, mlpot_set_properties,               &
            mlpot_update, mlpot_call, mlpot_is_set, mlpot_unset

contains

  subroutine func_set(new_func) bind(c)
    use energym, only: qeterm, user
    implicit none
    procedure(callback) :: new_func
    user_func => new_func
    qeterm(user) = .true.
  end subroutine func_set

  subroutine func_unset() bind(c)
    use energym, only: qeterm, user
    implicit none
    user_func => null()
    qeterm(user) = .false.
  end subroutine func_unset

  subroutine func_call(out_energy, natom, &
             x_pos, y_pos, z_pos, &
             dx, dy, dz) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none
    real(c_double) :: out_energy
    integer(c_int), value :: natom
    real(c_double), dimension(*) :: x_pos, y_pos, z_pos, &
         dx, dy, dz

    out_energy = user_func(natom, x_pos, y_pos, z_pos, &
         dx, dy, dz)
  end subroutine func_call

  logical function func_is_set()
    func_is_set = associated(user_func)
  end function func_is_set
  
  
  
  ! Addition Kai Toepfer Jan 2024
  subroutine mlpot_set_func(new_mlpot) bind(c)
    use energym, only: qeterm, user
    implicit none
    procedure(callback_mlpot) :: new_mlpot
    user_mlpot => new_mlpot
    qeterm(user) = .true.
    mlpot_is_init = .false.
  end subroutine mlpot_set_func

  subroutine mlpot_unset() bind(c)
    use energym, only: qeterm, user
    implicit none
    user_mlpot => null()
    qeterm(user) = .false.
    mlpot_is_init = .false.
  end subroutine mlpot_unset

  subroutine mlpot_set_properties(new_Nml, new_mlidx, new_mlidz) bind(c)
    use, intrinsic :: iso_c_binding, only: c_int
    implicit none

    integer(c_int), intent(in) :: new_Nml
    integer(c_int), dimension(*), intent(in) :: new_mlidx, new_mlidz

    integer(c_int) :: c, i, j


    ! Set number of ML atoms and ML atom indices
    Nml = new_Nml
    if(Nml .gt. max_Nml) then
      write(104,*) "Out of space - Nml"
    endif
    mlidx(1:Nml) = new_mlidx(1:Nml)

    ! Prepare central cell - central cell ML atom pair indices
    c = 0
    do i = 1, Nml
      do j = 1, Nml - 1
        c = c + 1
        mlidxi(c) = mlidx(i)
        if(j .lt. i) then
          mlidxj(c) = mlidx(j)
        elseif(j .ge. i) then
          mlidxj(c) = mlidx(j + 1)
        endif
      enddo
    enddo
    Nmlpcc = c

  end subroutine mlpot_set_properties

  subroutine mlpot_update(          &
             Natom, Ntrans, Natim,  &
             jnb, inblo,            &
             imattr, imjnb, imblo) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    use stream, only: outu,prnlev
    implicit none
    integer(c_int), value :: natom, ntrans, natim
    integer(c_int), dimension(*) :: jnb, inblo, imattr, imjnb, imblo
    integer(c_int) :: Niml
    integer(c_int) :: c, ic, i, j, u, v, w, n

    ! Update central and image to central atom index list
    do i = 1, natom
      idxp(i) = i
    enddo
    if(natom .gt. max_Npr) then
      write(104,*) "Out of space - idxp(:Natom)"
    endif
    if(Ntrans /= 0) then
      do j = natom + 1, natim
        idxp(j) = imattr(j)
      enddo
      if(natim .gt. max_Npr) then
        write(104,*) "Out of space - idxp(:Natim)"
      endif
    endif

    ! Identify ML atoms in image cells
    c = 0
    if(Ntrans /= 0) then
      do i = 1, Nml
        do j = natom + 1, natim
          if(mlidx(i) .eq. imattr(j)) then
            c = c + 1
            idxk(c) = j
            idxkp(c) = imattr(j)
          endif
        enddo
      enddo
      Niml = c
    endif

    ! Update ML(i)-ML(j) pair indices
    c = 1
    ic = 1
    if(Ntrans /= 0) then
      do i = 1, Nml
        ! Central cell indices
        idxi(ic:ic+Nml-2) = mlidxi(c:c+Nml-2)
        idxj(ic:ic+Nml-2) = mlidxj(c:c+Nml-2)
        idxjp(ic:ic+Nml-2) = mlidxj(c:c+Nml-2)
        c = c + Nml - 1
        ic = ic + Nml - 1
        ! Image cell indices
        idxi(ic:ic+Niml-1) = mlidx(i)
        idxj(ic:ic+Niml-1) = idxk(1:Niml)
        idxjp(ic:ic+Niml-1) = idxkp(1:Niml)
        ic = ic + Niml
      enddo
      Nmlp = ic - 1
    else
      idxi(1:Nmlpcc) = mlidxi(1:Nmlpcc)
      idxj(1:Nmlpcc) = mlidxj(1:Nmlpcc)
      idxjp(1:Nmlpcc) = mlidxj(1:Nmlpcc)
      Nmlp = Nmlpcc
    endif
    if(Nmlp .gt. max_Npr) then
      write(104,*) "Out of space - Nmlp"
    endif

    ! Identify ML(u)-MM(v) pair indices
    c = 1
    ! Iterate over ML atoms
    do u = 1, Nml
      ! INBLO: Number of non-bond atom pairs
      ! JNB(INBLO(i-1):INBLO(i)): List of non-bond atom pair indices with
      !                           indices larger than current atom index i

      ! Iterate over central atoms up to ML atom index
      ! (ML atom index mlidx(u) does not appear in JNB for v larger mlidx(u))
      do v = 1, mlidx(u)
        ! Start index
        if(v /= 1) then
          ic = inblo(v-1) + 1
        else
          ic = 1
        endif
        ! Iterate over non-bond pair atoms
        do w = ic, inblo(v)
          ! If non-bond pair atom is ML atom, add to ML-MM pair list
          if(jnb(w).eq.mlidx(u)) then
            idxu(c) = mlidx(u)
            idxup(c) = idxu(c)
            idxv(c) = v
            idxvp(c) = v
            c = c + 1
          endif
        enddo
      enddo

      ! Add Remaining MM atoms to ML-MM non-bond pair index list
      if(mlidx(u) /= 1) then
        n = inblo(mlidx(u)) - inblo(mlidx(u)-1)
        if(n.gt.0) idxv(c:c+n-1) = jnb(inblo(mlidx(u)-1)+1:inblo(mlidx(u)))
      else
        n = inblo(mlidx(u))
        if(n.gt.0) idxv(c:c+n-1) = jnb(1:inblo(mlidx(u)))
      endif
      ! Update second ML-MM non-bond pair atom list and pointer lists
      idxu(c:c+n-1) = mlidx(u)
      idxup(c:c+n-1) = idxu(c:c+n-1)
      idxvp(c:c+n-1) = idxv(c:c+n-1)
      ! Increment ML-MM non-bond atom pair counter
      c = c + n

      ! Identify MM atoms in image cells with ML central atom neighbors
      if(Ntrans /= 0) then
        do v = natom + 1, natim
          ! If image cell atom is ML atom, add second central cell non-bond
          ! pair atoms to list
          if(imattr(v) .eq. mlidx(u)) then
            n = imblo(v) - imblo(v-1)
            idxu(c:c+n-1) = v
            idxup(c:c+n-1) = imattr(v)
            idxv(c:c+n-1) = imjnb(imblo(v-1)+1:imblo(v))
            idxvp(c:c+n-1) = idxv(c:c+n-1)
            c = c + n
          ! Else if image cell atom is MM atom, check if second central cell
          ! non-bond pair atoms contain ML atoms
          else
            do w = imblo(v-1)+1, imblo(v)
              if(imjnb(w) .eq. mlidx(u)) then
                idxu(c) = mlidx(u)
                idxup(c) = idxu(c)
                idxv(c) = v
                idxvp(c) = imattr(v)
                c = c + 1
              endif
            enddo
          endif
        enddo
      endif

    enddo
    Nmlmmp = c - 1

    if(Nmlmmp .gt. max_Npr) then
      write(104,*) "Out of space - Nmlmmp"
    endif

    ! Shift indices by -1 to account for python indexing
    idxp = idxp - 1
    idxi = idxi - 1
    idxj = idxj - 1
    idxjp = idxjp - 1
    idxu = idxu - 1
    idxv = idxv - 1
    idxup = idxup - 1
    idxvp = idxvp - 1

    ! Set MLpot initialization flag
    mlpot_is_init = .true.

  end subroutine mlpot_update

  subroutine mlpot_call(            &
             out_euser,             &
             natom, ntrans, natim,  &
             x_pos, y_pos, z_pos,   &
             dx, dy, dz,            &
             jnb, inblo,            &
             imattr, imjnb, imblo) bind(c)
    use, intrinsic :: iso_c_binding, only: c_double, c_int
    implicit none
    real(c_double) :: out_energy
    real(c_double) :: out_euser
    integer(c_int), value :: natom, ntrans, natim
    real(c_double), dimension(*) :: x_pos, y_pos, z_pos, &
         dx, dy, dz
    integer(c_int), dimension(*) :: jnb, inblo, imattr, imjnb, imblo

    ! Update ML-ML and MM-MM atom pair lists
    if(.not. mlpot_is_init) then
        call mlpot_update(                  &
          natom, ntrans, natim,             &
          jnb, inblo, imattr, imjnb, imblo)
    endif

    ! Call ML potential function
    out_energy = user_mlpot(  &
         natom, ntrans, natim,&
         idxp,                &
         x_pos, y_pos, z_pos, &
         dx, dy, dz,          &
         Nmlp, Nmlmmp,        &
         idxi, idxj,          &
         idxjp,               &
         idxu, idxv,          &
         idxup, idxvp)

    ! Add ML potential and (if active) ML-MM electrostatic energy
    out_euser = out_euser + out_energy

  end subroutine mlpot_call

  logical function mlpot_is_set()
    mlpot_is_set = associated(user_mlpot)
  end function mlpot_is_set
  
end module api_func
