module gbswit
    use chm_kinds, only: chm_real

    implicit none

    logical, public, save :: qgbswit, louter
    real(chm_real), public, save :: c2ofnb, c2onnb

    logical, private, save :: is_initialized
    real(chm_real), private, save :: rul3, rul12

    contains

        subroutine gbswit_setup()
            use inbnd, only: lcons, lshft, lfswt
            use number, only: zero, one, twelve
            use inbnd, only: ctofnb, ctonnb

            implicit none

            qgbswit = lcons .and. .not. lshft .and. .not. lfswt
            c2ofnb = ctofnb * ctofnb
            c2onnb = ctonnb * ctonnb
            rul3 = zero
            rul12 = zero
            if (ctofnb  >  ctonnb) then
               rul3 = one / (c2ofnb - c2onnb)**3
               rul12 = twelve * rul3
            endif
            is_initialized = .true.
        end subroutine gbswit_setup

        subroutine es_switch(r_ij, fsw, dfsw)
          use number, only: one, zero, three

          implicit none

          real(chm_real), intent(in) :: r_ij
          real(chm_real), intent(out) :: fsw
          real(chm_real), optional, intent(out) :: dfsw

          real(chm_real) r_ij_l, r_ij_u

          if(.not. is_initialized) call gbswit_setup()

          fsw = one
          if ( present(dfsw) ) dfsw = zero 

          louter = r_ij > c2onnb
          if (qgbswit .and. louter) then
             r_ij_l = c2onnb - r_ij
             r_ij_u = c2ofnb - r_ij
             fsw = r_ij_u * r_ij_u * (r_ij_u - three * r_ij_l) * rul3
             if ( present(dfsw) ) &
                dfsw = r_ij_l * r_ij_u * rul12 / fsw
          endif
        end subroutine es_switch
end module gbswit
