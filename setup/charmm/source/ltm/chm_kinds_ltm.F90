module chm_kinds
  implicit none
  !--- From Lennart Nilsson 8/08 
  integer,parameter :: chm_real4 = selected_real_kind(6,37)
  integer,parameter :: chm_real8 = selected_real_kind(14,300)
  !      integer,parameter :: chm_int  = kind(1)

  integer,parameter :: chm_int2 = selected_int_kind(4)
  integer,parameter :: chm_int4 = selected_int_kind(9)
  integer,parameter :: chm_int8 = selected_int_kind(18) 
  integer, parameter :: int_byte = selected_int_kind(2)
#if KEY_SINGLE==1
  integer,parameter :: chm_real  = chm_real4
#else /**/
!  integer,parameter :: chm_real  = chm_real8
  integer,parameter :: chm_real  = selected_real_kind(8)
#endif 
  integer,parameter :: chm_cmpx = selected_real_kind(14,300)

  integer,save :: chm_bomlev,chm_bommin,chm_wrnlev
  integer,save :: bomlev,bommin,wrnlev

  ! GAMESS is still compiled with:
  ! -fdefault-integer-8 (gfortran),
  ! or -i8 (ifort)
  integer,parameter :: gms_int = selected_int_kind(18)

end module chm_kinds

