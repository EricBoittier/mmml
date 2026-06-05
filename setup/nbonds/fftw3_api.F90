!
! Wraps "modern" Fortran interface provided by FFTW 3.3 or later.
! http://www.fftw.org/
!
module fftw3
#if KEY_FFTW==1
   use, intrinsic :: iso_c_binding
   include 'fftw3.f03'
#endif 
end module fftw3

