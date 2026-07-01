module galgor_ltm
  ! Variables moved from from galgor to break dependency cycles - mkg 2011
  ! 20-May-2011, Antti-Pekka Hynninen, added: LstAngl 
#if KEY_GENETIC==0
  logical, parameter :: qGA_ener = .false.
#else /**/
  logical :: qGA_ener
  INTEGER LstAngl, LstPhi, LstImPhi
#endif 
end module galgor_ltm

