module qm2_elements
  use chm_kinds
  use dimens_fcm
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  use qm2_parameters,only:nelements 
#endif
#if KEY_SQUANTM==1 || KEY_QTURBO==1 || KEY_G09==1
  !------------------The element symbols     ----------------------------!
  CHARACTER (LEN = 2), DIMENSION(1:nelements), PARAMETER :: &
       ELEMENT_SYM = (/ &
       'H ','He', &
       'Li','Be','B ','C ','N ','O ','F ','Ne', &
       'Na','Mg','Al','Si','P ','S ','Cl','Ar', &
       'K ','Ca', &
       'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn', &
       'Ga','Ge','As','Se','Br','Kr', &
       'Rb','Sr', &
       'Y ','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd', &
       'In','Sn','Sb','Te','I ','Xe', &
       'Cs','Ba','La', &
       'Ce','Pr', &
       'Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu', &
       'Hf','Ta','W ','Re','Os','Ir','Pt','Au','Hg', &
       'Tl','Pb','Bi','Po','At','Rn'/)
  !------------------End of Elemental Symbols----------------------------!
#endif 
end module qm2_elements

