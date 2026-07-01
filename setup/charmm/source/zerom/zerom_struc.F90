 module zstruc
 use ztypes

 type(CONFSET),save,target :: CSR ! initial conformer set Read in (uncompressed)
 type(CONFSET),save,target :: CSW  ! Working conformer set (CSR after processing)

 end module zstruc
