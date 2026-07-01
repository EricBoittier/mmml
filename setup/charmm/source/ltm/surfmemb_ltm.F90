module surfmemb
  use chm_kinds
  use dimens_fcm

! File: surfmemb.fcm
!
!     Common block for solvent free energy as function
!     of solvent accessible Surface Area in the presence
!     of an Implicit membrane            
!
!     This block is a modified version of  surface.fcm to include
!     Implicit Membrane in ASPENER calculations. 
!     See            
!     V. Spassov, L.Yan, S.Szalma (2002) J.Phys. Chem, B, 8726 - 8738.
!
!----------------------------------------------------------------------
!     memb_dir   direction of membrane normal (1 (along X) 2(Y) or 3 (Z)
!     L_memb0    membrane thickness
!     mb_center  coordinate of membrane midplane 
!----------------------------------------------------------------------
#if KEY_ASPMEMB==1 /*surfmemb_fcm*/

      real(chm_real)  L_memb0, mb_center
      integer memb_dir

!     flag whether a membrane is present.
      logical membrane

!     flag which tells usere whether to read the solvent info.
      logical SOLVMEMB_ENTRD
!
contains
  subroutine surfmemb_iniall()
    solvmemb_entrd=.false.
    return
  end subroutine surfmemb_iniall

#endif /* (surfmemb_fcm)*/
end module surfmemb

