#if KEY_MC==1
SUBROUTINE MCUSRA(EU,X,Y,Z,DX,DY,DZ,NATOMX,IAMC,NAMC,INBLO,JNB)
  !
  !       Atom-based non-bonded list user energy routine.
  !
  !       EU   - Energy to be returned
  !       Everything else has its usual meaning.
  !
  !       Unmodified, the code just sets EU to zero.  
  !
  !       The argument list includes INBLO and JNB to show how
  !       the miniature non-bonded list is passed in the call 
  !       (see MCENER in mcener.src).  Note that INBLO is only
  !       filled for IAMC-1 to NAMC.
  !
  !       If images are also desired, the corresponding image arrays 
  !       must be added.
  !
  !       Aaron R. Dinner
  !
  use chm_kinds
  implicit none
  INTEGER NATOMX, IAMC, NAMC, INBLO(*), JNB(*)
  real(chm_real)  EU, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)

  EU = 0.0 

  RETURN
END SUBROUTINE MCUSRA

SUBROUTINE MCUSRG(EU,X,Y,Z,DX,DY,DZ,NGRPX,NATOMX, &
     INBLOG,JNBG,IBLO14,INB14)
  !
  !       Group-based non-bonded list user energy routine.
  !
  !       EU   - Energy to be returned
  !       Everything else has its usual meaning.
  !
  !       Unmodified, the code just sets EU to zero.  
  !
  !       The argument list includes INBLOG, JNBG, IBLO14, INB14 to 
  !       show how the miniature non-bonded list is passed in the call 
  !       (see MCENER in mcener.src).  
  !
  !       If images are also desired, the corresponding image arrays 
  !       must be added.
  !
  !       Aaron R. Dinner
  !
  use chm_kinds
  implicit none
  INTEGER NATOMX, NGRPX, INBLOG(*), JNBG(*)
  INTEGER IBLO14(*), INB14(*)
  real(chm_real)  EU, X(*), Y(*), Z(*), DX(*), DY(*), DZ(*)
  EU = 0.0 
  return
end SUBROUTINE MCUSRG


#endif 

SUBROUTINE NULL_MCUSER
  RETURN
END SUBROUTINE NULL_MCUSER

