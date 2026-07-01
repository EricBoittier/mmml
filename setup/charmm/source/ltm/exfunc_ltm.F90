!
! References to this junk drawer should be phased out. - mkg 2009
!
module exfunc
  use chm_kinds
  !     This include file contains some EXTERNAL FUNCTION declarations.
  !
  ! INTEGER functions
  !    FIND52 - Search a series array for a series of values
  !    GETATN - Get an atom number from a description
  !    GETRES - Get a residue number from a description
  !    GETRSN - Get a residue type from a description
  !    GETSEG - Get a segment number from a description
  !    LOCDIF - difference in address space between two arrays
  !    LUNASS - Find the next free I/O unit number
  !    MATOM  - Get an atom number from a residue number and type(name)
  !    NINDX  - Find the lowest index of a number in an array
  !    SRCHWD - Search a integer array for a value and return its index
  !    SRCHWS - Search a word array for word and return its index
  ! LOGICAL functions
  !    HYDROG - Is this atom a hydrogen?
  !    INITIA - Does this atom have a valid position?
  !    LONE   - Is this atom a lonepair type (no mass)?
  !    ORDER  - order of indicies of multiple-dimension integer array
  !    ORDER5 - order of indicies of multiple integer arrays
#if KEY_CMAP==1
  !    ORDER8 - order of indices of 8 integer arrays
#endif 
  !    ORDERR - order of indicies of multiple-dimension  real(chm_real) array
  !    QHAVCRD- Do all selected atoms have defined coordinates?
  !    QXFORM - Use extended format for I/O?
  ! real(chm_real) functions
  !    ECLOCK - Elapsed time
#if KEY_ADUMB==1
  !    UMFI   - Return value of umbrella potential
#endif 
  !
  !------------------------------------------------------------------
  ! General externals
  !
  INTEGER  FIND52, LUNASS, NINDX, SRCHWD, SRCHWS
  LOGICAL  ORDER, ORDER5, ORDERR
  !
  EXTERNAL FIND52, LUNASS, NINDX, SRCHWD, SRCHWS, &
       ORDER, ORDER5, ORDERR
  !
  !------------------------------------------------------------------
  ! Feature specific externals
  !
  !
#if KEY_DIMS==1
  INTEGER DIMSORDERR
  EXTERNAL DIMSORDERR
#endif
#if KEY_CMAP==1
  LOGICAL  ORDER8
  EXTERNAL ORDER8
#endif 
  !
#if KEY_ADUMB==1
  real(chm_real)   UMFI
  EXTERNAL UMFI
#endif 
  !yw 05-Aug-2008 remove ##NEWRNG
  !      INTEGER  MULMOD
  !      EXTERNAL MULMOD
  !
#if KEY_MMFF==1
  !--MFC-- ! mmff-specific external function definitions
  !--MFC-- !     Jay Banks 24 Oct 95: added MMFF external functions
  !
  CHARACTER(len=4) AtName
  external AtName

  CHARACTER(len=8) ElementName
  external ElementName

  CHARACTER(len=10) QNAME
  external QNAME

  integer  IATTCH, IBORDR, CONN12, CONN13, CONN14
  integer  LEQUIV, LPATH
  integer  nbndx, nbnd2, nbnd3, NTERMA
  external IATTCH, IBORDR, CONN12, CONN13, CONN14
  external LEQUIV, LPATH
  external nbndx, nbnd2, nbnd3, NTERMA

  real(chm_real) ElementMass
  external ElementMass
#endif 
  !------------------------------------------------------------------
  !
end module exfunc

