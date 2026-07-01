module spacdec
  use chm_kinds
  use dimens_fcm
  use parallel
  implicit none
  !
  !  Variables    Description
  !  ---------     ---------
  !      NROUND      Number of rounds for necessary communication
  !      PARTNER     Array of partners for each cpu, by round
  !      MYATMAP     Mapping of atom numbers within own cpu     
  !      YRATMAP     Mapping of atom numbers within other cpu
  !      MYATCNT     atom counts
  !      YRATCNT
  !      MYATMX      coordinates 
  !      MYATMY
  !      MYATMZ 
  !      YRATMX
  !      YRATMY
  !      YRATMZ
  !      YOURDX      forces
  !      YOURDY
  !      YOURDZ
  !      MINEDX
  !      MINEDY
  !      MINEDZ
  !      LSPACFRST   true in first pass of bycc for spacial decomp
  !      PRTATMLST  list of atoms to be communicated to each cpu partner
  !      NPARTATM   length of the list
  !      PRTATMHI   pointer into the list   
  !      PRTATMMNY  length of each partner cpu's part of the list
  !      NMYATM     number of atoms in this cpu
  !      MYATARR    mapping array for atoms in this cpu
  !      MYSATARR     same as above, but sorted array
  !      YRMAPLST  list of atoms to be received
  !      YRMAPHI  pointer into recmaplst, by node
  !      YRMAPMNY size info of recmaplst segments, by node
  !      ISTART(MAXNODE),IEND(MAXNODE)
  !     NMYBOND,NMYANGL,NMYDIHE,NMYIMPR  number of bonds, angles, dihe,
  !        and impropers assigned to my cpu
  !     MYBOND,MYANGL,MYDIHE,MYIMPR  mapping arrays for local bonds,
  !        angles, dihedrals and impropers (i.e. assigned to my cpu)
  !      
#if KEY_SPACDEC==1 /*spacdec_fcm*/
  INTEGER PRTATMLST(MAXA*5),PRTATMHI(MAXNODE),PRTATMMNY(MAXNODE),NPARTATM
  INTEGER NROUND,PARTNER(1000),MYATMAP(MAXA),YRATMAP(MAXA),YRATCNT,MYATCNT
  INTEGER NMYATM,MYATARR(MAXA),MYSATARR(MAXA)
  INTEGER YRMAPLST(MAXA),YRMAPHI(MAXNODE),YRMAPMNY(MAXNODE)
  INTEGER NMYBOND,NMYANGL,NMYDIHE,NMYIMPR
  INTEGER MYBOND(MAXB),MYANGL(MAXT),MYDIHE(MAXP),MYIMPR(MAXIMP)
  !
  real(chm_real) MYATMX(MAXA),MYATMY(MAXA),MYATMZ(MAXA), &
       YRATMX(MAXA),YRATMY(MAXA),YRATMZ(MAXA),MINEDX(MAXA), &
       MINEDY(MAXA),MINEDZ(MAXA),YOURDX(MAXA),YOURDY(MAXA), &
       YOURDZ(MAXA)
  !
  LOGICAL LSPACFRST
  !
contains
  subroutine spacdec_iniall
    lspacfrst=.true.
    return
  end subroutine spacdec_iniall

#endif /* (spacdec_fcm)*/
  !
end module spacdec

