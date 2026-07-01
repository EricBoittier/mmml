module grape
  use chm_kinds
  use dimens_fcm
  !CHARMM Element source/fcm/grape.fcm 1.1
  !-----------------------------------------------------------------------
  !     GRAPE control parameters
  !
  !     Some of these can go easily into the module (local to the module)!
  !
  !     LGRAPE  -  Energy and forces calculated on MDGRAPE processor
  !     LNOCUT  -  No cutoff GRAPE method
  !     QGRIM   -  Use Ewald in GRAPE
  !     QGRAPINI-  Initialize MD Grape board
  !     QGPUPD  -  flag for GPU nonbond update
  !     GRPVDW  -  van der Waals for images
  !     GRPELE  -  Electrostatic for images (Ewald real space)
  !     IGRAPE  -  flag to use MDGRAPE in several modes (-1 = don't use GRAPE)
  !     MDGINI  -  initialization time
  !     MDGELJ  -  nonbond energy+forces time
  !     MDGLJ   -  Lennard-Jones energy+forces time
  !     MDGRS   -  real space time
  !     MDGKS   -  k-space time
  !     MDG14   -  1-4 interactions
  !     QGPUSPLIT  flag to split parts of GPU/CPU calculations
  !
  !
#if KEY_GRAPE==1
  INTEGER*4 G2MAXTYP,G2MAXTYP2
  PARAMETER (G2MAXTYP=64,G2MAXTYP2=G2MAXTYP*G2MAXTYP)
  LOGICAL        LGRAPE,LNOCUT,QGRIM,QGRAPINI,QCNOLIST,QGPUPD,LFMM
  ! split calculations stuff
  logical :: qgpusplit
  LOGICAL        LMDVIS

  real(chm_real) GRPVDW,GRPELE,GSCALE(G2MAXTYP2),RSCALE(G2MAXTYP2), &
       FGSCALE(G2MAXTYP2),FRSCALE(G2MAXTYP2), &
       MDGINI,MDGELJ,MDGRS,MDGKS,MDGLJ,MDGEX,mdg14,CTOFNBNL

  INTEGER*4  IDIST,IGRAPE,MDVSOCK,MDVPORT
  !
contains

  subroutine grape_ltm_init()
    lfmm=.false.
    lgrape=.false.
    lnocut=.false.
    qgrapini=.false.
    qcnolist=.false.
    qgpusplit=.false.
    return
  end subroutine grape_ltm_init


#endif 
  !
end module grape

