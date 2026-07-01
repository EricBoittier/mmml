module defltsm
  use chm_kinds
  use dimens_fcm

  implicit none
  !
  !   This common file contains nonbond and hbond defaults which are read
  !   from the parameter file. They are processed in the routines GTHBCT
  !   and CTNBCT. NOTE: This is a structured common file and any changes
  !   may lead to errors or the obsolesence of old binary parameter files.
  !                               - BRB  1/85
  !-----------------------------------------------------------------------
  !   DFUSED - A flag specifying that a particular option or value has
  !            been specified in the parameter file.
  !   DEFLTS - The value of the corresponding variable or flag to be used
  !            as a default.
  !   Flags or values not present in this file may not be specified as
  !   defaults when reading a card parameter file is read.  See PARRDR
  !   for more information.  All variables here MUST also be parsed
  !   in PARRDR.
  ! 
  !
  INTEGER,PARAMETER :: MAXDEF=58 ! IF you change this, all old binary files 
  !                              ! parameter files become unreadable. - BRB
  !
  LOGICAL DFUSED(maxdef)
  INTEGER DEFLTS(maxdef)
  !
  !-----------------------------------------------------------------------
  !   NBOND
  !
  INTEGER DFNBXM
  real(chm_real) DFCTNB,DFCONB,DFCFNB,DFWMIN,DFE14F,DFEPS
  real(chm_real) DFGONB,DFGFNB
  LOGICAL DFLGRP,DFLCNS,DFLSHF,DFLVSH,DFLBYC,DFLFSW,DFLVFS,DFGEOM
  LOGICAL DFLGES,DFLGVS
  LOGICAL DUNBXM
  !
  !rjp..02-FEB-99 BYCC mods
  LOGICAL DULBCC,DFLBCC
#if KEY_IMCUBES==1
  LOGICAL DFLBYCI                                        
#endif
#if KEY_LRVDW==1
  LOGICAL DFLLRV                                         
#endif
  LOGICAL DUCTNB,DUCONB,DUCFNB,DUWMIN,DUE14F,DUEPS
  LOGICAL DULGRP,DULCNS,DULSHF,DULVSH,DULBYC,DULFSW,DULVFS,DUGEOM
  LOGICAL DUGONB,DUGFNB,DULGES,DULGVS
#if KEY_IMCUBES==1
  LOGICAL DULBYCI                                        
#endif
#if KEY_LRVDW==1
  LOGICAL DULLRV                                         
#endif
  !
  !-----------------------------------------------------------------------
  !   HBOND
  !
  real(chm_real) DFCTHB,DFCFHB,DFCOHB,DFCTHA,DFCFHA,DFCOHA
  LOGICAL DFLHBF,DFHBEX,DFBEST
  LOGICAL DUCTHB,DUCFHB,DUCOHB,DUCTHA,DUCFHA,DUCOHA
  LOGICAL DULHBF,DUHBEX,DUBEST
  !
  !  nonbond exclusion mode. (NBXMOD)
  EQUIVALENCE (DEFLTS(1),DFNBXM)
  EQUIVALENCE (DFUSED(1),DUNBXM)
  !  electostatics by groups? (LGROUP)
  EQUIVALENCE (DEFLTS(3),DFLGRP)
  EQUIVALENCE (DFUSED(3),DULGRP)
  !  constant dielectric? (LCONS)
  EQUIVALENCE (DEFLTS(5),DFLCNS)
  EQUIVALENCE (DFUSED(5),DULCNS)
  !  Shifted electrostatics? (LSHIFT)
  EQUIVALENCE (DEFLTS(7),DFLSHF)
  EQUIVALENCE (DFUSED(7),DULSHF)
  !  Gromacs shifted electrostatics? (LEGROM)
  EQUIVALENCE (DEFLTS(51),DFLGES)
  EQUIVALENCE (DFUSED(51),DULGES)
  !  Gromacs shifted VDW? (LVGROM)
  EQUIVALENCE (DEFLTS(53),DFLGVS)
  EQUIVALENCE (DFUSED(53),DULGVS)
  !  vdw shifted? (LVSHFT)
  EQUIVALENCE (DEFLTS(9),DFLVSH)
  EQUIVALENCE (DFUSED(9),DULVSH)
  !  nonbond list cutoff distance. (CUTNB)                 (11&12)
  EQUIVALENCE (DEFLTS(11),DFCTNB)                    
  EQUIVALENCE (DFUSED(11),DUCTNB)                    
  !  nonbond switching function on distance. (CTONNB)      (13&14)
  EQUIVALENCE (DEFLTS(13),DFCONB)                    
  EQUIVALENCE (DFUSED(13),DUCONB)                    
  !  nonbond switching or shifting off distance. (CTOFNB)  (15&16)
  EQUIVALENCE (DEFLTS(15),DFCFNB)                    
  EQUIVALENCE (DFUSED(15),DUCFNB)                    
  ! nonbond GROMACS elec switch on distance (CGONNB)
  EQUIVALENCE (DEFLTS(55),DFGONB)
  EQUIVALENCE (DFUSED(55),DUGONB)
  ! nonbond GROMACS elec switch off distance (CGONNB)
  EQUIVALENCE (DEFLTS(57),DFGFNB)
  EQUIVALENCE (DFUSED(57),DUGFNB)
  !  nonbond close contact warning distance. (WRNMIN)      (17&18)
  EQUIVALENCE (DEFLTS(17),DFWMIN)                    
  EQUIVALENCE (DFUSED(17),DUWMIN)                    
  !  1-4 electrostatic scaling factor. (E14FAC)            (19&20)
  EQUIVALENCE (DEFLTS(19),DFE14F)                    
  EQUIVALENCE (DFUSED(19),DUE14F)                    
  !  dielectric constant. (EPS)                            (21&22)
  EQUIVALENCE (DEFLTS(21),DFEPS)
  EQUIVALENCE (DFUSED(21),DUEPS)
  !  search by cubes? (LBYCU)
  EQUIVALENCE (DEFLTS(23),DFLBYC)
  EQUIVALENCE (DFUSED(23),DULBYC)
  !rjp..02-FEB-99 BYCC
  ! search by clusters in cubes? (LBYCC)
  EQUIVALENCE (DEFLTS(24),DFLBCC)
  EQUIVALENCE (DFUSED(24),DULBCC)
  ! Geometric combining rules for vdw (OPLS type)?
  EQUIVALENCE (DEFLTS(25),DFGEOM)
  EQUIVALENCE (DFUSED(25),DUGEOM)
  !
#if KEY_IMCUBES==1
  !  search images by cubes? (LBYCBIM)
  EQUIVALENCE (DEFLTS(48),DFLBYCI)
  EQUIVALENCE (DFUSED(48),DULBYCI)
#endif 
#if KEY_LRVDW==1
  !  Long-range vdw (LLRVDW)
  EQUIVALENCE (DEFLTS(49),DFLLRV)
  EQUIVALENCE (DFUSED(49),DULLRV)
#endif 
  !  Force bases electrostatics cutoffs? (LFSWT)
  EQUIVALENCE (DEFLTS(45),DFLFSW)
  EQUIVALENCE (DFUSED(45),DULFSW)
  !  Force based vdw cutoffs? (LVFSWT)
  EQUIVALENCE (DEFLTS(47),DFLVFS)
  EQUIVALENCE (DFUSED(47),DULVFS)
  !
  !
  !  hydrogen bond cutoff distance. (CUTHB)                (27&28)
  EQUIVALENCE (DEFLTS(27),DFCTHB)                    
  EQUIVALENCE (DFUSED(27),DUCTHB)                    
  !  hydrogen bond switching off distance. (CTOFHB)        (29&30)
  EQUIVALENCE (DEFLTS(29),DFCFHB)                    
  EQUIVALENCE (DFUSED(29),DUCFHB)                    
  !  hydrogen bond switching on distance. (CTONHB)         (31&32)
  EQUIVALENCE (DEFLTS(31),DFCOHB)                    
  EQUIVALENCE (DFUSED(31),DUCOHB)                    
  !  hydrogen bond cutoff off angle. (CUTHA)               (33&34)
  EQUIVALENCE (DEFLTS(33),DFCTHA)                    
  EQUIVALENCE (DFUSED(33),DUCTHA)                    
  !  hydrogen bond switching off angle. (CTOFHA)           (35&36)
  EQUIVALENCE (DEFLTS(35),DFCFHA)                    
  EQUIVALENCE (DFUSED(35),DUCFHA)                    
  !  hydrogen bond switching on angle. (CTONHA)            (37&38)
  EQUIVALENCE (DEFLTS(37),DFCOHA)
  EQUIVALENCE (DFUSED(37),DUCOHA)
  !  hydrogen bond antecedent used? (LHBFG)
  EQUIVALENCE (DEFLTS(39),DFLHBF)
  EQUIVALENCE (DFUSED(39),DULHBF)
  !  1-4 hbonds excluded from consideration? (HBEXCL)
  EQUIVALENCE (DEFLTS(41),DFHBEX)
  EQUIVALENCE (DFUSED(41),DUHBEX)
  !  Only allow best hbond for each donor? (BEST)
  EQUIVALENCE (DEFLTS(43),DFBEST)
  EQUIVALENCE (DFUSED(43),DUBEST)
  !
contains
  subroutine deflts_iniall()
    dfused(1:maxdef)=.false.
    deflts(1:maxdef)=0
    !  nonbond exclusion mode. (NBXMOD)
    DFNBXM=5
    !  electostatics by groups? (LGROUP)
    DFLGRP=.FALSE.
    !  constant dielectric? (LCONS)
    DFLCNS=.FALSE.
    !  shifted electrostatics? (LSHIFT)
    DFLSHF=.FALSE.
    !  vdw shifted? (LVSHFT)
    DFLVSH=.FALSE.
    !  gromacs electrostatics shift? (LEGROM)
    DFLGES=.FALSE.
    !  gromacs vdw shift? (LVGROM)
    DFLGVS=.FALSE.
    !  search by cubes? (LBYCU)
    DFLBYC=.FALSE.
    !  search by clusters in cubes?
    DFLBCC=.FALSE.
    !  use VDW geometric combining rules?
    DFGEOM=.FALSE.
    !
#if KEY_IMCUBES==1 /*cubes*/
    !  search images by cubes? (LBYCBIM)
    DFLBYCI=.FALSE.
#endif /*         (cubes)*/
#if KEY_LRVDW==1
    !  long-range vdW correction? (LLRVDW)
    DFLLRV=.FALSE.
#endif 
    !  Force bases electrostatics cutoffs? (LFSWT)
    DFLFSW=.FALSE.
    !  Force based vdw cutoffs? (LVFSWT)
    DFLVFS=.FALSE.
    !  nonbond list cutoff distance. (CUTNB)
    DFCTNB=8.0
    !  nonbond switching function on distance. (CTONNB)
    DFCONB=6.5
    !  nonbond switching or shifting off distance. (CTOFNB)
    DFCFNB=7.5
    !  GROMACS VDW switching on distance (CGONNB) 
    DFGONB=0.0
    !  GROMACS VDW switching off distance (CGOFNB)
    DFGFNB=10.0
    !  nonbond close contact warning distance. (WMIN)
    DFWMIN=1.5
    !  1-4 electrostatic scaling factor. (E14FAC)
    DFE14F=1.0
    !  dielectric constant. (EPS)
    DFEPS=1.0
    !
    !  hydrogen bond cutoff distance. (CUTHB)
    dfcthb=4.5
    !  hydrogen bond switching off distance. (CTOFHB)
    dfcfhb=4.0
    !  hydrogen bond switching on distance. (CTONHB)
    dfcohb=3.5
    !  hydrogen bond cutoff off angle. (CUTHA)
    dfctha=90.0
    !  hydrogen bond switching off angle. (CTOFHA)
    dfcfha=70.0
    !  hydrogen bond switching on angle. (CTONHA)
    dfcoha=50.0
    !  hydrogen bond antecedent used? (LHBFG)
    dflhbf=.true.
    !  1-4 hbonds excluded from consideration? (HBEXCL)
    dfhbex=.false.
    !  Only allow best hbond for each donor? (BEST)
    dfbest=.false.
    !
    return
  end subroutine deflts_iniall
end module defltsm

