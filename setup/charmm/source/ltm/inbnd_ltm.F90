module inbnd
  use chm_kinds
  use dimens_fcm
  implicit none
  character(len=*),private,parameter :: file_name   ="inbnd.src"
  !-----------------------------------------------------------------------
  !     Non-bonded List configuration variables
  !
  !     See chm_types_ltm for corresponding nonbondDataStructure members.
  !
  ! !!Important note!! The pointer assignments below MUST match the code
  ! in routines GETBN2 and SETBN2 (nbonds/nbutil.src). - BRB
  !
  ! THE NONBOND FLAGS:
  !
  !     LNBOPT    LOGICAL(NNBOPT)             The nonbond calculation options
  !     NNBOPT    integer                     max number of options, can be
  !                                           changed in the parameter statement
  !                                           below
  !
  !     (1) = LELEC    electrostatics activated (default .true.)
  !     (2) = LVDW     van der waals activated (default .true.)
  !     (3) = LGROUP   electrostatics by groups (default .false.)
  !     (4) = LCONS    constant dielectric       (default .false.)
  !     (5) = LSHFT    use a shifted potential for elec. (default .false.)
  !     (6) = LVATOM   to vdw energy temr by atoms
  !     (7) = LVSHFT   use a shifted potential for VDW's (default .false.)
  !     (8) = LBYCU    search by cubes instead of groups (default .false.)
  !     (9) = LFSWT    switch elec. forces, not energy (default .false.)
  !     (10)= LVFSWT   switch vdw forces, not energy   (default .false.)
  ! #if KEY_IMCUBES==1
  !     (11)= LBYCBIM  search images and primary by cubes
  ! #else /**/
  !     (11)=          unused
  ! #endif
  !     (12)= LFMA     use Fast Multipole Algorithm (default .false.)
  !--##IF MMFF
  !     (13)= LTRUNC   use truncated potential for elec. (default .false.)
  !     (14)= LVTRUNC  use truncated potential for vdw. (default .false.)
  !     (15)= LMSHFT   use modified shifted potential for elec. (default .false.)
  !--##ENDIF
  !--##IF SOFTVDW
  !     (16)= QGAMIN     use the soft core vdw and electrostatics
  !     (17)= QELEXP     ???
  !     (18)= QVDWEXP    ???
  !--##ENDIF
  !     (19)= LBYCC    search by clusters-in-cubes (default .false.)
  !     (20)= LLRVDW   use long-range VDW correction
  !--##IF NBIPS (nbips)
  !WXW Long range potential using isotropic periodic sum (IPS)
  !     (21)= LEIPS    use isotropic periodic sum for electrostatics (.false.)
  !     (22)= LVIPS    use isotropic periodic sum for VDW    (default .false.)
  !--##ENDIF (nbips)
  !     (23)= QETEN    Flag for 10-12 potential on/off
  !     (24)= LEGROM   use GROMACS shift for electrostatics (default .false)
  !                    turned on via GSHIft
  !     (25)= LVGROM   use GROMACS shift for VDW (default .false)
  !                    turned on via VGSHift
  !     (26)= QGRF     generalized reaction field -- Wei Chen 2015
  !--##IF OPENMM
  !     (27)= LOMMRXN  OpenMM reaction field logical (default .false.)
  !     (28)= LOMMSWI  OpenMM vdW switching logical (default .false.)
  !--##ENDIF
  !     (29)= QETSR    Flag for 10-12 potential on/off with switching term
  !     (30)= LDBEXP   Flag for vdw double exponential potential
  !     (31-40) = unused
  !-----------------------------------------------------------------------
  !     ATSX, ATSY, ATSZ  The position of the atoms at the last update.
  !-----------------------------------------------------------------------


  LOGICAL       LELEC,LVDW,LGROUP,LCONS,LSHFT,LVSHFT,LVATOM,LBYCU, &
       LFSWT,LVFSWT,LFMA &
       ,LTRUNC,LVTRUNC,LMSHFT &
       ,QGAMIN,QELEXP, QVDWEXP, LDBEXP

#if KEY_IMCUBES==1
  LOGICAL LBYCBIM
#endif
  LOGICAL LBYCC
#if KEY_LRVDW==1
  LOGICAL LLRVDW
#endif
  LOGICAL LLRVDW_MS
  ! LLRVDW_MS - LRC of VDW using Shirts's algorithm
#if KEY_NBIPS==1
  LOGICAL LEIPS,LVIPS
#endif
  LOGICAL LEGROM,LVGROM
  LOGICAL QETEN,QETSR
  LOGICAL QGRF ! Wei Chen 2015
  LOGICAL LOMMRXN, LOMMSWI, LOMMGB, qgbsw_omm, qgbmv_omm  !note lommgb is set in omm_cntrl
  LOGICAL qmses_omm ! AN EXAMPLE OF OPENMM PLUGIN 

  ! THE NONBOND REAL VALUES:
  !
  !     NBDIST    real(chm_real)(16)
  !
  !     (1) = CUTNB    electrostatics list cutoff (def 8.0)
  !     (2) = CTONNB   switching function turn on dist. (def 6.5)
  !     (3) = CTOFNB   swithcing function and shifting turn off (def 7.5)
  !     (4) = WRNMIN   electrostatics warning distance (def 1.5)
  !     (5) = WRNMXD   update warning step length (def 0.5)
  !     (6) = E14FAC   factor to multiply 1-4 electrostatic terms by
  !     (7) = EPS      The dielectric constant (for direct interactions)
  !     (8) =          unused
  !     (9) = NBSCAL   scale factor for guess at nonbond list size
  !     (10)= IMSCAL   scale factor for guess at image nonbond list size
  !--##IF MMFF
  !    (11) = V14FAC   factor to multiply 1-4 vdw terms by (for MMFF)
  !    (12) = DELQ     electrostatic "buffering" constant (for MMFF)
  !    (13) = GAMBUF   gamma "buffering" constant in 14-7 vdW (for MMFF)
  !    (14) = DELBUF   delta "buffering" constant in 14-7 vdW (for MMFF)
  !    (15) = CTVTRN   vdw cutoff when LVTRUNC = .true.
  !--##ENDIF
  !--##IF SOFTVDW
  !    (16)= RGAMIN     fraction of sigma which the soft core vdw and
  !                     electrostatics starts
  !    (17)=EGAMINA     Minimimum attractive energy for softvdw
  !    (18)=EGAMINR     Maximum repulsive energy for softvdw
  !--##ENDIF
  !    (19) = CGONNB   switching function turn on dist for LEGROM (def 0.0)
  !    (20) = CGOFNB   switching function turn off dist for LEGROM (def 10.0)
  !    (21) = RFCON    coefficient for GRF
  !    (22) = RFTEMP   temperature for GRF
  !    (23) = RFIONI   ionic strength for GRF
  !    (24) = RFKAP    inverse Debye screening length
  !    (25) = RFEPSO   dielectric constant outside cut-off radius
  !--##IF OPENMM
  !    (26) = omm_rxnfld_dielectric Reaction field dielectric for OpenMM generailzed reaction field (default 78.8)
  !--##ENDIF
  !    (27-40) = unused

  real(chm_real) CUTNB,CTONNB,CTOFNB,WRNMIN,WRNMXD,E14FAC,EPS, &
       NBSCAL,IMSCAL,CTVTRN &
       ,V14FAC,DELQ,GAMBUF,DELBUF &
       ,RGAMIN,EGAMINA,EGAMINR &
       ,CGONNB, CGOFNB &
       ,RFCON,RFTEMP,RFKAP,RFEPSO,RFIONI ! GRF parameters -- Wei Chen 2015
  real(chm_real) :: omm_rxnfld_dielectric

  ! THE NONBOND INTEGER VALUES:
  !
  !     NBINTS    INTEGER(10)
  !
  !     (1) = NBXMOD   The nonbond exclusion mode, see MAKINB
  !     (2) = NNNB     Total number of atom-atom interactions
  !     (3) = NNNBG    Tota number of group-group interactions
  !     (4) = NNB14    Total number of atom-atom exclusions
  !     (5) = NNG14    Total number of group-group exclusions
  !     (6-10) =          unused

  INTEGER       NNNB,NNNBG,NBXMOD,NNB14,NNG14 !

  real(chm_real),save,allocatable,dimension(:) :: ATSX, ATSY, ATSZ

contains
  subroutine inbnd_init()
    use number,only:one
    nbscal = one
    QETEN=.FALSE.
    QETSR=.FALSE.
    return
  end subroutine inbnd_init

  subroutine allocate_inbnd(natom)
    use memory
    integer,intent(in):: natom
    integer i
    character(len=*),parameter :: routine_name="allocate_inbnd"
    integer ilen

    if(allocated(atsx)) then
       if( natom > size(atsx) ) call deallocate_inbnd()
    endif
    if(.not.allocated(atsx)) then
       call chmalloc(file_name,routine_name,'atsx ',natom,crl=atsx)
       call chmalloc(file_name,routine_name,'atsy ',natom,crl=atsy)
       call chmalloc(file_name,routine_name,'atsz ',natom,crl=atsz)
       ! call chmalloc(file_name,routine_name,'e14ff',natom,crl=E14FF)
    endif
    !write(*,*)'inbnd_ltm: setting e14ff elements to ',e14fac
    !e14ff(1:natom) = e14fac
    return
  end subroutine allocate_inbnd

  subroutine deallocate_inbnd()
    use memory
    character(len=*),parameter :: routine_name="deallocate_inbnd"
    integer ilen
    if(allocated(atsx)) then
       ilen=size(atsx)
       call chmdealloc(file_name,routine_name,'atsx ',ilen,crl=atsx)
       call chmdealloc(file_name,routine_name,'atsy ',ilen,crl=atsy)
       call chmdealloc(file_name,routine_name,'atsz ',ilen,crl=atsz)
    endif

    return
  end subroutine deallocate_inbnd


  !
  INTEGER FUNCTION GETMAXNPR(NATOM,INBL)
    !
    !-----------------------------------------------------------------------
    !---     Return the maximum number of pairs from the
    !---     nonbond atom pair list.
    !
    !---     NATOM   - number of atoms
    !---     INBL    - nonbond atom pair list for index i
    !
    !----------------------------------------------------------------------
    !---     This is a helper function for allocation of the cache storage
    !---     necessary for the FAST EWald version of REWALD.
    !
    !---     February 5, 2002   Scott Brozell TSRI
    !
    !-----------------------------------------------------------------------
    !
    INTEGER NATOM
    INTEGER INBL(*)
#if KEY_DOMDEC==0
    !
    integer i
    integer itemp
    integer npr
    !
    getmaxnpr = 0
    itemp = 0
    do i = 1,natom
#if KEY_IMCUBES==1
       if(lbycbim) itemp = inbl(i + natom)
#endif
       npr = inbl(i) - itemp
       getmaxnpr = max(npr,getmaxnpr)
       itemp = inbl(i)
    enddo
#else /**/
    getmaxnpr = nnnb
#endif
    return
  end  function getmaxnpr



end module inbnd
