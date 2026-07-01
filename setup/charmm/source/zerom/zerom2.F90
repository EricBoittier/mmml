 module zmodule2
  use chm_kinds
#if KEY_PARALLEL==1
  use parallel,only: MYNODP,NUMNOD,MYNODGP 
#endif
  use zutil,only: loc_timer
  use new_timer,only: seconds 
  implicit none
!  real(chm_real) :: etime,ctime,etime2,ctime2
  logical,private :: QVERBOSE=.false.,QCVTIME=.false.,QWRITE=.false.
!  logical,private :: QZTIME=.true.

 contains
#if KEY_ZEROM==1 /*zerom_key*/
!
! ------------------------------------------------------
! ------------------------------------------------------
!    
SUBROUTINE ZCOMBOSET(SSSLST,SSSLS2,NTAKEN,PMODE, &
     MAXPOS,POSIT,LODOFL,HIDOFL,LOCONF,HICONF, &
     MSTDOF,MSTDFV,LDSSLS,ALILST,ALIREF,NATOMX)
  !  sets up zcombo
  use chm_kinds
  use ztypes
  use exfunc
  use zdata_mod
  use stream
  use energym
  use dimens_fcm
  use coord
  use contrl
  use memory
  use zstruc,only: CSR,CSW
  use psf,only: NATOM !temporary
  implicit none
  !
  logical,allocatable,dimension(:) :: HPQTK1_hv
  integer,allocatable,dimension(:) :: HPSVFI_hv
  logical,allocatable,dimension(:) :: HPQTAK_hv
  integer,allocatable,dimension(:) :: HPCURC_hv
  integer,allocatable,dimension(:) :: HPWSBS_hv
  integer,allocatable,dimension(:) :: HPSFLGC_hv
  integer,allocatable,dimension(:) :: HPCNFLG_hv
  integer,allocatable,dimension(:) :: HPINITR_hv
  integer,allocatable,dimension(:) :: HPINITF_hv
  INTEGER,dimension(:) :: SSSLST
  INTEGER NTAKEN,PMODE,NATOMX
  INTEGER,dimension(:) :: MAXPOS,POSIT
  INTEGER,dimension(:) :: SSSLS2
  INTEGER,dimension(:),target :: HIDOFL,LODOFL
  INTEGER,dimension(:),target :: LOCONF,HICONF
  INTEGER,dimension(:),target :: MSTDOF
  INTEGER,dimension(:) :: LDSSLS,ALILST,ALIREF
  real(chm_real),dimension(:),target ::  MSTDFV
  ! local variables
  INTEGER :: NCSAVE
  integer :: I
!  INTEGER ALLPNTS,NSLEFT
  INTEGER NSLEFT
  integer(chm_int8) :: ALLPNTS8
  INTEGER J,CBSLEN
  ! REAL
  real(chm_real) RGRDPT,FACT
! for call to ZCOMBO
  integer,pointer,dimension(:) :: LOCONF_ ,HICONF_,HIDOFL_,LODOFL_,MSTDOF_
  real(chm_real),pointer,dimension(:) :: MSTDFV_
!  real(chm_real) :: etime,ctime
  !
#if KEY_PARALLEL==1 
  if(MYNODP.eq.1) then  
#endif
   WRITE(6,'(A)') ' '
   WRITE(6,'(A42)') &
       'A SET OF GRID SEARCHES HAS BEEN REQUESTED'
#if KEY_PARALLEL==1
  endif
#endif

   if(QZTIME) call loc_timer('START of ZCOMBOSET')
  !
  IF(PMODE.EQ.3) THEN
     IF(NSSSLD.GT.NSUBME) CALL WRNDIE(-5,'<ZCOMBOSET>', &
          'NUMB OF SUBSPs and ALIASs > ALLOTTED MEMORY (ZMEM NSUB)')
  ENDIF
  !      DO I = 1,NSSSLD
  !       WRITE(6,*) I,'th loaded subspace ',LDSSLS(I)
  !       WRITE(6,*) 'CORRESPING ALIAS IS ',ALIREF(LDSSLS(I))
  !      ENDDO
  DO I = 1,NSSSLD
     !
     IF((PMODE.EQ.1).OR.(PMODE.EQ.3)) THEN
        IF(.NOT.QSUBNUL) THEN
           IF((LDSSLS(I).GT.NSUBSP).OR.(LDSSLS(I).LE.0)) THEN
              WRITE(OUTU,'(2X,A31,I8)')  &
                   'NO CONFORMERS READ FOR SUBSPACE',LDSSLS(I)
              CALL WRNDIE(-5,'<ZCOMBOSET>',' EMPTY SUBSPACE ')
           ENDIF
        ENDIF !if not qsubnul
     ENDIF
     SSSLST(I) = LDSSLS(I)
     !       WRITE(6,*) 'SETTING UP SUBSPACE # ',I,' = ',LDSSLS(I)
     IF ((PMODE.EQ.2).OR.(PMODE.EQ.3)) THEN
        IF((ALIREF(LDSSLS(I)).GT.NSUBSP).OR. &
             (ALIREF(LDSSLS(I)).LE.0)) THEN
           WRITE(OUTU,'(2X,A31,1X,I8,A13,I8)')  &
                'NO CONFORMERS READ FOR SUBSPACE', &
                ALIREF(LDSSLS(I)),' ALIAS OF SS ',LDSSLS(I)
           CALL WRNDIE(-5,'<ZCOMBOSET>',' EMPTY SUBSPACE ')
        ENDIF
     ENDIF
     SSSLS2(I) = ALIREF(LDSSLS(I))
  ENDDO
  !     
  ! calculated number of combinations or grids
  ZNCOMB = NSSSLD
  NSLEFT = NSSSLD - NTAKEN

  J = NSSSLD - 1
  DO WHILE (J.GT.NSLEFT)
     ZNCOMB = ZNCOMB*J
     J = J - 1
  ENDDO
  FACT = 1
  DO J = 1,NTAKEN
     FACT = FACT*J
  ENDDO
  RGRDPT = ZNCOMB
  ZNCOMB = INT(RGRDPT/FACT)   ! must be an integer
  !      WRITE(6,*)'NUMBER SUBSPACE COMBINATIONS = ',ZNCOMB

  IF((PMODE.EQ.1).OR.(PMODE.EQ.2)) THEN 
     CBSLEN = ZNCOMB*NTAKEN
  ELSE !PMODE 3
     CBSLEN = ZNCOMB*NSSSLD
  ENDIF

  ALLPNTS8 = 0
  call chmalloc('zerom2.src','ZCOMBOSET','HPQTK1_hv',NSSSLD,log=HPQTK1_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPZNCB_hv',ZNCOMB,ci8=HPZNCB_hv) !changed to ci8
  call chmalloc('zerom2.src','ZCOMBOSET','HPLCSS_hv',ZNCOMB,intg=HPLCSS_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPHCSS_hv',ZNCOMB,intg=HPHCSS_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPCBSS_hv',CBSLEN,intg=HPCBSS_hv)
  HPQTK1_hv = .false.
  HPZNCB_hv = 0
  HPLCSS_hv = 0
  HPHCSS_hv = 0
  HPCBSS_hv = 0

#if KEY_PARALLEL==0
  if(QZRANDM) then  
#endif
   LOCONF_ => CSW%LOCNF !LOCONFW
   HICONF_ => CSW%HICNF !HICONFW
   LODOFL_ => CSW%LODOF !LODOFLW
   HIDOFL_ => CSW%HIDOF !HIDOFLW
   MSTDOF_ => CSW%MSTDF !MSTDOFW
   MSTDFV_ => CSW%MSTDV !MSTDFVW
#if KEY_PARALLEL==0
  else
   LOCONF_ => LOCONF
   HICONF_ => HICONF
   LODOFL_ => LODOFL
   HIDOFL_ => HIDOFL
   MSTDOF_ => MSTDOF
   MSTDFV_ => MSTDFV
  endif
#endif 

  call TOTALGRID(SSSLST,SSSLS2,NTAKEN,LOCONF_,HICONF_,MAXPOS,POSIT,PMODE, &
       ALLPNTS8,HPQTK1_hv,HPZNCB_hv,HPLCSS_hv,HPHCSS_hv,HPCBSS_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPQTK1_hv',NSSSLD,log=HPQTK1_hv)
  !
  IF (QRECUT) THEN
#if KEY_PARALLEL==1
     if(MYNODP.eq.1) then 
#endif
     WRITE(6,*) ' '
     WRITE(6,'(2X,A24,F8.4,A10)')  &
          'ALLOTTING SPACE TO SAVE ', &
          ZSVFRC*100,' % OF ALL '
     WRITE(6,'(I18,A21)') ALLPNTS8,' CONFORMERS IN SEARCH'
#if KEY_PARALLEL==1
     endif 
#endif
     NCSAVE = INT(ALLPNTS8*ZSVFRC)
  ELSE
     NCSAVE = 0
  ENDIF
  !
  !       WRITE(6,*) 'CALLING WRITEIMINOR 1'
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
   WRITE(6,'(4x,A)') 'SEARCH PARAMETERS: '
   if(ZTMINO.EQ.0) THEN
     WRITE(6,'(4x,A18,I10)') 'GRIDS ',ZNCOMB
   endif
   WRITE(6,'(4x,A18,I10)') 'SUBSPACES ',NSSSLD
   if (PMODE.EQ.3) THEN 
     WRITE(6,'(4x,A18,I10)') 'SS/GRID ',NSSSLD 
   else 
     WRITE(6,'(4x,A18,I10)') 'SS/GRID ',NTAKEN
   endif
   if(ZTMINO.EQ.0) THEN
     WRITE(6,'(4x,A18,I18)') 'TOTAL GRID POINTS ',ALLPNTS8 
   else
     WRITE(6,'(4x,A18,I10)') 'MINOR SS/GRID ',ZTMINO
   endif
   WRITE(6,'(A)') ' '
#if KEY_PARALLEL==1
  endif 
#endif
  !
  call chmalloc('zerom2.src','ZCOMBOSET','HPSVFI_hv',NCSAVE,intg=HPSVFI_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPQTAK_hv',NSSSLD,log=HPQTAK_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPCURC_hv',NSSSLD*2,intg=HPCURC_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPWSBS_hv',NSSSLD*2,intg=HPWSBS_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPSFLGC_hv',NSSSLD,intg=HPSFLGC_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPCNFLG_hv',NCONFO,intg=HPCNFLG_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPINITR_hv',TINITA,intg=HPINITR_hv)
  call chmalloc('zerom2.src','ZCOMBOSET','HPINITF_hv',TINITA,intg=HPINITF_hv)
  HPSVFI_hv = 0
  HPQTAK_hv = .false.
  HPCURC_hv = 0
  HPWSBS_hv = 0
  HPSFLGC_hv = 0
  HPCNFLG_hv = 0
  HPINITR_hv = 0
  HPINITF_hv = 0
  !      
  !       WRITE(6,*) 'CALLING WRITEIMINOR 2'
  call WRITEIMINOR(HPIMISS_hv)

  call ZCOMBO(SSSLST,SSSLS2,NTAKEN,LOCONF_,HICONF_,LODOFL_,HIDOFL_,MSTDOF_, &
       MSTDFV_,MAXPOS,POSIT,PMODE,HPLOIE_hv,HPHIIE_hv,HPINVL_hv,HPSIAB_hv, &
       HPSIAE_hv,HPINIL_hv,HPSVFI_hv,HPWSBS_hv,HPCURC_hv,HPQTAK_hv,NCSAVE, &
       HPFLCI_hv,HPSATB_hv,HPSATE_hv,HPSALS_hv,HPIMISS_hv,HPALIA_hv, &
       HPSFLGC_hv,HPCNFLG_hv,HPINITR_hv,HPINITF_hv,HPQICBF_hv,INBFRQ)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPSVFI_hv',NCSAVE,intg=HPSVFI_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPQTAK_hv',NSSSLD,log=HPQTAK_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPCURC_hv',NSSSLD*2,intg=HPCURC_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPWSBS_hv',NSSSLD*2,intg=HPWSBS_hv)
  !
  IF((QRECUT).AND.(QZBINS)) &
       call ZBINSWRI(HPZBNS_hv,ALLPNTS8)
  !
  call chmdealloc('zerom2.src','ZCOMBOSET','HPZNCB_hv',ZNCOMB,ci8=HPZNCB_hv) !changed to int8
  call chmdealloc('zerom2.src','ZCOMBOSET','HPLCSS_hv',ZNCOMB,intg=HPLCSS_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPHCSS_hv',ZNCOMB,intg=HPHCSS_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPCBSS_hv',CBSLEN,intg=HPCBSS_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPSFLGC_hv',NSSSLD,intg=HPSFLGC_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPCNFLG_hv',NCONFO,intg=HPCNFLG_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPINITR_hv',TINITA,intg=HPINITR_hv)
  call chmdealloc('zerom2.src','ZCOMBOSET','HPINITF_hv',TINITA,intg=HPINITF_hv)

  if(QZTIME) call loc_timer('END of ZCOMBOSET')

  RETURN
END SUBROUTINE ZCOMBOSET
! 
SUBROUTINE ZCOMBO(SSSLST,SSSLS2, &
     NTAKEN,LOCONF,HICONF,LODOFL,HIDOFL,MSTDOF, &
     MSTDFV,MAXPOS,POSIT,PMODE,LOICEN, &
     HIICEN,INIVAL,SIABEG,SIAEND,INIATL, &
     SVCNFI,WSUBSP,CURCNF,QTAKEN,NCSAVE, &
     FLACTI,SATBEG,SATEND,SUBATL,IMINORSS,ALILST, &
     SSFLGCNT,CNFFLAG,INITRAR,INITFAR,ZQICBF,INBFRQX)

 ! *************************************************************
 ! find the combinations of NSSSLD subspaces taken NTAKEN
 ! at a time.
  !
  use chm_kinds
  use zdata_mod
  use exfunc
  use bases_fcm
  use dimens_fcm
  use psf
  use energym
  use coord
  use number
  use stream
  use intcor_module
  use memory
  use eintern_fast,only: MKBACTIVE
  use intcor2,only: MK_ICACTV
  use actclus_mod,only: ICACTVF,NICACTVF,ICACTVR,NICACTVR
#if KEY_PARALLEL==1
  use nbndcc_utilb,only: parstoperr 
#endif 
  implicit none

  !
  !  ZNCOMB  Total number of combinations
  !  NTAKEN   number of slots (how many ss taken at a time)
  !  NSSSLD  number of subspaces selected (passed)
  !
  ! local variablesi passed for memory allocation
  !  LOPDFL,HIPDFL lo and hi dof lines for a given conformer
  !  PACDOF,PACVAL  dofs and values packaged for passing to zmerge
  !  LOPCNF,HIPCNF lo and hi confs for a subspace to be passed
  ! local variables:
  !  SLOT      possible position of a subspace in combinatorics
  !  MAXPOS    highest # subspace a given slot can take
  !  POSIT     current position of slot
  !  NSLEFT dummy storage
  !  NPSBSP  counter of subspaces to be passed
  !  WSUBSP  dummy storage for subspaces
  !  LPCCNT, LPLINE !local conformer and line counts
  !  CURCNF  !memory pass to zmerge
  !  MINCNT !count of minor subspaces taken
  ! RMSMN1 !minimum rmsd over all grids
  !
  integer,allocatable,dimension(:) :: HPZATMN_hv
  real(chm_real),allocatable,dimension(:) :: HPZXCPS_hv
  real(chm_real),allocatable,dimension(:) :: HPZYCPS_hv
  real(chm_real),allocatable,dimension(:) :: HPZZCPS_hv
  real(chm_real),allocatable,dimension(:) :: HPZSTMX_hv
  real(chm_real),allocatable,dimension(:) :: HPZSTMY_hv
  real(chm_real),allocatable,dimension(:) :: HPZSTMZ_hv
  integer,allocatable,dimension(:) :: HPTEMPF_hv
  integer,allocatable,dimension(:) :: HPTEMPR_hv
  real(chm_real),allocatable,dimension(:) :: HPSAVX_hv
  real(chm_real),allocatable,dimension(:) :: HPSAVY_hv
  real(chm_real),allocatable,dimension(:) :: HPSAVZ_hv
  integer,allocatable,dimension(:) :: HPINIR_hv
  integer,allocatable,dimension(:) :: HPINIF_hv
  integer,allocatable,dimension(:) :: WRKINIT1, WRKINIT2  !work arrays
!  INTEGER SSFLGCNT(*),CNFFLAG(*),INITRAR(*),INITFAR(*)
  INTEGER,dimension(:) :: SSFLGCNT,CNFFLAG
  integer,dimension(:) :: INITRAR,INITFAR
  INTEGER,dimension(:) :: ZQICBF
  INTEGER,dimension(:) :: SSSLST,SSSLS2
  integer :: NTAKEN
  INTEGER,dimension(:) :: MAXPOS,POSIT !dimensioned by subspaces
  INTEGER,dimension(:) :: LOCONF,HICONF,LODOFL,HIDOFL
  INTEGER,dimension(:) :: SIABEG,SIAEND,INIATL
  INTEGER,dimension(:) :: MSTDOF
  integer :: PMODE
  INTEGER,dimension(:) :: LOICEN,HIICEN
!  integer,dimension(:) :: LOICEN,HIICEN
  real(chm_real),dimension(:) :: MSTDFV,INIVAL
!  INTEGER SVCNFI(*),FLACTI(*)
  INTEGER,dimension(:) :: SVCNFI
  INTEGER,dimension(:) :: FLACTI
  INTEGER,dimension(:) :: WSUBSP
  INTEGER,dimension(:) :: CURCNF
  INTEGER,dimension(:) :: SATBEG,SATEND,SUBATL
  LOGICAL,dimension(:) :: QTAKEN
  INTEGER,dimension(:) :: IMINORSS,ALILST
  INTEGER :: INBFRQX,NCSAVE
  ! local
  INTEGER J,FACT,I,II
  LOGICAL DONECOMB   !flag is true when a combination is finished
  INTEGER SLOT,NSLEFT
  real(chm_real) RGRDPT !real dummy storage
  INTEGER NPSBSP,NEWCONF
  INTEGER LPCCNT,LPLINE 
  INTEGER OKCCNT
  INTEGER TOTPNT,SUBSP,SSSCNT
  real(chm_real) MINIME
  INTEGER III,JJ,JJJ,ENTRYN,MSDLLN,SAVEGR,MINCNT,KK,SUBSP2
  real(chm_real) CURVAL,ZVACUT,RMSMN1
  INTEGER :: ZMINPT
  LOGICAL QWFIRS,LMATCH
  INTEGER ZNQVCUT, ZNQECUT, ZNQOBND, ZNMEET !counters
  INTEGER ZNQOBDF
  INTEGER NINITF,NINITR
! for random conformer selection
!  integer,allocatable,dimension(:) :: HICONFR,LOCONFR,LODOFLR,HIDOFLR,MSTDOFR
!  real(chm_real),allocatable,dimension(:) :: MSTDFVR 
  !
  !    reset new conformer counter

  NEWCONF = 0 
  ZMINPT = 0
  TOTPNT = 0
  OKCCNT = 0
  QWFIRS = .FALSE.
  ! set counters to zero
  ZNQVCUT = 0
  ZNQECUT = 0
  ZNQOBND = 0
  ZNQOBDF = 0
  ZNMEET = 0
  !
  !      MINIME = 9.9E14
  MINIME = 1.0D99
  ZVACUT = 1.0D99
  IF (QRECUT) THEN
     DO I =1,NCSAVE
        SVCNFI(I) = 0
     ENDDO
  ENDIF
  RMSMN1 = 1.0D99  !minimum rmsd 
  ! ******************************************************************
  ! ******************************************************************
  DO I = 1,NATOM
     FLACTI(I) = 0
     IF(QZCFIX) IMOVE(I)=1  !all fixed initially if qzcfix
  ENDDO
  DO SSSCNT = 1, NSSSLD  !activate atoms in all subspaces
     ! reset conformer counters
     !       SUBSP = SSSLST(I)
     !       SSFLGCNT(SUBSP) = 0
     !       DO JJ = LOCONF(SUBSP),HICONF(SUBSP)
     !        CNFFLAG(JJ) = 0
     !       ENDDO
     SUBSP = SSSLST(SSSCNT)
     !        WRITE(6,*) 'SSSCNT ',SSSCNT,' SUBSP ',SUBSP
     ! flag subs atoms to be activated
     DO JJ = SATBEG(SUBSP),SATEND(SUBSP)
        !          WRITE(6,*) 'SUBSP ',SUBSP,' ACTIVATING ',SUBATL(JJ)
        FLACTI(SUBATL(JJ)) = 1
     ENDDO
     !C  if necessary, determine which atoms are to be fixed
     IF (QZCFIX) THEN !if fixing atoms not moving in any grid
        DO JJ = SIABEG(SUBSP),SIAEND(SUBSP)
           ! flag the atoms that are moving as IMOVE(JJ) = 0
           IMOVE(INIATL(JJ)) = 0  !initialized atoms move
        ENDDO !loop over atoms in subspace SUBSP
     ENDIF
  ENDDO
  !   activate substructure atoms
  CALL ZNBACTV(FLACTI) !non-bonded
  CALL MKBACTIVE(FLACTI)  !bonded
  !   
  ! calculate the maximum number of positions and the starting
  ! positions for each of the NTAKEN subspace slots.
  DO SLOT = 1,NTAKEN
     MAXPOS(SLOT) = NSSSLD - SLOT + 1  !maximum position
     POSIT(SLOT) = NTAKEN - SLOT + 1  !positional marker
     !        WRITE(6,*)'SLOT ',SLOT,' MX ',MAXPOS(SLOT),' POS ',
     !     & POSIT(SLOT)
  ENDDO
  !
  !       WRITE(6,*) 'QRECUT ',QRECUT,' NXTIND ',NXTIND
  !       WRITE(6,*) 'NEXTOK ',NEXTOK,' OKCCNT ',OKCCNT
  !       WRITE(6,*) 'MINIME ',MINIME
  ! -----------------------------------------------------------------
  !  loop over subspace combinations
  !      WRITE(6,*) 'ZNCOMB is ',ZNCOMB
! allocate space for initialization work arrays
  if(allocated(WRKINIT1)) call chmdealloc('zerom2.src','ZCOMBO','WRKINIT1',NATOM,intg=WRKINIT1)
  call chmalloc('zerom2.src','ZCOMBO','WRKINIT1',NATOM,intg=WRKINIT1)
  if(allocated(WRKINIT2)) call chmdealloc('zerom2.src','ZCOMBO','WRKINIT2',NATOM,intg=WRKINIT2)
  call chmalloc('zerom2.src','ZCOMBO','WRKINIT2',NATOM,intg=WRKINIT2)
 
  DO I = 1,ZNCOMB
     IF (QWZCMP) QWFIRS = .TRUE.  !if compressing, will write out first conf
     ! 
     DO II=1,NSSSLD
        QTAKEN(II)=.FALSE.
     ENDDO
     ! Note that at I = 1, the NTAKEN slots have initial positions
     ! and therefore the NTAKEN subspaces have already been specified
     ! at this point in loop
     !        WRITE(6,*) 'PMODE IS ',PMODE
     NPSBSP = 0
     J = NTAKEN
     DO WHILE(J.GE.1) !loop over the NTAKEN slots
        NPSBSP = NPSBSP + 1
        IF((PMODE.EQ.1).OR.(PMODE.EQ.3)) THEN
           WSUBSP(NPSBSP) = SSSLST(POSIT(J)) !selected subspace from list1
        ELSE IF(PMODE.EQ.2) THEN
           WSUBSP(NPSBSP) = SSSLS2(POSIT(J)) !selected subspace from list2
        ELSE
           WRITE(6,*) 'WARNING: BAD PMODE '
        ENDIF
        QTAKEN(POSIT(J))=.TRUE.
        J = J - 1
     ENDDO  !(Loop over NTAKEN slots)
     IF(PMODE.EQ.3) THEN
        DO J = 1,NSSSLD
           IF(.NOT.(QTAKEN(J))) THEN
              NPSBSP = NPSBSP + 1
              WSUBSP(NPSBSP) = SSSLS2(J)
           ENDIF
        ENDDO
     ENDIF
     !        DO II = 1,NSSSLD
     !          WRITE(6,*) 'SUBSP# ',II,'SUBSP ',SSSLST(II),
     !     & 'QTAKEN ',QTAKEN(II)
     !        ENDDO
     ! ********************************************************************
     ! filter to determine whether we have the right number of minor and 
     !  major subspaces present
     !        WRITE(6,*) 'STARTING FILTER'
     MINCNT = 0
     DO II = 1,NPSBSP
        SUBSP = WSUBSP(II)
        !          WRITE(6,*) 'TAKING SUBSPACE ',SUBSP, ' IMINOR is ',
        !     & IMINORSS(SUBSP)
        IF(IMINORSS(SUBSP).EQ.1) THEN
           MINCNT = MINCNT + 1
        ENDIF
     ENDDO
     !        WRITE(6,*) 'MINOR SUBSPACES TAKEN = ',MINCNT,
     !     & ' ZTMINO ',ZTMINO
     IF(MINCNT.EQ.ZTMINO) THEN
        ! now filter to determine, if requested, whether any of the subspaces
        ! are effectively the same (redundant) 
        LMATCH = .FALSE.
        IF(.NOT.QREDUSS) THEN  !if not allowing redundancy
           JJ = 1
           DO WHILE ((JJ.LE.NPSBSP).AND.(.NOT.LMATCH))
              SUBSP = WSUBSP(JJ)
              DO KK = JJ + 1,NPSBSP
                 SUBSP2 = WSUBSP(KK)
                 !            WRITE(6,*) 'SUBSP ',SUBSP,' ALIAS ',ALILST(SUBSP)
                 !            WRITE(6,*) 'SUBSP2 ',SUBSP2,' ALIAS ',ALILST(SUBSP2)
                 IF(ALILST(SUBSP).EQ.SUBSP2) LMATCH = .TRUE. 
                 IF(ALILST(SUBSP2).EQ.SUBSP) LMATCH = .TRUE.
                 IF((ALILST(SUBSP).EQ.ALILST(SUBSP2)).AND. &
                      (ALILST(SUBSP).NE.0)) LMATCH = .TRUE.
                 GOTO 5674
                 IF(ALILST(SUBSP).EQ.SUBSP2) THEN
                    WRITE(6,*) 'MATCHES SUBSP ',SUBSP,' AND SUBSP ',SUBSP2
                 ENDIF
                 IF(ALILST(SUBSP2).EQ.SUBSP) THEN
                    WRITE(6,*) 'MATCHES SUBSP ',SUBSP,' AND SUBSP ',SUBSP2
                 ENDIF
                 IF((ALILST(SUBSP).EQ.ALILST(SUBSP2)).AND. &
                      (ALILST(SUBSP).NE.0)) THEN
                    WRITE(6,*) 'MATCHES SUBSP ',SUBSP,' AND SUBSP ',SUBSP2
                    WRITE(6,*) 'ALILST(SUBSP) = ',ALILST(SUBSP), &
                         'ALILST(SUBSP2) = ',ALILST(SUBSP2)
                 ENDIF
5674             CONTINUE
              ENDDO
              JJ = JJ + 1
           ENDDO
        ENDIF !if not qreduss
        !        WRITE(6,*) 'LMATCH is ',LMATCH
        IF(.NOT.LMATCH) THEN
           ! *********************************************************************
           ! For the subspaces not being "taken", reset to initial values,
           ! if pmode is not 3
           ! 
           IF((PMODE.EQ.1).OR.(PMODE.EQ.2)) THEN
              WRKINIT1 = 0
              WRKINIT2 = 0
              DO II = 1,NSSSLD
                 SUBSP = SSSLST(II)
                 ! change dof values in ic table
                 IF(.NOT.QTAKEN(II)) THEN
                    JJ = LOCONF(SUBSP)
                    !            WRITE(6,*) 'SUBSP ',SUBSP,' LOCONF ',LOCONF(SUBSP),
                    !     & ' LODOFL ',LODOFL(JJ),' HIDOFL ',HIDOFL(JJ)
                    DO III = LODOFL(JJ),HIDOFL(JJ)
                       JJJ = MSTDOF(III)
                       CURVAL = INIVAL(JJJ)
                       DO ENTRYN  = LOICEN(JJJ),HIICEN(JJJ)
                          !
                          CALL CHNGEICVL(icr_struct%B1ic, &
                               icr_struct%B2ic,icr_struct%T1ic, &
                               icr_struct%T2ic,icr_struct%PIC, &
                               HPLNLS_hv,HPCLST_hv, &
                               ENTRYN,CURVAL)
                       ENDDO  !loop over ic entries
                    ENDDO !loop over lines
!                 ENDIF !if subspace not taken
                 ENDIF
!at beginning of each grid, want to reset the entire structure to the initial values
!(inialize all nsssld selected subspaces, whether on this particular grid or not
                 ! initialize atom positions
                 !           WRITE(6,*) 'SUBSP is ',SUBSP,' SIABEG ',
                 !     & SIABEG(SUBSP),' SIAEND ',SIAEND(SUBSP)
                 ! store atom positions to be initialized
                  DO JJ = SIABEG(SUBSP),SIAEND(SUBSP)
                    ! store the atoms for the reverse ic build
                    IF(ZQICBF(SUBSP).EQ.0) THEN
                       WRKINIT1(INIATL(JJ)) = 1
                       ! store the atoms for the forward ic build
                    ELSE IF (ZQICBF(SUBSP).EQ.1) THEN
!                      if(MYNODP.eq.1) then
!                      write(6,*) 'original atoms to be initialized ',INIATL(JJ)
!                     endif
                       WRKINIT2(INIATL(JJ)) = 1
                    ENDIF
                  ENDDO
              ENDDO !loop over selected subspaces

              NINITR = 0
              NINITF = 0
              do II = 1,NATOM 
                if(WRKINIT1(II).eq.1) then
                  NINITR = NINITR + 1
                  INITRAR(NINITR) = II
                endif
                if(WRKINIT2(II).eq.1) then
                  NINITF = NINITF + 1
                  INITFAR(NINITF) = II
                endif
              enddo

!temporary
#if KEY_PARALLEL==1
            if(MYNODP.eq.1) then
#endif
!              DO II = 1,NINITF
!                WRITE(6,*) 'index ',II,' initializing atom ',INITFAR(II)
!              ENDDO
#if KEY_PARALLEL==1
            endif
#endif

              if(QVERBOSE) then
#if KEY_PARALLEL==1
               if(MYNODP.eq.1) then
#endif
                write(6,*) 'NINITF in zcombo is ',NINITF,' NINITR in zcombo is ',NINITR  
#if KEY_PARALLEL==1
               endif
#endif
              endif
              call MK_ICACTV(1,icr_struct%lenic,X,Y,Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               NATOM,INITFAR,NINITF,ICACTVF,NICACTVF)

               call MK_ICACTV(1,icr_struct%lenic,X,Y,Z, &
               icr_struct%B1ic,icr_struct%B2ic, &
               icr_struct%T1ic,icr_struct%T2ic, &
               icr_struct%PIC, icr_struct%IAR, &
               icr_struct%JAR, icr_struct%KAR, &
               icr_struct%LAR, icr_struct%TAR, &
               NATOM,INITRAR,NINITR,ICACTVR,NICACTVR)
              ! build from ic table, forward and reverse as needed
              CALL BILDCFR(NINITF,NINITR,INITFAR,INITRAR,NATOM)
           ENDIF !if not pmode 3
           !
           call chmalloc('zerom2.src','ZCOMBO','HPZATMN_hv',2*NATOM,intg=HPZATMN_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZXCPS_hv',NATOM,crl=HPZXCPS_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZYCPS_hv',NATOM,crl=HPZYCPS_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZZCPS_hv',NATOM,crl=HPZZCPS_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZSTMX_hv',NATOM,crl=HPZSTMX_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZSTMY_hv',NATOM,crl=HPZSTMY_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPZSTMZ_hv',NATOM,crl=HPZSTMZ_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPTEMPF_hv',NSATME,intg=HPTEMPF_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPTEMPR_hv',NSATME,intg=HPTEMPR_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPSAVX_hv',NATOM,crl=HPSAVX_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPSAVY_hv',NATOM,crl=HPSAVY_hv)
           call chmalloc('zerom2.src','ZCOMBO','HPSAVZ_hv',NATOM,crl=HPSAVZ_hv)
           HPZATMN_hv = 0
           HPZXCPS_hv = 0
           HPZYCPS_hv = 0
           HPZZCPS_hv = 0
           HPZSTMX_hv = 0
           HPZSTMY_hv = 0
           HPZSTMZ_hv = 0
           HPTEMPF_hv = 0
           HPTEMPR_hv = 0
           HPSAVX_hv = 0
           HPSAVY_hv = 0
           HPSAVZ_hv = 0

           !        WRITE(6,*) 'HPDISGAR ',HPDISGAR
           !        WRITE(6,*) 'HPDISLAR ',HPDISLAR
           call ZMERGE(WSUBSP,NPSBSP,LOCONF,HICONF,LODOFL,HIDOFL,MSTDOF,MSTDFV, &
                CURCNF,NEWCONF,HPLOIE_hv,HPHIIE_hv,HPSATB_hv,HPSATE_hv,HPSALS_hv, &
                HPSIAB_hv,HPSIAE_hv,HPINIL_hv,HPFLIN_hv,HPDOIN_hv,HPFLINR_hv, &
                HPDOINR_hv,HPGMNV_hv,SVCNFI,OKCCNT,MINIME,TOTPNT,NCSAVE,SSSLST, &
                ZMINPT,ZVACUT,QWFIRS,ALILST,HPZXCPS_hv,HPZYCPS_hv,HPZZCPS_hv, &
                HPZSTMX_hv,HPZSTMY_hv,HPZSTMZ_hv,HPZATMN_hv,RMSMN1,HPDISAB1_hv, &
                HPDISAE1_hv,HPDISAB2_hv,HPDISAE2_hv,HPDIATL_hv,HPLDDIS_hv, &
                HPDISGAR_hv,HPDISLAR_hv,ZNQVCUT, ZNQECUT,ZNQOBND,ZNQOBDF,ZNMEET, &
                HPTEMPF_hv,HPTEMPR_hv,HPQICBF_hv,HPSAVX_hv,HPSAVY_hv,HPSAVZ_hv, &
                INBFRQX,HPDFCGAR_hv,HPDFCLAR_hv,HPDFCMP_hv,HPLDDFC_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPSAVZ_hv',NATOM,crl=HPSAVZ_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPSAVY_hv',NATOM,crl=HPSAVY_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPSAVX_hv',NATOM,crl=HPSAVX_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPTEMPR_hv',NSATME,intg=HPTEMPR_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPTEMPF_hv',NSATME,intg=HPTEMPF_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZSTMZ_hv',NATOM,crl=HPZSTMZ_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZSTMY_hv',NATOM,crl=HPZSTMY_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZSTMX_hv',NATOM,crl=HPZSTMX_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZZCPS_hv',NATOM,crl=HPZZCPS_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZYCPS_hv',NATOM,crl=HPZYCPS_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZXCPS_hv',NATOM,crl=HPZXCPS_hv)
           call chmdealloc('zerom2.src','ZCOMBO','HPZATMN_hv',2*NATOM,intg=HPZATMN_hv)

        ENDIF !if lmatch or not
     ENDIF !if right number of minor subspaces
     ! **************************************************************
     ! set up the next grid point (next combination)
     SLOT = 1
     DONECOMB = .FALSE.        
     DO WHILE ((SLOT.LE.NTAKEN).AND.(.NOT.DONECOMB))
        POSIT(SLOT) = POSIT(SLOT) + 1
        IF (POSIT(SLOT).GT.MAXPOS(SLOT)) THEN
           SLOT = SLOT + 1
        ELSE
           J = SLOT
           DO WHILE (J.GE.2)
              J = J - 1
              POSIT(J) = POSIT(J+1)+1
           ENDDO
           DONECOMB = .TRUE.
        ENDIF
     ENDDO !(setting up next combination)
  ENDDO ! (loop over all combinations)
  call chmdealloc('zerom2.src','ZCOMBO','WRKINIT1',NATOM,intg=WRKINIT1)
  call chmdealloc('zerom2.src','ZCOMBO','WRKINIT2',NATOM,intg=WRKINIT2)
  ! ******************************************************************
  !  end loop over searches here
  ! ******************************************************************
  IF (QRECUT) CALL ZRECUTSET(WSUBSP,SSSLST,SVCNFI,OKCCNT, &
       LOCONF,HICONF,NATOM,LODOFL,HIDOFL,MSTDOF,MSTDFV, &
       HPLOIE_hv,HPHIIE_hv,NPSBSP,CURCNF,MINIME, &
       HPZBNS_hv,NEWCONF)
  !      reset to initial conformer
  NINITR = 0
  NINITF = 0
  DO II = 1,NSSSLD
     SUBSP = SSSLST(II)
     ! change dof values in ic table
     JJ = LOCONF(SUBSP)
     DO III = LODOFL(JJ),HIDOFL(JJ)
        JJJ = MSTDOF(III)
        CURVAL = INIVAL(JJJ)
        DO ENTRYN  = LOICEN(JJJ),HIICEN(JJJ)
           !
           CALL CHNGEICVL(icr_struct%B1ic, &
                icr_struct%B2ic,icr_struct%T1ic, &
                icr_struct%T2ic,icr_struct%PIC, &
                HPLNLS_hv,HPCLST_hv, &
                ENTRYN,CURVAL)
        ENDDO  !loop over ic entries
     ENDDO !loop over lines
     ! initialize atom positions
     !        DO JJ = SIABEG(SUBSP),SIAEND(SUBSP)
     !            X(INIATL(JJ)) = ANUM
     !            Y(INIATL(JJ)) = ANUM
     !            Z(INIATL(JJ)) = ANUM
     !        ENDDO
     ! initialize atom positions
     DO JJ = SIABEG(SUBSP),SIAEND(SUBSP)
        ! store the atoms for the reverse ic build
        IF(ZQICBF(SUBSP).EQ.0) THEN
           NINITR = NINITR + 1
           INITRAR(NINITR) = INIATL(JJ)
           ! store the atoms for the forward ic build
        ELSE IF (ZQICBF(SUBSP).EQ.1) THEN
           NINITF = NINITF + 1
           INITFAR(NINITF) = INIATL(JJ)
        ENDIF
        !             X(INIATL(JJ)) = ANUM
        !             Y(INIATL(JJ)) = ANUM
        !             Z(INIATL(JJ)) = ANUM
     ENDDO
  ENDDO !loop over selected subspaces
  ! build
  CALL BILDCFR(NINITF,NINITR,INITFAR,INITRAR,NATOM)
  !
  IF(QZENERG) THEN
     IF(NEWCONF.EQ.0) THEN 
        WRITE(6,'(4X,A49)') &
             'WARNING:  NO GRIDPOINTS WERE BELOW ENERGY CUTOFF'
        IF(QZMINI) WRITE(6,'(4X,A20)') &
             'NO MIMINUM ASSIGNED'
     ELSE     
        ! save the minimum from this search  !but don't we already have this in GMINVAL, MINIME 
        IF (MINIME.LT.LDSSMNE) THEN
! don't you need random access here to the uncompressed conformer list?
           call ZSAVEGRMIN(SSSLST,LOCONF,HICONF,HPZNCB_hv,HPLCSS_hv,HPHCSS_hv, &
                HPCBSS_hv,ZMINPT,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,NATOM,LODOFL,HIDOFL, &
                MSTDOF,MSTDFV,LOICEN,HIICEN,HPGMNV_hv,HPLDSMN_hv)
           LDSSMNE = MINIME
        ENDIF
        ! 
        IF(QZMINI) THEN !assign structure to minimum conformer
           call ZREGENMINX(SSSLST,LOCONF,HICONF,HPZNCB_hv,HPLCSS_hv,HPHCSS_hv, &
                HPCBSS_hv,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,NATOM,LODOFL,HIDOFL,MSTDOF, &
                MSTDFV,LOICEN,HIICEN,HPGMNV_hv,MINIME,ZQICBF)
        ELSE IF (QMINPRNT) THEN
           call ZMINPRINT(SSSLST,LOCONF,HICONF,HPZNCB_hv,HPLCSS_hv,HPHCSS_hv, &
                HPCBSS_hv,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,NATOM,LODOFL,HIDOFL,MSTDOF, &
                MSTDFV,LOICEN,HIICEN,HPGMNV_hv,MINIME)
        ENDIF !if assigning minimum or printing out minimum
     ENDIF !if newconf is 0 or not
  ENDIF !if qzenerg
  !
  !
#if KEY_PARALLEL==0
  WRITE(OUTU,'(3X,A)')  &
       '*************STATISTICS FOR SEARCH:************'
  WRITE(OUTU,'(3X,A48,I14)')  &
       'TOTAL NUMBER OF CONFORMERS EVALUATED ', &
       TOTPNT
  WRITE(OUTU,'(3X,A48,I14)') 'CONFORMERS OUT OF DISTANCE BOUNDS ', &
       ZNQOBND-ZNQOBDF
  WRITE(OUTU,'(3X,A48,I14)')  &
       'ADDTNAL CONFS OUT OF IC CNSTRNT BOUNDS ',ZNQOBDF
  WRITE(OUTU,'(3X,A48,I14)')  &
       'ADDITIONAL CONFORMERS NOT < CONSTANT ENER CUTOFF ',ZNQECUT
  WRITE(OUTU,'(3X,A48,I14)')  &
       'ADDITIONAL CONFORMERS NOT < VARYING ENER CUTOFF ',ZNQVCUT     
  WRITE(OUTU,'(3X,A48,I14)') &
       'TOTAL POINTS MEETING ALL CRITERIA ',ZNMEET 
#endif 

  if(QZTIME) call loc_timer('END of ZCOMBO')

  RETURN
END SUBROUTINE ZCOMBO
!
SUBROUTINE ZMERGE(WSUBSP,NPSUBS,LOCONF,HICONF, &
     LODOFL,HIDOFL,MSTDOF,MSTDFV,CURCNF,NEWCONF, &
     LOICEN,HIICEN,SATBEG,SATEND,SUBATL,SIABEG,SIAEND, &
     INIATL,FLINIT,DOINIT,FLINITR,DOINITR, &
     GMINVAL,SVCNFI,OKCCNT,MINIME,TOTPNT, &
     NCSAVE,SSSLST,ZMINPT,ZVACUT,QWFIRS,ALILST, &
     XCOPYS,YCOPYS,ZCOPYS,STEMPX,STEMPY,STEMPZ, &
     ATOMIN,RMSMN1,DISATBEG1,DISATEND1,DISATBEG2, &
     DISATEND2,DISTATL,LDDISLS,DISTGTAR,DISTLTAR, &
     ZNQVCUT, ZNQECUT,ZNQOBND,ZNQOBDF,ZNMEET, &
     INITFAR,INITRAR,ZQICBF,SAVEXX,SAVEYY,SAVEZZ, &
     INBFRQX,DOFCGAR,DOFCLAR,DOFCMAP,LDDOFCON)
  !------------------------------------------
  !
  !  Does a search of selected subspaces
  !  Expects conformers in each subspace to be consecutively
  !  numbered
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use zdata_mod
  use psf
  use bases_fcm
  use inbnd
  use coord
  use egrad
  use energym
  use eutil
  use coordc
  use ctitla
  use image
  use stream
  use deriv !temporary
  use code !temporary
#if KEY_PERT==1
  use pert 
#endif
  use intcor_module
  use memory
  use actclus_mod
  use intcor2
  use corsubs
  use cvio
  use zutil,only: trim_conf
#if KEY_PARALLEL==1
  use nbndcc_utilb,only: parstoperr !temporary
  use async_util2,only: CPUEXCHINT2  
  use parallel 
  use cpustruc,only: CPUIDENT,CPUONES,NUMNODH,NUMHOOD,IMYKEYP,HOODCOM,IMYNHDP,HEADCOM  
  use asynccomg
  use zmod_comm,only: COMBIN_RESLT 
#endif
  use block_ltm,only: NOFORC
#if KEY_FOURD==1
  use fourdm
#endif
  use heurist,only:updeci
  use nbexcl,only:makitc
  !
  implicit none
  !  passed variables
  !  INTEGERS
  !
  real(chm_real),allocatable,dimension(:) :: VARB_hv
  real(chm_real),allocatable,dimension(:) :: GRAD_hv
  real(chm_real),allocatable,dimension(:) :: VREF_hv
  real(chm_real),allocatable,dimension(:) :: VBEST_hv
  real(chm_real),allocatable,dimension(:) :: GBEST_hv
  real(chm_real),allocatable,dimension(:) :: HPDUMFDI_hv
  integer,allocatable,dimension(:) :: HPDUMIM4_hv
  integer,allocatable,dimension(:) :: PGRAD_hv
  logical,allocatable,dimension(:) :: PSGRAD_hv
  real(chm_real),allocatable,dimension(:) :: PTHTAN_hv
  real(chm_real),allocatable,dimension(:) :: RMASS_hv
  real(chm_real),allocatable,dimension(:) :: GPREV_hv
  real(chm_real),allocatable,dimension(:) :: VARFIN_hv
  INTEGER,dimension(:) :: WSUBSP
  INTEGER,dimension(:) :: HIDOFL,LODOFL
  INTEGER,dimension(:) :: HICONF,LOCONF,MSTDOF
  INTEGER,dimension(:) :: CURCNF!passed for memory alloc
  INTEGER :: NPSUBS,NEWCONF
  INTEGER,dimension(:) :: LOICEN,HIICEN
  INTEGER,dimension(:) :: SATBEG,SATEND,SUBATL
  integer,dimension(:) :: FLINIT,FLINITR
  integer,dimension(:) :: DOINIT,DOINITR
  INTEGER,dimension(:) :: SIABEG,SIAEND,INIATL
  INTEGER,dimension(:) :: SVCNFI
  INTEGER TOTPNT !current gridpoint
  INTEGER CURENPNT !current energy evaluation #
  INTEGER OKCCNT
  INTEGER GPTMNY,NCSAVE
  integer(chm_int8) :: RGPTMNY
  INTEGER,dimension(:) :: SSSLST,ALILST
  integer :: ZMINPT,MASTDF 
  INTEGER ATOMIN(2,NATOM)
  INTEGER,dimension(:) :: DISATBEG1,DISATEND1,DISATBEG2, &
       DISATEND2,DISTATL,LDDISLS
  INTEGER,dimension(:) :: INITFAR,INITRAR,ZQICBF
  integer :: INBFRQX
  INTEGER,dimension(:) :: LDDOFCON

! REAL
  real(chm_real) MINIME!minimum energy over all grids
  real(chm_real),dimension(:) :: XCOPYS,YCOPYS,ZCOPYS, &
       STEMPX,STEMPY,STEMPZ
  real(chm_real),dimension(:) :: MSTDFV
  real(chm_real) :: ZVACUT
  real(chm_real),dimension(:) :: GMINVAL !global minimum dof values (running over all calc)
  real(chm_real) RMSMN1
  real(chm_real),dimension(:) :: DISTGTAR,DISTLTAR
  INTEGER ZNQVCUT, ZNQECUT,ZNQOBND,ZNQOBDF,ZNMEET
  real(chm_real),dimension(:) :: SAVEXX,SAVEYY,SAVEZZ
  real(chm_real),dimension(:) :: DOFCGAR,DOFCLAR
  INTEGER,dimension(:) :: DOFCMAP
! LOGICAL
  LOGICAL QWFIRS
  integer :: pass=0,UNUMB!temporary
  integer :: passbuild !temp
  integer :: COUNT,COUNT1,COUNT2,COUNT3

#if KEY_PARALLEL==1  /*paradecl*/
#if KEY_PARALLEL==1
  integer,allocatable,dimension(:) :: NCONFSS 
#endif
  integer(chm_int8),allocatable,dimension(:) :: BEGPT,ENDPT
  integer(chm_int8),allocatable,dimension(:) :: BEGINME
  integer(chm_int8),allocatable,dimension(:,:) :: BEGCONF
  logical :: QALLPROCP
  integer(chm_int4),allocatable,dimension(:),save :: LOCSHAND,LOCRHAND
  type(arofar_i4),allocatable,dimension(:),save :: LOCRBUF,LOCSBUF
  type(arofar),allocatable,dimension(:),save :: LOCSBUFR,LOCRBUFR
  integer :: WORKICNT,WORKRCNT
  integer,allocatable,dimension(:) :: WORKIAR
  real(chm_real),allocatable,dimension(:,:) :: WORKRAR
  integer :: LASTCNF,NCOUNT,NCOUNT2,THISCNF
  integer,allocatable,dimension(:) :: NSSTAG_TMP,NEWCONF_TMP,MSTDOF_TMP
  real(chm_real),allocatable,dimension(:) :: CURVAL_TMP,MYPOTEN_TMP,RMST1_TMP
!  real(chm_real) :: NEWCUT
  integer :: NLOCCONF
  integer,dimension(1) :: TEMP
  integer :: SAV_NUMNOD,SAV_MYNODP,SAV_MYNOD,NUMNODES,MYHOOD,MYLOCRNKP
  integer(chm_int4) :: SAV_COMM,ALLCOMM
#endif /*paradecl*/
!
! local variables
!     
  INTEGER :: SSSCNT,SUBSPA,SUBSP,NAT3,LOCGRDPT,LOCMINPT
! dummy storage of local subspace moniker (1,2,3...)
  LOGICAL DONECHANGES,QFIRENE 
  !  done with changes in dofs for next conf
  INTEGER JJ,KK,J,I,II,LL,MSDLLN,ENTRYN,LO,HI
  INTEGER CURDOF,NINIT,NINITR,ATOMNB,LOI,HII,JJJ,III
  INTEGER CCNFOR,IDIFFE,OLDLEV,NPR,ATM
  real(chm_real) CURVAL,DIFFER
  real(chm_real) RMST1,XXR1,YYR1,ZZR1
  ! for distances
  real(chm_real) SUM1X,SUM2X,SUM1Y,SUM2Y,SUM1Z,SUM2Z
  INTEGER DISTCON,BEGIN1,BEGIN2,END1,END2
  INTEGER POINT1,POINT2,ATM1,ATM2,NATM1,NATM2
  real(chm_real) XMEAN1,YMEAN1,ZMEAN1,XXX,YYY,ZZZ
  real(chm_real) XMEAN2,YMEAN2,ZMEAN2,ZDISTSQ
  LOGICAL QOOBOUND 
  ! for trajectories
  real(chm_real) AREAL  !dummy variable for traj writing 
  real(chm_real) MYPOTEN
  ! for 1st-order minimization
  real(chm_real)   STEP, TOLGRD, TOLFUN, TOLSTP 
  INTEGER  CONVRG, NVAR, NCALLS,ATMS
  INTEGER ZUNIT !temp[
  real(chm_real) STEPSZ
  ! for dof constraints
  INTEGER THISDC,CURCONS
#if KEY_PARALLEL==1
  integer :: pp,ss  /*temporary*/
#endif
  LOGICAL QDUMMY
  real(chm_real) :: cumetime_03=0,cumetime_04=0,cumetime_05=0,time1,etime,ctime
  real(chm_real) :: time_01,time_02,time_03,time_04,time_05,cumetime_01=0,cumetime_02=0
  real(chm_real) :: time_06,time_07
  real(chm_real) :: cumetime_cv, cumetime_cvt
  integer :: NODEME
#if KEY_PARALLEL==1 /*paradecl2*/
  integer :: MYUNIT,SUM,TEST,BEG,END,NODE !temporary
  real(chm_real),allocatable,dimension(:) :: LOCMIN
  real(chm_real) :: MINGLOB
  integer :: IERR,MINNOD,THISDOF,NSRCHED,DOFCNT
  integer,dimension(:),allocatable :: SRCHDLST,DFSRCHED
  real(chm_real),dimension(:),allocatable :: SRCHDVAL
  real(chm_real),allocatable,dimension(:) :: SENDMIN
  include 'mpif.h'
  logical :: QZDATCOMM
  integer :: NSECTS,MYSECT
#endif /*paradecl2*/
!------------------------------6---------------------------
  if(QZTIME) then
   call loc_timer('START OF ZMERGE')
   cumetime_01 = 0
   cumetime_02 = 0
   cumetime_03 = 0 !temporary
   cumetime_04 = 0
   cumetime_05 = 0
   cumetime_cvt = 0
  endif

  NOFORC = .true.
  if(QZ1STMIN) NOFORC= .false.

#if KEY_PARALLEL==1 /*setcomm*/
  QZDATCOMM = .false.
  if((NUMHOOD.ne.1).and.(NUMNOD.gt.1)) QZDATCOMM = .true.

  if(NUMHOOD.eq.0) then
   SAV_NUMNOD = NUMNOD
   SAV_MYNODP = MYNODP
   NSECTS=NUMNOD
   MYSECT=MYNODP
   MYNODP=1   !assign nsects, mysect first, then reassign mynodp, numnod
   NUMNOD=1
  else
   NSECTS=NUMHOOD
   MYSECT=IMYNHDP
   if(NUMHOOD.gt.1) then
    SAV_NUMNOD = NUMNOD
    SAV_MYNODP = MYNODP
    SAV_MYNOD = MYNOD
    SAV_COMM = COMM_CHARMM
    
    NUMNOD = NUMNODH
    MYNODP = IMYKEYP
    MYNOD = MYNODP-1
    COMM_CHARMM  = HOODCOM
   endif
  endif
#endif /*setcomm*/

  pass = pass +1 !temporary
  passbuild = 0
!  cumetime_cc = 0  !for bycc
  COUNT=0
  COUNT1=0
  COUNT2=0
  COUNT3=0
!  WRITE(6,*) 'MYNODGP ',MYNODGP,' entering ZMERGE*********************** pass ',pass
#if KEY_PARALLEL==1
  OUTLNCNT = 0 
#endif
  !
  OLDLEV = PRNLEV
  PRNLEV = 1 
  QFIRENE = .TRUE.  !flag for first energy call
  !
  DO JJ = 1,NATOM
     FLINIT(JJ) = 0
     FLINITR(JJ) = 0
     !        IF(QZCFIX) IMOVE(JJ)=1  !all fixed initially
  ENDDO
  AREAL = 1.0
  !
  DO SSSCNT = 1,NPSUBS
     SUBSPA = WSUBSP(SSSCNT)
     ! create initialization list
     LOI = SIABEG(SUBSPA)
     HII = SIAEND(SUBSPA) 
     !          WRITE(6,*) 'INIATL ',INIATL(JJJ)
     IF(ZQICBF(SUBSPA).EQ.1) THEN
        DO JJJ = LOI,HII
           !          WRITE(6,*) 'INIATL ',INIATL(JJJ)
           FLINIT(INIATL(JJJ)) = 1
           !          IF(QZCFIX) THEN
           !            IMOVE(INIATL(JJJ)) = 0  !initialized atoms move
           !          ENDIF
        ENDDO
        ! for ic building in reverse
     ELSE IF (ZQICBF(SUBSPA).EQ.0) THEN
        DO JJJ = LOI,HII
           !          WRITE(6,*) 'INIATL ',INIATL(JJJ)
           FLINITR(INIATL(JJJ)) = 1
           !          IF(QZCFIX) THEN
           !            IMOVE(INIATL(JJJ)) = 0  !initialized atoms move
           !          ENDIF
        ENDDO
     ENDIF
  ENDDO !loop over subspaces

!test comparison coords if necessary
   if(QZRMSD) then
     if((XCOMP(ACTV(1)).ge.9999).or.(YCOMP(ACTV(1)).ge.9999).or.(ZCOMP(ACTV(1)).ge.9999)) then
         call WRNDIE(-5,'<ZMERGE>', &
        'SOME COMPARISON COORDINATES ARE ZERO')
     endif
    endif
!
  ! mkitc update if necessary
  IF(MUSTUP) THEN
     CALL MAKITC(NATOMT,NATOM,IAC &
#if KEY_PERT==1
          ,QPERT,PPIAC,PPIACNB & 
#endif
#if KEY_PERT==0
          ,.FALSE.,0,0 &         
#endif
          )
     ! do a code update
     CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC, &
          NBOND,IB,JB,NTHETA,IT,JT,KT, &
          NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
          .FALSE.,0,                        & ! DRUDE
#if KEY_CMAP==1
     ICCT,NCRTERM, &
          I1CT,J1CT,K1CT,L1CT, &
          I2CT,J2CT,K2CT,L2CT, &
#endif 
          !...##IF CMAP
          !     $                 0,0,          !cmap
          !     $                 0,0,0,0,      !cmap
          !     $                 0,0,0,0,      !cmap
          !...##ENDIF
     .FALSE.,.FALSE.)
     CALL MAKPHI
     MUSTUP=.FALSE.
  ENDIF !mustup
  !
  ! compress initialization info
  !
  NINITR = 0
  NINIT = 0
  DO I = 1,NATOM
     IF (FLINIT(I).GT.0) THEN
        NINIT = NINIT + 1
        DOINIT(NINIT) = I
     ENDIF
     IF (FLINITR(I).GT.0) THEN
        NINITR = NINITR + 1
        DOINITR(NINITR) = I
     ENDIF
  ENDDO
  if(QVERBOSE) then
   WRITE(6,*) 'in zmerge, NINIT is ',NINIT,' NINITR is ',NINITR, ' MYNODGP ',&
#if KEY_PARALLEL==1
    MYNODGP
#else
    '1'
#endif
  endif

! if requested, try to reduce the size of the active IC table
  call MK_ICACTV(1,icr_struct%lenic,X,Y,Z, &
    icr_struct%B1ic,icr_struct%B2ic, &
    icr_struct%T1ic,icr_struct%T2ic, &
    icr_struct%PIC, icr_struct%IAR, &
    icr_struct%JAR, icr_struct%KAR, &
    icr_struct%LAR, icr_struct%TAR, &
    NATOM,DOINIT,NINIT,ICACTVF,NICACTVF)

    call MK_ICACTV(1,icr_struct%lenic,X,Y,Z, &
    icr_struct%B1ic,icr_struct%B2ic, &
    icr_struct%T1ic,icr_struct%T2ic, &
    icr_struct%PIC, icr_struct%IAR, &
    icr_struct%JAR, icr_struct%KAR, &
    icr_struct%LAR, icr_struct%TAR, &
    NATOM,DOINITR,NINITR,ICACTVR,NICACTVR)

  !
  !
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
   WRITE(6,*) ' '
   WRITE(6,*) 'THE FOLLOWING SUBSPACES ARE BEING SEARCHED:'
   WRITE(6,'(13X,A23)') 'SUBSPACE   # CONFORMERS' 
#if KEY_PARALLEL==1
  endif 
#endif
#if KEY_PARALLEL==1
  if(allocated(NCONFSS)) call chmdealloc('zerom2.src','ZMERGE','NCONFSS',NSUBME,intg=NCONFSS)
  call chmalloc('zerom2.src','ZMERGE','NCONFSS',NSUBME,intg=NCONFSS)
  if(allocated(BEGPT)) call chmdealloc('zerom2.src','ZMERGE','BEGPT',NSECTS,ci8=BEGPT)
  call chmalloc('zerom2.src','ZMERGE','BEGPT',NSECTS,ci8=BEGPT)
  if(allocated(ENDPT)) call chmdealloc('zerom2.src','ZMERGE','ENDPT',NSECTS,ci8=ENDPT)
  call chmalloc('zerom2.src','ZMERGE','ENDPT',NSECTS,ci8=ENDPT)
  if(allocated(BEGCONF)) call chmdealloc('zerom2.src','ZMERGE','BEGCONF',NSECTS,NSUBME,ci8=BEGCONF)
  call chmalloc('zerom2.src','ZMERGE','BEGCONF',NSECTS,NSUBME,ci8=BEGCONF)
  if(allocated(BEGINME)) call chmdealloc('zerom2.src','ZMERGE','BEGINME',NPSUBS,ci8=BEGINME)
  call chmalloc('zerom2.src','ZMERGE','BEGINME',NPSUBS,ci8=BEGINME)
  NCONFSS=0
  BEGPT=0
  ENDPT=0
  BEGCONF=0
  BEGINME = 0
#endif

  RGPTMNY = 1
  DO I = 1,NPSUBS
     SUBSPA = WSUBSP(I)
     !      WRITE(6,*) 'FOR PASSED SS ',I,' SUBSPA = ',SUBSPA,
     !     & ' HICONF ',HICONF(SUBSPA)
     J = HICONF(SUBSPA) - LOCONF(SUBSPA) + 1
     RGPTMNY = J*RGPTMNY
     !        WRITE(6,*) ' NUMBER OF CONFORMERS IS ',J
#if KEY_PARALLEL==1
     if(MYNODGP.eq.1) then 
#endif
      WRITE(6,'(5X,I14,1X,I14)') SUBSPA,J 
#if KEY_PARALLEL==1
     endif 
#endif
#if KEY_PARALLEL==1
     NCONFSS(I) = J   /*number of conformers in each ss*/
#endif
  enddo
#if KEY_PARALLEL==1
  if(MYNODGP.eq.1) then 
#endif
   WRITE(6,'(A30,I18)') 'TOTAL # GRIDPOINTS IN SEARCH= ', &
       RGPTMNY
#if KEY_PARALLEL==1
  endif 
#endif
  !      DO II = 1,NPSUBS
  !        DO III = LOCONF(II),HICONF(II)
  !          DO JJ = LODOFL(III),HIDOFL(III)
  !            WRITE(6,*) II,III,JJ,MSTDOF(JJ),MSTDFV(JJ)
  !          ENDDO
  !        ENDDO
  !      ENDDO
  !
! if parallel, decide who calculates what part of grid
#if KEY_PARALLEL==1 /*(parazalloc)*/
! allocate memory
  if(QZDATCOMM) then
   if(allocated(NSSTAG_AR)) call chmdealloc('zerom2.src','ZMERGE','NSSTAG_AR',MAXLNCNT,intg=NSSTAG_AR)
   if(allocated(NEWCONF_AR)) call chmdealloc('zerom2.src','ZMERGE','NEWCONF_AR',MAXLNCNT,intg=NEWCONF_AR)
   if(allocated(MSTDOF_AR)) call chmdealloc('zerom2.src','ZMERGE','MSTDOF_AR',MAXLNCNT,intg=MSTDOF_AR)
   if(allocated(CURVAL_AR)) call chmdealloc('zerom2.src','ZMERGE','CURVAL_AR',MAXLNCNT,crl=CURVAL_AR)
   if(allocated(MYPOTEN_AR)) call chmdealloc('zerom2.src','ZMERGE','MYPOTEN_AR',MAXLNCNT,crl=MYPOTEN_AR)
   if(allocated(RMST1_AR)) call chmdealloc('zerom2.src','ZMERGE','RMST1_AR',MAXLNCNT,crl=RMST1_AR)
   call chmalloc('zerom2.src','ZMERGE','NSSTAG_AR',MAXLNCNT,intg=NSSTAG_AR)
   call chmalloc('zerom2.src','ZMERGE','NEWCONF_AR',MAXLNCNT,intg=NEWCONF_AR)
   call chmalloc('zerom2.src','ZMERGE','MSTDOF_AR',MAXLNCNT,intg=MSTDOF_AR)
   call chmalloc('zerom2.src','ZMERGE','CURVAL_AR',MAXLNCNT,crl=CURVAL_AR)
   call chmalloc('zerom2.src','ZMERGE','MYPOTEN_AR',MAXLNCNT,crl=MYPOTEN_AR)
   call chmalloc('zerom2.src','ZMERGE','RMST1_AR',MAXLNCNT,crl=RMST1_AR)
   NSSTAG_AR = 0
   NEWCONF_AR = 0
   MSTDOF_AR = 0
   CURVAL_AR = 0
   MYPOTEN_AR = 0
   RMST1_AR = 0
  endif

  QALLPROCP=.true.
  
  ! for testing
!  NPSUBS = 2
!  NCONFSS(1) = 100
!  NCONFSS(2) = 1000000
!WRITE(6,*) 'MYNODGP ',MYNODGP,' IMYKEYP ',IMYKEYP,' NUMNOD ',NUMNOD,' NUMNODG ',NUMNODG,' NUMHOOD ',NUMHOOD
!  if(NUMHOOD.gt.1) then  !if there are neighborhoods, and > 1
!   SAV_NUMNOD = NUMNOD
!   SAV_MYNODGP = MYNODGP
!   SAV_MYNOD = MYNOD
!   SAV_COMM = COMM_CHARMM
!   write(6,*) 'COMM_CHARMM is ',COMM_CHARMM,' MPI_COMM_WORLD is ',MPI_COMM_WORLD
!
!   NUMNOD = NUMNODH
!   MYNODGP = IMYKEYP
!   MYNOD = MYNODGP-1
!   COMM_CHARMM  = HOODCOM
!  endif

   call ZPART_GRID(NPSUBS,NCONFSS,BEGPT,ENDPT,BEGINME,NSECTS,QALLPROCP,BEGCONF,MYSECT)

!  do PP = 1,NUMNOD
!   do SS = 1,NPSUBS
!   UNUMB = MYNODGP+10*(pass+1) 
!    WRITE(UNUMB,*) 'processor ',PP,' subspace ',WSUBSP(SS),' begins at conformer ',BEGCONF(PP,SS), &
!   ' out of ',NCONFSS(SS),' pass ',pass
!   enddo
!  enddo
#endif /*(parazalloc)*/
  ! ****************************************************************
  ! ****************************************************************
  ! MAIN LOOP BEGINS HERE
  ! ****************************************************************
  ! ****************************************************************
!Note the subspace WSUBSP(1) "spins" the fastest
#if KEY_PARALLEL==1
  CURCNF(1) = BEGINME(1) + LOCONF(WSUBSP(1)) - 2 
!  WRITE(6,*) '>>INITMYNODGP ',MYNODGP,' initial setting CURCNF of SSS',1,' to ',CURCNF(1)
#else
  CURCNF(1) = LOCONF(WSUBSP(1)) 
#endif
!     WRITE(6,*) 'MYNODGP ',MYNODGP,' setting CURCNF of SS 1 to ',CURCNF(1)
  !
  ! set selected subspaces to starting conformations
  ! not necessary for 1st selected subspace (unless parallel)
!  WRITE(6,*) 'selecting starting conformations'
  do SSSCNT =1,NPSUBS
     !       WRITE(6,*) 'SETTING STARTING CONFORMATION FOR SUBSPACE ',
     !     & WSUBSP(SSSCNT)
     SUBSPA = WSUBSP(SSSCNT)
#if KEY_PARALLEL==1
     CURCNF(SSSCNT) = BEGINME(SSSCNT) + LOCONF(SUBSPA) - 1  
!     WRITE(6,*) '>>INITMYNODGP ',MYNODGP,' initial setting CURCNF of SS ',SSSCNT,' to ',CURCNF(SSSCNT)
!     WRITE(6,*) '>>INITMYNODGP ',MYNODGP,' SUBSPA ',SUBSPA,' LOCONF ',LOCONF(SUBSPA)
#else
     CURCNF(SSSCNT) = LOCONF(SUBSPA)   !starting conformations
#endif
     DO MSDLLN = LODOFL(CURCNF(SSSCNT)), &
          HIDOFL(CURCNF(SSSCNT))
        !            WRITE(6,*) ' MSDLLN ',MSDLLN
        CURDOF = MSTDOF(MSDLLN)
        CURVAL = MSTDFV(MSDLLN)
        DO ENTRYN  = LOICEN(CURDOF),HIICEN(CURDOF)
           !              WRITE(6,*) ' ENTRYN ',ENTRYN
           !
           CALL CHNGEICVL(icr_struct%B1ic, &
                icr_struct%B2ic,icr_struct%T1ic, &
                icr_struct%T2ic,icr_struct%PIC, &
                HPLNLS_hv,HPCLST_hv, &
                ENTRYN,CURVAL)
        ENDDO
     ENDDO
  ENDDO
  !
  !      ZNQOBND = 0 !count of points out of all constraints bounds
  !      ZNQOBDF = 0 !count of points out of ic constr bounds
  CURENPNT = 0 !count of energy evaluations
  !      WRITE(6,*) 'RGPTMNY is ', RGPTMNY
  CURCNF(1) = CURCNF(1) - 1
  if(QZTIME) call loc_timer('ZMERGE BEG GRID')

#if KEY_PARALLEL==1
  NLOCCONF = 0
  LOCGRDPT = 0

  do I = BEGPT(MYSECT),ENDPT(MYSECT)
#else
  GPTMNY = INT(RGPTMNY)
  DO I = 1,GPTMNY
#endif
     call seconds(etime,ctime)
     time_01 = etime
     TOTPNT = TOTPNT + 1
     LOCGRDPT = LOCGRDPT + 1
     DONECHANGES = .FALSE.
     SSSCNT = 1
     SUBSPA = WSUBSP(SSSCNT)
     DO WHILE((.NOT.DONECHANGES).AND.(SSSCNT.LE.NPSUBS)) 
        CURCNF(SSSCNT) = CURCNF(SSSCNT) + 1
        ! if current conf is above subspace's range, reset to low conf 
        ! and goto next subspace
        IF(CURCNF(SSSCNT).GT.HICONF(SUBSPA)) THEN
           CURCNF(SSSCNT) = LOCONF(SUBSPA)
           DO MSDLLN = LODOFL(CURCNF(SSSCNT)), &
                HIDOFL(CURCNF(SSSCNT))
              CURDOF = MSTDOF(MSDLLN)
              CURVAL = MSTDFV(MSDLLN)
              DO ENTRYN  = LOICEN(CURDOF),HIICEN(CURDOF)
                 CALL CHNGEICVL(icr_struct%B1ic, &
                      icr_struct%B2ic,icr_struct%T1ic, &
                      icr_struct%T2ic,icr_struct%PIC, &
                      HPLNLS_hv,HPCLST_hv, &
                      ENTRYN,CURVAL)
              ENDDO
           ENDDO
           !       WRITE(6,*) 'HI: SUBSPN ',SSSCNT,' SUBSPA ',SUBSPA,
           !     & ' CUR CONF ',CURCNF(SSSCNT)
           !          DO LL = LODOFL(CURCNF(SSSCNT)),HIDOFL(CURCNF(SSSCNT))
           !           WRITE(6,*) ' LINE IS ',LL,' DOF IS ', MSTDOF(LL)
           !          ENDDO
           SSSCNT = SSSCNT + 1
           SUBSPA = WSUBSP(SSSCNT)
           ! otherwise, you are finished with conformer advancement
        ELSE   ! conformer is not out of range 
           DONECHANGES = .TRUE.
           ! change the current subspace to the selected conformer

           do MSDLLN = LODOFL(CURCNF(SSSCNT)),HIDOFL(CURCNF(SSSCNT))
              CURDOF = MSTDOF(MSDLLN)
              CURVAL = MSTDFV(MSDLLN)
              DO ENTRYN  = LOICEN(CURDOF),HIICEN(CURDOF)
                 CALL CHNGEICVL(icr_struct%B1ic, &
                      icr_struct%B2ic,icr_struct%T1ic, &
                      icr_struct%T2ic,icr_struct%PIC, &
                      HPLNLS_hv,HPCLST_hv, &
                      ENTRYN,CURVAL)
              ENDDO
           ENDDO
        ENDIF
     ENDDO !do while not donechanges in conformer selection

     if(QZTIME) then
      call seconds(etime,ctime)
      time_02 = etime
      cumetime_01 = cumetime_01 + etime - time_01
     endif
! need to put the active IC tables in here.
     if(QCVTIME) then
      call BILDCFR(NINIT,NINITR,DOINIT,DOINITR,NATOM,ICACTVF=ICACTVF,NICACTVF=NICACTVF,ICACTVR=ICACTVR, &
       NICACTVR=NICACTVR,pnotest=.true.,cumetime_cv=cumetime_cv)
     else
      call BILDCFR(NINIT,NINITR,DOINIT,DOINITR,NATOM,ICACTVF=ICACTVF,NICACTVF=NICACTVF,ICACTVR=ICACTVR, &
       NICACTVR=NICACTVR,pnotest=.true.)
     endif
!     call BILDCFR(NINIT,NINITR,DOINIT,DOINITR,NATOM)
     passbuild = passbuild + 1
     if(QZTIME) then
      if(QCVTIME) cumetime_cvt = cumetime_cvt + cumetime_cv
      call seconds(etime,ctime)
      time_03 = etime
      cumetime_02 = cumetime_02 + etime - time_02
     endif
     ! if the gridpoint is to be sampled, continue
     ZSTEPN = ZSTEPN + 1
     ! ************************************************************
     ! check against geometric constraints
     ! ************************************************************
     !         WRITE(6,*) '---------------------------------------------'
     !         WRITE(6,*) '---------------New Gridpoint-----------------'
     QOOBOUND = .FALSE.
     IF(NDISTLD.GT.0) THEN
        III = 1
        DO WHILE((III.LE.NDISTLD).AND. &
             (.NOT.QOOBOUND))   !while not out of bounds
           !           WRITE(6,*) '********************************************'
           !           WRITE(6,*) '***********New Distant Constraint***********'
           DISTCON = LDDISLS(III)
           BEGIN1 = DISATBEG1(DISTCON)
           BEGIN2 = DISATBEG2(DISTCON)
           !           WRITE(6,*) 'DISATBEG1 ',DISATBEG1(DISTCON),
           !     & 'DISATBEG2 ',DISATBEG2(DISTCON)
           !           WRITE(6,*) ' up lim ',
           !     & DISTLTAR(DISTCON),' lo lim ',DISTGTAR(DISTCON)
           END1 = DISATEND1(DISTCON)
           END2 = DISATEND2(DISTCON)
           SUM1X = 0
           SUM1Y = 0
           SUM1Z = 0
           !
           DO POINT1 = BEGIN1,END1
              ATM1 = DISTATL(POINT1)
              !            WRITE(6,*) 'ATOM ',ATM1,' X(ATM1) ',X(ATM1)
              SUM1X = SUM1X + X(ATM1)
              SUM1Y = SUM1Y + Y(ATM1)
              SUM1Z = SUM1Z + Z(ATM1) 
           ENDDO
           !
           NATM1 = END1-BEGIN1+1
           XMEAN1 = SUM1X/NATM1
           YMEAN1 = SUM1Y/NATM1
           ZMEAN1 = SUM1Z/NATM1
           !           WRITE(6,*) 'AVG COORDINATES 1: ',XMEAN1,YMEAN1,ZMEAN1
           SUM2X = 0
           SUM2Y = 0
           SUM2Z = 0
           DO POINT2 = BEGIN2,END2
              ATM2 = DISTATL(POINT2)
              SUM2X = SUM2X + X(ATM2)
              SUM2Y = SUM2Y + Y(ATM2)
              SUM2Z = SUM2Z + Z(ATM2)
           ENDDO
           NATM2 = END2-BEGIN2+1
           XMEAN2 = SUM2X/NATM2
           YMEAN2 = SUM2Y/NATM2
           ZMEAN2 = SUM2Z/NATM2
           !           WRITE(6,*) 'AVG COORDINATES 2: ',XMEAN2,YMEAN2,ZMEAN2
           XXX = XMEAN2-XMEAN1
           XXX = XXX*XXX
           YYY = YMEAN2-YMEAN1
           YYY = YYY*YYY
           ZZZ = ZMEAN2-ZMEAN1
           ZZZ = ZZZ*ZZZ
           ZDISTSQ = XXX+YYY+ZZZ
           !           WRITE(6,*) 'DISTN ',III,' LOADED CONSTRAINT ',DISTCON,
           !     & ' DISTSQ ',ZDISTSQ
           !       WRITE(6,*) ' up lim ',
           !     & DISTLTAR(DISTCON),' lo lim ',DISTGTAR(DISTCON)
           IF(ZDISTSQ.GT.DISTLTAR(DISTCON)) QOOBOUND = .TRUE.
           IF(ZDISTSQ.LT.DISTGTAR(DISTCON)) QOOBOUND = .TRUE.
           !           WRITE(6,*) 'QOOBOUND IS ',QOOBOUND
           III = III + 1
        ENDDO !loop over loaded constraints while in bounds
     ENDIF !if there are any loaded distance constraints
     !       WRITE(6,*) 'FOR GRIDPOINT ',ZSTEPN,' QOOBOUND IS ',QOOBOUND
     !
     ! check internal coordinate constraints
     IF(.NOT.QOOBOUND) THEN
        IF(QICCNA) THEN !if dof constaints required (loaded)
           !         WRITE(6,*) '----------CHECKING INTERNAL COOR CONST------------'
           ! first you need to fill the ic table 
           QDUMMY = .FALSE.
           !
           !
           CALL FILLIC (icr_struct%lenic,&                       
                QDUMMY,QDUMMY,X,Y,Z, &
                icr_struct%B1ic,icr_struct%B2ic, &
                icr_struct%T1ic,icr_struct%T2ic, &
                icr_struct%PIC, icr_struct%IAR, &
                icr_struct%JAR, icr_struct%KAR, &
                icr_struct%LAR, icr_struct%TAR)
           JJJ = 1 
           DO WHILE((JJJ.LE.NDOFCLD).AND.(.NOT.QOOBOUND)) 
              CURCONS = LDDOFCON(JJJ) !constraint number
              !           WRITE(6,*) 'CURCONS is ',CURCONS
              THISDC = DOFCMAP(CURCONS) !dof 
              !           WRITE(6,*) 'THISDC is ',THISDC
              ENTRYN = LOICEN(THISDC)     
              !           WRITE(6,*) 'ENTRYN is ',ENTRYN
              CALL GETICVL(icr_struct%B1ic, &
                   icr_struct%B2ic,icr_struct%T1ic, &
                   icr_struct%T2ic,icr_struct%PIC, &
                   HPLNLS_hv,HPCLST_hv, &
                   ENTRYN,CURVAL)
              IF(CURVAL.LT.0) CURVAL = CURVAL + 360
              !           WRITE(6,*) 'LOADED IC CONS IS ',CURCONS,' DOF is ',
              !     & THISDC,' VAL IS ',CURVAL
              !           WRITE(6,*) 'UP LIM is ',DOFCLAR(CURCONS)
              !            WRITE(6,*) 'LOW LIM is ',DOFCGAR(CURCONS)
              IF(CURVAL.LE.DOFCGAR(CURCONS)) QOOBOUND = .TRUE. 
              IF(CURVAL.GE.DOFCLAR(CURCONS)) QOOBOUND = .TRUE.
              JJJ = JJJ + 1
           ENDDO
           IF(QOOBOUND) ZNQOBDF = ZNQOBDF+1
        ENDIF
     ENDIF
     !       
     IF(.NOT.QOOBOUND) THEN
        IF (QZENERG) THEN
           CURENPNT = CURENPNT + 1
           ! update the non-bonded list if necessary
           !          WRITE(6,*) 'CALCULATING ENERGIES'
           ! ******************************************************************
           !
           !           ZUNIT = 6
           !           CALL PRINTE(ZUNIT, EPROP, ETERM, 'STPD', 'MIN', .TRUE.,
           !     *            NCALLS, ZERO, Z1MSTP, .TRUE.)
           !          WRITE(6,*) 'before mini: Z1MSTP is ',Z1MSTP
8787       CONTINUE
           !      DO II =2383,2410
           !      WRITE(6,'(A,1X,I4,1X,A,1X,F20.16,1X,A,1X,F20.16,1X,A,1X,F20.16)')
           !     & 'ATM HERE',II,' X ',X(II),' Y ',Y(II),' Z ',Z(II)
           !      ENDDO
           !          WRITE(6,*) 'ooooooooo FILLING XYZ ooooooooooooooooooooooo'
           !      IF(CURENPNT.EQ.1) CALL FILLXYZ(X,Y,Z)
           !      IF(CURENPNT.EQ.2) CALL FILLXYZ2(X,Y,Z)
           !      DO II =2322,2410
           !      WRITE(6,'(A,1X,I4,1X,A,1X,F20.16,1X,A,1X,F20.16,1X,A,1X,F20.16)')
           !     & 'ATM ',II,' X ',X(II),' Y ',Y(II),' Z ',Z(II)
           !      ENDDO

           !         WRITE(6,*) '*** IN ZMERGE ENERGY IS = ',EPROP(EPOT)
           GOTO 8788
           WRITE(6,*) ' ELEC ',ETERM(ELEC)
           WRITE(6,*) ' VDW ',ETERM(VDW)
           WRITE(6,*) ' ASP ',ETERM(ASP)
           WRITE(6,*) ' DIHE ',ETERM(DIHE)
           WRITE(6,*) ' ANGL ',ETERM(ANGLE)
           WRITE(6,*) ' BOND ',ETERM(BOND)
           WRITE(6,*) ' IMPR ',ETERM(IMDIHE)
           WRITE(6,*) ' UREY ',ETERM(UREYB)
           WRITE(6,*) ' CMAP ',ETERM(CMAP)
           WRITE(6,*) ' SUM PE ',ETERM(ELEC) + ETERM(VDW) + &
                ETERM(ASP) + ETERM(DIHE) + ETERM(ANGLE) + ETERM(BOND) +  &
                ETERM(UREYB) + ETERM(IMDIHE) + ETERM(CMAP)
           WRITE(6,*) 'QZST1MIN ',QZ1STMIN
8788       CONTINUE

           ! ***************************************************************
           ! ************** start of 1st order  minimization ***************
           ! ***************************************************************
           IF(QZ1STMIN) THEN
              NCALLS=0
              !
              ! if minimizing, save the relavant atom positions before minimization
              ! update the non-bonded list, unless no updates being done
              IF(INBFRQX.NE.0) THEN
                 CALL NBONDS(X,Y,Z,BNBND,BIMAG)
              ENDIF
              DO III = 1,NACTVE
                 ATMS = ACTV(III) 
                 SAVEXX(ATMS) = X(ATMS)
                 SAVEYY(ATMS) = Y(ATMS)
                 SAVEZZ(ATMS) = Z(ATMS)
              ENDDO
              !            
              !            Z1STPRN = 5 !temporary, for conj grad
              !     Important: You have to reset the step size as follows because
              !     the minimization routines mess with it
              STEPSZ = Z1MSTP !reset stepsize 
              !
              CONVRG = 0
              !
              TOLFUN = ZERO
              TOLGRD = ZERO
              TOLSTP = ZERO

              CALL CALCNVAR(.FALSE.,(/0/),NVAR)

              !
              !       Allocate storage.
              !
              if(allocated(VARB_hv)) call chmdealloc('zerom2.src','ZMERGE','VARB_hv',NVAR,crl=VARB_hv)
              if(allocated(GRAD_hv)) call chmdealloc('zerom2.src','ZMERGE','GRAD_hv',NVAR,crl=GRAD_hv)
              if(allocated(VREF_hv)) call chmdealloc('zerom2.src','ZMERGE','VREF_hv',NVAR,crl=VREF_hv)
              if(allocated(VBEST_hv)) call chmdealloc('zerom2.src','ZMERGE','VBEST_hv',NVAR,crl=VBEST_hv)
              if(allocated(GBEST_hv)) call chmdealloc('zerom2.src','ZMERGE','GBEST_hv',NVAR,crl=GBEST_hv)
              if(allocated(HPDUMFDI_hv)) call chmdealloc('zerom2.src','ZMERGE','HPDUMFDI_hv',NATOM,crl=HPDUMFDI_hv)
              if(allocated(HPDUMIM4_hv)) call chmdealloc('zerom2.src','ZMERGE','HPDUMIM4_hv',NATOM,intg=HPDUMIM4_hv)
              if(allocated(VARFIN_hv)) call chmdealloc('zerom2.src','ZMERGE','VARFIN_hv',NVAR,crl=VARFIN_hv)

              call chmalloc('zerom2.src','ZMERGE','VARB_hv',NVAR,crl=VARB_hv)
              call chmalloc('zerom2.src','ZMERGE','GRAD_hv',NVAR,crl=GRAD_hv)
              call chmalloc('zerom2.src','ZMERGE','VREF_hv',NVAR,crl=VREF_hv)
              call chmalloc('zerom2.src','ZMERGE','VBEST_hv',NVAR,crl=VBEST_hv)
              call chmalloc('zerom2.src','ZMERGE','GBEST_hv',NVAR,crl=GBEST_hv)
              call chmalloc('zerom2.src','ZMERGE','HPDUMFDI_hv',NATOM,crl=HPDUMFDI_hv)
              call chmalloc('zerom2.src','ZMERGE','HPDUMIM4_hv',NATOM,intg=HPDUMIM4_hv)
              call chmalloc('zerom2.src','ZMERGE','VARFIN_hv',NVAR,crl=VARFIN_hv)
              VARB_hv=ZERO
              GRAD_hv=ZERO
              VREF_hv=ZERO
              VBEST_hv=ZERO
              GBEST_hv=ZERO
              HPDUMFDI_hv=ZERO
              HPDUMIM4_hv=0
              VARFIN_hv=ZERO
#if KEY_BROKEN==1
              !
              if(allocated(PGRAD_hv)) call chmdealloc('zerom2.src','ZMERGE','PGRAD_hv',NVAR,intg=PGRAD_hv)
              if(allocated(PSGRAD_hv)) call chmdealloc('zerom2.src','ZMERGE','PSGRAD_hv',NVAR,log=PSGRAD_hv)
              if(allocated(PTHTAN_hv)) call chmdealloc('zerom2.src','ZMERGE','PTHTAN_hv',NVAR,crl=PTHTAN_hv)
              if(allocated(RMASS_hv)) call chmdealloc('zerom2.src','ZMERGE','RMASS_hv',NVAR,crl=RMASS_hv)
              if(allocated(GPREV_hv)) call chmdealloc('zerom2.src','ZMERGE','GPREV_hv',NVAR,crl=GPREV_hv)
              call chmalloc('zerom2.src','ZMERGE','PGRAD_hv',NVAR,intg=PGRAD_hv)
              call chmalloc('zerom2.src','ZMERGE','PSGRAD_hv',NVAR,log=PSGRAD_hv)
              call chmalloc('zerom2.src','ZMERGE','PTHTAN_hv',NVAR,crl=PTHTAN_hv)
              call chmalloc('zerom2.src','ZMERGE','RMASS_hv',NVAR,crl=RMASS_hv)
              call chmalloc('zerom2.src','ZMERGE','GPREV_hv',NVAR,crl=GPREV_hv)
              PGRAD_hv=0
              PSGRAD_hv=.false.
              PTHTAN_hv=ZERO
              RMASS_hv=ZERO
              GPREV_hv=ZERO
#endif 
              !
              !       Fill the variable array with the coordinates to be optimized.
              !
              call GETVR1(.TRUE.,NATOM,VARB_hv,IMOVE,X,Y,Z,.FALSE., &
#if KEY_CHEQ==1
                   .FALSE.,(/ZERO/), &                
#endif
                   XTLTYP,XTLABC,XTLREF &
#if KEY_FLUCQ==1
                   ,.FALSE.,(/ZERO/),(/0/) &          
#endif
#if KEY_FOURD==0
                   ,.FALSE.,HPDUMFDI_hv,HPDUMIM4_hv & 
#endif
#if KEY_FOURD==1
                   ,DIM4,FDIM,IMOVE4 &                
#endif
                   )
              !
              ! decide whether it's SD or CONJ
              IF (Z1STTYP.EQ.0) THEN

                 call STEEP2(0,NVAR,VARB_hv,GRAD_hv,VREF_hv,Z1MNSTP,STEPSZ, &
                      TOLFUN,TOLGRD,TOLSTP,CONVRG,NCALLS,VBEST_hv,GBEST_hv, &
                      0, .FALSE.)
                 VARFIN_hv = VBEST_hv
              ELSE IF (Z1STTYP.EQ.1) THEN
                 !         Use VREF  in place of NEWV (CONJUG in conjug.src)
                 !         Use VBEST in place of P    (CONJUG in conjug.src)
                 call CONJG2(NVAR,VARB_hv,VREF_hv,GRAD_hv, &
#if KEY_CHEQ==1
                      NATOM, .FALSE., EPROP(SDMIN), &  
#endif
                      0,Z1MNSTP,100,PT9999,1,Z1STPRN,STEPSZ,VBEST_hv,OUTU,TOLFUN,TOLGRD, &
                      100,TOLSTP,CONVRG,NCALLS)
                 VARFIN_hv = VARB_hv
                 !
              ELSE
                 CALL WRNDIE (-5, '<ZMERGE>', &
                      'INTERNAL ERROR---UNKNOWN ALGORITHM')
              ENDIF

              !
              !       Fill the coordinate arrays with the optimized variables.
              !
              call PUTVR1(.TRUE.,NATOM,VARFIN_hv,IMOVE,X,Y,Z, & 
#if KEY_CHEQ==1
                   .FALSE.,(/ZERO/),(/ZERO/),(/ZERO/), &
#endif
                   .FALSE.,XTLTYP,XTLABC,XTLREF,.TRUE. &
#if KEY_FLUCQ==1
                   ,.FALSE.,(/ZERO/),(/0/) &
#endif
#if KEY_FOURD==0
                   ,.FALSE.,(/ZERO/),(/0/) &
#endif
#if KEY_FOURD==1
                   ,DIM4,FDIM,IMOVE4 &
#endif
                   )
              !       Clear up any temporary storage space.
              !
              call chmdealloc('zerom2.src','ZMERGE','GRAD_hv',NVAR,crl=GRAD_hv)
              call chmdealloc('zerom2.src','ZMERGE','VARB_hv',NVAR,crl=VARB_hv)
              call chmdealloc('zerom2.src','ZMERGE','VREF_hv',NVAR,crl=VREF_hv)
              call chmdealloc('zerom2.src','ZMERGE','VBEST_hv',NVAR,crl=VBEST_hv)
              call chmdealloc('zerom2.src','ZMERGE','GBEST_hv',NVAR,crl=GBEST_hv)
              call chmdealloc('zerom2.src','ZMERGE','HPDUMFDI_hv',NATOM,crl=HPDUMFDI_hv)
              !       'HPDUMFDI')
              call chmdealloc('zerom2.src','ZMERGE','HPDUMIM4_hv',NATOM,intg=HPDUMIM4_hv)
              !       'HPDUMIM4')
              call chmdealloc('zerom2.src','ZMERGE','VARFIN_hv',NVAR,crl=VARFIN_hv)
#if KEY_BROKEN==1
              !
              call chmdealloc('zerom2.src','ZMERGE','PGRAD_hv',NVAR,intg=PGRAD_hv)
              call chmdealloc('zerom2.src','ZMERGE','PSGRAD_hv',NVAR,log=PSGRAD_hv)
              call chmdealloc('zerom2.src','ZMERGE','PTHTAN_hv',NVAR,crl=PTHTAN_hv)
              call chmdealloc('zerom2.src','ZMERGE','RMASS_hv',NVAR,crl=RMASS_hv)
              call chmdealloc('zerom2.src','ZMERGE','GPREV_hv',NVAR,crl=GPREV_hv)
#endif 
           ENDIF !if minimizing
           ! ******************************************************************
           ! ***************  end of first-order minimization *****************
           ! ******************************************************************

           if(QZTIME) then
            call seconds(etime,ctime)
            time_04 = etime
            cumetime_03 = cumetime_03 + etime - time_03
           endif
           CALL UPDECI(CURENPNT,X,Y,Z,WMAIN,0,&
                (/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
           if(QZTIME) then
            call seconds(etime,ctime)
            time_05 = etime
            cumetime_04 = cumetime_04 + etime - time_04
           endif
           CALL GETE(X, Y, Z, X, Y, Z, 0)
           ! post waits
!
!
           if(QZTIME) then
            call seconds(etime,ctime) 
            cumetime_05 = cumetime_05 + (etime-time_05)
           endif
!           do NPR = 1,NACTVE
!            WRITE(MYNODGP+20,*) 'number ',NPR,' atom ',ACTV(NPR)
!           enddo
!           call parstoperr('<ZMERGE>','printed out nbactive atoms ')
           !
           !        WRITE(6,*) '*** IN ZMERGE AFTER ENERGY CALL = ',EPROP(EPOT)
          goto 7777
#if KEY_PARALLEL==1
#if KEY_PARALLEL==1
          if(MYNODGP.eq.1) then  
#endif
#endif
           WRITE(6,*) 'TEST3 ELEC ',ETERM(ELEC)
           WRITE(6,*) 'TEST3 VDW ',ETERM(VDW)
           WRITE(6,*) 'TEST3 ASP ',ETERM(ASP)
           WRITE(6,*) 'TEST3 DIHE ',ETERM(DIHE)
           WRITE(6,*) 'TEST3 ANGL ',ETERM(ANGLE)
           WRITE(6,*) 'TEST3 BOND ',ETERM(BOND)
           WRITE(6,*) 'TEST3 SUM PE ',ETERM(ELEC) + ETERM(VDW) +  &
              ETERM(ASP) + ETERM(DIHE) + ETERM(ANGLE) + ETERM(BOND)
           WRITE(6,*)  
           DO II = 1,NPSUBS
            SUBSPA = WSUBSP(II)
            WRITE(6,*) 'SUBSPACE ',II,' CURCNF ',CURCNF(II) 
            do MSDLLN = LODOFL(CURCNF(II)),HIDOFL(CURCNF(II))
              CURDOF = MSTDOF(MSDLLN)
              CURVAL = MSTDFV(MSDLLN)
              WRITE(6,*) 'DOF ',CURDOF,' VAL ',CURVAL
            enddo 
           ENDDO
#if KEY_PARALLEL==1
          endif  
           call parstoperr('<ZMERGE>','printed out energy term for conf 1') 
#endif
!           STOP
7777       CONTINUE
           IF (EPROP(EPOT).LT.MINIME) THEN
              ZMINPT = TOTPNT
              MINIME= EPROP(EPOT)
              LOCMINPT = LOCGRDPT
              ! **************************************************************************
              !  since input is compressed, you have to explicitly save the entire minimum
              DO III = 1,NSSSLD
                 SUBSP = SSSLST(III) 
                 DO MSDLLN = LODOFL(LOCONF(SUBSP)), &
                      !     only need the lowest (complete) conformer
                      HIDOFL(LOCONF(SUBSP))
                    MASTDF = MSTDOF(MSDLLN)
                    ENTRYN  = LOICEN(MASTDF)
                    CALL GETICVL(icr_struct%B1ic, &
                         icr_struct%B2ic,icr_struct%T1ic, &
                         icr_struct%T2ic,icr_struct%PIC, &
                         HPLNLS_hv,HPCLST_hv, &
                         ENTRYN,CURVAL)
                    GMINVAL(MASTDF) = CURVAL
                    !             WRITE(6,'(I14,I14,I14,F14.7,1X,F19.7)') !temporary
                    !     &  SUBSP,1,MSTDOF(MSDLLN),GMINVAL(MASTDF),EPROP(EPOT)
                 ENDDO  !loop over DOF's
              ENDDO !loop over subspaces
              ! ************ end of saving minimum *************************************
           ENDIF !if calculating energies (QZENERG)
        ENDIF !if not out of bounds
        ! tests to determine what get saved/written out
        !if (.not. qzenerg), then qzncut and qzvcut must be .false.
              COUNT1 = COUNT1 + 1
!              WRITE(6,*) 'MYNODGP ',MYNODGP,' TOTPNT ',TOTPNT,' EPROP ',EPROP(EPOT)
        IF((.NOT.QZNCUT).OR.(EPROP(EPOT).LT.ZENCUT)) THEN 
           COUNT2 = COUNT2 + 1
           IF((.NOT.QZVCUT).OR.(EPROP(EPOT).LT.ZVACUT)) THEN
              COUNT3 = COUNT3 + 1
              !**************************************************************************
              ZNMEET = ZNMEET + 1
              ! for rmsd calculation, do reorientation if necessary
              IF(QZRMSD) THEN
                 DO NPR = 1,NACTVE
                    JJ = ACTV(NPR)
                    XCOPYS(NPR) = X(JJ)
                    YCOPYS(NPR) = Y(JJ)
                    ZCOPYS(NPR) = Z(JJ)
                    STEMPX(NPR) = XCOMP(JJ)
                    STEMPY(NPR) = YCOMP(JJ)
                    STEMPZ(NPR) = ZCOMP(JJ)
                    ATOMIN(1,NPR)=NPR
                    ATOMIN(2,NPR)=NPR
!                    WRITE(6,*) 'ATOM ',JJ,' XX ',X(JJ),' YY ',Y(JJ),' ZZ ',Z(JJ) 
!                    WRITE(6,*) 'INDEX ',NPR,' XCOPYS ',XCOPYS(NPR), &
!                         ' YCOPYS ',YCOPYS(NPR),' ZCOPYS ',ZCOPYS(NPR)
!                    WRITE(6,*) 'INDEX ',NPR,' STEMPX ',STEMPX(NPR), &
!                       ' STEMPY ',STEMPY(NPR),' STEMPZ ',STEMPZ(NPR)
                    !              WRITE(6,*) 'INDEX ',NPR,' ATOMIN ',ATOMIN(1,NPR)
                 ENDDO
                 !
                 IF(QZORIEN) THEN
                    CALL ROTLSQ(STEMPX,STEMPY,STEMPZ,NACTVE, &
                         XCOPYS,YCOPYS,ZCOPYS,NACTVE,ATOMIN,NACTVE,.FALSE., &
                         AMASS,AMASS,.FALSE.,WMAIN,.FALSE.,.FALSE.)
                 ENDIF !if reorienting
                 RMST1 = 0
                 DO ATM = 1,NACTVE
                    XXR1 = XCOPYS(ATM) - STEMPX(ATM)
                    YYR1 = YCOPYS(ATM) - STEMPY(ATM)
                    ZZR1 = ZCOPYS(ATM) - STEMPZ(ATM)
                    XXR1 = XXR1*XXR1
                    YYR1 = YYR1*YYR1
                    ZZR1 = ZZR1*ZZR1
                    RMST1=RMST1+XXR1+YYR1+ZZR1
                 ENDDO
                 RMST1 = RMST1/NPR
                 IF (RMST1.LT.RMSMN1) RMSMN1 = RMST1
                 !           WRITE(6,*) 'RMSSQ IS ',RMST1
              ENDIF !if calculating rmsd
              ! ************************************************************************
              ! save the minimum
              ! recut is problematic, because you really can't generate all the
              ! conformers from the grid, since it's serial access
              IF (QZENERG) THEN
                 MYPOTEN = EPROP(EPOT)
              ELSE
                 MYPOTEN = 0
              ENDIF
              IF(QRECUT) THEN
                 OKCCNT = OKCCNT + 1
                 !         WRITE(6,*) 'TOTPNT ',TOTPNT,' OKCCNT ',OKCCNT,' MIN ',
                 !     & MINIME
                 SVCNFI(OKCCNT) = TOTPNT
                 IF(OKCCNT.GT.NCSAVE) CALL WRNDIE(-5,'<ZMERGE>', &
                      'TOO MANY SAVED CONFORMERS. REDUCE ZSEArch SVFRaction')
              ELSE  !no recut--write out result 
#if KEY_PARALLEL==0
                NEWCONF = NEWCONF + 1
#else
                if(QZDATCOMM) then
                 NLOCCONF = NLOCCONF + 1
                else
                 NEWCONF = NEWCONF + 1
                endif
#endif
!                 WRITE(6,*) 'NEWCONF ',NEWCONF,' QWFIRST ',QWFIRS,' NLOCCONF ',NLOCCONF
                 IF(QZSWRIT) THEN !if we're supposed to write the conformer file
                    IF(QWFIRS) THEN
                       call ZWRFIRST(SSSLST,NSSSLD,LODOFL,HIDOFL,LOCONF,LOICEN,MSTDOF, &
                            MSTDFV,HPLNLS_hv,HPCLST_hv,ZWRUNI,NEWCONF,NSSTAG,MYPOTEN)
                       QWFIRS=.FALSE.
                    ELSE  !not qwfirs
                       IF (QWZCMP) THEN  !if compression
                          DO II = 1,NPSUBS
                             CCNFOR = CURCNF(II)
                             !            WRITE(6,*)
                             !     & 'II = ',II,' WSUBSP= ',WSUBSP(II),' CCNFOR ',
                             !     & CCNFOR
                             DO JJ = LODOFL(CCNFOR),HIDOFL(CCNFOR)
                                WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
                                     NSSTAG,NEWCONF,MSTDOF(JJ),MSTDFV(JJ),MYPOTEN
                             ENDDO
                          ENDDO
                       ELSE !if not compressing output
#if KEY_PARALLEL==1
                          COUNT = COUNT + 1  /*temporary*/
#endif
                          DO III = 1,NSSSLD
                             SUBSP = SSSLST(III)
                             IF(ALILST(SUBSP).EQ.0) THEN !write out if not an alias
                                DO MSDLLN = LODOFL(LOCONF(SUBSP)), &
                                     HIDOFL(LOCONF(SUBSP))
                                   !            WRITE(6,*) ' MSDLLN ',MSDLLN
                                   !            WRITE(6,*) ' CURDOF ',CURDOF,' CURVAL ',
                                   !     & CURVAL
                                   !            WRITE(6,*) ' LOICEN ',LOICEN(CURDOF),
                                   !     & ' HIICEN ',HIICEN(CURDOF)
                                   ENTRYN  = LOICEN(MSTDOF(MSDLLN))
                                   CALL GETICVL(icr_struct%B1ic, &
                                        icr_struct%B2ic,icr_struct%T1ic, &
                                        icr_struct%T2ic,icr_struct%PIC, &
                                        HPLNLS_hv,HPCLST_hv, &
                                        ENTRYN,CURVAL)
#if KEY_PARALLEL==1
!               execute if neighborhoods not defined or if there are 2 or more neighborhoods
                                   if(QZDATCOMM) then
                                    OUTLNCNT = OUTLNCNT + 1
                                    if(OUTLNCNT.GT.MAXLNCNT) then
                                     call parstoperr('<ZMERGE>','output too large, increase ZMEM MXLN')
                                    endif

                                    NSSTAG_AR(OUTLNCNT) = NSSTAG
                                    NEWCONF_AR(OUTLNCNT) = NLOCCONF
!                                    MSTDOF_AR(OUTLNCNT) = MSTDOFW(MSDLLN) 
                                    MSTDOF_AR(OUTLNCNT) = MSTDOF(MSDLLN) 
                                    CURVAL_AR(OUTLNCNT) = CURVAL
                                    MYPOTEN_AR(OUTLNCNT) = MYPOTEN
                                    RMST1_AR(OUTLNCNT) = RMST1 
                                   else  !if single neighborhood or single CPU, write out as if serial
#endif
                                    if(QZRMSD) THEN
                                      WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7,F19.7)') &
                                           NSSTAG,NEWCONF,MSTDOF(MSDLLN),CURVAL,MYPOTEN,RMST1
                                    else
                                      WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
                                           NSSTAG,NEWCONF,MSTDOF(MSDLLN),CURVAL,MYPOTEN
                                    endif
#if KEY_PARALLEL==1
                                   ENDIF
#endif
                                ENDDO  !loop over DOF's
                             ENDIF !if subspace is not an alias
                          ENDDO !loop over subspaces
                          !            WRITE(6,*) 'POT ENER   ELEC   VDW    DIHE   ANGL ',
                          !     & '   BOND    IMPR  '
                          !            WRITE(6,*) 'ZENER>', EPROP(EPOT),ETERM(ELEC),ETERM(VDW),
                          !     & ETERM(DIHE),ETERM(ANGLE),ETERM(BOND),ETERM(IMDIHE)
                          !  
                          !
                       ENDIF !if compressing output or not
                    ENDIF !if first conformation to be written or not
                 ENDIF !if we're writing out the conformer file
                 ! write out trajectory if requested
                 ! ***********************************************
                 !           SUBROUTINE WRITCV(X,Y,Z,
#if KEY_CHEQ==1
                 !     $                  CG,QCG,                             
#endif
                 !     $                  NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF,
                 !     $                  DELTA,NSAVC,NSTEP,TITLE,NTITLE,IUNCRD,QVEL,
                 !     $                  QINCT,JCNTRL,DIM4,FDIM)
                 !      DO II =2383,2410
                 !          WRITE(6,*) 'ATM ',II,' X ',X(II),' Y ',Y(II),' Z ',Z(II)
                 !      ENDDO
                 !
                 IF(QZWRTRJ) THEN
                    NAT3 = 3*NATOM
                    CALL WRITCV(X,Y,Z, &
#if KEY_CHEQ==1
                         (/ ZERO /), .FALSE., &  
#endif
                         NATOM, (/ 0 /), NATOM, 1, ZNMEET, NAT3, &
                         AREAL,1,ZTNSTEP,TITLEA,NTITLA,ZWRTUNI, &
                         .FALSE., .FALSE., (/ 0 /), .FALSE., (/ ZERO /))
                 ENDIF
                 !
              ENDIF !if recutting or not
           ELSE !if no tolerance from minimum or if under 
              ZNQVCUT = ZNQVCUT + 1
           ENDIF
           IF(QZVCUT) ZVACUT = MINIME + ZVATOL
        ELSE !if no energy cutoff or if under cutoff
           ZNQECUT = ZNQECUT + 1
        ENDIF
        ! if minimized, restore the unminimized coordinates
        IF(QZ1STMIN) THEN
           !         WRITE(6,*) 'NACTVE IS ',NACTVE
           DO III = 1,NACTVE
              ATMS = ACTV(III)
              X(ATMS) = SAVEXX(ATMS) 
              Y(ATMS) = SAVEYY(ATMS) 
              Z(ATMS) = SAVEZZ(ATMS) 
              !           WRITE(6,*) 'ATOM ',ATMS,' Z IS ',Z(ATMS)
           ENDDO
        ENDIF
        ! temporary:
        !        CALL UPDECI(1,X,Y,Z,WMAIN,0,0,0,0,0,0,0)
        !      CALL GETE(X, Y, Z, X, Y, Z, 0)
        !       WRITE(6,*) 'ENERGY AFTER RESTORING COORDINATES IS: ',
        !     & EPROP(EPOT)
        !
        !       WRITE(6,*) 'ZMERGE HAS DONE THE FOLLOWING: '
        !       DO KK = 1,NPSUBS
        !       WRITE(6,*) 'SS # ',KK,' SUBSPA ',WSUBSP(KK),
        !     & ' CURCNF ',CURCNF(KK)
        !       ENDDO
     ELSE !if not out of bounds
        ZNQOBND = ZNQOBND + 1
     ENDIF
  ENDDO  !loop over gridpoints
!  call parstoperr('ZMERGE',' after gridpoints ')
  if(QZTIME) call loc_timer('END OF GRID')
  
  if(QZTIME) then
#if KEY_PARALLEL==1
   NODEME = MYNODGP
#else
   NODEME = 1
#endif
   write(6,*) 'mynodgp ',NODEME,' cumetime_01 for donechanges ',cumetime_01  
   write(6,*) 'mynodgp ',NODEME,' cumetime_02 for build ',cumetime_02  
   write(6,*) 'mynodgp ',NODEME,' cumetime_03 for up to updeci ',cumetime_03
   write(6,*) 'mynodgp ',NODEME,' cumetime_04 for updeci itself ',cumetime_04
   write(6,*) 'mynodgp ',NODEME,' cumetime_05 for energies ',cumetime_05
   if(QCVTIME) write(6,*) 'mynodgp ',NODEME,' cumetime_cvt for CARTCV ',cumetime_cvt
   if(QCVTIME) write(6,*) 'mynodgp ',NODEME,' cumetime_cv for ind CARTCV ',cumetime_cv,' pass ',passbuild
!   write(6,*) 'cumetime from bycc ',cumetime_cc
  endif
   if(QVERBOSE) write(6,*) 'mynodgp ',NODEME,'********************* done with gridpoints************************'
   if(QVERBOSE) write(6,*) 'mynodgp ',NODEME,'********************* ********************************************'
#if KEY_PARALLEL==1
  if(NUMHOOD.eq.0) then
   NUMNOD = SAV_NUMNOD
   MYNODP = SAV_MYNODP
  endif
  if(NUMHOOD.gt.1) then
! restore the normal charmm values for parallel variables
   NUMNOD = SAV_NUMNOD
   MYNODP = SAV_MYNODP
   MYNOD = SAV_MYNOD
   COMM_CHARMM = SAV_COMM
!   write(6,*) 'mynodgp ',mynodgp,' mynodp ',mynodp,' COMM_CHARMM at bottom is ',COMM_CHARMM
  endif
#endif
!  call parstoperr('ZMERGE','after all gridpoints')

#if KEY_PARALLEL==1 /*(combinepara)*/
! if numhood=1, no communication necessary
  if(NUMHOOD.eq.1) goto 9999
  if(NUMHOOD.eq.0) then   !neighborhoods not defined; communicate to all nodes
   ALLCOMM = MPI_COMM_WORLD
   NUMNODES = NUMNOD
   MYHOOD = MYNODP
   MYLOCRNKP = MYNODP
  else if(NUMHOOD.GT.1) then  !neighborhoods defined, and > 1; communicate between head nodes
   ALLCOMM = HEADCOM
   NUMNODES = NUMHOOD
   MYHOOD = IMYNHDP 
   MYLOCRNKP = IMYKEYP
  endif
  if(QVERBOSE) then
   if(MYNODP.eq.NUMNOD.or.MYNODP.eq.1) then  !temporary
   WRITE(6,'(A,I5,5(A,I8))') 'MYNODP ',MYNODP,' TOTPNT ',TOTPNT,' CNT1 ',COUNT1,' CT2 ',COUNT2, &
   ' NLOCCONF ',NLOCCONF,' OUTLNCNT ',OUTLNCNT,' NUMNOD ',NUMNOD
   endif
  endif
!  if(pass.eq.3) then
!  call parstoperr('<ZMERGE>','finished 3rd loop ')
!  endif
! if parallel and calculating energies, find the global minimum
 if(QZENERG) then
   if(allocated(LOCMIN)) call chmdealloc('zerom2.src','ZMERGE','LOCMIN',NUMNODES,crl=LOCMIN)
   call chmalloc('zerom2.src','ZMERGE','LOCMIN',NUMNODES,crl=LOCMIN)
   LOCMIN = 0
   if(allocated(SENDMIN)) call chmdealloc('zerom2.src','ZMERGE','SENDMIN',NUMNODES,crl=SENDMIN)
   call chmalloc('zerom2.src','ZMERGE','SENDMIN',NUMNODES,crl=SENDMIN)
   SENDMIN(1) = MINIME
  if(QZTIME) call loc_timer('BEFORE ALLGATH')

  if(QVERBOSE) write(6,*) 'after timing before allgather'
!  call parstoperr('ZMERGE','before first allgather')
  if((MYLOCRNKP.eq.1).or.(NUMHOOD.eq.0)) then
   call MPI_ALLGATHER(SENDMIN,1,MPI_DOUBLE_PRECISION,LOCMIN,1,MPI_DOUBLE_PRECISION,ALLCOMM,ierr)
  endif
  if(QVERBOSE) write(6,*) 'after allgather'
  if(QZTIME) call loc_timer('AFTER ALLGATH')

   MINGLOB = 1.0D99
   do II = 1,NUMNODES
    if(LOCMIN(II).LT.MINGLOB) then
     MINGLOB = LOCMIN(II)
     MINNOD = II
    endif
   enddo
   MINIME = MINGLOB

! now communicate the minimum to all the nodes

   if(allocated(DFSRCHED)) call chmdealloc('zerom2.src','ZMERGE','DFSRCHED',DDEFME,intg=DFSRCHED)
   call chmalloc('zerom2.src','ZMERGE','DFSRCHED',DDEFME,intg=DFSRCHED)
   DFSRCHED = 0 

  if(QVERBOSE) write(6,*) 'after allocation of dfsrched'
!  call parstoperr('ZMERGE','after first allocating dfsrched')

! find the set of searched DOF
   DOFCNT = 0 
   do III = 1,NSSSLD
     SUBSP = SSSLST(III)
!     do MSDLLN = LODOFLW(LOCONF(SUBSP)), &
!          HIDOFLW(LOCONF(SUBSP))
!        MASTDF = MSTDOFW(MSDLLN)
      do MSDLLN = LODOFL(LOCONF(SUBSP)), HIDOFL(LOCONF(SUBSP))
        MASTDF = MSTDOF(MSDLLN)
        DFSRCHED(MASTDF) = 1
        DOFCNT = DOFCNT + 1
     enddo !over DOFs
   enddo     
! make a list of the searched DOF
   NSRCHED = 0 
   do II = 1,DDEFME
    if(DFSRCHED(II).eq.1) then
     NSRCHED = NSRCHED + 1
    endif
   enddo

   if(QVERBOSE) WRITE(6,*) 'MYNODGP ',MYNODGP,' DOFCNT ',DOFCNT,' NSRCHED ',NSRCHED
   if(DOFCNT.GT.NSRCHED) then
     call parstoperr('<ZMERGE>','WARNING: SOME DOFs DOUBLE-SEARCHED',pqstop=.false.,pqerror=.false.)
   endif
   if(allocated(SRCHDLST)) call chmdealloc('zerom2.src','ZMERGE','SRCHDLST',NSRCHED,intg=SRCHDLST)
   call chmalloc('zerom2.src','ZMERGE','SRCHDLST',NSRCHED,intg=SRCHDLST)
   if(allocated(SRCHDVAL)) call chmdealloc('zerom2.src','ZMERGE','SRCHDVAL',NSRCHED,crl=SRCHDVAL)
   call chmalloc('zerom2.src','ZMERGE','SRCHDVAL',NSRCHED,crl=SRCHDVAL)
   SRCHDLST=0
   SRCHDVAL=0

   NSRCHED = 0 
   do II = 1,DDEFME
     if(DFSRCHED(II).eq.1) then
      NSRCHED = NSRCHED + 1
      SRCHDLST(NSRCHED)=II
      SRCHDVAL(NSRCHED)=GMINVAL(II)
     endif
   enddo
   if(MYHOOD.ne.MINNOD) SRCHDVAL = 0

! broadcast the minimum values from the appropriate node

  if(QZTIME) call loc_timer('BEFORE ZM2 BCAST')

   if((MYLOCRNKP.eq.1).or.(NUMHOOD.eq.0)) then
   call MPI_BCAST(SRCHDVAL,NSRCHED,MPI_DOUBLE_PRECISION,MINNOD-1,ALLCOMM,ierr)
   endif
   
  if(QVERBOSE) WRITE(6,*) 'just after broadcast in ZMERGE'
  if(QZTIME) call loc_timer('AFTER ZM2 BCAST')

!  call parstoperr('ZMERGE','after mpi_bcast1')
   if(MYHOOD.ne.MINNOD) then
    do II = 1,NSRCHED
     THISDOF = SRCHDLST(II) 
!     WRITE(6,*) 'MYNODGP ',MYNODGP,' DOF ',THISDOF,' SRCHDVAL ',SRCHDVAL(II)
     GMINVAL(THISDOF)= SRCHDVAL(II)
    enddo
   endif
!   do II = 1,DDEFME
!    WRITE(6,*) 'MYNODGP ',MYNODGP,' DOF ',II,' VAL ',GMINVAL(II)
!   enddo
  
!  call parstoperr('ZMERGE','after gminval')
! if parallel and variable cutoff, reduce size of data locally
   if(QZVCUT) then
    ZVACUT = MINIME + ZVATOL
    
     call trim_conf(ZVACUT,NSSTAG_AR,NEWCONF_AR,MSTDOF_AR,CURVAL_AR,MYPOTEN_AR,RMST1_AR,OUTLNCNT)

   endif !QZVCUT
 endif !QZENERG
!    WRITE(6,*) 'MYNODGP ',MYNODGP,' NCOUNT2 ',NCOUNT2
!    NCOUNT = 0
!    NSSTAG_AR(II),NEWCONF_AR(II),MSTDOF_AR(II),CURVAL_AR(II),MYPOTEN_AR(II),RMST1_AR(II)
!
!    do II = 1,OUTLNCNT
!     
!    enddo
!
  if(QZTIME) call loc_timer('BEFORE COMBIN')

! call parstoperr('ZMERGE','before COMBIN')

!if parallel, combine conformer results
  if(QZTIME) then
    call seconds(etime,ctime)
    time_06 = etime
  endif
  call COMBIN_RESLT 

  if(QZTIME) then
    call seconds(etime,ctime)
    time_07 = etime
    write(6,*) 'mynodgp ',mynodgp,' cumetime_07 for combin_reslt ',time_07-time_06
  endif

!  call parstoperr('ZMERGE','after COMBIN')
!  call MPI_BARRIER(MPI_COMM_WORLD,ierr) 
  
  if(QZTIME) call loc_timer('after barrier, END of COMBIN')

!  TEMP(1) = NLOCCONF !temporary
!  call IGCOMB(TEMP,1)  !temporary
!  NLOCCONF = TEMP(1) !temporary
  if(QVERBOSE) write(6,*) 'MYNODP ',MYNODP,' MYNODGP ',MYNODGP,' writing out values, OUTLNCNT is ',OUTLNCNT
  if(MYNODP.eq.1) then
   do II = 1,OUTLNCNT
    if(QZRMSD) THEN
    write(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7,F19.7)') &
      NSSTAG_AR(II),NEWCONF_AR(II)+NEWCONF,MSTDOF_AR(II),CURVAL_AR(II),MYPOTEN_AR(II),RMST1_AR(II)
    else
    write(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
      NSSTAG_AR(II),NEWCONF_AR(II)+NEWCONF,MSTDOF_AR(II),CURVAL_AR(II),MYPOTEN_AR(II)
    endif
   enddo
   if(OUTLNCNT.gt.0) then
    LASTCNF = NEWCONF_AR(OUTLNCNT)+NEWCONF
   else
    LASTCNF = NEWCONF
   endif
  if(QVERBOSE) WRITE(6,*) 'MYNODGP ',MYNODGP,' after Lastcnf ',' confllen ',confllen

   do II = 1,CONFLLEN 
    if(QZRMSD) THEN
     write(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7,F19.7)') &
      NSSTAG_GLB(II),LASTCNF + NEWCONF_GLB(II),MSTDOF_GLB(II),CURVAL_GLB(II),MYPOTEN_GLB(II),RMST1_GLB(II)
    else
     write(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
      NSSTAG_GLB(II),LASTCNF + NEWCONF_GLB(II),MSTDOF_GLB(II),CURVAL_GLB(II),MYPOTEN_GLB(II)
    endif
   enddo 
   NEWCONF = LASTCNF + NEWCONF_GLB(CONFLLEN)
  endif  !mynodgp = 1
  TEMP = NEWCONF 
!  call MPI_BARRIER(MPI_COMM_WORLD,ierr)
  if(QZTIME) call loc_timer('zmerge after barr, bef BCAST')
! is this really necessary:?
  call MPI_BCAST(TEMP,1,MPI_INTEGER,0,MPI_COMM_WORLD,ierr)
!  call parstoperr('ZMERGE','after mpi_bcast2')
  NEWCONF = TEMP(1) 
#endif /*(combinepara)*/
!  call parstoperr('<ZMERGE>','printed out values',pqstop=.false.,pqerror=.false.)
  if(QZTIME) call loc_timer('END OF ZMERGE')
9999  CONTINUE

  PRNLEV = OLDLEV
  RETURN 
END SUBROUTINE ZMERGE

!--------------------------------------------------------------------
SUBROUTINE CHNGEICVL(B1IC,B2IC,T1IC,T2IC,PIC,LNLIST, &
     COLLST,ENTRYN,ICVALU)
  !
  !  changes ic table value at line LNLIST(ENTRYN)
  !  and for column COLLST(ENTRYN), which corresponds
  !  to ic type
  use chm_kinds
  use exfunc
  use dimens_fcm
  use parallel,only: MYNODP !temporary
  !
  implicit none

  real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
  INTEGER COLLST(*),LNLIST(*),ENTRYN
  real(chm_real) ICVALU
  !
  INTEGER I,J     
  !       WRITE(6,*) 'IN CHNGEICVL B1IC(1) is ',B1IC(1),
  !     & ' B2IC(1) is ',B2IC(1)
  !
  !
  IF (COLLST(ENTRYN).EQ.3) THEN
     PIC(LNLIST(ENTRYN)) = ICVALU
!       WRITE(MYNODP+100,*) 'CHANGING DIHE at line ',LNLIST(ENTRYN),  &
!          ' TO ',ICVALU
      
  ELSE IF (COLLST(ENTRYN).EQ.4) THEN
     !        WRITE(6,*) 'CHANGING ANGL at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU 
     T1IC(LNLIST(ENTRYN)) = ICVALU
  ELSE IF (COLLST(ENTRYN).EQ.2) THEN
     !        WRITE(6,*) 'CHANGING ANGL at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     T2IC(LNLIST(ENTRYN)) = ICVALU
  ELSE IF (COLLST(ENTRYN).EQ.5) THEN
     !        WRITE(6,*) 'CHANGING BOND at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     B1IC(LNLIST(ENTRYN)) = ICVALU
  ELSE IF (COLLST(ENTRYN).EQ.1) THEN
     !         WRITE(6,*) 'CHANGING BOND at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     B2IC(LNLIST(ENTRYN)) = ICVALU
  ENDIF
  !      DO I = 1,8
  !        J = I + 36
  !        WRITE(6,*) J,B2IC(J),T2IC(J),PIC(J),T1IC(J),B1IC(J)
  !      ENDDO   
  RETURN
END SUBROUTINE CHNGEICVL

! ---------------------------------------------------------
SUBROUTINE ZNBACTV(FLACT)
  !
  ! This routine generates the arrays of active
  ! atoms (ACTV),  groups (ACTVG), and clusters
  ! (ACLHI/JACLS) for a given atom selection,
  ! flagged as "1" in FLACT.
  ! Modified from SUBROUTINE NBACTV.
  !   - RJP
  !
  use chm_kinds
  use dimens_fcm
  use coord
  use psf
  use stream
  use inbnd
  use actclus_mod
  use memory
  implicit none

  ! . Passed variables.
  INTEGER FLACT(*)
  !
  !. Local var
  INTEGER I,J,TT,CNT
  INTEGER AAA,IS,IQ,NAT
  INTEGER RCOUNT,BB,ACC
  INTEGER ACTCAT
  LOGICAL INCLU
  integer :: sizea,II
  ! End of variable declarations
  !
  ! -----------------------------------------------
! deallocate if previously allocated
    if(allocated(ACTV)) then
     call chmdealloc('zerom2','ACTVDEF','ACTV',sizea,intg=ACTV)
     call chmdealloc('zerom2','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
     call chmdealloc('zerom2','ACTVDEF','JACLS',sizea,intg=JACLS)
     call chmdealloc('zerom2','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
     call chmdealloc('zerom2','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
     call chmdealloc('zerom2','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
     call chmdealloc('zerom2','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)
    endif

    sizea = 0
    DO II = 1,NATOM
     if(FLACT(II) > 0) THEN
       sizea = sizea + 1
     endif
    ENDDO

    call chmalloc('zerom2','ACTVDEF','ACTV',sizea,intg=ACTV)
    call chmalloc('zerom2','ACTVDEF','ACAFLG',NATOM,intg=ACAFLG)
    call chmalloc('zerom2','ACTVDEF','JACLS',sizea,intg=JACLS)
    call chmalloc('zerom2','ACTVDEF','ACLPTN',CLNUM,intg=ACLPTN)
    call chmalloc('zerom2','ACTVDEF','ACLHI',CLNUM,intg=ACLHI)
    call chmalloc('zerom2','ACTVDEF','ACTVG',NGRP,intg=ACTVG)
    call chmalloc('zerom2','ACTVDEF','ACGFLG',NGRP,intg=ACGFLG)
    ACTV = 0
    ACAFLG = 0
    JACLS = 0
    ACLPTN = 0
    ACLHI = 0
    ACTVG = 0
    ACGFLG = 0

  IF(PRNLEV.GE.2) WRITE (OUTU, '(A)') &
       ' CREATING ACTIVE ARRAYS '
  !    create active atom array
  NACTVE = 0
  DO I=1,NATOM
     IF (FLACT(I).GT.0) THEN
        NACTVE = NACTVE + 1
        ACTV(NACTVE) = I
     ENDIF
  ENDDO
  IF (NACTVE.EQ.0) CALL WRNDIE(-5,'<NBACTV>', &
       'NO ACTIVE ATOMS SELECTED')
  !  create active cluster array
  ACTCAT = 0
  NACTC = 0
  IF((.NOT.LGROUP).OR.(LVATOM)) THEN
     IF (CLNUM.LE.0) WRITE(OUTU,*) &
          '******** WARNING: NO CLUSTERS FOUND ********'
  ENDIF
  DO I=1,CLNUM
     IQ = CLHIGH(I)
     IS = IQ - CLPRTN(I) + 1
     NAT=CLPRTN(I)
     IF(NAT.LE.0) CALL WRNDIE(-5,'<NBACTV>', &
         'BAD NUMBER OF ATOMS')
     INCLU =.FALSE.
     CNT = 0
     DO TT = IS,IQ
        J = JCLUS(TT)
        IF (FLACT(J).GT.0) THEN
           IF(.NOT.INCLU) THEN
              NACTC = NACTC + 1
              INCLU = .TRUE.
           ENDIF
           CNT = CNT + 1
           ACTCAT = ACTCAT + 1
           JACLS(ACTCAT) = J
        ENDIF
     ENDDO
     IF (INCLU) THEN
        ACLPTN(NACTC) = CNT
        ACLHI(NACTC) = ACTCAT
     ENDIF
  ENDDO
  !   create active group array
  ! fill ACGFLG array and ACTVG array
  NACTG = 0
  DO I=1,NGRP
     IS=IGPBS(I)+1
     IQ=IGPBS(I+1)
     NAT=IQ-IS+1
     IF(NAT.LE.0) CALL WRNDIE(-5,'<NBACTV>', &
        'BAD NUMBER OF ATOMS')
     TT = IS
     INCLU = .FALSE.
     CNT = 0
     DO J = IS,IQ
        IF (FLACT(J).GT.0) THEN
           IF(.NOT.INCLU) THEN
              NACTG = NACTG + 1
              !    Added 1 line below -RJP
              ACGFLG(I) = 1
              ACTVG(NACTG) = I
              INCLU = .TRUE.
           ENDIF
        ENDIF
     ENDDO
  ENDDO
  IF(PRNLEV.GE.2) WRITE (OUTU,'(I6,A,I6,A,I6,A,A)') &
       NACTVE,' ATMS ',NACTC,' CLUS ',NACTG,' GRPS ', &
       'ARE ACTIVE '
  RETURN
END SUBROUTINE ZNBACTV
! --------------------------------------------------------
! -----------------------------------------------------------
SUBROUTINE TOTALGRID(SSSLST,SSSLS2,NTAKEN, &
     LOCONF,HICONF, &
     MAXPOS,POSIT,PMODE,ALLPNTS8,QTAKEN,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS)
  !
  !    calculates the total number of points on all grids
  !
  use chm_kinds
  use zdata_mod
  implicit none
  !
  INTEGER SSSLST(*),SSSLS2(*),NTAKEN
  INTEGER LOCONF(*),HICONF(*),MAXPOS(*),POSIT(*)
  INTEGER PMODE,ALLPNTS
  LOGICAL QTAKEN(*)
  INTEGER LOCSSI(*),HICSSI(*),COMBSS(*)
!  INTEGER NZGRPB(*)
  integer(chm_int8) :: RNZGRPB(*)
  integer(chm_int8) :: ALLPNTS8
  !
  INTEGER I,NSLEFT,SLOT,LOCPNTS,II 
  INTEGER NPSBSP,J,SUBS,NCONF,ALLSSP
  real(chm_real) FACT,RGRDPT
  LOGICAL DONECOMB
  integer(chm_int8) :: RLOCPNTS

  DO SLOT = 1,NTAKEN
     MAXPOS(SLOT) = NSSSLD - SLOT + 1  !maximum position
     POSIT(SLOT) = NTAKEN - SLOT + 1  !positional marker
     !        WRITE(6,*)'SLOT ',SLOT,' MX ',MAXPOS(SLOT),' POS ',
     !     & POSIT(SLOT)
  ENDDO

  ALLSSP = 0
  DO I = 1,ZNCOMB
     !
     RNZGRPB(I)=ALLPNTS8 !store total # of gridpts occuring before each grid
     DO II=1,NSSSLD
        QTAKEN(II)=.FALSE.
     ENDDO
     ! Note that at I = 1, the NTAKEN slots have initial positions
     ! and therefore the NTAKEN subspaces have already been specified
     ! at this point in loop
     RLOCPNTS = 1
     NPSBSP = 0
     J = NTAKEN
     DO WHILE(J.GE.1) !loop over the NTAKEN slots
        NPSBSP = NPSBSP + 1
        IF((PMODE.EQ.1).OR.(PMODE.EQ.3)) THEN
           SUBS = SSSLST(POSIT(J)) !selected subspace from list1
           !          WSUBSP(NPSBSP) = SSSLST(POSIT(J)) !selected subspace from list1
        ELSE IF(PMODE.EQ.2) THEN
           !          WSUBSP(NPSBSP) = SSSLS2(POSIT(J)) !selected subspace from list2
           SUBS = SSSLS2(POSIT(J)) !selected subspace from list2
        ELSE
           WRITE(6,*) 'WARNING: BAD PMODE '
        ENDIF
        NCONF = HICONF(SUBS) - LOCONF(SUBS) + 1
        ALLSSP = ALLSSP + 1
        COMBSS(ALLSSP) = SUBS
        RLOCPNTS = RLOCPNTS*NCONF
        QTAKEN(POSIT(J))=.TRUE.
        J = J - 1
     ENDDO  !(Loop over NTAKEN slots)
     IF(PMODE.EQ.3) THEN
        DO J = 1,NSSSLD
           IF(.NOT.(QTAKEN(J))) THEN
              NPSBSP = NPSBSP + 1
              SUBS = SSSLS2(J)
              ALLSSP = ALLSSP + 1
              COMBSS(ALLSSP) = SUBS
              NCONF = HICONF(SUBS) - LOCONF(SUBS) + 1
              RLOCPNTS = RLOCPNTS*NCONF
           ENDIF
        ENDDO
     ENDIF
     ALLPNTS8 = ALLPNTS8 + RLOCPNTS 
     !        WRITE(6,*) 'COMBO ',I,' RNZGRPB ',RNZGRPB(I)
     !        WRITE(6,*) 'RLOCPNTS ',RLOCPNTS
     !        WRITE(6,*) 'ALLPNTS NOW ',ALLPNTS
     ! ---------------------------------------------------
     ! set up the next grid point (next combination)
     SLOT = 1
     DONECOMB = .FALSE.
     DO WHILE ((SLOT.LE.NTAKEN).AND.(.NOT.DONECOMB))
        POSIT(SLOT) = POSIT(SLOT) + 1
        IF (POSIT(SLOT).GT.MAXPOS(SLOT)) THEN
           SLOT = SLOT + 1
        ELSE
           J = SLOT
           DO WHILE (J.GE.2)
              J = J - 1
              POSIT(J) = POSIT(J+1)+1
           ENDDO
           DONECOMB = .TRUE.
        ENDIF
     ENDDO !(setting up next gridpoint)
     HICSSI(I) = ALLSSP
     LOCSSI(I) = ALLSSP - NPSBSP + 1
  ENDDO ! (loop over all gridpoints)

  RETURN
END SUBROUTINE TOTALGRID
! ----------------------------------------------------
!
SUBROUTINE GETICVL(B1IC,B2IC,T1IC,T2IC,PIC,LNLIST, &
     COLLST,ENTRYN,ICVALU)
  !
  !  retrieves a single value from the ic table 
  !  (line LNLIST(ENTRYN), column COLLST(ENTRYN))
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  implicit none
  !
  real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
  INTEGER COLLST(*),LNLIST(*),ENTRYN
  real(chm_real) ICVALU
  !
  !
  !      WRITE(6,*) 'ENTRYN ',ENTRYN,' LNLIST ',LNLIST(ENTRYN),
  !     & ' COLLST ',COLLST(ENTRYN)
  IF (COLLST(ENTRYN).EQ.3) THEN
     ICVALU = PIC(LNLIST(ENTRYN)) 
     !      WRITE(6,*) 'CHANGING DIHE at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     !
  ELSE IF (COLLST(ENTRYN).EQ.4) THEN
     !       WRITE(6,*) 'CHANGING ANGL at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     ICVALU = T1IC(LNLIST(ENTRYN)) 
  ELSE IF (COLLST(ENTRYN).EQ.2) THEN
     !       WRITE(6,*) 'CHANGING ANGL at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     ICVALU = T2IC(LNLIST(ENTRYN)) 
  ELSE IF (COLLST(ENTRYN).EQ.5) THEN
     !       WRITE(6,*) 'CHANGING BOND at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     ICVALU = B1IC(LNLIST(ENTRYN)) 
  ELSE IF (COLLST(ENTRYN).EQ.1) THEN
     !        WRITE(6,*) 'CHANGING BOND at line ',LNLIST(ENTRYN),
     !     & ' TO ',ICVALU
     ICVALU = B2IC(LNLIST(ENTRYN)) 
  ENDIF
  RETURN
END SUBROUTINE GETICVL
!
! ------------------------------------------------------------------
SUBROUTINE ZWRFIRST(SSSLST,NSSSLD,LODOFL, &
     HIDOFL,LOCONF,LOICEN,MSTDOF,MSTDFV,LNLIST,COLLST, &
     ZWRUNI,NEWCONF,NSSTAG,MYPOTEN)
  !
  !   writes out all values of all dof's of the new subspace
  !
  use chm_kinds
  use energym
  use bases_fcm
  use intcor_module
  !
  implicit none
  !
  INTEGER SSSLST(*),LODOFL(*),HIDOFL(*)
  INTEGER LOCONF(*),LOICEN(*),MSTDOF(*)
  INTEGER LNLIST(*),COLLST(*),ZWRUNI
  real(chm_real) MSTDFV(*),MYPOTEN
  INTEGER NEWCONF,NSSSLD,NSSTAG
  !
  INTEGER SSSCNT,SUBSP,MSDLLN,LINENT
  real(chm_real) CURVAL
  !
  DO SSSCNT=1,NSSSLD
     SUBSP=SSSLST(SSSCNT)
     DO MSDLLN=LODOFL(LOCONF(SUBSP)), &
          HIDOFL(LOCONF(SUBSP))
        LINENT=LOICEN(MSTDOF(MSDLLN))
        CALL GETICVL(icr_struct%B1ic, &
             icr_struct%B2ic,icr_struct%T1ic, &
             icr_struct%T2ic,icr_struct%PIC, &
             LNLIST,COLLST, &
             LINENT,CURVAL)
        WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
             NSSTAG,NEWCONF,MSTDOF(MSDLLN),CURVAL,MYPOTEN
     ENDDO
  ENDDO
  RETURN
END SUBROUTINE ZWRFIRST
! -------------------------------------------------------------
!
subroutine zrecut(SSSLST,LOCONF,HICONF,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS,INDEX,SIABEG,SIAEND,INIATL, &
     NATOMX,LODOFL,HIDOFL,MSTDOF,MSTDFV,LOICEN,HIICEN, &
     SATBEG,SATEND,SUBATL,INIVAL,SAVEGR,MINIME, &
     MINCUT,NEWCONF,ZEBINS,QWFIRS)
  !
  !     regenerates a conformer from an index number
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  use energym
  use eutil
  use stream
  use intcor_module
  use actclus_mod
  use intcor2
  use heurist,only : UPDECI

  implicit none
  !
  INTEGER SSSLST(*),LOCONF(*),HICONF(*),INDEX
!  INTEGER NZGRPB(*),ZEBINS(*)
  INTEGER ZEBINS(*)
  integer(chm_int8) :: RNZGRPB(*)
  INTEGER ZMINPT,LOCSSI(*),HICSSI(*),COMBSS(*)
  INTEGER LODOFL(*),HIDOFL(*),MSTDOF(*)
  INTEGER LOICEN(*),HIICEN(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)

  INTEGER SATBEG(*),SATEND(*),SUBATL(*)
  INTEGER SAVEGR,NEWCONF
  real(chm_real) INIVAL(*),MINCUT,MINIME
  LOGICAL QWFIRS
  !
  INTEGER COPYI,IRATIO,GRIDNB,DOF
  INTEGER LOCATI,ENTRYN,JJ,LL,NATOMX,I,PREVSS
  real(chm_real) RNCONF,FINALR,MSTDFV(*),DIVISOR,CURVAL
  INTEGER SUBSPA,LO,HI,J,FINALG,II,III,JJJ
  INTEGER LOIN,HIIN,IDIFFE,MSDLLN,OLDLEV
  real(chm_real) DIFFER
  !
  ! ---------------------------------------------------------
  OLDLEV = PRNLEV
  PRNLEV = 2
  !
  ! this routine needs to be rethought, because we don't have random
  ! access to the conformers.
  ! figure out which grid we are on
  COPYI = INDEX 
  GRIDNB = 1 
  DO WHILE ((GRIDNB.LE.ZNCOMB).AND.(COPYI.GT.0)) 
     FINALR = COPYI
     FINALG = GRIDNB
     GRIDNB = GRIDNB + 1
     COPYI = INDEX - RNZGRPB(GRIDNB)
  ENDDO
  GRIDNB = FINALG
  ! reset conformers to initial structure if not same grid
  IF(GRIDNB.NE.SAVEGR) THEN
     !        WRITE(ZWRUNI,*) 'QWFIRS BEFORE CHANGE IS ',QWFIRS
     QWFIRS=.TRUE.
     !        WRITE(ZWRUNI,*) 'QWFIRS AFTER CHANGE IS ',QWFIRS
     DO II = 1,NSSSLD
        SUBSPA = SSSLST(II)
        ! change dof values in ic table
        JJ = LOCONF(SUBSPA)
        DO III = LODOFL(JJ),HIDOFL(JJ)
           JJJ = MSTDOF(III)
           CURVAL = INIVAL(JJJ)
           DO ENTRYN  = LOICEN(JJJ),HIICEN(JJJ)
              !
              CALL CHNGEICVL(icr_struct%B1ic, &
                   icr_struct%B2ic,icr_struct%T1ic, &
                   icr_struct%T2ic,icr_struct%PIC, &
                   HPLNLS_hv,HPCLST_hv, &
                   ENTRYN,CURVAL)
           ENDDO  !loop over ic entries
        ENDDO !loop over lines
        ! initialize atom positions
        DO JJ = SIABEG(SUBSPA),SIAEND(SUBSPA)
           X(INIATL(JJ)) = ANUM
           Y(INIATL(JJ)) = ANUM
           Z(INIATL(JJ)) = ANUM
        ENDDO
     ENDDO !loop over selected subspaces
     ! build
     CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR, &
          NATOMX)
  ENDIF !not same grid

  ! calculate the largest divisor
  DIVISOR = 1
  DO I = LOCSSI(GRIDNB),HICSSI(GRIDNB)-1 
     SUBSPA = COMBSS(I)
     RNCONF = HICONF(SUBSPA) - LOCONF(SUBSPA) + 1
     DIVISOR = DIVISOR*RNCONF
  ENDDO
  FINALR = FINALR - 1
  LOIN = LOCSSI(GRIDNB)
  HIIN = HICSSI(GRIDNB)
  I = HICSSI(GRIDNB)
  DO WHILE(I.GE.LOCSSI(GRIDNB)) !loop over subspaces 
     !       WRITE(6,*) 'I ',I,' DIVISOR ',DIVISOR
     SUBSPA = COMBSS(I)
     !       WRITE(6,*) 'INDEX ',I,' SUBSPA ',SUBSPA 
     IRATIO = INT(FINALR/DIVISOR)
     LOCATI = IRATIO + LOCONF(SUBSPA)
     !       WRITE(6,*) 'CONFORMER LOCATION IS ',LOCATI
     DO LL = LODOFL(LOCATI),HIDOFL(LOCATI)
        DOF = MSTDOF(LL)
        CURVAL = MSTDFV(LL)
        DO ENTRYN  = LOICEN(DOF),HIICEN(DOF)
           !
           CALL CHNGEICVL(icr_struct%B1ic, &
                icr_struct%B2ic,icr_struct%T1ic, &
                icr_struct%T2ic,icr_struct%PIC, &
                HPLNLS_hv,HPCLST_hv, &
                ENTRYN,CURVAL)
        ENDDO  !loop over ic entries
     ENDDO !loop over lines in conformer file
     DO JJ = SIABEG(SUBSPA),SIAEND(SUBSPA)
        X(INIATL(JJ)) = ANUM
        Y(INIATL(JJ)) = ANUM
        Z(INIATL(JJ)) = ANUM
     ENDDO

     FINALR = FINALR - IRATIO*DIVISOR
     IF(I.GT.LOCSSI(GRIDNB)) THEN
        PREVSS = COMBSS(I-1) 
        RNCONF = HICONF(PREVSS) - LOCONF(PREVSS)  + 1
     ENDIF
     DIVISOR = DIVISOR/RNCONF
     I = I - 1
  ENDDO !loop over selected subspaces
  ! build
  CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
       icr_struct%B1ic,icr_struct%B2ic, &
       icr_struct%T1ic,icr_struct%T2ic, &
       icr_struct%PIC, icr_struct%IAR, &
       icr_struct%JAR, icr_struct%KAR, &
       icr_struct%LAR, icr_struct%TAR, &
       NATOMX)
  !
  CALL UPDECI(1,X,Y,Z,WMAIN,0,(/zero/),(/zero/),(/zero/),(/zero/),(/zero/),(/zero/))
  CALL GETE(X, Y, Z, X, Y, Z, 0)
  !        WRITE(6,*) 'PE is ',EPROP(EPOT),' cutoff ',MINCUT
  IF(QZBINS) THEN
     DIFFER = (EPROP(EPOT)-MINIME)/ZBINSZ
     IDIFFE = INT(DIFFER) + 1
     IF(IDIFFE.LE.NZBINS) ZEBINS(IDIFFE) = ZEBINS(IDIFFE) + 1
  ENDIF
  IF (EPROP(EPOT).LE.MINCUT) THEN
     !        LOCALI = LOCALI + 1
     NEWCONF = NEWCONF + 1
     IF(QZSWRIT) THEN
        IF(QWFIRS) THEN !if first point in grid to be written
           ! this call below is incorrect, need to include poten energy, not "0"
           call ZWRFIRST(SSSLST,NSSSLD,LODOFL,HIDOFL,LOCONF,LOICEN,MSTDOF, &
                MSTDFV,HPLNLS_hv,HPCLST_hv,ZWRUNI,NEWCONF,NSSTAG,EPROP(EPOT))
           QWFIRS=.FALSE.
        ELSE
           IF(QWZCMP) THEN
              DO II = LOIN,HIIN
                 SUBSPA = COMBSS(II)
                 DO MSDLLN = LODOFL(LOCONF(SUBSPA)), &
                      HIDOFL(LOCONF(SUBSPA))
                    ENTRYN  = LOICEN(MSTDOF(MSDLLN))
                    CALL GETICVL(icr_struct%B1ic, &
                         icr_struct%B2ic,icr_struct%T1ic, &
                         icr_struct%T2ic,icr_struct%PIC, &
                         HPLNLS_hv,HPCLST_hv, &
                         ENTRYN,CURVAL)
                    WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
                         NSSTAG,NEWCONF,MSTDOF(MSDLLN),CURVAL,EPROP(EPOT)
                 ENDDO
              ENDDO
           ELSE !no compression
              DO II = 1,NSSSLD
                 SUBSPA = SSSLST(II)
                 DO MSDLLN = LODOFL(LOCONF(SUBSPA)), &
                      HIDOFL(LOCONF(SUBSPA))
                    ENTRYN  = LOICEN(MSTDOF(MSDLLN))
                    CALL GETICVL(icr_struct%B1ic, &
                         icr_struct%B2ic,icr_struct%T1ic, &
                         icr_struct%T2ic,icr_struct%PIC, &
                         HPLNLS_hv,HPCLST_hv, &
                         ENTRYN,CURVAL)
                    WRITE(ZWRUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
                         NSSTAG,NEWCONF,MSTDOF(MSDLLN),CURVAL,EPROP(EPOT)
                 ENDDO
              ENDDO
           ENDIF !if compression  of output or not 
        ENDIF !if first conformation to be written or not
     ENDIF !if writing output or not
  ENDIF
  !
  SAVEGR=GRIDNB
  PRNLEV = OLDLEV
  RETURN
END SUBROUTINE ZRECUT
!
! ----------------------------------------------------------
SUBROUTINE ZRECUTSET(WSUBSP,SSSLST,SVCNFI,OKCCNT, &
     LOCONF,HICONF,NATOM,LODOFL,HIDOFL,MSTDOF,MSTDFV, &
     LOICEN,HIICEN,NPSUBS,CURCNF,MINIME,ZEBINS, &
     NEWCONF)
  !
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  use contrl
  use energym
  implicit none
  !
  !  retests a set of saved conformations
  !
  INTEGER SSSLST(*),SVCNFI(*),OKCCNT,NPSUBS
  INTEGER WSUBSP(*),NEWCONF
  INTEGER LOCONF(*),HICONF(*),LODOFL(*),HIDOFL(*)
  INTEGER MSTDOF(*),NATOM,LOICEN(*),HIICEN(*)
  INTEGER CURCNF(*),ZEBINS(*)
  real(chm_real) MSTDFV(*),MINIME
  !
  ! locals
  INTEGER I,INDEX,II,SSSCNT,SUBSPA,MSDLLN,ENTRYN
  INTEGER IDIFFE,JJ,CCNFOR,SAVEGR,OLDFRE
  real(chm_real) CURVAL,DIFFER,MINCUT
  LOGICAL QWFIRS
  !      
  !
  MINCUT = MINIME + RECUTF
!  WRITE(6,'(A)') ' '
  WRITE(6,'(4x,A)')  &
       'REMOVING CONFORMERS WITH ENERGIES HIGHER THAN'
  WRITE(6,'(4x,F19.7,A9)') MINCUT,' KCAL/MOL'
  WRITE(6,'(4x,A17,F16.9,A13,F16.9)')  &
       'MIMIMUM ENERGY = ',MINIME,' TOLERANCE = ',RECUTF
  ! set INBFRQ to update list every step
  OLDFRE = INBFRQ
  INBFRQ = 1
  ! initialize energy bins
  DO I = 1,NZBINS
     ZEBINS(I) = 0
  ENDDO
  SAVEGR = 0
  ! loop over saved conformation indices 
  QWFIRS = .FALSE.
  DO I = 1,OKCCNT
     INDEX = SVCNFI(I)
     call ZRECUT(SSSLST,LOCONF,HICONF,HPZNCB_hv,HPLCSS_hv,HPHCSS_hv, &
          HPCBSS_hv,INDEX,HPSIAB_hv,HPSIAE_hv,HPINIL_hv,NATOM,LODOFL,HIDOFL, &
          MSTDOF,MSTDFV,LOICEN,HIICEN,HPSATB_hv,HPSATE_hv,HPSALS_hv,HPINVL_hv, &
          SAVEGR,MINIME,MINCUT,NEWCONF,ZEBINS,QWFIRS)
     !
  ENDDO
  ! reset old frequency
  INBFRQ = OLDFRE
  !
  RETURN
END SUBROUTINE ZRECUTSET
! -------------------------------------------------------------
! -------------------------------------------------------------
subroutine zsavegrmin(SSSLST,LOCONF,HICONF,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS,INDEX,SIABEG,SIAEND,INIATL, &
     NATOMX,LODOFL,HIDOFL,MSTDOF,MSTDFV,LOICEN,HIICEN, &
     GMINVAL,LDDSSMN)
  !
  !     stores the minimum conformer from the search explicitly
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  implicit none
  !
  INTEGER SSSLST(*),LOCONF(*),HICONF(*),INDEX
  integer(chm_int8) :: RNZGRPB(*)
  INTEGER ZMINPT,LOCSSI(*),HICSSI(*),COMBSS(*)
  INTEGER LODOFL(*),HIDOFL(*),MSTDOF(*)
  INTEGER LOICEN(*),HIICEN(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)
  real(chm_real) GMINVAL(*),LDDSSMN(*)  !grid and loaded subspace minima
  !
  INTEGER COPYI,IRATIO,GRIDNB,DOF
  INTEGER LOCATI,ENTRYN,JJ,MSDLLN,NATOMX,I,PREVSS
  real(chm_real) RNCONF,FINALR,MSTDFV(*),DIVISOR,CURVAL
  INTEGER SUBSPA,LO,HI,J,FINALG,III,MASTDF
  INTEGER CONFMR, DOFL !temporary
  !
  !   write some stuff out
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
  WRITE(6,*) 'SAVING GRID MINIMUM'
#if KEY_PARALLEL==1
  endif 
#endif
  ! for testing
  GOTO 9991
  WRITE(6,*) 'THIS IS THE DATA BEFORE MINIMUM: '
  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     DO CONFMR = LOCONF(SUBSPA),HICONF(SUBSPA)
        DO DOFL = LODOFL(CONFMR),HIDOFL(CONFMR)
           WRITE(6,*) 'SUBSPAC ',SUBSPA,' CONF ',CONFMR, &
                ' DOFL ',DOFL,' MSTDOF ',MSTDOF(DOFL)
        ENDDO
     ENDDO
  ENDDO
9991 CONTINUE
  !
  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     CONFMR = LOCONF(SUBSPA)
     DO MSDLLN = LODOFL(CONFMR),HIDOFL(CONFMR)
        !     only need lowest (complete) conformer
        MASTDF = MSTDOF(MSDLLN)
        LDDSSMN(MASTDF) = GMINVAL(MASTDF)
     ENDDO !loop over lines in conformer file
  ENDDO !loop over selected subspaces
  RETURN
END SUBROUTINE ZSAVEGRMIN
! -----------------------------------------------------------------------
!
! -----------------------------------------------------------------------
subroutine zregenminx(SSSLST,LOCONF,HICONF,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS,SIABEG,SIAEND,INIATL, &
     NATOMX,LODOFL,HIDOFL,MSTDOF,MSTDFV,LOICEN,HIICEN, &
     GMINVAL,MINENER,ZQICBF)
  !
  !     regenerates the minimum conformer explicitly from
  ! stored values, assigns the main structure to it,
  !  and prints the minimum if necessary
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  use intcor_module
  use actclus_mod
  use intcor2
  use memory
#if KEY_PARALLEL==1
  use parallel,only: MYNODP 
#endif
  implicit none
  !
  INTEGER SSSLST(*),LOCONF(*),HICONF(*),INDEX
  integer(chm_int8) :: RNZGRPB(*)
  INTEGER ZMINPT,LOCSSI(*),HICSSI(*),COMBSS(*)
  INTEGER LODOFL(*),HIDOFL(*),MSTDOF(*)
  INTEGER LOICEN(*),HIICEN(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)
  real(chm_real) GMINVAL(*),MINENER
  integer :: ZQICBF(*)
  integer,allocatable,dimension(:) :: INITFAR,INITRAR
  !
  INTEGER COPYI,IRATIO,GRIDNB,DOF
  INTEGER LOCATI,ENTRYN,JJ,MSDLLN,NATOMX,I,PREVSS
  real(chm_real) RNCONF,FINALR,MSTDFV(*),DIVISOR,CURVAL
  INTEGER SUBSPA,LO,HI,J,FINALG,III,MASTDF
  INTEGER CONFMR, DOFL !temporary
  INTEGER NINITF,NINITR
  !
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
  WRITE(6,'(A)') ' '
  WRITE(6,'(4x,A31)') 'ASSIGNING STRUCTURE TO MINIMUM '
  WRITE(6,'(4x,A18,F19.8)') 'ASSOCIATED ENERGY ',MINENER
#if KEY_PARALLEL==1
  endif 
#endif
  !   write some stuff out
  ! for testing
  GOTO 9991
  WRITE(6,*) 'THIS IS THE DATA BEFORE MINIMUM: '
  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     DO CONFMR = LOCONF(SUBSPA),HICONF(SUBSPA)
        DO DOFL = LODOFL(CONFMR),HIDOFL(CONFMR)
           WRITE(6,*) 'SUBSPAC ',SUBSPA,' CONF ',CONFMR, &
                ' DOFL ',DOFL,' MSTDOF ',MSTDOF(DOFL)
        ENDDO
     ENDDO
  ENDDO
9991 CONTINUE
   if(allocated(INITRAR)) then
    call chmdealloc('zerom2.src','ZREGENX','INITRAR',TINITA,intg=INITRAR)
    call chmdealloc('zerom2.src','ZREGENX','INITFAR',TINITA,intg=INITFAR)
   endif
   call chmalloc('zerom2.src','ZREGENX','INITRAR',TINITA,intg=INITRAR)
   call chmalloc('zerom2.src','ZREGENX','INITFAR',TINITA,intg=INITFAR)
   INITRAR = 0
   INITFAR = 0
  !
10 FORMAT(I14,I14,I14,F14.7)
  NINITF = 0
  NINITR = 0
  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     CONFMR = LOCONF(SUBSPA)
     DO MSDLLN = LODOFL(CONFMR),HIDOFL(CONFMR)
        !     only need lowest (complete) conformer
        MASTDF = MSTDOF(MSDLLN)
        CURVAL = GMINVAL(MASTDF)
        IF(QMINPRNT) WRITE(ZMINUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
             SUBSPA,CONFMR,MASTDF,CURVAL,MINENER
#if KEY_PARALLEL==1
        if(MYNODP.eq.1) then  
#endif
        WRITE(6,'(A7,I8,A5,I8,A5,I8,A5,F14.7)')  &
         ' SUBSP ',SUBSPA,' CNF ',CONFMR,' DOF ',MASTDF,' VAL ',CURVAL
#if KEY_PARALLEL==1
        endif 
#endif
        !            WRITE(6,*) ' LOICEN ',LOICEN(CURDOF),
        !     & ' HIICEN ',HIICEN(CURDOF)
        !         WRITE(6,*) 'CURVAL IS ',CURVAL
        DO ENTRYN  = LOICEN(MASTDF),HIICEN(MASTDF)
           !          WRITE(6,*) 'ENTRYN IS ',ENTRYN
           !
           CALL CHNGEICVL(icr_struct%B1ic, &
                icr_struct%B2ic,icr_struct%T1ic, &
                icr_struct%T2ic,icr_struct%PIC, &
                HPLNLS_hv,HPCLST_hv, &
                ENTRYN,CURVAL)
        ENDDO  !loop over ic entries
     ENDDO !loop over lines in conformer file
! do we really need this?:
!     ! store atom positions to be initialized
!     do JJ = SIABEG(SUBSPA),SIAEND(SUBSPA)
!        ! store the atoms for the reverse ic build
!        if(ZQICBF(SUBSPA).EQ.0) THEN
!           NINITR = NINITR + 1
!           INITRAR(NINITR) = INIATL(JJ)
!           ! store the atoms for the forward ic build
!        else if (ZQICBF(SUBSPA).EQ.1) THEN
!           NINITF = NINITF + 1
!           INITFAR(NINITF) = INIATL(JJ)
!        endif
!     enddo
  ENDDO !loop over selected subspaces
  !      WRITE(6,*) 'IN REGEN MIN, NINITF is ',NINITF
  !      WRITE(6,*) 'IN REGEN MIN, NINITR is ',NINITR
  ! build
  !      WRITE(6,*) 'IN REGEN MIN, NINITF is ',NINITF
  !      WRITE(6,*) 'IN REGEN MIN, NINITR is ',NINITR
  CALL BILDCFR(NINITF,NINITR,INITFAR,INITRAR,NATOMX)

  call chmdealloc('zerom2.src','ZREGENX','INITRAR',TINITA,intg=INITRAR)
  call chmdealloc('zerom2.src','ZREGENX','INITFAR',TINITA,intg=INITFAR)

  RETURN
END SUBROUTINE ZREGENMINX
!
! -----------------------------------------------------------------------
! -----------------------------------------------------------------------
subroutine zminprint(SSSLST,LOCONF,HICONF,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS,SIABEG,SIAEND,INIATL, &
     NATOMX,LODOFL,HIDOFL,MSTDOF,MSTDFV,LOICEN,HIICEN, &
     GMINVAL,MINENER)
  !
  !     regenerates the minimum conformer explicitly from
  ! stored values and prints it (no assignment is done)
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  implicit none
  !
  INTEGER SSSLST(*),LOCONF(*),HICONF(*),INDEX
  integer(chm_int8) :: RNZGRPB(*)
  INTEGER ZMINPT,LOCSSI(*),HICSSI(*),COMBSS(*)
  INTEGER LODOFL(*),HIDOFL(*),MSTDOF(*)
  INTEGER LOICEN(*),HIICEN(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)
  real(chm_real) GMINVAL(*),MINENER
  !
  INTEGER COPYI,IRATIO,GRIDNB,DOF
  INTEGER LOCATI,ENTRYN,JJ,MSDLLN,NATOMX,I,PREVSS
  real(chm_real) RNCONF,FINALR,MSTDFV(*),DIVISOR,CURVAL
  INTEGER SUBSPA,LO,HI,J,FINALG,III,MASTDF
  INTEGER CONFMR, DOFL !temporary
  !
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
  WRITE(6,'(A)') ' '
  WRITE(6,'(4x,A31)') 'ASSIGNING STRUCTURE TO MINIMUM '
  WRITE(6,'(4x,A18,F19.8)') 'ASSOCIATED ENERGY ',MINENER
#if KEY_PARALLEL==1
  endif 
#endif
  !   write some stuff out
  ! for testing
  GOTO 9991
  WRITE(6,*) 'THIS IS THE DATA BEFORE MINIMUM: '

  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     DO CONFMR = LOCONF(SUBSPA),HICONF(SUBSPA)
        DO DOFL = LODOFL(CONFMR),HIDOFL(CONFMR)
           WRITE(6,*) 'SUBSPAC ',SUBSPA,' CONF ',CONFMR, &
                ' DOFL ',DOFL,' MSTDOF ',MSTDOF(DOFL)
        ENDDO
     ENDDO
  ENDDO
9991 CONTINUE
  !
10 FORMAT(I14,I14,I14,F14.7)
  DO III = 1,NSSSLD
     SUBSPA = SSSLST(III)
     CONFMR = LOCONF(SUBSPA)
     DO MSDLLN = LODOFL(CONFMR),HIDOFL(CONFMR)
        !     only need lowest (complete) conformer
        MASTDF = MSTDOF(MSDLLN)
        CURVAL = GMINVAL(MASTDF)
        IF(QMINPRNT) WRITE(ZMINUNI,'(I14,I14,I14,F14.7,1X,F19.7)') &
             SUBSPA,CONFMR,MASTDF,CURVAL,MINENER
     ENDDO !loop over lines in conformer file
  ENDDO !loop over selected subspaces
  RETURN
END SUBROUTINE ZMINPRINT
! -----------------------------------------------------------------------
! -----------------------------------------------------------------------
subroutine zregenmin(SSSLST,LOCONF,HICONF,RNZGRPB, &
     LOCSSI,HICSSI,COMBSS,INDEX,SIABEG,SIAEND,INIATL, &
     NATOMX,LODOFL,HIDOFL,MSTDOF,MSTDFV,LOICEN,HIICEN)
  !
  !     regenerates the minimum conformer from an index number
  use chm_kinds
  use dimens_fcm
  use zdata_mod
  use bases_fcm
  use coord
  use number
  use intcor_module
  use intcor2
  implicit none
  !
  INTEGER SSSLST(*),LOCONF(*),HICONF(*)
  integer(chm_int8),intent(in) :: INDEX
!  INTEGER NZGRPB(*)
  real(chm_real) :: RNZGRPB(*)
  INTEGER ZMINPT,LOCSSI(*),HICSSI(*),COMBSS(*)
  INTEGER LODOFL(*),HIDOFL(*),MSTDOF(*)
  INTEGER LOICEN(*),HIICEN(*)
  INTEGER SIABEG(*),SIAEND(*),INIATL(*)
  !
  INTEGER IRATIO,GRIDNB,DOF
  integer(chm_int8) :: COPYI
  INTEGER LOCATI,ENTRYN,JJ,LL,NATOMX,I,PREVSS
  real(chm_real) RNCONF,FINALR,MSTDFV(*),DIVISOR,CURVAL
  INTEGER SUBSPA,LO,HI,J,FINALG
  INTEGER CONF, DOFL, III !temporary
  !
  !   write some stuff out
  WRITE(6,*) 'THIS IS THE DATA BEFORE MINIMUM: '
  DO III = 1,2
     SUBSPA = SSSLST(III)
     DO CONF = LOCONF(SUBSPA),HICONF(SUBSPA)
        DO DOFL = LODOFL(CONF),HIDOFL(CONF)
           WRITE(6,*) 'SUBSPAC ',SUBSPA,' CONF ',CONF, &
                ' DOFL ',DOFL,' MSTDOF ',MSTDOF(DOFL)
        ENDDO
     ENDDO
  ENDDO
  ! figure out which grid we are on
  WRITE(6,*) 'INDEX IS ',INDEX
  COPYI = INDEX 
  GRIDNB = 1 
  DO WHILE ((GRIDNB.LE.ZNCOMB).AND.(COPYI.GT.0)) 
     FINALR = COPYI
     FINALG = GRIDNB
     GRIDNB = GRIDNB + 1
     COPYI = INDEX - RNZGRPB(GRIDNB)
  ENDDO
  GRIDNB = FINALG
  WRITE(6,*) 'MIN IS ON GRID NUMBER ',GRIDNB
  !      
  ! calculate the largest divisor
  DIVISOR = 1
  DO I = LOCSSI(GRIDNB),HICSSI(GRIDNB)-1 
     SUBSPA = COMBSS(I)
     RNCONF = HICONF(SUBSPA) - LOCONF(SUBSPA) + 1
     DIVISOR = DIVISOR*RNCONF
  ENDDO
  FINALR = FINALR - 1
  !      DO GRIDNB = 1,ZNCOMB
  !        LO = LOCSSI(GRIDNB)
  !        HI = HICSSI(GRIDNB)
  !        DO J = LO,HI
  !          WRITE(6,*) 'GRID ',GRIDNB,' SS ',COMBSS(J) 
  !        ENDDO
  !      ENDDO
  I = HICSSI(GRIDNB)
  DO WHILE(I.GE.LOCSSI(GRIDNB)) !loop over subspaces 
     SUBSPA = COMBSS(I)
!     WRITE(6,*) 'SUBSPACE IS ',SUBSPA
     IRATIO = INT(FINALR/DIVISOR)
     LOCATI = IRATIO + LOCONF(SUBSPA)
     WRITE(6,*) 'CONFORMER NUMBER IS: ',LOCATI
     WRITE(6,*) 'LODOFL ',LODOFL(LOCATI), &
          ' HIDOFL ',HIDOFL(LOCATI)
     DO LL = LODOFL(LOCATI),HIDOFL(LOCATI)
        WRITE(6,*) 'DOF LOCATOR IS ',LL
        DOF = MSTDOF(LL)
        CURVAL = MSTDFV(LL)
!        WRITE(6,*) 'DOF LOCATOR IS ',LL,' DOF ',DOF, &
!             ' CURVAL ',CURVAL
        DO ENTRYN  = LOICEN(DOF),HIICEN(DOF)
           WRITE(6,*) 'ENTRYN IS ',ENTRYN
           CALL CHNGEICVL(icr_struct%B1ic, &
                icr_struct%B2ic,icr_struct%T1ic, &
                icr_struct%T2ic,icr_struct%PIC, &
                HPLNLS_hv,HPCLST_hv, &
                ENTRYN,CURVAL)
!           WRITE(6,*) ' CHANGED DOF ',DOF,' ENTR ',ENTRYN, &
!                ' TO ',CURVAL
        ENDDO  !loop over ic entries
     ENDDO !loop over lines in conformer file
     !
     DO JJ = SIABEG(SUBSPA),SIAEND(SUBSPA)
        X(INIATL(JJ)) = ANUM
        Y(INIATL(JJ)) = ANUM
        Z(INIATL(JJ)) = ANUM
     ENDDO
     FINALR = FINALR - IRATIO*DIVISOR
     IF(I.GT.LOCSSI(GRIDNB)) THEN
        PREVSS = COMBSS(I-1) 
        RNCONF = HICONF(PREVSS) - LOCONF(PREVSS)  + 1
     ENDIF
     DIVISOR = DIVISOR/RNCONF
     I = I - 1
  ENDDO !loop over selected subspaces
  ! build
  CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
       icr_struct%B1ic,icr_struct%B2ic, &
       icr_struct%T1ic,icr_struct%T2ic, &
       icr_struct%PIC, icr_struct%IAR, &
       icr_struct%JAR, icr_struct%KAR, &
       icr_struct%LAR, icr_struct%TAR, &
       NATOMX)
#if KEY_PARALLEL==1
  if(MYNODP.eq.1) then 
#endif
  WRITE(6,'(A)') ' '
  WRITE(6,'(4x,A31)') 'ASSIGNING STRUCTURE TO MINIMUM '
#if KEY_PARALLEL==1
  endif 
#endif
  RETURN
END SUBROUTINE ZREGENMIN
!    
subroutine bildcfr(NINITF,NINITR,INITFAR,INITRAR,NATOM,ICACTVF,NICACTVF,ICACTVR, &
      NICACTVR,PNOTEST,CUMETIME_CV)
  ! calls bildc in both forward and reverse senses
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use stream
  use coord
  use number
  use zdata_mod
  use intcor2
  use intcor_module
#if KEY_PARALLEL==1
  use parallel  /*temporary*/
#endif
  use nbndcc_utilb,only: parstoperr
  implicit none
  !
  ! does initialization and ic build in both directions
  ! for passed arrays of atoms
  !
  ! passed variables
  integer,intent(in) :: NINITF,NINITR,NATOM
  integer,dimension(:),intent(in) :: INITFAR, INITRAR
  ! local var
  INTEGER III
  integer :: pass=1
  real(chm_real),intent(out),optional :: cumetime_cv
  integer,dimension(:),intent(in),optional :: ICACTVF,ICACTVR
  integer,intent(in),optional :: NICACTVF,NICACTVR
  logical,intent(in),optional :: PNOTEST
!local
  real(chm_real):: CUMETIME
  logical :: QNOTEST
!----------------------------
#if KEY_PARALLEL==1
  if(QVERBOSE)  write(6,*) 'BUILDING ',NINITF,' ATOMS FORWARD, MYNODP ',MYNODP,' NATOM ',NATOM
#endif
  QNOTEST=.false.
  if(present(PNOTEST)) QNOTEST=PNOTEST 
  !
  pass = pass + 1
  IF(NINITF.GT.0) THEN
     DO III = 1,NINITF
        X(INITFAR(III)) = ANUM
        Y(INITFAR(III)) = ANUM
        Z(INITFAR(III)) = ANUM
     ENDDO
     QICBREV  = .FALSE. !in zerom.f90
     IF(.NOT.QICFILL) CALL WRNDIE(-5, &
          '<ZCOMBO>',' IC TABLE EMPTY (USE IC FILL) ')
!     call BILDC(1,icr_struct%lenic,X,Y,Z, &
     if(present(ICACTVF)) then
       call BILDC(1,NICACTVF,X,Y,Z, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR, &
          NATOM,ICACTV=ICACTVF,pnotest=QNOTEST,cumetime=CUMETIME)
     else
       call BILDC(1,icr_struct%lenic,X,Y,Z, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR, &
          NATOM,pnotest=QNOTEST,cumetime=CUMETIME)
     endif
  ENDIF
  if(present(cumetime_cv)) cumetime_cv = cumetime
  ! build in reverse
   if(NINITR.GT.0) THEN
     DO III = 1,NINITR
        X(INITRAR(III)) = ANUM
        Y(INITRAR(III)) = ANUM
        Z(INITRAR(III)) = ANUM
     ENDDO
     QICBREV  = .TRUE.
     IF(.NOT.QICFILL) CALL WRNDIE(-5, &
          '<ZCOMBO>',' IC TABLE EMPTY (USE IC FILL) ')
     CALL BILDC(1,icr_struct%lenic,X,Y,Z, &
          icr_struct%B1ic,icr_struct%B2ic, &
          icr_struct%T1ic,icr_struct%T2ic, &
          icr_struct%PIC, icr_struct%IAR, &
          icr_struct%JAR, icr_struct%KAR, &
          icr_struct%LAR, icr_struct%TAR, &
          NATOM,cumetime=cumetime)
   endif
!  call printcoor(NATOM,X,Y,Z,MYNODP)
  RETURN
END SUBROUTINE BILDCFR
!
SUBROUTINE WRITEVARB(VARBY,NVAR)
  use chm_kinds
  implicit none
  !
  INTEGER NVAR,II
  real(chm_real) VARBY(*)
  !
  !      WRITE(6,*) '&&&&&&&&&&& WRITING OUT VARB BEFORE CALL STEEP2 &&&&'
  WRITE(6,'(A,I4,1x,A,1x,F20.16)') 'VAR ',76,' VARB ',VARBY(76)
  !      DO II = 1,NVAR
  !       WRITE(6,'(A,I4,1x,A,1x,F20.16)') 'VAR ',II,' VARB ',VARBY(II)
  !      ENDDO
  !
  !      SUBROUTINE WRITEGRAD(GRAD,NVAR)
  !...  use chm_kinds
  !      implicit none
  !C
  !      INTEGER NVAR,II
  !      real(chm_real) GRAD(*)
  !C
  !      WRITE(6,*) '&&&&&&&&&&& WRITING OUT VARB BEFORE CALL STEEP2 &&&&'
  !      WRITE(6,*) 'GRAD(2)= ',GRAD(2)
  !C      DO II = 1,NVAR
  !       WRITE(6,'(A,I4,1x,A,1x,F20.16)') 'VAR ',II,' VARB ',VARBY(II)
  !      ENDDO
  !  
  RETURN
END SUBROUTINE WRITEVARB
!
SUBROUTINE WRITEIC(B1IC,B2IC,T1IC,T2IC,PIC, &
     IAR,JAR,KAR,LAR,TAR)
  !
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  implicit none
  !
  real(chm_real) B1IC(*),B2IC(*),T1IC(*),T2IC(*),PIC(*)
  INTEGER  IAR(*),JAR(*),KAR(*),LAR(*)
  LOGICAL TAR(*)
  INTEGER IC
  WRITE(6,*) 'B1IC(1) ',B1IC(1),' B2IC(1) ',B2IC(1)
  WRITE(6,*) 'IAR(1) ',IAR(1),' JAR(1) ',JAR(1)
  WRITE(6,*) 'IAR(2) ',IAR(2),' JAR(2) ',JAR(2)
  IC = 2455
  WRITE(6,*) 'IAR(2455) ',IAR(IC),' JAR(2455) ',JAR(IC)
  RETURN
END SUBROUTINE WRITEIC
!
SUBROUTINE WRITEICSET
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use zdata_mod
  use consta
  use psf
  use bases_fcm
  use ctitla
  use stream
#if KEY_PARALLEL==1
  use parallel 
#endif
  use intcor_module
  implicit none
  ! 
  CALL WRITEIC(ICR_STRUCT%B1ic,ICR_STRUCT%B2ic, &
       ICR_STRUCT%T1ic,ICR_STRUCT%T2ic, &
       ICR_STRUCT%PIC,ICR_STRUCT%IAR, &
       ICR_STRUCT%JAR,ICR_STRUCT%KAR, &
       ICR_STRUCT%LAR,ICR_STRUCT%TAR)
  !
  RETURN
END SUBROUTINE WRITEICSET
!
SUBROUTINE WRITEIMINOR(IMINORSS)
  !
  use chm_kinds
  implicit none
  INTEGER IMINORSS(*)
  !      WRITE(6,*) 'IMINORSS of 1 is ',IMINORSS(1)
  return
end SUBROUTINE WRITEIMINOR

! -------------------------------------------------------
 subroutine zbinswri(ZEBINS,ALLPNTS8)
  !
  use chm_kinds
  use zdata_mod
  use stream
  implicit none
  !  writes out energy bins
  !
  integer(chm_int8) :: ALLPNTS8
  INTEGER ZEBINS(*),ALLPNTS
  INTEGER I,TOTCNF
  real(chm_real) ZENERB,ZENERE,ZFRACT,RTOTCNF,RALLPTS
  !
  TOTCNF = 0
  WRITE(OUTU,*) 'ENERGY DISTRIBUTION '
  do I = 1,NZBINS
     ZENERB = (I-1)*ZBINSZ
     ZENERE = I*ZBINSZ
     WRITE(6,'(A4,I8,A3,I12,A14,F11.4,A5,F11.4,A9)') &
          'BIN ',I,' : ',ZEBINS(I), &
          ' CONFS BETW ', &
          ZENERB,' AND ',ZENERE, ' KCAL/MOL'
     TOTCNF = TOTCNF + ZEBINS(I)
  enddo
  WRITE(OUTU,*) '  ABOVE MINIMUM'
  WRITE(OUTU,*) ' '
  RTOTCNF = TOTCNF
  ZFRACT = 100*(RTOTCNF/RALLPTS)
  WRITE(OUTU,'(I14,A14,I18,A4)')  &
       TOTCNF,' CONFS OUT OF ',ALLPNTS8,' OR '
  WRITE(OUTU,'(2X,F9.5,A23,F11.4,A20)')  &
       ZFRACT,' % of CONFS ARE WITHIN ', &
       ZBINSZ*NZBINS,' KCAL/MOL OF MINIMUM'
  return
 end subroutine zbinswri

!-----------------------------------------------------------------------------------------------
#if KEY_PARALLEL==1 /*zpart*/
  subroutine zpart_grid(NSUB,NC,BEGPT,ENDPT,BEGINME,NPROC,QALLPROCP,BEGCONF,MYNODPX)
! partitions grid over processors
  use memory
#if KEY_PARALLEL==1
  use nbndcc_utilb,only: parstoperr  /*temporary */
#endif
  implicit none

  integer,dimension(:),intent(in) :: NC !number of conformers in each subspace in grid
  integer(chm_int8),dimension(:),intent(out) :: BEGPT,ENDPT  !dimensioned over number of processors
  integer(chm_int8),dimension(:) :: BEGINME !starting point of subspaces in my node
  integer(chm_int8),dimension(:,:),intent(out),optional :: BEGCONF  !results over all processors,subspaces
  logical,intent(in),optional :: QALLPROCP !true if resulting for all processors (BEGCONF)
  integer,intent(in) :: NSUB !number of subspaces in grid
  integer,intent(in) :: MYNODPX !number of mynod (or neighborhood)
! local
  integer,intent(in) :: NPROC !number of processors
  integer :: SS
  integer(chm_int8) :: PP,PART,TOT,FAC,SUM,VAL,NUM
  integer(chm_int8),dimension(:),allocatable :: CPC !dimensioned nsub
!  real*8 :: DIV
  integer(chm_int8) :: DIV
  logical :: QALLPROC,QATTOT=.false.
  real(chm_real) :: RTOT

  if(allocated(CPC)) call chmdealloc('zerom2.src','ZMERGE','CPC',NSUB,ci8=CPC)
  call chmalloc('zerom2.src','ZMERGE','CPC',NSUB,ci8=CPC)

  QALLPROC = .false.
  if(present(QALLPROCP)) QALLPROC=QALLPROCP
!  NPROC = 16384

  if(MYNODGP.eq.1) then
   WRITE(6,'(4X,A34,1X,I8,1X,A13)') '*****Partitioning calculation into',NPROC,'sections*****'
  endif
  !real*8 :: 
!  NPROC = 16384
!  NSUB = 2
!  NC(1) = 100000 
!  NC(2) = 1000000
!  NC(3) = 2 
!  NC(4) = 7
!  NC(5) = 3
  TOT = 1
  FAC = 1
  do SS = 1,NSUB
    TOT = TOT*NC(SS) 
    CPC(SS) = FAC   !conformers per click (how many conformers subsumed by increase of 1 in this subspace)
!    write(6,*) 'subspace ',SS,' #conf per click ',CPC(SS)
    FAC = FAC*NC(SS)
  enddo
  
  if(TOT.LT.NPROC) then
   call parstoperr('<ZPART_GRID>','# CONFMRS ON A GRID CANNOT BE < # PROCESSES')
  endif
  RTOT = TOT
  if(QVERBOSE) then
   WRITE(6,*) 'MYNODPX ',MYNODPX,' RTOT/NPROC ',RTOT/NPROC,' NPROC ',NPROC,' TOT ',TOT 
  endif
  PART = dint(RTOT/NPROC)
  if (real(PART).ne.real(TOT)/NPROC) then
    PART = int(real(TOT)/NPROC + 1)
  endif
  if(QWRITE) then
   if(MYNODPX.eq.1) then 
   write(6,*) 'total #points is ',TOT,' per process: ',PART,' #processes ',NPROC
   endif 
  endif

  QATTOT = .false.
  do PP = 1,NPROC
   if(.not.QATTOT) then
    BEGPT(PP) = (PP-1)*PART +1
    ENDPT(PP) = PP*PART
    if(ENDPT(PP).GE.TOT) then
      ENDPT(PP) = TOT
      QATTOT = .true.
    endif
   else !at total already
    BEGPT(PP) = TOT
    ENDPT(PP) = TOT-1 
   endif
  enddo
!  WRITE(6,*) 'PP is NPROC ',NPROC,' BEGPT ',BEGPT(NPROC),' ENDPT ',ENDPT(NPROC)
!  call parstoperr('<ZPART_GRID>','end of printing BEGPT')
    
!  do PP = 1,NPROC
!   write(6,*) 'processor is ',PP,' begin gridpoint ',BEGPT(PP),' end gridpoint ',ENDPT(PP),' tot points ', &
!      ENDPT(PP)-BEGPT(PP) + 1
!  enddo
! I don't think you need every processor's information
  if(QALLPROC) then
   do PP = 1,NPROC
!    write(6,*) 'process ',PP
    NUM = BEGPT(PP)
    do SS = NSUB,1,-1
     DIV = CPC(SS)
!     WRITE(6,*) 'inside SS, NUM is ',NUM,' DIV IS ',DIV
!     if(NUM.lt.0) then
!       WRITE(6,*) 'NUM is lt zero! ',NUM,' BEGPT(PP) ',BEGPT(PP),' ENDPT(PP) ',ENDPT(PP)
!       call parstoperr('<ZPART_GRID>','num is lt zero ')
!     endif
     BEGCONF(PP,SS) = int((NUM-1)/DIV)  + 1 
     NUM = NUM - (BEGCONF(PP,SS)-1)*DIV
!    write(6,*) 'PP ',PP,' SS ',SS
!     write(6,*) 'subspace ',SS,' BEGCONF ', BEGCONF(PP,SS),' DIV ',DIV,' NUM ',NUM, &
!          'BEGCONF(1,2) is ',BEGCONF(1,2)
    enddo 
!     write(6,*) ''
   enddo
!  call parstoperr('<ZPART_GRID>','past first loop')
!   WRITE(6,*) 'BEGCONF(1,2) is ',BEGCONF(1,2)
   do PP = 1,NPROC
    VAL = BEGPT(PP)
    SUM = 0
    do SS = NSUB,1,-1
     SUM = SUM + CPC(SS)*(BEGCONF(PP,SS)-1) 
     if(PP.eq.131072) then
     WRITE(6,*) 'processor ',PP,' ss ',SS,' BEGCONF ',BEGCONF(PP,SS),'BEGCONF(1,2) is ',BEGCONF(1,2)
     endif !temporary
    enddo
!    write(6,*) 'result: processor ',PP,' start at gridpoint ',SUM+1,' VAL ',VAL
   enddo
  endif
! for my processor
  NUM = BEGPT(MYNODPX)
  do SS = NSUB,1,-1
    DIV = CPC(SS)
!    write(6,*) 'div is ',div,' num is ',num
    BEGINME(SS) = int((NUM-1)/DIV)  + 1
    NUM = NUM - (BEGINME(SS)-1)*DIV
!    write(6,*) 'MYNODPX ',MYNODPX,' subspace ',SS,' BEGINME ', BEGINME(SS),' DIV ',DIV,' NUM ',NUM
  enddo
!     write(6,*) ''
!  
!  write(6,*) 'size of CPC is ',size(CPC) 
!  write(6,*) 'NPROC is ',NPROC
  if(allocated(CPC)) call chmdealloc('zerom2.src','ZMERGE','CPC',NSUB,ci8=CPC)
  
!  call parstoperr('<ZPART_GRID>','end of zpart_grid')
  end subroutine zpart_grid
#endif /* (zpart)*/

#if KEY_PARALLEL==1 /*testcomm*/
 subroutine testcomm_sp(LINE,LENGTH)
 use parallel,only: MYNODP,NUMNOD
 use string
 use nbndcc_utilb,only: parstoperr 
 use paral4,only: po_comm_r,po_comm_i
 use cpustruc, only: HEADCOM,HOODCOM,IMYKEYP,NUMNODH
 implicit none 
 character(len=*),intent(in) :: LINE
 integer,intent(in) :: LENGTH
 integer :: NIGCOM,NGCOM,NALGA,NIALGA,NPOST,NIPOST
 integer,dimension(:),allocatable :: ITEST,IREC 
 real(chm_real),dimension(:),allocatable :: RTEST,RREC
 integer :: II,SENDSZ,DATSIZ,TOTSIZ,JJ,START
 logical :: QPRINT
#if KEY_PARALLEL==1
 include 'mpif.h' 
#endif
 integer :: COUNT,COUNT_RATE,COUNT_MAX,COUNT2,ierr

 NIGCOM = GTRMI(LINE,LENGTH,'IGCO',0)
 NGCOM = GTRMI(LINE,LENGTH,'GCOM',0) 
 NALGA = GTRMI(LINE,LENGTH,'ALGA',0)
 NIALGA = GTRMI(LINE,LENGTH,'IALG',0)
 NPOST = GTRMI(LINE,LENGTH,'POST',0)
 NIPOST = GTRMI(LINE,LENGTH,'IPOS',0)
 DATSIZ = GTRMI(LINE,LENGTH,'SIZE',1)
 QPRINT = INDXA(LINE,LENGTH,'PRIN').GT.0
 
! print format
10 FORMAT(A9,I11,A6,F16.8,A11,E16.8,A7,I8,A6,I8) 
 if(MYNODP.eq.1) then
 WRITE(6,*) 'IGCOMB CALLS ',NIGCOM
 WRITE(6,*) 'GCOMB CALLS ',NGCOM
 WRITE(6,*) 'ALLGATHER-R CALLS ',NALGA
 WRITE(6,*) 'ALLGATHER-I CALLS ',NIALGA
 WRITE(6,*) 'POST OFFICE-R CALLS ',NPOST
 WRITE(6,*) 'POST OFFICE-I CALLS ',NIPOST
 endif
 TOTSIZ = DATSIZ*NUMNOD
 START = (MYNODP-1)*DATSIZ

 if(NIGCOM.GT.0) then
  if(allocated(ITEST)) deallocate(ITEST)
  allocate(ITEST(TOTSIZ))
  ITEST = 0

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NIGCOM
   ITEST = 0
   do JJ = 1,DATSIZ
     ITEST(START+JJ) = START+JJ
   enddo
   call IGCOMB(ITEST,TOTSIZ)
  enddo
  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)
  if(MYNODP.eq.1) WRITE(6,10) 'IGCOMB # ',NIGCOM,' TIME ',REAL(COUNT2-COUNT)/COUNT_RATE, &
 ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NIGCOM),' NODES ',NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' IGCOMB TEST ',ITEST(II)
    enddo
   endif
  endif
  deallocate(ITEST)
 endif !if NIGCOM > 0
!-------------------------------------- 
! test GCOMB
 if(NGCOM.GT.0) then 
  if(allocated(RTEST)) deallocate(RTEST)
  allocate(RTEST(TOTSIZ))
  RTEST = 0

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NGCOM
   RTEST = 0
   do JJ = 1,DATSIZ
     RTEST(START+JJ) = REAL(START+JJ)/DATSIZ
   enddo
   call GCOMB(RTEST,TOTSIZ)
  enddo

  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)
  if(MYNODP.eq.1) WRITE(6,10) 'GCOMB # ',NGCOM,' TIME ',REAL(COUNT2-COUNT)/COUNT_RATE, &
  ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NGCOM),' NODES ',NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' GCOMB TEST ',RTEST(II)
    enddo   
   endif 
  endif
  deallocate(RTEST)
 endif !NGCOM >  0
!----------------------------------------
!  goto 10000
! allgather, reals
 if(NALGA.gt.0) then
  if(allocated(RTEST)) deallocate(RTEST)
  allocate(RTEST(DATSIZ))
  if(allocated(RREC)) deallocate(RREC)
  allocate(RREC(TOTSIZ))

! for allgather

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NALGA
   RREC = 0
   do JJ = 1,DATSIZ
     RTEST(JJ) = REAL(START+JJ)/DATSIZ
   enddo
   call MPI_ALLGATHER(RTEST,DATSIZ,MPI_DOUBLE_PRECISION,RREC,DATSIZ,MPI_DOUBLE_PRECISION,MPI_COMM_WORLD,ierr)
  enddo
  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)

  if(MYNODP.eq.1) WRITE(6,10) 'NALGA # ',NALGA,' TIME ', REAL(COUNT2-COUNT)/COUNT_RATE, &
   ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NALGA),' NODES ',NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' REAL ALLGATHER TEST ',RREC(II)
    enddo
   endif
  endif
  deallocate(RTEST,RREC)
 endif !if NALGA > 0
!-------------------------------------------------------------------------- 
! for allgather
 if(NIALGA.gt.0) then
  if(allocated(ITEST)) deallocate(ITEST)
  allocate(ITEST(DATSIZ))
  ITEST = 0
  if(allocated(IREC)) deallocate(IREC)
  allocate(IREC(TOTSIZ))
  IREC = 0

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NIALGA
   IREC=0
   do JJ = 1,DATSIZ
    ITEST(JJ) = START+JJ
   enddo
   call MPI_ALLGATHER(ITEST,DATSIZ,MPI_INTEGER,IREC,DATSIZ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
  enddo
  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)

  if(MYNODP.eq.1) WRITE(6,10) 'NIALGA # ',NIALGA,' TIME ', &
   REAL(COUNT2-COUNT)/COUNT_RATE, ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NIALGA),' NODES ', &
   NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' INT ALLGATHER TEST ',IREC(II)
    enddo
   endif
  endif
  deallocate(ITEST,IREC)
 endif !if NIALGA > 0

!----------------------------------------
 if(NPOST.gt.0) then
  if(allocated(RTEST)) deallocate(RTEST)
  allocate(RTEST(DATSIZ))
  if(allocated(RREC)) deallocate(RREC)
  allocate(RREC(TOTSIZ))

! for po algorithm

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NPOST
   RREC = 0
   do JJ = 1,DATSIZ
     RTEST(JJ) = REAL(START+JJ)/DATSIZ
   enddo
!   call po_comm_r(RTEST,DATSIZ,RREC,HOODCOM,HEADCOM)
   call po_comm_r(RTEST,DATSIZ,RREC,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)
  enddo
  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)

  if(MYNODP.eq.1) WRITE(6,10) 'NPOST # ',NPOST,' TIME ', REAL(COUNT2-COUNT)/COUNT_RATE, &
   ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NPOST),' NODES ',NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' REAL PO TEST ',RREC(II)
    enddo
   endif
  endif
  deallocate(RTEST,RREC)
 endif !if NPOST > 0

!--------------------------------------------------------------------------
 if(NIPOST.gt.0) then
  if(allocated(ITEST)) deallocate(ITEST)
  allocate(ITEST(DATSIZ))
  if(allocated(IREC)) deallocate(IREC)
  allocate(IREC(TOTSIZ))

! for po algorithm

  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
  do II = 1,NIPOST
  IREC = 0
   do JJ = 1,DATSIZ
     ITEST(JJ) = START+JJ
   enddo
   call po_comm_i(ITEST,DATSIZ,IREC,HOODCOM,NUMNODH,IMYKEYP,HEADCOM,NUMNOD)
  enddo
  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)

  if(MYNODP.eq.1) WRITE(6,10) 'NIPOST # ',NIPOST,' TIME ', REAL(COUNT2-COUNT)/COUNT_RATE, &
   ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NIPOST),' NODES ',NUMNOD,' SIZE ',DATSIZ

  if(QPRINT) then
   if(MYNODP.eq.1) then
    do II = 1,TOTSIZ
     WRITE(6,*) 'NODE ',II,' INT PO TEST ',IREC(II)
    enddo
   endif
  endif
  deallocate(ITEST,IREC)
 endif !if NIPOST > 0

!----------------------------------------

! for allgather
! if(NIPOST.gt.0) then
!  if(allocated(ITEST)) deallocate(ITEST)
!  allocate(ITEST(DATSIZ))
!  ITEST = 0
!  if(allocated(IREC)) deallocate(IREC)
!  allocate(IREC(TOTSIZ))
!  IREC = 0
!
!  call SYSTEM_CLOCK(COUNT,COUNT_RATE,COUNT_MAX)
!  do II = 1,NPOST
!   IREC=0
!   do JJ = 1,DATSIZ
!    ITEST(JJ) = MYNODP+JJ
!   enddo
!   call MPI_ALLGATHER(ITEST,DATSIZ,MPI_INTEGER,IREC,DATSIZ,MPI_INTEGER,MPI_COMM_WORLD,ierr)
!  enddo
!  call SYSTEM_CLOCK(COUNT2,COUNT_RATE,COUNT_MAX)
!
!  if(MYNODP.eq.1) WRITE(6,10) 'NIALGA # ',NIALGA,' TIME ', &
!   REAL(COUNT2-COUNT)/COUNT_RATE, ' TIME/CALL ', REAL(COUNT2-COUNT)/(COUNT_RATE*NIALGA),' NODES ', &
!   NUMNOD,' SIZE ',DATSIZ
!
!  if(QPRINT) then
!   if(MYNODP.eq.1) then
!    do II = 1,TOTSIZ
!     WRITE(6,*) 'NODE ',II,' INT ALLGATHER TEST ',IREC(II)
!    enddo
!   endif
!  endif
!  deallocate(ITEST,IREC)
! endif !if NIALGA > 0
!
! if(allocated(ITEST)) deallocate(ITEST)
! if(allocated(RTEST)) deallocate(RTEST)
! if(allocated(RREC)) deallocate(RREC)
! if(allocated(IREC)) deallocate(IREC)
!
!10000 CONTINUE
!! call parstoperr('<testcomm_sp>','end')
 end subroutine testcomm_sp


#endif /* (testcomm)*/

#endif  /*zerom_key*/

 subroutine NULL_ZEROM
  RETURN
 end subroutine NULL_ZEROM

 end module zmodule2
