#if KEY_SQUANTM==1 /*mainsquatn*/
SUBROUTINE SQMMME(CTOT,X,Y,Z,DX,DY,DZ)
  !-----------------------------------------------------------------------
  !
  !     Get energy and forces from New MOPAC
  !     It should do ???
  !
  !     Kwangho Nam, Ross Walker, and Mike Crowley July 2005
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use memory
  !
  use bases_fcm
  use datstr,only:useddt_nbond,useddt_image
  use consta
  use contrl
  use ewald_1m, only: lewald,EWVIRIAL
  !  use erfcd_mod,only: EWLDT
  use gamess_fcm
  use inbnd
  use squantm
  use quantm, only : natom_check,xim,yim,zim
  use nbndqm_mod
  use psf
  use stream
  ! 
  !     Adjust nonbonded group list for IMAGES.
  use image 
#if KEY_PBOUND==1
  use pbound   /* Adjust nonbonded group list for simple pbc.*/
#endif

  ! for parallel run
#if KEY_PARALLEL==1
  use parallel  
#endif
  use prssre
  !
  !
  implicit none

  real(chm_real):: CTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)

  INTEGER I, LPERIOD
  LOGICAL QDONE, QIMAGE
  !
  ! for new mopac qm/mm setup  and QM/MM-Ewald calculations
  real(chm_real)::  ESCF, VOLUME,XTLINV(6)
  real(chm_real)::  CTOT_1,CTOT_2,CTOT_3,ECLASS_1,ECLASS_2, &
       grad_fct_1,grad_fct_2,EWVIRIAL_local(9),EVALN(2)
  INTEGER:: natqm_1,natqm_2,Indx_mminb, ii,jj
  LOGICAL:: QFRST, QCHECK, OK, &
       Qdual_check(2),qmlay_local,qget_put, &
       qget_h,qsetup_h
  !
  !     Are there any QM atoms?
  IF (NATQM(1).EQ.0) RETURN
  !
  ! Initial check up
  QIMAGE =.FALSE.
  IF(.NOT.USEDDT_nbond(BNBND)) CALL WRNDIE(-3,'<SQMMME>', &
       'Nonbond data structure is not defined.')
#if KEY_PBOUND==1
  IF(.NOT.qBoun) THEN   
#endif
     IF(NTRANS.GT.0) THEN
        IF(LGROUP) THEN
           QIMAGE =.TRUE.
           IF(.NOT.USEDDT_image(BIMAG)) CALL WRNDIE(-3,'<SQMMME>', &
                'Image nonbond data structure is not defined.')
        ELSE
           CALL WRNDIE(-2,'<SQMMME>', &
                'QM/MM do not interact with Images under Atom Based Cutoff.')
        END IF
     END IF
#if KEY_PBOUND==1
  END IF                
#endif
  !

  !
  ! Swap coordinates, and make corresponding adjustments on
  ! MMINB array after Swap coordinates.
  ! first check whether number of atom is same
  IF(natom_check.NE.NATOM) THEN
     !     Free Heap allocation for temporary coordinates
     call chmdealloc('sqnt_ene.src','SQMMME','XIM',NATOM,crl=XIM)
     call chmdealloc('sqnt_ene.src','SQMMME','YIM',NATOM,crl=YIM)
     call chmdealloc('sqnt_ene.src','SQMMME','ZIM',NATOM,crl=ZIM)
     natom_check=natom
     !     New allocation of coordinates arrays
     call chmalloc('sqnt_ene.src','SQMMME','XIM',NATOM,crl=XIM)
     call chmalloc('sqnt_ene.src','SQMMME','YIM',NATOM,crl=YIM)
     call chmalloc('sqnt_ene.src','SQMMME','ZIM',NATOM,crl=ZIM)
  END IF
  CALL SwapXYZ_image(NATOM,X,Y,Z,XIM,YIM,ZIM,IMATTQ)

  !
  !     Get them ready for QM/MM calculations and SQUANTM call
  natqm_1=NATQM(1)
  CALL CH2SQM(natqm_1,IGMSEL,XIM,YIM,ZIM,.FALSE.)

  !
  !***********************************************************************
  ! Here's the part doing NewMopac preparation from Amber.
  ! Every energy evaluation.
  !
  !
  !***Note from namkh.
  !   Currenly, the atoms passed into New Mopac part only include
  !   atoms in primary cell (in case of Periodic boundary conditions.).
  !   Thus, all the wrapping should be done this interface before
  !   call any routine in New Mopac, thus making it easier to manage
  !   the program.
  !***End note


  ! 1) Allocates and setup for QM and QM-MM arrays, pair lists, and etc
  QCHECK=.TRUE.
  QFRST =.FALSE.
  !
  !    temporary use QIMAGE, but should be changed near future.
  If (QIMAGE) Then
     LPERIOD = 1
  Else
     LPERIOD = 0
  End if

  Qdual_check(1)= (QMFEP .or. QMLAY)  !
  Qdual_check(2)=.false.              ! 1st qm region
  grad_fct_1    = PreFact_dual(1)
  ! wrap before and after qm/mm setup...
  if(Qdual_check(1)) then
     qget_put      =.true.
     call PushPull_all_array(Qdual_check,qget_put)
  end if

  ! currently, I should assume MMINB array keep the Masking job,
  ! such that MM atoms MMINB(i) > 0 are within cutoff distance.
  !
  ! Use XIM,YIM,ZIM coordinates, which are already wrapped up to
  ! take account of image atoms
  Indx_mminb = 1
  CALL QMMM_SETUP(MAXA,NATOM,Indx_mminb, &
       mminb1_dual,mminb2_dual,LPERIOD,PRNLEV, &
       XIM,YIM,ZIM,CG,QCHECK,QFRST)
  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
       'The CHARMM will stop at QMMM_SETUP.')

  !
  ! setup distance arrays for QM-QM and QM-MM here.
  ! This array is used in QMMM_Ewald_Setup_And_Pot when "incore" used.
  QFRST =.FALSE.
  CALL QMMM_Distance_setup(NATOM,QFRST,Qdual_check)

  ! QM/MM-Ewald case
  ener_bg_cg_corr(1)=zero
  IF(LQMEWD) THEN 
     !
     ! Get the Volume and Reciprocal Space Lattice Vector
     CALL GETVOL(VOLUME)
     CALL INVT33S(XTLINV,XTLABC,OK)


     ! Setup Ktable and Kvec
     QCHECK=.TRUE.
     Indx_mminb = 1
     CALL QMMM_Ewald_Setup_And_Pot(Indx_mminb,VOLUME,XTLINV,X,Y,Z, &
          ener_bg_cg_corr,tot_cg_fep,EWVIRIAL, &
          QCG_corr,QCHECK)  ! EWLDT

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
          'The CHARMM will stop at QMMM_SETUP.')
  END IF

  ! Here's the part computing NewMopac energy and its preparation.
  ! Every energy evaluation.

  ! 2) Allocates and setup for QM and QM-MM arrays, pair lists, and etc
  QCHECK  =.TRUE.
  QFRST   =.FALSE.
  CTOT    = ZERO
  CTOT_1  = ZERO
  ECLASS_1= ZERO
  ESCF    = ZERO

  CALL QMMM_Energy(ESCF,NATOM,QFRST,QCHECK,Qdual_check, &
                   q_apply_tr_bomd,i_md_step)
  !
  !    energy is already in kcal/mol
#if KEY_PARALLEL==1
  if(mynod.eq.0) then             
#endif
     CTOT_1  = ESCF
#if KEY_PARALLEL==1
  end if                          
#endif

  ! note by nam.
  ! for ECLASS from GHO atom. (only, when QGRAD is .false., otherwise it will be
  ! calculated in routine QMMM_Gradient.)
  If ((.not. QGRAD) .and. QLINK(1)) then
     Call QMMM_GHO_Eclass(NATOM,XIM,YIM,ZIM,CG,ECLASS_1)
  End if

  ! some check
  IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
       'The CHARMM will stop at QMMM_Energy.')

  ! 
  ! Now compute gradients fro NewMopac, and already added into
  ! DX,DY,DZ arrays
  ! DTM gradient check
  IF (QGRAD) CALL QMMM_Gradient(NATOM,DX,DY,DZ, &
       XIM,YIM,ZIM,CG, &
       ECLASS_1,grad_fct_1,Qdual_check)
  !
  !
  ! Compute Kspace gradient contribution, but not added here.
  ! Added in subroutine KSPACE (ewalf.src)
  IF(LQMEWD) THEN
     ! do virial initialization here, (check ewaldf.src and pme.src)
     EWVIRIAL(1:9)=zero

     ! Volume and Reciprocal Space Lattice vector have been computed above.
     ! DTM gradient check
     IF (QGRAD) CALL QMMM_Ewald_Grad(VOLUME,XTLINV,X,Y,Z,DX,DY,DZ, &
          grad_fct_1,EWVIRIAL)  ! EWLDT
  END IF

  !
  ! Add the repulsion energy between MM atoms linked to the GHO boundary
  ! when the alpha correction term is included...PJ 12/2002
#if KEY_PARALLEL==1
  if(mynod.eq.0) then             
#endif
     IF(QLINK(1)) CTOT_1  = CTOT_1+ECLASS_1
     CTOT = CTOT_1                                    ! total energy
#if KEY_PARALLEL==1
  end if                          
#endif

  !
  if(Qdual_check(1)) then
     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)
  end if
  !***********************************************************************

  !***********************************************************************
  ! for dual quantum region
  if(QMFEP .or. QMLAY) then
     QCHECK=.TRUE.
     QFRST =.FALSE.
     If (QIMAGE) Then
        LPERIOD = 1
     Else
        LPERIOD = 0
     End if
     Qdual_check(2)=.true.
     qget_put      =.true.
     grad_fct_2    = PreFact_dual(2)
     natqm_2       = NATQM(2)
     ! handle charge
     do i=1,NATQM(1)
        ii     = iabs(qminb1_dual(i))
        cg(ii) = cginb(i)
     end do

     !cc  for H atom to be annihiliated in pKa FEP-QM/MM.
     !cc        if(QMFEP .and. QpKa_on) then
     !cc           cg(IpKa_Hatm)=cginb_pka(IpKa_Hatm_indx,1)
     !cc        end if

     ! wrap before and after energy/gradient call.
     call PushPull_all_array(Qdual_check,qget_put)

     ! handle h-link coords here.
     if(QMLAY .and. QH_Link) then
        qget_h=.true.
        qsetup_h=.false.
        call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST, &
             R_H_ref, &
             x,y,z,xyz_hlink, &
             qget_h,qsetup_h)
        CALL SwapXYZ_image(NATOM,X,Y,Z,XIM,YIM,ZIM,IMATTQ)

        ! gradient
        do i=1,NHLink
           ii=IHOSTGUEST(1,i)
           jj=IHOSTGUEST(2,i)

           dxyz_hlink(1,i)=dx(ii)
           dxyz_hlink(2,i)=dy(ii)
           dxyz_hlink(3,i)=dz(ii)

           dxyz_hlink(4,i)=dx(jj)
           dxyz_hlink(5,i)=dy(jj)
           dxyz_hlink(6,i)=dz(jj)

           dx(ii) = zero
           dy(ii) = zero
           dz(ii) = zero

           dx(jj) = zero
           dy(jj) = zero
           dz(jj) = zero
        end do
     end if

     Indx_mminb = 2
     CALL QMMM_SETUP(MAXA,NATOM,Indx_mminb, &
          mminb1_dual,mminb2_dual,LPERIOD,PRNLEV, &
          XIM,YIM,ZIM,CG,QCHECK,QFRST)
     ! some check
     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
          'The CHARMM will stop at 2nd QMMM_SETUP.')

     QFRST =.FALSE.
     CALL QMMM_Distance_setup(NATOM,QFRST,Qdual_check)

     ener_bg_cg_corr(2)=zero
     If(LQMEWD .and. .not. QMLAY) then
        QCHECK     =.TRUE.
        Indx_mminb = 2
        CALL QMMM_Ewald_Setup_And_Pot(Indx_mminb,VOLUME,XTLINV, &
             X,Y,Z, &
             ener_bg_cg_corr,tot_cg_fep,EWVIRIAL, &
             QCG_corr,QCHECK) ! EWLDT

        IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
             'The CHARMM will stop at QMMM_SETUP.')
     End if

     QCHECK  =.TRUE.
     QFRST   =.FALSE.
     CTOT_2  = ZERO
     CTOT_3  = ZERO
     ECLASS_2= ZERO
     ESCF    = ZERO

     CALL QMMM_Energy(ESCF,NATOM,QFRST,QCHECK,Qdual_check, &
                      q_apply_tr_bomd,i_md_step)  

#if KEY_PARALLEL==1
     if(mynod.eq.0) then             
#endif
        CTOT_2  = ESCF
#if KEY_PARALLEL==1
     end if                          
#endif

     If ((.not. QGRAD) .and. QLINK(2)) then
        Call QMMM_GHO_Eclass(NATOM,XIM,YIM,ZIM,CG,ECLASS_2)
     End if

     IF(.NOT.QCHECK) CALL WRNDIE(-5,'<SQMMME>', &
          'The CHARMM will stop at QMMM_Energy.')

     IF (QGRAD) CALL QMMM_Gradient(NATOM,DX,DY,DZ, &
          XIM,YIM,ZIM,CG, &
          ECLASS_2,grad_fct_2,Qdual_check)

     IF(LQMEWD .and. .not. QMLAY) THEN 
        !           make local copy and re-initialize the array.
        EWVIRIAL_local(1:9) = EWVIRIAL(1:9)
        EWVIRIAL            = zero
        IF (QGRAD) CALL QMMM_Ewald_Grad(VOLUME,XTLINV,X,Y,Z,DX,DY,DZ, &
             grad_fct_2,EWVIRIAL) ! EWLDT
        !           recover original value (for 1st qm region)
        EWVIRIAL(1:9)=EWVIRIAL(1:9)*grad_fct_2+EWVIRIAL_local(1:9)*grad_fct_1
     end if

     ! for pKa FEP-QM/MM: bonded energy terms.
     EVALN(1:2)=zero
     if(QMFEP .and. QpKa_on) then
        call PKAINTE_HMM(EVALN,grad_fct_2,XIM,YIM,ZIM,DX,DY,DZ)
        CTOT_2 = CTOT_2 + EVALN(1) + EVALN(2)
     end if

     ! for multi-layered qm/mm (high qm level).
     if(QMLAY) then
        call HighQM_ene_mlayer(NATOM,natqm_2,qminb2_dual, &
#if KEY_GAMESS==1
             mminb1_dual,               & 
#endif
             CTOT_3,XIM,YIM,ZIM,DX,DY,DZ,CG)
     end if

#if KEY_PARALLEL==1
     if(mynod.eq.0) then           
#endif
        IF(QLINK(2)) CTOT_2  = CTOT_2+ECLASS_2
        if(QMLAY) then
           CTOT = PreFact_dual(1)*CTOT_1+ &
                PreFact_dual(2)*(CTOT_2-CTOT_3)
        else
           CTOT = PreFact_dual(1)*CTOT_1+PreFact_dual(2)*CTOT_2
           if(LQMEWD) CTOT = CTOT &
                +PreFact_dual(1)*PreFact_dual(1)*ener_bg_cg_corr(1) &
                +PreFact_dual(2)*PreFact_dual(2)*ener_bg_cg_corr(2)

           ! for statistical averaging.
           EEQTRM(QPT1) = (one-lambda_qm(2))*CTOT_1+lambda_qm(2)*CTOT_2
           EEQTRM(QPT2) = (one-lambda_qm(3))*CTOT_1+lambda_qm(3)*CTOT_2
           if(LQMEWD) then
              EEQTRM(QPT1) = EEQTRM(QPT1) + &
                   (one-lambda_qm(2))*(one-lambda_qm(2))*ener_bg_cg_corr(1) + &
                   lambda_qm(2)*lambda_qm(2)*ener_bg_cg_corr(2)
              EEQTRM(QPT2) = EEQTRM(QPT2) +  &
                   (one-lambda_qm(3))*(one-lambda_qm(3))*ener_bg_cg_corr(1) + &
                   lambda_qm(3)*lambda_qm(3)*ener_bg_cg_corr(2)
           end if
        end if

#if KEY_PARALLEL==1
     end if                        
#endif

     ! handle h-link coords here.
     if(QMLAY .and. QH_Link) then
        qget_h=.false.
        qsetup_h=.false.
        call get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST, &
             R_H_ref, &
             x,y,z,xyz_hlink,qget_h, &
             qsetup_h)

        ! handle gradient for h-link atoms.
        call put_hlink_grads(NHLink,MAXCHL,IHOSTGUEST, &
             R_H_ref, &
             dx,dy,dz,xyz_hlink,.false.) 

        ! gradient
        do i=1,NHLink
           ii=IHOSTGUEST(1,i)
           jj=IHOSTGUEST(2,i)

           dx(ii) = dx(ii) + dxyz_hlink(1,i)
           dy(ii) = dy(ii) + dxyz_hlink(2,i)
           dz(ii) = dz(ii) + dxyz_hlink(3,i)

           dx(jj) = dx(jj) + dxyz_hlink(4,i)
           dy(jj) = dy(jj) + dxyz_hlink(5,i)
           dz(jj) = dz(jj) + dxyz_hlink(6,i)
        end do
     end if

     qget_put      =.false.
     call PushPull_all_array(Qdual_check,qget_put)

     ! handle charge
     do i=1,NATQM(1)
        cg(iabs(qminb1_dual(i))) =zero
     end do
  End if
  RETURN
END SUBROUTINE SQMMME
!***********************************************************************
!
!
#endif /*  (mainsquatn)*/
!
SUBROUTINE QM_ENE_BLANK
  !
  RETURN
END SUBROUTINE QM_ENE_BLANK

