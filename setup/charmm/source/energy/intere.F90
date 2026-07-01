SUBROUTINE INTEOP(COMLYN,COMLEN,ICYCLE)
  !-----------------------------------------------------------------------
  !     Process the interaction energy command.
  !-----------------------------------------------------------------------
  !     This routine computes the interaction energies and forces between
  !     selected sets of atoms. If only one selection is given, then
  !     a self energy is returned. The forces are returned in the force
  !     array. Some non-selected atoms may have a non-zero force.
  !
  !     By:  Bernard R. Brooks     22-OCT-1984
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number

  use code
  use coord
  use coordc
  use energym
  use stream
  use string

  use psf
  use inbnd
  use cnst_fcm
  use hbondm
  use image
#if KEY_FACTS==1
  use facts_module     
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
  implicit none
  !
  !     Passed variables.
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN, ICYCLE
  !     Local variables.
  INTEGER   UNIT
  LOGICAL   QCOMP, QPRINT, QKEEP
  INTEGER nskip

#if KEY_DOMDEC==1
  if (q_domdec) then
     CALL WRNDIE(-5,'<INTEOP>','INTERaction command not implemented for DOMDEC')
  endif
#endif 
  ! hwm: per BRB's suggestion on 220722
  !nskip = MAX(NBONDT,NTHETT,NPHIT,NIMPHT,NHB,NCSPHI &
  nskip = MAX(NBONDT2,NTHETT,NPHIT,NIMPHT,NHB,NCSPHI &
#if KEY_CMAP==1
       ,NCRTT &      
#endif
       )

  QCOMP = INDXA(COMLYN, COMLEN, 'COMP') .GT. 0
  QPRINT = .NOT. (INDXA(COMLYN, COMLEN, 'NOPR') .GT. 0)
  QKEEP = indxa(COMLYN, COMLEN, 'KEEP') .gt. 0

  IF (QCOMP) THEN
     call inte_coords(comlyn, comlen, icycle, XCOMP, YCOMP, ZCOMP, WCOMP, nskip, qkeep)
  ELSE
     call inte_coords(comlyn, comlen, icycle, X, Y, Z, WMAIN, nskip, qkeep)
  ENDIF

  IF (QPRINT) THEN
     UNIT = GTRMI(COMLYN, COMLEN, 'UNIT', OUTU)
     ICYCLE = ICYCLE + 1
     IF(UNIT.EQ.OUTU .AND. PRNLEV.LT.2) QPRINT=.FALSE.
     IF(UNIT.NE.OUTU .AND. IOLEV.LT.0) QPRINT=.FALSE.
     IF(QPRINT) CALL PRINTE(UNIT, EPROP, ETERM, 'INTE', 'ENR', &
          .TRUE.,ICYCLE, ZERO, ZERO, .TRUE.)
  ENDIF
  RETURN
END SUBROUTINE INTEOP

subroutine inte_coords(comlyn, comlen, icycle, xarg, yarg, zarg, warg, nskip, qkeep)
   use number
   use memory
   use bases_fcm, only: BIMAG
   use heurist, only: UPDECI
   use psf, only: NATOM, NATOMT
   use select, only: SELCTD
#if KEY_FLUCQ==1
   use flucq, only: QFLUC
   use flucqm, only: FQCFOR
#endif 
   implicit none

   character(len=*), parameter :: srcfile='intere.src', procname = 'inte_coords'
   character(len=*), intent(inout) :: comlyn
   integer, intent(inout) :: comlen
   integer, intent(inout) :: icycle
   integer, intent(in) :: nskip
   logical, intent(in) :: qkeep
   real(chm_real), intent(in) :: xarg(*), yarg(*), zarg(*), warg(*)
   integer :: iskip(nskip)
   real(chm_real) :: rtemp(NATOM)
   integer, allocatable :: islct(:), jslct(:)
   logical :: err

   call UPDATE(comlyn, comlen, xarg, yarg, zarg, warg, .false., &
        .true., .false., .false., .false., &
        0, [zero], [zero], [zero], [zero], [zero], [zero])
   call UPDECI(icycle, xarg, yarg, zarg, warg, 0, &
         [ZERO], [ZERO], [ZERO], [ZERO], [ZERO], [ZERO])

   call chmalloc(srcfile,procname,'ISLCT',NATOMT,intg=islct)
   call chmalloc(srcfile,procname,'JSLCT',NATOMT,intg=jslct)
   call SELCTD(comlyn, comlen, islct, jslct, &
         xarg, yarg, zarg, warg, .true., err)
   if (.not. err) then
      call inter2(xarg, yarg, zarg, islct, jslct, &
            iskip, rtemp, &
#if KEY_FLUCQ==1
            QFLUC, FQCFOR, & 
#endif
            BIMAG%IMATTR, BIMAG%IMATPT, &
#if KEY_GRID==1
            .true., .true., & 
#endif
            qkeep)
   endif
   call chmdealloc(srcfile,procname,'JSLCT',NATOMT,intg=jslct)
   call chmdealloc(srcfile,procname,'ISLCT',NATOMT,intg=islct)
end subroutine inte_coords

SUBROUTINE INTER2(X,Y,Z,ISLCT,JSLCT,ISKIP,RTEMP, &
#if KEY_FLUCQ==1
     QFLUC,FQCFOR,   & 
#endif
     IMTR,IMPT, &
#if KEY_GRID==1
     QDUP, QFRE,  & 
#endif
     QKEEP)
  use genborn, only: qgenborn, fillalpgb, gbsolv
#if KEY_BLOCK==1
  use genborn, only: fillalpgb1, fillalpgb2, gbsolv1, gbsolv2
#endif
  use gb_common,only: igentype
  !-----------------------------------------------------------------------
  !     This routine does the actual work of the interaction energies.
  !
  !     By Bernard R. Brooks   22-OCT-1984
  !     Jay L. Banks 19 October 1995: Added MMFF energy calls
  !
  use gbmv, only:rungbmv                          
  use chm_kinds
  use dimens_fcm
  use number
  use bases_fcm
  use deriv
  use econtmod
  use fast
  use psf
  use param
  use cnst_fcm
  use code
  use energym
  use hbondm
  use inbnd
  use image
  use image_routines_module, only: transi
  use stream
  use pbound
#if KEY_ASPENER==1
  use eef1_mod              
#endif
  use gbim, only: QGBIMb, FillAlpGBIM, GBSolvIM
  use grid_dock
  use block_fcm
  use ffieldm
#if KEY_PARALLEL==1
  use parallel          
#endif
#if KEY_MMFF==1
  use mmffm             
#endif
#if KEY_MMFF==1
  use escalar_mm        
#endif
  use datstr
  use usermod,only: usere
  use cmapm
  use eintern
  use enbond_mod
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif 
#if KEY_FACTS==1
  use facts_module     
#endif
#if KEY_FACTS==1
  use memory            
#endif
  implicit none

  real(chm_real) X(*),Y(*),Z(*)
  INTEGER ISLCT(*),JSLCT(*),ISKIP(*)
  real(chm_real) RTEMP(*)
  INTEGER IMTR(*),IMPT(*)
#if KEY_GRID==1
  Logical QDUP, QFRE 
#endif
  Logical QKEEP
#if KEY_FLUCQ==1
  LOGICAL QFLUC
  real(chm_real) FQCFOR(*)
#endif 

#if KEY_GBFIXAT==1
  INTEGER GbFixed
#endif 

  INTEGER I,J,ITEMP,ISTRT,IEND,ITRANS
  real(chm_real) EXX, ENBX, EELX, EST2X
  LOGICAL ERR
  type(nonbondDataStructure) BDUMMY
  !
#if KEY_PARALLEL==1
  INTEGER,PARAMETER :: NALLWK=LENENT+1
  real(chm_real) ALLWRK(NALLWK)
  INTEGER IPT
#endif 

#if KEY_FACTS==1
  integer :: ierr,ierr2      
#endif
#if KEY_MMFF==1
  INTEGER DERIVS
  DERIVS=1
#endif 
  ! copy atom selections to image atoms.
#if KEY_PBOUND==1
  IF(.NOT.QBOUN) THEN     
#endif
     IF(NTRANS.GT.0) THEN
        ITEMP=NATOM+1
        DO ITRANS=1,NTRANS
           ISTRT=ITEMP
           IEND=IMPT(ITRANS)
           ITEMP=IEND+1
           DO I=ISTRT,IEND
              J=IMTR(I)
              ISLCT(I)=ISLCT(J)
              JSLCT(I)=JSLCT(J)
           ENDDO
        ENDDO
        IF(NATIM.NE.IEND) CALL DIEWRN(-4) ! coding error
     ENDIF
#if KEY_PBOUND==1
  ENDIF                   
#endif
  !
  !     Now fill skip arrays for bonds, angles,...
  !
  DO I=1,LENENT
     ETERM(I) = ZERO
  ENDDO

  DO I=1,NATOMT
     DX(I)=ZERO
     DY(I)=ZERO
     DZ(I)=ZERO
  ENDDO
  !
  !     Zero the energy contribution array.
  IF(QECONT) THEN
     DO I=1,NATOMT
        ECONT(I)=ZERO
     ENDDO
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! User energy term
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) THEN   
#endif
     IF(QETERM(USER)) THEN
        CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM)
     ENDIF
#if KEY_PARALLEL==1
  ENDIF                 
#endif
  !
  !     USE NORMAL INTERNAL ENERGY ROUTINES
  !
  !-----------------------------------------------------------------------
  ! Bond energy term
  IF(NBOND.GT.0.AND.QETERM(BOND)) THEN
     DO I=1,NBOND
        IF (ISLCT(IB(I)).EQ.1 .AND. JSLCT(JB(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE IF (ISLCT(JB(I)).EQ.1 .AND. JSLCT(IB(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     ENDDO
     CALL EBOND(ETERM(BOND),IB,JB,ICB,NBOND,CBC,CBB,DX,DY,DZ, &
          X,Y,Z,QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Angle energy term
  IF(NTHETA.GT.0.AND.QETERM(ANGLE)) THEN
     DO I=1,NTHETA
        IF (ISLCT(JT(I)).EQ.1 .AND. JSLCT(JT(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     ENDDO
     CALL EANGLE(ETERM(ANGLE),IT,JT,KT,ICT,NTHETA,CTC,CTB,DX,DY,DZ, &
          X,Y,Z,QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
          )

  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Dihedral energy term
  IF(NPHI.GT.0.AND.QETERM(DIHE)) THEN
     DO I=1,NPHI
        IF (ISLCT(JP(I)).EQ.1 .AND. JSLCT(KP(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE IF (ISLCT(KP(I)).EQ.1 .AND. JSLCT(JP(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     ENDDO
     CALL EPHI(ETERM(DIHE),IP,JP,KP,LP,ICP,NPHI,CPC,CPD,CPB,CPCOS, &
          CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
          QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.)

  ENDIF
  !
  !-----------------------------------------------------------------------
#if KEY_MMFF==1 /*mmff_eterms*/
  IF (FFIELD.EQ.MMFF) THEN
     !
     !-----------------------------------------------------------------------
     ! Out-of-plane energy (MMFF)
     IF(NTHETA.GT.0.AND.QETERM(OOPL)) THEN
        DO I=1,NTHETA
           !     TREAT AS AN INTERACTION BETWEEN CENTRAL ATOM OF BOND ANGLE (JT) AND
           !     "OUT-OF-PLANE" ATOM (LTHETA); NOTE THAT AN OUT-OF-PLANE TERM IS ONLY
           !     DEFINED FOR ANGLE I IF LTHETA(I).GT.0
           IF(LTHETA(I).GT.0) THEN
              IF (ISLCT(JT(I)).EQ.1 .AND. JSLCT(LTHETA(I)).EQ.1) THEN
                 ISKIP(I)=0
              ELSE IF (ISLCT(LTHETA(I)).EQ.1 .AND. JSLCT(JT(I)).EQ.1) &
                   THEN
                 ISKIP(I)=0
              ELSE
                 ISKIP(I)=1
              ENDIF
           ENDIF
        ENDDO
        CALL EOOPL(ETERM(OOPL),IT,JT,KT,LTHETA, &
             icoop,ntheta,OoplFC, &
             X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
             QECONT,ECONT,1,ISKIP)
     ENDIF
     !
     !-----------------------------------------------------------------------
     ! Strech-Bend coupling energy (MMFF)
     IF(NTHETA.GT.0.AND.QETERM(STRB)) THEN
        DO I=1,NTHETA
           !     TREAT STRETCH-BENDS, LIKE THE ANGLES THEY CONTAIN, AS INTERACTIONS
           !     BETWEEN THE "OUTER" ATOMS (IT AND KT).
           IF (ISLCT(IT(I)).EQ.1 .AND. JSLCT(KT(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE IF (ISLCT(KT(I)).EQ.1 .AND. JSLCT(IT(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        ENDDO
        CALL ESTRBND(ETERM(STRB),IT,JT,KT,ICT, &
             ntheta,CTB, &
             IB,JB,ICB,CBB,StrbList,ICSTBN,STBNP, &
             X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
             QECONT,ECONT,1,ISKIP)
     ENDIF
  ELSE
#endif /* (mmff_eterms)*/
     !
     !-----------------------------------------------------------------------
     ! Urey-Bradley energy term
     IF(NTHETA.GT.0.AND.QETERM(UREYB)) THEN
        DO I=1,NTHETA
           IF (ISLCT(IT(I)).EQ.1 .AND. JSLCT(KT(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE IF (ISLCT(KT(I)).EQ.1 .AND. JSLCT(IT(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        ENDDO
        CALL EBOND(ETERM(UREYB),IT,KT,ICT,NTHETA,CTUC,CTUB,DX,DY,DZ, &
             X,Y,Z,QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

     ENDIF
     !
     !-----------------------------------------------------------------------
     ! Improper energy term
     IF(NIMPHI.GT.0.AND.QETERM(IMDIHE)) THEN
        DO I=1,NIMPHI
           IF (ISLCT(IM(I)).EQ.1 .AND. JSLCT(IM(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        ENDDO
        CALL EPHI(ETERM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI,CIC,CID,CIB, &
             CICOS,CISIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
             QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.)

     ENDIF
#if KEY_CMAP==1
     !---------------------------------------------------------------------
     ! . Crossterms
     IF(NCRTERM.GT.0.AND.QETERM(CMAP)) THEN
        DO I=1,NCRTERM
           IF (ISLCT(I1CT(I)).EQ.1 .AND. &
                JSLCT(I1CT(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        ENDDO
        CALL ECMAP(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
             I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
             DX,DY,DZ,X,Y,Z, &
             QECONT,ECONT,1,ISKIP, (/ZERO/), (/0/), .FALSE.)
     ENDIF

#endif 
     !
     !-----------------------------------------------------------------------
#if KEY_MMFF==1 /*mmff_endif*/
  ENDIF !IF(FFIELD.EQ.MMFF)
#endif /* (mmff_endif)*/
  !
  !-----------------------------------------------------------------------
  ! Nonbond energy term
  !
#if KEY_GRID==1
  !      If(QDUP .and. NTrans .le. 0) Then  
#endif
#if KEY_GRID==1
  If(QDUP) Then  
#endif
     !     Make sure counters, flags and cuttoffs, etc are correct.
     CALL GETBND(BNBND,.TRUE.)

     CALL FREEDT_nbond(BNBNDC)
     CALL DUPLDT_nbond(BNBNDC,BNBND)

     ! Inter3g added by Lazaridis, Apr 99
     ! MULTE addition, June 2008 bt
     IF (LGROUP) THEN
        IF (QKEEP) then
           CALL INTER3G('Normal Group Nonbond List',ISLCT,JSLCT, &
                bnbnd%INBLOG,bnbnd%JNBG, &
                bnbndc%INBLOG,bnbndc%JNBG)
        ELSE
           CALL INTER3G('Normal Group Nonbond List',ISLCT,JSLCT, &
                bnbndc%INBLOG,bnbndc%JNBG, &
                bnbnd%INBLOG,bnbnd%JNBG)
        ENDIF
     ELSE
        IF (QKEEP) then
           CALL INTER3('Normal Atom Nonbond List',ISLCT,JSLCT,NATOM,NGRP, &
                bnbnd%INBLO,bnbnd%JNB, &
                bnbndc%INBLO,bnbndc%JNB, &
                bnbndc%INBLOG &
#if KEY_IMCUBES==1
                ,lbycbim                         & 
#endif
                )
        ELSE
           CALL INTER3('Normal Atom Nonbond List',ISLCT,JSLCT,NATOM,NGRP, &
                bnbndc%INBLO,bnbndc%JNB, &
                bnbnd%INBLO,bnbnd%JNB, &
                bnbndc%INBLOG &
#if KEY_IMCUBES==1
                ,lbycbim                         & 
#endif
                )
        ENDIF
     ENDIF
#if KEY_GRID==1
  EndIf   
#endif
  !
  ! EEF1EN added by Lazaridis, Apr 99.  May2004: Moved before ENBOND
#if KEY_ASPENER==1
  IF (DOEEF1.AND.QETERM(ASP)) THEN
     ! added last 3 arguments for 2nd derivatives. I.A.
     CALL EEF1EN(ETERM(ASP),X,Y,Z,DX,DY,DZ,QECONT,ECONT,1,NATOM, &
          1,NGRP,bnbndc%JNBG,bnbndc%INBLOG, &
          bnbnd%INB14,bnbnd%IBLO14,.TRUE., &
          .FALSE. &
          )

  ENDIF
#endif /*  ASPENER*/
  !
  ! MULTE addition June 2008 bt
  IF (QKEEP) THEN
     CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBND, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/),.FALSE., &
          QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,   & 
#endif
          QETERM(ST2),NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )
  ELSE
     CALL ENBOND(ETERM(VDW),ETERM(ELEC),BNBNDC, &
          1,NATOM,CG,RSCLF,NGRP,IGPBS,IGPTYP,IAC,IACNB, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/),.FALSE., &
          QETERM(VDW),QETERM(ELEC), &
#if KEY_FLUCQ==1
          QFLUC,FQCFOR,   & 
#endif
          QETERM(ST2),NST2,ETERM(ST2),.FALSE. &
#if KEY_WCA==1
          ,.FALSE.,ONE,WCA        & 
#endif
#if KEY_LJPME==1
          ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
          )
  ENDIF

  !  Generalized Born Solvation energy term
  !
  !  First call set-up and get alphas and sigmas for this config
  !
  IF (QGenBorn) THEN
     If ( NTrans .gt. 0 ) Call WrnDie(-3,'<ENERGY>', &
          'Images not implemented w/ Generalized Born.')
     genborntype: select case(IGenType)
     case(1) ! IGenType == 1
#if KEY_BLOCK==1
        IF (QBLOCK) THEN
           Call FillAlpGB1( X, Y, Z, &
                bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
                .true., islct, jslct &
#if KEY_GBFIXAT==1
               ,GbFixed &
#endif
               )
           Call GBSolv1( ETERM(GBEnr), X, Y, Z, DX, DY, DZ, &
                bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
                .true., islct, jslct &
#if KEY_GBFIXAT==1
                ,GbFixed &
#endif
                )
        ELSE
#endif /* block */
           CALL WRNDIE(-3,'<ENERGY>', &
                'BLOCK should be on when IGenType = 1.')
#if KEY_BLOCK==1
        ENDIF !if QBLOCK
#endif /* block */
     case(2) ! IGenType == 2
#if KEY_BLOCK==1
        IF (QBLOCK) THEN
           Call FillAlpGB2( X, Y, Z, &
                bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
                .true., islct, jslct &
#if KEY_GBFIXAT==1
                ,GbFixed &
#endif
                )
           Call GBSolv2( ETERM(GBEnr), X, Y, Z, DX, DY, DZ, &
                bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
                .true., islct, jslct &
#if KEY_GBFIXAT==1
                ,GbFixed &
#endif
                )
        ELSE
#endif /* block */
           CALL WRNDIE(-3,'<ENERGY>', &
                'BLOCK should be on when IGenType = 2.')
#if KEY_BLOCK==1
        ENDIF ! if(qblock)
#endif /* block */
     case(20) ! iGenType == 20
        Call RunGBMV( ETERM(GBEnr), &
             bnbnd%JnB, bnbnd%INblO, bnbnd%InB14, bnbnd%IbLo14, &
             .true.,islct,jslct)
     case default ! Common Generalized Born model
        Call FillAlpGB( X, Y, Z, &
             bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
             .true., islct, jslct &
#if KEY_GBFIXAT==1
             ,GbFixed & 
#endif
             )
        Call GBSolv( ETERM(GBEnr), X, Y, Z, DX, DY, DZ, &
             bnbndc%JnB, bnbndc%INblO, bnbnd%InB14, bnbnd%IbLo14, &
             .true., islct, jslct &
#if KEY_GBFIXAT==1
             ,GbFixed & 
#endif
             )
     end select genborntype
  Endif
  !-----------------------------------------------------------------------
  !  Generalized Born Solvation energy model with Implicit Membrane
  !  GBIM is modification of Genborn
  !
  IF (QGBIMb) THEN
     If ( NTrans .gt. 0 ) Call WrnDie(-3,'<ENERGY>', &
          'Images not implemented w/ GBIM.')
     Call FillAlpGBIM( X, Y, Z, &
          BnBnDC%JnB, BnBnDC%INblO, &
          BnBnD%InB14, BnBnD%IbLo14, &
          .true., islct, jslct )

     Call GBSolvIM( ETERM(GBEnr), X, Y, Z, DX, DY, DZ,  &
          BnBnDC%JnB, BnBnDC%INblO, &
          BnBnD%InB14, BnBnD%IbLo14, &
          .true., islct, jslct )
  ENDIF
  !-----------------------------------------------------------------------
  ! FACTS screening interaction 06.06.2011
#if KEY_FACTS==1
   IF(FCTRUN) THEN
      if (.not. fctaim) then
         ! Create Copy of interaction arrays
         if(allocated(FCTBNDC%fct2ilo))write(outu,*) "FCT2IL copy already allocated"
         if(allocated(FCTBNDC%fct2jnb))write(outu,*) "FCT2IB copy already allocated"
         call chmalloc('intere.src','INTER2','FCT2ILOC', natom,  intg=FCTBNDC%fct2ilo)
         call chmalloc('intere.src','INTER2','FCT2JNBC', mxfcab, intg=FCTBNDC%fct2jnb)

         FCTBNDC%fct2ilo = 0
         FCTBNDC%fct2jnb = 0

         ! Select interaction lists
         CALL INTER3('FACTS Inter List',ISLCT,JSLCT,NATOM,NGRP, &
             FCTBNDC%fct2ilo , FCTBNDC%fct2jnb, &
             FCTBND%fct2ilo  , FCTBND%fct2jnb,&
             bnbndc%INBLOG &
#if KEY_IMCUBES==1
             ,lbycbim                         & 
#endif
             )

         if(.not. fctscro)then
            do i=1,natomt
               fctslfw(i) = one
               if(islct(i) /= 1) then
                  fctslfw(i) = zero
               endif
            enddo
         endif

         ! Call the FACTS energy subroutine
         !.fm.
         call fctene(eterm(ifctpol)                 ,&
                    eterm(ifctnpl)                 ,&
                    natom                          ,&
                    BNBND%inblo ,BNBND%jnb         ,&
                    FCTBND%fct1ilo,FCTBND%fct1jnb  ,&
                    FCTBNDC%fct2ilo,FCTBNDC%fct2jnb,&
                    natim                          ,&
                    BIMAG%imblo ,BIMAG%imjnb       ,&
                    BIMAG%imattr                   ,&
                    FCTBND%fct3ilo,FCTBND%fct3jnb  ,&
                    FCTBNDC%fct4ilo,FCTBNDC%fct4jnb)

         ! Deallocate copies of interaction arrays
         call chmdealloc('intere.src','INTER2','FCT2ILOC', natom,  intg=FCTBNDC%fct2ilo)
         call chmdealloc('intere.src','INTER2','FCT2JNBC', mxfcab, intg=FCTBNDC%fct2jnb)
      endif
   ENDIF
#endif 
  !-----------------------------------------------------------------------

#if KEY_GRID==1
  IF (QGrid .and. QGridOK) THEN
     !            Write(6,'(a)')' Call GridEnergy'
     Call GridEnergy(ETerm(GrvdW),ETerm(GrElec), ETerm(GrHBON), &
          XGridCenter, YGridCenter, ZGridCenter, &
          XGridMax, YGridMax, ZGridMax, DGrid, &
          GridForce, Cg, X, Y, Z, Dx, DY, DZ, &
          QETERM(GrvdW), QETerm(GrElec), QETerm(GrHBON), QGrHBon, &
          0, 0, .true., ISlct)

  Endif
#endif /* IF Grid*/
#if KEY_GRID==1
  !      If(QFRE .and. NTrans .le. 0) Then  
#endif

  ! F.M. 2011 (CALL FREEDT moved further down - Necessary for FACTS inter)
#if KEY_GRID==1
  ! If(QFRE) Then  
#endif
  !    CALL FREEDT_nbond(BNBNDC)
#if KEY_GRID==1
  ! EndIf  
#endif

#if KEY_PBOUND==1
  IF(.NOT.QBOUN) THEN     
#endif
     IF(NTRANS.GT.0) THEN
        CALL ALIASDT_nbond(BDUMMY, BNBND)
        CALL FREEDT_image(BIMAGC)
        CALL DUPLDT_image(BIMAGC,BIMAG)

        CALL NBSET_B14_FROM_IMG(BDUMMY, BIMAGC)
        CALL NBSET_G14_FROM_IMG(BDUMMY, BIMAGC)
        NNG14=bimagc%NIMING
        !
        !       Check if any self-energy terms are present
        !
        IF(bimag%NIMNBS.GT.0 .OR. bimag%NIMNBX.GT.0)THEN
           !         Self terms are present
           DO I=1,NATIM
              DX(I)=DX(I)*TWO
              DY(I)=DY(I)*TWO
              DZ(I)=DZ(I)*TWO
           ENDDO
           CALL NBSET_FROM_IMG_SX(BDUMMY, BIMAGC)
           NNNB =bimagc%NIMNBS
           NNNBG=bimagc%NIMNBX
           NNB14= 0
           CALL SETBND(BDUMMY)
           !
           CALL INTER3('Self Image Nonbond ',ISLCT,JSLCT,NATIM,NIMGRP, &
                BDUMMY%INBLO,BDUMMY%JNB, &
                bimag%IMBLOS,bimag%IMJNBS, &
                BDUMMY%INBLOG &
#if KEY_IMCUBES==1
                ,lbycbim                                  & 
#endif
                )
           CALL ENBOND(ETERM(IMVDW),ETERM(IMELEC),BDUMMY, &
                1,NATIM,CG,RSCLF,NIMGRP, &
                IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z, &
                QECONT,ECONT,.FALSE.,ZERO,(/ZERO/),(/0/),.FALSE., &
                QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,   & 
#endif
                QETERM(IMST2),NST2,ETERM(IMST2),.FALSE. &
#if KEY_WCA==1
                ,.FALSE.,ONE,WCA   & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )

           ETERM(IMVDW) = ETERM(IMVDW)*HALF
           ETERM(IMELEC)= ETERM(IMELEC)*HALF
           ETERM(IMST2) = ETERM(IMST2)*HALF
           DO I=1,NATIM
              DX(I)=DX(I)*HALF
              DY(I)=DY(I)*HALF
              DZ(I)=DZ(I)*HALF
           ENDDO
        ENDIF
        !
        !       compute image nonbonded energies
        !
        IF(bimag%NIMNB.GT.0 .OR. bimag%NIMNBG.GT.0) THEN
           !
           CALL NBSET_FROM_IMG_G(BDUMMY, BIMAGC)
           NNB14=bimagc%NIMINB
           NNNB=bimagc%NIMNB
           NNNBG=0
           CALL SETBND(BDUMMY)
           !
           CALL INTER3('Image Nonbond List ',ISLCT,JSLCT,NATIM,NIMGRP, &
                BDUMMY%INBLO,BDUMMY%JNB, &
                bimag%IMBLO,bimag%IMJNB, &
                BDUMMY%INBLOG &
#if KEY_IMCUBES==1
                ,lbycbim                      & 
#endif
                )
           CALL ENBOND(ENBX,EELX,BDUMMY,1,NATIM,CG,RSCLF,NIMGRP, &
                IGPBS,IGPTYP,IAC,IACNB,DX,DY,DZ,X,Y,Z, &
                QECONT,ECONT,QETERM(EWEXCL),ETERM(EWEXCL), &
                (/ZERO/),(/0/),.FALSE., &
                QETERM(IMVDW),QETERM(IMELEC), &
#if KEY_FLUCQ==1
                QFLUC,FQCFOR,    & 
#endif
                QETERM(IMST2),NST2,EST2X,.FALSE. &
#if KEY_WCA==1
                ,.FALSE.,ONE,WCA  & 
#endif
#if KEY_LJPME==1
                ,QETERM(LJEXC), ETERM(LJEXC)  & 
#endif
                )

!!!               ETERM(IMVDW) = ETERM(IMVDW) + ENBX
!!!               ETERM(IMELEC)= ETERM(IMELEC)+ EELX
!!!               ETERM(IMST2) = ETERM(IMST2) + EST2X
!!!
!!!            ENDIF
!!!!
           ETERM(IMVDW) = ETERM(IMVDW) + ENBX
           ETERM(IMELEC)= ETERM(IMELEC)+ EELX
           ETERM(IMST2) = ETERM(IMST2) + EST2X
           !
        ENDIF

  !-----------------------------------------------------------------------
! FACTS screening interaction 06.06.2011
#if KEY_FACTS==1
   IF(FCTRUN) THEN
      if (fctaim) then
         ! Create Copy of interaction arrays
         if(allocated(FCTBNDC%fct2ilo))write(outu,*) "FCT2IL copy already allocated"
         if(allocated(FCTBNDC%fct2jnb))write(outu,*) "FCT2IB copy already allocated"
         if(allocated(FCTBNDC%fct4ilo))write(outu,*) "FCT4ILO already allocated"
         if(allocated(FCTBNDC%fct4jnb))write(outu,*) "FCT4JNB already allocated"

         call chmalloc('intere.src','INTER2','FCT2ILOC', natom,  intg=FCTBNDC%fct2ilo)
         call chmalloc('intere.src','INTER2','FCT2JNBC', mxfcab, intg=FCTBNDC%fct2jnb)
         call chmalloc('intere.src','INTER2','FCT4ILOC', 26*natom, intg=FCTBNDC%fct4ilo)
         call chmalloc('intere.src','INTER2','FCT4JNBC', mxfcib  , intg=FCTBNDC%fct4jnb)

         FCTBNDC%fct2ilo = 0
         FCTBNDC%fct2jnb = 0
         FCTBNDC%fct4ilo = 0
         FCTBNDC%fct4jnb = 0

         ! Select interaction lists
         CALL INTER3('FACTS Inter List',ISLCT,JSLCT,NATOM,NGRP, &
             FCTBNDC%fct2ilo , FCTBNDC%fct2jnb, &
             FCTBND%fct2ilo  , FCTBND%fct2jnb,&
             bnbndc%INBLOG &
             ! BDUMMY%INBLOG &
#if KEY_IMCUBES==1
             ,lbycbim                         & 
#endif
             )

         CALL INTER3('FACTS Image Inter List',ISLCT,JSLCT,NATOM,NGRP, &
             FCTBNDC%fct4ilo , FCTBNDC%fct4jnb, &
             FCTBND%fct4ilo  , FCTBND%fct4jnb,&
             !bnbndc%INBLOG &
             BDUMMY%INBLOG &
#if KEY_IMCUBES==1
             ,lbycbim                         & 
#endif
             )

         if(.not. fctscro)then
            do i=1,natomt
               fctslfw(i) = one
               if(islct(i) /= 1) then
                  fctslfw(i) = zero
               endif
            enddo
         endif

         ! Call the FACTS energy subroutine
         call fctene(eterm(ifctpol)                 ,&
                    eterm(ifctnpl)                 ,&
                    natom                          ,&
                    ! BNBND%inblo ,BNBND%jnb         ,&
                    BNBNDC%inblo ,BNBNDC%jnb         ,&
                    FCTBND%fct1ilo,FCTBND%fct1jnb  ,&
                    FCTBNDC%fct2ilo,FCTBNDC%fct2jnb,&
                    natim                          ,&
                    ! BIMAG%imblo ,BIMAG%imjnb       ,&
                    BDUMMY%INBLO,BDUMMY%JNB        ,&
                    !BIMAG%imattr                   ,&
                    BIMAGC%imattr                   ,&
                    FCTBND%fct3ilo,FCTBND%fct3jnb  ,&
                    FCTBNDC%fct4ilo,FCTBNDC%fct4jnb)

         call chmdealloc('intere.src','INTER2','FCT2ILOC', natom,  intg=FCTBNDC%fct2ilo)
         call chmdealloc('intere.src','INTER2','FCT2JNBC', mxfcab, intg=FCTBNDC%fct2jnb)
         call chmdealloc('intere.src','INTER2','FCT4ILOC', 26*natom, intg=FCTBNDC%fct4ilo)
         call chmdealloc('intere.src','INTER2','FCT4JNBC', mxfcib  , intg=FCTBNDC%fct4jnb)
      endif
   ENDIF
#endif 
  !-----------------------------------------------------------------------
        !
        CALL FREEDT_nbond(BDUMMY)
        CALL FREEDT_image(BIMAGC)
     ENDIF
#if KEY_PBOUND==1
  ENDIF                  
#endif
  ! F.M. 2011
#if KEY_GRID==1
  If(QFRE) Then  
#endif
     CALL FREEDT_nbond(BNBNDC)
#if KEY_GRID==1
  EndIf  
#endif
  !
  !-----------------------------------------------------------------------
  ! Hydrogen-bond energy term
  IF(NHB.GT.0.AND.QETERM(HBOND)) THEN
#if KEY_PARALLEL==1
     IF(MYNOD.EQ.INODE(3)) THEN                         
#endif
        DO I=1,NHB
           IF (ISLCT(IHB(I)).EQ.1 .AND. JSLCT(JHB(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE IF (ISLCT(JHB(I)).EQ.1 .AND. JSLCT(IHB(I)).EQ.1) THEN
              ISKIP(I)=0
           ELSE
              ISKIP(I)=1
           ENDIF
        ENDDO
        CALL EHBOND(ETERM(HBOND),IHB,JHB,KHB,LHB,ICH,NHB,CHBA,CHBB, &
             DX,DY,DZ,X,Y,Z,QECONT,ECONT,0,0,1,ISKIP, &
             CTONHB,CTOFHB,CTONHA,CTOFHA,HBEXPN,0,0,.FALSE.)
#if KEY_PARALLEL==1
     ENDIF                                              
#endif
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Harmonic restraint energy term
  IF(QCNSTR.AND.QETERM(CHARM)) THEN
     DO I=1,NATOM
        IF (ISLCT(I).EQ.1 .AND. JSLCT(I).EQ.1) THEN
           RTEMP(I)=KCNSTR(I)
        ELSE
           RTEMP(I)=ZERO
        ENDIF
     ENDDO
     CALL ECNSTR(ETERM(CHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          QECONT,ECONT, (/ ZERO /), (/ 0 /), .FALSE. &
          ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
          )

  ENDIF
  !
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  ! PCA Harmonic restraint energy term
  IF(QCNSTR.AND.QETERM(PCHARM).and.numpca>0) THEN
     DO I=1,NATOM
        IF (ISLCT(I).EQ.1 .AND. JSLCT(I).EQ.1) THEN
           RTEMP(I)=KCNSTR(I)
        ELSE
           RTEMP(I)=ZERO
        ENDIF
     ENDDO
     CALL ECNSTR(ETERM(PCHARM),QCNSTR,REFX,REFY,REFZ,RTEMP,NATOM, &
          KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
          NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
          X,Y,Z,DX,DY,DZ, &
          QECONT,ECONT, (/ ZERO /), (/ 0 /), .FALSE. &
          ,numpca,pcax,pcay,pcaz &
          )
  ENDIF
  !
  !-----------------------------------------------------------------------
  ! Dihedral restraint energy term
  IF((NCSPHI.GT.0) .AND. QETERM(CDIHE)) THEN
     DO I=1,NCSPHI
        IF (ISLCT(JCS(I)).EQ.1 .AND. JSLCT(KCS(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE IF (ISLCT(KCS(I)).EQ.1 .AND. JSLCT(JCS(I)).EQ.1) THEN
           ISKIP(I)=0
        ELSE
           ISKIP(I)=1
        ENDIF
     ENDDO
#if KEY_DOMDEC==1
     if (q_domdec) then
        CALL WRNDIE(-5,'<INTER2>','HARMONIC RESTRAINTS NOT READY FOR DOMDEC')
     else
#endif 
        CALL EPHI(ETERM(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
             CCSC,CCSD,CCSB,CCSCOS,CCSSIN,DX,DY,DZ,X,Y,Z,.TRUE.,CCSW, &
             QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
             )
#if KEY_DOMDEC==1
     endif  
#endif

  ENDIF
  !
  !-----------------------------------------------------------------------
#if KEY_DOMDEC==1
  if (.not.q_domdec) then  
#endif
#if KEY_PBOUND==1
  IF(.NOT.QBOUN) THEN     
#endif
     ! . Backtransform image forces
     IF(NTRANS.GT.0) THEN
        !
        ! . Calculate crystal lattice first derivatives.
        CALL TRANSI(X,Y,Z,DX,DY,DZ,QECONT,ECONT,NATOM,NTRANS, &
             IMTRNS,bimag%IMATPT,bimag%IMATTR, &
             NOROT,NATIM,IMINV,IMFORC,IMTORQ &
#if KEY_FLUCQ==1
             ,QFLUC,FQCFOR    & 
#endif
             )
     ENDIF
#if KEY_PBOUND==1
  ENDIF                  
#endif
#if KEY_DOMDEC==1
  endif  
#endif
  !-----------------------------------------------------------------------
  ! Finish up...
  EXX=ZERO
  DO I=1,LENENT
     EXX=EXX+ETERM(I)
  ENDDO
  EPROP(EPOT)=EXX

#if KEY_PARALLEL==1
  IF(NUMNOD.GT.1) THEN
     CALL VDGSUM(DX,DY,DZ,0)
     CALL VDGBR(DX,DY,DZ,1)
     !
     ! Pack assorted results into a single array for efficiency
     ! in using the global sum operation.
     !
     !      VIRI    - internal virial
     !      VIRE    - external virial
     !      VIRKE   - virial energy ( <f|r> )
     !
     IPT=1
     ALLWRK(IPT)=EPROP(EPOT)
     DO I=1,LENENT
        IPT=IPT+1
        ALLWRK(IPT)=ETERM(I)
     ENDDO

     CALL GCOMB(ALLWRK,IPT)

     IPT=1
     EPROP(EPOT)=ALLWRK(IPT)
     DO I=1,LENENT
        IPT=IPT+1
        ETERM(I)=ALLWRK(IPT)
     ENDDO
  ENDIF
#endif 

  DO I=1,NATOM
     IF(IMOVE(I).GT.0) THEN
        DX(I)=ZERO
        DY(I)=ZERO
        DZ(I)=ZERO
     ENDIF
  ENDDO

  RETURN
END SUBROUTINE INTER2

SUBROUTINE INTER3(CALLNAME,ISLCT,JSLCT,NATOM,NGRP, &
     INBLO,JNB,INBLOX,JNBX,INBLOG &
#if KEY_IMCUBES==1
     ,lbycbim                               & 
#endif
     )
  !-----------------------------------------------------------------------
  ! Make new nonbond lists based on selected atoms.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  implicit none

  character(len=*), intent(in) :: CALLNAME
  INTEGER NATOM,NGRP
  INTEGER ISLCT(*),JSLCT(*)
  INTEGER INBLO(NATOM),INBLOX(NATOM),INBLOG(NGRP)
  INTEGER JNB(*),JNBX(*)

  INTEGER K,I,IFIRST,N,ILAST,J
#if KEY_IMCUBES==1
  logical lbycbim                          
#endif
  integer nmax,mmax

#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
  if(lbycbim)then                      
#endif
     nmax=0
  else
#endif 
     nmax=INBLOX(NATOM)
#if KEY_IMCUBES==1
  endif
#endif 

  IFIRST=1
  N=0
  DO I=1,NATOM
#if KEY_IMCUBES==1
#if KEY_IMCUBES==1
     if(lbycbim)then                   
#endif
        ifirst=INBLOX(natom+I)+1
        inblo(i+natom)=N
        nmax=max(nmax,inblox(i))
     endif
#endif 
     ILAST=INBLOX(I)
     IF (ISLCT(I).EQ.1 .AND. JSLCT(I).EQ.1) THEN
        DO J=IFIRST,ILAST
           K=JNBX(J)
           IF(K.LT.0) K=-K
           IF(ISLCT(K).EQ.1 .OR. JSLCT(K).EQ.1) THEN
              N=N+1
              JNB(N)=JNBX(J)
           ENDIF
        ENDDO
     ELSE IF (ISLCT(I).EQ.1) THEN
        DO J=IFIRST,ILAST
           K=JNBX(J)
           IF(K.LT.0) K=-K
           IF(JSLCT(K).EQ.1) THEN
              N=N+1
              JNB(N)=JNBX(J)
           ENDIF
        ENDDO
     ELSE IF (JSLCT(I).EQ.1) THEN
        DO J=IFIRST,ILAST
           K=JNBX(J)
           IF(K.LT.0) K=-K
           IF(ISLCT(K).EQ.1) THEN
              N=N+1
              JNB(N)=JNBX(J)
           ENDIF
        ENDDO
     ENDIF
     INBLO(I)=N
     IFIRST=ILAST+1
  ENDDO

  DO I=1,NGRP
     INBLOG(I)=0
  ENDDO

  IF(PRNLEV.GT.2) WRITE(OUTU,55) CALLNAME,N,nmax

55 FORMAT(' INTER:  ',A19,'selected',I12,' from a total of',I12)
#if KEY_DEBUG==1
  WRITE(20,'(A)') ' INTERACTION ATOM NONBOND LIST:'
  CALL PRNBD2(20,N,NATOM,INBLO,JNB)
#endif 
  !
  !-----------------------------------------------------------------------
  !
  RETURN
END SUBROUTINE INTER3

SUBROUTINE INTER3G(CALLNAME,ISLCT,JSLCT, &
     INBLOG,JNBG,INBLOGX,JNBGX)
  !-----------------------------------------------------------------------
  ! Just like inter3 but for group nonbonded list
  ! If any atom in a group is selected, the whole group is selected
  ! T. Lazaridis
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use psf
  implicit none

  CHARACTER(len=*), intent(in) :: CALLNAME
  INTEGER ISLCT(*),JSLCT(*)
  INTEGER INBLOG(NGRP),INBLOGX(NGRP)
  INTEGER JNBG(*),JNBGX(*)

  INTEGER K,I,ITEMP,N,ILAST,J,IRS,JRS,JRSPR,NPR,NB
  INTEGER IS,IQ,JS,JQ
  LOGICAL ILFIRST,ILSECND,ILBOTH
  LOGICAL JLFIRST,JLSECND,JLBOTH

  ITEMP=0
  N=0
  NB=0
  ! First group
  DO IRS=1,NGRP
     NPR= INBLOGX(IRS)- ITEMP
     ITEMP= INBLOGX(IRS)
     IS= IGPBS(IRS)+1
     IQ= IGPBS(IRS+1)
     ILBOTH=  .FALSE.
     ILFIRST= .FALSE.
     ILSECND= .FALSE.
     ! Atoms of first group
     DO I=IS,IQ
        IF (ISLCT(I).EQ.1 .AND. JSLCT(I).EQ.1) THEN
           ILBOTH= .TRUE.
        ELSEIF (ISLCT(I).EQ.1) THEN
           ILFIRST= .TRUE.
        ELSEIF (JSLCT(I).EQ.1) THEN
           ILSECND= .TRUE.
        ENDIF
     ENDDO
     ! Second group
     DO JRSPR=1,NPR
        N=N+1
        JRS=JNBGX(N)
        IF (JRS.LT.0) JRS=-JRS
        JS= IGPBS(JRS)+1
        JQ= IGPBS(JRS+1)
        JLBOTH=  .FALSE.
        JLFIRST= .FALSE.
        JLSECND= .FALSE.
        ! Atoms of second group
        DO J=JS,JQ
           IF (ISLCT(J).EQ.1 .AND. JSLCT(J).EQ.1) THEN
              JLBOTH= .TRUE.
           ELSEIF (ISLCT(J).EQ.1) THEN
              JLFIRST= .TRUE.
           ELSEIF (JSLCT(J).EQ.1) THEN
              JLSECND= .TRUE.
           ENDIF
        ENDDO
        ! Include or not include into new group list
        IF ((ILBOTH.AND.(JLBOTH.OR.JLFIRST.OR.JLSECND)) .OR. &
             (ILFIRST.AND.JLSECND).OR.(ILSECND.AND.JLFIRST)) THEN
           NB=NB+1
           JNBG(NB)= JNBGX(N)
        ENDIF
     ENDDO
     INBLOG(IRS)= NB
  ENDDO
  !-----------------------------------------------------------------------
  !
  RETURN
END SUBROUTINE INTER3G

