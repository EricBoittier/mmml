module ecntrl

contains

subroutine usefasterroutines(qf)
  ! the FASTER routines specifically strip out a lot of code for the
  ! purposes of efficiency. As per the comments preceeding the FASTER
  ! section of enefscal,
  !
  !     The following have been stripped, and thus must be ruled out in 
  !     ecntrl.src:
  !      ACTBOND (check NACTIVE, NACTC, ...)
  !      BLOCK (check QBLOCK)
  !      DOCK (handled by checking QBLOCK, but we also check QDOCK)
  !      GENETIC (check nChromos)
  !      LDM (cannot occur outside of BLOCK, so handled by QBLOCK)
#if KEY_ACTBOND==1
  use actclus_mod 
#endif
#if KEY_BLOCK==1
  use block_fcm
  use lambdam
  use pert  !Cc New PBLOCK
#endif 
#if KEY_TSALLIS==1
  use tsallis_module, only: qttsall, qtalpha
#endif 
#if KEY_GENETIC==1
  use galgor 
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
#endif
  logical qf
  qf = .true.
#if KEY_ACTBOND==1
  if (NACTIVE > 0 .or. NACTC > 0 .or. NACTG > 0 .or. &
       NACTBND > 0 .or. NACTANG > 0 .or. NACTDIH > 0 .or. &
       NACTIMP > 0) then
#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<ecntrl>','ACTBOND not supported in DOMDEC')
     endif
#endif
     qf = .false.
     return
  endif
#endif 
#if KEY_BLOCK==1
  if (QBLOCK) then
     qf = .false.
     return
  endif
#endif 
#if KEY_BLOCK==1
#if KEY_DOCK==1
  if (QDOCK) then
#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<ecntrl>','DOCK not supported in DOMDEC')
     endif
#endif
     qf = .false.
     return
  endif
#endif 
#endif
#if KEY_GENETIC==1
  if (nChromos.GT.0) then
#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<ecntrl>','GENETIC not supported in DOMDEC')
     endif
#endif
     qf = .false.
     return
  endif
#endif 
#if KEY_TSALLIS==1
  if(qttsall.or.qtalpha) then
#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-5,'<ecntrl>','TSALLIS not supported in DOMDEC')
     endif
#endif
     qf = .false.
     return
  endif
#endif 
return
end subroutine usefasterroutines

SUBROUTINE EBONDC(EB, IB, JB, ICB, NB, CBC, CBB, &
     DX, DY, DZ, X, Y, Z, &
     DD1, IUPT, QSECD, NAT)
  !
  ! Control routine for bond energy calculation
  !
  !      EB               - Bond energy returned
  !      IB(*),JB(*)      - Bond list
  !      ICB(*)           - Bond codes array
  !      NB               - Number of bonds to compute
  !      DX(*),DY(*),DZ(*)- Forces returned
  !      X(*),Y(*),Z(*)   - Coordinates
  !      DD1(*)           - Second derivative arrays
  !      IUPT(*)          - Upper triangle pointer array
  !      QSECD            - Second derivative flags
  !      NAT              - Number of atoms in calculation domain
  !
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use eintern
  use eintern_fast
  use fast
  use stream
  use ffieldm 
#if KEY_MMFF==1
  use mmffm
  use escalar_mm
#endif 
#if KEY_CFF==1
  use cff_fcm
#endif 
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec        
  use ebonded_domdec,only:ebond_domdec   
#endif
  use bond_tables,only:lbond_tables,ebond_table
  implicit none
  !
  real(chm_real) EB
  INTEGER IB(:),JB(:),ICB(:), NB
  real(chm_real) CBC(:), CBB(:)
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER NAT
  LOGICAL QFASTERROUTINES
  !
  !
#if KEY_CFF==1 || KEY_MMFF==1
  INTEGER DERIVS
#endif 
  !

  IF(NB.LE.0) RETURN

  call usefasterroutines(QFASTERROUTINES)

#if KEY_DOMDEC==1
  if (q_domdec) then
     if(prnlev > 6) write(outu,125) 'EBOND_DOMDEC'
     call ebond_domdec(ib, jb, icb, cbb, cbc, eb)
     return
  endif
#endif 

  if(lbond_tables)then
     call ebond_table(EB,IB,JB,NB,DX,DY,DZ,X,Y,Z,qsecd)
     return
  end if

  !=======================================================================
  !
  IF (LFAST.GE.0) THEN
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        CALL WRNDIE(-3,'<ENERGY>','No MMFF FAST code compiled.')
        RETURN
     ENDIF
#endif 
     !----------------------------------------------------------------------
     !   Scalar FAST routines
     ! . Bond terms.
#if KEY_CFF==1
     IF(FFIELD.EQ.CFF)THEN
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBONDFS_CFF'
        CALL EBONDFS_CFF(EB)
     ELSE
#endif 
        IF(QFASTERROUTINES) THEN
           IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBONDFASTER'
           CALL EBONDFASTER(EB,NB,IB,JB,ICB,CBB,CBC)
        ELSE
           IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBONDFS'
           CALL EBONDFS(EB,NB,IB,JB,ICB,CBB,CBC)
        ENDIF
#if KEY_CHEQ==1
        IF(PRNLEV.GT.6) WRITE(OUTU,*)'EBONDC: returned from EBONDFS'
#endif
#if KEY_CFF==1
     ENDIF
#endif 
     !
     !----------------------------------------------------------------------
     ! . Use the slow routines.
  ELSE
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        DERIVS=1
        IF(QSECD) DERIVS=3
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBOND_MM'
        CALL EBOND_MM(EB,IB,JB,ICB,NB,CBB,CBC,AuxPar(CSTR), &
             X,Y,Z,DX,DY,DZ,LTSD,IUPT,DERIVS, &
             QECONT,ECONT,0, (/ 0 /))
        RETURN
     ENDIF
#endif 
     !
#if KEY_CFF==1
     IF(FFIELD.EQ.CFF)THEN
        DERIVS=1
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBOND_CFF'
        CALL EBOND_CFF(EB,IB,JB,ICB,NB, &
             X,Y,Z,DX,DY,DZ,DD1,DERIVS,QSECD)
        RETURN
     ENDIF
#endif 
     IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBOND'
     CALL EBOND(EB,IB,JB,ICB,NB,CBC,CBB, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
          0,(/0/),DD1,IUPT,QSECD,2,.FALSE.)

     !
  ENDIF
  !
125 FORMAT(' EBONDC: Using routine ',A,' for energy calculation.')
  RETURN
END SUBROUTINE EBONDC

SUBROUTINE EANGLC(ET, ITT, JTT, KTT, &
     ICT, NT, DX, DY, DZ, X, Y, Z, &
     DD1, IUPT, QSECD, MTYPET, NAT, &
     QFARRAY, ICB, EBB, ESTRB, ESS)
  !
  ! Control routine for angle energy calculation
  !
  !      ET               - Angle energy returned
  !      ITT(*),JTT(*),KTT(*)- Angle list
  !      ICT(*)           - Angle codes array
  !      NT               - Number of bonds to compute
  !      DX(*),DY(*),DZ(*)- Forces returned
  !      X(*),Y(*),Z(*)   - Coordinates
  !      DD1(*)           - Second derivative arrays
  !      IUPT(*)          - Upper triangle pointer array
  !      QSECD            - Second derivative flags
  !      MTYPET(*)        - Atom type array (for MMFF)
  !      NAT              - Number of atoms in calculation domain
  !      ICB(*)           - Bond codes array (for CFF)
  !      EBB, ESTRB, ESS  - Bend-bend, stretch-bend and stretch-stretch
  !                         energies returned (for CFF)
  !
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use eintern
  use eintern_fast
  use fast
  use param
  use stream
#if KEY_MMFF==1 || KEY_CFF==1
  use ffieldm    
#endif
#if KEY_CFF==1
  use psf
  use cff_fcm
#endif 
#if KEY_MMFF==1
  use mmffm
  use escalar_mm
  use vangle_mm
#endif 
  use bond_tables,only:langle_tables,eangle_table
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec      
  use ebonded_domdec,only:eangle_domdec  
#endif
  !
  implicit none
  real(chm_real) ET
  INTEGER ITT(:), JTT(:), KTT(:), ICT(:), NT
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER MTYPET(*), NAT
  LOGICAL QFARRAY
  INTEGER ICB(*)
  real(chm_real) EBB, ESTRB, ESS
  LOGICAL QFASTERROUTINES
  !
#if KEY_MMFF==1 || KEY_CFF==1
  INTEGER DERIVS
#endif 
  !
  IF(NT.LE.0) RETURN
  IF (PRNLEV.GT.7) THEN
     WRITE(OUTU,*) 'EANGLC: QANGTYPE = ', QANGTYPE
  ENDIF
  call usefasterroutines(QFASTERROUTINES)

#if KEY_DOMDEC==1
  if (q_domdec) then
     if(prnlev > 6) write(outu,125) 'EANGLE_DOMDEC'
     call eangle_domdec(itt, jtt, ktt, ict, ctb, ctc, et)
     return
  endif
#endif 

  ! qangtype = 0 (mixed angle potentials) is not supported in the fast routines at all.
  ! eanglefs supports qangtype = 1.
  ! eanglefaster and eanglefastergrom support 1 and -1.
  if(langle_tables)then
     call eangle_table(et,itt,jtt,ktt,nt,dx,dy,dz,x,y,z,qsecd)
     return
  end if

  !=======================================================================
  !
  IF (LFAST.GE.0 .AND. QFARRAY .AND. &
       ((QANGTYPE.EQ.1).OR.(QFASTERROUTINES.AND.(QANGTYPE.EQ.-1))) &
       ) THEN
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        CALL WRNDIE(-3,'<ENERGY>','No MMFF FAST code compiled.')
        RETURN
     ENDIF
#endif 
     !----------------------------------------------------------------------
     !   Scalar FAST routines
     !
#if KEY_CFF==1
     IF(FFIELD.EQ.CFF)THEN
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGLFS_CFF'
        CALL EANGLFS_CFF(ET)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBNDANGFS'
        CALL EBNDANGFS(ESS,ESTRB)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGANGFS'
        CALL EANGANGFS(EBB)
     ELSE
#endif 
        IF(QFASTERROUTINES) THEN
           !              QANGTYPE .EQ. 0 rejected above
           IF(QANGTYPE .EQ. 1) THEN
              IF(PRNLEV .GT. 6) WRITE(OUTU,125) 'EANGLFASTER'
              CALL EANGLFASTER(ET)
           ELSE IF (QANGTYPE .EQ. -1) THEN
              IF (PRNLEV .GT. 6) WRITE(OUTU,125) 'EANGLFASTERGROM'
              CALL EANGLFASTERGROM(ET)
           ELSE
              CALL WRNDIE(-3,'<ENERGY>', &
                   'FAST routines not supported with mixed CHARMM and GROMACS angle potentials')
              RETURN
           ENDIF
        ELSE
           IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGLFS'
           CALL EANGLFS(ET)
        ENDIF
#if KEY_CFF==1
     ENDIF
#endif 
     !
     !----------------------------------------------------------------------
     ! . Use the slow routines.
  ELSE
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        IF (QANGTYPE .NE. 1) THEN
           CALL WRNDIE(-3,'<ENERGY>','MMFF is NOT COMPATIBLE with GROMACS-style angle parameters.')
           RETURN
        ENDIF
        DERIVS=1
        IF(QSECD) DERIVS=3
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGLE_MM'
        CALL EANGLE_MM(ET,ITT,JTT,KTT,ICT,NT,CTB,CTC,AuxPar(CBND), &
             X,Y,Z,DX,DY,DZ,LTSD,IUPT,DERIVS, &
             QECONT,ECONT,0, (/ 0 /))
        RETURN
     ENDIF
#endif 
     !
#if KEY_CFF==1
     IF (FFIELD.EQ.CFF) THEN
        IF(QANGTYPE .NE. 1) THEN
           CALL WRNDIE(-3,'<ENERGY>','CFF is NOT COMPATIBLE with GROMACS-style angle parameters.')
           RETURN
        ENDIF
        DERIVS=1
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGLE_CFF'
        CALL EANGLE_CFF(ET,ITT,JTT,KTT,ICT,NT,X,Y,Z,DX,DY,DZ,DD1, &
             DERIVS,QSECD)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBNDANG'
        CALL EBNDANG(ESS,ESTRB,ITT,JTT,KTT,ICT,NT,ICB,IAC, &
             X,Y,Z,DX,DY,DZ,DD1,DERIVS,QSECD)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGANG'
        ! BIOVIA Code Start : Bug fix
        CALL EANGANG(EBB,ICB,ICT,X,Y,Z,DX,DY,DZ,DD1,DERIVS,QSECD)
        ! BIOVIA Code End
        RETURN
     ENDIF
#endif 

     IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EANGLE'
     CALL EANGLE(ET,ITT,JTT,KTT,ICT,NT,CTC,CTB, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
          0,(/0/),DD1,IUPT,QSECD &
          )

     !
  ENDIF
  !
125 FORMAT(' EANGLC: Using routine ',A,' for energy calculation.')
  RETURN
END SUBROUTINE EANGLC

SUBROUTINE EPHIC(EP,IP,JP,KP,LP,ICP, NP, DX, DY, DZ, &
     X, Y, Z, DD1,IUPT,QSECD, NAT, &
     QDIM4,QFARRAY, IB, ICB, ICT, EBBT, EBNDT, &
     EEBST, EMBST, ESST)
  !
  ! . Proper dihedral terms.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use eintern
  use eintern_fast
  use fast
  use fourdm
  use param
  use stream
  use escalar_mm
  use vangle_mm
#if KEY_MMFF==1 || KEY_CFF==1
  use ffieldm        
#endif
  use bond_tables,only:ldihedral_tables,ephi_table
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec         
  use ebonded_domdec,only:edihe_domdec    
#endif
  implicit none
  !
  real(chm_real) EP,EBBT,EBNDT,EEBST,EMBST,ESST
  INTEGER IP(:), JP(:), KP(:), LP(:), ICP(:), NP
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER NAT
  LOGICAL QDIM4, QFARRAY
  INTEGER IB(*),ICB(*),ICT(*)
  LOGICAL QFASTERROUTINES
  !
  !
  !
#if KEY_MMFF==1 || KEY_CFF==1
  INTEGER DERIVS
#endif 
  !
  IF(NP.LE.0) RETURN
  !=======================================================================
  !

  call usefasterroutines(QFASTERROUTINES)

#if KEY_DOMDEC==1
  if (q_domdec) then
     if(prnlev > 6) write(outu,125) 'EDIHE_DOMDEC'
     call edihe_domdec(ip, jp, kp, lp, icp, cpd, cpc, cpsin, cpcos, ep)
     return
  endif
#endif 

#if KEY_FOURD==1 /*4ddihe*/
  !    In 4-D we have:
  IF(DIM4.AND.QDIM4) THEN
     IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHI4'
     CALL EPHI4(EP,IP,JP,KP,LP,ICP,NP,CPC,CPD,CPB, &
          DX,DY,DZ,X,Y,Z,QECONT,ECONT,0,0,DD1,IUPT, &
          QSECD)
     RETURN
  ENDIF
#endif /* (4ddihe)*/

  if (ldihedral_tables) then
     call ephi_table(ep,ip,jp,kp,lp,np,dx,dy,dz,x,y,z,qsecd)
     return
  endif

  !
  !----------------------------------------------------------------------
  IF (LFAST.GE.0 .AND. QFARRAY) THEN
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        CALL WRNDIE(-3,'<ENERGY>','No MMFF FAST code compiled.')
        RETURN
     ENDIF
#endif 
     !----------------------------------------------------------------------
     !   Scalar FAST routines
     !
#if KEY_CFF==1
     IF(FFIELD.EQ.CFF)THEN
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHIFS_CFF'
        CALL EPHIFS_CFF(EP,EBNDT,EEBST,EMBST,EBBT)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBNDBNDFS'
        CALL EBNDBNDFS(ESST)
     ELSE
#endif 
        IF(QFASTERROUTINES) THEN
           IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHIFASTER'
           CALL EPHIFASTER(EP)
        ELSE
           IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHIFS'
           CALL EPHIFS(EP)
        ENDIF
#if KEY_CFF==1
     ENDIF
#endif 
     !
     !----------------------------------------------------------------------
     ! . Use the slow routines.
  ELSE
     !
#if KEY_MMFF==1
     IF (FFIELD.EQ.MMFF) THEN
        DERIVS=1
        IF(QSECD) DERIVS=3
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHI_MM'
        CALL EPHI_MM(EP,IP,JP,KP,LP,ICP,NP,CPC, &
             X,Y,Z,DX,DY,DZ,LTSD,IUPT,DERIVS, &
             QECONT,ECONT,0, (/ 0 /))
        RETURN
     ENDIF
#endif 
     !
#if KEY_CFF==1
     IF(FFIELD.EQ.CFF)THEN
        DERIVS=1
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHI_CFF'
        CALL EPHI_CFF(EP,IP,JP,KP,LP,ICP,NP,ICT,IB,ICB, &
             X,Y,Z,DX,DY,DZ,DD1,DERIVS,QSECD, &
             EBNDT,EEBST,EMBST,EBBT)
        IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EBNDBND'
        CALL EBNDBND(ESST,IP,JP,KP,LP,ICP,IB,ICB, &
             X,Y,Z,DX,DY,DZ,DD1,DERIVS,QSECD)
        RETURN
     ENDIF
#endif 
     !
     IF(PRNLEV.GT.6) WRITE(OUTU,125) 'EPHI'
     CALL EPHI(EP,IP,JP,KP,LP,ICP,NP,CPC,CPD,CPB, &
          CPCOS,CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
          QECONT,ECONT,0,(/0/),DD1,IUPT,QSECD)
     !
  ENDIF
  !
125 FORMAT(' EPHIC: Using routine ',A,' for energy calculation.')
  RETURN
END SUBROUTINE EPHIC

SUBROUTINE EIMPHIC(EI,IM,JM,KM,LM,ICI, NI, DX, DY, DZ, &
     X, Y, Z, DD1,IUPT,QSECD, NAT, &
     QFARRAY)
  !
  ! . Improper dihedral terms.
  !
  use chm_kinds
  use dimens_fcm
  use number
  use econtmod
  use eintern
  use eintern_fast
  use fast
  use param
  use stream
#if KEY_MMFF==1
  !C  use mmffm
  use vangle_mm
#endif 
  use ffieldm
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec             
  use ebonded_domdec,only:eimdihe_domdec      
#endif
#if KEY_BLOCK==1
  use lambdam,only:qmld
  use block_fcm,only:qnoim
#endif
  !
  implicit none
  !
  real(chm_real) EI
  INTEGER IM(:), JM(:), KM(:), LM(:), ICI(:), NI
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER NAT
  LOGICAL QFARRAY
  LOGICAL QFASTERROUTINES
  !
  !
#if KEY_MMFF==1 || KEY_CFF==1
  INTEGER DERIVS
#endif 
  !
  !
  LOGICAL QF
  character(len=57) :: fmt125

  fmt125="(' EIMPHIC: Using routine ',a,' for energy calculation.')"

  if(ni.le.0) return
  !=======================================================================
  !

  call usefasterroutines(qfasterroutines)
#if KEY_DOMDEC==1
  if (q_domdec) then
     if(prnlev > 6) write(outu,fmt125) 'EIMDIHE_DOMDEC'
     call eimdihe_domdec(im, jm, km, lm, ici, cid, cic, cisin, cicos, ei)
     return
  endif
#endif 

#if KEY_BLOCK==1
  ! APH 11/7/2013: Warning added when scaling impropers with mld.
  ! These are not implemented in the bonded calculation
  if (qmld .and. .not.qnoim) then
     call wrndie(-5,'<ecntrl>',&
          'Warning: Improper torsion angle scaling are not implemented')
  endif
#endif 

#if KEY_AMBER==1
  qf=.false.
#else /**/
  ! Never use the fast improper routines with AMBER ff.
  qf=qfarray
#endif 
  if (lfast.ge.0 .and. qf) then
     !----------------------------------------------------------------------
     !   Scalar FAST routines
     select case(ffield)
#if KEY_CFF==1
     case(cff)
        if(prnlev.gt.6) write(outu,fmt125) 'EOPLNFS_CFF'
        call eoplnfs_cff(ei)
#endif 
     case(charmm,mmff)
        call usefasterroutines(qfasterroutines)
        if(qfasterroutines) then
           if(prnlev.gt.6) write(outu,fmt125) 'EIPHIFASTER'
           call eiphifaster(ei)
        else
           if(prnlev.gt.6) write(outu,fmt125) 'EIPHIFS'
           call eiphifs(ei)
        endif
     case(amberffn)
        call ephi(ei,im,jm,km,lm,ici,ni,cic,cid,cib, &
             cicos,cisin,dx,dy,dz,x,y,z,.false.,(/zero/), &
             qecont,econt,0,(/0/),dd1,iupt,qsecd &
             )
     case default
        call wrndie(-2,"ecntrl(enctrl.src) no valid forcefield", &
             "no call for impropers made") 
     end select

     !----------------------------------------------------------------------
     ! . Use the slow routines.
  else
     select case(ffield)
#if KEY_CFF==1
     case(cff)
        derivs=1
        if(prnlev.gt.6) write(outu,fmt125) 'EOPLN_CFF'
        call eopln_cff(ei,im,jm,km,lm,ici,ni, &
             x,y,z,dx,dy,dz,dd1,derivs,qsecd)
#endif 
     case(charmm,mmff)
        if(prnlev.gt.6) write(outu,fmt125)'EPHI'
        CALL EPHI(EI,IM,JM,KM,LM,ICI,NI,CIC,CID,CIB, &
             CICOS,CISIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
             QECONT,ECONT,0,(/0/),DD1,IUPT,QSECD)
     case(amberffn)
        call ephi(ei,im,jm,km,lm,ici,ni,cic,cid,cib, &
             cicos,cisin,dx,dy,dz,x,y,z,.false.,(/zero/), &
             qecont,econt,0,(/0/),dd1,iupt,qsecd)
     end select
  endif
  !
  return
end subroutine eimphic

#if KEY_CMAP==1 /*cmap*/
SUBROUTINE ECMAPC(EC,I1CT,J1CT,K1CT,L1CT, &
     I2CT,J2CT,K2CT,L2CT, &
     ICCT, NI, DX, DY, DZ, &
     X, Y, Z, DD1,IUPT,QSECD, NAT, &
     QFARRAY)
  !
  ! . Cross-term maps
  !
  use chm_kinds
  use dimens_fcm
  use econtmod
  use fast
  use param
  use stream
  use cmapm
  implicit none
  !
  real(chm_real) EC
  INTEGER I1CT(*), J1CT(*), K1CT(*), L1CT(*), &
       I2CT(*),J2CT(*),K2CT(*),L2CT(*),ICCT(*), NI
  real(chm_real) DX(*),DY(*),DZ(*)
  real(chm_real) X(*),Y(*),Z(*)
  real(chm_real) DD1(*)
  INTEGER IUPT(*)
  LOGICAL QSECD
  INTEGER NAT
  LOGICAL QFARRAY
  !
  !
  LOGICAL QF
  !
  IF(NI.LE.0) RETURN
  !
  IF(PRNLEV.GT.6) WRITE(OUTU,126) 'ECMAP'
  CALL ECMAP(EC,I1CT,J1CT,K1CT,L1CT, &
       I2CT,J2CT,K2CT,L2CT,ICCT,NI, &
       DX,DY,DZ,X,Y,Z, &
       QECONT,ECONT,0, (/0/), DD1,IUPT,QSECD)
  !
126 FORMAT(' ECMAPC: Using routine ',A, &
       ' for energy calculation.')
  RETURN
END SUBROUTINE ECMAPC
#endif /*  (cmap)*/

end module ecntrl

