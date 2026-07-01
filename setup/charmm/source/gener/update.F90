SUBROUTINE UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,QPARSE, &
     LCODES,LNBOND,LHBOND,LIMAGE,LDYNAM,XOLD,YOLD,ZOLD,VX,VY,VZ)
  !-----------------------------------------------------------------------
  !     This routine updates the nbond, hbond, image, and codes lists.
  !     Things are updated in the proper order.
  !
  !     By Bernard R. Brooks    1983
  !
  !     Rick Lapp 25-May-98: added modifications for CFF
  !
  !     Philippe Lagant and Roland Stote 11/01: SPASIBA force force added
  !     SPASIBA code removed for c34b1 and c35a1 releases
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
  use code
  use contrl
  use energym
  use fast
  use fourdm
  use hbondm
  use image
  use imgup
  use inbnd
  use param
  use pert
  use psf
  use shake
  use stream
  use string
  use tsms_mod
#if KEY_PARALLEL==1
  use parallel           
#endif
  use ffieldm
  use icfix
  use lonepr
#if KEY_RXNCONS==1
  use rxncons,only:irxcns  
#endif
  use shapes
  !
  use blockscc_fcm
  use sccdftb
  use sccdftbsrc
  use exclm
#if KEY_MMFF==1
  use vangle_mm     
#endif
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec
  use domdec,only:init_domdec1, init_domdec2
  use domdec_dr_common,only:q_direct_node
!!$  use domdec_local,only:build_local_vdwtype
!!$  use enbxfast,only:init_vdwparam
#endif
  use nbexcl,only:makitc
  use api_func, only: mlpot_update, mlpot_is_set   ! Kai Toepfer Jan. 2024
  !---   use nbutil_module,only:gtnbct
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  LOGICAL QPARSE
  LOGICAL LCODES,LNBOND,LHBOND,LIMAGE,LTEMP
  INTEGER LDYNAM
  real(chm_real) XOLD(*),YOLD(*),ZOLD(*),VX(*),VY(*),VZ(*)
#if KEY_PARALLEL==1
  INTEGER IGRP,IQ,NTRY
#endif
#if KEY_PERT==0
! arrays of zero to pass to makitc
! replace with optional argument version of makitc
  integer, dimension(natom) :: dummy_ppiac, dummy_ppiacnb
#endif
  !RCZ  25-OCT-91
  !     to exclud enonbond interactions among the selected segments
  !     1991/05/07 RYSZARD CZERMINSKI (RCZ)
  !     excl.f90 added to handle user specified exclusions from
  !     nonbonded interactions. UPDATE, NBONDS and NBONDG have been
  !     modified.  A logical function INLIST is added.  New routine
  !     RLIST which reads optional list of segment names following EXSG
  !     keyword.
  !     New key words for UPDATE :
  !     EXSG - nonbonded interactions between all segements will
  !            be excluded
  !            optionally it is possible to include list of
  !            segment names (exsg <list>). If list is not empty
  !            all nonbonded interactions with and within listed
  !            segments will be excluded.
  !     EXCL - to mark begining of the selection of the next group
  !     EXOF - to remove extra exclusions
  INTEGER I
  integer, dimension(1), target :: pzer1 = [0]
  integer, dimension(:), pointer :: tinb14, tiblo14
  !RCZ
  !
  ! make sure that the contraint flag is properly set
  !
  QSHAKE=(NCONST > 0)
  QHOLO = QSHAKE
#if KEY_NOST2==0
  IF(NST2 > 0)  QHOLO=.TRUE.        
#endif
#if KEY_TSM==1
  IF(IICF > 0)  QHOLO=.TRUE.        
#endif
#if KEY_LONEPAIR==1
  IF(NUMLP > 0) QHOLO=.TRUE.        
#endif
#if KEY_FOURD==1
  IF(DIM4 .AND. QSHAK4) QHOLO=.TRUE. 
#endif
#if KEY_SHAPES==1
  IF(NUMSHPR > 0) QHOLO=.TRUE.      
#endif
#if KEY_RXNCONS==1
  IF(IRXCNS > 0) QHOLO=.TRUE.       
#endif
  !
#if KEY_TSM==1
  !     Make sure piggyback coordinates are taken care of.
  IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
#if KEY_NOST2==0
  ! Fix up ST2's if present (on an image or nonbond update).
  IF(LIMAGE .OR. LNBOND) THEN
     IF(NST2 > 0) CALL FIXST2(X,Y,Z,-1)
  ENDIF
#endif 

  !
  ! Process an IMAGE update if requested.
  !
  IF(LIMAGE) THEN
     IF(NTRANS > 0) THEN
        IF(QPARSE) THEN
           IMGFRQ=GTRMI(COMLYN,COMLEN,'IMGF',IMGFRQ)
           IXTFRQ=GTRMI(COMLYN,COMLEN,'IXTF',IXTFRQ)
           IF(INDXA(COMLYN,COMLEN,'IMAL') > 0) LIMALL=.TRUE.
           IF(INDXA(COMLYN,COMLEN,'IMBR') > 0) LIMALL=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'INVE') > 0) LIMINV=.TRUE.
           IF(INDXA(COMLYN,COMLEN,'NOIN') > 0) LIMINV=.FALSE.
           IF(INDXA(COMLYN,COMLEN,'IMOR') > 0) LMKMA1=.TRUE.
           CUTIM=GTRMF(COMLYN,COMLEN,'CUTI',CUTIM)
        ENDIF
        ! APH: For DOMDEC, we include the "force_upimag_on_all" parameter
        !      to the upimag call to make sure the images are updated.
        !      This can be done because at this stage all cores have all
        !      coordinates
        !SF... 17-Feb-93 Bugfix  >  is replaced by  /= 
        if(imgfrq /= 0) then
           call upimag(x,y,z,wmain,ldynam,xold,yold,zold,vx,vy,vz,.true.)
        endif
     ENDIF
  ENDIF
  !
  ! Update the nonbond lists if requested.
  !
  LTEMP=.FALSE.
  IF(LNBOND) THEN
     IF(QPARSE) THEN
        INBFRQ=GTRMI(COMLYN,COMLEN,'INBF',INBFRQ)
        !RCZ  25-OCT-91
        IF(INDXA(COMLYN,COMLEN,'EXSG') > 0) THEN
           LEXS=.TRUE.
           IF(PRNLEV >= 2) WRITE(OUTU,'(2A)') ' UPDATE> NONBONDED', &
                ' interactions between segments will be excluded.'
           !              RLIST reads from the command line (COMLYN) list of
           !              segment names and adds it to the SLIST
           IF(.NOT.ALLOCATED(SLIST)) CALL ALLOCATE_EXCL_LTM
           CALL RLIST(COMLYN,COMLEN,NEXS,SLIST,MAXSEG)
           IF(NEXS > 0 .AND. PRNLEV >= 2) WRITE(OUTU, &
                '(A,I5,A,/,10(1X,A8))') &
                ' UPDATE> LIST OF ',NEXS,' SELECTED SEGMENTS', &
                (SLIST(I),I=1,NEXS)
        ENDIF
        IF(INDXA(COMLYN,COMLEN,'EXOF') > 0) THEN
           LEXS=.FALSE.
           NEXS=0
        ENDIF
        !RCZ
        IF(INBFRQ /= 0) THEN
           CALL GTNBCT(COMLYN,COMLEN,BNBND)
           LTEMP=LVSHFT
        ENDIF
     ENDIF
#if KEY_SPACDEC==1
     IF(INBFRQ < 0) CALL WRNDIE(-1,'<UPDATE>', &
          '<UPDATE> SPACDEC compilation requires INBFreq > 0')
#endif 
     !
  ENDIF

#if KEY_LJPME==1
  call update_ljpme
#endif

  !
  ! Initialize DOMDEC, part 1
  !
#if KEY_DOMDEC==1
  if (q_domdec) then
     ! Initialize domdec (non-bonds)
     call init_domdec1(x, y, z)
  endif
#endif 

  if(lnbond) then
#if KEY_DOMDEC==1
     if (.not.q_domdec .or. (q_domdec .and. q_direct_node)) then
#endif
        IF(INBFRQ /= 0) THEN
#if KEY_PARASCAL==1
           QPSRNB=(.NOT.LCODES)  
#endif
           CALL NBONDS(X,Y,Z,BNBND,BIMAG)
#if KEY_SCCDFTB==1
           if(qsccnb) call mkmmlst    
#endif
        ENDIF
#if KEY_DOMDEC==1
     endif                   
#endif
  endif

  !
  ! Update the fast energy lists if needed.
  !
!!$#if KEY_DOMDEC==1
!!$  if (q_direct_node) then 
!!$#endif
     IF(LCODES) THEN
        IF(MUSTUP .OR. LTEMP) THEN
#if KEY_PERT==0
           dummy_ppiac = 0
           dummy_ppiacnb = 0
#endif
           CALL MAKITC(NATOMT,NATOM,IAC, &
#if KEY_PERT==1
                QPERT,PPIAC,PPIACNB & 
#endif
#if KEY_PERT==0
                .FALSE., dummy_ppiac, dummy_ppiacnb &         
#endif
                )
        ENDIF
     !
#if KEY_PARALLEL==1
        if (associated(BNBND%JNB)) then
           CALL PARUPDATE(BNBND%JNB,BNBND%INBLO)
        else
           CALL PARUPDATE([0],[0])
        endif
#endif 
     ENDIF
!!$#if KEY_DOMDEC==1
!!$  endif              
!!$#endif
  !
  ! Update the hydrogen bond lists.
  !
!!$#if KEY_DOMDEC==1
!!$  if (q_direct_node) then 
!!$#endif
     IF(LHBOND) THEN
        IF(QPARSE) THEN
           IHBFRQ=GTRMI(COMLYN,COMLEN,'IHBF',IHBFRQ)
           IF(IHBFRQ /= 0) CALL GTHBCT(COMLYN,COMLEN)
        ENDIF
        IF(IHBFRQ /= 0) THEN
           if (associated(bnbnd%iblo14) ) then
              tiblo14 => bnbnd%iblo14
              tinb14 => bnbnd%inb14
           else
              tiblo14 => pzer1
              tinb14 => pzer1
           endif
           CALL HBONDS(OUTU,IDON,IHD1,NDON,IACC,IAC1,NACC, &
                IHB,JHB,KHB,LHB,ICH,NHB,MAXHB,CHBA,CHBB,HBEXPN, &
                CTONHB,CTOFHB,CTONHA,CTOFHA,CUTHB,CUTHBA,LHBFG, &
                NCH,KCH,IAC,ATC,NATC,IMOVE,BEST,HBEXCL, &
                tiblo14,tinb14,NNB14, &
                X,Y,Z,NATOM)
           !
           IF(NATIM > 0.AND.NTRANS.GT.0) CALL IMHBON(BIMAG,X,Y,Z)
        ENDIF
     ENDIF
!!$#if KEY_DOMDEC==1
!!$  endif 
!!$#endif
  !
  ! Update the various codes lists if requested.
  !
!!$#if KEY_DOMDEC==1
!!$  if (q_direct_node) then 
!!$#endif
     IF(LCODES) THEN
        IF(.NOT.LHBOND) CALL HCODES(ICH,NHB,IHB,JHB,IAC)
        !
        IF(MUSTUP) THEN
           CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC, &
                NBOND,IB,JB,NTHETA,IT,JT,KT, &
                NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
                QDRUDE,NBDRUDE,                        & ! DRUDE
#if KEY_CMAP==1
                ICCT,NCRTERM, &
                I1CT,J1CT,K1CT,L1CT, &
                I2CT,J2CT,K2CT,L2CT, &
#endif 
                .FALSE.,.FALSE.)
           !           make vector lists.
#if KEY_MMFF==1
           IF(FFIELD == MMFF) THEN
              CALL MAKPHI_MM
              CALL MAKNGL_MM
           ELSE
#endif 
              CALL MAKPHI
#if KEY_MMFF==1
           ENDIF
#endif 
           MUSTUP=.FALSE.
        ENDIF
        !
        IF(NTRANS > 0 .AND. NATIM.GT.0) THEN
           !           process codes for image energy terms
           !
           CALL CODES(ICB(NBOND+1),ICT(NTHETA+1), &
                ICP(NPHI+1),ICI(NIMPHI+1), &
                NATOM,IMOVE,IAC,NIMBON,IB(NBOND+1),JB(NBOND+1), &
                NIMANG,IT(NTHETA+1),JT(NTHETA+1),KT(NTHETA+1), &
                NIMDIH,IP(NPHI+1),JP(NPHI+1),KP(NPHI+1), &
                LP(NPHI+1),NIMIMP,IM(NIMPHI+1),JM(NIMPHI+1), &
                KM(NIMPHI+1),LM(NIMPHI+1), &
                .FALSE.,0,                                  & ! DRUDE
#if KEY_CMAP==1
                ICCT(NCRTERM+1),NIMCRT, &
                I1CT(NCRTERM+1),J1CT(NCRTERM+1), &
                K1CT(NCRTERM+1),L1CT(NCRTERM+1), &
                I2CT(NCRTERM+1),J2CT(NCRTERM+1), &
                K2CT(NCRTERM+1),L2CT(NCRTERM+1), &
#endif 
                .FALSE.,.FALSE.)
           !
        ENDIF
     ENDIF
     
!!$#if KEY_DOMDEC==1
!!$  endif 
!!$#endif

  !
  ! Initialize DOMDEC, part 2
  !
#if KEY_DOMDEC==1
  if (q_domdec) then
     ! Initialize domdec (bonds, shake, cons)
     call init_domdec2()
  endif
#endif 

  !
  ! Update ML-ML and ML-MM atom pair lists for MLpot module
  ! Kai Toepfer, Jan. 2024
  !
  if(mlpot_is_set().and.(LNBOND.or.LIMAGE)) then
     call mlpot_update(                        &
        natom, ntrans, natim,                  &
        bnbnd%jnb, bnbnd%inblo,                &
        bimag%imattr, bimag%imjnb, bimag%imblo)
  endif

  RETURN
END SUBROUTINE UPDATE

#if KEY_LJPME==1
subroutine update_ljpme
  use chm_kinds, only: chm_real
  use psf, only: ljc6, iac, natom
  use param, only: itc, vdwr, eff, c6def

  integer :: atom, ljptr
  real(chm_real) :: ljvdwr, ljeff
  ! Make the C6 coefficients, assuming a geometric mean combination rule
  do atom = 1, natom
      ljptr = ITC(IAC(atom))
 !     ljeff = EFF(ljptr)
 !     ljvdwr = VDWR(ljptr)
 !     ljc6(atom) = 8d0 * ljvdwr * ljvdwr * ljvdwr * SQRT(2d0 * ABS(ljeff))
      ljc6(atom) = sqrt(abs(C6DEF(ljptr)))
  enddo
end subroutine update_ljpme
#endif

SUBROUTINE CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC,NBOND,IB,JB, &
     NTHETA,IT,JT,KT,NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
     QDRUDE,NBDRUDE,                                        & ! DRUDE
#if KEY_CMAP==1
     ICTP,NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT,  & 
#endif
     QFEW,QUIET)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE SETS UP THE CODE ARRAYS (ICB,ICT,ETC.)
  !     THESE ARRAYS POINT TO THE INDEX OF THE PARAMETER ARRAYS FOR
  !     THE INDIVIDUAL INTERNAL COORDINATES
  !
  !     OVERHAULED 4-NOV-1982  BRB
  !
  !  ICB(NBOND),ICT(NTHETA),ICP(NPHI),ICI(NIMPHI)    =>  CODE arrays
  !  NATOM,IMOVE,IAC,NBOND,IB,JB,NTHETA,IT,JT,KT     <=  PSF data
  !  NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM             <=  PSF data
  !  QDRUDE,NBDRUDE                                  <=  Drude data
  !  ICTP(NCRTERM)                                   =>  CMAP CODE array
  !  NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT <=  CMAP data
  !  QFEW   <=  Flag indicating that only a partial list is used
  !             This is used to supress global codes calls (e.g. MMFF)
  !             and to preserve parameter use counts for non-UPDATE usage.
  !  QUIET  <=  Flag indicating that warnings and print are suppressed
  !
  use chm_kinds
  use exfunc
  use dimens_fcm
  use stream
  !
  use param
  use gamess_fcm
#if KEY_SCCDFTB==1
  use blockscc_fcm  
#endif
  !
  ! added for GHO-SCC-DFTB ... PJ 7/2004
#if KEY_SCCDFTB==1
  use stbgho
#endif 
  !
#if KEY_MNDO97==1
#if KEY_MNDO97==1
  !yw050728   use mndgho    
#endif
  use mndgho
#endif 
  !
#if KEY_SQUANTM==1
  use squantm
#endif 
#if KEY_CFF==1 || KEY_MMFF==1
  use ffieldm
  use mmffm
  use cff_fcm
#endif 
#if KEY_CMAP==1
  use cmapm  
#endif
  implicit none
  !
  ! passed
  INTEGER ICB(*),ICT(*),ICP(*),ICI(*)
  INTEGER NATOM,NBOND,NTHETA,NPHI,NIMPHI
  INTEGER IMOVE(*),IAC(*),IB(*),JB(*),IT(*),JT(*),KT(*)
  INTEGER IP(*),JP(*),KP(*),LP(*),IM(*),JM(*),KM(*),LM(*)
  LOGICAL QDRUDE            ! DRUDE
  INTEGER NBDRUDE           ! DRUDE
  INTEGER NTGROM,NTCHAR     ! GROMACS-style potentials
#if KEY_CMAP==1
  INTEGER ICTP(*)
  INTEGER NCRTERM
  INTEGER I1CT(*),J1CT(*),K1CT(*),L1CT(*), &
       I2CT(*),J2CT(*),K2CT(*),L2CT(*)
#endif 
  LOGICAL QFEW,QUIET
  !
  ! local
  INTEGER IOFF(MAXATC+1)
  INTEGER IFLAG,JFLAG,MAXFLAG
  INTEGER NATCSQ,NO,I,J,I1,J1,K1,L1
  integer(chm_int8) :: natc2,numb2,number
  integer :: number4
  INTEGER II,JJ,KK,LL,IIA,JJA,KKA
  INTEGER MULPHI
  CHARACTER(len=4) AI,AJ,AK,AL
  LOGICAL PHIFLG,QWARN
#if KEY_CMAP==1
  INTEGER(chm_int8) :: NUMB21,NUMB22,NUMBER1,NUMBER2,NO1,NO2
  INTEGER I3,J3,K3,L3
#endif 
  !
#if KEY_CFF==1 || KEY_MMFF==1
  INTEGER ICTPT2,I2,J2,K2,L2
  EXTERNAL EXCH5
  integer nmbr
#endif 
  integer itmp
#if KEY_FLEXPARM==1
  LOGICAL WCMATCH
  EXTERNAL WCMATCH
#endif 
  integer nindx8
  external nindx8
  !
  ! begin
  !
  QWARN=WRNLEV >= 2 .AND. .NOT.QUIET
  NTGROM=0
  NTCHAR=0
  !
#if KEY_MMFF==1
  IF(FFIELD == MMFF) THEN
     IF(QFEW) THEN
        ! Note: this use of CODES suboptimal, but has been
        ! done to fix an old bug. Partial CODES calls do not
        ! work for MMFF at present. This could to be fixed... - BRB
        DO I=1,NBOND
           ICB(I)=0
        ENDDO
        DO I=1,NTHETA
           ICT(I)=0
        ENDDO
        DO I=1,NPHI
           ICP(I)=0
        ENDDO
        DO I=1,NIMPHI
           ICI(I)=0
        ENDDO
#if KEY_CMAP==1 /*cmap_few*/
        DO I=1,NCRTERM
           ICTP(I)=0
        ENDDO
#endif /* (cmap_few)*/
     ELSE
        CALL CODES_MM
     ENDIF
     RETURN
  ENDIF
#endif 
  !
  IFLAG=0    ! reset missing paramter counter
  JFLAG=0    ! reset global missing paramter counter
  MAXFLAG=25
  !
#if KEY_CFF==1
  ! In the cff forcefield there are wildcard parameters designated with
  ! atom types of 0.  To allow for atom types of 0 the offset array has
  ! to be stretched out.
  !
  IF (FFIELD == CFF) THEN
     II=1
     DO I=1,NATC
        IOFF(I)=II
        II=II+I+1
     ENDDO
  ELSE
#endif 
     II=0
     DO I=1,NATC
        IOFF(I)=II
        II=II+I
     ENDDO
#if KEY_CFF==1
  ENDIF                  
#endif
  NATC2=NATC*(NATC+1)/2
  NATCSQ=NATC*NATC
  !
  !-----------------------------------------------------------------------
  IF(.NOT.QFEW) THEN
     !       Check to see if all atoms have a valid parameter type
     !       Also count number of atoms for each parameter type.
     DO J=1,MAXATC
        ATCCNT(J)=0
        IF(ATC(J) == ' ') ATCCNT(J)=-1
     ENDDO
     DO J=1,NATOM
        I=IAC(J)
        IF(ATCCNT(I) >= 0) THEN
           ATCCNT(I)=ATCCNT(I)+1
        ELSE IF(ATCCNT(I) < -1) THEN
           ATCCNT(I)=ATCCNT(I)-1
        ELSE
           IF(QWARN) WRITE(OUTU,28) J,I
28         FORMAT(' <CODES>: No atom parameters for atom',I8, &
                ' type',I5)
           IFLAG=IFLAG+1
           ATCCNT(I)=-2
        ENDIF
     ENDDO
     I=0
     DO J=1,MAXATC
        IF(ATCCNT(J) < -1) THEN
           ATCCNT(J)=-1-ATCCNT(J)
           I=I+ATCCNT(J)
        ELSE IF(ATCCNT(J) == -1) THEN
           ATCCNT(J)=0
        ENDIF
     ENDDO
     IF(I > 0 .AND. QWARN) WRITE(OUTU,29) I
29   FORMAT(' <CODES>: Total number of atoms with no paramter:',I8)
     JFLAG=IFLAG
  ENDIF
  !-----------------------------------------------------------------------
  !
  ! Here we handle the codes in the psf; bonds, angles, phis,
  !  and impropers.
  !-----------------------------------------------------------------------
  ! bonds
  !
  ! reset counters
  IF(NBOND > 0 .AND. .NOT.QFEW) THEN
     DO I=1,NCB
        ICBCNT(I)=0
     ENDDO
  ENDIF
  !
  ! BEGIN DRUDE (B. Roux and G. Lamoureux)
  IF(QDRUDE .AND. QWARN)THEN
     IF(NBDRUDE /= NBOND)THEN
        write(OUTU,'(1x,a)') 'UPDATE: There are Drude particles'
        write(OUTU,*) 'CODES:  Will skip bonds ',NBDRUDE  !old drude code (soon to be obsolete...)
     ENDIF
  ENDIF
  ! END DRUDE (B. Roux and G. Lamoureux)
  !
  IFLAG=0
  !
  DO I=1,NBOND
     ICB(I)=0
     !
     IF(IB(I) <= 0 .OR. JB(I).LE.0) GOTO 50
     IF(QDRUDE.AND.(I > NBDRUDE)) GOTO 50 ! DRUDE
     IF(IMOVE(IB(I)) > 0.AND.IMOVE(JB(I)).GT.0) GOTO 50
     IF(IB(I) > NATOM.AND.JB(I).GT.NATOM) GOTO 50
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM.and.qmused) THEN
#if KEY_SCCDFTB==1
        if(qpkac.and.((ib(i) == idxhyd).or.(jb(i).eq.idxhyd))) &
             goto 110
#endif 
        IF(ABS(IGMSEL(IB(I))) == 2) GOTO 50
        IF(ABS(IGMSEL(JB(I))) == 2) GOTO 50
        IF(((ABS(IGMSEL(IB(I))) == 1).OR.(ABS(IGMSEL(IB(I))).EQ.2)) &
             .AND.((ABS(IGMSEL(JB(I))) == 1).OR.(ABS(IGMSEL(JB(I))).EQ.2))) &
             GOTO 50
     ENDIF
#if KEY_SCCDFTB==1
110  continue  
#endif
#endif 
     I1=IAC(IB(I))
     J1=IAC(JB(I))
#if KEY_CFF==1
     ! for cff use the equivalence types
     IF (FFIELD == CFF) THEN
        I1=IBE(I1)
        J1=IBE(J1)
     ENDIF
#endif 
     IF(I1 <= 0.OR.J1.LE.0) GOTO 50
     !
#if KEY_FLEXPARM==1 /*flex_bond*/
     IF(QFLXPARM) THEN
        DO J=NCB,1,-1
           IF(WCMATCH(I1,CBAI(J))) THEN
              IF(WCMATCH(J1,CBAJ(J))) THEN
                 ICB(I)=J
                 GOTO 255
              ENDIF
           ELSEIF(WCMATCH(I1,CBAJ(J))) THEN
              IF(WCMATCH(J1,CBAI(J))) THEN
                 ICB(I)=J
                 GOTO 255
              ENDIF
           ENDIF
        ENDDO
255     CONTINUE
     ELSE
#endif /* (flex_bond)*/
        IF(I1 > J1) THEN
           NUMBER4=IOFF(I1)+J1
        ELSE
           NUMBER4=IOFF(J1)+I1
        ENDIF
        ICB(I)=NINDX(NUMBER4,KCB,NCB)
#if KEY_FLEXPARM==1
     ENDIF                                    
#endif
     !
#if KEY_CFF==1
     IF (FFIELD == CFF) THEN
        ! if explicit bond type not found then first look for bond order
        IF(ICB(I) == 0) THEN
           DO I2=NCB+1,NCBO
              IF (I1 == BID1(I2).AND.J1.EQ.BID2(I2).AND. &
                   BONDTYPE_cff(I) == BORD(I2)) ICB(I)=I2
           ENDDO
           ! if bond type still not found try automatic equivalence types
           IF (ICB(I) == 0) THEN
              I1=IBAE(IAC(IB(I)))
              J1=IBAE(IAC(JB(I)))
              IF(I1 > J1) THEN
                 NUMBER4=IOFF(I1)+J1
              ELSE
                 NUMBER4=IOFF(J1)+I1
              ENDIF
              ICB(I)=NINDX(NUMBER4,KCB,NCB)
           ENDIF
        ENDIF
     ENDIF
#endif 
1999 CONTINUE
     !
     IF(ICB(I) == 0) THEN
        IF(IFLAG < MAXFLAG .AND. QWARN) &
             WRITE(OUTU,31) I,ATC(I1)(1:idleng),ATC(J1)(1:idleng)
31      FORMAT(' <CODES>: No bond parameters for ',I5, &
             ' (',2(1X,A),')')
        IF(IFLAG == MAXFLAG .AND. QWARN) WRITE(OUTU,41)
41      FORMAT(' <CODES>: And more missing bond parameters.')
        IFLAG=IFLAG+1
     ELSE
        IF(.NOT.QFEW) ICBCNT(ICB(I))=ICBCNT(ICB(I))+1
     ENDIF
     !
50   CONTINUE
  ENDDO
  !
  JFLAG=JFLAG+IFLAG
  !-----------------------------------------------------------------------
  ! angles
  IF(NTHETA > 0 .AND. .NOT.QFEW) THEN
#if KEY_CFF==1
     !       the generation of the angle-angle cross terms requires that
     !       the angles be sorted
     CALL SORT(NTHETA,EXCH5,ORDER5,JT,IT,KT,0,0,0,0,3)
#endif 
     ! reset counters
     DO I=1,NCT
        ICTCNT(I)=0
     ENDDO
  ENDIF
  !
  IFLAG=0
  !
  DO I=1,NTHETA
     ICT(I)=0
     IF(IT(I) <= 0 .OR. JT(I).LE.0 .OR. KT(I).LE.0) GOTO 70
     IF(IMOVE(IT(I)) > 0 .AND. IMOVE(JT(I)).GT.0 .AND. &
          IMOVE(KT(I)) > 0)  GOTO 70
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM.and.qmused) THEN
#if KEY_SCCDFTB==1
        if(qpkac.and.((it(i) == idxhyd).or.(jt(i).eq.idxhyd).or. &
             (kt(i) == idxhyd))) goto 111
#endif 
        IF(ABS(IGMSEL(JT(I))) == 2) GOTO 70
        IF((ABS(IGMSEL(IT(I))) == 2) &
             .AND.((IGMSEL(JT(I)) == 0) &
             .OR.(IGMSEL(JT(I)) == 5))) GOTO 70
        IF((ABS(IGMSEL(KT(I))) == 2) &
             .AND.((IGMSEL(JT(I)) == 0) &
             .OR.(IGMSEL(JT(I)) == 5))) GOTO 70
        IF(((ABS(IGMSEL(IT(I))) == 1).OR.(ABS(IGMSEL(IT(I))).EQ.2)) &
             .AND. &
             ((ABS(IGMSEL(JT(I))) == 1).OR.(ABS(IGMSEL(JT(I))).EQ.2)) &
             .AND. &
             ((ABS(IGMSEL(KT(I))) == 1).OR.(ABS(IGMSEL(KT(I))).EQ.2))) &
             GOTO 70
     ENDIF
#if KEY_SCCDFTB==1
111  continue  
#endif
#endif 
     I1=IAC(IT(I))
     J1=IAC(JT(I))
     K1=IAC(KT(I))
#if KEY_CFF==1
     ! for cff use the equivalence atom types
     IF (FFIELD == CFF) THEN
        I1=ITE(I1)
        J1=ITE(J1)
        K1=ITE(K1)
     ENDIF
#endif 
     IF(I1 <= 0.OR.J1.LE.0.OR.K1.LE.0) GOTO 70
     !
#if KEY_FLEXPARM==1 /*flex_angle*/
     IF(QFLXPARM) THEN
        DO J=NCT,1,-1
           IF(WCMATCH(J1,CTAJ(J))) THEN
              IF(WCMATCH(I1,CTAI(J))) THEN
                 IF(WCMATCH(K1,CTAK(J))) THEN
                    ICT(I)=J
                    GOTO 355
                 ENDIF
              ELSEIF(WCMATCH(I1,CTAK(J))) THEN
                 IF(WCMATCH(K1,CTAI(J))) THEN
                    ICT(I)=J
                    GOTO 355
                 ENDIF
              ENDIF
           ENDIF
        ENDDO
355     CONTINUE
     ELSE
#endif /* (flex_angle)*/
        IF(I1 > K1) THEN
           NO=IOFF(I1)+K1
        ELSE
           NO=IOFF(K1)+I1
        ENDIF
        NUMBER4=NATC*NO+J1
        ICT(I)=NINDX(NUMBER4,KCT,NCT)
#if KEY_FLEXPARM==1
     ENDIF                                   
#endif
     !
#if KEY_CFF==1
     ! If a parameter is selected using automatic parameters it is assigned
     ! a -ve pointer.  In codes_cff another search will be made to see if
     ! a bond-ordered angle parameter should be used.
     IF (FFIELD == CFF) THEN
        IF(ICT(I) == 0) THEN
           ! if explicit angle type not found then try automatic equivalence types
           I1=ITEAE(IAC(IT(I)))
           J1=ITAAE(IAC(JT(I)))
           K1=ITEAE(IAC(KT(I)))
           IF(I1 > K1) THEN
              NO=IOFF(I1)+K1
           ELSE
              NO=IOFF(K1)+I1
           ENDIF
           NUMBER4=NATC*NO+J1
           ICT(I)=-NINDX(NUMBER4,KCT,NCT)
           IF(ICT(I) == 0) THEN
              ! if angle type still not found try wild card for atom i or k; if two
              ! wildcard types are found select the one with the higher priority
              NUMBER4=NATC*IOFF(I1)+J1
              ICT(I)=-NINDX(NUMBER4,KCT,NCT)
              IF (I1  /=  K1) THEN
                 NUMBER4 = NATC*IOFF(K1)+J1
                 ICTPT2=-NINDX(NUMBER4,KCT,NCT)
                 IF (ICT(I) /= 0 .AND. ICTPT2 .NE. 0) THEN
                    IF (TID1(-ICTPT2)  >  TID1(-ICT(I))) ICT(I) = ICTPT2
                 ELSEIF (ICT(I) == 0) THEN
                    ICT(I) = ICTPT2
                 ENDIF
              ENDIF
              ! if angle type still not found try wild card for both atoms i and k
              IF (ICT(I) == 0) THEN
                 NUMBER4 = J1
                 ICT(I)=-NINDX(NUMBER4,KCT,NCT)
              ENDIF
           ENDIF
        ENDIF
     ENDIF
#endif 
2999 CONTINUE
     IF(ICT(I) == 0) THEN
        IF(IFLAG < MAXFLAG .AND. QWARN) &
             WRITE(OUTU,32) I,ATC(I1)(1:idleng), &
             ATC(J1)(1:idleng),ATC(K1)(1:idleng)
32      FORMAT(' <CODES>: No angle parameters for ',I5, &
             ' (',3(1X,A),')')
        IF(IFLAG == MAXFLAG .AND. QWARN) WRITE(OUTU,42)
42      FORMAT(' <CODES>: And more missing angle parameters.')
        IFLAG=IFLAG+1
     ELSE
!        IF(.NOT.QFEW) ICTCNT(ICT(I))=ICTCNT(ICT(I))+1
            IF(.NOT.QFEW) THEN
               itmp = abs(ICT(I))
               ICTCNT(itmp) = ICTCNT(itmp) + 1
#if KEY_MMFF==1 || KEY_CFF==1
               IF(FFIELD == CHARMM .or. ffield == amberffn) THEN
#endif 
                  IF (CTB(itmp) >= 0) THEN
                     NTCHAR=NTCHAR+1
                  ELSE
                     NTGROM=NTGROM+1
                  ENDIF
#if KEY_MMFF==1 || KEY_CFF==1
               ENDIF
#endif 
            ENDIF
     ENDIF
     !
70   CONTINUE
  ENDDO
      IF(NTGROM+NTCHAR  >  0) THEN
         IF(NTCHAR == 0) THEN
            QANGTYPE=-1
         ELSE IF(NTGROM == 0) THEN
            QANGTYPE=1
         ELSE
            QANGTYPE=0
            CALL WRNDIE(1,'<CODES>', &
           'Mixing GROMACS and CHARMM angle parameter types.  OK?')
         ENDIF
      ENDIF
!      IF(QANGTYPE /= 1) THEN
!         write(*,*) 'UPDATE> GROMACS-style angles found. NTGROM',ntgrom,
!     &        'NTCHAR',ntchar,'QANGTYPE',qangtype
!      ENDIF


  !
  JFLAG=JFLAG+IFLAG
  !
  !-----------------------------------------------------------------------
  ! dihedrals
  !
  ! reset counters
  IF(NPHI > 0 .AND. .NOT.QFEW) THEN
     DO I=1,NCP
        ICPCNT(I)=0
     ENDDO
  ENDIF
  !
  IFLAG=0
  MULPHI=0
  PHIFLG=.TRUE.
  DO I=1,NPHI
     ICP(I)=0
     IF(IP(I) <= 0.OR.JP(I).LE.0) GOTO 100
     IF(KP(I) <= 0.OR.LP(I).LE.0) GOTO 100
     IF(IMOVE(IP(I)) > 0.AND.IMOVE(JP(I)).GT.0.AND. &
          IMOVE(KP(I)) > 0.AND.IMOVE(LP(I)).GT.0) GOTO 100
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM.and.qmused) THEN
        IF(ABS(IGMSEL(IP(I))) == 2) GOTO 100
        IF(ABS(IGMSEL(JP(I))) == 2) GOTO 100
        IF(ABS(IGMSEL(KP(I))) == 2) GOTO 100
        IF(ABS(IGMSEL(LP(I))) == 2) GOTO 100
        !
#if KEY_MNDO97==0 && KEY_SQUANTM==0
        !
        ! GHO-SCC-DFTB includes torsions around the Q-B bond ... PJ 7/2004
#if KEY_SCCDFTB==1 /*gho1*/
        !
        ! treat QM/MM torsion remove differently for GHO, remove only
        ! Q-Q-Q-Q type, but not X-Q-Q-X type
        !
        IF (QLINK) THEN
           IF (IGMSEL(IP(I)) > 0 .AND. IGMSEL(JP(I)).GT.0 &
                .AND. &
                IGMSEL(KP(I)) > 0 .AND. IGMSEL(LP(I)).GT.0) &
                GOTO 100
        ELSE
#endif /* (gho1)*/
           ! assume only second and third atoms matter.
           IF(((ABS(IGMSEL(JP(I))) == 1).OR.(ABS(IGMSEL(JP(I))).EQ.2)) &
                .AND. &
                ((ABS(IGMSEL(KP(I))) == 1).OR.(ABS(IGMSEL(KP(I))).EQ.2))) &
                GOTO 100
           !
           ! ... PJ 7/2004
#if KEY_SCCDFTB==1 /*gho2*/
        ENDIF
#endif /* (gho2)*/
#else /**/
        ! GHO-MNDO97 and GHO-SQUANTM also includes torsions around the Q-B bond
        ! only remove Q-Q-Q-Q type, since it includes Adjusted connection atoms.
        ! ??? Need to re-consider.. namkh
        ! Assumes no stupid QM-MM boundary setup.
#if KEY_SQUANTM==1
        IF (QLINK(1)) THEN
#else /**/
        IF (QLINK) THEN
#endif 
           IF ((IGMSEL(IP(I)) == 1 .OR. IGMSEL(IP(I)).EQ.2) .AND. &
                (IGMSEL(JP(I)) == 1 .OR. IGMSEL(JP(I)).EQ.2) .AND. &
                (IGMSEL(KP(I)) == 1 .OR. IGMSEL(KP(I)).EQ.2) .AND. &
                (IGMSEL(LP(I)) == 1 .OR. IGMSEL(LP(I)).EQ.2)) &
                GOTO 100
        ELSE
           IF(((ABS(IGMSEL(JP(I))) == 1).OR.(ABS(IGMSEL(JP(I))).EQ.2)) &
                .AND. &
                ((ABS(IGMSEL(KP(I))) == 1).OR.(ABS(IGMSEL(KP(I))).EQ.2))) &
                GOTO 100
        END IF
#endif 
        ENDIF
#endif 
        I1=IAC(IP(I))
        J1=IAC(JP(I))
        K1=IAC(KP(I))
        L1=IAC(LP(I))
        !
#if KEY_CFF==1
        ! for cff use the equivalence atom types
        IF (FFIELD == CFF) THEN
           I1=IPE(I1)
           J1=IPE(J1)
           K1=IPE(K1)
           L1=IPE(L1)
        ENDIF
#endif 
        !
        IF(I1 > 0.AND.J1.GT.0.AND.K1.GT.0.AND.L1.GT.0) THEN
           !
#if KEY_FLEXPARM==1 /*flex_dihe*/
           IF(QFLXPARM) THEN
              !             find the match with largest index
              DO J=NCP,1,-1
                 IF(WCMATCH(J1,CPAJ(J))) THEN
                    IF(WCMATCH(K1,CPAK(J))) THEN
                       IF(WCMATCH(I1,CPAI(J))) THEN
                          IF(WCMATCH(L1,CPAL(J))) THEN
                             ICP(I)=J
                             GOTO 455
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
                 IF(WCMATCH(J1,CPAK(J))) THEN
                    IF(WCMATCH(K1,CPAJ(J))) THEN
                       IF(WCMATCH(I1,CPAL(J))) THEN
                          IF(WCMATCH(L1,CPAI(J))) THEN
                             ICP(I)=J
                             GOTO 455
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDDO
455           CONTINUE
              !             point to first dihedral of a series (multiple dihedral)
              J=ICP(I)-1
              IF(J >= 1) THEN
                 IF(CPAI(J) == CPAI(J+1)) THEN
                    IF(CPAJ(J) == CPAJ(J+1)) THEN
                       IF(CPAK(J) == CPAK(J+1)) THEN
                          IF(CPAL(J) == CPAL(J+1)) THEN
                             ICP(I)=ICP(I)-1
                             GOTO 455
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
           ELSE
#endif /* (flex_dihe)*/
              IF(K1 > J1) THEN
                 NUMBER=IOFF(K1)+J1
              ELSE
                 NUMBER=IOFF(J1)+K1
              ENDIF
              IF(I1 > L1) THEN
                 NO=IOFF(I1)+L1
              ELSE
                 NO=IOFF(L1)+I1
              ENDIF
              NUMB2=NUMBER+NO*NATC2
              IF((J1-K1)*(I1-L1) < 0) NUMB2=NUMB2+NATC2*NATC2
              ICP(I)=NINDX8(NUMB2,KCP,NCP)
              IF(ICP(I) == 0) ICP(I)=NINDX8(NUMBER,KCP,NCP)
#if KEY_FLEXPARM==1
           ENDIF                                          
#endif
           !
#if KEY_CFF==1
           IF (FFIELD == CFF) THEN
              IF(ICP(I) == 0) THEN
                 I1 = IPEAE(IAC(IP(I)))
                 J1 = IPCAE(IAC(JP(I)))
                 K1 = IPCAE(IAC(KP(I)))
                 L1 = IPEAE(IAC(LP(I)))
                 IF (J1 > K1) THEN
                    I1 = IPEAE(IAC(LP(I)))
                    J1 = IPCAE(IAC(KP(I)))
                    K1 = IPCAE(IAC(JP(I)))
                    L1 = IPEAE(IAC(IP(I)))
                 ENDIF
                 IF(K1 > J1) THEN
                    NUMBER=IOFF(K1)+J1
                 ELSE
                    NUMBER=IOFF(J1)+K1
                 ENDIF
                 IF(I1 > L1) THEN
                    NO=IOFF(I1)+L1
                 ELSE
                    NO=IOFF(L1)+I1
                 ENDIF
                 NUMB2=NUMBER+NO*NATC2
                 IF((J1-K1)*(I1-L1) < 0) NUMB2=NUMB2+NATC2*NATC2
                 ICP(I)=-NINDX8(NUMB2,KCP,NCP)
                 IF(ICP(I) == 0) ICP(I)=-NINDX8(NUMBER,KCP,NCP)
              ENDIF
           ENDIF
#endif 
           IF(PHIFLG.AND.ICP(I) < 0.AND.QWARN) WRITE(OUTU,117)
117        FORMAT(' <CODES> NOTE: using multiple potential terms', &
                ' for some dihedrals'/)
           PHIFLG=.FALSE.
        ENDIF
        !
        IF(ICP(I) == 0) THEN
           IF(IFLAG < MAXFLAG .AND. QWARN) &
                WRITE(OUTU,33) I,ATC(I1)(1:idleng),ATC(J1)(1:idleng), &
                ATC(K1)(1:idleng),ATC(L1)(1:idleng)
33         FORMAT(' <CODES>: No torsion parameters for ', &
                I5,' (',4(1X,A),')')
           IF(IFLAG == MAXFLAG .AND. QWARN) WRITE(OUTU,43)
43         FORMAT(' <CODES>: And more missing torsion parameters.')
           IFLAG=IFLAG+1
        ELSE
           IF (.NOT. QFEW) THEN
              itmp = abs(ICP(I))
              ICPCNT(itmp) = ICPCNT(itmp) + 1
           ENDIF
        ENDIF
        !
        IF(I > 1) THEN
           IF(IP(I) == IP(I-1) .AND. JP(I).EQ.JP(I-1) .AND. &
                KP(I) == KP(I-1) .AND. LP(I).EQ.LP(I-1)) THEN
              MULPHI=MULPHI+1
              ICP(I)=0
           ENDIF
        ENDIF

100     CONTINUE

  ENDDO
  !
#if KEY_CFF==1
  IF(FFIELD == CFF .AND. .NOT.QFEW) THEN
     CALL CODES_CFF
  ENDIF
#endif 
  !
  IF (MULPHI > 0 .AND. QWARN) WRITE(OUTU,116) MULPHI
116 FORMAT (' <CODES>: ',I4, &
         ' multiple dihedral entries from RTF ignored')
  !
  JFLAG=JFLAG+IFLAG
  !
  !-----------------------------------------------------------------------
  ! improper dihedrals
  !
  ! reset counters
  IF(NIMPHI > 0 .AND. .NOT.QFEW) THEN
     DO I=1,NCI
        ICICNT(I)=0
     ENDDO
  ENDIF
  !
  IFLAG=0
  !
  DO I=1,NIMPHI
     ICI(I)=0
     IF(IM(I) <= 0.OR.JM(I).LE.0) GOTO 350
     IF(KM(I) <= 0.OR.LM(I).LE.0) GOTO 350
     IF(IMOVE(IM(I)) > 0.AND.IMOVE(JM(I)).GT.0.AND. &
          IMOVE(KM(I)) > 0.AND.IMOVE(LM(I)).GT.0) GOTO 350
     !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM.and.qmused) THEN
        IF(ABS(IGMSEL(IM(I))) == 2) GOTO 350
        IF(ABS(IGMSEL(JM(I))) == 2) GOTO 350
        IF(ABS(IGMSEL(KM(I))) == 2) GOTO 350 
        IF(ABS(IGMSEL(LM(I))) == 2) GOTO 350
#if KEY_MNDO97==0 && KEY_SQUANTM==0
        ! assume only first and last atoms matter.
        IF(((ABS(IGMSEL(IM(I))) == 1).OR.(ABS(IGMSEL(IM(I))).EQ.2)) &
             .AND. &
             ((ABS(IGMSEL(LM(I))) == 1).OR.(ABS(IGMSEL(LM(I))).EQ.2))) &
             GOTO 350
#else /**/
        ! GHO-MNDO97 and GHO-SQUANTM also includes torsions around the Q-B bond
        ! only remove Q-Q-Q-Q type, since it includes Adjusted connection atoms.
        ! ??? Need to re-consider.. namkh
        ! Assumes no stupid QM-MM boundary setup.
#if KEY_SQUANTM==1
        IF(QLINK(1)) THEN
#else /**/
        IF(QLINK) THEN
#endif 
           IF ((IGMSEL(IM(I)) == 1 .OR. IGMSEL(IM(I)).EQ.2) .AND. &
                (IGMSEL(JM(I)) == 1 .OR. IGMSEL(JM(I)).EQ.2) .AND. &
                (IGMSEL(KM(I)) == 1 .OR. IGMSEL(KM(I)).EQ.2) .AND. &
                (IGMSEL(LM(I)) == 1 .OR. IGMSEL(LM(I)).EQ.2)) &
                GOTO 350
        ELSE
           IF(((ABS(IGMSEL(IM(I))) == 1).OR.(ABS(IGMSEL(IM(I))).EQ.2)) &
                .AND. &
                ((ABS(IGMSEL(LM(I))) == 1).OR.(ABS(IGMSEL(LM(I))).EQ.2))) &
                GOTO 350
        END IF
#endif 
        ENDIF
#endif 
        !
        I1=IAC(IM(I))
        L1=IAC(LM(I))
        J1=IAC(JM(I))
        K1=IAC(KM(I))
        !
#if KEY_CFF==1
        IF (FFIELD == CFF) THEN
           I1=IOE(I1)
           J1=IOE(J1)
           K1=IOE(K1)
           L1=IOE(L1)
           IF (I1 > K1) THEN
              NMBR=I1
              I1=K1
              K1=NMBR
           ENDIF
           IF (I1 > L1) THEN
              NMBR=I1
              I1=L1
              L1=NMBR
           ENDIF
           IF (K1 > L1) THEN
              NMBR=K1
              K1=L1
              L1=NMBR
           ENDIF
        ENDIF
#endif 
        !
        IF(I1 > 0.AND.J1.GT.0.AND.K1.GT.0.AND.L1.GT.0) THEN
#if KEY_FLEXPARM==1 /*flex_imphi*/
           IF(QFLXPARM) THEN
              DO J=NCI,1,-1
                 IF(WCMATCH(J1,CIAJ(J))) THEN
                    IF(WCMATCH(K1,CIAK(J))) THEN
                       IF(WCMATCH(I1,CIAI(J))) THEN
                          IF(WCMATCH(L1,CIAL(J))) THEN
                             ICI(I)=J
                             GOTO 555
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
                 IF(WCMATCH(J1,CIAK(J))) THEN
                    IF(WCMATCH(K1,CIAJ(J))) THEN
                       IF(WCMATCH(I1,CIAL(J))) THEN
                          IF(WCMATCH(L1,CIAI(J))) THEN
                             ICI(I)=J
                             GOTO 555
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDDO
555           CONTINUE
           ELSE
#endif /* (flex_imphi)*/
              IF(I1 > L1) THEN
                 NUMBER=IOFF(I1)+L1
              ELSE
                 NUMBER=IOFF(L1)+I1
              ENDIF
              IF(J1 > K1) THEN
                 NO=IOFF(J1)+K1
              ELSE
                 NO=IOFF(K1)+J1
              ENDIF
              NUMB2=NUMBER+NO*NATC2
              IF((J1-K1)*(I1-L1) < 0) NUMB2=NUMB2+NATC2*NATC2
              ICI(I)=NINDX8(NUMB2,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              !
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              !
              ! check for 3 atom matches and amber type 2 atom matches
              !
              ! look for wildcard type X - A - B - C
              NUMBER=-(L1+NATC*(K1-1)+NATCSQ*J1)
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              !
              ! look for wildcard type X - A - B - X
              NUMBER=-(NO+NATC**3+NATCSQ)
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              !
              ! look for type  x - x - a - b
              NUMBER=-(L1+NATC*(K1-1))
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
#if KEY_FLEXPARM==1
           ENDIF                                     
#endif
#if KEY_CFF==1
           IF (FFIELD == CFF) THEN
              ! look for wildcard type x - a - x - x
              NUMBER=-(IOFF(J1)+NATC**3+NATCSQ)
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              I1=IOEAE(I1)
              J1=IOCAE(J1)
              K1=IOEAE(K1)
              L1=IOEAE(L1)
              IF (I1 > K1) THEN
                 NMBR=I1
                 I1=K1
                 K1=NMBR
              ENDIF
              IF (I1 > L1) THEN
                 NMBR=I1
                 I1=L1
                 L1=NMBR
              ENDIF
              IF (K1 > L1) THEN
                 NMBR=K1
                 K1=L1
                 L1=NMBR
              ENDIF
              IF(I1 > L1) THEN
                 NUMBER=IOFF(I1)+L1
              ELSE
                 NUMBER=IOFF(L1)+I1
              ENDIF
              IF(J1 > K1) THEN
                 NO=IOFF(J1)+K1
              ELSE
                 NO=IOFF(K1)+J1
              ENDIF
              NUMB2=NUMBER+NO*NATC2
              IF((J1-K1)*(I1-L1) < 0) NUMB2=NUMB2+NATC2*NATC2
              ICI(I)=NINDX8(NUMB2,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              !
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
              NUMBER=-(IOFF(J1)+NATC**3+NATCSQ)
              ICI(I)=NINDX8(NUMBER,KCI,NCI)
              IF(ICI(I) > 0) GOTO 300
           ENDIF
#endif 
           !
300        CONTINUE
           ! nothing found, error
           IF(ICI(I) == 0) THEN
              IF(IFLAG < MAXFLAG .AND. QWARN) &
                   WRITE(OUTU,34) I,ATC(I1)(1:idleng),ATC(J1)(1:idleng), &
                   ATC(K1)(1:idleng),ATC(L1)(1:idleng)
34            FORMAT(' <CODES>: No improper parameters for ',I5, &
                   ' (',4(1X,A),')')
              IF(IFLAG == MAXFLAG .AND. QWARN) WRITE(OUTU,44)
44            FORMAT(' <CODES>: And more missing improper parameters.')
              IFLAG=IFLAG+1
           ELSE
              IF(.NOT.QFEW) ICICNT(ICI(I))=ICICNT(ICI(I))+1
           ENDIF
        ENDIF
        !
350     CONTINUE
  ENDDO
  !
  JFLAG=JFLAG+IFLAG
  !
  !-----------------------------------------------------------------------
  ! cmap
#if KEY_CMAP==1
  ! reset counters
  IF(NCRTERM > 0 .AND. .NOT.QFEW) THEN
     DO I=1,NCTP
        ICTPCNT(I)=0
     ENDDO
  ENDIF
  !
  IFLAG=0
  !
  DO I=1,NCRTERM
     ICTP(I)=0

     IF(I1CT(I) <= 0.OR.J1CT(I).LE.0) GOTO 351
     IF(K1CT(I) <= 0.OR.L1CT(I).LE.0) GOTO 351
     IF(I2CT(I) <= 0.OR.J2CT(I).LE.0) GOTO 351
     IF(K2CT(I) <= 0.OR.L2CT(I).LE.0) GOTO 351

     IF(IMOVE(I1CT(I)) > 0.AND.IMOVE(J1CT(I)).GT.0.AND. &
          IMOVE(K1CT(I)) > 0.AND.IMOVE(L1CT(I)).GT.0.AND. &
          IMOVE(I2CT(I)) > 0.AND.IMOVE(J2CT(I)).GT.0.AND. &
          IMOVE(K2CT(I)) > 0.AND.IMOVE(L2CT(I)).GT.0) GOTO 351

#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
     IF(QGMREM.and.qmused) THEN
        IF(IGMSEL(I1CT(I)) == 2) GOTO 351
        IF(IGMSEL(J1CT(I)) == 2) GOTO 351
        IF(IGMSEL(K1CT(I)) == 2) GOTO 351 
        IF(IGMSEL(L1CT(I)) == 2) GOTO 351
        IF(IGMSEL(I2CT(I)) == 2) GOTO 351
        IF(IGMSEL(J2CT(I)) == 2) GOTO 351
        IF(IGMSEL(K2CT(I)) == 2) GOTO 351 
        IF(IGMSEL(L2CT(I)) == 2) GOTO 351
        ! assume only first and last atoms matter. BUG: Not good for excluded MM(5)
        IF(IGMSEL(I1CT(I)) > 0.AND. &
             IGMSEL(L1CT(I)) > 0.AND. &
             IGMSEL(I2CT(I)) > 0.AND. &
             IGMSEL(L2CT(I)) > 0) GOTO 351
     ENDIF
#endif 
     I1=IAC(I1CT(I))
     L1=IAC(L1CT(I))
     J1=IAC(J1CT(I))
     K1=IAC(K1CT(I))
     I3=IAC(I2CT(I))
     L3=IAC(L2CT(I))
     J3=IAC(J2CT(I))
     K3=IAC(K2CT(I))
     IF(I1 > 0.AND.J1.GT.0.AND.K1.GT.0.AND.L1.GT.0.AND. &
          I3 > 0.AND.J3.GT.0.AND.K3.GT.0.AND.L3.GT.0) THEN
        !
#if KEY_FLEXPARM==1 /*flex_dihe*/
        IF(QFLXPARM) THEN
           DO J=NCTP,1,-1
              IF(WCMATCH(J1,CTPA(J,1))) THEN
                 IF(WCMATCH(K1,CTPA(J,2))) THEN
                    IF(WCMATCH(I1,CTPA(J,3))) THEN
                       IF(WCMATCH(L1,CTPA(J,4))) THEN
                          IF(WCMATCH(J3,CTPA(J,5))) THEN
                             IF(WCMATCH(K3,CTPA(J,6))) THEN
                                IF(WCMATCH(I3,CTPA(J,7))) THEN
                                   IF(WCMATCH(L3,CTPA(J,8))) THEN
                                      ICTP(I)=J
                                      GOTO 655
                                   ENDIF
                                ENDIF
                             ENDIF
                          ENDIF
                       ENDIF
                    ENDIF
                 ENDIF
              ENDIF
           ENDDO
655        CONTINUE
        ELSE
#endif /* (flex_dihe)*/
           IF(J1 > K1) THEN
              NUMBER1=IOFF(J1)+K1
           ELSE
              NUMBER1=IOFF(K1)+J1
           ENDIF
           IF(I1 > L1) THEN
              NO1=IOFF(I1)+L1
           ELSE
              NO1=IOFF(L1)+I1
           ENDIF
           NUMB21=NUMBER1+NO1*NATC2
           IF((J1-K1)*(I1-L1) < 0) &
                NUMB21=NUMB21+NATC2*NATC2

           IF(J3 > K3) THEN
              NUMBER2=IOFF(J3)+K3
           ELSE
              NUMBER2=IOFF(K3)+J3
           ENDIF
           IF(I3 > L3) THEN
              NO2=IOFF(I3)+L3
           ELSE
              NO2=IOFF(L3)+I3
           ENDIF
           NUMB22=NUMBER2+NO2*NATC2
           IF((J3-K3)*(I3-L3) < 0) &
                NUMB22=NUMB22+NATC2*NATC2

           ICTP(I)=NINDX8(NUMB21+NUMB22,KCTP,NCTP)
           IF(ICTP(I) > 0) GOTO 301

           ICTP(I)=NINDX8(NUMBER1+NUMB22,KCTP,NCTP)
           IF(ICTP(I) > 0) GOTO 301

           ICTP(I)=NINDX8(NUMB21+NUMBER2,KCTP,NCTP)
           IF(ICTP(I) > 0) GOTO 301

           ICTP(I)=NINDX8(NUMBER1+NUMBER2,KCTP,NCTP)
           IF(ICTP(I) > 0) GOTO 301
#if KEY_FLEXPARM==1
        ENDIF                                      
#endif
        ! nothing found, error
301     CONTINUE
        IF(ICTP(I) <= 0) THEN
           IF(IFLAG < MAXFLAG .AND. QWARN) &
                WRITE(OUTU,94) I,ATC(I1)(1:idleng),ATC(J1)(1:idleng), &
                ATC(K1)(1:idleng),ATC(L1)(1:idleng), &
                ATC(I3)(1:idleng),ATC(J3)(1:idleng), &
                ATC(K3)(1:idleng),ATC(L3)(1:idleng)
94         FORMAT(' <CODES>: No cross-term map for ',I5, &
                ' (',4(1X,A),')', &
                ' (',4(1X,A),')')
           IF(IFLAG == MAXFLAG .AND. QWARN) WRITE(OUTU,95)
95         FORMAT(' <CODES>: And more missing cross-term parameters')
           IFLAG=IFLAG+1
        ELSE
           IF(.NOT.QFEW) ICTPCNT(ICTP(I))=ICTPCNT(ICTP(I))+1
        ENDIF
     ENDIF
351  CONTINUE
  ENDDO
#endif /*  CMAP*/
  !
  JFLAG=JFLAG+IFLAG
  !
  !-----------------------------------------------------------------------
  ! error messages
  !
  IF(JFLAG == 0) RETURN
  IF(QWARN) THEN
     WRITE(OUTU,83) JFLAG
83   FORMAT(' <CODES>: A TOTAL OF',I5,' MISSING PARAMETERS')
     CALL WRNDIE(-1,'<CODES>','CODES> MISSING PARAMETERS')
  ENDIF
  !
  RETURN
END SUBROUTINE CODES

SUBROUTINE HCODES(ICH,NHB,IHB,JHB,IAC)
  !
  !     THIS ROUTINE SETS UP THE CODE ARRAYS (ICH)
  !
  !      By Bernard R. Brooks    NOV-82
  !
  use chm_kinds
  use exfunc
  use stream
  use dimens_fcm
  !
  use param
  use gamess_fcm
  implicit none
  !
  INTEGER NHB
  INTEGER ICH(*),IAC(*)
  INTEGER IHB(*),JHB(*)
  !
  INTEGER I,J,I1,J1,NUMBER
#if KEY_FLEXPARM==1
  LOGICAL WCMATCH
  EXTERNAL WCMATCH
#endif 
  !
  ! here we find the codes for the hydrogen bonds
  !
  DO I=1,NHB
     ICH(I)=0
     IF(IHB(I) > 0.AND.JHB(I).GT.0) THEN
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QUANTUM==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
        IF(QGMREM.and.qmused) THEN
           IF(ABS(IGMSEL(IHB(I))) == 2) GOTO 100
           IF(ABS(IGMSEL(JHB(I))) == 2) GOTO 100
           IF(((ABS(IGMSEL(IHB(I))) == 1) &
                .OR.(ABS(IGMSEL(IHB(I))) == 2)) &
                .AND. &
                ((ABS(IGMSEL(JHB(I))) == 1) &
                .OR.(ABS(IGMSEL(JHB(I))) == 2))) GOTO 100
        ENDIF
#endif 
        I1=IAC(IHB(I))
        J1=IAC(JHB(I))
        IF(I1 > 0.AND.J1.GT.0) THEN
           !
#if KEY_FLEXPARM==1 /*flex_hbond*/
           IF(QFLXPARM) THEN
              DO J=NCH,1,-1
                 IF(WCMATCH(I1,CHAD(J))) THEN
                    IF(WCMATCH(J1,CHAA(J))) THEN
                       ICH(I)=J
                       GOTO 255
                    ENDIF
                 ENDIF
              ENDDO
255           CONTINUE
           ELSE
#endif /* (flex_hbond)*/
              NUMBER=NATC*(I1-1)+J1
              ICH(I)=NINDX(NUMBER,KCH,NCH)
#if KEY_FLEXPARM==1
           ENDIF                                   
#endif
           !
           IF(ICH(I) == 0) THEN
              IF(WRNLEV >= 2) WRITE(OUTU,35) &
                   I,ATC(I1)(1:idleng),ATC(J1)(1:idleng)
35            FORMAT(' <HCODES>: No hydrogen bond parameters for ', &
                   I5,' (',2(1X,A),')')
              CALL WRNDIE(-1,'<HCODES>', &
                   'HCODES> No hydrogen bond parameters')
           ENDIF
        ELSE
           IF(WRNLEV >= 2) WRITE(OUTU,36) &
                I,ATC(I1)(1:idleng),ATC(J1)(1:idleng)
36         FORMAT(' <HCODES>: Bad atom type(s) for hydrogen bond:', &
                I6,' (atoms:',2A,' )')
           CALL WRNDIE(-1,'<HCODES>', &
                'HCODES> Bad hydrogen bond')
        ENDIF
     ENDIF
100  CONTINUE
  ENDDO
  !
  RETURN
END SUBROUTINE HCODES

#if KEY_FLEXPARM==1 /*wcmatch_fp*/
LOGICAL FUNCTION WCMATCH(AI,BI)
  !
  ! Does atom type AI match atom type or group type BI?  - BRB
  !
  use chm_kinds
  use dimens_fcm
  use param
  !
  use stream
  implicit none
  !
  INTEGER AI,BI
  !
  IF(BI > 0) THEN
     !       check for exact match
     IF(AI == BI) THEN
        WCMATCH=.TRUE.
     ELSE
        WCMATCH=.FALSE.
     ENDIF
  ELSE IF(BI < 0) THEN
     !       check for wildcard match
     IF(ACTEQVM(AI,-BI) > 0) THEN
        WCMATCH=.TRUE.
     ELSE
        WCMATCH=.FALSE.
     ENDIF
  ELSE
     WCMATCH=.FALSE.
  ENDIF
  !
  !      if(bi < 0) then
  !        iz=ACTEQVM(AI,-BI)
  !        ac=acteqv(-bi)
  !      else
  !        iz=2
  !        ac=atc(bi)
  !      endif
  !      write(outu,45) ai,bi,iz,atc(ai),atc(i),ac,wcmatch
  ! 45   format(' WCMATCH:',3I6,3X,3A6,2X,L1)
  !
  RETURN
END FUNCTION WCMATCH
#endif /* (wcmatch_fp)*/

SUBROUTINE RLIST(COMLYN,COMLEN,N,LIST,NDIM)
  !-----------------------------------------------------------------------
  !     Reads names from COMLYN and stores them in LIST
  !-----------------------------------------------------------------------
  use chm_kinds
  use exfunc
  use stream
  use string

  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN,N,NDIM
  CHARACTER(len=*) LIST(NDIM)
  CHARACTER(len=8) NAME
  !
100 CONTINUE
  NAME=NEXTA8(COMLYN,COMLEN)
  IF(NAME == ' ') RETURN
  IF(N >= NDIM) THEN
     IF(WRNLEV >= 2) WRITE(OUTU,'(A,2I5)') &
          'RLIST> TOO MANY NAMES: N,NDIM=',N,NDIM
     RETURN
  ENDIF
  N=N+1
  LIST(N)=NAME
  GOTO 100
#if KEY_PARALLEL==1 /*paraupdate*/
END SUBROUTINE RLIST

SUBROUTINE PARUPDATE(JNB,INBLO)
  !
  ! Update arrays needed for parallel code.
  !
  !           Bernard R. Brooks - November 9, 1993
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use code
  use contrl
  use energym
  use fast
  use hbondm
  use image
  use param
  use psf
  use shake
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
  use parallel
#if KEY_DOMDEC==1
  use domdec_common,only:q_domdec  
#endif
  implicit none
  !
  INTEGER JNB(*),INBLO(*)
  INTEGER I,IGRP,IQ,NTRY
  !
#if KEY_DOMDEC==1
  if (q_domdec) return  
#endif
  !
#if KEY_PARAFULL==1
  ! decide where atom boundary between processors lie
  IPARPT(0)=0
  DO I=1,NUMNOD
     NTRY=(NATOM*I)/NUMNOD
     DO IGRP=1,NGRP
        IQ=IGPBS(IGRP+1)
        IF(IQ > NTRY) GOTO 250
     ENDDO
     IGRP=NGRP+1
250  IPARPT(I)=IGPBS(IGRP)
  ENDDO
#endif 

  ! We need the blockcount as well as the displacement
  DO I=1,NUMNOD
     NPARPT(I-1)=IPARPT(I)-IPARPT(I-1)
  ENDDO
  !
  RETURN
#if KEY_FSSHK==1 /*fsshk*/
END SUBROUTINE PARUPDATE

SUBROUTINE PARPTUPDATE()
  !
  ! Update parpt only used by shake setup
  !           M Crowley feb 2000, copied from above
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use code
  use contrl
  use energym
  use fast
  use hbondm
  use image
  use param
  use psf
  use shake
  use stream
#if KEY_TSM==1
  use tsms_mod
#endif 
  use parallel
!!$#if KEY_DOMDEC==1
!!$  use domdec_common,only:q_domdec  
!!$#endif
  implicit none
  INTEGER I,IGRP,IQ,NTRY
  !
!!$#if KEY_DOMDEC==1
!!$  if (q_domdec) return 
!!$#endif
  !
#if KEY_PARAFULL==1
  ! decide where atom boundary between processors lie
  IPARPT(0)=0
  DO I=1,NUMNOD
     NTRY=(NATOM*I)/NUMNOD
     DO IGRP=1,NGRP
        IQ=IGPBS(IGRP+1)
        IF(IQ > NTRY) GOTO 250
     ENDDO
     IGRP=NGRP+1
250  IPARPT(I)=IGPBS(IGRP)
  ENDDO
#endif 
  RETURN
#endif /* (fsshk)*/
#endif /* (paraupdate)*/
END SUBROUTINE

