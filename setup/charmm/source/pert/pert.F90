module sftc
  use chm_kinds
  use dimens_fcm
#if KEY_BLOCK==1
  !     The Perturbed Protein Structure File (PSF)
  LOGICAL QBPSSP,TBQPSSP
  real(chm_real) BALAMBD, BDLAMBD,BLAPSSP
  real(chm_real) bdvdl(6)
#endif 
end module sftc

module pert_mod

contains

#if KEY_PERT==1 /*pert1*/
SUBROUTINE PERTS(COMLYN,COMLEN)
  !
  ! THIS ROUTINE SAVES THE CURRENT PSF FOR SUBSEQUENT
  ! MODIFICATION.
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  !     Stefan Boresch, April 1995: Communication between
  !     SHAKE and PERT free energy module added; constraint
  !     correction added.  Changes in pert.src are minor;
  !     majority of changes in shake.src, dynamc(v).src and
  !     epert.src.
  !
  !     Stefan Boresch, Martin Leitgeb, May 2003: MMFP restraints
  !     are available for alchemical mutations (optional)
  !
  use chm_kinds
  use chm_types
  use memory
  use dimens_fcm
  use select
  use number
  !
  use chutil,only:makind
  use bases_fcm
  use psf
  use pert
  use param

  use dcm_fcm,only:dcchgcopy,qdcm 

  use energym
  use epert_mod
  use exelecm  ! NKB, exelec with pert
  use code
  use coord
  use contrl
  use inbnd
  use image
  use lookup,only:qlookup,qvu,quu,iwwflg,iwwenr
  use stream
  use string
  use shake
  use datstr,only:dupldt_nbond,freedt_nbond,freedt_image
#if KEY_MMFF==1
  use ffieldm, only: ffield, mmff
#endif
  ! QC: 11/17 for SCC PERT based on Xiya
#if KEY_SCCDFTB==1
  use blockscc_fcm
  use gamess_fcm, only: IGMSEL  !Xiya
  use sccdftb
  use sccdftbsrc
#endif
  ! QC: 11/17 Done
  !---   use nbutil_module,only:gtnbct,renbnd
  implicit none
  !
  INTEGER COMLEN
  character(len=*) COMLYN
  !
  INTEGER I,J,N !QC: 11/17 Add N
  !
  IF(INDXA(COMLYN,COMLEN,'OFF') > 0) THEN
     call pert_off
     return
  ENDIF
  ! Call DCM chg backup 1st to restore original CG array

  IF(QDCM) THEN
    CALL DCCHGCOPY(CG)
  ENDIF

#if KEY_MMFF==1
  if (ffield .eq. mmff) call wrndie(-2, '<PERTS>', 'PERT is incompatible with MMFF')
#endif

  !
  ! Allocate space for pert arrays
  !
  CALL INIPERT
  !
  !
  ! process atom selection for atoms to be modified.
  CALL SELCTA(COMLYN,COMLEN,PERTIP,X,Y,Z,WMAIN,.TRUE.)
  ! QC: 11/17 Add DFTB PERT based on Xiya; count before NIPERT
#if KEY_SCCDFTB==1
  QMMQM=(INDXA(COMLYN,COMLEN,'QMMM').GT.0)
  IF(QMMQM) THEN
    WRITE(OUTU,*)"PERT> Perform perturbation from QM (lambda=0) to MM (lambda=1)"
    FIRSTE = .TRUE.
    IF(NSCCTC >0) WRITE(OUTU,*)"WARNING: PERT QMMM have to be called before SCCDFTB"
  ENDIF
  IF(QSCCPERT) THEN
    WRITE(OUTU,*)"PERT> Perform perturbation from QM1+QM2 (lambda=0) to QM1+MM2 (lambda=1)"
  ENDIF
  IF (allocated(igmsel) .and. .not. qsccpert) THEN
     N=0
     DO I=1, NATOM
        IF(PERTIP(I) .EQ. 1) THEN
          IF((IGMSEL(I) .EQ. 1) .OR. (IGMSEL(I) .EQ. 2)) THEN
           N=N+1
         ENDIF
        ENDIF
     ENDDO
     IF(N .NE. NSCCTC) THEN
       WRITE(OUTU,*)"WARNING: Partial QM atoms are included in the PERT."
     ENDIF
  ENDIF
#endif
  ! QC: 11/17 Done
  NIPERT=NSELCT(NATOM,PERTIP)
  ! Store Selected Atoms for SSBP LRC - Y. Deng
  if(allocated(ppiatom)) &
       call chmdealloc('pert.src','PERTS:79','PPIATOM',size(ppiatom),intg=PPIATOM)
  call chmalloc('pert.src','PERTS','PPIATOM',NIPERT,intg=PPIATOM)
  CALL MAKIND(NATOM,PERTIP,PPIATOM,NIPERT) !QC: Appears O.K.
  !
  !  INVOKE THE FAST OPTIONS FOR OTHER TERMS
  !
  IF(INDXA(COMLYN,COMLEN,'RESE') > 0) THEN
     EPRTOT=0.0
     EPRTTW=0.0
     EPPRTT(:) = ZERO
     ETPRTT(:) = ZERO
     EVPRTT(:) = ZERO
     QPSSP=.FALSE.
     DLAMBD=FIVE
     ALAMBD=FIVE
     DCOTOT=ZERO
     DLJTOT=ZERO
     IF(PRNLEV >= 2) WRITE(OUTU,115)
115  FORMAT(' PERT: Free energy calculations reset.')
  ENDIF
  ! Check that we are not conflicting with LOOKUP
  IF(QLOOKUP)THEN
     IF(QUU .OR. QVU) CALL WRNDIE(-4, &
        '<PERTS>','LOOKUP with PERT MUST use NOUU and NOVU')
     IF(ANY(IWWFLG(1:NATOM) == 1 .AND. (IWWFLG(1:NATOM)==PERTIP(1:NATOM))))THEN
       CALL WRNDIE(-4, &
         '<PERTS>','LOOKUP and PERT selections MUST be disjoint')
     ENDIF
     IF(IWWENR /= 2) THEN
       CALL WRNDIE(1, &
        '<PERTS>','LOOKUP with PERT requires "ENERGY" flag; this has been set')
       IWWENR=2
     ENDIF
  ENDIF
  !
  !sb   To be on the safe side, I trap the case where a SHAKE
  !     command was issued before entering the PERT bracket.  While
  !     this may often be OK, it can cause havoc if a constraint
  !     correction is actually needed.
  ! Changed to non-fatal since this is normal with multiple PERT calls - BRB
  IF (QSHAKE) THEN
     CALL WRNDIE(1,'<PERTS>', &
          'Issue SHAKe command after PERT command!')
  ENDIF
  !
  !lni, similarly - check if images are present
  ! Changed to non-fatal since this is normal with multiple PERT calls - BRB
  IF(NTRANS  >  0) THEN
     CALL WRNDIE(1,'<PERTS>', &
          'Issue IMAGE/CRYSTL command after PERT command!')
  ENDIF

  QPERT=.TRUE.
  IF(PRNLEV >= 2) WRITE(OUTU,134)
134 FORMAT(' PERT: The current PSF is saved as lambda=0 for', &
       ' subsequent calculations.')
  !
  IF(PRNLEV >= 2) WRITE(OUTU,144) NIPERT
144 FORMAT(' PERT: Number of atoms treated as changing:',I6)

#if KEY_WCA==1
  ! shift softcore
  IF (INDXA(COMLYN, COMLEN, 'SCL0')  /=  0) LSOFTCORE0 = .TRUE.
  IF (INDXA(COMLYN, COMLEN, 'SCL1')  /=  0) LSOFTCORE1 = .TRUE.
  IF (LSOFTCORE0) THEN
     SCCUTR0 = GTRMF(COMLYN, COMLEN, 'SCR0', SCCUTR0)
     IF (PRNLEV  >=  2) THEN
        WRITE(OUTU, 2122)
        WRITE(OUTU, 2121) SCCUTR0
     ENDIF
  ENDIF

  IF (LSOFTCORE1) THEN
     SCCUTR1 = GTRMF(COMLYN, COMLEN, 'SCR1', SCCUTR1)
     IF (PRNLEV  >=  2) THEN
        WRITE(OUTU, 2125)
        WRITE(OUTU, 2124) SCCUTR1
     ENDIF
  ENDIF

2121 FORMAT(' PERTPS> SCCUTR0 = ',F10.5)
2122 FORMAT(' PERT: USING SOFT CORE FOR LAMBDA = 0')
2124 FORMAT(' PERTPS> SCCUTR1 = ',F10.5)
2125 FORMAT(' PERT: USING SOFT CORE FOR LAMBDA = 1')
#endif 

  !ML-----------------------------------------------------------------
  ! Check for MMFP during alchemical mutation
  IF(INDXA(COMLYN,COMLEN,'MMFP') > 0) THEN
     QMMFPE=.TRUE.
     IF(PRNLEV >= 2) WRITE(OUTU,154)
154  FORMAT(' PERT: MMFP restraints are enabled in alchemical ', &
          'mutations!')
  ELSE
     QMMFPE=.FALSE.
  ENDIF
  !ML-----------------------------------------------------------------
#if KEY_CHEMPERT==1
  !sbcp
  if(indxa(comlyn,comlen,'CHEM') > 0) then
     qchemp=.true.
     if(prnlev >= 2) write(outu,155)
155  format(' PERT: Experimental CHEM PERT code enabled')
  endif
  !sb
#endif 
  !
  ! Get the nonbond options (if specified,
  INBFRQ=GTRMI(COMLYN,COMLEN,'INBF',INBFRQ)
  IF(INBFRQ /= 0) CALL GTNBCT(COMLYN,COMLEN,BNBND)
  !
! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
  ! Build bonded list for lambda=0
  IF (QSCCPERT) THEN 
     igmsel(1:natom) = igmsel1(1:natom)
  ENDIF
#endif
! QC: 11/17 Done
  ! Fill the code arrays
  CALL CODES(ICB,ICT,ICP,ICI,NATOM,IMOVE,IAC,NBOND,IB,JB, &
       NTHETA,IT,JT,KT,NPHI,IP,JP,KP,LP,NIMPHI,IM,JM,KM,LM, &
       QDRUDE,NBDRUDE,                                 & ! DRUDE
#if KEY_CMAP==1
       ICCT,NCRTERM,I1CT,J1CT,K1CT,L1CT,I2CT,J2CT,K2CT,L2CT, &    
#endif
       .FALSE.,.FALSE.)
! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
  ! Restore igmsel for lambda=1
  IF (QSCCPERT) THEN 
     igmsel(1:natom) = igmsel2(1:natom)
  ENDIF
#endif
! QC: 11/17 Done
  !
  ! save current PSF, restraints, and codes.
  CALL PERTS2(PERTIP,PERTIG)
  !
  ! remove current nonbond list
  CALL RENBND(BNBND,0,NATOM,NGRP,0)
  ! duplicate nonbond list structure.
  CALL FREEDT_nbond(BNBNDP)
  CALL FREEDT_nbond(BNBNDR)
  CALL DUPLDT_nbond(BNBNDP,BNBND)
  CALL DUPLDT_nbond(BNBNDR,BNBND)
  !
  ! set default pert values if pert is not used
  IPSTRT=0
  IPSTP=0
  IPNTOT=0
  PUNIT=-1
  IPINCR=0
  IPEQUI=0
  QPWIND=.TRUE.
  QPAVER=.FALSE.
  !
  LAMDAP=ZERO
  LAMSTA=ZERO
  LAMSTOP=ZERO
  LAMDAI=ZERO
  LAMDAF=ZERO
  LAMDA=ZERO
  LAMDAM=ONE
  LAMDEL=LAMDAF-LAMDAI
  DELDRT=ZERO
  LAMINC=ZERO
  ! DBEXP parameters
  LAORDER=3
  LAMEXP=ZERO
  NLAMEXP=0
  ILAMEXP=0
  !
  EPDIFF=ZERO
  EPTOT1F=ZERO
  EPTOT1B=ZERO
  EPTOT2F=ZERO
  EPTOT2B=ZERO
  EPTOT3=ZERO
  EPTOT4=ZERO
  EPREF =ZERO
  EPREFF=ZERO
  EPREFB=ZERO
  EPSSBPLRC=ZERO
  EPSSBPLRCSQ=ZERO
  !
  EPPRTM(:) = ZERO
  EPPRTL(:) = ZERO
  EPPRTD(:) = ZERO
  EPPRTA(:) = ZERO
  EPPRTF(:) = ZERO
  EPPRTT(:) = ZERO
  QEPPRT(:) = QEPROP(:)

  ETPRTM(:) = ZERO
  ETPRTL(:) = ZERO
  ETPRTD(:) = ZERO
  ETPRTA(:) = ZERO
  ETPRTF(:) = ZERO
  ETPRTT(:) = ZERO
  QETPRT(:) = QETERM(:)

  EVPRTM(:) = ZERO
  EVPRTL(:) = ZERO
  EVPRTD(:) = ZERO
  EVPRTA(:) = ZERO
  EVPRTF(:) = ZERO
  EVPRTT(:) = ZERO
  QEVPRT(:) = QEPRSS(:)
  !
  IPNCAL=0
  EPSSLJ=ZERO
  EPSSCO=ZERO
  ECODLM=ZERO
  ECODLL=ZERO
  ELJDLM=ZERO
  ELJDLL=ZERO
! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
  if(nscctc > 0 .and. .not. qsccpert) then 
    if (lldep) then 
      do i=1,3*nscctc
        qla(i)=ql(i)
        qlb(i)=ql(i)
      enddo
    else 
      do i=1,nscctc
        qmata(i)=qmat(i)
        qmatb(i)=qmat(i)
      enddo
    endif
    if (lcolspin) then 
      do i=1,3*nscctc
        qlupa(i)=qlup(i)
        qlupb(i)=qlup(i)
        qldowna(i)=qldown(i)
        qldownb(i)=qldown(i)
      enddo
    endif
  endif
#endif
! QC: 11/17 END
  RETURN
END SUBROUTINE PERTS

!! REDUNDANT, ROUTINE MAKIND does the same thing.
!!SUBROUTINE FILLPPIATOM(IPERT,IATOM)
!!  !
!!  ! THIS ROUTINE FILLS IATOM ARRAY WITH SELECTED ATOMS
!! !
!!  !  Y. DENG APRIL 2004
!!  !
!!  use chm_kinds
!!  use dimens_fcm
!!  use exfunc
!!  use number
!!  use bases_fcm
!!  use psf
!!  use pert
!!
!!  implicit none
!!  !
!!  INTEGER I, J, IPERT(*), IATOM(*)
!!  character(len=8) SIDI,RIDI,RENI,ACI
!!
!! !
!!  J=1
!!  DO I=1, NATOM
!!     IF (IPERT(I)  ==  1) THEN
!!        IATOM(J)=I
!!        J=J+1
!!     ENDIF
!!  ENDDO
!!  RETURN
!!  END SUBROUTINE FILLPPIATOM

SUBROUTINE INIPERT
  !
  !  This routine allocates the space for the PERT datastructures
  !
  !        By B.R. Brooks - NIH - March 11, 1998
  !
  use memory
  use dimens_fcm
  use pert
  use psf
  use cnst_fcm
  use noem
  implicit none
  !
  ! Free space if it is alreay allocated (start over..)
  !
  ! Allocage general pert array space
  !   note: this is probably a lot more space than is needed... fix later..
  !
  !!      allocate(ptrsto(maxsto))

  if(allocated(pertip))return    !mfc assume everything already allocated

  call chmalloc('pert.src','INIPERT','PERTIP',MAXAIM,intg=PERTIP)
  call chmalloc('pert.src','INIPERT','PERTIG',MAXGRP,intg=PERTIG)
  call chmalloc('pert.src','INIPERT','PERTDX',MAXAIM,crl=PERTDX)
  call chmalloc('pert.src','INIPERT','PERTDY',MAXAIM,crl=PERTDY)
  call chmalloc('pert.src','INIPERT','PERTDZ',MAXAIM,crl=PERTDZ)
  call chmalloc('pert.src','INIPERT','PERTEP1',MAXAIM,crl=PERTEP1)
  call chmalloc('pert.src','INIPERT','PERTEP2',MAXAIM,crl=PERTEP2)
  call chmalloc('pert.src','INIPERT','PERTEPC',MAXAIM,crl=PERTEPC)
  call chmalloc('pert.src','INIPERT','PERTCONST',MAXSHK,intg=PERTCONST)
  call chmalloc('pert.src','INIPERT','PERTSHAUX',(2*MAXSHK),crl=PERTSHAUX)
  PERTIP = 0
  PERTIG = 0
  !
  ! Allocate pert PSF space
  !

  call chmalloc('pert.src','INIPERT','PPNICTT',(NSEGT+1),intg=PPNICTT)
  call chmalloc('pert.src','INIPERT','PPIBASE',MAXRES,intg=PPIBASE)
  call chmalloc('pert.src','INIPERT','PPIGPBS',MAXGRP,intg=PPIGPBS)
  call chmalloc('pert.src','INIPERT','PPIGPTP',MAXGRP,intg=PPIGPTP)
  call chmalloc('pert.src','INIPERT','PPIAC',MAXAIM,intg=PPIAC)
  call chmalloc('pert.src','INIPERT','PPIB',NBONDT,intg=PPIB)
  call chmalloc('pert.src','INIPERT','PPJB',NBONDT,intg=PPJB)
  call chmalloc('pert.src','INIPERT','PPIT',NTHETT,intg=PPIT)
  call chmalloc('pert.src','INIPERT','PPJT',NTHETT,intg=PPJT)
  call chmalloc('pert.src','INIPERT','PPKT',NTHETT,intg=PPKT)
  call chmalloc('pert.src','INIPERT','PPLT',NTHETT,intg=PPLT)
  call chmalloc('pert.src','INIPERT','PPSTRBL',2,NTHETT,intg=PPSTRBL)
  call chmalloc('pert.src','INIPERT','PPIP',NPHIT,intg=PPIP)
  call chmalloc('pert.src','INIPERT','PPJP',NPHIT,intg=PPJP)
  call chmalloc('pert.src','INIPERT','PPKP',NPHIT,intg=PPKP)
  call chmalloc('pert.src','INIPERT','PPLP',NPHIT,intg=PPLP)
  call chmalloc('pert.src','INIPERT','PPIM',NIMPHT,intg=PPIM)
  call chmalloc('pert.src','INIPERT','PPJM',NIMPHT,intg=PPJM)
  call chmalloc('pert.src','INIPERT','PPKM',NIMPHT,intg=PPKM)
  call chmalloc('pert.src','INIPERT','PPLM',NIMPHT,intg=PPLM)
#if KEY_CMAP==1
  call chmalloc('pert.src','INIPERT','PPI1CT',NCRTT,intg=PPI1CT)
  call chmalloc('pert.src','INIPERT','PPJ1CT',NCRTT,intg=PPJ1CT)
  call chmalloc('pert.src','INIPERT','PPK1CT',NCRTT,intg=PPK1CT)
  call chmalloc('pert.src','INIPERT','PPL1CT',NCRTT,intg=PPL1CT)
  call chmalloc('pert.src','INIPERT','PPI2CT',NCRTT,intg=PPI2CT)
  call chmalloc('pert.src','INIPERT','PPJ2CT',NCRTT,intg=PPJ2CT)
  call chmalloc('pert.src','INIPERT','PPK2CT',NCRTT,intg=PPK2CT)
  call chmalloc('pert.src','INIPERT','PPL2CT',NCRTT,intg=PPL2CT)
  call chmalloc('pert.src','INIPERT','PPICCT',NCRTT,intg=PPICCT)
#endif 
  call chmalloc('pert.src','INIPERT','PPIDON',NDON,intg=PPIDON)
  call chmalloc('pert.src','INIPERT','PPIHD1',NDON,intg=PPIHD1)
  call chmalloc('pert.src','INIPERT','PPIACC',NACC,intg=PPIACC)
  call chmalloc('pert.src','INIPERT','PPIAC1',NACC,intg=PPIAC1)
  call chmalloc('pert.src','INIPERT','PPINB',NNB,intg=PPINB)
  call chmalloc('pert.src','INIPERT','PPIBLO',MAXAIM,intg=PPIBLO)
  call chmalloc('pert.src','INIPERT','PPCG',MAXAIM,crl=PPCG)
  call chmalloc('pert.src','INIPERT','PPALPHA',MAXAIM,crl=PPALPHA)
  call chmalloc('pert.src','INIPERT','PPTHOLE',MAXAIM,crl=PPTHOLE)
  call chmalloc('pert.src','INIPERT','PPAMASS',MAXAIM,crl=PPAMASS)
  call chmalloc('pert.src','INIPERT','PPRSCLF',MAXAIM,crl=PPRSCLF)
#if KEY_WCA==1
  call chmalloc('pert.src','INIPERT','PPWCA',MAXAIM,crl=PPWCA)      
#endif
  call chmalloc('pert.src','INIPERT','PPICB',NBONDT,intg=PPICB)
  call chmalloc('pert.src','INIPERT','PPICT',NTHETT,intg=PPICT)
  call chmalloc('pert.src','INIPERT','PPICP',NPHIT,intg=PPICP)
  call chmalloc('pert.src','INIPERT','PPICI',NIMPHT,intg=PPICI)
  call chmalloc('pert.src','INIPERT','PPICOOP',NTHETT,intg=PPICOOP)
  call chmalloc('pert.src','INIPERT','PPICSTBN',NTHETT,intg=PPICSTBN)
  call chmalloc('pert.src','INIPERT','PPIACNB',MAXAIM,intg=PPIACNB)
  !
  ! Allocate pert restraint space
  !
  call chmalloc('pert.src','INIPERT','PRKCEXP',NUMHSETS,intg=PRKCEXP)
  call chmalloc('pert.src','INIPERT','PRICS',NCSPHI,intg=PRICS)
  call chmalloc('pert.src','INIPERT','PRJCS',NCSPHI,intg=PRJCS)
  call chmalloc('pert.src','INIPERT','PRKCS',NCSPHI,intg=PRKCS)
  call chmalloc('pert.src','INIPERT','PRLCS',NCSPHI,intg=PRLCS)
  call chmalloc('pert.src','INIPERT','PRICCS',NCSPHI,intg=PRICCS)
  call chmalloc('pert.src','INIPERT','PRCCSD',NCSPHI,intg=PRCCSD)
  call chmalloc('pert.src','INIPERT','PRTYPEH',NUMHSETS,intg=PRTYPEH)
  call chmalloc('pert.src','INIPERT','PRIHSET',NATOM,intg=PRIHSET)
  call chmalloc('pert.src','INIPERT','PRKCNST',NATOM,crl=PRKCNST)
  call chmalloc('pert.src','INIPERT','PRREFX',NATOM,crl=PRREFX)
  call chmalloc('pert.src','INIPERT','PRREFY',NATOM,crl=PRREFY)
  call chmalloc('pert.src','INIPERT','PRREFZ',NATOM,crl=PRREFZ)
  call chmalloc('pert.src','INIPERT','PRCCSC',NCSPHI,crl=PRCCSC)
  call chmalloc('pert.src','INIPERT','PRCCSB',NCSPHI,crl=PRCCSB)
  call chmalloc('pert.src','INIPERT','PRCCSW',NCSPHI,crl=PRCCSW)
  call chmalloc('pert.src','INIPERT','PRCCSCOS',NCSPHI,crl=PRCCSCOS)
  call chmalloc('pert.src','INIPERT','PRCCSSIN',NCSPHI,crl=PRCCSSIN)
  call chmalloc('pert.src','INIPERT','PRXHSCAL',NUMHSETS,crl=PRXHSCAL)
  call chmalloc('pert.src','INIPERT','PRYHSCAL',NUMHSETS,crl=PRYHSCAL)
  call chmalloc('pert.src','INIPERT','PRZHSCAL',NUMHSETS,crl=PRZHSCAL)
  call chmalloc('pert.src','INIPERT','PRQHNORT',NUMHSETS,log=PRQHNORT)
  call chmalloc('pert.src','INIPERT','PRQHNOTR',NUMHSETS,log=PRQHNOTR)
  call chmalloc('pert.src','INIPERT','PRNEIPT',NOENUM,intg=PRNEIPT)
  call chmalloc('pert.src','INIPERT','PRNEJPT',NOENUM,intg=PRNEJPT)
  call chmalloc('pert.src','INIPERT','PRNEINM',NOENUM,intg=PRNEINM)
  call chmalloc('pert.src','INIPERT','PRNEJNM',NOENUM,intg=PRNEJNM)
  call chmalloc('pert.src','INIPERT','PRNELIS',NOENUM,intg=PRNELIS)
  call chmalloc('pert.src','INIPERT','PRNERMN',NOENUM,crl=PRNERMN)
  call chmalloc('pert.src','INIPERT','PRNEKMN',NOENUM,crl=PRNEKMN)
  call chmalloc('pert.src','INIPERT','PRNERMX',NOENUM,crl=PRNERMX)
  call chmalloc('pert.src','INIPERT','PRNEKMX',NOENUM,crl=PRNEKMX)
  call chmalloc('pert.src','INIPERT','PRNEFMX',NOENUM,crl=PRNEFMX)
  call chmalloc('pert.src','INIPERT','PRNETCN',NOENUM,crl=PRNETCN)
  call chmalloc('pert.src','INIPERT','PRNEAVE',NOENUM,crl=PRNEAVE)
  call chmalloc('pert.src','INIPERT','PRNEEXP',NOENUM,crl=PRNEEXP)
  call chmalloc('pert.src','INIPERT','PRNEMIN',NOENUM,log=PRNEMIN)
  call chmalloc('pert.src','INIPERT','PRNERSW',NOENUM,crl=PRNERSW)
  call chmalloc('pert.src','INIPERT','PRNESEX',NOENUM,crl=PRNESEX)
  call chmalloc('pert.src','INIPERT','PRNERAM',NOENUM,intg=PRNERAM)
  !
  RETURN
END SUBROUTINE INIPERT

SUBROUTINE CLEARPERT
  !
  !  This routine deallocates the space for the PERT datastructures

  use memory
  use pert

  if(allocated(PERTIP))    &
       call chmdealloc('pert.src','clearpert','PERTIP',siz1=SIZE(PERTIP),intg=PERTIP)
  if(allocated(PERTIG))    &
       call chmdealloc('pert.src','clearpert','PERTIG',siz1=SIZE(PERTIG),intg=PERTIG)
  if(allocated(PERTDX))    &
       call chmdealloc('pert.src','clearpert','PERTDX',siz1=SIZE(PERTDX),crl=PERTDX)
  if(allocated(PERTDY))    &
       call chmdealloc('pert.src','clearpert','PERTDY',siz1=SIZE(PERTDY),crl=PERTDY)
  if(allocated(PERTDZ))    &
       call chmdealloc('pert.src','clearpert','PERTDZ',siz1=SIZE(PERTDZ),crl=PERTDZ)
  if(allocated(PERTEP1))   &
       call chmdealloc('pert.src','clearpert','PERTEP1',siz1=SIZE(PERTEP1),crl=PERTEP1)
  if(allocated(PERTEP2))   &
       call chmdealloc('pert.src','clearpert','PERTEP2',siz1=SIZE(PERTEP2),crl=PERTEP2)
  if(allocated(PERTEPC))   &
       call chmdealloc('pert.src','clearpert','PERTEPC',siz1=SIZE(PERTEPC),crl=PERTEPC)
  if(allocated(PERTCONST)) &
       call chmdealloc('pert.src','clearpert','PERTCONST',siz1=SIZE(PERTCONST),intg=PERTCONST)
  if(allocated(PERTSHAUX)) &
       call chmdealloc('pert.src','clearpert','PERTSHAUX',siz1=SIZE(PERTSHAUX),crl=PERTSHAUX)

  !
  ! Deallocate pert PSF space
  !

  if(allocated(PPNICTT))   &
       call chmdealloc('pert.src','clearpert','PPNICTT',siz1=SIZE(PPNICTT),intg=PPNICTT)
  if(allocated(PPIBASE))   &
       call chmdealloc('pert.src','clearpert','PPIBASE',siz1=SIZE(PPIBASE),intg=PPIBASE)
  if(allocated(PPIGPBS))   &
       call chmdealloc('pert.src','clearpert','PPIGPBS',siz1=SIZE(PPIGPBS),intg=PPIGPBS)
  if(allocated(PPIGPTP))   &
       call chmdealloc('pert.src','clearpert','PPIGPTP',siz1=SIZE(PPIGPTP),intg=PPIGPTP)
  if(allocated(PPIAC))     &
       call chmdealloc('pert.src','clearpert','PPIAC',siz1=SIZE(PPIAC),intg=PPIAC)
  if(allocated(PPIB))      &
       call chmdealloc('pert.src','clearpert','PPIB',siz1=SIZE(PPIB),intg=PPIB)
  if(allocated(PPJB))      &
       call chmdealloc('pert.src','clearpert','PPJB',siz1=SIZE(PPJB),intg=PPJB)
  if(allocated(PPIT))      &
       call chmdealloc('pert.src','clearpert','PPIT',siz1=SIZE(PPIT),intg=PPIT)
  if(allocated(PPJT))      &
       call chmdealloc('pert.src','clearpert','PPJT',siz1=SIZE(PPJT),intg=PPJT)
  if(allocated(PPKT))      &
       call chmdealloc('pert.src','clearpert','PPKT',siz1=SIZE(PPKT),intg=PPKT)
  if(allocated(PPLT))      &
       call chmdealloc('pert.src','clearpert','PPLT',siz1=SIZE(PPLT),intg=PPLT)
  if(allocated(PPSTRBL))   &
       call chmdealloc('pert.src','clearpert','PPSTRBL',2,SIZE(PPSTRBL)/2,intg=PPSTRBL)
  if(allocated(PPIP))      &
       call chmdealloc('pert.src','clearpert','PPIP',siz1=SIZE(PPIP),intg=PPIP)
  if(allocated(PPJP))      &
       call chmdealloc('pert.src','clearpert','PPJP',siz1=SIZE(PPJP),intg=PPJP)
  if(allocated(PPKP))      &
       call chmdealloc('pert.src','clearpert','PPKP',siz1=SIZE(PPKP),intg=PPKP)
  if(allocated(PPLP))      &
       call chmdealloc('pert.src','clearpert','PPLP',siz1=SIZE(PPLP),intg=PPLP)
  if(allocated(PPIM))      &
       call chmdealloc('pert.src','clearpert','PPIM',siz1=SIZE(PPIM),intg=PPIM)
  if(allocated(PPJM))      &
       call chmdealloc('pert.src','clearpert','PPJM',siz1=SIZE(PPJM),intg=PPJM)
  if(allocated(PPKM))      &
       call chmdealloc('pert.src','clearpert','PPKM',siz1=SIZE(PPKM),intg=PPKM)
  if(allocated(PPLM))      &
       call chmdealloc('pert.src','clearpert','PPLM',siz1=SIZE(PPLM),intg=PPLM)
#if KEY_CMAP==1
  if(allocated(PPI1CT))    &
       call chmdealloc('pert.src','clearpert','PPI1CT',siz1=SIZE(PPI1CT),intg=PPI1CT)
  if(allocated(PPJ1CT))    &
       call chmdealloc('pert.src','clearpert','PPJ1CT',siz1=SIZE(PPJ1CT),intg=PPJ1CT)
  if(allocated(PPK1CT))    &
       call chmdealloc('pert.src','clearpert','PPK1CT',siz1=SIZE(PPK1CT),intg=PPK1CT)
  if(allocated(PPL1CT))    &
       call chmdealloc('pert.src','clearpert','PPL1CT',siz1=SIZE(PPL1CT),intg=PPL1CT)
  if(allocated(PPI2CT))    &
       call chmdealloc('pert.src','clearpert','PPI2CT',siz1=SIZE(PPI2CT),intg=PPI2CT)
  if(allocated(PPJ2CT))    &
       call chmdealloc('pert.src','clearpert','PPJ2CT',siz1=SIZE(PPJ2CT),intg=PPJ2CT)
  if(allocated(PPK2CT))    &
       call chmdealloc('pert.src','clearpert','PPK2CT',siz1=SIZE(PPK2CT),intg=PPK2CT)
  if(allocated(PPL2CT))    &
       call chmdealloc('pert.src','clearpert','PPL2CT',siz1=SIZE(PPL2CT),intg=PPL2CT)
  if(allocated(PPICCT))    &
       call chmdealloc('pert.src','clearpert','PPICCT',siz1=SIZE(PPICCT),intg=PPICCT)
#endif 
  if(allocated(PPIDON))    &
       call chmdealloc('pert.src','clearpert','PPIDON',siz1=SIZE(PPIDON),intg=PPIDON)
  if(allocated(PPIHD1))    &
       call chmdealloc('pert.src','clearpert','PPIHD1',siz1=SIZE(PPIHD1),intg=PPIHD1)
  if(allocated(PPIACC))    &
       call chmdealloc('pert.src','clearpert','PPIACC',siz1=SIZE(PPIACC),intg=PPIACC)
  if(allocated(PPIAC1))    &
       call chmdealloc('pert.src','clearpert','PPIAC1',siz1=SIZE(PPIAC1),intg=PPIAC1)
  if(allocated(PPINB))     &
       call chmdealloc('pert.src','clearpert','PPINB',siz1=SIZE(PPINB),intg=PPINB)
  if(allocated(PPIBLO))    &
       call chmdealloc('pert.src','clearpert','PPIBLO',siz1=SIZE(PPIBLO),intg=PPIBLO)
  if(allocated(PPCG))      &
       call chmdealloc('pert.src','clearpert','PPCG',siz1=SIZE(PPCG),crl=PPCG)
  if(allocated(PPALPHA))      &
       call chmdealloc('pert.src','clearpert','PPALPHA',siz1=SIZE(PPALPHA),crl=PPALPHA)
  if(allocated(PPTHOLE))      &
       call chmdealloc('pert.src','clearpert','PPTHOLE',siz1=SIZE(PPTHOLE),crl=PPTHOLE)
  if(allocated(PPAMASS))   &
       call chmdealloc('pert.src','clearpert','PPAMASS',siz1=SIZE(PPAMASS),crl=PPAMASS)
  if(allocated(PPRSCLF))   &
       call chmdealloc('pert.src','clearpert','PPRSCLF',siz1=SIZE(PPRSCLF),crl=PPRSCLF)
#if KEY_WCA==1
  if(allocated(PPWCA))     &
       call chmdealloc('pert.src','clearpert','PPWCA',siz1=SIZE(PPWCA),crl=PPWCA)
#endif 
  if(allocated(PPICB))     &
       call chmdealloc('pert.src','clearpert','PPICB',siz1=SIZE(PPICB),intg=PPICB)
  if(allocated(PPICT))     &
       call chmdealloc('pert.src','clearpert','PPICT',siz1=SIZE(PPICT),intg=PPICT)
  if(allocated(PPICP))     &
       call chmdealloc('pert.src','clearpert','PPICP',siz1=SIZE(PPICP),intg=PPICP)
  if(allocated(PPICI))     &
       call chmdealloc('pert.src','clearpert','PPICI',siz1=SIZE(PPICI),intg=PPICI)
  if(allocated(PPICOOP))   &
       call chmdealloc('pert.src','clearpert','PPICOOP',siz1=SIZE(PPICOOP),intg=PPICOOP)
  if(allocated(PPICSTBN))  &
       call chmdealloc('pert.src','clearpert','PPICSTBN',siz1=SIZE(PPICSTBN),intg=PPICSTBN)
  if(allocated(PPIACNB))   &
       call chmdealloc('pert.src','clearpert','PPIACNB',siz1=SIZE(PPIACNB),intg=PPIACNB)
  !
  ! Deallocate pert restraint space
  !
  if(allocated(PRKCEXP))   &
       call chmdealloc('pert.src','clearpert','PRKCEXP',siz1=SIZE(PRKCEXP),intg=PRKCEXP)
  if(allocated(PRICS))     &
       call chmdealloc('pert.src','clearpert','PRICS',siz1=SIZE(PRICS),intg=PRICS)
  if(allocated(PRJCS))     &
       call chmdealloc('pert.src','clearpert','PRJCS',siz1=SIZE(PRJCS),intg=PRJCS)
  if(allocated(PRKCS))     &
       call chmdealloc('pert.src','clearpert','PRKCS',siz1=SIZE(PRKCS),intg=PRKCS)
  if(allocated(PRLCS))     &
       call chmdealloc('pert.src','clearpert','PRLCS',siz1=SIZE(PRLCS),intg=PRLCS)
  if(allocated(PRICCS))    &
       call chmdealloc('pert.src','clearpert','PRICCS',siz1=SIZE(PRICCS),intg=PRICCS)
  if(allocated(PRCCSD))    &
       call chmdealloc('pert.src','clearpert','PRCCSD',siz1=SIZE(PRCCSD),intg=PRCCSD)
  if(allocated(PRTYPEH))   &
       call chmdealloc('pert.src','clearpert','PRTYPEH',siz1=SIZE(PRTYPEH),intg=PRTYPEH)
  if(allocated(PRIHSET))   &
       call chmdealloc('pert.src','clearpert','PRIHSET',siz1=SIZE(PRIHSET),intg=PRIHSET)
  if(allocated(PRKCNST))   &
       call chmdealloc('pert.src','clearpert','PRKCNST',siz1=SIZE(PRKCNST),crl=PRKCNST)
  if(allocated(PRREFX))    &
       call chmdealloc('pert.src','clearpert','PRREFX',siz1=SIZE(PRREFX),crl=PRREFX)
  if(allocated(PRREFY))    &
       call chmdealloc('pert.src','clearpert','PRREFY',siz1=SIZE(PRREFY),crl=PRREFY)
  if(allocated(PRREFZ))    &
       call chmdealloc('pert.src','clearpert','PRREFZ',siz1=SIZE(PRREFZ),crl=PRREFZ)
  if(allocated(PRCCSC))    &
       call chmdealloc('pert.src','clearpert','PRCCSC',siz1=SIZE(PRCCSC),crl=PRCCSC)
  if(allocated(PRCCSB))    &
       call chmdealloc('pert.src','clearpert','PRCCSB',siz1=SIZE(PRCCSB),crl=PRCCSB)
  if(allocated(PRCCSW))    &
       call chmdealloc('pert.src','clearpert','PRCCSW',siz1=SIZE(PRCCSW),crl=PRCCSW)
  if(allocated(PRCCSCOS))  &
       call chmdealloc('pert.src','clearpert','PRCCSCOS',siz1=SIZE(PRCCSCOS),crl=PRCCSCOS)
  if(allocated(PRCCSSIN))  &
       call chmdealloc('pert.src','clearpert','PRCCSSIN',siz1=SIZE(PRCCSSIN),crl=PRCCSSIN)
  if(allocated(PRXHSCAL))  &
       call chmdealloc('pert.src','clearpert','PRXHSCAL',siz1=SIZE(PRXHSCAL),crl=PRXHSCAL)
  if(allocated(PRYHSCAL))  &
       call chmdealloc('pert.src','clearpert','PRYHSCAL',siz1=SIZE(PRYHSCAL),crl=PRYHSCAL)
  if(allocated(PRZHSCAL))  &
       call chmdealloc('pert.src','clearpert','PRZHSCAL',siz1=SIZE(PRZHSCAL),crl=PRZHSCAL)
  if(allocated(PRQHNORT))  &
       call chmdealloc('pert.src','clearpert','PRQHNORT',siz1=SIZE(PRQHNORT),log=PRQHNORT)
  if(allocated(PRQHNOTR))  &
       call chmdealloc('pert.src','clearpert','PRQHNOTR',siz1=SIZE(PRQHNOTR),log=PRQHNOTR)
  if(allocated(PRNEIPT))   &
       call chmdealloc('pert.src','clearpert','PRNEIPT',siz1=SIZE(PRNEIPT),intg=PRNEIPT)
  if(allocated(PRNEJPT))   &
       call chmdealloc('pert.src','clearpert','PRNEJPT',siz1=SIZE(PRNEJPT),intg=PRNEJPT)
  if(allocated(PRNEINM))   &
       call chmdealloc('pert.src','clearpert','PRNEINM',siz1=SIZE(PRNEINM),intg=PRNEINM)
  if(allocated(PRNEJNM))   &
       call chmdealloc('pert.src','clearpert','PRNEJNM',siz1=SIZE(PRNEJNM),intg=PRNEJNM)
  if(allocated(PRNELIS))   &
       call chmdealloc('pert.src','clearpert','PRNELIS',siz1=SIZE(PRNELIS),intg=PRNELIS)
  if(allocated(PRNERMN))   &
       call chmdealloc('pert.src','clearpert','PRNERMN',siz1=SIZE(PRNERMN),crl=PRNERMN)
  if(allocated(PRNEKMN))   &
       call chmdealloc('pert.src','clearpert','PRNEKMN',siz1=SIZE(PRNEKMN),crl=PRNEKMN)
  if(allocated(PRNERMX))   &
       call chmdealloc('pert.src','clearpert','PRNERMX',siz1=SIZE(PRNERMX),crl=PRNERMX)
  if(allocated(PRNEKMX))   &
       call chmdealloc('pert.src','clearpert','PRNEKMX',siz1=SIZE(PRNEKMX),crl=PRNEKMX)
  if(allocated(PRNEFMX))   &
       call chmdealloc('pert.src','clearpert','PRNEFMX',siz1=SIZE(PRNEFMX),crl=PRNEFMX)
  if(allocated(PRNETCN))   &
       call chmdealloc('pert.src','clearpert','PRNETCN',siz1=SIZE(PRNETCN),crl=PRNETCN)
  if(allocated(PRNEAVE))   &
       call chmdealloc('pert.src','clearpert','PRNEAVE',siz1=SIZE(PRNEAVE),crl=PRNEAVE)
  if(allocated(PRNEEXP))   &
       call chmdealloc('pert.src','clearpert','PRNEEXP',siz1=SIZE(PRNEEXP),crl=PRNEEXP)
  if(allocated(PRNEMIN))   &
       call chmdealloc('pert.src','clearpert','PRNEMIN',siz1=SIZE(PRNEMIN),log=PRNEMIN)
  if(allocated(PRNERSW))   &
       call chmdealloc('pert.src','clearpert','PRNERSW',siz1=SIZE(PRNERSW),crl=PRNERSW)
  if(allocated(PRNESEX))   &
       call chmdealloc('pert.src','clearpert','PRNESEX',siz1=SIZE(PRNESEX),crl=PRNESEX)
  if(allocated(PRNERAM))   &
       call chmdealloc('pert.src','clearpert','PRNERAM',siz1=SIZE(PRNERAM),intg=PRNERAM)
  if(allocated(prpcax)) then
     call chmdealloc('pert.src','clearpert','PRPCAX',size(prpcax,1),size(prpcax,2),crl=prpcax)
     call chmdealloc('pert.src','clearpert','PRPCAY',size(prpcay,1),size(prpcay,2),crl=prpcay)
     call chmdealloc('pert.src','clearpert','PRPCAZ',size(prpcaz,1),size(prpcaz,2),crl=prpcaz)
  endif

  RETURN
END SUBROUTINE CLEARPERT

SUBROUTINE PERTS2(IPERT,IGPERT)
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  !     Jay Banks, 20 October 1995: added MMFF-specific information
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use bases_fcm
  use psf
  use pert
  use code
  use fast
  use cnst_fcm
  use noem
  use stream
  use mmffm
  use ffieldm
  use exelecm  ! NKB, exelec with pert
  use mmfp
  use memory
  ! QC: 11/17 Add DFTB PERT based on Xiya
#if KEY_SCCDFTB==1
  use gamess_fcm, only: QGMREM,IGMSEL  
  use number
#endif
  ! QC: 11/17 End
  implicit none
  !
  INTEGER IPERT(:),IGPERT(:)
  !
  !
  INTEGER I,J,IS,IQ
  !
  ! Copy PSF data
  !
  NATOMP=         NATOM
  NRESP=          NRES
  NSEGP=          NSEG
  NBONDP=         NBOND
  NTHETP=         NTHETA
  NPHIP=          NPHI
  NIMPHP=         NIMPHI
#if KEY_CMAP==1
  NCRTP=          NCRTERM
#endif 
  NGRPP=          NGRP
  NATMTP=         NATOMT
  NRESTP=         NREST
  NSEGTP=         NSEGT
  NBONTP=         NBONDT
  NTHTTP=         NTHETT
  NPHITP=         NPHIT
  NIMPTP=         NIMPHT
  NGRPTP=         NGRPT
  NNBP=           NNB
  NDONP=          NDON
  NACCP=          NACC
  NST2P=          NST2
  CGTOTP=         CGTOT
  !
  DO J=1,NGRPT
     IS=IGPBS(J)+1
     IQ=IGPBS(J+1)
     IGPERT(J)=0
     DO I=IS,IQ
        IF(IPERT(I) == 1) IGPERT(J)=1
     ENDDO
  ENDDO
  !
  PPNICTT(1:NSEGT+1) = NICTOT(1:NSEGT+1)
  PPIBASE(1:NREST+1) = IBASE (1:NREST+1)
  PPIGPBS(1:NGRPT+1) = IGPBS (1:NGRPT+1)
  PPIGPTP(1:NGRPT)   = IGPTYP(1:NGRPT)
  PPIAC  (1:NATOMT)  = IAC   (1:NATOMT)
  PPIACNB(1:NATOMT)  = IACNB (1:NATOMT)
  PPIB   (1:NBONDT)  = IB    (1:NBONDT)
  PPJB   (1:NBONDT)  = JB    (1:NBONDT)
  PPIT   (1:NTHETT)  = IT    (1:NTHETT)
  PPJT   (1:NTHETT)  = JT    (1:NTHETT)
  PPKT   (1:NTHETT)  = KT    (1:NTHETT)
#if KEY_MMFF==1
  if(ffield == mmff) then
     pplt(1:nthett) = LTHETA(1:NTHETT)
     PPSTRBL(1:2,1:NTHETT) = StrbList(1:2,1:NTHETT)
  endif
#endif /*  MMFF*/
  PPIP(1:NPHIT)  = IP(1:NPHIT)
  PPJP(1:NPHIT)  = JP(1:NPHIT)
  PPKP(1:NPHIT)  = KP(1:NPHIT)
  PPLP(1:NPHIT)  = LP(1:NPHIT)
  PPIM(1:NIMPHT) = IM(1:NIMPHT)
  PPJM(1:NIMPHT) = JM(1:NIMPHT)
  PPKM(1:NIMPHT) = KM(1:NIMPHT)
  PPLM(1:NIMPHT) = LM(1:NIMPHT)
#if KEY_CMAP==1
  PPI1CT(1:NCRTT) = I1CT(1:NCRTT)
  PPJ1CT(1:NCRTT) = J1CT(1:NCRTT)
  PPK1CT(1:NCRTT) = K1CT(1:NCRTT)
  PPL1CT(1:NCRTT) = L1CT(1:NCRTT)
  PPI2CT(1:NCRTT) = I2CT(1:NCRTT)
  PPJ2CT(1:NCRTT) = J2CT(1:NCRTT)
  PPK2CT(1:NCRTT) = K2CT(1:NCRTT)
  PPL2CT(1:NCRTT) = L2CT(1:NCRTT)
#endif 

  PPIDON(1:NDON ) = IDON(1:NDON) 
  PPIHD1(1:NDON ) = IHD1(1:NDON)
  PPIACC(1:NACC ) = IACC(1:NACC)
  PPIAC1(1:NACC ) = IAC1(1:NACC)
  PPINB (1:NNB  ) = INB (1:NNB)
  PPIBLO(1:NATOMT) = IBLO(1:NATOMT)
  ppcg(1:natomt)    = CG   (1:NATOMT)
! QC: 11/17 Add DFTB PERT based on Xiya
! Though it is not used really?!
#if KEY_SCCDFTB==1
  ! Zero out charges on quantum atoms to remove from MM term. It hasn't been done for PPCG. Xiya
  !IF(QGMREM) THEN 
  !   DO I=1, NATOM
  !      IF((IGMSEL(I).GT.0).AND.(IGMSEL(I).LT.3)) PPCG(I)=ZERO
  !   ENDDO
  !ENDIF
#endif
! QC: 11/17 Done
  ppalpha(1:natomt) = ALPHADP(1:NATOMT)
  ppthole(1:natomt) = THOLEI(1:NATOMT)
  ppamass(1:natomt) = AMASS(1:NATOMT)
  pprsclf(1:natomt) = RSCLF(1:NATOMT)
#if KEY_WCA==1
  ppwca(1:natomt)   = WCA  (1:NATOMT)        
#endif
  !
  ! Copy CODES DATA
  !
  PPICB(1:NBONDT) = ICB(1:NBONDT)
  PPICT(1:NTHETT) = ICT(1:NTHETT)
#if KEY_MMFF==1
  if(ffield == mmff) then
     PPICOOP (1:nthett) = ICOOP (1:NTHETT)
     PPICSTBN(1:nthett) = ICSTBN(1:NTHETT)
  endif
#endif 
  PPICP (1:NPHIT ) = ICP (1:NPHIT)
  PPICI (1:NIMPHT) = ICI (1:NIMPHT)
#if KEY_CMAP==1
  PPICCT(1:NCRTT)  = ICCT(1:NCRTT)     
#endif
  !
  ! Copy Harmonic restraint data
  !
  if(.not. allocated(refx))then
     call allocate_cnst(natom)
  else
     if(natom > size(refx)) call allocate_cnst(natom)
  endif
  NUMHSETP=  NUMHSETS
  if (numpca > 0) then
     numpcap = numpca
     call chmalloc('pert.src','PERTS2','prpcax',numpcap,natom,crl=prpcax)
     call chmalloc('pert.src','PERTS2','prpcay',numpcap,natom,crl=prpcay)
     call chmalloc('pert.src','PERTS2','prpcaz',numpcap,natom,crl=prpcaz)
     prpcax(1:numpcap,1:natom) = pcax(1:numpca,1:natom)
     prpcay(1:numpcap,1:natom) = pcay(1:numpca,1:natom)
     prpcaz(1:numpcap,1:natom) = pcaz(1:numpca,1:natom)
  endif
  PRTYPEH(1:NUMHSETS) = TYPHSET(1:NUMHSETS)
  PRKCEXP(1:NUMHSETS) = KCEXPN (1:NUMHSETS)
  prxhscal(1:numhsets) = XHSCALE(1:NUMHSETS)
  pryhscal(1:numhsets) = YHSCALE(1:NUMHSETS)
  przhscal(1:numhsets) = ZHSCALE(1:NUMHSETS)
  PRQHNORT(1:NUMHSETS) = QHNORT(1:NUMHSETS)
  PRQHNOTR(1:NUMHSETS) = QHNOTR(1:NUMHSETS)
  PRIHSET (1:NATOM   ) = IHSET (1:NATOM)
  prkcnst(1:natom) = KCNSTR(1:NATOM)
  prrefx(1:natom) = REFX(1:NATOM)
  prrefy(1:natom) = REFY(1:NATOM)
  prrefz(1:natom) = REFZ(1:NATOM)
  NCSPHP=    NCSPHI
  PRICS (1:NCSPHI) = ICS (1:NCSPHI)
  PRJCS (1:NCSPHI) = JCS (1:NCSPHI)
  PRKCS (1:NCSPHI) = KCS (1:NCSPHI)
  PRLCS (1:NCSPHI) = LCS (1:NCSPHI)
  PRICCS(1:NCSPHI) = ICCS(1:NCSPHI)
  PRCCSD(1:NCSPHI) = CCSD(1:NCSPHI)
  prccsc(1:ncsphi)   = CCSC  (1:NCSPHI)
  prccsb(1:ncsphi)   = CCSB  (1:NCSPHI)
  prccsw(1:ncsphi)   = CCSW  (1:NCSPHI)
  prccscos(1:ncsphi) = CCSCOS(1:NCSPHI)
  prccssin(1:ncsphi) = CCSSIN(1:NCSPHI)
  QCNSRP=    QCNSTR
  !
#if KEY_NOMISC==0
  ! Copy NOE restraint data
  !
  NENUMP=    NOENUM
  NENM2P=    NOENM2
  NESCAP=    NOESCA
  PRNEIPT(1:noenum) = NOEIPT(1:NOENUM)
  PRNEJPT(1:noenum) = NOEJPT(1:NOENUM)
  PRNEINM(1:noenum) = NOEINM(1:NOENUM)
  PRNEJNM(1:noenum) = NOEJNM(1:NOENUM)
  prnermn(1:noenum) = NOERMN(1:NOENUM)
  prnekmn(1:noenum) = NOEKMN(1:NOENUM)
  prnermx(1:noenum) = NOERMX(1:NOENUM)
  prnekmx(1:noenum) = NOEKMX(1:NOENUM)
  prnefmx(1:noenum) = NOEFMX(1:NOENUM)
  prnetcn(1:noenum) = NOETCN(1:NOENUM)
  prneave(1:noenum) = NOEAVE(1:NOENUM)
  prneexp(1:noenum) = NOEEXP(1:NOENUM)
  prnersw(1:noenum) = NOERSW(1:NOENUM)
  prnesex(1:noenum) = NOESEX(1:NOENUM)
  PRNEMIN(1:NOENUM) = NOEMIN(1:NOENUM)
  PRNERAM(1:NOENUM) = NOERAM(1:NOENUM)
  PRNELIS(1:NOENM2) = NOELIS(1:NOENM2)
  !ML--------------------------------------------------------
  ! added MMFP-support---------------------------------------
  !
  !     allocate if GEO-restraint are set at lambda=0
  ! FIXME inappropriate intimacy
  IF(QMMFPE) THEN
     IF(QGEO) THEN

        !       Auxillary variable PMXGEO needed for FREHP
        PMXGEO = MAXGEO
        if (allocated(plsgeo))then
           call chmdealloc('pert.src','PERTS2','PLSGEO',PMXGEO,intg=PLSGEO)
           call chmdealloc('pert.src','PERTS2','PNGEO',PMXGEO,intg=PNGEO)
           call chmdealloc('pert.src','PERTS2','PIGEO',PMXGEO,intg=PIGEO)
           call chmdealloc('pert.src','PERTS2','PJGEO',PMXGEO,intg=PJGEO)
           call chmdealloc('pert.src','PERTS2','PXRGEO',PMXGEO,crl=PXRGEO)
           call chmdealloc('pert.src','PERTS2','PYRGEO',PMXGEO,crl=PYRGEO)
           call chmdealloc('pert.src','PERTS2','PZRGEO',PMXGEO,crl=PZRGEO)
           call chmdealloc('pert.src','PERTS2','PTRGEO',PMXGEO,crl=PTRGEO)
           call chmdealloc('pert.src','PERTS2','PXDGEO',PMXGEO,crl=PXDGEO)
           call chmdealloc('pert.src','PERTS2','PYDGEO',PMXGEO,crl=PYDGEO)
           call chmdealloc('pert.src','PERTS2','PZDGEO',PMXGEO,crl=PZDGEO)
           call chmdealloc('pert.src','PERTS2','PDRGEO',PMXGEO,crl=PDRGEO)
           call chmdealloc('pert.src','PERTS2','PDTGEO',PMXGEO,crl=PDTGEO)
           call chmdealloc('pert.src','PERTS2','PFCGEO',PMXGEO,crl=PFCGEO)
           call chmdealloc('pert.src','PERTS2','PP1GEO',PMXGEO,crl=PP1GEO)
           call chmdealloc('pert.src','PERTS2','PP2GEO',PMXGEO,crl=PP2GEO)
           call chmdealloc('pert.src','PERTS2','PP3GEO',PMXGEO,crl=PP3GEO)
           call chmdealloc('pert.src','PERTS2','PIUGEO',PMXGEO,intg=PIUGEO)
        endif
        call chmalloc('pert.src','PERTS2','PLSGEO',PMXGEO,intg=PLSGEO)
        call chmalloc('pert.src','PERTS2','PNGEO',PMXGEO,intg=PNGEO)
        call chmalloc('pert.src','PERTS2','PIGEO',PMXGEO,intg=PIGEO)
        call chmalloc('pert.src','PERTS2','PJGEO',PMXGEO,intg=PJGEO)
        call chmalloc('pert.src','PERTS2','PXRGEO',PMXGEO,crl=PXRGEO)
        call chmalloc('pert.src','PERTS2','PYRGEO',PMXGEO,crl=PYRGEO)
        call chmalloc('pert.src','PERTS2','PZRGEO',PMXGEO,crl=PZRGEO)
        call chmalloc('pert.src','PERTS2','PTRGEO',PMXGEO,crl=PTRGEO)
        call chmalloc('pert.src','PERTS2','PXDGEO',PMXGEO,crl=PXDGEO)
        call chmalloc('pert.src','PERTS2','PYDGEO',PMXGEO,crl=PYDGEO)
        call chmalloc('pert.src','PERTS2','PZDGEO',PMXGEO,crl=PZDGEO)
        call chmalloc('pert.src','PERTS2','PDRGEO',PMXGEO,crl=PDRGEO)
        call chmalloc('pert.src','PERTS2','PDTGEO',PMXGEO,crl=PDTGEO)
        call chmalloc('pert.src','PERTS2','PFCGEO',PMXGEO,crl=PFCGEO)
        call chmalloc('pert.src','PERTS2','PP1GEO',PMXGEO,crl=PP1GEO)
        call chmalloc('pert.src','PERTS2','PP2GEO',PMXGEO,crl=PP2GEO)
        call chmalloc('pert.src','PERTS2','PP3GEO',PMXGEO,crl=PP3GEO)
        call chmalloc('pert.src','PERTS2','PIUGEO',PMXGEO,intg=PIUGEO)

        !       copy lambda=0 data
        IF(PRNLEV >= 2) WRITE(OUTU,116)
116     FORMAT(' PERT: Copy MMFP data-structure')

        PNTGEO = NTGEO

        PLSGEO(1:PMXGEO) = LSTGEO(1:PMXGEO)
        PNGEO (1:PMXGEO) = NGEO  (1:PMXGEO)
        PIGEO (1:PMXGEO) = IGEO  (1:PMXGEO)
        PJGEO (1:PMXGEO) = JGEO  (1:PMXGEO)

        pxrgeo(1:pmxgeo) = XRGEO(1:PMXGEO)
        pyrgeo(1:pmxgeo) = YRGEO(1:PMXGEO)
        pzrgeo(1:pmxgeo) = ZRGEO(1:PMXGEO)
        ptrgeo(1:pmxgeo) = TRGEO(1:PMXGEO)

        pxdgeo(1:pmxgeo) = XDGEO(1:PMXGEO)
        pydgeo(1:pmxgeo) = YDGEO(1:PMXGEO)
        pzdgeo(1:pmxgeo) = ZDGEO(1:PMXGEO)
        pdrgeo(1:pmxgeo) = DRGEO(1:PMXGEO)
        pdtgeo(1:pmxgeo) = DTGEO(1:PMXGEO)
        pfcgeo(1:pmxgeo) = FCGEO(1:PMXGEO)
        pp1geo(1:pmxgeo) = P1GEO(1:PMXGEO)
        pp2geo(1:pmxgeo) = P2GEO(1:PMXGEO)
        pp3geo(1:pmxgeo) = P3GEO(1:PMXGEO)

        piugeo(1:pmxgeo) = IUGEO(1:PMXGEO)

        !       GEO data structure duplicated and used at lambda=0
        !       QZEGEO is needed! in epert.src
        QZEGEO= .TRUE.
     ENDIF

  ELSE
     !      if no GEO restraint at lambda=0, but intended at lambda=1
     QZEGEO=.FALSE.
  ENDIF
  !ML--------------------------------------------------------
#if KEY_CHEMPERT==1
  !sbcp chem pert additions
  if (qchemp) then
     call chmalloc('pert.src','PERTS2','cglig',maxaim,crl=cglig)
     cglig(1:natomt) = cg(1:natomt)
     call inichem(cg,cgtot,cglig,cgligt,natomt, &
          pertip)
  endif
  !sb
#endif 
#endif 
  !
  RETURN
END SUBROUTINE PERTS2

SUBROUTINE PERTDF
  !
  ! THIS ROUTINE PARSES ALL FREE ENERGY COMMAND
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  !
  use pert
  use comand
  use ctitla
  use stream
  use string
  implicit none
  !
  LOGICAL EOF
  !
  IWHAM=GTRMI(COMLYN,COMLEN,'WHAM',-1)
  IF(IWHAM > 0)THEN
     IF(PRNLEV >= 2) WRITE(OUTU,'(A,I3)') &
          ' PERTURBATION> write for WHAM analysis on punit',IWHAM
  ENDIF
  !
  PUNIT=GTRMI(COMLYN,COMLEN,'PUNI',-1)
  IF(PUNIT > 0) THEN
     CALL XTRANE(COMLYN,COMLEN,'PERTDF')
     IF (COMLEN > 0) CALL DIEWRN(-2)
     !
     CALL RDTITL(TITLEB,NTITLB,PUNIT,0)
     EOF=.FALSE.
     COMLEN=0
     DO WHILE(COMLEN == 0 .AND. .NOT.EOF)
        CALL RDCMND(COMLYN,MXCMSZ,COMLEN,PUNIT,EOF,.TRUE., &
             .TRUE.,'PUNIT> ')
     ENDDO
     IF(EOF) THEN
        ! end of commands, turn off perturbation.
        IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
             ' PERTURBATION> EOF on punit file: PERT cancelled.'
        PUNIT=-1
        COMLYN=' '
        COMLEN=0
     ENDIF
  ENDIF
  CALL PERTPS(COMLYN,COMLEN,.TRUE.)
  !
  RETURN
END SUBROUTINE PERTDF

SUBROUTINE PERTPS(COMLYN,COMLEN,QRESET)
  !
  ! THIS ROUTINE PARSES ALL FREE ENERGY OPTIONS
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use bases_fcm
  use psf
  use pert
  use energym
  use econtmod
  use epert_mod
  use stream
  use string
  use reawri
  use consta
  !
  implicit none
  !
  character(len=*) COMLYN
  INTEGER COMLEN
  LOGICAL QRESET
  !
  INTEGER I,NINC,IINC
  real(chm_real) RMARK
  real(chm_real) FACT1,FACT2,FACT3,FACT4
  !
  RMARK=-9999.0
  !
  ! Check limit values for recurring usage, or for the END command.
  PTERMN=.FALSE.
  IF(QRESET) THEN
     NLAMEXP=0
     LAMSTA=ZERO
     LAMSTOP=ZERO
  ELSE  
     IF(PUNIT < 0) THEN
        IF( ABS(LAMDAF-LAMSTOP) < RSMALL .AND. &
         ABS(LAMDAP-LAMSTOP) < RSMALL) PTERMN=.TRUE.
        IF(LAMINC < ZERO .AND. ABS(LAMDAF-LAMSTA) < RSMALL) PTERMN=.TRUE.
        IF(LAMINC == ZERO.AND.NLAMEXP==0) PTERMN=.TRUE.
     ENDIF
     IF(PTERMN) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
             ' PERTURBATION> Auto mode ends: PERT run terminated.'
        PUNIT=-1
        IPSTP=0
        IPSTRT=0
        PTERMN=.TRUE.
        IPNTOT=0
     ENDIF
     ! Check for END command.
     IF(INDXA(COMLYN,COMLEN,'END') /= 0) THEN
        IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
             ' PERTURBATION> END command issued: PERT run terminated.'
        PUNIT=-1
        IPSTP=0
        IPSTRT=0
        PTERMN=.TRUE.
        IPNTOT=0
     ENDIF
     IF(PTERMN) RETURN
  ENDIF
  !
  ! Parse all values and flags.
  IPINCR=GTRMI(COMLYN,COMLEN,'PINC',IPINCR)
  IPEQUI=GTRMI(COMLYN,COMLEN,'PEQU',IPEQUI)
  IPSTRT=GTRMI(COMLYN,COMLEN,'PSTA',IPSTP+IPEQUI)
  IPSTP =GTRMI(COMLYN,COMLEN,'PSTO',IPSTP+IPINCR)
  !
  IF(INDXA(COMLYN,COMLEN,'PWIN') /= 0) QPWIND=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'PSLO') /= 0) QPWIND=.FALSE.
  !
  ! use soft core / sep. shifted pot. ? ...
  IF(INDXA(COMLYN,COMLEN,'PSSP') /= 0) QPSSP=.TRUE.
  IF(INDXA(COMLYN,COMLEN,'NOPS') /= 0) QPSSP=.FALSE.
  IF (QPSSP) THEN
     DLAMBD=GTRMF(COMLYN,COMLEN,'DLAM',FIVE)
     ALAMBD=GTRMF(COMLYN,COMLEN,'ALAM',FIVE)
     LAORDER=GTRMI(COMLYN,COMLEN,'LAORDER',LAORDER)
     IF(PRNLEV >= 2) WRITE(OUTU,211)
211  FORMAT('PERTPS> PERT USES SOFT CORE INTERACTIONS')
     !  PARAMETER D FOR SOFT CORE POT ...
     IF(PRNLEV >= 2) WRITE(OUTU,212) DLAMBD
     IF(PRNLEV >= 2) WRITE(OUTU,213) ALAMBD
     IF(PRNLEV >= 2) WRITE(OUTU,215) LAORDER
212  FORMAT('PERTPS> DLAMBD = ',F10.5)
213  FORMAT('PERTPS> ALAMBD = ',F10.5)
215  FORMAT('PERTPS> LAORDER = ',I2)
  ENDIF
  !
  LAMINC=GTRMF(COMLYN,COMLEN,'LINC',LAMINC)
  ! number of lambdas 
  !NLAMEXP=GTRMF(COMLYN,COMLEN,'NLAMEXP',NLAMEXP)
  ! exponential increment constant
  LAMEXP=GTRMF(COMLYN,COMLEN,'LEXP',LAMEXP)
  !
  LAMSTA=GTRMF(COMLYN,COMLEN,'LSTA',LAMSTA)
  !
  LAMSTOP=GTRMF(COMLYN,COMLEN,'LSTO',LAMSTOP)
  !
  LAMDA=LAMDAP
  LAMDAP=GTRMF(COMLYN,COMLEN,'LAMB',RMARK)
  IF(LAMDAP /= RMARK) QPAVER=.FALSE.
  IF(INDXA(COMLYN,COMLEN,'LAVE') /= 0) QPAVER=.TRUE.
  IF(LAMDAP == RMARK) THEN
    IF(LAMEXP==ZERO)THEN
     IF(NLAMEXP==0)THEN
          IF(LAMINC/=ZERO)THEN
               NLAMEXP=ABS(NINT((LAMSTOP-LAMSTA)/LAMINC))
          ELSE
               NLAMEXP=1
          ENDIF
          ILAMEXP=0
          LAMDAF=LAMSTA
     ENDIF     
     IF(ILAMEXP==0)THEN
       LAMDAI=LAMSTA
       LAMDAP=LAMSTA
       LAMDAF=LAMSTA+HALF*(LAMSTOP-LAMSTA-(NLAMEXP-1)*LAMINC)
     ELSE IF(ILAMEXP.LT.NLAMEXP)THEN
       LAMDAI=LAMDAF
       LAMDAF=LAMDAF+LAMINC
       LAMDAP=HALF*(LAMDAI+LAMDAF)
     ELSE
       LAMDAI=LAMDAF
       LAMDAP=LAMSTOP
       LAMDAF=LAMSTOP
     ENDIF
     ILAMEXP=ILAMEXP+1
     IF(QPAVER) LAMDAP=HALF*(LAMDAI+LAMDAF)
     IF(.NOT.QPWIND) LAMDAP=LAMDAI
     IF(LAMDAP == RMARK) LAMDAP=LAMDA
     IF(LAMINC>ZERO.AND.LAMDAF>LAMSTOP)LAMDAF=LAMSTOP
     IF(LAMINC>ZERO.AND.LAMDAP>LAMSTOP)LAMDAP=LAMSTOP
     IF(LAMINC<ZERO.AND.LAMDAF<LAMSTOP)LAMDAF=LAMSTOP
     IF(LAMINC<ZERO.AND.LAMDAP<LAMSTOP)LAMDAP=LAMSTOP
    ELSE
     IF(NLAMEXP==0)THEN
       IF(LAMINC/=ZERO)THEN
          NLAMEXP=ABS(NINT((LAMSTOP-LAMSTA)/LAMINC))
       ELSE
          NLAMEXP=1
       ENDIF
       ILAMEXP=0
       LAMDAF=LAMSTA
     ENDIF
     IF(LAMEXP>ZERO)THEN
       FACT1=ONE-EXP(-LAMEXP*LAMEXP)
       FACT2=LAMEXP*ILAMEXP/NLAMEXP
       FACT3=(ONE-EXP(-FACT2*FACT2))/FACT1
       IF(ILAMEXP<NLAMEXP)FACT2=LAMEXP*(ILAMEXP+1)/NLAMEXP
       FACT4=HALF*(FACT3+(ONE-EXP(-FACT2*FACT2))/FACT1)
     ELSE
       FACT1=ONE-EXP(-LAMEXP*LAMEXP)
       FACT2=LAMEXP*(NLAMEXP-ILAMEXP)/NLAMEXP
       FACT3=(EXP(-FACT2*FACT2)+FACT1-ONE)/FACT1
       IF(ILAMEXP<NLAMEXP)FACT2=LAMEXP*(NLAMEXP-ILAMEXP-1)/NLAMEXP
       FACT4=HALF*(FACT3+(EXP(-FACT2*FACT2)+FACT1-ONE)/FACT1)
     ENDIF
     LAMDAI=LAMDAF
     !LAMDAP=LAMSTA+FACT3*(LAMSTOP-LAMSTA)
     LAMDAF=LAMSTA+FACT4*(LAMSTOP-LAMSTA)
     LAMDAP=HALF*(LAMDAI+LAMDAF)
     IF(ILAMEXP==0)LAMDAP=LAMDAI
     IF(ILAMEXP==NLAMEXP)LAMDAP=LAMDAF
     IF(.NOT.QPWIND) LAMDAP=LAMDAI
     ILAMEXP=ILAMEXP+1
    ENDIF 
  ELSE
     LAMDAI=LAMSTA
  !
     IF(ABS(LAMDAI-LAMDAF) > TENM5) THEN
          IF(WRNLEV >= 2 .and. prnlev > 2) WRITE(OUTU,222) LAMDAI,LAMDAF
222  FORMAT(' PERTPS> ::WARNING:: The current LSTART value',F10.5, &
               '  does not match the previous LSTOP value',F10.5)
       ENDIF
     
     LAMDAF=LAMSTOP
  ENDIF
  IF(QPWIND) THEN
     IF((LAMDAI-LAMDAP)*(LAMDAF-LAMDAP) > ZERO) THEN
        IF(WRNLEV >= 2 .and. prnlev > 2) WRITE(OUTU,233) LAMDAP,LAMDAI,LAMDAF
233     FORMAT(' PERTPS> ::WARNING:: LAMBDA value of',F12.6, &
             ' is not between LSTART and LSTOP',2F12.6)
        CALL WRNDIE(-2,'<PERTPS>','Bad LAMBDA value')
     ENDIF
     IF(LAMINC*(LAMSTOP-LAMSTA) < ZERO) THEN
        IF(WRNLEV >= 2 .and. prnlev > 2) WRITE(OUTU,234) LAMINC,LAMSTA,LAMSTOP
234     FORMAT(' PERTPS> ::WARNING:: LAMINC value of',F12.6, &
             ' is not match LSTART and LSTOP',2F12.6)
        CALL WRNDIE(-2,'<PERTPS>','Bad LAMINC value')
     ENDIF
  ENDIF
  !
  ! Set value limits
  IF(LAMDAI < ZERO) LAMDAI=ZERO
  IF(LAMDAP < ZERO) LAMDAP=ZERO
  IF(LAMDAF < ZERO) LAMDAF=ZERO
  IF(LAMDAI > ONE)  LAMDAI=ONE
  IF(LAMDAP > ONE)  LAMDAP=ONE
  IF(LAMDAF > ONE)  LAMDAF=ONE
  !
  ! set other values for initial conditions
  IPNTOT=0
  LAMDA=LAMDAP
  LAMDAM=ONE-LAMDA
  LAMDEL=LAMDAF-LAMDAI
  IF(FINALT > ZERO) THEN
     !br      DELDRT=-LAMDEL/(FINALT*KBOLTZ)
     DELDRT=-ONE/(FINALT*KBOLTZ)
  ELSE
     DELDRT=ZERO
  ENDIF
  IF(.NOT.QPWIND) THEN
     IF(IPSTP > IPSTRT) THEN
        LAMDEL=LAMDEL/(IPSTP-IPSTRT)
     ELSE
        LAMDEL=ZERO
     ENDIF
  ENDIF
  !
  ! Print out current status
  IF(PRNLEV >= 2) THEN
     IF(QRESET) THEN
        WRITE(OUTU,25) &
             'Free energy perturbation calculation initiated.'
        IF(NLAMEXP>0)THEN
          WRITE(OUTU,28) &
        'Number of LAMDA windows are automatically generated: ', NLAMEXP
        ELSE
          WRITE(OUTU,28) &
        'LAMDA windows are readin from UNIT: ', PUNIT
        ENDIF
     ELSE
        WRITE(OUTU,25) &
             'Free energy perturbation calculation continues.'
     ENDIF
25   FORMAT(' PERTURBATION> ',A)
     WRITE(OUTU,26) IPSTRT,IPSTP
26   FORMAT(' PERTURBATION> PSTART=',I12,'  PSTOP=',I12)
     WRITE(OUTU,27) LAMDAI,LAMDAF,LAMDAP
27   FORMAT(' PERTURBATION> LSTART=',F12.6,'  LSTOP=',F12.6, &
          '  LAMBDA=',F12.6)
28   FORMAT(' PERTURBATION> ',A,I8)
     IF(QPWIND) THEN
        WRITE(OUTU,25) 'Windowing will be used.'
     ELSE
        WRITE(OUTU,25) 'Slow growth will be used.'
     ENDIF
  ENDIF
  !
  ! reset accumulation variables and arrays
  EPDIFF=ZERO
  EPTOT1F=ZERO
  EPTOT1B=ZERO
  EPTOT2F=ZERO
  EPTOT2B=ZERO
  EPTOT3=ZERO
  EPTOT4=ZERO
  EPREF =ZERO
  EPREFF=ZERO
  EPREFB=ZERO
  EPSSBPLRC=ZERO
  EPSSBPLRCSQ=ZERO
  !
  EPPRTL(:) = ZERO   
  ETPRTL(:) = ZERO
  EVPRTL(:) = ZERO

  EPPRTM(:) = ZERO
  ETPRTM(:) = ZERO
  EVPRTM(:) = ZERO

  EPPRTD(:) = ZERO
  ETPRTD(:) = ZERO
  EVPRTD(:) = ZERO

  EPPRTA(:) = ZERO
  ETPRTA(:) = ZERO
  EVPRTA(:) = ZERO

  EPPRTF(:) = ZERO
  ETPRTF(:) = ZERO
  EVPRTF(:) = ZERO
  !
  EPSSLJ=ZERO
  EPSSCO=ZERO
  ELJDLM=ZERO
  ECODLM=ZERO
  ELJDLL=ZERO
  ECODLL=ZERO
  !
  IF(QRESET) THEN
     IPNCAL=0
     PERTEPC(1:NATOMT)=zero
  ENDIF
  !
  IF(QECONT) THEN
     PERTEPC(1:NATOMT)=zero
     PERTEP1(1:NATOMT)=zero
     PERTEP2(1:NATOMT)=zero
  ENDIF
  !
  RETURN
END SUBROUTINE PERTPS

SUBROUTINE PERTAN(QPARSE)
  !
  ! THIS ROUTINE ANALYZES AND PRINTS THE RESULTS AT THE
  ! END OF A SIMULATION.
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use pert
  implicit none
  !
  LOGICAL QPARSE
  !
  !
  CALL PERTAN2(QPARSE,PERTEPC,PERTEP1,PERTEP2)
  !
  RETURN
END SUBROUTINE PERTAN
!
SUBROUTINE PERTAN2(QPARSE,EPCONT,EPCON1,EPCON2)
  !
  ! THIS ROUTINE ANALYZES AND PRINTS THE RESULTS AT THE
  ! END OF A SIMULATION.
  !
  !  BERNARD R. BROOKS, NIH, DECEMBER,1987
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use psf
  use energym
  use econtmod
  use epert_mod
  use pert
  use stream
  use reawri
  use consta
  use comand
  use param_store, only: set_param

  implicit none
  !
  LOGICAL QPARSE
  real(chm_real) EPCONT(*),EPCON1(*),EPCON2(*)
  !
  !
  INTEGER I
  real(chm_real) ETEMP,LRCVDWC
  real(chm_real) ELJTMP,ECOTMP
  LOGICAL EOF
  !
  ! SSBP LONG RANGE CORRECTION
  IF(IPNTOT <= 0) GOTO 800
  LRCVDWC=4.0*3.14159265358979324*0.033427699/3.0
  EPSSBPLRC=EPSSBPLRC*LRCVDWC/IPNTOT
  EPSSBPLRCSQ=EPSSBPLRCSQ*LRCVDWC*LRCVDWC
  EPSSBPLRCSQ=SQRT(EPSSBPLRCSQ-IPNTOT*EPSSBPLRC*EPSSBPLRC)/ &
       (IPNTOT-1.0)
  call set_param('SSBPLRC',EPSSBPLRC)
  !sbbug      call set_param('SSBPLRCSD',EPSSBPLRCSQ)
  !     variable names must not be longer than 8 characters
  call set_param('SSBPLRCS',EPSSBPLRCSQ)


  IF(PRNLEV >= 2) WRITE(OUTU,25) 'Free energy perturbation results:'
25 FORMAT(' PERTURBATION> ',A)
  IF(PRNLEV >= 2) WRITE(OUTU,26) LAMDAI,LAMDAF,LAMDA,IPNTOT
26 FORMAT(' PERTURBATION> results, LSTART=',F12.6, &
       '  LSTOP=',F12.6,'  LLAST=',F12.6, &
       ' Number of steps used=',I10)
  !
  EPTOT1F=EPTOT1F/IPNTOT
  EPTOT1B=EPTOT1B/IPNTOT
  EPTOT2F=EPTOT2F/IPNTOT
  EPTOT2B=EPTOT2B/IPNTOT
  EPTOT2F=EPTOT2F-EPTOT1F**2
  EPTOT2B=EPTOT2B-EPTOT1B**2
  IF(EPTOT2F > 0.0) THEN
     EPTOT2F=SQRT(EPTOT2F)
  ELSE
     EPTOT2F=0.0
  ENDIF
  IF(EPTOT2B > 0.0) THEN
     EPTOT2B=SQRT(EPTOT2B)
  ELSE
     EPTOT2B=0.0
  ENDIF
  !
  EPTOT3=EPTOT3/IPNTOT
  EPTOT4=EPTOT4/IPNTOT
  EPTOT4=EPTOT4-EPTOT3**2
  IF(EPTOT4 > 0.0) THEN
     EPTOT4=SQRT(EPTOT4)
  ELSE
     EPTOT4=0.0
  ENDIF
  EPTOT3=EPTOT3*(LAMDAF-LAMDAI)
  EPTOT4=EPTOT4*ABS(LAMDAF-LAMDAI)
  !
  IF(PRNLEV >= 2) WRITE(OUTU,33) EPTOT1F,EPTOT1B, &
       EPTOT2F,EPTOT2B,EPTOT3,EPTOT4
33 FORMAT(' PERTURBATION> result: EXPAVE=',E12.6,1X,E12.6,' EXPFLC=', &
       E12.6,1X,E12.6,' DIFAVE=',F12.6,' DIFFLC=',F12.6)
  !
  DO I=1,LENENP
     EPPRTA(I) = (EPPRTA(I)/IPNTOT)
     EPPRTF(I) = (EPPRTF(I)/IPNTOT- EPPRTA(I)**2)*(LAMDAF-LAMDAI)**2
     EPPRTA(I) = EPPRTA(I)*(LAMDAF-LAMDAI)
     IF(EPPRTF(I) > ZERO) THEN
        EPPRTF(I) = SQRT(EPPRTF(I))
     ELSE
        EPPRTF(I)=ZERO
     ENDIF
  ENDDO
  DO I=1,LENENT
     ETPRTA(I) = (ETPRTA(I)/IPNTOT)
     ETPRTF(I) = (ETPRTF(I)/IPNTOT- ETPRTA(I)**2)*(LAMDAF-LAMDAI)**2
     ETPRTA(I) = ETPRTA(I)*(LAMDAF-LAMDAI)
     IF(ETPRTF(I) > ZERO) THEN
        ETPRTF(I) = SQRT(ETPRTF(I))
     ELSE
        ETPRTF(I)=ZERO
     ENDIF
  ENDDO
  DO I=1,LENENV
     EVPRTA(I) = (EVPRTA(I)/IPNTOT)
     EVPRTF(I) = (EVPRTF(I)/IPNTOT- EVPRTA(I)**2)*(LAMDAF-LAMDAI)**2
     EVPRTA(I) = EVPRTA(I)*(LAMDAF-LAMDAI)
     IF(EVPRTF(I) > ZERO) THEN
        EVPRTF(I) = SQRT(EVPRTF(I))
     ELSE
        EVPRTF(I)=ZERO
     ENDIF
  ENDDO
  !
  IF(QECONT) THEN
     DO I=1,NATOMT
        EPCON1(I) = (EPCON1(I)/IPNTOT)
        EPCON2(I) = (EPCON2(I)/IPNTOT - &
             EPCON1(I)**2)*(LAMDAF-LAMDAI)**2
        EPCON1(I) = EPCON1(I)*(LAMDAF-LAMDAI)
        EPCONT(I) = EPCONT(I)+EPCON1(I)
     ENDDO
  ENDIF
  !
  IF(QPWIND) THEN
     ! Window method
     !        IF(EPTOT1 > 0.0) THEN
     !           ETEMP=-(FINALT*KBOLTZ)*LOG(EPTOT1)
     !        ELSE
     !           ETEMP=1000.0
     !        ENDIF
     !        ETEMP=ETEMP+EPREF*(LAMDAF-LAMDAI)
     EPTOT1F = -(FINALT*KBOLTZ)*LOG(EPTOT1F)+EPREFF
     EPTOT1B = -(FINALT*KBOLTZ)*LOG(EPTOT1B)+EPREFB
     ETEMP = EPTOT1F-EPTOT1B
     EPRTTW=EPRTTW+ETEMP
     call set_param('TPDEL',ETEMP)
     call set_param('TPTOT',EPRTTW)
     IF(PRNLEV >= 2) THEN
        WRITE(OUTU,37) EPRTTW,EPTOT1F,EPTOT1B,EPREFF,EPREFB
37      FORMAT(' PERTURBATION> TP Windowing result, EPRTOT=',F12.6, &
             ' EFORWARD=',F12.6,' EBACKWARD=',F12.6, &
             ' EPREFF=',F12.6,' EPREFB=',F12.6)
     ENDIF

     !        IF(PRNLEV >= 2) WRITE(OUTU,34) EPRTTW,ETEMP,
     !    &                             EPREF*(LAMDAF-LAMDAI)
     ! 34     FORMAT(' PERTURBATION> TP Windowing result, EPRTOT=',F12.6,
     !    &          '  EFORWARD=',F12.6,' EPREF=',F12.6)

     !
     ETEMP=EPTOT3+EPREF*(LAMDAF-LAMDAI)
     EPRTOT=EPRTOT+ETEMP
     call set_param('TIDEL',ETEMP)
     call set_param('TITOT',EPRTOT)
     IF(PRNLEV >= 2) WRITE(OUTU,35) EPRTOT,ETEMP, &
          EPREF*(LAMDAF-LAMDAI)
35   FORMAT(' PERTURBATION> TI Windowing result, EPRTOT=',F12.6, &
          '  EFORWARD=',F12.6,' EPREF=',F12.6)
     EPPRTA(TOTE)=ETEMP
  ELSE
     ! Slow growth method
     ETEMP=EPTOT3+EPREF*(LAMDAF-LAMDAI)
     ETEMP=ETEMP+EPPRTA(TOTKE)
     EPRTOT=EPRTOT+ETEMP
     call set_param('SLDEL',ETEMP)
     call set_param('SLTOT',EPRTOT)
     IF(PRNLEV >= 2) WRITE(OUTU,39) EPRTOT,ETEMP, &
          EPREF*(LAMDAF-LAMDAI)
39   FORMAT(' PERTURBATION> Slow growth result, EPRTOT=',F12.6, &
          '  EFORWARD=',F12.6,' EPREF=',F12.6)
  ENDIF
  !
  IF(QPSSP) THEN
     !     the following is somewhat redundant, but it allows to separate
     !     and print out the complicated dU/dl parts at high print levels
     !     useful for debugging and needed in one test case!
     ELJTMP=EPSSLJ*(LAMDAF-LAMDAI)/IPNTOT
     ECOTMP=EPSSCO*(LAMDAF-LAMDAI)/IPNTOT
     DLJTOT=DLJTOT+ELJTMP
     DCOTOT=DCOTOT+ECOTMP
     !
     IF(PRNLEV >= 2) WRITE(OUTU,40) LAMDAI,LAMDAF,LAMDA,IPNTOT
     IF(PRNLEV >= 2) WRITE(OUTU,41) ELJTMP,ECOTMP
     IF(PRNLEV >= 2) WRITE(OUTU,42) DLJTOT,DCOTOT
     !
40   FORMAT(' PSSP> results, LSTART=',F8.4, &
          '  LSTOP=',F8.4,'  LLAST=',F8.4, &
          ' NSTEPS= ',I8)
41   FORMAT(' PSSP> DLJDLA=',F12.6,', DCODLA=',F12.6)
42   FORMAT(' PSSP> DLJTOT=',F12.6,', DCOTOT=',F12.6)
     !
     EPSSLJ=ZERO
     EPSSCO=ZERO
  ENDIF
  !
  IF(QLONGL) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,43) LAMDAI,LAMDAF,EPRTOT,ETEMP, &
          EPREF*(LAMDAF-LAMDAI),EPTOT3,EPTOT4
  ELSE
     IF(PRNLEV >= 2) WRITE(OUTU,44) LAMDAI,LAMDAF,EPRTOT,ETEMP, &
          EPREF*(LAMDAF-LAMDAI),EPTOT3,EPTOT4
  ENDIF
43 FORMAT(' PERTRES> LSTART=',F12.5,' LSTOP=',F12.5, &
       ' EPRTOT=',F12.5,'  EFORWARD=',F12.5,' EPREF=', &
       F12.5,' DIFAVE=',F12.5,' DIFFLC=',F12.5)
  !mu...25-Jul-93, M.E. Karpen, to avoid record overflow
44 FORMAT(' PERTRES> LSTART  =',F12.5,' LSTOP=',F12.5, &
       ' EPRTOT=',F12.5,/,10x,'EFORWARD=',F12.5,' EPREF=', &
       F12.5,' DIFAVE=',F12.5/10x,'DIFFLC  =',F12.5)
  !...  ln020628
  call set_param('DFLC',EPTOT4)
  !
  EPPRTT(:) = EPPRTT(:)+EPPRTA(:)
  ETPRTT(:) = ETPRTT(:)+ETPRTA(:)
  EVPRTT(:) = EVPRTT(:)+EVPRTA(:)
  ! . Print out the results.
  IF(PRNLEV >= 2) THEN
     IF(PRNLEV >= 2) WRITE (OUTU,'(A,I8,A)') &
          ' PERTURBATION> Averages for the last ', IPNTOT,'  steps:'
     CALL PRINTE(OUTU, EPPRTA, ETPRTA, 'PAVE', 'DYN', .TRUE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
     IF(PRNLEV >= 2) WRITE (OUTU,'(A,I8,A)') &
          ' PERTURBATION> Fluctuations for the last ', IPNTOT,'  steps:'
     CALL PRINTE(OUTU, EPPRTF, ETPRTF, 'PFLC', 'DYN', .FALSE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !...  ln020628
  call set_param('AVKE',EPPRTA(2))
  call set_param('KEFL',EPPRTF(2))
  !
800 CONTINUE
  IF(PRNLEV >= 2) THEN
     IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
          ' PERTURBATION> TOTALS since last reset:'
     CALL PRINTE(OUTU, EPPRTT, ETPRTT, 'PTOT', 'DYN', .FALSE., &
          IPNTOT, ZERO, ZERO, .TRUE.)
  ENDIF
  !
  IF(.NOT.QPARSE) RETURN
  !-----------------------------------------------------------------------
  ! Parse next phase of free energy run.
  IF(PUNIT > 0) THEN
     EOF=.FALSE.
     COMLEN=0
     DO WHILE(COMLEN == 0 .AND. .NOT.EOF)
        CALL RDCMND(COMLYN,MXCMSZ,COMLEN,PUNIT,EOF,.TRUE., &
             .TRUE.,'PUNIT> ')
     ENDDO
     IF(EOF) THEN
        ! end of commands, turn off perturbation.
        IF(PRNLEV >= 2) WRITE(OUTU,'(A)') &
             ' PERTURBATION> EOF on punit file: PERT in auto mode.'
        PUNIT=-1
     ENDIF
  ELSE
     COMLYN=' '
     COMLEN=0
  ENDIF
  CALL PERTPS(COMLYN,COMLEN,.FALSE.)
  RETURN
END SUBROUTINE PERTAN2
!
!sbcp the ugly construct serves to keep the original preprocessor
!     logic as much as possible; i.e., this hopefully compiles
!     meaningfully for all combinations of <nothing>, PERT, PERT + CHEMPERT
#if KEY_CHEMPERT==1 /*chempert*/
!sb   Aux. routine for chem pert with Ewald
subroutine inichem(cg,cgtot,cgl,cglt,natom,pertid)
  !
  !     Copy charges and set all charges not affected by PERT to zero
  !
  use chm_kinds
  use pert
  implicit none
  !
  real(chm_real) cg(*),cgl(*)
  real(chm_real) cgtot,cglt
  integer pertid(*)
  integer natom

  integer i

  cglt=0.0d0

  do i=1,natom
     !     make array of ligand charges, all others are zero
     cgl(i)=cg(i)*pertid(i)
     !     sum charges of ligand
     cglt=cglt+cgl(i)
  enddo
  return
end subroutine inichem

#else /* (chempert)*/
subroutine inichem
  return
end subroutine inichem
#endif /* (chempert)*/

subroutine pert_off
  use chm_kinds
  use pert
  use memory
  use bases_fcm
  use number
  use stream
  use datstr,only:dupldt_nbond,freedt_nbond,freedt_image
  implicit none

  QPERT=.FALSE.

#if KEY_WCA==1
  ! shift soft-core potential parameters
  LSOFTCORE0 = .FALSE.
  LSOFTCORE1 = .FALSE.
  SCCUTR0 = ONE
  SCCUTR1 = ONE
#endif 

  QPSSP=.FALSE.
  DLAMBD=FIVE
  ALAMBD=FIVE
#if KEY_CHEMPERT==1
  if (qchemp) then
     call chmdealloc('pert.src','PERTS','cglig',maxaim,crl=cglig)
     qchemp=.false.
  endif
#endif 
  if(allocated(ppiatom)) &
       call chmdealloc('pert.src','PERTS','PPIATOM',NIPERT,intg=PPIATOM)
  IF(PRNLEV >= 2) WRITE(OUTU,113)
113 FORMAT(' PERT: Free energy calculations turned off.')
  !        Free some space (if used)
  CALL FREEDT_nbond(BNBNDR)
  CALL FREEDT_nbond(BNBNDP)
  CALL FREEDT_image(BIMAGR)
  CALL FREEDT_image(BIMAGP)

  CALL CLEARPERT

  !ML-----------------------------------------------------------------
  ! Free space of lambda-dependent MMFP-term

  IF(QMMFPE.AND.QZEGEO) THEN
     IF(PRNLEV >= 2) WRITE(OUTU,114)
114  FORMAT(' PERT: Deactivate alchemical MMFP data structure')

     call chmdealloc('pert.src','PERTS','PLSGEO',PMXGEO,intg=PLSGEO)
     call chmdealloc('pert.src','PERTS','PNGEO',PMXGEO,intg=PNGEO)
     call chmdealloc('pert.src','PERTS','PIGEO',PMXGEO,intg=PIGEO)
     call chmdealloc('pert.src','PERTS','PJGEO',PMXGEO,intg=PJGEO)
     call chmdealloc('pert.src','PERTS','PXRGEO',PMXGEO, crl=PXRGEO)
     call chmdealloc('pert.src','PERTS','PYRGEO',PMXGEO, crl=PYRGEO)
     call chmdealloc('pert.src','PERTS','PZRGEO',PMXGEO, crl=PZRGEO)
     call chmdealloc('pert.src','PERTS','PTRGEO',PMXGEO, crl=PTRGEO)
     call chmdealloc('pert.src','PERTS','PXDGEO',PMXGEO, crl=PXDGEO)
     call chmdealloc('pert.src','PERTS','PYDGEO',PMXGEO, crl=PYDGEO)
     call chmdealloc('pert.src','PERTS','PZDGEO',PMXGEO, crl=PZDGEO)
     call chmdealloc('pert.src','PERTS','PDRGEO',PMXGEO, crl=PDRGEO)
     call chmdealloc('pert.src','PERTS','PDTGEO',PMXGEO, crl=PDTGEO)
     call chmdealloc('pert.src','PERTS','PFCGEO',PMXGEO, crl=PFCGEO)
     call chmdealloc('pert.src','PERTS','PP1GEO',PMXGEO, crl=PP1GEO)
     call chmdealloc('pert.src','PERTS','PP2GEO',PMXGEO, crl=PP2GEO)
     call chmdealloc('pert.src','PERTS','PP3GEO',PMXGEO, crl=PP3GEO)
     call chmdealloc('pert.src','PERTS','PIUGEO',PMXGEO,intg=PIUGEO)

  ENDIF

  QMMFPE=.FALSE.
  QZEGEO=.FALSE.
  !ML-----------------------------------------------------------------
  RETURN
end subroutine pert_off

#else /* (pert1)*/

SUBROUTINE PERTS(COMLYN,COMLEN)
  use stream
  INTEGER COMLEN
  character(len=*) COMLYN
  call wrndie(-2,"Pert not compiled","")
  return
end SUBROUTINE PERTS

#endif /*  (pert1)*/

end module pert_mod

