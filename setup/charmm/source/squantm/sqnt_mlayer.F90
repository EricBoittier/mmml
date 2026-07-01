#if KEY_SQUANTM==1 /*mainsquatn*/

SUBROUTINE Setup_option_Mlayer(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Initial setup options for Multi-layered QM/MM.
  !
  !     Kwangho Nam
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !
  use gamess_fcm
  use squantm
#if KEY_PARALLEL==1
  use parallel  
#endif
#if KEY_FLUCQ==1
  use flucq     
#endif
  use stream
  use string
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
  use replica_mod
  use block_fcm
#endif 

  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN

#if KEY_GAMESS==1 || KEY_GAMESSUK==1
  character(len=240)   name
  integer kstrm,koutu
  COMMON /CHGMIO/  KSTRM,KOUTU,name

  LOGICAL   QQINP
  INTEGER KFRAG,JFRAG
  !CCC  test code...
  logical mopac
  common /mpctst/mopac
  !
  character(len=200) tmpqms
#if KEY_REPLICA==1 /*replica*/
  INTEGER I,J,IPT,IREPL
#endif /* (replica)*/
#if KEY_GAMESSUK==1
  INTEGER INIT, ICODE,IVER     
#endif
#if KEY_GAMESSUK==1
  LOGICAL STARTUP              
#endif
  !
  CHARACTER(len=10) CFN
  CHARACTER(len=6) CRN
  INTEGER NQMPMEt1,NQMPMEt2,XI,YI,ZI
  LOGICAL QDONE

#endif 
  !
  !
#if KEY_FLUCQ==1
  IF(QFLUC) CALL WRNDIE(-2,'<MLAYer>', &
       'Current MLAYer QM/MM is not tested with FLUCQ.')
#endif 
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 /*guk_main*/
#if KEY_BLOCK==1 /*block*/
  IF(QBLOCK) CALL WRNDIE(-5,'<MLAYer>', &
       'BLOCK is not supported currenlty in MLAYer QM/MM.')
#endif /*  (block)*/
#if KEY_REPLICA==1
  IF(QREP) CALL WRNDIE(-5,'<MLAYer>', &
       'MLAYer QM/MM do not work with Replicas.')
#endif 

  CFN='gukini.src'
  CRN='GUKINI'
  !C
  !C  Disable interface to switch back to MM only
  !C  Does not restore MM charges to QM atoms
  !C
  !      IF(INDXA(COMLYN,COMLEN,'OFF').GT.0)THEN
  !         IF(PRNLEV.GE.2) THEN
  !            WRITE(OUTU,24)'OFF:  GAMESS Interface Disabled'
  !         ENDIF
  !c         CALL RSTCOPSEL
  !         QGMREM=.FALSE.
  !         NGAMES=0
  !         RETURN
  !      ENDIF
  !
  !    other options in gamess/gamess-uk

  QBLUCH=(INDXA(COMLYN,COMLEN,'BLUR').GT.0)
  IF(QBLUCH) THEN
     IF(PRNLEV.GE.2) WRITE(6,24) &
          ' BLUR: Blurred charges will be used on some atoms.'
  ENDIF
  IF(QBLUCH) CALL WRNDIE(-1,'<MLAYer>', &
       'MLAYer QM/MM is not tested with BLUR.')
  !
  QCUTFLAG=(INDXA(COMLYN,COMLEN,'CUTO').GT.0)
  IF(QCUTFLAG) THEN
     IF(PRNLEV.GE.2) WRITE(OUTU,24) &
          'CUTOFF: the cutoff method from NBOND will be used.'
  ENDIF
  !
  !
#if KEY_GAMESS==1
  QNOGU=(INDXA(COMLYN,COMLEN,'NOGU').GT.0)
  IF(QNOGU) THEN
     KGUES=0
     IF(PRNLEV.GE.2) WRITE(6,24) &
          ' NOGUess: Initial guess obtained from previous step.'
  ENDIF
  !
  QFMO=(INDXA(COMLYN,COMLEN,'FMO').GT.0)
  IF(QFMO) THEN
     CALL WRNDIE(-5,'<MLAYer>', &
          'FMO: Fragment MO method is not supported.')
     QFMO=.FALSE.
  ENDIF
  !
  KDIESL=GTRMI(COMLYN,COMLEN,'DIES',-1)
  IF(KDIESL.GE.0) THEN
     CALL WRNDIE(-5,'<MLAYer>', &
          'DIESel: Multi reference CI is not supported.')
     KDIESL=-1
  END IF
  !
#endif 
  NFRAG=GTRMI(COMLYN,COMLEN,'FRAG',0)
  IF(NFRAG.NE.0)THEN
     CALL WRNDIE(-5,'<MLAYer>', &
          'NFRAG is not supported with MLAYer QM/MM.')
     QFRAG=.FALSE.
  END IF
  MOPAC=(INDXA(COMLYN,COMLEN,'MOPAC').GT.0)
  IF(MOPAC) THEN
     CALL WRNDIE(-1,'<MLAYer>', &
          'MOPAC calculation is not supported with MLAYer QM/MM.')
     MOPAC=.FALSE.
  END IF
#endif /*  (guk_main)*/
  !
  !     default HRCUT=10.0D0
  HRCUT = GTRMF(COMLYN,COMLEN,'RCUT',10.0D0)
  IF(PRNLEV.GE.2) WRITE(6,26) &
       ' High Level Electrostatic Cutoff distance is set to', HRCUT

  !
  !     How to handle charge? QINP=.TRUE. use input MM charges
  !     for the charges on Low QM in High QM calculaitons
  QINP=.TRUE.
  !     QINP=(INDXA(COMLYN,COMLEN,'QINP').GT.0)
  IF(QINP.AND.PRNLEV.GE.2) WRITE(6,24) &
       ' Input MM charge will be used for High QM calculation.'

  !     Read the NSTEP for High level calculation during dynamics
  HIGHL =.FALSE.
  NHSTP = GTRMI(COMLYN,COMLEN,'NSTE',1)
  NMDSTP= NHSTP-1
  If (NHSTP.LE.0) then
     CALL WRNDIE(0,'<MLAYer>', &
          'Specify NSTEP for High Level calculation during MD.')
     IF(PRNLEV.GE.2) WRITE(6,28) &
          ' Default NSTEP for High Level during MD:',NHSTP
  Else if(NHSTP.gt.1) then
     IF(PRNLEV.GE.2) WRITE(6,30) &
          ' High Level calculation will be updated at every ', &
          NHSTP,'-th step during MD.' 
  End if
  !
  !     Read the LPLEV option to handle the force on H-link atom.
  LPLEV= GTRMI(COMLYN,COMLEN,'LPLE',1)
  IF(QH_Link) THEN
     IF(LPLEV.EQ.1) THEN
        IF(PRNLEV.GE.2) WRITE(6,24) &
             ' The force along H-QM atom will be projected out.'
     ELSE IF(LPLEV.EQ.2) THEN
        IF(PRNLEV.GE.2) WRITE(6,24) &
             ' The force on H-link atom will be ignored.'
     ELSE
        CALL WRNDIE(-1,'<MLAYer>', &
             'The default LPLEV will be used.')
        IF(PRNLEV.GE.2) WRITE(6,24) &
             ' The force along H-QM atom will be projected out.'
        LPLEV = 1
     END IF
  END IF
  !
24 FORMAT('MLAYer>',A)
26 FORMAT('MLAYer>',A,F8.4)
28 FORMAT('MLAYer>',A,I5)
30 FORMAT('MLAYer>',A,I5,A) 

  return
END SUBROUTINE Setup_option_Mlayer

SUBROUTINE Setup_high_qm
  !-----------------------------------------------------------------------
  !     Initial setup options for Multi-layered QM/MM.
  !
  !     Kwangho Nam
  !
  use chm_kinds
  use dimens_fcm
  !

  use gamess_fcm
  use squantm
#if KEY_PARALLEL==1
  use parallel  
#endif
  !
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
  !cc  use coord
  !cc  use replica_mod
  !cc  use block_fcm
#endif 

  implicit none

#if KEY_GAMESS==1 || KEY_GAMESSUK==1 /*gamess_part*/
  character(len=240)   name
  integer kstrm,koutu
  COMMON /CHGMIO/  KSTRM,KOUTU,name

  INTEGER KFRAG,JFRAG
  !
  character(len=200) tmpqms
#if KEY_GAMESSUK==1
  INTEGER INIT, ICODE,IVER     
#endif
#if KEY_GAMESSUK==1
  LOGICAL STARTUP              
#endif

  CHARACTER(len=10) CFN
  CHARACTER(len=6) CRN
  INTEGER NQMPMEt1,NQMPMEt2
  LOGICAL QDONE
  INTEGER I,JIATOM
#endif /*   (gamess_part)*/

  !cc##IF GAMESS GAMESSUK (rpath)
  !cc##IF REPLICA
  !cc##IF RPATH
  !ccc defaults for QM replica loop structure
  !cc      NPGRP      = 1                 ! number of parallel groups
  !cc      NREPQMNOD  = 1                 ! number of replicas on this node
  !cc
  !cc##IF PARALLEL (parallel)
  !ccc     save global node values
  !cc      QQMPAR=.FALSE.
  !cc      IF(NUMNOD.GT.1)QQMPAR=.TRUE.
  !ccC
  !cc##ENDIF (parallel)
  !cc##ENDIF
  !cc##ENDIF
  !
  !cc##IF REPLICA (replica)
  !cc##IF RPATH
  !cc##IF PARALLEL
  !ccC Save to be used
  !cc      NUMNOD_save=NUMNOD
  !cc      MYNOD_save=MYNOD
  !cc##ENDIF
  !cc##ENDIF 
  !cc##ENDIF (replica)
  !cc##ENDIF (rpath)

  !
  !     This initialize gamess data
  QINIGM=.TRUE.

#if KEY_GAMESS==1
  CALL CH2GMS(.TRUE.)
  CALL GAMESS
#endif 
  !
#if KEY_GAMESSUK==1 /*guk*/
  INIT=1
  STARTUP = .true.
  !cc##IF REPLICA
  !cc##IF RPATH
  !ccc
  !cc      do IREPL = IREPQM, IREPQM + NREPQMNOD - 1
  !ccC     Setup environment variables to separate group's I/O
  !cc      CALL ENVIGRP(IREPL)
  !cc##ENDIF
  !cc##ENDIF
  iver = 5
  CALL GAMESS(INIT,ICODE,STARTUP,LQMEWD,IVER)
  if (IVER.NE.-5) then
     write(6,*)iver
     CALL WRNDIE(-5,'<GAMESS-UK>','Code Version Mismatch')
  endif
  !cc##IF REPLICA
  !cc##IF RPATH
  !cc      enddo
  !cc##ENDIF
  !cc##ENDIF

#endif /* (guk)*/
  !
  !     Report QM/MM repulsion energy, also when no derivatives involved
#if KEY_GAMESS==1
  IF(PRNLEV.GE.2) CALL CGREPE(NATOM) 
#endif

  return
END SUBROUTINE Setup_high_qm


SUBROUTINE HighQM_ene_mlayer(NATOMX,NATQM_2,qminb2_dual, &
#if KEY_GAMESS==1
     mminb1_dual,               & 
#endif
     GTOT,X,Y,Z,DX,DY,DZ,CGX)
  !-----------------------------------------------------------------------
  !
  !     Get energy and forces from GAMESS,GAMESS-UK
  !
  use chm_kinds
  use dimens_fcm
  use number
  use consta
  use exfunc
  !
  !
  use gamess_fcm
  use stream
  use parallel
  use mndo97

  !cc  use replica_mod
  !cc  use pathm
  !cc  use block_fcm
  !cc  use heapm
  !cc  use scalar_module
  !ccC Add quantum energy to econt array for NEB usage ! jwchu
  !cc  use econtmod                ! jwchu

  implicit none
  INTEGER NATOMX,NATQM_2,QMINB2_dual(*)
#if KEY_GAMESS==1
  INTEGER mminb1_dual(MAXA,2)                       
#endif
  real(chm_real) GTOT,X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),CGX(*)
  !
#if KEY_GAMESSUK==1
  INTEGER INIT,ICODE,IVER,N1   
#endif
#if KEY_GAMESS==1
  INTEGER N1                   
#endif
#if KEY_GAMESSUK==1
  LOGICAL STARTUP              
#endif
#if KEY_GAMESS==1 /*gamess*/
  INTEGER ICHARM
  !
  real(chm_real) E, EG
  COMMON /FUNCT/ E, EG(3*MAXGMS)
  !
  INTEGER NAT,ICH,MUL,NUM,NX,NE,NA,NB,IAN
  real(chm_real) ZAN, C
  COMMON /INFOA / NAT,ICH,MUL,NUM,NX,NE,NA,NB, &
       ZAN(MAXGMS),C(3,MAXGMS),IAN(MAXGMS)
  !
  real(chm_real) HMLTN
  LOGICAL MPCWFN
  COMMON /MPCLNK/ HMLTN,MPCWFN
  !
  real(chm_real) QMMMRP, RBR, EDIESL
  INTEGER IPT,N
  logical mopac
#if KEY_REPLICA==0
  real(chm_real) BLFACTOR    
#endif
#endif /*  (gamess)*/
  !
  LOGICAL QDONE
  CHARACTER(len=10) CFN
  CHARACTER(len=6) CRN
  INTEGER I,IATOM,MM
#if KEY_GAMESSUK==1
  CHARACTER(len=3) TMPSTR        
#endif
#if KEY_GAMESSUK==1
  CHARACTER(len=40) MSG          
#endif
  real(chm_real) GTOTOLD,ECURRENT
#if KEY_GAMESSUK==1
  real(chm_real) GTOT_local(2)      
#endif

  LOGICAL LQMEWD

#if KEY_BLOCK==1
  !cc      INTEGER IBL,IBLQM,KK                    
#endif
#if KEY_PARALLEL==1
  !cc      INTEGER MYNODL            
#endif
#if KEY_REPLICA==1 || KEY_RPATH==1
  !cc      INTEGER IREPL             
#endif
  !
  !     Are there any QM atoms?
  !
  IF(NGAMES.EQ.0) RETURN
  !
  ! some initialization.
  CFN='gukini.src'
  CRN='GUKENE'
  BLFACTOR=ONE
  GTOT = ZERO
  GTOTOLD=ZERO
  LQMEWD=.false.                ! for multi-layered qm/mm
  !
  !     Zero the QM charges, in case they are not calculated
  QMMUL(1:MAXGMS) = ZERO
  QMLOW(1:MAXGMS) = ZERO
  QMKOL(1:MAXGMS) = ZERO
  !
  ! This forces a restart for all components of this job
#if KEY_GAMESSUK==1
  STARTUP = QINIGM
#endif 

#if KEY_GAMESS==1 /*gamess*/
  RBR=ONE/BOHRR
  !     Update coordinates
  N=0
  do i=1,natqm_2
     iatom        = iabs(qminb2_dual(i))
     n            = n + 1
     c(1,n)       = X(iatom)*RBR
     c(2,n)       = Y(iatom)*RBR
     c(3,n)       = Z(iatom)*RBR
  end do
  !
  CALL CH2GMS(.FALSE.)
#endif /*  (gamess)*/
  !
#if KEY_GAMESS==1 /*gamess*/
  CALL CGREP(NATOMX,QMMMRP,DX,DY,DZ)
  IF(PRNLEV.GE.6) WRITE(OUTU,'(A,2F17.6)') &
       'QM/MM repulsion is (au,kcal) ',QMMMRP,QMMMRP*TOKCAL
  !
  !     From gamess/grd1.src in case of other WFns.
  if(NCHMAT.ne.0) then
     do icharm=1,nchmat
        dxelmm(icharm)=zero
        dyelmm(icharm)=zero
        dzelmm(icharm)=zero
     end do
  end if
  !
  QINIGM=.FALSE.
  CALL GAMESS

#endif /*  (gamess)*/
  !
  !
#if KEY_GAMESSUK==1 /*gamessuk*/
  !
  QINIGM=.FALSE.
  INIT=0
  IVER=5
  CALL GAMESS(INIT,ICODE,STARTUP, LQMEWD, IVER)

  if(icode .ne. 0)then
     if(icode .eq. 1)then
        msg = 'SCF convergence failure '//tmpstr
     else if(icode .ne.0)then
        write(tmpstr,'(i3)')icode
        write(OUTU,*)'Return code '//tmpstr
     endif
     CALL WRNDIE(-1,'<GAMESS-UK>',msg)
  endif
  !
  !     Recover Energy and gradient here
  !
  GTOTOLD=GTOT              ! jwchu
  ! quantum charges written into first (QM) section of CGQMMM
  call gms2chm(gtot,dx,dy,dz,cgqmmm,gmsmap,blfactor,natomx)
  ECURRENT=GTOT-GTOTOLD     ! jwchu
#if KEY_PARALLEL==1
  if(numnod.gt.1) then
     GTOT_local(1)=GTOT
     GTOT_local(2)=ECURRENT
     call GCOMB(GTOT_local,2)
     if(mynod.eq.0) then
        GTOT     = GTOT_local(1)
        ECURRENT = GTOT_local(2)
     else
        GTOT = zero
        ECURRENT = zero
     end if
  end if
#endif 

#endif /* (gamessuk)*/
  !
#if KEY_GAMESS==1 /*gamess*/
  !
  !     If this is DIESEL calculation then get the energy
  CALL DIESELE(E)
  !
  !     NOTE: QMMMRP is already scaled by BLFACTOR
  GTOTOLD=GTOT
#if KEY_PARALLEL==1
  if(mynod.eq.0) then      
#endif
     GTOT= GTOT + E*TOKCAL*BLFACTOR+QMMMRP*TOKCAL
#if KEY_PARALLEL==1
     eles                     
#endif
#if KEY_PARALLEL==1
     GTOT= ZERO            
#endif
#if KEY_PARALLEL==1
  end if                   
#endif
  ECURRENT=GTOT-GTOTOLD
  !
  !
  !     MM atoms, without igmsel(i)=5, unless BLUR !!
  do i = 1,natqm_2
     iatom= iabs(qminb2_dual(i))
     dx(iatom)= dx(iatom) - DXELMM(i)*TOKCAL*RBR*BLFACTOR
     dy(iatom)= dy(iatom) - DYELMM(i)*TOKCAL*RBR*BLFACTOR
     dz(iatom)= dz(iatom) - DZELMM(i)*TOKCAL*RBR*BLFACTOR
     if(Prnlev.gt.6) then
        write(outu,334) iatom,i, - DXELMM(i)*TOKCAL*RBR &
             , - DYELMM(i)*TOKCAL*RBR &
             , - DZELMM(i)*TOKCAL*RBR
     endif
  end do
  !
  !     QM atoms (in parallel they are already summed up!)
#if KEY_PARALLEL==1
  IF(MYNOD.EQ.0) then           
#endif
     n = 0
     do i = natqm_2+1, natomx
        mm = mminb1_dual(i,2)
        if(mm.gt.0) then
           n  = n + 1
           ipt= 3*(n-1)+1
           dx(mm) = dx(mm)+EG(IPT)  *TOKCAL*RBR*BLFACTOR
           dy(mm) = dy(mm)+EG(IPT+1)*TOKCAL*RBR*BLFACTOR
           dz(mm) = dz(mm)+EG(IPT+2)*TOKCAL*RBR*BLFACTOR
           if(Prnlev.gt.6) write(outu,334) i,n,EG(IPT)*TOKCAL*RBR &
                , EG(IPT+1)*TOKCAL*RBR &
                , EG(IPT+2)*TOKCAL*RBR
        end if
     end do
#if KEY_PARALLEL==1
  End if                        
#endif
#endif /*  (gamess)*/

  ! do hard-weird here.
  !ccc     CALL GETQMCHG
  do i=1,natqm_2
     iatom = iabs(qminb2_dual(i))
     QMCMUL(iatom)=QMMUL(i)
     QMCLOW(iatom)=QMLOW(i)
     QMCKOL(iatom)=QMKOL(i)
  end do
  !
  return
END SUBROUTINE HighQM_ene_mlayer

SUBROUTINE COPSEL_Dual(ISLCT2,JLSLCT)
  !-----------------------------------------------------------------------
  !     Copies selection vector to common block
  !
  !     igmsel_dual(I) = 5  MM atom to be excluded from QM/MM non-bonded
  !                         interaction
  !     igmsel_dual(I) = 2  Hydrogen Link atom (QQH)
  !     igmsel_dual(I) = 1  QM atom
  !     igmsel_dual(I) = 0  MM atom
  !
  use chm_kinds
  use exfunc
  use chutil, only : atomid
  use dimens_fcm
  use number
  !
  use gamess_fcm
  use stream
  use psf
  use squantm
  !
  implicit none
  INTEGER ISLCT2(*), JLSLCT(*)

  INTEGER I,J,I1,I2,N,IS,IQ,NLATQ
  CHARACTER(len=4) SID, RID, REN, AC
  LOGICAL LNFLAG, qfind_local
  INTEGER LN
  integer nslct_1st,nslct_2nd,nhslct,ngslct,ii,jj,ncnt_t
  !
  ! Determine the number of quantum mechanical atoms, and
  ! find QM H-link atoms.
  ! Assume all 2nd qm region is within 1st qm region.
  do i=1,nqmax
     qminb2_dual(i)= 0
  end do
  do i=1,maxchl
     ihostguest(1,i)=0
     ihostguest(2,i)=0
  end do
  do i=1,maxa
     igmsel_dual(i) = 0
  end do

  nslct_1st = 0     ! 1st qm
  nslct_2nd = 0     ! 2nd qm
  NHLink    = 0     ! H-link atom (2nd qm)
  nhslct    = 0     ! h-link   selection
  ngslct    = 0     ! gho-atom selection
  if(QMFEP .or. QMLAY) then
     do i=1,natom            ! go over for errors.
        if(igmsel(i).eq.2) call wrndie(-5,'<DUALqm>', &
             'PERT and MLAYer QM/MM do not support QQH Link atom.')

        if(islct2(i).eq.1) then
           if(igmsel(i).ne.1) call wrndie(-5,'<DUALqm>', &
                '2nd QM region should be a subset of 1st QM region.')
        end if
        if(QH_Link .and. (JLSLCT(i).eq.1)) then
           if(igmsel(i).ne.1) call wrndie(-5,'<DUALqm>', &
                '2nd H-link atom should be a subset of 1st QM region.')
        end if
        if(QLINK(2).and. (JLSLCT(i).eq.1)) then
           if(igmsel(i).ne.1) call wrndie(-5,'<DUALqm>', &
                '2nd GHO atom should be a subset of 1st QM region.')
        end if

        igmsel_dual(i)=islct2(i)                 ! 2nd qm region, but
        ! no QQH atoms.
     end do
  end if
  !
  ! for multi-layered QM/MM calculations.
  if(QMLAY) then
     do i=1,natqm(1)                          ! assume no GHO atoms
        nslct_1st = nslct_1st + 1             ! for 2nd qm region.
        ii        = iabs(qminb1_dual(i))
        if(islct2(ii).eq.1) then
           nslct_2nd             = nslct_2nd + 1
           qminb2_dual(nslct_2nd)= ii
           cginb(nslct_1st)      = zero
           if(QH_Link .and. (JLSLCT(ii).eq.1)) then
              nhslct = nhslct + 1
              if(nhslct.gt.MAXCHL) call wrndie(-5,'<DUALqm>', &
                   'Too many QM H-link atoms. Reduce to < 20.')
              ihostguest(1,nhslct)   = ii     ! qm section
           end if
        end if
     end do
     if(QH_Link) then
        if(nhslct.le.0) call wrndie(0,'<DUALqm>', &
             'No H-link atoms selected.')
        NHLink = nhslct
        call Fill_HLink(MAXCHL,NHLink,nslct_2nd,IHOSTGUEST, &
             qminb2_dual)

        ! this H-link atom will replace MM guest atoms. So, put this to the end
        ! of 2nd qm selection. However, igmsel_dual will be "5," so it can be
        ! excluded from QM-MM parssing.
        do i=1,NHLink              ! put h-atom to the end
           ! of 2nd qm selection. 
           nslct_2nd             = nslct_2nd + 1
           ii                    = IHOSTGUEST(2,i)
           qminb2_dual(nslct_2nd)=-ii

           do j=1,natqm(1)               ! for charge of H-link atom
              jj=iabs(qminb1_dual(j))    !
              if(ii.eq.jj) cginb(j)=zero 
           end do
        end do
     end if

     ! for PERT (QM/MM-FEP), for example pKa calculations.
  else if(QMFEP) then
     do i=1,natqm(1)              ! 1st for normal qm atoms.
        nslct_1st = nslct_1st + 1
        ii        = iabs(qminb1_dual(i))
        if(islct2(ii).eq.1) then                   ! 2nd qm selection.
           if(QLINK(2)) then                       ! gho in use.
              if(JLSLCT(ii).ne.1) then             ! non-gho atoms.
                 nslct_2nd             = nslct_2nd + 1
                 qminb2_dual(nslct_2nd)= ii
                 cginb_pka(nslct_2nd,2)= cginb(nslct_1st)
                 cginb(nslct_1st)      = zero
              end if
           else                                    ! no gho in use.
              nslct_2nd             = nslct_2nd + 1
              qminb2_dual(nslct_2nd)= ii
              cginb_pka(nslct_2nd,2)= cginb(nslct_1st)
              cginb(nslct_1st)      = zero
           end if
        end if
     end do

!!! see below, after determine IpKa_Hatm_indx
!!!      if(QpKa_on) cginb(IpKa_Hatm_indx) = zero      ! for H atom annihilated.

     if(QLINK(2)) then           ! 2nd for gho atoms.
        ncnt_t =  0
        do i=1,natqm(1)
           ncnt_t = ncnt_t + 1
           ii     = iabs(qminb1_dual(i))
           if( (islct2(ii).eq.1) .and. (JLSLCT(ii).eq.1) ) then 
              nslct_2nd             = nslct_2nd + 1   
              ngslct                = ngslct + 1
              qminb2_dual(nslct_2nd)=-ii
              cginb_pka(nslct_2nd,2)= cginb(ncnt_t)
              cginb(ncnt_t)         = zero         ! remove for GHO on 2nd qm.
           end if
        end do
        if(ncnt_t .ne. nslct_1st) call wrndie(-1,'<DUALqm>', &
             'Atom selection error for GHO atoms.')
     end if
     NumGHO(2) = ngslct

     ! for SOLVation calculations.
  else if(QMSOLV) then                        ! 1st-qm=2nd-qm
     do i=1,natqm(1)                          ! so, last several atoms
        qminb2_dual(i)= qminb1_dual(i)
        cginb(i)      = zero
     end do
     nslct_1st = natqm(1)
     nslct_2nd = natqm(1)

     QLINK(2)  = QLINK(1)
     if(QLINK(2)) NumGHO(2) = NumGHO(1)

     do i=1,natom
        igmsel_dual(i) = igmsel(i)
     end do
  end if

  if(nslct_2nd .le. 0) call wrndie(-5,'<DUALqm>', &
       'No 2nd QM atoms selected.')

  NATMM_dual(1) = natom - nslct_1st
  NATMM_dual(2) = natom - nslct_2nd

  natqm(2)      = nslct_2nd

  !     find atoms to be excluded from QM/MM calculations.
  !     setup for COPSEL works.
  if(QH_Link) then
     do i=1,NHLink
        igmsel_dual(IHOSTGUEST(2,i)) = 2
     end do
  end if

  ! find which atom is annihilated in the pKa FEP-QM/MM calc.
  ! it assumes two qm regions only differ in the h-atom.
  IpKa_Hatm      =-1000
  IpKa_Hatm_indx =-1000
  if(QpKa_on) then
     do i=1,natqm(1)
        ii          = iabs(qminb1_dual(i))
        qfind_local =.true.
        do j=1,natqm(2)
           if(ii.eq.iabs(qminb2_dual(j))) then
              qfind_local =.false.
              goto 200
           end if
        end do
200     continue
        if(qfind_local) then
           if(IpKa_Hatm.ge.0) CALL WRNDIE(-5,'<DUALqm>', &
                'More than 1 atom is differ for pKa FEP-calc.')

           IpKa_Hatm      = ii             ! position in main array
           IpKa_Hatm_indx = i              ! position in qminb1_dual
           igmsel_dual(ii)= 5
           if(PRNLEV.GE.2) WRITE(OUTU,210) &
                'H-atom to be annihiliated in pKa FEP-calc.:',IpKa_Hatm
        end if
     end do

     if(QpKa_on) cginb(IpKa_Hatm_indx) = zero   ! for H atom annihilated.
  end if
210 FORMAT('DUALqm> ',A,I6)


  ! Check if (2nd) qm atom is connected to any of its neighbors. If "yes,"
  ! then that atom will not be included in QM/MM interactions. This is 
  ! sometimes necessary to prevent opposite charge collision, since QM
  ! cannot prevent this from happening.
  Do i=1,NBOND
     i1=IB(i)
     i2=JB(i)
     !
     ! For qm/mm boundary: link guest (mm) atom should be removed from
     ! the QM/MM SCF procedure
     if(igmsel_dual(i1).eq.1 .and. igmsel_dual(i2).eq.0) then ! qm-mm pair
        if(QLINK(2).and.JLSLCT(I1).eq.1) then
           continue
        else
           igmsel_dual(i2)=5
        end if
     end if
     if(igmsel_dual(i1).eq.0 .and. igmsel_dual(i2).eq.1) then ! mm-qm pair
        if(QLINK(2).and.JLSLCT(I2).eq.1) then
           continue
        else
           igmsel_dual(i1)=5
        end if
     end if

     ! For the link hydrogen atom: remove the link guest (mm) atom.
     ! this does not need for QH_Link, since igmsel_dual(x).eq.2, when x is
     ! the link guest (mm) atom.
     if(.not.QH_Link) then           
        if(igmsel_dual(i1).eq.2 .and. igmsel_dual(i2).eq.0)  &
             igmsel_dual(i2)=5
        if(igmsel_dual(i2).eq.2 .and. igmsel_dual(i1).eq.0)  &
             igmsel_dual(i1)=5
     end if
  End do

  !
  IF(PRNLEV.GE.2) THEN
     WRITE(OUTU,'(/,A)')'DUALqm> QM/MM SET'
     WRITE(OUTU,118)
     WRITE(OUTU,120) &
          'Classical atoms excluded from the 2nd QM calculation'
  ENDIF
118 FORMAT('------------------------------------------------')
120 FORMAT('DUALqm: ',A,':')
121 FORMAT('DUALqm: ',A,A,':')
122 FORMAT(10X,I5,4(1X,A4))
123 FORMAT(10X,I5,4(1X,A4),1X,'*')
124 FORMAT(10X,'NONE.')
  n=0
  do i=1,natom
     if(igmsel_dual(i).eq.5) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev.ge.2) write(outu,122) i,sid,rid,ren,ac
        n = n + 1
     end if
  end do
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     write(outu,120)'2nd Quantum mechanical atoms'
  end if

  n=0
  do i=1,natom
     if(igmsel_dual(i).eq.1) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev.ge.2) write(outu,122) i,sid,rid,ren,ac
        n = n + 1
     end if
  end do
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     if(QH_Link) then
        write(outu,121)'2nd quantum mechanical H-link atoms, ', &
             '(* atom is replaced by H)'
     else
        write(outu,120)'2nd quantum mechanical H-link atoms'
     end if
  end if
  n=0
  do i=1,natom
     if(igmsel_dual(i).eq.2) then
        call atomid(i,sid,rid,ren,ac)
        if(prnlev.ge.2) then
           if(QH_Link) then
              write(outu,123) i,sid,rid,ren,ac
           else
              write(outu,122) i,sid,rid,ren,ac
           end if
        end if
        n = n + 1
     end if
  end do
  if(QLINK(2)) then
     if(prnlev.ge.2) then
        if(n.eq.0) write(outu,124)
        write(outu,120) '2nd quantum mechanical GHO atoms'
     end if
     n=0
     ngslct = natqm(2)-NumGHO(2)
     do i=1,NumGHO(2)
        ngslct = ngslct + 1
        ii = iabs(qminb2_dual(ngslct))
        call atomid(ii,sid,rid,ren,ac)
        if(prnlev.ge.2) write(outu,122) ii,sid,rid,ren,ac
        n = n + 1
     end do
  end if
  if(prnlev.ge.2) then
     if(n.eq.0) write(outu,124)
     write(outu,118)
  end if
  !

  ! do the work for gamess/gamessuk
#if KEY_GAMESS==1 || KEY_GAMESSUK==1
  if(QMLAY) then
     NGAMES=natqm(2)

     ! all for partial charges on any qm atom
     n = 0
     do i=1,natom
        if(igmsel_dual(i).eq.1 .or. igmsel_dual(i).eq.2) then
           n = n + 1
           !cc               LNFLAG=.FALSE.
           !ccc first check the flag for QINP for all atoms.
           !cc               if(QQINP) then
           !cc                  if(.not.LNFLAG) FQQCHG(N)=CG(i)
           !cc               else

           IF(N.LE.MAXGMS) THEN
              FQQCHG(N)=-THOSND
           ELSE
              IF(PRNLEV.GT.2) WRITE(OUTU,'(A,I8)') &
                   'Number of QM atoms exceeded (REPLICA?)',N
              goto 100
           END IF

           !cc               end if
           !
           ! only output for the currrent qm region.
           if(igmsel_dual(i).gt.0) then
              if(prnlev.ge.6) WRITE(OUTU,'(A,2I10,A,F15.5)') &
                   'MLAYer: ATOM(',I,N,') has QNUC: ',FQQCHG(N)
           end if
100        continue
        end if
     end do
  end if
#endif 
  return
END SUBROUTINE COPSEL_Dual


SUBROUTINE CMNQUA_dual(NZUNC_2)
  !----------------------------------------------------------------------
  !     Find the atoms defined as QM atoms and get them ready
  !     for SQUANTM.
  !
  !     Paul D Lyne, September 1995
  !
  use chm_kinds
  use dimens_fcm
  use number
  use param
  use psf
  use stream
  use gamess_fcm
  use squantm
  use linkatom, only: findel
  use rtf, only: atct
  implicit none
  INTEGER NZUNC_2(*)
  !
  INTEGER I,N,NATMM,NACA
  CHARACTER(len=6) ELE
  real(chm_real)  aznuc
  logical qprt
  integer natqm_2,natqm_gho,natqm_hlink,natqm_clnk,nmm_excl
  !
  qprt = .true.
  !
  natqm_2    = natqm(2) 
  natqm_gho  = natqm_2 - NumGHO(2)
  natqm_hlink= natqm_2 - NHLink
  n          = 0

  do i = 1, natqm_2
     n       = iabs(qminb2_dual(i))
     call findel(ATCT(IAC(n)),AMASS(n),n,ELE,aznuc,qprt)

     ! copy for multi-layered qm/mm
     if(QMLAY) then
        if(igmsel_dual(n).eq.2) then                ! for H-link atom.
           AZNUC_high(i) = one
           CAATOM_high(i)= 'H         '
        else
           AZNUC_high(i) = aznuc
           CAATOM_high(i)= ELE
        end if
     end if

     !     assign nuclear charges
     if(QLINK(2) .and. (i.gt.natqm_gho)) then        ! GHO : 85
        nzunc_2(i) = 85
     else if(QH_Link .and. (i.gt.natqm_hlink)) then
        nzunc_2(i) = 1
     else
        nzunc_2(i) = aznuc
     end if

  end do

  natmm = natom -natqm_2

  ! number of H-link atoms
  !     NHLink:
  ! number of Adjusted connecion atoms
  !     no ACC atoms.
  naca = 0
  ! number of GHO atoms
  !     NumGHO(2):
  !
  !     Write out atomic information
  !
  IF (PRNLEV.GT.2) WRITE (OUTU,'(/,1x,A,/)') &
       ' DUALqm> Some atoms will be treated quantum mechanically.'
  IF (PRNLEV.GT.2) THEN
     WRITE (OUTU,'(8X,A,I5,/,8X,A,I5)') &
          ' The number of quantum mechanical atoms  (2nd) = ',natqm_2, &
          ' The number of QM/MM H-link atoms        (2nd) = ',NHLink

     IF(CLINK) WRITE (OUTU,'(8X,A,I5)') &
          ' The number of Adjusted Connection atoms (2nd) = ',naca
     IF(QLINK(1)) WRITE (OUTU,'(8X,A,I5)') &
          ' The number of GHO atoms                 (2nd) = ',NumGHO(2)

     nmm_excl = NATMM-NCUTOFF(2)
     WRITE (OUTU,'(2(8X,A,I5,/),/)') &
          ' The number of molecular mechanical atoms(2nd) = ',natmm, &
          ' The number of MM atoms excluded from QM (2nd) = ',nmm_excl
  END IF

  RETURN
END SUBROUTINE CMNQUA_dual


SUBROUTINE Fill_HLink(MAXCHL,NHLink,natqm_2,IHOSTGUEST,qminb2_dual)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !cc  use number
  !
  use psf
  use gamess_fcm
  !cc  use stream
  !
  implicit none

  INTEGER MAXCHL,NHLink,natqm_2,IHOSTGUEST(2,MAXCHL)
  INTEGER qminb2_dual(*)

  INTEGER I,J,k,ii,jj,Ncnt,iqm,jqm
  !
  ! Find QM host atoms that are connected to MM guest atoms
  ! to place H-link atoms
  !
  Ncnt = 0
  do i=1,NBOND
     ii = IB(i)
     jj = JB(i)
     if((igmsel(ii).eq.1 .or. igmsel(ii).eq.2) .or.    & ! either ii or jj is
          (igmsel(jj).eq.1 .or. igmsel(jj).eq.2)) then  ! 1st qm atom.
        iqm=-1000
        jqm=-1000
        do k=1,natqm_2
           if(ii.eq.iabs(qminb2_dual(k))) iqm=ii      ! if ii belongs 2nd qm.
           if(jj.eq.iabs(qminb2_dual(k))) jqm=jj      ! if jj belongs 2nd qm.
        end do
        if(iqm.eq.ii .and. jqm.eq.-1000) then      ! ii is 2nd qm, jj is 1st qm.
           do j=1,NHLink
              if (ii.eq.IHOSTGUEST(1,j)) then
                 Ncnt = Ncnt + 1
                 IHOSTGUEST(2,j) = jj
              end if
           end do
        else if(iqm.eq.-1000 .and. jqm.eq.jj) then ! ii is 1st qm, jj is 2nd qm.
           do j=1,NHLink
              if(jj.eq.IHOSTGUEST(1,j)) then
                 Ncnt = Ncnt + 1
                 IHOSTGUEST(2,j) = ii
              end if
           end do
        end if
     end if
  end do

  if(Ncnt.ne.NHLink) call wrndie(-5,'<DUALqm>', &
       'Number of counted H-link atoms mismatch. Check psf.')

  return
END SUBROUTINE Fill_HLink


SUBROUTINE get_hlink_coords(NHLink,MAXCHL,IHOSTGUEST, &
     R_H_ref,x,y,z, &
     xyz_hlink,Lxyz_get,Lref_save)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  !
  implicit none

  INTEGER NHLink,MAXCHL
  INTEGER IHOSTGUEST(2,MAXCHL)
  real(chm_real)  R_H_ref(*),X(*),Y(*),Z(*)
  real(chm_real)  xyz_hlink(6,MAXCHL)
  LOGICAL Lxyz_get,Lref_save

  ! local
  INTEGER I,Iatom,Jatom
  real(chm_real)  R_xyz_dist,R_xyz(3)
  real(chm_real), PARAMETER :: R_dist=one


  ! locate H-link atom at 1.0d0 distance away from xyz(Iatom).
  if(Lxyz_get) then
     Do i=1,NHLink
        Iatom=IHOSTGUEST(1,i)
        Jatom=IHOSTGUEST(2,i)
        xyz_hlink(1,i) = x(Iatom)
        xyz_hlink(2,i) = y(Iatom)
        xyz_hlink(3,i) = z(Iatom)

        xyz_hlink(4,i) = x(Jatom)
        xyz_hlink(5,i) = y(Jatom)
        xyz_hlink(6,i) = z(Jatom)

        R_xyz(1) = xyz_hlink(4,i)-xyz_hlink(1,i)
        R_xyz(2) = xyz_hlink(5,i)-xyz_hlink(2,i)
        R_xyz(3) = xyz_hlink(6,i)-xyz_hlink(3,i)

        if(Lref_save) then
           R_xyz_dist=SQRT(R_xyz(1)**2+R_xyz(2)**2+R_xyz(3)**2)
           R_H_ref(i)=one/R_xyz_dist
        end if

        x(Jatom) = x(Iatom)+R_H_ref(i)*R_xyz(1)
        y(Jatom) = y(Iatom)+R_H_ref(i)*R_xyz(2)
        z(Jatom) = z(Iatom)+R_H_ref(i)*R_xyz(3)
     End do
  else
     Do i=1,NHLink
        Jatom=IHOSTGUEST(2,i)
        x(Jatom) = xyz_hlink(4,i)
        y(Jatom) = xyz_hlink(5,i)
        z(Jatom) = xyz_hlink(6,i)
     End do
  end if

  return
END SUBROUTINE get_hlink_coords


SUBROUTINE put_hlink_grads(NHLink,MAXCHL,IHOSTGUEST, &
     R_H_ref,DX,DY,DZ, &
     xyz_hlink,Qnoproj)
  !
  !     This subroutine projects out the force along the H-link atom and
  !     QM atoms bonded to it.
  !
  use chm_kinds
  use dimens_fcm
  use number
  !
  implicit none

  INTEGER NHLink,MAXCHL
  INTEGER IHOSTGUEST(2,MAXCHL)
  real(chm_real)  R_H_ref(*),DX(*),DY(*),DZ(*)
  real(chm_real)  xyz_hlink(6,MAXCHL)
  LOGICAL Qnoproj
  !
  INTEGER I,J,Iatom,Jatom
  real(chm_real)  R_xyz(3),R_xyz_dist,DXI,DYI,DZI
  real(chm_real)  PRJF

  INTEGER IIC,ii
  real(chm_real)  SP,S2,SF2,SS,S_xyz(3),SMF(3,2),P_xyz(3)
  real(chm_real)  FACTF
  real(chm_real), PARAMETER :: R_TOLI=0.0001d0
  logical q_check(MAXCHL),DONE

  if(Qnoproj) then

     do i=1,NHLink
        !
        !    Find unit direction vector
        R_xyz(1) = xyz_hlink(4,i)-xyz_hlink(1,i)
        R_xyz(2) = xyz_hlink(5,i)-xyz_hlink(2,i)
        R_xyz(3) = xyz_hlink(6,i)-xyz_hlink(3,i)

        R_xyz_dist = SQRT(R_xyz(1)**2+R_xyz(2)**2+R_xyz(3)**2)
        R_xyz_dist = one/R_xyz_dist

        R_xyz(1) = R_xyz(1) * R_xyz_dist   ! R_H_ref(i)
        R_xyz(2) = R_xyz(2) * R_xyz_dist   ! R_H_ref(i)
        R_xyz(3) = R_xyz(3) * R_xyz_dist   ! R_H_ref(i)
        !
        ! PRJF = F_doc_E, F: force on atom, E: direction unit vector
        ! DXI  = PRJF*EX, DYI=...etc
        !    For link guest atom (mm-atom)
        iatom = IHOSTGUEST(1,i)
        jatom = IHOSTGUEST(2,i)
        PRJF=dx(jatom)*R_xyz(1)+dy(jatom)*R_xyz(2)+dz(jatom)*R_xyz(3)
        dxi =PRJF*R_xyz(1)
        dyi =PRJF*R_xyz(2)
        dzi =PRJF*R_xyz(3)
        !
        dx(jatom) = dx(jatom)-dxi
        dy(jatom) = dy(jatom)-dyi
        dz(jatom) = dz(jatom)-dzi
        !ccc only project force on 2nd qm-atom side, and ignore on mm (or 1st qm) atom side
        dx(jatom) = zero
        dy(jatom) = zero
        dz(jatom) = zero
        !
        !    For link host atom (qm-atom)
        PRJF=dx(iatom)*R_xyz(1)+dy(iatom)*R_xyz(2)+dz(iatom)*R_xyz(3)
        dxi =PRJF*R_xyz(1)
        dyi =PRJF*R_xyz(2)
        dzi =PRJF*R_xyz(3)
        !
        dx(iatom) = dx(iatom)-dxi
        dy(iatom) = dy(iatom)-dyi
        dz(iatom) = dz(iatom)-dzi
     end do

  else

     FACTF =1.0d-3
     FACTF = R_TOLI*R_TOLI*FACTF*FACTF   ! R_TOLI^2 * FACTF^2

     Do iic=1,NHLink
        q_check(iic)=.FALSE.
     End do

20   CONTINUE

     Do IIC=1,NHLink
        R_xyz(1)  = xyz_hlink(4,iic)-xyz_hlink(1,iic)
        R_xyz(2)  = xyz_hlink(5,iic)-xyz_hlink(2,iic)
        R_xyz(3)  = xyz_hlink(6,iic)-xyz_hlink(3,iic)

        R_xyz_dist= SQRT(R_xyz(1)**2+R_xyz(2)**2+R_xyz(3)**3)
        R_xyz_dist= ONE/R_xyz_dist
        SMF(1,2)  = R_xyz(1)*R_xyz_dist
        SMF(2,2)  = R_xyz(2)*R_xyz_dist
        SMF(3,2)  = R_xyz(3)*R_xyz_dist
        SMF(1,1)  =-SMF(1,2)
        SMF(2,1)  =-SMF(2,2)
        SMF(3,1)  =-SMF(3,2)

        SP = ZERO
        S2 = ZERO
        SF2= ZERO
        do i=1,2
           ii      =IHOSTGUEST(i,iic)
           S_xyz(1)=SMF(1,i)
           S_xyz(2)=SMF(2,i)
           S_xyz(3)=SMF(3,i)
           P_xyz(1)=dx(ii)*S_xyz(1)
           P_xyz(2)=dy(ii)*S_xyz(2)
           P_xyz(3)=dz(ii)*S_xyz(3)
           SP      =SP + P_xyz(1)   +P_xyz(2)   +P_xyz(3)
           S2      =S2 + S_xyz(1)**2+S_xyz(2)**2+S_xyz(3)**2
           SF2     =SF2+ dx(ii)**2  +dy(ii)**2  +dz(ii)**2
        end do

        SS = S2*SF2*FACTF               ! refer FACTF above.
        if(SP*SP.LT.SS) q_check(iic)=.TRUE.
        SP=SP/S2
        !
        ! ... Subtract parallel contribution
        !
        do i=1,2
           ii    =IHOSTGUEST(i,iic)
           dx(ii)=dx(ii)-SP*SMF(1,i)
           dy(ii)=dy(ii)-SP*SMF(2,i)
           dz(ii)=dz(ii)-SP*SMF(3,i)
        end do
     End do ! iic=1,NHLink
     !
     ! ... End of loop over constraints: Check convergence
     !
     DONE = .TRUE.
     do iic=1,NHLink
        DONE=(DONE.AND. q_check(iic))
     end do
     IF(.NOT.DONE)  GOTO 20


  end if

  RETURN
END SUBROUTINE put_hlink_grads


#if KEY_GAMESS==1 || KEY_GAMESSUK==1 /*highqm*/
#if KEY_GAMESS==1 /*gamess_only*/
SUBROUTINE CH2GMS_mlayer(natomx,nchmat,nbluch,ibluch, &
     xchm,ychm,zchm,qchm, &
     tmpblur,ebluch,cgblch,sgblch,cbluch)
  !-----------------------------------------------------------------------
  !     Define CHARMM atoms as point charges and copy to COMMON/CHMGMS/
  !
  !     (Not used in GAMESS-UK case)
  !
  !     To simplify changes in GAMESS we build contiguous arrays:
  !     (XCHM,YCHM,ZCHM,QCHM) for .not.QM atoms.
  !     [NOTE: Similar code is present also in CGREP for
  !     nuclear repulsion derivatives]
  !
  !     On QM atoms we don't care for cutoff, we just take all of the
  !     atoms in the system. This is not inconsistent with MM since QMs
  !     are special anyway. Also calculation time is linear with number
  !     of MM atoms!
  !
  !     [NOTE: It should be straightforward to implement usage of
  !     variety of cutoff methods implemented in the CHARMM:
  !     Just use nonbond array here and disable CGREP routine;
  !     but you need to specify charges on QM atoms as their
  !     atomic number in the RTF file!!!
  !
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use exfunc
  !
  use coord
  use squantm

  use psf
  use scalar_module
  !cc  use stackm
  !cc  use stream
  !cc  use gamess_fcm
  !
  implicit none

  INTEGER natomx,nchmat,nbluch
  INTEGER ibluch(*)
  real(chm_real)  xchm(*),ychm(*),zchm(*),qchm(*)
  real(chm_real)  tmpblur(*),ebluch(*),cgblch(*),sgblch(*),cbluch(*)
  !
  INTEGER J,I,N,NBLUR
  real(chm_real) SIGM1,SIGM2,SIGM3
  real(chm_real) RBR,FAC
  real(chm_real), parameter :: rtpipoh=0.5079490874739d0

  !cc      real(chm_real) TPIPOH

  !
  N=0
  NBLUR=0
  RBR=ONE/BOHRR
  IBLUCH(1:natomx) = 0

  n = 0
  if(QBLUCH) then
     !     fill in blurred charges array
     if(recallint.ne.-1) then
        do i=1,natomx
           !! this was wrong               tmpblur(i) = ptrsto(i)
           tmpblur(i)=ptrsto(recallint)%a(i)
        end do
     else
        do i=1,natomx
           tmpblur(i) = wmain(i)
        end do
     end if

     do i=natqm_2+1,natomx
        mm = mminb1_dual(i,2)
        if(mm.gt.0) then
           n = n + 1
           xchm(n) = x(mm)*RBR
           ychm(n) = y(mm)*RBR
           zchm(n) = z(mm)*RBR
           qchm(n) = cg(mm)

           if(abs(tmpblur(mm)).ge.rsmall) then
              if(tmpblur(mm).gt.NINE99) then
                 qchm(n) = zero
              else
                 nblur = nblur + 1
                 ibluch(nblur) = n
                 sigm1 = BOHRR/tmpblur(mm)
                 sigm2 = sigm1*sigm1
                 sigm3 = sigm2*sigm1
                 ebluch(nblur)=sigm2
                 cgblch(nblur)=cg(mm)   ! do not use BFIRST check.
                 sgblch(nblur)=tmpblur(mm)*RBR
                 ! it was:            cbluch(nblur)=cgblch(nblur)*sigm3/TPIPOH
                 !  or                cbluch(nblur)=cgblch(nblur)*sigm3*0.5079490874739d0
                 cbluch(nblur)=cgblch(nblur)*sigm3*rtpipoh

                 !     put to zero both mm charges
                 !     a) the one which goes to GAMESS (QCHM)
                 !ccC     b) the one which goes to CHARMM (CG)
                 !cc                     cg(mm)  = zero
                 qchm(mm)= zero

                 !ccc     for now put back mm charges
                 !ccc     they are dealt in CHARMM, and we only have
                 !ccc     blurred charges interacting with QM atoms.
                 !cc                     cg(mm)  = cgblch(nblur) 
              end if
           end if
        end if
     end do
  else
     do i=1,natqm_2+1,natomx
        mm = mminb1_dual(i,2)
        if(mm.gt.0) then
           n = n + 1
           xchm(n) = x(mm)*RBR
           ychm(n) = y(mm)*RBR
           zchm(n) = z(mm)*RBR
           FAC     = one            ! for block? (GAMESS only, also no blur yet)
           qchm(n) = cg(mm)*FAC
        end if
     end do
  end if
  !
  NCHMAT=N
  NBLUCH=NBLUR
  !
  RETURN
END SUBROUTINE CH2GMS_mlayer

SUBROUTINE CGREP_mlayer(NATOM,REPULS,DX,DY,DZ, &
     c,zan)
  !-----------------------------------------------------------------------
  !
  !     This routine calculates nuclear repulsion between
  !     CHARMM atoms and GAMESS atoms + derivatives
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use exfunc
  !
  use stream
  !cc  use coord
  use gamess_fcm
  use parallel
  use squantm
  !
  !
  implicit none

  INTEGER NATOM
  real(chm_real) REPULS,DX(*),DY(*),DZ(*)
  real(chm_real) ZAN(MAXGMS), C(3,MAZGMS)
  !
  INTEGER I,J,N,KBLUCH,KKBLCH,NN,im,iq
  real(chm_real) ERRF,ETMP,SIGM1,SIGM2,TSQP
  real(chm_real) Q1,Q2,X1,X2,Y1,Y2,Z1,Z2,R12,RR12,EL,ELR
  real(chm_real) X12,Y12,Z12,RBR
#if KEY_REPLICA==0
  real(chm_real) BLFACTOR          
#endif
  !
  !
  !     Put derivatives in the right places in DX,DY,DZ arrays
  !     and take Newton's 3rd law into account. 
  !
#if KEY_REPLICA==0
  BLFACTOR=ONE     
#endif
  RBR=ONE/BOHRR
  TSQP = TWO/SQRT(PI)
  REPULS = ZERO

#if KEY_PARALLEL==1
  IF (MYNOD.GT.0) RETURN
#endif 
  !
  !     This loop is for QM nuclei - MM atoms electrostatic interaction
  !     It deals also with QM nuclei - Blurred charge interaction
  !
  ! loop over qm atoms.
  do i= 1, natqm_2
     iq = iabs(qminb2_dual(i))
     q1 = zan(i)
     if(FQQCHG(i).GT.-nine99) q1 = FQQCHG(i)
     x1 = c(1,i)
     y1 = c(2,i)
     z1 = c(3,i)
     n      = 0
     kbluch = 1
     ! for mm atoms.
     do j=natqm_2+1,natom
        im=mminb1_dual(j,2)
        if(im.gt.0) then
           n = n + 1
           !cc               x2= xchm(n)
           !cc               y2= ychm(n)
           !cc               z2= zchm(n)
           !cc               q2= qchm(n)
           x12 = x1-xchm(n)   ! x1-x2
           y12 = y1-ychm(n)   ! y1-y2
           z12 = z1-zchm(n)   ! z1-z2
           rr12= x12*x12+y12*y12+z12*z12
           r12 = sqrt(rr12)
           if(QBLUCH .and. (N.eq.ibluch(kbluch))) then
              !
              !      qm nuclei - Blurred charge interaction
              q2    = cgblch(kbluch)
              etmp  = r12/sgblch(kbluch)
              el    = q1*q2/r12*BLFACTOR
              elr   = el/rr12 &
                   *(errf(etmp)-tsqp*etmp*exp(-etmp*etmp)) &
                   *tokcal*RBR
              el    = el*errf(etmp)
              kbluch= kbluch + 1
           else
              el    = q1*qchm(n)/r12*BLFACTOR
              elr   = el/rr12*tokcal*RBR 
           end if
           repuls   = repuls + el

           dx(iq) = dx(iq) - x12*elr
           dx(im) = dx(im) + x12*elr
           dy(iq) = dy(iq) - y12*elr
           dy(im) = dy(im) + y12*elr
           dz(iq) = dz(iq) - z12*elr
           dz(im) = dz(im) + z12*elr
        end if
     end do
  end do
  !
  !     This loop is for Blurred charges - MM atoms electrostatic interaction
  !     It deals also with the Blurred charge - Blurred charge interaction
  !
  !============================
  !
  !     SIMPLIFICATION: (????)
  !     For now we deal with blurred charges not interacting
  !     with QM region in the CHARMM as classical atoms.
  !     The following code is not good for MM - Blur and Blur - Blur,
  !     because it doesn't deal correctly with the bonded atoms!!
  !
  !     USE: bnbnd%inblo,bnbnd%jnb
  !
  !
  if(QBLUCH) return
  if(QBLUCH) then
     n      = 0
     kbluch = 1
     do i=natqm_2+1,natom
        iq=mminb1_dual(i,2)
        if(iq.gt.0) then
           n = n + 1
           if(n .eq. ibluch(kbluch)) then
              !
              !     found kbluch-th blurred charge.
              x1    = xchm(n)
              y1    = xchm(n)
              z1    = zchm(n)
              q1    = cgblch(kbluch)
              sigm1 = sgblch(kbluch)
              kbluch= kbluch + 1
              nn    = 0
              kkblch= 1
              do j=natqm_2+1,natom
                 im=mminb1_dual(j,2)
                 if(im.gt.0) then
                    !
                    !     either mm or blurred atom.
                    nn = nn + 1
                    x2 = xchm(nn)
                    y2 = ychm(nn)
                    z2 = zchm(nn)
                    x12= x1-x2
                    y12= y1-y2
                    z12= z1-z2
                    rr12= x12*x12+y12*y12+z12*z12
                    r12 = sqrt(rr12)
                    elr = zero
                    el  = zero
                    if(nn.eq.ibluch(kkblch)) then
                       if(kkblch .ge.kbluch) then
                          !
                          !     for blurred charge-blurred charge.
                          sigm2 = one/sqrt(ebluch(kkblch))
                          etmp  = r12/sqrt(sigm1*sigm1+sigm2*sigm2)
                          el    = q1*cgblch(kkblch)/r12
                          elr   = el/r12 &
                               *(errf(etmp)-tsqp*etmp*exp(-etmp*etmp)) &
                               *tokcal*RBR
                          el    = el*errf(etmp)
                       end if
                       kkblch = kkblch + 1
                    else
                       !
                       !     for blurred charge - mm charge.
                       etmp   = r12/sigm1
                       el     = q1*qchm(nn)/r12
                       elr    = el/r12 &
                            *(errf(etmp)-tsqp*etmp*exp(-etmp*etmp)) &
                            *tokcal*RBR
                       el     = el*errf(etmp)
                    end if
                    repuls    = repuls + el

                    dx(iq) = dx(iq) - x12*elr
                    dx(im) = dx(im) + x12*elr
                    dy(iq) = dy(iq) - y12*elr
                    dy(im) = dy(im) + y12*elr
                    dz(iq) = dz(iq) - z12*elr
                    dz(im) = dz(im) + z12*elr
                 end if
              end do
           end if
        end if
     end do
  end if
  !
  RETURN
END SUBROUTINE CGREP_mlayer
!
SUBROUTINE CGREPE_mlayer(NATOM,Prnlev,ZAN,C,E,EG)
  !-----------------------------------------------------------------------
  !
  !     This is here for the debugging purpose!
  !     This routine calculates nuclear repulsion between
  !     CHARMM atoms and GAMESS atoms - IT DOES NOT WORK WITH BLUR!!!
  !                                     Try to get rid of it
  !                                     by filling ETERM array with the
  !                                     terms calculated in the CGREP
  !                                     subroutine
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use number
  use exfunc
  !
  !cc  use stream
  use gamess_fcm
  use squantm
  !
  implicit none

  INTEGER NATOM,Prnlev
  real(chm_real) ZAN(MAXGMS),C(3,MAXGMS)
  real(chm_real) E, EG(3*MAXGMS)
  !
  INTEGER I,J,N,KBLUCH,iq,im
  real(chm_real) Q1,Q2,X1,X2,Y1,Y2,Z1,Z2,R12,RR12,EL
  real(chm_real) X12,Y12,Z12
  real(chm_real) REPULS
  !
  REPULS = ZERO

  !     loop over qm atoms.
  do i=1,natqm_2
     iq     = iabs(qminb2_dual(i))
     q1     = ZAN(i)
     if(FQQCHG(L).GT.-NINE99) q1=FQQCHG(i)
     x1     = c(1,i)
     y1     = c(2,i)
     z1     = c(3,i)

     !     loop over mm atoms.
     n      = 0
     kbluch = 1
     do j=natqm_2+1,natom
        im  = mminb1_dual(j,2)
        if(im.gt.0) then
           n = n + 1
           x12= x1-xchm(n)   ! x1-x2
           y12= y1-ychm(n)   ! y1-y2
           z12= z1-zchm(n)   ! z1-z2
           rr12= x12*x12+y12*y12+z12*z12
           r12 = sqrt(rr12)
           el  = q1*qchm(n)/r12
           repuls = repuls + el
        end if
     end do
  end do
  !
  IF(PRNLEV.GE.2) THEN
     WRITE(6,'(A,2F20.8)')'QM/MM repulsion (a.u.,kcal/mole) = ', &
          REPULS,REPULS*TOKCAL
     WRITE(6,'(A,2F20.8)')'QM/MM total en. (a.u.,kcal/mole) = ', &
          E+REPULS,(E+REPULS)*TOKCAL
  ENDIF
  !
  RETURN
END SUBROUTINE CGREPE_mlayer
#endif /* (gamess_only)*/

SUBROUTINE CHMDAT_mlayer(AATOM,AZNUC,CORD,NAT &
#if KEY_GAMESSUK==1
     ,nel,expo,wght,maxatg       & 
#endif
     ,X,Y,Z,WMAIN &
     )
  !-----------------------------------------------------------------------
  !     Multi-layered QM/MM version of CHMDAT.
  !
  !     Define the set of quantum mechanical atoms and set up the data
  !     structures.
  !
  !     NB - for the GAMESS-UK interface, this deals with the
  !     classical atoms as well, since all interactions are
  !     handled within GAMESS-UK
  !
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  !
  !cc  use coord
  use consta
  use psf
  use param
  use stream
  use gamess_fcm
  use parallel
#if KEY_GAMESSUK==1
  use scalar_module       
#endif
  use squantm
  !
  !     COMMON from GAMESS
  !
  !     The following is defined in GAMESS after call to CHMDAT
  !CCC  test code...
  logical mopac
  common /mpctst/mopac
  !CCC  end test code...
  !
  implicit none

#if KEY_GAMESS==1 /*gamess*/
  CHARACTER(len=10) AATOM(MAXGMS)
  real(chm_real) AZNUC(MAXGMS), CORD(MAXGMS,3)
#endif /* (gamess)*/
#if KEY_GAMESSUK==1 /*gamessuk*/
  CHARACTER(len=10) AATOM(*)
  real(chm_real) AZNUC(*), CORD(3,*)
  real(chm_real) expo(*),wght(*),SIGM1,SIGM2,totnuc
  INTEGER NEL, IATOM, MAXATG, K
  real(chm_real) AZN, EXP1, WGH, TESTNE
  CHARACTER(len=6) atype
  logical qm,obq
#endif /* (gamessuk)*/

  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)

  INTEGER NAT,natqm_2
  INTEGER I,NSLCT,NATMM_high,NATQM_high,NATLNK,n,mm
  CHARACTER(len=6) ELE
  real(chm_real) tmpblur(maxa)
  real(chm_real) RBR

  !  Counter for true QM atoms
  natqm_2   =natqm(2)
  NATQM_high=0
  RBR       =one/BOHRR

#if KEY_GAMESSUK==1 /*gamessuk*/

  !
  ! for consistency with GAMESS(US) this is
  ! a counter for BQ-class centres and Blurred centres
  ! (not useful to the QM code)
  !
  NCHMAT=0
  totnuc = zero

  ! to obtain position in the charmm list of
  ! atom igms in the GAMESS-UK list use
  !
  ! ichm = GMSMAP(igms)
  DO I = 1, NATOM
     GMSMAP(I) = -1
  ENDDO
  !
  if(natqm_2.gt.MAXATG) call wrndie(0,'<CHMDAT_mlayer>', &
       'Too many atoms, redimension GAMESS-UK')

  ! ensure mass vector is only referenced for initialisation
  ! step (helps use lone pair assignments)
  ! 
  if(QINIGM) then
     do i = 1, natom
        if(igmsel_dual(i).eq.1) then
           A2MASS(i) = AMASS(i)
        else if(igmsel_dual(i).eq.2) then  ! it is 2 for H-link atom.
           A2MASS(i) = 1.00800d0
        else
           A2MASS(i) = ZERO      ! right?
        end if

        ! for charges
        QMCMUL(i) = CG(i)
        QMCLOW(i) = CG(i)
        QMCKOL(i) = CG(i)
     end do
  end if

  !  First loop Quantum atoms
  iatom = 0
  do i = 1,natqm_2
     n            = iabs(qminb2_dual(i)) 
     iatom        = iatom + 1
     natqm_high   = natqm_high + 1
     gmsmap(iatom)= n

     cord(1,iatom)= x(n)*RBR 
     cord(2,iatom)= y(n)*RBR 
     cord(3,iatom)= z(n)*RBR 

     expo(iatom)  =  minone
     wght(iatom)  =  zero 
     !cc         AZNUC(iatom) =  zero 

     aatom(iatom) = CAATOM_high(iatom)
     uznuc(iatom) = AZNUC_high(iatom)
     aznuc(iatom) = AZNUC_high(iatom)

     ! Store sum of nuclear charges
     !
     totnuc       = totnuc + aznuc(IATOM)
  end do

  ! just keep here, though may not be used.
  !  Secon loop for blurred atoms.
  !
  ! it has a high chance to be wrong....!!! (iatom sequence is not same as
  ! original implementation!!! (namkh)
  !
  if(QBLUCH)then
     ! it is not set in gamess.f90 (namkh)
     !        if(recallint.ne.-1)then
     !           do k=1,natom
     !              tmpblur(k)=PTRSTO(k)
     !           enddo
     !        else
     do k=1,natom
        tmpblur(k)=WMAIN(k)
     enddo
     !        endif

     do i=1,natom
        if(igmsel_dual(i).eq.0) then
           IF (ABS(tmpblur(i)) .GE. RSMALL) THEN
              IF(tmpblur(i).GT.NINE99) THEN
                 !
                 ! explicitly excluded atom
                 !
              ELSE
                 !
                 ! add a blurred centre
                 !
                 IATOM = IATOM + 1
                 if (IATOM .GT. MAXATG)  &
                      call wrndie(0,'<CHMDAT_mlayer>', &
                      'Too many atoms, redimension GAMESS-UK')

                 gmsmap(iatom) = i
                 NCHMAT = NCHMAT + 1
                 AZNUC(IATOM)=ZERO
                 AATOM(IATOM)='BQ        '
                 CORD(1,IATOM)=X(I)*RBR
                 CORD(2,IATOM)=Y(I)*RBR
                 CORD(3,IATOM)=Z(I)*RBR
                 NBLUCH=NBLUCH+1
                 SIGM1=BOHRR/tmpblur(I)
                 SIGM2=SIGM1*SIGM1
                 expo(IATOM) = SIGM2
                 wght(IATOM) = CG(I)
              ENDIF
           ENDIF
        end if                    ! igmsel_dual(i).eq.0
     end do
  end if                          ! QBLUCH
  !
  !  Third loop to assign BQ atoms
  !
  do i = natqm_2+1,natom
     mm = mminb1_dual(i,2)
     if(mm.gt.0) then
        OBQ = .TRUE.

        !           skip blurred centres (already included above)
        !           this statement also skips explicitly excluded atoms,
        !           these have WMAIN(I).GT.NINE99
        if(QBLUCH) OBQ = .NOT. (ABS(tmpblur(mm)) .GE. RSMALL)

        if(OBQ) then
           iatom = iatom + 1
           if(iatom .gt. maxatg) call wrndie(0,'<CHMDAT_mlayer>', &
                'Too many atoms, redimension GAMESS-UK')
           gmsmap(iatom) = mm
           NCHMAT        = NCHMAT + 1
           aatom(iatom)  = 'BQ        '
           cord(1,iatom) = x(mm)*RBR 
           cord(2,iatom) = y(mm)*RBR 
           cord(3,iatom) = z(mm)*RBR 
           expo(iatom)   = minone
           wght(iatom)   = zero
           aznuc(iatom)  = CG(mm)
        end if
     end if
  end do
  !
  nat    = iatom
  nel    = NINT(totnuc)
  TESTNE = NEL
  TESTNE = TESTNE - TOTNUC
  if ( dabs(testne) .gt. PT0001) then
     write (6,9568) totnuc,1.0d0*nel,testne
9568 format(1x,'non-integral charge found',1x,3e15.8)
     CALL WRNDIE(0,'<CHMDAT_mlayer>','non-integral QM charge')
  endif
  !
#endif /*  (gamessuk)*/
  !
#if KEY_GAMESS==1 /*gamess*/
  !
  do i = 1,natqm_2
     n = iabs(qminb2_dual(i))
     natqm_high = natqm_high
     cord(natqm_high,1)=x(n)
     cord(natqm_high,2)=y(n)
     cord(natqm_high,3)=z(n)
     cuniq(natqm_high,1)=cord(natqm_high,1)
     cuniq(natqm_high,2)=cord(natqm_high,2)
     cuniq(natqm_high,3)=cord(natqm_high,3)

     if(QINIGM) THEN
        aatom(natqm_high)=CAATOM_high(natqm_high)
        uatom(natqm_high)=aatom(natqm_high)
        aznuc(natqm_high)=AZNUC_high(natqm_high)
        uznuc(natqm_high)=AZNUC_high(natqm_high)
     end if
  end do
  !
  NAT=NATQM_high
  NATREL=NAT
  UATOM(NAT+1)='$END      '
  !
#endif /* (gamess)*/
  !
  IF (NATQM_high .LE. 0) CALL WRNDIE(0,'<CHMDAT_mlayer>', &
       'No quantum mechanical atoms selected.')
  NATMM_high = NATOM - NATQM_high
  NGAMES = NATQM_high
  !
  !     Write out some information and options requested.
  !
  IF(QBLUCH.AND.PRNLEV.GE.2.AND.QINIGM) WRITE (OUTU,'(/,8X,A,I10)') &
       ' The number of blurred MM charges         = ',NBLUCH
  !
  RETURN
END SUBROUTINE CHMDAT_mlayer
#endif /* (highqm)*/
! .. #else /*  (mainsquatn)*/
! SUBROUTINE QM_ENE_BLANK
!
!  RETURN
! END SUBROUTINE QM_ENE_BLANK
#endif /*  (mainsquatn)*/

