module replica_mod
  use chm_kinds
  use dimens_fcm
  use replica_ltm

  implicit none
  character(len=*),private,parameter :: file_name   ="replica.src"

#if KEY_REPLICA==1 /*replica_fcm*/
  !-----------------------------------------------------------------------
  !# <caves>-Jul-31-1991 (Leo Caves) Add some convenient arrays for Replica
  !
  !     qRep : are replicas in use at present ?
  !     nSub : number of REPLicated sub-systems present (includes primary)
  !     nRepl: total number of replicates in all sub-systems (includes
  !            primary) 
  !
  !     repNoA - for each atom denotes individual replica (nClone) unit
  !              membership 
  !     repNoG - as above for groups
  !     repID  - identifies replicates with their sub-system 
  !     repWt  - for each atom denotes its degree of replication (1/no.
  !              of clones)  
  !
  !     The following variables are useful, so as not to have to change
  !     calling sequences in non-bond list generation routines.  They are
  !     included in the integer common block. 
  !     nRepXA - number of atom pairs excluded from non-bond list due to
  !              replication 
  !     nRepXG - as above for group pairs 
  !
  !     QQCRP   - Logical to tell if Q-Chem is doing replica path/neb
  !     QRPMINP - Logical to tell Q-Chem that it needs individual input files
  !               per replica (only works with 2 replicas currently)  
  !
  !     NEXT 3 REFER TO DOING OFF-PATH SIMULATIONS
  !     BFIJ   - Store the current rms bestfit (J -> I(ref))
  !     BFKJ   - Store the current rms bestfit (J -> K(ref))
  !     BFJJ   - Store the current rms bestfit (J -> J(ref))
  !
  !-----------------------------------------------------------------------
  logical :: qRepDB
  logical, save :: QQCRP = .false.
  logical, save :: QRPMINP = .false.
  INTEGER nSub
  integer,allocatable,dimension(:) :: repNoA, repNoG, repID
  INTEGER nRepXA, nRepXG
  real(chm_real),allocatable,dimension(:) ::  repWt
  real(chm_real) :: BFIJ(100),BFKJ(100),BFJJ(100)
  !
#endif /* (replica_fcm)*/
  !

contains
#if KEY_REPLICA==1 /*allocate*/
  subroutine allocate_replica()
    use memory
    character(len=*),parameter :: routine_name="allocate_replica"
    call chmalloc(file_name,routine_name,'repnoa',maxa,intg=repnoa)
    call chmalloc(file_name,routine_name,'repnog',maxgrp,intg=repnog)
    call chmalloc(file_name,routine_name,'repid ',maxseg,intg=repid)
    call chmalloc(file_name,routine_name,'repwt ',maxa,crl=repwt)
    return
  end subroutine allocate_replica
#endif /* (allocate)*/
#if KEY_REPLICA==0
  SUBROUTINE Replica
    CALL WRNDIE(-1,'<REPLICA>','REPLICA code not compiled.')
    RETURN
  end SUBROUTINE Replica
#else /**/
  SUBROUTINE Replica
    !-----------------------------------------------------------------------
    !     Generate replicas of arbitrary regions of the psf.
    !     May-9-1991 (Leo Caves)
    !
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use psf
    use param
    use code
    use bases_fcm
    use comand
    use select
    use stream
    use string
    use number
    use pert
    use econtmod
    use chm_types
    use memory
    implicit none
    !
    ! Local Pointers. 
    integer, PARAMETER :: nPtr=6,pFlg=1,pInd=2,pGrp=3,pRes=4,pFl2=5,pMap=6
    !      INTEGER SRepWk(2,nPtr), ptr(2)
    type(chm_iptr),dimension(nptr) :: SRepWk

    ! Local variables.
    CHARACTER(len=8) newSeg
    INTEGER i, nSel, nCopy, onSeg
    LOGICAL qCOMP,qSetIC,qReset
    !
    ! Begin Executable Code. 
    !
    ! reset ? 
    qReset = (IndxA(comLyn,comLen,'RESE').GT.0)
    IF (qReset) THEN
       IF (qRep) CALL RepReset
       qRep = .FALSE.
       if(prnlev >= 2) write(outu,*) ' Replica: Reset. ' // &
            'A single (primary) subsystem is assumed.'
       RETURN
    ENDIF
    !
#if KEY_PERT==1
    IF(QPERT) THEN
       CALL WRNDIE(-3,'<REPLICA>', &
            'REPLICA must be invoked before PERT')
       RETURN
    ENDIF
#endif 
    ! move on...
    !
    ! invokation of replica command requires subsequent use of econt array.
    ! check and allocate if neccsary. cb3
    if(.not. allocated(econt)) call allocate_econt
    !
    !
    ! invokation of replica command requires subsequent use of econt array.
    ! check and allocate if neccsary. cb3
    if(.not. allocated(repnoa)) call allocate_replica
    !
    !
    onSeg = nSeg
    newSeg = '    '
    ! Initialize the pointer array for working arrays.
    !------------------------------------------------------
    ! Allocate wk arrays of size nAtom
    DO i = 1, 4 
       call chmalloc('replica.src','replica','SRepWk(i)',nAtom,intgp=SRepWk(i)%a)
       SRepWk(i)%a(1:nAtom) = 0
    enddo
    ! Allocate wk arrays of size nAtom+1
    DO i = 5, 6 
       call chmalloc('replica.src','replica','SRepWk(i)',nAtom+1,intgp=SRepWk(i)%a)
       SRepWk(i)%a(1:nAtom+1) = 0
    enddo
    !------------------------------------------------------

    ! Parse the command line
    !
    ! the number of replicas to create
    nCopy = GtrmI(comLyn,comLen,'NREP',2)
    !
    ! Comparison coordinates for selection ?
    qCOMP  = (IndxA(comLyn, comLen, 'COMP') .GT. 0)
    ! Replicate IC entries ?
    qSetIC = (IndxA(comLyn,comLen,'SETU').GT.0)
    ! DEBUG output from replica routines 
    qRepDB = (IndxA(comLyn,comLen,'DEBU').GT.0)
    ! the atoms to be replicated.
    nSel = AtmSel (comLyn, comLen, SRepWk(pFlg)%a, qCOMP) 
    !
    !...the new segment identifier
    !...parse this last to be consistent with GENERate command. Allows blank segid.
    !...in the event of a blank segid, the segid will be the replica number
    newSeg = NextA8(comLyn,comLen)
    !
    ! Need to check for duplicate segment names here. <=======
    !
    IF (nSel .GT. 0) THEN 
       !
       ! Initialize the replica flags for the primary system
       ! this could be placed in the standard PSF generation routines.
       IF (.NOT. qRep) CALL RepReset

       ! do the stuff...

       ! NB:   nSub is the number of sub-systems 
       !       nRepl is the total number of replicas in all sub-systems
       !       nCopy is the number of replicas in the current sub-system
       !
       nSub = nSub + 1

       CALL Repl2(  &
            SRepWk(pFlg)%a, SRepWk(pFl2)%a,  &
            SRepWk(pInd)%a, SRepWk(pMap)%a,  &
            SRepWk(pGrp)%a, SRepWk(pRes)%a, &
            newSeg, nCopy, qSetIC )

       !
       ! need someway of wiping out replica handling
       IF ( .NOT. qRep ) qRep = .TRUE. 

       if (prnlev >= 2) write(outu, '(A, I0, A, I0, A)') &
            ' REPLIcate> Segments ', onSeg+1, &
            ' to ', nSeg, ' have been generated.'
       if (prnlev >= 2) write(outu, '(2A, I0, A, I0)') &
            ' REPLIcate> Their identifiers are ',  &
            newSeg, 1, ' to ', nCopy 
       !
       ! Print out the structure file counters (this also forces a reset of the lists)
       CALL PsfSum ( outu )

    ELSE
       CALL WrnDie(0,'<Replica>','No atoms selected')
    ENDIF ! (nSel.GT.0)

    !DEBUG
    IF (qRepDB)THEN
       if(prnlev >= 2) write(outu,*)' GROUP FLAGS'
       if(prnlev >= 2) write(outu,*)' ID, NO '
       DO i = 1, nGrp
          if(prnlev >= 2) write(outu,*) repID(repNoG(i)),repNoG(i)
       ENDDO
       if(prnlev >= 2) write(outu,*)' ATOM  FLAGS'
       if(prnlev >= 2) write(outu,*)' ID, NO, WT'
       DO i = 1, nAtom
          if(prnlev >= 2) write(outu,*)  &
               repID(repNoA(i)),repNoA(i),repWt(i)
       ENDDO
    ENDIF

    !...Free up space.
    do i = 1, 6
       if (associated(SRepWk(i)%a)) then
          call chmdealloc('replica.src','replica','SRepWk(i)', &
               size(SRepWk(i)%a),intgp=SRepWk(i)%a)
       endif
    enddo

    !...Exit.
    RETURN
  END SUBROUTINE Replica

  SUBROUTINE Repl2( flag, flag2, index, map, grp, rs,  &
       newSeg, nCopy, qSetIC )
    !-----------------------------------------------------------------------
    !# <caves>-May-9-1991 (Leo Caves)
    !
    !...Replicate a selected region of the PSF (nCopy times)
    !...The replicated atoms are placed in nCopy new segments
    !...The new segments are called nSegidxxx (where xxx is the
    !...sequential number of replica (up to nCopy)
    !...Residue names and resids for replicas are copied from the original.
    !...IC entries are duplicated if qSetIC is .TRUE.
    !...NB: At the moment the primary selection must be deleted 
    !...immediately following the REPLica statement --- otherwise it is
    !..."visible" to its respective replicated subsytem.
    !
    !...The routine is essentially DUPLIC of Axel T. Brunger from X-PLOR
    !...It has been modified for generating many duplicates (replicas)
    !...and for the CHARMM PSF format. (some handles into REPLica data
    !...structure have been added).
    !
    ! WISH LIST:
    ! Replicate run-time attributes of the system such as constraints
    ! BLOCK membership, SHAKE stuff etc., SBOUnd stuff...
    !
    !# <caves>-Aug-6-1993 (Leo Caves) DISCLAIMER ! 
    !  This code is horribly inefficient. it uses the tallest loop I have ever had
    !  the misfortune to write. My only defence is that when written (some
    !  time ago) it was the only way I could "think" about the PSF data structures.
    !  Now, I could do a much better job and actually started to recode this stuff,
    !  more out of embarrassment than need.  But commonsense prevailed, and I have
    !  decided to let out this original code, until our knowledge of the use of
    !  REPLICAS demands a revamped version. This code has been in use for some time
    !  now and appears to be reasonably robust and free of (major) errors !  It is
    !  generally only called once, at the beginning of a job and as such its
    !  inefficiency should not be a major inconvenience. --- Leo
    !
    use chm_kinds
    use intcor_module, only: reintc_new, icr_struct
    use intcor2,only:geticel,puticel
    use dimens_fcm
    use number
    use bases_fcm
    use psf
    use coord
    use coordc
    use deriv
    use cnst_fcm
    use econtmod
    use comand
    use stream
    use string
    use lonepr
#ifdef KEY_RESIZE
    use resize
#endif
    
    implicit none
    !
    !...Passed Variables.
    INTEGER flag(*),index(*), grp(*), rs(*),   &
         flag2(0:*), map(0:*), nCopy  
    CHARACTER(len=*) newSeg
    LOGICAL qSetIC
    !
    !...Global Variables.
    !
    !...Functions
    !!      LOGICAL BOrTag2, BOrTag3, BOrTag4
    !!      LOGICAL BOrTag8
    !
    !...Local Variables. 
    INTEGER i, j, n, iAt, iGp, iRs, iSg, nFlag,  &
         replAt, origAt, expNum
    CHARACTER(len=8) segNam,cSg
    INTEGER segLen, cSgLen
    !...old PSF (and IC) counters.
    INTEGER onAtom, onRes, onSeg, onBond, onTheta, onPhi,  &
         onImPhi, onNB, onDon, onAcc, onGrp, onST2, olenIC
#if KEY_CMAP==1
    INTEGER onCrTerm
#endif 
    INTEGER II,JJ,KK,LL,IRSZ
    LOGICAL TT
    real(chm_real)  BB1,BB2,TT1,TT2,PP
#if KEY_LONEPAIR==1
    INTEGER ILP,IPT
    INTEGER ONUMLP
#endif 
#ifdef KEY_RESIZE
    integer na0,nb0,nt0,np0,ni0,nc0,nd0,nac0,nn0,nr0,ng0,ns0
    na0=natom; nb0=nbond; nt0=ntheta; np0=nphi; ni0=nimphi
    nc0=ncrterm; nd0=ndon; nac0=nacc; nn0=nnb; nr0=nres
    ng0=ngrp; ns0=nseg
#endif
    !
    !...Begin Executable Code. 
    !
    !...Initialize the map array.
    map(0)  = 0
    DO  iAt = 1, nAtom
       map(iAt) = iAt
    enddo

    !...Initialize flag2 
    flag2(0)= 0
    flag2(1:natom) = flag(1:nAtom )
    !
    !...Get copies of the selected atom flags into Index
    index(1:natom) = flag(1:nAtom )
    !
    !...Make a compressed index array of selected atoms
    n = 0 
    DO iAt=1,nAtom
       IF (index(iAt).EQ.1) THEN
          n = n + 1
          index(n) = iAt
       ENDIF
    ENDDO  ! iAt
    !
    !...The number selected.
    nFlag = n 
    expNum = nAtom + (nFlag*nCopy)
    !
    ! MH-14: this will speedup things...
    call allocate_cnst(expnum)

    !...Get the residue membership of the original atoms.
    DO iRs = 1, nRes
       DO iAt = iBase(iRs)+1, iBase(iRs+1)
          rs(iAt) = iRs
       ENDDO ! iAt
    ENDDO ! iRs
    !

    !...Get the group membership of the original atoms.
    DO iGp = 1, nGrp
       DO iAt = iGpBs(iGp)+1, iGpBs(iGp+1)
          grp(iAt) = iGp
       ENDDO ! iAt
    ENDDO ! iGp
    !
    !...Keep track of the original PSF counters. (see psf.f90)
    onAtom  = nAtom 
    onRes   = nRes
    onSeg   = nSeg
    onBond  = nBond
    onTheta = nTheta
    onPhi   = nPhi
    onImPhi = nImPhi
#if KEY_CMAP==1
    onCrTerm = nCrTerm
#endif 
    onNB    = nNB
    onDon   = nDon
    onAcc   = nAcc
    onGrp   = nGrp
    onST2   = nST2
    !
    !lb...B950321 fix of error in IC propagation for Replica 06/29/95
    !     Internal coordinates (if equested)
    IF (qSetIC) olenIC = icr_struct%lenIC
    !lb...
    !...R E P L I C A T E : Loop over the number of replicas 
    DO iSg = 1, nCopy
       !
       !...Increment replica count (global)
       nRepl = nRepl + 1
       !...Identify this replica segment with others of its sub-system 
       repID(nRepl) = nSub

       !...Generate the new segment ID (segNam)
       segNam = newSeg 
       segLen = Len(segNam)
       CALL TrimA(segNam,segLen)
       CALL EncodI(iSg,cSg,Len(cSg),cSgLen) 
       CALL AddSt(segNam,Len(segNam),segLen,cSg,cSgLen)

       IF(qRepDB.and.prnlev >= 2) WRITE(outu,*)  &
            ' Processing Segment=', segNam


       !...Replicate ATOMic properties 
       DO iAt = 1, nFlag
#ifndef KEY_RESIZE
          !...Check for array bounds.
          IF ( nAtom.GE.maxA ) THEN 
             if(prnlev >= 2) write(outu,*)  &
                  ' Current=',nAtom,' Max=',maxA
             CALL WrnDie(-5,'<Replica>', &
                  'PSF: MAXA exceeded - Redimension.' )
          ENDIF
#endif
          nAtom = nAtom + 1
#ifdef KEY_RESIZE
          if (natom .gt. na0) then
             na0=natom+drsz0
             call resize_psf('replica.F90','Repl2','NATOM',na0, .true.)
             call resize_coord('replica.F90','Repl2',na0, .true.)
             call resize_coordc('replica.F90','Repl2',na0, .true.)
             call resize_deriv('replica.F90','Repl2',na0, .true.)
          endif
#endif    

          replAt = nAtom
          origAt = index(iAt)

          IF(qRepDB.and.prnlev >= 2)WRITE(outU,*)  &
               'Atom    old:',origAt
          IF(qRepDB.and.prnlev >= 2)WRITE(outU,*)  &
               'Atom    new:',replAt

          !----------------
          !...PSF ARRAYS.
          !----------------

          !...IUPAC name.
          atype(replAt)  = atype(origAt)
          !...Partial Charge.
          cg(replAt)    = cg(origAt)
          !...Mass.
          aMass(replAt) = aMass(origAt)
          !...Parameter type code. 
          iAC(replAt)   = iAC(origAt)
          !...Fixed Atom Flag.
          iMove(replAt) = iMove(origAt)
          RSCLF(replAt) = RSCLF(origAt)
#if KEY_WCA==1
          WCA(replAt) = WCA(origAt)          
#endif
          !----------------
          !...COOR ARRAYS.
          !----------------

          !...MAIN coordinate set. 
          x(replAt)   = x(origAt)
          y(replAt)   = y(origAt)
          z(replAt)   = z(origAt)
          wMain(replAt)   = wMain(origAt)
          !...COMP coordinate set. 
          xComp(replAt)   = xComp(origAt) 
          yComp(replAt)   = yComp(origAt) 
          zComp(replAt)   = zComp(origAt)
          wComp(replAt)   = wComp(origAt)
          !...REFErence  coordinate set. 
!          if(.not. allocated(refx))  then
!             write(outu,*)"REPLICA>> Possible error, refx being filled before allocated"
!          endif
!!!
!!! Performance bug - MH-14
!!! shuffling zeros :-(
!!!       call allocate_cnst(natom)
!!! Maybe also some other bug ??? Why is refx here ???
!!!
          refX(replAt)   = refX(origAt) 
          refY(replAt)   = refY(origAt) 
          refZ(replAt)   = refZ(origAt)
          !...Energy Contributions
          eCont(replAt)  = eCont(origAt)
          !...Forces
          dX(replAt)   = dX(origAt) 
          dY(replAt)   = dY(origAt) 
          dZ(replAt)   = dZ(origAt)
          !...Velocities ( kept in COMParison set during/following DYNAmics )
          !              vX(replAt)   = vX(origAt) 
          !              vY(replAt)   = vY(origAt) 
          !              vZ(replAt)   = vZ(origAt)
          !...Friction Coefficient
          fBeta(replAt) = fBeta(origAt)
          !...Constraints.
          kCnstr(replAt) = kCnstr(origAt)
          IHSET(replAt ) = IHSET(origAt)
          !...Replica
          repNoA(replAt) = nRepl
          repWt (replAt) = one / nCopy
          !
          !...The atom map. (maps orig atom number onto new...)
          !...map(i) = atom no. of the replicate of i , otherwise it is primary atom no.
          map(origAt)   = replAt 

       ENDDO ! iAt
       !
       !...Duplication of atom list information done for this replica.
       !
       !...Now deal with the topology of this replica
       !
       !...Bonds begin

       DO i = 1, onBond
          IF (BOrTag2(flag2(1),1,iB(i),jB(i))) THEN
             nBond = nBond +1
#ifdef KEY_RESIZE
             if (nbond .gt. nb0) then
                nb0=nbond+drsz0
                call resize_psf('replica.F90','Repl2','NBOND',nb0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nBond.GT.maxB) THEN 
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nBond,' Max=',maxB
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXB exceeded - Redimension.' )
             ENDIF
#endif
             IF (qRepDB) THEN
                if(prnlev >= 2) write(outu,*) 'Bond    old:',iB(i),jB(i)
                if(prnlev >= 2) write(outu,*)  &
                     'Bond    new:',map(iB(i)),map(jB(i))
             ENDIF
             !
             !...Create new bond entry.
             iB(nBond) = map(iB(i))
             jB(nBond) = map(jB(i))

          ENDIF ! (BOrTag2)
       ENDDO  ! i
       !...Bonds end

       !
       !...Angles begin 
       DO i = 1, onTheta
          IF (BOrTag3(flag2(1),1,iT(i),jT(i),kT(i))) THEN
             nTheta = nTheta +1
#ifdef KEY_RESIZE
             if (ntheta .gt. nt0) then
                nt0=ntheta+drsz0
                call resize_psf('replica.F90','Repl2','NTHETA',nt0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nTheta.GT.maxT) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nTheta,' Max=',maxT
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXT exceeded - Redimension.' )
             ENDIF
#endif
             IF (qRepDB)  THEN
                if(prnlev >= 2) write(outu,*)  &
                     'Angle   old:',iT(i),jT(i),kT(i)
                if(prnlev >= 2) write(outu,*) 'Angle   new:', &
                     map(iT(i)),map(jT(i)),map(kt(i))
             ENDIF
             !
             !...Create new angle entry.
             iT(nTheta) = map(iT(i))
             jT(nTheta) = map(jT(i))
             kT(nTheta) = map(kT(i))

          ENDIF ! (BOrTag3)
       ENDDO ! i 
       !...Angles end 

       !
       !...Dihedrals begin 
       DO i = 1, onPhi
          IF (BOrTag4(flag2(1),1,iP(i),jP(i),kP(i),lP(i))) THEN
             nPhi = nPhi + 1
#ifdef KEY_RESIZE
             if (nphi .gt. np0) then
                np0=nphi+drsz0
                call resize_psf('replica.F90','Repl2','NPHI',np0, .true.)
             endif
#else             
             !...Check for array bounds.
             IF (nPhi.GT.maxP) THEN 
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nPhi,' Max=',maxP
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXP exceeded - Redimension.' )
             ENDIF
#endif
             IF (qRepDB)  THEN
                if(prnlev >= 2) write(outu, 418)  &
                     'Dihedral old:',iP(i),jP(i),kP(i),lP(i)
                if(prnlev >= 2) write(outu, 418) 'Dihedral new:', &
                     map(iP(i)),map(jP(i)),map(kP(i)),map(lP(i))
             ENDIF

             !
             !...Create new dihedral entry.
             iP(nPhi) = map(iP(i))
             jP(nPhi) = map(jP(i))
             kP(nPhi) = map(kP(i))
             lP(nPhi) = map(lP(i))

          ENDIF ! (BOrTag4)
       ENDDO ! i
       !...Dihedrals end 

       !
       !...Improper dihedrals begin 
       DO i = 1, onImPhi
          IF (BOrTag4(flag2(1),1,iM(i),jM(i),kM(i),lM(i))) THEN
             nImPhi = nImPhi +1
#ifdef KEY_RESIZE
             if (nimphi .gt. ni0) then
                ni0=nimphi+drsz0
                call resize_psf('replica.F90','Repl2','NIMPHI',ni0, .true.)
             endif
#else             
             !...Check for array bounds.
             IF (nImPhi.GT.maxImp) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nImPhi,' Max=',maxImp
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXIMP exceeded - Redimension.' )
             ENDIF
#endif
             IF (qRepDB)  THEN
                if(prnlev >= 2) write(outu, 418)  &
                     'Improper old:',iM(i),jM(i),kM(i),lM(i)
                if(prnlev >= 2) write(outu, 418) 'Improper new:', &
                     map(iM(i)),map(jM(i)),map(kM(i)),map(lM(i))
             ENDIF
             !
             !...Create new improper dihedral entry.
             iM(nImPhi) = map(iM(i))
             jM(nImPhi) = map(jM(i))
             kM(nImPhi) = map(kM(i))
             lM(nImPhi) = map(lM(i))

          ENDIF ! (BOrTag4)
       ENDDO  ! i
       !...Improper dihedrals end 
418    format (A, 4I8)

#if KEY_CMAP==1
       !...Cross term maps begin 
       DO i = 1, onCrTerm
          IF (BOrTag8(flag2(1),1,i1CT(i),j1CT(i),k1CT(i),l1CT(i) &
               ,i2CT(i),j2CT(i),k2CT(i),l2CT(i))) THEN
             !CC         IF (BOrTag4(flag2(1),1,iM(i),jM(i),kM(i),lM(i))) THEN
             nCrTerm = nCrTerm +1
#ifdef KEY_RESIZE
             if (ncrterm .gt. nc0) then
                nc0=ncrterm+drsz0
                call resize_psf('replica.F90','Repl2','NCRTERM',nc0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nCrTerm.GT.maxCrt) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nCrTerm,' Max=',maxCrt
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXCRT exceeded - Redimension.' )
             ENDIF
#endif
             IF (qRepDB)  THEN
                if(prnlev >= 2) write(outu, 818) 'Cross-term old:', &
                     i1ct(i),j1ct(i),k1ct(i),l1ct(i), &
                     i2ct(i),j2ct(i),k2ct(i),l2ct(i)
                if(prnlev >= 2) write(outu, 818) 'Cross-term new:', &
                     map(i1ct(i)),map(j1ct(i)), &
                     map(k1ct(i)),map(l1ct(i)), &
                     map(i2ct(i)),map(j2ct(i)), &
                     map(k2ct(i)),map(l2ct(i))

             ENDIF
             !
             !...Create new improper dihedral entry.
             i1ct(nCrTerm) = map(i1ct(i))
             j1ct(nCrTerm) = map(j1ct(i))
             k1ct(nCrTerm) = map(k1ct(i))
             l1ct(nCrTerm) = map(l1ct(i))
             i2ct(nCrTerm) = map(i2ct(i))
             j2ct(nCrTerm) = map(j2ct(i))
             k2ct(nCrTerm) = map(k2ct(i))
             l2ct(nCrTerm) = map(l2ct(i))
          ENDIF ! (BOrTag4)
       ENDDO  ! i
       !...Cross term end 
818    format (A, 8I8)
#endif 


       !
       !...Donors begin
       DO i = 1, onDon
          IF (BOrTag2(flag2(1),1,iHD1(i),iDon(i))) THEN
             nDon = nDon +1
#ifdef KEY_RESIZE
             if (ndon .gt. nd0) then
                nd0=ndon+drsz0
                call resize_psf('replica.F90','Repl2','NDON',nd0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nDon.GT.maxPAD) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nDon,' Max=',maxPAD
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXPAD exceeded - Redimension.' )
             ENDIF
#endif
             !...Create new donor entry.
             iHD1(nDon) = map(iHD1(i))
             iDon(nDon) = map(iDon(i))

          ENDIF ! (BOrTag2)
       ENDDO  ! i
       !...Donors end

       !
       !...Acceptors begin
       DO i = 1, onAcc
          IF (BOrTag2(flag2(1),1,iAcc(i),iAc1(i))) THEN
             nAcc = nAcc + 1
#ifdef KEY_RESIZE
             if (nacc .gt. nac0) then
                nac0=nacc+drsz0
                call resize_psf('replica.F90','Repl2','NACC',nac0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nAcc.GT.maxPAD) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nAcc,' Max=',maxPAD
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXPAD exceeded - Redimension.' )
             ENDIF
#endif
             !...Create new acceptor entry.
             iAcc(nAcc) = map(iAcc(i))
             iAc1(nAcc) = map(iAc1(i))

          ENDIF ! (BOrTag2)
       ENDDO  ! i
       !...Acceptors end
       !
#if KEY_LONEPAIR==1
       !...Lonepairs begin
       ! note: only add a new lonepair if the lonepair atom is replicated.
       !       we don't care about the host atoms...
       !
       ONUMLP  = NUMLP
       DO ILP = 1, ONUMLP
          N=LPNHOST(ILP)+1
          IPT=LPHPTR(ILP)
          I=LPHOST(IPT)
          IF(flag2(I).EQ.1) THEN
             !
             !...Check for array bounds.
             IF(NUMLP.GE.MAXLP) THEN
                CALL WRNDIE(-3,'<Replica>', &
                     'Maximum number of lone-pair atoms exceeded')
             ELSE
                IF(NUMLPH+N.GE.MAXLPH) THEN
                   CALL WRNDIE(-3,'<Replica>', &
                        'Maximum number of lone-pair hosts exceeded')
                ELSE
                   !
                   IF (qRepDB) THEN
                      if(prnlev >= 2) write(outu,*) 'Lonepairold:',I      
                      if(prnlev >= 2) write(outu,*) 'Lonepairnew:',map(I)
                   ENDIF
                   !
                   !...Create new lonepair entry.
                   NUMLP = NUMLP +1
                   LPNHOST(NUMLP)=LPNHOST(ILP)
                   LPHPTR(NUMLP)=NUMLPH+1
                   DO I=1,N
                      LPHOST(NUMLPH+I)=MAP(LPHOST(IPT+I-1))
                   ENDDO
                   NUMLPH= NUMLPH+N
                   !
                   LPWGHT(NUMLP)=LPWGHT(ILP)
                   LPVALUE(1,NUMLP)=LPVALUE(1,ILP)
                   LPVALUE(2,NUMLP)=LPVALUE(2,ILP)
                   LPVALUE(3,NUMLP)=LPVALUE(3,ILP)
                ENDIF
             ENDIF
          ENDIF
       ENDDO  ! ilp
       !...Lonepairs end
#endif 
       !
       !...Internal coordinates (if requested) 
       IF (qSetIC) THEN
          !lb...B950321 fix of error in IC propagation for Replica 06/29/95
          !     Internal coordinates (if equested)
          !        olenIC = lenIC
          !lb...
          !...Internal coordinates begin.
          DO i = 1, olenIC
             CALL GETICEL(II,JJ,KK,LL,TT,BB1,BB2,TT1,TT2,PP,I, &
                  icr_struct%B1ic,icr_struct%B2ic, &
                  icr_struct%T1ic,icr_struct%T2ic, &
                  icr_struct%PIC, icr_struct%IAR, &
                  icr_struct%JAR, icr_struct%KAR, &
                  icr_struct%LAR, icr_struct%TAR)
             IF (BOrTag4(flag2(1),1,II,JJ,KK,LL)) THEN
                !
                !...Check for array bounds.
                IF(icr_struct%intlen.LE.icr_struct%lenic) THEN  ! get more space...
                   IRSZ=icr_struct%lenic+200
                   CALL reintc_new(IRSZ,icr_struct)
                Endif
                icr_struct%lenIC = icr_struct%lenIC +1
                !
                !...Create new internal coordinate entries.
                IF(II.gt.0) II = map(II)
                IF(JJ.gt.0) JJ = map(JJ)
                IF(KK.gt.0) KK = map(KK)
                IF(LL.gt.0) LL = map(LL)
                CALL PUTICEL(II,JJ,KK,LL,TT,BB1,BB2,TT1,TT2,PP,&
                     icr_struct%lenic, &
                     icr_struct%B1ic,icr_struct%B2ic, &
                     icr_struct%T1ic,icr_struct%T2ic, &
                     icr_struct%PIC, icr_struct%IAR, &
                     icr_struct%JAR, icr_struct%KAR, &
                     icr_struct%LAR, icr_struct%TAR)

             ENDIF ! (BOrTag4)
          ENDDO ! i 
          !...Internal coordinates end 
       ENDIF ! (qSetIC)
       !
       ! Explicit non-bonded exclusions (begin)
       DO i = 1, nFlag
          IF (index(i).GT.1) THEN
             iAt = iBlo(index(i)-1) + 1
          ELSE
             iAt = i
          ENDIF ! (index(i).GT.1)
          !
          DO j = iAt, iBlo(index(i))

             nNB = nNB + 1
#ifdef KEY_RESIZE
             if ((nnb .gt. nn0) .and. (nnb .gt. natom/2)) then
                nn0=nnb+drsz0
                call resize_psf('replica.F90','Repl2','NNB',nn0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nNB.GT.maxNB) THEN 
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nNB,' Max=',maxNB
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXNB exceeded - Redimension.' )
             ENDIF
#endif
             iNB(nNB) = map(iNB(j))
             !
          ENDDO ! j
          !         
          iBlo(map(index(i))) = nNB
          !
       ENDDO ! i 
       ! Explicit non-bonded exclusions (end)

       !
       !...Resid list (begin)
       iRs = -99
       DO i = 1, nFlag
          IF (rs(index(i)).NE.iRs) THEN
             nRes = nRes + 1
#ifdef KEY_RESIZE
             if (nres .gt. nr0) then
                nr0=nres+drsz0
                call resize_psf('replica.F90','Repl2','NRES',nr0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nRes.GT.maxRes) THEN 
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nRes,' Max=',maxRes
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXRES exceeded - Redimension.' )
             ENDIF
#endif
             iRs = rs(index(i))
             iBase(nRes) = map(index(i)) - 1
             !...residue name
             res(nRes) = res(iRs)
             !...resid
             resID(nRes) = resID(iRs)
             !
             !...check code
             IF (iBase(nRes).NE.((iSg-1)*nFlag)+onAtom+i-1)  &
                  CALL WrnDie(-5,'<Replica>','Programming Error')

          ENDIF ! (rs(index(i)).NE.iRs)
          !
       ENDDO ! i
       !
       iBase(nRes+1)=nAtom
       !
       !...Resid list (end)
       !
       !...Group list (begin)
       iGp = -99
       DO i = 1, nFlag
          IF (grp(index(i)).NE.iGp) THEN
             nGrp = nGrp + 1
#ifdef KEY_RESIZE
             if (ngrp .gt. ng0) then
                ng0=ngrp+drsz0
                call resize_psf('replica.F90','Repl2','NGRP',ng0, .true.)
             endif
#else
             !...Check for array bounds.
             IF (nGrp.GT.maxGrp) THEN
                if(prnlev >= 2) write(outu,*)  &
                     ' Current=',nGrp,' Max=',maxGrp
                CALL WrnDie(-5,'<Replica>', &
                     'PSF: MAXGRP exceeded - Redimension.' )
             ENDIF
#endif
             iGp = grp(index(i))
             iGpBs(nGrp) = map(index(i)) - 1
             !
             !...Group-Type and Group Move flag.
             iGpTyp(nGrp) = iGpTyp(iGp)
             iMoveG(nGrp) = iMoveG(iGp)
             !LSDC Replica
             repNoG(nGrp) = nRepl
             !
             !...check code
             IF (iGpBs(nGrp).NE.((iSg-1)*nFlag)+onAtom+i-1) &
                  CALL WrnDie(-5,'<Replica>','Programming Error')

          ENDIF ! (grp(index(i)).NE.onGrp)

       ENDDO ! i

       iGpBs(nGrp+1)=nAtom
       !
       !...Group list (end)
       !
       !...Update the segment extent (in terms of residues) array.
       nSeg = nSeg + 1
#ifdef KEY_RESIZE
       if (nseg .gt. ns0) then
          ns0=nseg+drsz0
          call resize_psf('replica.F90','Repl2','NSEG',ns0, .true.)
       endif
#else       
       !
       !...Check for array bounds.
       IF (nSeg.GT.maxSeg) THEN 
          if(prnlev >= 2) write(outu,*)  &
               ' Current=',nSeg,' Max=',maxSeg
          CALL WrnDie(-5,'<Replica>', &
               'PSF: MAXSEG exceeded - Redimension.' )
       ENDIF
#endif
       !...segid
       segID(nSeg) = segNam
       nICTot(nSeg+1) = nRes

    ENDDO ! iSg
#ifdef KEY_RESIZE
    if (natom .lt. na0) then
       call resize_psf('replica.F90','Repl2','NATOM',natom, .true.)
       call resize_coord('replica.F90','Repl2',natom, .true.)
       call resize_coordc('replica.F90','Repl2',natom, .true.)
       call resize_deriv('replica.F90','Repl2',natom, .true.)
    endif
    if (nbond .lt. nb0) &
         call resize_psf('replica.F90','Repl2','NBOND',nbond, .true.)
    if (ntheta .lt. nt0) &
         call resize_psf('replica.F90','Repl2','NTHETA',ntheta, .true.)
    if (nphi .lt. np0) &
         call resize_psf('replica.F90','Repl2','NPHI',nphi, .true.)
    if (nimphi .lt. ni0) &
         call resize_psf('replica.F90','Repl2','NIMPHI',nimphi, .true.)         
    if (ncrterm .lt. nc0) &
         call resize_psf('replica.F90','Repl2','NCRTERM',ncrterm, .true.)         
    if (natom .lt. nd0) & ! set to natom
         call resize_psf('replica.F90','Repl2','NDON',natom, .true.)         
    if (natom .lt. nac0) &
         call resize_psf('replica.F90','Repl2','NACC',natom, .true.)         
!    if (nnb .lt. nn0)  & ! resizing done via NATOM
!         call resize_psf('replica.F90','Repl2','NNB',nnb, .true.)         
    if (nres .lt. nr0) &
         call resize_psf('replica.F90','Repl2','NRES',nres, .true.)         
    if (ngrp .lt. ng0) &
         call resize_psf('replica.F90','Repl2','NGRP',ngrp, .true.)         
    if (nseg .lt. ns0) &
         call resize_psf('replica.F90','Repl2','NSEG',nseg, .true.)         
#endif
    
    IF (nAtom .NE. expNum)  &
         CALL WrnDie(-3,'<Replica>','Programming Error')

    ! DEBUG from now on
    !
    !...Get the group membership of the atoms 
    !      DO 202 iGp = 1, nGrp
    !            WRITE(outu,*) ' Group=', iGp, ' Atom Range=',
    !     &                    iGpBs(iGp)+1, iGpBs(iGp+1),
    !     &                    iGpBs(iGp+1) - iGpBs(iGp)+1
    !202    CONTINUE
    !
    !...Get the residue membership of the atoms 
    !      DO 201 iRs = 1, nRes
    !            WRITE(outu,*) ' Residue=', iRs, ' Atom Range=',
    !     &                    iBase(iRs)+1, iBase(iRs+1),
    !     &                    iBase(iRs+1) - iBase(iRs)+1
    !201    CONTINUE
    !
    !...Get the segment membership of the residues 
    !      DO 200 iSg = 1, nSeg
    !            WRITE(outu,*) ' Segment=', iSg, ' Residue Range=',
    !     &                    nICTot(iSg)+1, nICTot(iSg+1), 
    !     &                    nICTot(iSg+1) - nICTot(iSg)+1
    !200    CONTINUE

    !...Exit.  
    RETURN
  END subroutine Repl2

  SUBROUTINE RepReset
    !-----------------------------------------------------------------------
    !# <caves>-Feb-10-1993 (Leo Caves)
    !
    !...Reset the Replica data structure
    !
    use chm_kinds
    !...Global Variables.
    use dimens_fcm
    use psf
    !...  use replica_mod
    use number
    implicit none
    !
    !...Passed Variables.
    !
    !...Local Variables. 
    !
    INTEGER i
    !...Begin Executable Code. 
    !
    nSub = 1
    nRepl = 1
    repID(nSub) = 1
    !      CALL VSetI(repNoG,  1, nGrp)
    !      CALL VSetI(repNoA,  1, nAtom)
    !      CALL VSetF(repWt ,one, nAtom)
    !# <caves>-Aug-4-1993 (Leo Caves) not adding my vector libraries: so inline.
    DO i = 1, nGrp
       repNoG(i) = 1
    ENDDO
    DO i = 1, nAtom
       repNoA(i) = 1
       repWt(i) = one
    ENDDO
    !
    !...Exit.
    RETURN
  END SUBROUTINE RepReset

  LOGICAL FUNCTION BOrTag2 ( tagArr, tagFlg, id1, id2 )
    !-----------------------------------------------------------------------
    !# <caves>-May-13-1991 (Leo Caves)
    !
    !...Performs boolean .OR. on 2 (given) elements of tagArr.
    !...ie IF (tagArr(id1).EQ.tagFlg .OR. tagArr(id2).EQ.tagFlg) BOrTag2 = .TRUE. 
    !
    use chm_kinds
    implicit none
    !...Passed Variables.
    INTEGER tagArr(*), tagFlg, id1, id2
    !
    !...Begin Executable Code. 
    !
!    BOrTag2 = tagArr(id1).EQ.tagFlg .OR. tagArr(id2).EQ.tagFlg 
    BOrTag2 = .FALSE.
    IF(id1.gt.0) BOrTag2 = (tagArr(id1).EQ.tagFlg) .or. BOrTag2
    IF(id2.gt.0) BOrTag2 = (tagArr(id2).EQ.tagFlg) .or. BOrTag2

    !...Exit.
    RETURN
  END FUNCTION BOrTag2

  LOGICAL FUNCTION BOrTag3 ( tagArr, tagFlg, id1, id2, id3 )
    !-----------------------------------------------------------------------
    !# <caves>-May-13-1991 (Leo Caves)
    !
    !...Performs boolean .OR. on 3 (given) elements of tagArr.
    !
    use chm_kinds
    implicit none
    !...Passed Variables.
    INTEGER tagArr(*), tagFlg, id1, id2, id3
    !
    !...Begin Executable Code. 
    !
!    BOrTag3 = tagArr(id1).EQ.tagFlg .OR. tagArr(id2).EQ.tagFlg .OR. &
!         tagArr(id3).EQ.tagFlg 
    BOrTag3 = .FALSE.
    IF(id1.gt.0) BOrTag3 = (tagArr(id1).EQ.tagFlg) .or. BOrTag3
    IF(id2.gt.0) BOrTag3 = (tagArr(id2).EQ.tagFlg) .or. BOrTag3
    IF(id3.gt.0) BOrTag3 = (tagArr(id3).EQ.tagFlg) .or. BOrTag3

    !...Exit.
    RETURN
  END FUNCTION BOrTag3

  LOGICAL FUNCTION BOrTag4 ( tagArr, tagFlg, id1, id2, id3, id4 )
    !-----------------------------------------------------------------------
    !# <caves>-May-13-1991 (Leo Caves)
    !
    !...Performs boolean .OR. on 4 (given) elements of tagArr.
    !
    use chm_kinds
    implicit none
    !...Passed Variables.
    INTEGER tagArr(*), tagFlg, id1, id2, id3, id4
    !
    !...Begin Executable Code. 
    !
    BOrTag4 = .FALSE.
    IF(id1.gt.0) BOrTag4 = (tagArr(id1).EQ.tagFlg) .or. BOrTag4
    IF(id2.gt.0) BOrTag4 = (tagArr(id2).EQ.tagFlg) .or. BOrTag4
    IF(id3.gt.0) BOrTag4 = (tagArr(id3).EQ.tagFlg) .or. BOrTag4
    IF(id4.gt.0) BOrTag4 = (tagArr(id4).EQ.tagFlg) .or. BOrTag4
    !
    ! old code
    !C      BOrTag4 = tagArr(id1).EQ.tagFlg .OR. tagArr(id2).EQ.tagFlg .OR.
    !C     &          tagArr(id3).EQ.tagFlg .OR. tagArr(id4).EQ.tagFlg 
    ! 
    !...Exit.
    RETURN
  END FUNCTION BOrTag4

#if KEY_CMAP==1

  LOGICAL FUNCTION BOrTag8 ( tagArr, tagFlg, id1, id2, id3, id4, &
       id5, id6, id7, id8)
    !-----------------------------------------------------------------------
    !...Performs boolean .OR. on 8 (given) elements of tagArr.
    !
    use chm_kinds
    implicit none
    !...Passed Variables.
    INTEGER tagArr(*), tagFlg, id1, id2, id3, id4, &
         id5,id6,id7,id8

    BOrTag8 = .FALSE.
    IF(id1.gt.0) BOrTag8 = (tagArr(id1).EQ.tagFlg) .or. BOrTag8
    IF(id2.gt.0) BOrTag8 = (tagArr(id2).EQ.tagFlg) .or. BOrTag8
    IF(id3.gt.0) BOrTag8 = (tagArr(id3).EQ.tagFlg) .or. BOrTag8
    IF(id4.gt.0) BOrTag8 = (tagArr(id4).EQ.tagFlg) .or. BOrTag8
    IF(id5.gt.0) BOrTag8 = (tagArr(id5).EQ.tagFlg) .or. BOrTag8
    IF(id6.gt.0) BOrTag8 = (tagArr(id6).EQ.tagFlg) .or. BOrTag8
    IF(id7.gt.0) BOrTag8 = (tagArr(id7).EQ.tagFlg) .or. BOrTag8
    IF(id8.gt.0) BOrTag8 = (tagArr(id8).EQ.tagFlg) .or. BOrTag8
    RETURN
  END FUNCTION BOrTag8

#endif 
#endif 

end module replica_mod

