#if KEY_MMFF==1
!  =====================================================================
SUBROUTINE ADDH(IATOM,MODE,nh,NADDH,NBORS, &
     NATOM,X,Y,Z,MTYPE,ATNUM,IB,JB,BondType,NBOND, &
     ITAB,AtNames,RESNAME,KRES)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "IT" to "TYP" to avoid
  !                       conflict with CHARMM name
  !  27 Nov 95 Jay Banks: replaced XYZ array with X,Y,Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  !...  use mmffm
  use string
  use stream
  implicit none
  !
  integer ATNUM(*)
  integer IB(*),JB(*)
  integer BondType(*)
  integer MTYPE(*)
  integer NATOM
  integer NBOND
  real(chm_real)  X(*),Y(*),Z(*)
  integer ITAB(6,NATOM)
  character(len=*) AtNames(*)
  character(len=*) RESNAME(*)
  character(len=*) KRES(*)
  !
  CHARACTER(len=4) HNAME
  real(chm_real)  adrot, bang, hposx(3), hposy(3), hposz(3)
  integer iang, iatom
  integer TYP, ita, ityp, jh
  integer kkount, maxseq, mode, NADDH, nb
  integer nbors ,ndbl, newh, newseq
  integer nh, ntrpl
  !
  KKOUNT=NBORS
  ITA=ATNUM(IATOM)
  !   KKOUNT IS THE NUMBER OF NEIGHBORS OF IATOM
  BANG=109.5
  ITYP=1
  IANG=3
  ADROT=0.
  TYP=MTYPE(IATOM)
  !   ADD H'S BASED ON WHAT TYPE OF ATOM THIS IS
  IF(MODE.EQ.0) THEN
     !
     !   ASSIGN "ROUGH" ATOM TYPE
     !
     NDBL=NBND2(IATOM,ITAB)
     NTRPL=NBND3(IATOM,ITAB)
     TYP=1
     IF(NDBL.EQ.1) TYP=2
     IF(NDBL.EQ.2.OR.NTRPL.EQ.1) TYP=4
     !       WRITE(6,*) ' ATOM',IATOM,'  TYP =',TYP,' nh =',nh
  ENDIF
  IF(MODE.EQ.0.OR.MODE.EQ.1) THEN
     ! ADD H OR H'S TO CARBON
     IF(TYP.EQ.1) GOTO (400,1100,1300),KKOUNT
     IF(TYP.EQ.2.OR.TYP.EQ.3.OR.TYP.EQ.37) THEN
        GOTO (1000,1200) KKOUNT
        GOTO 100
     ENDIF
     IANG=1
     BANG=180.
     IF(TYP.EQ.4) GOTO 1400
  ELSE IF(MODE.EQ.2) THEN
     !  HETEROATOM
     IF((TYP.EQ.6.OR.TYP.EQ.8.OR.TYP.EQ.15.OR.TYP.EQ.34).AND. &
          KKOUNT.EQ.1) GOTO 400
     IF((TYP.EQ.9.OR.TYP.EQ.10).AND.KKOUNT.EQ.1) GOTO 1000
     IF((TYP.EQ.10.OR.TYP.EQ.38.OR.TYP.EQ.39).AND. &
          KKOUNT.EQ.2)GOTO 1200
     IF((TYP.EQ.8.OR.TYP.EQ.34).AND.KKOUNT.EQ.2) GOTO 1100
     IF(TYP.EQ.34.AND.KKOUNT.EQ.3) GOTO 1300
  ENDIF
  !   IF STILL HERE, ERROR.
100 if(wrnlev.gt.2) WRITE(OUTU,200) TYP,KKOUNT,nh
  !     IF(IZ.NE.0) WRITE(IZ,200) TYP,KKOUNT,nh
200 FORMAT(/' H ADDITION NOT PROGRAMMED FOR ATOM TYPE',I3,' WITH', &
       I3,' NEIGHBORS AND',I3, ' H MISSING')
  if(wrnlev.gt.2) WRITE(OUTU,300) IATOM
  !     IF(IZ.NE.0) WRITE(IZ,300) IATOM
300 FORMAT(' POSSIBLE ERROR FOR ATOM',I4,' ... CHECK DATA')
  GOTO 1500
400 CONTINUE
  !   IF CAME HERE DIRECTLY, ADD nh H'S TO A TERMINAL SINGLY-BONDED
  !   ATOM, SUCH AS AN ALCOHOL O OR A PRIMARY AMINE N
  !     WRITE(6,*) ' HADD',IATOM,ITYP,nh,KKOUNT,IANG,BANG
  !D     WRITE(OUTU,*) ' HADD',IATOM,ITYP,nh,KKOUNT,IANG,BANG
  !     IF(IZ.NE.0) WRITE(IZ,*) ' HADD',IATOM,ITYP,nh,KKOUNT,IANG,BANG
  CALL HIDEAL(IATOM,ITYP,IANG,ADROT,BANG,HPOSX,HPOSY,HPOSZ,X,Y,Z, &
       NATOM,nh,ATNUM)
500 CONTINUE
  MAXSEQ=NATOM+NADDH
  !     WRITE(6,*) ' IATOM,nh,MAXSEQ',IATOM,nh,MAXSEQ
  DO JH=1,nh
     NEWH=NATOM+JH+NADDH
     ATNUM(NEWH)=1
     MTYPE(NEWH)=0
     nb=NBOND+JH
     IB(nb)=IATOM
     JB(nb)=NEWH
     BondType(nb)=1
     X(NEWH)=HPOSX(JH)
     Y(NEWH)=HPOSY(JH)
     Z(NEWH)=HPOSZ(JH)
     HNAME='H   '
     HNAME(2:2)='X'
     IF(ITA.EQ.6) HNAME(2:2)='C'
     IF(ITA.EQ.7) HNAME(2:2)='N'
     IF(ITA.EQ.8) HNAME(2:2)='O'
     IF(ITA.EQ.15) HNAME(2:2)='P'
     IF(ITA.EQ.16) HNAME(2:2)='S'
     !
     NEWSEQ=0 ! what should be this value ? RCZ940202
     !
     IF(NEWSEQ.LT.10) THEN
        WRITE(HNAME(3:3),'(I1)') NEWSEQ
     ELSE IF(NEWSEQ.LT.100) THEN
        WRITE(HNAME(3:4),'(I2)') NEWSEQ
     ELSE IF(NEWSEQ.LT.1000) THEN
        WRITE(HNAME(2:4),'(I3)') NEWSEQ
     ELSE IF(NEWSEQ.GE.1000) THEN
        WRITE(SCRTCH,'(A,I8,A)') &
             'ATTEMPT TO CREATE H ATOM ',NEWSEQ,' DISALLOWED'
        !        IF(IZ.NE.0) WRITE(IZ,600) NEWSEQ
        CALL WRNDIE(-5,'<addh>',SCRTCH(:44))
     ENDIF
     AtNames(NEWH)=HNAME
     RESNAME(NEWH)=RESNAME(IATOM)
     KRES(NEWH)=KRES(IATOM)
     !     IF(IZ.NE.0) WRITE(IZ,800) NEWSEQ,X(NEWH),Y(NEWH),Z(NEWH),IATOM,
     !    .  X(IATOM),Y(IATOM),Z(IATOM)
800  FORMAT(' NEWH',I4,3F8.3,I4,3F8.3)
  ENDDO
  NBOND=NBOND+nh
  !     NENTRY=NBOND
  NADDH=NADDH+nh
  GOTO 1500
1000 CONTINUE
  !   ADD nh H'S TO A TERMINAL DOUBLY BONDED ATOM, SUCH AS AN IMINE N
  IANG=2
  BANG=120.
  GOTO 400
1100 CONTINUE
  !   ADD nh H'S TO A SECONDARY AMINE OR PHOSPHINE OR QUATERNARY N+
  ITYP=3
  GOTO 400
1200 CONTINUE
  !   ADD nh=1 H'S TO AN AMIDE NITROGEN
  ITYP=4
  GOTO 400
1300 CONTINUE
  !   ADD METHINE OR NR3+ H
  ITYP=5
  GOTO 400
1400 CONTINUE
  !   ADD ACETYLENIC H
  ITYP=1
  GOTO 400
1500 CONTINUE
  RETURN
END SUBROUTINE ADDH

! ======================================================================
SUBROUTINE BONDTB
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use stream
  implicit none
  !
  integer i
  integer j, l, m
  integer IATJ(4), index(4)

  do m=1,4
     index(M)=M
  enddo
  !
  !  USES IAT TABLE in psf.f90.
  !
  DO J=1,NATOM
     L=0
     DO M=1,4
        IF(ITAB(M,J).GT.0) then
           L=M
           IATJ(M)=ITAB(M,J)
        endif
     ENDDO
     ! T.A. Halgren change: fix error in logic for case IC <= 0 (B980629.wy)
     !     if(L.le.0 .or. L.gt.4)
     if(L.gt.4) &
          ! end of T.A. Halgren change
          call wrndie(-5,'<bondtb>','L.le.0 .or. L.gt.4')
     ITAB(6,J)=L
     LOCAT(J)=L
     !     CALL KSORTS(L,4,IATJ)
     !     write(outu,'(a,4i5)') ' bondtb> L,IATJ =',L,(IATJ(I),I=1,L)
     call sorth(L,iatj,index)
     !     write(outu,'(a,4i5)') ' bondtb> L,IATJ =',L,(IATJ(I),I=1,L)
     DO I=1,L
        !     IAT(I,J)=IATJ(I)
        ITAB(I,J)=IATJ(I)
     ENDDO
     !     WRITE(6,350) J,(IAT(I,J),I=1,L)
     !     WRITE(12,350) J,(IAT(I,J),I=1,L)
     ! 350 FORMAT(' ATOM',I4,'   BONDED TO ',4I4)
  ENDDO
  !
  !  now setup the bonds list in BondList(,)
  RETURN
END SUBROUTINE BONDTB

subroutine GenerateRings ! generates ring information
  !
  !  Jay Banks 22 Nov 95: replaced in-line implicit none with ##INCLUDE
  !
  use chm_kinds
  use memory
  use dimens_fcm
  !...##INCLUDE '~/charmm_fcm/debug.f90'
  use exfunc
  use psf
  use stream
  use string
  use mmffm
  implicit none
  !
  logical,allocatable,dimension(:) :: bmask
  integer,allocatable,dimension(:) :: paths
  integer i,ier
  !
  IRINGS=0
  call chmalloc('datastruc.src','GenerateRings','bmask',NBOND,log=bmask)
  call chmalloc('datastruc.src','GenerateRings','paths', &
       2*MAXPATHS*(MAX_TO_SEARCH-1),intg=paths)
  call ring_find(NATOM,NBOND,MAX_TO_SEARCH,IB,JB,ier, &
       bmask,paths)
  call chmdealloc('datastruc.src','GenerateRings','bmask',NBOND,log=bmask)
  call chmdealloc('datastruc.src','GenerateRings','paths', &
       2*MAXPATHS*(MAX_TO_SEARCH-1),intg=paths)

  !...##IF DEBUG
  !      call getenv('DEBUG_GenerateRings',SCRTCH)
  !      DEBUG=SCRTCH.ne.' '
  !      if(DEBUG) then
  !         write(OUTU,'(a,i5)') 'DEBUG_GenerateRings> IRINGS=',IRINGS
  !         do i=1,NATOM
  !            write(OUTU,'(a,2i5)')
  !     &      'DEBUG_GenerateRings> atom,INRING=',I,INRING(I)
  !         enddo
  !      endif
  !...##ENDIF
  return
end subroutine GenerateRings

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!
!       SUBROUTINE RING_FIND: FIND CONNECTED RING SYSTEMS IN A MOLECULE.
!
!       WRITTEN BY JAY L. BANKS OF MOLECULAR SIMULATIONS, INC., AT AND
!       FOR MERCK & CO., INC.
!
!       NOTE: THE IDEA BEHIND THE 'PATHS' DATA STRUCTURE -- REPRESENTING
!       A PATH OF LENGTH 'N' AS A PAIR OF PATHS OF LENGTH 'N-1' -- WAS
!       MADE KNOWN TO THE AUTHOR THROUGH THE WORK OF BROND LARSON OF
!       THINKING MACHINES CORP., AND SHOULD *NOT* BE CONSIDERED THE
!       PROPERTY OF EITHER MOLECULAR SIMULATIONS, INC., OR MERCK & CO., INC.
!       THE IMPLEMENTATION OF THIS STRUCTURE USING SIGNED INTEGERS TO
!       INDICATE THE DIRECTION OF TRAVERSAL OF THE PATH OF LENGTH 'N-1',
!       AND THE USE OF RECURSIVE ROUTINES TO EXPAND A PATH OF ARBITRARY
!       SIZE TO ITS CONSTITUENT ATOMS, WERE DEVELOPED AT MERCK BY T. HALGREN
!       AND THE AUTHOR.
!
!      ARGUMENTS (FOR OPTIMOL/MMFF/CHARMM IMPLEMENTATION)
!      natoms         integer            input         number of atoms
!      nbonds         integer            input         number of bonds
!      maxsize        integer            input         largest ring to find
!      IBOND          integer array      input         first atom in bond
!      JBOND          integer array      input         second atom in bond
!      IRINGS         integer            output        number of ring systems
!      INRING         integer array      output        ring index per atom
!      ier            integer            output        error code
!
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

subroutine ring_find(natoms, nbonds, maxsize, IBOND, JBOND, &
     ier, bmask, paths)
  !
  !  Jay Banks 22-Nov-95: added IF (PRNLEV.GE.2) to WRITE statement.
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2),
  !  and nested inside other IF (rather than .AND. in one IF), which
  !  requires action whether WRITE executed or not.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.  Likewise "end do"s and "end if"s.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use mmffm
  implicit none

  integer natoms, nbonds, maxsize
  integer IBOND(*), JBOND(*)
  integer  ier
  logical bmask(*)
  integer paths(2, MAXPATHS, 3:*)
  integer atoms_left
  integer i, j
  integer npaths
  logical has_any

  ier = 0
  do i = 1, natoms
     INRING(i) = 0
  enddo
  do i = 1, nbonds
     bmask(i) = .true.
  enddo
  atoms_left = natoms
  call strip_term(natoms, nbonds, IBOND, JBOND, bmask, atoms_left)
  if (atoms_left .le. 0) then
     !       no atoms not stripped off: graph is acyclic
     ier = 2
     return
  endif

  !       FIND ALL PAIRS OF BONDS WITH AN ATOM IN COMMON, AND STORE THEM AS
  !       PATHS OF LENGTH THREE
  npaths = 0
  do i = 1, nbonds-1
     do j = i+1, nbonds
        if (bmask(i) .and. bmask(j) .and. &
             (IBOND(i) .eq. IBOND(j) .or. &
             JBOND(i) .eq. IBOND(j) .or. &
             IBOND(i) .eq. JBOND(j) .or. &
             JBOND(i) .eq. JBOND(j))) &
             then                        ! have found a path of length 3
           if (npaths .ge. MAXPATHS) then
              IF (WRNLEV.GE.2) THEN
                 write(OUTU, 6001)
6001             format('ring_find: Path list full for length 3')
              ENDIF
              goto 20
           endif
           npaths = npaths + 1
           !     mark "backwards" bond with negative index.
           !     CONVENTION: bond "i" (lower index) is ALWAYS traversed first
           if (IBOND(i) .eq. IBOND(j)) then
              !             use atom order (2, i), (1, i) = (1, j), (2, j)
              paths(1, npaths, 3) = -i
              paths(2, npaths, 3) = j
           else if (JBOND(i) .eq. IBOND(j)) then
              !             use atom order (1, i), (2, i) = (1, j), (2, j)
              paths(1, npaths, 3) = i
              paths(2, npaths, 3) = j
           else if (IBOND(i) .eq. JBOND(j)) then
              !             use atom order (2, i), (1, i) = (2, j), (1, j)
              paths(1, npaths, 3) = -i
              paths(2, npaths, 3) = -j
           else if (JBOND(i) .eq. JBOND(j)) then
              !             use atom order (1, i), (2, i) = (2, j), (1, j)
              paths(1, npaths, 3) = i
              paths(2, npaths, 3) = -j
           endif
        endif
     enddo          !inner loop over bonds
  enddo            !outer loop over bonds
20 continue        !break out of above loops
  do j = 3, maxsize
     call get_next_size(j, natoms, IBOND, JBOND, paths, npaths)
  enddo                          ! loop on ring sizes

  !     clear out empty ring systems
  i = 1
  do while (i .le. IRINGS)
     has_any = .false.
     j = 1
     do while (j .le. natoms .and. (.not. has_any))
        if (INRING(j) .eq. i) then
           has_any = .true.
        endif
        j = j + 1
     enddo
     if (.not. has_any) then
        do j = 1, natoms
           if (INRING(j) .gt. i) then
              INRING(j) = INRING(j) - 1
           endif
        enddo
        IRINGS = IRINGS - 1
     else
        i = i + 1
     endif
  enddo

  return
end subroutine ring_find
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!
!       SUBROUTINE STRIP_TERM: STRIP "TERMINAL" ATOMS AND BONDS FROM A MOLECULE.
!
!       WRITTEN BY JAY L. BANKS OF MOLECULAR SIMULATIONS, INC., AT AND
!       FOR MERCK & CO., INC.

!      ARGUMENTS (FOR OPTIMOL/MMFF/CHARMM IMPLEMENTATION)
!      natoms          integer            input            number of atoms
!      nbonds          integer            input            number of bonds
!      IBOND           integer array      input            first atom in bond
!      JBOND           integer array      input            second atom in bond
!      bmask           logical array      output           mask for bonds left
!      atoms_left      integer            output           number of atoms left
!
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!
! 28 Nov 95 Jay Banks: took spaces out of "end do"s and "end if"s.
!
subroutine strip_term(natoms, nbonds, IBOND, JBOND, &
     bmask, atoms_left)
  use memory
  use chm_kinds
  use dimens_fcm
  implicit none

  integer natoms, nbonds, atoms_left
  integer IBOND(*), JBOND(*)
  logical bmask(*)

  integer,allocatable,dimension(:) :: nnbr
  integer i, nterm

  call chmalloc("datastruc.src","strip_term","nnbr",natoms,intg=nnbr)

  do i = 1, natoms
     nnbr(i) = 0
  enddo
  do i = 1, nbonds
     if (bmask(i)) then
        nnbr(IBOND(i)) = nnbr(IBOND(i)) + 1
        nnbr(JBOND(i)) = nnbr(JBOND(i)) + 1
     endif
  enddo

  nterm = 0
  do i = 1, natoms
     if (nnbr(i) .eq. 1) then
        nterm = nterm + 1
     endif
  enddo

  do while (nterm .gt. 0 .and. atoms_left .gt. 0)
     do i = 1, nbonds
        if (bmask(i) .and. (nnbr(IBOND(i)) .eq. 1 .or. &
             nnbr(JBOND(i)) .eq. 1)) &
             then
           bmask(i) = .false.
           !       Decrement number of atoms for EACH "terminal" atom in the current bond.
           !       In other words, if BOTH atoms are terminal (the molecule consists
           !       of only the two atoms connected by the current bond), decrement number
           !       of atoms by 2.
           if (nnbr(IBOND(i)) .eq. 1) then
              atoms_left = atoms_left - 1
           endif
           if (nnbr(JBOND(i)) .eq. 1) then
              atoms_left = atoms_left - 1
           endif
           nnbr(IBOND(i)) = nnbr(IBOND(i)) - 1
           nnbr(JBOND(i)) = nnbr(JBOND(i)) - 1
        endif
     enddo
     nterm = 0
     do i = 1, natoms
        if (nnbr(i) .eq. 1) then
           nterm = nterm + 1
        endif
     enddo
  enddo
  call chmdealloc("datastruc.src","strip_term","nnbr",natoms,intg=nnbr)

  return
end subroutine strip_term
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!
!       SUBROUTINE GET_NEXT_SIZE: FIND PATHS OF LENGTH 'N+1.'
!
!       WRITTEN BY JAY L. BANKS OF MOLECULAR SIMULATIONS, INC., AT AND
!       FOR MERCK & CO., INC.
!
!       NOTE: THE IDEA BEHIND THE 'PATHS' DATA STRUCTURE -- REPRESENTING
!       A PATH OF LENGTH 'N' AS A PAIR OF PATHS OF LENGTH 'N-1' -- WAS
!       MADE KNOWN TO THE AUTHOR THROUGH THE WORK OF BROND LARSON OF
!       THINKING MACHINES CORP., AND SHOULD *NOT* BE CONSIDERED THE
!       PROPERTY OF EITHER MOLECULAR SIMULATIONS, INC., OR MERCK & CO., INC.
!       THE IMPLEMENTATION OF THIS STRUCTURE USING SIGNED INTEGERS TO
!       INDICATE THE DIRECTION OF TRAVERSAL OF THE PATH OF LENGTH 'N-1',
!       AND THE USE OF RECURSIVE ROUTINES TO EXPAND A PATH OF ARBITRARY
!       SIZE TO ITS CONSTITUENT ATOMS, WERE DEVELOPED AT MERCK BY T. HALGREN
!       AND THE AUTHOR.
!
!
!      ARGUMENTS (FOR OPTIMOL/MMFF/CHARMM IMPLEMENTATION)
!      cursize        integer            input           ring size to find
!      natoms         integer            input           number of atoms
!      IBOND          integer array      input           first atom in bond
!      JBOND          integer array      input           second atom in bond
!      paths          integer array      both            list of paths
!      npaths         integer            both            number at current size
!      IRINGS         integer            output          number of ring systems
!      INRING         integer array      output          ring index per atom
!
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

subroutine get_next_size(cursize, natoms, IBOND, JBOND, &
     paths, npaths)
  !
  !  Jay Banks 22-Nov-95: added IF (PRNLEV.GE.2) to WRITE statements
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2),
  !  and nested inside other IF (rather than .AND. in one IF), which
  !  requires action whether WRITE executed or not.
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.  Also took spaces out of "end do"s and "end if"s.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use mmffm
  implicit none

  integer cursize, natoms, npaths
  integer IBOND(*), JBOND(*)
  integer paths(2, MAXPATHS, 3:*)

  integer alist(MAX_RINGSIZE+1)
  integer nnewpaths, newsize, i, j, ii, jj, k, m, ier
  integer cur_sys, prev_sys
  logical in_sys

  newsize = cursize + 1
  nnewpaths = 0
  !       construct all paths of length "newsize" from overlapping pairs of
  !       paths of length "cursize."
  do i = 1, npaths-1
     do j = i+1, npaths
        if (paths(1, i, cursize) .eq. -paths(1, j, cursize) .or. &
             paths(2, i, cursize) .eq. paths(1, j, cursize) .or. &
             paths(1, i, cursize) .eq. paths(2, j, cursize) .or. &
             paths(2, i, cursize) .eq. -paths(2, j, cursize)) &
             then                          ! have found a path of length "newsize"
           !       mark "backwards" path with negative index.
           !       CONVENTION: path "i" is ALWAYS traversed first
           if (paths(1,i,cursize) .eq. -paths(1,j,cursize)) then
              !             use order (2, i), (1, i) = (1, j), (2, j)
              ii = -i
              jj = j
           else if (paths(2,i,cursize) .eq. paths(1,j,cursize)) then
              !             use order (1, i), (2, i) = (1, j), (2, j)
              ii = i
              jj = j
           else if (paths(1,i,cursize) .eq. paths(2,j,cursize)) then
              !             use order (2, i), (1, i) = (2, j), (1, j)
              ii = -i
              jj = -j
           else if (paths(2,i,cursize) .eq. -paths(2,j,cursize)) then
              !             use order (1, i), (2, i) = (2, j), (1, j)
              ii = i
              jj = -j
           endif
           call exp_to_atoms(cursize, ii, jj, paths, IBOND, JBOND, &
                alist, ier)
           if (ier .lt. 0) then
              IF (WRNLEV.GE.2) THEN
                 WRITE(OUTU, 6002) ier
6002             format('get_next_size: ERROR: ier = ', i4, &
                      ' from exp_to_atoms')
              ENDIF
              return
           endif
           if (alist(1) .eq. alist(cursize+1)) then
              !       It's a ring!  Copy atom numbers to ring list
              !       If desired, can check here (alist vs. rlist(*, cursize, *)) whether
              !      it's new.  To do so, need to uncomment the following block so
              !      rlist will be stored.  (Also declarations for rlist and nsize)

              !       find "ring system" to which current ring belongs

              in_sys = .false.
              k = 1
              do while (k .le. cursize .and. (.not. in_sys))
                 if (INRING(alist(k)) .gt. 0) then
                    cur_sys = INRING(alist(k))
                    in_sys = .true.
                 endif
                 k = k + 1
              enddo
              if (.not. in_sys) then
                 IRINGS = IRINGS + 1
                 cur_sys = IRINGS
              endif
              do k = 1, cursize
                 prev_sys = INRING(alist(k))
                 if (prev_sys.ne.cur_sys .and. prev_sys.gt.0) then
                    do m = 1, natoms
                       if (INRING(m) .eq. prev_sys) then
                          INRING(m) = cur_sys
                       endif
                    enddo
                 endif
                 INRING(alist(k)) = cur_sys
              enddo                    ! reassign system numbers
           else                        ! put in list of paths of new size
              if (nnewpaths .ge. MAXPATHS) then
                 IF (WRNLEV.GE.2) then
                    write(OUTU, 6001) newsize
6001                format('get_next_size: Path list full for length ', &
                         i4)
                 ENDIF
              else
                 nnewpaths = nnewpaths + 1
                 paths(1, nnewpaths, newsize) = ii
                 paths(2, nnewpaths, newsize) = jj
              endif
           endif                      !is new path a ring or not?
        endif                        !path of length "newsize" found
     enddo                          !inner loop on paths of length "cursize"
  enddo                            !outer loop on paths of length "cursize"
  npaths = nnewpaths

  return

end subroutine get_next_size
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!
!       SUBROUTINE EXP_TO_ATOMS: FIND CONSTITUENT ATOMS OF TWO
!       OVERLAPPING PATHS.
!
!       WRITTEN BY JAY L. BANKS OF MOLECULAR SIMULATIONS, INC., AT AND
!       FOR MERCK & CO., INC.
!
!       NOTE: THE IDEA BEHIND THE 'PATHS' DATA STRUCTURE -- REPRESENTING
!       A PATH OF LENGTH 'N' AS A PAIR OF PATHS OF LENGTH 'N-1' -- WAS
!       MADE KNOWN TO THE AUTHOR THROUGH THE WORK OF BROND LARSON OF
!       THINKING MACHINES CORP., AND SHOULD *NOT* BE CONSIDERED THE
!       PROPERTY OF EITHER MOLECULAR SIMULATIONS, INC., OR MERCK & CO., INC.
!       THE IMPLEMENTATION OF THIS STRUCTURE USING SIGNED INTEGERS TO
!       INDICATE THE DIRECTION OF TRAVERSAL OF THE PATH OF LENGTH 'N-1',
!       AND THE USE OF RECURSIVE ROUTINES TO EXPAND A PATH OF ARBITRARY
!       SIZE TO ITS CONSTITUENT ATOMS, WERE DEVELOPED AT MERCK BY T. HALGREN
!       AND THE AUTHOR.
!
!      ARGUMENTS (FOR OPTIMOL/MMFF/CHARMM IMPLEMENTATION)
!      cursize        integer            input            size of input paths
!      in1            integer            input            index of first path
!      in2            integer            input            index of second path
!      paths          integer array      input            list of all paths
!      IBOND          integer array      input            first atom in bond
!      JBOND          integer array      input            second atom in bond
!      alist          integer array      output           list of atoms
!      ier            integer            output           error code
!
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
recursive subroutine exp_to_atoms(cursize, in1, in2, paths, &
     IBOND, JBOND, alist, ier)
  !       Takes two paths of length "cursize" (indices i1 and i2), returns (in
  !       alist) the list of "cursize+1" atoms in their union.  Returns an error
  !       if they don't overlap at "cursize-1" points (ier = -2), or if cursize,
  !       i1, or i2 is out of range (ier = -1).
  !
  ! Jay Banks, 22-Nov-95: added IF(PRNLEV.GE.2) to WRITE statement
  !
  !  28 Nov 95 Jay Banks: changed PRNLEV test to WRNLEV for printing error
  !  conditions.  Also took spaces out of "end do"s and "end if"s.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use mmffm
  implicit none

  integer cursize, in1, in2
  integer paths(2, MAXPATHS, 3:*)
  integer IBOND(*), JBOND(*)
  integer alist(MAX_RINGSIZE+1)
  integer ier

  integer i1, i2, itemp, j11, j12, j21, j22
  logical switched
  integer templist(2, MAX_RINGSIZE+1)
  integer tempindex, tempsize, templen
  integer jn11, jn12, jn21, jn22
  integer k, k1, k2, klast, jlastn1, jlastn2, jlast1, jlast2
  integer mid1, mid2, out1, out2, kn1, kn2, klastn, midlast, outlast

  i1 = abs(in1)
  i2 = abs(in2)
  switched = .false.
  if (i2 .lt. i1) then              !I don't think this can happen, but...
     itemp = i1
     i1 = i2
     i2 = itemp
     switched = .true.
     IF (WRNLEV.GE.2) write(OUTU, 6001)
6001 format('exp_to_atoms: indices switched')
  endif
  ier = 0


  if (cursize .lt. 1 .or. i1 .lt. 1 .or. i2 .lt. 1 .or. &
       i1 .eq. i2 .or. cursize .gt. MAX_RINGSIZE .or. &
       i1 .gt. MAXPATHS .or. i2 .gt. MAXPATHS) &
       then
     ier = -1
     CALL WRNDIE(-5, '<EXPAND>', 'size or index out of range')
     return
  else if (cursize .eq. 1) then
     alist(1) = i1
     alist(2) = i2
     return
  else if (cursize .eq. 2) then
     if (.not. switched) then
        if (in1 .lt. 0 .and. in2 .gt. 0 .and. &
             IBOND(i1) .eq. IBOND(i2)) &
             then
           alist(1) = JBOND(i1)
           alist(2) = IBOND(i1)
           alist(3) = JBOND(i2)
        else if (in1 .gt. 0 .and. in2 .gt. 0 .and. &
             JBOND(i1) .eq. IBOND(i2)) &
             then
           alist(1) = IBOND(i1)
           alist(2) = JBOND(i1)
           alist(3) = JBOND(i2)
        else if (in1 .lt. 0 .and. in2 .lt. 0 .and. &
             IBOND(i1) .eq. JBOND(i2)) &
             then
           alist(1) = JBOND(i1)
           alist(2) = IBOND(i1)
           alist(3) = IBOND(i2)
        else if (in1 .gt. 0 .and. in2 .lt. 0 .and. &
             JBOND(i1) .eq. JBOND(i2)) &
             then
           alist(1) = IBOND(i1)
           alist(2) = JBOND(i1)
           alist(3) = IBOND(i2)
        else
           ier = -1
           CALL WRNDIE(-5, '<EXPAND>', &
                'index out of range or nonoverlap')
        endif
     else                    ! switched
        if (in2 .lt. 0 .and. in1 .gt. 0 .and. &
             IBOND(i1) .eq. IBOND(i2)) &
             then
           alist(1) = JBOND(i1)
           alist(2) = IBOND(i1)
           alist(3) = JBOND(i2)
        else if (in2 .gt. 0 .and. in1 .gt. 0 .and. &
             JBOND(i1) .eq. IBOND(i2)) &
             then
           alist(1) = IBOND(i1)
           alist(2) = JBOND(i1)
           alist(3) = JBOND(i2)
        else if (in2 .lt. 0 .and. in1 .lt. 0 .and. &
             IBOND(i1) .eq. JBOND(i2)) &
             then
           alist(1) = JBOND(i1)
           alist(2) = IBOND(i1)
           alist(3) = IBOND(i2)
        else if (in2 .gt. 0 .and. in1 .lt. 0 .and. &
             JBOND(i1) .eq. JBOND(i2)) &
             then
           alist(1) = IBOND(i1)
           alist(2) = JBOND(i1)
           alist(3) = IBOND(i2)
        else
           ier = -1
           CALL WRNDIE(-5, '<EXPAND>', &
                'index out of range or nonoverlap')
        endif
     endif                  ! switched (cursize .eq. 2)
     return
  else                      ! cursize > 2, iterate down to 2
     jn11 = paths(1, i1, cursize)
     jn12 = paths(2, i1, cursize)
     jn21 = paths(1, i2, cursize)
     jn22 = paths(2, i2, cursize)
     j11 = abs(jn11)
     j12 = abs(jn12)
     j21 = abs(jn21)
     j22 = abs(jn22)
     if (.not. switched) then
        if (in1 .gt. 0) then
           mid1 = jn12
           out1 = jn11
        else if (in1 .lt. 0) then
           mid1 = -jn11
           out1 = -jn12
        endif
        if (in2 .gt. 0) then
           mid2 = jn21
           out2 = jn22
        else if (in2 .lt. 0) then
           mid2 = -jn22
           out2 = -jn21
        endif
     else
        if (in2 .gt. 0) then
           mid1 = jn12
           out1 = jn11
        else if (in1 .lt. 0) then
           mid1 = -jn11
           out1 = -jn12
        endif
        if (in1 .gt. 0) then
           mid2 = jn21
           out2 = jn22
        else if (in2 .lt. 0) then
           mid2 = -jn22
           out2 = -jn21
        endif
     endif                   ! switched
     if (mid1 .ne. mid2) then
        ier = -2
        CALL WRNDIE(-5, '<EXPAND>', 'paths do not overlap (1)')
        return
     endif
     tempindex = 1
     templist(1, 1) = out1
     templist(1, 2) = mid1
     templist(1, 3) = out2
     if (cursize .gt. 3) then
        do tempsize = cursize, 4, -1
           templen = cursize - tempsize + 3
           do k = 1, templen - 1, 2
              !       Fill in templist(3-tempindex, 1:templen+1) with indices of
              !       paths of length (tempsize-1).
              kn1 = templist(tempindex, k)
              kn2 = templist(tempindex, k + 1)
              k1 = abs(kn1)
              k2 = abs(kn2)
              if (k1 .eq. k2 .or. k1 .eq. 0 .or. k2 .eq. 0) then
                 ier = -1
                 CALL WRNDIE(-5, '<EXPAND>', &
                      'index out of range (k1=k2 or either=0)')
                 return
              endif

              jn11 = paths(1, k1, tempsize-1)
              jn12 = paths(2, k1, tempsize-1)
              jn21 = paths(1, k2, tempsize-1)
              jn22 = paths(2, k2, tempsize-1)
              j11 = abs(jn11)
              j12 = abs(jn12)
              j21 = abs(jn21)
              j22 = abs(jn22)
              if (kn1 .gt. 0) then
                 mid1 = jn12
                 out1 = jn11
              else if (kn1 .lt. 0) then
                 mid1 = -jn11
                 out1 = -jn12
              endif
              if (kn2 .gt. 0) then
                 mid2 = jn21
                 out2 = jn22
              else if (kn2 .lt. 0) then
                 mid2 = -jn22
                 out2 = -jn21
              endif
              if (mid2 .ne. mid1 .or. &
                   (k .gt. 1 .and. out1 .ne. templist(3-tempindex, k))) &
                   then
                 ier = -2
                 CALL WRNDIE(-5, '<EXPAND>', 'paths do not overlap (2)')
                 return
              endif
              if (k .eq. 1) then
                 templist(3-tempindex, 1) = out1
              endif
              templist(3-tempindex, k+1) = mid1
              templist(3-tempindex, k+2) = out2
              if (k .eq. templen-2) then
                 klastn = templist(tempindex, templen)
                 klast = abs(klastn)
                 jlastn1 = paths(1, klast, tempsize-1)
                 jlastn2 = paths(2, klast, tempsize-1)
                 jlast1 = abs(jlastn1)
                 jlast2 = abs(jlastn2)
                 if (klastn .gt. 0) then
                    midlast = jlastn1
                    outlast = jlastn2
                 else if (klastn .lt. 0) then
                    midlast = -jlastn2
                    outlast = -jlastn1
                 else
                    ier = -1
                    CALL WRNDIE(-5, '<EXPAND>', &
                         'index out of range (klast=0)')
                    return
                 endif
                 if (midlast .ne. out2) then
                    ier = -2
                    CALL WRNDIE(-5, '<EXPAND>', &
                         'paths do not overlap (3)')
                    return
                 endif
                 templist(3-tempindex, templen+1) = outlast
              endif               ! k .eq. templen-2
           enddo                ! k = 1, templen-1, 2
           tempindex = 3 - tempindex
        enddo                  ! tempsize = cursize, 4, -1
     endif                     ! cursize .gt. 3
     !       When we get here, the next value of templen = cursize, and
     !       templist(tempindex, 1:cursize) is a list of indices of paths of
     !       size 2, i.e., bonds.  Read atom indices off these to fill alist.
     do k = 1, cursize-1, 2
        kn1 = templist(tempindex, k)
        kn2 = templist(tempindex, k + 1)
        k1 = abs(kn1)
        k2 = abs(kn2)
        if (k1 .eq. k2 .or. k1 .eq. 0 .or. k2 .eq. 0) then
           ier = -1
           CALL WRNDIE(-5, '<EXPAND>', &
                'last loop: index out of range (k1=k2 or either=0)')
           return
        endif

        if (kn1 .gt. 0) then
           j11 = IBOND(k1)
           j12 = JBOND(k1)
        else if (kn1 .lt. 0) then
           j11 = JBOND(k1)
           j12 = IBOND(k1)
        endif
        if (kn2 .gt. 0) then
           j21 = IBOND(k2)
           j22 = JBOND(k2)
        else if (kn2 .lt. 0) then
           j21 = JBOND(k2)
           j22 = IBOND(k2)
        endif
        if (j12 .ne. j21 .or. &
             (k .gt. 1 .and. j11 .ne. alist(k))) &
             then
           ier = -2
           CALL WRNDIE(-5, '<EXPAND>', 'paths do not overlap (4)')
           return
        endif
        if (k .eq. 1) then
           alist(1) = j11
        endif
        alist(k+1) = j12
        alist(k+2) = j22
        if (k .eq. cursize-2) then
           klastn = templist(tempindex, cursize)
           klast = abs(klastn)
           if (klastn .gt. 0) then
              jlast1 = IBOND(klast)
              jlast2 = JBOND(klast)
           else if (klastn .lt. 0) then
              jlast1 = JBOND(klast)
              jlast2 = IBOND(klast)
           else
              ier = -1
              CALL WRNDIE(-5, '<EXPAND>', &
                   'last loop: index out of range (klast=0)')
              return
           endif
           if (jlast1 .ne. j22) then
              ier = -2
              CALL WRNDIE(-5, '<EXPAND>', 'paths do not overlap (5)')
              return
           endif
           alist(cursize+1) = jlast2
        endif                 ! k .eq. cursize-2
     enddo                  ! k = 1, cursize-1, 2
  endif                    ! cursize > 2
  !
  RETURN
END subroutine exp_to_atoms

! ======================================================================
! SUBROUTINE GOMEGA : COMPUTES THE NUMBER OF DIHEDRAL ANGLES (NPHI)
! AND THE NUMBERS OF THE 4 ATOMS DEFINING THE ANGLE, DiheList
! ======================================================================
SUBROUTINE GOMEGA
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  !     include 'param.f90'
  use psf
  use mmffm
  use stream
  use string
#ifdef KEY_RESIZE
  use resize
#endif  
  implicit none
  !
  ! ib, jb modified to bi, bj because of name conflict with charmm
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to AI, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.  Likewise "END IF"s.
  !
  integer i, ai, bi, ic, ii, j, ja, bj, jc
  integer nn
#ifdef KEY_RESIZE
  integer np0
  np0=nphi
#endif  
  !  
  NPHI=1
  NN=NTHETA-1
  DO I=1,NN
     AI=IT(I) ! AnglList(I,2)
     BI=JT(I) ! AnglList(I,3)
     IC=KT(I) ! AnglList(I,4)
     II=I+1
     DO J=II,NTHETA
        JA=IT(J) ! AnglList(J,2)
        BJ=JT(J) ! AnglList(J,3)
        JC=KT(J) ! AnglList(J,4)
        IF (bi.NE.JA) GOTO 200
        IF (IC.NE.bj) GOTO 100
        !  OMEGA AI-bi-IC-JC  (bi=JA & IC=bj)
        !  CHECK FOR THREE MEMBERED RING
        IF(AI.EQ.JC) GOTO 500
        IF (AI.LE.JC) THEN
           !                                     DiheList(,NPHI)=IPACK(AI,BI,IC,JC)
           IP(NPHI)=AI
           JP(NPHI)=BI
           KP(NPHI)=IC
           LP(NPHI)=JC
        ENDIF
        IF (AI.GT.JC) THEN
           !                                     DiheList(,NPHI)=IPACK(JC,IC,BI,AI)
           IP(NPHI)=JC
           JP(NPHI)=IC
           KP(NPHI)=BI
           LP(NPHI)=AI
        ENDIF
        GOTO 400
100     IF (AI.NE.bj) GOTO 500
        !  OMEGA JC-AI-bi-IC  (bi=JA & AI=bj)
        !  CHECK FOR THREE MEMBERED RING
        IF(JC.EQ.IC) GOTO 500
        IF (JC.LE.IC) THEN
           !                                       DiheList(,NPHI)=IPACK(JC,AI,BI,IC)
           IP(NPHI)=JC
           JP(NPHI)=AI
           KP(NPHI)=BI
           LP(NPHI)=IC
        ENDIF
        IF (JC.GT.IC) THEN
           !                                       DiheList(,NPHI)=IPACK(IC,BI,AI,JC)
           IP(NPHI)=IC
           JP(NPHI)=BI
           KP(NPHI)=AI
           LP(NPHI)=JC
        ENDIF
        GOTO 400
200     IF (bi.NE.JC) GOTO 500
        IF (IC.NE.bj) GOTO 300
        !  OMEGA AI-bi-IC-JA  (bi=JC & IC=bj)
        !  CHECK FOR THREE MEMBERED RING
        IF(AI.EQ.JA) GOTO 500
        IF (AI.LE.JA) THEN
           IP(NPHI)=AI
           JP(NPHI)=BI
           KP(NPHI)=IC
           LP(NPHI)=JA
        ENDIF
        IF (AI.GT.JA) THEN
           IP(NPHI)=JA
           JP(NPHI)=IC
           KP(NPHI)=BI
           LP(NPHI)=AI
        ENDIF
        GOTO 400
300     IF (AI.NE.bj) GOTO 500
        !  OMEGA JA-AI-bi-IC  (bi=JC & AI=bj)
        !  CHECK FOR THREE MEMBERED RING
        IF(JA.EQ.IC) GOTO 500
        IF (JA.LE.IC) THEN
           IP(NPHI)=JA
           JP(NPHI)=AI
           KP(NPHI)=BI
           LP(NPHI)=IC
        ENDIF
        IF (JA.GT.IC) THEN
           IP(NPHI)=IC
           JP(NPHI)=BI
           KP(NPHI)=AI
           LP(NPHI)=JA
        ENDIF
400     continue
        NPHI=NPHI+1
#ifdef KEY_RESIZE
        if (nphi .gt. np0) then
           np0=nphi+drsz0
           call resize_psf('psfres.F90','GOMEGA','NPHI',np0, .true.)
        endif
#else
        IF(NPHI.GT.MAXP) THEN
           WRITE(SCRTCH,'(A,2(1X,I5))') &
                'to many torsions:NPHI,MAXP=',NPHI,MAXP
           !     IF(IZ.NE.0) WRITE(IZ,410) NPHI,MAXP
           CALL wrndie(-5,'<gomega>',SCRTCH(:40))
        ENDIF
#endif
        !     WRITE(6,450) (DiheList(NPHI-1,L),L=1,4)
        !     WRITE(12,450) (DiheList(NPHI-1,L),L=1,4)
450     FORMAT(' DIHEDRAL ',4I5)
500     CONTINUE
     ENDDO
  ENDDO
  NPHI=NPHI-1
#ifdef KEY_RESIZE
  if (nphi .lt. np0) &
       call resize_psf('psfres.F90','GOMEGA','NPHI',nphi, .true.)
#endif  
  !  SORT THE DiheList ARRAY.
  IF (NPHI.GT.1) CALL KSORT(NPHI,IP,JP,KP,LP)
  !
  RETURN
END SUBROUTINE GOMEGA

! =====================================================================
! SUBROUTINE HADD :
! =====================================================================
!
#ifdef KEY_RESIZE
SUBROUTINE HADD(MTYPE,ATNUM,BondType, &
     NADDH,MODE,NIMPH,OK,ITAB,LOCAT,AtNames,RESNAME,KRES)
#else
SUBROUTINE HADD(NATOM,X,Y,Z,MTYPE,ATNUM,IB,JB,BondType,NBOND, &
     NADDH,MODE,NIMPH,OK,ITAB,LOCAT,AtNames,RESNAME,KRES)
#endif
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27-Nov-95: changed IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for
  !  WRITE statement followed by CALL WRNDIE.
  !
  !  Jay Banks 27-Nov-95: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use cnst_fcm, only: cnst_is_allocated, allocate_cnst
  use dimens_fcm
  use exfunc
  use stream
#ifdef KEY_RESIZE
  use psf ! hwm: check
  use resize
  use coord
  use coordc
  use deriv
#endif
  implicit none
  !
#ifndef KEY_RESIZE
  integer NATOM
  real(chm_real)  X(*),Y(*),Z(*)
  integer IB(*),JB(*)
  integer NBOND
#endif
  integer MTYPE(*)
  integer ATNUM(*)
  integer BondType(*)
  integer NIMPH(*)
  integer ITAB(6,NATOM)
  integer LOCAT(NATOM)
  character(len=*) AtNames(*)
  character(len=*) RESNAME(*)
  character(len=*) KRES(*)
  !
  integer iatom, ita
  integer mode, NADDH, nbors, nchar
  integer nexph, nh, nmode
  !
  CHARACTER(len=10) :: DIGIT='1234567890'
  CHARACTER(len=4) ADD
  LOGICAL OK
#ifdef KEY_RESIZE
  integer na0,nb0
#endif
  !
  OK=.TRUE.
  !   CHECK CARBON AND HETEROATOMS AND ADD ANY MISSING HYDROGENS
  NADDH=0
  IF(MODE.EQ.1) THEN
100  CONTINUE
     CALL QUERY(OUTU, &
          ' DO YOU WANT TO ADD HYDROGENS TO CARBON ATOMS? (Y/N): ')
     !        CALL QUERY(IZ,' DO YOU WANT TO ADD HYDROGENS TO CARBON ATOMS?'
     !    .   //' (Y/N): ')
     CALL RDWORD(1,1,1,NCHAR,ADD)
     IF(ADD.NE.'N'.AND.ADD.NE.'Y') GOTO 100
     IF(ADD.EQ.'N') RETURN
  ENDIF
#ifdef KEY_RESIZE
  na0=natom
  nb0=nbond
#endif  
  DO IATOM=1,NATOM
     ITA=ATNUM(IATOM)
     IF(ITA.LE.1) GOTO 600
     IF(MODE.EQ.0.OR.MODE.EQ.1) THEN
        IF(ITA.NE.6) GOTO 600
        IF(NIMPH(IATOM).LE.0) GOTO 600
        CALL HCOUNT(IATOM,ATNUM,IB,JB,BondType,NBOND, &
             NEXPH,nh,NBORS)
        !       WRITE(OUTU,*) NEXPH,nh,NBORS,NIMPH(IATOM)
        !       IF(IZ.NE.0) WRITE(IZ,*) NEXPH,nh,NBORS,NIMPH(IATOM)
        nh=NIMPH(IATOM)
        GOTO 500
     ELSE
        IF(ITA.EQ.6) GOTO 600
        CALL HCOUNT(IATOM,ATNUM,IB,JB,BondType,NBOND, &
             NEXPH,nh,NBORS)
        IF(nh.LE.0) GOTO 600
        GOTO 500
     ENDIF
     !   nh HYDROGENS ARE MISSING.  ADD THEM IF THIS IS A STAND-ALONE RUN --
     !   OTHERWISE, WRITE "NO PERMISSION" MESSAGE AND SET "OK" TO .FALSE.
500  CONTINUE
     !      IF(SALONE) THEN
#ifdef KEY_RESIZE
     if (natom+nh .gt. na0) then
        na0=natom+nh+drsz0
        call resize_psf('psfres.F90','HADD','NATOM',na0, .true.)
        call resize_coord('psfres.F90','HADD',na0, .true.)
        call resize_coordc('psfres.F90','HADD',na0, .true.)
        call resize_deriv('psfres.F90','HADD',na0, .true.)
     endif
     if (nbond+nh .gt. nb0) then
        nb0=nbond+nh+drsz0
        call resize_psf('psfres.F90','HADD','NBOND',nb0, .true.)
     endif
#endif
     CALL ADDH(IATOM,MODE,nh,NADDH,NBORS,NATOM,X,Y,Z,MTYPE, &
          ATNUM,IB,JB,BondType,NBOND,ITAB,AtNames,RESNAME,KRES)
     IF(MODE.EQ.0) NIMPH(IATOM)=0
     !      ELSE
     !         OK=.FALSE.
     !         if(prnlev.ge.2) WRITE(OUTU,550) nh,QNAME(IATOM)
     !C     IF(IZ.NE.0) WRITE(IZ,550) nh,QNAME(IATOM)
     !  550    FORMAT(' *** ATTEMPT TO ADD',I2,' HYDROGEN(S) TO ATOM ',
     !     .   A,' REJECTED ***')
     !      ENDIF
600  CONTINUE
  ENDDO
  IF(.NOT.OK) THEN
     !   WRITE "PERMISSION DENIED" MESSAGE, CITING REASON
     if(wrnlev.ge.2) WRITE(OUTU,650)
     !     IF(IZ.NE.0) WRITE(IZ,650)
650  FORMAT(/' *** ONLY THE STAND-ALONE VERSION OF MMFF ', &
          'HAS THE RIGHT TO ADD HYDROGENS ***'/ &
          ' *** EXECUTION MUST TERMINATE - RETURNING TO HOST ', &
          'PROGRAM ***')
     CALL WRNDIE(-5,'<hadd>','ERROR MSG')
  ENDIF
#ifdef KEY_RESIZE
  if (natom .lt. na0) then
     call resize_psf('psfres.F90','HADD','NATOM',natom, .true.)
     call resize_coord('psfres.F90','HADD',natom, .true.)
     call resize_coordc('psfres.F90','HADD',natom, .true.)
     call resize_deriv('psfres.F90','HADD',natom, .true.)
  endif
  if (nbond .lt. nb0) &
       call resize_psf('psfres.F90','HADD','NBOND',nbond, .true.)
#endif  
  NATOM=NATOM+NADDH

  if (cnst_is_allocated) call allocate_cnst(natom)
  
  IF(MODE.EQ.2.AND.NADDH.NE.0 .and. prnlev.ge.2) &
       WRITE(OUTU,700) NADDH
  !     IF(MODE.EQ.2.AND.NADDH.NE.0.AND.IZ.NE.0) WRITE(IZ,700) NADDH
700 FORMAT(/I3,' HYDROGENS WERE ADDED TO HETEROATOMS')
  IF(MODE.EQ.0.OR.MODE.EQ.1 .and. prnlev.ge.2) WRITE(OUTU,800) NADDH
  !     IF((MODE.EQ.0.OR.MODE.EQ.1).AND.IZ.NE.0) WRITE(IZ,800) NADDH
800 FORMAT(/I3,' HYDROGENS WERE ADDED TO CARBON ATOMS')
  !   RECOMPUTE BOND TABLE IF H'S WERE ADDED
  NMODE=1
  IF(MODE.EQ.0) NMODE=2
  IF(NADDH.NE.0) &
       CALL TABINT(NMODE,IB,JB,BondType,NATOM,NBOND,ITAB,LOCAT)
  !
  RETURN
END SUBROUTINE HADD

!  =====================================================================
SUBROUTINE HCOUNT(IATOM,ATNUMX,IB,JB,BondTypeX,NBOND, &
     NEXPH,NIMPH,NBORS)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed arguments ATNUM and BondType to ATNUMX
  ! and BondtypeX, to avoid conflict with common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use mmffm
  implicit none
  !
  integer ATNUMX(*)
  integer IB(*),JB(*),Bondtypex(*)
  integer NBOND
  !
  integer iatom, ibd, mt, nb
  integer nbors, nexph, nimph, nsum, MINVAL
  integer :: NVAL(54)=(/1,0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1,0, &
       1,2,0,0,0,0,0,0,0,0,0,0,3,4,3,2,1,0, &
       0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, &   !16*0, &
       1,0/)
  !
  !   COUNT THE NUMBERS OF EXPLICIT AND IMPLICIT HYDROGENS ON IATOM
  NSUM=0
  NBORS=0
  MT=MTYPE(IATOM)
  NEXPH=0
  DO nb=1,NBOND
     IF(IB(nb).EQ.IATOM) GOTO 100
     IF(JB(nb).EQ.IATOM) GOTO 200
     GOTO 300
100  CONTINUE
     NBORS=NBORS+1
     IBD=Bondtypex(nb)
     NSUM=NSUM+IBD
     IF(ATNUMX(JB(nb)).EQ.1) NEXPH=NEXPH+1
     GOTO 300
200  CONTINUE
     IBD=Bondtypex(nb)
     NSUM=NSUM+IBD
     NBORS=NBORS+1
     IF(ATNUMX(IB(nb)).EQ.1) NEXPH=NEXPH+1
300  CONTINUE
  ENDDO
  !
  !   MVALMIN(MT) IS THE MINIMUM PERMITTED VALENCE FOR AN ATOM OF MMFF
  !   TYPE MT;  IT IS NORMALLY THE FULL EXPECTED VALENCE (MVALMAX(MT)), BUT
  !   CAN BE LESS FOR ATOMS IN DELOCALIZED CHARGED SYSTEMS.  BECAUSE
  !   SUCH SYSTEMS ON INPUT TO MMFF ARE REPRESENTED BY SINGLE RESONANCE
  !   STRUCTURES, SOME ACTUALLY-EQUIVALENT ATOMS WILL HAVE MORE BONDS THAN
  !   OTHERS.  EXAMPLES ARE THE TWO OXYGENS IN A CARBOXYLATE ANION AND THE
  !   THREE NITROGENS IN A GUANIDINIUM CATION.  MMFF WISHES TO MARK AS
  !   NEEDING HYDROGENS ONLY ATOMS FOR WHICH EVEN THE MINIMUM VALENCE IS
  !   NOT SATISFIED.
  !
  IF(MT.GT.0) THEN
     MINVAL=MVALMIN(MT)
  ELSE
     MINVAL=NVAL(ATNUMX(IATOM))
  ENDIF
  NIMPH=MINVAL-NSUM
  !
  RETURN
END SUBROUTINE HCOUNT

! ======================================================================
! SUBROUTINE HIDEAL : CALCULATES IDEAL HYDROGEN POSTIONS
! **********************************************************************
! HIDEAL
! BY R.C. COLLINS  FOLLOWING SUGGESTIONS OF R.E. DAVIS AND R. HARLOW
! PROGRAM MODIFIED FOR SDP PACKAGE BY R. HARLOW
! CONVERTED TO SUBROUTINE FORM BY D. PENATOMK
!     CONVERTED TO STANDALONE FORM BY J. J. WENDOLOSKI
! THIS PROGRAM CALCULATES IDEAL HYDROGEN POSTIONS AND WRITES THEM INTO
! THE ARRAYS HPOSX, HPOSY, AND HPOSZ
! **********************************************************************
SUBROUTINE HIDEAL(IIM,ITYPE,JANG,ADROT,BANG,HPOSX,HPOSY,HPOSZ, &
     X,Y,Z,NATOM,NH,ATNUM)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable name "IM" to "IIM"
  !                       to avoid conflict with CHARMM name
  !
  !  27 Nov 95 Jay Banks: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use consta
  use stream
  implicit none
  !
  real(chm_real) HPOSX(3),HPOSY(3),HPOSZ(3),X(*),Y(*),Z(*)
  INTEGER ATNUM(*)

  integer iang, ifl1, ihyd, iim, imin1, imin2, imin3, in2
  integer in3, itype, jang, nang, NATOM, nh
  real(chm_real) adrot, bang, banp, bcos, bond, bsin,  &
       cntrx, cntry, cntrz
  real(chm_real) cosb, cprx, cpry, cprz, dang,  &
       dist1, dist2, dist3, fac1
  real(chm_real) fac2, fac3, fac4, fac5, fac6, hplx, hply, hplz
  real(chm_real) hvx, hvy, hvz, hx, hy, hz, pcx, pcy, pcz
  real(chm_real) pvx, pvy, pvz, px, py, pz, rcos, rho,  &
       rpio, rrho, rsin
  real(chm_real) scale, sinb, test, th, ux, uy, uz, vx, vy, vz
  real(chm_real) :: wx, wy, wz, zck=0.00001_chm_real
  !
  IFL1 = 1
  ! READ CALCULATION INFORMATION
  IF (ITYPE.EQ.0   ) GOTO 800
  BOND = 1.09
  IF(ATNUM(IIM).EQ.8) BOND=0.97
  IF(ATNUM(IIM).EQ.7) BOND=1.0
  IF(BANG.LT.0.0) BANG=109.5
  IN2=0
  IN3=0
  IF (ITYPE.NE.1    ) GOTO 100
  IANG=JANG
  IF (IANG.EQ.0) IANG = 3
  IF (IANG.GT.3) IANG = 3
  ! THE PROGRAM ASKS FOR AND PRINTS THE NAMES OF THE HYDROGEN ATOMS
  ! BUT DOES NOT WRITE THEM ON THE PARA.PKS FILE.
  IHYD = IANG
  ! CALCULATE HYDROGEN POSITIONS AND STORE THEM IN PARA.PKS
100 IF (ITYPE.EQ.1   ) GOTO 500
  CALL MIND(IIM, IMIN1, IMIN2, IMIN3, DIST1, DIST2, DIST3, &
       NATOM,X,Y,Z,HX,HY,HZ)
  CNTRX = X(IIM)
  CNTRY = Y(IIM)
  CNTRZ = Z(IIM)
  !       C1 IS IM CARBON ATOM C2,C3,C4 ARE 3 NEAREST ATOMS
  !       COMPUTE UNIT VECTORS V AND W ALONG C2-C1 AND C3-C1 BONDS
  VX = (CNTRX-X(IMIN1))/DIST1
  VY = (CNTRY-Y(IMIN1))/DIST1
  VZ = (CNTRZ-Z(IMIN1))/DIST1
  WX = (CNTRX-X(IMIN2))/DIST2
  WY = (CNTRY-Y(IMIN2))/DIST2
  WZ = (CNTRZ-Z(IMIN2))/DIST2
  IF (ITYPE.EQ. 5  ) GOTO 300
  !
  !       COMPUTE UNIT VECTOR HPL IN PLANE OF C1,C2,C3 BisecTING C2-C1-C3
  HPLX = VX + WX
  HPLY = VY + WY
  HPLZ = VZ + WZ
  CALL NORMLL(HPLX, HPLY, HPLZ)
  IF (ITYPE.EQ.3   ) GOTO 200
  !       COMPUTE H PLANAR POSTION  AND BOND ANGLE
  HX = BOND*HPLX + CNTRX
  HY = BOND*HPLY + CNTRY
  HZ = BOND*HPLZ + CNTRZ
  DANG = ACOS(-VX*HPLX-VY*HPLY-VZ*HPLZ)*RADDEG
  HPOSX(IFL1)=HX
  HPOSY(IFL1)=HY
  HPOSZ(IFL1)=HZ
  IFL1=IFL1+1
  GOTO 800
  !       COMPUTE CROSS PRODUCT OF HPL AND V ,W
200 WX = VY*HPLZ - VZ*HPLY
  WY = VZ*HPLX - VX*HPLZ
  WZ = VX*HPLY - VY*HPLX
  CALL NORMLL(WX, WY, WZ)
  !       HYDROGENS LIE IN HPL W PLANE AT ANGLE BANP FROM HPL
  BANP = BANG*DEGRAD*.5
  COSB = COS(BANP)
  SINB = SIN(BANP)
  HPLX = HPLX*COSB
  HPLY = HPLY*COSB
  HPLZ = HPLZ*COSB
  WX = WX*SINB
  WY = WY*SINB
  WZ = WZ*SINB
  HX = HPLX + WX
  HY = HPLY + WY
  HZ = HPLZ + WZ
  DANG = ACOS(-VX*HX-HY*VY-VZ*HZ)*RADDEG
  HX = HX*BOND + CNTRX
  HY = HY*BOND + CNTRY
  HZ = HZ*BOND + CNTRZ
  HPOSX(IFL1)=HX
  HPOSY(IFL1)=HY
  HPOSZ(IFL1)=HZ
  IFL1=IFL1+1
  IF(NH.EQ.1) GOTO 800
  HX = (HPLX-WX)*BOND + CNTRX
  HY = (HPLY-WY)*BOND + CNTRY
  HZ = (HPLZ-WZ)*BOND + CNTRZ
  HPOSX(IFL1)=HX
  HPOSY(IFL1)=HY
  HPOSZ(IFL1)=HZ
  IFL1=IFL1+1
  GOTO 800
  !       COMPUTE UNIT VECTOR C4-C1, U
300 UX = (CNTRX-X(IMIN3))/DIST3
  UY = (CNTRY-Y(IMIN3))/DIST3
  UZ = (CNTRZ-Z(IMIN3))/DIST3
  !       SOLVE 3X3 SET OF EQUATIONS  H DOT U,H DOTV,H DOT W = CONST =1
  FAC1 = UX/VX
  FAC2 = UX/WX
  FAC3 = VY*FAC1 - UY
  FAC4 = VZ*FAC1 - UZ
  FAC5 = WY*FAC2 - UY
  FAC6 = WZ*FAC2 - UZ
  FAC1 = FAC1 - 1.
  FAC2 = FAC2 - 1.
  HZ = ((FAC3*FAC2)/FAC5-FAC1)/(FAC6*FAC3/FAC5-FAC4)
  HY = (FAC1-HZ*FAC4)/FAC3
  HX = (1.-HZ*UZ-HY*UY)/UX
  !   FIND AND NORMALIZE V+W+U RESULTANT
  !     WRITE(OUTU,'(4I4,3F10.5)') IIM,IMIN1,IMIN2,IMIN3,DIST1,DIST2,DIST3
  HX=UX+VX+WX
  HY=UY+VY+WY
  HZ=UZ+VZ+WZ
  CALL NORMLL(HX, HY, HZ)
  TEST = -HX*UX - HY*UY - HZ*UZ
  !     IF (TEST.LE.0.) GOTO 7
  !     BOND = -BOND
  !     TEST = -TEST
400 HX = BOND*HX + CNTRX
  HY = BOND*HY + CNTRY
  HZ = BOND*HZ + CNTRZ
  BOND = ABS(BOND)
  !     DANG = ACOS(TEST)/PI
  HPOSX(IFL1)=HX
  HPOSY(IFL1)=HY
  HPOSZ(IFL1)=HZ
  IFL1=IFL1+1
  GOTO 800
500 IF (IN2.GT.0) GOTO 600
  CALL MIND(IIM, IN2, IN3, IMIN3, DIST1, DIST2, DIST3, &
       NATOM,X,Y,Z,HX,HY,HZ)
600 TH = (180.-BANG)*DEGRAD
  BCOS = BOND*COS(TH)
  BSIN = BOND*SIN(TH)
  ! IIM IS SEQ NO OF ATOM IN LIST FOR METHYL CARBON
  ! IN2 IS SEQ NO OF ATOM IN LIST BONDED TO METHYL
  ! IN3 IS SEQ NO OF ATOM  TO BE TRANS TO HYDROGEN
  ! IIM =C1  IN2=C2  IN3=C3
  ! FORM VECTOR HV  C3 TO C2
  HVX = X(IN2) - X(IN3)
  HVY = Y(IN2) - Y(IN3)
  HVZ = Z(IN2) - Z(IN3)
  ! PC IS EXTENTSION VECTOR  ALONG C2 TO C1
  ! P IS POINT IN CENTER OF TRIANGLE OF H ATOMS
  PCX = X(IIM) - X(IN2)
  PCY = Y(IIM) - Y(IN2)
  PCZ = Z(IIM) - Z(IN2)
  CALL NORMLL(PCX, PCY, PCZ)
  PX = X(IIM) + BCOS*PCX
  PY = Y(IIM) + BCOS*PCY
  PZ = Z(IIM) + BCOS*PCZ
  ! FIND A VECTOR PV FROM P TOA POINT ALONG VECTOR HV ORIGINATING FROM C1
  ! WITH PV NORMAL TO P-C1
  ! H1 IS IN THE DIRECTION PV FROM P
  SCALE=((PX-X(IIM))*PCX+(PY-Y(IIM))*PCY+ &
       (PZ-Z(IIM))*PCZ)/(HVX*PCX+HVY*PCY+HVZ*PCZ)
  PVX = X(IIM) + SCALE*HVX - PX
  PVY = Y(IIM) + SCALE*HVY - PY
  PVZ = Z(IIM) + SCALE*HVZ - PZ
  CALL NORMLL(PVX, PVY, PVZ)
  ! CALCULATE CROSS PRODUCT OF PC AND PV
  !  PC AND PV DEFINE PLANE OF H ATOMS
  CPRX = (PCY*PVZ-PCZ*PVY)*BSIN
  CPRY = (PCZ*PVX-PCX*PVZ)*BSIN
  CPRZ = (PCX*PVY-PCY*PVX)*BSIN
  PVX = PVX*BSIN
  PVY = PVY*BSIN
  PVZ = PVZ*BSIN
  ! FOR EACH ATOM IIM, THREE H POSTIONS ARE CALCULATED AT ROTATION ANGLES
  ! OF 120 ABOUT C-C BOND FROM POSTION TRANS TO ATOM IN3
  ! ROTATE PV VECTOR ANGLE RHO
  RRHO = 360./IANG
  RHO = -RRHO + ADROT
  DO NANG=1,IANG
     IF(NANG.GT.NH) GOTO 800
     RHO = RHO + RRHO
     IF (RHO.GE.360.) RHO = RHO - 360.
     IF (RHO.LT.-ZCK) RHO = RHO + 360.
     RPIO = RHO*DEGRAD
     RCOS = COS(RPIO)
     RSIN = SIN(RPIO)
     HX = RCOS*PVX + RSIN*CPRX + PX
     HY = RCOS*PVY + RSIN*CPRY + PY
     HZ = RCOS*PVZ + RSIN*CPRZ + PZ
     HPOSX(IFL1)=HX
     HPOSY(IFL1)=HY
     HPOSZ(IFL1)=HZ
     IFL1=IFL1+1
  ENDDO
800 CONTINUE
  !
  RETURN
END SUBROUTINE HIDEAL

! ====================================================================
! SUBROUTINE IMPLCT : FIND THE NUMBER OF IMPLICIT HYDROGENS (IF ANY)
! ON EACH CARBON;  THESE MISSING HYDROGENS WILL LATER NEED TO BE ADDED
! FOR MMFF TO RUN CORRECTLY
! ====================================================================
SUBROUTINE IMPLCT(NIMPH,NATOM,ATNUM,MTYPE,IB,JB,BondType, &
     NBOND,NICHG)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  implicit none
  !
  integer NIMPH(*)
  integer NATOM
  integer ATNUM(*)
  integer MTYPE(*)
  integer IB(*),JB(*),BondType(*)
  integer NBOND
  integer NICHG(*)
  !
  integer i
  integer nbors, nexp, nimp
  !
  DO I=1,NATOM
     NIMPH(I)=0
     IF(ATNUM(I).NE.6) GOTO 200
     CALL HCOUNT(I,ATNUM,IB,JB,BondType,NBOND, &
          NEXP,NIMP,NBORS)
     IF(NIMP.GE.1) NIMPH(I)=NIMP
     !   SPECIAL RULE FOR NEGATIVELY CHARGED CARBONS IN ACETYLIDE ANIONS
     !   AND IN ISONITRILES - each should form just three bonds (to a
     !   triply bonded neighbor, rather than the usual 4
     IF(NICHG(I).EQ.2) NIMPH(I)=NIMPH(I)-1
     !     WRITE(OUTU,20) I,QNAME(I),NIMPH(I),NIMP,NEXP,NBORS
     !     IF(IZ.NE.0) WRITE(IZ,20) I,QNAME(I),NIMPH(I),NIMP,NEXP,NBORS
     ! 100 FORMAT(' IMPLICIT H:',I4,1X,A,4I4)
200  CONTINUE
  ENDDO
  !
  RETURN
END SUBROUTINE IMPLCT

!==================================================================
! SUBROUTINE KSORT(NK,K) : SORTS THE K-ARRAY IN ASCENDING ORDER.
!==================================================================
SUBROUTINE KSORT(NK,IPH,JPH,KPH,LPH)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !  16 Oct 95 Jay Banks: changed variable names IP, JP, KP, LP, KT to
  !       IPH, JPH, KPH, LPH, KTMP to avoid conflict with CHARMM names
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use memory
  use chm_kinds
  use dimens_fcm
  implicit none
  !
  INTEGER NK, IPH(*),JPH(*),KPH(*),LPH(*)
  !
  integer i, j, l, m
  INTEGER,allocatable,dimension(:,:) :: ktmp
  INTEGER,allocatable,dimension(:) :: n


  call chmalloc("datastruc.src","ksort","ktmp",maxp*2,4,intg=KTMP)
  call chmalloc("datastruc.src","ksort","n",maxp,intg=n)

  !
  !  TRANSFER K TO KTMP, THE TEMPORARY STORAGE ARRAY; SET ALL N=1.
  DO I=1,NK
     !     KTMP(,I)=K(,I)
     KTMP(I,1)=IPH(I)
     KTMP(I,2)=JPH(I)
     KTMP(I,3)=KPH(I)
     KTMP(I,4)=LPH(I)
     N(I)=1
  ENDDO
  !  CALCULATE THE RANK FOR EACH ENTRY IN KTMP
  M=NK-1
  DO I=1,M
     J=I+1
     DO L=J,NK
        !     IF(KTMP(,I).GT.KTMP(,L)) GOTO 30
        IF (KTMP(I,1).GT.KTMP(L,1)) GOTO 300
        IF ((KTMP(I,1).EQ.KTMP(L,1)).AND.(KTMP(I,2).GT.KTMP(L,2))) &
             GOTO 300
        IF ((KTMP(I,1).EQ.KTMP(L,1)).AND.(KTMP(I,2).EQ.KTMP(L,2)) &
             .AND.(KTMP(I,3).GT.KTMP(L,3))) GOTO 300
        IF ((KTMP(I,1).EQ.KTMP(L,1)).AND.(KTMP(I,2).EQ.KTMP(L,2)) &
             .AND.(KTMP(I,3).EQ.KTMP(L,3)).AND.(KTMP(I,4).GT.KTMP(L,4)))  &
             GOTO 300
200     N(L)=N(L)+1
        GOTO 400
300     N(I)=N(I)+1
400     CONTINUE
     ENDDO
  ENDDO
  !  RESTORE THE K-ARRAY.  THE VALUE OF N FOR EACH ENTRY IN KTMP BECOMES THE
  !  SUBSCRIPT OF THIS ENTRY IN THE K-ARRAY.
  DO I=1,NK
     !     K(,N(I))=KTMP(,I)
     IPH(N(I))=KTMP(I,1)
     JPH(N(I))=KTMP(I,2)
     KPH(N(I))=KTMP(I,3)
     LPH(N(I))=KTMP(I,4)
  ENDDO
  !
  call chmdealloc("datastruc.src","ksort","ktmp",maxp*2,4,intg=KTMP)
  call chmdealloc("datastruc.src","ksort","n",maxp,intg=n)
  RETURN
END SUBROUTINE KSORT

! ======================================================================
! SUBROUTINE MIND : FIND THE CLOSEST ATOM, GIVING PREFERENCE
! TO THE DIRECTLY ATTACHED ATOMS
! ======================================================================
SUBROUTINE MIND(K, IMIN1, IMIN2, IMIN3, DIST1, DIST22, DIST3, &
     NATOM,X,Y,Z,HX,HY,HZ)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27 Nov 95: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  implicit none
  !
  integer imin1, imin2, imin3, j, k, NATOM
  real(chm_real)  del, dist1, dist22, dist3, distq, hx, hy, hz
  !
  real(chm_real) X(*),Y(*),Z(*)
  !
  DIST1 = 50000.
  DIST22 = 50000.
  DIST3 = 50000.
  IMIN1 = 0
  IMIN2 = 0
  IMIN3 = 0
  !
  DO J=1,NATOM
     IF (J.EQ.K) GOTO 300
     DEL=10000.
     IF(CONN14(J,K).EQ.1) DEL=1000.
     IF(CONN13(J,K).EQ.1) DEL=100.
     IF(CONN12(J,K).EQ.1) DEL=10.
     DISTQ=DEL+SQRT((X(J)-X(K))**2+(Y(J)-Y(K))**2+ &
          (Z(J)-Z(K))**2)
     IF(DISTQ.LT.DIST1) THEN
        DIST1=DISTQ
        IMIN1=J
     ENDIF
300  CONTINUE
  ENDDO
  !
  !   NOW FIND THE NEXT CLOSEST ATOM
  !
  DO J=1,NATOM
     IF (J.EQ.K.OR.J.EQ.IMIN1) GOTO 1300
     DEL=10000.
     IF(CONN14(J,K).EQ.1) DEL=1000.
     IF(CONN13(J,K).EQ.1) DEL=100.
     IF(CONN12(J,K).EQ.1) DEL=10.
     DISTQ=DEL+SQRT((X(J)-X(K))**2+(Y(J)-Y(K))**2+ &
          (Z(J)-Z(K))**2)
     IF(DISTQ.LT.DIST22) THEN
        DIST22=DISTQ
        IMIN2=J
     ENDIF
1300 CONTINUE
  ENDDO
  !
  !   SIMILARLY, FIND THE THIRD CLOSEST ATOM
  !
  DO J=1,NATOM
     IF (J.EQ.K.OR.J.EQ.IMIN1.OR.J.EQ.IMIN2) GOTO 2300
     DEL=10000.
     IF(CONN14(J,K).EQ.1) DEL=1000.
     IF(CONN13(J,K).EQ.1) DEL=100.
     IF(CONN12(J,K).EQ.1) DEL=10.
     DISTQ=DEL+SQRT((X(J)-X(K))**2+(Y(J)-Y(K))**2+ &
          (Z(J)-Z(K))**2)
     IF(DISTQ.LT.DIST3) THEN
        DIST3=DISTQ
        IMIN3=J
     ENDIF
2300 CONTINUE
  ENDDO
  !
  !   NOW CALCULATE ACTUAL (UNBIASED) DISTANCES AND RETURN
  !
  DISTQ=SQRT((X(IMIN1)-X(K))**2+ &
       (Y(IMIN1)-Y(K))**2+(Z(IMIN1)-Z(K))**2)
  IF(IMIN2.NE.0) DIST22=SQRT((X(IMIN2)-X(K))**2+ &
       (Y(IMIN2)-Y(K))**2+(Z(IMIN2)-Z(K))**2)
  IF(IMIN3.NE.0) DIST3=SQRT((X(IMIN3)-X(K))**2+ &
       (Y(IMIN3)-Y(K))**2+(Z(IMIN3)-Z(K))**2)
  !
  RETURN
END SUBROUTINE MIND

! ======================================================================
! SUBROUTINE NORML : normalizes vector
! ======================================================================
SUBROUTINE NORMLL(X, Y, Z)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  implicit none
  real(chm_real) dist, x, y, z
  DIST = SQRT(X**2+Y**2+Z**2)
  IF(DIST.LE. 0.001) DIST=1.0
  X = X/DIST
  Y = Y/DIST
  Z = Z/DIST
  !
  RETURN
END SUBROUTINE NORMLL

! ===========================================================
! SUBROUTINE TABINT : CREATES THE SPECIAL REDUNDANT CONN. TAB.
! THIS ROUTINE CREATES THE SPECIAL REDUNDANT CONN. TAB. NEEDED
! BY THE NEXT ATOM ROUTINE.  IN THIS TABLE EACH ATOM INDEXES
! ALL ATOMS BONDED TO IT.
! ===========================================================
SUBROUTINE TABINT(MODE,IB,JB,BondType,NATOM,NBOND,ITAB, &
     LOCAT)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 22 Nov 95: added IF (PRNLEV.GE.2) to a WRITE statement.
  !  Several others already have IF (WRNLEV.GT.0), but I don't know if
  !  that's enough "protection" (e.g., from multiple nodes on a parallel
  !  machine executing the same WRITE).  Left it for now.
  !
  !  Jay Banks 27 Nov 95: changed above (WRNLEV.GT.0) to (WRNLEV.GE.2),
  !  which seems to be the standard in the rest of CHARMM.  Also changed
  !  IF (PRNLEV.GE.2) to IF (WRNLEV.GE.2) for a WRITE followed by CALL
  !  WRNDIE.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use stream
  use string
  use chutil,only:atomid
  implicit none
  !
  integer IB(*),JB(*)
  integer BondType(*)
  integer NATOM
  integer NBOND
  integer ITAB(6,NATOM)
  integer LOCAT(NATOM)
  !
  integer i, ATOMI, ATOMJ
  integer j, loc1, loc2
  integer mode, nb, ntimes, ERROR
  character(len=8) SID,RID,REN,IUPAC
  !
  ERROR = 0
#if KEY_DEBUG==1
  if(prnlev.gt.6) then
     write(outu,'(a,3i5)') &
          'tabint> debug: mode,natom,nbond=',mode,natom,nbond
     write(outu,'(a,7(1x,i2))') 'IAT1=',(IB(nb),nb=1,7)
     write(outu,'(a,7(1x,i2))') 'IAT2=',(JB(nb),nb=1,7)
     write(outu,'(a,7(1x,i2))') 'BondType =',(BondType(nb),nb=1,7)
  endif
#endif /*  DEBUG*/

  LOCAT(1:NATOM  ) = 0   ! ZERO OUT THE NEW CONN. TAB.
  ITAB(1:6,1:NATOM) = -1
  !
  ! PICK UP THE NUMBER OF ENTRYS
  !     NENTRY=NATOM
  ! LOOP OVER ALL BONDS IN THE ORIGINAL CONN. TAB. AND MAKE THE
  ! NEW CONN. TAB.
  DO nb=1,NBOND
     ATOMI=IB(nb)
     ATOMJ=JB(nb)
     NTIMES=BondType(nb)
     IF(NTIMES.LE.0 .or. NTIMES.gt.3) then
        ERROR=ERROR+1
        if(WRNLEV.ge.2) write(OUTU,'(a,4i6)') &
             'tabint> BondType out of range: nb,IB(nb),JB(nb),'// &
             'BondType(nb)=',nb,IB(nb),JB(nb),BondType(nb)
     endif
     IF(ABS(MODE).EQ.1) NTIMES=1
     !   RECORD THIS BOND TWICE IF A DOUBLE BOND, ETC
#if KEY_DEBUG==1
     if(prnlev.gt.6) then
        write(outu,'(a,4i3)') 'tabint> debug: nb,ATOMI,ATOMJ,NTIMES=', &
             nb,ATOMI,ATOMJ,NTIMES
     endif
#endif /*  DEBUG*/
     DO J=1,NTIMES
        ! SKIP 0 ENTRIES IN THE CONN TAB.
        IF(ATOMI.EQ.0.OR.ATOMJ.EQ.0)GOTO 400
        IF(ATOMI.EQ.ATOMJ) then
           ERROR=ERROR+1
           if(WRNLEV.ge.2) write(OUTU,'(a,2i6)') &
                'tabint> TWO ATOMS BONDED ARE THE SAME: ATOMI,ATOMJ=', &
                ATOMI,ATOMJ
        endif
        !
        LOC1=LOCAT(ATOMI) ! GET THE COLUMN NUMBER FOR EACH ENTRY
        LOC2=LOCAT(ATOMJ)
        LOC1=LOC1+1        ! UPDATE IT
        LOC2=LOC2+1
#if KEY_DEBUG==1
        if(prnlev.gt.6) then
           write(outu,'(a,4i3)') 'tabint> debug: j,ATOMI,ATOMJ,NTIMES=', &
                j,ATOMI,ATOMJ,NTIMES
           write(outu,'(a,2i3)') 'tabint> debug: LOC1,LOC2=',LOC1,LOC2
        endif
#endif /*  DEBUG*/
        IF(LOC1.GT.4) THEN
           !   REPORT THE ATOM SEQUENCE NUMBER, IF IT IS DEFINED
           ERROR=ERROR+1
           if(WRNLEV.ge.2) then
              call ATOMID(ATOMI,SID,RID,REN,IUPAC)
              WRITE(OUTU,'(A,I5,4(1x,A),A)') ' tabint>ERROR: ATOM', &
                   ATOMI,SID(1:idleng),RID(1:idleng),REN(1:idleng), &
                   IUPAC(1:idleng),' HAS MORE THAN FOUR BONDS'
           ENDIF
        ENDIF
        IF(LOC2.GT.4) THEN
           ERROR=ERROR+1
           if(WRNLEV.ge.2) then
              call ATOMID(ATOMJ,SID,RID,REN,IUPAC)
              WRITE(OUTU,'(A,I5,4(1x,A4),A)') ' tabint>ERROR: ATOM', &
                   ATOMJ,SID(1:idleng),RID(1:idleng),REN(1:idleng), &
                   IUPAC(1:idleng),' HAS MORE THAN FOUR BONDS'
           ENDIF
        ENDIF
        LOCAT(ATOMI)=LOC1       ! RESET IT
        LOCAT(ATOMJ)=LOC2
        ITAB(LOC1,ATOMI)=ATOMJ ! PLACE THE ATOMS INT THE NEW CONN. TAB.
        ITAB(LOC2,ATOMJ)=ATOMI
     ENDDO
400  CONTINUE
  ENDDO
  !
  !   STORE THE NUMBERS OF ATOMS ATTACHED TO EACH ATOM IN ITAB(6,)
  !   (ACTUALLY, IF ABS(MODE)=2 THIS IS THEN NUMBER OF BONDS MADE
  !   TO THE ATOM IN QUESTION)
  !
  DO I=1,NATOM
     ITAB(6,I)=LOCAT(I)
  ENDDO

  if(ERROR.gt.0) then
     IF (WRNLEV.GE.2) WRITE(OUTU,'(a)')  &
          ' tabint>NOTE:  MMFF REQUIRES THAT "DATIVE" BONDS BE'// &
          ' REPRESENTED AS SINGLE BONDS', &
          '        EXAMPLES: S-O BONDS IN SULFONAMIDES, P-O BONDS'// &
          ' IN PHOSPHINE OXIDES'
     call wrndie(-5,'<tabint>','errors in structure')
  endif
  !
  ! OK NOW THE LOCATION POINTERS ARE NOT NEEDED,SO USE STORAGE FOR
  ! CYCLE POINTERS WHICH WILL BE USED IN nextat.
  IF(MODE.NE.-1) LOCAT(1:NATOM) = 1
  !
  RETURN
END SUBROUTINE TABINT

! ======================================================================
! SUBROUTINE THETA : COMPUTES THE NUMBER OF BOND ANGLES (NTHETA).
! THIS SUBROUTINE COMPUTES THE NUMBER OF BOND ANGLES (NTHETA) AND SETS 
! UP THE nglList ARRAY, WHERE nglList IS THE PACKED INTEGER
! IAMMFFIBmmffICID.  IAMMFF,IBmmff, IC, & ID ARE THE ATOM NUMBERS OF THE
! ANGLE A-B-C AND ID IS A THIRD ATTACHED ATOM FOR OUT-OF-PLANE BENDING.
! ======================================================================

SUBROUTINE THETA
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 25 Oct 95: changed variable IA to IAmmff, to avoid conflict
  !  with variable in common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use string
#ifdef KEY_RESIZE
  use resize
#endif  
  implicit none
  !
  integer i, iammff,  ibmmff, ic, id, itmmff, j
#ifdef KEY_RESIZE
  integer nt0
  nt0=ntheta
#endif
  !
  NTHETA=0          ! angle bending counter
  !     nop=0             ! oopl counter
  !
  DO I=1,NATOM
     ibmmff=ITAB(2,I)
     IF (ibmmff.LE.0) GOTO 200
     IAMMFF=ITAB(1,I)
     IC=ITAB(3,I)
     ID=ITAB(4,I)
     itmmff=MTYPE(I)
     !     WRITE(6,100) I,IAMMFF,ibmmff,IC,ID,itmmff
     !     WRITE(12,100) I,IAMMFF,ibmmff,IC,ID,itmmff
100  FORMAT(' NUMBER',I4,'   ANGLE',4I4, '   IT',I3)
     !  IF ATOM-I IS TRIGONAL, SET itmmff=0 TO SIGNIFY THAT ALL X-I-Y
     !  ANGLES WILL HAVE OUT-OF-PLANE BENDING.
     !
     !   FOR MMFF, INCLUDE ALL TRI-COORDINATE ATOMS
     !
     !      WRITE(6,*) I,itmmff,ID
     j=itmmff
     IF(MCOORD(j).EQ.3.AND.ID.LE.0) itmmff=0
     NTHETA=NTHETA+1
#ifdef KEY_RESIZE
     if (ntheta .gt. nt0) then
        nt0=ntheta+drsz0
        call resize_psf('psfres.F90','THETA','NTHETA',nt0, .true.)
     endif
#else
     IF(NTHETA.GT.MAXT) goto 900
#endif     
     IT(NTHETA)=IAMMFF
     JT(NTHETA)=I
     KT(NTHETA)=IBMMFF
     LTHETA(NTHETA)=0
     !  IF ANGLE IS INVOLVED IN O-P-B, PUT THE THIRD ATOM ATTACHED TO I IN
     !  nglList.
     !     IF (ITMMFF.EQ.0) nglList(,NTHETA)=nglList(,NTHETA)+IC
     ! T.A. Halgren change: fix error in logic for case IC <= 0 (B980629.wy)
     !     IF (ITMMFF.EQ.0) LTHETA(NTHETA)=IC
     IF (ITMMFF.EQ.0.and.ic.gt.0) LTHETA(NTHETA)=IC
     ! end of T.A. Halgren change
     IF (IC.LE.0) GOTO 200
     NTHETA=NTHETA+1
#ifdef KEY_RESIZE
     if (ntheta .gt. nt0) then
        nt0=ntheta+drsz0
        call resize_psf('psfres.F90','THETA','NTHETA',nt0, .true.)
     endif
#else
     IF(NTHETA.GT.MAXT) goto 900
#endif
     IT(NTHETA)=IAMMFF
     JT(NTHETA)=I
     KT(NTHETA)=IC
     LTHETA(NTHETA)=0
     !  CHECK FOR O-P-B
     !     IF (ITMMFF.EQ.0) nglList(,NTHETA)=nglList(,NTHETA)+IBMMFF
     IF (ITMMFF.EQ.0) LTHETA(NTHETA)=IBMMFF
     NTHETA=NTHETA+1
#ifdef KEY_RESIZE
     if (ntheta .gt. nt0) then
        nt0=ntheta+drsz0
        call resize_psf('psfres.F90','THETA','NTHETA',nt0, .true.)
     endif
#else
     IF(NTHETA.GT.MAXT) goto 900
#endif
     IT(NTHETA)=IBMMFF
     JT(NTHETA)=I
     KT(NTHETA)=IC
     LTHETA(NTHETA)=0
     !  CHECK FOR O-P-B
     !     IF (ITMMFF.EQ.0) nglList(,NTHETA)=nglList(,NTHETA)+IAMMFF
     IF (ITMMFF.EQ.0) LTHETA(NTHETA)=IAMMFF
     IF (ID.LE.0) GOTO 200
     NTHETA=NTHETA+1
#ifdef KEY_RESIZE
     if (ntheta .gt. nt0) then
        nt0=ntheta+drsz0
        call resize_psf('psfres.F90','THETA','NTHETA',nt0, .true.)
     endif
#else
     IF(NTHETA.GT.MAXT) goto 900
#endif
     !     nglList(,NTHETA)=nglList(,NTHETA-1)
     IT(NTHETA)=IT(NTHETA-1)
     JT(NTHETA)=JT(NTHETA-1)
     KT(NTHETA)=KT(NTHETA-1)
     LTHETA(NTHETA)=LTHETA(NTHETA-1)
     !
     IT(NTHETA-1)=IAMMFF
     JT(NTHETA-1)=I
     KT(NTHETA-1)=ID
     LTHETA(NTHETA-1)=0
     !
     IT(NTHETA+1)=IBMMFF
     JT(NTHETA+1)=I
     KT(NTHETA+1)=ID
     LTHETA(NTHETA+1)=0
     !
     IT(NTHETA+2)=IC
     JT(NTHETA+2)=I
     KT(NTHETA+2)=ID
     LTHETA(NTHETA+2)=0
     NTHETA=NTHETA+2
#ifdef KEY_RESIZE
     if (ntheta .gt. nt0) then
        nt0=ntheta+drsz0
        call resize_psf('psfres.F90','THETA','NTHETA',nt0, .true.)
     endif
#endif
200  CONTINUE
  ENDDO
#ifdef KEY_RESIZE
  if (ntheta .lt. nt0) &
       call resize_psf('psfres.F90','THETA','NTHETA',ntheta, .true.)
#endif
  !C      IF (NTHETA.EQ.0) RETURN
  !CC  CONVERT THE nglList LIST (FOR ANGLE A-B-C) FROM ABCX TO XABC.
  !C      DO 400 I=1,NTHETA
  !CC     CALL UNPACK(nglList(,I),IAMMFF,IBMMFF,IC,ID)
  !C      IAMMFF=IT(I)
  !C      IBMMFF=JT(I)
  !C      IC=KT(I)
  !C      ID=LTHETA(I)
  !C      IT(I)=ID
  !C      JT(I)=IAMMFF
  !C      KT(I)=IBMMFF
  !C      LTHETA(I)=IC
  !CC     WRITE(6,450) ID,IAMMFF,ibmmff,IC
  !CC     WRITE(12,450) ID,IAMMFF,ibmmff,IC
  !CC 450 FORMAT(' ANGLE',4I5)
  !C  400 CONTINUE
  RETURN
  !
900 WRITE(SCRTCH,'(A,2(1X,I5))') &
       'to many bond angles:NTHETA,MAXT=',NTHETA,MAXT
  !     IF(IZ.NE.0) WRITE(IZ,310) NTHETA,MAXT
  CALL wrndie(-5,'<theta>',SCRTCH(:45))
  !
  RETURN
END SUBROUTINE THETA

subroutine sorth(n,ra,ia)
  !...
  !... sorts an array ra of length n into ascending numerical order
  !... using the HEAPSORT algorithm
  !...
  !
  !  Jay Banks 22 Nov 95: replaced in-line implicit none with ##INCLUDE
  use chm_kinds
  implicit none
  !
  integer n
  integer ra(n)
  integer ia(n)
  !
  integer i,iia,ir,j,l
  integer rra
  !
  if(n.le.1) return
  l  = n/2 + 1
  ir = n
10 continue
  if(l.gt.1) then
     l = l-1
     rra = ra(l)
     iia = ia(l)
  else
     rra = ra(ir)
     iia = ia(ir)
     ra(ir) = ra(1)
     ia(ir) = ia(1)
     ir = ir - 1
     if(ir.eq.1) then
        ra(1) = rra
        ia(1) = iia
        return
     endif
  endif
  i = l
  j = l + l
20 if(j.le.ir) then
     if(j.lt.ir) then
        if(ra(j).lt.ra(j+1)) j = j+1
     endif
     if(rra.lt.ra(j)) then
        ra(i) = ra(j)
        ia(i) = ia(j)
        i = j
        j = j + j
     else
        j = ir + 1
     endif
     goto 20
  endif
  ra(i) = rra
  ia(i) = iia
  goto 10
  !.....sorth
  !
end subroutine sorth

#endif 
SUBROUTINE NULL_datstr
  RETURN
END SUBROUTINE NULL_datstr

