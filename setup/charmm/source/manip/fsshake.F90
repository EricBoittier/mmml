module fstshk
  use chm_kinds
  use dimens_fcm
  implicit none
  private

  !    MFC fast non-vector shake common
#if KEY_FSSHK==1 /*fsshk_main*/
  !
  !     Heap pointers and constants for Cray fast shake:
  !
  !   HMASSI   -
  !   HMASSJ   -
  !   D2TOL2   -
  !   IFIRSTP  -
  !   ILASTP   -
  !   MXBONI   -
  !   LHONLY   -
  !
  !   OM1      -
  !   OM2      -
  !   OM3      -
  !   CA11     -
  !   CA12     -
  !   CA13     -
  !   CA21     -
  !   CA22     -
  !   CA23     -
  !   CA31     -
  !   CA32     -
  !   CA33     -
  !   CB11     -
  !   CB12     -
  !   CB13     -
  !   CB21     -
  !   CB22     -
  !   CB23     -
  !   CB31     -
  !   CB32     -
  !   CB33     -
  !   A1       -
  !   A2       -
  !   A3       -
  !   B        -
  !   TOLIJ    -
  !   TOLJK    -
  !   TOLKI    -
  !   R12SQ    -
  !   R23SQ    -
  !   R31SQ    -
  !   NUMWATER -
  !   NSTWAT   -
  !
  real(chm_real),allocatable,dimension(:) :: HMASSI,HMASSJ,D2TOL2,ammi
  integer,allocatable,dimension(:) :: shkgp,nshkgp,bshkgp
  INTEGER IFIRSTP,ILASTP
  INTEGER MXBONI
  LOGICAL LHONLY
  !
  !     Heap pointers and constants for Cray water shake:
  !
  real(chm_real) OM1,OM2,OM3
  real(chm_real) CA11,CA12,CA13,CA21,CA22,CA23,CA31,CA32,CA33
  real(chm_real) CB11,CB12,CB13,CB21,CB22,CB23,CB31,CB32,CB33

  real(chm_real),allocatable,dimension(:,:) :: A1,A2,A3
  real(chm_real),allocatable,dimension(:) :: TOLIJ,TOLJK,TOLKI,R12SQ,R23SQ,R31SQ
  INTEGER NUMWATER,NSTWAT
  integer,parameter :: MAXNSHK=6
  integer nsh1,nsh2,nsh3
  integer nsh4,NOTHER,NWATGR

  ! Public variables
  public nother, nwatgr, nsh1, nsh2, nsh3, nsh4, lhonly
#if KEY_DOMDEC==1
  public bshkgp, nstwat, numwater
  public hmassi, hmassj, ammi
#endif

  ! Public subroutines
  public fsrscshk, fsshkini, fstshake

contains

#if KEY_FSSHK==1 /*fsshake*/
  SUBROUTINE FSRSCSHK
    !-----------------------------------------------------------------------
    !     De-allocates memory and initializes variables used by
    !     scalar fast shake routines.
    !
    use exfunc
    use machdep
    use shake
    use psf
    use memory
    !
    IF(NCONST_pll <= 0) RETURN
    !
    !     Give up shake memory.
    !
    call chmdealloc('fsshake.src','FSRSCSHK','HMASSI',NCONST_pll,crl=HMASSI)
    call chmdealloc('fsshake.src','FSRSCSHK','HMASSJ',NCONST_pll,crl=HMASSJ)
    call chmdealloc('fsshake.src','FSRSCSHK','D2TOL2',NCONST_pll,crl=D2TOL2)
    call chmdealloc('fsshake.src','FSRSCSHK','ammi',7*NCONST_pll,crl=ammi)
    call chmdealloc('fsshake.src','FSRSCSHK','shkgp',natom,intg=shkgp)
    call chmdealloc('fsshake.src','FSRSCSHK','nshkgp',nconst_pll,intg=nshkgp)
    call chmdealloc('fsshake.src','FSRSCSHK','bshkgp',nconst_pll,intg=bshkgp)
    IF(NWATGR > 0) then
       call chmdealloc('fsshake.src','FSRSCSHK','A1',3,NWATGR,crl=A1)
       call chmdealloc('fsshake.src','FSRSCSHK','A2',3,NWATGR,crl=A2)
       call chmdealloc('fsshake.src','FSRSCSHK','A3',3,NWATGR,crl=A3)
       call chmdealloc('fsshake.src','FSRSCSHK','TOLIJ',NWATGR,crl=TOLIJ)
       call chmdealloc('fsshake.src','FSRSCSHK','TOLJK',NWATGR,crl=TOLJK)
       call chmdealloc('fsshake.src','FSRSCSHK','TOLKI',NWATGR,crl=TOLKI)
       call chmdealloc('fsshake.src','FSRSCSHK','R12SQ',NWATGR,crl=R12SQ)
       call chmdealloc('fsshake.src','FSRSCSHK','R23SQ',NWATGR,crl=R23SQ)
       call chmdealloc('fsshake.src','FSRSCSHK','R31SQ',NWATGR,crl=R31SQ)
    ENDIF
    LHONLY = .FALSE.
    NOTHER = 0
    NWATGR = 0
    RETURN
  END SUBROUTINE FSRSCSHK

  SUBROUTINE fsSHKINI(ICONB,RENWAT)
    use exfunc
    use machdep
    use psf
    use shake
    use stream
#if KEY_PARALLEL==1
    use parallel      
#endif
    use memory

#if KEY_PARALLEL==1
    real(chm_real) tot_nshake(7)
    integer itot_nshake(7),itot

#else /**/
    integer numnod
#endif 
    !
    INTEGER ICONB
    character(len=*) renwat
    !
    !
#if KEY_PARALLEL==0
    numnod=1
#endif 
    LHONLY = (ICONB == 1)
    call chmalloc('fsshake.src','fsSHKINI','HMASSI',NCONST_pll,crl=HMASSI)
    call chmalloc('fsshake.src','fsSHKINI','HMASSJ',NCONST_pll,crl=HMASSJ)
    call chmalloc('fsshake.src','fsSHKINI','D2TOL2',NCONST_pll,crl=D2TOL2)
    call chmalloc('fsshake.src','fsSHKINI','ammi',7*NCONST_pll,crl=ammi)

    call chmalloc('fsshake.src','fsSHKINI','shkgp',natom,intg=shkgp)
    call chmalloc('fsshake.src','fsSHKINI','nshkgp',nconst_pll,intg=nshkgp)
    call chmalloc('fsshake.src','fsSHKINI','bshkgp',nconst_pll,intg=bshkgp)

    CALL fsSHKIN2(natom,AMASS,IMOVE,HMASSI,HMASSJ, &
         D2TOL2,MXBONI,LHONLY, &
         NOTHER,NWATGR,renwat &
         ,shkapr,nconst_pll,idgf2,constr,shktol &
         ,shkgp,nshkgp ,nsh1,nsh2,nsh3,nsh4 &
         ,bshkgp &
         ,ammi &
         )
    IF(NWATGR > 0) THEN
       call chmalloc('fsshake.src','fsSHKINI','A1',3,NWATGR,crl=A1)
       call chmalloc('fsshake.src','fsSHKINI','A2',3,NWATGR,crl=A2)
       call chmalloc('fsshake.src','fsSHKINI','A3',3,NWATGR,crl=A3)
       call chmalloc('fsshake.src','fsSHKINI','TOLIJ',NWATGR,crl=TOLIJ)
       call chmalloc('fsshake.src','fsSHKINI','TOLJK',NWATGR,crl=TOLJK)
       call chmalloc('fsshake.src','fsSHKINI','TOLKI',NWATGR,crl=TOLKI)
       call chmalloc('fsshake.src','fsSHKINI','R12SQ',NWATGR,crl=R12SQ)
       call chmalloc('fsshake.src','fsSHKINI','R23SQ',NWATGR,crl=R23SQ)
       call chmalloc('fsshake.src','fsSHKINI','R31SQ',NWATGR,crl=R31SQ)
       CALL fsWATINI(NWATGR,NOTHER,A1,A2,A3, &
            R12SQ,R23SQ,R31SQ,TOLIJ,TOLJK, &
            TOLKI,CA11,CA12,CA13,CA21,CA22,CA23,CA31,CA32,CA33,CB11, &
            CB12,CB13,CB21,CB22,CB23,CB31,CB32,CB33,OM1,OM2,OM3,NUMWATER, &
            NSTWAT &
            ,shkapr,nconst_pll,idgf2,constr,shktol)
    ENDIF
    !

    IF(PRNLEV > 2) THEN
       WRITE(OUTU,45) NOTHER,NWATGR
45     FORMAT(' FSSHKINI: Fast shake initialized with',I8, &
            ' bond contraints and ',I8,' water constraints (3-center).')
       WRITE(OUTU,46) nother,nwatgr
46     FORMAT(' FSSHKINI: ',I8,' constraints are in groups of: ',I8)
       WRITE(OUTU,47) nsh1,nsh2,nsh3,nsh4
47     FORMAT(' FSSHKINI: ',I6,' 2-body (CH), ', &
            I6,' 3-body(CH2), ',I6,' 4-body(CH3), ',I6,' >4-body. ')
    ENDIF
    !
    RETURN
  END SUBROUTINE fsSHKINI

  subroutine fsshkin2(natom,amass,imove,hmassi,hmassj,d2tol2, &
       mxboni,lhonly, &
       nother,nwatgr,renwat &
       ,shkapr,nconst,idgf2,constr,shktol &
       ,shkgp,nshkgp  ,nsh1,nsh2,nsh3 ,nsh4 &
       ,bshkgp,ammi )
    !-----------------------------------------------------------------------
    !     Sort the constraint index arrays according to the schemes
    !     required for the vector-parallel shake algorithms, and
    !     store some quantities which only need to be computed once.
    !
    !     Authors: Doug Tobias and John Mertz
    !
    use exfunc
    use number
    use stream
#if KEY_PARALLEL==1
    use parallel        
#endif
    use memory
    use chutil,only:hydrog,atomid

    integer nconst
    integer shkapr(2,MAXSHK),idgf2(*)
    real(chm_real) constr(*),shktol
    integer shkgp(*),nshkgp(*),ishkgp,mshkgp,nsh1,nsh2,nsh3,nsh4
    integer bshkgp(*),igp
    real(chm_real) ammi(*)
    !
#if KEY_PARALLEL==0
    integer mynod
#endif 
    !
    real(chm_real) amass(*),hmassi(*),hmassj(*),d2tol2(*)

    integer mxboni,nother,nwatgr,natom
    integer imove(*)
#if KEY_PARALLEL==1
    logical qperror        
#endif
    logical lhonly
    character(len=*) renwat
    INTEGER ATFRST,ATLAST
    integer i,j,k,l,i1,i2
    real(chm_real) mi,mj,mk,ml
    !
    integer iconst,ibegin,iend,ix,jx,kx
    integer nboni,nused,nwater,ngroup
    character(len=8) sid,rid,ren,ac,renbb
    character(len=8) sidi,sidj,ridi,ridj,reni,renj
    external exch5
    real(chm_real) RMASS,TOL2,AMSI,AMSJ,RMHALF
    !
    !     Only 3-site TIP3P waters are currently handled by the fast
    !     fast matrix water shake.  This can easily be made more general.
    !      (and was... BRB - 12/4/97)
    !
    integer :: nbdwat=3
    !
    integer,allocatable,dimension(:) :: ITEMP,JTEMP,KTEMP,gTEMP,igroup
    real(chm_real),allocatable,dimension(:) :: RTEMP
    logical,allocatable,dimension(:) :: LWATER


    call chmalloc('fsshake.src','fsshkin2','ITEMP',Nconst,intg=ITEMP)
    call chmalloc('fsshake.src','fsshkin2','JTEMP',Nconst,intg=JTEMP)
    call chmalloc('fsshake.src','fsshkin2','KTEMP',Nconst,intg=KTEMP)
    call chmalloc('fsshake.src','fsshkin2','gTEMP',Nconst,intg=gTEMP)
    call chmalloc('fsshake.src','fsshkin2','RTEMP',Nconst,crl=RTEMP)
    call chmalloc('fsshake.src','fsshkin2','LWATER',Nconst,log=LWATER)
    call chmalloc('fsshake.src','fsshkin2','IGROUP',Nconst,intg=IGROUP)

    !     Fill the temporary arrays for sorting.  For hydrogen-only constrints,
    !     the first indices (shkapr(1,i)) are the heavy atom numbers, except
    !     for the tip3 h-h constraint, where the lowest hydrogen atom number
    !     comes first.  For the general case, we put the lowest atom numbers
    !     first.
    !
    nwater = 0
    nother = 0
    ATFRST=1
    ATLAST=NATOM

#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
       !     Define the atom bounds for this processor.
       ATFRST=1+IPARPT(MYNOD)
       ATLAST=IPARPT(MYNODP)
#else /**/
       call wrndie(-5,'<FSSHKIN2>', &
            'FAST SHAKE will not work with PARASCAL')
#endif 
#else /**/
       mynod=0
#endif 

    !
    do  iconst = 1,nconst
       lwater(iconst)=.false.
    enddo

    do iconst = 1,nconst
       i = shkapr(1,iconst)
       IF(I == 0) cycle
       j = shkapr(2,iconst)
#if KEY_PARALLEL==1
          IF(I < ATFRST .OR. I > ATLAST) THEN
             IF(J < ATFRST .OR. J > ATLAST) cycle
             QPERROR=.TRUE.
             cycle
          elseIF(J < ATFRST .OR. J > ATLAST) THEN
             QPERROR=.TRUE.
             cycle
          ENDIF
#endif 
       call atomid(i,sid,rid,ren,ac)
       !
       if(ren == renwat) then
          if(idgf2(i) == 4 .and. idgf2(j) == 4) then
             lwater(iconst) = .true.
             nwater = nwater + 1
          endif
       endif
       !
       if(hydrog(i).and..not.hydrog(j)) then
          itemp(iconst) = j
          jtemp(iconst) = i
       else if(.not.hydrog(i).and.hydrog(j)) then
          itemp(iconst) = i
          jtemp(iconst) = j
       else if(hydrog(i).and.hydrog(j)) then
          if(i > j) then
             itemp(iconst) = j
             jtemp(iconst) = i
          else
             itemp(iconst) = i
             jtemp(iconst) = j
          endif
       else
          if(lhonly) call wrndie(-4,'<CSHKIN2>', &
               'Programmer error: found constraint not involving an H atom.')
          if(i > j) then
             itemp(iconst) = j
             jtemp(iconst) = i
          else
             itemp(iconst) = i
             jtemp(iconst) = j
          endif
       endif
       rtemp(iconst) = constr(iconst)
       ktemp(iconst) = iconst
    enddo
    !
    !     Now we sort the arrays so that the first constraint index is ascending.
    !     For the hydrogen-only case, the second index will be ascending also.
    !     In addition, we first rearrange the arrays so the water constraints
    !     (if there are any) are last.
    !
    i = nconst

    iswat_loop: do while( lwater(i))
       i = i - 1
       if(i == 0 ) exit iswat_loop
    enddo iswat_loop
    j = i
    do iconst = 1,j
       if(iconst >= i) cycle
       if(lwater(iconst)) then
          ix = itemp(iconst)
          jx = jtemp(iconst)
          kx = ktemp(iconst)
          itemp(iconst) = itemp(i)
          jtemp(iconst) = jtemp(i)
          ktemp(iconst) = ktemp(i)
          itemp(i) = ix
          jtemp(i) = jx
          ktemp(i) = kx
          i = i - 1

          iswat2_loop: do while(lwater(i))
             i = i - 1
             ! BIOVIA Code Start : Bug fix
             if(i < iconst) exit iswat2_loop
             ! BIOVIA Code End
          enddo iswat2_loop
       endif
    enddo
    nother = nconst - nwater
    if(nother > 0) &
         call sort(nother,exch5,order5,itemp,jtemp,ktemp,0,0,0,0,3)
    if(nwater > 0)then
       ibegin = nother + 1
       call sort(nwater,exch5,order5,itemp(ibegin),jtemp(ibegin), &
            ktemp(ibegin),0,0,0,0,3)
    endif
    !
    !     Now we copy the sorted temporary arrays into the constraint arrays.
    !     We also find the maximum number of constraints involving a given
    !     heavy atom, to determine the stride through the vector loops,
    !     and we fill the group array with constraint group numbers to assist
    !     the partitioning of the constraint arrays into sections (see
    !     below).  Also, we compute and store the quantities which only
    !     need to be computed once.
    !
    mxboni = 1
    nboni = 1
    ngroup = 1
    tol2 = two*shktol
    do iconst = 1,nother
       if(iconst > 1) then
          i = itemp(iconst)
          j = itemp(iconst-1)
          call atomid(i,sidi,ridi,ren,ac)
          call atomid(j,sidj,ridj,ren,ac)
          if(i == j.or.(ren == renwat.and.sidi == sidj.and. &
               ridi == ridj)) then
             nboni = nboni + 1
          else
             nboni = 1
             ngroup = ngroup + 1
          endif
          if(nboni > mxboni) mxboni = nboni
       endif
       shkapr(1,iconst) = itemp(iconst)
       shkapr(2,iconst) = jtemp(iconst)
       constr(iconst) = rtemp(ktemp(iconst))
       d2tol2(iconst) = constr(iconst)*tol2
       igroup(iconst) = ngroup
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       !mu...25-Jul-93 Correct expressions for hmassi and hmassj with fixed atoms
       rmass=amass(i)*amass(j)/(amass(i)+amass(j))
       rmhalf= half*rmass
       if(imove(i) > 0) then
          amsi = zero
       else
          amsi = one/amass(i)
       endif
       if(imove(j) > 0) then
          amsj = zero
       else
          amsj = one/amass(j)
       endif
       hmassi(iconst) = amsi*rmhalf
       hmassj(iconst) = amsj*rmhalf
       !mu...25-Jul-93
    enddo
    do iconst = nother+1,nconst
       shkapr(1,iconst) = itemp(iconst)
       shkapr(2,iconst) = jtemp(iconst)
       constr(iconst) = rtemp(ktemp(iconst))
       igroup(iconst) = ngroup
    enddo
    nwatgr = nwater/nbdwat
    !---------------------------------------------------------------
    mshkgp=0
    do i=1,natom
       shkgp(i)=0
    enddo
    do i=1,nother
       nshkgp(i)=0
    enddo
    do iconst=1,nother
       i=shkapr(1,iconst)
       j=shkapr(2,iconst)
       ishkgp=shkgp(i)
       if(ishkgp == 0)then
          if(shkgp(j) == 0)then
             !           ----- New group -----
             mshkgp=mshkgp+1
             !              ---- back-pointer to the first constraint # i this group ---
             bshkgp(mshkgp)=iconst
             !              ---- number of atoms in this group ----
             nshkgp(mshkgp)=1
             !              ---- identity of group for this constraint ----
             shkgp(i)=mshkgp
             shkgp(j)=mshkgp
          else
             CALL WRNDIE(-1,'<fsshkin2>', &
                  'atom already in another shake group')
          endif
       else
          if(shkgp(j) == 0)then
             !              ---- add this atom to atom i's group ----
             shkgp(j)=ishkgp
             nshkgp(ishkgp)=nshkgp(ishkgp)+1
          elseif(shkgp(j) == ishkgp)then
             nshkgp(ishkgp)=nshkgp(ishkgp)+1
          else
             CALL WRNDIE(-1,'<fsshkin2>', &
                  'atom already in another shake group')
          endif
       endif
    enddo
    !---------------------------------------------------------------
    if(nother > 0) &
         call sort(mshkgp,exch5,order5,nshkgp,bshkgp,0,0,0,0,0,2)
    nsh1=0
    nsh2=0
    nsh3=0
    nsh4=0
    !     ----- Initialize mask array for untreated constraints -----
    do i1=1,nconst
       gtemp(i1)=1
    enddo
    !     ----- Count number of each kind of constraint -----
    do igp=1,mshkgp
       if(nshkgp(igp) == 1)then
          nsh1=nsh1+1
       elseif(nshkgp(igp) == 2)then
          nsh2=nsh2+1
       elseif(nshkgp(igp) == 3)then
          nsh3=nsh3+1
       else
          nsh4=nsh4+1
       endif
    enddo
    do i1 = 1,nsh1
       i2=i1*2-1
       iconst = bshkgp(i1)
       gtemp(iconst)=0
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       mi=amass(i)
       mj=amass(j)
       ammi(i2)=one/mi
       ammi(i2+1)=one/mj
    enddo
    i2=2*nsh1+1
    do i1 = nsh1+1,nsh2+nsh1
       iconst = bshkgp(i1)
       gtemp(iconst)=0
       gtemp(iconst+1)=0
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       k = shkapr(2,iconst+1)
       mi=amass(i)
       mj=amass(j)
       mk=amass(k)
       ammi(i2)=one/mi
       ammi(i2+1)=one/mj
       ammi(i2+2)=one/mk
       ammi(i2+3)=(mi+mj)/(mi*mj)
       ammi(i2+4)=(mi+mk)/(mi*mk)
       i2=i2+5
    enddo
    i2=2*nsh1+5*nsh2+1
    do i1 = nsh2+nsh1+1,nsh2+nsh1+nsh3
       iconst = bshkgp(i1)
       gtemp(iconst)=0
       gtemp(iconst+1)=0
       gtemp(iconst+2)=0
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       k = shkapr(2,iconst+1)
       l = shkapr(2,iconst+2)
       mi=amass(i)
       mj=amass(j)
       mk=amass(k)
       ml=amass(l)
       ammi(i2)=one/mi
       ammi(i2+1)=one/mj
       ammi(i2+2)=one/mk
       ammi(i2+3)=one/ml
       ammi(i2+4)=(mi+mj)/(mi*mj)
       ammi(i2+5)=(mi+mk)/(mi*mk)
       ammi(i2+6)=(mi+ml)/(mi*ml)
       i2=i2+7
    enddo
    nsh4=0
    do i1=1,nother
       if(gtemp(i1) == 1)then
          nsh4=nsh4+1
       endif
    enddo
    if(prnlev  > 2)then
       WRITE(OUTU,'(a,i8,a,i8,a)') &
            ' FSSHKINI: Fast shake initialized with',   nother, &
            ' bond contraints and ',   NWATGR, &
            ' water constraints (3-center).'
       WRITE(OUTU,'(a,i8,a,i8)') ' FSSHKINI: ',nother, &
            ' constraints are in groups of: ',nwatgr
       WRITE(OUTU,'(a,i6,a,i6,a,i6,a,i6,a)') &
            ' FSSHKINI: ',   nsh1,' 2-body (CH), ',nsh2, &
            ' 3-body(CH2), ',nsh3,' 4-body(CH3), ',nsh4,' >4-body. '
       if(nsh4 > 0)then
          write(OUTU,'(a)') &
               "FAST SHAKE SETUP found a group with more than 3 constraints."
          call wrndie(-2,'<FSSHKIN2>', &
               'FAST SHAKE will not treat  >  4-body constratins')
       endif
    endif
    !-----------------------------------------------------------------------
    call chmdealloc('fsshake.src','fsshkin2','ITEMP',Nconst,intg=ITEMP)
    call chmdealloc('fsshake.src','fsshkin2','JTEMP',Nconst,intg=JTEMP)
    call chmdealloc('fsshake.src','fsshkin2','KTEMP',Nconst,intg=KTEMP)
    call chmdealloc('fsshake.src','fsshkin2','gTEMP',Nconst,intg=gTEMP)
    call chmdealloc('fsshake.src','fsshkin2','RTEMP',Nconst,crl=RTEMP)
    call chmdealloc('fsshake.src','fsshkin2','LWATER',Nconst,log=LWATER)
    call chmdealloc('fsshake.src','fsshkin2','IGROUP',Nconst,intg=IGROUP)
    return
  end subroutine fsshkin2

  subroutine fswatini(nwatgr,nother,a1,a2,a3,r12sq,r23sq,r31sq, &
       tolij,toljk,tolki,ca11,ca12,ca13,ca21,ca22,ca23,ca31,ca32, &
       ca33,cb11,cb12,cb13,cb21,cb22,cb23,cb31,cb32,cb33,om1,om2,om3, &
       numwater,nstwat &
       ,shkapr,nconst,idgf2,constr,shktol)

    !-----------------------------------------------------------------------
    !     Initialization of shake variables for water.
    !
    !     Authors: John Mertz and Doug Tobias
    !
    !     Sort must be made in this order: O H1, O H2, H1 H2.
    !
    use number
    use psf
    implicit none

    integer nwatgr,nother,numwater,nstwat
    integer ibond,iwater
    real(chm_real) a1(3,*),a2(3,*),a3(3,*)
    real(chm_real) r12sq(*),r23sq(*),r31sq(*)
    real(chm_real) tolij(*),toljk(*),tolki(*)
    real(chm_real) ca11,ca12,ca13,ca21,ca22,ca23,ca31,ca32,ca33
    real(chm_real) cb11,cb12,cb13,cb21,cb22,cb23,cb31,cb32,cb33
    real(chm_real) om1,om2,om3
    real(chm_real) constr(*),shktol
    integer shkapr(2,MAXSHK),nconst,idgf2(*)
    !
    !
    !     Calculate and store some invariant coefficients used in
    !     the matrix water shake.
    !
    if(nwatgr <= 0) return
    !
    nstwat=nother+1
    om1= one/amass(shkapr(1,nstwat))
    om2= one/amass(shkapr(2,nstwat))
    om3= one/amass(shkapr(2,nstwat))
    ca11 =   two   * ( om1 + om2 ) * om2
    ca12 = - two   * om1 * om2
    ca13 =   two   * ( om1 + om2 ) * om1
    ca21 =   two   * ( om2 + om3 ) * om2
    ca22 =   two   * ( om2 + om3 ) * om3
    ca23 = - two   * om2 * om3
    ca31 = - two   * om1 * om3
    ca32 =   two   * ( om1 + om3 ) * om3
    ca33 =   two   * ( om1 + om3 ) * om1
    cb11 =   two   * ( om1 + om2 )
    cb12 = - two   *   om2
    cb13 = - two   *   om1
    cb21 = - two   *   om2
    cb22 =   two   * ( om2 + om3 )
    cb23 = - two   *   om3
    cb31 = - two   *   om1
    cb32 = - two   *   om3
    cb33 =   two   * ( om1 + om3 )
    !
    iwater=0
    do ibond=nstwat,nconst,3
       iwater=iwater+1
       r12sq(iwater) = constr(ibond)
       r31sq(iwater) = constr(1+ibond)
       r23sq(iwater) = constr(2+ibond)
       tolij(iwater) = shktol*sqrt(r12sq(iwater))
       toljk(iwater) = shktol*sqrt(r23sq(iwater))
       tolki(iwater) = shktol*sqrt(r31sq(iwater))
       a1(1,iwater) = - r12sq(iwater) * ( om1 + om2 ) ** 2
       a2(1,iwater) = - r12sq(iwater) * om2 ** 2
       a3(1,iwater) = - r12sq(iwater) * om1 ** 2
       a1(2,iwater) = - r23sq(iwater) * om2 ** 2
       a2(2,iwater) = - r23sq(iwater) * ( om2 + om3 ) ** 2
       a3(2,iwater) = - r23sq(iwater) * om3 ** 2
       a1(3,iwater) = - r31sq(iwater) * om1 ** 2
       a2(3,iwater) = - r31sq(iwater) * om3 ** 2
       a3(3,iwater) = - r31sq(iwater) * ( om1 + om3 ) ** 2
    enddo
    !
    numwater=iwater
    !
    return
  end subroutine fswatini

#if KEY_DOMDEC==1
  ! *
  ! * Use SETTLE algorithm to constrain water using an analytical formula
  ! *
  ! * (x0, y0, z0) = coordinates at time t
  ! * (x1, y1, z1) = coordinates at time t + delta t
  ! *
  ! * Antti-Pekka Hynninen, May 2013
  ! * NOTE: parts of this subroutine are copied form Amber shake.F90
  ! *
  ! * SETTLE algorithm is described in
  ! * Shuichi Miyamoto and Peter A. Kollman, Journal of Computational Chemistry 13 p. 952 (1992)
  ! *
  ! * NOTE: we use the same notation as in the article, where the atoms in the water
  ! *       molecule are refered to as: A = O, B = H, and C = H
  ! *
  subroutine settle_water(x0, y0, z0, x1, y1, z1, numwater, nstwat, shkapr)
    use number,only:one,half,two,four
    use fsshake_kernel,only:settle_water_kernel_d0
    use domdec_common,only:divide_thread_work, q_test
    use domdec_shake,only:nshakewater, shakewater_ind, mO, mH, rHHsq, rOHsq
    use fsshake_kernel,only:settle_water_kernel_d1
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: x0(*), y0(*), z0(*)
    real(chm_real), intent(inout) :: x1(*), y1(*), z1(*)
    integer, intent(in) :: numwater, nstwat
    integer, intent(in) :: shkapr(2,MAXSHK)
    ! Variables
    real(chm_real) mH2O, mO_div_mH2O, mH_div_mH2O
    real(chm_real) ra, rc, rb, ra_inv, rc2
    integer imol_start, imol_end  

    ! Setup masses
    mH2O = mO + mH + mH
    mO_div_mH2O = mO/mH2O
    mH_div_mH2O = mH/mH2O

    ! Setup ra, rb, rc
    ra = mH_div_mH2O*sqrt(four*rOHsq - rHHsq)
    ra_inv = one/ra
    rb = ra*mO/(two*mH)
    rc = sqrt(rHHsq)/two
    rc2 = two*rc

!$omp parallel private(imol_start, imol_end)
    call divide_thread_work(nshakewater, imol_start, imol_end)
    call settle_water_kernel_d1(imol_start, imol_end, shakewater_ind, &
         x0, y0, z0, x1, y1, z1, mO_div_mH2O, mH_div_mH2O, ra, rc, rb, ra_inv, rc2)
!$omp end parallel
    if (q_test) then
       call test_settle_water(nshakewater, shakewater_ind, rOHsq, rHHsq, x1, y1, z1)
    endif

    return
  end subroutine settle_water

  ! *
  ! * Tests settle_water -subroutine by checking that the O-H and H-H distances equal the given
  ! * constants
  ! *
  subroutine test_settle_water(nmol, shakewater_ind, rOHsq, rHHsq, x, y, z)
    use stream,only:outu, prnlev
    implicit none
    ! Input / Output
    integer, intent(in) :: nmol, shakewater_ind(:)
    real(chm_real), intent(in) :: rOHsq, rHHsq, x(*), y(*), z(*)
    ! Parameters
    real(chm_real), parameter :: errtol = 1.0e-10_chm_real
    ! Variables
    real(chm_real) rsq
    integer imol, i, j, k

    do imol=1,nmol
       i = shakewater_ind(imol*3-2)      ! O
       j = shakewater_ind(imol*3-1)      ! H
       k = shakewater_ind(imol*3)        ! H

       rsq = (x(i) - x(j))**2 + (y(i) - y(j))**2 + (z(i) - z(j))**2
       if (abs(rsq - rOHsq) > errtol) then
          write (outu,'(a,i8,2f12.6,g12.4)') 'imol, rsq, rOHsq, diff=',&
               imol,rsq,rOHsq,abs(rsq - rOHsq)
          call wrndie(-5,'<fsshake>','test_settle_water: incorrect O-H distance')
       endif

       rsq = (x(i) - x(k))**2 + (y(i) - y(k))**2 + (z(i) - z(k))**2
       if (abs(rsq - rOHsq) > errtol) then
          write (outu,'(a,i8,2f12.6,g12.4)') 'imol, rsq, rOHsq, diff=',&
               imol,rsq,rOHsq,abs(rsq - rOHsq)
          call wrndie(-5,'<fsshake>','test_settle_water: incorrect O-H distance')
       endif
       
       rsq = (x(k) - x(j))**2 + (y(k) - y(j))**2 + (z(k) - z(j))**2
       if (abs(rsq - rHHsq) > errtol) then
          write (outu,'(a,i8,2f12.6,g12.4)') 'imol, rsq, rHHsq, diff=',&
               imol,rsq,rHHsq,abs(rsq - rHHsq)
          call wrndie(-5,'<fsshake>','test_settle_water: incorrect H-H distance')
       endif

    enddo

    if (prnlev > 2) write (outu,'(a)') 'test_settle_water OK'

    return
  end subroutine test_settle_water
#endif

  subroutine fsshakwat(niter,x,y,z,xref,yref,zref,nwatgr,nother, &
       a1,a2,a3,tolij,toljk,tolki,r12sq,r23sq,r31sq,ca11,ca12,ca13, &
       ca21,ca22,ca23,ca31,ca32,ca33,cb11,cb12,cb13,cb21,cb22,cb23, &
       cb31,cb32,cb33,om1,om2,om3,numwater,nstwat &
       ,shkapr,nconp,idgf2,constr,shktol,mxiter)
    !-----------------------------------------------------------------------
    !     Matrix shake routine for three-site water molecules.
    !
    !     Authors: John Mertz and Doug Tobias
    !
    !     APH: (1) Index order changed in diffwat (formerly diff) and bwat (formerly b)
    !          (2) Memory allocation moved outside the function call
    !          (3) Indexing changed for watxyz (formerly space_xyz) to minimize cache misses
    !          xyzwat(1,i) = xij
    !          xyzwat(2,i) = yij
    !          xyzwat(3,i) = zij
    !          xyzwat(4,i) = xjk
    !          xyzwat(5,i) = yjk
    !          xyzwat(6,i) = zjk
    !          xyzwat(7,i) = xki
    !          xyzwat(8,i) = yki
    !          xyzwat(9,i) = zki
    !          xyzwat(10,i) = dij
    !          xyzwat(11,i) = djk
    !          xyzwat(12,i) = dki
    !          
    !
    use number
    use memory

    integer, intent(in) :: nconp,idgf2(*),shkapr(2,MAXSHK),mxiter
    real(chm_real), intent(in) :: constr(nconp),shktol
    !
    integer, intent(inout) :: niter
    integer, intent(in) :: nwatgr,nother,numwater,nstwat
    real(chm_real), intent(inout) :: x(*),y(*),z(*)
    real(chm_real), intent(in) :: xref(*),yref(*),zref(*)
    real(chm_real), intent(in) :: a1(3,*),a2(3,*),a3(3,*)
    real(chm_real), intent(in) :: tolij(*),toljk(*),tolki(*)
    real(chm_real), intent(in) :: r12sq(*),r23sq(*),r31sq(*)

    real(chm_real), intent(in) :: ca11,ca12,ca13,ca21,ca22,ca23,ca31,ca32,ca33
    real(chm_real), intent(in) :: cb11,cb12,cb13,cb21,cb22,cb23,cb31,cb32,cb33
    real(chm_real), intent(in) :: om1,om2,om3
    !
    real(chm_real) c11,c12,c13,c21,c22,c23,c31,c32,c33
    real(chm_real) xpij,ypij,zpij,xpki,ypki,zpki,xpjk,ypjk,zpjk
    real(chm_real) d1,d2,d3
    real(chm_real) drij,drjk,drki
    real(chm_real) drij_new,drjk_new,drki_new
    real(chm_real) xij, yij, zij
    real(chm_real) xki, yki, zki
    real(chm_real) xjk, yjk, zjk
    real(chm_real) bwat1, bwat2, bwat3, bwat4, bwat5, bwat6, bwat7, bwat8, bwat9
    real(chm_real) diffwat1, diffwat2, diffwat3
    real(chm_real) a12, a22, a32, a14, a24, a34, a16, a26, a36
    integer imol,nmol,ipointer,i,j,k
    integer ifirst,ilast
    real(chm_real) XNORM,R1
    logical donea
    !
    integer, parameter :: dim=128
#if KEY_DEBUG==1
    integer jjj
#endif 

    !
    !     Loop over water molecules and not bonds.
    !

    nmol = numwater
    do imol=1,nmol
       ipointer=3*(imol-1)+nstwat
       i = shkapr(1,ipointer)
       j = shkapr(2,ipointer)
       k = shkapr(2,ipointer+1)
       
       xij = xref(i)-xref(j)
       yij = yref(i)-yref(j)
       zij = zref(i)-zref(j)
       xki = xref(k)-xref(i)
       yki = yref(k)-yref(i)
       zki = zref(k)-zref(i)
       xjk = xref(j)-xref(k)
       yjk = yref(j)-yref(k)
       zjk = zref(j)-zref(k)
       
       r1 = xij*xjk + yij*yjk + zij*zjk
       a12 = ca11*r1
       a22 = ca21*r1
       a32 = ca31*r1
       r1 = xjk*xki + yjk*yki + zjk*zki
       a14 = ca12*r1
       a24 = ca22*r1
       a34 = ca32*r1
       r1 = xki*xij + yki*yij + zki*zij
       
       a16 = ca13*r1
       a26 = ca23*r1
       a36 = ca33*r1
       xpij=x(i)-x(j)
       ypij=y(i)-y(j)
       zpij=z(i)-z(j)
       xpki=x(k)-x(i)
       ypki=y(k)-y(i)
       zpki=z(k)-z(i)
       xpjk=x(j)-x(k)
       ypjk=y(j)-y(k)
       zpjk=z(j)-z(k)
       diffwat1 = r12sq(imol)-xpij*xpij-ypij*ypij-zpij*zpij
       diffwat2 = r23sq(imol)-xpjk*xpjk-ypjk*ypjk-zpjk*zpjk
       diffwat3 = r31sq(imol)-xpki*xpki-ypki*ypki-zpki*zpki
       c11=(xpij*xij+ypij*yij+zpij*zij)*cb11
       c12=(xpij*xjk+ypij*yjk+zpij*zjk)*cb12
       c13=(xpij*xki+ypij*yki+zpij*zki)*cb13
       c21=(xpjk*xij+ypjk*yij+zpjk*zij)*cb21
       c22=(xpjk*xjk+ypjk*yjk+zpjk*zjk)*cb22
       c23=(xpjk*xki+ypjk*yki+zpjk*zki)*cb23
       c31=(xpki*xij+ypki*yij+zpki*zij)*cb31
       c32=(xpki*xjk+ypki*yjk+zpki*zjk)*cb32
       c33=(xpki*xki+ypki*yki+zpki*zki)*cb33

       bwat1 = (c22*c33-c23*c32)
       bwat2 = (c31*c23-c33*c21)
       bwat3 = (c21*c32-c22*c31)
       xnorm=one/(c11*bwat1+c12*bwat2+c13*bwat3)
       bwat1 = bwat1*xnorm
       bwat2 = bwat2*xnorm
       bwat3 = bwat3*xnorm
       bwat4 = (c32*c13-c33*c12)*xnorm
       bwat5 = (c11*c33-c13*c31)*xnorm
       bwat6 = (c31*c12-c32*c11)*xnorm
       bwat7 = (c12*c23-c13*c22)*xnorm
       bwat8 = (c21*c13-c23*c11)*xnorm
       bwat9 = (c11*c22-c12*c21)*xnorm
          
       ! Iterate
       niter = 0
       drij  = zero
       drjk  = zero
       drki  = zero
       
       donea = .false.
       
       do while (niter <= mxiter)
          c11=drij*drij
          c12=drij*drjk
          c22=drjk*drjk
          c23=drjk*drki
          c33=drki*drki
          c31=drki*drij
          
          d1=a1(1,imol)*c11+a12*c12+a1(2,imol)*c22+ &
               a14*c23+a1(3,imol)*c33+a16*c31+ &
               diffwat1
          d2=a2(1,imol)*c11+a22*c12+a2(2,imol)*c22+ &
               a24*c23+a2(3,imol)*c33+a26*c31+ &
               diffwat2
          d3=a3(1,imol)*c11+a32*c12+a3(2,imol)*c22+ &
               a34*c23+a3(3,imol)*c33+a36*c31+ &
               diffwat3
          drij_new=bwat1*d1+bwat4*d2+bwat7*d3
          drjk_new=bwat2*d1+bwat5*d2+bwat8*d3
          drki_new=bwat3*d1+bwat6*d2+bwat9*d3

          donea=((abs(drij_new-drij) <= tolij(imol)).and. &
               (abs(drjk_new-drjk) <= toljk(imol)).and. &
               (abs(drki_new-drki) <= tolki(imol)))
          drij = drij_new
          drjk = drjk_new
          drki = drki_new

          niter = niter + 1
          if (donea) exit
       enddo
       
       ! BIOVIA Code Start : Bug fix
       if (niter > mxiter) then
          call wrndie(10,'<fsshakwat>','SHAKE not converged')
       endif
       ! BIOVIA Code End

       x(i)=x(i)+om1*(drij*xij-drki*xki)
       y(i)=y(i)+om1*(drij*yij-drki*yki)
       z(i)=z(i)+om1*(drij*zij-drki*zki)
       x(j)=x(j)+om2*(drjk*xjk-drij*xij)
       y(j)=y(j)+om2*(drjk*yjk-drij*yij)
       z(j)=z(j)+om2*(drjk*zjk-drij*zij)
       x(k)=x(k)+om3*(drki*xki-drjk*xjk)
       y(k)=y(k)+om3*(drki*yki-drjk*yjk)
       z(k)=z(k)+om3*(drki*zki-drjk*zjk)

    enddo

    !
    return
  end subroutine fsshakwat

  !-----------------------------------------------------------------------
  !     ***********************************
  !       ------------ FSSHAKPH -------
  !     ***********************************
  !-----------------------------------------------------------------------
  subroutine fsshakph(x,y,z,xref,yref,zref,niter,nconst, &
       hmassi,hmassj,d2tol2,nother, &
       shkapr,nconp,constr,shktol,mxiter, &
       nsh1,nsh2,nsh3,nsh4,bshkgp,ammi)
    !-----------------------------------------------------------------------
    !     Parallel routine for shake constraints consisting of
    !     only bonds to hydrogen atoms.
    !     Sorted into groups of 1, 2, and 3 constraint groups
    !        each solved analytically.
    !     Anything more than 3 is done at once iteratively.
    !
    !     Author: M. Crowley
    !

    use stream
    use number
    use memory
    use fsshake_kernel,only:fsshakph_kernel2_d0, fsshakph_kernel3_d0, fsshakph_kernel4_d0
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, divide_thread_work
    use domdec_shake,only:nshakepair, nshaketrip, nshakequad,&
         shakepair_ind, shakepair_mass, shakepair_constr, &
         shaketrip_ind, shaketrip_mass, shaketrip_constr, &
         shakequad_ind, shakequad_mass, shakequad_constr
    use fsshake_kernel,only:fsshakph_kernel2_d1, fsshakph_kernel3_d1, fsshakph_kernel4_d1
#endif 
    ! Input / Output
    integer, intent(in) :: nconp,shkapr(2,MAXSHK),mxiter,nconst
    real(chm_real), intent(in) :: constr(nconp), shktol
    real(chm_real), intent(inout) :: x(*),y(*),z(*)
    real(chm_real), intent(in) :: xref(*),yref(*),zref(*)
    real(chm_real), intent(in) :: hmassi(*),hmassj(*),d2tol2(*)
    real(chm_real), intent(in) :: ammi(:)
    integer, intent(in) :: nother
    integer, intent(in) :: nsh1,nsh2,nsh3,nsh4,bshkgp(:)
    integer, intent(inout) :: niter
    ! Variables
    real(chm_real) a12tol,a13tol,a14tol
    integer iconst, i, j, niter3, niter4
#if KEY_DOMDEC==1
    integer istart, iend
#endif 

    !
    a13tol=shktol
    a12tol=shktol
    a14tol=shktol

#if KEY_DOMDEC==1
    if (q_domdec) then
       niter3 = 0 
       niter4 = 0
!$omp parallel private(istart, iend) reduction(max:niter3,niter4)
       call divide_thread_work(nshakepair, istart, iend)
       call fsshakph_kernel2_d1(istart, iend, shakepair_ind, shakepair_constr, shakepair_mass, &
            xref, yref, zref, x, y, z)
       call divide_thread_work(nshaketrip, istart, iend)
       call fsshakph_kernel3_d1(istart, iend, shaketrip_ind, shaketrip_constr, shaketrip_mass, &
            a12tol, a13tol, xref, yref, zref, x, y, z, mxiter, niter3)
       call divide_thread_work(nshakequad, istart, iend)
       call fsshakph_kernel4_d1(istart, iend, shakequad_ind, shakequad_constr, shakequad_mass, &
            a12tol, a13tol, a14tol, xref, yref, zref, x, y, z, d2tol2, mxiter, niter4)
!$omp end parallel
    else
#endif 
       call fsshakph_kernel2_d0(1, nsh1, bshkgp, shkapr, constr, hmassi, hmassj, &
            xref, yref, zref, x, y, z)
       niter3 = 0
       call fsshakph_kernel3_d0(nsh1+1, nsh2+nsh1, bshkgp, shkapr, constr, ammi, 2*nsh1-4, &
            a12tol, a13tol, xref, yref, zref, x, y, z, mxiter, niter3)
       niter4 = 0
       call fsshakph_kernel4_d0(nsh1+nsh2+1, nsh1+nsh2+nsh3, bshkgp, shkapr, &
            constr, ammi, 2*nsh1+5*nsh2-6, a12tol, a13tol, a14tol, &
            xref, yref, zref, x, y, z, d2tol2, mxiter, niter4)
#if KEY_DOMDEC==1
    endif  
#endif

    if(niter3 >= mxiter) then
       write(outu,'(1x,a,1x,/,a,1x,i6,1x,a,/)') &
            '***** Error in fsshakph_kernel3 *****', &
            'Coordinate resetting was not accomplished within', niter3 , &
            'steps BY FAST METHOD.'
       call diewrn(-2)
       return
    end if

    if(niter4 >= mxiter) then
       write(outu,'(1x,a,1x,a,/,i6,1x,a,/)') &
            '***** Error in fsshakph_kernel4 *****', &
            'Coordinate resetting was not accomplished within', niter4 , &
            'steps BY FAST METHOD.'
       call diewrn(-2)
       return
    end if

    niter = max(niter3, niter4)

    return
  end subroutine fsshakph

  subroutine fsshakpg(x,y,z,xref,yref,zref,nconst, &
       natom,niter,nother, &
       hmassi,hmassj,d2tol2, &
       ibegin,iend &
       ,shkapr,nconp,idgf2,constr,shktol,mxiter)
    !-----------------------------------------------------------------------
    !     Parallel and partial vector routine for general shake
    !     constraints.
    !
    !     Authors: Doug Tobias and John Mertz
    !
    use stream
    use memory

    integer nconp,idgf2(*),shkapr(2,MAXSHK),mxiter,nconst
    real(chm_real) constr(nconp),shktol
    real(chm_real) x(*),y(*),z(*),xref(*),yref(*),zref(*)
    real(chm_real) hmassi(*),hmassj(*),d2tol2(*)
    real(chm_real) xpij,ypij,zpij,ratio,diff,corri,corrj
    integer natom,niter,nother
    integer iconst,i,j,ibegin,iend
    logical adjust
    logical done
    real(chm_real),allocatable,dimension(:),target :: space_xij
    real(chm_real),pointer,dimension(:) :: xij,yij,zij

    call chmalloc('fsshake.src','fsshakpg','space_xij',3*nconst,crl=space_xij)
    xij=>space_xij(         1:  nconst)
    yij=>space_xij(  nconst+1:2*nconst)
    zij=>space_xij(2*nconst+1:3*nconst)

    !
    niter = 0
    !
    do iconst = 1,nother
       i = shkapr(1,iconst)
       j = shkapr(2,iconst)
       xij(iconst) = xref(i) - xref(j)
       yij(iconst) = yref(i) - yref(j)
       zij(iconst) = zref(i) - zref(j)
    end do
    !
    done = .false.
    do while( .not. done )
       done = .true.
       do iconst = ibegin,iend
          i = shkapr(1,iconst)
          j = shkapr(2,iconst)
          xpij = x(i) - x(j)
          ypij = y(i) - y(j)
          zpij = z(i) - z(j)
          diff = constr(iconst) - xpij*xpij - ypij*ypij - zpij*zpij
          adjust = (abs(diff) > d2tol2(iconst))
          if(adjust) then
             ratio = diff/(xij(iconst)*xpij + yij(iconst)*ypij &
                  + zij(iconst)*zpij)
             corri =  hmassi(iconst)*ratio
             corrj = -hmassj(iconst)*ratio
             x(i) = x(i) + xij(iconst)*corri
             y(i) = y(i) + yij(iconst)*corri
             z(i) = z(i) + zij(iconst)*corri
             x(j) = x(j) + xij(iconst)*corrj
             y(j) = y(j) + yij(iconst)*corrj
             z(j) = z(j) + zij(iconst)*corrj
          endif
       end do
       !
       niter = niter + 1
       do iconst = 1,nother
          if(adjust) then
             if(niter > mxiter) then
                done = .true.
             else
                done = .false.
             end if
          end if
       end do
    end do
    if(niter > mxiter) then
       write(outu,'(1x,a,1x,a,/,i6,1x,a,/)') &
            '***** Error in SHAKPG ***** Coordinate resetting was not', &
            'accomplished in',mxiter,'iterations.'
       call diewrn(-2)
       call chmdealloc('fsshake.src','fsshakpg','space_xij',3*nconst,crl=space_xij)
       return
    endif
    !      niter = max(niter,niter)
    !
    !
    call chmdealloc('fsshake.src','fsshakpg','space_xij',3*nconst,crl=space_xij)
    return
  end subroutine fsshakpg

  subroutine fstshake(nconst,niter,x,y,z,xref,yref,zref,natom &
       ,shkapr,nconp,idgf2,constr,shktol,mxiter,amass &
       )
    !-----------------------------------------------------------------------
    !     Call appropriate scalar fast shake routines.
    !
    !
    use exfunc
    use machdep
#if KEY_DOMDEC==1
    use stream,only:outu,prnlev
    use domdec_common,only:q_domdec
#endif
#if KEY_DOMDEC_GPU==1
    use domdec_common,only:q_gpu
    use domdec_util_gpu_mod,only:range_start, range_stop
#endif

    integer nconp,idgf2(*),shkapr(2,MAXSHK),mxiter
    real(chm_real) constr(nconp),shktol,amass(*)
    !
    real(chm_real) x(*),y(*),z(*),xref(*),yref(*),zref(*)
    integer nconst,natom
    integer xijcp,yijcp,zijcp,adjust,niter,donep
    !

#if KEY_DOMDEC==1
    if (q_domdec) then
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_start('Shake solvent')  
#endif
       call settle_water(xref, yref, zref, x, y, z, numwater, nstwat, shkapr)
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()  
       if (q_gpu) call range_start('Shake solute')  
#endif
       if(lhonly) then
          call fsshakph(x,y,z,xref,yref,zref,niter,nconst, &
               hmassi,hmassj,d2tol2,nother &
               ,shkapr,nconp,constr,shktol,mxiter &
               ,nsh1,nsh2,nsh3,nsh4,bshkgp,ammi)
       endif
#if KEY_DOMDEC_GPU==1
       if (q_gpu) call range_stop()  
#endif
    else
#endif

       if(nwatgr > 0) then
          call fsshakwat(niter,x,y,z,xref,yref,zref,nwatgr,nother, &
               a1,a2,a3,tolij,toljk,tolki, &
               r12sq,r23sq,r31sq,ca11,ca12,ca13,ca21,ca22, &
               ca23,ca31,ca32,ca33,cb11,cb12,cb13,cb21,cb22,cb23,cb31,cb32, &
               cb33, &
               om1,om2,om3,numwater,nstwat, &
               shkapr,nconp,idgf2,constr,shktol,mxiter)
       endif
       if(nother > 0) then
          if(lhonly) then
             call fsshakph(x,y,z,xref,yref,zref,niter,nconst, &
                  hmassi,hmassj,d2tol2,nother &
                  ,shkapr,nconp,constr,shktol,mxiter &
                  ,nsh1,nsh2,nsh3,nsh4,bshkgp,ammi)
          else
             call fsshakpg(x,y,z,xref,yref,zref,nconst, &
                  natom,niter,nother, &
                  hmassi,hmassj,d2tol2, &
                  1,nother &
                  ,shkapr,nconp,idgf2,constr,shktol,mxiter)
          endif
       endif

#if KEY_DOMDEC==1
    endif
#endif

    return
  end subroutine fstshake
#endif /* (fsshake)*/

#endif /* (fsshk_main)*/
  !
end module fstshk


subroutine null_cshk
  return
end subroutine null_cshk

