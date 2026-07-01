module merck_io


  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*), private,parameter :: file_name="merckio.src"


  ! I-Jen Chen was here.
  ! Local variables
  
  ! clb3 replaced 20000 by allocatable/reallocatable arrays
  
  !real(chm_real)  tmpX(20000),tmpY(20000),tmpZ(20000), &
  !     t2X(20000),t2Y(20000),t2Z(20000),nonarX(20000), &
  !     nonarY(20000),nonarZ(20000)
  
  real(chm_real), private, allocatable, dimension(:) ::  tmpX,tmpY,tmpZ, &
       t2X,t2Y,t2Z,nonarX, &
       nonarY,nonarZ
  
  !integer tmpAtNum(20000),nonarAtNum(20000),t2AtNum(20000), &
  !     tmpIB(20000),tmpJB(20000),tmploc(20000),t2loc(20000), &
  !     tempib(20000),tempjb(20000),found(20000), &
  !     IB1(20000),JB1(20000),fuatom(20000), &
  !     arib(20000),arjb(20000),loc(20000),nonarloc(20000), &
  !     id2(20000),atm_friend(20000),nextfuse(20000),next2(20000), &
  !     tmpvelec(20000),t2velec(20000),velec(20000),fc(20000), &
  !     selec(20000),nonarvelec(20000),lonepair(20000)
  
  integer, private, allocatable, dimension(:) :: tmpAtNum,nonarAtNum,t2AtNum, &
       tmpIB,tmpJB,tmploc,t2loc, &
       tempib,tempjb,found, &
       IB1,JB1,fuatom, &
       arib,arjb,loc,nonarloc, &
       id2,atm_friend,nextfuse,next2, &
       tmpvelec,t2velec,velec,fc, &
       selec,nonarvelec,lonepair

  integer, private :: nallocated

  !character(len=2) tmpbond(20000),arbond(20000),tmpbo(20000), &
  !     bond(20000),aratom(20000),atname2(20000)
  
  !     character(len=4) tmpAtNames(20000),nonarAtNames(20000),t2AtNames(20000)
  
  character(len=2), private, allocatable, dimension(:) :: tmpbond,arbond,tmpbo, &
       bond,aratom,atname2
  character(len=4), private, allocatable, dimension(:) :: tmpAtNames,nonarAtNames,t2AtNames
  
contains

#if KEY_MMFF==1 /*merckio*/

  subroutine alloc_merckio(nallocated)
    use memory
    character(len=*),parameter :: routine_name="alloc_merckio"
    integer :: nallocated
    
    write(*,*)'**************************CALLING ALLOC, nallocated=', nallocated

    ! chm_real
    call chmalloc(file_name,routine_name,'tmpX',nallocated,crl=tmpX)
    call chmalloc(file_name,routine_name,'tmpY',nallocated,crl=tmpY)
    call chmalloc(file_name,routine_name,'tmpZ',nallocated,crl=tmpZ)
    call chmalloc(file_name,routine_name,'t2X',nallocated,crl=t2X)
    call chmalloc(file_name,routine_name,'t2Y',nallocated,crl=t2Y)
    call chmalloc(file_name,routine_name,'t2Z',nallocated,crl=t2Z)
    call chmalloc(file_name,routine_name,'nonarX',nallocated,crl=nonarX)
    call chmalloc(file_name,routine_name,'nonarY',nallocated,crl=nonarY)
    call chmalloc(file_name,routine_name,'nonarZ',nallocated,crl=nonarZ)
    
    ! integers
    call chmalloc(file_name,routine_name,'tmpAtNum',nallocated,intg=tmpAtNum)
    call chmalloc(file_name,routine_name,'nonarAtNum',nallocated,intg=nonarAtNum)
    call chmalloc(file_name,routine_name,'t2AtNum',nallocated,intg=t2AtNum)
    call chmalloc(file_name,routine_name,'tmpIB',nallocated,intg=tmpIB)
    call chmalloc(file_name,routine_name,'tmpJB',nallocated,intg=tmpJB)
    call chmalloc(file_name,routine_name,'tmploc',nallocated,intg=tmploc)
    call chmalloc(file_name,routine_name,'t2loc',nallocated,intg=t2loc)
    call chmalloc(file_name,routine_name,'tempib',nallocated,intg=tempib)
    call chmalloc(file_name,routine_name,'tempjb',nallocated,intg=tempjb)
    call chmalloc(file_name,routine_name,'found',nallocated,intg=found)
    call chmalloc(file_name,routine_name,'IB1',nallocated,intg=IB1)
    call chmalloc(file_name,routine_name,'JB1',nallocated,intg=JB1)
    call chmalloc(file_name,routine_name,'fuatom',nallocated,intg=fuatom)
    call chmalloc(file_name,routine_name,'arib',nallocated,intg=arib)
    call chmalloc(file_name,routine_name,'arjb',nallocated,intg=arjb)
    call chmalloc(file_name,routine_name,'loc',nallocated,intg=loc)
    call chmalloc(file_name,routine_name,'nonarloc',nallocated,intg=nonarloc)
    call chmalloc(file_name,routine_name,'id2',nallocated,intg=id2)
    call chmalloc(file_name,routine_name,'atm_friend',nallocated,intg=atm_friend)
    call chmalloc(file_name,routine_name,'nextfuse',nallocated,intg=nextfuse)
    call chmalloc(file_name,routine_name,'next2',nallocated,intg=next2)
    call chmalloc(file_name,routine_name,'tmpvelec',nallocated,intg=tmpvelec)
    call chmalloc(file_name,routine_name,'t2velec',nallocated,intg=t2velec)
    call chmalloc(file_name,routine_name,'velec',nallocated,intg=velec)
    call chmalloc(file_name,routine_name,'fc',nallocated,intg=fc)
    call chmalloc(file_name,routine_name,'selec',nallocated,intg=selec)
    call chmalloc(file_name,routine_name,'nonarvelec',nallocated,intg=nonarvelec)
    call chmalloc(file_name,routine_name,'lonepair',nallocated,intg=lonepair)
    
    ! ch2
    call chmalloc(file_name,routine_name,'tmpbond',nallocated,ch2=tmpbond)
    call chmalloc(file_name,routine_name,'arbond',nallocated,ch2=arbond)
    call chmalloc(file_name,routine_name,'tmpbo',nallocated,ch2=tmpbo)
    call chmalloc(file_name,routine_name,'bond',nallocated,ch2=bond)
    call chmalloc(file_name,routine_name,'aratom',nallocated,ch2=aratom)
    call chmalloc(file_name,routine_name,'atname2',nallocated,ch2=atname2)
    !
    ! ch4
    call chmalloc(file_name,routine_name,'tmpAtNames',nallocated,ch4=tmpAtNames)
    call chmalloc(file_name,routine_name,'nonarAtNames',nallocated,ch4=nonarAtNames)
    call chmalloc(file_name,routine_name,'t2AtNames',nallocated,ch4=t2AtNames)
    
  end subroutine alloc_merckio

  subroutine realloc_merckio(nallocated)
    use memory
    character(len=*), parameter :: routine_name="realloc_merckio"
    integer :: nallocated
    
    write(*,*)'**************************CALLING REALLOC, nallocated=', nallocated
    
    ! chm_real
    call chmrealloc(file_name,routine_name,'tmpX',nallocated,crl=tmpX)
    call chmrealloc(file_name,routine_name,'tmpY',nallocated,crl=tmpY)
    call chmrealloc(file_name,routine_name,'tmpZ',nallocated,crl=tmpZ)
    call chmrealloc(file_name,routine_name,'t2X',nallocated,crl=t2X)
    call chmrealloc(file_name,routine_name,'t2Y',nallocated,crl=t2Y)
    call chmrealloc(file_name,routine_name,'t2Z',nallocated,crl=t2Z)
    call chmrealloc(file_name,routine_name,'nonarX',nallocated,crl=nonarX)
    call chmrealloc(file_name,routine_name,'nonarY',nallocated,crl=nonarY)
    call chmrealloc(file_name,routine_name,'nonarZ',nallocated,crl=nonarZ)
    
    ! integers
    call chmrealloc(file_name,routine_name,'tmpAtNum',nallocated,intg=tmpAtNum)
    call chmrealloc(file_name,routine_name,'nonarAtNum',nallocated,intg=nonarAtNum)
    call chmrealloc(file_name,routine_name,'t2AtNum',nallocated,intg=t2AtNum)
    call chmrealloc(file_name,routine_name,'tmpIB',nallocated,intg=tmpIB)
    call chmrealloc(file_name,routine_name,'tmpJB',nallocated,intg=tmpJB)
    call chmrealloc(file_name,routine_name,'tmploc',nallocated,intg=tmploc)
    call chmrealloc(file_name,routine_name,'t2loc',nallocated,intg=t2loc)
    call chmrealloc(file_name,routine_name,'tempib',nallocated,intg=tempib)
    call chmrealloc(file_name,routine_name,'tempjb',nallocated,intg=tempjb)
    call chmrealloc(file_name,routine_name,'found',nallocated,intg=found)
    call chmrealloc(file_name,routine_name,'IB1',nallocated,intg=IB1)
    call chmrealloc(file_name,routine_name,'JB1',nallocated,intg=JB1)
    call chmrealloc(file_name,routine_name,'fuatom',nallocated,intg=fuatom)
    call chmrealloc(file_name,routine_name,'arib',nallocated,intg=arib)
    call chmrealloc(file_name,routine_name,'arjb',nallocated,intg=arjb)
    call chmrealloc(file_name,routine_name,'loc',nallocated,intg=loc)
    call chmrealloc(file_name,routine_name,'nonarloc',nallocated,intg=nonarloc)
    call chmrealloc(file_name,routine_name,'id2',nallocated,intg=id2)
    call chmrealloc(file_name,routine_name,'atm_friend',nallocated,intg=atm_friend)
    call chmrealloc(file_name,routine_name,'nextfuse',nallocated,intg=nextfuse)
    call chmrealloc(file_name,routine_name,'next2',nallocated,intg=next2)
    call chmrealloc(file_name,routine_name,'tmpvelec',nallocated,intg=tmpvelec)
    call chmrealloc(file_name,routine_name,'t2velec',nallocated,intg=t2velec)
    call chmrealloc(file_name,routine_name,'velec',nallocated,intg=velec)
    call chmrealloc(file_name,routine_name,'fc',nallocated,intg=fc)
    call chmrealloc(file_name,routine_name,'selec',nallocated,intg=selec)
    call chmrealloc(file_name,routine_name,'nonarvelec',nallocated,intg=nonarvelec)
    call chmrealloc(file_name,routine_name,'lonepair',nallocated,intg=lonepair)
    
    ! ch2
    call chmrealloc(file_name,routine_name,'tmpbond',nallocated,ch2=tmpbond)
    call chmrealloc(file_name,routine_name,'arbond',nallocated,ch2=arbond)
    call chmrealloc(file_name,routine_name,'tmpbo',nallocated,ch2=tmpbo)
    call chmrealloc(file_name,routine_name,'bond',nallocated,ch2=bond)
    call chmrealloc(file_name,routine_name,'aratom',nallocated,ch2=aratom)
    call chmrealloc(file_name,routine_name,'atname2',nallocated,ch2=atname2)
    
    call chmrealloc(file_name,routine_name,'tmpAtNames',nallocated,ch4=tmpAtNames)
    call chmrealloc(file_name,routine_name,'nonarAtNames',nallocated,ch4=nonarAtNames)
    call chmrealloc(file_name,routine_name,'t2AtNames',nallocated,ch4=t2AtNames)
    
  end subroutine realloc_merckio

  subroutine dealloc_merckio(nallocated)
    use memory
    character(len=*),parameter :: routine_name="dealloc_merckio"
    integer :: nallocated

     write(*,*)'**************************DEALLOCATING, nallocated=',nallocated
 
   ! chm_real
    call chmdealloc(file_name,routine_name,'tmpX',nallocated,crl=tmpX)
    call chmdealloc(file_name,routine_name,'tmpY',nallocated,crl=tmpY)
    call chmdealloc(file_name,routine_name,'tmpZ',nallocated,crl=tmpZ)
    call chmdealloc(file_name,routine_name,'t2X',nallocated,crl=t2X)
    call chmdealloc(file_name,routine_name,'t2Y',nallocated,crl=t2Y)
    call chmdealloc(file_name,routine_name,'t2Z',nallocated,crl=t2Z)
    call chmdealloc(file_name,routine_name,'nonarX',nallocated,crl=nonarX)
    call chmdealloc(file_name,routine_name,'nonarY',nallocated,crl=nonarY)
    call chmdealloc(file_name,routine_name,'nonarZ',nallocated,crl=nonarZ)
    
    ! integers
    call chmdealloc(file_name,routine_name,'tmpAtNum',nallocated,intg=tmpAtNum)
    call chmdealloc(file_name,routine_name,'nonarAtNum',nallocated,intg=nonarAtNum)
    call chmdealloc(file_name,routine_name,'t2AtNum',nallocated,intg=t2AtNum)
    call chmdealloc(file_name,routine_name,'tmpIB',nallocated,intg=tmpIB)
    call chmdealloc(file_name,routine_name,'tmpJB',nallocated,intg=tmpJB)
    call chmdealloc(file_name,routine_name,'tmploc',nallocated,intg=tmploc)
    call chmdealloc(file_name,routine_name,'t2loc',nallocated,intg=t2loc)
    call chmdealloc(file_name,routine_name,'tempib',nallocated,intg=tempib)
    call chmdealloc(file_name,routine_name,'tempjb',nallocated,intg=tempjb)
    call chmdealloc(file_name,routine_name,'found',nallocated,intg=found)
    call chmdealloc(file_name,routine_name,'IB1',nallocated,intg=IB1)
    call chmdealloc(file_name,routine_name,'JB1',nallocated,intg=JB1)
    call chmdealloc(file_name,routine_name,'fuatom',nallocated,intg=fuatom)
    call chmdealloc(file_name,routine_name,'arib',nallocated,intg=arib)
    call chmdealloc(file_name,routine_name,'arjb',nallocated,intg=arjb)
    call chmdealloc(file_name,routine_name,'loc',nallocated,intg=loc)
    call chmdealloc(file_name,routine_name,'nonarloc',nallocated,intg=nonarloc)
    call chmdealloc(file_name,routine_name,'id2',nallocated,intg=id2)
    call chmdealloc(file_name,routine_name,'atm_friend',nallocated,intg=atm_friend)
    call chmdealloc(file_name,routine_name,'nextfuse',nallocated,intg=nextfuse)
    call chmdealloc(file_name,routine_name,'next2',nallocated,intg=next2)
    call chmdealloc(file_name,routine_name,'tmpvelec',nallocated,intg=tmpvelec)
    call chmdealloc(file_name,routine_name,'t2velec',nallocated,intg=t2velec)
    call chmdealloc(file_name,routine_name,'velec',nallocated,intg=velec)
    call chmdealloc(file_name,routine_name,'fc',nallocated,intg=fc)
    call chmdealloc(file_name,routine_name,'selec',nallocated,intg=selec)
    call chmdealloc(file_name,routine_name,'nonarvelec',nallocated,intg=nonarvelec)
    call chmdealloc(file_name,routine_name,'lonepair',nallocated,intg=lonepair)
    
    ! ch2
    call chmdealloc(file_name,routine_name,'tmpbond',nallocated,ch2=tmpbond)
    call chmdealloc(file_name,routine_name,'arbond',nallocated,ch2=arbond)
    call chmdealloc(file_name,routine_name,'tmpbo',nallocated,ch2=tmpbo)
    call chmdealloc(file_name,routine_name,'bond',nallocated,ch2=bond)
    call chmdealloc(file_name,routine_name,'aratom',nallocated,ch2=aratom)
    call chmdealloc(file_name,routine_name,'atname2',nallocated,ch2=atname2)
    
    call chmdealloc(file_name,routine_name,'tmpAtNames',nallocated,ch4=tmpAtNames)
    call chmdealloc(file_name,routine_name,'nonarAtNames',nallocated,ch4=nonarAtNames)
    call chmdealloc(file_name,routine_name,'t2AtNames',nallocated,ch4=t2AtNames)

    nallocated = 0
 
  end subroutine dealloc_merckio
  
! ===========================================================
! SUBROUTINE MOLIN : READ THE MOL COORD FILE FROM THE DISK,
! AND FILL THE COMMON AREAS.
! ===========================================================
!23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
!

!
! I-Jen Chen was here.
!      SUBROUTINE MOLIN(imolf,SNAME,ERROR,
#ifdef KEY_RESIZE
SUBROUTINE MOLIN(imolf,SNAME,ERROR,KEY,CMPD)
#else
SUBROUTINE MOLIN(imolf,SNAME,ERROR,KEY,CMPD, &
     ! I-Jen left. 
     NATOM,X,Y,Z,ATNUM,AtNames,NICHG,  & ! ATOMS
     NBOND,IB,JB,BondType,            & ! BONDS
     IGPBS,NGRP,                            & ! GROUPS
     RESID,RES,IBASE,NRES,RESNAME,KRES,     & ! RESIDUES
     SEGID,NICTOT,NSEG)                    ! SEGMENTS
#endif
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  Jay Banks 27 Nov 95: changed XYZ arrays to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.  Likewise "END DO"s and "END IF"s.
  !
  use cnst_fcm, only: cnst_is_allocated, allocate_cnst
  use ctitla
  use stream
  use string
#ifdef KEY_RESIZE
  use mmffm
  use psf
  use coord
  use coordc
  use deriv
  use resize
#endif
  implicit none
  !
  integer AOFF, NAT
#ifndef KEY_RESIZE  
  integer NATOM
  real(chm_real)  X(*),Y(*),Z(*)
  integer  NBOND, IB(*),JB(*),BondType(*)
  integer ATNUM(*)
#endif  
  integer BOFF
  CHARACTER(len=*) SNAME
#ifndef KEY_RESIZE  
  character(len=*) AtNames(*)
  character(len=*) RESID(*)
  character(len=*) RES(*),SEGID(*)
  character(len=*) RESNAME(*),KRES(*)
#endif  
  integer ridlen      ! length or RESNAME (residue ID) string
#ifndef KEY_RESIZE    
  integer NICTOT(*)   ! segment pointers
  integer NSEG ! number of segments
  integer IBASE(*)    ! residue pointers
  integer NRES        ! number of residues
  integer IGPBS(*)    ! group   pointers
  integer NGRP        ! number of groups
  integer NICHG(*)    ! formal charge
#endif  
  integer NSEGM    ! formal charge
  !
  integer i, imolf, nb, nichgi, idummy, fcount
  CHARACTER(len=8) ATN, old_resname, old_segn, segn
  CHARACTER(len=NRES0) ANRES
  character(len=KRES0) AKRES
  LOGICAL ERROR,NTHERE,KTHERE
  CHARACTER(len=2) ElementSymbol
   external ElementSymbol
  !
  integer fubond,cmpdloc,fuse_num
  integer neighbor
  integer totalb
  integer tempi,tempj,tmpib2,numb,numb2,numb3,numb4,numb5,numb6
  integer count,arbn,pairs,last,futotal,loc1,loc2,nb2,abc,eof
  integer junk1,junk2,junk3,j,k,l,m,n,o,p,s,zero,id,apperance
  integer nextnum,diff,next2num,show

  character(len=2) tempab, junk4*6
  character(len=4) KEY
  character(len=20) CMPD
  logical no_cmpd,switch,fuse_yes
#ifdef KEY_RESIZE
  integer na0,nb0,ng0,nr0,ns0
#endif
 
  no_cmpd = .true.
  fuse_yes = .false.
  ! I-Jen Chen left

  ERROR=.FALSE.

#ifdef KEY_RESIZE
  na0=natom
  nb0=nbond
  ng0=ngrp
  nr0=nres
  ns0=nseg
#endif  
  ! I-Jen Chen was here.
!!!!!!!!!!!!!!!!!!READ MEERCK FILE!!!!!!!!!!!!!!!!!
  if(KEY.eq.'MERC')then

     ! I-Jen left.

     fcount=0
     do while (.true.) ! read concatenated files...
        !
        ! READ THE TITLE
        !   READ THE SECOND TITLE CARD
        READ(imolf,'(A)',END=4200) SCRTCH ! this should allow to read concatenated files
        if(SCRTCH(1:3).eq.'END') return   ! from input script
        TITLEB(1)=SCRTCH
        READ(imolf,'(A)',END=4000) TITLEB(2)
        NTITLB=2
        !
        old_resname='?XXX'
        if(NATOM.gt.0) old_resname=resname(NATOM)
        old_segn='?XXX'
        if(NSEG.gt.0) old_segn=SEGID(NSEG)
        !
        AOFF=NATOM
        BOFF=NBOND
        READ(imolf,*,END=4000) NATOM,NBOND
        NATOM=NATOM+AOFF
        NBOND=NBOND+BOFF
#ifdef KEY_RESIZE
        if (natom .gt. na0) then
           na0=natom+drsz0
           call resize_psf('psfres.F90','molin','NATOM',na0, .true.)
           call resize_coord('psfres.F90','molin',na0, .true.)
           call resize_coordc('psfres.F90','molin',na0, .true.)
           call resize_deriv('psfres.F90','molin',na0, .true.)
        endif
        if (nbond .gt. nb0) then
           nb0=nbond+drsz0
           call resize_psf('psfres.F90','molin','NBOND',nb0, .true.)
        endif
#endif
        !
        IF(NATOM.LT.1.OR.NATOM.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>',' ERROR IN ATOM COUNT,CHECK FILE')
           return
        endif
        IF(NBOND.LT.0.OR.NBOND.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>','ERROR IN BOND COUNT,CHECK FILE')
           RETURN
        endif

        !           (3(F10.4,X), I5, X,A2,X,I1,X,I5, X,3(A4),F8.4) < from 8 March 1993  Bruce Bush
        !2200 FORMAT(3(F10.4,1X),I5,1X,I2, I2,1X,I5,1X, 3A4,1X,F8.4) < from molout
        !                              ^   ^                ^

        !     NAT=AOFF
        !     do while (NAT.lt.NATOM)
        DO NAT=AOFF+1,NATOM ! READ THE X,Y,Z VALUES AND ATOM ATTRIBUTES, IF PRESENT
           !
           READ(imolf,'(a)',END=4000,ERR=3200) SCRTCH
           !        if(SCRTCH(1:1).ne.'!') then
           !           NAT=NAT+1
           !
           !           comments should not be allowed as long as NSEQI
           !           is ignored.... They might be allowed if NSEQI is used
           !           to code bond information...
           !
           read(SCRTCH( 1:10),'(f10.4)') X(NAT)   ! X - coordinate
           read(SCRTCH(12:21),'(f10.4)') Y(NAT)   ! Y - coordinate
           read(SCRTCH(23:32),'(f10.4)') Z(NAT)   ! Z - coordinate
           read(SCRTCH(34:38),'(i5)')    AtNum(NAT)   ! atomic number
           !           read(SCRTCH(40:41),'(i2)')    CMTYPE       ! ignored (atomic type)
           read(SCRTCH(43:43),'(i1)')    NICHGI       ! Charge Code
           !           read(SCRTCH(45:49),'(i5)')    NSEQI        ! ignored (sequential number)
           read(SCRTCH(51:54),'(a4)')    AtNames(NAT) ! atomic name
           read(SCRTCH(55:58),'(a4)')    resname(NAT) ! residue id (number)
           read(SCRTCH(59:62),'(a4)')    KRES(NAT)    ! residue type
           !           read(SCRTCH(63:70),'(f8.4)')  charge       ! ignored
           read(SCRTCH(77:80),'(a4)')    segn         ! segment id
           ! KRES - residue type (ALA, GLY etc...)
           ! RESNAME - residue ID
           !    Left-justify the residue ID  JLB 06 Oct 95
           if (resname(NAT) .ne. ' ') then
              ridlen = LEN(resname(NAT))
              do while (resname(NAT)(1:1).EQ.' ')
                 resname(NAT) = resname(NAT)(2:ridlen)//' '
              enddo
           endif
           NICHG(NAT)=NICHGI
           if(resname(NAT).ne.old_resname .or. segn.ne.old_segn) then
              !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
              old_resname=resname(NAT)
              nres=nres+1
#ifdef KEY_RESIZE
              if (nres .gt. nr0) then
                 nr0=nres+drsz0
                 call resize_psf('psfres.F90','molin','NRES',nr0, .true.)
              endif
#endif
              ibase(nres)=NAT-1
              resid(nres)=resname(NAT)
              res(nres)=kres(NAT)
              ngrp=ngrp+1            ! group = reside
#ifdef KEY_RESIZE
              if (ngrp .gt. ng0) then
                 ng0=ngrp+drsz0
                 call resize_psf('psfres.F90','molin','NGRP',ng0, .true.)
              endif
#endif
              igpbs(ngrp)=NAT-1
           endif
           if(segn.ne.old_segn) then
              old_segn=segn
              nsegm=nseg
              nseg=nseg+1
#ifdef KEY_RESIZE
              if (nseg .gt. ns0) then
                 ns0=nseg+drsz0
                 call resize_psf('psfres.F90','molin','NSEG',ns0, .true.)
              endif
#endif              
              segid(nseg)=segn
              IF (SEGID(NSEG) .EQ. ' ') then
                 CALL ENCODI(NSEG, SEGID(NSEG), 8, IDUMMY)
                 IF(NSEG.EQ.1) SEGID(NSEG)='MMFF'
              endif
              nictot(nseg)=nres-1 !????
              !...........Check for duplicate segment names.
              DO I = 1,NSEGM
                 IF(SEGID(I) .EQ. SEGID(NSEG)) THEN
                    NSEG = NSEGM
                    CALL WRNDIE(-2,'<molin>','DUPLICATE SEGMENT NAMES')
                    RETURN
                 ENDIF
              ENDDO ! DO I = 1,NSEGM
              NSEGM=NSEG
           endif ! if(segn.ne.old_segn) then
           !        endif ! if(SCRTCH(1:1).ne.'!') then
           !     ENDDO ! do while (NAT.lt.NATOM)
        ENDDO ! DO NAT=AOFF+1,NATOM
        nictot(nseg+1)=nres
        ibase(nres+1)=natom
        igpbs(ngrp+1)=natom
        !
        if((NBOND-BOFF).gt.0) READ(imolf,*,END=3800)  & ! READ THE BOND INFORMATION.
             (IB(nb),JB(nb),BondType(nb),nb=BOFF+1,NBOND)

        if(AOFF.gt.0) then
           do nb=BOFF+1,NBOND
              IB(nb)=IB(nb)+AOFF
              JB(nb)=JB(nb)+AOFF
           enddo
        endif
        !
        NTHERE=.TRUE. !  SEE IF RESIDUE NAMES ARE PRESENT IN THE MOL FILE
        ANRES(1:NRES0)=SNAME(1:NRES0)
        DO I=AOFF+1,NATOM
           IF(RESNAME(I).EQ.' ') NTHERE=.FALSE.
           IF(RESNAME(I).NE.' ') ANRES=RESNAME(I)
        ENDDO
        !
        AKRES(1:KRES0)=SNAME(1:KRES0) ! SEE IF RESIDUE TYPES ARE PRESENT IN THE MOL FILE
        KTHERE=.TRUE.
        DO I=AOFF+1,NATOM
           IF(KRES(I).EQ.' ') KTHERE=.FALSE.
           IF(KRES(I).NE.' ') AKRES=KRES(I)
        ENDDO
        IF(KTHERE) GOTO 3100
        !   ASSIGN RESIDUE AND ATOM TYPES
3100    CONTINUE
        !   NOW ASSIGN ATOM NAMES, ETC., IF NOT ALREADY PRESENT
        DO I=AOFF+1,NATOM
           IF(AtNames(I).eq.' ') then
              ATN(1:2)=ElementSymbol(AtNum(i))
              if(ATN(1:2).eq.' ') call wrndie(-5,'<molin>', &
                   'error in name assignment')
              IF(I.GE.100) THEN
                 WRITE(ATN(2:4),'(I3)') I
              ELSEIF(I.GE.10) THEN
                 WRITE(ATN(2:3),'(I2)') I
              ELSE
                 WRITE(ATN(2:2),'(I1)') I
              ENDIF
              AtNames(I)=ATN(1:4) ! PLACE CONSTRUCTED ATOM NAME IN AtNames(I)
           endif
           IF(.NOT.NTHERE) RESNAME(I)=ANRES
           IF(.NOT.KTHERE) KRES(I)=AKRES
        ENDDO
        !      write(6,*) ' molin: file read successfully'
        !
        fcount=fcount+1
        !
     enddo !  do while (.true.) ! read concatenated files...
     !
3200 ERROR=.TRUE.
     WRITE(SCRTCH,'(A,I4)') 'ERROR IN READING XYZ CARD FOR ATOM',NAT
     CALL WRNDIE(-5,'<molin>',SCRTCH(:44))
     return
3800 ERROR=.TRUE.
     CALL WRNDIE(-5,'<molin>','ERROR IN reading bonds, CHECK FILE')
     RETURN
4000 ERROR=.TRUE.
     CALL WRNDIE(1,'<molin>','Input file incomplete or unreadable')
     RETURN
4200 CONTINUE
     if(fcount.eq.0) then
        ERROR=.TRUE.
        CALL WRNDIE(1,'<molin>','--- End of MOL-file input ---')
     endif
#ifdef KEY_RESIZE
     if (natom .lt. na0) then
        call resize_psf('psfres.F90','molin','NATOM',natom, .true.)
        call resize_coord('psfres.F90','molin',natom, .true.)
        call resize_coordc('psfres.F90','molin',natom, .true.)
        call resize_deriv('psfres.F90','molin',natom, .true.)
     endif
     if (nbond .lt. nb0) &
        call resize_psf('psfres.F90','molin','NBOND',nbond, .true.)
     if (ngrp .lt. ng0) &
        call resize_psf('psfres.F90','molin','NGRP',ngrp, .true.)
     if (nres .lt. nr0) &
        call resize_psf('psfres.F90','molin','NRES',nres, .true.)
     if (nseg .lt. ns0) &
        call resize_psf('psfres.F90','molin','NSEG',nseg, .true.)
#endif
     if (cnst_is_allocated) call allocate_cnst(natom)
     RETURN

     !
     ! I-Jen Chen was here.
!!!!!!!!!!!!!!!!!!READ MOL2 FILE!!!!!!!!!!!!!!!!!

  else if(KEY.eq.'MOL2')then
     fcount = 0

     do while (.true.) ! read concatenated files...

        READ(imolf,'(A)',END=4201) SCRTCH ! this should allow to read concatenated files
        if(SCRTCH(1:3).eq.'END') then
!write(*,*)' Exiting 1'
           if(allocated(tmpX)) call dealloc_merckio(nallocated)
           return   ! from input script
        endif
        TITLEB(1)=SCRTCH
        READ(imolf,'(A)',END=4001) TITLEB(2)

        !
        ! I-Jen Chen was here.
        !
500     READ(imolf,'(A)',END=4002) SCRTCH
        if(SCRTCH(10:17).ne.'MOLECULE') goto 500
4002    continue

        READ(imolf,'(A)') SCRTCH
        ! I-Jen left.
        !
        NTITLB=2
        !
        old_resname='?XXX'
        if(NATOM.gt.0) old_resname=resname(NATOM)
        old_segn='?XXX'
        if(NSEG.gt.0) old_segn=SEGID(NSEG)
        !
        AOFF=NATOM
!write(*,*)' Before NATOM=', natom,' nallocated=',nallocated
        BOFF=NBOND
        READ(imolf,*,END=4001) NATOM,NBOND,junk1,junk2,junk3
!write(*,*)' After NATOM=', natom,' nallocated=',nallocated

        NATOM=NATOM+AOFF
        NBOND=NBOND+BOFF
#ifdef KEY_RESIZE
        if (natom .gt. na0) then
           na0=natom+drsz0
           call resize_psf('psfres.F90','molin','NATOM',na0, .true.)
           call resize_coord('psfres.F90','molin',na0, .true.)
           call resize_coordc('psfres.F90','molin',na0, .true.)
           call resize_deriv('psfres.F90','molin',na0, .true.)
        endif
        if (nbond .gt. nb0) then
           nb0=nbond+drsz0
           call resize_psf('psfres.F90','molin','NBOND',nb0, .true.)
        endif
#endif
        if (cnst_is_allocated) call allocate_cnst(natom)
        ! Allocate temporary space
        if(.not. allocated(tmpX)) then
           nallocated = max(natom,nbond)
           call alloc_merckio(nallocated)

        ! reallocate with additional size
        elseif(allocated(tmpX) .and. nallocated < max(natom,nbond)) then 
           nallocated = max(natom,nbond)
           call realloc_merckio(nallocated)
        endif
!
        IF(NATOM.LT.1.OR.NATOM.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>',' ERROR IN ATOM COUNT,CHECK FILE')
           return
        endif

        IF(NBOND.LT.0.OR.NBOND.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>','ERROR IN BOND COUNT,CHECK FILE')
           RETURN
        endif
        !


        !
        ! I-Jen Chen was here.
        !
501     READ(imolf,'(A)',END=4003) SCRTCH
        if(SCRTCH(10:13).ne.'ATOM') goto 501
4003    continue


        ! ==== START COOR READ ====

        numb = 0
        numb2 = 0

        DO NAT=AOFF+1,NATOM ! READ THE X,Y,Z VALUES AND ATOM ATTRIBUTES, IF PRESENT

           READ(imolf,'(a)',END=4001,ERR=3201) SCRTCH
           read(SCRTCH(9:12),'(a4)')    tmpAtNames(NAT) ! atomic name
           read(SCRTCH(17:26),'(f10.4)') tmpX(NAT)   ! X - coordinate
           read(SCRTCH(27:36),'(f10.4)') tmpY(NAT)   ! Y - coordinate
           read(SCRTCH(37:46),'(f10.4)') tmpZ(NAT)   ! Z - coordinate

           ! ==== ATOMIC NUMBER CONVERSION ====

           if(SCRTCH(48:49) .eq. 'Al') tmpAtNum(NAT) = 13
           if(SCRTCH(48:49) .eq. 'As') tmpAtNum(NAT) = 33
           if(SCRTCH(48:49) .eq. 'Ag') tmpAtNum(NAT) = 47
           if(SCRTCH(48:49) .eq. 'At') tmpAtNum(NAT) = 85
           if(SCRTCH(48:49) .eq. 'Au') tmpAtNum(NAT) = 79

           if(SCRTCH(48:49) .eq. 'B ') tmpAtNum(NAT) =  5
           if(SCRTCH(48:49) .eq. 'Be') tmpAtNum(NAT) =  4
           if(SCRTCH(48:49) .eq. 'Ba') tmpAtNum(NAT) = 56
           if(SCRTCH(48:49) .eq. 'Br')then
              tmpAtNum(NAT) = 35
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Bi') tmpAtNum(NAT) = 83

           if(SCRTCH(48:49) .eq. 'C.')then
              tmpAtNum(NAT) =  6
              tmpvelec(NAT) =  4
           endif
           if(SCRTCH(48:49) .eq. 'Cl')then
              tmpAtNum(NAT) = 17
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Ca') tmpAtNum(NAT) = 20
           if(SCRTCH(48:49) .eq. 'Cr') tmpAtNum(NAT) = 24
           if(SCRTCH(48:49) .eq. 'Cu') tmpAtNum(NAT) = 29
           if(SCRTCH(48:49) .eq. 'Cd') tmpAtNum(NAT) = 48
           if(SCRTCH(48:49) .eq. 'Cs') tmpAtNum(NAT) = 55

           if(SCRTCH(48:49) .eq. 'Du') tmpAtNum(NAT) = 0

           if(SCRTCH(48:49) .eq. 'F ' .or. &
                SCRTCH(48:49) .eq. 'F.')then
              tmpAtNum(NAT) =  9
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Fe') tmpAtNum(NAT) = 26

           if(SCRTCH(48:49) .eq. 'Ga') tmpAtNum(NAT) = 31
           if(SCRTCH(48:49) .eq. 'Ge') tmpAtNum(NAT) = 32

           if(SCRTCH(48:49) .eq. 'H ')then
              tmpAtNum(NAT) =  1
              tmpvelec(NAT) =  1
           endif

           if(SCRTCH(48:49) .eq. 'Hg') tmpAtNum(NAT) = 80

           if(SCRTCH(48:49) .eq. 'I ')then
              tmpAtNum(NAT) = 53
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'In') tmpAtNum(NAT) = 49

           if(SCRTCH(48:49) .eq. 'K ') tmpAtNum(NAT) = 19

           if(SCRTCH(48:49) .eq. 'Li') tmpAtNum(NAT) =  3
           if(SCRTCH(48:49) .eq. 'Lu') tmpAtNum(NAT) = 71

           if(SCRTCH(48:49) .eq. 'Mg') tmpAtNum(NAT) = 12
           if(SCRTCH(48:49) .eq. 'Mn') tmpAtNum(NAT) = 25
           if(SCRTCH(48:49) .eq. 'Mo') tmpAtNum(NAT) = 42

           if(SCRTCH(48:49) .eq. 'N.'.or. &
                SCRTCH(48:49) .eq. 'N ')then
              tmpAtNum(NAT) =  7
              tmpvelec(NAT) =  5
           endif
           if(SCRTCH(48:49) .eq. 'Ne') tmpAtNum(NAT) = 10
           if(SCRTCH(48:49) .eq. 'Na') tmpAtNum(NAT) = 11
           if(SCRTCH(48:49) .eq. 'Ni') tmpAtNum(NAT) = 28
           if(SCRTCH(48:49) .eq. 'Nb') tmpAtNum(NAT) = 41

           if(SCRTCH(48:49) .eq. 'O.'.or. &
                SCRTCH(48:49) .eq. 'O ')then
              tmpAtNum(NAT) =  8
              tmpvelec(NAT) =  6
           endif

           if(SCRTCH(48:49) .eq. 'P '.or. &
                SCRTCH(48:49) .eq. 'P.') tmpAtNum(NAT) = 15
           if(SCRTCH(48:49) .eq. 'Pd') tmpAtNum(NAT) = 46
           if(SCRTCH(48:49) .eq. 'Pt') tmpAtNum(NAT) = 78
           if(SCRTCH(48:49) .eq. 'Pb') tmpAtNum(NAT) = 82

           if(SCRTCH(48:49) .eq. 'Rb') tmpAtNum(NAT) = 37
           if(SCRTCH(48:49) .eq. 'Re') tmpAtNum(NAT) = 75
           if(SCRTCH(48:49) .eq. 'Ru') tmpAtNum(NAT) = 44
           if(SCRTCH(48:49) .eq. 'Rh') tmpAtNum(NAT) = 45

           if(SCRTCH(48:49) .eq. 'S.'.or. &
                SCRTCH(48:49) .eq. 'S ')then
              tmpAtNum(NAT) = 16
              tmpvelec(NAT) =  6
           endif

           if(SCRTCH(48:49) .eq. 'Si') tmpAtNum(NAT) = 14
           if(SCRTCH(48:49) .eq. 'Sr') tmpAtNum(NAT) = 38
           if(SCRTCH(48:49) .eq. 'Sc') tmpAtNum(NAT) = 21
           if(SCRTCH(48:49) .eq. 'Sn') tmpAtNum(NAT) = 50
           if(SCRTCH(48:49) .eq. 'Sb') tmpAtNum(NAT) = 51
           if(SCRTCH(48:49) .eq. 'Se') tmpAtNum(NAT) = 34

           if(SCRTCH(48:49) .eq. 'Ti') tmpAtNum(NAT) = 22
           if(SCRTCH(48:49) .eq. 'Zn') tmpAtNum(NAT) = 30

           tmploc(NAT) = NAT

           if(SCRTCH(50:51) .eq. 'ar')then
              numb = numb + 1
              aratom(numb) = SCRTCH(9:12)
              t2loc(numb) = numb
              t2AtNames(numb) = tmpAtNames(NAT)
              t2X(numb) = tmpX(NAT)
              t2Y(numb) = tmpY(NAT)
              t2Z(numb) = tmpZ(NAT)
              t2AtNum(numb) = tmpAtNum(NAT)
              t2velec(numb) = tmpvelec(NAT)
           else
              if(SCRTCH(50:51) .ne. 'ar')then
                 numb2 = numb2 + 1

                 nonarloc(numb2) = numb2

                 nonarAtNames(numb2) = tmpAtNames(NAT)

                 nonarX(numb2) = tmpX(NAT)

                 nonarY(numb2) = tmpY(NAT)

                 nonarZ(numb2) = tmpZ(NAT)

                 nonarAtNum(numb2) = tmpAtNum(NAT)

                 nonarvelec(numb2) = tmpvelec(NAT)
              endif
           endif

           !
           ! NICHGI values will be assigned after reading the BOND info.
           !
           ! NICHGI = (valence e) -1/2(shared e) - (lone pair e)
           !
           ! I-Jen Chen  
           !

           read(SCRTCH(55:58),'(a4)')    resname(NAT) ! residue id (number)
           read(SCRTCH(60:63),'(a4)')    KRES(NAT)    ! residue type
           read(SCRTCH(67:70),'(a4)')    segn         ! segment id

           if (resname(NAT) .ne. ' ') then
              ridlen = LEN(resname(NAT))
              do while (resname(NAT)(1:1).EQ.' ')
                 resname(NAT) = resname(NAT)(2:ridlen)//' '
              enddo
           endif

           if(resname(NAT).ne.old_resname.or.segn.ne.old_segn)then
              !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
              old_resname=resname(NAT)
              nres=nres+1
#ifdef KEY_RESIZE
              if (nres .gt. nr0) then
                 nr0=nres+drsz0
                 call resize_psf('psfres.F90','molin','NRES',nr0, .true.)
              endif
#endif
              ibase(nres)=NAT-1
              resid(nres)=resname(NAT)
              res(nres)=kres(NAT)
              ngrp=ngrp+1            ! group = reside
#ifdef KEY_RESIZE
              if (ngrp .gt. ng0) then
                 ng0=ngrp+drsz0
                 call resize_psf('psfres.F90','molin','NGRP',ng0, .true.)
              endif
#endif
              igpbs(ngrp)=NAT-1
           endif

           if(segn.ne.old_segn) then
              old_segn=segn
              nsegm=nseg
              nseg=nseg+1
#ifdef KEY_RESIZE
              if (nseg .gt. ns0) then
                 ns0=nseg+drsz0
                 call resize_psf('psfres.F90','molin','NSEG',ns0, .true.)
              endif
#endif              
              segid(nseg)=segn
              IF (SEGID(NSEG) .EQ. ' ') then
                 CALL ENCODI(NSEG, SEGID(NSEG), 8, IDUMMY)
                 IF(NSEG.EQ.1) SEGID(NSEG)='MMFF'
              endif
              nictot(nseg)=nres-1 !????
              !...........Check for duplicate segment names.
              DO I = 1,NSEGM
                 IF(SEGID(I) .EQ. SEGID(NSEG)) THEN
                    NSEG = NSEGM
                    CALL WRNDIE(-2,'<molin>','DUPLICATE SEGMENT NAMES')
                    RETURN
                 ENDIF
              ENDDO ! DO I = 1,NSEGM
              NSEGM=NSEG
           endif ! if(segn.ne.old_segn) then

        ENDDO ! DO NAT=AOFF+1,NATOM

        ! ==== END OF COOR READ ====

        !
        ! ASSIGN ATNAME, XYZ, AND ATNUM OF AROMATIC ATOMS INTO THE
        ! FINAL ARRAY.
        ! 

        !           numb3 = 0
        do loc1=1, numb
           !              numb3 = numb3 + 1
           loc(loc1) = loc1
           AtNames(loc1) = t2AtNames(loc1)
           X(loc1) = t2X(loc1)
           Y(loc1) = t2Y(loc1)
           Z(loc1) = t2Z(loc1)
           AtNum(loc1) = t2AtNum(loc1)
           velec(loc1) = t2velec(loc1)
        enddo

        !
        ! ASSIGN ATNAME, XYZ, AND ATNUM OF NONAROMATIC ATOMS INTO THE
        ! FINAL ARRAY.
        !

        do loc2 = 1, numb2

           loc(loc2+numb) = loc2 + numb

           AtNames(loc2+numb) = nonarAtNames(loc2)

           X(loc2+numb) = nonarX(loc2)

           Y(loc2+numb) = nonarY(loc2)

           Z(loc2+numb) = nonarZ(loc2)

           AtNum(loc2+numb) = nonarAtNum(loc2)

           velec(loc2+numb) = nonarvelec(loc2)

        enddo

        nictot(nseg+1)=nres
        ibase(nres+1)=natom
        igpbs(ngrp+1)=natom
        !

        ! Skip the line that separates the coor and the bond info.

        READ(imolf,'(A)',END=4001)

        ! ==== START BOND INFO READ ====

        do nb = BOFF+1,NBOND
           READ(imolf,100,END=3801)tmpIB(nb),tmpJB(nb),tmpbond(nb)
        enddo

100     format(6x,i5,i5,1x,a2)

        ! I-Jen Chen 07/28/99
        ! CHANGE THE ATOM PAIRS IN THE BOND LIST IN ACCORD WITH THE NEW ATOM
        ! LIST.
        !
        do nb = 1,NBOND
           do nat = 1,NATOM
              if(tmpAtNames(tmpIB(nb)).eq.AtNames(nat))IB1(nb)=nat
              if(tmpAtNames(tmpJB(nb)).eq.AtNames(nat))JB1(nb)=nat
              bond(nb) = tmpbond(nb)
           enddo
        enddo

        ! ==== BOND TYPE REASSIGNMENT ====
        !
        arbn = 0
        do nb = BOFF+1,NBOND
           if(bond(nb)(1:2) .eq. 'ar')then
              arbn = arbn + 1
              if(IB1(nb) .gt. JB1(nb))then
                 arib(arbn) = JB1(nb)
                 arjb(arbn) = IB1(nb)
              else
                 arib(arbn) = IB1(nb)
                 arjb(arbn) = JB1(nb)
              endif
              arbond(arbn) = bond(nb)
           endif
        enddo

        pairs = arbn - 1
1019    if(pairs .gt. 0)then
           last = 1
           do i= 1, pairs
              if(arib(i) .ge. arib(i+1))then

                 tempj = arjb(i)
                 arjb(i) = arjb(i+1)
                 arjb(i+1) = tempj

                 tempi = arib(i)
                 arib(i) = arib(i+1)
                 arib(i+1) = tempi

                 tempab = arbond(i)
                 arbond(i) = arbond(i+1)
                 arbond(i+1) = tempab

                 last = i

              endif
           enddo
           pairs = last -1
           goto 1019

        endif

        pairs = arbn - 1
1021    if(pairs .gt. 0)then
           last = 1
           do i= 1, pairs
              if(arib(i).eq.arib(i+1).and.arjb(i).gt.arjb(i+1))then

                 tempj = arjb(i)
                 arjb(i) = arjb(i+1)
                 arjb(i+1) = tempj

                 tempi = arib(i)
                 arib(i) = arib(i+1)
                 arib(i+1) = tempi

                 tempab = arbond(i)
                 arbond(i) = arbond(i+1)
                 arbond(i+1) = tempab

                 last = i

              endif
           enddo
           pairs = last -1
           goto 1021
        endif

        k = 1
        do i = 1, arbn
           if(arib(i).ne.9999)then
              tempib(k) = arib(i)
              tempjb(k) = arjb(i)
              do j = i+1, arbn
                 if((arib(j).ne.9999).and.((arjb(i).eq.arjb(j)).or. &
                      (arjb(i).eq.arib(j))))then
                    found(k) = j
                    k = k + 1
                    tempib(k) = arib(j)
                    tempjb(k) = arjb(j)
                    arib(j) = 9999
                    arjb(j) = 9999
                 endif
              enddo
              k = k + 1
           endif
        enddo

        ! Count the number of neighbors each aromatic atom has

        do i = 1, numb
           neighbor = 0
           do j = 1, arbn
              if(atnames(i).eq.atnames(tempib(j)).or.atnames(i).eq. &
                   atnames(tempjb(j))) neighbor = neighbor + 1
           enddo
           atm_friend(i) = neighbor
        enddo

        do i = 1, arbn
           if((atm_friend(tempib(i))-atm_friend(tempjb(i))).eq.-1) &
                then
              tempi = tempib(i)
              tempib(i) = tempjb(i)
              tempjb(i) = tempi

           endif
        enddo

        l = 0
        do j = 1, arbn
           if((atm_friend(tempib(j))-atm_friend(tempjb(j))).eq.1) &
                then
              l = l + 1
              arib(l) = tempib(j)
              arjb(l) = tempjb(j)
              tempib(j) = 999
              tempjb(j) = 999
              do k = 1, arbn
                 if(tempib(k).ne.999.and.arib(j).eq.tempib(k))then
                    l = l + 1
                    arib(l) = tempib(k)
                    arjb(l) = tempjb(k)
                    tempib(k) = 999
                    tempjb(k) = 999
                 endif
              enddo
           endif
        enddo

        n = l
        do m = 1, arbn
           if(tempib(m).ne.999)then
              n = n + 1
              arib(n) = tempib(m)
              arjb(n) = tempjb(m)
           endif
        enddo

        count = arbn

        if (arbn .gt. 0) then
           IB(1) = arib(1)
           JB(1) = arjb(1)
           if(atm_friend(IB(1))-atm_friend(JB(1)).eq.0)then
              BondType(1) = 1
           else if(atm_friend(IB(1))-atm_friend(JB(1)).eq.1)then
              BondType(1) = 1
              show = 1
           endif
        end if

        do i = 2, arbn
           IB(i) = arib(i)
           JB(i) = arjb(i)
           if((atm_friend(IB(i))-atm_friend(JB(i))) &
                .eq.1)then
              show = show + 1
              if(mod(show,2).eq.0) BondType(i) = 2
              if(mod(show,2).eq.1) BondType(i) = 1
              write(outu,*) 'show=',show
           else
              BondType(i) = 99   !temporary bondtype
           endif
        enddo

        do nb = 2, arbn
           if(BondType(nb).eq.99)then
              p = 0
              fubond = 0
              do o = 1, arbn
                 if(BondType(o).ne.99)then
                    if(IB(nb).eq.IB(o).or.IB(nb).eq.JB(o).or. &
                         JB(nb).eq.IB(o).or.JB(nb).eq.JB(o))then
                       p = p + 1
                       fuatom(p) = IB(o)
                       fubond = fubond + BondType(o)
                    endif
                 endif
              enddo

              if(p.eq.1)then
                 if(fubond.eq.1) BondType(nb) = 2
                 if(fubond.eq.2) BondType(nb) = 1
              else if(p.ge.2)then
                 if(fubond.eq.2) BondType(nb) = 2
                 if(fubond.ge.3) BondType(nb) = 1
              else if(p.eq.0)then
                 BondType(nb) = 1
              endif
           endif
        enddo

        do nb = BOFF+1,NBOND
           if(bond(nb)(1:2) .ne. 'ar')then
              count = count + 1
              IB(count) = IB1(nb)
              JB(count) = JB1(nb)
              if(bond(nb)(1:2) .eq. 'am') BondType(count) = 1
              if(bond(nb)(1:1) .eq. '1') BondType(count) = 1
              if(bond(nb)(1:1) .eq. '2') BondType(count) = 2
              if(bond(nb)(1:1) .eq. '3') BondType(count) = 3
           endif
        enddo


        ! I-Jen Chen 08/12/99
        ! Formal Charge Assignment
        !

        do nat=1,NATOM
           totalb=0
           nb=1
           do while(nb.le.NBOND)
              if(IB(nb).eq.nat.or.JB(nb).eq.nat)then
                 totalb=totalb+BondType(nb)
              endif
              nb= nb+1
           enddo
           selec(nat) = totalb

           if(AtNum(nat).eq.6)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.7)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.8)then
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.1)lonepair(nat)=3
           endif

           if(AtNum(nat).eq.16)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.9.or.AtNum(nat).eq.17.or.AtNum(nat).eq. &
                35.or.AtNum(nat).eq.54)then
              if(selec(nat).eq.1)lonepair(nat)=3
              if(selec(nat).eq.0)lonepair(nat)=4
           endif

           if(AtNum(nat).ne.8.and.AtNum(nat).ne.7.and. &
                AtNum(nat).ne.16.and.AtNum(nat).ne.9.and. &
                AtNum(nat).ne.17.and.AtNum(nat).ne.35.and. &
                AtNum(nat).ne.54.and.AtNum(nat).ne.6.) &
                lonepair(nat)=0

           NICHG(nat) = velec(nat)-selec(nat)-lonepair(nat)*2

           if(AtNum(nat).eq.3) NICHG(nat)=1
           if(AtNum(nat).eq.19) NICHG(nat)=1
           if(AtNum(nat).eq.20) NICHG(nat)=2
           if(AtNum(nat).eq.12) NICHG(nat)=2
           if(AtNum(nat).eq.11) NICHG(nat)=1
           if(AtNum(nat).eq.30) NICHG(nat)=2
           if(AtNum(nat).eq.55) NICHG(nat)=1
           if(AtNum(nat).eq.25) NICHG(nat)=2
           if(AtNum(nat).eq.29.and.ngrp.eq.2) NICHG(nat)=1
           if(AtNum(nat).eq.29.and.ngrp.eq.4) NICHG(nat)=2


        enddo

        ! I-Jen left.

        if(AOFF.gt.0) then
           do nb=BOFF+1,NBOND
              IB(nb)=IB(nb)+AOFF
              JB(nb)=JB(nb)+AOFF
           enddo
        endif
        !
        NTHERE=.TRUE. !  SEE IF RESIDUE NAMES ARE PRESENT IN THE MOL FILE
        ANRES(1:NRES0)=SNAME(1:NRES0)
        DO I=AOFF+1,NATOM
           IF(RESNAME(I).EQ.' ') NTHERE=.FALSE.
           IF(RESNAME(I).NE.' ') ANRES=RESNAME(I)
        ENDDO
        !
        AKRES(1:KRES0)=SNAME(1:KRES0) ! SEE IF RESIDUE TYPES ARE PRESENT IN THE MOL FILE
        KTHERE=.TRUE.
        DO I=AOFF+1,NATOM
           IF(KRES(I).EQ.' ') KTHERE=.FALSE.
           IF(KRES(I).NE.' ') AKRES=KRES(I)
        ENDDO
        IF(KTHERE) GOTO 3101
3101    CONTINUE


        DO I=AOFF+1,NATOM
           IF(AtNames(I).eq.' ') then
              ATN(1:2)=ElementSymbol(AtNum(i))
              if(ATN(1:2).eq.' ') call wrndie(-5,'<molin>', &
                   'error in name assignment')
              IF(I.GE.100) THEN
                 WRITE(ATN(2:4),'(I3)') I
              ELSEIF(I.GE.10) THEN
                 WRITE(ATN(2:3),'(I2)') I
              ELSE
                 WRITE(ATN(2:2),'(I1)') I
              ENDIF
              AtNames(I)=ATN(1:4) ! PLACE CONSTRUCTED ATOM NAME IN AtNames(I)
           endif
           IF(.NOT.NTHERE) RESNAME(I)=ANRES
           IF(.NOT.KTHERE) KRES(I)=AKRES
        ENDDO

        fcount=fcount+1

        !
        ! I-Jen Chen was here.
502     READ(imolf,'(A)',END=4004) SCRTCH
        if(SCRTCH(1:3).ne.'END') goto 502
4004    continue
        backspace(imolf)
        ! I-Jen left.
        !
     enddo !  do while (.true.) ! read concatenated files...
3201 ERROR=.TRUE.
     WRITE(SCRTCH,'(A,I4)') 'ERROR IN READING XYZ CARD FOR ATOM',NAT
     !
     ! I-Jen Chen was here.
     !           CALL WRNDIE(-5,'<molin>',SCRTCH(:44))
     CALL WRNDIE(-5,'<molin>',SCRTCH(:46))
     ! I-Jen left.
     return
3801 ERROR=.TRUE.
     CALL WRNDIE(-5,'<molin>','ERROR IN reading bonds, CHECK FILE')
     RETURN
4001 ERROR=.TRUE.
     CALL WRNDIE(1,'<molin>','Input file incomplete or unreadable')
     RETURN
4201 CONTINUE

     if(fcount.eq.0) then
        ERROR=.TRUE.
        CALL WRNDIE(1,'<molin>','--- End of MOL-file input ---')
     endif   !  if(fcount.eq.0)

!!!!!!!!!!!!!!!!!!READ DB FILE!!!!!!!!!!!!!!!!!

  else if(KEY.eq.'DB  ')then

     ! Start scanning the whole database to identify the target compound 

     rewind(imolf)
     READ(imolf,'(A)',iostat=eof) SCRTCH
     do while(eof.ge.0)
        if(SCRTCH(1:20).eq.cmpd)then
           no_cmpd = .false.
           goto 4005
        else
           READ(imolf,'(A)',iostat=eof) SCRTCH
        endif
     enddo

     if(no_cmpd) &
          write(outu,'(a10/,a11,1x,a20,a12)') &
          'WARNING!!!','THERE IS NO',cmpd,'IN THE FILE.'
     return
     ! End scanning
4005 continue


     do i = 1, 5
        backspace(imolf)
     enddo

     fcount = 0

     do while (.true.) ! read concatenated files...

        READ(imolf,'(A)',END=4202) SCRTCH ! this should allow to read concatenated files
        if(SCRTCH(1:3).eq.'END') then   ! from input script
!write(*,*)' Exiting 2'
           if(allocated(tmpX)) call dealloc_merckio(nallocated)
           return
        endif
        TITLEB(1)=SCRTCH
        READ(imolf,'(A)',END=4009) TITLEB(2)

        !
        ! I-Jen Chen was here.

503     READ(imolf,'(A)',END=4007) SCRTCH
        if(SCRTCH(10:17).ne.'MOLECULE') goto 503
4007    continue

        READ(imolf,'(A)') SCRTCH
        ! I-Jen left.
        !
        NTITLB=2
        !
        old_resname='?XXX'
        if(NATOM.gt.0) old_resname=resname(NATOM)
        old_segn='?XXX'
        if(NSEG.gt.0) old_segn=SEGID(NSEG)
        !
        AOFF=NATOM
 !write(*,*)' Before DB NATOM=', natom,' nallocated=',nallocated
       BOFF=NBOND
        READ(imolf,*,END=4009) NATOM,NBOND,junk1,junk2,junk3
!write(*,*)' After DB NATOM=', natom,' nallocated=',nallocated

        NATOM=NATOM+AOFF
        NBOND=NBOND+BOFF
#ifdef KEY_RESIZE
        if (natom .gt. na0) then
           na0=natom+drsz0
           call resize_psf('psfres.F90','molin','NATOM',na0, .true.)
           call resize_coord('psfres.F90','molin',na0, .true.)
           call resize_coordc('psfres.F90','molin',na0, .true.)
           call resize_deriv('psfres.F90','molin',na0, .true.)
        endif
        if (nbond .gt. nb0) then
           nb0=nbond+drsz0
           call resize_psf('psfres.F90','molin','NBOND',nb0, .true.)
        endif
#endif
        if (cnst_is_allocated) call allocate_cnst(natom)
        ! Allocate temporary space
        if(.not. allocated(tmpX)) then
           nallocated = max(natom,nbond)
           call alloc_merckio(nallocated)

        ! reallocate with additional size
        elseif(allocated(tmpX) .and. nallocated < max(natom,nbond)) then 
           nallocated = max(natom,nbond)
           call realloc_merckio(nallocated)
        endif

        !
        IF(NATOM.LT.1.OR.NATOM.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>',' ERROR IN ATOM COUNT,CHECK FILE')
           return
        endif

        IF(NBOND.LT.0.OR.NBOND.GT.MAXAIM) then
           ERROR=.TRUE.
           CALL WRNDIE(-5,'<molin>','ERROR IN BOND COUNT,CHECK FILE')
           RETURN
        endif
        !


        !
        ! I-Jen Chen was here.
        !
505     READ(imolf,'(A)',END=4008) SCRTCH
        if(SCRTCH(10:13).ne.'ATOM') goto 505
4008    continue


        ! ==== START COOR READ ====

        numb = 0
        numb2 = 0

        DO NAT=AOFF+1,NATOM ! READ THE X,Y,Z VALUES AND ATOM ATTRIBUTES, IF PRESENT

           READ(imolf,'(a)',END=4009,ERR=3202) SCRTCH
           read(SCRTCH(9:12),'(a4)')    tmpAtNames(NAT) ! atomic name
           read(SCRTCH(17:26),'(f10.4)') tmpX(NAT)   ! X - coordinate
           read(SCRTCH(27:36),'(f10.4)') tmpY(NAT)   ! Y - coordinate
           read(SCRTCH(37:46),'(f10.4)') tmpZ(NAT)   ! Z - coordinate

           ! ==== ATOMIC NUMBER CONVERSION ====

           if(SCRTCH(48:49) .eq. 'Al') tmpAtNum(NAT) = 13
           if(SCRTCH(48:49) .eq. 'As') tmpAtNum(NAT) = 33
           if(SCRTCH(48:49) .eq. 'Ag') tmpAtNum(NAT) = 47
           if(SCRTCH(48:49) .eq. 'At') tmpAtNum(NAT) = 85
           if(SCRTCH(48:49) .eq. 'Au') tmpAtNum(NAT) = 79

           if(SCRTCH(48:49) .eq. 'B ') tmpAtNum(NAT) =  5
           if(SCRTCH(48:49) .eq. 'Be') tmpAtNum(NAT) =  4
           if(SCRTCH(48:49) .eq. 'Ba') tmpAtNum(NAT) = 56
           if(SCRTCH(48:49) .eq. 'Br')then
              tmpAtNum(NAT) = 35
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Bi') tmpAtNum(NAT) = 83

           if(SCRTCH(48:49) .eq. 'C.')then
              tmpAtNum(NAT) =  6
              tmpvelec(NAT) =  4
           endif
           if(SCRTCH(48:49) .eq. 'Cl')then
              tmpAtNum(NAT) = 17
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Ca') tmpAtNum(NAT) = 20
           if(SCRTCH(48:49) .eq. 'Cr') tmpAtNum(NAT) = 24
           if(SCRTCH(48:49) .eq. 'Cu') tmpAtNum(NAT) = 29
           if(SCRTCH(48:49) .eq. 'Cd') tmpAtNum(NAT) = 48
           if(SCRTCH(48:49) .eq. 'Cs') tmpAtNum(NAT) = 55

           if(SCRTCH(48:49) .eq. 'Du') tmpAtNum(NAT) = 0

           if(SCRTCH(48:49) .eq. 'F ' .or. &
                SCRTCH(48:49) .eq. 'F.')then
              tmpAtNum(NAT) =  9
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'Fe') tmpAtNum(NAT) = 26

           if(SCRTCH(48:49) .eq. 'Ga') tmpAtNum(NAT) = 31
           if(SCRTCH(48:49) .eq. 'Ge') tmpAtNum(NAT) = 32

           if(SCRTCH(48:49) .eq. 'H ')then
              tmpAtNum(NAT) =  1
              tmpvelec(NAT) =  1
           endif

           if(SCRTCH(48:49) .eq. 'Hg') tmpAtNum(NAT) = 80

           if(SCRTCH(48:49) .eq. 'I ')then
              tmpAtNum(NAT) = 53
              tmpvelec(NAT) =  7
           endif
           if(SCRTCH(48:49) .eq. 'In') tmpAtNum(NAT) = 49

           if(SCRTCH(48:49) .eq. 'K ') tmpAtNum(NAT) = 19

           if(SCRTCH(48:49) .eq. 'Li') tmpAtNum(NAT) =  3
           if(SCRTCH(48:49) .eq. 'Lu') tmpAtNum(NAT) = 71

           if(SCRTCH(48:49) .eq. 'Mg') tmpAtNum(NAT) = 12
           if(SCRTCH(48:49) .eq. 'Mn') tmpAtNum(NAT) = 25
           if(SCRTCH(48:49) .eq. 'Mo') tmpAtNum(NAT) = 42

           if(SCRTCH(48:49) .eq. 'N.'.or. &
                SCRTCH(48:49) .eq. 'N ')then
              tmpAtNum(NAT) =  7
              tmpvelec(NAT) =  5
           endif
           if(SCRTCH(48:49) .eq. 'Ne') tmpAtNum(NAT) = 10
           if(SCRTCH(48:49) .eq. 'Na') tmpAtNum(NAT) = 11
           if(SCRTCH(48:49) .eq. 'Ni') tmpAtNum(NAT) = 28
           if(SCRTCH(48:49) .eq. 'Nb') tmpAtNum(NAT) = 41

           if(SCRTCH(48:49) .eq. 'O.'.or. &
                SCRTCH(48:49) .eq. 'O ')then
              tmpAtNum(NAT) =  8
              tmpvelec(NAT) =  6
           endif

           if(SCRTCH(48:49) .eq. 'P '.or. &
                SCRTCH(48:49) .eq. 'P.') tmpAtNum(NAT) = 15
           if(SCRTCH(48:49) .eq. 'Pd') tmpAtNum(NAT) = 46
           if(SCRTCH(48:49) .eq. 'Pt') tmpAtNum(NAT) = 78
           if(SCRTCH(48:49) .eq. 'Pb') tmpAtNum(NAT) = 82

           if(SCRTCH(48:49) .eq. 'Rb') tmpAtNum(NAT) = 37
           if(SCRTCH(48:49) .eq. 'Re') tmpAtNum(NAT) = 75
           if(SCRTCH(48:49) .eq. 'Ru') tmpAtNum(NAT) = 44
           if(SCRTCH(48:49) .eq. 'Rh') tmpAtNum(NAT) = 45

           if(SCRTCH(48:49) .eq. 'S.'.or. &
                SCRTCH(48:49) .eq. 'S ')then
              tmpAtNum(NAT) = 16
              tmpvelec(NAT) =  6
           endif

           if(SCRTCH(48:49) .eq. 'Si') tmpAtNum(NAT) = 14
           if(SCRTCH(48:49) .eq. 'Sr') tmpAtNum(NAT) = 38
           if(SCRTCH(48:49) .eq. 'Sc') tmpAtNum(NAT) = 21
           if(SCRTCH(48:49) .eq. 'Sn') tmpAtNum(NAT) = 50
           if(SCRTCH(48:49) .eq. 'Sb') tmpAtNum(NAT) = 51
           if(SCRTCH(48:49) .eq. 'Se') tmpAtNum(NAT) = 34

           if(SCRTCH(48:49) .eq. 'Ti') tmpAtNum(NAT) = 22 
           if(SCRTCH(48:49) .eq. 'Zn') tmpAtNum(NAT) = 30

           tmploc(NAT) = NAT

           if(SCRTCH(50:51) .eq. 'ar')then
              numb = numb + 1
              aratom(numb) = SCRTCH(9:12)
              t2loc(numb) = numb
              t2AtNames(numb) = tmpAtNames(NAT)
              t2X(numb) = tmpX(NAT)
              t2Y(numb) = tmpY(NAT)
              t2Z(numb) = tmpZ(NAT)
              t2AtNum(numb) = tmpAtNum(NAT)
              t2velec(numb) = tmpvelec(NAT)
           else
              if(SCRTCH(50:51) .ne. 'ar')then
                 numb2 = numb2 + 1

                 nonarloc(numb2) = numb2

                 nonarAtNames(numb2) = tmpAtNames(NAT)

                 nonarX(numb2) = tmpX(NAT)

                 nonarY(numb2) = tmpY(NAT)

                 nonarZ(numb2) = tmpZ(NAT)

                 nonarAtNum(numb2) = tmpAtNum(NAT)

                 nonarvelec(numb2) = tmpvelec(NAT)
              endif
           endif

           !
           ! NICHGI values will be assigned after reading the BOND info.
           !
           ! NICHGI = (valence e) -1/2(shared e) - (lone pair e)
           !
           ! I-Jen Chen  
           !

           read(SCRTCH(55:58),'(a4)')    resname(NAT) ! residue id (number)
           read(SCRTCH(60:63),'(a4)')    KRES(NAT)    ! residue type
           read(SCRTCH(67:70),'(a4)')    segn         ! segment id

           if (resname(NAT) .ne. ' ') then
              ridlen = LEN(resname(NAT))
              do while (resname(NAT)(1:1).EQ.' ')
                 resname(NAT) = resname(NAT)(2:ridlen)//' '
              enddo
           endif

           if(resname(NAT).ne.old_resname.or.segn.ne.old_segn)then
              !23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
              old_resname=resname(NAT)
              nres=nres+1
#ifdef KEY_RESIZE
              if (nres .gt. nr0) then
                 nr0=nres+drsz0
                 call resize_psf('psfres.F90','molin','NRES',nr0, .true.)
              endif
#endif
              ibase(nres)=NAT-1
              resid(nres)=resname(NAT)
              res(nres)=kres(NAT)
              ngrp=ngrp+1            ! group = reside
#ifdef KEY_RESIZE
              if (ngrp .gt. ng0) then
                 ng0=ngrp+drsz0
                 call resize_psf('psfres.F90','molin','NGRP',ng0, .true.)
              endif
#endif
              igpbs(ngrp)=NAT-1
           endif

           if(segn.ne.old_segn) then
              old_segn=segn
              nsegm=nseg
              nseg=nseg+1
#ifdef KEY_RESIZE
              if (nseg .gt. ns0) then
                 ns0=nseg+drsz0
                 call resize_psf('psfres.F90','molin','NSEG',ns0, .true.)
              endif
#endif              
              segid(nseg)=segn
              IF (SEGID(NSEG) .EQ. ' ') then
                 CALL ENCODI(NSEG, SEGID(NSEG), 8, IDUMMY)
                 IF(NSEG.EQ.1) SEGID(NSEG)='MMFF'
              endif
              nictot(nseg)=nres-1 !????
              !...........Check for duplicate segment names.
              DO I = 1,NSEGM
                 IF(SEGID(I) .EQ. SEGID(NSEG)) THEN
                    NSEG = NSEGM
                    CALL WRNDIE(-2,'<molin>','DUPLICATE SEGMENT NAMES')
                    RETURN
                 ENDIF
              ENDDO ! DO I = 1,NSEGM
              NSEGM=NSEG
           endif ! if(segn.ne.old_segn) then

        ENDDO ! DO NAT=AOFF+1,NATOM

        ! ==== END OF COOR READ ====

        !
        ! ASSIGN ATNAME, XYZ, AND ATNUM OF AROMATIC ATOMS INTO THE
        ! FINAL ARRAY.
        ! 

        !           numb3 = 0
        do loc1=1, numb
           !              numb3 = numb3 + 1
           loc(loc1) = loc1
           AtNames(loc1) = t2AtNames(loc1)
           X(loc1) = t2X(loc1)
           Y(loc1) = t2Y(loc1)
           Z(loc1) = t2Z(loc1)
           AtNum(loc1) = t2AtNum(loc1)
           velec(loc1) = t2velec(loc1)
        enddo

        !
        ! ASSIGN ATNAME, XYZ, AND ATNUM OF NONAROMATIC ATOMS INTO THE
        ! FINAL ARRAY.
        !

        do loc2 = 1, numb2

           loc(loc2+numb) = loc2 + numb

           AtNames(loc2+numb) = nonarAtNames(loc2)

           X(loc2+numb) = nonarX(loc2)

           Y(loc2+numb) = nonarY(loc2)

           Z(loc2+numb) = nonarZ(loc2)

           AtNum(loc2+numb) = nonarAtNum(loc2)

           velec(loc2+numb) = nonarvelec(loc2)

        enddo

        nictot(nseg+1)=nres
        ibase(nres+1)=natom
        igpbs(ngrp+1)=natom
        !

        ! Skip the line that separates the coor and the bond info.

        READ(imolf,'(A)',END=4009)

        ! ==== START BOND INFO READ ====

        do nb = BOFF+1,NBOND
           READ(imolf,100,END=3802)tmpIB(nb),tmpJB(nb),tmpbond(nb)
        enddo

        ! I-Jen Chen 07/28/99
        ! CHANGE THE ATOM PAIRS IN THE BOND LIST IN ACCORD WITH THE NEW ATOM
        ! LIST.
        !

        do nb = 1,NBOND
           do nat = 1,NATOM
              if(tmpAtNames(tmpIB(nb)).eq.AtNames(nat))IB1(nb)=nat
              if(tmpAtNames(tmpJB(nb)).eq.AtNames(nat))JB1(nb)=nat
              bond(nb) = tmpbond(nb)
           enddo
        enddo

        ! ==== BOND TYPE REASSIGNMENT ====
        !
        arbn = 0
        do nb = BOFF+1,NBOND
           if(bond(nb)(1:2) .eq. 'ar')then
              arbn = arbn + 1
              if(IB1(nb) .gt. JB1(nb))then
                 arib(arbn) = JB1(nb)
                 arjb(arbn) = IB1(nb)
              else
                 arib(arbn) = IB1(nb)
                 arjb(arbn) = JB1(nb)
              endif
              arbond(arbn) = bond(nb)
           endif
        enddo

        ! Sorting the bond pair list in the order of ARIB

        pairs = arbn - 1
1056    if(pairs .gt. 0)then
           last = 1
           do i= 1, pairs
              if(arib(i) .ge. arib(i+1))then

                 tempj = arjb(i)
                 arjb(i) = arjb(i+1)
                 arjb(i+1) = tempj

                 tempi = arib(i)
                 arib(i) = arib(i+1)
                 arib(i+1) = tempi

                 tempab = arbond(i)
                 arbond(i) = arbond(i+1)
                 arbond(i+1) = tempab

                 last = i

              endif
           enddo
           pairs = last -1
           goto 1056

        endif

        pairs = arbn - 1
1058    if(pairs .gt. 0)then
           last = 1
           do i= 1, pairs
              if(arib(i).eq.arib(i+1).and.arjb(i).gt.arjb(i+1))then

                 tempj = arjb(i)
                 arjb(i) = arjb(i+1)
                 arjb(i+1) = tempj

                 tempi = arib(i)
                 arib(i) = arib(i+1)
                 arib(i+1) = tempi

                 tempab = arbond(i)
                 arbond(i) = arbond(i+1)
                 arbond(i+1) = tempab

                 last = i

              endif
           enddo
           pairs = last -1
           goto 1058
        endif

        k = 1

        do i = 1, arbn
           if(arib(i).ne.9999)then
              tempib(k) = arib(i)
              tempjb(k) = arjb(i)
              do j = i+1, arbn
                 if((arib(j).ne.9999).and.((arjb(i).eq.arjb(j)).or. &
                      (arjb(i).eq.arib(j))))then
                    found(k) = j
                    k = k + 1
                    tempib(k) = arib(j)
                    tempjb(k) = arjb(j)
                    arib(j) = 9999
                    arjb(j) = 9999
                 endif
              enddo
              k = k + 1
           endif
        enddo

        ! Count the number of neighbors each aromatic atom has

        do i = 1, numb
           neighbor = 0
           do j = 1, arbn
              if(atnames(i).eq.atnames(tempib(j)).or.atnames(i).eq. &
                   atnames(tempjb(j))) neighbor = neighbor + 1
           enddo
           atm_friend(i) = neighbor
        enddo
        ! Go through the bond list and put the pair with two fused atoms in the first

        do i = 1, arbn
           if((atm_friend(tempib(i))-atm_friend(tempjb(i))).eq.-1) &
                then
              tempi = tempib(i)
              tempib(i) = tempjb(i)
              tempjb(i) = tempi

           endif
        enddo

        l = 0
        do j = 1, arbn
           if((atm_friend(tempib(j))-atm_friend(tempjb(j))).eq.1) &
                then
              l = l + 1
              arib(l) = tempib(j)
              arjb(l) = tempjb(j)
              tempib(j) = 999
              tempjb(j) = 999
              do k = 1, arbn
                 if(tempib(k).ne.999.and.arib(j).eq.tempib(k))then
                    l = l + 1
                    arib(l) = tempib(k)
                    arjb(l) = tempjb(k)
                    tempib(k) = 999
                    tempjb(k) = 999
                 endif
              enddo
           endif
        enddo

        n = l
        do m = 1, arbn
           if(tempib(m).ne.999)then
              n = n + 1
              arib(n) = tempib(m)
              arjb(n) = tempjb(m)
           endif
        enddo

        count = arbn
        
        if (arbn .gt. 0) then
           IB(1) = arib(1)
           JB(1) = arjb(1)
           if(atm_friend(IB(1))-atm_friend(JB(1)).eq.0)then
              BondType(1) = 1
           else if(atm_friend(IB(1))-atm_friend(JB(1)).eq.1)then
              BondType(1) = 1
              show = 1
           endif
        end if

        do i = 2, arbn
           IB(i) = arib(i)
           JB(i) = arjb(i)
           if((atm_friend(IB(i))-atm_friend(JB(i))) &
                .eq.1)then
              show = show + 1
              if(mod(show,2).eq.0) BondType(i) = 2
              if(mod(show,2).eq.1) BondType(i) = 1
              write(outu,*) 'show=',show
           else
              BondType(i) = 99   !temporary bondtype
           endif
        enddo

        do nb = 2, arbn
           if(BondType(nb).eq.99)then
              p = 0
              fubond = 0
              do o = 1, arbn
                 if(BondType(o).ne.99)then
                    if(IB(nb).eq.IB(o).or.IB(nb).eq.JB(o).or. &
                         JB(nb).eq.IB(o).or.JB(nb).eq.JB(o))then
                       p = p + 1
                       fuatom(p) = IB(o)
                       fubond = fubond + BondType(o)
                    endif
                 endif
              enddo

              if(p.eq.1)then
                 if(fubond.eq.1) BondType(nb) = 2
                 if(fubond.eq.2) BondType(nb) = 1
              else if(p.ge.2)then
                 if(fubond.eq.2) BondType(nb) = 2
                 if(fubond.ge.3) BondType(nb) = 1
              else if(p.eq.0)then
                 BondType(nb) = 1
              endif

           endif
        enddo

        do nb = BOFF+1,NBOND
           if(bond(nb)(1:2) .ne. 'ar')then
              count = count + 1
              IB(count) = IB1(nb)
              JB(count) = JB1(nb)
              if(bond(nb)(1:2) .eq. 'am') BondType(count) = 1
              if(bond(nb)(1:1) .eq. '1') BondType(count) = 1
              if(bond(nb)(1:1) .eq. '2') BondType(count) = 2
              if(bond(nb)(1:1) .eq. '3') BondType(count) = 3
           endif
        enddo

        ! I-Jen Chen 08/12/99
        ! Formal Charge Assignment
        !

        do nat=1,NATOM
           totalb=0
           nb=1
           do while(nb.le.NBOND)
              if(IB(nb).eq.nat.or.JB(nb).eq.nat)then
                 totalb=totalb+BondType(nb)
              endif
              nb= nb+1
           enddo
           selec(nat) = totalb

           if(AtNum(nat).eq.6)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.7)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.8)then
              if(selec(nat).eq.3)lonepair(nat)=1
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.1)lonepair(nat)=3
           endif

           if(AtNum(nat).eq.16)then
              if(selec(nat).eq.2)lonepair(nat)=2
              if(selec(nat).eq.4)lonepair(nat)=0
           endif

           if(AtNum(nat).eq.9.or.AtNum(nat).eq.17.or.AtNum(nat).eq. &
                35.or.AtNum(nat).eq.54)then
              if(selec(nat).eq.1)lonepair(nat)=3
              if(selec(nat).eq.0)lonepair(nat)=4
           endif

           if(AtNum(nat).ne.8.and.AtNum(nat).ne.7.and. &
                AtNum(nat).ne.16.and.AtNum(nat).ne.9.and. &
                AtNum(nat).ne.17.and.AtNum(nat).ne.35.and. &
                AtNum(nat).ne.54.and.AtNum(nat).ne.6.) &
                lonepair(nat)=0

           NICHG(nat) = velec(nat)-selec(nat)-lonepair(nat)*2

           if(AtNum(nat).eq.3) NICHG(nat)=1
           if(AtNum(nat).eq.19) NICHG(nat)=1
           if(AtNum(nat).eq.20) NICHG(nat)=2
           if(AtNum(nat).eq.12) NICHG(nat)=2
           if(AtNum(nat).eq.11) NICHG(nat)=1
           if(AtNum(nat).eq.30) NICHG(nat)=2
           if(AtNum(nat).eq.55) NICHG(nat)=1
           if(AtNum(nat).eq.25) NICHG(nat)=2
           if(AtNum(nat).eq.29.and.ngrp.eq.2) NICHG(nat)=1
           if(AtNum(nat).eq.29.and.ngrp.eq.4) NICHG(nat)=2
        enddo

        ! I-Jen left.

        if(AOFF.gt.0) then
           do nb=BOFF+1,NBOND
              IB(nb)=IB(nb)+AOFF
              JB(nb)=JB(nb)+AOFF
           enddo
        endif
        !
        NTHERE=.TRUE. !  SEE IF RESIDUE NAMES ARE PRESENT IN THE MOL FILE
        ANRES(1:NRES0)=SNAME(1:NRES0)
        DO I=AOFF+1,NATOM
           IF(RESNAME(I).EQ.' ') NTHERE=.FALSE.
           IF(RESNAME(I).NE.' ') ANRES=RESNAME(I)
        ENDDO
        !
        AKRES(1:KRES0)=SNAME(1:KRES0) ! SEE IF RESIDUE TYPES ARE PRESENT IN THE MOL FILE
        KTHERE=.TRUE.
        DO I=AOFF+1,NATOM
           IF(KRES(I).EQ.' ') KTHERE=.FALSE.
           IF(KRES(I).NE.' ') AKRES=KRES(I)
        ENDDO
        IF(KTHERE) GOTO 3102
3102    CONTINUE


        DO I=AOFF+1,NATOM
           IF(AtNames(I).eq.' ') then
              ATN(1:2)=ElementSymbol(AtNum(i))
              if(ATN(1:2).eq.' ') call wrndie(-5,'<molin>', &
                   'error in name assignment')
              IF(I.GE.100) THEN
                 WRITE(ATN(2:4),'(I3)') I
              ELSEIF(I.GE.10) THEN
                 WRITE(ATN(2:3),'(I2)') I
              ELSE
                 WRITE(ATN(2:2),'(I1)') I
              ENDIF
              AtNames(I)=ATN(1:4) ! PLACE CONSTRUCTED ATOM NAME IN AtNames(I)
           endif
           IF(.NOT.NTHERE) RESNAME(I)=ANRES
           IF(.NOT.KTHERE) KRES(I)=AKRES
        ENDDO

        fcount=fcount+1

        !
        ! I-Jen Chen was here.
504     READ(imolf,'(A)',END=4006) SCRTCH
        if(SCRTCH(1:3).ne.'END') goto 504
4006    continue
        backspace(imolf)
        ! I-Jen left.
        !
     enddo !  do while (.true.) ! read concatenated files...

3202 ERROR=.TRUE.
     WRITE(SCRTCH,'(A,I4)') 'ERROR IN READING XYZ CARD FOR ATOM',NAT
     !
     ! I-Jen Chen was here.
     !           CALL WRNDIE(-5,'<molin>',SCRTCH(:44))
     CALL WRNDIE(-5,'<molin>',SCRTCH(:46))
     ! I-Jen left.
     return
3802 ERROR=.TRUE.
     CALL WRNDIE(-5,'<molin>','ERROR IN reading bonds, CHECK FILE')
     RETURN
4009 ERROR=.TRUE.
     CALL WRNDIE(1,'<molin>','Input file incomplete or unreadable')
     RETURN
4202 CONTINUE

     if(fcount.eq.0) then
        ERROR=.TRUE.
        CALL WRNDIE(1,'<molin>','--- End of MOL-file input ---')
     endif   !  if(fcount.eq.0)

     !        endif      !  if(KEY.eq.'MOL2')

  endif        !  if(KEY.eq.'MERC')
#ifdef KEY_RESIZE
  if (natom .lt. na0) then
     call resize_psf('psfres.F90','molin','NATOM',natom, .true.)
     call resize_coord('psfres.F90','molin',natom, .true.)
     call resize_coordc('psfres.F90','molin',natom, .true.)
     call resize_deriv('psfres.F90','molin',natom, .true.)
  endif
  if (nbond .lt. nb0) &
       call resize_psf('psfres.F90','molin','NBOND',nbond, .true.)
  if (ngrp .lt. ng0) &
       call resize_psf('psfres.F90','molin','NGRP',ngrp, .true.)
  if (nres .lt. nr0) &
       call resize_psf('psfres.F90','molin','NRES',nres, .true.)
  if (nseg .lt. ns0) &
       call resize_psf('psfres.F90','molin','NSEG',nseg, .true.)
#endif  
  !write(*,*)' Deallocating, End'
  if(allocated(tmpX)) call dealloc_merckio(nallocated)
  RETURN

END SUBROUTINE MOLIN

! ===============================================================
! SUBROUTINE MOLOUT : OUTPUTS A MOLEDIT FILE OF THE SUBSTRATE STRUCTURE.
! ===============================================================
!23456789 123456789 123456789 123456789 123456789 123456789 123456789 12
!
SUBROUTINE MOLOUT(TITLE,IUNIT, &
     NATOM,X,Y,Z,ATNUM,AtNames,NICHG,PARTLQ,MTYPE, &
     NBOND,IB,JB,BondType, &
     RESID,RES,IBASE,NRES, &
     SEGID,NICTOT,NSEG)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  27 Nov 95 Jay Banks: changed XYZ array to X, Y, Z.
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.  Likewise "END DO"s and "END IF"s.
  !
  !  12 Dec 95 Jay Banks: Changed CALL IDATE to CALL DAYTIM, for machine
  !  independence (Cray in particular).
  !
  use ctitla
  use energym
  use stream
  use machutil,only:daytim
  use startup
  implicit none
  !
  CHARACTER(len=*) TITLE(*)

  integer NATOM                          ! ATOMS
  real(chm_real)  X(*),Y(*),Z(*)
  integer ATNUM(*)
  real(chm_real)  PARTLQ(*)
  character(len=*) AtNames(*)
  integer NICHG(*)
  integer MTYPE(*)

  integer NBOND                          ! BONDS
  integer IB(*),JB(*),BondType(*)

  integer NRES,IBASE(*)                  ! RESIDUES
  character(len=*) RESID(*),RES(*)
  character(len=80) cur_resid
  integer ridlen, irid

  character(len=*) SEGID(*)                 ! SEGMENTS
  integer NSEG,NICTOT(*)
  !
  integer i, idd, iddd, IHOUR, IMINUTE, ISEC
  integer imm, ires, iseg, iunit, iverme, iyy, k, kmm, ldir
  integer lscr, lusrn, m
  integer nb, nseqi
  !
  CHARACTER(len=7) FFIELD_name
  !     CHARACTER*4 NEWN
  INTEGER :: IDAMON(12)=(/31,28,31,30,31,30,31,31,30,31,30,31/)
  !     CHARACTER*80 NTITLE ! ,OUTFIL
  CHARACTER(len=8) HHMMSS
  !     CHARACTER*4 IYN,IFNEW
  COMMON/USRINF/LUSRN,LDIR,LSCR, &
       USRNAM,GROUP,USRNUM,USRDIR,SCRDIR
  CHARACTER(len=12) USRNAM
  CHARACTER(len=3) GROUP, USRNUM
  CHARACTER(len=72) USRDIR, SCRDIR
  !     DOUBLE PRECISION XYZ2N
  !     LOGICAL ERROR
  !
  !   BRANCH IF NEXT=.TRUE., INDICATING THAT THIS STRUCTURE IS TO BE
  !   ADDED TO THE OUTPUT FILE BEING ACCUMULATED
  !     IF(NEXT.AND..NOT.FIRST) GOTO 450
  !   GET A NAME FOR AND OPEN THE OUTPUT FILE
  !     CALL OPNOUT(IUNIT,OUTFIL,' Enter a name for the output file',
  !    . 'MOL',USRDIR(1:LDIR),.TRUE.,ERROR)
  !     IF(ERROR) RETURN
  !   FIND OUT IF TITLE SHOULD BE CHANGED
  ! 450 CONTINUE
  !     WRITE(OUTU,500) TITLE(1)
  !     IF(IZ.NE.0) WRITE(IZ,500) TITLE(1)
  ! 500 FORMAT(' CURRENT TITLE IS:'/1X,A)
  !     CALL QUERY(OUTU,
  !    &' Do you want to specify a NEW TITLE? (Y/N) [N]: ')
  !     CALL QUERY(IZ,' Do you want to specify a NEW TITLE? (Y/N) [N]: ')
  !     CALL RDWORD(1,1,1,NCHAR,IYN)
  !     IF(IYN.EQ.' ') IYN='N'
  !     IF(IYN.NE.'Y'.AND.IYN.NE.'N') GOTO 450
  !     IF(IYN.EQ.'Y') THEN
  !        WRITE(OUTU,550)
  !     IF(IZ.NE.0) WRITE(IZ,550)
  ! 550    FORMAT(/' Enter new title  ............(must fit in this space)
  !    .............')
  !        CALL RDLINE(0,1,1,LENT,NTITLE(1:70))
  !     ELSE
  !        IYN='N'
  !        NTITLE(:70)=TITLE(1)(:70)
  !     ENDIF
  !   GENERATE THE MOLEDIT JULIAN DATE/TIME YYDDDHHMM, COLS 71-79
  CALL DAYTIM(IMM,IDD,IYY,IHOUR,IMINUTE,ISEC)
  IDDD=0
  KMM=0
1700 KMM=KMM+1
  IF(KMM.GE.IMM) GOTO 1800
  IDDD=IDDD+IDAMON(KMM)
  GOTO 1700
1800 CONTINUE
  IF(MOD(IYY,4).EQ.0.AND.IMM.GE.3) IDDD=IDDD+1
  IDDD=IDDD+IDD
  !CC      CALL TIME(HHMMSS)
  WRITE(HHMMSS,'(I2,'':'',I2,'':'',I2)')IHOUR,IMINUTE,ISEC
  ! -- MOLEDIT VERSION NUMBER
  IVERME=1
  ! WRITE TITLE.
  !...##IF DEBUG
  IYY=94            ! fix time stamp
  IDDD=103          ! for ease of automatic output
  HHMMSS='15:35:13' ! comparison
  !...##ENDIF
  WRITE(IUNIT,1900) TITLEB(1)(:70),IYY,IDDD,HHMMSS(1:2), &
       !      WRITE(IUNIT,1900) TITLE(1)(:70),IYY,IDDD,HHMMSS(1:2),
       HHMMSS(4:5),IVERME
1900 FORMAT(A70,I2,I3,2A2,I1)
  !   TEST COORDS TO SEE IF ENERGY IS CURRENT
  !     XYZ2N=0.
  !     DO 1950 I=1,NATOM
  !     DO 1950 K=1,3
  !     XYZ2N=XYZ2N+XYZ(I,K)**2
  !1950 CONTINUE
  FFIELD_name='MMFF94'
  !     IF(ABS(XYZ2N-XYZ2).LT.1.E00.AND.EPROP(GRMS).EQ.0.) THEN
  !        WRITE(IUNIT,2000) USRNAM(1:9),eterm(epot),FFIELD_name
  !2000    FORMAT('MOL ',A,'O  E = ',F12.4,2X,A5,40X,' ')
  !     ELSE IF(ABS(XYZ2N-XYZ2).LT.1.E00.AND.EPROP(GRMS).NE.0.) THEN
  ! T.A. Halgren change (B980629.wy)
  !        USRNAM='halgren'
  USRNAM=' '
  CALL GETNAM(USRNAM)
  !C         CALL GETUNAMEF(USRNAM, LEN(USRNAM))
  !        WRITE(IUNIT,2005) USRNAM(1:9),eterm(epot),EPROP(GRMS),FFIELD_name
  WRITE(IUNIT,2005) USRNAM(1:9),EPROP(epot),EPROP(GRMS),FFIELD_name
  ! end of change
2005 FORMAT('MOL ',A,'O  E = ',F12.4,2X, &
       ' G =',1PE10.2,2X,A7)
  !     ELSE
  !        IF(USRNAM.EQ.'HALGREN') THEN
  !           WRITE(OUTU,*) ' XYZ2N, XYZ2',XYZ2N,XYZ2
  !           IF(IZ.NE.0) WRITE(IZ,*) ' XYZ2N, XYZ2',XYZ2N,XYZ2
  !        ENDIF
  !        WRITE(IUNIT,2010) USRNAM(1:9)
  !2010    FORMAT('MOL ',A,'O ',64X,' ')
  !     ENDIF
  ! WRITE ATOM # AND BOND #.
  WRITE(IUNIT,2100) NATOM,NBOND
2100 FORMAT(I5,1X,I5)
  ! OUTPUT COORDS AND ATOM ATTRIBUTES
  do iseg=1,nseg
     do ires=NICTOT(ISEG)+1,NICTOT(ISEG+1)
        !    Right-justify the residue ID  (JLB 06 Oct 95)
        ridlen = LEN(resid(ires))
        cur_resid = resid(ires)
        if (cur_resid .ne. ' ') then
           do while (cur_resid(ridlen:ridlen).EQ.' ')
              !yw... The string copy does not work on Convex and IBM RS/6000
              !yw              cur_resid = ' '//cur_resid(1:ridlen-1)
              do irid=1,ridlen-1
                 cur_resid(ridlen-irid+1:ridlen-irid+1) = &
                      cur_resid(ridlen-irid:ridlen-irid)
              enddo
              cur_resid(1:1) = ' '
           enddo
        endif
        DO I=IBASE(IRES)+1,IBASE(IRES+1)
           NSEQI=I
           WRITE(IUNIT,2200) &
                X(I),Y(I),Z(I),ATNUM(I),MTYPE(I),NICHG(I), &
                NSEQI,AtNames(I),CUR_RESID,RES(IRES),PARTLQ(I),SEGID(ISEG)
2200       FORMAT(3(F10.4,1X),I5,1X,I2,I2,1X,I5,1X,3A4,F8.4,6X,A4)
        enddo ! DO I=IBASE(IRES)+1,IBASE(IRES+1)
     enddo ! do ires=NICTOT(ISEG)+1,NICTOT(ISEG+1)
  enddo ! do iseg=1,nseg
  ! OUTPUT THE BONDS.
  WRITE(IUNIT,2400) (IB(nb),JB(nb),BondType(nb),nb=1,NBOND)
2400 FORMAT(5(I5,1X,I5,1X,I2,2X))
  if(prnlev.ge.2) WRITE(OUTU,2500) NATOM,NBOND
  !     IF(IZ.NE.0) WRITE(IZ,2500) NATOM,NBOND
2500 FORMAT(' MOLEDIT FILE WRITTEN WITH ',I6,' ATOMS,',I6,' BONDS')
  !   CLOSE THE FILE
  !     IF(.NOT.NEXT) CALL VCLOSE(IUNIT,'KEEP',ERROR)
  !
  RETURN
END SUBROUTINE MOLOUT
#else /* (merckio)*/
SUBROUTINE NULL_merckio
  RETURN
END SUBROUTINE NULL_merckio

#endif /* (merckio)*/
end module merck_io

