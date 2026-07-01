MODULE alloccomp
  !
  !     Module for comparing allocation and deallocation
  !     databases
  !
  !                                --RJP Nov 2008
  use chm_kinds
  implicit none

CONTAINS 
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE compalloc(qnofile,qnoproc,qnoname,qnotype,qnorank, &
       qnosize,qcaseins,qmmverb)
    !
  use allocdat
  use deallocdat 
  use allocation
  use deallocation

    ! This routine compares the allocation and deallocation
    ! databases.
    !
    implicit none
    logical,optional,intent(in) :: qnofile,qnoproc,qnoname, &
         qnotype,qnorank,qnosize,qcaseins,qmmverb
    !
    integer,allocatable,dimension(:) :: snar,sdnar
    real(kind=chm_real),allocatable,dimension(:) :: rarsizcp, &
         rdarsizcp
    integer :: ii,jj,aa,dd,first,storefirst,ncounta,ncountd,cnt
    integer :: nnalloc,totcnt
    integer :: ncountma,ncountmd,nndeall,kk,storedsize,storeasize
    integer,allocatable,dimension(:) :: missmatchaar,missmatchdar
    integer,allocatable,dimension(:) :: multmatchaar,multmatchdar
    integer,allocatable,dimension(:) :: netmatalloc,netmatdeall
    logical :: qmatch,qmatchlst,qover,qloopmatch,qmat
    real(kind=chm_real) :: aasize,ddsize,storeaasz,storeddsz, &
         mallocsum,mdeallsum,fmallocsum,fmdeallsum,mallmissum,ssum, &
         mdeallmissum
    real(kind=chm_real) :: allwrdsum,deallwrdsum,fallwrdsum, &
         fdeallwrdsum
    integer,allocatable,dimension(:) :: deallocar,allocar
    integer,allocatable,dimension(:) :: cntdealloc,cntalloc
    integer :: orig,ierr,cntcnt
    logical :: qnoprocl,qnonamel,qnotypel,qnorankl, &
         qnosizel,qnofilel,qcaseinsl,qmmverbl
    character(len=3) :: chfile,chproc,chname,chtype,chrank,chsize, &
         chverb
    character(len=6) :: chcase
    !***************************************************************
    write(6,'(5X,A70)') &
         '**************************************************************'
    write(6,'(13X,A62)') &
         '-------COMPARING ALLOCATION AND DEALLOCATION DATABASES--------'
    write(6,'(5X,A70)') &
         '**************************************************************'
    qmmverbl = .false.
    qnofilel = .false.
    qnoprocl = .false.
    qnonamel = .false.
    qnotypel = .false.
    qnorankl = .false.
    qnosizel = .false.
    qcaseinsl = .false.
    if(present(qmmverb)) qmmverbl=qmmverb
    if(present(qnofile)) qnofilel=qnofile
    if(present(qnoproc)) qnoprocl=qnoproc
    if(present(qnoname)) qnonamel=qnoname
    if(present(qnotype)) qnotypel=qnotype
    if(present(qnorank)) qnorankl=qnorank
    if(present(qnosize)) qnosizel=qnosize
    if(present(qcaseins)) qcaseinsl=qcaseins
    write(6,'(30X,A29)') 'CRITERIA USED IN COMPARISONS:'
    !      chverb = 'No'
    chfile = 'Yes'
    chproc = 'Yes'
    chname = 'Yes'
    chtype = 'Yes'
    chrank = 'Yes'
    chsize = 'Yes'
    chcase = 'Sens'
    !      if(qmmverbl) chverb = 'Yes'
    if(qnofilel) chfile = 'No'
    if(qnoprocl) chproc = 'No'
    if(qnonamel) chname = 'No'
    if(qnotypel) chtype = 'No'
    if(qnorankl) chrank = 'No'
    if(qnosizel) chsize = 'No'
    if(qcaseinsl) chcase = 'Insens'
    write(6,'(13X,A10,A3,A17,A3,A13,A3,A13,A3)') &
         'File Name-',chfile,'; Procedure Name-',chproc, &
         '; Array Name-',chname,'; Array Type-',chtype
    !
    write(6,'(13X,A11,A3,A13,A3,A7,A6,A10,A3)')  &
         'Array Rank-',chrank,'; Array Size-',chsize,'; Case-',chcase
    !     & '; Verbose-',chverb
    !
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    !
    !      allocdbarsz = 1000  !these two have to be defined whether there are alloc/dealloc or not
    !      dealldbarsz = 1000  !this has to be de
    !      fdealldbarsz = 1000
    if(.not.allocated(rarsizcp)) then
       allocate(rarsizcp(allocdbarsz),rdarsizcp(dealldbarsz), &
            snar(allocdbarsz),sdnar(dealldbarsz), &
            missmatchaar(allocdbarsz),missmatchdar(dealldbarsz), &
            multmatchaar(allocdbarsz),multmatchdar(dealldbarsz), &
            allocar(allocdbarsz),deallocar(dealldbarsz), &
            cntalloc(allocdbarsz),cntdealloc(dealldbarsz), &
            netmatalloc(allocdbarsz),netmatdeall(dealldbarsz))
    endif
    !
    ! initialize all allocated arrays
    do ii = 1,allocdbarsz
       rarsizcp(ii) = 0
       snar(ii) = 0
       missmatchaar(ii) = 0
       multmatchaar(ii) = 0
       allocar(ii) = 0
       cntalloc(ii) = 0
       netmatalloc(ii) = 0
    enddo
    do jj = 1,dealldbarsz
       rdarsizcp(jj) = 0
       sdnar(jj) = 0
       missmatchdar(jj) = 0
       multmatchdar(jj) = 0
       deallocar(jj) = 0
       cntdealloc(jj) = 0
       netmatdeall(jj) = 0
    enddo
    !
    !----------------------------------------------------------------
    ! SORT ALLOCATIONS/DEALLOCATIONS BY SIZE
    !----------------------------------------------------------------
    !sort the allocations
    mallocsum = 0
    allwrdsum = 0
    do ii = 1,nallocentr
       snar(ii) = ii
       rarsizcp(ii) = arsizear(ii)
       cntdealloc(ii) = 0 !redundancy count for each entry
       allocar(ii) = 0  ! non-redundant array entries
       netmatalloc(ii) = 0 !count of net allocation/dealloc matches
       mallocsum = mallocsum + arsizear(ii)*arkindar(ii)
       allwrdsum = allwrdsum + arsizear(ii)
    enddo
    call qsort(rarsizcp,nallocentr,snar) 
    !        
    !sort the deallocations
    mdeallsum=0
    deallwrdsum = 0
    do jj = 1,ndeallocentr
       sdnar(jj) = jj 
       rdarsizcp(jj) = darsizear(jj)
       cntdealloc(jj) = 0 !redundancy count for each entry
       deallocar(jj) = 0  ! non-redundant array entries
       netmatdeall(jj) = 0 !count of net allocation/dealloc matches
       mdeallsum=mdeallsum + darsizear(jj)*darkindar(jj)
       deallwrdsum = deallwrdsum + darsizear(jj)
    enddo
    call qsort(rdarsizcp,ndeallocentr,sdnar)
    !
    !-------------------------------------------------------------
    ! END OF SORTING
    !-------------------------------------------------------------
    if(qmmverbl) then
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(22X,A40)') '(REDUNDANT) SORTED ARRAY ALLOCATION LIST'
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4)') &
            'number','file','procedu','name','type','rank','size','kind'
       do jj = 1,nallocentr
          aa = snar(jj)
          call prn_allocdpt('ALLREDN>',aa,filnamar(aa),procnamar(aa), &
               arnamar(aa),artypar(aa),arrankar(aa),arsizear(aa), &
               arkindar(aa))
       enddo
       if(qmmverbl) write(6,'(20X,A12,I9)') &
            'Array count:' ,nallocentr
    endif
    ! count and remove redundancies
    !first for allocations
    if(nallocentr.gt.0) then
       first = 1
       nnalloc = 1 
       allocar(nnalloc) = snar(1) 
       !       if (nallocentr.gt.0) then
       storeasize = arsizear(snar(1))
       cntalloc(snar(1)) = 1
       !       endif
       do jj = 2,nallocentr
          qloopmatch = .false.
          aa = snar(jj)
          if (arsizear(aa).eq.storeasize) then
             do kk = first,nnalloc  !looping over the new list
                orig = allocar(kk)  !this gives original line number
                call compdbentr(qmat=qmatch,qnofile=qnofilel, &
                     qnoproc=qnoprocl,qnoname=qnonamel,qnotype=qnotypel, &
                     qnorank=qnorankl,qnosize=.false.,qcaseins=qcaseinsl, &
                     a1=aa,a2=orig)
                if (qmatch) then
                   qloopmatch = .true.
                   cntalloc(orig) = cntalloc(orig) + 1
                endif
             enddo
          else if (arsizear(aa).gt.storeasize) then
             first = nnalloc + 1
          else
             write(6,*) 'sorted list out of order' 
          endif
          if(.not.qloopmatch) then
             nnalloc = nnalloc + 1
             allocar(nnalloc) = aa
             cntalloc(aa) = 1
          endif
          storeasize = arsizear(aa)
       enddo
    else !no allocations
       nnalloc = 0
    endif
    if(qmmverbl) then
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(20X,A40)') 'NON-REDUNDANT LIST OF ALLOCATIONS'
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6, &
            '(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4,1X,A3)') &
            'number','file','procedu','name','type','rank','size', &
            'kind','cnt'
    endif
    totcnt = 0
    do jj = 1,nnalloc
       aa = allocar(jj)
       cnt = cntalloc(aa)
       netmatalloc(aa) = cnt !initialize net allocation counts
       if(qmmverbl) then
          call prn_allocdpt('ALLNONR>',aa,filnamar(aa),procnamar(aa), &
               arnamar(aa),artypar(aa),arrankar(aa),arsizear(aa), &
               arkindar(aa),cnt)
          totcnt = totcnt + cnt
       endif
    enddo
    if(qmmverbl) write(6,'(20X,A12,I9)')  &
         'Array count:' ,totcnt
    !--------------------------------------------------------------------------
    !then for deallocations
    if(qmmverbl) then
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(22X,A42)') '(REDUNDANT) SORTED ARRAY DEALLOCATION LIST'
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6, &
            '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4)') &
            'number','file','procedu','name', &
            'type','rank','size','intended size','kind'
       do jj = 1,ndeallocentr
          dd = sdnar(jj)
          call prn_deallocdpt('DEALLRD>',dd,dfilnamar(dd),dprocnamar(dd), &
               darnamar(dd),dartypar(dd),darrankar(dd),darsizear(dd), &
               diarsizear(dd),darkindar(dd))
       enddo
       if(qmmverbl) write(6,'(20X,A12,I9)') &
            'Array count:' ,ndeallocentr
    endif
    !
    if(ndeallocentr.gt.0) then
       first = 1
       nndeall = 1 
       deallocar(nndeall) = sdnar(1) 
       storedsize = darsizear(sdnar(1))
       cntdealloc(sdnar(1)) = 1
       do jj = 2,ndeallocentr
          qloopmatch = .false.
          dd = sdnar(jj)
          if (darsizear(dd).eq.storedsize) then
             do kk = first,nndeall
                orig = deallocar(kk)
                call compdbentr(qmat=qmatch,qnofile=qnofilel, &
                     qnoproc=qnoprocl,qnoname=qnonamel,qnotype=qnotypel, &
                     qnorank=qnorankl,qnosize=.false.,qcaseins=qcaseinsl, &
                     d1=dd,d2=orig)
                if (qmatch) then
                   qloopmatch = .true.
                   cntdealloc(orig) = cntdealloc(orig) + 1
                endif
             enddo
          else if (darsizear(dd).gt.storedsize) then
             first = nndeall + 1
          else
             write(6,*) 'sorted list out of order' 
          endif
          if(.not.qloopmatch) then
             nndeall = nndeall + 1
             deallocar(nndeall) = dd
             cntdealloc(dd) = 1
          endif
          storedsize = darsizear(dd)
       enddo
    else !no deallocations 
       nndeall = 0
    endif
    !
    if(qmmverbl) then
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(18X,A42)') 'NON-REDUNDANT LIST OF DEALLOCATIONS'
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6, &
            '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4,1X,A3)') &
            'number','file','procedu','name', &
            'type','rank','size','intended size','kind','cnt'
    endif
    totcnt = 0
    do jj = 1,nndeall
       dd = deallocar(jj)
       cnt = cntdealloc(dd)
       netmatdeall(dd) = cnt  !intialize net deallocation counts
       if(qmmverbl) then
          call prn_deallocdpt('DEALCNR>',dd,dfilnamar(dd),dprocnamar(dd), &
               darnamar(dd),dartypar(dd),darrankar(dd),darsizear(dd), &
               diarsizear(dd),darkindar(dd),cnt)
       endif
       totcnt = totcnt + cnt
    enddo
    if(qmmverbl) write(6,'(20X,A12,I9)') &
         'Array count:' ,totcnt
    !-----------------------------------------------------------------
    ! compare the allocation and deallocation lists
    ! start main loop through the allocations
    !-----------------------------------------------------------------
    ! loop through the non-redundant allocation list and compare 
    ! with blocks of arrays having the same size in deallocation list
    !      write(6,*) 'WE KNOW NNALLOC is',NNALLOC
    !      write(6,*) 'WE KNOW NNDEALL is',NNDEALL
    if((nnalloc.gt.0).and.(nndeall.gt.0)) then
       storeaasz = -1 
       qover = .false.
       dd = deallocar(1)
       storeddsz = darsizear(dd)
       storefirst = 1
       first = 1  !first line of block in deallocations list
       jj = 1
       do ii = 1,nnalloc
          aa = allocar(ii) 
          aasize = arsizear(aa)  !size of allocd array
          qmatch=.false.
          ! check where we are in dealloc list
          ! if jj over upper bound, decrement
          if(jj.gt.nndeall) then
             qover = .true.
             jj = jj - 1
             first =  storefirst
          endif
          dd = deallocar(jj)
          ddsize = darsizear(dd)  !size of dealloc array
          ! decide where to start testing in dealloc list
          if((qmatchlst).or.(qover)) then
             if(aasize.eq.ddsize) then 
                jj = first
                dd = deallocar(jj) 
                ddsize = darsizear(dd)  !size of dealloc array
             endif
          else
             if(aasize.eq.storeddsz) then
                first = storefirst
                jj = first
                dd = deallocar(jj)
                ddsize = darsizear(dd)  !size of dealloc array
             endif
          endif
          ! loop over the rest of the block and check for matches
          do while((aasize.ge.ddsize).and.(.not.qmatch).and.  &
               (jj.le.nndeall)) 
             ! do testing for matches
             call compdbentr(qmat=qmatch,qnofile=qnofilel, &
                  qnoproc=qnoprocl,qnoname=qnonamel,qnotype=qnotypel, &
                  qnorank=qnorankl,qnosize=.false.,qcaseins=qcaseinsl, &
                  a1=aa,d1=dd)
             if (qmatch) then
                netmatalloc(aa) = netmatalloc(aa) - cntdealloc(dd)
                netmatdeall(dd) = netmatdeall(dd) - cntalloc(aa)
             endif
             !if no match, increment the line number
             if(.not.qmatch) then
                jj = jj + 1
                dd = deallocar(jj)
                if(darsizear(dd).gt.ddsize) then
                   storeddsz = ddsize
                   storefirst = first
                   first = jj
                endif
                ddsize = darsizear(dd)
             endif
          end do
          if(.not.qmatch) then
             qmatchlst = .false.
          else
             qmatchlst = .true.
          endif
          storeaasz = aasize
       enddo
    endif !if at least one alloc and one dealloc
    !-----------------------------------------------------------------
    ! end of main loop through allocations
    !-----------------------------------------------------------------
    !
    ncounta = 0
    ncountma = 0
    do ii = 1,nnalloc
       aa = allocar(ii)
       if(netmatalloc(aa).gt.0) then
          ncounta = ncounta + 1
          missmatchaar(ncounta) = aa  ! original database number
       endif
       if(netmatalloc(aa).lt.0) then
          ncountma = ncountma + 1
          multmatchaar(ncountma) = aa
       endif
    enddo
    !
    ncountd = 0
    ncountmd = 0
    do jj = 1,nndeall
       dd = deallocar(jj)
       if(netmatdeall(dd).gt.0) then
          ncountd = ncountd + 1
          write(6,*) 'jj ',jj,' countd ',ncountd, &
               ' settng missmatchdar to ',dd
          missmatchdar(ncountd) = dd !original database number 
       endif
       if(netmatdeall(dd).lt.0) then
          ncountmd = ncountmd + 1
          multmatchdar(ncountmd) = dd 
       endif
    enddo
    !
    write(6,'(13X,A62)') &
         '-------------SUMMARY of ALLOC/DEALLOC COMPARISON--------------'
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    if((qnosizel).or.(qnotypel)) then
       write(6,'(13X,A62)') &
            '***WARNING: total size and byte results are unreliable here***'
       write(6,'(13X,A61)') &
            '***if either array size or type are excluded from criteria***'
    endif
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    if(ncounta.gt.0) then
       write(6,'(12X,A63)') &
            'These allocatd arrys had missing matching explict deallocatns:'
       write(6, &
            '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4,1X,3A)') &
            'number','file','procedu','name','type','rank','size','kind', &
            'cnt'
       cntcnt = 0
       mallmissum = 0
       ssum = 0
       do ii = 1,ncounta
          aa = missmatchaar(ii)
          call prn_allocdpt('ALLOC-> ',aa,filnamar(aa),procnamar(aa), &
               arnamar(aa),artypar(aa),arrankar(aa),arsizear(aa), &
               arkindar(aa),netmatalloc(aa)) 
          mallmissum = mallmissum + arsizear(aa)*arkindar(aa)* &
               netmatalloc(aa)
          ssum = ssum + arsizear(aa)*netmatalloc(aa)
          cntcnt = cntcnt + netmatalloc(aa)
       enddo
       write(6,'(A56,7X,F15.0,2X,I5)') &
            'Total:',ssum,cntcnt
       write(6,'(4X,A57,2X,F15.0)') &
            'Estimated total bytes with no corresponding deallocations', &
            mallmissum
       !       write(6,*) 'qnosizel is ',qnosizel,' qnotypel is ',qnotypel
    else
       write(6,'(13X,A62)') &
            'No allocated arrys had missing matching explict deallocations.'
    endif
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    if(ncountd.gt.0) then
       write(6,'(12X,A63)') &
            'These deallocted arrys had missing matching explict allocatns:'
       write(6, &
            '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4,1X,A3)') &
            'number','file','procedu','name', &
            'type','rank','size','intended size','kind','cnt'
       cntcnt = 0
       mdeallmissum = 0
       ssum = 0
       do ii = 1,ncountd
          dd = missmatchdar(ii)
          call prn_deallocdpt('DEALC-> ',dd,dfilnamar(dd),dprocnamar(dd), &
               darnamar(dd),dartypar(dd),darrankar(dd),darsizear(dd), &
               diarsizear(dd),darkindar(dd),netmatdeall(dd))
          mdeallmissum = mdeallmissum + darsizear(dd)*darkindar(dd)* &
               netmatdeall(dd)
          ssum = ssum + darsizear(dd)*netmatdeall(dd)
          cntcnt = cntcnt +  netmatdeall(dd)
       enddo
       write(6,'(A56,7X,F15.0,2X,I5)') &
            'Total:',ssum,cntcnt
       write(6,'(6X,A55,2X,F15.0)') &
            'Estimated total bytes with no corresponding allocations:', &
            mdeallmissum
       !       write(6,*) 'qnosizel is ' ,qnosizel,' qnotypel is ',qnotypel
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
    endif
    !
    if(ncountma.gt.0) then
       write(6,'(12X,A63)') &
            'These allocatd arrays had extra matching explicit deallocatns:'
       write(6, &
            '(10X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,3X,A4,1X,A3)') &
            'number','file','procedu','name','type','rank','size','kind', &
            'cnt'
       do ii = 1,ncountma
          aa = multmatchaar(ii)
          call prn_allocdpt('ALLOC+> ',aa,filnamar(aa),procnamar(aa), &
               arnamar(aa),artypar(aa),arrankar(aa),arsizear(aa), &
               arkindar(aa),abs(netmatalloc(aa)))
       enddo
    endif
    if(ncountmd.gt.0) then
       write(6,'(5X,A70)') &
            '--------------------------------------------------------------'
       write(6,'(12X,A63)') &
            'These deallocted arrays had extra matching explicit allocatns:'
       write(6, &
            '(9X,A8,1X,A6,5X,A7,1X,A8,2X,A6,3X,A4,1X,A12,5X,A13,2X,A4,1X,A3)') &
            'number','file','procedu','name', &
            'type','rank','size','intended size','kind','cnt'
       do ii = 1,ncountmd
          dd = multmatchdar(ii)
          call prn_deallocdpt('DEALC+> ',dd,dfilnamar(dd),dprocnamar(dd), &
               darnamar(dd),dartypar(dd),darrankar(dd),darsizear(dd), &
               diarsizear(dd),darkindar(dd),abs(netmatdeall(dd)))
       enddo
    else
       write(6,'(13X,A62)') &
            'No deallocated arrays had extra matching explicit allocations.'
    endif
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    ! calculate memory in failed allocations/deallocations
    fmallocsum = 0
    fallwrdsum = 0
    do ii = 1,nfallocentr
       fmallocsum = fmallocsum + farsizear(ii)*farkindar(ii) 
       fallwrdsum = fallwrdsum +farsizear(ii)
    enddo
    ! these are only non-zero when array to be deallocated has been
    ! allocated
    fmdeallsum = 0
    fdeallwrdsum = 0
    do ii = 1,nfdeallocentr
       fmdeallsum = fmdeallsum + fdarsizear(ii)*fdarkindar(ii)
       fdeallwrdsum = fdeallwrdsum + fdarsizear(ii)
    enddo
    write(6,'(5X,A70)') &
         '--------------------------------------------------------------'
    write(6,'(10X,A29)') 'Calls to chmalloc/chmdealloc:'
    write(6,'(40X,A6,10X,A5,6X,A16)') 'Number','Words', &
         'Est Mem (bytes)'
    write(6,'(10X,A22,5X,I12,1X,F15.0,1X,F15.0)')  &
         'SUCCESSFUL ALLOCATIONS', &
         nallocentr,allwrdsum,mallocsum
    write(6,'(10X,A24,3X,I12,1X,F15.0,1X,F15.0)')  &
         'SUCCESSFUL DEALLOCATIONS',ndeallocentr,deallwrdsum,mdeallsum
    write(6,'(10X,A18,9X,I12,1X,F15.0,1X,F15.0)')  &
         'FAILED ALLOCATIONS', &
         nfallocentr,fallwrdsum,fmallocsum
    write(6,'(10X,A20,7X,I12,1X,F15.0,1X,F15.0)')  &
         'FAILED DEALLOCATIONS', &
         nfdeallocentr,fdeallwrdsum,fmdeallsum
    write(6,'(10X,A5,22X,I12)') 'Total',nallocentr+ &
         ndeallocentr+nfallocentr+nfdeallocentr
    !
    write(6,'(13X,A51)')  &
         'Estimates of memory in bytes = sum(kind*size)'

    write(6,'(13X,A62)') &
         '***************END of ALLOC/DEALLOC COMPARISON****************'
    write(6,*) ' '
    deallocate(rarsizcp,rdarsizcp,snar,sdnar, &
         missmatchaar,missmatchdar,multmatchaar,multmatchdar, &
         allocar,deallocar,cntalloc,cntdealloc, &
         netmatalloc,netmatdeall)
    !
  end SUBROUTINE compalloc
  !______________________________________________________________________ 
  !______________________________________________________________________
  SUBROUTINE compdbentr(qmat,qnofile,qnoproc,qnoname, &
       qnotype,qnorank,qnosize,qcaseins,a1,a2,d1,d2)
    !
    ! This routine compares two individual database entries
    ! (datapoints).
    !
    ! Not sure if qnotype, qnorank, qnosize will ever be useful, but have
    ! left them in for now.
    !
    !                                   --RJP Nov 2008
  use stringutil
  use allocdat
  use deallocdat
    !
    logical,intent(out) :: qmat  !returned value true if match
    logical,optional,intent(in) :: qnofile,qnoproc,qnoname, &
         qnotype,qnorank,qnosize,qcaseins
    integer,optional,intent(in) :: a1,a2,d1,d2
    !
    character(len=8),dimension(4) :: file,proc,name,eltype
    integer,dimension(4) :: rank,size
    integer :: fir,sec
    logical :: qfileok,qprocok,qnameok
    !***************************************************************
    !      qcaseins = .false.  
    !
    ! check to see what to compare
    !
    fir = 0
    sec = 0
    if(present(a1)) then
       file(1) = filnamar(a1)
       proc(1) = procnamar(a1)
       name(1) = arnamar(a1)
       eltype(1) = artypar(a1)
       rank(1) = arrankar(a1)
       size(1) = arsizear(a1)
       fir = 1
       !        write(6,*) 'size of a1 is ',size(1)
    endif
    if(present(a2)) then
       file(2) = filnamar(a2)
       proc(2) = procnamar(a2)
       name(2) = arnamar(a2)
       eltype(2) = artypar(a2)
       rank(2) = arrankar(a2)
       size(2) = arsizear(a2)
       if(fir.eq.1) then
          sec = 2
       else 
          fir = 2
       endif
    endif
    if(present(d1)) then
       file(3) = dfilnamar(d1)
       proc(3) = dprocnamar(d1)
       name(3) = darnamar(d1)
       eltype(3) = dartypar(d1)
       rank(3) = darrankar(d1)
       size(3) = darsizear(d1)
       if(fir.ge.1) then
          sec = 3
       else
          fir = 3
       endif
    endif
    if(present(d2)) then
       file(4) = dfilnamar(d2)
       proc(4) = dprocnamar(d2)
       name(4) = darnamar(d2)
       eltype(4) = dartypar(d2)
       rank(4) = darrankar(d2)
       size(4) = darsizear(d2)
       if(fir.ge.1) then
          sec = 4
       else
          fir = 4
       endif
    endif
    ! do actual testing
    qmat = .false.
    ! file name comparison
    qfileok = .true.
    if(.not.qnofile) then
       if (.not.qcaseins) then
          qfileok = file(fir).eq.file(sec)
       else
          qfileok = cmpstr_ins(file(fir),file(sec))
       endif
    endif
    ! procedure name comparison
    qprocok = .true.
    if(.not.qnoproc) then
       if (.not.qcaseins) then
          qprocok = proc(fir).eq.proc(sec)
       else 
          qprocok = cmpstr_ins(proc(fir),proc(sec))
       endif
    endif
    !      write(6,*) 'procf is ',proc(fir)
    !      write(6,*) 'procs is ',proc(sec),' qprocok is ',qprocok
    !
    ! array name comparison
    qnameok = .true.
    if(.not.qnoname) then
       if (.not.qcaseins) then
          qnameok = name(fir).eq.name(sec)
       else 
          qnameok = cmpstr_ins(name(fir),name(sec))
       endif
    endif

    !      write(6,*) 'namef is ',name(fir)
    !      write(6,*) 'names is ',name(sec),' qnameok is ',qnameok

    if (qfileok) then
       if (qprocok) then
          if (qnameok) then
             if ((eltype(fir).eq.eltype(sec)).or.qnotype) then
                if ((rank(fir).eq.rank(sec)).or.qnorank) then
                   if ((size(fir).eq.size(sec)).or.qnosize) then
                      qmat = .true.
                   endif
                endif
             endif
          endif
       endif
    endif
    !
  END SUBROUTINE compdbentr
  !______________________________________________________________________
  !______________________________________________________________________
  SUBROUTINE qsort(a,n,t)
  use chm_kinds

    !     NON-RECURSIVE STACK VERSION OF QUICKSORT FROM N.WIRTH'S PASCAL
    !     BOOK, 'ALGORITHMS + DATA STRUCTURES = PROGRAMS'.

    !     SINGLE PRECISION, ALSO CHANGES THE ORDER OF THE ASSOCIATED ARRAY T.

    IMPLICIT NONE

    INTEGER, INTENT(IN)    :: n
    real(kind=chm_real) :: a(n)
    INTEGER, INTENT(INOUT) :: t(n)

    !     Local Variables

    INTEGER                :: i, j, k, l, r, s, stackl(15), stackr(15), ww
    real(kind=chm_real)    :: w, x

    s = 1
    stackl(1) = 1
    stackr(1) = n

    !     KEEP TAKING THE TOP REQUEST FROM THE STACK UNTIL S = 0.

10  CONTINUE
    l = stackl(s)
    r = stackr(s)
    s = s - 1

    !     KEEP SPLITTING A(L), ... , A(R) UNTIL L >= R.

20  CONTINUE
    i = l
    j = r
    k = (l+r) / 2
    x = a(k)

    !     REPEAT UNTIL I > J.

    DO
       DO
          IF (a(i).LT.x) THEN                ! Search from lower end
             i = i + 1
             CYCLE
          ELSE
             EXIT
          END IF
       END DO

       DO
          IF (x.LT.a(j)) THEN                ! Search from upper end
             j = j - 1
             CYCLE
          ELSE
             EXIT
          END IF
       END DO

       IF (i.LE.j) THEN                     ! Swap positions i & j
          w = a(i)
          ww = t(i)
          a(i) = a(j)
          t(i) = t(j)
          a(j) = w
          t(j) = ww
          i = i + 1
          j = j - 1
          IF (i.GT.j) EXIT
       ELSE
          EXIT
       END IF
    END DO

    IF (j-l.GE.r-i) THEN
       IF (l.LT.j) THEN
          s = s + 1
          stackl(s) = l
          stackr(s) = j
       END IF
       l = i
    ELSE
       IF (i.LT.r) THEN
          s = s + 1
          stackl(s) = i
          stackr(s) = r
       END IF
       r = j
    END IF

    IF (l.LT.r) GO TO 20
    IF (s.NE.0) GO TO 10
    RETURN
  END SUBROUTINE qsort
  !______________________________________________________________________ 
  !______________________________________________________________________
END MODULE alloccomp

