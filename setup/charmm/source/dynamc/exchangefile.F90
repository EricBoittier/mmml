! handle high efficiency exchange files
! Tim Miller -- December, 2012

! Note, I make an assumptiuon that we only deal with one exchange file
! at a time. If we try to open two exchange files up, things will get trashed
! with undefined results. YOU HAVE BEEN WARNED!
module exchangefile
   use chm_kinds
   implicit none


   ! Note, the number of exchanges do not include
   ! repeats, for example if ytou exchange at 10,000
   ! different points (time steps) in your simulation,
   ! but you do iterative exchange 5 times, the
   ! number of exchanges is still 10,000. In this case
   ! there are 10,000. If we ran 1,000,000 steps exchanging
   ! every 100 steps, order(*,1) is valid for steps 1-100,
   ! ... , order(*,10000) is valid for step 999,901-1,000,000.

   ! The orders themselves are a permutation that maps from 
   ! replica space to temperature space, e.g. if order(*,N) =
   ! (3,4,1,2), it means that the lowest temperature is in
   ! replica 3, the second lowest in replica 4, and the highest
   ! in replica 2.
   type exfile_typ
      integer                                 :: nrep,nexch,rexch,exchfreq,nrepeat
      real(chm_real),allocatable,dimension(:) :: temps
      integer,allocatable,dimension(:,:)      :: orders
      logical                                 :: is_active
   end type exfile_typ

   type(exfile_typ),save  :: ef

contains

   subroutine map_totempspace(istep,iresnum,temprep)
      use stream

      integer, intent(in)  :: istep,iresnum
      integer, intent(out) :: temprep

      integer              :: critnum

      critnum=(istep-1)/ef%exchfreq
      if(critnum.gt.ef%rexch) &
         call wrndie(-3,'<MAP_TOTEMPSPACE>','BOUNDS EXCEEDED')

      if(critnum.eq.0) then
         temprep=iresnum
      else
         temprep=ef%orders(iresnum,critnum)
      endif

      if(prnlev.gt.6) &
         write(outu,'(a,i9,a,i3a,i3,a,i3)') &
         'MAP_TOTEMPSPACE> ISTEP ',istep,' CRITNUM ',critnum,' MAP REP ',iresnum,' TO TEMP ',temprep

   end subroutine map_totempspace

   subroutine readhdr(line,linelen,ixnum,istep,irepe)
      use string
      use stream

      character(len=100),intent(inout) :: line
      integer,intent(inout)            :: linelen
      integer,intent(out)              :: ixnum,istep,irepe

      integer               :: wrdlen
      character(len=100)    :: junk


      call nextwd(line,linelen,junk,100,wrdlen)
      if(junk /= '#') call wrndie(-5,'<READHDR>','PARSE ERROR')
      call nextwd(line,linelen,junk,100,wrdlen)
      if(junk /= 'Exchange') call wrndie(-5,'<READHDR>','PARSE ERROR')
      call nextwd(line,linelen,junk,100,wrdlen)
      ixnum=nexti(junk(1:wrdlen-1),wrdlen-1)

      call nextwd(line,linelen,junk,100,wrdlen)
      if(junk /= 'STEP') call wrndie(-5,'<READHDR>','PARSE ERROR')
      call nextwd(line,linelen,junk,100,wrdlen)
      istep=nexti(junk(1:wrdlen-1),wrdlen-1)

      call nextwd(line,linelen,junk,100,wrdlen)
      if(junk /= 'REPEAT') call wrndie(-5,'<READHDR>','PARSE ERROR')
      call nextwd(line,linelen,junk,100,wrdlen)
      irepe=nexti(junk,wrdlen)
   end subroutine readhdr

   subroutine read_exchange(unum,qinit)
      use stream

      integer,parameter      :: maxln = 100
      integer,intent(in)     :: unum
      logical,intent(in)     :: qinit

      integer             :: ixnum,istep,irepe,i,j
      integer             :: replica,neighbor,result,linelen
      real(chm_real)      :: reptemp,repepot,nbrtemp,nbrepot,prob,p
      real(chm_real)      :: newtemp
      logical             :: qexch,qfind
      integer,save        :: lastst=0,pos=1
      character(len=maxln)  :: line

      if(ef%rexch.gt.ef%nexch) then
         call wrndie(-1,'<READ_EXCHANGE>','TRYING TO READ MORE THAN NREP EXCHANGES')
         return
      endif

      if(qinit) then 
         read(unum,*)
         read(unum,'(a)') line
         linelen=len(line)

         write(outu,*) 'READ_EXCHANGE> GOT ', LINE
         call flush(outu)
         call readhdr(line,linelen,ixnum,istep,irepe)

         ef%exchfreq = istep
         lastst = 0

         ! get all of the temperatures
         do i=1,ef%nrep 
            read(unum,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L,x,I2)') &
                 replica,reptemp,repepot,neighbor,nbrtemp,nbrepot,prob,p,qexch,result
            ef%temps(i)=reptemp ! temps in order at step 1
         enddo
         
         rewind(unum)
         read(unum,*)
      endif

      ! Loop over any repeats that don't matter
      do
         read(unum,'(a)') line
         linelen=len(line)
         write(outu,*) 'READ_EXCHANGE> GOT ', LINE
         call flush(outu)
         call readhdr(line,linelen,ixnum,istep,irepe)

         if(irepe.eq.ef%nrepeat) exit
         do i=1,ef%nrep
            read(unum,*)
         enddo
         ef%rexch=ef%rexch+1
      enddo

      if(istep.ne.lastst+ef%exchfreq) then
         call wrndie(0,'<READ_EXCHANGE>','CHANGING EXCHANGE FREQUENCIES!')
      endif
      lastst = istep

      if(prnlev.gt.6) write(outu,'(a,i7)') '--- EXCHANGE ', pos
      do i=1,ef%nrep
         read(unum,'(I2,x,F12.6,x,F15.6,x,I2,x,F12.6,x,F15.6,x,F5.3,x,F5.3,x,L,x,I2)') &
              replica,reptemp,repepot,neighbor,nbrtemp,nbrepot,prob,p,qexch,result

         if(qexch) then
            newtemp=nbrtemp
         else
            newtemp=reptemp
         endif

         qfind=.false.
         do j=1,ef%nrep
            if(ef%temps(j).eq.newtemp) then
               ef%orders(replica,pos)=j
               qfind=.true.
               exit
            endif
         enddo
         if(.not.qfind) &
            call wrndie(-4,'<READ_EXCHANGE>','UNKNOWN TEMPERATURE ENCOUNTERED')
         if(prnlev.gt.6) write(outu,'(a,i3,a,i3)') 'rep order ', replica, ' --temp-order--> ', ef%orders(replica,pos)
      enddo
      ef%rexch=ef%rexch+1
      pos=pos+1

   end subroutine read_exchange

   subroutine delete_exfile()
      use memory

      if(ef%is_active) then
         call chmdealloc('exchangefile.src','destroy_exfile','ef%temps',ef%nrep,crl=ef%temps)
         deallocate(ef%orders)

         ef%nrep = 0
         ef%nexch = 0
         ef%rexch = 0
         ef%is_active = .FALSE.
      endif
   end subroutine delete_exfile

   subroutine create_exfile(eunit,nexch,nrep,nrepeat)
      use stream
      use memory

      integer,intent(in)  :: eunit,nexch,nrep,nrepeat
      integer             :: i

      if(ef%is_active) call wrndie(-5,'<CREATE_EXFILE>','EXFILE ALREADY ACTIVE')

      ef%nrep = nrep
      ef%nexch = nexch
      ef%rexch = 0
      ef%nrepeat = nrepeat

      allocate(ef%orders(nrep,nexch))
      call chmalloc('exchangefile.src','create_exfile','ef%temps',ef%nrep,crl=ef%temps)

      rewind(eunit)
      call read_exchange(eunit,.true.)

      do i=2,nexch/nrepeat
         call read_exchange(eunit,.false.)
      enddo

   end subroutine create_exfile

end module exchangefile

