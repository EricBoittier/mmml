module edsmod
  use chm_kinds
  use chm_types
  implicit none

#if KEY_EDS==1

   type :: eds_subs
      character(len=20)  :: mskey
      real(chm_real)     :: eoffset
   end type

   logical,save,public                            :: qeds,qedsdynoff
   real(chm_real),save,public                     :: edstemp
   integer, save, public                          :: neds
   type(eds_subs),allocatable,save,dimension(:)   :: edslist

contains

   subroutine eds_init()
      qeds = .false.
      qedsdynoff=.false.
   end subroutine eds_init

   subroutine eds_cleanup()
     if(allocated(edslist)) &
        deallocate(edslist)
   end subroutine eds_cleanup

   subroutine process_eds(comlyn,comlen)
      use stream
      use string
      use memory
      implicit none

      character(len=*) :: comlyn
      integer          :: comlen,i

      edstemp = gtrmf(comlyn,comlen,'TEMP',-1.0)
      neds    = gtrmi(comlyn,comlen,'NEDS',-1)
      qedsdynoff = (indxa(comlyn,comlen,'DYNO').gt.0)

      if(qedsdynoff.and.prnlev.gt.4) &
         write(outu,'(a)') 'PROCESS_EDS> DYNAMIC ENERGY PARTITIONING ACTIVATED'

      if(neds.le.0) &
         call wrndie(-4,'<PROCESS_EDS>','At least one EDS subsystem is needed')

      if(allocated(edslist)) then
         call wrndie(-1,'<PROCESS_EDS>','EDS list already specified; resetting')
         deallocate(edslist)
      endif

      ! I would use chmalloc for this, but I don't think it supports
      ! user-defined types
      allocate(edslist(neds))
      do i=1,neds
         edslist(i)%mskey = gtrma(comlyn,comlen,'TERM')
         if(edslist(i)%mskey.eq.'    ') &
            call wrndie(-4,'<PROCESS_EDS>','Invalid subsystem key name')
         edslist(i)%eoffset = nextf(comlyn,comlen)

         if(prnlev.gt.5) then
            write(outu,'(a,i3,a,a,a,f12.4)') 'EDS DEBUG> ENDPOINT ',i,' KEY NAME ',edslist(i)%mskey, &
                                             ' OFFSET ',edslist(i)%eoffset
         endif
      enddo
      qeds = .true.

   end subroutine process_eds

#endif

end module edsmod

