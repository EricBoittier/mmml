module epmf
#if KEY_EPMF==1 /*epmf_main*/
!
! See doc/epmfmodule.doc for usage information
!
! Srinivasa Murthy gopal & Michael Feig, 2008,2010
!            Michigan State University(MSU) 
!
   use chm_kinds
   use chm_types
   use cmapm,only : cmapspl,cmapspi
   implicit none

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!
!   2012 MARCH
!   hbpmf has been modified
!   pmfn now is i+/-N>=6 and new pmfs pmf4 and pmf5 are defined
!   
!   new datastructures:
!      hbnei4,hbnei5  
!   modified
!      hbpmf,hbscale,loghb
!!
!   new variables
!
!
!  
!   Most of charmm COMMON (global) variables are typed in UPPER CASE
!   while epmf variables are in lower case. Same holds also for
!   external subroutines called here.
!
!   List of some important variables/datastructures used by epmf module
! 
!    ===================================================================
!    Variable               Purpose                             Default  
!                     explanation and usage syntax example    (MAX VALUE)   
!    ===================================================================
!
!    qinit =      Initialize datastructures.                   .true.
!    qepmf =      Initiate epmf calculations.                  .false.
!    hbpmf =      Datastructure for storing hbpmf data.       
!                 i:pmfno. j:pmftype  k:angle, l:distance dimensions
!                   e=hbpmf(i,j)%map2d%a(k,l,1)
!                   de/d(angl)=hbpmf(i,j)%map2d%a(k,l,2)
!                   de/d(dist)=hbpmf(i,j)%map2d%a(k,l,3)
!                   de/d(angl)(dist)=hbpmf(i,j)%map2d%a(k,l,4)  
!                   angl grid = hbpmf(i,j)%binx
!                   dist grid = hbpmf(i,j)%biny
!
!    dtpmf        Datastructure for storing 1d epmf data. 
!                    i:pmfno. j:pmftype  k:dtpmf grid point(1..npts)
!                    x=dtpmf(i,j)%map1d%a(k,1)          
!                    y=dtpmf(i,j)%map1d%(k,2)
!                    d/dx2=dtpmf(i,j)%map1d%(k,3)
!                    min=dtpmf(i,j)%minx
!                    max=dtpmf(i,j)%maxx
!                    binsize=dtpmf(i,j)%binx
!                    num of data points=dtpmf(i,j)%map1d%len1            
!        
!
!    dtdnrs/       Stores the donors/acceptors
!    dtacps        jth donr of ith pmf: dtdnrs(i)%a(j)
!                  no of ith pmf donrs: dtdnrs(i)%len
!
!    dtnei0        neighbor list of different pmfs
!    dtnei1        x=0,1,...p is the pmf type
!    dtnei2                    
!    dtnei3                          
!    dtnein                   pmfno
!    dtneim                      ^
!    dtneip                      |  
!                    dtnei0(i)%list(p)%a(i)
!                                   |    |
!                                   v    v
!                                  dono  acceptor
!
!
!
!    hbdnrs
!    hbacps       Analagous to the above definations
!    hbnei*   
!
!
!    ndtpmf       No. of 1d distance pmfs                   0 (MAXDTPMF)
!    nhbpmf       No. of hb pmfs                            0 (MAXDTPMF)
!
!    ehb          hbpmf energies.       ehb(i)
!    edt          dtpmf energies.       edt(i)
!       
!    loghb        find which hbpmf is defined.  
!                 loghb(i,j) i: pmfno. j: 0,1,2,3,N th neighbors of ith res
!    logdt        find which dtpmf is defined. 
!                 logdt(i,j) j: 0,+/-1,+/-2,+/-3,N,-1,+1 
!
!
!    upfrn        update frequency for nth neigh list             25
!                 upfrn is same for all pmfs and is read only
!                 once(first entry in stream file)
!
!    hbcutf       cutoff distance for     neighbor for hbpmf      CUTNB
!    dtcutf       same as that of above, but for dtpmf            CUTNB  
!
!    fa,fb,blen   parameters for constructing hb atom on the donor
!                 fa(i)  fb(i)  blen(i)
!  
!    hbscale      scaling factor for hbpmf
!                 hbscale(i,j)  i:hbpmf no. j: 0,1,2,3,N pmf
!
!    ehbmin       Energy below which hbpmf calculations           minehb
!                 are turned off
!
!    dtscale      scaling factor for dtpmf
!                 dtscale(i,j)  i:dtpmf no. j:  0,+/-1,+/-2,+/-3,N,-1,+1
!
!    hbtyp        type of construction of h atom
!                 hbtyp(i)='defa'/'geom'
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     integer, parameter :: maxhbpmf=50,maxdtpmf=250
     real(chm_real),parameter :: minehb=-10.0


     integer, save :: nhbpmf=0,ndtpmf=0
     integer, save :: upfrn
     logical, save :: qinit=.true., qepmf=.false., qmesg=.false.

     logical,allocatable,dimension(:,:),save :: loghb,logdt

     real(chm_real),allocatable,dimension(:),save :: ehb,edt,hbcutf,&
                dtcutf,fa,fb,blen,ehbmin
     real(chm_real),allocatable,dimension(:,:),save :: hbscale,dtscale 

     character(len=4),allocatable,dimension(:),save :: hbtyp
     character(len=8),allocatable,dimension(:),save :: cha1,cha2


     type dtmap
         type(chm_ptr_2d) :: map1d
         real(chm_real) :: minx,maxx,binx
     end type dtmap


     type ptr3d
          real(chm_real),pointer,dimension(:,:,:) :: a => null()
          integer :: len1=0,len2=0,len3=0
     end type ptr3d


     type hbmap
          type(ptr3d) :: map2d
          real(chm_real) :: minx,maxx,miny,maxy,binx,biny
     end type hbmap


     type alist
         type(chm_iptr),allocatable,dimension(:) :: list
     end type alist


     type(dtmap),allocatable,dimension(:,:),save   :: dtpmf
     type(chm_iptr),allocatable,dimension(:),save  :: dtdnrs,dtacps
     type(alist),allocatable,dimension(:),save:: dtnei0,dtnei1, &
         dtnei2,dtnei3,dtnein,dtneim,dtneip


     type(hbmap),allocatable,dimension(:,:),save   :: hbpmf
     type(chm_iptr),allocatable,dimension(:),save  :: hbdnrs,hbacps,&
                    hbatm1,hbatm2
     type(alist),allocatable,dimension(:),save:: hbnei0,hbnei1, &
         hbnei2,hbnei3,hbnei4,hbnei5,hbnein
     
     contains

     subroutine epmf_set(COMLYN,COMLEN)
!  use select
  use dimens_fcm
  use psf
  use stream
  use number
  use param
  use string
  use econtmod
  use contrl
  use inbnd
#if KEY_PARALLEL==1
  use parallel
#endif 

      character(len=*),intent(inout) :: COMLYN
      integer,intent(inout) :: COMLEN

!     local vars
      integer :: leng,ii,dummy
      character(len=4) :: comnd
      real(chm_real) :: sc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      if(qinit) then
         qinit = .false.
         qepmf = .true.
         call epmf_init()
      endif

#if KEY_PARALLEL==1
       processparall: if(MYNODP .eq. 1)then
         if((ndtpmf .eq. 0) .and. (nhbpmf .eq. 0)) then
         write(OUTU,'(/,a40,/)')'*******************************'
         write(OUTU,'(a40,i8)')'EPMF> Number of nodes:',NUMNOD
         write(OUTU,'(a40,i8)')'EPMF> Processing input on node:',MYNODP 
         write(OUTU,'(a40,i8)')'EPMF> Calculations on node:',MYNODP
         write(OUTU,'(/,a40,/)')'*******************************'   
         endif
#endif 

      comnd=NEXTA4(COMLYN,COMLEN)
      if(comnd .eq. 'DIST') then

!     read update frequency only once ie first call of epmf
         if(ndtpmf .eq. 0 .and. nhbpmf .eq. 0) then
           upfrn=GTRMI(COMLYN,COMLEN,'UPFR',25)
         endif

!        Ignore UPFR when they were read before          
         dummy=GTRMI(COMLYN,COMLEN,'UPFR',25)

         ndtpmf=ndtpmf+1
         if(ndtpmf .gt. maxdtpmf) then
           call WRNDIE(-5,'epmf:epmf_set',&
                'Maximum no. of DIST pmfs exceeded')
         endif

         dtcutf(ndtpmf)=GTRMF(COMLYN,COMLEN,'CUTF',CUTNB)
!        set the scaling factor for each dtpmf 
         dtscale(ndtpmf,1)=GTRMF(COMLYN,COMLEN,'SCL0',ONE)
         dtscale(ndtpmf,2)=GTRMF(COMLYN,COMLEN,'SCL1',ONE)
         dtscale(ndtpmf,3)=GTRMF(COMLYN,COMLEN,'SCL2',ONE)
         dtscale(ndtpmf,4)=GTRMF(COMLYN,COMLEN,'SCL3',ONE)
         dtscale(ndtpmf,5)=GTRMF(COMLYN,COMLEN,'SCLN',ONE)
         dtscale(ndtpmf,6)=GTRMF(COMLYN,COMLEN,'SCLM',ONE)
         dtscale(ndtpmf,7)=GTRMF(COMLYN,COMLEN,'SCLP',ONE)

         if(PRNLEV .gt. 5) then
          write(OUTU,'(/,a60,/)')"*** Scaling factors for epmf(dt) ***"
          write(OUTU,'(a15,1x,i2,1x,a25,1x,7f8.3)') & 
         'dtpmf number:',ndtpmf,'uses scaling factors:',&
          dtscale(ndtpmf,1),dtscale(ndtpmf,2),dtscale(ndtpmf,3),&
          dtscale(ndtpmf,4),dtscale(ndtpmf,5),dtscale(ndtpmf,6),&
          dtscale(ndtpmf,7)
          write(OUTU,'(/,a60,/)')"*** cutoff dist. for dt neigh list***"
          write(OUTU,'(a15,1x,i4,f8.3,a10)') &
              'dtpmf number:',ndtpmf,dtcutf(ndtpmf),'angstrom'
         endif

         call tabdtpmf(COMLYN,COMLEN,ndtpmf)
         call selcdtpmf(COMLYN,COMLEN,ndtpmf)

      elseif(comnd .eq. 'HBON') then

         if(ndtpmf .eq. 0 .and. nhbpmf .eq. 0) then
           upfrn=GTRMI(COMLYN,COMLEN,'UPFR',25)
         endif
         dummy=GTRMI(COMLYN,COMLEN,'UPFR',25)

!        Select type of build h type          
         nhbpmf=nhbpmf+1
         if(nhbpmf .gt. maxhbpmf) then
           call WRNDIE(-5,'epmf:epmf_set',&
                'Maximum no. of HB pmfs exceeded')
         endif

         if(INDXA(COMLYN,COMLEN,'GEOM') .GT. 0) then
           hbtyp(nhbpmf)='GEOM'
         elseif(INDXA(COMLYN,COMLEN,'DEFA') .GT. 0) then
           hbtyp(nhbpmf)='DEFA'
         else
           hbtyp(nhbpmf)='DEFA'
         endif

         if(PRNLEV .gt. 5) then
           write(OUTU,'(/,a40,i4,a3,a5/)')&
              'EPMF> hbuild type for',nhbpmf,' : ',hbtyp(nhbpmf)
         endif

!        cutf for each hbpmf
         hbcutf(nhbpmf)=GTRMF(COMLYN,COMLEN,'CUTF',CUTNB)
         if(PRNLEV .gt. 5) then
            write(OUTU,'(/,a60,/)') &
            "*** cutoff dist. for dist neigh list***"
            write(OUTU,'(a15,1x,i4,f8.3,a10)') &
            'hbpmf number:',nhbpmf,hbcutf(nhbpmf),'angstrom'
         endif

!        set the scaling factor for each hbpmf    
         hbscale(nhbpmf,1)=GTRMF(COMLYN,COMLEN,'SCL0',ONE)
         hbscale(nhbpmf,2)=GTRMF(COMLYN,COMLEN,'SCL1',ONE)
         hbscale(nhbpmf,3)=GTRMF(COMLYN,COMLEN,'SCL2',ONE)
         hbscale(nhbpmf,4)=GTRMF(COMLYN,COMLEN,'SCL3',ONE)
         hbscale(nhbpmf,5)=GTRMF(COMLYN,COMLEN,'SCL4',ONE)
         hbscale(nhbpmf,6)=GTRMF(COMLYN,COMLEN,'SCL5',ONE)
         hbscale(nhbpmf,7)=GTRMF(COMLYN,COMLEN,'SCLN',ONE)

!        minimum energy for hbpmf
         ehbmin(nhbpmf)=GTRMF(COMLYN,COMLEN,'EMIN',MINEHB)

         if(PRNLEV .gt. 5) then
          write(OUTU,'(/,a60,/)')"*** Scaling factors for epmf(hb) ***"
          write(OUTU,'(a15,1x,i2,1x,a25,1x,7f8.3)') &
               'hbpmf number:',nhbpmf,'uses scaling factors:', &
          hbscale(nhbpmf,1),hbscale(nhbpmf,2),hbscale(nhbpmf,3), &
          hbscale(nhbpmf,4),hbscale(nhbpmf,5),hbscale(nhbpmf,6),&
          hbscale(nhbpmf,7)
          write(OUTU,'(/,a60,/)')"*** cutoff dist. for hb neigh list***"
          write(OUTU,'(a15,1x,i4,f8.3,a10)') &
               'hbpmf number:',nhbpmf,hbcutf(nhbpmf),'angstrom'
          write(OUTU,'(/,a60,/)')"*** min allowed interaction energy***"
          write(OUTU,'(a15,1x,i4,f8.3,a30)') &
         'hbpmf number:',nhbpmf,ehbmin(nhbpmf),' kcal/mol per residue'
        endif

         cha1(nhbpmf)='HAL'
         cha2(nhbpmf)='HAL'
         call GTRMWD(COMLYN,COMLEN,'ATM1',4,cha1(nhbpmf),8,leng)
         call GTRMWD(COMLYN,COMLEN,'ATM2',4,cha2(nhbpmf),8,leng)
         fa(nhbpmf)=GTRMF(COMLYN,COMLEN,'F1',MINSIX)
         fb(nhbpmf)=GTRMF(COMLYN,COMLEN,'F2',MINSIX)
         blen(nhbpmf)=GTRMF(COMLYN,COMLEN,'BLEN',MINSIX)

         if(fa(nhbpmf) .eq. MINSIX .or. fb(nhbpmf) .eq. MINSIX .or. &
             blen(nhbpmf) .eq. MINSIX .or. cha1(nhbpmf) .eq. 'HAL' &
             .or. cha2(nhbpmf) .eq. 'HAL') then

             write(OUTU,'(/,a,i8)')"EPMF> error in hbond",nhbpmf
             if(fa(nhbpmf) .eq. MINSIX) then
                write(OUTU,'(/,a40)')'EPMF> Missing parameter F1'
             elseif(fb(nhbpmf) .eq. MINSIX) then
                write(OUTU,'(/,a40)')'EPMF> Missing parameter F2'
             elseif(blen(nhbpmf) .eq. MINSIX) then
                write(OUTU,'(/,a40)')'EPMF> Missing parameter BLEN'
             elseif(cha1(nhbpmf) .eq. 'HAL') then
                write(OUTU,'(/,a40)')'EPMF> Missing parameter ATM1'
             else
                write(OUTU,'(/,a40)')'EPMG> Missing parameter ATM2'
             endif
             call WRNDIE(-5,'epmf_set()',&
                  'Could not read necessary parameters')
         endif
!   check any errors in parameters       
        if(hbtyp(nhbpmf) .eq. 'GEOM') then
          if(fa(nhbpmf) .lt. 0 .or. fa(nhbpmf) .gt. 180) then
           write(OUTU,'(/,a,i8/)')"EPMF> error in hbond",nhbpmf
           write(OUTU,'(/,a,f8.3/)')"given fa=",fa(nhbpmf)
           call WRNDIE(-5,'epmf_set()', &
               'wrong fa values. valid fa values: 0<=fa<=180')
          endif
          if(fb(nhbpmf) .lt. -180 .or. fb(nhbpmf) .gt. 180) then
           write(OUTU,'(/,a,i8/)')"EPMF> error in hbond",nhbpmf
           write(OUTU,'(/,a,f8.3/)')"given fb=",fb(nhbpmf)
           call WRNDIE(-5,'epmf_set()', &
             'wrong fa values. valid fa values: -180<=fb<=180')
          endif
        else
          if(fa(nhbpmf) .lt. -1.0 .or. fa(nhbpmf) .gt. 1.0) then
           write(OUTU,'(/,a,i8/)')"EPMF> error in hbond",nhbpmf
           write(OUTU,'(/,a,f8.3/)')"given fa=",fa(nhbpmf)
           call WRNDIE(-5,'epmf_set()', &
              'wrong fa values. valid fa values: -1.0<=fa<=1.0')
          endif
          if(fb(nhbpmf) .lt. -1.0 .or. fb(nhbpmf) .gt. 1.0) then
           write(OUTU,'(/,a,i8/)')"EPMF> error in hbond",nhbpmf
           write(OUTU,'(/,a,f8.3/)')"given fb=",fb(nhbpmf)
           call WRNDIE(-5,'epmf_set()', &
              'wrong fa values. valid fa values: -1.0<=fb<=1.0')
          endif
       endif

         call tabhbpmf(COMLYN,COMLEN,nhbpmf)
         call selchbpmf(COMLYN,COMLEN,nhbpmf)

!     this option is not supported..
      elseif(comnd .eq. 'ANGL') then
         call parseangl(COMLYN,COMLEN)

      elseif(comnd .eq. 'CLEA') then
          write(*,*)'EPMF> Invoked CLEAR'
          call epmf_free()
!
      else
          write(OUTU,*)'option: ',comnd
          call WRNDIE(-5,'epmf_set()',&
               'Unknown option for epmf')
      endif

#if KEY_PARALLEL==1
      endif processparall
#endif 
     end subroutine epmf_set  


     subroutine epmf_init()
  use stream
  use memory
  use number

      integer :: ii,jj,astat

      call CHMALLOC('epmfmodule.src','epmf_init','edt',  &
             maxdtpmf,crl=edt,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','logdt',&
             maxdtpmf,7,log=logdt,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','dtcuf',&
             maxdtpmf,crl=dtcutf,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','dtscale',&
                   maxdtpmf,7,crl=dtscale,qdie=.true.)

      allocate(dtpmf(maxdtpmf,7),dtdnrs(maxdtpmf), &
               dtacps(maxdtpmf),dtnei0(maxdtpmf),dtnei1(maxdtpmf), &
               dtnei2(maxdtpmf),dtnei3(maxdtpmf),dtnein(maxdtpmf), &
               dtneim(maxdtpmf),dtneip(maxdtpmf),stat=astat)
      if(astat/=0) call WRNDIE(-5,'epmf_init()', &
           'Memory allocation for dist pmf failed')

      allocate(hbpmf(maxhbpmf,7),hbdnrs(maxdtpmf), &
               hbacps(maxhbpmf),hbnei0(maxhbpmf),hbnei1(maxhbpmf), &
               hbnei2(maxhbpmf),hbnei3(maxhbpmf),hbnei4(maxhbpmf),&
               hbnei5(maxhbpmf),hbnein(maxhbpmf),&
               hbatm1(maxhbpmf),hbatm2(maxhbpmf),stat=astat)
      if(astat/=0) call WRNDIE(-5,'epmf_init()', &
           'Memory allocation for hb pmf failed')

      call CHMALLOC('epmfmodule.src','epmf_init','loghb',&
             maxhbpmf,7,log=loghb,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','fa',   &
                   maxhbpmf,crl=fa,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','fb',   &
                   maxhbpmf,crl=fb,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','blen', &
                   maxhbpmf,crl=blen,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','ehb',  &
                   maxhbpmf,crl=ehb,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','hbcutf', &
                   maxhbpmf,crl=hbcutf,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','hbscale',&
                   maxhbpmf,7,crl=hbscale,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','ehbmin',&
                   maxhbpmf,crl=ehbmin,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','hbtyp',&
                   maxhbpmf,ch4=hbtyp,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','cha1',&
                   maxhbpmf,ch8=cha1,qdie=.true.)
      call CHMALLOC('epmfmodule.src','epmf_init','cha2',&
                   maxhbpmf,ch8=cha2,qdie=.true.)
      

!     Allocate default values 
      do ii=1,maxdtpmf
         edt(ii)=ZERO
         do jj=1,7
           logdt(ii,jj)=.true. 
         enddo
      enddo
      do ii=1,maxhbpmf
         ehb(ii)=ZERO
         do jj=1,7
            loghb(ii,jj)=.true.
         enddo
      enddo

      if(PRNLEV .gt. 5)write(OUTU,*) &
             'EPMF> Successfully allocated epmf datastructures'

     end subroutine epmf_init

     subroutine epmf_free()
  use stream
  use memory

      integer :: astat

      if (ndtpmf .ne. 0 .and. nhbpmf .ne. 0) then
        ndtpmf=0
        nhbpmf=0
        qinit=.true.
        qepmf=.false.

        if(ndtpmf .ne. 0) then
          call CHMDEALLOC('epmfmodule.src','epmf_free','edt',  &
                   maxdtpmf,crl=edt,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','logdt',&
                   maxdtpmf,7,log=logdt,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','dtcuf',&
                   maxdtpmf,crl=dtcutf,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','dtscale',&
                   maxdtpmf,7,crl=dtscale,qdie=.true.)
          deallocate(dtpmf,dtdnrs,dtacps,dtnei0,dtnei1,dtnei2,dtnei3,&
                   dtnein,dtneim,dtneip,stat=astat)
          if(astat/=0) call WRNDIE(-5,'epmf_free()', &
              'Could not free memory associated with DIST pmf')
        endif

        if(nhbpmf .ne. 0) then
          call CHMDEALLOC('epmfmodule.src','epmf_free','loghb',&
                    maxhbpmf,5,log=loghb,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','fa',   &
                    maxhbpmf,crl=fa,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','fb',   &
                   maxhbpmf,crl=fb,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','blen', &
                   maxhbpmf,crl=blen,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','ehb',  &
                   maxhbpmf,crl=ehb,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','hbcutf', &
                   maxhbpmf,crl=hbcutf,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','hbscale',&
                   maxhbpmf,5,crl=hbscale,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','ehbmin',&
                   maxhbpmf,crl=ehbmin,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','hbtyp',&
                   maxhbpmf,ch4=hbtyp,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','cha1',&
                   maxhbpmf,ch8=cha1,qdie=.true.)
          call CHMDEALLOC('epmfmodule.src','epmf_free','cha2',&
                   maxhbpmf,ch8=cha2,qdie=.true.)
          deallocate(hbpmf,hbdnrs,hbacps,hbnei0,hbnei1,hbnei2,&
                   hbnei3,hbnei4,hbnei5,hbnein,&
                   hbatm1,hbatm2,stat=astat)
          if(astat/=0) call WRNDIE(-5,'epmf_free()', &
            'Could not free memory associated with HB pmf')
        endif

      else
         call WRNDIE(-5,'epmf_free()',&
              'CLEAR called before allocating EPMF data structures')
      endif

     end subroutine epmf_free


     subroutine tabdtpmf(COMLYN,COMLEN,num)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use string
  use memory

     
     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     integer,intent(in) :: num

!     local vars
      integer :: i,j,p,td1,td0
      integer :: kk
      integer :: dd(7),fstat,td
      real(chm_real) :: tx,ty,tz,tx0,dx,tol
      logical :: qfl
      real(chm_real),allocatable,dimension(:) :: y,u,y2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
     dd=-1
!    find which pmfs are defined
     dd(1)=GTRMI(COMLYN,COMLEN,'PMF0',-1)
     if(dd(1) .eq. -1) then
       logdt(num,1)=.false.
     else
       logdt(num,1)=.true.
     endif

     dd(2)=GTRMI(COMLYN,COMLEN,'PMF1',-1)
     if(dd(2) .eq. -1) then
       logdt(num,2)=.false.
     else
       logdt(num,2)=.true.
     endif

     dd(3)=GTRMI(COMLYN,COMLEN,'PMF2',-1)
     if(dd(3) .eq. -1) then
       logdt(num,3)=.false.
     else
       logdt(num,3)=.true.
     endif

     dd(4)=GTRMI(COMLYN,COMLEN,'PMF3',-1)
     if(dd(4) .eq. -1) then
        logdt(num,4)=.false.
     else
        logdt(num,4)=.true.
     endif

     dd(5)=GTRMI(COMLYN,COMLEN,'PMFN',-1)
     if(dd(5) .eq. -1) then
        logdt(num,5)=.false.
     else
        logdt(num,5)=.true.
     endif

     dd(6)=GTRMI(COMLYN,COMLEN,'PMFM',-1)
     if(dd(6) .eq. -1) then
        logdt(num,6)=.false.
     else
        logdt(num,6)=.true.
     endif

     dd(7)=GTRMI(COMLYN,COMLEN,'PMFP',-1)
     if(dd(7) .eq. -1) then
        logdt(num,7)=.false.
     else
        logdt(num,7)=.true.
     endif

!       atleast pmf has to be defined        
     j=0
     do i=1,7
       if(dd(i) .eq. -1) j=j+1
     enddo
     if(j .eq. 7)then
       call WRNDIE(-5,'tabdtpmf',&
           'Specify atleast one of following: PMF0,PMF1,PMF2,..')
     endif

!
!
     j=0
     loop0: do kk=1,7
        chkpmf: if(logdt(num,kk)) then

!  check if pmf data file exists, if it does and is already read
!  rewind the file
           p=0
           fstat=0
!    read header
           inquire(unit=dd(kk),exist=qfl)  
           if(.not. qfl) call WRNDIE(-5,'tabdtpmf',&
              'Cannot find the dist PMF data file')
           rewind(dd(kk))
           do
             read(dd(kk),*,iostat=fstat)tx,ty,tz,td0
             if(fstat /= 0) then
                call WRNDIE(-5,'tabdtpmf', &
                'Cannot read the header section of dist PMF data file')
             endif
             p=p+1
             if(p .ge. 1) exit
           enddo
           dtpmf(num,kk)%maxx=dble(tx)
           dtpmf(num,kk)%binx=dble(ty)
           dtpmf(num,kk)%minx=dble(tz)

!          calculate dimension of the pmf data array and allocate memory 
           td1=nint((dtpmf(num,kk)%maxx-dtpmf(num,kk)%minx)/&
                   dtpmf(num,kk)%binx)+1
           call CHMALLOC("epmfmodule.src","tabdtpmf",&
             "dtpmf(num,kk)%map1d%a",td1,3,crlp=dtpmf(num,kk)%map1d%a)
           dtpmf(num,kk)%map1d%len1 = td1  !num of data points
           dtpmf(num,kk)%map1d%len2 =  3   !x,y,d2/dx2

           p=0 !!! counts number of lines in file now
           do
             read(dd(kk),*,iostat=fstat)tx,ty
             if(fstat .eq. -1) then
                exit ! end of file 
             elseif(fstat /= 0) then
                write(*,'(a30,2i8)')'EPMF> Error on line no.:',p+1,fstat
                call WRNDIE(-5,'tabdtpmf',&
                 'Could not read the dist PMF data file')
             else
                p=p+1
!           fill in the dist pmf data points 
!           ignore the extra lines(if exists) in the file
                if(p .le. dtpmf(num,kk)%map1d%len1) then
                  dtpmf(num,kk)%map1d%a(p,1)=tx
                  dtpmf(num,kk)%map1d%a(p,2)=ty
                  if(p .gt. 2) then
                   tx0=dtpmf(num,kk)%map1d%a(p-1,1)
!            check ordering of data
                   if( (tx-tx0) .lt. ZERO) then
                     write(*,*)&
                     'EPMF> PMF data file not in ascending order'
                      call WRNDIE(-5,"tabdtpmf",&
                      'Could not read the dist PMF data file properly')
                   endif
!            check if the data points are equi-spaced
                   tol=abs(tx-tx0-dtpmf(num,kk)%binx)
                   if(tol .gt. 1e-6) then
                   write(*,*)'EPMF> dist PMF file is not equi-spaced'
                   write(*,'(/,a30,f9.6,a10,f9.6,a10,i4)') &
                   'EPMF> expected: ',dtpmf(num,kk)%binx, &
                   'got:  ',(tx-tx0),'at line ',p
                   call WRNDIE(-5,"tabdtpmf",&
                      'Could not read the dist PMF data file properly')
                  endif
                 endif
                endif
             endif
           enddo
           
!          Assign derivatives using subroutine from ecmap subroutine
           p=dtpmf(num,kk)%map1d%len1  !now p is number of data points
           allocate(y(p),u(p),y2(p))
           dx=dtpmf(num,kk)%binx
           do i=1,p
              y(i)=dtpmf(num,kk)%map1d%a(i,2)
           enddo
           call CMAPSPL(dx,y,p,u,y2)
           do i=1,p
              dtpmf(num,kk)%map1d%a(i,3)=y2(i)
           enddo

           deallocate(y,u,y2)
        endif chkpmf
     enddo loop0
     end subroutine tabdtpmf   


     subroutine tabhbpmf(COMLYN,COMLEN,num)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use string
  use memory
     
     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     integer,intent(in) :: num

!     local vars
      integer :: i,j,dd(7),fstat,kk,p,td0,tdx,tdy,astat
      logical :: qfl
      real(chm_real) :: tx,ty,tz,mnx,mny,ddx,ddy
      real(chm_real),allocatable,dimension(:) :: xd,yd,ed

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     dd=-1
     dd(1)=GTRMI(COMLYN,COMLEN,'PMF0',-1)
     if(dd(1) .eq. -1) then
       loghb(num,1)=.false.
     else
       loghb(num,1)=.true.
     endif

     dd(2)=GTRMI(COMLYN,COMLEN,'PMF1',-1)
     if(dd(2) .eq. -1) then
       loghb(num,2)=.false.
     else
       loghb(num,2)=.true.
     endif

     dd(3)=GTRMI(COMLYN,COMLEN,'PMF2',-1)
     if(dd(3) .eq. -1) then
       loghb(num,3)=.false.
     else
       loghb(num,3)=.true.
     endif

     dd(4)=GTRMI(COMLYN,COMLEN,'PMF3',-1)
     if(dd(4) .eq. -1) then
        loghb(num,4)=.false.
     else
        loghb(num,4)=.true.
     endif

     dd(5)=GTRMI(COMLYN,COMLEN,'PMF4',-1)
     if(dd(5) .eq. -1) then
        loghb(num,5)=.false.
     else
        loghb(num,5)=.true.
     endif

     dd(6)=GTRMI(COMLYN,COMLEN,'PMF5',-1)
     if(dd(6) .eq. -1) then
        loghb(num,6)=.false.
     else
        loghb(num,6)=.true.
     endif

     dd(7)=GTRMI(COMLYN,COMLEN,'PMFN',-1)
     if(dd(7) .eq. -1) then
        loghb(num,7)=.false.
     else
        loghb(num,7)=.true.
     endif

!       atleast pmf has to be defined        
     j=0
     do i=1,7
       if(dd(i) .eq. -1) j=j+1
     enddo
     if(j .eq. 7)then
       call WRNDIE(-5,'tabhbpmf',&
           'Specify atleast one of following: PMF0,PMF1,PMF2,..')
     endif

!
     j=0
     loop0: do kk=1,7
       chkpmf: if(loghb(num,kk)) then
!  check if pmf data file exists, if it does and is already read
!  rewind the file
           p=0
           fstat=0
!    read header
           inquire(unit=dd(kk),exist=qfl)  
           if(.not. qfl) call WRNDIE(-5,'tabdtpmf',&
              'Cannot find the hb PMF data file')
           rewind(dd(kk))
           do 
              read(dd(kk),*,iostat=fstat)tx,ty,tz,td0
              if(fstat /=0) then
                 call WRNDIE(-5,'tabhbpmf',&
                      'Cannot read header section of hb PMF data file')
              endif 
              p=p+1
              if(p .eq. 1) then
                hbpmf(num,kk)%maxy=dble(tx)  !dist max
                hbpmf(num,kk)%biny=dble(ty)  !     bin
                hbpmf(num,kk)%miny=dble(tz)  !     min
              else
                hbpmf(num,kk)%maxx=dble(tx)  !angle max
                hbpmf(num,kk)%binx=dble(ty)  !      bin
                hbpmf(num,kk)%minx=dble(tz)  !      min
              endif
               if(p .ge. 2) exit
           enddo
           tdx=nint((hbpmf(num,kk)%maxx-hbpmf(num,kk)%minx)/&
                   hbpmf(num,kk)%binx)+1
           tdy=nint((hbpmf(num,kk)%maxy-hbpmf(num,kk)%miny)/&
                   hbpmf(num,kk)%biny)+1

           allocate(hbpmf(num,kk)%map2d%a(tdx,tdy,4),&
            xd(tdx*tdy),yd(tdx*tdy),ed(tdx*tdy),stat=astat)
           if(astat /=0) call WRNDIE(-5,'tabhbpmf',&
                         'Cannot allocate memory for hb PMF data')
           hbpmf(num,kk)%map2d%len1=tdx  !angle
           hbpmf(num,kk)%map2d%len2=tdy  !dist
           hbpmf(num,kk)%map2d%len3=4

           td0=tdx*tdy !!! expected number of lines
           p=0   !!! counts number of lines
          
           do  !read the data points..No error checks here..
              p=p+1
              read(dd(kk),*,iostat=fstat)tx,ty,tz
              if(fstat .eq. -1) then
                  exit  !end of the file
              elseif(fstat /= 0) then
                  write(*,'(a30,2i8)')'EPMF> Error on line no.:',p,fstat
                  call WRNDIE(-5,'tabhbpmf',&
                  'Could not read the hb PMF data file')
              else
!               ignore extra lines(if any) in the file              
                if(p .le. td0) then
                     xd(p)=dble(tx)
                     yd(p)=dble(ty)
                     ed(p)=dble(tz)
                endif
              endif
           enddo 

!
!      Reformat the data, calculate paritial derivatives and
!      load it into main data structure, hbpmf
!      HBPMF DATA FILE FORMAT
!      -----------------------
!       row,col:  (dist,angle) : (d0 a1),(d0 a2),(d0 a3) | (d1 a1),(d1 a2),(d1 a3) ..
!      DATA MATRIX:dlenXalen     
!
!      HBPMF DATA STRUC FORMAT
!      -----------------------
!       row,col:  (angle,dist) : (a1 d0),(a2 d0),(a3,d0) | (a1 d1),(a2 d1),(a3 d1) ..  
!       DATA MATRIX:alenXalen
!
           tdx=hbpmf(num,kk)%map2d%len1 !angle
           tdy=hbpmf(num,kk)%map2d%len2 !dist
           td0=(tdx*tdy)-1
           do p=0,td0
              j=int(p/tdx)+1
              i=mod(p,tdx)+1
              if(j .le. tdy .and. i .le. tdx) hbpmf(num,kk)%map2d%a(i,j,1)=ed(p+1)
           enddo

           do j=1,tdy  !dist dimension
              do i=1,tdx  !angle dimension
                 if( (i .eq. 1) .or. (i .eq. tdx) .or. (j .eq. 1) & 
                     .or. (j .eq. tdy) ) then
                    hbpmf(num,kk)%map2d%a(i,j,2)=ZERO
                    hbpmf(num,kk)%map2d%a(i,j,3)=ZERO
                    hbpmf(num,kk)%map2d%a(i,j,4)=ZERO
                 else
!    =======================D E R I V A T I V E S========================
!
!                   D/Dangl
                    hbpmf(num,kk)%map2d%a(i,j,2)=&
      (hbpmf(num,kk)%map2d%a(i+1,j,1)-hbpmf(num,kk)%map2d%a(i-1,j,1))/&
       (2*hbpmf(num,kk)%binx)
!                   D/Ddist
                    hbpmf(num,kk)%map2d%a(i,j,3)=&
      (hbpmf(num,kk)%map2d%a(i,j+1,1)-hbpmf(num,kk)%map2d%a(i,j-1,1))/&
      (2*hbpmf(num,kk)%biny)
!                   D/Dangl Ddist
                    hbpmf(num,kk)%map2d%a(i,j,4)=&
      (hbpmf(num,kk)%map2d%a(i+1,j+1,1)-hbpmf(num,kk)%map2d%a(i+1,j-1,1)-&
      hbpmf(num,kk)%map2d%a(i-1,j+1,1)+hbpmf(num,kk)%map2d%a(i-1,j-1,1))/&
      (4*hbpmf(num,kk)%binx*hbpmf(num,kk)%biny)

!
!    =======================D E R I V A T I V E S========================
                 endif
              enddo
           enddo

       endif chkpmf
        deallocate(xd,yd,ed,stat=astat)
     enddo loop0  
     end subroutine tabhbpmf


     subroutine parseangl(COMLYN,COMLEN)
  use stream
  use dimens_fcm
  use psf
  use number
  use string
  use coord
  use memory
  use select
     
     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     integer :: i,j,dd(7),leng
     character(len=8) :: cha1,cha2
     integer, dimension(natom) :: islct
     real(chm_real), dimension(natom) :: wt

     dd=-1
     dd(1)=GTRMI(COMLYN,COMLEN,'PMF0',-1)
     dd(2)=GTRMI(COMLYN,COMLEN,'PMF1',-1)
     dd(3)=GTRMI(COMLYN,COMLEN,'PMF2',-1)
     dd(4)=GTRMI(COMLYN,COMLEN,'PMF3',-1)
     dd(5)=GTRMI(COMLYN,COMLEN,'PMFN',-1)

     call GTRMWD(COMLYN,COMLEN,'ATM1',4,cha1,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ATM2',4,cha2,8,leng)

     if(INDXA(COMLYN,COMLEN,'ACCP') .gt. 0) then
       call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
     endif

     if(INDXA(COMLYN,COMLEN,'DONO') .gt. 0) then
       call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
     endif


     end subroutine parseangl

     subroutine selcdtpmf(COMLYN,COMLEN,num)
  use select
  use stream
  use string
  use dimens_fcm
  use psf
  use number
  use coord
  use memory
  use chutil,only:getres,getseg,atomid

     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     integer,intent(in) :: num

!    vars for parsing atom selctions
     real(chm_real), dimension(natom) :: wt  
     integer, dimension(natom) :: islct
     character(len=8) :: sid,rid,ren,ac,at1,at2
     integer :: isega,isegb


!    local vars
     integer :: ns,i,ai,ii,jj,ll,kk,mm,pp,nt(7),astat,ndnrs,nacps
     real(chm_real) :: x1,y1,z1,dis,x2,y2,z2
     integer,dimension(:),allocatable :: n0,n1,n2,n3,nn,nm,np
     

     dnrselct: if(INDXA(COMLYN,COMLEN,'DONO') .gt. 0) then
        call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
        ns=NSELCT(NATOM,islct)
        if(ns .eq. 0) then
           if(PRNLEV .gt. 5)&
            write(OUTU,'(a40,i8)')'EPMF> No donors for pmf: ',num
        else
           call CHMALLOC("epmfmodule.src","selcdtpmf",&
           "dtdnrs(num)%a",ns,intgp=dtdnrs(num)%a)
           dtdnrs(num)%len=ns
           ai=0
           do i=1,natom
              if(islct(i) .eq. 1) then
                 ai=ai+1
                 if(ai .le. dtdnrs(num)%len) dtdnrs(num)%a(ai)=i 
              endif
           enddo
        endif
     else
        call WRNDIE(-5,'epmf:selcdtpmf', &
             'Missing DONO keyword for DIST pmf')
     endif dnrselct
!
     acpselct: if(INDXA(COMLYN,COMLEN,'ACCP') .gt. 0) then
        call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
        ns=NSELCT(NATOM,islct)
        if(ns .eq. 0) then
           if(PRNLEV .gt. 5)&
            write(OUTU,'(a40,i8)')'EPMF> No acceptors for pmf: ',num
        else
           call CHMALLOC("epmfmodule.src","selecdtpmf",&
           "dtacps(num)%a",ns,intgp=dtacps(num)%a)
           dtacps(num)%len=ns
           ai=0
           do i=1,natom
              if(islct(i) .eq. 1) then
                 ai=ai+1
                 if(ai .le. dtacps(num)%len) dtacps(num)%a(ai)=i
              endif
           enddo
         endif  
     else
        call WRNDIE(-5,'epmf:selcdtpmf', &
             'Missing ACCP keyword for DIST pmf')
     endif acpselct

     if(PRNLEV .gt. 5) then
        write(OUTU,'(a8,a40,i6,a2,i6)')'EPMF> ',&
        'No. of donors for 1D PMF line',num,':',dtdnrs(num)%len
        write(OUTU,'(a8,a40,i6,a2,i6)')'EPMF> ',&
        'No. of acceptors for 1D PMF line',num,':',dtacps(num)%len
     endif

!
!     Neigh selections (different type of PMFS)
!
     ndnrs=dtdnrs(num)%len
     nacps=dtacps(num)%len

     neiselct:if( (ndnrs .ne. 0) .and. (nacps .ne. 0) )then
!     
!  all neighbor list memory is allocated for every donor even though
!  only one pmf type is defined(see example below).This is done inorder 
!  to account for multiple pmf type definations
!  examples:
!   extreme case: epmf dist PMF0 11 PMF1 12 PMF2 13 PMF3 14 PMFN 15 PMFP 16 PMFM 17 ..
!   simple case: epmf dist PMF0 11 ..
!
      allocate(dtnei0(num)%list(ndnrs),dtnei1(num)%list(ndnrs), &
               dtnei2(num)%list(ndnrs),dtnei3(num)%list(ndnrs), &
               dtnein(num)%list(ndnrs),dtneim(num)%list(ndnrs), &
               dtneip(num)%list(ndnrs),stat=astat)
      if(astat /= 0) then
        call WRNDIE(-5,'epmf:selcdtpmf',&
             'Cannot allocate memory for neighbors of dist pmfs')
      endif

      do ii=1,ndnrs
        allocate(n0(nacps),n1(nacps),n2(nacps),n3(nacps), &
           nn(nacps),np(nacps),nm(nacps))
        n0=-1
        n1=-1
        n2=-1
        n3=-1
        nn=-1
        nm=-1
        np=-1

        nt=0
        kk=dtdnrs(num)%a(ii) 
        pp=GETRES(kk,IBASE,NRES)  ! residue to which kkth donr belongs
        isega=GETSEG(pp,NICTOT,NSEG)
        x1=X(kk)
        y1=Y(kk)
        z1=Z(kk)

        do jj=1,nacps
           ll=dtacps(num)%a(jj)
           mm=GETRES(ll,IBASE,NRES)  !residue to which jjth accp belongs
           isegb=GETSEG(mm,NICTOT,NSEG)
           x2=X(ll)
           y2=Y(ll)
           z2=Z(ll)
           if(mm .eq. pp)  then 
              nt(1)=nt(1)+1        !ii+0
              n0(jj)=ll         !this is right accp for pmf0 
           elseif(mm .eq. pp+1 .or. mm .eq. pp-1) then 
              if(isega .eq. isegb) then
                nt(2)=nt(2)+1
                n1(jj)=ll
                if(mm .eq. pp+1) then
                   nt(7)=nt(7)+1        !ii+1
                   np(jj)=ll
                else
                   nt(6)=nt(6)+1        !ii-1
                   nm(jj)=ll
                endif
              endif
           elseif(mm .eq. pp+2 .or. mm .eq. pp-2) then 
              nt(3)=nt(3)+1
              n2(jj)=ll
           elseif(mm .eq. pp+3 .or. mm .eq. pp-3) then 
              nt(4)=nt(4)+1
              n3(jj)=ll
           else 
              dis=sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
              if(dis .le. dtcutf(num)) then
                 nt(5)=nt(5)+1  !ignore this value, exists only for consistency of names
                 nn(jj)=ll
              endif
           endif
        enddo

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtnei0(num)%list(ii)%a",nt(1),intgp=dtnei0(num)%list(ii)%a) 
        dtnei0(num)%list(ii)%len=nt(1)
        dtnei0(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtnei1(num)%list(ii)%a",nt(2),intgp=dtnei1(num)%list(ii)%a) 
        dtnei1(num)%list(ii)%len=nt(2)
        dtnei1(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtnei2(num)%list(ii)%a",nt(3),intgp=dtnei2(num)%list(ii)%a) 
        dtnei2(num)%list(ii)%len=nt(3)
        dtnei2(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtnei3(num)%list(ii)%a",nt(4),intgp=dtnei3(num)%list(ii)%a) 
        dtnei3(num)%list(ii)%len=nt(4)
        dtnei3(num)%list(ii)%a=-1

!       number of 'N' neighbors can change during simulation
!       so it is equal to number of accps
        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtnein(num)%list(ii)%a",nacps,intgp=dtnein(num)%list(ii)%a) 
        dtnein(num)%list(ii)%len=nacps
        dtnein(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtneim(num)%list(ii)%a",nt(6),intgp=dtneim(num)%list(ii)%a) 
        dtneim(num)%list(ii)%len=nt(6)
        dtneim(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selcdtpmf", & 
             "dtneip(num)%list(ii)%a",nt(7),intgp=dtneip(num)%list(ii)%a) 
        dtneip(num)%list(ii)%len=nt(7)
        dtneip(num)%list(ii)%a=-1

        if(PRNLEV .gt. 5) then
          write(OUTU,'(a15,i5,a22,i5,a2,7i5)')'EPMF> pmfno.',num,&
          'No. of neigbors for ',ii,':',nt
        endif

        nt=0
        do jj=1,nacps
           if(n0(jj) .ne. -1 .and. &
              nt(1) .le. dtnei0(num)%list(ii)%len) then
              nt(1)=nt(1)+1
              dtnei0(num)%list(ii)%a(nt(1))=n0(jj)
           endif

           if(n1(jj) .ne. -1 .and. &
              nt(2) .le. dtnei1(num)%list(ii)%len) then
              nt(2)=nt(2)+1
              dtnei1(num)%list(ii)%a(nt(2))=n1(jj)
           endif

           if(n2(jj) .ne. -1 .and. &
              nt(3) .le. dtnei2(num)%list(ii)%len) then
              nt(3)=nt(3)+1
              dtnei2(num)%list(ii)%a(nt(3))=n2(jj)
           endif

           if(n3(jj) .ne. -1 .and. &
              nt(4) .le. dtnei3(num)%list(ii)%len) then
              nt(4)=nt(4)+1
              dtnei3(num)%list(ii)%a(nt(4))=n3(jj)
           endif

           if(nn(jj) .ne. -1 ) then 
              dtnein(num)%list(ii)%a(jj)=nn(jj)
           endif

           if(nm(jj) .ne. -1 .and. & 
              nt(6) .le. dtneim(num)%list(ii)%len) then
              nt(6)=nt(6)+1
              dtneim(num)%list(ii)%a(nt(6))=nm(jj)
           endif

           if(np(jj) .ne. -1 .and. &
              nt(7) .le. dtneip(num)%list(ii)%len) then
              nt(7)=nt(7)+1
              dtneip(num)%list(ii)%a(nt(7))=np(jj)
           endif
        enddo

       deallocate(n0,n1,n2,n3,nn,np,nm)
      enddo
     endif neiselct

     end subroutine selcdtpmf


     subroutine selchbpmf(COMLYN,COMLEN,num)
  use select
  use stream
  use string
  use dimens_fcm
  use psf
  use number
  use coord
  use memory
  use chutil,only:getres,getseg,atomid

     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     integer,intent(in) :: num

!    vars for parsing atom selctions
     real(chm_real), dimension(natom) :: wt  
     integer, dimension(natom) :: islct
     character(len=8) :: sid,rid,ren,ac,at1,at2,tt1,tt2

!    local vars
     integer :: ns,i,j,ai,aj,ii,jj,ll,kk,mm,pp,nt(7),astat,ndnrs,nacps,&
                isegd,isega,isegb
     real(chm_real) :: x1,y1,z1,dis,x2,y2,z2
     integer,dimension(:),allocatable :: n0,n1,n2,n3,n4,n5,nn
     character(len=1) :: pfx

     dnrselct:if(INDXA(COMLYN,COMLEN,'DONO') .gt. 0) then
        call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
        ns=NSELCT(NATOM,islct)
        if(ns .eq. 0) then
           if(PRNLEV .gt. 5)&
            write(OUTU,'(a40,i8)')'EPMF> No donors for pmf: ',num
        else
           call CHMALLOC("epmfmodule.src","selchbpmf",&
           "hbdnrs(num)%a",ns,intgp=hbdnrs(num)%a)
           hbdnrs(num)%len=ns
           call CHMALLOC("epmfmodule.src","selchbpmf",&
           "hbatm1(num)%a",ns,intgp=hbatm1(num)%a)
           hbatm1(num)%len=ns
           call CHMALLOC("epmfmodule.src","selchbpmf",&
           "hbatm2(num)%a",ns,intgp=hbatm2(num)%a)
           hbatm2(num)%len=ns

           ai=0
           do i=1,natom
              if(islct(i) .eq. 1) then
                 call ATOMID(i,sid,rid,ren,ac)
                 ai=ai+1
                 if(ai .le. hbdnrs(num)%len) hbdnrs(num)%a(ai)=i 
              endif
           enddo
        endif
     else
        call WRNDIE(-5,'epmf:selchbpmf', &
             'Missing DONO keyword for HB pmf')
     endif dnrselct

     acpselct: if(INDXA(COMLYN,COMLEN,'ACCP') .gt. 0) then
        call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,wt,.TRUE.)
        ns=NSELCT(NATOM,islct)
        if(ns .eq. 0) then
           if(PRNLEV .gt. 5)&
            write(OUTU,'(a40,i8)')'EPMF> No acceptors for pmf: ',num
        else
           call CHMALLOC("epmfmodule.src","selechbpmf",&
           "hbacps(num)%a",ns,intgp=hbacps(num)%a)
           hbacps(num)%len=ns
           ai=0
           do i=1,natom
              if(islct(i) .eq. 1) then
                 call ATOMID(i,sid,rid,ren,ac)
                 ai=ai+1
                 if(ai .le. hbacps(num)%len) hbacps(num)%a(ai)=i
              endif
           enddo
         endif  
     else
        call WRNDIE(-5,'epmf:selchbpmf', &
             'Missing ACCP keyword for HB pmf')
     endif acpselct

     if(PRNLEV .gt. 5) then
        write(OUTU,'(a8,a40,i6,a2,i6)')'EPMF> ',&
        'No. of donors for HB PMF line',num,':',hbdnrs(num)%len
        write(OUTU,'(a8,a40,i6,a2,i6)')'EPMF> ',&
        'No. of acceptors for HB PMF line',num,':',hbacps(num)%len
     endif

     
     ndnrs=hbdnrs(num)%len
     nacps=hbacps(num)%len
     neiselct: if( (ndnrs .ne. 0) .and. (nacps .ne. 0) ) then
        allocate(hbnei0(num)%list(ndnrs),hbnei1(num)%list(ndnrs),&
                 hbnei2(num)%list(ndnrs),hbnei3(num)%list(ndnrs),& 
                 hbnei4(num)%list(ndnrs),hbnei5(num)%list(ndnrs),&
                 hbnein(num)%list(ndnrs),stat=astat)
        if(astat /= 0) then
          call WRNDIE(-5,'epmf:selchbpmf',&
             'Cannot allocate memory for neighbors of hb pmfs [0]')
        endif
        do ii=1,ndnrs
           allocate(n0(nacps),n1(nacps),n2(nacps),n3(nacps), &
           n4(nacps),n5(nacps),nn(nacps),stat=astat)
           if(astat /= 0) then
             call WRNDIE(-5,'epmf:selchbpmf',&
             'Cannot allocate memory for neighbors of hb pmfs [1]')
           endif

           n0=-1
           n1=-1
           n2=-1
           n3=-1
           n4=-1
           n5=-1
           nn=-1

           nt=0
           kk=hbdnrs(num)%a(ii)
           pp=GETRES(kk,IBASE,NRES)
           x1=X(kk)
           y1=Y(KK)
           z1=Z(kk)

           do jj=1,nacps
              ll=hbacps(num)%a(jj)
              mm=GETRES(ll,IBASE,NRES)
              x2=X(ll)
              y2=Y(ll)
              z2=Z(ll)
              if(mm .eq. pp) then
                 nt(1)=nt(1)+1
                 n0(jj)=ll
              elseif(mm .eq. pp+1 .or. mm .eq. pp-1) then
                 nt(2)=nt(2)+1
                 n1(jj)=ll
              elseif(mm .eq. pp+2 .or. mm .eq. pp-2) then
                 nt(3)=nt(3)+1
                 n2(jj)=ll
              elseif(mm .eq. pp+3 .or. mm .eq. pp-3) then
                 nt(4)=nt(4)+1
                 n3(jj)=ll
              elseif(mm .eq. pp+4 .or. mm .eq. pp-4) then
                 nt(5)=nt(5)+1
                 n4(jj)=ll
              elseif(mm .eq. pp+5 .or. mm .eq. pp-5) then
                 nt(6)=nt(6)+1
                 n5(jj)=ll
              else   
                 dis=sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
                 if(dis .le. hbcutf(num)) then
                     nt(7)=nt(7)+1 ! dummy 
                     nn(jj)=ll
                 endif
               endif
           enddo  

        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei0(num)%list(ii)%a",nt(1),intgp=hbnei0(num)%list(ii)%a) 
        hbnei0(num)%list(ii)%len=nt(1)
        hbnei0(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei1(num)%list(ii)%a",nt(2),intgp=hbnei1(num)%list(ii)%a) 
        hbnei1(num)%list(ii)%len=nt(2)
        hbnei1(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei2(num)%list(ii)%a",nt(3),intgp=hbnei2(num)%list(ii)%a) 
        hbnei2(num)%list(ii)%len=nt(3)
        hbnei2(num)%list(ii)%a=-1

        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei3(num)%list(ii)%a",nt(4),intgp=hbnei3(num)%list(ii)%a) 
        hbnei3(num)%list(ii)%len=nt(4)
        hbnei3(num)%list(ii)%a=-1


        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei4(num)%list(ii)%a",nt(5),intgp=hbnei4(num)%list(ii)%a) 
        hbnei4(num)%list(ii)%len=nt(5)
        hbnei4(num)%list(ii)%a=-1


        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnei5(num)%list(ii)%a",nt(6),intgp=hbnei5(num)%list(ii)%a) 
        hbnei5(num)%list(ii)%len=nt(6)
        hbnei5(num)%list(ii)%a=-1


        call CHMALLOC("epmfmodule.src","selchbpmf", & 
             "hbnein(num)%list(ii)%a",nacps,intgp=hbnein(num)%list(ii)%a) 
        hbnein(num)%list(ii)%len=nacps
        hbnein(num)%list(ii)%a=-1

        if(PRNLEV .gt. 5) then
          write(OUTU,'(a23,i5,a22,i5,a2,7i5)')'EPMF> hb pmfno.',num,&
          'No. of neigbors for ',ii,':',nt
        endif
          
        nt=0
        do jj=1,nacps
           if(n0(jj) .ne. -1 .and. &
              nt(1) .le. hbnei0(num)%list(ii)%len) then
              nt(1)=nt(1)+1
              hbnei0(num)%list(ii)%a(nt(1))=n0(jj)
           endif
        
           if(n1(jj) .ne. -1 .and. &
              nt(2) .le. hbnei1(num)%list(ii)%len) then
              nt(2)=nt(2)+1
              hbnei1(num)%list(ii)%a(nt(2))=n1(jj)
           endif

           if(n2(jj) .ne. -1 .and. &
              nt(3) .le. hbnei2(num)%list(ii)%len) then
              nt(3)=nt(3)+1
              hbnei2(num)%list(ii)%a(nt(3))=n2(jj)
           endif

           if(n3(jj) .ne. -1 .and. &
              nt(4) .le. hbnei3(num)%list(ii)%len) then
              nt(4)=nt(4)+1
              hbnei3(num)%list(ii)%a(nt(4))=n3(jj)
           endif

           if(n4(jj) .ne. -1 .and. &
              nt(5) .le. hbnei4(num)%list(ii)%len) then
              nt(5)=nt(5)+1
              hbnei4(num)%list(ii)%a(nt(5))=n4(jj)
           endif

           if(n5(jj) .ne. -1 .and. &
              nt(6) .le. hbnei5(num)%list(ii)%len) then
              nt(6)=nt(6)+1
              hbnei5(num)%list(ii)%a(nt(6))=n5(jj)
           endif

           if(nn(jj) .ne. -1) then
              hbnein(num)%list(ii)%a(jj)=nn(jj)
           endif
        enddo
        deallocate(n0,n1,n2,n3,n4,n5,nn)
      enddo

!
!                ATM1/ATM2  selections
!
      do ii=1,ndnrs
          kk=hbdnrs(num)%a(ii)
          pp=GETRES(kk,IBASE,NRES)         
          tt1=trim(cha1(num))
          if(tt1(1:1) .eq. '+') then
            pfx='+'
            at1=tt1(2:)
          elseif(tt1(1:1) .eq. '-') then
            pfx='-'
            at1=tt1(2:)
          else
            pfx=' '
            at1=tt1
          endif
          call selctatoms(pp,at1,pfx,mm)
          hbatm1(num)%a(ii)=mm

          tt2=trim(cha2(num))
          if(tt2(1:1) .eq. '+') then
            pfx='+'
            at2=tt2(2:)
          elseif(tt2(1:1) .eq. '-') then
            pfx='-'
            at2=tt2(2:)
          else
            pfx=' '
            at2=tt2
          endif
          call selctatoms(pp,at2,pfx,ll)
          hbatm2(num)%a(ii)=ll

          pp=GETRES(hbdnrs(num)%a(ii),IBASE,NRES)
          isegd=GETSEG(pp,NICTOT,NSEG)

          if(hbatm1(num)%a(ii) .ne. -1 .and. &
             hbatm2(num)%a(ii) .ne. -1 ) then

             pp=GETRES(hbatm1(num)%a(ii),IBASE,NRES)
             isega=GETSEG(pp,NICTOT,NSEG)
             pp=GETRES(hbatm2(num)%a(ii),IBASE,NRES)
             isegb=GETSEG(pp,NICTOT,NSEG)
             
             if(isega .ne. isegd .or. isegb .ne. isegd) then
                hbatm1(num)%a(ii)=-1
                hbatm2(num)%a(ii)=-1 
             endif
          endif

       enddo
     endif neiselct
     end subroutine selchbpmf


     subroutine edtpmf(edt,ndt,x,y,z,dx,dy,dz)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl
  use memory
     
     real(chm_real),dimension(:),intent(inout) :: DX,DY,DZ
     real(chm_real),dimension(:),intent(in) :: X,Y,Z
     real(chm_real),intent(out) :: edt
     integer,intent(in) :: ndt


!    local vars
     real(chm_real),allocatable,dimension(:) :: ya,y2a
     integer,allocatable,dimension(:) :: tn
     real(chm_real) :: xmin,xbin,xmax
     integer :: i,j,k,ii,jj,kk,astat,npts,ndnrs,nacps,nn
     real(chm_real) :: tdx,tdy,tdz,dlen,ene,drv
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     ndnrs=dtdnrs(ndt)%len
     nacps=dtacps(ndt)%len
     ene=ZERO
     edt=ZERO                 !!!! 1d pmf energy !!!!

     nodnracps: if(ndnrs .ne. 0 .and. nacps .ne. 0) then
     dologdt: do ii=1,7 !pmftype
        iflogdt:if(logdt(ndt,ii)) then
            
            npts=dtpmf(ndt,ii)%map1d%len1
            call CHMALLOC('epmfmodule.src','edtpmf','ya',&
                 npts,crl=ya,qdie=.true.)
            call CHMALLOC('epmfmodule.src','edtpmf','y2a',&
                 npts,crl=y2a,qdie=.true.)
            do jj=1,npts
               ya(jj)=dtpmf(ndt,ii)%map1d%a(jj,2)
               y2a(jj)=dtpmf(ndt,ii)%map1d%a(jj,3)
            enddo
            xmin=dtpmf(ndt,ii)%minx
            xmax=dtpmf(ndt,ii)%maxx
            xbin=dtpmf(ndt,ii)%binx
                    
            loopdnrs: do jj=1,ndnrs  !donors
               if(ii .eq. 1) then
                  nn=dtnei0(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtnei0(ndt)%list(jj)%a
               elseif(ii .eq. 2) then
                  nn=dtnei1(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtnei1(ndt)%list(jj)%a
               elseif(ii .eq. 3) then
                  nn=dtnei2(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtnei2(ndt)%list(jj)%a
               elseif(ii .eq. 4) then
                  nn=dtnei3(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtnei3(ndt)%list(jj)%a
               elseif(ii .eq. 5) then
                  nn=dtnein(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtnein(ndt)%list(jj)%a
               elseif(ii .eq. 6) then
                  nn=dtneim(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtneim(ndt)%list(jj)%a
               else
                  nn=dtneip(ndt)%list(jj)%len
                  call CHMALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
                  tn=dtneip(ndt)%list(jj)%a
               endif

!               
!           Main energy loop over selected accps stored in array tn
!    
               if(nn .gt. 0) then !accploop
               do i=1,nn !accps  
                if(tn(i) .ne. -1) then    !comes from dtnein
                  tdx=x(tn(i))-x(dtdnrs(ndt)%a(jj))
                  tdy=y(tn(i))-y(dtdnrs(ndt)%a(jj))
                  tdz=z(tn(i))-z(dtdnrs(ndt)%a(jj))
                  dlen=sqrt(tdx**2+tdy**2+tdz**2)
                  if(dlen .lt. xmax .and. dlen .gt. xmin) then
                     call CMAPSPI(xmin,xbin,ya,y2a,dlen,ene,drv)
                     edt=edt+dtscale(ndt,ii)*ene      !!!! 1D pmf energy  !!!!   
                     dx(tn(i))=dx(tn(i))+((drv*dtscale(ndt,ii)*tdx)/dlen)
                     dy(tn(i))=dy(tn(i))+((drv*dtscale(ndt,ii)*tdy)/dlen)
                     dz(tn(i))=dz(tn(i))+((drv*dtscale(ndt,ii)*tdz)/dlen)
                     dx(dtdnrs(ndt)%a(jj))=dx(dtdnrs(ndt)%a(jj)) &
                                -((drv*dtscale(ndt,ii)*tdx)/dlen)
                     dy(dtdnrs(ndt)%a(jj))=dy(dtdnrs(ndt)%a(jj)) &
                                -((drv*dtscale(ndt,ii)*tdy)/dlen)
                     dz(dtdnrs(ndt)%a(jj))=dz(dtdnrs(ndt)%a(jj)) &
                                -((drv*dtscale(ndt,ii)*tdz)/dlen)
                  else
                    npts=dtpmf(ndt,ii)%map1d%len1
                    if(dlen .le. xmin) then
                        ene=ya(1)
                        edt=edt+(dtscale(ndt,ii)*ene) !!!! 1D pmf energy  !!!!  
                    else
                        ene=ya(npts)
                        edt=edt+(dtscale(ndt,ii)*ene)!!!! 1D pmf energy  !!!! 
                    endif
        
                  endif

                endif
               enddo 
               endif !accploop

             call CHMDEALLOC('epmfmodule.src','edtpmf','tn',nn,intg=tn)
            enddo loopdnrs
            call CHMDEALLOC('epmfmodule.src','edtpmf','y2a',npts,crl=y2a)
            call CHMDEALLOC('epmfmodule.src','edtpmf','ya',npts,crl=ya)
        else
            continue  !edt=ZERO
        endif iflogdt
        

     enddo dologdt
     endif nodnracps
     end subroutine edtpmf 


     subroutine ehbpmf(ehb,nhb,x,y,z,dx,dy,dz)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl
  use memory
  use consta
  use chutil,only:getres

     real(chm_real),dimension(:),intent(inout) :: DX,DY,DZ
     real(chm_real),dimension(:),intent(in) :: X,Y,Z
     real(chm_real),intent(out) :: ehb
     integer,intent(in) :: nhb


!    local vars
     integer,allocatable,dimension(:) :: tn
     real(chm_real) :: xmin,xbin,xmax,ymin,ybin,ymax
     integer :: i,j,k,ii,jj,kk,ndnrs,nacps,&
                nn,resdnr,resacp,twrn,tlen
     real(chm_real),dimension(3) :: dvdx1,dvdx2,dvdx3,dvdy1,dvdy2,&
                dvdy3,dvdz1,dvdz2,dvdz3,rh
     real(chm_real) :: ax,ay,az,bx,by,bz,rnx,rny,rnz,eres(nres)
     real(chm_real) :: v0(3),v1(3),v2(3),v3(3),racp(3),rdon(3),rhyd(3),&
                       cosa,xcosa,tbon,tbmax,tbmin,ene,t1dx,t1dy,&
                       t1dz,d1len,dlen,d2len,tdx,tdy,tdz,df1,df2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


     ene=ZERO
     ehb=ZERO
     eres=ZERO   !!! ehb energy / residue
     ndnrs=hbdnrs(nhb)%len
     nacps=hbacps(nhb)%len

     nodnracps: if( (ndnrs .ne. 0) .and. (nacps .ne. 0) ) then
     dologhb: do ii=1,7  !pmftype
!!!!!!!    
       ifloghb: if(loghb(nhb,ii)) then
       twrn=0
       dnrloop: do jj=1,ndnrs

          resdnr=GETRES(hbdnrs(nhb)%a(jj),IBASE,NRES)

!      make local neighbor list         
       if(ii .eq. 1) then
         nn=hbnei0(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei0(nhb)%list(jj)%a
         tlen=nn
       elseif(ii .eq. 2) then
         nn=hbnei1(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei1(nhb)%list(jj)%a
         tlen=nn
       elseif(ii .eq. 3) then
         nn=hbnei2(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei2(nhb)%list(jj)%a
         tlen=nn
       elseif(ii .eq. 4) then
         nn=hbnei3(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei3(nhb)%list(jj)%a
         tlen=nn
       elseif(ii .eq. 5) then
         nn=hbnei4(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei4(nhb)%list(jj)%a
         tlen=nn
       elseif(ii .eq. 6) then
         nn=hbnei5(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnei5(nhb)%list(jj)%a
         tlen=nn
       else
         nn=hbnein(nhb)%list(jj)%len
         call CHMALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)
         tn=hbnein(nhb)%list(jj)%a
         tlen=nn
       endif



!!!!!!!!!!!!!!!! 
       chkhbtype: if(hbtyp(nhb) .eq. 'DEFA') then !!!!!!!!!DEFA!!!!!!!!!
         if(hbatm1(nhb)%a(jj) .eq. -1 .or. hbatm2(nhb)%a(jj) .eq. -1)then
            ene=ZERO  !no ehb calculation
         else
            
            ax=X(hbatm1(nhb)%a(jj))-X(hbdnrs(nhb)%a(jj))
            ay=Y(hbatm1(nhb)%a(jj))-Y(hbdnrs(nhb)%a(jj))
            az=Z(hbatm1(nhb)%a(jj))-Z(hbdnrs(nhb)%a(jj))
            bx=X(hbatm2(nhb)%a(jj))-X(hbdnrs(nhb)%a(jj))
            by=Y(hbatm2(nhb)%a(jj))-Y(hbdnrs(nhb)%a(jj))
            bz=Z(hbatm2(nhb)%a(jj))-Z(hbdnrs(nhb)%a(jj))
!           distance between hyd(jj) and donr(jj)
            rnx=fa(nhb)*ax+fb(nhb)*bx
            rny=fa(nhb)*ay+fb(nhb)*by
            rnz=fa(nhb)*az+fb(nhb)*bz
            rhyd(1)=rnx+X(hbdnrs(nhb)%a(jj))                    !!HB(JJ)
            rhyd(2)=rny+Y(hbdnrs(nhb)%a(jj))                    !!COORDS
            rhyd(3)=rnz+Z(hbdnrs(nhb)%a(jj))                    !!
            rdon=(/X(hbdnrs(nhb)%a(jj)),Y(hbdnrs(nhb)%a(jj)),&  !!DONR(JJ) 
               Z(hbdnrs(nhb)%a(jj))/)                           !!COORDS
!           check bond distances
            tbon=sqrt((rhyd(1)-rdon(1))**2+(rhyd(2)-rdon(2))**2+&
                    (rhyd(3)-rdon(3))**2)
            tbmax=blen(nhb)+(0.1*blen(nhb))
            tbmin=blen(nhb)-(0.1*blen(nhb))
            if(tbon .ge. tbmax .or. tbon .le. tbmin) twrn=twrn+1


!==========================   DEFA ENERGIES   ============================|

!
!       Main energy loop over selected acceptors
!
         
          do i=1,tlen
             if(tn(i) .ne. -1) then !comes from hbnein
                resacp=GETRES(tn(i),IBASE,NRES)
                racp=(/X(tn(i)),Y(tn(i)),Z(tn(i))/) !!ACP(I) COORDS

!                       ^    ^      ^   ^
!               cosa =  rha. rnh = -rah.rnh             
!
!               distance between hyd(jj) and acp(i)
                t1dx=rhyd(1)-racp(1)     
                t1dy=rhyd(2)-racp(2)
                t1dz=rhyd(3)-racp(3)
                d1len=sqrt(t1dx**2+t1dy**2+t1dz**2)
!               distance between dnr(jj) and acp(i)             
                tdx=racp(1)-rdon(1)
                tdy=racp(2)-rdon(2)
                tdz=racp(3)-rdon(3)
                dlen=sqrt(tdx**2+tdy**2+tdz**2)
                d2len=(t1dx*rnx+t1dy*rny+t1dz*rnz)
                xcosa=-d2len/(d1len*blen(nhb))
                if(xcosa .gt. 1.0) then
                   cosa=1.0
                elseif(xcosa .lt. -1.0) then
                   cosa=-1.0
                else
                   cosa=xcosa
                endif
              
                if( eres(resdnr) .le. ehbmin(nhb) .or.  &
                    eres(resacp) .le. ehbmin(nhb)  ) then
                    ene=ZERO
                else
                    call calcene1(ene,nhb,ii,dlen,cosa, &
                         tdx,tdy,tdz,t1dx,t1dy,t1dz,rnx,rny,rnz,&
                         tn(i),hbdnrs(nhb)%a(jj),hbatm1(nhb)%a(jj),&
                         hbatm2(nhb)%a(jj),DX,DY,DZ)   
                endif
                ehb=ehb+ene !!!ehb energy donr(jj)-acp(i)
                eres(resdnr)=eres(resdnr)+(HALF*ene)
                eres(resacp)=eres(resacp)+(HALF*ene)  
  
             endif
          enddo
         

!==========================   DEFA ENERGIES   ============================|

         endif 
          
       else  !!!!!!!!!!!!!!GEOM!!!!!!!!!!!!!!!!!
         if(hbatm1(nhb)%a(jj) .eq. -1 .or. hbatm2(nhb)%a(jj) .eq. -1)then 
           ene=ZERO
         else
          v1=(/X(hbdnrs(nhb)%a(jj)),Y(hbdnrs(nhb)%a(jj)),&
              Z(hbdnrs(nhb)%a(jj))/)
          v2=(/X(hbatm1(nhb)%a(jj)),Y(hbatm1(nhb)%a(jj)),&
              Z(hbatm1(nhb)%a(jj))/)
          v3=(/X(hbatm2(nhb)%a(jj)),Y(hbatm2(nhb)%a(jj)),&
              Z(hbatm2(nhb)%a(jj))/)
          call buildatom(blen(nhb),fa(nhb),fb(nhb),v1,v2,v3,v0, &
              dvdx1,dvdy1,dvdz1,dvdx2,dvdy2,dvdz2,dvdx3,dvdy3,dvdz3)

          rhyd=v0
          rdon=(/X(hbdnrs(nhb)%a(jj)),Y(hbdnrs(nhb)%a(jj)),&
               Z(hbdnrs(nhb)%a(jj))/)

!         check bond distances
          tbon=sqrt((rhyd(1)-rdon(1))**2+(rhyd(2)-rdon(2))**2+&
                   (rhyd(3)-rdon(3))**2)
          tbmax=blen(nhb)+(0.1*blen(nhb))
          tbmin=blen(nhb)-(0.1*blen(nhb))
          if(tbon .ge. tbmax .or. tbon .le. tbmin) twrn=twrn+1
         
!          write(*,*) 'GDH',hbatm1(nhb)%a(jj),hbatm2(nhb)%a(jj) 
!          write(*,*) 'GDON',hbdnrs(nhb)%a(jj)
!          write(*,*) 'GHYD',rhyd(1),rhyd(2),rhyd(3)


!==========================   GEOM ENERGIES   ============================|

!
!       Main energy loop over selected acceptors
!
          do i=1,tlen
             if(tn(i) .ne. -1) then !comes from hbnein
                resacp=GETRES(tn(i),IBASE,NRES)
                racp=(/X(tn(i)),Y(tn(i)),Z(tn(i))/) !!ACP(I) COORDS
                dlen=sqrt( (racp(1)-rdon(1))**2+(racp(2)-rdon(2))**2+&
                           (racp(3)-rdon(3))**2 )
!                write(*,*) 'GD',tn(i) 
                
                call getcosangl3(rdon,rhyd,racp,xcosa)
                if(xcosa .gt. 1.0) then
                   cosa=1.0
                elseif(xcosa .lt. -1.0) then
                   cosa=-1.0
                else
                   cosa=xcosa
                endif

                if( eres(resdnr) .le. ehbmin(nhb) .or.  &
                    eres(resacp) .le. ehbmin(nhb)  ) then
                    ene=ZERO
                else
                    call calcene2(ene,nhb,ii,dlen,cosa,&
                         rdon,rhyd,racp,tn(i),hbdnrs(nhb)%a(jj),&
                         hbatm1(nhb)%a(jj),hbatm2(nhb)%a(jj),dvdx1,&
                         dvdy1,dvdz1,dvdx2,dvdy2,dvdz2,dvdx3,dvdy3,&
                         dvdz3,X,Y,Z,DX,DY,DZ)
!                    write(*,*) 'GHBPMF',ii,i,jj,ene
                endif
                ehb=ehb+ene !!!ehb energy donr(jj)-acp(i)
                eres(resdnr)=eres(resdnr)+(HALF*ene)
                eres(resacp)=eres(resacp)+(HALF*ene)             
             endif
          enddo
!==========================   GEOM ENERGIES   ============================|
         endif    
       endif chkhbtype
!!!!!!!!!!!!!!!!!!  
       call CHMDEALLOC('epmfmodule.src','ehbpmf','tn',nn,intg=tn)

       enddo dnrloop

         
       endif ifloghb
!!!!!!
     enddo dologhb
     endif nodnracps


     end subroutine ehbpmf

     subroutine calcene(nhb,typ,dist,angl,ene,tdf1,tdf2)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym

     integer,intent(in) :: nhb,typ
     real(chm_real),intent(in) :: dist,angl
     real(chm_real),intent(out) :: tdf1,tdf2,ene

!     local vars
      integer :: ia,id,j,alen,dlen
      real(chm_real) :: amn,amx,dmn,dmx,tu,tt,gd,ga,da,dd
      real(chm_real),dimension(4,4) :: tc4
      real(chm_real),dimension(4) :: ty,ty1,ty2,ty12
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!       
     

!    angle specs
     da=hbpmf(nhb,typ)%binx
     amn=hbpmf(nhb,typ)%minx
     amx=hbpmf(nhb,typ)%maxx-da
     alen=hbpmf(nhb,typ)%map2d%len1

!    dist specs
     dd=hbpmf(nhb,typ)%biny
     dlen=hbpmf(nhb,typ)%map2d%len2
     dmn=hbpmf(nhb,typ)%miny
     dmx=hbpmf(nhb,typ)%maxy-dd

     ene=ZERO
     tt=ZERO
     tu=ZERO
     tdf1=ZERO
     tdf2=ZERO

!
!        checkout bounds and assign (partial) derivatives
!
!           (3)                  (C)                (2)         
!                (am,DM)---------------------(AM,DM)
!                   |==========================|    
!           (D)     |==========================|    (B)
!                   |==========================|       
!   dist         (am,dm)---------------------(AM,dm)
!     ^     (0)                   (A)               (1)   
!     |
!     ---> angle
!
     ia=1
     id=1

     cutoffs: if(angl .le. amx .and. angl .ge. amn .and. &
                 dist .le. dmx .and. dist .ge. dmn ) then
        gd=dist-dmn
        ga=angl-amn
        ia=int(ga/da)+1
        id=int(gd/dd)+1
        call calene2dspl(nhb,typ,ia,id,da,dd,tc4)
        tt=(gd-(dble(id-1)*dd))/dd
        tu=(ga-(dble(ia-1)*da))/da
        do j=4,1,-1
          ene=tt*ene+((tc4(j,4)*tu+tc4(j,3))*tu+tc4(j,2))*tu+tc4(j,1)
          tdf1=tu*tdf1+(THREE*tc4(4,j)*tt+TWO*tc4(3,j))*tt+tc4(2,j) 
          tdf2=tt*tdf2+(THREE*tc4(j,4)*tu+TWO*tc4(j,3))*tu+tc4(j,2) 
        enddo
!                              (A)
     elseif(angl .ge. amn .and. angl .le. amx .and. dist .lt. dmn) then
        ga=angl-amn
        gd=ZERO
        ia=int(ga/da)+1
        id=int(gd/dd)+1
        call calene2dspl(nhb,typ,ia,id,da,dd,tc4)
        tt=(gd-(dble(id-1)*dd))/dd
        tu=(ga-(dble(ia-1)*da))/da
        do j=4,1,-1
          ene=tt*ene+((tc4(j,4)*tu+tc4(j,3))*tu+tc4(j,2))*tu+tc4(j,1)
          tdf1=ZERO
          tdf2=tt*tdf2+(THREE*tc4(j,4)*tu+TWO*tc4(j,3))*tu+tc4(j,2) 
        enddo
!                              (C)
     elseif(angl .ge. amn .and. angl .le. amx .and. dist .gt. dmx) then
        ga=angl-amn
        gd=dmx-dmn
        ia=int(ga/da)+1
        id=int(gd/dd)+1
        call calene2dspl(nhb,typ,ia,id,da,dd,tc4)
        tt=(gd-(dble(id-1)*dd))/dd
        tu=(ga-(dble(ia-1)*da))/da
        do j=4,1,-1
          ene=tt*ene+((tc4(j,4)*tu+tc4(j,3))*tu+tc4(j,2))*tu+tc4(j,1)
          tdf1=ZERO
          tdf2=tt*tdf2+(THREE*tc4(j,4)*tu+TWO*tc4(j,3))*tu+tc4(j,2) 
        enddo
!                               (B)
     elseif(dist .ge. dmn .and. dist .le. dmx .and. angl .gt. amx) then
        ga=amx-amn
        gd=dist-dmn
        ia=int(ga/da)+1
        id=int(gd/dd)+1
        call calene2dspl(nhb,typ,ia,id,da,dd,tc4)
        tt=(gd-(dble(id-1)*dd))/dd
        tu=(ga-(dble(ia-1)*da))/da
        do j=4,1,-1
          ene=tt*ene+((tc4(j,4)*tu+tc4(j,3))*tu+tc4(j,2))*tu+tc4(j,1)
          tdf1=tu*tdf1+(THREE*tc4(4,j)*tt+TWO*tc4(3,j))*tt+tc4(2,j) 
          tdf2=ZERO
        enddo
!                               (D)
     elseif(dist .ge. dmn .and. dist .le. dmx .and. angl .gt. amx) then
        ga=ZERO
        gd=dist-dmn
        ia=int(ga/da)+1
        id=int(gd/dd)+1
        call calene2dspl(nhb,typ,ia,id,da,dd,tc4)
        tt=(gd-(dble(id-1)*dd))/dd
        tu=(ga-(dble(ia-1)*da))/da
        do j=4,1,-1
          ene=tt*ene+((tc4(j,4)*tu+tc4(j,3))*tu+tc4(j,2))*tu+tc4(j,1)
          tdf1=tu*tdf1+(THREE*tc4(4,j)*tt+TWO*tc4(3,j))*tt+tc4(2,j) 
          tdf2=ZERO
        enddo
     elseif(angl .lt. amn .and. dist .lt. dmn) then !!(0)
        ia=1
        id=1
        ene=hbpmf(nhb,typ)%map2d%a(ia,id,1)
        tdf1=ZERO
        tdf2=ZERO        
     elseif(angl .gt. amx .and. dist .lt. dmn) then !!(1)
        ia=alen
        id=1
        ene=hbpmf(nhb,typ)%map2d%a(ia,id,1)
        tdf1=ZERO
        tdf2=ZERO    
     elseif(angl .gt. amx .and. dist .gt. dmx) then !!(2)
        ia=alen
        id=dlen
        ene=hbpmf(nhb,typ)%map2d%a(ia,id,1)
        tdf1=ZERO
        tdf2=ZERO
     elseif(angl .lt. amn .and. dist .gt. dmx) then !!(3)
        ia=1
        id=dlen
        ene=hbpmf(nhb,typ)%map2d%a(ia,id,1)
        tdf1=ZERO
        tdf2=ZERO
     endif cutoffs
       
     end subroutine calcene


    subroutine calene2dspl(nhb,typ,ia,id,da,dd,tc4)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use cmapm
      integer,intent(in) :: nhb,typ,ia,id
      real(chm_real),intent(in)  :: da,dd
      real(chm_real),dimension(4,4),intent(out) :: tc4

!     local vars
      real(chm_real),dimension(4) :: ty,ty1,ty2,ty12
      integer :: j


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
!       Z=f(x,y)        
        ty(1)=hbpmf(nhb,typ)%map2d%a(ia,  id,  1)
        ty(2)=hbpmf(nhb,typ)%map2d%a(ia,  id+1,1)
        ty(3)=hbpmf(nhb,typ)%map2d%a(ia+1,id+1,1)
        ty(4)=hbpmf(nhb,typ)%map2d%a(ia+1,id,  1)
!       dZ/dx
        ty1(1)=hbpmf(nhb,typ)%map2d%a(ia,  id,  2)
        ty1(2)=hbpmf(nhb,typ)%map2d%a(ia,  id+1,2)
        ty1(3)=hbpmf(nhb,typ)%map2d%a(ia+1,id+1,2)
        ty1(4)=hbpmf(nhb,typ)%map2d%a(ia+1,id,  2)
!       dZ/dy
        ty2(1)=hbpmf(nhb,typ)%map2d%a(ia,  id,  3)
        ty2(2)=hbpmf(nhb,typ)%map2d%a(ia,  id+1,3)
        ty2(3)=hbpmf(nhb,typ)%map2d%a(ia+1,id+1,3)
        ty2(4)=hbpmf(nhb,typ)%map2d%a(ia+1,id,  3)
!       dZ/dxdy
        ty12(1)=hbpmf(nhb,typ)%map2d%a(ia,  id  ,4)
        ty12(2)=hbpmf(nhb,typ)%map2d%a(ia,  id+1,4)
        ty12(3)=hbpmf(nhb,typ)%map2d%a(ia+1,id+1,4)
        ty12(4)=hbpmf(nhb,typ)%map2d%a(ia+1,id,  4)
        call GCSCF(ty,ty1,ty2,ty12,dd,da,tc4)
     end subroutine calene2dspl

     subroutine calcene1(ene,nhb,typ,dist,cangl,v1x,v1y,v1z,&
          v2x,v2y,v2z,v3x,v3y,v3z,acp,don,atm1,atm2,DX,DY,DZ)

!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl

     integer,intent(in) :: nhb,typ,acp,don,atm1,atm2
     real(chm_real),intent(in) :: dist,cangl,v1x,v1y,v1z,v2x,v2y,v2z,&
                                  v3x,v3y,v3z
     real(chm_real),intent(inout),dimension(:) :: DX,DY,DZ
     real(chm_real),intent(out)  :: ene

!     local vars
      integer :: j,p,q
      real(chm_real) :: tdf1,tdf2,v2,v3,df1,df2
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
     v2=sqrt(v2x**2+v2y**2+v2z**2)
     v3=v2x*v3x+v2y*v3y+v2z*v3z
     ene=ZERO
     tdf1=ZERO
     tdf2=ZERO
     df1=ZERO
     df2=ZERO

     call calcene(nhb,typ,dist,cangl,ene,tdf1,tdf2)
     ene=hbscale(nhb,typ)*ene

     if(tdf1 .ne. 0 .or. tdf2 .ne. 0) then
      df1=hbscale(nhb,typ)*(tdf1/hbpmf(nhb,typ)%biny) !dist
      df2=hbscale(nhb,typ)*(tdf2/hbpmf(nhb,typ)%binx) !angl

!         dist derv
       DX(acp)=DX(acp)+df1*v1x/dist
       DY(acp)=DY(acp)+df1*v1y/dist
       DZ(acp)=DZ(acp)+df1*v1z/dist
       DX(don)=DX(don)-df1*v1x/dist
       DY(don)=DY(don)-df1*v1y/dist
       DZ(don)=DZ(don)-df1*v1z/dist

!         cosa derv
       DX(acp)=DX(acp)-df2*v2x*v3/(blen(nhb)*(v2**3))+&
               df2*v3x/(blen(nhb)*v2)
       DY(acp)=DY(acp)-df2*v2y*v3/(blen(nhb)*(v2**3))+&
              df2*v3y/(blen(nhb)*v2)
       DZ(acp)=DZ(acp)-df2*v2z*v3/(blen(nhb)*(v2**3))+&
               df2*v3z/(blen(nhb)*v2)

       DX(atm1)=DX(atm1)+df2*v2x*v3/(blen(nhb)*(v2**3))*fa(nhb)-&
                df2/(blen(nhb)*v2)*(v2x+v3x)*fa(nhb)
       DY(atm1)=DY(atm1)+df2*v2y*v3/(blen(nhb)*(v2**3))*fa(nhb)-&
                df2/(blen(nhb)*v2)*(v2y+v3y)*fa(nhb)
       DZ(atm1)=DZ(atm1)+df2*v2z*v3/(blen(nhb)*(v2**3))*fa(nhb)-&
               df2/(blen(nhb)*v2)*(v2z+v3z)*fa(nhb)


       DX(atm2)=DX(atm2)+df2*v2x*v3/(blen(nhb)*(v2**3))*fb(nhb)-&
                df2/(blen(nhb)*v2)*(v2x+v3x)*fb(nhb)
       DY(atm2)=DY(atm2)+df2*v2y*v3/(blen(nhb)*(v2**3))*fb(nhb)-&
                df2/(blen(nhb)*v2)*(v2y+v3y)*fb(nhb)
       DZ(atm2)=DZ(atm2)+df2*v2z*v3/(blen(nhb)*(v2**3))*fb(nhb)-&
                df2/(blen(nhb)*v2)*(v2z+v3z)*fb(nhb)

       DX(don)=DX(don)+df2*v2x*v3/(blen(nhb)*(v2**3))*&
              (ONE-fa(nhb)-fb(nhb))-df2/(blen(nhb)*v2)*&
              (v3x*(ONE-fa(nhb)-fb(nhb))-(fa(nhb)+fb(nhb))*v2x)
       DY(don)=DY(don)+df2*v2y*v3/(blen(nhb)*(v2**3))*&
              (ONE-fa(nhb)-fb(nhb))-df2/(blen(nhb)*v2)*&
              (v3y*(ONE-fa(nhb)-fb(nhb))-(fa(nhb)+fb(nhb))*v2y)
       DZ(don)=DZ(don)+df2*v2z*v3/(blen(nhb)*(v2**3))*&
              (ONE-fa(nhb)-fb(nhb))-df2/(blen(nhb)*v2)*&
              (v3z*(ONE-fa(nhb)-fb(nhb))-(fa(nhb)+fb(nhb))*v2z)
               
     endif

     end subroutine calcene1


     subroutine calcene2(ene,nhb,typ,dist,cangl,vd,vh,va,&
                        acp,don,atm1,atm2,dhx1,dhy1,dhz1,&
                        dhx2,dhy2,dhz2,dhx3,dhy3,dhz3,X,Y,Z,DX,DY,DZ)
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl
  use vector
       
     integer,intent(in) :: nhb,typ,acp,don,atm1,atm2
     real(chm_real),intent(in) :: dist,cangl,vd(3),vh(3),va(3),dhx1(3),&
                                  dhy1(3),dhz1(3),dhx2(3),dhy2(3),&
                                  dhz2(3),dhx3(3),dhy3(3),dhz3(3)
     real(chm_real),intent(inout),dimension(:) :: DX,DY,DZ
     real(chm_real),intent(in),dimension(:)    :: X,Y,Z
     real(chm_real),intent(out)  :: ene

!     local vars
      integer :: i
      real(chm_real) :: tdf1,tdf2,df1,df2,rm(3),rn(3),tt,da(3),&
                  drm,drn,nrn(3),nrm(3),t1,t2,&
                  tmxd(3),tmyd(3),tmzd(3),tmx2(3),tmy2(3),tmz2(3),&
                  tmx3(3),tmy3(3),tmz3(3),tmxa(3),tmya(3),tmza(3),&
                  tnxd(3),tnyd(3),tnzd(3),tnx2(3),tny2(3),tnz2(3),&
                  tnx3(3),tny3(3),tnz3(3),tnxa(3),tnya(3),tnza(3),&
      dfrmxd(3),dfrmyd(3),dfrmzd(3),dfrmx2(3),dfrmy2(3),dfrmz2(3),&
      dfrmx3(3),dfrmy3(3),dfrmz3(3),dfrmxa(3),dfrmya(3),dfrmza(3),&
      dfrnxd(3),dfrnyd(3),dfrnzd(3),dfrnx2(3),dfrny2(3),dfrnz2(3),&
      dfrnx3(3),dfrny3(3),dfrnz3(3),dfrnxa(3),dfrnya(3),dfrnza(3),&
      dcosqd(3),dcosqa(3),dcosq2(3),dcosq3(3)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!     
     
     ene=ZERO
     tdf1=ZERO
     tdf2=ZERO
     df1=ZERO
     df2=ZERO
     call calcene(nhb,typ,dist,cangl,ene,tdf1,tdf2)
     ene=hbscale(nhb,typ)*ene
!     write(*,*) 'CE2',dist,cangl,ene,hbscale(nhb,typ)


     splderv:if(tdf1 .ne. 0 .or. tdf2 .ne. 0) then
      df1=hbscale(nhb,typ)*(tdf1/hbpmf(nhb,typ)%biny) !dist
      df2=hbscale(nhb,typ)*(tdf2/hbpmf(nhb,typ)%binx) !angl
!         dist derv
       da=(/X(acp)-X(don),Y(acp)-Y(don),Z(acp)-Z(don)/)
       DX(acp)=DX(acp)+df1*da(1)/dist
       DY(acp)=DY(acp)+df1*da(2)/dist
       DZ(acp)=DZ(acp)+df1*da(3)/dist
       DX(don)=DX(don)-df1*da(1)/dist
       DY(don)=DY(don)-df1*da(2)/dist
       DZ(don)=DZ(don)-df1*da(3)/dist

!         cosa derv
      
       rm=vh-vd
       rn=va-vh
       drm=sqrt(rm(1)**2+rm(2)**2+rm(3)**2)
       drn=sqrt(rn(1)**2+rn(2)**2+rn(3)**2)
       nrm=rm/drm
       nrn=rn/drn

!     
!       dRM     drH    dVD
!       --- =   ---  - ---       
!       dqk     dqk    dqk
!
        tmxd=DHx1-(/1.0,0.0,0.0/)
        tmyd=DHy1-(/0.0,1.0,0.0/)
        tmzd=DHz1-(/0.0,0.0,1.0/)
        tmx2=DHx2
        tmy2=DHy2
        tmz2=DHz2
        tmx3=DHx3
        tmy3=DHy3
        tmz3=DHz3
        tmxa=ZERO
        tmya=ZERO
        tmza=ZERO


!      
!      d|RM|         dRM
!      ----  = ( RM. --   )/|RM|
!      dxk           dxk
!
!       ^
!      dRM            dRM       d|RM|
!      ---   = { |RM| --  -  RM ---- }/|RM|^2
!      dxk            dxk       dxk
!
!
     

      call DOTPR(rm,tmxd,3,tt)
      dfrmxd=( (drm*tmxd) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmyd,3,tt)
      dfrmyd=( (drm*tmyd) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmzd,3,tt)
      dfrmzd=( (drm*tmzd) - (rm*tt)/drm)/(drm**2)

      call DOTPR(rm,tmx2,3,tt)
      dfrmx2=( (drm*tmx2) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmy2,3,tt)
      dfrmy2=( (drm*tmy2) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmz2,3,tt)
      dfrmz2=( (drm*tmz2) - (rm*tt)/drm)/(drm**2)

      call DOTPR(rm,tmx3,3,tt)
      dfrmx3=( (drm*tmx3) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmy3,3,tt)
      dfrmy3=( (drm*tmy3) - (rm*tt)/drm)/(drm**2)
      call DOTPR(rm,tmz3,3,tt)
      dfrmz3=( (drm*tmz3) - (rm*tt)/drm)/(drm**2)

      dfrmxa=ZERO
      dfrmya=ZERO
      dfrmza=ZERO


!    
!       dRN     dVA    drH
!       --- =   ---  - ---       
!       dqk     dqk    dqk
!
        tnxd=-DHx1
        tnyd=-DHy1
        tnzd=-DHz1
        tnx2=-DHx2
        tny2=-DHy2
        tnz2=-DHz2
        tnx3=-DHx3
        tny3=-DHy3
        tnz3=-DHz3
        tnxa=(/1.0,0.0,0.0/)
        tnya=(/0.0,1.0,0.0/)
        tnza=(/0.0,0.0,1.0/)
!       
!      d|RN|         dRN
!      ----  =   RN. --
!      dxk           dxk
!
!       ^
!      dRN            dRN        d|RN|
!      ---   = { |RN| --  -  RN  ---- }/|RN|^2
!      dxk            dxk        dxk
!
!

      call DOTPR(rn,tnxd,3,tt)
      dfrnxd=( (drn*tnxd) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnyd,3,tt)
      dfrnyd=( (drn*tnyd) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnzd,3,tt)
      dfrnzd=( (drn*tnzd) - (rn*tt/drn))/(drn**2)

      call DOTPR(rn,tnx2,3,tt)
      dfrnx2=( (drn*tnx2) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tny2,3,tt)
      dfrny2=( (drn*tny2) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnz2,3,tt)
      dfrnz2=(  (drn*tnz2) - (rn*tt/drn))/(drn**2)

      call DOTPR(rn,tnx3,3,tt)
      dfrnx3=( (drn*tnx3) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tny3,3,tt)
      dfrny3=( (drn*tny3) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnz3,3,tt)
      dfrnz3=( (drn*tnz3) - (rn*tt/drn))/(drn**2)

      call DOTPR(rn,tnxa,3,tt)
      dfrnxa=( (drn*tnxa) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnya,3,tt)
      dfrnya=( (drn*tnya) - (rn*tt/drn))/(drn**2)
      call DOTPR(rn,tnza,3,tt)
      dfrnza=( (drn*tnza) - (rn*tt/drn))/(drn**2)

!                      ^           ^
! dcosth        ^     drn         drm   ^
! ------    =   rm . -----   +   ----- .rn
!  dqk                dqk         dqk
!

      call DOTPR(nrm,dfrnxd,3,t1)
      call DOTPR(dfrmxd,nrn,3,t2)
      dcosqd(1)=t1+t2
      call DOTPR(nrm,dfrnyd,3,t1)
      call DOTPR(dfrmyd,nrn,3,t2)
      dcosqd(2)=t1+t2
      call DOTPR(nrm,dfrnzd,3,t1)
      call DOTPR(dfrmzd,nrn,3,t2)
      dcosqd(3)=t1+t2


      call DOTPR(nrm,dfrnxa,3,t1)
      call DOTPR(dfrmxa,nrn,3,t2)
      dcosqa(1)=t1+t2
      call DOTPR(nrm,dfrnya,3,t1)
      call DOTPR(dfrmya,nrn,3,t2)
      dcosqa(2)=t1+t2
      call DOTPR(nrm,dfrnza,3,t1)
      call DOTPR(dfrmza,nrn,3,t2)
      dcosqa(3)=t1+t2


      call DOTPR(nrm,dfrnx3,3,t1)
      call DOTPR(dfrmx3,nrn,3,t2)
      dcosq3(1)=t1+t2
      call DOTPR(nrm,dfrny3,3,t1)
      call DOTPR(dfrmy3,nrn,3,t2)
      dcosq3(2)=t1+t2
      call DOTPR(nrm,dfrnz3,3,t1)
      call DOTPR(dfrmz3,nrn,3,t2)
      dcosq3(3)=t1+t2


      call DOTPR(nrm,dfrnx2,3,t1)
      call DOTPR(dfrmx2,nrn,3,t2)
      dcosq2(1)=t1+t2
      call DOTPR(nrm,dfrny2,3,t1)
      call DOTPR(dfrmy2,nrn,3,t2)
      dcosq2(2)=t1+t2
      call DOTPR(nrm,dfrnz2,3,t1)
      call DOTPR(dfrmz2,nrn,3,t2)
      dcosq2(3)=t1+t2
!
!
!
      dx(don)=dx(don)+df2*dcosqd(1)
      dy(don)=dy(don)+df2*dcosqd(2)
      dz(don)=dz(don)+df2*dcosqd(3)

      dx(acp)=dx(acp)+df2*dcosqa(1)
      dy(acp)=dy(acp)+df2*dcosqa(2)
      dz(acp)=dz(acp)+df2*dcosqa(3)

      dx(atm1)=dx(atm1)+df2*dcosq2(1)
      dy(atm1)=dy(atm1)+df2*dcosq2(2)
      dz(atm1)=dz(atm1)+df2*dcosq2(3)

      dx(atm2)=dx(atm2)+df2*dcosq3(1)
      dy(atm2)=dy(atm2)+df2*dcosq3(2)
      dz(atm2)=dz(atm2)+df2*dcosq3(3)

     endif splderv  
     end subroutine calcene2



     subroutine updtn(ndt,steps)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use coord
  use contrl
  use memory
  use chutil,only:getres

     integer,intent(in) :: ndt,steps

!     local vars
      integer :: ndnr,nacp,ii,jj,kk,ll,mm,pp
      real(chm_real) :: x1,y1,z1,x2,y2,z2,dis
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      
      updtloop: if(steps .gt. 0 .and. (mod(steps,upfrn) .eq. 0)) then
          ndnr=dtdnrs(ndt)%len
          nacp=dtacps(ndt)%len
          do ii=1,ndnr
             kk=dtdnrs(ndt)%a(ii)
             pp=GETRES(kk,IBASE,NRES)
             x1=X(kk)
             y1=Y(kk)
             z1=Z(kk)
             do jj=1,nacp
                 ll=dtacps(ndt)%a(jj)
                 mm=GETRES(ll,IBASE,NRES)
                 x2=X(ll)
                 y2=Y(ll)
                 z2=Z(ll)
                 if( (mm .ge. (pp+4)) .or. (mm .le. (pp-4)) ) then
                      dis=sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
                      if(dis .le. dtcutf(ndt)) then
                          dtnein(ndt)%list(ii)%a(jj)=ll
                      endif
                 endif
               enddo      
            enddo
      endif updtloop
     end subroutine updtn


     subroutine uphbn(nhb,steps)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use coord
  use contrl
  use memory
  use chutil,only:getres

     integer,intent(in) :: nhb,steps
!     local vars
      integer :: ndnr,nacp,ii,jj,kk,ll,mm,pp,qq
      real(chm_real) :: x1,y1,z1,x2,y2,z2,dis
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      uphbloop: if(steps .gt. 0 .and. (mod(steps,upfrn) .eq. 0)) then
          ndnr=hbdnrs(nhb)%len
          nacp=hbacps(nhb)%len
          do ii=1,ndnr
             kk=hbdnrs(nhb)%a(ii)
             pp=GETRES(kk,IBASE,NRES)
             x1=X(kk)
             y1=Y(kk)
             z1=Z(kk)
             do jj=1,nacp
                 ll=hbacps(nhb)%a(jj)
                 mm=GETRES(ll,IBASE,NRES)
                 x2=X(ll)
                 y2=Y(ll)
                 z2=Z(ll)
                 if( (mm .ge. (pp+6)) .or. (mm .le. (pp-6)) ) then
                      dis=sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
                      if(dis .le. hbcutf(nhb)) then
                          hbnein(nhb)%list(ii)%a(jj)=ll
                          qq=qq+1
                      endif
                 endif
               enddo      
            enddo
      endif uphbloop

     end subroutine uphbn

     subroutine totepmf(ep1d,ep2d,X,Y,Z,DX,DY,DZ)
!  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl
#if KEY_PARALLEL==1
  use parallel
#endif 

      real(chm_real),intent(inout) :: ep1d,ep2d
      real(chm_real),dimension(natom),intent(inout)::X,Y,Z,DX,DY,DZ

!     local vars
      integer :: i,mdt,mhb,nd,na
      
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      ep1d=ZERO
      ep2d=ZERO

#if KEY_PARALLEL==1
      ecalcparall: if(MYNODP .eq. 1) then
         if(qepmf) then
#else /**/
         if(qepmf) then
#endif 

     
           if(.not. qmesg) then 
               mdt=0
               do i=1,ndtpmf
                 nd=dtdnrs(i)%len
                 na=dtacps(i)%len
                 if(nd .ne. 0 .and. na .ne. 0) mdt=mdt+1
               enddo
               mhb=0
               do i=1,nhbpmf
                 nd=hbdnrs(i)%len
                 na=hbacps(i)%len
                 if(nd .ne. 0 .and. na .ne. 0) mhb=mhb+1
               enddo
              
               write(OUTU,'(/,a,i5)')"EPMF> NO. OF READ  DIST PMFS =",ndtpmf
               write(OUTU,'(a,i5)')  "EPMF> NO. OF VALID DIST PMFS =",mdt
               write(OUTU,'(a,i5)')  "EPMF> NO. OF READ  HB   PMFS =",nhbpmf
               write(OUTU,'(a,i5)')  "EPMF> NO. OF VALID HB   PMFS =",mhb
               write(OUTU,'(a,i5,/)')"EPMF> NEIG. UPDATE FREQUENCY =",upfrn
               qmesg=.true.
           endif

            if(ndtpmf .ne. 0) then
              do i=1,ndtpmf
                 if(DYNAMQ .and. logdt(i,5)) call updtn(i,mdstep)
                 edt(i)=ZERO
                 call edtpmf(edt(i),i,X,Y,Z,DX,DY,DZ)
                 ep1d=ep1d+edt(i)
              enddo
            endif
!
!
            if(nhbpmf .ne. 0) then
               do i=1,nhbpmf
                  if(DYNAMQ .and. loghb(i,5)) call uphbn(i,mdstep)
                  ehb(i)=ZERO
                  call ehbpmf(ehb(i),i,X,Y,Z,DX,DY,DZ)
                  ep2d=ep2d+ehb(i)
                enddo
            endif
         endif  !qepmf

#if KEY_PARALLEL==1
      endif ecalcparall
#endif 
     end subroutine totepmf

     subroutine selctatoms(nr,atyp,apfx,ainx)
!
!   Return index of atom of type 'atyp' of prev, curr and next residues whence
!   apfx is '-', ' ', or '+'
!
!  use select
  use stream
  use number
  use string
  use psf
  use chutil,only:getres
  use contrl
     integer,intent(in)  :: nr
     integer,intent(out) :: ainx
     character(len=1),intent(in) :: apfx   
     character(len=8),intent(in) :: atyp

!    local vars
     integer :: i,n,mm,astat,ta,tc,tp,tn,nc,np,nn
!=====================================================================!
     ainx=-1

     n=0
     do i=1,natom
        if(atype(i) .eq. atyp) then
           mm=GETRES(i,ibase,nres)
           if(mm .eq. nr) n=n+1
        endif
     enddo

     if(n .ne. 0) then
        if(n .gt. 1) write(OUTU,*)&
         'EPMF:SELCTA> More than one atom for current selection'
      
      nc=0
      np=0
      nn=0
      tn=-1
      tp=-1
      tc=-1
      do i=1,natom
         if(atype(i) .eq. atyp) then
           mm=GETRES(i,ibase,nres)
           if(mm .eq. nr-1) then
              np=np+1
              tp=i
              if(np .gt. 1 .and. PRNLEV .gt. 5) write(OUTU,*)&
              'EPMF:SELCTA> More than one atom for -atom selection'
           elseif(mm .eq. nr) then
              nc=nc+1
              tc=i
              if(nc .gt. 1 .and. PRNLEV .gt. 5) write(OUTU,*)&
              'EPMF:SELCTA> More than one atom for atom selection'
           elseif(mm .eq. nr+1) then
              nn=nn+1
              tn=i
              if(nn .gt. 1 .and. PRNLEV .gt. 5) write(OUTU,*)&
              'EPMF:SELCTA> More than one atom for +atom selection'             
           else
              continue
           endif
         endif
       enddo

       if(apfx .eq. '+') then
         ainx=tn
       elseif(apfx .eq. '-') then
         ainx=tp
       else
        ainx=tc
       endif

      else  !noselections
        ainx=-1
      endif

      end subroutine selctatoms

      subroutine mtxmult(mat,w)
      real(chm_real),intent(out) :: w(3)
      real(chm_real),intent(in)  :: mat(4,4)
      real(chm_real) :: oldvect(4),newvect(4)
      integer :: i,j
      oldvect(1)=w(1)
      oldvect(2)=w(2)
      oldvect(3)=w(3)
      oldvect(4)=1
      newvect=0.0
      do j=1,4
        do i=1,4
          newvect(j)= newvect(j)+oldvect(i)*mat(i,j)
        enddo
      enddo
      w=(/ newvect(1),newvect(2),newvect(3)/)
      end subroutine mtxmult

     subroutine nmcrosprod2(v1,v2,v3)
      use vector
      real(chm_real),dimension(3),intent(in)  :: v1,v2
      real(chm_real),dimension(3),intent(out) :: v3
      real(chm_real) :: v
      call CROSS3(v1,v2,v3)
      call NORMALL(v3,3)
     end subroutine nmcrosprod2


     subroutine getcosangl3(v1,v2,v3,d)  
!         1---->2----->3  r12.r23
!  use select
      real(chm_real),dimension(3),intent(in) :: v1,v2,v3
      real(chm_real),intent(out)  :: d
      real(chm_real),dimension(3) :: v12,v23
      real(chm_real) :: d12,d23
      v12=(/v2(1)-v1(1), v2(2)-v1(2), v2(3)-v1(3)/)
      v23=(/v3(1)-v2(1), v3(2)-v2(2), v3(3)-v2(3)/)
      d12=sqrt(v12(1)**2+v12(2)**2+v12(3)**2)
      d23=sqrt(v23(1)**2+v23(2)**2+v23(3)**2)
      d=((v12(1)*v23(1))+(v12(2)*v23(2))+(v12(3)*v23(3)))/(d12*d23)
     end subroutine getcosangl3

     subroutine getcosangl2(v1,v2,v3,d)
!         1<----2----->3   r21.r23     
!  use select
      real(chm_real),dimension(3),intent(in) :: v1,v2,v3
      real(chm_real),intent(out)  :: d
      real(chm_real),dimension(3) :: v21,v23
      real(chm_real) :: d21,d23
      v21=(/v1(1)-v2(1), v1(2)-v2(2), v1(3)-v2(3)/)
      v23=(/v3(1)-v2(1), v3(2)-v2(2), v3(3)-v2(3)/)
      d21=sqrt(v21(1)**2+v21(2)**2+v21(3)**2)
      d23=sqrt(v23(1)**2+v23(2)**2+v23(3)**2)
      d=((v21(1)*v23(1))+(v21(2)*v23(2))+(v21(3)*v23(3)))/(d21*d23)
      if(d .le. -1.0) d=-1.0
      if(d .ge.  1.0) d=1.0
     end subroutine

       subroutine nmcrosprod(x1,y1,z1,x2,y2,z2,x3,y3,z3)
!     Returns the normal of the cross product(surface normal)
       real(chm_real),intent(in)  :: x1,y1,z1,x2,y2,z2
       real(chm_real),intent(out) :: x3,y3,z3
       real(chm_real) ::  dis,x,y,z
       x=y1*z2 - y2*z1
       y=z1*x2 - z2*x1
       z=x1*y2 - x2*y1
       dis=sqrt((x*x)+(y*y)+(z*z))
       x3=x/dis
       y3=y/dis
       z3=z/dis
       end subroutine nmcrosprod
 
     subroutine viewat(m,invm,p1,p2,p3)
!
!      Based on routine rebuild from MMTSB toolset
!      Transforms p1,p2,p3 / p1=(0,0,0) p2=(0,0,z2') p3=(0,y3',z3')
!
      real(chm_real),intent(in)  :: p1(4),p2(4),p3(4)
      real(chm_real),intent(out) :: m(4,4),invm(4,4)
      real(chm_real) :: d12,p120,p121,p122,p130,p131,p132
      
      p120=p2(1)-p1(1)
      p121=p2(2)-p1(2)
      p122=p2(3)-p1(3)
      p130=p3(1)-p1(1)
      p131=p3(2)-p1(2)
      p132=p3(3)-p1(3)
      d12=sqrt((p120*p120)+(p121*p121)+(p122*p122))
      m=0
      invm=0
      invm(3,1)=p120/d12
      invm(3,2)=p121/d12
      invm(3,3)=p122/d12
      m(1,3)=invm(3,1)
      m(2,3)=invm(3,2)
      m(3,3)=invm(3,3)

      call nmcrosprod(p130,p131,p132,p120,p121,p122,&
                      m(1,1),m(2,1),m(3,1))
      invm(1,1)=m(1,1)
      invm(1,2)=m(2,1)
      invm(1,3)=m(3,1)
      m(1,4)=0.
      m(2,4)=0.
      m(3,4)=0.
      invm(1,4)=M(1,4)
      invm(2,4)=M(2,4)
      invm(3,4)=M(3,4)
 
      call nmcrosprod(m(1,3),m(2,3),m(3,3),m(1,1),m(2,1),m(3,1),&
                      m(1,2),m(2,2),m(3,2) )
      invm(2,1)=m(1,2)
      invm(2,2)=m(2,2)
      invm(2,3)=m(3,2)
      invm(4,1)=p1(1)
      invm(4,2)=p1(2)
      invm(4,3)=p1(3)
      invm(4,4)=1.0
      m(4,1)=-p1(1)*m(1,1)-p1(2)*m(2,1)-p1(3)*m(3,1)
      m(4,2)=-p1(1)*m(1,2)-p1(2)*m(2,2)-p1(3)*m(3,2)
      m(4,3)=-p1(1)*m(1,3)-p1(2)*m(2,3)-p1(3)*m(3,3)
      m(4,4)=1.0
      end subroutine viewat


      subroutine buildatom (bnd,ang,dih,v1,v2,v3,v0,&
                 dv0dx1,dv0dy1,dv0dz1,dv0dx2,dv0dy2,dv0dz2,&
                 dv0dx3,dv0dy3,dv0dz3)
!
!                <0> <-constructed
!             b   .    
!                 .
!                [1]
!     theta,phi /    \
!              /      \
!           [2]        [3] 
!

  use consta
  use number
  use stream
  use vector
       real(chm_real),intent(in)  :: v1(3),v2(3),v3(3)
       real(chm_real),intent(in)  :: bnd,ang,dih
       real(chm_real),intent(out) :: v0(3),dv0dx1(3),dv0dx2(3),dv0dx3(3),&
        dv0dy1(3),dv0dy2(3),dv0dy3(3),dv0dz1(3),dv0dz2(3),dv0dz3(3) 
       
       integer :: i
       real(chm_real) :: mat(4,4),invmat(4,4),angr,dihr
       real(chm_real) :: p1(4),p2(4),p3(4),tv(4),u(3),v(3),w(3),tv0(3)
       real(chm_real) :: t1,t2,t3,v12(3),v13(3),uvec(3)

!          w derivatives        
       real(chm_real) :: dfwx1(3),dfwy1(3),dfwz1(3),dfwx2(3),dfwy2(3),&
                       dfwz2(3),dfwq1(3),dfwq2(3),d12      
!          u derivatives
       real(chm_real) :: dfux1(3),dfuy1(3),dfuz1(3),dfux2(3),dfuy2(3),&
                       dfuz2(3),dfux3(3),dfuy3(3),dfuz3(3),tux1(3),&
                       tuy1(3),tuz1(3),tux2(3),tuy2(3),tuz2(3),&
                       tux3(3),tuy3(3),tuz3(3),tuq1(3),tuq2(3),&
                       tuq3(3),r1(3),r2(3),ddx(3),ddy(3),ddz(3),&
                       tx,ty,tz,du,d2u,dfuq1(3),dfuq2(3),dfuq3(3)
!         v derivatives
       real(chm_real) :: dfvx1(3),dfvy1(3),dfvz1(3),dfvx2(3),dfvy2(3),&
                       dfvz2(3),dfvx3(3),dfvy3(3),dfvz3(3),dfvq1(3),&
                       dfvq2(3),dfvq3(3)
 
       angr=DEGRAD*ang
       dihr=DEGRAD*dih
!      this is the spherical unit vector which later gets transformed     
       tv(1)=bnd*sin(angr)*sin(dihr)
       tv(2)=bnd*sin(angr)*cos(dihr)
       tv(3)=bnd*cos(angr)
       tv(4)=1
!      this is copy of the above which will be used in derivatives
       tv0(1)=tv(1)
       tv0(2)=tv(2)
       tv0(3)=tv(3)
       p1(1)=v1(1)
       p1(2)=v1(2)
       p1(3)=v1(3)
       p2(1)=v2(1)
       p2(2)=v2(2)
       p2(3)=v2(3)
       p3(1)=v3(1)
       p3(2)=v3(2)
       p3(3)=v3(3)
       p1(4)=1.
       p2(4)=1.
       p3(4)=1.
       call viewat(mat,invmat,p1,p2,p3)
       u=(/mat(1,1),mat(2,1),mat(3,1)/)
       v=(/mat(1,2),mat(2,2),mat(3,2)/)
       w=(/mat(1,3),mat(2,3),mat(3,3)/)
       call mtxmult(mat,p1)
       call mtxmult(mat,p2)
       call mtxmult(mat,p3)
       uvec=ZERO
       call CROSS3(v3-v1,w,uvec)
       call mtxmult(invmat,tv)
       v0=(/ tv(1), tv(2), tv(3) /)



!      get derivatives ...
!--------------------------------------------------------------------------
!      First W wrt q1,q2
       
       v12=ZERO
       t1=ZERO
       t2=ZERO
       t3=ZERO

       v12=v2-v1
       d12=sqrt(v12(1)**2+v12(2)**2+v12(3)**2)
!      Derivatives of w wrt r2       
       t1=(1/d12)-((v12(1)**2)/(d12**3))
       t2=-(v12(1)*v12(2))/(d12**3)
       t3=-(v12(1)*v12(3))/(d12**3)
       DFwx2=(/t1, t2, t3 /)
       t1=-(v12(2)*v12(1))/(d12**3)
       t2=(1/d12)-((v12(2)**2)/(d12**3))       
       t3=-(v12(2)*v12(3))/(d12**3)
       DFwy2=(/t1, t2, t3 /)
       t1=-(v12(3)*v12(1))/(d12**3)
       t2=-(v12(3)*v12(2))/(d12**3)
       t3=(1/d12)-((v12(3)**2)/(d12**3))
       DFwz2=(/t1, t2, t3 /)
       DFwq2=DFwx2+DFwy2+DFwz2

!      Derivatives of w wrt r1
       t1=(-1/d12)+((v12(1)**2)/(d12**3))
       t2=(v12(1)*v12(2))/(d12**3)
       t3=(v12(1)*v12(3))/(d12**3)
       DFwx1=(/t1, t2, t3 /)
       t1=(v12(2)*v12(1))/(d12**3)
       t2=(-1/d12)+((v12(2)**2)/(d12**3))       
       t3=(v12(2)*v12(3))/(d12**3)
       DFwy1=(/t1, t2, t3 /)
       t1=(v12(3)*v12(1))/(d12**3)
       t2=(v12(3)*v12(2))/(d12**3)
       t3=(-1/d12)+((v12(3)**2)/(d12**3))
       DFwz1=(/t1, t2, t3 /)
       DFwq1=DFwx1+DFwy1+DFwz1
!--------------------------------------------------------------------------
!       Derivatives of U
!  
!    First VEC (U)
!
!    DU         Dr13               Dw      
!    --   =     --- X w  +   r13 X ---   
!    Dxk        Dxk                Dxk     
!
!   
!    DU         DU      DU     DU
!    --    =    ---  +  --  +  --
!    Dqk        Dxk     Dyk    Dzk
!
!    k=1,2,3
!-------------------------------------------------------------------------
       tuq1=ZERO
       tuq2=ZERO
       tuq3=ZERO
       r1=ZERO
       r2=ZERO
       tux1=ZERO
       tuy1=ZERO
       tuz1=ZERO
       tux2=ZERO
       tuy2=ZERO
       tuz2=ZERO
       tux3=ZERO
       tuy3=ZERO
       tuz3=ZERO 
 
       ddx=(/1.0, 0.0, 0.0/)
       ddy=(/0.0, 1.0, 0.0/)
       ddz=(/0.0, 0.0, 1.0/)

!      du/dq1       
       v13=v3-v1
       call CROSS3(-ddx,w,r1)
       call CROSS3(v13,dfwx1,r2)
       tux1=r1+r2
       call CROSS3(-ddy,w,r1)
       call CROSS3(v13,dfwy1,r2)
       tuy1=r1+r2
       call CROSS3(-ddz,w,r1)
       call CROSS3(v13,dfwz1,r2)
       tuz1=r1+r2
       tuq1=tux1+tuy1+tuz1

!      du/dq2
       call CROSS3(v13,dfwx2,tux2)
       call CROSS3(v13,dfwy2,tuy2)
       call CROSS3(v13,dfwz2,tuz2)
       tuq2=tux2+tuy2+tuz2

!      du/dq3
       call CROSS3(ddx,w,tux3)
       call CROSS3(ddy,w,tuy3)
       call CROSS3(ddz,w,tuz3)
       tuq3=tux3+tuy3+tuz3
!       
!      d|u|           du
!      ----  =   ( u. --  )/|u|
!      dxk            dxk
!
!       ^
!      du            du       d|u|
!      ---   = { |u| --  -  u ---- }/|u|^2
!      dxk           dxk      dxk
!
!
       dfuq1=ZERO
       dfuq2=ZERO
       dfuq3=ZERO
       du=sqrt(uvec(1)**2+uvec(2)**2+uvec(3)**2)
       d2u=du**2

       call DOTPR(uvec,tux1,3,tx)
       dfux1=( (du*tux1) - (uvec*(tx/du)) )/d2u
       call DOTPR(uvec,tuy1,3,ty)
       dfuy1=( (du*tuy1) - (uvec*(ty/du)) )/d2u
       call DOTPR(uvec,tuz1,3,tz)
       dfuz1=( (du*tuz1) - (uvec*(tz/du)) )/d2u
       dfuq1=dfux1+dfuy1+dfuz1

       call DOTPR(uvec,tux2,3,tx)
       dfux2=( (du*tux2) - (uvec*(tx/du)) )/d2u
       call DOTPR(uvec,tuy2,3,ty)
       dfuy2=( (du*tuy2) - (uvec*(ty/du)) )/d2u
       call DOTPR(uvec,tuz2,3,tz)
       dfuz2=( (du*tuz2) - (uvec*(tz/du)) )/d2u
       dfuq2=dfux2+dfuy2+dfuz2

       call DOTPR(uvec,tux3,3,tx)
       dfux3=( (du*tux3) - (uvec*(tx/du)) )/d2u
       call DOTPR(uvec,tuy3,3,ty)
       dfuy3=( (du*tuy3) - (uvec*(ty/du)) )/d2u
       call DOTPR(uvec,tuz3,3,tz)
       dfuz3=( (du*tuz3) - (uvec*(tz/du)) )/d2u
       dfuq3=dfux3+dfuy3+dfuz3

!------------------------------------------------
! Finally v
!         
!    ^     ^            ^
!   dv    dw   ^   ^   du
!   --  = -- X u + w X --
!   dxk   dxk          dxk
!
!-------------------------------------------------
       dfvq1=ZERO
       dfvq2=ZERO
       dfvq3=ZERO

       r1=ZERO
       r2=ZERO
       call CROSS3(dfwx1,u,r1)
       call CROSS3(w,dfux1,r2)
       dfvx1=r1+r2
       call CROSS3(dfwy1,u,r1)
       call CROSS3(w,dfuy1,r2)
       dfvy1=r1+r2
       call CROSS3(dfwz1,u,r1)
       call CROSS3(w,dfuz1,r2)
       dfvz1=r1+r2
       dfvq1=dfvx1+dfvy1+dfvz1

       r1=ZERO
       r2=ZERO
       call CROSS3(dfwx2,u,r1)
       call CROSS3(w,dfux2,r2)
       dfvx2=r1+r2
       call CROSS3(dfwy2,u,r1)
       call CROSS3(w,dfuy2,r2)
       dfvy2=r1+r2
       call CROSS3(dfwz2,u,r1)
       call CROSS3(w,dfuz2,r2)
       dfvz2=r1+r2
       dfvq2=dfvx2+dfvy2+dfvz2

       r1=ZERO
       r2=ZERO
       call CROSS3(w,dfux3,r2)
       dfvx3=r1+r2
       call CROSS3(w,dfuy3,r2)
       dfvy3=r1+r2
       call CROSS3(w,dfuz3,r2)
       dfvz3=r1+r2
       dfvq3=dfvx3+dfvy3+dfvz3
!
!      Dv0/Dq1
!
       tx=tv0(1)*dfux1(1)+tv0(2)*dfvx1(1)+tv0(3)*dfwx1(1)+1
       ty=tv0(1)*dfux1(2)+tv0(2)*dfvx1(2)+tv0(3)*dfwx1(2)
       tz=tv0(1)*dfux1(3)+tv0(2)*dfvx1(3)+tv0(3)*dfwx1(3)
       dv0dx1=(/tx, ty, tz/)
       tx=tv0(1)*dfuy1(1)+tv0(2)*dfvy1(1)+tv0(3)*dfwy1(1)
       ty=tv0(1)*dfuy1(2)+tv0(2)*dfvy1(2)+tv0(3)*dfwy1(2)+1
       tz=tv0(1)*dfuy1(3)+tv0(2)*dfvy1(3)+tv0(3)*dfwy1(3)
       dv0dy1=(/tx, ty, tz/)
       tx=tv0(1)*dfuz1(1)+tv0(2)*dfvz1(1)+tv0(3)*dfwz1(1)
       ty=tv0(1)*dfuz1(2)+tv0(2)*dfvz1(2)+tv0(3)*dfwz1(2)
       tz=tv0(1)*dfuz1(3)+tv0(2)*dfvz1(3)+tv0(3)*dfwz1(3)+1
       dv0dz1=(/tx, ty, tz/)
!
!       Dv0/Dq2
!
       tx=tv0(1)*dfux2(1)+tv0(2)*dfvx2(1)+tv0(3)*dfwx2(1)
       ty=tv0(1)*dfux2(2)+tv0(2)*dfvx2(2)+tv0(3)*dfwx2(2)
       tz=tv0(1)*dfux2(3)+tv0(2)*dfvx2(3)+tv0(3)*dfwx2(3)
       dv0dx2=(/tx, ty, tz/)
       tx=tv0(1)*dfuy2(1)+tv0(2)*dfvy2(1)+tv0(3)*dfwy2(1)
       ty=tv0(1)*dfuy2(2)+tv0(2)*dfvy2(2)+tv0(3)*dfwy2(2)
       tz=tv0(1)*dfuy2(3)+tv0(2)*dfvy2(3)+tv0(3)*dfwy2(3)
       dv0dy2=(/tx, ty, tz/)
       tx=tv0(1)*dfuz2(1)+tv0(2)*dfvz2(1)+tv0(3)*dfwz2(1)
       ty=tv0(1)*dfuz2(2)+tv0(2)*dfvz2(2)+tv0(3)*dfwz2(2)
       tz=tv0(1)*dfuz2(3)+tv0(2)*dfvz2(3)+tv0(3)*dfwz2(3)
       dv0dz2=(/tx, ty, tz/)

!
!       Dv0/Dq3
!
       tx=tv0(1)*dfux3(1)+tv0(2)*dfvx3(1)
       ty=tv0(1)*dfux3(2)+tv0(2)*dfvx3(2)
       tz=tv0(1)*dfux3(3)+tv0(2)*dfvx3(3)
       dv0dx3=(/tx, ty, tz/)
       tx=tv0(1)*dfuy3(1)+tv0(2)*dfvy3(1)
       ty=tv0(1)*dfuy3(2)+tv0(2)*dfvy3(2)
       tz=tv0(1)*dfuy3(3)+tv0(2)*dfvy3(3)
       dv0dy3=(/tx, ty, tz/)
       tx=tv0(1)*dfuz3(1)+tv0(2)*dfvz3(1)
       ty=tv0(1)*dfuz3(2)+tv0(2)*dfvz3(2)
       tz=tv0(1)*dfuz3(3)+tv0(2)*dfvz3(3)
       dv0dz3=(/tx, ty, tz/)
 
       end subroutine buildatom

       subroutine prnmat(m)
       real(chm_real),dimension(4,4):: m
       write(*,'(4f10.3)')m(1,1),m(1,2),m(1,3),m(1,4)
       write(*,'(4f10.3)')m(2,1),m(2,2),m(2,3),m(2,4)
       write(*,'(4f10.3)')m(3,1),m(3,2),m(3,3),m(3,4)
       write(*,'(4f10.3)')m(4,1),m(4,2),m(4,3),m(4,4)
       end subroutine prnmat

!     
!
! ======================================================================
!
#else /* (epmf_main)*/
!
!  Dummy module when epmf is not needed
!  
  use chm_kinds
  use select
  use dimens_fcm
   implicit none
   contains
     subroutine epmf_set(COMLYN,COMLEN)
      character(len=*),intent(inout) :: COMLYN
      integer,intent(inout) :: COMLEN
      call WRNDIE(-5,'epmf:epmf_set','epmf module was not compiled')
     end subroutine epmf_set

#endif /* EPMF (epmf_main)*/
end module epmf

