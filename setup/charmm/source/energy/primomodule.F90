! Srinivasa Murthy gopal & Michael Feig, 2010
!            Michigan State University(MSU) 
!

   module primomodule
#if KEY_PRIMO==1 /*primo_main*/
   use chm_kinds
   use chm_types
   use epmf, only : buildatom,selctatoms,getcosangl2
   implicit none

   type primo_ds1
!calculate every cstep
       integer :: cstep      
!To/Notto construct virtual sites vs0,vs1,vs2,vs3     
       logical :: vs1,vs2,vs3,vs0
!parameters ie bond/angle/dihd parms for vs0, vs1 and vs2
       real(chm_real) :: b1,t1,q1,b2,t2,q2,b0,t0,q0
!atoms involved in virtual vs0,vs1,vs2,vs3
       integer :: catm1,catm2,catm3,catm4,catm5,batm1,batm2,batm3 
!flags for potentials involving virtual atoms
       logical :: qangl1,qangl2,qangl3,qdist1,qdist2,qdist3,qangl0,qdist0
!atoms involved in potentials(distances/angles) 
       type(chm_iptr) :: asel1,asel2,asel3,asel4,asel5,asel6,&
       dsel1,dsel2,dsel3,zsel1,zsel2,dsel0
!number of atoms involved in above potential        
       integer :: na1,na2,na3,nd1,nd2,nd3,na0,nd0
!spring constants and thetas for angl potential
       real(chm_real) :: kth1,kth2,kth3,th1,th2,th3,kth0,th0
!spring constants and distances for dist potential
       real(chm_real) :: kdt1,kdt2,kdt3,dt1,dt2,dt3,kdt0,dt0
   end type primo_ds1


     type(primo_ds1),allocatable,dimension(:),save :: vatms 
     logical,save :: qprminit=.true., qprimo=.false.,qmesg=.false.           
     logical,allocatable,dimension(:),save :: qvatm   !finds which residues have vatoms


     contains
     subroutine primo_set(COMLYN,COMLEN)
  use exfunc
  use stream
  use string
  use contrl

#if KEY_PARALLEL==1
  use parallel
#endif 

       character(len=*),intent(inout) :: COMLYN
       integer,intent(inout) :: COMLEN

!      local vars
       integer :: i
       character(len=4) :: comnd

!=====================================================================!

       if(qprminit) then
         qprminit=.false.
         qprimo=.true.
         call primo_init()
       endif

#if KEY_PARALLEL==1
       processparall: if(MYNODP .eq. 1)then
#endif 

       comnd=NEXTA4(COMLYN,COMLEN)
       if(comnd .eq. 'RESN' .or. comnd .eq. 'RESI') then
           call primo_process_stream(COMLYN,COMLEN,comnd)
       elseif(comnd .eq. 'CLEA') then
          write(OUTU,*)'PRIMO> Clearing the memory'
          call primo_free()
       else
          write(OUTU,*)'PRIMO> option: ',comnd,' is not a valid one'
          write(OUTU,*)'PRIMO> First option is always RESN'
          call WRNDIE(-5,'primo:primo_set','Unknown option')
       endif

#if KEY_PARALLEL==1
       endif processparall
#endif 
     end subroutine primo_set  

   

     subroutine primo_init()
  use exfunc
  use stream
  use number
  use param
  use psf

        integer :: i,astat
!=====================================================================!
        allocate(vatms(nres),qvatm(nres),stat=astat)
        if(astat /= 0) then
           call WRNDIE(-5,'primo:primo_set()',&
           'Couldnot allocate memory to primo datastructures')
        endif

        do i=1,nres
          qvatm(i)=.false.      !default one vatoms on any residue
          vatms(i)%cstep=1      !calculation every step
          vatms(i)%vs1=.false.  !no vs1
          vatms(i)%vs2=.false.  !no vs2
          vatms(i)%vs3=.false.  !no vs3
          vatms(i)%vs0=.false.  !no vs0

!         parameters for vs1          
          vatms(i)%b1=NINE99     
          vatms(i)%t1=NINE99
          vatms(i)%q1=NINE99
          vatms(i)%catm1=-1  !not defined
          vatms(i)%catm2=-1
          vatms(i)%catm3=-1
          vatms(i)%catm4=-1
          vatms(i)%catm5=-1

!         parameters for vs2         
          vatms(i)%b2=NINE99     
          vatms(i)%t2=NINE99
          vatms(i)%q2=NINE99

!         parameters for vs0
          vatms(i)%b0=NINE99
          vatms(i)%t0=NINE99
          vatms(i)%q0=NINE99
          vatms(i)%batm1=-1
          vatms(i)%batm2=-1
          vatms(i)%batm3=-1

!           parameters for the potentials 
          vatms(i)%qangl1=.false.    !no potentials by default
          vatms(i)%qangl2=.false.
          vatms(i)%qangl3=.false.
          vatms(i)%qdist1=.false.
          vatms(i)%qdist2=.false.
          vatms(i)%qdist3=.false.
          vatms(i)%qdist0=.false.
          vatms(i)%qangl0=.false.

          vatms(i)%nd1=0           !no. of atoms associated with dist potential
          vatms(i)%nd2=0
          vatms(i)%nd3=0
          vatms(i)%nd0=0
          vatms(i)%na1=0           !no. of atoms associated with angl potential
          vatms(i)%na2=0
          vatms(i)%na3=0 
          vatms(i)%na0=0


          vatms(i)%kth1=NINE99         !dummy spring constants
          vatms(i)%kth2=NINE99
          vatms(i)%kth3=NINE99
          vatms(i)%kth0=NINE99
          vatms(i)%kdt1=NINE99
          vatms(i)%kdt2=NINE99
          vatms(i)%kdt3=NINE99
          vatms(i)%kdt0=NINE99
          vatms(i)%th1=NINE99          !dummy minimas
          vatms(i)%th2=NINE99
          vatms(i)%th3=NINE99
          vatms(i)%th0=NINE99
          vatms(i)%dt1=NINE99
          vatms(i)%dt2=NINE99
          vatms(i)%dt3=NINE99
          vatms(i)%dt0=NINE99
       enddo
!=====================================================================!
     end subroutine primo_init

     subroutine primo_free()
  use exfunc
  use stream
  use number
  use param
  use psf
     
     integer :: astat
!=====================================================================!     
     if(allocated(vatms) .and. allocated(qvatm) ) then
        deallocate(vatms,qvatm,stat=astat)
        if(astat /= 0) then
           call WRNDIE(-5,'primo:primo_free()',&
           'Couldnot free primo datastructures')
        endif
         qprminit=.true.
         qprimo=.false.

     else
        call WRNDIE(-5,'primo:primo_free()', &
             'Deallocation invoked before init')
     endif
     end subroutine primo_free


     subroutine primo_process_stream(COMLYN,COMLEN,comnd)
  use select
  use stream
  use dimens_fcm
  use psf
  use number
  use string
  use memory
  use string
  use coord
  use chutil,only:getres,getseg,atomid
  use contrl

     character(len=4),intent(in) :: comnd
     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN

     !local vars
     character(len=8) :: resn
     integer :: resi,i,istp,atmax,j,ai,aj,pp,&
                iseg1,iseg2,iseg3,iseg4,iseg5,hseg1,hseg2,hseg3
     real(chm_real), dimension(natom) :: wt
     integer, dimension(natom) :: islct
     character(len=8) :: at1,at2,tt1,tt2
     logical :: qvs1,qvs3,qvs2,qvs0,qdis1,qdis2,qdis3,qdis0,&
                qang1,qang2,qang3,qang0
     real(chm_real) :: rlb1,rlt1,rlq1,rlb2,rlt2,rlq2,rlb0,rlt0,rlq0,&
                       rlkd1,rldt1,rlkd2,rldt2,rlkd3,rldt3,rlkd0,rldt0,&
                       rlkt1,rlth1,rlkt2,rlth2,rlkt3,rlth3,rlkt0,rlth0
     integer :: leng,ainx,na1,a1,a2,a3,a4,a5,a6,z1,z2,c1,c2,d1,d2,d3,d0
     character(len=8) :: satm1,satm2,satm3,satm4,satm5,ratm1,ratm2,ratm3,cht1,&
                         atm1,atm2,atm3,atm4,atm5,ztm1,ztm2,ztm3
     character(len=1) :: pfx
     integer,dimension(natom) :: ia1,ia2,ia3,ia4,ia5,ia6,ha1,ha2,id1,id2,id3,id0

!=====================================================================!


     resn='NORES'
     resi=-9999

     qvs1=.false.
     qvs3=.false.
     qvs2=.false.
     qvs0=.false.

     qdis1=.false.
     qdis2=.false.
     qdis3=.false.
     qdis0=.false.
     qang1=.false.
     qang2=.false.
     qang3=.false.
     qang0=.false.

     rlkd1=NINE99
     rlkd2=NINE99
     rlkd3=NINE99
     rlkd0=NINE99
     rldt1=NINE99
     rldt2=NINE99
     rldt3=NINE99
     rldt0=NINE99
     rlkt1=NINE99
     rlkt2=NINE99
     rlkt3=NINE99
     rlkt0=NINE99
     rlth1=NINE99
     rlth2=NINE99
     rlth3=NINE99
     rlth0=NINE99

     a1=0
     a2=0
     a3=0
     a4=0
     a5=0
     a6=0
     z1=0
     z2=0
     c1=0
     c2=0
     d1=0
     d2=0
     d3=0
     d0=0

     ia1=0   !no selections for asel1,2,3,4,5,6
     ia2=0
     ia3=0
     ia4=0
     ia5=0
     ia6=0
     ha1=0
     ha2=0
     id1=0   !no selections for dsel1,2,3
     id2=0
     id3=0
     id0=0

!-------------------- begin parsing -----------------------------------|

!    parse residue option [RESI/RESN]
     if(comnd .eq. 'RESN') resn=NEXTA8(COMLYN,COMLEN)
     if(comnd .eq. 'RESI') resi=NEXTI(COMLYN,COMLEN)

     if(resn .eq. 'NORES' .and. resi .eq. -9999) then
        write(*,*)resn,resi
        call WRNDIE(-5,'primo:process_stream',&
             'Invaild options for RESN or RESI')
     endif


        
     istp=GTRMI(COMLYN,COMLEN,'CSTEP',1)
!    parse virtual atom selections [VS1,VS3,VS2,VS0]
     if(INDXA(COMLYN,COMLEN,'VS1') .gt. 0)  qvs1=.true.
     if(INDXA(COMLYN,COMLEN,'VS2') .gt. 0)  qvs2=.true.
     if(INDXA(COMLYN,COMLEN,'VS3') .gt. 0)  qvs3=.true.
     if(INDXA(COMLYN,COMLEN,'VS0') .gt. 0)  qvs0=.true.

!    check if relevant virtual atoms are asked for     
     if(qvs2 .and. (.not. qvs1) ) then
        call WRNDIE(-5,'primo:process_stream',&
            'VS2 is possible when VS1 is defined')
     endif

     if(qvs3 .and. (.not. qvs2) .and. (.not. qvs1) ) then
         call WRNDIE(-5,'primo:process_stream',&
            'VS3 is possible when both VS1 and VS2 are defined')
     endif


!    parse atm1-atm5 selections
     satm1='UNKNOWN'
     satm2='UNKNOWN'
     satm3='UNKNOWN'
     satm4='UNKNOWN'
     satm5='UNKNOWN'
     ratm1='UNKNOWN'
     ratm2='UNKNOWN'
     ratm3='UNKNOWN'
     atm1='UNKNOWN'
     atm2='UNKNOWN'
     atm3='UNKNOWN'
     atm4='UNKNOWN'
     atm5='UNKNOWN'
     ztm1='UNKNOWN'
     ztm2='UNKNOWN'
     ztm3='UNKNOWN'
     iseg1=-1
     iseg2=-1
     iseg3=-1
     iseg4=-1
     iseg5=-1
     hseg1=-1
     hseg2=-1
     hseg3=-1

     call GTRMWD(COMLYN,COMLEN,'ATM1',4,satm1,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ATM2',4,satm2,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ATM3',4,satm3,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ATM4',4,satm4,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ATM5',4,satm5,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ZTM1',4,ratm1,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ZTM2',4,ratm2,8,leng)
     call GTRMWD(COMLYN,COMLEN,'ZTM3',4,ratm3,8,leng)


!    parse parameters/potentials for VS0 (if any)
     if(qvs0) then
        rlb0=GTRMF(COMLYN,COMLEN,'B0',NINE99)
        if(rlb0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter B0')
        rlt0=GTRMF(COMLYN,COMLEN,'T0',NINE99)
        if(rlt0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter T0')
        rlq0=GTRMF(COMLYN,COMLEN,'Q0',NINE99)
        if(rlq0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter Q0')

!     check ztm1,ztm2,ztm3
         if(ratm1 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ZTM1')
         if(ratm2 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ZTM2')
         if(ratm3 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ZTM3')

!       parse dist potential of VS0 
        if(INDXA(COMLYN,COMLEN,'DIS0') .gt. 0) then
           qdis0=.true.
           rlkd0=GTRMF(COMLYN,COMLEN,'KDT0',NINE99)
           if(rlkd0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KDT0] for DIS0 potential')
           rldt0=GTRMF(COMLYN,COMLEN,'MND0',NINE99)
           if(rldt0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MND0] for DIS0 potential')

            if(INDXA(COMLYN,COMLEN,'DSEL0') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,id0,X,Y,Z,wt,.TRUE.)
              d0=NSELCT(NATOM,id0)          
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [DSEL0] for DIS0 potential')        
            endif

         endif

!       parse angl potential of VS0
        if(INDXA(COMLYN,COMLEN,'ANG0') .gt. 0) then
           qang0=.true.
           rlkt0=GTRMF(COMLYN,COMLEN,'KTH0',NINE99)
           if(rlkt0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KTH0] for ANG0 potential')
           rlth0=GTRMF(COMLYN,COMLEN,'MNT0',NINE99)
           if(rlth0 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MNT0] for ANG0 potential')

            if(INDXA(COMLYN,COMLEN,'ZSEL1') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ha1,X,Y,Z,wt,.TRUE.)
              z1=NSELCT(NATOM,ha1)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ZSEL1] for ANG0 potential')             
            endif

     
            if(INDXA(COMLYN,COMLEN,'ZSEL2') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ha2,X,Y,Z,wt,.TRUE.)
              z2=NSELCT(NATOM,ha2)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ZSEL2] for ANG0 potential')            
            endif
        endif
     endif

     
!    parse parameters/potentials for VS1 (if any)
     if(qvs1) then
        rlb1=GTRMF(COMLYN,COMLEN,'B1',NINE99)
        if(rlb1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter B1')
        rlt1=GTRMF(COMLYN,COMLEN,'T1',NINE99)
        if(rlt1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter T1')
        rlq1=GTRMF(COMLYN,COMLEN,'Q1',NINE99)
        if(rlq1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter Q1')

!     check atm1,atm2,atm3
         if(satm1 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ATM1')
         if(satm2 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ATM2')
         if(satm3 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ATM3')

!       parse dist potential of VS1 
        if(INDXA(COMLYN,COMLEN,'DIS1') .gt. 0) then
           qdis1=.true.
           rlkd1=GTRMF(COMLYN,COMLEN,'KDT1',NINE99)
           if(rlkd1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KDT1] for DIS1 potential')
           rldt1=GTRMF(COMLYN,COMLEN,'MND1',NINE99)
           if(rldt1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MND1] for DIS1 potential')

            if(INDXA(COMLYN,COMLEN,'DSEL1') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,id1,X,Y,Z,wt,.TRUE.)
              d1=NSELCT(NATOM,id1)          
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [DSEL1] for DIS1 potential')        
            endif
        endif

!       parse angl potential of VS1
        if(INDXA(COMLYN,COMLEN,'ANG1') .gt. 0) then
           qang1=.true.
           rlkt1=GTRMF(COMLYN,COMLEN,'KTH1',NINE99)
           if(rlkt1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KTH1] for ANG1 potential')
           rlth1=GTRMF(COMLYN,COMLEN,'MNT1',NINE99)
           if(rlth1 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MNT1] for ANG1 potential')

            if(INDXA(COMLYN,COMLEN,'ASEL1') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia1,X,Y,Z,wt,.TRUE.)
              a1=NSELCT(NATOM,ia1)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL1] for ANG1 potential')             
            endif

     
            if(INDXA(COMLYN,COMLEN,'ASEL2') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia2,X,Y,Z,wt,.TRUE.)
              a2=NSELCT(NATOM,ia2)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL2] for ANG1 potential')            
            endif
        endif

     endif



!    parse parameters/potentials for VS2 (if any)
     if(qvs2) then
        rlb2=GTRMF(COMLYN,COMLEN,'B2',NINE99)
        if(rlb2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Could not read parameter b2')
        rlt2=GTRMF(COMLYN,COMLEN,'T2',NINE99)
        if(rlt2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Could not read parameter t2')
        rlq2=GTRMF(COMLYN,COMLEN,'Q2',NINE99)
        if(rlq2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
                 'Could not read parameter q2')

!       parse dist potential of VS2 
        if(INDXA(COMLYN,COMLEN,'DIS2') .gt. 0) then
           qdis2=.true.
           rlkd2=GTRMF(COMLYN,COMLEN,'KDT2',NINE99)
           if(rlkd2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KDT2] for DIS2 potential')
           rldt2=GTRMF(COMLYN,COMLEN,'MND2',NINE99)
           if(rldt2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MND2] for DIS2 potential')

            if(INDXA(COMLYN,COMLEN,'DSEL2') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,id2,X,Y,Z,wt,.TRUE.)
              d2=NSELCT(NATOM,id2)          
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [DSEL2] for DIS2 potential')        
            endif

        endif

!       parse angl potential of VS2
        if(INDXA(COMLYN,COMLEN,'ANG2') .gt. 0) then
           qang2=.true.
           rlkt2=GTRMF(COMLYN,COMLEN,'KTH2',NINE99)
           if(rlkt2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KTH2] for ANG2 potential')
           rlth2=GTRMF(COMLYN,COMLEN,'MNT2',NINE99)
           if(rlth2 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MNT2] for ANG2 potential')

            if(INDXA(COMLYN,COMLEN,'ASEL3') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia3,X,Y,Z,wt,.TRUE.)
              a3=NSELCT(NATOM,ia3)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL3] for ANG2 potential')             
            endif

            if(INDXA(COMLYN,COMLEN,'ASEL4') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia4,X,Y,Z,wt,.TRUE.)
              a4=NSELCT(NATOM,ia4)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL4] for ANG2 potential')            
            endif

        endif

!     check atm4
         if(satm4 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ATM4')

     endif

!    parse potentials for VS3
     if(qvs3) then
!       parse dist potential of VS3 
        if(INDXA(COMLYN,COMLEN,'DIS3') .gt. 0) then
           qdis3=.true.
           rlkd3=GTRMF(COMLYN,COMLEN,'KDT3',NINE99)
           if(rlkd3 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KDT3] for DIS3 potential')
           rldt3=GTRMF(COMLYN,COMLEN,'MND3',NINE99)
           if(rldt3 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MND3] for DIS3 potential')

            if(INDXA(COMLYN,COMLEN,'DSEL3') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,id3,X,Y,Z,wt,.TRUE.)
              d3=NSELCT(NATOM,id3)          
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [DSEL3] for DIS3 potential')        
            endif

        endif


!       parse angl potential of VS3
        if(INDXA(COMLYN,COMLEN,'ANG3') .gt. 0) then
           qang3=.true.
           rlkt3=GTRMF(COMLYN,COMLEN,'KTH3',NINE99)
           if(rlkt3 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
           'Missing spring constant[KTH3] for ANG3 potential')
           rlth3=GTRMF(COMLYN,COMLEN,'MNT3',NINE99)
           if(rlth3 .eq. NINE99) call WRNDIE(-5,'primo:process_stream',&
            'Missing minima[MNT3] for ANG3 potential')

            if(INDXA(COMLYN,COMLEN,'ASEL5') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia5,X,Y,Z,wt,.TRUE.)
              a5=NSELCT(NATOM,ia5)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL5] for ANG3 potential')             
            endif

            if(INDXA(COMLYN,COMLEN,'ASEL6') .gt. 0) then
              call SELCTA(COMLYN,COMLEN,ia6,X,Y,Z,wt,.TRUE.)
              a6=NSELCT(NATOM,ia6)
            else
              call WRNDIE(-5,'primo:process_stream',&
                'Missing selection [ASEL6] for ANG3 potential')            
            endif

        endif

!     check atm5
         if(satm5 .eq. 'UNKNOWN') call WRNDIE(-5,'primo:process_stream',&
                 'Missing parameter ATM5')

     endif

!--------------assign values from parsed data ------------------------|

     resloop: do i=1,nres
    
        chkreslp: if(res(i) .eq. resn .or. i .eq. resi) then
           qvatm(i)=.true.
           vatms(i)%cstep=istp
           vatms(i)%vs1=qvs1
           vatms(i)%vs2=qvs2
           vatms(i)%vs3=qvs3
           vatms(i)%vs0=qvs0

           vs1loop: if(qvs1) then
              vatms(i)%b1=rlb1
              vatms(i)%t1=rlt1
              vatms(i)%q1=rlq1

              cht1=trim(satm1)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 atm1=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 atm1=cht1(2:)
              else
                 pfx=' '
                 atm1=cht1
              endif
              call selctatoms(i,atm1,pfx,ainx)
              vatms(i)%catm1=ainx

              cht1=trim(satm2)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 atm2=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 atm2=cht1(2:)
              else
                 pfx=' '
                 atm2=cht1
              endif
              call selctatoms(i,atm2,pfx,ainx)
              vatms(i)%catm2=ainx

              cht1=trim(satm3)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 atm3=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 atm3=cht1(2:)
              else
                 pfx=' '
                 atm3=cht1
              endif
              call selctatoms(i,atm3,pfx,ainx)
              vatms(i)%catm3=ainx


              if(qdis1) then
                 vatms(i)%qdist1=.true.
                 vatms(i)%kdt1=rlkd1
                 vatms(i)%dt1=rldt1
              endif

              if(qang1) then
                 vatms(i)%qangl1=.true.
                 vatms(i)%kth1=rlkt1
                 vatms(i)%th1=rlth1
              endif

           endif vs1loop

           vs0loop: if(qvs0) then
              vatms(i)%b0=rlb0
              vatms(i)%t0=rlt0
              vatms(i)%q0=rlq0

              cht1=trim(ratm1)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 ztm1=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 ztm1=cht1(2:)
              else
                 pfx=' '
                 ztm1=cht1
              endif
              call selctatoms(i,ztm1,pfx,ainx)
              vatms(i)%batm1=ainx

              cht1=trim(ratm2)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 ztm2=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 ztm2=cht1(2:)
              else
                 pfx=' '
                 ztm2=cht1
              endif
              call selctatoms(i,ztm2,pfx,ainx)
              vatms(i)%batm2=ainx

              cht1=trim(ratm3)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 ztm3=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 ztm3=cht1(2:)
              else
                 pfx=' '
                 ztm3=cht1
              endif
              call selctatoms(i,ztm3,pfx,ainx)
              vatms(i)%batm3=ainx


              if(qdis0) then
                 vatms(i)%qdist0=.true.
                 vatms(i)%kdt0=rlkd0
                 vatms(i)%dt0=rldt0
              endif

              if(qang0) then
                 vatms(i)%qangl0=.true.
                 vatms(i)%kth0=rlkt0
                 vatms(i)%th0=rlth0
              endif

           endif vs0loop

           vs2loop: if(qvs2) then
              vatms(i)%b2=rlb2
              vatms(i)%t2=rlt2
              vatms(i)%q2=rlq2

              cht1=trim(satm4)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 atm4=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 atm4=cht1(2:)
              else
                 pfx=' '
                 atm4=cht1
              endif
              call selctatoms(i,atm4,pfx,ainx)
              vatms(i)%catm4=ainx

              if(qdis2) then
                 vatms(i)%qdist2=.true.
                 vatms(i)%kdt2=rlkd2
                 vatms(i)%dt2=rldt2
              endif

              if(qang2) then
                 vatms(i)%qangl2=.true.
                 vatms(i)%kth2=rlkt2
                 vatms(i)%th2=rlth2
              endif             

                   
           endif vs2loop


           vs3loop:if(qvs3) then

              if(qdis3) then
                 vatms(i)%qdist3=.true.
                 vatms(i)%kdt3=rlkd3
                 vatms(i)%dt3=rldt3
              endif

              if(qang3) then
                 vatms(i)%qangl3=.true.
                 vatms(i)%kth3=rlkt3
                 vatms(i)%th3=rlth3
              endif             

              cht1=trim(satm5)
              if(cht1(1:1) .eq. '+') then
                 pfx='+'
                 atm5=cht1(2:)
              elseif(cht1(1:1) .eq. '-') then
                 pfx='-'
                 atm5=cht1(2:)
              else
                 pfx=' '
                 atm5=cht1
              endif
              call selctatoms(i,atm5,pfx,ainx)
              vatms(i)%catm5=ainx
           endif vs3loop

!              
!    See if catm1,catm2,catm3,catm4,catm5,batm1,batm2,atm3 are 
!    part of same isegment ...  if not set all to -1
!
          if(vatms(i)%vs1) then
              if(vatms(i)%catm1 .ne. -1 .and. vatms(i)%catm2 .ne. -1 &
                 .and.  vatms(i)%catm3 .ne. -1) then 
                 pp=GETRES(vatms(i)%catm1,IBASE,NRES)
                 iseg1=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%catm2,IBASE,NRES)
                 iseg2=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%catm3,IBASE,NRES)
                 iseg3=GETSEG(pp,NICTOT,NSEG)
                 if(iseg2 .ne. iseg1 .or. iseg3 .ne. iseg1) then
                     vatms(i)%catm1=-1
                     vatms(i)%catm2=-1
                     vatms(i)%catm3=-1
                 endif
              else
                 vatms(i)%catm1=-1
                 vatms(i)%catm2=-1
                 vatms(i)%catm3=-1
              endif
          endif
!
!         segment checks for vs0       
!
          if(vatms(i)%vs0) then
              if(vatms(i)%batm1 .ne. -1 .and. vatms(i)%batm2 .ne. -1 &
                 .and.  vatms(i)%batm3 .ne. -1) then 
                 pp=GETRES(vatms(i)%batm1,IBASE,NRES)
                 hseg1=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%batm2,IBASE,NRES)
                 hseg2=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%batm3,IBASE,NRES)
                 hseg3=GETSEG(pp,NICTOT,NSEG)
                 if(hseg2 .ne. hseg1 .or. hseg3 .ne. hseg1) then
                     vatms(i)%batm1=-1
                     vatms(i)%batm2=-1
                     vatms(i)%batm3=-1
                 endif
              else
                 vatms(i)%batm1=-1
                 vatms(i)%batm2=-1
                 vatms(i)%batm3=-1
              endif
          endif

!        check for catm4..if vs1 is defined and valid, 
!                   else.. disable only vs2 calculation
          if(vatms(i)%vs2) then
             if(vatms(i)%catm1 .ne. -1 .and. vatms(i)%catm4 .ne. -1) then
                 pp=GETRES(vatms(i)%catm1,IBASE,NRES)
                 iseg1=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%catm4,IBASE,NRES)
                 iseg4=GETSEG(pp,NICTOT,NSEG)
                 if(iseg4 .ne. iseg1) then
                     vatms(i)%catm4=-1
                 endif
             endif
          endif

          
!        check for catm5..if vs2 is defined and valid
!                  else.. disable only vs3 calculation
          if(vatms(i)%vs3) then
             if(vatms(i)%catm4 .ne. -1 .and. vatms(i)%catm5 .ne. -1) then
                 pp=GETRES(vatms(i)%catm4,IBASE,NRES)
                 iseg4=GETSEG(pp,NICTOT,NSEG)
                 pp=GETRES(vatms(i)%catm5,IBASE,NRES)
                 iseg5=GETSEG(pp,NICTOT,NSEG)
                 if(iseg5 .ne. iseg4) then
                     vatms(i)%catm5=-1
                 endif
             endif 
          endif

         call selctpotatoms(i,d1,a1,a2,id1,ia1,ia2,&
                     d2,a3,a4,id2,ia3,ia4,&
                     d3,a5,a6,id3,ia5,ia6,&
                     d0,z1,z2,id0,ha1,ha2)

        endif chkreslp

      enddo resloop
     end subroutine primo_process_stream


 
     subroutine selctpotatoms(nr,d1,a1,a2,id1,ia1,ia2,d2,a3,a4,id2,ia3,ia4,&
                      d3,a5,a6,id3,ia5,ia6,d0,z1,z2,id0,ha1,ha2)
  use exfunc
  use stream
  use number
  use string
  use psf
  use chutil,only:getres,atomid,getseg
  use contrl
  use memory

     integer,intent(in)  :: nr,d1,d2,d3,d0,a1,a2,a3,a4,a5,a6,z1,z2
     integer,dimension(natom),intent(in) :: id1,id2,id3,id0,&
                          ia1,ia2,ia3,ia4,ia5,ia6,ha1,ha2

!    local vars
     integer :: i,j,astat,pp,jj,ai,aj
     character(len=8) :: sid,rid,ren,ac     
     integer,dimension(natom) :: isel,jsel!copy of id* used for selction
     integer :: na11,na12,na21,na22,na31,na32,na01,na02,nd0,nd1,nd2,nd3
     integer,allocatable,dimension(:) :: asel1,asel2,dsel
!=====================================================================!
     vatms(nr)%nd1=0
     vatms(nr)%nd2=0
     vatms(nr)%nd3=0
     vatms(nr)%nd0=0
     vatms(nr)%na1=0
     vatms(nr)%na2=0
     vatms(nr)%na3=0
     vatms(nr)%na0=0

     na11=0
     na12=0
     na21=0
     na22=0
     na31=0
     na32=0
     na01=0
     na02=0
     nd1=0
     nd2=0
     nd3=0
     nd0=0

!    dsel1 
     isel=0
     if(vatms(nr)%qdist1) then
        isel=id1
        call distselcs(nr,d1,nd1,isel)
        if(nd1 .gt. 0) then
             vatms(nr)%nd1=nd1
             call CHMALLOC('primomodule','selctpotatoms',&
                  'vatms(nr)%dsel1%a',nd1,intgp=vatms(nr)%dsel1%a,qdie=.true.)
             vatms(nr)%dsel1%len=nd1
             allocate(dsel(nd1),stat=astat)
             ai=0
             do i=1,natom
               if(isel(i) .eq. 1) then
                 ai=ai+1
                 dsel(ai)=i
               endif
             enddo
             do j=1,nd1
               vatms(nr)%dsel1%a(j)=dsel(j)
             enddo
             deallocate(dsel)
        endif
     endif 

!    dsel2
     isel=0 
     if(vatms(nr)%qdist2) then
        isel=id2
        call distselcs(nr,d2,nd2,isel)
        if(nd2 .gt. 0) then
             vatms(nr)%nd2=nd2
             call CHMALLOC('primomodule','selctpotatoms',&
                 'vatms(nr)%dsel2%a',nd2,intgp=vatms(nr)%dsel2%a,qdie=.true.)
              vatms(nr)%dsel2%len=nd2
              allocate(dsel(nd2),stat=astat)
              ai=0
              do i=1,natom
                if(isel(i) .eq. 1) then
                  ai=ai+1
                  dsel(ai)=i
                endif
              enddo
              do j=1,nd2
                vatms(nr)%dsel2%a(j)=dsel(j)
              enddo
              deallocate(dsel)
        endif     
     endif 

!    dsel3
     isel=0
     if(vatms(nr)%qdist3) then
        isel=id3
        call distselcs(nr,d3,nd3,isel)
        if(nd3 .gt. 0) then
             vatms(nr)%nd3=nd3
             call CHMALLOC('primomodule','selctpotatoms',&
                  'vatms(nr)%dsel3%a',nd3,intgp=vatms(nr)%dsel3%a,qdie=.true.)
             vatms(nr)%dsel3%len=nd3
             allocate(dsel(nd3),stat=astat)
             ai=0
             do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  dsel(ai)=i
               endif
             enddo
             do j=1,nd3
               vatms(nr)%dsel3%a(j)=dsel(j)
             enddo
             deallocate(dsel)
        endif 
     endif

!     dsel0   
     isel=0
     if(vatms(nr)%qdist0) then
        isel=id0
        call distselcs(nr,d0,nd0,isel)
        if(nd0 .gt. 0) then
             vatms(nr)%nd0=nd0
             call CHMALLOC('primomodule','selctpotatoms',&
                  'vatms(nr)%dsel0%a',nd0,intgp=vatms(nr)%dsel0%a,qdie=.true.)
             vatms(nr)%dsel0%len=nd0
             allocate(dsel(nd0),stat=astat)
             ai=0
             do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  dsel(ai)=i
               endif
             enddo
             do j=1,nd0
                vatms(nr)%dsel0%a(j)=dsel(j)
             enddo
             deallocate(dsel)
         endif  
     endif 

     isel=0
     jsel=0
     pota1:if(vatms(nr)%qangl1) then
        isel=ia1
        jsel=ia2
        call anglselcs(nr,a1,a2,na11,na12,isel,jsel)
        if(na11 .ne. 0 .and. na12 .ne. 0) then

           if(na11 .ne. na12)call WRNDIE(-5,'primo:selctpotatoms',&
                  'No. of asel1 atoms != No. of asel2 atoms')

            allocate(asel1(na11),asel2(na11),stat=astat)
            if(astat /=0)call WRNDIE(-5,'primo:selctpotatoms',&
            'memory allocation failed [1]')

            ai=0
            aj=0
            do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  asel1(ai)=i
               elseif(jsel(i) .eq. 1) then
                  aj=aj+1
                  asel2(aj)=i
               else      
                  continue
               endif
            enddo 
                    
            vatms(nr)%na1=na11
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel1%a',&
                       na11,intgp=vatms(nr)%asel1%a,qdie=.true.)
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel2%a',&
                       na11,intgp=vatms(nr)%asel2%a,qdie=.true.)
            vatms(nr)%asel1%len=na11
            vatms(nr)%asel2%len=na11

            do j=1,na11
                vatms(nr)%asel1%a(j)=asel1(j)
            enddo
            do j=1,na11
                vatms(nr)%asel2%a(j)=asel2(j)
            enddo
            deallocate(asel1,asel2)
        endif
     endif pota1

     isel=0
     jsel=0
     pota2:if(vatms(nr)%qangl2) then
        isel=ia3
        jsel=ia4
        call anglselcs(nr,a3,a4,na21,na22,isel,jsel)
        if(na21 .ne. 0 .and. na22 .ne. 0) then

            if(na21 .ne. na22)call WRNDIE(-5,'primo:selctpotatoms',&
                  'No. of asel3 atoms != No. of asel4 atoms')
            allocate(asel1(na22),asel2(na22),stat=astat)

            if(astat /=0)call WRNDIE(-5,'primo:selctpotatoms',&
            'memory allocation failed [2]')

            ai=0
            aj=0
            do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  asel1(ai)=i
               elseif(jsel(i) .eq. 1) then
                  aj=aj+1
                  asel2(aj)=i
               else      
                  continue
               endif
            enddo 
                    
            vatms(nr)%na2=na22
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel3%a',&
                       na22,intgp=vatms(nr)%asel3%a,qdie=.true.)
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel4%a',&
                       na22,intgp=vatms(nr)%asel4%a,qdie=.true.)
            vatms(nr)%asel3%len=na22
            vatms(nr)%asel4%len=na22

            do j=1,na22
                vatms(nr)%asel3%a(j)=asel1(j)
            enddo
            do j=1,na22
                vatms(nr)%asel4%a(j)=asel2(j)
            enddo
            deallocate(asel1,asel2)
        endif
     endif pota2

     isel=0
     jsel=0
     pota3:if(vatms(nr)%qangl3) then
        isel=ia5
        jsel=ia6
        call anglselcs(nr,a5,a6,na31,na32,isel,jsel)
        if(na31 .ne. 0 .and. na32 .ne. 0) then

            if(na31 .ne. na32)call WRNDIE(-5,'primo:selctpotatoms',&
                  'No. of asel5 atoms != No. of asel6 atoms')

            allocate(asel1(na32),asel2(na32),stat=astat)
            if(astat /=0)call WRNDIE(-5,'primo:selctpotatoms',&
            'memory allocation failed [3]')

            ai=0
            aj=0
            do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  asel1(ai)=i
               elseif(jsel(i) .eq. 1) then
                  aj=aj+1
                  asel2(aj)=i
               else      
                  continue
               endif
            enddo 
                    
            vatms(nr)%na3=na32
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel5%a',&
                       na32,intgp=vatms(nr)%asel5%a,qdie=.true.)
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%asel6%a',&
                       na32,intgp=vatms(nr)%asel6%a,qdie=.true.)
            vatms(nr)%asel5%len=na32
            vatms(nr)%asel6%len=na32

            do j=1,na32
                vatms(nr)%asel5%a(j)=asel1(j)
            enddo
            do j=1,na32
                vatms(nr)%asel6%a(j)=asel2(j)
            enddo
            deallocate(asel1,asel2)
        endif
     endif pota3
        
     isel=0
     jsel=0
     pota0:if(vatms(nr)%qangl0) then
        isel=ha1
        jsel=ha2
        call anglselcs(nr,z1,z2,na01,na02,isel,jsel)
        if(na01 .ne. 0 .and. na02 .ne. 0) then

           if(na01 .ne. na02)call WRNDIE(-5,'primo:selctpotatoms',&
                  'No. of zsel1 atoms != No. of zsel2 atoms')

            allocate(asel1(na02),asel2(na02),stat=astat)
            if(astat /=0)call WRNDIE(-5,'primo:selctpotatoms',&
            'memory allocation failed [0]')

 
            ai=0
            aj=0
            do i=1,natom
               if(isel(i) .eq. 1) then
                  ai=ai+1
                  asel1(ai)=i
               elseif(jsel(i) .eq. 1) then
                  aj=aj+1
                  asel2(aj)=i
               else      
                  continue
               endif
            enddo 
                    
            vatms(nr)%na0=na02
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%zsel1%a',&
                       na02,intgp=vatms(nr)%zsel1%a,qdie=.true.)
            call CHMALLOC('primomodule','selctpotatoms','vatms(nr)%zsel2%a',&
                       na02,intgp=vatms(nr)%zsel2%a,qdie=.true.)
            vatms(nr)%zsel1%len=na02
            vatms(nr)%zsel2%len=na02

            do j=1,na02
                vatms(nr)%zsel1%a(j)=asel1(j)
            enddo
            do j=1,na02
                vatms(nr)%zsel2%a(j)=asel2(j)
            enddo
            deallocate(asel1,asel2)
        endif
     endif pota0

     end subroutine selctpotatoms


     subroutine distselcs(nr,dx,nd,idx)
  use exfunc
  use stream
  use number
  use string
  use psf
  use chutil,only:getres
  use contrl
  use memory

     integer,intent(in)   :: nr,dx
!    read input iselct array. if selection exists and
!    doesnot match current residue, reset idx(i) to zero
     integer,dimension(natom),intent(inout) :: idx
     integer,intent(out)  :: nd
!    local vars
     integer :: i,pp
     if(dx .ne. 0) then
        nd=0
        do i=1,natom
           if( idx(i) .eq. 1) then
               pp=GETRES(i,IBASE,NRES)
               if(pp .eq. nr) then
                  nd=nd+1
               else
                  idx(i)=0   !reset idx(i)
               endif
           endif
        enddo
     else
        nd=0
     endif

     end subroutine distselcs


     subroutine anglselcs(nr,da,db,na,nb,ida,idb)
  use exfunc
  use stream
  use number
  use string
  use psf
  use chutil,only:getres
  use contrl
  use memory

     integer,intent(in)   :: nr,da,db
     integer,dimension(natom),intent(inout) :: ida,idb
     integer,intent(out)  :: na,nb
!    local vars
     integer :: i,pp

     if(da .ne. 0 .and. db .ne. 0) then
        na=0
        nb=0
        pp=0

        do i=1,natom
          if( ida(i) .eq. 1 .and. idb(i) .eq. 1) then   
              call WRNDIE(-5,'primo:anglselcs',&
                 'No angle possible between given selections')
          else
             if( ida(i) .eq. 1) then
                pp=GETRES(i,IBASE,NRES)
                if(pp .eq. nr) then
                   na=na+1
                else
                   ida(i)=0  !reset on residue mismatch
                endif      
             elseif( idb(i) .eq. 1) then
                pp=GETRES(i,IBASE,NRES)
                if(pp .eq. nr) then
                  nb=nb+1
                else
                  idb(i)=0  !reset on residue mismatch
                endif   
             else
                continue
             endif
          endif  
        enddo
      else
        na=0
        nb=0
      endif

     end subroutine anglselcs


     subroutine print_primo_ds1(foo)
  use exfunc
  use stream
  use string
  use contrl

     type(primo_ds1),intent(in) :: foo
     character(len=75) :: fl

     fl='(/,a8,6i8,a5,/,2x,l3,a8,3f10.3,a8,l3,2f10.3,l3,2f10.3)'
     write(OUTU,fl)'PRIMO> ',foo%catm1,foo%catm2,foo%catm3,&
        foo%catm4,foo%catm5,foo%cstep, 'VS1:', foo%vs1, 'parms:',&
        foo%b1, foo%t1, foo%q1,'pot:',foo%qdist1,foo%dt1,foo%kdt1,&
         foo%qangl1,foo%th1,foo%kth1

     fl='(2a5,/,2x,l3,a8,3f10.3,a8,l3,2f10.3,l3,2f10.3)'
     write(OUTU,fl)'PRIMO> ','VS2:', foo%vs2, 'parms:', foo%b2, &
        foo%t2, foo%q2,'pot:',foo%qdist2,foo%dt2,foo%kdt2,&
         foo%qangl2,foo%th2,foo%kth2

     fl='(2a5,/,2x,l3,a8,l3,2f10.3,l3,2f10.3)'
     write(OUTU,fl)'PRIMO> ','VS3:', foo%vs3,'pot:',foo%qdist3,foo%dt3,&
           foo%kdt3,foo%qangl3,foo%th3,foo%kth3   

     fl='(2a5,/,2x,l3,a8,l3,2f10.3,l3,2f10.3)'
     write(OUTU,fl)'PRIMO> ','VS0:', foo%vs0,'pot:',foo%qdist0,foo%dt0,&
           foo%kdt0,foo%qangl0,foo%th0,foo%kth0   


     end subroutine print_primo_ds1


     subroutine calcprimoene(ev,nr,X,Y,Z,DX,DY,DZ)

  use exfunc
  use stream
  use dimens_fcm
  use psf
  use number
  use econtmod
  use energym
  use contrl
  use memory
  use consta
        
     integer,intent(in) :: nr
     real(chm_real),intent(out) :: ev
     real(chm_real),dimension(:),intent(inout) :: DX,DY,DZ
     real(chm_real),dimension(:),intent(in) :: X,Y,Z

 !   local vars
     logical :: qcal1,qcal2,qcal3,qcal0
     real(chm_real) :: et,tt
     real(chm_real),dimension(3) :: tqc1(3),tqc2(3),tqc3(3),&
             tqc4(3),tqc5(3),&!derivatives of catm1,2,3,4,5$&
             sqc1(3),sqc2(3),sqc3(3),&!derivatives of batm1,2,3
             tqd1(3),tqd2(3),tqd3(3),tqd0(3),&!derivatives of dsel1,2,3,0
     tqa1(3),tqa2(3),tqa3(3),tqa4(3),tqa5(3),tqa6(3),&!deriv of asel1,6
     sqa1(3),sqa2(3) !deriv of zsel1,2
     
     integer :: i,j,ii,jj
     
     real(chm_real) :: tvs1(3),tvs2(3),tvs3(3),tvs0(3),&
                tv0x1(3),tv0y1(3),tv0z1(3),tv0x2(3),tv0y2(3),tv0z2(3),&
                tv0x3(3),tv0y3(3),tv0z3(3),tv0x4(3),tv0y4(3),tv0z4(3),&
                tv0x5(3),tv0y5(3),tv0z5(3),sv0x1(3),sv0y1(3),sv0z1(3),&
                sv0x2(3),sv0y2(3),sv0z2(3),sv0x3(3),sv0y3(3),sv0z3(3),&
                v1(3),v2(3),v3(3),v4(3),v5(3),u1(3),u2(3),u3(3),&
                svec(3),tvec(3)

     real(chm_real) :: rvs1(3),rvs2(3),rvs3(3),rvs0(3),&
      drvs1dx1(3),drvs1dy1(3),drvs1dz1(3),drvs1dx2(3),drvs1dy2(3),drvs1dz2(3),&
      drvs1dx3(3),drvs1dy3(3),drvs1dz3(3),drvs1(45),drvs2dx1(3),drvs2dy1(3),&
      drvs2dz1(3),drvs2dx2(3),drvs2dy2(3),drvs2dz2(3),drvs2dx3(3),drvs2dy3(3),&
      drvs2dz3(3),drvs2dx4(3),drvs2dy4(3),drvs2dz4(3),drvs2(45),drvs3(45),&
      drvs0cx1(3),drvs0cy1(3),drvs0cz1(3),drvs0cx2(3),drvs0cy2(3),drvs0cz2(3),&
      drvs0cx3(3),drvs0cy3(3),drvs0cz3(3),drvs0(45)

     logical :: qvs1,qvs2,qvs3,qvs0



     qvs1=.false.
     qvs1=.false.
     qvs3=.false.
     qvs0=.false.
     
     rvs1=ZERO
     rvs2=ZERO                                                               
     rvs3=ZERO
     rvs0=ZERO
     
     tvs1=ZERO
     tvs2=ZERO
     tvs3=ZERO
     tvs0=ZERO
     
     drvs1=ZERO
     drvs2=ZERO
     drvs3=ZERO
     drvs0=ZERO
     
     tv0x1=ZERO
     tv0y1=ZERO
     tv0z1=ZERO
     tv0x2=ZERO
     tv0y2=ZERO
     tv0z2=ZERO
     tv0x3=ZERO
     tv0y3=ZERO
     tv0z3=ZERO
     tv0x4=ZERO
     tv0y4=ZERO
     tv0z4=ZERO
     tv0x5=ZERO
     tv0y5=ZERO
     tv0z5=ZERO
     sv0x1=ZERO
     sv0y1=ZERO
     sv0z1=ZERO
     sv0x2=ZERO
     sv0y2=ZERO
     sv0z2=ZERO
     sv0x3=ZERO
     sv0y3=ZERO
     sv0z3=ZERO
     tvec=ZERO
     svec=ZERO

     qcal1=vatms(nr)%qdist1 .or. vatms(nr)%qangl1
     qcal2=vatms(nr)%qdist2 .or. vatms(nr)%qangl2 
     qcal3=vatms(nr)%qdist3 .or. vatms(nr)%qangl3
     qcal0=vatms(nr)%qdist0 .or. vatms(nr)%qangl0

!===========================   build vs1     ==========================!
     if(vatms(nr)%vs1 .and. vatms(nr)%catm1 .ne. -1 .and. &
        vatms(nr)%catm2 .ne. -1 .and. vatms(nr)%catm3 .ne. -1) then

        v1=(/X(vatms(nr)%catm1),Y(vatms(nr)%catm1),Z(vatms(nr)%catm1)/)
        v2=(/X(vatms(nr)%catm2),Y(vatms(nr)%catm2),Z(vatms(nr)%catm2)/)
        v3=(/X(vatms(nr)%catm3),Y(vatms(nr)%catm3),Z(vatms(nr)%catm3)/)
        call buildatom(vatms(nr)%b1,vatms(nr)%t1,vatms(nr)%q1,v1,v2,&
                       v3,tvs1,tv0x1,tv0y1,tv0z1,tv0x2,tv0y2,tv0z2,&
                       tv0x3,tv0y3,tv0z3)

        tv0x4=(/ZERO, ZERO, ZERO/)  !dummy arrays
        tv0y4=(/ZERO, ZERO, ZERO/)  !needed to run calcdist123
        tv0z4=(/ZERO, ZERO, ZERO/)  !subroutine 
        tv0x5=(/ZERO, ZERO, ZERO/)  !
        tv0y5=(/ZERO, ZERO, ZERO/)  !
        tv0z5=(/ZERO, ZERO, ZERO/)  !


        drvs1=(/tv0x1, tv0y1, tv0z1, tv0x2, tv0y2, tv0z2, tv0x3,&
               tv0y3, tv0z3, tv0x4, tv0y4, tv0z4, tv0x5, tv0y5,&
               tv0z5/)

!   if any of vs2 atom is requested
        if(vatms(nr)%vs2) then
           rvs1=tvs1
           drvs1dx1=tv0x1
           drvs1dy1=tv0y1
           drvs1dz1=tv0z1
           drvs1dx2=tv0x2
           drvs1dy2=tv0y2
           drvs1dz2=tv0z2
           drvs1dx3=tv0x3
           drvs1dy3=tv0y3
           drvs1dz3=tv0z3
           qvs1=.true.
        endif
     endif


!===========================   build vs0     ==========================!
     if(vatms(nr)%vs0 .and. vatms(nr)%batm1 .ne. -1 .and. &
        vatms(nr)%batm2 .ne. -1 .and. vatms(nr)%batm3 .ne. -1) then

        u1=(/X(vatms(nr)%batm1),Y(vatms(nr)%batm1),Z(vatms(nr)%batm1)/)
        u2=(/X(vatms(nr)%batm2),Y(vatms(nr)%batm2),Z(vatms(nr)%batm2)/)
        u3=(/X(vatms(nr)%batm3),Y(vatms(nr)%batm3),Z(vatms(nr)%batm3)/)
        call buildatom(vatms(nr)%b0,vatms(nr)%t0,vatms(nr)%q0,u1,u2,&
                       u3,tvs0,sv0x1,sv0y1,sv0z1,sv0x2,sv0y2,sv0z2,&
                       sv0x3,sv0y3,sv0z3)
        tvec=ZERO
        drvs0=(/sv0x1, sv0y1, sv0z1, sv0x2, sv0y2, sv0z2, sv0x3,&
                sv0y3, sv0z3,  tvec,  tvec,  tvec,  tvec, tvec,&
                tvec /)
     endif
!===========================   build vs2    ==========================!
     if(vatms(nr)%catm4 .ne. -1 .and. qvs1) then
        v3=(/X(vatms(nr)%catm3),Y(vatms(nr)%catm3),Z(vatms(nr)%catm3)/)
        v4=(/X(vatms(nr)%catm4),Y(vatms(nr)%catm4),Z(vatms(nr)%catm4)/)

        call buildvs2(tvs2,tv0x1,tv0y1,tv0z1,tv0x2,tv0y2,tv0z2,tv0x3,&
           tv0y3,tv0z3,tv0x4,tv0y4,tv0z4,rvs1,vatms(nr)%b2,vatms(nr)%t2,&
           vatms(nr)%q2,v1,v3,v4,drvs1dx1,drvs1dy1,drvs1dz1,drvs1dx2,drvs1dy2,&
           drvs1dz2,drvs1dx3,drvs1dy3,drvs1dz3)
        tv0x5=(/ZERO, ZERO, ZERO/)  !dummy arrays
        tv0y5=(/ZERO, ZERO, ZERO/)  !needed to run calcangl23
        tv0z5=(/ZERO, ZERO, ZERO/)  !subroutine 
        drvs2=(/tv0x1, tv0y1, tv0z1, tv0x2, tv0y2, tv0z2, tv0x3,&
                tv0y3, tv0z3, tv0x4, tv0y4, tv0z4, tv0x5, tv0y5,&
                tv0z5/)

!   if vs3 atom is requested
        if(vatms(nr)%vs3) then
           rvs2=tvs2
           drvs2dx1=tv0x1
           drvs2dy1=tv0y1
           drvs2dz1=tv0z1
           drvs2dx2=tv0x2
           drvs2dy2=tv0y2
           drvs2dz2=tv0z2
           drvs2dx3=tv0x3
           drvs2dy3=tv0y3
           drvs2dz3=tv0z3
           drvs2dx4=tv0x4
           drvs2dy4=tv0y4
           drvs2dz4=tv0z4
           qvs2=.true.
        endif
     endif

!===========================   build vs3    ==========================!
     if(vatms(nr)%catm5 .ne. -1 .and. qvs1 .and. qvs2) then
        v5=(/X(vatms(nr)%catm5),Y(vatms(nr)%catm5),Z(vatms(nr)%catm5)/)
        call buildvs3(tvs3,tv0x1,tv0y1,tv0z1,tv0x2,tv0y2,tv0z2,tv0x3,&
          tv0y3,tv0z3,tv0x4,tv0y4,tv0z4,tv0x5,tv0y5,tv0z5,rvs2,v5,&
          drvs2dx1,drvs2dy1,drvs2dz1,drvs2dx2,drvs2dy2,drvs2dz2,drvs2dx3,&
          drvs2dy3,drvs2dz3,drvs2dx4,drvs2dy4,drvs2dz4)
          drvs3=(/tv0x1, tv0y1, tv0z1, tv0x2, tv0y2, tv0z2, tv0x3,&
                  tv0y3, tv0z3, tv0x4, tv0y4, tv0z4, tv0x5, tv0y5,&
                  tv0z5/)

          rvs3=tvs3
          qvs3=.true.
     endif


!====================== energies =====================================! 

     ev=ZERO
     et=ZERO
     tqc1=ZERO
     tqc2=ZERO
     tqc3=ZERO
     tqc4=ZERO
     tqc5=ZERO
     tqd1=ZERO
     tqd2=ZERO
     tqd3=ZERO
     tqa1=ZERO
     tqa2=ZERO
     tqa3=ZERO
     tqa4=ZERO
     tqa5=ZERO
     tqa6=ZERO
     sqc1=ZERO
     sqc2=ZERO
     sqc3=ZERO
     tqd0=ZERO
     sqa1=ZERO
     sqa2=ZERO


!-----------------  dist0  and angl0  energy calculations--------------!
  calcvs0: if(vatms(nr)%batm1 .ne. -1 .and. vatms(nr)%batm2 .ne. -1 .and. &
     vatms(nr)%batm3 .ne. -1 .and. qcal0) then
       if(vatms(nr)%qdist0 .and. vatms(nr)%nd0 .ne. 0) then
       do j=1,vatms(nr)%nd0
           et=ZERO    
           call calcdist0(et,sqc1,sqc2,sqc3,tqd0,tvs0,drvs0,&
                 vatms(nr)%kdt0,vatms(nr)%dt0,vatms(nr)%dsel0%a(j),X,Y,Z)
           ev=ev+et
           DX(vatms(nr)%batm1)=DX(vatms(nr)%batm1)+sqc1(1)
           DY(vatms(nr)%batm1)=DY(vatms(nr)%batm1)+sqc1(2)
           DZ(vatms(nr)%batm1)=DZ(vatms(nr)%batm1)+sqc1(3)
           DX(vatms(nr)%batm2)=DX(vatms(nr)%batm2)+sqc2(1)
           DY(vatms(nr)%batm2)=DY(vatms(nr)%batm2)+sqc2(2)
           DZ(vatms(nr)%batm2)=DZ(vatms(nr)%batm2)+sqc2(3)
           DX(vatms(nr)%batm3)=DX(vatms(nr)%batm3)+sqc3(1)
           DY(vatms(nr)%batm3)=DY(vatms(nr)%batm3)+sqc3(2)
           DZ(vatms(nr)%batm3)=DZ(vatms(nr)%batm3)+sqc3(3)
           DX(vatms(nr)%dsel0%a(j))=DX(vatms(nr)%dsel0%a(j))+tqd0(1)
           DY(vatms(nr)%dsel0%a(j))=DY(vatms(nr)%dsel0%a(j))+tqd0(2)
           DZ(vatms(nr)%dsel0%a(j))=DZ(vatms(nr)%dsel0%a(j))+tqd0(3)
       enddo
       endif

       if(vatms(nr)%qangl0 .and. vatms(nr)%na0 .ne. 0 ) then
       do j=1,vatms(nr)%na0
           et=ZERO    
           call calcangl1(et,sqc1,sqc2,sqc3,sqa1,sqa2,tvs0,&
                drvs0,vatms(nr)%kth0,vatms(nr)%th0,&
                vatms(nr)%zsel1%a(j),vatms(nr)%zsel2%a(j),X,Y,Z)
           ev=ev+et  
           DX(vatms(nr)%batm1)=DX(vatms(nr)%batm1)+sqc1(1)
           DY(vatms(nr)%batm1)=DY(vatms(nr)%batm1)+sqc1(2)
           DZ(vatms(nr)%batm1)=DZ(vatms(nr)%batm1)+sqc1(3)
           DX(vatms(nr)%batm2)=DX(vatms(nr)%batm2)+sqc2(1)
           DY(vatms(nr)%batm2)=DY(vatms(nr)%batm2)+sqc2(2)
           DZ(vatms(nr)%batm2)=DZ(vatms(nr)%batm2)+sqc2(3)
           DX(vatms(nr)%batm3)=DX(vatms(nr)%batm3)+sqc3(1)
           DY(vatms(nr)%batm3)=DY(vatms(nr)%batm3)+sqc3(2)
           DZ(vatms(nr)%batm3)=DZ(vatms(nr)%batm3)+sqc3(3)
           DX(vatms(nr)%zsel1%a(j))=DX(vatms(nr)%zsel1%a(j))+sqa1(1)
           DY(vatms(nr)%zsel1%a(j))=DY(vatms(nr)%zsel1%a(j))+sqa1(2)
           DZ(vatms(nr)%zsel1%a(j))=DZ(vatms(nr)%zsel1%a(j))+sqa1(3)
           DX(vatms(nr)%zsel2%a(j))=DX(vatms(nr)%zsel2%a(j))+sqa2(1)
           DY(vatms(nr)%zsel2%a(j))=DY(vatms(nr)%zsel2%a(j))+sqa2(2)
           DZ(vatms(nr)%zsel2%a(j))=DZ(vatms(nr)%zsel2%a(j))+sqa2(3)
       enddo
       endif
  
  endif calcvs0
  
!-----------------  dist1  and angl1  energy calculations--------------!
  calcvs1: if(vatms(nr)%catm1 .ne. -1 .and. vatms(nr)%catm2 .ne. -1 .and. &
     vatms(nr)%catm3 .ne. -1 .and. qcal1) then
       if(vatms(nr)%qdist1 .and. vatms(nr)%nd1 .ne. 0) then
       do j=1,vatms(nr)%nd1
           et=ZERO    
           call calcdist123(et,'VS1',tqc1,tqc2,tqc3,tqc4,tqc5,&
                 tqd1,tvs1,drvs1,vatms(nr)%kdt1,vatms(nr)%dt1,&
                 vatms(nr)%dsel1%a(j),X,Y,Z)
           ev=ev+et
           DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
           DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
           DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
           DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
           DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
           DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
           DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
           DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
           DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
           DX(vatms(nr)%dsel1%a(j))=DX(vatms(nr)%dsel1%a(j))+tqd1(1)
           DY(vatms(nr)%dsel1%a(j))=DY(vatms(nr)%dsel1%a(j))+tqd1(2)
           DZ(vatms(nr)%dsel1%a(j))=DZ(vatms(nr)%dsel1%a(j))+tqd1(3)
       enddo
       endif
 
       if(vatms(nr)%qangl1 .and. vatms(nr)%na1 .ne. 0 ) then
       do j=1,vatms(nr)%na1
           et=ZERO    
           call calcangl1(et,tqc1,tqc2,tqc3,tqa1,tqa2,tvs1,&
                drvs1,vatms(nr)%kth1,vatms(nr)%th1,&
                vatms(nr)%asel1%a(j),vatms(nr)%asel2%a(j),X,Y,Z)
           ev=ev+et 
           DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
           DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
           DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
           DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
           DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
           DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
           DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
           DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
           DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
           DX(vatms(nr)%asel1%a(j))=DX(vatms(nr)%asel1%a(j))+tqa1(1)
           DY(vatms(nr)%asel1%a(j))=DY(vatms(nr)%asel1%a(j))+tqa1(2)
           DZ(vatms(nr)%asel1%a(j))=DZ(vatms(nr)%asel1%a(j))+tqa1(3)
           DX(vatms(nr)%asel2%a(j))=DX(vatms(nr)%asel2%a(j))+tqa2(1)
           DY(vatms(nr)%asel2%a(j))=DY(vatms(nr)%asel2%a(j))+tqa2(2)
           DZ(vatms(nr)%asel2%a(j))=DZ(vatms(nr)%asel2%a(j))+tqa2(3)
       enddo
       endif
    endif calcvs1

!-----------------  dist2  and angl2  energy calculations--------------!
     calcvs2: if (vatms(nr)%catm4 .ne. -1 .and. qvs1 .and. qcal2) then
! dist2 energy calculations 
      if(vatms(nr)%qdist2 .and. vatms(nr)%nd2 .ne. 0 ) then
      do j=1,vatms(nr)%nd2 
           et=ZERO   
           call calcdist123(et,'VS2',tqc1,tqc2,tqc3,tqc4,tqc5,&
                tqd2,tvs2,drvs2,vatms(nr)%kdt2,vatms(nr)%dt2,&
                vatms(nr)%dsel2%a(j),X,Y,Z)
           ev=ev+et


!           write(outu,'(a5,5i5,f8.3)')'DBG2d>',nr,vatms(nr)%catm1,&
!           vatms(nr)%catm2,vatms(nr)%catm3,vatms(nr)%catm4,et

           DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
           DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
           DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
           DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
           DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
           DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
           DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
           DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
           DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
           DX(vatms(nr)%catm4)=DX(vatms(nr)%catm4)+tqc4(1)
           DY(vatms(nr)%catm4)=DY(vatms(nr)%catm4)+tqc4(2)
           DZ(vatms(nr)%catm4)=DZ(vatms(nr)%catm4)+tqc4(3)
           DX(vatms(nr)%dsel2%a(j))=DX(vatms(nr)%dsel2%a(j))+tqd2(1)
           DY(vatms(nr)%dsel2%a(j))=DY(vatms(nr)%dsel2%a(j))+tqd2(2)
           DZ(vatms(nr)%dsel2%a(j))=DZ(vatms(nr)%dsel2%a(j))+tqd2(3)
       enddo
       endif

 ! angl2 energy calculations  
       if(vatms(nr)%qangl2 .and. vatms(nr)%na2 .ne. 0 ) then
       do j=1,vatms(nr)%na2
          et=ZERO     
          call calcangl23(et,'VS2',tqc1,tqc2,tqc3,tqc4,tqc5,tqa3,&
          tqa4,tvs2,drvs2,vatms(nr)%kth2,vatms(nr)%th2,&
          vatms(nr)%asel3%a(j),vatms(nr)%asel4%a(j),X,Y,Z)

!          write(outu,'(a5,5i5,f8.3)')'DBG2a>',nr,vatms(nr)%catm1,&
!          vatms(nr)%catm2,vatms(nr)%catm3,vatms(nr)%catm4,et

          ev=ev+et
          DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
          DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
          DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
          DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
          DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
          DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
          DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
          DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
          DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
          DX(vatms(nr)%catm4)=DX(vatms(nr)%catm4)+tqc4(1)
          DY(vatms(nr)%catm4)=DY(vatms(nr)%catm4)+tqc4(2)
          DZ(vatms(nr)%catm4)=DZ(vatms(nr)%catm4)+tqc4(3)
          DX(vatms(nr)%asel3%a(j))=DX(vatms(nr)%asel3%a(j))+tqa3(1)
          DY(vatms(nr)%asel3%a(j))=DY(vatms(nr)%asel3%a(j))+tqa3(2)
          DZ(vatms(nr)%asel3%a(j))=DZ(vatms(nr)%asel3%a(j))+tqa3(3)
          DX(vatms(nr)%asel4%a(j))=DX(vatms(nr)%asel4%a(j))+tqa4(1)
          DY(vatms(nr)%asel4%a(j))=DY(vatms(nr)%asel4%a(j))+tqa4(2)
          DZ(vatms(nr)%asel4%a(j))=DZ(vatms(nr)%asel4%a(j))+tqa4(3)
       enddo
       endif
     endif calcvs2

!-----------------  dist3  and angl3 energy calculations--------------!
     calcvs3: if (vatms(nr)%catm5 .ne. -1 .and. qvs1 .and. qvs2 .and. qcal3) then

!dist3 energy calculations    
      if(vatms(nr)%qdist3 .and. vatms(nr)%nd3 .ne. 0 ) then
      do j=1,vatms(nr)%nd3
         et=ZERO     
         call calcdist123(et,'VS3',tqc1,tqc2,tqc3,tqc4,tqc5,&
            tqd3,tvs3,drvs3,vatms(nr)%kdt3,vatms(nr)%dt3,&
            vatms(nr)%dsel3%a(j),X,Y,Z)
         ev=ev+et

!          write(outu,'(a5,6i5,f8.3)')'DBG3b>',nr,vatms(nr)%catm1,&
!          vatms(nr)%catm2,vatms(nr)%catm3,vatms(nr)%catm4,&
!          vatms(nr)%catm5,et

         DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
         DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
         DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
         DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
         DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
         DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
         DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
         DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
         DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
         DX(vatms(nr)%catm4)=DX(vatms(nr)%catm4)+tqc4(1)
         DY(vatms(nr)%catm4)=DY(vatms(nr)%catm4)+tqc4(2)
         DZ(vatms(nr)%catm4)=DZ(vatms(nr)%catm4)+tqc4(3)
         DX(vatms(nr)%catm5)=DX(vatms(nr)%catm5)+tqc5(1)
         DY(vatms(nr)%catm5)=DY(vatms(nr)%catm5)+tqc5(2)
         DZ(vatms(nr)%catm5)=DZ(vatms(nr)%catm5)+tqc5(3)
         DX(vatms(nr)%dsel3%a(j))=DX(vatms(nr)%dsel3%a(j))+tqd3(1)
         DY(vatms(nr)%dsel3%a(j))=DY(vatms(nr)%dsel3%a(j))+tqd3(2)
         DZ(vatms(nr)%dsel3%a(j))=DZ(vatms(nr)%dsel3%a(j))+tqd3(3)
      enddo
      endif

!angl3 energy calculations
      if(vatms(nr)%qangl3 .and. vatms(nr)%na3 .ne. 0 ) then
      do j=1,vatms(nr)%na3
        et=ZERO      
        call calcangl23(et,'VS3',tqc1,tqc2,tqc3,tqc4,tqc5,tqa5,&
           tqa6,tvs3,drvs3,vatms(nr)%kth3,vatms(nr)%th3,&
           vatms(nr)%asel5%a(j),vatms(nr)%asel6%a(j),X,Y,Z)
!          write(outu,'(a5,6i5,f8.3)')'DBG3a>',nr,vatms(nr)%catm1,&
!          vatms(nr)%catm2,vatms(nr)%catm3,vatms(nr)%catm4,&
!          vatms(nr)%catm5,et
        ev=ev+et
        DX(vatms(nr)%catm1)=DX(vatms(nr)%catm1)+tqc1(1)
        DY(vatms(nr)%catm1)=DY(vatms(nr)%catm1)+tqc1(2)
        DZ(vatms(nr)%catm1)=DZ(vatms(nr)%catm1)+tqc1(3)
        DX(vatms(nr)%catm2)=DX(vatms(nr)%catm2)+tqc2(1)
        DY(vatms(nr)%catm2)=DY(vatms(nr)%catm2)+tqc2(2)
        DZ(vatms(nr)%catm2)=DZ(vatms(nr)%catm2)+tqc2(3)
        DX(vatms(nr)%catm3)=DX(vatms(nr)%catm3)+tqc3(1)
        DY(vatms(nr)%catm3)=DY(vatms(nr)%catm3)+tqc3(2)
        DZ(vatms(nr)%catm3)=DZ(vatms(nr)%catm3)+tqc3(3)
        DX(vatms(nr)%catm4)=DX(vatms(nr)%catm4)+tqc4(1)
        DY(vatms(nr)%catm4)=DY(vatms(nr)%catm4)+tqc4(2)
        DZ(vatms(nr)%catm4)=DZ(vatms(nr)%catm4)+tqc4(3)
        DX(vatms(nr)%catm5)=DX(vatms(nr)%catm5)+tqc5(1)
        DY(vatms(nr)%catm5)=DY(vatms(nr)%catm5)+tqc5(2)
        DZ(vatms(nr)%catm5)=DZ(vatms(nr)%catm5)+tqc5(3)
        DX(vatms(nr)%asel5%a(j))=DX(vatms(nr)%asel5%a(j))+tqa5(1)
        DY(vatms(nr)%asel5%a(j))=DY(vatms(nr)%asel5%a(j))+tqa5(2)
        DZ(vatms(nr)%asel5%a(j))=DZ(vatms(nr)%asel5%a(j))+tqa5(3)
        DX(vatms(nr)%asel6%a(j))=DX(vatms(nr)%asel6%a(j))+tqa6(1)
        DY(vatms(nr)%asel6%a(j))=DY(vatms(nr)%asel6%a(j))+tqa6(2)
        DZ(vatms(nr)%asel6%a(j))=DZ(vatms(nr)%asel6%a(j))+tqa6(3)
      enddo
      endif
     endif calcvs3

     end subroutine calcprimoene

     subroutine calcangl1(ee,dfrq1,dfrq2,dfrq3,dfrq6,dfrq7,v0,&
                   dv0drv,kth,th0,pa1,pa2,X,Y,Z)
!--------------------------------------------!      
!                                            !
!             <0>                            !
!             .   \                          ! 
!            .     \  THETA (th0)            ! 
!   [3]----[1]      \                        !
!           |       [6]------[7]             !
!           |                                !
!          [2]                               !
!                                            !
!                                            !
!--------------------------------------------!

  use vector
  use stream
  use consta
  use number
  use contrl
  use psf
     integer,intent(in) :: pa1,pa2 !atoms involved in angl potential.
     real(chm_real),intent(in) :: kth,th0,v0(3),dv0drv(45)
     real(chm_real),intent(out) :: ee,dfrq1(3),dfrq2(3),dfrq3(3),&
               dfrq6(3),dfrq7(3)
     real(chm_real),intent(in),dimension(natom) :: X,Y,Z

!    local vars
     real(chm_real) :: tf,tht,r60(3),r67(3),n60(3),n67(3),d60,d67,&
                       tx,ty,tz,td,th1,t1,t2,t3,dfee,dftf

     real(chm_real),dimension(3) :: s1,s2,dv0dx1,dv0dy1,&
                   dv0dz1,dv0dx2,dv0dy2,dv0dz2,dv0dx3,dv0dy3,dv0dz3,&
     dfr60dx1,dfr60dy1,dfr60dz1,dfr60dx2,dfr60dy2,dfr60dz2,dfr60dx3,&
     dfr60dy3,dfr60dz3,dfr60dx6,dfr60dy6,dfr60dz6,dfn60dx1,dfn60dy1,&
     dfn60dz1,dfn60dx2,dfn60dy2,dfn60dz2,dfn60dx3,dfn60dy3,dfn60dz3,&
     dfn60dx6,dfn60dy6,dfn60dz6,dfr67dx6,dfr67dy6,dfr67dz6,dfn67dx6,&
     dfn67dy6,dfn67dz6,dfr67dx7,dfr67dy7,dfr67dz7,dfn67dx7,dfn67dy7,&
     dfn67dz7,dfthtdq1,dfthtdq2,dfthtdq3,dfthtdq6,dfthtdq7
     

     ee=ZERO
     dfee=ZERO
     dfrq1=ZERO
     dfrq2=ZERO
     dfrq3=ZERO
     dfrq6=ZERO
     dfrq7=ZERO
        
     s1=(/X(pa1), Y(pa1), Z(pa1)/) !q6
     s2=(/X(pa2), Y(pa2), Z(pa2)/) !q7

     dv0dx1=dv0drv(1:3)   !derv q1 (catm1)
     dv0dy1=dv0drv(4:6)
     dv0dz1=dv0drv(7:9)
     dv0dx2=dv0drv(10:12) !derv q2 (catm2)
     dv0dy2=dv0drv(13:15)
     dv0dz2=dv0drv(16:18)
     dv0dx3=dv0drv(19:21) !derv q3 (catm3)
     dv0dy3=dv0drv(22:24)
     dv0dz3=dv0drv(25:27)

!            r60         r67
!    (v0,0)------(r6,s1)-----(r7,s2)
!        
     r67=(/ s2(1)-s1(1), s2(2)-s1(2), s2(3)-s1(3) /)
     r60=(/ v0(1)-s1(1), v0(2)-s1(2), v0(3)-s1(3) /)
     d60=sqrt(r60(1)**2+r60(2)**2+r60(3)**2)
     d67=sqrt(r67(1)**2+r67(2)**2+r67(3)**2)
     n60=r60/d60
     n67=r67/d67
 
!==================================================!    
!                     qk=1  D[v0]/D[q1]            !  
!                     qk=2  D[v0]/D[q2]            !
!     D[r60]/D[qk] =  qk=3  D[v0]/D[q3]            !  
!                     qk=6  -1                     !
!                     qk=7   0                     !
!==================================================!

!     D[r60]/D[q1]
      dfr60dx1=dv0dx1
      dfr60dy1=dv0dy1
      dfr60dz1=dv0dz1

!     D[r60]/D[q2]
      dfr60dx2=dv0dx2
      dfr60dy2=dv0dy2
      dfr60dz2=dv0dz2

!     D[r60]/D[q3]
      dfr60dx3=dv0dx3
      dfr60dy3=dv0dy3
      dfr60dz3=dv0dz3
           
!     D[r60]/D[q4]
      dfr60dx6=(/-1.0,   0.0,    0.0/)
      dfr60dy6=(/ 0.0,  -1.0,    0.0/)
      dfr60dz6=(/ 0.0,   0.0,   -1.0/)

!==================================================================!
!                                                                  !
!    D[d60]/D[qk] = r60 . D[r60]/D[qk]/d60                         !
!                   _                             _                !
!                  |       D[r60]          D[d60]  |    /          !
!    D[n60]/D[qk] =| d60 * ------  - r60 * ------  |   /   d60**2  !
!                  |_      D[qk]           D[qk]  _|  /            !
!                                                                  !
!==================================================================!
!     D[d60]/D[q1]  &&  D[n60]/D[q1]
      call nvecdrv(r60,dfr60dx1,dfn60dx1)
      call nvecdrv(r60,dfr60dy1,dfn60dy1)
      call nvecdrv(r60,dfr60dz1,dfn60dz1)

!     D[d60]/D[x2]  &&  D[n60]/D[x2]
      call nvecdrv(r60,dfr60dx2,dfn60dx2)
      call nvecdrv(r60,dfr60dy2,dfn60dy2)
      call nvecdrv(r60,dfr60dz2,dfn60dz2)

!     D[d60]/D[x3]  &&  D[n60]/D[x3]
      call nvecdrv(r60,dfr60dx3,dfn60dx3)
      call nvecdrv(r60,dfr60dy3,dfn60dy3)
      call nvecdrv(r60,dfr60dz3,dfn60dz3)

!     D[d60]/D[x4]  &&  D[n60]/D[x4]
      call nvecdrv(r60,dfr60dx6,dfn60dx6)
      call nvecdrv(r60,dfr60dy6,dfn60dy6)
      call nvecdrv(r60,dfr60dz6,dfn60dz6)


!==================================================!    
!                     qk=1  0                      !  
!                     qk=2  0                      !
!     D[r67]/D[qk] =  qk=3  0                      !  
!                     qk=4 -1                      !
!                     qk=5  1                      !
!==================================================!

      dfr67dx6=(/ MINONE,  ZERO,   ZERO   /)
      dfr67dy6=(/ ZERO,    MINONE, ZERO   /)
      dfr67dz6=(/ ZERO,    ZERO,   MINONE /)

      dfr67dx7=(/ ONE,     ZERO,   ZERO   /)
      dfr67dy7=(/ ZERO,    ONE,    ZERO   /)
      dfr67dz7=(/ ZERO,    ZERO,   ONE    /)

!==================================================================!
!                                                                  !
!    D[d67]/D[qk] = r67 . D[r67]/D[qk]/d67                         !
!                   _                             _                !
!                  |       D[r67]          D[d67]  |    /          !
!    D[n67]/D[qk] =| d67 * ------  - r67 * ------  |   /   d67**2  !
!                  |_      D[qk]           D[qk]  _|  /            !
!                                                                  !
!==================================================================!


!     D[d67]/D[x4]  &&  D[n67]/D[x6]
      call nvecdrv(r67,dfr67dx6,dfn67dx6)
      call nvecdrv(r67,dfr67dy6,dfn67dy6)
      call nvecdrv(r67,dfr67dz6,dfn67dz6)

!     D[d67]/D[x5]  &&  D[n67]/D[x7]
      call nvecdrv(r67,dfr67dx7,dfn67dx7)
      call nvecdrv(r67,dfr67dy7,dfn67dy7)
      call nvecdrv(r67,dfr67dz7,dfn67dz7)

!
!     Energies and Derivatives
!

     call getcosangl2(v0,s1,s2,tf) ! q0---q6---q7
     tht=acos(tf)
     dftf=-1.0/sqrt(1-(tf**2))
     th1=DEGRAD*th0
     ee=kth*(tht-th1)**2
     dfee=2.0*kth*(tht-th1)


!
!   tht=acos(tf)
!   dftf=-1/sqrt(1-tf**2)
!   D[tht]/D[qk] =  dftf*[ (n60.D[n67]/D[qk]) + (D[n60]/D[qk].n67) ]
!

     call DOTPR(n67,dfn60dx1,3,t1)
     call DOTPR(n67,dfn60dy1,3,t2)
     call DOTPR(n67,dfn60dz1,3,t3)
     dfthtdq1=(/ t1, t2, t3 /)

     call DOTPR(n67,dfn60dx2,3,t1)
     call DOTPR(n67,dfn60dy2,3,t2)
     call DOTPR(n67,dfn60dz2,3,t3)
     dfthtdq2=(/ t1, t2, t3 /)

     call DOTPR(n67,dfn60dx3,3,t1)
     call DOTPR(n67,dfn60dy3,3,t2)
     call DOTPR(n67,dfn60dz3,3,t3)
     dfthtdq3=(/ t1, t2, t3 /)

     call DOTPR(n60,dfn67dx6,3,tx) 
     call DOTPR(n60,dfn67dy6,3,ty)
     call DOTPR(n60,dfn67dz6,3,tz)
     call DOTPR(n67,dfn60dx6,3,t1)
     call DOTPR(n67,dfn60dy6,3,t2)
     call DOTPR(n67,dfn60dz6,3,t3)
     dfthtdq6=(/ t1+tx, t2+ty, t3+tz /)

     call DOTPR(n60,dfn67dx7,3,tx) 
     call DOTPR(n60,dfn67dy7,3,ty)
     call DOTPR(n60,dfn67dz7,3,tz)
     dfthtdq7=(/ tx, ty, tz /)

     dfrq1=dfee*dftf*dfthtdq1
     dfrq2=dfee*dftf*dfthtdq2
     dfrq3=dfee*dftf*dfthtdq3
     dfrq6=dfee*dftf*dfthtdq6
     dfrq7=dfee*dftf*dfthtdq7
      end subroutine calcangl1

     subroutine calcdist0(ee,dfrp1,dfrp2,dfrp3,dfrp6,u0,du0drv,kdt,dt0,pa,X,Y,Z)
  use vector
  use stream
  use consta
  use number
  use contrl
  use psf

     integer,intent(in) :: pa !atoms involved in dist potential.
     real(chm_real),intent(in) :: kdt,dt0,u0(3),du0drv(45)
     real(chm_real),intent(out) :: ee,dfrp1(3),dfrp2(3),dfrp3(3),dfrp6(3)
     real(chm_real),intent(in),dimension(natom) :: X,Y,Z
!    local vars
     real(chm_real) :: dt,tx,ty,tz,tt,dfee
     real(chm_real),dimension(3) :: s1,du0dx1,du0dy1,du0dz1,du0dx2,&
     du0dy2,du0dz2,du0dx3,du0dy3,du0dz3,du0dx4,du0dy4,du0dz4,du0dx5,&
     du0dy5,du0dz5,r06,tr1,tr2,tr3

     ee=ZERO
     dfrp1=ZERO
     dfrp2=ZERO
     dfrp3=ZERO
     s1=(/X(pa), Y(pa), Z(pa)/) !q6

     du0dx1=du0drv(1:3)  !derv q1 (batm1)
     du0dy1=du0drv(4:6)
     du0dz1=du0drv(7:9)
     du0dx2=du0drv(10:12) !derv q2 (batm2)
     du0dy2=du0drv(13:15)
     du0dz2=du0drv(16:18)
     du0dx3=du0drv(19:21) !derv q3 (batm3)
     du0dy3=du0drv(22:24)
     du0dz3=du0drv(25:27)

     r06=(/s1(1)-u0(1), s1(2)-u0(2), s1(3)-u0(3)/)
     dt=sqrt(r06(1)**2+r06(2)**2+r06(3)**2)
     ee=kdt*(dt-dt0)**2
     dfee=2*kdt*(dt-dt0)

!    D[q6]        
     tx=dfee*(r06(1)/dt)
     ty=dfee*(r06(2)/dt)
     tz=dfee*(r06(3)/dt)
     dfrp6=(/tx, ty, tz /)

!    D[qk] = (r06.D[u0]/dqk)/dt
     call DOTPR(du0dx1,r06,3,tt)
     tx=(-1.0*dfee*tt)/dt
     call DOTPR(du0dy1,r06,3,tt)
     ty=(-1.0*dfee*tt)/dt
     call DOTPR(du0dz1,r06,3,tt)
     tz=(-1.0*dfee*tt)/dt
     dfrp1=(/tx, ty, tz /)

     call DOTPR(du0dx2,r06,3,tt)
     tx=(-1.0*dfee*tt)/dt
     call DOTPR(du0dy2,r06,3,tt)
     ty=(-1.0*dfee*tt)/dt
     call DOTPR(du0dz2,r06,3,tt)
     tz=(-1.0*dfee*tt)/dt
     dfrp2=(/tx, ty, tz/)

     call DOTPR(du0dx3,r06,3,tt)
     tx=(-1.0*dfee*tt)/dt
     call DOTPR(du0dy3,r06,3,tt)
     ty=(-1.0*dfee*tt)/dt
     call DOTPR(du0dz3,r06,3,tt)
     tz=(-1.0*dfee*tt)/dt
     dfrp3=(/tx, ty, tz/)
     end subroutine calcdist0

     subroutine calcdist123(ee,vtyp,dfrq1,dfrq2,dfrq3,dfrq4,dfrq5,&
             dfrq6,v0,dv0drv,kdt,dt0,pa,X,Y,Z)

  use vector
  use stream
  use consta
  use number
  use contrl
  use psf

     integer,intent(in) :: pa !atoms involved in dist potential.
     real(chm_real),intent(in) :: kdt,dt0,v0(3),dv0drv(45)
     real(chm_real),intent(out) :: ee,dfrq1(3),dfrq2(3),dfrq3(3),&
               dfrq4(3),dfrq5(3),dfrq6(3)
     real(chm_real),intent(in),dimension(natom) :: X,Y,Z
     character(len=3),intent(in) :: vtyp

!    local vars
     real(chm_real) :: dt,tx,ty,tz,tt,dfee

     real(chm_real),dimension(3) :: s1,dv0dx1,dv0dy1,dv0dz1,dv0dx2,&
     dv0dy2,dv0dz2,dv0dx3,dv0dy3,dv0dz3,dv0dx4,dv0dy4,dv0dz4,dv0dx5,&
     dv0dy5,dv0dz5,r06,tr1,tr2,tr3


     ee=ZERO
     dfrq1=ZERO
     dfrq2=ZERO
     dfrq3=ZERO
     dfrq4=ZERO
     dfrq5=ZERO
     dfrq6=ZERO
        
     s1=(/X(pa), Y(pa), Z(pa)/) !q6

     dv0dx1=dv0drv(1:3)   !derv q1 (catm1)
     dv0dy1=dv0drv(4:6)
     dv0dz1=dv0drv(7:9)
     dv0dx2=dv0drv(10:12) !derv q2 (catm2)
     dv0dy2=dv0drv(13:15)
     dv0dz2=dv0drv(16:18)
     dv0dx3=dv0drv(19:21) !derv q3 (catm3)
     dv0dy3=dv0drv(22:24)
     dv0dz3=dv0drv(25:27)

     if (vtyp .eq. 'VS2' .or. vtyp .eq. 'VS3') then    
        dv0dx4=dv0drv(28:30) !derv q4 (catm4)
        dv0dy4=dv0drv(31:33)
        dv0dz4=dv0drv(34:36)

        if (vtyp .eq. 'VS3') then
          dv0dx5=dv0drv(37:39) !derv q5 (catm5)
          dv0dy5=dv0drv(40:42)
          dv0dz5=dv0drv(43:45)
        else
          dv0dx5=ZERO
          dv0dy5=ZERO
          dv0dz5=ZERO
        endif
     else
        dv0dx4=ZERO
        dv0dy4=ZERO
        dv0dz4=ZERO
        dv0dx5=ZERO
        dv0dy5=ZERO
        dv0dz5=ZERO
     endif

        r06=(/s1(1)-v0(1), s1(2)-v0(2), s1(3)-v0(3)/)
        dt=sqrt(r06(1)**2+r06(2)**2+r06(3)**2)
        ee=kdt*(dt-dt0)**2
        dfee=2*kdt*(dt-dt0)

!       D[q6]        
        tx=dfee*(r06(1)/dt)
        ty=dfee*(r06(2)/dt)
        tz=dfee*(r06(3)/dt)
        dfrq6=(/tx, ty, tz /)

!       D[qk] = (r06.D[v0]/dqk)/dt
        call DOTPR(dv0dx1,r06,3,tt)
        tx=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dy1,r06,3,tt)
        ty=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dz1,r06,3,tt)
        tz=(-1.0*dfee*tt)/dt
        dfrq1=(/tx, ty, tz /)

        call DOTPR(dv0dx2,r06,3,tt)
        tx=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dy2,r06,3,tt)
        ty=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dz2,r06,3,tt)
        tz=(-1.0*dfee*tt)/dt
        dfrq2=(/tx, ty, tz/)

        call DOTPR(dv0dx3,r06,3,tt)
        tx=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dy3,r06,3,tt)
        ty=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dz3,r06,3,tt)
        tz=(-1.0*dfee*tt)/dt
        dfrq3=(/tx, ty, tz/)

!       D[q4] non zero for dist2 and dist3        
        call DOTPR(dv0dx4,r06,3,tt)
        tx=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dy4,r06,3,tt)
        ty=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dz4,r06,3,tt)
        tz=(-1.0*dfee*tt)/dt
        dfrq4=(/tx, ty, tz/)

!       D[q5] valid only for dist3        
        call DOTPR(dv0dx5,r06,3,tt)
        tx=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dy5,r06,3,tt)
        ty=(-1.0*dfee*tt)/dt
        call DOTPR(dv0dz5,r06,3,tt)
        tz=(-1.0*dfee*tt)/dt
        dfrq5=(/tx, ty, tz/)

     end subroutine calcdist123


     subroutine calcangl23(ee,vtyp,dfrq1,dfrq2,dfrq3,dfrq4,dfrq5,dfrq6,&
          dfrq7,v0,dv0drv,kth,th0,pa1,pa2,X,Y,Z)

!----------------------------------------!-------------------------------------!     
!*****************CALCANGL2**************|**************CALCANGL3**************!
!      <vs1>                 [6]         |  <vs1>                 [5]      [6]!
!         .                /             |    .                 /   .    /     !
!          .              /              |     .               /     .  /      !
!         [1]----[3]....<0>  THETA (th0) |      [1]----[3]..<vs2>    <0> THETA !  
!         /        \       \             |      /        \              \ (th0)!
!        /          \       \            |     /          \              \     !
!    [2]            [4]      [7]         |   [2]         [4]             [7]   !
!                                        |                                     !
!                                        |                                     !
!----------------------------------------!-------------------------------------!
  use vector
  use stream
  use consta
  use number
  use contrl
  use psf

     integer,intent(in) :: pa1,pa2 !atoms involved in angl potential.
     real(chm_real),intent(in) :: kth,th0,v0(3),dv0drv(45)
     real(chm_real),intent(out) :: ee,dfrq1(3),dfrq2(3),dfrq3(3),&
               dfrq4(3),dfrq5(3),dfrq6(3),dfrq7(3)
     real(chm_real),intent(in),dimension(natom) :: X,Y,Z
     character(len=3) :: vtyp

!    local vars
     real(chm_real) :: tf,tht,r06(3),r07(3),n06(3),n07(3),d06,d07,&
                       tx,ty,tz,td,th1,t1,t2,t3,dfee,dftf

     real(chm_real),dimension(3) :: s1,s2,dv0dx1,dv0dy1,dv0dz1,dv0dx2,&
     dv0dy2,dv0dz2,dv0dx3,dv0dy3,dv0dz3,dv0dx4,dv0dy4,dv0dz4,dv0dx5,&
     dv0dy5,dv0dz5,tr1,tr2,tr3

     real(chm_real),dimension(3)::dfr06dx1,dfr06dy1,dfr06dz1,dfr06dx2,&
     dfr06dy2,dfr06dz2,dfr06dx3,dfr06dy3,dfr06dz3,dfr06dx4,dfr06dy4,&
     dfr06dz4,dfr06dx6,dfr06dy6,dfr06dz6,dfn06dy1,dfn06dz1,dfn06dx2,&
     dfn06dy2,dfn06dz2,dfn06dx3,dfn06dy3,dfn06dz3,dfn06dx4,dfn06dy4,&
     dfn06dz4,dfn06dx6,dfn06dy6,dfn06dz6,dfn06dx1,dfr06dx5,dfr06dy5,&
     dfr06dz5,dfn06dx5,dfn06dy5,dfn06dz5

     real(chm_real),dimension(3)::dfr07dx1,dfr07dy1,dfr07dz1,dfr07dx2,&
     dfr07dy2,dfr07dz2,dfr07dx3,dfr07dy3,dfr07dz3,dfr07dx4,dfr07dy4,&
     dfr07dz4,dfr07dx7,dfr07dy7,dfr07dz7,dfn07dx1,dfn07dy1,dfn07dz1,&
     dfn07dx2,dfn07dy2,dfn07dz2,dfn07dx3,dfn07dy3,dfn07dz3,dfn07dx4,&
     dfn07dy4,dfn07dz4,dfn07dx7,dfn07dy7,dfn07dz7,dfr07dx5,dfr07dy5,&
     dfr07dz5,dfn07dx5,dfn07dy5,dfn07dz5,&
     dfthtdq1,dfthtdq2,dfthtdq3,dfthtdq4,dfthtdq5,dfthtdq6,dfthtdq7
     

     ee=ZERO
     dfrq1=ZERO
     dfrq2=ZERO
     dfrq3=ZERO
     dfrq4=ZERO
     dfrq5=ZERO
     dfrq6=ZERO
     dfrq7=ZERO
        
     s1=(/X(pa1), Y(pa1), Z(pa1)/) !q6
     s2=(/X(pa2), Y(pa2), Z(pa2)/) !q7


     dv0dx1=dv0drv(1:3)   !derv q1 (catm1)
     dv0dy1=dv0drv(4:6)
     dv0dz1=dv0drv(7:9)
     dv0dx2=dv0drv(10:12) !derv q2 (catm2)
     dv0dy2=dv0drv(13:15)
     dv0dz2=dv0drv(16:18)
     dv0dx3=dv0drv(19:21) !derv q3 (catm3)
     dv0dy3=dv0drv(22:24)
     dv0dz3=dv0drv(25:27)
     dv0dx4=dv0drv(28:30) !derv q4 (catm4)
     dv0dy4=dv0drv(31:33)
     dv0dz4=dv0drv(34:36)

     if (vtyp .eq. 'VS3') then
        dv0dx5=dv0drv(37:39) !derv q5 (catm5)
        dv0dy5=dv0drv(40:42)
        dv0dz5=dv0drv(43:45)
     else
        dv0dx5=ZERO
        dv0dy5=ZERO
        dv0dz5=ZERO
     endif

!        r06    r07
!    (6)----(0)----(7)
!   

     r06=(/ s1(1)-v0(1), s1(2)-v0(2), s1(3)-v0(3) /)
     r07=(/ s2(1)-v0(1), s2(2)-v0(2), s2(3)-v0(3) /)
     d06=sqrt(r06(1)**2+r06(2)**2+r06(3)**2)
     d07=sqrt(r07(1)**2+r07(2)**2+r07(3)**2)
     n06=r06/d06
     n07=r07/d07

!==================================================!    
!                     qk=1..5  D[v0]/D[qk]         !
!     D[r06]/D[qk] =                               !  
!                     qk=6  1                      !
!                     qk=7  0                      !
!==================================================!

!   D[r06]/D[qk]
    dfr06dx1=-dv0dx1
    dfr06dy1=-dv0dy1
    dfr06dz1=-dv0dz1
    dfr06dx2=-dv0dx2
    dfr06dy2=-dv0dy2
    dfr06dz2=-dv0dz2
    dfr06dx3=-dv0dx3
    dfr06dy3=-dv0dy3
    dfr06dz3=-dv0dz3
    dfr06dx4=-dv0dx4
    dfr06dy4=-dv0dy4
    dfr06dz4=-dv0dz4
    dfr06dx5=-dv0dx5
    dfr06dy5=-dv0dy5
    dfr06dz5=-dv0dz5
    dfr06dx6=(/ ONE,   ZERO,  ZERO /)
    dfr06dy6=(/ ZERO,  ONE,   ZERO /)
    dfr06dz6=(/ ZERO,  ZERO,  ONE  /)


!   D[n06]/D[qk] 
    call nvecdrv(r06,dfr06dx1,dfn06dx1)
    call nvecdrv(r06,dfr06dy1,dfn06dy1)
    call nvecdrv(r06,dfr06dz1,dfn06dz1)
 
    call nvecdrv(r06,dfr06dx2,dfn06dx2)
    call nvecdrv(r06,dfr06dy2,dfn06dy2)
    call nvecdrv(r06,dfr06dz2,dfn06dz2)

    call nvecdrv(r06,dfr06dx3,dfn06dx3)
    call nvecdrv(r06,dfr06dy3,dfn06dy3)
    call nvecdrv(r06,dfr06dz3,dfn06dz3)

    call nvecdrv(r06,dfr06dx4,dfn06dx4)
    call nvecdrv(r06,dfr06dy4,dfn06dy4)
    call nvecdrv(r06,dfr06dz4,dfn06dz4)

    call nvecdrv(r06,dfr06dx5,dfn06dx5)
    call nvecdrv(r06,dfr06dy5,dfn06dy5)
    call nvecdrv(r06,dfr06dz5,dfn06dz5)

    call nvecdrv(r06,dfr06dx6,dfn06dx6)
    call nvecdrv(r06,dfr06dy6,dfn06dy6)
    call nvecdrv(r06,dfr06dz6,dfn06dz6)


!==================================================!    
!                     qk=1..5  D[v0]/D[qk]         !
!     D[r07]/D[qk] =                               !  
!                     qk=6  0                      !
!                     qk=7  1                      !
!==================================================!
!   D[r07]/D[qk]
    dfr07dx1=-dv0dx1
    dfr07dy1=-dv0dy1
    dfr07dz1=-dv0dz1
    dfr07dx2=-dv0dx2
    dfr07dy2=-dv0dy2
    dfr07dz2=-dv0dz2
    dfr07dx3=-dv0dx3
    dfr07dy3=-dv0dy3
    dfr07dz3=-dv0dz3
    dfr07dx4=-dv0dx4
    dfr07dy4=-dv0dy4
    dfr07dz4=-dv0dz4
    dfr07dx5=-dv0dx5
    dfr07dy5=-dv0dy5
    dfr07dz5=-dv0dz5
    dfr07dx7=(/ ONE,   ZERO,  ZERO /)
    dfr07dy7=(/ ZERO,  ONE,   ZERO /)
    dfr07dz7=(/ ZERO,  ZERO,  ONE  /)

!   D[n07]/D[qk] 
    call nvecdrv(r07,dfr07dx1,dfn07dx1)
    call nvecdrv(r07,dfr07dy1,dfn07dy1)
    call nvecdrv(r07,dfr07dz1,dfn07dz1)
 
    call nvecdrv(r07,dfr07dx2,dfn07dx2)
    call nvecdrv(r07,dfr07dy2,dfn07dy2)
    call nvecdrv(r07,dfr07dz2,dfn07dz2)

    call nvecdrv(r07,dfr07dx3,dfn07dx3)
    call nvecdrv(r07,dfr07dy3,dfn07dy3)
    call nvecdrv(r07,dfr07dz3,dfn07dz3)

    call nvecdrv(r07,dfr07dx4,dfn07dx4)
    call nvecdrv(r07,dfr07dy4,dfn07dy4)
    call nvecdrv(r07,dfr07dz4,dfn07dz4)

    call nvecdrv(r07,dfr07dx5,dfn07dx5)
    call nvecdrv(r07,dfr07dy5,dfn07dy5)
    call nvecdrv(r07,dfr07dz5,dfn07dz5)

    call nvecdrv(r07,dfr07dx7,dfn07dx7)
    call nvecdrv(r07,dfr07dy7,dfn07dy7)
    call nvecdrv(r07,dfr07dz7,dfn07dz7)

!   tht=acos(tf)
!   dftf=-1/sqrt(1-tf**2)
!   D[tht]/D[qk] =  dftf*[ (n07.D[n07]/D[qk]) + (D[n07]/D[qk].n07) ]
!
     call getcosangl2(s1,v0,s2,tf) ! q6---q0---q7
     tht=acos(tf)
     dftf=-1.0/sqrt(1-(tf**2))
     th1=DEGRAD*th0
     ee=kth*(tht-th1)**2
     dfee=2.0*kth*(tht-th1)

     call DOTPR(n07,dfn06dx1,3,t1)
     call DOTPR(n07,dfn06dy1,3,t2)
     call DOTPR(n07,dfn06dz1,3,t3)
     call DOTPR(n06,dfn07dx1,3,tx)
     call DOTPR(n06,dfn07dy1,3,ty)
     call DOTPR(n06,dfn07dz1,3,tz)
     dfthtdq1=(/t1+tx, t2+ty, t3+tz/)

     call DOTPR(n07,dfn06dx2,3,t1)
     call DOTPR(n07,dfn06dy2,3,t2)
     call DOTPR(n07,dfn06dz2,3,t3)
     call DOTPR(n06,dfn07dx2,3,tx)
     call DOTPR(n06,dfn07dy2,3,ty)
     call DOTPR(n06,dfn07dz2,3,tz)
     dfthtdq2=(/t1+tx, t2+ty, t3+tz/)

     call DOTPR(n07,dfn06dx3,3,t1)
     call DOTPR(n07,dfn06dy3,3,t2)
     call DOTPR(n07,dfn06dz3,3,t3)
     call DOTPR(n06,dfn07dx3,3,tx)
     call DOTPR(n06,dfn07dy3,3,ty)
     call DOTPR(n06,dfn07dz3,3,tz)
     dfthtdq3=(/t1+tx, t2+ty, t3+tz/)

     call DOTPR(n07,dfn06dx4,3,t1)
     call DOTPR(n07,dfn06dy4,3,t2)
     call DOTPR(n07,dfn06dz4,3,t3)
     call DOTPR(n06,dfn07dx4,3,tx)
     call DOTPR(n06,dfn07dy4,3,ty)
     call DOTPR(n06,dfn07dz4,3,tz)
     dfthtdq4=(/t1+tx, t2+ty, t3+tz/)

     call DOTPR(n07,dfn06dx5,3,t1)
     call DOTPR(n07,dfn06dy5,3,t2)
     call DOTPR(n07,dfn06dz5,3,t3)
     call DOTPR(n06,dfn07dx5,3,tx)
     call DOTPR(n06,dfn07dy5,3,ty)
     call DOTPR(n06,dfn07dz5,3,tz)
     dfthtdq5=(/t1+tx, t2+ty, t3+tz/)

     call DOTPR(n07,dfn06dx6,3,t1)
     call DOTPR(n07,dfn06dy6,3,t2)
     call DOTPR(n07,dfn06dz6,3,t3)
     dfthtdq6=(/t1, t2, t3/)

     call DOTPR(n06,dfn07dx7,3,tx)
     call DOTPR(n06,dfn07dy7,3,ty)
     call DOTPR(n06,dfn07dz7,3,tz)
     dfthtdq7=(/tx, ty, tz/)

     dfrq1=dfee*dftf*dfthtdq1
     dfrq2=dfee*dftf*dfthtdq2
     dfrq3=dfee*dftf*dfthtdq3
     dfrq4=dfee*dftf*dfthtdq4
     dfrq5=dfee*dftf*dfthtdq5
     dfrq6=dfee*dftf*dfthtdq6
     dfrq7=dfee*dftf*dfthtdq7

     end subroutine calcangl23

      subroutine buildvs2(vs2,dvs2x1,dvs2y1,dvs2z1,dvs2x2,dvs2y2,dvs2z2,&
           dvs2x3,dvs2y3,dvs2z3,dvs2x4,dvs2y4,dvs2z4,vs1,bnd2,ang2,dih2,&
           v1,v3,v4,dv0dx1,dv0dy1,dv0dz1,dv0dx2,dv0dy2,dv0dz2,&
           dv0dx3,dv0dy3,dv0dz3)

  use vector
  use stream
  use consta
  use number
  use contrl
  use psf

     real(chm_real),intent(in)  :: bnd2,ang2,dih2,vs1(3),v1(3),v3(3),v4(3),&
     dv0dx1(3),dv0dy1(3),dv0dz1(3),dv0dx2(3),dv0dy2(3),dv0dz2(3),&
     dv0dx3(3),dv0dy3(3),dv0dz3(3)

     real(chm_real),intent(out) :: vs2(3),dvs2x1(3),dvs2y1(3),dvs2z1(3),&
     dvs2x2(3),dvs2y2(3),dvs2z2(3),dvs2x3(3),dvs2y3(3),dvs2z3(3),&
     dvs2x4(3),dvs2y4(3),dvs2z4(3)

!    local vars
     real(chm_real),dimension(3) :: t0x1,t0y1,t0z1,t0x2,t0y2,t0z2,t0x3,&
          t0y3,t0z3,t0x4,t0y4,t0z4  !!dummy variables for buildatom
!    local copy of virtual atom derivatives     
     real(chm_real),dimension(3) :: d0x1,d0y1,d0z1,d0x2,d0y2,d0z2,d0x3,&
          d0y3,d0z3,d0x4,d0y4,d0z4  !!dummy variables for buildatom

     real(chm_real),dimension(3) :: nu,nv,nw,u,r34,dfwdx3,dfwdy3,dfwdz3,&
        dfwdx4,dfwdy4,dfwdz4,tux1,tuy1,tuz1,tux2,tuy2,tuz2,tux3,tuy3,&
        tuz3,tux4,tuy4,tuz4,dfudx1,dfudy1,dfudz1,dfudx2,dfudy2,dfudz2,&
        dfudx3,dfudy3,dfudz3,dfudx4,dfudy4,dfudz4,dfvdx1,dfvdy1,dfvdz1,&
        dfvdx2,dfvdy2,dfvdz2,dfvdx3,dfvdy3,dfvdz3,dfvdx4,dfvdy4,dfvdz4,&
        dpdx,dpdy,dpdz,dmdx,dmdy,dmdz,r1,r2,tr1,tr2,rtcb,vv3

     real(chm_real) :: d34,du,t1,t2,t3,tx,ty,tz,alpha,alphad

!     
!  q1: catm1  q2: catm2   q3: catm3  q4: catm4
!  vs1:  q1,q2,q3  
!  vs2: vs1,q3,q4
!

     d0x1=dv0dx1
     d0y1=dv0dy1
     d0z1=dv0dz1
     d0x2=dv0dx2
     d0y2=dv0dy2
     d0z2=dv0dz2
     d0x3=dv0dx3
     d0y3=dv0dy3
     d0z3=dv0dz3
     d0x4=ZERO
     d0y4=ZERO
     d0z4=ZERO

     dvs2x1=ZERO
     dvs2y1=ZERO
     dvs2z1=ZERO
     dvs2x2=ZERO
     dvs2y2=ZERO
     dvs2z2=ZERO
     dvs2x3=ZERO
     dvs2y3=ZERO
     dvs2z3=ZERO
     dvs2x4=ZERO
     dvs2y4=ZERO
     dvs2z4=ZERO

     dpdx=(/ ONE,   ZERO,  ZERO/)
     dpdy=(/ ZERO,  ONE,   ZERO/)
     dpdz=(/ ZERO,   ZERO,  ONE/)

     dmdx=(/ MINONE,   ZERO,     ZERO/)
     dmdy=(/ ZERO,     MINONE,   ZERO/)
     dmdz=(/ ZERO,     ZERO,     MINONE/)

!
!              <<vs2>>
!                 |    
!              b2 |
!                [3]
!         t2,q2 /    \
!              /      \
!           [4]     <<vs1>> 
!                  ([1],[2],[3])
!
!

!>>2012
     call getcosangl2(v4,v3,vs1,alpha)
     alphad=RADDEG*acos(alpha)
     if ( alphad .le. 165 ) then
!>>2012
        call buildatom(bnd2,ang2,dih2,v3,v4,vs1,vs2,t0x1,t0y1,t0z1,t0x2,&
                    t0y2,t0z2,t0x3,t0y3,t0z3)
!>>2012
    else
       vv3=v3
       d0x1=dv0dx1
       d0y1=dv0dy1
       d0z1=dv0dz1
       d0x2=dv0dx2
       d0y2=dv0dy2
       d0z2=dv0dz2
       d0x3=dv0dx3
       d0y3=dv0dy3
       d0z3=dv0dz3
       d0x4=ZERO
       d0y4=ZERO
       d0z4=ZERO

       if(PRNLEV .gt. 5) then
          write(*,'(a6,4f8.3,a6)',advance='no')'PA> ',vv3,alphad,' => ' 
       endif

       call pushatom(vv3,vs1,v1,v3,v4,d0x1,d0y1,d0z1,d0x2,d0y2,&
                    d0z2,d0x3,d0y3,d0z3,d0x4,d0y4,d0z4)
       call getcosangl2(v4,vv3,vs1,alpha)
       alphad=RADDEG*acos(alpha)

       if(PRNLEV .gt. 5) then
           write(*,'(4f8.3)',advance='yes')vv3,alphad
       endif

       call buildatom(bnd2,ang2,dih2,vv3,v4,vs1,vs2,t0x1,t0y1,t0z1,t0x2,t0y2,&
                      t0z2,t0x3,t0y3,t0z3)
    endif
!>>2012       

     r34=v4-v3  
     d34=sqrt(r34(1)**2+r34(2)**2+r34(3)**2)
     nw=r34/d34
     call CROSS3(vs1-v3,nw,u)
     du=sqrt(u(1)**2+u(2)**2+u(3)**2)
     nu=u/du
     call CROSS3(nw,nu,nv)

!      get derivatives ...
!--------------------------------------------------------------------------
!      First W wrt q3,q4

       t1=(1/d34)-((r34(1)**2)/(d34**3))
       t2=-(r34(1)*r34(2))/(d34**3)
       t3=-(r34(1)*r34(3))/(d34**3)
       dfwdx4=(/t1, t2, t3 /)
       t1=-(r34(2)*r34(1))/(d34**3)
       t2=(1/d34)-((r34(2)**2)/(d34**3))       
       t3=-(r34(2)*r34(3))/(d34**3)
       dfwdy4=(/t1, t2, t3 /)
       t1=-(r34(3)*r34(1))/(d34**3)
       t2=-(r34(3)*r34(2))/(d34**3)
       t3=(1/d34)-((r34(3)**2)/(d34**3))
       dfwdz4=(/t1, t2, t3 /)
      
       t1=(-1/d34)+((r34(1)**2)/(d34**3))
       t2=(r34(1)*r34(2))/(d34**3)
       t3=(r34(1)*r34(3))/(d34**3)
       dfwdx3=(/t1, t2, t3 /)
       t1=(r34(2)*r34(1))/(d34**3)
       t2=(-1/d34)+((r34(2)**2)/(d34**3))       
       t3=(r34(2)*r34(3))/(d34**3)
       dfwdy3=(/t1, t2, t3 /)
       t1=(r34(3)*r34(1))/(d34**3)
       t2=(r34(3)*r34(2))/(d34**3)
       t3=(-1/d34)+((r34(3)**2)/(d34**3))
       dfwdz3=(/t1, t2, t3 /)

!--------------  derivatives of UVEC ---------------
! 
! u = (vs1-v3) X nw
!            k=1     D(vs1-v3) X nw              
!  du        k=2     D(vs1-v3) X nw
!  ---   =   k=3     (vs1-v3)  X Dnw + D(vs1-v3) X nw
!  dqk       k=4     (vs1-v3)  X Dnw
!----------------------------------------------------

  
!    du/dq1
     call CROSS3(d0x1,nw,tux1)
     call CROSS3(d0y1,nw,tuy1)
     call CROSS3(d0z1,nw,tuz1)

!    du/dq2        
     call CROSS3(d0x2,nw,tux2)
     call CROSS3(d0y2,nw,tuy2)
     call CROSS3(d0z2,nw,tuz2)

!    du/dq3
  
     r1=vs1-v3
     call CROSS3(r1,dfwdx3,tr1)
     r2=d0x3-dpdx
     call CROSS3(r2,nw,tr2)
     tux3=tr1+tr2

     r1=vs1-v3
     call CROSS3(r1,dfwdy3,tr1)
     r2=d0y3-dpdy
     call CROSS3(r2,nw,tr2)
     tuy3=tr1+tr2

     r1=vs1-v3
     call CROSS3(r1,dfwdz3,tr1)
     r2=d0z3-dpdz
     call CROSS3(r2,nw,tr2)
     tuz3=tr1+tr2

!     du/dq4
      r1=vs1-v3
      call CROSS3(r1,dfwdx4,tux4)
      call CROSS3(r1,dfwdy4,tuy4)
      call CROSS3(r1,dfwdz4,tuz4)

!
! d|u|/dqk = (u.du/dqk)/|u|
! dnu/dqk  = { |u|.du/dqk - u*d|u|/dqk }/|u|**2
!
!    dnu/dq1
     call nvecdrv(u,tux1,dfudx1)
     call nvecdrv(u,tuy1,dfudy1)
     call nvecdrv(u,tuz1,dfudz1)

!    dnu/dq2
     call nvecdrv(u,tux2,dfudx2)
     call nvecdrv(u,tuy2,dfudy2)
     call nvecdrv(u,tuz2,dfudz2)

!    dnu/dq3
     call nvecdrv(u,tux3,dfudx3)
     call nvecdrv(u,tuy3,dfudy3)
     call nvecdrv(u,tuz3,dfudz3)

!    dnu/dq4
     call nvecdrv(u,tux4,dfudx4)
     call nvecdrv(u,tuy4,dfudy4)
     call nvecdrv(u,tuz4,dfudz4)

!
!             k=1   nw X dnu/dq1
! dnv/dqk =   k=2   nw X dnu/dq2
!             k=3   nw X dnu/dq3 + dnw/dq3 X nu
!             k=4   nw X dnu/dq4 + dnw/dq4 X nu
!

!        dnw/dq1
      call CROSS3(nw,dfudx1,dfvdx1)
      call CROSS3(nw,dfudy1,dfvdy1)
      call CROSS3(nw,dfudz1,dfvdz1)

!        dnw/dq2
      call CROSS3(nw,dfudx2,dfvdx2)
      call CROSS3(nw,dfudy2,dfvdy2)
      call CROSS3(nw,dfudz2,dfvdz2)

!        dnw/dq3
      call CROSS3(nw,dfudx3,r1)
      call CROSS3(dfwdx3,nu,r2)
      dfvdx3=r1+r2
      call CROSS3(nw,dfudy3,r1)
      call CROSS3(dfwdy3,nu,r2)
      dfvdy3=r1+r2
      call CROSS3(nw,dfudz3,r1)
      call CROSS3(dfwdz3,nu,r2)
      dfvdz3=r1+r2

!        dnw/dq4
      call CROSS3(nw,dfudx4,r1)
      call CROSS3(dfwdx4,nu,r2)
      dfvdx4=r1+r2
      call CROSS3(nw,dfudy4,r1)
      call CROSS3(dfwdy4,nu,r2)
      dfvdy4=r1+r2
      call CROSS3(nw,dfudz4,r1)
      call CROSS3(dfwdz4,nu,r2)
      dfvdz4=r1+r2
    
!
!  Derivatives of vs2
!       ---                                    --- 
!       | nux*rtcb(1)+nvx*rtcb(2)+nwx*rtcb(3)+x3 |  
!  rcb =| nuy*rtcb(1)+nvy*rtcb(2)+nwy*rtcb(3)+y3 |
!       | nuz*rtcb(1)+nvz*rtcb(2)+nwz*rtcb(3)+z3 |
!       ---                                    ---
!

     rtcb(1)=bnd2*sin(DEGRAD*ang2)*sin(DEGRAD*dih2)
     rtcb(2)=bnd2*sin(DEGRAD*ang2)*cos(DEGRAD*dih2)
     rtcb(3)=bnd2*cos(DEGRAD*ang2)

!    Dvs2/Dq1
     tx=rtcb(1)*dfudx1(1)+rtcb(2)*dfvdx1(1)
     ty=rtcb(1)*dfudx1(2)+rtcb(2)*dfvdx1(2)
     tz=rtcb(1)*dfudx1(3)+rtcb(2)*dfvdx1(3)
     dvs2x1=(/tx, ty, tz/)

     tx=rtcb(1)*dfudy1(1)+rtcb(2)*dfvdy1(1)
     ty=rtcb(1)*dfudy1(2)+rtcb(2)*dfvdy1(2)
     tz=rtcb(1)*dfudy1(3)+rtcb(2)*dfvdy1(3)
     dvs2y1=(/tx, ty, tz/)

     tx=rtcb(1)*dfudz1(1)+rtcb(2)*dfvdz1(1)
     ty=rtcb(1)*dfudz1(2)+rtcb(2)*dfvdz1(2)
     tz=rtcb(1)*dfudz1(3)+rtcb(2)*dfvdz1(3)
     dvs2z1=(/tx, ty, tz/)

!    Dvs2/Dq2
     tx=rtcb(1)*dfudx2(1)+rtcb(2)*dfvdx2(1)
     ty=rtcb(1)*dfudx2(2)+rtcb(2)*dfvdx2(2)
     tz=rtcb(1)*dfudx2(3)+rtcb(2)*dfvdx2(3)
     dvs2x2=(/tx, ty, tz/)

     tx=rtcb(1)*dfudy2(1)+rtcb(2)*dfvdy2(1)
     ty=rtcb(1)*dfudy2(2)+rtcb(2)*dfvdy2(2)
     tz=rtcb(1)*dfudy2(3)+rtcb(2)*dfvdy2(3)
     dvs2y2=(/tx, ty, tz/)

     tx=rtcb(1)*dfudz2(1)+rtcb(2)*dfvdz2(1)
     ty=rtcb(1)*dfudz2(2)+rtcb(2)*dfvdz2(2)
     tz=rtcb(1)*dfudz2(3)+rtcb(2)*dfvdz2(3)
     dvs2z2=(/tx, ty, tz/)

!    Dvs2/Dq3
     tx=1+rtcb(1)*dfudx3(1)+rtcb(2)*dfvdx3(1)+rtcb(3)*dfwdx3(1)
     ty=  rtcb(1)*dfudx3(2)+rtcb(2)*dfvdx3(2)+rtcb(3)*dfwdx3(2)
     tz=  rtcb(1)*dfudx3(3)+rtcb(2)*dfvdx3(3)+rtcb(3)*dfwdx3(3)
     dvs2x3=(/tx, ty, tz/)

     tx=  rtcb(1)*dfudy3(1)+rtcb(2)*dfvdy3(1)+rtcb(3)*dfwdy3(1)
     ty=1+rtcb(1)*dfudy3(2)+rtcb(2)*dfvdy3(2)+rtcb(3)*dfwdy3(2)
     tz=  rtcb(1)*dfudy3(3)+rtcb(2)*dfvdy3(3)+rtcb(3)*dfwdy3(3)
     dvs2y3=(/tx, ty, tz/)

     tx=  rtcb(1)*dfudz3(1)+rtcb(2)*dfvdz3(1)+rtcb(3)*dfwdz3(1)
     ty=  rtcb(1)*dfudz3(2)+rtcb(2)*dfvdz3(2)+rtcb(3)*dfwdz3(2)
     tz=1+rtcb(1)*dfudz3(3)+rtcb(2)*dfvdz3(3)+rtcb(3)*dfwdz3(3)
     dvs2z3=(/tx, ty, tz/)

!    Dvs2/Dq4
     tx=rtcb(1)*dfudx4(1)+rtcb(2)*dfvdx4(1)+rtcb(3)*dfwdx4(1)
     ty=rtcb(1)*dfudx4(2)+rtcb(2)*dfvdx4(2)+rtcb(3)*dfwdx4(2)
     tz=rtcb(1)*dfudx4(3)+rtcb(2)*dfvdx4(3)+rtcb(3)*dfwdx4(3)
     dvs2x4=(/tx, ty, tz/)

     tx=rtcb(1)*dfudy4(1)+rtcb(2)*dfvdy4(1)+rtcb(3)*dfwdy4(1)
     ty=rtcb(1)*dfudy4(2)+rtcb(2)*dfvdy4(2)+rtcb(3)*dfwdy4(2)
     tz=rtcb(1)*dfudy4(3)+rtcb(2)*dfvdy4(3)+rtcb(3)*dfwdy4(3)
     dvs2y4=(/tx, ty, tz/)

     tx=rtcb(1)*dfudz4(1)+rtcb(2)*dfvdz4(1)+rtcb(3)*dfwdz4(1)
     ty=rtcb(1)*dfudz4(2)+rtcb(2)*dfvdz4(2)+rtcb(3)*dfwdz4(2)
     tz=rtcb(1)*dfudz4(3)+rtcb(2)*dfvdz4(3)+rtcb(3)*dfwdz4(3)
     dvs2z4=(/tx, ty, tz/)

     end subroutine buildvs2

     subroutine pushatom(vv3,vs1,v1,v3,v4,tv0tx1,tv0ty1,tv0tz1,tv0tx2,tv0ty2,&
                         tv0tz2,tv0tx3,tv0ty3,tv0tz3,tv0tx4,tv0ty4,tv0tz4)
  use vector
  use stream
  use consta
  use number
  use contrl
  use psf

     real(chm_real),dimension(3),intent(in)  :: vs1(3),v1(3),v3(3),v4(3)
     real(chm_real),dimension(3),intent(inout) :: tv0tx1,tv0ty1,tv0tz1,tv0tx2,&
           tv0ty2,tv0tz2,tv0tx3,tv0ty3,tv0tz3,tv0tx4,tv0ty4,tv0tz4
!   local vars
     real(chm_real) :: alphad,alpha,delta,dvs11,d3s1,tx,ty,tz,d34,&
                       dnum,dden,dfac,beta,lambda
     real(chm_real),dimension(3) :: vv3,vs11,n34,n3s1,nvs11,r3s1,&
     tvs11qk,tnvs11qk,trijqk,tnijqk,dpdx,dpdy,dpdz,dmdx,dmdy,dmdz,r34,&
     dfr34dx3,dfr34dy3,dfr34dz3,dfr34dx4,dfr34dy4,dfr34dz4,&
     dfn34dx3,dfn34dy3,dfn34dz3,dfn34dx4,dfn34dy4,dfn34dz4,&
     dfr3s1dx1,dfr3s1dy1,dfr3s1dz1,dfr3s1dx2,dfr3s1dy2,dfr3s1dz2,dfr3s1dx3,&
     dfr3s1dy3,dfr3s1dz3,dfn3s1dx1,dfn3s1dy1,dfn3s1dz1,dfn3s1dx2,dfn3s1dy2,&
     dfn3s1dz2,dfn3s1dx3,dfn3s1dy3,dfn3s1dz3,tva,tvb,tvc
     real(chm_real) :: Tdeltatq(12),DnvecDq(36)
     integer :: kk
!     
!  q1: catm1  q2: catm2   q3: catm3  q4: catm4
!  vs1:  q1,q2,q3  
!
     dpdx=(/ ONE,   ZERO,  ZERO/)
     dpdy=(/ ZERO,  ONE,   ZERO/)
     dpdz=(/ ZERO,   ZERO,  ONE/)
     dmdx=(/ MINONE,   ZERO,     ZERO/)
     dmdy=(/ ZERO,     MINONE,   ZERO/)
     dmdz=(/ ZERO,     ZERO,     MINONE/)
     Tdeltatq=ZERO  !DdeltaD(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4)
     DnvecDq=ZERO   !Dnvs11D(x1(3),y1(3),z1(3),x2(3),y2(3),z2(3),
                    !        x3(3),y3(3),z3(3),x4(3),y4(3),z4(3))
     call getcosangl2(v4,v3,vs1,alpha)
     alphad=RADDEG*acos(alpha)
     vv3=v3
     vs11=v1-vs1
     dvs11=sqrt(vs11(1)**2+vs11(2)**2+vs11(3)**2)
     nvs11=vs11/dvs11
     dnum=-0.125
     beta=300.0
     lambda=294.0
     dden=(1+exp(beta*alpha+lambda))
     delta=dnum/dden
     vv3=vv3+delta*nvs11   !v3' = v3 + delta*nvs11
!
!                         ________________________________
!       dvs11    dv1     |  dvs1       q=x,y,z  k=1,2,3       
!       ---   =  ---   - |  ----     these derivatives are        
!       dqk      dqk     |  dqk      from vs1 construction
!                        |      vs1=F(catm1,catm2,catm3)
!                         --------------------------------- 
!
!     dnvs11           dvs11           dds11
!     ------  = { ds11 -----  -   vs11 ----- }/ds11^2 
!     dqk              dqk             dqk
!
      tvs11qk=ZERO
      tnvs11qk=ZERO
!       q1
      tvs11qk=dpdx-tv0tx1
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(1)=tnvs11qk(1)
      DnvecDq(2)=tnvs11qk(2)
      DnvecDq(3)=tnvs11qk(3)
      tvs11qk=dpdy-tv0ty1
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(4)=tnvs11qk(1)
      DnvecDq(5)=tnvs11qk(2)
      DnvecDq(6)=tnvs11qk(3)
      tvs11qk=dpdz-tv0tz1
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(7)=tnvs11qk(1)
      DnvecDq(8)=tnvs11qk(2)
      DnvecDq(9)=tnvs11qk(3)
!       q2
      tvs11qk=-tv0tx2
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(10)=tnvs11qk(1)
      DnvecDq(11)=tnvs11qk(2)
      DnvecDq(12)=tnvs11qk(3)
      tvs11qk=-tv0ty2
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(13)=tnvs11qk(1)
      DnvecDq(14)=tnvs11qk(2)
      DnvecDq(15)=tnvs11qk(3)
      tvs11qk=-tv0tz2
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(16)=tnvs11qk(1)
      DnvecDq(17)=tnvs11qk(2)
      DnvecDq(18)=tnvs11qk(3)
!     q3
      tvs11qk=-tv0tx3
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(19)=tnvs11qk(1)
      DnvecDq(20)=tnvs11qk(2)
      DnvecDq(21)=tnvs11qk(3)
      tvs11qk=-tv0ty3
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(22)=tnvs11qk(1)
      DnvecDq(23)=tnvs11qk(2)
      DnvecDq(24)=tnvs11qk(3)
      tvs11qk=-tv0tz3
      call nvecdrv(vs11,tvs11qk,tnvs11qk)
      DnvecDq(25)=tnvs11qk(1)
      DnvecDq(26)=tnvs11qk(2)
      DnvecDq(27)=tnvs11qk(3)
!     q4
      DnvecDq(28)=ZERO
      DnvecDq(29)=ZERO
      DnvecDq(30)=ZERO
      DnvecDq(31)=ZERO
      DnvecDq(32)=ZERO
      DnvecDq(33)=ZERO
      DnvecDq(34)=ZERO
      DnvecDq(35)=ZERO
      DnvecDq(36)=ZERO
!
!      Dalpha          Dn3s1    Dn34
!      ------  =  n34. -----  + ---- .n3s1
!      Dqk             Dqk      Dqk
!
      r34=v4-v3
      d34=sqrt(r34(1)**2+r34(2)**2+r34(3)**2)
      r3s1=vs1-v3
      d3s1=sqrt(r3s1(1)**2+r3s1(2)**2+r3s1(3)**2)
      n34=(/    r34(1)/d34,    r34(2)/d34,   r34(3)/d34  /)
      n3s1=(/ r3s1(1)/d3s1,  r3s1(2)/d3s1, r3s1(3)/d3s1  /)
      call DOTPR(n34,n3s1,3,tx)
!
!     Dn34             0  k=1
!    -----.n3s1  ==    0  k=2
!     Dqk              
!
!       q=1
      Tdeltatq(1)=ZERO
      Tdeltatq(2)=ZERO
      Tdeltatq(3)=ZERO
!       q=2
      Tdeltatq(4)=ZERO
      Tdeltatq(5)=ZERO
      Tdeltatq(6)=ZERO
!       q=3
      trijqk=dmdx
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,tx)
      Tdeltatq(7)=tx
      trijqk=dmdy
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,ty)
      Tdeltatq(8)=ty
      trijqk=dmdz
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,tz)
      Tdeltatq(9)=tz
!     q=4
      trijqk=dpdx
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,tx)
      Tdeltatq(10)=tx
      trijqk=dpdy
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,ty)
      Tdeltatq(11)=ty
      trijqk=dpdz
      call nvecdrv(r34,trijqk,tnijqk)
      call DOTPR(tnijqk,n3s1,3,tz)
      Tdeltatq(12)=tz

!
!           Dn3s1         0  k=4
!      n34. -----    =     
!           Dqk    
!

!       q=1
      trijqk=tv0tx1
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tx)
      Tdeltatq(1)=Tdeltatq(1)+tx
      trijqk=tv0ty1
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,ty)
      Tdeltatq(2)=Tdeltatq(2)+ty
      trijqk=tv0tz1
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tz)
      Tdeltatq(3)=Tdeltatq(3)+tz
!     q=2
      trijqk=tv0tx2
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tx)
      Tdeltatq(4)=Tdeltatq(4)+tx
      trijqk=tv0ty2
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,ty)
      Tdeltatq(5)=Tdeltatq(5)+ty
      trijqk=tv0tz2
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tz)
      Tdeltatq(6)=Tdeltatq(6)+tz
!     q=3
      trijqk=tv0tx3+dmdx
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tx)
      Tdeltatq(4)=Tdeltatq(4)+tx
      trijqk=tv0ty3+dmdy
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,ty)
      Tdeltatq(5)=Tdeltatq(5)+ty
      trijqk=tv0tz3+dmdz
      call nvecdrv(r3s1,trijqk,tnijqk)
      call DOTPR(tnijqk,n34,3,tz)
      Tdeltatq(6)=Tdeltatq(6)+tz

!
!       Update all the derivatives of q1,q2,q3 before vs2 is constructed       
!       with new v3
!
!       Dvv3       Dv3              Dnvs11      Ddelta
!       ----   =   ---   +  delta * ------   +  ------ * nvs11
!       Dqk        Dqk              Dqk         Dqk
! 
!
!      
!      Ddelta                   (dden-1)     Dalpha 
!      -------  = { - dnum*beta* --------} * -----
!      Dqk                       dden^2      Dqk
!

     dfac=-(dnum*beta*(dden-1))/(dden**2)
!    q1 (x)
     tva=ZERO
     tvb=(/ DnvecDq(1), DnvecDq(2), DnvecDq(3) /)
     tvb=delta*tvb
     tvc=Tdeltatq(1)*dfac*nvs11
     tv0tx1=tv0tx1+tva+tvb+tvc     
!    q1 (y)
     tva=ZERO
     tvb=(/ DnvecDq(4), DnvecDq(5), DnvecDq(6) /)
     tvb=delta*tvb
     tvc=Tdeltatq(2)*dfac*nvs11
     tv0ty1=tv0ty1+tva+tvb+tvc     
!    q1 (z)
     tva=ZERO
     tvb=(/ DnvecDq(7), DnvecDq(8), DnvecDq(9) /)
     tvb=delta*tvb
     tvc=Tdeltatq(3)*dfac*nvs11
     tv0tz1=tv0tz1+tva+tvb+tvc
!    q2 (x)
     tva=ZERO
     tvb=(/ DnvecDq(10), DnvecDq(11), DnvecDq(12) /)
     tvb=delta*tvb
     tvc=Tdeltatq(4)*dfac*nvs11
     tv0tx2=tv0tx2+tva+tvb+tvc     
!    q2 (y)
     tva=ZERO
     tvb=(/ DnvecDq(13), DnvecDq(14), DnvecDq(15) /)
     tvb=delta*tvb
     tvc=Tdeltatq(5)*dfac*nvs11
     tv0ty2=tv0ty2+tva+tvb+tvc     
!    q2 (z)
     tva=ZERO
     tvb=(/ DnvecDq(16), DnvecDq(17), DnvecDq(18) /)
     tvb=delta*tvb
     tvc=Tdeltatq(6)*dfac*nvs11
     tv0tz2=tv0tz2+tva+tvb+tvc
!    q3 (x)
     tva=ZERO
     tvb=(/ DnvecDq(19), DnvecDq(20), DnvecDq(21) /)
     tvb=delta*tvb
     tvc=Tdeltatq(7)*dfac*nvs11
     tv0tx3=tv0tx3+tva+tvb+tvc     
!    q3 (y)
     tva=ZERO
     tvb=(/ DnvecDq(22), DnvecDq(23), DnvecDq(24) /)
     tvb=delta*tvb
     tvc=Tdeltatq(8)*dfac*nvs11
     tv0ty3=tv0ty3+tva+tvb+tvc     
!    q3 (z)
     tva=ZERO
     tvb=(/ DnvecDq(25), DnvecDq(26), DnvecDq(27) /)
     tvb=delta*tvb
     tvc=Tdeltatq(9)*dfac*nvs11
     tv0tz3=tv0tz3+tva+tvb+tvc
!    q4 (x)
     tva=ZERO
     tvb=(/ DnvecDq(28), DnvecDq(29), DnvecDq(30) /)
     tvb=delta*tvb
     tvc=Tdeltatq(10)*dfac*nvs11
     tv0tx4=tv0tx4+tva+tvb+tvc
!    q4 (y)
     tva=ZERO
     tvb=(/ DnvecDq(31), DnvecDq(32), DnvecDq(33) /)
     tvb=delta*tvb
     tvc=Tdeltatq(11)*dfac*nvs11
     tv0ty4=tv0ty4+tva+tvb+tvc
!    q4 (z)
     tva=ZERO
     tvb=(/ DnvecDq(34), DnvecDq(35), DnvecDq(36) /)
     tvb=delta*tvb
     tvc=Tdeltatq(12)*dfac*nvs11
     tv0tz4=tv0tz4+tva+tvb+tvc
     end subroutine pushatom

     subroutine buildvs3(vs3,dvs3x1,dvs3y1,dvs3z1,dvs3x2,dvs3y2,dvs3z2,&
           dvs3x3,dvs3y3,dvs3z3,dvs3x4,dvs3y4,dvs3z4,dvs3x5,dvs3y5,&
           dvs3z5,vs2,v5,dv0dx1,dv0dy1,dv0dz1,dv0dx2,dv0dy2,dv0dz2,&
           dv0dx3,dv0dy3,dv0dz3,dv0dx4,dv0dy4,dv0dz4)
  use exfunc
  use stream
  use consta
  use number
  use contrl
  use psf

     real(chm_real),intent(in)  :: vs2(3),v5(3),dv0dx1(3),dv0dy1(3),&
     dv0dz1(3),dv0dx2(3),dv0dy2(3),dv0dz2(3),dv0dx3(3),dv0dy3(3),&
     dv0dz3(3),dv0dx4(3),dv0dy4(3),dv0dz4(3)

     real(chm_real),intent(out) :: vs3(3),dvs3x1(3),dvs3y1(3),dvs3z1(3),&
     dvs3x2(3),dvs3y2(3),dvs3z2(3),dvs3x3(3),dvs3y3(3),dvs3z3(3),&
     dvs3x4(3),dvs3y4(3),dvs3z4(3),dvs3x5(3),dvs3y5(3),dvs3z5(3)

     vs3=2*v5-vs2
     dvs3x1=-dv0dx1
     dvs3y1=-dv0dy1
     dvs3z1=-dv0dz1

     dvs3x2=-dv0dx2
     dvs3y2=-dv0dy2
     dvs3z2=-dv0dz2

     dvs3x3=-dv0dx3
     dvs3y3=-dv0dy3
     dvs3z3=-dv0dz3

     dvs3x4=-dv0dx4
     dvs3y4=-dv0dy4
     dvs3z4=-dv0dz4

     dvs3x5=(/2.0, 0.0, 0.0/)
     dvs3y5=(/0.0, 2.0, 0.0/)
     dvs3z5=(/0.0, 0.0, 2.0/)

     end subroutine buildvs3


     subroutine nvecdrv(r,drdq,dndq)
!    returns derivative of a normal vector wrt generic q given
!    the vector and the derivative of vector(wrt q)
  use vector
  use number
     real(chm_real),dimension(3),intent(in)  :: r,drdq
     real(chm_real),dimension(3),intent(out) :: dndq
!    local vars
     real(chm_real) :: rd,tx,td

     rd=sqrt(r(1)**2+r(2)**2+r(3)**2)
     if (rd .le. 1e-06) call WRNDIE(-5,'primomodule:nvecdrv',&
           ' processed zero length vector ')
     call DOTPR(r,drdq,3,tx)
     td=tx/rd
     dndq=( (rd*drdq) - (r*td) )/ (rd**2)
    end subroutine nvecdrv

     subroutine primostats(st)
  use exfunc
  use stream
  use dimens_fcm
  use psf

     integer,dimension(12),intent(out) :: st
     integer :: i
     st=0
   
     do i=1,nres

        if(vatms(i)%vs1) then
           st(1)=st(1)+1
           if(vatms(i)%qdist1)  st(2)=st(2)+1
           if(vatms(i)%qangl1)  st(3)=st(3)+1
        endif

        if(vatms(i)%vs2) then
           st(4)=st(4)+1
           if(vatms(i)%qdist2)  st(5)=st(5)+1
           if(vatms(i)%qangl2)  st(6)=st(6)+1
        endif

        if(vatms(i)%vs3) then
           st(7)=st(7)+1
           if(vatms(i)%qdist3)  st(8)=st(8)+1
           if(vatms(i)%qangl3)  st(9)=st(9)+1
        endif

        if(vatms(i)%vs0) then
           st(10)=st(10)+1
           if(vatms(i)%qdist0)  st(11)=st(11)+1
           if(vatms(i)%qangl0)  st(12)=st(12)+1
        endif


      enddo

     end subroutine primostats

     subroutine eprimo(ene,X,Y,Z,DX,DY,DZ)
  use exfunc
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
     real(chm_real),intent(out) :: ene
     real(chm_real),intent(in),dimension(natom) :: X,Y,Z
     real(chm_real),intent(inout),dimension(natom) :: DX,DY,DZ
    
     !local vars
     integer :: i,st(12),steps
     real(chm_real) :: ev

#if KEY_PARALLEL==1
      ecalcparall: if(MYNODP .eq. 1) then
         if(qprimo) then
#else /**/
         if(qprimo) then
#endif 

         if(.not. qmesg) then
            st=0
            call primostats(st)
            write(OUTU,'(/,a)')'PRIMO>  *** STATS ***'
            write(OUTU,'(a,i5)')'PRIMO> No. of residues: ',nres 

            write(OUTU,'(a,i5)')'PRIMO> No. of virtual sites VS1',st(1)
            write(OUTU,'(a,i5)')'PRIMO>          No. of DIST1 calc.',st(2)
            write(OUTU,'(a,i5)')'PRIMO>          No. of ANGL1 calc.',st(3)

            write(OUTU,'(a,i5)')'PRIMO> No. of virtual sites VS2',st(4)
            write(OUTU,'(a,i5)')'PRIMO>          No. of DIST2 calc.',st(5)
            write(OUTU,'(a,i5)')'PRIMO>          No. of ANGL2 calc.',st(6)

            write(OUTU,'(a,i5)')'PRIMO> No. of virtual sites VS3',st(7)
            write(OUTU,'(a,i5)')'PRIMO>          No. of DIST3 calc.',st(8)
            write(OUTU,'(a,i5)')'PRIMO>          No. of ANGL3 calc.',st(9)

            write(OUTU,'(a,i5)')'PRIMO> No. of virtual sites VS0',st(10)
            write(OUTU,'(a,i5)')'PRIMO>          No. of DIST3 calc.',st(11)
            write(OUTU,'(a,i5)')'PRIMO>          No. of ANGL3 calc.',st(12)

            
            qmesg=.true.
         endif

         if (DYNAMQ) then
            steps=mdstep
         else
            steps=0
         endif

         ene=ZERO
         do i=1,nres
              if(steps .eq. 0 .or. mod(steps,vatms(i)%cstep) .eq. 0) then
                  ev=ZERO
                  if(qvatm(i)) then
                      call calcprimoene(ev,i,X,Y,Z,DX,DY,DZ)
                      ene=ene+ev
                  endif
              endif
         enddo

      endif
#if KEY_PARALLEL==1
      endif ecalcparall
#endif 
     end subroutine eprimo

#else /* (primo_main)*/

   implicit none
   contains
   subroutine primo_set(COMLYN,COMLEN)
     use exfunc
     character(len=*),intent(inout) :: COMLYN
     integer,intent(inout) :: COMLEN
     call WRNDIE(-5,'primo:primo_set',"primo module not compiled")
   end subroutine primo_set

#endif /* (primo_main)*/
   end module primomodule

