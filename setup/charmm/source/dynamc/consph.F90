module consph

  ! Constant pH module
  ! Tim Miller, June/July, 2011; reimplementing earlier
  ! code by Satoru Itoh with improvements.

  use chm_kinds
  use chm_types
  implicit none

  ! we rather lamely assume that there aren't more than 20 atom
  ! types to be modified per residue
  type :: cph_desc
    character(len=4)                :: resname
    integer                         :: nmod,nstate ! nmod <= 20, nstate <= 3
    real(chm_real), dimension(2)    :: del_ph,del_dg
    character(len=4), dimension(20) :: atnames
    real(chm_real), dimension(20,3) :: chrgs
    real(chm_real), dimension(20,3) :: vdws  ! for future expansion
  end type

  integer, parameter                       :: maxtitrtyp = 20
  integer, save                            :: ntres,ntitr ! number of titratable residue types 
                                                          ! and selected residues
  integer, allocatable, save, dimension(:)       :: rslct, tstate ! map of which residues are selected and
                                                                  ! their titration states (1, 2, or 3)
  type(cph_desc), dimension(maxtitrtyp), save    :: cph_params ! allow up to 20 typres of titratable residues
  logical, save                                  :: prm_parsed,qstartrsvr
  integer, save                                  :: recptr,phrsvrunum

contains

  subroutine consph_init
    ntres=0
    ntitr=0
    phrsvrunum=-1
    prm_parsed=.false.
    qstartrsvr=.false.
  end subroutine consph_init

  subroutine consph_cleanup
    use psf
    use memory

    if(allocated(rslct)) call chmdealloc('consph.src','consph_cleanup','rslct',nres,intg=rslct)
    if(allocated(tstate)) call chmdealloc('consph.src','consph_cleanup','tstate',nres,intg=tstate)
  end subroutine consph_cleanup

  subroutine phwrirsvr(x,y,z &
#if KEY_CHEQ==1
      ,cg &
#endif          
    )
    use psf,only:natom,nres
    use memory
    use stream

    real(chm_real),intent(in)        :: x(natom),y(natom),z(natom)
#if KEY_CHEQ==1
    real(chm_real),intent(in)        :: cg(natom)
#endif          
    integer,allocatable,dimension(:) :: tmptstate
    integer                          :: i

    if(phrsvrunum.le.0) return

    if(.not.qstartrsvr) then
       recptr=1
       qstartrsvr=.true.
    endif
    call chmalloc('consph.src','phwrirsvr','tmptstate',natom,intg=tmptstate)
    do i=1,nres
       tmptstate(i)=tstate(i)
    enddo
    do i=nres+1,natom
       tmptstate(i)=-1 ! make sure these are invalid
    enddo

    write(phrsvrunum,rec=recptr)   (sngl(x(i)),i=1,natom)
    write(phrsvrunum,rec=recptr+1) (sngl(y(i)),i=1,natom)
    write(phrsvrunum,rec=recptr+2) (sngl(z(i)),i=1,natom)
#if KEY_CHEQ==1
    write(phrsvrunum,rec=recptr+3) (sngl(cg(i)),i=1,natom)
#endif          
    write(phrsvrunum,rec=recptr+4) tmptstate
    recptr=recptr+5
    call chmdealloc('consph.src','phwrirsvr','tmptstate',natom,intg=tmptstate)
  end subroutine phwrirsvr

  subroutine phwrirstrt(unum)
    use psf
    use stream
    use ctitla

    integer, intent(in) :: unum
    integer             :: i,tindx

    if(iolev.gt.0) then
       write(titleb,'(A)') '* Constant pH restart file'
       call wrtitl(titleb,ntitlb,unum,0)
       tindx=0

       do i=1,nres
          if(rslct(i).ne.0) tindx=tindx+1
          write(unum,'(I5,I5,I2,I2)') i,tindx,rslct(i),tstate(i)
       enddo
       write(unum,'(A)') 'END'
    endif
  end subroutine phwrirstrt

  subroutine phrdrstrt(unum)
    use psf
    use string
    use stream
    use ctitla
#if KEY_PARALLEL==1
    use parallel 
#endif

    integer, parameter    :: mxclen = 500
    integer, intent(in)   :: unum
    integer               :: resnum,tindx,select,state,ios,ntitle
    character(len=mxclen) :: inpln,errstr,mytitle
    character(len=4)      :: resname, segname
    logical               :: error

#if KEY_PARALLEL==1
    if(mynod.eq.0) then
#endif
       if(iolev.gt.0) then
          ntitle=0
          call rdtitl(mytitle,ntitle,unum,0)
          do
             read(unum,'(A)',end=100) inpln
             if(len(inpln).le.0) cycle
             if(eqsta(inpln,1,'*').or.eqsta(inpln,1,'#')) cycle
             if(inpln=='END'.or.inpln=='end') exit

             read(inpln,'(I5,I5,I2,I2)',iostat=ios) resnum,tindx,select,state
             if(ios /= 0) then
                call wrndie(-4,'<PHRDRSTRT>','MALFORMED LINE IN PH RESTART FILE')
                cycle
             endif
             if(resnum.lt.1.or.resnum.gt.nres) then
                call wrndie(-4,'<PHRDRSTRT>','BAD RESIDUE NUMBER IN PH RESTART FILE')
                cycle
             endif
             if(select.ne.rslct(resnum)) then
                call wrndie(-4,'<PHRDRSTRT>','MISMATCH IN RESIDUE SELECTION WHEN READING CONS PH RESTART')
                cycle
             endif

             if(select /= 0) then
                ! wow, this residue is titratable, I'm honored...
                if(state.lt.1.or.state.gt.3) then
                   write(errstr,'(A,I5)') 'ERROR IN PH RESTART FILE: BAD STATE FOR RESIDUE ', resnum
                   call wrndie(-4,'<PHRDRSTRT>',errstr)
                   cycle
                endif
                call change_titration_state(resnum,state,resname,segname)
                tstate(resnum)=state
                if(prnlev.ge.5) then
                   write(outu,'(A,I5,A,I5,A,I3)') 'PHRDRSTRT> SET TITRATION STATE OF RESIDUE ', resnum, &
                                                  ' (TINDX = ', tindx, ') TO ', tstate(resnum)
                   call flush(outu)
                endif
             endif
          enddo
100       call vclose(unum,'KEEP',error)
       endif
#if KEY_PARALLEL==1
    endif 
#endif

#if KEY_PARALLEL==1
        ! broadcast charges
        !!call psnd4(tstate,nres)
        call psnd8(cg,natom)
#endif
  end subroutine phrdrstrt

  subroutine parsephprm(unum)
    use stream
    use string
    use ctitla
    use parallel
    implicit none

    integer, parameter    :: mxclen = 500
    integer, intent(in)   :: unum
    integer               :: clen,pidx,aidx,i,j
    character(len=mxclen) :: title,inpln
    character(len=4)      :: curres,curat,wrd
    character(len=200)    :: scratch
    logical               :: eof

    if(iolev.gt.0) then

       if(prm_parsed) &
          call wrndie(-4,'<PARSEPHPRM>','CONST PH PARAMS ALREADY READ')

       pidx=-1
       call rdtitl(titleb,ntitlb,unum,0)

       eof=.false.
       do
         call rdcmnd(inpln,mxclen,clen,unum,eof,.false.,.false.,'PARSEPHPRM> ')
         if(eof) exit
         if(clen.le.0) cycle

         wrd = nexta4(inpln,clen)
         if(wrd.eq.'RESI') then
            curres=nexta4(inpln,clen)

            pidx=-1
            do i=1,ntres
               if(cph_params(i)%resname == curres) then
                  call wrndie(-1,'<PARSEPHPRM>', 'DUPLICATE RESNAME, OVERWRITING OLD PARAMETERS')
                  pidx=i
                  exit
               endif
            enddo
            if(pidx.le.0) then
               ntres=ntres+1
               pidx=ntres
            endif

            ! fill in a default bogus set of parameters so we know if
            ! the user doesn't specify them.
            cph_params(pidx)%resname = curres
            cph_params(pidx)%nmod = 0 ! number of modified atoms this residue can have
            cph_params(pidx)%del_ph(1) = 0.0
            cph_params(pidx)%del_ph(2) = 0.0
            cph_params(pidx)%del_dg(1) = 0.0
            cph_params(pidx)%del_dg(2) = 0.0
            do i=1,20
               cph_params(pidx)%atnames(i)='NONE' ! what atoms get modified
               cph_params(pidx)%chrgs(i,1)=0.0    ! what are their various titration states
               cph_params(pidx)%chrgs(i,2)=0.0
               cph_params(pidx)%chrgs(i,3)=0.0
               ! to do vDW params
            enddo

            ! get the number of modifications
            cph_params(pidx)%nstate = nexti(inpln,clen) ! number of atoms that are modified
            if(cph_params(pidx)%nstate < 2 .or. cph_params(pidx)%nstate > 3) &
               call wrndie(-4,'<PARSEPHPRM>','Wrong number of different states for residue')

         else if(wrd.eq.'RFPK') then
            if(pidx.le.0) call wrndie(-4,'<PARSEPHRM>','RFPK NOT IN RESIDUE')

            cph_params(pidx)%del_ph(1)=nextf(inpln,clen)
            if(cph_params(pidx)%del_ph(1) < -13.9 .or. cph_params(pidx)%del_ph(1) > 13.9) &
               call wrndie(-4,'<PARSEPHPRM>','BAD pH VALUE GIVEN')

            if(cph_params(pidx)%nstate == 3) then
               cph_params(pidx)%del_ph(2)=nextf(inpln,clen)
               if(cph_params(pidx)%del_ph(2) < -13.9 .or. cph_params(pidx)%del_ph(2) > 13.9) &
                  call wrndie(-4,'<PARSEPHPRM>','BAD pH VALUE GIVEN')
            else
               cph_params(pidx)%del_ph(2)=cph_params(pidx)%del_ph(1)
            endif

         else if(wrd.eq.'RFDG') then
            if(pidx.le.0) call wrndie(-4,'<PARSEPHRM>','RFPH NOT IN RESIDUE')

            cph_params(pidx)%del_dg(1)=nextf(inpln,clen)

            if(cph_params(pidx)%nstate == 3) then
               cph_params(pidx)%del_dg(2)=nextf(inpln,clen)
            else
               cph_params(pidx)%del_dg(2)=cph_params(pidx)%del_dg(1)
            endif

         else if(wrd.eq.'ATOM') then
            if(pidx.le.0) call wrndie(-4,'<PARSEPHRM>','ATOM NOT IN RESIDUE')
            curat=nexta4(inpln,clen)

            ! check and make sure we don't already have a record for this atom
            aidx=-1
            do i=1,cph_params(pidx)%nmod
               if(cph_params(pidx)%atnames(i).eq.curat) then
                  aidx=i
                  write(outu,*) 'DUPLICATE PARAMS FOR ', curat, &
                                               ' IN RESNAME ', cph_params(pidx)% resname, &
                                               '. OLD PARAMS WILL BE OVERWRITTEN'
                  call wrndie(-1,'<PARSEPHRM>','DUPLICATE PARAMS')
                  exit
               endif
            enddo
            if(aidx.le.0) then
               cph_params(pidx)%nmod=cph_params(pidx)%nmod+1
               aidx=cph_params(pidx)%nmod
            endif

            ! now fill in the information
            cph_params(pidx)%atnames(aidx)=curat
            cph_params(pidx)%chrgs(aidx,1)=nextf(inpln,clen)
            cph_params(pidx)%chrgs(aidx,2)=nextf(inpln,clen)
            if(cph_params(pidx)%nstate == 3) then
               cph_params(pidx)%chrgs(aidx,3)=nextf(inpln,clen)
            else
               cph_params(pidx)%chrgs(aidx,3)=cph_params(pidx)%chrgs(aidx,2)
            endif
         else if(wrd.eq.'END') then
            exit
         else
            call wrndie(-2,'<PARSEPHRM>','UNKNOWN KEY-WORD!')
         endif
       enddo ! end of reading the file

       prm_parsed = .true.

       if(prnlev.ge.3) then
          ! let's print out the params for the user
          write(outu,'(a)') ' '
          write(outu,'(a)') 'PARSEPHPRM> PARAMETER DUMP: PLEASE REVIEW CAREFULLY'
          write(outu,'(a)') '---------------------------------------------------'
          write(outu,'(a)') ' '
    
          do i=1,ntres
             write(outu,'(a,i3,2a)') 'RESIDUE ', i, ' NAME ', cph_params(i)%resname
             write(outu,'(a,i2)') 'NUMBER OF DIFFERENT PROTONATION STATES IS ', cph_params(i)%nstate

             if(cph_params(i)%nstate == 2) then
                write(outu,'(a,f6.2)') 'pKa,w = ', cph_params(i)%del_ph(1)
                write(outu,'(a,f6.2)') 'delFele,w = ', cph_params(i)%del_dg(1)
                do j=1,cph_params(i)%nmod
                   write(outu,'(a,i3,3a,2f6.2)') &
                         'ATOM ', j, ' NAME ', cph_params(i)%atnames(j), ' CHARGES ', &
                         cph_params(i)%chrgs(j,1), cph_params(i)%chrgs(j,2)
                enddo
             else
                write(outu,'(a,2f6.2)') 'pKa,w = ', cph_params(i)%del_ph(1), cph_params(i)%del_ph(2)
                write(outu,'(a,2f6.2)') 'delFele,w = ', cph_params(i)%del_dg(1), cph_params(i)%del_dg(2)
                do j=1,cph_params(i)%nmod
                   write(outu,'(a,i3,3a,3f6.2)') &
                         'ATOM ', j, ' NAME ', cph_params(i)%atnames(j), ' CHARGES ', &
                         cph_params(i)%chrgs(j,1), cph_params(i)%chrgs(j,2), cph_params(i)%chrgs(j,3)
                enddo
             endif

             write(outu,'(a)') ' '
          enddo
       endif
    endif ! end iolev wrapper

#if KEY_PARALLEL==1
    ! broadcast the parameters accross all nodes
    call psnd4(ntres,1)
    do i=1,ntres
       call psndc(cph_params(i)%resname,4)
       call psnd4(cph_params(i)%nmod,1)
       call psnd4(cph_params(i)%nstate,1)
       call psnd8(cph_params(i)%del_ph,2)
       call psnd8(cph_params(i)%del_dg,2)
       do j=1,cph_params(i)%nmod
          call psndc(cph_params(i)%atnames(j),20)
          call psnd8(cph_params(i)%chrgs(j,1:3),3)
          call psnd8(cph_params(i)%vdws(j,1:3),3)
       enddo
    enddo
#endif

  end subroutine parsephprm

  subroutine getphresidues(comlyn,comlen)
     use memory
     use psf
     use coord
     use select, only: atmsel
     use stream
     use string
#if KEY_PARALLEL==1
     use parallel
#endif
     implicit none
     
     character(len=*)                   :: comlyn
     integer, intent(in)                :: comlen
     integer                            :: i,j,k,rnum,nsel
     integer, allocatable, dimension(:) :: islct
     character(len=4)                   :: wrd,resname
     character(len=60)                  :: errmsg

#if KEY_PARALLEL==1
     if(mynod.ne.0) return
#endif

     wrd=nexta4(comlyn,comlen)

     if(.not.allocated(rslct)) then
        call chmalloc('consph.src','getphresidues','rslct',nres,intg=rslct)
        do i=1,nres
           rslct(i) = 0
        enddo
     endif
     if(.not.allocated(tstate)) then
        call chmalloc('consph.src','getphresidues','tstate',nres,intg=tstate)
        do i=1,nres
           tstate(i) = 0
        enddo
     endif

     call chmalloc('consph.src','getphresidues','islct',natom,intg=islct)
     if(wrd.eq.'ADD') then
        nsel = atmsel(comlyn,comlen,islct,.false.)
        if(nsel > 0) then
           do i=1,natom
              if(islct(i).gt.0) then
                 call getresbyat(i,rnum)
                 if(rslct(rnum) /= 1) then
                    ntitr=ntitr+1
                    rslct(rnum)=1
                    tstate(rnum)=1  ! state 1 is protonated
                 endif
              endif
           enddo
        endif
     else if(wrd.eq.'DEL') then
        nsel = atmsel(comlyn,comlen,islct,.false.)
        if(nsel > 0) then
           do i=1,natom
              if(islct(i).gt.0) then
                 call getresbyat(i,rnum)
                 if(rslct(rnum) /= 0) then
                    ntitr=ntitr-1
                    rslct(rnum)=0
                    tstate(rnum)=0
                 endif
              endif
           enddo 
        endif
     else if(wrd.eq.'CLEA') then
        ntitr=0
        do i=1,nres
           tstate(i)=0
           rslct(i)=0
        enddo
     endif
     call chmdealloc('consph.src','getphresidues','islct',natom,intg=islct)

     ! test and make sure that we have parameters for titratable residues
     do i=1,nres
        if(rslct(i) > 0) then
           resname=res(i)

           k=-1
           do j=1,ntres
              if(cph_params(j)%resname.eq.resname) then
                 k=j
                 exit
              endif
           enddo
           if(k.le.0) then
              write(errmsg,'(2a)') 'NO PH PARAMS FOR RESIDUE ', resname
              call wrndie(-3,'<GETPHRESIDUES>',errmsg)
           endif
        endif
     enddo
  
     if(prnlev.ge.3) write(outu,'(A,I4,A)') 'GETPHRESIDUES> ', ntitr, &
                                            ' RESIDUES ARE TITRATABLE.'

  end subroutine getphresidues

  subroutine dophmc(nstep,phval,igvopt,iasvel,iseed,x,y,z,dx,dy,dz,phunum,phtemp,istart)
     use number
     use consta
     use stream
     use psf
     use energym
     use memory
#if KEY_PARALLEL==1
     use parallel 
#endif
     use clcg_mod,only: random
     use bases_fcm,only: bnbnd,bimag
     implicit none

     integer, intent(in)           :: nstep,iseed,phunum,istart
     integer, intent(inout)        :: igvopt,iasvel
     real(chm_real), intent(inout) :: phval
     real(chm_real), intent(inout) :: x(:), y(:), z(:), dx(:), dy(:), dz(:)
     real(chm_real), intent(in)    :: phtemp

     character(len=4)                   :: resname, segname
     integer                            :: i,j,k,ii,oldstate,rnum
     integer                            :: trgt,newstate,trlstate
     integer, allocatable, dimension(:) :: oslist   ! list of original states
     type(cph_desc)                     :: parm
     logical                            :: found, lsuccess
     real(chm_real)                     :: expr,deprot_ene,prot_ene,ref_ph,ref_dg,prob,p
     real(chm_real)                     :: start_ene,mod_ene
     real(chm_real)                     :: oldcg,scalef,usetemp
     character(len=100)                 :: errmsg

     if(prnlev.ge.6) then
         write(OUTU,'(A)') ' '
         write(OUTU,'(A,I10,A)') 'DOPHMC> START DOPHMC AT STEP ', istart-1, ' current (unmodified) energy:'
         call printe(outu, eprop, eterm, 'PHMC', 'ENR', .true., 0, 0, 0, .false.)
     endif

#if KEY_PARALLEL==1
     if(mynod.eq.0) then
#endif
        call chmalloc('consph.src','dophmc','oslist',nres,intg=oslist)
        do i=1,nres
           oslist(i)=tstate(i)
        enddo
#if KEY_PARALLEL==1
     endif
#endif

     do i=1,nstep

        ! pick a random residue to mutate
#if KEY_PARALLEL==1
        if(mynod.eq.0) then
#endif
           trgt=ceiling(random(iseed)*ntitr)
           if(trgt==0) trgt=1

           ! get the index number in the array of all residues of the random
           ! residue that has been selected from the array of titratable residues.
           k=0
           do j=1,nres
              if(rslct(j).gt.0) then
                 k=k+1
                 if(k==trgt) exit
              endif
           enddo
           resname=res(j)

           ! search the parameter array for the pH parameters corresponding
           ! to the residue type that we picked.
           found = .false.
           do k=1,ntres
              if(cph_params(k)%resname.eq.resname) then
                 found = .true.
                 parm=cph_params(k)
                 exit
              endif
           enddo
           if(.not.found) &
              call wrndie(-4,'<CHANGE_TITRATION_STATE>','CODE ERROR: NO PARAMS FOR RESIDUE TYPE')

           ! remember here, state 1 is ALWAYS PROTONATED
           oldstate=tstate(j)
           if(tstate(j)==1) then
              ! residue is PROTONATED, need to deprotonate it
              prot_ene=eprop(epot)
              if(parm%nstate==2) then
                 call change_titration_state(j,2,resname,segname)
              else
                 p=random(iseed)
                 if(p.gt.0.5) then
                    call change_titration_state(j,3,resname,segname)
                 else
                    call change_titration_state(j,2,resname,segname)
                 endif
              endif
           else
              ! residue is DEPROTONATED, need to protonate it
              deprot_ene=eprop(epot)
              call change_titration_state(j,1,resname,segname)
           endif

           if(prnlev.ge.3) write(outu,'(a,i3,3a,i2,a,i2)') &
                           'DOPHMC> TRY MOVING RESIDUE ', j, ' (', resname, ') FROM STATE ', &
                           oldstate, ' TO ', tstate(j)
           trlstate = tstate(j)
          
#if KEY_PARALLEL==1
        endif

        call psnd8(cg,natom)
        call psnd4(trlstate,1)
#endif

        ! we did not move any atom positions, so we don't need to
        ! update nonbond or image lists, so we can just call
        ! energy here
        ! remember, state 1 is protonated!!!!
        call energy(x,y,z,dx,dy,dz,bnbnd,bimag,1)

        ! set the current energy (deprot_ene or prot_ene, depending on
        ! tstate), to the correct value.
#if KEY_PARALLEL==1
        if(mynod.eq.0) then 
#endif
           if(tstate(j)==1) then
              ! we are PROTONATED!
              if(parm%nstate == 2) then
                 ref_dg=parm%del_dg(1)
                 ref_ph=parm%del_ph(1)
              else
                 if(oldstate == 2) then
                    ref_dg=parm%del_dg(1)
                    ref_ph=parm%del_ph(1)
                 else
                    ref_dg=parm%del_dg(2)
                    ref_ph=parm%del_ph(2)
                 endif
              endif
              prot_ene=eprop(epot)
           else
              ! tstate(j) is not 1 so that means we are DEPROTONATED!
              if(tstate(j)==2) then
                 ref_dg=parm%del_dg(1)
                 ref_ph=parm%del_ph(1)
              else
                 ref_dg=parm%del_dg(2)
                 ref_ph=parm%del_ph(2)
              endif
              deprot_ene=eprop(epot)
           endif
           if(prnlev.gt.5) then
              write(outu,'(a)') 'DOPHMC> PRINT MUTATED ENERGY'
              call printe(outu, eprop, eterm, 'PHMC', 'ENR', .true., 0, 0, 0, .false.)
           endif

           ! test the old vs. new energy to see if the swap is accepted
           if(prnlev.ge.6) then
              write(outu,'(A,F6.2)') 'DOPHMC> phval  = ', phval
              write(outu,'(A,F6.2)') 'DOPHMC> ref_ph = ', ref_ph
              write(outu,'(A,F6.2)') 'DOPHMC> ref_dg = ', ref_dg
              write(outu,'(A,F6.2)') 'DOPHMC> temp   = ', eprop(temps)
              write(outu,'(A,F12.3)') 'DOPHMC> prot_ene = ', prot_ene
              write(outu,'(A,F12.3)') 'DOPHMC> deprot_ene = ', deprot_ene
           endif
           if(phtemp.gt.0.0) then
              usetemp=phtemp
           else
              usetemp=eprop(temps)
           endif
           if(prnlev.ge.3) write(outu,*) 'DOPHMC> use temperature ', usetemp, ' for constant pH.'
           expr=kboltz*usetemp*(phval-ref_ph)*log(10.0)+(prot_ene-deprot_ene)-ref_dg

           ! ok so now we have del f(d->p), if we're trying to move p->d then we need to
           ! multiply by -1.
           if(tstate(j) > 1) then
              expr=-1.0*expr
           endif

           if(expr.le.zero) then
              prob=one
           else
              prob=exp(-expr/(kboltz*usetemp))
           endif
           if(prnlev.ge.3) write(outu,'(A,F7.4)') 'DOPHMC> PROB = ', prob
           p=random(iseed)
           if(prnlev.ge.3) write(outu,'(A,F6.4)') 'DOPHMC> P = ', p

           if(p > prob) then
              lsuccess = .false.
              if(prnlev.ge.3) write(outu,*) 'DOPHMC> SWAP FAILED' 

              ! swap fails, reset everything
              call change_titration_state(j,oldstate,resname,segname)
           else
              lsuccess = .true.
              if(prnlev.ge.3) write(outu,*) 'DOPHMC> SWAP SUCCEEDED'
           endif

#if KEY_PARALLEL==1
        endif 

        call psnd8(cg,natom)
        call psnd4(lsuccess,1)
#endif
        ! If the swap failed, re-do the energy call.
        if (lsuccess .eqv. .false.) call energy(x,y,z,dx,dy,dz,bnbnd,bimag,1)

        ! print out what just happened
#if KEY_PARALLEL==1
        if(mynod.eq.0) then 
#endif
50         format('PHMC SUMMARY> ',I3,1X,I3,1X,'->',I3,1X,F8.3,1X,F5.3,1X,F5.3,1X,L1,1X,I3)
           if(prnlev > 1) write(outu,50) j,oldstate,trlstate,prot_ene-deprot_ene,prob,p,(p <= prob),tstate(j)
           if(prnlev > 3) then
              write(outu,'(a)') 'DOPHMC>       STEP   TEMP   PH     RESID'
              write(outu,'(a)') 'DOPHMC PROB>  SWAP?  REF_PH REF_DG DEPROT_ENE PROT_ENE DIFF'
              write(outu,'(a)') 'DOPHMC EXPR>  EPH    ENE    EXPR   PROB   P'
              write(outu,'(a)') '-------------------------------------------------------------------------'
              write(outu,'(a,x,i4,x,2f10.4,x,i6)') 'DOPHMC>      ', i, eprop(temps), phval, j
              write(outu,'(a,l1,x,x,2f8.2,x,3f12.3)') 'DOPHMC PROB>    ', (p <= prob), &
                         ref_ph, ref_dg, deprot_ene, prot_ene, prot_ene-deprot_ene
              write(outu,'(a,3f12.6,2f8.3)') 'DOPHMC EXPR> ', kboltz*usetemp*(phval-ref_ph)*log(10.0), &
                         (prot_ene-deprot_ene)-ref_dg,expr,prob, p
           endif
#if KEY_PARALLEL==1
        endif
#endif

     enddo ! end of main MC loop
     !!!!call write_ph_state(phunum,istart,oslist)
#if KEY_PARALLEL==1
     if(mynod.eq.0) then 
#endif
        call chmdealloc('consph.src','dophmc','oslist',nres,intg=oslist)
#if KEY_PARALLEL==1
     endif 
#endif

  end subroutine dophmc

  subroutine getresbyat(anum,rnum)
     use psf
     implicit none

     integer, intent(in)  :: anum
     integer, intent(out) :: rnum
     logical              :: rfound
     integer              :: j

     rfound=.false.
     do j=1,nres-1
        if(ibase(j) < anum .and. anum <= ibase(j+1)) then
           rfound=.true.
           rnum=j
           exit        
        endif
     enddo
     if(.not.rfound) rnum=nres

  end subroutine getresbyat

  subroutine getsegbyres(rnum,snum)
     use psf
     implicit none

     integer, intent(in)  :: rnum
     integer, intent(out) :: snum
     logical              :: sfound
     integer              :: j

     do j=1,nseg
        if(nictot(j) < rnum .and. rnum <= nictot(j+1)) then
           sfound = .true.
           snum=j
           exit
        endif
     enddo
     if(.not.sfound) call wrndie(-5,'<getsegbyres>','could not find seg')

  end subroutine getsegbyres

  subroutine change_titration_state(grnum,state,resname,segname)
    use psf
    use stream
    use bases_fcm
    use chm_types
    use image

    integer, intent(in)              :: grnum,state
    character(len=4), intent(out)    :: resname, segname
    type(cph_desc)                   :: parm
    integer                          :: k,l,lrnum,pidx
    logical                          :: found
    character(len=200)               :: errmsg

    resname=res(grnum)
    call getsegbyres(grnum,k)
    segname=segid(k)

    tstate(grnum) = state ! reset the titration state

    ! search the parameter array for the pH parameters corresponding
    ! to the residue type that we picked.
    found = .false.
    do k=1,ntres
       if(cph_params(k)%resname.eq.resname) then
          found = .true.
          parm=cph_params(k)
          exit
       endif
    enddo
    if(.not.found) &
       call wrndie(-4,'<CHANGE_TITRATION_STATE>','CODE ERROR: NO PARAMS FOR RESIDUE TYPE')

    ! sanity check to make sure that the residue is in a protonation state that we know
    ! about.        
    if(tstate(grnum) < 1 .or. tstate(grnum) > 3) then
       write(errmsg,'(a,i2,a,i4)') 'UNKNOWN PROTONATION STATE (', tstate(grnum), &
                                   ') FOR RESIDUE ', grnum
       call wrndie(-3,'<CHANGE_TITRATION_STATE>',errmsg)
    endif

    do k = 1,parm%nmod
       found = .false.
       do l=1,natom 
          call getresbyat(l,lrnum)
          if(lrnum /= grnum) cycle

          if(atype(l) == parm%atnames(k)) then
             ! this is the atom that we want to change
             cg(l)=parm%chrgs(k,tstate(grnum))
             found=.true.
             exit
          endif  
       enddo
       if(.not.found) then
          write(outu,*) 'CODE ERROR: COULD NOT FIND ATOM FOR ', parm%atnames(l)
          call wrndie(-4,'<CHANGE_TITRATION_STATE>','COULD NOT FIND ATOM')
       endif
    enddo

    ! change the titration state of any images atoms
    if(ntrans > 0) then
       do k=natom+1,natim
          pidx=bimag%IMATTR(k)
          if(pidx > 0) then
             cg(k)=cg(pidx)
          endif
       enddo
    endif

  end subroutine change_titration_state

  subroutine write_ph_state(phunum,istart,oslist)
     use stream
     use psf
#if KEY_PARALLEL==1
     use parallel 
#endif

     integer, intent(in)               :: phunum,istart
     integer, intent(in), dimension(:) :: oslist
     integer                           :: i

#if KEY_PARALLEL==1
     if(mynod.eq.0) then 
#endif
        if(phunum > 0 .and. iolev > 0) then
           write(phunum,'(i10,a)',advance='no') istart-1, ' '
           do i=1,nres
              if(tstate(i).ne.0) write(phunum,'(I3,A,I3,A)',advance='no') oslist(i), '->', tstate(i), ' '
           enddo
           write(phunum,'(a)') ' '
           call flush(phunum)
        endif
#if KEY_PARALLEL==1
     endif
#endif
  end subroutine write_ph_state

end module consph
