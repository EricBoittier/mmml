module pins
  !   by Florent Hedin & Markus Meuwly, Department of Chemistry, University of Basel, Switzerland
  !   by Nuria Plattner, Department of Mathematics and Computer Science, Freie Universitaet Berlin, Germany
  !
  !   florent.hedin@unibas.ch
  !   April, 2015 -- February, 2016
  !
  !   See the following references:
  !
  !   1) N. Plattner et al., J. Chem. Phys., 135, 134111, (2011), DOI: 10.1063/1.3643325
  !   2) P. Dupuis et al., Multiscale Model. Simul., 10(3), 986–1022, DOI: 10.1137/110853145
  !   3) J. D. Doll et al., J. Chem. Phys., 137, 204112, (2012), DOI: 10.1063/1.4765060
  !   4) N. Plattner et al., J. Chem. Theory Comput., 2013, 9 (9), pp 4215–4224, DOI: 10.1021/ct400355g
  !   5) J. D. Doll et al., J. Chem. Phys. 142, 024111 (2015); DOI: 10.1063/1.4904890
  !   6) F. Hedin et al., J. Chem. Theory Comput., Submitted

  use chm_kinds, only: chm_int4,chm_real

  implicit none

  !logical variable set to true in ensemble.src-->ensrex if infinite swapping enabled for the current replica exchange simulation (i.e. if 'ENSEMBLE EXCHANGE PINS' is found in the input file)
  logical,save :: lpins=.false.
#if KEY_ENSEMBLE==1 /* ensemble required for compiling infinite swapping */
#if KEY_PINS==1 /* partial inf swapping */

  !Variables for infinite swapping

  ! maxblocks : maximal number of blocks for set1|set2, defined as max(nbm1,nbm2)
  ! maxperm : ?? Any way of calculating the value from simulation parameters instead of chossing a too large value as 800 ?
  ! maxtempset : ?? Any way of calculating the value from simulation parameters instead of chossing a too large value as 100 ?
  integer(chm_int4),save :: maxblocks=100, maxperm=800, maxtempset=100

  !NTSET   number of temperature sets
  !NBM1   number of symmetrized blocks in set 1
  !NBM2   number of symmetrized blocks in set 2
  !MSEL   currently selected block
  integer(chm_int4),save :: ntset, nbm1, nbm2, msel

  !SWPREP inverse index to repswp
  integer(chm_int4),save,allocatable,dimension(:) :: swprep

  !NSWT  (2, maxblocks) number of temperatures to be symmetrized per block
  !NPERM   (2, maxblocks) number of permutation (NSWT!)
  integer(chm_int4),save,allocatable,dimension(:,:) :: nswt, nperm

  !IIND  (2, maxblocks, maxperm, maxtempset) swapping index
  integer(chm_int4),save,allocatable,dimension(:,:,:,:) :: iind

  !Restrict access to variables and subroutines which are not required from outside of this module
  private
  public  :: lpins
  public  :: iswini, iswfin, iswparse, iswprintrex, ptswap, isw_ensswl

  contains

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------

  !for allocating arrays used by this module
  subroutine iswalloc()
    use memory, only: chmalloc
    use stream, only: outu,prnlev
    use ensemble, only: nensem

    implicit none

    if (prnlev >= 2) then
      write(outu,'(a)') 'ISWALLOC> Allocating memory for infinite swapping routines ...'
    endif

    call chmalloc('pins.src','iswalloc','swprep',nensem,ci4=swprep)
    call chmalloc('pins.src','iswalloc','nswt',2,maxblocks,ci4=nswt)
    call chmalloc('pins.src','iswalloc','nperm',2,maxblocks,ci4=nperm)
    call chmalloc('pins.src','iswalloc','iind',2,maxblocks,maxperm,maxtempset,ci4=iind)

    return
  end subroutine iswalloc

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------

  !for deallocating arrays used by this module
  subroutine iswdealloc()
    use memory, only: chmdealloc
    use stream, only: outu,prnlev
    use ensemble, only: nensem

    implicit none

    if (prnlev >= 2) then
      write(outu,'(a)') 'ISWDEALLOC> Deallocating memory for infinite swapping routines ...'
    endif

    call chmdealloc('pins.src','iswdealloc','swprep',nensem,ci4=swprep)
    call chmdealloc('pins.src','iswdealloc','nswt',2,maxblocks,ci4=nswt)
    call chmdealloc('pins.src','iswdealloc','nperm',2,maxblocks,ci4=nperm)
    call chmdealloc('pins.src','iswdealloc','iind',2,maxblocks,maxperm,maxtempset,ci4=iind)

    return
  end subroutine iswdealloc

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  !read infinite swapping parameters from command line
  subroutine iswparse(comlyn,comlen)
  
    use string, only: gtrmi,nexti
    
    implicit none
    
    !received
    character(len=*),intent(inout) :: comlyn
    integer(chm_int4),intent(in)   :: comlen
    
    !local
    integer(chm_int4) :: i
    
    nbm1=gtrmi(comlyn,comlen,'NBM1',0)
    nbm2=gtrmi(comlyn,comlen,'NBM2',0)
    
    !default size of the max* variables is too large so adjust it for reducing memory consumption when allocating
    !maxblocks = max(nbm1,nbm2)
    !maxperm=...
    !maxtempset=...
    
    call iswalloc()
    
    do i=1,nbm1
      nswt(1,i) = nexti(comlyn,comlen)
    enddo
    
    do i=1,nbm2
      nswt(2,i) = nexti(comlyn,comlen)
    enddo
  
    return
    
  end subroutine iswparse
  
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  !Summarize simulation params to main output unit if required
  subroutine iswprintrex()

    use stream, only: outu,prnlev

    implicit none

    integer(chm_int4) :: i

    if(prnlev>=2) then

      write(outu,'(A,I8,A)') ' ENSREX> ',nbm1,' blocks in 1st PINS set'
      do i=1, nbm1
        write(outu,'(A,I8,A,I8)') ' ENSREX> PINS BLOCK ', i, 'NSWT ', nswt(1,i)
      enddo
      write(outu,'(A,I8,A)') ' ENSREX> ',nbm2,' blocks in 2nd PINS set'
      do i=1, nbm2
        write(outu,'(A,I8,A,I8)') ' ENSREX> PINS BLOCK ', i, 'NSWT ', nswt(2,i)
      enddo

    endif

    return

  end subroutine iswprintrex

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  !subroutine isw_get_optimal_size()
  !  implicit none
  !
  !end subroutine isw_get_optimal_size
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------

  !additional subroutine for infinite swapping initialization
  subroutine iswini()
    !-----------------------------------------------------------------------
    !Setup permutations
    !-----------------------------------------------------------------------
    use stream, only: outu,prnlev
    use ensemble, only: nensem
    
    implicit none

    !local
    integer(chm_int4)  :: i, j, n1, n2, n3, n4, n5, n6, m, indc, tsel
    integer(chm_int4),dimension(6) :: lab
    
    if (prnlev >= 2) then
      write(outu,'(A,I8)') 'ISWINI> Initializing the infinite swapping module with nensem = ',nensem
      write(outu,'(A,I8)') 'ISWINI> maxblocks = ',maxblocks
      write(outu,'(A,I8)') 'ISWINI> maxperm = ',maxperm
      write(outu,'(A,I8)') 'ISWINI> maxtempset = ',maxtempset
    endif

    !    start with block 1
    msel = 1

    do m = 1, 2 !do level 1

      if (m .eq. 1) then
        ntset = nbm1
      else
        ntset = nbm2
      endif

      tsel = 1

      do i = 1, ntset !do level 2

        indc = 0
        nperm(m, i) = nswt(m, i)

        do j = 1, (nswt(m, i) - 2) !do level 3
          nperm(m, i) = nperm(m, i)*(nswt(m, i) - j)
        enddo !do level 3

        do n1 = tsel, (tsel + nswt(m, i) - 1) !do level 3

!           write(outu,*) 'filling lab(1)'
          lab(1) = n1

          do n2 = tsel, (tsel + nswt(m, i) - 1) !do level 4

            if (n2 .ne. lab(1)) then
!               write(outu,*) 'filling lab(2)'
              lab(2) = n2
              do n3 = tsel, (tsel + nswt(m, i) - 1) !do level 5
                if ((n3 .ne. n2) .and. (n3 .ne. n1)) then
!                   write(outu,*) 'filling lab(3)'
                  lab(3) = n3
                  if (nswt(m, i) .eq. 3) then
                    indc = indc + 1
                    iind(m, i, indc, tsel) = lab(1)
                    iind(m, i, indc, tsel + 1) = lab(2)
                    iind(m, i, indc, tsel + 2) = lab(3)
                  else if (nswt(m, i) > 3) then
                    do n4 = tsel, (tsel + nswt(m, i) - 1) !do level 6
                      if ((n4 .ne. n1).and.(n4 .ne. n2).and.(n4 .ne. n3)) then
!                         write(outu,*) 'filling lab(4)'
                        lab(4) = n4
                        if (nswt(m, i) .eq. 4) then
                          indc = indc + 1
                          iind(m, i, indc, tsel) = lab(1)
                          iind(m, i, indc, tsel + 1) = lab(2)
                          iind(m, i, indc, tsel + 2) = lab(3)
                          iind(m, i, indc, tsel + 3) = lab(4)
                        else if (nswt(m, i) > 4) then
                          do n5 = tsel, (tsel + nswt(m, i) - 1) !do level 7
                            if ((n5 .ne. n1).and.(n5 .ne. n2).and.(n5 .ne. n3).and.(n5 .ne. n4)) then
!                               write(outu,*) 'filling lab(5)'
                              lab(5) = n5
                              if (nswt(m, i) .eq. 5) then
                                indc = indc + 1
                                iind(m, i, indc, tsel) = lab(1)
                                iind(m, i, indc, tsel + 1) = lab(2)
                                iind(m, i, indc, tsel + 2) = lab(3)
                                iind(m, i, indc, tsel + 3) = lab(4)
                                iind(m, i, indc, tsel + 4) = lab(5)
                              else if (nswt(m, i) .eq. 6) then
                                do n6 = tsel, (tsel + nswt(m, i) - 1) !do level 8
                                  if ((n6 .ne. n1).and.(n6 .ne. n2).and.(n6 .ne. n3).and.(n6 .ne. n4).and.(n6 .ne. n5)) then
!                                     write(outu,*) 'filling lab(6)'
                                    lab(6) = n6
                                    indc = indc + 1
                                    iind(m, i, indc, tsel) = lab(1)
                                    iind(m, i, indc, tsel + 1) = lab(2)
                                    iind(m, i, indc, tsel + 2) = lab(3)
                                    iind(m, i, indc, tsel + 3) = lab(4)
                                    iind(m, i, indc, tsel + 4) = lab(5)
                                    iind(m, i, indc, tsel + 5) = lab(6)
                                  endif
                                enddo !do level 8
                              endif
                            endif
                          enddo !do level 7
                        endif
                      endif
                    enddo !do level 6
                  endif
                endif
              enddo !do level 5
            endif
          enddo !do level 4
        enddo !do level 3
        tsel = tsel + nswt(m, i)
      enddo !do level 2
!       write(outu,*) 'indc ',indc,' tsel ',tsel
    enddo !do level 1

    if (prnlev >= 2) then
      write(outu,'(a)') 'ISWINI> Initialisation of the infinite swapping module done !'
    endif

    return
  end subroutine iswini

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  
  !for ending simulation properly, mainly deallocating memory
  subroutine iswfin()
    
    implicit none

    call iswdealloc()

    return

  end subroutine iswfin

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  
  subroutine ptswap(tset,epot,iperm,m,nsl)
  !-----------------------------------------------------------------------
  !     calculates rho-values infinite swapping exchange
  !     and selects a permutation
  !-----------------------------------------------------------------------      
    use stream, only: outu,prnlev
    use consta, only: kboltz
    use reawri,only: iseed
    use clcg_mod,only:random
    use ensemble, only: enstem

    implicit none

    !received
    integer(chm_int4)  :: tset,iperm,m,nsl
    real(chm_real),dimension(maxblocks) :: epot

    !local
    integer(chm_int4) :: np,npi,npj,j
    real(chm_real)    :: rnum,rhosum,eevl,mysum,eval
    real(chm_real),dimension(maxblocks,maxperm) :: rho
    real(chm_real),dimension(maxperm)     :: arg

    !  evaluation of rho values
    do np=1,nperm(m,nsl)
      arg(np)     = 0.0d0
      do j=tset, (tset+nswt(m,nsl)-1)
!          write(outu,*) "PTSWAP> in loop np,j=",np,j 
        eevl = epot(iind(m,nsl,np,j))/((enstem(j)*kboltz))
        arg(np)     = arg(np) + eevl
      enddo
    enddo

    do npi=1,nperm(m,nsl)
      mysum = 0d0
      do npj=1,nperm(m,nsl)
        eval = -arg(npj)+arg(npi)
        mysum  = mysum + dexp(eval)
      enddo
      rho(nsl,npi)  = 1.0d0/mysum
!      if (npi < 10) then
!        if (prnlev >= 3) then
!          write(outu,'(a,i8,f7.3)') 'PTSWAP> RHO',npi, rho(nsl,npi)
!        endif
!      endif
    enddo

    !   permutation selection
    rnum = random(iseed)
    rhosum = 0d0
    do np=1,nperm(m,nsl)
      rhosum = rhosum + rho(nsl,np)
!      if (prnlev >= 3) then
!        write(outu,'(a,f7.3,a,f7.3)') "PTSWAP> rnum ",rnum," rho(nsl,np) ",rho(nsl,np)
!      endif
      if(rnum<=rhosum) then
        iperm = np
        return
      endif
    enddo

    call wrndie(-5,'<PTSWAP>','Something wrong with permutation choice, see pins.src, &
                    subroutine ptswap, block of code marked with permutation selection')

    return

  end subroutine ptswap

  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  ! modified copy of ensswl(...) from ensemble.src
  subroutine isw_ensswl(xold,yold,zold,xnew,ynew,znew,xcomp,ycomp,zcomp,natom)
  
    use reawri, only: iseed
    use clcg_mod, only:random
    use coord, only: x,y,z,wmain
    use deriv, only: dx,dy,dz
    use bases_fcm, only: bimag,bnbnd
    use comand, only: comlyn,comlen
    use parallel, only: comm_charmm,mynod
    use consta, only: kboltz
    use number, only: zero,one,two
    use stream, only: prnlev,outu
    use mpi, only: mpi_double_precision
    use energym, only: epot,eprop,energy
    use ensemble
    
    implicit none
    
    !received
    integer(chm_int4) :: natom
    real(chm_real),dimension(natom) :: xold,yold,zold,xnew,ynew,znew,xcomp,ycomp,zcomp
    
    !local from ensswl
    integer(chm_int4) :: i,j,k,ierror,ri,rj,partner,tmpi,ii,s,atfrst,atlast,kk,tempi,status
    integer(chm_int4), parameter :: myenel=4
    real(chm_real) :: tempd,eii,eij,ejj,eji,cdel,rnum,scafac,oldt,db
    real(chm_real) :: p_i,v_i,p_j,v_j
    real(chm_real) :: pivi,pivj,pjvi,pjvj
    logical :: qswp,sendfirst,debug,swapped
    real(chm_real),dimension(3) :: pist,pistmp
    real(chm_real),dimension(6) :: tmp6
    real(chm_real),dimension(maxswp) :: randtbl
    real(chm_real),dimension(myenel) :: myene
    real(chm_real),dimension(myenel*maxens) :: ensene
    integer(chm_int4),dimension(maxswp) :: swtmpi,swtmpj,ssave
    
    !local for infinite swapping
    integer(chm_int4) :: nl,nbl,ne,n2,iperm,partnr,partns,ml,nsw
    integer(chm_int4),dimension(nensem) :: prw,rtm,rte,mind
    real(chm_real),dimension(nensem)  :: swpene
    
    debug=.false.
    
    !     ROOT decides which nodes will attempt a swap
    if (mastermaster) then
    
      if(prnlev >= 2) then
        write(outu,'(a)') ' ******************************************************'
        write(outu,'(a)') ' **   Testing whether replicas need to be swapped ...**'
      endif

      if (msel .eq. 1) then
        nbl = nbm2
      else
        nbl = nbm1
      endif
          
      do k=1,nensem
        repswp(k)=0
        swprep(k)=0
        ensatt(k) = ensatt(k)+1
        mind(k) = k
      enddo
          
      s = 1
      do nl=1, nbl
        i=ensisw(s)
        j=ensjsw(s)
        if(prnlev >= 2) then
          write(outu,'(a,i3,a,i3)') ' ATTEMPTING TO SWAP REPLICAS ',i,' TO ',(i+nswt(msel,nl)-1)
        endif
        call flush(outu)
        s = s + nswt(msel,nl)
      enddo
    endif

    !     ROOT broadcasts swap array to other nodes
    call ens_bcast_all(repswp,nensem)

    myene(1)=eprop(epot)


    call ens_global_barrier(ierror)
    
    if (mynod == 0) then
        call mpi_allgather( &
            myene, myenel,mpi_double_precision, &
            ensene,myenel,mpi_double_precision, &
            comm_master,ierror)
    endif
      
    !     ROOT decides if TRIAL NODES will swap
    if (ensmasternod == 0) then

      do ne=1, nensem
        swpene(ne) = ensene((ne-1)*myenel+1)
!        if(prnlev >= 3) then
!          write(outu,'(a,i8,a,f7.3,a,f7.3)') "ne ",ne," swpene(ne) ",swpene(ne)," enstem(ne)*kboltz ",enstem(ne)*kboltz
!        endif
        rte(ne) = swpene(ne)
      enddo
        
      do ml=1,2
      
        s = 1
        
        do nl=1, nbl
        
          i=s
          qswp=.false.
          iperm=0

          call ptswap(s,swpene,iperm,msel,nl)
          
!          if(prnlev >= 2) then
!            write(outu,'(A,I3,A,I3)') 'chain ', msel, '  selected permutation= ', iperm
!            write(outu,'(6I3)') iind(msel,nl,iperm,s),iind(msel,nl,iperm,(s+1)), iind(msel,nl,iperm,(s+2)),iind(msel,nl,iperm,(s+3)), iind(msel,nl,iperm,(s+4)),iind(msel,nl,iperm,(s+5))
!          endif
          
          call flush(outu)

          if (iperm .ne. 1) then
            qswp=.true.
          else
            qswp=.false.
          endif
          
          if (qswp) then
          
!            if(prnlev >= 2) then
!              write(outu,'(a,i3,a,i3)')  &
!                    'swapping replicas for nodes ', i, &
!                    ' to ', (i+nswt(msel,nl)-1)
!              call flush(outu)
!            endif
            
            ! this just tracks which rep is at which temp for post-analysis
            do n2=i, (i+nswt(msel,nl)-1)
              if (ml .eq. 1) then 
                swpene(n2) = rte(iind(msel,nl,iperm,n2))
                mind(n2) = iind(msel,nl,iperm,n2)
                rtm(n2)=t2rep(n2)
              else
                rtm(n2)=t2rep(n2)
              endif
            enddo
            
            do n2=i, (i+nswt(msel,nl)-1)
            
              if (ml .eq. 1) then
                t2rep(n2)= rtm(iind(msel,nl,iperm,n2))
                rep2t(rtm(n2))= iind(msel,nl,iperm,n2)
                repswp(n2)= -iind(msel,nl,iperm,n2)
                swprep(iind(msel,nl,iperm,n2)) = n2
              else
                nsw = mind(iind(msel,nl,iperm,n2))
                repswp(n2)= -nsw
                swprep(nsw) = n2
                t2rep(n2)= rtm(iind(msel,nl,iperm,n2))
                rep2t(rtm(n2))= iind(msel,nl,iperm,n2)
              endif
            
              if (abs(repswp(n2)) .ne. n2 .and. ml .eq. 2) then
                enssuc(n2)=enssuc(n2)+1
              endif
              
            enddo

          endif
          
          s = s + nswt(msel,nl)
          
        enddo
        
        if (ml .eq. 1) then
          if (msel .eq. 1) then
            msel = 2
            nbl = nbm2
          else
            msel = 1
            nbl = nbm1
          endif
        endif
        
      enddo !DO ML=1,2
    endif ! end if (ensmasternode)

    !     ROOT broadcasts swap array to other nodes
    if(lmasternode)then 
        call psnd4_comm(comm_master,repswp,nensem)
        call psnd4_comm(comm_master,swprep,nensem)
        call psnd4_comm(comm_master,rep2t,nensem)
        call psnd4_comm(comm_master,t2rep,nensem)
!        if(prnlev >= 3) then
!          write(outu,'(a,10i4)') 'rep2t ', rep2t(1:nensem)
!          write(outu,'(a,10i4)') 't2rep ', t2rep(1:nensem)
!          call flush(outu)
!        endif
    endif
    
    !other nodes
    call psnd4_comm(comm_charmm,repswp,nensem)
    call psnd4_comm(comm_charmm,swprep,nensem)
    call psnd4_comm(comm_charmm,rep2t,nensem)
    call psnd4_comm(comm_charmm,t2rep,nensem)

    !     TRIAL NODES swap coordinates if success
    !     else also swap other dynamics arrays
    if (repswp(whoiam+1) /= 0 .and. abs(repswp(whoiam+1)).ne.(whoiam+1) ) then
    
      partner=abs(repswp(whoiam+1))
      partnr=abs(repswp(whoiam+1))
      partns=swprep(whoiam+1)
      
      do n2 = 1, nensem
        if (n2 .eq. (whoiam+1)) then
          prw(n2) = -partns
        else if (n2 .eq. partnr) then
          prw(n2) = partnr
        else
          prw(n2) = 0
        endif
      enddo

      if (repswp(whoiam+1) < 0) then
        !           ... success
        !           swap coordinates used in leapfrog algorithm
        if(partnr .eq. partns)then
        
          if (partner>(whoiam+1)) then
            sendfirst=.true.
          else
            sendfirst=.false.
          endif
        
          call ensscv(xold,ensbuf,natom,partner-1,sendfirst)
          call ensscv(yold,ensbuf,natom,partner-1,sendfirst)
          call ensscv(zold,ensbuf,natom,partner-1,sendfirst)
          call ensscv(xnew,ensbuf,natom,partner-1,sendfirst)
          call ensscv(ynew,ensbuf,natom,partner-1,sendfirst)
          call ensscv(znew,ensbuf,natom,partner-1,sendfirst)
          call ensscv(xcomp,ensbuf,natom,partner-1,sendfirst)
          call ensscv(ycomp,ensbuf,natom,partner-1,sendfirst)
          call ensscv(zcomp,ensbuf,natom,partner-1,sendfirst)
          
        else
        
          call ens2scv(xold,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(yold,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(zold,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(xnew,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(ynew,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(znew,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(xcomp,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(ycomp,ensbuf,natom,partns-1,partnr-1,prw)
          call ens2scv(zcomp,ensbuf,natom,partns-1,partnr-1,prw)
          
        endif
        
        !           update nb
        call update(comlyn,comlen,xcomp,ycomp,zcomp,wmain, &
              .false.,.false.,.true., &
              .false.,.true.,0,0,0,0,0,0,0)

        if (repswp(whoiam+1)<0 ) then 
          oldt=enstem(partner)
          scafac=sqrt(ensmyt/oldt)
          if (jensc) then
              do k=1,natom
                xnew(k)=(two*scafac-one)*xold(k)+scafac*(xnew(k)-xold(k))
                ynew(k)=(two*scafac-one)*yold(k)+scafac*(ynew(k)-yold(k))
                znew(k)=(two*scafac-one)*zold(k)+scafac*(znew(k)-zold(k))
              enddo
          endif
        endif


      else ! else of : if (repswp(whoiam+1) < 0) then
      
        call update(comlyn,comlen,xcomp,ycomp,zcomp,wmain, &
              .false.,.false., &
              .true.,.false.,.true.,0,0,0,0,0,0,0)
              
      endif !endif of : if (repswp(whoiam+1) < 0) then
        
    endif ! endif of : if (repswp(whoiam+1) /= 0 .and. abs(repswp(whoiam+1)).ne.(whoiam+1) ) then
    
  end subroutine isw_ensswl
  
  !-----------------------------------------------------------------------
  !-----------------------------------------------------------------------
  ! additional coordinates swapping routine used only with infinite swapping simulations
  subroutine ens2scv(array,tmpv,natom,partnr,partns,prw)
  
    use ensemble, only: nensem, comm_master, lmasternode
    use parallel
    use mpi
    
    implicit none
    
    !received
    integer(chm_int4) :: natom,partnr,partns
    integer(chm_int4),dimension(nensem) :: prw
    real(chm_real),dimension(natom) :: array,tmpv
    
    !local
    integer(chm_int4) :: ierror,stat(mpi_status_size),itag,icom
    integer(chm_int4) :: i,n1
    
    itag = 9
    
    do n1=1, nensem
    
      itag = itag+1
      icom = abs(prw(n1))-1 

      if (prw(n1)<0) then
        if(lmasternode)then
            call mpi_send(array,natom,mpi_double_precision,icom,itag, &
                comm_master,ierror)
        endif
      else if (prw(n1)>0) then
        if(lmasternode)then
            call mpi_recv(tmpv,natom,mpi_double_precision,icom,itag, &
                comm_master,stat,ierror)
        endif
      endif
    
    enddo
    
    call psnd8_comm(comm_charmm,tmpv,natom)

    array=tmpv
    
    return
    
  end subroutine ens2scv

#endif /* partial inf. swapping */
#endif /* ensemble module */
end module pins
