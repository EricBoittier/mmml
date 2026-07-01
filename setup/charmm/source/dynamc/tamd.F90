module TAMDmodule
#if KEY_TAMD==1
  use chm_kinds
  use chm_types
  implicit none
  !-----------------------------------------------------------------------
  !     Torsion Angle Molecular Dynamics: tree topologies and internal 
  !     coordinate quantities
  !
  ! ... dimensions (move to dimens.f90 later)
  !
  !     TADVER:   relavant only in TAMD related I/O (restart, tree files)
  !     MAXCLUS:  maximum number of clusters
  !     MAXCHILD: maximum number of children per cluster
  !
  integer,parameter :: TADVER=2,MAXCLUS=3000,MAXCHILD=200,MAXSEG=100
  !
  ! ... tree topology
  integer,save :: NClus     ! number of clusters
  integer,save :: TreeReady ! whether complete tree topology is defined
  !  integer,dimension(1:MAXSEG+1),save :: NCSeg 
  integer,allocatable,dimension(:),save :: NCSeg 
  !     cluster partitions in segments. NCSeg(1)=0, NCSeg(NSeg+1)=NClus 

  !  integer,dimension(1:MAXCLUS),save :: OrigPt,HingePt,Parent,NChild
  integer,allocatable,dimension(:),save :: OrigPt,HingePt,Parent,NChild
  integer,allocatable,dimension(:),save :: ClusID
  !  integer,dimension(1:MAXCHILD,1:MAXCLUS),save :: Child
  integer,allocatable,dimension(:,:),save :: Child
  !
  ! ... Dynamics parameters (variable have same meaning as in reawri.f90)
  integer,save :: iseed, nstep, nsavc, iprfrq, nsavv, npriv, ntrfrq, &
       iasors, iunrea, iunwri, iuncrd, iunvel, isvfrq
  !
  !     QREF: response time of thermo coupling in unit of timesteps. 
  real(chm_real),save :: timest, delta, tref, qref1, echeck, firstt

  ! ... most auxillary subroutine and functions are private
  private PreClus, SetupTree, PrnClus, AClus19, AClus22, CheckTree, &
       tadmini, GenSpOp1, GetTc, tadcntrl, GetKen,  & !TryOneStep,
       AssSpVel, GetTe, GenSpOp2, UpdateCor2, UpdateBase,  &
       UpdateSpVel2, GetScalVel, SPTAD2, makepk, makephik,  &
       WriteTree, ReadTree, readtadyn, writadyn, addatom, AddClus
#endif 
contains
#if KEY_TAMD==1 /*TAMDMAIN*/

  SUBROUTINE TAMD
    !     
    !     Torsion Angle Molecular Dynamics (TAMD): 
    !
    !     This subroutine handles all TAMD related commands including
    !     tree build facilities, torsion angle dynamics and simple energy
    !     minimization in torsion space. 
    !
    !     Several CHARMM commands are also parsed and passed to the 
    !     appropriate subroutines.
    !
    !     v 0.1: single chain; no loop allowed
    !     v 0.2: multiple chains; no loop allowed
    !

#if KEY_CHEQ==1
    use cheq,only: qcg,   &                  
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH                   
#endif
    use cstran_mod,only:mkfrat,cstran
    use dimens_fcm
    use number
    use consta
    use stream
    use coord
    use comand
    use corman_mod,only:corcom
    use eutil
    use inbnd
    use deriv
    use shake
    use image
    use psf
    use contrl
    use memory
    use select
    use string
#if KEY_TSM==1
    use tsms_mod
#endif 
    use heurist,only:updeci

    implicit none
    !
    ! ... local
    integer iunit
    ! ... TAMD variables
    !-------------------------------------------------------------------------
    !     SpVel,                ! spatial velocity
    !     vhk,                  ! unit vector h(k):=q(k,orig)-q(k,hinge)
    !     VOrigk,               ! vOrigk=q(k,orig)-q(p(k),orig), i.e.,l(k+1,k)
    !     VCMk,                 ! vector to Center of Mass(CM):=q(k,CM)-q(k,orig)
    !     tmk,                  ! total mass of each cluster
    !     ITk,                  ! cluster intertia tensor w/ repect q(k,orig)
    !     Tk,                   ! hinge foce along h(k)
    !     Theta,                ! torsional angle increments of h(k)
    !     dTheta,               ! torsional speed of h(k)
    !     ddTheta,              ! torsional acceleration of h(k)
    !     dThetaNew, dThetaOld  ! new and old torsional velocities
    !     VX,VY,VZ              ! cartesian velocities
    !     XOLD,YOLD,ZOLD        !(for calling nonbond update only)
    real(chm_real),allocatable, dimension(:) ::  &
         tmk,tk,theta,dtheta,ddtheta,dThetaOld,dThetaNew, &
         vx,vy,vz,xold,yold,zold
    real(chm_real),allocatable, dimension(:,:) :: &
         vhk,vorigk,vcmk,spvel
    real(chm_real),allocatable, dimension(:,:,:) :: ITk 
    !
    integer,allocatable,dimension(:) :: islct, freeat
    integer :: alloc_err

    integer nfreat, prnold, k,nn
    character(len=4) wrd
    logical done,eof,lused
    !
    ! ... begin
    if(prnlev >= 1) write(outu,100) 'Torsion Angle Molecular Dynamics'
100 format(/,6x,2A)
    !
    ! allocate base arrays if not allocated
    if(.not.allocated(NCSeg)) then
       call chmalloc('tamd.src','tamd','NCSeg',MAXSEG+1,intg=NCSeg)
    endif
    if(.not.allocated(OrigPt)) then
       call chmalloc('tamd.src','tamd','OrigPt',MAXCLUS,intg=OrigPt)
       call chmalloc('tamd.src','tamd','HingePt',MAXCLUS,intg=HingePt)
       call chmalloc('tamd.src','tamd','Parent',MAXCLUS,intg=Parent)
       call chmalloc('tamd.src','tamd','NChild',MAXCLUS,intg=NChild)
    endif
    if(.not.allocated(Child)) then
       call chmalloc('tamd.src','tamd','Child',MAXCHILD,MAXCLUS,intg=Child)
    endif

    ! allocate arrays used in tree building
    if (.not. allocated(clusid)) then
       call chmalloc('tamd.src', 'tamd', 'clusid', natom, intg=clusid)
    else if (size(clusid) < natom) then
       call chmrealloc('tamd.src', 'tamd', 'clusid', natom, intg=clusid)
    end if

    allocate(islct(natom),stat = alloc_err)
    if(alloc_err /= 0) &
         CALL WrnDie(-1,'<TAMD>', &
         'Failed to allocate memory for islct array')

    eof=.false.
    done=.false.
    do while(.not.done)
       !
110    call xtrane(COMLYN,COMLEN,'TAMD')
       call rdcmnd(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
            'TAMD> ')
       !
       !     This should not happen. no stream handling inside TAMD
       IF(EOF) then
          call wrndie(0,'<TAMD>', &
               'EOF reached prematurally. abort')
          go to 9999
       endif
       !
       CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED) ! handle miscellaneous commands
       if(LUSED) goto 110
       !
       wrd=NEXTA4(COMLYN,COMLEN)
       !     
       if(wrd == '    ') then
          !
          continue              ! blank line
          !
       else if(wrd == 'RESE') then ! reset and intialization
          !
          TreeReady=0
          IUNREA=-1
          IUNWRI=-1
          IUNCRD=-1
          IUNVEL=-1
          NSAVC=10
          NSAVV=10
          NTRFRQ=500
          IASORS=1              
          NSTEP=100
          DELTA=0.0
          TIMEST=0.002
          TREF=ROOMT
          ECHECK=20.0
          ISEED=76052727
          NClus=0
          do k=1,NATOM
             ClusID(k)=0
          enddo
          !
       else if(wrd == 'CLUS') then ! make single cluster from selection
          !
          if(TreeReady == 1) then
             if(prnlev >= 3) write(outu,'(A)')  &
                  'TAMD> CLUS command should precede TREE command'
             call wrndie(3,'<TAMD>', &
                  'Tree already set up. nothing done.')
          else
             call selcta(COMLYN,COMLEN,islct,X,Y,Z,WMAIN,.true.)
             call PreClus(islct)
          endif
          !          
       else if(wrd == 'TREE') then ! handle tree topology
          !
          wrd=nexta4(COMLYN,COMLEN)
          if(wrd == 'SETU') then
             call SetupTree(COMLYN,COMLEN)
          else if(wrd == 'CHEC') then
             allocate(tmk(nclus),stat = alloc_err)
             if(alloc_err /= 0) &
                  CALL WrnDie(-1,'<TAMD>', &
                  'Failed to allocate memory for tmk array')
             call CheckTree(tmk)
             deallocate(tmk)
          else if(wrd == 'READ') then
             iunit=gtrmi(comlyn,comlen,'UNIT',-1)
             if(iunit /= -1) then
                if(NClus > 0) call wrndie(3,'<TAMD>', &
                     'Overwrite old tree specifications')
                call ReadTree(iunit)
             else
                call wrndie(0,'<TAMD>', &
                     'No unit specified. nothing done.')
             endif
          else if(wrd == 'WRIT') then
             if(TreeReady /= 1) call wrndie(3,'<TAMD>', &
                  'Write an incomplete tree topology')
             iunit=gtrmi(comlyn,comlen,'UNIT',outu)
             call WriteTree(iunit)
          else if(wrd == 'PRIN') then
             if(TreeReady /= 1) call wrndie(3,'<TAMD>', &
                  'Print an incomplete tree topology')
             call PrintTree()
          else
             !             more keywords from TREE command?
             if(prnlev >= 3) write(outu,'(2A)')  &
                  ' TAMD> Unrecognized keyword in TREE ', WRD
          endif
          !
       else if(wrd == 'MINI') then ! energy minimization
          !
          if(TreeReady /= 1) then
             if(prnlev >= 3) then
                write(outu,120) 
120             format(' TAMD>', &
                     ' The tree topology has not been properly set up yet.',/, &
                     ' Energy minimization in internal coordinate requires proper', &
                     ' tree topology.',/, ' The tree topology setup is finalized', &
                     ' by TREE command.')
             endif
             call wrndie(-3,'<TAMD>', &
                  'Minimization w/o proper tree setup')
             continue           ! refuse to do anything
          endif
          !
          if(XATIM /= 0) call wrndie(-3,'<TAMD>', &
               'IMAGE not allowed in TAMD')
          !
          nn=NClus
          allocate(tk(nn),vhk(3,nn),vorigk(3,nn),theta(nn), &
               stat = alloc_err)
          if(alloc_err /= 0) &
               CALL WrnDie(-1,'<TAMD>', &
               'Failed to allocate memory for tk arrays')

          prnold=prnlev
          if(prnlev == 5) prnlev=4 ! reduce updeci() print
          call tadmini( COMLYN,COMLEN, &
               Tk, vhk,  &
               VOrigk, theta )
          prnlev=prnold

          deallocate(tk,vhk,vorigk,theta)

       else if(wrd == 'DYNA') then ! torsion angle dyamics
          if(TreeReady /= 1) then
             if(prnlev >= 3) then
                write(outu,130) 
130             format(' TAMD>', &
                     ' The tree topology has not been properly set up yet.',/, &
                     ' Torsion angle dynamics can been only initiated with proper', &
                     ' tree topology.',/, ' The tree topology setup is finalized', &
                     ' by TREE command.')
             endif
             call wrndie(-3,'<TAMD>', &
                  'Dyanamics requested w/o proper tree')
             continue             ! refuse to do anything
          endif
          !
          ! ... check for IMAGES, SHAKE, CHEQ, IMOVE flags
          if(QSHAKE) call wrndie(-3,'<TAMD>', &
               'SHAKE not allowed in TAMD')
          if(XATIM /= 0) call wrndie(-3,'<TAMD>', &
               'IMAGE not supported in TAMD')
#if KEY_CHEQ==1
          if(QCG) call wrndie(-3,'<TAMD>','QCG not allowed(?) in TAMD')
#endif 
#if KEY_TSM==1
          IF(QTSM) call wrndie(-3,'<TAMD>','TSM not allowed in TAMD')
#endif 
          !
          allocate(freeat(NATOM),stat = alloc_err)
          if(alloc_err /= 0) &
               CALL WrnDie(-1,'<TAMD>', &
               'Failed to allocate memory for freeat array')
          !          
          call mkfrat(imove,natom,freeat,nfreat)
          if(nfreat /= natom) call wrndie(-3,'<TAMD>', &
               'All atoms have to be free in TAMD')        
          !
          ! ... parse the dynamic options
          irest=0
          if(indxa(COMLYN, COMLEN, 'STAR') > 0) irest = 0
          if(indxa(COMLYN, COMLEN, 'STRT') > 0) irest = 0
          if(indxa(COMLYN, COMLEN, 'REST') > 0) irest = 1
          !
          ! ... allocates
          allocate(vhk(3,NClus), tmk(Nclus), Vorigk(3, NClus), &
               VCMk(3,NClus), SpVel(6,NClus), ITk(3,3,NClus), &
               Tk(NClus), Theta(NClus), dTheta(NClus),  &
               ddTheta(NClus), dThetaOld(NClus), dThetaNew(NClus), &
               VX(NATOM), VY(NATOM), VZ(NATOM), &
               XOLD(NATOM), YOLD(NATOM), ZOLD(NATOM), &
               stat = alloc_err)
          if(alloc_err /= 0) &
               CALL WrnDie(-1,'<TAMD>', &
               'Failed to allocate memory for vhk array')

          prnold=prnlev
          if(prnlev == 5) prnlev=4 ! reduce updeci() print
          call tadcntrl(COMLYN,COMLEN,XOLD,YOLD,ZOLD, &
               freeat,nfreat, &
               SpVel,Theta,dTheta, &
               ddTheta,vhk,VOrigk,VCMk, &
               tmk,ITk,Tk,dThetaOld,dThetaNew, &
               VX,VY,VZ) 
          prnlev=prnold
          !     
          !     free
          deallocate(vhk,tmk,VOrigk,VCMk,SpVel,ITk,Tk,Theta, &
               dTheta ,ddTheta,dThetaOld,dThetaNew,VX,VY,VZ, &
               XOLD,YOLD,ZOLD )
          deallocate(freeat)
          !
          ! ... pass commands for CONS COOR ENERgy GETE WRIT
       ELSE IF (WRD == 'CONS') THEN
          CALL CSTRAN(COMLYN,COMLEN)
       ELSE IF (WRD == 'COOR') THEN
          CALL CORCOM(COMLYN,COMLEN)
       ELSE IF (WRD == 'ENER') THEN
          CALL GETE0('ENER', COMLYN, COMLEN)
       ELSE IF (WRD == 'GETE') THEN
          CALL GETE0('GETE', COMLYN, COMLEN)
          !       ELSE IF (WRD == 'READ') THEN
          !          CALL MAINIO(WRD)
       ELSE IF (WRD == 'WRIT') THEN
          CALL MAINIO(WRD)
          !
       else if(wrd == 'END ') then ! exit TAMD module
          done=.true.            
       else 
          if(prnlev >= 3) write(outu,'(2A)') &
               ' TAMD> Unrecognized command: ', WRD          
       endif                    ! if(wrd....)
       !       
    enddo                     ! do while(.not.done)
    !
    if(allocated(islct))deallocate(islct)
    !
9999 if(prnlev >= 1) write(outu,100) 'Exiting TAMD module'
    RETURN
  END      subroutine tamd                 ! of subroutine TAMD()
  !
  !*******************************
  !     tree build facilities    *
  !*******************************
  !
  !======================================================================
  subroutine PreClus(ISLCT)
    !
    !     pre-cluster a group of selected atoms
    !      
    use dimens_fcm
    use stream
    use psf
    use chutil,only:atomid
    !
    integer ISLCT(NATOM), n, nsel
    character(len=8) SIDI,RIDI,RENI,ACI
    !
    nsel=0
    do n=1,NATOM              ! checking
       if(ISLCT(n) /= 0) then
          nsel=nsel+1
          if(ClusID(n) /= 0) then
             call wrndie(0,'<TAMD>', &
                  'ATOM ALREADY ASSIGNED TO A CLUSTER')
             return
          endif
       endif
    enddo
    if(nsel == 0) then
       if(wrnlev >= 0) write(outu,'(A)') &
            ' PreClus: zero atom selection. no atom clustered.'
       return
    endif
    !
    if(NClus > 0) call wrndie(0,'<TAMD>', &
         'Pre-cluster atom while NClus > 0')
    NClus=NClus-1
    if(prnlev >= 2) write(outu,'(A)') &
         ' PreClus: following atoms will be grouped into a cluster'
    do n=1,NATOM
       if(ISLCT(n) /= 0) then
          if(ClusID(n) /= 0) then
             call wrndie(0,'<TAMD>','Attempt to re-cluster atoms')
             return
          else
             ClusID(n)=NClus
             if(prnlev >= 2) then
                call ATOMID(n,SIDI,RIDI,RENI,ACI)
                write(outu,100) n, SIDI(1:idleng), RIDI(1:idleng), &
                     RENI(1:idleng), ACI(1:idleng)
100             format('    ATOM: ',I6,4(1X,A))
             endif
          endif
       endif
    enddo
    return
  end subroutine PreClus
  !
  !======================================================================
  subroutine SetupTree(COMLYN,COMLEN)
    !
    !     Finalize the tree topology setup by calling appropriate automatic
    !     atom clustering subroutines.
    !
    !     v 0.1: handle single chain protein with standard residues (CHARMM 22)
    !     v 0.2: handle CHARMM toph19 (no nonpolar hydrogen)
    !
    use dimens_fcm
    use number
    use stream
    use string
    use psf
    use chutil,only:atomid
    !
    integer COMLEN
    character(len=*) COMLYN
    ! ... local variables
    integer iclus,topver, error!,matom2
    integer j, k, n, nb, nh, no, newhinge, iseg, fstclus, lstclus
    character(len=8) SIDI, RENI, RIDI, ACI
    logical prflag, notdone
    parameter (prflag=.false.) ! print error messages in MATOM2() calls?
    !
    if(TreeReady == 1) then
       call wrndie(2, '<TAMD>', 'Tree topology already setup.')
       return
    endif
    !
    !      if(NSEG > 1) then        ! need better way check for mutiple chains?
    !         call wrndie(-3,'<TAMD>','Multiple chains not supported.')
    !         return
    !      endif
    !
    if(NRES <= 0) then
       call wrndie(0,'<TAMD>','NRES=0. Blank PSF?')
       return
    endif
    !
    !------------------------------------------------------------------
    !     Automatic clustering of rigid fragments.
    !
    !     CHARMM topology version: if HA present, all atom; otherwise, toph19
    !
    error=0
    topver=GTRMI(COMLYN,COMLEN,'TOPV',0)
    j=(nres+1)/2
    if(topver == 0) then
       if (matom2(j,'HA',atype,ibase,prflag) > 0 .or.  &
            matom2(j,'HA1',atype,ibase,prflag) > 0) then
          topver=22
       else
          topver=19
       endif
    endif
    if(topver == 19) then
       call AClus19(COMLYN,COMLEN)
    else if(topver == 22) then
       call AClus22(COMLYN,COMLEN)
    else
       if(prnlev >= 2) write(outu,'(A,I3,A)') ' SetupTree: topver = ', &
            topver, ' version is not supported. Only 19, 22 supported.'
       call wrndie(-3,'<TAMD>','Unsupported Topology Version.')
       return
    endif
    !
    ! ... assign cluster ID for pre-specified clusters that have not been 
    !     incorporated during AClus19/AClus22; adjust all cluster IDs if
    !     necessary for segment partition integrity.
    !
    nh=0
    do n=1,NATOM
       if(ClusID(n) < 0) then ! atom is pre-clustered
          nh=nh+1
          nb=ClusID(n)
          NClus=NClus+1
          if(NClus > MAXCLUS) then
             call wrndie(-1,'<TAMD>','MAXCLUS EXCEEDED')
             NClus=NClus-1
          endif
          ClusID(n)=NClus
          do no=n+1,NATOM
             if(ClusID(no) == nb) ClusID(no)=NClus
          enddo
       endif
    enddo
    if(nh > 0.and.prnlev >= 2) write(outu,'(A,I4,A)') ' SetupTree: ',  &
         nh, ' clusters are added based on previous specifications.'

    !---------------------------------------------------------------------------
    !     double check completeness of atom clustering
    !
    nb=0
    do n=1,NATOM
       if(ClusID(n) == 0) then
          nb=nb+1
          if(prnlev >= 2) then
             if(nb == 1) write(outu,'(/2A,I8)') ' SetupTree: ', &
                  'Following atoms are not assigned to any cluster:'
             call ATOMID(n,SIDI,RIDI,RENI,ACI)
             write(outu,'(12X,I6,4(1X,A))')  n, &
                  SIDI(1:idleng), RIDI(1:idleng), &
                  RENI(1:idleng), ACI(1:idleng)
          endif
       endif
    enddo
    if(nb > 0) then
       if(prnlev >= 2) write(outu,'(2A,I8)') ' SetupTree: ', &
            'total number of unclustered atoms: ', nb
       call wrndie(0,'<TAMD>','Some atoms unrecognized in SetupTree')
       return 
    endif
    !
    if(prnlev >= 2) write(outu,80) NClus
80  format(' SetupTree: atoms are grouped into ',I6,' clusters')     
    !
    !----------------------------------------------------------------------------
    !     establish the connectivity and assign Parent, OrigPt, HingePt
    !
    do k=1,NClus
       Parent(k)=0
       OrigPt(k)=0
       NChild(k)=-1           ! -1: marks unprocessed clusters
       HingePt(k)=0
    enddo
    !
    if(indxa(COMLYN,COMLEN,'BASE') > 0) then 
       !     process selection command and find the cluster to be the base
    endif
    !
    do iseg=1,nseg

       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       !      write(*,*) "#####", iseg, fstclus, lstclus

       Parent(lstclus)=-1          ! mark the last cluster as base
       notdone=.true.
       do while(notdone)
          newhinge=0
          loop88: do k=lstclus,fstclus,-1
             if(Parent(k) /= 0.and.NChild(k) < 0) then ! find all its children
                NChild(k)=0          ! this cluster has been processed
                loop86: do nb=1,NBOND
                   ! ... find the hinges (bonds that connect two clusters)
                   if(ClusID(IB(nb)) == k.and.ClusID(JB(nb)) /= k) then
                      nh=IB(nb)
                      no=JB(nb)
                   else if(ClusID(IB(nb)) /= k.and.ClusID(JB(nb)) == k) then
                      nh=JB(nb)
                      no=IB(nb)
                   else
                      cycle loop86
                   endif
                   !
                   iclus=ClusID(no)
                   if(iclus == Parent(k)) cycle loop86 ! hinge to its parent
                   !
                   ! ... check for conflictions and add connectivity
                   if(Parent(iclus) /= 0) then    ! not a new connectivity
                      if(Parent(iclus) /= k) then
                         if(wrnlev >= 2) write(outu,92) iclus
                         if(prnlev > 5) then
                            write(outu,'(3(A,I6))') 'new child of ',k, &
                                 'th cluster iclus = ', iclus,  &
                                 ' already has parent = ', Parent(iclus)
                            call PrnClus(k)
                            call PrnClus(iclus)
                         endif
                         error=3
                         !                    goto 9994
                      else
                         if(no /= OrigPt(iclus)) then
                            if(wrnlev >= 2) write(outu,94) k,iclus
                            if(prnlev > 5) then
                               call PrnClus(k)
                               call PrnClus(iclus)
                            endif
                            error=3
                            !                       goto 9994
                         else
                            if(wrnlev >= 2) write(outu,96) k,iclus
                            if(prnlev > 5) then
                               call PrnClus(k)
                               call PrnClus(iclus)
                            endif
                         endif
                      endif
                   else
                      newhinge=newhinge+1
                      NChild(k)=NChild(k)+1
                      if(NChild(k) > MAXCHILD) then
                         call wrndie(-3,'<TAMD>', &
                              'MAXCHILD exceeded in SetupTree')
                         return
                      endif
                      Parent(iclus)=k
                      OrigPt(iclus)=no
                      HingePt(iclus)=nh
                      Child(NChild(k),k)=iclus
                   endif
                enddo loop86
             endif
          enddo loop88
          !
          notdone=newhinge /= 0    ! otherwise all connectivity found, thus done
          !
       enddo                     ! do while(notdone)
       !
       ! ... set OrigPt() and HingePt() of base cluster
       HingePt(lstclus)=-1       ! hinge point of base is origin (0,0,0)
       if(NChild(lstclus) > 0) then
          OrigPt(lstclus)=HingePt(Child(1,lstclus)) 
       else                      ! single cluster, OrigPt set to 1st atom
          do k=1,NATOM
             if(ClusID(k) == lstclus) then
                OrigPt(lstclus)=k
                exit
             endif
          enddo
       endif
       !
       do k=fstclus,lstclus
          if(Parent(k) == 0) then
             error=4
             if(wrnlev >= 2) write(outu,98) k, segid(iseg)
             if(prnlev > 5) then
                call PrnClus(k)
             endif
          endif
       enddo

    enddo                     ! do iseg=1, nseg
    !
90  format(' SetupTree: multiple hinge points found for cluster: ',I6)
92  format(' SetupTree: multiple parents found for cluster: ',I6)
94  format(' SetupTree: multiple hinges found between clusters:', &
         2(1X,I6))
96  format(' SetupTree: multiple bonds found between clusters:  ', &
         2(1X,I6))
98  format(' SetupTree: segment: ',A,' cluster: ',I6, &
         ' is not covalently connected!')
    !
    if(error == 0) then
       TreeReady=1
       if(prnlev >= 2) write(outu,'(/,A,/)') &
            ' SetupTree: the connectivity has been successfully established.'
    else if(error == 3) then
       !
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       ! ... Error code 3: Errors while searching connectivity between clusters.
9994   TreeReady=0
       if(prnlev >= 2) write(outu,'(A)') &
            ' SetupTree: Errors in connectivity between clusters.'
       call wrndie(0,'<TAMD>','ERROR in SetupTree: error code=3')
    else if(error == 4) then
       !
       ! ... Error code 4: Unable to establish connectivity between clusters.
9996   TreeReady=0
       call wrndie(0,'<TAMD>','ERROR in SetupTree: error code=4')
    endif
    !___________________________________________________________________________
    !
    ! ... print tree
    if(prnlev > 5) call PrintTree()
    !
    return
  end subroutine SetupTree
  !
  ! ... print out the details of a tree
  subroutine PrintTree()
    use dimens_fcm
    use number
    use stream
    use psf
    use chutil,only:atomid
    !
    integer k, n, j
    character(len=8) SIDI, RENI, RIDI, ACI
    !
    write(outu,200) NSeg, NClus, NATOM
    do k=1, NClus
       if(NChild(k) > 0) then
          write(outu,210) k, Parent(k), NChild(k), &
               (Child(j,k),j=1,NChild(k))
       else
          write(outu,212) k, Parent(k)
       endif
       do n=1,NATOM
          if(ClusID(n) == k) then
             call ATOMID(n,SIDI,RIDI,RENI,ACI)
             write(outu,220,advance='no') '   Atom: ',n, &
                  SIDI(1:idleng),RIDI(1:idleng), &
                  RENI(1:idleng),ACI(1:idleng)
             if(n == OrigPt(k)) write(outu,'(A)',advance='no') ' OrigPt '
             do j=1,NChild(k)
                if(n == HingePt(Child(j,k)))  &
                     write(outu,'(A,I1)',advance='no')' hPt',j
             enddo
             write(outu,'(A)')
          endif
       enddo
    enddo
200 format(/,3X, &
         '++++++++++++++++++ Details of The Tree ++++++++++++++++++',/, &
         3X,' NSeg = ', I4, '  NClus = ', I5, '  NATOM = ', I6)
210 format(3X,'---------------------------------------------',/, &
         3X,' Cluster NO. ', I6,/ &
         3X,'   Parent =', I6, '  Child(1:',I1,') =',8(1X,I6))
212 format(3X,'---------------------------------------------',/, &
         3X,' Cluster No. ', I6,/ &
         3X,'   Parent =', I6, '  No Child')
220 format(6X,A,I6,4(1X,A))
    !
    return
  end subroutine PrintTree
  !
  ! Print all atoms in a cluster
  subroutine PrnClus(iclus)
    use dimens_fcm
    use number
    use stream
    use psf
    use chutil,only:atomid
    integer iclus,n,j
    character(len=8) SIDI, RENI, RIDI, ACI
    write(outu,'("Cluster No. ",I5," Parrent =",I5, " NChild =",I4)')  &
         iclus, Parent(iclus), NChild(iclus)
    do n=1,NATOM
       if(ClusID(n) == iclus) then
          call ATOMID(n,SIDI,RIDI,RENI,ACI)
          write(outu,220,advance='no') '   Atom: ',n, &
               SIDI(1:idleng),RIDI(1:idleng), &
               RENI(1:idleng),ACI(1:idleng)
          if(n == OrigPt(iclus)) write(outu,'(A)',advance='no') ' OrigPt '
          do j=1,NChild(iclus)
             if(n == HingePt(Child(j,iclus)))  &
                  write(outu,'(A,I1)',advance='no')' hPt',j
          enddo
          write(outu,'(A)')
       endif
    enddo
220 format(6X,A,I6,4(1X,A))
    return
  end    subroutine PrnClus
  !======================================================================
  subroutine AClus19(COMLYN,COMLEN)
    !
    !     Automatic atom clustering for a single chain protein with standard 
    !     residues as defined in CHARMM toph19.inp
    !
    use dimens_fcm
    use number
    use stream
    use psf
    use chutil,only:getres,getseg
    !
    integer COMLEN
    character(len=*) COMLYN
    ! ... local variables
    integer ires,iseg,iclus,inre,npre,fstres,lstres,err,n,nn,tmpid
    character(len=4) NTER, CTER
    character(len=8) SIDI, RENI, RIDI, ACI
    logical prflag,stdres
    parameter (prflag=.false.) ! print error messages in MATOM2() calls?
    integer MAXATOM, ncgrp
    parameter (MAXATOM=20)    ! this number should be sufficiently large for
    integer atid(MAXATOM)     ! all clusterable groups of atoms 
    !
    if(TreeReady == 1) then
       call wrndie(2, '<TAMD>', 'Tree topology already setup.')
       return
    endif
    !
    if(NRES <= 0) then
       call wrndie(0,'<TAMD>','NRES=0. Blank PSF?')
       return
    endif
    !
    !---------------------------------------------------------------------
    !     group all atoms in PSF into clusters
    !
    if(NClus < 0) then
       npre=-NClus
       if(prnlev >= 2) write(outu,'(A,I6)')  &
            ' AClus19: number of user specified clusters is: ', npre
    endif
    NClus=0
    !
    !------------------------------------------------------------------
    !     Go over each segments
    do iseg=1,nseg
       !      
       fstres=nictot(iseg)+1
       lstres=nictot(iseg+1)
       !
       NCSeg(iseg)=NClus
       !__________________________________________________________________
       !     C-terminus patch identification: CTER, CBXP (defined in MMTSB)
       !
       CTER=' '
       if (MATOM2(lstres,'NCB',atype,ibase,prflag) > 0) then
          CTER='CBXP'
       else if (MATOM2(lstres,'OT1',atype,ibase,prflag) > 0) then
          CTER='CTER'
       else 
          if(prnlev >= 2) write(outu,'(A,/,A)')  &
               ' AClus19: cluster commands should be used to define cluster', &
               '          of atoms for non-standard C-terminus patches'
          call wrndie(2,'<TAMD>','No C-terminus patch recognized')
       endif
       !___________________________________________________________________
       ! ... N-terminus patch identification: NTER/GLYP/PROP, AMNP(defined MMTSB)
       !
       NTER=' '
       if (MATOM2(fstres,'HT1 ',atype,ibase,prflag) > 0) then
          if(res(fstres) == 'GLY ') then ! GLYP
             NTER='GLYP'
          else if(res(fstres) == 'PRO') then
             NTER='PROP'
          else
             NTER='NTER'
          endif
       else if (MATOM2(fstres,'CLM ',atype,ibase,prflag) > 0) then
          NTER='AMNP'
       else
          if(prnlev >= 2) write(outu,'(A,/,A)')  &
               ' AClus19: cluster commands should be used to define cluster', &
               '          of atoms for non-standard N-terminus patches'
          call wrndie(2,'<TAMD>','No N-terminus patch recognized')
       endif
       !_____________________________________________________________________
       !     N-Terminus
       !
       RENI=NTER
       RIDI=RESID(fstres)
       if(NTER == 'NTER'.or.NTER.eq.'GLYP') then
          ! ... -NH3
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'HT1 ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'HT2 ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'HT3 ',fstres,atype,ibase,prflag)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
       else if(NTER == 'PROP') then
          ! ... everything is rigid
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'HT1 ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'HT2 ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'CB  ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'CD  ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'CG  ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'CA  ',fstres,atype,ibase,prflag)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
       else if(NTER == 'AMNP') then
          ! ... C-CO-NH
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'CLM ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'CAM ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'OAM ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prflag)) err=err+1
          if(addatom(atid,ncgrp,'H   ',fstres,atype,ibase,prflag)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
       endif
       !
       do ires=fstres,lstres
          RENI=RES(ires)
          RIDI=RESID(ires)

          stdres=.true.
          inre=ires+1
          !_____________________________________________________________________
          !     side-chains
          !
          if(res(ires) == 'ACE'.or.res(ires).eq.'AMN') then
             ! ... C-CO-NH
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CH3 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'H   ',inre,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'ALA') then
             !
          else if(res(ires) == 'ARG') then
             ! ... C(Z)NH2NH2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NH1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HH11',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HH12',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NH2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HH21',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HH22',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... N(E)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'NE  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(D)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'ASN') then
             ! ... C(G)O-N(D2)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'ND2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HD21',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HD22',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'ASP') then
             ! ... C(G)O2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OD2 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'CYS') then
             ! ... C(B)-S(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'SG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'GLN') then
             ! ... C(D)O-N(E)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE21',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE22',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !        
          else if(res(ires) == 'GLU') then
             ! ... C(D)O2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'OE2 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !        
          else if(res(ires) == 'GLY') then
             ! ... no sidechain
          else if(res(ires) == 'HIS') then
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !        
          else if(res(ires) == 'HSD') then ! neutral HIS, H on NE2 (instead of ND1)
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'HSC') then ! similar to HSP in param22 
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'ILE') then
             ! ... C(G1)-C(D)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CG1 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)-C(G2)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'LEU') then
             ! ... C(G1)-C(D1)C(D2)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'LYS') then
             ! ... N(Z)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'NZ  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HZ1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HZ2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HZ3 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(E)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CE  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(D)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'MET') then
             ! ... S(D)-C(E)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CE  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'SD  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(G)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'PEN') then
             ! ... C(B)-S(G)C(G1)C(G2)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'SG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
          else if(res(ires) == 'PHE') then
             ! ... benzene ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'PRO') then 
             ! ... all as a rigid body: clustered in backbone proceessing section
             !
          else if(res(ires) == 'SER') then
             ! ... O(G)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HG  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'THR') then
             ! ... O(G)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OG1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)-C(G2)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'TRP') then
             ! ... rings
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'NE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE3 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CZ3 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CH2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CZ2 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI              
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'TYR') then
             ! ... O(H)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OH  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'HH  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             ! ... C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
          else if(res(ires) == 'VAL') then
             ! ... C(B)-C(G1)C(G2)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG1 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             !
             !      else if(res(ires) == 'HEME') then
          else 
             stdres=.false.
             if(wrnlev >= 2) write(outu,'(A,A4,A)') &
                  ' AClus19: unrecognized residue: ', res(ires), &
                  ' OK if all members pre-clustered by CLUS commands'
          endif
          !_____________________________________________________________________
          !     backbone: (including C-Terminus)
          !
          if(res(ires) == 'ALA') then ! CA
             ! ... C(A)-C(B)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CA  ',ires,atype,ibase,prflag)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI    
          else if(res(ires) /= 'PRO'.and.res(ires).ne.'ACE' &
               .and.res(ires) /= 'AMN'.and.res(ires).ne.'CBX' ) then
             ! ... -C(A)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CA  ',ires,atype,ibase,prflag)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI    
          endif
          !
          if(ires < lstres) then    ! peptide bond
             if(res(inre) == 'PRO') then
                ! ... everything as a rigid body
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CB  ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CD  ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CG  ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CA  ',inre,atype,ibase,prflag)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI       
             else if(res(inre) == 'CBX') then
                ! ... CO-NH-CA
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'H   ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CA  ',inre,atype,ibase,prflag)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             else if(stdres) then
                ! ... CO-NH
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'H   ',inre,atype,ibase,prflag)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             endif
             !
          else                     ! C-Terminus
             !
             RENI=CTER
             if(CTER == 'CTER') then
                ! ... CO2
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'OT1 ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'OT2 ',ires,atype,ibase,prflag)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI      
             else if(CTER == 'CBXP') then
                ! ... CO-NHCA
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'NCB ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'HCB ',ires,atype,ibase,prflag)) err=err+1
                if(addatom(atid,ncgrp,'CAC ',ires,atype,ibase,prflag)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             endif
             ! 
          endif                     ! if(ires < NRES)
          !            
       enddo                     ! enddo of do ires=fstres,lstres
       !
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       !     ERROR MODES:
       !
       ! ... Error code 1: some members of a group of atoms are either not
       !                   found or redundant.
       !
       ! ... Error code 2: members of the group of atoms are previously 
       !                   assigned to different clusters.
100    format( &
            ' AClus22: some members of the same group are pre-assigned',/, &
            '          to different clusters in resude: ',A,A)
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       !
       ! ... assign cluster ID for pre-specified clusters that have not been 
       !     identified above
       !
       ncgrp=0
       do n=1,NATOM
          if(ClusID(n) < 0) then ! atom is pre-clustered
             ires=GETRES(n,IBASE,NREST)
             if(GETSEG(ires,NICTOT,NSEGT) == iseg) then ! within the same segment
                ncgrp=ncgrp+1
                tmpid=ClusID(n)
                NClus=NClus+1
                if(NClus > MAXCLUS) then
                   call wrndie(-1,'<TAMD>','MAXCLUS EXCEEDED')
                   NClus=NClus-1
                endif
                ClusID(n)=NClus
                do nn=n+1,NATOM
                   if(ClusID(nn) == tmpid) ClusID(nn)=NClus
                enddo
             endif
          endif
       enddo
       if(ncgrp > 0.and.prnlev >= 2) write(outu,'(A,I4,A,A)')  &
            ' AClus22: ',ncgrp, &
            ' remaining predefined clusters added to segment ',segid(iseg)
       !
    enddo                     ! enddo of: do iseg=1,nseg
    NCSeg(nseg+1)=NClus
    !
    !
    ! ... successful return
    return
    !
  end subroutine AClus19
  !
  !======================================================================
  subroutine AClus22(COMLYN,COMLEN)
    !
    !     Automatic atom clustering for a single chain protein with 
    !     standard residues as defined in CHARMM top_all22_prot.inp
    !
    use dimens_fcm
    use number
    use stream
    use psf
    use chutil,only:getres,getseg
    !
    integer COMLEN
    character(len=*) COMLYN
    ! ... local variables
    integer ires,iseg,inre,npre,fstres,lstres,tmpid,n,nn,err
    character(len=4) NTER, CTER
    character(len=8) SIDI, RENI, RIDI, ACI
    logical prf,stdres               
    parameter (prf=.false.)   ! print error messages in MATOM2() calls?
    integer MAXATOM, ncgrp
    parameter (MAXATOM=20)    ! this number is sufficient for all
    integer atid(MAXATOM)     ! foreseeable clusterable groups of atoms 
    !
    if(TreeReady == 1) then
       call wrndie(2, '<TAMD>', 'Tree topology already setup.')
       return
    endif
    !
    if(NRES <= 0) then
       call wrndie(0,'<TAMD>','NRES=0. Blank PSF?')
       return
    endif
    !
    !---------------------------------------------------------------------
    !     group all atoms in PSF into clusters
    !
    if(NClus < 0) then
       npre=-NClus
       if(prnlev >= 2) write(outu,'(A,I6)')  &
            ' AClus22: number of user specified clusters is: ', npre
    endif
    NClus=0
    !
    !------------------------------------------------------------------
    !     Go over each segments
    do iseg=1,nseg
       !      
       fstres=nictot(iseg)+1
       lstres=nictot(iseg+1)
       !
       NCSeg(iseg)=NClus
       !__________________________________________________________________
       !     C-terminus identification
       !
       CTER=' '
       if (MATOM2(lstres,'CAT',atype,ibase,prf) > 0) then
          CTER='CT3'
       else if (MATOM2(lstres,'CT',atype,ibase,prf) > 0) then
          CTER='CT1'
       else if (MATOM2(lstres,'NT',atype,ibase,prf) > 0) then
          CTER='CT2'
       else if (MATOM2(lstres,'OT1',atype,ibase,prf) > 0) then
          CTER='CTER'
       else if (MATOM2(lstres,'OCT1',atype,ibase,prf) > 0) then ! lpdb CTER
          CTER='CTE2'
       else 
          if(prnlev >= 2) write(outu,'(A,A)')  &
               ' AClus22: pre-cluster necessary for non-standard C-ter: ', &
               segid(iseg)
          call wrndie(2,'<TAMD>','Unrecognized C-terminus')
       endif
       !___________________________________________________________________
       ! ... N-terminus: by looking at HT1, HY1, HN1 to tell the four(five) 
       !     possible N-terimus patchs
       !
       NTER=' '
       if (MATOM2(fstres,'HT1 ',atype,ibase,prf) > 0) then
          if(res(fstres) == 'GLY ') then ! GLYP
             NTER='GLYP'
          else if(res(fstres) == 'PRO ') then ! PROP with HT1/HT2
             NTER='PROP'
          else
             NTER='NTER'
          endif
       endif
       if (MATOM2(fstres,'HY1 ',atype,ibase,prf) > 0) then
          if(NTER == ' '.or.(lstres.eq.fstres.and.CTER /= 'CTER')) then
             NTER='ACE '         ! or ACP
          else                   ! test positive for multiple markers
             call wrndie(fstres,'<TAMD>','Unrecognized N-terminus')
          endif
       endif
       if (MATOM2(fstres,'HN1 ',atype,ibase,prf) > 0) then
          if(NTER == ' ') then
             NTER='PROP'
          else                   ! test positive for multiple markers
             call wrndie(fstres,'<TAMD>','Ambiguous N-terminus')
          endif
       endif
       if(NTER == ' ') then
          if(prnlev >= 2) write(outu,'(A,A,A)')   &
               ' AClus22: pre-cluster necessary for non-standard N-ter: ', &
               segid(iseg)
          call wrndie(2,'<TAMD>','Unrecognized N-terminus')
       endif

       if(prnlev > 3) then
          write(outu,*) "Segment: ", segid(iseg), ' NTER= ', NTER, &
               ' CTER= ', CTER
          write(outu,*)
       endif
       !_____________________________________________________________________
       !     N-Terminus
       !
       RENI=NTER
       RIDI=RESID(fstres)
       if(NTER == 'NTER'.or.NTER.eq.'GLYP') then
          ! ... -NH3
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HT1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HT2 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HT3 ',fstres,atype,ibase,prf)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
       else if(NTER == 'PROP') then
          ! ... everything is rigid
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prf)) err=err+1
          ! ... PROP with HN1/2 or HT1/2
          if(addatom(atid,ncgrp,'HN1 ',fstres,atype,ibase,prf).and. &
               addatom(atid,ncgrp,'HT1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HN2 ',fstres,atype,ibase,prf).and. &
               addatom(atid,ncgrp,'HT2 ',fstres,atype,ibase,prf)) err=err+1
          !
          if(addatom(atid,ncgrp,'CB  ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HB1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HB2 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'CD  ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HD1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HD2 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'CG  ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HG1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HG2 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'CA  ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HA  ',fstres,atype,ibase,prf)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
       else if(NTER == 'ACE') then
          ! ... CH3
          ncgrp=0
          err=0
          if(addatom(atid,ncgrp,'CAY ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HY1 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HY2 ',fstres,atype,ibase,prf)) err=err+1
          if(addatom(atid,ncgrp,'HY3 ',fstres,atype,ibase,prf)) err=err+1
          if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
               .and.prnlev >= 2) write(outu,100) RIDI,RENI
          !
          if(res(fstres) == 'PRO') then 
             ! ... ACP: everything is rigid for now
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CY  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OY  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CB  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CG  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CA  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HA  ',fstres,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
          else
             ! ... CO-NH
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CY  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OY  ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'N   ',fstres,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HN  ',fstres,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
          endif
       endif
       !
       do ires=fstres,lstres
          RENI=RES(ires)
          RIDI=RESID(ires)

          stdres=.true.  ! assuming it is a standard residue
          inre=ires+1       
          !_____________________________________________________________________
          !     side-chains
          !
          if(res(ires) == 'ALA') then
             ! ... C(B)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB3 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'ARG'.or.res(ires).eq.'ARGN') then
             ! ... C(Z)NH2NH2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NH1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HH11',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HH12',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NH2 ',ires,atype,ibase,prf)) err=err+1
             ! ... ARGN only has HH21
             if(addatom(atid,ncgrp,'HH21',ires,atype,ibase,prf).or. &
                  addatom(atid,ncgrp,'HH22',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... N(E)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'NE  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(D)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'ASN') then
             ! ... C(G)O-N(D2)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ND2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD22',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'ASP'.or.res(ires).eq.'ASPH'.or. &
               res(ires) == 'ASPA'.or.res(ires).eq.'ASPB') then
             ! ... C(G)O2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OD2 ',ires,atype,ibase,prf)) err=err+1
             ! ... protonated ASP (ASPH/ASPA: od1-hd; ASPB: od2-hd)
             if(MATOM2(ires,'HD  ',atype,ibase,prf) > 0.and. &
                  addatom(atid,ncgrp,'HD  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'CYS') then
             ! ... S(G)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'SG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'HG  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'CYSS') then
             ! ... C(B)H2-S
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'SG  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'CYSZ') then
             ! ... S(G)-Zn
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'SG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ZN  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'GLN') then
             ! ... C(D)O-N(E)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE22',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !        
          else if(res(ires) == 'GLU'.or.res(ires).eq.'GLUH') then
             ! ... C(D)O2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'OE2 ',ires,atype,ibase,prf)) err=err+1
             if(MATOM2(ires,'HE  ',atype,ibase,prf) > 0.and. &
                  addatom(atid,ncgrp,'HE  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !        
          else if(res(ires) == 'GLY') then
             ! ... no sidechain
          else if(res(ires) == 'HIS') then  ! HIS is same as HSD
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !        
          else if(res(ires) == 'HSD') then
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !        
          else if(res(ires) == 'HSE') then
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'HSP') then
             ! ... 5-member ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'ND1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'ILE') then
             ! ... C(D)HD3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'HD11',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'HD12',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD3 ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'HD13',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G1)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG11',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG12',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G2)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG22',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG23',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'LEU') then
             ! ... C(D1)HD3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD11',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD12',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD13',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(D2)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD22',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD23',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G1)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'LYS'.or.res(ires).eq.'LYSN') then
             ! ... N(Z)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'NZ  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HZ1 ',ires,atype,ibase,prf)) err=err+1
             ! ... protonated LYSN only has HZ2
             if(addatom(atid,ncgrp,'HZ2 ',ires,atype,ibase,prf).or. &
                  addatom(atid,ncgrp,'HZ3 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(E)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CE  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(D)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CD  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'MET') then
             ! ... C(E)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CE  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE3 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... S(D)
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'SD  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'PHE') then
             ! ... benzene ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HZ  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'PRO') then 
             ! ... all as a rigid body: clustered in backbone proceessing section
             !
          else if(res(ires) == 'SER') then
             ! ... O(G)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf).and. &
                  addatom(atid,ncgrp,'HG  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'THR') then
             ! ... O(G)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG1 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G2)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG22',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG23',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'TRP') then
             ! ... rings
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'NE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE3 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE3 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CZ3 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HZ3 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CH2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HH2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CZ2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HZ2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'TYR') then
             ! ... O(H)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'OH  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HH  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... ring
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CZ  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HE2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'CD2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HD2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
          else if(res(ires) == 'VAL') then
             ! ... C(G1)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG11',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG12',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG13',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(G2)H3
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CG2 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG21',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG22',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HG23',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             ! ... C(B)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CB  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HB  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
             !
             !      else if(res(ires) == 'HEME') then
          else 
             stdres=.false.
             if(wrnlev >= 2) write(outu,'(A,A4,/,9X,A)') &
                  ' AClus22: unrecognized residue: ', res(ires), &
                  ' OK if all members pre-clustered by CLUSter commands.'
          endif
          !_____________________________________________________________________
          !     backbone: (including C-Terminus)
          !
          if(res(ires) == 'GLY') then ! CA
             ! ... -C(A)H2
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CA  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HA1 ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HA2 ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
          else if(stdres) then
             ! ... C(A)H
             ncgrp=0
             err=0
             if(addatom(atid,ncgrp,'CA  ',ires,atype,ibase,prf)) err=err+1
             if(addatom(atid,ncgrp,'HA  ',ires,atype,ibase,prf)) err=err+1
             if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                  .and.prnlev >= 2) write(outu,100) RIDI,RENI
          endif
          ! 
          if(ires < lstres) then    ! peptide bond
             if(res(inre) == 'PRO') then
                ! ... everything as a rigid body
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'CB  ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HB1 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HB2 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'CD  ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HD1 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HD2 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'CG  ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HG1 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HG2 ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'CA  ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HA  ',inre,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             else if(stdres) then
                ! ... CO-NH
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'N   ',inre,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HN  ',inre,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             endif
             !
          else                     ! C-Terminus
             !
             RENI=CTER
             if(CTER == 'CTER') then
                ! ... CO2
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OT1 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OT2 ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             else if(CTER == 'CTE2') then
                ! ... CO2
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OCT1',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OCT2',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             else if(CTER == 'CT1') then
                ! ... CO2
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OT1 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'OT2 ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
                ! ... C(T)H3
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'CT  ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT1 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT2 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT3 ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             else if(CTER == 'CT2') then
                ! ... CO-(T)H2
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'NT  ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT1 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT2 ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             else if(CTER == 'CT3') then
                ! ... CO-N(T)H
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'C   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'O   ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'NT  ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HNT ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
                ! ... C(AT)H3
                ncgrp=0
                err=0
                if(addatom(atid,ncgrp,'CAT ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT1 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT2 ',ires,atype,ibase,prf)) err=err+1
                if(addatom(atid,ncgrp,'HT3 ',ires,atype,ibase,prf)) err=err+1
                if(err == 0 .and. AddClus(atid, ncgrp, NATOM) &
                     .and.prnlev >= 2) write(outu,100) RIDI,RENI
             endif
             ! 
          endif                     ! if(ires < NRES)
          !            
       enddo                     ! enddo of: do ires=fstres,lstres
       !
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       !     ERROR MODES:
       !
       ! ... Error code 1: some members of a group of atoms are either not
       !                   found or redundant.
       !
       ! ... Error code 2: members of the group of atoms are previously 
       !                   assigned to different clusters.
100    format( &
            ' AClus22: some members of the same group are pre-assigned',/, &
            '          to different clusters in resude: ',A,A)
       !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
       !
       ! ... assign cluster ID for pre-specified clusters that have not been 
       !     identified above
       !
       ncgrp=0
       do n=1,NATOM
          if(ClusID(n) < 0) then ! atom is pre-clustered
             ires=GETRES(n,IBASE,NREST)
             if(GETSEG(ires,NICTOT,NSEGT) == iseg) then ! within the same segment
                ncgrp=ncgrp+1
                tmpid=ClusID(n)
                NClus=NClus+1
                if(NClus > MAXCLUS) then
                   call wrndie(-1,'<TAMD>','MAXCLUS EXCEEDED')
                   NClus=NClus-1
                endif
                ClusID(n)=NClus
                do nn=n+1,NATOM
                   if(ClusID(nn) == tmpid) ClusID(nn)=NClus
                enddo
             endif
          endif
       enddo
       if(ncgrp > 0.and.prnlev >= 2) write(outu,'(A,I4,A,A)')  &
            ' AClus22: ',ncgrp, &
            ' remaining predefined clusters added to segment ',segid(iseg)
       !
    enddo                     ! enddo of: do iseg=1,nseg
    NCSeg(nseg+1)=NClus
    !
    ! ... successful return
    return
    !
  end subroutine AClus22
  !
  !======================================================================
  logical function addatom(atomid,ncgrp,ATOM,ires,atype,ibase,prflag)
    !
    !     Add an atom to an atomic ID array, which is to be used by AddClus()
    !     to add specified atoms as a new cluster to the tree.
    !
    !     On return,
    !       addtom  = .true. if unsuccessful and wrnlev >= 3 
    !               = .false. if succesful (no error) or wrnlev < 3
    !
    !     ( return a ".false." value even in case of error will permit the
    !       calling subroutine to continue. It might be necessary for 
    !       non-standard residues such with truncations or patches. )
    !
    use stream
    integer atomid(*), ncgrp, ires, ibase(*),  id, i !, matom2
    character(len=*) ATOM, atype(*)
    logical prflag,wrnflag
    !
    wrnflag = wrnlev >= 3
    !
    if(ncgrp < 0) ncgrp=0
    id=matom2(ires,ATOM,atype,ibase,prflag)
    if(id <= 0) then
       addatom=wrnflag
       if(prnlev >= 5) write(outu,'(A,I4,1X,2A)') &
            ' AddAtom: atom ', ires, ATOM,  &
            ' is not found. nothing added.'
       return
    else
       do i=1,ncgrp
          if(atomid(i) == id) then
             addatom=wrnflag
             if(prnlev >= 5) write(outu,'(A,I4,1X,2A)') &
                  ' AddAtom:  atom ', ires, ATOM, &
                  ' is already in the array. nothing added.'
             return
          endif
       enddo
       !
       ! ... no error, now add the atom to the array
       addatom=.false.
       ncgrp=ncgrp+1
       atomid(ncgrp)=id
       return
       !
    endif
  end function addatom
  !
  !======================================================================
  logical function AddClus(atomid, ncgrp, NATOM)
    !
    !     cluster a group of atoms, On return, 
    !     AddClus = = .true. if unsuccessful and wrnlev >= 3 
    !               = .false. if succesful (no error) or wrnlev < 3
    !
    !     ( return a ".false." value even in case of error will permit the
    !       calling subroutine to continue )
    !      
    use stream
    integer atomid(*), ncgrp, NATOM, i, n, id
    logical preclus,wrnflag
    !
    wrnflag = wrnlev >= 3
    !
    if (NClus < 0 .or. ncgrp < 1) then
       if(prnlev >= 3) write(outu,'(A)') &
            ' AddClus: invalid NClus and/or ncgrp. nothing done'
       AddClus=wrnflag
       return
    endif
    !
    preclus=.false.
    do i=1,ncgrp
       if(ClusID(atomid(i)) /= 0) preclus=.true.
    enddo
    if(preclus) then          ! atoms are pre-clustered
       ! ... make sure that all atoms in atomid have the same cluster ID
       do i=2,ncgrp
          if(ClusID(atomid(i)) /= ClusID(atomid(1))) then
             AddClus=.true.
             call wrndie(0,'<TAMD>','ERROR in AddClus: error code=2')
             return
          endif
       enddo
    endif
    !
    id=ClusID(atomid(1))
    if(id == 0) then          ! add new cluster         
       NClus=NClus+1
       if(NClus > MAXCLUS) then
          call wrndie(0,'<TAMD>','MAXCLUS EXCEEDED')
          NClus=NClus-1
          AddClus=.true.
          return
       endif
       do i=1,ncgrp
          ClusID(atomid(i))=NClus
       enddo
    else if(id < 0) then     ! add pre-defined cluster
       NClus=NClus+1
       if(NClus > MAXCLUS) then
          call wrndie(0,'<TAMD>','MAXCLUS EXCEEDED')
          NClus=NClus-1
          AddClus=.true.
          return
       endif
       do n=1,NATOM
          if(ClusID(n) == id) ClusID(n)=NClus
       enddo
    endif                     ! id>0: nothing to be done
    AddClus=.false.           ! no error
    return
    !
    return
  end function AddClus
  !
  !======================================================================
  INTEGER FUNCTION MATOM2(IRES,ATOM,ATYPE,IBASE,LW)
    !
    !     A simplified MATOM() w/o the fancy stuffs.
    !
    use stream
    LOGICAL LW
    CHARACTER(len=*) ATYPE(*), ATOM
    INTEGER I, IRES, IBASE(*)
    DO I=IBASE(IRES)+1,IBASE(IRES+1)
       IF(ATOM == ATYPE(I)) THEN
          MATOM2=I
          RETURN
       ENDIF
    ENDDO
    IF(LW .AND. WRNLEV >= 2) WRITE(OUTU,13) ATOM(1:idleng),IRES
13  FORMAT(' *** IN MATOM *** ''',A,''' ATOM TYPE NOT FOUND FOR ', &
         'RESIDUE',I5)
    MATOM2=-99999
    RETURN
  END function matom2
  !
  !======================================================================
  subroutine CheckTree(tmk)
    !
    !     Check the integrity of the tree topology.
    !
    use dimens_fcm
    use number
    use stream
    use psf
    !
    integer k, n, nc, iseg, fstclus, lstclus
    real(chm_real)  tmk(NClus)
    logical flag
    !
    if(TreeReady /= 1) then
       if(prnlev >= 3) write(outu,'(A)')  &
            ' The Tree toplogy is not ready yet.'
       return
    endif
    !
10  format(' CheckTree: ', &
         'Total number of child clusters is not NClus-NSeg')
15  format(' CheckTree: ', &
         'Segment: ',A, ' has invalid cluster partition')
20  format(' CheckTree: ', &
         I5,' th cluster has inconsistent cluster IDs for', &
         ' origin and/or hinge atoms.')
30  format(' CheckTree: ', &
         I5,' th cluster has null child: Child(',I1,')= ',I5)
40  format(' CheckTree: ', &
         I6,' th atom has invalid cluster ID = ', I5,  &
         ' ( totol number of clusters is ', I5, ')')
50  format(' CheckTree: ', &
         I5,' th child of ',I5,' th cluster has an inconsistent', &
         ' parent ID of ', I5, '.')
60  format(' CheckTree: ', 'Segment: ', A, &
         ' last cluster is not the root. Its parent ID is ', I5)
70  format(' <CheckTree: ', &
         I5," th cluster's parent is itself, which is not allowed.")
80  format(' CheckTree: ', &
         I5,' th cluster has inconsistent number of children with', &
         ' Parent ID array.')
100 format(' CheckTree: ', &
         I5,' th cluster has zero total mass.')
    !
    do n=1,NATOM
       if(ClusID(n) <= 0.or.ClusID(n) > NClus) then
          TreeReady=0
          if(prnlev >= 3) write(outu,40) n, ClusID(n), NClus
       endif
    enddo
    n=0
    do k=1,NClus
       n=n+NChild(k)
    enddo
    if(n /= NClus-NSeg) then
       TreeReady=0
       if(prnlev >= 3) write(outu,10)
    endif
    !
    do iseg=1,nseg
       fstclus=ncseg(iseg)+1
       lstclus=ncseg(iseg+1)
       if(lstclus < fstclus) write(outu,15) segid(iseg)
       do k=fstclus,lstclus
          flag=.false.
          if(ClusID(OrigPt(k)) /= k) flag=.true.
          if (k /= lstclus) then
             if (ClusID(HingePt(k)) /= Parent(k)) flag=.true.
          endif
          if(flag) then
             TreeReady=0
             if(prnlev >= 3) write(outu,20) k
          endif
          do n=1,NChild(k)
             if(Child(n,k) <= 0) then
                TreeReady=0
                if(prnlev >= 3) write(outu,30) k, n, Child(n,k)
             endif
          enddo
          do n=1,NChild(k)
             if(Parent(Child(n,k)) /= k) then
                TreeReady=0
                if(prnlev >= 3) write(outu,50) n,k,Parent(Child(n,k))
             endif
          enddo
          if(Parent(k) == k) then
             TreeReady=0
             if(prnlev >= 3) write(outu,70) k
          endif
          nc=0
          do n=1,NClus
             if(Parent(n) == k) nc=nc+1
          enddo
          if(nc /= NChild(k)) then
             TreeReady=0
             if(prnlev >= 3) write(outu,80) k
          endif
       enddo
       if(Parent(lstclus) /= -1) then
          TreeReady=0
          if(prnlev >= 3) write(outu,60) segid(iseg),Parent(lstclus)
       endif
    enddo
    !
    do k=1,NClus
       tmk(k)=0d0
    enddo
    do n=1,NATOM
       tmk(ClusID(n))=tmk(ClusID(n))+AMASS(n)
    enddo
    do k=1, NClus
       if(tmk(k) <= 0d0) then 
          TreeReady=0
          if(prnlev >= 3) write(outu,100) k
       endif
    enddo
    !
    if(prnlev >= 3) then
       if(TreeReady == 1) then
          write(outu,'(A)')  &
               ' CheckTree: The tree topology is self-consistent.'
       else
          write(outu,'(A)')  &
               ' CheckTree: The tree topology contains errors.'
       endif
    endif
    return
  end subroutine CheckTree
  !
  !*********************************************************************
  !     Torsion Angle Dynamics and Energy Minimization Subroutines     *
  !*********************************************************************
  !
  !=======================================================================
  subroutine tadmini(COMLYN,COMLEN, &
       Tk,Vhk,VOrigk,theta)
    !
    !     Handles energy minimization in torsion space
    !     
    !     SD  : steepest decents
    !     POWE: conjugent gradient powell method
    !
    use new_timer,only:timer_start,timer_stop,T_10extra    
    use dimens_fcm
    use number
    use bases_fcm
    use contrl
    use coord
    use deriv
    use energym
    use hbondm
    use inbnd
    use psf
    use stream
    use string
    character(len=*) COMLYN
    integer COMLEN
    real(chm_real)  Tk(NClus),Vhk(3,NClus), &
         VOrigk(3,NClus),theta(NClus)
    !
    ! ... local variables
    integer imethod,ierr,nt,convrg,k,iseg,fstclus,lstclus,nsegclus
    real(chm_real)  step,TOLFUN,TOLGRD,TOLSTP,XOLD,YOLD,ZOLD,VX,VY,VZ, &
         cstep,fold,func,qBase(3,NSeg),RBase(3,3,NSeg),bTk(6,NSeg)
    real(chm_real) MAXTHETA
    !
    integer alloc_err
    logical,allocatable, dimension(:) :: flag
    integer,allocatable, dimension(:) :: rcnt
    real(chm_real),allocatable, dimension(:,:) :: zk, fck
    real(chm_real),allocatable, dimension(:,:,:) :: phik
    !
    if(NClus == 1) then
       if(prnlev >= 2) write(outu,'(2A)') ' TADMINI: ', &
            'NClus=1, nothing to minimize.'
       COMLEN=0               ! suppress warnings
       return
    endif
    !
    allocate(phik(6,6,NClus),zk(6,NClus),fck(6,NClus), &
         rcnt(NClus),flag(NClus), &
         stat = alloc_err)
    if(alloc_err /= 0) &
         CALL WrnDie(-1,'<TADMINI>', &
         'Failed to allocate memory for phik array')
    !
    NSTEP=GTRMI(COMLYN,COMLEN,'NSTE',100)
    NPRINT=GTRMI(COMLYN,COMLEN,'NPRI',10)
    STEP=GTRMF(COMLYN,COMLEN,'STEP',PT001)
    TOLFUN=GTRMF(COMLYN,COMLEN,'TOLE',zero)
    TOLGRD=GTRMF(COMLYN,COMLEN,'TOLG',zero)
    TOLSTP=GTRMF(COMLYN,COMLEN,'TOLS',zero)
    MAXTHETA=GTRMF(COMLYN,COMLEN,'MAXT',PT01)
    !
    if(NSTEP <= 0) goto 999
    if(STEP == 0)  STEP=0.001d0
    !
    imethod=1                 ! default is SD
    if(indxa(COMLYN, COMLEN, 'SD  ') > 0) imethod=1
    if(indxa(COMLYN, COMLEN, 'POWE') > 0) imethod=2
    if(indxa(COMLYN, COMLEN, 'ABNR') > 0) imethod=3
    if(indxa(COMLYN, COMLEN, 'CONJ') > 0) imethod=4
    if(indxa(COMLYN, COMLEN, 'TN  ') > 0) imethod=5
    if(indxa(COMLYN, COMLEN, 'NRAP') > 0) imethod=6      
    !
    if(imethod == 1) then
       if(prnlev >= 2) then
          write(outu,'(/A)') &
               ' TADMINI: A steepest decents energy minimization is requested.'
          write(outu,'(/,A,I12,A,I12,/,3(A,F12.7,A,F12.7,/))') &
               '   NSTEP  = ',NSTEP, '   NPRINT = ',NPRINT, &
               '   STEP   = ',STEP,  '   TOLFUN = ',TOLFUN, &
               '   TOLGRD = ',TOLGRD,'   TOLSTP = ',TOLSTP, &
               '   MAXTHETA = ',MAXTHETA
       endif
    else if(imethod == 2) then
       if(prnlev >= 2) then
          write(outu,'(/A)') &
               ' TADMINI: A Powell energy minimization is requested.'
       endif
    else
       call wrndie(0,'<TAMD>','Unsurpported minimization method')
       goto 999
    endif
    ! note: LDYNAM,XOLD,YOLD,ZOLD,VX,VY,VZ are only relevant with images,
    !           which is not supported by current TAMD. Their values passed 
    !           to UPDATE here are meaningless and will cause segment fault.
    !
    call timer_start(T_10extra)                        
    CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN, &
         .true.,               & ! QPARSE (parse nbond specifications)
         .TRUE.,               & ! LCODES
         .TRUE.,               & ! LNBOND
         .TRUE.,               & ! LHBOND
         .TRUE.,               & ! LIMAGE
         0,                    & ! LDYNAM (only relevant in images:PROCATOMS)
         XOLD,YOLD,ZOLD,       & ! (only relevant in images:PROCATOMS)
         VX,VY,VZ)            ! (only relevant in images:PROCATOMS)
    call timer_stop(T_10extra)                        
    !
    call energy(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    if(prnlev >= 3) then
       call printe(outu,EPROP,ETERM,'MINI','MIN', &
            .true.,0,0d0,0d0,.false.)
    endif
    !
    do iseg=1,NSeg
       lstclus=NCSeg(iseg+1)
       k=OrigPt(lstclus)
       qBase(1,iseg)=X(k)
       qBase(2,iseg)=Y(k)
       qBase(3,iseg)=Z(k)
       do k=1,3
          do nt=1,3
             RBase(k,nt,iseg)=0d0
          enddo
          RBase(k,k,iseg)=1d0
       enddo
    enddo
    !
    convrg=0
    cstep=step
    func=EPROP(EPOT)
    fold=func
    nt=0
    !
100 nt=nt+1
    !
    !     compute torsion gradient from Cartesian gradients
    call GenSpOp1(NATOM,X,Y,Z,Vhk,VOrigk,ierr)
    call GetTc(NATOM,X,Y,Z,DX,DY,DZ,NSeg, &
         Vhk,VOrigk,Tk,bTk, &
         phik,fck,zk,rcnt,flag)
    !     
    if(imethod == 1) then     ! SD
       do iseg=1,NSeg
          fstclus=NCSeg(iseg)+1
          lstclus=NCSeg(iseg+1)
          nsegclus=lstclus-fstclus+1

          do k=fstclus,lstclus-1
             Theta(k)=-Tk(k)*cstep
             if(Theta(k) > MAXTHETA) then
                Theta(k)=MAXTHETA
             else if(Theta(k) < -MAXTHETA) then
                Theta(k)=-MAXTHETA
             endif
          enddo
          !     it appears to be tricky to update the base positions: chain reaction?
          !            qBase(1,iseg)=qBase(1,iseg)-bTk(1,iseg)*cstep
          !            qBase(2,iseg)=qBase(2,iseg)-bTk(2,iseg)*cstep
          !            qBase(3,iseg)=qBase(3,iseg)-bTk(3,iseg)*cstep
       enddo
       call UpdateCor2(NATOM,X,Y,Z,NSeg, &
            Vhk,VOrigk,Theta,qBase,RBase, &
            fck,phik,rcnt,flag)
       call energy(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       func=EPROP(EPOT)
       if(fold <= func) then
          cstep=cstep/2d0
       else
          !            cstep=min(MAXSTEP,cstep*1.1d0)
       endif
       if(nt >= nstep)              convrg=4
       if(abs(fold-func) <= tolfun) convrg=3
       if(cstep < tolstp)          convrg=1
       fold=func
       !     
       if(prnlev >= 3.and.nprint > 0.and. &
            (convrg /= 0.or.mod(nt,nprint) == 0)) then
          call printe(outu,EPROP,ETERM,'MINI','MIN', &
               .false.,nt,0d0,cstep,.false.)
       endif
       !     
       if(convrg /= 0.and.prnlev >= 2) then
          if(convrg == 1) then
             write(outu,'(/,A,F10.7,A,/)') &
                  ' TADMINI: Minimization exiting with step tolerance (', &
                  TOLSTP,') satisfied.'
          ELSE IF (CONVRG  ==  2) THEN
             WRITE (OUTU,'(/,A,F10.7,A,/)') &
                  ' TADMINI: Minimization exiting with gradient tolerance (', &
                  TOLGRD,') satisfied.'
          ELSE IF (CONVRG  ==  3) THEN
             WRITE (OUTU,'(/,A,F10.7,A,/)') &
                  ' TADMINI: Minimization exiting with function tolerance (', &
                  TOLFUN,') satisfied.'
          ELSE IF (CONVRG  ==  4) THEN
             WRITE (OUTU,'(/,A,I10,A,/)') &
                  ' TADMINI: Minimization exiting with number of steps limit (', &
                  NSTEP,') exceeded.'
          ELSE
             WRITE (OUTU,'(/,A,I5,A,/)') &
                  ' TADMINI: Unknown convergence status (',CONVRG,').'
          ENDIF
       endif
       !
       if(convrg /= 0) then
          goto 999
       else
          goto 100
       endif
       !     
    else if(imethod == 2) then ! Powell
       stop 'no Powell yet'            
    else
       stop 'no valid minimization metho ID'
    endif
    !
999 continue
    !
    deallocate(phik,zk,fck,rcnt,flag)
    !
    return
  end subroutine tadmini
  !
  !======================================================================
  subroutine GenSpOp1(NTot,X,Y,Z, &
                                !... outputs
       Vhk,VOrigk,ierr)
    !     
    !     given tree topology and coordinates, compute: Vhk, VOrigk
    !
    use dimens_fcm
    use stream
    integer NTot
    real(chm_real)  X(NTot),Y(NTot),Z(NTot)
    integer k,i,j,ierr
    real(chm_real)  Vhk(3,*),VOrigk(3,*),hl
    !
    ierr=0
    !
    !     Vhk and VOrigk
    do k=1, NClus-1
       if(Parent(k) > 0) then ! not a base cluster
          i=OrigPt(k)
          j=HingePt(k)
          Vhk(1,k)=X(i)-X(j)
          Vhk(2,k)=Y(i)-Y(j)
          Vhk(3,k)=Z(i)-Z(j)
          hl=Vhk(1,k)*Vhk(1,k)+Vhk(2,k)*Vhk(2,k)+Vhk(3,k)*Vhk(3,k)
          hl=sqrt(hl)
          if(hl < 0.000001) then
             if(wrnlev >= 2) write(outu,'(A)') &
                  ' GenSpOp1: invalid hinge with very small norm'
             call wrndie(-2,'<TAMD>','fatal error in GenSpOp1')
             ierr=1
             return
          endif
          Vhk(1,k)=Vhk(1,k)/hl
          Vhk(2,k)=Vhk(2,k)/hl
          Vhk(3,k)=Vhk(3,k)/hl
          !     
          j=OrigPt(Parent(k))
          VOrigk(1,k)=X(i)-X(j)
          VOrigk(2,k)=Y(i)-Y(j)
          VOrigk(3,k)=Z(i)-Z(j)
       endif
    enddo
    return
  end       subroutine GenSpOp1
  !
  !======================================================================
  subroutine GetTc(NATOM,X,Y,Z,DX,DY,DZ,NSeg, &
       Vhk,VOrigk, &
                                ! ... outputs
       Tk,bTk, &
                                ! ... auxillary matrices
       phik,Fck,zk,readychild,notdone)
    !
    !     given tree topology, Cartesian coordinates and gradient,
    !     compute torsion gradient: Tc = H Phi Fc
    !
    !       zk(1)   = Fc(1)
    !       zk(k+1) = Phi(k+1,k) * zk(k) + Fc(k)
    !
    !       Tc(k) = H(k) zk(k)
    !
    use stream
    use number
    integer NATOM, NSeg
    integer readyChild(NClus)
    real(chm_real)  Vhk(3,NClus),VOrigk(3,NClus), &
         X(NATOM),Y(NATOM),Z(NATOM), &
         DX(NATOM),DY(NATOM),DZ(NATOM),Tk(NClus),bTk(6,NSeg), &
         fck(6,NClus),phik(6,6,NClus),zk(6,NClus)
    logical notdone(NClus)
    integer j,k,kc,n,nk0,done,iseg,fstclus,lstclus,nsegclus
    real(chm_real)  vk(3)
    !
    do k=1,NClus
       do j=1,6
          FCk(j,k)=0d0
       enddo
    enddo
    !
    !     f^(c)_k: Cartesian spatial force
    !
    do n=1,NATOM
       k=ClusID(n)            ! cluster id
       nk0=OrigPt(k)          ! origin of kth cluster
       vk(1)=X(n)-X(nk0)
       vk(2)=Y(n)-Y(nk0)
       vk(3)=Z(n)-Z(nk0)
       FCk(1,k)=FCk(1,k)+vk(2)*DZ(n)-vk(3)*DY(n)
       FCk(2,k)=FCk(2,k)+vk(3)*DX(n)-vk(1)*DZ(n)
       FCk(3,k)=FCk(3,k)+vk(1)*DY(n)-vk(2)*DX(n)
       FCk(4,k)=FCk(4,k)+DX(n)
       FCk(5,k)=FCk(5,k)+DY(n)
       FCk(6,k)=FCk(6,k)+DZ(n)
    enddo
    !
    !     zk = Phi Fck, Tc(k) = H(k) zk(k)
    do iseg=1,NSeg
       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       nsegclus=lstclus-fstclus+1
       do k=fstclus,lstclus
          notdone(k)=.true.
          readychild(k)=0
       enddo
       done=0
       do while(done < nsegclus) ! .true. ==> some nodes not processed yet
          do k=fstclus,lstclus
             if(notdone(k).and.(readychild(k) == NChild(k))) then ! ready 
                do j=1,6
                   zk(j,k)=Fck(j,k)
                enddo
                do n=1,NChild(k)
                   kc=Child(n,k)
                   call matxvec('n',6,ONE,phik(1,1,kc),zk(1,kc), &
                        ONE,zk(1,k))
                enddo
                if(k < lstclus) then ! not base cluster with 6 DOF hinge     
                   call makephik(phik(1,1,k),VOrigk(1,k))
                   Tk(k)=0d0
                   do j=1,3
                      Tk(k)=Tk(k)+Vhk(j,k)*zk(j,k)
                   enddo
                   readychild(Parent(k))=readychild(Parent(k))+1
                else
                   do j=1,6
                      bTk(j,iseg)=zk(j,lstclus)
                   enddo
                endif
                notdone(k)=.false.
                done=done+1
             endif
          enddo
       enddo
    enddo
    return
  end subroutine GetTc
  !
  !=======================================================================
  subroutine tadcntrl(COMLYN,COMLEN,XOLD,YOLD,ZOLD, &
       freeat, nfreat, &
       SpVel,Theta,dTheta,ddTheta, &
       Vhk,VOrigk,VCMk,tmk,ITk,Tk,dThetaOld,dThetaNew,VX,VY,VZ)
    !
    !     TADCNTRL dispatches the torsion angle dynamics loops
    !
    !     Arguments: (see below)
    !     
    !     PSF, BNBND and TREE stuctures are obtained from common blocks.
    !
    !     Jianhan Chen, DEC-2003
    !

    use new_timer,only:timer_start,timer_stop,timer_stpstrt,   & 
         T_10extra,T_dcntrl,T_list,T_dynamc                   
#if KEY_CHEQ==1
    use cheq,only: qcg,   &                  
#endif
#if KEY_CHEQ==1
         DCH,SUMDCH,DDCH                   
#endif

    use dimens_fcm
    use number
    use bases_fcm
    use consta
    use contrl
    use coord
    use ctitla
    use deriv
    use cvio
    use energym
    use averfluc
    use eutil
    use hbondm
    use inbnd
    use psf
    use stream
    use string
    use heurist,only:updeci
    use dynutil, only: assvel
#if KEY_TMD==1
    use tmd,only:inrt  
#endif
    !
    character(len=*) COMLYN
    integer COMLEN, NFREAT
    integer FREEAT(NATOM)
    real(chm_real)  XOLD(NATOM),YOLD(NATOM),ZOLD(NATOM) 
    real(chm_real)  VX(NATOM),VY(NATOM),VZ(NATOM)
    real(chm_real)  SpVel(6,NClus),    & ! spatial velocity
         Theta(NClus),         & ! torsional increments of h(k)
         dTheta(NClus),        & ! torsional speed of h(k)
         ddTheta(NClus),       & ! torsional acceleration of h(k)
         Vhk(3,NClus),         & ! unit vector h(k):=q(k,0)-q(p(k),h)
         VOrigk(3,NClus),      & ! Orig(k)=q(k,0)-q(p(k),0), i.e.,l(k+1,k)
         VCMk(3,NClus),        & ! vector to Center of Mass:=q(k,CM)-q(k,0)
         tmk(NClus),           & ! total mass of each cluster
         ITk(3,3,NClus),       & ! cluster intertia tensor with repect q(k,0)
         Tk(NClus),            & ! hinge forces
         dThetaOld(NClus),     & ! dtheta(t-delt/2)
         dThetaNew(NClus)     ! dtheta(t+delt/2)
    !
    ! ... local variables
    integer NDEGF,i,j,k,n,ierr,jhstrt,curtime,ncycle, &
         istep,ist1,istpsa,numstp,trialcnt,MAXTRIAL
    parameter (MAXTRIAL=10)   ! maximum number of trials in velocity assignments
    real(chm_real)  seed,DNUM,DNM1,DNP1,DNPM1,RVAL,FLUCTD
    real(chm_real)  XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
    logical QALWRT(6),debug,reject
    real(chm_real)  bddTheta(6,NSeg),                  & ! base acceleration
         bSpVelOld(6,NSeg),bSpVelNew(6,NSeg),  & ! base spatial velocities
         QuatOld(0:3,NSeg),Quat(0:3,NSeg),     & ! base orientation 
         qBase(3,NSeg),                        & ! base translation
         RBase(3,3,NSeg),                      & ! base rotation
         bTk(6,NSeg)                          ! base hinge force vector
    real(chm_real) jhtemp,avetem,gamma,lambda,TNEW,tmpa(3), &
         phi1,totmass,entot0,totken,totkeo,time
    !
    !     
    ! ... auxiliary arrays for recursive TAD algorithm
    !      integer iphik, idk, ifck, iak, ibk, igk, iepsi, 
    !     &     izkpl, ipkpl, ialph, ircnt, iflag       
    integer alloc_err
    logical,allocatable, dimension(:) :: flag
    integer,allocatable, dimension(:) :: rcnt
    real(chm_real),allocatable, dimension(:) :: dk
    real(chm_real),allocatable, dimension(:,:) ::  &
         fck,ak,bk,gk,epsi,zkpl,alph
    real(chm_real),allocatable, dimension(:,:,:) :: phik,pkpl
    !
    integer iseg,fstclus,lstclus,nsegclus ! for multiple chains

#if KEY_CONSHELIX==1
      lqstprt = .FALSE.
      dynamq = .TRUE.
#endif 
    !
    ! ... allocate for auxillary matrices
    ist1=NClus

    allocate(phik(6,6,ist1),dk(ist1),fck(6,NClus), &
         ak(6,NClus),bk(6,NClus),gk(6,ist1),epsi(6,ist1), &
         zkpl(6,NClus),pkpl(6,6,NClus),alph(6,NClus), &
         rcnt(NClus),flag(NClus), &
         stat = alloc_err)
    if(alloc_err /= 0) &
         CALL WrnDie(-1,'<TADCNTRL>', &
         'Failed to allocate memory for phik array')
    !
    IUNREA=GTRMI(COMLYN,COMLEN,'IUNR',-1)
    IUNWRI=GTRMI(COMLYN,COMLEN,'IUNW',-1)
    IUNCRD=GTRMI(COMLYN,COMLEN,'IUNC',-1)
    IUNVEL=GTRMI(COMLYN,COMLEN,'IUNV',-1)
    IF(PRNLEV >= 2) WRITE(OUTU,5) IUNREA,IUNWRI,-1,IUNCRD,IUNVEL,-1
5   FORMAT('  IUNREA =',I3,7X,'  IUNWRI =',I3,7X,'   IUNOS =',I3,/, &
         '  IUNCRD =',I3,7X,'  IUNVEL =',I3,7X,'   KUNIT =',I3,7X)
    !
    ! ... read restart if necessary
    if(irest == 1.and.iunrea <= 0) then
       call wrndie(-2,'<TAMD>','NO RESTART FILE')
       goto 999
    endif
    if(irest == 1) then
       call readtadyn(iunrea,NATOM,X,Y,Z,NSeg, &
            TADVER,dTheta,dThetaOld,SpVel,bSpVelOld, &
            npriv,jhstrt,ndegf,nstep,nsavc,nsavv,seed,avetem,istpsa)
       IF(PRNLEV >= 2) WRITE(OUTU,15) NSTEP,JHSTRT
15     FORMAT(' NSTEP  =',I6,'  JHSTRT =',I6)
       JHTEMP=AVETEM*JHSTRT
       if(ndegf /= NClus+5*NSeg) then
          call wrndie(0, '<TAMD>', &
               'NDEGF does not match with NCLUS/NSEG')
          goto 999
       endif
    else
       npriv=0
       jhstrt=0
       jhtemp=0d0
    endif
    !
    ! ... read in dynamics variables
    !
    if(indx(COMLYN,COMLEN,'ISEE',4) > 0.and.irest == 1) &
         call wrndie(3,'<TAMD>','ISEED SPECIFIED WHILE RESTART')
    ISEED=GTRMI(COMLYN,COMLEN,'ISEE',ISEED)
    if(irest == 1) iseed=int(seed) ! use seed from restart file if applicable
    NSTEP=GTRMI(COMLYN,COMLEN,'NSTE',NSTEP)
    NPRINT=GTRMI(COMLYN,COMLEN,'NPRI',NPRINT)
    NTRFRQ=GTRMI(COMLYN,COMLEN,'NTRF',NTRFRQ)
    IASORS=GTRMI(COMLYN,COMLEN,'IASO',IASORS)
    IPRFRQ=GTRMI(COMLYN,COMLEN,'IPRF',IPRFRQ)
    NSAVC=GTRMI(COMLYN,COMLEN,'NSAVC',NSAVC)
    NSAVV=GTRMI(COMLYN,COMLEN,'NSAVV',NSAVV)
    ISVFRQ=GTRMI(COMLYN,COMLEN,'ISVF',0)
    TIMEST=GTRMF(COMLYN,COMLEN,'TIME',ZERO)
    ECHECK=GTRMF(COMLYN,COMLEN,'ECHE',ECHECK)
    TREF=GTRMF(COMLYN,COMLEN,'TREF',TREF)
    FIRSTT=GTRMF(COMLYN,COMLEN,'FIRS',FIRSTT)
    QREF1=GTRMF(COMLYN,COMLEN,'QREF',QREF1)
    debug=indxa(COMLYN, COMLEN, 'DEBUG') > 0
    !
    DELTA=TIMEST/TIMFAC       ! convert ps to AKMA time unit
    if(delta <= 0d0) then
       call wrndie(-3,'<TAMD>','Zero time step specified')
       goto 999
    endif
    !
    ! ... check frequency specifications (note: ncycle is not actually used; 
    !     the sole purpose of calling fincyc() is to double check frequencies)
    i=0
    j=0
    n=0
    ist1=0
    call fincyc(ncycle,i,j,ntrfrq,isvfrq,inbfrq,ihbfrq, &
         ilbfrq,n,ist1,0,nstep &
#if KEY_TMD==1
         ,inrt &
#endif
    )
    !         
    if(nstep <= 0) then 
       call wrndie(0,'<TAMD>', 'Invalid number of dynamic steps')
       goto 999
    endif
    if(iprfrq > nstep) then
       iprfrq=nstep
    ELSE IF(IPRFRQ > 0) THEN
       IPRFRQ=(IPRFRQ/NCYCLE)*NCYCLE
       IF(IPRFRQ <= 0) IPRFRQ=NCYCLE
    ELSE
       IPRFRQ=NSTEP
    ENDIF
    if(nprint <= 0.or.nprint > iprfrq) nprint=iprfrq
    !
    !     note: LDYNAM,XOLD,YOLD,ZOLD,VX,VY,VZ are only relevant with images,
    !           which is not supported by current TAMD. Their values passed 
    !           to UPDATE here are not related to image and will cause problems
    !           if image is active (should not happen here).
    !
    call timer_start(T_10extra)                        
    CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN, &
         .true.,               & ! QPARSE (parse nbond specifications)
         .TRUE.,               & ! LCODES
         .TRUE.,               & ! LNBOND
         .TRUE.,               & ! LHBOND
         .TRUE.,               & ! LIMAGE
         0,                    & ! LDYNAM (only relevant in images:PROCATOMS)
         XOLD,YOLD,ZOLD,       & ! (only relevant in images:PROCATOMS)
         VX,VY,VZ)            ! (only relevant in images:PROCATOMS)
    call timer_stpstrt(T_10extra,T_dcntrl)                        
    !
    ! ... print out state of simulation
    if(prnlev >= 2) then
       if(qref1 <= 1d-8) then
          write(outu,'(A)') 'TADCNTRL> Constant energy requested.'
       else
          write(outu,'(A)') 'TADCNTRL> Constant temperature requested.'
          write(outu,'(2(8X,A,F10.3,A,/))') &
               '  Reference temperature         = ',TREF,' K.', &
               '  Temperature coupling constant = ',QREF1,' time steps.'
       endif
       if(debug) write(outu,'(A,/)')  &
            ' +++++++++++++++++++++ DEBUGGING MODE +++++++++++++++++++++++++'

       WRITE(OUTU,40) NSTEP,NSAVC,NSAVV, -1,-1,IASORS, &
            -1,-1,NTRFRQ, -1,-1,NPRINT, INBFRQ,IHBFRQ,IPRFRQ, &
            -1,-1,ISEED, ISVFRQ,ncycle,-1
40     FORMAT('   NSTEP =',I10,  '   NSAVC =',I6,4X,'   NSAVV =',I6/ &
            '  ISCALE =',I6,4X,'  ISCVEL =',I6,4X,'  IASORS =',I6/ &
            '  IASVEL =',I6,4X,'  ICHECW =',I6,4X,'  NTRFRQ =',I6/ &
            '  IHTFRQ =',I6,4X,'  IEQFRQ =',I6,4X,'  NPRINT =',I6/ &
            '  INBFRQ =',I6,4X,'  IHBFRQ =',I6,4X,'  IPRFRQ =',I6/ &
            '  ILBFRQ =',I6,4X,'  IMGFRQ =',I6,4X,'   ISEED =',I20/ &
            '  ISVFRQ =',I6,4X,'  NCYCLE =',I6,4X,'   NSNOS =',I6)
       WRITE(OUTU,45) TREF,0d0,TREF,TREF,-1d0,-1d0, &
            DELTA,DELTA*TIMFAC
45     FORMAT('  FIRSTT =',F10.3,'  TEMINC =',F10.3,'  TSTRUC =',F10.3/ &
            '  FINALT =',F10.3,'  TWINDH =',F10.3,'  TWINDL =',F10.3/ &
            /'  TIME STEP = ',1PG12.5,' AKMA      ',1PG12.5,' PS'/)

    endif
    !
    IF(INBFRQ >= 50) THEN
       CALL WRNDIE(1,'<DCNTRL>', &
            'Nonbond bond update frequency may be too large.')
    ENDIF
    IF(IHBFRQ >= 50 .AND. CUTHB > 1.0) THEN
       CALL WRNDIE(1,'<DCNTRL>', &
            'Hydrogen bond update frequency may be too large.')
    ENDIF
    !
    ! ... initialization
    !
    NDEGF=NClus+5*NSeg        ! total number of degree-of-freedoms (DOFs)
    write(outu,55) NDEGF      ! (assuming single 6-DOF hinge)
55  format(' NUMBER OF DEGREES OF FREEDOM = ',I6)
    do j=1,6
       QALWRT(j)=.false.      ! remove all 6 global degrees of freedom
    enddo
    totmass=0d0
    do k=1,NClus
       tmk(k)=0d0
    enddo
    do n=1, NATOM
       totmass=totmass+AMASS(n)
       tmk(ClusID(n))=tmk(ClusID(n))+AMASS(n)
    enddo
    call GenSpOp2(NATOM,X,Y,Z,AMASS,tmk,Vhk,VOrigk,VCMk,ITk,ierr)
    !
    do iseg=1,NSeg
       Quat(0,iseg)=1d0       ! initial body-fixed frame = inertial frame
       Quat(1,iseg)=0d0
       Quat(2,iseg)=0d0
       Quat(3,iseg)=0d0
    enddo
    ! 
    if(qref1 <= 1d-8) then    ! no coupling to thermo bath
       gamma=0d0
    else
       gamma=1d0/qref1
    endif
    !
    ! ... prepare initial spatial velocities
    if(irest == 1) then
       !         do iseg=1,NSeg
       !            fstclus=NCSeg(iseg)+1
       !            lstclus=NCSeg(iseg+1)
       !            nsegclus=lstclus-fstclus+1
       !            do k=fstclus,lstclus-1
       !               dThetaOld(k)=dTheta(k)
       !            enddo
       !            do j=1,6
       !               SpVel(j,lstclus)=bSpVelOld(j,iseg)
       !            enddo
       !         enddo
       call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
    else                     
       if(firstt <= 1d0) then ! set intitial velecities to be zero
          do k=1,NClus
             dTheta(k)=0d0
             dThetaOld(k)=0d0
          enddo
          do k=1,NClus
             do j=1,6
                SpVel(j,k)=0d0
             enddo
          enddo
          do iseg=1,NSeg
             do j=1,6
                bSpVelOld(j,iseg)=0d0
             enddo
          enddo
       else                   ! assign initial velocity (Gaussian)
          trialcnt=0
60        call ASSVEL ( firstt,X,Y,Z,VX,VY,VZ, &
               AMASS,ISEED,1,i,NATOM,IMOVE)
          !     
          !     remove center of mass (CM) rotation and translation
          call cenmss2(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
               VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          call ROTSTP(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
               VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          !
          !     map Cartesian velocity onto internal coordinate
          !     (based on Eqn[78] of GM Clore's J. Magn. Reson. 2001 paper)
          !
          call GetTe(NSeg,Vhk,VOrigk,NATOM,X,Y,Z,AMASS,VX,VY,VZ, &
               Tk,bTk, &
               fck,phik,ak,rcnt,flag)
          call SPTAD2(NATOM,X,Y,Z,DX,DY,DZ,NSeg, &
               Vhk,VOrigk,VCMk,tmk,ITk,dTheta,SpVel,Tk,bTk,.false., &
               dTheta,bSpVelOld, &
               phik,dk,fck,ak,bk,gk,epsi,zkpl,pkpl,alph,rcnt,flag)
          !
          !     compute the spatial velocities based on mapped torsion velocities
          do iseg=1,NSeg
             lstclus=NCSeg(iseg+1)
             do j=1,6
                SpVel(j,lstclus)=bSpVelOld(j,iseg)
             enddo
          enddo
          call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
          !
          !     scale SpVel to get the desired temperature
          call GetKen(NClus,ITk,tmk,VCMk,SpVel,totken)
          TNEW=totken/NDEGF/KBOLTZ
          if(TNEW > 1d-2) then ! scale up velocities unless TNEW is too small
             lambda=sqrt(firstt/TNEW)
             do k=1,NClus
                dTheta(k)=dTheta(k)*lambda
                do j=1,6
                   SpVel(j,k)=SpVel(j,k)*lambda
                enddo
             enddo
          endif
          !
          if(prnlev >= 5) then ! print info about CM again
             call GetScalVel(NATOM,X,Y,Z,SpVel, &
                  VX,VY,VZ)
             call cenmss2(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
                  VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          endif
          do k=1,NClus
             dThetaOld(k)=dTheta(k)
          enddo
          do iseg=1,NSeg
             lstclus=NCSeg(iseg+1)
             do j=1,6
                bSpVelOld(j,iseg)=SpVel(j,lstclus)
             enddo
          enddo
          !
       endif                  ! if(firstt <= 1d0)
       !     
    endif                     ! if(irest == 1) 
    !________________________________________________________________________
    !
    !     At this point all of the input line should be parsed so anything
    !     remaining must be an error.  To avoid expensive mistakes we test
    !     and bomb out.
    !
    CALL XTRANE(COMLYN,COMLEN,'TADCNTRL')
    IF (COMLEN > 0) CALL DIEWRN(-2)
    !________________________________________________________________________
    !     preparing for dynamics: if fresh start, not back propagation to 
    !     estimate state at -delt/2.
    !
    ! ... get previous energy for printing, get force for first step
    call timer_stop (T_dcntrl) 
    call timer_start(T_list)   
    CALL UPDECI(1,X,Y,Z,WMAIN,2,XOLD,YOLD,ZOLD,(/zero/),(/zero/),(/zero/))
    call timer_stop(T_list)   

    call energy(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
    call timer_start(T_dcntrl) 
    !
    do k=1,NClus
       Tk(k)=0d0              ! no net torsional residual force (all in Fck)
    enddo
    do iseg=1,NSeg
       do j=1,6
          bTk(j,iseg)=0d0
       enddo
    enddo
    !
    !     compute kenetic energy
    call GetKen(NClus,ITk,tmk,VCMk,SpVel,totken)
    TNEW=totken/NDEGF/KBOLTZ
    totken=totken*0.5d0
    totkeo=totken             ! first step only
    !
    EPROP(GRMS)=0d0
    EPROP(TEMPS)=TNEW
    EPROP(TOTKE)=totken
    EPROP(KEPR)=totkeo
    EPROP(KEPR2)=EPROP(KEPR)
    EPROP(PEPR)=EPROP(EPOT)
    EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)
    EPROP(HFCKE) = HALF * (TOTKEN + EPROP(KEPR))     
    EPROP(EHFC) = (EPROP(KEPR) - EPROP(TOTKE) + EPROP(PEPR) - &
         EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEN)/TWELVE
    EPROP(HFCTE)=EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
    EPROP(TEPR)=EPROP(HFCTE)
    entot0=EPROP(TOTE)
    !
    time=TIMFAC*NPRIV*DELTA 
    IF(PRNLEV >= 2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM,'DYNA','DYN', &
            .TRUE.,0, TIME, ZERO, .false.)
    ENDIF
    !     
    ! ... initialize accumulation arrays
    if(jhstrt == 0) then
       JHTEMP=ZERO
       FITP=ZERO
       call avfl_reset_lt()
    endif
    !_________________________________________________________________________
    ! ... start the main dynamic loop
    !
    call timer_start(T_dynamc) 
    do istep=1, NStep            ! main loop of dynamics
       jhstrt=jhstrt+1
       npriv=npriv+1
       !     
       ! ... initialize accumulation arrays (every iprfrq steps)
       IF(MOD(ISTEP-1,IPRFRQ) == 0) THEN
          ist1=istep-1
          call avfl_reset()
          FITA = ZERO
       ENDIF
       !
       ! ... compute the torsional acceleration using Jain's recursive algorithm
       call SPTAD2(NATOM,X,Y,Z,DX,DY,DZ,NSeg, &
            Vhk,VOrigk,VCMk,tmk,ITk,dTheta,SpVel,Tk,bTk,.true., &
            ddTheta,bddTheta, &
            phik,dk,fck,ak,bk,gk,epsi,zkpl,pkpl,alph,rcnt,flag)
       !
       !     modified leap-frog integration
       !     
       do iseg=1,NSeg
          fstclus=NCSeg(iseg)+1
          lstclus=NCSeg(iseg+1)
          nsegclus=lstclus-fstclus+1

          do k=fstclus,lstclus-1
             dThetaNew(k)=dThetaOld(k)+delta*ddTheta(k)
             Theta(k)=dThetaNew(k)*delta
             dTheta(k)=1.5d0*dThetaNew(k)-0.5d0*dThetaOld(k) ! for force calc.
             dThetaOld(k)=dThetaNew(k)
          enddo
          !
          !     update body orientation and base spatial velocity
          call UpdateBase(NATOM,X,Y,Z,nsegclus,OrigPt(fstclus), &
               Quat(0,iseg),delta, &
               bddTheta(1,iseg),bSpVelOld(1,iseg),bSpVelNew(1,iseg), &
               SpVel(1,fstclus),qBase(1,iseg),RBase(1,1,iseg),ierr)
       enddo                  ! do iseg=1,NSeg
       !
       !     update the atomic coordinates and spatial velocities
       call UpdateCor2(NATOM,X,Y,Z,NSeg, &
            Vhk,VOrigk,Theta,qBase,RBase, &
            fck,pkpl,rcnt,flag)
       !
       call GenSpOp2(NATOM,X,Y,Z,AMass,tmk,Vhk,VOrigk,VCMk,ITk,ierr)
       !
       call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
       !______________________________________________________________________
       !     stop global rotation/tranlation
       !
       if(ntrfrq > 0.and.mod(istep,ntrfrq) == 0) then ! re-assign
          if(IASORS /= 0) then
             call ASSVEL ( TREF,X,Y,Z,VX,VY,VZ, &
                  AMASS,ISEED,1,i,NATOM,IMOVE)
          else 
             !
             !     compute Cartesian velocities from spatial velocities
             call GetScalVel(NATOM,X,Y,Z,SpVel, &
                  VX,VY,VZ)
          endif
          !
          !     remove R/T
          call cenmss2(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
               VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          call ROTSTP(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
               VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          !
          !     map new Cartesian velocities back to internal velocities
          call GetTe(NSeg,Vhk,VOrigk,NATOM,X,Y,Z,AMASS,VX,VY,VZ, &
               Tk,bTk,fck,phik, &
               ak,rcnt,flag)
          call SPTAD2(NATOM,X,Y,Z,DX,DY,DZ,NSeg, &
               Vhk,VOrigk,VCMk,tmk,ITk,dTheta,SpVel,Tk,bTk,.false., &
               dTheta,bSpVelOld, &
               phik,dk,fck,ak,bk,gk,epsi,zkpl,pkpl,alph,rcnt,flag)
          !
          do iseg=1,NSeg
             lstclus=NCSeg(iseg+1)
             do j=1,6
                SpVel(j,lstclus)=bSpVelOld(j,iseg)
             enddo
          enddo
          call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
          !
          call GetKen(NClus,ITk,tmk,VCMk,SpVel,totken)
          !
          TNEW=totken/NDEGF/KBOLTZ
          if(TNEW > 1d-2) then 
             lambda=sqrt(tref/TNEW)
             do k=1,NClus
                dTheta(k)=dTheta(k)*lambda
                do j=1,6
                   SpVel(j,k)=SpVel(j,k)*lambda
                enddo
             enddo
          endif
          !
          !     print new information about CM
          if(prnlev >= 5) then
             call GetScalVel(NATOM,X,Y,Z,SpVel, &
                  VX,VY,VZ)
             call cenmss2(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
                  VXCM,VYCM,VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT)
          endif
          !
          !     clean up
          do k=1,NClus
             Tk(k)=0d0        ! clean up values from GetTe()
             dThetaOld(k)=dTheta(k)
          enddo
          do iseg=1,NSeg
             lstclus=NCSeg(iseg+1)
             do j=1,6
                bTk(j,iseg)=0d0 ! clean up values from GetTe()
                bSpVelOld(j,iseg)=SpVel(j,lstclus)
             enddo
          enddo
          !            
       endif                  ! if(ntrfrq > 0...
       !------------------------------------------------------------------------
       !     compute the instantanous temperature
       call GetKen(NClus,ITk,tmk,VCMk,SpVel,totken)
       TNEW=totken/NDEGF/KBOLTZ
       totken=totken*HALF
       !
       !     temperature control via velocity rescaling
       !
       if(gamma > 0d0) then
          lambda=sqrt(1d0+(TREF-TNEW)*gamma/TNEW)
          do k=1,NClus
             dThetaOld(k)=lambda*dThetaOld(k)
             dTheta(k)=lambda*dTheta(k)
             do j=1,6
                SpVel(j,k)=SpVel(j,k)*lambda
             enddo
          enddo
          do iseg=1,NSeg
             do j=1,6
                bSpVelOld(j,iseg)=lambda*bSpVelOld(j,iseg)
             enddo
          enddo
          TNEW=TNEW*lambda*lambda
          totken=totken*lambda*lambda
       endif

       ! ... write coordinate trajectory
       if (nsavc > 0) then
          if ((iolev > 0) .and. mod(istep, nsavc) == 0) then
             call writcv(x,y,z, &
#if KEY_CHEQ==1
                  CG,QCG,  & 
#endif
                  NATOM,FREEAT,NFREAT,NPRIV, &
                  ISTEP,NDEGF,DELTA,NSAVC,NSTEP,TITLEA, &
                  NTITLA,IUNCRD,.FALSE.,.FALSE., (/ 0 /), &
                  .FALSE., (/ ZERO /))
          end if
       end if

       ! ... write velocity trajectory (who would need it?)
       !         IF(NSAVV > 0.and.MOD(ISTEP,NSAVV) == 0) THEN
       !            IF(IOLEV > 0) THEN
       !               call GetScalVel(NATOM,X,Y,Z,AMASS,ClusID,NClus,OrigPt,
       !     &                         VX,VY,VZ,VK,SpVel)
       !               CALL WRITCV(VX,VY,VZ,
#if KEY_CHEQ==1
       !     &                     CG,QCG,     /* CG should be VCG here*/
#endif
       !     $                     NATOM,FREEAT,NFREAT,NPRIV,ISTEP,NDEGF,
       !     &                     DELTA,NSAVV,NSTEP,TITLEA,NTITLA,IUNVEL,
       !     $                     .TRUE.,.FALSE.,0,.FALSE.,0)
       !            ENDIF
       !         ENDIF
       !
       EPROP(PEPR)=EPROP(EPOT)
       EPROP(TEPR)=EPROP(HFCTE)
       !
       ! ... nobond updating and compute new force
       call timer_stop (T_dynamc) 
       call timer_stop(T_dcntrl)  
       call timer_start(T_list)   
       CALL UPDECI(ISTEP,X,Y,Z,WMAIN, &
            2,XOLD,YOLD,ZOLD,(/zero/),(/zero/),(/zero/))
       call timer_stop(T_list)    
       !
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1)
       call timer_start(T_dcntrl) 
       call timer_start(T_dynamc) 
       !
       EPROP(TEMPS)=TNEW
       EPROP(KEPR)=EPROP(TOTKE)
       EPROP(TOTKE)=totken
       EPROP(KEPR2)=EPROP(KEPR)
       EPROP(TOTE)=EPROP(EPOT)+EPROP(TOTKE)
       EPROP(HFCKE) = HALF * (TOTKEN + EPROP(KEPR))     
       EPROP(EHFC) = (EPROP(KEPR) - EPROP(TOTKE) + EPROP(PEPR) - &
            EPROP(EPOT))/THREE + (EPROP(KEPR2)- TOTKEN)/TWELVE
       EPROP(HFCTE)=EPROP(EPOT) + EPROP(TOTKE) + EPROP(EHFC)
       !
       ! ... check total energy change
       if(jhstrt > 2) then
          if(abs(EPROP(TEPR)-EPROP(HFCTE)) >  &
               max(ECHECK,PTONE*EPROP(HFCKE))) then
             IF(WRNLEV >= 2) WRITE(OUTU,2000) &
                  ECHECK,EPROP(TEPR),EPROP(HFCTE),EPROP(HFCKE)
2000         FORMAT(' TOTAL ENERGY CHANGE EXCEEDED'/G12.2, &
                  ' KCAL AND 10% OF THE TOTAL KINETIC ENERGY IN THE LAST STEP'/ &
                  ' PREVIOUS E =',G14.4,' CURRENT E =',G14.4,' KINETIC =',G14.4)
             !
             !     do something here? or just bail out?
             !               
             !               CALL WRNDIE(-2,'<TAMD>',
             !     &              'ENERGY CHANGE TOLERANCE EXCEEDED')
          endif
       endif
       !
       if(nprint > 0.and.mod(istep,nprint) == 0) then 
          time=TIMFAC*NPRIV*DELTA 
          IF(PRNLEV > 0) THEN
             CALL PRINTE(OUTU, EPROP, ETERM,'DYNA','DYN', &
                  .false.,istep, TIME, ZERO, .false.)
          ENDIF
       endif
       !
       ! ... accumulate energy statistics (sum(X) and sum(X^2))
       JHTEMP = JHTEMP + EPROP(TEMPS)
       call avfl_update(EPROP, ETERM, EPRESS)
       ISTPSA=MOD(IST1,IPRFRQ)+ISTEP-IST1
       FITA = FITA + ISTPSA * EPROP(HFCTE)
       FITP = FITP + JHSTRT * EPROP(HFCTE)
       !
       !         if(mod(istep,iprfrq) == 0.or.istep.eq.nstep) then 
       !         (this will mess up the statistics if restart)
       if(mod(istep,iprfrq) == 0) then                    
          !
          ! ... accumulate the long-time statistics                    
          call avfl_update_lt()
          !
          NUMSTP = ISTPSA
          DNUM   = NUMSTP
          DNM1   = DNUM - ONE
          DNP1   = DNUM + ONE
          DNPM1  = DNM1 * DNP1
          IF(NUMSTP > 1) THEN
             DRIFTA = (TWO*FITA-DNP1*EPRPA(HFCTE))*SIX/(DNPM1*DNUM)
             EAT0A  = EPRPA(HFCTE)/DNUM-DRIFTA*DNP1/TWO
             RVAL = EPRP2A(HFCTE)/DNUM - (EPRPA(HFCTE)/DNUM)**2
             IF(RVAL > ZERO) THEN
                CORRA=DNPM1/RVAL/TWELVE
                CORRA=DRIFTA*SQRT(CORRA)
             ELSE
                CORRA=ZERO
             ENDIF
             !
             ! ... Compute statistics.
             call avfl_compute(DNUM)
          ENDIF
          ! ... Print out the results.
          IF(PRNLEV >= 2) THEN
             call avfl_print_aver(NUMSTP, TIME, tag='TAMD>')
             call avfl_print_fluc(NUMSTP, TIME, tag='TAMD>')
          ENDIF
          !
          DRIFTP = DRIFTA
          EAT0P  = EAT0A
          CORRP  = CORRA
          IF (JHSTRT <= NUMSTP) GOTO 190 ! no need for long time statistics
          NUMSTP = JHSTRT
          DNUM   = NUMSTP
          DNM1   = DNUM - ONE
          DNP1   = DNUM + ONE
          DNPM1  = DNM1 * DNP1
          IF(NUMSTP > 1) THEN
             DRIFTP = (TWO*FITP - DNP1*EPRPP(HFCTE))*SIX/(DNPM1*DNUM)
             EAT0P  = EPRPP(HFCTE)/DNUM-DRIFTP*DNP1/TWO
             RVAL=EPRP2P(HFCTE)/DNUM - (EPRPP(HFCTE)/DNUM)**2
             IF(RVAL > ZERO) THEN
                CORRP=DNPM1/RVAL/TWELVE
                CORRP=DRIFTP*SQRT(CORRP)
             ELSE
                CORRP=ZERO
             ENDIF
          ENDIF
          !
          call avfl_compute_lt(DNUM)
          ! 
          IF(PRNLEV >= 2) THEN
             call avfl_print_aver_lt(NUMSTP, TIME, tag='TAMD>')
             call avfl_print_fluc_lt(NUMSTP, TIME, tag='TAMD>')
          ENDIF
          !
190       continue
          WRITE(OUTU,195)DRIFTA,DRIFTP,EAT0A,EAT0P,CORRA,CORRP
195       FORMAT(/5X,'DRIFT/STEP (LAST-TOTAL): ',1P,2G17.8, &
               /5X,'E AT STEP 0            : ',1P,2G17.8, &
               /5X,'CORR. COEFFICIENT      : ',1P,2G17.8)
          !
       endif                  ! if(mod(istep,iprfrq) == 0) then ...
       !
       ! ... write restart file
       if(iunwri > 0.and.iolev.gt.0) then
          if(isvfrq > 0.and. &
               (mod(istep,isvfrq) == 0.or.istep.eq.nstep)) then
             avetem=jhtemp/jhstrt
             seed=iseed
             call writadyn(iunwri,NATOM,X,Y,Z,NSeg, &
                  TADVER,dTheta,dThetaOld,SpVel,bSpVelOld, &
                  NPRIV,JHSTRT,NDEGF,NSTEP,NSAVC,NSAVV,seed,avetem,0)
          endif
       endif
#if KEY_CONSHELIX==1
       mdstep=istep
#endif 
       !
       ! ... debugging records
       if(debug.and.nprint > 0.and.mod(istep,nprint) == 0) then 
          curtime=istep*1000*delta*TIMFAC
          phi1=0d0
          do j=1,6
             phi1=phi1+SpVel(j,NClus)*SpVel(j,NClus)
          enddo
          phi1=sqrt(phi1)
          write(68,100) curtime,dTheta,phi1 
          !            write(17,100) curtime,(SpVel(j,NClus),j=1,6) ! base velocity
          !            write(20,100) curtime,ddTheta  ! acceleration
          !            write(21,100) curtime,bddTheta ! base acceleration           
          write(69,100) curtime,EPROP(TOTKE),EPROP(EPOT),EPROP(TOTE), &
               100d0*(EPROP(TOTE)-entot0)/entot0, TNEW ! energy
          tmpa(1)=0d0
          tmpa(2)=0d0
          tmpa(3)=0d0
          do n=1,NATOM
             tmpa(1)=tmpa(1)+X(n)*AMass(n)
             tmpa(2)=tmpa(2)+Y(n)*AMass(n)
             tmpa(3)=tmpa(3)+Z(n)*AMass(n)
          enddo
          tmpa(1)=tmpa(1)/totmass
          tmpa(2)=tmpa(2)/totmass
          tmpa(3)=tmpa(3)/totmass
          phi1=tmpa(1)*tmpa(1)+tmpa(2)*tmpa(2)+tmpa(3)*tmpa(3)
          phi1=sqrt(phi1)
          write(70,100) curtime, tmpa, phi1 ! center of mass
       endif
100    format(I8,16(1X,F12.4))
       !
    enddo                     ! do nt=1,NStep
    call timer_stop (T_dynamc) 
    call timer_stop(T_dcntrl)  

999 continue
    !
    ! ... free spaces
    deallocate(phik,dk,fck,ak,bk,gk,epsi,zkpl,pkpl,alph,rcnt,flag)
    !
    return
  end subroutine tadcntrl
  !     **** ( end of subroutine tadcntrl ) ****
  !
  !======================================================================
  subroutine GetKen(NClus,ITk,tmk,VCMk,SpVel,KEN)
    !
    !     note: KEN/2 gives the total kenetic energy
    !
    use number
    integer NClus, k
    real(chm_real)  ITk(3,3,NClus),tmk(NClus),SpVel(6,NClus), &
         VCMk(3,NClus),KEN
    real(chm_real)  tmpa(3),tmpb(3)
    KEN=0d0
    do k=1, NClus
       call matxvec('n',3,ONE,ITk(1,1,k),SpVel(1,k),ZERO,tmpa)
       tmpb(1)=SpVel(2,k)*VCMk(3,k)-SpVel(3,k)*VCMk(2,k)
       tmpb(2)=SpVel(3,k)*VCMk(1,k)-SpVel(1,k)*VCMk(3,k)
       tmpb(3)=SpVel(1,k)*VCMk(2,k)-SpVel(2,k)*VCMk(1,k)
       KEN=KEN+tmk(k)*( SpVel(4,k)*SpVel(4,k)+SpVel(5,k)*SpVel(5,k) &
            +SpVel(6,k)*SpVel(6,k) ) + SpVel(1,k)*tmpa(1)+ &
            SpVel(2,k)*tmpa(2)+SpVel(3,k)*tmpa(3) + &
            2d0*tmk(k)*( SpVel(4,k)*tmpb(1)+ &
            SpVel(5,k)*tmpb(2)+SpVel(6,k)*tmpb(3) )
    enddo
    return
  end subroutine GetKen

  !======================================================================
  subroutine AssSpVel(TEMP,dTheta,SpVel, &
       Vhk,VOrigk,VCMk,tmk,ITk,NDEGF,NSeg, &
       NATOM,X,Y,Z,AMASS,IMOVE,VX,VY,VZ)
    !
    !     initial velocity assignment in internal coordinates 
    !     (no removal of global rotation and translation, do not use!!!)
    !
    use dimens_fcm
    use consta
    use number
    use stream
    !
    integer NATOM,NDEGF,NSeg,IMOVE(NATOM)
    real(chm_real) TEMP,dTheta(NClus-1),SpVel(6,NClus),Vhk(3,NClus-1), &
         VOrigk(3,NClus-1),VCMk(3,NClus),tmk(NClus),ITk(3,3,NClus), &
         X(NATOM),Y(NATOM),Z(NATOM),AMASS(NATOM), &
         VX(NATOM),VY(NATOM),VZ(NATOM)
    ! ... local variables
    integer j,k
    real(chm_real)  ken,tmpa(3),tmpb(3) !,rang
    !
    do j=1,6
       SpVel(j,NClus)=rang(iseed,ZERO,ONE)
    enddo
    if(NClus == 1) then
       if(wrnlev >= 2) write(outu,'(A,A)') ' AssSpVel: ', &
            'NClus=1, nothing to be assigned. Initial temperature is 0 K'
       return
    endif
    if(TEMP <= 1d-8) then
       do k=1, NClus-1
          dTheta(k)=0d0
       enddo
    else
100    continue
       do k=1,NClus-1
          dTheta(k)=rang(iseed,ZERO,ONE)
       enddo
       !     
       call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
       !     
       ken=0                 
       do k=1, NClus
          call matxvec('n',3,ONE,ITk(1,1,k),SpVel(1,k),ZERO,tmpa)
          tmpb(1)=SpVel(2,k)*VCMk(3,k)-SpVel(3,k)*VCMk(2,k)
          tmpb(2)=SpVel(3,k)*VCMk(1,k)-SpVel(1,k)*VCMk(3,k)
          tmpb(3)=SpVel(1,k)*VCMk(2,k)-SpVel(2,k)*VCMk(1,k)
          ken=ken+tmk(k)*( SpVel(4,k)*SpVel(4,k)+SpVel(5,k)*SpVel(5,k) &
               +SpVel(6,k)*SpVel(6,k) )+ SpVel(1,k)*tmpa(1)+ &
               SpVel(2,k)*tmpa(2)+SpVel(3,k)*tmpa(3) + &
               2d0*tmk(k)*( SpVel(4,k)*tmpb(1)+ &
               SpVel(5,k)*tmpb(2)+SpVel(6,k)*tmpb(3) )
       enddo
       ken=ken/NDEGF/KBOLTZ   ! temperature
       if(ken <= 1d-8) then
          if(wrnlev >= 4) write(outu,'(A)') &
               ' AssSpVel: very small intiatial velocity assignment.'
          goto 100            ! re-assign
       endif
       !
       ken=sqrt(TEMP/ken)
       do k=1,NClus-1
          dTheta(k)=ken*dTheta(k)
       enddo
       do j=1,6
          SpVel(j,NClus)=ken*SpVel(j,NClus)
       enddo
    endif
    !
    call UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
    !
    if(prnlev >= 3) write(outu,'(A,F10.3,A2)') &
         ' AssSpVel: velocity assigned at TEMP = ', TEMP, ' K'
    return
  end subroutine AssSpVel
  !
  !     a gaussian random number generator
  real(chm_real) function rang(idum,am,sd)
    integer idum
    real(chm_real) am, sd, a, b!, ran0
    a=SQRT(-2d0*LOG(ran0(idum)))
    b=6.2831853072*ran0(idum)
    rang=sd*a*cos(b)+am
    return
  end function rang
  !
  !     a random number generator: ran0=[0:1]
  real(chm_real) function ran0(idum)
    real(chm_real) AM
    integer idum,IA,IM,IQ,IR,MASK
    parameter(IA=16807,IM=2147483647,AM=1./IM, &
         IQ=127773,IR=2836,MASK=123459876)
    integer k
    idum=ieor(idum,MASK)
    k=idum/IQ
    idum=IA*(idum-k*IQ)-IR*k
    if(idum < 0) idum=idum+IM
    ran0=AM*idum
    idum=ieor(idum,MASK)
    return
  end  function ran0
  !
  !======================================================================
  subroutine GetTe(NSeg,Vhk,VOrigk,NATOM,X,Y,Z,AMASS,VX,VY,VZ, &
                                ! ... outputs
       Tk,bTk, &
                                ! ... auxillary matrices
       BMV,phik,PBMV,readychild,notdone)
    !
    !     given tree topology, Cartesian coordinates and velocities, 
    !     compute: Te = J^T Mc Vc (see Clore JMR 2001)
    !
    !       BMV = B Mc Vc 
    !       BMV(k) = Sum_(i~k) mi * Col [ (r(i)-r(k,0)) x V(i)  V(i) ] 
    !
    !       PBMV(1) = BMV(1)
    !       PBMV(k+1) = Phi(k+1,k) * PBMV(k) + BMV(k)
    !
    !       Te(k) = H(k) PBMV(k)
    !
    use stream
    use number
    integer NATOM, NSeg
    integer readyChild(NClus)
    real(chm_real)  Vhk(3,NClus),VOrigk(3,NClus), &
         X(NATOM),Y(NATOM),Z(NATOM),AMASS(NATOM), &
         VX(NATOM),VY(NATOM),VZ(NATOM),Tk(NClus),bTk(6,NSeg), &
         BMV(6,NClus),phik(6,6,NClus),PBMV(6,NClus)
    logical notdone(NClus)
    integer j,k,kc,n,n0,done,iseg,fstclus,lstclus,nsegclus
    real(chm_real)  vkn(3),mv(3),rxmv(3),mn
    !
    ! ... BMV = B Mc Vc
    do k=1, NClus
       do j=1, 6
          BMV(j,k)=0d0
       enddo
    enddo
    do n=1, NATOM
       k=ClusID(n)
       n0=OrigPt(k)
       mn=AMASS(n)
       mv(1)=mn*VX(n)
       mv(2)=mn*VY(n)
       mv(3)=mn*VZ(n)
       if(n /= n0) then
          vkn(1)=X(n)-X(n0)
          vkn(2)=Y(n)-Y(n0)
          vkn(3)=Z(n)-Z(n0)
          rxmv(1)=vkn(2)*mv(3)-vkn(3)*mv(2)
          rxmv(2)=vkn(3)*mv(1)-vkn(1)*mv(3)
          rxmv(3)=vkn(1)*mv(2)-vkn(2)*mv(1)
          BMV(1,k)=BMV(1,k)+rxmv(1)
          BMV(2,k)=BMV(2,k)+rxmv(2)
          BMV(3,k)=BMV(3,k)+rxmv(3)
       endif
       BMV(4,k)=BMV(4,k)+mv(1)
       BMV(5,k)=BMV(5,k)+mv(2)
       BMV(6,k)=BMV(6,k)+mv(3)
    enddo
    !
    !     PBMV = Phi BMV, Te(k) = H(k) PBMV(k)
    do iseg=1,NSeg
       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       nsegclus=lstclus-fstclus+1
       do k=fstclus,lstclus
          notdone(k)=.true.
          readychild(k)=0
       enddo
       done=0
       do while(done < nsegclus) ! .true. ==> some nodes not processed yet
          do k=fstclus,lstclus
             if(notdone(k).and.(readychild(k) == NChild(k))) then ! ready 
                do j=1,6
                   PBMV(j,k)=BMV(j,k)
                enddo
                do n=1,NChild(k)
                   kc=Child(n,k)
                   call matxvec('n',6,ONE,phik(1,1,kc),PBMV(1,kc), &
                        ONE,PBMV(1,k))
                enddo
                if(k < lstclus) then ! not base cluster with 6 DOF hinge     
                   call makephik(phik(1,1,k),VOrigk(1,k))
                   Tk(k)=0d0
                   do j=1,3
                      Tk(k)=Tk(k)+Vhk(j,k)*PBMV(j,k)
                   enddo
                   readychild(Parent(k))=readychild(Parent(k))+1
                else
                   do j=1,6
                      bTk(j,iseg)=PBMV(j,lstclus)
                   enddo
                endif
                notdone(k)=.false.
                done=done+1
             endif
          enddo
       enddo
    enddo
    return
  end  subroutine GetTe
  !     
  !======================================================================
  SUBROUTINE CENMSS2(X,Y,Z,VX,VY,VZ,AMASS,XCM,YCM,ZCM, &
       VXCM,VYCM, &
       VZCM,AXCM,AYCM,AZCM,NATOM,IMOVE,QALWRT &
       )
    !
    !     DETERMINES THE CENTER OF MASS, CENTER OF MASS MOTION, AND ANGULAR
    !     MOMENTUM OF THE MOLECULE.
    !
    !     Authors: S. Swaminathan
    !              Bernie Brooks
    !
    !     modified to print out kinetic energy associated with global 
    !     translation and rotation,    by Jianhan Chen (02/05/2004)
    !
    use dimens_fcm
    use number
    use stream
    !
    real(chm_real) X(*),Y(*),Z(*),VX(*),VY(*),VZ(*)
    real(chm_real) XCM,YCM,ZCM,VXCM,VYCM,VZCM,AXCM,AYCM,AZCM
    INTEGER NATOM
    real(chm_real) AMASS(*)
    INTEGER IMOVE(*)
    LOGICAL QALWRT(6)
    !
    !
    real(chm_real) TMASS,AMASSI,EKCM,XI,YI,ZI,VXI,VYI,VZI, &
         OXCM,OYCM,OZCM, &
         XX,XY,XZ,YY,YZ,ZZ,TCM(3,3),U(3,3),SCR(24),AMOM(6),EKCMR
    INTEGER I,J,K,NUMFR
    !
    VXCM=ZERO
    VYCM=ZERO
    VZCM=ZERO
    AXCM=ZERO
    AYCM=ZERO
    AZCM=ZERO
    XCM=ZERO
    YCM=ZERO
    ZCM=ZERO
    TMASS=ZERO
    DO I=1,NATOM
       IF(IMOVE(I) == 0) THEN
          AMASSI=AMASS(I)
          TMASS=TMASS+AMASSI
          XI=X(I)
          YI=Y(I)
          ZI=Z(I)
          VXI=VX(I)
          VYI=VY(I)
          VZI=VZ(I)
          VXCM=VXCM+VXI*AMASSI
          VYCM=VYCM+VYI*AMASSI
          VZCM=VZCM+VZI*AMASSI
          XCM=XCM+XI*AMASSI
          YCM=YCM+YI*AMASSI
          ZCM=ZCM+ZI*AMASSI
          AXCM=AXCM+(YI*VZI-ZI*VYI)*AMASSI
          AYCM=AYCM+(ZI*VXI-XI*VZI)*AMASSI
          AZCM=AZCM+(XI*VYI-YI*VXI)*AMASSI
       ENDIF
    ENDDO
    !
    NUMFR=0
    IF(QALWRT(4)) NUMFR=NUMFR+1
    IF(QALWRT(5)) NUMFR=NUMFR+1
    IF(QALWRT(6)) NUMFR=NUMFR+1
    IF(NUMFR > 0 .AND. NUMFR < 3) THEN
       !     We have a system with axial symmetry.  Determine which axis
       !     and assign centers to appropriate axis.
       IF(.NOT.QALWRT(4)) THEN
          YCM=ZERO
          ZCM=ZERO
       ENDIF
       IF(.NOT.QALWRT(5)) THEN
          XCM=ZERO
          ZCM=ZERO
       ENDIF
       IF(.NOT.QALWRT(6)) THEN
          XCM=ZERO
          YCM=ZERO
       ENDIF
    ENDIF
    !
    AXCM=AXCM-(YCM*VZCM-ZCM*VYCM)/TMASS
    AYCM=AYCM-(ZCM*VXCM-XCM*VZCM)/TMASS
    AZCM=AZCM-(XCM*VYCM-YCM*VXCM)/TMASS
    XCM=XCM/TMASS
    YCM=YCM/TMASS
    ZCM=ZCM/TMASS
    VXCM=VXCM/TMASS
    VYCM=VYCM/TMASS
    VZCM=VZCM/TMASS
    EKCM=VXCM**2+VYCM**2+VZCM**2
    EKCM=EKCM*TMASS/TWO
    !
    XX=ZERO
    XY=ZERO
    XZ=ZERO
    YY=ZERO
    YZ=ZERO
    ZZ=ZERO
    DO I=1,NATOM
       IF(IMOVE(I) == 0) THEN
          XI=X(I)-XCM
          YI=Y(I)-YCM
          ZI=Z(I)-ZCM
          AMASSI=AMASS(I)
          XX=XX+XI*XI*AMASSI
          XY=XY+XI*YI*AMASSI
          XZ=XZ+XI*ZI*AMASSI
          YY=YY+YI*YI*AMASSI
          YZ=YZ+YI*ZI*AMASSI
          ZZ=ZZ+ZI*ZI*AMASSI
       ENDIF
    ENDDO
    !
    AMOM(1)=YY+ZZ
    AMOM(2)=-XY
    AMOM(3)=-XZ
    AMOM(4)=XX+ZZ
    AMOM(5)=-YZ
    AMOM(6)=XX+YY
    !
    CALL DIAGQ(3,3,AMOM,U,SCR(4),SCR(7),SCR(10),SCR(13),SCR(1), &
         SCR(16),SCR(19),SCR(22),0)
    !
    DO I=1,3
       IF(ABS(SCR(I)) < TENM5) SCR(I)=SIGN(TENM5,SCR(I))
       SCR(I)=ONE/SCR(I)
    ENDDO
    !
    DO I=1,3
       DO J=1,3
          TCM(I,J)=0.0
          DO K=1,3
             TCM(I,J)=TCM(I,J)+U(I,K)*U(J,K)*SCR(K)
          ENDDO
       ENDDO
    ENDDO
    OXCM=AXCM*TCM(1,1)+AYCM*TCM(1,2)+AZCM*TCM(1,3)
    OYCM=AXCM*TCM(2,1)+AYCM*TCM(2,2)+AZCM*TCM(2,3)
    OZCM=AXCM*TCM(3,1)+AYCM*TCM(3,2)+AZCM*TCM(3,3)
    EKCMR=0.5d0*(OXCM*AXCM+OYCM*AYCM+OZCM*AZCM)

    IF(PRNLEV >= 2) WRITE(OUTU,101) XCM,YCM,ZCM,VXCM,VYCM,VZCM, &
         AXCM,AYCM,AZCM,EKCM,EKCMR,EKCM+EKCMR
101 FORMAT(/3X,'DETAILS ABOUT CENTRE OF MASS (CM)',/, &
         3X,'POSITION          :',1P,3G17.8,/, &
         3X,'VELOCITY          :',1P,3G17.8,/, &
         3X,'ANGULAR MOMENTUM  :',1P,3G17.8,/, &
         3X,'KINE ENER (T,R,A) :',1P,3G17.8,/)
    !
    !      AMASSI=0d0
    !      do i=1,NATOM
    !         AMASSI=AMASSI+AMASS(i)*(VX(i)*VX(i)+VY(i)*VY(i)+VZ(i)*VZ(i))
    !      enddo
    !      AMASSI=AMASSI*0.5d0
    !      write(70,'(4(G17.8,1X))') EKCM, EKCMR, EKCM+EKCMR,AMASSI
    !
    RETURN
  END SUBROUTINE CENMSS2
  !
  !======================================================================
  subroutine GenSpOp2(NTot,X,Y,Z,Mass,tmk, &
                                !... outputs
       Vhk,VOrigk,VCMk,ITk,ierr)
    !     
    !     given tree topology and current conformation, compute all relavent
    !     coordinate dependant properties: Vhk, VOrigk, VCMk, ITk
    !
    use dimens_fcm
    use stream
    integer NTot
    real(chm_real)  X(NTot),Y(NTot),Z(NTot),Mass(NTot),tmk(NClus)
    integer n,k,i,j,n0,ierr
    real(chm_real)  Vhk(3,*),VOrigk(3,*),VCMk(3,*),ITk(3,3,*)
    real(chm_real)  hl,vkn(3),vknsq(3)
    !
    ierr=0
    !
    !     Vhk and VOrigk
    do k=1, NClus
       if(Parent(k) > 0) then ! not a base cluster
          i=OrigPt(k)
          j=HingePt(k)
          Vhk(1,k)=X(i)-X(j)
          Vhk(2,k)=Y(i)-Y(j)
          Vhk(3,k)=Z(i)-Z(j)
          hl=Vhk(1,k)*Vhk(1,k)+Vhk(2,k)*Vhk(2,k)+Vhk(3,k)*Vhk(3,k)
          hl=sqrt(hl)
          if(hl < 0.000001) then
             if(wrnlev >= 2) write(outu,'(A)') &
                  ' GenSpOp2: invalid hinge with very small norm'
             call wrndie(-2,'<TAMD>','fatal error in GenSpOp2')
             ierr=1
             return
          endif
          Vhk(1,k)=Vhk(1,k)/hl
          Vhk(2,k)=Vhk(2,k)/hl
          Vhk(3,k)=Vhk(3,k)/hl
          !     
          j=OrigPt(Parent(k))
          VOrigk(1,k)=X(i)-X(j)
          VOrigk(2,k)=Y(i)-Y(j)
          VOrigk(3,k)=Z(i)-Z(j)
       endif
    enddo
    !
    !     VCMk and ITk
    do k=1,NClus
       do i=1, 3
          VCMk(i,k)=0d0
          do j=1,i
             ITk(j,i,k)=0d0
          enddo
       enddo
    enddo
    do n=1, NTot
       k=ClusID(n)
       n0=OrigPt(k)       ! origin of kth cluster
       if(n /= n0) then
          vkn(1)=X(n)-X(n0)
          vkn(2)=Y(n)-Y(n0)
          vkn(3)=Z(n)-Z(n0)
          VCMk(1,k)=VCMk(1,k)+Mass(n)*vkn(1)
          VCMk(2,k)=VCMk(2,k)+Mass(n)*vkn(2)
          VCMk(3,k)=VCMk(3,k)+Mass(n)*vkn(3)
          !
          vknsq(1)=Mass(n)*vkn(1)*vkn(1)
          vknsq(2)=Mass(n)*vkn(2)*vkn(2)
          vknsq(3)=Mass(n)*vkn(3)*vkn(3)
          ITk(1,1,k)=ITk(1,1,k)+vknsq(2)+vknsq(3)
          ITk(2,2,k)=ITk(2,2,k)+vknsq(1)+vknsq(3)
          ITk(3,3,k)=ITk(3,3,k)+vknsq(1)+vknsq(2)
          ITk(1,2,k)=ITk(1,2,k)-Mass(n)*vkn(1)*vkn(2)
          ITk(1,3,k)=ITk(1,3,k)-Mass(n)*vkn(1)*vkn(3)
          ITk(2,3,k)=ITk(2,3,k)-Mass(n)*vkn(2)*vkn(3)
       endif
    enddo
    do k=1, NClus
       hl=1d0/tmk(k)
       VCMk(1,k)=VCMk(1,k)*hl
       VCMk(2,k)=VCMk(2,k)*hl
       VCMk(3,k)=VCMk(3,k)*hl
       ITk(2,1,k)=ITk(1,2,k)
       ITk(3,1,k)=ITk(1,3,k)
       ITk(3,2,k)=ITk(2,3,k)
    enddo
    return
  end   subroutine GenSpOp2
  !
  !======================================================================
  subroutine UpdateCor2(NTot,X,Y,Z,NSeg, &
       Vhk,VOrigk,Theta,qknseg,Rki, &
                                ! ... auxillary arrays 
       qBase,Rk,ready,notdone)
    !
    !     given torsion increments (theta) and new base position (qknseg,Rki)
    !     update the coordinates of the subtree
    !
    use number
    integer NTot, NSeg
    real(chm_real)  X(NTot),Y(NTot),Z(NTot),Vhk(3,NClus), &
         Vorigk(3,NClus), &
         Theta(NClus),qBase(3,NClus),Rk(3,3,NClus),Rki(3,3,NSeg), &
         qknseg(3,NSeg)
    integer ready(NClus)
    logical notdone(NClus)
    ! ... local variables
    integer k,n,kc,ik0,ierr,done,iseg,fstclus,lstclus,nsegclus
    real(chm_real)  qkn(3),vk(3)
    !
    do iseg=1,NSeg
       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       nsegclus=lstclus-fstclus+1

       done=0
       do k=fstclus,lstclus-1
          ready(k)=0
          notdone(k)=.true.
       enddo
       do k=1,3
          qkn(k)=qknseg(k,iseg)
          qBase(k,lstclus)=qkn(k)
          do n=1,3
             Rk(n,k,lstclus)=Rki(n,k,iseg)
          enddo
       enddo
       ready(lstclus)=1
       notdone(lstclus)=.true.
       do while(done < nsegclus)
          do k=lstclus,fstclus,-1 
             if(ready(k) == 1.and.notdone(k)) then ! ready to update
                !
                !     update the coordinates of current cluster
                ik0=OrigPt(k)    
                do n=1, NTot
                   if(ClusID(n) == k.and.n /= ik0) then
                      vk(1)=X(n)-X(ik0)
                      vk(2)=Y(n)-Y(ik0)
                      vk(3)=Z(n)-Z(ik0)
                      call matxvec('n',3,ONE,Rk(1,1,k),vk,ZERO,qkn)
                      X(n)=qBase(1,k)+qkn(1)
                      Y(n)=qBase(2,k)+qkn(2)
                      Z(n)=qBase(3,k)+qkn(3)
                   endif
                enddo
                X(ik0)=qBase(1,k)
                Y(ik0)=qBase(2,k)
                Z(ik0)=qBase(3,k)
                !
                !     get the children ready: compute rotation of children and 
                !                             displacement of their origins
                !
                do n=1,NChild(k)
                   kc=Child(n,k)
                   call rotmat(Vhk(1,kc),Theta(kc),Rki,ierr) ! get Rk^(i)
                   call matxmat('n','n',3,ONE,Rk(1,1,k),Rki, &
                        ZERO,Rk(1,1,kc))                     ! Rk=Rkp*Rki
                   !
                   call matxvec('n',3,ONE,Rk(1,1,k),VOrigk(1,kc),ZERO,qkn)
                   qBase(1,kc)=X(ik0)+qkn(1)
                   qBase(2,kc)=Y(ik0)+qkn(2)
                   qBase(3,kc)=Z(ik0)+qkn(3)
                   ready(kc)=1
                enddo
                done=done+1
                notdone(k)=.false.
             endif
          enddo
       enddo
       !
    enddo                     ! do iseg=1,NSeg
    return
  end subroutine UpdateCor2
  !
  !======================================================================
  subroutine UpdateBase(NTot,X,Y,Z,NClus,OrigPt,Quat,delt, &
       bddTheta,bSpVelOld,bSpVelNew,SpVel,qBase,RBase,ierr)
    !
    !     Give base spatial acceleration, update the body
    !     orientation using quaternion representation (Allen&Tildesley) and the
    !     spatial velocity with modified FIQA
    !
    use stream
    use number
    integer NTot,NClus,OrigPt(NClus),ierr
    real(chm_real) X(NTot),Y(NTot),Z(NTot),Quat(0:3),delt, &
         RBase(3,3),qBase(3)
    real(chm_real) bddTheta(6),bSpVelOld(6),bSpVelNew(6), &
         SpVel(6,NClus)
    !
    integer j,iter,maxits
    parameter(maxits=10)      ! maximum number of iterations for FIQA
    real(chm_real) qn(0:3),qold(0:3),dq(0:3),dqn(0:3),wb(3),rotm(3,3), &
         rotm2(3,3),eps,convergence
    parameter(convergence=1d-8) ! convergence criteria for FIQA
    !
    !     get rotational matrix: A(t)=A(q(t))
    call q2rot(Quat,rotm)
    !
    !     compute angular velocity in body-fixed frame: wb(t)=A(t)*ws(t)
    call matxvec('n',3,ONE,rotm,SpVel(1,NClus),ZERO,wb)
    !
    !     dq(t)=Q(t)*(0,wb(t))
    dq(0)=0.5d0*(-Quat(1)*wb(1)-Quat(2)*wb(2)-Quat(3)*wb(3))
    dq(1)=0.5d0*( Quat(0)*wb(1)-Quat(3)*wb(2)+Quat(2)*wb(3))
    dq(2)=0.5d0*( Quat(3)*wb(1)+Quat(0)*wb(2)-Quat(1)*wb(3))
    dq(3)=0.5d0*(-Quat(2)*wb(1)+Quat(1)*wb(2)+Quat(0)*wb(3))
    !
    !     update base spatial velicity: bSpVelOld(t+delt/2),SpVel(t+delt)
    do j=1,6
       bSpVelNew(j)=bSpVelOld(j)+delt*bddTheta(j)
       SpVel(j,NClus)=1.5d0*bSpVelNew(j)-0.5d0*bSpVelOld(j)
       bSpVelOld(j)=bSpVelNew(j)
    enddo
    !
    !     initial estimation of q(t+delt)
    qn(0)=Quat(0)+delt*dq(0)
    qn(1)=Quat(1)+delt*dq(1)
    qn(2)=Quat(2)+delt*dq(2)
    qn(3)=Quat(3)+delt*dq(3)
    !
    !     Fincham's implicit quaternion algorithm
    !
    eps=1d0
    iter=0
    do while(eps > convergence.and.iter <= maxits)
       !     compute wb(t+delt)
       call q2rot(qn,RBase)
       call matxvec('n',3,ONE,RBase,SpVel(1,NClus),ZERO,wb)
       !
       !     compute dq(t+delt)=Q(t+delt)*(0,wb(t+delt))
       dqn(0)=-0.5d0*(qn(1)*wb(1)+qn(2)*wb(2)+qn(3)*wb(3))
       dqn(1)=0.5d0*( qn(0)*wb(1)-qn(3)*wb(2)+qn(2)*wb(3))
       dqn(2)=0.5d0*( qn(3)*wb(1)+qn(0)*wb(2)-qn(1)*wb(3))
       dqn(3)=0.5d0*(-qn(2)*wb(1)+qn(1)*wb(2)+qn(0)*wb(3))

       qold(0)=qn(0)
       qold(1)=qn(1)
       qold(2)=qn(2)
       qold(3)=qn(3)
       !
       !     update q(t+delt)
       qn(0)=Quat(0)+0.5d0*delt*(dq(0)+dqn(0))
       qn(1)=Quat(1)+0.5d0*delt*(dq(1)+dqn(1))
       qn(2)=Quat(2)+0.5d0*delt*(dq(2)+dqn(2))
       qn(3)=Quat(3)+0.5d0*delt*(dq(3)+dqn(3))

       eps=sqrt( (qn(0)-qold(0))**2 + (qn(1)-qold(1))**2+ &
            (qn(2)-qold(2))**2 + (qn(3)-qold(3))**2 )               
       iter=iter+1
    enddo
    !
    !     renormalize Quat()
    eps=qn(0)*qn(0)+qn(1)*qn(1)+ &
         qn(2)*qn(2)+qn(3)*qn(3)
    eps=sqrt(eps)
    if(eps < 0.000001) then ! something must be wrong
       if(wrnlev >= 2) write(outu,'(A)') &
            ' UpdateBase: norm of new quaternion is very small!'
       call wrndie(-1,'<TAMD>','fatal error in UpdateBase')
       ierr=1
       return
    endif
    Quat(0)=qn(0)/eps
    Quat(1)=qn(1)/eps
    Quat(2)=qn(2)/eps
    Quat(3)=qn(3)/eps
    !
    j=OrigPt(NClus)
    qBase(1)=delt*bSpVelNew(4)+X(j) ! new position of base origin
    qBase(2)=delt*bSpVelNew(5)+Y(j)
    qBase(3)=delt*bSpVelNew(6)+Z(j)
    call q2rot(Quat,rotm2)   ! new orientation A(t+delt)
    call matxmat('t','n',3,ONE,rotm2,rotm,ZERO,RBase) ! net rotation
    !
    return
  end subroutine UpdateBase
  !
  !======================================================================
  subroutine UpdateSpVel2(NSeg,Vhk,VOrigk,dTheta,SpVel)
    !
    !     Given torsional speed and the base spatial velocity, 
    !     update the spatial velocity of the whole tree
    !
    integer NSeg
    real(chm_real) Vhk(3,NClus),VOrigk(3,NClus),dTheta(NClus)
    ! ... outputs
    real(chm_real) SpVel(6,NClus)
    ! ... local variables
    integer k,kp,done,iseg,fstclus,lstclus,nsegclus
    real(chm_real)  vv(3)
    logical notdone(NClus)
    !
    do iseg=1,NSeg
       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       nsegclus=lstclus-fstclus+1
       do k=fstclus,lstclus-1
          notdone(k)=.true.
       enddo
       done=1
       notdone(lstclus)=.false.
       do while(done < nsegclus)
          do k=lstclus-1,1,-1
             if(notdone(k).and..not.notdone(Parent(k))) then !ready to be done
                kp=Parent(k)
                !     w(k)=w(kp)+h(k)*theta'
                SpVel(1,k)=SpVel(1,kp)+dTheta(k)*Vhk(1,k) 
                SpVel(2,k)=SpVel(2,kp)+dTheta(k)*Vhk(2,k)
                SpVel(3,k)=SpVel(3,kp)+dTheta(k)*Vhk(3,k)
                !     v(k)=v(kp)+w(kp)xVOrigk(k)
                vv(1)=SpVel(2,kp)*VOrigk(3,k)-SpVel(3,kp)*VOrigk(2,k)
                vv(2)=SpVel(3,kp)*VOrigk(1,k)-SpVel(1,kp)*VOrigk(3,k)
                vv(3)=SpVel(1,kp)*VOrigk(2,k)-SpVel(2,kp)*VOrigk(1,k)
                SpVel(4,k)=SpVel(4,kp)+vv(1)
                SpVel(5,k)=SpVel(5,kp)+vv(2)
                SpVel(6,k)=SpVel(6,kp)+vv(3)
                !
                done=done+1
                notdone(k)=.false.
             endif
          enddo
       enddo
    enddo
    return
  end subroutine UpdateSpVel2
  !
  !======================================================================
  subroutine GetScalVel(NTot,X,Y,Z, &
       SpVel,VX,VY,VZ)
    !
    !     Given spatial velocities of clusters, compute scalar (Cartesian)
    !     velocities of all atoms
    !
    integer NTot
    real(chm_real)  X(NTot),Y(NTot),Z(NTot)
    real(chm_real)  SpVel(6,NClus),VX(NTot),VY(NTot),VZ(NTot)
    ! ... local variables
    integer k, n, nk0
    real(chm_real) vn(3), vv(3)
    do k=1, NClus
       nk0=OrigPt(k)
       VX(nk0)=SpVel(4,k)
       VY(nk0)=SpVel(5,k)
       VZ(nk0)=SpVel(6,k)
       do n=1, NTot
          if(ClusID(n) == k) then
             if(n /= nk0) then
                vn(1)=X(n)-X(nk0)
                vn(2)=Y(n)-Y(nk0)
                vn(3)=Z(n)-Z(nk0)
                vv(1)=SpVel(2,k)*vn(3)-SpVel(3,k)*vn(2)
                vv(2)=SpVel(3,k)*vn(1)-SpVel(1,k)*vn(3)
                vv(3)=SpVel(1,k)*vn(2)-SpVel(2,k)*vn(1)
                VX(n)=VX(nk0)+vv(1)
                VY(n)=VY(nk0)+vv(2)
                VZ(n)=VZ(nk0)+vv(3)
             endif
             !               VK(n)=VX(n)*VX(n)+VY(n)*VY(n)+VZ(n)*VZ(n)
             !               VK(n)=VK(n)*Mass(n)
          endif
       enddo
    enddo
    return
  end subroutine GetScalVel
  !
  !======================================================================
  subroutine SPTAD2( &
                                ! ... inputs (unchanged)
       NTot,X,Y,Z,DX,DY,DZ,NSeg, &
       Vhk,VOrigk,VCMk,tmk,ITk,dTheta,SpVel,Tk,bTk,compFckAkBk, &
                                ! ... outputs (changed)
       ddTheta,bddTheta, &
                                ! ... auxillary matrices (changed)
       phik,Dk,FCk,ak,bk,Gk,epsilon,zkplus,Pkplus,alphak, &
       readychild,notdone)
    !
    !     given force in cartesian space and everything about the system
    !     compute the torsional acceleration based on the recursive algorithm
    !     of Jain et. al, JCP 106, 258-268 (1993)
    !
    !     version 2: handles a single tree (Jianhan Chen, DEC-2003)
    !     version 3: handles multiple tree (Jianhan Chen, Feb-2007)
    !
    !     arguments:
    !     ----------
    !
    !     NTOT:     number of atoms
    !     X,Y,Z:    current Cartesian coordinates
    !     DX,DY,DZ: gradient of potential ("Cartesian foreces")
    !     NClus:    number of clusters of corresponding tree topology
    !
    !     ClusID,OrigPt,Parent,NClild,Child,MAXCHILD: defines the tree (see tamd.f90)
    !     
    !     Vhk,VOrigk,VCMk,tmk,ITk: current status of tree (see tadcntrl())
    !
    !     dTheta:  torsional velocities
    !     Tk,bTk:  projection of spatial force along hinges
    !     compFckAkBk:  wether to compute Fck, ak and bk (.true. for dynamics 
    !                   and minimization; .false. for velocity mapping)
    !     ddTheta,bddTheta: torsion accelerations
    !
    !     All the rest are auxillary matrices for Jain's recursive algorithm
    !
    !
    use stream
    use number
    ! ...input (unchanged)
    integer NTot, NSeg
    real(chm_real)  tmk(NClus),Vhk(3,NClus),VOrigk(3,NClus), &
         VCMk(3,NClus), &
         ITk(3,3,NClus),dTheta(NClus),SpVel(6,NClus),Tk(NClus), &
         bTk(6,NSeg)
    real(chm_real)  X(NTot),Y(NTot),Z(NTot),DX(NTot),DY(NTot),DZ(NTot)
    ! ...ouptut (changed)
    real(chm_real)  ddTheta(NClus),bddTheta(6,NSeg)
    ! ...SP TAD auxillary matrices
    real(chm_real) phik(6,6,NClus),Dk(NClus),FCk(6,NClus),  &
         ak(6,NClus),  &
         bk(6,NClus),Gk(6,NClus),epsilon(NClus),zkplus(6,NClus), &
         zk(6),Pk(6,6),Pkplus(6,6,NClus),alphak(6,NClus),alphakplus(6)
    ! ...local temporary variables
    integer n,k,i,j,kc,kp,nk0,INFO,done,readychild(NClus)
    integer iseg,fstclus,lstclus,nsegclus
    real(chm_real)  vk(3),mat6(6,6),mat3(3,3),r1,r2,r3
    logical compFckAkBk,notdone(NClus)
    !
    do k=1,NClus
       do j=1,6
          FCk(j,k)=0d0
          ak(j,k)=0d0
          bk(j,k)=0d0
       enddo
    enddo
    !
    if(compFckAkBk) then      ! compute Fck, ak, bk
       !
       !     f^(c)_k: Cartesian spatial force
       !
       do n=1,NTot
          k=ClusID(n)         ! cluster id
          nk0=OrigPt(k)       ! origin of kth cluster
          vk(1)=X(n)-X(nk0)
          vk(2)=Y(n)-Y(nk0)
          vk(3)=Z(n)-Z(nk0)
          FCk(1,k)=FCk(1,k)+vk(2)*DZ(n)-vk(3)*DY(n)
          FCk(2,k)=FCk(2,k)+vk(3)*DX(n)-vk(1)*DZ(n)
          FCk(3,k)=FCk(3,k)+vk(1)*DY(n)-vk(2)*DX(n)
          FCk(4,k)=FCk(4,k)+DX(n)
          FCk(5,k)=FCk(5,k)+DY(n)
          FCk(6,k)=FCk(6,k)+DZ(n)
       enddo
       !
       !     ak: Coriolis term   (6x1)
       !     bk: gyroscopic term (6x1)
       !     
       do k=1, NClus
          kp=Parent(k)
          if(kp > 0) then    ! not base cluster
             vk(1)=SpVel(2,k)*Vhk(3,k)-SpVel(3,k)*Vhk(2,k)
             vk(2)=SpVel(3,k)*Vhk(1,k)-SpVel(1,k)*Vhk(3,k)
             vk(3)=SpVel(1,k)*Vhk(2,k)-SpVel(2,k)*Vhk(1,k)
             ak(1,k)=dTheta(k)*vk(1)
             ak(2,k)=dTheta(k)*vk(2)
             ak(3,k)=dTheta(k)*vk(3)
             vk(1)=SpVel(4,k)-SpVel(4,kp)
             vk(2)=SpVel(5,k)-SpVel(5,kp)
             vk(3)=SpVel(6,k)-SpVel(6,kp)
             ak(4,k)=SpVel(2,kp)*vk(3)-SpVel(3,kp)*vk(2)
             ak(5,k)=SpVel(3,kp)*vk(1)-SpVel(1,kp)*vk(3)
             ak(6,k)=SpVel(1,kp)*vk(2)-SpVel(2,kp)*vk(1)
          endif
          !
          call matxvec('n',3,ONE,ITk(1,1,k),SpVel(1,k),ZERO,vk)
          bk(1,k)=SpVel(2,k)*vk(3)-SpVel(3,k)*vk(2)
          bk(2,k)=SpVel(3,k)*vk(1)-SpVel(1,k)*vk(3)
          bk(3,k)=SpVel(1,k)*vk(2)-SpVel(2,k)*vk(1)
          r1=VCMk(1,k)*SpVel(1,k)+VCMk(2,k)*SpVel(2,k) &
               +VCMk(3,k)*SpVel(3,k)
          r2=SpVel(1,k)*SpVel(1,k)+SpVel(2,k)*SpVel(2,k) &
               +SpVel(3,k)*SpVel(3,k)
          bk(4,k)=tmk(k)*(r1*SpVel(1,k)-r2*VCMk(1,k))
          bk(5,k)=tmk(k)*(r1*SpVel(2,k)-r2*VCMk(2,k))
          bk(6,k)=tmk(k)*(r1*SpVel(3,k)-r2*VCMk(3,k))
       enddo
       !
    endif                     ! if(compFckAkBk)
    !
    !     compute Pk (accumlate inertia), zk (residual spatial force)
    !     (==> efficiency on matrix multiplications can be improved!!!)
    !     (==> efficiency might be improved by expand small do-loops)
    !
    do iseg=1,NSeg
       fstclus=NCSeg(iseg)+1
       lstclus=NCSeg(iseg+1)
       nsegclus=lstclus-fstclus+1
       !
       if(Parent(lstclus) > 0) then
          call wrndie(-3,'<TAMD>', &
               'Base cluster is not the last cluster')
          return
       endif
       !
       do k=fstclus,lstclus
          notdone(k)=.true.
          readychild(k)=0
       enddo
       done=0
       do while(done < nsegclus)   ! .true. ==> some nodes not processed yet
          do k=fstclus,lstclus
             if(notdone(k).and.(readychild(k) == NChild(k))) then ! ready 
                !     
                !     Pk: sum_kc { phi(kc)*Pkplus(kc)*phi(kc)^T} + Mk
                call makepk(Pk,tmk(k),VCMk(1,k),ITk(1,1,k))
                do n=1,NChild(k)
                   kc=Child(n,k)
                   call matxmat('n','t',6,ONE,Pkplus(1,1,kc), &
                        phik(1,1,kc),ZERO,mat6)
                   call matxmat('n','n',6,ONE,phik(1,1,kc), &
                        mat6,ONE,Pk)
                enddo
                !     
                !     zk: sum_kc { phi(kc)*zkplus(kc)} + Pk*ak + bk + FCk
                do j=1,6
                   zk(j)=bk(j,k)+FCk(j,k)
                enddo
                call matxvec('n',6,ONE,Pk,ak(1,k),ONE,zk)
                do n=1, NChild(k)
                   kc=Child(n,k)
                   call matxvec('n',6,ONE,phik(1,1,kc),zkplus(1,kc), &
                        ONE,zk)
                enddo
                !     
                if(k /= lstclus) then ! not base cluster with 6 DOF hinge
                   !     
                   call makephik(phik(1,1,k),VOrigk(1,k))
                   !     
                   !     Dk=Hk*Pk*Hk^T = hk^T*Pk11*hk
                   r1=Vhk(1,k)*Vhk(2,k) ! xy
                   r2=Vhk(1,k)*Vhk(3,k) ! xz
                   r3=Vhk(2,k)*Vhk(3,k) ! yz
                   Dk(k)=Pk(1,1)*Vhk(1,k)*Vhk(1,k) + &
                        Pk(2,2)*Vhk(2,k)*Vhk(2,k) + &
                        Pk(3,3)*Vhk(3,k)*Vhk(3,k) +  &
                        r1*(Pk(1,2)+Pk(2,1)) + &
                        r2*(Pk(1,3)+Pk(3,1)) + &
                        r3*(Pk(2,3)+Pk(3,2))
                   if(Dk(k) /= 0) then
                      Dk(k) = 1d0/Dk(k)
                   else
                      Dk(k) = 0d0
                   endif
                   !     
                   !     GK=Pk*hk^T/Dk=col[P11*hk P21*hk]/Dk
                   do j=1,3
                      do i=1,3
                         mat3(j,i)=Pk(j,i) ! P11
                      enddo
                   enddo
                   call matxvec('n',3,Dk(k),mat3,Vhk(1,k),ZERO,Gk(1,k))
                   do j=1,3
                      do i=1,3
                         mat3(j,i)=Pk(3+j,i) ! P21
                      enddo
                   enddo
                   call matxvec('n',3,Dk(k),mat3,Vhk(1,k),ZERO,Gk(4,k))
                   !     
                   !     Pkplus=(1-Gk*Hk)*Pk
                   do j=1,6
                      do i=1,3
                         mat6(j,i)=-Gk(j,k)*Vhk(i,k)
                         mat6(j,i+3)=0d0
                      enddo
                      mat6(j,j)=mat6(j,j)+1d0
                   enddo
                   call matxmat('n','n',6,ONE,mat6,Pk,ZERO,Pkplus(1,1,k))
                   !     
                   !     epsilonk=Tk-Hk*Zk
                   epsilon(k)=Tk(k)-(Vhk(1,k)*zk(1)+ &
                        Vhk(2,k)*zk(2)+Vhk(3,k)*zk(3))
                   !     
                   !     zkplus=zk+epsilonk*Gk
                   do j=1,6
                      zkplus(j,k)=zk(j)+epsilon(k)*Gk(j,k)
                   enddo
                   !     
                   !     record processing log:
                   readychild(Parent(k))=readychild(Parent(k))+1
                   !     
                else             ! base cluster
                   !     
                   !     bddTheta=Dk^(-1) * epsilonk = Pk^(-1) * epsilonk
                   do j=1,6      ! epsilonk=Tk-Hk*zk=Tk-zk
                      bddTheta(j,iseg)=bTk(j,iseg)-zk(j)
                   enddo
                   call GAUSSJt(6,Pk,bddTheta(1,iseg),info)
                   if(info /= 0) then
                      if(wrnlev >= 2) write(outu,'(A,I3)')  &
                           ' SPTAD2: GAUSSJt exits with error = ',info
                      call wrndie(-2,'<TAMD>','fatal error in GAUSSJt')
                      return
                   endif
                   !     
                endif            ! if(k /= nsegclus)
                !     
                !     record processing flag and count
                notdone(k)=.false.
                done=done+1
             endif               ! if(notdone(k).and...)
          enddo                  ! do k=fstclus,lstclus
       enddo                     ! do while(more)
       !
       !     compute ddTheta: torsional acceleration
       !
       !      base cluster: alphak=ddTheta(k)+ak
       do j=1,6
          alphak(j,lstclus)=bddTheta(j,iseg)+ak(j,lstclus)
       enddo
       done=1
       notdone(lstclus)=.false.
       do k=fstclus,lstclus-1
          notdone(k)=.true.
       enddo
       !      
       do while(done < nsegclus)
          do k=lstclus-1,1,-1
             if(notdone(k).and..not.notdone(Parent(k))) then ! ready to be done
                kp=Parent(k)
                !     alphakplus=phi(k)^T*alpha(k+1)
                call matxvec('T',6,ONE,phik(1,1,k),alphak(1,kp), &
                     ZERO,alphakplus)
                !     ddTheta(k)=epsilonk/Dk-Gk^T*alphakplus
                ddTheta(k)=epsilon(k)*Dk(k)
                do j=1,6
                   ddTheta(k)=ddTheta(k)-Gk(j,k)*alphakplus(j)
                enddo
                !     aplhak=alphakplus+Hk^T*ddTheta(k)+ak
                do j=1,3
                   alphak(j,k)=ddTheta(k)*Vhk(j,k)+alphakplus(j)+ak(j,k)
                   alphak(j+3,k)=alphakplus(j+3)+ak(j+3,k)
                enddo
                done=done+1
                notdone(k)=.false.
             endif
          enddo
       enddo
       !
    enddo                     !do iseg=1,NSeg
    !
    return
  end subroutine SPTAD2
  !
  !======================================================================
  subroutine makepk(Pk,mk,qck,ITk)
    !        { ITk       mk*A(Qck)}               { 0  -vz  vy}
    !     Pk=|                    |    where A(v)=| vz  0  -vx|
    !        {-mk*A(Qck)   mk*1   }               {-vy  vx  0 }
    !
    integer i, j
    real(chm_real) mk, qck(3), ITk(3,3), Pk(6,6), x, y, z
    do i=1, 3
       do j=1, 3
          Pk(j,i)=ITk(j,i)
          !     Pk(3+j,3+i)=ITk(j,i)
          Pk(3+j,3+i)=0d0
       enddo
       j=3+i
       Pk(j,j)=mk
       Pk(i,j)=0d0
       Pk(j,i)=0d0
    enddo
    x=mk*qck(1)
    y=mk*qck(2)
    z=mk*qck(3)
    Pk(1,5)=-z
    Pk(1,6)=y
    Pk(2,4)=z
    Pk(2,6)=-x
    Pk(3,4)=-y
    Pk(3,5)=x
    Pk(5,1)=-z
    Pk(6,1)=y
    Pk(4,2)=z
    Pk(6,2)=-x
    Pk(4,3)=-y
    Pk(5,3)=x
    return
  end subroutine makepk
  !
  !======================================================================
  subroutine makephik(phik,lk)
    !                            {  1    A(l(k+1,k)) }
    !     make phi(k)=phi(k+1,k)=|                   |
    !                            {  0        1      }
    integer i, j, m, n
    real(chm_real) phik(6,6),lk(3)      
    do i=1, 3
       j=3+i
       phik(i,i)=1d0
       phik(j,j)=1d0
       phik(i,j)=0d0
       phik(j,i)=0d0
       do j=1, i-1
          m=3+i
          n=3+j
          phik(j,i)=0d0
          phik(i,j)=0d0
          phik(m,n)=0d0
          phik(n,m)=0d0
          phik(m,j)=0d0
          phik(n,i)=0d0
       enddo
    enddo
    phik(1,5)=-lk(3)
    phik(1,6)=lk(2)
    phik(2,4)=lk(3)
    phik(2,6)=-lk(1)
    phik(3,4)=-lk(2)
    phik(3,5)=lk(1)
    return
  end subroutine makephik
  !
  !======================================================================
  subroutine rotmat(u, theta, mat, ierr)
    !
    !     generate rotation matrix for given unit axis vector and rotation angle
    !
    !     u=[ux,uy,uz]: unit vector in axis orientation (unchanged on exit)
    !     theta:        angle given in radian (unchanged on exit)
    !     mat:          3x3 rotation matrix (output)
    !
    integer ierr, i, j
    real(chm_real) u(3), theta, mat(3,3)
    real(chm_real) x, y, z, w, cost, sint, xsq, ysq, zsq, wsq
    !
    ierr=0
    if(theta == 0d0) then
       do i=1, 3  
          mat(i,i)=1d0
          do j=1, i-1
             mat(j,i)=0d0
             mat(i,j)=0d0
          enddo
       enddo
       return
    endif
    !
    cost=cos(0.5d0*theta)
    sint=sin(0.5d0*theta)
    x=sint*u(1)
    y=sint*u(2)
    z=sint*u(3)
    w=cost
    xsq=x*x
    ysq=y*y
    zsq=z*z
    wsq=w*w
    !
    mat(1,1)=wsq+xsq-ysq-zsq
    mat(1,2)=2*(x*y-w*z)
    mat(1,3)=2*(x*z+w*y)
    mat(2,1)=2*(x*y+w*z)
    mat(2,2)=wsq-xsq+ysq-zsq
    mat(2,3)=2*(y*z-w*x)
    mat(3,1)=2*(x*z-w*y)
    mat(3,2)=2*(y*z+w*x)
    mat(3,3)=wsq-xsq-ysq+zsq
    return
  end subroutine rotmat
  !
  !======================================================================
  subroutine q2rot(q, A)
    !
    !     given a quaternion, make a rotation matrix
    !
    real(chm_real) q(0:3),A(3,3), &
         q0sq,q1sq,q2sq,q3sq,q12,q03,q13,q02,q23,q01
    q0sq=q(0)*q(0)
    q1sq=q(1)*q(1)
    q2sq=q(2)*q(2)
    q3sq=q(3)*q(3)
    q12=q(1)*q(2)
    q03=q(0)*q(3)
    q13=q(1)*q(3)
    q02=q(0)*q(2)
    q23=q(2)*q(3)
    q01=q(0)*q(1)
    !
    A(1,1)=q0sq+q1sq-q2sq-q3sq
    A(2,2)=q0sq-q1sq+q2sq-q3sq
    A(3,3)=q0sq-q1sq-q2sq+q3sq
    A(1,2)=2*(q12+q03)
    A(2,1)=2*(q12-q03)
    A(1,3)=2*(q13-q02)
    A(3,1)=2*(q13+q02)
    A(2,3)=2*(q23+q01)
    A(3,2)=2*(q23-q01)
    return
  end subroutine q2rot
  !
  !======================================================================
  subroutine matxvec(TRANS, N, ALPHA, A, X, BETA, Y)
    !
    !     square matrix vector multiplication (simplified based on BLAS DGEMV)
    !
    !       y := alpha*TRANS(A) + beta*y        TRANS(A) = A or A'
    !
    use stream
    integer N, i, j
    character(len=1) trans
    real(chm_real) alpha, beta, A(N,N), X(N), Y(N), temp
    real(chm_real) ONE, ZERO
    PARAMETER  ( ONE = 1.0D+0, ZERO = 0.0D+0 )
    !
    if(N < 0) then
       if(wrnlev >= 0) write(outu,'(A)')  &
            ' MATXVEC: invalid input: N<0 (in TAMD)'
       return
    endif
    !
    !     Quick return if possible.
    IF( (N == 0).OR.((ALPHA.EQ.ZERO ).AND.( BETA.EQ.ONE ))) RETURN
    !      
    if(beta == zero) then
       do i=1,N
          Y(i)=ZERO
       enddo
    else if(beta /= ONE) then
       do i=1,N
          Y(i)=beta*Y(i)
       enddo
    endif
    !
    if(alpha == ZERO) then
       return
    else
       if(trans == 't'.or.trans.eq.'T') then
          do j=1,N
             temp=ZERO
             do i=1,N
                temp=temp+A(i,j)*X(i)
             enddo
             Y(j)=Y(j)+alpha*temp
          enddo
       else
          do j=1,N
             if(X(J) /= ZERO) then
                temp=alpha*X(j)
                do i=1,N
                   Y(I)=Y(I)+temp*A(I,J)
                enddo
             endif
          enddo
       endif
    endif
    !
    return
  end subroutine matxvec
  !
  !======================================================================
  subroutine matxmat(TRANSA, TRANSB, N, ALPHA, A, B, BETA, C)
    !
    !     square matrix multiplication (simplified from BLAS DGEMM)
    !
    !       C = TRANSA(A) * TRANSB(B) + BETA * C
    !
    use stream
    character(len=1) transa, transb
    integer i, j, l, N
    real(chm_real) A(N,N), B(N,N), C(N,N), ALPHA, BETA, temp
    real(chm_real) ONE, ZERO
    PARAMETER  ( ONE = 1.0D+0, ZERO = 0.0D+0 )
    logical taflag, tbflag
    !
    if(N < 0) then
       if(wrnlev >= 0) write(outu,'(A)')  &
            ' MATXMAT: invalid input: N<0 (in TAMD)'
       return
    endif
    !
    if(N == 0) return
    !
    taflag=(transa == 't'.or.transa.eq.'T') ! transpose A
    tbflag=(transb == 't'.or.transb.eq.'T') ! transpose B
    !
    if(taflag) then
       if(tbflag) then            ! C = A'*B' + Beta*C
          DO J = 1, N
             DO I = 1, N
                TEMP = ZERO
                DO L = 1, N
                   TEMP = TEMP + A( L, I )*B( J, L )
                ENDDO
                IF( BETA == ZERO )THEN
                   C( I, J ) = ALPHA*TEMP
                ELSE
                   C( I, J ) = ALPHA*TEMP + BETA*C( I, J )
                END IF
             ENDDO
          ENDDO
       else                   ! C = A'*B  + Beta*C
          DO J = 1, N
             DO I = 1, N
                TEMP = ZERO
                DO L = 1, N
                   TEMP = TEMP + A( L, I )*B( L, J )
                ENDDO
                IF( BETA == ZERO )THEN
                   C( I, J ) = ALPHA*TEMP
                ELSE
                   C( I, J ) = ALPHA*TEMP + BETA*C( I, J )
                END IF
             ENDDO
          ENDDO
       endif
    else
       if(tbflag) then            ! C = A*B' + Beta*C
          DO J = 1, N
             IF( BETA == ZERO )THEN
                DO I = 1, N
                   C( I, J ) = ZERO
                ENDDO
             ELSE IF( BETA /= ONE )THEN
                DO I = 1, N
                   C( I, J ) = BETA*C( I, J )
                ENDDO
             END IF
             DO L = 1, N
                IF( B( J, L ) /= ZERO )THEN
                   TEMP = ALPHA*B( J, L )
                   DO I = 1, N
                      C( I, J ) = C( I, J ) + TEMP*A( I, L )
                   ENDDO
                END IF
             ENDDO
          ENDDO
       else                   ! C = A*B  + Beta*C
          DO J = 1, N
             IF( BETA == ZERO )THEN
                DO I = 1, N
                   C( I, J ) = ZERO
                ENDDO
             ELSE IF( BETA /= ONE )THEN
                DO I = 1, N
                   C( I, J ) = BETA*C( I, J )
                ENDDO
             END IF
             DO L = 1, N
                IF( B( L, J ) /= ZERO )THEN
                   TEMP = ALPHA*B( L, J )
                   DO I = 1, N
                      C( I, J ) = C( I, J ) + TEMP*A( I, L )
                   ENDDO
                END IF
             ENDDO
          ENDDO
       endif
    endif
    !
    return
  end subroutine matxmat
  !
  !======================================================================
  SUBROUTINE GAUSSJt(N,A,X,INFO)
    !
    !     gauss-jordan elimination (originally from Numerical Recipe), trimmed
    !
    !      solve A X = Y,
    ! 
    !        where A=A(N,N), X=X(N), Y=Y(N)  (initial Y is stored as X)
    !
    !      on exit, A is replaced by its inverse, X is replaced by the solution
    !               INFO=0 (normal return), 1(N<0 on input), 2(signular matrix)
    !
    INTEGER N,INFO
    real(chm_real) A(N,N),X(N)
    INTEGER IPIV(N),INDXR(N),INDXC(N)
    INTEGER I,J,K,L,LL,IROW,ICOL
    real(chm_real)  BIG,DUM,PIVINV
    real(chm_real) ONE, ZERO
    PARAMETER  ( ONE = 1.0D+0, ZERO = 0.0D+0 )
    !
    INFO=0
    if(N < 0) then
       INFO=1
       return
    endif
    !
    DO J=1,N
       IPIV(J)=0
    ENDDO
    DO I=1,N
       BIG=ZERO
       DO J=1,N
          IF(IPIV(J) /= 1)THEN
             DO K=1,N
                IF (IPIV(K) == 0) THEN
                   IF (ABS(A(J,K)) >= BIG)THEN
                      BIG=ABS(A(J,K))
                      IROW=J
                      ICOL=K
                   ENDIF
                ELSE IF (IPIV(K) > 1) THEN
                   INFO=2
                   return
                ENDIF
             ENDDO
          ENDIF
       ENDDO
       IPIV(ICOL)=IPIV(ICOL)+1
       IF (IROW /= ICOL) THEN
          DO L=1,N
             DUM=A(IROW,L)
             A(IROW,L)=A(ICOL,L)
             A(ICOL,L)=DUM
          ENDDO
          DUM=X(IROW)
          X(IROW)=X(ICOL)
          X(ICOL)=DUM
       ENDIF
       INDXR(I)=IROW
       INDXC(I)=ICOL
       IF (A(ICOL,ICOL) == ZERO) then
          info=2
          return
       endif
       PIVINV=ONE/A(ICOL,ICOL)
       A(ICOL,ICOL)=ONE
       DO L=1,N
          A(ICOL,L)=A(ICOL,L)*PIVINV
       ENDDO
       X(ICOL)=X(ICOL)*PIVINV
       DO LL=1,N
          IF(LL /= ICOL)THEN
             DUM=A(LL,ICOL)
             A(LL,ICOL)=zero
             DO L=1,N
                A(LL,L)=A(LL,L)-A(ICOL,L)*DUM
             ENDDO
             X(LL)=X(LL)-X(ICOL)*DUM
          ENDIF
       ENDDO
    ENDDO
    DO L=N,1,-1
       IF(INDXR(L) /= INDXC(L))THEN
          DO K=1,N
             DUM=A(K,INDXR(L))
             A(K,INDXR(L))=A(K,INDXC(L))
             A(K,INDXC(L))=DUM
          ENDDO
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE GAUSSJt
  !
  !****************************************************************
  !*     TAD I/O subroutines: tree toplogy and restart files      *
  !****************************************************************
  !
  !====================================================================
  subroutine WriteTree(iunit)
    !
    !     Write the tree structure to a file
    !
    use version
    use dimens_fcm
    use stream
    use psf
    use ctitla
    !     
    integer iunit,j,k,istart,istop
    logical notdone
    !
    IF(IOLEV < 0.or.iunit <= 0) RETURN
    WRITE(iunit,'(A4,2I6,2X,A)') &
         'TREE',VERNUM,TADVER,'!THIS IS A TAMD TREE FILE'
    CALL WRTITL(TITLEA,NTITLA,0,+2)
    WRITE(iunit,'(/I8,A)') NTITLA,' !NTITLE followed by title'
    WRITE(iunit,'(A)') (TITLEA(j),j=1,NTITLA)
    !  
    write(iunit,'(/A)') ' !NATOM, NCLUS, NSeg'
    write(iunit,'(I8,1X,I6,1X,I4)') NATOM, NClus, NSeg
    !
    write(iunit,'(/A)') ' !ClusID(1:NATOM)'
    notdone=.true.
    istart=1
    istop=min(10,NATOM)
    do while(notdone)
       notdone=istop < NATOM
       write(iunit,'(10(I6,1X))') (ClusID(k),k=istart,istop)
       istart=istart+10
       istop=min(NATOM,istop+10)
    enddo
    !      
    write(iunit,'(/A)')  &
         ' !ClusID, OrigPt, HingePt, Parent, NChild, Child(1:*)'
    do k=1,NClus
       write(iunit,'(4(I6,1X),I4,1X,10(I6,1X))') &
            k,OrigPt(k),HingePt(k),Parent(k), &
            NChild(k),(Child(j,k),j=1,NChild(k))
    enddo
    !
    write(iunit,'(/A)')  ' !NCSeg(1:Nseg+1)'
    notdone=.true.
    istart=1
    istop=min(10,NSeg+1)
    do while(notdone)
       notdone=istop < NSeg+1
       write(iunit,'(10(I4,1X))') (NCSeg(k),k=istart,istop)
       istart=istart+10
       istop=min(NSeg+1,istop+10)
    enddo
    !
    CALL SAVEIT(iunit)
    IF(PRNLEV >= 2) WRITE(OUTU,'(A,I8)') &
         ' WriteTree: TREE file was written on unit ', iunit
    !
    return
  end subroutine WriteTree
  !
  !======================================================================
  subroutine ReadTree(iunit)
    !
    !     Read the tree structure from a file
    !
    use version
    use dimens_fcm
    use stream
    use psf
    use ctitla
    !     
    character(len=4) HDR
    character(len=120) LINE
    logical notdone
    integer iunit, ntot, j, k, iver, itadver, istart, istop, iseg
    !
    IF(IOLEV < 0) return
    read(iunit,'(A4,2I6,2X,A)') HDR, iver, itadver
    if(HDR /= 'TREE') then
       if(prnlev >= 2) write(outu,'(A)') &
            ' ReadTree: wrong file header (not TREE).'
       call wrndie(0,'<TAMD>','ERROR during read tree')
       return
    endif
    if(itadver > TADVER) then
       call wrndie(1,'<TAMD>', &
            'The file has a higher TAD version number.')
    endif
    READ(iunit,'(/I8)',END=100) NTITLB
    READ(iunit,'(A)',END=100) (TITLEB(k),k=1,NTITLB)
    IF(PRNLEV >= 2) CALL WRTITL(TITLEB,NTITLB,OUTU,+1)
    !
    read(iunit,'(/A)',end=100) LINE
    read(iunit,'(I8,1X,I6,1X,I4)',end=100) ntot, k, iseg
    if(ntot /= NATOM) then
       if(prnlev >= 2) write(outu,'(2A)') ' ReadTree: ', &
            'atom number in the file is different from PSF'
       call wrndie(0,'<TAMD>','ERROR during read tree')
       return
    endif
    if(iseg /= NSEG) then
       if(prnlev >= 2) write(outu,'(2A)') ' ReadTree: ', &
            'segment number in the file is different from PSF'
       call wrndie(0,'<TAMD>','ERROR during read tree')
    endif
    NClus=k
    !
    read(iunit,'(/A)',end=100) LINE
    notdone=.true.
    istart=1
    istop=min(NATOM,10)
    do while(notdone)
       notdone=istop < NATOM
       read(iunit,'(10(I6,1X))',end=100) (ClusID(k),k=istart,istop)
       istart=istart+10
       istop=min(NATOM,istop+10)
    enddo
    !
    read(iunit,'(/A)',end=100) LINE
    do k=1,NClus
       read(iunit,'(4(I6,1X),I4,1X,10(I6,1X))',end=100) &
            ntot,OrigPt(k),HingePt(k),Parent(k), &
            NChild(k),(Child(j,k),j=1,NChild(k))
    enddo
    !
    read(iunit,'(/A)',end=100) LINE
    notdone=.true.
    istart=1
    istop=min(NSEG+1,10)
    do while(notdone)
       notdone=istop < NSEG+1
       read(iunit,'(10(I4,1X))',end=100) (NCSeg(k),k=istart,istop)
       istart=istart+10
       istop=min(NSEG+1,istop+10)
    enddo
    !
    TreeReady=1
    if(prnlev > 2) write(outu,'(A,I4)') &
         ' ReadTree: successfully read TREE from unit ', iunit
    return
100 TreeReady=0
    call wrndie(0,'<TAMD>','EOF during read tree')
    return
  end subroutine ReadTree
  !
  !======================================================================
  subroutine readtadyn(U,NATOM,X,Y,Z,NSeg, &
       TADVER,dTheta,dThetaOld,SpVel,bSpVelOld, &
       npriv,jhstrt,ndegf,nstep,nsavc,nsavv,seed,avetem,istpsa)
    !
    !     read TAMD restart file
    !
    use stream
    use string
    use energym
    use averfluc
    use version
    use ctitla
    use number
    !      
    ! ... arguments
    integer U, NATOM, TADVER, NSeg
    integer npriv,jhstrt,ndegf,nstep,nsavc,nsavv,istpsa
    real(chm_real)  X(NATOM),Y(NATOM),Z(NATOM)
    real(chm_real)  dTheta(NClus),dThetaOld(NClus)
    real(chm_real)  SpVel(6,NClus),bSpVelOld(6,NSeg),avetem,seed
    ! ... local variables
    integer i,j,NATOMQ,NCLUSQ,ILENEP,ILENET,ivers,itadvers,NSEGQ
    character(len=1) BIT
    character(len=4) HDR
    character(len=128) LINE
    logical MISMAT,QET
    !
    IF(IOLEV < 0.or.U <= 0) RETURN
    !
    REWIND(UNIT=U)
    READ(U,'(A4,2I6)',END=9) HDR,IVERS,ITADVERS
    IF(HDR /= 'REST') THEN
       IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
            ' READTADYN: wrong file header (not REST)'
       GOTO 8
    ENDIF
    if(ITADVERS > TADVER) call wrndie(0,'<TAMD>', &
         'RESTart file has higher TAMD version')
    READ(U,'(/I8)',END=9) NTITLB
    READ(U,'(A)',END=9) (TITLEB(I),I=1,NTITLB)
    IF(PRNLEV >= 2) CALL WRTITL(TITLEB,NTITLB,OUTU,+1)
    !      
    READ(U,'(/A)',END=9) LINE
    READ(U,'(7I12,D22.15,2I12)',END=9) &
         NATOMQ,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NCLUSQ,NSEGQ
    if(NATOM /= NATOMQ) then
       if(wrnlev >= 2) write(outu,'(A)') &
            ' READTADYN: NATOM - psf mismatch'
       goto 8
    endif
    if(NCLUS /= NCLUSQ) then
       if(wrnlev >= 2) write(outu,'(A)') &
            ' READTADYN: NCLUS - TAMD tree mismatch'
       goto 8
    endif
    if(NSEG /= NSEGQ) then
       if(wrnlev >= 2) write(outu,'(A)') &
            ' READTADYN: NSEG - TAMD tree mismatch'
       goto 8
    endif
    !-----------------------------------------------------------------------
    ! Read current flags, energies and statistics
    !
    ! Just in case we're reading on old restart file, zero the arrays
    ! so what isn't read has zeros.
    EPROP = ZERO
    ETERM = ZERO
    EPRESS = ZERO
    call avfl_reset_lt()
    call avfl_reset()
    !
    READ(U,'(/A)',END=9) LINE
    READ(U,'(A)',END=9) LINE  ! read QEPROP
    ILENEP=LEN(LINE)
    CALL TRIMA(LINE,ILENEP)
    ! Check to see if the energy property flags match
    J=MIN(ILENEP,LENENP)
    MISMAT=.FALSE.
    DO I=1,J
       BIT=LINE(I:I)  
       READ(BIT,'(L1)') QET
       IF(QET.NEQV.QEPROP(I)) MISMAT=.TRUE.
    ENDDO
    IF(MISMAT) CALL WRNDIE(-1,'<TAMD>', &
         'Energy property flags in the restart file do not match')
    !
    READ(U,'(A)',END=9) LINE  ! read QETERM
    ILENET=LEN(LINE)
    CALL TRIMA(LINE,ILENET)
    ! Check to see if the energy term flags match
    J=MIN(ILENET,LENENT)
    MISMAT=.FALSE.
    DO I=1,J
       BIT=LINE(I:I)  
       READ(BIT,'(L1)') QET
       IF(QET.NEQV.QETERM(I)) MISMAT=.TRUE.
    ENDDO
    IF(MISMAT) CALL WRNDIE(-1,'<TAMD>', &
         'Energy term flags in the restart file do not match')
    !
    ! XXX duplicates dynio
    IF (ILENEP > LENENP .OR. ILENET.GT.LENENT) THEN
       ! A future restart file??
       CALL WRNDIE(-4,'<READYN>', &
            'Cannot read a future version restart file')
    ELSE IF (ILENEP == LENENP .AND. ILENET.EQ.LENENT) THEN
       ! A current restart file
       READ(U,'(I8,3D22.15)',END=9) ISTPSA,FITA,FITP,AVETEM
       READ(U,'(3D22.15)',END=9) (EPROP(I),EPRPP(I),EPRP2P(I),I=1, &
            LENENP)
       READ(U,'(2D22.15)',END=9) (EPRPA(I),EPRP2A(I),I=1,LENENP)
       READ(U,'(3D22.15)',END=9) (ETERM(I),ETRMP(I),ETRM2P(I),I=1, &
            LENENT)
       READ(U,'(2D22.15)',END=9) (ETRMA(I),ETRM2A(I),I=1,LENENT)
       READ(U,'(3D22.15)',END=9) (EPRESS(I),EPRSP(I),EPRS2P(I), &
            I=1,LENENV)
       READ(U,'(2D22.15)',END=9) (EPRSA(I),EPRS2A(I),I=1,LENENV)
    ELSE IF (ILENEP >= 50 .AND. ILENET.GE.50) THEN
       ! A post version 21 restart file (it should not happen)
       READ(U,'(I8,3D22.15)',END=9) ISTPSA,FITA,FITP,AVETEM
       READ(U,'(3D22.15)',END=9) (EPROP(I),EPRPP(I),EPRP2P(I),I=1, &
            ILENEP)
       READ(U,'(2D22.15)',END=9) (EPRPA(I),EPRP2A(I),I=1,ILENEP)
       READ(U,'(3D22.15)',END=9) (ETERM(I),ETRMP(I),ETRM2P(I),I=1, &
            ILENET)
       READ(U,'(2D22.15)',END=9) (ETRMA(I),ETRM2A(I),I=1,ILENET)
       IF(IVERS > 23) THEN
          READ(U,'(3D22.15)',END=9) (EPRESS(I),EPRSP(I),EPRS2P(I), &
               I=1,LENENV)
          READ(U,'(2D22.15)',END=9) (EPRSA(I),EPRS2A(I),I=1,LENENV)
       ENDIF
    ELSE
       ! an old restart file - no go.... (definitely should not happen)
       CALL WRNDIE(-4,'<READYN>', &
            'Cannot read a version 21 or earlier restart file')
    ENDIF
    !
    !-----------------------------------------------------------------------
    ! read positions and velocities
    READ(U,'(/A)',END=9) LINE
    READ(U,'(3D22.15)',END=9) (X(I),Y(I),Z(I),I=1,NATOM)
    READ(U,'(/A)',END=9) LINE
    READ(U,'(2D22.15)',END=9) (dTheta(I),dThetaOld(I),I=1,NClus)
    READ(U,'(/A)',END=9) LINE
    do J=1,NSEG
       NSEGQ=NCSeg(J+1)
       READ(U,'(2D22.15)',END=9) (SpVel(I,NSEGQ),bSpVelOld(I,J),I=1,6)     
    ENDDO
    !
    ! ... done
    if(prnlev >= 2) write(outu,'(2A,I8)') &
         ' READTADYN: TAMD dynamics restart file was read.', &
         ' Current step=', NPRIV      
    return
    !
    ! ... error cases
8   call wrndie(-3,'<TAMD>','ERROR during read RESTart')
    return
9   call wrndie(-3,'<TAMD>','EOF during read RESTart')
    return
  end subroutine readtadyn
  !
  !======================================================================
  subroutine writadyn(U,NATOM,X,Y,Z,NSeg, &
       TADVER,dTheta,dThetaOld,SpVel,bSpVelOld, &
       npriv,jhstrt,ndegf,nstep,nsavc,nsavv,seed,avetem,istpsa)
    !
    !     write TAMD restart file: tree structure not recorded
    !
    use stream
    use energym
    use averfluc
    use version
    use ctitla
    !
    integer U, NATOM, TADVER, NSeg
    integer npriv,jhstrt,ndegf,nstep,nsavc,nsavv,istpsa
    real(chm_real)  X(NATOM),Y(NATOM),Z(NATOM)
    real(chm_real)  dTheta(NClus),dThetaOld(NClus)
    real(chm_real)  SpVel(6,NClus),bSpVelOld(6,NSeg),avetem,seed
    !
    ! ... local variables
    integer i,iseg,lstclus
    !
    IF(IOLEV < 0.or.U <= 0) RETURN
    REWIND(UNIT=U)
    WRITE(U,'(A4,2I6,2X,A)') &
         'REST',VERNUM,TADVER,'!THIS IS A TAMD RESTART FILE'
    CALL WRTITL(TITLEA,NTITLA,0,+2)
    WRITE(U,'(/I8,A)') NTITLA,' !NTITLE followed by title'
    WRITE(U,'(A)') (TITLEA(I),I=1,NTITLA)
    !
    WRITE(U,'(/A)') &
         ' !NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NCLUS,NSeg'
    WRITE(U,'(7I12,D22.15,2I12)') &
         NATOM,NPRIV,NSTEP,NSAVC,NSAVV,JHSTRT,NDEGF,SEED,NCLUS,NSeg
    !
    !
    ! write current flags, energies and statistics
    ! XXX duplicates dynio
    WRITE(U,'(/A)') ' !ENERGIES and STATISTICS'
    WRITE(U,'(128L1)') (QEPROP(I), I = 1,LENENP)
    WRITE(U,'(128L1)') (QETERM(I), I = 1,LENENT)
    WRITE(U,'(I8,3D22.15)') ISTPSA,FITA,FITP,AVETEM
    WRITE(U,'(3D22.15)') (EPROP(I),EPRPP(I),EPRP2P(I),I=1,LENENP)
    WRITE(U,'(2D22.15)') (EPRPA(I),EPRP2A(I),I=1,LENENP)
    WRITE(U,'(3D22.15)') (ETERM(I),ETRMP(I),ETRM2P(I),I=1,LENENT)
    WRITE(U,'(2D22.15)') (ETRMA(I),ETRM2A(I),I=1,LENENT)
    WRITE(U,'(3D22.15)') (EPRESS(I),EPRSP(I),EPRS2P(I),I=1,LENENV)
    WRITE(U,'(2D22.15)') (EPRSA(I),EPRS2A(I),I=1,LENENV)
    !
    ! write positions and velocities
    WRITE(U,'(/A)') ' !X, Y, Z'
    WRITE(U,'(3D22.15)') (X(I),Y(I),Z(I),I=1,NATOM)
    WRITE(U,'(/A)') ' !dTheta, dThetaOld'
    WRITE(U,'(2D22.15)') (dTheta(I),dThetaOld(I),I=1,NClus)
    WRITE(U,'(/A)') ' !bSpVel, bSpVelOld'
    do iseg=1,NSeg
       lstclus=NCSeg(iseg+1)
       WRITE(U,'(2D22.15)') (SpVel(I,lstclus),bSpVelOld(I,iseg),I=1,6)
    enddo
    ! 
    CALL SAVEIT(U)
    IF(PRNLEV >= 2) WRITE(OUTU,'(A,I8)') &
         ' WRITADYN: RESTart file was written at step', NPRIV
    !      
    return
  end subroutine writadyn
  !
  !************************************
  !*     debugging related routines   *
  !************************************
  !
  !======================================================================
  !     given 4 atomic coordinates, compute dihedral(I-J-K-L)
  !     (copied from QUICKA.SRC, CHARMM) (for debugging only)
  subroutine GetDihe(NTot,X,Y,Z,iat,jat,kat,lat,phi,ierr)
    implicit none
    integer ierr, iat, jat, kat, lat, NTot
    real(chm_real) X(*),Y(*),Z(*),phi

    real(chm_real) COSMAX,ONE
    PARAMETER (COSMAX=0.9999999999D0,ONE=1.0d0)
    real(chm_real) FX,FY,FZ,FR,HX,HY,HZ,HR,GX,GY,GZ,GR,CST,AX,AY,AZ, &
         BX,BY,BZ, &
         RA,RB,CX,CY,CZ
    !
    ierr=0     
    if(iat > NTot.or.jat.gt.NTot.or.kat.gt.NTot.or.lat.gt.NTot) then
       ierr=9
999    write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       return
    endif
    !
    FX=X(IAT)-X(JAT)
    FY=Y(IAT)-Y(JAT)
    FZ=Z(IAT)-Z(JAT)
    FR=FX*FX + FY*FY + FZ*FZ
    FR=SQRT(FR)
    IF(FR < 0.000001) THEN
       ierr=1
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    FX=FX/FR
    FY=FY/FR
    FZ=FZ/FR
    !     
    HX=X(LAT)-X(KAT)
    HY=Y(LAT)-Y(KAT)
    HZ=Z(LAT)-Z(KAT)
    HR=HX*HX + HY*HY + HZ*HZ
    HR=SQRT(HR)
    IF(HR < 0.000001) THEN
       ierr=2
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    HX=HX/HR
    HY=HY/HR
    HZ=HZ/HR
    !     
    GX=X(JAT)-X(KAT)
    GY=Y(JAT)-Y(KAT)
    GZ=Z(JAT)-Z(KAT)
    GR=GX*GX+GY*GY+GZ*GZ
    GR=SQRT(GR)
    IF(GR < 0.000001) THEN
       ierr=3
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    GX=GX/GR
    GY=GY/GR
    GZ=GZ/GR
    !     
    CST=-FX*GX-FY*GY-FZ*GZ
    IF(ABS(CST) >= COSMAX) THEN
       ierr=4
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    CST=HX*GX+HY*GY+HZ*GZ
    IF(ABS(CST) >= COSMAX) THEN
       ierr=5
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    !     
    AX=FY*GZ-FZ*GY
    AY=FZ*GX-FX*GZ
    AZ=FX*GY-FY*GX
    BX=HY*GZ-HZ*GY
    BY=HZ*GX-HX*GZ
    BZ=HX*GY-HY*GX
    !     
    RA=AX*AX+AY*AY+AZ*AZ
    RB=BX*BX+BY*BY+BZ*BZ
    RA=SQRT(RA)
    RB=SQRT(RB)
    IF(RA <= 0.000001) THEN
       ierr=6
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    IF(RB <= 0.000001) THEN
       ierr=7
       write(*,*) 'Error: GetDihe: No valid dihedral, ierr=',ierr
       RETURN
    ENDIF
    !     
    AX=AX/RA
    AY=AY/RA
    AZ=AZ/RA
    BX=BX/RB
    BY=BY/RB
    BZ=BZ/RB
    !     
    CST=AX*BX+AY*BY+AZ*BZ
    IF(ABS(CST) >= ONE) CST=SIGN(ONE,CST)
    PHI=ACOS(CST)
    CX=AY*BZ-AZ*BY
    CY=AZ*BX-AX*BZ
    CZ=AX*BY-AY*BX
    IF(GX*CX+GY*CY+GZ*CZ > 0.0) PHI=-PHI
    !      PHI=PHI*RADDEG
    !      WRITE(*,*) ' GetDihe: the dihedral is:', phi*180/acos(-1d0)
    return
  end subroutine GetDihe
  !
  !     print matrix (for debugging only)
  subroutine prmat(ounit, mat, M, N)
    implicit none
    integer ounit, M, N, i, j
    real(chm_real)  mat(M,N)
    write(ounit,100) 'r\c', (i,i=1,N)
100 format(2X,A4,20(I7,3X))
    do i=1, M
       write(ounit,200) i,(mat(i,j),j=1,N)
200    format(I4,1X,20(1X,F9.4))
    enddo
    return
  end subroutine prmat
  !
  subroutine cprod(A, B, C) !(for debugging only)
    implicit none
    real(chm_real)  A(3), B(3), C(3)
    C(1)=A(2)*B(3)-A(3)*B(2)
    C(2)=A(3)*B(1)-A(1)*B(3)
    C(3)=A(1)*B(2)-A(2)*B(1)
    return
  end subroutine cprod

#else /* (TAMDMAIN)*/
subroutine TAMD
  use stream
  if(prnlev >= 1) write(outu,'(A)')  &
       'Torsion Angle Molecular Dynamics not available (not compiled)'
end subroutine TAMD
#endif /* (TAMDMAIN)*/

end module tamdmodule

