#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
SUBROUTINE NBNDQM(X,Y,Z)
  !-----------------------------------------------------------------------
  !     Main driver of MM/MM, electrostatic QM/MM and van der Waals
  !     QM/MM interaction/exclusion list generation routines.
  !     Called by NBONDS <nbonds/nbonds.src>.
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use memory
  !
  use bases_fcm
  use datstr
  use inbnd
  use nbndqm_mod

#if KEY_QUANTUM==1
  use quantm      
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  use gamess_fcm  
#endif
#if KEY_MNDO97==1
  use mndo97      
#endif
#if KEY_SQUANTM==1
  use squantm     
#endif

  use psf
  use stream
  !
  !
  use image
#if KEY_PBOUND==1
  use pbound  
#endif
  implicit none
  !
  !
  real(chm_real)  x(*),y(*),z(*)
  integer nqmlen,nnnbq 
  integer i, numqm,imem_size,num_qm_grp

#if KEY_QUANTUM==1
  numqm = natqm     
#endif
#if KEY_SQUANTM==1
  numqm = natqm(1)  
#endif
#if KEY_MNDO97==1
  numqm = numat     
#endif

  !
  !     copy normal non-bond data structure pointers to /NBONDG/
  !     common block.  This is an intermediate code to introduce QM/MM
  !     force field program into CHARMM.  22-Mar-91 YDW.
  !
  if (useddt_nbond(bnbnd)) then
     nnbgrp = nnnbg
     inbgrp%a => bnbnd%inblog
     inbgrp%len = size(bnbnd%inblog)
     jnbgrp%a => bnbnd%jnbg
     jnbgrp%len = size(bnbnd%jnbg)
     ngrpex = nng14
     igrpex%a => bnbnd%iglo14
     igrpex%len = size(bnbnd%iglo14)
     jgrpex%a => bnbnd%ing14
     jgrpex%len = size(bnbnd%ing14)
     natmex = nnb14
     iatmex%a => bnbnd%iblo14
     iatmex%len = size(bnbnd%iblo14)
     jatmex%a => bnbnd%inb14
     jatmex%len = size(bnbnd%inb14)
  else
     call wrndie(-3,'<NBNDQM>', &
          'Nonbond data structure is not defined.')
  endif
  !
  !     for coordinates swaping, for images.
  if(allocated(imattq) .and. size(imattq) .ne. natom ) then
     call chmdealloc('qmnbnd.src','NBNDQM','imattq',size(imattq),intg=imattq)
  end if
  if(.not.allocated(imattq)) call chmalloc('qmnbnd.src','NBNDQM','imattq',natom,intg=imattq)
  IMATTQ(1:natom) = 0
  !
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0) THEN
        IF(LGROUP) THEN
           !              nnbgrp = nnnbg + NIMNBG
           if(useddt_image(bimag)) then
              mnbgrp    = bimag%NIMNBG
              imbgrp%a => bimag%imblog
              imbgrp%len = size(bimag%imblog)
              jmbgrp%a => bimag%imjnbg
              jmbgrp%len = size(bimag%imjnbg)
           else
              call wrndie(-3,'<NBNDQM>', &
                   'Image nonbond data structure is not defined.')
           end if
        ELSE
#if KEY_MNDO97==0
           call wrndie(-2,'<NBNDQM>', &
                'Atom based cutoff do not support QM/MM with Image.')
#endif
        END IF
     END IF
#if KEY_PBOUND==1
  endif                 
#endif
  !
  !     Rearrange the normal nonbond lists for MM/MM atoms
  !     We do not use QM/MM atom list in this version.
  !
  ! namkh 08/08/04
  ! QM/MM-Ewald and non-bonded interactions (GROUP or ATOM based cutoff)
  !    Comments: It doesn't matter Group based cutoff to any boundary
  !              conditions...(and also in nbndq2 subroutine)
  !              Currently, QM/MM-Ewald and IMAGE on PBC only support
  !              GROUP based cutoff.
  !              In addition, in IMAGE, self-term of QM atoms with its own
  !              image will be ignored. Actually, it shouldn't happen in QM/MM,
  !              and also in QM/MM-Ewald for real space mulitpolar
  !              QM/MM interactions.
  !    *******
  !   *Important*
  !              Currently, it allows only MINIMUM image QM/MM interactions.
  !              Thus, the cutoff distance should be less than half of the
  !              box length.
  !    *******
  if(allocated(qgrpmm) .and. size(qgrpmm) .ne. ngrp) then
     call chmdealloc('qmnbnd.src','NBNDQM','qgrpmm',size(qgrpmm),log=qgrpmm)
     call chmdealloc('qmnbnd.src','NBNDQM','qgrpqm',size(qgrpqm),log=qgrpqm)
  end if
  if(.not.allocated(qgrpmm)) call chmalloc('qmnbnd.src','NBNDQM','qgrpmm',ngrp,log=qgrpmm)  ! flag for mm group
  if(.not.allocated(qgrpqm)) call chmalloc('qmnbnd.src','NBNDQM','qgrpqm',ngrp,log=qgrpqm)  ! flag for qm group
  
  IF(LGROUP) THEN
     !if(allocated(qgrpmm) .and. size(qgrpmm) .ne. ngrp) then
     !   call chmdealloc('qmnbnd.src','NBNDQM','qgrpmm',size(qgrpmm),log=qgrpmm)
     !   call chmdealloc('qmnbnd.src','NBNDQM','qgrpqm',size(qgrpqm),log=qgrpqm)
     !end if
     !if(.not.allocated(qgrpmm)) call chmalloc('qmnbnd.src','NBNDQM','qgrpmm',ngrp,log=qgrpmm)  ! flag for mm group
     !if(.not.allocated(qgrpqm)) call chmalloc('qmnbnd.src','NBNDQM','qgrpqm',ngrp,log=qgrpqm)  ! flag for qm group

     ! on return, qgrpmm=.true. for all mm and mixed qm/mm groups, while all qm groups are .false.
     Call Grpmm (ngrp,natom,igpbs,numqm,num_qm_grp  &
#if KEY_QUANTUM==1
                 ,qatlab                            &
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
                 ,igmsel                            &
#endif 
                ,qgrpmm)
     call nbndq2(ngrp,nnnbg,bnbnd%inblog,bnbnd%jnbg,qgrpmm)

     ! This part cause later heap_ error. I am not sure now why it happends.
     ! Since later GUPQME will take care all the QM/MM interactions with
     ! primary image and their images, and the QM charge will be set to
     ! zero, thus, it wouldn't cause any problem without modifying it.
     ! But, GUPQME should only loop over primary image groups..
     ! --------------namkh 09/28/04
     !#if KEY_PBOUND==1
     !        if(.not.qboun) then   
     !#endif
     !           IF(NTRANS.GT.0) THEN
     !              call nbndq2(ngrp,BIMAG%NIMNBG,bimag%imjnbg,bimag%imblog,qgrpmm)
     !           END IF
     !#if KEY_PBOUND==1
     !        end if                 
     !#endif

  ELSE
     ! QM/MM-Ewald
#if KEY_QUANTUM==1 || KEY_SQUANTM==1
     IF(LQMEWD) call wrndie(-2,'<NBNDQM>','QM/MM-Ewald do not work with Atom based cutoff.')
#elif KEY_MNDO97==1
     !IF(LQMEWD) call wrndie(1,'<NBNDQM>','QM/MM-Ewald do not tested throughly with Atom based cutoff.')
     !
     ! NOTE: Kwangho Nam, 2015-0420.
     !
     ! What we are doing?
     ! for MM-MM interactions: use atom-based cutoff.
     ! for QM-MM interactions: use group based-cutoff (electrostatics and vdW interactions).
     !
     ! Why we are doing this way, though it may sounds not right? However, there is
     ! nothing wrong by doing this way, as the MM-MM and QM-MM interactions are handled
     ! separately in the current implementation of QM/MM.
     ! The benefit of doing this is that MM part scales better in running parallel
     ! environment. In particular, this allows us to use BYCBim when using crystal and PME.
     ! This definitely scales better with increasing number of processors.
     ! 
     ! However, note that this approach is not rigirous and causes some systematic shift,
     ! if you are computing potential of mean force. So, use it with your own risk.
     ! Nevertheless, there is a future plan to implement fully atom-based cutoff scheme.
     !
#endif
     !
     nqmlen=numqm*(natom-numqm)
     if(nqmlen.gt.nnnb) nqmlen=nnnb
     if(nqmlen_nbnd.ne.nqmlen) then   ! has to reallocate memory.
        if(allocated(inblos)) call chmdealloc('qmnbnd.src','NBNDQM','inblos',size(inblos),intg=inblos)
        if(allocated(jnbs))   call chmdealloc('qmnbnd.src','NBNDQM','jnbs',size(jnbs),intg=jnbs)
        if(allocated(inbloq)) call chmdealloc('qmnbnd.src','NBNDQM','inbloq',size(inbloq),intg=inbloq)
        if(allocated(jnbq))   call chmdealloc('qmnbnd.src','NBNDQM','jnbq',size(jnbq),intg=jnbq)
        nqmlen_nbnd = nqmlen
     end if
     if(.not.allocated(inblos)) call chmalloc('qmnbnd.src','NBNDQM','inblos',natom,intg=inblos)
     if(.not.allocated(jnbs))   call chmalloc('qmnbnd.src','NBNDQM','jnbs',nqmlen,intg=jnbs)
     if(.not.allocated(inbloq)) call chmalloc('qmnbnd.src','NBNDQM','inbloq',natom,intg=inbloq)
     if(.not.allocated(jnbq))   call chmalloc('qmnbnd.src','NBNDQM','jnbq',nqmlen,intg=jnbq)

     ! for the flag for mm and qm atoms.
     if(allocated(qatmmm)) then
        if(size(qatmmm).ne.natom) call chmdealloc('qmnbnd.src','NBNDQM','qatmmm',natom,log=qatmmm)
     end if
     if(.not.allocated(qatmmm)) call chmalloc('qmnbnd.src','NBNDQM','qatmmm',natom,log=qatmmm)

     ! fill qatmmm array.
     call QMMMATM(natom,qatmmm   &
#if KEY_QUANTUM==1
                 ,qatlab         & 
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
                 ,igmsel         & 
#endif
                  )
     ! now clean-up non-bond list.
     call nbndq1(natom,nnnb,bnbnd%inblo,bnbnd%jnb,inblos,jnbs,nnnbq, &
                 inbloq,jnbq,qatmmm                                  &
#if KEY_IMCUBES==1
                ,lbycbim         &
#endif
                 )
  END IF
  !
  !     Update the QM/MM electrostatic group list
  call gupqme(x,y,z)
  !
  !     Update the QM/MM van der Waals group list
  Call gupqmv(x,y,z)
  !
  return
end SUBROUTINE NBNDQM

SUBROUTINE NBNDQ1(NATOM,NNNB,INBLO,JNB,INBLOS,JNBS,NNNBQ, &
                  INBLOQ,JNBQ,qatmmm                      &
#if KEY_IMCUBES==1
                 ,lbycbim                                 &
#endif
                  )
  !-----------------------------------------------------------------------
  !     NBNDQM rearranges the normal non-bond lists into two lists.
  !     One of these (INBLO/JNB) contains all the MM/MM atom interactions
  !     whilst the other (INBLOQ/JNBQ) has the QM/MM interactions. The
  !     INBLOS and JNBS arrays are scratch arrays.
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  implicit none
  !
  integer :: inblo(*),inblos(*),jnb(*),jnbs(*),natom,nnnb,nnnbq
  integer :: inbloq(*),jnbq(*)
  logical :: qatmmm(natom)
#if KEY_IMCUBES==1
  logical :: lbycbim
#endif
  !
  integer i, itemp, j, jpr, k, nb, nbnew, nbq, npr
  integer jj
  !
  !     Remove all QM/QM interactions from the non-bond list and put
  !     all QM/MM interactions into INBLOS/JNBS.
  !
  itemp = 0
  nb    = 0
  nbnew = 0
  nbq   = 0

#if KEY_IMCUBES==1
  ! we do not change the size of jnb, but for the ij pairs with
  ! any qm atoms will be skipped, such that inblo(i+natom) points
  ! toward the npr number.
  if(lbycbim) then
     do i = 1,natom
        itemp = inblo(i+natom)
        npr   = inblo(i) - itemp
        nbnew = 0                ! local counter
        if (npr .gt. 0) then
           ! only do the cycle, if i is mm atom. (so, skip if i is qm.)
           if (qatmmm(i)) then
              do jpr = 1,npr
                 j  = jnb(itemp+jpr)
                 jj = iabs(j)
                 if(.not.qatmmm(jj)) cycle   ! qm atom, skip this pair.

                 nbnew = nbnew + 1
                 jnb(itemp+nbnew) = jnb(itemp+jpr)
              end do
           end if
        endif
        inblo(i)  = itemp + nbnew

        ! What this means is that inblo(i+natom) is the same,
        ! so does jnb(itemp+i). By just makeing inblo changed such that
        ! inblo(i) - itemp returns new NPR value, which is nbnew.
     end do
     !nnnb  = nbnew  ; nnnb is not changed, just jnb array is skipped if qm atoms.
  else
#endif
     ! otherwise, it probably will be the same as original one.
     do i = 1,natom
        npr = inblo(i) - itemp
        itemp = inblo(i)
        if (npr .gt. 0) then
           if(.not.qatmmm(i)) then
              ! i is qm atom. qatmmm(i)==.false.
              do jpr = 1,npr
                 nb = nb + 1
                 j  = jnb(nb)
                 jj = iabs(j)
                 if(qatmmm(jj)) then   ! jj is mm atom
                    ! qm-mm pair.
                    nbq = nbq + 1
                    jnbs(nbq) = j
                 endif
              end do
           else
              ! i is mm atom. qatmmm(i)==.true.
              do jpr = 1,npr
                 nb = nb + 1
                 j  = jnb(nb)
                 jj = iabs(j)
                 if(.not.qatmmm(jj)) then ! jj is qm atom
                    ! mm-qm pair.
                    nbq = nbq + 1
                    jnbs(nbq) = j
                 else
                    nbnew = nbnew + 1
                    jnb(nbnew) = jnb(nb)
                 endif
              end do
           endif
        endif
        inblo(i)  = nbnew
        inblos(i) = nbq
     end do
     !
     nnnb  = nbnew
     nnnbq = nbq
     !
     !     Sort the INBLOS/JNBS arrays into INBLOQ/JNBQ making use of the
     !     fact that the atom labels in the JNBS list are always greater
     !     than the label of the indexing atom.
     !
     nbq  = 0
     do i = 1,natom
        if (.not.qatmmm(i)) then
           ! i is qm atom.
           itemp = 0
           do j = 1,(i-1)
              npr   = inblos(j) - itemp
              nb    = itemp
              itemp = inblos(j)
              if(qatmmm(j)) then  
                 ! j is mm atom.
                 do jpr = 1,npr
                    nb = nb + 1
                    k = jnbs(nb)
                    if (iabs(k) .eq. i) then
                       nbq = nbq + 1
                       jnbq(nbq) = sign(j,k)
                       EXIT    ! goto 50
                    endif
                 end do
              endif
              !50         continue
           end do
           !
           npr = inblos(i) - itemp
           nb  = itemp
           do jpr = 1,npr
              nb  = nb + 1
              nbq = nbq + 1
              jnbq(nbq) = jnbs(nb)
           end do
        endif
        !
        inbloq(i) = nbq
     end do
     !
     if (nbq .ne. nnnbq) then
        IF(PRNLEV >= 2) THEN
           write ( outu, '(a,i5)' ) ' Nbnew = ', nbnew
           !
           write ( outu, '(a)' ) ' Inbloq = '
           write ( outu, '(20i5)' ) ( inbloq(i), i = 1,natom )
           write ( outu, '(a)' ) ' Jnbq = '
           write ( outu, '(20i5)' ) ( jnbq(i), i = 1,nbq )
           !
           write ( outu, '(a)' ) ' Inblos = '
           write ( outu, '(20i5)' ) ( inblos(i), i = 1,natom )
           write ( outu, '(a)' ) ' Jnbs = '
           write ( outu, '(20i5)' ) ( jnbs(i), i = 1,nnnbq )
           !
           write ( outu, '(a,2i5)' ) ' Nbq, Nnnbq = ', nbq, nnnbq
        END IF
        call wrndie(-5,'<NBNDQM>','Error in non-bond list processing.')
     endif
#if KEY_IMCUBES==1
  end if
#endif
  !
  return
end SUBROUTINE NBNDQ1
!
! namkh 08/08/04
! QM/MM-Ewald
SUBROUTINE NBNDQ2(NGRP,NNNBG,INBLOG,JNBG,QGRPMM)
  !-----------------------------------------------------------------------
  !     NBNDQ2 rearranges the normal non-bond lists into two lists.
  !     One of these (INBLOG/JNBG) contains all the MM/MM group interactions
  !     whilst the other (INBLOQ/JNBQ) has the QM/MM interactions. The
  !     INBLOS and JNBS arrays are scratch arrays.
  !-----------------------------------------------------------------------
  use chm_kinds
  use stream
  implicit none
  !
  integer inblog(*), jnbg(*), nnnbg, ngrp
  logical qgrpmm(*)
  !
  integer i, itemp, j, jpr, k, nb, nbnew, nbq, npr
  !
  !     Remove all QM/QM interactions from the non-bond list and put
  !     all QM/MM interactions into INBLOS/JNBS.
  !
  itemp = 0
  nb    = 0
  nbnew = 0
  nbq   = 0
  do i = 1,ngrp
     npr = inblog(i) - itemp
     itemp = inblog(i)
     if (npr .gt. 0) then
        if (.not. qgrpmm(i)) then
           do jpr = 1,npr
              nb = nb + 1
              j  = jnbg(nb)
              if ( qgrpmm(iabs(j)) ) nbq = nbq + 1
           end do
        else
           do jpr = 1,npr
              nb = nb + 1
              j  = jnbg(nb)
              if (.not. qgrpmm(iabs(j))) then
                 nbq = nbq + 1
              else
                 nbnew = nbnew + 1
                 jnbg(nbnew) = jnbg(nb)
              endif
           end do
        endif
     endif
     inblog(i)  = nbnew
  end do
  nnnbg  = nbnew
  !
  return
end SUBROUTINE NBNDQ2

SUBROUTINE GUPQME(X,Y,Z)
  !------------------------------------------------------------------------
  !     Create the QM/MM electrostatic group interaction lists.
  use chm_kinds
  use dimens_fcm
  use memory
  use number, only : half
  !
  use bases_fcm
  use inbnd
  use nbndqm_mod

#if KEY_QUANTUM==1
  use quantm      
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  use gamess_fcm  
#endif
#if KEY_MNDO97==1
  use mndo97      
#endif
#if KEY_SQUANTM==1
  use squantm     
#endif

  use psf
  use stream
  !
  !     Adjust nonbonded group list for IMAGES.
  use image
#if KEY_PBOUND==1
  use pbound   
#endif
  !---   use nbutil_module,only: prnbd3
  implicit none
  !
  real(chm_real)   X(*),Y(*),Z(*)
  Integer  maxnnb, nnbx, nnbex

  !!integer, parameter :: Maxqmg=200  ! moved to nbndqm_mod

  integer  imxnb(Maxqmg)
  !
  Logical  qdone, qImage
  !
  Integer:: numqm,i,is,iq,num_mm_atm,num_qm_grp
  integer, allocatable,dimension(:) :: jnbx, jnbex
  real(chm_real) :: box_dim(3)   ! use for approximate way to compute the distance cutoff
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  Logical QMCUTF   
#endif

#if KEY_QUANTUM==1
  numqm = natqm    
#endif
#if KEY_MNDO97==1
  numqm = numat    
#endif
#if KEY_SQUANTM==1
  numqm = natqm(1) 
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  QMCUTF= .TRUE.   
  QNBGRP= .TRUE.    /* it assume GROUP, for QUANTUM, it sets somewhere.*/
#endif
  !
  !========================================================================
  !     Check whether IMAGE is used
  !========================================================================
  qImage =.FALSE.
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0) THEN
        qImage=.TRUE.
        ! half of the box dimension(x,y,z direction).
        box_dim(1) = half*sqrt(xtlabc(1)**2+xtlabc(2)**2+xtlabc(4)**2)   ! x-side distance
        box_dim(2) = half*sqrt(xtlabc(2)**2+xtlabc(3)**2+xtlabc(5)**2)   ! y-side distance
        box_dim(3) = half*sqrt(xtlabc(4)**2+xtlabc(5)**2+xtlabc(6)**2)   ! z-side distance
     END IF
#if KEY_PBOUND==1
  endif                 
#endif

  !========================================================================
  !     Initialise the group lists.
  !========================================================================
  nullify(iqmgpe)
  nullify(jqmgpe)
  nullify(iqmgex)
  nullify(jqmgex)
  !========================================================================
  !     Allocate scratch space.
  !========================================================================
  call chmalloc('qmnbnd.src','GUPQME','iqmgpe',ngrp,intgp=iqmgpe)
  call chmalloc('qmnbnd.src','GUPQME','iqmgex',ngrp,intgp=iqmgex)

  if(allocated(qgrpmm) .and. size(qgrpmm) .ne. ngrp) then
     call chmdealloc('qmnbnd.src','GUPQME','qgrpmm',size(qgrpmm),log=qgrpmm)
  end if
  if(allocated(xyzg) .and. size(xyzg).ne.(3*ngrp)) then
     call chmdealloc('qmnbnd.src','GUPQME','xyzg',3,size(xyzg,2),crl=xyzg)
  end if
  if(.not.allocated(qgrpmm)) call chmalloc('qmnbnd.src','GUPQME','qgrpmm',ngrp,log=qgrpmm)
  if(.not.allocated(xyzg))   call chmalloc('qmnbnd.src','GUPQME','xyzg',3,ngrp,crl=xyzg)

  !-------
  !jG 103000 changed to include only qm/mm pairs in qm/mm non-bonded list.
  !jG   maxnnb = (ngrp * (ngrp + 1)) / 2
  !jG   call chmalloc('qmnbnd.src','GUPQME','jnbx',maxnnb,intg=jnbx)
  !jG   call chmalloc('qmnbnd.src','GUPQME','jnbex',maxnnb,intg=jnbex)
  !-------
  !========================================================================
  !     Calculate the group centres of geometry.
  !========================================================================
  Call Grpcen_G(ngrp,igpbs,x,y,z,xyzg) 
  !========================================================================
  !     Fill the Qgrpmm array.
  !========================================================================
  ! on return, qgrpmm=.true. for all mm and mixed qm/mm groups, while all qm groups are .false.
  !            qgrpqm=.false. for all pure mm groups, and .true. for all qm and mixed qm/mm groups.
  Call Grpmm_el(ngrp,natom,igpbs,numqm,num_qm_grp &
#if KEY_QUANTUM==1
            ,qatlab &
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
            ,igmsel &
#endif
            ,qgrpmm,qgrpqm)

  !-------
  !jG 103000 allcate space for qm/mm non-bonded list
#if KEY_MNDO97==1
  call alnbsp_el(ngrp,qgrpqm,maxnnb,imxnb)
  num_qm_group = num_qm_grp ! number of qm groups (see call Grpmm).
#else
  call alnbsp(ngrp,qgrpmm,maxnnb,imxnb)
#endif
  !
  maxnnb = maxnnb*ngrp
  call chmalloc('qmnbnd.src','GUPQME','jnbx',maxnnb,intg=jnbx)
  call chmalloc('qmnbnd.src','GUPQME','jnbex',maxnnb,intg=jnbex)
  !jG
  !C-------
  !========================================================================
  !     Calculate the group lists.
  !========================================================================
  If(qImage) then
#if KEY_MNDO97==0
     Call Guqme3 (ngrp,natom,igpbs,iatmex%a,jatmex%a,nnbx,iqmgpe,jnbx,nnbex, &
          iqmgex,jnbex,nbond,ib,jb,qdone,qgrpmm,qgrpqm,cutnb,xyzg,box_dim, &
          maxnnb, ntheta,it,kt,QMCUTF,QNBGRP,numqm,IMATTQ,NTRANS,IMTRNS,NOROT, &
          num_mm_atm)
#else
     Call Guqme3_mndo(ngrp,natom,igpbs,iatmex%a,jatmex%a,nnbx,iqmgpe,jnbx,nnbex, &
          iqmgex,jnbex,nbond,ib,jb,qdone,qgrpmm,qgrpqm,cutnb,xyzg,box_dim, &
          maxnnb, ntheta,it,kt,QMCUTF,QNBGRP,numqm,IMATTQ,NTRANS,IMTRNS,NOROT, &
          num_mm_atm)
#endif
  Else
     Call Guqme2 (ngrp,natom,igpbs,iatmex%a,jatmex%a,nnbx,iqmgpe,jnbx,nnbex, &
          iqmgex,jnbex,nbond,ib,jb,qdone,qgrpmm,qgrpqm,cutnb,xyzg, &
          maxnnb, ntheta,it,kt,QMCUTF,QNBGRP,numqm)
  End if
  !
  If (qdone) then
     nqmgpe = nnbx
     nqmgex = nnbex
     call chmalloc('qmnbnd.src','GUPQME','jqmgpe',nnbx,intgp=jqmgpe)
     call chmalloc('qmnbnd.src','GUPQME','jqmgex',nnbex,intgp=jqmgex)
     jqmgpe(1:nnbx ) = jnbx(1:nnbx)
     jqmgex(1:nnbex) = jnbex(1:nnbex)
     IF(PRNLEV.GE.2) THEN
        IF(QNBGRP) THEN
           Write (outu,'(a,i6,a)') ' GUPQME> ',nnbx,' unique QM molecule-MM group interactions generated.'
        ELSE
           Write (outu,'(a,i6,a)') ' GUPQME> ',nnbx,' QM/MM group interactions generated.'
        ENDIF
        Write (outu,'(a,i6,a)') ' GUPQME> ',nnbex,' QM/MM group exclusions generated.'
     END IF
#if KEY_MNDO97==1
     if(qImage .and. q_cut_by_group .or. qmswtch_qmmm) then
       num_mm_group = nqmgpe ! + num_qm_group ! additional buffer..
       num_mmatm_in_list = num_mm_atm ! + 100 ! total number of mm and qm atoms in the list and extra atoms (buffer).
       if(associated(map_allatm_to_group)) deallocate(map_allatm_to_group)
       if(associated(map_mmatom_to_group)) deallocate(map_mmatom_to_group)
       if(associated(map_qmatom_to_group)) deallocate(map_qmatom_to_group)
       if(associated(map_mmgrp_to_group))  deallocate(map_mmgrp_to_group)

       ! this array maps each atom to its corresponding group; so, the inverse of igpbs.
       ! map_mmatom_to_group: map mm atom to its group in ngrp.
       ! map_qmatom_to_group: map qm atom to its group in NQMGRP(1).
       ! map_allatm_to_group: map each atom to its group in ngrp.
       if(.not.associated(map_allatm_to_group)) then
          allocate(map_allatm_to_group(natom))
          map_allatm_to_group(1:natom) = -1
          do i=1,ngrp
             is=igpbs(i)+1
             iq=igpbs(i+1)
             map_allatm_to_group(is:iq) = i  ! i-th group.
          end do
          ! check errors.
          do i=1,natom
             if(map_allatm_to_group(i) < 0 ) then
                if(prnlev >= 2) write(outu,'(a,i7,a)') 'Missing group definition for,',i,'-th atom.'
                call wrndie(-5,'<GUPQME>','Missing group info for map_allatm_to_group.')
             end if
          end do
       end if
       if(.not.associated(map_mmatom_to_group)) allocate(map_mmatom_to_group(num_mmatm_in_list))
       if(.not.associated(map_qmatom_to_group)) allocate(map_qmatom_to_group(numqm))
       if(.not.associated(map_mmgrp_to_group))  allocate(map_mmgrp_to_group(num_mm_group))
     end if
#endif
  Else
     Call Wrndie(-5,'<GUPQME>','QM/MM group update error.')
  Endif
  !========================================================================
  !     Free working stack_ space.
  !========================================================================
  call chmdealloc('qmnbnd.src','GUPQME','jnbx',size(jnbx),intg=jnbx)
  call chmdealloc('qmnbnd.src','GUPQME','jnbex',size(jnbex),intg=jnbex)
  call chmdealloc('qmnbnd.src','GUPQME','xyzg',3,size(xyzg,2),crl=xyzg)
  !
  IF(PRNLEV.GT.7) THEN
     WRITE(OUTU,24) ' The QM/MM electrostatic group list'
24   FORMAT(1X,A/)
     CALL PRNBD3(OUTU,NQMGPE,NGRP,iqmgpe,jqmgpe)
     WRITE(OUTU,24) ' The QM/MM electrotatic group exclusion list'
     CALL PRNBD3(OUTU,NQMGEX,NGRP,iqmgex,jqmgex)
  ENDIF
  !
  Return
END SUBROUTINE GUPQME

SUBROUTINE GUQME2 (NGRP,NATOM,IGPBS,IATMEX,JATMEX,NB,INB,JNB, &
     NBEX, INBEX, JNBEX, NBOND, IB, JB, QDONE, &
     QGRPMM,qgrpqm,CUTNB, XYZ_g, MAXNNB,NTHETA,IT,KT, &
     QMCUTF,QNBGRP,NUMQM)
  !------------------------------------------------------------------------
  !
  !     Does the work of GUPQME. The lists are generated with the QM
  !     group first and the MM group next even if the index of the QM
  !     group is larger than that of the MM group. This is in contrast
  !     to the generation of the other lists in which the ordering is
  !     implicit and assumed in the code. The QM/MM code permits any
  !     ordering.
  !
  !     Ngrp           - the number of groups. (In IMAGE, it is NIMGRP)
  !     Natom          - the number of atoms.
  !     Igpbs          - the group atom array.
  !     Iatmex, Jatmex - the atom exclusion arrays.
  !     Nb, Nbex       - the number of QM/MM interactions and exclusions.
  !     Inb, Jnb       - the QM/MM interaction arrays.
  !     Inbex, Jnbex   - the QM/MM exclusion arrays.
  !     Nbond          - the number of bonds.
  !     Ib, Jb         - bond arrays.
  !     Qdone          - logical flag indicating list generation status.
  !     Qgrpmm         - logical array indicating whether groups are MM.
  !     Cutnb          - non-bonded cutoff.
  !     Xg, Yg, Zg     - group centres of geometry.
  !     Maxnnb         - the maximum number of group interactions allowed.
  !     QMCUTF         - Flag: whether to use cutoffs
  !     Nqmgrp         - number of qm groups
  !     Iqmgrp         - index for qm groups
  !
  use chm_kinds
  use sizes

#if KEY_QUANTUM==1
  use qmlinkm  
#endif
#if KEY_MNDO97==1
  use mndgho   
#endif
#if KEY_SQUANTM==1
  use squantm, only: QLINK     
#endif
  !
#if KEY_PBOUND==1
  use pbound   
#endif
  use number 
  use nbndqm_mod,only : Maxqmg

  implicit none

#if KEY_PBOUND==1
  real(chm_real) CORR   
#endif
  !
  Integer iatmex(*), ib(*), igpbs(*), inb(*), inbex(*), jatmex(*), &
       jb(*), jnb(*), jnbex(*), maxnnb, nb, nbex, nbond, ngrp, &
       natom
  INTEGER NTHETA,IT(*),KT(*)
  Logical qdone, qgrpmm(*),qgrpqm(*)
  real(chm_real)  cutnb, xyz_g(3,ngrp) 
  !
  !!integer, parameter :: Maxqmg=200  ! moved to nbndqm_mod

  Integer i, ibond, matom, mfirst, mgrp, mlast, mstop, mstart, &
       qatom, qfirst, qgrp, qlast, qstop, qstart, IANGL, NUMQM
  Logical qex123, QNBGRP, QMCUTF
  real(chm_real)  cutnb2, xyz_cen(3), rij2, rr2
  real(chm_real):: delta_xyz(3),xyz_local(3),box_inv(3),box_reg(3)
  integer nqmgrp,iqmgrp(Maxqmg),ii,ij,iigrp
  Logical QGHOL, q_do_this_group


#if KEY_QUANTUM==1
  QGHOL = QMLINK   
#endif
#if KEY_MNDO97==1
  QGHOL = QLINK    
#endif
#if KEY_SQUANTM==1
  QGHOL = QLINK(1) 
#endif
  !
  cutnb2 = cutnb * cutnb
  !
  nb    = 0
  nbex  = 0
  qdone = .false.
  !
#if KEY_PBOUND==1 /*pbound*/
  IF(qBoun .and. qTOBoun) CALL WRNDIE(-5,'qmnbnd>','TOBoundary not supported')
  box_inv(1)=BOXINV
  box_inv(2)=BOYINV
  box_inv(3)=BOZINV
  box_reg(1)=XSIZE
  box_reg(2)=YSIZE
  box_reg(3)=ZSIZE
#endif /* (pbound)*/

#if KEY_MNDO97==1
  call alnbsp_el(ngrp,qgrpqm,nqmgrp,iqmgrp)
#else
  call alnbsp(ngrp,qgrpmm,nqmgrp,iqmgrp)
#endif

  Do qgrp = 1,ngrp
#if KEY_MNDO97==1
     if(qgrpqm(qgrp)) then          ! qm group (for mndo97, includes mixed qm/mm group as qm group.)
#else
     If (.not. qgrpmm(qgrp)) then
        ! if not mm group == qm group (qgrp), including mixed qm-mm group (for GHO).
#endif
        xyz_cen(1) = xyz_g(1,qgrp)
        xyz_cen(2) = xyz_g(2,qgrp)
        xyz_cen(3) = xyz_g(3,qgrp)
        !
        qstart = igpbs(qgrp) + 1
        qstop  = igpbs(qgrp + 1)
        !
        Do mgrp = 1,ngrp
           If (qgrpmm(mgrp)) then
              ! for mndo97, it is possible that mgrp is the same as qgrp as the mixed qm/mm group
              !             is considered both the qm and mm group.
              ! mgrp is here mm group.
              delta_xyz(1) = xyz_g(1,mgrp) - xyz_cen(1)
              delta_xyz(2) = xyz_g(2,mgrp) - xyz_cen(2)
              delta_xyz(3) = xyz_g(3,mgrp) - xyz_cen(3)
              ! JG 6/1/01
              ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
              IF(qBoun) THEN
                 IF(qCUBoun.or.qTOBoun) THEN
                    delta_xyz(1:3)=box_inv(1:3)*delta_xyz(1:3)
                    do ij=1,3
                       if(delta_xyz(ij).gt. half) delta_xyz(ij)=delta_xyz(ij)-one
                       if(delta_xyz(ij).lt.-half) delta_xyz(ij)=delta_xyz(ij)+one
                    end do
                    delta_xyz(1:3)=box_reg(1:3)*delta_xyz(1:3)
                 ENDIF
              ENDIF
#endif /*  (pbound)*/
              rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
              !
              ! JG 3/21/01
              ! USE THE FIRST SHORTEST distance for a QM/MM PAIR IN QM/MM group list.
              q_do_this_group=.true.
!#if KEY_MNDO97==1
!              if(mgrp /= qgrp) then
!                 ! The opposite occurs when this group (mgrp and qgrp) is the mixed qm/mm group for GHO.
!                 ! In this case, we includes this group in the list (as mm group).
!#endif
#if KEY_QUANTUM==1
                 IF(QNBGRP) THEN
#endif
                    Do II = 1,nqmgrp
                       iigrp = iqmgrp(ii)
                       if(iigrp.ne.qgrp) then
                          xyz_local(1)= xyz_g(1,mgrp) - xyz_g(1,iigrp)
                          xyz_local(2)= xyz_g(2,mgrp) - xyz_g(2,iigrp)
                          xyz_local(3)= xyz_g(3,mgrp) - xyz_g(3,iigrp)
                          ! JG 6/1/01
                          ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
                          IF(qBoun) THEN
                             IF(qCUBoun.or.qTOBoun) THEN
                                xyz_local(1:3)=box_inv(1:3)*xyz_local(1:3)
                                do ij=1,3
                                   if(xyz_local(ij).gt. half) xyz_local(ij)=xyz_local(ij)-one
                                   if(xyz_local(ij).gt.-half) xyz_local(ij)=xyz_local(ij)+one
                                end do
                                xyz_local(1:3)=box_reg(1:3)*xyz_local(1:3)
                             END IF
                          END IF
#endif /*  (pbound)*/
                          rr2 = xyz_local(1)*xyz_local(1)+xyz_local(2)*xyz_local(2)+xyz_local(3)*xyz_local(3) 
                          if(rr2.lt.rij2 ) then
                             q_do_this_group=.false.
                             EXIT        ! goto 85
                          end if
                       end if
                    End do
#if KEY_QUANTUM==1
                 END IF
#endif
!#if KEY_MNDO97==1
!              end if
!#endif
              !
              !--JG
              if(q_do_this_group) then
                 If ((.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2)) then
#if KEY_QUANTUM==1 /*quantm_specific*/
                    mstart = igpbs(mgrp) + 1
                    mstop  = igpbs(mgrp + 1)

                    !========================================================================
                    ! . Determine the type of interaction.
                    qex123 = .false.
                    !
                    !      IN QM LINK ATOM CASE, EVERYTHING IS INCLUDED INCLUDING THE CHARGE
                    !      ON THE LINK ATOM.  HOWEVER, "SPECIAL" TREATMENT MUST BE MADE SO
                    !      THAT IT DOES NOT INTERACT WITH ITSELF IN THE QM/MM PART.  IT IS
                    !      TREATED AS A NORMAL QM NUCLEUS CHARGE IN THE QM PART...JG 5/17/97
                    !      c.f. qm/mm energy routines
                    !
                    !      only do this with QUANTUM, since MNDO97 this will be taken cared by
                    !      checking IGMSEL(i)=0 or 5 to exclude excluded MM atoms.

                    If (.not. QGHOL) then          ! skip for GHO atoms.
                       Do qatom = qstart,qstop
                          If (qatom .eq. 1) then
                             qfirst = 1
                          Else
                             qfirst = iatmex(qatom - 1) + 1
                          Endif
                          qlast = iatmex(qatom)
                          ! 1-2.
                          do matom = mstart,mstop
                             do ibond = 1,nbond
                                If (qatom.eq.ib(ibond) .and. matom.eq.jb(ibond) .or. &
                                     qatom.eq.jb(ibond) .and. matom.eq.ib(ibond)) then
                                   If (nbex .ge. maxnnb) then
                                      Call Wrndie (-5,'<GUQME2>', 'Exclusion list error.')
                                      return
                                   Else
                                      nbex = nbex + 1
                                      jnbex(nbex) = - mgrp
                                      Goto 70
                                   Endif
                                Endif
                             end do
                             !-------------------------------------------------------------------------
                             !  bonds, just in case and make sure...JG 12/96
                             DO IANGL = 1,NTHETA
                                IF((QATOM.EQ.IT(IANGL) .AND. MATOM.EQ.KT(IANGL)) .OR. &
                                     (QATOM.EQ.KT(IANGL) .AND. MATOM.EQ.IT(IANGL)))THEN
                                   IF(NBEX.GE.MAXNNB) THEN
                                      CALL WRNDIE (-5,'<GUQME2>','Exclusion list error.')
                                      return   
                                   ELSE
                                      NBEX = NBEX+1
                                      JNBEX(NBEX) = -MGRP
                                      GOTO 70
                                   ENDIF
                                ENDIF
                             ENDDO
                             !-------------------------------------------------------------------------
                             ! 1-3.
                             If (matom .eq. 1) then
                                mfirst = 1
                             Else
                                mfirst = iatmex(matom - 1) + 1
                             End if
                             mlast = iatmex(matom)
                             If (qatom .le. matom) then
                                do i = qfirst,qlast
                                   If (matom .eq. jatmex(i)) then
                                      qex123 = .true.
                                      Goto 60
                                   Endif
                                end do
                             Else
                                do i = mfirst,mlast
                                   If (qatom .eq. jatmex(i)) then
                                      qex123 = .true.
                                      Goto 60
                                   Endif
                                end do
                             Endif
                          end do     !  matom = mstart,mstop
                       End do        !  qatom = qstart,qstop
                    End if      ! .not. QGHOL
60                  continue
#endif /* (quantm_specific)*/

                    If (nb .ge. maxnnb) then
                       Call Wrndie (-5,'<GUQME2>','QM/MM group list error.')
                       return
#if KEY_QUANTUM==1
                    Else if (qex123) then
                       nb = nb + 1
                       jnb(nb) = - mgrp
#endif
                    Else
                       nb = nb + 1
                       jnb(nb) = mgrp
                    Endif
#if KEY_QUANTUM==1
70                  Continue
#endif
                    !========================================================================
                 Endif      ! (.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2)
              end if     ! q_do_this_group
           Endif         ! qgrpmm(mgrp)
        End do           ! mgrp = 1,ngrp
     Endif               ! (.not. qgrpmm(qgrp))
     inb(qgrp)   = nb
     inbex(qgrp) = nbex
  End do                 ! qgrp = 1,ngrp
  qdone = .true.
  !
999 Return
END SUBROUTINE GUQME2
!
#if KEY_MNDO97==0  /*MNDO97 case*/
SUBROUTINE GUQME3 (NGRP,NATOM,IGPBS,IATMEX,JATMEX,NB,INB, &
     JNB,NBEX, INBEX, JNBEX, NBOND, IB, JB, QDONE, &
     QGRPMM,qgrpqm,CUTNB, XYZ_G, box_dim, &
     MAXNNB,NTHETA,IT,KT, &
     QMCUTF,QNBGRP,NUMQM, IMATTQ, &
     NTRANS,IMTRNS,NOROT,num_mm_atm)
  !------------------------------------------------------------------------
  !
  !     Does the work of GUPQME. The lists are generated with the QM
  !     group first and the MM group next even if the index of the QM
  !     group is larger than that of the MM group. This is in contrast
  !     to the generation of the other lists in which the ordering is
  !     implicit and assumed in the code. The QM/MM code permits any
  !     ordering.
  !
  !     Ngrp           - the number of groups.
  !     Natom          - the number of atoms.
  !     Igpbs          - the group atom array.
  !     Iatmex, Jatmex - the atom exclusion arrays.
  !     Nb, Nbex       - the number of QM/MM interactions and exclusions.
  !     Inb, Jnb       - the QM/MM interaction arrays.
  !     Inbex, Jnbex   - the QM/MM exclusion arrays.
  !     Nbond          - the number of bonds.
  !     Ib, Jb         - bond arrays.
  !     Qdone          - logical flag indicating list generation status.
  !     Qgrpmm         - logical array indicating whether groups are MM.
  !     Cutnb          - non-bonded cutoff.
  !     Xg, Yg, Zg     - group centres of geometry.
  !     Maxnnb         - the maximum number of group interactions allowed.
  !     QMCUTF         - Flag: whether to use cutoffs
  !     Nqmgrp         - number of qm groups
  !     Iqmgrp         - index for qm groups
  !     Imattq         - IMATTR mapping index into Primary image
  !     Norot          - whether have no-image rotation
  !     Ntrans         - number of image transformation
  !     IMTRNS         - image transformation matrix
  !
  use chm_kinds
  use dimens_fcm
  use sizes

#if KEY_QUANTUM==1
  use qmlinkm   
#endif
#if KEY_MNDO97==1
  use mndgho    
#endif
#if KEY_SQUANTM==1
  use squantm, only: QLINK     
#endif
#if KEY_PBOUND==1
  use pbound    
#endif
  use number
  use nbndqm_mod, only : Maxqmg
#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none

#if KEY_PBOUND==1
  real(chm_real) CORR   
#endif
  !
  Integer iatmex(*), ib(*), igpbs(*), inb(*), inbex(*), jatmex(*), &
       jb(*), jnb(*), jnbex(*), maxnnb, nb, nbex, nbond, ngrp, &
       natom, num_mm_atm
  INTEGER NTHETA,IT(*),KT(*),IMATTQ(*),NTRANS
  real(chm_real)  IMTRNS(*)
  Logical qdone, qgrpmm(*),qgrpqm(*),NOROT
  real(chm_real)  cutnb, xyz_g(3,ngrp)
  !
  Integer       :: i, ibond, matom, mfirst, mgrp, mlast, mstop, mstart, &
       qatom, qfirst, qgrp, qlast, qstop, qstart, IANGL, NUMQM
  integer       :: nqmgrp,ii,iigrp, mstart2, mstop2, ncount,imtrans
  real(chm_real):: cutnb2, rij2, rr2
  real(chm_real):: delta_xyz(3),xyz_local(3), box_dim(3)
  Logical       :: qex123, QNBGRP, QMCUTF
  Logical       :: QGHOL, q_do_this_group

  !!Integer,PARAMETER:: Maxqmg=200  ! moved to nbndqm_mod
  integer          :: iqmgrp(Maxqmg)
  real(chm_real)   :: xyzqm(3,Maxqmg), XYZIM(3,MAXTRN+1), cut_box_sq

  integer, save :: mgrp_start,mgrp_end
#if KEY_PARALLEL==1
  integer,       pointer,save :: imtrans_local(:)=>Null()
  integer,       pointer,save :: iqmmm_pair(:)=>Null()
  real(chm_real),pointer,save :: rij2_local(:)=>Null()
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif

#if KEY_QUANTUM==1
  QGHOL = QMLINK    
#endif
#if KEY_MNDO97==1
  QGHOL = QLINK     
#endif
#if KEY_SQUANTM==1
  QGHOL = QLINK(1)  
#endif
  !
  cutnb2 = cutnb * cutnb
  cut_box_sq= box_dim(1)**2+box_dim(2)**2+box_dim(3)**2  ! square of half-of-box-dimension
  !
  nb    = 0
  nbex  = 0
  qdone =.false.
  ncount= 0
  num_mm_atm = 0
  !
#if KEY_MNDO97==1
  call alnbsp_el(ngrp,qgrpqm,nqmgrp,iqmgrp)
#else
  call alnbsp(ngrp,qgrpmm,nqmgrp,iqmgrp)
#endif
  !
  do ii=1,nqmgrp
     iigrp       = iqmgrp(ii)
     xyzqm(1,ii) = xyz_g(1,iigrp)
     xyzqm(2,ii) = xyz_g(2,iigrp)
     xyzqm(3,ii) = xyz_g(3,iigrp)
  end do

#if KEY_PARALLEL==1
  ! memory.
  if(associated(rij2_local))    deallocate(rij2_local)
  if(associated(imtrans_local)) deallocate(imtrans_local)
  if(associated(iqmmm_pair))    deallocate(iqmmm_pair)
  !
  allocate(rij2_local(ngrp))
  allocate(imtrans_local(ngrp))
  allocate(iqmmm_pair(ngrp))
  !
  if(numnod>1) then
     !call PARA_RANGE_nbnd(1,ngrp,numnod,mynod,mgrp_start,mgrp_end)
     !
     ! Prepare array for vector allgather calls using VDGBRE
     ! mapping for each node (Hard weird).
     JPARPT_local(0)=0
     do i=1,numnod
        JPARPT_local(i)= ngrp*i/numnod 
     end do
     mgrp_start = JPARPT_local(mynod)+1
     mgrp_end   = JPARPT_local(mynod+1)
  else
     mgrp_start = 1
     mgrp_end   = ngrp
  end if
#else
  mgrp_start = 1
  mgrp_end   = ngrp
#endif

  !
  ! Loop over groups in primary and image cell, since QM atom could
  ! be exist as images when QM atoms are close to boundary
  Do qgrp = 1,ngrp
#if KEY_MNDO97==1
     if(qgrpqm(qgrp)) then          ! qm group (for mndo97, includes mixed qm/mm group as qm group.)
#else
     If (.not. qgrpmm(qgrp)) then   ! qm group (does not include mixed qm/mm group as qm group.)
        ! if not mm group == qm group (qgrp), including mixed qm-mm group (for GHO).
#endif
        ncount = ncount + 1
        qstart = igpbs(qgrp) + 1
        qstop  = igpbs(qgrp + 1)
        num_mm_atm = num_mm_atm + (qstop-qstart+1) ! add number of mm atoms in the qm group for extra buffer.

#if KEY_PARALLEL==1
        rij2_local(mgrp_start:mgrp_end)   = zero
        imtrans_local(mgrp_start:mgrp_end)= 0
        iqmmm_pair(mgrp_start:mgrp_end)   = 0
#endif
        !
        ! Here need to loop over all the groups in primary groups
        ! with all image transformations to be checked.
        Do mgrp = mgrp_start,mgrp_end           ! 1,ngrp
           If (qgrpmm(mgrp)) then  ! mm group 
              ! for mndo97, it is possible that mgrp is the same as qgrp as the mixed qm/mm group
              !             is considered both the qm and mm group. 
              ! mgrp is here mm group.
              xyzim(1,1) = xyz_g(1,mgrp)
              xyzim(2,1) = xyz_g(2,mgrp)
              xyzim(3,1) = xyz_g(3,mgrp)

              !              Get the image transformed coordinates of MM group
              !              and find closet distance^2, and returns closet MM images
              !              (including primary image) as xyzim(1:3,1)
              imtrans = 0
              call IMTRNQM(NOROT,ncount,imtrans,cutnb2,NTRANS,IMTRNS,xyzqm,XYZIM, &
                           box_dim,cut_box_sq)

              delta_xyz(1:3) = xyzim(1:3,1) - xyzqm(1:3,ncount)
              rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3) 

              ! USE THE FIRST SHORTEST distance for a QM/MM PAIR IN QM/MM group list.
              q_do_this_group=.true.
!#if KEY_MNDO97==1
!              if(mgrp /= qgrp) then
!                 ! The opposite occurs when this group (mgrp and qgrp) is the mixed qm/mm group for GHO.
!                 ! In this case, we includes this group in the list (as mm group).
!#endif
#if KEY_QUANTUM==1
                 if(QNBGRP) then
#endif
                    do II = 1,nqmgrp
                       if(ii /= ncount) then
                          xyz_local(1:3)=xyzim(1:3,1)-xyzqm(1:3,ii)
                          rr2 = xyz_local(1)*xyz_local(1)+xyz_local(2)*xyz_local(2)+xyz_local(3)*xyz_local(3)
                          if (rr2 < rij2 ) then
                             q_do_this_group=.false.
                             EXIT
                          end if
                       end if
                    end do
#if KEY_QUANTUM==1
                 end if
#endif
!#if KEY_MNDO97==1
!              end if
!#endif
#if KEY_PARALLEL==1
              rij2_local(mgrp)   =rij2
              imtrans_local(mgrp)=imtrans
              if(q_do_this_group) iqmmm_pair(mgrp) = 1
           Endif            ! (qgrpmm(mgrp))
        End do              ! mgrp = 1,ngrp
        ! go broadcast.
        if(numnod>1) then
           !call gcomb(rij2_local,ngrp)
           !call igcomb(imtrans_local,ngrp)
           !call igcomb(iqmmm_pair,ngrp)

           call VDGBRE(rij2_local,JPARPT_local)
           call IVDGBRE(imtrans_local,JPARPT_local)
           call IVDGBRE(iqmmm_pair,JPARPT_local)
        end if

        ! we loop over all mm groups.
        Do mgrp = 1,ngrp
           If (qgrpmm(mgrp)) then   ! mm group (including mixed qm/mm groups)
              rij2   = rij2_local(mgrp)
              imtrans= imtrans_local(mgrp)
              if(iqmmm_pair(mgrp) >= 1) then
                 q_do_this_group =.true.
              else
                 q_do_this_group =.false.
              end if
#endif
              !--JG
              if(q_do_this_group) then
                 If ((.NOT.QMCUTF) .OR. (rij2 < cutnb2)) then
                    mstart = igpbs(mgrp) + 1
                    mstop  = igpbs(mgrp + 1)

                    ! Image transform mapping index for MM atoms to swap around the qm atoms. 
                    If (imtrans.gt.0) IMATTQ(mstart:mstop)=imtrans

#if KEY_QUANTUM==1 /*quantm_specific*/
                    !========================================================================
                    ! . Determine the type of interaction.
                    qex123 = .false.
                    !
                    !      IN QM LINK ATOM CASE, EVERYTHING IS INCLUDED INCLUDING THE CHARGE
                    !      ON THE LINK ATOM.  HOWEVER, "SPECIAL" TREATMENT MUST BE MADE SO
                    !      THAT IT DOES NOT INTERACT WITH ITSELF IN THE QM/MM PART.  IT IS
                    !      TREATED AS A NORMAL QM NUCLEUS CHARGE IN THE QM PART...JG 5/17/97
                    !      c.f. qm/mm energy routines
                    !
                    !      only do this with QUANTUM, since MNDO97 this will be taken cared by
                    !      checking IGMSEL(i)=0 or 5 to exclude excluded MM atoms.
                    !
                    IF(.not. QGHOL) then
                       do qatom = qstart,qstop
                          If (qatom .eq. 1) then
                             qfirst = 1
                          Else
                             qfirst = iatmex(qatom - 1) + 1
                          Endif
                          qlast = iatmex(qatom)
                          !
                          ! 1-2.
                          do matom = mstart,mstop
                             do ibond = 1,nbond
                                If (qatom.eq.ib(ibond) .and. matom.eq.jb(ibond) .or. &
                                     qatom.eq.jb(ibond) .and. matom.eq.ib(ibond)) then
                                   If (nbex .ge. maxnnb) then
                                      Call Wrndie (-5,'<GUQME2>','Exclusion list error.')
                                      return
                                   Else
                                      nbex = nbex + 1
                                      jnbex(nbex) = - mgrp
                                      Goto 70
                                   Endif
                                Endif
                             end do
                             !-------------------------------------------------------------------------
                             !  bonds, just in case and make sure...JG 12/96
                             DO IANGL = 1,NTHETA
                                IF((QATOM.EQ.IT(IANGL) .AND. MATOM.EQ.KT(IANGL)) .OR. &
                                     (QATOM.EQ.KT(IANGL) .AND. MATOM.EQ.IT(IANGL)))THEN
                                   IF(NBEX.GE.MAXNNB) THEN
                                      CALL WRNDIE (-5,'<GUQME2>','Exclusion list error.')
                                      return
                                   ELSE
                                      NBEX = NBEX+1
                                      JNBEX(NBEX) = -MGRP
                                      GOTO 70
                                   ENDIF
                                ENDIF
                             END DO
                             !-------------------------------------------------------------------------
                             ! 1-3.
                             If (matom .eq. 1) then
                                mfirst = 1
                             Else
                                mfirst = iatmex(matom - 1) + 1
                             End if
                             mlast = iatmex(matom)
                             If (qatom .le. matom) then
                                do i = qfirst,qlast
                                   If (matom .eq. jatmex(i)) then
                                      qex123 = .true.
                                      Goto 60
                                   Endif
                                end do
                             Else
                                do i = mfirst,mlast
                                   If (qatom .eq. jatmex(i)) then
                                      qex123 = .true.
                                      Goto 60
                                   Endif
                                end do
                             Endif
                          end do       ! matom = mstart,mstop
                       end do          ! qatom = qstart,qstop
                    End if    ! .not.QGHOL
60                  continue
#endif /*   (quantm_specific)*/

                    If (nb .ge. maxnnb) then
                       Call Wrndie (-5,'<GUQME2>','QM/MM group list error.')
                       return
#if KEY_QUANTUM==1
                    Else if (qex123) then
                       nb = nb + 1
                       jnb(nb) = - mgrp
#endif
                    Else
                       nb = nb + 1
                       jnb(nb) = mgrp
                       num_mm_atm = num_mm_atm + (mstop-mstart+1)
                    Endif
#if KEY_QUANTUM==1
70                  Continue
#endif
                    !========================================================================
                 Endif         ! ((.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2))
              end if        ! q_do_this_group
           Endif            ! (qgrpmm(mgrp))
        End do              ! mgrp = 1,ngrp
     Endif                  ! (.not. qgrpmm(qgrp))  ! or qgrpqm(qgrp) for mndo97
     inb(qgrp)   = nb
     inbex(qgrp) = nbex
  End do                    ! qgrp = 1,ngrp
  qdone = .true.

#if KEY_PARALLEL==1
  ! memory.
  deallocate(rij2_local)
  deallocate(imtrans_local)
  deallocate(iqmmm_pair)
#endif
  !
  Return
END SUBROUTINE GUQME3
#else     /*MNDO97 case*/
!
SUBROUTINE GUQME3_MNDO(NGRP,NATOM,IGPBS,IATMEX,JATMEX,NB,INB, &
     JNB,NBEX, INBEX, JNBEX, NBOND, IB, JB, QDONE, &
     QGRPMM,qgrpqm,CUTNB, XYZ_G, box_dim, &
     MAXNNB,NTHETA,IT,KT, &
     QMCUTF,QNBGRP,NUMQM, IMATTQ, &
     NTRANS,IMTRNS,NOROT,num_mm_atm)
  !------------------------------------------------------------------------
  !
  !     Copy of GUQME3 subroutine adopted for MNDO97 module.
  !
  !
  !     Does the work of GUPQME. The lists are generated with the QM
  !     group first and the MM group next even if the index of the QM
  !     group is larger than that of the MM group. This is in contrast
  !     to the generation of the other lists in which the ordering is
  !     implicit and assumed in the code. The QM/MM code permits any
  !     ordering.
  !
  !     Ngrp           - the number of groups.
  !     Natom          - the number of atoms.
  !     Igpbs          - the group atom array.
  !     Iatmex, Jatmex - the atom exclusion arrays.
  !     Nb, Nbex       - the number of QM/MM interactions and exclusions.
  !     Inb, Jnb       - the QM/MM interaction arrays.
  !     Inbex, Jnbex   - the QM/MM exclusion arrays.
  !     Nbond          - the number of bonds.
  !     Ib, Jb         - bond arrays.
  !     Qdone          - logical flag indicating list generation status.
  !     Qgrpmm         - logical array indicating whether groups are MM.
  !     Cutnb          - non-bonded cutoff.
  !     Xg, Yg, Zg     - group centres of geometry.
  !     Maxnnb         - the maximum number of group interactions allowed.
  !     QMCUTF         - Flag: whether to use cutoffs
  !     Nqmgrp         - number of qm groups
  !     Iqmgrp         - index for qm groups
  !     Imattq         - IMATTR mapping index into Primary image
  !     Norot          - whether have no-image rotation
  !     Ntrans         - number of image transformation
  !     IMTRNS         - image transformation matrix
  !
  use chm_kinds
  use dimens_fcm
  use sizes

  use mndgho
#if KEY_PBOUND==1
  use pbound    
#endif
  use number
  use nbndqm_mod, only : Maxqmg,i_map_box
#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none

#if KEY_PBOUND==1
  real(chm_real) CORR   
#endif
  !
  Integer iatmex(*), ib(*), igpbs(*), inb(*), inbex(*), jatmex(*), &
       jb(*), jnb(*), jnbex(*), maxnnb, nb, nbex, nbond, ngrp, &
       natom, num_mm_atm
  INTEGER NTHETA,IT(*),KT(*),IMATTQ(*),NTRANS
  real(chm_real)  IMTRNS(*)
  Logical qdone, qgrpmm(*),qgrpqm(*),NOROT
  real(chm_real)  cutnb, xyz_g(3,ngrp)
  !
  Integer       :: i, ibond, matom, mfirst, mgrp, mlast, mstop, mstart, &
                   qatom, qfirst, qgrp, qlast, qstop, qstart, IANGL, NUMQM
  integer       :: nqmgrp,ii,iigrp, mstart2, mstop2, ncount,imtrans,icnt
  real(chm_real):: cutnb2, rij2, rr2
  real(chm_real):: delta_xyz(3),xyz_local(3), box_dim(3)
  Logical       :: qex123, QNBGRP, QMCUTF
  Logical       :: QGHOL, q_do_this_group

  !!Integer,PARAMETER:: Maxqmg=200  ! moved to nbndqm_mod
  integer          :: iqmgrp(Maxqmg)
  real(chm_real)   :: xyzqm(3,Maxqmg), XYZIM(3,MAXTRN+1), cut_box_sq

  integer, save :: mgrp_start,mgrp_end,cell_start,cell_end
#if KEY_PARALLEL==1
  integer,       pointer,save :: imtrans_local(:)=>Null()
  integer,       pointer,save :: iqmmm_pair(:)=>Null()
  integer,save  :: JPARPT_local(0:MAXNODE),KPARPT_local(0:MAXNODE),LPARPT_local(0:MAXNODE)

  integer :: j,k,jj,kk,ij,ii_qm,jj_qm,kk_qm,ii_qm_old,jj_qm_old,kk_qm_old
  real(chm_real)      :: box_x,box_y,box_z,r_cutnb,xh_size(3),xyzim_ref(3)
  real(chm_real),save :: x_size=0.0d0, y_size=0.0d0, z_size=0.0d0 ! 
  integer,       save :: n_x=0,n_y=0,n_z=0,cell_dim=0,kk_min=0,kk_max=0,icnt_st=0

  ! memory
  integer, pointer,save :: i_grp_box(:,:,:)=>Null()
  real(chm_real),pointer,save :: x_cen(:)=>Null(), y_cen(:)=>Null(), z_cen(:)=>Null()
#endif

  QGHOL = QLINK     
  !
  cutnb2 = cutnb * cutnb
  cut_box_sq= box_dim(1)**2+box_dim(2)**2+box_dim(3)**2  ! square of half-of-box-dimension
  !
  nb    = 0
  nbex  = 0
  qdone =.false.
  ncount= 0
  num_mm_atm = 0
  !
  call alnbsp_el(ngrp,qgrpqm,nqmgrp,iqmgrp)
  !
  do ii=1,nqmgrp
     iigrp       = iqmgrp(ii)
     xyzqm(1,ii) = xyz_g(1,iigrp)
     xyzqm(2,ii) = xyz_g(2,iigrp)
     xyzqm(3,ii) = xyz_g(3,iigrp)
  end do

#if KEY_PARALLEL==1
  ! memory.
  if(associated(imtrans_local)) deallocate(imtrans_local)
  if(associated(iqmmm_pair))    deallocate(iqmmm_pair)
  !
  allocate(imtrans_local(ngrp))
  allocate(iqmmm_pair(ngrp))
  !
  if(numnod>1) then
     ! Prepare array for vector allgather calls using VDGBRE
     ! mapping for each node (Hard weird).
     JPARPT_local(0)=0
     do i=1,numnod
        JPARPT_local(i)= ngrp*i/numnod 
     end do
     mgrp_start = JPARPT_local(mynod)+1
     mgrp_end   = JPARPT_local(mynod+1)
  else
     mgrp_start = 1
     mgrp_end   = ngrp
  end if
#else
  mgrp_start = 1
  mgrp_end   = ngrp
#endif

  ! initialization
#if KEY_PARALLEL==1
  do i=mgrp_start,mgrp_end
     !rij2_local(i)   = zero
     imtrans_local(i)= 0
     iqmmm_pair(i)   = 0
  end do
#endif

#if KEY_PARALLEL==1
  ! find x_min,x_max,y_min,y_max,z_min,z_max
  ! Assume that the center of the box is origin.
  box_x = two*box_dim(1)  ! x,y,z box dimension.
  box_y = two*box_dim(2)
  box_z = two*box_dim(3)
  if(box_x > x_size .or. box_y > y_size .or. box_z > z_size) then
     ! box size has changed. updat info.
     x_size = box_x + two*cutnb + five
     y_size = box_y + two*cutnb + five
     z_size = box_z + two*cutnb + five

     ! n_boxes
     n_x = ceiling(x_size/cutnb)
     n_y = ceiling(y_size/cutnb)
     n_z = ceiling(z_size/cutnb)

     ! memories
     if(associated(i_grp_box)) then
        deallocate(i_grp_box)
        deallocate(i_map_box)
        deallocate(x_cen)
        deallocate(y_cen)
        deallocate(z_cen)
     end if
     if(.not.associated(i_grp_box)) allocate(i_grp_box(n_x,n_y,n_z))
     if(.not.associated(i_map_box)) allocate(i_map_box(3,ngrp))
     if(.not.associated(x_cen))     allocate(x_cen(n_x))
     if(.not.associated(y_cen))     allocate(y_cen(n_y))
     if(.not.associated(z_cen))     allocate(z_cen(n_z))

     ! fill the center of each box.
     x_cen(1) = - half*x_size + half*cutnb
     y_cen(1) = - half*y_size + half*cutnb
     z_cen(1) = - half*z_size + half*cutnb
     do i=2,n_x
        x_cen(i) = x_cen(i-1) + cutnb
     end do
     do j=2,n_y
        y_cen(j) = y_cen(j-1) + cutnb
     end do
     do k=2,n_z
        z_cen(k) = z_cen(k-1) + cutnb
     end do

     ! for i_grp_box parallelization.
     cell_dim = n_x*n_y*n_z
     KPARPT_local(0)=0
     LPARPT_local(0)=0
     do i=1,numnod
        KPARPT_local(i)= cell_dim*i/numnod 
        LPARPT_local(i)= JPARPT_local(i)*3
     end do
     cell_start = KPARPT_local(mynod)+1
     cell_end   = KPARPT_local(mynod+1)
     icnt   = 0
     kk_min = n_z
     kk_max = 1
     do kk=1,n_z
        do jj=1,n_y
           do ii=1,n_x
              icnt = icnt + 1
              if(icnt >= cell_start .and. icnt <= cell_end) then
                 kk_min = MIN(kk_min,kk)
                 kk_max = MAX(kk_max,kk)
              end if
           end do
        end do
     end do
     icnt_st = (kk_min-1)*n_y*n_x
  end if

  ! count how many groups are within each box.
  icnt = icnt_st
  do kk=kk_min,kk_max  ! 1,n_z
     do jj=1,n_y
        do ii=1,n_x
           icnt = icnt + 1
           if(icnt >= cell_start .and. icnt <= cell_end) i_grp_box(ii,jj,kk) =-1  ! default no checking
        end do
     end do
  end do
  r_cutnb    = one/cutnb
  xh_size(1) = half*x_size
  xh_size(2) = half*y_size
  xh_size(3) = half*z_size

  ! so in checking, skip n_grp_box(ii,jj,kk) = 0, which is empty.
  ii_qm_old = -9999
  jj_qm_old = -9999
  kk_qm_old = -9999
  do i=1,nqmgrp
     ! The box containing qm group
     ii_qm= INT((xyzqm(1,i)+xh_size(1))*r_cutnb) + 1
     jj_qm= INT((xyzqm(2,i)+xh_size(2))*r_cutnb) + 1
     kk_qm= INT((xyzqm(3,i)+xh_size(3))*r_cutnb) + 1
     !
     xyzim_ref(1) = x_cen(ii_qm)
     xyzim_ref(2) = y_cen(jj_qm)
     xyzim_ref(3) = z_cen(kk_qm)
     !
     if(ii_qm /= ii_qm_old .or. jj_qm /= jj_qm_old .or. kk_qm /= kk_qm_old) then
        ! loop over all boxes to find if image transform is needed.
        icnt = icnt_st
        do kk=kk_min,kk_max  ! 1,n_z
           xyzim(3,1) = z_cen(kk)
           do jj=1,n_y
              xyzim(2,1) = y_cen(jj)
              do ii=1,n_x
                 icnt = icnt + 1
                 if(icnt >= cell_start .and. icnt <= cell_end .and. i_grp_box(ii,jj,kk) <= 0) then
                    ! not empty box
                    xyzim(1,1) = x_cen(ii) 

                    ! check, if the present box is neighboring the box of qm.
                    if(abs(ii-ii_qm)<=1 .and. abs(jj-jj_qm)<=1 .and. abs(kk-kk_qm)<=1) then
                       imtrans = 0
                    else
                       ! Get the image transformed coordinates of MM group
                       ! and find closet distance^2, and returns closet MM images
                       ! (including primary image) as xyzim(1:3,1)
                       imtrans = -1
                       call IMTRNQM_cell(NOROT,imtrans,cutnb2,NTRANS,IMTRNS,xyzim_ref,XYZIM, &
                                         box_dim,cut_box_sq)
                    end if
                    i_grp_box(ii,jj,kk) = MAX(i_grp_box(ii,jj,kk),imtrans)
                 end if
              end do
           end do
        end do
        !
        ii_qm_old = ii_qm
        jj_qm_old = jj_qm
        kk_qm_old = kk_qm
     end if
  end do
  ! filling in i_map_box array.
  do mgrp = mgrp_start,mgrp_end           ! 1,ngrp
     ! find the box index
     ii= INT((xyz_g(1,mgrp)+xh_size(1))*r_cutnb) + 1
     jj= INT((xyz_g(2,mgrp)+xh_size(2))*r_cutnb) + 1
     kk= INT((xyz_g(3,mgrp)+xh_size(3))*r_cutnb) + 1
     i_map_box(1,mgrp) = ii
     i_map_box(2,mgrp) = jj
     i_map_box(3,mgrp) = kk
  end do
  if(numnod>1) then
     call IVDGBRE(i_grp_box,KPARPT_local)
     call IVDGBRE(i_map_box,LPARPT_local)
  end if

  !
  ! Loop over groups in primary and image cell, since QM atom could
  ! be exist as images when QM atoms are close to boundary.
  do qgrp = 1,ngrp
     if(qgrpqm(qgrp)) then          ! qm group (for mndo97, includes mixed qm/mm group as qm group.)
        ncount = ncount + 1
        !
        ! Here need to loop over all the groups in primary groups
        ! with all image transformations to be checked.
        do mgrp = mgrp_start,mgrp_end           ! 1,ngrp
           if (qgrpmm(mgrp)) then  ! mm group 
              ! for mndo97, it is possible that mgrp is the same as qgrp as the mixed qm/mm group
              !             is considered both the qm and mm group. 
              ! mgrp is here mm group.
              xyzim(1,1) = xyz_g(1,mgrp)
              xyzim(2,1) = xyz_g(2,mgrp)
              xyzim(3,1) = xyz_g(3,mgrp)

              ! find the box index
              !ii= INT((xyzim(1,1)+xh_size(1))*r_cutnb) + 1
              !jj= INT((xyzim(2,1)+xh_size(2))*r_cutnb) + 1
              !kk= INT((xyzim(3,1)+xh_size(3))*r_cutnb) + 1
              ii = i_map_box(1,mgrp)
              jj = i_map_box(2,mgrp)
              kk = i_map_box(3,mgrp)

              !              Get the image transformed coordinates of MM group
              !              and find closet distance^2, and returns closet MM images
              !              (including primary image) as xyzim(1:3,1)
              imtrans = i_grp_box(ii,jj,kk)
              if(imtrans >= 0) then
                 ! this group (including image transformed case) is within the cutoff checking box.
                 if(imtrans > 0) then
                    ! this box needs to be checked.
                    call IMTRNQM(NOROT,ncount,imtrans,cutnb2,NTRANS,IMTRNS,xyzqm,XYZIM, &
                                 box_dim,cut_box_sq)
                 end if

                 delta_xyz(1:3) = xyzim(1:3,1) - xyzqm(1:3,ncount)
                 rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3) 

                 ! USE THE FIRST SHORTEST distance for a QM/MM PAIR IN QM/MM group list.
                 q_do_this_group=.true.
                 do ij = 1,nqmgrp
                    if(ij /= ncount) then
                       xyz_local(1:3)=xyzim(1:3,1)-xyzqm(1:3,ij)
                       rr2 = xyz_local(1)*xyz_local(1)+xyz_local(2)*xyz_local(2)+xyz_local(3)*xyz_local(3)
                       if (rr2 < rij2 ) then
                          q_do_this_group=.false.
                          EXIT
                       end if
                    end if
                 end do
                 if(q_do_this_group .and. rij2 < cutnb2) then
                    imtrans_local(mgrp)=imtrans
                    iqmmm_pair(mgrp)   =ncount
                 end if
              end if
           end if           !(qgrpmm(mgrp))
        end do              ! mgrp = 1,ngrp
     end if                 ! qgrpqm(qgrp) for mndo97
  end do                    ! qgrp = 1,ngrp
  ! go broadcast.
  if(numnod>1) then
     call IVDGBRE(imtrans_local,JPARPT_local)
     call IVDGBRE(iqmmm_pair,JPARPT_local)
  end if

  ! now loop over again.
  ncount  = 0
  do qgrp = 1,ngrp
     if(qgrpqm(qgrp)) then          ! qm group (for mndo97, includes mixed qm/mm group as qm group.)
        ncount = ncount + 1
        qstart = igpbs(qgrp) + 1
        qstop  = igpbs(qgrp + 1)
        num_mm_atm = num_mm_atm + (qstop-qstart+1) ! add number of mm atoms in the qm group for extra buffer.

        ! we loop over all mm groups.
        do mgrp = 1,ngrp
           ! This group is within cutoff (mm group including mixed qm/mm groups).
           if(iqmmm_pair(mgrp) == ncount) then  ! this group already satisfies rij2 < cutnb2.
              mstart = igpbs(mgrp) + 1
              mstop  = igpbs(mgrp + 1)

              ! Image transform mapping index for MM atoms to swap around the qm atoms. 
              if (imtrans>0) IMATTQ(mstart:mstop)=imtrans_local(mgrp)

              if (nb >= maxnnb) then
                 Call Wrndie (-5,'<GUQME3_MNDO>','QM/MM group list error.')
                 return
              else
                 nb = nb + 1
                 jnb(nb) = mgrp
                 num_mm_atm = num_mm_atm + (mstop-mstart+1)
              endif
              !========================================================================
           end if           ! (qgrpmm(mgrp))
        end do              ! mgrp = 1,ngrp
     end if                 ! qgrpqm(qgrp) for mndo97
     inb(qgrp)   = nb
     inbex(qgrp) = nbex
  end do                    ! qgrp = 1,ngrp
#else
  ! non-parallel case.
  !
  ! Loop over groups in primary and image cell, since QM atom could
  ! be exist as images when QM atoms are close to boundary
  Do qgrp = 1,ngrp
     if(qgrpqm(qgrp)) then          ! qm group (for mndo97, includes mixed qm/mm group as qm group.)
        ncount = ncount + 1
        qstart = igpbs(qgrp) + 1
        qstop  = igpbs(qgrp + 1)
        num_mm_atm = num_mm_atm + (qstop-qstart+1) ! add number of mm atoms in the qm group for extra buffer.
        !
        ! Here need to loop over all the groups in primary groups
        ! with all image transformations to be checked.
        Do mgrp = 1,ngrp
           If (qgrpmm(mgrp)) then  ! mm group 
              ! for mndo97, it is possible that mgrp is the same as qgrp as the mixed qm/mm group
              !             is considered both the qm and mm group. 
              ! mgrp is here mm group.
              xyzim(1,1) = xyz_g(1,mgrp)
              xyzim(2,1) = xyz_g(2,mgrp)
              xyzim(3,1) = xyz_g(3,mgrp)

              !              Get the image transformed coordinates of MM group
              !              and find closet distance^2, and returns closet MM images
              !              (including primary image) as xyzim(1:3,1)
              imtrans = 0
              call IMTRNQM(NOROT,ncount,imtrans,cutnb2,NTRANS,IMTRNS,xyzqm,XYZIM, &
                           box_dim,cut_box_sq)

              delta_xyz(1:3) = xyzim(1:3,1) - xyzqm(1:3,ncount)
              rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3) 

              ! USE THE FIRST SHORTEST distance for a QM/MM PAIR IN QM/MM group list.
              q_do_this_group=.true.
              do II = 1,nqmgrp
                 if(ii /= ncount) then
                    xyz_local(1:3)=xyzim(1:3,1)-xyzqm(1:3,ii)
                    rr2 = xyz_local(1)*xyz_local(1)+xyz_local(2)*xyz_local(2)+xyz_local(3)*xyz_local(3)
                    if (rr2 < rij2 ) then
                       q_do_this_group=.false.
                       EXIT
                    end if
                 end if
              end do
              !--JG
              if(q_do_this_group) then
                 If ((.NOT.QMCUTF) .OR. (rij2 < cutnb2)) then
                    mstart = igpbs(mgrp) + 1
                    mstop  = igpbs(mgrp + 1)

                    ! Image transform mapping index for MM atoms to swap around the qm atoms. 
                    If (imtrans.gt.0) IMATTQ(mstart:mstop)=imtrans

                    If (nb .ge. maxnnb) then
                       Call Wrndie (-5,'<GUQME3_MNDO>','QM/MM group list error.')
                       return
                    Else
                       nb = nb + 1
                       jnb(nb) = mgrp
                       num_mm_atm = num_mm_atm + (mstop-mstart+1)
                    Endif
                    !========================================================================
                 Endif         ! ((.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2))
              end if        ! q_do_this_group
           Endif            ! (qgrpmm(mgrp))
        End do              ! mgrp = 1,ngrp
     Endif                  ! (.not. qgrpmm(qgrp))  ! or qgrpqm(qgrp) for mndo97
     inb(qgrp)   = nb
     inbex(qgrp) = nbex
  End do                    ! qgrp = 1,ngrp
#endif
  qdone = .true.

#if KEY_PARALLEL==1
  ! memory.
  if(associated(imtrans_local)) deallocate(imtrans_local)
  if(associated(iqmmm_pair))    deallocate(iqmmm_pair)
#endif
  !
  Return
END SUBROUTINE GUQME3_MNDO
!
#endif    /*MNDO97 case*/
!
SUBROUTINE GUPQMV(X,Y,Z)
  !------------------------------------------------------------------------
  !
  !     Create the QM/MM van der Waal's group interaction list.
  !
  use chm_kinds
  use dimens_fcm
  use memory
  !
  use bases_fcm
  !...  use coord
  use inbnd
  use nbndqm_mod

#if KEY_QUANTUM==1
  use quantm     
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  use gamess_fcm 
#endif
#if KEY_MNDO97==1
  use mndo97     
#endif
#if KEY_SQUANTM==1
  use squantm    
#endif
#if KEY_PARALLEL==1
  use parallel
#endif

  use psf
  use stream
  !
  ! namkh 09/25/04
  !     Adjust nonbonded group list for IMAGES.
  use image
#if KEY_PBOUND==1
  use pbound     
#endif
  !---   use nbutil_module,only: prnbd3
  implicit none
  !
  real(chm_real)   X(*),Y(*),Z(*)
  Integer  mxnbgp, nnbg, nnbg_local
  !
  Logical  qdone,qImage

  !!integer, parameter :: Maxqmg=200  ! moved to nbndqm_mod
  integer  imxnb(Maxqmg)

  integer, allocatable,dimension(:) :: jnbgx
  logical, save :: q_done_print=.false.
  !
  Integer  numqm,num_qm_grp
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  Logical QMCUTF    
#endif

#if KEY_QUANTUM==1
  numqm = natqm  
#endif
#if KEY_MNDO97==1
  numqm = numat  
#endif
#if KEY_SQUANTM==1
  numqm = natqm(1) 
#endif
#if KEY_MNDO97==1 || KEY_SQUANTM==1
  QMCUTF= .TRUE.   
  QNBGRP= .TRUE.   
#endif
  !
#if KEY_MNDO97==1  /* MNDO97 specific */
  !========================================================================
  !     Check whether IMAGE is used
  !========================================================================
  qImage =.FALSE.
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0) qImage=.TRUE.
#if KEY_PBOUND==1
  endif
#endif
#endif             /* MNDO97 specific */
  !
  !========================================================================
  !     Initialise the group lists.
  !========================================================================
  nullify(iqmgpv)
  nullify(jqmgpv)
  !========================================================================
  !     Allocate scratch space.
  !========================================================================
  call chmalloc('qmnbnd.src','GUPQMV','iqmgpv',ngrp,intgp=iqmgpv)
  !--------
  !jG 103000
  !jG   mxnbgp = (ngrp * (ngrp + 1)) / 2
  !jG   call chmalloc('qmnbnd.src','GUPQMV','jnbgx',mxnbgp,intg=jnbgx)
  !--------
  if(allocated(qgrpmm) .and. size(qgrpmm) .ne. ngrp) then
     call chmdealloc('qmnbnd.src','GUPQMV','qgrpmm',size(qgrpmm),log=qgrpmm)
  end if
  if(allocated(xyzg) .and. size(xyzg).ne.(3*ngrp)) then
     call chmdealloc('qmnbnd.src','GUPQMV','xyzg',3,size(xyzg,2),crl=xyzg)
  end if
  if(.not.allocated(qgrpmm)) call chmalloc('qmnbnd.src','GUPQMV','qgrpmm',ngrp,log=qgrpmm)
  if(.not.allocated(xyzg))   call chmalloc('qmnbnd.src','GUPQMV','xyzg',3,ngrp,crl=xyzg)

  !========================================================================
  !     Calculate the group centres of geometry.
  !========================================================================
  Call Grpcen_G(ngrp,igpbs,x,y,z,xyzg)
  !========================================================================
  !     Fill the Qgrpmm array.
  !========================================================================
  ! on return, qgrpmm=.true. for all mm and mixed qm/mm groups, while all qm groups are .false.
  Call Grpmm (ngrp,natom,igpbs,numqm,num_qm_grp &
#if KEY_QUANTUM==1
       ,qatlab &  
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
       ,igmsel &  
#endif
       ,qgrpmm)
  !--------
  !jG  Changed to include qm/mm group pairs in the qm/mm non-bonded list
  call alnbsp(ngrp,qgrpmm,mxnbgp,imxnb)
  mxnbgp = mxnbgp*ngrp
  call chmalloc('qmnbnd.src','GUPQMV','jnbgx',mxnbgp,intg=jnbgx)
  !--------
  !========================================================================
  !     Calculate the group lists.
  !========================================================================
#if KEY_MNDO97==1
  if(qImage) then
     Call Guqmv2_mndo(ngrp,qgrpmm,igrpex%a,jgrpex%a,xyzg, &
                      cutnb,mxnbgp,iqmgpv,jnbgx,nnbg,qdone,QMCUTF,.FALSE.)
  else
#endif
     Call Guqmv2 (ngrp,qgrpmm,igrpex%a,jgrpex%a,xyzg, &
                  cutnb,mxnbgp,iqmgpv,jnbgx,nnbg,qdone,QMCUTF,.FALSE.)
     !jG 0701                                                  QNBGRP)
#if KEY_MNDO97==1
  end if
#endif
  !
  If (qdone) then
     nqmgpv = nnbg
     call chmalloc('qmnbnd.src','GUPQMV','jqmgpv',nnbg,intgp=jqmgpv)
     jqmgpv(1:nnbg) = jnbgx(1:nnbg)
#if KEY_PARALLEL==1
     if(.not.q_done_print) then
        ! print info for each node. only first time.
        write (outu,'(a,i5,a,i6,a,i6,a)') ' GUPQMV> (node',mynod,') ',nnbg, &
          ' group interactions generated for ',ngrp,' groups.'
       q_done_print=.true.
     else
        ! only print the master node.
        if(prnlev.ge.2) then
        write (outu,'(a,i5,a,i6,a,i6,a)') ' GUPQMV> (node',mynod,') ',nnbg, &
          ' group interactions generated for ',ngrp,' groups.'
        end if
     end if
#else
     IF(PRNLEV.GE.2) Write (outu,'(a,i6,a,i6,a)') ' GUPQMV> ',nnbg, &
          ' group interactions generated for ',ngrp,' groups.'
#endif
  Else
     Call Wrndie(-5,'<GUPQMV>','Group update error.')
  Endif
  !========================================================================
  !     Free working stack_ space.
  !========================================================================
  call chmdealloc('qmnbnd.src','GUPQMV','xyzg',3,size(xyzg,2),crl=xyzg)
  call chmdealloc('qmnbnd.src','GUPQMV','jnbgx',size(jnbgx),intg=jnbgx)
  !
  IF(PRNLEV.GT.7) THEN
     WRITE(OUTU,24) ' The QM/MM van der Waal group list'
24   FORMAT(1X,A/)
     CALL PRNBD3(OUTU,NQMGPV,NGRP,iqmgpv,jqmgpv)
     WRITE(OUTU,24) ' The standard group exclusion list'
     CALL PRNBD3(OUTU,NNG14,NGRP,bnbnd%iglo14,bnbnd%ing14)
  ENDIF
  !
  Return
END SUBROUTINE GUPQMV

SUBROUTINE GUQMV2 (NGRP,QGRPMM,IGRPEX,JGRPEX,XYZ_G, &
     CUTNB, MXNBGP, INBG, JNBG, NNBG, QDONE, &
     QMCUTF, QNBGRP)
  !------------------------------------------------------------------------
  !
  !     Does the work of GUPQMV.
  !
  !     Ngrp           - the number of groups.
  !     Qgrpmm         - a logical array indicating whether the group is to be
  !                      included in the list generation.
  !     Igrpex, Jgrpex - the group exclusion lists.
  !     Xg, Yg, Zg     - the group centres of geometry.
  !     Cutnb          - the non-bond cutoff distance.
  !     Mxnbgp         - the maximum number of group interactions permitted.
  !     Inbg, Jnbg     - the group list arrays.
  !     Nnbg           - the number of group interactions generated.
  !     Qdone          - a logical flag indicating that list generation has
  !                      been successful.
  !     QMCUTF         - Flag: whether to use cutoffs
  !     QNBGRP         - a logical flag for QM-MM group interactions..jg
  !
  use chm_kinds
  ! JG 0601
#if KEY_PBOUND==1
  use pbound  
  use number  
#endif
#if KEY_PARALLEL==1
  use parallel
#endif
  use nbndqm_mod,only : Maxqmg

  implicit none
  ! JG 0601
  !
  Integer igrpex(*), inbg(*), jgrpex(*), jnbg(*), mxnbgp, ngrp, &
       nnbg
  Logical QMCUTF, QNBGRP, qdone, qgrpmm(*)
  real(chm_real)  cutnb, xyz_g(3,ngrp) 
  !
  !!integer, parameter :: Maxqmg=200  ! moved to nbndqm_mod

  real(chm_real) :: cutnb2, rij2,  rr2
  real(chm_real) :: delta_xyz(3),xyz_cen(3),xyz_local(3),box_inv(3),box_reg(3)
  integer :: iex, ifirst, igrp, ilast, jgrp
  integer :: nqmgrp,iqmgrp(Maxqmg),ii,ij,iigrp, mstop2
  logical :: qexc14, q_do_this_group, q_no_skip, q_grpmm_i
#if KEY_PARALLEL==1
  integer :: icnt_par,mmynod,nnumnod

  !
  mmynod = mynod
  nnumnod= numnod
#endif
  !
  cutnb2 = cutnb * cutnb

#if KEY_PBOUND==1 /*pbound*/
  If (qBoun) then
     if(.not.qCUBoun.or.qTOBoun) then
        if(qTOBoun) CALL WRNDIE(-5,'qmnbnd>','TOBoundary not supported')
     else
        CALL WRNDIE(-5,'qmnbnd>','QM/MM PBC not supported')
     end if
  End if
  box_inv(1)=BOXINV
  box_inv(2)=BOYINV
  box_inv(3)=BOZINV
  box_reg(1)=XSIZE
  box_reg(2)=YSIZE
  box_reg(3)=ZSIZE
#endif /* (pbound)*/

  call alnbsp(ngrp,qgrpmm,nqmgrp,iqmgrp)
  !
  nnbg  = 0
  ! Only loop over groups in primary cells.
  Do igrp = 1,ngrp
     xyz_cen(1)=xyz_g(1,igrp)
     xyz_cen(2)=xyz_g(2,igrp)
     xyz_cen(3)=xyz_g(3,igrp)
     !
     If (igrp .eq. 1) then
        ifirst = 1
     Else
        ifirst = igrpex(igrp - 1) + 1
     Endif
     ilast  = igrpex(igrp)

     q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
#if KEY_PARALLEL==1
     icnt_par = 0
#endif
     ! the following parallelization should be careful, if using QMPI. (check with DMT.)

     !
     ! Loop over all the groups.
     loopjg: do jgrp = igrp,ngrp
#if KEY_PARALLEL==1
        icnt_par = icnt_par + 1
        if(mmynod .ne. mod(icnt_par-1,nnumnod) .and. .not.QMPI) cycle loopjg 
#endif
        ! only loop over qm-mm grp pairs.
        if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
           qexc14 = .false.
           do iex = ifirst,ilast
              if (jgrpex(iex) .eq. jgrp) then
                 qexc14 = .true.
                 exit
              endif
           end do
           !
           delta_xyz(1) = xyz_g(1,jgrp) - xyz_cen(1)
           delta_xyz(2) = xyz_g(2,jgrp) - xyz_cen(2)
           delta_xyz(3) = xyz_g(3,jgrp) - xyz_cen(3)
           ! JG 6/1/01
           ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
           IF(qBoun) THEN
              IF(qCUBoun.or.qTOBoun) THEN
                 delta_xyz(1:3)=box_inv(1:3)*delta_xyz(1:3)
                 do ij=1,3
                    if(delta_xyz(ij).gt. half) delta_xyz(ij)=delta_xyz(ij)-one
                    if(delta_xyz(ij).lt.-half) delta_xyz(ij)=delta_xyz(ij)+one
                 end do
                 delta_xyz(1:3)=box_reg(1:3)*delta_xyz(1:3)
              ENDIF
           ENDIF
#endif /*  (pbound)*/
           rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
           !
           ! JG 3/21/01
           ! USE THE FIRST SHORTEST distance for a QM/MM PAIR IN QM/MM group list.
           q_do_this_group =.true.
#if KEY_MNDO97==0
           ! QNBGRP is set to be .false. in the subroutine call. for other qm packages.
           IF(QNBGRP) THEN     ! QNBGRP is set to be .false. in the subroutine call.
              do II = 1,nqmgrp
                 iigrp = iqmgrp(ii)
                 q_no_skip =.true.
                 IF(.NOT.QGRPMM(IGRP)) THEN
                    if(iigrp.eq.igrp) then
                       q_no_skip =.false.
                    else 
                       xyz_local(1) = xyz_g(1,jgrp) - xyz_g(1,iigrp)
                       xyz_local(2) = xyz_g(2,jgrp) - xyz_g(2,iigrp)
                       xyz_local(3) = xyz_g(3,jgrp) - xyz_g(3,iigrp)
                    end if
                 ELSEIF(.NOT.QGRPMM(JGRP)) THEN
                    if(IIGRP.eq.JGRP) then
                       q_no_skip =.false.
                    else 
                       xyz_local(1) = xyz_g(1,IGRP) - xyz_g(1,IIGRP)
                       xyz_local(2) = xyz_g(2,IGRP) - xyz_g(2,IIGRP)
                       xyz_local(3) = xyz_g(3,IGRP) - xyz_g(3,IIGRP)
                    end if
                 ENDIF
                 if(q_no_skip) then
                    ! JG 6/1/01
                    ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
                    IF(qBoun) THEN
                       IF(qCUBoun.or.qTOBoun) THEN
                          xyz_local(1:3)=box_inv(1:3)*xyz_local(1:3)
                          do ij=1,3                           
                             if(xyz_local(ij).gt. half) xyz_local(ij)=xyz_local(ij)-one
                             if(xyz_local(ij).gt.-half) xyz_local(ij)=xyz_local(ij)+one
                          end do
                          xyz_local(1:3)=box_reg(1:3)*xyz_local(1:3)
                       END IF
                    END IF
#endif /*  (pbound)*/
                    rr2 = xyz_local(1)*xyz_local(1)+xyz_local(2)*xyz_local(2)+xyz_local(3)*xyz_local(3)
                    if (rr2.lt.rij2 ) then
                       q_do_this_group =.false.
                       EXIT        ! do II = 1,nqmgrp
                    end if
                 end if
              end do               ! II = 1,nqmgrp
           ENDIF                   ! QNBGRP
#endif
           !--JG
           if(q_do_this_group) then
              if ((.NOT.QMCUTF) .OR. (rij2 < cutnb2)) then
                 if (nnbg .gt. mxnbgp) then
                    qdone = .false.
                    return
                 else
                    nnbg = nnbg + 1
                    if (qexc14) then
                       jnbg(nnbg) = - jgrp
                    else
                       jnbg(nnbg) =   jgrp
                    endif
                 endif
              endif
           end if             ! q_do_this_group
        endif                 ! qgrpmm(igrp) .and. .not. qgrpmm(jgrp) .or. ...
     end do  loopjg           ! jgrp = igrp,ngrp

     inbg(igrp) = nnbg
  End do                      ! igrp = 1,ngrp
  qdone = .true.
  !
  Return
END SUBROUTINE GUQMV2
!
#if KEY_MNDO97==1  /*MNDO97 case*/
SUBROUTINE GUQMV2_MNDO (NGRP,QGRPMM,IGRPEX,JGRPEX,XYZ_G, &
     CUTNB, MXNBGP, INBG, JNBG, NNBG, QDONE, &
     QMCUTF, QNBGRP)
  !------------------------------------------------------------------------
  !
  !     Does the work of GUPQMV.
  !
  !     Ngrp           - the number of groups.
  !     Qgrpmm         - a logical array indicating whether the group is to be
  !                      included in the list generation.
  !     Igrpex, Jgrpex - the group exclusion lists.
  !     Xg, Yg, Zg     - the group centres of geometry.
  !     Cutnb          - the non-bond cutoff distance.
  !     Mxnbgp         - the maximum number of group interactions permitted.
  !     Inbg, Jnbg     - the group list arrays.
  !     Nnbg           - the number of group interactions generated.
  !     Qdone          - a logical flag indicating that list generation has
  !                      been successful.
  !     QMCUTF         - Flag: whether to use cutoffs
  !     QNBGRP         - a logical flag for QM-MM group interactions..jg
  !
  use chm_kinds
  ! JG 0601
#if KEY_PBOUND==1
  use pbound  
  use number  
#endif
#if KEY_PARALLEL==1
  use parallel
#endif
!#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!  use omp_lib
!#endif                 /*OpenMP specific*/
  ! note GUQME3_MNDO: i_map_box is filled in.
  use nbndqm_mod,only : Maxqmg,i_map_box

  implicit none
  ! JG 0601
  !
  Integer igrpex(*), inbg(*), jgrpex(*), jnbg(*), mxnbgp, ngrp, &
       nnbg
  Logical QMCUTF, QNBGRP, qdone, qgrpmm(*)
  real(chm_real)  cutnb, xyz_g(3,ngrp) 
  !
  !!integer, parameter :: Maxqmg=200  ! moved to nbndqm_mod

  real(chm_real) :: cutnb2, rij2,  rr2
  real(chm_real) :: delta_xyz(3),xyz_cen(3),xyz_local(3),box_inv(3),box_reg(3)
  integer :: iex, ifirst, igrp, ilast, jgrp
  integer :: nqmgrp,iqmgrp(Maxqmg),ii,ij,iigrp, mstop2
  logical :: qexc14, q_do_this_group, q_no_skip, q_grpmm_i
#if KEY_PARALLEL==1
  integer :: icnt_par,mmynod,nnumnod,ii_ig,jj_ig,kk_ig,ii_jg,jj_jg,kk_jg
#if KEY_MNDOOPENMP==1
  integer,pointer :: jnbg_aux(:)=>Null()
  integer         :: id,nnbg_aux,nnbg_max_1,nnbg_max_2,nnbg_tot
  real(chm_real)  :: delta_xyz_aux(3),xyz_cen_aux(3),xyz_local_aux(3)
  integer,save    :: nnbg_max=0
  logical,save    :: q_opmm_init=.false.  ! if not initialized.
  logical         :: qdone_aux
#endif
  !
  mmynod = mynod
  nnumnod= numnod
  if(QMPI) then
     mmynod = 0
     nnumnod= 1
  end if
#endif
  !
  cutnb2 = cutnb * cutnb

#if KEY_PBOUND==1 /*pbound*/
  If (qBoun) then
     if(.not.qCUBoun.or.qTOBoun) then
        if(qTOBoun) CALL WRNDIE(-5,'qmnbnd>','TOBoundary not supported')
     else
        CALL WRNDIE(-5,'qmnbnd>','QM/MM PBC not supported')
     end if
  End if
  box_inv(1)=BOXINV
  box_inv(2)=BOYINV
  box_inv(3)=BOZINV
  box_reg(1)=XSIZE
  box_reg(2)=YSIZE
  box_reg(3)=ZSIZE
#endif /* (pbound)*/

  call alnbsp(ngrp,qgrpmm,nqmgrp,iqmgrp)
  !
#if KEY_PARALLEL==1
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
  if(.not.q_opmm_init) then
     ! if not initialized.
     nnbg_max    = 0
     nnbg_max_1  = 0
     nnbg_max_2  = 0
!$omp parallel private(id,igrp,q_grpmm_i,icnt_par,jgrp) NUM_THREADS(2)
     id = OMP_get_thread_num()
     if(id.eq.0) then
        ! Only loop over groups in primary cells.
        do igrp = 1,ngrp/2
           q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
           loopjg3: do jgrp = igrp+mmynod,ngrp,nnumnod
              ! only loop over qm-mm grp pairs.
              if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
                 nnbg_max_1 = igrp
              end if
           end do loopjg3
        end do
     else
        do igrp = ngrp/2+1,ngrp
           q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
           loopjg4: do jgrp = igrp+mmynod,ngrp,nnumnod
              ! only loop over qm-mm grp pairs.
              if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
                 nnbg_max_2 = igrp
              end if
           end do loopjg4
        end do
     end if
!$omp barrier
!$omp single
     nnbg_max = MAX(nnbg_max_1,nnbg_max_2)
!$omp end single
!$omp end parallel
     q_opmm_init=.true.
  end if

  nnbg_aux    = 0
  if(associated(jnbg_aux)) deallocate(jnbg_aux)
  allocate(jnbg_aux(mxnbgp))
  qdone       =.true.
  qdone_aux   = qdone
#endif                 /*OpenMP specific*/
#endif

  nnbg  = 0
#if KEY_PARALLEL==1    /*Parallel specific*/
#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
!$omp parallel NUM_THREADS(2)  &
!$omp & private(id,igrp,ifirst,ilast,q_grpmm_i,jgrp,qexc14,iex) &
!$omp & private(ii,ij,rij2,ii_ig,jj_ig,kk_ig,ii_jg,jj_jg,kk_jg) 
  id = OMP_get_thread_num()
  if(id.eq.0) then
     ! first thread
     do igrp = 1,nnbg_max/2
        xyz_cen(1)=xyz_g(1,igrp)
        xyz_cen(2)=xyz_g(2,igrp)
        xyz_cen(3)=xyz_g(3,igrp)
        !
        if (igrp .eq. 1) then
           ifirst = 1
        else
           ifirst = igrpex(igrp - 1) + 1
        end if
        ilast  = igrpex(igrp)

        q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
        ii_ig    = i_map_box(1,igrp)  ! location of box for igrp
        jj_ig    = i_map_box(2,igrp)  ! i_map_box are setup in GUQME3_MNDO routine.
        kk_ig    = i_map_box(3,igrp)
        ! the following parallelization should be careful, if using QMPI. (check with DMT.)

        !
        ! Loop over all the groups.
        loopjg: do jgrp = igrp+mmynod,ngrp,nnumnod
           ! only loop over qm-mm grp pairs.
           if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
              ii_jg  = i_map_box(1,jgrp)  ! location of box for jgrp
              jj_jg  = i_map_box(2,jgrp)
              kk_jg  = i_map_box(3,jgrp)
              ! do the check of this pair, if the two groups are within neighboring boxes.
              if(abs(ii_jg-ii_ig)<=1 .and. abs(jj_jg-jj_ig)<=1 .and. abs(kk_jg-kk_ig)<=1) then
                 ! this igrp and jgrp are neighboring or self-box.
                 qexc14 = .false.
                 do iex = ifirst,ilast
                    if (jgrpex(iex) .eq. jgrp) then
                       qexc14 = .true.
                       exit
                    endif
                 end do
                 !
                 delta_xyz(1) = xyz_g(1,jgrp) - xyz_cen(1)
                 delta_xyz(2) = xyz_g(2,jgrp) - xyz_cen(2)
                 delta_xyz(3) = xyz_g(3,jgrp) - xyz_cen(3)
                 ! JG 6/1/01
                 ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
                 IF(qBoun) THEN
                    IF(qCUBoun.or.qTOBoun) THEN
                          delta_xyz(1:3)=box_inv(1:3)*delta_xyz(1:3)
                       do ij=1,3
                          if(delta_xyz(ij).gt. half) delta_xyz(ij)=delta_xyz(ij)-one
                          if(delta_xyz(ij).lt.-half) delta_xyz(ij)=delta_xyz(ij)+one
                       end do
                       delta_xyz(1:3)=box_reg(1:3)*delta_xyz(1:3)
                    ENDIF
                 ENDIF
#endif /*  (pbound)*/
                 rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
                 if ((.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2)) then
                    if (nnbg .gt. mxnbgp) then
                       qdone = .false.
                    else
                       nnbg = nnbg + 1
                       if (qexc14) then
                          jnbg(nnbg) = - jgrp
                       else
                          jnbg(nnbg) =   jgrp
                       endif
                    endif
                 endif
              end if
           endif                 ! qgrpmm(igrp) .and. .not. qgrpmm(jgrp) .or. ...
        end do  loopjg           ! jgrp = igrp,ngrp

        inbg(igrp) = nnbg
     end do                      ! igrp = 1,ngrp
  else
     !second thread
     do igrp = nnbg_max/2+1,nnbg_max
        xyz_cen_aux(1)=xyz_g(1,igrp)
        xyz_cen_aux(2)=xyz_g(2,igrp)
        xyz_cen_aux(3)=xyz_g(3,igrp)
        !
        if (igrp .eq. 1) then
           ifirst = 1
        else
           ifirst = igrpex(igrp - 1) + 1
        end if
        ilast  = igrpex(igrp)

        q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
        ii_ig    = i_map_box(1,igrp)  ! location of box for igrp
        jj_ig    = i_map_box(2,igrp)  ! i_map_box are setup in GUQME3_MNDO routine.
        kk_ig    = i_map_box(3,igrp)
        ! the following parallelization should be careful, if using QMPI. (check with DMT.)

        !
        ! Loop over all the groups.
        loopjg2: do jgrp = igrp+mmynod,ngrp,nnumnod
           ! only loop over qm-mm grp pairs.
           if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
              ii_jg  = i_map_box(1,jgrp)  ! location of box for jgrp
              jj_jg  = i_map_box(2,jgrp)
              kk_jg  = i_map_box(3,jgrp)
              ! do the check of this pair, if the two groups are within neighboring boxes.
              if(abs(ii_jg-ii_ig)<=1 .and. abs(jj_jg-jj_ig)<=1 .and. abs(kk_jg-kk_ig)<=1) then
                 ! this igrp and jgrp are neighboring or self-box.
                 qexc14 = .false.
                 do iex = ifirst,ilast
                    if (jgrpex(iex) .eq. jgrp) then
                       qexc14 = .true.
                       exit
                    endif
                 end do
                 !
                 delta_xyz_aux(1) = xyz_g(1,jgrp) - xyz_cen_aux(1)
                 delta_xyz_aux(2) = xyz_g(2,jgrp) - xyz_cen_aux(2)
                 delta_xyz_aux(3) = xyz_g(3,jgrp) - xyz_cen_aux(3)
                 ! JG 6/1/01
                 ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
                 IF(qBoun) THEN
                    IF(qCUBoun.or.qTOBoun) THEN
                       delta_xyz_aux(1:3)=box_inv(1:3)*delta_xyz(1:3)
                       do ij=1,3
                          if(delta_xyz_aux(ij).gt. half) delta_xyz_aux(ij)=delta_xyz_aux(ij)-one
                          if(delta_xyz_aux(ij).lt.-half) delta_xyz_aux(ij)=delta_xyz_aux(ij)+one
                       end do
                       delta_xyz_aux(1:3)=box_reg(1:3)*delta_xyz_aux(1:3)
                    ENDIF
                 ENDIF
#endif /*  (pbound)*/
                 rij2  = delta_xyz_aux(1)*delta_xyz_aux(1)+delta_xyz_aux(2)*delta_xyz_aux(2) &
                        +delta_xyz_aux(3)*delta_xyz_aux(3)

                 if ((.NOT.QMCUTF) .OR. (rij2 .lt. cutnb2)) then
                    if (nnbg_aux .gt. mxnbgp) then
                       qdone_aux = .false.
                    else
                       nnbg_aux = nnbg_aux + 1
                       if (qexc14) then
                          jnbg_aux(nnbg_aux) = - jgrp
                       else
                          jnbg_aux(nnbg_aux) =   jgrp
                       endif
                    endif
                 endif
              end if
           endif                 ! qgrpmm(igrp) .and. .not. qgrpmm(jgrp) .or. ...
        end do  loopjg2           ! jgrp = igrp,ngrp

        inbg(igrp) = nnbg_aux
     end do                      ! igrp = 1,ngrp
  endif
!$omp barrier
!$omp single
  nnbg_tot = nnbg + nnbg_aux
!$omp end single
!$omp do
  do ii= nnbg_max+1,ngrp
     inbg(ii) = nnbg_tot
  enddo
!$omp end do
!$omp end parallel
  do ii =1,nnbg_aux
     ij = ii + nnbg
     jnbg(ij) = jnbg_aux(ii)
  enddo
  
  do ii= nnbg_max/2+1,nnbg_max
     inbg(ii) = inbg(ii) + nnbg
  enddo
  nnbg = nnbg_tot

  if(qdone .and. qdone_aux) then
     continue
  else
     qdone=.false.
     return          ! error 
  end if

#else                  /*OpenMP specific*/
  ! Only loop over groups in primary cells.
  loopig: do igrp = 1,ngrp
     xyz_cen(1)=xyz_g(1,igrp)
     xyz_cen(2)=xyz_g(2,igrp)
     xyz_cen(3)=xyz_g(3,igrp)
     !
     If (igrp .eq. 1) then
        ifirst = 1
     Else
        ifirst = igrpex(igrp - 1) + 1
     Endif
     ilast  = igrpex(igrp)

     q_grpmm_i= qgrpmm(igrp)       ! local copy of qgrpmm(igrp)
     ii_ig    = i_map_box(1,igrp)  ! location of box for igrp
     jj_ig    = i_map_box(2,igrp)  ! i_map_box are setup in GUQME3_MNDO routine.
     kk_ig    = i_map_box(3,igrp)
     ! the following parallelization should be careful, if using QMPI. (check with DMT.)

     !
     ! Loop over all the groups.
     loopjg: do jgrp = igrp+mmynod,ngrp,nnumnod
        ! only loop over qm-mm grp pairs.
        if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
           ii_jg  = i_map_box(1,jgrp)  ! location of box for jgrp
           jj_jg  = i_map_box(2,jgrp)
           kk_jg  = i_map_box(3,jgrp)
           ! do the check of this pair, if the two groups are within neighboring boxes.
           if(abs(ii_jg-ii_ig)<=1 .and. abs(jj_jg-jj_ig)<=1 .and. abs(kk_jg-kk_ig)<=1) then
              ! this igrp and jgrp are neighboring or self-box.
              qexc14 = .false.
              do iex = ifirst,ilast
                 if (jgrpex(iex) .eq. jgrp) then
                    qexc14 = .true.
                    exit
                 endif
              end do
              !
              delta_xyz(1) = xyz_g(1,jgrp) - xyz_cen(1)
              delta_xyz(2) = xyz_g(2,jgrp) - xyz_cen(2)
              delta_xyz(3) = xyz_g(3,jgrp) - xyz_cen(3)
              ! JG 6/1/01
              ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION
#if KEY_PBOUND==1 /*pbound*/
              IF(qBoun) THEN
                 IF(qCUBoun.or.qTOBoun) THEN
                    delta_xyz(1:3)=box_inv(1:3)*delta_xyz(1:3)
                    do ij=1,3
                       if(delta_xyz(ij).gt. half) delta_xyz(ij)=delta_xyz(ij)-one
                       if(delta_xyz(ij).lt.-half) delta_xyz(ij)=delta_xyz(ij)+one
                    end do
                    delta_xyz(1:3)=box_reg(1:3)*delta_xyz(1:3)
                 ENDIF
              ENDIF
#endif /*  (pbound)*/
              rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
              !
              if ((.NOT.QMCUTF) .OR. (rij2 < cutnb2)) then
                 if (nnbg .gt. mxnbgp) then
                    qdone = .false.
                    return
                 else
                    nnbg = nnbg + 1
                    if (qexc14) then
                       jnbg(nnbg) = - jgrp
                    else
                       jnbg(nnbg) =   jgrp
                    endif
                 endif
              endif
           endif
        endif                 ! qgrpmm(igrp) .and. .not. qgrpmm(jgrp) .or. ...
     end do  loopjg           ! jgrp = igrp,ngrp

     inbg(igrp) = nnbg
  end do  loopig                    ! igrp = 1,ngrp
#endif                 /*OpenMP specific*/
#else                  /*Parallel specific*/
  ! non-parallel case.
  loopig: do igrp = 1,ngrp
     xyz_cen(1)=xyz_g(1,igrp)
     xyz_cen(2)=xyz_g(2,igrp)
     xyz_cen(3)=xyz_g(3,igrp)
     !
     If (igrp .eq. 1) then
        ifirst = 1
     Else
        ifirst = igrpex(igrp - 1) + 1
     Endif
     ilast  = igrpex(igrp)

     q_grpmm_i= qgrpmm(igrp)  ! local copy of qgrpmm(igrp)
     !
     ! Loop over all the groups.
     loopjg: do jgrp = igrp,ngrp
        ! only loop over qm-mm grp pairs.
        if ((q_grpmm_i .and. .not. qgrpmm(jgrp)) .or. (.not. q_grpmm_i .and. qgrpmm(jgrp))) then
           qexc14 = .false.
           do iex = ifirst,ilast
              if (jgrpex(iex) .eq. jgrp) then
                 qexc14 = .true.
                 exit
              endif
           end do
           !
           delta_xyz(1) = xyz_g(1,jgrp) - xyz_cen(1)
           delta_xyz(2) = xyz_g(2,jgrp) - xyz_cen(2)
           delta_xyz(3) = xyz_g(3,jgrp) - xyz_cen(3)
           ! JG 6/1/01
           ! ADD SIMPLE PERIODIC BOUNDARY CONDITIONS FOR QM/MM INTERACTION

#if KEY_PBOUND==1 /*pbound*/
           IF(qBoun) THEN
              IF(qCUBoun.or.qTOBoun) THEN
                 delta_xyz(1:3)=box_inv(1:3)*delta_xyz(1:3)
                 do ij=1,3
                    if(delta_xyz(ij).gt. half) delta_xyz(ij)=delta_xyz(ij)-one
                    if(delta_xyz(ij).lt.-half) delta_xyz(ij)=delta_xyz(ij)+one
                 end do
                 delta_xyz(1:3)=box_reg(1:3)*delta_xyz(1:3)
              ENDIF
           ENDIF
#endif /*  (pbound)*/
           rij2  = delta_xyz(1)*delta_xyz(1)+delta_xyz(2)*delta_xyz(2)+delta_xyz(3)*delta_xyz(3)
           !
           if ((.NOT.QMCUTF) .OR. (rij2 < cutnb2)) then
              if (nnbg .gt. mxnbgp) then
                 qdone = .false.
                 return
              else
                 nnbg = nnbg + 1
                 if (qexc14) then
                    jnbg(nnbg) = - jgrp
                 else
                    jnbg(nnbg) =   jgrp
                 endif
              endif
           endif
        endif                 ! qgrpmm(igrp) .and. .not. qgrpmm(jgrp) .or. ...
     end do  loopjg           ! jgrp = igrp,ngrp

     inbg(igrp) = nnbg
  end do  loopig                    ! igrp = 1,ngrp
#endif    /*Parallel specific*/
  qdone = .true.

#if KEY_MNDOOPENMP==1  /*OpenMP specific*/
  ! memory.
  if(associated(jnbg_aux)) deallocate(jnbg_aux)
#endif                 /*OpenMP specific*/
  !
  Return
END SUBROUTINE GUQMV2_MNDO
#endif    /*MNDO97 case*/

SUBROUTINE QMMMATM(NATOM,qatmmm,QATLAB)
  !-------------------------------------------------------------------------
  !     Fill the logical array, qatmmm, to indicate whether a atom is
  !     molecular mechanics or quantum mechanics atom.
  !
  use chm_kinds
  use sizes
  implicit none

  integer :: natom,QATLAB(natom)
  logical :: qatmmm(natom)

  integer :: i,j

  do i=1,natom
     qatmmm(i)=.true.    ! mm atom

     ! if qm atoms: set as false.
#if KEY_QUANTUM==1
     if(QATLAB(i) .ge. 0) qatmmm(i)=.false.
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
     if(QATLAB(i).eq.1 .or. QATLAB(i).eq.2) qatmmm(i)=.false.
#endif
  end do

  return
END SUBROUTINE QMMMATM


SUBROUTINE GRPMM (NGRP, NATOM, IGPBS, NUMQM, num_qm_grp, QATLAB, QGRPMM)
  !-------------------------------------------------------------------------
  !     Fill the logical array, QGRPMM, to indicate whether a group consists
  !     of molecular mechanics or quantum mechanics atoms.
  !
  use chm_kinds
  use sizes
#if KEY_QUANTUM==1
  use qmlinkm        
#endif
#if KEY_MNDO97==1
  use mndgho         
#endif
#if KEY_SQUANTM==1
  use squantm, only : QLINK  
#endif
  implicit none
  !
  INTEGER  NGRP, NATOM, IGPBS(*), NUMQM, num_qm_grp, QATLAB(*)
  LOGICAL  QGRPMM(*)
  !
  INTEGER  IATOM, IGRP, ISTOP, ISTRT, NATM, NQM
  LOGICAL  QGHOL
  !
#if KEY_QUANTUM==1
  QGHOL = QMLINK    
#endif
#if KEY_MNDO97==1
  QGHOL = QLINK     
#endif
#if KEY_SQUANTM==1
  QGHOL = QLINK(1)  
#endif

  IF (NUMQM .GT. 0) THEN
     num_qm_grp = 0  ! counter of the number of qm groups, including mixed case.
     DO IGRP = 1,NGRP
        ISTRT = IGPBS(IGRP) + 1
        ISTOP = IGPBS(IGRP + 1)
        NATM  = ISTOP - ISTRT + 1
        NQM   = 0

        DO IATOM = ISTRT,ISTOP
#if KEY_QUANTUM==1
           IF (QATLAB(IATOM) .GE. 0) NQM = NQM + 1                      
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
           IF (QATLAB(IATOM).EQ.1.OR.QATLAB(IATOM).EQ.2) NQM = NQM + 1  
#endif
        END DO

        IF (NQM .EQ. NATM) THEN
           ! pure qm group
           QGRPMM(IGRP) = .FALSE.
           num_qm_grp   = num_qm_grp + 1
        ELSE IF (NQM .EQ. 0) THEN
           ! pure mm group
           QGRPMM(IGRP) = .TRUE.
        ELSE
           !-------------------------------------------------------------------------
           !      ALLOWING MIXED QM/MM GROUP FOR QM LINK ATOMS...JG 5/17/97
           !
           ! mixed qm-mm group
           IF(QGHOL) THEN
              num_qm_grp   = num_qm_grp + 1
              QGRPMM(IGRP) = .TRUE.         ! considered to be MM group.
           ELSE
              CALL WRNDIE(-5,'<GRPMM>','MIXED QM/MM GROUPS.')
           ENDIF
           !-------------------------------------------------------------------------
        ENDIF
     END DO
  ELSE
     num_qm_grp    = 0
     QGRPMM(1:NGRP)=.TRUE.
  ENDIF
  !
  RETURN
END SUBROUTINE GRPMM
!
SUBROUTINE GRPMM_EL (NGRP, NATOM, IGPBS, NUMQM, num_qm_grp, QATLAB, QGRPMM, QGRPQM)
  !-------------------------------------------------------------------------
  !     Fill the logical array, QGRPMM, to indicate whether a group consists
  !     of molecular mechanics or quantum mechanics atoms.
  !
  ! NOTE (namkh 2015-0521)
  !
  ! Special routine for qm/mm electrostatics, particularly for MNDO97
  ! In using GHO, mixed qm/mm group here is included as the QM group,
  ! not MM group. On the other hand, (in nbndq1 and nbndq2 and gupqmv),
  ! the mixed qm/mm group is considered to be MM group.
  !
  ! This routine is necessary, because the mixed qm/mm group interaction 
  ! with the MM group (for classical electrostatics and vdW interactiosn)
  ! has to be the same as before as the group contains MM atoms and their
  ! interactions with the rest MM atoms have to be handled in the nbonds
  ! routine. On the other hand, the QM atoms in the mixed qm/mm group,
  ! have to have list for MM groups for qm/mm non-bond interactions.
  !
  ! For now, this routine is only called when MNDO97 is used.
  !
  use chm_kinds
  use sizes
#if KEY_QUANTUM==1
  use qmlinkm        
#endif
#if KEY_MNDO97==1
  use mndgho         
#endif
#if KEY_SQUANTM==1
  use squantm, only : QLINK  
#endif
  implicit none
  !
  INTEGER  NGRP, NATOM, IGPBS(*), NUMQM, num_qm_grp, QATLAB(*)
  LOGICAL  QGRPMM(*),QGRPQM(*)
  !
  INTEGER  IATOM, IGRP, ISTOP, ISTRT, NATM, NQM
  LOGICAL  QGHOL
  !
#if KEY_QUANTUM==1
  QGHOL = QMLINK    
#endif
#if KEY_MNDO97==1
  QGHOL = QLINK     
#endif
#if KEY_SQUANTM==1
  QGHOL = QLINK(1)  
#endif

  IF (NUMQM .GT. 0) THEN
     num_qm_grp = 0  ! counter of the number of qm groups, including mixed case.
     DO IGRP = 1,NGRP
        ISTRT = IGPBS(IGRP) + 1
        ISTOP = IGPBS(IGRP + 1)
        NATM  = ISTOP - ISTRT + 1
        NQM   = 0

        DO IATOM = ISTRT,ISTOP
#if KEY_QUANTUM==1
           IF (QATLAB(IATOM) .GE. 0) NQM = NQM + 1
#elif KEY_MNDO97==1 || KEY_SQUANTM==1
           IF (QATLAB(IATOM).EQ.1.OR.QATLAB(IATOM).EQ.2) NQM = NQM + 1  
#endif
        END DO

        IF (NQM .EQ. NATM) THEN
           ! pure qm group
           QGRPMM(IGRP) = .FALSE.  ! not mm group
           QGRPQM(IGRP) = .TRUE.   ! qm group
           num_qm_grp   = num_qm_grp + 1
        ELSE IF (NQM .EQ. 0) THEN
           ! pure mm group
           QGRPMM(IGRP) = .TRUE.   ! mm group
           QGRPQM(IGRP) = .FALSE.  ! not qm group
        ELSE
           !-------------------------------------------------------------------------
           !      ALLOWING MIXED QM/MM GROUP FOR QM LINK ATOMS...JG 5/17/97
           !
           ! mixed qm-mm group
           IF(QGHOL) THEN
              num_qm_grp   = num_qm_grp + 1
              QGRPMM(IGRP) = .TRUE.  ! consider this group as both qm and mm groups.
              QGRPQM(IGRP) = .TRUE.  ! 
           ELSE
              CALL WRNDIE(-5,'<GRPMM>','MIXED QM/MM GROUPS.')
           ENDIF
           !-------------------------------------------------------------------------
        ENDIF
     END DO
  ELSE
     num_qm_grp    = 0
     QGRPMM(1:NGRP)=.TRUE.   ! mm group
     QGRPQM(1:NGRP)=.FALSE.  ! not qm group
  ENDIF
  !
  RETURN
END SUBROUTINE GRPMM_EL

!
SUBROUTINE GRPCEN_G(NGRP,IGPBS,X,Y,Z,XYZG)
  !-----------------------------------------------------------------------
  !     returns geometric group centers (no mass weighting)
  !
  use chm_kinds
  use number
#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none
  Integer ngrp, igpbs(*)
  real(chm_real)  x(*),y(*),z(*)
  real(chm_real) :: xyzg(3,ngrp) 
  !
  Integer irs,i,is,iq
  real(chm_real)  xsum,ysum,zsum,rnat
  integer, save :: old_N = 0
  integer, save :: istrt,ifalt
#if KEY_PARALLEL==1
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif

  !
  if(old_N .ne. ngrp) then
     old_N = ngrp
     istrt = 1
     ifalt = ngrp
#if KEY_PARALLEL==1
     if(numnod>1) then
        !call PARA_RANGE_nbnd(1,ngrp,numnod,mynod,istrt,ifalt)
        ! Prepare array for vector allgather calls using VDGBRE
        ! mapping for each node (Hard weird).
        JPARPT_local(0)=0
        do i=1,numnod
           JPARPT_local(i)= 3*(ngrp*i/numnod)
        end do
        istrt = ngrp*mynod/numnod + 1
        ifalt = ngrp*(mynod+1)/numnod 
     end if
#endif
  end if
  !
  do irs=istrt,ifalt        ! 1,ngrp
     is=igpbs(irs)+1
     iq=igpbs(irs+1)
     rnat=ONE/float(iq-is+1)
     xsum=ZERO
     ysum=ZERO
     zsum=ZERO
     do i=is,iq
        xsum=xsum+x(i)
        ysum=ysum+y(i)
        zsum=zsum+z(i)
     end do
     xyzg(1,irs)=xsum*rnat
     xyzg(2,irs)=ysum*rnat
     xyzg(3,irs)=zsum*rnat
  end do

#if KEY_PARALLEL==1
  if(numnod>1) call VDGBRE(xyzg,JPARPT_local)
#endif
  !
  RETURN
END SUBROUTINE GRPCEN_G

SUBROUTINE GRPCEN (NGRP,IGPBS,X,Y,Z,XG,YG,ZG)
  !-----------------------------------------------------------------------
  !     returns geometric group centers (no mass weighting)
  !
  use chm_kinds
  use number

#if KEY_PARALLEL==1
  use parallel
#endif
  implicit none
  Integer ngrp, igpbs(*)
  real(chm_real)  x(*),y(*),z(*),xg(*),yg(*),zg(*)
  !
  Integer irs,i,is,iq
  real(chm_real)  xsum,ysum,zsum,rnat
  integer, save :: old_N = 0
  integer, save :: istrt=1,ifalt=1
#if KEY_PARALLEL==1
  integer,save  :: JPARPT_local(0:MAXNODE)
#endif

  !
  if(old_N .ne. ngrp) then
     old_N = ngrp
#if KEY_PARALLEL==1
     JPARPT_local(0)=0
     do i=1,numnod
        JPARPT_local(i)= ngrp*i/numnod
     end do
     istrt = JPARPT_local(mynod)+1
     ifalt = JPARPT_local(mynod+1)
#else
     istrt = 1
     ifalt = ngrp
#endif
  end if
  !
  Do irs=istrt,ifalt        ! 1,ngrp
     is=igpbs(irs)+1
     iq=igpbs(irs+1)
     rnat=ONE/float(iq-is+1)
     xsum=ZERO
     ysum=ZERO
     zsum=ZERO
     Do i=is,iq
        xsum=xsum+x(i)
        ysum=ysum+y(i)
        zsum=zsum+z(i)
     end do
     xg(irs)=xsum*rnat
     yg(irs)=ysum*rnat
     zg(irs)=zsum*rnat
  End do

#if KEY_PARALLEL==1
  if(numnod>1) then
     call VDGBRE(xg,JPARPT_local)
     call VDGBRE(yg,JPARPT_local)
     call VDGBRE(zg,JPARPT_local)
  end if
#endif
  !
  RETURN
END SUBROUTINE GRPCEN

SUBROUTINE ALNBSP(ngrp,qgrpmm,maxnnb,iqmgrp)
  !  Allocates only space that is needed for qm-mm interactions.
  !  JG 103000
  implicit none
  integer ngrp, i, maxnnb
  integer iqmgrp(*)
  logical qgrpmm(*)

  maxnnb = 0
  do i = 1,ngrp
     if(.not.qgrpmm(i)) then
        maxnnb=maxnnb+1
        iqmgrp(maxnnb) = i
     endif
  enddo
  !
  RETURN
END SUBROUTINE ALNBSP

SUBROUTINE ALNBSP_EL(ngrp,qgrpqm,maxnnb,iqmgrp)
  !  Allocates only space that is needed for qm-mm interactions.
  !  JG 103000
  implicit none
  integer ngrp, i, maxnnb
  integer iqmgrp(*)
  logical qgrpqm(*)

  maxnnb = 0
  do i = 1,ngrp
     if(qgrpqm(i)) then  ! this group contains pure qm groups and mixed qm/mm groups.
        maxnnb=maxnnb+1
        iqmgrp(maxnnb) = i
     endif
  enddo
  !
  RETURN
END SUBROUTINE ALNBSP_EL
!
SUBROUTINE IMTRNQM(NOROT,IQM,IMTRANS,cutnb2,NTRANS, &
     IMTRNS,XYZQM,XYZIM,box_dim,cut_box_sq)
  !-----------------------------------------------------------------------
  !     Coordinate transform for 26 images and check whether
  !     which images are closet from QM atoms and return as
  !     XYZIM(1:3,1).
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  !
  LOGICAL NOROT
  INTEGER IQM,IMTRANS,NTRANS
  real(chm_real)  cutnb2,XYZQM(3,*),XYZIM(3,*),IMTRNS(*)
  real(chm_real) :: box_dim(3),cut_box_sq

  INTEGER ITRANS,IPT,Ncount
  real(chm_real)  delxyz(3) ,rij2
  real(chm_real)  deli(3),rijim2

  delxyz(1)=xyzim(1,1)-xyzqm(1,iqm)
  delxyz(2)=xyzim(2,1)-xyzqm(2,iqm)
  delxyz(3)=xyzim(3,1)-xyzqm(3,iqm)
  ! if distance is less than half of the box.
  if(delxyz(1).le.box_dim(1) .and. delxyz(2).le.box_dim(2)          &
                             .and. delxyz(3).le.box_dim(3)) return

  rij2     =delxyz(1)*delxyz(1)+delxyz(2)*delxyz(2)+delxyz(3)*delxyz(3)

  ! assumption: 
  ! only loop over, if the rij2 > cut_box_sq
  !
  !if(rij2.le.cutnb2) return
  if(rij2.le.cut_box_sq) return

  Ncount = 1
  if (NOROT) then
     do ITRANS=1,NTRANS
        IPT   =(ITRANS-1)*12
        Ncount=Ncount+1
        XYZIM(1,Ncount)=XYZIM(1,1)+IMTRNS(IPT+10)
        XYZIM(2,Ncount)=XYZIM(2,1)+IMTRNS(IPT+11)
        XYZIM(3,Ncount)=XYZIM(3,1)+IMTRNS(IPT+12)
     end do
  else
     do ITRANS=1,NTRANS
        IPT   =(ITRANS-1)*12
        Ncount=Ncount+1
        XYZIM(1,Ncount)=XYZIM(1,1)*IMTRNS(IPT+1)+ &
                        XYZIM(2,1)*IMTRNS(IPT+2)+ &
                        XYZIM(3,1)*IMTRNS(IPT+3)+ IMTRNS(IPT+10)
        XYZIM(2,Ncount)=XYZIM(1,1)*IMTRNS(IPT+4)+ &
                        XYZIM(2,1)*IMTRNS(IPT+5)+ &
                        XYZIM(3,1)*IMTRNS(IPT+6)+ IMTRNS(IPT+11)
        XYZIM(3,Ncount)=XYZIM(1,1)*IMTRNS(IPT+7)+ &
                        XYZIM(2,1)*IMTRNS(IPT+8)+ &
                        XYZIM(3,1)*IMTRNS(IPT+9)+ IMTRNS(IPT+12)
     end do
  end if

  !        Find closest distance square
  Ncount = 1
  DO ITRANS=1,NTRANS
     Ncount=Ncount+1
     deli(1)=XYZIM(1,Ncount)-xyzqm(1,iqm)
     deli(2)=XYZIM(2,Ncount)-xyzqm(2,iqm)
     deli(3)=XYZIM(3,Ncount)-xyzqm(3,iqm)
     rijim2=deli(1)*deli(1)+deli(2)*deli(2)+deli(3)*deli(3)

     if(rijim2 < rij2) then
        IMTRANS=ITRANS
        rij2   =rijim2
        if(rij2 <= cutnb2) EXIT
     end if
  END DO

  if(IMTRANS.NE.0) then
     Ncount    =IMTRANS+1
     xyzim(1,1)=xyzim(1,Ncount)
     xyzim(2,1)=xyzim(2,Ncount)
     xyzim(3,1)=xyzim(3,Ncount)
  end if

  RETURN
END SUBROUTINE IMTRNQM
!
#if KEY_MNDO97==1
SUBROUTINE IMTRNQM_cell(NOROT,IMTRANS,cutnb2,NTRANS,IMTRNS,XYZIM_ref,XYZIM, &
                        box_dim,cut_box_sq)
  !-----------------------------------------------------------------------
  !     Coordinate transform for 26 images and check whether
  !     which images are closet from QM atoms and return as
  !     XYZIM(1:3,1).
  !
  use chm_kinds
  use dimens_fcm
  use number
  implicit none
  !
  LOGICAL NOROT
  INTEGER IMTRANS,NTRANS
  real(chm_real)  cutnb2,XYZIM_ref(3),XYZIM(3,*),IMTRNS(*)
  real(chm_real) :: box_dim(3),cut_box_sq

  INTEGER ITRANS,IPT,Ncount
  real(chm_real)  delxyz(3) ,rij2
  real(chm_real)  deli(3),rijim2,cut_nb2_3

  !
  cut_nb2_3 = three*cutnb2 + 0.1d0 ! a bit of extra distance.

  delxyz(1)=xyzim(1,1)-xyzim_ref(1)
  delxyz(2)=xyzim(2,1)-xyzim_ref(2)
  delxyz(3)=xyzim(3,1)-xyzim_ref(3)
  rij2     =delxyz(1)*delxyz(1)+delxyz(2)*delxyz(2)+delxyz(3)*delxyz(3)

  if(rij2 <= cut_nb2_3) then
     IMTRANS=0       ! default value.
  else  ! (rij2 > cut_nb2_3)
     IMTRANS= -1 ! default that this box does not need to check.
     
     ! if the distance is less than half of the box.
     if(delxyz(1).le.box_dim(1) .and. delxyz(2).le.box_dim(2)          &
                                .and. delxyz(3).le.box_dim(3)) return
     
     ! assumption: 
     ! only loop over, if the rij2 > cut_box_sq
     !
     if(rij2 <= cut_box_sq) return

     Ncount = 1
     if (NOROT) then
        do ITRANS=1,NTRANS
           IPT   =(ITRANS-1)*12
           Ncount=Ncount+1
           XYZIM(1,Ncount)=XYZIM(1,1)+IMTRNS(IPT+10)
           XYZIM(2,Ncount)=XYZIM(2,1)+IMTRNS(IPT+11)
           XYZIM(3,Ncount)=XYZIM(3,1)+IMTRNS(IPT+12)
        end do
     else
        do ITRANS=1,NTRANS
           IPT   =(ITRANS-1)*12
           Ncount=Ncount+1
           XYZIM(1,Ncount)=XYZIM(1,1)*IMTRNS(IPT+1)+ &
                           XYZIM(2,1)*IMTRNS(IPT+2)+ &
                           XYZIM(3,1)*IMTRNS(IPT+3)+ IMTRNS(IPT+10)
           XYZIM(2,Ncount)=XYZIM(1,1)*IMTRNS(IPT+4)+ &
                           XYZIM(2,1)*IMTRNS(IPT+5)+ &
                           XYZIM(3,1)*IMTRNS(IPT+6)+ IMTRNS(IPT+11)
           XYZIM(3,Ncount)=XYZIM(1,1)*IMTRNS(IPT+7)+ &
                           XYZIM(2,1)*IMTRNS(IPT+8)+ &
                           XYZIM(3,1)*IMTRNS(IPT+9)+ IMTRNS(IPT+12)
        end do
     end if

     ! Find closest distance square
     Ncount = 1
     do ITRANS=1,NTRANS
        Ncount=Ncount+1
        deli(1)=XYZIM(1,Ncount)-xyzim_ref(1)
        deli(2)=XYZIM(2,Ncount)-xyzim_ref(2)
        deli(3)=XYZIM(3,Ncount)-xyzim_ref(3)
        rijim2=deli(1)*deli(1)+deli(2)*deli(2)+deli(3)*deli(3)

        if(rijim2 < rij2) then
           IMTRANS=ITRANS
           rij2   =rijim2
           if(rij2 <= cut_nb2_3) exit
        end if
     end do
  
     if(rij2 <= cut_nb2_3) then
        ! meaning this box is within the box to check after image transform.
        continue   ! IMTRANS is ITRANS
     else
        IMTRANS= -1 ! meaning this box does not need to check.
     end if
  end if
  !! debug: write(6,'(I3,2F12.5)') IMTRANS,sqrt(rij2),sqrt(three*cutnb2)
  RETURN
END SUBROUTINE IMTRNQM_cell
#endif
!
SUBROUTINE SWAPXYZ_IMAGE(NATOM,X,Y,Z,XTMP,YTMP,ZTMP,IMATTQ)
  !-----------------------------------------------------------------------
  !     Swap the images atoms that are included into QM/MM interactions
  !     into XTMP,YTMP,ZTMP temporary coordinates array
  !
  use chm_kinds
  use dimens_fcm
  !
  use image
  !     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1 /*pbound*/
  use pbound
#endif /*     (pbound)*/
  implicit none
  !
  real(chm_real)  X(*),Y(*),Z(*),XTMP(*),YTMP(*),ZTMP(*)
  INTEGER NATOM,IMATTQ(*)
  INTEGER I, IPT
  !
  !  For any cases
  XTMP(1:NATOM) = X(1:NATOM)
  YTMP(1:NATOM) = Y(1:NATOM)
  ZTMP(1:NATOM) = Z(1:NATOM)
  !
  !  Now handle with image swap
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0) THEN
        IF(NOROT) THEN
           DO I= 1,NATOM
              IF(IMATTQ(I) > 0) THEN
                 IPT = (IMATTQ(I)-1)*12
                 XTMP(I)=X(I)+IMTRNS(IPT+10)
                 YTMP(I)=Y(I)+IMTRNS(IPT+11)
                 ZTMP(I)=Z(I)+IMTRNS(IPT+12)
              END IF
           END DO
        ELSE
           DO I= 1,NATOM
              IF(IMATTQ(I) > 0) THEN
                 IPT = (IMATTQ(I)-1)*12
                 XTMP(I)=X(I)*IMTRNS(IPT+1)+Y(I)*IMTRNS(IPT+2)+  &
                      Z(I)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                 YTMP(I)=X(I)*IMTRNS(IPT+4)+Y(I)*IMTRNS(IPT+5)+ &
                      Z(I)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                 ZTMP(I)=X(I)*IMTRNS(IPT+7)+Y(I)*IMTRNS(IPT+8)+ &
                      Z(I)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
              END IF
           END DO
        END IF
     END IF
#if KEY_PBOUND==1
  endif                 
#endif

  !
  return
end SUBROUTINE SWAPXYZ_IMAGE

!!#if KEY_PARALLEL==1  /*paracheck*/
!!SUBROUTINE PARA_RANGE_nbnd(N1,N2,NPROCS,IRANK,ISTA,IEND)
!!  ! Divide a loop over processors
!!  ! From IBM MPI Programming publication
!!  ! N1 & N2 - beginning & end of non-parallel loop
!!  ! NPROCS - total no. of processors
!!  ! IRANK - processor rank
!!  ! ISTA & IEND - returned beginning & end of parallel loop for processor
!!  !               with rank IRANK
!!
!!  implicit none
!!
!!  INTEGER N1,N2,NPROCS,IRANK,ISTA,IEND,IWORK1,IWORK2
!!
!!  IWORK1 = (N2-N1+1)/NPROCS
!!  IWORK2 = MOD(N2-N1+1,NPROCS)
!!  ISTA = IRANK*IWORK1 + N1 + MIN(IRANK,IWORK2)
!!  IEND = ISTA + IWORK1 - 1
!!  IF (IWORK2 > IRANK) IEND = IEND + 1
!!
!!  RETURN
!!END SUBROUTINE PARA_RANGE_nbnd
!!#endif /*paracheck*/

#endif 
SUBROUTINE NULL_NBNDQM
  RETURN
END SUBROUTINE NULL_NBNDQM

