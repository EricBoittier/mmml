#if KEY_SQUANTM==1 /*squantm*/
      SUBROUTINE CH2SQM(NUMQM,IGMSEL,X,Y,Z,LFIRST)
!-----------------------------------------------------------------------
!     Setup QM/MM-non-bonded part for SQUANTM
!
!     Kwangho Nam 2004
!     Changes to support Group based Non-bond cutoff methods
!
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use memory

  use bases_fcm
  use datstr
  use contrl
  use inbnd
  use nbndqm_mod
  use psf
!
!     Adjust nonbonded group list for IMAGES.
  use image
!     Adjust nonbonded group list for simple pbc.
#if KEY_PBOUND==1
  use pbound  
#endif
      implicit none

      INTEGER NUMQM,num_qm_grp,IGMSEL(*)
      real(chm_real)  X(*),Y(*),Z(*)
      LOGICAL LFIRST

      LOGICAL Qimage

!-----------------------------------------------------------------------
      Qimage =.FALSE.
      IF(.NOT.USEDDT_nbond(BNBND)) CALL WRNDIE(-3,'<CH2SQM>','Nonbond data structure is not defined.')
#if KEY_PBOUND==1
      IF(.NOT.qBoun) THEN   
#endif
         IF(NTRANS.GT.0) THEN
            IF(LGROUP) THEN
               Qimage =.TRUE.
               IF(.NOT.USEDDT_image(BIMAG)) CALL WRNDIE(-3,'<CH2SQM>','Image nonbond data structure is not defined.')
            ELSE
               CALL WRNDIE(-2,'<CH2SQM>','QM/MM do not interact with Images under Atom Based Cutoff.')
            END IF
         END IF
#if KEY_PBOUND==1
      END IF                
#endif

!     Prepare the list for QM/MM-one-electron interaction list
!new
!     note xyzg(3,ngrp), this is different for mndo97/quantum
      if(ngrp_old.ne.ngrp) then
         if(allocated(xyzg))    call chmdealloc('sqnt_nbnd.src','CH2SQM','xyzg',size(xyzg,1),3,crl=xyzg)
         if(allocated(xyzg_sq)) call chmdealloc('sqnt_nbnd.src','CH2SQM','xyzg_sq',3,size(xyzg_sq,2),crl=xyzg_sq)
         if(allocated(igrpg))   call chmdealloc('sqnt_nbnd.src','CH2SQM','igrpg',size(igrpg),intg=igrpg)
         if(allocated(igrpg2))  call chmdealloc('sqnt_nbnd.src','CH2SQM','igrpg2',size(igrpg2),intg=igrpg2)
         if(allocated(qgrpmm))  call chmdealloc('sqnt_nbnd.src','CH2SQM','qgrpmm',size(qgrpmm),log=qgrpmm)
         ngrp_old=ngrp
      end if
      if(.not.allocated(xyzg))    call chmalloc('sqnt_nbnd.src','CH2SQM','xyzg',ngrp,3,crl=xyzg)
      if(.not.allocated(xyzg_sq)) call chmalloc('sqnt_nbnd.src','CH2SQM','xyzg_sq',3,ngrp,crl=xyzg_sq)
      if(.not.allocated(igrpg))   call chmalloc('sqnt_nbnd.src','CH2SQM','igrpg',ngrp,intg=igrpg)
      if(.not.allocated(igrpg2))  call chmalloc('sqnt_nbnd.src','CH2SQM','igrpg2',ngrp,intg=igrpg2)
      if(.not.allocated(qgrpmm))  call chmalloc('sqnt_nbnd.src','CH2SQM','qgrpmm',ngrp,log=qgrpmm)
!new

!     Calculate the group centres of geometry.
      Call Grpcen_sq (ngrp, igpbs, x, y, z, xyzg_sq)

!     Fill the QM group array.
      Call Grpmm (ngrp,natom,igpbs,numqm,num_qm_grp,igmsel,qgrpmm)

!     Fill MMINB array and checkup for non-list list
      Call Ch2sqm2(x,y,z,xyzg_sq,igmsel,iqmgpe(1),jqmgpe(1),igrpg,igrpg2,qgrpmm,LFIRST)

!-----------------------------------------------------------------------
      RETURN
      END

      SUBROUTINE CH2SQM2(X,Y,Z,XYZG,IGMSEL,INBX,JNBX,IGRPG,IGRPG2,QGRPMM,LFIRST)
!-----------------------------------------------------------------------
!     Do the actual work of CH2SQM:
!        Define CHARMM atoms as point charges and copy for SQUANTM
!        interface. Currently, only support Group based non-bond
!        cutoff
!
!     Author: Kwangho Nam (2005)
!
!     For atom-based cutoffs:
!        All atoms which are not quantum contribute to electrostatic
!        interaction in the QM part. See MJF reference:
!        J. Comp. Chem., Vol. 11, No. 6, 700-733 (1990)
!
!     For group-based cutoffs:
!        Use different scheme, See Nam et al..
!        J. Chem. Theor. Comput., Vol. 1, No. 1, 2-13 (2005)
!
!        1) Changes to support Group based Non-bond cutoff methods
!           with multiple QM groups.
!        2) Image atoms are included into QM/MM interaction when
!           group based non-bond cutoff methods are used.
!
!
  use chm_kinds
  use number
  use dimens_fcm
  use exfunc
  use consta
!
  use inbnd
  use squantm
  use psf
  use stream

#if KEY_PBOUND==1
  use pbound   
#endif
#if KEY_PARALLEL==1
  use parallel  
#endif

      implicit none

      real(chm_real)  X(*),Y(*),Z(*),XYZG(3,NGRP)
      INTEGER IGMSEL(*),INBX(*),JNBX(*),IGRPG(*),IGRPG2(*)
      LOGICAL QGRPMM(*),LFIRST
!
      real(chm_real) C2OFNB, C2ONNB,DELR(3), SCENT
      INTEGER I,J,IS,IQ,JS,JQ,IRS,JRS,NB,ITEMP,NLAT
      INTEGER NFLAGS,MM,NGFND,IGRP,JRSPR,NPR,natqm_1,natqm_2
      LOGICAL QPBCHECK,QCHECK,Qdual_check(2),qmswitch_local
      integer mstart,mstop
      integer ISTRT_CHECK                 ! for external function
!
!
!
      C2OFNB=CTOFNB*CTOFNB
      C2ONNB=CTONNB*CTONNB
      QPBCHECK=.FALSE.
#if KEY_PBOUND==1 /*pbound*/
      IF (qBoun) QPBCHECK=.TRUE.
#endif /* (pbound)*/

      IF(LFIRST) THEN
! first for QM atoms
         DO I = 1,NATQM(1) 
            mm               = iabs(qminb1_dual(i)) 
            mminb1_dual(i,1) = MM
            mminb2_dual(MM,1)= I
         END DO
!
         NQMGRP(1:NMAXGRP) = 0
         qm_grp_dual(1:maxgrp,1:2)=.false.    ! initialze
!
! Initialize the MMINB flag for the selection of MM atoms
         NFLAGS = -1

         ITEMP = 0
         DO I = 1, NATOM
            IF(IGMSEL(I).EQ.5.OR.IGMSEL(I).EQ.0) THEN
               ITEMP     = ITEMP + 1
               MM        = NATQM(1) + ITEMP
               mminb1_dual(MM,1)= I * NFLAGS
               mminb2_dual(i,1) = MM
            END IF
         END DO
!
! Let's find QM group number
         NGFND  = 1
         DO I = 1, NGRP
            IS = IGPBS(I) + 1
            IQ = IGPBS(I+1)
            NLAT = 0
            DO J = IS, IQ
               IF(IGMSEL(J).EQ.1.OR.IGMSEL(J).EQ.2) THEN
                  NLAT = NLAT + IGMSEL(J)
               END IF
            END DO
            IF(NLAT.GT.0) THEN
               IF(NGFND.GT.NMAXGRP) CALL WRNDIE(-5,'<CH2SQM2>', &
             'Too Many QM groups. Decrease less than 30')
               NGFND = NGFND + 1
               NQMGRP(NGFND) = I
               qm_grp_dual(i,1) =.true.
            END IF
         END DO
         NQMGRP(1) = NGFND - 1

! for dual quantum region
! setup mminb1_dual(i,2) and mminb2_dual(i,2) array here.
         if(QMFEP .or. QMLAY) then
            do i=1,natqm(2)
               mm               = IABS(qminb2_dual(i))
               mminb1_dual(i,2) = mm
               mminb2_dual(mm,2)= i
            end do

            nflags=-1
            itemp = 0
            if(QMFEP .or. QMLAY) then
               do i=1,natom
                  if(igmsel_dual(i).EQ.0 .or. igmsel_dual(i).eq.5) then
                     itemp             = itemp + 1
                     mm                = natqm(2) + itemp   
                     mminb1_dual(mm,2) = i*nflags
                     mminb2_dual(i,2)  = mm
                  end if
               end do
            end if

            do i=1,ngrp
               is   = igpbs(i) + 1
               iq   = igpbs(i+1)
               nlat =  0
               do j=is,iq
                  if(igmsel_dual(j).eq.1 .or. igmsel_dual(j).eq.2)  &
                                                             nlat=nlat+1
               end do
               if(nlat.gt.0) qm_grp_dual(i,2)=.true.
            end do   
         end if
      END IF
!
!
! Prepare to pass the charges and coordinates for MM atoms
! Reinitialize MMINB array for rechecking
      NFLAGS = -1
      natqm_1=NATQM(1)
      mminb1_dual(natqm_1+1:NATOM,1)= &
                             IABS(mminb1_dual(natqm_1+1:NATOM,1))*NFLAGS
!
! QGRPMM array,    if QM group: QGRPMM(I)=.FALSE.
!                     MM group: QGRPMM(I)=.TRUE.
!
! It only uses Group based cutoff for QM-MM interactions.
! First check of MM atoms in same group with QM atoms
      NLAT  = 0
      DO IRS = 1, NQMGRP(1)
         IGRP = NQMGRP(IRS+1)
         IS   = IGPBS(IGRP)+1
         IQ   = IGPBS(IGRP+1)
         DO I = IS, IQ
!      On the MMINB array when any MM atoms that need
!      to be included in QM/MM
            IF(IGMSEL(I).EQ.0) THEN
               MM               = mminb2_dual(i,1)
               mminb1_dual(MM,1)= IABS(mminb1_dual(MM,1))
               NLAT             = NLAT + 1
            END IF
         END DO
      END DO
!
! Initialize IGRPG array
      IGRPG(1:NGRP)  = 0
      IGRPG2(1:NGRP) = 0
!
!
!    Always IRS is for QM group and JRS is for MM group
      NB    = 0
      ITEMP = 0
      DO IRS=1,NGRP
         NPR  = INBX(IRS) - ITEMP
         ITEMP= INBX(IRS)

!        Loop over the MM groups
         If(npr .gt. 0) then

! down-grade: not separate between nodes.
!           mstart=ISTRT_CHECK(mstop,NPR)       ! get nstart abd mstop

            DO JRSPR=1,NPR
               JRS = IABS(JNBX(NB+JRSPR))

               DELR(1:3) = xyzg(1:3,irs)-xyzg(1:3,jrs)
#if KEY_PBOUND==1 /*pbound*/
               IF(QPBCHECK) CALL PBCHECK(DELR(1),DELR(2),DELR(3))
#endif /*      (pbound)*/
               SCENT=DELR(1)*DELR(1)+DELR(2)*DELR(2)+DELR(3)*DELR(3)
       
               IF(SCENT.LT.C2OFNB) THEN
                  IGRPG(JRS) = IGRPG(JRS) + 1
                  IF(SCENT.GT.C2ONNB) IGRPG2(JRS) = IRS
               END IF
            END DO

            NB = NB + NPR
         End if
      END DO

!...##IF PARALLEL
!      if(numnod.gt.1) then
!         call IGCOMB(IGRPG,NGRP)
!         call IGCOMB(IGRPG2,NGRP)
!      end if
!...##ENDIF

!
      NFLAGS=1
      IF(LQMEWD.AND.EWMODE.EQ.2) NFLAGS=-1
      DO IRS=1,NGRP
         IF(IGRPG(IRS).GE.1) THEN
            IS=IGPBS(IRS)+1
            IQ=IGPBS(IRS+1)
            DO I = IS, IQ
               IF(IGMSEL(I).EQ.0) THEN
                  MM   = mminb2_dual(i,1) 
                  IF(mminb1_dual(MM,1).LT.0) THEN
                     mminb1_dual(MM,1)=NFLAGS*IABS(mminb1_dual(MM,1)) 
                     NLAT      = NLAT + 1
                  END IF
               END IF
            END DO
         END IF
      END DO
!
      NCUTOFF(1)  = NLAT

!
! Need to work on mminb2_dual here.
      NLAT  = 0
      if (QMFEP .or. QMLAY) then
         NFLAGS = -1
         natqm_2=NATQM(2)
         mminb1_dual(natqm_2+1:NATOM,2)= &
                             IABS(mminb1_dual(natqm_2+1:NATOM,2))*NFLAGS
         do irs = 1, nqmgrp(1)     ! it should work, since 2nd qm region
            igrp = nqmgrp(irs+1)   ! is subset of 1st qm region.
            is   = igpbs(igrp)+1
            iq   = igpbs(igrp+1)
            do i=is,iq
               if(igmsel_dual(i).eq.0) then
                  mm               = mminb2_dual(i,2)
                  mminb1_dual(mm,2)= IABS(mminb1_dual(mm,2))
                  nlat             = nlat + 1
               end if
            end do
         end do

!  initialize IGRPG array.
         igrpg(1:ngrp) = 0
         igrpg2(1:ngrp)= 0
!
!  always irs is for QM group and jrs is for MM group
         nb    = 0
         itemp = 0
         do irs = 1, ngrp
            npr  = inbx(irs) - itemp
            itemp= inbx(irs)
!  loop over MM groups.
            if(npr .gt. 0) then
               if(qm_grp_dual(irs,2)) then  ! for 2nd qm region.
                 do jrspr = 1, npr
                   jrs = iabs(jnbx(nb+jrspr))
                   delr(1:3)=xyzg(1:3,irs)-xyzg(1:3,jrs)
#if KEY_PBOUND==1 /*pbound*/
                   if(QPBCHECK) call PBCHECK(delr(1),delr(2),delr(3))
#endif /*      (pbound)*/
                   scent=delr(1)*delr(1)+delr(2)*delr(2)+delr(3)*delr(3)

                   if(scent .lt. C2OFNB) then
                      igrpg(jrs) = igrpg(jrs) + 1
                      if(scent.gt.C2ONNB) igrpg2(jrs)=irs
                   end if
                 end do
               end if
               nb = nb + npr
            end if
         end do

! 
         NFLAGS=1
         if(LQMEWD.AND.EWMODE.EQ.2 .and. .not. QMLAY) NFLAGS=-1
         do irs=1,ngrp
            if(igrpg(irs).ge.1) then
               is=igpbs(irs)+1
               iq=igpbs(irs+1)
               do i=is,iq
                  if(igmsel_dual(i).eq.0) then
                     mm                  =mminb2_dual(i,2)
                     if(mminb1_dual(mm,2).lt.0) then
                        mminb1_dual(mm,2)=NFLAGS*IABS(mminb1_dual(mm,2))
                        nlat             =nlat + 1
                     end if
                  end if
               end do
            end if              ! igrpg(irs).ge.1
         end do                 ! irs=1,ngrp
      end if                    ! QMFEP .or. QMLAY
      NCUTOFF(2)  = NLAT
!
!     Setup for Switching function...
      Qdual_check(1) = (QMFEP .or. QMLAY)
      IF(QMSWTCH) THEN
         QCHECK         =.TRUE.
         Qdual_check(2) = .false.
         Call QMMM_Nbnd_Switch(NATOM,NGRP,natqm_1,qminb1_dual, &
                               IGPBS,IGRPG2, &
                               NQMGRP,C2OFNB,C2ONNB,XYZG, &
                               QMSWTCH,QPBCHECK,LFIRST,QCHECK, &
                               Qdual_check)
         IF(.NOT.QCHECK) THEN
            CALL WRNDIE(-1,'<CH2SQM2>', &
            'Error in setup Switching function for QM-MM interaction.')

!           Pass here..then turn off Switching..
            QMSWTCH=.FALSE.
            if(Prnlev.ge.2) WRITE(6,*) &
               'Set as do not use Switch function.'
         END IF
         if(QMFEP .or. QMLAY) then
            Qdual_check(2)=.true.
            if(QMFEP) then
               qmswitch_local = QMSWTCH
            else if(QMLAY) then                  ! should be false for QMLAY.
               qmswitch_local =.false.
            end if
            Call QMMM_Nbnd_Switch(NATOM,NGRP,natqm_2,qminb2_dual, &
                                  IGPBS,IGRPG2, &
                                  NQMGRP,C2OFNB,C2ONNB,XYZG, &
                                  qmswitch_local,QPBCHECK,LFIRST,QCHECK, &
                                  Qdual_check)
            if (.not.QCHECK) then               ! it shouldn't go further..
               CALL WRNDIE(-5,'<CH2SQM2>', &
         'Error in setup Switching function for 2nd QM-MM interaction.')
            end if
         end if
      END IF

      RETURN
      END

      SUBROUTINE GRPCEN_SQ (NGRP,IGPBS,X,Y,Z,XYZG)
!-----------------------------------------------------------------------
!     returns geometric group centers (no mass weighting)
!     replicate of subroutine GRPCEN (qmnbnd.src)
!
  use chm_kinds
  use number
#if KEY_PARALLEL==1
  use parallel 
#endif

      implicit none

      Integer ngrp, igpbs(*)
      real(chm_real)  x(*),y(*),z(*),xyzg(3,NGRP)
!
      Integer irs,i,is,iq,istrt,ifalt,ncnt
      real(chm_real)  xsum(3),rnat
      integer ISTRT_CHECK                 ! for external function
!
! initialize
      istrt = 1
      ifalt = ngrp
#if KEY_PARALLEL==1 /*paramain*/
      if(numnod.gt.1) then
         istrt = ISTRT_CHECK(ifalt,ngrp)

         xyzg=zero
      end if
#endif /* (paramain)*/

      Do irs=istrt,ifalt        ! 1,ngrp
         is=igpbs(irs)+1
         iq=igpbs(irs+1)
!        rnat=ONE/(iq-is+1)                          !CA, Minneapolis 09/06/2000
         rnat=ONE/float(iq-is+1)
         xsum(1:3)=zero
         Do i=is,iq
            xsum(1)=xsum(1)+x(i)
            xsum(2)=xsum(2)+y(i)
            xsum(3)=xsum(3)+z(i)
         End do
         xyzg(1:3,irs)=xsum(1:3)*rnat
      End do

#if KEY_PARALLEL==1
      if(numnod.gt.1) call GCOMBR(xyzg,3*ngrp)
#endif 
!!!!
!!!!     RETURN
!!!!     END
!!!!
#else /* (squantm)*/

      SUBROUTINE CH2SQM_DUMMY

#endif /* (squantm)*/
      RETURN
      END

