! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
! minimization for finite-temperature string
      module ftsm_min
#if (KEY_STRINGM==1) /*  automatically protect all code */
!
#if (KEY_STRINGM==1)
!
      use chm_kinds
      use ftsm_var
      use ftsm_util
      use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3 ! , only : RMSBestFit, rmsd
      use heurist, only: reuris ! only for dependency checking
      implicit none
      private
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
      logical, save, public :: ftsm_mini_initialized=.false.
      integer, save :: ftsm_mini_method=0 ! minimization method
      integer, parameter :: sd=1, conj=2 ! minimization method codes
      integer, save :: ftsm_mini_bath_iterations=50, ftsm_mini_forced_iterations=10 ! minimizer iterations
      real(chm_real), save :: ftsm_mini_step ! minimization step
!
       type(nonbondDataStructure), save :: ftsm_nbond_copy
       type(imageDataStructure), save :: ftsm_image_copy
!=====================================================================
! SUBROUTINES
      public ftsm_mini_init
      public ftsm_mini_done
      public ftsm_mini
!
      contains
!======================================================================
      subroutine ftsm_mini_init(comlyn, comlen)
      use ftsm_var
      use stream
      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use mpi
      use number

      use datstr

!======================================================================
      character(len=*) :: comlyn
      integer :: comlen
!
      integer :: isd=0, iconj=0, mlen
!
      logical :: qprint, qroot
!
      character(len=20) :: methods(2)
      data methods/ 'STEEPEST DESCENT','CONJUGATE GRADIENT'/
!
      character(len=len("FTSM_MINI_INIT>") ),parameter::whoami="FTSM_MINI_INIT>";!macro
!
      qroot=MPI_COMM_STRNG.ne.MPI_COMM_NULL
      qprint=qroot.and.ME_STRNG.eq.0
!
      ftsm_mini_method=0
      ftsm_mini_initialized=.false.
!
      isd=indxa(comlyn, comlen, 'SD') ; if (isd.gt.0) ftsm_mini_method=sd
      iconj=indxa(comlyn, comlen, 'CONJ') ; if (iconj.gt.0) ftsm_mini_method=conj
!
      if ((iconj).gt.0) then
        call wrndie(0,whoami,trim(' ONLY SD MINIMIZATION IS CURRENTLY SUPPORTED. NOTHING DONE'))
        return
      endif
!ccccccc CHECK FOR MULTIPLE OR MISSING OPTIONS
      if ((abs(isd)+abs(iconj)) .eq. 0) then
       if (qprint) then ; write(info,665) whoami ; write(OUTU,'(A)') pack(info,info.ne.'');info=''; ; endif ! qprint
 665 FORMAT(A,' MINIMIZATION METHOD NOT SPECIFIED. WILL USE STEEPEST DESCENT.')
       ftsm_mini_method=sd
      elseif ((isd+iconj) .gt. 1) then
       call wrndie(0,whoami,trim('MORE THAN ONE MINIMIZATION METHOD SPECIFIED. NOTHING DONE'))
       return
      endif
!
      ftsm_mini_step=gtrmf(comlyn, comlen, 'STEP', one*0.01) ! cast to correct kind
      if (ftsm_mini_step.lt.zero) then
         call wrndie(0,whoami,trim('MINIMIZATION STEP MUST NOT BE NEGATIVE. SET TO 0.01 .'))
         ftsm_mini_step=0.01
      endif
! number of minimization steps
      ftsm_mini_bath_iterations=gtrmi(comlyn, comlen, 'BITER', ione*50)
      if (ftsm_mini_bath_iterations.lt.0) then
         call wrndie(0,whoami,trim('NUMBER OF MINIMIZATION ITERATIONS CANNOT BE NEGATIVE. NOTHING DONE.'))
         return
      endif
!
      ftsm_mini_forced_iterations=gtrmi(comlyn, comlen, 'SITER', ione*10)
      if (ftsm_mini_forced_iterations.lt.0) then
         call wrndie(0,whoami,trim('NUMBER OF MINIMIZATION ITERATIONS CANNOT BE NEGATIVE. NOTHING DONE.'))
         return
      endif
!
! print summary
!
      if (qprint) then
       mlen=len_trim(methods(ftsm_mini_method))
       write(info(1),667) whoami, methods(ftsm_mini_method)(1:mlen)
       write(info(3),669) whoami, ftsm_mini_bath_iterations, ftsm_mini_forced_iterations
       write(info(2),668) whoami, ftsm_mini_step
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
!
 667 format(A,'  WILL MINIMIZE STRING USING ',A,' MINIMIZATION')
 669 format(A,'  FOR ',I5,' BATH ITERATIONS AND ',I5,' STRING ITERATIONS')
 668 format(A,'  WITH INITIAL STEP ',F10.5)
!
      endif
!
      ftsm_mini_initialized=.true.
      ftsm_nbond_image_data_initialized=.false. ! data for computing energy using CHARMM
!
      end subroutine ftsm_mini_init
!=====================================================================
      subroutine ftsm_mini_done()

      use datstr, only : FREEDT_nbond, FREEDT_image
      call FREEDT_nbond(ftsm_nbond_copy)
      call FREEDT_image(ftsm_image_copy)
      ftsm_nbond_image_data_initialized=.false.

      ftsm_mini_method=0
      ftsm_mini_initialized=.false.
      end subroutine ftsm_mini_done
!=====================================================================
      subroutine ftsm_mini(x, y, z &

     & ,wmain, nbond_data, image_data &
     & ,fx, fy, fz &

     & )

      use string
#if (KEY_MULTICOM==1)
      use multicom_aux; 
#endif
      use stream
      use number

      use chm_kinds, only : wrnlev
      use chm_types, only : nonbondDataStructure, imageDataStructure
      use datstr, only : DUPLDT_nbond, DUPLDT_image
      use energym, only : eprop, eterm, epot &


     & , energy ! module contains a wrapper

      use heurist & ! contains interface to updeci


& ; ! necessary to work regardless of 39
!
      use shake, only : qholo ! for turning off shake during the (unphysical) minimization
! use facts_module, only : FactsDataStructure, FCTBND, FCTBNDC, fctaim, fctrun ! an extremely ugly workaround for compatibility with FACTS

!
      real(chm_real) :: x(:), y(:), z(:)

! to restore forces : some energy functions will access x and dx directly (sigh)
      real(chm_real), optional :: fx(:), fy(:), fz(:)
      real(chm_real) :: fxt(size(x,1)), fyt(size(x,1)), fzt(size(x,1))
      logical :: saveforces
      integer :: wrnlev_
      real(chm_real) :: dummy(0)

!
! local variables
      logical :: qprint
 logical :: qshake
!
      real(chm_real) :: u(3,3)= Id3
      real(chm_real), pointer :: r_com(:), ow(:)
      real(chm_real), pointer, dimension(:,:) :: roi, roc, rfc, roc_rot, rfc_rot
!
      integer :: i, j, k
!
! temporary coordinate and force arrays
      real(chm_real) :: xt(size(x,1))
      real(chm_real) :: yt(size(x,1))
      real(chm_real) :: zt(size(x,1))
      real(chm_real) :: dxt(size(x,1))
      real(chm_real) :: dyt(size(x,1))
      real(chm_real) :: dzt(size(x,1))
      integer :: stringatoms(size(x,1))
      integer :: natom, ind, iter, nbath, nfree, ibeg
! other variables for minimization
      real(chm_real) :: oonbath, oonfree, gradnorm, mini_step, norm_step, energy_new, energy_old
!

! CHARMM - dependent energy evaluation routines/vars
      real(chm_real) :: wmain(:), wt(size(x,1))
      type(nonbondDataStructure) :: nbond_data
      type(imageDataStructure) :: image_data
! type(FactsDataStructure) :: fctbnd_copy, fctbndc_copy !##FACTS
      real(chm_real) :: eprop_save(size(eprop)), eterm_save(size(eterm)) ! arrays to save energy values (not certain how essential this is)
!
!
      character(len=len("FTSM_MINI>") ),parameter::whoami="FTSM_MINI>";!macro
!
      if (ftsm_com_on) then
       call wrndie(0,whoami,trim(' FTSM IN COM COORDINATES DOES NOT SUPPORT IMAGE MINIMIZATION. ABORT.'))
       return
      endif
!
! check if the user has made an initialization call to the minimizer
!
      if (.not.ftsm_mini_initialized) then
       call wrndie(0,whoami,trim('NO MINIMIZATION OPTIONS SELECTED. NOTHING DONE.'))
       return
      endif
!
! qroot =MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.SIZE_STRNG.gt.1
! qslave=MPI_COMM_LOCAL.ne.MPI_COMM_NULL.and.SIZE_LOCAL.gt.1
      qprint=MPI_COMM_STRNG.ne.MPI_COMM_NULL.and.ME_STRNG.eq.0
!
! initialize nbond data structure on first use
      if (.not.ftsm_nbond_image_data_initialized) then
       call DUPLDT_nbond(ftsm_nbond_copy, nbond_data)
       call DUPLDT_image(ftsm_image_copy, image_data)
       ftsm_nbond_image_data_initialized=.true.
      endif
!
      if (qprint) then
       write(info,691) whoami ; if(prnlev.ge. 5) write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
 691 format(/A,' PERFORMING STRING MINIMIZATION.')
!
! create coordinate arrays
!
      ow=>orientWeights
      r_com=>rcom(:,instant)
      roi=>r_o(:,:,instant)
      roc=>r_o(:,:,center)
      rfc=>r_f(:,:,center)
      roc_rot=>r_o(:,:,center_rot)
      rfc_rot=>r_f(:,:,center_rot)
      natom=size(x,1)
! write(0,*) 'natom:', natom
! use instantaneous coordinates to fill missing coords
! save forces:
      saveforces=present(fx).and.present(fy).and.present(fz)
      if (saveforces) then ; do i=1, natom ; fxt(i)=fx(i) ; fyt(i)=fy(i) ; fzt(i)=fz(i) ; enddo ; endif
!
      do i=1, natom ; xt(i)=x(i) ; yt(i)=y(i) ; zt(i)=z(i) ; enddo ! copy current coords
      stringatoms=0;
! copy string coordinates to corresponding instantaneous coordinates
! first, align string coordinates with the instantaneous coordinates, if needed
!
      if (.not. restraint_force_on) then ! load coordinates, unless restraints on, in which case, they are loaded
!
       if (qorient) then
! load orientation coordinates
        do k=1,norient
         ind=iatom_o(k)
         roi(k,1)=xt(ind)
         roi(k,2)=yt(ind)
         roi(k,3)=zt(ind)
        enddo
!
! translate forced atoms to centroid
!
        r_com(:)=0d0;
        do j=1,3 ; do k=1, norient;
         r_com(j) = r_com(j)+ow(k)*roi(k,j)
        enddo ; enddo
!
        roi(:,1)=roi(:,1)-r_com(1)
        roi(:,2)=roi(:,2)-r_com(2)
        roi(:,3)=roi(:,3)-r_com(3)
!
       endif ! qorient
      endif ! .not. restrained on
!
      if (qorient) then ! orient center image w.r.t. instantaneous coordinates
       call RMSBestFit(roi,roc,ow,u) ! superpose roi onto roc (assuming string is COM-free)
!
       rfc_rot = matmul(rfc, u) ! apply transpose (=inverse) of u to rfc
       rfc_rot(:,1)=rfc_rot(:,1)+r_com(1)
       rfc_rot(:,2)=rfc_rot(:,2)+r_com(2)
       rfc_rot(:,3)=rfc_rot(:,3)+r_com(3)
!
       if (qdiffrot) then
        roc_rot = matmul(roc, u) ! apply transpose (=inverse) of u to roc
! move to COM of the instantaneous coordinates
        roc_rot(:,1)=roc_rot(:,1)+r_com(1)
        roc_rot(:,2)=roc_rot(:,2)+r_com(2)
        roc_rot(:,3)=roc_rot(:,3)+r_com(3)
! insert orientation coordinates into all-atom coordinate array
        do k=1,norient
         ind=iatom_o(k)
         x(ind)=roc_rot(k,1)
         y(ind)=roc_rot(k,2)
         z(ind)=roc_rot(k,3)
         stringatoms(ind)=-1 ! these coordinates are fixed through all minimization (unless they are also forced atoms)
        enddo
       endif ! qdiffrot
      else ! no orientation
       rfc_rot=>rfc
      endif ! qorient
!
! insert forced coordinates into all-atom coordinate array
!
      do k=1,nforced
       ind=iatom_f(k)
       x(ind)=rfc_rot(k,1)
       y(ind)=rfc_rot(k,2)
       z(ind)=rfc_rot(k,3)
       stringatoms(ind)=1 ! these coordinates will be minimized (but at the end)
      enddo
!
! perform minimization with string coordinates fixed
!
      wt=wmain
      qshake=qholo ; if (qholo) qholo=.false. ! turn off shake for speed
      eprop_save=eprop ; eterm_save=eterm ! save current energy value, since CHARMM uses it to determine stability
      wrnlev_=wrnlev ! save warning level
!
      mini_step=ftsm_mini_step ! initial minimization step
!
      nbath=ithree*count(stringatoms.eq.0) ; if (nbath.gt.0) oonbath=one/nbath ;
      nfree=ithree*count(stringatoms.ne.-1) ; if (nfree.gt.0) oonfree=one/nfree ;
!
      if (nbath.eq.0) then ; ibeg = ftsm_mini_bath_iterations+1 ; else ; ibeg=1 ; endif ! skip bath iterations is there are no bath atoms
!
      do iter=ibeg, ftsm_mini_bath_iterations+ftsm_mini_forced_iterations
       call UPDECI(iter, x, y, z, wt, 0, dummy, dummy, dummy, dummy, dummy, dummy) ! following calling format in SD
! someone in his/her infinite wisdom decided to change arguments in energy routine from c37 to c39 to accommodate optional args....
       call ENERGY(x, y, z, dxt, dyt, dzt, ftsm_nbond_copy, ftsm_image_copy, 1, 0, dummy)
       energy_new=eprop(epot)
! SD hardwired for now:
! adaptive minimization strategy (a la CHARMM)
!
       if (iter .eq. ftsm_mini_bath_iterations+1) then
 wrnlev=wrnlev_ ! restore old level (works only is siter>0)
        mini_step=ftsm_mini_step ! reset step for string minimization
       elseif (energy_new .lt. energy_old .and. iter .gt. ibeg) then
 wrnlev=-1 ! turn off warnings for the first part of minimization
        mini_step=mini_step*1.5d0 ! accelerate
       else
        mini_step=mini_step*half ! decelerate
       endif
       energy_old=energy_new
!
! note that the evolution is not parallel
       if (iter.le.ftsm_mini_bath_iterations) then
! first, minimize the instantaneous atoms with the string atoms fixed
        where(stringatoms.ne.0) ! zero out gradients corresponding to string atoms
         dxt=zero ; dyt=zero; dzt=zero
        endwhere
        gradnorm = sqrt ( ( dot_product(dxt,dxt)+dot_product(dyt,dyt)+dot_product(dzt,dzt) ) * oonbath)
!
       else ! now minimize all coordinates except the string orientation coordinates
        where(stringatoms.eq.-1) ! gradients on orientation atoms zero-ed
         dxt=zero ; dyt=zero; dzt=zero
        endwhere
        gradnorm = sqrt ( ( dot_product(dxt,dxt)+dot_product(dyt,dyt)+dot_product(dzt,dzt) ) * oonfree)
       endif
!
       norm_step = mini_step/max(gradnorm,RSMALL) ! using RSMALL might lead to oscillations
       x = x - norm_step * dxt
       y = y - norm_step * dyt
       z = z - norm_step * dzt
!
!write(0,*) 'step: ', iter, mini_step, norm_step, oonfree, oonbath
!
      enddo ! iterations
!
! put minimized string coordinates back into r_f array
      do k=1,nforced
        ind=iatom_f(k)
        rfc_rot(k,1)=x(ind)
        rfc_rot(k,2)=y(ind)
        rfc_rot(k,3)=z(ind)
      enddo
!
      if (qorient) then
        u=transpose(u)
        rfc = matmul(rfc_rot, u) ! rotate back for consistency
        if (qdiffrot) call ftsm_update_overlap_coor(ione) ! update orientation coordinates
      endif
!
      call ftsm_save_com() ! remove COM from center coordinates
!
      do i=1, natom ; x(i)=xt(i) ; y(i)=yt(i) ; z(i)=zt(i) ; enddo ! restore original coordinates
!
      if (saveforces) then ; do i=1, natom ; fx(i)=fxt(i) ; fy(i)=fyt(i) ; fz(i)=fzt(i) ; enddo ; endif ! restore forces
      eprop=eprop_save ; eterm=eterm_save ! restore current energy values
      qholo=qshake
      wrnlev=wrnlev_
!
      end subroutine ftsm_mini
!========================================================================
#endif
#endif /* automatically protect all code */
      end module ftsm_min
