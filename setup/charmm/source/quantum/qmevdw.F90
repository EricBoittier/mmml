#if KEY_QUANTUM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1
SUBROUTINE EVDWQM (EVDW, X, Y, Z, DX, DY, DZ)
  !------------------------------------------------------------------------
  !
  !     Calculate the van der Waal's energies and first derivatives using a
  !     group based cutoff between the quantum mechanical and molecular
  !     mechanical atoms. A switching function is applied to interactions
  !     on a group/group basis.
  !
  !     Evdw   - the van der Waal's energy.
  !     Qupdat - a logical flag indicating that the group lists should be
  !              updated.
  !
  use chm_kinds
  use dimens_fcm
  use memory
  use bases_fcm
  use inbnd
  use psf
  use quantm, only : natom_check,dxm_qmmm
  use image
#if KEY_PBOUND==1
  use pbound   
#endif
  implicit none
  !
  real(chm_real)  EVDW, X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
  Logical Qimage
  !
  Qimage =.FALSE.
#if KEY_PBOUND==1
  if(.not.qboun) then   
#endif
     IF(NTRANS.GT.0) THEN
        IF(LGROUP) THEN
           Qimage =.TRUE.
        END IF
     END IF
#if KEY_PBOUND==1
  endif                 
#endif
  !----------------------------------------------------------------------
  ! temporariliy ignore QM/MM vdW interaction with images..
  ! this will be handled in regular non-bond interaction routine.
  ! Just, here only incldue vdW interactions of QM groups with
  ! MM groups in primary cell.
  ! ------------------------namkh 09/28/04
  Qimage =.FALSE.
  !----------------------------------------------------------------------
  if(natom_check.ne.natom) then
     if(allocated(dxm_qmmm)) call chmdealloc('qmevdw.src','EVDWQM','dxm_qmmm',size(dxm_qmmm),crl=dxm_qmmm)
     natom_check=natom
  end if
  if(.not.allocated(dxm_qmmm))call chmalloc('qmevdw.src','EVDWQM','dxm_qmmm',3*natom,crl=dxm_qmmm)
  !
  !=======================================================================
  !
  If(Qimage) then
     CALL EVDWQM3 (EVDW,X,Y,Z,DX,DY,DZ, &
          dxm_qmmm(1:natom),dxm_qmmm(natom+1:2*natom),dxm_qmmm(2*natom+1:3*natom)) 
  Else
     CALL EVDWQM2 (EVDW,X,Y,Z,DX,DY,DZ, &
          dxm_qmmm(1:natom),dxm_qmmm(natom+1:2*natom),dxm_qmmm(2*natom+1:3*natom))
  End if
  !
  !=======================================================================
  !
  RETURN
END SUBROUTINE EVDWQM
!
SUBROUTINE EVDWQM2 (EVDW,X,Y,Z,DX,DY,DZ,DXM,DYM,DZM)
  !------------------------------------------------------------------------
  !
  !     Does the work of EVDWQM
  !
  !     Evdw   - the van der Waal's energy.
  !     Qupdat - a logical flag indicating that the group lists should be
  !              updated.
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor        
#endif

  use chm_kinds
  use dimens_fcm
  use memory
  use number
  use exelecm
  use inbnd
  use nbndqm_mod
  use enbondg
  use param
  use psf
#if KEY_FLUCQ==1
  use flucq  
#endif
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel  
#endif
#if KEY_LJPME==1
  use energym, only: qeterm, eterm, ljexc
#endif
  implicit none
  !
  real(chm_real)  etemp, evdw,evdwt,x(*),y(*),z(*),dx(*),dy(*),dz(*)
  real(chm_real)  dxm(*),dym(*),dzm(*)
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,I
  !
  !========================================================================
  !     Calculate the van der Waal's energies.
  !========================================================================

  if(allocated(xyzg)) then
     if(size(xyzg,1).lt.ngrp) call chmdealloc('qmevdw.src','EVDWQM2','xyzg',size(xyzg,1),3,crl=xyzg)
  end if
  if(allocated(qg_qmmm)) then
     if(size(qg_qmmm).lt.ngrp) call chmdealloc('qmevdw.src','EVDWQM2','qg_qmmm',size(qg_qmmm),crl=qg_qmmm)
  end if
  if(allocated(ioff_qmmm)) then
     if(size(ioff_qmmm).lt.natc) call chmdealloc('qmevdw.src','EVDWQM2','ioff_qmmm',size(ioff_qmmm),intg=ioff_qmmm)
     natc_old=natc
  end if
  if(.not.allocated(xyzg))      call chmalloc('qmevdw.src','EVDWQM2','xyzg',ngrp,3,crl=xyzg)
  if(.not.allocated(qg_qmmm))   call chmalloc('qmevdw.src','EVDWQM2','qg_qmmm',ngrp,crl=qg_qmmm)
  if(.not.allocated(ioff_qmmm)) call chmalloc('qmevdw.src','EVDWQM2','ioff_qmmm',natc,intg=ioff_qmmm)

  !
  evdw = zero
  evdwt= zero
  !
  dxm(1:natom)=Zero
  dym(1:natom)=Zero
  dzm(1:natom)=Zero
  !
  ! Set the "lshft" flag to false in this call regardless of the specified
  ! nonbond optiopns.  This avoids the use of the hidden "shift" option
  ! in EGROUP for van der Waal interaction (i.e. vdw truncation). - BRB
  !
  ! Ben Webb, 2000, Call changed to #ass LCENTR=-1 (not .true.) as EGROUP
  ! expects an integer, not a logical! -BW
  CALL EGROUP(etemp,evdwt,natom,NST2,jqmgpv, &
       iqmgpv,cg,RSCLF,dxm,dym,dzm,1,ngrp,igpbs, &
       igptyp,jatmex%a,iatmex%a, &
       .false.,.true.,lcons,.false.,lfswt,ldbexp, &
       cnba,cnbb,maxcn,itc,iac,natc,ioff_qmmm,x,y,z, &
       ctonnb,ctofnb,eps,e14fac,wrnmxd,.false., (/ ZERO /), &
#if KEY_NBIPS==1
       LVIPS,LEIPS,                          & 
#endif
       QETEN,QETSR,                                 & 
#if KEY_MC==1
       -1,                                         & 
#endif
       xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),qg_qmmm, &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,                               & 
#endif
       (/ ZERO /), (/ 0 /), .false. &
#if KEY_WCA==1
       ,.FALSE.,ONE,WCA                            & 
#endif
#if KEY_LJPME==1
       , QETERM(LJEXC), ETERM(LJEXC)  &
#endif

       ,QEXTND)                                   ! qc050701
  !
  do i=1,natom  ! ATFRST,ATLAST as each node has different non-bond list. (see GUQMV2)
     dx(i)=dx(i)+dxm(i)
     dy(i)=dy(i)+dym(i)
     dz(i)=dz(i)+dzm(i)
  end do
  !
  ! go parallel, each node has different value, that will be summed up somewhere.
  evdw = evdwt
  !========================================================================
  RETURN
END SUBROUTINE EVDWQM2
!
SUBROUTINE EVDWQM3 (EVDW,X,Y,Z,DX,DY,DZ,DXM,DYM,DZM)
  !------------------------------------------------------------------------
  !
  !     Does the work of EVDWQM at Image
  !
  !     Evdw   - the van der Waal's energy.
  !     Qupdat - a logical flag indicating that the group lists should be
  !              updated.
  !
#if KEY_FLUCQ==1
  use flucqm,only:fqcfor          
#endif

  use chm_kinds
  use dimens_fcm
  use memory
  use number
  use inbnd
  use nbndqm_mod
  use enbondg
  use param
  use psf
#if KEY_FLUCQ==1
  use flucq  
#endif
  !
  !     QC: UW_10_04 fix for fast off
  !         so that extended is consistent with fast "on"
  use exelecm
  !
  ! for parallel run
#if KEY_PARALLEL==1
  use parallel  
#endif
#if KEY_LJPME==1
  use energym, only: qeterm, eterm, ljexc
#endif
  implicit none
  !
  real(chm_real)  etemp, evdw,evdwt,x(*),y(*),z(*),dx(*),dy(*),dz(*)
  real(chm_real)  dxm(*),dym(*),dzm(*)
  !
  ! go parallel
  INTEGER ATFRST,ATLAST,I,J
  !
  !========================================================================
  !     Calculate the van der Waal's energies.
  !========================================================================

  if(allocated(xyzg)) then
     if(size(xyzg,1).lt.ngrp) call chmdealloc('qmevdw.src','EVDWQM3','xyzg',size(xyzg,1),3,crl=xyzg)
  end if
  if(allocated(qg_qmmm)) then
     if(size(qg_qmmm).lt.ngrp) call chmdealloc('qmevdw.src','EVDWQM3','qg_qmmm',size(qg_qmmm),crl=qg_qmmm)
  end if
  if(allocated(ioff_qmmm)) then
     if(size(ioff_qmmm).lt.natc) call chmdealloc('qmevdw.src','EVDWQM3','ioff_qmmm',size(ioff_qmmm),intg=ioff_qmmm)
     natc_old=natc
  end if
  if(.not.allocated(xyzg))      call chmalloc('qmevdw.src','EVDWQM3','xyzg',ngrp,3,crl=xyzg)
  if(.not.allocated(qg_qmmm))   call chmalloc('qmevdw.src','EVDWQM3','qg_qmmm',ngrp,crl=qg_qmmm)
  if(.not.allocated(ioff_qmmm)) call chmalloc('qmevdw.src','EVDWQM3','ioff_qmmm',natc,intg=ioff_qmmm)

  !
  evdw = Zero
  evdwt= zero
  !
  dxm(1:natom)=Zero
  dym(1:natom)=Zero
  dzm(1:natom)=Zero
  !
  ! Set the "lshft" flag to false in this call regardless of the specified
  ! nonbond optiopns.  This avoids the use of the hidden "shift" option
  ! in EGROUP for van der Waal interaction (i.e. vdw truncation). - BRB
  !
  ! Ben Webb, 2000, Call changed to #ass LCENTR=-1 (not .true.) as EGROUP
  ! expects an integer, not a logical! -BW
  CALL EGROUP(etemp,evdwt,natom,NST2,jqmgpv, &
       iqmgpv,cg,RSCLF,dxm,dym,dzm,1,ngrp,igpbs, &
       igptyp,jatmex%a,iatmex%a, &
       .false.,.true.,lcons,.false.,lfswt,ldbexp, &
       cnba,cnbb,maxcn,itc,iac,natc,ioff_qmmm,x,y,z, &
       ctonnb,ctofnb,eps,e14fac,wrnmxd,.false., (/ ZERO /), &
#if KEY_NBIPS==1
       LVIPS,LEIPS,                          & 
#endif
       QETEN,QETSR,                                 & 
#if KEY_MC==1
       -1,                                         & 
#endif
       xyzg(1:ngrp,1),xyzg(1:ngrp,2),xyzg(1:ngrp,3),qg_qmmm, &
#if KEY_FLUCQ==1
       QFLUC,FQCFOR,                               & 
#endif
       (/ ZERO /), (/ 0 /), .false. &
#if KEY_WCA==1
       ,.FALSE.,ONE,WCA                            & 
#endif
#if KEY_LJPME==1
       , QETERM(LJEXC), ETERM(LJEXC)  &
#endif
       ,QEXTND)

  !     QC: UW_031205 fix for fast off
  !         so that extended is consistent with fast "on"

  !
  !
  do i=1,natom   ! ATFRST,ATLAST as each node has different non-bond list. (see GUQMV2)
     dx(i)=dx(i)+dxm(i)
     dy(i)=dy(i)+dym(i)
     dz(i)=dz(i)+dzm(i)
  end do
  !
  ! go parallel, each node has different value, that will be summed up somewhere.
  evdw = evdwt
  !========================================================================
  RETURN
END SUBROUTINE EVDWQM3
#else /**/
subroutine noqmevdw
  return
end subroutine noqmevdw
#endif 

