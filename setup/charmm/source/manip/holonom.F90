module holonom
  use chm_kinds
  use dimens_fcm
  implicit none

  !  This module deals with all holonomic (i.e. rigid) constraints
  !  in CHARMM.  A central routine is important since an atom may be
  !  affected by more than one contraint types and both need to be solved
  !  in a consistent (i.e. iterative) manner.
  !
  !  The following holonomic constraints are considered in this module
  !
  !     SHAKE     (distance constraints)
  !           manip/shake.src       fcm/shake.f90
  !     LONEPAIR  (general massless particle contraints)
  !           dynamc/lonepair.src   fcm/lonepr.f90  ##LONEPAIR
  !     FIX       (original CONS FIX feature)
  !           energy/energy.src...  fcm/psf.f90
  !     ST2       (old ST2 5-site water model with lonepairs)
  !           energy/enst2.src      fcm/psf.f90     ##.not.NOST2
  !     ICFIX     (Internal coordinate constrains..angles,dihedrals..)
  !           pert/icfix.src   fcm/icfix.f90        ##TSM
  !     BODY      (Rigid body shapes)
  !           shapes/shapedyn.src   fcm/shapes.f90  ##SHAPES
  !     SHAKA4    (4th dimension constraints)
  !                                 fcm/fourd.f90   ##FOURD
  !     PATH      (path constraints -- hyperplane)
  !
  !  The following holonomic constraints are NOT considered in this module
  !  due to implementation difficulties and/or scattered development
  !
  !     TMD       (targeted molecular dynamics)
  !           dynamc/tmd.src...     fcm/tmd.f90   ##TMD
  !
  !                          B.R. Brooks - NIH - September 15, 2003
  !
  !-----------------------------------------------------------------------

  private

  ! APH: These variables are allocated but NOT deallocated. 
  !      We should introduce Init/Uninit routines for this module!
  integer,allocatable,dimension(:) :: ISKP
  real(chm_real),allocatable,target,dimension(:) :: xyz_space

  public holonoma, holonomf

contains

  SUBROUTINE HOLONOMA(X,Y,Z,XREF,YREF,ZREF,LMASS,LDYNA,QOK)
    !-----------------------------------------------------------------------
    !  This routine is the control routine for holonomic constraints
    !  when applied to coordinates.
    !
    !  X,Y,Z          <-> coordinates to modify
    !  XREF,YREF,ZREF <-  reference coordinates
    !  LMASS          <-  Flag for mass weighting
    !  LDYNA          <-  Flag for dynamics
    !  QOK             -> Return flag for convergence
    !
    !     Determine which version of SHAKE to use.
    !     Also handle lone pairs, internal coordinate constraints,
    !     and ST2 water   - B. Brooks - NIH - 11/8/97 and 7/15/03
    !

    use new_timer,only:timer_start,timer_stop,T_shake    
    use number
    use exfunc
    use psf
    use bases_fcm
    use shake
    use stream
    use fourdm
    use icfix
    use parallel
    use lonepr
    use pert
    use pshake
    use shapes
#if KEY_RXNCONS==1
    use rxncons  
#endif
#if KEY_RXNCONS==1
    use rxcons1  
#endif
#if KEY_RXNCONS==1
    use rxcons2  
#endif
#if KEY_RXNCONS==1
    use rxcons3  
#endif
#if KEY_RXNCONS==1
    use rxcons4  
#endif
    !   use rxncons2,only:rxncons_2
#if KEY_FSSHK==1
    use fstshk,only: fstshake   
#endif
    use memory
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_d2d_comm,only:transfer_coord
#endif
    implicit none
    !
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*)
    LOGICAL LMASS,LDYNA,QOK
    !
    !
    INTEGER NITER,NIT
    LOGICAL QFSHK
    INTEGER ATFRST, ATLAST, I
    !-----------------------------------------------------------------------
    !
    QOK=.TRUE.
    if (.not.allocated(iskp)) then
       call chmalloc('holonom.src','HOLONOMA','ISKP',NATOM,intg=ISKP)
    elseif (size(iskp) < natom) then
       call chmdealloc('holonom.src','HOLONOMA','ISKP',size(iskp),intg=ISKP)
       call chmalloc('holonom.src','HOLONOMA','ISKP',NATOM,intg=ISKP)
    endif
    !
    !-----------------------------------------------------------------------
    ! First do only non-zero mass contraints.
    !
    call timer_start(T_shake)
    ! Should we allow any fast SHAKE routines?
    QFSHK=QFSHAKE
#if KEY_PARALLEL==1
#if KEY_FSSHK==0
    IF(NUMNOD > 1) QFSHK=.FALSE.  ! use generic if parallel
#endif 
#endif 
    !
#if KEY_PERT==1
    !sb   Make sure that constraint list is up to date with respect to
    !     lambda!!
    IF (QPSHAKE) THEN
       CALL PSHKSET(PERTCONST,PERTSHAUX)
       QFSHK=.FALSE.
    ENDIF
#endif 
    !SBp
    !
    NITER=0
    !
    IF(NCONST == 0) THEN
       !       no shake constraints, but still process ICs and lonepairs.
#if KEY_TSM==1
       IF(NICF > 0) CALL ICFCNS(XREF,YREF,ZREF,X,Y,Z,AMASS,NATOM) 
#endif
       !
#if KEY_FSSHK==1 /*qfshake_call*/
    ELSE IF(QFSHK.AND.LMASS.and.ldyna) THEN
34     CONTINUE
       NITER = 0
       CALL fstSHaKe(NCONST_pll,NITER,X,Y,Z,XREF,YREF,ZREF,NATOM &
            ,shkapr,nconst_pll,idgf2,constr,shktol,mxiter,amass &
            )
       IF(PRNLEV > 6) WRITE(OUTU,125) 'FSTSHAKE'
       IF(PRNLEV > 6) WRITE(OUTU,*)  &
            '         contraints: ',nconst
#if KEY_TSM==1
       IF(NICF > 0) THEN
          ANYADJ = .FALSE.
          CALL ICFCNS(XREF,YREF,ZREF,X,Y,Z,AMASS,NATOM)
          IF(ANYADJ) GOTO 34
       ENDIF
#endif 
#endif /* (qfshake_call)*/
       !
    ELSE
       ! Use the generic SHAKE version
       !
       CALL SHAKEA2(X,Y,Z,XREF,YREF,ZREF,AMASS,LMASS,LDYNA,NATOM, &
            NITER,IMOVE,ISKP, &
#if KEY_PERT==1
            PERTCONST,PERTSHAUX & 
#endif
#if KEY_PERT==0
            0,0 &                 
#endif
            )
       !
       IF(PRNLEV > 6) WRITE(OUTU,125) 'SHAKEA2'
       IF(PRNLEV > 6) WRITE(OUTU,*)  &
            '         contraints: ',nconst
    ENDIF
    !
#if KEY_SHAPES==1
    !     do rigid body transformations
    IF(NUMSHP > 0) THEN
       CALL SHAKBODY(X,Y,Z,XREF,YREF,ZREF,AMASS,LMASS,LDYNA,NATOM, &
            IMOVE,SHPATP,NUMSHP,SHPTYP)
    ENDIF
#endif 
    !
#if KEY_FOURD==1
    IF(DIM4.AND.QSHAK4) CALL SHAKA4(FDIM,AMASS,NATOM,IMOVE4)
#endif 

    !
    ! Finished with non-zero mass contraints.
    !-----------------------------------------------------------------------
    ! Now do constraints for massless particles.
    !
#if KEY_LONEPAIR==1
    ! position lonepairs based on new positions
    !
    !     NOTE on parallel:
    !     Parallelize LONEPRC only for the dynamics (LDYNA)
    !     There maybe a code which doesn't want parallel LONEPRC
    !     In this case we could replace the code below
    !     with ATFRST=1;ATLAST=NATOM
    !     VV2 integrator calls LONEPRC directly (is this OK?) with
    !     ATFRST, ATLAST adjusted for parallel!
    !
    !
    IF(LDYNA)THEN
       !
#if KEY_DOMDEC==1
       if (q_domdec) then
          atfrst = 1
          atlast = natom
       else
#endif
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
          ATFRST=1+IPARPT(MYNOD)
          ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
          ATFRST=1
          ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
          ATFRST=1
          ATLAST=NATOM
#endif /* (paramain)*/
#if KEY_DOMDEC==1
       endif
#endif
       !
    ELSE
       ATFRST=1
       ATLAST=NATOM
    ENDIF
    !
    IF(NUMLP > 0) THEN
       CALL LONEPRC(ATFRST,ATLAST,X,Y,Z,XREF,YREF,ZREF,AMASS, &
            NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
#if KEY_DOMDEC==1
       if (q_domdec .and. .not.ldyna) then
          ! We need to distribute lone pairs to import volume
          call transfer_coord(x, y, z, .false.)
       endif
#endif
    ENDIF
#endif 
#if KEY_NOST2==0
    !     now position ST2 lonepairs (if present)
    NIT=0
    CALL SHKST2(X,Y,Z,XREF,YREF,ZREF,LMASS,NIT)
    IF(NIT > NITER) NITER=NIT
#endif 
    !
#if KEY_RXNCONS==1 /*jc070701-1*/
    ! . Reaction coordinate constraint  ! jwchu
    !
    IF(LRXCNS) THEN
       IF(IRXCNS == 1)THEN
          CALL RXNCONS_1(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
               NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MAXITR, &
               PLAGRANM,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               XREFC,YREFC,ZREFC, &
               DTSQ,ZFAC,GFAC)
       ELSEIF(IRXCNS == 2)THEN
          CALL RXNCONS2(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
               NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MAXITR, &
               PLAGRANM,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               XREFC,YREFC,ZREFC, &
               DTSQ,ZFAC,GFAC)
       ELSEIF(IRXCNS == 3)THEN
          CALL RXNCONS_3(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
               NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,FRCONS, &
               MAXITR,PLAGRANM,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               XREFC,YREFC,ZREFC, &
               DTSQ,ZFAC,GFAC,RCNSDL,RCNSDS,RCNSDSM,&
               RCNSDC,RCNSDCM,RCNSDR,RCNSDRC,RCNSAMF, &
               IMOVE,NUMLP, &
               LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)
       ELSEIF(IRXCNS == 4)THEN
          CALL RXNCONS_4(X,Y,Z,XREF,YREF,ZREF,AMASS,NATOM,IRXCNS, &
               NRXATM,LRXCNS,LDYNA,VRXNCS,LAGRANM,MAXITR, &
               PLAGRANM,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               XREFC,YREFC,ZREFC, &
               DTSQ,ZFAC,GFAC,NRXNCS,IMOVE, &
               NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT, &
               LPRID)
       ENDIF
    ENDIF
    !
#endif /* (jc070701-1)*/
    call timer_stop(T_shake)
125 FORMAT(' HOLONOMA: Using routine ',A,' for shake calculation.')
    !

    RETURN
  END SUBROUTINE HOLONOMA


  SUBROUTINE HOLONOMF(DX,DY,DZ,XREF,YREF,ZREF,LMASS,QZEROM,QOK)
    !
    !  This routine is the control routine for holonomic constraints
    !  when applied to forces.
    !
    !   DX,DY,DZ       <-> Forces to modify
    !   XREF,YREF,ZREF <-  Reference coordinates
    !   LMASS          <-  Mass weight flag
    !   QZEROM         <-  Flag to only do zero mass particles
    !   QOK             -> Return flag for convergence
    !
    !      BRB - 3/6/83 and 7/15/03
    !
    use exfunc
    use psf
    use bases_fcm
    use shake
    use energym
    use fourdm
    use stream
    use pert
    use pshake
    use icfix
    use lonepr
    use shapes
#if KEY_RXNCONS==1
    use rxncons  
#endif
#if KEY_RXNCONS==1
    use rxcons1  
#endif
#if KEY_RXNCONS==1
    use rxcons2  
#endif
#if KEY_RXNCONS==1
    use rxcons3  
#endif
#if KEY_RXNCONS==1
    use rxcons4  
#endif
    !   use rxncons2,only:rxncons_2f
    use parallel
    use memory
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,natoml,atoml  
#endif
    use prssre
    implicit none
    !
    real(chm_real),pointer,dimension(:) :: xIX,yIY,zIZ

    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) XREF(*),YREF(*),ZREF(*)
    LOGICAL LMASS      ! use mass weighting
    LOGICAL QZEROM     ! only process zero mass atoms
    LOGICAL QOK        ! returned flag, .true. if converged
    !
    !
    real(chm_real) AMSI,AMSJ,XIJ,YIJ,ZIJ,XPIJ,YPIJ,ZPIJ,DIFF,RRPR,ACOR
    INTEGER I,J,K,LL

    INTEGER IX,IY,IZ
    INTEGER NITER
    real(chm_real) TOLGRD
    LOGICAL READY
    INTEGER ATFRST, ATLAST
    logical workdone
    !
    ! Save the current forces (for virial correction)

    workdone = .false.

    if (.not.allocated(iskp)) then
       call chmalloc('holonom.src','HOLONOMF','ISKP',NATOM,intg=ISKP)
    elseif (size(iskp) < natom) then
       call chmdealloc('holonom.src','HOLONOMF','ISKP',size(iskp),intg=ISKP)
       call chmalloc('holonom.src','HOLONOMF','ISKP',NATOM,intg=ISKP)
    endif

    if (.not.allocated(xyz_space)) then
       call chmalloc('holonom.src','HOLONOMF','xyz_space',3*NATOM,crl=xyz_space)
    elseif (size(xyz_space) < 3*natom) then
       call chmdealloc('holonom.src','HOLONOMF','xyz_space',size(xyz_space),crl=xyz_space)
       call chmalloc('holonom.src','HOLONOMF','xyz_space',3*NATOM,crl=xyz_space)
    endif

    xix=>xyz_space(        1:  natom)
    yiy=>xyz_space(  natom+1:2*natom)
    ziz=>xyz_space(2*natom+1:3*natom)

#if KEY_DOMDEC==1
    if (q_domdec) then
       ! For DOMDEC, only atoms in the homebox are affected
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = atoml(i)
          xix(j) = dx(j)
          yiy(j) = dy(j)
          ziz(j) = dz(j)
       enddo
!$omp end parallel do
    else
#endif 
       xix(1:natom) = dx(1:natom)
       yiy(1:natom) = dy(1:natom)
       ziz(1:natom) = dz(1:natom)
#if KEY_DOMDEC==1
    endif  
#endif
    !
    !-----------------------------------------------------------------------
    ! First do all of the zero mass particles
    !
    ! Note: Even if the mass weight flag is not set, these atoms
    ! still get zero weight.
    !
#if KEY_LONEPAIR==1
    !
#if KEY_DOMDEC==1
    if (q_domdec) then
       atfrst = 1
       atlast = natom
    else
#endif
#if KEY_PARALLEL==1 /*paramain*/
#if KEY_PARAFULL==1 /*parfmain*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Illegal parallel compile options'
#endif /* (parstest)*/
       ATFRST=1+IPARPT(MYNOD)
       ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /*parfmain*/
       ATFRST=1
       ATLAST=NATOM
#else /* (parfmain)*/
#error  'Illegal parallel compile options'
#endif /* (parfmain)*/
#else /* (paramain)*/
       ATFRST=1
       ATLAST=NATOM
#endif /* (paramain)*/
#if KEY_DOMDEC==1
    endif
#endif
    !
    ! Remove forces from lonepairs (if any)
    IF(NUMLP > 0) THEN
       workdone = .true.
       CALL LONEPRF(ATFRST,ATLAST,XREF,YREF,ZREF,AMASS,DX,DY,DZ, &
            NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
    ENDIF
#endif 
#if KEY_NOST2==0
    !     Now remove ST2 forces if present
    if (nst2 > 0) then
       workdone = .true.
       CALL FXFST2(XREF,YREF,ZREF,DX,DY,DZ,LMASS)
    endif
#endif 
    IF(QZEROM) GOTO 800  ! exit if we're only doing zero mass atoms
    !
    !-----------------------------------------------------------------------
    ! Now do the rest (non-zero mass atoms)...
    !
#if KEY_PERT==1
    !sb   Make sure that constraint list is up to date with respect to
    !     lambda!!
    IF (QPSHAKE) THEN
       CALL PSHKSET(PERTCONST,PERTSHAUX)
    ENDIF
#endif 
    !
#if KEY_SHAPES==1 /*shapef*/
    !     Remove deformation forces for rigid bodies
    IF(NUMSHP > 0) THEN
       workdone = .true.
       CALL SHAKBODYF(DX,DY,DZ,XREF,YREF,ZREF,AMASS,LMASS,NATOM, &
            IMOVE,SHPATP,NUMSHP,SHPTYP)
    ENDIF
#endif /*     (shapef)*/
    !
#if KEY_RXNCONS==1 /*jc070701-2*/
    ! . Reaction coordinate constraint  ! jwchu
    !
    IF(LRXCNS) THEN
       IF(IRXCNS == 1)THEN
          workdone = .true.
          CALL RXNCONS_1F(XREF,YREF,ZREF,DX,DY,DZ, &
               NATOM,IRXCNS,NRXATM,LRXPRN,IRXATM)
       ELSEIF(IRXCNS == 2)THEN
          workdone = .true.
          CALL RXNCONS_2F(XREF,YREF,ZREF,DX,DY,DZ, &
               NATOM,IRXCNS,NRXATM,LRXPRN,IRXATM)
       ELSEIF(IRXCNS == 3)THEN
          workdone = .true.
          CALL RXNCONS_3F(XREF,YREF,ZREF,DX,DY,DZ, &
               NATOM,IRXCNS,NRXATM,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               IMOVE,AMASS,NUMLP, &
               LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT,LPRID)
       ELSEIF(IRXCNS == 4)THEN
          workdone = .true.
          CALL RXNCONS_4F(XREF,YREF,ZREF,DX,DY,DZ,NATOM,IRXCNS, &
               NRXATM,NRXNCS,LRXPRN,IRXATM, &
               XCORD,YCORD,ZCORD, &
               XREFC,YREFC,ZREFC, &
               IMOVE,NUMLP,LPNHOST,LPHPTR,LPHOST, &
               LPVALUE,LPWGHT,AMASS,LPRID)

       ENDIF
    ENDIF
    !
#endif /* (jc070701-2)*/
    ! added lines to call routine removing forces from ICFIXed coordinates
    ! K.Kuczera 14-Mar-97
    IF(NCONST == 0) THEN
#if KEY_TSM==1
       IF(NICF > 0) then
          workdone = .true.
          CALL ICFCNF(DX,DY,DZ,XREF,YREF,ZREF,AMASS)
       endif
#endif 
       GOTO 800
    ENDIF
    !
    workdone = .true.
    CALL SHAKEF(DX,DY,DZ,XREF,YREF,ZREF,AMASS,LMASS,NATOM, &
         IMOVE,NITER,ISKP)
    !
#if KEY_TSM==1
    !  Remove forces along fixed internal coordinates K.Kuczera 14-Mar-97
    ANYADJ=.FALSE.
    IF (NICF > 0) then
       workdone = .true.
       CALL ICFCNF(DX,DY,DZ,XREF,YREF,ZREF,AMASS)
    endif
    READY = .NOT. ANYADJ
#endif 
#if KEY_FOURD==1
    IF(DIM4.AND.QSHAK4) CALL SHAKF4(DFDIM,AMASS,NATOM,IMOVE4)
#endif 
    !
800 CONTINUE
    !-----------------------------------------------------------------------
    ! Update the internal virial..
    !
#if KEY_DOMDEC==1
    if (q_domdec) then
       ! If no work is done, just return, no need for virshk
       if (.not.workdone) return
!$omp parallel do schedule(static) private(i, j)
       do i=1,natoml
          j = atoml(i)
          xix(j) = dx(j) - xix(j)
          yiy(j) = dy(j) - yiy(j)
          ziz(j) = dz(j) - ziz(j)
       enddo
!$omp end parallel do
    else
#endif 
       xix(1:natom) = dx(1:natom) - xix(1:natom)
       yiy(1:natom) = dy(1:natom) - yiy(1:natom)
       ziz(1:natom) = dz(1:natom) - ziz(1:natom)
#if KEY_DOMDEC==1
    endif  
#endif
    CALL VIRSHK(EPRESS(VIXX:VIZZ),NATOM,XREF,YREF,ZREF, &
         xIX,yIY,zIZ)
    !

    RETURN
  END SUBROUTINE HOLONOMF
end module holonom

