module heurist

  use chm_kinds
  use chm_types
  use number
  implicit none
  private

  real(chm_real) reuris

  ! True if heuristic_check was called
  logical :: heuristic_check_called = .false.

#if KEY_DOMDEC==1
  ! True if last call to UPDECI updated the neighborlist
  logical q_donb_updeci
#endif

  real(chm_real8) timebuffer(256)

  public reuris, heuristic_check, updeci
#if KEY_DOMDEC==1
  public q_donb_updeci
#endif

contains

subroutine heuristic_check(istep, x, y, z, xstep, ystep, zstep, stepflag)
#if KEY_PARALLEL==1
  use parallel  
#endif
  use psf,only:natom
  use contrl,only:inbfrq
  use new_timer,only:timer_start,timer_stop,T_heurcheck 
  use inbnd,only:atsx,atsy,atsz,cutnb,ctofnb
#if KEY_DOMDEC==1
  use domdec_common,only:natoml,atoml,q_domdec  
#endif
#if KEY_DOMDEC_GPU==1
  use domdec_common,only:q_gpu
#endif
  implicit none
  ! Input / Output parameters
  integer istep
  real(chm_real) x(*), y(*), z(*)
  real(chm_real) xstep(*), ystep(*), zstep(*)
  logical stepflag
  ! Variables
  integer i, atfrst, atlast
  real(chm_real) ddumy1, ddumy2, ddumy3, sqrbuf
#ifdef _OPENMP
  real(chm_real) ddumy_max  
#endif
#if KEY_DOMDEC_GPU==1
  integer checkflag
#endif
  integer ia

  ! Heuristic update-testing :
  ! If any atom moves by more than half CUTNB-CTOFNB, then an
  ! nonbond-list update is performed.
  !
  !-----------------------------------------------------------------------------
  ! Note :
  !     This current implementation will work - without modifications -
  ! to decide whether the image/crystal non-bond lists need updating,
  ! provided the periodicity parameters don't change (i.e. constant
  ! Volume).
  !     The algorythm is easily adapted to variable Volume
  ! dynamics/minimizations by using
  !              DUMMY1 = ABS(CUTNB-CTOFNB - L )/TWO
  ! instead of   DUMMY1 = ABS(CUTNB-CTOFNB)/TWO       below, where L is
  ! the maximum amount by which the images/cells have come closer to
  ! each other (i.e. the decrease in the lattice sides a,b,c) since the
  ! last update was done.
  !-----------------------------------------------------------------------------
  ! Comment on Note: I don't know who wrote this, but it's wrong - BRB  11/22/96
  ! Consider the assymetric unit of an alanine crystal with a long cutoff ~20A.
  ! In this case, the test is not stringent enough and interactions can be missed.
  ! Also,in the case where the box length is much larger than the cutoff,
  ! then this test is too conservative leading to too many updates.  I don't
  ! believe that there is a good inexpensive solution for the general case, but
  ! it does warrant further consideration.  If you run with CPT, be careful!
  !-----------------------------------------------------------------------------
  !

  heuristic_check_called = .true.

  reuris = zero

  IF (INBFRQ < 0) THEN
     IF (MOD(ISTEP,-INBFRQ) == 0) THEN
        !
        DDUMY1 = ABS(CUTNB-CTOFNB)/TWO
        SQRBUF = DDUMY1*DDUMY1

#if KEY_DOMDEC==1
        if (q_domdec) then
           atfrst = 1
           atlast = natoml
        else
#endif 
           atfrst = 1
           atlast = natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
           atfrst = 1+IPARPT(MYNOD)
           atlast = IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
        endif
#endif

        if (stepflag) then

#if KEY_DOMDEC==1 /*domdec*/
           if (q_domdec) then
#ifdef _OPENMP
              ddumy_max = zero
!$omp parallel do schedule(static) private(ia, i, ddumy1, ddumy2, ddumy3) &
!$omp&                             reduction(max:ddumy_max)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 DDUMY1 = X(I) + XSTEP(I) - ATSX(I)
                 DDUMY2 = Y(I) + YSTEP(I) - ATSY(I)
                 DDUMY3 = Z(I) + ZSTEP(I) - ATSZ(I)
                 DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                 ddumy_max = max(ddumy_max, ddumy1)
              enddo
!$omp end parallel do
              if (ddumy_max > sqrbuf) reuris = one
#else /**/
              do ia=atfrst,atlast
                 i = atoml(ia)
                 DDUMY1 = X(I) + XSTEP(I) - ATSX(I)
                 DDUMY2 = Y(I) + YSTEP(I) - ATSY(I)
                 DDUMY3 = Z(I) + ZSTEP(I) - ATSZ(I)
                 DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                 IF(DDUMY1 > SQRBUF) THEN
                    REURIS = ONE
                    return
                 ENDIF
              enddo
#endif 
           else
#endif /* (domdec)*/
              do i=atfrst,atlast
#if KEY_SPACDEC==1
                 IF(ICPUMAP(I) == MYNOD) THEN  
#endif
#if KEY_PARASCAL==1
                 IF(JPBLOCK(I) == MYNOD) THEN  
#endif
                    DDUMY1 = X(I) + XSTEP(I) - ATSX(I)
                    DDUMY2 = Y(I) + YSTEP(I) - ATSY(I)
                    DDUMY3 = Z(I) + ZSTEP(I) - ATSZ(I)
                    DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                    IF(DDUMY1 > SQRBUF) THEN
                       REURIS = ONE
                       return
                    ENDIF
#if KEY_PARASCAL==1
                 ENDIF
#endif
#if KEY_SPACDEC==1
                 ENDIF
#endif
              enddo
#if KEY_DOMDEC==1
           endif
#endif
        else ! (stepflag)

#if KEY_DOMDEC==1 /*domdec*/
           if (q_domdec) then
#ifdef _OPENMP
              ddumy_max = zero
!$omp parallel do schedule(static) private(ia, i, ddumy1, ddumy2, ddumy3) &
!$omp&                             reduction(max:ddumy_max)
              do ia=atfrst,atlast
                 i = atoml(ia)
                 DDUMY1 = X(I) - ATSX(I)
                 DDUMY2 = Y(I) - ATSY(I)
                 DDUMY3 = Z(I) - ATSZ(I)
                 DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                 ddumy_max = max(ddumy_max, ddumy1)
              enddo
!$omp end parallel do
              if (ddumy_max > sqrbuf) reuris = one
#else /**/
              do ia=atfrst,atlast
                 i = atoml(ia)
                 DDUMY1 = X(I) - ATSX(I)
                 DDUMY2 = Y(I) - ATSY(I)
                 DDUMY3 = Z(I) - ATSZ(I)
                 DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                 IF(DDUMY1 > SQRBUF) THEN
                    REURIS = ONE
                    return
                 ENDIF
              enddo
#endif 
           else
#endif /* (domdec)*/
              do i=atfrst,atlast
#if KEY_SPACDEC==1
                 IF(ICPUMAP(I) == MYNOD) THEN  
#endif
#if KEY_PARASCAL==1
                 IF(JPBLOCK(I) == MYNOD) THEN  
#endif
                    DDUMY1 = X(I) - ATSX(I)
                    DDUMY2 = Y(I) - ATSY(I)
                    DDUMY3 = Z(I) - ATSZ(I)
                    DDUMY1 = DDUMY1*DDUMY1 + DDUMY2*DDUMY2 + DDUMY3*DDUMY3
                    IF(DDUMY1 > SQRBUF) THEN
                       REURIS = ONE
                       return
                    ENDIF
#if KEY_PARASCAL==1
                 ENDIF
#endif
#if KEY_SPACDEC==1
                 ENDIF
#endif
              ENDDO
#if KEY_DOMDEC==1
           endif
#endif
        endif ! stepflag
     ENDIF
  ENDIF

  return
end subroutine heuristic_check

SUBROUTINE UPDECI(ISTEP,X,Y,Z,WMAIN,LDYNAM,XOLD,YOLD,ZOLD,VX,VY,VZ)
  !-----------------------------------------------------------------------
  ! By Stefan Fischer.
  !
  ! Makes the decision whether to update the various non-bond lists.
  !
  ! UPDECI() is controled through INBFRQ (in CONTRL.FCM)
  !  and ISTEP as follows :
  !
  ! If INBFRQ = +n --> non-bond list is performed when MOD(ISTEP,n) == 0
  !                    Image and H-bond lists are updated according to
  !                    IMGFRQ and IHBFRQ.
  ! If INBFRQ =  0 --> non-bond list update is not performed. Image and
  !                    H-bond lists are updated according to IMGFRQ
  !                    and IHBFRQ.
  ! If INBFRQ = -n --> non-bond list is updated when necessary (heuristic
  !                    test) if n=-1. If n < -1, then update-testing is done
  !                    every n steps (not recommended !). Heuristic will
  !                    be used for Image and H-bond list-updates, but only
  !                    if IMGFRQ and IHBFRQ are also = -1 .
  !
  use new_timer,only:timer_start,timer_stop,timer_stpstrt,&  
       T_images,T_list,T_crdcomm,T_heurcheck,T_nbonds
  
  use dimens_fcm
  !
  use bases_fcm
  use contrl
  use energym
  use hbondm
  use fast
  use image
  use imgup
  use inbnd
  use param
  use psf
  use stream
  use exelecm
  use tsms_mod
  use ffieldm
#if KEY_PARALLEL==1
  use parallel      
#endif
#if KEY_ZEROM==1
  use zdata_mod, only: QZMOD
#endif
#if KEY_SCCDFTB==1
  use sccdftb       
#endif
#if KEY_SCCDFTB==1
  use sccdftbsrc     /*qc_010110*/
#endif
  use memory
#if KEY_PHMD==1
  use phmd, only: qphimg, qphnbd  
#endif
  use machutil,only:eclock
#if KEY_DOMDEC==1
  use domdec_d2d_comm,only:transfer_coord, update_groupl
  use domdec_common,only:q_domdec, q_gpu, set_box
  use domdec_grouped,only:build_groupedtbl
  use domdec_shake,only:build_shaketbl
  use domdec_dlb,only:q_load_balance, load_balance
  use domdec_local,only:update_local_home_coord, update_local_import_coord, build_local_coord, &
       build_local_vdwtype
  use enbxfast,only:enbcalc_xfast_home
#endif 
#if KEY_DOMDEC_GPU==1
  use domdec_util_gpu_mod,only:range_start, range_stop
#endif
  use api_func, only: mlpot_update, mlpot_is_set   ! Kai Toepfer Jan. 2024

  implicit none
#if KEY_PARALLEL==1
  real(chm_real) TIMMER
#endif 
  !
  ! Passed variables
  INTEGER ISTEP
  real(chm_real) X(*),Y(*),Z(*),WMAIN(*)
  INTEGER LDYNAM
  real(chm_real) XOLD(*),YOLD(*),ZOLD(*),VX(*),VY(*),VZ(*)
  !
  ! Local variables :
  integer,allocatable,dimension(:) :: ISLCT
  real(chm_real),allocatable,dimension(:,:,:) :: transf
  INTEGER  I
  ! Update-decision variables :
  LOGICAL  QEURIS,QDONB,QDOHB,QDOIM,QDOXT
  !
  ! Don't update anything if just starting (avoid duplicate updates).
#if KEY_DOMDEC==1
  if (.not.q_domdec .and. istep <= 0) return  
#endif
#if KEY_DOMDEC==0
  IF(ISTEP <= 0) RETURN        
#endif
  !
#if KEY_TSM==1
  !     Make sure all piggyback atoms are correctly set.
  IF(QTSM.AND.PIGSET) CALL PIGCVSET(X,Y,Z)
#endif 
  !
#if KEY_PARALLEL==1
!  CALL PSYNC()
  TIMMER = ECLOCK()
#endif 
  !
#if KEY_ZEROM==1
  if(QZMOD) heuristic_check_called=.true.
#endif

  if (.not.heuristic_check_called) then
     reuris = zero
     call heuristic_check(istep, x, y, z, (/zero/), (/zero/), (/zero/), .false.)
#if KEY_PARALLEL==1
     call timer_start(T_heurcheck)                  
     call gcomb(reuris,1)
     call timer_stop(T_heurcheck)                  
#endif 
  else
     ! NOTE (APH): If heuristic_check was called, then gcomb(reuris) was also done outside this
     !             subroutine. Therefore, no call to gcomb necessary here.
  endif

  ! Reset the flag
  heuristic_check_called = .false.

  qeuris = reuris > zero

  !
  ! Process an CRYSTAL and IMAGE update if requested.
  !
  QDOIM =.FALSE.
#if KEY_PHMD==1
  QPHIMG = QDOIM  
#endif
  QDOXT =.FALSE.
     IF(NTRANS > 0) THEN
        IF(IMGFRQ > 0) QDOIM = (MOD(ISTEP,IMGFRQ) == 0)
        IF(IXTFRQ > 0 .AND. XTLTYP /= '    ') THEN
           QDOXT = (MOD(ISTEP,IXTFRQ) == 0)
        ENDIF
        IF(IMGFRQ < 0) QDOIM = QEURIS
        IF(QDOXT) THEN
           IF(PRNLEV >= 4) WRITE(OUTU,100) 'Crystal',ISTEP
           call chmalloc('heurist.src','UPDECI','ISLCT',natom,intg=ISLCT)
           call chmalloc('heurist.src','UPDECI','transf',3,4,xnsymm,crl=transf)
           islct(1:natom)=1
           CALL XBUILD2(NATOM,X,Y,Z,ISLCT,TRANSF)
           call chmdealloc('heurist.src','UPDECI','ISLCT',Natom,intg=ISLCT)
           call chmdealloc('heurist.src','UPDECI','transf',3,4,xnsymm,crl=transf)
           ! Always do an image update after a crystal update. - BRB
           QDOIM = .TRUE.
#if KEY_PHMD==1
           QPHIMG = QDOIM  
#endif
        ENDIF
        IF(QDOIM) THEN
           IF(PRNLEV >= 4) WRITE(OUTU,100) 'Image',ISTEP
           call timer_start( T_images )                  
           CALL UPIMAG(X,Y,Z,WMAIN,LDYNAM, &
                XOLD,YOLD,ZOLD,VX,VY,VZ)
           call timer_stop( T_images )                  
           QEURIS=.TRUE.
        ENDIF
     ENDIF
  !
  ! Update the hydrogen-bond list :
  !
  QDOHB=.FALSE.
  IF(IHBFRQ > 0) QDOHB = (MOD(ISTEP,IHBFRQ) == 0) .OR. QDOIM
  IF(IHBFRQ < 0) QDOHB = QEURIS
  IF(QDOHB) THEN
     CALL HBONDS(OUTU,IDON,IHD1,NDON,IACC,IAC1,NACC, &
          IHB,JHB,KHB,LHB,ICH,NHB,MAXHB,CHBA,CHBB,HBEXPN, &
          CTONHB,CTOFHB,CTONHA,CTOFHA,CUTHB,CUTHBA,LHBFG, &
          NCH,KCH,IAC,ATC,NATC,IMOVE,BEST,HBEXCL, &
          BNBND%IBLO14,BNBND%INB14,NNB14, &
          X,Y,Z,NATOM)
     IF(NTRANS > 0.AND.NATIM.GT.0) CALL IMHBON(BIMAG,X,Y,Z)
  ENDIF
  !
  ! Process nonbond update.
  !
  QDONB=.FALSE.
  IF(INBFRQ > 0) QDONB = (MOD(ISTEP,INBFRQ) == 0) .OR. QDOIM
  IF(INBFRQ < 0) QDONB = QEURIS
 
#if KEY_PHMD==1
  QPHNBD = QDONB 
#endif

#if KEY_DOMDEC==1
  q_donb_updeci = qdonb 
#endif

  ! *** In sync ***

  IF(QDONB) THEN
     !
     !     This is now outside the dynamc loop as CALL SPACBR()!!!
     !         write(*,*)'UPDECI>me,call to spacsr',mynod
#if KEY_SPACDEC==1
     !C         CALL SPACSR(XOLD,YOLD,ZOLD)                      
#endif
     !     This is a good place to do XOLD. Here we can decide on either
     !     SPACSR() - cheap or SPACBR() - costly
#if KEY_PARALLEL==1
#if KEY_SPACDEC==1
     TMERI(TIMNBOND) = TMERI(TIMNBOND) + ECLOCK() - TIMMER
     TIMMER=ECLOCK()
     CALL SPACBR(xold,natom,icpumap)
     CALL SPACBR(yold,natom,icpumap)
     CALL SPACBR(zold,natom,icpumap)
     TMERI(TIMGCOMM)=TMERI(TIMGCOMM)+ECLOCK()-TIMMER
     TIMMER=ECLOCK()
#endif 
#endif 
     !
     ! JG 5/02
     IF (RCENT) THEN
        CALL CRDMOV(NATOM,X,Y,Z,XOLD,YOLD,ZOLD, &
             NRES,IBASE)
     ENDIF
     !
#if KEY_DOMDEC==1
     if (q_domdec) then
        call timer_stpstrt(T_list,T_crdcomm)                     
        ! Do dynamic load balancing
        if (q_load_balance) call load_balance()

        ! Update group home boxes
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('update_groupl')  
#endif
        if (ldynam .eq. 0) then
          call update_groupl(x, y, z)
        else
          call update_groupl(x, y, z, xold, yold, zold)
        end if
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()  
#endif
        ! Communicate coordinates
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('transfer_coord')  
#endif        
        call transfer_coord(x, y, z, .true.)
        call build_local_coord(x, y, z, cg &
#if KEY_LJPME==1
        , ljc6 &
#endif
        )
        call build_local_vdwtype()
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()  
#endif

        call timer_stpstrt(T_crdcomm,T_list)                     
     endif
#endif 
     
     IF(PRNLEV >= 4) WRITE(OUTU,100) 'Nonbond',ISTEP
     call timer_start(T_nbonds)                  
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_start('NBONDS')  
#endif
     CALL NBONDS(X,Y,Z,BNBND,BIMAG)
#if KEY_DOMDEC_GPU==1
     if (q_gpu) call range_stop()  
#endif
     call timer_stop(T_nbonds)                  

#if KEY_DOMDEC==1
     if (q_domdec) then
        ! Build bonded lists
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('build_groupedtbl / build_shaketbl')  
#endif

        call build_groupedtbl()
        call build_shaketbl()
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()  
#endif

     endif
#endif

     ! QC:UW_031205
#if KEY_PARALLEL==1
     IF(MYNOD == 0) THEN
#endif 
#if KEY_SCCDFTB==1
        if(qsccnb) call mkmmlst         
#endif
#if KEY_PARALLEL==1
     ENDIF
#endif 
     ! QC:UW_031205_END
     TOTUPD = TOTUPD + 1
  ELSE
#if KEY_DOMDEC==1
     if (q_domdec) then
        call timer_stpstrt(T_list,T_crdcomm)                     

#if KEY_DOMDEC_GPU==1
        if (q_gpu) then
           call update_local_home_coord(x, y, z)
           !! Box must be set correctly for NPT
           call set_box()
           call enbcalc_xfast_home()
        else
#endif
           call update_local_home_coord(x, y, z)
#if KEY_DOMDEC_GPU==1
        endif
#endif

        ! APH: Do not rebuild non-bond list, just update coordinates
        ! NOTE: transfer_coord only changes the import coordinates
        !       home zone coordinates are already up-to-date
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_start('transfer_coord (updeci)')
#endif
        call transfer_coord(x, y, z, .false.)
#if KEY_DOMDEC_GPU==1
        if (q_gpu) call range_stop()
#endif

        ! Update import coordinates
        call update_local_import_coord(x, y, z)

        call timer_stpstrt(T_crdcomm,T_list)                     
     endif
#endif 
  ENDIF

  !
  ! Update ML-ML and ML-MM atom pair lists for MLpot module
  ! Kai Toepfer, Jan. 2024
  !
  if(mlpot_is_set().and.(QDONB.or.QDOIM.or.QDOXT)) then                                               
    call mlpot_update(                        &                                                       
      natom, ntrans, natim,                   &
      bnbnd%jnb, bnbnd%inblo,                 &  
      bimag%imattr, bimag%imjnb, bimag%imblo)    
    if(prnlev >= 4) write(outu,100) 'MLpot',istep
  endif
  ! Out of sync

100 FORMAT(' UPDECI: ',A,' update at step',I10)
  !
#if KEY_PARALLEL==1
  TMERI(TIMNBOND) = TMERI(TIMNBOND) + ECLOCK() - TIMMER     
#endif
  !
  RETURN
END SUBROUTINE UPDECI

end module heurist

