module rmsdyn_mod
  use chm_kinds
  use dimens_fcm
  implicit none

  real(chm_real),allocatable,dimension(:,:),save :: ID
  integer :: siz_id

contains

#if KEY_NOMISC==0
  subroutine rmsdyn(cname,x,y,z,wmain,xcomp,ycomp,zcomp,wcomp,natom,amass)
    !
    !     MODIFIED CORMAN ROUTINE TO DO RMS FITS BETWEEN ALL NON-EQUAL PAIRS
    !     OF DYNAMICS CORDINATE FILES
    !
    !     CORMAN by Bernard R. Brooks  (developed from 1981 to 1983)
    !     Modified by William D. Laidig
    !     Modified by Lennart Nilsson, June 2002, to include approximate
    !     projection onto 2D plane. A Sammon projection: Sammon JW: A Nonlinear Mapping
    !     for Data Structure Analysis. IEEE Trans Comput 1969, C 18(5):401-409. See also 
    !     M.Levitt, JMB 168, 621 (1983)
    !     Allow number of frames in the two trajectories to be different
    !     L Nilsson March 2015: Added option to use interatomic distance RMSD as metric (DRMS)
    !     L Nilsson Nov 2019: Added option to use interatomic average coordinate
    !     difference, w/o superpositiion, as metric (DIFF)
    !
    use number
    use select
    use comand
    use ctitla
    use cvio,only:trjspc 
    use string
    use stream
    use memory
    use parallel,only:mynod

    character(len=*) cname
    character(len=4) wrd,hdrr
    integer natom
    real(chm_real) x(natom),y(natom),z(natom),wmain(natom)
    real(chm_real) xcomp(natom),ycomp(natom),zcomp(natom),wcomp(natom)
    real(chm_real) rmsr, f
    real(chm_real) amass(natom)

    real(chm_real4),allocatable,dimension(:) ::   temp
    integer,allocatable,dimension(:) :: freeat
    integer :: nfreat,firstu,secndu,icntrl(20), natrec
    integer begin,skip,stop,begin2,stop2,skip2,nunit,nunit2
    integer nslct,nslct2,outrms,pqseed,pqunit
    integer i,j,k,nposit, np, np2
    integer,allocatable,dimension(:) ::  islct,jslct
    integer iopt,nsig,maxfn
    logical lrms,lmass,lnoro,lweig, lmatrix,lsymm,ldrms,ldiff
    !
    integer,allocatable,dimension(:) :: iscr
    real(chm_real),allocatable,dimension(:) :: pq,h,g,w,mass2
    real(chm_real4),allocatable,dimension(:,:,:),target :: x1,x2t
    real(chm_real4),pointer,dimension(:,:,:) :: x2
    integer :: sizx2

    call chmalloc('rmsdyn.src','rmsdyn','temp',natom,cr4=temp)
    call chmalloc('rmsdyn.src','rmsdyn','islct',natom,intg=islct)
    call chmalloc('rmsdyn.src','rmsdyn','freeat',natom,intg=freeat)

    lweig=(indxa(comlyn,comlen,'weig') > 0)
    wrd=nexta4(comlyn,comlen)
    ! for now this is the only operation
    if(wrd == 'ORIE') then
       ! process-orie-command
       LMASS=(INDXA(COMLYN,COMLEN,'MASS') > 0)
       LNORO=(INDXA(COMLYN,COMLEN,'NORO') > 0)
       LRMS=(INDXA(COMLYN,COMLEN,'RMS') > 0)
       LDRMS=(INDXA(COMLYN,COMLEN,'DRMS') > 0)
       LDIFF=(INDXA(COMLYN,COMLEN,'DIFF') > 0)
       i=0
       IF(LRMS) i=i+1
       IF(LDRMS) i=i+1
       IF(LDIFF) i=i+1
       IF(i > 1 ) &
         CALL WRNDIE(-3,'<RMSDYN>','RMS, DRMS and DIFF are mutually exclusive')
       LMATRIX=(INDXA(COMLYN,COMLEN,'MATR') > 0)
       PQUNIT=GTRMI(COMLYN,COMLEN,'PQUN',-1)
       IF(PQUNIT > 0) LMATRIX=.TRUE.
       PQSEED=GTRMI(COMLYN,COMLEN,'PQSE',4711)
       FIRSTU=GTRMI(COMLYN,COMLEN,'IREA',-1)
       ! Default to using just one trajectory
       SECNDU=GTRMI(COMLYN,COMLEN,'JREA',FIRSTU)
       if(firstu ==  -1)then
          ! new syntax then. defaults could be extracted from trajectory.
          call trjspc(comlyn,comlen,nunit,firstu,begin,skip,stop)
          SECNDU=GTRMI(COMLYN,COMLEN,'SECU',FIRSTU)
          NUNIT2=GTRMI(COMLYN,COMLEN,'NUN2',NUNIT)
          BEGIN2=GTRMI(COMLYN,COMLEN,'BEG2',BEGIN)
          SKIP2=GTRMI(COMLYN,COMLEN,'SKP2',SKIP)
          STOP2=GTRMI(COMLYN,COMLEN,'STP2',STOP)
          !
          if(skip <= 0 .or. skip2  <= 0) &
               CALL WRNDIE(-2,'<RMSDYN>','SKIP or SKIP2 NOT POSITIVE')
          np=1 +(stop-begin)/skip
          np2=1 +(stop2-begin2)/skip2
          if(pqunit > 0 .and. np /= np2) &
               call wrndie(-3,'<RMSDYN>', &
               'For PQ-analysis # of frames must be same in both trajectories')
       else
          begin=gtrmi(comlyn,comlen,'BEGI',-1)
          stop=gtrmi(comlyn,comlen,'STOP',-1)
          skip=gtrmi(comlyn,comlen,'SKIP',1)
          nunit=1
          if(skip <= 0) &
               CALL WRNDIE(-2,'<RMSDYN>','SKIP IS NOT POSITIVE')
          np= 1 + (stop-begin)/skip
          np2 = np
          nunit2=nunit
          begin2=begin
          stop2=stop
          skip2=skip
       endif
       lsymm=(firstu == secndu)
       IF(NP <= 0) CALL WRNDIE(-3,'<RMSDYN>','BEGIN  >  STOP ???')
       IF(NP2 <= 0) CALL WRNDIE(-3,'<RMSDYN>','BEG2  >  STP2 ???')
       outrms=gtrmi(comlyn,comlen,'IWRI',-1)
       if(prnlev >= 2)then
          write(outu,220) firstu,secndu,begin,skip,stop
220       FORMAT(' TRAJ: INITIATING READ OF TRAJECTORIES, OPTIONS;'/, &
               '    FIRSTU = ',I3,' SECNDU = ',I3,' BEGIN = ',I10, &
               ' SKIP=',I7,' STOP = ',I10)
          if(lnoro) write(outu,*) 'ORIENTING USING TRANSLATIONS ONLY'
       endif 

       ! Since this command just prints results, don't execute on a slave.
       ! Not strictly true. Some CHARMM variables are set, and subsequent use
       ! of these may cause problems?
!       if(iolev < 0) then
!          call chmdealloc('rmsdyn.src','rmsdyn','TEMP',NATOM,cr4=TEMP)
!          call chmdealloc('rmsdyn.src','rmsdyn','ISLCT',Natom,intg=ISLCT)
!          call chmdealloc('rmsdyn.src','rmsdyn','FREEAT',NATom,intg=FREEAT)
!          return
!       endif

       ! Select-atoms
       call selcta(comlyn,comlen,islct,x,y,z,wmain,.FALSE.)
       nslct=nselct(natom,islct)
       if(nslct == 0) then  
          CALL WRNDIE(0,'<RMSDYN>','ATOM SELECTION ERROR')
          call chmdealloc('rmsdyn.src','rmsdyn','TEMP',NATOM,cr4=TEMP)
          call chmdealloc('rmsdyn.src','rmsdyn','ISLCT',Natom,intg=ISLCT)
          call chmdealloc('rmsdyn.src','rmsdyn','FREEAT',NATom,intg=FREEAT)
          return
       endif
       nslct2=0
       if(ldrms)then
       ! If no second selection present leave nslct2=0 and do not allocate jslct
          if(indx(comlyn,comlen,'SELE',4) > 0 .AND. indx(comlyn,comlen,'END',3) > 0 )THEN
            call chmalloc('rmsdyn.src','rmsdyn','JSLCT',natom,intg=jslct)
            call selcta(comlyn,comlen,jslct,x,y,z,wmain,.FALSE.)
            nslct2=nselct(natom,jslct)
          endif
       endif

       ! Done with the preliminaries

       ! IOPT=0 tells ZXMIN to initalize Hessian to Identity matrix.
       ! IOPT=3 (ZXMIN should initialize Hessian) does not seem to work well.
       ! These options are not documented, and should not need user modification
       iopt=gtrmi(comlyn,comlen,'IOPT',0)
       nsig=gtrmi(comlyn,comlen,'NSIG',3)
       maxfn=gtrmi(comlyn,comlen,'MAXF',0)
       if (maxfn < 1) maxfn=min(50000,200*np)
       call chmalloc('rmsdyn.src','rmsdyn','iscr',natom+natom,intg=iscr)
       siz_id=1
       if(lmatrix)siz_id=np
       call chmalloc('rmsdyn.src','rmsdyn','id_hv',siz_id,siz_id,crl=id)
       if(pqunit > 0)then
          call chmalloc('rmsdyn.src','rmsdyn','pq',2*np,crl=pq)
          call chmalloc('rmsdyn.src','rmsdyn','h',np*(2*np+1),crl=h)
          call chmalloc('rmsdyn.src','rmsdyn','g',2*np,crl=g)
          call chmalloc('rmsdyn.src','rmsdyn','w',6*np,crl=w)
       endif

       call chmalloc('rmsdyn.src','rmsdyn','x1',3,nslct+nslct2,np,cr4=x1)
       x2 => x1
       if(.not. lsymm) then
          call chmalloc('rmsdyn.src','rmsdyn','x2',3,nslct+nslct2,np2,cr4=x2t)
          x2 => x2t
       endif
       call chmalloc('rmsdyn.src','rmsdyn','mass2',nslct,crl=mass2)

       ! Get trajectories into memory

       call rdtrj(natom,islct,np,nslct,jslct,nslct2,x1,firstu,nunit,begin,skip,stop)

       IF(FIRSTU /= SECNDU)THEN
          CALL RDTRJ(NATOM,ISLCT,NP2,NSLCT,jslct,nslct2,X2, &
               SECNDU,NUNIT2,BEGIN2,SKIP2,STOP2)
       ENDIF
       IF(LRMS) CALL CPMSL(NATOM,ISLCT,AMASS,MASS2)
       IF(PRNLEV > 2)THEN
         IF(LDRMS) THEN
           WRITE(OUTU,*) ' Calculating pairwise RMSDs based on interatomic distances'
         ELSE IF (LRMS) THEN
           WRITE(OUTU,*) ' Calculating pairwise RMSDs based on atom positions'
         ELSE IF (LDIFF) THEN
           WRITE(OUTU,*) ' Calculating pairwise RMSDs w/o superpos based on atom positions'
         ELSE
           CALL WRNDIE(-2,'<RMSDYN>','Unknown option')
         ENDIF
       ENDIF
       CALL RMSDY2(X1,X2,NSLCT,NSLCT2,ISLCT,ISCR, &
            OUTRMS,PQUNIT,PQSEED,MASS2,LMASS, &
            LSYMM,LMATRIX,LWEIG,LRMS,LDRMS,LDIFF,LNORO,ID,NP,NP2,F,PQ, &
            H,G,W,IOPT,NSIG,MAXFN)
       !
       call chmdealloc('rmsdyn.src','RMSDYN','ISCR',NATOM+NATOM,intg=ISCR)
       nullify(x2)
       call chmdealloc('rmsdyn.src','RMSDYN','X1',3,nslct+nslct2,np,cr4=X1)
       if(.not. lsymm) call chmdealloc('rmsdyn.src','RMSDYN','X2',3,nslct+nslct2,np2,cr4=X2t)
       call chmdealloc('rmsdyn.src','RMSDYN','MASS2',NSLCT,crl=MASS2)
       call chmdealloc('rmsdyn.src','RMSDYN','ID_hv',siz_id,siz_id,crl=ID)
       IF(PQUNIT > 0)THEN
          call chmdealloc('rmsdyn.src','RMSDYN','PQ',2*NP,crl=PQ)
          call chmdealloc('rmsdyn.src','RMSDYN','H',NP*(2*NP+1),crl=H)
          call chmdealloc('rmsdyn.src','RMSDYN','G',2*NP,crl=G)
          call chmdealloc('rmsdyn.src','RMSDYN','W',6*NP,crl=W)
       ENDIF
    ELSE
       CALL WRNDIE(-1,'<RMSDYN>','UNSUPPORTED OPTION')
    ENDIF
    call chmdealloc('rmsdyn.src','rmsdyn','TEMP',NATOM,cr4=TEMP)
    call chmdealloc('rmsdyn.src','rmsdyn','ISLCT',Natom,intg=ISLCT)
    if(ldrms .and. nslct2>0) call chmdealloc('rmsdyn.src','rmsdyn','JSLCT',Natom,intg=JSLCT)
    call chmdealloc('rmsdyn.src','rmsdyn','FREEAT',NATom,intg=FREEAT)
    RETURN
  END SUBROUTINE RMSDYN


  subroutine rdtrj(natom,islct,np,nsl,jslct,nsl2,x1,firstu,nunit,begin,skip,stop)

    ! Read coordinates for selected atoms from trajectory into memory, storing
    ! only these coordinates in a compact (contracted) form.
    !
    ! X1(1,II,J)=X(I;t)
    ! X1(2,II,J)=Y(I;t)
    ! X1(3,II,J)=Z(I;t)
    ! t=J*DT
    ! II runs from 1 to NSEL, simply indicating the II:th selected atom
    !
    ! If these coordinates are to be used together with regular PSF atom indexing
    ! a mapping has to be created:
    !  CALL REVIDX(NATOM,ISLCT,ISEL), which returns
    !  II=ISEL(I)
    !
    ! L. Nilsson, October 2002

    use coord
    use ctitla
    use exfunc
    use number
    use cvio
    use memory

    integer natom,np,nsl,nsl2,islct(*),jslct(*)
    integer firstu,nunit,begin,skip,stop
    real(chm_real4) x1(3,nsl+nsl2,np)

    integer i,j,k,nfreat
    integer istep,istats,ndegf,nfile,iunit,nsavv
    real(chm_real) delta
    character(len=4) :: corhd='CORD',VELHD='VELD'
    real(chm_real4),allocatable,dimension(:) :: tmparr
    integer,allocatable,dimension(:) :: freeat

    if(np /=  1 +(stop-begin)/skip)then
       CALL WRNDIE(-3,'<RDTRJ>','MISMATCH IN STEP SPECIFICATION')
    endif

    call chmalloc('rmsdyn.src','rdtrj','tmparr',natom,cr4=tmparr)
    call chmalloc('rmsdyn.src','rdtrj','freeat',natom,intg=freeat)
    iunit=firstu
    istats=1
    do i=1,np
       call readcv(x,y,z, &
#if KEY_CHEQ==1
            (/ zero /), .false., &  
#endif
            tmparr,natom,freeat,nfreat, &
            firstu,nunit,iunit,nfile, &
            istep,istats,ndegf,delta, &
            begin,stop,skip,nsavv,corhd,velhd, &
            titleb,ntitlb,.false., (/ zero /), .true.)
       j=0
       do k=1,natom
          if(islct(k) == 1)then
             j=j+1
             x1(1,j,i)=x(k)
             x1(2,j,i)=y(k)
             x1(3,j,i)=z(k)
          endif
       enddo
       if(nsl2 > 0)then
         do k=1,natom
           if(jslct(k) == 1)then
             j=j+1
             x1(1,j,i)=x(k)
             x1(2,j,i)=y(k)
             x1(3,j,i)=z(k)
           endif
         enddo
       endif
    enddo

    call chmdealloc('rmsdyn.src','RDTRJ','TMParr',NATOM,cr4=TMParr)
    call chmdealloc('rmsdyn.src','RDTRJ','FREEAT',NATOM,intg=FREEAT)
    return
  end subroutine rdtrj

  subroutine cpmsl(natom,islct,amass,amass2)

    ! Copy masses for selected atoms to first consecutive positions in
    ! AMASS2; also set first NSLCT positions of ISLCT=1 so it can be
    ! used for a contracted coordinate array

    integer natom,islct(*)
    real(chm_real) amass(*),amass2(*)
    integer i,j
    !
    j=0
    do i=1,natom
       if(islct(i) == 1)then
          j=j+1
          amass2(j)=amass(i)
          islct(j)=1
       endif
    enddo
    return
  end subroutine cpmsl

  subroutine rmsdy2(x1,x2, &
       natom,nslct2,islct,iscr,outrms,pqunit,pqseed,amass,lmass, &
       lsymm,lmatrix,lweig,lrms,ldrms,ldiff,lnoro,d,np,np2,f,pq, &
       h,g,w,iopt,nsig,maxfn)

    use clcg_mod,only:random
    use number
    use exfunc
    use coord
    use coordc
    use corsubs,only:orintc
    use stream
    use param_store, only: get_param, set_param

    implicit none

    integer natom,islct(*),iscr(*),nslct2
    integer outrms,pqunit,pqseed,np,np2
    logical lmatrix,lrms,ldrms,ldiff,lmass,lnoro,lweig,lsymm
    real(chm_real) amass(*),d(np,np2),pq(*),h(*),g(*),w(*),f
    ! Coordinates are stored as REAL*4, and we save much on these possibly
    ! large arrays by keeping this precision
    real(chm_real4) x1(3,natom+nslct2,np),x2(3,natom+nslct2,np2)

    integer i,j,jj,k,nposit,k1,j1,ks
    real(chm_real) rmsr,pqrange,xi,yi,zi,xci,yci,zci,xij,yij,zij,xcij,ycij,zcij,rij,rcij,totrms
    integer npar,nsig,maxfn,iopt,ier,pqs1,iout,nbpair
    logical lprint
    character(len=20) matfmt

    lprint=(prnlev >= 8)
    pqs1=pqseed
    totrms=zero
    if(lmatrix) d=zero
    do j = 1,np
       do i=1,natom+nslct2
          x(i)=x1(1,i,j)
          y(i)=x1(2,i,j)
          z(i)=x1(3,i,j)
       enddo
       ks=1
       if(lsymm) ks=j+1
       do k= ks,np2
          do i=1,natom+nslct2
             xcomp(i)=x2(1,i,k)
             ycomp(i)=x2(2,i,k)
             zcomp(i)=x2(3,i,k)
          enddo

          if(LRMS) THEN
            call orintc(natom,x,y,z,xcomp,ycomp,zcomp,amass,lmass,lrms, &
               iscr,islct,lweig,wmain,lnoro,lprint)
            call get_param('RMS ', RMSR)
            totrms=totrms+rmsr
          else if (LDIFF)then
            rmsr=ZERO
            do i=1,natom
              xij=x(i)-xcomp(i)
              yij=y(i)-ycomp(i)
              zij=z(i)-zcomp(i)
              rmsr=rmsr+xij**2+yij**2+zij**2
            enddo
            rmsr=sqrt(rmsr/natom)
            totrms=totrms+rmsr
          else if(LDRMS) THEN
            NBPAIR=0
            RMSR=ZERO
            if(nslct2 == 0)then
              DO I=1,NATOM
                XI=X(I)
                YI=Y(I)
                ZI=Z(I)
                XCI=XCOMP(I)
                YCI=YCOMP(I)
                ZCI=ZCOMP(I)
                DO JJ=I+1,NATOM
                  NBPAIR=NBPAIR+1
                  XIJ=(XI-X(JJ))**2
                  YIJ=(YI-Y(JJ))**2
                  ZIJ=(ZI-Z(JJ))**2
                  XCIJ=(XCI-XCOMP(JJ))**2
                  YCIJ=(YCI-YCOMP(JJ))**2
                  ZCIJ=(ZCI-ZCOMP(JJ))**2
                  RIJ=SQRT(XIJ+YIJ+ZIJ)
                  RCIJ=SQRT(XCIJ+YCIJ+ZCIJ)
                  RMSR=RMSR+(RIJ-RCIJ)**2
                ENDDO
              ENDDO
            else 
              DO I=1,NATOM
                XI=X(I)
                YI=Y(I)
                ZI=Z(I)
                XCI=XCOMP(I)
                YCI=YCOMP(I)
                ZCI=ZCOMP(I)
                DO JJ=NATOM+1,NATOM+NSLCT2
                  XIJ=(XI-X(JJ))**2
                  YIJ=(YI-Y(JJ))**2
                  ZIJ=(ZI-Z(JJ))**2
                  RIJ=(XIJ+YIJ+ZIJ)
                  ! exclude distance=0, most likely the two atoms are the same
                  IF(RIJ /= ZERO) THEN
                    NBPAIR=NBPAIR+1
                    XCIJ=(XCI-XCOMP(JJ))**2
                    YCIJ=(YCI-YCOMP(JJ))**2
                    ZCIJ=(ZCI-ZCOMP(JJ))**2
                    RIJ=SQRT(RIJ) 
                    RCIJ=SQRT(XCIJ+YCIJ+ZCIJ)
                    RMSR=RMSR+(RIJ-RCIJ)**2
                  ENDIF
                ENDDO
              ENDDO
            endif
            rmsr=sqrt(rmsr/nbpair)
            totrms=totrms+rmsr
          endif
          if(lmatrix)then
             d(j,k)=rmsr
             if(lsymm) d(k,j)=rmsr
          else
             if(prnlev >= 2) write(outrms,1314)j,k,rmsr
          endif
1314      format(1x,2i5,f12.5)
!
          if(prnlev > 8) then
             if(lnoro) then
                write(outu,52) 'TRANSLATED'
             else
                write(outu,52) 'ORIENTED'
             endif
52           format(' SELECTED COORDINATES ',A,' IN THE MAIN SET')
          endif
       enddo
    enddo

    if (lmatrix.and.lsymm) then
       do i=1,np
          d(i,i)=0.0
       enddo
    endif
    IF(LMATRIX.AND.OUTRMS > 0.AND.PRNLEV >= 2)THEN
       ! Output rmsdistance matrix in matrix form
       WRITE(MATFMT,'(A,I5,A)')'(',NP2,'G16.8)'
       DO I=1,NP
          WRITE(OUTRMS,MATFMT) (D(I,J),J=1,NP2)
       ENDDO
    ENDIF
!
! ?TOTRMS can be used for automatic evaluation of testcases
    call set_param('TOTRMS',totrms)

    IF(PQUNIT > 0)THEN
       !
       ! Estimate of {Pi,Qi} according to Levitt
       !
       ! Initial guesses, uniform random in interval of same magnitude as
       ! RMSDs
       PQRANGE=0.0
       DO I=1,NP
          DO J=I+1,NP
             PQRANGE=MAX(PQRANGE,D(J,I))
          ENDDO
       ENDDO
       NPAR=2*NP
       DO I=1,NPAR
          PQ(I)=PQRANGE*RANDOM(PQSEED)-PQRANGE/2.0
       ENDDO
       !
       IF(PRNLEV >= 6)THEN
          WRITE(OUTU,'(A,I10)') 'INITIAL GUESSES. SEED: ',PQS1
          WRITE(OUTU,'(7X,A)') 'I        Pi         Qi'
          WRITE(OUTU,'(I8,2G15.3)') (I,PQ(I),PQ(I+NP),I=1,NP)
          CALL PQDIST(NPAR,PQ,F)
          WRITE(OUTU,'(/,A,G12.3,/)') &
               ' GIVING INITIAL VALUE OF TARGET FUNCTION: ', F
       ENDIF
       !
       !      NSIG=3
       !      MAXFN=100*NPAR
       !      IOPT=0
       IOUT=-1
       IF(PRNLEV >= 6) IOUT=OUTU
       CALL ZXMIN(PQDIST,NPAR,NSIG,MAXFN,IOPT,PQ,H,G,F,W,IER)
       ! Error and convergence checks
       IF(IER == 129)THEN
          CALL WRNDIE(-2,'<RMSDYN/ZXMIN>', &
               'HESSIAN NOT POSITIVE DEFINITE')
       ELSEIF(IER == 130)THEN
          CALL WRNDIE(1,'<RMSDYN/ZXMIN>', &
               'ROUNDING ERRORS DOMINATE')
       ELSEIF(IER == 131)THEN
          CALL WRNDIE(1,'<RMSDYN/ZXMIN>', &
               'MAX FUNCTION EVALUATIONS REACHED')
          IF(PRNLEV >= 2) WRITE(OUTU,'(A,I10/)') ' **** MAXFN=',MAXFN
       ENDIF
       !
       call set_param('PQRES',F)
       ! Results
       IF(PRNLEV  >=  2)THEN
          IF(IER > 0) WRITE(OUTU,'(/6X,A,/)') &
               '*** *** POSSIBLY NOT  CONVERGED!!'
          WRITE(OUTU,100) PQS1,NP,F,W(1),W(2),W(3)
100       FORMAT(/'  2D-PROJECTION OF TRAJECTORY. RANDOM START:',I10, &
               /'        NUMBER OF (P,Q)-pairs:',I12, &
               /'      FINAL VALUE OF RESIDUAL:',G12.3, &
               /'                     GRADIENT:',G12.3, &
               /'    # OF FUNCTION EVALUATIONS:',F12.0, &
               /'      # OF SIGNIFICANT DIGITS:',F12.0/)
          !
          WRITE(PQUNIT,'(7X,A)') 'I        Pi         Qi'
          WRITE(PQUNIT,'(I8,2G15.3)') (I,PQ(I),PQ(I+NP),I=1,NP)
       ENDIF
       !
    ENDIF
    RETURN
  END SUBROUTINE RMSDY2

  SUBROUTINE PQDIST(N,PQ,F)

    INTEGER N
    real(chm_real) PQ(N),F

    CALL PQD2(N,PQ,F,ID)
    RETURN
  END SUBROUTINE PQDIST

  SUBROUTINE PQD2(N,PQ,F,D)

    INTEGER N
    real(chm_real) PQ(N),F,D(N/2,N/2)
    INTEGER NP,I,J

    !
    NP=N/2
    F=0.0
    DO I=1,NP
       DO J=I+1,NP
          F=F+(D(J,I)-SQRT((PQ(I)-PQ(J))**2+(PQ(I+NP)-PQ(J+NP))**2))**2
       ENDDO
    ENDDO
    !
  end SUBROUTINE PQD2
#endif 
  SUBROUTINE NULL_RD
    RETURN
  END SUBROUTINE NULL_RD

end module rmsdyn_mod

