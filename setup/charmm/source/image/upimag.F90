module imgup
  implicit none
  private

  public upimag0, upimag, imhbon

contains

  subroutine UPIMAG0(X, Y, Z, WMAIN, LDYNA)
    use chm_kinds
    use number
    implicit none
    real(chm_real) :: X(*), Y(*), Z(*), WMAIN(*)
    integer :: LDYNA
    real(chm_real),parameter :: upim0(1) = (/ ZERO /)

    call UPIMAG(X, Y, Z, WMAIN, LDYNA, upim0, upim0, upim0, upim0, upim0, upim0)
    return
  end subroutine UPIMAG0

  SUBROUTINE UPIMAG(X,Y,Z,WMAIN,LDYNA, &
       XOLD,YOLD,ZOLD,VX,VY,VZ,force_upimag_on_all)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE IS THE CONTROL ROUTINE FOR THE PROCESS OF
    !     GENERATING IMAGE ATOMS AND UPDATING ALL ARRAYS PERTAINING TO
    !     THESE ATOMS.
    !
    !     By Bernard R. Brooks    8/9/83
    !
    use memory
#if KEY_LOOKUP==1
    use LOOKUP,only:qvv,qvu,mkwwflg,iwwflg,nwwoim  
#endif
#if KEY_FLUCQ==1
    use flucqm,only:fqcfor                         
#endif
    use chm_kinds
    use chm_types
    use intcor_module
    use dimens_fcm
    use bases_fcm
    use psf
    use inbnd
    use image
    use image_routines_module, only: imcent, transo
    use number, only: zero
    use stream
    use param_store, only: set_param
    use pbound
    use pert
    use coord, only: tstcrd
#if KEY_GRAPE==1
    use grape       
#endif
#if KEY_FLUCQ==1
    use flucq
#endif 
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,testcoord_domdec  
#endif
    use fast,only:lfast
    !.ab.
#if KEY_BLOCK==1
    use trunk,only:ptable 
#endif
    use upimag_util,only:upimnb
    !.ab.
#ifdef KEY_RESIZE
    use resize,only:rszf
#endif    

    implicit none
    !
    integer,allocatable,dimension(:) :: ITR
    integer,allocatable,dimension(:) :: JMATPT
    integer,allocatable,dimension(:) :: JMATTR
    integer,allocatable,dimension(:) :: IMJBLO
    real(chm_real) X(*),Y(*),Z(*),XOLD(*),YOLD(*),ZOLD(*),WMAIN(*)
    real(chm_real) VX(*),VY(*),VZ(*)
    INTEGER LDYNA
    logical, optional :: force_upimag_on_all

    INTEGER NATIMO
#ifdef KEY_RESIZE
    integer i0
#endif
    LOGICAL LOK
    real(chm_real),allocatable,dimension(:) :: R2MIN
    real(chm_real),allocatable,dimension(:) :: RSCMX, RSCMY, RSCMZ
    real(chm_real),allocatable,dimension(:) :: RSXMAX, RSYMAX, RSZMAX

    logical check_groups

    ! Set check_groups = .true. for DOMDEC
    check_groups = .false.
    if (.not.present(force_upimag_on_all)) then
#if KEY_DOMDEC==1
       if (q_domdec) check_groups = .true.
#endif 
    endif

    NATIMO = 0

    IF(NTRANS.EQ.0) RETURN
    ! we need imall=true for parallel gpu code (pressure calcs)
    ! unfortunately igrape flag is not parsed yet so we do this for all
    ! GPU methods... I think in most of them this is ignored anyway!
#if KEY_GRAPE==1
    if(lgrape) limall=.true.                 
#endif

#if KEY_PBOUND==1 /*pbound*/
    IF (NATIM.EQ.0.and. &
         (.not.qboun ) ) THEN
#else /* (pbound)*/
    IF (NATIM.EQ.0) THEN
#endif /* (pbound)*/
#if KEY_NOMISC==0
#if KEY_DOMDEC==1
       if (q_domdec .and. .not.present(force_upimag_on_all)) then
          call testcoord_domdec(lok,x,y,z)
       else
#endif 
          CALL TSTCRD(LOK,X,Y,Z,NATOM)
#if KEY_DOMDEC==1
       endif
#endif
       IF (.NOT.(LOK)) THEN
          CALL WRNDIE(-1,'<UPIMAG>','NO IMAGE ATOMS GENERATED')
          RETURN
       ENDIF
#endif 


#if KEY_PERT==1
       IF(NIMBON.GT.0 .AND. QPERT) THEN
          CALL WRNDIE(-4,'<UPIMAG>','No code exists for using PERT with primary-image bonds')
       ENDIF
#endif

       BIMAG%IMATPT(1:NTRANS) = 0
       BIMAG%IMIBLO(1:NATOM) = 0
    ELSE
#if KEY_PBOUND==1 /*pbound*/
       If (.not.  ( qBoun ) ) Then
#endif /* (pbound)*/
          NATIMO=NATIM
          ! rmv apr2013; next 6 lines relocated from below to fix IMPATCH problem
          ! old mapping must be saved prior to calling MKIMAT or MKIMAT2
          call chmalloc('upimag.src','UPIMAG','ITR',NATIM,intg=ITR)
          call chmalloc('upimag.src','UPIMAG','JMATPT',NTRANS,intg=JMATPT)
          call chmalloc('upimag.src','UPIMAG','JMATTR',NATIM,intg=JMATTR)
          ! SAVE CURRENT MAPPING OF PRIMARY TO IMAGE ATOMS
          JMATPT(1:NTRANS) = BIMAG%IMATPT(1:NTRANS)
          JMATTR(1:NATIM) = BIMAG%IMATTR(1:NATIM)
#if KEY_PBOUND==1 /*pbound*/
       Endif
#endif /* (pbound)*/
    ENDIF
    !
    IF(LIMCEN) THEN
       !       PROCESS CENTERING IF REQUSTED
       CALL IMCENT(IMXCEN,IMYCEN,IMZCEN,BIMAG%IMCENF, &
            NTRANS,IMTRNS,IMNAME,X,Y,Z,LDYNA,XOLD,YOLD,ZOLD,VX,VY,VZ,check_groups, &
            NumLattice,ILATT,lattice_vector)
    ENDIF
#if KEY_PBOUND==1 /*pbound*/
    If (qBoun  ) Return
#endif /* (pbound)*/

#if KEY_GRAPE==1
    !  write(*,*)'UPIMAG>qgpupd=',qgpupd
    qgpupd = .true.
#endif 

    if (check_groups) return
#ifdef KEY_RESIZE
    i0=ngrp!*rszf*5/4 ! give 25% room
    call chmalloc('upimag.src','UPIMAG','RSCMX',i0,crl=RSCMX)
    call chmalloc('upimag.src','UPIMAG','RSCMY',i0,crl=RSCMY)
    call chmalloc('upimag.src','UPIMAG','RSCMZ',i0,crl=RSCMZ)
    call chmalloc('upimag.src','UPIMAG','RSXMAX',i0,crl=RSXMAX)
    call chmalloc('upimag.src','UPIMAG','RSYMAX',i0,crl=RSYMAX)
    call chmalloc('upimag.src','UPIMAG','RSZMAX',i0,crl=RSZMAX)
#else    
    call chmalloc('upimag.src','UPIMAG','RSCMX',MAXGRP,crl=RSCMX)
    call chmalloc('upimag.src','UPIMAG','RSCMY',MAXGRP,crl=RSCMY)
    call chmalloc('upimag.src','UPIMAG','RSCMZ',MAXGRP,crl=RSCMZ)
    call chmalloc('upimag.src','UPIMAG','RSXMAX',MAXGRP,crl=RSXMAX)
    call chmalloc('upimag.src','UPIMAG','RSYMAX',MAXGRP,crl=RSYMAX)
    call chmalloc('upimag.src','UPIMAG','RSZMAX',MAXGRP,crl=RSZMAX)
#endif
    call chmalloc('upimag.src','UPIMAG','R2MIN',NTRANS,crl=R2MIN)
    !  update images
    !.ab.
#if KEY_BLOCK==1
    ptable=-1 
#endif
    !.ab.
#if KEY_LJPME==1
    ! We need to ensure that the C6 coefficients are defined for atoms before
    ! updating images.
    call update_ljpme
#endif
    IF (.not.LMKMA1) THEN  !"linearized" update
       CALL MKIMAT2(X,Y,Z,WMAIN,CUTIM,NTRANS,NIMRES,NATIM,NIMGRP, &
#if KEY_PERT==1
            PERTIG,PPIGPTP,PPIGPBS,PERTIP, &
            PPIAC,PPIACNB,PPCG,PPRSCLF, &
#endif 
            IMTRNS,NOROT,BIMAG%IMATPT,BIMAG%IMATTR, &
            IMNAME,LIMINV,LIMALL,IMINV,RSCMX,RSCMY, &
            RSCMZ,RSXMAX,RSYMAX,RSZMAX, &
            R2MIN &
#if KEY_WCA==1
            ,PPWCA           & 
#endif
            )
    ELSE  !original update code

       CALL MKIMAT(X,Y,Z,WMAIN,CUTIM,NTRANS,NIMRES,NATIM,NIMGRP, &
#if KEY_PERT==1
            PERTIG,PPIGPTP,PPIGPBS,PERTIP, &
            PPIAC,PPIACNB,PPCG,PPRSCLF, &
#endif 
            IMTRNS,NOROT,BIMAG%IMATPT,BIMAG%IMATTR, &
            IMNAME,LIMINV,LIMALL,IMINV,RSCMX,RSCMY, &
            RSCMZ,RSXMAX,RSYMAX,RSZMAX, &
            R2MIN &
#if KEY_WCA==1
            ,PPWCA           & 
#endif
            )
    ENDIF
#ifdef KEY_RESIZE    
    call chmdealloc('upimag.src','UPIMAG','RSCMX',i0,crl=RSCMX)
    call chmdealloc('upimag.src','UPIMAG','RSCMY',i0,crl=RSCMY)
    call chmdealloc('upimag.src','UPIMAG','RSCMZ',i0,crl=RSCMZ)
    call chmdealloc('upimag.src','UPIMAG','RSXMAX',i0,crl=RSXMAX)
    call chmdealloc('upimag.src','UPIMAG','RSYMAX',i0,crl=RSYMAX)
    call chmdealloc('upimag.src','UPIMAG','RSZMAX',i0,crl=RSZMAX)
#else
    call chmdealloc('upimag.src','UPIMAG','RSCMX',MAXGRP,crl=RSCMX)
    call chmdealloc('upimag.src','UPIMAG','RSCMY',MAXGRP,crl=RSCMY)
    call chmdealloc('upimag.src','UPIMAG','RSCMZ',MAXGRP,crl=RSCMZ)
    call chmdealloc('upimag.src','UPIMAG','RSXMAX',MAXGRP,crl=RSXMAX)
    call chmdealloc('upimag.src','UPIMAG','RSYMAX',MAXGRP,crl=RSYMAX)
    call chmdealloc('upimag.src','UPIMAG','RSZMAX',MAXGRP,crl=RSZMAX)
#endif
    call chmdealloc('upimag.src','UPIMAG','R2MIN',NTRANS,crl=R2MIN)

    CALL set_param('NATI',NATIM)
    !
#if KEY_LOOKUP==1
    IF(QVV.OR.QVU) CALL MKWWFLG(NATIM,NWWOIM)                  
#endif
    !
    !  Create image coordinates
    !
    CALL TRANSO(X, Y, Z, &
         (/ zero /),  (/ zero /),  (/ zero /), &
         .FALSE., .FALSE., &
         (/ zero /), &
         NATOM,NTRANS,IMTRNS, &
         BIMAG%IMATPT,BIMAG%IMATTR,NOROT,NATIM &
#if KEY_FLUCQ==1
         ,QFLUC,CG,FQCFOR    & 
#endif
    )
    !
    !     NOW REMAP ALL INTERNAL AND H-BOND STUFF
    !
    IF (NATIMO > 0) THEN
       CALL IMMAP(NTRANS,BIMAG%IMATPT, &
            BIMAG%IMATTR,JMATPT,JMATTR,ITR, &
            NIMHB,NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP, &
            NIMEXCL,IMEXCLI,IMEXCLJ,IMEXCLT, &
#if KEY_CMAP==1
            NIMCRT, &  
#endif
            icr_struct%lenic, icr_struct%iar, &
            icr_struct%jar, icr_struct%kar, &
            icr_struct%lar)

       call chmdealloc('upimag.src','UPIMAG','ITR',NATIMO,intg=ITR)
       call chmdealloc('upimag.src','UPIMAG','JMATPT',NTRANS,intg=JMATPT)
       call chmdealloc('upimag.src','UPIMAG','JMATTR',NATIMO,intg=JMATTR)
    ENDIF

    CALL UPIMNB(BIMAG)
    !
    RETURN
  END SUBROUTINE UPIMAG

  SUBROUTINE MKIMAT(X,Y,Z,WMAIN,CUTIM,NTRANS,NIMRES,NATIM,NIMGRP,&
#if KEY_PERT==1
       IGPERT,IGPTPP,IGPBSP,IPERT, &
#endif
#if KEY_PERT==1
       IACP,IACNBP,CGP,RSCLFP,                & 
#endif
       IMTRNS,NOROT,IMATPT,IMATTR,IMNAME, &
       LIMINV,LIMALL,IMINV,RSCMX,RSCMY,RSCMZ, &
       RSXMAX,RSYMAX,RSZMAX,R2MIN &
#if KEY_WCA==1
       ,WCAP                                   & 
#endif
       )
    !-----------------------------------------------------------------------
    !     This routine constructs the image atom lists
    !
    !     By Bernard R. Brooks  3-AUG-1983
    !
    !     Coding Tuning by Scott R. Brozell May 2002
    !     This subroutine calculates the distance between pairs of groups
    !     and compares it to a threshold.
    !     The classic optimization of the straightforward algorithm,
    !     wherein the sum of squares is followed by the comparison test,
    !     is to perform successive comparisons on the partial sums.
    !     This trades floating point operations for conditional evaluations.
    !     In addition, a clever mathematical expression based on the fact
    !     that IMOVEG(IRS) is either 0 or 1 was transformed into a
    !     loop-invariant if-statement that was then hoisted out of the loop.
    !     Since this subroutine contains duplicate copies of the distance
    !     algorithm (for the serial and the parallel code sections),
    !     the tuning occurs twice.
    !     The values formerly designated as Min-Distance in the output
    !     are now merely upper bounds to the minimum distances.
    !
#if KEY_LOOKUP==1
    use LOOKUP              
#endif
#if KEY_CHEQ==1
    use cheq,only:ech,eha,qcg,qpartbin,allocate_cheq   
#endif

    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use stream
    use pert
    use fast
    use timerm
    use memory
    !EPO - variable LJ cutoff
    use varcutm
    !EPO
    use block_ltm
    use tbmts_ltm
#if KEY_SCCDFTB==1
    use sccdftb
    use gamess_fcm
#endif 
    use parallel
#if KEY_ASPENER==1
    use eef1_mod           
#endif
#if KEY_PIPF==1
    use pipfm              
#endif
    use chutil,only:hydrog
    use machutil,only:timrb,timre
    use mtp_fcm
    use mtpl_fcm

    use dcm_fcm,only: QDCM,IMGINIT,INDQ,DCMSETIM

    implicit none

    integer,allocatable,dimension(:) :: IPKMAT
    real(chm_real)  X(*),Y(*),Z(*),WMAIN(*)
    real(chm_real)  CUTIM
    INTEGER NTRANS,NATIM,NIMGRP,NIMRES(*),IMINV(*)
#if KEY_PERT==1
    INTEGER IGPERT(*),IGPTPP(*),IGPBSP(*),IPERT(*),IACP(*),IACNBP(*)
    real(chm_real) CGP(*),RSCLFP(*)
#endif 
#if KEY_WCA==1
    real(chm_real) WCAP(*)              
#endif
    real(chm_real) IMTRNS(*)
    LOGICAL NOROT,LIMALL,LIMINV
    INTEGER IMATPT(*),IMATTR(*)
    CHARACTER(len=*) IMNAME(*)
    real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
    real(chm_real)  RSXMAX(*),RSYMAX(*),RSZMAX(*),R2MIN(*)
    !
#if KEY_PARALLEL==1 /*pardecl*/
    INTEGER IMYNOD,VAL,JPT,KPT,IPKMSZ,VMASK
#endif /* (pardecl)*/
    !
    LOGICAL ACCEPT,NEWTRN
    !
    real(chm_real) XMAXP,YMAXP,ZMAXP,XMINP,YMINP,ZMINP
    real(chm_real) XMAX,YMAX,ZMAX,XMIN,YMIN,ZMIN,XD,YD,ZD
    real(chm_real) RSCXP,RSCYP,RSCZP,RSXMXP,RSYMXP,RSZMXP
    real(chm_real) RSCX,RSCY,RSCZ,RSXMX,RSYMX,RSZMX,XD1,YD1,ZD1
    real(chm_real) CTIMSQ,R2MINA,R2MINB,R2
    INTEGER I,J,K,IPT,ISTRT,ILAST,ITMP
    INTEGER IS,IQ,ITRANS,IRES,IRS,JRS
    INTEGER NAT,NATM,NATMO,KRS,KRSO
    INTEGER NSGIM,NRSIM
    CHARACTER(len=8) CHX
    real(chm_real) R2BIG
    DATA   R2BIG/9.99D12/
    !
    !
    IF (TIMER.GT.0) CALL TIMRB
    IF(PRNLEV.GE.5) WRITE(OUTU,'(/,A)') &
         ' <MKIMAT>: updating the image atom lists and remapping'
    IF(PRNLEV.GE.5) WRITE(OUTU,'(A)') &
         ' Transformation   Atoms  Groups  Residues  Upper-Bound '
    !    $  ' Transformation   Atoms  Groups  Residues  Min-Distance'
    ! Code tuning has changed this value from a minimum distance to
    ! an upper bound to the minimum distance.  SRB, May 2002.
    !
    !-----------initialize array for atomic multipoles--MTP and MTPL----
    IF (QMTP) IMOL = MOLS
#if KEY_MTPL==1
    IF (QMTPL) IMOLL = MOLSL
#endif 

    IF (QDCM) THEN
     INDQ=0
    ENDIF

    DO I = 1,NATOM
      IMATTR(I)=I
    ENDDO

    !-------------------------------------------------------------------
    CTIMSQ=CUTIM*CUTIM
    !
    !     Set up group centers and sizes
    !
    XMAXP=X(1)
    XMINP=X(1)
    YMAXP=Y(1)
    YMINP=Y(1)
    ZMAXP=Z(1)
    ZMINP=Z(1)
    DO I=1,NGRP
       IS=IGPBS(I)+1
       IQ=IGPBS(I+1)
       NAT=IQ-IS+1
       IF(NAT.LE.0) CALL WRNDIE(-5,'<MKIMAT>', &
            'Group with no atoms found')
       XMIN=X(IS)
       XMAX=XMIN
       YMIN=Y(IS)
       YMAX=YMIN
       ZMIN=Z(IS)
       ZMAX=ZMIN
       DO J=IS+1,IQ
          IF(X(J).LT.XMIN) XMIN=X(J)
          IF(Y(J).LT.YMIN) YMIN=Y(J)
          IF(Z(J).LT.ZMIN) ZMIN=Z(J)
          IF(X(J).GT.XMAX) XMAX=X(J)
          IF(Y(J).GT.YMAX) YMAX=Y(J)
          IF(Z(J).GT.ZMAX) ZMAX=Z(J)
       ENDDO
       !
       !       Size of rectangular box surrounding group
       !
       RSXMAX(I)=(XMAX-XMIN)*0.5
       RSYMAX(I)=(YMAX-YMIN)*0.5
       RSZMAX(I)=(ZMAX-ZMIN)*0.5
       !       CENTER OF GROUP
       RSCMX(I)=(XMAX+XMIN)*0.5
       RSCMY(I)=(YMAX+YMIN)*0.5
       RSCMZ(I)=(ZMAX+ZMIN)*0.5
       !       GLOBAL EXTREEMS
       XMAXP=MAX(XMAXP,XMAX)
       XMINP=MIN(XMINP,XMIN)
       YMAXP=MAX(YMAXP,YMAX)
       YMINP=MIN(YMINP,YMIN)
       ZMAXP=MAX(ZMAXP,ZMAX)
       ZMINP=MIN(ZMINP,ZMIN)
    ENDDO
    !
    !
    !     Find spatial extent of primary space
    !
    RSCXP=(XMAXP+XMINP)*0.5
    RSCYP=(YMAXP+YMINP)*0.5
    RSCZP=(ZMAXP+ZMINP)*0.5
    RSXMXP=XMAXP-RSCXP
    RSYMXP=YMAXP-RSCYP
    RSZMXP=ZMAXP-RSCZP
    !
#if KEY_PARALLEL==1 /*parainit*/
    !     If compiled with PARALLEL and there is more than one node,
    !     setup a packed list.
    !
       ! initialize packing indicies
       KPT=0
       VAL=0
       IPKMSZ=NTRANS*NGRP/30+5
       call chmalloc('upimag.src','MKIMAT','IPKMAT',IPKMSZ,intg=IPKMAT)
       jpt = 1
       !
       !     Now decide how to treat each group using a rectangular
       !     box comparison with primary atoms
       !
       DO ITRANS=1,NTRANS
          IPT=(ITRANS-1)*12
          R2MINA=R2BIG
          R2MINB=R2BIG
          IF(LIMALL) THEN
             CONTINUE
          ELSE IF(.NOT.LIMINV) THEN
             CONTINUE
          ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
             GOTO 400
          ENDIF
          !
          DO IRS=1,NGRP
             ACCEPT=.FALSE.
#if KEY_PARAFULL==1 /*parfgroup*/
#if KEY_PARASCAL==1 /*parstest*/
#error  'Bad parallel nonbond compile option'
#endif /* (parstest)*/
             IF(MYNOD.EQ.MOD(ITRANS+IRS,NUMNOD)) THEN
#elif KEY_PARASCAL==1 /*parfgroup*/
                IS=IGPBS(IRS)+1
                JS=IGPBS(JRS)+1
             IF(MYNOD.EQ.IPMAT(IPBLOCK(IS),IPBLOCK(JS))) THEN
#elif KEY_SPACDEC==1 /*parfgroup*/
             IF(MYNOD == ICPUMAP(IS)) THEN
                CALL WRNDIE(-5,'<UPIMAG>','SPACDEC not supported.')
#else /* (parfgroup)*/
#error  'Bad parallel compile option'
#endif /* (parfgroup)*/
                !
                !
                IF(NOROT) THEN
                   RSCX=RSCMX(IRS)+IMTRNS(IPT+10)
                   RSCY=RSCMY(IRS)+IMTRNS(IPT+11)
                   RSCZ=RSCMZ(IRS)+IMTRNS(IPT+12)
                   RSXMX=RSXMAX(IRS)
                   RSYMX=RSYMAX(IRS)
                   RSZMX=RSZMAX(IRS)
                ELSE
                   RSCX=RSCMX(IRS)*IMTRNS(IPT+1)+RSCMY(IRS)*IMTRNS(IPT+2)+ &
                        RSCMZ(IRS)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                   RSCY=RSCMX(IRS)*IMTRNS(IPT+4)+RSCMY(IRS)*IMTRNS(IPT+5)+ &
                        RSCMZ(IRS)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                   RSCZ=RSCMX(IRS)*IMTRNS(IPT+7)+RSCMY(IRS)*IMTRNS(IPT+8)+ &
                        RSCMZ(IRS)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
                   RSXMX=MAX(RSXMAX(IRS),RSYMAX(IRS),RSZMAX(IRS))
                   RSYMX=RSXMX
                   RSZMX=RSXMX
                ENDIF
                !
                ! Now check if this image group is close enough to primary atoms
                !
                XD=MAX(ABS(RSCX-RSCXP)-RSXMX-RSXMXP,ZERO)
                YD=MAX(ABS(RSCY-RSCYP)-RSYMX-RSYMXP,ZERO)
                ZD=MAX(ABS(RSCZ-RSCZP)-RSZMX-RSZMXP,ZERO)
                R2=XD*XD+YD*YD+ZD*ZD
                IF(R2.GE.CTIMSQ) THEN
                   IF(R2.LT.R2MINA) R2MINA=R2
                   GOTO 500
                ENDIF
                !
                ! Now check to see if this group is its own image
                !
                XD=RSCX-RSCMX(IRS)
                YD=RSCY-RSCMY(IRS)
                ZD=RSCZ-RSCMZ(IRS)
                R2=XD*XD+YD*YD+ZD*ZD
                IF(R2.LT.PT001) THEN
                   IF(PRNLEV.GT.7) WRITE(OUTU,265) IRS,IMNAME(ITRANS)
                   GOTO 500
                ENDIF
                !
                ! Now decide how to treat each group pair using a rectangular
                !
                IF (LIMALL) THEN
                   ISTRT=1
                ELSE IF (.NOT.LIMINV) THEN
                   ISTRT=IRS
                ELSE IF (IMINV(ITRANS).EQ.ITRANS) THEN
                   ISTRT=IRS
                ELSE IF (IMINV(ITRANS).GT.ITRANS) THEN
                   ISTRT=1
                ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
                   GOTO 500
                ENDIF
                !
                !
                IF(IMOVEG(IRS).EQ.0) THEN
                   DO JRS = ISTRT,NGRP
                      XD = RSCX - RSCMX(JRS)
                      XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
                      R2 = XD1*XD1
                      IF(R2.LT.CTIMSQ) THEN
                         YD = RSCY - RSCMY(JRS)
                         YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
                         R2 = R2 + YD1*YD1
                         IF(R2.LT.CTIMSQ) THEN
                            ZD = RSCZ - RSCMZ(JRS)
                            ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
                            R2 = R2 + ZD1*ZD1
                            R2MINB = MIN(R2MINB,R2)
                            IF(R2.LT.CTIMSQ) THEN
                               ACCEPT = .TRUE.
                               R2MINB = ZERO
                               GOTO 500
                            ENDIF
                         ENDIF
                      ENDIF
                   ENDDO
                ELSE IF(IMOVEG(IRS).EQ.1) THEN
                   DO JRS = ISTRT,NGRP
                      XD = RSCX - RSCMX(JRS)
                      XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
                      R2 = XD1*XD1
                      IF(R2.LT.CTIMSQ) THEN
                         YD = RSCY - RSCMY(JRS)
                         YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
                         R2 = R2 + YD1*YD1
                         IF(R2.LT.CTIMSQ) THEN
                            ZD = RSCZ - RSCMZ(JRS)
                            ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
                            R2 = R2 + ZD1*ZD1
                            R2MINB = MIN(R2MINB,R2)
                            R2 = R2 + IMOVEG(JRS)*CTIMSQ
                            IF(R2.LT.CTIMSQ) THEN
                               ACCEPT = .TRUE.
                               R2MINB = ZERO
                               GOTO 500
                            ENDIF
                         ENDIF
                      ENDIF
                   ENDDO
                ELSE
                   IF(WRNLEV.GE.0) THEN
                      WRITE(OUTU,944) IMOVEG(IRS)
944                   FORMAT(' <MKIMAT>: Warning: IMOVEG(IRS) is ' &
                           ,I5,' It should be 0 or 1')
                   ENDIF
                ENDIF
                ! The untuned original.
                !            DO JRS = ISTRT,NGRP
                !              XD = RSCX - RSCMX(JRS)
                !              YD = RSCY - RSCMY(JRS)
                !              ZD = RSCZ - RSCMZ(JRS)
                !              XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
                !              YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
                !              ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
                !              R2 = XD1*XD1 + YD1*YD1 + ZD1*ZD1
                !              R2MINB = MIN(R2MINB,R2)
                !              R2 = R2 + IMOVEG(IRS)*IMOVEG(JRS)*CTIMSQ
                !              IF(R2.LT.CTIMSQ) THEN
                !                 ACCEPT = .TRUE.
                !                 R2MINB = ZERO
                !                 GOTO 500
                !              ENDIF
                !            ENDDO
                !
             ENDIF
500          CONTINUE
             !
             KPT=KPT+1
             VAL=VAL+VAL
             IF(ACCEPT) VAL=VAL+1
             IF(KPT.EQ.30) THEN
                ipkmat(jpt) = val
                VAL=0
                JPT=JPT+1
                KPT=0
             ENDIF
          ENDDO
          !
          IF(R2MINB.EQ.R2BIG) R2MINB=R2MINA
400       CONTINUE
          R2MIN(ITRANS)=R2MINB
       ENDDO
       !
420    CONTINUE
       KPT=KPT+1
       VAL=VAL+VAL
       IF(KPT.LT.30) GOTO 420
       ipkmat(jpt) = val

       !
       !    Now do a global OR operation on all of the packed lists.
       CALL GBOR(IPKMAT,JPT)
       !
       KPT=0
       jpt = 1
       val = ipkmat(jpt)
       !        use 30 bits to a word
       VMASK=2**29
    !
#endif /* (parainit)*/
    !
    ! main loop for image atom generation
    !
    NSGIM=NSEG
    NRSIM=NRES
    !
    KRS=NGRP
    NATM=NATOM
    KRSO=KRS
    NATMO=NATM
    !
    DO ITRANS=1,NTRANS
       IPT=(ITRANS-1)*12
       NEWTRN=.TRUE.
       ILAST=0
       IRES=0
       R2MINA=R2BIG
       R2MINB=R2BIG
#if KEY_PARALLEL==1
       IF(NUMNOD.GT.1) R2MINB=R2MIN(ITRANS)
#endif 
       IF (LIMALL) THEN
          CONTINUE
       ELSE IF (.NOT.LIMINV) THEN
          CONTINUE
       ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
          GOTO 700
       ENDIF
       !
       DO IRS=1,NGRP
          !
#if KEY_PARALLEL==1
          IF(NUMNOD.GT.1) THEN
             ACCEPT=(VAL.GE.VMASK)
             IF(ACCEPT) VAL=VAL-VMASK
             VAL=VAL+VAL
             KPT=KPT+1
             IF(KPT.EQ.30) THEN
                JPT=JPT+1
                val = ipkmat(jpt)
                KPT=0
             ENDIF
          ELSE
#endif 
             !
             ACCEPT=.FALSE.
             !
             IF(NOROT) THEN
                RSCX=RSCMX(IRS)+IMTRNS(IPT+10)
                RSCY=RSCMY(IRS)+IMTRNS(IPT+11)
                RSCZ=RSCMZ(IRS)+IMTRNS(IPT+12)
                RSXMX=RSXMAX(IRS)
                RSYMX=RSYMAX(IRS)
                RSZMX=RSZMAX(IRS)
             ELSE
                RSCX=RSCMX(IRS)*IMTRNS(IPT+1)+RSCMY(IRS)*IMTRNS(IPT+2)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                RSCY=RSCMX(IRS)*IMTRNS(IPT+4)+RSCMY(IRS)*IMTRNS(IPT+5)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                RSCZ=RSCMX(IRS)*IMTRNS(IPT+7)+RSCMY(IRS)*IMTRNS(IPT+8)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
                RSXMX=MAX(RSXMAX(IRS),RSYMAX(IRS),RSZMAX(IRS))
                RSYMX=RSXMX
                RSZMX=RSXMX
             ENDIF
             !
             ! Now check if this image group is close enough to primary atoms
             !
             XD=MAX(ABS(RSCX-RSCXP)-RSXMX-RSXMXP,ZERO)
             YD=MAX(ABS(RSCY-RSCYP)-RSYMX-RSYMXP,ZERO)
             ZD=MAX(ABS(RSCZ-RSCZP)-RSZMX-RSZMXP,ZERO)
             R2=XD*XD+YD*YD+ZD*ZD
             IF(R2.GE.CTIMSQ) THEN
                IF(R2.LT.R2MINA) R2MINA=R2
                GOTO 600
             ENDIF
             !
             ! Now check to see if this group is its own image
             !
             XD=RSCX-RSCMX(IRS)
             YD=RSCY-RSCMY(IRS)
             ZD=RSCZ-RSCMZ(IRS)
             R2=XD*XD+YD*YD+ZD*ZD
             IF(R2.LT.PT001) THEN
                IF(PRNLEV.GT.7) WRITE(OUTU,265) IRS,IMNAME(ITRANS)
265             FORMAT(' MKIMAT: Group',I6, &
                     ' excluded in transformation "', &
                     A,'" because it is its own image.')
                GOTO 600
             ENDIF
             !
             ! Now decide how to treat each group pair using a rectangular
             !
             IF (LIMALL) THEN
                ISTRT=1
             ELSE IF (.NOT.LIMINV) THEN
                ISTRT=IRS
             ELSE IF (IMINV(ITRANS).EQ.ITRANS) THEN
                ISTRT=IRS
             ELSE IF (IMINV(ITRANS).GT.ITRANS) THEN
                ISTRT=1
             ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
                GOTO 600
             ENDIF
             !
             ! Code tuning.  SRB, May 2002.
             ! IMOVEG(IRS) is either 0 or 1.
             !
             IF(IMOVEG(IRS).EQ.0) THEN
                DO JRS = ISTRT,NGRP
                   XD = RSCX - RSCMX(JRS)
                   XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
                   R2 = XD1*XD1
                   IF(R2.LT.CTIMSQ) THEN
                      YD = RSCY - RSCMY(JRS)
                      YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
                      R2 = R2 + YD1*YD1
                      IF(R2.LT.CTIMSQ) THEN
                         ZD = RSCZ - RSCMZ(JRS)
                         ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
                         R2 = R2 + ZD1*ZD1
                         R2MINB = MIN(R2MINB,R2)
                         IF(R2.LT.CTIMSQ) THEN
                            ACCEPT = .TRUE.
                            R2MINB = ZERO
                            GOTO 600
                         ENDIF
                      ENDIF
                   ENDIF
                ENDDO
             ELSE IF(IMOVEG(IRS).EQ.1) THEN
                DO JRS = ISTRT,NGRP
                   XD = RSCX - RSCMX(JRS)
                   XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
                   R2 = XD1*XD1
                   IF(R2.LT.CTIMSQ) THEN
                      YD = RSCY - RSCMY(JRS)
                      YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
                      R2 = R2 + YD1*YD1
                      IF(R2.LT.CTIMSQ) THEN
                         ZD = RSCZ - RSCMZ(JRS)
                         ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
                         R2 = R2 + ZD1*ZD1
                         R2MINB = MIN(R2MINB,R2)
                         R2 = R2 + IMOVEG(JRS)*CTIMSQ
                         IF(R2.LT.CTIMSQ) THEN
                            ACCEPT = .TRUE.
                            R2MINB = ZERO
                            GOTO 600
                         ENDIF
                      ENDIF
                   ENDIF
                ENDDO
             ELSE
                IF(WRNLEV.GE.0) THEN
                   WRITE(OUTU,945) IMOVEG(IRS)
945                FORMAT(' <MKIMAT>: Warning: IMOVEG(IRS) is ' &
                        ,I5,' It should be 0 or 1')
                ENDIF
             ENDIF
             ! The untuned original.
             !            DO JRS = ISTRT,NGRP
             !              XD = RSCX - RSCMX(JRS)
             !              YD = RSCY - RSCMY(JRS)
             !              ZD = RSCZ - RSCMZ(JRS)
             !              XD1 = MAX(ABS(XD)-RSXMX-RSXMAX(JRS),ZERO)
             !              YD1 = MAX(ABS(YD)-RSYMX-RSYMAX(JRS),ZERO)
             !              ZD1 = MAX(ABS(ZD)-RSZMX-RSZMAX(JRS),ZERO)
             !              R2 = XD1*XD1 + YD1*YD1 + ZD1*ZD1
             !              R2MINB = MIN(R2MINB,R2)
             !              R2 = R2 + IMOVEG(IRS)*IMOVEG(JRS)*CTIMSQ
             !              IF(R2.LT.CTIMSQ) THEN
             !                 ACCEPT = .TRUE.
             !                 R2MINB = ZERO
             !                 GOTO 600
             !              ENDIF
             !            ENDDO
             !
600          CONTINUE
             !
#if KEY_PARALLEL==1
          ENDIF
#endif 
          !
          IF (ACCEPT) THEN
             !
             IS=IGPBS(IRS)+1
             IQ=IGPBS(IRS+1)
             !
             !         Now add this image group in if it has been accepted
             !
             KRS=KRS+1
#ifdef KEY_RESIZE
             IF(KRS.GT.size(imoveg)) THEN
                IF(WRNLEV.GE.2) WRITE(OUTU,35) ITRANS,IRS,KRS
35              FORMAT(//' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT'/ &
                     ' ITRANS=',I5,' IRS=',I5,' KRS=',I5)
#else      
             IF(KRS.GT.MAXGRP) THEN
                IF(WRNLEV.GE.2) WRITE(OUTU,35) ITRANS,IRS,KRS,MAXGRP
35              FORMAT(//' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT'/ &
                     ' ITRANS=',I5,' IRS=',I5,' KRS=',I5,' MAXGRP=',I5)
#endif                
                CALL WRNDIE(-5,'<MKIMAT>', &
                     ' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT')
             ENDIF
             !
             IMOVEG(KRS)=IMOVEG(IRS)
             IGPTYP(KRS)=IGPTYP(IRS)
             IGPBS(KRS+1)=IQ-IS+1+NATM
#if KEY_PERT==1
             IF(QPERT) THEN
                IGPERT(KRS)=IGPERT(IRS)
                IGPTPP(KRS)=IGPTPP(IRS)
                IGPBSP(KRS+1)=IQ-IS+1+NATM
             ENDIF
#endif 
             !-----------addition loop for atomic multipoles--MTP-------
             IF (QMTP) CALL IMTP(NATM,IS,IQ)
#if KEY_MTPL==1
             IF (QMTPL) CALL IMTPL(NATM,IS,IQ)
#endif 

             IF (QDCM) CALL IMGINIT(NATM,IS,IQ)

             !----------------------------------------------------------
             !
             K=NATM
             DO I=IS,IQ
                K=K+1
#ifdef KEY_RESIZE
                IF(K.GT.size(cg)) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,25) ITRANS,IRS,IQ,IS
25                 FORMAT(/' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT'/ &
                        ' ITRANS=',I5,' IRS=',I5,' IQ=',I5,' IS=',I5)
#else
                IF(K.GT.MAXAIM) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,25) ITRANS,IRS,IQ,IS,MAXAIM
25                 FORMAT(/' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT'/ &
                        ' ITRANS=',I5,' IRS=',I5,' IQ=',I5,' IS=',I5,' MAXAIM=',I5)
#endif
                   CALL WRNDIE(-5,'<MKIMAT>', &
                        ' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT')
                ENDIF
                !
                !             Copy other arrays dimensioned NATOM
                !
                IMATTR(K)=I
                CHX=ATYPE(I)
                ATYPE(K)=CHX
                CG(K)=CG(I)
#if KEY_LJPME==1
                LJC6(K)=LJC6(I)
#endif
                AMASS(K)=AMASS(I)
                ICAAUTO(K)=ICAAUTO(I)
                if(qvarcut) VARCUT(K)=VARCUT(I)
                ! Drude force field stuff
                ISDRUDE(K)=ISDRUDE(I)
                ALPHADP(K)=ALPHADP(I)
                THOLEI(K)=THOLEI(I)
#if KEY_CHEQ==1
                if(qcg) then
                   ECH(K)=ECH(I)
                   EHA(K)=EHA(I)
                endif
#endif 

#if KEY_PIPF==1
                ! PJ 06/2005
                IF (QPIPF .AND. QPFDYN) THEN
                   !yw                  CALL UINDASG(K,I,IUINDSV)
                   CALL UINDASG(K,I,UIND)
                ENDIF
#endif 
                IAC(K)=IAC(I)
                IACNB(K)=IACNB(I)
                IMOVE(K)=IMOVE(I)
                RSCLF(K)=RSCLF(I)
                WMAIN(K)=WMAIN(I)
#if KEY_LOOKUP==1
                !LNI flag solvent atoms selected for nb lookup
                !CCCC How about lone-pairs and TIP4P????
                IF(QVV.OR.QVU) THEN
                   ITMP=IWWFLG(I)
                   IF(ITMP.GT.0)THEN
                      IF(.NOT.HYDROG(I))THEN
                         IWWFLG(K)=K
                      ELSE IF(.NOT.HYDROG(I-1))THEN
                         IWWFLG(K)=K-1
                      ELSE IF(.NOT.HYDROG(I-2))THEN
                         IWWFLG(K)=K-2
                      ENDIF
                   ELSE
                      IWWFLG(K)=0
                   ENDIF
                ENDIF
#endif 
#if KEY_WCA==1
                WCA(K)=WCA(I)        
#endif
#if KEY_SCCDFTB==1
                if(qmused)IGMSEL(K)=IGMSEL(I)  
#endif

#if KEY_ASPENER==1
                if(doeef1) then
                   GFREEI(K)=GFREEI(I)
                   VOLMI(K)=VOLMI(I)
                   SIGWI(K)=SIGWI(I)
                   FSIGWI(K)=FSIGWI(I)
                   SIGWI2(K)=SIGWI2(I)
                   FSIGWI2(K)=FSIGWI2(I)
                endif
#endif 
                !
#if KEY_MTS==1
                IF (QTBMTS) THEN
                   IMTS(K)=IMTS(I)
                   IF(TBHY1) THEN
                      IMTF(K)=IMTF(I)
                   ENDIF
                ENDIF
#endif 
                !
#if KEY_PERT==1
                IF(QPERT) THEN
                   IPERT(K)=IPERT(I)
                   CGP(K)=CGP(I)
                   IACP(K)=IACP(I)
                   IACNBP(K)=IACNBP(I)
                   RSCLFP(K)=RSCLFP(I)
#if KEY_WCA==1
                   WCAP(K)=WCAP(I)   
#endif

                   PPCG(K)=PPCG(I)
                   PPALPHA(K)=PPALPHA(I)
                   PPTHOLE(K)=PPTHOLE(I)
                   PPAMASS(K)=PPAMASS(I)
                ENDIF
#endif 
#if KEY_BLOCK==1
                !SBblock: copy content of IBLOCK/IBLCKP array of real atoms
                !         to respective image atoms.  This code is crude as
                !         it accesses the hp directly.  However, the only
                !         alternative would be to change the calling sequence of
                !         MKIMAT.  BTW, using the function I4VAL doesn't really
                !         help and seems to just adding another function call.
                !         Don't know whether this runs on a 64bit machine; maybe
                !         I4VAL has to be used there after all...
                IF(QBLOCK) THEN
                   IBLCKP(K-1+1) = IBLCKP(I-1+1)
                ENDIF
#endif 
                !
                !             CHECK FOR NEW SEGMENT BOUNDARY
                IF(NEWTRN) THEN
                   NEWTRN=.FALSE.
                   NSGIM=NSGIM+1
#ifdef KEY_RESIZE
                   IF(NSGIM.GT.size(segid)) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "Segment array size exceeded: ",nsgim
#else
                   IF(NSGIM.GT.MAXSEG) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,2i10)') &
                           "maxseg exceeded: ",maxseg,nsgim
#endif
                      CALL WRNDIE(-3,'<MKIMAT>','SEGMENT NUMBER OVERFLOW')
                   ENDIF
                   SEGID(NSGIM)=IMNAME(ITRANS)
                   NICTOT(NSGIM)=NRSIM
                ENDIF
                !
                !             CHECK FOR NEW RESIDUE BOUNDARY
                IF (I.GT.IBASE(IRES+1)) THEN
380                CONTINUE
                   IRES=IRES+1
                   IF (I.GT.IBASE(IRES+1)) GOTO 380
                ENDIF
                IF(IRES.GT.ILAST) THEN
                   ILAST=IRES
                   NRSIM=NRSIM+1
#ifdef KEY_RESIZE
                   IF(NRSIM.GT.size(resid)) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "Residue array size exceeded: ",nrsim
#else
                   IF(NRSIM.GT.MAXRES) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "maxres exceeded: ",maxres
#endif
                      CALL WRNDIE(-3,'<MKIMAT>','RESIDUE NUMBER OVERFLOW')
                   ENDIF
                   CHX=RES(IRES)
                   RES(NRSIM)=CHX
                   CHX=RESID(IRES)
                   RESID(NRSIM)=CHX
                   IBASE(NRSIM)=K-1
                ENDIF
                !
             ENDDO
             NATM=K
          ENDIF
       ENDDO
       !
700    CONTINUE
       IMATPT(ITRANS)=NATM
       NIMRES(ITRANS)=NRSIM
       IF (ITRANS.EQ.1) THEN
          I=NRSIM-NRES
       ELSE
          I=NRSIM-NIMRES(ITRANS-1)
       ENDIF
       IF(R2MINB.EQ.R2BIG) R2MINB=R2MINA
       IF(R2MINB.LT.R2BIG) THEN
          R2MINB=SQRT(R2MINB)
          IF(PRNLEV.GE.5) THEN
             WRITE(OUTU,45) ITRANS,IMNAME(ITRANS),NATM-NATMO,KRS-KRSO, &
                  I,R2MINB
45           FORMAT(I5,2X,A,' has',I8,I8,I8,F12.2)
          ENDIF
       ENDIF
       KRSO=KRS
       NATMO=NATM
    ENDDO
    !
    NATIM=NATM
    NIMGRP=KRS
    IBASE(NRSIM+1)=NATIM
    NICTOT(NSGIM+1)=NRSIM
    !
    IF(QDCM) CALL DCMSETIM(.FALSE.,NATIM) !UPDATE NATIM IN DCM ROUTINE
    IF(PRNLEV.GE.5) THEN
       if (NATIM.GE.100000 .OR. NIMGRP.GE.100000 .OR. NRSIM.GE.100000) &
            then
          WRITE(OUTU,56) NATIM,NIMGRP,NRSIM
       else
          WRITE(OUTU,55) NATIM,NIMGRP,NRSIM
       endif
    ENDIF
55  FORMAT(' Total of',I5,' atoms and',I5,' groups and',I5, &
         ' residues were included'/)
56  FORMAT(' Total of',I10,' atoms and',I10,' groups and',I10, &
         ' residues were included'/)

    !
    NATOMT=NATIM
    NGRPT=NIMGRP
    NREST=NRSIM
    NSEGT=NSGIM
    !
#if KEY_PARALLEL==1 /*parafin*/
    IF(NUMNOD.GT.1) THEN
       call chmdealloc('upimag.src','MKIMAT','IPKMAT',IPKMSZ,intg=IPKMAT)
    ENDIF
#endif /* (parafin)*/
    !
#if KEY_CHEQ==1
    if (qcg) then
       call allocate_cheq(natim,ngrp)
    endif
#endif 
    IF (TIMER.EQ.1) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,A)') 'TOTAL TIME IN MKIMAT'
       CALL TIMRE
       CALL TIMRB
    ENDIF
    !
    RETURN
  END SUBROUTINE MKIMAT
  
  SUBROUTINE MKIMAT2(X,Y,Z,WMAIN,CUTIM,NTRANS,NIMRES,NATIM,NIMGRP, &
#if KEY_PERT==1
       IGPERT,IGPTPP,IGPBSP,IPERT,            & 
#endif
#if KEY_PERT==1
       IACP,IACNBP,CGP,RSCLFP,                & 
#endif
       IMTRNS,NOROT,IMATPT,IMATTR,IMNAME, &
       LIMINV,LIMALL,IMINV,RSCMX,RSCMY,RSCMZ, &
       RSXMAX,RSYMAX,RSZMAX,R2MIN &
#if KEY_WCA==1
       ,WCAP                                   & 
#endif
       )
    !-----------------------------------------------------------------------
    !    Linearized image list builder, made partly from MKIMAT.
    !    The general scheme is to make two passes through the images.
    !    In the preliminary pass, the image groups that are within cutim
    !    of the box circumscribing the primary system are identified (as in
    !    MKIMAT), and the dimensions of the box circumscribing these "prelimary"
    !    image groups and the primary groups are then calculated.  The preliminary
    !    image groups are stored in a list, pointed to by the transformations.
    !    In the main pass, the system (box) is partitioned into cubes of side length
    !    cutim plus a margin length corresponding to the size of the largest group.
    !    For each prelim image group, each of the surrounding 27 nearest-
    !    neighbor cubes is checked to determine whether any of them contains primary
    !    atoms.  For each cube that does, distance checks are performed between the
    !    atoms it contains and the image group.  If the image group is within
    !    cutim of any primary atom, it is accepted for inclusion in the final
    !    image list.  In this way, the algorithm does distance checks only near
    !    the interface of the primary and image systems, rather than between
    !    all primary atoms and all image atoms near the primary system.
    !
    !    The portions of the code that are particular to BYCC and active atom
    !    selections are meant to be "protected" by the qnbacton flag.
    !
    !    Thanks to Bernie Brooks --RJP  5.15.08
    !
    !
    use actclus_mod,only:clnum,nactg,qnbacton,actvg,clusnum
#if KEY_LOOKUP==1
    use LOOKUP              
#endif
#if KEY_CHEQ==1
    use cheq,only:ech,eha,qcg,qpartbin,allocate_cheq   
#endif
    !
    use chm_kinds
    use dimens_fcm
    use number
    use psf
    use stream
    use pert
    use fast
    use timerm
    !EPO - variable LJ cutoff
    use varcutm
    !EPO
    use block_ltm
    use tbmts_ltm
#if KEY_SCCDFTB==1
    use sccdftb
    use gamess_fcm
#endif 
    use parallel
#if KEY_ASPENER==1
    use eef1_mod           
#endif
    !
#if KEY_PIPF==1
    use pipfm              
#endif
    use chutil,only:hydrog
    use machutil,only:timrb,timre
    use mtp_fcm
    use mtpl_fcm

    use dcm_fcm,only: QDCM,IMGINIT,INDQ,DCMSETIM

    implicit none
    !
    real(chm_real)  X(*),Y(*),Z(*),WMAIN(*)
    real(chm_real)  CUTIM
    INTEGER NTRANS,NATIM,NIMGRP,NIMRES(*),IMINV(*)
#if KEY_PERT==1
    INTEGER IGPERT(*),IGPTPP(*),IGPBSP(*),IPERT(*),IACP(*),IACNBP(*)
    real(chm_real) CGP(*),RSCLFP(*)
#endif 
#if KEY_WCA==1
    real(chm_real) WCAP(*)              
#endif
    real(chm_real) IMTRNS(*)
    LOGICAL NOROT,LIMALL,LIMINV
    INTEGER IMATPT(*),IMATTR(*)
    CHARACTER(len=*) IMNAME(*)
    real(chm_real)  RSCMX(*),RSCMY(*),RSCMZ(*)
    real(chm_real)  RSXMAX(*),RSYMAX(*),RSZMAX(*),R2MIN(*)
    !
    !
#if KEY_PARALLEL==1 /*pardecl*/
    INTEGER IMYNOD,VAL,JPT,KPT,IPKMAT,IPKMSZ,VMASK
    integer,allocatable,dimension(:) :: ACCVAL
#endif /* (pardecl)*/
    LOGICAL ACCEPT,NEWTRN
    !
    real(chm_real) XMAXP,YMAXP,ZMAXP,XMINP,YMINP,ZMINP
    real(chm_real) XMAX,YMAX,ZMAX,XMIN,YMIN,ZMIN,XD,YD,ZD
    real(chm_real) RSCXP,RSCYP,RSCZP,RSXMXP,RSYMXP,RSZMXP
    real(chm_real) RSCX,RSCY,RSCZ,RSXMX,RSYMX,RSZMX,XD1,YD1,ZD1
    real(chm_real) CTIMSQ,R2
    INTEGER I,J,K,IPT,ISTRT,ILAST,ITMP
    INTEGER IS,IQ,ITRANS,IRES,IRS,JRS
    INTEGER NAT,NATM,NATMO,KRS,KRSO
    INTEGER NSGIM,NRSIM
    CHARACTER(len=8) CHX
    real(chm_real) R2BIG
    DATA   R2BIG/9.99D12/
    ! *****************************************************************
    !     declarations for linearization
    integer,allocatable,dimension(:) :: trgrlist ! prelim-pass list of groups
    integer,allocatable,dimension(:) :: trgrptlo,trgrpthi !pointers to trgrlist
    integer :: grpcnt,grpend,grpstrt,ii,ierr,locsumat
    integer, save :: grpcntal !estimated length of trgrlist, for mem alloc
    logical,save :: q1stpas=.true. !only true at beginning of charmm run
    logical :: done
    logical :: checkall !true if all groups in primary systems must be
    ! checked (not half)
    real(kind=chm_real), parameter :: buffac=1.1 ! mem for trgrlist in next
    !call to MKIMAT2 = current length times this factor
    integer :: loctratm !local count of atoms in transformation
    !      real(kind=chm_real) :: glbrsxmx,glbrsxmy,glbrsxmz
    ! second loop
    integer :: numbx,numby,numbz  !size of box circumscribing entire sys
    integer :: totcub !total number of cubes in the system
    integer :: nxy,fldm
    integer :: locnactg  !the number of active groups
    integer :: cc,acc1,ac2,qq,ac !indices
    ! cube-based arrays:
    integer, allocatable, dimension(:) :: lstm0, cntn, mmm
    ! group based arrays:
    integer, allocatable, dimension(:) :: lstn0,ordcbn,locactg
    !
    integer :: x00,y00,z00,ix,iy,iz,ttx,tty,ttz
    ! dimensions of primary system (as defined by group centers)
    real(kind=chm_real) grxmin,grxmax,grymin,grymax,grzmin,grzmax
    ! dimensions of images (as defined by group centers)
    real(kind=chm_real) imxmin,imxmax,imymin,imymax,imzmin,imzmax
    ! dimensions of overall system
    real(kind=chm_real) ovxmin,ovxmax,ovymin,ovymax,ovzmin,ovzmax
    !
    real(kind=chm_real) rcutim
    real(kind=chm_real), allocatable, dimension(:) :: r2mina
    !      integer sumat !temporary
    ! for image clusters:
    integer :: clscnt,clus
    integer, allocatable, dimension (:) :: locclsflg
    real(kind=chm_real) :: glbrsmax
    integer, allocatable, dimension(:),save :: trcllist,trclshi, &
         trclslo
    logical :: qrepeat
    integer, parameter :: trgrmemmin=1000  !minimum size of trgrlist (main) work array
    !
    ! *****************************************************************
    !      end of declarations for MKIMAT2
    ! *****************************************************************
    !      start loop through primary system for geometry
    ! *****************************************************************
    !
    !mmpass = mmpass + 1 ! counts calls to this routine

    IF (TIMER.GT.0) CALL TIMRB
    IF(PRNLEV.GE.5) WRITE(OUTU,'(/,A)') &
         ' <MKIMAT2>: updating the image atom lists and remapping'
    IF(PRNLEV.GE.5) WRITE(OUTU,'(A)') &
                                !        ' Transformation   Atoms  Groups  Residues  Upper-Bound '
         ' Transformation   Atoms  Groups  Residues  Min-Distance'
    !
    !-----------initialize array for atomic multipoles--MTP-------------
    IF (QMTP) IMOL = MOLS
#if KEY_MTPL==1
    IF (QMTPL) IMOLL = MOLSL
#endif 

    IF (QDCM) THEN
     INDQ=0
    ENDIF

    !-------------------------------------------------------------------
    CTIMSQ=CUTIM*CUTIM
    !
    !     Set up group centers and sizes
    !
    XMAXP=X(1)
    XMINP=X(1)
    YMAXP=Y(1)
    YMINP=Y(1)
    ZMAXP=Z(1)
    ZMINP=Z(1)
    !
    grxmin = rbig
    grxmax = -rbig
    grymin = rbig
    grymax = -rbig
    grzmin = rbig
    grzmax = -rbig

    glbrsmax = -rbig  !largest x,y, or z width of any group
    !
    if (qnbacton) then
       locnactg = nactg
    else
       locnactg = ngrp
    endif
    ! allocate group-based arrays
    allocate(locactg(ngrp),ordcbn(ngrp),lstn0(ngrp),STAT=ierr)
    !
    if (ierr.gt.0) call wrndie(-5, &
         '<MKIMAT2>','group-based arrays not allocated')
    !
    DO I=1,NGRP
       ordcbn(i) = 0 !initialize
       lstn0(i) = 0 !initialize
       if(qnbacton) THEN !if active atom arrays filled
          if (i.le.locnactg) then
             locactg(i)=actvg(i) !only the first locnactg elements are valid
          else
             locactg(i)=0 !initialize
          endif
       else !active atom arrays are not filled
          locactg(i)=i
       endif
       IS=IGPBS(I)+1
       IQ=IGPBS(I+1)
       NAT=IQ-IS+1
       IF(NAT.LE.0) CALL WRNDIE(-5,'<MKIMAT2>', &
            'Group with no atoms found')
       XMIN=X(IS)
       XMAX=XMIN
       YMIN=Y(IS)
       YMAX=YMIN
       ZMIN=Z(IS)
       ZMAX=ZMIN
       DO J=IS+1,IQ
          IF(X(J).LT.XMIN) XMIN=X(J)
          IF(Y(J).LT.YMIN) YMIN=Y(J)
          IF(Z(J).LT.ZMIN) ZMIN=Z(J)
          IF(X(J).GT.XMAX) XMAX=X(J)
          IF(Y(J).GT.YMAX) YMAX=Y(J)
          IF(Z(J).GT.ZMAX) ZMAX=Z(J)
       ENDDO
       !
       !       Size of rectangular box surrounding group
       !
       RSXMAX(I)=(XMAX-XMIN)*0.5 ! half of x-width of group
       RSYMAX(I)=(YMAX-YMIN)*0.5 ! half of y-width
       RSZMAX(I)=(ZMAX-ZMIN)*0.5 ! half of z-width
       glbrsmax =max(glbrsmax,RSXMAX(I)) !largest dimension of
       glbrsmax =max(glbrsmax,RSYMAX(I))  !any group
       glbrsmax =max(glbrsmax,RSZMAX(I))
       !
       ! the global maxima for box sizes:
       !       CENTER OF GROUP
       RSCMX(I)=(XMAX+XMIN)*0.5
       RSCMY(I)=(YMAX+YMIN)*0.5
       RSCMZ(I)=(ZMAX+ZMIN)*0.5
       !       GLOBAL EXTREEMS
       XMAXP=MAX(XMAXP,XMAX)
       XMINP=MIN(XMINP,XMIN)
       YMAXP=MAX(YMAXP,YMAX)
       YMINP=MIN(YMINP,YMIN)
       ZMAXP=MAX(ZMAXP,ZMAX)
       ZMINP=MIN(ZMINP,ZMIN)
       ! extremes based on group coordinates
       grxmin = min(grxmin,RSCMX(I))
       grxmax = max(grxmax,RSCMX(I))
       grymin = min(grymin,RSCMY(I))
       grymax = max(grymax,RSCMY(I))
       grzmin = min(grzmin,RSCMZ(I))
       grzmax = max(grzmax,RSCMZ(I))
    ENDDO
    !
    !     Find spatial extent of primary space
    !
    RSCXP=(XMAXP+XMINP)*0.5 ! center of primary box
    RSCYP=(YMAXP+YMINP)*0.5 ! center of primary box
    RSCZP=(ZMAXP+ZMINP)*0.5 ! center of primary box
    RSXMXP=XMAXP-RSCXP ! 1/2 of x-width of primary box
    RSYMXP=YMAXP-RSCYP ! 1/2 of y-width of primary box
    RSZMXP=ZMAXP-RSCZP ! 1/2 of z-width of primary box
    !
    !***********************************************************************C
    ! preliminary loop through the system images starts here
    !***********************************************************************
    ! initialize bounding box around images
    !
    imxmin = rbig
    imxmax = -rbig
    imymin = rbig
    imymax = -rbig
    imzmin = rbig
    imzmax = -rbig
    !
    ! allocate transformation-group pointer arrays
    allocate(trgrptlo(ntrans),trgrpthi(ntrans),STAT=ierr)
    if (ierr.gt.0) call wrndie(-5, &
         '<MKIMAT2>','trgrptlo and hi arrays not allocated')
    ! allocate transformation-based arrays
    allocate(r2mina(ntrans),STAT=ierr)
    if (ierr.gt.0) call wrndie(-5, &
         '<MKIMAT2>','r2mina array not allocated')
    !
    done = .false.
    do while (.not. done)
       ! initialize arrays
       do itrans = 1,ntrans
          trgrpthi(itrans) = 0 !pointers into trgrlist
          trgrptlo(itrans) = 0
       enddo
       !
       qrepeat = .false.  !true if need to repeat loop for memory reasons
       ! if it the estimate of memory for the main work array is less than
       ! some prespecified minimum, set it to the minimum
       if(grpcntal.lt.trgrmemmin) grpcntal=trgrmemmin
       ! allocate work array memory
       if(.not.allocated(trgrlist)) allocate(trgrlist(grpcntal),stat=ierr)
       if(ierr.gt.0) &
            call wrndie(-5,'<MKIMAT2>','trgrlist arrays not allocated')
       grpcnt = 0 !reset counter for number of included image groups
       trgrptlo(1) = 1
       locsumat = 0 ! total atoms close to prim system in this pass
       !
       ! note that none of the transformations is the identity transformation--
       ! i.e. the primary system is not included in the (ITRANS = 1,NTRANS) loop
       DO ITRANS=1,NTRANS
          IPT=(ITRANS-1)*12
          r2mina(itrans)=R2BIG
          if (qnbacton) loctratm = 0
          IF (LIMALL) THEN
             CONTINUE
          ELSE IF (.NOT.LIMINV) THEN
             CONTINUE
          ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
             GOTO 400
          ENDIF
          !
          DO IRS=1,NGRP
             ! determine the number of atoms in each group
             IS=IGPBS(IRS)+1
             IQ=IGPBS(IRS+1)
             nat = IQ-IS+1
             IF(NOROT) THEN
                RSCX=RSCMX(IRS)+IMTRNS(IPT+10)
                RSCY=RSCMY(IRS)+IMTRNS(IPT+11)
                RSCZ=RSCMZ(IRS)+IMTRNS(IPT+12)
                RSXMX=RSXMAX(IRS)
                RSYMX=RSYMAX(IRS)
                RSZMX=RSZMAX(IRS)
             ELSE
                RSCX=RSCMX(IRS)*IMTRNS(IPT+1)+RSCMY(IRS)*IMTRNS(IPT+2)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                RSCY=RSCMX(IRS)*IMTRNS(IPT+4)+RSCMY(IRS)*IMTRNS(IPT+5)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                RSCZ=RSCMX(IRS)*IMTRNS(IPT+7)+RSCMY(IRS)*IMTRNS(IPT+8)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
                RSXMX=MAX(RSXMAX(IRS),RSYMAX(IRS),RSZMAX(IRS))
                RSYMX=RSXMX
                RSZMX=RSXMX
             ENDIF
             !
             ! Now check if this image group is close enough to primary atoms
             !
             XD=MAX(ABS(RSCX-RSCXP)-RSXMX-RSXMXP,ZERO)
             YD=MAX(ABS(RSCY-RSCYP)-RSYMX-RSYMXP,ZERO)
             ZD=MAX(ABS(RSCZ-RSCZP)-RSZMX-RSZMXP,ZERO)
             R2=XD*XD+YD*YD+ZD*ZD
             IF(R2.LT.r2mina(itrans)) r2mina(itrans)=R2
             IF(R2.GE.CTIMSQ) GOTO 300
             !
             ! Now check to see if this group is its own image
             !
             XD=RSCX-RSCMX(IRS)
             YD=RSCY-RSCMY(IRS)
             ZD=RSCZ-RSCMZ(IRS)
             R2=XD*XD+YD*YD+ZD*ZD
             IF(R2.LT.PT001) THEN
                IF(PRNLEV.GT.7) WRITE(OUTU,265) IRS,IMNAME(ITRANS)
265             FORMAT(' MKIMAT2: Group',I6, &
                     ' excluded in transformation "', &
                     A,'" because it is its own image.')
                GOTO 300
             ENDIF
             !
             grpcnt = grpcnt + 1
             if (qnbacton) loctratm = loctratm + nat
             if(grpcnt.le.grpcntal) then !ifn over allocd space
                trgrlist(grpcnt) = irs
                ! global extremes of image group positions
                imxmin = min(imxmin,RSCX)
                imymin = min(imymin,RSCY)
                imzmin = min(imzmin,RSCZ)
                imxmax = max(imxmax,RSCX)
                imymax = max(imymax,RSCY)
                imzmax = max(imzmax,RSCZ)
                !
             else !not enough memory allocated.  record that.
                qrepeat = .true.
             endif
300          CONTINUE
          ENDDO !loop over groups
400       trgrpthi(itrans) = grpcnt
          if (itrans .gt. 1) trgrptlo(itrans) = trgrpthi(itrans-1) + 1
          if (qnbacton) then
             locsumat = locsumat + loctratm  !add loc # atms to tot for pass
          endif
       ENDDO !loop over transformations
       grpcntal = int(grpcnt*buffac)  !size of trgrlist mem to be allocated next
       if (.not.qrepeat) then
          done = .true.
       else
          ! (at this point trgrlist should always have been allocated)
          if (allocated(trgrlist)) deallocate(trgrlist)
          !           IF(PRNLEV.GE.5) WRITE(OUTU,'(/,A)') &
          !        ' <MKIMAT2>: repeating the image list build with more memory'
       endif
500    continue
    enddo !do while .not. done
    !
    ! size of overall box is determined by max/min of both primary box
    ! and box circumscribing images (as defined by group centers).
    ! add +/- 1 ang to each extreme to avoid any problems caused by
    ! by rounding errors (e.g. if system is transformed)
    !
    ovxmin = min(imxmin,grxmin) - 1
    ovymin = min(imymin,grymin) - 1
    ovzmin = min(imzmin,grzmin) - 1
    ovxmax = max(imxmax,grxmax) + 1
    ovymax = max(imymax,grymax) + 1
    ovzmax = max(imzmax,grzmax) + 1
    !     real cutim
    rcutim = cutim + glbrsmax*2 ! accnt for finite width of groups
    ! **********************************************************************
    ! end of preliminary loop section, beginning of binning section
    ! **********************************************************************
    ! bin the groups of the primary system into cubes.
    ! first calculate the size of the overall grid.
    !
    !    find number of cubes necessary in each direction
    numbx = int((ovxmax-ovxmin)/rcutim) + 1
    numby = int((ovymax-ovymin)/rcutim) + 1
    numbz = int((ovzmax-ovzmin)/rcutim) + 1
    totcub = numbx*numby*numbz
    !
    nxy = numbx*numby
    ! allocate cubic arrays
    allocate(lstm0(totcub),cntn(totcub),   &
         mmm(totcub),stat=ierr)
    if (ierr.gt.0) call wrndie(-5, &
         '<MKIMAT2>','cube-based arrays not allocated')
    !
    ! ---initialize cube arrays
    do ii = 1,totcub
       lstm0(ii) = 0
       cntn(ii) = 0
       mmm(ii) =0
    enddo
    !
    ! check to see which cubes are "filled" (non-empty) in the
    ! primary system only (image groups need not be binned) and
    ! for the filled ones, store the number of groups they
    ! contain. Store the total number of filled cubes (FLDM)
    !
    fldm = 0
    do ii = 1,locnactg
       ac = locactg(ii)
       x00 = int((RSCMX(ac)-ovxmin)/rcutim)
       y00 = int((RSCMY(ac)-ovymin)/rcutim)
       z00 = int((RSCMZ(ac)-ovzmin)/rcutim)
       ordcbn(ac) = x00 + numbx*y00 + nxy*z00 + 1
       cc = ordcbn(ac)
       if (cc.gt.totcub) then
          write(6,*) 'cube overflow'
          write(6,*) 'cc is ',cc,' totcub ',totcub
       endif
       if (cntn(cc).eq.0) THEN
          fldm = fldm + 1
          mmm(fldm) = cc
          !  Note:  MMM not sorted
       endif
       cntn(cc) = cntn(cc) + 1
    enddo
    !
    !    Store the groups in each cube in a linked list.
    !    For each cube, the highest-numbered group contained
    !    in the cube is stored (in LSTM0()).  This group is
    !    also linked to the next highest group in number
    !    contained in the cube, such that  LSTN0(high) =
    !    next highest. This is done for all groups in a given
    !    cube.  Hence , the groups within the same cube are
    !    linked to each other in a "chain" and
    !    they are all linked to the cube number via
    !    the high group number. Linked list.
    do ii = 1, locnactg
       acc1 = locactg(ii)
       cc = ordcbn(acc1)
       lstn0(acc1) = lstm0(cc)
       lstm0(cc) = acc1
    enddo
    !****************************************************************
    ! end of binning section
    ! beginning of main section for image atom generation
    !****************************************************************
    ! For each (transformed) group--determine what cube
    ! it is in and then check the surrounding 27 cubes to
    ! see if there are any real (primary) atoms present.
    ! If there are, do a distance tests.
    ! Image list, by transformation number, is thus produced.
    !
    !*************************************************************************
    !*************************************************************************
    if (qnbacton) then
       !    allocate space for local cluster flag array (for use with BYCC)
       allocate(locclsflg(clnum)) !alloc space for flagging all clusters
       !
       if(allocated(trcllist)) deallocate(trcllist,trclshi,trclslo)
       ! need to know how much memory to allocate here to trcllist--
       ! can use the total number of atoms in grpcnt groups (locsumat),
       ! since # of clusters cannot be > this number.
       allocate(trcllist(locsumat),trclshi(ntrans),trclslo(ntrans),  &
            STAT=ierr)
       if (ierr.gt.0) then
          call wrndie(-5, &
               '<MKIMAT2>','tranf-cluster list/pointers not allocated')
       endif
       ! initialize transformation-cluster pointer arrays
       do itrans = 1,ntrans
          trclshi(itrans) = 0  !from image module
          trclslo(itrans) = 0  !from image module
       enddo
       clscnt = 0  !active cluster count
       trclslo(1) = 1
    endif
    !*************************************************************************
    !*************************************************************************
    !
    NSGIM=NSEG
    NRSIM=NRES
    !
    KRS=NGRP
    NATM=NATOM
    KRSO=KRS
    NATMO=NATM
    !
    checkall = .false.
    !*************************************************************************
    ! ***************** Main loop starts here ********************************
    !*************************************************************************
#if KEY_PARALLEL==1 /*parainit*/
    !     If compiled with PARALLEL and there is more than one node,
    !     setup a packed list.
    !
    KPT=0
    VAL=0
    IPKMSZ=NTRANS*NGRP/30+5
    allocate(ACCVAL(IPKMSZ),STAT=ierr)
    JPT = 1

#endif /* (parainit)*/

    do itrans=1,ntrans
       if (qnbacton) then
          ! initialize the cluster flag array for this transf
          do ii = 1,clnum
             locclsflg(ii) = 0
          enddo
       endif
       NEWTRN=.TRUE.
       ILAST=0
       IRES=0
       IPT=(ITRANS-1)*12
       IF (LIMALL) THEN
          checkall=.true.
       ELSE IF (.NOT.LIMINV) THEN
          checkall=.false.
       ELSE IF (IMINV(ITRANS).EQ.ITRANS) THEN
          checkall=.false.
       ELSE IF (IMINV(ITRANS).GT.ITRANS) THEN
          checkall=.true.
       ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
#if KEY_PARALLEL==1
          GOTO 650
#else /**/
          GOTO 700
#endif 
       ENDIF
       !
       grpstrt = trgrptlo(itrans)
       grpend =  trgrpthi(itrans)

       !
       ! decision was made here to recalculate the image coords, rather than
       ! store them in an array in preliminary loop, for memory reasons
       !
       do ii=grpstrt,grpend
          IRS = trgrlist(ii)
          IS=IGPBS(IRS)+1 !temporary
          IQ=IGPBS(IRS+1) !temporary
          NAT = IQ-IS+1 !temporary
          ACCEPT=.FALSE.
#if KEY_PARALLEL==1
          IF(MYNOD.EQ.MOD(ITRANS+IRS,NUMNOD)) THEN 
#endif
             IF(NOROT) THEN
                RSCX=RSCMX(IRS)+IMTRNS(IPT+10)
                RSCY=RSCMY(IRS)+IMTRNS(IPT+11)
                RSCZ=RSCMZ(IRS)+IMTRNS(IPT+12)
                RSXMX=RSXMAX(IRS)
                RSYMX=RSYMAX(IRS)
                RSZMX=RSZMAX(IRS)
             ELSE
                RSCX=RSCMX(IRS)*IMTRNS(IPT+1)+RSCMY(IRS)*IMTRNS(IPT+2)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+3)+IMTRNS(IPT+10)
                RSCY=RSCMX(IRS)*IMTRNS(IPT+4)+RSCMY(IRS)*IMTRNS(IPT+5)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+6)+IMTRNS(IPT+11)
                RSCZ=RSCMX(IRS)*IMTRNS(IPT+7)+RSCMY(IRS)*IMTRNS(IPT+8)+ &
                     RSCMZ(IRS)*IMTRNS(IPT+9)+IMTRNS(IPT+12)
                RSXMX=MAX(RSXMAX(IRS),RSYMAX(IRS),RSZMAX(IRS))
                RSYMX=RSXMX
                RSZMX=RSXMX
             ENDIF
             !
             ! find cube coordinates of the image group
             x00 = int((RSCX-ovxmin)/rcutim) !(these are "zeros"
             y00 = int((RSCY-ovymin)/rcutim) !not "ohs")
             z00 = int((RSCZ-ovzmin)/rcutim)
             ! loops over the 27 surrounding cubes (including central cube)
             do ix = -1,1
                ttx = x00+ix
                if ((ttx.ge.0).and.(ttx.lt.numbx)) then
                   do iy = -1,1
                      tty = y00+iy
                      if ((tty.ge.0).and.(tty.lt.numby)) then
                         do iz = -1,1
                            ttz = z00+iz
                            if ((ttz.ge.0).and.(ttz.lt.numbz)) then
                               cc = ttx + tty*numbx + ttz*nxy + 1
                               qq = 1
                               ac2 = lstm0(cc)
                               do while (qq.le.cntn(cc))
                                  ! need to take care of all the different cases here
                                  if((ac2.ge.irs).or.(checkall)) then
                                     !
                                     XD1 = MAX(ABS(RSCX - RSCMX(ac2)) &
                                          -RSXMX-RSXMAX(ac2),ZERO)
                                     YD1 = MAX(ABS(RSCY - RSCMY(ac2)) &
                                          -RSYMX-RSYMAX(ac2),ZERO)
                                     ZD1 = MAX(ABS(RSCZ - RSCMZ(ac2)) &
                                          -RSZMX-RSZMAX(ac2),ZERO)
                                     R2 = XD1*XD1 + YD1*YD1 + ZD1*ZD1 + &
                                          IMOVEG(IRS)*IMOVEG(AC2)*CTIMSQ
                                     IF(R2.LT.CTIMSQ) THEN
                                        ACCEPT = .TRUE.
                                        GOTO 600
                                     ENDIF
                                  endif                  !if doing distance check
                                  ac2 = lstn0(ac2)
                                  qq = qq + 1
                               enddo !loop over all primary groups in cube
                            endif !if cube z-coordinate not out of bounds
                         enddo ! loop over z-coordinates of surrounding cubes
                      endif !if cube y-coordinate not out of bounds
                   enddo ! loop over y-coors
                endif !if cube x-coordinate not out of bounds
             enddo ! loop over x-coors
             !
             !***************************************************************
             ! image group included here only if it is within cutim of an active
             ! group (or of any group if no active groups selected [.not. qnbacton]).
             !
#if KEY_PARALLEL==1
          endif !if for mynode  
#endif
600       CONTINUE
#if KEY_PARALLEL==1 /*paraclose*/
          KPT=KPT+1
          VAL=VAL+VAL
          IF(ACCEPT) VAL=VAL+1
          IF(KPT.EQ.30) THEN
             ACCVAL(JPT)=VAL
             VAL=0
             JPT=JPT+1
             KPT=0
          ENDIF
       enddo !loop over groups
       !
650    CONTINUE
    enddo !loop over transformations
    !
670 CONTINUE
    KPT=KPT+1
    VAL=VAL+VAL
    IF(KPT.LT.30) GOTO 670
    ACCVAL(JPT)=VAL
    !
    !    Now do a global OR operation on all of the packed lists.
    !
    CALL GBOR(ACCVAL,JPT)
    KPT=0
    JPT = 1
    VAL=ACCVAL(JPT)
    !   use 30 bits to a word
    VMASK=2**29
    !
    !  need to repeat start of main loop
    !
    NSGIM=NSEG
    NRSIM=NRES
    !
    KRS=NGRP
    NATM=NATOM
    KRSO=KRS
    NATMO=NATM
    !
    checkall = .false.
    ! ***************** Main loop starts here ********************************

    do itrans=1,ntrans
       if (qnbacton) then
          ! initialize the cluster flag array for this transf
          do ii = 1,clnum
             locclsflg(ii) = 0
          enddo
       endif
       NEWTRN=.TRUE.
       ILAST=0
       IRES=0
       IPT=(ITRANS-1)*12
       IF (LIMALL) THEN
          checkall=.true.
       ELSE IF (.NOT.LIMINV) THEN
          checkall=.false.
       ELSE IF (IMINV(ITRANS).EQ.ITRANS) THEN
          checkall=.false.
       ELSE IF (IMINV(ITRANS).GT.ITRANS) THEN
          checkall=.true.
       ELSE IF (IMINV(ITRANS).LT.ITRANS) THEN
          GOTO 700
       ENDIF
       !
       grpstrt = trgrptlo(itrans)
       grpend =  trgrpthi(itrans)
       !
       do ii=grpstrt,grpend
          IRS = trgrlist(ii)
          !
          ACCEPT=(VAL.GE.VMASK)
          IF(ACCEPT) VAL=VAL-VMASK
          VAL=VAL+VAL
          KPT=KPT+1
          IF(KPT.EQ.30) THEN
             JPT=JPT+1
             VAL=ACCVAL(JPT)
             KPT=0
          ENDIF
#endif /* (paraclose)*/
          !
          IF (ACCEPT) THEN
             !
             IS=IGPBS(IRS)+1
             IQ=IGPBS(IRS+1)
             !
             !         Now add this image group in if it has been accepted
             !
             KRS=KRS+1
#ifdef KEY_RESIZE
             IF(KRS.GT.size(imoveg)) THEN
                IF(WRNLEV.GE.2) WRITE(OUTU,35) ITRANS,IRS,KRS
35              FORMAT(//' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT2'/ &
                     ' ITRANS=',I5,' IRS=',I5,' KRS=',I5)
#else
             IF(KRS.GT.MAXGRP) THEN
                IF(WRNLEV.GE.2) WRITE(OUTU,35) ITRANS,IRS,KRS,MAXGRP
35              FORMAT(//' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT2'/ &
                     ' ITRANS=',I5,' IRS=',I5,' KRS=',I5,' MAXGRP=',I5)
#endif
                CALL WRNDIE(-5,'<MKIMAT2>', &
                     ' OVERFLOW ON NUMBER OF IMAGE GROUPS IN MKIMAT2')
             ENDIF
             !
             IMOVEG(KRS)=IMOVEG(IRS)
             IGPTYP(KRS)=IGPTYP(IRS)
             IGPBS(KRS+1)=IQ-IS+1+NATM
#if KEY_PERT==1
             IF(QPERT) THEN
                IGPERT(KRS)=IGPERT(IRS)
                IGPTPP(KRS)=IGPTPP(IRS)
                IGPBSP(KRS+1)=IQ-IS+1+NATM
             ENDIF
#endif 
             !-----------addition loop for atomic multipoles--MTP-------
             IF (QMTP) CALL IMTP(NATM,IS,IQ)
#if KEY_MTPL==1
             IF (QMTPL) CALL IMTPL(NATM,IS,IQ)
#endif 

             IF (QDCM) CALL IMGINIT(NATM,IS,IQ)

             !----------------------------------------------------------
             !
             K=NATM
             DO I=IS,IQ
                !             IF ((.NOT.QNBACTON).OR.(ACAFLG(I).EQ.1)) THEN !if active atms not used,
                !                                                         ! or atm is active
                ! flag active image clusters
                if (qnbacton) locclsflg(clusnum(i)) = 1
                !
                K=K+1
#ifdef KEY_RESIZE
                IF(K.GT.size(cg)) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,25) ITRANS,IRS,IQ,IS
25                 FORMAT(/' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT2'/ &
                        ' ITRANS=',I5,' IRS=',I5,' IQ=',I5,' IS=',I5)
#else
                IF(K.GT.MAXAIM) THEN
                   IF(WRNLEV.GE.2) WRITE(OUTU,25) ITRANS,IRS,IQ,IS,MAXAIM
25                 FORMAT(/' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT2'/ &
                        ' ITRANS=',I5,' IRS=',I5,' IQ=',I5,' IS=',I5,' MAXAIM=',I5)
#endif
                   CALL WRNDIE(-5,'<MKIMAT2>', &
                        ' OVERFLOW ON NUMBER OF IMAGE ATOMS IN MKIMAT2')
                ENDIF
                !
                !             Copy other arrays dimensioned NATOM
                !
                IMATTR(K)=I
                CHX=ATYPE(I)
                ATYPE(K)=CHX
                CG(K)=CG(I)
#if KEY_LJPME==1
                LJC6(K)=LJC6(I)
#endif
                AMASS(K)=AMASS(I)
                ICAAUTO(K)=ICAAUTO(I)
                if(qvarcut) VARCUT(K)=VARCUT(I)
                ! Drude force field stuff
                ISDRUDE(K)=ISDRUDE(I)
                ALPHADP(K)=ALPHADP(I)
                THOLEI(K)=THOLEI(I)
#if KEY_CHEQ==1
                if(qcg) then
                   ECH(K)=ECH(I)
                   EHA(K)=EHA(I)
                endif
#endif 
#if KEY_PIPF==1
                ! PJ 06/2005
                IF (QPIPF .AND. QPFDYN) THEN
                   !yw                  CALL UINDASG(K,I,IUINDSV)
                   CALL UINDASG(K,I,UIND)
                ENDIF
#endif 
                IAC(K)=IAC(I)
                IACNB(K)=IACNB(I)
                IMOVE(K)=IMOVE(I)
                RSCLF(K)=RSCLF(I)
                WMAIN(K)=WMAIN(I)
#if KEY_LOOKUP==1
                !LNI flag solvent atoms selected for nb lookup
                !CCCC How about lone-pairs and TIP4P????
                IF(QVV.OR.QVU) THEN
                   ITMP=IWWFLG(I)
                   IF(ITMP.GT.0)THEN
                      IF(.NOT.HYDROG(I))THEN
                         IWWFLG(K)=K
                      ELSE IF(.NOT.HYDROG(I-1))THEN
                         IWWFLG(K)=K-1
                      ELSE IF(.NOT.HYDROG(I-2))THEN
                         IWWFLG(K)=K-2
                      ENDIF
                   ELSE
                      IWWFLG(K)=0
                   ENDIF
                ENDIF
#endif 
#if KEY_WCA==1
                WCA(K)=WCA(I)        
#endif
#if KEY_SCCDFTB==1
                if(qmused)IGMSEL(K)=IGMSEL(I)  
#endif

#if KEY_ASPENER==1
                if(doeef1) then
                   GFREEI(K)=GFREEI(I)
                   VOLMI(K)=VOLMI(I)
                   SIGWI(K)=SIGWI(I)
                   FSIGWI(K)=FSIGWI(I)
                   SIGWI2(K)=SIGWI2(I)
                   FSIGWI2(K)=FSIGWI2(I)
                endif
#endif 
                !
#if KEY_MTS==1
                IF (QTBMTS) THEN
                   IMTS(K)=IMTS(I)
                   IF(TBHY1) THEN
                      IMTF(K)=IMTF(I)
                   ENDIF
                ENDIF
#endif 
                !
#if KEY_PERT==1
                IF(QPERT) THEN
                   IPERT(K)=IPERT(I)
                   CGP(K)=CGP(I)
                   IACP(K)=IACP(I)
                   IACNBP(K)=IACNBP(I)
                   RSCLFP(K)=RSCLFP(I)
#if KEY_WCA==1
                   WCAP(K)=WCAP(I)   
#endif

                   PPCG(K)=PPCG(I)
                   PPALPHA(K)=PPALPHA(I)
                   PPTHOLE(K)=PPTHOLE(I)
                   PPAMASS(K)=PPAMASS(I)
                ENDIF
#endif 
#if KEY_BLOCK==1
                !SBblock: copy content of IBLOCK/IBLCKP array of real atoms
                !         to respective image atoms.  This code is crude as
                !         it accesses the hp directly.  However, the only
                !         alternative would be to change the calling sequence of
                !         MKIMAT2.  BTW, using the function I4VAL doesn't really
                !         help and seems to just adding another function call.
                !         Don't know whether this runs on a 64bit machine; maybe
                !         I4VAL has to be used there after all...
                IF(QBLOCK) THEN
                   IBLCKP(K-1+1) = IBLCKP(I-1+1)
                ENDIF
#endif 
                !
                !             CHECK FOR NEW SEGMENT BOUNDARY
                IF(NEWTRN) THEN
                   NEWTRN=.FALSE.
                   NSGIM=NSGIM+1
#ifdef KEY_RESIZE
                   IF(NSGIM.GT.size(segid)) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "Segment array size exceeded: ",nsgim
#else
                   IF(NSGIM.GT.MAXSEG) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,2i10)') &
                           "maxseg exceeded: ",maxseg,nsgim
#endif
                      CALL WRNDIE(-3,'<MKIMAT2>','SEGMENT NUMBER OVERFLOW')
                   ENDIF
                   SEGID(NSGIM)=IMNAME(ITRANS)
                   NICTOT(NSGIM)=NRSIM
                ENDIF
                !
                !             CHECK FOR NEW RESIDUE BOUNDARY
                IF (I.GT.IBASE(IRES+1)) THEN
680                CONTINUE
                   IRES=IRES+1
                   IF (I.GT.IBASE(IRES+1)) GOTO 680
                ENDIF
                IF(IRES.GT.ILAST) THEN
                   ILAST=IRES
                   NRSIM=NRSIM+1
#ifdef KEY_RESIZE
                   IF(NRSIM.GT.size(resid)) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "Residue array size exceeded: ",nrsim
#else
                   IF(NRSIM.GT.MAXRES) THEN
                      IF(WRNLEV.GE.2) WRITE(OUTU,'(a,i10)') &
                           "maxres exceeded: ",maxres
#endif
                      CALL WRNDIE(-3,'<MKIMAT2>','RESIDUE NUMBER OVERFLOW')
                   ENDIF
                   CHX=RES(IRES)
                   RES(NRSIM)=CHX
                   CHX=RESID(IRES)
                   RESID(NRSIM)=CHX
                   IBASE(NRSIM)=K-1
                ENDIF
                !             ENDIF !if atom is active, or no active atoms
             ENDDO !loop over atoms in groups
             NATM=K
          ENDIF !if accepted group
       ENDDO !loop over groups
       !
700    CONTINUE
       IMATPT(ITRANS)=NATM !pointer to the number of image atoms
       ! associated with that transformation
       NIMRES(ITRANS)=NRSIM
       IF (ITRANS.EQ.1) THEN
          I=NRSIM-NRES
       ELSE
          I=NRSIM-NIMRES(ITRANS-1)
       ENDIF
       IF(r2mina(itrans).LT.R2BIG) then
          IF(PRNLEV.GE.5) THEN
             WRITE(OUTU,45) ITRANS,IMNAME(ITRANS),NATM-NATMO,KRS-KRSO, &
                  I,sqrt(r2mina(itrans))
45           FORMAT(I5,2X,A,' has',I8,I8,I8,F12.2)
          ENDIF
       ENDIF
       KRSO=KRS
       NATMO=NATM
       ! compile the transformation-cluster list
       if (qnbacton) then
          do ii = 1, clnum
             if (locclsflg(ii) .eq. 1) then
                clscnt = clscnt + 1
                trcllist(clscnt) = ii
             endif
          enddo
          trclshi(itrans) = clscnt
          if (itrans.gt.1) trclslo(itrans) = trclshi(itrans-1) + 1
       endif
       !
    ENDDO !loop over transformations
    !  temporary:
    !      if(qnbacton) then
    !       DO itrans = 1,ntrans
    !         write(6,*) 'FOR TRANSFORMATION ',ITRANS,' the clusters are:'
    !         write(6,*) 'start is ',trclslo(itrans)
    !         write(6,*) 'end is ',trclshi(itrans)
    !         do ii = trclslo(itrans), trclshi(itrans)
    !          clus = trcllist(ii)
    !          write(6,*) ii, clus
    !         enddo
    !       ENDDO
    !      endif
    !
    NATIM=NATM
    NIMGRP=KRS
    IBASE(NRSIM+1)=NATIM
    NICTOT(NSGIM+1)=NRSIM
    !
    IF(QDCM) CALL DCMSETIM(.FALSE.,NATIM) !UPDATE NATIM IN DCM ROUTINE
    IF(PRNLEV.GE.5) THEN
       if (NATIM.GE.100000 .OR. NIMGRP.GE.100000 .OR. NRSIM.GE.100000) &
            then
          WRITE(OUTU,56) NATIM,NIMGRP,NRSIM
       else
          WRITE(OUTU,55) NATIM,NIMGRP,NRSIM
       endif
    ENDIF

55  FORMAT(' Total of',I5,' atoms and',I5,' groups and',I5, &
         ' residues were included'/)
56  FORMAT(' Total of',I10,' atoms and',I10,' groups and',I10, &
         ' residues were included'/)

    !
    NATOMT=NATIM
    NGRPT=NIMGRP
    NREST=NRSIM
    NSEGT=NSGIM
    !
#if KEY_CHEQ==1
    if (qcg) then
       call allocate_cheq(natim,ngrp)
    endif
#endif 
    IF (TIMER.EQ.1) THEN
       IF(PRNLEV.GE.2) WRITE(OUTU,'(1X,A)') 'TOTAL TIME IN MKIMAT2'
       CALL TIMRE
       CALL TIMRB
    ENDIF
    ! deallocate local subroutine arrays
    deallocate(trgrlist,trgrptlo,trgrpthi)
    deallocate(lstm0,cntn,mmm)
    deallocate(locactg,ordcbn,lstn0)
    deallocate(r2mina)
    if (qnbacton) deallocate(locclsflg)
    !
    RETURN
  END SUBROUTINE MKIMAT2
    
  SUBROUTINE IMMAP(NTRANS,IMATPT,IMATTR,JMATPT,JMATTR,ITR,NIMHB, &
       NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP,NIMEXCL,IMEXCLI,IMEXCLJ,IMEXCLT, &
#if KEY_CMAP==1
       NIMCRT, &
#endif 
       LENIC,IAR,JAR,KAR,LAR)
    !-----------------------------------------------------------------------
    !
    !     THIS ROUTINE REMAPS ALL HBOND, AND INTERNAL ENERGY ATOM POINTERS
    !     INVOLVING IMAGE TO PRIMARY ATOM INTERACTIONS.
    !     THIS IS REQUIRED WHEN GENERATING A NONBOND LIST AND DIFFERENT IMAGE
    !     HAVE BEEN SELECTED BASED ON DISTANCE AND ORDERING REQUIREMENTS.
    !
    !     By Bernard R. Brooks    1983
    !
    use chm_kinds
    use dimens_fcm
    use psf
    use hbondm
    use stream
    !
    implicit none
    !
    INTEGER NTRANS,NIMHB,NIMBON,NIMANG,NIMDIH,NIMAUTO,NIMIMP
#if KEY_CMAP==1
    INTEGER NIMCRT
#endif 
    INTEGER IMATPT(*),IMATTR(*)
    INTEGER JMATPT(*),JMATTR(*)
    INTEGER ITR(*)
    INTEGER LENIC,IAR(*),JAR(*),KAR(*),LAR(*)
    INTEGER NIMEXCL,IMEXCLI(*),IMEXCLJ(*),IMEXCLT(*)
    !
    LOGICAL DROP
    INTEGER NDROP,IR,ITRANS,I,J,ITEMP,JTEMP,IPT
    !
    DO I=1,NATOM
       ITR(I)=I
    ENDDO

    ITEMP=NATOM
    JTEMP=NATOM
    DO ITRANS=1,NTRANS
       DO IR=1,NATOM
          IF (JTEMP.LT.JMATPT(ITRANS)) THEN
             J=JMATTR(JTEMP+1)
          ELSE
             J=0
          ENDIF
          IF (ITEMP.LT.IMATPT(ITRANS)) THEN
             I=IMATTR(ITEMP+1)
          ELSE
             I=0
          ENDIF
          IF (IR.EQ.J .AND. IR.EQ.I) THEN
             ITEMP=ITEMP+1
             JTEMP=JTEMP+1
             ITR(JTEMP)=ITEMP
          ELSE IF (IR.EQ.J) THEN
             JTEMP=JTEMP+1
             ITR(JTEMP)=0
          ELSE IF (IR.EQ.I) THEN
             ITEMP=ITEMP+1
          ENDIF
       ENDDO
    ENDDO
    !
    !     NOW MAPPING IS COMLETE. TRANSLATE ITEMS
    !
    !     DO HBONDS
    !
    NDROP=0
    DO I=1,NIMHB
       J=I+NHB
       DROP=.FALSE.
       IF(IHB(J).GT.0) THEN
          IHB(J)=ITR(IHB(J))
          IF(IHB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JHB(J).GT.0) THEN
          JHB(J)=ITR(JHB(J))
          IF(JHB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KHB(J).GT.0) THEN
          KHB(J)=ITR(KHB(J))
          IF(KHB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(LHB(J).GT.0) THEN
          LHB(J)=ITR(LHB(J))
          IF(LHB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IHB(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,33) NDROP
33  FORMAT(' ***',I5,' IMAGE HYDROGEN BONDS WERE DROPPED IN ', &
         'REMAPPING ***')
    !
    !     DO BONDS
    !
    NDROP=0
    DO I=1,NIMBON
       J=I+NBOND
       DROP=.FALSE.
       IF(IB(J).GT.0) THEN
          IB(J)=ITR(IB(J))
          IF(IB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JB(J).GT.0) THEN
          JB(J)=ITR(JB(J))
          IF(JB(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IB(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,44) NDROP
44  FORMAT(' ***',I5,' IMAGE BONDS WERE DROPPED IN ', &
         'REMAPPING ***')
    !
    !     DO ANGLES
    !
    NDROP=0
    DO I=1,NIMANG
       J=I+NTHETA
       DROP=.FALSE.
       IF(IT(J).GT.0) THEN
          IT(J)=ITR(IT(J))
          IF(IT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JT(J).GT.0) THEN
          JT(J)=ITR(JT(J))
          IF(JT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KT(J).GT.0) THEN
          KT(J)=ITR(KT(J))
          IF(KT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IT(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,55) NDROP
55  FORMAT(' ***',I5,' IMAGE ANGLES WERE DROPPED IN ', &
         'REMAPPING ***')
    !
    !     DO DIHEDRALS
    !
    NDROP=0
    DO I=1,NIMDIH
       J=I+NPHI
       DROP=.FALSE.
       IF(IP(J).GT.0) THEN
          IP(J)=ITR(IP(J))
          IF(IP(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JP(J).GT.0) THEN
          JP(J)=ITR(JP(J))
          IF(JP(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KP(J).GT.0) THEN
          KP(J)=ITR(KP(J))
          IF(KP(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(LP(J).GT.0) THEN
          LP(J)=ITR(LP(J))
          IF(LP(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IP(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,66) NDROP
66  FORMAT(' ***',I5,' IMAGE DIHEDRALS WERE DROPPED IN ', &
         'REMAPPING ***')

    !
    !     DO AUTOGEN ANGLES AND DIHEDRALS
    !
    NDROP=0
    DO I=1,NIMAUTO
       J=I+NPAUTO
       DROP=.FALSE.
       IF(IPAUTO(J).GT.0) THEN
          IPAUTO(J)=ITR(IPAUTO(J))
          IF(IPAUTO(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JPAUTO(J).GT.0) THEN
          JPAUTO(J)=ITR(JPAUTO(J))
          IF(JPAUTO(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KPAUTO(J).GT.0) THEN
          KPAUTO(J)=ITR(KPAUTO(J))
          IF(KPAUTO(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(LPAUTO(J).GT.0) THEN
          LPAUTO(J)=ITR(LPAUTO(J))
          IF(LPAUTO(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IPAUTO(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,67) NDROP
67  FORMAT(' ***',I5,' IMAGE AUTOGEN TERMS WERE DROPPED IN ', &
         'REMAPPING ***')
    !
    !     DO IMPROPER DIHEDRALS
    !
    NDROP=0
    DO I=1,NIMIMP
       J=I+NIMPHI
       DROP=.FALSE.
       IF(IM(J).GT.0) THEN
          IM(J)=ITR(IM(J))
          IF(IM(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JM(J).GT.0) THEN
          JM(J)=ITR(JM(J))
          IF(JM(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KM(J).GT.0) THEN
          KM(J)=ITR(KM(J))
          IF(KM(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(LM(J).GT.0) THEN
          LM(J)=ITR(LM(J))
          IF(LM(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          IM(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,77) NDROP
77  FORMAT(' ***',I5,' IMAGE IMPROPER DIHEDRALS WERE DROPPED IN ', &
         'REMAPPING ***')

#if KEY_CMAP==1
    !
    !     DO CROSSTERMS
    !
    NDROP=0
    DO I=1,NIMCRT
       J=I+NCRTERM
       DROP=.FALSE.
       IF(I1CT(J).GT.0) THEN
          I1CT(J)=ITR(I1CT(J))
          IF(I1CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(J1CT(J).GT.0) THEN
          J1CT(J)=ITR(J1CT(J))
          IF(J1CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(K1CT(J).GT.0) THEN
          K1CT(J)=ITR(K1CT(J))
          IF(K1CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(L1CT(J).GT.0) THEN
          L1CT(J)=ITR(L1CT(J))
          IF(L1CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(I2CT(J).GT.0) THEN
          I2CT(J)=ITR(I2CT(J))
          IF(I2CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(J2CT(J).GT.0) THEN
          J2CT(J)=ITR(J2CT(J))
          IF(J2CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(K2CT(J).GT.0) THEN
          K2CT(J)=ITR(K2CT(J))
          IF(K2CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(L2CT(J).GT.0) THEN
          L2CT(J)=ITR(L2CT(J))
          IF(L2CT(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          I1CT(J)=0
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,977) NDROP
977 FORMAT(' ***',I5,' IMAGE CROSSTERMS WERE DROPPED IN ', &
         'REMAPPING ***')
#endif 
    !
    !     DO IMAGE EXCLUSIONS

    return
    !
    NDROP=0
    IPT=1
    DO J=1,NIMEXCL
       DROP=.FALSE.
       IF(IMEXCLI(J).GT.0) THEN
          IMEXCLI(J)=ITR(IMEXCLI(J))
          IF(IMEXCLI(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(IMEXCLJ(J).GT.0) THEN
          IMEXCLJ(J)=ITR(IMEXCLJ(J))
          IF(IMEXCLJ(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          NDROP=NDROP+1
       ELSE
          IF(IPT+1.LT.J) THEN
             IMEXCLI(IPT)=IMEXCLI(J)
             IMEXCLJ(IPT)=IMEXCLJ(J)
             IMEXCLT(IPT)=IMEXCLT(J)
             IPT=IPT+1
          ENDIF
       ENDIF
    ENDDO
    IF(NDROP.GT.0) THEN
       NIMEXCL=NIMEXCL-NDROP
       IF(WRNLEV.GE.2) WRITE(OUTU,82) NDROP
82  FORMAT(' ***',I5,' IMAGE EXCLUSIONS WERE DROPPED IN ', &
         'REMAPPING ***')
    ENDIF
    !
    !     DO INTERNAL COORDINATES
    !
    NDROP=0
    DO J=1,LENIC
       DROP=.FALSE.
       IF(IAR(J).GT.0) THEN
          IAR(J)=ITR(IAR(J))
          IF(IAR(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(JAR(J).GT.0) THEN
          JAR(J)=ITR(JAR(J))
          IF(JAR(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(KAR(J).GT.0) THEN
          KAR(J)=ITR(KAR(J))
          IF(KAR(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(LAR(J).GT.0) THEN
          LAR(J)=ITR(LAR(J))
          IF(LAR(J).EQ.0) DROP=.TRUE.
       ENDIF
       IF(DROP) THEN
          NDROP=NDROP+1
       ENDIF
    ENDDO
    !
    IF(NDROP.GT.0 .AND. WRNLEV.GE.2) WRITE(OUTU,88) NDROP
88  FORMAT(' ***',I5,' INTERNAL COORDINATES WERE DROPPED IN ', &
         'REMAPPING ***')
    !
    RETURN
  END SUBROUTINE IMMAP
  
  SUBROUTINE IMHBON(BIMAG,X,Y,Z)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE HANDLES THE SET UP OF THE IMAGE HBOND ARRAYS
    !
    !     By Bernard R. Brooks    1983
    !
    use chm_kinds
    use chm_types
    use dimens_fcm
    use memory
    use hbondm
    use image
    use param
    use psf
    use stream
    implicit none
    !
    !!      INTEGER BIMAG(*)
    type(imageDataStructure) BIMAG
    real(chm_real) X(*),Y(*),Z(*)
    !     LOCAL STORAGE
    integer,allocatable,dimension(:) :: IHBA,JHBA,KHBA,LHBA
    integer,allocatable,dimension(:) :: IHBB,JHBB,KHBB,LHBB
    integer NHBA, NHBB
    integer,allocatable,dimension(:) :: IMACC,IMAC1,IMIHD,IMDON,IATTR
    integer NIMACC,NIMDON
    integer,allocatable,dimension(:) :: IMATTN
    real(chm_real),allocatable,dimension(:) :: IENER
    INTEGER LEN,MAXHBX,MAXN
    real(chm_real) RLEN
    LOGICAL QPRINT
    !
    QPRINT=(PRNLEV.GE.5)
    MAXHBX=MAXHB-NHB
    LEN=MAXHBX
    call chmalloc('upimag.src','IMHBON','IHBA',LEN,intg=IHBA)
    call chmalloc('upimag.src','IMHBON','JHBA',LEN,intg=JHBA)
    call chmalloc('upimag.src','IMHBON','KHBA',LEN,intg=KHBA)
    call chmalloc('upimag.src','IMHBON','LHBA',LEN,intg=LHBA)
    call chmalloc('upimag.src','IMHBON','IHBB',LEN,intg=IHBB)
    call chmalloc('upimag.src','IMHBON','JHBB',LEN,intg=JHBB)
    call chmalloc('upimag.src','IMHBON','KHBB',LEN,intg=KHBB)
    call chmalloc('upimag.src','IMHBON','LHBB',LEN,intg=LHBB)
    RLEN=NDON+NACC
    MAXN=(RLEN*NATIM)/NATOM+100
    LEN=MAXN
    call chmalloc('upimag.src','IMHBON','IMACC',LEN,intg=IMACC)
    call chmalloc('upimag.src','IMHBON','IMAC1',LEN,intg=IMAC1)
    call chmalloc('upimag.src','IMHBON','IMIHD',LEN,intg=IMIHD)
    call chmalloc('upimag.src','IMHBON','IMDON',LEN,intg=IMDON)
    call chmalloc('upimag.src','IMHBON','IATTR',NATOM,intg=IATTR)
    !
    CALL NEWHBL(NATOM,NTRANS, &
         BIMAG%IMATPT,BIMAG%IMATTR, &
         MAXN,NACC,IACC,IAC1, &
         IATTR,NIMACC,IMACC,IMAC1, &
         NDON,IDON,IHD1,NIMDON, &
         IMDON,IMIHD)
    !
    IF(PRNLEV.GE.3) THEN
       IF(BEST) WRITE(OUTU,'(2A,/)') &
            ' *** WARNING: HBOND OPTION -BEST- SHOULD NOT BE USED', &
            ' WITH IMAGES'
    ENDIF
    !
    CALL HBFIND(OUTU,QPRINT,IDON,IHD1,1,NDON, &
         IMACC,IMAC1,1,NIMACC,X,Y,Z, &
         IHBA,JHBA,KHBA,LHBA,1, &
         NHBA,MAXHBX,LHBFG,CUTHB,CUTHBA)
    !
    IF(PRNLEV.GE.5) WRITE(OUTU,'(21X,I5,A)') NHBA, &
         ' primary donor to image acceptor hbonds found'
    !
    CALL HBFIND(OUTU,QPRINT,IMDON,IMIHD,1,NIMDON, &
         IACC,IAC1,1,NACC,X,Y,Z, &
         IHBB,JHBB,KHBB,LHBB,1, &
         NHBB,MAXHBX,LHBFG,CUTHB,CUTHBA)
    call chmdealloc('upimag.src','IMHBON','IMACC',LEN,intg=IMACC)
    call chmdealloc('upimag.src','IMHBON','IMAC1',LEN,intg=IMAC1)
    call chmdealloc('upimag.src','IMHBON','IMIHD',LEN,intg=IMIHD)
    call chmdealloc('upimag.src','IMHBON','IMDON',LEN,intg=IMDON)
    call chmdealloc('upimag.src','IMHBON','IATTR',NATOM,intg=IATTR)
    !
    IF(PRNLEV.GE.5) WRITE(OUTU,'(21X,I5,A)') NHBB, &
         ' image donor to primary acceptor hbonds found'
    !
    call chmalloc('upimag.src','IMHBON','IMATTN',NATIM,intg=IMATTN)
    CALL IMHBFX(NATOM,BIMAG%IMATTR,BIMAG%IMATPT, &
         IMATTN,NTRANS,LIMINV,IMINV, &
         NHBA,IHBA,JHBA,KHBA,LHBA, &
         NHBB,IHBB,JHBB,KHBB,LHBB, &
         MAXHBX,NIMHB,IHB(NHB+1), &
         JHB(NHB+1),KHB(NHB+1),LHB(NHB+1))
    call chmdealloc('upimag.src','IMHBON','IHBA',LEN,intg=IHBA)
    call chmdealloc('upimag.src','IMHBON','JHBA',LEN,intg=JHBA)
    call chmdealloc('upimag.src','IMHBON','KHBA',LEN,intg=KHBA)
    call chmdealloc('upimag.src','IMHBON','LHBA',LEN,intg=LHBA)
    call chmdealloc('upimag.src','IMHBON','IHBB',LEN,intg=IHBB)
    call chmdealloc('upimag.src','IMHBON','JHBB',LEN,intg=JHBB)
    call chmdealloc('upimag.src','IMHBON','KHBB',LEN,intg=KHBB)
    call chmdealloc('upimag.src','IMHBON','LHBB',LEN,intg=LHBB)
    call chmdealloc('upimag.src','IMHBON','IMATTN',NATIM,intg=IMATTN)
    !
    CALL HCODES(ICH(NHB+1),NIMHB,IHB(NHB+1),JHB(NHB+1),IAC)
    !
    call chmalloc('upimag.src','IMHBON','IENER',NIMHB,crl=IENER)
    CALL HBEDIT(OUTU,X,Y,Z,IMOVE,NATIM, &
         IHB(NHB+1),JHB(NHB+1),KHB(NHB+1), &
         LHB(NHB+1),NIMHB,MAXHBX, &
         ICH(NHB+1),CHBA,CHBB,HBEXPN, &
         BEST,HBEXCL, &
         BIMAG%IMIBLO,BIMAG%IMINB,BIMAG%NIMINB, &
         CTONHB,CTOFHB,CTONHA,CTOFHA,IENER)
    call chmdealloc('upimag.src','IMHBON','IENER',NIMHB,crl=IENER)
    !
    RETURN
  END SUBROUTINE IMHBON

  SUBROUTINE NEWHBL(NATOM,NTRANS,IMATPT,IMATTR,MAXN,NACC,IACC,IAC1, &
       IATTR,NIMACC,IMACC,IMAC1,NDON,IDON,IHD1,NIMDON,IMDON,IMIHD)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE SETS UP THE TEMPORARY IMAGE DONER AND IMAGE ACCEPTOR
    !     ARRAYS NEEDED TO GENERATE IMAGE-PRIMARY AND PRIMARY-IMAGE HBONDS.
    !
    !     By Bernard R. Brooks    1983
    !
    use chm_kinds
    use stream
    implicit none
    !
    INTEGER NATOM,NTRANS,MAXN,NACC,NIMACC,NDON,NIMDON
    INTEGER IMATPT(*),IMATTR(*)
    INTEGER IACC(*),IAC1(*),IATTR(*),IMACC(*),IMAC1(*)
    INTEGER IDON(*),IHD1(*),IMDON(*),IMIHD(*)
    INTEGER ITEMP,ITRANS,ISTRT,IEND,I,J,ID,IDM,IA,IAM
    !
    NIMACC=0
    NIMDON=0
    !
    ITEMP=NATOM+1
    DO ITRANS=1,NTRANS
       DO J=1,NATOM
          IATTR(J)=0
       ENDDO
       ISTRT=ITEMP
       IEND=IMATPT(ITRANS)
       ITEMP=IEND+1
       DO I=ISTRT,IEND
          J=IMATTR(I)
          IATTR(J)=I
       ENDDO
       DO ID=1,NDON
          IDM=IATTR(IDON(ID))
          IF(IDM.GT.0) THEN
             NIMDON=NIMDON+1
             IF(NIMDON.GT.MAXN) CALL WRNDIE(-4,'<NEWHBL>', &
                  'Too many image Hbond donors generated')
             IMDON(NIMDON)=IDM
             IF (IHD1(ID).GT.0) THEN
                IMIHD(NIMDON)=IATTR(IHD1(ID))
             ELSE
                IMIHD(NIMDON)=0
             ENDIF
          ENDIF
       ENDDO
       DO IA=1,NACC
          IAM=IATTR(IACC(IA))
          IF(IAM.GT.0) THEN
             NIMACC=NIMACC+1
             IF(NIMACC.GT.MAXN) CALL WRNDIE(-4,'<NEWHBL>', &
                  'Too many image Hbond acceptors generated')
             IMACC(NIMACC)=IAM
             IF (IAC1(IA).GT.0) THEN
                IMAC1(NIMACC)=IATTR(IAC1(IA))
             ELSE
                IMAC1(NIMACC)=0
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !
    RETURN
  END SUBROUTINE NEWHBL

  SUBROUTINE IMHBFX(NATOM,IMATTR,IMATPT,IMATTN,NTRANS,LIMINV,IMINV, &
       NHBA,IHBA,JHBA,KHBA,LHBA,NHBB,IHBB,JHBB,KHBB,LHBB, &
       MAXHB,NIMHB,IHB,JHB,KHB,LHB)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE COMBINES THE TWO LISTS (P-I AND I-P) OF HBONDS INTO
    !     ONE AND DELETES ANY REDUNDANT ENTRIES.
    !
    !     By Bernard R. Brooks    1983
    !
    !
    use chm_kinds
    use stream
    implicit none
    !
    INTEGER NATOM,NTRANS,NHBA,NHBB,MAXHB,NIMHB
    INTEGER IMATTR(*),IMATPT(*),IMATTN(*)
    INTEGER IHBA(*),JHBA(*),KHBA(*),IHBB(*),JHBB(*),KHBB(*)
    INTEGER LHBA(*),LHBB(*),LHB(*)
    INTEGER IHB(*),JHB(*),KHB(*)
    INTEGER IMINV(*)
    LOGICAL LIMINV
    INTEGER ITEMP,ISTRT,IEND,I,IT
    !
    ITEMP=NATOM+1
    DO IT=1,NTRANS
       ISTRT=ITEMP
       IEND=IMATPT(IT)
       ITEMP=IEND+1
       DO I=ISTRT,IEND
          IMATTN(I)=IT
       ENDDO
    ENDDO
    !
    NIMHB=0
    !
    !     Image JHBA atoms
    !
    DO I=1,NHBA
       IT=IMATTN(JHBA(I))
       IF (LIMINV .AND. IMINV(IT).LT.IT) THEN
       ELSE IF ((LIMINV.AND.IMINV(IT).GT.IT).OR.(IHBA(I).GE. &
            IMATTR(JHBA(I)))) THEN
          NIMHB=NIMHB+1
          IF(NIMHB.GT.MAXHB) CALL WRNDIE(-4,'<IMHBFX>', &
               'Too many image Hbonds generated')
          IHB(NIMHB)=IHBA(I)
          JHB(NIMHB)=JHBA(I)
          KHB(NIMHB)=KHBA(I)
          LHB(NIMHB)=LHBA(I)
       ENDIF
    ENDDO
    !
    DO I=1,NHBB
       IT=IMATTN(IHBB(I))
       IF (LIMINV .AND. IMINV(IT).LT.IT) THEN
       ELSE IF ((LIMINV.AND.IMINV(IT).GT.IT).OR.(JHBB(I).GT. &
            IMATTR(IHBB(I)))) THEN
          NIMHB=NIMHB+1
          IF(NIMHB.GT.MAXHB) CALL WRNDIE(-4,'<IMHBFX>', &
               'Too many image Hbonds generated')
          IHB(NIMHB)=IHBB(I)
          JHB(NIMHB)=JHBB(I)
          KHB(NIMHB)=KHBB(I)
          LHB(NIMHB)=LHBB(I)
       ENDIF
    ENDDO
    !
    IF(PRNLEV.GE.5) WRITE(OUTU,55) NIMHB
55  FORMAT(21X,I5,' unique image hbonds found')
    !
    RETURN
  END SUBROUTINE IMHBFX

end module imgup
