module dmcons
  use chm_kinds
  use dimens_fcm
  !  Distance Map restraint
  !  DMC0   Reference value for distance map restraint potential
  !  DMCUT  Cutoff value for consideration of contact
  !  DMFORCE Force constant for distance map harmonic restraint
  !  DUNIT  Unit to output information about contact map values
  !  NSVDMC Frequency to save output
  !  NCOUNT Current count of dynamics steps
  !  DSTEP  Count of dynamics steps
  !  ICATM\
  !  JCATM \ Selection lists
  !  ICRES /
  !  JCRES/
  !  DMWEI  Contact matrix weight values
  !  DMNCO  Number of contacts to be read in setting up contact potential
  !  QSAVE logical flag indicating saves to file
  !  QDMC  logical flag indicating DMC potential to be used
  !  dmc_atomlist(1:dmc_natomlist) = list of all atoms involved in restraint
#if KEY_DMCONS == 0

  integer, parameter :: ndmc = 0
  logical, parameter :: qdmc = .false.
  
#else /* KEY_DMCONS */
  private

  INTEGER :: NCOUNT, DSTEP
  INTEGER :: NDMC
  integer,allocatable,dimension(:) :: DMNCO, DUNIT, NSVDMC
  integer,allocatable,dimension(:,:) :: ICATM, JCATM, ICRES, JCRES
  integer, allocatable, dimension(:,:) :: dmc_atomlist
  integer,allocatable,dimension(:) :: dmc_natomlist
  real(chm_real),allocatable,dimension(:,:) :: DMWEI,DMCUT_con
  real(chm_real),allocatable,dimension(:) :: DMFORCE, DMC0, DMCUT
  LOGICAL :: QDMC, QSAVE

  ! Public variables
  public qdmc, dstep, dmc_atomlist, dmc_natomlist, ndmc

  ! Public subroutines
  public dmcons_init, dmcset, edmc, write_edmc

contains

  subroutine dmcons_init()
    use memory

    if(allocated(DMFORCE)) call chmdealloc('dmcons.src','dmcons_init','DMFORCE',size(DMFORCE),crl=DMFORCE)
    if(allocated(DMC0)) call chmdealloc('dmcons.src','dmcons_init','DMC0',size(DMC0),crl=DMC0)
    if(allocated(DMCUT)) call chmdealloc('dmcons.src','dmcons_init','DMCUT',size(DMCUT),crl=DMCUT)
    if(allocated(DMNCO)) call chmdealloc('dmcons.src','dmcons_init','DMNCO',size(DMNCO),intg=DMNCO)
    if(allocated(DUNIT)) call chmdealloc('dmcons.src','dmcons_init','DUNIT',size(DUNIT),intg=DUNIT)
    if(allocated(NSVDMC)) call chmdealloc('dmcons.src','dmcons_init','NSVDMC',size(NSVDMC),intg=NSVDMC)

    if(allocated(ICATM)) call chmdealloc('dmcons.src','dmcons_init','ICATM',size(ICATM,1),size(ICATM,2),intg=ICATM)
    if(allocated(JCATM)) call chmdealloc('dmcons.src','dmcons_init','JCATM',size(JCATM,1),size(JCATM,2),intg=JCATM)
    if(allocated(ICRES)) call chmdealloc('dmcons.src','dmcons_init','ICRES',size(ICRES,1),size(ICRES,2),intg=ICRES)
    if(allocated(JCRES)) call chmdealloc('dmcons.src','dmcons_init','JCRES',size(JCRES,1),size(JCRES,2),intg=JCRES)
    if(allocated(DMWEI)) call chmdealloc('dmcons.src','dmcons_init','DMWEI',size(DMWEI,1),size(DMWEI,2),crl=DMWEI)

    if(allocated(DMCUT_con)) &
         call chmdealloc('dmcons.src','dmcons_init','DMCUT_con', &
                size(DMCUT_con,1),size(DMCUT_con,2),crl=DMCUT_con)
    if(allocated(dmc_atomlist)) &
         call chmdealloc('dmcons.src','dmcons_init','dmc_atomlist', &
                size(dmc_atomlist,1),size(dmc_atomlist,2),intg=dmc_atomlist)
    if(allocated(dmc_natomlist)) &
         call chmdealloc('dmcons.src','dmcons_init','dmc_natomlist', &
                size(dmc_natomlist),intg=dmc_natomlist)

    qsave=.false.
    qdmc=.false.
    ndmc=0
    return
  end subroutine dmcons_init


  SUBROUTINE DMCSET
    !
    ! Sets up Distance Map constraint force field.
    !
    !   SYNTAX:  DMCOnstrain FORCe real REFErence real OUTPut_unit integer      
    !            NSAVe_output integer CUTOff real NCONtact integer   
    !            {SELE {atom selection} WEIGt real}(ncontact times) 
    !
    use psf
    use comand
    use coord
    use stream
    use string
    use number
    use exfunc
    use parallel
    use memory
    implicit none
    !
    ! parse the command line info
    !
    real(chm_real)    CUTD
    character(LEN=4) :: WRD

    WRD = CURRA4(COMLYN,COMLEN)
    IF (WRD .EQ. "CLEA") then
       WRITE(OUTU,'(a,I4,a)') ' Clearing ', ndmc, ' sets of Distance Map constraints'
       WRD = NEXTA4(COMLYN,COMLEN) 
       CALL dmcons_init
    ELSE
       ndmc = ndmc + 1

       IF (NDMC .EQ. 1) THEN
          QSAVE = .FALSE.
          QDMC = .FALSE.
          NCOUNT=0
          CUTD=SIX+HALF
       ENDIF
   
       if (NDMC .gt. 1) then
          call chmrealloc('dmcons.src','DMCSET','DMFORCE',NDMC,crl=DMFORCE)
          call chmrealloc('dmcons.src','DMCSET','DMC0',NDMC,crl=DMC0)
          call chmrealloc('dmcons.src','DMCSET','DMCUT',NDMC,crl=DMCUT)
          call chmrealloc('dmcons.src','DMCSET','DMNCO',NDMC,intg=DMNCO)
          call chmrealloc('dmcons.src','DMCSET','DUNIT',NDMC,intg=DUNIT)
          call chmrealloc('dmcons.src','DMCSET','NSVDMC',NDMC,intg=NSVDMC)
       else
          call chmalloc('dmcons.src','DMCSET','DMFORCE',NDMC,crl=DMFORCE)
          call chmalloc('dmcons.src','DMCSET','DMC0',NDMC,crl=DMC0)
          call chmalloc('dmcons.src','DMCSET','DMCUT',NDMC,crl=DMCUT)
          call chmalloc('dmcons.src','DMCSET','DMNCO',NDMC,intg=DMNCO)
          call chmalloc('dmcons.src','DMCSET','DUNIT',NDMC,intg=DUNIT)
          call chmalloc('dmcons.src','DMCSET','NSVDMC',NDMC,intg=NSVDMC)
       endif

       DMFORCE(ndmc) = GTRMF(COMLYN,COMLEN,'FORC',ZERO)
       DMC0(ndmc) = GTRMF(COMLYN,COMLEN,'REFE',ZERO)
       DMCUT(ndmc) = GTRMF(COMLYN,COMLEN,'CUTO',CUTD)
       DMNCO(ndmc) = GTRMI(COMLYN,COMLEN,'NCON',0)
       DUNIT(ndmc) = GTRMI(COMLYN,COMLEN,'OUTP',6)
       NSVDMC(ndmc) = GTRMI(COMLYN,COMLEN,'NSAV',0)
       
       CALL DMCSET2

       IF((NSVDMC(NDMC).GT.0).AND.(DUNIT(NDMC).GT.0)) QSAVE = .TRUE.
       IF((QSAVE.OR.(DMFORCE(NDMC).GE.ZERO)).AND.(DMNCO(NDMC).GT.0)) QDMC = .TRUE.
    
       !
       ! Print the info
       !
       IF(PRNLEV.GE.5.AND.MYNOD.EQ.0) THEN
          IF(QDMC) THEN
             WRITE(OUTU,2) NDMC 
             WRITE(OUTU,1) DMFORCE(ndmc),DMC0(ndmc),NSVDMC(ndmc),DUNIT(ndmc),DMNCO(ndmc)
          ELSE
             WRITE(OUTU,'(a)') ' DMCSET : No constraint set nothing done '
          ENDIF
       ENDIF
       
1      FORMAT('DMCSET: FORCe = ',F12.6,' REFE = ',F10.6,' NSAVe = ',I4, &
            ' ON UNIT = ',I4,' NCON = ',I4)
2      FORMAT('DMCSET : dmat constraint number ',I3,' has been set')
       
    END IF
    RETURN
  END SUBROUTINE DMCSET

  SUBROUTINE DMCSET2

    use stream
    use string
    use number
    use psf
    use comand
    use coord
    use exfunc
    use memory
    use chutil,only:atomid
    use select
#if KEY_DOMDEC==1
    use domdec_common,only:domdec_system_changed
#endif 
    implicit none

    INTEGER IC,I,NI,NJ

    integer,dimension(NATOM) :: SELCT
    CHARACTER(len=8) :: RID0, sid0
    CHARACTER(len=8) :: SID,RID,REN,AC
    LOGICAL EOF
    ! Atom count in large residue (nucleic acids 33, some lipids > 100)
    integer,parameter :: LGRES = 50
    integer :: DMMXAT
    integer :: PREVNI, PREVNJ, MAXNATOM, MAXDMNCO
    logical :: BADGROUP
    ! temporary list: in_atomlist(i) = .true., atom i is already in dmc_atomlist()
    logical, allocatable, dimension(:) :: in_atomlist

    MAXDMNCO = 0

    DO I=1,NDMC
       IF (DMNCO(I) .gt. MAXDMNCO) then
          MAXDMNCO = DMNCO(I)
       ENDIF
    ENDDO

    DMMXAT = LGRES * MAXDMNCO

    IF (NDMC .gt. 1) then
       call realloc_2d('dmcons.src','DMCSET2','ICATM',NDMC,DMMXAT,intg=ICATM)
       call realloc_2d('dmcons.src','DMCSET2','JCATM',NDMC,DMMXAT,intg=JCATM)
       call realloc_2d('dmcons.src','DMCSET2','ICRES',NDMC,MAXDMNCO,intg=ICRES)
       call realloc_2d('dmcons.src','DMCSET2','JCRES',NDMC,MAXDMNCO,intg=JCRES)
       call realloc_2d('dmcons.src','DMCSET2','DMWEI',NDMC,MAXDMNCO,crl=DMWEI)
       call realloc_2d('dmcons.src','DMCSET2','DMCUT_con',NDMC,MAXDMNCO,crl=DMCUT_con)
       call realloc_2d('dmcons.src','DMCSET2','dmc_atomlist',NDMC,natom,intg=dmc_atomlist)
       call chmrealloc('dmcons.src','DMCSET2','dmc_natomlist',NDMC,intg=dmc_natomlist)
    ELSE
       call alloc_2d('dmcons.src','DMCSET2','ICATM',NDMC,DMMXAT,intg=ICATM)
       call alloc_2d('dmcons.src','DMCSET2','JCATM',NDMC,DMMXAT,intg=JCATM)
       call alloc_2d('dmcons.src','DMCSET2','ICRES',NDMC,MAXDMNCO,intg=ICRES)
       call alloc_2d('dmcons.src','DMCSET2','JCRES',NDMC,MAXDMNCO,intg=JCRES)
       call alloc_2d('dmcons.src','DMCSET2','DMWEI',NDMC,MAXDMNCO,crl=DMWEI)
       call alloc_2d('dmcons.src','DMCSET2','DMCUT_con',NDMC,MAXDMNCO,crl=DMCUT_con)
       call alloc_2d('dmcons.src','DMCSET2','dmc_atomlist',NDMC,natom,intg=dmc_atomlist)
       call chmalloc('dmcons.src','DMCSET2','dmc_natomlist',NDMC,intg=dmc_natomlist)
    ENDIF
    ! XXX the above are deallocated only in dmcons_init (but are reallocated when new DMC sets are added, as in above)

    call chmalloc('dmcons.src','DMCSET2','in_atomlist',natom,log=in_atomlist)
    in_atomlist(1:natom) = .false.
    
    dmc_natomlist(NDMC) = 0

    ICATM(NDMC,1:DMMXAT) = 0
    JCATM(NDMC,1:DMMXAT) = 0

    EOF = .FALSE.

    NI=0
    NJ=0
    BADGROUP = .FALSE.
    DO IC=1,DMNCO(NDMC)
       PREVNI = NI
       PREVNJ = NJ
       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
            'DMCONS> ')
       IF(EOF) THEN
          CALL WRNDIE(0,'<DMCSET2>','SELECTION ERROR')
       ENDIF
       RID0= '    '
       sid0='    '
       CALL SELCTA(COMLYN,COMLEN,SELCT,X,Y,Z,WMAIN,.TRUE.)
       DO I=1,NATOM
          IF(SELCT(I).EQ.1) THEN
             CALL ATOMID(I,SID,RID,REN,AC)
             IF(RID0.EQ.'    ') RID0=RID
             if(sid0.eq.'    ') sid0=sid
             IF(RID.EQ.RID0 .and. sid.eq.sid0) THEN
                NI=NI+1
                IF (NI > DMMXAT) CALL WRNDIE(-2, '<DMCSET2>', &
                     'ICATM ARRAY OVERFLOW')
                ICATM(NDMC,NI)=I
             ELSE
                NJ=NJ+1
                IF (NJ > DMMXAT) CALL WRNDIE(-2, '<DMCSET2>', &
                     'JCATM ARRAY OVERFLOW')
                JCATM(NDMC,NJ)=I
             ENDIF
             ! Add atom i to atomlist
             if (.not.in_atomlist(i)) then
                dmc_natomlist(NDMC) = dmc_natomlist(NDMC) + 1
                dmc_atomlist(NDMC,dmc_natomlist(NDMC)) = i
                in_atomlist(i) = .true.
             endif
          ENDIF
       ENDDO
       ICRES(NDMC,IC)=NI
       JCRES(NDMC,IC)=NJ
       DMWEI(NDMC,IC) = GTRMF(COMLYN,COMLEN,'WEIG',ONE)
       DMCUT_con(NDMC,IC) = GTRMF(COMLYN,COMLEN,'CUTO',DMCUT(NDMC))

       IF (NI == PREVNI) THEN
          BADGROUP = .TRUE.
          CALL WRNDIE(2, '<DMCSET2>', '0 atoms selected in either residue')
       ELSE IF (NJ == PREVNJ) THEN
          BADGROUP = .TRUE.
          CALL WRNDIE(2, '<DMCSET2>', '0 atoms selected in one residue')
       ENDIF
    ENDDO
    IF (BADGROUP) CALL WRNDIE(0, '<DMCSET2>', 'BAD CONTACT GROUPS')

#if KEY_DOMDEC==1
    if (dmc_natomlist(NDMC) > 0) then
       call domdec_system_changed()
    endif
#endif

    ! Re-size dmc_atomlist
    ! get max dmc_natomlist
    MAXNATOM = 0
    DO I=1,NDMC
       IF (dmc_natomlist(I) > MAXNATOM) then
          MAXNATOM = dmc_natomlist(I)
       ENDIF
    ENDDO

    call realloc_2d('dmcons.src','DMCSET2','dmc_atomlist',NDMC,MAXNATOM,intg=dmc_atomlist)
    call chmdealloc('dmcons.src','DMCSET2','in_atomlist',natom,log=in_atomlist)

    RETURN
  END SUBROUTINE DMCSET2

  SUBROUTINE EDMC(ECDMC,RHO,X,Y,Z,DX,DY,DZ,NATOM,vpress)
    !
    ! THIS ROUTINE ADDS A Quadratic POTENTIAL to restrain the
    ! reaction coordinate. The reaction coordinate is defined
    ! as a weighted sum of contacts.
    !
    !                                   2
    !      E= 1/2 * CONST * (RHO - DMC0)
    !
    !  where
    !
    !  RHO  = SUM (Weigt * (1-STATE );
    !          i        i          i
    !
    !                        1
    !  STATE = ----------------------------
    !       i   1 + EXP(20*(DIST - (CUTOff+0.25)))
    !                            i
    !  DIST - distance between centers of geometry of residues
    !      i
    !         forming contact i
    !
    !
    use number
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: ECDMC(NDMC), RHO(NDMC)
!    real(chm_real) :: ECDMC_arr(NDMC), RHO_arr(NDMC)
    real(chm_real), intent(in) :: X(*),Y(*),Z(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real), optional, intent(out) :: vpress(9)
    ! Variables
    INTEGER NI,I,IFIRST,ILAST,JFIRST,JLAST, &
         NATOM, DMC
    real(chm_real) XCG1, YCG1, ZCG1, NORM, &
         XCG2, YCG2, ZCG2, DIST, DMSTATE, &
         DEN1,DEN2,FCTR, fctrx, fctry, fctrz
    ! Forces from potential
    real(chm_real),dimension(NATOM) :: CONX, CONY, CONZ
    !mfc..
    real(chm_real) ETMP,LOGRBIG
    ! preventitive measure to insure that exp(ETMP) below
    ! does not blow up. logrbig will be the largest
    ! allowed value of etmp.
    LOGRBIG=LOG(RBIG)
    !mfc..

    DO DMC=1,NDMC
!       RHO_arr(DMC)=ZERO
       RHO(DMC)=ZERO
       NORM=ZERO

       DO I=1,NATOM
          CONX(I)=ZERO
          CONY(I)=ZERO
          CONZ(I)=ZERO
       ENDDO

       DO I=1,DMNCO(DMC)
          XCG1=ZERO
          YCG1=ZERO
          ZCG1=ZERO
          XCG2=ZERO
          YCG2=ZERO
          ZCG2=ZERO
          DEN1=ZERO
          DEN2=ZERO

          IF(I.EQ.1) THEN
             IFIRST=1
          ELSE
             IFIRST=1+ICRES(DMC,I-1)
          ENDIF
          ILAST=ICRES(DMC,I)
          DO NI=IFIRST,ILAST
             XCG1=XCG1+X(ICATM(DMC,NI))
             YCG1=YCG1+Y(ICATM(DMC,NI))
             ZCG1=ZCG1+Z(ICATM(DMC,NI))
             DEN1=DEN1+ONE
          ENDDO
          XCG1=XCG1/DEN1
          YCG1=YCG1/DEN1
          ZCG1=ZCG1/DEN1

          IF(I.EQ.1) THEN
             JFIRST=1
          ELSE
             JFIRST=1+JCRES(DMC,I-1)
          ENDIF
          JLAST=JCRES(DMC,I)
          DO NI=JFIRST,JLAST
             XCG2=XCG2+X(JCATM(DMC,NI))
             YCG2=YCG2+Y(JCATM(DMC,NI))
             ZCG2=ZCG2+Z(JCATM(DMC,NI))
             DEN2=DEN2+ONE
          ENDDO
          XCG2=XCG2/DEN2
          YCG2=YCG2/DEN2
          ZCG2=ZCG2/DEN2

          DIST=DSQRT((XCG1-XCG2)*(XCG1-XCG2)+ &
               (YCG1-YCG2)*(YCG1-YCG2)+ &
               (ZCG1-ZCG2)*(ZCG1-ZCG2))
          !mfc..
          ! DMSTATE=ONE/(ONE+EXP(TWENTY*(DIST-(DMCUT+PT25))))  !"EXP"?
          ETMP=TWENTY*(DIST-(DMCUT_con(DMC,I)+PT25))
          IF(ETMP.GT.LOGRBIG)THEN
             DMSTATE=ZERO
          ELSE
             DMSTATE=ONE/(ONE+EXP(ETMP))
          ENDIF
          !mfc..
          FCTR = DMFORCE(DMC)*TWENTY*DMWEI(DMC,I)*DMSTATE &
               * (ONE-DMSTATE)/(DIST*DEN1)

          DO NI=IFIRST,ILAST
             CONX(ICATM(DMC,NI))= CONX(ICATM(DMC,NI))+FCTR*(XCG1-XCG2)
             CONY(ICATM(DMC,NI))= CONY(ICATM(DMC,NI))+FCTR*(YCG1-YCG2)
             CONZ(ICATM(DMC,NI))= CONZ(ICATM(DMC,NI))+FCTR*(ZCG1-ZCG2)
          ENDDO
          FCTR = FCTR*DEN1/DEN2
          DO NI=JFIRST,JLAST
             CONX(JCATM(DMC,NI))= CONX(JCATM(DMC,NI))+FCTR*(XCG2-XCG1)
             CONY(JCATM(DMC,NI))= CONY(JCATM(DMC,NI))+FCTR*(YCG2-YCG1)
             CONZ(JCATM(DMC,NI))= CONZ(JCATM(DMC,NI))+FCTR*(ZCG2-ZCG1)
          ENDDO

!          RHO_arr(DMC)=RHO_arr(DMC)+DMWEI(DMC,I)*(ONE-DMSTATE)
          RHO(DMC)=RHO(DMC)+DMWEI(DMC,I)*(ONE-DMSTATE)
          NORM=NORM+DMWEI(DMC,I)
       ENDDO
!       RHO_arr(DMC)=RHO_arr(DMC)/NORM
       RHO(DMC)=RHO(DMC)/NORM
       ! Compute energy
!       ECDMC_arr(DMC)=HALF*DMFORCE(DMC)*(RHO_arr(DMC)-DMC0(DMC))*(RHO_arr(DMC)-DMC0(DMC))
       ECDMC(DMC)=HALF*DMFORCE(DMC)*(RHO(DMC)-DMC0(DMC))*(RHO(DMC)-DMC0(DMC))

       ! Compute all forces
!       fctr = (RHO_arr(DMC)-DMC0(DMC))/NORM
       fctr = (RHO(DMC)-DMC0(DMC))/NORM

       if (present(vpress)) then
          do ni=1,dmc_natomlist(DMC)
             i = dmc_atomlist(DMC,ni)
             fctrx = conx(i)*fctr
             fctry = cony(i)*fctr
             fctrz = conz(i)*fctr
             dx(i) = dx(i) + fctrx
             dy(i) = dy(i) + fctry
             dz(i) = dz(i) + fctrz          
             vpress(1) = vpress(1) - x(i)*fctrx
             vpress(2) = vpress(2) - x(i)*fctry
             vpress(3) = vpress(3) - x(i)*fctrz
             vpress(4) = vpress(4) - y(i)*fctrx
             vpress(5) = vpress(5) - y(i)*fctry
             vpress(6) = vpress(6) - y(i)*fctrz
             vpress(7) = vpress(7) - z(i)*fctrx
             vpress(8) = vpress(8) - z(i)*fctry
             vpress(9) = vpress(9) - z(i)*fctrz
          enddo
       else
          do ni=1,dmc_natomlist(DMC)
             i = dmc_atomlist(DMC,ni)
             fctrx = conx(i)*fctr
             fctry = cony(i)*fctr
             fctrz = conz(i)*fctr
             dx(i) = dx(i) + fctrx
             dy(i) = dy(i) + fctry
             dz(i) = dz(i) + fctrz
          enddo
       endif
    ENDDO

!    ECDMC = 0
!    RHO = 0
!    DO I=1,NDMC
!       ECDMC = ECDMC + ECDMC_arr(I)
!       RHO = RHO + RHO_arr(I)
!    ENDDO
!    RHO = RHO/NDMC

    ! ARD:  the ECDMC value saved for writing is the sum
    ! of all DMC sets, and the RHO value saved is the average

    ! APH: Writing to file moved to write_dmc -subroutine. See below.
    ! in domdec with split to direct and recip cores, 
    ! recip nodes calculates ecdmc, rho, and the forces.
    ! Recip nodes send the forces to the direct nodes.

    RETURN
  END SUBROUTINE EDMC

  !
  ! Write dmcons output
  ! NOTE: this separation of calculating ecdmc and writing it to file was done
  !       in order to allow the recip processors in domdec to calculate dmcons
  !       and direct root node to write the output
  !
  subroutine write_edmc(ecdmc, rho)
    use stream, only:prnlev, outu
    use contrl, only:irest
    use parallel, only:mynod
    use param_store, only: set_param
    implicit none
    ! Input
    real(chm_real), intent(in) :: ecdmc(NDMC), rho(NDMC)
    ! Variables
    integer dmc
    logical wflag

    call set_param("DMRHO",rho(1))
    IF(NDMC.gt.1)then
       call set_param("DMRHO2",rho(2))
    ENDIF

    if(mynod.ne.0)return

    !
    ! NOW WRITE OUT FILE IF APPROPRIATE
    !
    IF(QSAVE) THEN

       IF((IREST.EQ.1).AND.(NCOUNT.EQ.0)) THEN
          NCOUNT=DSTEP-1 ! for restart dynamics
          IF(PRNLEV.GE.5) &
               WRITE(OUTU,*) 'ECDMC: NCOUNT updated with restart'
       ENDIF ! restart
       IF(NCOUNT.LT.DSTEP) THEN
          NCOUNT = NCOUNT + 1
          DO DMC=1,NDMC
             WFLAG = ((MOD(NCOUNT,NSVDMC(DMC)).EQ.0).AND.(NCOUNT.NE.0))
             IF(WFLAG) WRITE(DUNIT(DMC),'(4(1X,F15.8),1X,I8)') &
                  RHO(DMC), ECDMC(DMC), DMFORCE(DMC), DMC0(DMC), NCOUNT
          END DO
       ENDIF ! increment ncount and write outfile
    ENDIF !IF(QSAVE)

    return
  End subroutine write_edmc
#endif 

End module dmcons
