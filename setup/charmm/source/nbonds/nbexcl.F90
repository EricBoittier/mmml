module nbexcl
  implicit none
  private

  ! Public subroutines
  public upinb, makitc, makitc_clr, suextab

contains

  SUBROUTINE UPINB(BNBNDX)
    !-----------------------------------------------------------------------
    !     Do an INB update as specified. Front end for MKINB.
    !     Gets guesses for storage space allocations from "working"
    !     PSF.
    !
    !      By Bernard R. Brooks    1983
    !
    use chm_kinds
    use chm_types
    !  use dimens_fcm
    use number
    use bases_fcm
    use datstr
    use image
    use imgup
    use inbnd
    use machdep
    use psf
    use stream
    use tsms_mod
    use tsmh
    use replica_mod
#if KEY_SCCDFTB==1
    use sccdftb                
#endif
    use actclus_mod,only: qlbycc
    use memory
#if KEY_BLOCK==1
    use lambdam,only: qmld      /*ldm*/
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec       
    use domdec_common,only:q_inb_changed  
#endif
    use upimag_util,only:upimnb
    use nbexcl_util,only:makgrp
    !---   use nbutil_module,only:setbnd,getbnd
#ifdef KEY_RESIZE
    use resize,only:resize_psf,drsz0
#endif    
    implicit none
    !
    !-----------------------------------------------------------------------
    !  Miscellaneous:
    !
    !  MAXING - The maximum number of atoms in any electrostatic group.
    !
    integer,parameter :: MAXING=1000
    !-----------------------------------------------------------------------
    integer,allocatable,dimension(:) :: IATBON
    integer,allocatable,dimension(:),target :: space_pk
    integer,pointer,dimension(:) :: IPK,JPK,IPK14,JPK14
    integer,allocatable,dimension(:) :: NATBON
    type(nonbondDataStructure) BNBNDX
    type(imageDataStructure) BIMAGX
    !
    INTEGER IATMXB, I, MAXNNG

    integer,allocatable,dimension(:),target :: space_i
    INTEGER,pointer,dimension(:) ::  IGR,IPT,ISTOP
    !
    LOGICAL   CMPLTD
    INTEGER NATIML
#ifdef KEY_RESIZE
    integer nnb0, nnb1
    nnb0=nnb; nnb1=nnb
#endif
    !
    IF(NATOM <= 0) RETURN
    CALL GETBND(BNBNDX,.TRUE.)
    !
    !     wipe out existing nbond lists
    !     this is to free whatever space it may currenlty hold
    !
    IF(NNNB > 0 .OR. NNNBG > 0) THEN
       NNNB=0
       NNNBG=0
       ! No need to resize here. - BRB
       !C       CALL RESIZE(BNBNDX,LNBNDX,SNBND,JNBG,0,1)
       !C       CALL RESIZE(BNBNDX,LNBNDX,SNBND,JNB,0,1)
    ENDIF
    !
    IF(IABS(NBXMOD) > 5) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,288) NBXMOD
288    FORMAT(' <UPINB>: NBXMOD MODE IS OUT OF RANGE.',I5)
       CALL DIEWRN(0)
    ENDIF
    !
    NATIML=NATOM
    IF(NATIM > NATOM) NATIML=NATIM
    call chmalloc('nbexcl.src','UPINB','NATBON',NATIML,intg=NATBON)
    !
    IF(QDRUDE) THEN
       IATMXB=20
    ELSE
       IATMXB=8
    ENDIF
#if KEY_BLOCK==1
    if (qmld) iatmxb = 28  /*ldm*/
#endif

    IF(NATIML < 200) IATMXB=20
#if KEY_REPLICA==1
    !# <caves>-Aug-4-1993 (Leo Caves) accomodate covalently bound replicas
    !# this is a quick fix. the max number could be properly accounted for in the
    !# replication process.
    !C      IF (qRep) IATMXB = nRepl * IATMXB
    !  Assume that each replica will have no more than one bond to
    !  non-replicated atoms, thus a sum is needed. BRB - 12/5/95
    IF (qRep) IATMXB = nRepl + IATMXB
#if KEY_SCCDFTB==1
    !  QC: We notice here that for SCC replica, we might want to increase
    !  the value of IATMXB
    IF (LSCCRP) IATMXB = NSCCRP + IATMXB
#endif 
#endif /*  REPLICA*/
    call chmalloc('nbexcl.src','UPINB','IATBON',IATMXB*NATIML,intg=IATBON)

    call NBGROW(bnbndx%LASTX, NATOM)
    call NBGROW(bnbndx%LASTY, NATOM)
    call NBGROW(bnbndx%LASTZ, NATOM)
    call NBGROW(bnbndx%IBLO14, NATOM)

! make sure NBONDT2 is set when images are not in use
    IF(NATOM.EQ.NATOMT) THEN
       NBONDT=NBOND
       NBONDT2=NBOND
    ENDIF

    I=3*NBONDT2
    IF(IABS(NBXMOD) > 3) I=I*2
#if KEY_REPLICA==1
    !  Allocate some extra space if replica is active
    IF (qRep) THEN
       IF(nSub > 1) I=I*nSub
    ENDIF
#endif 
    IF(NBXMOD > 0) I=MAX(NNB+1,I)
    IF(QDRUDE) I=I*1.5
    call NBGROW(bnbndx%INB14, I)

    ! Mark flag that tells that iblo14 and inb14 have changed
#if KEY_DOMDEC==1
    q_inb_changed = .true.  
#endif

    CMPLTD=.FALSE.
    DO WHILE(.NOT.CMPLTD)
       call chmalloc('nbexcl.src','UPINB','space_PK',4*I,intg=space_PK)
       IPK   => space_pk(0*i+1 : 1*i)
       JPK   => space_pk(1*i+1 : 2*i)
       IPK14 => space_pk(2*i+1 : 3*i)
       JPK14 => space_pk(3*i+1 : 4*i)
       !!QQQQ
       !!       call PRINTDT_nbond('MAKINB_call','BNBNDX',BNBNDX)
#ifdef KEY_RESIZE
       nnb1=nnb1+I
       if (nnb0 .lt. nnb1) then
          nnb0=nnb0+drsz0
          call resize_psf('nbexcl.F90','UPINB','NNB',nnb0, .true.)
       endif
#endif

       CALL MAKINB(NATOM,NATIML,IB,JB,NBONDT2,INB,IBLO,NNB,BNBNDX%INB14, &
            BNBNDX%IBLO14,NNB14,(I), &
            NBXMOD,NATBON,IATBON,IATMXB,CMPLTD, &
#if KEY_TSM==1
            QTSM,REACLS,PRODLS, &       
#endif
            IPK,JPK,IPK14,JPK14, &
            ISDRUDE)

       !!QQQQ
       !!      call PRINTDT_nbond('MAKINB_return','BNBNDX',BNBNDX)


       call chmdealloc('nbexcl.src','UPINB','space_PK',4*I,intg=space_PK)
       IF(.NOT.CMPLTD) THEN
          call NBGROW(bnbndx%INB14, 1.5, 10)
          I = size(bnbndx%INB14)
       ENDIF
    ENDDO
#ifdef KEY_RESIZE
    if (nnb0 .gt. nnb1) then ! reduce to fit
       call resize_psf('nbexcl.F90','UPINB','NNB',nnb1, .true.)
    endif
#endif
    call chmdealloc('nbexcl.src','UPINB','IATBON',IATMXB*NATIML,intg=IATBON)
    !
    ! Generate group exclusion lists
    !

    call NBGROW(bnbndx%IGLO14, NGRP)
    I=NNB14/4+10
    call NBGROW(bnbndx%ING14, I)

    CMPLTD=.FALSE.
    DO WHILE(.NOT.CMPLTD)
       MAXNNG=I
       call chmalloc('nbexcl.src','UPINB','space_I',natom+2*maxing,intg=space_I)
       IGR   => space_i(1              : natom)
       IPT   => space_i(natom+1        : natom+maxing)
       ISTOP => space_i(natom+maxing+1 : natom+2*maxing)

       CALL MAKGRP(1,NGRP,NGRP,BNBNDX%IGLO14, &
            BNBNDX%ING14,NNG14,MAXNNG, &
            BNBNDX%IBLO14,BNBNDX%INB14,IGPBS, &
#ifdef KEY_RESIZE
            IGR,CMPLTD,IPT,ISTOP,size(iac))
#else
            IGR,CMPLTD,IPT,ISTOP)
#endif

       call chmdealloc('nbexcl.src','UPINB','space_I',I+2*maxing,intg=space_I)
       IF(.NOT.CMPLTD) THEN
          call NBGROW(bnbndx%ING14, 1.5, 10)
          I = size(bnbndx%ING14)
       ENDIF
    ENDDO
    !
    call chmdealloc('nbexcl.src','UPINB','NATBON',NATIML,intg=NATBON)

#if KEY_DOMDEC==1
    if (.not.q_domdec) then  
#endif
       IF(NTRANS > 0) CALL UPIMNB(BIMAG)
#if KEY_DOMDEC==1
    endif
#endif

    !
    !!QQQQ
    !      call PRINTDT_nbond('SUEXTAB_call','BNBNDX',BNBNDX)

    CALL SETBND(BNBNDX)
    IF (QLBYCC) THEN
       CALL SUEXTAB(BNBNDX%IBLO14,BNBNDX%INB14, &
            BNBNDX%IGLO14,BNBNDX%ING14, &
            NATOM,NGRP)
    ENDIF

    !!QQQQ
    !!      call PRINTDT_nbond('UPINB_return','BNBNDX',BNBNDX)


    RETURN
  END SUBROUTINE UPINB

  SUBROUTINE MAKINB(NATOM,NATIML,IB,JB,NBOND,INB,IBLO,NNB, &
       INB14,IBLO14,NNB14,MXNB14,MODE,NATBON,IATBON,IATMXB,CMPLTD, &
#if KEY_TSM==1
       LTSM,REACLS,PRODLS, &    
#endif
       IPK,JPK,IPK14,JPK14,ISDRUDE)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE TAKES THE PSF INFORMATION AND GENERATES
    !     A NONBONDED EXCLUSION LIST (INB14 and IBLO14).
    !
    !     MODE =    0        LEAVE THE EXISTING INB14/IBLO14 LISTS
    !     MODE = +- 1        INCLUDE NOTHING
    !     MODE = +- 2        INCLUDE ONLY 1-2 (BOND) INTERACTIONS
    !     MODE = +- 3        INCLUDE 1-2 AND 1-3 (BOND AND ANGLE)
    !     MODE = +- 4        INCLUDE 1-2 1-3 AND 1-4's
    !     MODE = +- 5        INCLUDE 1-2 1-3 AND SPECIAL 1-4 interactions
    !     A POSITIVE MODE VALUE CAUSES THE EXISTING INB ARRAY TO BE ADDED.
    !
    !
    !     ORIGINAL VERSION  OCT-81  BY DJS
    !     OVERHAULED TO ACT ON BOND LIST ONLY  SEP-82  BRB
    !
    use chm_kinds
    use dimens_fcm
    use stream
    use timerm
    use replica_mod
    use exfunc,only:order5
    use memory
    use machutil,only:die
    use gopair, only : qgopair, ngopair, GoPair_exclusions
#if KEY_BLOCK==1
    use block_fcm, only : qblock_excld, nblock_excldPairs, block_exclusions
#endif
    implicit none
    !
    integer,allocatable,dimension(:) :: ipdrude,iblo14x,inb14x
    INTEGER NATOM,NATIML,NBOND,NNB,NNB14,MXNB14,MODE,IATMXB
    INTEGER IB(*),JB(*)
    INTEGER INB(*),IBLO(*),INB14(*),IBLO14(*)
    INTEGER NATBON(*),IATBON(IATMXB,*)
    INTEGER IPK(*),JPK(*),IPK14(*),JPK14(*)
    LOGICAL ISDRUDE(*)
    LOGICAL LEX14,CMPLTD
#if KEY_TSM==1
    LOGICAL LTSM
    INTEGER REACLS(*),PRODLS(*)
#endif 
    !
    EXTERNAL  EXCH5
    INTEGER MODEX,NPAIR,NPAIR4,MAXWRK,MXWRK4,ILAST
    INTEGER NX14,IATOM,NEXT14
    INTEGER I2,J2,I,J,K,IJ,IJA,IK,IKA,I2L,J2L,IBT,JBT
    LOGICAL OK

    CMPLTD=.FALSE.
    MODEX=IABS(MODE)
    IF(MODEX == 0) THEN
       IF(NNB14 <= 0) THEN
          DO I=1,NATOM
             IBLO14(I)=0
          ENDDO
       ENDIF
       IF(NNB14 /= IBLO14(NATOM)) THEN
          CALL WRNDIE(-1,'<MAKINB>', &
               'NBXMod=0 not allowed without existing exclusion list')
       ENDIF
       CMPLTD=.TRUE.
       RETURN
    ENDIF
    IF(MODEX > 5) MODEX=5
    !
    NPAIR=0
    MAXWRK=MXNB14
    !
    IF(MODEX == 5) THEN
       NPAIR4=0
       MXWRK4=MXNB14
    ENDIF
    !
    !     FOR MODE GREATER THAN ZERO INCLUDE THE EXISTING INB/IBLO LISTS,
    !
    IF(MODE > 0) THEN
       IF(IBLO(NATOM) /= NNB) THEN
          CALL WRNDIE(-4,'<MAKINB>','Bad exclusion list pointers')
       ENDIF
       ILAST=0
       DO I=1,NATOM
          IF (IBLO(I) > ILAST) THEN
             DO J=ILAST+1,IBLO(I)
                IF(I > INB(J)) THEN
                   I2=INB(J)
                   J2=I
                ELSE
                   I2=I
                   J2=INB(J)
                ENDIF
                IF (I2 > 0) THEN
                   NPAIR=NPAIR+1
                   IF(NPAIR > MAXWRK) THEN
                      !                       Ran out of space sb070507
                      IF(WRNLEV >= 2) WRITE(OUTU,988)
                      RETURN
                   ENDIF
                   IPK(NPAIR)=I2
                   JPK(NPAIR)=J2
                ENDIF
             ENDDO
             ILAST=IBLO(I)
          ENDIF
       ENDDO
    ENDIF
    !
    ! Compile a list of all the specified interactions
    !
    IF(MODEX > 1) THEN
       DO I=1,NBOND
          IF(IB(I) > JB(I)) THEN
             I2=JB(I)
             J2=IB(I)
          ELSE
             I2=IB(I)
             J2=JB(I)
          ENDIF
          IF(I2 > 0 .AND. J2 <= NATOM) THEN
             NPAIR=NPAIR+1
             IF(NPAIR > MAXWRK) THEN
                !                       Ran out of space sb070507
                IF(WRNLEV >= 2) WRITE(OUTU,988)
                RETURN
             ENDIF
             IPK(NPAIR)=I2
             JPK(NPAIR)=J2
          ENDIF
       ENDDO
    ENDIF
    !
    !     Next make a list of all the possible 1-3 and 1-4 interactions.
    !     This is based on the bond list.  First make a list of all bonds
    !     for every atom.
    !
    IF(MODEX > 2) THEN
       DO I=1,NATIML
          NATBON(I)=0
       ENDDO
       DO I=1,NBOND
          IBT=IB(I)
          JBT=JB(I)
          IF(IBT > 0 .AND. JBT > 0) THEN
             NATBON(IBT)=NATBON(IBT)+1
             IF(NATBON(IBT) > IATMXB) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,335) IBT
335             FORMAT(' <MAKINB>: Too many bonds for atom',I5, &
                     ' Check code')
                CALL DIEWRN(-4)
             ENDIF
             IATBON(NATBON(IBT),IBT)=I
             NATBON(JBT)=NATBON(JBT)+1
             IF(NATBON(JBT) > IATMXB) THEN
                IF(WRNLEV >= 2) WRITE(OUTU,335) JBT
                CALL DIE
             ENDIF
             IATBON(NATBON(JBT),JBT)=-I
          ENDIF
       ENDDO
       !
       !     Now make the unsorted list of 1-3 interactions by taking bonds
       !     and extending in a direction one bond.
       !
       DO I=1,NBOND
          IBT=IB(I)
          JBT=JB(I)
          DO J=1,NATBON(IBT)
             IJ=IATBON(J,IBT)
             IF(ABS(IJ)  >  I) THEN
                IF(IJ > 0) THEN
                   IJA=JB(IJ)
                ELSE
                   IJA=IB(ABS(IJ))
                ENDIF
                IF(IJA > JBT) THEN
                   I2=JBT
                   J2=IJA
                ELSE
                   I2=IJA
                   J2=JBT
                ENDIF
                IF (I2 > 0 .AND. I2 /= J2 .AND. J2 <= NATOM) THEN
                   NPAIR=NPAIR+1
                   IF(NPAIR > MAXWRK) THEN
                      !                       Ran out of space
                      IF(WRNLEV >= 2) WRITE(OUTU,988)
                      RETURN
                   ENDIF
                   IPK(NPAIR)=I2
                   JPK(NPAIR)=J2
                ENDIF
             ENDIF
          ENDDO
          DO J=1,NATBON(JBT)
             IJ=IATBON(J,JBT)
             IF(ABS(IJ)  >  I) THEN
                IF(IJ > 0) THEN
                   IJA=JB(IJ)
                ELSE
                   IJA=IB(ABS(IJ))
                ENDIF
                IF(IJA > IBT) THEN
                   I2=IBT
                   J2=IJA
                ELSE
                   I2=IJA
                   J2=IBT
                ENDIF
                IF(I2 > 0 .AND. I2 /= J2 .AND. J2 <= NATOM) THEN
                   NPAIR=NPAIR+1
                   IF(NPAIR > MAXWRK) THEN
                      !                       Ran out of space
                      IF(WRNLEV >= 2) WRITE(OUTU,988)
                      RETURN
                   ENDIF
                   IPK(NPAIR)=I2
                   JPK(NPAIR)=J2
                ENDIF
             ENDIF
          ENDDO
       ENDDO
    ENDIF

    !
    !
    !     Now make the unsorted list of 1-4 interactions by taking bonds
    !     and extending in each direction one bond.
    !
    IF(MODEX > 3) THEN
       DO I=1,NBOND
          IBT=IB(I)
          JBT=JB(I)
          DO J=1,NATBON(IBT)
             IJ=IATBON(J,IBT)
             IF(ABS(IJ)  /=  I) THEN
                IF(IJ > 0) THEN
                   IJA=JB(IJ)
                ELSE
                   IJA=IB(ABS(IJ))
                ENDIF
                DO K=1,NATBON(JBT)
                   IK=IATBON(K,JBT)
                   IF(ABS(IK)  /=  I) THEN
                      IF(IK > 0) THEN
                         IKA=JB(IK)
                      ELSE
                         IKA=IB(ABS(IK))
                      ENDIF
                      IF(IJA > IKA) THEN
                         I2=IKA
                         J2=IJA
                      ELSE
                         I2=IJA
                         J2=IKA
                      ENDIF
                      IF(I2 > 0 .AND. I2 /= J2 .AND. J2 <= NATOM) THEN
                         NPAIR=NPAIR+1
                         IF(NPAIR > MAXWRK) THEN
                            !                             Ran out of space
                            IF(WRNLEV >= 2) WRITE(OUTU,988)
                            RETURN
                         ENDIF
                         IPK(NPAIR)=I2
                         JPK(NPAIR)=J2
                         IF(MODEX == 5) THEN
                            NPAIR4=NPAIR4+1
                            IF(NPAIR4 > MXWRK4) THEN
                               !                                Ran out of space
                               IF(WRNLEV >= 2) WRITE(OUTU,988)
                               RETURN
                            ENDIF
                            IPK14(NPAIR4)=I2
                            JPK14(NPAIR4)=J2
                         ENDIF
                      ENDIF
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
       ENDDO
    ENDIF
    if(qgopair) then
       ! Add exclusions from explicit Go-odel pairs
       ! 1) Check that npair won't be blown then
       ! 2) Call GoPair_exclusions
       if( (npair+ngopair) > maxwrk) then
          npair = maxwrk + 1
          if(wrnlev >= 2) write(outu,988)
          return
       else
          call GoPair_exclusions(ipk, jpk, npair)
       endif
    endif
    !
#if KEY_BLOCK==1
    if(qblock_excld) then
       ! Add exclusions from explicit block pairs
       ! 1) Check that npair won't be blown then
       ! 2) Call block_exclusions
       if( (npair+nblock_excldPairs) > maxwrk) then
          npair = maxwrk + 1
          if(wrnlev >= 2) write(outu,988)
          return
       else
          call block_exclusions(ipk, jpk, npair)
       endif
    endif
#endif
    !
    !     Next sort the list of all the possible 1-4 interactions.
    !
    IF(MODEX == 5) THEN
       NPAIR4=NPAIR4+1
       IF(NPAIR4 > MXWRK4) THEN
          !           Ran out of space
          IF(WRNLEV >= 2) WRITE(OUTU,988)
          RETURN
       ENDIF
       IPK14(NPAIR4)=NATOM
       JPK14(NPAIR4)=NATOM
       CALL SORT(NPAIR4,EXCH5,ORDER5,IPK14,JPK14,0,0,0,0,0,2)
    ENDIF
    !
    ! Sort the pair list.
    !
    CALL SORT(NPAIR,EXCH5,ORDER5,IPK,JPK,0,0,0,0,0,2)
    !
    ! Process the sorted pair list to make inb. check that there are not
    ! multiple entries.
    !
    NNB14=0
    NX14=0
    I2L=0
    J2L=0
    IATOM=1
    NEXT14=1
    LEX14=.FALSE.
    DO I=1,NPAIR
       I2=IPK(I)
       J2=JPK(I)
       IF(MODEX == 5) THEN
          LEX14=I2 == IPK14(NEXT14) .AND. J2 == JPK14(NEXT14)
          IF(LEX14) NEXT14=NEXT14+1
       ENDIF
       !
       OK=.TRUE.
#if KEY_TSM==1
       IF(LTSM) THEN
          !           get rid of exclusions between reactants and products.
          IF(REACLS(J2) == 1 .AND. PRODLS(I2) == 1) OK=.FALSE.
          IF(REACLS(I2) == 1 .AND. PRODLS(J2) == 1) OK=.FALSE.
       ENDIF
#endif 
#if KEY_REPLICA==1
       IF (qRep) THEN
          !           get rid of exclusions between replicas
          IF (repID(repNoA(I2))  ==  repID(repNoA(J2)) .AND. &
               repNoA(I2)   /=        repNoA(J2)) OK=.FALSE.
       ENDIF
#endif 
       IF(OK) THEN
          IF(I2 == I2L .AND. J2 == J2L) THEN
             !             this is a repition. remove 1-4 sign if applied.
             IF(.NOT.LEX14 .AND. INB14(NNB14) < 0) THEN
                INB14(NNB14)=-INB14(NNB14)
                NX14=NX14-1
             ENDIF
          ELSE
             !             this is a new one.
             I2L=I2
             J2L=J2
             IF(I2 > IATOM) THEN
                DO J=IATOM,I2-1
                   IBLO14(J)=NNB14
                ENDDO
                IATOM=I2
             ENDIF
             NNB14=NNB14+1
             !
             IF(NNB14 > MXNB14) THEN
                !                Ran out of space
                IF(WRNLEV >= 2) WRITE(OUTU,988)
                RETURN
             ENDIF
             !
             IF(LEX14) THEN
                INB14(NNB14)=-J2
                NX14=NX14+1
             ELSE
                INB14(NNB14)=J2
             ENDIF
          ENDIF
       ENDIF
    ENDDO
    !
    DO J=IATOM,NATOM
       IBLO14(J)=NNB14
    ENDDO

    !------------------------------------------------------------------------------
    !   Map the 1-2, 1-3, 1-4 non-bonded exclusions of the heavy atoms to their
    !   associated drude particle
    !
    call chmalloc('nbexcl.src','MAKINB','ipdrude',natom,intg=ipdrude)
    call chmalloc('nbexcl.src','MAKINB','iblo14x',natom+10,intg=iblo14x)
    call chmalloc('nbexcl.src','MAKINB','inb14x',2*nnb14,intg=inb14x)

    call MAKDRUDEINB(natom,isdrude,ipdrude,iblo14,inb14, &
         nnb14,nx14,iblo14x,inb14x,MXNB14)

    call chmdealloc('nbexcl.src','MAKINB','ipdrude',natom,intg=ipdrude)
    call chmdealloc('nbexcl.src','MAKINB','iblo14x',natom+10,intg=iblo14x)
    call chmdealloc('nbexcl.src','MAKINB','inb14x',2*nnb14,intg=inb14x)


    !-------------Start added by Lei Huang ---------------
    IF(NNB14 > MXNB14) THEN
       !           Ran out of space
       IF(WRNLEV >= 2) WRITE(OUTU,988)
       RETURN
    ENDIF

#if KEY_LONEPAIR==1
    call MAKLONEPAIRINB(natom,iblo14,inb14,nnb14,nx14,MXNB14)
#endif 
    !-------------End   added by Lei Huang ---------------


    IF(NNB14 > MXNB14) THEN
       !           Ran out of space
       IF(WRNLEV >= 2) WRITE(OUTU,988)
       RETURN
    ENDIF


    !------------------------------------------------------------------------------

    CMPLTD=.TRUE.
    I=NNB14-NX14

    IF(PRNLEV >= 2) WRITE(OUTU,432) MODE,I,NX14
432 FORMAT(' <MAKINB> with mode',I4,' found',I7,' exclusions and',I7, &
         ' interactions(1-4)')
    !
    IF(PRNLEV > 8) THEN
       WRITE(OUTU,433) 'Nonbond exclusions pointers'
       WRITE(OUTU,435) (IBLO14(J),J=1,NATOM)
       WRITE(OUTU,433) 'Nonbond exclusions'
       WRITE(OUTU,435) (INB14(J),J=1,NNB14)
433    FORMAT(' MAKINB: ',A)
435    FORMAT(20I5)
    ENDIF
    !
    RETURN
    !yw
988 FORMAT(' <MAKINB>: Ran out of space. RESIZING')
    !
  END SUBROUTINE MAKINB

  SUBROUTINE MAKITC(NATOMX,NATOM,IAC,QPERT,IACP,IACNBP)
    !
    !-----------------------------------------------------------------------
    !      This routine sets up the arrays needed for the vdw calculation
    !           -  BRB  1/4/85, overhauled 3/19/98
    !
    use nb_module
    use chm_kinds
    use dimens_fcm
    use fast
    use inbnd
    use param
    use ffieldm
    use exfunc,only:order5
    use memory
    implicit none
    INTEGER  NATOMX
    LOGICAL  QPERT
    INTEGER  NATOM,IAC(*),IACP(*),IACNBP(*)
    !
    !
    INTEGER I,J
    integer :: alloc_err
    !sblrc
#if KEY_PERT==1
    INTEGER K
#endif 
    !sb
    !brb..07-FEB-99 Initialize ITC array when MMFF is in use
#if KEY_MMFF==1
    IF(FFIELD == MMFF) THEN
       DO I=1,NATC
          ITC(I)=I
       ENDDO
    ENDIF
#endif 
    !
    DO I=1,NATC
       ICCOUNT(I)=0
       LOWTP(I)=(I*(I-3))/2
    ENDDO
    !sblrc
#if KEY_PERT==1
#if KEY_LRVDW==1
    if (qpert) then
       DO I=1,NATC
          ICCOUNTM(I)=0
          ICCOUNTL(I)=0
       ENDDO
    endif
#endif 
#endif 
    !sb
    DO I=1,NATOM
       J=ITC(IAC(I))
       ICCOUNT(J)=ICCOUNT(J)-1
    ENDDO
#if KEY_PERT==1
    IF(QPERT) THEN
       DO I=1,NATOM
          J=ITC(IACP(I))
          !sblrc
          K=ITC(IAC(I))
          !sb
          ICCOUNT(J)=ICCOUNT(J)-1
          !sblrc        new: M ==> 1-lam, L ==> lam
#if KEY_LRVDW==1
          ICCOUNTM(J)=ICCOUNTM(J)-1         
#endif
#if KEY_LRVDW==1
          ICCOUNTL(K)=ICCOUNTL(K)-1         
#endif
          !sb
       ENDDO
    ENDIF
#endif 
    ! Sort atoms by frequency in the PSF (most frequent first).
    ! This will (hopefully) reduce the number of cache hits when
    ! processing vdw terms.  Create the tables in lower triangular form.
    !
    CALL SORTP(NATC,IGCNB,ORDER5,ICCOUNT,0,0,0,0,0,0,1)
    !
    DO I=1,NATC
       IF(ICCOUNT(IGCNB(I)) /= 0) THEN
          NITCC=I
          ICUSED(IGCNB(I))=I
       ENDIF
    ENDDO
    !
#if KEY_DEBUG==1
    write(6,88) 'natc,nitcc: ',natc,nitcc
88  format(A,2I5)
    write(6,89) 'iccount',(iccount(i),i=1,natc)
    write(6,89) 'igcnb  ',(igcnb(i),i=1,natc)
89  format(A/,20I5)
    write(6,89) 'icused ',(icused(i),i=1,natc)
#endif 
    !
    !
    ! Make the new IACNB arrays
    DO I=1,NATOMX
       IACNB(I)=ICUSED(ITC(IAC(I)))
    ENDDO
#if KEY_PERT==1
    IF(QPERT) THEN
       DO I=1,NATOMX
          IACNBP(I)=ICUSED(ITC(IACP(I)))
       ENDDO
    ENDIF
#endif 
    !
    NITCC2=(NITCC*(NITCC+1))/2
    IF(MOD(NITCC2,2) == 1) NITCC2=NITCC2+1  ! make sure it's even!!
    !
    ! Allocate space for the vdw coefficient tables
    !
    IF(lccnba < NITCC2*2) THEN
       if(allocated(ccnba))then
          call chmdealloc("nbexcl.src","makitc","ccnba",lccnba,crl=ccnba)
          call chmdealloc("nbexcl.src","makitc","ccnbb",lccnba,crl=ccnbb)
       endif
       LCCNBA=NITCC2*2
       call chmalloc("nbexcl.src","makitc","ccnba",lccnba,crl=ccnba)
       call chmalloc("nbexcl.src","makitc","ccnbb",lccnba,crl=ccnbb)
    ENDIF
    IF(LVSHFT) THEN           ! only allocate space for VSHIFT if necessary..
       IF(LCCNBD < NITCC2*2) THEN
          if(allocated(ccnbc))then
             lccnbd=size(ccnbc)
             call chmdealloc("nbexcl.src","makitc","ccnbc",lccnbd,crl=ccnbc)
             call chmdealloc("nbexcl.src","makitc","ccnbd",lccnbd,crl=ccnbd)
          endif
          LCCNBD=LCCNBA
          call chmalloc("nbexcl.src","makitc","ccnbc",lccnbd,crl=ccnbc)
          call chmalloc("nbexcl.src","makitc","ccnbd",lccnbd,crl=ccnbd)
       ENDIF
    ELSE
       IF(LCCNBD > 0) THEN
          if(allocated(ccnbc))then
             lccnbd=size(ccnbc)
             call chmdealloc("nbexcl.src","makitc","ccnbc",lccnbd,crl=ccnbc)
             call chmdealloc("nbexcl.src","makitc","ccnbd",lccnbd,crl=ccnbd)
          endif

       ENDIF
       LCCNBD=0
    ENDIF
    !
    ! Now fill the arrays...
    !
#if KEY_MMFF==1
    IF(FFIELD == MMFF) THEN
       CALL MAKITC_MM(NITCC,NITCC2,IGCNB,LOWTP,LVDW,LVSHFT,CTOFNB, &
            CCNBA,CCNBB,CCNBC, &
            CCNBD,GAMBUF,DELBUF)
    ELSE
#endif 
       CALL MAKITC2(NITCC,NITCC2,IGCNB,LOWTP,LVDW,LVSHFT,CTOFNB, &
            CCNBA,CCNBB,CCNBC,CCNBD)
#if KEY_MMFF==1
    ENDIF
#endif 
#if KEY_LRVDW==1
    !    COMPUTE LONG RANGE CORRECTIONS
    !sblrc
    CALL LRCOR(CCNBA,CCNBB,CCNBC,CCNBD,QPERT)
#endif 
    !
    RETURN
  END SUBROUTINE MAKITC

  SUBROUTINE MAKITC2(NITCC,NITCC2,IGCNB,LOWTP,LVDW,LVSHFT,CTOFNB, &
       CCNBA,CCNBB,CCNBC,CCNBD)

    !-----------------------------------------------------------------------
    !      This routine does the actual work of filling the vdw tables.
    !           - BRB    1/4/85, overhauled  3/19/98
    !
    use chm_kinds
    use dimens_fcm
    use param
#if KEY_CFF==1
    use ffieldm
    use cff_fcm
#endif 
    implicit none
    !
    INTEGER NITCC,NITCC2,IGCNB(*),LOWTP(*)
    LOGICAL LVDW,LVSHFT
    real(chm_real)  CTOFNB
    real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
    !
    !
    INTEGER I,J,IU,JU,IPT,JPT
    real(chm_real) CTOF2
    real(chm_real) CACX,CBCX,CADX,CBDX,CCDX,CA,CB,CC,CD
    !
    IF(LVDW) THEN
       !        vdw shifting coefficients
       IF(LVSHFT) THEN
          CTOF2=CTOFNB**2
          IF(CTOFNB < 50.0) THEN
             CACX=-2.0/CTOF2**9
             CBCX=1.0/CTOF2**6
             CADX=-1.0/CTOF2**6
             CBDX=1.0/CTOF2**3
             CCDX=CTOF2**3
          ELSE
             CACX=0.0
             CBCX=0.0
             CADX=0.0
             CBDX=0.0
             CCDX=0.0
          ENDIF
       ENDIF
       !
       ! Do van der waals
       !
       !     E = CA/R**12 - CB/R**6 - CC*R**6 + CD
       !
#if KEY_CFF==1
       IF(FFIELD == CFF)THEN
          JPT=0
          DO IU=1,NITCC
             DO JU=1,IU
                JPT=JPT+1
                IPT=MNO(IGCNB(IU),IGCNB(JU))
                CA=CNBA(IPT)
                CB=CNBB(IPT)
                CCNBA(JPT)=CA
                CCNBB(JPT)=CB
                !
                !     do 1-4 interaction vdw parameters
                !
                CCNBA(JPT+NITCC2)=CA
                CCNBB(JPT+NITCC2)=CB
             ENDDO
          ENDDO
       ELSE
#endif 
          JPT=0
          DO IU=1,NITCC
             DO JU=1,IU
                JPT=JPT+1
                IPT=MAX(IGCNB(IU),IGCNB(JU))
                IPT=LOWTP(IPT) + IGCNB(IU) + IGCNB(JU)
                CA=CNBB(IPT)*CNBA(IPT)**6
                CB=2.0*CNBB(IPT)*CNBA(IPT)**3
                CCNBA(JPT)=CA
                CCNBB(JPT)=CB
                IF(LVSHFT) THEN
                   CC=CACX*CA+CBCX*CB
                   CD=CADX*CA+CBDX*CB+CCDX*CC
                   CCNBC(JPT)=CC
                   CCNBD(JPT)=CD
                ENDIF
                !
                !     do 1-4 interaction vdw parameters
                !
                IPT=IPT+MAXCN
                CA=CNBB(IPT)*CNBA(IPT)**6
                CB=2.0*CNBB(IPT)*CNBA(IPT)**3
                CCNBA(JPT+NITCC2)=CA
                CCNBB(JPT+NITCC2)=CB
                IF(LVSHFT) THEN
                   CC=CACX*CA+CBCX*CB
                   CD=CADX*CA+CBDX*CB+CCDX*CC
                   CCNBC(JPT+NITCC2)=CC
                   CCNBD(JPT+NITCC2)=CD
                ENDIF
             ENDDO
          ENDDO
#if KEY_CFF==1
       ENDIF
#endif 
       !
    ELSE
       DO I=1,NITCC2
          CCNBA(I)=0.0
          CCNBB(I)=0.0
          IF(LVSHFT) THEN
             CCNBC(I)=0.0
             CCNBD(I)=0.0
          ENDIF
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE MAKITC2

  SUBROUTINE MAKITC_clr()
    !
    !-----------------------------------------------------------------------
    !      This routine clears up the arrays needed for the vdw calculation
    !           mfc added to c28a*
    !
    use nb_module
    use chm_kinds
    use dimens_fcm
    use fast
    use inbnd
    use param
    use ffieldm
    use memory
    implicit none
    INTEGER  NATOMX
    LOGICAL  QPERT
    !     INTEGER  NATOM,IAC(*),IACP(*),IACNBP(*)
    !
    IF(LCCNBA > 0) THEN
       if(allocated(ccnba))then
          call chmdealloc("nbexcl.src","makitc","ccnba",lccnba,crl=ccnba)
          call chmdealloc("nbexcl.src","makitc","ccnbb",lccnba,crl=ccnbb)
       endif
    ENDIF
    IF(LCCNBD > 0) THEN
       if(allocated(ccnbc))then
          call chmdealloc("nbexcl.src","makitc","ccnbc",lccnba,crl=ccnbc)
          call chmdealloc("nbexcl.src","makitc","ccnbd",lccnba,crl=ccnbd)
       endif
    ENDIF
    return
  end SUBROUTINE MAKITC_clr

  SUBROUTINE SUEXTAB(IBLO14,INB14,IGLO14,ING14, &
       NATOMX,NGRPX)
    ! This routine sets up the generation of the
    ! exclusion table. It calculates the dimensions
    ! of the atom and group exclusion tables and the
    ! necessary pointer arrays.
    ! It allocates the appropriate space
    ! and calls the table-generating
    ! subroutine (MKEXTAB).   -RJPetrella 1.6.99
    !
    use chm_kinds
    use dimens_fcm
    use mexclar
    use stream
    use memory
    implicit none
    INTEGER INB14(*),IBLO14(*),NATOMX
    INTEGER ING14(*),IGLO14(*),NGRPX
    ! local variables
    INTEGER AT1,GR1,NXI,NXIMAX
    INTEGER I,J
    INTEGER DIFF,MAXDIF,MAXDIFG
    INTEGER EXCL
    ! -- Added variables below--RJP
    !rjp      INTEGER EXDMMX,EXDMMG          !maximum size of table
    !rjp      PARAMETER(EXDMMX = 2*MAXA,EXDMMG = 2*MAXA)
    INTEGER,save :: SVEXCD,SVGEXC,SVNATO,SVNGRP !save allo sizes
    LOGICAL QEXTALL !true if allocated for excl table
    DATA QEXTALL/.FALSE./
    DATA SVEXCD/0/,SVGEXC/0/,SVNATO/0/,SVNGRP/0/
    ! eliminated below --RJP
    ! ------------------------------------------------------------
    !  END of variable declarations for SUEXTAB
    !      IF(PRNLEV > 2) WRITE(OUTU,'(A)')
    !     & 'SETTING UP ATOM EXCLUSION TABLE '
    EXCL = 0
    EXCDIM = 0
    DO AT1 = 1,NATOMX
       MAXDIF = 0
       DIFF = 0
       ! added amx below RJP 7.20.99
       IF(AT1 > 1) THEN
          NXI=IBLO14(AT1-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(AT1)
       DO J = NXI,NXIMAX
          EXCL = EXCL + 1
          DIFF = IABS(AT1-IABS(INB14(J)))
          IF (DIFF < MAXDIF) &
               CALL WRNDIE(-5,'<SUEXTAB>', &
               'EXCLUSIONS ARE OUT OF ORDER')
          MAXDIF = DIFF
       ENDDO
       EXCDIM = EXCDIM + MAXDIF
    ENDDO
    ! Added below--RJP 2.19.99
    !      IF (EXCDIM > EXDMMX) CALL WRNDIE(-5,
    !     & '<SUEXTAB>',
    !     & 'ATOM EXCLUSION TABLE TOO BIG')
    ! ----------------------------------------------------
    !      IF(PRNLEV > 2) THEN
    !        WRITE(OUTU,'(A,I9)')
    !     & 'NUMBER OF ATOM EXCLUSIONS = ',EXCL
    !        WRITE(OUTU,'(A,I9)')
    !     & 'NUMBER OF ELMNTS IN ATOM EXCL TABLE = ',EXCDIM
    !      ENDIF
    ! set up group exclusion table
    EXCL = 0
    GEXCDIM = 0
    !      IF(PRNLEV > 2) WRITE(OUTU,'(A)')
    !     & 'SETTING UP GROUP EXCLUSION TABLE '
    DO GR1 = 1,NGRPX
       MAXDIFG = 0
       DIFF = 0
       IF(GR1 > 1) THEN
          NXI=IGLO14(GR1-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IGLO14(GR1)
       DO J = NXI,NXIMAX
          EXCL = EXCL + 1
          DIFF = IABS(GR1-IABS(ING14(J)))
          IF (DIFF < MAXDIFG) &
               CALL WRNDIE(-5,'<SUEXTAB>', &
               'EXCLUSIONS ARE OUT OF ORDER')
          MAXDIFG = DIFF
       ENDDO
       GEXCDIM = GEXCDIM + MAXDIFG
    ENDDO
    !  Added below --RJP 2.19.99
    !      IF (GEXCDIM > EXDMMG) CALL WRNDIE(-5,
    !     & '<SUEXTAB>',
    !     & 'GROUP EXCLUSION TABLE TOO BIG')
    !----------------------------------------------------------
    !      IF(PRNLEV > 2) THEN
    !        WRITE(OUTU,'(A,I9)')
    !     & 'NUMBER OF GRP EXCLUSIONS = ',EXCL
    !        WRITE(OUTU,'(A,I9)')
    !     & 'NUMBER OF ELMNTS IN GRP EXCL TABLE = ',GEXCDIM
    !      ENDIF
    ! ------------- ALLOCATE WORK SPACE--------
    IF(QEXTALL) THEN !if exclusion table previously allocated
       if(allocated(HPAEXL)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPAEXL',SVEXCD,intg=HPAEXL)
       if(allocated(HPGEXL)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPGEXL',SVGEXC,intg=HPGEXL)
       if(allocated(HPAEXP)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPAEXP',SVNATO,intg=HPAEXP)
       if(allocated(HPGEXP)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPGEXP',SVNGRP,intg=HPGEXP)
       if(allocated(HPNEXA)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPNEXA',SVNATO,intg=HPNEXA)
       if(allocated(HPNEXG)) &
            call chmdealloc('nbexcl.src','SUEXTAB','HPNEXG',SVNGRP,intg=HPNEXG)
    ENDIF
    !
    call chmalloc('nbexcl.src','SUEXTAB','HPAEXL',EXCDIM,intg=HPAEXL)
    call chmalloc('nbexcl.src','SUEXTAB','HPGEXL',GEXCDIM,intg=HPGEXL)
    call chmalloc('nbexcl.src','SUEXTAB','HPAEXP',NATOMX,intg=HPAEXP)
    call chmalloc('nbexcl.src','SUEXTAB','HPGEXP',NGRPX,intg=HPGEXP)
    call chmalloc('nbexcl.src','SUEXTAB','HPNEXA',NATOMX,intg=HPNEXA)
    call chmalloc('nbexcl.src','SUEXTAB','HPNEXG',NGRPX,intg=HPNEXG)
    QEXTALL=.TRUE. !indicate allocation
    SVEXCD = EXCDIM
    SVGEXC = GEXCDIM
    SVNATO = NATOMX
    SVNGRP = NGRPX
    !
    CALL MKEXTAB(IBLO14,INB14,IGLO14,ING14, &
         HPAEXL,HPGEXL,HPAEXP, &
         HPGEXP,HPNEXA,HPNEXG, &
         NATOMX,NGRPX)
    RETURN
  END SUBROUTINE SUEXTAB

  SUBROUTINE MKEXTAB(IBLO14,INB14,IGLO14,ING14, &
       EXCLAR,EXCLARG,ATEXPT,GREXPT,NUEXCA,NUEXCG, &
       NATOMX,NGRPX)
    use chm_kinds
    use dimens_fcm
    use psf
    use mexclar
    use stream
    implicit none
    !
    !  This subroutine creates the EXCLUSION TABLE, which
    !  is a partially compressed two-dimensional table
    !  containing all the atom exclusions. The table is
    !  stored as a list, in which the exclusions for a given
    !  atom are indexed by the difference in atom numbers.
    !  Hence the size of the table for one of the N
    !  atoms in the system is the largest difference between
    !  the number of the atom and the number of any of its
    !  associated exclusions.
    !                                 --RJ Petrella 8.20.01
    !      Passed variables
    INTEGER INB14(*),IBLO14(*),NATOMX
    INTEGER ING14(*),IGLO14(*),NGRPX
    INTEGER EXCLAR(*),EXCLARG(*)
    INTEGER ATEXPT(*),GREXPT(*)
    INTEGER NUEXCA(*),NUEXCG(*)
    !      Local variables
    INTEGER AT1,NXI,NXIMAX,J,T,GR1
    !  eliminated 2 lines below --RJP 7.20.99
    INTEGER DIFF,BASENM
    !
    IF(PRNLEV > 2) THEN
       WRITE(OUTU,'(A)') &
            ' GENERATING ATOM EXCLUSION TABLE '
    ENDIF
    ! line below not necessary RJP 7.20.99
    DO J = 1,EXCDIM
       EXCLAR(J) = 1
    ENDDO
    BASENM = 0
    DO AT1 = 1,NATOM
       DIFF = 0
       ATEXPT(AT1) = BASENM
       IF(AT1 > 1) THEN
          NXI=IBLO14(AT1-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IBLO14(AT1)
       DO J = NXI,NXIMAX
          DIFF = IABS(AT1-IABS(INB14(J)))
          IF (INB14(J) < 0) THEN
             EXCLAR(BASENM+DIFF) = -1
          ELSE
             EXCLAR(BASENM+DIFF) = 0
          ENDIF
       ENDDO
       NUEXCA(AT1) = DIFF
       BASENM = BASENM + DIFF
    ENDDO
    !
    IF(PRNLEV > 2) THEN
       WRITE(OUTU,'(A)') &
            ' GENERATING GROUP EXCLUSION TABLE '
    ENDIF
    DO J = 1,GEXCDIM
       EXCLARG(J) = 1
    ENDDO
    BASENM = 0
    DO GR1 = 1,NGRP
       DIFF = 0
       GREXPT(GR1) = BASENM
       IF(GR1 > 1) THEN
          NXI=IGLO14(GR1-1)+1
       ELSE
          NXI=1
       ENDIF
       NXIMAX=IGLO14(GR1)
       DO J = NXI,NXIMAX
          DIFF = IABS(GR1-IABS(ING14(J)))
          IF (ING14(J) < 0) THEN
             EXCLARG(BASENM+DIFF) = -1
          ELSE
             EXCLARG(BASENM+DIFF) = 0
          ENDIF
       ENDDO
       NUEXCG(GR1) = DIFF
       BASENM = BASENM + DIFF
    ENDDO
    QEXCTB = .TRUE.
    RETURN
  END SUBROUTINE MKEXTAB

#if KEY_LRVDW==1 /*lrvdw_lrcor*/
  
  !sblrc, June 2006, making routine LRCOR compatible with PERT
  !     Some comments: (1) This code ought to be cleaned up (who
  !         wrote/contributed this originally); in particular, some
  !         lines (pointed out below) are never needed
  !     (2) Any PERT related code is ##IF PERT
  !         protected; in addition, it is bracketed by csblrc / csb pairs
  !
  !sb      SUBROUTINE LRCOR(CCNBA,CCNBB,CCNBC,CCNBD) !,LRC)
  SUBROUTINE LRCOR(CCNBA,CCNBB,CCNBC,CCNBD,QPERT) !,LRC)
    !...  use nb_module
    use chm_kinds
    use dimens_fcm
    use fast
    use psf
    use inbnd
    use param
    use ffieldm
    use consta
    use number
    implicit none
    INTEGER  NATOMX,itype,jtype,itemp,jtemp,ni,nj,IVECT
    LOGICAL  QPERT
    !      INTEGER  NATOM,IAC(*),IACP(*),IACNBP(*)
    real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*) !LRC,rcut
    !
    real(chm_real) ct3,ct9,prefac1,prefac2,prefac3,prefac4
    !sb      real(chm_real) lrcn,lrcp,lrcap,lrcbp,ninth
    real(chm_real) lrcn,lrcp,ninth
    real(chm_real) lrcn1,lrcp1
    !sblrc
#if KEY_PERT==1
    integer nip,njp,ivec2
#endif 
    !sb
    INTEGER count,count_P
    real(chm_real) sumOfAs, sumOfBs, LJAvgA, LJAvgB
    real(chm_real) sumOfAs_P, sumOfBs_P, LJAvgA_P, LJAvgB_P
    real(chm_real) rcut,rcut2, rcut3, rcut4, rcut5
    real(chm_real) rswitch,rswitch2, rswitch3, rswitch4, rswitch5

    LRC  = 0.0
    LRC2 = 0.0

    ct3 = one/CTofNB**3
    ct9 = CT3**3
    ninth=third*third
    LRCa  = 0.0
    LRCb  = 0.0
#if KEY_PERT==1
    IF (QPERT) THEN
       LRCAP  = ZERO
       LRCBP  = ZERO
    ENDIF
#endif 

    DO itemp = 1,natc
       DO jtemp = itemp,natc ! 1,itemp !  natc

          !sblrc if PERT is active, we need to do accounting for l=0 and l=1
          !     vdw types separately.
#if KEY_PERT==1
          if (.not.qpert) then
#endif 
             ni = abs(iccount(itemp))
             nj = abs(iccount(jtemp))
#if KEY_PERT==1
#if KEY_LRVDW==1
          else
             ni = abs(iccountl(itemp))
             nj = abs(iccountl(jtemp))
             nip = abs(iccountm(itemp))
             njp = abs(iccountm(jtemp))
          endif
#endif 
#endif 
          !sb
          itype=icused(itemp)
          jtype=icused(jtemp)

          IF ( (ni > 0).and.(nj > 0)) THEN
             IVECT = LOWTP(MAX(jtype,itype))+jtype+itype
             LRCb = LRCb - ni*nj*CCNBB(IVECT)
             lrca = lrca + ni*nj*CCNBA(IVECT)
             !sb   lines below are not used as far as I can tell ...
             LRC = LRC - ((TWOPI/3.0)*ni*nj*CCNBB(IVECT)/CTofNB**3) + &
                  ((TWOPI/9.0)*ni*nj*CCNBA(IVECT)/CTofNB**9)

             LRC2 = LRC2 - ((4.0*PI*ni*nj*CCNBB(IVECT)/CTofNB**3)) + &
                  (8.0*PI*ni*nj*CCNBA(IVECT)/(3.0*CTofNB**9))
             !sb ... end of unused code

          ENDIF
          !sblrc
#if KEY_PERT==1
          IF ((QPERT).AND.((NIP > 0).AND.(NJP > 0))) THEN
             IVEC2 = LOWTP(MAX(JTYPE,ITYPE))+JTYPE+ITYPE
             LRCBP = LRCBP - NIP*NJP*CCNBB(IVEC2)
             LRCAP = LRCAP + NIP*NJP*CCNBA(IVEC2)
          ENDIF
#endif 
          !sb
       ENDDO
    ENDDO
    !sb more lines that (apparently) are not used ...
    LRC2 = LRC2/3.0

    lrcn = (lrcb + lrca*ct3*ct3*third) *lrvdw_const*ct3
    lrcp = two*lrcn
    !sb ... end of unused code

    ! start the code for LRC of VDW using Shirts's algorithm. -lei Huang
    rcut = CTofNB
    rcut2= CTofNB * CTofNB
    rcut3= rcut2 * CTofNB
    rcut4= rcut2 * rcut2
    rcut5= rcut2 * rcut3
    rswitch  = ctonnb
    rswitch2 = ctonnb * ctonnb
    rswitch3 = rswitch2 * ctonnb
    rswitch4 = rswitch2 * rswitch2
    rswitch5 = rswitch2 * rswitch3

#if KEY_PERT==1
    IF (QPERT) THEN
       count = 0
       count_P = 0
       sumOfAs = 0.0
       sumOfAs_P = 0.0
       sumOfBs = 0.0
       sumOfBs_P = 0.0

       DO itemp = 1,natc
          ni = abs(iccountl(itemp))
          nip = abs(iccountm(itemp))
          DO jtemp = itemp,natc
             nj = abs(iccountl(jtemp))
             njp = abs(iccountm(jtemp))
             itype=icused(itemp)
             jtype=icused(jtemp)
             ! APH 5/6/2014: Added this safety check because lowtp starts from index 1
             if (MAX(jtype,itype) == 0) then
                IVECT = 0
             else   
                IVECT = LOWTP(MAX(jtype,itype))+jtype+itype
             endif

             IF( (ni > 0) .AND. (nj > 0) ) THEN
                IF(itemp == jtemp) THEN
                   sumOfAs = sumOfAs + (ni - 1) * nj * CCNBA(IVECT)
                   sumOfBs = sumOfBs + (ni - 1) * nj * CCNBB(IVECT)
                   count = count + (ni - 1) * nj
                ELSE
                   sumOfAs = sumOfAs + 2.0*ni*nj*CCNBA(IVECT)
                   sumOfBs = sumOfBs + 2.0*ni*nj*CCNBB(IVECT)
                   count = count + 2.0*ni*nj
                ENDIF
             ENDIF

             IF( (nip > 0) .AND. (njp > 0) ) THEN
                IF(itemp == jtemp) THEN
                   sumOfAs_P = sumOfAs_P + (nip - 1) * njp * CCNBA(IVECT)
                   sumOfBs_P = sumOfBs_P + (nip - 1) * njp * CCNBB(IVECT)
                   count_P = count_P + (nip - 1) * njp
                ELSE
                   sumOfAs_P = sumOfAs_P + 2.0*nip*njp*CCNBA(IVECT)
                   sumOfBs_P = sumOfBs_P + 2.0*nip*njp*CCNBB(IVECT)
                   count_P = count_P + 2.0*nip*njp
                ENDIF
             ENDIF

          ENDDO
       ENDDO

       LJAvgA = sumOfAs / count
       LJAvgB = sumOfBs / count

       LJAvgA_P = sumOfAs_P / count_P
       LJAvgB_P = sumOfBs_P / count_P

       lrvdw_const_ms_m = (16*nAtom*nAtom*PI*(-105*LJAvgB_P*rcut5*rswitch5 +   &
            LJAvgA_P*(3*rcut4 + 9*rcut3*rswitch + 11*rcut2*rswitch2 +         &
            9*rcut*rswitch3 + 3*rswitch4)))/(315*rcut5*rswitch5*            &
            ((rcut + rswitch)*(rcut + rswitch)*(rcut + rswitch)))
       lrvdw_const_ms_l = (16*nAtom*nAtom*PI*(-105*LJAvgB*rcut5*rswitch5 +   &
            LJAvgA*(3*rcut4 + 9*rcut3*rswitch + 11*rcut2*rswitch2 +         &
            9*rcut*rswitch3 + 3*rswitch4)))/(315*rcut5*rswitch5*            &
            ((rcut + rswitch)*(rcut + rswitch)*(rcut + rswitch)))

    ELSE
#endif 
       ! no pert, regular simulation
       count = 0
       sumOfAs = 0.0
       sumOfBs = 0.0
       DO itemp = 1,natc
          ni = abs(iccount(itemp))
          IF(ni > 0) THEN
             DO jtemp = itemp,natc
                nj = abs(iccount(jtemp))
                IF(nj > 0) THEN
                   itype=icused(itemp)
                   jtype=icused(jtemp)
                   IVECT = LOWTP(MAX(jtype,itype))+jtype+itype

                   IF(itemp == jtemp) THEN
                      sumOfAs = sumOfAs + (ni - 1) * nj * CCNBA(IVECT)
                      sumOfBs = sumOfBs + (ni - 1) * nj * CCNBB(IVECT)
                      count = count + (ni - 1) * nj
                   ELSE
                      sumOfAs = sumOfAs + 2.0*ni*nj*CCNBA(IVECT)
                      sumOfBs = sumOfBs + 2.0*ni*nj*CCNBB(IVECT)
                      count = count + 2.0*ni*nj
                   ENDIF
                ENDIF
             ENDDO
          ENDIF
       ENDDO

       LJAvgA = sumOfAs / count
       LJAvgB = sumOfBs / count

       lrvdw_const_ms = (16*nAtom*nAtom*PI*(-105*LJAvgB*rcut5*rswitch5 +   &
            LJAvgA*(3*rcut4 + 9*rcut3*rswitch + 11*rcut2*rswitch2 +         &
            9*rcut*rswitch3 + 3*rswitch4)))/(315*rcut5*rswitch5*            &
            ((rcut + rswitch)*(rcut + rswitch)*(rcut + rswitch)))

#if KEY_PERT==1
    ENDIF
#endif 


    ! end   the code for LRC of VDW using Shirts's algorithm  -lei Huang


    RETURN
  END SUBROUTINE LRCOR
#endif /* (lrvdw_lrcor)*/

  SUBROUTINE MAKDRUDEINB(natom,isdrude,ipdrude,iblo14,inb14, &
       nnb14,nx14,iblo14x,inb14x,MXNB14)
    use stream
    integer natom, ipdrude(*)
    logical isdrude(*)
    integer nx14
    integer iblo14(*),  inb14(*),  nnb14
    integer iblo14x(*), inb14x(*), nnb14x
    integer i,j,j2, sign
    integer MXNB14

    !     Construct a temporary list of polarizable atom possessing a drude (ipdrude=+1),
    !     of their drudes (ipdrude=-1), and of the rest (ipdrude=0)
    !
    do i = 1,natom
       ipdrude(i) = 0
    enddo

    do i = 1,natom
       if(isdrude(i))then
          ipdrude(i-1) = 1  !heavy atom
          ipdrude(i) = -1   !drude particle
       endif
    enddo

    if(prnlev >= 10) &
         write(outu,'(a,i10,a)') 'Call MAKDRUDEINB for ',natom,' atoms'

    nnb14x = 0
    j2=1

    do i=1,natom
       iblo14x(i) = 0
    enddo

    do i=1,natom

       iblo14x(i) = nnb14x

       IF(NNB14x > MXNB14)THEN
          nnb14 = nnb14x
          write(*,*) NNB14x,nnb14,MXNB14
          RETURN  ! Ran out of space
       ENDIF

       if(ipdrude(i) == 0)then

          if(prnlev > 10) &
               write(*,*) ' atom  ',i,ipdrude(i)

          do j=j2,iblo14(i)

             if(ipdrude(abs(inb14(j))) == 0)then
                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = inb14(j)

             elseif(ipdrude(abs(inb14(j))) == 1)then
                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = inb14(j)
                sign = inb14(j)/abs(inb14(j))
                if(prnlev > 10) &
                     write(*,*) '     +its drude', &
                     (abs(inb14(j))+1)*sign
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = (abs(inb14(j))+1)*sign

             endif

          enddo

       elseif(ipdrude(i) == 1)then

          if(prnlev > 10) &
               write(*,*) ' atom  ',i,ipdrude(i)
          if(prnlev > 10) &
               write(*,*) '  add pair: ',i,i+1,ipdrude(i+1)
          nnb14x = nnb14x + 1
          iblo14x(i) = iblo14x(i) + 1
          inb14x(nnb14x) = i + 1

          do j=j2,iblo14(i)

             if(ipdrude(abs(inb14(j))) == 0)then

                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = inb14(j)

             elseif(ipdrude(abs(inb14(j))) == 1)then

                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = inb14(j)
                sign = inb14(j)/abs(inb14(j))
                if(prnlev > 10) &
                     write(*,*) '     +its drude', &
                     (abs(inb14(j))+1)*sign
                nnb14x = nnb14x + 1
                iblo14x(i) = iblo14x(i) + 1
                inb14x(nnb14x) = (abs(inb14(j))+1)*sign

             endif

          enddo

          if(prnlev > 10) write(*,*)
          if(prnlev > 10) &
               write(*,*) '+drude ',i+1,ipdrude(i+1)

          do j=j2,iblo14(i)

             if(ipdrude(abs(inb14(j))) == 0)then

                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i+1,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i+1) = iblo14x(i+1) + 1
                inb14x(nnb14x) = inb14(j)

             elseif(ipdrude(abs(inb14(j))) == 1)then

                if(prnlev > 10) &
                     write(*,*) '  add pair: ',i+1,inb14(j),ipdrude(abs(inb14(j)))
                nnb14x = nnb14x + 1
                iblo14x(i+1) = iblo14x(i+1) + 1
                inb14x(nnb14x) = inb14(j)
                sign = inb14(j)/abs(inb14(j))
                if(prnlev > 10) &
                     write(*,*) '     +its drude', &
                     (abs(inb14(j))+1)*sign
                nnb14x = nnb14x + 1
                iblo14x(i+1) = iblo14x(i+1) + 1
                inb14x(nnb14x) = (abs(inb14(j))+1)*sign

             endif

          enddo

       endif

       j2=IBLO14(i)+1

       if(prnlev > 10) &
            write(*,'(A,3I8)') 'iblo14x, nnb14x ',i,iblo14x(i),nnb14x

    enddo

    IF(prnlev > 10)THEN
       write(outu,'(a,i10,a)') 'Call MAKDRUDEINB for ',natom,' atoms'
       write(outu,'(a,i10)') 'MAKDRUDEINB excluded list ',nnb14x
       j2=1
       do i=1,natom
          write(*,'(A,4I8)') 'atom ',i,ipdrude(i),j2,iblo14x(i)
          do j=j2,iblo14x(i)
             write(*,*) inb14x(j),ipdrude(abs(inb14x(j)))
          enddo
          j2=iblo14x(i)+1
          write(*,*)
       enddo
    ENDIF

    ! copy back onto the original array
    if(prnlev >= 8) &
         write(*,*) 'number of pairs ', nnb14 , nnb14x
    nnb14 = nnb14x
    IF(NNB14x > MXNB14)then
       write(*,*) NNB14x,nnb14,MXNB14
       RETURN  ! Ran out of space
    ENDIF

    do i=1,natom
       iblo14(i) = iblo14x(i)
    enddo

    nx14=0
    do i=1,nnb14x
       inb14(i) = inb14x(i)
       if(inb14(i) < 0) nx14=nx14+1
       !     write(*,*) i,'iblo14(1) ',iblo14(1), iblo14x(1),
       !    &              inb14(i), inb14x(i)
    enddo

    IF(prnlev > 10)THEN
       write(*,*)
       write(*,*) 'Exclusion list'
       j2=1
       do i=1,natom
          write(*,'(A,4I8)') 'atom ',i,ipdrude(i),j2,iblo14(i)
          do j=j2,iblo14(i)
             write(*,*) inb14(j),ipdrude(abs(inb14(j)))
          enddo
          j2=iblo14(i)+1
          write(*,*)
       enddo
    ENDIF
    RETURN
  END SUBROUTINE MAKDRUDEINB


  !-------------Start added by Lei Huang ---------------
  !---- Sort abs(ARRAY(1..N)) in ascending order via Quicksort ---------
  !Modified from qsort_ints() by Lei Huang
  subroutine qsort_absints(ARRAY, N)
    integer,intent(in) :: N
    integer,dimension(N),intent(inout) :: ARRAY

    logical :: DONE
    integer :: RECURS
    real :: RNDVAL
    ! array indices
    integer,dimension(N) :: BOUNDS
    integer :: LO, HI, PRL, PRH, RNDIND
    ! array element values
    integer :: SWP, PIVOT
    !
    !     In any given iteration of Quicksort the task is to sort the
    !     subarray in the range of indices LO..HI.
    !
    !     The basic operation is "partitioning": First we choose a
    !     PIVOT, which is a value that hopefully will split the array
    !     into two pieces of roughly equal size.  Using a pair of
    !     cursors PRL and PRH we sweep from the edges and proceed
    !     inward, swapping elements between the two cursors as
    !     necessary until they meet in the middle.  At that point,
    !     everything above the cursors exceeds PIVOT, and everthing
    !     below the cursor is less than PIVOT.
    !
    !     More precisely:  at the end of the partitioning, we must
    !     end up with ARRAY(LO..PRL)  <=  PIVOT, and
    !     ARRAY(PRH+1..HI)  >=  PIVOT.  If PIVOT happens to be the
    !     largest element in ARRAY, then (PRL,PRH) should end up equal to
    !     (HI,HI).  If PIVOT happens to be the smallest
    !     element in ARRAY, then (PRL,PRH) should end up equal to
    !     (LO,LO).
    !
    !     This particular implementation is taken from Sedgewick,
    !     "Algorithms", Addison-Wesley, 1988, p. 118.  It is modified
    !     so that arrays of length 2 or less are handled as special
    !     cases, and that pivots are chosen at random.
    !
    !     After partitioning we sort the arrays LO..PRL and
    !     PRH+1..HI recursively.  Since FORTRAN is not recursive,
    !     we must emulate recursion using a stack.  I use BOUNDS for
    !     this:  I save the bounds of one subarray in a pair of
    !     locations in ARRAY, then go on to do the other subarray
    !     immediately.
    !
    LO = 1
    HI = N

    RECURS = 0
    !
    !     At the top of the loop we have a number of subarrays to sort:
    !     LO..HI, and anything on the stack.  Our top priority
    !     is to sort LO..HI; the stack contains deferred tasks.
    !     The body of the loop is a giant IF-ELSE:  either
    !     LO..HI is small enough to be done by a direct method
    !     or it must be done by recursive Quicksort.  In the former
    !     case, we end by grabbing a new task from the stack.
    !
    !     None of the loops in this sort are vectorizable.
    !
    DONE = .FALSE.
    DO WHILE (.NOT. DONE)
       !
       IF (HI - LO <= 1) THEN

          !           If exactly two elements, swap if necessary
          IF (HI - LO == 1) THEN ! Exactly two elems
             IF (abs(ARRAY(LO)) > abs(ARRAY(HI))) THEN
                SWP = ARRAY(LO)
                ARRAY(LO) = ARRAY(HI)
                ARRAY(HI) = SWP
             ENDIF
          ENDIF

          !           LO..HI is done, so grab next task off stack
          DONE = (RECURS < 2)
          IF (.NOT. DONE) THEN
             LO = BOUNDS(RECURS - 1)
             HI = BOUNDS(RECURS)
             RECURS = RECURS - 2
          ENDIF

       ELSE

          !           Choose PIVOT at random from among elements in subarray
          CALL RANDOM_NUMBER(RNDVAL)
          RNDIND = INT((HI - LO + 0.9) * RNDVAL) + LO
          PIVOT = ARRAY(RNDIND)
          ! Move PIVOT to HI so we're free to rearrange other elements
          ARRAY(RNDIND) = ARRAY(HI)
          ARRAY(HI) = PIVOT

          !           Partition subarray LO..HI-1 about PIVOT by
          !           sweeping inward from edges using cursors PRL and PRH
          PRL = LO
          PRH = HI - 1
          DO
             DO WHILE (PRL < HI .AND. abs(ARRAY(PRL)) < abs(PIVOT))
                PRL = PRL + 1
             ENDDO
             DO WHILE (PRH > LO .AND. abs(ARRAY(PRH)) > abs(PIVOT))
                PRH = PRH - 1
             ENDDO
             IF (PRL >= PRH) EXIT
             ! elements PRL and PRH are each on the wrong side of PIVOT
             SWP = ARRAY(PRL)
             ARRAY(PRL) = ARRAY(PRH)
             ARRAY(PRH) = SWP
             PRL = PRL + 1
             PRH = PRH - 1
          ENDDO
          IF (PRL < HI) THEN
             ! restore PIVOT to its proper place
             SWP = ARRAY(PRL)
             ARRAY(PRL) = ARRAY(HI)
             ARRAY(HI) = SWP
          ENDIF

          !           Save larger subarray on stack for later sorting;
          !           return to sort smaller one immediately
          RECURS = RECURS + 2
          IF (PRL - LO > HI - PRL) THEN
             BOUNDS(RECURS - 1) = LO
             BOUNDS(RECURS) = PRL - 1
             LO = PRL + 1
          ELSE
             BOUNDS(RECURS - 1) = PRL + 1
             BOUNDS(RECURS) = HI
             HI = PRL - 1
          ENDIF

       ENDIF

    ENDDO
  end subroutine qsort_absints


  !lonepair.src

  !NUMLP - # of lone pair
  !NUMLPH = # of items in LPHOST()
  !LPHOST() - the list of all lone pairs and their hosts
  !LPNHOST(I) - the list of # of atoms as the host of each lone pair
  !LPHPTR(I) - the pointer to the first item (lone pair) in LPHOST()

  ! to introduce the exclusion list of host atoms to the exclusion list of the lone pair
#if KEY_LONEPAIR==1 /*lp*/
  SUBROUTINE MAKLONEPAIRINB(natom,iblo14,inb14,nnb14,nx14,MXNB14)
    use lonepr
    use stream
    IMPLICIT NONE

    !  INTEGER natom, prnlev
    INTEGER natom
    INTEGER iblo14(*), inb14(*), nnb14, nnb14x, nx14
    INTEGER i,j,k,j2,i_LP, i_LP_2, j_LP, NumHosts
    INTEGER Atm_Host_J, Atm_LP_1, Atm_LP_2, sign
    INTEGER Atm_Idx_Host, Atm_Idx_LP, i_N_LP, j_N_LP
    INTEGER MXNB14, iError
    INTEGER, PARAMETER :: N_MAX_LP = 4
    INTEGER, PARAMETER :: N_MAX_Exc = 256
    !  INTEGER, PARAMETER :: N_MAX_Exc = 5

    TYPE ROW
       INTEGER, DIMENSION(:), POINTER :: List
    END TYPE ROW
    TYPE LP_LIST
       INTEGER N_LP ! the number of lone pair involved for this host. Also as flags. >= 1 - host; -1 - LP; == 0 - others
       INTEGER, DIMENSION(N_MAX_LP) :: AtomLP = (/0,0,0,0/) ! the index of LP atom out of nAtom
    END TYPE LP_LIST
    TYPE (ROW), DIMENSION(:), ALLOCATABLE :: inb14x ! arrays of type ROW,
    TYPE (LP_LIST), DIMENSION(:), ALLOCATABLE :: Host_LP ! the number of LP for this atom and LP list
    integer, DIMENSION(:), ALLOCATABLE :: ListNumofExcl ! number of items in exclusion list
    integer, DIMENSION(:), ALLOCATABLE :: ArrayLenList ! record the length of inb14x(i)%List()
    integer, DIMENSION(:), POINTER :: pBuff ! the pointer to a temprary buffer used for resizing memory of inb14x(i)%List()
    integer nAddSize ! the size of memory increased when resizing memory

!!$    interface ReSizeMem
!!$       SUBROUTINE ReSizeMem(lpMem, lpBuff, nAtom, nCurLen, nAddSize)
!!$         INTEGER, DIMENSION(:), POINTER :: lpMem ! the pointer to the existing array
!!$         INTEGER, DIMENSION(:), POINTER :: lpBuff ! the pointer to the buffer
!!$         INTEGER nAtom, nCurLen, nNewLen, nAddSize, i
!!$       END SUBROUTINE ReSizeMem
!!$    end interface


    !  prnlev = 11

    !  NumHosts = NUMLPH - NUMLP ! wrong. One lone pair has only one host here.
    nAddSize = N_MAX_Exc
    NumHosts = NUMLP

    ALLOCATE(pBuff(natom), ArrayLenList(natom), STAT=iError)
    if (iError > 0) CALL wrndie(-5, '<MAKLONEPAIRINB>', &
         'Failed to allocate memory: pBuff(natom), ArrayLenList(natom)')

    ALLOCATE(ListNumofExcl(natom), STAT=iError)
    if (iError > 0) CALL wrndie(-5, '<MAKLONEPAIRINB>', &
         'Failed to allocate memory: ListNumofExcl(natom)')

    ALLOCATE(Host_LP(natom), inb14x(natom), STAT=iError)
    if (iError > 0) CALL wrndie(-5, '<MAKLONEPAIRINB>', &
         'Failed to allocate memory: Host_LP(natom), inb14x(natom)')

    DO i = 1, natom
       ArrayLenList(i) = min(natom+1-i, N_MAX_Exc)
       ALLOCATE (inb14x(i)%List(1:ArrayLenList(i)), STAT=iError) ! allocate storage for each row
       if (iError > 0) CALL wrndie(-5, '<MAKLONEPAIRINB>', &
            'Failed to allocate memory: inb14x(i)%List(ArrayLenList(i))')
    END DO


    DO i = 1, natom
       Host_LP(i)%N_LP = 0 ! default as others
    END DO

    ! start to set up the flags for host, LP and others; then build LP list for each host
    DO i = 1, NUMLP
       Atm_Idx_LP = LPHOST(LPHPTR(i))
       !   DO j = 1, LPNHOST(i) ! considering all related atoms are hosts. Not correct. Only the first one is used.
       DO j = 1, 1          ! only the first host atom is the real host
          Atm_Idx_Host = LPHOST(LPHPTR(i)+j)
          Host_LP(Atm_Idx_Host)%N_LP = Host_LP(Atm_Idx_Host)%N_LP + 1
          Host_LP(Atm_Idx_Host)%AtomLP(Host_LP(Atm_Idx_Host)%N_LP) = Atm_Idx_LP
       END DO

       Host_LP(Atm_Idx_LP)%N_LP = -1 ! the flag for LP
    END DO
    ! end to set up the flags for host, LP and others; then build LP list for each host


    if(prnlev >= 10) &
         write(*,'(a,i10,a)') 'Call MAKLONEPAIRINB for ',natom,' atoms'

    DO i=1,natom
       ListNumofExcl(i) = 0
    END DO
    nnb14x = 0 ! to store the total length of exclusion list

    j2=1
    DO i=1,natom ! map all exclusion list from inb14() to the new list, inb14x(,)
       IF(nnb14x > MXNB14)THEN
          nnb14 = nnb14x
          write(*,*) nnb14x,nnb14,MXNB14

          DEALLOCATE(ListNumofExcl)
          DO j = 1, natom
             DEALLOCATE (inb14x(j)%List)
          END DO
          DEALLOCATE(Host_LP, inb14x)
          DEALLOCATE(pBuff, ArrayLenList)

          RETURN  ! Ran out of space
       ENDIF

       i_N_LP = Host_LP(i)%N_LP
       IF(i_N_LP == 0) THEN ! atom i is other than host and LP. other
          DO j=j2,iblo14(i)
             Atm_Host_J = abs(inb14(j))
             j_N_LP = Host_LP(Atm_Host_J)%N_LP

             IF(j_N_LP == 0) THEN ! atom j is other than host and LP. other
                ListNumofExcl(i) = ListNumofExcl(i) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                   CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                ENDIF

                inb14x(i)%List(ListNumofExcl(i)) = inb14(j) ! other - other
             ELSEIF(j_N_LP > 0) THEN ! atom j is a host atom
                ListNumofExcl(i) = ListNumofExcl(i) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                   CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                ENDIF

                inb14x(i)%List(ListNumofExcl(i)) = inb14(j) ! i - j; other - host

                DO j_LP=1, j_N_LP ! add all associated other - LP
                   Atm_Idx_LP = Host_LP(Atm_Host_J)%AtomLP(j_LP)

                   IF(Atm_Idx_LP < i) THEN ! add to list of atom Atm_Idx_LP
                      ListNumofExcl(Atm_Idx_LP) = ListNumofExcl(Atm_Idx_LP) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_Idx_LP) > ArrayLenList(Atm_Idx_LP)) THEN
                         CALL ReSizeMem(inb14x(Atm_Idx_LP)%List, pBuff, natom, ArrayLenList(Atm_Idx_LP), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(Atm_Idx_LP)%List(ListNumofExcl(Atm_Idx_LP)) = i*sign ! i - j; other - host's LP
                   ELSE ! add to list of atom i
                      ListNumofExcl(i) = ListNumofExcl(i) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                         CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(i)%List(ListNumofExcl(i)) = Atm_Idx_LP*sign ! i - j; other - host's LP
                   ENDIF
                ENDDO
             ENDIF
          ENDDO

       ELSEIF(i_N_LP > 0) THEN ! atom i is a host atom
          DO i_LP=1, i_N_LP ! add all host/LPs and local LP/LP
             Atm_LP_1 = Host_LP(i)%AtomLP(i_LP)

             IF(Atm_LP_1 < i) THEN ! add to Atm_LP_1's list
                ListNumofExcl(Atm_LP_1) = ListNumofExcl(Atm_LP_1) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(Atm_LP_1) > ArrayLenList(Atm_LP_1)) THEN
                   CALL ReSizeMem(inb14x(Atm_LP_1)%List, pBuff, natom, ArrayLenList(Atm_LP_1), nAddSize)
                ENDIF

                inb14x(Atm_LP_1)%List(ListNumofExcl(Atm_LP_1)) = i ! host - its LP
             ELSE ! add to i's list
                ListNumofExcl(i) = ListNumofExcl(i) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                   CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                ENDIF

                inb14x(i)%List(ListNumofExcl(i)) = Atm_LP_1 ! host - its LP
             ENDIF

             DO i_LP_2=i_LP+1, i_N_LP ! between LPs of the same host
                Atm_LP_2 = Host_LP(i)%AtomLP(i_LP_2)
                IF(Atm_LP_1 < Atm_LP_2) THEN ! add to Atm_LP_1's list
                   ListNumofExcl(Atm_LP_1) = ListNumofExcl(Atm_LP_1) + 1
                   nnb14x = nnb14x + 1
                   IF(ListNumofExcl(Atm_LP_1) > ArrayLenList(Atm_LP_1)) THEN
                      CALL ReSizeMem(inb14x(Atm_LP_1)%List, pBuff, natom, ArrayLenList(Atm_LP_1), nAddSize)
                   ENDIF

                   inb14x(Atm_LP_1)%List(ListNumofExcl(Atm_LP_1)) = Atm_LP_2 ! LPs from the same host
                ELSE ! add to Atm_LP_2's list
                   ListNumofExcl(Atm_LP_2) = ListNumofExcl(Atm_LP_2) + 1
                   nnb14x = nnb14x + 1
                   IF(ListNumofExcl(Atm_LP_2) > ArrayLenList(Atm_LP_2)) THEN
                      CALL ReSizeMem(inb14x(Atm_LP_2)%List, pBuff, natom, ArrayLenList(Atm_LP_2), nAddSize)
                   ENDIF

                   inb14x(Atm_LP_2)%List(ListNumofExcl(Atm_LP_2)) = Atm_LP_1 ! LPs from the same host
                ENDIF
             ENDDO
          ENDDO

          DO  j=j2,iblo14(i)
             Atm_Host_J = abs(inb14(j))
             j_N_LP = Host_LP(Atm_Host_J)%N_LP

             IF(j_N_LP == 0) THEN ! atom j is other than host and LP. other
                ListNumofExcl(i) = ListNumofExcl(i) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                   CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                ENDIF

                inb14x(i)%List(ListNumofExcl(i)) = inb14(j) ! host - other
             ELSEIF(j_N_LP > 0) THEN ! atom j is a host atom
                ListNumofExcl(i) = ListNumofExcl(i) + 1
                nnb14x = nnb14x + 1
                IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                   CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                ENDIF

                inb14x(i)%List(ListNumofExcl(i)) = inb14(j) ! i - j; host - host

                DO j_LP=1, j_N_LP ! add all associated LP to host i
                   Atm_Idx_LP = Host_LP(Atm_Host_J)%AtomLP(j_LP)

                   IF(Atm_Idx_LP < i) THEN ! add to list of atom Atm_Idx_LP
                      ListNumofExcl(Atm_Idx_LP) = ListNumofExcl(Atm_Idx_LP) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_Idx_LP) > ArrayLenList(Atm_Idx_LP)) THEN
                         CALL ReSizeMem(inb14x(Atm_Idx_LP)%List, pBuff, natom, ArrayLenList(Atm_Idx_LP), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(Atm_Idx_LP)%List(ListNumofExcl(Atm_Idx_LP)) = i*sign ! i - j; host - host's LP
                   ELSE ! add to list of atom i
                      ListNumofExcl(i) = ListNumofExcl(i) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(i) > ArrayLenList(i)) THEN
                         CALL ReSizeMem(inb14x(i)%List, pBuff, natom, ArrayLenList(i), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(i)%List(ListNumofExcl(i)) = Atm_Idx_LP*sign ! i - j; host - host's LP
                   ENDIF
                ENDDO
             ENDIF
          ENDDO

          DO i_LP=1, i_N_LP ! add all host's LPs / all other atoms
             Atm_LP_1 = Host_LP(i)%AtomLP(i_LP)

             DO  j=j2,iblo14(i)
                Atm_Host_J = abs(inb14(j))
                j_N_LP = Host_LP(Atm_Host_J)%N_LP

                IF(j_N_LP == 0) THEN ! atom j is other than host and LP. other
                   IF(Atm_LP_1 < Atm_Host_J) THEN ! add to Atm_LP_1's list
                      ListNumofExcl(Atm_LP_1) = ListNumofExcl(Atm_LP_1) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_LP_1) > ArrayLenList(Atm_LP_1)) THEN
                         CALL ReSizeMem(inb14x(Atm_LP_1)%List, pBuff, natom, ArrayLenList(Atm_LP_1), nAddSize)
                      ENDIF

                      inb14x(Atm_LP_1)%List(ListNumofExcl(Atm_LP_1)) = inb14(j) ! LP - other
                   ELSE ! add to Atm_Host_J's list
                      ListNumofExcl(Atm_Host_J) = ListNumofExcl(Atm_Host_J) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_Host_J) > ArrayLenList(Atm_Host_J)) THEN
                         CALL ReSizeMem(inb14x(Atm_Host_J)%List, pBuff, natom, ArrayLenList(Atm_Host_J), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(Atm_Host_J)%List(ListNumofExcl(Atm_Host_J)) = Atm_LP_1*sign ! LP - other
                   ENDIF
                ELSEIF(j_N_LP > 0) THEN ! atom j is a host atom
                   IF(Atm_LP_1 < Atm_Host_J) THEN ! add to Atm_LP_1's list
                      ListNumofExcl(Atm_LP_1) = ListNumofExcl(Atm_LP_1) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_LP_1) > ArrayLenList(Atm_LP_1)) THEN
                         CALL ReSizeMem(inb14x(Atm_LP_1)%List, pBuff, natom, ArrayLenList(Atm_LP_1), nAddSize)
                      ENDIF

                      inb14x(Atm_LP_1)%List(ListNumofExcl(Atm_LP_1)) = inb14(j) ! LP - host
                   ELSE ! add to Atm_Host_J's list
                      ListNumofExcl(Atm_Host_J) = ListNumofExcl(Atm_Host_J) + 1
                      nnb14x = nnb14x + 1
                      IF(ListNumofExcl(Atm_Host_J) > ArrayLenList(Atm_Host_J)) THEN
                         CALL ReSizeMem(inb14x(Atm_Host_J)%List, pBuff, natom, ArrayLenList(Atm_Host_J), nAddSize)
                      ENDIF

                      sign = inb14(j)/abs(inb14(j))
                      inb14x(Atm_Host_J)%List(ListNumofExcl(Atm_Host_J)) = Atm_LP_1*sign ! LP - host
                   ENDIF

                   DO j_LP=1, j_N_LP ! add all associated LPs
                      Atm_LP_2 = Host_LP(Atm_Host_J)%AtomLP(j_LP)

                      IF(Atm_LP_1 < Atm_LP_2) THEN ! add to list of atom Atm_LP_1
                         ListNumofExcl(Atm_LP_1) = ListNumofExcl(Atm_LP_1) + 1
                         nnb14x = nnb14x + 1
                         IF(ListNumofExcl(Atm_LP_1) > ArrayLenList(Atm_LP_1)) THEN
                            CALL ReSizeMem(inb14x(Atm_LP_1)%List, pBuff, natom, ArrayLenList(Atm_LP_1), nAddSize)
                         ENDIF

                         sign = inb14(j)/abs(inb14(j))
                         inb14x(Atm_LP_1)%List(ListNumofExcl(Atm_LP_1)) = Atm_LP_2*sign ! i - j; host - host's LP
                      ELSE ! add to list of atom Atm_LP_2
                         ListNumofExcl(Atm_LP_2) = ListNumofExcl(Atm_LP_2) + 1
                         nnb14x = nnb14x + 1
                         IF(ListNumofExcl(Atm_LP_2) > ArrayLenList(Atm_LP_2)) THEN
                            CALL ReSizeMem(inb14x(Atm_LP_2)%List, pBuff, natom, ArrayLenList(Atm_LP_2), nAddSize)
                         ENDIF

                         sign = inb14(j)/abs(inb14(j))
                         inb14x(Atm_LP_2)%List(ListNumofExcl(Atm_LP_2)) = Atm_LP_1*sign ! i - j; host - host's LP
                      ENDIF
                   ENDDO
                ENDIF
             ENDDO


          ENDDO


       ENDIF

       j2 = iblo14(i) + 1
    END DO


    ! sort the exclusion list for each atom after new exclusions added
    ! also compile the full exclusion list, ibn14(), and pointer list iblo14()
    nnb14 = 0
    nx14 = 0
    DO i = 1, natom
       call qsort_absints(inb14x(i)%List, ListNumofExcl(i))

       DO j = 1, ListNumofExcl(i)
          nnb14 = nnb14 + 1
          IF(nnb14 > MXNB14)THEN
             write(*,*) '<MAKLONEPAIRINB> nnb14 > MXNB14. Need to resize memory allocated.'

             DEALLOCATE(ListNumofExcl)
             DO k = 1, natom
                DEALLOCATE (inb14x(k)%List)
             END DO
             DEALLOCATE(Host_LP, inb14x)
             DEALLOCATE(pBuff, ArrayLenList)

             RETURN  ! Ran out of space
          ENDIF
          inb14(nnb14) = inb14x(i)%List(j)

          IF (inb14(nnb14) < 0) THEN
             nx14 = nx14 + 1
          ENDIF
       END DO
       iblo14(i) = nnb14
    END DO

    IF(prnlev > 10)THEN
       write(*,*)
       write(*,*) 'From MAKLONEPAIRINB(). Exclusion list'
       j2=1
       do i=1,natom
          write(*,'(A,1I8)') 'atom ',i
          do j=j2,iblo14(i)
             write(*,*) inb14(j)
          enddo
          j2=iblo14(i)+1
          write(*,*)
       enddo
    ENDIF

    if(prnlev >= 10) &
         write(*,'(a,i10,a)') 'MAKLONEPAIRINB: total  ',nnb14,' exclusion pairs'

    DEALLOCATE(ListNumofExcl)

    DO i = 1, natom
       DEALLOCATE (inb14x(i)%List)
    END DO
    DEALLOCATE(Host_LP, inb14x)

    DEALLOCATE(pBuff, ArrayLenList)

    RETURN
  END SUBROUTINE MAKLONEPAIRINB

  SUBROUTINE ReSizeMem(lpMem, lpBuff, nAtom, nCurLen, nAddSize)
    use stream
    INTEGER, DIMENSION(:), POINTER :: lpMem ! the pointer to the existing array
    INTEGER, DIMENSION(:), POINTER :: lpBuff ! the pointer to the buffer
    INTEGER nAtom, nCurLen, nNewLen, nAddSize, i, iError

    DO i=1, nCurLen
       lpBuff(i) = lpMem(i) ! backup the existing data
    ENDDO

    DEALLOCATE(lpMem)

    nNewLen = nCurLen + nAddSize
    ALLOCATE(lpMem(nNewLen), STAT=iError)
    if (iError > 0) CALL wrndie(-5, '<ReSizeMem>', &
         'Failed to allocate memory: lpMem(nNewLen)')

    DO i=1, nCurLen
       lpMem(i) = lpBuff(i) ! restore data
    ENDDO

    nCurLen = nNewLen

  END SUBROUTINE ReSizeMem

#endif /* (lp)*/
  !-------------End   added by Lei Huang ---------------

end module nbexcl
