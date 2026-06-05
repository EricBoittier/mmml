module upimag_util
  implicit none
  private

  public upimnb, mkimnb, IMAUTGEN

contains

  SUBROUTINE UPIMNB(BIMAG)
    !-----------------------------------------------------------------------
    !     GENERATE EXCLUSION LISTS FOR IMAGE-PRIMARY ATOMS IF PRESENT
    !
    use chm_kinds
    use chm_types
    !  use dimens_fcm
    use number
    !
    use machdep
    use psf
    use inbnd
    use image
    use memory
    use stream
    use pbound
    use datstr, only: imgrow
    use nbexcl_util,only:makgrp
    use api_func, only: mlpot_is_set
    implicit none
    !
    !-----------------------------------------------------------------------
    !  Miscellaneous:
    !
    !  MAXING - The maximum number of atoms in any electrostatic group.
    !
    integer,parameter :: MAXING=1000
    !-----------------------------------------------------------------------
    integer,allocatable,dimension(:) :: IGR, IPT, ISTOP

    type(imageDataStructure) BIMAG
    !
    INTEGER I,MAXNNG,IATMXB
    LOGICAL CMPLTD
    !
    !
    IF(NTRANS.EQ.0) RETURN
    IF(NBXMOD.LT.-5 .OR. NBXMOD.GT.5) RETURN
    !
#if KEY_PBOUND==1
    If (qBoun) Return
#endif
    !
    !     GENERATE EXCLUSION LISTS FOR IMAGE-PRIMARY ATOMS IF PRESENT
    !
    NBONDT=NBOND+NIMBON
    NBONDT2=NBONDT ! hwm220722
    I=NIMEXCL+1
    call IMGROW(bimag%IMINB, I)

    CALL MKIMNB(NATOM,NATIM,NIMEXCL,IMEXCLI,IMEXCLJ,IMEXCLT, &
                    BIMAG%IMINB,BIMAG%IMIBLO,BIMAG%NIMINB,(I), &
                    NBXMOD,CMPLTD)
    
    IF(MLPOT_IS_SET()) THEN
        CALL MKIMNB_MLPOT(NATOM,NATIM,NTRANS,INB,IBLO,NNB, &
                          BIMAG%IMATPT,BIMAG%IMATTR, &
                          BIMAG%IMINB,BIMAG%IMIBLO,BIMAG%NIMINB,(I))
    END IF

    ! generate image group exclusion lists

    maxnng = BIMAG%NIMINB / 2 + 10
    call IMGROW(bimag%IMING, maxnng)

    do
       call chmalloc('upimag_util.src','UPIMNB','IGR', natom, intg=IGR)
       call chmalloc('upimag_util.src','UPIMNB','IPT', MAXING, intg=IPT)
       call chmalloc('upimag_util.src','UPIMNB','ISTOP', MAXING, intg=ISTOP)

       CALL MAKGRP(NGRP+1,NGRPT,NGRP,BIMAG%IMIGLO, &
            BIMAG%IMING,BIMAG%NIMING, size(bimag%iming), &
            BIMAG%IMIBLO,BIMAG%IMINB,IGPBS,IGR, &
#ifdef KEY_RESIZE
            CMPLTD,IPT,ISTOP,size(iac))
#else
            CMPLTD,IPT,ISTOP)
#endif

       call chmdealloc('upimag_util.src','UPIMNB','IGR', natom, intg=IGR)
       call chmdealloc('upimag_util.src','UPIMNB','IPT',MAXING,intg=IPT)
       call chmdealloc('upimag_util.src','UPIMNB','ISTOP',MAXING,intg=ISTOP)

       IF (CMPLTD) EXIT
       call IMGROW(bimag%IMING, 1.5, 10)
    end do
    RETURN
  END SUBROUTINE UPIMNB

  SUBROUTINE MKIMNB(NATOM,NATIM,NIMEXCL,IMEXCLI,IMEXCLJ,IMEXCLT, &
                    INB14,IBLO14,NNB14,MXNB14,MODE,CMPLTD)
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
    !
    !     By Bernard R. Brooks    August 1983
    !              BRB overhauled August 2019
    !
    use chm_kinds
    use stream
    use exfunc
    implicit none
    !
    INTEGER NATOM,NATIM,NTRANS
    INTEGER NNB14,MXNB14,MODE

    INTEGER INB14(*),IBLO14(*)
    INTEGER NIMEXCL,IMEXCLI(*),IMEXCLJ(*),IMEXCLT(*)
    LOGICAL LEX14,CMPLTD,LFOUND

    !
    !     LOCAL STORAGE
    INTEGER I,J,I2,J2,IATOM,NX14,NEXT14,MODEX
    !
    CMPLTD=.FALSE.
    MODEX=IABS(MODE)
    !
    IF(NIMEXCL.EQ.0) THEN
       MODEX=0
       NNB14=0
    ENDIF


!    write(6,281) NIMEXCL,1
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)
!281 format('   NIMEXCL:',2I5)
!282 format(10x,A,30I5)


    !
    IF(MODEX.EQ.0) THEN
       IF(NNB14.EQ.0) THEN
          DO I=1,NATIM
             IBLO14(I)=0
          ENDDO
       ENDIF
       IF(NNB14.NE.IBLO14(NATIM)) THEN
          CALL WRNDIE(-1,'<MKIMNB>', &
               'NBXMod=0 not allowed without existing exclusion list')
       ENDIF
       CMPLTD=.TRUE.
       RETURN
    ENDIF
    !
    IF(MODEX.GT.5) MODEX=5
    IF(MODEX.EQ.4) MODEX=6
    !
    !     FOR MODE GREATER THAN ZERO INCLUDE THE EXISTING INB/IBLO LISTS,
    !     BUT NO IMAGE INTERACTIONS ARE ON EXCLUDED LIST, so we ignore
    !
    !     COMPILE A LIST OF ALL THE SPECIFIED INTERACTIONS
    !
    !     This is based on the permanent remapable exclusion list from IMAUTGEN
    !
    NNB14=0
    NX14=0
    IATOM=1
    NEXT14=1
    LEX14=.FALSE.
    DO I=1,NIMEXCL
       I2=IMEXCLI(I)
       J2=IMEXCLJ(I)
       LEX14=IMEXCLT(I).GE.4  ! make all 1-4 and cross 6-member ring interactions 1-4
       IF(IMEXCLT(I).LE.MODEX) THEN
          IF(J2.GT.IATOM) THEN
             DO J=IATOM,J2-1
                IBLO14(J)=NNB14
             ENDDO
             IATOM=J2
          ENDIF
          NNB14=NNB14+1
          !
          IF (NNB14.GT.MXNB14) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,988)
988 FORMAT(' <MKIMNB>: Ran out of space. RESIZING')
             RETURN
          ENDIF
          !
!    if(J2.LE.NATOM) then
!       WRITE(OUTU,*) ' '
!       WRITE(OUTU,*) ' MKIMNB: OK. We have a problem here.'
!       WRITE(OUTU,*) ' MKIMNB: You have a complex system where some primary-primary interactions should be excluded,'
!       WRITE(OUTU,*) ' MKIMNB: but they are not, because the intervening bonded atoms are image atoms.'
!       WRITE(OUTU,*) ' MKIMNB: So... These new exclusions need to be inserted into the main exclusion list.'
!       WRITE(OUTU,*) ' MKIMNB: A fix is coming soon... --BRB Jan-2022.'
!       CALL WRNDIE(-5,'<MKIMNB>','No code yet for this special case.')
!    endif
          !
          IF (LEX14) THEN
             INB14(NNB14)=-I2
             NX14=NX14+1
          ELSE
             INB14(NNB14)=I2
          ENDIF
       ENDIF
    ENDDO
    !
    DO J=IATOM,NATIM
       IBLO14(J)=NNB14
    ENDDO
    CMPLTD=.TRUE.
    I=NNB14-NX14
    !
    IF(PRNLEV.GE.5) WRITE(OUTU,45) MODE,I,NX14
45  FORMAT(' <MKIMNB> with mode',I4,' found',I7,' exclusions and',I7, &
         ' interactions(1-4)')

!    write(6,282) 'nnb14',NNB14
!    write(6,282) 'iblo14',(iblo14(i),i=1,30)
!    write(6,282) 'inb14',(inb14(i),i=1,nnb14)

    RETURN
  END SUBROUTINE MKIMNB


! Addition Kai Toepfer May 2022
  SUBROUTINE MKIMNB_MLPOT(NATOM,NATIM,NTRANS,INB,IBLO,NNB, &
       IMATPT,IMATTR,INB14,IBLO14,NNB14,MXNB14)
    !-----------------------------------------------------------------------
    !     THIS ROUTINE TAKES THE PSF INFORMATION AND GENERATES
    !     A NONBONDED EXCLUSION LIST (INB14 and IBLO14) 
    !     FOR MLPOT TREATED ATOMS.
    !
    !     By Kai Toepfer May 2022
    !
    use chm_kinds
    use stream
    use exfunc
    implicit none
    INTEGER NATOM,NATIM,NTRANS,NNB
    INTEGER INB(*),IBLO(*),INB14(*),IBLO14(*)
    INTEGER NNB14,MXNB14
    INTEGER IMATTR(*),IMATPT(*)
    EXTERNAL  EXCH5
    !
    !     LOCAL STORAGE
    INTEGER IPK(MXNB14),JPK(MXNB14)
    INTEGER NPAIR,MAXWRK,ILAST,ITRANS
    INTEGER I,J,K,I2,J2,J3,I2L,J2L,IATOM,NX14,NEXT14
    LOGICAL LEX14

    ! Get primary cell exclusion pairs
    NPAIR=0
    MAXWRK=MXNB14
    !
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
                DO K=NATOM+1,IMATPT(NTRANS-1)+1
                   J3=IMATTR(K)
                   IF(J2.EQ.J3) THEN
                      NPAIR=NPAIR+1
                      IF(NPAIR > MAXWRK) THEN
                         !                       Ran out of space sb070507
                         IF(WRNLEV >= 2) WRITE(OUTU,988)
                         RETURN
                      ENDIF
                      IPK(NPAIR)=K
                      JPK(NPAIR)=I2
                   ENDIF
                ENDDO
             ENDIF
          ENDDO
          ILAST=IBLO(I)
       ENDIF
    ENDDO
    !
    !     SORT THE PAIR LIST.
    !
    CALL SORT(NPAIR,EXCH5,ORDER5,IPK,JPK,0,0,0,0,0,2)
    !
    !     PROCESS THE SORTED PAIR LIST TO MAKE INB. CHECK THAT THERE ARE NOT
    !     MULTIPLE ENTRIES.
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
       IF (I2.EQ.I2L .AND. J2.EQ.J2L) THEN
          !         THIS IS A REPITION. REMOVE 1-4 SIGN IF APPLIED.
          IF(.NOT.LEX14 .AND. INB14(NNB14).LT.0) THEN
             INB14(NNB14)=-INB14(NNB14)
             NX14=NX14-1
          ENDIF
       ELSE
          !         THIS IS A NEW ONE.
          I2L=I2
          J2L=J2
          IF (I2.GT.IATOM) THEN
             DO J=IATOM,I2-1
                IBLO14(J)=NNB14
             ENDDO
             IATOM=I2
          ENDIF
          NNB14=NNB14+1
          !
          IF (NNB14.GT.MXNB14) THEN
             !           RAN-OUT-OF-SPACE
             IF(WRNLEV.GE.2) WRITE(OUTU,988)
             RETURN
          ENDIF
          !
          IF (LEX14) THEN
             INB14(NNB14)=-J2
             NX14=NX14+1
          ELSE
             INB14(NNB14)=J2
          ENDIF
       ENDIF
    ENDDO
    !
    DO J=IATOM,NATIM
       IBLO14(J)=NNB14
    ENDDO
    I=NNB14-NX14
    !
    IF(PRNLEV.GE.5) WRITE(OUTU,45) I,NX14
45  FORMAT(' <MKIMNB_MLPOT> found',I7,' exclusions and',I7, &
         ' interactions(ml-iml)')
988 FORMAT(' <MKIMNB>: Ran out of space. RESIZING')
    
    RETURN
  
  END SUBROUTINE MKIMNB_MLPOT


      SUBROUTINE IMAUTGEN(QAUTOG,QREMDUP,QSORT)
!
! The autogeneration of primary-image angles, dihedrals, and exclusions
!
  use chm_kinds
  use chm_types
!  use dimens_fcm
  use exfunc
  !
  use bases_fcm
  use memory
  use image
  use psf
  implicit none

  logical QAUTOG,QREMDUP,QSORT

  ! . Local variables.

  integer,allocatable,dimension(:,:) :: IGMATR,TRONTR
  integer,allocatable,dimension(:) :: ITRANS
  INTEGER NTRANSP
  integer MAXCON

  integer,allocatable,dimension(:,:) :: IPRB,IIMB,IATBON,IPRBCD,IIMBCD,IATBONCD
  integer,allocatable,dimension(:) :: IPRA,IIMA,IPRACD,IIMACD,NIMB,NPRB,NATBON,NATSCR

  !
  !
    if(MAXIMEX.LT.NIMBON*100) then
! Make new or larger re-mapable primry-image exclusion list, if needed
!  first, deallocate prior list, if exists.
      IF(allocated(IMEXCLI)) THEN
        call chmdealloc('images.src','IMPATC','IMEXCLI',MAXIMEX,intg=IMEXCLI)
        call chmdealloc('images.src','IMPATC','IMEXCLJ',MAXIMEX,intg=IMEXCLJ)
        call chmdealloc('images.src','IMPATC','IMEXCLT',MAXIMEX,intg=IMEXCLT)
      ENDIF
! Now reallocate for new list
      MAXIMEX=NIMBON*100
      call chmalloc('images.src','IMPATC','IMEXCLI',MAXIMEX,intg=IMEXCLI)
      call chmalloc('images.src','IMPATC','IMEXCLJ',MAXIMEX,intg=IMEXCLJ)
      call chmalloc('images.src','IMPATC','IMEXCLT',MAXIMEX,intg=IMEXCLT)
    endif
    NIMEXCL=0


 ! now do autogeneration of primary-image angles, dihedrals, and exclusions
    NTRANSP=NTRANS+1
    call chmalloc('images.src','IMPATC','IGMATR',NATOMT,NTRANSP,intg=IGMATR)
    call chmalloc('images.src','IMPATC','TRONTR',NTRANSP,NTRANSP,intg=TRONTR)
    call chmalloc('images.src','IMPATC','ITRANS',natomt,intg=ITRANS)

    call chmalloc('upimag_util.src','UPIMNB','NATBON',NATOMT,intg=NATBON)
    call chmalloc('upimag_util.src','UPIMNB','NATSCR',NATOMT,intg=NATSCR)

    !!MAXCON=IATBMX
    MAXCON=8  ! maximum number of bonds for any atom
    IF(NATOMT.LT.200) MAXCON=20  ! allow small weird things
    call chmalloc('images.src','IMPATC','IATBON',MAXCON,NATOMT,intg=IATBON)
    call chmalloc('images.src','IMPATC','IPRA',MAXCON,intg=IPRA)
    call chmalloc('images.src','IMPATC','IIMA',MAXCON,intg=IIMA)
    call chmalloc('images.src','IMPATC','IPRB',MAXCON,MAXCON,intg=IPRB)
    call chmalloc('images.src','IMPATC','IIMB',MAXCON,MAXCON,intg=IIMB)

    call chmalloc('images.src','IMPATC','IATBONCD',MAXCON,NATOMT,intg=IATBONCD)
    call chmalloc('images.src','IMPATC','IPRACD',MAXCON,intg=IPRACD)
    call chmalloc('images.src','IMPATC','IIMACD',MAXCON,intg=IIMACD)
    call chmalloc('images.src','IMPATC','IPRBCD',MAXCON,MAXCON,intg=IPRBCD)
    call chmalloc('images.src','IMPATC','IIMBCD',MAXCON,MAXCON,intg=IIMBCD)

    call chmalloc('images.src','IMPATC','NIMB',MAXCON,intg=NIMB)
    call chmalloc('images.src','IMPATC','NPRB',MAXCON,intg=NPRB)


    CALL IMAUTGEN2(NATOM,NTRANS,BIMAG%IMATPT,BIMAG%IMATTR,ITRANS,TRONTR,IGMATR, &
          IPRA,IIMA,IPRB,IIMB,IPRACD,IIMACD,IPRBCD,IIMBCD,NIMB,NPRB, &
          NATBON,IATBON,IATBONCD,MAXCON,NATSCR,QAUTOG,QREMDUP,QSORT)
 !
     call chmdealloc('images.src','IMPATC','IGMATR',NATOMT,NTRANSP,intg=IGMATR)
     call chmdealloc('images.src','IMPATC','TRONTR',NTRANSP,NTRANSP,intg=TRONTR)
     call chmdealloc('images.src','IMPATC','ITRANS',natomt,intg=ITRANS)

     call chmdealloc('upimag_util.src','UPIMNB','NATBON',NATOMT,intg=NATBON)
     call chmdealloc('upimag_util.src','UPIMNB','NATSCR',NATOMT,intg=NATSCR)

     call chmdealloc('images.src','IMPATC','IATBON',MAXCON,NATOMT,intg=IATBON)
     call chmdealloc('images.src','IMPATC','IPRA',MAXCON,intg=IPRA)
     call chmdealloc('images.src','IMPATC','IIMA',MAXCON,intg=IIMA)
     call chmdealloc('images.src','IMPATC','IPRB',MAXCON,MAXCON,intg=IPRB)
     call chmdealloc('images.src','IMPATC','IIMB',MAXCON,MAXCON,intg=IIMB)

     call chmdealloc('images.src','IMPATC','IATBONCD',MAXCON,NATOMT,intg=IATBONCD)
     call chmdealloc('images.src','IMPATC','IPRACD',MAXCON,intg=IPRACD)
     call chmdealloc('images.src','IMPATC','IIMACD',MAXCON,intg=IIMACD)
     call chmdealloc('images.src','IMPATC','IPRBCD',MAXCON,MAXCON,intg=IPRBCD)
     call chmdealloc('images.src','IMPATC','IIMBCD',MAXCON,MAXCON,intg=IIMBCD)

     call chmdealloc('images.src','IMPATC','NIMB',MAXCON,intg=NIMB)
     call chmdealloc('images.src','IMPATC','NPRB',MAXCON,intg=NPRB)


  RETURN
END SUBROUTINE IMAUTGEN

    SUBROUTINE IMAUTGEN2(NATOMX,NTRANSX,IMATPT,IMATTR,ITRANS,TRONTR,IGMATR, &
          IPRA,IIMA,IPRB,IIMB,IPRACD,IIMACD,IPRBCD,IIMBCD,NIMB,NPRB, &
          NATBON,IATBON,IATBONCD,MAXCON,NATSCR,QAUTOG,QREMDUP,QSORT)
!
! The autogeneration of primary-image angles, dihedrals, and exclusions
!
! This routine also:
!     puts primary-image bonds     in canonical order and removes duplicates
!     puts primary-image angles    in canonical order and removes duplicates
!     puts primary-image dihedrals in canonical order and removes duplicates
!     puts primary-image impropers in canonical order and removes duplicates
!     checks for cyclic structures and modifies exclusion list for 6-member rings
!
!  canonical order means: primary atoms first, and use image atoms with lowest number.
!
!        NATOMX         - another version of NATOM
!        NTRANS         - number of image transformations
!        IMATPT(NTRANS) - image transformation pointer
!        IMATTR(NATOMT)  - corresponding primary atom
!        ITRANS(NATOMT)  - Image number for each atom (0=primary)
!        TRONTR(0:NTRANS,0:NTRANS) - transformation multiplication table
!        (Gets the image transformation number for a product of two transformations)
!        IGMATR(NATOMT,0:NTRANSX) - inverse of IMATTR, image atom numbers for each primary atom
!                                       (and now extends into image atoms)
!        
!        IPRA(MAXCON)      - list of atoms bonded to primary atom of bond
!        IIMA(MAXCON)      - list of atoms bonded to image atom of bond
!        IPRB(MAXCON,MAXCON) - list of atoms 1-3 bonded to primary atom of bond
!        IIMB(MAXCON,MAXCON) - list of atoms 1-3 bonded to image atom of bond
!        NIMB(MAXCON)
!        NPRB(MAXCON)
!        NATBON(NATOMT)    - Number of bonds for each atom
!        IATBON(MAXCON,NATOMT) -
!          " CD            - Bond autogeneration code for each bond (5 arrays)
!        MAXCON            - Maximum number of bonds for any atom
!        NATSCR(NATOMT)    - Flags of image atoms with exclusions with primary
!
!  Other arrays used:
!        NIMEXCL           - The number of image exclusions
!        IMEXCLI(MAXIMEX)  - primary atom of exclusion (not an image)
!        IMEXCLJ(MAXIMEX)  - image atom of exclusion -- must reorder on image update
!        IMEXCLT(MAXIMEX)  - exclusion type (2,3,4)  -- must reorder on image update
!        MAXIMEX           - maximum number of image exclusions
!
!
    use chm_kinds
    use stream
    use exfunc
    use image
    use psf
    use machutil,only:die
    implicit none

    integer NATOMX,NTRANSX
    integer IMATPT(*),IMATTR(*),ITRANS(*)
    integer TRONTR(0:NTRANSX,0:NTRANSX),IGMATR(NATOMT,0:NTRANSX)

    integer MAXCON
    INTEGER NATBON(*),IATBON(MAXCON,*),IATBONCD(MAXCON,*),NATSCR(*)
    INTEGER IPRA(MAXCON),IIMA(MAXCON),IPRB(MAXCON,MAXCON),IIMB(MAXCON,MAXCON)
    INTEGER IPRACD(MAXCON),IIMACD(MAXCON),IPRBCD(MAXCON,MAXCON),IIMBCD(MAXCON,MAXCON)
    INTEGER NIMB(MAXCON),NPRB(MAXCON)
    LOGICAL QAUTOG,QREMDUP,QSORT,KEEP

    !     LOCAL STORAGE
    INTEGER NIMA,NPRA
    INTEGER IBOND,IS,IQ
    INTEGER I,J,IJ,IPR,IBT,JBT,IIM,IA,NA,JA ,I2,J2,K2,L2
    INTEGER I3,J3,K3,L3,I4,J4,K4,L4,I5,J5,K5,L5,I6,J6,K6,I7,J7,K7
    INTEGER II,JJ,ITR,JTR,KTR,IPT,JPT,KPT,MP,MTR,MXTR
    INTEGER ICD,JCD,KCD
    logical skip,removed,found,qcodes
    !
            
    real(chm_real) RMAT(12)
    integer, parameter :: MARK=99999999
!    integer, parameter :: MARK=999 
    real,parameter :: TOLER=0.0001
    EXTERNAL  EXCH5, EXCH8

    real(chm_real) UNCHK

! Create temporary arrays, IGMATR(NATOMT,0:NTRANS)

     DO I=1,NATOM
        ITRANS(I)=0
        IMATTR(I)=I
     ENDDO
     DO I=1,NATOMT
        IGMATR(I,0)=I
     ENDDO
     IS=NATOM+1
     DO ITR=1,NTRANS
        DO I=1,NATOMT
           IGMATR(I,ITR)=-1
        ENDDO
        IQ=IMATPT(ITR)
        DO I=IS,IQ
           IGMATR(IMATTR(I),ITR)=I
           ITRANS(I)=ITR
        ENDDO
        IS=IQ+1
     ENDDO


!      write(6,241)
!    DO I=1,NATOMT
!      write(6,241) I,(IGMATR(I,J),J=0,NTRANS)
!241 format('   I,IGMATR(I,J),:',25I5)
!    ENDDO


! Create temporary arrays, TRONTR(0:NTRANS,0:NTRANS)
! Gets the image transformation number for a product of two transformations, 
!   returns MARK if not present
!     example: IXTR = TRONTR(ITR,JTR) -- IXTR is the image IRT becomes when operated on by IMINV(JTR)

!     write(6,251)


      TRONTR(0,0)=0
      DO ITR=1,NTRANS
         TRONTR(0,ITR)=IMINV(ITR) 
         TRONTR(ITR,0)=ITR
         TRONTR(ITR,ITR)=0

!         IPT=(ITR-1)*12      
!      write(6,251) ITR,0,(IMTRNS(IPT+J),J=1,12)
      ENDDO

!     write(6,251)

      DO ITR=1,NTRANS
         IPT=(ITR-1)*12
         DO JTR=1,NTRANS
           IF(ITR.NE.JTR) THEN
              TRONTR(ITR,JTR) = MARK ! number corresponding for missing transformation product
           
           ! multiply ITR by the inverse of JTR
              JPT=(IMINV(JTR)-1)*12
              CALL MULTTRMR(IMTRNS(IPT+1),IMTRNS(JPT+1),RMAT)


!      write(6,251) ITR,JTR,RMAT
!251 format('   ITR,JTR,RMAT:',2I5,12F10.4)


              DO KTR=1,NTRANS
                KPT=(KTR-1)*12
                UNCHK=0.0
                DO I=1,12
                  UNCHK=UNCHK+ABS(RMAT(I)-IMTRNS(KPT+I))
                ENDDO


!      IF(itr.eq.1 .and. jtr.eq.7) write(6,252) ITR,JTR,KTR,UNCHK
!      IF(itr.eq.1 .and. jtr.eq.9) write(6,252) ITR,JTR,KTR,UNCHK
!      IF(itr.eq.3 .and. jtr.eq.4) write(6,252) ITR,JTR,KTR,UNCHK
!         write(6,252) ITR,JTR,KTR,UNCHK
!252 format('   ITR,JTR,KTR,UNCHK:',3I5,F16.8)


                IF(UNCHK.LE.TOLER) THEN
                  TRONTR(ITR,JTR)=KTR
                  GOTO 170
                ENDIF
              ENDDO
170          CONTINUE
 
           ENDIF
         ENDDO
      ENDDO

!  now create the image to image transformation indices
     DO ITR=1,NTRANS
        DO I=NATOM+1,NATOMT
           MP=IMATTR(I)
           MTR=ITRANS(I)
           MXTR=TRONTR(MTR,IMINV(ITR))
           IF(MXTR.NE.MARK) THEN
             J=IGMATR(MP,MXTR)
             IF(J.GT.0) IGMATR(I,ITR)=J
           ENDIF
        ENDDO 
     ENDDO

!      write(6,241)
!    DO I=1,NATOMT
!      write(6,241) I,(IGMATR(I,J),J=0,NTRANS)
!241 format('   I,IGMATR(I,J),:',25I5)
!    ENDDO



!      write(6,231)
!    DO I=0,NTRANS
!      write(6,231) I,(TRONTR(J,I),J=0,NTRANS)
!231 format('   I,TRONTR(J,I),:',25I5)
!    ENDDO




!    remove any bad image bonds
      IPT=NBOND
      JPT=NBOND
      DO I=1,NIMBON
        IPT=IPT+1
        IF(IB(IPT).GT.0 .AND. JB(IPT).GT.0) THEN
           JPT=JPT+1
           IF(IPT.GT.JPT) THEN
              IB(JPT)=IB(IPT)
              JB(JPT)=JB(IPT)
              ICBAUTO(JPT)=ICBAUTO(IPT)
           ENDIF
        ENDIF
      ENDDO
!   we just deleted IPT-JPT bonds
     IF(PRNLEV.GE.2) WRITE(OUTU,24) IPT-JPT,'bonds'
24   FORMAT(' <IMAUTGEN>:',I5,' bad image ',A,' deleted')
      NBONDT=JPT
      NIMBON=NBONDT-NBOND
      NBONDT2=NBONDT

! Make image bond list canonical (i.e. connected to the lower image number)
  
      DO I = 1,NIMBON
         J=I+NBOND
         CALL MKIMCANON(IB(J),JB(J),0,0,2,1, &
                        NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
      ENDDO

! Sort image bonds and remove duplicate bonds

      CALL SORT(NIMBON,EXCH5,ORDER5,IB(NBOND+1),JB(NBOND+1),ICBAUTO(NBOND+1),0,0,0,0,3)
!
!    remove any duplicate bonds
      IPT=NBOND+1
      JPT=NBOND+1
      DO I=2,NIMBON
        IF(IB(IPT).NE.IB(IPT+1) .OR. JB(IPT).NE.JB(IPT+1)) JPT=JPT+1
        IPT=IPT+1
        IF(IPT.GT.JPT) THEN
           IB(JPT)=IB(IPT)
           JB(JPT)=JB(IPT)
           ICBAUTO(JPT)=ICBAUTO(IPT)
        ENDIF
      ENDDO
!   we just deleted IPT-JPT bonds
     IF(PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'bonds'
25   FORMAT(' <IMAUTGEN>:',I5,' duplicate image ',A,' deleted')
      NBONDT=JPT
      NIMBON=NBONDT-NBOND
      NBONDT2=NBONDT
!
      if(QAUTOG) THEN
         ! remove existing angles and dihedrals for autogen terms

          j=ntheta
          do i=ntheta+1,nthett
            ! keep any angle involving any atoms without autogen
            keep=.false.
            if(iand(ICAAUTO(JT(i)),32).gt.0) then
               keep=.true.  ! keep if central atom is not autogenerating angles
            endif
            if(keep) then
               j=j+1
               IT(j)=IT(i)
               JT(j)=JT(i)
               KT(j)=KT(i)
            endif
          enddo
          nthett=j
          nimang=nthett-ntheta


          j=nphi
          do i=nphi+1,nphit
            ! keep any dihedral involving either central atom without autogen
            keep=.false.
            if(iand(ICAAUTO(JP(i)),64).gt.0) then
               keep=.true.  ! keep if a central atom is not autogenerating dihedrals
            elseif(iand(ICAAUTO(KP(i)),64).gt.0) then
               keep=.true.  ! keep if a central atom is not autogenerating dihedrals
            endif
            if(keep) then
               j=j+1
               IP(j)=IP(i)
               JP(j)=JP(i)
               KP(j)=KP(i)
               LP(j)=LP(i)
            endif
          enddo
          nphit=j
          nimdih=nphit-nphi
      ENDIF
      
!
!  add nonbond exclusions for these bonds
      NIMEXCL=NIMBON
      DO I = 1,NIMBON
         J=I+NBOND
         IMEXCLI(I) = IB(J)
         IMEXCLJ(I) = JB(J)  
         IMEXCLT(I) = 2
      ENDDO 


!    write(6,281) NIMEXCL,2
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)
!281 format('   NIMEXCL:',2I5)
!282 format(10x,A,20I5,(/20x,20I5))


! create connectivity list for each atom (NATOMT)

      DO I=1,NATOMT
         NATBON(I)=0
         NATSCR(I)=0
      ENDDO

      DO I=1,NBONDT
         IBT = IB(I)
         JBT = JB(I)
         ICD=ICBAUTO(I)
         call add_to_bond_list 
      ENDDO
      J=0
      DO I=NBOND+1,NBONDT
         IBT=IB(I)
         JBT=JB(I)
         IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
         IF(IBT.LE.NATOM .OR. JBT.LE.NATOM) THEN
            IF(IBT.GT.NATOM .and. NATSCR(IBT).eq.0) THEN
               NATSCR(IBT)=1
               J=J+1
            ENDIF
            IF(JBT.GT.NATOM .and. NATSCR(JBT).eq.0) THEN
               NATSCR(JBT)=1
               J=J+1
            ENDIF
          ENDIF
         ENDIF
      ENDDO

      DO ITR=1,NTRANS
         DO I=NBOND+1,NBONDT
            IBT=IGMATR(IB(I),ITR)
            JBT=IGMATR(JB(I),ITR)
            IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
               IF(IBT.LE.NATOM .OR. JBT.LE.NATOM) THEN
                 IF(IBT.GT.NATOM .and. NATSCR(IBT).eq.0) THEN
                    NATSCR(IBT)=1
                    J=J+1
                 ENDIF
                 IF(JBT.GT.NATOM .and. NATSCR(JBT).eq.0) THEN
                    NATSCR(JBT)=1
                    J=J+1
                 ENDIF
               ENDIF
            ENDIF
         ENDDO
      ENDDO
      IF(PRNLEV.GE.2 .and. j.gt.0) WRITE(OUTU,*) ' IMAUTGEN: Found ',J,' type 1-2 interactions with images.'

      J=0
      DO ITR=1,NTRANS
         DO I=1,NBONDT
            IBT=IGMATR(IB(I),ITR)
            JBT=IGMATR(JB(I),ITR)
            IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
             IF(NATSCR(IBT).EQ.1 .or. NATSCR(JBT).EQ.1) THEN
                 IF(IBT.GT.NATOM .and. NATSCR(IBT).eq.0) THEN
                    NATSCR(IBT)=3
                    J=J+1

!      WRITE(OUTU,*) ' IMAUTGEN: adding IBT ',IBT,' type 1-3 interactions',JBT 

                 ENDIF
                 IF(JBT.GT.NATOM .and. NATSCR(JBT).eq.0) THEN
                    NATSCR(JBT)=3
                    J=J+1

!      WRITE(OUTU,*) ' IMAUTGEN: adding JBT ',JBT,' type 1-3 interactions',IBT 


                 ENDIF
             ENDIF
            ENDIF
         ENDDO
      ENDDO
      IF(PRNLEV.GE.2 .and. j.gt.0) WRITE(OUTU,*) ' IMAUTGEN: Found ',J,' type 1-3 interactions with images.'

      J=0
      DO ITR=1,NTRANS
         DO I=1,NBONDT
            IBT=IGMATR(IB(I),ITR)
            JBT=IGMATR(JB(I),ITR)
            IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
             IF(NATSCR(IBT).EQ.3 .or. NATSCR(JBT).EQ.3) THEN
                 IF(IBT.GT.NATOM .and. NATSCR(IBT).eq.0) THEN
                    NATSCR(IBT)=7
                    J=J+1
                 ENDIF
                 IF(JBT.GT.NATOM .and. NATSCR(JBT).eq.0) THEN
                    NATSCR(JBT)=7
                    J=J+1
                 ENDIF
             ENDIF
            ENDIF
         ENDDO
      ENDDO
      IF(PRNLEV.GE.2 .and. j.gt.0) WRITE(OUTU,*) ' IMAUTGEN: Found ',J,' type 1-4 interactions with images.'

      DO I=1,NATOM
         NATSCR(I)=1
      ENDDO
      DO ITR=1,NTRANS
         DO I=1,NBONDT
            IBT=IGMATR(IB(I),ITR)
            JBT=IGMATR(JB(I),ITR)
            IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
              IF(NATSCR(IBT).gt.0 .and. NATSCR(JBT).gt.0) THEN
                IF(IBT.GT.NATOM .or. JBT.GT.NATOM) THEN
                  ICD=ICBAUTO(I)
                  call add_to_bond_list 
                  nbondt2=nbondt2+1
                  IF(NBONDT2.GT.MAXB)THEN
                    CALL WRNDIE(-3,'<IMPAT2>','Maximum number of bonds exceeded')
                    return
                  ENDIF
                  IB(nbondt2)=IBT
                  JB(nbondt2)=JBT
                  ICBAUTO(nbondt2)=ICBAUTO(I)
                ENDIF
              ENDIF
            ENDIF
         ENDDO
      ENDDO


!    DO I=1,NATOMT
!    write(6,221) I,NATBON(I),(IATBON(J,I),J=1,NATBON(I))
!221 format('   NATBON(I),IATBON(J,I),:',20I5)
!    ENDDO

!
! loop over image bonds, creating all angles (2 cases) and all dihedrals (3 cases), and all exclusions(6 cases)
! Also convert all angles, dihedrals, and exclusions to canonical.
!
       DO IBOND=NBOND+1,NBONDT

          !
          !  set up linkage arrays for this bond (exclude backtracking)
          !
          IPR=IB(IBOND)
          IIM=JB(IBOND)
          NPRA=0
          NA=NATBON(IPR)
          DO J=1,NA
             JA=IATBON(J,IPR)
             ICD=IATBONCD(J,IPR)
             IF(JA.NE.IIM) THEN
                NPRA=NPRA+1
                IPRA(NPRA)=JA
                IPRACD(NPRA)=ICD
             ENDIF
          ENDDO
          !
          NIMA=0
          NA=NATBON(IIM)
          DO J=1,NA
             JA=IATBON(J,IIM)
             ICD=IATBONCD(J,IIM)
             IF(JA.NE.IPR) THEN
                NIMA=NIMA+1
                IIMA(NIMA)=JA
                IIMACD(NIMA)=ICD
             ENDIF
          ENDDO
          !
          DO I=1,NPRA
             NPRB(I)=0
             IA=IPRA(I)
             NA=NATBON(IA)
             DO J=1,NA
                JA=IATBON(J,IA)
                ICD=IATBONCD(J,IA)
                IF(JA.NE.IPR) THEN
                   NPRB(I)=NPRB(I)+1
                   IPRB(NPRB(I),I)=JA
                   IPRBCD(NPRB(I),I)=ICD
                ENDIF
             ENDDO
          ENDDO
             !
          DO I=1,NIMA
             NIMB(I)=0
             IA=IIMA(I)
             NA=NATBON(IA)
             DO J=1,NA
                JA=IATBON(J,IA)
                ICD=IATBONCD(J,IA)
                IF(JA.NE.IIM) THEN
                   NIMB(I)=NIMB(I)+1
                   IIMB(NIMB(I),I)=JA
                   IIMBCD(NIMB(I),I)=ICD
                ENDIF
             ENDDO
          ENDDO


! print out arrays
!    write(6,220) IBOND,IPR,IIM
!220 format('Processing bond#, IPR, IIM:',3I5)
!    write(6,223) NPRA,(IPRA(J),J=1,NPRA)
!223 format('   NPRA,:',20I5)
!    write(6,224) NIMA,(IIMA(J),J=1,NIMA)
!224 format('   NIMA,:',20I5)

          ICD=ICBAUTO(IBOND)

          !
          ! do first angle set centered on IIM
          !
          DO I=1,NIMA
             I2=IPR
             J2=IIM
             K2=IIMA(I)
             JCD=IIMACD(I)              
             CALL MKIMCANON(I2,K2,J2,0,3,2, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)

                NIMEXCL=NIMEXCL+1
                IMEXCLI(NIMEXCL) = MIN(I2,K2)
                IMEXCLJ(NIMEXCL) = MAX(I2,K2) 
                IMEXCLT(NIMEXCL) = 3

             if(QAUTOG) then
! AUTOGEN: check 3 atom flags, 2 bond flags
               skip=        IAND(ICAAUTO(J2),1).gt.0
               skip=skip.or.IAND(ICAAUTO(I2),2).gt.0
               skip=skip.or.IAND(ICAAUTO(K2),2).gt.0
               skip=skip.or.IAND(ICD,2).gt.0
               skip=skip.or.IAND(JCD,2).gt.0
               if(.not.skip) then
                   NIMANG=NIMANG+1
                   NTHETT=NTHETT+1
                   IT(NTHETT)=I2
                   JT(NTHETT)=J2
                   KT(NTHETT)=K2
                endif
             endif
          ENDDO
          !
          ! do second angle set centered on IPR
          !
          DO I=1,NPRA
             I2=IPRA(I)
             J2=IPR
             K2=IIM
             JCD=IPRACD(I) 
             CALL MKIMCANON(I2,K2,J2,0,3,3, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
                NIMEXCL=NIMEXCL+1
                IMEXCLI(NIMEXCL) = MIN(I2,K2)
                IMEXCLJ(NIMEXCL) = Max(I2,K2) 
                IMEXCLT(NIMEXCL) = 3

             if(QAUTOG) then
! AUTOGEN: check 3 atom flags, 2 bond flags
                skip=        IAND(ICAAUTO(J2),1).gt.0
                skip=skip.or.IAND(ICAAUTO(I2),2).gt.0
                skip=skip.or.IAND(ICAAUTO(K2),2).gt.0
                skip=skip.or.IAND(ICD,2).gt.0
                skip=skip.or.IAND(JCD,2).gt.0
                if(.not.skip) then
                   NIMANG=NIMANG+1
                   NTHETT=NTHETT+1
                   IT(NTHETT)=I2
                   JT(NTHETT)=J2
                   KT(NTHETT)=K2
                endif
             endif
          ENDDO


!    write(6,281) NIMEXCL,31
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)

          qcodes=.false.

          !
          ! do first dihedral set centered on IIM-IIMA
          !
          DO I=1,NIMA
             DO J=1,NIMB(I)    
                I2=IPR
                J2=IIM
                K2=IIMA(I)
                JCD=IIMACD(I)
                L2=IIMB(J,I)
                KCD=IIMBCD(J,I)
                CALL MKIMCANON(I2,L2,J2,K2,4,4, &
                               NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
                   NIMEXCL=NIMEXCL+1
                   IMEXCLI(NIMEXCL) = MIN(I2,L2)
                   IMEXCLJ(NIMEXCL) = MAX(I2,L2) 
                   IMEXCLT(NIMEXCL) = 5

                if(QAUTOG) then
! AUTOGEN: check 4 atom flags, 3 bond flags
                   skip=        IAND(ICAAUTO(I2),8).gt.0
                   skip=skip.or.IAND(ICAAUTO(J2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(K2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(L2),8).gt.0
                   skip=skip.or.IAND(ICD,8).gt.0
                   skip=skip.or.IAND(JCD,4).gt.0
                   skip=skip.or.IAND(KCD,8).gt.0
                   if(.not.skip) then
                      NIMDIH=NIMDIH+1
                      NPHIT=NPHIT+1
                      IP(NPHIT)=I2
                      JP(NPHIT)=J2
                      KP(NPHIT)=K2
                      LP(NPHIT)=L2
                      qcodes=qcodes.or.(iand(ICAAUTO(J2),16).eq.0 .and. iand(ICAAUTO(K2),16).eq.0)
                   endif
                endif
             ENDDO
          ENDDO
          !

!    write(6,281) NIMEXCL,32
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)


          ! do second dihedral set centered on IPR-IPRA
          !
          DO I=1,NPRA
             DO J=1,NPRB(I)    
                I2=IPRB(J,I)
                KCD=IPRBCD(J,I)
                J2=IPRA(I)
                JCD=IPRACD(I)
                K2=IPR
                L2=IIM
                CALL MKIMCANON(I2,L2,J2,K2,4,5, &
                               NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
                   NIMEXCL=NIMEXCL+1
                   IMEXCLI(NIMEXCL) = MIN(I2,L2)
                   IMEXCLJ(NIMEXCL) = MAX(I2,L2) 
                   IMEXCLT(NIMEXCL) = 5
                if(QAUTOG) then
! AUTOGEN: check 4 atom flags, 3 bond flags
                   skip=        IAND(ICAAUTO(I2),8).gt.0
                   skip=skip.or.IAND(ICAAUTO(J2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(K2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(L2),8).gt.0
                   skip=skip.or.IAND(ICD,8).gt.0
                   skip=skip.or.IAND(JCD,4).gt.0
                   skip=skip.or.IAND(KCD,8).gt.0
                   if(.not.skip) then
                      NIMDIH=NIMDIH+1
                      NPHIT=NPHIT+1
                      IP(NPHIT)=I2
                      JP(NPHIT)=J2
                      KP(NPHIT)=K2
                      LP(NPHIT)=L2
                      qcodes=qcodes.or.(iand(ICAAUTO(J2),16).eq.0 .and. iand(ICAAUTO(K2),16).eq.0)
                   endif
                endif
             ENDDO
          ENDDO
          !


!    write(6,281) NIMEXCL,33
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)


          ! do third dihedral set centered on IPR-IMA
          !
          DO I=1,NPRA
             DO J=1,NIMA
                I2=IPRA(I)
                JCD=IPRACD(I)
                J2=IPR
                K2=IIM
                L2=IIMA(J)
                KCD=IIMACD(J)
                CALL MKIMCANON(I2,L2,J2,K2,4,6, &
                               NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
                   NIMEXCL=NIMEXCL+1
                   IMEXCLI(NIMEXCL) = MIN(I2,L2)
                   IMEXCLJ(NIMEXCL) = MAX(I2,L2) 
                   IMEXCLT(NIMEXCL) = 5
                if(QAUTOG) then
! AUTOGEN: check 4 atom flags, 3 bond flags
                   skip=        IAND(ICAAUTO(I2),8).gt.0
                   skip=skip.or.IAND(ICAAUTO(J2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(K2),4).gt.0
                   skip=skip.or.IAND(ICAAUTO(L2),8).gt.0
                   skip=skip.or.IAND(ICD,4).gt.0
                   skip=skip.or.IAND(JCD,8).gt.0
                   skip=skip.or.IAND(KCD,8).gt.0
                   if(.not.skip) then
                      NIMDIH=NIMDIH+1
                      NPHIT=NPHIT+1
                      IP(NPHIT)=I2
                      JP(NPHIT)=J2
                      KP(NPHIT)=K2
                      LP(NPHIT)=L2
                      qcodes=qcodes.or.(iand(ICAAUTO(J2),16).eq.0 .and. iand(ICAAUTO(K2),16).eq.0)
                   endif
                endif
             ENDDO
          ENDDO
       ENDDO

!    write(6,281) NIMEXCL,34
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)


! now process image-primary ICPAUTO information, removing or adding angles and dihedrals.


    IF(QAUTOG) THEN

!     Now search the extra/deleted list of angles
      removed=.false.
      DO I=NPAUTO+1,NPAUTOT
        if(LPAUTO(I).eq.0) then  ! this is an angle type
           I2=IPAUTO(I)
           J2=JPAUTO(I)
           K2=KPAUTO(I)
           CALL MKIMCANON(I2,K2,J2,0,3,7, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
           I3=KPAUTO(I)
           J3=JPAUTO(I)
           K3=IPAUTO(I)
           CALL MKIMCANON(I3,K3,J3,0,3,8, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
           found=.false.
           DO J=NTHETA+1, NTHETT     ! search new angles to see if angle is found
             IF(JT(J).EQ.J2) then ! central atoms match
               IF(IT(J).EQ.I2) then ! 1st atoms match
                 IF(KT(J).EQ.K2) then ! 3rd atoms match!
                   found=.true.
                   IF(IAND(ICPAUTO(I),2).gt.0) then
                     IT(J)=0  ! remove unwanted angle
                     removed=.true.
                   ENDIF
                 endif
               endif
             endif
             IF(JT(J).EQ.J3) then ! central atoms match
               IF(KT(J).EQ.K3) then ! 1st atoms match the other way
                 IF(IT(J).EQ.I3) then ! 3rd atoms match!
                   found=.true.
                   IF(IAND(ICPAUTO(I),2).gt.0) then
                     IT(J)=0 ! remove unwanted angle
                     removed=.true.
                   ENDIF
                 endif
               endif
             endif
           ENDDO
           if(.not.found) then
              IF(IAND(ICPAUTO(I),2).eq.0) then ! add this angle
                  NTHETT=NTHETT+1
                  NIMANG=NIMANG+1
                  IT(NTHETT)=MIN(IPAUTO(I),KPAUTO(I))
                  JT(NTHETT)=JPAUTO(I)
                  KT(NTHETT)=MAX(IPAUTO(I),KPAUTO(I))
              ELSE   
                                     ! angle to deleted could not be found.
                 IF(WRNLEV.GE.2)then
                   write(outu,*) 'Requested angle for deletion not found:',IPAUTO(I),JPAUTO(I),KPAUTO(I)
                   CALL WRNDIE(0,'<AUTGEN>','Requested angle to delete not Found. Nothing done')
                 ENDIF
              ENDIF
           endif
        endif
      ENDDO


!write (outu,*) 'Testing AUTOGEN D:',nthett,nphit

      IF(removed) THEN  ! Remove any unwanted angles that were flagged above
        j=ntheta
        do i=ntheta+1,nthett
          ! keep any angle involving unselected atoms
          if(IT(I).ne.0) then
             j=j+1
             IT(j)=IT(i)
             JT(j)=JT(i)
             KT(j)=KT(i) 
          endif
        enddo
        nthett=j
        NIMANG=nthett-ntheta
      ENDIF

!write (outu,*) 'Testing AUTOGEN E:',nthett,nphit



!write (outu,*) 'Testing AUTOGEN F:',ntheta,nphi


!     Now search the extra/deleted list of dihedrals
      removed=.false.
      DO I=1,NPAUTO

!write (outu,*) ' Processing: i,IPAUTO(i),...',I,IPAUTO(I),JPAUTO(I),KPAUTO(I),LPAUTO(I),ICPAUTO(I)

                                                ! this is an angle type
         if(LPAUTO(I).eq.0) then
           if(IAND(ICPAUTO(I),12).gt.0) then  ! this is an angle type with request to prevent dihedrals

!      Process all eight possibilities for matching dihedrals. 
!      Make sure everything remains canonical before matching in every case.

             I2=IPAUTO(I)
             J2=JPAUTO(I)
             K2=KPAUTO(I)
             CALL MKIMCANON(I2,K2,J2,0,3,9, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
 
             I3=KPAUTO(I)
             J3=JPAUTO(I)
             K3=IPAUTO(I)
             CALL MKIMCANON(I3,K3,J3,0,3,10, &
                           NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)

             found=.false.
             DO J=NPHI+1, NPHIT     ! search new angles to see if angle is found

               I6=IP(J)
               J6=JP(J)
               K6=KP(J)
               CALL MKIMCANON(I6,K6,J6,0,3,11, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
               I7=KP(J)
               J7=JP(J)
               K7=IP(J)
               CALL MKIMCANON(I7,K7,J7,0,3,12, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
               I4=JP(J)
               J4=KP(J)
               K4=LP(J)
               CALL MKIMCANON(I4,K4,J4,0,3,13, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
               I5=LP(J)
               J5=KP(J)
               K5=JP(J)
               CALL MKIMCANON(I5,K5,J5,0,3,14, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)

               IF(J2.EQ.J4) then ! 2nd atoms match
                 IF(I2.EQ.I4) then ! 1st atoms match
                   IF(K2.EQ.K4) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J2.EQ.J5) then ! 2nd atoms match
                 IF(I2.EQ.I5) then ! 1st atoms match
                   IF(K2.EQ.K5) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J2.EQ.J6) then ! 2nd atoms match
                 IF(I2.EQ.I6) then ! 1st atoms match
                   IF(K2.EQ.K6) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J2.EQ.J7) then ! 2nd atoms match
                 IF(I2.EQ.I7) then ! 1st atoms match
                   IF(K2.EQ.K7) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J3.EQ.J4) then ! 2nd atoms match
                 IF(I3.EQ.I4) then ! 1st atoms match
                   IF(K3.EQ.K4) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J3.EQ.J5) then ! 2nd atoms match
                 IF(I3.EQ.I5) then ! 1st atoms match
                   IF(K3.EQ.K5) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J3.EQ.J6) then ! 2nd atoms match
                 IF(I3.EQ.I6) then ! 1st atoms match
                   IF(K3.EQ.K6) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

               IF(J3.EQ.J7) then ! 2nd atoms match
                 IF(I3.EQ.I7) then ! 1st atoms match
                   IF(K3.EQ.K7) then ! 3rd atoms match
                     IP(J)=0  ! remove unwanted dihedral
                     removed=.true.
                   endif
                 endif
               endif

             ENDDO
           endif
         else
                                           ! this is a dihedral type
!      Process all four possibilities for matching dihedrals. 
!      Make sure everything remains canonical before matching in every case.
            I2=IPAUTO(I)
            J2=JPAUTO(I)
            K2=KPAUTO(I)
            L2=LPAUTO(I)
            CALL MKIMCANON(I2,L2,J2,K2,4,15, &
                           NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
 
            I3=LPAUTO(I)
            J3=KPAUTO(I)
            K3=JPAUTO(I)
            L3=IPAUTO(I)
            CALL MKIMCANON(I3,L3,J3,K3,4,16, &
                           NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)

            found=.false.
            DO J=NPHI+1, NPHIT    ! search new dihedral to see if dihedral is found
               I4=IP(J)
               J4=JP(J)
               K4=KP(J)
               L4=LP(J)

!write(outu,*) '     INNER LOOP AT 17: J,I4,J4,K4,L4,NPHIT',J,I4,J4,K4,L4,NPHIT

               CALL MKIMCANON(I4,L4,J4,K4,4,17, &
                           NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
               I5=LP(J)
               J5=KP(J)
               K5=JP(J)
               L5=IP(J)
               CALL MKIMCANON(I5,L5,J5,K5,4,18, &
                           NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)

               IF(J4.EQ.J2) then ! 2nd atoms match
                 IF(K4.EQ.K2) then ! 3rd atoms match
                   IF(I4.EQ.I2) then ! 1st atoms match
                     IF(L4.EQ.L2) then ! 4th atoms match
                       found=.true.
                       IF(IAND(ICPAUTO(I),12).gt.0) then
                         IP(J)=0 ! remove unwanted angle
                         removed=.true.
                       ENDIF
                     ENDIF
                   ENDIF
                 ENDIF
               ENDIF

               IF(J4.EQ.J3) then ! 2nd atoms match
                 IF(K4.EQ.K3) then ! 3rd atoms match
                   IF(I4.EQ.I3) then ! 1st atoms match
                     IF(L4.EQ.L3) then ! 4th atoms match
                       found=.true.
                       IF(IAND(ICPAUTO(I),12).gt.0) then
                         IP(J)=0 ! remove unwanted angle
                         removed=.true.
                       ENDIF
                     ENDIF
                   ENDIF
                 ENDIF
               ENDIF

               IF(J5.EQ.J2) then ! 2nd atoms match
                 IF(K5.EQ.K2) then ! 3rd atoms match
                   IF(I5.EQ.I2) then ! 1st atoms match
                     IF(L5.EQ.L2) then ! 4th atoms match
                       found=.true.
                       IF(IAND(ICPAUTO(I),12).gt.0) then
                         IP(J)=0 ! remove unwanted angle
                         removed=.true.
                       ENDIF
                     ENDIF
                   ENDIF
                 ENDIF
               ENDIF

               IF(J5.EQ.J3) then ! 2nd atoms match
                 IF(K5.EQ.K3) then ! 3rd atoms match
                   IF(I5.EQ.I3) then ! 1st atoms match
                     IF(L5.EQ.L3) then ! 4th atoms match
                       found=.true.
                       IF(IAND(ICPAUTO(I),12).gt.0) then
                         IP(J)=0 ! remove unwanted angle
                         removed=.true.
                       ENDIF
                     ENDIF
                   ENDIF
                 ENDIF
               ENDIF

            ENDDO
            if(.not.found) then
               IF(IAND(ICPAUTO(I),12).eq.0) then ! add this dihedral
                 NPHIT=NPHIT+1
                 IP(NPHIT)=IPAUTO(I)
                 JP(NPHIT)=JPAUTO(I)
                 KP(NPHIT)=KPAUTO(I)
                 LP(NPHIT)=LPAUTO(I)
               ELSE   
                                   ! angle to deleted could not be found.
                 IF(WRNLEV.GE.2)then
                   write(outu,*) 'Requested dihedral to delete not found:', &
                                  IPAUTO(I),JPAUTO(I),KPAUTO(I),LPAUTO(I)
                   CALL WRNDIE(0,'<AUTGEN>','Requested dihedral to delete not Found. Nothing done')
                 ENDIF
               ENDIF
            endif
         endif
      ENDDO

!write (outu,*) 'Testing AUTOGEN G:',ntheta,nphi

 ! Remove any unwanted angles that were flagged above
 ! Also, if requested, search through all new dihedrals and weed out 
 ! the ones with linear angles based on the parameters.

      j=nphi
      do i=nphi+1,nphit
        skip=(IP(I).le.0)
        if(.not.skip) then
           if(iand(ICAAUTO(JP(i)),16).eq.0 .and. iand(ICAAUTO(KP(i)),16).eq.0) then   
             CALL CHECKDH(IP(i),JP(i),KP(i),LP(i),SKIP)
           endif
        endif
        if(.not.skip) then
           j=j+1
           IP(j)=IP(i)
           JP(j)=JP(i)
           KP(j)=KP(i) 
           LP(j)=LP(i) 
        endif
      enddo
      nphit=j
      NIMDIH=nphit-nphi


!write (outu,*) 'Testing AUTOGEN H:',ntheta,nphi


    ENDIF ! IF(QAUTOG)

    IF(NIMANG.GT.0) THEN
      if(QSORT) then
        if(.NOT.QAUTOG) then
          DO I = 1,NIMANG
             J=I+NTHETA
             CALL MKIMCANON(IT(J),KT(J),JT(J),0,3,19, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
          ENDDO
        endif
        CALL SORT(NIMANG,EXCH5,ORDER5,IT(NTHETA+1),JT(NTHETA+1),KT(NTHETA+1),0,0,0,0,3)
      endif
      if(QREMDUP) then
         IPT=NTHETA+1
         JPT=NTHETA+1
         DO I=2,NIMANG
           IF(IT(IPT).NE.IT(IPT+1) .OR. JT(IPT).NE.JT(IPT+1) &
                                   .OR. KT(IPT).NE.KT(IPT+1)) JPT=JPT+1
           IPT=IPT+1
           IF(IPT.GT.JPT) THEN
              IT(JPT)=IT(IPT)
              JT(JPT)=JT(IPT)
              KT(JPT)=KT(IPT)
           ENDIF
         ENDDO
!   we just deleted IPT-JPT angles
         IF(PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'angles'
         NTHETT=JPT
         NIMANG=NTHETT-NTHETA
      endif
    ENDIF




!    write(6,281) NIMEXCL,4
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)


       IF(NIMEXCL.GT.MAXIMEX) THEN
          IF(WRNLEV.GE.2) WRITE(OUTU,988) ! out of space
988 FORMAT(' <IMAUTGEN>: Ran out of space for exclusions. Increase MAXIMEX.')
          CALL DIE
       ENDIF

! Sort angles and remove any duplicates.

!  do i=1,nimang
!    write(6,666) i,IT(NTHETA+i),JT(NTHETA+i),KT(NTHETA+i)
!666 format(' angles before sorting:',4I5)
!  enddo



! Sort dihedrals and remove any duplicates.

    IF(NIMDIH.GT.0) THEN
      if(QSORT) then
        if(.NOT.QAUTOG) then
          DO I = 1,NIMDIH
             J=I+NPHI
             CALL MKIMCANON(IP(J),LP(J),JP(J),KP(J),4,20, &
                            NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
          ENDDO
        endif
        CALL SORT(NIMDIH,EXCH5,ORDER5,IP(NPHI+1),JP(NPHI+1),KP(NPHI+1),LP(NPHI+1),0,0,0,4)
      endif
      if(QREMDUP) then
         IPT=NPHI+1
         JPT=NPHI+1
         DO I=2,NIMDIH
           IF(IP(IPT).NE.IP(IPT+1) .OR. JP(IPT).NE.JP(IPT+1) .OR. &
              KP(IPT).NE.KP(IPT+1) .OR. LP(IPT).NE.LP(IPT+1)) JPT=JPT+1
           IPT=IPT+1
           IF(IPT.GT.JPT) THEN
              IP(JPT)=IP(IPT)
              JP(JPT)=JP(IPT)
              KP(JPT)=KP(IPT)
              LP(JPT)=LP(IPT)
           ENDIF
         ENDDO
!   we just deleted IPT-JPT dihedrals
         IF(PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'dihedrals'
         NPHIT=JPT
         NIMDIH=NPHIT-NPHI
      endif
    ENDIF

! sort and search exclusions for duplicates (rings = type 4). Remove duplicates.

    IF(NIMEXCL.GT.0) THEN
      CALL SORT(NIMEXCL,EXCH5,ORDER5,IMEXCLJ,IMEXCLI,IMEXCLT,0,0,0,0,3)

!    write(6,281) NIMEXCL,5
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)

      IPT=1
      JPT=1
      DO I=2,NIMEXCL
         IF(IMEXCLI(IPT).NE.IMEXCLI(IPT+1) .OR. IMEXCLJ(IPT).NE.IMEXCLJ(IPT+1)) THEN
            JPT=JPT+1
         ELSE
            IMEXCLT(IPT+1)=IMEXCLT(IPT) ! It's a ring. Keep smaller value 
         ENDIF
         IPT=IPT+1
         IF(IPT.GT.JPT) THEN
            IMEXCLI(JPT)=IMEXCLI(IPT)
            IMEXCLJ(JPT)=IMEXCLJ(IPT)
            IMEXCLT(JPT)=IMEXCLT(IPT)
         ENDIF
      ENDDO
!   we just deleted IPT-JPT exclusions
     IF(PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'exclusions'
      NIMEXCL=JPT

!    write(6,281) NIMEXCL,6
!    write(6,282) 'IMEXCLI:',(IMEXCLI(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLJ:',(IMEXCLJ(I),I=1,NIMEXCL)
!    write(6,282) 'IMEXCLT:',(IMEXCLT(I),I=1,NIMEXCL)

    ENDIF
  
! now check impropers and crossterms for any duplicates. Remove, if requested.


!  
!  sort impropers and remove any duplicates
!
    IF(NIMIMP.GT.0) THEN
      if(QSORT) then
        CALL SORT(NIMIMP,EXCH5,ORDER5,IM(NIMPHI+1),JM(NIMPHI+1),KM(NIMPHI+1),LM(NIMPHI+1),0,0,0,4)
      endif
      if(QREMDUP) then
        IPT=NIMPHI+1
        JPT=NIMPHI+1
        DO I=2,NIMIMP
          IF(IM(IPT).NE.IM(IPT+1) .OR. JM(IPT).NE.JM(IPT+1) .OR. &
             KM(IPT).NE.KM(IPT+1) .OR. LM(IPT).NE.LM(IPT+1)) JPT=JPT+1
          IPT=IPT+1
          IF(IPT.GT.JPT) THEN
             IM(JPT)=IM(IPT)
             JM(JPT)=JM(IPT)
             KM(JPT)=KM(IPT)
             LM(JPT)=LM(IPT)
          ENDIF
        ENDDO
!   we just deleted IPT-JPT impropers
        IF(PRNLEV.GE.2) WRITE(OUTU,25) IPT-JPT,'impropers'
        NIMPHT=JPT
        NIMIMP=NIMPHT-NIMPHI
      endif
    ENDIF
!
    RETURN

contains

    subroutine add_to_bond_list

!    write(6,214) IBT,JBT
!214 format('Adding to bond list: IBT,JBT:',20I5) 

         IF(IBT.GT.0 .AND. JBT.GT.0) THEN 
           NATBON(IBT)=NATBON(IBT)+1
           IF(NATBON(IBT).GT.MAXCON) THEN
              IF(WRNLEV.GE.2) WRITE(OUTU,35) IBT
              CALL DIEWRN(-4)
           ENDIF
           IATBON(NATBON(IBT),IBT)=JBT
           IATBONCD(NATBON(IBT),IBT)=ICD

           NATBON(JBT)=NATBON(JBT)+1
           IF(NATBON(JBT).GT.MAXCON) THEN
              IF(WRNLEV.GE.2) WRITE(OUTU,35) JBT
              CALL DIEWRN(-4)
           ENDIF
           IATBON(NATBON(JBT),JBT)=IBT
           IATBONCD(NATBON(JBT),JBT)=ICD

         ENDIF
35  FORMAT(' <IMAUTGEN>: Too many bonds for atom',I5,' Check code')
         return
    end subroutine add_to_bond_list

   END SUBROUTINE IMAUTGEN2


   SUBROUTINE MKIMCANON(I,J,K,L,MODE,code, &
                        NATOMT,NTRANS,IMATTR,ITRANS,TRONTR,IGMATR)
! 
! This routine redefines an internal energy term list so that the first atom 
! is in the primary space, and the last atom is in the lower transformation#
! of two possible choices.
!
!   where mode=2,3,4,8 (make work if both i and j are image atoms)
!     bonds      -- MKIMCANON(i,j,0,0,0,0,0,0,2,...)
!     angles     -- MKIMCANON(i,k,j,0,0,0,0,0,3,...)
!     dihedrals  -- MKIMCANON(i,l,j,k,0,0,0,0,4,...) (also swap j and k when swapping i and l)
!     exclusions -- MKIMCANON(i,j,0,0,0,0,0,0,2,...)
!     
!
! when calling, do not use loop counters for I,J,K,L
!

  use exfunc
  use stream
  use image,only:limall

  implicit none

  integer I,J,K,L,MODE,NATOMT,NTRANS,code
  integer IMATTR(*),ITRANS(*),TRONTR(0:NTRANS,0:NTRANS),IGMATR(NATOMT,0:NTRANS)


  integer IP,JP,KP,LP,MP,IX,I1,J1
  integer ITR,JTR,KTR,LTR,MTR,IXTR,JXTR,MXTR,NTUSED
  logical SWAP
  integer, parameter :: MARK=99999999 


!write(6,*) ' Calling MKIMCANON: I,J,K,L,MODE,code',I,J,K,L,MODE,code



  I1=I
  J1=J
!
! get primary atom numbers for I and J
     IP=IMATTR(I)
     JP=IMATTR(J)
! get transformation numbers
     ITR=ITRANS(I)
     JTR=ITRANS(J)
     NTUSED=1
     IF(ITR.NE.JTR) NTUSED=2
     IF(MODE.GE.3) THEN
        KP=IMATTR(K)
        KTR=ITRANS(K)
        IF(KTR.NE.ITR .AND. KTR.NE.JTR) NTUSED=NTUSED+1
     ENDIF
     IF(MODE.GE.4) THEN
        LP=IMATTR(L)
        LTR=ITRANS(L)
        IF(LTR.NE.ITR .AND. LTR.NE.JTR .AND. LTR.NE.KTR) NTUSED=NTUSED+1
     ENDIF
     IF(NTUSED>2 .and. .not.LIMALL) then
       ! we have a situation where there is complex patching, but IMALL is not invoked.
       ! This might be OK, but could be a mess.  Too difficult to figure out here...
       CALL WRNDIE(-4,'<MKIMCANON','Complex image autogenerate without IMALL. Please use IMALL')
     ENDIF

! get cross transformation numbers
     IXTR=TRONTR(ITR,JTR)
     JXTR=TRONTR(JTR,ITR)
! check for swap condition


       ! swap if I's image upon centering J is lower than J's image upon centering I
        SWAP=(IXTR.LT.JXTR)
       ! swap if both atoms are primary and J is before I
!!        IF(ITR.EQ.0 .AND. JTR.EQ.0) SWAP=(I.GT.J)
        IF(IXTR.EQ.JXTR) SWAP=(I.GT.J)
        IF(SWAP) THEN
       ! swapping now
           IX=I
           I=J
           J=IX
           ITR=ITRANS(I)
           IF(MODE.EQ.4) THEN
       ! for dihedrals, also swap inner atoms when swapping outer atoms
              IX=K
              K=L
              L=IX
           ENDIF
        ENDIF

     IF(ITR.LE.0) RETURN

!  Make I a primary atom (and adjust the others, if not)
        I=IMATTR(I)

           MP=IMATTR(J)
           MTR=ITRANS(J)
           MXTR=TRONTR(MTR,ITR)
           IF(MXTR.EQ.MARK) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,2,I1,J1,J,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing cross transformation')
           ENDIF
           J=IGMATR(MP,MXTR)
           if(J<0) then
             IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,102,I1,J1,J,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing 2nd atom in transformation')
           endif

        IF(MODE.GE.3) THEN
           MP=IMATTR(K)
           MTR=ITRANS(K)
           MXTR=TRONTR(MTR,ITR)
           IF(MXTR.EQ.MARK) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,3,I1,J1,K,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing cross transformation')
           ENDIF
           K=IGMATR(MP,MXTR)
           if(K<0) then
             IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,103,I1,J1,K,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing central atom after transformation')
           endif
        ENDIF

        IF(MODE.GE.4) THEN 

           MP=IMATTR(L)
           MTR=ITRANS(L)
           MXTR=TRONTR(MTR,ITR)
           IF(MXTR.EQ.MARK) THEN
           IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,4,I1,J1,L,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing cross transformation')
           ENDIF
           L=IGMATR(MP,MXTR)
           if(L<0) then
             IF(WRNLEV.GE.2) WRITE(OUTU,35) mode,104,I1,J1,L,MP,MTR,ITR,MXTR
             CALL WRNDIE(-4,'<MKIMCANON>','missing central atom after transformation')
           endif
        ENDIF

35     FORMAT('Error info: mode,I1,J1,M7,MP,MTR,ITR,MXTR:',20I5)

     RETURN
     END SUBROUTINE MKIMCANON



end module upimag_util
