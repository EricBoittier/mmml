module mbh_m
  implicit none

contains

  !========================================================
  SUBROUTINE MBH_INITBLK(NATOM,NSUBS,NBLK,BLKINFO)
    !
    ! This subroutine initiates the main variables of the
    ! Mobile Block Hessian approach (MBH).
    ! Added by An Ghysels november 2007
    !
    ! NATOM    number of atoms
    ! NSUBS    number of subsystem atoms
    ! NBLK     number of blocks
    ! BLKINFO  indicates to which block an atom belongs
    !          eg blkinfo(i)=2    atom i belongs to block 2
    !             blkinfo(i)=0    atom i is in subsystem (default)
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
    implicit none
    ! Passed variables
    INTEGER I,NATOM
    INTEGER NSUBS,NBLK
    INTEGER BLKINFO(*)

    ! initialisation of block information
    NSUBS=NATOM
    NBLK=0
    BLKINFO(1:NATOM)=0

    IF(PRNLEV.LT.2) RETURN
    WRITE(OUTU,900)
900 FORMAT(/15X,'BLOCK INFORMATION FOR MBH INITIALIZED')

    RETURN
  END SUBROUTINE MBH_INITBLK
  !========================================================
  SUBROUTINE MBH_ADDBLK(NATOM,NSUBS,NBLK,BLKINFO,ISLCT)
    !
    ! This subroutine adds another block. The added block consists
    ! of the selected atoms in islct. Variables nsubs, nblk, and
    ! blkinfo are changed.
    ! Added by An Ghysels november 2007
    !
    ! NATOM    number of atoms
    ! NSUBS    number of subsystem atoms
    ! NBLK     number of blocks
    ! BLKINFO  array that indicates to which block each atom belongs
    !          eg blkinfo(i)=2    atom i belongs to block 2
    !             blkinfo(i)=0    atom i is in subsystem (default)
    ! ISLCT    array that indicates which atoms are selected
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
    implicit none
    ! Passed variables
    INTEGER I,NATOM
    INTEGER NSUBS,NBLK,NSLCT
    INTEGER ISLCT(*),BLKINFO(*)

    ! Get nb of atoms in block
    NSLCT=0
    DO I=1,NATOM
       IF(ISLCT(I).NE.0) THEN
          NSLCT=NSLCT+1
       ENDIF
    ENDDO
    ! Check blocks
    !     Block is empty, change nothing
    IF (NSLCT.EQ.0) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,903)
903       FORMAT(/15X,'NO BLOCK ADDED, BECAUSE BLOCK IS EMPTY')
       ENDIF

       ! Block is just 1 atom, change nothing
    ELSE IF (NSLCT.EQ.1) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,904)
904       FORMAT(/15X,'NO BLOCK ADDED, BECAUSE BLOCK IS JUST 1 ATOM')
       ENDIF

       ! Block is 2 atoms, change nothing
       ! TODO  to be implemented: linear blocks !!!
    ELSE IF (NSLCT.EQ.2) THEN
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,905)
905       FORMAT(/15X,'NO BLOCK ADDED, BECAUSE BLOCK IS LINEAR')
       ENDIF
       ! FORMAT(/15X,'LINEAR BLOCKS NOT YET IMPLEMENTED')

       ! Block is 3 atoms or more, adapt block information
    ELSE
       DO I=1,NATOM
          IF(ISLCT(I).NE.0) THEN
             !
             ! TODO linked blocks not yet implemented
             ! at present: if blkinfo(i) is already initialized,
             ! it will not add the block
             !
             IF (BLKINFO(I).NE.0) THEN
                IF(PRNLEV.GE.2) THEN
                   WRITE(OUTU,902)
902                FORMAT(/15X,'NO BLOCK ADDED, BECAUSE BLOCKS OVERLAP')
                   ! FORMAT(/15X,'ATOM BELONGED ALREADY TO OTHER BLOCK')
                   RETURN
                ENDIF
             ELSE
                NSUBS=NSUBS-1
                BLKINFO(I)=NBLK+1
             ENDIF

          ENDIF
       ENDDO
       NBLK=NBLK+1
       IF(PRNLEV.GE.2) THEN
          WRITE(OUTU,901)
901       FORMAT(/15X,'BLOCK ADDED')
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE MBH_ADDBLK

  !========================================================
  SUBROUTINE MBH_GEN_BLKINFO(NATOM,NBLK,BLKINFO, &
       BLKREORDER,BLKSIZES,BLKCUMSIZES)
    !
    ! This subroutine generates all necessary block information
    ! before the actual Mobile Block Hessian (MBH) approach starts.
    ! According to the values of nblk/blkinfo, the subroutine fills
    ! the variables blkreorder, blksizes and blkcumsizes.
    ! Added by An Ghysels november 2007
    !
    ! NATOM    number of atoms
    ! NBLK     number of blocks
    ! BLKINFO  array that indicates to which block each atom belongs
    !          e.g. blkinfo(i)=2    atom i belongs to block 2
    !               blkinfo(i)=0    atom i is in subsystem (default)
    ! BLKSIZES        array of length nblk,
    !                 blksizes(k) is nb of atoms in block k
    ! BLKCUMSIZES     array of length nblk+1,
    !                 blkcumsizes(k) is cumulative nb of atoms
    !                 in previous k-1 blocks
    ! BLKREORDER      reordering array of length natom, that gets
    !                 first the atoms of block 1, then block 2, and
    !                 so on, and finally the subsystem atoms
    !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use stream
  use gamess_fcm,only: qmused_qchem
    implicit none
    ! Passed variables
    INTEGER NATOM,NBLK
    INTEGER BLKINFO(NATOM),BLKREORDER(NATOM)
    INTEGER BLKSIZES(NBLK),BLKCUMSIZES(NBLK+1)
    ! Local variables
    INTEGER IPT,I,K,BLKNATOM
    ! BLKNATOM    nb of atoms in certain block

    ! fill blksizes, blkcumsizes, blkreorder
    IPT=0
    BLKCUMSIZES(1)=0
    DO K=1,NBLK
       BLKNATOM=0
       DO I=1,NATOM
          IF(BLKINFO(I).EQ.K) THEN
             BLKNATOM=BLKNATOM+1
             IPT=IPT+1
             BLKREORDER(IPT)=I
          ENDIF
       ENDDO
       BLKSIZES(K)=BLKNATOM
       BLKCUMSIZES(K+1)=BLKCUMSIZES(K)+BLKNATOM
    ENDDO

    ! add subs atoms to blkreorder array
    DO I=1,NATOM
       IF(BLKINFO(I).EQ.0) THEN
          IPT=IPT+1
          BLKREORDER(IPT)=I
       ENDIF
    ENDDO
    !
    !      IF(PRNLEV.LT.2) RETURN
    !      WRITE(OUTU,902)
    ! 902  FORMAT(/15X,'Prepared for MBH...')
    !

    ! Add QCHEM routine...
    if(qmused_qchem) CALL MBH_GEN_QCHEM_FCM(NATOM,NBLK,BLKREORDER,BLKSIZES)

    RETURN
  END SUBROUTINE MBH_GEN_BLKINFO

  !========================================================

  SUBROUTINE MBH_GEN_QCHEM_FCM(NATOM,NBLK,BLKREORDER,BLKSIZES)
  use chm_kinds
  use dimens_fcm
  use gamess_fcm
    implicit none

    INTEGER BLKSIZES(*), BLKREORDER(*)
    INTEGER NBLK,NATOM

    QCBLKREORDER(1:NATOM)=BLKREORDER(1:NATOM)

    QCBLKSIZES(1:NBLK)=BLKSIZES(1:NBLK)

    QCNBLK=NBLK

    RETURN
  END SUBROUTINE MBH_GEN_QCHEM_FCM

  !========================================================

  SUBROUTINE MBH(NATOM,NSUBS,NBLK,BLKINFO,X,Y,Z,NAT3, &
       BNBND,BIMAG,NFREQ,AMASS, &
       DDV,DDM,DDF,DDEV,DDSCR,NADD,LRAISE, &
       LFINIT,STEP,LDSCF,LENTRO)
    !
    ! This routine does a vibrational analysis with the
    ! (Multiple) Mobile Block Hessian approach (MBH).
    !
    ! By An Ghysels  november 2007
    !
    ! See Journal of Chemical Physics 126, 224102 (2007)
    ! and Journal of Chemical Physics 127, 164108 (2007)
    !
    ! * do energy calculation: fill dd1 with second derivatives
    ! * do all heaping, stacking...
    ! * set up all block information variables
    ! * perform MBH
    ! * free allocated space
    !
    ! Notations:
    !
    ! NATOM           number of atoms
    ! NSUBS           number of subsystem atoms
    ! NBLK            number of blocks
    ! BLKINFO         array that indicates to which block each atom belongs
    !                 e.g. blkinfo(i)=2    atom i belongs to block 2
    !                      blkinfo(i)=0    atom i is in subsystem (default)
    ! BLKSIZES        array of length nblk,
    !                 blksizes(k) is nb of atoms in block k
    ! BLKCUMSIZES     array of length nblk+1,
    !                 blkcumsizes(k) is cumulative nb of atoms
    !                 in previous k-1 blocks
    ! BLKREORDER      reordering array of length natom, that gets
    !                 first the atoms of block 1, then block 2, and
    !                 so on, and finally the subsystem atoms

    ! X,Y,Z           Cartesian coordinates
    ! NAT3            3*nb of atoms
    ! BNBND
    ! BIMAG
    ! NFREQ           nb of freqs
    ! DDV             eigenvectors (modes)
    ! DDM             mass matrix
    ! DDF             frequencies
    ! DDEV            eigenvalues
    ! DDSCR           a scratch matrix
    ! NADD            offset nb of freqs
    ! LFINIT          whether derivatives are calculated with finit diffs
    ! STEP
    ! LDSCF
    ! LENTRO          whether to perform an entropy analysis

    ! Defaults:
    ! nadd   = 0
    ! nfre = nat3 = nfreq (see vibran.src)
    ! lentro = false (see vibran.src)
    ! lfinit = false (see vibran.src)
    ! lraise = false (always)
    ! bnbnd  =
    ! bimag  =
    ! ldscf  =
    !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use exfunc
  use deriv
  use consta
  use energym
  use stream
  use memory
  use vibcom, only: gensd2
  use vibsub, only: entropds
    implicit none

    ! Passed variables
    INTEGER NATOM,NSUBS,NBLK,NFREQ,NAT3,NADD
    type(nonbondDataStructure) BNBND
    type(imageDataStructure) BIMAG
    INTEGER BLKINFO(*)
    real(chm_real) :: X(:), Y(:), Z(:)
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)

    LOGICAL LFINIT,LDSCF,LENTRO,LRAISE
    real(chm_real)  STEP

    ! Local variables
    INTEGER I,NATP,N6
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    real(chm_real),allocatable,dimension(:) :: XREF,YREF,ZREF
    real(chm_real),allocatable,dimension(:) :: DXF,DYF,DZF
    INTEGER ISPACE,JSPACE,KSPACE,N3S
    ! extra for block things (array names used as pointers)
    INTEGER MBHDIM,MBHDIM6
    integer,allocatable,dimension(:) :: BLKREORDER,BLKSIZES,BLKCUMSIZES
    real(chm_real),allocatable,dimension(:) :: DD1,DDS
    real(chm_real),allocatable,dimension(:) :: HEFF,VHEFF
    INTEGER MEFF,VMEFF
    real(chm_real),allocatable,dimension(:) :: MBLK,MSUBS

    ! ******END of declarations***********

    ! dimension of arrays
    MBHDIM=6*NBLK+3*NSUBS
    N6=(NAT3*(NAT3+1))/2
    MBHDIM6=(MBHDIM*(MBHDIM+1))/2
    NFREQ=MBHDIM
    ! TODO maybe include here some checking... for nb of freqs
    ! IF(NFREQ.GT.MBHDIM-NADD) NFREQ=MBHDIM-NADD
    ! IF(NFREQ.LE.0) RETURN
    !

    ! -------------------------------
    ! ALLOCATE NEEDED SPACE
    ! -------------------------------
    ! for MBH matrices
    ISPACE=MBHDIM6
    call chmalloc('mbh.src','MBH','HEFF',ISPACE,crl=HEFF)
    JSPACE=MBHDIM**2
    call chmalloc('mbh.src','MBH','VHEFF',JSPACE,crl=VHEFF)
    KSPACE=NSUBS
    call chmalloc('mbh.src','MBH','MSUBS',KSPACE,crl=MSUBS)
    KSPACE=21*NBLK
    call chmalloc('mbh.src','MBH','MBLK',KSPACE,crl=MBLK)
    ! for diagonalizations: dds,...
    !     ddv,ddm,ddf,ddev,ddscr are already heaped - size determined by nnmds
    !     dds is usually scratch vector of 8*(nat3+1) (8 times nat3+1)
    !         use here mbhdim instead of nat3
    KSPACE=(MBHDIM+1)*8
    call chmalloc('mbh.src','MBH','DDS',KSPACE,crl=DDS)
    ! for block things
    call chmalloc('mbh.src','MBH','BLKREORDER',NATOM,intg=BLKREORDER)
    call chmalloc('mbh.src','MBH','BLKSIZES',NBLK,intg=BLKSIZES)
    call chmalloc('mbh.src','MBH','BLKCUMSIZES',NBLK+1,intg=BLKCUMSIZES)

    ! -------------------------------
    ! BLOCK INFO GENERATION
    ! -------------------------------
    CALL MBH_GEN_BLKINFO(NATOM,NBLK,BLKINFO, &
         BLKREORDER,BLKSIZES,BLKCUMSIZES)

    ! -------------------------------
    ! DO ENERGY CALCULATION
    ! -------------------------------
    IF (NAT3.GT.3600) CALL WRNDIE(-1,'<MBH>', &
         'NAT3 for second derivatives is >3600')
    call chmalloc('mbh.src','MBH','DD1',N6,crl=DD1)
    DD1(1:N6)=0.0
    ! -------------------------------
    ! calculate energy, gradient, second derivatives.
    IF(LFINIT) THEN
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,0)
       call chmalloc('mbh.src','MBH','XREF',NATOM,crl=XREF)
       call chmalloc('mbh.src','MBH','YREF',NATOM,crl=YREF)
       call chmalloc('mbh.src','MBH','ZREF',NATOM,crl=ZREF)
       call chmalloc('mbh.src','MBH','DXF',NATOM,crl=DXF)
       call chmalloc('mbh.src','MBH','DYF',NATOM,crl=DYF)
       call chmalloc('mbh.src','MBH','DZF',NATOM,crl=DZF)

       CALL GENSD2(NAT3,X,Y,Z,XREF,YREF,ZREF, &
            DD1,DXF,DYF,DZF,STEP,LDSCF, &
            3*NATOM,.FALSE.) ! JZ_UW12)

       call chmdealloc('mbh.src','MBH','DZF',NATOM,crl=DZF)
       call chmdealloc('mbh.src','MBH','DYF',NATOM,crl=DYF)
       call chmdealloc('mbh.src','MBH','DXF',NATOM,crl=DXF)
       call chmdealloc('mbh.src','MBH','ZREF',NATOM,crl=ZREF)
       call chmdealloc('mbh.src','MBH','YREF',NATOM,crl=YREF)
       call chmdealloc('mbh.src','MBH','XREF',NATOM,crl=XREF)
    ELSE
       CALL ENERGY(X,Y,Z,DX,DY,DZ,BNBND,BIMAG,1,NAT3,DD1)
    ENDIF

    IF(PRNLEV.GE.2) THEN
       CALL PRINTE(OUTU, EPROP, ETERM, 'VIBR', 'ENR', .TRUE., &
            1, ZERO, ZERO, .TRUE.)
    ENDIF

    ! Remove translational/rotational modes from Cartesian Hessian if requested
    ! Do NOT mass-weight - that will happen later.
    ! use ddv as a scratch vector
    ! raise both trans and rot modes
    IF(LRAISE) THEN
       CALL RAISE_BEFOREMW(.TRUE.,.TRUE.,NAT3,NATOM,AMASS,DD1, &
            DDV,X,Y,Z,0,0,0,.FALSE.)
    ENDIF

    ! -------------------------------
    ! Do MBH
    ! -------------------------------
    CALL MULTMBH(MBHDIM,NATOM,NSUBS,NBLK,BLKINFO, &
         BLKREORDER,BLKSIZES, &
         BLKCUMSIZES,X,Y,Z,DX,DY,DZ,NAT3,NFREQ, &
         AMASS,DDV,DDM,DDF,DDEV,DDSCR,DD1,DDS, &
         NADD,HEFF,VHEFF,MSUBS,MBLK)

    ! -------------------------------
    ! FREE ALLOCATED SPACE
    ! -------------------------------
    call chmdealloc('mbh.src','MBH','DD1',N6,crl=DD1)
    ! for MBH matrices
    ISPACE=MBHDIM6
    call chmdealloc('mbh.src','MBH','HEFF',ISPACE,crl=HEFF)
    JSPACE=MBHDIM**2
    call chmdealloc('mbh.src','MBH','VHEFF',JSPACE,crl=VHEFF)
    KSPACE=NSUBS
    call chmdealloc('mbh.src','MBH','MSUBS',KSPACE,crl=MSUBS)
    KSPACE=21*NBLK
    call chmdealloc('mbh.src','MBH','MBLK',KSPACE,crl=MBLK)
    ! for diagonalizations
    KSPACE=(MBHDIM+1)*8
    call chmdealloc('mbh.src','MBH','DDS',KSPACE,crl=DDS)
    ! for block things
    call chmdealloc('mbh.src','MBH','BLKCUMSIZES',NBLK+1,intg=BLKCUMSIZES)
    call chmdealloc('mbh.src','MBH','BLKSIZES',NBLK,intg=BLKSIZES)
    call chmdealloc('mbh.src','MBH','BLKREORDER',NATOM,intg=BLKREORDER)

    IF(PRNLEV.LT.2) RETURN
    WRITE(OUTU,903)
903 FORMAT(/15X,'MOBILE BLOCK HESSIAN - NMA COMPLETED')
    WRITE(OUTU,904)
904 FORMAT(/15X,'FREQUENCIES'/)
    CALL FREQPR(DDF,NFREQ,OUTU)

    ! entropy calculation
    IF(LENTRO) THEN
       CALL ENTROPDS(X,Y,Z,NAT3,NFREQ,AMASS,DDF,NADD,LRAISE)
    ENDIF

    RETURN
  END SUBROUTINE MBH

  !========================================================

  SUBROUTINE MULTMBH(MBHDIM,NATOM,NSUBS,NBLK,BLKINFO, &
       BLKREORDER,BLKSIZES,BLKCUMSIZES, &
       X,Y,Z,DX,DY,DZ, &
       NAT3,NFREQ,AMASS, &
       DDV,DDM,DDF,DDEV,DDSCR,DD1,DDS, &
       NADD,HEFF,VHEFF,MSUBS,MBLK)
    !
    ! This subroutine does the actual Mobile Block Hessian
    ! (MBH) calculation.
    !
    ! By An Ghysels  november 2007
    !
    ! See Journal of Chemical Physics 126, 224102 (2007)
    ! and Journal of Chemical Physics 127, 164108 (2007)
    !
    !          S C H E M E
    !
    ! * Calculate effective Hessian matrix (MBH) HEFF.
    !   This is the first step because dd1 will be destroyed by diagq.
    !   Store in UPT (upper triangular) form.
    !
    ! * Calculate effective mass matrix (MBH) MEFF.
    !   Inverse square root of this matrix is stored in two parts
    !                 - MBLK
    !                 - MSUBS
    !   MSUBS (dim: NSUBS) contains inverse square root of each atom
    !                      in the subsystem
    !   MBLK (dim: 21*NBLK) : do for each block
    !              1 construct 6x6 mass matrix of block in UPT form
    !              2 construct inverse square root of this 6x6 mass matrix
    !              3 add the 21 elements to MBLK
    !
    ! * Mass weighting.
    !   MEFF^-.5 . HEFF . MEFF^-.5 is stored in HEFF
    !   Is calculated in 3 parts.
    !
    ! * Diagonalization of (mass weighted) HEFF
    !
    ! * Back transformation
    !   collect eigenvalues/eigenfrequencies in DDEV/DDF
    !   collect eigenvectors in DDEV
    !

  use chm_kinds
  use dimens_fcm
  use number
  use vector
  use consta
  use stream

    implicit none

    ! Passed variables
    INTEGER MBHDIM,NATOM,NSUBS,NBLK,BLKINFO(*),NAT3,NFREQ,NADD
    INTEGER BLKREORDER(*),BLKSIZES(NBLK),BLKCUMSIZES(NBLK+1)
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real) AMASS(*)
    real(chm_real) DDV(NAT3,*),DDM(*),DDF(*),DDEV(*),DDSCR(*)
    real(chm_real) DD1(*),DDS(*)
    real(chm_real) HEFF(*)
    real(chm_real) VHEFF(MBHDIM,MBHDIM)

    ! Local variables
    INTEGER MBHDIM6
    INTEGER NATP,N6,I,J,K,I2,J2,I3,J3,I4,J4,I5,J5
    INTEGER IST,JST
    INTEGER II,JJ,KK,IPT,JPT,KPT
    INTEGER IH1,IH2,IH3,IH4,IH5,IH6,IH7,IH8
    real(chm_real) VCUT,AMS
    real(chm_real) ALLDDTROT(NAT3,6),P,Q,R,S,T,U
    real(chm_real) BLKMASS
    ! !!!!!!!!!!! changed
    real(chm_real) TMP,TMP_21(21),VTMP_21(6,6),TMP_36(6,6),TMP_6(6)
    real(chm_real) MSUBS(NSUBS),MBLK(21*NBLK)

    !**********END of declarations********

    MBHDIM6=(MBHDIM*(MBHDIM+1))/2
    VCUT=TENM5
    ! NATOM=NAT3/3

    ! -------------------------------
    ! Construct allddtrot
    ! -------------------------------
    ! construct the 6 translation/rotation vectors
    DO K=1,6
       DO I=1,NATOM
          DDSCR(3*I-2)=0.0
          DDSCR(3*I-1)=0.0
          DDSCR(3*I)=0.0
          IF(K.EQ.1) THEN
             DDSCR(3*I-2)=1.0
          ELSE IF(K.EQ.2) THEN
             DDSCR(3*I-1)=1.0
          ELSE IF(K.EQ.3) THEN
             DDSCR(3*I)=1.0
          ELSE IF(K.EQ.4) THEN
             DDSCR(3*I-1)=-Z(I)
             DDSCR(3*I)=Y(I)
          ELSE IF(K.EQ.5) THEN
             DDSCR(3*I-2)=Z(I)
             DDSCR(3*I)=-X(I)
          ELSE
             DDSCR(3*I-2)=-Y(I)
             DDSCR(3*I-1)=X(I)
          ENDIF
          ALLDDTROT(3*I-2,K)=DDSCR(3*I-2)
          ALLDDTROT(3*I-1,K)=DDSCR(3*I-1)
          ALLDDTROT(3*I,K)=DDSCR(3*I)
       ENDDO
    ENDDO
    !
    ! length of ddscr vector is minimum nat3, so should be okay
    ! Note:
    ! not mass weighted
    ! and not orthogonalized (so does not work if linear dependent)
    ! and not normalized
    ! and actually not necessary to use DDSCR ??!!!!!! TODO
    !
    ! -------------------------------
    ! CONSTRUCT H EFFECTIVE (the MBH Hessian matrix)
    ! -------------------------------
    !
    ! Construct heff first, because dd1 will be destroyed by diagonalizations
    !
    HEFF(1:MBHDIM6)=0.0

    ! blk-blk part
    !     IPT pointer in Heff, mbhdim UPT
    !     JPT pointer in H cartesian, nat3 UPT
    DO I=1,NBLK
       DO J=I,NBLK
          DO II=1,6
             IF(I.EQ.J) THEN
                JST=II
             ELSE
                JST=1
             ENDIF
             DO JJ=JST,6

                I2=6*(I-1)+II
                J2=6*(J-1)+JJ
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2
                !  every index must contribute to tmp !
                TMP=0.0
                DO I3=1,BLKSIZES(I)
                   DO J3=1,BLKSIZES(J)
                      I4=BLKREORDER(BLKCUMSIZES(I)+I3)
                      J4=BLKREORDER(BLKCUMSIZES(J)+J3)
                      DO I5=1,3
                         DO J5=1,3
                            I2=3*(I4-1)+I5
                            J2=3*(J4-1)+J5
                            IF (I2.LE.J2) THEN
                               JPT=NAT3*I2+J2-NAT3+(I2-I2**2)/2
                            ELSE
                               JPT=NAT3*J2+I2-NAT3+(J2-J2**2)/2
                            ENDIF
                            TMP=TMP+DD1(JPT)*ALLDDTROT(I2,II) &
                                 *ALLDDTROT(J2,JJ)
                         ENDDO
                      ENDDO
                   ENDDO
                ENDDO

                HEFF(IPT)=TMP
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ! EXTRA blk-blk part CORRECTION for gradients
    ! This correction is essential if structure is partially optimized
    DO I=1,NBLK
       P=0.0
       Q=0.0
       R=0.0
       S=0.0
       T=0.0
       U=0.0
       DO J=1,BLKSIZES(I)
          I4=BLKREORDER(BLKCUMSIZES(I)+J)
          P=P+DX(I4)*X(I4)
          Q=Q+DY(I4)*Y(I4)
          R=R+DZ(I4)*Z(I4)
          S=S+DY(I4)*X(I4)
          T=T+DZ(I4)*X(I4)
          U=U+DZ(I4)*Y(I4)
       ENDDO
       ! write(6,'(A)') 'p,q,r'
       ! write (6,'(6F15.8)') P,Q,R,S,T,U

       I2=6*(I-1)+4
       IPT=MBHDIM*I2+I2-MBHDIM+(I2-I2**2)/2
       HEFF(IPT)=HEFF(IPT)-Q-R
       HEFF(IPT+1)=HEFF(IPT+1)+S
       HEFF(IPT+2)=HEFF(IPT+2)+T

       I2=6*(I-1)+5
       IPT=MBHDIM*I2+I2-MBHDIM+(I2-I2**2)/2
       HEFF(IPT)=HEFF(IPT)-P-R
       HEFF(IPT+1)=HEFF(IPT+1)+U

       I2=6*(I-1)+6
       IPT=MBHDIM*I2+I2-MBHDIM+(I2-I2**2)/2
       HEFF(IPT)=HEFF(IPT)-P-Q
    ENDDO

    ! subs-subs part
    !     IPT index in Heff, mbhdim UPT
    !     JPT index in Hessian, nat3 UPT
    !     KPT total nb of atoms in blocks
    KPT=NATOM-NSUBS
    DO I=1,NSUBS
       DO J=I,NSUBS
          I4=BLKREORDER(KPT+I)
          J4=BLKREORDER(KPT+J)
          DO I3=1,3
             IF(I.EQ.J) THEN
                JST=I3
             ELSE
                JST=1
             ENDIF
             DO J3=JST,3

                I2=6*NBLK+3*(I-1)+I3
                J2=6*NBLK+3*(J-1)+J3
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2

                I2=3*(I4-1)+I3
                J2=3*(J4-1)+J3
                IF (I2.LE.J2) THEN
                   JPT=NAT3*I2+J2-NAT3+(I2-I2**2)/2
                ELSE
                   JPT=NAT3*J2+I2-NAT3+(J2-J2**2)/2
                ENDIF

                HEFF(IPT)=DD1(JPT)
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ! blk-subs part
    !     I,II indices blk, nblk
    !     J,JJ indices subs, nsubs
    !     IPT index in Heff, mbhdim UPT
    !     JPT index in Hessian, nat3 UPT
    !     KPT total nb of atoms in blocks
    KPT=NATOM-NSUBS
    DO I=1,NBLK
       DO J=1,NSUBS
          DO II=1,6
             DO JJ=1,3

                I2=6*(I-1)+II
                J2=6*NBLK+3*(J-1)+JJ
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2

                TMP=0.0
                DO I3=1,BLKSIZES(I)
                   DO I5=1,3
                      I4=BLKREORDER(BLKCUMSIZES(I)+I3)
                      J4=BLKREORDER(KPT+J)
                      I2=3*(I4-1)+I5
                      J2=3*(J4-1)+JJ
                      IF (I2.LE.J2) THEN
                         JPT=NAT3*I2+J2-NAT3+(I2-I2**2)/2
                      ELSE
                         JPT=NAT3*J2+I2-NAT3+(J2-J2**2)/2
                      ENDIF
                      TMP=TMP+DD1(JPT)*ALLDDTROT(I2,II)
                   ENDDO
                ENDDO

                HEFF(IPT)=TMP
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ! -------------------------------
    ! CONSTRUCT M EFFECTIVE (the MBH mass matrix)
    ! -------------------------------
    !   Inverse square root of this matrix is stored in two parts
    !                 - MSUBS
    !                 - MBLK
    !   because MEFF is block-diagonal in the blk-blk part
    !   and diagonal in the subs part.
    !
    ! *** PART 1 : MSUBS ***
    !   MSUBS (dim: NSUBS) contains inverse square root of each atom
    !                      in the subsystem
    DO I=1,NSUBS
       I4=BLKREORDER(BLKCUMSIZES(NBLK+1)+I)
       MSUBS(I)=1/SQRT(AMASS(I4))
    ENDDO

    ! *** PART 2 : MBLK ***
    !   MBLK (dim: 21*NBLK) : do for each block
    !              1 construct 6x6 mass matrix of block in UPT form
    !              2 construct inverse square root of this 6x6 mass matrix
    !              3 add the 21 elements to MBLK
    !
    MBLK(:)=0.0

    ! do for each block:
    IPT=0
    DO I=1,NBLK

       ! 1. construct 6x6 matrix of block in UPT form
       TMP_21(:)=0.0

       ! fill in upper left part
       BLKMASS=0.0
       DO J=1,BLKSIZES(I)
          I4=BLKREORDER(BLKCUMSIZES(I)+J)
          BLKMASS=BLKMASS+AMASS(I4)
       ENDDO
       TMP_21(1)=BLKMASS
       TMP_21(7)=BLKMASS
       TMP_21(12)=BLKMASS

       ! fill in upper right part
       ! JPT is pointer in tmp_21
       DO I2=1,3
          DO J2=4,6
             JPT=6*I2+J2-6+(I2-I2**2)/2

             TMP=0.0
             DO J=1,BLKSIZES(I)
                I4=BLKREORDER(BLKCUMSIZES(I)+J)
                AMS=AMASS(I4)
                I5=3*(I4-1)+I2
                TMP=TMP+AMS*ALLDDTROT(I5,J2)
                ! TODO !!!!!!! could be    TMP=TMP+AMS*ALLDDTROT(I5,J2-3)
                ! !!!!!!!!!!! to be changed: 3 first columns of allddtrot
                ! !!!!!!!!!!! would be more economical implementation...
             ENDDO

             TMP_21(JPT)=TMP
          ENDDO
       ENDDO

       ! fill in lower right part
       ! JPT is pointer in tmp_21
       DO I2=4,6
          DO J2=I2,6
             JPT=6*I2+J2-6+(I2-I2**2)/2

             TMP=0.0
             DO J=1,BLKSIZES(I)
                I4=BLKREORDER(BLKCUMSIZES(I)+J)
                AMS=AMASS(I4)
                DO K=1,3
                   I5=3*(I4-1)+K
                   TMP=TMP+AMS*ALLDDTROT(I5,I2)*ALLDDTROT(I5,J2)
                ENDDO
             ENDDO

             TMP_21(JPT)=TMP
          ENDDO
       ENDDO

       ! 2. construct inverse square root of this 6x6 mass matrix
       !        diagonalize tmp_21
       IH1=1
       NATP=6+1
       IH2=IH1+NATP
       IH3=IH2+NATP
       IH4=IH3+NATP
       IH5=IH4+NATP
       IH6=IH5+NATP
       IH7=IH6+NATP
       IH8=IH7+NATP
       CALL DIAGQ(6,6,TMP_21,VTMP_21,DDS(IH2),DDS(IH3), &
            DDS(IH4),DDS(IH5),DDEV,DDS(IH6),DDS(IH7),DDS(IH8),0)

       ! invert the eigenvalues and take root
       DO K=1,MBHDIM
          IF(ABS(DDEV(K)).LT.VCUT) DDEV(K)=SIGN(VCUT,DDEV(K))
          DDEV(K)=ONE/SQRT(DDEV(K))
       ENDDO

       ! construct inverse root
       IPT=21*(I-1)
       DO II=1,6
          DO JJ=II,6
             IPT=IPT+1
             DO K=1,6
                MBLK(IPT)=MBLK(IPT)+DDEV(K)*VTMP_21(II,K)*VTMP_21(JJ,K)
             ENDDO
          ENDDO
       ENDDO

    ENDDO

    ! -------------------------------
    ! CALCULATE MASS WEIGHTED H EFFECTIVE
    ! -------------------------------
    !   MEFF^-.5 . HEFF . MEFF^-.5 is stored in HEFF
    !   Is calculated in 3 parts.
    !
    ! 1. blk-blk part
    DO I=1,NBLK
       DO J=I,NBLK
          ! for each blk-blk' combination:
          ! replace heff_blk-blk' by mass weighted heff_blk-blk'
          !
          ! => put heff . meff(-.5) in tmp_36
          !    this matrix is NOT symmetric!
          TMP_36(:,:)=0.0

          DO II=1,6
             DO JJ=1,6
                TMP=0.0
                DO K=1,6

                   ! IPT index in heff (mbhdim)
                   I2=6*(I-1)+II
                   J2=6*(J-1)+K
                   IF (I2.LE.J2) THEN
                      IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2
                   ELSE
                      IPT=MBHDIM*J2+I2-MBHDIM+(J2-J2**2)/2
                   ENDIF
                   ! JPT index in mblk (21*nblk)
                   I2=MIN(K,JJ)
                   J2=MAX(K,JJ)
                   JPT=6*I2+J2-6+(I2-I2**2)/2
                   JPT=JPT+21*(J-1)

                   TMP=TMP+HEFF(IPT)*MBLK(JPT)

                ENDDO

                TMP_36(II,JJ)=TMP
             ENDDO
          ENDDO

          ! => put meff(-.5) . temp_36 in heff
          !    this matrix is symmetric
          DO II=1,6
             IF (I.EQ.J) THEN
                JST=II
             ELSE
                JST=1
             ENDIF
             DO JJ=JST,6
                TMP=0.0

                DO K=1,6
                   ! JPT index in mblk (21*nblk)
                   I2=MIN(II,K)
                   J2=MAX(II,K)
                   JPT=6*I2+J2-6+(I2-I2**2)/2
                   JPT=JPT+21*(I-1)
                   TMP=TMP+MBLK(JPT)*TMP_36(K,JJ)
                ENDDO

                ! IPT index in heff (mbhdim)
                I2=6*(I-1)+II
                J2=6*(J-1)+JJ
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2

                HEFF(IPT)=TMP
             ENDDO
          ENDDO

       ENDDO
    ENDDO

    ! 2. subs-subs part
    !    for each subs-subs' combination:
    !    replace heff_subs-subs' by mass weighted heff_subs-subs'
    DO I=1,NSUBS
       DO J=I,NSUBS

          DO I3=1,3
             IF (I.EQ.J) THEN
                JST=I3
             ELSE
                JST=1
             ENDIF
             DO J3=JST,3
                I2=6*NBLK+3*(I-1)+I3
                J2=6*NBLK+3*(J-1)+J3
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2
                HEFF(IPT)=HEFF(IPT)*MSUBS(I)*MSUBS(J)
             ENDDO
          ENDDO

       ENDDO
    ENDDO
    !
    ! 3. blk-subs part
    !
    DO I=1,NSUBS
       AMS=MSUBS(I)
       DO I3=1,3
          DO J=1,NBLK

             ! for each blk-subs combination:
             ! replace heff_blk-subs by mass weighted heff_blk-subs
             !
             ! => construct tmp_6
             TMP_6(:)=0.0
             DO JJ=1,6
                TMP=0.0
                DO K=1,6

                   ! IPT index in heff (mbhdim)
                   I2=6*(J-1)+K
                   J2=6*NBLK+3*(I-1)+I3
                   IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2

                   ! JPT index in mblk (21*nblk)
                   I2=MIN(JJ,K)
                   J2=MAX(JJ,K)
                   JPT=6*I2+J2-6+(I2-I2**2)/2
                   JPT=JPT+21*(J-1)

                   TMP=TMP+AMS*HEFF(IPT)*MBLK(JPT)
                ENDDO
                TMP_6(JJ)=TMP
             ENDDO

             ! => fill in in heff
             DO JJ=1,6
                I2=6*(J-1)+JJ
                J2=6*NBLK+3*(I-1)+I3
                IPT=MBHDIM*I2+J2-MBHDIM+(I2-I2**2)/2
                HEFF(IPT)=TMP_6(JJ)
             ENDDO

          ENDDO
       ENDDO
    ENDDO

    ! -------------------------------
    ! DIAGONALIZE MASS WEIGHTED H EFFECTIVE
    ! -------------------------------
    ! NMA with mass weighted Heff
    IH1=1
    NATP=MBHDIM+1
    IH2=IH1+NATP
    IH3=IH2+NATP
    IH4=IH3+NATP
    IH5=IH4+NATP
    IH6=IH5+NATP
    IH7=IH6+NATP
    IH8=IH7+NATP
    CALL DIAGQ(MBHDIM,MBHDIM,HEFF,VHEFF,DDS(IH2),DDS(IH3), &
         DDS(IH4),DDS(IH5),DDEV,DDS(IH6),DDS(IH7),DDS(IH8),0)

    ! -----------------------------------
    ! eigenvectors in Cartesian coordinates: DDV

    ! Note: 2D arrays are stored by column
    DDV(:,1:MBHDIM)=0.0

    ! fill rows of free atoms
    DO I=1,NSUBS
       I4=BLKREORDER(BLKCUMSIZES(NBLK+1)+I)
       DO I3=1,3
          IPT=3*(I4-1)+I3
          I5=6*NBLK+3*(I-1)+I3
          DDV(IPT,1:MBHDIM)=VHEFF(I5,1:MBHDIM)
       ENDDO
    ENDDO

    ! fill rows of atoms in blocks

    DO K=1,NBLK
       DO I=(BLKCUMSIZES(K)+1),(BLKCUMSIZES(K+1))
          I4=BLKREORDER(I)
          DO I3=1,3
             IPT=3*(I4-1)+I3
             DO J=1,MBHDIM

                TMP=0.0
                DO II=1,6
                   DO JJ=1,6
                      I2=MIN(II,JJ)
                      J2=MAX(II,JJ)
                      JPT=6*I2+J2-6+(I2-I2**2)/2
                      JPT=JPT+21*(K-1)
                      KPT=6*(K-1)+II

                      TMP=TMP+ALLDDTROT(IPT,JJ) &
                           *MBLK(JPT)*VHEFF(KPT,J)

                   ENDDO
                ENDDO
                DDV(IPT,J)=TMP*SQRT(AMASS(I4))
                ! DDV(IPT,J)=TMP
             ENDDO
          ENDDO
       ENDDO
    ENDDO

    ! normalize columns ddv
    DO J=1,MBHDIM
       CALL NORMALL(DDV(1,J),NAT3)
    ENDDO

    ! -----------------------------------
    ! Convert eigenvalues to frequencies
    DO I=1,MBHDIM
       DDF(I)=CNVFRQ*SQRT(ABS(DDEV(I)))
       IF(DDEV(I).LT.0.0) DDF(I)=-DDF(I)
    ENDDO

    RETURN
  END SUBROUTINE MULTMBH

end module mbh_m

