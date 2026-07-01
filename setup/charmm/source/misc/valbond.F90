module VALBOND
  use chm_kinds
  implicit none
      
 !  ********************************************************************
 !  ********************************************************************
 !  *********         VALBOND GLOBAL VARIABLES            **************
 !  ********************************************************************
 !  ********************************************************************

      INTEGER,ALLOCATABLE,DIMENSION(:),SAVE :: &
       VBZ,      & ! ATOMIC NUMBER
       VBLP,     & ! NUMBER OF LONE PAIRS PER ATOM
       VBDE,     & ! NUMBER OF FORMAL D ELECTRONS
       VBVAL,    & ! VALENCE (NUMBER OF NEIGHBORS)
       VBRAWP,   & ! RAW P HYBRIDIZATION OF ATOM (0-3)
       VBRAWD      ! RAW D HYBRIDIZATION (0-5)

      LOGICAL,ALLOCATABLE,DIMENSION(:),SAVE :: &
       VBHYP,  &  ! TRUE FOR HYPERVALENT ATOMS
       VBMET,  &  ! TRUE FOR METAL ATOMS
       VBSKIP     ! TRUE IF ATOM HAS CHARMM ANGLES

      ! VARIABLES INDEXED BY ATOM AND BOND (I,J) 
      ! (FORMERLY DIMENSIONED MAXA,VBMAXV)
      INTEGER,ALLOCATABLE,DIMENSION(:,:),SAVE :: &
        VBNEI,  & ! NEIGHBOR LIST
        VBB      ! BOND LIST

      REAL(chm_real),ALLOCATABLE,DIMENSION(:,:),SAVE :: &
        VBP,    & ! VALBOND P HYBRIDIZATION FOR BOND J
        VBD,    & ! SAME FOR D
        VBSMAX    ! MAX. STRENGTH FUNCTION
    
      ! ARRAY SIZE PARAMETERS
      INTEGER VBMAXV            ! MAXIMUM VALENCE
      PARAMETER (VBMAXV = 8)
      INTEGER VBMAXZ            ! MAXIMUM ATOMIC NUMBER
      PARAMETER (VBMAXZ = 100)

      ! VALBOND PARAMETERS
      REAL(chm_real) VBWT(VBMAXZ,VBMAXZ)   ! WEIGHT FOR A BOND BETWEEN ELEMENTS I AND J
      REAL(chm_real) VBK(VBMAXZ,VBMAXZ)    ! SCALING FACTOR FOR BOND BETWEEN ELEMS I,J
      REAL(chm_real) VBKHV(VBMAXZ)         ! SCALING FOR HYPERVALENT BOND AT ELEM I
      REAL(chm_real) VBWTLP(VBMAXZ)     ! WEIGHT FOR A LONE PAIR ON ELEMENT I
      REAL(chm_real) VBBLI(VBMAXZ)      ! BOND-LENGTHENING INFLUENCE
      REAL(chm_real) VBBLS(VBMAXZ)      ! BOND-LENGTHENING SENSITIVITY

      REAL(chm_real) VBTR(VBMAXZ,VBMAXZ)  ! TRANS-INFLUENCE OFFSET

      ! OTHER BASIC ELEMENTAL PROPERTIES
      INTEGER VBVALZ(VBMAXZ)     ! NORMAL VALENCE OF ELEMENT I
      REAL(chm_real)  VBEN(VBMAXZ)       ! ELECTRONEGATIVITY OF ELEMENT I

      ! FLAGS
      LOGICAL VBINID          ! TRUE IF VBINIT HAS BEEN CALLED
      LOGICAL VBDOND          ! TRUE IF VBDONE HAS BEEN CALLED
      LOGICAL VBPRUT          ! TRUE IF VERBOSE OUTPUT ALREADY 'PRUNT'
      LOGICAL VBRESD          ! TRUE IF VBRESE HAS BEEN CALLED

 !       COMMON /VALBON/ VBEN,
 !      .  VBWT,VBK,VBKHV,VBWTLP,VBBLI,VBBLS,VBTR,VBINID,VBDOND,
 !      .  VBVALZ,VBPRUT,VBRESD

      ! UFF2 FORCE FIELD
      INTEGER U2MAXZ            ! MAXIMUM NUMBER OF ATOM TYPES
      PARAMETER (U2MAXZ = 50)
      REAL(chm_real) U2R(U2MAXZ)        ! NONBOND RADIUS
      REAL(chm_real) U2E(U2MAXZ)        ! NONBOND ENERGY
      REAL(chm_real) U2S(U2MAXZ)        ! NONBOND SCALING

 !      COMMON /UFF2/ U2R,U2E,U2S

 ! ====================================================================
 !  Leave global array definition and enter module subroutine section
 ! ====================================================================

      contains
     
#if KEY_VALBOND==1 /*valbond*/

 !  ********************************************************************
 !  ********************************************************************
 !  *********         VALBOND COMMAND PARSING             **************
 !  ********************************************************************
 !  ********************************************************************

      SUBROUTINE VBCOMM
 !      PROCESS THE COMMAND LINE FOR THE VALBOND MODULE.
 !      CALLED BY MISCOM WHEN COMMAND LINE STARTS WITH 'VALB'
  use dimens_fcm
  use comand
  use exfunc
  use string
      INTEGER NLP,ISHYP,ZI,ZJ
      REAL(chm_real) NP,ND,VAL
 ! 
      CHARACTER*4 wrd,ANAME,BNAME
      CHARACTER*40 MSG

      WRD = NEXTA4(COMLYN,COMLEN)

      IF(.NOT.VBINID) CALL VBINIT()

      IF (WRD .EQ. 'INIT') THEN
         VBINID = .FALSE.
         CALL VBINIT()
      ELSE IF (WRD .EQ. 'RESE') THEN
         CALL VBRESE()
      ELSE IF (WRD.EQ.'LP  ') THEN
         ANAME = NEXTA4(COMLYN,COMLEN)
         NLP = NEXTI(COMLYN,COMLEN)
         CALL VBSLP(ANAME,NLP)
      ELSE IF (WRD.EQ.'E ') THEN
         ANAME = NEXTA4(COMLYN,COMLEN)
         NLP = NEXTI(COMLYN,COMLEN)
         CALL VBSE(ANAME,NLP)
      ELSE IF (WRD.EQ.'HYBR'.OR.WRD.EQ.'HYBB') THEN
         ANAME = NEXTA4(COMLYN,COMLEN)
         BNAME = NEXTA4(COMLYN,COMLEN)
         NP = NEXTF(COMLYN,COMLEN)
         ND = NEXTF(COMLYN,COMLEN)
         ISHYP = NEXTI(COMLYN,COMLEN)
         CALL VBSHYB(ANAME,BNAME,NP,ND,ISHYP)
      ELSE IF (WRD.EQ.'HYBA') THEN
         ANAME = NEXTA4(COMLYN,COMLEN)
         NP = NEXTF(COMLYN,COMLEN)
         ND = NEXTF(COMLYN,COMLEN)
         ISHYP = NEXTI(COMLYN,COMLEN)
         CALL VBSHYA(ANAME,NP,ND,ISHYP)
      ELSE IF (WRD.EQ.'PARA') THEN
         ANAME = NEXTA4(COMLYN,COMLEN)
         ZI = NEXTI(COMLYN,COMLEN)
         ZJ = NEXTI(COMLYN,COMLEN)
         VAL = NEXTF(COMLYN,COMLEN)
         CALL VBPARA(ANAME,ZI,ZJ,VAL)
      ELSE IF (WRD.EQ.'PRIN') THEN
         VBPRUT = .FALSE.
      ELSE IF (WRD.EQ.'DONE') THEN
         CALL VBDONE()
      ELSE IF (WRD.EQ.'INCL') THEN
         CALL VBINCLUDE(COMLYN,COMLEN)
      ELSE IF (WRD.EQ.'SKIP') THEN
         CALL VBEXCLUDE(COMLYN,COMLEN)
      ELSE
        WRITE(MSG,*) 'INVALID VALB SUBCOMMAND: ', WRD
         CALL WRNDIE(-1,'<VALB>', MSG)
      ENDIF

      RETURN
      END SUBROUTINE VBCOMM

      subroutine vbexclude(comlyn,comlen)

      use psf, only : natom
      use dimens_fcm
      use coord
      use stream
      use select

      implicit none

      integer::i,comlen
      integer,dimension(:),allocatable::itemp
      character(len=*)::comlyn

      allocate(itemp(natom))

      call selcta(comlyn,comlen,itemp,x,y,z,wmain,.true.)

      do i=1,natom
        if(itemp(i)==1) vbskip(i)=.true.
      enddo

      deallocate(itemp)

      end subroutine vbexclude

      subroutine vbinclude(comlyn,comlen)

      use psf, only : natom
      use dimens_fcm
      use coord
      use stream
      use select

      implicit none

      integer::i,comlen
      integer,dimension(:),allocatable::itemp
      character(len=*)::comlyn

      allocate(itemp(natom))

      call selcta(comlyn,comlen,itemp,x,y,z,wmain,.true.)

      do i=1,natom
        if(itemp(i)==1) vbskip(i)=.false.
      enddo

      deallocate(itemp)

      end subroutine vbinclude

      SUBROUTINE VBPARA(ANAME,ZI,ZJ,VAL)
      ! CHANGE THE VALUES OF VALBOND PARAMETERS
  use dimens_fcm
  use stream
      CHARACTER*4 ANAME
      CHARACTER*40 MSG
      INTEGER ZI,ZJ
      REAL(chm_real) VAL
      IF(ANAME.EQ.'KHV ') THEN
        VBKHV(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'K   ') THEN
        VBK(ZI,ZJ) = VAL
        WRITE(OUTU, 910) ANAME,ZI,ZJ,VAL
      ELSE IF(ANAME.EQ.'WT  ') THEN
        VBWT(ZI,ZJ) = VAL
        WRITE(OUTU, 910) ANAME,ZI,ZJ,VAL
      ELSE IF(ANAME.EQ.'LP  ') THEN
        VBWTLP(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'U2R ') THEN
        U2R(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'U2S ') THEN
        U2S(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'U2E ') THEN
        U2E(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'VAL ') THEN
        VBVALZ(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'EN  ') THEN
        VBEN(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'TR  ') THEN
        VBTR(ZI,ZJ) = VAL
        VBTR(ZJ,ZI) = VAL
        WRITE(OUTU, 910) ANAME,ZI,ZJ,VAL
      ELSE IF(ANAME.EQ.'BLI ') THEN
        VBBLI(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE IF(ANAME.EQ.'BLS ') THEN
        VBBLS(ZI) = VAL
        WRITE(OUTU, 900) ANAME,ZI,VAL
      ELSE
        WRITE(MSG,*) 'PARAMETER TYPE NOT FOUND: ', ANAME
        CALL WRNDIE(-1,'<VALB PARA>', MSG)
      ENDIF
      RETURN
 900  FORMAT (' SETTING VALBOND PARAMETER ',A4,'(',I3,') = ',G15.5)
 910  FORMAT (' SETTING VALBOND PARAMETER ',A4,'(',I3,',',I3, &
              ') = ',G15.5)
      END SUBROUTINE VBPARA


      SUBROUTINE VBSE(ANAME,NE)
 !      SET THE NUMBER OF D-ELECTRONS ON AN ATOM
 !        ANAME: PDB NAME OF THE ATOM
 !        NE:    NUMBER OF D ELECTRONS
        use dimens_fcm
        use psf
        use stream
      INTEGER NE, I
      CHARACTER*4 ANAME
      CHARACTER*30 MSG
      DO I = 1, NATOM
         IF (ATYPE(I)(1:4).EQ.ANAME) THEN
            if (prnlev >= 2) write(outu, '(''VB: D ELEC. COUNT(''A4'') = ''I4)') ANAME,NE
            VBDE(I) = NE
            GOTO 10 ! BREAK
         ENDIF
      ENDDO
      WRITE(MSG,*) 'ATOM NOT FOUND: ', ANAME
      CALL WRNDIE(-1,'<VALB E>', MSG)
   10 CONTINUE
      RETURN 
      END SUBROUTINE VBSE


      SUBROUTINE VBSLP(ANAME,NLP)
      ! SET THE NUMBER OF LONE PAIRS ON AN ATOM
        use dimens_fcm
        use psf
        use stream
      INTEGER NLP, I
      CHARACTER*4 ANAME
      CHARACTER*30 MSG
      DO I = 1, NATOM
         IF (ATYPE(I)(1:4).EQ.ANAME) THEN
            if (prnlev >= 2) write(outu, '(''VB: LP(''A4'') = ''I4)') ANAME,NLP
            VBLP(I) = NLP
            GOTO 10 ! BREAK
         ENDIF
      ENDDO
      WRITE(MSG,*) 'ATOM NOT FOUND: ', ANAME
      CALL WRNDIE(-1,'<VALB LP>', MSG)
   10 CONTINUE
      RETURN 
      END SUBROUTINE VBSLP

      SUBROUTINE VBSHYB(ANAME,BNAME,NP,ND,ISHYP)
      ! SET THE HYBRIDIZATION OF A BOND
  use dimens_fcm
  use psf
  use stream
      CHARACTER*4 ANAME,BNAME
      INTEGER ISHYP
      REAL(chm_real) NP, ND
      INTEGER I,J
      CHARACTER*30 MSG
      LOGICAL FOUND
      FOUND = .FALSE.
      !CALL VBINIT()
      DO I = 1, NATOM
        IF (ATYPE(I)(1:4).EQ.ANAME) THEN
          IF(ISHYP.NE.0) THEN
            VBHYP(I) = .TRUE. 
            if (prnlev >= 2) WRITE (OUTU,900) I, ANAME
  900       FORMAT('VALBOND: SETTING ATOM ',I4,' (',A4, &
                   ') AS HYPERVALENT')
          ENDIF
          DO J = 1, VBVAL(I)
            IF (BNAME.EQ.'*   '.OR.ATYPE(VBNEI(I,J))(1:4).EQ.BNAME) THEN
              VBP(I,J) = NP
              VBD(I,J) = ND
              if (prnlev >= 2) write(outu,910) ANAME,ATYPE(VBNEI(I,J)),NP,ND
  910         FORMAT('VALBOND: SET BOND ',A4,'-',A4, &
                     ' AS SP',F7.3,'D',F7.3)
              FOUND = .TRUE.
            ENDIF
          ENDDO
        ENDIF
      ENDDO
      IF(.NOT.FOUND) THEN
        WRITE(MSG,*) 'BOND NOT FOUND: ', ANAME,'-',BNAME
        CALL WRNDIE(-1,'<VALB SHYB>', MSG)
      ENDIF
      RETURN
      END SUBROUTINE VBSHYB
      
      SUBROUTINE VBSHYA(ANAME,NP,ND,ISHYP)
      ! SET THE HYBRIDIZATION OF AN ATOM
  use dimens_fcm
  use psf
  use stream
      CHARACTER*4 ANAME
      INTEGER ISHYP
      REAL(chm_real) NP, ND
      CALL VBSHYB(ANAME,'*   ',NP,ND,ISHYP)
      RETURN
      END SUBROUTINE VBSHYA



 !  ********************************************************************
 !  ********************************************************************
 !  *********         VALBOND INITIALIZATION              **************
 !  ********************************************************************
 !  ********************************************************************

      SUBROUTINE VBRESE()
        ! RESET ALL VALBOND VARIABLES TO THEIR DEFAULT VALUES
        ! XXX: IT MIGHT BE A GOOD IDEA TO MOVE THEM TO A DATA FILE...
        use dimens_fcm
        use param
        use psf
        use code
        use stream
        INTEGER I, J
        if (prnlev >= 2) WRITE(outu,*) 'INITIALIZING VALBOND'
        VBINID = .FALSE.
        VBDOND = .FALSE.
        VBPRUT = .FALSE.
        VBRESD = .TRUE.


        DO J = 1, VBMAXZ
           DO I = 1, VBMAXZ
              VBK(I,J) = 220.0 ! DEFAULT K VALUE ACCORDING TO 1996 REF
              VBWT(I,J) = 1.0 ! DEFAULT WT VALUE (UNOFFICIAL)
              VBTR(I,J) = 0.0 ! DEFAULT TRANS INFLUENCE
           ENDDO

           VBKHV(J) = 220.0 ! DEFAULT HYPERVALENT K VALUE
           VBWTLP(J) = 1.0  ! DEFAULT LONE PAIR WEIGHT (UNOFFICIAL)
           VBBLI(J) = 0.    ! DEFAULT TRANS BOND LENGTHENING EFFECT
           VBBLS(J) = 1.0   ! DEFAULT BOND LENGTHENING SENSITIVITY
           
           ! METAL PARAMETERS FROM 1998 PAPER
           VBK(21,J) = 40.0
           VBK(22,J) = 40.0
           VBK(23,J) = 50.0
           VBK(24,J) = 35.0
           VBK(25,J) = 60.0
           VBK(26,J) = 70.0
           VBK(27,J) = 85.0
           VBK(28,J) = 100.0
           VBK(29,J) = 100.0
           VBK(30,J) = 100.0

           VBK(39,J) = 40.0
           VBK(40,J) = 45.0
           VBK(41,J) = 55.0
           VBK(42,J) = 65.0
           VBK(43,J) = 70.0
           VBK(44,J) = 75.0
           VBK(45,J) = 150.0
           VBK(46,J) = 100.0
           VBK(47,J) = 100.0
           VBK(48,J) = 100.0

           VBK(57,J) = 40.0
           VBK(72,J) = 44.0
           VBK(73,J) = 44.0
           VBK(74,J) = 44.0
           VBK(75,J) = 44.0
           VBK(76,J) = 90.0
           VBK(77,J) = 100.0
           VBK(78,J) = 150.0
           VBK(79,J) = 100.0
           VBK(80,J) = 100.0
        ENDDO

        ! WEIGHTS FROM 1993 PAPER
        VBWT(6,1) = 1.05
        VBWT(6,6) = 1.0
        VBWT(6,7) = 1.0
        VBWT(6,8) = 0.875
        VBWT(6,15) = 1.03
        VBWT(7,1) = 1.04
        VBWT(7,6) = 0.851
        VBWT(7,7) = 1.0
        VBWT(7,8) = 1.10
        VBWT(8,1) = 1.016
        VBWT(8,6) = 0.922
        VBWT(8,7) = 1.151
        VBWT(8,8) = 1.0
        VBWT(15,1) = 1.152
        VBWT(15,6) = 0.96
        VBWT(15,7) = 0.397
        VBWT(15,8) = 0.425

        ! K'S FROM 1993 PAPER
        VBK(6,1) = 163
        VBK(6,6) = 212
        VBK(6,7) = 220
        VBK(6,8) = 220
        VBK(6,15) = 220
        VBK(7,1) = 145
        VBK(7,6) = 220
        VBK(7,7) = 190
        VBK(7,8) = 300
        VBK(8,1) = 163
        VBK(8,6) = 220
        VBK(8,7) = 220
        VBK(8,8) = 220
        VBK(15,1) = 122
        VBK(15,6) = 220
        VBK(15,7) = 220
        VBK(15,8) = 220
        VBK(74,1) = 15.0

        ! LONE PAIR WEIGHTS FROM 1993 PAPER
        VBWTLP(6) = 1.658
        VBWTLP(7) = 0.9
        VBWTLP(8) = 0.902
        VBWTLP(15) = 0.23
        VBWTLP(16) = 0.69

        ! NORMAL VALENCE, USED TO COMPUTE NUMBER OF OMEGA BONDS
        ! IN HYPERVALENT COMPOUNDS
        VBVALZ(54) = 0
        VBVALZ(14) = 4
        VBVALZ(15) = 3
        VBVALZ(16) = 2
        VBVALZ(17) = 1
        VBVALZ(18) = 0
        VBVALZ(35) = 1
        VBVALZ(53) = 1
        VBVALZ(77) = 0 ! XXX HACK!!!

        ! ELECTRONEGATIVITIES, USED FOR "OFFSETS" IN HV-VB
        VBEN(1) = 2.20
        VBEN(6) = 2.55
        VBEN(7) = 3.04
        VBEN(8) = 3.44
        VBEN(9) = 3.98
        VBEN(14) = 1.90
        VBEN(15) = 2.19
        VBEN(16) = 2.58
        VBEN(17) = 3.16
        VBEN(35) = 2.96
        VBEN(53) = 2.66
        VBEN(54) = 3.00
        !CALL VBTBL()

        ! TRANS INFLUENCE OFFSETS
        !VBTR(1,15)  = -6.115098546
        !VBTR(1,7)   = -10.20411333
        !VBTR(1,17)  = -18.78890841
        !VBTR(7,15)  = -1.138616191
        !VBTR(15,17) = -5.811380475
        !VBTR(7,17)  = -2.677920416

        ! BOND LENGTHENING INFLUENCE
        !VBBLI(6) = 9.706979169
        !VBBLI(17) = 0.238002001
        !VBBLI(1) = 9.833049934
        !VBBLI(7) = 2.066686353
        !VBBLI(8) = 0.357412528
        !VBBLI(15) = 4.386005081

        !! BOND LENGTHENING SENSITIVITY
        !VBBLS(6) = 0.422193547
        !VBBLS(17) =1.19108921
        !VBBLS(1) = 0.812711355
        !VBBLS(7) = 1.051542802
        !VBBLS(8) = 1.230776736
        !VBBLS(15) =0.973864492


        ! SYMMETRIZE THE VBTR MATRIX
        DO I = 1, VBMAXZ
           DO J = I+1, VBMAXZ
              VBTR(J,I) = VBTR(I,J)
           ENDDO
        ENDDO

        ! UFF2 MORSE PARAMETERS
        DO I = 1, U2MAXZ
           U2R(I) = 0.
           U2S(I) = 0.
           U2E(I) = 0.
        ENDDO

        !U2R(1) = 3.248
        !U2E(1) = 0.034 * 23.06
        !U2S(1) = 1.432

        !U2R(6) = 3.906
        !U2E(6) = 0.094 * 23.06
        !U2S(6) = 1.367

      END SUBROUTINE VBRESE

      SUBROUTINE VBINIT()
      ! initialize structure-dependent valbond variables, such as
      ! neighbor lists, valencies, and atomic numbers
      ! also allocates memory
  use dimens_fcm
  use param
  use psf
  use code
  use coord
  use stream
      INTEGER I,J
      !INTEGER VBZF

      IF(VBINID) RETURN

      IF(.NOT.VBRESD) CALL VBRESE()

      IF(ALLOCATED(VBZ))THEN
        DEALLOCATE(VBZ,VBLP,VBDE,VBVAL,VBRAWP,VBRAWD,VBHYP,VBMET, &
         VBSKIP,VBNEI,VBB,VBP,VBD,VBSMAX)
      ENDIF

      ALLOCATE(VBZ(NATOM),VBLP(NATOM),VBDE(NATOM),VBVAL(NATOM), &
       VBRAWP(NATOM),VBRAWD(NATOM),VBHYP(NATOM),VBMET(NATOM), &
       VBSKIP(NATOM), &
       VBNEI(NATOM,VBMAXV),VBB(NATOM,VBMAXV), &
       VBP(NATOM,VBMAXV),VBD(NATOM,VBMAXV),VBSMAX(NATOM,VBMAXV))
      
      DO I = 1,NATOM
         VBRAWP(I) = 0
         VBRAWD(I) = 0
         VBVAL(I) = 0
         VBLP(I) = -1 ! -1 = UNDEFINED
         !VBZ(I) = 0
         VBLP(I) = 0
         VBDE(I) = -1
         VBHYP(I) = .FALSE.
         VBMET(I) = .FALSE.
         VBSKIP(I) = .FALSE.
         DO J = 1, VBMAXV
            VBP(I,J) = -1.
            VBD(I,J) = 0.
            VBSMAX(I,J) = 0.
            VBNEI(I,J) = 0
            VBB(I,J) = 0
         ENDDO
      ENDDO

      ! SKIP ATOMS THAT ARE AT THE CENTER OF AN ANGLE
      DO I = 1, NTHETA
        VBSKIP(JT(I)) = .TRUE.
      ENDDO

      if (prnlev >= 2) WRITE(OUTU,*) 'VALBOND: INITIALIZING CONNECTION TABLE'

      ! COUNT NUMBER OF BONDS PER ATOM AND BUILD NEIGHBOR LIST
      DO I = 1, NBOND
         VBVAL(IB(I)) = VBVAL(IB(I))+1
         VBVAL(JB(I)) = VBVAL(JB(I))+1
         VBNEI(IB(I),VBVAL(IB(I))) = JB(I)
         VBNEI(JB(I),VBVAL(JB(I))) = IB(I)
         VBB(IB(I),VBVAL(IB(I))) = I
         VBB(JB(I),VBVAL(JB(I))) = I
      ENDDO

      ! FIGURE OUT THE ATOMIC NUMBERS AND TRANSITION METAL CHARACTER
      DO I = 1, NATOM
         VBZ(I) = VBZF(I)

         ! SKIP HYDROGEN ATOMS
         IF (VBZ(I).EQ.1) VBSKIP(I)=.TRUE.

         IF (VBZ(I).GE.21.AND.VBZ(I).LE.30) VBMET(I)=.TRUE.
         IF (VBZ(I).GE.39.AND.VBZ(I).LE.48) VBMET(I)=.TRUE.
         IF (VBZ(I).GE.57.AND.VBZ(I).LE.80) VBMET(I)=.TRUE.
         IF (VBZ(I).GE.89.AND.VBZ(I).LE.112) VBMET(I)=.TRUE.
         IF(.NOT.VBPRUT) THEN
           IF(VBSKIP(I)) THEN
             if (prnlev >= 2) WRITE(OUTU,900) I,ATYPE(I)
 900         FORMAT('VALBOND: ATOM ',I4, '(',A4,')', &
              ' WILL BE SKIPPED BECAUSE IT HAS CHARMM ANGLES')
              ! ' WILL BE SKIPPED ')
          ENDIF
        ENDIF
      ENDDO

      VBINID = .TRUE.

      RETURN
      END SUBROUTINE VBINIT

      SUBROUTINE VBDONE()
      ! CALCULATE VALBOND HYBRIDIZATIONS
  use dimens_fcm
  use param
  use psf
  use code
  use coord
  use stream
      INTEGER I,J
      REAL(chm_real) DENP(MAXA),PP
      !INTEGER VBZF
      CHARACTER*60 MSG

      !CALL VBINIT()
      ! COMPUTE AND PRINT HYBRIDIZATIONS
      if (prnlev >= 2) WRITE(OUTU,'(/ ''VALBOND: HYBRIDIZATIONS'')')
      DO 10 I = 1, NATOM
         IF (VBSKIP(I)) GOTO 10 ! NEXT
         IF (VBVAL(I).LE.1) GOTO 10 ! NEXT
         ! RAW HYBRIDIZATION
         IF (.NOT.VBMET(I)) THEN
           IF(VBLP(I).EQ.-1) CALL VBLPAU(I)
           VBRAWP(I) = VBVAL(I) + VBLP(I) - 1
           if (prnlev >= 2) WRITE(OUTU, 900) ATYPE(I), VBRAWP(I)
 900       FORMAT(' ATOM ',A4,' RAW HYBRIDIZATION: S P',I1)

           ! VALBOND HYBRIDIZATION (EQ. 18,19)
           DENP(I) = VBLP(I) * VBWTLP(VBZ(I))
           DO J = 1, VBVAL(I)
              DENP(I) = DENP(I) + VBWT(VBZ(I),VBZ(VBNEI(I,J)))
           ENDDO
         ENDIF
         IF(VBMET(I)) THEN
           IF(VBHYP(I)) THEN
             IF (VBDE(I).LT.0) THEN
               WRITE(MSG,905) ATYPE(I)
 905           FORMAT('D-ELECTRONS FOR HYPERVALENT METAL ' &
                      ,A4, ' NOT SET EXPLICITLY')
               CALL WRNDIE(0,'<VALB VBDONE>', MSG)
             ENDIF
           ENDIF
         ENDIF
         DO J = 1, VBVAL(I)
            IF (VBP(I,J).LT.0.0) THEN
               IF(VBMET(I)) THEN
                 VBP(I,J) = 0.0
                 IF (VBDE(I) + 2*VBVAL(I) > 12) THEN
                    if (prnlev >= 2) WRITE (OUTU,906) I, ATYPE(I)
 906                FORMAT('VALBOND: SETTING ATOM ',I4,' (',A4, &
                    ') AS HYPERVALENT')
                   VBHYP(I) = .TRUE.
                   VBD(I,J) = (12 - VBDE(I))/2 - 1
                 ELSE
                   VBHYP(I) = .FALSE.
                   VBD(I,J) = VBVAL(I) - 1
                 ENDIF
               ELSE
                 PP = VBRAWP(I) * VBWT(VBZ(I),VBZ(VBNEI(I,J))) / DENP(I)
                 VBP(I,J) = PP/(1.-PP)
               ENDIF
            ENDIF
            if (prnlev >= 2) WRITE(OUTU, 910) ATYPE(I),ATYPE(VBNEI(I,J)), &
                 VBP(I,J), VBD(I,J)
 910        FORMAT('     BOND ',A4,'-',A4,' HYBRIDIZATION: S P',F5.2, &
                ' D',F5.2)
            ! SMAX, EQ. 13
            VBSMAX(I,J) = SQRT(1/(1+VBP(I,J)+VBD(I,J))) &
               *(1+SQRT(3*VBP(I,J))+SQRT(5*VBD(I,J)))
         ENDDO
         VBDOND = .TRUE.
   10 CONTINUE
      if (prnlev >= 2) WRITE(OUTU, *)

      END SUBROUTINE VBDONE


      

 !  ********************************************************************
 !  ********************************************************************
 !  *********         VALBOND ENERGY CALCULATION          **************
 !  ********************************************************************
 !  ********************************************************************

      SUBROUTINE EANGVB(ET,X,Y,Z,DX,DY,DZ,NATOMX)
 !      Calculates VALBOND bond angles, angle energies and forces
 !      loops through all angles
 ! 
  use number
  use stream
  use consta
  use dimens_fcm
 ! 
      REAL(chm_real) ET
      REAL(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
 ! 
      INTEGER I,J,K,NATOMX,II,KK
      REAL(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,DF,E
 ! 
 !       WRITE(OUTU,*) "#IN : ",I,J,K,NATOMX
 !       WRITE(OUTU,*) X(I),Y(I),Z(I)
 !       WRITE(OUTU,*) X(J),Y(J),Z(J)
 !       WRITE(OUTU,*) X(K),Y(K),Z(K)
      
      ET=ZERO
      IF(.NOT.VBINID) RETURN
 ! 
      ! LOOP OVER ALL ATOMS, AND THEN OVER ALL PAIRS OF NEIGHBORS 
      ! (I.E., ALL ANGLES)
      ! THE ANGLE IS DEFINED AS I-J-K; J IS THE CENTRAL ATOM
      IF(.NOT.VBPRUT) THEN
        if (prnlev >= 2) WRITE(outu,*) 'VALBOND: COMPUTING VALBOND ENERGY (EANGVB)'
      ENDIF
      DO J = 1, NATOMX
        IF (VBSKIP(J)) GOTO 10 ! NEXT
        IF (VBHYP(J)) THEN ! HYPERVALENT CENTER?
          CALL VBHV(J,X,Y,Z,DX,DY,DZ,NATOMX,E)
          ET = ET+E
        ENDIF

        ! EVEN IF HYPERVALENT, CALL THE 'NORMAL' EANGV1 TO INCLUDE
        ! THE 1,3 NONBONDED INTERACTIONS IF NECESSARY
        DO II = 1, VBVAL(J)
          I = VBNEI(J,II)
          DO KK = II+1, VBVAL(J)
            K = VBNEI(J,KK)
            CALL EANGV1(E,I,J,K,II,KK,X,Y,Z,DX,DY,DZ,NATOMX)
            ET = ET + E
          ENDDO
        ENDDO

  10  CONTINUE
      ENDDO
      if (prnlev >= 2) WRITE(outu,*) 'VALBOND E = ', ET
      RETURN
      END SUBROUTINE EANGVB

      SUBROUTINE VBE(M,N,K,SMAX,A,E,DE)
      ! VALBOND ENERGY TERM FOR NORMAL BONDS
 !      N: VALBOND P hybridization
 !      M: VALBOND D hybridization
 !      K: VALBOND scaling factor
 !      SMAX: Smax (maximum strength)
 !      A: angle
 !      E: output energy
 !      DE: output derivative dE/dA
        use consta
        use dimens_fcm
        use stream
      REAL(chm_real) N,M,K,SMAX,A,E,DE
      REAL(chm_real) D,DD,DP,ST

      ! D = OVERLAP (DELTA IN VALBOND EQUATIONS)
      D = (1+M*COS(A)+N/2*(3*COS(A)**2-1))/(1+M+N)
      ! DD = dD/dA
      DD = -(M*SIN(A)+N*3*COS(A)*SIN(A))/(1+M+N)
      DP = SQRT(1-D**2)
      ! ST = S/SMAX
      ST = SQRT(1-(1-DP)/2)
      E  = K*SMAX*(1-ST)
      IF (DP.NE.0) THEN
        DE = K*SMAX*D*DD/(4*ST*DP)
      ELSE
        DE = 0 ! LIMITING CASE FOR 180-DEG ANGLES, TO AVOID DIV BY 0
      ENDIF
      !IF(.NOT.VBPRUT) WRITE (6,901) M,N,K,SMAX,A
      !IF(.NOT.VBPRUT) WRITE (6,902) D,DD,DP,ST
      !IF(.NOT.VBPRUT) WRITE (6,900) A*RADDEG,E,DE
      IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE (outu,910) A*RADDEG,K,SMAX,E,DE

  900 FORMAT ('       VBE A,E,DE: ',3G12.3)
  901 FORMAT ('       VBE M,N,K,SMAX,A: ',5G12.3)
  902 FORMAT ('       VBE D,DD,DP,ST: ',5G12.3)
  910 FORMAT ('       VBE(',F6.2,';K=',F6.2,',SMAX=',F6.3,') = ', &
             G12.3,'; DE =',G12.3)
      RETURN
      END SUBROUTINE VBE


      SUBROUTINE VBHV(J,X,Y,Z,DX,DY,DZ,NATOMX,ET)
      ! Compute the valbond energy contribution from a single
      ! hypervalent center, including numerical derivatives for the
      ! center and its ligands
        use dimens_fcm
        use stream
      INTEGER J,NATOMX
      REAL(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
      REAL(chm_real) ET
      REAL(chm_real) DEL,E2,TMP
      INTEGER I,II

      DEL = 1.0E-9

      ET = 0.
      CALL VBCOMB(J,X,Y,Z,NATOMX,ET)
      ET = ET * 1.0
      DO II = 0, VBVAL(J)
        IF (II.EQ.0) THEN
          I = J
        ELSE
          I = VBNEI(J,II)
        ENDIF

        ! USE SIMPLE FINITE FORWARD DIFFERENCE
        TMP = X(I)
        X(I) = X(I) + DEL
        CALL VBCOMB(J,X,Y,Z,NATOMX,E2)
        DX(I) = DX(I) + (E2-ET)/DEL
        !WRITE(6,*) ' VBHV ET1 = ',ET
        IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(outu,*) ' VBHV DX I=',I, (E2-ET)/DEL
        X(I) = TMP

        TMP = Y(I)
        Y(I) = Y(I) + DEL
        CALL VBCOMB(J,X,Y,Z,NATOMX,E2)
        DY(I) = DY(I) + (E2-ET)/DEL
        IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(outu,*) ' VBHV DY I=',I, (E2-ET)/DEL
        Y(I) = TMP

        TMP = Z(I)
        Z(I) = Z(I) + DEL
        CALL VBCOMB(J,X,Y,Z,NATOMX,E2)
        DZ(I) = DZ(I) + (E2-ET)/DEL
        IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(outu,*) ' VBHV DZ I=',I, (E2-ET)/DEL
        Z(I) = TMP

      ENDDO
      RETURN 
      END SUBROUTINE VBHV


      SUBROUTINE VBCOMB(J,X,Y,Z,NATOMX,ET)
      ! generates all hypervalent valbond configurations (combinations) 
      ! for a hypervalent center
      ! J = HYPERVALENT ATOM
      ! X,Y,Z = COORDINATE ARRAYS
      ! NATOMX = NUMBER OF ATOMS
      ! ET (OUTPUT) = ENERGY DUE TO ANGLES CENTERED ON J
  use dimens_fcm
  use stream
  use psf
      INTEGER J,NATOMX
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
      INTEGER N,NP
      INTEGER NCOMB
      INTEGER P(VBMAXV)
      INTEGER POS(VBMAXV)
      LOGICAL USED(VBMAXV)
      INTEGER DEPTH, STATE
      INTEGER I,K
      REAL(chm_real) ET,CT,E,C,ETR,ETR1
      !INTEGER FACTOR


      N  = VBVAL(J)               ! NUMBER OF BONDS
      IF(VBMET(J)) THEN
        NP = (N*2 + VBDE(J) - 12)/2  ! NO. OF OMEGA-BONDS
      ELSE
        NP = (N - VBVALZ(VBZ(J)))/2 ! NO. OF OMEGA-BONDS
      ENDIF
      IF(.NOT.VBPRUT) THEN
        NCOMB = FACTOR(N)/FACTOR(NP)/FACTOR(N-2*NP)/(2**NP)
        if (prnlev >= 2) WRITE (OUTU,900) ATYPE(J),N,NP,NCOMB
 900    FORMAT(/,' VALBOND: HYPERVALENT ATOM ',A4,' HAS ',I2, &
              ' BONDS; ',I2,' HYPERVALENT (',I3,' CONFIGURATIONS)')
      ENDIF

      DO I = 1, VBMAXV
        USED(I) = .FALSE.
        P(I) = 0      ! ATOM PAIRS INVOLVED IN W-BONDS FOR THIS CONFIG
        POS(I) = 0    ! ITERATION STACK
      ENDDO

      DEPTH = 1
      NCOMB = 0           ! NUMBER OF CONFIGURATIONS SO FAR
      STATE = 0
      ET = 0.
      ETR = 0.
      CT = 0.

      ! This loop is a state machine with a stack that emulates a
      ! recursive algorithm for generating all combinations of NP pairs
      ! of numbers between 1 and N. State 0 is the default (iterate),
      ! state 1 is backtrack

  10  CONTINUE ! WHILE .TRUE.
        IF (STATE.EQ.1) THEN ! BACKTRACKING
          DEPTH = DEPTH-1
          IF(DEPTH.LE.0) GOTO 20      ! DONE
          USED(POS(DEPTH)) = .FALSE.
          STATE = 0
        ELSE IF (DEPTH.GT.NP*2) THEN  ! REACHED THE BOTTOM
          NCOMB = NCOMB+1
          IF (.NOT.VBPRUT) THEN
            if (prnlev >= 2) WRITE (OUTU,910) NCOMB, (P(K), K=1,NP*2)
 910        FORMAT('   CONFIG ',I2,' USES HYP. BONDS ', &
                  3(I2,'-',I2,', '))
          ENDIF
          CALL VBHCFG(J,P,NP,X,Y,Z,NATOMX,C,E,ETR1)
          ET  = ET + C*E
          CT  = CT + C
          ETR = ETR + C*ETR1
          STATE = 1
        ELSE                          ! DEFAULT; ITERATE/RECURSE
          POS(DEPTH) = POS(DEPTH)+1
          I = POS(DEPTH)
          IF (I.GT.N) THEN
            STATE = 1
            GOTO 10 ! NEXT
          ENDIF
          IF (USED(I)) GOTO 10 ! NEXT
          IF (MOD(DEPTH,2).EQ.0.AND.P(DEPTH-1).GT.I) GOTO 10 ! NEXT
          IF (MOD(DEPTH,2).EQ.1.AND.DEPTH.GT.2.AND.P(DEPTH-2).GT.I)  &
           GOTO 10 ! NEXT
          P(DEPTH) = I
          USED(I) = .TRUE.
          DEPTH = DEPTH+1
          POS(DEPTH) = 0
        ENDIF

      GOTO 10 ! NEXT
  20  CONTINUE

      ET = ET/CT
      ETR = ETR/CT

      IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,920) ET, ETR
 920  FORMAT(' TOTAL ENERGY FOR THIS HYPERVALENT ATOM: ', &
      F9.3,/,' TOTAL TRANS OFFSET: ',F9.3,/)

      VBPRUT = .TRUE.

      RETURN
      END SUBROUTINE VBCOMB

      SUBROUTINE VBHCFG(J,P,NP,X,Y,Z,NATOMX,C,ET,ETR)
      ! Calculate the energy for a single HV-VB configuration
      ! RETURNS C,ET
      ! P IS A FLAT LIST OF PAIRS OF ATOMS INVOLVED IN HYPERVALENT BONDS
      ! FOR THIS CONFIGURATION
      ! They are not actual atom numbers, but numbers from 1 to N where
      ! N is the valence of the central atom. For example, for a
      ! octahedral compound, P might contain (1,6,2,5,3,4), meaning that
      ! 1 and 6 form an omega bond, 2 and 5, and 3 and 4.
  use dimens_fcm
  use consta
  use psf
  use stream
  use param
  use code
      INTEGER NP
      INTEGER P(VBMAXV)
      INTEGER NATOMX
      INTEGER PLOOK(VBMAXV)
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
      INTEGER I,J,K,IIP,II,KK,ZI,ZJ,ZK
      REAL(chm_real) C,E,ET,THETA,D,DELTA,CTOT,EI,EK,EOFF,DELTA1,DELTA2,ETR
      REAL(chm_real) ETR1,ETR2
      REAL(chm_real) N,M,BOF
      !REAL(chm_real) ANGIJK
      INTEGER IC
      REAL(chm_real) RX,RY,RZ,S2,S,DB,DF,EB,R0

      ZJ = VBZ(J)

      ! CALCULATE CONTRIBUTION TO COEFFICIENT
      C = 1.
      ET = 0.
      ETR = 0.
      EB  = 0.

      ! INITIALIZE LOOKUP TABLE
      DO I = 1,VBMAXV
        PLOOK(I) = 0
      ENDDO

      DO IIP = 0, NP-1
        II = P(IIP*2+1)
        KK = P(IIP*2+2)
        PLOOK(II) = KK
        PLOOK(KK) = II
      ENDDO

      ! LOOP OVER ALL POSSIBLE ANGLES
      DO II = 1, VBVAL(J)
        I = VBNEI(J,II)
        ZI = VBZ(I)
        DO KK = II+1, VBVAL(J)
          K = VBNEI(J,KK)
          ZK = VBZ(K)
          BOF = 1.
          IF (PLOOK(II).NE.0) BOF = BOF*0.5
          IF (PLOOK(KK).NE.0) BOF = BOF*0.5
          THETA = ANGIJK(I,J,K,X,Y,Z,NATOMX)
          IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,900) ATYPE(I),ATYPE(J),ATYPE(K), &
           RADDEG*THETA, BOF
 900      FORMAT('     ANGLE ',A4,'-',A4,'-',A4,' = ',F7.2,' DEG', &
                '; BOF = ', F4.2)
          IF (PLOOK(II).EQ.KK) THEN ! USE HV FORMULA
            ! XXX: AMBIGUITY IN PAPER: SHOULD WE USE II OR KK??? 
            ! LET'S DO THE AVERAGE OF BOTH...
            M = VBP(J,II)
            N = VBD(J,II)
            DELTA1 = ((1+M*COS(THETA+PI)+N/2*(3*COS(THETA+PI)**2-1))/ &
                   (1+M+N))**2
            E = BOF*VBKHV(ZJ)*(1-DELTA1)
            IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,910) E
 910        FORMAT('       HV VBE = ',F9.3)
            IF(VBMET(J)) THEN
              ETR1 = VBTR(ZI,ZK)*(DELTA1)
              IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,915) ETR1,DELTA1
              ! ADD BOND ENERGY
              ! R0 = R0MIN * (1+VBBLI(ZI)/100*VBBLS(ZK)*DELTA1)
              ! R = ...
              ! EBND1 = K * (R - R0)**2
              !WRITE(OUTU,*) 'VBB',J,II,VBB(J,II)
              IC=ICB(VBB(J,II))
              !WRITE(OUTU,*) 'IC', ICB(VBB(J,II))
              RX=X(I)-X(J)
              RY=Y(I)-Y(J)
              RZ=Z(I)-Z(J)
              S2=RX*RX + RY*RY + RZ*RZ
              S=SQRT(S2)
              R0=CBB(IC)*(1+VBBLI(ZK)/100*VBBLS(ZI)*DELTA1)
              DB=S-R0
              DF=CBC(IC)*DB
              EB=DF*DB
              !WRITE(OUTU,916) EB
              ! XXX
            ENDIF
 915        FORMAT('       TRANS OFFSET = ',F9.3,', DELTA**2=',F9.3)
 916        FORMAT('       STRETCHED BOND ENERGY = ',F9.3)
            ET = ET + (E+ETR1)/2 + EB

            ! NOW DO KK
            M = VBP(J,KK)
            N = VBD(J,KK)
            DELTA2 = (1+M*COS(THETA+PI)+N/2*(3*COS(THETA+PI)**2-1))/ &
                   (1+M+N)
            E = BOF*VBKHV(ZJ)*(1-DELTA2**2)
            IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,910) E
            IF(VBMET(J)) THEN
              ETR2 = VBTR(ZI,ZK)*(DELTA2**2)
              IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,915) ETR2,DELTA2**2
              ! ADD BOND ENERGY
              IC=ICB(VBB(J,KK))
              RX=X(K)-X(J)
              RY=Y(K)-Y(J)
              RZ=Z(K)-Z(J)
              S2=RX*RX + RY*RY + RZ*RZ
              S=SQRT(S2)
              R0=CBB(IC)*(1+VBBLI(ZI)/100*VBBLS(ZK)*DELTA1)
              DB=S-R0
              DF=CBC(IC)*DB
              EB=DF*DB
              !WRITE(OUTU,916) EB
            ENDIF
            ET = ET + (E+ETR2)/2 + EB
            ETR = ETR + (ETR1+ETR2)/2

            ! SUBSTRACT "OFFSET" ENERGY (EQ 9,10)
            ! DON'T DO IT FOR METALS (YET)
            IF (.NOT.VBMET(J)) THEN
              EI = 30 * (VBEN(ZI) - VBEN(ZJ))
              IF (EI.LT.0) EI = EI * 2
              EK = 30 * (VBEN(ZK) - VBEN(ZJ))
              IF (EK.LT.0) EK = EK * 2
              EOFF = (EI+EK)/2
              IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(outu,*) '    EOFF= ',EOFF,VBEN(ZI), &
               VBEN(ZJ),VBEN(ZK)
              ET = ET-EOFF  ! SEEMS TO REPRODUCE PUBLISHED RESULTS...
            ELSE
              EOFF = 0.0
            ENDIF

            ! CONTRIBUTION TO THE WEIGHT C FOR THIS CONFIG
            IF (N.LE.1) THEN
              ! FORMULA FOR P-BLOCK HYPERVALENT COMPOUNDS:
              C = C * COS(THETA)**2
            ELSE
              ! FORMULA FOR TRANSITION METALS
              DELTA = (DELTA1+DELTA2)/2
              C = C * DELTA**2
            ENDIF
          ELSE ! USE NORMAL FORMULA
            CALL VBE(VBP(J,II),VBD(J,II),VBK(ZJ,ZI), &
             VBSMAX(J,II),THETA,E,D)
            ET=ET+BOF*E
            CALL VBE(VBP(J,KK),VBD(J,KK),VBK(ZJ,ZK), &
             VBSMAX(J,KK),THETA,E,D)
            ET=ET+BOF*E
          ENDIF
        ENDDO
        ! For metals, now add the bond energy terms for the normal 
        ! bonds (the ones that are not part of a hypervalent pair)
        IF (PLOOK(II).EQ.0.AND.VBMET(J)) THEN
          IC=ICB(VBB(J,II))
          RX=X(I)-X(J)
          RY=Y(I)-Y(J)
          RZ=Z(I)-Z(J)
          S2=RX*RX + RY*RY + RZ*RZ
          S=SQRT(S2)
          R0=CBB(IC)
          DB=S-R0
          DF=CBC(IC)*DB
          EB=DF*DB
          !WRITE(OUTU,916) EB
          ET=ET+EB
        ENDIF
      ENDDO
      IF(.NOT.VBPRUT .and. prnlev >= 2) WRITE(OUTU,920) ET,ETR,C
 920  FORMAT('   CONFIG END; E = ',G15.5,'; ETR = ',G15.5, &
      '; C = ',G12.3)

      RETURN
      END SUBROUTINE VBHCFG

      SUBROUTINE EANGV1(ET,I,J,K,II,KK,X,Y,Z,DX,DY,DZ,NATOMX)
 !      Calculates VALBOND bond angles, angle energies and forces
 !      for a single angle, defined by atoms I,J,K
 !      Based on subroutine EANGLE() from energy/eintern.src
 ! 
  use number
  use stream
  use consta
  use dimens_fcm
 ! 
      REAL(chm_real) ET
      INTEGER ATOMX
      REAL(chm_real) DX(NATOMX),DY(NATOMX),DZ(NATOMX)
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
 ! 
 ! 
      INTEGER I,J,K,NATOMX,II,KK
      REAL(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,RI2,RJ2,RI,RJ
      REAL(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,CST,AT,DA,DF,E,D
      REAL(chm_real) ST2R,STR,DTXI,DTXJ,DTYI,DTYJ,DTZI,DTZJ
      REAL(chm_real) DFX,DFY,DFZ,DGX,DGY,DGZ,SMALLV
      REAL(chm_real) RIK,B,C,DXK,DYK,DZK,EX,KE
 ! 
 !       WRITE(OUTU,*) "#IN : ",I,J,K,NATOMX
 !       WRITE(OUTU,*) X(I),Y(I),Z(I)
 !       WRITE(OUTU,*) X(J),Y(J),Z(J)
 !       WRITE(OUTU,*) X(K),Y(K),Z(K)
      
      ET=ZERO
      SMALLV=RPRECI
 ! 

       DXI=X(I)-X(J)
       DYI=Y(I)-Y(J)
       DZI=Z(I)-Z(J)
       DXJ=X(K)-X(J)
       DYJ=Y(K)-Y(J)
       DZJ=Z(K)-Z(J)
       RI2=DXI*DXI+DYI*DYI+DZI*DZI
       RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

       RI=SQRT(RI2)
       RJ=SQRT(RJ2)
       RIR=ONE/RI
       RJR=ONE/RJ
       DXIR=DXI*RIR
       DYIR=DYI*RIR
       DZIR=DZI*RIR
       DXJR=DXJ*RJR
       DYJR=DYJ*RJR
       DZJR=DZJ*RJR
       CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

       IF(ABS(CST).GE.COSMAX) THEN
         IF(ABS(CST).GT.ONE) CST=SIGN(ONE,CST)
         AT=ACOS(CST)
 !                  DA=AT-TEQR !!!
 !                  IF(ABS(DA).GT.0.1) THEN
 !                    WRITE(OUTU,10) I,J,K
 !           10       FORMAT(' WARNING FROM EANGLE. Angle is almost linear.',
 !               &          /' Derivatives may be affected for atoms:',3I5)
 !                    WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
 !                    WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
 !                    WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
 !                    WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
 !                    WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
 !                    WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
 !           101      FORMAT(5X,A,5F15.5)
 !                  ENDIF
       ENDIF

       AT=ACOS(CST)

 !                DA=AT-TEQR

 !               WRITE(OUTU,*) "#MID : ",I,J,K,AT

 !                DF=FORCE*DA !!!
 !                E=DF*DA
 !                ET=ET+E
 !                DF=DF+DF

       IF (.NOT.VBHYP(J)) THEN ! SKIP FOR HYPERVALENT CENTER...
         ! COMPUTE VALBOND ENERGY AND DERIVATIVE HERE
         !   CALL TWICE TO COUNT THE CONTRIBUTIONS FROM THE POINT OF VIEW
         !   OF THE TWO ORBITALS INVOLVED...
         IF(.NOT.VBPRUT) THEN
           WRITE(6,*) 'VALBOND TERMS FOR ATOM NUMBERS',I,J,K
           WRITE(6,*) 'VALBOND TERMS FOR ELEMENT NUMBERS ',VBZ(I),VBZ(J),VBZ(K)
         ENDIF
         CALL VBE(VBP(J,II),VBD(J,II),VBK(VBZ(J),VBZ(I)), &
                 VBSMAX(J,II),AT,E,D)
         ET=ET+E
         DF=D
         CALL VBE(VBP(J,KK),VBD(J,KK),VBK(VBZ(J),VBZ(K)), &
                 VBSMAX(J,KK),AT,E,D)
         DF=DF+D
         ET=ET+E

         IF(ABS(CST).GE.0.999) THEN
           ST2R=ONE/(ONE-CST*CST+SMALLV)
           STR=SQRT(ST2R)
 !                  IF(TEQR.LT.PT001) THEN !!!
 !                    DF=MINTWO*FORCE*(ONE+DA*DA*SIXTH) !!!
 !                  ELSE IF(PI-TEQR.LT.PT001) THEN !!!
 !                    DF=TWO*FORCE*(ONE+DA*DA*SIXTH) !!!
 !                  ELSE
             DF=-DF*STR
 !                  ENDIF
         ELSE
           ST2R=ONE/(ONE-CST*CST)
           STR=SQRT(ST2R)
           DF=-DF*STR
         ENDIF

         ! THIS BLOCK NEEDS VALUES FOR R.R, CST, AND D..R
         DTXI=RIR*(DXJR-CST*DXIR)
         DTXJ=RJR*(DXIR-CST*DXJR)
         DTYI=RIR*(DYJR-CST*DYIR)
         DTYJ=RJR*(DYIR-CST*DYJR)
         DTZI=RIR*(DZJR-CST*DZIR)
         DTZJ=RJR*(DZIR-CST*DZJR)

         DFX=DF*DTXI
         DGX=DF*DTXJ
         DX(I)=DX(I)+DFX
         DX(K)=DX(K)+DGX
         DX(J)=DX(J)-DFX-DGX

         DFY=DF*DTYI
         DGY=DF*DTYJ
         DY(I)=DY(I)+DFY
         DY(K)=DY(K)+DGY
         DY(J)=DY(J)-DFY-DGY

         DFZ=DF*DTZI
         DGZ=DF*DTZJ
         DZ(I)=DZ(I)+DFZ
         DZ(K)=DZ(K)+DGZ
         DZ(J)=DZ(J)-DFZ-DGZ
      ENDIF

      ! ADD 1,3 MORSE POTENTIAL FOR ANGLES CENTERED ON
      ! TRANSITION METAL ATOMS
      IF (VBMET(J)) THEN
        ! CALC RIK
        DXK = X(I)-X(K)
        DYK = Y(I)-Y(K)
        DZK = Z(I)-Z(K)
        RIK = SQRT(DXK*DXK + DYK*DYK + DZK*DZK)

        ! CALC ENERGY
        KE = SQRT(U2E(VBZ(I))*U2E(VBZ(K)))
        B = 0.5*(U2S(VBZ(I))+U2S(VBZ(K)))
        C = 0.5*(U2S(VBZ(I))*U2R(VBZ(I)) + U2S(VBZ(K))*U2R(VBZ(K)))
        EX = EXP(C-B*RIK)-1
        E  = KE*EX*EX
        ET = ET+E
        !WRITE(6,*)'    MORSE ENERGY: ', E

        ! CALC DERIVATIVE
        DF  = -2*B*KE*EX*EXP(C-B*RIK)
        !WRITE(6,*) 'XXX E,DF = ', E,DF
        DFX = DXK/RIK*DF
        DFY = DYK/RIK*DF
        DFZ = DZK/RIK*DF

        DX(I)=DX(I)+DFX
        DY(I)=DY(I)+DFY
        DZ(I)=DZ(I)+DFZ

        DX(K)=DX(K)-DFX
        DY(K)=DY(K)-DFY
        DZ(K)=DZ(K)-DFZ
      ENDIF
       
      RETURN
      END SUBROUTINE EANGV1


 !  ********************************************************************
 !  ********************************************************************
 !  *********         HELPER ROUTINES AND FUNCTIONS       **************
 !  ********************************************************************
 !  ********************************************************************

      INTEGER FUNCTION FACTOR(N)
      ! FACTORIAL FUNCTION
      INTEGER N,I
      FACTOR = 1
      DO I = 1,N
        FACTOR = FACTOR * I
      ENDDO
      RETURN
      END FUNCTION FACTOR

      INTEGER FUNCTION VBZF(NUM)
 !      RETURNS THE ATOMIC NUMBER (Z) OF ATOM NUM
  use dimens_fcm
  use psf
  use linkatom

      INTEGER NUM,Z
      CHARACTER(chm_real) ELE
      REAL(chm_real) ZR
      CALL FINDEL(ATYPE(NUM),AMASS(NUM),NUM,ELE,ZR,.TRUE.)
      !SUBROUTINE FINDEL(NAME,MASS,NUM,ELE,ZNUM,QINIGM)
      Z = ZR
      IF (Z .EQ. 0) THEN               !avoid array index 0
        WRITE(6,*) 'VALBOND: ATOMIC NUMBER 0 REASSIGNED TO 100'
        Z = 100
      ENDIF
      VBZF = Z
      RETURN
      END FUNCTION VBZF


      REAL(chm_real) FUNCTION ANGIJK(I,J,K,X,Y,Z,NATOMX)
 !      COMPUTE THE ANGLE FOR ATOMS I,J,K
 ! 
  use number
  use stream
  use consta
  use dimens_fcm
 ! 
      INTEGER NATOMX
      REAL(chm_real) X(NATOMX),Y(NATOMX),Z(NATOMX)
 ! 
 ! 
      INTEGER I,J,K
      REAL(chm_real) DXI,DYI,DZI,DXJ,DYJ,DZJ,RI2,RJ2,RI,RJ
      REAL(chm_real) RIR,RJR,DXIR,DYIR,DZIR,DXJR,DYJR,DZJR,CST,AT,DA,DF,E,D
      REAL(chm_real) DFX,DFY,DFZ,DGX,DGY,DGZ,SMALLV

      SMALLV=RPRECI
 ! 

      !WRITE(6,'(A7,9F8.3)')'ANGIJK ', X(I) 
      DXI=X(I)-X(J)
      DYI=Y(I)-Y(J)
      DZI=Z(I)-Z(J)
      DXJ=X(K)-X(J)
      DYJ=Y(K)-Y(J)
      DZJ=Z(K)-Z(J)
      RI2=DXI*DXI+DYI*DYI+DZI*DZI
      RJ2=DXJ*DXJ+DYJ*DYJ+DZJ*DZJ

      RI=SQRT(RI2)
      RJ=SQRT(RJ2)
      RIR=ONE/RI
      RJR=ONE/RJ
      DXIR=DXI*RIR
      DYIR=DYI*RIR
      DZIR=DZI*RIR
      DXJR=DXJ*RJR
      DYJR=DYJ*RJR
      DZJR=DZJ*RJR
      CST=DXIR*DXJR+DYIR*DYJR+DZIR*DZJR

      IF(ABS(CST).GE.COSMAX) THEN
        IF(ABS(CST).GT.ONE) CST=SIGN(ONE,CST)
        AT=ACOS(CST)
 !                  DA=AT-TEQR !!!
 !                  IF(ABS(DA).GT.0.1) THEN
 !                    WRITE(OUTU,10) I,J,K
 !           10       FORMAT(' WARNING FROM EANGLE. Angle is almost linear.',
 !               &          /' Derivatives may be affected for atoms:',3I5)
 !                    WRITE(OUTU,101) 'I ATOM:',X(I),Y(I),Z(I)
 !                    WRITE(OUTU,101) 'J ATOM:',X(J),Y(J),Z(J)
 !                    WRITE(OUTU,101) 'K ATOM:',X(K),Y(K),Z(K)
 !                    WRITE(OUTU,101) 'DXIR  :',DXIR,DYIR,DZIR
 !                    WRITE(OUTU,101) 'DXJR  :',DXJR,DYJR,DZJR
 !                    WRITE(OUTU,101) 'CST   :',CST,AT*RADDEG,DA*RADDEG
 !           101      FORMAT(5X,A,5F15.5)
 !                  ENDIF
      ENDIF

      AT=ACOS(CST)
      ANGIJK = AT

      RETURN
      END FUNCTION ANGIJK


      SUBROUTINE VBLPAU(I)
      ! GUESS THE NUMBER OF LONE PAIRS FOR ATOM I AUTOMATICALLY
      ! ONLY WORKS FOR SOME OF THE SIMPLE "ORGANIC" CASES
  use dimens_fcm
  use psf
  use rtf, only: atct
      INTEGER I
      CHARACTER*4 T
      VBLP(I) = 0
      T = ATCT(IAC(I))
 !       WRITE(6,*) 'AUTOMATIC LP ASSIGNMENT TO ATOM ',I,ATYPE(I),IAC(I),
 !      .   ATCT(IAC(I))
      !IF (HEAP(VBPZ+I-1).EQ.8.OR.HEAP(VBPZ+I-1).EQ.16) THEN
      IF (VBZ(I).EQ.8.OR.VBZ(I).EQ.16) THEN
         IF (VBVAL(I).EQ.3) THEN
            VBLP(I) = 1
         ELSE
            VBLP(I) = 2
         ENDIF
      ELSE IF (VBZ(I).EQ.7.OR.VBZ(I).EQ.15) THEN
         IF (VBVAL(I).EQ.4.OR.T.EQ.'NH1 '.OR.T.EQ.'NH2 ' &
          .OR.T.EQ.'NR1 ') THEN
            VBLP(I) = 0
         ELSE IF (VBVAL(I).EQ.3) THEN
            VBLP(I) = 1
         ELSE IF (VBVAL(I).EQ.2) THEN
            VBLP(I) = 1
         ENDIF
      ELSE IF (VBZ(I).EQ.17.OR.VBZ(I).EQ.35.OR.VBZ(I).EQ.53) THEN
         VBLP(I) = 4 - VBVAL(I)
      ENDIF
      RETURN
      END SUBROUTINE VBLPAU

#else /* (valbond)*/

      SUBROUTINE NULL_VB
      RETURN
      END SUBROUTINE NULL_VB

#endif /* (valbond)*/
      

 ! =====================================================================
 !  End of module VALBOND
 ! =====================================================================      
      end module VALBOND
      
   

