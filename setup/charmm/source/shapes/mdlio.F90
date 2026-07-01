#if KEY_SHAPES==1 /*shapes_main*/
SUBROUTINE RDMDL(IUNIT,COMLYN,COMLEN)
  !
  !   This routine reads an MDL card file and
  !   generates a new segment for the PSF
  !
  !   Note:: This routine modifies the PSF and COORD common blocks
  !
  ! SYNTAX:
  !   READ MDL [UNIT int] segid [ match-name ] [POSItion]
  !                             [ NEXT       ]
  !                             [ ALL        ]
  !                             
  !
  !      By Bernard R. Brooks   December 6, 1995
  !
  use cnst_fcm, only: cnst_is_allocated, allocate_cnst
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use stream
  use string
  use psf
  use param
  use coord
  use ctitla
  use mmffm
  use ffieldm
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  INTEGER IUNIT
  !
  !
  INTEGER LEN
  INTEGER, PARAMETER :: MAXLEN=100
  CHARACTER(len=MAXLEN) LINE,MATCH
  !
  INTEGER I,J,NAT,IAT,NBON,IBON,JAT,BNDORD,IEL,ILAST
  INTEGER IS,IQ
  LOGICAL QNEXT,QALL,QMATCH,QPOSIT
  CHARACTER(len=8) NEWID,TYPEQ,TYPEZ
  INTEGER, PARAMETER :: NUMEL=100
  INTEGER NTYPE(NUMEL)
  CHARACTER(len=2) ATNAMX

  SAVE
  !
  CHARACTER(len=2) :: ELEMNT(NUMEL)=(/'H ','He','Li','Be','B ','C ','N ','O ','F ','Ne', &
       'Na','Mg','Al','Si','P ','S ','Cl','Ar','K ','Ca', &
       'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn', &
       'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y ','Zr', &
       'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', &
       'Sb','Te','I ','Xe','Cs','Ba','La','Ce','Pr','Nd', &
       'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', &
       'Lu','Hf','Ta','W ','Re','Os','Ir','Pt','Au','Hg', &
       'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', &
       'Pa','U ','Np','Pu','Am','Cm','Bk','Cf','Es','Fm'/)
  !
  !     Atomic masses of the most common isotopes for each atom.
  !
  integer, parameter :: crl = chm_real
  real(chm_real) :: AMS(NUMEL) = (/ &
       1.00790_crl,   4.00260_crl,   6.94000_crl,   9.01218_crl,   10.81000_crl, &
       12.01100_crl,  14.00670_crl,  15.99940_crl,  18.99840_crl,  20.17900_crl, &
       22.98977_crl,  24.30500_crl,  26.98154_crl,  28.08550_crl,  30.97376_crl, &
       32.06000_crl,  35.45300_crl,  39.94800_crl,  39.09830_crl,  40.08000_crl, &
       44.95590_crl,  47.90000_crl,  50.94150_crl,  51.99600_crl,  54.93800_crl, &
       55.84700_crl,  58.93320_crl,  58.71000_crl,  63.54600_crl,  65.38000_crl, &
       69.73500_crl,  72.59000_crl,  74.92160_crl,  78.96000_crl,  79.90400_crl, &
       83.80000_crl,  85.46780_crl,  87.62000_crl,  88.90590_crl,  91.22000_crl, &
       92.90640_crl,  95.94000_crl,  98.90620_crl,  101.07000_crl, 102.90550_crl, &
       106.40000_crl, 107.86800_crl, 112.41000_crl, 114.82000_crl, 118.69000_crl, &
       121.75000_crl, 127.60000_crl, 126.90450_crl, 131.30000_crl, 132.90540_crl, &
       137.33000_crl, &
       (ZERO,i=1,15), 178.49000_crl, 180.94790_crl, 183.85000_crl, 186.20700_crl, &
       190.20000_crl, 192.22000_crl, 195.09000_crl, 196.96650_crl, 200.59000_crl, &
       204.37000_crl, 207.20000_crl, 208.98040_crl, &
       (ZERO,i=1,17) /)
  !
  IF(IOLEV.GT.0) THEN
     !
     QPOSIT=(INDXA(COMLYN,COMLEN,'POSI').GT.0)
     NEWID=NEXTA8(COMLYN,COMLEN)
     CALL TRIMA(COMLYN,COMLEN)
     IF(COMLEN.LE.0) THEN
        CALL WRNDIE(0,'<RDMDL>','BAD SYNTAX')
        RETURN
     ENDIF
     IF(COMLEN.GT.MAXLEN)THEN
        CALL WRNDIE(0,'<RDMDL>','MATCH NAME TOO LONG, TRUNCATED')
     ENDIF
     MATCH=COMLYN(1:COMLEN)
     comlen = 0
     QNEXT=MATCH(1:4).EQ.'NEXT'
     QALL =MATCH(1:4).EQ.'ALL '
     QMATCH=.NOT.(QNEXT .OR. QALL)
     QNEXT=.TRUE.
     LINE=' '
     !
     IF(QPOSIT) THEN
        DO WHILE(LINE(1:4).NE.'$$$$')
           READ(IUNIT,22,END=940) LINE
        ENDDO
     ENDIF
     IF(QMATCH) THEN
        DO WHILE(LINE.NE.MATCH)
           READ(IUNIT,22,END=940) LINE
        ENDDO
     ELSE
        READ(IUNIT,22) LINE
     ENDIF
22   FORMAT(A)

     ILAST=NRES
100  CONTINUE
     !  now process this section
     !
     LEN=MAXLEN
     CALL TRIME(LINE,LEN)

     titleb(1) = line
     READ(IUNIT,22,END=920) titleb(2)
     READ(IUNIT,22,END=920) titleb(3)
     ntitlb = 3
     READ(IUNIT,45,END=920,ERR=920) NAT,NBON
45   FORMAT(2I3)

     IF(PRNLEV.GT.3) WRITE(OUTU,105) LINE(1:LEN)
105  FORMAT('RDMDL:  Adding a new section: "',A,'"')
     !
     IF(NATOM+NAT.GT.MAXA .OR. &
          NSEG+1.GT.MAXSEG .OR. &
          NRES+1.GT.MAXRES .OR. &
          NGRP+1.GT.MAXGRP .OR. &
          NBOND+NBON.GT.MAXB) THEN
        CALL WRNDIE(-2,'<RDMDL>','Size limit exceeded')
        RETURN
     ENDIF
     !
     DO I=1,NUMEL
        NTYPE(I)=0
     ENDDO
     !
     DO I=1,NAT
        IAT=I+NATOM
        READ(IUNIT,55) X(IAT),Y(IAT),Z(IAT),ATNAMX
55      FORMAT(3F10.4,1X,A2)
        !
        IEL=-1
        DO J=1,NUMEL
           IF(ELEMNT(J).EQ.ATNAMX) IEL=J
        ENDDO
        IF(IEL.EQ.-1) IEL=1
        NTYPE(IEL)=NTYPE(IEL)+1
        !  
        ! construct the new name based on atom number counts
        CALL CNVTUC(ATNAMX, 2)
        TYPEZ=ATNAMX
        WRITE(TYPEQ,'(I4)') NTYPE(IEL)
        IS=3
        IF(ATNAMX(2:2).EQ.' ') IS=2
        IQ=1
        IF(TYPEQ(1:1).EQ.' ') IQ=2
        IF(TYPEQ(2:2).EQ.' ') IQ=3
        IF(TYPEQ(3:3).EQ.' ') IQ=4
        IF(IS.GT.IQ) IQ=IS
        TYPEZ(IS:4)=TYPEQ(IQ:4)
        ATYPE(IAT)=TYPEZ
        !
        AMASS(IAT)=AMS(IEL)
#if KEY_MMFF==1
        if(ffield == mmff) AtNum(IAT)=IEL 
#endif
        CG(IAT)=ZERO   ! mmff will reassign
        IAC(IAT)=1     ! mmff will reassign
        IBLO(IAT)=0
        IMOVE(IAT)=0
        RSCLF(IAT)=1.0
#if KEY_WCA==1
        WCA(I)=1.0     
#endif
        WMAIN(IAT)=ZERO
     ENDDO
     !
     NGRP=NGRP+1
     IMOVEG(NGRP)=0
     IGPBS(NGRP)=NATOM
     IGPTYP(NGRP)=0
     !
     DO I=1,NBON
        READ(IUNIT,58) IAT,JAT,BNDORD
58      FORMAT(3I3)
        NBOND=NBOND+1
        IF(JAT.GT.IAT) THEN
           IB(NBOND)=IAT + natom
           JB(NBOND)=JAT + natom
        ELSE
           IB(NBOND)=JAT + natom
           JB(NBOND)=IAT + natom
        ENDIF
#if KEY_MMFF==1
        if(ffield == mmff) BondType(NBOND)=BNDORD 
#endif
     ENDDO
     !
     IF(QNEXT) THEN
        NSEG=NSEG+1
        NICTOT(NSEG)=NRES
        SEGID(NSEG)=NEWID
        QNEXT=.FALSE.
     ENDIF
     !
     NRES=NRES+1
     IBASE(NRES)=NATOM

     i = indx (line, maxlen, ')', 1)
     j = indx (line, maxlen, '(', 1)

     if (i .gt. 4) then
        if (j .lt. i .and. j .gt. 0) then
           if (j .lt. i - 5) j = i - 5
           res(nres) = line (j + 1: i - 1)
        else
           RES(NRES)=LINE(i-4:i-1)
        end if
     else
        res(nres) = line
     end if

     CALL ENCODI(NRES-ILAST,RESID(NRES),8, i)
     !
     NATOM=NATOM+NAT
     !
     IBASE(NRES+1)=NATOM
     IGPBS(NGRP+1)=NATOM
     NICTOT(NSEG+1)=NRES
     !
     DO WHILE(LINE(1:4).NE.'$$$$')
        READ(IUNIT,22,END=920) LINE
     ENDDO
     IF(QALL) THEN
        READ(IUNIT,22,END=920) LINE
        GOTO 100
     ENDIF

     if (cnst_is_allocated) call allocate_cnst(natom)
  ENDIF
  !
  !===========================================================
  ! At end of file (when jump), no more sections to read.
920 CONTINUE
  IF(QNEXT) THEN
     CALL WRNDIE(0,'<RDMDL>','No matching set found')
  ENDIF
  !CC      ELSE
  !CC##IF PARALLEL
  !CC         CALL PSND4(NATOM,1)
  !CC         CALL PSNDC(ATYPE(IAT),1)
  !CC         CALL PSNDC(RESTYP,1)
  !CC         CALL PSNDC(LINE,1)
  !CC##ENDIF
  CALL PSFSUM(OUTU)
  RETURN
  !===========================================================
  ! Cant find a match
940 CONTINUE
  CALL WRNDIE(0,'<RDMDL>','No matching name found')
  RETURN
END SUBROUTINE RDMDL

SUBROUTINE WRMDL(IUNIT,COMLYN,COMLEN)
  !
  !   This routine reads an MDL card file and
  !   generates a new segment for the PSF
  !
  !   Note:: This routine modifies the PSF and COORD common blocks
  !
  ! SYNTAX:
  !   WRIT MDL [UNIT int] atom-selection [TITLe title]
  !
  !      By Bernard R. Brooks   December 6, 1995
  !
  use chm_kinds
  use dimens_fcm
  use number
  use memory
  use select
  use stream
  use string
  use psf
  use coord
  use ctitla
  use mmffm
  use ffieldm
  use linkatom, only: findel
  !
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER COMLEN
  INTEGER IUNIT

  !
  !
  INTEGER LEN
  INTEGER, PARAMETER :: MAXLEN=100
  CHARACTER(len=MAXLEN) LINE
  !
  INTEGER I,J,NAT,IAT,NBON,IBON,JAT,BNDORD,IEL,ILAST
  integer nsel
  CHARACTER(len=8) NEWID
  CHARACTER(len=8) ATYPEI,AIEL
  !
  INTEGER, PARAMETER :: NUMEL=100
  CHARACTER(len=2) ATNAMX
  real(chm_real) ZNUM
  integer,allocatable,dimension(:):: islct
  !
  CHARACTER(len=2) :: ELEMNT(NUMEL)=(/'H ','He','Li','Be','B ','C ','N ','O ','F ','Ne', &
       'Na','Mg','Al','Si','P ','S ','Cl','Ar','K ','Ca', &
       'Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn', &
       'Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y ','Zr', &
       'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn', &
       'Sb','Te','I ','Xe','Cs','Ba','La','Ce','Pr','Nd', &
       'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb', &
       'Lu','Hf','Ta','W ','Re','Os','Ir','Pt','Au','Hg', &
       'Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th', &
       'Pa','U ','Np','Pu','Am','Cm','Bk','Cf','Es','Fm'/)
  !
  IF(IOLEV.GT.0) THEN
     call chmalloc('mdlio.src','wrmdl','ISLCT',natom,intg=ISLCT) 
     call SELCTA(COMLYN,COMLEN,islct,X,Y,Z,WMAIN,.true.)

     CALL TRIMA(COMLYN,COMLEN)

     newid = nexta8 (comlyn, comlen)
     if (newid(1:4) .eq. 'TITL') then
        if (comlen .gt. 0) then
           titleb(1) = comlyn(1:comlen)
        else
           titleb(1) = ' '
        end if
        titleb(2) = ' '
        titleb(3) = ' '
        ntitlb = 3
        comlen = 0
     end if

     LINE=' '
     !
     ILAST=NRES

     !  now process this section
     !
     nbon = 0
     do i = 1, nbond
        if (islct(ib(i)) + islct(jb(i)) .eq. 2) nbon = nbon + 1
     end do

     nsel = 0
     do i = 1, natom
        if (islct(i) .eq. 1) then
           nsel = nsel + 1
           islct(i) = nsel
        end if
     end do

     if (ntitlb .gt. 3) ntitlb = 3
     if (ntitlb .lt. 3) then
        if (ntitlb .lt. 0) ntitlb = 0
        titleb(ntitlb + 1) = ' '
        titleb(ntitlb + 2) = ' '
        titleb(ntitlb + 3) = ' '
        ntitlb = 3
     end if

     do i = 1, ntitlb
        j = 80
        call trime(titleb(i), j)
        if (j .eq. 0) j = 1
        write (iunit, '(A)') titleb(i)(1:j)
     end do

     write(IUNIT,45) nsel,NBON
45   FORMAT(2I3)

     do i = 1, natom
        if (islct (i) .gt. 0) then
           iat = islct(i)
#if KEY_MMFF==1
           if(ffield == mmff) IEL=AtNum(I)
#else /**/
           ATYPEI=ATYPE(I)
           CALL FINDEL(ATYPEI,AMASS(I),I,AIEL,ZNUM,.FALSE.)
           IEL=NINT(ZNUM)
#endif 
           atnamx=ELEMNT(IEL)
           call cnvtlc (atnamx(2:2), 1)
           write(IUNIT,55) X(I),Y(I),Z(I),ATNAMX
        end if
     end do
55   FORMAT(3F10.4,1X,A2)

     DO I=1,NBOND
        if (islct(ib(i)) .gt. 0 .and. islct(jb(i)) .gt. 0) then
           iat = islct(ib(i))
           jat = islct(jb(i))
#if KEY_MMFF==1
           if(ffield == mmff) bndord = BondType(i) 
#else /**/
           bndord = 1
#endif 
           WRITE(IUNIT,58) IAT,JAT,BNDORD
        end if
     end do
58   FORMAT(3I3)

     write (iunit, '(A)') '$$$$'
     call chmdealloc('mdlio.src','wrmdl','ISLCT',natom,intg=ISLCT) 
  ENDIF
  !===========================================================
  RETURN
END SUBROUTINE WRMDL
#endif /*  (shapes_main)*/
SUBROUTINE NULL_MDL
  RETURN
END SUBROUTINE NULL_MDL

