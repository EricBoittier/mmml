module misc_mm

#if KEY_MMFF==0

  implicit none

contains

  SUBROUTINE NULL_misc_MM
    RETURN
  END SUBROUTINE NULL_misc_MM

end module misc_mm

#else /* KEY_MMFF */  

  use chm_kinds,only:chm_real
  use number,only:zero

  integer, parameter :: MaxSymbolN=100
  character(len=2) :: ElementSymbolList(MaxSymbolN)=(/ &
       'H ','HE','LI','BE','B ','C ','N ','O ','F ','NE', &
       'NA','MG','AL','SI','P ','S ','CL','AR','K ','CA', &
       'SC','TI','V ','CR','MN','FE','CO','NI','CU','ZN', &
       'GA','GE','AS','SE','BR','KR','RB','SR','Y ','ZR', &
       'NB','MO','TC','RU','RH','PD','AG','CD','IN','SN', &
       'SB','TE','I ','XE','CS','BA','LA','CE','PR','ND', &
       'PM','SM','EU','GD','TB','DY','HO','ER','TM','YB', &
       'LU','HF','TA','W ','RE','OS','IR','PT','AU','HG', &
       'TL','PB','BI','PO','AT','RN','FR','RA','AC','TH', &
       'PA','U ','NP','PU','AM','CM','BK','CF','ES','FM'/)
  save ElementSymbolList
 
  integer, parameter :: MaxElementNameN=55
  character(len=8) :: ElementNameList(MaxElementNameN)=(/ &
       'HYDROGEN','HELIUM  ','LITHIUM ','BERYLIUM','BXXXXXXX', & !  5
       'CARBON  ','NITROGEN','OXYGEN  ','FLUORINE','NEON    ', & ! 10
       'SODIUM  ','MAGNESIU','ALUMINIU','SILICON ','PHOSPHOR', & ! 15
       'SULFUR  ','CHLORINE','ARGON   ','POTASIUM','CALCIUM ', & ! 20
       'SCXXXXX ','TIXXX   ','VXXXX   ','CRXXXXXX','MNXXXXX ', & ! 25
       'FERRUM  ','COBALT  ','NIXXX   ','COPPER  ','ZINC    ', & ! 30
       'GALIUM  ','GERMANIU','ASXXXX  ','SELENIUM','BROMINE ', & ! 35
       'XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX', & ! 40
       'XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX', & ! 45
       'XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX','XXXXXXXX', & ! 50
       'XXXXXXXX','XXXXXXXX','IODINE  ','XXXXXXXX','XXXXXXXX'/)! 55
  save ElementNameList
  
  integer, parameter :: crl = chm_real
  integer, parameter :: MaxMassN=55
  real(chm_real), save :: ElementMassList(MaxMassN) = (/ &
       1.00794_crl,   4.002602_crl, 6.941_crl,     9.012182_crl, 10.811_crl, &
       12.011_crl,    14.00674_crl, 15.9994_crl,   18.9984_crl,  20.1797_crl, &
       22.989768_crl, 24.3050_crl,  26.981539_crl, 28.0855_crl,  30.973762_crl, &
       32.066_crl,    35.4527_crl,  ZERO,          39.098_crl,   40.078_crl, &
       (ZERO,i=1,5), &
       55.84700_crl,  ZERO,         ZERO,          63.546_crl,   65.39_crl, &
       (ZERO,i=1,4),  78.918336_crl, &
       (ZERO,i=1,17), 126.90447_crl, ZERO, ZERO /)
  !
  ! atomic masses
  !
  !    & 1.007825D0, 4.002603D0, 6.015121D0, 9.012182D0,10.012937D0,
  !    &12.000000D0,14.003074D0,15.994914D0,18.998403D0,19.992436D0,
  !    &22.989768D0,23.985042D0,26.981539D0,27.976927D0,30.973762D0,
  !    &31.972071D0,34.968853D0, 0.00000000, 0.00000000,39.962591D0,
  !    & 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  !    & 0.00000000, 0.00000000, 0.00000000,62.929599D0,63.929145D0,
  !    & 0.00000000, 0.00000000, 0.00000000, 0.00000000,78.918336D0,
  !    & 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  !    & 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  !    & 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
  !    & 0.00000000, 0.00000000,126.904473,0.00000000, 0.0  /

end module misc_mm
    
!  ================================================================
!   MATCH ATOM NUMBER TO ATOM NAME
CHARACTER(len=8) FUNCTION AtName(I)
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer i
  !
  AtName=' '
  IF(I.GT.0.AND.I.LT.MAXAIM) AtName=AtNames(I)
  RETURN
END FUNCTION AtName

INTEGER FUNCTION AtomType(atom)
  !
  use chm_kinds
  use dimens_fcm
  use exfunc
  use mmffm
  use psf
  use string
  implicit none
  !
  integer atom
  !
  AtomType=MTYPE(atom)
  IF(AtomType.LE.0.OR.AtomType.GT.MAXATC) THEN
     WRITE(SCRTCH,'(A,I5,2A,1X,I5)') &
          'Invalid atom type ',AtomType,' for atom ',QNAME(atom),atom
     call wrndie(-3,'<AtomType>',SCRTCH(:50))
     AtomType=0
  ENDIF
  RETURN
END FUNCTION AtomType

INTEGER FUNCTION b_class(nb)
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer nb
  !
  b_class=MDBOND(nb)
  RETURN
END FUNCTION b_class

! INTEGER FUNCTION CONN12(I,J) : CONN12=1 IF ATOMS I AND J ARE DIRECTLY BONDED
! =====================================================================
INTEGER FUNCTION CONN12(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer i, j, m
  !
  CONN12=0
  DO M=1,5
     IF(ITAB(M,I).LT.0) RETURN
     IF(ITAB(M,I).EQ.J) THEN
        CONN12=1
        RETURN
     ENDIF
  ENDDO
  RETURN
END FUNCTION CONN12

! INTEGER FUNCTION CONN13(I,J) : CONN13=1 IF ATOMS I AND J ARE 1,3 NONBONDED
! ====================================================================
INTEGER FUNCTION CONN13(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer i, mi, j, jn, m, max, n, nn
  !
  CONN13=0
  MAX=5
  loop300: DO M=1,5
     mi=ITAB(M,I)
     IF(mi.LT.0) RETURN
     loop100: DO N=1,MAX
        NN=N
        JN=ITAB(N,J)
        IF(JN.LT.0) then
           MAX=NN-1
           cycle loop300
        endif
        IF(JN == mi) then
           CONN13=1
           RETURN
        endif
     enddo loop100
  enddo loop300
  !
  RETURN
END FUNCTION CONN13

! ====================================================================
! INTEGER FUNCTION CONN14(I,J) : CONN14=1 IF I AND J ARE 1,4-NONBONDED.
!   THIS FUNCTION SUBPROGRAM RETURNS CONN14=1 IF I AND J ARE 1,4-NONBONDED.
!   IT ASSUMES THAT I AND J ARE NEITHER 1,2-BONDED OR 1,3-NONBONDED - I.
!   THAT FUNCTIONS CONN12(I,J) AND CONN13(I,J) HAVE ALREADY BEEN CALLED AND
!   HAVE RETURNED ZERO.
! ====================================================================
INTEGER FUNCTION CONN14(I,J)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  ! Jay Banks 25 Oct 95: changed local variable name IA to IIA, to avoid 
  ! conflict with variable in common (mmff.f90).
  !
  !  28 Nov 95 Jay Banks: removed space from all "GO TO"s, as suggested in
  !  developer.doc.
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer i, iia, j, ja, l, li, m, mj
  !
  integer CONN12
  external CONN12
  !
  CONN14=0
  LI=LOCAT(I)
  loop200: DO L=1,LI
     IIA=ITAB(L,I)
     IF(IIA.LE.0) cycle loop200
     MJ=LOCAT(J)
     loop100: DO M=1,MJ
        JA=ITAB(M,J)
        IF(JA.LE.0) cycle loop100
        IF(CONN12(IIA,JA).EQ.1) THEN
           CONN14=1
           RETURN
        ENDIF
     enddo loop100
  enddo loop200
  !
  RETURN
END FUNCTION CONN14

!
!  Jay Banks 22 Nov 95: added ##INCLUDE impnon.f90 to all functions.
!  (Replaced in-line implicit none in ElementSymbol and AtomicNumber.)
!
CHARACTER(len=2) FUNCTION ElementSymbol(AtNum)
  !-----------------------------------------------------------------------
  use chm_kinds
  use misc_mm
  implicit none
  integer AtNum

  if(AtNum.le.MaxSymbolN .and. AtNum.gt.0) then
     ElementSymbol=ElementSymbolList(AtNum)
  else
     ElementSymbol=' '
  endif
  return
END FUNCTION ElementSymbol

CHARACTER(len=8) FUNCTION ElementName(AtNum)
  !-----------------------------------------------------------------------
  use chm_kinds
  use misc_mm
  implicit none
  integer AtNum

  if(AtNum.le.MaxElementNameN .and. AtNum.gt.0) then
     ElementName=ElementNameList(AtNum)
  else
     ElementName=' '
  endif
  return
END FUNCTION ElementName

FUNCTION ElementMass(AtNum) result(emass)
  !-----------------------------------------------------------------------
  use chm_kinds
  use misc_mm
  implicit none
  integer AtNum

  real(chm_real) emass
  !
  ! averaged atomic masses
  !
  
  if(AtNum.le.MaxMassN .and. AtNum.gt.0) then
     EMass=ElementMassList(AtNum)
  else
     EMass=0.
  endif
  return
END FUNCTION ElementMass

INTEGER FUNCTION AtomicNumber(SymbolName) 
  !-----------------------------------------------------------------------
  use chm_kinds
  use misc_mm
  implicit none
  character(len=2) SymbolName
  
  AtomicNumber=0
  do while (AtomicNumber.lt.MaxSymbolN)
     AtomicNumber=AtomicNumber+1
     if(SymbolName .eq. ElementSymbolList(AtomicNumber)) return
  enddo
  AtomicNumber=0
  !
  RETURN
END FUNCTION AtomicNumber
!--MFC--  MFC Jan2010 removed, it is an intrinsic in f95
!--MFC-- !
!--MFC-- ! Created 16 Oct 95 by Jay Banks, from machdep/fortran90.src in MSI version.
!--MFC-- !
!--MFC--       INTEGER FUNCTION LEN_TRIM(STRING)
!--MFC--   use chm_kinds
!--MFC--       implicit none
!--MFC--       CHARACTER*(*) STRING
!--MFC--       integer ITAB
!--MFC--       parameter(ITAB=9)
!--MFC-- !
!--MFC--       LEN_TRIM=LEN(STRING)
!--MFC--       do while (LEN_TRIM.gt.0 .and. (STRING(LEN_TRIM:LEN_TRIM).eq.' ' &
!--MFC--                         .or. ICHAR(STRING(LEN_TRIM:LEN_TRIM)).eq.ITAB))
!--MFC--         LEN_TRIM=LEN_TRIM-1
!--MFC--       enddo
!--MFC-- !
!--MFC--       RETURN
!--MFC--       END

!
! INTEGER FUNCTION LPATH : Finds length of path specification
! in full file name
!
INTEGER FUNCTION LPATH(FILE)
  !
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !   Adapted by R. Czerminski of Molecular Simulations, Inc. from
  !  code developed at Merck and Co., Inc. by T. Halgren and others.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  !
  use chm_kinds
  implicit none
  CHARACTER(len=*) FILE
  integer i, l
  !
  L=LEN(FILE)
  DO I=L,1,-1
     LPATH=I
     IF(FILE(LPATH:LPATH).EQ.'/') RETURN
     !         IF(FILE(LPATH:LPATH).EQ.']') RETURN  ! vax
     !         IF(FILE(LPATH:LPATH).EQ.':') RETURN  ! vax
  ENDDO
  LPATH=0
  !
  RETURN
END FUNCTION LPATH

INTEGER FUNCTION p_class(nph)
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer nph
  !
  p_class=MDOMEG(nph)
  !
  RETURN
END FUNCTION p_class

INTEGER FUNCTION t_class(nth)
  !
  use chm_kinds
  use dimens_fcm
  use psf
  use mmffm
  implicit none
  !
  integer nth
  !
  t_class=MDTHET(nth)
  !
  RETURN
END FUNCTION t_class

#endif /* KEY_MMFF */
