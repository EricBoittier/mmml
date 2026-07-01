module vmod
  use chm_kinds

  !
  !     Mode-vector constraint io
  !
  !     Variable  Purpose
  !
  !     UOUT      unit for writing the mode coordinate and related quantities
  !     NSVQ      number of steps for writing mode coordinates
  !     ISVQ      counter for writing the mode coordinates
  !     ISVQR     real counter for mode coordinates
  !
  INTEGER UOUT, ISVQ, NSVQ, UVMODP, UMDN
  real(chm_real)  ISVQR
end module vmod

module mmfp
  use chm_kinds
  use dimens_fcm
  implicit none
  !-----------------------------------------------------------------------
  ! Common block:
  ! QMFP              logical flag to setup the Miscelaneous 
  !                   Mean Field Potentials
  !
  ! geometrical terms:
  ! QGEO              logical flag to setup the Geometrical terms
  ! MAXGEO            maximum number of constraints on the heap
  ! JGEO              type of constraints
  ! LSTGEO            list of atoms under constraints
  ! NGEO              number of atoms under constraints
  ! IGEO              type of geometrical constraints
  ! XRGEO             reference position
  ! YRGEO             reference position
  ! ZRGEO             reference position
  ! TRGEO             reference angle
  ! XDGEO             unit vector 
  ! YDGEO             unit vector
  ! ZDGEO             unit vector
  ! DRGEO             offset distance
  ! DTGEO             D_Theta offset from Theta_REF for angle/dihedral restraints. 
  ! FCGEO             force constant
  ! P1GEO             parameter 1 
  ! P2GEO             parameter 2 
  ! P3GEO             parameter 3 
  ! IUGEO             position output file  for each IGEO
  !
  ! Liquid-Xtal-Mayer-Saupe
  ! QLXMS
  ! 
  ! Miscelaneous Dipole constraints
  ! QMDIP  
  ! DIP0
  ! PWMDIP
  !
  ! Primary Shell Solvation model
  ! QBHEL 
  ! QSHEL
  ! NTBHEL 
  ! NTSHEL 
  ! LSTSHL 
  ! LPOINT 
  ! UPDF
  ! NEILIS
  ! TEMPLIST
  ! NUMNEI
  ! WATLIST
  ! NWAT
  ! NPATM
  ! CHFR
  ! SPACE
  ! NPROT
  ! NPWH
  ! LPROT
  ! LPWH
  ! NONPOLAR
  ! DRSH 
  ! FOCO1
  ! FOCO2
  ! CUT
  ! MOLVOL
  ! RWEL
  ! XALER
  ! SCO
  ! AVEP
  ! CHCO
  ! PFINAL
  ! WATV
  ! TOTV
  !
  ! IUMMFP             output print unit

  LOGICAL QMMFP

  LOGICAL :: QGEO, QLXMS, QMDIP, QVMOD, QVCARD, QSYMM, QPRNT

  ! INTEGER
  ! GEO
  ! TODO type mmfp_geo_t with scalar members
  INTEGER MAXGEO, NTGEO
  integer,allocatable,dimension(:) :: &
       IGEO,  JGEO, lstgeo,ngeo,iugeo
  real(chm_real),allocatable,dimension(:) ::  &
       XRGEO,  YRGEO, ZRGEO, TRGEO, &
       XDGEO,  YDGEO, ZDGEO,  &
       DRGEO,  DTGEO, &
       FCGEO,  P1GEO, P2GEO, P3GEO
  ! MDIP
  INTEGER MAXMDIP,NTMDIP

  integer,allocatable,dimension(:) :: &
       LSTMDIP, NMDIP,  PWMDIP
  real(chm_real),allocatable,dimension(:) :: &
       XRMDIP,  YRMDIP,  ZRMDIP,  &
       XDMDIP,  YDMDIP,  ZDMDIP,  &
       FCMDIP,  D0MDIP

  ! LXMS
  INTEGER MAXLXMS,nlxms
  INTEGER,allocatable,dimension(:) ::  ILXMS, JLXMS    
  real(chm_real),allocatable,dimension(:) :: RLXMS, ALXMS

  ! SHEL
  integer :: NWAT, NPATM

  ! VMOD
  INTEGER  VMAST, MXMD,  NTMD,  NATOMV
  real(chm_real) :: rvmast
  real(chm_real),allocatable,dimension(:) :: &
       xini,yini,zini, XMODN, YMODN, ZMODN,RVMASS,CMPMODN, VSCRTCH, &
       QN, QMODN, KMODNU
  integer,allocatable,dimension(:) :: IMODNU, ISLCTV
  ! REAL
  ! SHEL
  REAL(chm_real) :: AVEP, TOTV

  ! VMOD
  real(chm_real)  KROTA, KCGRA

! -------------------------------------------------------------------------------
! EPR (B. Roux, 2010)
  real(chm_real) delppp, kenpp, sig2ppp 
  integer nppp
  integer nspinlabels, nrep_EPR

  real(chm_real),allocatable,dimension(:) ::  &
       Hpexp, Hpcalc, HdistE, TVEC
  integer,allocatable,dimension(:) ::  &
       Hspinlabel, Hspinlb1, Hspinlb2

      integer nptot
      logical QEPRR, QTVEC

contains

#if KEY_NOMISC==1
  SUBROUTINE MMFP0
    CALL WRNDIE(-1,'<CHARMM>','MMFP code is not compiled.')
    RETURN
  END SUBROUTINE MMFP0
#else /**/
  SUBROUTINE MMFP0
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use contrl
    use comand
    use primsh, only: bhelp, shelp
    use ssbpm, only: ssbp0
    use number
    use stream
    use string
    use psf
    use memory

    implicit none
    !-----------------------------------------------------------------------

    ! Addition of code in this version of mmfp.src ---------------------------
    ! June 2000
    ! Note : addition of code by Nilesh K. Banavali (NKB), Alex MacKerell, Jr.
    ! in this file pertaining to MMFP constraints on angles and dihedrals of
    ! center of masses of sets of atoms instead of individual atoms. Changes
    ! made were designed to be parallel to similar MMFP constraints on distance
    ! between center of masses previously implemented
    ! December 2002
    ! Addition of Molecular axis distance restraint "ADIStance". T.W.Allen.
    ! Addition of Flat bottom Dihedral restraint. T.W.Allen.
    ! Writing to unit  IUMMFP in each IGEO: Useful for Umbrella sampling. T.W.Allen
    ! addition for mmfp angle and dihedral codes, NKB
    !ML Stefan Boresch, Martin Leitgeb, May 2003:
    !ML addition of SAXON-WOOD-type restraint
    ! -------------------------------------------------------------------------

    !  Miscelaneous Local variables
    ! Local variable
    integer,allocatable,dimension(:) :: ISLCT,JSLCT

    CHARACTER(len=4)   WRD
    LOGICAL       DONE,EOF,LUSED,OK
    !
    OK    = .TRUE.
    LUSED = .FALSE.
    DONE  = .FALSE.
    EOF   = .FALSE.
    !     IUMMFP = GTRMI(COMLYN,COMLEN,'IUMM',-1)
    IF(PRNLEV.GE.2)THEN
       WRITE(OUTU,100) 'MISCELANEOUS MEAN FIELD POTENTIALS'
100    FORMAT(/,6X,A)
    ENDIF
    !-----------------------------------------------------------------------
1000 CALL XTRANE(COMLYN,COMLEN,'MMFP')
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
         '  MMFP> ')
    IF(EOF)THEN
       CALL PPSTRM(OK)
       IF(.NOT.OK)  RETURN
    ENDIF
    CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
    IF(LUSED) GOTO 1000

    WRD  = '    '
    WRD=NEXTA4(COMLYN,COMLEN)
    IF(WRD.EQ.'    ') GOTO 1000
    !.......................................................................

    IF(WRD.EQ.'GEO ')THEN
       !        -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       CALL GEO1(ISLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
    !.......................................................................
      ELSEIF(WRD.EQ.'EPR ')THEN
         call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
         call chmalloc('mmfp.src','MMFP0','JSLCT',natom,intg=JSLCT)
         CALL EPR000(ISLCT,JSLCT)
         call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
         call chmdealloc('mmfp.src','MMFP0','JSLCT',natom,intg=JSLCT)
       !.......................................................................
    ELSEIF(WRD.EQ.'LXMS')THEN
       !            -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       CALL LXMS1(ISLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)

       !.......................................................................
    ELSEIF(WRD.EQ.'MDIP')THEN
       !            -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       CALL MDIP1(ISLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)

       !.......................................................................
    ELSEIF(WRD.EQ.'SSBP')THEN
       !            -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       call chmalloc('mmfp.src','MMFP0','JSLCT',natom,intg=JSLCT)
       CALL SSBP0(ISLCT,JSLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       call chmdealloc('mmfp.src','MMFP0','JSLCT',natom,intg=JSLCT)

       !......................................................................
    ELSEIF(WRD.EQ.'BHEL')THEN
       !            -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       CALL BHELP(ISLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)


    ELSEIF(WRD.EQ.'SHEL')THEN
       !            -------------
       call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       CALL SHELP(ISLCT)
       call chmdealloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
       !.....................................................................
       ! VMOD
    ELSEIF(WRD.EQ.'VMOD')THEN
       !            -------------
       CALL VMOD1
       !.....................................................................
    ELSEIF(WRD.EQ.'BPCM')THEN                            !!!
       !            -------------                        !!!
       CALL BPCMAP( )                                    !!!
       !            -------------                        !!!
       !.....................................................................
    ELSEIF(WRD.EQ.'END ')THEN
       !            -------------
       QMMFP = QGEO .OR. QLXMS .OR. QMDIP .OR. QVMOD
       DONE  = .TRUE.
       !.......................................................................
    ELSE
       !
       CALL WRNDIE(-1,'<MMFP>','NOT A COMMAND:  '//WRD)
       DONE  = .TRUE.

    ENDIF
    !.......................................................................
    IF(DONE) RETURN
    GOTO 1000

  END SUBROUTINE MMFP0
  
 
!=========================================================================

 SUBROUTINE BPCMAP( )     
    use chm_kinds
    use comand
    use number
    use stream
    use string
    use memory
    use bpcmap_mod
    implicit none

300 FORMAT(/,6X,A,I6)                              
    QBPC = .TRUE.
    nbtp = GTRMI(COMLYN,COMLEN,'NBTP',-1)  
    WRITE(OUTU,300) 'Number of BPC-term maps is : ',nbtp
   
310 FORMAT(/,6X,A)         
    IF(NBTP.LE.0) THEN
       WRITE(OUTU,310) 'NO DIHIDRAL WERE SELECT FOR BPC MODULE.'
       RETURN
    END IF

    call chmalloc('mmfp.src','BMAP_INPUT','BMAP_DIM',nbtp,intg=BMAP_DIM)
    call chmalloc('mmfp.src','BMAP_INPUT','BMAP_UNT',nbtp,intg=BMAP_UNT)
    call chmalloc('mmfp.src','BMAP_INPUT','BMAP_DIM',nbtp,8,intg=BMAP_IND)
    call chmalloc('mmfp.src','BMAP_INPUT','BMAP_LAM',nbtp,crl=BMAP_LAM)

    CALL BMAP_INPUT( )    !!(nbtp, BMAP_DIM,BMAP_UNT, BMAP_IND,BMAP_LAM)

    RETURN   
     
END SUBROUTINE BPCMAP
 
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------------
subroutine BMAP_INPUT( )  !!(nbtp, BMAP_DIM,BMAP_UNT, BMAP_IND,BMAP_LAM)
    use chm_kinds
    use memory
    use comand
    use stream
    use string
    USE select
    use number
    use psf
    use coord
    use bpcmap_mod
    implicit none
     
    LOGICAL           ::  EOF    
    INTEGER           ::  I, J, K,M,N,KK,II
    CHARACTER(len=4)  ::  WRD

    INTEGER           ::  IMODE
    integer,allocatable,dimension(:) :: ISLCT

    call chmalloc('mmfp.src','MMFP0','ISLCT',natom,intg=ISLCT)
    EOF   = .FALSE.
    DO I=1, NBTP
        BMAP_LAM(I) = 1.d0
        BMAP_DIM(I) = 24
    END DO

    I=1 
    K=1
    II=1   
220 FORMAT(/,6X,A,F10.8)  
230 FORMAT(/,6X,A,8(I4))  
1900 IF(I.ne.II) THEN 
       WRITE(OUTU,220)  'LAMBDA = ', BMAP_LAM(II)
       WRITE(OUTU,230)  'DIM    = ', BMAP_DIM(II)
       WRITE(OUTU,230)  'UNIT   = ', BMAP_UNT(II)
       WRITE(OUTU,230)  'DIHE   = ',(BMAP_IND(II,J),J=1,8)
       II=I
    END IF
     
    IF(I.eq.(nbtp+1)) goto 1950
        
    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
         '  BPCMAP> ')
    IF (EOF) GOTO 1950
    CALL TRIME(COMLYN,COMLEN)
    IF (COMLEN.LE.0) GOTO 1900
        
1920 WRD='    '
    WRD=NEXTA4(COMLYN,COMLEN)
            
    IF(WRD.EQ.'LAMB') THEN
       BMAP_LAM(I)=NEXTF(COMLYN,COMLEN)
       IF (COMLEN.LE.0) GOTO 1900
       GOTO 1920
    END IF
  
    IF(WRD.EQ.'DIM ')THEN 
       BMAP_DIM(I) = NEXTI(COMLYN,COMLEN)
       IF (COMLEN.LE.0) GOTO 1900
       GOTO 1920
    END IF
     
    IF(WRD.EQ.'UNIT')THEN 
       BMAP_UNT(I) = NEXTI(COMLYN,COMLEN)
       K=K+1
       IF (COMLEN.LE.0) GOTO 1900  
       GOTO 1920
    END IF

    IF(WRD.EQ.'DIHE') THEN
       J=1
       DO N=1, 8 
          IMODE=0       !IMPLIES DEFAULT = ALL ATOMS SELECTED
          CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
                     .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
                     .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
          IF(IMODE.NE.0) THEN
             CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
          ENDIF
                  
          M=0
          DO KK=1, NATOM
             IF( ISLCT(KK)==1) THEN
                BMAP_IND(I,J)=KK
                J=J+1
                M=M+1      
                ISLCT(KK)=0
             END IF  
          END DO
       
          IF(M.ne.1) THEN
               CALL WRNDIE(-1,'<GEO>',' ERRO IN INPUT FILE WHEN USING MMFP/BPCMap/ ATOM SELECTION')
          END IF
       END DO
       I=I+1
       IF (COMLEN.LE.0) GOTO 1900
       GOTO 1920
    END IF
     
GOTO 1900

1950  CONTINUE  
240 FORMAT(/,6X,A)  
    IF(K.ne.I) THEN
       WRITE(OUTU,240) 'ERRO in BPCMAP INPUT REGION'
       CALL WRNDIE(-1,'<GEO>','ERRO in BPCMAP INPUT REGION')
    END IF 

    RETURN
END SUBROUTINE BMAP_INPUT


!!=====================================================
 

  SUBROUTINE GEO1(ISLCT)
    !-----------------------------------------------------------------------
    use memory
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    !---  use mmfp
    implicit none
    INTEGER ISLCT(*)
    !-----------------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4)   WRD
    INTEGER I, IGEO2, IMODE, JGEO2
    real(chm_real)  XREF, YREF, ZREF, TREF, XDIR, YDIR, ZDIR,  &
         DROFF, DTOFF
    real(chm_real)  FORC, P1, P2, P3
    INTEGER IUMMFP
    real(chm_real)  NORM
    LOGICAL QRCM, QDIST
    ! added by NKB for angles and dihedrals _______________________________
    LOGICAL QANGL, QDIHE
    ! end of addition by NKB_______________________________________________
    INTEGER IUNIT
    LOGICAL QADIST,QADPERP   ! Distance along a molecular axis
    LOGICAL QAANG
    !
    IF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN
       IF(QGEO)THEN
          IF(PRNLEV.GE.2)THEN
             WRITE(OUTU,100)'GEO RESET TO ZERO'
          ENDIF
          call chmdealloc('mmfp.src','GEO1','XRGEO',MAXGEO,crl=XRGEO)
          call chmdealloc('mmfp.src','GEO1','YRGEO',MAXGEO,crl=YRGEO)
          call chmdealloc('mmfp.src','GEO1','ZRGEO',MAXGEO,crl=ZRGEO)
          call chmdealloc('mmfp.src','GEO1','TRGEO',MAXGEO,crl=TRGEO)
          call chmdealloc('mmfp.src','GEO1','XDGEO',MAXGEO,crl=XDGEO)
          call chmdealloc('mmfp.src','GEO1','YDGEO',MAXGEO,crl=YDGEO)
          call chmdealloc('mmfp.src','GEO1','ZDGEO',MAXGEO,crl=ZDGEO)
          call chmdealloc('mmfp.src','GEO1','DRGEO',MAXGEO,crl=DRGEO)
          call chmdealloc('mmfp.src','GEO1','DTGEO',MAXGEO,crl=DTGEO)
          call chmdealloc('mmfp.src','GEO1','FCGEO',MAXGEO,crl=FCGEO)
          call chmdealloc('mmfp.src','GEO1','P1GEO',MAXGEO,crl=P1GEO)
          call chmdealloc('mmfp.src','GEO1','P2GEO',MAXGEO,crl=P2GEO)
          call chmdealloc('mmfp.src','GEO1','P3GEO',MAXGEO,crl=P3GEO)

          call chmdealloc('mmfp.src','GEO1','IGEO',MAXGEO,intg=IGEO)
          call chmdealloc('mmfp.src','GEO1','JGEO',MAXGEO,intg=JGEO)
          call chmdealloc('mmfp.src','GEO1','LSTGEO',MAXGEO,intg=LSTGEO)

          call chmdealloc('mmfp.src','GEO1','NGEO',MAXGEO,intg=NGEO)

          call chmdealloc('mmfp.src','GEO1','IUGEO',MAXGEO,intg=IUGEO)

          QGEO    = .FALSE.
       ELSE
          CALL WRNDIE(0,'<GEO>','GEO NOT SETUP')
       ENDIF
       !
    ELSEIF(INDXA(COMLYN,COMLEN,'PRIN').GT.0)THEN
       CALL PRIGEO(NATOM,LSTGEO,NGEO,NTGEO, &
            IGEO,JGEO, &
            XRGEO,YRGEO,ZRGEO,TRGEO, &
            XDGEO,YDGEO,ZDGEO, &
            DRGEO,DTGEO,FCGEO,P1GEO, &
            P2GEO,P2GEO,IUGEO)
       QPRNT = .TRUE.

       !
       ! Get parameter values
    ELSE
       !
       ! Initialize if necessary
       IF(.NOT.QGEO)THEN
          MAXGEO = 2+GTRMI(COMLYN,COMLEN,'MAXG',NATOM)
          IF(PRNLEV.GE.2)THEN
             WRITE(OUTU,100) 'GEO INITIALIZED, MAXGEO=',MAXGEO
100          FORMAT(1X,A,I5)
          ENDIF
          call chmalloc('mmfp.src','GEO1','XRGEO',MAXGEO,crl=XRGEO)
          call chmalloc('mmfp.src','GEO1','YRGEO',MAXGEO,crl=YRGEO)
          call chmalloc('mmfp.src','GEO1','ZRGEO',MAXGEO,crl=ZRGEO)
          call chmalloc('mmfp.src','GEO1','TRGEO',MAXGEO,crl=TRGEO)
          call chmalloc('mmfp.src','GEO1','XDGEO',MAXGEO,crl=XDGEO)
          call chmalloc('mmfp.src','GEO1','YDGEO',MAXGEO,crl=YDGEO)
          call chmalloc('mmfp.src','GEO1','ZDGEO',MAXGEO,crl=ZDGEO)
          call chmalloc('mmfp.src','GEO1','DRGEO',MAXGEO,crl=DRGEO)
          call chmalloc('mmfp.src','GEO1','DTGEO',MAXGEO,crl=DTGEO)
          call chmalloc('mmfp.src','GEO1','FCGEO',MAXGEO,crl=FCGEO)
          call chmalloc('mmfp.src','GEO1','P1GEO',MAXGEO,crl=P1GEO)
          call chmalloc('mmfp.src','GEO1','P2GEO',MAXGEO,crl=P2GEO)
          call chmalloc('mmfp.src','GEO1','P3GEO',MAXGEO,crl=P3GEO)
          call chmalloc('mmfp.src','GEO1','IUGEO',MAXGEO,intg=IUGEO)
          call chmalloc('mmfp.src','GEO1','IGEO',MAXGEO,intg=IGEO)
          call chmalloc('mmfp.src','GEO1','JGEO',MAXGEO,intg=JGEO)
          call chmalloc('mmfp.src','GEO1','LSTGEO',MAXGEO,intg=LSTGEO)
          call chmalloc('mmfp.src','GEO1','NGEO',MAXGEO,intg=NGEO)
          NTGEO  = 0
          QGEO   = .TRUE.
       ELSE
          IF(INDXA(COMLYN,COMLEN,'MAXM').GT.0)THEN
             CALL WRNDIE(-1,'<GEO>','MUST FIRST RESET GEO')
          ENDIF
       ENDIF

       IF(INDXA(COMLYN,COMLEN,'READ').GT.0)THEN
          IUNIT = GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
          IF(PRNLEV.GE.2)THEN
             WRITE(OUTU,100) 'Reading MMFP constraints from unit ',IUNIT
          ENDIF
          CALL RDGEO(IUNIT,NATOM,LSTGEO,NGEO,NTGEO, &
               IGEO,JGEO, &
               XRGEO,YRGEO,ZRGEO,TRGEO, &
               XDGEO,YDGEO,ZDGEO, &
               DRGEO,DTGEO,FCGEO,P1GEO, &
               P2GEO,P2GEO,IUGEO)
          RETURN

       ENDIF

       IUMMFP = GTRMI(COMLYN,COMLEN,'IUMM',-1)
       XREF = GTRMF(COMLYN,COMLEN,'XREF',ZERO)
       YREF = GTRMF(COMLYN,COMLEN,'YREF',ZERO)
       ZREF = GTRMF(COMLYN,COMLEN,'ZREF',ZERO)
       XDIR = GTRMF(COMLYN,COMLEN,'XDIR',ZERO)
       YDIR = GTRMF(COMLYN,COMLEN,'YDIR',ZERO)
       ZDIR = GTRMF(COMLYN,COMLEN,'ZDIR',ZERO)
       !     DREF = GTRMF(COMLYN,COMLEN,'DREF',ZERO)
       DROFF= GTRMF(COMLYN,COMLEN,'DROF',ZERO)
       TREF = GTRMF(COMLYN,COMLEN,'TREF',DROFF)
       DTOFF= GTRMF(COMLYN,COMLEN,'DTOF',ZERO)
       FORC = GTRMF(COMLYN,COMLEN,'FORC',ZERO)
       P1   = GTRMF(COMLYN,COMLEN,'P1',ZERO)
       P2   = GTRMF(COMLYN,COMLEN,'P2',ZERO)
       P3   = GTRMF(COMLYN,COMLEN,'P3',ZERO)
       QRCM = INDXA(COMLYN,COMLEN,'RCM ') .GT. 0
       IF(QRCM.AND.PRNLEV.GE.2) &
            WRITE(OUTU,100) 'Center of mass constraint'
       QADPERP = INDXA(COMLYN,COMLEN,'PERP') .GT. 0
       IF((IUMMFP.gt.0).AND.PRNLEV.GE.2) &
            WRITE(OUTU,'(a,i10)') 'Position output to Unit ',IUMMFP

       ! Geometrical boundary
       IF(INDXA(COMLYN,COMLEN,'SPHE').GT.0)THEN
          IGEO2 = 1
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Sphere constraint'
       ELSEIF(INDXA(COMLYN,COMLEN,'CYLI').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Cylinder constraint'
          IGEO2 = 2
       ELSEIF(INDXA(COMLYN,COMLEN,'PLAN').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Plane constraint'
          IGEO2 = 3
       ELSE
          CALL WRNDIE(-1,'<GEO>','GEOMETRY SHAPE UNDEFINED')
       ENDIF

       IF(IGEO2.NE.1)THEN
          NORM = SQRT(XDIR**2+YDIR**2+ZDIR**2)
          IF((NORM.EQ.ZERO).AND.(IGEO2.NE.1))THEN
             CALL WRNDIE(-1,'<GEO>','ZERO UNIT VECTOR')
          ELSE
             XDIR = XDIR/NORM
             YDIR = YDIR/NORM
             ZDIR = ZDIR/NORM
          ENDIF
       ENDIF

       ! Handle distance constraints
       QDIST= INDXA(COMLYN,COMLEN,'DIST') .GT. 0
       IF(QDIST)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Distance constraint'
          IGEO2 = -IGEO2
       ENDIF

       ! added by NKB for angle and dihedral constraints____________________
       QANGL= INDXA(COMLYN,COMLEN,'ANGL') .GT. 0
       IF(QANGL)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Angle constraint'
          IGEO2 = -4*IGEO2
       ENDIF

       QDIHE= INDXA(COMLYN,COMLEN,'DIHE') .GT. 0
       IF(QDIHE)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Dihedral constraint'
          IGEO2 = -5*IGEO2
       ENDIF

       ! end of addition by NKB________________________________________

       ! Distance along a molecular axis
       QADIST= INDXA(COMLYN,COMLEN,'ADIS') .GT. 0
       IF(QADIST.AND.PRNLEV.GE.2)THEN
          WRITE(OUTU,100) 'Molecular Axis Distance constraint'
          if(.not. QADPERP) then
             IGEO2 = -6*IGEO2
          ELSE
             IF(PRNLEV.GE.2) &
                  WRITE(OUTU,100) 'Constraint on Perpendicular distance.'
             IGEO2 = -7*IGEO2
          endif
       ENDIF

       ! Angle between two axes
       QAANG= INDXA(COMLYN,COMLEN,'AANG') .GT. 0
       IF(QAANG)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Axis angle constraint'
          IGEO2 = -8*IGEO2
       ENDIF

       ! Write header to IUMMFP, with info useful for umbrella sampling:
       if(IUMMFP.gt.0) then
          IF((QANGL.OR.QDIHE).AND.PRNLEV.GE.2)THEN
             write(IUMMFP,90) TREF, forc
          ELSE
             IF(PRNLEV.GE.2) &
                  write(IUMMFP,92) Droff, forc
          ENDIF
90        format('# TREF  = ',f14.8,' K = ',f14.8)
92        format('# DROFF = ',f14.8,' K = ',f14.8)
       endif

       ! Type of restraining function
       IF(INDXA(COMLYN,COMLEN,'HARM').GT.0)THEN
          JGEO2 = 1
          IF(INDXA(COMLYN,COMLEN,'INSI').GT.0)THEN
             JGEO2 = 1
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'Harmonic constraint, keep inside the maximum radius'
          ELSEIF(INDXA(COMLYN,COMLEN,'OUTS').GT.0)THEN
             JGEO2 = 2
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'Harmonic constraint, keep outside the maximum radius'
          ELSEIF(INDXA(COMLYN,COMLEN,'SYMM').GT.0)THEN
             JGEO2 = 3
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'Harmonic constraint, symmetric constraint'
       ! VN inserted RMAX option in c38 based on CR' work [QC: 11/17]
          ELSEIF(INDXA(COMLYN,COMLEN,'RMAX').GT.0)THEN
             ! can only be spherical
             IGEO2 = 1
             ! new code for RMAX restraint
             JGEO2 = 9
             IF(PRNLEV.GE.2) WRITE(OUTU,100) &
                  'Harmonic constraint, radius maximum of inner sphere'
       ! [QC: 11/17 Done] 
          ENDIF

       ELSEIF(INDXA(COMLYN,COMLEN,'QUAR').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Quartic constraint'
          JGEO2 = 4
       ELSEIF(INDXA(COMLYN,COMLEN,'EXPO').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Exponential constraint'
          JGEO2 = 5
       ELSEIF(INDXA(COMLYN,COMLEN,'GAUS').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Gaussian constraint'
          JGEO2 = 6
          QSYMM=INDXA(COMLYN,COMLEN,'SYMM').GT.0
          IF(PRNLEV.GE.2)THEN
            IF(QSYMM)THEN
              WRITE(OUTU,100) 'Gaussian constraint with symmetry '
            ELSE
              WRITE(OUTU,100) 'Gaussian constraint'
            ENDIF
          ENDIF
          !ML--------------------------------------------------------------
          !     SAXON-WOOD-type exponential flat-bottom restraint
       ELSEIF(INDXA(COMLYN,COMLEN,'SAWO').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Exp.Saxon-Wood restraint'
          JGEO2 = 7
       ELSEIF(INDXA(COMLYN,COMLEN,'SAWE').GT.0)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100) 'Exp.flat-bottom restraint'
          JGEO2 = 8
          !ML--------------------------------------------------------------
       ELSE
          JGEO2 = 1
          IF(PRNLEV.GE.2) WRITE(OUTU,100) &
               'Harmonic constraint, keep inside the maximum radius'
       ENDIF

       IF(JGEO2.eq.1) THEN
          !       WRITE(OUTU,100)'Harmonic constraint,keep inside maximum radius'
          IF(IGEO2.eq.-4 .or. IGEO2.eq.-5 .or. IGEO2.eq.-8)THEN
             if (prnlev >= 2) then
                WRITE(OUTU,100)'ANGL/DIHE restraint, flat-bottom potential'
                WRITE(OUTU,'(a,f10.3)') ' Angular offset from TREF is ',DTOFF
             endif
          ENDIF
       ELSEIF(JGEO2.eq.2) THEN
          !       WRITE(OUTU,100)'Harmonic constraint,keep outside maximum radius'
          IF(IGEO2.eq.-4 .or. IGEO2.eq.-5 .or. IGEO2.eq.-8)THEN
             if (prnlev >= 2) then
                WRITE(OUTU,100) 'ANGL/DIHE restraint, flat-outside potential'
                WRITE(OUTU,'(a,f10.3)') ' Angular offset from TREF is ',DTOFF
             endif
          ENDIF
       ENDIF

       ! NKB, Allen, Jan 2003
       ! check for negative offset with insi/outs options and angle/dihedral
       ! constraints, check for any offset specified with symmetric angle/dihedral
       ! constraints, check that only harmonic potential specified with angle/dihedral
       ! constraints (only harmonic has been tested)
       IF(IGEO2.eq.-4 .or. IGEO2.eq.-5 .or. IGEO2.eq.-8)THEN
          IF((JGEO2.EQ.1).OR.(JGEO2.EQ.2)) THEN
             IF(DTOFF.LT.ZERO)THEN
                IF(WRNLEV.GE.2 .and. prnlev >= 2)THEN
                   WRITE(OUTU,100) 'MMFP> WARNING: NEGATIVE OFFSET NOT&
                        & COMPATIBLE WITH ANGLE/DIHEDRAL CONSTRAINTS;'
                   WRITE(OUTU,100) '     ABSOLUTE VALUE OF DTOFF USED INSTEAD'
                ENDIF
                DTOFF=ABS(DTOFF)
             ENDIF
          ELSEIF(JGEO2.EQ.3)THEN
             IF(ABS(DTOFF).GT.RSMALL)THEN
                IF(WRNLEV.GE.2 .and. prnlev >= 2)THEN
                   WRITE(OUTU,100) 'MMFP> WARNING: ANGLE/DIHEDRAL SYMMETRIC &
                        &CONSTRAINT NOT COMPATIBLE WITH    '
                   WRITE(OUTU,100) '      DTOFF, DTOFF SET TO ZERO'
                ENDIF
                DTOFF=ZERO
             ENDIF
             !ML----------------------------------------------------------
          ELSEIF(JGEO2.EQ.7 .OR. JGEO2.EQ.8)THEN
             if (prnlev >= 2) WRITE(OUTU,100) 'MMFP> WARNING: ANGLE/DIHEDRAL SAWO'
             !ML----------------------------------------------------------
          ELSE
             CALL WRNDIE(-1,'MMFP>','FUNCTION NOT COMPATIBLE &
                  &WITH ANGLE/DIHEDRAL CONSTRAINT')
          ENDIF
       ENDIF

       ! NKB, Allen, Jan 2003

          ! QC: 11/17 Add Haiyun's change
          ! haiyun
          ! to define a moving center , we need a third selection
          ! If we have "MOVE" keyword, the first selection will be the center
          if (JGEO2 .EQ. 9) then 
            ! now restore to 1 9
            IGEO2 = 0      ! igeo is the type of geometrical constraint:
            JGEO2 = 11      ! the first selection will be used as ref if have "MOVE" 

            IMODE=0 !IMPLIES DEFAULT = NO ATOMS SELECTED
            CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,0,IMODE, &
               .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
               .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
            IF(IMODE.NE.0)THEN
              CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
            ENDIF

          CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
               IGEO2,IGEO,JGEO2,JGEO, &
               XREF,YREF,ZREF,TREF, &
               XRGEO,YRGEO,ZRGEO,TRGEO, &
               XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
               DROFF,DRGEO,DTOFF,DTGEO, &
               FORC,FCGEO, &
               P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
               IUGEO)

            WRITE(OUTU,*) "<GEO> RMAX: first  selection (reference) done"

            IGEO2 = 1      ! igeo is the type of geometrical constraint:
            JGEO2 = 9      ! the first selection will be used as ref if have "MOVE" 

          end if 

          !QC: 11/17 END

       IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
       ENDIF

       CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
            IGEO2,IGEO,JGEO2,JGEO, &
            XREF,YREF,ZREF,TREF, &
            XRGEO,YRGEO,ZRGEO,TRGEO, &
            XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
            DROFF,DRGEO,DTOFF,DTGEO, &
            FORC,FCGEO, &
            P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
            IUGEO)
	
	   ! BIOVIA Code Start : Bug fix	
       IF(JGEO2 .EQ. 9) THEN 
	        WRITE(OUTU,*) "<GEO> RMAX: second selection (inner) done"

       !QC: 11/17
       !VN Above CALL SELRPN(...) will define the inner atoms within RMAX sphere, i.e, JGEO2 = 9, by 
       ! the first selection
       !CR if we are using an RMAX restraint parse second selection for atoms outside the sphere of RMAX 
       !VN Below IF THEN is to define atoms ouside the RMAX sphere, i.e., JGEO2 is now 10.  
       ! BIOVIA Code End
            IGEO2 = 0
            JGEO2 = 10 

            IMODE=0 !IMPLIES DEFAULT = NO ATOMS SELECTED

            CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,0,IMODE, &
               .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
               .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
            IF(IMODE.NE.0)THEN
              CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
            ENDIF

          CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
               IGEO2,IGEO,JGEO2,JGEO, &
               XREF,YREF,ZREF,TREF, &
               XRGEO,YRGEO,ZRGEO,TRGEO, &
               XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
               DROFF,DRGEO,DTOFF,DTGEO, &
               FORC,FCGEO, &
               P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
               IUGEO)
           WRITE(OUTU,*) "<GEO> RMAX: third  selection (inner) done"
       ENDIF
       !QC: 11/17 END

       !      IF(QDIST.OR.QADIST)THEN
       ! Second set for distance constraints
       !      IGEO2 = 0

       ! next line added by NKB__________________
       IF(QDIST .OR. QADIST .OR. QANGL .OR. QDIHE .OR. QAANG) THEN
          
          IGEO2 = 0
          IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED

          CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
               .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
               .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
          IF(IMODE.NE.0)THEN
             CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
          ENDIF

          CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
               IGEO2,IGEO,JGEO2,JGEO, &
               XREF,YREF,ZREF,TREF, &
               XRGEO,YRGEO,ZRGEO,TRGEO, &
               XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
               DROFF,DRGEO,DTOFF,DTGEO, &
               FORC,FCGEO, &
               P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
               IUGEO)

          ! Third set for angle constraints
          IF(QANGL .OR. QDIHE .OR. QADIST .OR. QAANG) THEN

             ! addtion by NKB to correct for negative value of angle or dihedral
             !        IF(DROFF.LT.ZERO) DROFF=DROFF+360.0
             IF(TREF.LT.ZERO) TREF=TREF+360.0

             IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
             CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
                  .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
                  .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
             IF(IMODE.NE.0)THEN
                CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
             ENDIF

             CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
                  IGEO2,IGEO,JGEO2,JGEO, &
                  XREF,YREF,ZREF,TREF, &
                  XRGEO,YRGEO,ZRGEO,TRGEO, &
                  XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
                  DROFF,DRGEO,DTOFF,DTGEO, &
                  FORC,FCGEO, &
                  P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
                  IUGEO)
          ENDIF

          IF(QDIHE .OR. QAANG)THEN
             ! Fourth set for dihedral constraints
             !      IGEO2 = 0
             !      ENDIF

             IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
             CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
                  .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
                  .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
             IF(IMODE.NE.0)THEN
                CALL WRNDIE(-1,'<GEO>','ATOM SELECTION PARSING ERROR')
             ENDIF

             CALL STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
                  IGEO2,IGEO,JGEO2,JGEO, &
                  XREF,YREF,ZREF,TREF, &
                  XRGEO,YRGEO,ZRGEO,TRGEO, &
                  XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
                  DROFF,DRGEO,DTOFF,DTGEO, &
                  FORC,FCGEO, &
                  P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP, &
                  IUGEO)
          ENDIF

       ENDIF

    ENDIF

    RETURN
  END SUBROUTINE GEO1
  
  SUBROUTINE STGEO(NATOM,MAXGEO,ISLCT,LSTGEO,NGEO,NTGEO, &
       IGEO2,IGEO,JGEO2,JGEO, &
       XREF,YREF,ZREF,TREF,XRGEO,YRGEO,ZRGEO,TRGEO, &
       XDIR,YDIR,ZDIR,XDGEO,YDGEO,ZDGEO, &
       DROFF,DRGEO,DTOFF,DTGEO, &
       FORC,FCGEO, &
       P1,P1GEO,P2,P2GEO,P3,P3GEO,QRCM,IUMMFP,IUGEO)
    !----------------------------------------------------------------------
    use chm_kinds
    use stream
    implicit none
    INTEGER NATOM, MAXGEO, ISLCT(*), NGEO(*), NTGEO
    INTEGER LSTGEO(*), IGEO2, IGEO(*), JGEO2, JGEO(*)
    real(chm_real)  XREF, YREF, ZREF, TREF
    real(chm_real)  XRGEO(*), YRGEO(*), ZRGEO(*), TRGEO(*)
    real(chm_real)  XDIR, YDIR, ZDIR
    real(chm_real)  XDGEO(*), YDGEO(*), ZDGEO(*)
    real(chm_real)  DROFF, DRGEO(*), DTOFF, DTGEO(*)
    real(chm_real)  FORC, FCGEO(*)
    real(chm_real)  P1,P1GEO(*),P2,P2GEO(*),P3,P3GEO(*)
    INTEGER IUMMFP,IUGEO(*)
    LOGICAL QRCM
    ! Local variables
    INTEGER I, J, ISTART, ICOUNT, IOFF
    INTEGER NTOLD

    NGEO(1)=1
    NTOLD=NTGEO


    IF(NTGEO.EQ.0)THEN
       ISTART=1
    ELSE
       ISTART=NGEO(NTGEO+1)
    ENDIF
    ICOUNT=0

    DO I=1,NATOM
       IF(ISLCT(I).EQ.1)THEN
          ICOUNT=ICOUNT+1
          IOFF=ISTART+ICOUNT-1

          IF(IOFF.GT.MAXGEO)THEN
             CALL WRNDIE(-4,'<GEO>','NUMBER OF CONSTRAINT OVERFLOW')
          ENDIF
          LSTGEO(IOFF)=I

          IF(QRCM)THEN
             ! Center of MASS collective constraints
             NTGEO=NTOLD+1
             NGEO(NTGEO+1)=NGEO(NTGEO)+ICOUNT
          ELSE
             ! Individual constraints
             NTGEO=NTGEO+1
             NGEO(NTGEO+1)=NGEO(NTGEO)+1
          ENDIF

          IGEO(NTGEO)=IGEO2
          JGEO(NTGEO)=JGEO2
          XRGEO(NTGEO)=XREF
          YRGEO(NTGEO)=YREF
          ZRGEO(NTGEO)=ZREF
          TRGEO(NTGEO)=TREF
          XDGEO(NTGEO)=XDIR
          YDGEO(NTGEO)=YDIR
          ZDGEO(NTGEO)=ZDIR
          DRGEO(NTGEO)=DROFF
          DTGEO(NTGEO)=DTOFF
          FCGEO(NTGEO)=FORC
          P1GEO(NTGEO)=P1
          P2GEO(NTGEO)=P2
          P3GEO(NTGEO)=P3
          IUGEO(NTGEO)=IUMMFP

          IF(PRNLEV.GT.5)THEN
             WRITE(OUTU,100) NTGEO, NGEO(NTGEO), I, IGEO2, JGEO2, &
                  XREF, YREF, ZREF, TREF, &
                  XDIR, YDIR, ZDIR, DROFF, DTOFF, &
                  FORC, P1, P2, P3
!100          FORMAT(1X,5I4,12F8.3)
100          FORMAT(1X,5I10,13F8.3)
             IF(PRNLEV.GE.2) WRITE(OUTU,'(3(a,i10))') &
                  'ISTART=',ISTART,' ICOUNT=',ICOUNT,' IOFF=',IOFF
          ENDIF
       ENDIF
    enddo

    IF(PRNLEV.GE.2)THEN
       WRITE(OUTU,101) 'new constraints applied on ',ICOUNT,' atoms'
!101    FORMAT(1X,4(A,I4))
101    FORMAT(1X,4(A,I10))
       WRITE(OUTU,101) 'the total number of constraints is ',NTGEO
       WRITE(OUTU,101) 'the total number of atoms affected is ',IOFF
    ENDIF
    IF(PRNLEV.GT.5)THEN
       DO I=1,NTGEO
          WRITE(OUTU,'(1x)')
          WRITE(OUTU,101) 'constraint: ',I, &
               ' affecting ',NGEO(I+1)-NGEO(I),' atoms'
          WRITE(OUTU,102) &
               ' GEOM=',IGEO(I), ' TYPE=',JGEO(I), &
               ' DROFF=',DRGEO(I), ' DTOFF=',DTGEO(I), &
               ' FORC=',FCGEO(I),' P1=',P1GEO(I), ' P2=',P2GEO(I),' P3=',P3GEO(I)
102       FORMAT(1X,2(A,I3),6(A,F8.3))
          WRITE(OUTU,103)'REF(x,y,z,t)=',XRGEO(I),YRGEO(I),ZRGEO(I),TRGEO(I)
103       FORMAT(1X,A,3F8.3)
          WRITE(OUTU,103) 'VEC=',XDGEO(I), YDGEO(I), ZDGEO(I)
          DO J=NGEO(I),NGEO(I+1)-1
             WRITE(OUTU,101) 'applied on atom ',LSTGEO(J)
          ENDDO
       enddo
    ENDIF

    RETURN
  END SUBROUTINE STGEO
  
  SUBROUTINE RDGEO(IUNIT,NATOM,LSTGEO,NGEO,NTGEO,IGEO,JGEO, &
       XRGEO,YRGEO,ZRGEO,TRGEO,XDGEO,YDGEO,ZDGEO,DRGEO, &
       DTGEO,FCGEO,P1GEO,P2GEO,P3GEO,IUGEO)
    !-----------------------------------------------------------------------
    use chm_kinds
    use dimens_fcm
    use string
    use stream
    use number
    use exfunc
    use comand

    implicit none
    INTEGER  IUNIT, NATOM, NGEO(*), NTGEO
    INTEGER  LSTGEO(*), IGEO(*), JGEO(*)
    real(chm_real)   XRGEO(*), YRGEO(*), ZRGEO(*), TRGEO(*)
    real(chm_real)   XDGEO(*), YDGEO(*), ZDGEO(*)
    real(chm_real)   DRGEO(*), DTGEO(*)
    real(chm_real)   FCGEO(*), P1GEO(*), P2GEO(*), P3GEO(*)
    INTEGER  IUGEO(*)
    ! Local variables
    INTEGER I, J
    LOGICAL       EOF
    !
    EOF   = .FALSE.

    CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
         'RDMMFP> ')

    NTGEO = GTRMI(COMLYN,COMLEN,'NTGE',0)
    IF(PRNLEV.GE.2) &
         WRITE(OUTU,100) 'The total number of constraints is ',NTGEO
100 FORMAT(1X,4(A,I4))

    NGEO(1)  = 1
    DO I=1,NTGEO

       CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.TRUE., &
            'RDMMFP> ')

       NGEO(I+1)  = GTRMI(COMLYN,COMLEN,'NGEO',0)
       IGEO(I)  = GTRMI(COMLYN,COMLEN,'GEOM',0)
       JGEO(I)  = GTRMI(COMLYN,COMLEN,'TYPE',0)
       DRGEO(I) = GTRMF(COMLYN,COMLEN,'DROF',ZERO)
       DTGEO(I) = GTRMF(COMLYN,COMLEN,'DTOF',ZERO)

       XRGEO(I) = GTRMF(COMLYN,COMLEN,'XREF',ZERO)
       YRGEO(I) = GTRMF(COMLYN,COMLEN,'YREF',ZERO)
       ZRGEO(I) = GTRMF(COMLYN,COMLEN,'ZREF',ZERO)
       TRGEO(I) = GTRMF(COMLYN,COMLEN,'TREF',ZERO)

       FCGEO(I) = GTRMF(COMLYN,COMLEN,'FORC',ZERO)
       P1GEO(I) = GTRMF(COMLYN,COMLEN,'P1',ZERO)
       P2GEO(I) = GTRMF(COMLYN,COMLEN,'P2',ZERO)
       P3GEO(I) = GTRMF(COMLYN,COMLEN,'P3',ZERO)
       IUGEO(I) = GTRMI(COMLYN,COMLEN,'IUMMFP',-1)

       DO J=NGEO(I),NGEO(I+1)-1
          CALL RDCMND(COMLYN,MXCMSZ,COMLEN,IUNIT,EOF,.TRUE.,.FALSE., &
               'RDMMFP> ')
          LSTGEO(J)=NEXTI(COMLYN,COMLEN)
          !      READ(IUNIT,'(I5)') LSTGEO(J)
          IF(PRNLEV.GE.2) WRITE(OUTU,'(I5)') LSTGEO(J)
       enddo
    enddo
    RETURN
  END SUBROUTINE RDGEO
  
  SUBROUTINE PRIGEO(NATOM,LSTGEO,NGEO,NTGEO,IGEO,JGEO, &
       XRGEO,YRGEO,ZRGEO,TRGEO,XDGEO,YDGEO,ZDGEO,DRGEO, &
       DTGEO,FCGEO,P1GEO,P2GEO,P3GEO,IUGEO)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    use chutil,only:atomid

    implicit none

    INTEGER  NATOM, NGEO(*), NTGEO
    INTEGER  LSTGEO(*), IGEO(*), JGEO(*)
    real(chm_real)   XRGEO(*), YRGEO(*), ZRGEO(*), TRGEO(*)
    real(chm_real)   XDGEO(*), YDGEO(*), ZDGEO(*)
    real(chm_real)   DRGEO(*), DTGEO(*)
    real(chm_real)   FCGEO(*), P1GEO(*), P2GEO(*), P3GEO(*)
    INTEGER  IUGEO(*)
    ! Local variables
    INTEGER I, J
    CHARACTER(len=8) AT, REN, SGID, RID

    IF(PRNLEV.LT.2) RETURN
    WRITE(OUTU,100) 'The total number of constraints is ',NTGEO
100 FORMAT(1X,4(A,I4))

    DO I=1,NTGEO
       WRITE(OUTU,'(1x)')
       WRITE(OUTU,100) 'Constraint:  ',I, &
            ' affecting ',NGEO(I+1)-NGEO(I),' atoms'
       WRITE(OUTU,101) ' GEOM=',IGEO(I),' TYPE=',JGEO(I), &
            ' DROFF=',DRGEO(I),' DTOFF=',DTGEO(I),' FORC=',FCGEO(I), &
            ' P1=',P1GEO(I),' P2=',P2GEO(I),' P3=',P3GEO(I)
101    FORMAT(1X,2(A,I3),6(A,F8.3))
       WRITE(OUTU,102) 'REF=',XRGEO(I), YRGEO(I), ZRGEO(I), TRGEO(I)
       WRITE(OUTU,102) 'VEC=',XDGEO(I), YDGEO(I), ZDGEO(I)
102    FORMAT(1X,A,4F8.3)
       DO J=NGEO(I),NGEO(I+1)-1
          CALL ATOMID(LSTGEO(J),SGID,RID,REN,AT)
          WRITE(OUTU,103) 'applied on atom ',LSTGEO(J), &
               AT(1:idleng),REN(1:idleng),SGID(1:idleng),RID(1:idleng)
       ENDDO
!103    FORMAT(1X,A,I4,4(1X,A))
103    FORMAT(1X,A,I10,4(1X,A))
    enddo

    RETURN
  END SUBROUTINE PRIGEO
  
  SUBROUTINE GEO2(EGEO,NATOM,X,Y,Z,DX,DY,DZ, &
       LSTGEO,NGEO,NTGEO, &
       IGEO,JGEO,AMASS, &
       XRGEO,YRGEO,ZRGEO,TRGEO, &
       XDGEO,YDGEO,ZDGEO, &
       DRGEO,DTGEO,FCGEO,P1GEO,P2GEO,P3GEO,IUGEO)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    use number
    use consta
    use contrl
    use energym
    use param_store, only: set_param
#if KEY_GCMC==1
    use dimens_fcm
    use gcmc
#endif 

    implicit none

    real(chm_real)   EGEO
    INTEGER  NATOM, NGEO(*), NTGEO
    real(chm_real)   X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    INTEGER  LSTGEO(*), IGEO(*), JGEO(*)
    real(chm_real)   AMASS(*)
    real(chm_real)   XRGEO(*), YRGEO(*), ZRGEO(*), TRGEO(*)
    real(chm_real)   XDGEO(*), YDGEO(*), ZDGEO(*)
    real(chm_real)   DRGEO(*), DTGEO(*)
    real(chm_real)   FCGEO(*), P1GEO(*), P2GEO(*), P3GEO(*)
    INTEGER  IUGEO(*), IUMMFP
    ! Local variables
    ! previous line addition by NKB to get raddeg and cosmax def _____________

    INTEGER I, J, IAT
    real(chm_real) XCM, YCM, ZCM, MTOT, R
    real(chm_real) XCM2, YCM2, ZCM2, MTOT2, R2
    real(chm_real) FX, FY, FZ, FR, GX, GY, GZ, RG, GR
    real(chm_real) HX, HY, HZ, HR
    real(chm_real) AX, AY, AZ, RA, BX, BY, BZ, RB
    real(chm_real) CX, CY, CZ

    ! addition by NKB for center of mass angles and dihedrals ________________
    real(chm_real) FXR, FYR, FZR, GXR, GYR, GZR, CST, RG2
    real(chm_real) HXR, HYR, HZR
    real(chm_real) AXR, AYR, AZR, RA2, BXR, BYR, BZR, RB2
    real(chm_real) CXR, CYR, CZR
    real(chm_real) RGI, RA2I, RB2I, RABI
    real(chm_real) GAA,GBB,FGA,HGB
    real(chm_real) DFX,DFY,DFZ,DHX,DHY,DHZ,DGX,DGY,DGZ
    real(chm_real) DTFX,DTFY,DTFZ,DTHX,DTHY,DTHZ,DTGX,DTGY,DTGZ
    real(chm_real) XCM3, YCM3, ZCM3, MTOT3, R3
    real(chm_real) XCM4, YCM4, ZCM4, MTOT4, R4
    real(chm_real) RA2R, RB2R, RG2R, RAR, RBR, RGR, FG, HG
    real(chm_real) DTXI, DTXJ, DTYI, DTYJ, DTZI, DTZJ

    ! NKB, Dec, 2002, addition
    real(chm_real) ST2R,STR,DA,SMALLV
    ! end of NKB, Dec, 2002 addition

    ! end of NKB addition _____________________________________________________

    real(chm_real) XREF, YREF, ZREF, TREF, XDIR, YDIR, ZDIR
    real(chm_real) DROFF, DTOFF, DELTA, FORC, P1, P2, P3
    real(chm_real) E, DE
    real(chm_real) DELX, DELY, DELZ
    !ML added EXPDP2 to support Saxon-Wood Potential
    real(chm_real) EXPDP1, EXPDP2
    real(chm_real) RPRINT
    ! Local logical for inside/outside testing:
    LOGICAL ROUTS

    ! For axial distance:
    real(chm_real) DELR,DELX0, DELY0, DELZ0
    real(chm_real) DAX,DAY,DAZ,RPERP
    real(chm_real) DRDX1,DRDY1,DRDZ1
    real(chm_real) DRDX2,DRDY2,DRDZ2
    real(chm_real) DRDX3,DRDY3,DRDZ3
    real(chm_real) DATOT

    ! QC: 11/17
    ! CR for RMAX restraint
    real(chm_real) RMAX, ERMAX
    INTEGER IATRMAX
    LOGICAL RMAXON, RMAXPRINT

    ! haiyun for RMAX moving center
    real(chm_real) weight_HY
    
    RMAXON=.FALSE.

    ! QC: 11/17 END

    EGEO=ZERO

    loop10: DO I=1,NTGEO

       IF(IGEO(I).EQ.0) cycle loop10
       E     = ZERO
       XCM   = ZERO
       YCM   = ZERO
       ZCM   = ZERO
       MTOT  = ZERO
       XCM2  = ZERO
       YCM2  = ZERO
       ZCM2  = ZERO
       MTOT2 = ZERO
       XREF  = XRGEO(I)
       YREF  = YRGEO(I)
       ZREF  = ZRGEO(I)
       TREF  = TRGEO(I)
       XDIR  = XDGEO(I)
       YDIR  = YDGEO(I)
       ZDIR  = ZDGEO(I)
       DROFF = DRGEO(I)
       DTOFF = DTGEO(I)
       FORC  = FCGEO(I)
       P1    = P1GEO(I)
       P2    = P2GEO(I)
       P3    = P3GEO(I)
       IUMMFP = IUGEO(I)

       ! added by NKB for angles and dihedrals ______________________________
       XCM3  = ZERO
       YCM3  = ZERO
       ZCM3  = ZERO
       MTOT3 = ZERO
       XCM4  = ZERO
       YCM4  = ZERO
       ZCM4  = ZERO
       MTOT4 = ZERO

       ! NKB, Dec, 2002, addition
       SMALLV = RPRECI
       ! end of NKB, Dec, 2002 addition

       ! end of addition by NKB ______________________________________________

       loop20: DO J=NGEO(I),NGEO(I+1)-1
          IAT=LSTGEO(J)
#if KEY_GCMC==1
          if(qgcmc) then
             IF (.NOT. GCMCON(IAT)) cycle loop20
          endif
#endif 
          MTOT=MTOT+AMASS(IAT)
          XCM=XCM+X(IAT)*AMASS(IAT)
          YCM=YCM+Y(IAT)*AMASS(IAT)
          ZCM=ZCM+Z(IAT)*AMASS(IAT)
          !    write(6,'(i5,3f10.5)') iat,x(iat),y(iat),z(iat)
       enddo loop20

       IF (MTOT .NE. 0.0) THEN
          XCM=XCM/MTOT
          YCM=YCM/MTOT
          ZCM=ZCM/MTOT
          DELX=XCM-XREF
          DELY=YCM-YREF
          DELZ=ZCM-ZREF
       ENDIF
       call set_param('XCM ',XCM)
       call set_param('YCM ',YCM)
       call set_param('ZCM ',ZCM)
          !  write(6,'(5x,A,4f10.5)') 'xcm, ycm, zcm, mtot=',xcm,ycm,zcm,mtot
          !  write(6,'(5x,A,3f10.5)') 'delx, dely, delz=',delx,dely,delz

       IF(IGEO(I).LT.0)THEN
          loop21: DO J=NGEO(I+1),NGEO(I+2)-1
             IAT=LSTGEO(J)
             MTOT2=MTOT2+AMASS(IAT)
             XCM2=XCM2+X(IAT)*AMASS(IAT)
             YCM2=YCM2+Y(IAT)*AMASS(IAT)
             ZCM2=ZCM2+Z(IAT)*AMASS(IAT)
             !   write(6,'(i5,3f10.5)') iat,x(iat),y(iat),z(iat)
          enddo loop21

          XCM2=XCM2/MTOT2
          YCM2=YCM2/MTOT2
          ZCM2=ZCM2/MTOT2
          DELX=XCM-XCM2
          DELY=YCM-YCM2
          DELZ=ZCM-ZCM2
          call set_param('XCM2',XCM2)
          call set_param('YCM2',YCM2)
          call set_param('ZCM2',ZCM2)
          !    write(6,'(5x,A,4f10.5)') 'xcm2,ycm2,zcm2,mtot2=',xcm2,ycm2,zcm2,mtot2
          !    write(6,'(5x,A,3f10.5)') 'delx,dely,delz=',delx,dely,delz
          !    write(6,'(5x,A,f10.5)') 'dist = ',sqrt(delx**2+dely**2+delz**2)
       ENDIF
       !

       ! added by NKB for angles and dihedrals ________________________
       IF(IGEO(I).eq.-4 .or. IGEO(I).eq.-5 .or. IGEO(I).eq.-8)THEN
          ! adjust for tref being in degrees instead of radians, NKB ____
          TREF = TREF/RADDEG
          DTOFF = DTOFF/RADDEG
       ENDIF

       IF(IGEO(I).eq.-4 .or. IGEO(I).eq.-5 .or. &
            IGEO(I).eq.-6 .or. IGEO(I).eq.-7 .or. &
            IGEO(I).eq.-8) THEN
          loop22: DO J=NGEO(I+2),NGEO(I+3)-1
             IAT=LSTGEO(J)
             MTOT3=MTOT3+AMASS(IAT)
             XCM3=XCM3+X(IAT)*AMASS(IAT)
             YCM3=YCM3+Y(IAT)*AMASS(IAT)
             ZCM3=ZCM3+Z(IAT)*AMASS(IAT)
             !     write(6,'(i5,3f10.5)') iat,x(iat),y(iat),z(iat)
          enddo loop22

          XCM3=XCM3/MTOT3
          YCM3=YCM3/MTOT3
          ZCM3=ZCM3/MTOT3
          DELX=XCM-XCM3
          DELY=YCM-YCM3
          DELZ=ZCM-ZCM3
          call set_param('XCM3',XCM3)
          call set_param('YCM3',YCM3)
          call set_param('ZCM3',ZCM3)
       ENDIF
       !

       IF(IGEO(I).EQ.-5 .or. IGEO(I).EQ.-8)THEN
          loop23: DO J=NGEO(I+3),NGEO(I+4)-1
             IAT=LSTGEO(J)
             MTOT4=MTOT4+AMASS(IAT)
             XCM4=XCM4+X(IAT)*AMASS(IAT)
             YCM4=YCM4+Y(IAT)*AMASS(IAT)
             ZCM4=ZCM4+Z(IAT)*AMASS(IAT)
             !     write(6,'(i5,3f10.5)') iat,x(iat),y(iat),z(iat)
          enddo loop23

          XCM4=XCM4/MTOT4
          YCM4=YCM4/MTOT4
          ZCM4=ZCM4/MTOT4
          DELX=XCM-XCM4
          DELY=YCM-YCM4
          DELZ=ZCM-ZCM4
          call set_param('XCM4',XCM4)
          call set_param('YCM4',YCM4)
          call set_param('ZCM4',ZCM4)
       ENDIF
       !
       ! end of addition by NKB ____________________________________________

       ! GEOMETRY
       ! Sphere
       IF(ABS(IGEO(I)).EQ.1)THEN

          R=DELX*DELX+DELY*DELY+DELZ*DELZ
          IF(ABS(R).GT.RSMALL)THEN
             R=SQRT(R)
             DELX=DELX/R
             DELY=DELY/R
             DELZ=DELZ/R

             ! Write statement added by NKB ______________________________________
             !      IF(PRNLEV.GT.2)THEN
             !      WRITE(OUTU,199) 'MMFP> Distance between center of masses = ', R
             !199   FORMAT(1X,A,F10.2)
             !      ENDIF
             ! end of addition by NKB ___________________________________________

          ELSE
             cycle loop10
          ENDIF

          ! Cylinder
       ELSEIF(ABS(IGEO(I)).EQ.2)THEN
          R=DELX*XDIR+DELY*YDIR+DELZ*ZDIR
          DELX=DELX-R*XDIR
          DELY=DELY-R*YDIR
          DELZ=DELZ-R*ZDIR
          R=(DELX*DELX+DELY*DELY+DELZ*DELZ)
          IF(ABS(R).GT.RSMALL)THEN
             R=SQRT(R)
             DELX=DELX/R
             DELY=DELY/R
             DELZ=DELZ/R
          ELSE
             cycle loop10
          ENDIF

          ! Plane
       ELSEIF(ABS(IGEO(I)).EQ.3)THEN
          R=DELX*XDIR+DELY*YDIR+DELZ*ZDIR
          DELX=XDIR
          DELY=YDIR
          DELZ=ZDIR
          IF(R.LT.ZERO)THEN
             IF(JGEO(I).NE.3)THEN
                R=ABS(R)
                DELX=-XDIR
                DELY=-YDIR
                DELZ=-ZDIR
             ENDIF
          ENDIF

          ! added by NKB for angle center of mass constraints ________________

       ELSE IF(ABS(IGEO(I)).EQ.4 .OR. ABS(IGEO(I)).EQ.8)THEN

          FX=XCM-XCM2
          FY=YCM-YCM2
          FZ=ZCM-ZCM2
          FR=SQRT(FX*FX + FY*FY + FZ*FZ)
          IF(FR.LT.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') ' MMFP> No valid angle.'
             RETURN
          ENDIF
          ! added by NKB, Dec, 2002, addition
          FXR=FX/FR
          FYR=FY/FR
          FZR=FZ/FR
          ! end of NKB, Dec, 2002, addition
          !
          IF (ABS(IGEO(I)).EQ.4) THEN
          GX=XCM3-XCM2
          GY=YCM3-YCM2
          GZ=ZCM3-ZCM2
          ELSE ! ABS(IGEO(I)).EQ.8
          GX=XCM3-XCM4
          GY=YCM3-YCM4
          GZ=ZCM3-ZCM4
          ENDIF
          GR=SQRT(GX*GX+GY*GY+GZ*GZ)
          IF(GR.LT.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') ' MMFP> No valid angle.'
             RETURN
          ENDIF
          ! modified by NKB, Dec, 2002, addition
          GXR=GX/GR
          GYR=GY/GR
          GZR=GZ/GR
          CST=FXR*GXR+FYR*GYR+FZR*GZR

          DTXI=(GXR-CST*FXR)/FR
          DTXJ=(FXR-CST*GXR)/GR
          DTYI=(GYR-CST*FYR)/FR
          DTYJ=(FYR-CST*GYR)/GR
          DTZI=(GZR-CST*FZR)/FR
          DTZJ=(FZR-CST*GZR)/GR
          ! end of NKB, Dec, 2002, addition
          !
          IF(ABS(CST).GE.COSMAX) THEN
             R=NINETY-SIGN(NINETY,CST)
             ! added by NKB, Dec, 2002, addition
             R=R/RADDEG
             ! end of NKB, Dec, 2002 addition
          ELSE
             !            R=ACOS(CST)*RADDEG
             R=ACOS(CST)
          ENDIF

          ! NKB, Allen, addition, Jan 2003
24        CONTINUE
          IF(R.GT.PI) R=R-(2.0*PI)
          IF(R.LT.(-1.0*PI)) R=R+(2.0*PI)
          IF(TREF.GT.PI) TREF=TREF-(2.0*PI)
          IF(TREF.LT.(-1.0*PI)) TREF=TREF+(2.0*PI)

          IF((R.GT.PI).OR.(R.LT.(-1.0*PI))) GOTO 24
          IF((TREF.GT.PI).OR.(TREF.LT.(-1.0*PI))) GOTO 24
          ! NKB, Allen, addition, Jan 2003

          ! correct for reference value
          R = R - TREF

          ! added by NKB, Dec, 2002, addition
          IF(ABS(CST).GE.COSMAX) THEN
             DA=R-DTOFF
             IF(ABS(DA).GT.0.1) THEN
                IF(WRNLEV.GE.2) THEN
                   WRITE(OUTU,50)
50                 FORMAT(' WARNING FROM MMFP. Angle', &
                        '  is almost linear.', &
                        /' Derivatives may be affected')
                ENDIF
             ENDIF
          ENDIF
          ! end of NKB, Dec, 2002 addition

          ! Write statement added by NKB ----------------------------------------
          !      IF(PRNLEV.GT.2)THEN
          !      WRITE(OUTU,200) 'MMFP> Angle of center of masses = ', R*RADDEG
          !200   FORMAT(1X,A,F10.2)
          !      ENDIF
          ! ---------------------------------------------------------------------

          ! added by NKB for dihedral center of mass constraints ________________

       ELSE IF(ABS(IGEO(I)).EQ.5)THEN

          FX=XCM-XCM2
          FY=YCM-YCM2
          FZ=ZCM-ZCM2
          FR=FX*FX + FY*FY + FZ*FZ
          FR=SQRT(FR)
          IF(FR.LT.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral '
             RETURN
          ENDIF
          FXR=FX/FR
          FYR=FY/FR
          FZR=FZ/FR
          !
          HX=XCM4-XCM3
          HY=YCM4-YCM3
          HZ=ZCM4-ZCM3
          HR=HX*HX + HY*HY + HZ*HZ
          HR=SQRT(HR)
          IF(HR.LT.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral '
             RETURN
          ENDIF
          HXR=HX/HR
          HYR=HY/HR
          HZR=HZ/HR
          !
          GX=XCM2-XCM3
          GY=YCM2-YCM3
          GZ=ZCM2-ZCM3
          RG=GX*GX+GY*GY+GZ*GZ
          RG=SQRT(RG)
          IF(RG.LT.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral '
             RETURN
          ENDIF
          GXR=GX/RG
          GYR=GY/RG
          GZR=GZ/RG
          !
          CST=-FXR*GXR-FYR*GYR-FZR*GZR
          IF(ABS(CST).GE.COSMAX) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral '
             RETURN
          ENDIF
          CST=HXR*GXR+HYR*GYR+HZR*GZR
          IF(ABS(CST).GE.COSMAX) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral '
             RETURN
          ENDIF
          !
          AX=FY*GZ-FZ*GY
          AY=FZ*GX-FX*GZ
          AZ=FX*GY-FY*GX
          BX=HY*GZ-HZ*GY
          BY=HZ*GX-HX*GZ
          BZ=HX*GY-HY*GX

          AXR=FYR*GZR-FZR*GYR
          AYR=FZR*GXR-FXR*GZR
          AZR=FXR*GYR-FYR*GXR
          BXR=HYR*GZR-HZR*GYR
          BYR=HZR*GXR-HXR*GZR
          BZR=HXR*GYR-HYR*GXR
          !
          RA2R=AXR*AXR+AYR*AYR+AZR*AZR
          RB2R=BXR*BXR+BYR*BYR+BZR*BZR
          RG2R=GXR*GXR+GYR*GYR+GZR*GZR
          RAR=SQRT(RA2R)
          RBR=SQRT(RB2R)
          RGR=SQRT(RG2R)

          RA2=AX*AX+AY*AY+AZ*AZ
          RB2=BX*BX+BY*BY+BZ*BZ
          ! added by NKB, Dec, 2002 addition
          ! next line contained bug, commented out
          !         RG2=GX*GX+GYR*GYR+GZ*GZ
          ! next line is new line with corrected code
          RG2=GX*GX+GY*GY+GZ*GZ
          ! end of NKB, Dec, 2002 addition
          RA=SQRT(RA2)
          RB=SQRT(RB2)
          RG=SQRT(RG2)

          IF(RAR.LE.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral.'
             RETURN
          ENDIF
          IF(RBR.LE.0.000001) THEN
             IF(WRNLEV.GE.2) WRITE(OUTU,'(A)') &
                  ' MMFP> No valid dihedral.'
             RETURN
          ENDIF
          !
          AXR=AX/RA
          AYR=AY/RA
          AZR=AZ/RA
          BXR=BX/RB
          BYR=BY/RB
          BZR=BZ/RB
          !
          CST=AXR*BXR+AYR*BYR+AZR*BZR
          IF(ABS(CST).GE.ONE) CST=SIGN(ONE,CST)
          R=ACOS(CST)
          CXR=AYR*BZR-AZR*BYR
          CYR=AZR*BXR-AXR*BZR
          CZR=AXR*BYR-AYR*BXR

          CX=AY*BZ-AZ*BY
          CY=AZ*BX-AX*BZ
          CZ=AX*BY-AY*BX

          IF(GXR*CXR+GYR*CYR+GZR*CZR.GT.0.0) R=-R
          !         R=R*RADDEG

          ! NKB, Allen, addition, Jan 2003
25        CONTINUE
          IF(R.GT.PI) R=R-(2.0*PI)
          IF(R.LT.(-1.0*PI)) R=R+(2.0*PI)
          IF(TREF.GT.PI) TREF=TREF-(2.0*PI)
          IF(TREF.LT.(-1.0*PI)) TREF=TREF+(2.0*PI)

          IF((R.GT.PI).OR.(R.LT.(-1.0*PI))) GOTO 25
          IF((TREF.GT.PI).OR.(TREF.LT.(-1.0*PI))) GOTO 25
          ! NKB, Allen, addition, Jan 2003

          ! correct for reference value
          R = R - TREF

          ! Write statement added by NKB -------------------------------------
          !      IF(PRNLEV.GT.2)THEN
          !      WRITE(OUTU,201) 'MMFP> Phi of center of masses = ', R*RADDEG
          !201   FORMAT(1X,A,F10.2)
          !      ENDIF
          ! ------------------------------------------------------------------

          ! addition of next section by NKB to compute first derivatives
          ! wrt cartesian coordinates for distribution of forces later ____________

          RGI=ONE/RG
          RA2I=ONE/RA2
          RB2I=ONE/RB2
          RABI=SQRT(RA2I*RB2I)

          ! GAA=-|G|/A^2, GBB=|G|/B^2, FG=F.G, HG=H.G
          !  FGA=F.G/(|G|A^2), HGB=H.G/(|G|B^2)
          FG=FX*GX+FY*GY+FZ*GZ
          HG=HX*GX+HY*GY+HZ*GZ
          FGA=FG*RA2I*RGI
          HGB=HG*RB2I*RGI
          GAA=-RA2I*RG
          GBB=RB2I*RG
          ! DTFi=d(phi)/dFi, DTGi=d(phi)/dGi, DTHi=d(phi)/dHi. (used in 2nd deriv)
          DTFX=GAA*AX
          DTFY=GAA*AY
          DTFZ=GAA*AZ
          DTGX=FGA*AX-HGB*BX
          DTGY=FGA*AY-HGB*BY
          DTGZ=FGA*AZ-HGB*BZ
          DTHX=GBB*BX
          DTHY=GBB*BY
          DTHZ=GBB*BZ
          ! DFi=dE/dFi, DGi=dE/dGi, DHi=dE/dHi.

          ! end of addition by NKB ________________________________________________

       ELSEIF(IGEO(I).EQ.-6 .or. IGEO(I).EQ.-7)THEN
          !      Distance along molecular axis:
          DELX=XCM3-XCM2
          DELy=yCM3-yCM2
          DELz=zCM3-zCM2
          DELr=sqrt(delx**2+dely**2+delz**2)
          DELX=delx/delr
          DELy=dely/delr
          DELz=delz/delr
          DELX0=0.5*(XCM2+XCM3)
          DELy0=0.5*(yCM2+yCM3)
          DELz0=0.5*(zCM2+zCM3)
          dax=XCM - DELx0
          day=yCM - DELy0
          daz=zCM - DELz0
          R=dax*DELX+day*DELY+daz*DELZ
          datot=sqrt(dax**2+day**2+daz**2)
          Rperp=sqrt(datot**2 - R**2)
          !      Partial derivatives dR/dx for forces:
          dRdx1= DELX
          dRdy1= DELy
          dRdz1= DELz
          dRdx2=dax/delr*(DELx**2-1.0)-0.5*DELx+ &
               delX/delr*(day*dely+daz*delz)
          dRdy2=day/delr*(DELy**2-1.0)-0.5*DELy+ &
               delY/delr*(dax*delx+daz*delz)
          dRdz2=daz/delr*(DELz**2-1.0)-0.5*DELz+ &
               delZ/delr*(dax*delx+day*dely)
          dRdx3=dax/delr*(1.0-DELx**2)-0.5*DELx- &
               delX/delr*(day*dely+daz*delz)
          dRdy3=day/delr*(1.0-DELy**2)-0.5*DELy- &
               delY/delr*(dax*delx+daz*delz)
          dRdz3=daz/delr*(1.0-DELz**2)-0.5*DELz- &
               delZ/delr*(dax*delx+day*dely)
          if(IGEO(I).EQ.-7) then
             !       Distance along perpendicular molecular axis:
             dRdx1= (dax-R*dRdx1)/Rperp
             dRdy1= (day-R*dRdy1)/Rperp
             dRdz1= (daz-R*dRdz1)/Rperp
             dRdx2=(-0.5*dax-R*dRdx2)/Rperp
             dRdy2=(-0.5*day-R*dRdy2)/Rperp
             dRdz2=(-0.5*daz-R*dRdz2)/Rperp
             dRdx3=(-0.5*dax-R*dRdx3)/Rperp
             dRdy3=(-0.5*day-R*dRdy3)/Rperp
             dRdz3=(-0.5*daz-R*dRdz3)/Rperp
             datot=Rperp
             Rperp=R
             R=datot
          endif

       ENDIF

       ! Restore angles by removing offsets
       IF(IGEO(I).eq.-4 .or. IGEO(I).eq.-5 .or. IGEO(I).eq.-8)THEN
          Rprint=(R+TREF)*RADDEG
          IF(Rprint.LT.ZERO) Rprint=Rprint+360.0  ! angle in degrees
          IF(Rprint.GT.360.0) Rprint=Rprint-360.0 ! angle in degrees
       ELSE
          Rprint=R
       ENDIF

       ! Write position to IUMMFP:
       IF(IUMMFP.gt.0) then
          IF(QDYNCALL) then ! MMFP fix 11/08
             IF(IGEO(I).eq.-4 .or. IGEO(I).eq.-8)THEN
                write(IUMMFP,190) Rprint
             ELSEIF(IGEO(I).eq.-5)THEN
                write(IUMMFP,192) Rprint
             ELSE
                write(IUMMFP,194) Rprint
             ENDIF
190          format(1x,f14.8) ! MMFP fix 11/08
192          format(1x,f14.8) ! MMFP fix 11/08
194          format(1x,f14.8) ! MMFP fix 11/08
          ENDIF   ! MMFP fix 11/08
       ENDIF

       call set_param('RGEO',RPRINT)
       ! Note that RGEO is already in degrees before the SETMSR call.

       ! Potential type E and its derivatives DE
       IF(IGEO(I).eq.-4 .or. IGEO(I).eq.-5 .or. IGEO(I).eq.-8)THEN

          ! NKB, Allen, addition, Jan 2003
26        CONTINUE
          IF(R.GT.PI) R=R-(2.0*PI)
          IF(R.LT.(-1.0*PI)) R=R+(2.0*PI)
          IF(DTOFF.GT.PI) DTOFF=DTOFF-(2.0*PI)
          IF(DTOFF.LT.(-1.0*PI)) DTOFF=DTOFF+(2.0*PI)
          IF((R.GT.PI).OR.(R.LT.(-1.0*PI))) GOTO 26
          IF((DTOFF.GT.PI).OR.(DTOFF.LT.(-1.0*PI))) GOTO 26

          ! negative values of dtoff are ignored because of the inside and outside
          ! options that can be used for the same purpose
          IF(JGEO(I).EQ.1)THEN
             ! inside
             DELTA=R-abs(DTOFF)
          ELSEIF(JGEO(I).EQ.2)THEN
             ! outside
             DELTA=R+abs(DTOFF)
          ELSEIF(JGEO(I).EQ.3)THEN
             ! symmetric (no dtoff)
             DELTA=R
             !ML-----symmetric SAWO-------------------------------------------
          ELSEIF(JGEO(I).EQ.7 .OR. JGEO(I).EQ.8)THEN 
             DELTA=R
             !ML--------------------------------------------------------------
          ELSE
             CALL WRNDIE(-1,'MMFP>','FUNCTION NOT COMPATIBLE WITH &
                  &ANGLE/DIHEDRAL CONSTRAINT')
          ENDIF
       ELSE
          DELTA=R-DROFF
       ENDIF

100    FORMAT(1x,A)

       ! NKB, Allen, addition, Jan 2003

       ROUTS=.FALSE.

       IF(IGEO(I).EQ.-4 .OR. IGEO(I).EQ.-5 .OR. IGEO(I).EQ.-8)THEN
          IF(DELTA.GT.ZERO) ROUTS=.TRUE.
       ELSE
          IF(R.GT.DROFF) ROUTS=.TRUE.
       ENDIF


       ! Half-Harmonic boundary inside
       IF(JGEO(I).EQ.1)THEN
          IF(ROUTS)THEN
             E  = HALF*FORC*DELTA**2
             DE = FORC*DELTA
          ELSE
             E = ZERO
             DE = ZERO
          ENDIF


          ! Half-Harmonic boundary outside
       ELSEIF(JGEO(I).EQ.2)THEN
          IF(.NOT.ROUTS)THEN
             E  = HALF*FORC*DELTA**2
             DE = FORC*DELTA
          ELSE
             E = ZERO
             DE = ZERO
          ENDIF


          ! Full-Harmonic boundary (inside and outside)
       ELSEIF(JGEO(I).EQ.3)THEN
          E  = HALF*FORC*DELTA**2
          DE = FORC*DELTA


          ! Half-Quartic boundary
       ELSEIF(JGEO(I).EQ.4)THEN
          IF(ROUTS)THEN
             E  = FORC*DELTA**2*(DELTA**2-P1)
             DE = FORC*(FOUR*DELTA**3-TWO*P1*DELTA)
          ELSE
             E = ZERO
             DE = ZERO
          ENDIF


          ! Decaying exponential (Olle Edholm membrane potential,
          !                       use with planar geometry)
       ELSEIF(JGEO(I).EQ.5)THEN
          IF(DELTA.GE.ZERO)THEN
             EXPDP1 = HALF*FORC*EXP(-DELTA/P1)
             E  = EXPDP1
             DE = -EXPDP1/P1
          ELSEIF(DELTA.LT.ZERO)THEN
             EXPDP1 = HALF*FORC*EXP(+DELTA/P1)
             E  = FORC - EXPDP1
             DE = -EXPDP1/P1
          ENDIF

          ! Gaussian boundary (for membrane proteins)
       ELSEIF(JGEO(I).EQ.6)THEN
          IF(ROUTS.OR.QSYMM)THEN
             EXPDP1 = EXP(-(DELTA/P1)**2)
             E  =  FORC*EXPDP1
             DE = -FORC*EXPDP1*(TWO*DELTA/P1**2)
          ELSE
             E  = FORC
             DE = ZERO
          ENDIF
          !ML----------------------------------------------------------
          ! Saxon-Wood-typ potential (flat bottom restraint)
          ! P3=0: default SW-potential,  P3=1: no limit 

       ELSEIF(JGEO(I).EQ.7)THEN
          IF(DELTA.GE.ZERO)THEN
             EXPDP1 = Exp((P2-DELTA)/P1)
             EXPDP2 = Exp(P2/P1)
             E  = FORC/(1-P3+EXPDP1)-FORC/(1-P3+EXPDP2)   
             DE = (FORC*EXPDP1)/(P1*(1-P3+EXPDP1)**2) 

             IF(PRNLEV.GE.6) WRITE(OUTU,101) DE,E
101          FORMAT(' DE,E: ',F10.5, F10.5)

          ELSEIF(DELTA.LT.ZERO)THEN
             EXPDP1 = Exp((P2+DELTA)/P1)
             EXPDP2 = Exp(P2/P1)
             E  = FORC/(1-P3+EXPDP1)-FORC/(1-P3+EXPDP2)
             DE = -(FORC*EXPDP1)/(P1*(1-P3+EXPDP1)**2)

             IF(PRNLEV.GE.6) WRITE(OUTU,102) DE,E
102          FORMAT(' DE,E: ',F10.5, F10.5)

          ENDIF

          ! Exponential flat bottom restraint (experimental stuff)
          ! The small offset 1D-06 avoids zero-division

       ELSEIF(JGEO(I).EQ.8)THEN
          IF(DELTA.GE.ZERO)THEN
             EXPDP1 =  Exp(FORC-(P2/(DELTA + 1D-06)))
             E  = EXPDP1
             DE = (P2*EXPDP1)/((1D-06+DELTA)**2)

             IF(PRNLEV.GE.6) WRITE(OUTU,103) DE,E
103          FORMAT(' DE,E: ',F10.5, F10.5)

          ELSEIF(DELTA.LT.ZERO)THEN
             EXPDP1 =  Exp(FORC-(P2/(-DELTA + 1D-06)))
             E  = EXPDP1
             DE = -(P2*EXPDP1)/((1D-06-DELTA)**2)

             IF(PRNLEV.GE.6) WRITE(OUTU,104) DE,E
104          FORMAT(' DE,E: ',F10.5, F10.5)

          ENDIF
        ! QC: 11/17
        ! the RMAX restraint - this should be done for all the atoms in the selections only once 
        ! so RMAXON is used to flag if this restraint has already been imposed
        ELSEIF(JGEO(I).EQ.9) THEN 
           RMAXON=.TRUE.


          !ML----------------------------------------------------------

       ENDIF

       IF(E.NE.ZERO)THEN
          EGEO = EGEO+E
          ! for distance center of mass constraint, distribution of forces
          IF((IGEO(I).LT.0).AND.(IGEO(I).GT.-4))THEN
             loop30: DO J=NGEO(I),NGEO(I+1)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DE*DELX*AMASS(IAT)/MTOT
                DY(IAT)=DY(IAT)+DE*DELY*AMASS(IAT)/MTOT
                DZ(IAT)=DZ(IAT)+DE*DELZ*AMASS(IAT)/MTOT
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop30
             loop31: DO J=NGEO(I+1),NGEO(I+2)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)-DE*DELX*AMASS(IAT)/MTOT2
                DY(IAT)=DY(IAT)-DE*DELY*AMASS(IAT)/MTOT2
                DZ(IAT)=DZ(IAT)-DE*DELZ*AMASS(IAT)/MTOT2
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop31

             ! for angle center of mass constraint, distribution of forces
          ELSEIF(IGEO(I).EQ.-4 .or. IGEO(I).EQ.-8)THEN

             ! added by NKB, Dec, 2002 addition
             IF(ABS(CST).GE.0.999) THEN
                ST2R=ONE/(ONE-CST*CST+SMALLV)
                STR=SQRT(ST2R)
                IF(TREF-DTOFF.LT.PT001) THEN
                   DE=MINONE*FORC*(ONE+DA*DA*SIXTH)
                ELSE IF(PI-TREF-DTOFF.LT.PT001) THEN
                   DE=FORC*(ONE+DA*DA*SIXTH)
                ELSE
                   DE=-DE*STR
                ENDIF
             ELSE
                ST2R=ONE/(ONE-CST*CST)
                STR=SQRT(ST2R)
                DE=-DE*STR
             ENDIF
             ! end of NKB, Dec, 2002 addition

             DFX=DE*DTXI
             DFY=DE*DTYI
             DFZ=DE*DTZI
             DGX=DE*DTXJ
             DGY=DE*DTYJ
             DGZ=DE*DTZJ
             ! first center of mass set of atoms
             loop32: DO J=NGEO(I),NGEO(I+1)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DFX*AMASS(IAT)/MTOT
                DY(IAT)=DY(IAT)+DFY*AMASS(IAT)/MTOT
                DZ(IAT)=DZ(IAT)+DFZ*AMASS(IAT)/MTOT
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop32
             ! second center of mass set of atoms
             loop33: DO J=NGEO(I+1),NGEO(I+2)-1
                IAT=LSTGEO(J)
                if (IGEO(I).eq.-4) then
                DX(IAT)=DX(IAT)-(DFX+DGX)*AMASS(IAT)/MTOT2
                DY(IAT)=DY(IAT)-(DFY+DGY)*AMASS(IAT)/MTOT2
                DZ(IAT)=DZ(IAT)-(DFZ+DGZ)*AMASS(IAT)/MTOT2
                else ! IGEO(I).eq.8
                DX(IAT)=DX(IAT)-DFX*AMASS(IAT)/MTOT2
                DY(IAT)=DY(IAT)-DFY*AMASS(IAT)/MTOT2
                DZ(IAT)=DZ(IAT)-DFZ*AMASS(IAT)/MTOT2
                endif
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop33
             ! third center of mass set of atoms
             loop34: DO J=NGEO(I+2),NGEO(I+3)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DGX*AMASS(IAT)/MTOT3
                DY(IAT)=DY(IAT)+DGY*AMASS(IAT)/MTOT3
                DZ(IAT)=DZ(IAT)+DGZ*AMASS(IAT)/MTOT3
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop34
             if (IGEO(I).eq.-8) then
             ! fourth center of mass set of atoms
             loop341: DO J=NGEO(I+3),NGEO(I+4)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)-DGX*AMASS(IAT)/MTOT4
                DY(IAT)=DY(IAT)-DGY*AMASS(IAT)/MTOT4
                DZ(IAT)=DZ(IAT)-DGZ*AMASS(IAT)/MTOT4
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop341
             endif

             ! for dihedral center of mass constraint, distribution of forces
          ELSEIF(IGEO(I).EQ.-5)THEN
             DFX=DE*DTFX
             DFY=DE*DTFY
             DFZ=DE*DTFZ
             DGX=DE*DTGX
             DGY=DE*DTGY
             DGZ=DE*DTGZ
             DHX=DE*DTHX
             DHY=DE*DTHY
             DHZ=DE*DTHZ
             ! first center of mass set of atoms
             loop35: DO J=NGEO(I),NGEO(I+1)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DFX*AMASS(IAT)/MTOT
                DY(IAT)=DY(IAT)+DFY*AMASS(IAT)/MTOT
                DZ(IAT)=DZ(IAT)+DFZ*AMASS(IAT)/MTOT
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop35
             ! second center of mass set of atoms
             loop36: DO J=NGEO(I+1),NGEO(I+2)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+((DGX-DFX)*AMASS(IAT)/MTOT2)
                DY(IAT)=DY(IAT)+((DGY-DFY)*AMASS(IAT)/MTOT2)
                DZ(IAT)=DZ(IAT)+((DGZ-DFZ)*AMASS(IAT)/MTOT2)
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop36
             ! third center of mass set of atoms
             loop37: DO J=NGEO(I+2),NGEO(I+3)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)-((DGX+DHX)*AMASS(IAT)/MTOT3)
                DY(IAT)=DY(IAT)-((DGY+DHY)*AMASS(IAT)/MTOT3)
                DZ(IAT)=DZ(IAT)-((DGZ+DHZ)*AMASS(IAT)/MTOT3)
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop37
             ! fourth center of mass set of atoms
             loop38: DO J=NGEO(I+3),NGEO(I+4)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DHX*AMASS(IAT)/MTOT4
                DY(IAT)=DY(IAT)+DHY*AMASS(IAT)/MTOT4
                DZ(IAT)=DZ(IAT)+DHZ*AMASS(IAT)/MTOT4
                !     write(6,'(i5,3f10.5)') iat,dx(iat),dy(iat),dz(iat)
             enddo loop38

             ! For AXIAL distance center of mass constraint, T.W.Allen 2002
          ELSEIF(igeo(i).eq.-6 .or. igeo(i).eq.-7)THEN
             loop40: DO J=NGEO(I),NGEO(I+1)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DE*dRdx1*AMASS(IAT)/MTOT
                DY(IAT)=DY(IAT)+DE*dRdy1*AMASS(IAT)/MTOT
                DZ(IAT)=DZ(IAT)+DE*dRdz1*AMASS(IAT)/MTOT
             enddo loop40
             loop41: DO J=NGEO(I+1),NGEO(I+2)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DE*dRdx2*AMASS(IAT)/MTOT2
                DY(IAT)=DY(IAT)+DE*dRdy2*AMASS(IAT)/MTOT2
                DZ(IAT)=DZ(IAT)+DE*dRdz2*AMASS(IAT)/MTOT2
             enddo loop41
             loop42: DO J=NGEO(I+2),NGEO(I+3)-1
                IAT=LSTGEO(J)
                DX(IAT)=DX(IAT)+DE*dRdx3*AMASS(IAT)/MTOT3
                DY(IAT)=DY(IAT)+DE*dRdy3*AMASS(IAT)/MTOT3
                DZ(IAT)=DZ(IAT)+DE*dRdz3*AMASS(IAT)/MTOT3
             enddo loop42

             ! for single center of mass distribution of forces
          ELSE
             loop49: DO J=NGEO(I),NGEO(I+1)-1
                IAT=LSTGEO(J)
#if KEY_GCMC==1
                if(qgcmc) then
                   IF (.NOT. GCMCON(IAT)) cycle loop49
                endif
#endif 
                DX(IAT)=DX(IAT)+DE*DELX*AMASS(IAT)/MTOT
                DY(IAT)=DY(IAT)+DE*DELY*AMASS(IAT)/MTOT
                DZ(IAT)=DZ(IAT)+DE*DELZ*AMASS(IAT)/MTOT
             enddo loop49
          ENDIF

       ENDIF

    enddo loop10

    !QC: 11/17[w/ Haiyun's addition]
    !CR RMAX restraint 
    IF(RMAXON .eqv. .TRUE.) THEN
       ! loop over all atoms with MMFP restraints
       ! act on the ones with with RMAX property
       RMAX=0
       ERMAX=0
       IATRMAX=-1

        ! haiyun add the following
        do i = 1, ntgeo
            if (JGEO(i) .eq. 11) then 
                  weight_HY = 0.0
                  xref = 0.0
                  yref = 0.0
                  zref = 0.0
                 DO J=NGEO(I),NGEO(I+1)-1
                    IAT=LSTGEO(J)
                    !wmain(IAT)   ! weighting at the last
                    ! find mass
                    xref = xref + X(IAT)*amass(IAT)
                    yref = yref + y(IAT)*amass(IAT)
                    zref = zref + z(IAT)*amass(IAT)
                    weight_HY = weight_HY + amass(IAT) 
                 enddo
                 xref = xref /weight_HY 
                 yref = yref /weight_HY 
                 zref = zref /weight_HY 
            end if 
        end do 
       ! end of haiyun add

       loop50: DO I=1,NTGEO    !  ntgeo is number of constrain
          IF(JGEO(I).EQ.9) THEN   
          ! jgeo is the type of constrain
            FORC=FCGEO(I)
            IUMMFP = IUGEO(I)


            !determine Rmax (inner) for this step
            loop51: DO J=NGEO(I),NGEO(I+1)-1
               IAT=LSTGEO(J)
               !R=X(IAT)*X(IAT)+Y(IAT)*Y(IAT)+Z(IAT)*Z(IAT)      ! haiyun add ref
               R=(X(IAT)-xref)*(X(IAT)-xref) + &           
                 (y(IAT)-yref)*(y(IAT)-yref) + &
                 (z(IAT)-zref)*(z(IAT)-zref) 
               IF(R.GT.RMAX) THEN
                 RMAX=R
                 IATRMAX=IAT
               ENDIF
! QC: debug
!               IF(PRNLEV.GE.6)
! WRITE(OUTU,*) 'IN IAT ', IAT, 'R ', R , ' RMAX ', RMAX
            ENDDO loop51
          ENDIF
       ENDDO loop50

       RMAX=SQRT(RMAX)

       RMAXPRINT=.FALSE.
!       IF(PRNLEV.GE.6) WRITE(OUTU,*) 'RMAX ', RMAX, ' IAT ',  IATRMAX

       loop52: DO I=1,NTGEO
          IF(JGEO(I).EQ.10) THEN
            ! apply forces on outer atoms inside Rmax
            loop53: DO J=NGEO(I),NGEO(I+1)-1     !  ngeo is the number of atoms under cons
               IAT=LSTGEO(J)
               !R=X(IAT)*X(IAT)+Y(IAT)*Y(IAT)+Z(IAT)*Z(IAT)   ! haiyun add ref
               R=(X(IAT)-xref)*(X(IAT)-xref)+ &           !  haiyun add ref
                 (Y(IAT)-yref)*(Y(IAT)-yref)+ &
                 (Z(IAT)-zref)*(Z(IAT)-zref) 
               R=SQRT(R)
! QC: debug
!              WRITE(OUTU,*) 'OUT IAT ', IAT, 'R ', R

               IF(R.LT.RMAX) THEN
                 RMAXPRINT=.TRUE.
                 ! apply repulsive force to J
                 DELTA=R-RMAX
                 E  = HALF*FORC*DELTA**2
                 ERMAX = ERMAX+E
                 DE = FORC*DELTA
                 !DELX=X(IAT)/R     ! haiyun add ref
                 !DELY=Y(IAT)/R     ! haiyun add ref
                 !DELZ=Z(IAT)/R     ! haiyun add ref
                 DELX=(X(IAT)-xref)/R     ! haiyun add ref
                 DELY=(Y(IAT)-yref)/R     ! haiyun add ref
                 DELZ=(Z(IAT)-zref)/R     ! haiyun add ref
                 DX(IAT)=DX(IAT)+DE*DELX
                 DY(IAT)=DY(IAT)+DE*DELY
                 DZ(IAT)=DZ(IAT)+DE*DELZ

                 ! apply opposite force to RMAX atom
                 !DX(IATRMAX)=DX(IATRMAX)-DE*X(IATRMAX)/RMAX  ! haiyun add ref
                 !DY(IATRMAX)=DY(IATRMAX)-DE*Y(IATRMAX)/RMAX  ! haiyun add ref
                 !DZ(IATRMAX)=DZ(IATRMAX)-DE*Z(IATRMAX)/RMAX  ! haiyun add ref
                 DX(IATRMAX)=DX(IATRMAX)-DE*(X(IATRMAX)-xref)/RMAX  ! haiyun add ref
                 DY(IATRMAX)=DY(IATRMAX)-DE*(Y(IATRMAX)-yref)/RMAX  ! haiyun add ref
                 DZ(IATRMAX)=DZ(IATRMAX)-DE*(Z(IATRMAX)-zref)/RMAX  ! haiyun add ref
               ENDIF
             ENDDO loop53
           ENDIF
       ENDDO loop52
       EGEO=EGEO+ERMAX
       IF(IUMMFP.GT.0) then
         WRITE(IUMMFP, '(A5,1X,f14.8,f14.8)') 'RMAX ', RMAX, ERMAX
       ENDIF
    ENDIF
    !QC: 11/17 END

    RETURN
  END SUBROUTINE GEO2
  
  SUBROUTINE LXMS1(ISLCT)
    !-----------------------------------------------------------------------
    use memory
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    implicit none
    INTEGER ISLCT(*)
    ! Local variables
    CHARACTER(len=4)   WRD
    INTEGER I, IMODE
    real(chm_real)  REF, AMPL

    IF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN
       IF(QLXMS)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)'LXMS RESET TO ZERO'
          call chmdealloc('mmfp.src','LXMS1','RLXMS',MAXLXMS,crl=RLXMS)
          call chmdealloc('mmfp.src','LXMS1','ALXMS',MAXLXMS,crl=ALXMS)
          call chmdealloc('mmfp.src','LXMS1','ILXMS',MAXLXMS,intg=ILXMS)
          call chmdealloc('mmfp.src','LXMS1','JLXMS',MAXLXMS,intg=JLXMS)
          QLXMS = .FALSE.
       ELSE
          CALL WRNDIE(0,'<LXMS1>','LXMS NOT SETUP')
       ENDIF

    ELSEIF(INDXA(COMLYN,COMLEN,'PRIN').GT.0)THEN
       !
       ! Get parameter values
    ELSE
       !
       ! Initialize if necessary
       IF(.NOT.QLXMS)THEN
          MAXLXMS = GTRMI(COMLYN,COMLEN,'MAXL',NATOM)
          IF(PRNLEV.GE.2) &
               WRITE(OUTU,100) 'LXMS INITIALIZED, MAXLXMS=',MAXLXMS
100       FORMAT(1X,A,I5)
          call chmalloc('mmfp.src','LXMS1','RLXMS',MAXLXMS,crl=RLXMS)
          call chmalloc('mmfp.src','LXMS1','ALXMS',MAXLXMS,crl=ALXMS)
          call chmalloc('mmfp.src','LXMS1','ILXMS',MAXLXMS,intg=ILXMS)
          call chmalloc('mmfp.src','LXMS1','JLXMS',MAXLXMS,intg=JLXMS)
          NLXMS  = 0
          QLXMS  = .TRUE.
       ELSE
          IF(INDXA(COMLYN,COMLEN,'MAXL').GT.0)THEN
             CALL WRNDIE(-1,'<LXMS1>','MUST FIRST RESET LXMS')
          ENDIF
       ENDIF

       REF  = GTRMF(COMLYN,COMLEN,'REF',ZERO)
       AMPL = GTRMF(COMLYN,COMLEN,'AMPL',ZERO)

       IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(-1,'<LXMS1>','ATOM SELECTION PARSING ERROR')
       ENDIF

       CALL STLXMS(NATOM,ISLCT,REF,AMPL)

    ENDIF

    RETURN
  END SUBROUTINE LXMS1
  
  SUBROUTINE STLXMS(NATOM,ISLCT,REF,AMPL)
    !-----------------------------------------------------------------------
    use chm_kinds
    !---  use mmfp
    use stream
    implicit none
    INTEGER NATOM, ISLCT(*)
    real(chm_real)  REF, AMPL
    INTEGER I, IL, JL
    IL=0
    JL=0
    loop1: DO I=1,NATOM
       IF(ISLCT(I).EQ.1)THEN
          IF(IL.EQ.0)THEN
             IL=I
          ELSEIF(JL.EQ.0)THEN
             JL=I
          ELSE
             CALL WRNDIE(-1,'<STLXMS>','ATOM SELECTION PARSING ERROR')
          ENDIF
       ENDIF
    enddo loop1

    IF((IL.NE.0).AND.(JL.NE.0))THEN
       NLXMS = NLXMS + 1
       ILXMS(NLXMS) = IL
       JLXMS(NLXMS) = JL
       RLXMS(NLXMS) = REF
       ALXMS(NLXMS) = AMPL
       IF(PRNLEV.GT.5)THEN
          WRITE(OUTU,'(2i10,2f10.5)') IL,JL,REF,AMPL
       ENDIF
    ELSE
       CALL WRNDIE(-1,'<STLXMS>','A PAIR OF ATOMS WAS NOT SELECTED')
    ENDIF

    RETURN
  END SUBROUTINE STLXMS
#if KEY_UNUSED==1 /*lxms2_unused*/
  
  SUBROUTINE LXMS2(ELXMS,X,Y,Z,DX,DY,DZ, &
       NLXMS,ILXMS,JLXMS,RLXMS,ALXMS)
    !-----------------------------------------------------------------------
    !
    !  Liquid Xtal Mayer-Saupe potential oriented along the Z axis
    !     ELXMS = ampl*[P2(cos)-ref]**2
    !     where P2(cos) = 0.5*(3*cos**2-1) and cos is taken with the Z-axis
    !
    use chm_kinds
    use number
    use stream
    implicit none
    INTEGER  NLXMS, ILXMS(*), JLXMS(*)
    real(chm_real) ELXMS, X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real) RLXMS(*), ALXMS(*)
    !  local variables
    INTEGER I, i1, i2
    real(chm_real) ampl,ref,tempo, elxms2
    real(chm_real) uv(3),uv2, p2
    real(chm_real) ddx,ddy,ddz

    IF(PRNLEV.GT.5) WRITE(OUTU,'(a)') 'LXMS2'

    elxms=zero
    loop1: do i=1,nlxms
       i1=ilxms(i)
       i2=jlxms(i)
       ampl=alxms(i)
       ref=rlxms(i)
       !
       !  Principal axis
       uv(1)=x(i1)-x(i2)
       uv(2)=y(i1)-y(i2)
       uv(3)=z(i1)-z(i2)
       uv2=dsqrt(uv(1)**2+uv(2)**2+uv(3)**2)
       uv(1)=uv(1)/uv2
       uv(2)=uv(2)/uv2
       uv(3)=uv(3)/uv2
       p2 = half*(three*uv(3)**2-one)

       elxms2=ampl*(p2-ref)**2
       elxms=elxms+elxms2

       IF(PRNLEV.GT.5)THEN
          WRITE(OUTU,100) I, I1, I2, AMPL, REF, P2, ELXMS2
100       FORMAT(1x,3I5,4F10.4)
       ENDIF

       !  Derivatives
       tempo=2*ampl*(p2-ref)*(three*uv(3))/uv2

       ddx=tempo*(-uv(3)*uv(1))
       ddy=tempo*(-uv(3)*uv(2))
       ddz=tempo*(one-uv(3)*uv(3))

       dx(i1)=dx(i1)+ddx
       dy(i1)=dy(i1)+ddy
       dz(i1)=dz(i1)+ddz
       dx(i2)=dx(i2)-ddx
       dy(i2)=dy(i2)-ddy
       dz(i2)=dz(i2)-ddz

    enddo loop1

    RETURN
  END SUBROUTINE LXMS2
#endif /* (lxms2_unused)*/
  
  SUBROUTINE MDIP1(ISLCT)
    !-----------------------------------------------------------------------
    ! MDIPole mean fields constraints (MDIP)
    !
    ! E = (1/POWER) * FORC *
    !               { [MDIPX,MDIPY,MDIPZ]*[XDIR,YDIR,ZDIR] - DIP0 } ** POWER
    !
    ! the dipole is calculated with respect to (XREF,YREF,ZREF)
    !
    use memory
    use chm_kinds
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use coord
    !---  use mmfp

    implicit none
    INTEGER ISLCT(*)
    !
    !-----------------------------------------------------------------------
    ! Local variables
    CHARACTER(len=4)   WRD
    INTEGER I, IMODE
    real(chm_real)  XREF, YREF, ZREF, XDIR, YDIR, ZDIR
    real(chm_real)  FORC, DIP0
    INTEGER POWER
    real(chm_real)  NORM

    IF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN
       IF(QMDIP)THEN
          IF(PRNLEV.GE.2) WRITE(OUTU,100)'MDIP RESET TO ZERO'
          call chmdealloc('mmfp.src','MDIP1','XRMDIP',MAXMDIP,crl=XRMDIP)
          call chmdealloc('mmfp.src','MDIP1','YRMDIP',MAXMDIP,crl=YRMDIP)
          call chmdealloc('mmfp.src','MDIP1','ZRMDIP',MAXMDIP,crl=ZRMDIP)
          call chmdealloc('mmfp.src','MDIP1','XDMDIP',MAXMDIP,crl=XDMDIP)
          call chmdealloc('mmfp.src','MDIP1','YDMDIP',MAXMDIP,crl=YDMDIP)
          call chmdealloc('mmfp.src','MDIP1','ZDMDIP',MAXMDIP,crl=ZDMDIP)
          call chmdealloc('mmfp.src','MDIP1','FCMDIP',MAXMDIP,crl=FCMDIP)
          call chmdealloc('mmfp.src','MDIP1','D0MDIP',MAXMDIP,crl=D0MDIP)
          call chmdealloc('mmfp.src','MDIP1','PWMDIP',MAXMDIP,intg=PWMDIP)
          call chmdealloc('mmfp.src','MDIP1','LSTMDIP',MAXMDIP,intg=LSTMDIP)
          call chmdealloc('mmfp.src','MDIP1','NMDIP',MAXMDIP,intg=NMDIP)
          QMDIP    = .FALSE.
       ELSE
          CALL WRNDIE(0,'<MDIP>','MDIP NOT SETUP')
       ENDIF

    ELSEIF(INDXA(COMLYN,COMLEN,'PRIN').GT.0)THEN
       CALL PRIMDIP(NATOM,LSTMDIP,NMDIP,NTMDIP, &
            XRMDIP,YRMDIP,ZRMDIP, &
            XDMDIP,YDMDIP,ZDMDIP, &
            FCMDIP,D0MDIP,PWMDIP)
       !
       ! Get parameter values
    ELSE
       !
       ! Initialize if necessary
       IF(.NOT.QMDIP)THEN
          MAXMDIP = GTRMI(COMLYN,COMLEN,'MAXM',NATOM)
          IF(PRNLEV.GE.2) &
               WRITE(OUTU,100) 'MDIP INITIALIZED, MAXMDIP=',MAXMDIP
100       FORMAT(1X,A,I5)
          call chmalloc('mmfp.src','MDIP1','XRMDIP',MAXMDIP,crl=XRMDIP)
          call chmalloc('mmfp.src','MDIP1','YRMDIP',MAXMDIP,crl=YRMDIP)
          call chmalloc('mmfp.src','MDIP1','ZRMDIP',MAXMDIP,crl=ZRMDIP)
          call chmalloc('mmfp.src','MDIP1','XDMDIP',MAXMDIP,crl=XDMDIP)
          call chmalloc('mmfp.src','MDIP1','YDMDIP',MAXMDIP,crl=YDMDIP)
          call chmalloc('mmfp.src','MDIP1','ZDMDIP',MAXMDIP,crl=ZDMDIP)
          call chmalloc('mmfp.src','MDIP1','FCMDIP',MAXMDIP,crl=FCMDIP)
          call chmalloc('mmfp.src','MDIP1','D0MDIP',MAXMDIP,crl=D0MDIP)
          call chmalloc('mmfp.src','MDIP1','PWMDIP',MAXMDIP,intg=PWMDIP)
          call chmalloc('mmfp.src','MDIP1','LSTMDIP',MAXMDIP,intg=LSTMDIP)
          call chmalloc('mmfp.src','MDIP1','NMDIP',MAXMDIP,intg=NMDIP)
          NTMDIP  = 0
          QMDIP   = .TRUE.
       ELSE
          IF(INDXA(COMLYN,COMLEN,'MAXD').GT.0)THEN
             CALL WRNDIE(-1,'<MDIP>','MUST FIRST RESET MDIP')
          ENDIF
       ENDIF

       XREF  = GTRMF(COMLYN,COMLEN,'XREF',ZERO)
       YREF  = GTRMF(COMLYN,COMLEN,'YREF',ZERO)
       ZREF  = GTRMF(COMLYN,COMLEN,'ZREF',ZERO)
       XDIR  = GTRMF(COMLYN,COMLEN,'XDIR',ZERO)
       YDIR  = GTRMF(COMLYN,COMLEN,'YDIR',ZERO)
       ZDIR  = GTRMF(COMLYN,COMLEN,'ZDIR',ZERO)
       FORC  = GTRMF(COMLYN,COMLEN,'FORC',ZERO)
       DIP0  = GTRMF(COMLYN,COMLEN,'DIP0',ZERO)
       POWER = GTRMI(COMLYN,COMLEN,'POWE',2)

       NORM = SQRT(XDIR**2+YDIR**2+ZDIR**2)
       IF(NORM.EQ.ZERO)THEN
          CALL WRNDIE(-1,'<MDIP>','ZERO UNIT VECTOR')
       ELSE
          XDIR = XDIR/NORM
          YDIR = YDIR/NORM
          ZDIR = ZDIR/NORM
       ENDIF
       !
       ! IMPLIES DEFAULT = ALL ATOMS SELECTED
       IMODE=0
       CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
       IF(IMODE.NE.0)THEN
          CALL WRNDIE(-1,'<MDIP>','ATOM SELECTION PARSING ERROR')
       ENDIF

       CALL STMDIP(NATOM,MAXMDIP,ISLCT,LSTMDIP,NMDIP,NTMDIP, &
            XREF,YREF,ZREF,XRMDIP,YRMDIP,ZRMDIP, &
            XDIR,YDIR,ZDIR,XDMDIP,YDMDIP,ZDMDIP, &
            FORC,FCMDIP, &
            DIP0,D0MDIP,POWER,PWMDIP)

    ENDIF

    RETURN
  END SUBROUTINE MDIP1
  
  SUBROUTINE STMDIP(NATOM,MAXMDIP,ISLCT,LSTMDIP,NMDIP,NTMDIP, &
       XREF,YREF,ZREF,XRMDIP,YRMDIP,ZRMDIP, &
       XDIR,YDIR,ZDIR,XDMDIP,YDMDIP,ZDMDIP, &
       FORC,FCMDIP, &
       DIP0,D0MDIP,POWER,PWMDIP)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    implicit none
    INTEGER NATOM, MAXMDIP, ISLCT(*), NMDIP(*), NTMDIP
    INTEGER LSTMDIP(*)
    real(chm_real)  XREF, YREF, ZREF
    real(chm_real)  XRMDIP(*), YRMDIP(*), ZRMDIP(*)
    real(chm_real)  XDIR, YDIR, ZDIR
    real(chm_real)  XDMDIP(*), YDMDIP(*), ZDMDIP(*)
    real(chm_real)  FORC, FCMDIP(*)
    real(chm_real)  DIP0,D0MDIP(*)
    INTEGER POWER,PWMDIP(*)
    ! Local variables
    INTEGER I, J, ISTART, ICOUNT, IOFF
    INTEGER NTOLD

    NMDIP(1)=1
    NTOLD=NTMDIP

    IF(NTMDIP.EQ.0)THEN
       ISTART=1
    ELSE
       ISTART=NMDIP(NTMDIP+1)
    ENDIF
    ICOUNT=0

    loop2: DO I=1,NATOM
       IF(ISLCT(I).EQ.1)THEN
          ICOUNT=ICOUNT+1
          IOFF=ISTART+ICOUNT-1
          IF(IOFF.GT.MAXMDIP)THEN
             CALL WRNDIE(-4,'<MDIP>','NUMBER OF CONSTRAINT OVERFLOW')
          ENDIF
          LSTMDIP(IOFF)=I

          NTMDIP=NTOLD+1
          NMDIP(NTMDIP+1)=NMDIP(NTMDIP)+ICOUNT

          XRMDIP(NTMDIP)=XREF
          YRMDIP(NTMDIP)=YREF
          ZRMDIP(NTMDIP)=ZREF
          XDMDIP(NTMDIP)=XDIR
          YDMDIP(NTMDIP)=YDIR
          ZDMDIP(NTMDIP)=ZDIR
          FCMDIP(NTMDIP)=FORC
          D0MDIP(NTMDIP)=DIP0
          PWMDIP(NTMDIP)=POWER

          IF(PRNLEV.GT.5)THEN
             WRITE(OUTU,100) NTMDIP, NMDIP(NTMDIP), I, POWER, &
                  XREF, YREF, ZREF, &
                  XDIR, YDIR, ZDIR, FORC, DIP0
100          FORMAT(1X,4I4,12F8.3)
             IF(PRNLEV.GE.2) WRITE(OUTU,'(1x,3(a,i10))') &
                  'ISTART=',ISTART,' ICOUNT=',ICOUNT,' IOFF=',IOFF
          ENDIF
       ENDIF
    enddo loop2
    IF(PRNLEV.GE.2)THEN
       WRITE(OUTU,101) 'new constraints applied on ',ICOUNT,' atoms'
101    FORMAT(1X,4(A,I4))
       WRITE(OUTU,101) 'the total number of constraints is ',NTMDIP
       WRITE(OUTU,101) 'the total number of atoms affected is ',IOFF
    ENDIF
    IF(PRNLEV.GT.5)THEN
       LOOP3: DO I=1,NTMDIP
          WRITE(OUTU,'(1x)')
          WRITE(OUTU,101) 'constraint: ',I, &
               ' affecting ',NMDIP(I+1)-NMDIP(I),' atoms'
          WRITE(OUTU,102) &
               ' POWER=',PWMDIP(I),' FORC=',FCMDIP(I),' DIP0=',D0MDIP(I)
102       FORMAT(1X,2(A,I3),6(A,F8.3))
          WRITE(OUTU,103) 'REF=',XRMDIP(I), YRMDIP(I), ZRMDIP(I)
103       FORMAT(1X,A,3F8.3)
          WRITE(OUTU,103) 'VEC=',XDMDIP(I), YDMDIP(I), ZDMDIP(I)
          DO J=NMDIP(I),NMDIP(I+1)-1
             WRITE(OUTU,101) 'applied on atom',LSTMDIP(J)
          ENDDO
       ENDDO LOOP3
    ENDIF

    RETURN
  END SUBROUTINE STMDIP
  
  SUBROUTINE PRIMDIP(NATOM,LSTMDIP,NMDIP,NTMDIP, &
       XRMDIP,YRMDIP,ZRMDIP,XDMDIP,YDMDIP,ZDMDIP, &
       FCMDIP,D0MDIP,PWMDIP)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    use chutil,only:atomid
    implicit none
    INTEGER  NATOM, NMDIP(*), NTMDIP
    INTEGER  LSTMDIP(*)
    real(chm_real)   XRMDIP(*), YRMDIP(*), ZRMDIP(*)
    real(chm_real)   XDMDIP(*), YDMDIP(*), ZDMDIP(*)
    real(chm_real)   FCMDIP(*), D0MDIP(*)
    INTEGER  PWMDIP(*)
    ! Local variables
    INTEGER I, J
    CHARACTER(len=8) AT, REN, SGID, RID
    IF(PRNLEV.LT.2)RETURN
    WRITE(OUTU,100) 'The total number of constraints is',NTMDIP
100 FORMAT(1X,4(A,I4))

    loop3: DO I=1,NTMDIP
       WRITE(OUTU,'(1x)')
       WRITE(OUTU,100) 'Constraint:',I, &
            ' affecting',NMDIP(I+1)-NMDIP(I),' atoms'
       WRITE(OUTU,101) &
            ' POWER=',PWMDIP(I),' FORC=',FCMDIP(I),' DIP0=',D0MDIP(I)
101    FORMAT(1X,A,I3,6(A,F8.3))
       WRITE(OUTU,102) 'REF=',XRMDIP(I), YRMDIP(I), ZRMDIP(I)
       WRITE(OUTU,102) 'VEC=',XDMDIP(I), YDMDIP(I), ZDMDIP(I)
102    FORMAT(1X,A,3F8.3)
       DO J=NMDIP(I),NMDIP(I+1)-1
          CALL ATOMID(LSTMDIP(J),SGID,RID,REN,AT)
          WRITE(OUTU,103) 'applied on atom',LSTMDIP(J), &
               AT(1:idleng),REN(1:idleng),SGID(1:idleng),RID(1:idleng)
       ENDDO
103    FORMAT(1X,A,I4,4(1X,A))
    enddo loop3

    RETURN
  END SUBROUTINE PRIMDIP
  
  SUBROUTINE MDIP2(EMDIP,NATOM,X,Y,Z,DX,DY,DZ,CG, &
       LSTMDIP,NMDIP,NTMDIP, &
       XRMDIP,YRMDIP,ZRMDIP,XDMDIP,YDMDIP,ZDMDIP, &
       FCMDIP,D0MDIP,PWMDIP)
    !-----------------------------------------------------------------------
    use chm_kinds
    use stream
    use number
    implicit none
    real(chm_real)   EMDIP
    INTEGER  NATOM
    real(chm_real)   X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real)   CG(*)
    INTEGER  LSTMDIP(*), NMDIP(*), NTMDIP
    real(chm_real)   XRMDIP(*), YRMDIP(*), ZRMDIP(*)
    real(chm_real)   XDMDIP(*), YDMDIP(*), ZDMDIP(*)
    real(chm_real)   FCMDIP(*), D0MDIP(*)
    INTEGER  PWMDIP(*)
    ! Local variables
    INTEGER I, J, IAT
    real(chm_real) DIPX, DIPY, DIPZ, R
    real(chm_real) XREF, YREF, ZREF, XDIR, YDIR, ZDIR
    real(chm_real) DIP0, DELTA, FORC
    INTEGER POWER
    real(chm_real) E, DE
    real(chm_real) DELX, DELY, DELZ, DIPOL

    EMDIP=ZERO

    loop10: DO I=1,NTMDIP

       DIPX  = ZERO
       DIPY  = ZERO
       DIPZ  = ZERO
       XREF  = XRMDIP(I)
       YREF  = YRMDIP(I)
       ZREF  = ZRMDIP(I)
       XDIR  = XDMDIP(I)
       YDIR  = YDMDIP(I)
       ZDIR  = ZDMDIP(I)
       DIP0  = D0MDIP(I)
       FORC  = FCMDIP(I)
       POWER = PWMDIP(I)

       loop20: DO J=NMDIP(I),NMDIP(I+1)-1
          IAT=LSTMDIP(J)
          DIPX=DIPX+(X(IAT)-XREF)*CG(IAT)
          DIPY=DIPY+(Y(IAT)-YREF)*CG(IAT)
          DIPZ=DIPZ+(Z(IAT)-ZREF)*CG(IAT)
       enddo loop20
       DIPOL  = DIPX*XDIR+DIPY*YDIR+DIPZ*ZDIR
       DELTA  = DIPOL - DIP0
       E  = (ONE/POWER)*FORC*DELTA**POWER
       DE = FORC*DELTA**(POWER-1)

       EMDIP = EMDIP + E
       loop30: DO J=NMDIP(I),NMDIP(I+1)-1
          IAT=LSTMDIP(J)
          DX(IAT)=DX(IAT)+DE*XDIR*CG(IAT)
          DY(IAT)=DY(IAT)+DE*YDIR*CG(IAT)
          DZ(IAT)=DZ(IAT)+DE*ZDIR*CG(IAT)
       enddo loop30

    enddo loop10

    RETURN
#endif 
  END SUBROUTINE MDIP2
  

  SUBROUTINE VMOD1
    !-----------------------------------------------------------------------
    ! VMOD
    ! Normal mode (or other vector) coordinate restraints
    !
    ! E = (KMOD)/2 * ( Q - Q0 ) ** 2
    ! DE = KMOD * ( Q -Q0) S sqrt(M)              
    !
    !     David Perahia, Sylvain Frederic, and Charles Robert 2002-2006
    !
    !-----------------------------------------------------------------------
    !
    use memory
    use chm_kinds
    use consta
    use reawri
    use dimens_fcm
    use number
    use comand
    use psf
    use select
    use stream
    use string
    use parallel
    use coord
    use coordc
    use ctitla
    use vibio, only: rdnmd0
    use vmod

    implicit none
    !
    !-----------------------------------------------------------------------

    ! Local variables
    INTEGER I,NFREQ,NDIM

    INTEGER IMODN,IDUMMY,NTMDC
    LOGICAL QDUMMY
    real(chm_real)  QNV,KMODN,EVMOD,KMODCURR,QNCURR

    IF(IOLEV.GT.0) THEN

       IF(INDXA(COMLYN,COMLEN,'INIT').GT.0)THEN

          IF(QVMOD)THEN
             CALL WRNDIE(0,'<VMOD>','VMOD MUST BE RESET FIRST')
          ENDIF

          QVMOD  = .TRUE.

          QVCARD  = INDXA(COMLYN,COMLEN,'CARD').GT.0
          QDUMMY  = INDXA(COMLYN,COMLEN,'FILE').GT.0
          NATOMV  = GTRMI(COMLYN,COMLEN,'NATM',NATOM)
          MXMD   =  GTRMI(COMLYN,COMLEN,'MXMD',1)

          NTMD    =  0
          call chmalloc('mmfp.src','VMOD1','XINI',NATOMV,crl=XINI)
          call chmalloc('mmfp.src','VMOD1','YINI',NATOMV,crl=YINI)
          call chmalloc('mmfp.src','VMOD1','ZINI',NATOMV,crl=ZINI)
          call chmalloc('mmfp.src','VMOD1','XMODN',NATOMV*MXMD,crl=XMODN)
          call chmalloc('mmfp.src','VMOD1','YMODN',NATOMV*MXMD,crl=YMODN)
          call chmalloc('mmfp.src','VMOD1','ZMODN',NATOMV*MXMD,crl=ZMODN)
          call chmalloc('mmfp.src','VMOD1','RVMASS',NATOMV,crl=RVMASS)
          call chmalloc('mmfp.src','VMOD1','CMPMODN',NATOMV*3,crl=CMPMODN)
          call chmalloc('mmfp.src','VMOD1','VSCRTCH',NATOMV*3,crl=VSCRTCH)
          call chmalloc('mmfp.src','VMOD1','QN',MXMD,crl=QN)
          call chmalloc('mmfp.src','VMOD1','QMODN',MXMD,crl=QMODN)
          call chmalloc('mmfp.src','VMOD1','KMODNU',MXMD,crl=KMODNU)
          call chmalloc('mmfp.src','VMOD1','IMODNU',MXMD,intg=IMODNU)
          call chmalloc('mmfp.src','VMOD1','ISLCTV',NATOMV,intg=ISLCTV)

          KROTA  = 0.000001
          KROTA  = GTRMF(COMLYN,COMLEN,'KROT',KROTA)
          KCGRA  = 1000.0
          KCGRA  = GTRMF(COMLYN,COMLEN,'KCGR',KCGRA)
          UMDN   = GTRMI(COMLYN,COMLEN,'UMDN',-1)
          UOUT   = GTRMI(COMLYN,COMLEN,'UOUT',98)
          NSVQ   = GTRMI(COMLYN,COMLEN,'NSVQ',10)

          CALL SELCTA(COMLYN,COMLEN,ISLCTV, &
               XINI,YINI,ZINI,WMAIN,.FALSE.)
          ISVQ   = 0
          ISVQR  = 0.0

          CALL ORIMOD(XINI,YINI,ZINI,X,Y,Z, &
               AMASS,RVMASS,RVMAST,NATOMV,ISLCTV)

          IF (PRNLEV.GT.2) THEN
             WRITE(OUTU,'(''VMOD INITIALIZED, MXMD='',I5)') MXMD
             WRITE(OUTU,'(''VMOD MODE FILE, OUTPUT FILE, NSVQ:'',3I5)') &
                  UMDN,UOUT,NSVQ 
             WRITE(OUTU,'(''RESTRAINED ATOMS='',I5)') NATOMV
             !         Write the header to UOUT (not OUTU)
             WRITE(UOUT,'(20A11)') 'Time, ps','EMODES', &
                  'ETRANS','EROTAT','MOD COORDS'
          ENDIF


       ELSEIF(INDXA(COMLYN,COMLEN,'RESE').GT.0)THEN

          IF(QVMOD)THEN
             IF (PRNLEV.GT.2) WRITE(OUTU,'(''VMOD RESET TO ZERO'')')
             call chmdealloc('mmfp.src','VMOD1','XINI',NATOMV,crl=XINI)
             call chmdealloc('mmfp.src','VMOD1','YINI',NATOMV,crl=YINI)
             call chmdealloc('mmfp.src','VMOD1','ZINI',NATOMV,crl=ZINI)
             call chmdealloc('mmfp.src','VMOD1','XMODN',NATOMV*MXMD,crl=XMODN)
             call chmdealloc('mmfp.src','VMOD1','YMODN',NATOMV*MXMD,crl=YMODN)
             call chmdealloc('mmfp.src','VMOD1','ZMODN',NATOMV*MXMD,crl=ZMODN)
             call chmdealloc('mmfp.src','VMOD1','RVMASS',NATOMV,crl=RVMASS)
             call chmdealloc('mmfp.src','VMOD1','CMPMODN',NATOMV*3,crl=CMPMODN)
             call chmdealloc('mmfp.src','VMOD1','VSCRTCH',NATOMV*3,crl=VSCRTCH)
             call chmdealloc('mmfp.src','VMOD1','QN',MXMD,crl=QN)
             call chmdealloc('mmfp.src','VMOD1','QMODN',MXMD,crl=QMODN)
             call chmdealloc('mmfp.src','VMOD1','KMODNU',MXMD,crl=KMODNU)
             call chmdealloc('mmfp.src','VMOD1','IMODNU',MXMD,intg=IMODNU)
             call chmdealloc('mmfp.src','VMOD1','ISLCTV',NATOMV,intg=ISLCTV)
             QVMOD    = .FALSE.
          ELSE
             CALL WRNDIE(0,'<VMOD>','VMOD NOT SETUP')
          ENDIF

       ELSEIF(INDXA(COMLYN,COMLEN,'PRIN').GT.0)THEN

          IF (PRNLEV.GT.2) THEN
             CALL PRVMOD(AMASS,RVMASS,RVMAST,EVMOD,NATOMV,X,Y,Z,  &
                  XINI,YINI,   ZINI, &
                  XMODN,YMODN, ZMODN,QN, &
                  QMODN,KMODNU,KCGRA,KROTA,IMODNU,  &
                  MXMD,NTMD,QNV)
          ENDIF

       ELSEIF(INDXA(COMLYN,COMLEN,'ADD').GT.0) THEN

          !       Add/change restraint parameters for a mode restraint

          IMODN  = GTRMI(COMLYN,COMLEN,'IMDN',0)
          KMODN  = GTRMF(COMLYN,COMLEN,'KMDN',ZERO)
          QNV    = GTRMF(COMLYN,COMLEN,'QN',ZERO)

          NTMD=NTMD+1
          IF(NTMD.GT.MXMD.AND.PRNLEV.GT.2)THEN
             CALL WRNDIE(-4,'<VMOD>','NUMBER OF RESTRAINTS EXCEEDED')
          ENDIF

          !  Read requested mode vector from file

          !  CHR Use extracted VIBRAN RDNMD0 routine to read normal modes file to get vec info.
          !  NATOMV (default NATOM) must correspond to the number of atoms in the mode file.
          !  Note CMPMODN receives the DDV vectors, other info goes into scratch arrays.
          NFREQ=0
          NDIM=0
          REWIND (UNIT=UMDN)
          CALL RDNMD0(QVCARD,NFREQ,MXMD,NATOMV*3,NDIM, &
               CMPMODN,VSCRTCH,VSCRTCH,VSCRTCH, &
               UMDN,IMODN,IMODN,1,IDUMMY)

          CALL STMOD(NATOMV,NATOMV*3,ISLCTV,CMPMODN, &
               UMDN,IMODN,KMODN,XMODN,YMODN,ZMODN, &
               QN,IMODNU,KMODNU,MXMD,NTMD,QNV)

          IF (PRNLEV.GT.2) THEN
             WRITE(OUTU,'(''VMOD RESTRAINT'',I5,'' ADDED'')') NTMD
             WRITE(OUTU,'(''KMODN, QN VALUES:'',2F8.1)') KMODN,QNV
          ENDIF

       ELSEIF(INDXA(COMLYN,COMLEN,'CHAN').GT.0) THEN

          !       Change parameters for existing mode restraint number NTMDC
          NTMDC  = GTRMI(COMLYN,COMLEN,'NRES',1)

          IF(NTMDC.GT.MXMD.AND.PRNLEV.GT.2)THEN
             CALL WRNDIE(-4,'<VMOD>','NUMBER OF RESTRAINTS EXCEEDED')
          ENDIF

          KMODCURR=KMODNU(NTMDC)
          QNCURR=QN(NTMDC)

          KMODN  = GTRMF(COMLYN,COMLEN,'KMDN',KMODCURR)
          QNV    = GTRMF(COMLYN,COMLEN,'QN',QNCURR) 

          kmodnu(ntmdc) = kmodn
          qn(ntmdc)=qnv

          IF (PRNLEV.GT.2) THEN
             WRITE(OUTU,'(''VMOD RESTRAINT'',I5,'' CHANGED'')') NTMDC
             WRITE(OUTU,'(''NEW KMODN, QN VALUES:'',2F8.1)') KMODN,QNV
          ENDIF

       ENDIF

       !     endif IOLEV .gt. 0
    ENDIF

    RETURN
  END SUBROUTINE VMOD1


  SUBROUTINE ORIMOD(XINI,YINI,ZINI,X,Y,Z,AMASS,RVMASS,RVMAST, &
       NATOMV,ISLCTV)
    !-----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    ! Read MAIN coordinates, which defines origin of mode restraint
    ! Define root mass array and total for selected atoms.
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    use ctitla
    implicit none
    !
    !-----------------------------------------------------------------------
    !
    INTEGER NATOMV, I
    INTEGER ISLCTV(*)
    real(chm_real) XINI(*), YINI(*), ZINI(*)
    real(chm_real) X(*), Y(*), Z(*)
    real(chm_real) AMASS(*),RVMASS(*),RVMAST,VMAST

    VMAST=ZERO
    DO I=1,NATOMV
       XINI(I)=X(I)
       YINI(I)=Y(I)
       ZINI(I)=Z(I)
       IF (ISLCTV(I).GT.0) THEN
          RVMASS(I)=DSQRT(AMASS(I))
          VMAST=VMAST+AMASS(I)
       ELSE
          RVMASS(I)=ZERO
       ENDIF
    ENDDO
    RVMAST=DSQRT(VMAST)

    RETURN
  END SUBROUTINE ORIMOD


  SUBROUTINE STMOD(NATOMV,NATV3,ISLCTV,CMPMODN,UMDN,IMODN,KMODN, &
       XMODN, YMODN, ZMODN,QN,IMODNU,KMODNU, &
       MXMD,NTMD,QNV)
    !-----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    ! Store (normalized) mode, restraint force constant, target projection
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    use ctitla
    use stream
    use parallel
    use vector
    implicit none
    !
    !-----------------------------------------------------------------------
    !
    INTEGER NDIM,NFREQ
    INTEGER NATV3, MXMD, NATOMV, ISLCTV(*)
    real(chm_real) XMODN(NATOMV,MXMD),YMODN(NATOMV,MXMD), &
         ZMODN(NATOMV,MXMD)
    real(chm_real) CMPMODN(NATV3),QNV,QN(*),KMODN,KMODNU(*)
    CHARACTER(len=4) HDR
    real(chm_real) dot,alpha
    INTEGER I, IAT, NTMD, IMODNU(*)
    INTEGER UMDN, IMODN
    INTEGER ICNTRL(20)

    LOGICAL vmodtest
    vmodtest=.false.
    !      vmodtest=.true.

    ! Store the mode number and target projection for this mode

    QN(NTMD)=QNV
    IMODNU(NTMD)=IMODN
    KMODNU(NTMD)=KMODN

    IAT=0
    DOT=ZERO
    DO I=1,NATV3,3
       IAT=IAT+1
       IF (ISLCTV(IAT).GT.0) THEN
          XMODN(IAT,NTMD)=CMPMODN(I)
          YMODN(IAT,NTMD)=CMPMODN(I+1)
          ZMODN(IAT,NTMD)=CMPMODN(I+2)
          DOT=DOT+CMPMODN(I)**2+CMPMODN(I+1)**2+CMPMODN(I+2)**2
       ELSE
          XMODN(IAT,NTMD)=ZERO
          YMODN(IAT,NTMD)=ZERO
          ZMODN(IAT,NTMD)=ZERO
       ENDIF
    ENDDO

    !     Normalize after atom selection
    ALPHA=DSQRT(ONE/DOT)
    dot=ZERO
    DO IAT=1,NATOMV
       XMODN(IAT,NTMD)=alpha*XMODN(IAT,NTMD)
       YMODN(IAT,NTMD)=alpha*YMODN(IAT,NTMD)
       ZMODN(IAT,NTMD)=alpha*ZMODN(IAT,NTMD)
       if (vmodtest) dot=dot+XMODN(IAT,NTMD)**2 + &
            YMODN(IAT,NTMD)**2+ZMODN(IAT,NTMD)**2
    ENDDO

    if (vmodtest) then
       write(6,'(a)') "STMOD:"
       write(6,'(a,2i10)') "* * * NATOMV, NATV3 is",NATOMV,NATV3
       write(6,'(i5,3E20.12)') NTMD,(CMPMODN(i),i=1,3)
       do i=1,1
          write(6,'(i5,3E20.12)') NTMD, &
               XMODN(i,ntmd),YMODN(i,ntmd),ZMODN(i,ntmd)
       enddo
       write(6,'(a,i10,f10.5)') &
            "* * * lengthsq of atom-selected vector",NTMD,dot
       call dotpr(CMPMODN,CMPMODN,3*NATOMV,dot)
       write(6,'(a,i10,f10.5)') &
            "* * * lengthsq of normal mode vector",NTMD,dot
    endif

    RETURN
  END SUBROUTINE STMOD


  SUBROUTINE VMOD2(AMASS,RVMASS,RVMAST,EVMOD,NATOMV, &
       X,Y,Z,DX,DY,DZ, &
       XINI,YINI,ZINI,XMODN,YMODN,ZMODN, &
       QN,QMODN,KMODNU,ISLCTV,KCGRA,KROTA,NTMD)
    !-----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    ! Harmonic potential restraining mode coordinate to target value QN
    !
    ! E = (KMOD)/2 * ( QMOD - Q0 ) ** 2
    ! DE = KMOD * ( QMOD -Q0) S sqrt(M)              
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    use stream
    use consta 
    use ctitla
    use vmod
    use reawri
    implicit none
    !
    !-----------------------------------------------------------------------
    ! Local variables
    logical testing
    INTEGER I,NATOMV,NTMD,INMD
    INTEGER ISLCTV(*)
    real(chm_real) EVMODM,EVMOD
    real(chm_real) EVMODT,EVMODR,KCGRA,KROTA
    real(chm_real) QN(*),KMODNU(*)
    real(chm_real) XMODN(NATOMV,*), YMODN(NATOMV,*), ZMODN(NATOMV,*)
    real(chm_real) RVMASS(*)
    real(chm_real) XINI(*),YINI(*),ZINI(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    real(chm_real) AMASS(*)
    real(chm_real) QMODN(*),QMODV
    real(chm_real) RVMAST,KDER,MINVAL
    real(chm_real) vmodvec(100)

    testing=.false.

    ! CHR Note: QMODV and QN are mrms values
    !     EVMODN is now (correctly) summmed over modes as EVMODM

    EVMODM=ZERO
    DO INMD=1,NTMD

       !       Note: XMODN,YMODN,ZMODN,RVMASS are zero for unselected atoms
       QMODV=ZERO
       DO I=1,NATOMV    
          QMODV = QMODV + RVMASS(I)*( &
               (X(I) - XINI(I))*XMODN(I,INMD) + &
               (Y(I) - YINI(I))*YMODN(I,INMD) +  &
               (Z(I) - ZINI(I))*ZMODN(I,INMD) )
       ENDDO

       QMODV=QMODV/RVMAST
       QMODN(INMD)=QMODV

       if (testing) then
          write(6,'(a,f8.2,2f8.3,i5,3f8.3,i5,3f8.3)') &
               "kmodnu(inmd),QMODV,QN(INMD),1,x,xini,xn,2,x,xini,xn", &
               kmodnu(inmd),QMODV,QN(INMD), &
               1,X(1),XINI(1),XMODN(1,1), &
               3,X(3),XINI(3),XMODN(3,1)
       endif

       EVMODM = EVMODM + (KMODNU(INMD)/2)*( QMODV - QN(INMD))**2

       ! CHR (Mass-weighting denominator included in KDER)
       KDER=KMODNU(INMD)*(QMODV-QN(INMD))/RVMAST

       !       Note: XMODN,YMODN,ZMODN,RVMASS are zero for unselected atoms
       DO I=1,NATOMV
          DX(I)=DX(I)+KDER*RVMASS(I)*XMODN(I,INMD)
          DY(I)=DY(I)+KDER*RVMASS(I)*YMODN(I,INMD)
          DZ(I)=DZ(I)+KDER*RVMASS(I)*ZMODN(I,INMD)
       ENDDO

    ENDDO
    !
    ! Global translation restraint
    CALL VMODTR(EVMODT,XINI,YINI,ZINI,KCGRA, &
         NATOMV,X,Y,Z,DX,DY,DZ,AMASS,ISLCTV)
    !
    ! Global rotation restraint
    CALL VMODRO(EVMODR,XINI,YINI,ZINI,KROTA, &
         NATOMV,X,Y,Z,DX,DY,DZ,AMASS,ISLCTV)

    !  TOTAL EVMOD,DX,DY,DZ    

    EVMOD = EVMODM + EVMODT + EVMODR

    ! print the projections and the restraint energies
    ISVQ=ISVQ+1
    ISVQR=ISVQR+1.0
    IF(ISVQ.EQ.NSVQ)THEN
       IF (PRNLEV.GT.2) WRITE(UOUT,'(20F11.4)')ISVQR*TIMEST, &
            EVMODM,EVMODT,EVMODR, &
            (QMODN(I),I=1,NTMD)
       ISVQ=0 
    END IF
    RETURN
  END SUBROUTINE VMOD2


  SUBROUTINE VMODTR(EVMODT,XINI,YINI,ZINI,KCGRA, &
       NATOMV,X,Y,Z,DX,DY,DZ,AMASS,ISLCTV)
    !----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    !     CALCULATES THE ENERGY OF A HARMONIC POSITIONAL CONSTRAINT
    !     IN TERM OF GRAVITY CENTER .
    !     XINI, YINI, AND ZINI GIVES THE INITIAL COORDINATES.
    !     KCGRA GIVES THE FORCE CONSTANT,
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    !-----------------------------------------------------------------------
    real(chm_real) EVMODT
    real(chm_real) XINI(*),YINI(*),ZINI(*)
    real(chm_real) KCGRA
    INTEGER NATOMV
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real) AMASS(*)
    INTEGER ISLCTV(*)
    !
    real(chm_real) AX,AY,AZ,XCGR,YCGR,ZCGR,AMAST
    INTEGER I,IADD,J,K,II,III
    !
    AX=0.0
    AY=0.0
    AZ=0.0
    XCGR=0.0
    YCGR=0.0
    ZCGR=0.0
    AMAST=0.0
    DO I=1,NATOMV
       IF (ISLCTV(I).EQ.1) THEN
          AMAST = AMAST+AMASS(I)
          AX=AX+(AMASS(I)*X(I))
          AY=AY+(AMASS(I)*Y(I))
          AZ=AZ+(AMASS(I)*Z(I))
          XCGR=XCGR+(AMASS(I)*XINI(I))
          YCGR=YCGR+(AMASS(I)*YINI(I))
          ZCGR=ZCGR+(AMASS(I)*ZINI(I))
       ENDIF
    ENDDO
    !      write(6,'(" AMAST =",f15.5)')AMAST

    EVMODT=(KCGRA/((AMAST**2)*2))*(((AX-XCGR)**2)+((AY-YCGR) &
         **2)+((AZ-ZCGR)**2))

    DO I=1,NATOMV
       IF (ISLCTV(I).EQ.1) THEN
          DX(I)=DX(I)+((KCGRA/(AMAST**2))*(AX-XCGR)*AMASS(I))
          DY(I)=DY(I)+((KCGRA/(AMAST**2))*(AY-YCGR)*AMASS(I))
          DZ(I)=DZ(I)+((KCGRA/(AMAST**2))*(AZ-ZCGR)*AMASS(I))
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE VMODTR


  SUBROUTINE VMODRO(EVMODR,XINI,YINI,ZINI,KROTA, &
       NATOMV,X,Y,Z,DX,DY,DZ,AMASS,ISLCTV)
    !----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    !     CALCULATES THE ENERGY OF A HARMONIC POSITIONAL CONSTRAINT
    !     IN TERM OF ROTATION .
    !     XINI, YINI, AND ZINI GIVES THE INITIAL COORDINATES.
    !     KROTA GIVES THE FORCE CONSTANT,
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    implicit none
    !
    !-----------------------------------------------------------------------
    real(chm_real) EVMODR
    real(chm_real) XINI(*),YINI(*),ZINI(*)
    real(chm_real) KROTA
    INTEGER NATOMV
    real(chm_real) X(*),Y(*),Z(*),DX(*),DY(*),DZ(*)
    real(chm_real) AMASS(*)
    INTEGER ISLCTV(*)
    !
    real(chm_real)  AXREF,AYREF,AZREF,AMAST,X1,Y1,Z1,X2,Y2,Z2
    real(chm_real)  KX,KY,KZ,X22,Y22,Z22
    INTEGER I,IADD,K,II,III,J
    !
    AXREF=ZERO
    AYREF=ZERO
    AZREF=ZERO
    AMAST=ZERO
    DO I=1,NATOMV
       IF (ISLCTV(I).EQ.1) THEN
          AMAST=AMAST+AMASS(I)
          AXREF=AXREF+(AMASS(I)*XINI(I))
          AYREF=AYREF+(AMASS(I)*YINI(I))
          AZREF=AZREF+(AMASS(I)*ZINI(I))
       ENDIF
    ENDDO
    AXREF=AXREF/AMAST
    AYREF=AYREF/AMAST
    AZREF=AZREF/AMAST
    KX=ZERO
    KY=ZERO
    KZ=ZERO
    DO I=1,NATOMV
       IF (ISLCTV(I).EQ.1) THEN
          X1=X(I)-AXREF
          Y1=Y(I)-AYREF
          Z1=Z(I)-AZREF
          X2=XINI(I)-AXREF
          Y2=YINI(I)-AYREF
          Z2=ZINI(I)-AZREF
          KX=KX+(AMASS(I)*(Y1*(Z1-Z2)-Z1*(Y1-Y2)))
          KY=KY+(AMASS(I)*(Z1*(X1-X2)-X1*(Z1-Z2)))
          KZ=KZ+(AMASS(I)*(X1*(Y1-Y2)-Y1*(X1-X2)))
       ENDIF
    ENDDO
    EVMODR = (KROTA/2)*((KX**2)+(KY**2)+(KZ**2))
    !
    DO I=1,NATOMV
       IF (ISLCTV(I).EQ.1) THEN
          X2=XINI(I)-AXREF
          Y2=YINI(I)-AYREF
          Z2=ZINI(I)-AZREF
          DX(I)=DX(I)+(KROTA*AMASS(I)*(KY*Z2-KZ*Y2))
          DY(I)=DY(I)+(KROTA*AMASS(I)*(KZ*X2-KX*Z2))
          DZ(I)=DZ(I)+(KROTA*AMASS(I)*(KX*Y2-KY*X2))
       ENDIF
    ENDDO
    RETURN
  END SUBROUTINE VMODRO



  SUBROUTINE PRVMOD(AMASS,RVMASS,RVMAST,EVMOD,NATOMV,X,Y,Z, &
       XINI,YINI,ZINI, &
       XMODN,YMODN,ZMODN,QN,QMODN,KMODNU,KCGRA,KROTA,IMODNU, &
       MXMD,NTMD,QNV)
    !-----------------------------------------------------------------------
    ! (VMOD, D Perahia, S Frederic, CH Robert.)
    ! Print normal mode coordinates and corresponding restraint energies
    !
    ! E = (KMOD)/2 * ( QMOD - Q0 ) ** 2
    !
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use number
    use stream
    !
    !-----------------------------------------------------------------------
    implicit none
    ! Local variables
    INTEGER I,NATOMV,NTMD,INMD,MXMD,IMODNU(*)
    real(chm_real) EVMODN,EVMOD
    real(chm_real) EVMODT,EVMODR,KCGRA,KROTA
    real(chm_real) QN(*),KMODNU(*)
    real(chm_real) XMODN(NATOMV,*), YMODN(NATOMV,*), ZMODN(NATOMV,*)
    real(chm_real) XINI(*),YINI(*),ZINI(*)
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) AMASS(*),RVMASS(*)
    real(chm_real) QMODN(*),QMODV
    real(chm_real) TOTMAS, RVMAST, QNV, RTOTMAS

    TOTMAS=ZERO
    DO I=1,NATOMV
       TOTMAS=AMASS(I) + TOTMAS
    ENDDO
    RTOTMAS=DSQRT(TOTMAS)

    IF (PRNLEV.GT.2) THEN
       WRITE(OUTU, &
            '(22x,''Q-target Q-current'','' Force-const E-restraint'')')
    ENDIF

    DO INMD=1,NTMD
       QMODV=ZERO
       DO I=1,NATOMV    
          QMODV = RVMASS(I)*( &
               (X(I) - XINI(I))*XMODN(I,INMD) + &
               (Y(I) - YINI(I))*YMODN(I,INMD) +  &
               (Z(I) - ZINI(I))*ZMODN(I,INMD) ) + QMODV
       ENDDO

       QMODV=QMODV/RVMAST
       QMODN(INMD)=QMODV
       EVMODN = (KMODNU(INMD)/2)*( QMODV - QN(INMD))**2

       IF (PRNLEV.GT.2) THEN
          WRITE(OUTU,'(''RESTRAINT'',I4,'' MOD'',I4,2F8.3,2F12.2)') &
               INMD,IMODNU(INMD),QN(INMD),QMODV,KMODNU(INMD),EVMODN
       ENDIF

    ENDDO

    !
    ! Global translation restraint
    !      CALL VMODTR(EVMODT,XINI,YINI,ZINI,KCGRA,
    !     &     NATOMV,X,Y,Z,DX,DY,DZ,AMASS)
    !
    ! Global rotation restraint
    !      CALL VMODRO(EVMODR,XINI,YINI,ZINI,KROTA,
    !     &     NATOMV,X,Y,Z,DX,DY,DZ,AMASS)

    !  TOTAL EVMOD,DX,DY,DZ    

    !      EVMOD = EVMODN + EVMODT + EVMODR

    ! print the projections and the restraint energies
    !     ISVQ=ISVQ+1
    !     ISVQR=ISVQR+1.0
    !     IF(ISVQ.EQ.NSVQ)THEN
    !       WRITE(UOUT,'(20F11.4)')ISVQR*TIMEST,
    !    &                        EVMODN,EVMODT,EVMODR,
    !    &                        (QMODN(I),I=1,NTMD)
    !       ISVQ=0 
    !     END IF
    RETURN
  END SUBROUTINE PRVMOD

      SUBROUTINE EPR000(ISLCT,LIST)
!-----------------------------------------------------------------------
    use dimens_fcm
    use number
    use exfunc
    use comand
    use psf
    use stream
    use coord
    use deriv
    use string
    use memory
    use select
    use parallel

    INTEGER ISLCT(*), LIST(*)
!-----------------------------------------------------------------------
! Local variables
      CHARACTER*4   WRD 
      integer I, J, IMODE, iunit, n, icount
      logical qverb
      
#if KEY_PARALLEL==1
      CALL PSND4(COMLEN,1)
      IF(COMLEN > 0) CALL PSNDC(COMLYN(1:COMLEN),1)
#endif

      if(QEPRR.and.INDXA(COMLYN,COMLEN,'RESE').GT.0)then
         call chmdealloc('mmfp.src','EPR000','Hpexp',nppp*nptot,crl=Hpexp)
         call chmdealloc('mmfp.src','EPR000','Hpcalc',nppp*nptot,crl=Hpcalc)
         call chmdealloc('mmfp.src','EPR000','HdistE',nrep_EPR*nrep_EPR,crl=HdistE)

         call chmdealloc('mmfp.src','EPR000','Hspinlabel',nrep_EPR*nspinlabels,intg=Hspinlabel)
         call chmdealloc('mmfp.src','EPR000','Hspinlb1',nptot,intg=Hspinlb1)
         call chmdealloc('mmfp.src','EPR000','Hspinlb2',nptot,intg=Hspinlb2)

         QEPRR = .false.
         IF(MYNOD.EQ.0) write(outu,'(a)') ' Resetting all EPR restraints '
         IF(MYNOD.EQ.0) write(outu,*)
         return

      endif
   
      IF(MYNOD.EQ.0) write(outu,'(a)') ' Setting up EPR restraints'
      iunit       = GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
      kenpp       = GTRMF(COMLYN,COMLEN,'KENP',ZERO)
      sig2ppp     = GTRMF(COMLYN,COMLEN,'SIG2',ONE)
      nrep_EPR    = GTRMI(COMLYN,COMLEN,'NREP',0)
      nspinlabels = GTRMI(COMLYN,COMLEN,'NSPI',0)
      nppp        = GTRMI(COMLYN,COMLEN,'NPPP',0)
      delppp      = GTRMF(COMLYN,COMLEN,'DELP',ONE)
      nptot       = GTRMI(COMLYN,COMLEN,'NPTO',0)
      qverb       = INDXA(COMLYN,COMLEN,'VERB').GT.0
      qtvec       = INDXA(COMLYN,COMLEN,'TVEC').GT.0

      IF(MYNOD.EQ.0) THEN
         write(outu,'(a,i4)') ' nreplica ',nrep_EPR
         write(outu,'(a,i4)') ' nspinlabels ',nspinlabels
         write(outu,'(a,i4)') ' Total number of spin label pairs (nptot) ',   &
                                            nptot
         write(outu,'(a,i4)')    &
           ' Number of data points in distance histogram (nppp) ',nppp
      ENDIF

      if(qtvec .AND. (MYNOD.EQ.0) ) write(outu,'(a)') ' Will read translation vectors'

      IF(ALLOCATED(Hpexp)) THEN
         call CHMREALLOC('mmfp.src','EPR000','Hpexp',nppp*nptot,crl=Hpexp)
      ELSE
         call chmalloc('mmfp.src','EPR000','Hpexp',nppp*nptot,crl=Hpexp)
      ENDIF

      IF(ALLOCATED(Hpcalc)) THEN
         call CHMREALLOC('mmfp.src','EPR000','Hpcalc',nppp*nptot,crl=Hpcalc)
      ELSE
         call chmalloc('mmfp.src','EPR000','Hpcalc',nppp*nptot,crl=Hpcalc)
      ENDIF

      IF(ALLOCATED(HdistE)) THEN
         call CHMREALLOC('mmfp.src','EPR000','HdistE',nrep_EPR*nrep_EPR,crl=HdistE)
      ELSE
         call chmalloc('mmfp.src','EPR000','HdistE',nrep_EPR*nrep_EPR,crl=HdistE)
      ENDIF

      IF(ALLOCATED(Hspinlabel)) THEN
         call CHMREALLOC('mmfp.src','EPR000','Hspinlabel',nrep_EPR*nspinlabels,intg=Hspinlabel)
      ELSE
         call chmalloc('mmfp.src','EPR000','Hspinlabel',nrep_EPR*nspinlabels,intg=Hspinlabel)
      ENDIF

      IF(ALLOCATED(Hspinlb1)) THEN
         call CHMREALLOC('mmfp.src','EPR000','Hspinlb1',nptot,intg=Hspinlb1)
      ELSE
         call chmalloc('mmfp.src','EPR000','Hspinlb1',nptot,intg=Hspinlb1)
      ENDIF

      IF(ALLOCATED(Hspinlb2)) THEN
         call CHMREALLOC('mmfp.src','EPR000','Hspinlb2',nptot,intg=Hspinlb2)
      ELSE
         call chmalloc('mmfp.src','EPR000','Hspinlb2',nptot,intg=Hspinlb2)
      ENDIF


      if(QTVEC)then
         IF(ALLOCATED(TVEC)) THEN
            call CHMREALLOC('mmfp.src','EPR000','TVEC',3*nspinlabels,crl=TVEC)
         ELSE
            call chmalloc('mmfp.src','EPR000','TVEC',3*nspinlabels,crl=TVEC)
         ENDIF
      else
         IF(ALLOCATED(TVEC)) THEN
            call CHMREALLOC('mmfp.src','EPR000','TVEC',1,crl=TVEC)
         ELSE
            call chmalloc('mmfp.src','EPR000','TVEC',1,crl=TVEC)
         ENDIF
      endif

      IMODE=0 !IMPLIES DEFAULT = ALL ATOMS SELECTED
      CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE,   &
            .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG,   &
            .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
      IF(IMODE.NE.0)THEN
         CALL WRNDIE(-1,'<EPR0>','ATOM SELECTION PARSING ERROR')
      ENDIF

      call EPR001(ISLCT, list, nrep_EPR, nspinlabels, nppp, nptot,   &
           Hpexp, Hpcalc, Hspinlabel, Hspinlb1,                      &
           Hspinlb2, HdistE, delppp, kenpp, sig2ppp, iunit,           &
           qverb,qtvec,tvec)

      QEPRR = .true.

      return
      END SUBROUTINE EPR000


      SUBROUTINE EPR001(ISLCT, list, nrep_EPR, nspinlabels, nppp,   &
                 nptot, pexp, pcalc, spinlabel, spinlb1, spinlb2,   &
                 dist, delppp, kenpp, sig2ppp, iunit, qverb,        &
                 qtvec, tvec) 
      use dimens_fcm
      use number
      use stream
      use coord
      use psf
      use deriv
      use chutil,only:atomid
      use parallel

      integer ISLCT(*), list(*), nrep_EPR, nspinlabels
      integer nppp, nptot
      real*8  pexp(nppp,nptot)
      real*8  pcalc(nppp,nptot)
      real*8  dist(nrep_EPR*nrep_EPR)
      integer spinlabel(nrep_EPR,nspinlabels)
      integer spinlb1(nptot), spinlb2(nptot)
      real*8 delppp, kenpp, sig2ppp 
      integer iunit
      logical qverb, qtvec
      real*8  tvec(nspinlabels,3)


      integer i, j, i1, j1, n, icount, ii
      integer i2, j2 
      real*8 sum, rr, enpp
      character*6 SEGIDX, RESIDX, RESNX, TYPEX

!     Make the list of the spin labels with their replicas
!     ----------------------------------------------------

      IF(MYNOD .EQ. 0) THEN
         icount = 0 
         do i=1,natom
            if(ISLCT(i).eq.1)then
               icount = icount + 1
               list(icount) = i
               if( prnlev.ge.9 )then
!                 write(OUTU,'(a,3i12)') ' selection ',
!    &                      i, icount, list(icount)
                  call ATOMID(i,SEGIDX,RESIDX,RESNX,TYPEX)
                  write(outu,'(2i12,1x,4a,3f10.5)')         &
                     i,icount,SEGIDX, RESIDX, RESNX, TYPEX,    &
                     x(i), y(i), z(i)
               endif
            endif
         enddo

         if( icount.ne.nrep_EPR*nspinlabels)then
            write(OUTU,'(a)') ' Problem with nreplica and nspinlabels'
            CALL WRNDIE(-1,'<CHARMM>','EPR0 dimension mismatch.')
         endif

         icount = 0
         do i=1,nrep_EPR
            do j=1,nspinlabels
               icount = icount + 1
               spinlabel(i,j)=list(icount)
            enddo
         enddo

         if(qverb)then
            write(*,*) 'write the full list of spinlabels', nspinlabels
            do j=1,nspinlabels
               call ATOMID(spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX)
               write(*,*) j,spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX
            enddo
         endif

!        Read the experimental distance histograms
!        -----------------------------------------

         write(outu,'(a,i4)') ' Reading experimental data from unit ',iunit
         write(outu,'(a,i5)') ' Number of data point ',nppp 
         write(outu,'(a,f12.3)') ' Distance increment ',delppp 
         write(outu,'(a,i5)')    ' Number of spin label pairs ',nptot
      
         do icount=1,nptot
            read(iunit,*) i1, j1
            write(outu,'(a,2i5)')     &
               ' Experimental P(r) for residue pair: ', i1, j1

            do i=1,nspinlabels
               call ATOMID(spinlabel(1,i),SEGIDX,RESIDX,RESNX,TYPEX)
               read(RESIDX,*) i2
!              write(*,*) i1,i2,spinlabel(1,i),SEGIDX,RESIDX,RESNX,TYPEX
               if(i1.eq.i2) then
                  spinlb1(icount)=i
!                 write(*,*) 'found it'
                  write(outu,'(2i9,1x,4a)') i,spinlabel(1,i),    &
                                SEGIDX, RESIDX, RESNX, TYPEX
                  goto 1000
               endif
            enddo
            write(outu,*) 'spinlabel not found'

 1000       do j=1,nspinlabels
               call ATOMID(spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX)
               read(RESIDX,*) j2
!              write(*,*) j1,j2,spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX
               if(j1.eq.j2) then
                  spinlb2(icount)=j
!                 write(*,*) 'found it'
                  write(outu,'(2i9,1x,4a)') j,spinlabel(1,j),    &
                                SEGIDX, RESIDX, RESNX, TYPEX
                  goto 2000
               endif 
            enddo
            write(outu,*) 'spinlabel not found'

 2000       write(outu,'(i5,a,2i5)') icount, ' Spin label pair: ',    &
                                spinlb1(icount),spinlb2(icount)
            write(outu,*)

            sum  = 0.0      
            do n=1,nppp
               read(iunit,*) rr,pexp(n,icount)
               sum = sum + pexp(n,icount)
               if( prnlev.ge.9 ) write(outu,'(2f12.4,2f15.8)')    &
                               rr,n*delppp,pexp(n,icount),sum
               if(abs(rr-n*delppp).gt.TENM5)then  
                  write(OUTU,'(a)') ' Problem with r-axis of histogram '
                  CALL WRNDIE(-1,'<CHARMM>','EPR0 axis mismatch.')
               endif
            enddo

            if(abs(sum-1.0).gt.0.05)then  
               write(OUTU,'(a)') ' Problem with histogram'
               CALL WRNDIE(-1,'<CHARMM>','EPR0 normalization problem.')
            endif

         enddo     
      
!        read translation vectors for each system
         write(outu,*) 'Reading the translation vectors ',qtvec
         if(qtvec) then
            do i=1,nspinlabels
               read(iunit,*) ii, (tvec(i,j), j=1,3)
               if(qverb)then
                  write(outu,'(i5,3f12.3)')  ii, (tvec(i,j), j=1,3)
               endif
            enddo
         endif

!        do icount=1,nptot
!        write(*,*) icount,spinlb1(icount),spinlb2(icount)
!        enddo
      ENDIF

#if KEY_PARALLEL==1
      CALL PSND4m(spinlabel,nrep_EPR*nspinlabels) ! this is for warning!!!???
      CALL PSND4(spinlb1,nptot)
      CALL PSND4(spinlb2,nptot)
      CALL PSND8m(pexp,nppp*nptot)
      IF(qtvec) CALL PSND8m(tvec,3*nspinlabels)
#endif


!     test call to the energy restraint subroutine
!     --------------------------------------------

      if(prnlev.ge.9)then
         IF(MYNOD.EQ.0) write(outu,'(a)') ' Coordinates'
         do j=1,nspinlabels
         do i=1,nrep_EPR
            i1 = spinlabel(i,j)
            if(qtvec)then
               IF(MYNOD.EQ.0) write(outu,'(3i9,3f10.3,4x,3f10.3)')         &
                  j,i,i1, x(i1), y(i1), z(i1), tvec(j,1),tvec(j,2),tvec(j,3)
            else
               IF(MYNOD.EQ.0) write(outu,'(3i9,3f10.3,4x,3f10.3)')         &
                  j,i,i1, x(i1), y(i1), z(i1)
            endif
         enddo
         enddo
      endif

      do i=1,natom
         dx(i) = 0.0
         dy(i) = 0.0
         dz(i) = 0.0
      enddo

      call EPR002(x,y,z,dx,dy,dz,enpp,kenpp,    &
           nrep_EPR, nspinlabels, spinlabel,    &
           nptot, spinlb1, spinlb2,             &
           pcalc,pexp,nppp,delppp,sig2ppp,dist,qverb,   &
           qtvec,tvec)

      if(prnlev.ge.9)then
#if KEY_PARALLEL==1
         CALL GCOMB(dx,natom)
         CALL GCOMB(dy,natom)
         CALL GCOMB(dz,natom)
#endif

         IF(MYNOD.EQ.0) write(outu,'(a)') ' Output of first derivatives'
         sum = 0.0
         do j=1,nspinlabels
         do i=1,nrep_EPR
            i1 = spinlabel(i,j)
            IF(MYNOD.EQ.0) write(outu,'(2i9,3f12.6)') i,i1, dx(i1), dy(i1), dz(i1)
            sum = dx(i1)**2+dy(i1)**2+dz(i1)**2
         enddo
         enddo
         sum = sqrt(sum/(nrep_EPR*nspinlabels))
         IF(MYNOD.EQ.0) write(outu,'(a,f15.5)') ' DRMS       ',sum
      endif

!     Output the current distance histograms
!     --------------------------------------

      if(qverb)then
         IF(MYNOD.EQ.0) write(outu,'(a)') ' # EPR Distance Distribution'
         sum  = 0.0      
         do icount=1,nptot
            i=spinlb1(icount)
            j=spinlb2(icount)
            call ATOMID(spinlabel(1,i),SEGIDX,RESIDX,RESNX,TYPEX)
            IF(MYNOD.EQ.0) write(outu,'(2i9,1x,4a)') i,spinlabel(1,i),            &
                                   SEGIDX, RESIDX, RESNX, TYPEX
            call ATOMID(spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX)
            IF(MYNOD.EQ.0) write(outu,'(2i9,1x,4a)') j,spinlabel(1,j),            &
                                   SEGIDX, RESIDX, RESNX, TYPEX
            sum = 0.0
            do n=1,nppp
               sum = sum + pcalc(n,icount)
               IF(MYNOD.EQ.0) write(outu,'(f12.4,3f15.8)') n*delppp, pcalc(n,icount),   &
                                            sum, pexp(n,icount)
            enddo
            IF(MYNOD.EQ.0) write(outu,*)
         enddo
      endif

      IF(MYNOD.EQ.0) write(outu,*)

      return
      END SUBROUTINE EPR001

      SUBROUTINE EPR002(x,y,z,dx,dy,dz,enpp,kenpp,     &
           nrep_EPR, nspinlabels, spinlabel,           &
           nptot, spinlb1, spinlb2,                    &
           pcalc,pexp,nppp,delppp,sig2ppp,dist,qverb,  &
           qtvec,tvec)
      use stream
      use chutil,only:atomid
      use parallel

      real*8 x(*),y(*),z(*),dx(*),dy(*),dz(*)
      real*8 enpp, kenpp
      integer nrep_EPR, nspinlabels
      integer spinlabel(nrep_EPR,nspinlabels)
      integer nptot, spinlb1(nptot), spinlb2(nptot)
      real*8 pexp(nppp,nptot)
      real*8 pcalc(nppp,nptot)
      integer nppp
      real*8 delppp, sig2ppp
      real*8 dist(nrep_EPR,nrep_EPR)
      logical qverb, qtvec
      real*8 tvec(nspinlabels,3)
      real*8 tvec1(3), tvec2(3)

!     local
      integer i, j, n, ip, i1
      character*6 SEGIDX, RESIDX, RESNX, TYPEX

      enpp = 0.0 

      do ip=1, nptot
         i=spinlb1(ip)
         j=spinlb2(ip)

         if(qverb)then
            IF(MYNOD.EQ.0) write(outu,'(a,i9)') ' Calling EPR003 with pair',ip 
            call ATOMID(spinlabel(1,i),SEGIDX,RESIDX,RESNX,TYPEX)
            IF(MYNOD.EQ.0) write(outu,'(2i9,1x,4a)') i,spinlabel(1,i),            &
                                   SEGIDX, RESIDX, RESNX, TYPEX
            call ATOMID(spinlabel(1,j),SEGIDX,RESIDX,RESNX,TYPEX)
            IF(MYNOD.EQ.0) write(outu,'(2i9,1x,4a)') j,spinlabel(1,j),            &
                                   SEGIDX, RESIDX, RESNX, TYPEX
         endif
     
         if(qtvec) then
            tvec1(1) = tvec(i,1)
            tvec1(2) = tvec(i,2)
            tvec1(3) = tvec(i,3)
            tvec2(1) = tvec(j,1)
            tvec2(2) = tvec(j,2)
            tvec2(3) = tvec(j,3)
         endif
     
         call EPR003(x,y,z,dx,dy,dz,enpp,kenpp,                     &
              nspinlabels, nrep_EPR,spinlabel(1,i),spinlabel(1,j),  &
              pcalc(1,ip),pexp(1,ip),nppp,delppp,sig2ppp,           &
              dist,qverb,qtvec, tvec1, tvec2)
     
         if(qverb .AND. (MYNOD.EQ.0)) write(outu,'(a,f15.5)') ' ENER       ', enpp

      enddo
      
      if(qverb .AND. (MYNOD.EQ.0)) write(outu,'(a,f15.5)') ' ENER TOT:  ', enpp

      return
      END SUBROUTINE EPR002 

      SUBROUTINE EPR003(x,y,z,dx,dy,dz,enpp,kenpp,            &
                 nspinlabels,nrep_EPR,spinlabel1,spinlabel2,  &
                 pcalc,pexp,nppp,del,sig2,d,qverb,            &
                 qtvec,tvec1, tvec2) 
      use stream
      use parallel
!
!     EPR000 = Electron Paramagnetic Resonance Pair Probability distribution energy restraint
!     (first implemented in charmm 36a2)
!
      real*8 x(*),y(*),z(*)
      real*8 dx(*),dy(*),dz(*)
      integer nspinlabels, nrep_EPR
      integer spinlabel1(*), spinlabel2(*)
      real*8 pcalc(*), pexp(*)
      integer nppp
      real*8 del, sig2, enpp, kenpp
      real*8 d(nrep_EPR,nrep_EPR)
      logical qverb, qtvec
      real*8 tvec1(3), tvec2(3), Inv_sig2, Half_Inv_sig2


!     local
      real*8 avedist,avedist2
      real*8 fact,dxij,dyij,dzij,r2,dist,sum,dpn,Fnij,dFnij
      integer npair,n,i,j,i1,j1

      Inv_sig2 = 1.0 / sig2
      Half_Inv_sig2 = 0.5 * Inv_sig2
      npair = nrep_EPR*nrep_EPR
      fact = 1.0d0/(sqrt(2*3.1415*sig2))/npair

      do n=1,nppp
         pcalc(n) = 0.0
      enddo

      avedist = 0.0
      avedist2 = 0.0
      
      do i1=1,nrep_EPR
         i = spinlabel1(i1)      
         do j1=1,nrep_EPR
            j = spinlabel2(j1)      
            if(qtvec)then
!              write(*,'(i9,6f10.3)') 
!    &             i,x(i),y(i),z(i),tvec1(1),tvec1(2),tvec1(3)
!              write(*,'(i9,6f10.3)') 
!    &             j,x(j),y(j),z(j),tvec2(1),tvec2(2),tvec2(3)
               dxij = x(i)-x(j) -tvec1(1)+tvec2(1)
               dyij = y(i)-y(j) -tvec1(2)+tvec2(2)
               dzij = z(i)-z(j) -tvec1(3)+tvec2(3)
            else
               dxij = x(i)-x(j) 
               dyij = y(i)-y(j) 
               dzij = z(i)-z(j) 
            endif
            r2 = dxij**2 + dyij**2 + dzij**2
            dist = sqrt(r2)
!           write(*,*) i,j,'dist',dist
            d(i1,j1) = dist
            avedist = avedist + dist
            avedist2 = avedist2 + dist**2
            
            do n=MYNOD+1,nppp,NUMNOD
               pcalc(n) = pcalc(n) + exp(-(n*del-dist)**2*Half_Inv_sig2)
            enddo
         enddo
      enddo
      
      do n=MYNOD+1,nppp,NUMNOD
         pcalc(n) = fact*pcalc(n)
      ENDDO

#if KEY_PARALLEL==1 && KEY_GCMC!=1
      CALL GCOMB(pcalc,nppp)
#endif

      avedist = avedist/npair 
      avedist2 = avedist2/npair 
      avedist2 = sqrt(avedist2-avedist**2)

      if(qverb .AND. (MYNOD.EQ.0)) write(outu,*) 'Average distance: ',    &
                avedist, avedist2 

      IF(MYNOD .EQ. 0) THEN
         sum  = 0.0      
         do n=1,nppp
            sum = sum + pcalc(n)
!           IF(MYNOD.EQ.0) write(*,*) n,n*del, pcalc(n), sum, pexp(n)
            enpp = enpp + 0.5*kenpp*(pcalc(n)-pexp(n))**2
         enddo

         if(sum.lt.0.95)then
            IF(MYNOD.EQ.0) write(*,*) i,j,'normalization problem',sum
            sum  = 0.0      
            do n=1,nppp
               sum = sum + pcalc(n)
               IF(MYNOD.EQ.0) write(*,*) n,n*del, pcalc(n), sum, pexp(n)
            enddo
         endif
      ENDIF


!     First derivatives
      do n=MYNOD+1,nppp,NUMNOD
         dpn = pcalc(n)-pexp(n)

         do i1=1,nrep_EPR
            i = spinlabel1(i1)      
            do j1=1,nrep_EPR
               j = spinlabel2(j1)      

               Fnij  = fact*exp(-(n*del-d(i1,j1))**2*Half_Inv_sig2)
               dFnij = Fnij * (n*del-d(i1,j1))*Inv_sig2

               if(qtvec)then
                  dxij = (x(i)-x(j) -tvec1(1)+tvec2(1))/d(i1,j1)
                  dyij = (y(i)-y(j) -tvec1(2)+tvec2(2))/d(i1,j1)
                  dzij = (z(i)-z(j) -tvec1(3)+tvec2(3))/d(i1,j1)
               else
                  dxij = (x(i)-x(j))/d(i1,j1)
                  dyij = (y(i)-y(j))/d(i1,j1)
                  dzij = (z(i)-z(j))/d(i1,j1)
               endif

               dx(i) = dx(i) + kenpp*dpn*dFnij*dxij
               dx(j) = dx(j) - kenpp*dpn*dFnij*dxij

               dy(i) = dy(i) + kenpp*dpn*dFnij*dyij
               dy(j) = dy(j) - kenpp*dpn*dFnij*dyij

               dz(i) = dz(i) + kenpp*dpn*dFnij*dzij
               dz(j) = dz(j) - kenpp*dpn*dFnij*dzij

            enddo
         enddo
      enddo

      RETURN
      END SUBROUTINE EPR003


end module mmfp

