module scpismm
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="scpism.src"

  !---mfc--- Needs allocation routines

#if KEY_SCPISM==1 /*scpism_fcm*/
  !
  !     common block for the SCPISM
  !
  LOGICAL SCPISM,HYPBIC,INTESCP,LFORM
  INTEGER UISM
  CHARACTER(len=2),allocatable,dimension(:) :: HBFLAG
  real(chm_real) DCERO,DSOLT,DSUM,DSUMI,COEFFK,OFF2,OFF2PH,ccm,ccmph

  real(chm_real),dimension(:),allocatable :: ALFFAS,SQALFA,GPOLAR,SRFTNS, &
       ACM,BCM,DCM,DCMPH,JRI,ATRADW,PARNTE,RIPNP,SANDRA,SANDPH

#endif /* (scpism_fcm)*/

contains

#if KEY_SCPISM==1 /*scpism_main*/


  subroutine allocate_scpism(natom)
    use memory
    integer :: natom
    character(len=*),parameter :: routine_name="allocate_scpism"
    call chmalloc(file_name,routine_name,'hbflag ',natom,ch2=hbflag)
    call chmalloc(file_name,routine_name,'ALFFAS ',natom,crl=ALFFAS)
    call chmalloc(file_name,routine_name,'SQALFA ',natom,crl=SQALFA)
    call chmalloc(file_name,routine_name,'GPOLAR ',natom,crl=GPOLAR)
    call chmalloc(file_name,routine_name,'SRFTNS ',natom,crl=SRFTNS)
    call chmalloc(file_name,routine_name,'ACM    ',natom,crl=ACM   )
    call chmalloc(file_name,routine_name,'BCM    ',natom,crl=BCM   )
    call chmalloc(file_name,routine_name,'DCM    ',natom,crl=DCM   )
    call chmalloc(file_name,routine_name,'DCMPH  ',natom,crl=DCMPH )
    call chmalloc(file_name,routine_name,'JRI    ',natom,crl=JRI   )
    call chmalloc(file_name,routine_name,'ATRADW ',natom,crl=ATRADW)
    call chmalloc(file_name,routine_name,'PARNTE ',natom,crl=PARNTE)
    call chmalloc(file_name,routine_name,'RIPNP  ',natom,crl=RIPNP )
    call chmalloc(file_name,routine_name,'SANDRA ',natom,crl=SANDRA)
    call chmalloc(file_name,routine_name,'SANDPH ',natom,crl=SANDPH)

    return
  end subroutine allocate_scpism




  subroutine deallocate_scpism(natom)
    use memory
    integer :: natom
    character(len=*),parameter :: routine_name="deallocate_scpism"
    call chmdealloc(file_name,routine_name,'hbflag ',natom,ch2=hbflag)
    call chmdealloc(file_name,routine_name,'ALFFAS ',natom,crl=ALFFAS)
    call chmdealloc(file_name,routine_name,'SQALFA ',natom,crl=SQALFA)
    call chmdealloc(file_name,routine_name,'GPOLAR ',natom,crl=GPOLAR)
    call chmdealloc(file_name,routine_name,'SRFTNS ',natom,crl=SRFTNS)
    call chmdealloc(file_name,routine_name,'ACM    ',natom,crl=ACM   )
    call chmdealloc(file_name,routine_name,'BCM    ',natom,crl=BCM   )
    call chmdealloc(file_name,routine_name,'DCM    ',natom,crl=DCM   )
    call chmdealloc(file_name,routine_name,'DCMPH  ',natom,crl=DCMPH )
    call chmdealloc(file_name,routine_name,'JRI    ',natom,crl=JRI   )
    call chmdealloc(file_name,routine_name,'ATRADW ',natom,crl=ATRADW)
    call chmdealloc(file_name,routine_name,'PARNTE ',natom,crl=PARNTE)
    call chmdealloc(file_name,routine_name,'RIPNP  ',natom,crl=RIPNP )
    call chmdealloc(file_name,routine_name,'SANDRA ',natom,crl=SANDRA)
    call chmdealloc(file_name,routine_name,'SANDPH ',natom,crl=SANDPH)

    return
  end subroutine deallocate_scpism



  SUBROUTINE SCPPARSE
    !
    !     this routine parses commands of the SCP continuum model
    !     and initializes a number of variables 
    !
    !     S. A. Hassan, December 2002
    !     Main Refs: 
    !     [1] S A Hassan, F Guarnieri and E L Mehler; J Phys Chem 104, 6478 (2000)
    !     [2] S A Hassan, F Guarnieri and E L Mehler; J Phys Chem 104, 6490 (2000)
    !     [3] S A Hassan and E L Mehler; PROTEINS 47, 45 (2002)
    !     [4] S A Hassan, E L Mehler, D Zhang and H Weinstein; PROTEINS 51, 109 (2003)
    !
    !     Peter J. Steinbach, August 2006.
    !     Code optimized to run 30% faster and use less memory.
    !     Large 2D arrays no longer needed.
    !     scpism efficiency imrpoved: 1.5 times more expensive than vacuum (>6000 atoms) SAH
    !
    !     Parallelization improved substantially (Milan Hodoscek, 2011)
    !  
    !     Benchmark for ATPase (>15000 atoms, 100 steps dynamics) using 32 core AMD box:
    !
    !CPUs                scpism (charmm 35)      |       scpism (charmm 36)
    !               time (sec)   speedup   eff   |   time (sec)   speedup   eff
    ! 1               61.7         1.00   100%   |   61.8           1.00   100%
    ! 2               43.5         1.42    71%   |   33.0           1.87    94%
    ! 4               58.4         1.06    26%   |   17.0           3.63    91%
    ! 8              105.4         0.59     7%   |    9.0           6.87    86%
    !16              218.4         0.28     2%   |    5.1          12.12    76%
    !32              453.4         0.14     0%   |    3.3          18.73    59%             
    !    
    !
    use chm_kinds
    use dimens_fcm
    use comand
    use exfunc
    use stream
    use string
    use number
    use psf
    !--mfc--   use scpismm

    implicit none
    !
    CHARACTER(len=4) WISM
    !
    UISM=GTRMI(COMLYN,COMLEN,'UISM',-1)
    LFORM=INDXA(COMLYN,COMLEN,'LFORM')>0
    WISM=NEXTA4(COMLYN,COMLEN)
    !
    IF (WISM(:3) .EQ. 'END') THEN
       SCPISM=.FALSE.
    ELSE IF (WISM .EQ. 'HYDR') THEN
       SCPISM=.TRUE.
       HYPBIC=.TRUE.
    ELSE
       SCPISM=.TRUE.
       HYPBIC=.FALSE.
    ENDIF

    if(scpism) then
       ! Assume if scpism us true here we allocate scipsm arrays. clb3
       if(.not.allocated(hbflag)) call allocate_scpism(natom)
       !
    else
       ! Assume we turn off so we can deallocate. clb3
       if(allocated(hbflag)) call deallocate_scpism(natom)
       !
    endif

    IF (PRNLEV .GT. 2) THEN
       IF (.NOT. SCPISM) THEN
          WRITE(OUTU,'(A)') ' SCPISM> SCP-ISM has been turned off'
          WRITE(OUTU,'(A)') &
               ' SCPISM> default electrostatics will be used unless otherwise specified'
       ELSE IF (SCPISM) THEN 
          WRITE(OUTU,'(A)') ' SCPISM> SCP-ISM has been requested'
          WRITE(OUTU,'(A)') ' SCPISM> Shifting electrostatics used'
          IF (UISM < 0) CALL WRNDIE(-5,'<SCPISM>', &
               'No unit number for reading SCP-ISM parameters')
          IF (.NOT. HYPBIC) THEN
             WRITE(OUTU,'(A)') ' SCPISM> No Hydrophobic term in use'
          ELSE IF (HYPBIC) THEN
             WRITE(OUTU,'(A)') ' SCPISM> Hydrophobic term requested'
          ENDIF
       ENDIF
    ENDIF
    !
    IF (SCPISM) CALL INITSCPISM
    !
    RETURN
  END SUBROUTINE SCPPARSE
  !
  !-----------------------------------------------------------------------
  !
  SUBROUTINE INITSCPISM
    !
    !     this initializes variables for the run
    !-----------------------------------------------------------------------
    !
    use chm_kinds
    use consta
    use dimens_fcm
    use exfunc
    use number
    use param
    use fast
    use rtf, only: atct
    use psf
    use parallel,only: psnd4,psnd8
#if KEY_MULTICOM==1 /*  VO to work with stringm */
    use parallel         
#endif
    use stream
    !--   use scpismm
    implicit none
    !
    integer,PARAMETER :: MXSCPAT=270
    INTEGER I,J,IBOND,ITYPE
    LOGICAL PRONTO
    CHARACTER(len=2) :: SCPLX8(MXSCPAT)
    CHARACTER(len=8) :: SCPLX1(MXSCPAT),ISID,IRID,IREN,IAT
    real(chm_real) GCONST,RPROBE,R,RIW,AREAI,JUNK,GNONP,RICOV
    real(chm_real) SCPLX2(MXSCPAT),SCPLX3(MXSCPAT)
    real(chm_real) SCPLX4(MXSCPAT),SCPLX5(MXSCPAT)
    real(chm_real) SCPLX6(MXSCPAT),SCPLX7(MXSCPAT)
    real(chm_real) SCPLX9(MXSCPAT),SCPLXA(MXSCPAT)
    real(chm_real) SCPLXB(MXSCPAT),SCPLXC(MXSCPAT)
    real(chm_real), PARAMETER :: FOURPI = FOUR*PI
    !
    !     default parameters
    !
    DCERO=ONE
    DSOLT=80.0D0
    RPROBE=1.4D0
    DSUM=DCERO+DSOLT
    DSUMI=ONE/DSUM
    COEFFK=(DSOLT-ONE)/(DCERO+ONE)
    !
    !     Read the SCP parameters from a file.
    !     SCPLX7 is the atomic surface tension
    !
    I=0
    PRONTO=.FALSE.
#if KEY_MULTICOM==0 /*  VO stringm */
    IF(IOLEV .GT. 0) THEN
#else
    IF(MYNOD .eq. 0) THEN
#endif
       DO WHILE (.NOT.PRONTO)
          I=I+1
          IF(LFORM)THEN
          READ(UISM,668) SCPLX1(I),SCPLX2(I),SCPLX3(I), &
               SCPLX4(I),SCPLX5(I),SCPLX6(I), &
               SCPLX7(I),SCPLX8(I)
          ELSE
          READ(UISM,667) SCPLX1(I),SCPLX2(I),SCPLX3(I), &
               SCPLX4(I),SCPLX5(I),SCPLX6(I), &
               SCPLX7(I),SCPLX8(I)
          ENDIF
          IF ((SCPLX1(I)(1:3).EQ.'end') .OR.  &
               (SCPLX1(I)(1:3).EQ.'END')) PRONTO=.TRUE.
       ENDDO
667    FORMAT(A4,6(1X,F7.4),1X,A2)
668    FORMAT(A8,6(1X,F7.4),1X,A2)
       !
       REWIND(UISM)
    ENDIF
#if KEY_PARALLEL==1
#if KEY_MULTICOM==1 /*  VO could have a string of 1-cpu systems */
    if (numnod.gt.1) then      
#endif
    CALL PSND4(I,1)
    CALL PSNDC(SCPLX1,I)
    CALL PSND8(SCPLX2,I)
    CALL PSND8(SCPLX3,I)
    CALL PSND8(SCPLX4,I)
    CALL PSND8(SCPLX5,I)
    CALL PSND8(SCPLX6,I)
    CALL PSND8(SCPLX7,I)
    CALL PSNDC(SCPLX8,I)
#if KEY_MULTICOM==1
    endif                      
#endif
#endif 
    !
    !     For each atom, assign parameters according to its atom type
    !
    loop981: DO I=1,NATOM
       !
       R = VDWR(ITC(IAC(I)))+RPROBE
       AREAI = FOURPI*R*R
       PRONTO=.FALSE.
       J=0
       DO WHILE (.NOT.PRONTO)
          J=J+1
          IF (SCPLX1(J) .EQ. ATCT(IAC(I))) THEN
             PRONTO=.TRUE.
             ALFFAS(I)=SCPLX2(J)
             GPOLAR(I)=SCPLX3(J)
             GNONP    =SCPLX4(J)
             SQALFA(I)=SCPLX5(J)
             RICOV    =SCPLX6(J)
             SRFTNS(I)=SCPLX7(J)
             HBFLAG(I)=SCPLX8(J)
             !
             IF (ALFFAS(I) .EQ. ZERO) THEN
                IF(PRNLEV.GT.2) WRITE(OUTU,810) I,ATCT(IAC(I))
810             FORMAT(' SCPISM> No SCPISM parameters found for', &
                     ' atom',I6,' of type ',A8,'!')
                CALL WRNDIE(-5,'<SCPISM>','PARAMETERS MISSING******')
             ENDIF
             !
             !     See PROTEINS 2003, top of page 112 for discussion of GCONST.
             !                           
             IF (CG(I).GT.ZERO) GCONST=0.85D0
             IF (CG(I).LE.ZERO) GCONST=0.35D0
             !                         
             RIW=RICOV+GCONST
             RIPNP(I)=RIW+GNONP
             JRI(I)=(RIW-RIPNP(I))/AREAI
             !
          ELSE IF (SCPLX1(J).EQ.'end' .OR. SCPLX1(J).EQ.'END') THEN
             IF(PRNLEV.GT.2) WRITE(OUTU,810) I,ATCT(IAC(I))
             CALL WRNDIE(-5,'<SCPISM>','PARAMETERS MISSING******')
          ENDIF
       ENDDO
       IF (ATYPE(I)(1:1) == 'C') THEN
          ACM(I)=31.00D0 
          BCM(I)=7.92D0
          DCM(I)=0.32D0
       ELSE IF (ATYPE(I)(1:1) == 'O') THEN
          ACM(I)=40.0D0
          BCM(I)=8.52D0
          DCM(I)=0.29D0
       ELSE IF (ATYPE(I)(1:1) == 'N') THEN
          ACM(I)=60.00D0
          BCM(I)=9.66D0
          DCM(I)=0.22D0
       ELSE IF (ATYPE(I)(1:1) == 'S') THEN
          ACM(I)=60.0D0
          BCM(I)=9.10D0
          DCM(I)=0.22D0
       ELSE IF (ATYPE(I)(1:1) == 'H') THEN
          IF (HBFLAG(I)(1:2) .EQ. 'PH') THEN
             ACM(I)=1.20D0
             BCM(I)=3.68D0
             DCM(I)=0.80D0
             DCMPH(I)=0.80D0
          ELSE IF (HBFLAG(I)(1:2) .NE. 'PH') THEN
             ACM(I)=17.00D0
             BCM(I)=9.00D0
             DCM(I)=0.50D0
          ENDIF
       ENDIF
    enddo loop981
    CCM=0.04D0
    OFF2=ONE/CCM
    CCMPH=0.04D0
    OFF2PH=ONE/CCMPH
    !
    RETURN
  ENd SUBROUTINE INITSCPISM
  !
  !
  !-----------------------------------------------------------------
  SUBROUTINE SCPEVDWF(ENB,EEL,LELECX,LVDWX,IFRSTA,NATOMX, &
       CGX,JNBL,INBL, &
       IACNB,NITCC2,LOWTP, &
#if KEY_BLOCK==1
       IBLOCK,BLCOE,BLCOV,BLCOVR,BLCOVA, &    
#endif
       LUSED)
    !----------------------------------------------------------------------
    !     this subroutine calculates the nonbond energy terms
    !     for continuum electrostatics based on SCP {shifting} and
    !     vdW {any cutoff option supported}
    !
    !     December 2002, S. A. Hassan
    !
    !     definitions follow mainly PROTEINS 51, 109 (2003)
    !     EELONLY: interaction component of the electrostatic energy
    !              (first sum in the rhs of Eq.(1))
    !     SELFEN: self-energy component of the electrostatic energy
    !             (second sum in the rhs of Eq.(1))
    !     ENPOLR: non-polar energy of the ISM (proportional to an approximate 
    !             SASA based on a contact model)
    !     HBFLAG: flag for acceptor, polar hydrogen (col.8 in parameters file)
    !     RIW: Riw in Eq.(2)
    !     RIPNP: Rip in Eq.(2)
    !     SCRN: Screening function Ds(r) in Eq.(1), defined explicitely in
    !           Eq.(11)
    !     SANDRA: sum in the contact model for Born radii [see Eq.(3)]
    !     GPOLAR,GNONP  The "g" extensions to Born radii in H-bonds (see papers)
    !     ALFFA: parameter alpha that defines the screening function Ds(r_ij)
    !            (these are the alpha_ij in Eq.(11), product of values
    !            in col.5 in parameters file)
    !     ALFFAS: parameter alpha (col.2 in parameters file) that defines
    !             the screening function Ds(R), where R is the Born radius
    !     ATRADW: Born radius R [see Eqs.(4) and (6)]; note that the
    !             approximation of Eq.(6) has been slightly modified for
    !             practical reasons and efficiency
    !-----------------------------------------------------------------------

    use nb_module            ! has ccnba thru d
    use chm_kinds
    use consta
    use dimens_fcm
    use exfunc
    use number
    use coord
    use deriv
    use param
    use inbnd
    use psf
    !--   use scpismm
    use parallel
    use stream
    use chutil,only:getres
    implicit none
    real(chm_real)  ENB,EEL
    LOGICAL LELECX,LVDWX
    INTEGER NATOMX,JNBL(*),INBL(*)
    real(chm_real) CGX(*)
    INTEGER IACNB(*),NITCC2,LOWTP(*),IFRSTA
    LOGICAL LUSED
#if KEY_BLOCK==1 /* fix on charmm.org */
    INTEGER IBLOCK(*)
    real(chm_real) BLCOE(*),BLCOV(*),BLCOVR(*),BLCOVA(*)
#endif 
    !
    !
    INTEGER IVECT,JVECT,KVECT
    real(chm_real) CA,CC,CH,ENE,ENN
    real(chm_real) TF,TX,TY,TZ,DTX,DTY,DTZ
    real(chm_real) S2,TR2,TR6,FSW,DFSW
    real(chm_real) ON3,ON6,OFF3,OFF6,R1,R3,ENEVDW
    real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
         CR6,CR12,RJUNK3,RJUNK6
    !
    real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
    real(chm_real) CGF,HAFCGF,CGT,CRXI,CRYI,CRZI
    INTEGER ITEMP,I,J,NPR,IACI,IHEAVY,JHEAVY,IHP1,JHP1
    LOGICAL ELECFG,LOUTER,LVSW,LVSH,LVFSW
    !
    real(chm_real) SELFEN,EELONLY,ENPOLR,JUNK
    INTEGER IS
    real(chm_real) ALFFA,SCRNRI,SCRN,R1SCRN,CH1,CH2,DERVDI,CHENER
    real(chm_real) DISTR,SCRNP1,SHFUNC,TFFCOMP,TFF,SRCTFI,SRCPHI, &
         BCMPHI
    INTEGER JJCC,JJCCMX,LXY
    !

    !---------- Sanity check -------------------------------------
    if(.not. allocated(ccnba))then
       ! How we got here without vdw table filled, who knows?
       call wrndie(-4,"scpevdwf<misc/scpism.src>", &
            "CCNBA not allocated")
    endif

    LUSED=.TRUE.
    ENB=ZERO
    EEL=ZERO 
    SELFEN=ZERO
    ENPOLR=ZERO
    CGF=ZERO
    ELECFG=LELECX
    IF (.NOT.(LVDWX.OR.ELECFG)) RETURN
    IF (ELECFG) CGF=CCELEC
    HAFCGF = HALF*CGF
    !
    LVFSW=       LVFSWT                    .AND. LVDWX
    LVSH = .NOT. LVFSWT .AND.       LVSHFT .AND. LVDWX
    LVSW = .NOT. LVFSWT .AND. .NOT. LVSHFT .AND. LVDWX
    !
    C2OFNB=CTOFNB*CTOFNB
    C2ONNB=CTONNB*CTONNB
    CTROF2=-ONE/C2OFNB
    C4ROF2=FOUR*CTROF2
    !
    IF(LVSW .AND. CTOFNB.GT.CTONNB) THEN
       RUL3=ONE/(C2OFNB-C2ONNB)**3
       RUL12=TWELVE*RUL3
    ELSE IF (LVFSW) THEN
       OFF3=C2OFNB*CTOFNB
       OFF6=OFF3*OFF3
       RECOF6=ONE/OFF6
       IF (CTONNB .LT. CTOFNB) THEN
          ON3=C2ONNB*CTONNB
          ON6=ON3*ON3
          RECOF3=ONE/OFF3
          OFDIF6=OFF6/(OFF6-ON6)
          OFDIF3=OFF3/(OFF3-ON3)
          ONOFF6=RECOF6/ON6
          ONOFF3=RECOF3/ON3
       ELSE
          ONOFF6=RECOF6*RECOF6
          ONOFF3=RECOF6
       END IF
    END IF
    !
    DO I=1,NATOMX
       SANDRA(I)=ZERO
       SANDPH(I)=ZERO
    ENDDO
    !
    !    This first I,J loop calculates some of the SASA-related quantities needed for
    !    the self-energy and nonpolar terms; it is not needed if only the interaction
    !    term is to be computed (if INTESCP=T).  
    !
    JJCC=0
    IF(.NOT. INTESCP) THEN
       loop60: DO I=IFRSTA,NATOMX
          !
          IF (I.GT.1) THEN
             ITEMP=INBL(I-1)
             NPR=INBL(I)-ITEMP
          ELSE
             NPR=INBL(I)
             ITEMP=0
          ENDIF
          IF (NPR /= 0) then      !  GOTO 55
             !
             !    Note: 1 <= I <~ NATOM-3, since NPR=0 for the last few atoms.
             !
             CGT=CGF*CGX(I)
             CRXI=X(I)
             CRYI=Y(I)
             CRZI=Z(I)
             !
             !    In this "DO 30" loop, compute r_I,JVECT contribution to SANDRA(I)
             !    and to SANDRA(JVECT); note JVECT > I.
             !    Check if I is polar hydrogen 'PH' and JVECT is 'PA'.  Also,
             !    check if JVECT is 'PH' and I is 'PA'.
             !
             loop30: DO J=1,NPR
                KVECT=JNBL(ITEMP+J)
                JVECT=ABS(KVECT)
                TX=CRXI-X(JVECT)
                TY=CRYI-Y(JVECT)
                TZ=CRZI-Z(JVECT)
                S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
                IF(S2.LT.C2OFNB) THEN
                   CH=CGT*CGX(JVECT)
                   IF(ELECFG) THEN
                      IF(S2 .LE. OFF2) THEN
                         DISTR=SQRT(S2)
                         SRCTFI=ONE-CCM*S2
                         JUNK = SRCTFI*SRCTFI
                         SANDRA(I) = SANDRA(I) + JUNK*EXP(-DCM(I)*DISTR)
                         SANDRA(JVECT) = SANDRA(JVECT) &
                              + JUNK*EXP(-DCM(JVECT)*DISTR)
                         IF(HBFLAG(I)(1:2) .EQ. 'PH' .AND.  &
                              HBFLAG(JVECT)(1:2) .EQ. 'PA') THEN
                            IF(S2 .LE. OFF2PH) THEN 
                               IF(GETRES(I,IBASE,NREST) .NE.  &
                                    GETRES(JVECT,IBASE,NREST)) THEN
                                  ! bb hb
                                  IF (ATYPE(I) == 'HN  ' .AND. &
                                       ATYPE(JVECT) == 'O   ') THEN
                                     BCMPHI=-0.378*GPOLAR(JVECT)
                                  ELSE
                                     BCMPHI=GPOLAR(I)*GPOLAR(JVECT)
                                  ENDIF
                                  SRCPHI=ONE-CCMPH*S2
                                  SANDPH(I)=SANDPH(I)+BCMPHI*SRCPHI* &
                                       SRCPHI*EXP(-DCMPH(I)*DISTR)
                               ENDIF
                            ENDIF
                         ELSE IF(HBFLAG(JVECT)(1:2) .EQ. 'PH' .AND.  &
                              HBFLAG(I)(1:2) .EQ. 'PA') THEN
                            IF(S2 .LE. OFF2PH) THEN 
                               IF(GETRES(I,IBASE,NREST) .NE.  &
                                    GETRES(JVECT,IBASE,NREST)) THEN
                                  ! bb hb
                                  IF (ATYPE(JVECT) == 'HN  ' .AND. &
                                       ATYPE(I) == 'O   ') THEN
                                     BCMPHI=-0.378*GPOLAR(I)
                                  ELSE
                                     BCMPHI=GPOLAR(JVECT)*GPOLAR(I)
                                  ENDIF
                                  SRCPHI=ONE-CCMPH*S2
                                  SANDPH(JVECT)=SANDPH(JVECT)+BCMPHI*SRCPHI* &
                                       SRCPHI*EXP(-DCMPH(JVECT)*DISTR)
                               ENDIF
                            ENDIF
                         ENDIF
                      ENDIF
                   ENDIF
                ENDIF
             enddo loop30
          endif
          !
          IF (ELECFG) THEN
             !
             !   For all atoms except polar hydrogens, the Born radius is:
             !   ATRADW = Rprot + (Rwat - Rprot)/4piR2 (ACM - BCM SANDRA)
             !   (Eq 4, PROTEINS 2003)
             !
             SANDRA(I) = BCM(I)*SANDRA(I)
          ENDIF
          !
          ITEMP=INBL(I)
       enddo loop60
       !
    ENDIF
! New communication:
#if KEY_PARALLEL==1
    call gcomb(sandra,natom)
    call gcomb(sandph,natom)   ! we could squeeze this one in sandra, maybe ????
#endif 
    ! we need to perform some extra summations for parallel
    ! this could go into next loop, maybe !!!!????
    do i = ifrsta,natomx
       ATRADW(I) = RIPNP(I) + JRI(I)*(ACM(I)-sandra(I))
       !
       !Modify Born radius for polar hydrogens (approximation to Eq 7, PROTEINS 2003):
       !
       IF(HBFLAG(I)(1:2) .EQ. 'PH') THEN
          ATRADW(I)=ATRADW(I)+sandPH(I)
          IF(ATRADW(I) .LE. ZERO) ATRADW(I)=ATRADW(I)-sandPH(I)
       ENDIF
       !
       !   Eself = 1/2 Sum_i qq/ATRADW [1/D(ATRADW) - 1].  SCRN = D(ATRADW).
       !   DERVDI = dD/dATRADW(I), Eq 12 of PROTEINS 2003.
       !
       SCRN=DSUM/(ONE+COEFFK*EXP(-ALFFAS(I)*ATRADW(I))) - DCERO
       SCRNRI=ONE/(SCRN*ATRADW(I))
       JUNK = MINONE/ATRADW(I)
       CHENER=HAFCGF*CGX(I)*CGX(I)*(SCRNRI+JUNK)
       SELFEN=SELFEN+CHENER
       SCRNP1 = SCRN + ONE
       DERVDI = ALFFAS(I)*DSUMI*SCRNP1*(DSUM-SCRNP1)
       PARNTE(I)=CHENER*(JUNK+DERVDI/(SCRN*(SCRN-ONE)))
    enddo
#if KEY_PARALLEL==1
    if(mynod > 0) selfen=zero                         
#endif
    !
    !   Calculation of self energy is done.
    !   Calculate energy and forces for interaction and nonpolar terms.
    !
    loop6011: do I=IFRSTA,NATOMX
       ! cavity+vdW term
       IF (HYPBIC .AND. .NOT.INTESCP) THEN
          ENPOLR = ENPOLR + SRFTNS(I)*(ACM(I)-SANDRA(I))
#if KEY_PARALLEL==1
          if(mynod > 0) enpolr=zero                       
#endif
       ENDIF
       !
       IF (I.GT.1) THEN
          ITEMP=INBL(I-1)
          NPR=INBL(I)-ITEMP
       ELSE
          NPR=INBL(I)
          ITEMP=0
       ENDIF
       IF (NPR.EQ.0) GOTO 5511 
       IACI=IACNB(I)
       CGT=CGF*CGX(I)
       CRXI=X(I)
       CRYI=Y(I)
       CRZI=Z(I)
       DTX=ZERO
       DTY=ZERO
       DTZ=ZERO
       DO 3011 J=1,NPR
          KVECT=JNBL(ITEMP+J)
          JVECT=ABS(KVECT)
          TX=CRXI-X(JVECT)
          TY=CRYI-Y(JVECT)
          TZ=CRZI-Z(JVECT)
          !
          S2=MAX(RSMALL,TX*TX+TY*TY+TZ*TZ)
          IF (S2.LT.C2OFNB) THEN
             LOUTER=(S2.GT.C2ONNB)
             IVECT=LOWTP(MAX(IACNB(JVECT),IACI))+IACNB(JVECT)+IACI 
             CH=CGT*CGX(JVECT)
             IF (KVECT.LT.0) THEN
                CH=CH*E14FAC
                IVECT=IVECT+NITCC2
             ENDIF
             !
             DISTR = SQRT(S2)
             R1 =ONE/DISTR
             TR2=R1*R1
             TR6=TR2*TR2*TR2
             !
             ! Interaction component of electrostatic energy (only for nonzero charges).
             ! Screened Coulomb Potential with shift cutoff: ENE = Sum(i<j) SHFUNC qi qj / (Dr)
             !
             IF(ELECFG .AND. CH.NE.ZERO) THEN
                ALFFA=SQALFA(I)*SQALFA(JVECT)
                SCRN = DSUM/(ONE+COEFFK*EXP(-ALFFA*DISTR)) - DCERO
                R1SCRN=R1/SCRN
                CH1=CH*R1SCRN
                CH2=S2*CTROF2
                SHFUNC=ONE+CH2*(TWO+CH2)
                ENE=CH1*SHFUNC
                SCRNP1=SCRN+ONE
                TFFCOMP=TR2+SCRNP1*(DSUM-SCRNP1)*R1SCRN*ALFFA*DSUMI
                TFF=-CH1*TFFCOMP
                TF=SHFUNC*TFF+CH1*(C4ROF2*(ONE+CH2))
             ELSE
                ENE=ZERO
                TF=ZERO
             ENDIF
             !
             ! van der Waals interaction energy and forces
             ! vdw shift
             IF (LVSH) THEN
                CA=CCNBA(IVECT)*TR6*TR6
                ENEVDW=CA-CCNBB(IVECT)*TR6
                CC=S2*S2*S2*CCNBC(IVECT)
                ENN=ENEVDW-CC+CCNBD(IVECT)
                TF=TF+MINSIX*(ENEVDW+CA+CC)*TR2
                ! vdw force switch
             ELSE IF (LVFSW) THEN
                IF (LOUTER) THEN
                   IF (.NOT.LCONS .OR. CH.EQ.ZERO) R1=SQRT(TR2)
                   R3=R1*TR2
                   RJUNK6=TR6-RECOF6
                   RJUNK3=R3-RECOF3
                   CR12=CCNBA(IVECT)*OFDIF6*RJUNK6
                   CR6=CCNBB(IVECT)*OFDIF3*RJUNK3
                   ENN=CR12*RJUNK6-CR6*RJUNK3
                   TF=TF+TR2*(SIX*CR6*R3-TWELVE*CR12*TR6)
                ELSE
                   CA=CCNBA(IVECT)*TR6*TR6
                   ENEVDW=CA-CCNBB(IVECT)*TR6
                   ENN=ENEVDW+CCNBB(IVECT)*ONOFF3-CCNBA(IVECT)*ONOFF6
                   TF=TF+MINSIX*TR2*(ENEVDW+CA)
                ENDIF
                ! vdw switch
             ELSE IF (LVSW) THEN
                CA=CCNBA(IVECT)*TR6*TR6
                IF (LOUTER) THEN
                   RIJL=C2ONNB-S2
                   RIJU=C2OFNB-S2
                   FSW=RIJU*RIJU*(RIJU-THREE*RIJL)*RUL3
                   DFSW=RIJL*RIJU*RUL12
                   ENEVDW=CA-CCNBB(IVECT)*TR6
                   ENN=ENEVDW*FSW
                   TF=TF+ENEVDW*DFSW-SIX*TR2*(ENN+CA*FSW)
                ELSE
                   ENN=CA-CCNBB(IVECT)*TR6
                   TF=TF+MINSIX*TR2*(ENN+CA)
                ENDIF
                ! no vdw
             ELSE
                ENN=ZERO
             ENDIF
             !
             !    ENB = sum of Lennard-Jones.  EEL = sum of SCP interaction term.
             !
             ENB=ENB+ENN
2011         EEL=EEL+ENE
             !
             !    To this point, TF = sum of 1/r dV/dr for interaction terms (SCP and vdW).
             !    Add force due to self-energies, and if HYPBIC, add cavity-term force.
             !
             IF (ELECFG .AND. .NOT.INTESCP .AND. S2.LE.OFF2) &
                  CALL SCDERIVS(I,JVECT,DISTR,R1,S2,OFF2,TF)
             !
             TX=TX*TF
             TY=TY*TF
             TZ=TZ*TF
             DTX=DTX+TX
             DTY=DTY+TY
             DTZ=DTZ+TZ
             !       
             DX(JVECT)=DX(JVECT)-TX
             DY(JVECT)=DY(JVECT)-TY
             DZ(JVECT)=DZ(JVECT)-TZ
          ENDIF
3011   ENDDO
       !
       DX(I)=DX(I)+DTX
       DY(I)=DY(I)+DTY
       DZ(I)=DZ(I)+DTZ
5511   CONTINUE
       ITEMP=INBL(I) 
    enddo loop6011
    !
    IF (ELECFG) THEN
       EELONLY=EEL
       IF (INTESCP) THEN
          EEL=EELONLY
       ELSE
          EEL=EELONLY+SELFEN+ENPOLR
       END IF
       INTESCP=.FALSE. 
    ENDIF
    !
    RETURN
  END SUBROUTINE SCPEVDWF
  !
  !
  ! ------------------------------------------------------------------------
  SUBROUTINE SCDERIVS(I,J,DISTR,R1,S2,CMOFF2,TF)
    !
    !    Adds forces due to self-energies, and cavity term (if HYPBIC).
    !
    use chm_kinds
    use dimens_fcm
    use number
    use exfunc
    use psf
    use chutil,only:getres
    implicit none
    !
    INTEGER I,J
    real(chm_real) BCEXI,BCEXJ
    real(chm_real) FSELFE,FSELFEXI,FSELFEXJ,TF,DISTR,R1,S2,CMOFF2, &
         BIXI,BIXJ,FOURR,CCM4R
    real(chm_real) SRCTFI,SRCPHI,SRCPHJ,BCMPHI,BCMPHJ,CC
    real(chm_real) FHYDR
    !
    FOURR = FOUR*DISTR
    CCM4R = CCM*FOURR
    SRCTFI = ONE-CCM*S2
    BIXI = BCM(I)*SRCTFI*EXP(-DCM(I)*DISTR)* &
         (DCM(I)*SRCTFI+CCM4R)
    BIXJ = BCM(J)*SRCTFI*EXP(-DCM(J)*DISTR)* &
         (DCM(J)*SRCTFI+CCM4R)
    BCEXI = BIXI*JRI(I)
    BCEXJ = BIXJ*JRI(J)
    !
    IF(HBFLAG(I)(1:2).EQ.'PH' .AND. HBFLAG(J)(1:2).EQ.'PA') THEN
       IF(S2 .LE. OFF2PH) THEN
          IF(GETRES(I,IBASE,NREST) .NE. GETRES(J,IBASE,NREST)) THEN
             ! bb hb
             IF (ATYPE(I) == 'HN  ' .AND. ATYPE(J) == 'O   ') THEN
                BCMPHI=-0.378*GPOLAR(J)
             ELSE
                BCMPHI=GPOLAR(I)*GPOLAR(J)
             ENDIF
             SRCPHI=ONE-CCMPH*S2
             BCEXI=BCEXI-BCMPHI*SRCPHI*EXP(-DCMPH(I)*DISTR)* &
                  (DCM(I)*SRCPHI+FOURR*CCMPH)
          ENDIF
       ENDIF
    ENDIF
    !
    IF(HBFLAG(J)(1:2).EQ.'PH' .AND. HBFLAG(I)(1:2).EQ.'PA') THEN
       IF(S2 .LE. OFF2PH) THEN
          IF(GETRES(I,IBASE,NREST) .NE. GETRES(J,IBASE,NREST)) THEN
             ! bb hb
             IF(ATYPE(J) == 'HN  ' .AND. ATYPE(I) == 'O   ') THEN
                BCMPHJ=-0.378*GPOLAR(I)
             ELSE
                BCMPHJ=GPOLAR(I)*GPOLAR(J)
             ENDIF
             SRCPHJ=ONE-CCMPH*S2
             BCEXJ=BCEXJ-BCMPHJ*SRCPHJ*EXP(-DCMPH(J)*DISTR)* &
                  (DCM(J)*SRCPHJ+FOURR*CCMPH)
          ENDIF
       ENDIF
    ENDIF
    !
    FSELFEXI=PARNTE(I)*BCEXI
    FSELFEXJ=PARNTE(J)*BCEXJ
    FSELFE=FSELFEXI+FSELFEXJ 
    TF=TF+FSELFE*R1
    !                            Cavity-term force
    IF(HYPBIC) THEN
       FHYDR=SRFTNS(I)*BIXI + SRFTNS(J)*BIXJ
       TF=TF+FHYDR*R1
    ENDIF
    RETURN
  END SUBROUTINE SCDERIVS
#else /* (scpism_main)*/
  SUBROUTINE SCPPARSE
    CALL WRNDIE(-1,'<SCPPARSE>','SCPISM code is not compiled.')
    return
  end SUBROUTINE SCPPARSE
#endif /* (scpism_main)*/

end module scpismm

