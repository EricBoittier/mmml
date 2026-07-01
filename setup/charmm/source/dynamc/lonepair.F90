module lonepr
  use chm_kinds
  use dimens_fcm
  implicit none

  character(len=*),private,parameter :: file_name   ="lonepair.src"

#if KEY_LONEPAIR==1 /*lonepair_fcm*/
  !
  !     The Lone-Pair structure data
  !
  !     Purpose:
  !
  !     To store the lists needed to define and effect the handling of
  !     Lone Pairs in CHARMM.  This information is an extension of PSF
  !     data and is written and read and patched with the PSF.
  !
  !     I/O: Unformatted and formatted I/O and print: io/psfres.src
  !
  !     Notes:    The index entry gives the domain of the data.
  !
  !     Variable  Index    Purpose
  !
  !     NUMLP             Number of lone-pairs
  !     NUMLPH            Number of LP hosts in the host table 
  !     LPNHOST   lp      Number of lone-pair hosts
  !     LPVALUE  (3,lp)   Lone pair types and values:
  !           0 hosts -  fixed dummy position (but scales with CPT box)
  !           1 host  -  colocate (only) (values unused)
  !           2 hosts -  colinear (sum of following specs)
  !                 val1 - absolute dist from h1 away from h2
  !                 val2 - relative dist from h1 away from h2
  !                          (value of -0.5 is geometric center)
  !           3 hosts - relative (val1>0)
  !                 val1 - distance from h1
  !                 val2 - angle (lp-h1-h2) (in degrees)
  !                 val3 - dihedral (lp-h1-h2-h3) (in degrees)
  !           3 hosts - bisector h2-h3 (val1<0)
  !                 val1 - negative distance from h1
  !                 val2 - angle (lp-h1-bisector) (in degrees)
  !                 val3 - dihedral (lp-h1-bisector-h2) (in degrees)
  !           3 hosts - center position (val1=0) (other values unused)
  !           4 or more hosts - center position (values unused)
  !     LPHPTR    lp      Lone-pair host pointer
  !     LPWGHT    lp      Weighting option for force and center calcs.
  !                        .false. - use center of geometry (no weights)
  !                        .true.  - use mass weighting (center of mass)
  !     LPHOST    lpptr   Lone-pair host indicies
  !
  !     MAXLP      Maximum number of lone-pair atoms
  !     MAXLPH     Maximum number of lone-pair hosts
  !
  REAL(chm_real),save,DIMENSION(:,:), ALLOCATABLE :: LKATMS

  ! integers
  INTEGER NUMLP,NUMLPH
  integer,allocatable,dimension(:) :: LPNHOST,LPHPTR,LPHOST

  ! logicals
  LOGICAL,allocatable,dimension(:) :: LPWGHT

  ! reals
  real(chm_real),allocatable,dimension(:,:) ::LPVALUE

  private :: LONEPRS2

#endif /* (lonepair_fcm)*/

contains

#if KEY_LONEPAIR==1 /*lonepair_main*/
  subroutine lonepair_init
    numlp=0
    numlph=0
    return
  end subroutine lonepair_init

  subroutine allocate_lonepair()
    use memory
    character(len=*),parameter :: routine_name="allocate_lonepair"
    call chmalloc(file_name,routine_name,'lpnhost',maxlp,intg=lpnhost)
    call chmalloc(file_name,routine_name,'lphptr',maxlp,intg=lphptr)
    call chmalloc(file_name,routine_name,'lphost',MAXLPH,intg=lphost)
    call chmalloc(file_name,routine_name,'lpwght',maxlp, log=lpwght)
    call chmalloc(file_name,routine_name,'lpvalue',3,maxlp,crl=lpvalue)
    return
  end subroutine allocate_lonepair

  SUBROUTINE LONEPRS(COMLYN,COMLEN)
    !
    ! This routine parses the lone-pair command which converts existing
    ! atoms to lone-pairs in the PSF.
    !
    ! Syntax:
    ! 
    ! LONEpair { FIXEd   atom-spec   [ xloc-yloc-zloc ]          } [MASS]
    !          { CENTer  atom-spec  {  atom-selection   }        }
    !          {                    { repeat(atom-spec) }        }
    !          {                                                 }
    !          { COLOcate { 2x(atom-selection) }                 } 
    !          {          { 2x(atom-spec)      }                 } 
    !          {                                                 }
    !          { { COLInear distance-spec } { 3x(atom-selection) }
    !          { { CEN2                   } { 3x(atom-spec)      }
    !          {                                                 }
    !          { { RELAtive } { 4x(atom-selection) position-spec }
    !          { { BISEctor } { 4x(atom-selection)               }
    !          { { CEN3     }                                    }
    !          {                                                 }
    !          { CLEAr                                           }
    !          { PRINt                                           }
    ! 
    ! atom-spec::= { residue-number atom-name }
    !              { segid  resid atom-name   }
    !              { BYNUm  atom-number       }
    ! 
    ! atom-selection ::= see *note select:(chmdoc/select.doc).
    ! 
    ! xloc-yloc-zloc::= three real numbers identifying the new position
    ! 
    ! distance-spec::= [DISTance real] [SCAled real]
    !                    (def 0.0)       (def 0.0)
    ! 
    ! postition-spec::= [DISTance real] [ANGLe real] [DIHEdral real]
    !                      (def 0.0)      (def 0.0)    (def 0.0)
    ! 
    !
    !                     Bernard R. Brooks, NIH, October, 1997
    !
    use memory
    use psf
    use shake
    use param_store, only: set_param

    INTEGER COMLEN
    character(len=*) COMLYN
    !
    if(.not. allocated(lpnhost)) call allocate_lonepair 
    ! Save the current counts
    CALL LONEPRS2(COMLYN,COMLEN,NATOM)
    !
    IF(NUMLP > 0) QHOLO=.TRUE.
    !

    RETURN
  END SUBROUTINE LONEPRS

  SUBROUTINE LONEPRS2(COMLYN,COMLEN,NAT)
    use number
    use coord
    use psf
    use stream
    use select
    use string
    use param_store, only: set_param 

    INTEGER COMLEN,NAT
    character(len=*) COMLYN
    INTEGER ISLCT(NAT,4) ! changed to automatic array - cb3
    !
    INTEGER IPAa(1),ipa,CURPT,ISEL
    INTEGER NSEL(4)
    character(len=4) WRD
    LOGICAL QMASS,QWEIGH,OK,ERROR,QPRINT
    real(chm_real)  RSCAL,RDIST,RANGL,RDIHE,SETMAS
    real(chm_real)  WA,WB,SA
    INTEGER QSELE,NIPA,I,J,K,N,LPSTRT,ILP,OLDNLP,OLDNLPH,NWARN
    !
    OLDNLP =NUMLP
    OLDNLPH=NUMLPH
    ERROR=.FALSE.
    !
    !-----------------------------------------------------------------------
    ! Parse the atom selections (up to 4 sets)
    QSELE=0
    DO ISEL=1,4 
       IF(INDX(COMLYN,COMLEN,'SELE',4) > 0) THEN
          QSELE=ISEL
          CALL SELCTA(COMLYN,COMLEN,ISLCT(1,ISEL),X,Y,Z,WMAIN,.TRUE.)
          NSEL(ISEL)=NSELCT(NATOM,ISLCT(1,ISEL))
       ENDIF
    ENDDO
    !-----------------------------------------------------------------------
    ! Clear the facility
    WRD=NEXTA4(COMLYN,COMLEN)
    IF(WRD == 'CLEA') THEN
       DO ILP=1,NUMLP
          I=LPHOST(LPHPTR(ILP))
          IMOVE(I)=0
       ENDDO
       NUMLP=0
       NUMLPH=0
       IF(PRNLEV > 3) WRITE(OUTU,25) 'Lone-pair facility is cleared'
25     FORMAT('LONEPR: ',A)
       CALL set_param('NUMLP',NUMLP)
       !
       NWARN=0
       DO I=1,NATOM
          IF(IMOVE(I) == -1) THEN
             IMOVE(I)=0
             NWARN=NWARN+1
          ENDIF
       ENDDO
       IF(NWARN > 0 .AND. WRNLEV.GT.2) WRITE(OUTU,26) NWARN
26     FORMAT('LONEPR:  Number of unassigned lonepairs reset:',I5)
       NWARN=0
       !
       SETMAS=GTRMF(COMLYN,COMLEN,'MASS',ZERO)
       IF(SETMAS > ZERO) THEN
          DO I=1,NATOM
             IF(AMASS(I) <= ZERO) AMASS(I)=SETMAS
          ENDDO
       ELSE
          DO I=1,NATOM
             IF(AMASS(I) <= ZERO) NWARN=NWARN+1
          ENDDO
       ENDIF
       IF(NWARN > 0 .AND. WRNLEV.GT.2) WRITE(OUTU,27) NWARN
27     FORMAT(' *****  WARNING  ***** Number of atoms with', &
            ' zero mass that are no longer lonepairs:',I5)
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    ! Print the current lonepair information
    IF(WRD == 'PRIN') THEN
       IF(PRNLEV > 2) THEN
          WRITE(OUTU,46) NUMLP,NUMLPH
46        FORMAT('LONEPR:  Number of lone-pairs:',I5, &
               '  Number of host pointers:',I5)
          DO ILP=1,NUMLP

             WRITE(OUTU,47) ILP,LPNHOST(ILP),LPHPTR(ILP)
47           FORMAT(I5,5X,' Number of hosts:',I5,'  Host pointer:',I5)
             WRITE(OUTU,48) (LPHOST(LPHPTR(ILP)+K),K=0,LPNHOST(ILP))
48           FORMAT(10X,'Atom pointers:',20I5)
             WRD='GEOM'
             IF(LPWGHT(ILP)) WRD='MASS'
             WRITE(OUTU,49) WRD,(LPVALUE(K,ILP),K=1,3)
49           FORMAT(10X,'Weight option: ',A4,'  Values:',3F12.5)
          ENDDO
          WRITE(OUTU,50) NSELCTV(NATOM,IMOVE,-1)
50        FORMAT(10X,'Number of atoms flagged as lonepairs:',I8)
          CALL PRNTATSL(IMOVE,NATOM,-1,RESID,RES,IBASE,ATYPE, &
               SEGID,NICTOT,NSEG)
       ENDIF
       RETURN
    ENDIF
    !-----------------------------------------------------------------------
    !
    QMASS = (INDXA(COMLYN,COMLEN,'MASS') > 0)
    QPRINT= (INDXA(COMLYN,COMLEN,'PRIN') > 0)
    !
    ! Check limits and data set integrity
    IF(NUMLP >= MAXLP) THEN
       CALL WRNDIE(-3,'<LONEPRS>', &
            'Maximum number of lone-pair atoms exceeded')
       RETURN
    ENDIF
    !
    IF(NUMLP > 0) THEN
       IF(NUMLPH /= LPHPTR(NUMLP)+LPNHOST(NUMLP)) THEN
          CALL WRNDIE(-3,'<LONEPRS>', &
               'Atom pointer count mismatch - Check code')
          RETURN
       ENDIF
    ENDIF
    IF(NUMLPH >= MAXLPH) THEN
       CALL WRNDIE(-3,'<LONEPRS>', &
            'Maximum number of lone-pair hosts exceeded')
       RETURN
    ENDIF
    !
    CURPT=NUMLPH+1
    LPSTRT=NUMLP+1
    !
    !------------------------------------------------------------------
    ! The fixed atom or the the centered atom
    ! add just one lone-pair
    IF(WRD == 'FIXE' .OR. WRD.EQ.'CENT') THEN
       CALL NXTATM(IPAa,NIPA,1,COMLYN,COMLEN,ISLCT, &
            SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
       IF(NIPA == 0) THEN
          CALL WRNDIE(0,'<LONEPRS>', &
               'Bad lone-pair atom specification')
          RETURN
       ENDIF
       ipa=ipaa(1)
       !
       AMASS(IPA)=ZERO     ! no mass for lonepairs
       IF(IMOVE(IPA) < 0) THEN
          CALL WRNDIE(0,'<LONEPRS>', &
               'Selected lone-pair atom is already a lonepair')
          ERROR=.TRUE.
       ENDIF
       IMOVE(IPA)=-1       ! define it to be a lonepair

       icaauto(ipa)=ior(icaauto(ipa),15) ! no autogen for lonepairs

       !
       NUMLP=NUMLP+1
       LPNHOST(NUMLP)=0
       LPHPTR(NUMLP)=CURPT
       LPHOST(CURPT)=IPA
       LPWGHT(NUMLP)=QMASS
       NUMLPH=CURPT
       CURPT=NUMLPH+1
       DO I=1,3
          LPVALUE(I,NUMLP)=0.0
       ENDDO
       CALL TRIMA(COMLYN,COMLEN)
       !
       IF(WRD == 'FIXE') THEN
          IF(COMLEN > 0) THEN
             X(IPA)=NEXTF(COMLYN,COMLEN)
             Y(IPA)=NEXTF(COMLYN,COMLEN)
             Z(IPA)=NEXTF(COMLYN,COMLEN)
          ENDIF
       ELSE IF(WRD == 'CENT') THEN
          IF(QSELE == 0) THEN
             DO WHILE(COMLEN > 0)
                CALL NXTATM(IPAa,NIPA,1,COMLYN,COMLEN,ISLCT, &
                     SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
                IF(NIPA == 0) THEN
                   CALL WRNDIE(0,'<LONEPRS>', &
                        'Bad lone-pair host atom specification')
                   ERROR=.TRUE.
                   GOTO 900
                ENDIF
                ipa=ipaa(1)
                IF(IMOVE(IPA) < 0) THEN
                   CALL WRNDIE(0,'<LONEPRS>', &
                        'Selected host atom is already a lonepair')
                   ERROR=.TRUE.
                ENDIF
                LPHOST(CURPT)=IPA
                NUMLPH=NUMLPH+1
                IF(NUMLPH >= MAXLPH) THEN
                   CALL WRNDIE(-3,'<LONEPRS>', &
                        'Maximum number of lone-pair hosts exceeded')
                   NUMLP =OLDNLP
                   NUMLPH=OLDNLPH
                   RETURN
                ENDIF
                CURPT=CURPT+1
                LPNHOST(NUMLP)=LPNHOST(NUMLP)+1
                CALL TRIMA(COMLYN,COMLEN)
             ENDDO
          ELSE
             NUMLPH=NUMLPH+NSEL(1)
             IF(NUMLPH >= MAXLPH) THEN
                CALL WRNDIE(-3,'<LONEPRS>', &
                     'Maximum number of lone-pair hosts exceeded')
                NUMLP =OLDNLP
                NUMLPH=OLDNLPH
                RETURN
             ENDIF
             LPNHOST(NUMLP)=LPNHOST(NUMLP)+NSEL(1)
             DO IPA=1,NATOM
                IF(ISLCT(IPA,1) == 1) THEN
                   IF(IMOVE(IPA) < 0) THEN
                      CALL WRNDIE(0,'<LONEPRS>', &
                           'Selected host atom is already a lonepair')
                      ERROR=.TRUE.
                   ENDIF
                   LPHOST(CURPT)=IPA
                   CURPT=CURPT+1
                ENDIF
             ENDDO
          ENDIF
          IF(LPNHOST(NUMLP) == 0) THEN
             CALL WRNDIE(0,'<LONEPRS>', &
                  'Null center list')
             ERROR=.TRUE.
          ELSE IF(LPNHOST(NUMLP) == 1) THEN
             CALL WRNDIE(0,'<LONEPRS>', &
                  'Center spec for only one atom (use COLOcate)')
             ERROR=.TRUE.
          ENDIF
       ENDIF
       GOTO 900
    ENDIF
    !------------------------------------------------------------------
    ! parse all remaining values.
    IF(WRD == 'COLI') THEN
       RDIST=GTRMF(COMLYN,COMLEN,'DIST',ZERO)
       RSCAL=GTRMF(COMLYN,COMLEN,'SCAL',ZERO)
    ENDIF
    IF(WRD == 'RELA' .OR. WRD.EQ.'BISE') THEN
       RDIST=ABS(GTRMF(COMLYN,COMLEN,'DIST',ZERO))
       RANGL=GTRMF(COMLYN,COMLEN,'ANGL',ZERO)
       RDIHE=GTRMF(COMLYN,COMLEN,'DIHE',ZERO)
    ENDIF
    !------------------------------------------------------------------
    ! handle case where atom-spec is used (instead of atom selections).
    IF(QSELE == 0) THEN
       ! add just one lone-pair (no atom selections)
       CALL TRIMA(COMLYN,COMLEN)
       ISEL=0
       DO WHILE(COMLEN > 0)
          CALL NXTATM(IPAa,NIPA,1,COMLYN,COMLEN,ISLCT, &
               SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
          CALL TRIMA(COMLYN,COMLEN)
          IF(NIPA == 0) THEN
             CALL WRNDIE(0,'<LONEPRS>', &
                  'Bad lone-pair atom host specification')
             RETURN
          ENDIF
          ipa=ipaa(1)
          ISEL=ISEL+1
          IF(ISEL > 4) THEN
             CALL WRNDIE(0,'<LONEPRS>', &
                  'Too many atoms parsed')
          ENDIF
          NSEL(ISEL)=1
          DO I=1,NATOM
             ISLCT(I,ISEL)=0
          ENDDO
          ISLCT(IPA,ISEL)=1
       ENDDO
       QSELE=ISEL
       IF(QSELE == 0) THEN
          CALL WRNDIE(0,'<LONEPRS>','No atoms found in parsing')
          ERROR=.TRUE.
       ENDIF
    ENDIF
    !------------------------------------------------------------------
    ! add a set of lone-pairs
    !        first check to see if everything is OK.
    N=NSEL(1)
    OK=.FALSE.
    IF(WRD == 'COLO' .AND. QSELE.EQ.2) OK=.TRUE.
    IF(WRD == 'COLI' .AND. QSELE.EQ.3) OK=.TRUE.
    IF(WRD == 'CEN2' .AND. QSELE.EQ.3) OK=.TRUE.
    IF(WRD == 'RELA' .AND. QSELE.EQ.4) OK=.TRUE.
    IF(WRD == 'BISE' .AND. QSELE.EQ.4) OK=.TRUE.
    IF(WRD == 'CEN3' .AND. QSELE.EQ.4) OK=.TRUE.
    IF(.NOT.OK) THEN
       CALL WRNDIE(0,'<LONEPRS>', &
            'Wrong number of atom selections')
       ERROR=.TRUE.
       GOTO 900
    ENDIF
    DO ISEL=1,QSELE
       IF(N /= NSEL(ISEL)) OK=.FALSE.
    ENDDO
    IF(.NOT.OK) THEN
       CALL WRNDIE(0,'<LONEPRS>', &
            'Wrong number of atoms selected')
       ERROR=.TRUE.
       GOTO 900
    ENDIF
    !------------------------------------------------------------------
    IF(NUMLP+N >= MAXLP) THEN
       CALL WRNDIE(-3,'<LONEPRS>', &
            'Maximum number of lone-pair atoms exceeded')
       GOTO 900
    ENDIF
    IF(NUMLPH+N*QSELE >= MAXLPH) THEN
       CALL WRNDIE(-3,'<LONEPRS>', &
            'Maximum number of lone-pair hosts exceeded')
       GOTO 900
    ENDIF
    !
    NUMLP=NUMLP+N
    NUMLPH=NUMLPH+N*QSELE
    !
    OK=.TRUE.
    DO I=LPSTRT,NUMLP
       LPNHOST(I)=QSELE-1
       LPHPTR(I)=CURPT+(I-LPSTRT)*QSELE
       LPWGHT(I)=QMASS
       DO J=1,3
          LPVALUE(J,I)=0.0
       ENDDO
       IF(WRD == 'COLI') THEN
          LPVALUE(1,I)=RDIST
          LPVALUE(2,I)=RSCAL
          IF(RDIST == ZERO .AND. RSCAL.EQ.0) OK=.FALSE.
       ENDIF
       IF(WRD == 'RELA') THEN
          LPVALUE(1,I)=RDIST
          LPVALUE(2,I)=RANGL
          LPVALUE(3,I)=RDIHE
          IF(RDIST == ZERO) OK=.FALSE.
       ENDIF
       IF(WRD == 'BISE') THEN
          LPVALUE(1,I)=-RDIST
          LPVALUE(2,I)=RANGL
          LPVALUE(3,I)=RDIHE
          IF(RDIST == ZERO) OK=.FALSE.
       ENDIF
    ENDDO
    IF(.NOT.OK) THEN
       CALL WRNDIE(0,'<LONEPRS>', &
            'Bad distances specified')
       ERROR=.TRUE.
    ENDIF
    DO ISEL=1,QSELE
       NSEL(ISEL)=0
    ENDDO
    DO I=1,NATOM
       DO ISEL=1,QSELE
          IF(ISLCT(I,ISEL) == 1) THEN
             NSEL(ISEL)=NSEL(ISEL)+1
             LPHOST(CURPT+(NSEL(ISEL)-1)*QSELE+ISEL-1)=I
             IF(IMOVE(I) < 0) THEN
                IF(ISEL == 1) CALL WRNDIE(0,'<LONEPRS>', &
                     'Selected lone-pair atom is already a lonepair')
                IF(ISEL > 1) CALL WRNDIE(0,'<LONEPRS>', &
                     'Selected host atom is already a lonepair')
                ERROR=.TRUE.
             ENDIF
             IF(ISEL == 1) THEN
                AMASS(I)=ZERO
                IMOVE(I)=-1
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !------------------------------------------------------------------
900 CONTINUE
    CALL set_param('NUMLP',NUMLP)
    !------------------------------------------------------------------
    ! Print the current lonepair information
    IF(QPRINT) THEN
       IF(PRNLEV > 2) THEN
          WRITE(OUTU,46) NUMLP-LPSTRT+1,NUMLPH
          DO ILP=LPSTRT,NUMLP
             WRITE(OUTU,47) ILP,LPNHOST(ILP),LPHPTR(ILP)
             WRITE(OUTU,48) (LPHOST(LPHPTR(ILP)+K),K=0,LPNHOST(ILP))
             WRD='GEOM'
             IF(LPWGHT(ILP)) WRD='MASS'
             WRITE(OUTU,49) WRD,(LPVALUE(K,ILP),K=1,3)
          ENDDO
       ENDIF
    ENDIF
    !------------------------------------------------------------------
    IF(ERROR) THEN
       CALL WRNDIE(-3,'<LONEPRS>', &
            'Errors were encountered creating lonepairs - None created')
       NUMLP =OLDNLP
       NUMLPH=OLDNLPH
    ENDIF
    !------------------------------------------------------------------
    RETURN
  END SUBROUTINE LONEPRS2

  SUBROUTINE LONEPRD(MARK)
    !
    ! This routine deletes lone-pairs for marked atoms
    !
    ! Bernard R. Brooks   December, 1997
    !
  use psf
  use stream
    !
    !
    INTEGER MARK
    !
    INTEGER NDEL,NMISS,ILP,JLP,JLPH
    INTEGER I,N,IPT
    LOGICAL MISSING
    !
    ! Some lone-pair atom(s) have been deleted.
    ! Eliminate them from the data structure.
    !
    NDEL=0
    NMISS=0
    JLP=0
    JLPH=0
    !
    DO ILP=1,NUMLP
       N=LPNHOST(ILP)+1
       IPT=LPHPTR(ILP)-1
       I=LPHOST(IPT+1)

       IF(I == MARK) THEN
          NDEL=NDEL+1
       ELSE
          MISSING=.FALSE.
          DO I=1,N
             IF(LPHOST(I+IPT) == MARK) MISSING=.TRUE.
          ENDDO
          IF(MISSING) THEN
             IF(NMISS == 0) CALL WRNDIE(-1,'<LONEPRD>', &
                  'Lonepair atom host deleted, but not the lonepair')
             NMISS=NMISS+1
          ELSE
             ! keep this one
             JLP=JLP+1
             LPNHOST(JLP)=LPNHOST(ILP)
             LPHPTR(JLP)=JLPH+1
             LPWGHT(JLP)=LPWGHT(ILP)
             LPVALUE(1,JLP)=LPVALUE(1,ILP)
             LPVALUE(2,JLP)=LPVALUE(2,ILP)
             LPVALUE(3,JLP)=LPVALUE(3,ILP)
             DO I=1,N
                LPHOST(JLPH+I)=LPHOST(IPT+I)
             ENDDO
             JLPH=JLPH+N
          ENDIF
       ENDIF
    ENDDO
    !
    NUMLP=JLP
    NUMLPH=JLPH
    !
    IF(NDEL > 0 .AND. PRNLEV >= 2) WRITE(OUTU,22) &
         NDEL,'lonepair atoms'
    IF(NMISS > 0 .AND. PRNLEV >= 2) WRITE(OUTU,22) &
         NMISS,'lonepair atoms with missing hosts'
22  FORMAT(' LONEPRD:',I5,1X,A,' deleted')
    !
    RETURN
  END SUBROUTINE LONEPRD

  SUBROUTINE LONEPRC(ATFRST,ATLAST,X,Y,Z,XREF,YREF,ZREF,AMASS, &
       NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
    !
    ! This routine positions lone-pair atoms relative to their hosts.
    !
    ! Bernard R. Brooks   October, 1997
    !
    !  use lonepr,only: LKATMS
    use number
    use parallel
    use intcor2,only:cartcv
    use psf,only:natom
    use gamess_fcm,only:qmused_qchem,qmused_turbo,qmused_g09
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_lonepair,only:lonepr_comm_coord, nloneprlist, loneprlist
#endif
    !
    !     NOTE on parallelization of this:
    !            in the main DO loop check if (only in PARALLEL & numnod>1)
    !            I is within atfrst, atlast
    !            if no then skip this ILP
    !            if yes then check for each J,K,L if they are
    !            within the same limits (for drude they are in the same group?!)
    !
    real(chm_real) X(*),Y(*),Z(*),XREF(*),YREF(*),ZREF(*),AMASS(*)
    INTEGER NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    LOGICAL LPWGHT(*)
    real(chm_real) LPVALUE(3,*)
    !
    INTEGER ILP,N,I,J,K,L,IPT,ATFRST,ATLAST
    real(chm_real) TOL,V1,V2,V3,AM,MTOT,RX,RY,RZ,DR
    LOGICAL ERROR,OK,QCENT
    integer lp, lp_start, lp_end

    TOL=TENM5
    ERROR=.FALSE.
    ! KEY_QCHEM + friends now a default compile:
    if(qmused_qchem.or.qmused_turbo.or.qmused_g09) then
       IF (.not.allocated(LKATMS)) THEN 
          ALLOCATE(LKATMS(3,natom))
       ENDIF
       ! cnrAug22-2015
       ! check if size is incorrect for current number of atoms
       if(size(LKATMS,2).ne.NATOM) THEN
          DEALLOCATE(LKATMS)
          ALLOCATE(LKATMS(3,natom))
       ENDIF

       !hlw_080705
       DO J = 1, NATOM
          DO I = 1, 3
             LKATMS(I, J) = -10
          ENDDO
       ENDDO
       !hlw_080705
    endif

#if KEY_DOMDEC==1
    if (q_domdec) then
       !call wrndie(-5,'<lonepair>','loneprc not implemented for domdec')
       call lonepr_comm_coord(x, y, z)
       lp_start = 1
       lp_end = nloneprlist
    else
#endif
       lp_start = 1
       lp_end = numlp
#if KEY_DOMDEC==1
    endif
#endif

    !
    do lp=lp_end,lp_start,-1
       ilp = lp
#if KEY_DOMDEC==1
       if (q_domdec) then
          ilp = loneprlist(lp)
       endif
#endif
       N=LPNHOST(ILP)
       IPT=LPHPTR(ILP)
       I=LPHOST(IPT)
       !     For parallel do only lonepairs for valid forces & coordinates
       !     Works only if lonepairs within the group boundary.
#if KEY_PARALLEL==1
       IF((I < ATFRST).OR.(I > ATLAST)) GOTO 99    
#endif
       !
       QCENT=(N > 3)
       IF(N < 0) THEN
          ERROR=.TRUE.
       ELSE IF(N == 0) THEN
          ! fixed in fractional coords
          X(I)=XREF(I)
          Y(I)=YREF(I)
          Z(I)=ZREF(I)
       ELSE IF(N == 1) THEN
          ! simple colocate
          J=LPHOST(IPT+1)

#if KEY_PARALLEL==1
          IF((J < ATFRST).OR.(J > ATLAST))THEN
             CALL WRNDIE(-5,'<LONEPRC>', &
                  'In parallel lonepairs must be within group boundary!')
          ENDIF
#endif 
          X(I)=X(J)
          Y(I)=Y(J)
          Z(I)=Z(J)
       ELSE IF(N == 2) THEN
          ! colinear
          J=LPHOST(IPT+1)
          K=LPHOST(IPT+2)
#if KEY_PARALLEL==1
          IF((J < ATFRST).OR.(J > ATLAST) &
               .OR.(K < ATFRST).OR.(K > ATLAST))THEN
             CALL WRNDIE(-5,'<LONEPRC>', &
                  'In parallel lonepairs must be within group boundary!')
          ENDIF
#endif 
          RX=X(J)-X(K)
          RY=Y(J)-Y(K)
          RZ=Z(J)-Z(K)
          DR=SQRT(RX*RX+RY*RY+RZ*RZ)
          V1=LPVALUE(1,ILP)
          V2=LPVALUE(2,ILP)
          IF(V1 == ZERO .AND. V2.EQ.ZERO) THEN
             QCENT=.TRUE.
          ELSE IF(DR > TOL) THEN
             DR=V2+V1/DR
             !hlw_080705 yihan
             if(qmused_qchem.or.qmused_turbo.or.qmused_g09) then
                LKATMS(1,I) = J+0.01
                LKATMS(2,I) = K+0.01
                LKATMS(3,I) = DR
             endif
             !hlw_080705 end yihan
             X(I)=X(J)+DR*RX
             Y(I)=Y(J)+DR*RY
             Z(I)=Z(J)+DR*RZ
          ELSE
             ! colinear atoms at same position - error
             ERROR=.TRUE.
          ENDIF
       ELSE IF(N == 3) THEN
          ! relative positioning
          J=LPHOST(IPT+1)
          K=LPHOST(IPT+2)
          L=LPHOST(IPT+3)
#if KEY_PARALLEL==1
          IF((J < ATFRST).OR.(J > ATLAST) &
               .OR.(K < ATFRST).OR.(K > ATLAST) &
               .OR.(L < ATFRST).OR.(L > ATLAST))THEN
             CALL WRNDIE(-5,'<LONEPRC>', &
                  'In parallel lonepairs must be within group boundary!')
          ENDIF
#endif 
          V1=LPVALUE(1,ILP)
          V2=LPVALUE(2,ILP)
          V3=LPVALUE(3,ILP)
          IF(V1 == ZERO) THEN
             QCENT=.TRUE.
          ELSE IF(V1 > ZERO) THEN
             CALL CARTCV(X,Y,Z,L,K,J,I,V1,V2,V3,OK)
             IF(.NOT.OK) ERROR=.TRUE.
          ELSE
             V1=-V1
             X(I)=(X(K)+X(L))*HALF
             Y(I)=(Y(K)+Y(L))*HALF
             Z(I)=(Z(K)+Z(L))*HALF
             CALL CARTCV(X,Y,Z,L,I,J,I,V1,V2,V3,OK)
             IF(.NOT.OK) ERROR=.TRUE.
          ENDIF
       ENDIF
       IF(QCENT) THEN
          ! center of mass/geometry
          RX=ZERO
          RY=ZERO
          RZ=ZERO
          MTOT=ZERO
          DO K=1,N
             J=LPHOST(IPT+K)
             AM=ONE
             IF(LPWGHT(ILP)) AM=AMASS(J)
             MTOT=MTOT+AM
             RX=RX+X(J)*AM
             RY=RY+Y(J)*AM
             RZ=RZ+Z(J)*AM
          ENDDO
          X(I)=RX/MTOT
          Y(I)=RY/MTOT
          Z(I)=RZ/MTOT
       ENDIF
99     CONTINUE
    ENDDO
    !
    IF(ERROR) THEN
       CALL WRNDIE(0,'<LONEPRC>', &
            'Error found in positioning lone-pair')
    ENDIF
    !
    RETURN
  END SUBROUTINE LONEPRC

  SUBROUTINE LONEPRF(ATFRST,ATLAST,X,Y,Z,AMASS,DX,DY,DZ, &
       NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
    !
    ! This routine transforms lone-pair forces to its hosts.
    ! This is done in a manner which preserves net force, torque and virial.
    !
    ! Bernard R. Brooks   October, 1997
    !
    use number
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_lonepair,only:nloneprlist, loneprlist, lonepr_zero_force, lonepr_comm_force
#endif
    !
    INTEGER ATFRST,ATLAST
    real(chm_real) X(*),Y(*),Z(*),AMASS(*)
    real(chm_real) DX(*),DY(*),DZ(*)
    INTEGER NUMLP,LPNHOST(*),LPHPTR(*),LPHOST(*)
    LOGICAL LPWGHT(*)
    real(chm_real) LPVALUE(3,*)
    !
#if KEY_DEBUG==1
    real(chm_real) fneti(3),tneti(3),fnetf(3),tnetf(3)  
#endif
    !
    INTEGER ILP,N,I,J,K,L,IPT
    real(chm_real) RX,RY,RZ,SX,SY,SZ,TX,TY,TZ,AM,MTOT,DR,FACT,TOL, &
         V1,V2,V3
    LOGICAL OK,QBIS,QCENT,ERROR
    !
    real(chm_real) VX,VY,VZ,QX,QY,QZ,PX,PY,PZ
    real(chm_real) RRDOT,SRDOT,STDOT,TTDOT,FDOT,PDOT,QDOT,RDOT, &
         SDOT,TDOT,FPDOT
    real(chm_real) FPX,FPY,FPZ,FQX,FQY,FQZ,FRX,FRY,FRZ
    real(chm_real) FTX,FTY,FTZ,RRX,RRY,RRZ,TTX,TTY,TTZ
    integer lp, lp_start, lp_end
    !
    ERROR=.FALSE.
    TOL=RSMALL
    !
    ! process all of the lonepairs (in reverse order)
    !

#if KEY_DOMDEC==1
    if (q_domdec) then
       call lonepr_zero_force(3, 2, dx, dy, dz)
       !call wrndie(-5,'<lonepair>','loneprf not implemented for domdec')
       lp_start = 1
       lp_end = nloneprlist
    else
#endif
       lp_start = 1
       lp_end = numlp
#if KEY_DOMDEC==1
    endif
#endif

    !
    do lp=lp_end,lp_start,-1
       ilp = lp
#if KEY_DOMDEC==1
       if (q_domdec) then
          ilp = loneprlist(lp)
       endif
#endif
       N=LPNHOST(ILP)
       IPT=LPHPTR(ILP)
       I=LPHOST(IPT)
       !     For parallel do only lonepairs for valid forces & coordinates
       !     Works only if lonepairs within the group boundary
       IF((I < ATFRST).OR.(I > ATLAST)) GOTO 99

#if KEY_DEBUG==1 /*debug_a*/
       ! temp code for debug
       ! save net force and torque relative to the origin
       fneti(1)=zero
       fneti(2)=zero
       fneti(3)=zero
       tneti(1)=zero
       tneti(2)=zero
       tneti(3)=zero
       do k=0,n
          j=lphost(ipt+k)
          fneti(1)=fneti(1)+dx(j)
          fneti(2)=fneti(2)+dy(j)
          fneti(3)=fneti(3)+dz(j)
          tneti(1)=tneti(1)+dy(j)*z(j)-dz(j)*y(j)
          tneti(2)=tneti(2)+dz(j)*x(j)-dx(j)*z(j)
          tneti(3)=tneti(3)+dx(j)*y(j)-dy(j)*x(j)
       enddo
#endif /* (debug_a)*/

       QCENT=(N > 3)
       IF(N < 0) THEN
          ERROR=.TRUE.
       ELSE IF(N == 1) THEN
          ! simple colocate
          J=LPHOST(IPT+1)
          DX(J)=DX(J)+DX(I)
          DY(J)=DY(J)+DY(I)
          DZ(J)=DZ(J)+DZ(I)
       ELSE IF(N == 2) THEN
          ! colinear
          J=LPHOST(IPT+1)
          K=LPHOST(IPT+2)
          RX=X(J)-X(K)
          RY=Y(J)-Y(K)
          RZ=Z(J)-Z(K)
          DR=SQRT(RX*RX+RY*RY+RZ*RZ)
          V1=LPVALUE(1,ILP)
          V2=LPVALUE(2,ILP)
          IF(V1 == ZERO .AND. V2.EQ.ZERO) THEN
             QCENT=.TRUE.
          ELSE IF(DR > TOL) THEN
             V1=V1/DR
             FDOT=V1*(DX(I)*RX+DY(I)*RY+DZ(I)*RZ)/(DR*DR)
             DX(J)=DX(J) + (ONE+V2+V1)*DX(I) - RX*FDOT
             DY(J)=DY(J) + (ONE+V2+V1)*DY(I) - RY*FDOT
             DZ(J)=DZ(J) + (ONE+V2+V1)*DZ(I) - RZ*FDOT
             DX(K)=DX(K) - (V2+V1)*DX(I) + RX*FDOT
             DY(K)=DY(K) - (V2+V1)*DY(I) + RY*FDOT
             DZ(K)=DZ(K) - (V2+V1)*DZ(I) + RZ*FDOT
          ELSE
             ! colinear atoms at same position - error
             ERROR=.TRUE.
          ENDIF
       ELSE IF(N == 3) THEN
          ! absolute positioning
          J=LPHOST(IPT+1)
          K=LPHOST(IPT+2)
          L=LPHOST(IPT+3)
          V1=LPVALUE(1,ILP)
          V2=LPVALUE(2,ILP)
          V3=LPVALUE(3,ILP)
          IF(V1 == ZERO) THEN
             QCENT=.TRUE.
          ELSE
             QBIS=.FALSE.
             IF(V1 < ZERO) THEN
                V1=-V1
                QBIS=.TRUE.
             ENDIF
             RX=X(I)-X(J)
             RY=Y(I)-Y(J)
             RZ=Z(I)-Z(J)
             RDOT=RX*RX+RY*RY+RZ*RZ
             FDOT=(DX(I)*RX+DY(I)*RY+DZ(I)*RZ)/RDOT
             RDOT=SQRT(RDOT)
             FRX=RX*FDOT
             FRY=RY*FDOT
             FRZ=RZ*FDOT
             !
             !  Rx is the radial vector
             !  Px is the out-of-plane vector (unnormalized)
             !  Sx is the j-k vector
             !  Tx is the k-l vector
             !
             !  FRx is the radial force
             !  FPx is the torsional force
             !  FTx is the angular force
             !
             ! process the radial force
             !
             DX(J)=DX(J)+FRX
             DY(J)=DY(J)+FRY
             DZ(J)=DZ(J)+FRZ
             !
             ! prepare for angle and torsion forces
             IF(QBIS) THEN
                SX=X(J)-HALF*(X(K)+X(L))
                SY=Y(J)-HALF*(Y(K)+Y(L))
                SZ=Z(J)-HALF*(Z(K)+Z(L))
                TX=HALF*(X(K)-X(L))
                TY=HALF*(Y(K)-Y(L))
                TZ=HALF*(Z(K)-Z(L))
             ELSE
                SX=X(J)-X(K)
                SY=Y(J)-Y(K)
                SZ=Z(J)-Z(K)
                TX=X(K)-X(L)
                TY=Y(K)-Y(L)
                TZ=Z(K)-Z(L)
             ENDIF
             SDOT=SQRT(SX*SX+SY*SY+SZ*SZ)
             TDOT=SQRT(TX*TX+TY*TY+TZ*TZ)
             PX=RY*SZ-RZ*SY
             PY=RZ*SX-RX*SZ
             PZ=RX*SY-RY*SX
             PDOT=SQRT(PX*PX+PY*PY+PZ*PZ)
             QX=SY*TZ-SZ*TY
             QY=SZ*TX-SX*TZ
             QZ=SX*TY-SY*TX
             QDOT=SQRT(QX*QX+QY*QY+QZ*QZ)
             IF(PDOT < TOL) THEN
                !  process as linear (P vector is zero)
                FTX=DX(I)-FRX
                FTY=DY(I)-FRY
                FTZ=DZ(I)-FRZ
                FACT=(SDOT-RDOT)/SDOT
                DX(J)=DX(J)+FTX*FACT
                DY(J)=DY(J)+FTY*FACT
                DZ(J)=DZ(J)+FTZ*FACT
                FACT=RDOT/SDOT
                IF(QBIS) THEN
                   FACT=FACT*HALF
                   DX(K)=DX(K)+FTX*FACT
                   DY(K)=DY(K)+FTY*FACT
                   DZ(K)=DZ(K)+FTZ*FACT
                   DX(L)=DX(L)+FTX*FACT
                   DY(L)=DY(L)+FTY*FACT
                   DZ(L)=DZ(L)+FTZ*FACT
                ELSE
                   DX(K)=DX(K)+FTX*FACT
                   DY(K)=DY(K)+FTY*FACT
                   DZ(K)=DZ(K)+FTZ*FACT
                ENDIF
             ELSE
                FPDOT=(DX(I)*PX+DY(I)*PY+DZ(I)*PZ)/PDOT
                ! Note: FPX, FPY, and FPZ are set incorrectly, but can be scaled
                ! by an arbitrary amount, or even set to 0, and the error will
                ! cancel out in the angular and torsional calculations as long
                ! as FPDOT is correct. Ryan Hayes 2022-03-18
                FPX=PX*FPDOT ! PX*FPDOT/PDOT
                FPY=PY*FPDOT ! PY*FPDOT/PDOT
                FPZ=PZ*FPDOT ! PZ*FPDOR/PDOT
                FTX=DX(I)-FRX-FPX
                FTY=DY(I)-FRY-FPY
                FTZ=DZ(I)-FRZ-FPZ
                !
                ! process the angular force
                !
                DX(J)=DX(J)+FTX
                DY(J)=DY(J)+FTY
                DZ(J)=DZ(J)+FTZ
                VX=RY*FTZ-RZ*FTY  ! torque
                VY=RZ*FTX-RX*FTZ
                VZ=RX*FTY-RY*FTX
                FACT=ONE/SDOT**2
                FTX=(SY*VZ-SZ*VY)*FACT
                FTY=(SZ*VX-SX*VZ)*FACT
                FTZ=(SX*VY-SY*VX)*FACT
                DX(J)=DX(J)-FTX
                DY(J)=DY(J)-FTY
                DZ(J)=DZ(J)-FTZ
                IF(QBIS) THEN
                   DX(K)=DX(K)+FTX*HALF
                   DY(K)=DY(K)+FTY*HALF
                   DZ(K)=DZ(K)+FTZ*HALF
                   DX(L)=DX(L)+FTX*HALF
                   DY(L)=DY(L)+FTY*HALF
                   DZ(L)=DZ(L)+FTZ*HALF
                ELSE
                   DX(K)=DX(K)+FTX
                   DY(K)=DY(K)+FTY
                   DZ(K)=DZ(K)+FTZ
                ENDIF
                !
                ! process the torsional force
                SRDOT=(SX*RX+SY*RY+SZ*RZ)/SDOT**2
                RRX=RX-SX*SRDOT
                RRY=RY-SY*SRDOT
                RRZ=RZ-SZ*SRDOT
                RRDOT=SQRT(RRX*RRX+RRY*RRY+RRZ*RRZ)
                STDOT=(SX*TX+SY*TY+SZ*TZ)/SDOT**2
                TTX=TX-SX*STDOT
                TTY=TY-SY*STDOT
                TTZ=TZ-SZ*STDOT
                TTDOT=SQRT(TTX*TTX+TTY*TTY+TTZ*TTZ)
                FACT=(RRDOT*FPDOT)/(TTDOT*QDOT)
                FQX=QX*FACT
                FQY=QY*FACT   
                FQZ=QZ*FACT
                DX(L)=DX(L)+FQX
                DY(L)=DY(L)+FQY
                DZ(L)=DZ(L)+FQZ
                DX(J)=DX(J)+FPX*(ONE+SRDOT)+FQX*STDOT
                DY(J)=DY(J)+FPY*(ONE+SRDOT)+FQY*STDOT
                DZ(J)=DZ(J)+FPZ*(ONE+SRDOT)+FQZ*STDOT
                FTX=FQX*(ONE+STDOT)+FPX*SRDOT
                FTY=FQY*(ONE+STDOT)+FPY*SRDOT
                FTZ=FQZ*(ONE+STDOT)+FPZ*SRDOT
                IF(QBIS) THEN
                   DX(K)=DX(K)-FTX*HALF
                   DY(K)=DY(K)-FTY*HALF
                   DZ(K)=DZ(K)-FTZ*HALF
                   DX(L)=DX(L)-FTX*HALF
                   DY(L)=DY(L)-FTY*HALF
                   DZ(L)=DZ(L)-FTZ*HALF
                ELSE
                   DX(K)=DX(K)-FTX
                   DY(K)=DY(K)-FTY
                   DZ(K)=DZ(K)-FTZ
                ENDIF
             ENDIF
          ENDIF
       ENDIF
       IF(QCENT) THEN
          ! center of mass/geometry
          MTOT=ZERO 
          DO K=1,N
             J=LPHOST(IPT+K)
             AM=ONE
             IF(LPWGHT(ILP)) AM=AMASS(J)
             MTOT=MTOT+AM
          ENDDO
          MTOT=ONE/MTOT
          DO K=1,N
             J=LPHOST(IPT+K)
             AM=ONE
             IF(LPWGHT(ILP)) AM=AMASS(J)
             DX(J)=DX(J)+DX(I)*AM*MTOT
             DY(J)=DY(J)+DY(I)*AM*MTOT
             DZ(J)=DZ(J)+DZ(I)*AM*MTOT
          ENDDO
       ENDIF
       DX(I)=ZERO
       DY(I)=ZERO
       DZ(I)=ZERO
       !
       !
#if KEY_DEBUG==1 /*debug_b*/
       ! temp code for debug
       ! save net force and torque relative to the origin
       fnetf(1)=zero
       fnetf(2)=zero
       fnetf(3)=zero
       tnetf(1)=zero
       tnetf(2)=zero
       tnetf(3)=zero
       do k=1,n
          J=LPHOST(IPT+K)
          fnetf(1)=fnetf(1)+dx(j)
          fnetf(2)=fnetf(2)+dy(j)
          fnetf(3)=fnetf(3)+dz(j)
          tnetf(1)=tnetf(1)+dy(j)*z(j)-dz(j)*y(j)
          tnetf(2)=tnetf(2)+dz(j)*x(j)-dx(j)*z(j)
          tnetf(3)=tnetf(3)+dx(j)*y(j)-dy(j)*x(j)
       enddo
       ! do they match
       ok=.true.
       do k=1,3
          if(abs(fneti(k)-fnetf(k)) > tol) ok=.false.
          if(abs(tneti(k)-tnetf(k)) > tol) ok=.false.
       enddo
       if(.not.ok .and. n > 0) then
          write(6,888) ILP,(LPHOST(IPT+K),k=0,n)
888       format(' LONEPR:: Bad force or torque:',10I5)
          write(6,887) fneti,fnetf
887       format(' LONEPR:: fneti,fnetf:',6F12.5)
          write(6,886) tneti,tnetf
886       format(' LONEPR:: tneti,tnetf:',6F12.5)
          do k=0,n
             i=LPHOST(IPT+K)
             write(6,885) i,X(I),Y(I),Z(I)
885          format(' LONEPR:: i,x(i),..:',I5,3F12.5)
          enddo
       endif
#endif /* (debug_b)*/
99     CONTINUE
    ENDDO
    !
#if KEY_DOMDEC==1
    if (q_domdec) then
       call lonepr_comm_force(dx, dy, dz)
    endif
#endif

    RETURN
  END SUBROUTINE LONEPRF


#else /* (lonepair_main)*/
  SUBROUTINE LONEPRS(COMLYN,COMLEN)
    INTEGER COMLEN
    character(len=*) COMLYN
    CALL WRNDIE(0,'<LONEPRS>','Lonepair code not compiled')
    return
  end SUBROUTINE LONEPRS
#endif /* (lonepair_main)*/

end module lonepr

