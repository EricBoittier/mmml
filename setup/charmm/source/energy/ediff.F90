module ediff
contains
   SUBROUTINE EDIFFOP(COMLYN,COMLEN,ICYCLE)
      !-----------------------------------------------------------------------
      ! Process the EDIFF (energy difference) command.
      !
      ! Syntax:    EDIFF  {      }  atom-selection
      !                   { MAIN }
      !                   { COMP }
      !
      !-----------------------------------------------------------------------
      ! This routine computes the difference in energy of two coordinate
      ! sets based on a selected set of atoms that have potentially moved.
      ! This is a non-list based method.
      !
      ! By:  Bernard R. Brooks     22-OCT-1984
      ! SPASIBA Force Field added by P. Lagant and R. Stote (11/01)
      ! SPASIBA Force Field removed for c34b2 and c35a1 releases
      !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number

  use code
  use coord
  use coordc
  use energym
  use select
  use stream
  use string

  use bases_fcm
  use psf
  use hbondm
  use image
  use memory
      implicit none

      ! Passed variables.
      CHARACTER(len=*) :: COMLYN
      INTEGER   COMLEN, ICYCLE
      ! Local variables.
      INTEGER   UNIT
      LOGICAL   QMAIN, QCOMP, QPRINT

      integer,allocatable,dimension(:) :: ISLCT

      ! Do an update only if really necessary...
      IF(MUSTUP) CALL UPDATE(COMLYN,COMLEN,X,Y,Z,WMAIN,.TRUE., &
           .TRUE.,.TRUE.,.TRUE.,.TRUE., &
           0,[zero],[zero],[zero],[zero],[zero],[zero])

      call chmalloc('ediff.src','EDIFFOP','ISLCT',NATOMT,intg=ISLCT)

      CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

      QMAIN = INDXA(COMLYN, COMLEN, 'MAIN') .GT. 0
      QCOMP = INDXA(COMLYN, COMLEN, 'COMP') .GT. 0
      IF(.NOT.QCOMP .AND. .NOT.QMAIN) THEN
         QMAIN=.TRUE.
         QCOMP=.TRUE.
      ENDIF

      QPRINT = .NOT. (INDXA(COMLYN, COMLEN, 'NOPR') .GT. 0)

      !XXX in edifftest, these members of BIMAG are null - mkg 07/2009
      CALL EDIFF2(X,Y,Z,XCOMP,YCOMP,ZCOMP, &
            ISLCT,BIMAG%IMATTR,BIMAG%IMATPT, &
            QMAIN,QCOMP)

      call chmdealloc('ediff.src','EDIFFOP','ISLCT',NATOMT,intg=ISLCT)

      IF (QPRINT) THEN
         UNIT = GTRMI(COMLYN, COMLEN, 'UNIT', OUTU)
         ICYCLE = ICYCLE + 1
         IF(UNIT.EQ.OUTU .AND. PRNLEV.LT.2) QPRINT=.FALSE.
         IF(UNIT.NE.OUTU .AND. IOLEV.LT.0) QPRINT=.FALSE.
         IF(QPRINT) CALL PRINTE(UNIT, EPROP, ETERM, 'EDIF', 'ENR', &
               .TRUE.,ICYCLE, ZERO, ZERO, .TRUE.)
      ENDIF

      RETURN
   END SUBROUTINE EDIFFOP

   SUBROUTINE EDIFF2(X,Y,Z,XM,YM,ZM, &
        ISLCT,IMTR,IMPT, &
        QMAIN,QCOMP)

     use nb_module
     !---   use nbutil_module,only:getbnd

     !-----------------------------------------------------------------------
     ! This routine does the actual work of the EDIFF command.
     !
     ! By Bernard R. Brooks - NIH - December 15, 1997
     !
     use chm_kinds
     use chm_types
     use dimens_fcm
     use number
     use bases_fcm
     use block_fcm
     use deriv
     use econtmod
     use eintern
     use fast
     use psf
     use param
     use cnst_fcm
     use code
     use energym
     use hbondm
     use inbnd
     use image
     use stream
     use memory
     use lonepr
     use parallel
     use usermod,only: usere
     use cmapm
#if KEY_PNM==1
     use pnm, only : pnm_ene 
#endif
#if KEY_MMFF==1
     use ffieldm
     use mmffm
     use escalar_mm
#endif
#if KEY_DOMDEC==1
     use domdec_common,only:q_domdec
#endif
     implicit none

      real(chm_real) X(*),Y(*),Z(*),XM(*),YM(*),ZM(*)
      INTEGER ISLCT(*)
      INTEGER IMTR(*),IMPT(*)
      LOGICAL QMAIN,QCOMP

      integer :: SKIPLEN
      integer,allocatable,dimension(:) :: ISKIP
      real(chm_real),allocatable,dimension(:) :: RTEMP
      real(chm_real),allocatable,dimension(:) :: ECONTM, ETERMM
      real(chm_real),allocatable,dimension(:) :: DXM, DYM, DZM
      INTEGER I,J,ITEMP,ISTRT,IEND,ITRANS
      real(chm_real) EXX, ENBX, EELX, EST2X
      LOGICAL ERR
      INTEGER ATFRST,ATLAST

#if KEY_MMFF==1
      INTEGER DERIVS
      DERIVS=1
#endif
      ! hwm: per BRB's suggestion on 220722
      !SKIPLEN = MAX(NBONDT,NTHETT,NPHIT,NIMPHT,NHB,NCSPHI &
      SKIPLEN = MAX(NBONDT2,NTHETT,NPHIT,NIMPHT,NHB,NCSPHI &
#if KEY_CMAP==1
            ,NCRTT &  
#endif
            )
      call chmalloc('ediff.src','EDIFF2','ISKIP',SKIPLEN,intg=ISKIP)
      call chmalloc('ediff.src','EDIFF2','RTEMP',NATOM,crl=RTEMP)
      call chmalloc('ediff.src','EDIFF2','ECONTM',NATOMT,crl=ECONTM)
      call chmalloc('ediff.src','EDIFF2','ETERMM',LENENT,crl=ETERMM)
      call chmalloc('ediff.src','EDIFF2','DXM',NATOMT,crl=DXM)
      call chmalloc('ediff.src','EDIFF2','DYM',NATOMT,crl=DYM)
      call chmalloc('ediff.src','EDIFF2','DZM',NATOMT,crl=DZM)

      ! copy atom selections to image atoms.
      IF(NTRANS.GT.0) THEN
        ITEMP=NATOM+1
        DO ITRANS=1,NTRANS
          ISTRT=ITEMP
          IEND=IMPT(ITRANS)
          ITEMP=IEND+1
          DO I=ISTRT,IEND
            J=IMTR(I)
            ISLCT(I)=ISLCT(J)
          ENDDO
        ENDDO
      ENDIF
      !
      ! Now fill skip arrays for bonds, angles,...
      !
      DO I=1,LENENT
        ETERM(I) = ZERO
        ETERMM(I) = ZERO
      ENDDO

      DO I=1,NATOM
        IF(IMOVE(I).LE.0) THEN
          DX(I)=ZERO
          DY(I)=ZERO
          DZ(I)=ZERO
          DXM(I)=ZERO
          DYM(I)=ZERO
          DZM(I)=ZERO
        ENDIF
      ENDDO

      ! Zero the energy contribution array.
      IF(QECONT) THEN
        DO I=1,NATOM
          ECONT(I)=ZERO
          ECONTM(I)=ZERO
        ENDDO
      ENDIF

      !-----------------------------------------------------------------------
      ! User energy term
      IF(QETERM(USER)) THEN
        IF(QMAIN) CALL USERE(ETERM(USER),X,Y,Z,DX,DY,DZ, &
              QECONT,ECONT,NATOM)
        IF(QCOMP) CALL USERE(ETERMM(USER),XM,YM,ZM,DXM,DYM,DZM, &
              QECONT,ECONTM,NATOM)
      ENDIF
#if KEY_PNM==1 /* (pnm) */
      ! compute PNM energy, 0504PJ07
      IF(QETERM(PNME)) THEN
        IF(QMAIN) CALL PNM_ENE(ETERM(PNME),X,Y,Z,DX,DY,DZ, &
&               QECONT,ECONT,NATOM)
        IF(QCOMP) CALL PNM_ENE(ETERMM(PNME),XM,YM,ZM,DXM,DYM,DZM, &
&               QECONT,ECONTM,NATOM)

      ENDIF
#endif /*(pnm) */
      !
      ! USE NORMAL INTERNAL ENERGY ROUTINES
      !
      !-----------------------------------------------------------------------
      ! Bond energy term
      IF(NBOND.GT.0.AND.QETERM(BOND)) THEN
        DO I=1,NBOND
          IF (ISLCT(IB(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JB(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) THEN
          CALL EBOND(ETERM(BOND),IB,JB,ICB,NBOND,CBC,CBB, &
                DX,DY,DZ,X,Y,Z,QECONT,ECONT, &
                1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)
        ENDIF
        IF(QCOMP) THEN
          CALL EBOND(ETERMM(BOND),IB,JB,ICB,NBOND,CBC,CBB, &
                DXM,DYM,DZM,XM,YM,ZM,QECONT,ECONTM, &
                1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)
        ENDIF
      ENDIF

      !-----------------------------------------------------------------------
      ! Angle energy term
      IF(NTHETA.GT.0.AND.QETERM(ANGLE)) THEN
        DO I=1,NTHETA
          IF (ISLCT(IT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) THEN
          CALL EANGLE(ETERM(ANGLE),IT,JT,KT,ICT,NTHETA, &
                CTC,CTB,DX,DY,DZ,X,Y,Z, &
                QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
                )
        ENDIF
        IF(QCOMP) THEN
          CALL EANGLE(ETERMM(ANGLE),IT,JT,KT,ICT,NTHETA, &
                CTC,CTB,DXM,DYM,DZM,XM,YM,ZM, &
                QECONT,ECONTM,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
                )
        ENDIF
      ENDIF

      !-----------------------------------------------------------------------
      ! Dihedral energy term
      IF(NPHI.GT.0.AND.QETERM(DIHE)) THEN
        DO I=1,NPHI
          IF (ISLCT(IP(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JP(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KP(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(LP(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) CALL EPHI(ETERM(DIHE),IP,JP,KP,LP,ICP,NPHI, &
             CPC,CPD,CPB,CPCOS, &
             CPSIN,DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
             QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.)

        IF(QCOMP) CALL EPHI(ETERMM(DIHE),IP,JP,KP,LP,ICP,NPHI, &
             CPC,CPD,CPB,CPCOS, &
             CPSIN,DXM,DYM,DZM,XM,YM,ZM,.FALSE.,(/ZERO/), &
             QECONT,ECONTM,1,ISKIP,(/ZERO/),(/0/),.FALSE.)

      ENDIF

      !-----------------------------------------------------------------------
#if KEY_MMFF==1 /* (mmff_eterms) */
      IF (FFIELD.EQ.MMFF) THEN

        !-----------------------------------------------------------------------
        ! Out-of-plane energy (MMFF)
        IF(NTHETA.GT.0.AND.QETERM(OOPL)) THEN
          DO I=1,NTHETA
            ! Treat as an interaction between central atom of bond angle (JT) and
            ! "out-of-plane" atom (LTHETA); note that an out-of-plane term is only
            ! defined for angle I if LTHETA(I).GT.0
            IF(LTHETA(I).GT.0) THEN
              IF (ISLCT(IT(I)).EQ.1) THEN
                ISKIP(I)=0
              ELSE IF (ISLCT(JT(I)).EQ.1) THEN
                ISKIP(I)=0
              ELSE IF (ISLCT(KT(I)).EQ.1) THEN
                ISKIP(I)=0
              ELSE IF (ISLCT(LTHETA(I)).EQ.1) THEN
                ISKIP(I)=0
              ELSE
                ISKIP(I)=1
              ENDIF
            ENDIF
          ENDDO
          IF(QMAIN) CALL EOOPL(ETERM(OOPL),IT,JT,KT,LTHETA, &
                ICOOP,NTHETA,OOPLFC, &
                X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                QECONT,ECONT,1,ISKIP)
          IF(QCOMP) CALL EOOPL(ETERMM(OOPL),IT,JT,KT,LTHETA, &
                ICOOP,NTHETA,OOPLFC, &
                XM,YM,ZM,DXM,DYM,DZM,LTSD,DERIVS, &
                QECONT,ECONTM,1,ISKIP)
        ENDIF

        !-----------------------------------------------------------------------
        ! Strech-Bend coupling energy (MMFF)
        IF(NTHETA.GT.0.AND.QETERM(STRB)) THEN
          DO I=1,NTHETA
            ! Treat stretch-bends, like the angles they contain, as interactions
            ! between the "outer" atoms (IT and KT).
            IF (ISLCT(IT(I)).EQ.1) THEN
              ISKIP(I)=0
            ELSE IF (ISLCT(JT(I)).EQ.1) THEN
              ISKIP(I)=0
            ELSE IF (ISLCT(KT(I)).EQ.1) THEN
              ISKIP(I)=0
            ELSE
              ISKIP(I)=1
            ENDIF
          ENDDO
          IF(QMAIN) CALL ESTRBND(ETERM(STRB),IT,JT,KT,ICT,NTHETA, &
                CTB,IB,JB,ICB,CBB,STRBLIST,ICSTBN,STBNP, &
                X,Y,Z,DX,DY,DZ,LTSD,DERIVS, &
                QECONT,ECONT,1,ISKIP)
          IF(QCOMP) CALL ESTRBND(ETERMM(STRB),IT,JT,KT,ICT,NTHETA, &
                CTB,IB,JB,ICB,CBB,STRBLIST,ICSTBN,STBNP, &
                XM,YM,ZM,DXM,DYM,DZM,LTSD,DERIVS, &
                QECONT,ECONTM,1,ISKIP)
        ENDIF
      ELSE
#endif /*(mmff_eterms) */

      !-----------------------------------------------------------------------
      ! Urey-Bradley energy term
      IF(NTHETA.GT.0.AND.QETERM(UREYB)) THEN
        DO I=1,NTHETA
          IF (ISLCT(IT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) CALL EBOND(ETERM(UREYB),IT,KT,ICT,NTHETA, &
              CTUC,CTUB,DX,DY,DZ,X,Y,Z,QECONT, &
              ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

        IF(QCOMP) CALL EBOND(ETERMM(UREYB),IT,KT,ICT,NTHETA, &
              CTUC,CTUB,DXM,DYM,DZM,XM,YM,ZM,QECONT, &
              ECONTM,1,ISKIP,(/ZERO/),(/0/),.FALSE.,2,.FALSE.)

      ENDIF

      !-----------------------------------------------------------------------
      ! Improper energy term
      IF(NIMPHI.GT.0.AND.QETERM(IMDIHE)) THEN
        DO I=1,NIMPHI
          IF (ISLCT(IM(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JM(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KM(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(LM(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) CALL EPHI(ETERM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI, &
             CIC,CID,CIB,CICOS,CISIN, &
             DX,DY,DZ,X,Y,Z,.FALSE.,(/ZERO/), &
             QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE.)
           
        IF(QCOMP) CALL EPHI(ETERMM(IMDIHE),IM,JM,KM,LM,ICI,NIMPHI, &
             CIC,CID,CIB,CICOS,CISIN, &
             DXM,DYM,DZM,XM,YM,ZM,.FALSE.,(/ZERO/), &
             QECONT,ECONTM,1,ISKIP,(/ZERO/),(/0/),.FALSE.)

      ENDIF

#if KEY_CMAP==1
      !-----------------------------------------------------------------------
      ! Cross-term map
      IF(NCRTERM.GT.0.AND.QETERM(CMAP)) THEN
        DO I=1,NCRTERM
          IF (ISLCT(I1CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(J1CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(K1CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(L1CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(I2CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(J2CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(K2CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(L2CT(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) CALL ECMAP(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
              I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
              DX,DY,DZ,X,Y,Z, &
              QECONT,ECONT,1,ISKIP, (/ZERO/), (/0/), .FALSE.)
        IF(QCOMP) CALL ECMAP(ETERM(CMAP),I1CT,J1CT,K1CT,L1CT, &
              I2CT,J2CT,K2CT,L2CT,ICCT,NCRTERM, &
              DXM,DYM,DZM,XM,YM,ZM, &
              QECONT,ECONTM,1,ISKIP, (/ZERO/), (/0/), .FALSE.)
      ENDIF

#endif

      !-----------------------------------------------------------------------
#if KEY_MMFF==1 /* (mmff_endif) */
      ENDIF !IF(FFIELD.EQ.MMFF)
#endif /*(mmff_endif) */

      !-----------------------------------------------------------------------
      ! Nonbond energy term
      !
      ! Make sure counters, flags and cuttoffs, etc are correct.
      CALL GETBND(BNBND,.TRUE.)

      IF(QMAIN) CALL NBONDD(X,Y,Z,DX,DY,DZ,ISLCT,CG, &
            BNBND%INB14,BNBND%IBLO14,CTOFNB, &
            CCNBA,CCNBB,CCNBC,CCNBD, &
            IACNB,NITCC2,LOWTP, &
            ETERM(VDW),ETERM(ELEC),QETERM(ELEC),QETERM(VDW), &
            NATOM,1,NATOM,LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT, &
            CTONNB,E14FAC,EPS &
            ,QETEN,QETSR)

      IF(QCOMP) CALL NBONDD(XM,YM,ZM,DXM,DYM,DZM,ISLCT,CG, &
            BNBND%INB14,BNBND%IBLO14,CTOFNB, &
            CCNBA,CCNBB,CCNBC,CCNBD, &
            IACNB,NITCC2,LOWTP, &
            ETERMM(VDW),ETERMM(ELEC),QETERM(ELEC),QETERM(VDW), &
            NATOM,1,NATOM,LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT, &
            CTONNB,E14FAC,EPS, &
            QETEN,QETSR)

      IF(NTRANS.GT.0) THEN
        IF(BIMAG%NIMNBS.GT.0 .OR. BIMAG%NIMNBX.GT.0)THEN
!         Self terms are present
          CALL WRNDIE(-2,'<EDIFF>', &
                'Image self terms present. Energy is wrong')
        ENDIF
        IF(LIMALL) CALL WRNDIE(-2,'<EDIFF>', &
              'The IMALL keyword causes bad EDIFF energies')

        IF(QMAIN) CALL NBONDD(X,Y,Z,DX,DY,DZ,ISLCT,CG, &
             BIMAG%IMINB,BIMAG%IMIBLO,CTOFNB, &
              CCNBA,CCNBB,CCNBC,CCNBD, &
              IACNB,NITCC2,LOWTP, &
              ETERM(IMVDW),ETERM(IMELEC), &
              QETERM(IMELEC),QETERM(IMVDW), &
              NATOM,NATOM+1,NATOMT,LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT, &
              CTONNB,E14FAC,EPS, &
              QETEN,QETSR)

        IF(QCOMP) CALL NBONDD(XM,YM,ZM,DXM,DYM,DZM,ISLCT,CG, &
              BIMAG%IMINB,BIMAG%IMIBLO,CTOFNB, &
              CCNBA,CCNBB,CCNBC,CCNBD, &
              IACNB,NITCC2,LOWTP, &
              ETERMM(IMVDW),ETERMM(IMELEC), &
              QETERM(IMELEC),QETERM(IMVDW), &
              NATOM,NATOM+1,NATOMT,LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT, &
              CTONNB,E14FAC,EPS, &
              QETEN,QETSR)

      ENDIF

      !-----------------------------------------------------------------------
      ! Hydrogen-bond energy term
      IF(NHB.GT.0.AND.QETERM(HBOND)) THEN
        DO I=1,NHB
          IF (ISLCT(IHB(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JHB(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KHB(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (lhb(i) > 0 ) then 
             if (ISLCT(LHB(I)).EQ.1) ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
        IF(QMAIN) CALL EHBOND(ETERM(HBOND),IHB,JHB,KHB,LHB,ICH,NHB, &
              CHBA,CHBB,DX,DY,DZ,X,Y,Z, &
              QECONT,ECONT,0,0,1,ISKIP, &
              CTONHB,CTOFHB,CTONHA,CTOFHA, &
              HBEXPN,0,0,.FALSE.)
        IF(QCOMP) CALL EHBOND(ETERMM(HBOND),IHB,JHB,KHB,LHB,ICH,NHB, &
              CHBA,CHBB,DXM,DYM,DZM,XM,YM,ZM, &
              QECONT,ECONTM,0,0,1,ISKIP, &
              CTONHB,CTOFHB,CTONHA,CTOFHA, &
              HBEXPN,0,0,.FALSE.)
      ENDIF

      !-----------------------------------------------------------------------
      ! Harmonic restraint energy term
      IF(QCNSTR.AND.QETERM(CHARM)) THEN
        DO I=1,NATOM
          IF (ISLCT(I).EQ.1) THEN
            RTEMP(I)=KCNSTR(I)
          ELSE
            RTEMP(I)=ZERO
          ENDIF
        ENDDO
        ! Just do the absolute types.
        IF(QMAIN) CALL ECNSTR(ETERM(CHARM),QCNSTR,REFX,REFY,REFZ,RTEMP, &
             NATOM,KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             X,Y,Z,DX,DY,DZ, &
             QECONT,ECONT, (/ ZERO /), (/ 0 /), .FALSE. &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )
        IF(QCOMP) CALL ECNSTR(ETERMM(CHARM),QCNSTR,REFX,REFY,REFZ,RTEMP, &
             NATOM,KCEXPN,XHSCALE,YHSCALE,ZHSCALE,1, &
             NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
             XM,YM,ZM,DXM,DYM,DZM, &
             QECONT,ECONTM, (/ ZERO /), (/ 0 /), .FALSE. &
             ,(/0/),(/0,0/),(/0,0/),(/0,0/) &
             )
      ENDIF

      !-----------------------------------------------------------------------
      !-----------------------------------------------------------------------
      ! Harmonic restraint energy term
      IF(QCNSTR.AND.QETERM(PCHARM).and.numpca>0) THEN
         DO I=1,NATOM
            IF (ISLCT(I).EQ.1) THEN
               RTEMP(I)=KCNSTR(I)
            ELSE
               RTEMP(I)=ZERO
            ENDIF
         ENDDO
         ! Just do the absolute types.
         IF(QMAIN) CALL ECNSTR(ETERM(PCHARM),QCNSTR,REFX,REFY,REFZ,RTEMP, &
              NATOM,KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
              NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
              X,Y,Z,DX,DY,DZ, &
              QECONT,ECONT, (/ ZERO /), (/ 0 /), .FALSE. &
              ,numpca,pcax,pcay,pcaz &
              )
         IF(QCOMP) CALL ECNSTR(ETERMM(PCHARM),QCNSTR,REFX,REFY,REFZ,RTEMP, &
              NATOM,KCEXPN,XHSCALE,YHSCALE,ZHSCALE,2, &
              NUMHSETS,TYPHSET,IHSET,QHNORT,QHNOTR, &
              XM,YM,ZM,DXM,DYM,DZM, &
              QECONT,ECONTM, (/ ZERO /), (/ 0 /), .FALSE. &
              ,numpca,pcax,pcay,pcaz &
              )
      ENDIF

      !-----------------------------------------------------------------------
      ! Dihedral restraint energy term
      IF((NCSPHI.GT.0) .AND. QETERM(CDIHE)) THEN
        DO I=1,NCSPHI
          IF (ISLCT(ICS(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(JCS(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(KCS(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE IF (ISLCT(LCS(I)).EQ.1) THEN
            ISKIP(I)=0
          ELSE
            ISKIP(I)=1
          ENDIF
        ENDDO
#if KEY_DOMDEC==1
        if (q_domdec) CALL WRNDIE(-5,'<EDIFF>','HARMONIC RESTRAINTS NOT READY FOR DOMDEC')
#endif
        IF(QMAIN) CALL EPHI(ETERM(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
              CCSC,CCSD,CCSB,CCSCOS,CCSSIN, &
              DX,DY,DZ,X,Y,Z,.TRUE.,CCSW, &
              QECONT,ECONT,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
              )

        IF(QCOMP) CALL EPHI(ETERMM(CDIHE),ICS,JCS,KCS,LCS,ICCS,NCSPHI, &
              CCSC,CCSD,CCSB,CCSCOS,CCSSIN, &
              DXM,DYM,DZM,XM,YM,ZM,.TRUE.,CCSW, &
              QECONT,ECONTM,1,ISKIP,(/ZERO/),(/0/),.FALSE. &
              )
      ENDIF

      !-----------------------------------------------------------------------
      ! Finish up...
      EXX=ZERO
      DO I=1,LENENT
        EXX=EXX+ETERM(I)-ETERMM(I)
      ENDDO
      EPROP(EPOT)=EXX

#if KEY_LONEPAIR==1

#if KEY_PARALLEL==1 /* (paramain) */
#if KEY_PARAFULL==1 /* (parfmain) */
      ATFRST=1+IPARPT(MYNOD)
      ATLAST=IPARPT(MYNODP)
#elif KEY_PARASCAL==1 || KEY_SPACDEC==1 /* (parfmain) */
      ATFRST=1
      ATLAST=NATOM
#endif /*(parfmain) */
#else /* (paramain) */
      ATFRST=1
      ATLAST=NATOM
#endif /*(paramain) */

      IF(NUMLP.GT.0) THEN
        IF(QMAIN) CALL LONEPRF(ATFRST,ATLAST,X,Y,Z,AMASS,DX,DY,DZ,NUMLP, &
              LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
        IF(QCOMP) CALL LONEPRF(ATFRST,ATLAST,XM,YM,ZM,AMASS,DXM,DYM,DZM, &
              NUMLP,LPNHOST,LPHPTR,LPHOST,LPVALUE,LPWGHT)
      ENDIF
#endif

      DO I=1,NATOM
        IF(IMOVE(I).GT.0) THEN
          DX(I)=ZERO
          DY(I)=ZERO
          DZ(I)=ZERO
        ELSE
          DX(I)=DX(I)-DXM(I)
          DY(I)=DY(I)-DYM(I)
          DZ(I)=DZ(I)-DZM(I)
        ENDIF
      ENDDO

      call chmdealloc('ediff.src','EDIFF2','DZM',NATOMT,crl=DZM)
      call chmdealloc('ediff.src','EDIFF2','DYM',NATOMT,crl=DYM)
      call chmdealloc('ediff.src','EDIFF2','DXM',NATOMT,crl=DXM)
      call chmdealloc('ediff.src','EDIFF2','ETERMM',LENENT,crl=ETERMM)
      call chmdealloc('ediff.src','EDIFF2','ECONTM',NATOMT,crl=ECONTM)
      call chmdealloc('ediff.src','EDIFF2','RTEMP',NATOM,crl=RTEMP)
      call chmdealloc('ediff.src','EDIFF2','ISKIP',SKIPLEN,intg=ISKIP)

      RETURN
   END SUBROUTINE EDIFF2

   SUBROUTINE NBONDD(X,Y,Z,DX,DY,DZ,ISLCT,CG, &
                        INB14,IBLO14,CTOFNB, &
                        CCNBA,CCNBB,CCNBC,CCNBD,IACNB,NITCC2,LOWTP, &
                        ENB,EEL,LELECX,LVDWX,NATOMX,IATMIN,IATMAX, &
                        LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT, &
                        CTONNB,E14FAC,EPS, &
                        QETEN,QETSR)
!
!
  use chm_kinds
  use dimens_fcm
  use number
  use stream
  use consta
  use param
  use block_fcm
  use pbound
!
      implicit none
!
      real(chm_real)  X(*),Y(*),Z(*)
      real(chm_real)  DX(*),DY(*),DZ(*)
      INTEGER ISLCT(*)
      real(chm_real)  CG(*)
      INTEGER INB14(*),IBLO14(*)
      real(chm_real)  CTOFNB
      real(chm_real)  CCNBA(*),CCNBB(*),CCNBC(*),CCNBD(*)
      INTEGER IACNB(*),NITCC2,LOWTP(*)
!
      real(chm_real)  ENB,EEL
      LOGICAL LELECX,LVDWX
      INTEGER NATOMX,IATMIN,IATMAX
      LOGICAL LCONS,LSHFT,LFSWT,LVFSWT,LVSHFT
      real(chm_real)  CTONNB,E14FAC,EPS
!
      INTEGER I,J,IS,IQ,NAT,NGAT,IRS,NXI,NXIMAX
      INTEGER JRS,JS,JQ,IRST,JRST,INBX,IX14,IX14P
      real(chm_real) XMIN,XMAX,YMIN,YMAX,ZMIN,ZMAX,XD,YD,ZD
      real(chm_real) XD1,YD1,ZD1,R2,R,XI,YI,ZI,DXT,DYT,DZT
      LOGICAL MOVEFG
!
      INTEGER IVECT,JVECT
      real(chm_real) CA,CC,CH,ENE,ENN
      real(chm_real) TF,DTX,DTY,DTZ
      real(chm_real) TFELEC,TFVDW
      real(chm_real) S2,TR2,TR6,FSW,DFSW,FSH,TTPW1,TTPW2,TTP12,SWTMP
      real(chm_real) EADD,ON3,ON6,ONOFF2,OFF3,OFF4,OFF5,OFF6, &
           R1,R3,DENOM, &
             ACOEF,BCOEF,CCOEF,DCOEF,COVER3,DOVER5,CONST,ENEVDW
      real(chm_real) RECOF3,RECOF6,OFDIF3,OFDIF6,ONOFF3,ONOFF6, &
             CR6,CR12,RJUNK3,RJUNK6,MIN2OF
!
      real(chm_real) C2ONNB,C2OFNB,CTROF2,C4ROF2,RUL3,RUL12,RIJL,RIJU
      real(chm_real) CGF,CGT
      real(chm_real) CORR
      INTEGER ITEMP,NPR,IACI
      INTEGER IBL,JBL,KK
      LOGICAL ELECFG,LOUTER,RSHFT,RSWIT,CSHIFT,CFSWIT,CSHFT,CSWIT, &
                  QETEN,QETSR,SWITCH,LVSW,LVSH,LVFSW,RSHIFT,RFSWIT
!
!-----------------------------------------------------------------------
!
      ENB=ZERO
      EEL=ZERO
      ELECFG=(LELECX.AND.(EPS.NE.ZERO))
      IF (.NOT.(LVDWX.OR.ELECFG)) RETURN
      CGF=ZERO
      IF (ELECFG) CGF=CCELEC/EPS
!
! Set flags for electrostatics options (6 options supported)
      RSHIFT= .NOT.LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
      RFSWIT= .NOT.LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
      RSHFT = .NOT.LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
      RSWIT = .NOT.LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
      CSHIFT=      LCONS .AND.      LSHFT .AND.      LFSWT .AND. ELECFG
      CFSWIT=      LCONS .AND. .NOT.LSHFT .AND.      LFSWT .AND. ELECFG
      CSHFT =      LCONS .AND.      LSHFT .AND. .NOT.LFSWT .AND. ELECFG
      CSWIT =      LCONS .AND. .NOT.LSHFT .AND. .NOT.LFSWT .AND. ELECFG
!
      IF(RSHIFT .OR. RFSWIT) THEN
         RETURN
      ENDIF
!
      LVFSW=      LVFSWT                   .AND. LVDWX
      LVSH = .NOT.LVFSWT .AND.      LVSHFT .AND. LVDWX
      LVSW = .NOT.LVFSWT .AND. .NOT.LVSHFT .AND. LVDWX
      SWITCH= RSWIT.OR.CSWIT.OR.LVSW
!
      C2OFNB=CTOFNB*CTOFNB
      C2ONNB=CTONNB*CTONNB
      CTROF2=-ONE/C2OFNB
      C4ROF2=FOUR*CTROF2
      IF (CSHIFT) MIN2OF = MINTWO/CTOFNB
!
      IF (SWITCH) THEN
         IF (CTOFNB.GT.CTONNB) THEN
            RUL3=ONE/(C2OFNB-C2ONNB)**3
            RUL12=TWELVE*RUL3
         ENDIF
      ENDIF
      IF (CFSWIT) THEN
!       force-based cdie switching coeffs
        IF(CTONNB .LT. CTOFNB) THEN
          ONOFF2 = C2ONNB*C2OFNB
          ON3    = C2ONNB*CTONNB
          OFF3   = C2OFNB*CTOFNB
          OFF4   = C2OFNB*C2OFNB
          OFF5   = OFF3*C2OFNB
          DENOM  = ONE/(C2OFNB-C2ONNB)**3
          EADD   = (ONOFF2*(CTOFNB-CTONNB)-(OFF5-ON3*C2ONNB)/FIVE)* &
                   EIGHT*DENOM
          ACOEF  = OFF4*(C2OFNB-THREE*C2ONNB)*DENOM
          BCOEF  = SIX*ONOFF2*DENOM
          COVER3 = -(C2ONNB+C2OFNB)*DENOM
          CCOEF  = THREE*COVER3
          DCOEF  = TWO*DENOM
          DOVER5 = DCOEF/FIVE
          CONST  = BCOEF*CTOFNB-ACOEF/CTOFNB+COVER3*OFF3+DOVER5*OFF5
        ELSE
          EADD  = -ONE/CTOFNB
        END IF
      ENDIF
      IF (LVFSW) THEN
        OFF3 = C2OFNB*CTOFNB
        OFF6 = OFF3*OFF3
        RECOF6 = ONE/OFF6
        IF(CTONNB .LT. CTOFNB) THEN
          ON3 = C2ONNB*CTONNB
          ON6 = ON3*ON3
          RECOF3 = ONE/OFF3
          OFDIF6 = OFF6/(OFF6 - ON6)
          OFDIF3 = OFF3/(OFF3 - ON3)
          ONOFF6 = RECOF6/ON6
          ONOFF3 = RECOF3/ON3
        ELSE
          ONOFF6 = RECOF6*RECOF6
          ONOFF3 = RECOF6
        END IF
      END IF

!=======================================================================
!  Expand control section
!-------------------------------------------------------------------
! (disable expand when debug is active)

!-------------------------------------------------------------------
! Do PBOUND expansion of code
#if KEY_PBOUND == 1 && KEY_DEBUG == 0
      
      IF(QBOUN) THEN
#define EDIFF_PBOUND 1
#include "ediff.inc"
#undef EDIFF_PBOUND
      ELSE
#include "ediff.inc"
      ENDIF
      
#else  /* KEY_PBOUND == 1 && KEY_DEBUG == 0 */
      
#undef EDIFF_PBOUND
#define EDIFF_DEBUG
#include "ediff.inc"
#undef EDIFF_DEBUG
      
#endif  /* KEY_PBOUND == 1 && KEY_DEBUG == 0 */
      RETURN
   END SUBROUTINE NBONDD
end module ediff
