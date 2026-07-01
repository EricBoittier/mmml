module travelsub
  ! TRAVEL file-names facilitate compilation with previous versions of CHARMM.

  !******************************************************************************
  !                                                                             *
  !           TReK: a program for Trajectory REfinement and Kinematics.         *
  !                                                                             *
  !                        Version 2.10 , July  5-2003.                         *
  !                                                                             *
  !******************************************************************************
  !         Please report problems or send suggestions to the author:           *
  !                                                                             *
  !                              Stefan Fischer                                 *
  !                         Tel. (49)6221-548879                                *
  !                e-mail: stefan.fischer@iwr.uni-heidelberg.de                 *
  !                                                                             *
  !            Check for application examples and bug-fixes under :             *
  !                                                                             *
  !            http://www.iwr.uni-heidelberg.de/iwr/biocomp/fischer             *
  !                                                                             *
  !******************************************************************************

  use travelsub2

contains

#if KEY_TRAVEL==0 /*travel*/
  SUBROUTINE NULL_AI
    return
  end SUBROUTINE NULL_AI

#else /* (travel)*/

  SUBROUTINE CPRRUN( LMVCPR,IDXSAD, NMAXP,NPOINT,XBAS,YBAS,ZBAS, &
       TOTUSD,SERIAL,IDXPRO,IDPREC,IDNEXT,LENSEG, &
       SEGSTP, PTENE,PTGRAD,STPMAX,ENEMAX,QUP,QDOWN, &
       NEXMIN,PREMIN,NLINMN,DYNBRA, NFREEP,IFREEP, &
       NMAXI,SRTIDX,SRTENE, NEIBOR,NEIBR1,NEIBR2, &
       ISLCT,XREF,YREF,ZREF,IMGAXI,IMGROT, &
       ITEMP,DTEMP, NATOM,NFIXED,VERBEU,NDISPL, &
       XDUM,YDUM,ZDUM,XS0,YS0,ZS0, X,Y,Z )



    use chm_kinds
    use chm_types
    use consta
    use contrl
    use dimens_fcm
    use energym
    use number
    use param_store, only: set_param
    use stream
    use trek1
    use trek2

#if KEY_PARALLEL==1
    use parallel
#endif 

    implicit none

    LOGICAL      LMVCPR

    INTEGER      IDXSAD

    INTEGER      NMAXP,NPOINT
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER      TOTUSD,SERIAL(*),IDXPRO,IDPREC(*),IDNEXT(*)
    real(chm_real)       LENSEG(*),SEGSTP(*)

    real(chm_real)       PTENE(*),PTGRAD(*)
    INTEGER      STPMAX(*)
    real(chm_real)       ENEMAX(*)
    LOGICAL      QUP(*),QDOWN(*)
    INTEGER      NEXMIN(*),PREMIN(*),NLINMN(*)
    real(chm_real)       DYNBRA(*)

    INTEGER      NFREEP,IFREEP(*)

    INTEGER      NMAXI,SRTIDX(*)
    real(chm_real)       SRTENE(*)

    INTEGER      NEIBOR,NEIBR1(*),NEIBR2(*)

    INTEGER      ISLCT(*)
    real(chm_real)       XREF(*),YREF(*),ZREF(*)
    INTEGER      IMGAXI
    LOGICAL      IMGROT

    INTEGER      NATOM,NFIXED

    INTEGER      VERBEU,NDISPL

    INTEGER      ITEMP(*)
    real(chm_real)       DTEMP(*)

    real(chm_real)       XDUM(*),YDUM(*),ZDUM(*),XS0(*),YS0(*),ZS0(*)

    real(chm_real)       X(*),Y(*),Z(*)

    LOGICAL      QERROR
    INTEGER      I,J,K,L,i_temp
    INTEGER      IDUM1,IDUM2,IDUM3,IDX,IDXNEW,IDXPRE,IDXFOL
    real(chm_real)       SNORM,JRN,DDUM1,DDUM2,DDUM3,DDUM4,DDUM5,DDUM6

    INTEGER      IDIM
    real(chm_real)       DDIM

    LOGICAL      QPREMA,QMOVD1,QMOVD2
    INTEGER      ICYCLE

    INTEGER      NTERPL
    LOGICAL      LFOUND

    LOGICAL      LPAKED

    LOGICAL      QUSMIN

    LOGICAL      LBOTH
    INTEGER      II,LOCDEC,NSTEP
    real(chm_real)       MXPARA

    INTEGER      FRAM2,NEWCYC,IHIST(HISLEN)
    real(chm_real)       EHIST(HISLEN),GHIST(HISLEN),NEWMIN,NEWMAX
    real(chm_real)       OLDMIN,OLDMAX
    SAVE  IHIST,EHIST,GHIST

    INTEGER      NCOLUM,NLINES, POS1ST,POSLST,IDX1ST,NSHOWP

    INTEGER      I7050,I7070,I7090,I7110,I7130,I7150

    real(chm_real), PARAMETER :: PT03=PT01*THREE,PT09=PT01*NINE


    IDIM  = 3*(NATOM - NFIXED)
    DDIM  = IDIM

    CALL W0(' ')
    CALL W0(' Refining the path :')
    CALL W0(' -------------------')
    CALL WI( ' # of path segments = ',NPOINT-1)
    CALL WI( ' # of segments with at least 1 local max. = ',NMAXI)
    IF (VERBEU >= 2) THEN
       CALL W0(' ')
       CALL W0(' Sorted list of maxima and their segment index :')
       CALL W0(' -----------------------------------------------')
       DO I = 1,NMAXI
          WRITE(OUTU,1070) SRTENE(I), SRTIDX(I)
1070      FORMAT(' E = ',F15.7,'    Index  = ',I7)
       ENDDO
    ENDIF


    ICYCLE = 1
11071 CONTINUE

    IF (NMAXI  <  0) THEN
       CALL W0(' ')
       CALL WI(' NMAXI < 0 , bug1 in CPRRUN !!!  NMAXI = ', NMAXI )
       CALL W0(' ')
    ENDIF

    IF (NMAXI  ==  0) THEN
       CALL W0(' ')
       CALL W0(' Exiting CPR: all maxima along the path are'// &
            ' saddle-points.')
       CALL WI(' After completing a number of CPR-cycles =',ICYCLE-1)
       CALL W0(' =========================================')
       CALL W0(' The path is FULLY refined, because')
       IF (IDXSAD > 0) THEN
          CALL W0( ' all energy-peaks along the path satisfy'// &
               ' the used saddle-point criteria.')
       ELSE
          CALL W0(' there are no energy maxima along the path.')
       ENDIF
       GOTO  75
    ENDIF


    IF ( LHISAD .AND. IDXSAD > 0 )then
       if( PTENE(IDXSAD) > SRTENE(1) .AND. NMAXI > 0 ) THEN
          CALL W0(' ')
          CALL W0(' Exiting CPR: the GLOBAL energy-maximum along'// &
               ' the path is a saddle-point.')
          GOTO  71
       ENDIF
    endif

    IF (ICYCLE > NCYCLE) THEN
       CALL W0(' ')
       CALL W0(' Exiting CPR: did all requested NCYCLE cycles.')
       GOTO  71
    ENDIF


    IF (ECALLS >= MAXCAL) THEN
       CALL W0(' ')
       CALL WI(' Exiting CPR: number of new energy-calls'// &
            ' exceed NECALL =', NECALL)
       GOTO  71
    ENDIF


    LMVCPR = .TRUE.
    if( func7110() ) goto 75
    TOTCYC = TOTCYC + 1
    if( func7150() ) goto 75

    DO I = 1, HISLEN-1
       IHIST(I) = IHIST(I+1)
       EHIST(I) = EHIST(I+1)
       GHIST(I) = GHIST(I+1)
    ENDDO
    IHIST(HISLEN) =        NEWCYC
    EHIST(HISLEN) =  PTENE(NEWCYC)
    GHIST(HISLEN) = PTGRAD(NEWCYC)

    ICYCLE = ICYCLE+1
    GOTO  11071
71  CONTINUE


    ICYCLE = ICYCLE-1
    CALL WI(' After completing a number of CPR-cycles =', ICYCLE )
    CALL W0(' =============================================')
    CALL W0(' The path is NOT fully refined for the used'// &
         ' saddle-point criteria.')

75  CALL W0(' ')
    CALL W0(' Since the last time CPR was initialized :')
    CALL W0(' -----------------------------------------')
    CALL WI( ' The total number of CPR-cycles           =',TOTCYC)
    CALL WI( ' The total number of line-minimizations   =',TOTMIN)
    CALL WI( ' Total of energy calls (w. & wo. grad.)   =',ECALLS)
    CALL WI( ' Total updates of the non-bond list       =',TOTUPD)

    IF (INSMIN > 0) &
         CALL WI( ' # of flanking minima inserted into path  =',INSMIN)
    IF (RMVMIN > 0) &
         CALL WI( ' # of such pseudo-minima later removed    =',RMVMIN)

    IF (TOTLUP > 0) &
         CALL WI( ' Total occurences of looping              =',TOTLUP)

    IF (TOTOSC > 0) THEN
       CALL WI( ' Total occurences of algorithm-oscillation=',TOTOSC)
       IF (PROJIN /= ONE) THEN
          CALL W0( ' The final oscillation-reduced projection '// &
               'tolerances are :')
          CALL WR( '         PRTOL1 =', PRTOL1*(DDIM-ONE))
          CALL WR( '         PRTOL2 =', PRTOL2*(DDIM-ONE))
       ENDIF
    ENDIF

    IF ( ABS(ORISTP-STEPSZ) > RSMALL ) THEN
       CALL W0(' ')
       CALL WR( ' The final reduced STEP-SiZe = ', STEPSZ/SQRT(DDIM))
       CALL W0(' Set that value with the STEP keyword in the first')
       CALL W0(' (and ONLY the first!) call to CPR when continuing')
       CALL W0(' the refinement in a future TReK calculation !!!')
    ENDIF

    IF (NMAXI  <  0) THEN
       CALL W0(' ')
       CALL WI(' NMAXI < 0 , bug2 in CPRRUN !!!  NMAXI = ', NMAXI )
       CALL W0(' ')
    ENDIF

    CALL W0(' ')
    CALL W0(' Description of the resulting path :')
    CALL W0(' -----------------------------------')
    CALL WI(' The number of path-points             NPOINT =', NPOINT)
    CALL WR(' The length of the path (sum of all segments) =', &
         PATHLN(NPOINT,IDNEXT,LENSEG)/SQRT(DDIM) )

    IF (IDXSAD > 0) THEN
       CALL W0(' ')
       CALL WI(' The highest point satisfying the'// &
            ' saddle-criteria has index=', IDXSAD)
       CALL WR(' Its Energy =', PTENE(IDXSAD))
       CALL WR(' Its RMS gradient =', PTGRAD(IDXSAD))
       IF ( PTENE(IDXSAD) > SRTENE(1) .OR. NMAXI == 0 ) THEN
          CALL W0(' It is the GLOBAL energy max. along the path.')
       ELSE
          CALL W0(' It is NOT the path`s global energy-maximum.')
       ENDIF
    ELSEIF ( NMAXI > 0 ) THEN
       CALL W0(' ')
       CALL W0(' NO saddle-point has been found so far, which')
       CALL W0(' would satisfy the saddle-point criteria.')
    ENDIF

    IF ( NMAXI > 0 ) THEN
       CALL WI(' The number of remaining peaks (non-saddle) =', NMAXI )
       CALL WR(' of which the highest peak-energy =', SRTENE(1))
       CALL WI(' is found on segment index        =', SRTIDX(1))
    ENDIF

    CALL LISTSD( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )
    CALL LISTMN( NPOINT,IDNEXT,NLINMN, PTENE,PTGRAD )

    CALL set_param('SADI', IDXSAD )
    CALL set_param('SADO', 0 )

    IF (IDXSAD > 0) THEN
       call set_param('SADE',  PTENE(IDXSAD) )
       CALL set_param('SADO', POSITN(IDXSAD, NPOINT,IDNEXT) )
    ENDIF

    RETURN
    
    !-------------------- Contained recursive subroutines ----------------------
  contains

    logical function func7090() result(err)

      TOTLUP = TOTLUP + 1
      LBOTH  = .FALSE.

      MXPARA = PARAMX( LENSEG(IDXPRE),LENSEG(IDX), &
           PTENE(IDXPRE),PTENE(IDX),PTENE(IDXFOL) )
150   II = IDX
      IF (MXPARA < ZERO)  II = IDXPRE

      NTERPL = MAX( NINT( LENSEG(II)/SEGSTP(II) ) , 1 )
      LOCDEC = 4
      IF (NTERPL > 1)  LOCDEC = 2
      CALL DSUM2( XS0,YS0,ZS0, &
           XBAS(IDNEXT(II))%a,YBAS(IDNEXT(II))%a,ZBAS(IDNEXT(II))%a, &
           -ONE, XBAS(II)%a,YBAS(II)%a,ZBAS(II)%a, NATOM )
      SNORM = LENSEG(II)
      DDUM1 = ZERO
      DDUM2 = ONE/(LOCDEC*NTERPL)
      NSTEP  = LOCDEC*NTERPL - 1
      IF (MXPARA < ZERO)  DDUM2 = DDUM2*NSTEP
      DDUM3 = ONE
      DDUM4 = PTENE(II)
      DDUM6 = PTENE(IDNEXT(II))
      CALL SERCH1( DDUM1,DDUM2,DDUM3, DDUM4,DDUM5,DDUM6, &
           E1DIM,XBAS(II)%a,YBAS(II)%a,ZBAS(II)%a, &
           XS0,YS0,ZS0, SNORM, NATOM, &
           NSTEP, .FALSE., .FALSE.,.TRUE.,.FALSE., LFOUND )

      IF (.NOT.LFOUND) THEN
         IF (.NOT.LBOTH) THEN
            MXPARA = -MXPARA
            LBOTH = .TRUE.
            GOTO  150
         ENDIF

         if( func7130() ) then
            err=.true.
            return
         endif
      ELSE
         IF (II == IDXPRE)  LPAKED = .TRUE.

         IF (.NOT.LPAKED) THEN
            NMAXI = NMAXI - 1
            DO J=1, NMAXI
               SRTENE(J) = SRTENE(J+1)
               SRTIDX(J) = SRTIDX(J+1)
            ENDDO
            LPAKED = .TRUE.
         ENDIF

         IDX    = II
         IDXPRE = IDPREC(IDX)
         IDXFOL = IDNEXT(IDX)

         IDXNEW = GETIDX( QERROR, NFREEP,IFREEP, NPOINT,NMAXP, &
              XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
         if(qerror)then
            err=.true.
            return
         endif

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Simple looping:        cycle & added point-IDX =', &
                 ICYCLE, IDXNEW )
         ENDIF


         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XS0,YS0,ZS0,LENSEG(IDX), &
              PRTOL2, STPTOL,ENETOL,GRATOL,MAXTOL, BRASTP, &
              BRAMAG,ATOMAX,VERBEU,NATOM,IDIM, PTGRAD(IDXNEW), &
              XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
              LINMIN, LXEVAL, DELTAS, SADGRD, BRASCA,SADMIN, &
              DYNBRA(IDXNEW),QMOVD2, MODXIT, .FALSE., TOTMIN, &
              PTENE(IDXNEW), NLINMN(IDXNEW), QPREMA )

         IF (LORIEN)        CALL ORIENT( IDXNEW, &
              XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
              XREF,YREF,ZREF, NATOM, &
              NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         IDPREC(IDXNEW) = IDX
         IDNEXT(IDXNEW) = IDXFOL
         IDPREC(IDXFOL) = IDXNEW
         IDNEXT(IDX)    = IDXNEW

         LASTMV = IDXNEW
         CALL ADDPRL(SERIAL(IDX),SERIAL(IDXNEW), NEIBR1,NEIBR2,NEIBOR)
         CALL ADDPRL(SERIAL(IDXNEW),SERIAL(IDXFOL),NEIBR1,NEIBR2,NEIBOR)

         I = IDX
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXNEW
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         NEWCYC = IDXNEW
      ENDIF

      IF ( REDLUP > 0 .AND. STEPSZ/SQRT(TWO) >= STEPLW ) THEN

         IF ( MOD(TOTLUP,REDLUP) == 0 ) THEN

            STEPSZ = STEPSZ/SQRT(TWO)
            IF (VERBEU >= 1) THEN
               CALL W0(' ')
               CALL WI(' Rescanning traj. with smaller STEPSZ'// &
                    ' after looping-occurence #', TOTLUP)
            ENDIF

            LSCAN1 = .FALSE.
            LSCAN2 = .FALSE.
            LSCAN3 = .FALSE.

            CALL SCANPATH( NPOINT,IDXPRO,IDXSAD,SADMIN,SADGRD,NLINMN, &
                 IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
                 LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
                 SRTENE,SRTIDX,SEGSTP,PTGRAD, &
                 XBAS,YBAS,ZBAS, NATOM,VERBEU)

            LSCAN1 = .FALSE.
            LSCAN2 = .FALSE.
            LSCAN3 = .TRUE.
         ENDIF
      ENDIF
      err=.false.
      return
    end function func7090
    
    logical function func7130() result(err)

      IF ( SERIAL(IDXPRE) /= CPLUP1 .OR. &
           SERIAL(IDXFOL) /= CPLUP2  ) THEN
         LTMPRJ = .TRUE.
         CPLUP1 = SERIAL(IDXPRE)
         CPLUP2 = SERIAL(IDXFOL)

         NPOINT = NPOINT - 1
         NFREEP = NFREEP + 1
         IFREEP(NFREEP) = IDX
         IDNEXT(IDXPRE) = IDXFOL
         IDPREC(IDXFOL) = IDXPRE

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         LASTMV = 0
         IF ( SERIAL(IDX) < 0 )  RMVMIN = RMVMIN + 1

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Complex looping:     cycle & removed point-IDX =', &
                 ICYCLE, IDX )
         ENDIF
         IF (VERBEU >= 2) THEN
            CALL W0('      -> increasing PRTOL2 for one cycle only.')
         ENDIF

      ELSE

         IF (IDXPRE == 1.AND.IDXFOL == IDXPRO) THEN
            CALL W0(' ')
            CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            CALL W0(' Aborting due to fatal looping over segment going')
            CALL W0('    from the REACTANT to the PRODUCT structures')
            CALL WI('    if removing the point    IDX =',IDX)
            CALL WI('    after doing number of editing cycle =', ICYCLE)
            CALL W0(' ================================================')
            err=.true.
            return
         ELSEIF ( IDXPRE == 1 .OR. &
              ( ABS(NLINMN(IDXPRE)) > ABS(NLINMN(IDXFOL)) ) ) THEN
            II = IDXFOL
         ELSEIF ( IDXFOL == IDXPRO .OR. &
              ( ABS(NLINMN(IDXPRE)) < ABS(NLINMN(IDXFOL)) ) ) THEN
            II = IDXPRE
         ELSEIF (PTENE(IDXPRE) > PTENE(IDXFOL)) THEN
            II = IDXPRE
         ELSE
            II = IDXFOL
         ENDIF

         LPAKED = .TRUE.

         IDX = II
         IDXPRE = IDPREC(IDX)
         IDXFOL = IDNEXT(IDX)

         NPOINT = NPOINT - 1
         NFREEP = NFREEP + 1
         IFREEP(NFREEP) = IDX
         IDNEXT(IDXPRE) = IDXFOL
         IDPREC(IDXFOL) = IDXPRE

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDXFOL),NEIBR1,NEIBR2,NEIBOR)
         LASTMV = 0
         IF ( SERIAL(IDX) < 0 )  RMVMIN = RMVMIN + 1

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Complex looping: cycle & removed adjacent point=', &
                 ICYCLE, IDX )
         ENDIF
         IF (VERBEU >= 2) THEN
            CALL W0('      because increasing PRTOL2 did not help.')
         ENDIF
      ENDIF

      DYNBRA(IDX) =  FIRSTP
      NEWCYC = IDX
      err=.false.
      return
    end function func7130

    logical function func7110() result(err)
      OSSKIP = OSSKIP - 1
      IF (OSSKIP < 0) THEN
         FRAM2  = 2*FRAME
         NEWMIN =  RBIG
         NEWMAX = -RBIG
         DO J = HISLEN-FRAME+1, HISLEN
            IF ( EHIST(J) < NEWMIN )  NEWMIN = EHIST(J)
            IF ( EHIST(J) > NEWMAX )  NEWMAX = EHIST(J)
         enddo
         OLDMIN =  RBIG
         OLDMAX = -RBIG
         DO J = HISLEN-FRAM2+1, HISLEN-FRAME
            IF ( EHIST(J) < OLDMIN )  OLDMIN = EHIST(J)
            IF ( EHIST(J) > OLDMAX )  OLDMAX = EHIST(J)
         enddo
         DDUM1 = MIN(OLDMAX-OLDMIN,NEWMAX-NEWMIN)
         IF (ABS(NEWMIN-OLDMIN)/DDUM1 > OSCTOL)  GOTO  189
         IF (ABS(NEWMAX-OLDMAX)/DDUM1 > OSCTOL)  GOTO  189
         NEWMIN =  RBIG
         NEWMAX = -RBIG
         DO J = HISLEN-FRAME+1, HISLEN
            IF ( GHIST(J) < NEWMIN )  NEWMIN = GHIST(J)
            IF ( GHIST(J) > NEWMAX )  NEWMAX = GHIST(J)
         enddo
         OLDMIN =  RBIG
         OLDMAX = -RBIG
         DO J = HISLEN-FRAM2+1, HISLEN-FRAME
            IF ( GHIST(J) < OLDMIN )  OLDMIN = GHIST(J)
            IF ( GHIST(J) > OLDMAX )  OLDMAX = GHIST(J)
         enddo
         DDUM1 = MIN(OLDMAX-OLDMIN,NEWMAX-NEWMIN)
         IF (ABS(NEWMIN-OLDMIN)/DDUM1 > OSCTOL)  GOTO  189
         IF (ABS(NEWMAX-OLDMAX)/DDUM1 > OSCTOL)  GOTO  189
         loop181:DO J = HISLEN-FRAME+1, HISLEN
            DO K = HISLEN-FRAM2+1, HISLEN-FRAME
               IF ( IHIST(J) == IHIST(K) )  cycle loop181
            enddo
            GOTO  189
         enddo loop181
         loop183:DO K = HISLEN-FRAM2+1, HISLEN-FRAME
            DO J = HISLEN-FRAME+1, HISLEN
               IF ( IHIST(J) == IHIST(K) ) cycle loop183
            enddo
            GOTO  189
         enddo loop183

         CALL W0(' ')
         CALL WI( '!! Oscillation detected !! TOTOSC =', TOTOSC)
         TOTOSC = TOTOSC + 1

         IF (TOTOSC > OSCMAX) THEN
            CALL W0(' ')
            CALL W0('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            CALL W0('  Exceding allowed number of oscillations OSCMAX.')
            CALL WI(' Aborting after number of editing cycle =', ICYCLE)
            CALL W0(' ================================================')
            err=.true.
            return
         ENDIF

         IF ( STEPSZ/SQRT(TWO) >= STEPLW .OR. &
              PRTOL1 < PT03 .OR. PRTOL2 < PT09 ) THEN

            CALL WR('Increasing PRTOL1/2 by a factor of ', PROJIN)
            PRTOL1 = MIN( PROJIN*PRTOL1, PT03 )
            PRTOL2 = MIN( PROJIN*PRTOL2, PT09 )
            CALL W2R('         PRTOL1 & PRTOL2 =', PRTOL1,PRTOL2)

            IF ( STEPSZ/SQRT(TWO) >= STEPLW ) THEN

               CALL W0('         Rescanning traj. with smaller STEPSZ.')
               STEPSZ = STEPSZ/SQRT(TWO)

               LSCAN1 = .FALSE.
               LSCAN2 = .FALSE.
               LSCAN3 = .FALSE.

               CALL SCANPATH( NPOINT,IDXPRO,IDXSAD,SADMIN,SADGRD,NLINMN, &
                    IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
                    LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
                    SRTENE,SRTIDX,SEGSTP,PTGRAD, &
                    XBAS,YBAS,ZBAS, NATOM,VERBEU )

               LSCAN1 = .FALSE.
               LSCAN2 = .FALSE.
               LSCAN3 = .TRUE.
            ENDIF

            OSSKIP = FRAME

         ELSE

            CALL W0(' ')
            CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            CALL W0('     Not allowed to reduce STEPSZ below STEPLW ')
            CALL W0('          or to increase PRTOL1/2 further.')
            CALL WI(' Aborting after number of editing cycle =', ICYCLE)
            CALL W0(' ================================================')
            err=.true.
            return
         ENDIF
      ENDIF
189   continue
      err=.false.
      return
    end function func7110
    
    logical function func7150() result(err)
      IDX    = SRTIDX(1)
      IDXPRE = IDPREC(IDX)
      IDXFOL = IDNEXT(IDX)
      LPAKED = .FALSE.

      IF ( STPMAX(IDX) > 0 .OR. NLINMN(IDX) < 0 ) THEN

         IDXNEW = GETIDX( QERROR, NFREEP,IFREEP, NPOINT,NMAXP, &
              XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
         if(qerror) then
            err=.true.
            return
         endif
         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Adding path-point:     cycle & added point-IDX =', &
                 ICYCLE, IDXNEW )
         ENDIF

         CALL DSUM2( XS0, YS0, ZS0, &
              XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
              -ONE , XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, NATOM )

         DDUM1 = ONE/NINT( LENSEG(IDX)/SEGSTP(IDX) )
         DDUM2 = ABS(STPMAX(IDX))*DDUM1
         DDUM3 = DDUM2 + DDUM1
         DDUM1 = DDUM2 - DDUM1

         DDUM4 = PRTOL2
         IF (LTMPRJ) THEN
            DDUM4 = MIN( THREE*PRTOL2, PT09 )
            LTMPRJ = .FALSE.
         ENDIF

         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XS0,YS0,ZS0,LENSEG(IDX), &
              DDUM4, STPTOL,ENETOL,GRATOL,MAXTOL, BRASTP,BRAMAG, &
              ATOMAX,VERBEU,NATOM,IDIM, PTGRAD(IDXNEW), &
              XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
              LINMIN, LXEVAL, DELTAS, SADGRD, BRASCA,SADMIN, &
              DYNBRA(IDXNEW),QMOVD2, MODXIT, .FALSE., TOTMIN, &
              PTENE(IDXNEW), NLINMN(IDXNEW), QPREMA )

         IF (LORIEN)        CALL ORIENT( IDXNEW, &
              XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
              XREF,YREF,ZREF, NATOM, &
              NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         IDPREC(IDXNEW) = IDX
         IDNEXT(IDXNEW) = IDXFOL
         IDPREC(IDXFOL) = IDXNEW
         IDNEXT(IDX)    = IDXNEW

         LASTMV = IDXNEW
         CALL ADDPRL(SERIAL(IDX),SERIAL(IDXNEW), NEIBR1,NEIBR2,NEIBOR)
         CALL ADDPRL(SERIAL(IDXNEW),SERIAL(IDXFOL),NEIBR1,NEIBR2,NEIBOR)

         I = IDX
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXNEW
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         NEWCYC = IDXNEW
      ELSE

         CALL DSUM2( X,Y,Z, XBAS(IDPREC(IDX))%a, &
              YBAS(IDPREC(IDX))%a,ZBAS(IDPREC(IDX))%a, &
              -ONE, XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, NATOM )

         CALL DSUM2( XDUM,YDUM,ZDUM, XBAS(IDNEXT(IDX))%a, &
              YBAS(IDNEXT(IDX))%a,ZBAS(IDNEXT(IDX))%a,  -ONE, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, NATOM )

         CALL DSUM1( XS0,YS0,ZS0, LENSEG(IDPREC(IDX))/LENSEG(IDX), &
              XDUM, YDUM, ZDUM, -ONE, X,Y,Z, NATOM )


         SNORM = DSDOT1( XS0,YS0,ZS0, NATOM )

         DDUM1 = DDOT2(X,Y,Z, XS0,YS0,ZS0, NATOM)
         JRN = NINT( LENSEG(IDPREC(IDX))/SEGSTP(IDPREC(IDX)) )
         DDUM1 = DDUM1 / ( SNORM * MAX(ONE,JRN))
         DDUM2 = ZERO
         DDUM3 = DDOT2( XDUM,YDUM,ZDUM, XS0,YS0,ZS0, NATOM)
         JRN = NINT( LENSEG(IDX)/SEGSTP(IDX) )
         DDUM3 = DDUM3 / ( SNORM * MAX(ONE,JRN))

         SNORM  = SQRT(SNORM)
         DDUM5 = PTENE(IDX)

         CALL SERCH1( DDUM1,DDUM2,DDUM3, DDUM4,DDUM5,DDUM6, &
              E1DIM,XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XS0,YS0,ZS0, SNORM, NATOM, &
              NTANGT, .FALSE., .TRUE.,.FALSE.,.TRUE., LFOUND )

         QMOVD1 = (DDUM2 /= ZERO)

         IF (LFOUND) THEN

            IF (VERBEU >= 1) THEN
               CALL W0(' ')
               CALL W2I('Refining path-point:         cycle' &
                    //' & point-IDX =', ICYCLE, IDX )
            ENDIF

            CALL SADLE1( DDUM1,DDUM2,DDUM3, &
                 XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
                 XS0,YS0,ZS0,SNORM,PRTOL1, STPTOL,ENETOL, &
                 GRATOL,MAXTOL, BRASTP,BRAMAG, &
                 ATOMAX, VERBEU,NATOM,IDIM, PTGRAD(IDX), &
                 XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
                 LINMIN, LXEVAL, DELTAS, SADGRD, BRASCA, SADMIN, &
                 DYNBRA(IDX), QMOVD2, MODXIT, .FALSE., TOTMIN, &
                 PTENE(IDX), NLINMN(IDX), QPREMA )

            IF ((.NOT.QMOVD1).AND.(.NOT.QMOVD2)) THEN
               IF (NLINMN(IDX) > 0) THEN
                  NLINMN(IDX) = 2*IDIM + NLINMN(IDX)
                  CALL W0('!!! Warning: peak could not be moved !!!')
                  CALL WI('Tentatively flagged as'// &
                       ' saddle-point: IDX=', IDX)
               ELSE
                  CALL W0(' ')
                  CALL W0(' !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                  CALL WI(' Aborting: could not refine path-point IDX =', IDX)
                  CALL WI(' after doing number of editing cycle =', ICYCLE)
                  CALL W0(' since all line-extremizations exit before the')
                  CALL W0(' point is moved.')
                  CALL W0(' Use more stringent extremization exit-criteria.')
                  CALL W0(' ================================================')
                  err=.true.
                  return
               ENDIF
            ELSEIF (QPREMA.AND.NLINMN(IDX) > 0) THEN
               NLINMN(IDX) = 2*IDIM + NLINMN(IDX)
               CALL WI('Tentatively flagged as saddle-point: IDX=', &
                    IDX )
            ENDIF

            NEWCYC = IDX
            IF (IDX /= LASTMV) THEN
               CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
               TOTUSD = TOTUSD + 1
               SERIAL(IDX) = TOTUSD
               LASTMV = IDX
               CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDX),NEIBR1,NEIBR2,NEIBOR)
               CALL ADDPRL(SERIAL(IDX),SERIAL(IDXFOL),NEIBR1,NEIBR2,NEIBOR)
            ENDIF

            I = IDXPRE
            CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
                 NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
                 IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
                 LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

            I = IDX
            CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
                 LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
                 SRTENE,SRTIDX,SEGSTP,PTGRAD, &
                 XBAS,YBAS,ZBAS, NATOM,VERBEU )
            CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
                 NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

            I = IDXFOL
            CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
                 NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )
         ELSE

            if( func7050() ) then
               err=.true.
               return
            endif

         ENDIF
      ENDIF

      IF (.NOT.LPAKED) THEN
         NMAXI = NMAXI - 1
         DO J=1, NMAXI
            SRTENE(J) = SRTENE(J+1)
            SRTIDX(J) = SRTIDX(J+1)
         ENDDO
         LPAKED = .TRUE.
      ENDIF

      CALL SETIHI( IDXSAD,NPOINT,IDNEXT,PTGRAD,SADGRD,QUP,QDOWN, &
           NLINMN,SADMIN,PTENE )

      IF (VERBEU >= 3) THEN
         WRITE(OUTU,'(A12,I6,5X,A20,I7,4X,A20,I7)') &
              ' Line-min. =', TOTMIN - LSTOT1, &
              '          E calls  =', ECALLS - LSTOT2
      ENDIF

      call sub7070
      LSTOT1 = TOTMIN
      LSTOT2 = ECALLS
      err=.false.
      return
    end function func7150

    subroutine sub7070()

#if KEY_PARALLEL==1
      IF (MYNOD == 0) THEN
#endif 

         NCOLUM = INT( NDISPL/10 )

         IF (DISPLP > 0.AND.NPOINT >= 2.AND.NCOLUM > 0) THEN
            IDUM1 = POSITN( IDPREC(IDXFOL), NPOINT,IDNEXT )
            POS1ST = MAX(   1  , IDUM1 - INT(DISPLP/2) )
            POSLST = MIN(NPOINT, POS1ST + DISPLP - 1 )
            POS1ST = MAX(   1  , POSLST - DISPLP + 1 )
            IDX1ST = PINDEX( POS1ST, NPOINT,IDNEXT)
            NSHOWP = POSLST-POS1ST+1

            I = IDX1ST
            J = IDX1ST
            K = IDX1ST
            IDUM2 = 1 - NCOLUM
            NLINES = INT( (NSHOWP-1)/NCOLUM ) + 1
            DO L=1,NLINES
               IDUM2 = IDUM2 + NCOLUM
               IDUM3 = MIN( IDUM2+NCOLUM-1 , NSHOWP )
               DO IDUM1=IDUM2,IDUM3
                  ITEMP(IDUM1) = I
                  I = IDNEXT(I)
               ENDDO
               WRITE(OUTU,1111) ( ITEMP(IDUM1), IDUM1=IDUM2,IDUM3 )
               DO IDUM1=IDUM2,IDUM3
                  DTEMP(IDUM1) = PTENE(J)
                  J = IDNEXT(J)
               ENDDO
               WRITE(OUTU,1112) ( DTEMP(IDUM1), IDUM1=IDUM2,IDUM3 )
               DO IDUM1=IDUM2,IDUM3
                  DTEMP(IDUM1) = PTGRAD(K)
                  K = IDNEXT(K)
               ENDDO
               WRITE(OUTU,1113) ( DTEMP(IDUM1), IDUM1=IDUM2,IDUM3 )
               WRITE(OUTU,1114) ( '----------' , IDUM1=IDUM2,IDUM3 )
            ENDDO
         ENDIF

1111     FORMAT(   I6,3X,100(1X,I6,3X)    )
1112     FORMAT(1PG9.4E1,100(1X,1PG9.4E1) )
1113     FORMAT(  1PG9.2,100(1X,1PG9.3)   )
1114     FORMAT(      A9,100(A10)         )

#if KEY_PARALLEL==1
      ENDIF
#endif 
      return
    end subroutine sub7070

    !========================== 7050 =====================================
    logical function func7050() result(err)


      DYNBRA(IDX) =  FIRSTP
      QUSMIN = .TRUE.
      IF ( MODREM <= 0 .AND. &
           (PREMIN(IDX) > 0 .OR. NEXMIN(IDX) > 0) ) THEN
         QUSMIN = .FALSE.
         DO I = 1,NEIBOR
            IF ( SERIAL(IDXPRE) == NEIBR1(I) .AND. &
                 SERIAL(IDXFOL) == NEIBR2(I)    )  QUSMIN = .TRUE.
         ENDDO
      ENDIF

      IF (QUSMIN.AND.PREMIN(IDX) > 0.AND.NEXMIN(IDX) > 0) THEN

         IDXNEW = GETIDX( QERROR, NFREEP,IFREEP, NPOINT,NMAXP, &
              XBAS,YBAS,ZBAS, NATOM,TOTUSD,SERIAL,NLINMN )
         if(qerror)then
            err=.true.
            return
         endif
         NEWCYC = IDXNEW

         CALL DSUM2( XS0, YS0, ZS0, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%A, -ONE , &
              XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
              NATOM )

         DDUM1 = ONE/NINT( LENSEG(IDXPRE)/SEGSTP(IDXPRE) )
         DDUM2 = PREMIN(IDX)*DDUM1
         DDUM3 = DDUM2 + DDUM1
         DDUM1 = DDUM2 - DDUM1

         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
              XS0,YS0,ZS0,LENSEG(IDXPRE),PRTOL1, STPTOL, &
              ENETOL,GRATOL,MAXTOL, BRASTP,BRAMAG,ATOMAX, &
              VERBEU,NATOM,IDIM, PTGRAD(IDXNEW), &
              XBAS(IDXNEW)%a,YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, &
              ABS(MODREM), LXEVAL,DELTAS,SADGRD, BRASCA, 1, &
              DYNBRA(IDXNEW),QMOVD2, MODXIT, .TRUE., TOTMIN, &
              PTENE(IDXNEW), 0, QPREMA )

         NLINMN(IDXNEW) = 0

         IF (LORIEN)     CALL ORIENT( IDXNEW, XBAS(IDXNEW)%a, &
              YBAS(IDXNEW)%a,ZBAS(IDXNEW)%a, XREF,YREF,ZREF, &
              NATOM, NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         CALL DSUM2( XS0, YS0, ZS0, &
              XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
              -ONE , XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, NATOM )

         DDUM1 = ONE/NINT( LENSEG(IDX)/SEGSTP(IDX) )
         DDUM2 = NEXMIN(IDX)*DDUM1
         DDUM3 = DDUM2 + DDUM1
         DDUM1 = DDUM2 - DDUM1

         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XS0,YS0,ZS0,LENSEG(IDX),PRTOL1, STPTOL,ENETOL, &
              GRATOL,MAXTOL, BRASTP,BRAMAG,ATOMAX, &
              VERBEU,NATOM,IDIM, PTGRAD(IDX), &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              ABS(MODREM), LXEVAL, DELTAS,SADGRD,BRASCA,1, &
              DYNBRA(IDX),QMOVD2, MODXIT, .TRUE., TOTMIN, &
              PTENE(IDX), 0, QPREMA )

         NLINMN(IDX) = 0

         IF (LORIEN)  CALL ORIENT( IDX, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XREF,YREF,ZREF, NATOM, &
              NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         IDNEXT(IDXPRE) = IDXNEW
         IDPREC(IDXNEW) = IDXPRE
         IDNEXT(IDXNEW) = IDX
         IDPREC(IDX)    = IDXNEW

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDXNEW
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDX
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         SERIAL(IDXNEW) = -SERIAL(IDXNEW)
         CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDXNEW),NEIBR1,NEIBR2,NEIBOR)
         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         TOTUSD = TOTUSD + 1
         SERIAL(IDX) = -TOTUSD
         CALL ADDPRL(SERIAL(IDXNEW),SERIAL(IDX), NEIBR1,NEIBR2,NEIBOR)
         CALL ADDPRL(SERIAL(IDX),SERIAL(IDXFOL), NEIBR1,NEIBR2,NEIBOR)

         LASTMV = 0
         INSMIN = INSMIN + 2

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Removing path-point: cycle & removed point-IDX =', &
                 ICYCLE, IDX )
            CALL W0('The two minima surrounding it become path-points.')
         ENDIF
         IF (VERBEU >= 3) THEN
            CALL WIR('The first one: IDXNEW & Energy =', IDXNEW, &
                 PTENE(IDXNEW))
            CALL WIR('The 2nd one  :    IDX & Energy =', IDX, PTENE(IDX))
         ENDIF
      ELSEIF (QUSMIN.AND.PREMIN(IDX) > 0) THEN
         NEWCYC = IDX
         CALL DSUM2( XS0, YS0, ZS0, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, -ONE , &
              XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
              NATOM )

         DDUM1 = ONE/NINT( LENSEG(IDXPRE)/SEGSTP(IDXPRE) )
         DDUM2 = PREMIN(IDX)*DDUM1
         DDUM3 = DDUM2 + DDUM1
         DDUM1 = DDUM2 - DDUM1

         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDXPRE)%a,YBAS(IDXPRE)%a,ZBAS(IDXPRE)%a, &
              XS0,YS0,ZS0,LENSEG(IDXPRE),PRTOL1, STPTOL,ENETOL, &
              GRATOL,MAXTOL, BRASTP,BRAMAG,ATOMAX, &
              VERBEU,NATOM,IDIM, PTGRAD(IDX), &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              ABS(MODREM), LXEVAL, DELTAS, SADGRD, BRASCA, 1, &
              DYNBRA(IDX),QMOVD2, MODXIT, .TRUE., TOTMIN, &
              PTENE(IDX), 0, QPREMA )

         NLINMN(IDX) = 0

         IF (LORIEN)  CALL ORIENT( IDX, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XREF,YREF,ZREF, NATOM, &
              NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         TOTUSD = TOTUSD + 1
         SERIAL(IDX) = -TOTUSD
         CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDX), NEIBR1,NEIBR2,NEIBOR)
         CALL ADDPRL(SERIAL(IDX),SERIAL(IDXFOL), NEIBR1,NEIBR2,NEIBOR)

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDX
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         LASTMV = 0
         INSMIN = INSMIN + 1

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Removing path-point: cycle & removed point-IDX =', &
                 ICYCLE, IDX )
            CALL W0('The minimum preceding it becomes a path-point.')
         ENDIF
         IF (VERBEU >= 3) &
              CALL WIR('Its IDX & Energy =', IDX, PTENE(IDX))
      ELSEIF (QUSMIN.AND.NEXMIN(IDX) > 0) THEN
         NEWCYC = IDX
         CALL DSUM2( XS0, YS0, ZS0, &
              XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
              -ONE , XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, NATOM )

         DDUM1 = ONE/NINT( LENSEG(IDX)/SEGSTP(IDX) )
         DDUM2 = NEXMIN(IDX)*DDUM1
         DDUM3 = DDUM2 + DDUM1
         DDUM1 = DDUM2 - DDUM1

         CALL SADLE1( DDUM1,DDUM2,DDUM3, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XS0,YS0,ZS0,LENSEG(IDX),PRTOL1, STPTOL,ENETOL, &
              GRATOL,MAXTOL, BRASTP,BRAMAG,ATOMAX, &
              VERBEU,NATOM,IDIM, PTGRAD(IDX), &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              ABS(MODREM), LXEVAL, DELTAS, SADGRD, BRASCA, 1, &
              DYNBRA(IDX),QMOVD2, MODXIT, .TRUE., TOTMIN, &
              PTENE(IDX), 0, QPREMA )

         NLINMN(IDX) = 0

         IF (LORIEN)  CALL ORIENT( IDX, &
              XBAS(IDX)%a,YBAS(IDX)%a,ZBAS(IDX)%a, &
              XREF,YREF,ZREF, NATOM, &
              NFIXED,IMGAXI,IMGROT, VERBEU, ISLCT )

         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         TOTUSD = TOTUSD + 1
         SERIAL(IDX) = -TOTUSD
         CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDX), NEIBR1,NEIBR2,NEIBOR)
         CALL ADDPRL(SERIAL(IDX),SERIAL(IDXFOL), NEIBR1,NEIBR2,NEIBOR)

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDX
         CALL SGSCAN( L,I,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
              LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
              SRTENE,SRTIDX,SEGSTP,PTGRAD, &
              XBAS,YBAS,ZBAS, NATOM,VERBEU )
         CALL UPDNEW( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX, &
              NLINMN,PTGRAD,SADGRD,SADMIN,ENEMAX )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         LASTMV = 0
         INSMIN = INSMIN + 1

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Removing path-point: cycle & removed point-IDX =', &
                 ICYCLE, IDX )
            CALL W0('The minimum following it becomes a path-point.')
         ENDIF
         IF (VERBEU >= 3) &
              CALL WIR('Its IDX & Energy =', IDX, PTENE(IDX))
      ELSE
         ! MH21: serious bug !!??
         ! used to be just I, which is global in this routine and func7090 is messing with I
         ! changed I -> I_temp locally in this loop:
         DO I_temp = 1,NEIBOR
            IF ( SERIAL(IDXPRE) == NEIBR1(I_temp) .AND. &
                 SERIAL(IDXFOL) == NEIBR2(I_temp)       ) THEN
               if(func7090() ) then
                  err=.true.
                  return
               endif
               GOTO  94
            ENDIF
         ENDDO

         NEWCYC = IDX
         NPOINT = NPOINT - 1
         NFREEP = NFREEP + 1
         IFREEP(NFREEP) = IDX
         IDNEXT(IDXPRE) = IDXFOL
         IDPREC(IDXFOL) = IDXPRE

         I = IDXPRE
         CALL UPDPRE(I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX, &
              IDNEXT,LSCAN1,LSCAN2,LSCAN3,SEGSTP,NATOM,VERBEU, &
              LENSEG,NEXMIN,PREMIN,XBAS,YBAS,ZBAS )

         I = IDXFOL
         CALL UPDEND( I,LPAKED,SRTENE,SRTIDX,NMAXI,STPMAX,QUP,QDOWN, &
              NLINMN,PTGRAD,SADGRD,SADMIN,PTENE,ENEMAX )

         CALL REMSER( SERIAL(IDX), NEIBR1,NEIBR2, NEIBOR )
         CALL ADDPRL(SERIAL(IDXPRE),SERIAL(IDXFOL),NEIBR1,NEIBR2,NEIBOR)
         LASTMV = 0
         IF ( SERIAL(IDX) < 0 )  RMVMIN = RMVMIN + 1

         IF (VERBEU >= 1) THEN
            CALL W0(' ')
            CALL W2I('Removing path-point: cycle & removed point-IDX =', &
                 ICYCLE, IDX )
         ENDIF
      ENDIF
94    CONTINUE
      err=.false.
      return
    end function func7050

  END subroutine cprrun

  SUBROUTINE SCANPATH( NPOINT,IDXPRO, IDXSAD,SADMIN,SADGRD,NLINMN, &
       IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
       LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
       SRTENE,SRTIDX,SEGSTP,PTGRAD, &
       XBAS,YBAS,ZBAS, NATOM,VERBEU)


    use chm_kinds
    use chm_types
    use stream
    implicit none

    LOGICAL   QUP(*),QDOWN(*)
    LOGICAL   LSCAN1,LSCAN2,LSCAN3
    INTEGER   NPOINT,IDXPRO,IDXSAD,SADMIN, NLINMN(*)
    INTEGER   IDNEXT(*),NMAXI, STPMAX(*),NEXMIN(*),PREMIN(*)
    INTEGER   SRTIDX(*), NATOM,VERBEU
    real(chm_real)    PTENE(*),LENSEG(*),ENEMAX(*),SRTENE(*),SEGSTP(*)
    real(chm_real)    PTGRAD(*), SADGRD
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    INTEGER   IDX, L

    CALL W0(' ')
    CALL W0(' Scanning for local E max. along the trajectory.')
    CALL W0(' -----------------------------------------------')
    NMAXI  = 0
    IDX = 1

    DO L=1, NPOINT-1

       CALL SGSCAN( L,IDX,IDNEXT,PTENE,NMAXI,LSCAN1,LSCAN2,LSCAN3, &
            LENSEG,STPMAX,NEXMIN,ENEMAX,QUP,QDOWN,PREMIN, &
            SRTENE,SRTIDX,SEGSTP,PTGRAD, &
            XBAS,YBAS,ZBAS, NATOM,VERBEU )

       IF (STPMAX(IDX) == 0.AND.NLINMN(IDX) < 0) THEN
          NMAXI = NMAXI - 1
       ELSEIF (STPMAX(IDX) < 0.AND.NLINMN(IDX) < 0) THEN
          SRTENE(NMAXI) = ENEMAX(IDX)
       ELSEIF ( (.NOT.QUP(IDX)).OR.(.NOT.QDOWN(IDX)) ) THEN
          NLINMN(IDX) = ABS(NLINMN(IDX))
       ENDIF

       IDX = IDNEXT(IDX)
    ENDDO

    IF (VERBEU >= 1)  WRITE(OUTU,1030)  IDXPRO, PTENE(IDXPRO)
1030 FORMAT(/,'  Path end-point Index',I5,3X,'(no interpolations),', &
         6X,'E=',F12.4)

    CALL DSORT1(SRTENE, SRTIDX, NMAXI, .FALSE.)

    CALL SETIHI( IDXSAD,NPOINT,IDNEXT,PTGRAD,SADGRD,QUP,QDOWN, &
         NLINMN,SADMIN,PTENE )

    RETURN
  END subroutine scanpath
  

  SUBROUTINE SCM( NMAXP, NPOINT, XBAS,YBAS,ZBAS, NATOM,DDIM, &
       IDPREC,IDNEXT,IDXPRO,NLINMN, COMLYN,COMLEN,VERBEU,LSCM,QMOVED, &
       XS0,YS0,ZS0, TANGX,TANGY,TANGZ, PRNORM,DYNBRA, PTENE,PTGRAD, &
       NFIXED,IMGAXI,IMGROT,ISLCT, XREF,YREF,ZREF )


    use chm_kinds
    use chm_types
    use number
    use dimens_fcm
    use exfunc
    use consta
    use contrl
    use coord
    use deriv
    use memory
    use energym
    use stream
    use string
    implicit none


    real(chm_real),allocatable,dimension(:) :: DUM1X
    real(chm_real),allocatable,dimension(:) :: DUM1Y
    real(chm_real),allocatable,dimension(:) :: DUM1Z
    real(chm_real),allocatable,dimension(:) :: DUM2X
    real(chm_real),allocatable,dimension(:) :: DUM2Y
    real(chm_real),allocatable,dimension(:) :: DUM2Z
    real(chm_real),allocatable,dimension(:) :: DUM3X
    real(chm_real),allocatable,dimension(:) :: DUM3Y
    real(chm_real),allocatable,dimension(:) :: DUM3Z
    real(chm_real),allocatable,dimension(:) :: PREVX
    real(chm_real),allocatable,dimension(:) :: PREVY
    real(chm_real),allocatable,dimension(:) :: PREVZ
    real(chm_real),allocatable,dimension(:) :: DOLDX
    real(chm_real),allocatable,dimension(:) :: DOLDY
    real(chm_real),allocatable,dimension(:) :: DOLDZ
    real(chm_real),allocatable,dimension(:) :: DKX
    real(chm_real),allocatable,dimension(:) :: DKY
    real(chm_real),allocatable,dimension(:) :: DKZ

    LOGICAL     LSCM, QMOVED, IMGROT
    INTEGER     NMAXP,NPOINT,IDXPRO, NATOM, VERBEU, NFIXED,IMGAXI
    INTEGER     ISLCT(*)
    type(chm_array)    XBAS(*),YBAS(*),ZBAS(*)
    INTEGER     IDPREC(*),IDNEXT(*),NLINMN(*)
    real(chm_real) TANGX(*),TANGY(*),TANGZ(*),  &
         XS0(*),YS0(*),ZS0(*),DDIM
    real(chm_real)      PRNORM(*), DYNBRA(*)
    real(chm_real)      XREF(*),YREF(*),ZREF(*)
    CHARACTER(len=*) COMLYN
    INTEGER     COMLEN
    real(chm_real)      PTENE(*),PTGRAD(*)

    LOGICAL     LORIEN, QDONE
    INTEGER     MAXUPD,MINUPD,NCYCLE, NCONJG
    real(chm_real)      PRJTOL, MAXANG

    INTEGER     BRKCYC,MODXIT,LXEVAL, STUCK
    real(chm_real)  AX,BB,CX, FA,FB,FC, FIRSTP, OLDANG,NEWANG,ATOMAX
    real(chm_real)  STPTOL,ENETOL,GRATOL,BRASCA,BRASTP,BRAMAG,DNORM2

    INTEGER     I,J,K, IDX,IDXPRE,IDXFOL, BX,BY,BZ, ILARGS
    INTEGER     IDUM1,IDUM2, TOTCJG,LRGLMN,NMOVED,TOTMIN
    INTEGER     NSADDL
    real(chm_real)      DDUM1,DDUM2,DDUM3,DDUM4,DDUM5,TGNORM,GRNORM
    real(chm_real)      LARGST, S0NORM, OLDPRJ, ATODIS

    SAVE   MAXUPD,MINUPD,NCYCLE,PRJTOL, FIRSTP, LORIEN, MAXANG,ATOMAX
    SAVE   MODXIT,LXEVAL,STPTOL,ENETOL,GRATOL,BRASCA,BRASTP,BRAMAG
    IF (NPOINT < 3) THEN
       CALL WRNDIE(-1,'<SCM>','Number of path-points must be > 2.')
       RETURN
    ENDIF

    call chmalloc('travel.src','SCM','DUM1X',NATOM,crl=DUM1X)
    call chmalloc('travel.src','SCM','DUM1Y',NATOM,crl=DUM1Y)
    call chmalloc('travel.src','SCM','DUM1Z',NATOM,crl=DUM1Z)
    call chmalloc('travel.src','SCM','DUM2X',NATOM,crl=DUM2X)
    call chmalloc('travel.src','SCM','DUM2Y',NATOM,crl=DUM2Y)
    call chmalloc('travel.src','SCM','DUM2Z',NATOM,crl=DUM2Z)
    call chmalloc('travel.src','SCM','DUM3X',NATOM,crl=DUM3X)
    call chmalloc('travel.src','SCM','DUM3Y',NATOM,crl=DUM3Y)
    call chmalloc('travel.src','SCM','DUM3Z',NATOM,crl=DUM3Z)
    call chmalloc('travel.src','SCM','PREVX',NATOM,crl=PREVX)
    call chmalloc('travel.src','SCM','PREVY',NATOM,crl=PREVY)
    call chmalloc('travel.src','SCM','PREVZ',NATOM,crl=PREVZ)
    call chmalloc('travel.src','SCM','DOLDX',NATOM,crl=DOLDX)
    call chmalloc('travel.src','SCM','DOLDY',NATOM,crl=DOLDY)
    call chmalloc('travel.src','SCM','DOLDZ',NATOM,crl=DOLDZ)
    call chmalloc('travel.src','SCM','DKX',NATOM,crl=DKX)
    call chmalloc('travel.src','SCM','DKY',NATOM,crl=DKY)
    call chmalloc('travel.src','SCM','DKZ',NATOM,crl=DKZ)

    IF (.NOT.LSCM) THEN
       PRJTOL = PT01  * SQRT(DDIM)
       BRASTP = PTONE * SQRT(DDIM)
       STPTOL = TENM5*TENM5 * SQRT(DDIM)
       GRATOL = PT05
       ENETOL = PT001*PT0001
       FIRSTP = PT05
       BRASCA = TWO
       BRAMAG = FIVE
       ATOMAX = ZERO
       MAXANG = NINETY
       MODXIT =       3
       LXEVAL =      15
       MINUPD =       2
       MAXUPD =      20
       NCYCLE =       1
       LORIEN = .TRUE.
    ENDIF

    NCYCLE = GTRMI(COMLYN,COMLEN,'NCYC', NCYCLE )
    IF (COMLEN > 3 .OR. .NOT.LSCM) THEN

       DDUM4 = GTRMF(COMLYN,COMLEN,'PROJ', PRJTOL/SQRT(DDIM) )
       DDUM4 = DDUM4*SQRT(DDIM)
       IF (DDUM4 < PRJTOL-RSMALL) THEN
          DO I = 1,NMAXP
             IF ( ABS(PRNORM(I)) > DDUM4)  PRNORM(I) = ABS(PRNORM(I))
          ENDDO
       ENDIF
       PRJTOL = DDUM4

       MODXIT = GTRMI(COMLYN,COMLEN,'EXIT', MODXIT )
       LXEVAL = GTRMI(COMLYN,COMLEN,'LXEV', LXEVAL )
       BRAMAG = GTRMF(COMLYN,COMLEN,'BRKM', BRAMAG )
       ATOMAX = GTRMF(COMLYN,COMLEN,'BRKM', ATOMAX )
       BRASCA = GTRMF(COMLYN,COMLEN,'BRKS', BRASCA )
       ENETOL = GTRMF(COMLYN,COMLEN,'TOLE', ENETOL )
       GRATOL = GTRMF(COMLYN,COMLEN,'TOLG', GRATOL )

       MAXANG = GTRMF(COMLYN,COMLEN,'ANGL', MAXANG )
       MAXUPD = GTRMI(COMLYN,COMLEN,'MAXU', MAXUPD )
       MINUPD = GTRMI(COMLYN,COMLEN,'MINU', MINUPD )

       FIRSTP = GTRMF(COMLYN,COMLEN,'FIRS', FIRSTP )
       BRASTP = GTRMF(COMLYN,COMLEN,'BRAK', BRASTP/SQRT(DDIM) )
       BRASTP = BRASTP*SQRT(DDIM)
       STPTOL = GTRMF(COMLYN,COMLEN,'TOLS', STPTOL/SQRT(DDIM) )
       STPTOL = STPTOL*SQRT(DDIM)

       IF (INDXA(COMLYN,COMLEN,'ORIE') > 0) LORIEN = .TRUE.
       IF (INDXA(COMLYN,COMLEN,'NOOR') > 0) LORIEN = .FALSE.

       IF (MINUPD > MAXUPD)  MAXUPD = 0

       IF (VERBEU >= 0) THEN
          CALL W0(' ')
          CALL W0(' Synchronous Chain Minimization parameters :')
          CALL W0(' -------------------------------------------')
          CALL WI( ' Max. number of SCM cycles               = ',NCYCLE)
          CALL WR( ' RMS tolerance for projected gradient    = ', &
               PRJTOL/SQRT(DDIM) )
          CALL WI( ' Min. # of lin.min. between plane-updates= ',MINUPD)
          CALL WI( ' Max. # of lin.min. between plane-updates= ',MAXUPD)
          CALL WR( ' Max. allowed path curvature angle [deg] = ',MAXANG)
          CALL W0(' ')
          CALL W0(' Braketing and line-minimization parameters :')
          CALL W0(' --------------------------------------------')
          CALL WR( ' Maximal 1-D Braketing Step       BRASTP = ', &
               BRASTP/SQRT(DDIM) )
          CALL WR( ' First   1-D Braketing Step       FIRSTP = ',FIRSTP)
          CALL WR( ' Dynamic BRaKeting-step SCAling   BRASCA = ',BRASCA)
          CALL WR( ' Max. Braket Magnification factor BRAMAG = ',BRAMAG)
          CALL WR( ' Toler. grad. cosin. 1-D minimiz. GRATOL = ',GRATOL)
          CALL WR( ' Smallest 1 dim. extremiz. Step   STPTOL = ', &
               STPTOL/SQRT(DDIM) )
          CALL WR( ' Smallest fractional Ener. change ENETOL = ',ENETOL)
          CALL WI( ' Exit-Mode with respect to GRATOL MODXIT = ',MODXIT)
          CALL WI( ' Line eXtrem. energy-Evaluations  LXEVAL = ',LXEVAL)
          CALL WR( ' Max. allowed atomic displacement ATOMAX = ',ATOMAX)
       ENDIF
    ENDIF

    CALL XTRANE(COMLYN,COMLEN,'SCM')

    IF (.NOT.LSCM) THEN
       LSCM = .TRUE.
       QMOVED = .FALSE.
       ECALLS = 0
       TOTUPD = 0
       TOTMIN = 0
       LARGST = -RBIG
       NSADDL = 0
       DO I = 1,NMAXP
          PRNORM(I) = +RBIG
          DYNBRA(I) =  FIRSTP/BRASCA
          IF (NLINMN(I) < 0) THEN
             PRNORM(I) = ZERO
             NSADDL = NSADDL + 1
          ENDIF
       ENDDO
       PRNORM(1) = ZERO
       PRNORM(IDXPRO) = ZERO
    ENDIF

    IF (VERBEU == 2) THEN
       CALL W0(' ')
       CALL W0( '                                            '// &
            '  Remaining')
       CALL W0( '                                            '// &
            '  points w.')
       CALL W0( '       Max.#   Avg.#                  IDX w.'// &
            '   PRNORM')
       CALL W0( '       line-   line-      Largest     largst'// &
            '      >')
       CALL W0( 'Cycle  minim.  minim.     PRNORM      PRNORM'// &
            '    PRJTOL')
       CALL W0( '-----  ------  ------  -------------  ------'// &
            '  ---------')
    ENDIF

    IF (MINUPD > MAXUPD)  MAXUPD = 0
    IF (MAXUPD <= 0) THEN
       NCONJG = MINUPD
    ELSE
       NCONJG = MAXUPD
    ENDIF

    loop201:DO I = 1, NCYCLE
       QDONE = .FALSE.
       LRGLMN = 0
       TOTCJG = TOTMIN
       NMOVED = 0

200    IDX = 1
       CALL COP3D( XBAS(1)%a,YBAS(1)%a,ZBAS(1)%a, &
            PREVX,PREVY,PREVZ, NATOM)

       loop212: DO J=2, NPOINT-1

          IDXPRE = IDX
          IDX    = IDNEXT(IDX)
          IDXFOL = IDNEXT(IDX)

          IF (         (NLINMN(IDX) < 0) .OR. &
               (PRNORM(IDXPRE) <= ZERO.AND.PRNORM(IDX) < ZERO &
               .AND.PRNORM(IDXFOL) <= ZERO.AND..NOT.QDONE) ) THEN
             CALL COP3D( XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, &
                  PREVX,PREVY,PREVZ, NATOM)
             IF (VERBEU >= 3) THEN
                CALL W0(' ')
                CALL WI('Doing point IDX=', IDX )
                CALL WR('PRNORM =', PRNORM(IDX)/SQRT(DDIM) )
                CALL W0('Point is either a saddle or is done.')
             ENDIF
             cycle loop212
          ENDIF

          CALL DSUM2( XS0,YS0,ZS0, XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, -ONE, &
               PREVX,PREVY,PREVZ, NATOM)
          DDUM1 = SQRT( DSDOT1(XS0,YS0,ZS0,NATOM) )
          CALL DSUM2( TANGX,TANGY,TANGZ, &
               XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
               -ONE, XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, NATOM)
          DDUM2 = SQRT( DSDOT1(TANGX,TANGY,TANGZ,NATOM) )

          IF (DDUM1 == ZERO.OR.DDUM2 == ZERO) THEN
             CALL W0(' ')
             CALL WI( '!!! WARNING, point IDX =', IDX)
             CALL W0('is identical to one of its neighbors !!!')
             CALL W0('The path direction at that point cannot be')
             CALL W0('defined. Remove that point and restart SCM.')
             call chmdealloc('travel.src','SCM','DUM1X',NATOM,crl=DUM1X)
             call chmdealloc('travel.src','SCM','DUM1Y',NATOM,crl=DUM1Y)
             call chmdealloc('travel.src','SCM','DUM1Z',NATOM,crl=DUM1Z)
             call chmdealloc('travel.src','SCM','DUM2X',NATOM,crl=DUM2X)
             call chmdealloc('travel.src','SCM','DUM2Y',NATOM,crl=DUM2Y)
             call chmdealloc('travel.src','SCM','DUM2Z',NATOM,crl=DUM2Z)
             call chmdealloc('travel.src','SCM','DUM3X',NATOM,crl=DUM3X)
             call chmdealloc('travel.src','SCM','DUM3Y',NATOM,crl=DUM3Y)
             call chmdealloc('travel.src','SCM','DUM3Z',NATOM,crl=DUM3Z)
             call chmdealloc('travel.src','SCM','PREVX',NATOM,crl=PREVX)
             call chmdealloc('travel.src','SCM','PREVY',NATOM,crl=PREVY)
             call chmdealloc('travel.src','SCM','PREVZ',NATOM,crl=PREVZ)
             call chmdealloc('travel.src','SCM','DOLDX',NATOM,crl=DOLDX)
             call chmdealloc('travel.src','SCM','DOLDY',NATOM,crl=DOLDY)
             call chmdealloc('travel.src','SCM','DOLDZ',NATOM,crl=DOLDZ)
             call chmdealloc('travel.src','SCM','DKX',NATOM,crl=DKX)
             call chmdealloc('travel.src','SCM','DKY',NATOM,crl=DKY)
             call chmdealloc('travel.src','SCM','DKZ',NATOM,crl=DKZ)
             RETURN
          ENDIF

          OLDANG = DDOT1( TANGX,TANGY,TANGZ, XS0,YS0,ZS0, NATOM )
          OLDANG = OLDANG/(DDUM1*DDUM2)
          IF ( OLDANG < -ONE )  OLDANG = -ONE
          IF ( OLDANG >  ONE )  OLDANG =  ONE
          OLDANG = RADDEG*ACOS( OLDANG )

          CALL DSUM2(TANGX,TANGY,TANGZ, XS0,YS0,ZS0, DDUM1/DDUM2, &
               TANGX,TANGY,TANGZ, NATOM)

          TGNORM = SQRT( DSDOT1(TANGX,TANGY,TANGZ,NATOM) )
          CALL COP3D( XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, &
               PREVX,PREVY,PREVZ, NATOM)

          OLDPRJ = ABS( PRNORM(IDX) )

          CALL COP3D( XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, X,Y,Z, NATOM)
          FB = ENERG( X,Y,Z, DX,DY,DZ, .TRUE., .FALSE., ZERO )
          CALL PROJKT( DX,DY,DZ, DDUM1, &
               TANGX,TANGY,TANGZ, TGNORM, NATOM )
          PRNORM(IDX) = SQRT( DSDOT1( DX,DY,DZ, NATOM) )

          IF (VERBEU >= 3) THEN
             CALL W0(' ')
             CALL WI('Doing point IDX=', IDX )
             CALL WR('PRNORM =', PRNORM(IDX)/SQRT(DDIM) )
          ENDIF

          IF ( PRNORM(IDX) <= PRJTOL ) THEN
             IF ( OLDPRJ <= PRJTOL )  PRNORM(IDX) = -PRNORM(IDX)
             IF (VERBEU >= 3) &
                  CALL W0('Point is done and will not be moved.')
             cycle loop212
          ENDIF

          IF (QDONE)  cycle loop212

          CALL COP3D( DX,DY,DZ, XS0,YS0,ZS0, NATOM)
          S0NORM = PRNORM(IDX)
          CALL COP3D( DX,DY,DZ, &
               DOLDX,DOLDY,DOLDZ, NATOM)
          DNORM2 = PRNORM(IDX)**2

          STUCK  = 0
          NMOVED = NMOVED + 1

          loop210:DO K = 1, NCONJG
             TOTMIN = TOTMIN + 1

             AX = ZERO
             FA = FB
             BRKCYC = 0

             BB = DYNBRA(IDX)*BRASCA
             IF ( BB > BRASTP ) THEN
                IF (VERBEU >= 4)   CALL WR( 'Was limited to BRASTP:'// &
                     ' braket-step too large =', BB/SQRT(DDIM) )
                BB = BRASTP
             ENDIF
             BB = MAX( BB, STPTOL )
             BB = -BB/S0NORM

             CALL BRAKET( AX,BB,CX, FA,FB,FC, E1DIM, BRAMAG, VERBEU, &
                  XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, XS0,YS0,ZS0, &
                  DKX,DKY,DKZ, S0NORM, &
                  NATOM , .TRUE., BRKCYC)

             IF (CX > ZERO) THEN
                CALL W0(' ')
                CALL W0(' Warning: braketing in the wrong direction')
                CALL WI( ' while minimizing point IDX =', IDX)
                GOTO  211
             ENDIF

             IF ( BB == ZERO ) THEN
                CALL COP3D( DOLDX,DOLDY,DOLDZ, &
                     DX,DY,DZ,NATOM )
             ELSE
                CALL COP3D( DKX,DKY,DKZ, DX,DY,DZ, &
                     NATOM )
                CALL PROJKT( DX,DY,DZ, DDUM1, &
                     TANGX,TANGY,TANGZ, TGNORM, NATOM )
             ENDIF

             IF ( ABS(MODXIT) > 1 ) &
                  CALL COP3D( DOLDX,DOLDY,DOLDZ, &
                  DKX,DKY,DKZ, NATOM )

             CALL LNMINI( AX,BB,CX, E1DIM,DE1DIM, FB,DDUM1, &
                  DKX,DKY,DKZ, STPTOL, &
                  ENETOL,GRATOL, VERBEU, XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, &
                  XS0,YS0,ZS0,S0NORM, NATOM, .TRUE. , &
                  LXEVAL, MODXIT, GRNORM)

             IF ( BB == ZERO ) THEN
                STUCK = STUCK + 1
                IF (VERBEU >= 2) THEN
                   CALL W0(' ')
                   CALL W0('Warning: LNMINI cannot minimize further !')
                ENDIF
                IF (STUCK >= 2) THEN
                   IF (VERBEU >= 2)  CALL W2I('Interrupting'// &
                        ' line-min. K & point-IDX =', K,IDX)
                   GOTO  211
                ENDIF
             ELSE
                STUCK = 0
             ENDIF

             IF (ATOMAX > ZERO) &
                  DDUM1 = NORMAX( IDUM1, XS0,YS0,ZS0, NATOM )
             ATODIS = BB*DDUM1
             IF (ATODIS > ATOMAX .AND. ATOMAX > ZERO) THEN
                CALL W0(' ')
                CALL W2I('! SCM> Warning: exceeded ATOMAX in'// &
                     ' line-min. K & point-IDX =', K,IDX )
                CALL WIR('! Atom-number & atom-displacement =', IDUM1,DDUM1)
                BB = ATOMAX/DDUM1
                FB = E1DIM( BB, XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, &
                     XS0,YS0,ZS0, NATOM, .TRUE., .FALSE., ZERO )
                CALL COP3D( DX,DY,DZ, DKX,DKY,DKZ, &
                     NATOM )
             ENDIF

             CALL DSUM2( DUM1X,DUM1Y,DUM1Z, &
                  XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, BB, XS0,YS0,ZS0, NATOM)

             NEWANG = RBIG
             IF ( K >= MINUPD) THEN
                CALL DSUM2( DUM2X,DUM2Y,DUM2Z, &
                     DUM1X,DUM1Y,DUM1Z, -ONE, &
                     PREVX,PREVY,PREVZ, NATOM)
                DDUM1 = SQRT( DSDOT1(DUM2X,DUM2Y, &
                     DUM2Z, NATOM) )
                CALL DSUM2( DUM3X,DUM3Y,DUM3Z, &
                     XBAS(IDXFOL)%a,YBAS(IDXFOL)%a,ZBAS(IDXFOL)%a, &
                     -ONE, DUM1X,DUM1Y,DUM1Z, NATOM)

                DDUM2 = SQRT( DSDOT1(DUM3X,DUM3Y, &
                     DUM3Z, NATOM) )

                NEWANG = DDOT1( DUM3X,DUM3Y,DUM3Z, &
                     DUM2X,DUM2Y,DUM2Z, NATOM )

                NEWANG = NEWANG/(DDUM1*DDUM2)
                IF ( NEWANG < -ONE )  NEWANG = -ONE
                IF ( NEWANG >  ONE )  NEWANG =  ONE
                NEWANG = RADDEG*ACOS( NEWANG )
             ENDIF

             IF ( NEWANG > OLDANG .AND. NEWANG > MAXANG .AND. &
                  K > MINUPD)  GOTO  211
             OLDANG = NEWANG

             CALL COP3D( DUM1X,DUM1Y,DUM1Z, &
                  XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, NATOM)
             PTENE(IDX) = FB
             IF ( ABS(MODXIT) > 1 )  THEN
                GRNORM = SQRT( &
                     DSDOT1( DKX,DKY,DKZ, NATOM) )
             ENDIF
             PTGRAD(IDX) = GRNORM/SQRT(DDIM)

             QMOVED = .TRUE.
             IF ( BB /= ZERO )  DYNBRA(IDX) = ABS(BB)*S0NORM

             CALL PROJKT( DKX,DKY,DKZ, DDUM1, &
                  TANGX,TANGY,TANGZ, TGNORM, NATOM )
             PRNORM(IDX) = SQRT( &
                  DSDOT1( DKX,DKY,DKZ, NATOM) )

             IF (VERBEU >= 4) THEN
                CALL WI( 'In line-minimization K=', K )
                CALL W2R('PRNORM & Angle =',PRNORM(IDX)/SQRT(DDIM),NEWANG)
             ENDIF

             IF ( PRNORM(IDX) <= PRJTOL )  GOTO  211

             DDUM1 = DDOT2( DOLDX,DOLDY,DOLDZ, &
                  DKX,DKY,DKZ, NATOM)
             CALL COP3D( DKX,DKY,DKZ, &
                  DOLDX,DOLDY,DOLDZ, NATOM)
             CALL DSUM2( XS0,YS0,ZS0, DKX,DKY,DKZ, &
                  (PRNORM(IDX)**2-DDUM1)/DNORM2, XS0,YS0,ZS0, NATOM)
             S0NORM = SQRT( DSDOT1( XS0,YS0,ZS0, NATOM) )
             DNORM2 = PRNORM(IDX)**2

          enddo loop210
          K = K - 1

211       IF (LORIEN)  CALL ORIENT( IDX, XBAS(IDX)%A,YBAS(IDX)%A,ZBAS(IDX)%A, &
               XREF,YREF,ZREF, NATOM,NFIXED,IMGAXI,IMGROT,VERBEU, ISLCT)

          IF (VERBEU >= 3) THEN
             CALL WI( 'After nb. of conjugate line-minimizations =',K)
             CALL WR( '   E      =', FB)
             CALL WR( '   PRNORM =', PRNORM(IDX)/SQRT(DDIM))
             CALL WR( '   OLDANGle     =', OLDANG)
             CALL WR( '   NEWANGle     =', NEWANG)
          ENDIF

          IF ( K > LRGLMN)  LRGLMN = K
       enddo loop212

       LARGST = -RBIG
       IDUM2 = 0
       IDX = 1
       DO J=2, NPOINT-1
          IDX    = IDNEXT(IDX)
          IF ( ABS(PRNORM(IDX)) > PRJTOL) IDUM2 = IDUM2 + 1
          IF ( NLINMN(IDX) >= 0 ) THEN
             IF ( ABS(PRNORM(IDX)) > LARGST ) THEN
                LARGST = ABS(PRNORM(IDX))
                ILARGS = IDX
                IDUM1 = J
             ENDIF
          ENDIF
       ENDDO

       IF ((LARGST <= PRJTOL.OR.I >= NCYCLE).AND..NOT.QDONE) THEN
          QDONE = .TRUE.
          IF (MAXUPD > 0)  GOTO  200
       ENDIF

       IF (VERBEU >= 3) THEN
          CALL W0(' ')
          CALL W0( '                                            '// &
               '  Remaining')
          CALL W0( '                                            '// &
               '  points w.')
          CALL W0( '       Max.#   Avg.#                  IDX w.'// &
               '   PRNORM')
          CALL W0( '       line-   line-      Largest     largst'// &
               '      >')
          CALL W0( 'Cycle  minim.  minim.     PRNORM      PRNORM'// &
               '    PRJTOL')
          CALL W0( '-----  ------  ------  -------------  ------'// &
               '  ---------')
       ENDIF
       IF (VERBEU >= 2 .AND. NMOVED > 0) THEN
          DDUM5 = (TOTMIN - TOTCJG)/NMOVED
          WRITE(OUTU,1001) I, LRGLMN, NINT(DDUM5), &
               LARGST/SQRT(DDIM), ILARGS, IDUM2
1001      FORMAT(I5,2X,I6,2X,I6,2X,1PE13.7E1,2X,I6,2X,I9)
       ENDIF

       IF ((LARGST <= PRJTOL.OR.I >= NCYCLE).AND.QDONE)  GOTO  202
    enddo loop201
    I = I-1

202 CONTINUE
    CALL W0(' ')

    IF (LARGST <= PRJTOL) THEN
       CALL WI( 'Exiting SCM after doing number of cycles =', I)
       CALL W0('------------------------------------------')
       CALL W0('The path is FULLY relaxed for the used PRJTOL.')
    ELSE
       CALL WI( 'Exiting SCM after doing max. allowed cycles =',I)
       CALL W0('---------------------------------------------')
       CALL W0('The path is NOT fully relaxed for the used PRJTOL.')
    ENDIF

    CALL WI('Number of points still with (grad.proj.>PRJTOL) =', &
         IDUM2)
    CALL W0(' ')
    CALL WR('The largest RMS gradient-projection is          =', &
         LARGST/SQRT(DDIM) )
    CALL WI('which is located on point IDX =', ILARGS)
    CALL WI('which is the Nth point along the path =', IDUM1)
    CALL W0(' ')
    CALL W0(' Since the last time SCM was initialized :')
    CALL W0(' -----------------------------------------')
    CALL WI(' Average numb. of line-minimizations/point=', &
         TOTMIN/(NPOINT-2-NSADDL)  )
    CALL WI(' The total number of line-minimizations   =',TOTMIN)
    CALL WI(' Total of energy calls (w. & wo. grad.)   =',ECALLS)
    CALL WI(' Total updates of the non-bond list       =',TOTUPD)

    call chmdealloc('travel.src','SCM','DUM1X',NATOM,crl=DUM1X)
    call chmdealloc('travel.src','SCM','DUM1Y',NATOM,crl=DUM1Y)
    call chmdealloc('travel.src','SCM','DUM1Z',NATOM,crl=DUM1Z)
    call chmdealloc('travel.src','SCM','DUM2X',NATOM,crl=DUM2X)
    call chmdealloc('travel.src','SCM','DUM2Y',NATOM,crl=DUM2Y)
    call chmdealloc('travel.src','SCM','DUM2Z',NATOM,crl=DUM2Z)
    call chmdealloc('travel.src','SCM','DUM3X',NATOM,crl=DUM3X)
    call chmdealloc('travel.src','SCM','DUM3Y',NATOM,crl=DUM3Y)
    call chmdealloc('travel.src','SCM','DUM3Z',NATOM,crl=DUM3Z)
    call chmdealloc('travel.src','SCM','PREVX',NATOM,crl=PREVX)
    call chmdealloc('travel.src','SCM','PREVY',NATOM,crl=PREVY)
    call chmdealloc('travel.src','SCM','PREVZ',NATOM,crl=PREVZ)
    call chmdealloc('travel.src','SCM','DOLDX',NATOM,crl=DOLDX)
    call chmdealloc('travel.src','SCM','DOLDY',NATOM,crl=DOLDY)
    call chmdealloc('travel.src','SCM','DOLDZ',NATOM,crl=DOLDZ)
    call chmdealloc('travel.src','SCM','DKX',NATOM,crl=DKX)
    call chmdealloc('travel.src','SCM','DKY',NATOM,crl=DKY)
    call chmdealloc('travel.src','SCM','DKZ',NATOM,crl=DKZ)

    RETURN
  END subroutine scm


  SUBROUTINE SADLE1( YA,Y0,YC, PX,PY,PZ, XS0,YS0,ZS0,S0NORM, &
       PRTOLJ, STPTOL,ENETOL,GRATOL,MAXTOL, BRASTP,BRAMAG,ATOMAX, &
       VERBEU, N, IDIM, PTGRAD, SADDLX,SADDLY,SADDLZ, LINMIN, &
       LXEVAL,DELTAS, SADGRD, BRASCA, SADMIN, DYNBRK, QMOVED,MODXIT, &
       QMINI,ADDMIN, ESADDL, SUCGRA, QPREMA )


    use chm_kinds
    use exfunc
    use number
    use dimens_fcm
    use coord
    use deriv
    use memory
    use stream
    implicit none

    real(chm_real),allocatable,dimension(:) :: HX
    real(chm_real),allocatable,dimension(:) :: HY
    real(chm_real),allocatable,dimension(:) :: HZ
    real(chm_real),allocatable,dimension(:) :: SX
    real(chm_real),allocatable,dimension(:) :: SY
    real(chm_real),allocatable,dimension(:) :: SZ
    real(chm_real),allocatable,dimension(:) :: DBX
    real(chm_real),allocatable,dimension(:) :: DBY
    real(chm_real),allocatable,dimension(:) :: DBZ
    real(chm_real),allocatable,dimension(:) :: DKX
    real(chm_real),allocatable,dimension(:) :: DKY
    real(chm_real),allocatable,dimension(:) :: DKZ
    LOGICAL   QMOVED,QMINI, QPREMA
    INTEGER   N,IDIM, VERBEU, LINMIN,LXEVAL, SADMIN, MODXIT
    INTEGER   ADDMIN, SUCGRA
    real(chm_real)    YA,Y0,YC, PTGRAD, SADGRD, DELTAS, ESADDL
    real(chm_real)    PX(*),PY(*),PZ(*), XS0(*),YS0(*),ZS0(*),S0NORM
    real(chm_real)    SADDLX(*),SADDLY(*),SADDLZ(*)
    real(chm_real)    PRTOLJ,STPTOL,ENETOL,GRATOL,MAXTOL
    real(chm_real)    BRASTP,BRAMAG,BRASCA,DYNBRK,ATOMAX

    INTEGER   IDUM1
    INTEGER   I, BRKCYC, SUCBX0, IEXIT,IEVAL, IORTHO
    real(chm_real)  Y1, FB,DFB, AX,BX,CX, FA,FC, SQRGRA, SNORM, GNORM2
    real(chm_real)    DDUM1,DDUM2, PROJ, GRATIO, RIDIM, ATODIS

    IF ((Y0-YA)*(YC-Y0)  <=  ZERO) THEN
       CALL W0(' ')
       CALL W0(' SADLE1 was called with inadequate YA,Y0,YC :')
       CALL WR( ' YA =', YA)
       CALL WR( ' Y0 =', Y0)
       CALL WR( ' YC =', YC)
       STOP
    ENDIF

    QMOVED = .FALSE.
    QPREMA = .FALSE.
    RIDIM  = IDIM
    SUCBX0 = 0
    GRATIO = ZERO
    IORTHO = 0

    call chmalloc('travel.src','SADLE1','HX',N,crl=HX)
    call chmalloc('travel.src','SADLE1','HY',N,crl=HY)
    call chmalloc('travel.src','SADLE1','HZ',N,crl=HZ)
    call chmalloc('travel.src','SADLE1','SX',N,crl=SX)
    call chmalloc('travel.src','SADLE1','SY',N,crl=SY)
    call chmalloc('travel.src','SADLE1','SZ',N,crl=SZ)
    call chmalloc('travel.src','SADLE1','DBX',N,crl=DBX)
    call chmalloc('travel.src','SADLE1','DBY',N,crl=DBY)
    call chmalloc('travel.src','SADLE1','DBZ',N,crl=DBZ)
    call chmalloc('travel.src','SADLE1','DKX',N,crl=DKX)
    call chmalloc('travel.src','SADLE1','DKY',N,crl=DKY)
    call chmalloc('travel.src','SADLE1','DKZ',N,crl=DKZ)

    FB = E1DIM( Y0, PX,PY,PZ, XS0,YS0,ZS0, N, .TRUE., .FALSE., ZERO)
    Y1 = Y0
    ESADDL = FB
    SQRGRA = DSDOT1( DX,DY,DZ, N )
    PTGRAD = SQRT(SQRGRA/RIDIM)
    CALL COP3D(DX,DY,DZ, DKX,DKY,DKZ, N)

    IF (S0NORM <= ZERO) THEN
       CALL W0(' ')
       CALL W0('! SADLE1> Warning: passed vector XS0 = 0 ,'// &
            ' doing nothing and returning.')
       GOTO  102
    ENDIF

    IEVAL = 2*LXEVAL
    IDUM1 = -1
    IF (IDUM1 == 2 .OR. IDUM1 == 3) &
         CALL COP3D(DX,DY,DZ, DBX,DBY,DBZ, N)

    CALL LNMINI( YA,Y1,YC, E1DIM,DE1DIM, &
         FB,DFB,DBX,DBY,DBZ, &
         STPTOL,ENETOL,MAXTOL, VERBEU, PX,PY,PZ, &
         XS0,YS0,ZS0,S0NORM, N, QMINI , IEVAL, &
         IDUM1, DDUM1)

    IF (IDUM1 == 2 .OR. IDUM1 == 3) &
         DDUM1 = SQRT(DSDOT1(DBX,DBY,DBZ, N))


    IF (  ((.NOT.QMINI).AND. FB > ESADDL) .OR. &
         (      QMINI .AND. FB < ESADDL)      ) THEN
       QMOVED = (Y1 /= Y0)
       ESADDL = FB
       PTGRAD = DDUM1/SQRT(RIDIM)
       SQRGRA = DDUM1*DDUM1
       PROJ   = (DFB*DFB)/SQRGRA

    ELSE
       CALL W0(' ')
       IF (VERBEU >= 2)  CALL W0('! SADLE1> Warning: no energy'// &
            ' increase during single maximization.')
       IF (ABS(FB-ESADDL) > ZERO.AND.VERBEU >= 3) THEN
          CALL WR('! Log10|FB - ESADDL| =', LOG10(ABS(FB-ESADDL)) )
       ELSEIF(VERBEU >= 3) THEN
          CALL W0('! |FB - ESADDL| = 0' )
       ENDIF

       Y1 = Y0
       PROJ = ZERO
       CALL COP3D( DKX,DKY,DKZ, &
            DBX,DBY,DBZ, N)
    ENDIF

    IF (DELTAS == ZERO) THEN
       IORTHO = 1
       CALL COP3D( XS0,YS0,ZS0,  HX,HY,HZ,N)
       CALL MULTD2(ONE/S0NORM**2,HX,HY,HZ,N)

    ELSE
       IF ( DELTAS < ZERO .OR. Y1 == Y0 ) THEN
          Y0 = Y1 + ABS(DELTAS)/S0NORM
          DDUM2 = E1DIM( Y0, PX,PY,PZ, XS0,YS0,ZS0, N, .TRUE., &
               .FALSE., ABS(YC-YA)*S0NORM+ABS(DELTAS) )
       ELSE
          CALL COP3D( DKX,DKY,DKZ, DX,DY,DZ, N)
       ENDIF

       CALL  DSUM2( HX,HY,HZ, &
            DBX,DBY,DBZ, &
            -ONE, DX,DY,DZ, N )
       DDUM1 = DDOT1( HX,HY,HZ, XS0,YS0,ZS0, N)
       IF (DDUM1 == ZERO) THEN
          CALL W0(' ')
          CALL W0('! SADLE1> Warning:  HX*XS0 = 0 '// &
               ' after unique line-maximization.' )
          GOTO  102
       ENDIF
       CALL MULTD2( ONE/DDUM1, HX,HY,HZ, N )
    ENDIF

    CALL  DSUM2( SADDLX,SADDLY,SADDLZ, PX,PY,PZ, Y1, XS0,YS0,ZS0, N )
    Y0 = Y1
    IF (LINMIN == 0)  GOTO  101

    DDUM1 = DDOT1( DBX,DBY,DBZ, &
         HX,HY,HZ, N )
    CALL  DSUM1( SX,SY,SZ, &
         -ONE, DBX,DBY,DBZ, &
         DDUM1, XS0,YS0,ZS0, N )
    SNORM = SQRT( DSDOT1(SX,SY,SZ,N) )

    loop100:DO I = 1, LINMIN
       IF (SNORM == ZERO) THEN
          CALL W0(' ')
          CALL WI('! SADLE1> Warning: vector SX = 0'// &
               ' during line-min. I =', I )
          GOTO  102
       ENDIF

       ADDMIN = ADDMIN + 1
       AX = ZERO
       FA = ESADDL
       BX = DYNBRK/SNORM
       BRKCYC = 0
       CALL BRAKET( AX,BX,CX, FA,FB,FC, E1DIM, BRAMAG, VERBEU, &
            SADDLX,SADDLY,SADDLZ, SX,SY,SZ, &
            DKX,DKY,DKZ, SNORM, &
            N , .TRUE., BRKCYC)

       IF (CX < ZERO.AND.VERBEU >= 3) THEN
          CALL W0(' ')
          CALL W0( '! SADLE1> Warning: braketed in wrong')
          CALL W2I('! direction.  Line-min. & BRKCYC =', I, BRKCYC)
       ENDIF

       IF ( BX == ZERO ) THEN
          CALL COP3D( DBX,DBY,DBZ, DX,DY,DZ, N)
       ELSE
          CALL COP3D( DKX,DKY,DKZ, DX,DY,DZ, N)
       ENDIF

       CALL COP3D( DBX,DBY,DBZ, &
            DKX,DKY,DKZ, N)

       IEXIT  = MODXIT
       IEVAL  = LXEVAL
       DDUM2 = GRATOL
       IF (I == 1 .OR. SUCBX0 > 0)  THEN
          IEXIT  = -ABS(MODXIT)
          IEVAL  = 2*LXEVAL
          DDUM2 = MAXTOL
       ENDIF
       CALL LNMINI( AX,BX,CX, E1DIM,DE1DIM, FB,DFB, &
            DBX,DBY,DBZ, STPTOL, &
            ENETOL,DDUM2, VERBEU, SADDLX,SADDLY,SADDLZ, &
            SX,SY,SZ,SNORM, N, &
            .TRUE., IEVAL, IEXIT, DDUM1)

       IF (ATOMAX > ZERO) &
            DDUM1 = NORMAX( IDUM1, SX,SY,SZ, N )
       ATODIS = BX*DDUM1

       IF (ATODIS > ATOMAX .AND. ATOMAX > ZERO) THEN
          IF (VERBEU >= 1) THEN
             CALL W0(' ')
             CALL WI('! SADLE1> Exceeded ATOMAX'// &
                  ' in line-min. I =', I )
             CALL WIR('! Atom-number & displacement =', &
                  IDUM1,DDUM1)
          ENDIF
          BX = ATOMAX/DDUM1
          FB = E1DIM( BX, SADDLX,SADDLY,SADDLZ, SX,SY, &
               SZ, N, .TRUE., .FALSE., ZERO)
          CALL COP3D( DX,DY,DZ, DBX,DBY,DBZ, N)
       ENDIF

       IF (FB < ESADDL) THEN
          CALL  DSUM2( SADDLX,SADDLY,SADDLZ, &
               SADDLX,SADDLY,SADDLZ, &
               BX, SX,SY,SZ, N )
          ESADDL = FB
          QMOVED = ((BX /= ZERO).OR.QMOVED)

          IF (BX <= ZERO.AND.VERBEU >= 3) THEN
             CALL W0(' ')
             CALL WI('! SADLE1> Warning:  BX<0 in line-min. I =',I)
             IF (ABS(BX) > ZERO) THEN
                CALL WR('! Unusual, since FB < ESADDL.  Log10|BX| =', &
                     LOG10(ABS(BX)) )
             ELSE
                CALL W0('! Bizarre, since FB < ESADDL, while  BX = 0' )
             ENDIF
          ENDIF

          GNORM2 = DSDOT1( DBX,DBY,DBZ, N )
          IF (GNORM2 == ZERO.OR.SQRGRA == ZERO) THEN
             CALL W0(' ')
             CALL WI('! SADLE1> Warning: gradient = 0'// &
                  ' during line-min. I =', I )
             GOTO  102
          ENDIF
          GRATIO = GNORM2/SQRGRA
          SQRGRA = GNORM2
          PTGRAD = SQRT(SQRGRA/RIDIM)

          DDUM1 = DDOT1(DBX,DBY,DBZ,XS0,YS0,ZS0,N)
          PROJ   = DDUM1/GNORM2
          DDUM1 = DDUM1/(S0NORM*S0NORM)
          PROJ   = PROJ*DDUM1

          DYNBRK = MAX( MIN(BRASTP,ABS(BX)*SNORM*BRASCA), STPTOL )

          SUCBX0 = 0

       ELSE
          CALL COP3D( DKX,DKY,DKZ, &
               DBX,DBY,DBZ, N)
          CALL W0(' ')
          CALL WI('! SADLE1> Warning: no energy decrease'// &
               ' during line-min. I =', I )
          IF ((FB-ESADDL) > ZERO.AND.VERBEU >= 3) THEN
             CALL W2R('! BX & Log10(FB-ESADDL) =', BX,LOG10(FB-ESADDL))
          ELSEIF(VERBEU >= 3) THEN
             CALL W0('! FB = ESADDL ,  BX = 0 .' )
          ENDIF

          SUCBX0 = SUCBX0 + 1
          IF (GRATIO == ZERO) THEN
             IF (IORTHO == 0) THEN
                CALL W0('! Will CONTINUE  with minimization'// &
                     ' in hyper-space orthogonal to XS0.')
                IORTHO = I + 1
                CALL COP3D( XS0,YS0,ZS0,  HX,HY,HZ,N)
                CALL MULTD2(ONE/S0NORM**2,HX,HY,HZ,N)
                GRATIO = ZERO

             ELSE
                GOTO  102
             ENDIF

          ELSE
             IF (SUCBX0 <= 3) THEN
                CALL W0('! Will try to CONTINUE  normally.')
                GRATIO = ONE

             ELSE
                CALL W0('! Reseting the series of conjugate directions')
                GRATIO = ZERO
             ENDIF
          ENDIF
       ENDIF

       IF (VERBEU >= 3) THEN
          IF (I == 1) &
               CALL W0( ' Minim.  Log10(PTGRAD)     Log10(PROJ)    '// &
               'Dist.moved BX      Ener. FB' )
          DDUM1 = -RBIG
          IF (PTGRAD > ZERO)  DDUM1 = LOG10(PTGRAD)
          DDUM2 = -RBIG
          IF (  PROJ > ZERO)  DDUM2 = LOG10(PROJ)
          WRITE(OUTU,1000) I, DDUM1, DDUM2, BX*SNORM, FB
1000      FORMAT(1X,I5,2X,1PG15.8,2X,1PG15.8,2X,1PG15.8,2X,1PG15.8)
       ENDIF

       IF (PTGRAD <= SADGRD) THEN
          SUCGRA = SUCGRA + 1
          IF (SUCGRA >= SADMIN) THEN
             IF (VERBEU >= 3) THEN
                CALL W0(' ')
                CALL WI( ' Exiting SADLE1 with satisfied SADMIN'// &
                     ' after line-minimizations I =', I )
             ENDIF
             GOTO  210
          ENDIF
       ELSE
          SUCGRA = 0
       ENDIF

       IF (PROJ > PRTOLJ) THEN
          IF (VERBEU >= 3) THEN
             CALL W0(' ')
             CALL WI( ' Exiting SADLE1 with exceeded PRTOLJ'// &
                  ' after line-minimizations I =', I )
          ENDIF
          GOTO  210
       ENDIF

       CALL  DSUM1( SX,SY,SZ, &
            -ONE,    DBX,DBY,DBZ, &
            GRATIO, SX,SY,SZ, N )
       DDUM1 = DDOT1( DBX,DBY,DBZ, &
            HX,HY,HZ, N )
       CALL  DSUM2( SX,SY,SZ, &
            SX,SY,SZ, &
            DDUM1, XS0,YS0,ZS0, N )
       SNORM  = SQRT( DSDOT1(SX,SY,SZ,N) )

    enddo loop100

101 IF (VERBEU >= 3) THEN
       CALL W0(' ')
       CALL WI( ' Exiting SADLE1 with satisfied LINMIN =', LINMIN)
    ENDIF
    GOTO  210

102 IF (VERBEU >= 1)  CALL W0('! Exiting SADLE1 prematurely.')
    QPREMA = .TRUE.

210 IF (VERBEU >= 3) &
         CALL W2R(' Current energy & RMS-gradient =', ESADDL,PTGRAD )

    call chmdealloc('travel.src','SADLE1','HX',N,crl=HX)
    call chmdealloc('travel.src','SADLE1','HY',N,crl=HY)
    call chmdealloc('travel.src','SADLE1','HZ',N,crl=HZ)
    call chmdealloc('travel.src','SADLE1','SX',N,crl=SX)
    call chmdealloc('travel.src','SADLE1','SY',N,crl=SY)
    call chmdealloc('travel.src','SADLE1','SZ',N,crl=SZ)
    call chmdealloc('travel.src','SADLE1','DBX',N,crl=DBX)
    call chmdealloc('travel.src','SADLE1','DBY',N,crl=DBY)
    call chmdealloc('travel.src','SADLE1','DBZ',N,crl=DBZ)
    call chmdealloc('travel.src','SADLE1','DKX',N,crl=DKX)
    call chmdealloc('travel.src','SADLE1','DKY',N,crl=DKY)
    call chmdealloc('travel.src','SADLE1','DKZ',N,crl=DKZ)

    RETURN
  END subroutine sadle1
#endif /* (travel)*/

#if KEY_HFB==1 /*hfb*/
  !-----------------------------------------------------------------------
  SUBROUTINE FRCPATH(XBAS,YBAS,ZBAS, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
       LINITFP,IDSTART,IDSTOP,ITRUNC, &
       LDENS,LNOROTATE, &
       XS0,YS0,ZS0, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       QERROR,NFREEP,IFREEP, &
       TOTUSD,SERIAL,NLINMN, &
       IDPREC,IDNEXT, &
       NQGRID)
    ! IVK: This routine superimposes the given replicas to the first one
    ! and then computes the corresponding Fourier expansion for every
    ! coordinate of all the atoms in the selection subspace.
    ! Once this is done the routine integrates the resulting smooth path
    ! to find its full length by using first order line integral.
    ! Basically finds the length of the overall curve.
    ! Once the length is known as a function of the parameter alpha 
    ! (or lambda) the values of alpha that correspond to equidistant
    ! arc lengths are found and the corresponding coordinates are 
    ! then generated. The coordinates are then available in one of the 
    ! coordinate sets (either main or comparison). This is good enough to 
    ! start MD simulations with RMSD restraints.
    ! 
    use chm_kinds
    use chm_types
    use stream
    implicit none
    ! QMEPINIT - if true the path will be initialized/reset
    ! this is stored in the common array
    ! The main subroutine for the Lagrangian MEP method
    !ivk      LOGICAL QMEPREVR,QMEPANAL
    LOGICAL LINITFP,LDENS,lnorotate
    INTEGER NSELR,NATOM,NPOINT,NMAXP
    INTEGER FPSLCT(*)
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    !ivk  Note that in this version SMASS is onlu NSELR long
    real(chm_real) SMASS(*)
    real(chm_real) XS0(*),YS0(*),ZS0(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    LOGICAL QERROR
    INTEGER NFREEP,IFREEP(*)
    INTEGER TOTUSD,SERIAL(*),NLINMN(*)
    INTEGER IDPREC(*),IDNEXT(*)
    INTEGER IDSTART,IDSTOP,ITRUNC,NQGRID

    ! Local variables

    if (.not. lnorotate) then
       CALL initfrcp(XBAS,YBAS,ZBAS, &
            FPSLCT,NSELR,NATOM,NPOINT,SMASS)
    endif

    IF (LINITFP) THEN
       CALL W0('Performing line interpolation')
       IF (NPOINT  ==  2) THEN
          IDSTART = 1
          IDSTOP = 2
          CALL W2I('Interpolating between points:',IDSTART,IDSTOP)
       ELSE
          CALL W2I('Interpolating between points:',IDSTART,IDSTOP)
       ENDIF

       IF (IDSTART + 1  /=  IDSTOP) THEN
          CALL WRNDIE(-1,'<FRCPATH>','Only adjacent points allowed')
          RETURN
       ENDIF

       CALL InterpolateFP(XBAS,YBAS,ZBAS, &
            IDSTART,IDSTOP, &
            XS0,YS0,ZS0, &
            XKEEP,YKEEP,ZKEEP, &
            XDUM,YDUM,ZDUM, &
            FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
            QERROR,NFREEP,IFREEP, &
            TOTUSD,SERIAL,NLINMN, &
            IDPREC,IDNEXT)
    ELSEIF (LDENS) THEN
       CALL W0('Performing Path Activation')
       CALL ActivateFPMax(XBAS,YBAS,ZBAS, &
            XS0,YS0,ZS0, &
            XKEEP,YKEEP,ZKEEP, &
            XDUM,YDUM,ZDUM, &
            FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
            QERROR,NFREEP,IFREEP, &
            TOTUSD,SERIAL,NLINMN, &
            IDPREC,IDNEXT)
    ENDIF

    ! This should have superimposed configurations
    ! Call a subroutine to compute the Fourier Transform of the 
    ! coordinates for the selected atoms using alpha variable [0;1]
    ! I can write such a routine myself but check first if charmm already
    ! has one of this kind. No could not find any simple enough to use.
    ! writing my own.

    ! Now let's do the Fourier transformation on the provided path
    ! Remember to properly handle the linear term in front of the
    ! Fourier series: it needs to be taken care of when doing FT
    ! Need to store the Fourier coefficients somehow:
    ! ax(*),ay(*),az(*)
    ! need the alpha variable as a vector of size (nrep)
    ! need a line integral variable probably of the same size.
    ! I can always recompute the length as a function of alpha
    ! though not sure if this would be the most efficient way.
    ! REP = 1 corresponds to alpha = 0
    ! REP = N corresponds to alpha = 1
    ! First of, subtract the linear term in alpha from each replica

    !ivk Define the number of grid points for the quadrature first.

    IF (NQGRID  <  0) THEN
       NQGRID = NMAXP * 4
       CALL WI('The number of qudrature grid-points is ', &
            NQGRID )
    ELSEIF (NQGRID  <  NMAXP * 4) THEN
       NQGRID = NMAXP * 4
       CALL WI('The number of grid-points too small, increase to ', &
            NQGRID )
    ENDIF


    CALL SETFPATH(XBAS,YBAS,ZBAS, &
         FPSLCT,NSELR,NATOM,NPOINT,NMAXP,ITRUNC, &
         LDENS, &
         XS0,YS0,ZS0, &
         XKEEP,YKEEP,ZKEEP, &
         XDUM,YDUM,ZDUM, &
         NQGRID)

    ! Upon exit we should have X, Y and Z corresponding to
    ! equally separated structures from a Fourier path. These
    ! structures should be used as references for RMSD restrained
    ! molecular dynamics.

    ! This is inserted to be able to perform path analysis
    !ivk      IF (QMEPANAL) THEN
    !ivk       CALL ANALMEP(X,Y,Z,REFPX,REFPY,REFPZ,
    !ivk     $              LMEPSLCT,NSELR,NATREP,NREP,NATOM,SMASS)
    !ivk      ENDIF

    RETURN
  END SUBROUTINE FRCPATH
  !-----------------------------------------------------------------------
  SUBROUTINE WRKFRCPATH(XBAS,YBAS,ZBAS, &
       FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
       LINITFP,IDSTART,IDSTOP,ITRUNC, &
       LDENS,LFEND,LPRJCT,LPSTRUCT,lnorotate, &
       XS0,YS0,ZS0, &
       XKEEP,YKEEP,ZKEEP, &
       XDUM,YDUM,ZDUM, &
       QERROR,NFREEP,IFREEP, &
       TOTUSD,SERIAL,NLINMN, &
       IDPREC,IDNEXT, &
       IRTYPE,RRFRC,RSTEP, &
       IPROFU,NBEADS,NQGRID)

    ! IVK: The purpose of this subroutine is to 
    ! 1(2)) Read in the evolved beads coordinates, also without any
    !    kind of superposition and use them to compute the mean forces on each bead
    !    and use the bead as the origin of the coordinate system
    !    Note: In the correct implementation I need to use the averaged beads as the
    !          origin of the coordinate system!
    ! 2(1)) Read in the reference beads from a trajectory and get the Fourier 
    !    amplitudes without any kind of superposition just as they are
    !    Note: In the correct implementation the amplitudes need to be computed
    !          from the averaged (evolved) beads, not the reference!
    ! 3) Do the Fourier transform of the forces or the force vector field
    !    that is in the coordinate system with bead as its origin
    !    keep the Fourier coefficients
    !    Note: Once again the origin should be the averaged bead!
    ! 4) Compute the line integrals of the second order and then
    !    sum them up to get the generalized line integral that is the WORK
    ! 5) write the work out to a file as a function of the parameter alpha, and enjoy.
    !ivk I just realized that if I simply swap the reference beads and the evolved beads
    !ivk it should get me the correct profile up to the sign! Let's try that!

    ! Will do this option later.
    !ivk It is time to get down to business Thu Aug 31 15:38:09 PDT 2006
    ! 6) Need to get the mean forces' components that are directly orthogonal to 
    !    the path at the bead points. Use these forces to step a certain length in
    !    the direction of the forces to make a new reference coordinates, and
    !    if necessary may be redistribute the beads on the new proposed path. 
    !    
    !    For now let's read both reference and the evolved beads consequitively
    !    into the same big fat trajectory with double the number of points
    !    Yes I know I don't like it either, but can't figure out a simple enough
    !    alternative way.

    use chm_kinds
    use chm_types
    use stream
    implicit none
    ! QMEPINIT - if true the path will be initialized/reset
    ! this is stored in the common array
    ! The main subroutine for the Lagrangian MEP method
    !ivk      LOGICAL QMEPREVR,QMEPANAL
    LOGICAL LINITFP,LDENS,LFEND,LPRJCT,LPSTRUCT,lnorotate
    INTEGER NSELR,NATOM,NPOINT,NMAXP
    INTEGER FPSLCT(*)
    type(chm_array) XBAS(:),YBAS(:),ZBAS(:)
    !ivk  Note that in this version SMASS is onlu NSELR long
    real(chm_real) SMASS(*)
    real(chm_real) XS0(*),YS0(*),ZS0(*)
    real(chm_real) XKEEP(*),YKEEP(*),ZKEEP(*)
    real(chm_real) XDUM(*),YDUM(*),ZDUM(*)
    LOGICAL QERROR
    INTEGER NFREEP,IFREEP(*)
    INTEGER TOTUSD,SERIAL(*),NLINMN(*)
    INTEGER IDPREC(*),IDNEXT(*)
    INTEGER IDSTART,IDSTOP,ITRUNC,IRTYPE,IPROFU,NBEADS,NQGRID
    real(chm_real) RRFRC,RSTEP

    ! Local variables

    IF (LDENS .or. LPSTRUCT) THEN
       CALL W0('Performing Path Activation')
       CALL ActivateFPMax(XBAS,YBAS,ZBAS, &
            XS0,YS0,ZS0, &
            XKEEP,YKEEP,ZKEEP, &
            XDUM,YDUM,ZDUM, &
            FPSLCT,NSELR,NATOM,NPOINT,NMAXP,SMASS, &
            QERROR,NFREEP,IFREEP, &
            TOTUSD,SERIAL,NLINMN, &
            IDPREC,IDNEXT)
    ENDIF

    ! Call a subroutine to perform the Fourier Transform of the 
    ! coordinates for the selected atoms using alpha variable [0;1]

    ! Now let's do the Fourier transformation on the reference path
    ! Need to store this first set of the Fourier coefficients
    ! First of, subtract the linear term in alpha from each replica
    ! Note that in this part of the code NPOINT is actually doubled

    CALL PRCSSFPATH(XBAS,YBAS,ZBAS, &
         FPSLCT,NSELR,NATOM,NPOINT,NMAXP,ITRUNC, &
         LDENS,LFEND,LPRJCT,LPSTRUCT, &
         XS0,YS0,ZS0, &
         XKEEP,YKEEP,ZKEEP, &
         XDUM,YDUM,ZDUM, &
         SMASS,IRTYPE,RRFRC,RSTEP, &
         IPROFU,NBEADS,NQGRID)

    IF (LPSTRUCT) THEN
       !ivk The NPOINT is already reset to the NQGRID
       CALL WI('The structures generated on the quadrature grid ', &
            NQGRID )
       RETURN
    ENDIF

    ! Upon exit we should have X, Y and Z corresponding to
    ! equally separated structures from a Fourier path. These
    ! structures should be used as references for RMSD restrained
    ! molecular dynamics.

    !ivk Need to figure out ways to remove the second set of points
    !ivk Let's try the simplest possible solution first:

    NPOINT = NPOINT/2

    if (.not. lnorotate) then
       CALL initfrcp(XBAS,YBAS,ZBAS, &
            FPSLCT,NSELR,NATOM,NPOINT,SMASS)
    endif

    CALL SETFPATH(XBAS,YBAS,ZBAS, &
         FPSLCT,NSELR,NATOM,NPOINT,NMAXP,ITRUNC, &
         LDENS, &
         XS0,YS0,ZS0, &
         XKEEP,YKEEP,ZKEEP, &
         XDUM,YDUM,ZDUM, &
         NQGRID)


    ! This is inserted to be able to perform path analysis
    !ivk      IF (QMEPANAL) THEN
    !ivk       CALL ANALMEP(X,Y,Z,REFPX,REFPY,REFPZ,
    !ivk     $              LMEPSLCT,NSELR,NATREP,NREP,NATOM,SMASS)
    !ivk      ENDIF
    RETURN
  END SUBROUTINE WRKFRCPATH
  !-----------------------------------------------------------------------
#endif /* (hfb)*/

end module travelsub

