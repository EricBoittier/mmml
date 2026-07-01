!Puja's zeta put in c43 by Tanmoy 10-16-2017 [QC: 11/17]
module rxdefs

  use chm_types
  use dimens_fcm
  use rxncom
  implicit none

contains

#if KEY_RXNCOR==1 /*rxndefs_main*/
  !---------------------------------------------------------------
  !       RXPARS  called from charm_main.src 
  !---------------------------------------------------------------
  subroutine rxpars
    !
    !     ---- to parse the rxncor command and to transfer control
    !     ---- to the appropriate routine
    !
    use number
    use exfunc
    use comand
    use stream
    use string
    use consta
    use reawri
    use memory
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec, q_cons_node
    use domdec_io,only:send_nostr_to_root
#endif

    !
    !
    ! Modified by SRD 2/6/92
    !   Moved this section above the declarations to prevent
    !   unresolved external reference when linking.
    !
    real(chm_real),allocatable,dimension(:) :: DELP
    integer i, rxnunt, freq
    character(len=4) word
    !     ARD and MFH 03-05-13
    !     Added for multiple reaction coordinates
    INTEGER IFMTMP, IR, INDEX
    real(chm_real)  KUMTMP, PUMTMP, DT1, DT2, DT3, D
#if KEY_SMD==1
    real(chm_real)  SMDTMP 
#endif
    !     Pointer to real
    CHARACTER(len=4) RXNNAM
    !
#if KEY_TPS==1 /*tps_basvar*/
    INTEGER TMPUNI
    real(chm_real) TMPALO, TMPAHI, TMPBLO, TMPBHI
#endif /*  (tps_basvar)*/
    !
    logical :: first=.true.
    ! Node where the rxnene computation was performed
    integer comp_nod

    !       ---- Initialization
    !
    if (first) then
       ngraph = 0
       trcfrq(1:maxnod) = 0

       old_MDSTEP = -1

       rxncnt = 0
       rxntrn = 0
       first  = .false.
    end if
    !
    word = nexta4 (comlyn,comlen)
    !
    cmd_sel: select case(word)
    case('DEFI') cmd_sel
       call rxndef
    case('SET ') cmd_sel
       call rxnset
    case('UMBR' )  cmd_sel
       !       ARD and MFH 03-05-13
       !       Re-written to allow for mulitiple reaction coordinates.
       IF (NRXNCR  <=  0) CALL WRNDIE(-2, '<RXPARS>', &
            'Cannot set UMBRELLA until reaction coordinates set')

       CALL GTRMWD(COMLYN,COMLEN,'NAME',4,RXNNAM,80,IR)
       IF (IR  ==  0) THEN
          CALL WRNDIE (2, '<RXPARS>', &
               'NO NAME SPECIFIED:  DEFAULTING TO FIRST COORDINATE')
          IR = 1
       ELSE
          IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)
          !         Error handling taken care of in INDGRF.
          IF (IR  ==  0) IR = 1
       ENDIF

       IFMTMP = GTRMI(COMLYN,COMLEN,'FORM',1)
       KUMTMP = GTRMF(COMLYN,COMLEN,'KUMB',ZERO)
       !       ARD and Jie Hu 06-06-30 for periodicity
       PUMTMP = GTRMF(COMLYN,COMLEN,'PERI',ZERO)
       !       ARD and Jie Hu 06-06-30 for SMD based on RXNCOR
#if KEY_SMD==1
       SMDTMP = GTRMF(COMLYN,COMLEN,'SMDD',ZERO) 
#endif
       DT1    = GTRMF(COMLYN,COMLEN,'DEL0',ZERO)
       DT2    = NEXTF(COMLYN,COMLEN)
       DT3    = NEXTF(COMLYN,COMLEN)

       !       Normalize DEL0 if it is a direction
       INDEX = TREELO(IR)
       IF (      DEFTYP(INDEX)  ==  DIREPTPT_TYPE &
            .OR. DEFTYP(INDEX)  ==  DIREDRDR_TYPE) THEN
          D = SQRT(DT1*DT1 + DT2*DT2 + DT3*DT3)
          DT1 = DT1 / D
          DT2 = DT2 / D
          DT3 = DT3 / D
       ENDIF

       ! possible bug fix: /MH09/
       ! Problems with allocating these arrays! It seems to me
       ! that it depends on the order of the commands in the input
       ! script, but the conversion to fortran 2003/2008 must go on !!!
       ! I am not sure which order of allocation is the best so
       ! a special routine takes care of it ... FIX it properly later!
       !        write(*,*)'rxpars-umbr>basXXX allocation'
#if KEY_TPS==1
       call allocmixorder    
#endif

       umbfrm(ir) = ifmtmp
       KUMBPR(IR) = KUMTMP
       PUMBPR(IR) = PUMTMP
#if KEY_SMD==1
       SMDDEL(IR) = SMDTMP 
#endif
       dl0ptr(3*(IR-1)+1) = dt1
       dl0ptr(3*(IR-1)+2) = dt2
       dl0ptr(3*(IR-1)+3) = dt3

       !       Set RXNIND to a non-zero number so RXNENE is called
       !       (see comment in rxnset).
       RXNIND = 1
       !
    case('TRAC' )  cmd_sel
       rxntrn = rxntrn + 1
       if (rxntrn > maxnod) then
          call wrndie (-1, '<RXPARS>', &
               'No space for so many traces.  Ignored')
          rxntrn = rxntrn - 1
       else
          trcfrq(rxntrn) = gtrmi (comlyn, comlen, 'FREQ', 10)
          trunit(rxntrn) = gtrmi (comlyn, comlen, 'UNIT', outu)
          rxntrc(rxntrn) = indgrf(nexta4(comlyn,comlen), &
               ngraph+1,nrxn,nodnam,0) ! now optional parameter/MH09/
          if (rxntrc(rxntrn) == 0) then
             call wrndie (-1, '<RXPARS>', &
                  'Not clear what to trace.  Ignored')
             rxntrn = rxntrn - 1
          end if
       end if
       !       Set RXNIND to a non-zero number so RXNENE is called
       !       (see comment in rxnset).
       RXNIND = 1
       !
    case('STAT' )  cmd_sel
       !       Read the start and the file name if they are given
       STTSTP = GTRMI(COMLYN, COMLEN, 'STAR',   1)
       RXNUNT = GTRMI(COMLYN, COMLEN, 'READ',  -1)

       !       ARD 03-05-13 Change:  Calling STATistics reinitializes DELSTT
       IF (allocated(DELSTP)) THEN
          ir = nmlstt(1)*nrxstt(1)
          call chmdealloc('rxndef.src','rxpars','DELSTP',IR,intg=DELSTP)
       ELSE
          call chmalloc('rxndef.src','rxpars','NMLSTT',NRXNCR,intg=NMLSTT)
          call chmalloc('rxndef.src','rxpars','NRXSTT',NRXNCR,intg=NRXSTT)
          call chmalloc('rxndef.src','rxpars','LODEL',NRXNCR,crl=LODEL)
          call chmalloc('rxndef.src','rxpars','HIDEL',NRXNCR,crl=HIDEL)
          call chmalloc('rxndef.src','rxpars','DELDEL',NRXNCR,crl=DELDEL)
       ENDIF

       !       Allocate space for basin limits for path sampling
       !        write(*,*)'rxpars-stat>basXXX allocation'
       call allocmixorder     ! moved from below
       !       Go through each of the reaction coordinates.

       CALL SETSTT(COMLYN,COMLEN,IR,LODEL,HIDEL, &
            DELDEL,NRXSTT,NMLSTT, &
            TREELO,NRXNCR,NODNAM)
       ir = nmlstt(1)*nrxstt(1)
       call chmalloc('rxndef.src','rxpars','DELSTP',IR,intg=DELSTP)
       !       Read the old statistics if available, otherwise initialize
       call chmalloc('rxndef.src','rxpars','DELP',3*NRXNCR,crl=DELP)

       CALL RDSTT(RXNUNT,DELSTP,NRXNCR,NMLSTT,NRXSTT, &
            LODEL,DELDEL,DELP)

       call chmdealloc('rxndef.src','rxpars','DELP',3*NRXNCR,crl=DELP)
       !=====================================================

    case('WRIT' )  cmd_sel

       RXNUNT = GTRMI(COMLYN, COMLEN, 'UNIT', OUTU)
       call chmalloc('rxndef.src','rxpars','DELP',3*NRXNCR,crl=DELP)
       IR=1   ! Bugfix: select first one (needs to be fixed later)- BRB
       INDEX = TREELO(IR)
       !
       comp_nod = 0
#if KEY_DOMDEC==1
       ! Check that rxncor computing was done on a single node and returns
       ! the node number that performed the computation
       call check_comp_nod(comp_nod)
#endif

#if KEY_PARALLEL==1
       IF(MYNOD == comp_nod) THEN
#endif
#if KEY_DOMDEC==1
          ! This tells wrstt() to send the data to root node instead of writing it in file
          if (comp_nod /= 0) rxnunt = -1
#endif
! DTM debug - only write stats if data collected
!          CALL WRSTT(RXNUNT,DELSTP,NRXNCR,NMLSTT,NRXSTT, &
          if (RXNCNT  >  STTSTP) then
             CALL WRSTT(RXNUNT,DELSTP,NRXNCR,NMLSTT,NRXSTT, &
               TREELO,UMBFRM,KUMBPR,DL0PTR, &
               NBIASP,RBIASP,CBIASP,LODEL, &
               DELDEL,DELP,RXNCNT,DEFTYP(INDEX), &
               PUMBPR &
#if KEY_SMD==1
               ,SMDDEL,BASALO,BASAHI,  & 
#endif
#if KEY_SMD==1
               BASBLO,BASBHI,QFLIP    & 
#endif
               )
#if KEY_DOMDEC==1
          else
             ! For DOMDEC, in case constraint node didn't go to WRSTT -subroutine, 
             ! we need to send a 0 length string to terminate the root node that
             ! is waiting to write in wrstt_root()
             call send_nostr_to_root()
#endif
          endif
#if KEY_DOMDEC==1
       else if (mynod == 0) then
          ! For DOMDEC, root node goes here and does the actual writing in file
          call wrstt_root(rxnunt)
#endif
#if KEY_PARALLEL==1
       ENDIF
#endif

       call chmdealloc('rxndef.src','rxpars','DELP',3*NRXNCR,crl=DELP)
       !
       !=====================================================
       ! begin: umbrella sampling implementation. CAR August 2000
    case('BIAS' )  cmd_sel
       !       ARD and MFH 03-05-13
       !       Re-written to allow for mulitiple reaction coordinates.
       IF (NRXNCR  <=  0) CALL WRNDIE(-2, '<RXPARS>', &
            'Cannot set UMBRELLA until reaction coordinates set')

       CALL GTRMWD(COMLYN,COMLEN,'NAME',4,RXNNAM,80,IR)
       IF (IR  ==  0) THEN
          CALL WRNDIE (2, '<RXPARS>', &
               'NO NAME SPECIFIED:  DEFAULTING TO FIRST COORDINATE')
          IR = 1
       ELSE
          IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)
          !         Error handling taken care of in INDGRF.
          IF (IR  ==  0) IR = 1
       ENDIF
       !
#if KEY_PARALLEL==1
       IF (MYNOD == 0) THEN
#endif 
          CALL RXNBF(IR,NBIASP,RBIASP,CBIASP)
#if KEY_PARALLEL==1
       ENDIF
#endif 
       !
       ! end  : umbrella sampling implementation. CAR August 2000
       !
       !=====================================================

    case('TPUN' )  cmd_sel
       !     ARD and M. Hagan for TPS
#if KEY_TPS==1 /*tps_tpunit*/
       !       Enable writeout of order parameter values
       IF (NRXNCR  <=  0) CALL WRNDIE(-2, '<RXPARS>', &
            'Cannot set basin writeout until reaction coordinates set')

       !       Allocate array to hold units for printing of order parameters
       call chmalloc('rxndef.src','rxpars','TPOPUN',NRXNCR,intg=TPOPUN)
       !       Go through each of the reaction coordinates.
       DO I=1,NRXNCR
          RXNNAM = NEXTA4(COMLYN, COMLEN)
          IF(RXNNAM == ' ') CALL WRNDIE(-2, '<RXPARS>', &
               'Need a name to set basin writeout')

          !         To which element in tree does this refer?
          IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)
          TMPUNI = NEXTI(COMLYN,COMLEN)
          tpopun(ir) = tmpuni
       ENDDO
#else /*   (tps_tpunit)*/
       CALL WRNDIE (-1,'<RXPARS>', &
            'TPUNit requires TPS compilation')
#endif /*  (tps_tpunit)*/

    case('BASI' )  cmd_sel
       !       ARD and MFH for TPS 03-05-13
       !       Define the reaction and product basins in TPS using rxncor.
#if KEY_TPS==1 /*tps_basdef*/
       !       Make sure reaction coordinates have been set
       IF (NRXNCR  <=  0) CALL WRNDIE(-2, '<RXPARS>', &
            'Cannot set basins until reaction coordinates set')

       !       Allocate space for basin limits for path sampling
       !        write(*,*)'rxpars>basXXX-2 allocation'
       call allocmixorder      ! NOTE: this was the original allocation /MH09/
       !       Go through each of the reaction coordinates.
       DO I=1,NRXNCR
          RXNNAM = NEXTA4(COMLYN, COMLEN)
          IF(RXNNAM == ' ') CALL WRNDIE(-2, '<RXPARS>', &
               'Need a name to set basin')

          !         To which element in tree does this refer?
          IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)

          TMPALO = NEXTF(COMLYN,COMLEN)
          TMPAHI = NEXTF(COMLYN,COMLEN)
          TMPBLO = NEXTF(COMLYN,COMLEN)
          TMPBHI = NEXTF(COMLYN,COMLEN)

          basalo(ir) = tmpalo
          basahi(ir) = tmpahi
          basblo(ir) = tmpblo
          basbhi(ir) = tmpbhi

          !         ARD and Jie Hu 06-06-30 for periodicity and TPS
          IF(DDPERI(TMPAHI-TMPALO,PUMBPR(IR))* &
               DDPERI(TMPBHI-TMPBLO,PUMBPR(IR))  <  ZERO) &
               CALL WRNDIE (-2,'<RXPARS>', 'IMPOSSIBLE BASINS!')
       ENDDO

       IF (PRNLEV  >=  0) THEN
          WRITE (OUTU,*)
          WRITE (OUTU,887) '  RXNCOR>','NAME','ALOW','AHIGH','BLOW', &
               'BHIGH'
          DO IR = 1, NRXNCR
             WRITE (OUTU,888) ' ',NODNAM(TREELO(IR)), &
                  BASALO(IR),BASAHI(IR), &
                  BASBLO(IR),BASBHI(IR)
          ENDDO
887       FORMAT(A,1X,A9,1X,A9,1X,A9,1X,A9,1X,A9,1X,A9)
888       FORMAT(1X,A9,1X,A9,F9.3,1X,F9.3,1X,F9.3,1X,F9.3)
       end IF
#else /*   (tps_basdef)*/
       CALL WRNDIE (-1,'<RXPARS>', &
            'BASIn requires TPS compilation')
#endif /*  (tps_basdef)*/

    case default  cmd_sel
       !
       call wrndie (-1,'<RXPARS>', word//' is unknown command')
       !
    end select cmd_sel
    !
    return
  end subroutine rxpars



  !---------------------------------------------------------------
  !       RXNDEF
  !---------------------------------------------------------------
  subroutine rxndef
    !
    !     ---- to define point, line, plane, distance and angle
    !     ---- for defining the reaction coordinate
    !
    !     ---- global variables
    !
    use number
    use comand
    use stream
    use string
    !     ---- local variables
    !
    integer, parameter :: maxjun=20
    integer lenjun, ind1, ind2, tmpind, ind3, &
         dstype, type1, type2, type3, comtyp, n, dummy
    character(len=4) word, key, dtype(5), name, name1, name2, name3
    character(len=maxjun) junk
    real(chm_real) wt1, wt2, sum
    logical done, gotpt
    !
    !     ---- set ngraph and name
    !
    ngraph = ngraph + 1
    nodnam(ngraph) = nexta4 (comlyn,comlen)
    word  = nexta4 (comlyn, comlen)
    !
    !     ---- remove ineffective words from command line:
    !     ---- POIN, AND, TO, LINE, PLAN, BETW
    !
    done = .true.
    do while(.not. done)
       done = .true.
       call gtrmwa (comlyn, comlen, 'POIN', 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
       call gtrmwa (comlyn, comlen, 'AND' , 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
       call gtrmwa (comlyn, comlen, 'TO'  , 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
       call gtrmwa (comlyn, comlen, 'LINE', 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
       call gtrmwa (comlyn, comlen, 'PLAN', 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
       call gtrmwa (comlyn, comlen, 'BETW', 4, junk, maxjun, lenjun)
       done = done .and. lenjun == 0
    enddo
    !
    !     ---- this subroutine will be easier to read if read side by side
    !     ---- with the file rxncom.f90
    !     ---- The meanings of deftyp codes are listed there.
    !
    defcmnd: select case (word)
    case ('POIN') defcmnd
       call poidef
    case ('DIRE') defcmnd
       name1 = nexta4 (comlyn,comlen)
       name2 = nexta4 (comlyn,comlen)
       ind1 = indgrf (name1,1,ngraph,nodnam,0) ! 0->out:optional
       ind2 = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
       refnod(1,ngraph) = ind1
       refnod(2,ngraph) = ind2
       type1 = deftyp(ind1)
       type2 = deftyp(ind2)
#if KEY_ROLLRXNCOR==1
       if (type1 == POINT_TYPE .and. type2 == POINT_TYPE) then
          deftyp(ngraph)   = DIREPTPT_TYPE
       else if((type1 == DIREPTPT_TYPE .or. type1 == DIREDRDR_TYPE .or. type1 == 81) .and.   &
            (type2 == DIREPTPT_TYPE .or. type2 == DIREDRDR_TYPE .or. type2 == 81)) then
          deftyp(ngraph) = DIREDRDR_TYPE
       end if
#else /**/
       if (type1 >= LAST_TYPE .and. type2 >= LAST_TYPE) then
          deftyp(ngraph)   = DIREPTPT_TYPE
       else if((type1 == DIREPTPT_TYPE .or. type1 == DIREDRDR_TYPE) .and.   &
            (type2 == DIREPTPT_TYPE .or. type2 == DIREDRDR_TYPE)) then
          deftyp(ngraph) = DIREDRDR_TYPE
       end if
#endif 
    case('DIST') defcmnd
       name1 = nexta4 (comlyn,comlen)
       name2 = nexta4 (comlyn,comlen)
       ind1 = indgrf (name1,1,ngraph,nodnam,0) ! 0->out: optional
       ind2 = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
       !       ---- one of these must be a point
       if (deftyp(ind1) < POINT_TYPE) then
          !         ---- interchange
          tmpind = ind1
          ind1 = ind2
          ind2 = tmpind
       end if
       if (deftyp(ind1) < POINT_TYPE) call wrndie (-3, '<RXNDEF>', &
            ' Error in distance definition')
       !       ---- now ind1 is a point
       refnod(1,ngraph) = ind1
       refnod(2,ngraph) = ind2
       dstype = deftyp(ind2)/10
       disttype: select case (dstype)
       case(POINT_ITEM) disttype
          !         ---- ind2 is a point, so distance is between two points
          deftyp(ngraph) = DISTPTPT_TYPE
       case(LINE_ITEM) disttype
          !         ---- ind2 is a line, so distance between point and line
          deftyp(ngraph) = DISTPTLN_TYPE
       case(PLANE_ITEM) disttype
          !         ---- ind2 is a plane, so distance between point and plane
          deftyp(ngraph) = DISTPTPL_TYPE
       case default disttype
          call wrndie (-3, ' <RXNDEF>', ' error in distance definition')
       end select disttype
       !
    case('ANGL') defcmnd
       !
       name1 = nexta4 (comlyn, comlen)
       name2 = nexta4 (comlyn, comlen)
       name3 = nexta4 (comlyn, comlen)
       ind1 = indgrf(name1,1,ngraph,nodnam,0) ! 0->out: optional
       ind2 = indgrf(name2,1,ngraph,nodnam,0) ! 0->out
       ind3 = indgrf(name3,1,ngraph,nodnam,0) ! 0->out
       type1 = deftyp(ind1) / 10
       type2 = deftyp(ind2) / 10
       type3 = deftyp(ind3) / 10
       comtyp = type1*100+type2*10+type3
       select case (comtyp)
       case(LIN_LIN_DIR)
          deftyp(ngraph) = ANGLLNLN_TYPE
       case(LIN_PLN_DIR, PLN_LIN_DIR)
          deftyp(ngraph) = ANGLLNPL_TYPE
       case(PLN_PLN_DIR)
          deftyp(ngraph) = ANGLPLPL_TYPE
#if KEY_ROLLRXNCOR==0
       case(DIR_DIR_DIR) 
#endif
#if KEY_ROLLRXNCOR==1
       case(DIR_DIR_DIR, DIR_DIR_COM, COM_COM_COM) 
#endif
          deftyp(ngraph) = ANGLDRDR_TYPE
       case default
          call wrndie (-3, '<RXNDEF>', ' Error in angle definition')
       end select
       refnod(1,ngraph) = ind1
       refnod(2,ngraph) = ind2
       refnod(3,ngraph) = ind3
    case('RATI') defcmnd
       deftyp(ngraph) = RATIO_TYPE
       name1 = nexta4 (comlyn,comlen)
       name2 = nexta4 (comlyn,comlen)
       refnod(1,ngraph) = indgrf (name1,1,ngraph,nodnam,0) ! 0->out: optional
       refnod(2,ngraph) = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
    case('COMB', 'VCOM') defcmnd
       !       ---- strip comlyn of WEIG. repeat till all gone
       lenjun=-1
       loop20: do while(lenjun /= 0) 
          call gtrmwa (comlyn, comlen, 'WEIG', 4, junk, maxjun, lenjun)
       enddo loop20

       name2 = nexta4 (comlyn,comlen)
       ind2  = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
       wt2  = nextf (comlyn,comlen)
       name = nodnam(ngraph)
       ngraph = ngraph - 1

       loop30: do while(comlen > 0)
          name1 = nexta4 (comlyn, comlen)
          if (name1 == '    ') exit loop30
          ind1  = indgrf (name1,1,ngraph,nodnam,0) ! 0->out
          wt1  = nextf (comlyn, comlen)
          ngraph = ngraph + 1
          nodnam(ngraph) = '    '
          IF (WORD == 'COMB') THEN
             deftyp(ngraph) = COMBI_TYPE
          ELSE
             deftyp(ngraph) = 81
          ENDIF
          refnod(1,ngraph) = ind1
          refnod(2,ngraph) = ind2
          !         sum = wt1 + wt2                            !JG 12/96
          sum = one
          !         ARD and Jie Hu 06-06-30 for VCOM
          delval(4,ngraph) = wt1 / sum
          delval(5,ngraph) = wt2 / sum
          ind2 = ngraph
          wt2  = sum
          call trima(comlyn, comlen)
       enddo loop30

       nodnam(ngraph) = name
       !
    case('SCOM') defcmnd
       name2 = nexta4 (comlyn,comlen)
       ind2  = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
       wt2  = nextf (comlyn,comlen)
       name = nodnam(ngraph)
       ngraph = ngraph - 1

       loop50: do while(comlen > 0)
          name1 = nexta4 (comlyn, comlen)
          if (name1 == '    ') exit loop50
          ind1  = indgrf (name1,1,ngraph,nodnam,0) ! 0->out
          wt1  = nextf (comlyn, comlen)
          ngraph = ngraph + 1
          nodnam(ngraph) = '    '
          deftyp(ngraph) = COMBI_TYPE
          refnod(1,ngraph) = ind1
          refnod(2,ngraph) = ind2
          delval(4,ngraph) = wt1
          delval(5,ngraph) = wt2
          ind2 = ngraph
          !
          !         ARD 06-06-30 
          !         I think wt2 must be one for SCOM to be able to handle
          !         more then two terms as suggested in the documentation.
          !         For example, for s = w1*val1 + w2*val2 + w3*val3, the
          !         values are saved in a binary tree, so 
          !             WANT s = w1*val1 + 1*(w2*val2 + w3*val3), 
          !             NOT  s = w1*val1 + 0*(w2*val2 + w3*val3).
          !
          !         wt2  = 0
          wt2  = ONE
          call trima(comlyn, comlen)
       enddo loop50

       nodnam(ngraph) = name
        !
        !Puja [QC: 11/17]
     case('CECM') defcmnd
         ! Proton transfer channel
         ! modified center of excess charge.
         ! Nilanjan Ghosh, Peter H. Koenig
         ! debug
         ! write(*,*) 'cecm type, got here~'
         deftyp(ngraph) = CECM_TYPE
         call protonchdef
 
     case('CEC2') defcmnd
          ! Proton transfer channel
          ! modified center of excess charge.
          ! additional term for coupled donor-acceptor
          ! Peter H. Koenig
         deftyp(ngraph) = CEC2_TYPE
         call protonchdef2
 
     case('VSUM') defcmnd
          ! Vector sum
          ! Peter H. Koenig
         deftyp(ngraph) = VSUM_TYPE
         name1 = nexta4 (comlyn,comlen)
         name2 = nexta4 (comlyn,comlen)
         refnod(1,ngraph) = indgrf (name1,1,ngraph,nodnam,0) ! 0->out
         refnod(2,ngraph) = indgrf (name2,1,ngraph,nodnam,0) ! 0->out
        ! Puja [QC: 11/17]
        !

    case default defcmnd
       !       ---- line and plane cases
       !
       gotpt = .false.
       n = 0
       loop210: do dummy = 1, 3
          key = nexta4 (comlyn, comlen)
          if (key == '    ') then
             continue
          else if (key == 'THRO' .and. .not. gotpt) then
             name = nexta4 (comlyn,comlen)
             refnod(1,ngraph) = indgrf (name,1,ngraph,nodnam,0) ! 0->out
             gotpt = .true.
          else
             n = n + 1
             dtype(n) = key
             name = nexta4 (comlyn,comlen)
             refnod(n+1,ngraph) = indgrf (name,1,ngraph,nodnam,0) ! 0->out
          end if
       enddo loop210
       !
       call xtrane (comlyn, comlen, '<RXNDEF>')
       !
       ln_pln_case: select case(word)
       case('LINE') ln_pln_case
          if (n == 1 .and. dtype(1) == 'THRO') then
             deftyp(ngraph) = LINEPTPT_TYPE
          else if (n == 1 .and. dtype(1) == 'PARA') then
             deftyp(ngraph) = LINEPTLP_TYPE
          else if (n == 2 .and. dtype(1) == 'PERP' .and. &
               dtype(2) == 'PERP') then
             deftyp(ngraph) = LINEPTPP_TYPE
          else if (n == 1 .and. dtype(1) == 'NORM') then
             deftyp(ngraph) = LINEPTPL_TYPE
          else
             call wrndie (-1, '<RXNDEF>', &
                  ' Syntax error in line definition')
          end if
          !
       case('PLAN') ln_pln_case
          if (n == 2 .and. dtype(1) == 'THRO' .and. &
               dtype(2) == 'THRO') then
             deftyp(ngraph) = PLAN3PT_TYPE
          else if (n == 1 .and. dtype(1) == 'CONT') then
             deftyp(ngraph) = PLANPTLN_TYPE
          else if (n == 1 .and. dtype(1) == 'PERP') then
             deftyp(ngraph) = PLANPTNM_TYPE
          else if (n == 2 .and. dtype(1) == 'PARA' .and. &
               dtype(2) == 'PARA') then
             deftyp(ngraph) = PLANPTLL_TYPE
          else if (n == 1 .and. dtype(1) == 'PARA') then
             deftyp(ngraph) = PLANPTPL_TYPE
          else
             call wrndie (-1, 'RXNDEF>', &
                  'Syntax error in plane definition')
          end if
          !
       case default ln_pln_case
          call wrndie (-1, '<RXNDEF>', ' syntax error in'// &
               word// ' definition')
       end select ln_pln_case
    end select defcmnd
    !
    if (ngraph > maxnod) then
       IF(WRNLEV >= 2) write (outu,*) &
            ' This version of the program is not large', &
            ' enough to hold ', ngraph, ' elements'
       call wrndie (-3, '<RXNDEF>', ' maxnod exceeded')
    end if
    !
    return
  end subroutine rxndef



  !---------------------------------------------------------------
  !       UMBPOT
  !---------------------------------------------------------------
  SUBROUTINE  UMBPOT(EUMB,DEUMB,DELTA,IR,UMFORM,KUMB,DELTA0, &
       NBIASP,CBIASP,RBIASP,DSTYPE,PUMB &
#if KEY_SMD==1
       ,SMDDEL,BASALO,BASAHI,BASBLO,BASBHI,QFLIP  & 
#endif
       )
    !       Get the energetic contribution and derivative of umbrella IR.
    !       Modified for multiple reaction coordinates by ARD 03-05-13
    use number,only:two
#if KEY_SMD==1
    use smd           
#endif
    implicit none
    !
    INTEGER NBIASP(:)
    type(chm_array) :: CBIASP(:), RBIASP(:)

    integer ir, umform(:), dstype
#if KEY_SMD==1
    integer qflip(:) 
#endif
    ! rank has changed for deumb,delta, so (:) is not so good,(*) better
    ! proper solution is to explicitly rank them as are originals in rxncom
    real(chm_real)  eumb, deumb(3), delta(3), kumb(:), pumb(:),  &
         delta0(3,*)
#if KEY_SMD==1
    real(chm_real)  smddel(:), basalo(:), basahi(:), basbhi(:),basblo(:)     
#endif

    !
    real(chm_real)  DD

    !       The following lines are the same for umforms 1 and 5
    IF (dstype == DIREPTPT_TYPE .OR. dstype == DIREDRDR_TYPE .OR.  &
         dstype == POINT_TYPE .OR. dstype == 81) THEN
       eumb = kumb(ir) * ((delta(1) - delta0(1,ir)) ** 2 + &
            (delta(2) - delta0(2,ir)) ** 2 + &
            (delta(3) - delta0(3,ir)) ** 2 )
       deumb(1) = 2.0d0 * kumb(ir) * (delta(1) - delta0(1,ir))
       deumb(2) = 2.0d0 * kumb(ir) * (delta(2) - delta0(2,ir))
       deumb(3) = 2.0d0 * kumb(ir) * (delta(3) - delta0(3,ir))

       IF (UMFORM(IR)  ==  5) CALL  WRNDIE(-2, '<UMBPOT>',  &
            'BIAS umbrella form not defined for vectors')
    else
#if KEY_SMDbad==1
       call smdadd(ir,delta,delta0,pumb,smddel,basalo,basahi, &
            basblo,basbhi,qflip)
#endif 
       dd = delta(1) - delta0(1,ir)
       dd = ddperi(dd,pumb(ir))   ! adjust if periodic coordinate
       eumb = kumb(ir) * dd * dd
       deumb(1) = two * kumb(ir) * dd
    endif

    if (umform(ir)  ==  5) then
       call biaspt(eumb,deumb,delta,nbiasp(ir), &
            rbiasp(ir)%a,cbiasp(ir)%a)
    endif
    return
  end subroutine umbpot


  !---------------------------------------------------------------
  !       BIASPT
  !---------------------------------------------------------------
  ! from rxnene.src : for dependency problem... /MH09/
  SUBROUTINE  BIASPT(EUMB,DEUMB,DELTA,NBIASP,RBIAS,CBIAS)
    !
    !       Get the bias contribution to the umbrella potential.
    !
    INTEGER NBIASP
    real(chm_real)  EUMB, RBIAS(:), CBIAS(4,*) 
    real(chm_real)  deumb(*)   ! was scalar / big problem with the new fortran/
    !without (3) - possible bug/MH09/
    real(chm_real)  delta(*)   ! was scalar
    !without (3) - possible bug/MH09/
    !
    real(chm_real)  DELX,EB2
    INTEGER I, KNDX
    !
    DO I = 1,NBIASP-1
       KNDX = I
       IF(DELTA(1) >= RBIAS(I) .AND. DELTA(1) < RBIAS(I+1)) GOTO 20
    ENDDO
    CALL  WRNDIE(0, '<BIAPOT>', 'Wrong or bad RXNCOR value')
20  DELX = DELTA(1)-RBIAS(KNDX)
    EB2 = ((CBIAS(1,KNDX)*DELX+CBIAS(2,KNDX))*DELX+ &
         CBIAS(3,KNDX))*DELX+CBIAS(4,KNDX)
    EUMB = EUMB+EB2
    DEUMB(1) = DEUMB(1)  &
         +(3.0D0*CBIAS(1,KNDX)*DELX+2.0D0*CBIAS(2,KNDX))*DELX &
         + CBIAS(3,KNDX)
    RETURN 
  END SUBROUTINE BIASPT

  !---------------------------------------------------------------
  !       POIDEF
  !---------------------------------------------------------------
  subroutine poidef
    !
    use number
    use string
    use select
    use comand
    use psf
    use coord
    use memory
#if KEY_DOMDEC==1
    use domdec_common,only:domdec_system_changed
#endif 

    !     ---- local variables
    integer,allocatable,dimension(:) :: iats
    integer i, k, n
    real(chm_real)  dummy(2), sum
    logical masflg
    character(len=4) word

    !     ---- array on stack
    call chmalloc('rxndef.src','poidef','iats',natom,intg=iats)

    !     ---- read atom selections and weights
    call selcta (comlyn, comlen, iats, x, y, z, &
         dummy, .true.)
    word = nexta4(comlyn,comlen)
    masflg = word == 'MASS'

    !     ---- count number of atoms involved
    n = 0
    do  i = 1, natom
       if (iats(i) == 1) then
          n = n + 1
       endif
    end do

#if KEY_DOMDEC==1
    if (n > 0) then
       call domdec_system_changed()
    endif
#endif

    !    write(*,*)'poidef>selection: ngraph, n=',ngraph,n

    !
    call chmalloc('rxndef.src','poidef','WTaa(ngraph)',N,crl=WTaa(ngraph)%a)
    call chmalloc('rxndef.src','poidef','iataa(ngraph)',n,intg=iataa(ngraph)%a)
    !
    !     ---- collect atom indices and weighting factors
    !
    k = 0
    do i = 1, natom
       if (iats(i) == 1) then
          if (masflg) then
             wtaa(ngraph)%a(k+1) = amass(i)
          else
             wtaa(ngraph)%a(k+1) = one
          end if
          iataa(ngraph)%a(k+1) = i
          k = k + 1
       end if
    end do

    !
    !     ---- normalize the weighting factors
    !
    sum = zero
    do  i = 1, n
       SUM = SUM + WTaa(ngraph)%a(i)
    end do
    do i = 1, n
       wtaa(ngraph)%a(i) = wtaa(ngraph)%a(i) / sum
    end do
    !
    !     ---- finalize the point definition
    !
    ! Conversion process (MH09) : we dont have heap
    ! so we just put 0 here and work with iat and wt directly
    ! in other routines
    deftyp(ngraph) = POINT_TYPE
    refnod(1,ngraph) = n
    refnod(2,ngraph) = ngraph
    ! now in case of dstype=POINT_TYPE we use this for
    ! index in which the iat data was stored
    refnod(3,ngraph) = 0 
    !      write(*,*)'poidef>refnod=',(refnod(i,ngraph),i=1,3)
    !      write(*,*)'poidef>iataa(ngraph)%a(1)=',iataa(ngraph)%a(1)
    !
    !     ---- return stack arrays
    !
    call chmdealloc('rxndef.src','poidef','iats',natom,intg=iats)
    !
    return
  end subroutine poidef
  !

  !---------------------------------------------------------------
  !       RXNSET
  !---------------------------------------------------------------
  subroutine rxnset
    !
    use number
    use stream,only:outu
    use string
    use memory
    use comand,only:comlyn,comlen

    !     ---- local variables
    !
    integer i, dstype, line, plane, line1, line2, plane1, &
         plane2, dir1, dir2, dir3, point, dir, k, low, high, &
         j, igraph, tmpnod, IR
    logical branch
    character(len=4) rxnnam
    !
    !     ---- convert all lines and planes to point-direction form
    !     ---- MFC09 comment above seems incorrect, not what this part does.
    !     ---- MFC09 rather, it puts the points defining the line into the refnod
    !     ----       for the item using the line instead of the reference to the
    !     ----       line, thus the item goes directly to the points.
    !
    loop300: do i = 1, ngraph
       dstype = deftyp(i)

       select case(dstype)
       case(DISTPTLN_TYPE)
          line = refnod(2,i)
          refnod(2,i) = refnod(1,line)
          refnod(3,i) = refnod(2,line)
       case(DISTPTPL_TYPE)
          plane = refnod(2,i)
          refnod(2,i) = refnod(1,plane)
          refnod(3,i) = refnod(2,plane)
       case(ANGLLNLN_TYPE)
          line1 = refnod(1,i)
          line2 = refnod(2,i)
          refnod(1,i) = refnod(2,line1)
          refnod(2,i) = refnod(2,line2)
          deftyp(i) = ANGLDRDR_TYPE
       case(ANGLLNPL_TYPE)
          line = refnod(1,i)
          plane = refnod(2,i)
          refnod(1,i) = refnod(2,line)
          refnod(2,i) = refnod(2,plane)
          deftyp(i) = ANGLDRDR_TYPE
       case(ANGLPLPL_TYPE)
!jms 4/2011
!this part doesn't work, this node has been bypassed, see modification below
#if KEY_ROLLRXNCOR==0
          plane1 = refnod(1,i)
          plane2 = refnod(2,i)
          refnod(1,i) = refnod(2,plane1)
          refnod(2,i) = refnod(2,plane2)
#endif 
          deftyp(i) = ANGLDRDR_TYPE
       case(PLAN3PT_TYPE)
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREPTPT_TYPE
          refnod(1,ngraph) = refnod(1,i)
          refnod(2,ngraph) = refnod(2,i)
          dir1 = ngraph
          !         ---- define direction2 from point1 to point3
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREPTPT_TYPE
          refnod(1,ngraph) = refnod(1,i)
          refnod(2,ngraph) = refnod(3,i)
          dir2 = ngraph
#if KEY_ROLLRXNCOR==1
! jms 2/1/2011: instead, define node directly as normal vector to plane                                                                                                                       
          deftyp(i) = DIREDRDR_TYPE
          refnod(1,i) = dir1
          refnod(2,i) = dir2
          !write(outu,*) "rxndef PLAN3PT_TYPE"
          !call printtree(1,ngraph)
#else /**/
          !         ---- define direction3 using direction1 and direction2
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREDRDR_TYPE
          refnod(1,ngraph) = dir1
          refnod(2,ngraph) = dir2
          dir3 = ngraph
          !         ---- set point-direction form for the plane
          refnod(2,i) = dir3
          refnod(3,i) = 0
          deftyp(i) = PLANPTDR_TYPE
#endif 
       case(PLANPTLN_TYPE)
          point = refnod (1,i)
          line = refnod(2,i)
          !         ---- define direction1 from point to line-point
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREPTPT_TYPE
          refnod(1,ngraph) = point
          refnod(2,ngraph) = refnod(1,line)
          dir1 = ngraph
          !         ---- define direction2 using direction1 and line-direction
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREDRDR_TYPE
          refnod(1,ngraph) = dir1
          refnod(2,ngraph) = refnod(2,line)
          dir2 = ngraph
          !         ---- set point-direction form for plane
          refnod(2,i) = dir2
          deftyp(i) = PLANPTDR_TYPE
       case(PLANPTNM_TYPE)
          line = refnod(2,i)
          refnod(2,i) = refnod(2,line)
          deftyp(i) = PLANPTDR_TYPE
       case(PLANPTLL_TYPE)
          line1 = refnod(1,i)
          line2 = refnod(2,i)
          !         ---- define direction using line1-direction and line2-direction
          ngraph = ngraph + 1
          deftyp(ngraph) = DIREDRDR_TYPE
          refnod(1,ngraph) = refnod(1,line1)
          refnod(2,ngraph) = refnod(2,line2)
          dir = ngraph
          !         ---- set point-direction form for plane
          refnod(2,i) = dir
          refnod(3,i) = 0
          deftyp(i) = PLANPTDR_TYPE
       case(PLANPTPL_TYPE)
          plane = refnod(2,i)
          refnod(2,i) = refnod(2,plane)
          deftyp(i) = PLANPTDR_TYPE
       end select
       !
    enddo loop300

    !     ARD and MFH 03-05-13
    !     Process the command line down here so we can loop.

    !     Number of reaction coordinates
    NRXNCR = GTRMI (COMLYN, COMLEN, 'NRXN', 1)

    call chmalloc('rxndef.src','rxnset','TREEHI',NRXNCR,intg=TREEHI)
    call chmalloc('rxndef.src','rxnset','TREELO',NRXNCR,intg=TREELO)
    !     Allocate arrays for force constant and center of umbrella
    !     as well as for dodgy biasing potential
    call chmalloc('rxndef.src','rxnset','UMBFRM',NRXNCR,intg=UMBFRM)
    call chmalloc('rxndef.src','rxnset','KUMBPR',NRXNCR,crl=KUMBPR)
    call chmalloc('rxndef.src','rxnset','PUMBPR',NRXNCR,crl=PUMBPR)
    call chmalloc('rxndef.src','rxnset','DL0PTR',3*NRXNCR,crl=DL0PTR)
    call chmalloc('rxndef.src','rxnset','NBIASP',NRXNCR,intg=NBIASP)
    call chmalloc_chm_array('rxndef.src','rxnset','RBIASP',NRXNCR,RBIASP)
    call chmalloc_chm_array('rxndef.src','rxnset','CBIASP',NRXNCR,CBIASP)
#if KEY_SMD==1
    call chmalloc('rxndef.src','rxnset','QFLIP',NRXNCR,intg=QFLIP) 
#endif
#if KEY_SMD==1
    call chmalloc('rxndef.src','rxnset','SMDDEL',NRXNCR,crl=SMDDEL)
#endif
    !     Now intitialize these in case rxnene is called without umbrellas
    DO IR = 1, NRXNCR
       umbfrm(ir) = 1
       kumbpr(ir) = zero
       pumbpr(ir) = zero
       dl0ptr(3*(ir-1)+1) = zero
       dl0ptr(3*(ir-1)+2) = zero
       dl0ptr(3*(ir-1)+3) = zero
       nbiasp(ir) = 0
       qflip(ir) = 0
#if KEY_SMD==1
       smddel(ir) = zero 
#endif
    ENDDO

    !     Initialize the looping construct
    IR = 1
    treelo(1) = ngraph + 1
    RXNNAM = NEXTA4 (COMLYN, COMLEN)

    !     Begin loop over reaction coordinates
    loop350: do while(.true.)
       RXNIND = INDGRF (RXNNAM,1,NGRAPH,NODNAM,0) ! 0->out,optional
       K = TREELO(IR)
       LOW = K
       HIGH = LOW
       !
       !     ---- make tree
       !     ---- (deftyp(i=low,high) is used to store igraph temporarily)
       !
       deftyp(k) = rxnind

       branch=.true.
       loop20: do while(branch)
          branch = .false.
          loop320: do i = low, high
             igraph = deftyp(i)
             nodnam(i)   = nodnam(igraph)
             if (deftyp(igraph) == POINT_TYPE) then
                deftyp(i) = POINT_TYPE
                refnod(1,i) = refnod(1,igraph)
                refnod(2,i) = refnod(2,igraph)
                refnod(3,i) = refnod(3,igraph)
             else
                branch = .true.
                deftyp(i) = deftyp(igraph)
                loop310: do j = 1, 3
                   tmpnod = refnod(j,igraph)
                   if (tmpnod /= 0) then
                      k = k + 1
                      IF (K > MAXNOD) CALL WRNDIE(-5,'<RXNSET>', &
                           'Tree exceeded MAXNOD')
                      deftyp(k) = tmpnod
                      refnod(j,i) = k
                   end if
                enddo loop310
                delval(1,i) = delval(1,igraph)
                delval(2,i) = delval(2,igraph)
                delval(3,i) = delval(3,igraph)
                !           ARD and Jie Hu 06-06-30 for VCOM
                delval(4,i) = delval(4,igraph)
                delval(5,i) = delval(5,igraph)
             end if
          enddo loop320
          low = high + 1
          high = k
       enddo loop20
       !     End inner loop for tree

       !     Each tree spans nodes TREELO (root) to TREEHI (branch tips)
       TREEHI(IR) = k
       !     See if there is another reaction coordinate
       RXNNAM = NEXTA4(COMLYN,COMLEN)
       IF (RXNNAM  /=  ' ') THEN
          IR = IR + 1
          treelo(ir) = k + 1
       else
          exit loop350
       ENDIF
       !     End loop over reaction coordinates
    enddo loop350

    IF (NRXNCR  /=  IR) THEN
       CALL WRNDIE(0,'<RXNSET>', &
            'Resetting NRXN to number of reaction coordinates entered')
       NRXNCR = IR
    ENDIF

    nrxn = k
    !     Set RXNIND to zero to suppress calling rxnene to use coordinates
    !     for TPS and not umbrella sampling.  Setting the umbrella or trace
    !     restores RXNIND.
    RXNIND = 0
    !
    return
  end subroutine rxnset

 !----Puja---------- [QC: 11/17]
    subroutine protonchdef
    use exfunc
    use comand
    use psf
    use coord
    use rxncom
    use memory
    use string
    use select
    !
    ! Peter H. Koenig, Nilanjan Ghosh, Mai 2004, August 2004
    ! Define a proton transfer channel
    ! for CEC or mCEC
    !
    ! create data structures needed for computing the CEC
    integer:: i, k1, k2
    integer:: n1, n2
    integer, allocatable :: iats1(:), iats2(:)
    real(chm_real)::  dummy(2)
    character(len=4):: word
    real(chm_real):: rsw, dsw
    ! debug
    !write(*,*) 'protonchdef, got here'
    allocate(iats1(natom))
    allocate(iats2(natom))
    ! get selections passed as arguments to the command
    call selcta (comlyn, comlen, iats1, x, y, z,dummy, .true.)
    call selcta (comlyn, comlen, iats2, x, y, z,dummy, .true.)
    n1 = 0
    n2 = 0
    do i = 1, natom
       if (iats1(i)==1) n1 = n1 + 1
       if (iats2(i)==1) n2 = n2 + 1
    enddo

    ! allocate space on heap for remembering which
    ! atoms are included in the coordinate
    call chmalloc('rxndef.src','protonchdef','iat11',n1+1,intg=iat11)
    call chmalloc('rxndef.src','protonchdef','iat12',n2+1,intg=iat12)
    call chmalloc('rxndef.src','protonchdef','wt1',n1+2,crl=wt1)
    iat11(1) = n1
    iat12(1) = n2
    n1 = 0
    n2 = 0
    rsw= GTRMF(COMLYN,COMLEN,'RSW', 1.2d0)
    dsw= GTRMF(COMLYN,COMLEN,'DSW', 0.04d0)
    write(*,*) "mCEC: Using r_sw=",rsw," and d_sw=",dsw
    wt1(1)=rsw
    wt1(2)=dsw
    ! copy over the weights and atom numbers to the arrays over to
    ! the heap
    do i=1,natom
       if (iats1(i)==1) then
          n1 = n1 + 1
          iat11(n1+1) = i
          wt1(n1+2)=WMAIN(i)
          ! debug
          !write(*,*) 'oxygen atom index is: ', i
       endif
    enddo
    do i=1,natom
       if (iats2(i)==1) then
          n2 = n2 + 1
          iat12(n2+1) = i
       endif
    enddo
  ! refnod(1,ngraph) = iat1
  ! refnod(2,ngraph) = iat2
  ! refnod(3,ngraph) = wt
    deallocate(iats1)
    deallocate(iats2)

     return

     END  subroutine protonchdef

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     subroutine protonchdef2

     use exfunc
     use comand
     use psf
     use coord
     use rxncom
     use memory
     use string
     use select
     !
     ! Peter H. Koenig, Nilanjan Ghosh, Mai 2004, August 2004
     ! Define a proton transfer channel
     ! for CEC or mCEC
     !
     ! create data structures needed for computing the CEC


     integer:: i, k1, k2
     integer:: n1, n2
     integer, allocatable :: iats1(:), iats2(:)
     real(chm_real)::  dummy(2)
     character(len=4):: word
     real(chm_real):: rsw, dsw

     ! debug
     !write(*,*) 'protonchdef, got here'

     allocate(iats1(natom))
     allocate(iats2(natom))

     ! get selections passed as arguments to the command
     call selcta (comlyn, comlen, iats1, x, y, z,dummy, .true.)
     call selcta (comlyn, comlen, iats2, x, y, z,dummy, .true.)

     n1 = 0
     n2 = 0

     do i = 1, natom
        if (iats1(i)==1) n1 = n1 + 1
        if (iats2(i)==1) n2 = n2 + 1
     enddo

     ! allocate space on heap for remembering which
     ! atoms are included in the coordinate


     call chmalloc('rxndef.src','protonchdef','iat21',n1+1,intg=iat21)
     call chmalloc('rxndef.src','protonchdef','iat22',n2+1,intg=iat22)
     call chmalloc('rxndef.src','protonchdef','wt2',n1+2,crl=wt2)

     iat21(1) = n1
     iat22(1) = n2

     n1 = 0
     n2 = 0

     rsw= GTRMF(COMLYN,COMLEN,'RSW', 1.2d0)
     dsw= GTRMF(COMLYN,COMLEN,'DSW', 0.04d0)

     write(*,*) "mCEC: Using r_sw=",rsw," and d_sw=",dsw

     wt2(1)=rsw
     wt2(2)=dsw

     ! copy over the weights and atom numbers to the arrays over to
     ! the heap

     do i=1,natom
        if (iats1(i)==1) then
           n1 = n1 + 1
           iat21(n1+1) = i
           wt2(n1+2)=WMAIN(i)
           ! debug
           !write(*,*) 'oxygen atom index is: ', i
        endif
     enddo

     do i=1,natom
        if (iats2(i)==1) then
           n2 = n2 + 1
           iat22(n2+1) = i
        endif
     enddo

   ! refnod(1,ngraph) = iat1
   ! refnod(2,ngraph) = iat2
   ! refnod(3,ngraph) = wt

     deallocate(iats1)
     deallocate(iats2)

     return

     END  subroutine protonchdef2

  !----Puja---------- [QC: 11/17]


  !---------------------------------------------------------------
  !      INDGRF
  !---------------------------------------------------------------
  INTEGER FUNCTION INDGRF (NAME,IS,N,NAMLST,MODE,INDLST)
    !
    !       Find the index of the name in namlst.
    !       Generalized a bit by Aaron R. Dinner 03-05-13
    !
    integer,optional :: indlst(:)
    INTEGER IS, N, MODE
    CHARACTER(len=4) NAME, NAMLST(:)
    !
    INTEGER I, J, K
    !
    INDGRF = 0
    DO I = IS, N
       IF (MODE  ==  1) THEN
          K = INDLST(I)
       ELSE
          K = I
       ENDIF
       IF (NAMLST(K) == NAME) INDGRF = I
    enddo
    !
    IF (INDGRF == 0) CALL WRNDIE (-3, 'rxndef.src<INDGRF>', &
         NAME // ' NOT DEFINED' )
    !
    RETURN
  END function indgrf

  !---------------------------------------------------------------
  !       SETSTT
  !---------------------------------------------------------------
  SUBROUTINE SETSTT(COMLYN,COMLEN,IR,LODEL,HIDEL,DELDEL, &
       NRXSTT,NMLSTT,TREELO,NRXNCR,NODNAM)
    !
    !       Setup for statistics for multiple reaction coordinates.
    !       ARD 03-05-13 based on RXPARS code.
    !
    use number
    use string
    INTEGER COMLEN
    INTEGER IR, NRXSTT(:), NMLSTT(:), TREELO(:), NRXNCR
    real(chm_real)  LODEL(:), HIDEL(:), DELDEL(:)
    CHARACTER(len=*) COMLYN
    !
    INTEGER NR, ILEN
    CHARACTER(len=4) RXNNAM, NODNAM(:)
    !
    !
    !       Initialize the arrays in case they are not named.
    DO IR = 1, NRXNCR
       LODEL(IR)  = MINONE
       HIDEL(IR)  = ONE
       DELDEL(IR) = PTONE
       NRXSTT(IR) = nint((HIDEL(IR) - LODEL(IR)) / DELDEL(IR))
    ENDDO
    !
    !       Setup for the looping construct.
    CALL GTRMWD(COMLYN,COMLEN,'NAME',4,RXNNAM,80,ILEN)
    IF (ILEN  ==  0) THEN
       CALL WRNDIE (2, '<SETSTT>', &
            'NO NAME SPECIFIED:  DEFAULTING TO FIRST COORDINATE')
       IR = 1
    ELSE
       IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)
       IF (IR  ==  0) IR = 1
    ENDIF

    NR = 0
    !       Loop over the command line to get all the ranges.
10  NR = NR + 1
    LODEL(IR)  = GTRMF(COMLYN, COMLEN, 'LOWD', MINONE)
    HIDEL(IR)  = GTRMF(COMLYN, COMLEN, 'HIDE',    ONE)
    DELDEL(IR) = GTRMF(COMLYN, COMLEN, 'DELD',  PTONE)
    NRXSTT(IR) = nint((HIDEL(IR) - LODEL(IR)) / DELDEL(IR))

    CALL GTRMWD(COMLYN,COMLEN,'NAME',4,RXNNAM,80,ILEN)
    IF (ILEN  >  0) THEN
       IR = INDGRF(RXNNAM,1,NRXNCR,NODNAM,1,TREELO)
       IF (IR  ==  0) IR = 1
       GOTO 10
    ENDIF

    !       Repeating the same name multiple times breaks this check
    IF (NRXNCR  /=  NR) CALL WRNDIE(-2,'<SETSTT>', &
         'Number of ranges is not equal to number of coordinates')

    !       Setup an array for indexing convenience (see RXNSTT).
    !       Do it backwards so fastest varying coordinates are last.
    NMLSTT(NRXNCR) = 1
    DO IR = NRXNCR-1, 1, -1
       NMLSTT(IR) = NMLSTT(IR+1)*NRXSTT(IR+1)
    ENDDO
    RETURN
  END SUBROUTINE SETSTT

  !---------------------------------------------------------------
  !       RDSTT
  !---------------------------------------------------------------
  SUBROUTINE RDSTT(RXNUNT,DELSTT,NRXNCR,NMLSTT,NRXSTT, &
       LODEL,DELDEL,DEL)
    !
    !       Read statistics for multiple reaction coordinates.
    !       ARD 03-05-13 based on RXPARS code.
    !
    use stream

    INTEGER RXNUNT, NRXNCR, NMLSTT(:), NRXSTT(:), DELSTT(:)
    real(chm_real)  DEL(:), LODEL(:), DELDEL(:)
    !
    INTEGER I, J, K, N, FREQ
    real(chm_real)  HELMHO

    IF (RXNUNT  <=  0) THEN
       N = NMLSTT(1)*NRXSTT(1)
       DO I = 1, N
          DELSTT(I) = 0
       ENDDO
       RETURN
    ENDIF
    !       Begin loop over file
10  CONTINUE
    !         Read the line
    IF (IOLEV  >  0) READ (RXNUNT,*,END=20,ERR=30) &
         (DEL(I),I=1,NRXNCR), HELMHO, FREQ
    !         Determine the one-dimensional index and save FREQ in DELSTT
    ! DTM debug
    ! K = 0
    K = 1
    DO I = 1, NRXNCR
    ! DTM debug
    !   J = INT((DEL(I)-LODEL(I))/DELDEL(I)) + 1
       J = INT((DEL(I)-LODEL(I))/DELDEL(I))
       K = K + NMLSTT(I)*J
       DELSTT(K) = FREQ
    ENDDO
    GOTO 10
    !       End loop over file

20  RETURN

    !       Error handling
30  CALL WRNDIE (-5, '<RDSTT>', &
         'Error while reading previous statistics')
    !
    RETURN
  END SUBROUTINE RDSTT

  !---------------------------------------------------------------
  !       WRSTT
  !---------------------------------------------------------------
  SUBROUTINE WRSTT(RXNUNT,DELSTT,NRXNCR,NMLSTT,NRXSTT,TREELO, &
       UMFORM,KUMB,DELTA0,NBIASP,RBIASP,CBIASP, &
       LODEL,DELDEL,DEL,RXNCNT,DEFT,PUMB &
#if KEY_SMD==1
       ,SMDDEL,BASALO,BASAHI,BASBLO,BASBHI,QFLIP  & 
#endif
       )
    !
    !       Write statistics for multiple reaction coordinates.
    !       ARD 03-05-13 based on RXPARS code.
    !
    use consta
    use number
    use reawri
    use stream
    use string
#if KEY_PARALLEL==1
    use parallel   
#endif
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
    use domdec_io,only:send_str_to_root, send_nostr_to_root
#endif

    INTEGER RXNUNT, DELSTT(:), NMLSTT(:), NRXSTT(:), NRXNCR
    INTEGER TREELO(:), UMFORM(:), NBIASP(:), RXNCNT
    !       RBIAS and CBIAS are arrays of pointers to real
    type(chm_array) RBIASP(:), CBIASP(:)
    real(chm_real)  LODEL(:), DELDEL(*), DEL(*)
    real(chm_real)  DELTA0(*), KUMB(:), PUMB(:)
    INTEGER DEFT
#if KEY_SMD==1
    real(chm_real)  SMDDEL(:), BASALO(:), BASAHI(:), BASBLO(:),  &
         BASBHI(:)
    INTEGER QFLIP(:)
#endif 
    !
    INTEGER I, J, K, N, IR
    real(chm_real)  EUMB, D(3), E, HELMHO
    CHARACTER(len=4)   NSTRING
    CHARACTER(len=100) FSTRING
#if KEY_DOMDEC==1
    ! NOTE: this limits the line length to 2048 characters
    character(2048) rxnstr
#endif
    !
    !       Code added by Ben Webb, 2000
    !       This code must only be run on node zero, as it writes to files...
    !  APH 10/11/2014, node zero restriction lifted since we want to allow DOMDEC constraint node
    !  to go here as well. decision on what nodes enter here must be made before calling this routine.
!!$#if KEY_PARALLEL==1
!!$    IF (MYNOD == 0) THEN                     
!!$#endif
!!$       !
!!$       IF (PRNLEV  <  2 .OR. RXNUNT  <  0) RETURN

       !       Format varies with number of reaction coordinates
       CALL ENCODI(NRXNCR+1,NSTRING,4,N)
       FSTRING = '(5X,'//NSTRING//'(F10.5, 5X), I10, 5X, 2(F10.5, 5X))'

       !       Figure out total size of DELSTT array
       N = NMLSTT(1)*NRXSTT(1)

       DO I = 1, N
          IF (DELSTT(I)  >  0) THEN

             EUMB = ZERO
             K = I
             DO IR = 1, NRXNCR
                !             Get index
                J = INT((K-1)/NMLSTT(IR))
                K = K - J*NMLSTT(IR)
                J = J + 1
                !             Convert to real
                E = LODEL(IR) + (J - 0.5)*DELDEL(IR)
                DEL(IR) = LODEL(IR) + (J - 0.5)*DELDEL(IR)
                !             Get energy
                ! MH09: This is a bug              CALL UMBPOT(E,D,DEL(IR),IR,UMFORM, &
                ! MH09: del should be defined something like del(3,nrxncr)
                ! MH09: leave it as was in the original code. When called from energy
                ! MH09: i think it is OK! ( or maybe not: it is coming from delval(5,maxnod)???)
                CALL UMBPOT(E,D,DEL(IR),IR,UMFORM, &
                     KUMB,DELTA0,NBIASP,CBIASP,RBIASP,DEFT,PUMB &
#if KEY_SMD==1
                     ,SMDDEL,BASALO,BASAHI,BASBLO,BASBHI,QFLIP  & 
#endif
                     )
                EUMB = EUMB + E
             ENDDO

             !           Write
! DTM debug - account for data not collected. Routine is only called if RXNCNT>STTSTP
!             HELMHO = -KBOLTZ*FIRSTT*LOG(DBLE(DELSTT(I))/DBLE(RXNCNT))
             HELMHO = -KBOLTZ*FIRSTT*LOG(DBLE(DELSTT(I))/DBLE(RXNCNT-STTSTP))

             !           Old format was without HELMO and EUMB on end and was selected
             !           based on whether BIAS was present.  With multiple coordinates,
             !           some could have BIAS and others not, so write full information
             !           always.
             !
             ! APH 11/10/2014, In case of DOMDEC, the root node might not have the results,
             ! in that case, send the results to the root node from the constraint node
#if KEY_DOMDEC==1
             if (rxnunt < 0) then
                WRITE (rxnstr,FSTRING) (DEL(IR),IR=1,NRXNCR), &
                     HELMHO-EUMB, DELSTT(I), HELMHO, EUMB
                call send_str_to_root(trim(rxnstr))
             else
#endif
                WRITE (RXNUNT,FSTRING) (DEL(IR),IR=1,NRXNCR), &
                     HELMHO-EUMB, DELSTT(I), HELMHO, EUMB
#if KEY_DOMDEC==1
             endif
#endif

          ENDIF
       ENDDO
!!$#if KEY_PARALLEL==1
!!$    ENDIF                                    
!!$#endif
#if KEY_DOMDEC==1
       if (rxnunt < 0) then
          call send_nostr_to_root()
       endif
#endif

    RETURN
  END SUBROUTINE WRSTT
  
#if KEY_DOMDEC==1
  ! *
  ! * The DOMDEC root node receives the results from the constraint node
  ! * and writes the result to file
  ! *
  subroutine wrstt_root(rxnunt)
    use domdec_io,only:recv_str_from_node
    use domdec_dr_common,only:cons_node
    implicit none
    ! Input
    integer, intent(in) :: rxnunt
    ! Variables
    ! NOTE: this limits the line length to 2048 characters
    character(2048) str
    integer nstr

    do while (.true.)
       call recv_str_from_node(str, nstr, cons_node)
       if (nstr <= 0) exit
       write (rxnunt,'(a)') str(1:nstr)
    enddo

    return
  end subroutine wrstt_root

  ! *
  ! * Check to make sure the calls to rxnene were made from the same node.
  ! *
  subroutine check_comp_nod(comp_nod)
    use rxncom,only:q_comp_nod
    use mpi,only:mpi_success, mpi_integer, mpi_sum
    use parallel,only:mynod, comm_charmm
    implicit none
    ! Output
    integer, intent(out) :: comp_nod
    ! Variables
    integer comp_int, comp_int_sum
    integer ierror
    
    if (q_comp_nod) then
       comp_int = mynod
    else
       comp_int = 0
    endif
    
    call mpi_allreduce(comp_int, comp_int_sum, 1, mpi_integer, mpi_sum, comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<rxndef>',&
            'Error calling mpi_reduce in check_comp_nod')
    endif

    if (q_comp_nod) then
       if (comp_int_sum /= mynod) then
          call wrndie(-5,'<rxndef>',&
               'Calls to RXNENE made from different nodes. &
               This can happen when mixing DOMDEC with non-DOMDEC dynamics or when NDIR is changed &
               between dynamics calls.')
       endif
    endif

    call mpi_barrier(comm_charmm, ierror)
    if (ierror /= mpi_success) then
       call wrndie(-5,'<rxndef>',&
            'Error calling mpi_barrier in check_comp_nod')
    endif

    comp_nod = comp_int_sum

    return
  end subroutine check_comp_nod
#endif

  !---------------------------------------------------------------
  !       RXNBF
  !---------------------------------------------------------------
  SUBROUTINE RXNBF(IR,NBIASP,RBIASP,CBIASP)
    !
    use ctitla
    use memory
    use stream
    use string
    use comand

    !yw...01/26/2001
    ! Variables already defined in rxncom.f90, delete:
    INTEGER, PARAMETER :: NPOINTS=25
    !      INTEGER NBIASP
    !      real(chm_real)  CBIAS(4,NPOINTS), RBIAS(NPOINTS), UBIAS(NPOINTS)
    !yw
    INTEGER IR, NBIASP(:)
    type(chm_array) :: RBIASP(:), CBIASP(:)
    !
    INTEGER IUNIT,I
    real(chm_real) Y,S(NPOINTS),A(NPOINTS,4),UBIAS(NPOINTS)
    integer j,kndx
    real(chm_real) delx,eb2,x0

    IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',5)
    IF(PRNLEV >= 2) WRITE(OUTU,100) IUNIT
100 FORMAT( &
         ' RXNBF: biasing potential for rxncor being read from unit',I4)
    CALL RDTITL(TITLEB,NTITLB,IUNIT,0)
    READ (IUNIT,*) NBIASP(IR)
    IF(NBIASP(IR) <= 0 .OR. NBIASP(IR) > NPOINTS) THEN
       CALL WRNDIE(0,'RNXBIA','Error reading biasing potential')
       CALL WRNDIE(0,'RNXBIA','Maximum number of points = 25  ')
    ENDIF
    call chmalloc('rxndef.src','RXNBF','RBIASP(IR)',NBIASP(IR)*4,crl=RBIASP(IR)%a)
    call chmalloc('rxndef.src','RXNBF','CBIASP(IR)',NBIASP(IR)*4,crl=CBIASP(IR)%a)
    CALL RDBIAS(IUNIT,NBIASP(IR),RBIASP(IR)%a,UBIAS)
    !
    !     INITIALIZING SPLINE COEFFIECIENTS
    !
    Y = UBIAS(1)
    DO I = 1,NBIASP(IR)
       UBIAS(I) = UBIAS(I) - Y
    ENDDO
    CALL SETSPLN(RBIASP(IR)%a,UBIAS,A,S,NBIASP(IR),1)
    CALL BIACOE(CBIASP(IR)%a,RBIASP(IR)%a,UBIAS,S,NBIASP(IR))
    !
    RETURN
  END SUBROUTINE RXNBF

  !---------------------------------------------------------------
  !       RDBIAS
  !---------------------------------------------------------------
  SUBROUTINE RDBIAS(IUNIT,N,RBIAS,UBIAS)
    !
    INTEGER IUNIT,N
    real(chm_real)  RBIAS(:), UBIAS(:)
    INTEGER I
    DO I = 1,N
       READ(IUNIT,*,ERR=999,END=999) RBIAS(I),UBIAS(I)
    ENDDO
    RETURN
999 CALL WRNDIE(-3,'<RDBIAS>','Error during read')
    RETURN
  END SUBROUTINE RDBIAS

  !---------------------------------------------------------------
  !       SETSPLN
  !---------------------------------------------------------------
  SUBROUTINE SETSPLN(X,Y,A,S,N,IEND)
    use number
    !
    INTEGER N,IEND, I,J,NM2,NM1
    real(chm_real) X(N),Y(N),A(N,4),S(N)
    !
    real(chm_real)  DX1,DX2,DY1,DY2,DXN1,DXN2

    DO I = 1,N
       A(I,1) = zero
       A(I,2) = zero
       A(I,3) = zero
       A(I,4) = zero
    ENDDO
    NM2 = N-2
    NM1 = N-1
    DX1 = X(2)-X(1)
    DY1 = (Y(2)-Y(1))/DX1*six
    DO I = 1,NM2
       DX2 = X(I+2)-X(I+1)
       DY2 = (Y(I+2)-Y(I+1))/DX2*six
       A(I,1) = DX1
       A(I,2) = 2.0D0 * (DX1+DX2)
       A(I,3) = DX2
       A(I,4) = DY2 - DY1
       DX1 = DX2
       DY1 = DY2
    ENDDO

    select case(iend)
    case(2)
       A(1,2) = A(1,2) + X(2) - X(1)
       A(NM2,2) = A(NM2,2) + X(N) - X(NM1)
    case(3)
       DX1    = X(2) - X(1)
       DX2    = X(3) - X(2)
       A(1,2) = (DX1+DX2)*(DX1+2.0D0*DX2)/DX2
       A(1,3) = (DX2**2 - DX1**2)/DX2
       DXN2   = X(NM1) - X(NM2)
       DXN1   = X(N) - X(NM1)
       A(NM2,1) = (DXN2**2 - DXN1**2)/DXN2
       A(NM2,2) = (DXN1 + DXN2) * (DXN1 + 2.0D0*DXN2)/DXN2
    end select

    loop110: DO I = 2,NM2
       A(I,2) = A(I,2)-A(I,1)/A(I-1,2)*A(I-1,3)
       A(I,4) = A(I,4)-A(I,1)/A(I-1,2)*A(I-1,4)
    enddo loop110
    !
    A(NM2,4) = A(NM2,4)/A(NM2,2)
    DO I = 2,NM2
       J = NM1 - I
       A(J,4) = (A(J,4) - A(J,3) * A(J+1,4))/A(J,2)
    enddo

    DO  I = 1,NM2
       S(I+1) = A(I,4)
    enddo

    select case (iend)
    case(1)
       S(1) = zero
       S(N) = zero
    case(2)
       S(1) = S(2)
       S(N) = S(N-1)
    case(3)
       S(1) = ((DX1+DX2)*S(2) + DX1 * S(3))/DX2
       S(N) = ((DXN2+DXN1)*S(NM1) - DXN1*S(NM2))/DXN2
    end select
    RETURN
  END subroutine setspln

  !---------------------------------------------------------------
  !       BIACOE
  !---------------------------------------------------------------
  SUBROUTINE BIACOE(C,X,Y,S,N)
    !
    INTEGER N,I
    real(chm_real) X(N),Y(N),C(4,N),S(N),DELX
    !
    DO I = 1,N-1
       DELX = X(I+1)-X(I)
       C(1,I) = (S(I+1)-S(I))/(6.0D0*DELX)
       C(2,I) = S(I)/2.0D0
       C(3,I) = (Y(I+1)-Y(I))/DELX-(2.0D0*DELX*S(I)+DELX* &
            S(I+1))/6.0D0
       C(4,I) = Y(I)
    ENDDO
    RETURN
  END SUBROUTINE BIACOE


#else /*  (rxndefs_main)*/
  subroutine rxpars
    CALL WRNDIE(-1,'<RXPARS>','RXNCOR code is not compiled.')
    RETURN
  end subroutine rxpars
#endif /*  (rxndefs_main)*/

  !---------------------------------------------------------------
  !       AllocMixOrder
  !---------------------------------------------------------------
  subroutine allocmixorder
    use memory
    use number
#if KEY_TPS==1
    if(.not.allocated(basalo)) then
       call chmalloc('rxndef.src','rxpars','BASALO',NRXNCR,crl=BASALO)
       call chmalloc('rxndef.src','rxpars','BASAHI',NRXNCR,crl=BASAHI)
       call chmalloc('rxndef.src','rxpars','BASBLO',NRXNCR,crl=BASBLO)
       call chmalloc('rxndef.src','rxpars','BASBHI',NRXNCR,crl=BASBHI)
    endif
    BASALO(1:NRXNCR) = ZERO
    BASAHI(1:NRXNCR) = ZERO
    BASBLO(1:NRXNCR) = ZERO
    BASBHI(1:NRXNCR) = ZERO
#endif 
    return
  end subroutine allocmixorder

#if KEY_ROLLRXNCOR==1
!jms 9/8/11
  subroutine printtree(start,end)
  use stream
  integer start,end
  integer i
  character(len=4) name
        do i=start,end
          name=nodnam(i)
          if(ichar(name(1:1)).eq.0) name="----"
          write(outu,'(I4,1X,A4,4I4,3F12.5)') i,name,deftyp(i),&
            refnod(1,i),refnod(2,i),refnod(3,i),delval(1,i),delval(2,i),&
            delval(3,i)
        end do
  end subroutine printtree
#endif 

end module rxdefs

