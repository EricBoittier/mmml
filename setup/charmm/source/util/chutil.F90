module chutil
  use chm_kinds
  implicit none

contains
  SUBROUTINE MAKIND(NATOM,ISLCT,LSTPRP,NTPRP)
    !-----------------------------------------------------------------------
    !  Make the list of the atoms that from a selection
    !  All atoms are chosen by default.
    !
    INTEGER NATOM, ISLCT(*), LSTPRP(*), NTPRP

    ! Local variables
    INTEGER I
    !
    NTPRP=0
    DO I=1,NATOM
       IF(ISLCT(I) == 1)THEN
          NTPRP=NTPRP+1
          LSTPRP(NTPRP)=I
       ENDIF
    ENDDO

    RETURN
  END SUBROUTINE MAKIND

  INTEGER FUNCTION GETATN(SEG,RES,ATOM, &
       SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
    !-----------------------------------------------------------------------
    !     THIS FUNCTION RETURNS THE ATOM NUMBER FOR THE ATOM NAMED BY ITS
    !     SEGID, RESIDE, AND IUPAC NAME.
    !
    !     Author: David States
    !
  use stream
    CHARACTER(LEN=*) ATYPE(*),RESID(*),SEGID(*)
    INTEGER NICTOT(*)
    INTEGER IBASE(*)
    CHARACTER(LEN=*) SEG,RES,ATOM
    INTEGER NSEG
    INTEGER IRES,ISEG,ISTOP,IATOM
    INTEGER ISTART  ! Handle * as the 1st character,June2008 bt
    !
    ! FIRST FIND THE SEGMENT NUMBER.
    !
    ISEG=1
    IF (NSEG == 0) GOTO 999
    DO WHILE (SEGID(ISEG) /= SEG)
       ISEG=ISEG+1
       IF (ISEG > NSEG) GOTO 999
    ENDDO
    !
    ! NOW FIND THE RESID USING NICTOT TO GET THE RESIDUES IN THE SEGMENT.
    !
    IRES=NICTOT(ISEG)+1
    ISTOP=NICTOT(ISEG+1)
    IF (IRES > ISTOP) GOTO 998
    DO WHILE (RESID(IRES) /= RES)
       IRES=IRES+1
       IF (IRES > ISTOP) GOTO 998
    ENDDO
    !
    !     FINALLY WE GET THE ATOM NUMBER.
    !
    !     IATOM=IBASE(IRES)+1
    !      ISTOP=IBASE(IRES+1)
    !      IF (IATOM > ISTOP) GOTO 997
    !      DO WHILE (ATYPE(IATOM) /= ATOM)
    !         IATOM=IATOM+1
    !         IF (IATOM > ISTOP) GOTO 997
    !      ENDDO
    !
    ! Handle * as the first character, June 2008   bt
    ISTART=NICTOT(ISEG)+1
    ISTOP=NICTOT(ISEG+1)
    IATOM=MATOM(IRES,ATOM,ATYPE,IBASE,ISTART,ISTOP,.TRUE.)
    ! June 2008 bt
    !
    GETATN=IATOM
    RETURN
999 IF(WRNLEV >= 2) WRITE(OUTU,2000) 'SEGMENT', &
         SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
    GETATN=-1
    RETURN
998 IF(WRNLEV >= 2) WRITE(OUTU,2000) 'RESIDUE', &
         SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
    GETATN=-1
    RETURN
997 IF(WRNLEV >= 2) WRITE(OUTU,2000) 'ATOM', &
         SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
    GETATN=-1
    RETURN
2000 FORMAT(' *****  WARNING  ***** GETATN could not find the ',A, &
         ' SEG="',A,'" RES="',A,'" ATOM="',A,'"')
  END FUNCTION GETATN

  INTEGER FUNCTION GETRES(ATOM,IBASE,NRES)
    !
    !     GIVEN AN ATOM NUMBER, THIS FUNCTION RETURNS THE RESIDUE NUMBER
    !     TO WHICH THE ATOM BELONGS.
    !
    !     Author: Robert Bruccoleri
    !
  use stream
    INTEGER ATOM,IBASE(*)
    INTEGER NRES
    INTEGER START,STOP,TRY
    !
    !     WE DO A BINARY SEARCH THROUGH IBASE.
    !
    START=1
    STOP=NRES
1   TRY=(START+STOP)/2
    IF(ATOM > IBASE(TRY) .AND. ATOM <= IBASE(TRY+1)) GOTO 2
    IF(ATOM <= IBASE(TRY)) STOP=TRY-1
    IF(ATOM > IBASE(TRY+1)) START=TRY+1
    IF(START <= STOP) GOTO 1
    IF(WRNLEV >= 2) WRITE(OUTU,200)ATOM
200 FORMAT(' ERROR IN GETRES. COULDNT FIND ATOM ',I5,'.')
    !     CALL DIE
    TRY=0
2   GETRES=TRY
    RETURN
  END FUNCTION GETRES

  INTEGER FUNCTION GETSEG(IRES,NICTOT,NSEG)
    !
    !     GIVEN AN RESIDUE NUMBER IN IRES, THIS FUNCTION RETURNS THE SEGMENT
    !     NUMBER TO WHICH THE RESIDUE BELONGS.
    !
    !     Author: Robert Bruccoleri
    !
  use stream
  use machutil,only:die
    INTEGER IRES,NSEG
    INTEGER NICTOT(*)
    INTEGER START,STOP,TRY
    !
    !     WE DO A BINARY SEARCH THROUGH NICTOT.
    !
    START=1
    STOP=NSEG
1   TRY=(START+STOP)/2
    IF(IRES > NICTOT(TRY) .AND. IRES <= NICTOT(TRY+1)) GOTO 2
    IF(IRES <= NICTOT(TRY)) STOP=TRY-1
    IF(IRES > NICTOT(TRY+1)) START=TRY+1
    IF(START <= STOP) GOTO 1
    IF(WRNLEV >= 2) WRITE(OUTU,200) IRES
200 FORMAT(' ERROR IN GETSEG. COULDNT FIND RESIDUE ',I5,'.')
    CALL DIE
2   GETSEG=TRY
    RETURN
  END FUNCTION GETSEG

  INTEGER FUNCTION GETRSN(SEG,RES,ATOM, &
       SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG)
    !
    !     THIS FUNCTION RETURNS THE RESIDUE NUMBER FOR THE ATOM NAMED BY ITS
    !     SEGID, RESIDE, AND IUPAC NAME.  IF ATOM IS ' ' THEN IT IS IGNORED.
    !
  use stream
    CHARACTER(LEN=*) SEG,RES,ATOM
    INTEGER NSEG
    CHARACTER(LEN=*) ATYPE(*),RESID(*),SEGID(*)
    INTEGER NICTOT(*)
    INTEGER IBASE(*)
    INTEGER IRES,ISEG,ISTOP,IATOM
    !
    !     FIRST FIND THE SEGMENT NUMBER.
    !
    ISEG=1
    IF (NSEG == 0) GOTO 999
    DO WHILE (SEGID(ISEG) /= SEG)
       ISEG=ISEG+1
       IF (ISEG > NSEG) GOTO 999
    ENDDO
    !
    !   NOW FIND THE RESID USING NICTOT TO GET THE RESIDUES IN THE SEGMENT.
    !
    IRES=NICTOT(ISEG)+1
    ISTOP=NICTOT(ISEG+1)
    IF (IRES > ISTOP) GOTO 998
    DO WHILE (RESID(IRES) /= RES)
       IRES=IRES+1
       IF (IRES > ISTOP) GOTO 998
    ENDDO
    IF (ATOM /= '    ') THEN
       !
       !     FINALLY WE GET THE ATOM NUMBER.
       !
       IATOM=IBASE(IRES)+1
       ISTOP=IBASE(IRES+1)
       IF (IATOM > ISTOP) GOTO 997
       DO WHILE (ATYPE(IATOM) /= ATOM)
          IATOM=IATOM+1
          IF (IATOM > ISTOP) GOTO 997
       ENDDO
    ENDIF
    GETRSN=IRES
    RETURN
999 CONTINUE
    IF(WRNLEV >= 2) THEN
       IF(ATOM /= '    ') THEN
          WRITE(OUTU,2000) 'SEGMENT', &
               SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
       ELSE
          WRITE(OUTU,2001) 'SEGMENT', &
               SEG(1:idleng),RES(1:idleng)
       ENDIF
    ENDIF
    GETRSN=-1
    RETURN
998 CONTINUE
    IF(WRNLEV >= 2) THEN
       IF (ATOM /= '    ') THEN
          WRITE(OUTU,2000) 'RESIDUE', &
               SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
       ELSE
          WRITE(OUTU,2001) 'RESIDUE', &
               SEG(1:idleng),RES(1:idleng)
       ENDIF
    ENDIF
    GETRSN=-1
    RETURN
997 IF(WRNLEV >= 2) WRITE(OUTU,2000) 'ATOM', &
         SEG(1:idleng),RES(1:idleng),ATOM(1:idleng)
    GETRSN=-1
    RETURN
2000 FORMAT(' **  WARNING  ** GETRSN could not find the ',A, &
         ' SEG="',A,'" RES="',A,'" ATOM="',A,'"')
2001 FORMAT(' **  WARNING  ** GETRSN could not find the ',A, &
         ' SEG="',A,'" RES="',A,'"')
  END FUNCTION GETRSN

  LOGICAL FUNCTION HYDROG(I)
    !
    !     THIS FUNCTION RETURNS A TRUE IF THE ATOM IS DEEMED TO BE A
    !     HYDROGEN ATOM. ALL TESTS FOR HYDROGEN ATOMS SHOULD USE THIS
    !     ROUTINE.    BRB 10/27/82
    !
  use dimens_fcm
    !
  use psf
    INTEGER I
    LOGICAL H
    !
    IF(I <= 0 .OR. I > NATOMT) THEN
       HYDROG=.FALSE.
       RETURN
    ENDIF 
    !clb3: changed from 3.5 to 5 to account for hydrogen mass renormalization
    H= ( AMASS(I) < 5 ) &
         .and. ( (index(atype(i),"H")==1) .or. (index(atype(i),"D")==1) ) &
         .and. ( .NOT. LONE(I) ) 
    HYDROG=H
    RETURN
  END FUNCTION HYDROG

  LOGICAL FUNCTION LONE(I)
    !
    !     THIS FUNCTION RETURNS A TRUE IF THE ATOM IS DEEMED TO BE A
    !     LONE PAIR. ALL TESTS FOR HYDROGEN ATOMS SHOULD USE THIS
    !     ROUTINE.    BRB 11/29/82
    !
  use dimens_fcm
    !
  use psf
    INTEGER I
    !
    IF(I <= 0 .OR. I > NATOMT) THEN
       LONE=.FALSE.
       RETURN
    ENDIF
    LONE=AMASS(I) < 0.002_chm_real
    RETURN
  END FUNCTION LONE

  LOGICAL FUNCTION INITIA(ATOM,X,Y,Z)
    !
    !     Function true if coordinates for ATOM defined.
    !
  use number
    !
    INTEGER ATOM
    real(chm_real) X(*),Y(*),Z(*)
    !
    IF (ATOM > 0) THEN
       INITIA=(X(ATOM) < ANUM .AND. Y(ATOM).LT.ANUM .AND. &
            Z(ATOM) < ANUM)
    ELSE
       INITIA=.FALSE.
    ENDIF
    RETURN
  END FUNCTION INITIA


  ! Given the atom number ATOM this routine returns the SEGID, RESID,
  ! RESIDUE and IUPAC atom name or ??? if atom number is invalid.
  subroutine atomid(atom, sid, rid, ren, ac)
    use psf, only: atype, ibase, nrest, res, resid, segid, nictot, nsegt

    implicit none

    ! input
    integer, intent(in) :: atom

    ! output
    character(len = *), intent(out) :: sid, rid, ren, ac

    ! local vars
    integer :: atomx, iacrs
    character(len=5) :: atom_index_str, res_index_str

    sid = '???'
    rid = '???'
    ren = '???'
    ac = '???'

    if (atom <= 0 .or. atom > size(atype)) then
      write(atom_index_str, '(i5)') atom
      call wrndie(-2, 'atomid', 'invalid atom index ' // atom_index_str)
      return
    end if

    atomx = atom
    ac = atype(atomx)

    iacrs = getres(atomx, ibase, nrest)

    if (iacrs <= 0 .or. iacrs > size(res)) then
      write(atom_index_str, '(i5)') atomx
      write(res_index_str, '(i5)') iacrs
      call wrndie(-2, 'atomid', &
        'invalid residue index ' // res_index_str // &
        ' found for atom index ' // atom_index_str)
      return
    end if

    ren = res(iacrs)
    rid = resid(iacrs)
    sid = segid(getseg(iacrs, nictot, nsegt))
  end subroutine atomid

  SUBROUTINE GROUPID(GROUP,SID,RID,REN,AC)
    !-----------------------------------------------------------------------
    !     Given the group number, this routine returns the SEGID, RESID,
    !     RESIDUE and IUPAC atom name for the first atom of the group.
    !
    !     Remark: the information about the PSF is passed via the psf module
  use dimens_fcm
  use psf
    !
    INTEGER GROUP
    CHARACTER(LEN=*) SID, RID, REN, AC
    !
    INTEGER ATOM
    !
    ATOM=IGPBS(GROUP)+1
    CALL ATOMID(ATOM,SID,RID,REN,AC)
    RETURN
  END SUBROUTINE GROUPID


  INTEGER FUNCTION MATOM(IRES,ATOM,ATYPE,IBASE,ISTART,ISTOP,LW)
    !
    !     THIS ROUTINE RETURNS THE ATOM NUMBER GIVEN A RESIDUE AND
    !     ATOM TYPE -- AN * AS THE FIRST CHARACTER IS IGNORED -- BRB
    !
  use stream
  use string
    LOGICAL LW
    CHARACTER(LEN=*) ATYPE(*),ATOM
    CHARACTER(LEN=8)   WORD
    INTEGER IRES,ISTART,ISTOP
    INTEGER  IBASE(*)
    !
    INTEGER IDUM,ISTA,ISTO,I
    INTEGER :: MARK=-99999999
    !
    WORD=ATOM
    IF(IRES >= ISTART .AND. IRES <= ISTOP) THEN
       IF (EQSTA(WORD,1,'*')) THEN
          !yw            CALL COPSUB(WORD,4,IDUM,ATOM,2,4)
          !yw            CALL FILSPC(WORD,4,3)
          CALL COPSUB(WORD,8,IDUM,ATOM,2,LEN(ATOM))
          CALL FILSPC(WORD,8,IDUM)
       ENDIF
       ISTA=IBASE(IRES)+1
       ISTO=IBASE(IRES+1)
       DO I=ISTA,ISTO
          IF(WORD == ATYPE(I)) THEN
             MATOM=I
             RETURN
          ENDIF
       ENDDO
    ENDIF
    IF(LW .AND. WRNLEV >= 2) WRITE(OUTU,13) WORD(1:idleng),IRES
13  FORMAT(' *** IN MATOM *** ''',A,''' ATOM TYPE NOT FOUND FOR ', &
         'RESIDUE',I5)
    MATOM=MARK
    RETURN
  END FUNCTION MATOM

  LOGICAL FUNCTION QHAVCRD(NATOM,SELCT,X)
    !
    ! True if ALL selected atoms have defined coordinates
    ! IF SELCT(1)=-1 all atoms (1,..., NATOM) are checked,and SELCT is not used
    ! Empty selection returns false
    ! L. Nilsson June 2001. Updated 2015
    !
  use number
    !
    INTEGER NATOM,SELCT(NATOM)
    real(chm_real) X(NATOM)
    !
    QHAVCRD=.TRUE.
    IF(SELCT(1) == -1) THEN
      IF(ANY(X(1:NATOM) == ANUM)) QHAVCRD=.FALSE.
    ELSE
      IF(ALL(SELCT(1:NATOM) == 0)) THEN
        QHAVCRD=.FALSE.
      ELSE
        IF(ANY(X(1:NATOM)==ANUM .AND. SELCT(1:NATOM)==1)) QHAVCRD=.FALSE.
      ENDIF
    ENDIF
    RETURN
    END FUNCTION QHAVCRD

  subroutine value_error(caller, msg, val, iunit, line)
    character(len=*), intent(in) :: caller, msg
    integer, intent(in) :: val, iunit
    integer, intent(in), optional :: line
    integer :: errcode
    character(len=100) :: fname
    character(len=200) :: fullmsg

    inquire (unit=iunit, name=fname, iostat=errcode)
    if (errcode == 0) then
       if (present(line)) then
          write (fullmsg, '(A,X,I0,A,I0,2A)') &
                msg, val, ' on line ', line, ' of ', fname
       else
          write (fullmsg, '(A,X,I0,2A)') &
                msg, val, ' in ', fname
       endif
    else
       write (fullmsg, '(A,I0,/,A,I0)') &
             msg, val, 'value_error: cannot get name of unit ', iunit
    endif
    call wrndie(-2, caller, trim(fullmsg))
  end subroutine value_error


  SUBROUTINE ATOMID2(ATOM,SID,RID,REN,AC, &
       SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEGT,NREST)
    !-----------------------------------------------------------------------
    !     Given the atomnumber ATOM this routine returns the SEGID, RESID,
    !     RESIDUE and IUPAC atom name.
    !
    !     Remark: this version use the specified arrays instead of the
    !             psf module
    !
    !     adpated from ATOMID by Nathan Desdouits, 4/2012
    !
  use dimens_fcm
    !
    !
    INTEGER ATOM,NREST,NSEGT
    CHARACTER(LEN=*) SID, RID, REN, AC
    CHARACTER(LEN=8) SEGID(*),RES(*),RESID(*),ATYPE(*)
    INTEGER IBASE(*),NICTOT(*)
    INTEGER ATOMX
    INTEGER IACRS
    !
    ATOMX=ATOM
    IF(ATOM <= 0) THEN
       SID='???'
       RID='???'
       REN='???'
       AC='???'
       RETURN
    ENDIF
    !
    IACRS=GETRES(ATOMX,IBASE,NREST)
    REN=RES(IACRS)
    RID=RESID(IACRS)
    SID=SEGID(GETSEG(IACRS,NICTOT,NSEGT))
    AC=ATYPE(ATOMX)
    RETURN
  END SUBROUTINE ATOMID2

  SUBROUTINE GROUPID2(GROUP,SID,RID,REN,AC, &
       SEGID,RES,RESID,ATYPE,NICTOT,IBASE,IGPBS,NSEGT,NREST)
    !-----------------------------------------------------------------------
    !     Given the group number, this routine returns the SEGID, RESID,
    !     RESIDUE and IUPAC atom name for the first atom of the group.
    !
    !     Remark: this version use the specified arrays instead of the
    !             psf module
    !
    !     adapted from GROUPID by Nathan Desdouits, 4/2012
    !
  use dimens_fcm
    !
    INTEGER GROUP,NREST,NSEGT
    CHARACTER(LEN=*) SID, RID, REN, AC
    CHARACTER(LEN=8) SEGID(*),RES(*),RESID(*),ATYPE(*)
    INTEGER IBASE(*),NICTOT(*),IGPBS(*)
    !
    INTEGER ATOM
    !
    ATOM=IGPBS(GROUP)+1
    CALL ATOMID2(ATOM,SID,RID,REN,AC, &
       SEGID,RES,RESID,ATYPE,NICTOT,IBASE,NSEGT,NREST)
    RETURN
  END SUBROUTINE GROUPID2

end module chutil
