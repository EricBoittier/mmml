! CHARMM Element source/energy/multe.src $Revision: 1.13 $
SUBROUTINE multe0(comlyn,comlen)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use energym
  use machio,only:vopen
  use memory
  use stream
  use string
  use timerm

  implicit none

  character(len=*) comlyn
  integer comlen
  integer atomnumi, atomnumj
  integer length1, length2
  integer numatom1, numatom2
  integer numrot1, numrot2
  integer totcoor1, totcoor2
  integer coorunit, outunit
  integer i,k
  character(len=4) eword
  integer length
  integer,parameter :: maxeterm=10
  integer eind(maxeterm)
  integer eindlen
  integer,parameter :: maxlen=128
  character(len=maxlen) file1, file2, outfile
  logical QERROR, FOUND
  logical qprfile, qprtote
  !
  !  coorp1 and coorp2 keep track of atomic coordinates
  !
  real(chm_real),allocatable,dimension(:),save :: coorp1, coorp2

  qprfile = .false.
  qprtote = .false.
  coorunit = 21
  outunit = 22
  atomnumi = 0
  atomnumj = 0
  atomnumi = GTRMI(COMLYN,COMLEN,'ATNI',atomnumi)
  atomnumj = GTRMI(COMLYN,COMLEN,'ATNJ',atomnumj)
  if (prnlev >= 2) write(outu,'(a,i10,a,i10)') "atomnumi ", atomnumi, &
       "  atomnumj ", atomnumj
  qprtote = INDXA(COMLYN,COMLEN,'TOTE') .gt. 0
  if (atomnumi .le. 0) then
     call wrndie(-5,'<MULTE>','NO INITIAL ATOM1 NUM GIVEN')
  endif
  ! version 6 of multe--allow atomnumj to be zero in case one wants
  !    to vary the coordinates of only one group in the structure.
  !      if (atomnumj .le. 0) then
  !         call wrndie(-5,'<MULTE>','NO INITIAL ATOM2 NUM GIVEN')
  !      endif
  CALL GTRMWD(COMLYN,COMLEN,'FILI',4,FILE1,MAXLEN,length1)
  IF (length1.EQ.0) THEN
     CALL WRNDIE(0,'<MULTE>','NO FILE NAME1 GIVEN')
     RETURN
  ENDIF
  ! remove any embedded double quotes from @param substitution
  K = length1
  DO I=1,length1
     IF (FILE1(I:I).EQ.'"') THEN
        FILE1(I:) = FILE1(I+1:)
        K = K-1
     ENDIF
  ENDDO
  IF (K.NE.length1) length1 = K

  CALL GTRMWD(COMLYN,COMLEN,'FILJ',4,FILE2,MAXLEN,length2)
  IF ((length2.EQ.0) .and. (atomnumj.gt.0)) THEN
     CALL WRNDIE(0,'<MULTE>','NO FILE NAME2 GIVEN')
     RETURN
  ENDIF
  ! remove any embedded double quotes from @param substitution
  K = length2
  DO I=1,length2
     IF (FILE2(I:I).EQ.'"') THEN
        FILE2(I:) = FILE2(I+1:)
        K = K-1
     ENDIF
  ENDDO
  IF (K.NE.length2) length2 = K

  CALL GTRMWD(COMLYN,COMLEN,'OUTF',4,OUTFILE,MAXLEN,length)
  IF (length.EQ.0) THEN
     outfile = 'energy.tab'
  ENDIF
  ! remove any embedded double quotes from @param substitution
  K = length
  DO I=1,length
     IF (OUTFILE(I:I).EQ.'"') THEN
        OUTFILE(I:) = OUTFILE(I+1:)
        K = K-1
     ENDIF
  ENDDO
  IF (K.NE.length) length = K

  length = 1
  eindlen = 0
  call gtrmwd(comlyn,comlen,'ETER',4,EWORD,4,length)
  do while (length .gt. 0)
     FOUND = .FALSE.
     DO I = 1,LENENT
        IF (CETERM(I) .EQ. EWORD) THEN
           eindlen = eindlen + 1
           if (eindlen .gt. maxeterm) then
              call wrndie(-1,'<MULTE>',"TOO MANY ENERGY TERMS")
           endif
           eind(eindlen) = i
           FOUND    = .TRUE.
           exit
        ENDIF
     ENDDO
     IF (.NOT.(FOUND)) THEN
        IF(WRNLEV.GE.2) WRITE (OUTU,'(A,A4,A)') &
             ' MULTE>  **** ERROR **** Unrecognised energy term = ', &
             EWORD, '.'
        CALL DIEWRN(0)
     ENDIF
     call gtrmwd(comlyn,comlen,'ETER',4,EWORD,4,length)
  enddo
  if (eindlen .gt. 0) qprfile = .true.
  if (qprtote) qprfile = .true.

  call mlte_open(coorunit,file1,length1,numatom1,numrot1,totcoor1)
  call chmalloc('multe.src','multe0','coorp1',totcoor1,crl=coorp1)
  call mlteread(coorunit,coorp1,totcoor1)

  ! second file
  if (atomnumj .gt. 0) then
     call mlte_open(coorunit,file2,length2,numatom2,numrot2,totcoor2)
     call chmalloc('multe.src','multe0','coorp2',totcoor2,crl=coorp2)
     call mlteread(coorunit,coorp2,totcoor2)
     CALL VCLOSE(coorunit,'KEEP',QERROR)
  else
     numatom2 = 0
     numrot2 = 1
  endif

  ! -- output file --
  if (qprfile) then
     CALL VOPEN(OUTUNIT,outfile,'FORMATTED','WRITE',QERROR,0)
     IF(QERROR) THEN
        WRITE(OUTU,'(3a,i4,a)') "OPEN ",outfile," AS UNIT ", &
             OUTUNIT," FAILED."
        CALL WRNDIE(-1,'<MULTE>',"CAN NOT OPEN OUTFILE")
     ENDIF
  endif
  !-----------------------------
  call multe(comlyn,comlen,coorp1,coorp2,atomnumi, &
       atomnumj,numatom1,numatom2,numrot1,numrot2, &
       qprfile,qprtote,eind,eindlen,outunit)
  if (qprfile) then
     CALL VCLOSE(outunit,'KEEP',QERROR)
  endif
  return
end SUBROUTINE multe0

subroutine multe(comlyn,comlen,coor1,coor2,atomnumi, &
     atomnumj,numatom1,numatom2,numrot1,numrot2, &
     qprfile,qprtote,eind,eindlen,outunit)
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use energym
  use stream
  use coord
  use eutil
  use parallel,only:mynod
  implicit none
  character(len=*) comlyn
  integer comlen
  real(chm_real) coor1(*), coor2(*)
  integer atomnumi, atomnumj
  integer numatom1, numatom2, numrot1, numrot2
  logical qprfile, qprtote
  integer eind(*)
  integer eindlen, outunit
  integer i,j,r,s,k
  integer oldprnlev

  integer icoor,jcoor
  if (qprfile) then
     oldprnlev = PRNLEV
     PRNLEV = 1
  endif
  ! now i,j refer to the real coordinates
  icoor = 1
  do r=1,numrot1
     do i=atomnumi,atomnumi+numatom1-1
        x(i) = coor1(icoor)
        y(i) = coor1(icoor+1)
        z(i) = coor1(icoor+2)
        icoor = icoor + 3
        jcoor = 1
     enddo
     do s=1,numrot2
        !  the j loop will be skipped if numatom2 is zero, so we will vary the
        !       coordinates of only one group
        do j=atomnumj,atomnumj+numatom2-1
           x(j) = coor2(jcoor)
           y(j) = coor2(jcoor+1)
           z(j) = coor2(jcoor+2)
           jcoor = jcoor + 3
        enddo

        call gete(X,Y,Z,X,Y,Z,0)

        if(mynod == 0)then
           if (qprfile) then
              if (qprtote) then
                 write(outunit,100) EPROP(EPOT), &
                      (ETERM(eind(k)),k=1,eindlen)
              else
                 write(outunit,100) (ETERM(eind(k)),k=1,eindlen)
              endif
100           format(10(1x,f13.5))
           endif
        endif
     enddo
  enddo
  if (qprfile) then
     PRNLEV = oldprnlev
  endif
  return
end subroutine multe

subroutine mlte_open(u,file,flen,numatom,numrot,totcoor)
  use stream,only:outu,prnlev
  use machio,only:vopen
  use parallel,only:mynod,numnod,psnd4,psnd8
  implicit none
  integer,intent(in) :: u,flen
  character(len=flen),intent(in) :: file
  integer,intent(out) :: numatom,numrot,totcoor
  logical qerror
  
  if(mynod == 0)then
     call vopen(u,file,'FORMATTED','READ',qerror,0)
     if(qerror) then
        write(outu,'(3a,i4,a)') "OPEN ",file," AS UNIT ",u, &
             " FAILED."
        call wrndie(-1,'<MULTE>',"CAN NOT OPEN COORFILE1")
     endif
     read(u,*) numatom,numrot
     totcoor = 3*numatom*numrot
     if (prnlev >= 2) then
        write(outu,'(a,a,i10,a,i10)') file, " NUMATOM ", numatom, &
             "  NUMROT1 ", numrot
        write(outu,'(2a,i10)') file, " TOTCOOR ", totcoor
        write(outu,'(2a,i10)') file, " coor length ", totcoor
     endif
  endif
#if KEY_PARALLEL==1
  if(numnod > 1) then
     call psnd4(numatom,1)
     call psnd4(numrot,1)
     call psnd4(totcoor,1)
  endif
#endif /* */
end subroutine mlte_open

subroutine mlteread(coorunit,coor,totcoor)
  !-----------------------------------------------------------------------
  use chm_kinds
  use parallel,only:mynod,psnd8
  implicit none
  integer coorunit
  integer totcoor
  real(chm_real) coor(totcoor)
  integer i,j,numrd
  logical qerror

  if(mynod == 0 ) then
     if (mod(totcoor,3) .ne. 0) CALL WRNDIE(-4,'<MULTE>', &
          'number of coordinates not mod 3')
     numrd = totcoor/3
     do i=1,numrd
        read(coorunit,'(3f12.4)') (coor(j),j=3*i-2,3*i)
     enddo
     CALL VCLOSE(coorunit,'KEEP',QERROR)
  endif
#if KEY_PARALLEL==1
  call psnd8(coor,totcoor)     
#endif

  return
end subroutine mlteread

