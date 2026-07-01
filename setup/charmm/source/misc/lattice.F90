Subroutine RLATT(unit,islct)
  !-----------------------------------------------------------------------
  !  This is the main routine to read lattice structures from
  !  Skolnick dynamic MC programs and convert them to CHARMM Ca
  !  or Cb atomic positions
  !  Syntax:
  !          read coor lattice unit <integer> scale <real> -
  !           <side> select (...) end
  !
  !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use number
  use exfunc
  use comand
  use psf
  use stream
  use string
  use coord

  implicit none
  INTEGER ISLCT(*), unit, nread, natm, i
  LOGICAL LSIDE
  real(chm_real) Scale
  !
  Scale = 1.22
  Scale = GTRMF(COMLYN,COMLEN,'SCAL',Scale)
  LSide = (INDXA(COMLYN,COMLEN,'SIDE').GT.0)

  nread = 0
  Do i=1, natom
     if(islct(i).ne.0) nread = nread + 1
  Enddo

  Write(6,'(5x,i5,2x,a,2x,i5)')  &
       nread,'ATOMS SELECTED FOR READING FROM UNIT', unit
  Write(6,'(5x,a,2x,f10.4,2x,a)') &
       'A SCALE FACTOR OF', Scale,'WILL BE APPLIED TO COORDINATES'

  if(.not.lside) then
     read(unit,*) natm
     if( (natm - nread).ne.0 ) then
        Write(6,'(5x,a)')'FIRST AND LAST ATOM POSITIONS IGNORED'
        Read(unit,*) natm, natm, natm
     endif
  endif

  Do i=1,natom

     if(islct(i).ne.0) then
        read(unit,*) x(i), y(i), z(i)
        x(i) = Scale * x(i)
        y(i) = Scale * y(i)
        z(i) = Scale * z(i)
     endif
  Enddo
  RETURN
END Subroutine RLATT
