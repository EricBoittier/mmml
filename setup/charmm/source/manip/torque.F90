module torque
  use chm_kinds
  use chm_types
  implicit none

#if KEY_TORQUE==1 /*tourqe_main*/

  logical qtorque                    !  Torque centers setup?
  integer ntrqcenter,maxtrqcenter    !  number of torque centers
  integer,allocatable,dimension(:) :: trqati,trqatf,trqtyp ! first and last atom of and type of each body
  real(chm_real),allocatable,dimension(:,:,:) :: trqrotm      ! determined rotation matrices
  real(chm_real),allocatable,dimension(:,:) :: trqtxyz        ! computed torques

contains

subroutine torque_iniall()
  qtorque=.false.
  return
end subroutine torque_iniall


SUBROUTINE TORQUE_PARSE
  !-----------------------------------------------------------------------
  !     THIS ROUTINE PARSES THE TORQUE COMMAND
  !
  !     BRB Oct 27, 2010
  !
  !
  !  Syntax:
  !     TORQue { SET   } { WATEr } [ ALL       ]  [atom-selection]
  !            { ADD   } { BODY  } [ BYSEgment ]
  !            { CLEAr }           [ BYREsidue ]
  !                                [ BYGRoup   ]
  !
  !  Example of usage: 
  !   Create a set of torque bodies for all waters (segment named "SOLV"):
  !          TORQUE SET WATER BYRES sele segid SOLV end
  ! 


  use chm_kinds
  use dimens_fcm
  use psf
  use coord

  use comand
  use stream
  use string
  use select

  use memory
  !
  implicit none
  integer,allocatable,dimension(:) :: islct,iwork

  !
  CHARACTER(len=4) WRD

  INTEGER   I,J,ISEG,IRES,IST,NSL,INUMBR,ICODE,IS,IQ,LLEN
  INTEGER   BTYPE,ERR


  WRD=NEXTA4(COMLYN,COMLEN)

  call chmalloc('torque.src','torque','FLAGS',NATOM,intg=islct)
  CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)

  IF (WRD == 'ADD') THEN
    if(.not.qtorque) WRD='SET'
  ENDIF

  IF (WRD == 'CLEA') THEN
    ! clear the data structure
    !
    if(qtorque) then
      call TORQUE_CLEAR
      return
    endif
  ELSE IF (WRD == 'SET') THEN
       ! setup data for new bodies

    qtorque = .true.
    maxtrqcenter = NATOM/3 + 4
      call chmalloc('torque.src','torque_clear','trqati',maxtrqcenter,intg=trqati)
      call chmalloc('torque.src','torque_clear','trqatf',maxtrqcenter,intg=trqatf)
      call chmalloc('torque.src','torque_clear','trqtyp',maxtrqcenter,intg=trqtyp)
      allocate(trqrotm(3,3,maxtrqcenter),trqtxyz(3,maxtrqcenter),stat=err)
      ntrqcenter=0

  ELSE IF (WRD == 'ADD') THEN
        ! Remove all existing torque centers from the selection (so don't double count)
      DO I=1,ntrqcenter
        DO J=trqati(I),trqatf(I)
          islct(J)=0
        ENDDO
      ENDDO
  ELSE
     CALL WRNDIE(1,'<TORQUE_PARSE>','Unrecognized TORQUE subcommand')
     return
  ENDIF

  !   Get body type
     WRD=NEXTA4(COMLYN,COMLEN)
     IF(WRD /= 'WATE') THEN
        CALL WRNDIE(-1,'<TORQUE_PARSE>','WATER is the only body type currently implemented')
        return
     ELSE
        BTYPE=1
     ENDIF

     ICODE=0
     IF(INDXA(COMLYN,COMLEN,'BYGR') > 0) ICODE=4
     IF(INDXA(COMLYN,COMLEN,'BYRE') > 0) ICODE=3
     IF(INDXA(COMLYN,COMLEN,'BYSE') > 0) ICODE=2
     IF(INDXA(COMLYN,COMLEN,'ALL') > 0) ICODE=1

     !
     IF (ICODE == 1) THEN
        !         DO FOR ALL ATOMS
        IS=1
        IQ=NATOM
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              call TORQUE_ADDBODY(I,IQ,BTYPE,islct)
           ENDIF
        ENDDO
        !----
     ELSE IF (ICODE == 2) THEN
        !     DO FOR EACH SEGMENT
        ISEG=0
        IF (ISEG >= NSEG) GOTO 420
400     CONTINUE
        ISEG=ISEG+1
        IS=IBASE(NICTOT(ISEG)+1)+1
        IQ=IBASE(NICTOT(ISEG+1)+1)
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              call TORQUE_ADDBODY(I,IQ,BTYPE,islct)
           ENDIF
        ENDDO

        !----
        IF (.NOT.(ISEG >= NSEG)) GOTO 400
420     CONTINUE
     ELSE IF (ICODE == 3) THEN
        !         DO FOR EACH RESIDUE
        IRES=0
        IF (IRES >= NRES) GOTO 450
430     CONTINUE
        IRES=IRES+1
        IS=IBASE(IRES)+1
        IQ=IBASE(IRES+1)
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              call TORQUE_ADDBODY(I,IQ,BTYPE,islct)
           ENDIF
        ENDDO

        !----
        IF (.NOT.(IRES >= NRES)) GOTO 430
450     CONTINUE
     ELSE IF (ICODE == 4) THEN
        !     DO FOR EACH GROUP
        IRES=0
        IF (IRES >= NGRP) GOTO 480
470     CONTINUE
        IRES=IRES+1
        IS=IGPBS(IRES)+1
        IQ=IGPBS(IRES+1)
        DO I=IS,IQ
           IF (ISLCT(I) == 1) THEN
              call TORQUE_ADDBODY(I,IQ,BTYPE,islct)
           ENDIF
        ENDDO

        !----
        IF (.NOT.(IRES >= NGRP)) GOTO 470
480     CONTINUE
     ELSE
        CALL WRNDIE(0,'<TORQUE_PARSE>','Division field not specified')
     ENDIF
 
  call chmdealloc('torque.src','torque','FLAGS',NATOM,intg=islct)

  RETURN
END SUBROUTINE TORQUE_PARSE

SUBROUTINE TORQUE_ADDBODY(IS,IQ,BTYPE,islct)

  use chm_kinds
  use number
  use dimens_fcm
  use psf
  use coord
  implicit none

  integer is,iq,btype,islct(*)
  integer ix,iy

  iy=iq
  DO ix=iq,is,-1
    if(islct(ix) /= 1) iy=ix-1
  ENDDO
  DO ix=is,iy
    islct(ix)=0
  ENDDO

  IF(btype == 1) then
    if(IY-IS == 7) then
      call wrndie(3,'<TORQUE_ADDBODY>','Assuming you are using new spiffy proto SSDQO water')
    else if(IY-IS /= 2) then
      CALL WRNDIE(-3,'<TORQUE_ADDBODY>','Wrong number of atoms selected for WATER body type')
      return
    endif

    ntrqcenter=ntrqcenter+1    !  number of torque centers
    trqati(ntrqcenter)=IS
    trqatf(ntrqcenter)=IY
    trqtyp(ntrqcenter)=btype

  ELSE
    CALL WRNDIE(-3,'<TORQUE_ADDBODY>','Unimplemented torque body type')
  ENDIF

  return
END SUBROUTINE TORQUE_ADDBODY

SUBROUTINE TORQUE_GETROT(islct)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE COMPUTES THE ROTATION MATRIX FOR EACH BODY
  !
  !     BRB Oct 27, 2010
  !

  use chm_kinds
  use number
  use dimens_fcm
  use psf
  use coord
  use memory
  use stream, only:prnlev
  use vector
  implicit none

  integer islct(*)
  !
  integer icenter,iw,i  

  I=0
  DO icenter=1,ntrqcenter
    if(trqtyp(icenter)==1) then
 !    This is a rigid water type. Create rot matrix directly from coordinates
      if(islct(trqati(icenter))==1) then
        I=I+1
        iw=trqati(icenter)
           ! 3rd axis is dipole axis
        trqrotm(1,3,I) = X(iw+2)+X(iw+1)-TWO*X(iw)
        trqrotm(2,3,I) = Y(iw+2)+Y(iw+1)-TWO*Y(iw)
        trqrotm(3,3,I) = Z(iw+2)+Z(iw+1)-TWO*Z(iw)
        IF(PRNLEV.GE.7) THEN
          WRITE(*,'(A,3F10.6)') 'TORQUE_GETROT DEBUG> PRE-NORMALL VEC3 = ', &
                trqrotm(1,3,I), trqrotm(2,3,I), trqrotm(3,3,I)
        ENDIF
        CALL NORMALL(trqrotm(1,3,I),3)
        IF(PRNLEV.GE.7) THEN
          WRITE(*,'(A,3F10.6)') 'TORQUE_GETROT DEBUG> POST-NORMALL VEC3 = ', &
                trqrotm(1,3,I), trqrotm(2,3,I), trqrotm(3,3,I)
        ENDIF
           ! 2nd axis is H-H vector
        trqrotm(1,2,I) = X(iw+1)-X(iw+2)
        trqrotm(2,2,I) = Y(iw+1)-Y(iw+2)
        trqrotm(3,2,I) = Z(iw+1)-Z(iw+2)
        IF(PRNLEV.GE.7) THEN
          WRITE(*,'(A,3F10.6)') 'TORQUE_GETROT DEBUG> PRE-ORTHOG/NORMALL VEC2 = ', &
                trqrotm(1,2,I), trqrotm(2,2,I), trqrotm(3,2,I)
        ENDIF
        CALL ORTHOG(trqrotm(1,2,I),trqrotm(1,3,I),3)
        CALL NORMALL(trqrotm(1,2,I),3)
        IF(PRNLEV.GE.7) THEN
          WRITE(*,'(A,3F10.6)') 'TORQUE_GETROT DEBUG> POST-ORTHOG/NORMALL VEC2 = ', &
                trqrotm(1,2,I), trqrotm(2,2,I), trqrotm(3,2,I)
        ENDIF
        CALL CROSS3(trqrotm(1,2,I),trqrotm(1,3,I),trqrotm(1,1,I))
        IF(PRNLEV.GE.7) THEN
          WRITE(*,'(A,3F10.6)') 'TORQUE_GETROT DEBUG> POST CROSS3 VEC1 = ', &
                trqrotm(1,1,I), trqrotm(2,1,I), trqrotm(3,1,I)
        ENDIF
      endif
    else
      CALL WRNDIE(-3,'<TORQUE_GETROT>','Unimplemented torque body type')
      return
    endif

  ENDDO

  RETURN

END SUBROUTINE TORQUE_GETROT

SUBROUTINE TORQUE_MAKEFORCE(islct)
  !-----------------------------------------------------------------------
  !     THIS ROUTINE COMPUTES FORCES CONSISTENT WITH APPLIED TORQUES
  !     torques are assumed to be in the molecular frame.
  !
  !     BRB Oct 27, 2010
  !

  use chm_kinds
  use chm_types
  use number
  use dimens_fcm
  use psf
  use coord
  use deriv
  implicit none

  !
  integer islct(*)
  !
  integer icenter,iw,i  
  real(chm_real) df(3,3),rf(3,3),rhh,rob,twxa,twy,twz,twxb

  I=0
  DO icenter=1,ntrqcenter
    if(trqtyp(icenter).eq.1) then
 !    This is a rigid water type. Make forces directly from torques
      if(islct(trqati(icenter)).eq.1) then
        I=I+1
        iw=trqati(icenter)

        rhh=SQRT((X(iw+2)-X(iw+1))**2 + (Y(iw+2)-Y(iw+1))**2 + (Z(iw+2)-Z(iw+1))**2)
        rob=SQRT((X(iw)-HALF*(X(iw+2)+X(iw+1)))**2 + &
                 (Y(iw)-HALF*(Y(iw+2)+Y(iw+1)))**2 + & 
                 (Z(iw)-HALF*(Z(iw+2)+Z(iw+1)))**2)
        
  !    Compute forces in molecule frame due to torques (non-unique solution)

        twxa = trqtxyz(1,I)/rhh * half
        twxb = trqtxyz(1,I)/rob * half
        twy = -trqtxyz(2,I)/rob
        twz = -trqtxyz(3,I)/rhh

        df(1,1) =  twy
        df(1,2) =  twz -twy*half
        df(1,3) = -twz -twy*half

        df(2,1) = twxb
        df(2,2) = -twxb*half
        df(2,3) = -twxb*half

        df(3,1) =  0.0
        df(3,2) =  twxa
        df(3,3) = -twxa

  !       Rotate new forces to lab frame
        call mulnxn(rf,trqrotm(1,1,I),df,3)
  !       Add new forces to the force arrays
        DX(iw)=DX(iw)+rf(1,1)
        DY(iw)=DY(iw)+rf(2,1)
        DZ(iw)=DZ(iw)+rf(3,1)
        DX(iw+1)=DX(iw+1)+rf(1,2)
        DY(iw+1)=DY(iw+1)+rf(2,2)
        DZ(iw+1)=DZ(iw+1)+rf(3,2)
        DX(iw+2)=DX(iw+2)+rf(1,3)
        DY(iw+2)=DY(iw+2)+rf(2,3)
        DZ(iw+2)=DZ(iw+2)+rf(3,3)

     endif
    else
      CALL WRNDIE(-3,'<TORQUE_GETROT>','Unimplemented torque body type')
      return
    endif
  ENDDO

  RETURN

END SUBROUTINE TORQUE_MAKEFORCE

SUBROUTINE TORQUE_CLEAR

  use chm_kinds
  use chm_types
  use stream
  use memory
  !
  implicit none
  integer err

    if(qtorque) then
      call chmdealloc('torque.src','torque_clear','trqati',maxtrqcenter,intg=trqati)
      call chmdealloc('torque.src','torque_clear','trqatf',maxtrqcenter,intg=trqatf)
      call chmdealloc('torque.src','torque_clear','trqtyp',maxtrqcenter,intg=trqtyp)
      deallocate(trqrotm,trqtxyz,stat=err)
      ntrqcenter=0
      maxtrqcenter=0

      write(outu,'(A)') ' TORQUE_CLEAR> TORQUE body data cleared.'
      qtorque=.false.
    endif

#else /* (tourqe_main)*/
contains
SUBROUTINE TORQUE_CLEAR
  use stream
  implicit none
    write(outu,'(A)') ' TORQUE_CLEAR> TORQUE code not compiled.'
#endif /* (tourqe_main)*/
    return
END SUBROUTINE TORQUE_CLEAR

end module torque

