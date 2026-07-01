module cveloci_mod
  use chm_kinds
  use dimens_fcm
  implicit none
  !CHARMM Element source/fcm/cveloci.fcm $Revision: 1.2 $
  !
#if KEY_CVELOCI==1 /*cveloci_fcm*/
  !     impulse
  !     Purpose:
  !     atoms that assigned a constant velocity
  !
  !     Variable        Purpose
  !     LCVEL           logical: is impulse on or not
  !     NCVEL           total number in FCVEL
  !     FCVEL(NCVEL)    array with impulsed atom ids
  !     VCVEL(3,NCVEL)  array with impule velocity (Ang/AKMA time)
  !
  INTEGER,PARAMETER :: MAXCVEL=10000
  !
  LOGICAL LCVEL
  real(chm_real)  VCVEL(3,maxcvel)
  INTEGER NCVEL,FCVEL(MAXCVEL)                    

#endif /* (cveloci_fcm)*/
  !

  !
  !   NIMP1 - the rate of cveloci (velocity) Angst/ps ????
  !   VIMP  - the cos direc vector between the two atoms selected(cveloci scalar 
  !
contains
  SUBROUTINE CVELOCI(COMLYN,COMLEN)
    !-----------------------------------------------------------------------
    !     This routine interprets commands dealing with 
    !-----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  use exfunc
  use number
  use coord
  use coordc
    !
  use select
  use stream
  use string
  use psf
    implicit none

    !.passed variables to cveloci.src
    character(len=*) COMLYN
    INTEGER COMLEN

    !.local variables
    real(chm_real) X2,Y2,Z2,X1,Y1,Z1
    real(chm_real) NIMP1, VIMP(3),RIMP
    INTEGER     I,J,K, ID1
    character(len=4) WRD

    !.used by selprn
    INTEGER ISLCT(natom),JSLCT(natom)
    INTEGER IMODE, NSLCT, NSLCT2
    LOGICAL LSEL2 

#if KEY_CVELOCI==1 /*cvel_main*/

    NIMP1=ZERO
    VIMP(1)=ZERO
    VIMP(2)=ZERO
    VIMP(3)=ZERO
    RIMP=ZERO

    CALL TRIMA(COMLYN,COMLEN)
    IF(COMLEN <= 0) THEN
       LCVEL=.FALSE.
       RETURN
    ENDIF
    NIMP1=NEXTF(COMLYN,COMLEN)
    IF(NIMP1 == 0.0) THEN
       LCVEL=.FALSE.
       RETURN
    ENDIF
    LCVEL=.TRUE.
    !
    !=======================================================================
    ! to select first-atoms.
    IMODE=0
    CALL SELRPN(COMLYN,COMLEN,ISLCT,NATOM,1,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE /= 0) THEN
       CALL WRNDIE(0,'<CVELOCI>','ATOM SELECTION ERROR')
       RETURN
    ENDIF
    !
    NSLCT=0
    DO K=1,NATOM
       IF(ISLCT(K) == 1) THEN
          NSLCT=NSLCT+1
          X1=X(K)
          Y1=Y(K)
          Z1=Z(K)
          ID1=K
          !            print*,k,x(k),y(k),z(k) 
       ENDIF
    ENDDO
    IF(NSLCT > MAXCVEL) THEN
       CALL WRNDIE(-1,'<CVELOCI>','TOO MANY ATOMS SELECTED')
       RETURN
    ENDIF
    IF(NSLCT == 0) THEN
       CALL WRNDIE(0,'<CVELOCI>','ZERO ATOMS SELECTED')
       LCVEL=.FALSE.
       RETURN
    ENDIF
    !
    !=======================================================================
    ! to select-second-atoms
    IMODE=0
    CALL SELRPN(COMLYN,COMLEN,JSLCT,NATOM,0,IMODE, &
         .FALSE.,1,' ',0,RESID,RES,IBASE,SEGID,NICTOT,NSEG, &
         .TRUE.,X,Y,Z,.TRUE.,1,WMAIN)
    IF(IMODE /= 0) THEN
       CALL WRNDIE(0,'<CVELOCI>','ATOM SELECTION ERROR FOR SECOND')
       RETURN
    ENDIF
    !
    NSLCT2=0
    DO K=1,NATOM
       IF(JSLCT(K) == 1) THEN
          NSLCT2=NSLCT2+1
          X2=X(K)
          Y2=Y(K)
          Z2=Z(K)
          !            print*,k,x(k),y(k),z(k) 
       ENDIF
    ENDDO

    LSEL2=(NSLCT2 > 0)
    !
    IF(LSEL2) THEN
       ! Use old syntax (two atoms...)
       !
       !.get id of atom to be cveloci: use array for now
       j=1
       DO I=1,NATOM
          IF(JSLCT(I) > 0) THEN
             FCVEL(j)=I
             j=j+1
             !             print*,FCVEL(I)
          ENDIF
       ENDDO
       NCVEL = NSLCT2

       write(outu,*)'tot cvel atoms found',ncvel

       WRITE(OUTU,986) NCVEL
986    FORMAT(/,' CVELOCI> ',I6,' Atoms selected for cveloci')
       WRITE(OUTU,988) NIMP1
988    FORMAT(/,' CVELOCI> Velocity (Angst./ps) assigned is ',F10.5)

       !-------------------------
       !.figure out cveloci vector info

       RIMP=SQRT((X2-X1)**2 + (Y2-Y1)**2 +(Z2-Z1)**2) 

       WRITE(OUTU,1110) RIMP 
1110   FORMAT(/,' CVELOCI>  Impulse scalar is: ',F10.5)

       VIMP(1)=(X2-X1)/RIMP
       VIMP(2)=(Y2-Y1)/RIMP
       VIMP(3)=(Z2-Z1)/RIMP

       WRITE(OUTU,1100) VIMP(1),VIMP(2),VIMP(3)
1100   FORMAT(/,' CVELOCI> Direction Cos:  AX = ',F10.5,/, &
            ' CVELOCI>                 AY = ',F10.5,/, &
            ' CVELOCI>                 AZ = ',F10.5,/)

       !-------------------------
       !.now move dummy atom to coor of begin atom
       !.NOTE:::adjust x coor just a little (0.0001) because somehow
       ! duplicate coord causes charmm dyn to blow up 

       DO I=1,NCVEL 
          j=FCVEL(i)
          X(j) = X(ID1)+(0.0001 * VIMP(1))
          Y(j) = Y(ID1)+(0.0001 * VIMP(2))
          Z(j) = Z(ID1)+(0.0001 * VIMP(3))

          WRITE(OUTU,1101) j,X(j), Y(j), Z(j)
1101      FORMAT(/,' CVELOCI>  Coor of atomid ',i5,' reset to ',3F10.5)
          !             ENDIF
       ENDDO
       !
       DO I=1,NCVEL
          VCVEL(1,I)=VIMP(1)*NIMP1
          VCVEL(2,I)=VIMP(2)*NIMP1
          VCVEL(3,I)=VIMP(3)*NIMP1
       ENDDO
    ELSE
       ! Use the new syntax (one atom selection..main and comp coordinates)
       j=0
       DO I=1,NATOM
          IF(ISLCT(I) > 0) THEN
             j=j+1
             FCVEL(j)=I
             VCVEL(1,J)=(XCOMP(I)-X(I))/NIMP1
             VCVEL(2,J)=(YCOMP(I)-Y(I))/NIMP1
             VCVEL(3,J)=(ZCOMP(I)-Z(I))/NIMP1
          ENDIF
       ENDDO
       NCVEL = NSLCT
    ENDIF
    !
#else /*  (cvel_main)*/
    CALL WRNDIE(0,'<CVELOCI>','CVELOCI code not compiled')
#endif /* (cvel_main)*/
    RETURN
  END subroutine cveloci

#if KEY_CVELOCI==1
  subroutine cveloci_init()
    lcvel=.false.
    return
  end subroutine cveloci_init
#endif 
end module cveloci_mod

