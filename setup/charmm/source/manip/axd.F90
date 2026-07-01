module axd_module

 use chm_kinds
 implicit none

!
! The AXD/BXD functionality was originally implemented into c36a4 in june 2010.
! For additional information see: Glowacki, Paci, & Shalashilin, J. Phys. Chem. B 2009, 113, 16603 - 16611
! Please cite the above if you use this code.
! Dave Glowacki, 5 jun 2010
!

! data available to the following module subroutines: AXDINI, LEAPAXD, VVAXD

 LOGICAL,save :: INVFLAG,INVTEST,LAXD,STORMIN,STORMAX,PRNTALL,PRNGFLG
 LOGICAL,save :: LESTHAN,MORTHAN,BDIS
 INTEGER,save :: NSEL,IUNJUJ,MAXSEL
 INTEGER,allocatable,dimension(:),save :: EMSLCT
 INTEGER,allocatable,dimension(:),save :: ISEL
 real(chm_real),save :: RXCMAX,RXCMIN,RHON,RHOT,PMIN,PMAX

!stuff for a sequence of boxes (BXD in the paper)
 LOGICAL,save :: BXFLAG,DESCEND,ASCEND
 INTEGER,save :: NBOUND, EVENTS
 real(chm_real),allocatable,dimension(:),save :: BOUNDS

! end data available to all module subroutines
!===============================================================
#if KEY_AXD==1 /*(axd_outer)*/

contains
!---------------------------------------------------------------

 SUBROUTINE AXDINI(NATOMS)

  use memory
  use comand
  use coord
  use number
  use select
  use stream
  use string
  implicit none

  real(chm_real) :: RXCVMAX,RXCVMIN,BD,PM,PMX,QMARK
  INTEGER :: I, BX, EV, IMARK
  INTEGER,INTENT(IN) :: NATOMS

! see subroutine AXD for definitions of variables below
  INVFLAG=.FALSE.
  INVTEST=.FALSE.
  LAXD=.TRUE.
  LESTHAN=.FALSE.
  MORTHAN=.FALSE.
  RXCVMAX=0.0
  RXCVMIN=0.0
  BXFLAG=.FALSE.
  DESCEND=.FALSE.
  ASCEND=.FALSE.
  PRNGFLG=.FALSE.
  BDIS=.FALSE.
  
! davidglo: error checking
!  WRITE(*,*)'entering AXDINI execution!'
!  WRITE(*,*)'value of LAXD = ',LAXD
! end error checking

! check if the AXD keyword is turned on

  IF (INDXA(COMLYN, COMLEN, 'RESE') .GT. 0) THEN
     if (prnlev > 2) WRITE(OUTU,*)' AXD Module turned off'
     RETURN
  ENDIF
  
  IUNJUJ=GTRMI(COMLYN,COMLEN,'IUNJ',96)

! Different reaction coordinates may be available; new ones may be included by
! adding in some ELSEIF statements for other coordinates
! presently, only distance is implemented

  IF (INDXA(COMLYN, COMLEN, 'DIS' ) .GT. 0) THEN 
    BDIS =.TRUE.
! bond distance is set as the default if there is no rxn coordinate keyword
  ELSE 
    BDIS =.TRUE.
  ENDIF

! EMSLCT is a vector of length NATOMS, with EMSLCT(I)=1 if atom I
!   is selected, and EMSLCT(I)=0 otherwise.
   
  call chmalloc("axd.src","axdini","emslct",natoms,intg=emslct)

! select the atoms whose distance will be calculated
   
  CALL SELCTA(COMLYN,COMLEN,EMSLCT,X,Y,Z,WMAIN,.TRUE.)
      
  NSEL = 0

! ISEL is a vector of length NSEL where NSEL is the number of atoms 
!   selected, with ISEL(I) equal to the idx of the atom(s) selected.
! For BDIS, NSEL has must equal 2, so only ISEL(1) and ISEL(2) will be defined.

  IF(BDIS)THEN
    call chmalloc("axd.src","axdini","isel",2,intg=isel)
    DO I=1,NATOMS
      IF (EMSLCT(I) .EQ. 1) THEN
        NSEL = NSEL + 1
        ISEL(NSEL) = I
      ENDIF
    ENDDO

!   davidglo check which atoms are selected
    if (prnlev > 2) then
       write (OUTU,*) 'AXD> ATOM A has index ',ISEL(1)
       write (OUTU,*) 'AXD> ATOM B has index ',ISEL(2)
    endif

!   Check that a pair of distinct atoms is selected
    IF (NSEL .GT. 2) THEN
      WRITE(OUTU,'(2(A,I5))')' AXDINI> NSEL',NSEL,' IS LARGER THAN 2!!!!'
      CALL WRNDIE(-1, '<AXDINI>', 'YOU CANNOT DEFINE A DISTANCE FOR MORE THAN 2 ATOMS')
    ELSE IF (NSEL .LT. 2) THEN
      WRITE(OUTU,'(2(A,I5))')' AXDINI> NSEL',NSEL,' IS LESS THAN 2!!!!'
      CALL WRNDIE(-1, '<AXDINI>', 'YOU CANNOT DEFINE A DISTANCE FOR LESS THAN 2 ATOMS')
    ELSE IF (ISEL(1).EQ.ISEL(2)) THEN
      WRITE(OUTU,'(2(A,I5))')' AXDINI> NSEL',NSEL,' THE 2 ATOMS SELECTED ARE THE SAME!!!!'
      CALL WRNDIE(-1, '<AXDINI>', 'YOU CANNOT SELECT THE SAME 2 ATOMS')
    ENDIF
! if you add another rxn coordinate, then the ENDIF below would become an ELSEIF
  ENDIF
  
! print out the number of atoms selected
  IF (PRNLEV.GE.2) WRITE(OUTU,*)'AXDINI> ',nsel,' ATOMS SELECTED'

! print options follow- there are 2: PRNALL & PRANGE
! PRNALL: option to print all data points to the output file

  IF (INDXA(COMLYN, COMLEN, 'PRNALL') .GT. 0) THEN
    PRNTALL =.TRUE.
    if (prnlev > 2) WRITE(OUTU,*)'AXDINI> PRNALL: AXD OUTPUT WILL PRINT EVERY TIMESTEP'
  ENDIF

! PRANGE: option to print between a range of specified data points 
  IF(INDX(COMLYN, COMLEN, 'PRANGE', 6) .GT. 0)THEN
!   be sure than both PRNALL and PRANGE arent specified
    IF(PRNTALL)THEN
      CALL WRNDIE(-1,'<AXDINI>',' YOU HAVE SPECIFIED BOTH PRNALL AND PRANGE. CHOOSE ONE OR THE OTHER')          
    ENDIF
!   get the first PRANGE value, PMIN
    PM=GTRMF(COMLYN,COMLEN,'PRANGE',FMARK)
!   check that the rxn coordinate is GE.0; this is specific to bond distance being the reaction coordinate
    IF(BDIS)THEN
      MAXSEL=2
      IF(PM.GE.0) THEN
        PMIN=PM
        if (prnlev > 2) WRITE(OUTU,*)'AXDINI> PRANGE INPUT: PMIN=',PMIN
!       get the second PRANGE value, PMAX
        PMX=NEXTF(COMLYN,COMLEN)
          IF(PMX.GE.0)THEN
            PMAX=PMX
            if (prnlev > 2) WRITE(OUTU,*)'AXDINI> PRANGE INPUT: PMAX=',PMAX
          ELSE
            CALL WRNDIE(-1, '<AXDINI>', &
                 'REQUIRED 2nd VALUE FOLLOWING PRANGE MUST BE GT OR EQ TO ZERO. CHECK YOUR INPUT')
          ENDIF
      ELSE
        CALL WRNDIE(-1, '<AXDINI>', 'THE REQUIRED 1st VALUE FOLLOWING PRANGE MUST BE GT OR EQ ZERO. CHECK YOUR INPUT')
      ENDIF
    ENDIF
!   check to be sure than PMIN<PMAX
    IF(PMIN.GE.PMAX)THEN
      if (prnlev > 2) WRITE (OUTU, *) 'PMIN=',PMIN,'PMAX=',PMAX
      CALL WRNDIE(-1, '<AXDINI>', 'THE 1st INPUT, PMIN, MUST BE LT THE 2nd INPUT, PMAX')
    ELSEIF(PMIN.LT.PMAX)THEN
      PRNGFLG=.TRUE.
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXDINI> PRANGE HAS BEEN SPECIFIED:' 
         WRITE(OUTU,*)'AXD WILL PRINT OUTPUT WHEN THE REACTION COORD IS BETWEEN',PMIN,'AND',PMAX,'ANGSTROMS'
      endif
    ENDIF

  ENDIF

!  What to do if NBOUND is specified - i.e., we are reading in a set of boundaries
!  NBOUND = number of boundaries
!  EVENTS = number of inversion events after which we reset the box bounds
!  BOUNDS = a list of the boundaries, in either ascending or descending order

  IF(INDX(COMLYN, COMLEN, 'NBOUND', 6) .GT. 0) THEN
!                  -----routine for reading in inversion events----
    EV=GTRMI(COMLYN,COMLEN,'EVENTS',IMARK)
    IF(EV.GT.0)THEN
      EVENTS=EV
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXDINI> TRANSITION BETWEEN BOUNDS WILL BE AFTER',EVENTS,&
              ' INVERSION EVENTS OCCUR'
      endif
    ELSE
      if (prnlev > 2) WRITE(OUTU,*)'AXDINI> INVERSION EVENTS =',EVENTS
      CALL WRNDIE(-1, '<AXDINI>', 'AXD COULD NOT FIND SPECIFIED NUMBER OF INVERSION EVENTS. CHECK YOUR INPUT')
    ENDIF
!                  ----routine for reading in number of BOUNDS----
    BX=GTRMI(COMLYN,COMLEN,'NBOUND',IMARK)
    IF(BX.GT.0) THEN
      NBOUND=BX
      BXFLAG =.TRUE.
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXDINI> ',NBOUND,' BOUNDS ARE SPECIFIED'
         WRITE(OUTU,*)'AXDINI> AXD EXPECTS ',(NBOUND),' BOUNDARIES FOLLOWING THE BOUND INPUT'
      endif
!                  ------code for reading box boundaries----
      call chmalloc("axd.src","AXDINI","BOUNDS",NBOUND,crl=BOUNDS)
      IF (INDXA(COMLYN, COMLEN, 'BOUNDS') .GT. 0) THEN
        if (prnlev > 2) WRITE(OUTU,*)'AXDINI> READING THE INPUT FOLLOWING BOUNDS'
        DO I=1,NBOUND
          BD=NEXTF(COMLYN,COMLEN)
!         check that the rxn coordinate bounds are GE.0; this is specific to bond distance being the reaction coordinate
          IF(BDIS)THEN
            IF(BD.GE.0)THEN
              BOUNDS(I)=BD
              if (prnlev > 2) WRITE(OUTU,*)'AXDINI> BOUNDARY',I,' = ',BD                
            ELSE
              if (prnlev > 2) WRITE(OUTU,*)'AXDINI> BOUNDARY',I,' = ',BD                
              CALL WRNDIE(-1, '<AXDINI>', 'AXD REQUIRES POSITIVE BOX BOUNDARIES. CHECK YOUR INPUT')
            ENDIF
          ENDIF
        ENDDO
      ELSE
        CALL WRNDIE(-1, '<AXDINI>', 'AXD COULD NOT FIND BOUNDS, REQUIRED WITH NBOUND KEYWORD. CHECK INPUT')
      ENDIF
!                -----end code for reading box boundaries---
    ELSE
      if (prnlev > 2) WRITE(OUTU,'(2(A,I5))')' AXDINI> NBOUND',BX,' IS EITHER UNDEFINED OR LESS THAN/EQUAL TO ZERO'
      CALL WRNDIE(-1, '<AXDINI>', 'AXD ERROR READING SPECIFICATION OF NBOUND. CHECK YOUR INPUT')
    ENDIF

!   CALL CHECKANDSETBOXES in order to do some error checking, see whether they are in  
!   ascending (ASCEND) or descending (DESCEND) order.  Then, initialize RXCMAX,
!   RXCMIN, ASCEND, DESCEND

    CALL CHECKANDSETBOXES(BOUNDS,NBOUND,RXCMAX,RXCMIN,ASCEND,DESCEND)

!   set both LESTHAN and MORTHAN so they correspond to the values of the first 'box' (see comments below)

    LESTHAN=.TRUE.
    MORTHAN=.TRUE.

! if NBOUND isnt specified, the AXD schemes available are: 
! 1) a retaining wall (set LESTHAN) 
! 2) an excluding wall (set MORTHAN) 
! 3) a box (set both LESTHAN and MORTHAN)
! The code below determines how to set LESTHAN and MORTHAN based on whether MAX or MIN 
! or both are specified

  ELSE
    IF (INDX(COMLYN, COMLEN, 'MAX', 3) .GT. 0) THEN
      LESTHAN  =.TRUE.
    ENDIF

    IF (INDX(COMLYN, COMLEN, 'MIN', 3) .GT. 0) THEN
      MORTHAN =.TRUE.
    ENDIF

!   get the AXD distances between atoms and do some error checking:
!   1) if LT, the distance read in is the maximum allowed A-B distance
!   2) if GT, the distance read in is the minimum allowed A-B distance
!   3) if BOX, then we read in both the maximum & minimum allowed A-B distances 

!   read in the maximum separation if MAX was specified in the input

    IF(LESTHAN) THEN
      RXCVMAX=GTRMF(COMLYN,COMLEN,'MAX',FMARK)
!     check that value is GE.0; this is specific to bond distance being the reaction coordinate
      IF(BDIS)THEN
        IF(RXCVMAX.GT.0) THEN
          RXCMAX=RXCVMAX
          if (prnlev > 2) WRITE(OUTU,*)'AXDINI> MAX',RXCMAX,&
               ' IS THE MAXIMUM ALLOWED DISTANCE BETWEEN A AND B'
        ELSE
          if (prnlev > 2) WRITE(OUTU,'(A,F19.8)')'AXDINI> MAX',RXCVMAX,&
               ' IS EITHER UNDEFINED OR LESS THAN/EQUAL TO ZERO'
          CALL WRNDIE(-1, '<AXDINI>', 'AXD ERROR REGARDING SPECIFICATION OF MAX. CHECK YOUR INPUT')
        ENDIF
      ENDIF
    ENDIF

!   read in the minimum separation if MIN was specified in the input

    IF(MORTHAN) THEN
      RXCVMIN=GTRMF(COMLYN,COMLEN,'MIN',FMARK)
!     check that value is GE.0; this is specific to bond distance being the reaction coordinate
      IF(BDIS)THEN
        IF(RXCVMIN.GE.0) THEN
          RXCMIN=RXCVMIN
          if (prnlev > 2) WRITE(OUTU,*)'AXDINI> MIN',RXCMIN,&
               ' IS THE MINIMUM ALLOWED DISTANCE BETWEEN A AND B'
        ELSE
          if (prnlev > 2) WRITE(OUTU,'(A,F19.8)')'AXDINI> MIN',RXCVMIN,&
               ' IS EITHER UNDEFINED OR LESS THAN/EQUAL TO ZERO'
          CALL WRNDIE(-1, '<AXDINI>', 'AXD ERROR REGARDING SPECIFICATION OF MIN. CHECK YOUR INPUT')
        ENDIF
      ENDIF
    ENDIF

!   if both MIN and MAX are specified in the input, make sure than RXCVMIN < RXCVMAX

    IF(MORTHAN.AND.LESTHAN) THEN
      IF(RXCVMIN.GE.RXCVMAX) THEN
        CALL WRNDIE(-1, '<AXDINI>', 'IF YOU SPECIFY BOTH MIN AND MAX, MIN MUST&
             & BE LESS THAN MAX. CHECK YOUR INPUT')
      ENDIF
    ENDIF

  ENDIF

  IF (.NOT.(LESTHAN) .AND. .NOT.(MORTHAN)) THEN
    CALL WRNDIE(-1, '<AXDINI>', 'YOU HAVE TO GIVE AN AXD COMMAND (MIN/MAX/NBOUND)')
  ENDIF

  RETURN

 END SUBROUTINE AXDINI


 SUBROUTINE LEAPAXD(QCX,QCY,QCZ,VXDT,VYDT,VZDT,VXDTP,VYDTP,VZDTP,MASS,NATOMX,STEP,LASTEP)
!-----------------------------------------------------------------------
!     THIS SUBROUTINE IS THE AXD IMPLEMENTATION FOR THE LEAPFROG INTEGRATION
!     ALGORITHM (CALLED FROM DYNAMC.SRC)
!
!     QCX,QCY,QCZ - coordinates at time=t-1 passed into AXD at the beginning 
!                   of the main dynamics loop in dynamc (XCOMP,YCOMP,ZCOMP)
!     VXDT,VYDT,VZDT - the velocities at time=t-1/2 which are passed into
!                   AXD (XOLD,YOLD,ZOLD) at the beginning of the main 
!                   dynamics loop in dynamc
!     VXDTP,VYDTP,VZDTP - the velocities at time=t-3/2 which are passed 
!                   into AXD (XNEW,YNEW,ZNEW) at the beginning of the main 
!                   dynamics loop in dynamc.
!                **note that the velocities are not exactly the velocities; 
!                  rather they are equal to the velocities*dt
!     NATOMX - Number of atoms
!     STEP - integrator iteration
!     LASTEP - final integrator iteration
!
!     ***note: RXCV is an abbreviation for 'Reaction Coordinate Value', and 
!              should be interpreted as such wherever it occurs in related
!              variants (e.g., RXCVMAX, RXCVMIN, etc).   
!
!     Author: Dave Glowacki 3-3-09
!-------------------------------------------------------------------------

  use contrl
  use chm_kinds
  use stream
  use memory
  use reawri
  use dimens_fcm
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none

  LOGICAL :: INVIN, INVOUT

! following arrays hold coords for time=t
  real(chm_real),dimension(NATOMX),intent(in) :: QCX,QCY,QCZ
! following arrays hold velocities for time=t+1/2
  real(chm_real),dimension(NATOMX),intent(inout) :: VXDT,VYDT,VZDT
! following arrays hold velocities for time=t-1/2
  real(chm_real),dimension(NATOMX),intent(in) :: VXDTP,VYDTP,VZDTP
! following arrays hold coords for next time step
  real(chm_real),allocatable,dimension(:) :: QCXN,QCYN,QCZN

  INTEGER :: CTR, STEP
  INTEGER,intent(in) :: NATOMX,LASTEP
! data for getting masses
  real(chm_real),save :: MA, MB
  real(chm_real),dimension(NATOMX),intent(in) :: MASS
! arrays to hold necessary data for velocity inversion in the (CM) frame
  real(chm_real),dimension(3) :: VA,VB,QA,QB
  real(chm_real) :: XIJ,YIJ,ZIJ,RHOP,RXCV
  real(chm_real),save :: RHO0

!
! what to do the first time that AXD is called,including calculating the
! distance RIJ at t=0
!
  IF (STEP.EQ. 1) THEN

!   print-out some information
    CALL INITIAL_PRINTS(LESTHAN,MORTHAN,BXFLAG,RXCMAX,RXCMIN,NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX)
 
    IF(BDIS)THEN
!     the subroutine below evaluates the bond distance as RXCV 
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX,RXCV)
        RHOT = RXCV
!       get the masses, MA and MB, of atoms A and B 
        MA=MASS(ISEL(1))
        MB=MASS(ISEL(2))
    ENDIF

!   and assigns the reaction coordinate value to RHO0, which is the value of the rxn coordinate at time = 0
    RHO0 = RHOT

    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)')'AXD> INITIAL VALUE OF RXN COORDINATE = ',RHO0

!   if BXFLAG, then set the values of RXCMIN and RXCMAX based on the initial value of the reaction coordinate
    IF(BXFLAG) THEN
      CALL SET_INITIAL_BOX(RHO0,BOUNDS,NBOUND,RXCMAX,RXCMIN,ASCEND,DESCEND)
    ENDIF

!   if the RHO < RXCMAX & we have MORTHAN, or if RHO > RXCMIN & we have LESTHAN, 
!   then set the flag to turn on the inversion test.  the inversion tes wont be
!   turned on until the first time that these conditions are met

    IF(RHO0.LT.RXCMAX.AND.LESTHAN) INVTEST=.TRUE.

    IF(RHO0.GT.RXCMIN.AND.MORTHAN) INVTEST=.TRUE.

    IF(BXFLAG) THEN
      CALL BOUNMANAGER(BOUNDS,NBOUND,RXCMAX,RXCMIN,STEP,EVENTS,INVFLAG,INVIN,INVOUT,ASCEND,DESCEND)
    ENDIF

  ENDIF

! what to do if
!   1) this isnt the first time AXD is called
!   OR
!   2) INVTEST was set to TRUE the first time that AXD was called

  IF (STEP.NE.1.OR.INVTEST) THEN

    call chmalloc("axd.src","LEAPAXD","QCXN",NATOMX,crl=QCXN)
    call chmalloc("axd.src","LEAPAXD","QCYN",NATOMX,crl=QCYN)
    call chmalloc("axd.src","LEAPAXD","QCZN",NATOMX,crl=QCZN)

!   if INVFLAG is true, it tells dynamc that we are inverting the velocity 
!   coordinates. As the inversion test is yet to be performed, set it to false
 
    INVFLAG=.FALSE.
    INVIN=.FALSE.
    INVOUT=.FALSE.

    IF(BDIS)THEN
!     calculate RXCV (bond distance)
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX,RXCV)
!     assign RXCV to RHOT (value of rxn coordinate at time T)     
      RHOT=RXCV
    ENDIF

    IF(BDIS)THEN
!     calculate the coordinates of A and B at the next time step (QCXN,QCYN,QCZN)
      QCXN(ISEL(1))=QCX(ISEL(1))+VXDT(ISEL(1))
      QCYN(ISEL(1))=QCY(ISEL(1))+VYDT(ISEL(1))
      QCZN(ISEL(1))=QCZ(ISEL(1))+VZDT(ISEL(1))
      QCXN(ISEL(2))=QCX(ISEL(2))+VXDT(ISEL(2))
      QCYN(ISEL(2))=QCY(ISEL(2))+VYDT(ISEL(2))
      QCZN(ISEL(2))=QCZ(ISEL(2))+VZDT(ISEL(2))
!     use QCXN,QCYN,QCZN to calculate RXCV at the next time step
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QCXN,QCYN,QCZN,NATOMX,RXCV)
!     assign RXCV to RHON i.e.- RHO at time T+1 -i.e, the Next geometry
      RHON=RXCV
    ENDIF

!   if the following conditions are met, then dynamc will execute an incorrect step
!   and we need to invert VXDT, VYDT, and VZDT 

    IF(RHOT.LT.RXCMAX.AND.RHON.GT.RXCMAX.AND.LESTHAN) THEN
      INVFLAG=.TRUE. 
      INVIN=.TRUE.
    ENDIF

    IF(RHOT.GT.RXCMIN.AND.RHON.LT.RXCMIN.AND.MORTHAN) THEN
      INVFLAG=.TRUE. 
      INVOUT=.TRUE.
    ENDIF

    IF(BXFLAG) THEN
      CALL BOUNMANAGER(BOUNDS,NBOUND,RXCMAX,RXCMIN,STEP,EVENTS,INVFLAG,INVIN,INVOUT,ASCEND,DESCEND)
    ENDIF
         
    IF(INVFLAG) THEN

!     write some output to indicate that we are inverting

     if (prnlev > 2) WRITE(OUTU,FMT=100)'AXD> THRESHOLD CROSSED AT STEP ',STEP,&
          ': RHO(T)= ',RHOT,'; RHO(T+1)=',RHON
100  FORMAT(A31,I13,A11,F10.3,A12,F10.3)

! davidglo some error checking
!      write(*,*)'axd:QA(1-3),QB(1-3) at 327:',QCX(ISEL(1)),QCY(ISEL(1)),QCZ(ISEL(1)),QCX(ISEL(2)),QCY(ISEL(2)),QCZ(ISEL(2))
!      write(*,*)'axd:QA(1-3),QB(1-3) at 327:',QCXN(ISEL(1)),QCYN(ISEL(1)),QCZN(ISEL(1)),QCXN(ISEL(2)),QCYN(ISEL(2)),QCZN(ISEL(2))
! end error checking

      IF(BDIS)THEN
!       set up QA, QB, VA, and VB.  QA and QB are 3 element vectors holding the x,y,z
!       coords.  VA and VB hold V(x,y,z)*dt of atoms A & B, and will be inverted
        QA(1)=0.5D+0*(QCX(ISEL(1))+QCXN(ISEL(1)))
        QA(2)=0.5D+0*(QCY(ISEL(1))+QCYN(ISEL(1)))
        QA(3)=0.5D+0*(QCZ(ISEL(1))+QCZN(ISEL(1)))
        QB(1)=0.5D+0*(QCX(ISEL(2))+QCXN(ISEL(2)))
        QB(2)=0.5D+0*(QCY(ISEL(2))+QCYN(ISEL(2)))
        QB(3)=0.5D+0*(QCZ(ISEL(2))+QCZN(ISEL(2)))

        VA(1)=VXDT(ISEL(1))
        VA(2)=VYDT(ISEL(1))
        VA(3)=VZDT(ISEL(1))
        VB(1)=VXDT(ISEL(2))
        VB(2)=VYDT(ISEL(2))
        VB(3)=VZDT(ISEL(2))

!       invert the velocities in the CM frame
        CALL INVERTDIST(QA,QB,VA,VB,MA,MB)

!       assign inverted velocities to the appropriate vectors
        VXDT(ISEL(1))=VA(1)
        VYDT(ISEL(1))=VA(2)
        VZDT(ISEL(1))=VA(3)
        VXDT(ISEL(2))=VB(1)
        VYDT(ISEL(2))=VB(2)
        VZDT(ISEL(2))=VB(3)
      ENDIF
    ENDIF
    call chmdealloc("axd.src","LEAPAXD","QCXN",NATOMX,crl=QCXN)
    call chmdealloc("axd.src","LEAPAXD","QCYN",NATOMX,crl=QCYN)
    call chmdealloc("axd.src","LEAPAXD","QCZN",NATOMX,crl=QCZN)
  ENDIF
  
! print the RHO at time T to an output file
  CALL AXDPRINT(PRNTALL,INVIN,INVOUT,STEP,RHOT,RXCMIN,RXCMAX,NPRINT,IUNJUJ,PRNGFLG,PMIN,PMAX)

 RETURN
  
 END SUBROUTINE LEAPAXD


 SUBROUTINE VVAXD(QCX,QCY,QCZ,QPX,QPY,QPZ,VCX,VCY,VCZ,DCX,DCY,DCZ,MASS,NATOMX,STEP,LASTEP)
!-----------------------------------------------------------------------
!     THIS SUBROUTINE IS THE AXD IMPLEMENTATION FOR THE VELOCITY VERLET (VV2) 
!     INTEGRATION ALGORITHM (CALLED FROM DYNVV2.SRC)
!
!     QCX,QCY,QCZ - (X,Y,Z) coordinates at Current time t passed into AXD
!     QPX,QPY,QPZ - (XOLD,YOLD,ZOLD) coordinates at Previous time t-1 passed into AXD
!     VCX,VCY,VCZ - (VX,VY,VZ) velocities at Current time t passed into AXD
!     VPX,VPY,VPZ - velocities at Previous time t-1 stored by AXD
!     DCX,DCY,DCZ - (DX,DY,DZ) forces at Current time t passed into AXD
!     DPX,DPY,DPZ - forces at Previous time t-1 stored by AXD
!     MASS - array of masses for all atoms in the system
!     NATOMX - Number of atoms
!     STEP - integrator iteration
!     LASTEP - final integrator iteration
!
!     ***note: RXCV is an abbreviation for 'Reaction Coordinate Value', and 
!              should be interpreted as such wherever it occurs in related
!              variants (e.g., RXCVMAX, RXCVMIN, etc).  
!
!     Author: Dave Glowacki 3-3-09
!-----------------------------------------------------------------------

  use contrl
  use chm_kinds
  use stream
  use memory
  use reawri
  use dimens_fcm
#if KEY_PARALLEL==1
  use parallel
#endif 
  implicit none

  INTEGER,intent(in) :: NATOMX,LASTEP

! following arrays hold info for Current time=t
  real(chm_real),dimension(NATOMX),intent(inout) :: QCX,QCY,QCZ
  real(chm_real),dimension(NATOMX),intent(inout) :: VCX,VCY,VCZ
  real(chm_real),dimension(NATOMX),intent(inout) :: DCX,DCY,DCZ

! following arrays store info for Previous time t-1 (need not save)
  real(chm_real),dimension(NATOMX),intent(inout) :: QPX,QPY,QPZ
! following arrays store info for Previous time t-1 (need to save)
  real(chm_real),allocatable,dimension(:),save :: VPX,VPY,VPZ
  real(chm_real),allocatable,dimension(:),save :: DPX,DPY,DPZ

  LOGICAL :: INVIN, INVOUT
  INTEGER :: CTR, STEP
! data for getting masses
  real(chm_real),save :: MA, MB
  real(chm_real),dimension(NATOMX),intent(in) :: MASS
! arrays to hold necessary data for velocity inversion in the (CM) frame
  real(chm_real),dimension(3) :: VA,VB,QA,QB
  real(chm_real) :: XIJ,YIJ,ZIJ,RHOP,RXCV
  real(chm_real),save :: RHO0

! the code that follows is for execution the first time that AXD is called
! and includes calculating the distance RIJ at t=0

  IF (STEP.EQ. 1) THEN

    call chmalloc("axd.src","VVAXD","VPX",NATOMX,crl=VPX)
    call chmalloc("axd.src","VVAXD","VPY",NATOMX,crl=VPY)
    call chmalloc("axd.src","VVAXD","VPZ",NATOMX,crl=VPZ)
    call chmalloc("axd.src","VVAXD","DPX",NATOMX,crl=DPX)
    call chmalloc("axd.src","VVAXD","DPY",NATOMX,crl=DPY)
    call chmalloc("axd.src","VVAXD","DPZ",NATOMX,crl=DPZ)

!   print-out some information

    CALL INITIAL_PRINTS(LESTHAN,MORTHAN,BXFLAG,RXCMAX,RXCMIN,NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX)

!   the subroutine below evaluates the distance and assigns it to RHO0,
!   which is the value of the rxn coordinate at time = 0

    IF(BDIS)THEN
!     the subroutine below evaluates the bond distance as RXCV 
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX,RXCV)
      RHOT = RXCV
!     get the masses, MA and MB, of atoms A and B 
      MA=MASS(ISEL(1))
      MB=MASS(ISEL(2))
    ENDIF

!   assign the reaction coordinate value to RHO0, which is the value of the rxn coordinate at time = 0
    RHO0 = RHOT

    IF (PRNLEV.GE.2) WRITE(OUTU,'(A,F13.4)')'AXD> INITIAL VALUE OF RXN COORDINATE = ',RHO0

!   set the flag for storing information to false
    STORMAX=.FALSE.
    STORMIN=.FALSE.
    INVFLAG=.FALSE.
    INVIN=.FALSE.
    INVOUT=.FALSE.

!   if BXFLAG, then set the values of RXCMIN and RXCMAX based on the initial value 
!   of the reaction coordinate

    IF(BXFLAG) THEN
      CALL SET_INITIAL_BOX(RHO0,BOUNDS,NBOUND,RXCMAX,RXCMIN,ASCEND,DESCEND)
    ENDIF

!   if the distance between A and B, RHO, is less than RXCMAX, then set the flag to store information.  
!   if RHO > RXCMAX, the info wont be stored until the first time that RHO < RXCMAX

    IF(RHO0.LT.RXCMAX.AND.LESTHAN) THEN
      STORMAX=.TRUE.
    ENDIF

    IF(RHO0.GT.RXCMIN.AND.MORTHAN) THEN
      STORMIN=.TRUE.
    ENDIF

    IF(BXFLAG) THEN
      CALL BOUNMANAGER(BOUNDS,NBOUND,RXCMAX,RXCMIN,STEP,EVENTS,INVFLAG,INVIN,INVOUT,ASCEND,DESCEND)
    ENDIF

  ENDIF

! what to do if this isnt the first time AXD is called

  IF (STEP.NE.1) THEN

! set the flag for performing inversion to false

    INVFLAG=.FALSE.
    INVIN=.FALSE.
    INVOUT=.FALSE.

    IF(BDIS)THEN
!     calculate bond distance
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QCX,QCY,QCZ,NATOMX,RXCV)
!     assign the bond distance, RXCV, to RHOT (RHO at time T)
      RHOT=RXCV
    ENDIF
   
!   the first time that RHOT is less than RXCMAX, turn on STORE, and leave it on
!   for the duration of the trajectory

    IF(RHOT.LT.RXCMAX.AND.LESTHAN) THEN
      STORMAX=.TRUE.
    ENDIF

    IF(RHOT.GT.RXCMIN.AND.MORTHAN) THEN
      STORMIN=.TRUE.
    ENDIF

!   if RHOT >= RXCMAX and STORE is on, we executed an incorrect step and need 
!   a velocity inversion; INVIN is on if we did an inversion wrt to RXCMAX
!   INVOUT is on if we did an inversion wrt to RXCMIN

    IF(RHOT.GE.RXCMAX.AND.STORMAX.AND.LESTHAN) THEN 
      INVFLAG=.TRUE.
      INVIN=.TRUE.
    ENDIF

    IF(RHOT.LE.RXCMIN.AND.STORMIN.AND.MORTHAN) THEN 
      INVFLAG=.TRUE.
      INVOUT=.TRUE.
    ENDIF

!   if we are doing dynamics in boxes, then call BOUNMANAGER to determine
!   if the boundaries need to be updated

    IF(BXFLAG) THEN
      CALL BOUNMANAGER(BOUNDS,NBOUND,RXCMAX,RXCMIN,STEP,EVENTS,INVFLAG,INVIN,INVOUT,ASCEND,DESCEND)
    ENDIF

  ENDIF

! if STORE is on, INVFLAG is false, and RHO<RXCMAX or RHO>RXCMIN, then 
! assign information from time=t vectors to those holding information for t-1

  IF(.NOT.(INVFLAG)) THEN
    IF (RHOT.LT.RXCMAX.AND.STORMAX.AND.LESTHAN) THEN
!      write(*,*)'axd: storing data from present step to P vectors'
      DO CTR=1,NATOMX
        VPX(CTR)=VCX(CTR)
        VPY(CTR)=VCY(CTR)
        VPZ(CTR)=VCZ(CTR)
        DPX(CTR)=DCX(CTR)
        DPY(CTR)=DCY(CTR)
        DPZ(CTR)=DCZ(CTR)
      ENDDO
    ENDIF
    IF (RHOT.GT.RXCMIN.AND.STORMIN.AND.MORTHAN) THEN
!      write(*,*)'axd: storing data from present step to P vectors'
      DO CTR=1,NATOMX
        VPX(CTR)=VCX(CTR)
        VPY(CTR)=VCY(CTR)
        VPZ(CTR)=VCZ(CTR)
        DPX(CTR)=DCX(CTR)
        DPY(CTR)=DCY(CTR)
        DPZ(CTR)=DCZ(CTR)
      ENDDO
    ENDIF
      
! the coordinate transform and subsequent inversion

  ELSEIF(INVFLAG) THEN

    IF(BDIS)THEN
!     calculate bond distance for the previous set of coordinates
      CALL CALCULATE_DISTANCE(NSEL,ISEL,MAXSEL,QPX,QPY,QPZ,NATOMX,RXCV)
!     assign RXCV to RHOP
      RHOP=RXCV
    ENDIF

!   write some output to indicate that we are inverting

    if (prnlev > 2) WRITE(OUTU,FMT=100)'AXD> THRESHOLD CROSSED AT STEP ',STEP,&
         ': RHO(T)= ',RHOT,'; RHO(T+1)=',RHON
100 FORMAT(A31,I13,A11,F10.3,A12,F10.3)

    IF(BDIS)THEN
!     QA,QB and VA,VB are 3 element vectors holding the x,y,z coords and velocities of atoms A & B from time (t-1)  
      QA(1)=QPX(ISEL(1))
      QA(2)=QPY(ISEL(1))
      QA(3)=QPZ(ISEL(1))
      QB(1)=QPX(ISEL(2))
      QB(2)=QPY(ISEL(2))
      QB(3)=QPZ(ISEL(2))

      VA(1)=VPX(ISEL(1))
      VA(2)=VPY(ISEL(1))
      VA(3)=VPZ(ISEL(1))
      VB(1)=VPX(ISEL(2))
      VB(2)=VPY(ISEL(2))
      VB(3)=VPZ(ISEL(2))

!     invert the velocities in the CM frame
      CALL INVERTDIST(QA,QB,VA,VB,MA,MB)
!     assign the inverted velocities to the appropriate elements in the t-1 vector
      VPX(ISEL(1))=VA(1)
      VPY(ISEL(1))=VA(2)
      VPZ(ISEL(1))=VA(3)
      VPX(ISEL(2))=VB(1)
      VPY(ISEL(2))=VB(2)
      VPZ(ISEL(2))=VB(3)
          
! if INVFLAG is turned on, then inversion in AXD has occurred and we need to:
! 1) overwrite the present coordinates (X,Y,Z) with those from the previous 
!    step: XOLD,YOLD,ZOLD  (overwrite QC with QP)
! 2) overwrite the present forces (DX,DY,DZ) with those from the previous step: 
!    DPX,DPY,DPZ, which are stored by AXD (overwrite DC with DP)
! 3) overwrite the present velocities (VX,VY,VZ) with inverted velocities from
!    the previous step: VPX,VPY,VPZ, which are stored and calculated by AXD
!    (overwrite VC with VP) 
 
    ENDIF

    DO CTR=1,NATOMX
      QCX(CTR)=QPX(CTR)
      QCY(CTR)=QPY(CTR)
      QCZ(CTR)=QPZ(CTR)
      DCX(CTR)=DPX(CTR)
      DCY(CTR)=DPY(CTR)
      DCZ(CTR)=DPZ(CTR)
      VCX(CTR)=VPX(CTR)
      VCY(CTR)=VPY(CTR)
      VCZ(CTR)=VPZ(CTR)
    ENDDO    
                                
  ENDIF

! if PRNTALL is specified, print to the output file RHO at each timestep.  
! otherwise, print at the specified print frequency.  also if an inversion event 
! occurred, print the wall at which the inversion occurred

  CALL AXDPRINT(PRNTALL,INVIN,INVOUT,STEP,RHOT,RXCMIN,RXCMAX,NPRINT,IUNJUJ,PRNGFLG,PMIN,PMAX)

! if we're at the last dynamics step, then deallocate memory
  IF (STEP.EQ.LASTEP) THEN
    call chmdealloc("axd.src","VVAXD","VPX",NATOMX,crl=VPX)
    call chmdealloc("axd.src","VVAXD","VPY",NATOMX,crl=VPY)
    call chmdealloc("axd.src","VVAXD","VPZ",NATOMX,crl=VPZ)
    call chmdealloc("axd.src","VVAXD","DPX",NATOMX,crl=DPX)
    call chmdealloc("axd.src","VVAXD","DPY",NATOMX,crl=DPY)
    call chmdealloc("axd.src","VVAXD","DPZ",NATOMX,crl=DPZ)
  ENDIF

 RETURN

 END SUBROUTINE VVAXD

!===============================================================
! the subroutines below are used by AXD module subroutines, but dont
! require access to the module data members 
!

SUBROUTINE CHECKANDSETBOXES(BDS,NBOUN,SMAX,SMIN,ASC,DES)

!-----------------------------------------------------------------------
!
! This routine is for checking that the boxes are in either ascending or 
! descending order, and for setting SMIN and SMAX based on the initial box
! boundaries within the BDS array
! 
! NBOUN - number of boundaries
! BDS - array holding the boundaries
! SMIN & SMAX - RXCMIN & RXCMAX
! ASC & DES - ASCEND & DESCEND
!
! dave glowacki, 25 feb 09
!
!-----------------------------------------------------------------------

  use stream
  use chm_kinds
  implicit none

  LOGICAL,intent(inout) :: ASC,DES
  LOGICAL :: BXOK
  INTEGER,intent(in) :: NBOUN
  INTEGER :: CTR, J
  real(chm_real), intent(inout) :: SMIN,SMAX
  real(chm_real),dimension(NBOUN),intent(in) :: BDS

  ASC=.FALSE.
  DES=.FALSE.      

! check that the boundaries are in either ascending or descending order
! first, check if they are in descending order (i.e., ctr(i)>ctr(i+1))

  J=1
  DO CTR=1,NBOUN-1
    IF(BDS(J).GT.BDS(J+1)) THEN 
      BXOK=.TRUE.
      J=J+1
    ELSE 
      BXOK=.FALSE.
    ENDIF
  ENDDO

  IF(BXOK) THEN
    if (prnlev > 2) WRITE(OUTU,*)'AXDINI> BOUNDARIES ARE IN DESCENDING ORDER'
    DES=.TRUE.
  ENDIF

! if they are in descending order, then BXOK = TRUE.  if BXOK is still
! false, check if they are ascending order (i.e., ctr(i)<ctr(i+1))

  J=1
  IF(.NOT.(BXOK)) THEN
    DO CTR=1,NBOUN-1
      IF(BDS(J).LT.BDS(J+1)) THEN
        BXOK=.TRUE.
        J=J+1
      ELSE 
        BXOK=.FALSE.
      ENDIF
    ENDDO
    IF(BXOK) THEN 
      if (prnlev > 2) WRITE(OUTU,*)'AXDINI> BOUNDARIES ARE IN ASCENDING ORDER'
      ASC=.TRUE.
    ENDIF
  ENDIF

! if they are neither ascending nor descending order, then BXOK = FALSE and we need an error

  IF(.NOT.(BXOK)) THEN      
    CALL WRNDIE(-1, '<AXDINI>', 'AXD REQUIRES THAT THE BOXES BE IN EITHER&
         & ASCENDING OR DESCENDING ORDER. CHECK YOUR INPUT')
  ENDIF

! if the boundaries are in ascending order, then we will assume that the user
! is trying to go from smaller to larger separations, and we will set 
! RXCMAX and RXCMIN accordingly

  IF(ASC) THEN
    SMAX=BDS(2)
    SMIN=BDS(1)        
  ENDIF

! if the boundaries are in descending order, then we will assume that the user
! is trying to go from larger to smaller separations, and we will set 
! RXCMAX and RXCMIN accordingly

  IF(DES) THEN
    SMAX=BDS(1)
    SMIN=BDS(2)        
  ENDIF

  if (prnlev > 2) then
     WRITE(OUTU,*)'AXDINI> AXD BOUNDARIES SET AS FOLLOWS:'
     WRITE(OUTU,*)'AXDINI> ',SMAX,'IS THE MAX ALLOWED RXN COORD VALUE'
     WRITE(OUTU,*)'AXDINI> ',SMIN,'IS THE MIN ALLOWED RXN COORD VALUE'
  endif
  
  RETURN

END SUBROUTINE CHECKANDSETBOXES


SUBROUTINE INITIAL_PRINTS(LT,MT,BXFLG,MAXRXC,MINRXC,NUMSEL,IDXSEL,MAXIDX,CRDX,CRDY,CRDZ,NATOMS)

!-----------------------------------------------------------------------
!     LT,MT - equivalent to LESTHAN, MORTHAN
!     BXFLG - equivalent to BOXFLAG
!     MAXRXC,MINRXC - eq to RXCMAX, RXCMIN
!     NUMSEL - number of atoms selected (2 for a bond distance coordinate)
!     IDXSEL - array that holds the indices of the selected atoms
!     MAXIDX - largest possible dimension of IDXSEL array (see axd.fcm)
!     CRDX,CRDY,CRDZ - arrays holding x,y,z coords for every atom in the system
!     NATOMS - max number of possible atoms in the CRD arrays
!
! This routine is for carrying out actions at the first timestep for the case
! where the reaction coordinate is a bond distance between two atoms
!                                          dave glowacki, 25 feb 09
!-----------------------------------------------------------------------

  use stream
  use chm_kinds
  implicit none
     
  INTEGER :: II,I,J
  INTEGER,intent(in) :: NUMSEL,MAXIDX,NATOMS
  INTEGER,dimension(*),intent(in) :: IDXSEL(MAXIDX)
  LOGICAL,intent(in) :: LT,MT,BXFLG
  real(chm_real),intent(in) ::  MAXRXC,MINRXC
  real(chm_real),dimension(*),intent(in) ::  CRDX,CRDY,CRDZ

  IF (PRNLEV.GE.2) THEN
    WRITE(OUTU,'(A)')'AXD> CALLING AXD'
            
    IF(LT .AND. .NOT.BXFLG) THEN
      WRITE(OUTU,'(A,F19.8)')'AXD> MAXIMUM VALUE OF RXN COORDINATE:',MAXRXC  
    ENDIF

    IF(MT .AND. .NOT.BXFLG) THEN
      WRITE(OUTU,'(A,F19.8)')'AXD> MINIMUM VALUE OF RXN COORDINATE:',MINRXC  
    ENDIF

    WRITE(OUTU,'(A)')'AXD> THE PHASE SPACE IS TRUNCATED'

  ENDIF

  IF (PRNLEV.GE.6) WRITE(OUTU,'(A)')'AXD> COORDINATES OF SELECTED ATOMS IN INITIAL CONFIG.:'
    DO II=1,NUMSEL
      I = IDXSEL(II)
      IF (PRNLEV.GE.6) WRITE(OUTU,'(A9,I5,3F12.4)') 'AXD> ',I,CRDX(I),CRDY(I),CRDZ(I)
    ENDDO
          
  RETURN
     
END SUBROUTINE INITIAL_PRINTS


SUBROUTINE CALCULATE_DISTANCE(NUMSEL,IDXSEL,MAXIDX,COORDX,COORDY,COORDZ,NATOMS,SEP)

!-----------------------------------------------------------------------
!     NUMSEL - number of atoms selected (2 for a bond distance coordinate)
!     IDXSEL - array that holds the indices of the selected atoms
!     MAXIDX - largest possible dimension of IDXSEL array (see axd.fcm)
!     COORDX,COORDY,COORDZ - arrays holding x,y,z coords for every atom in the system
!     NATOMS - max number of possible atoms in the COORD arrays
!     SEP - calculated separation between atoms A and B
!
! This routine is for evaluating the rxn coordinate where it is the
!  bond distance between two atoms
!                                          dave glowacki, 25 feb 09
!-----------------------------------------------------------------------
     
  use chm_kinds
  implicit none

  INTEGER,intent(in) :: NUMSEL,MAXIDX,NATOMS
  INTEGER,dimension(*),intent(in) :: IDXSEL
  real(chm_real) :: XDIST,YDIST,ZDIST
  real(chm_real),intent(inout) :: SEP
  real(chm_real),dimension(*),intent(in) :: COORDX,COORDY,COORDZ
     
  XDIST=COORDX(idxsel(1))-COORDX(idxsel(2))
  YDIST=COORDY(idxsel(1))-COORDY(idxsel(2))
  ZDIST=COORDZ(idxsel(1))-COORDZ(idxsel(2))

  SEP = SQRT(XDIST*XDIST+YDIST*YDIST+ZDIST*ZDIST)
               
  RETURN
     
END SUBROUTINE CALCULATE_DISTANCE


SUBROUTINE SET_INITIAL_BOX(INITVAL,BDS,NBOUN,SMAX,SMIN,ASC,DES)

!-----------------------------------------------------------------------
!
! This routine is for setting SMIN and SMAX based on the initial
! value of the reaction coordinate
! 
! INITVAL - initial value of the reaction coordinate
! NBOUN - number of boundaries
! BDS - array holding the boundaries
! SMIN & SMAX - RXCMIN & RXCMAX
! ASC & DES - ASCEND & DESCEND
!
! dave glowacki, 25 feb 09
!-----------------------------------------------------------------------

  use stream
  use chm_kinds
  implicit none

  LOGICAL,intent(in) :: ASC,DES
  INTEGER,intent(in) :: NBOUN
  INTEGER :: I
  real(chm_real),dimension(NBOUN),intent(in) :: BDS
  real(chm_real),intent(inout) :: SMIN,SMAX
  real(chm_real),intent(in) :: INITVAL
      
! if the boundaries are in ascending order, then we will assume that the user
! is trying to go from smaller to larger separations, and we will set 
! RXCMAX and RXCMIN accordingly, setting the first box boundaries to enclose the
! intial value of the reaction coordinate

  IF(ASC) THEN

!   what to do if the initval is smaller than the smallest boundary        
     IF(INITVAL.LE.BDS(1)) THEN
       if (prnlev > 2) then
          WRITE(OUTU,*) 'AXDINI> THE INITIAL VALUE OF THE RXN COORDINATE IS LESS THAN THE SMALLEST BOUNDARY'
       endif
       SMIN=BDS(1)
       SMAX=BDS(2)

!   what to do if the initval is larger than the largest boundary              
    ELSEIF(INITVAL.GE.BDS(NBOUN)) THEN
       if (prnlev > 2) then
          WRITE(OUTU,*) 'AXDINI> THE INITIAL VALUE OF THE RXN COORDINATE IS GREATER THAN THE LARGEST BOUNDARY'
       endif
       SMIN=BDS(NBOUN-1)
       SMAX=BDS(NBOUN)
          
!   what to do if the initval is an intermediate value
    ELSE      
      I=1
      DO WHILE(INITVAL.GT.BDS(I))
        SMIN=BDS(I)
        SMAX=BDS(I+1)
        I=I+1
      ENDDO
    ENDIF
        
  ENDIF

! if the boundaries are in descending order, then we will assume that the user
! is trying to go from larger to smaller separations, and we will set 
! RXCMAX and RXCMIN accordingly, setting the first box boundaries to enclose the
! intial value of the reaction coordinate

  IF(DES) THEN
      
!   what to do if the initval is greater than the largest boundary
     IF(INITVAL.GE.BDS(1)) THEN
        if (prnlev > 2) then
           WRITE(OUTU,*)'AXDINI> THE INITIAL VALUE OF THE RXN COORDINATE IS LARGER THAN THE LARGEST BOUNDARY'
        endif
        SMAX=BDS(1)
        SMIN=BDS(2)
          
!   what to do if the initval is smaller than the smallest boundary        
     ELSEIF(INITVAL.LE.BDS(NBOUN)) THEN
        if (prnlev > 2) then
           WRITE(OUTU,*)'AXDINI> THE INITIAL VALUE OF THE RXN COORDINATE IS SMALLER THAN THE SMALLEST BOUNDARY'
        endif
        SMAX=BDS(NBOUN-1)
        SMIN=BDS(NBOUN)

!   what to do if the initval is an intermediate value
    ELSE
      I=1
      DO WHILE(INITVAL.LT.BDS(I))
        SMAX=BDS(I)
        SMIN=BDS(I+1)
        I=I+1
      ENDDO
    ENDIF
      
  ENDIF

  if (prnlev > 2) then
     WRITE(OUTU,*)'AXDINI> BASED ON INITIAL VALUE OF REACTION COORDINATE, AXD BOUNDARIES SET AS FOLLOWS:'
     WRITE(OUTU,*)'AXDINI> ',SMAX,'IS THE MAX ALLOWED RXN COORD VALUE'
     WRITE(OUTU,*)'AXDINI> ',SMIN,'IS THE MIN ALLOWED RXN COORD VALUE'
  endif

  RETURN
      
END SUBROUTINE SET_INITIAL_BOX

 
SUBROUTINE BOUNMANAGER(BOUNDRYS,NBOUN,SMAX,SMIN,STP,EVNTS,IFLAG,INSMAX,INSMIN,ASC,DES)

!-----------------------------------------------------------------------
!
! This routine keeps track of the number of inversion events that have 
! occurred, and resets the inversion boundaries, RXCMAX and RXCMIN, 
! according to the value of EVENTS set in AXDINIT
! 
! NBOUN - number of boundaries
! BOUNDRYS - array holding the boundaries
! SMAX - RXCMAX
! SMIN - RXCMIN
! STP - integrator iteration
! EVNTS - number of events we want before resetting boundaries
! IFLAG - INVFLAG, it is true if an inversion is requested
! INSMAX - true if the inversion is wrt RXCMAX
! INSMIN - true if the inversion is wrt RXCMIN
! ASC - true if boundaries are in ascending order
! DES - true if boundaries are in descending order
!
! dave glowacki, 25 feb 09
!-----------------------------------------------------------------------

  use stream
  use chm_kinds
  implicit none

  LOGICAL,intent(inout) :: ASC,DES                 ! yw 12-Jul-2010
  LOGICAL,intent(inout) :: IFLAG,INSMAX,INSMIN
  INTEGER,intent(in) :: NBOUN,STP,EVNTS
  INTEGER,save :: IEVENTS,BSHIFTS
  INTEGER :: CTR,I
  real(chm_real),dimension(NBOUN), intent(inout) :: BOUNDRYS
  real(chm_real),dimension(NBOUN) :: BDSDUM
  real(chm_real),intent(inout) :: SMIN,SMAX

! in general, we need to keep track of the number of inversion events, the number of 
! times we change boundaries (BSHIFTS), and the number of times we change direction 
! (i.e., whether we are moving through the boundaries in an ascending order or a 
! descending order)

! what to do on the first call 
  IF(STP.EQ.1) THEN

! set IEVENTS to zero
    IEVENTS=0
        
! now we have to set BSHIFTS, the number of box boundary crossings thus far.
! in the initial iteration of this code, BXD would only turn on when we were in 
! box #1, but in using the code, we decided that we we wanted it to be flexible enough 
! so that BXD will 'turn on' based on the initial value of the reaction coordinate, 
! and place it in the correct initial starting box. however, to do this, then we have
! to do some bookkeeping to calculate BSHIFT, and that's what's included below:
  
!   if bounds are in ascending order and the initial box is the smallest box, we're starting
!   from scratch (this is all that was initially available)
    IF(ASC.AND.SMIN.EQ.BOUNDRYS(1).AND.SMAX.EQ.BOUNDRYS(2)) THEN
      BSHIFTS=0

!   if bounds are in descending order and the initial box is the biggest box, we're starting
!   from scratch (this is all that was initially available)
    ELSEIF(DES.AND.SMAX.EQ.BOUNDRYS(1).AND.SMIN.EQ.BOUNDRYS(2)) THEN
      BSHIFTS=0

!   if bounds are in ascending order and the initial box is the biggest box, it's like we 
!   already crossed all the boundaries and arrived at the last box, so bshifts is maximum
    ELSEIF(ASC.AND.SMIN.EQ.BOUNDRYS(NBOUN-1).AND.SMAX.EQ.BOUNDRYS(NBOUN))THEN
      BSHIFTS=NBOUN-2

!   if bounds are in descending order and we're starting in the smallest box, it's like we 
!   already crossed all the boundaries and arrived at the last box, so bshifts is maximum
    ELSEIF(DES.AND.SMAX.EQ.BOUNDRYS(NBOUN-1).AND.SMIN.EQ.BOUNDRYS(NBOUN))THEN
      BSHIFTS=NBOUN-2

!   if bounds are in ascending order and we're starting in an intermediate box, we need to
!   calculate the number of 'apparent' bshifts we underwent to get there
    ELSEIF(ASC)THEN
      I=1
      BSHIFTS=0
      DO WHILE(SMIN.NE.BOUNDRYS(I))
        I=I+1
        BSHIFTS=BSHIFTS+1
      ENDDO

!   if it's descending order and we're starting in an intermediate box, we need to
!   calculate the number of 'apparent' bshifts we underwent to get there
    ELSEIF(DES)THEN
      I=1
      BSHIFTS=0
      DO WHILE(SMAX.NE.BOUNDRYS(I))
        I=I+1
        BSHIFTS=BSHIFTS+1
      ENDDO
          
    ENDIF

  ENDIF

! be sure that the number of shifts isnt greater than is possible with the
! number of boundaries: in general for N boundaries, there will be N-2
! shifts that are possible.
! if IFLAG is true, then an inversion is requested, and we need to keep
! track of how many total inversion events have been performed

  IF(BSHIFTS.LT.(NBOUN-2))THEN
    IF(IFLAG) IEVENTS=IEVENTS + 1
      
! if IEVENTS > EVENTS, boundaries are in descending order, and the next inversion is wrt RXCMIN
!                              or
! if IEVENTS > EVENTS, boundaries are in ascending order, and the next inversion is wrt RXCMAX
!                             then
! we change the boundaries, increment the shift counter, reset IEVENTS, 
! turn off INVFLAG (so that we dont invert), and then turn off either 
! INSMIN or INSMAX (so that we dont do an inappropriate print)
      
    IF(IEVENTS.GT.EVNTS.AND.DES.AND.INSMIN)THEN
      BSHIFTS=BSHIFTS+1
      IFLAG=.FALSE.
      INSMIN=.FALSE.
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXD> THERE HAVE BEEN',IEVENTS,'INVERSION EVENTS BETWEEN BOX BOUNDARIES', &
              SMIN,' AND',SMAX
         WRITE(OUTU,*)'AXD> THE AXD BOX BOUNDARIES ARE BEING UPDATED:'
      endif
      SMIN=BOUNDRYS(BSHIFTS+2)
      SMAX=BOUNDRYS(BSHIFTS+1)
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXD> ',SMAX,'IS THE MAX ALLOWED RXN COORD VALUE'
         WRITE(OUTU,*)'AXD> ',SMIN,'IS THE MIN ALLOWED RXN COORD VALUE'
      endif
      IEVENTS=0

    ELSEIF(IEVENTS.GT.EVNTS.AND.ASC.AND.INSMAX)THEN
      BSHIFTS=BSHIFTS+1
      IFLAG=.FALSE.
      INSMAX=.FALSE.
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXD> THERE HAVE BEEN',IEVENTS,'INVERSION EVENTS BETWEEN BOX BOUNDARIES', &
              SMIN,'AND',SMAX
         WRITE(OUTU,*)'AXD> THE AXD BOX BOUNDARIES ARE BEING UPDATED:'
      endif
      SMAX=BOUNDRYS(BSHIFTS+2)
      SMIN=BOUNDRYS(BSHIFTS+1)
      if (prnlev > 2) then
         WRITE(OUTU,*)'AXD> ',SMAX,'IS THE MAX ALLOWED RXN COORD VALUE'
         WRITE(OUTU,*)'AXD> ',SMIN,'IS THE MIN ALLOWED RXN COORD VALUE'
      endif
      IEVENTS=0
    ENDIF
! what to do if we're in the last box
  ELSEIF(BSHIFTS.EQ.(NBOUN-2))THEN
    IF(IFLAG) IEVENTS=IEVENTS + 1
!   if number of inversion events is exceeded, restart the box dynamics, going in the opposite direction
    IF(IEVENTS.GT.EVNTS)THEN
!   reverse the order of the bounds in the BOUNDRYS vector
      DO CTR=1,NBOUN 
        BDSDUM(NBOUN+1-CTR)=BOUNDRYS(CTR)
      ENDDO
      DO CTR=1,NBOUN 
        BOUNDRYS(CTR)=BDSDUM(CTR)
      ENDDO
!     set ASC and DES and set values of SMAX and SMIN
      CALL CHECKANDSETBOXES(BOUNDRYS,NBOUN,SMAX,SMIN,ASC,DES)
!     reset BSHIFTS and IEVENTS
      BSHIFTS=0
      IEVENTS=0
    ENDIF 
  ENDIF

  RETURN

END SUBROUTINE BOUNMANAGER

SUBROUTINE INVERTDIST(CA,CB,VA,VB,MA,MB)
!-----------------------------------------------------------------------
!     CA,CB - x,y,z Coordinates for atoms A and B
!     VA,VB - x,y,z Velocities for atoms A and B
!     MA,MB - masses of atoms A and B
!
! This routine is for inverting the velocities of two atoms in their 
! center of mass (CM) frame.  See the comments below for more specific 
! details on how the inversion is performed.  dave glowacki, 25 feb 09
!-----------------------------------------------------------------------

  use chm_kinds
  implicit none

  INTEGER :: CTR
  real(chm_real),dimension(3), intent(inout) :: CA,CB,VA,VB
  real(chm_real),intent(in) :: MA,MB
  real(chm_real) :: NORM 
  real(chm_real),dimension(3) :: N12,VACM,VBCM,VCM,VAPROPL,VBPROPL,VAPROPR,VBPROPR

! davidglo: error checking
!        write(*,*)'INVERTDIST- preinvert' 
!        write(*,*)'VA(1) - VA(3); VB(1) - VB(3)'
!        write(*,*)VA(1),VA(2),VA(3),VB(1),VB(2),VB(3)
! end error checking

! invert the velocities 

! straight inversion in cartesian frame (only for testing)
!      DO CTR=1, 3
!         VA(CTR)=VA(CTR)-VA(CTR)-VA(CTR)
!         VB(CTR)=VB(CTR)-VB(CTR)-VB(CTR)                                     
!      ENDDO
! end straight cartesian inversion

! what follows is for inversion in CM frame
! calculate the velocity of the center of mass (CM)

  DO CTR=1, 3
    VCM(CTR)=(MA*VA(CTR)+MB*VB(CTR))/(MA+MB)                                     
  ENDDO

! calculate velocities of A and B in the CM frame

  DO CTR=1, 3
    VACM(CTR)=VA(CTR)-VCM(CTR)
    VBCM(CTR)=VB(CTR)-VCM(CTR)
  ENDDO

! calculate the unit vector, N12 pointing from A to B

  NORM=SQRT((CA(1)-CB(1))*(CA(1)-CB(1))+(CA(2)-CB(2))*(CA(2)-CB(2))+(CA(3)-CB(3))*(CA(3)-CB(3)))

  DO CTR=1, 3
    N12(CTR)=(CA(CTR)-CB(CTR))/NORM
  ENDDO

! calculate the parallel projection of A & B (in the CM frame) on N12

!  note: VACM x N12 =(VACM(1)*N12(1)+VACM(2)*N12(2)+VACM(3)*N12(3))
!  note: VBCM x N12 =(VBCM(1)*N12(1)+VBCM(2)*N12(2)+VBCM(3)*N12(3)) 

  DO CTR=1, 3
    VAPROPL(CTR)=N12(CTR)*(VACM(1)*N12(1)+VACM(2)*N12(2)+VACM(3)*N12(3))
    VBPROPL(CTR)=N12(CTR)*(VBCM(1)*N12(1)+VBCM(2)*N12(2)+VBCM(3)*N12(3)) 
  ENDDO

! The commented lines below calculate the perpendicular projection of A & B 
! in the CM frame so that we can invert BOTH the perpendicular AND parallel 
! projections of A and B in the CM frame.  This is what Emilio did previously
! (see JCTC, 2006(2), 912-919).  But in my tests, inverting only the parallel
! projections in the CM frame works better for the following reasons:
!  1) perturbs the trajectories less 
!  2) gives better conservation of energy
!  3) conserves the total angular momentum of the CM; whereas inverting BOTH 
!     parallel AND perpendicular projections inverts the CM angular momentum.  
!     dave glowacki, 19 feb 09

!      DO CTR=1, 3
!        VAPROPR(CTR)=VACM(CTR)-VAPROPL(CTR)
!        VBPROPR(CTR)=VBCM(CTR)-VBPROPL(CTR) 
!      ENDDO

!      DO CTR=1, 3
!        VACM(CTR)=0.0D+0-VAPROPR(CTR)-VAPROPL(CTR)
!        VBCM(CTR)=0.0D+0-VBPROPR(CTR)-VBPROPL(CTR) 
!      ENDDO

! invert the velocities VACM and VBCM in the CM frame 

  DO CTR=1, 3
    VACM(CTR)=VACM(CTR)-VAPROPL(CTR)-VAPROPL(CTR)
    VBCM(CTR)=VBCM(CTR)-VBPROPL(CTR)-VBPROPL(CTR) 
  ENDDO

! transform to the cartesian lab frame

  DO CTR=1, 3
    VA(CTR)=VACM(CTR)+VCM(CTR)
    VB(CTR)=VBCM(CTR)+VCM(CTR) 
  ENDDO          

! davidglo: error checking
!        write(*,*)'INVERTDIST post-invert:'
!        write(*,*)'VA(1) - VA(3); VB(1) - VB(3)'
!        write(*,*)VA(1),VA(2),VA(3),VB(1),VB(2),VB(3)
! end error checking

  RETURN

END SUBROUTINE INVERTDIST

SUBROUTINE AXDPRINT(PALL,IN,OUT,STP,RHO,SMIN,SMAX,NPR,JUJ,PRG,PMN,PMX)
!-----------------------------------------------------------------------
!     PALL - flag to print at every step
!     IN - flag that is true if inversion wrt RXCMAX ooccured
!     OUT - flag that is true if inversion wrt RXCMIN ooccured
!     STP - numerical integrator iteration
!     RHO - RHO at time T
!     SMIN & SMAX - see above code for specifications
!     NPR - frequency of prints in main output
!     JUJ - axd output file
!     PRG - flag that is true if a specified print range is on
!     PMN - PMIN specified in AXDINIT
!     PMX - PMAX specified in AXDINIT
!
! This routine is for printing to the AXD output file. 
! dave glowacki, 25 feb 09
!-----------------------------------------------------------------------

  use reawri
  use chm_kinds
  implicit none

  LOGICAL, intent(in) :: PALL,IN,OUT
  LOGICAL :: PRG
  INTEGER,intent(in) :: STP,NPR,JUJ
  real(chm_real),intent(in) :: RHO,SMIN,SMAX,PMN,PMX

! what to do if print all is on
  IF(PALL) THEN
    IF(IN) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMAX
    ELSEIF(OUT) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMIN        
    ELSE
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO
    ENDIF
! if print all is not on, prange is on and, pmin<rho<pmax, print
  ELSEIF(PRG.AND.RHO.LT.PMX.AND.RHO.GT.PMN) THEN
    IF(IN) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMAX
    ELSEIF(OUT) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMIN        
    ELSE
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO
    ENDIF
! if pall isnt on and the prange conditions arent satisfied, print at 
! the standard print freqency and when inversion occurs
  ELSE
    IF(IN) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMAX
    ELSEIF(OUT) THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO,SMIN        
    ELSEIF(MOD(STP,NPR).EQ.0)THEN
      WRITE(JUJ,'(I9,4(1X,F14.6))')STP,RHO
    ENDIF
  ENDIF
   
  RETURN

END SUBROUTINE AXDPRINT

#endif /*       (axd_outer)  */

!
! end of the axd_module
!===============================================================
end module axd_module


