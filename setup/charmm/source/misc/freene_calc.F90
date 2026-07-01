! A general purpose calculator for various free energy differences
! using BAR, NBB, and the Zwanzig equation. Also includes a feature
! to create dummy atoms at the parameter file level and a feature
! to adapt some of the bonded terms to a target structure. 
!
! Initial version by Gerhard Koenig, somewhat redesigned
! for inclusion into CHARMM by Tim Miller.
!
! anno domini MMXV in festo beate angele de fulgineo hoc opus est inceptum per Gerhardum Koenigem,
! regnante in laboratorio Bernardo R. Brookso.
!
! Questions?  Send an email to gerhard@mdy.univie.ac.at

module freeene_calc
  use chm_kinds
  implicit none
  ! Some basic definitions:
  ! 0 denotes the inital state of the free energy calculation
  ! 1 denotes the final state of the free energy calculation
  ! U_0 is the potential energy of the inital state 0
  ! U_1 is the potential energy of the final state 1 
  ! (0) means that coordinates from a trajectory of state 0 are employed
  ! dU are potential energy differences (dU0=U_1(0)-U_0(0), dU1=U_0(1)-U_1(1))

  ! For reweighting:
  ! S0 is a sampling state that is employed to explore the conformational space of target state 0 
  ! S1 is a sampling state that is employed to explore the conformational space of target state 1 
  ! U_S0 is the potential energy of the sampling state of 0 
  ! U_S1 is the potential energy of the sampling state of 1   
  ! Vb are the biasing potentials (Vb0=U_S0(S0)-U_0(S0), Vb1=US_1(S1)-U_1(S1))
  ! note that dU is now (dU0=U_1(S0)-U_0(S0), dU1=U_0(S1)-U_1(S1)) since we are not directly sampling state 0 and 1 
  ! Even more confused now? Read: Koenig G, Boresch S. J. Comput. Chem. 32:1082-1090 (2011) and/or send me an email 

  type free_ene_diff                                 
     integer                                        :: n0  !! number of (effective) data points for trajectory 0 ("forward perturbation")  - excluding outliers
     integer                                        :: n1  !! number of (effective) data points for trajectory 1 ("backward perturbation") - excluding outliers
     real(chm_real),allocatable,dimension(:)        :: dU0 !! Potential energy difference dU0 = U_1 - U_0 for trajectory 0 ("forward delta U")
     real(chm_real),allocatable,dimension(:)        :: dU1 !! Potential energy difference dU1 = U_0 - U_1 for trajectory 1 ("backward delta U")
     real(chm_real),allocatable,dimension(:)        :: Vb0 !! forward bias
     real(chm_real),allocatable,dimension(:)        :: Vb1 !! backward bias
     real(chm_real),allocatable,dimension(:)        :: U00,U01,U10,U11,US0,US1 !! All possible potential energies involved
     integer                                        :: ndU0,ndU1,nVb0,nVb1,nU00,nU01,nU10,nU11,nUS0,nUS1 ! number of read data points  
     logical                                        :: q_bias0,q_bias1 !! whether there is bias data for the initial (0) or end (1) state
     integer                                        :: l_order !! reserved for future use
     real(chm_real)                                 :: l_start,l_stop !! reserved for future use
  end type free_ene_diff

  ! ToDo: we eventually want to have multiple free_ene_diff types
  ! to allow for lambda intermediate points.
  ! So, this will become an allocatable array and the user will
  ! pass "NLAM x" as an argument to FREN and this array will get
  ! dynamically allocated.
  type(free_ene_diff) :: dA

  ! make some stuff public and private

contains
! --------------------------------------------------------------------------------------------------------------------------------------------------
    !
    !     This routine replaces the equilibrium bond and angle terms in the parameter file by the current coordinates
    !     (mainly intended to reproduce QM structures)
    !
    !            By Bernard R. Brooks and Gerhard Koenig, 2015
    !
    subroutine qmfix(comlyn,comlen)
      use chm_kinds
      use chm_types
      use bases_fcm
      use code
      use consta,only: degrad
      use coord
      use dimens_fcm
      use deriv
      use energym
      use intcor2
      use memory
      use number
      use param
      use psf
      use select
      use stream
      use string
      use vector
      implicit none
      character,intent(inout) :: comlyn*(*)
      integer,intent(inout)   :: comlen

      logical lbonds ! fix bonds
      logical langles  ! fix angles
      logical lnoupdate ! Don't do an update before accessing psf/parameter data   
      logical lmini    ! Minimize equilibrium bond length/angle based on forces 
      logical lnoor    ! No orthogonalization in minimization 
      integer nsel     ! Number of selected atoms 
      integer oldncb, oldnct   ! Old number of bond/angle types 
      integer tmpi     ! Temperorary integer variable 
      integer,allocatable,dimension(:) :: ISLCT ! atom selection 
      real(chm_real) atmdu(3), normatmdu(3)  ! du/dx,y,z of current atom  + normalized vector
      real(chm_real) wilson(9), SCR2(9), SCR3(9)           ! scratch 
      real(chm_real),allocatable,dimension(:) :: CTCTMP ! temporary storage for angle force constants
      real(chm_real),allocatable,dimension(:) :: CTUCTMP ! temporary storage for Urey-Bradley force constants
      real(chm_real) :: distance, myangle       ! current distance/angle from coordinates
      real(chm_real) :: normforce, oldnormforce ! 1-Norm of the force vectors (To check convergence)
      real(chm_real) :: fdrop, expfdrop         ! Drop of 1-Norm of force and expected drop of 1-Norm (based on NRAP)
      real(chm_real) :: rfdrop,oldrfdrop        ! ratio of drop and expected drop 
      real(chm_real) :: fconvergence      ! Convergence criterion for force-norm 
      real(chm_real) :: proj      ! projection of force
      real(chm_real) :: psum      ! sum of projections on force 
      real(chm_real) :: stepsize  ! step size for minimization 
      real(chm_real) :: shift     ! shift of parameter to generate counterforce 
      real(chm_real) :: bla       ! temporary variable 
      integer  iteration, maxiter ! if multiple iterations are requested 
      integer  i,j,ic

      bla = zero 

      lbonds=(INDXA(COMLYN,COMLEN,'BOND').GT.0)
      langles=(INDXA(COMLYN,COMLEN,'ANGL').GT.0)
      lnoupdate=(INDXA(COMLYN,COMLEN,'NOUP').GT.0)
      lmini=(INDXA(COMLYN,COMLEN,'MINI').GT.0)
      lnoor=(INDXA(COMLYN,COMLEN,'NOOR').GT.0)
      maxiter=gtrmi(comlyn,comlen,'MAXI',10000)  
      fconvergence = gtrmf(comlyn, comlen, 'CONV',1.0d-5)
      stepsize = gtrmf(comlyn, comlen, 'STEP', one)

      if (.not. lbonds .and. .not. langles) then
         lbonds = .true.
         langles = .true.
      endif

      ! Do an update to make sure that all required arrays are filled 
      if(.NOT. lnoupdate) then 
         IF(PRNLEV >= 3) WRITE(OUTU, *)  "QMFIX> PERFORMING UPDATE"
         call update(comlyn,comlen,x,y,z,wmain,.true.,.true.,.true., &
             .false.,.true.,0,[zero],[zero],[zero],[zero],[zero],[zero])
      endif

      call chmalloc('freeene_calc.src','QMFIX','ISLCT',natom,intg=ISLCT)   
      islct(1:natom) = 1 
      CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
      nsel = SUM(ISLCT(1:natom))
      
      if (lbonds .and. ncb+nsel>maxcb) then 
         call wrndie(-1,'<QMFIX>','NUMBER OF NEW BOND TYPES EXCEEDS MAXCB')
      endif
      
      if (langles .and. nct+nsel>maxct) then 
         call wrndie(-1,'<QMFIX>','NUMBER OF NEW ANGLE TYPES EXCEEDS MAXCT')
      endif

      if (.not. lmini .and. lbonds) then
         IF(PRNLEV >= 3) WRITE(OUTU, *)  "QMFIX> ADJUSTING ALL SELECTED BOND PARAMETERS TO CURRENT BOND LENGTHS"
         oldncb = ncb 
         j = 1
         do i = 1,nbond
            if(ISLCT(IB(I)) .eq. 0 .or. ISLCT(JB(I)) .eq. 0) cycle  ! Skip bond if atoms are not selected 
            ! Get pointer for bond force constant parameters and set new pointer
            ic = icb(i)
            icb(i) = oldncb+j
            ! Obtain distance (dist)
            distance = 99999.99     ! very wrong initial value to make errors very obvious
            CALL GETICV(IB(I),JB(I),0,0,.FALSE.,distance,bla,bla,bla,bla,X,Y,Z)
            ! Replace equilibrium bond distance by current distance
            cbb(oldncb+j) =  distance
            ! Replace force constant
            cbc(icb(i)) = cbc(ic)
            ncb=ncb+1
            j = j+1 
         enddo
       endif

      if (.not. lmini .and. langles) then
         IF(PRNLEV >= 3) WRITE(OUTU, *)  "QMFIX> ADJUSTING ALL SELECTED ANGLE PARAMETERS TO CURRENT BOND ANGLES"
         oldnct = nct 
         j=1 
         do i = 1,ntheta
            if(ISLCT(IT(I)) .eq. 0 .or. ISLCT(JT(I)) .eq. 0 .or. ISLCT(KT(I)) .eq. 0 ) cycle  ! Skip angle if atoms are not selected 
            ! Get pointer for bond force constant parameters and set new pointer
            ic = ict(i)
            ict(i) = oldnct+j
            ! Obtain angle
            myangle = 0.0         ! very wrong initial value to make errors very obvious
            CALL GETICV(IT(I),JT(I),KT(I),0,.FALSE.,bla,myangle,bla,bla,bla,X,Y,Z)
            ! Convert angle to radians
            myangle=myangle*degrad
            ! Replace equilibrium angle by current angle
            ctb(ict(i)) = myangle
            ! Take old force constant
            ctc(ict(i)) = ctc(ic)  

            ! And now the Urey-Bradley term
            ! Note by GK: Technically, it might be necessary to change the force constant of the
            ! Urey-Bradley term as you change the equilibrium angle. However, I am not sure how
            ! to do that correctly, as this matter is not documented, so I just copy the old value.
            distance = 99999.99     ! very wrong initial value to make errors very obvious
            CALL GETICV(IT(I),KT(I),0,0,.FALSE.,distance,bla,bla,bla,bla,X,Y,Z)
            ! Change Urey-Bradley equilibrium value
            ctub(ict(i)) = distance
            ctuc(ict(i)) = ctuc(ic)
            nct=nct+1
            j = j+1 
         enddo
      endif
     
      if (lmini) then 
         normforce = zero 
         oldnormforce = zero 
         fdrop = one
         rfdrop = one
         do iteration=1,maxiter 
            ! If a minimization was requested, calculate energy/forces 
            IF(PRNLEV >= 6) WRITE(OUTU, *)  "QMFIX> RECALCULATING ENERGY AND FORCES FOR STEP ", iteration 
            call energy(x,y,z,dx,dy,dz,bnbnd,bimag,0)
            oldnormforce = normforce 
            normforce = SUM(ABS(DX(1:natom))) + SUM(ABS(DY(1:natom))) + SUM(ABS(DZ(1:natom)))

            ! Check convergence 
            fdrop = normforce-oldnormforce
            oldrfdrop=rfdrop
            if(iteration .gt. 1 ) rfdrop = fdrop/expfdrop 
            if(iteration .gt. 1 .and. PRNLEV >= 6) WRITE(OUTU, *)  "QMFIX> NORM OF FORCE ARRAYS IS ",&
                 & real(normforce) , " AFTER ITERATION  ", iteration , " DROP ", real(fdrop), &
                 & "EFFICIENCY ", real(rfdrop), "STEP SIZE ", real(stepsize)
            ! If the convergence criterion is met, or the forces are negligible, we can stop 
            if (abs(fdrop) .le. fconvergence ) exit 
            if (abs(normforce) .le. fconvergence ) exit 

            ! Go through all atoms and check whether they are selected 
            do i=1,natom
               if ( ISLCT(i) .ne. 1) cycle 
               ! Save forces and normalized forces 
               ATMDU(1)=DX(i)
               ATMDU(2)=DY(i)
               ATMDU(3)=DZ(i)
               NORMATMDU(1:3)=ATMDU(1:3)
               call normall(NORMATMDU(1:3),3) ! Normalize   
               psum = zero 
               ! See which bonds are associated with the atom 
               if (lbonds) then
                  do j=1,nbond 
                     if(IB(j) .ne. i .and. JB(j) .ne. i) cycle  ! Skip bond if atom i is not involved
                     ! Generate Wilson vectors for bond 
                     WILSON(1) = X(IB(j)) - X(JB(j))
                     WILSON(2) = Y(IB(j)) - Y(JB(j))
                     WILSON(3) = Z(IB(j)) - Z(JB(j))                     
                     if(JB(j) .eq. i) WILSON(1:3)=-WILSON(1:3)
                     ! Check bond length (if it is too short die) 
                     distance = SQRT(SUM(WILSON(1:3)**2))
                     if(distance .lt. 0.00001 .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                          &"QMFIX> WARNING: BOND BETWEEN ATOMS ", IB(j), " AND " , JB(j), &
                          &" IS VERY SHORT (", real(distance),"ANG)"
                     if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC BOND')
                     WILSON(1:3)=  WILSON(1:3)/distance ! Normalize 
                     proj=dot_product(NORMATMDU(1:3),WILSON(1:3))
                     psum=psum+abs(proj) 
                  enddo ! j 
               endif ! lbonds
               
               if (langles) then
                  do j = 1,ntheta
                     ! Skip angle if atom is not part of it  
                     if(IT(j) .ne. i .and. JT(j) .ne. i .or. KT(j) .ne. i ) cycle  
                     ! Get pointer for angle parameters
                     ic = ict(j)
                     
                     ! Generate Wilson vectors for angle 
                     ! Calculate normal vector for plane defined by the three atoms
                     ! Vector for first bond between atom 2 and 1 
                     WILSON(1)=X(IT(j))-X(JT(j))
                     WILSON(2)=Y(IT(j))-Y(JT(j))
                     WILSON(3)=Z(IT(j))-Z(JT(j))
                     distance = SQRT(SUM(WILSON(1:3)**2))
                     if(distance .lt. 0.00001  .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                          &"QMFIX> WARNING: BOND BETWEEN ATOMS ", IT(j), " AND " , JT(j), &
                          &" IS VERY SHORT (", real(distance),"ANG)"
                     if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE')
                     WILSON(1:3)=WILSON(1:3)/distance !Normalize 
                     ! Vector for second bond between atom 2 and 3
                     WILSON(7)=X(KT(j))-X(JT(j))
                     WILSON(8)=Y(KT(j))-Y(JT(j))
                     WILSON(9)=Z(KT(j))-Z(JT(j))
                     distance = SQRT(SUM(WILSON(7:9)**2))
                     if(distance .lt. 0.00001 .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                          &"QMFIX> WARNING: BOND BETWEEN ATOMS ", JT(j), " AND " , KT(j), &
                          &" IS VERY SHORT (", real(distance),"ANG)"
                     if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE') 
                     WILSON(7:9)=WILSON(7:9)/distance !Normalize 
                     
                     ! Terminal atoms of angle
                     if(IT(j) .eq. i .or. KT(j) .eq. i) then  
                        ! Now we store the normal vector of the plane
                        WILSON(4:6) = crossproduct3(WILSON(1:3),WILSON(7:9))
                        distance = SQRT(SUM(WILSON(4:6)**2))
                        if(distance .lt. 0.00001 .and. PRNLEV >= 5 ) WRITE(OUTU, *)  &
                             &"QMFIX> SKIPPING LINEAR ANGLE BETWEEN ATOMS ", IT(i), JT(i), KT(i) 
                        if(distance .lt. 0.00001) CYCLE 
                        call normall(WILSON(4:6),3) 
                        ! The tangential vectors can be obtained from the crossproducts
                        ! between the bond_21 and the normal vector of the plane of 123
                        WILSON(1:3) = crossproduct3(WILSON(1:3),WILSON(4:6))
                        ! between the bond_23 and the normal vector of the plane of 123
                        WILSON(7:9) = crossproduct3(WILSON(4:6),WILSON(7:9))
                        ! Normalize 
                        call normall(WILSON(1:3),3)
                        call normall(WILSON(7:9),3) 
                        if(IT(j) .eq. i) proj=dot_product(NORMATMDU(1:3),WILSON(1:3))
                        if(KT(j) .eq. i) proj=dot_product(NORMATMDU(1:3),WILSON(7:9))
                        psum=psum+abs(proj) 
                     endif
                     ! Apex atom 
                     if(jT(j) .eq. i) then  
                        ! The vector of the apex atom is given by the two other vectors 
                        WILSON(4:6) = -WILSON(1:3)-WILSON(7:9)
                        call normall(WILSON(4:6),3) 
                        proj=dot_product(NORMATMDU(1:3),WILSON(4:6))
                        psum=psum+abs(proj) 
                     endif
                                         
                     ! Urey-Bradley terms 
                     if (abs(ctuc(ic)) .gt. 0.00001) then 
                        ! Generate Wilson vectors for Urey Bradley  
                        WILSON(1) = X(IT(j)) - X(KT(j))
                        WILSON(2) = Y(IT(j)) - Y(KT(j))
                        WILSON(3) = Z(IT(j)) - Z(KT(j))                     
                        if(KT(j) .eq. i) WILSON(1:3)=-WILSON(1:3)
                        ! Check UB length (if it is too short die) 
                        distance = SQRT(SUM(WILSON(1:3)**2))
                        if(distance .lt. 0.00001 .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                             &"QMFIX> WARNING: UREY-BRADLEY BETWEEN ATOMS ", IT(j), " AND " , KT(j), &
                             &" IS VERY SHORT (", real(distance),"ANG)"
                        if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE')
                        WILSON(1:3)=  WILSON(1:3)/distance ! Normalize 
                        proj=dot_product(NORMATMDU(1:3),WILSON(1:3))
                        psum=psum+abs(proj) 
                     endif
                  enddo ! j
               endif !langle 
               


               ! Now that we have determined how the force has to be distributed, 
               ! we change all associated degrees of freedom 
               if (lbonds) then
                  do j=1,nbond 
                     if(IB(j) .ne. i .and. JB(j) .ne. i) cycle  ! Skip bond if atom i is not involved
                     ! Get pointer for bond force constant parameters 
                     ic = icb(j)
                     ! Generate Wilson vectors for bond 
                     WILSON(1) = X(IB(j)) - X(JB(j))
                     WILSON(2) = Y(IB(j)) - Y(JB(j))
                     WILSON(3) = Z(IB(j)) - Z(JB(j))                     
                     if(JB(j) .eq. i) WILSON(1:3)=-WILSON(1:3)
                     call normall(WILSON(1:3),3) ! Normalize   
                     ! Project forces into bond Wilson vector 
                     ! and divide by psum (every associated dof gets a fair share of the force)
                     proj=dot_product(ATMDU(1:3),WILSON(1:3))/psum 
                     ! Save expected drop of total force 
                     expfdrop = expfdrop - abs(proj*stepsize)
                     ! divide projected dU/dr by the force constant to determine the shift  ( dU/dr=2*k*r)
                     ! CHARMM force constant needs a factor of 2 
                     ! Since we only did this for one of the two atoms of the bond, divide shift by two 
                     shift =  stepsize*proj/(TWO*CBC(ic))/TWO   
                     ! Since we only did this for one of the a
                     IF(PRNLEV >= 7) WRITE(OUTU, *)  "QMFIX> Adjusting bond ", IB(j), JB(j), " by " , &
                          & real(shift) , " ANG PROJ", real(proj), "CBB:", real(CBB(ic)), "CBC:", &
                          & real(CBC(ic))   
                     ! Shift equilibrium bond distance to neutralize the forces 
                     CBB(ic) = CBB(ic) + shift   
                  enddo ! j 
               endif ! lbonds

               if (langles) then
                  do j = 1,ntheta
                     ! Skip angle if atom is not part of it  
                     if(IT(j) .ne. i .and. JT(j) .ne. i .or. KT(j) .ne. i ) cycle  
                     ! Get pointer for angle parameters
                     ic = ict(j)
                     
                     ! Generate Wilson vectors for angle 
                     ! Calculate normal vector for plane defined by the three atoms
                     ! Vector for first bond between atom 2 and 1 
                     WILSON(1)=X(IT(j))-X(JT(j))
                     WILSON(2)=Y(IT(j))-Y(JT(j))
                     WILSON(3)=Z(IT(j))-Z(JT(j))
                     distance = SQRT(SUM(WILSON(1:3)**2))
                     if(distance .lt. 0.00001  .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                          &"QMFIX> WARNING: BOND BETWEEN ATOMS ", IT(j), " AND " , JT(j), &
                          &" IS VERY SHORT (", real(distance),"ANG)"
                     if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE')
                     WILSON(1:3)=WILSON(1:3)/distance !Normalize 
                     ! Vector for second bond between atom 2 and 3
                     WILSON(7)=X(KT(j))-X(JT(j))
                     WILSON(8)=Y(KT(j))-Y(JT(j))
                     WILSON(9)=Z(KT(j))-Z(JT(j))
                     distance = SQRT(SUM(WILSON(7:9)**2))
                     if(distance .lt. 0.00001 .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                          &"QMFIX> WARNING: BOND BETWEEN ATOMS ", JT(j), " AND " , KT(j), &
                          &" IS VERY SHORT (", real(distance),"ANG)"
                     if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE') 
                     WILSON(7:9)=WILSON(7:9)/distance !Normalize 
                     
                     ! Terminal atoms of angle
                     if(IT(j) .eq. i .or. KT(j) .eq. i) then  
                        ! Now we store the normal vector of the plane
                        WILSON(4:6) = crossproduct3(WILSON(1:3),WILSON(7:9))
                        distance = SQRT(SUM(WILSON(4:6)**2))
                        if(distance .lt. 0.00001 .and. PRNLEV >= 5 ) WRITE(OUTU, *)  &
                             &"QMFIX> SKIPPING LINEAR ANGLE BETWEEN ATOMS ", IT(i), JT(i), KT(i) 
                        if(distance .lt. 0.00001) CYCLE 
                        call normall(WILSON(4:6),3) 
                        ! The tangential vectors can be obtained from the crossproducts
                        ! between the bond_21 and the normal vector of the plane of 123
                        WILSON(1:3) = crossproduct3(WILSON(1:3),WILSON(4:6))
                        ! between the bond_23 and the normal vector of the plane of 123
                        WILSON(7:9) = crossproduct3(WILSON(4:6),WILSON(7:9))
                        ! Normalize 
                        call normall(WILSON(1:3),3)
                        call normall(WILSON(7:9),3) 
                        if(IT(j) .eq. i) proj=dot_product(NORMATMDU(1:3),WILSON(1:3))
                        if(KT(j) .eq. i) proj=dot_product(NORMATMDU(1:3),WILSON(7:9))
                        psum=psum+abs(proj) 
                     endif
                     ! Apex atom 
                     if(jT(j) .eq. i) then  
                        ! The vector of the apex atom is given by the two other vectors 
                        WILSON(4:6) = -WILSON(1:3)-WILSON(7:9)
                        call normall(WILSON(4:6),3) 
                        proj=dot_product(NORMATMDU(1:3),WILSON(4:6))
                        psum=psum+abs(proj) 
                     endif
                                         
                     ! Urey-Bradley terms 
                     if (abs(ctuc(ic)) .gt. 0.00001) then 
                        ! Generate Wilson vectors for Urey Bradley  
                        WILSON(1) = X(IT(j)) - X(KT(j))
                        WILSON(2) = Y(IT(j)) - Y(KT(j))
                        WILSON(3) = Z(IT(j)) - Z(KT(j))                     
                        if(KT(j) .eq. i) WILSON(1:3)=-WILSON(1:3)
                        ! Check UB length (if it is too short die) 
                        distance = SQRT(SUM(WILSON(1:3)**2))
                        if(distance .lt. 0.00001 .and. PRNLEV >= 3 ) WRITE(OUTU, *)  &
                             &"QMFIX> WARNING: UREY-BRADLEY BETWEEN ATOMS ", IT(j), " AND " , KT(j), &
                             &" IS VERY SHORT (", real(distance),"ANG)"
                        if(distance .lt. 0.00001) call wrndie(-1,'<QMFIX>','PROBLEMATIC ANGLE')
                        WILSON(1:3)=  WILSON(1:3)/distance ! Normalize 
                        proj=dot_product(NORMATMDU(1:3),WILSON(1:3))
                        psum=psum+abs(proj) 
                     endif
                  enddo ! j
               endif !langle 
            enddo ! i 
         enddo !  iteration 
      endif ! lmini

      IF(PRNLEV >= 3) WRITE(OUTU, *)  "QMFIX> REMINDER: DO NOT CHANGE PSF OR ALL CHANGES WILL BE LOST"
      call chmdealloc('freeene_calc.src','QMFIX','ISLCT',natom,intg=ISLCT)   
      RETURN
  END SUBROUTINE QMFIX
  
! --------------------------------------------------------------------------------------------------------------------------------------------------
    !
    !     This routine writes a parameter file in which the selected atoms are turned into dummy atoms.
    !     It automatically generates the corresponding bonded + VdW terms. 
    !     IT DOES NOT WORK WITH CMAP OR NBFIXES! 
    !
    !            By Gerhard Koenig, 2015
    !


  subroutine MKDUMMY(comlyn,comlen)
    use chm_kinds
    use chm_types
    use code
    use consta, only: degrad,raddeg
    use coord
    use ctitla
    use dimens_fcm
    use io 
    use intcor2
    use memory
    use number
    use psf
    use param
    use parmiom
    use rtf,only:atct,natct,rtfrdr
    use select
    use stream
    use string
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec
#endif

    implicit none

    character,intent(inout) :: comlyn*(*)
    integer,intent(inout)   :: comlen
    integer,allocatable,dimension(:) :: ISLCT ! atom selection (atoms that are going to be turned to dummies)
    integer,allocatable,dimension(:) :: SELA  ! List of atom numbers that were selected 
    integer                                        :: nsel    ! number of selected atom  
    integer                                        :: oldnatc ! number of atom types before introduction of new ones
    integer                                        :: oldncb  ! number of bond types before introduction of new ones
    integer                                        :: oldnct   ! number of angle types before introduction of new ones
    integer                                        :: oldncp  ! number of dihedral types before introduction of new ones
    integer                                        :: oldnci   ! number of improper types before introduction of new ones
    integer                                        :: unt, untstr, untpar, unttop ! unit to write stream/parameter/topology file to 
    integer                                        :: iter, maxiter ! Number of iterations for charge redistribution
    character(LEN=100)                              :: cval    ! Temporary variable for strings
    character(LEN=1)                                :: prefix    ! Prefix for dummy atoms 
    logical                                         :: chkopen ! Check if file is open
    logical                                         :: chkexist  ! Check if parameter already exists
    logical                                         :: lrdi         ! Redistribute charges of dummy atoms? 
    logical                                         :: lnostr     ! you don't want to automatically stream the file?
    integer                                        :: i,j,k, stat,wdlength
    integer                                        :: DI, DJ, DK, DL ! indices of atom types involved
    integer                                        :: II, IJ, IK, IL ! atom types involved
    character(LEN=8)                       :: CI, CJ, CK, CL ! atom type codes of atoms involved
    real(chm_real)                            :: DCBB, DCBC  ! dummy bond terms
    real(chm_real)                            :: DCTB, DCTC, DCTUB, DCTUC ! dummy angle terms 
    real(chm_real)                            :: DCPB, DCPC ! dummy torsion terms 
    integer                                        :: DCPD,  DCID   ! dummy torsion + improper multiplicities 
    real(chm_real)                            :: DCIB,  DCIC ! dummy improper torsion term 
    real(chm_real)                            :: DEPS, DRMIN, DEPS14, DRMIN14 ! dummy VdW parameters
    real(chm_real)                            :: CSCAL ! Scaling factor for charges 
    real(chm_real)                            :: DCHAR ! dummy charge
    real(chm_real)                            :: RSCAL ! Scaling factor for VdW radius
    real(chm_real)                            :: totchar,tmpchar,tmpsca 

#if KEY_FLEXPARM==1

#if KEY_DOMDEC==1
    if(q_domdec) &
       call wrndie(-5,'<MKDUMMY>','SORRY, DOMDEC IS NOT SUPPORTED (BUT YOU DONT NEED DOMDEC TO GENERATE THE FILES)')
#endif

    wdlength = 1 
    prefix = 'D' 
    ! Allow user to set the prefix of the dummy atoms - if not, use default: D
    call gtrmwd(comlyn,comlen,'PREF',4,prefix,1,wdlength)

    ! Determine which atoms have been selected 
    call chmalloc('freeene_calc.src','MKDUMMY','ISLCT',NATOM,intg=ISLCT)   
    CALL SELCTA(COMLYN,COMLEN,ISLCT,X,Y,Z,WMAIN,.TRUE.)
    NSEL = SUM(ISLCT(1:NATOM))
    call chmalloc('freeene_calc.src','MKDUMMY','SELA',NSEL,intg=SELA)   

    ! Go through selection
    j = 1 
    do i= 1, NATOM
       if (ISLCT(i) .eq. 1) then
          SELA(j) = i 
          ! Check if atom is already a dummy atom 
          ! write(cval, '(a)') adjustl(ATC(IAC(SELA(j)))) 
          if (CVAL(1:wdlength) .ne. prefix) then 
             j = j + 1 
          else 
             IF(PRNLEV >= 3 .and. iolev > 0) write(outu,'(a,i9,a)') "MKDUMMY> THE ATOM TYPE OF ATOM ", i, " ALREADY SEEMS TO BE A DUMMY ATOM - SKIPPED" 
          endif
       endif
    end do
    nsel = j - 1 

    ! Check units
    untstr=gtrmi(comlyn,comlen,'USTR',-1)  
    untpar=gtrmi(comlyn,comlen,'UPAR',-1)  
    unttop=gtrmi(comlyn,comlen,'UTOP',-1)  

    if(untstr .eq. -1 .and. untpar .eq. -1 .and. unttop .eq. -1) call wrndie(-2,'<MKDUMMY>', "NO UNIT FOR STREAM, PARAMETER OR TOPOLOGY FILE SPECIFIED")

    if(untstr .ne. -1 .and. untpar .ne. -1) call wrndie(-2,'<MKDUMMY>', "CAN'T CREATE STREAM FILE AND PARAMETER FILE AT THE SAME TIME")
    if(untstr .ne. -1 .and. unttop .ne. -1) call wrndie(-2,'<MKDUMMY>', "CAN'T CREATE STREAM FILE AND TOPOLOGY FILE AT THE SAME TIME")

    if(iolev > 0) then
       if(untstr .ne. -1) inquire(untstr, OPENED=chkopen)
       if(untstr .ne. -1 .and. .not. chkopen)  call wrndie(-5,'<MKDUMMY>', "UNIT FOR STREAM FILE NOT OPEN")

       if(untpar .ne. -1) inquire(untpar, OPENED=chkopen)
       if(untpar .ne. -1 .and. .not. chkopen)  call wrndie(-5,'<MKDUMMY>', "UNIT FOR PARAMETER FILE NOT OPEN")
 
       if(unttop .ne. -1) inquire(unttop, OPENED=chkopen)
       if(unttop .ne. -1 .and. .not. chkopen)  call wrndie(-5,'<MKDUMMY>', "UNIT FOR TOPOLOGY FILE NOT OPEN")
    endif


    IF(PRNLEV >= 3 .and. iolev > 0) write(outu, '(a)')' MKDUMMY> GENERATING BONDED PARAMETERS FOR DUMMY ATOMS'         

    oldnatc = natc 
    oldncb=ncb
    oldnct=nct
    oldncp=ncp
    oldnci=nci

    
    if (untstr .ne. -1) unt = untstr 

    ! Write header
    if (untstr .ne. -1) then 
       unt = untstr 
       if(iolev > 0) then 
          write(unt,"(a)") '* Additional topology and parameter file for dummy atoms' 
          write(unt,"(a)") '* ' 
          write(unt,"(a)") ' ' 
          write(unt,"(a)") 'read rtf card append' 
          write(unt,"(a)") '* Additional topology file for dummy atoms' 
          write(unt,"(a)") '* ' 
          write(unt,"(a)") ' '
       endif
    endif 
    if (unttop .ne. -1) then 
       unt = unttop 
       if(iolev > 0) then 
          write(unt,"(a)") '* Additional topology file for dummy atoms' 
          write(unt,"(a)") '* ' 
          write(unt,"(a)") ' ' 
       endif
    endif 
    ! Write new atom types 
    do i= 1, NSEL 
       if (NATC+ I .GT. MAXATC)   call wrndie(-2,'<MKDUMMY>', "MAXIMUM NUMBER OF CHEMICAL TYPES EXCEEDED")
       if(iolev > 0) then 
          write(cval(2:8),'(i7)') SELA(i) 
          write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
          write(unt,'(a,x,i3,x,a,x,f8.5)') 'MASS', NATC+ I , CVAL(1:8) , AMASS(SELA(i)) 
       endif
    end do

    if(iolev > 0) then 
       write(unt,"(a)") ' ' 
       write(unt,"(a)") 'END' 
       call flush(unt)
    endif

    if (untstr .ne. -1 .and. iolev > 0) then 
       unt = untstr 
       write(unt,"(a)") ' ' 
       write(unt,"(a)") 'read para card flex append' 
       write(unt,"(a)") '* Additional parameter file for dummy atoms' 
       write(unt,"(a)") '* ' 
       write(unt,"(a)") ' ' 
    endif 
    if (untpar .ne. -1 .and. iolev > 0) then 
       unt = untpar 
       write(unt,"(a)") '* Additional parameter file for dummy atoms' 
       write(unt,"(a)") '* ' 
       write(unt,"(a)") ' ' 
    endif 
    if(iolev > 0) then 
       write(unt,"(a)") 'ATOMS ' 
       write(unt,"(a)") ' ' 
    endif
    ! Write new atom types 
    do i= 1, NSEL 
       if (NATC+ I .GT. MAXATC)   call wrndie(-2,'<MKDUMMY>', "MAXIMUM NUMBER OF CHEMICAL TYPES EXCEEDED")
       write(cval(2:8), '(i7)') SELA(i) 
       write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
       !write(cval, 'a1,a7)'), 'D', adjustl(cval(2:8))
       if(iolev > 0) write(unt,'(a,x,i3,x,a,x,f8.5)') 'MASS', NATC+ I , CVAL(1:8) , AMASS(SELA(i))  
       ATC(NATC+ I) = CVAL(1:8)
    end do
    NATC=NATC+NSEL ! Now change number of atom types to new value
    ATCCNT=ATCCNT+NSEL ! Change number of active atom types

    ! Write new bond parameters
    if(iolev > 0) write(unt,"(a)") ' ' 
    if(iolev > 0) write(unt,"(a)") 'BONDS ' 
    if(iolev > 0) write(unt,"(a)") ' ' 
    ! Find all bonds associated with new dummy atoms and write new parameters
    do i = 1, nbond
       DCBB = CBB(ICB(i)) ! save equilibrium bond distance 
       DCBC = CBC(ICB(i)) ! save force constant 
       II = IAC(IB(i)) ! save atom type of first atom
       IJ = IAC(JB(i)) ! save atom type of second atom 
       CI = ATC(IAC(IB(i))) ! save atom type code of first atom
       CJ = ATC(IAC(JB(i))) ! save atom type code of second atom 

       do j = 1 , nsel 
          if (IB(i) .eq. SELA(j)) then ! if the first atom is a dummy
             write(cval(2:8), '(i7)') SELA(j) 
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             !write(cval, '(a1,a7)'), 'D', adjustl(cval(2:8))
             CI = cval(1:8)
             II = oldnatc + j  
          endif
          if (JB(i) .eq. SELA(j)) then ! if the second atom is dummy
             write(cval(2:8), '(i7)')SELA(j)
             write(cval, '(a1,a7)')prefix(1:wdlength), adjustl(cval(2:8))
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             CJ = cval(1:8) 
             IJ = oldnatc + j  
          endif
       enddo
       ! If one of the atoms has been a dummy atom, write new parameters      
       if (CI .ne. ATC(IAC(IB(i))) .or. CJ .ne. ATC(IAC(JB(i)))) then 
          ! First check whether there is already a parameter of that type 
          chkexist = .FALSE. 
          do k = oldncb , ncb
             IF ( CI .eq.  ATC(CBAI(k)) .and.   CJ .eq.  ATC(CBAJ(k)) )   chkexist = .TRUE. 
             IF ( CJ .eq.  ATC(CBAI(k)) .and.   CI .eq.  ATC(CBAJ(k)) )   chkexist = .TRUE. 
          enddo
          if (.not. chkexist) then 
             ncb = ncb + 1 
             CBB(ncb) = DCBB
             CBC(ncb) = DCBC
             CBAI(ncb) =  II
             CBAJ(ncb) = IJ
             if(iolev > 0) write(unt,'(a,a,f12.4,f12.4)') CI, CJ, DCBC,  DCBB                
          endif
       endif
    enddo

    ! Write new angle parameters
    if(iolev > 0) write(unt,"(a)") ' ' 
    if(iolev > 0) write(unt,"(a)") 'ANGLES ' 
    if(iolev > 0) write(unt,"(a)") ' ' 
    ! Find all angles associated with new dummy atoms and write new parameters
    do i = 1, ntheta
       DCTB = CTB(ICT(i)) ! save equilibrium bond angle    
       DCTC = CTC(ICT(i)) ! save force constant 
       DCTUB = CTUB(ICT(i)) ! Urey-Bradley equilibrium 1-3 length
       DCTUC = CTUC(ICT(i)) ! Urey-Bradley 1-3 bond force constant
       II = IAC(IT(i)) ! save atom type of first atom
       IJ = IAC(JT(i)) ! save atom type of second atom 
       IK = IAC(KT(i)) ! save atom type of third atom 
       CI = ATC(IAC(IT(i)))   ! save atom type of first atom
       CJ = ATC(IAC(JT(i)))   ! save atom type of second atom 
       CK = ATC(IAC(KT(i))) ! save atom type of third atom 

       do j = 1 , nsel 
          if (IT(i) .eq. SELA(j)) then ! if the first atom is a dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CI = cval(1:8) 
             II = oldnatc + j  
          endif
          if (JT(i) .eq. SELA(j)) then ! if the second atom is dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CJ = cval(1:8) 
             IJ = oldnatc + j  
          endif
          if (KT(i) .eq. SELA(j)) then ! if the third atom is dummy
             write(cval(2:8), '(i7)') SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CK = cval(1:8) 
             IK = oldnatc + j  
          endif
       enddo
       ! If one of the atoms has been a dummy atom, write new parameters      
       if (CI .ne. ATC(IAC(IT(i))) .or. CJ .ne. ATC(IAC(JT(i))) .or.   CK .ne. ATC(IAC(KT(i)))  ) then 
          ! First check whether there is already a parameter of that type 
          chkexist = .FALSE. 
          do k = oldnct , nct
             IF ( CI .eq.  ATC(CTAI(k)) .and.   CJ .eq.  ATC(CTAJ(k)) .and. CK .eq.  ATC(CTAK(k)) )   chkexist = .TRUE. 
             IF ( CK .eq.  ATC(CTAI(k)) .and.   CJ .eq.  ATC(CTAJ(k)) .and. CI .eq.  ATC(CTAK(k)) )   chkexist = .TRUE. 
          enddo
          if (.not. chkexist) then 
             nct = nct + 1 
             CTB(nct) = DCTB
             CTC(nct) = DCTC
             CTUC(nct)=DCTUC
             CTUB(nct)=DCTUB
             CTAI(nct) =  II
             CTAJ(nct) = IJ
             CTAK(nct) = IK
             if(iolev > 0) write(unt,'(a,a,a,f12.4,f12.4,f12.4,f12.4)') CI, CJ, CK,  DCTC, REAL(DCTB)*raddeg, DCTUC , DCTUB             
          endif
       endif
    enddo

    ! Write new dihedral parameters
    if(iolev > 0) write(unt,"(a)") ' ' 
    if(iolev > 0) write(unt,"(a)") 'DIHEDRALS ' 
    if(iolev > 0) write(unt,"(a)") ' ' 
    ! Find all dihedrals associated with new dummy atoms and write new parameters
    do i = 1, nphi
       DCPB = CPB(ICP(i)) ! save phase shift               
       DCPC = CPC(ICP(i)) ! save force constant 
       DCPD = CPD(ICP(i)) ! periodicity 
       II = IAC(IP(i))   ! save atom type of first atom
       IJ = IAC(JP(i))   ! save atom type of second atom 
       IK = IAC(KP(i)) ! save atom type of third atom 
       IL = IAC(LP(i)) ! save atom type of third atom 
       CI = ATC(IAC(IP(i)))   ! save atom type of first atom
       CJ = ATC(IAC(JP(i)))   ! save atom type of second atom 
       CK = ATC(IAC(KP(i))) ! save atom type of third atom 
       CL = ATC(IAC(LP(i))) ! save atom type of third atom 

       do j = 1 , nsel 
          if (IP(i) .eq. SELA(j)) then ! if the first atom is a dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CI = cval(1:8) 
             II = oldnatc + j  
          endif
          if (JP(i) .eq. SELA(j)) then ! if the second atom is dummy
             write(cval(2:8), '(i7)')SELA(j) 
             ! write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)')prefix(1:wdlength), adjustl(cval(2:8))
             CJ = cval(1:8) 
             IJ = oldnatc + j  
          endif
          if (KP(i) .eq. SELA(j)) then ! if the third atom is dummy
             write(cval(2:8), '(i7)') SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CK = cval(1:8) 
             IK = oldnatc + j  
          endif
          if (LP(i) .eq. SELA(j)) then ! if the fourth atom is dummy
             write(cval(2:8), '(i7)') SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CL = cval(1:8) 
             IL = oldnatc + j  
          endif
       enddo
       ! If one of the atoms has been a dummy atom, write new parameters      
       if (CI .ne. ATC(IAC(IP(i))) .or. CJ .ne. ATC(IAC(JP(i))) .or.   CK .ne. ATC(IAC(KP(i)))  .or.   CL .ne. ATC(IAC(LP(i)))  ) then 
          ! First check whether there is already a parameter of that type 
          chkexist = .FALSE. 
          do k = oldncp , ncp
             IF ( CI .eq. ATC(CPAI(k)) .and. CJ .eq. ATC(CPAJ(k)) .and. CK .eq. ATC(CPAK(k)) .and. CL .eq. ATC(CPAL(k)))   chkexist = .TRUE. 
             IF ( CL .eq. ATC(CPAI(k)) .and. CK .eq. ATC(CPAJ(k)) .and. CJ .eq. ATC(CPAK(k)) .and. CI .eq. ATC(CPAL(k)))   chkexist = .TRUE. 
          enddo
          if (.not. chkexist) then 
             ncp = ncp + 1 
             CPB(ncp) = DCPB
             CPC(ncp) = DCPC
             CPD(ncp)= DCPD        
             CPAI(ncp) =  II
             CPAJ(ncp) = IJ
             CPAK(ncp) = IK
             CPAL(ncp) = IL
             if(iolev > 0) write(unt,'(a,a,a,a,f12.4,i12,f12.4)') CI, CJ, CK, CL,  DCPC, ABS(DCPD), REAL(DCPB)*raddeg
             j = 1 
             do while (DCPD .lt. 0) ! take care of multiple dihedral potentials (indicated by negative multiplicities)
                ! Check that the next row still has the same atom types 
                chkexist = .FALSE.               
                if(  CPAI(ICP(i)) .ne.  CPAI(ICP(i)+J) ) chkexist = .TRUE. 
                if(  CPAJ(ICP(i)) .ne.  CPAJ(ICP(i)+J) ) chkexist = .TRUE. 
                if(  CPAK(ICP(i)) .ne.  CPAK(ICP(i)+J) ) chkexist = .TRUE. 
                if(  CPAL(ICP(i)) .ne.  CPAL(ICP(i)+J) ) chkexist = .TRUE. 
                if (chkexist) exit 
                ! Check that no multiplicity occurs twice 
                chkexist = .FALSE.               
                do k = 1,j-1
                   if( ABS(CPD(ICP(i)+J))  .eq.  ABS(CPD(ICP(i)+K)))  chkexist = .TRUE. 
                enddo
                DCPB = CPB(ICP(i)+J) 
                DCPC = CPC(ICP(i)+J) 
                DCPD = CPD(ICP(i)+J)   
                j = j + 1 
                if (chkexist) cycle   
                ncp = ncp + 1 
                CPB(ncp) = DCPB
                CPC(ncp) = DCPC
                CPD(ncp)= DCPD        
                CPAI(ncp) =  II
                CPAJ(ncp) = IJ
                CPAK(ncp) = IK
                CPAL(ncp) = IL                  
                if(iolev > 0) write(unt,'(a,a,a,a,f12.4,i12,f12.4)') CI, CJ, CK, CL,DCPC,ABS(DCPD),REAL(DCPB)*raddeg
             enddo
          endif
       endif
    enddo

    ! Write new improper parameters
    if(iolev > 0) write(unt,"(a)") ' ' 
    if(iolev > 0) write(unt,"(a)") 'IMPROPER ' 
    if(iolev > 0) write(unt,"(a)") ' ' 
    ! Find all dihedrals associated with new dummy atoms and write new parameters
    do i = 1, nimphi
       DCIB = CIB(ICI(i)) ! save phase shift               
       DCIC = CIC(ICI(i)) ! save force constant 
       DCID = CID(ICI(i)) ! periodicity 
       II = IAC(IM(i))   ! save atom type of first atom
       IJ = IAC(JM(i))   ! save atom type of second atom 
       IK = IAC(KM(i)) ! save atom type of third atom 
       IL = IAC(LM(i)) ! save atom type of third atom 
       CI = ATC(IAC(IM(i)))   ! save atom type of first atom
       CJ = ATC(IAC(JM(i)))   ! save atom type of second atom 
       CK = ATC(IAC(KM(i))) ! save atom type of third atom 
       CL = ATC(IAC(LM(i))) ! save atom type of third atom 

       do j = 1 , nsel 
          if (IM(i) .eq. SELA(j)) then ! if the first atom is a dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CI = cval(1:8) 
             II = oldnatc + j  
          endif
          if (JM(i) .eq. SELA(j)) then ! if the second atom is dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CJ = cval(1:8) 
             IJ = oldnatc + j  
          endif
          if (KM(i) .eq. SELA(j)) then ! if the third atom is dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CK = cval(1:8) 
             IK = oldnatc + j  
          endif
          if (LM(i) .eq. SELA(j)) then ! if the fourth atom is dummy
             write(cval(2:8), '(i7)')SELA(j) 
             !write(cval, '(a1,a7)')'D', adjustl(cval(2:8))
             write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
             CL = cval(1:8) 
             IL = oldnatc + j  
          endif
       enddo

       ! If one of the atoms has been a dummy atom, write new parameters      
       if (CI .ne. ATC(IAC(IM(i))) .or. CJ .ne. ATC(IAC(JM(i))) .or.   CK .ne. ATC(IAC(KM(i)))  .or.   CL .ne. ATC(IAC(LM(i)))  ) then 
          ! First check whether there is already a parameter of that type 
          chkexist = .FALSE. 
          do k = oldnci , nci
             IF ( CI .eq. ATC(CIAI(k)) .and. CJ .eq. ATC(CIAJ(k)) .and. CK .eq. ATC(CIAK(k)) .and. CL .eq. ATC(CIAL(k)))   chkexist = .TRUE. 
             IF ( CL .eq. ATC(CIAI(k)) .and. CK .eq. ATC(CIAJ(k)) .and. CJ .eq. ATC(CIAK(k)) .and. CI .eq. ATC(CIAL(k)))   chkexist = .TRUE. 
          enddo
          if (.not. chkexist) then 
             nci = nci + 1 
             CIB(nci) = DCIB
             CIC(nci) = DCIC
             CID(nci)= DCID        
             CIAI(nci) =  II
             CIAJ(nci) = IJ
             CIAK(nci) = IK
             CIAL(nci) = IL
             if(iolev > 0) write(unt,'(a,a,a,a,f12.4,i12,f12.4)') CI, CJ, CK, CL,  DCIC, DCID, REAL(DCIB)*raddeg
          endif
       endif
    enddo

    ! Scale charges 
    cscal = gtrmf(comlyn, comlen, 'CSCA',0.0d0)
    IF(PRNLEV >= 3 .and. iolev > 0) write(outu, '(a,f12.4)')' MKDUMMY> CHARGES OF SELECTED ATOMS ARE SCALED BY ', REAL(CSCAL)
    do j = 1 , nsel        
       dchar = CSCAL*CG(SELA(j))
       CG(SELA(j)) = dchar
    enddo

    rscal = gtrmf(comlyn, comlen, 'RSCA',0.0d0)
    IF(PRNLEV >= 3 .and. iolev > 0) write(outu, '(a,f12.4)')' MKDUMMY> VDW PARAMETERS OF SELECTED ATOMS ARE SCALED BY ', REAL(RSCAL)

     if(iolev > 0) write(unt,"(a)") 'NONBONDED' 
     if(iolev > 0) write(unt,"(a)") ' ' 

    ! Lennard Jones parameters 
    do j = 1 , nsel        
       ! Find offset in triangular matrix of non-bonded pairs
       k = 0
       do I=1,ITC(IAC(SELA(j)))
          k = k+i 
       enddo
       write(cval(2:8), '(i7)') SELA(j) 
       !write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
       write(cval, '(a1,a7)') prefix(1:wdlength), adjustl(cval(2:8))
       deps = -RSCAL*CNBB(k)
       CNBB(k) = -deps
       drmin = RSCAL*SQRT(CNBA(K))/2.0
       CNBA(K) = (2.0*drmin)**2.0 
       deps14 = -RSCAL*CNBB(k+MAXCN)
       CNBB(k+MAXCN) = -deps14
       drmin14 = RSCAL*SQRT(CNBA(K+MAXCN))/2.0
       CNBA(K+MAXCN) = (2.0*drmin14)**2.0 
       if ( drmin .lt. 0.4 .and. rscal .lt. 1.0) then 
          deps = 0.0 
          drmin = 0.0 
          deps14 = 0.0 
          drmin14 = 0.0 
          IF(PRNLEV .ge. 5 .and. rscal .ne. 0.0d0 ) write(outu, '(a,i6,a)') ' MKDUMMY> NEW RMIN OF ATOM ', SELA(j), ' BELOW 0.4 ANGSTROM - SET TO ZERO' 
       endif
       if(iolev > 0) write(unt,'(a,f12.1,f12.4,f12.4,f12.1,f12.4,f12.4)') CVAL(1:8), 0.0, REAL(deps), REAL(drmin), 0.0, REAL(deps14), REAL(drmin14)
    enddo

    ! Redistribute charges of dummies? 
    lrdi = (INDXA(COMLYN,COMLEN,'REDI').NE.0)
    if (lrdi) then 
       IF(PRNLEV .ge. 5 ) write(outu, '(a)')' MKDUMMY> REDISTRIBUTING CHARGES OF DUMMY ATOMS IN SELECTION' 
       ! Make sure that (almost) dummy atoms don't have charges 
       totchar = 0.0 
       do j = 1 , nsel        
          k = 0
          do I=1,ITC(IAC(SELA(j)))
             k = k+i 
          enddo
          drmin = SQRT(CNBA(K))/2.0
          if ( drmin .lt. 0.4 .and. rscal .lt. 1.0) then   
             totchar = totchar +  ABS(CG(SELA(j)))
          endif
       enddo

       !Loop until charges on dummies are zero  
       iter = 0 
       maxiter=gtrmi(comlyn,comlen,'MAXI',500)  
       do while( abs(totchar) .gt. 0.000005 )
          do j = 1 , nsel        
             ! Find offset in triangular matrix of non-bonded pairs
             k = 0
             do I=1,ITC(IAC(SELA(j)))
                k = k+i 
             enddo
             ! Find VdW Radius again
             drmin = SQRT(CNBA(K))/2.0
             ! If this is a dummy, get rid of charges 
             if ( drmin .lt. 0.4 .and. rscal .lt. 1.0) then 
                dchar = CG(SELA(j))
                if(dchar .ne. 0.0) then 
                   CG(SELA(j)) = 0.0 ! set charge to zero, otherwise terrible things might happen
                   k = 0 
                   tmpchar = 0.0 
                   do i = 1, nbond  ! To neutralize charge, find atoms bonded to this atom and of opposite charge
                      if (IB(i) .eq. SELA(j) .and. CG(JB(i))*dchar .lt. -0.0001 ) then 
                         k=k+1
                         tmpchar = tmpchar +  CG(JB(i))
                      elseif (JB(i) .eq. SELA(j) .and. CG(IB(i))*dchar .lt. -0.0001 ) then
                         k=k+1
                         tmpchar = tmpchar +  CG(IB(i))
                      elseif  (IB(i) .eq. SELA(j) .or. JB(i) .eq. SELA(j)) then
                         k=k+1 
                      endif
                   enddo
                   if (k .ne. 0)  then  ! If there are neighbors, we have to worry about neutralizing the charge locally
                      tmpsca = 1.0 
                      if (abs(tmpchar) .gt. 0.0001) tmpsca = 1.0+dchar/tmpchar ! scaling factor for opposite charges 
                      do i = 1, nbond  
                         if (IB(i) .eq. SELA(j) .and. CG(JB(i))*dchar .lt. -0.0001 ) then 
                            dchar = dchar + (1.0 - tmpsca) * CG(JB(i))
                            CG(JB(i)) = tmpsca * CG(JB(i))
                         elseif (JB(i) .eq. SELA(j) .and. CG(IB(i))*dchar .lt. -0.0001 ) then 
                            dchar = dchar + (1.0 - tmpsca) * CG(IB(i))
                            CG(IB(i)) = tmpsca * CG(IB(i))
                         endif
                      enddo
                      ! If the net charge is still not zero, distribute the rest of the charge among all neighbors 
                      ! (pray that nothing bad happens - otherwise the user has to deal with it )
                      tmpsca = dchar / real(k)
                      do i = 1, nbond  
                         if (IB(i) .eq. SELA(j)) then 
                            CG(JB(i)) = CG(JB(i)) + tmpsca
                         elseif (JB(i) .eq. SELA(j)) then 
                            CG(IB(i)) = CG(IB(i)) + tmpsca
                         endif
                      enddo
                   endif
                endif
             endif
          enddo

          ! Recalculate total charge of dummies
          totchar = 0.0 
          do j = 1 , nsel        
             k = 0
             do I=1,ITC(IAC(SELA(j)))
                k = k+i 
             enddo
             drmin = SQRT(CNBA(K))/2.0
             if ( drmin .lt. 0.4 .and. rscal .lt. 1.0) then   
                totchar = totchar +  ABS(CG(SELA(j)))
             endif          
          enddo

          iter = iter + 1 
          if (iter .gt. maxiter) then
             call wrndie(-1,'<MKDUMMY>', "CHARGE REDISTRIBUTION OF DUMMY ATOMS DID NOT CONVERGE (CHANGE MAXITER?)")
             exit 
          endif
       enddo

       ! Once we are done, round the charges
       do j = 1 , nsel     
          CG(SELA(j)) =   REAL(AINT(CG(SELA(j))*1000000.0))*0.0000001 
          if ( ABS(CG(SELA(j))) .lt. 1.0E-6 ) CG(SELA(j)) = 0.0 
       enddo
    endif



    if(iolev > 0) write(unt,"(a)") ' ' 
    if(iolev > 0) write(unt,"(a)") 'END' 
    if(iolev > 0) write(unt,"(a)") 'RETURN' 
    if(iolev > 0) call flush(unt)

    ! Determine whether user wants to use the stream/par/top files or not (NOSTREAM keyword)
    lnostr = (INDXA(COMLYN,COMLEN,'NOST').NE.0)

    !Now change the atom types to the new ones 
    do j = 1 , nsel        
       write(cval(2:8), '(i7)') SELA(j) 
       ! write(cval, '(a1,a7)') 'D', adjustl(cval(2:8))
       write(cval, '(a1,a7)')prefix(1:wdlength), adjustl(cval(2:8))
       IAC(SELA(j)) = oldnatc+ J
       if (lnostr) ATC(IAC(SELA(j))) = CVAL
       if (.not. lnostr) ATC(IAC(SELA(j))) = ''
    enddo

    IF(PRNLEV > 3  .and. iolev > 0 ) write(outu, '(a)')' MKDUMMY> CMAP AND NBFIXES REMOVED FROM SELECTED ATOMS ' 



    if (.not. lnostr) then
       ! Get rid of all the evidence of how much we have messed with the innards of CHARMM 
       ! (avoids error messages because of repeated parameters when we read in the parameters in next step)
       CBB(oldncb+1:ncb) = 0.0
       CBC(oldncb+1:ncb) = 0.0  
       CBAI(oldncb+1:ncb) =  0
       CBAJ(oldncb+1:ncb) = 0
       CTB(oldnct+1:nct) = 0.0
       CTC(oldnct+1:nct) = 0.0
       CTUC(oldnct+1:nct)= 0.0
       CTUB(oldnct+1:nct)= 0.0
       CTAI(oldnct+1:nct) =  0
       CTAJ(oldnct+1:nct) = 0
       CTAK(oldnct+1:nct) = 0
       CPB(oldncp+1:ncp) = 0.0
       CPC(oldncp+1:ncp) = 0.0
       CPD(oldncp+1:ncp)= 0       
       CPAI(oldncp+1:ncp) =  0
       CPAJ(oldncp+1:ncp) = 0
       CPAK(oldncp+1:ncp) = 0
       CPAL(oldncp+1:ncp) = 0    
       CIB(oldnci+1:nci) = 0.0
       CIC(oldnci+1:nci) = 0.0
       CID(oldnci+1:nci)= 0        
       CIAI(oldnci+1:nci) =  0
       CIAJ(oldnci+1:nci) = 0
       CIAK(oldnci+1:nci) = 0
       CIAL(oldnci+1:nci) = 0
       natc = oldnatc 
       ncb= oldncb
       nct= oldnct
       ncp= oldncp
       nci= oldnci

       ! Now we stream the freshly generated file 
       if (untstr .ne. -1) then 
          rewind(untstr)
          if (unt .ge. 10000 )  then
             call wrndie(-2,'<MKDUMMY>', "UNIT NUMBER TOO LONG FOR AUTOMATIC STREAMING")
          else
             IF(PRNLEV >= 3  .and. iolev > 0 ) write(outu, '(a,i6)')' MKDUMMY> STREAMING NEW PARAMETERS FROM UNIT ', untstr
             write(cval, '(a12,i6)') 'STREAM UNIT ', unt
             if(iolev > 0) call flush(outu)
             call JOINWD(COMLYN,MXCMSZ,COMLEN,CVAL(1:18),18)
             call MISCOM(COMLYN,MXCMSZ,COMLEN,CHKEXIST)
          endif
       endif
 
      if (unttop .ne. -1 .and. untpar .ne. -1) then 
          rewind(unttop)
          rewind(untpar)
          if (unttop .ge. 10000 .or. untpar .ge. 10000 )  then
             call wrndie(-2,'<MKDUMMY>', "UNIT NUMBER TOO LONG FOR AUTOMATIC STREAMING")
          else
             IF(PRNLEV >= 3 .and. iolev > 0) write(outu, '(a,i6)')' MKDUMMY> STREAMING NEW TOPOLOGY FROM UNIT ', unttop
             !write(cval, '(a19,i6)') 'READ RTF CARD UNIT ', unttop
              if(iolev > 0) call flush(outu)
             !call JOINWD(COMLYN,MXCMSZ,COMLEN,CVAL(1:26),26)
             !call MISCOM(COMLYN,MXCMSZ,COMLEN,CHKEXIST)
             call rtfrdr(unttop,titleb,ntitlb,1,.false.,.true.)

             IF(PRNLEV >= 3 .and. iolev > 0) write(outu, '(a,i6)')' MKDUMMY> STREAMING NEW PARAMETERS FROM UNIT ', untpar
             write(cval, '(a4)')'FLEX'
             call flush(outu)
             call JOINWD(COMLYN,MXCMSZ,COMLEN,CVAL(1:4),4)
             !call MISCOM(COMLYN,MXCMSZ,COMLEN,CHKEXIST)
             call parmio(untpar,ntitlb,titleb,1,outu,natct,atct,.true.)
             !call parrdr(untpar,ntitlb,titleb,1,.false.,.true.)
          endif
       endif
       
       mustup = .true. 
    endif
    
    call chmdealloc('freeene_calc.src','MKDUMMY','SELA',NSEL,intg=SELA)
    call chmdealloc('freeene_calc.src','MKDUMMY','ISLCT',NATOM,intg=ISLCT)


    RETURN
#else
   call wrndie(-3,'<MKDUMMY>','MKDUMMY REQUIRES FLEXPARAM')
#endif
  END SUBROUTINE MKDUMMY
  
! --------------------------------------------------------------------------------------------------------------------------------------------------

  ! Free up everything allocated upon exit
  subroutine free_diff()
     use memory
     implicit none


     dA%n0 = 0
     dA%n1 = 0
     dA%ndU0 = 0
     dA%ndU1 = 0
     dA%nVb0 = 0
     dA%nVb1 = 0
     dA%nU00 = 0
     dA%nU01 = 0
     dA%nU10 = 0
     dA%nU11 = 0
     dA%nUS0 = 0
     dA%nUS1  = 0
     dA%q_bias0 = .false. 
     dA%q_bias1 = .false. 


     if(allocated(dA%Vb0)) call chmdealloc('freeene_calc.src','FREE_DA','DA%VB0',dA%nVb0,crl=dA%Vb0)
     if(allocated(dA%Vb1)) call chmdealloc('freeene_calc.src','FREE_DA','DA%VB1',dA%nVb1,crl=dA%Vb1)
     if(allocated(dA%dU0)) call chmdealloc('freeene_calc.src','FREE_DA','DA%DU01',dA%ndU0,crl=dA%dU0)
     if(allocated(dA%dU1)) call chmdealloc('freeene_calc.src','FREE_DA','DA%DU10',dA%ndU1,crl=dA%dU1)

     if(allocated(dA%U00)) call chmdealloc('freeene_calc.src','FREE_DA','DA%U00',dA%nU00,crl=dA%U00)
     if(allocated(dA%U01)) call chmdealloc('freeene_calc.src','FREE_DA','DA%U01',dA%nU01,crl=dA%U01)
     if(allocated(dA%US0)) call chmdealloc('freeene_calc.src','FREE_DA','DA%US0',dA%nUS0,crl=dA%US0)

     if(allocated(dA%U10)) call chmdealloc('freeene_calc.src','FREE_DA','DA%U10',dA%nU10,crl=dA%U10)
     if(allocated(dA%U11)) call chmdealloc('freeene_calc.src','FREE_DA','DA%U11',dA%nU11,crl=dA%U11)
     if(allocated(dA%US1)) call chmdealloc('freeene_calc.src','FREE_DA','DA%US1',dA%nUS1,crl=dA%US1)

  end subroutine free_diff

! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! This routine reads column data into an array
  subroutine read_column_data(unt,col,skip,offset,npt,out)
     use ctitla
     use stream
     use string
     use comand
     use memory
     implicit none

     integer, intent(in)                                    :: unt,col,skip,offset     
     integer, intent(inout)                               :: npt ! if npt < 0, figure it out ourselves
     real(chm_real),allocatable,dimension(:) :: out

     character(len=mxcmsz)                         :: line,cdata
     character(len=120)                                :: errmsg
     integer                                                    :: i,k, l, stat, lnlen, cdlen
     integer                                                    :: nline, nread, readlim, sz
     integer, parameter                                 :: maxlines = 900000000
     logical                                                     :: chkopen

     ! Check if unit is open 
     if(iolev > 0) inquire(unt, OPENED=chkopen)

     if (.not. chkopen .and. iolev > 0)  call wrndie(-5,'<READ_COLUMN_DATA>', "UNIT NOT OPEN")

     if(iolev <= 0) return
     rewind(unt)
     !!call rdtitl(titleb,ntitlb,unt,0)
     stat = 0
     nline = 0
     nread = 0

     if(npt >= 0) then
        readlim = npt
        sz = npt
     else
        readlim = maxlines

        ! assume we have to keep reallocating the output array
        sz = 128
        call chmalloc('FREEENE_CALC.SRC','READ_COLUMN_DATA','OUT',sz,crl=out)
     endif

     read(unt,"(a)",iostat=stat) line
     call flush(outu)
     do while(stat == 0 .and. nread < readlim)
        nline = nline + 1
        if(nline <= offset) then 
           read(unt,"(a)",iostat=stat) line
           call flush(outu)
           cycle
        endif 
        if(mod((nline-offset-1),skip) /= 0) then 
           read(unt,"(a)",iostat=stat) line
           call flush(outu)
           cycle
        endif 

        ! Replace stupid tabs by spaces 
        k = scan(line, char(9)) 
        do while(k .ne. 0) 
           line(k:k) = " "
           k = scan(line, char(9)) 
        end do

        ! Replace other stupid type of tabs by spaces 
        k = scan(line, char(11)) 
        do while(k .ne. 0) 
           line(k:k) = " "
           k = scan(line, char(11)) 
        end do

        line = adjustl(line)
        lnlen = len_trim(line) 

        if(lnlen .eq. 0) cycle

        ! This code has a bit of extra redundancy in it,
        ! but it makes the error handling a bit nicer.
        do i=1,col
           call nextwd(line,lnlen,cdata,mxcmsz,cdlen)
           if(cdlen == 0) then
              write(errmsg,'(a,i3,a,i6)') 'COLUMN ',col,' CANNOT BE READ ON LINE ',nline
              call wrndie(-2,'<READ_COLUMN_DATA>',errmsg)
              exit
           endif
        enddo

         if(nread + 1 > sz) then
           sz = sz*2
           call chmrealloc('FREEENE_CALC.SRC','READ_COLUMN_DATA','OUT',sz,crl=out)
        endif
        out(nread+1)=decodf(cdata,cdlen)
        nread=nread+1

        if(prnlev > 5 .and. nread == 1) then
           write(outu,'(a,f14.4)') 'READ_COLUMN_DATA> READ      FIRST LINE  DATA READ = ',out(nread)
           call flush(outu)
        endif
        if(prnlev > 6 .and. mod(nread,1000) == 0) then
           write(outu,'(a,i9,a,f14.4)') 'READ_COLUMN_DATA> READ  ',nread,' LINES DATA READ = ',out(nread)
           call flush(outu)
        endif

        read(unt,"(a)",iostat=stat) line
     enddo

     if(npt < 0) then
        npt = nread
        call chmrealloc('FREEENE_CALC.SRC','READ_COLUMN_DATA','OUT',npt,crl=out)
     else if(nread /= npt) then
        write(errmsg,'(a,i6,a,i6)') 'EXPECTED ',npt,' DATA POINTS BUT ONLY READ ',nread
        call wrndie(-5,'<READ_COLUMN_DATA>',errmsg)
     endif

     !write(outu,'(a)') 'READ_COLUMN_DATA> ALL DONE BUH-BYE'
     call flush(outu)
  end subroutine read_column_data

! --------------------------------------------------------------------------------------------------------------------------------------------------
  elemental real(chm_real) function fermi(x)
     implicit none
     real(chm_real), intent(in)   :: X
     fermi = 1/(1+exp(x))
  end function fermi

! --------------------------------------------------------------------------------------------------------------------------------------------------
  function crossproduct3(a,b) result(axb)
     implicit none
     real(chm_real), dimension(3)               :: axb
     real(chm_real), dimension(3), intent(in)   :: a,b
     axb(1) = a(2)*b(3) - a(3)*b(2)
     axb(2) = a(3)*b(1) - a(1)*b(3)
     axb(3) = a(1)*b(2) - a(2)*b(1)     
  end function crossproduct3


! --------------------------------------------------------------------------------------------------------------------------------------------------
! Calculate mean value of a vector 
  subroutine mean(d,n,result)
     use number
     implicit none

     real(chm_real),intent(in),dimension(:) :: d
     integer,intent(in)                     :: n
     real(chm_real),intent(out)             :: result

     integer        :: i

     result = d(1)
     do i=2,n   
        result = result + (d(i)-result)/real(i)
     enddo

  end subroutine mean

! --------------------------------------------------------------------------------------------------------------------------------------------------
! Calculate standard deviation of a vector given a mean 
  subroutine stddev(d,n,ave,result)
     use number
     implicit none

     real(chm_real),intent(in),dimension(:) :: d
     real(chm_real),intent(in)              :: ave
     integer,intent(in)                     :: n
     real(chm_real),intent(out)             :: result

     real(chm_real) :: sum
     integer        :: i

     sum=zero
     do i=1,n
        sum = sum + ((d(i) - ave)**2)/real(i)         
     enddo
     result = sqrt( sum*(real(n)/real(n-1)) )
  end subroutine stddev
! --------------------------------------------------------------------------------------------------------------------------------------------------
! Calculate the skewness given a mean and a standard deviation 
  subroutine skew(d,n,ave,stddev,result)
     use number
     implicit none

     real(chm_real),intent(in),dimension(:) :: d
     real(chm_real),intent(in)              :: ave
     real(chm_real),intent(in)              :: stddev
     integer,intent(in)                     :: n
     real(chm_real),intent(out)             :: result

     real(chm_real) :: sum
     integer        :: i

     sum=zero
     do i=1,n
        sum = sum + ((d(i) - ave)**3 - sum)/real(i)   ! Replaced the sum by an average for better stability           
     enddo
     result = sum/(stddev**3)
  end subroutine skew 
! --------------------------------------------------------------------------------------------------------------------------------------------------
! Calculate the kurtosis given a mean and a standard deviation 
  subroutine kurt(d,n,ave,stddev,result)
     use number
     implicit none

     real(chm_real),intent(in),dimension(:) :: d
     real(chm_real),intent(in)              :: ave
     real(chm_real),intent(in)              :: stddev
     integer,intent(in)                     :: n
     real(chm_real),intent(out)             :: result

     real(chm_real) :: sum
     integer        :: i

     sum=zero
     do i=1,n
        sum = sum + ((d(i) - ave)**4 - sum)/real(i)    ! Replaced the sum by an average for better stability 
      enddo
      result = sum/(stddev**4)
  end subroutine kurt
! --------------------------------------------------------------------------------------------------------------------------------------------------

! Detect corrupted data in a vector based on the ten-sigma rule of thumb (or at least 10 kcal/mol difference)
! (what's good enough for Higgs and CERN is not good enough for us)
  subroutine outlier1(x,n,traj)
    use number
    use stream
    implicit none
    
    real(chm_real),intent(inout),dimension(:) :: x        ! Vector containing data 
    integer,intent(inout)                                    :: n        ! Number of data points 
    integer,intent(in)                                       :: traj     ! Which trajectory (0 or 1)?
    real(chm_real)                                           :: ave    ! Average value of vector  
    real(chm_real)                                           :: sd      ! Standard deviation
    real(chm_real)                                           :: skewness  ! Skewness of distribution - for quality check
    real(chm_real)                                           :: kurtosis  ! Kurtosis of distribution - for quality check 
    real(chm_real)                                           :: dev    ! deviation from mean
    integer                                                  :: i, newn 

    call mean(x,n,ave)
    call stddev(x,n,ave,sd)
    call skew(x,n,ave,sd,skewness)
    call kurt(x,n,ave,sd,kurtosis)

    if(prnlev > 6) then
       write(outu,'(a,e12.4,a,e12.4,a,f8.2a,f8.2)') " OUTLIER>  DISTRIBUTION MEAN", real(ave), " SD ", real(sd) ," SKEWNESS " ,  real(skewness) , " KURTOSIS ", real(kurtosis)
    endif   


    if(abs(skewness) .gt. two) return ! If the distribution is too skew, we cannot really determine outliers
    if(abs(kurtosis-3) .gt. two) return ! If the tailedness of the distribution is wrong, we also cannot really determine outliers

    newn = n 
    
    do i=1,n
       dev = abs(x(i)-ave)
       if (  dev > MAX(sd*10.0,10.0)) then 
          x(i:(newn-1)) = x((i+1):newn)
          x(newn) = zero
          newn = newn - 1 
          if(prnlev > 6) then
             write(outu,'(a,i12,a,i1,a,f14.4,a)') "OUTLIER REMOVED AT LINE ", i, " OF TRAJECTORY ", traj ," DUE TO DEVIATION OF " ,  dev , " KCAL/MOL FROM MEAN"
          endif
       endif
       dev = x(i) - ave 
    enddo
    
    if (newn .ne. n) write(outu,'(a,i1,a,i12,a,i12,a)') "NUMBER OF DATA POINTS IN TRAJECTORY ", traj, " CHANGED FROM ", n, " TO " ,  newn, " DUE TO DATA CORRUPTION" 
    n = newn


  end subroutine outlier1
  
! Detect corrupted data in a vector and also delete corresponding line in some other data set 
  subroutine outlier2(x,y,n,traj)
    use number
    use stream
    implicit none
    
    real(chm_real),intent(inout),dimension(:) :: x,y      ! Vector containing data 
    integer,intent(inout)                                  :: n        ! Number of data points 
    integer,intent(in)                                       :: traj     ! Which trajectory (0 or 1)?
    real(chm_real)                                           :: ave    ! Average value of vector  
    real(chm_real)                                           :: sd      ! Standard deviation
    real(chm_real)                                           :: skewness  ! Skewness of distribution - for quality check
    real(chm_real)                                           :: kurtosis  ! Kurtosis of distribution - for quality check 
    real(chm_real)                                           :: dev    ! deviation from mean
    integer                                                       :: i, newn 
    
    call mean(x,n,ave)
    call stddev(x,n,ave,sd)
    call skew(x,n,ave,sd,skewness)
    call kurt(x,n,ave,sd,kurtosis)

     if(prnlev > 6) then
       write(outu,'(a,e12.4,a,e12.4,a,f8.2a,f8.2)') " OUTLIER> DISTRIBUTION MEAN", real(ave), " SD ", real(sd) ," SKEWNESS " ,  real(skewness) , " KURTOSIS ", real(kurtosis)
    endif   
    
    if(abs(skewness) .gt. two) return ! If the distribution is too skew, we cannot really determine outliers
    if(abs(kurtosis-3) .gt. two) return ! If the tailedness of the distribution is wrong, we also cannot really determine outliers
      
    newn = n 
    
    do i=1,n
       dev = abs(x(i)-ave)
       if (  dev > MAX(sd*10.0,10.0)) then 
          x(i:(newn-1)) = x((i+1):newn)
          y(i:(newn-1)) = y((i+1):newn)
          x(newn) = zero
          y(newn) = zero
          newn = newn - 1 
          if(prnlev > 6) then
             write(outu,'(a,i12,a,i1,a,f14.4,a)') "OUTLIER REMOVED AT LINE ", i, " OF TRAJECTORY ", traj ," DUE TO DEVIATION OF " ,  dev , " KCAL/MOL FROM MEAN"
          endif
       endif
       dev = x(i) - ave 
    enddo
    
    if (newn .ne. n) write(outu,'(a,i1,a,i12,a,i12,a)') "NUMBER OF DATA POINTS IN TRAJECTORY ", traj, " CHANGED FROM ", n, " TO " ,  newn, " DUE TO DATA CORRUPTION" 
    n = newn

  end subroutine outlier2


! --------------------------------------------------------------------------------------------------------------------------------------------------
  subroutine barsort(qrw)
     use number
     use memory
     implicit none
     logical,intent(in) :: qrw ! Will reweighting happen or not?
     ! Local variables
     real(chm_real),allocatable,dimension(:)  :: scr0,scr1

     if(qrw) then ! if reweighting is happening, sort based on Vb - if it exists
        if (dA%n0 > 0) then 
           if (dA%q_bias0) then
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)    
              call chmalloc('freeene_calc.src','BARSORT','SCR1',dA%n0,crl=scr1)
              call barsort2(dA%n0,dA%Vb0,dA%dU0,scr0,scr1)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR1',dA%n0,crl=scr1)
           else 
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)        
              call barsort1(dA%n0,dA%dU0,scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)
           endif
        endif
        
        if (dA%n1 > 0) then 
           if (dA%q_bias1) then
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)    
              call chmalloc('freeene_calc.src','BARSORT','SCR1',dA%n1,crl=scr1)
              call barsort2(dA%n1,dA%Vb1,dA%dU1,scr0,scr1)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR1',dA%n1,crl=scr1)
           else 
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)        
              call barsort1(dA%n1,dA%dU1,scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)
           endif
        endif
     else ! if no reweighting is happening, sort based on dU
        if (dA%n0 > 0) then 
           if (dA%q_bias0) then
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)    
              call chmalloc('freeene_calc.src','BARSORT','SCR1',dA%n0,crl=scr1)
              call barsort2(dA%n0,dA%dU0,dA%Vb0,scr0,scr1)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR1',dA%n0,crl=scr1)
           else 
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)        
              call barsort1(dA%n0,dA%dU0,scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n0,crl=scr0)
           endif
        endif

        if (dA%n1 > 0) then 
           if (dA%q_bias1) then
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)    
              call chmalloc('freeene_calc.src','BARSORT','SCR1',dA%n1,crl=scr1)
              call barsort2(dA%n1,dA%dU1,dA%Vb1,scr0,scr1)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR1',dA%n1,crl=scr1)
           else 
              call chmalloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)        
              call barsort1(dA%n1,dA%dU1,scr0)
              call chmdealloc('freeene_calc.src','BARSORT','SCR0',dA%n1,crl=scr0)
           endif
        endif
     endif

  end subroutine barsort

 
  ! Sort a vector with potential energy data with merge sort (for better numerical stability)
  subroutine barsort1(nframes,d,buffer)
    implicit none

    integer, intent(in)  :: nframes
    real(chm_real),dimension(nframes),intent(inout)    :: d
    real(chm_real),dimension(nframes),intent(inout)    :: buffer

    integer              :: i,j,l,r,ende,width

    buffer(1:nframes) = 0.0

    ! Sort segments of increasing width and merge them with each other
    width=1
    do while(width < nframes)
       do i= 1,nframes,2*width
          l = i
          r = min(i+width,nframes)
          ende = min(i+2*width-1,nframes)
          do j=l,ende
             if(r>ende) then
                buffer(j) = d(l)
                l = l+1
             else if(l>=i+width) then
                buffer(j) = d(r)
                r = r+1
             else if(d(l) <= d(r)) then
                buffer(j) = d(l)
                l = l+1
             else
                buffer(j) = d(r)
                r = r+1
             endif
          enddo
       enddo
       ! Copy buffer to array
       d(1:nframes)  = buffer(1:nframes)
       width = 2*width
    enddo

  end subroutine barsort1


! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! Sort two columns of U with merge sort based on the values of column a.  (for better numerical stability)
  subroutine barsort2(nframes,cola,colb,scratcha,scratchb)
    use number
    implicit none
    integer, intent(in)  :: nframes
    real(chm_real),dimension(nframes), intent(inout)  :: cola, colb
    real(chm_real),dimension(nframes),intent(inout)   :: scratcha, scratchb
 
    integer              :: i,j,l,r,ende,width
 
    scratcha(1:nframes) = zero
    scratchb(1:nframes) = zero

   ! Sort segments of increasing width and merge them with each other
    width=1
    do while (width < nframes)
       do i= 1, nframes, 2*width
          l = i
          r = min(i+width,nframes)
          ende = min(i+2*width-1,nframes)
          do j=l,ende
             if (r>ende) then
                scratcha(j) = cola(l)
                scratchb(j) = colb(l)
                l = l+1
             else if (l>=i+width) then
                scratcha(j) = cola(r)
                scratchb(j) = colb(r)
                r = r+1 
             else if (cola(l) <= cola(r)) then
                scratcha(j) = cola(l)
                scratchb(j) = colb(l)
                l = l+1
             else
                scratcha(j) = cola(r)
                scratchb(j) = colb(r)
                r = r+1 
             end if
          end do          
       end do
       ! Copy buffer to array
       cola(1:nframes) =  scratcha(1:nframes)
       colb(1:nframes) =  scratchb(1:nframes) 
       width = 2*width
    end do

  end subroutine  barsort2

! --------------------------------------------------------------------------------------------------------------------------------------------------
  subroutine barhistogram(kt,ovl,maxovl,lrw)
    use stream
    use memory
    use number
    implicit none

    real(chm_real),intent(in)  :: kt
    real(chm_real),intent(out) :: ovl,maxovl  ! Overlap and dU with maximum overlap
    logical,intent(in)         :: lrw         ! reweight or not ?

    real(chm_real), dimension(4) :: minu ! Minimal energies of each column
    real(chm_real), dimension(4) :: maxu ! Maximum energies of each column
    real(chm_real)               :: valmaxovl  ! value of maximum ovlerlap
    real(chm_real)               :: avew0   ! average weight of trajectory 0
    real(chm_real)               :: avew1   ! average weight of trajectory 1
    real(chm_real), dimension(8) :: minmax ! list of minima and maxima involved
    real(chm_real)               :: binsize ! bin size for histogram
    real(chm_real)               :: histmin ! starting point of histogram
    real(chm_real)               :: histmax ! end point of histogram

    real(chm_real),allocatable,dimension(:) :: scr0,scr1   ! scratch arrays for weights
    real(chm_real),allocatable,dimension(:) :: hist0,hist1 ! histograms

    integer :: i,j
    integer :: bins     ! Number of bins for the histogram
    integer :: maxspace ! Max number of data points

    minu(1:4) = 0.0
    maxu(1:4) = 0.0
    minu(1) = MINVAL(dA%dU0(1:dA%n0))
    maxu(1) = MAXVAL(dA%dU0(1:dA%n0))
    minu(2) = MINVAL(dA%dU1(1:dA%n1))
    maxu(2) = MAXVAL(dA%dU1(1:dA%n1))

    if(dA%q_bias0) then
       minu(3) = MINVAL(dA%Vb0(1:dA%n0))
       maxu(3) = MAXVAL(dA%Vb0(1:dA%n0))
    else
       minu(3) = zero
       maxu(3) = zero
    endif
    if(dA%q_bias1) then
       minu(4) = MINVAL(dA%Vb1(1:dA%n1))
       maxu(4) = MAXVAL(dA%Vb1(1:dA%n1))
    else
       minu(4) = zero
       maxu(4) = zero
    endif

  
    call chmalloc('FREEENE_CALC.SRC','BARHISTOGRAM','SCR0',dA%n0,crl=scr0)
    call chmalloc('FREEENE_CALC.SRC','BARHISTOGRAM','SCR1',dA%n1,crl=scr1)

    scr0(1:dA%n0) = zero
    scr1(1:dA%n1) = zero
    
    ! Deal with biasing potentials 
    if (lrw) then    
       ! Replace biasing potentials by weights
       if(dA%q_bias0) then
          scr0(1:dA%n0) = exp((dA%Vb0(1:dA%n0)-maxu(3))/kt)
       else
          scr0(1:dA%n0) = 1.0
       endif

       if(dA%q_bias1) then
          scr1(1:dA%n1) = exp((dA%Vb1(1:dA%n1)-maxu(4))/kt)
       else
          scr1(1:dA%n1) = 1.0
       endif
       
       ! Calculate average weights
       call mean(scr0(1:dA%n0),dA%n0,avew0)
       call mean(scr1(1:dA%n1),dA%n1,avew1)
       
       ! Normalize weights
       scr0(1:dA%n0) = scr0(1:dA%n0)/avew0
       scr1(1:dA%n1) = scr1(1:dA%n1)/avew1
       
    else
       scr0(1:dA%n0) = 1.0
       scr1(1:dA%n1) = 1.0
    endif

    ! Get bounds of the histogram
    minmax(1:8) = 0.0
    minmax(1) = MINVAL(dA%dU0(1:dA%n0))
    minmax(2) = -MINVAL(dA%dU1(1:dA%n1))
    minmax(3) = MAXVAL(dA%dU0(1:dA%n0))
    minmax(4) = -MAXVAL(dA%dU1(1:dA%n1))

    ! Sort minima and maxima    (minmax(5:8) is just buffer)
    call barsort1(4,minmax(1:4),minmax(5:8))

    ! Maximum number of bins in histogram
    maxspace = 1003
    ! First, try using the global minimum and maximum dU
    histmin = real(floor(minmax(1)))
    histmax = real(ceiling(minmax(4)))

    ! Check if the array is large enough for full histogram - if not, use minimal bounds
    if(abs(histmax-histmin)/0.25 > real(maxspace-3) ) then
       histmin = real(floor(minmax(2)))
       histmax = real(ceiling(minmax(3)))
    endif

    ! if the boundaries are too close together, there is no overlap, or something is weird
    if(histmax-histmin < epsilon(histmin)) then
       valmaxovl = 0.0
       maxovl = histmin
       ovl = 0.0
       return
    endif

    ! Determine bin size of the histogram  (default =0.1 kcal/mol)
    ! Make sure that there are not more bins than space in the array
    binsize = MAX(1.0E-1,abs(histmax-histmin)/real(maxspace-3))
    bins = int((histmax-histmin)/binsize)+3 ! first and last bin reserved for non-overlap

    call chmalloc('FREEENE_CALC.SRC','BARHISTOGRAM','HIST0',bins,crl=hist0)
    call chmalloc('FREEENE_CALC.SRC','BARHISTOGRAM','HIST1',bins,crl=hist1)

    hist0(1:bins) = zero
    hist1(1:bins) = zero

    ! Histogram of the forward perturbation
    do i=1,dA%n0
       ! Data outside the histogram region gets stored in first or last field of array
       j = MAX(nint((dA%dU0(i)-histmin)/binsize)+2,1)
       j = MIN(j,bins)
       hist0(j) = hist0(j)+scr0(i)/real(dA%n0)
    enddo
    
    ! Histogram of the backward perturbation
    do i=1,dA%n1
       ! Data outside the histogram region gets stored in first and last field of array
       j = MAX(nint((-dA%dU1(i)-histmin)/binsize)+2,1)
       j = MIN(j,bins)
       hist1(j) = hist1(j)+scr1(i)/real(dA%n1)
    enddo
    
    ! Calculate overlap
    valmaxovl = 0.0
    maxovl = 0.0
    ovl = 0.0

    if(prnlev > 6) then
       if (lrw) then
          write(outu,'(a)') 'BARHISTOGRAM> REWEIGHTED OVERLAP HISTOGRAM'
       else 
          write(outu,'(a)') 'BARHISTOGRAM> OVERLAP HISTOGRAM'
       end if
       write(outu,'(3a14)') 'dU','p0','p1'
    endif

    do i=2,bins-1
       if (prnlev > 6)  write (outu,'(3f14.3)') real(i-2)*binsize+histmin,hist0(i),hist1(i)
       if (hist0(i) == 0.0 .or. hist1(i) == 0.0) cycle
       hist0(i) =  2.0*hist0(i)*hist1(i)/(hist0(i)+hist1(i))
       ovl = ovl + hist0(i)
       if(hist0(i)>valmaxovl) then
          valmaxovl = hist0(i)
          maxovl = histmin + real(I-2)*binsize
       endif
    enddo

    ! clean up
    call chmdealloc('FREEENE_CALC.SRC','BARHISTOGRAM','SCR0',dA%n0,crl=scr0)
    call chmdealloc('FREEENE_CALC.SRC','BARHISTOGRAM','SCR1',dA%n1,crl=scr1)
    call chmdealloc('FREEENE_CALC.SRC','BARHISTOGRAM','HIST0',bins,crl=hist0)
    call chmdealloc('FREEENE_CALC.SRC','BARHISTOGRAM','HIST1',bins,crl=hist1)

  end subroutine barhistogram

! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! Do BAR/NBB iterations
  subroutine barit(temp,qrw,daguess,newbar)
     use consta,only:kboltz
     use number
     use stream
     use memory
     implicit none

     real(chm_real),intent(in)    :: temp,daguess
     real(chm_real),intent(out)   :: newbar
     logical,intent(in)           :: qrw ! Do reweighting?

    integer           :: maxbarit ! Maximum number of BAR iterations
    real(chm_real)    :: tolbar   ! Tolerance of BAR iteration convergence
    real(chm_real)    :: avew0    ! average weight of trajectory 0
    real(chm_real)    :: avew1    ! average weight of trajectory 1
    real(chm_real)    :: offvb0   ! offset for biasing potential 0
    real(chm_real)    :: offvb1   ! offset for biasing potential 1
    real(chm_real)    :: C        ! BAR result of previous iteration / offset for BAR calculation
    real(chm_real)    :: fw       ! forward perturbation Fermi average
    real(chm_real)    :: bw       ! backward  perturbation Fermi average
    real(chm_real)    :: var      ! internal variance estimate
    real(chm_real)    :: testfw   ! forward perturbtation contribution to the variance estimate
    real(chm_real)    :: testbw   ! backward perturbtation contribution to the variance estimate
    real(chm_real)    :: avg_dU0, avg_dU1, sdv_dU0, sdv_dU1, avg_bias0, avg_bias1
    real(chm_real)    :: sdv_bias0, sdv_bias1
    integer           :: i,j

    real(chm_real)                          :: kt
    real(chm_real),allocatable,dimension(:) :: scr_du0,scr_du1,scr_w0,scr_w1

    if(prnlev > 5) write(outu,'(a,f14.4)') "BAR> Initial free energy guess used for C ", daguess

    kt = kboltz*temp

    call mean(dA%dU0,dA%n0,avg_dU0)
    call mean(dA%dU1,dA%n1,avg_dU1)
    call stddev(dA%dU0,dA%n0,avg_dU0,sdv_dU0)
    call stddev(dA%dU1,dA%n1,avg_dU1,sdv_dU1)

    call chmalloc('freeene_calc.src','BARIT','SCR_DU0',dA%n0,crl=scr_du0)
    call chmalloc('freeene_calc.src','BARIT','SCR_DU1',dA%n1,crl=scr_du1)
    call chmalloc('freeene_calc.src','BARIT','SCR_W0',dA%n0,crl=scr_w0)
    call chmalloc('freeene_calc.src','BARIT','SCR_W1',dA%n1,crl=scr_w1)

    if(qrw .and. dA%q_bias0) then
       call mean(dA%Vb0,dA%n0,avg_bias0)
       call stddev(dA%Vb0,dA%n0,avg_bias0,sdv_bias0)      
    else
       avg_bias0 = zero
       sdv_bias0 = zero
       scr_w0(1:dA%n0) = 1.0
    endif
    if(qrw .and. dA%q_bias1) then
       call mean(dA%Vb1,dA%n1,avg_bias1)
       call stddev(dA%Vb1,dA%n1,avg_bias1,sdv_bias1)
    else
       avg_bias1 = zero
       sdv_bias1 = zero
       scr_w1(1:dA%n1) = 1.0
    endif

    ! Determine the offsets for the biasing potential - if the fluctuations are large, use maximum value
    ! to avoid overflows
    if(qrw) then
       if(dA%q_bias0) then
          if(sdv_bias0 < 10) then
             offvb0 = avg_bias0
          else
             offvb0 = maxval(dA%Vb0(1:dA%n0)) - 8.0
          endif
       else
          offvb0 = zero
       endif
       if(dA%q_bias1) then
          if(sdv_bias1 < 10) then
             offvb1 = avg_bias1
          else
             offvb1 = maxval(dA%Vb1(1:dA%n1)) - 8.0
          endif
       else
          offvb1 = zero
       endif

       if(prnlev > 5) &
          write(outu,'(a,f14.4,a,f14.4,a)') "BAR> Using offsets of ", offvb0 , " for Vb0 and" , offvb1, " for Vb1"

       ! Replace biasing potentials by weights
       scr_w0(1:dA%n0) = exp((dA%Vb0(1:dA%n0)-offvb0)/kt)
       scr_w1(1:dA%n1) = exp((dA%Vb1(1:dA%n1)-offvb1)/kt)

    endif

    ! Calculate average weights
    call mean(scr_w0(1:dA%n0),dA%n0,avew0)
    call mean(scr_w1(1:dA%n1),dA%n1,avew1)
 
    if(prnlev > 5) write(outu,'(a,f14.4,a,f14.4,a)') "BAR> Average weights are ", avew0 , " for 0 and ", &
                   avew1, " for 1"

    ! Divide all weights by average weight to get normalized weights
    scr_w0(1:dA%n0) = scr_w0(1:dA%n0)/avew0
    scr_w1(1:dA%n1) = scr_w1(1:dA%n1)/avew1

    ! convergence tolerance -- ToDo make these abd number of iterations user-settable
    tolbar = 0.000001
 
    ! Maximum number of iterations
    maxbarit = 500
    newbar = 0.0
    C = daguess


 
    scr_du0(1:dA%n0) = (dA%dU0(1:dA%n0) )
    scr_du1(1:dA%n1) = (dA%dU1(1:dA%n1) )

    do j=1,maxbarit

       ! Subtract the free energy estimate
       scr_du0(1:dA%n0) = (dA%dU0(1:dA%n0) - C)/kt
       scr_du1(1:dA%n1) = (dA%dU1(1:dA%n1) + C)/kt

       ! Calculate ensemble average for trajectory 0 
       fw = FERMI(scr_du0(1))*scr_w0(1)
       do i=2,dA%n0
          fw = fw + (FERMI(scr_du0(i))*scr_w0(i)-fw)/real(i)
       enddo
       
       ! Calculate ensemble average for trajectory 1 
       bw = FERMI(scr_du1(1))*scr_w1(1)
       do i=2,dA%n1
          bw = bw + (FERMI(scr_du1(i))*scr_w1(i)-bw)/real(i)
       enddo
      
       ! New free energy estimate
       newbar = -kt*log(fw/bw) + c 

       if(prnlev > 5) write(outu,'(a,i4,a,f14.4,a,f14.4,a,f14.4)') "BAR> ITERATION ", j, " RATIO ", newbar , " FW ", fw, " BW ", bw
       if(fw/bw < epsilon(fw) .or. bw/fw < epsilon(bw)) then
          write (outu,'(a,f14.4,a,f14.4)') &
                'CALCULATION IS BECOMING NUMERICALLY UNSTABLE DUE TO WRONG C! CURRENT BAR RESULT: ', newbar, ' BAR RATIO ', fw/bw
       endif

       if (abs(newbar-c) < tolbar) exit
       c = newbar
    enddo

    if (j == maxbarit) then
       if(prnlev > 3) then
          write(outu,'(a)') "BAR> Warning: Something is wrong. Check the free energy data, check your simulation."
          write(outu,'(a)') "BAR> Try increasing overlap (more lambda points) or adding more data (longer simulations)."
          write(outu,'(a)') "BAR> You can also try to provide a better initial free energy guess."
       endif
       call wrndie(-2,'<BAR>','BAR COULD NOT CONVERGE AFTER 500 ITERATIONS')
    endif

    ! deallocate memory
    call chmdealloc('freeene_calc.src','BARIT','SCR_DU0',dA%n0,crl=scr_du0)
    call chmdealloc('freeene_calc.src','BARIT','SCR_DU1',dA%n1,crl=scr_du1)
    call chmdealloc('freeene_calc.src','BARIT','SCR_W0',dA%n0,crl=scr_w0)
    call chmdealloc('freeene_calc.src','BARIT','SCR_W1',dA%n1,crl=scr_w1)


  end subroutine barit




! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! Do BAR/NBB iterations on a subset of the data 
  subroutine subbarit(temp,qrw,daguess,newbar,start0,stop0,start1,stop1)
    use consta,only:kboltz
    use number
    use stream
    use memory
    implicit none

    real(chm_real),intent(in)    :: temp,daguess
    real(chm_real),intent(out)   :: newbar
    logical,intent(in)           :: qrw ! Do reweighting?
    integer,intent(in)           :: start0,stop0, start1,stop1 
   
    integer           :: maxbarit ! Maximum number of BAR iterations
    real(chm_real)    :: tolbar   ! Tolerance of BAR iteration convergence
    real(chm_real)    :: avew0    ! average weight of trajectory 0
    real(chm_real)    :: avew1    ! average weight of trajectory 1
    real(chm_real)    :: offvb0   ! offset for biasing potential 0
    real(chm_real)    :: offvb1   ! offset for biasing potential 1
    real(chm_real)    :: C        ! BAR result of previous iteration / offset for BAR calculation
    real(chm_real)    :: fw       ! forward perturbation Fermi average
    real(chm_real)    :: bw       ! backward  perturbation Fermi average
    real(chm_real)    :: var      ! internal variance estimate
    real(chm_real)    :: testfw   ! forward perturbtation contribution to the variance estimate
    real(chm_real)    :: testbw   ! backward perturbtation contribution to the variance estimate
    real(chm_real)    :: avg_dU0, avg_dU1, sdv_dU0, sdv_dU1, avg_bias0, avg_bias1
    real(chm_real)    :: sdv_bias0, sdv_bias1
    integer           :: i,j

    real(chm_real)                          :: kt
    real(chm_real),allocatable,dimension(:) :: scr_du0,scr_du1,scr_w0,scr_w1
    integer           :: newn0,newn1 

    if(prnlev > 5) write(outu,'(a,f14.4)') "BAR> Initial free energy guess used for C ", daguess

    kt = kboltz*temp

    newn0 = stop0 - start0 + 1  
    newn1 = stop1 - start1 + 1 

    call mean(dA%dU0(start0:stop0),newn0,avg_dU0)
    call mean(dA%dU1(start1:stop1),newn1,avg_dU1)
    call stddev(dA%dU0(start0:stop0),newn0,avg_dU0,sdv_dU0)
    call stddev(dA%dU1(start1:stop1),newn1,avg_dU1,sdv_dU1)

    call chmalloc('freeene_calc.src','BARIT','SCR_DU0',newn0,crl=scr_du0)
    call chmalloc('freeene_calc.src','BARIT','SCR_DU1',newn1,crl=scr_du1)
    call chmalloc('freeene_calc.src','BARIT','SCR_W0',newn0,crl=scr_w0)
    call chmalloc('freeene_calc.src','BARIT','SCR_W1',newn1,crl=scr_w1)

    if(qrw .and. dA%q_bias0) then
       call mean(dA%Vb0(start0:stop0),newn0,avg_bias0)
       call stddev(dA%Vb0(start0:stop0),newn0,avg_bias0,sdv_bias0)      
    else
       avg_bias0 = zero
       sdv_bias0 = zero
       scr_w0(1:newn0) = 1.0
    endif
    if(qrw .and. dA%q_bias1) then
       call mean(dA%Vb1(start1:stop1),newn1,avg_bias1)
       call stddev(dA%Vb1(start1:stop1),newn1,avg_bias1,sdv_bias1)
    else
       avg_bias1 = zero
       sdv_bias1 = zero
       scr_w1(1:newn1) = 1.0
    endif

    ! Determine the offsets for the biasing potential - if the fluctuations are large, use maximum value
    ! to avoid overflows
    if(qrw) then
       if(dA%q_bias0) then
          if(sdv_bias0 < 10.0) then
             offvb0 = avg_bias0
          else
             offvb0 = maxval(dA%Vb0(start0:stop0)) - 8.0
          endif
       else
          offvb0 = zero
       endif
       if(dA%q_bias1) then
          if(sdv_bias1 < 10.0) then
             offvb1 = avg_bias1
          else
             offvb1 = maxval(dA%Vb1(start1:stop1)) - 8.0
          endif
       else
          offvb1 = zero
       endif

       if(prnlev > 5) &
            write(outu,'(a,f14.4,a,f14.4,a)') "BAR> Using offsets of ", offvb0 , " for Vb0 and" , offvb1, " for Vb1"

       ! Replace biasing potentials by weights
       scr_w0(1:newn0) = exp((dA%Vb0(start0:stop0)-offvb0)/kt)
       scr_w1(1:newn1) = exp((dA%Vb1(start1:stop1)-offvb1)/kt)

    endif

    ! Calculate average weights
    call mean(scr_w0(1:newn0),newn0,avew0)
    call mean(scr_w1(1:newn1),newn1,avew1)

    if(prnlev > 5) write(outu,'(a,f14.4,a,f14.4,a)') "BAR> Average weights are ", avew0 , " for 0 and ", &
         avew1, " for 1"

    ! Divide all weights by average weight to get normalized weights
    scr_w0(1:newn0) = scr_w0(1:newn0)/avew0
    scr_w1(1:newn1) = scr_w1(1:newn1)/avew1

    ! convergence tolerance -- ToDo make these abd number of iterations user-settable
    tolbar = 0.000001

    ! Maximum number of iterations
    maxbarit = 500
    newbar = 0.0
    C = daguess
    
    scr_du0(1:newn0) = (dA%dU0(start0:stop0) )
    scr_du1(1:newn1) = (dA%dU1(start1:stop1) )

    do j=1,maxbarit

       ! Subtract the free energy estimate
       scr_du0(1:newn0) = (dA%dU0(start0:stop0) - C)/kt
       scr_du1(1:newn1) = (dA%dU1(start1:stop1) + C)/kt

       ! Calculate ensemble average for trajectory 0 
       fw = FERMI(scr_du0(1))*scr_w0(1)
       do i=2,newn0
          fw = fw + (FERMI(scr_du0(i))*scr_w0(i)-fw)/real(i)
       enddo

       ! Calculate ensemble average for trajectory 1 
       bw = FERMI(scr_du1(1))*scr_w1(1)
       do i=2,newn1
          bw = bw + (FERMI(scr_du1(i))*scr_w1(i)-bw)/real(i)
       enddo

       ! New free energy estimate
       newbar = -kt*log(fw/bw) + c 

       if(prnlev > 5) write(outu,'(a,i4,a,f14.4,a,f14.4,a,f14.4)') "BAR> ITERATION ", j, " RATIO ", newbar , " FW ", fw, " BW ", bw
       if(fw/bw < epsilon(fw) .or. bw/fw < epsilon(bw)) then
          write (outu,'(a,f14.4,a,f14.4)') &
               'CALCULATION IS BECOMING NUMERICALLY UNSTABLE DUE TO WRONG C! CURRENT BAR RESULT: ', newbar, ' BAR RATIO ', fw/bw
       endif

       if (abs(newbar-c) < tolbar) exit
       c = newbar
    enddo

    if (j == maxbarit) then
       if(prnlev > 3) then
          write(outu,'(a)') "BAR> Warning: Something is wrong. Check the free energy data, check your simulation."
          write(outu,'(a)') "BAR> Try increasing overlap (more lambda points) or adding more data (longer simulations)."
          write(outu,'(a)') "BAR> You can also try to provide a better initial free energy guess."
       endif
       call wrndie(-2,'<BAR>','BAR COULD NOT CONVERGE AFTER 500 ITERATIONS')
    endif

    ! deallocate memory
    call chmdealloc('freeene_calc.src','BARIT','SCR_DU0',newn0,crl=scr_du0)
    call chmdealloc('freeene_calc.src','BARIT','SCR_DU1',newn1,crl=scr_du1)
    call chmdealloc('freeene_calc.src','BARIT','SCR_W0',newn0,crl=scr_w0)
    call chmdealloc('freeene_calc.src','BARIT','SCR_W1',newn1,crl=scr_w1)


  end subroutine subbarit





  ! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! Calculate free energy difference with Zwanzig/Exponential Formula/"Free energy perturbation"
  ! and, optionally, with reweighting  
  subroutine nbzwanzig(nframes,tp,ener_offset,bias_offset,delta,qrw,bias,zwanzig)
    use consta,only:kboltz
    use stream
    use number
    implicit none

    integer,intent(in)                     :: nframes
    real(chm_real),intent(in)              :: tp         ! temperatures
    real(chm_real),intent(in)              :: ener_offset,bias_offset
    real(chm_real),intent(in),dimension(:) :: delta,bias
    logical,intent(in)                     :: qrw        ! do we reweight?
    real(chm_real),intent(out)             :: zwanzig    ! result

    ! local variables
    real(chm_real) :: kt, ediff, avew
    integer        :: i

    if (prnlev .gt. 6 ) write(outu,'(a,f12.6)') ' ZWANZIG> ENERGY OFFSET IS ', ener_offset         

    kt = kboltz*tp
    avew = zero
    !write(*,*) "OFFSET"  , ener_offset ,  " MIN " , MINVAL(delta(1:nframes))   ! , " MAX " , MAXVAL(delta) ! , " first" , delta(1)

    if(qrw) then ! For reweighting
       ! Calculate average weight (subtracting an offset for numerical reasons)
       avew = exp((bias(1)-bias_offset)/kt)
       do i=2,nframes
          avew = avew + (exp((bias(i)-bias_offset)/kt)-avew)/real(i)
       enddo
       !write(outu,'(a,f12.6)') 'NBZWANZIG> AVERAGE BIAS (ACCOUNTING FOR OFFSET) IS ',avew        

       zwanzig = exp((bias(1)-bias_offset)/kt)/avew * exp(-(delta(1)-ener_offset)/kt)
       do i=2,nframes
          zwanzig = zwanzig + ((exp((bias(i)-bias_offset)/kt)/avew*(exp(-(delta(i)-ener_offset)/kt)))-zwanzig)/real(i)
       enddo
       zwanzig = -kt * log(zwanzig) + ener_offset

    else ! Normal Zwanzig
       zwanzig = exp(-(delta(1)-ener_offset)/kt)
       do i=2,nframes
          zwanzig = zwanzig + (exp(-(delta(i)-ener_offset)/kt)-zwanzig)/real(i)
       enddo
       zwanzig = -kt * log(zwanzig) + ener_offset

   endif

  

  end subroutine nbzwanzig


  ! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! check that data has been input correctly and consistently
  subroutine datacheck(tp,qbias0,qbias1,maxovl,zwanzig,qrw,qnosort,qnocheck)
    use stream
    use number
    use consta,only: kboltz
    use param_store, only: set_param
    implicit none

    real(chm_real),intent(in)  :: tp                ! temperature
    real(chm_real),intent(out) :: maxovl            ! position for max overlap - may be needed as initial BAR guess
    logical,intent(in)         :: qbias0,qbias1      ! what are we doing
    integer,intent(in)         :: zwanzig            ! = 0, no Zwanzig, = 1 forward, = 2 backwards
    logical,intent(in)         :: qrw ! Will reweighting happen or not?
    logical,intent(in)         :: qnosort ! Will the data be sorted? 
    logical,intent(in)         :: qnocheck !  Check for outliers 

    ! local variables
    real(chm_real)                         :: avg_dU0,avg_dU1,sdv_dU0,sdv_dU1
    real(chm_real)                         :: avg_Vb0,avg_Vb1,sdv_Vb0,sdv_Vb1
    real(chm_real)                         :: kt,fzwanzig,bzwanzig,ovl

    kt = kboltz*tp

    ! calculate averages and standard deviations - remove corrupt data (outliers)
    avg_dU0 = zero 
    avg_dU1 = zero 
    sdv_dU0 = zero 
    sdv_dU1 = zero 

    ! If we do BAR, check everything
    if (.not. qnocheck ) then
       if(zwanzig == 0) then 
          if (dA%q_bias0) then
             call outlier2(dA%dU0,dA%Vb0,dA%n0,0)
             call outlier2(dA%Vb0,dA%dU0,dA%n0,0)           
          else 
             call outlier1(dA%dU0,dA%n0,0)
          endif
          if (dA%q_bias1) then
             call outlier2(dA%dU1,dA%Vb1,dA%n1,1)
             call outlier2(dA%Vb1,dA%dU1,dA%n1,1)           
          else 
             call outlier1(dA%dU1,dA%n1,1)
          endif
       endif
       
       ! Otherwise, just check data that is going to be used
       if(zwanzig == 1) then 
          if (dA%q_bias0) then
             call outlier2(dA%dU0,dA%Vb0,dA%n0,0)
             call outlier2(dA%Vb0,dA%dU0,dA%n0,0)           
          else 
             call outlier1(dA%dU0,dA%n0,0)
          endif
       endif
       
       if(zwanzig == 2) then 
          if (dA%q_bias1) then
             call outlier2(dA%dU1,dA%Vb1,dA%n1,1)
             call outlier2(dA%Vb1,dA%dU1,dA%n1,1)           
          else 
             call outlier1(dA%dU1,dA%n1,1)
          endif
       endif
    endif
    
    ! Do a quick sanity check on the fluctuations if we're doing Zwanzig
    if(zwanzig == 1) then
       if(sdv_dU0 > 3*kt) &
            call wrndie(1,'<DATACHECK>','FLUCTUATION OF FORWARD PERTURBATION IS > 3 KT')
    else if(zwanzig == 2) then
       if(sdv_dU1 > 3*kt) &
            call wrndie(1,'<DATACHECK>','FLUCTUATION OF BACKWARD PERTURBATION IS > 3 KT')
    else if(zwanzig /= 0) then
       call wrndie(-4,'<DATACHECK>','UNKNOWN CALCULATION TYPE - YOU ARE ON YOUR OWN.')
       return
    endif

    if(avg_dU0*avg_dU1 > 0.0) &
         call wrndie(1,'<BAR>','SAME SIGN OF AVERAGE FORWARD AND BACKWARD PERTURBATIONS')

    if(qbias0) then
       if(dA%q_bias0) then
          call mean(dA%Vb0,dA%n0,avg_Vb0)
          call stddev(dA%Vb0,dA%n0,avg_Vb0,sdv_Vb0)

          if(sdv_Vb0 > 3*kt) &
               call wrndie(1,'<DATACHECK>','FLUCTUATIONS OF BIAS 0 > 3 KT')
       else
          call wrndie(-4,'<DATACHECK>','BIAS REQUESTED WITH NO DATA')
       endif
    endif
    if(qbias1) then
       if(dA%q_bias1) then
          call mean(dA%Vb1,dA%n1,avg_Vb1)
          call stddev(dA%Vb1,dA%n1,avg_Vb1,sdv_Vb1)

          if(sdv_Vb1 > 3*kt) &
               call wrndie(1,'<DATACHECK>','FLUCTUATIONS OF BIAS 1 > 3 KT')
       else
          call wrndie(-4,'<DATACHECK>','BIAS REQUESTED WITH NO DATA')
       endif
    endif

    ! Sort data for numerical stability
    if(.not. qnosort) call barsort(qrw)

    ! Calculate overlap
    if(zwanzig == 0) then
       call barhistogram(kt,ovl,maxovl,.false.)
       if(prnlev >= 5) &
            write(outu,'(a,f7.3,a)') 'DATACHECK> OVERLAP BETWEEN END STATES IS ',real(ovl*100.0),'%'
       if(qrw) then
          call barhistogram(kt,ovl,maxovl,.true.)
          if(prnlev >= 5) &
               write(outu,'(a,f7.3,a)') 'DATACHECK> OVERLAP BETWEEN END STATES *AFTER* REWEIGHTING IS ',real(ovl*100.0),'%'
       endif
       ! Save overlap to variable 
       !CALL SETMSR('FOVL',ovl*100.0)
       CALL SET_PARAM('FOVL',ovl*100.0)

       if(ovl<0.01) call wrndie(0,'<BAR>','OVERLAP BETWEEN STATES IS TOO SMALL - ADD MORE LAMBDA POINTS')
    endif

  end subroutine datacheck



  ! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! --------------------------------------------------------------------------------------------------------------------------------------------------
  ! This is the main driver routine that drives
  ! the calculator command loop.
  subroutine frencalc(comlyn,comlen)
    use stream
    use string
    use number
    use memory,only:chmalloc,chmdealloc,chmrealloc
    use param_store, only: set_param

    implicit none

    character,intent(inout) :: comlyn*(*)
    integer,intent(inout)   :: comlen

    logical                                 :: lused,eof,qdelta,qback,qforw,q_userw
    logical                                 :: qrw, qnocheck, qnosort , qnoof, qmedian 
    real(chm_real),allocatable,dimension(:) :: scr0,scr1 ! scratch arrays
    real(chm_real)                          :: duoffset, vboffset 
    real(chm_real)                          :: result,overlap,maxovl,temp,barguess,stddev 
    character(len=4)                        :: wrd
    integer                                 :: npt,un0,un1,cl0,cl1,sk0,sk1,of0,of1
    integer                                 :: blocks   ! Number of blocks for calculation of standard deviation 
    real(chm_real),allocatable,dimension(:) :: dablocks ! free energy results of blocks 
    ! Now we specify unit numbers (u) for a couple of possible states
    integer :: du0u,du1u,vb0u,vb1u,u00u,u01u,u10u,u11u,us0u,us1u,uewr
    integer :: i,j 
    
    ! Initialize flags 
    qrw = .false. 
    qnocheck = .false. 
    qnosort = .false. 
    qnoof=.false. 
    qmedian=.false. 

    if(iolev .le. 0) return

    call xtrane(comlyn,comlen,'FREENE')

    dA%n0 = 0
    dA%n1 = 0
    eof=.false.
    do while(.not.eof)
       lused=.true.
       do while(lused .and. (.not.eof))
          call rdcmnd(comlyn,mxcmsz,comlen,istrm,eof,.true., &
               .true.,'FREEENE> ')
          call miscom(comlyn,mxcmsz,comlen,lused)
          !!write(outu,'(a,2l1)') 'FREEENE> lused, eof = ',lused,eof
       enddo
       wrd=nexta4(comlyn,comlen)

       ! ToDo: for now this routine only allows the setting
       ! up of two states (0 and 1). Moving forward, we want
       ! to have a flexible way of setting up lambda intermediates
       ! and calculating a total free energy difference.

       if(wrd == 'END')  then
          write(outu,'(a)') 'CHARMM> CLOSING FREE ENERGY SUBCOMMAND PARSER'
          if(allocated(scr0)) call chmdealloc('freeene_calc.src','FRENCALC','SCR0',size(scr0),crl=scr0)
          if(allocated(scr1)) call chmdealloc('freeene_calc.src','FRENCALC','SCR1',size(scr1),crl=scr1)
          call free_diff()
          return
#if KEY_GERHARD==1
          ! Create an MNDO file 
       else if(wrd == 'MNDO') then
          CALL MKMNDO(COMLYN,COMLEN)
#endif
          ! Load data 
       else if(wrd == 'LOAD') then

          ! All possible options of potential energies/differences/biasing potentials 
          du0u=gtrmi(comlyn,comlen,'DU0',-1)  
          du1u=gtrmi(comlyn,comlen,'DU1',-1)
          vb0u=gtrmi(comlyn,comlen,'VB0',-1)
          vb1u=gtrmi(comlyn,comlen,'VB1',-1)
          u00u=gtrmi(comlyn,comlen,'U00',-1)
          u01u=gtrmi(comlyn,comlen,'U01',-1)
          u10u=gtrmi(comlyn,comlen,'U10',-1)
          u11u=gtrmi(comlyn,comlen,'U11',-1)
          us0u=gtrmi(comlyn,comlen,'US0',-1)
          us1u=gtrmi(comlyn,comlen,'US1',-1)
          uewr=gtrmi(comlyn,comlen,'UEWR',-1)

          ! Alternative set of commands 
          du0u=gtrmi(comlyn,comlen,'DFW',du0u)  
          du1u=gtrmi(comlyn,comlen,'DBW',du1u)
          vb0u=gtrmi(comlyn,comlen,'VB0',vb0u)
          vb1u=gtrmi(comlyn,comlen,'VB1',vb1u)
          u00u=gtrmi(comlyn,comlen,'U0L0',u00u)
          u01u=gtrmi(comlyn,comlen,'U0L1',u01u)
          u10u=gtrmi(comlyn,comlen,'U1L0',u10u)
          u11u=gtrmi(comlyn,comlen,'U1L1',u11u)
          us0u=gtrmi(comlyn,comlen,'S0L0',us0u)
          us1u=gtrmi(comlyn,comlen,'S1L1',us1u)

          if(MIN(du0u,du1u,vb0u,vb1u,u00u,u01u,u10u,u11u,us0u,us1u) < -1) call wrndie(-5,'<FRENCALC>','INVALID UNIT')

          npt=gtrmi(comlyn,comlen,'NPT',-1)  ! Number of points 
          if(npt == 0) call wrndie(-4,'<FRENCALC>','NO DATA POINTS?')

          cl0=gtrmi(comlyn,comlen,'COLU', 1)  ! Which column contains the data?
          if(cl0 < 0) call wrndie(-4,'<FRENCALC>','NO NEGATIVE COLUMN NUMBERS')

          sk0 = gtrmi(comlyn,comlen,'SKIP',1) ! Skip rate 
          if(sk0 <= 0) call wrndie(-4,'<FRENCALC>','NO SUCH SKIP RATES')

          of0 = gtrmi(comlyn,comlen,'OFFS',0) ! Offset for reading data 
          if(of0 < 0) call wrndie(-4,'<FRENCALC>','NO NEGATIVE OFFSETS')

          if((du0u .ne. -1) .and. (.not. allocated(dA%dU0))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%dU0',npt,crl=dA%dU0)
             call read_column_data(du0u,cl0,sk0,of0,npt,dA%dU0)
             dA%ndU0=npt 
             dA%n0=npt
             qforw = .true. 
          endif

          if((du1u .ne. -1) .and. (.not. allocated(dA%dU1))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%dU1',npt,crl=dA%dU1)
             call read_column_data(du1u,cl0,sk0,of0,npt,dA%dU1)
             dA%ndU1=npt
             dA%n1=npt
             qback = .true.
          endif

          if((vb0u .ne. -1) .and. (.not. allocated(dA%vb0))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%vb0',npt,crl=dA%vb0)
             call read_column_data(vb0u,cl0,sk0,of0,npt,dA%vb0)
             dA%nVb0=npt
             dA%n0=npt
             dA%q_bias0 = .true.
          endif

          if((vb1u .ne. -1) .and. (.not. allocated(dA%vb1))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%vb1',npt,crl=dA%vb1)
             call read_column_data(vb1u,cl0,sk0,of0,npt,dA%vb1)
             dA%nVb1=npt
             dA%n1=npt
             dA%q_bias1 = .true.
          endif

          if (uewr .ne. -1) then
             u00u = uewr 
             cl0 = 2
          endif

          if((u00u .ne. -1) .and. (.not. allocated(dA%u00))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%u00',npt,crl=dA%u00)
             call read_column_data(u00u,cl0,sk0,of0,npt,dA%u00)
             dA%nU00=npt
             dA%n0=npt
          endif

          if (uewr .ne. -1) then
             u01u = uewr 
             cl0 = 3
          endif

          if((u01u .ne. -1) .and. (.not. allocated(dA%u01))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%u01',npt,crl=dA%u01)
             call read_column_data(u01u,cl0,sk0,of0,npt,dA%u01)
             dA%nU01=npt
             dA%n0=npt
          endif

          if (uewr .ne. -1) then
             u10u = uewr 
             cl0 = 4
          endif

          if((u10u .ne. -1) .and. (.not. allocated(dA%u10))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%u10',npt,crl=dA%u10)
             call read_column_data(u10u,cl0,sk0,of0,npt,dA%u10)
             dA%nU10=npt
             dA%n1=npt
          endif

          if (uewr .ne. -1) then
             u11u = uewr 
             cl0 = 5
          endif

          if((u11u .ne. -1) .and. (.not. allocated(dA%u11))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%u11',npt,crl=dA%u11)
             call read_column_data(u11u,cl0,sk0,of0,npt,dA%u11)
             dA%nU11=npt
             dA%n1=npt
          endif

          if((us0u .ne. -1) .and. (.not. allocated(dA%us0))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%us0',npt,crl=dA%us0)
             call read_column_data(us0u,cl0,sk0,of0,npt,dA%us0)
             dA%nUS0=npt
             dA%n0=npt
             dA%q_bias0 = .true.
          endif

          if((us1u .ne. -1) .and. (.not. allocated(dA%us1))) then 
             if (npt .ne. -1) call chmalloc('freeene_calc.src','FRENCALC','dA%us1',npt,crl=dA%us1)
             call read_column_data(us1u,cl0,sk0,of0,npt,dA%us1)
             dA%nUS1=npt
             dA%n1=npt
             dA%q_bias1 = .true.
          endif

          ! Check if we can generate dU's from U data 
          if (.not. allocated(dA%dU0) .and. allocated(dA%U00) .and. allocated(dA%U01)) then 
             !check for consistency
             if (dA%nU00 .eq. dA%nU01) then
                dA%ndU0 = dA%nU00
                call chmalloc('freeene_calc.src','FRENCALC','dA%dU0',dA%ndU0,crl=dA%dU0)
                dA%dU0(1:dA%ndU0) = dA%U01(1:dA%ndU0) - dA%U00(1:dA%ndU0)
                qforw = .true. 
             else 
                call wrndie(-4,'<FRENCALC>','U00 AND U01 HAVE INCONSISTENT LENGTH')
             endif
          endif

          if (.not. allocated(dA%dU1) .and. allocated(dA%U10) .and. allocated(dA%U11)) then 
             !check for consistency
             if (dA%nU10 .eq. dA%nU11) then
                dA%ndU1 = dA%nU11
                call chmalloc('freeene_calc.src','FRENCALC','dA%dU1',dA%ndU1,crl=dA%dU1)
                dA%dU1(1:dA%ndU1) = dA%U10(1:dA%ndU1) - dA%U11(1:dA%ndU1)                 
                qback = .true. 
             else 
                call wrndie(-4,'<FRENCALC>','U10 AND U11 HAVE INCONSISTENT LENGTH')
             endif
          endif

          ! Check if we can generate Vb's from U data 
          if (.not. allocated(dA%Vb0) .and. allocated(dA%U00) .and. allocated(dA%US0)) then 
             !check for consistency
             if (dA%nU00 .eq. dA%nUS0) then
                dA%nVb0 = dA%nUS0
                call chmalloc('freeene_calc.src','FRENCALC','dA%Vb0',dA%nVb0,crl=dA%Vb0)
                dA%Vb0(1:dA%nVb0) = dA%US0(1:dA%nVb0) - dA%U00(1:dA%nVb0)
                dA%q_bias0 = .true.
             else 
                call wrndie(-4,'<FRENCALC>','US0 AND U00 HAVE INCONSISTENT LENGTH')
             endif
          endif

          if (.not. allocated(dA%Vb1) .and. allocated(dA%U11) .and. allocated(dA%US1)) then 
             !check for consistency
             if (dA%nUS1 .eq. dA%nU11) then
                dA%nVb1 = dA%nUS1
                call chmalloc('freeene_calc.src','FRENCALC','dA%Vb1',dA%nVb1,crl=dA%Vb1) 
                dA%Vb1(1:dA%nVb1) = dA%US1(1:dA%nVb1) - dA%U11(1:dA%nVb1)
                dA%q_bias1 = .true.
             else 
                call wrndie(-4,'<FRENCALC>','US1 AND U11 HAVE INCONSISTENT LENGTH')
             endif
          endif

       else if(wrd == 'NBB' .or. wrd == 'BAR')  then
          qrw = (wrd == 'NBB')

          if(dA%n0 == 0 .or. dA%n1 == 0) then
             call wrndie(-2,'<FRENCALC>','NO DATA')
             cycle
          endif
          if(qrw .and. (.not.dA%q_bias0) .and. (.not.dA%q_bias1)) then
             call wrndie(-2,'<FRENCALC>','NBB REQUESTED BUT NO BIASES LOADED - ABORTING')
             cycle
          endif
          if(.not.qback .and. .not.qforw) then
             call wrndie(-2,'<FRENCALC>','NEED BOTH BACKWARDS AND FORWARDS DATA FOR NBB OR BAR')
             cycle
          endif

          temp = gtrmf(comlyn,comlen,'TEMP',300.0d0)
          barguess = gtrmf(comlyn,comlen,'IGUE',-zero)
          qnocheck= (INDXA(COMLYN,COMLEN,'NOCH').GT.0)
          qnosort= (INDXA(COMLYN,COMLEN,'NOSO').GT.0)
          qmedian = (INDXA(COMLYN,COMLEN,'MEDI').GT.0)

          ! If the number of blocks has been specified, determine standard deviation based on block averages 
          blocks = gtrmi(comlyn,comlen,'BLOC',0)
          if (blocks .ne. 0) qnosort = .TRUE. 

          if(qrw) then
             call datacheck(temp,dA%q_bias0,dA%q_bias1,maxovl,0,qrw,qnosort,qnocheck)
          else
             call datacheck(temp,.false.,.false.,maxovl,0,qrw,qnosort,qnocheck)
          endif

          ! Use position of maximum overlap as a first guess (unless one is specified by user)
          if(barguess == -zero) barguess = maxovl

          if (blocks .ne. 0) then
             call chmalloc('freeene_calc.src','FRENCALC','DABLOCKS',blocks,crl=dablocks)
             
             do i=0,blocks-1 
                call subbarit(temp,qrw,barguess,result,i*(dA%n0/blocks)+1,(i+1)*(dA%n0/blocks),i*(dA%n1/blocks)+1,(i+1)*(dA%n1/blocks))             
                if(wrd == 'NBB') write (outu,'(a,i6,a,f14.4,a)') "NBB> BLOCK ", i+1, " NBB FREE ENERGY ESTIMATE: ", real(result), " KCAL/MOL"
                if(wrd == 'BAR') write (outu,'(a,i6,a,f14.4,a)') "BAR> BLOCK ", i+1, " BAR FREE ENERGY ESTIMATE: ", real(result), " KCAL/MOL"
                dablocks(i+1) = result
             enddo
          endif
 
          call barit(temp,qrw,barguess,result)

          if(wrd == 'NBB') write (outu,'(a,f14.4,a)') "NBB> NBB FREE ENERGY ESTIMATE: ", real(result), " KCAL/MOL"
          if(wrd == 'BAR') write (outu,'(a,f14.4,a)') "BAR> BAR FREE ENERGY ESTIMATE: ", real(result), " KCAL/MOL"

          ! Save free energy to ?FREN
          !CALL SETMSR('FREN',result)
          CALL SET_PARAM('FREN',result)

          
          ! calculate standard deviation 
          if (blocks .ne. 0) then 
             stddev =  sqrt( sum( (dablocks(1:blocks)-result)**2 )/real(blocks-1)) 
             call chmdealloc('freeene_calc.src','FRENCALC','DABLOCKS',blocks,crl=dablocks)
             if(wrd == 'NBB') write (outu,'(a,f14.4,a)') "NBB> NBB STANDARD DEVIATION:   ", real(stddev), " KCAL/MOL"
             if(wrd == 'BAR') write (outu,'(a,f14.4,a)') "BAR> BAR STANDARD DEVIATION:   ", real(stddev), " KCAL/MOL"
             !CALL SETMSR('FSTD',stddev)
             CALL SET_PARAM('FSTD',stddev)
          endif

       else if(wrd == 'ZWAN' .or. wrd == 'NBZW') then
          if(dA%n0 == 0 .and. dA%n1 == 0) then
             call wrndie(-3,'<FRENCALC>','NO DATA')
             cycle
          endif

          qrw = .FALSE.
          qback   = (indxa(comlyn,comlen,'BACK') > 0)
          qrw = (indxa(comlyn,comlen,'REWE') > 0)

          if( wrd == 'NBZW' ) qrw = .TRUE.

          duoffset = zero
          vboffset = zero 



          duoffset = gtrmf(comlyn,comlen,'OFDU',duoffset)     ! Offset for potential energy difference
          vboffset = gtrmf(comlyn,comlen,'OFVB',vboffset)     ! Offset for biasing potential
          qmedian = (INDXA(COMLYN,COMLEN,'MEDI').GT.0)
          temp   = gtrmf(comlyn,comlen,'TEMP',300.0d0)
          qnoof= (INDXA(COMLYN,COMLEN,'NOOF').GT.0)
          qnocheck= (INDXA(COMLYN,COMLEN,'NOCH').GT.0)
          qnosort= (INDXA(COMLYN,COMLEN,'NOSO').GT.0)
          blocks = gtrmi(comlyn,comlen,'BLOC',0)
          if (blocks .ne. 0) qnosort = .TRUE. 

          if(qback) then
             if(.not.allocated(dA%dU1)) &
                  call wrndie(-5,'<FRENCALC>','YOU DID NOT LOAD ANY BACKWARDS DATA')

             if(qrw .and.(.not.dA%q_bias1)) &
                  call wrndie(-5,'<FRENCALC>','REWEIGHTING REQUESTED BUT NO DATA LOADED')

             if (blocks .ne. 0) then
                call chmalloc('freeene_calc.src','FRENCALC','DABLOCKS',blocks,crl=dablocks)
                
                do i=0,blocks-1 
                   if ( .not. qnoof )        duoffset = minval(dA%dU1(i*(dA%ndU1/blocks)+1:(i+1)*(dA%ndU1/blocks))) 
                   !  call mean(dA%dU1(i*(dA%ndU1/blocks)+1:(i+1)*(dA%ndU1/blocks)),(dA%ndU1/blocks),duoffset)
                   if (qrw .and. qback)       call mean(dA%Vb1(i*(dA%ndU1/blocks)+1:(i+1)*(dA%ndU1/blocks)),(dA%ndU1/blocks),vboffset)
                   call datacheck(temp,.false.,qrw,maxovl,2,qrw,qnosort,qnocheck)
                   call nbzwanzig((dA%ndU1/blocks),temp,duoffset,vboffset,&
                        & dA%dU1(i*(dA%ndU1/blocks)+1:(i+1)*(dA%ndU1/blocks)),&
                        & qrw,dA%Vb1(i*(dA%ndU1/blocks)+1:(i+1)*(dA%ndU1/blocks)),result)
                   dablocks(i+1) = result
                   if (qrw)       write(outu,'(a,i6,a,f14.4,a)')  " NBZWANZIG> BLOCK ", i+1, " ESTIMATED FREE ENERGY DIFFERENCE ",result, " KCAL/MOL"
                   if (.not. qrw) write(outu,'(a,i6,a,f14.4,a)')  " ZWANZIG> BLOCK ", i+1, " ESTIMATED FREE ENERGY DIFFERENCE ",result, " KCAL/MOL"
                enddo
             endif


             ! Use averages as offsets for exponential average
             if (.not. qnoof )     duoffset = minval(dA%dU1(1:dA%ndU1))    ! call mean(dA%dU1(1:dA%ndU1),dA%ndU1,duoffset)
             if (qrw )            call mean(dA%Vb1(1:dA%nVb1),dA%nVb1,vboffset)
             call datacheck(temp,.false.,qrw,maxovl,2,qrw,qnosort,qnocheck)
             if ( qmedian .and. ( .not. qnosort) .and. (.not. qnocheck) )      duoffset =  dA%dU1((dA%ndU1/2))
             call nbzwanzig(dA%n1,temp,duoffset,vboffset,dA%dU1,qrw,dA%Vb1,result)
             if (qrw)       write(outu,'(a,f14.4,a)') ' NBZWANZIG> ESTIMATED FREE ENERGY DIFFERENCE ',result, " KCAL/MOL"
             if (.not. qrw) write(outu,'(a,f14.4,a)') ' ZWANZIG> ESTIMATED FREE ENERGY DIFFERENCE ',result, " KCAL/MOL"
             ! Save free energy to ?FREN
             !CALL SETMSR('FREN',result)
             CALL SET_PARAM('FREN',result)
          else
             if(.not.allocated(dA%dU0)) &
                  call wrndie(-5,'<FRENCALC>','YOU DID NOT LOAD ANY FORWARDS DATA')

             if(qrw .and.(.not.dA%q_bias0)) &
                  call wrndie(-5,'<FRENCALC>','REWEIGHTING REQUESTED BUT NO DATA LOADED')


             if (blocks .ne. 0) then
                call chmalloc('freeene_calc.src','FRENCALC','DABLOCKS',blocks,crl=dablocks)
                
                do i=0,blocks-1 
                   if ( .not. qnoof  )       duoffset = minval(dA%dU0(i*(dA%ndU0/blocks)+1:(i+1)*(dA%ndU0/blocks)))   
                   !call mean(dA%dU0(i*(dA%ndU0/blocks)+1:(i+1)*(dA%ndU0/blocks)),(dA%ndU0/blocks),duoffset)
                   if (qrw )            call mean(dA%Vb0(i*(dA%ndU0/blocks)+1:(i+1)*(dA%ndU0/blocks)),(dA%ndU0/blocks),vboffset)
                   call datacheck(temp,.false.,qrw,maxovl,2,qrw,qnosort,qnocheck)
                   call nbzwanzig((dA%ndU0/blocks),temp,duoffset,vboffset, &
                        & dA%dU0((i*(dA%ndU0/blocks)+1):((i+1)*(dA%ndU0/blocks))) ,qrw, &
                        & dA%Vb0((i*(dA%ndU0/blocks)+1):((i+1)*(dA%ndU0/blocks))) ,result)
                   dablocks(i+1) = result
                   if (qrw)       write(outu,'(a,i6,a,f14.4,a)')  " NBZWANZIG> BLOCK ", i+1, " ESTIMATED FREE ENERGY DIFFERENCE ",result, " KCAL/MOL"
                   if (.not. qrw) write(outu,'(a,i6,a,f14.4,a)')  " ZWANZIG> BLOCK ", i+1, " ESTIMATED FREE ENERGY DIFFERENCE ",result, " KCAL/MOL"
                enddo
             endif


             ! Use averages as offsets for exponential average
             if (.not. qnoof ) duoffset =  minval(dA%dU0(1:dA%ndU0)) ! call mean(dA%dU0(1:dA%ndU0),dA%ndU0,duoffset)      
             if ( qrw ) call mean(dA%Vb0(1:dA%nVb0),dA%nVb0,vboffset)

             call datacheck(temp,qrw,.false.,maxovl,1,qrw,qnosort,qnocheck)
             if ( qmedian .and. ( .not. qnosort) .and. (.not. qnocheck) )      duoffset =  dA%dU0((dA%ndU0/2))
             call nbzwanzig(dA%n0,temp,duoffset,vboffset,dA%dU0,qrw,dA%Vb0,result)

             if (qrw)       write(outu,'(a,f14.4,a)') ' NBZWANZIG> ESTIMATED FREE ENERGY DIFFERENCE ',result, " KCAL/MOL"
             if (.not. qrw) write(outu,'(a,f14.4,a)') ' ZWANZIG> ESTIMATED FREE ENERGY DIFFERENCE ',result, " KCAL/MOL"
 
             ! Save free energy to ?FREN
             !CALL SETMSR('FREN',result)
             CALL SET_PARAM('FREN',result)
          endif
          
          ! calculate standard deviation 
          if (blocks .ne. 0) then 
             stddev =  sqrt( sum( (dablocks(1:blocks)-result)**2 )/real(blocks-1)) 
             call chmdealloc('freeene_calc.src','FRENCALC','DABLOCKS',blocks,crl=dablocks)
             if (qrw) write (outu,'(a,f14.4,a)') " NBZWANZIG> STANDARD DEVIATION:              ", real(stddev), " KCAL/MOL"
             if (.not. qrw) write (outu,'(a,f14.4,a)') " ZWANZIG> STANDARD DEVIATION:              ", real(stddev), " KCAL/MOL"
             !CALL SETMSR('FSTD',stddev)
             CALL SET_PARAM('FSTD',stddev)

          endif

       else if(wrd == ' ') then
          ! Do nothing

       else
          call wrndie(-5,'<FRENCALC>','BAD COMMAND')
       endif



    enddo

    if(allocated(scr0)) call chmdealloc('freeene_calc.src','FRENCALC','SCR0',npt,crl=scr0)
    if(allocated(scr1)) call chmdealloc('freeene_calc.src','FRENCALC','SCR1',npt,crl=scr1)
    call free_diff()

  end subroutine frencalc

end module freeene_calc
