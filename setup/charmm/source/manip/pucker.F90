module pucker_mod
  use chm_kinds
  use dimens_fcm
  use memory
  implicit none
  public epucker, pucker, pucker_constraint_output, setup_pucker_constraint, &
       pucka,pucker6cp,ncspuck, print_pucker_constraints
  private
  real(chm_real) :: puck_eq,puck_etheta,puck_ephi
  integer :: pkfrq,pucker_unit
  real(chm_real),allocatable,dimension(:) :: q_puck,theta_puck, phi_puck
  !  Puckering restraints
  !     ncspuck      Number of puckering restraints
  !     indcspuck    Index of puckering restraints, size (6,ncspuck)
  !     kQ_tab       "Spring" constants of restraint potential energy
  !     ktheta_tab
  !     kphi_tab
  !     Q0_tab       Rest values for restraint potential energy
  !     theta0_tab
  !     phi0_tab
  !     expQ_tab     Exponents of restraint potential energy
  !     exptheta_tab
  !     expphi_tab
  !
  integer :: ncspuck=0
  integer, allocatable, dimension(:,:) :: indcspuck
  real(chm_real), allocatable, dimension(:) :: kQ_tab, ktheta_tab, kphi_tab
  real(chm_real), allocatable, dimension(:) :: Q0_tab, theta0_tab, phi0_tab
  real(chm_real), allocatable, dimension(:) :: expQ_tab, exptheta_tab, expphi_tab
  real(chm_real),allocatable,dimension(:) :: qtmp,qtmp2


contains
  SUBROUTINE PUCKER(SID,ID,PHASE,AMPL,PDEF,X,Y,Z)
    !-----------------------------------------------------------------------
    !     Locate correct residue and try to find the five correct atoms...
    !     Lennart Nilsson, NOV-87
    !     ln october 1993:
    !     Modified to allow Cremer&Pople (PDEF='CP') or Altona&Sundaralingam
    !     (PDEF='AS') pucker definitions
    !
  use psf
  use stream
    !
    character(len=*) SID,ID,PDEF
    real(chm_real) PHASE,AMPL,X(*),Y(*),Z(*)
    !
    INTEGER I,ISEG,IRES,NPUCK,J
    INTEGER,parameter :: MXPUCK=5
    real(chm_real) XYZ(3,MXPUCK)
    INTEGER IPATM(MXPUCK)
    character(len=8) PUCNAM(MXPUCK)
    !
    !     The following DATA statement  D E F I N E S the atoms to be used
    !     in the pucker calculation (and their order).
    !     For now it is only set up for the standard CHARMM (TOPRNA10)
    !     (deoxy)ribose heavy atoms (=IUPAC)
    !++LN FIX APR 90:
    DATA PUCNAM /'O4'' ','C1'' ','C2'' ','C3'' ','C4'' '/
    !--
    !
    PHASE=-9999.99
    AMPL=-9999.99
    !     Locate correct residue (default to first segment)
    IF(SID  ==  '@@@@') SID=SEGID(1)
    DO I=1,NSEG
       IF(SID  ==  SEGID(I)) ISEG=I
    ENDDO
    DO I=NICTOT(ISEG)+1,NICTOT(ISEG+1)
       IF(ID == RESID(I)) IRES=I
    ENDDO
    !
    !     and now the (five) atoms
    NPUCK=0
    DO I=IBASE(IRES)+1,IBASE(IRES+1)
       DO J=1,MXPUCK
          IF (ATYPE(I) == PUCNAM(J)) THEN
             NPUCK=NPUCK+1
             IF(PDEF == 'CP')THEN
                XYZ(1,J)=X(I)
                XYZ(2,J)=Y(I)
                XYZ(3,J)=Z(I)
             ELSE
                IPATM(J)=I
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !
    !     In case atomnames are wrong...(STILL PENTOSES ONLY)
    IF(NPUCK  /=  5) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,100) NPUCK,SID(1:idleng), &
            ID(1:idleng),(PUCNAM(I)(1:idleng),I=1,MXPUCK)
       CALL DIEWRN(-3)
    ENDIF
100 FORMAT(/' %PUCKER-ERROR: Could only locate ',I5,' (deoxy)ribose', &
         /'   atoms for residue ',A,1X,A,' - expected atomnames are:', &
         /5(2X,A)/)
    !
    IF(PDEF == 'CP')THEN
       CALL PUCKCP(NPUCK,XYZ,PHASE,AMPL)
    ELSE
       CALL PUCK1(IPATM,X,Y,Z,PHASE,AMPL)
    ENDIF
    !
    RETURN
  END SUBROUTINE PUCKER

  SUBROUTINE PUCKA(IRES,PHASE,AMPL,PDEF,X,Y,Z)
    !-----------------------------------------------------------------------
    !     Same as PUCKER, but with the actual residue number IRES as input
    !
    use psf
    use stream
    !
    INTEGER IRES
    real(chm_real) PHASE,AMPL,X(*),Y(*),Z(*)
    character(len=*) PDEF
    !
    INTEGER I,J,NPUCK
    INTEGER MXPUCK
    PARAMETER (MXPUCK=5)
    real(chm_real) XYZ(3,MXPUCK)
    INTEGER IPATM(MXPUCK)
    character(len=8) PUCNAM(MXPUCK)
    !     The following DATA statement  D E F I N E S the atoms to be used
    !     in the pucker calculation (and their order).
    !     For now it is only set up for the standard CHARMM (TOPRNA10)
    !     (deoxy)ribose heavy atoms (=IUPAC)
    !++LN FIX APR 90:
    DATA PUCNAM /'O4'' ','C1'' ','C2'' ','C3'' ','C4'' '/
    !--
    PHASE=-9999.99
    AMPL=-9999.99
    NPUCK=0
    DO I=IBASE(IRES)+1,IBASE(IRES+1)
       DO J=1,MXPUCK
          IF (ATYPE(I) == PUCNAM(J)) THEN
             NPUCK=NPUCK+1
             IF(PDEF == 'AS')THEN
                IPATM(J)=I
             ELSE
                XYZ(1,J)=X(I)
                XYZ(2,J)=Y(I)
                XYZ(3,J)=Z(I)
             ENDIF
          ENDIF
       ENDDO
    ENDDO
    !
    !     In case atomnames are wrong...(STILL PENTOSES ONLY)
    IF(NPUCK  /=  5) THEN
       IF(WRNLEV >= 2) WRITE(OUTU,100) NPUCK,IRES, &
            (PUCNAM(I)(1:idleng),I=1,MXPUCK)
       CALL DIEWRN(-3)
    ENDIF
100 FORMAT(/' %PUCKER-ERROR: Could only locate ',I5,' (deoxy)ribose', &
         /'   atoms for residue number',I6,' - expected atomnames are:', &
         /5(2X,A)/)
    !
    IF(PDEF == 'CP')THEN
       CALL PUCKCP(NPUCK,XYZ,PHASE,AMPL)
    ELSE
       CALL PUCK1(IPATM,X,Y,Z,PHASE,AMPL)
    ENDIF
    !
    RETURN
  END SUBROUTINE PUCKA

  SUBROUTINE PUCKCP(N,XYZ,PHASE,AMPL)
    !-----------------------------------------------------------------------
    !     Compute pucker coordinates PHASE&AMPL [(qt,q,phi)] for the
    !     N-atom ring system whose coordinates are in XYZ
    !     [ Needed dimensions: q(n/2-1), phi((n-1)/2-1) ]
    !
    !     Algorithm from: Cremer&Pople, JACS 97:6, 1354-1358, 1975.
    !     see also Harvey&Prabhakaran, JACS 108,6128-6136, 1986.
    !     1983-10-25/LN Harvard Univ., Chemistry VAX
    !     1987-11-19/LN Adapted to use in CHARMM
    !     (For pentoses)
    !
    use consta
    use number,only:zero
    use vector
    !
    INTEGER N,NMAX
    PARAMETER (NMAX=6)
    real(chm_real) Q(NMAX),PHI(NMAX)
    real(chm_real) PHASE
    real(chm_real) AN,CSUM,SSUM,P,FACT,SUM
    real(chm_real) XYZ(3,*)
    real(chm_real) RC(3),R1(3),R2(3),RJ(3),RN(3)
    INTEGER I,J,M,NHALF,N2
    real(chm_real) C,S,AMPL,TMP
    !
    !     Geometrical center
    RC(1:3)=zero
    DO I=1,N
       CALL ADDVEC(XYZ(1,I),RC,RC,3)
    ENDDO
    AN=1./N
    CALL SCALR8(RC,3,AN)
    !     Normal vector
    R1(1:3)=zero
    R2(1:3)=zero
    DO J=1,N
       S=SIN(TWOPI*(J-1)/N)
       C=COS(TWOPI*(J-1)/N)
       CALL SUBVEC(XYZ(1,J),RC,RJ,3)
       CALL SCALR8(RJ,3,S)
       CALL ADDVEC(R1,RJ,R1,3)
       CALL SUBVEC(XYZ(1,J),RC,RJ,3)
       CALL SCALR8(RJ,3,C)
       CALL ADDVEC(R2,RJ,R2,3)
    ENDDO
    CALL CROSS3(R1,R2,RN)
    CALL NORMALL(RN,3)
    !     Coordinate eqns
    N2=(N-1)/2
    DO M=2,N2
       CSUM=0.0
       SSUM=0.0
       DO J=1,N
          CALL SUBVEC(XYZ(1,J),RC,RJ,3)
          CSUM=CSUM+DOTVEC(RJ,RN,3) * COS(TWOPI*M*(J-1)/N)
          SSUM=SSUM+DOTVEC(RJ,RN,3) * SIN(TWOPI*M*(J-1)/N)
       enddo
       CSUM=CSUM*SQRT(2./N)
       SSUM=-SSUM*SQRT(2./N)
       !       Sums done, solve for q & phi
       !       Q*COS(PHI)=CSUM
       !       Q*SIN(PHI)=SSUM
       !       And get 0<phi<360
       !++ LN FIX APR 90:  everything (22 lines) removed up to FIN !DO
       !       and replaced by new code
       Q(M-1)=SQRT(CSUM**2+SSUM**2)
       IF(Q(M-1)  ==  0)THEN
          PHI(M-1)=0.0
       ELSE
          P=ATAN2(SSUM,CSUM)+TWOPI/4.
          IF(P  <  0) P=TWOPI+P
          PHI(M-1)=P*360./TWOPI
       ENDIF
       !--
    enddo
    !
    !     One more q for even N
    IF(EVEN(N))THEN
       SUM=0.0
       FACT=1.0
       DO J=1,N
          CALL SUBVEC(XYZ(1,J),RC,RJ,3)
          SUM = SUM + FACT * DOTVEC(RJ, RN, 3)
          FACT=-FACT
       ENDDO
       NHALF=N/2
       TMP=N
       Q(NHALF)=SUM/SQRT(TMP)
       N2=N2+1
    ENDIF
    N2=N2-1
    AMPL = SQRT(DOTVEC(Q, Q, N2))
    !
    !     One phase only, ie for pentoses...
    PHASE=PHI(1)
    RETURN
  END SUBROUTINE PUCKCP

  SUBROUTINE PUCK1(IPATM,X,Y,Z,PHASE,AMPL)
    !-----------------------------------------------------------------------
    !     Compute pucker coordinates PHASE&AMPL  for the
    !     5-atom ring system whose coordinates are in XYZ
    !    
    !
    !     Algorithm from: Altona&Sundaralingam, JACS 94, 8205-8212, 1972.
    !     see also Harvey&Prabhakaran, JACS 108,6128-6136, 1986.
    !     1983-10-25/LN Harvard Univ., Chemistry VAX
    !     1987-11-19/LN Adapted to use in CHARMM
    !     1993-10-01/LN A&S instead of Cremer&Pople 
    !     (For pentoses)
    !
  use consta
  use exfunc
  use number
  use intcor2,only:geticv
    !
    INTEGER IPATM(*)
    real(chm_real) X(*),Y(*),Z(*),AMPL,PHASE
    !
    INTEGER I,J,K,L,M,N
    real(chm_real)  T(0:4),A,B
    real(chm_real)  D1,D2,D3,D4 
    !
    ! GET RINGTORSIONS
    I=IPATM(1)
    J=IPATM(2)
    K=IPATM(3)
    L=IPATM(4)
    M=IPATM(5)
    CALL GETICV(J,K,L,M,.FALSE.,D1,D2,T(0),D3,D4,X,Y,Z)
    CALL GETICV(K,L,M,I,.FALSE.,D1,D2,T(1),D3,D4,X,Y,Z)
    CALL GETICV(L,M,I,J,.FALSE.,D1,D2,T(2),D3,D4,X,Y,Z)
    CALL GETICV(M,I,J,K,.FALSE.,D1,D2,T(3),D3,D4,X,Y,Z)
    CALL GETICV(I,J,K,L,.FALSE.,D1,D2,T(4),D3,D4,X,Y,Z)
    !
    ! CONSTRUCT PHASE & AMPLITUDE (FOURIER VARIANT)
    A=ZERO
    B=ZERO
    DO N=0,4
       A=A+T(N)*COS(N*FOUR*PI/FIVE)
       B=B+T(N)*SIN(N*FOUR*PI/FIVE)
    ENDDO
    A= PTFOUR*A
    B=-PTFOUR*B
    AMPL=SQRT(A**2+B**2)
    ! 
    IF (AMPL  <=  ZERO)THEN
       PHASE=ZERO
    ELSE
       !ab...B940601.ab sign flip problem fixed.
       !ab     PHASE=ATAN(B/A)*180.0/PI
       !ab     IF(T(0)  <  ZERO) PHASE=PHASE+180.0
       PHASE=ATAN2(B,A)*RADDEG
       IF (PHASE  <  ZERO) PHASE=PHASE+360.0
    ENDIF
    RETURN
  END SUBROUTINE PUCK1

  ! *
  ! * Calculates the Cremer-Pople ring puckering parameters
  ! * for 6 member ring
  ! *
  ! * See: D. Cremer and J. A. Pople, JACS 97:6, page 1354, 1975
  ! * By: Antti-Pekka Hynninen, January 2012
  ! *
  subroutine pucker6cp(xin, yin, zin, Q, theta, phi, &
       x0, y0, z0, zpo, wx, wy, wz, q2, q3, f, g, ux, uy, uz, vx, vy, vz)
    use chm_kinds
    use number
    use consta
    use vector
    implicit none
    ! Input / Output
    real(chm_real), intent(in) :: xin(6), yin(6), zin(6)
    real(chm_real), intent(out) :: Q, theta, phi
    real(chm_real), optional, intent(out) :: x0(6), y0(6), z0(6), zpo(6) ! optional output values
    real(chm_real), optional, intent(out) :: wx, wy, wz, q2, q3, f, g, ux, uy, uz, vx, vy, vz
    ! Variables
    real(chm_real) x(6), y(6), z(6), zp(6)
    real(chm_real) xcm, ycm, zcm
    real(chm_real) Rd(3), Rdd(3), n(3)          ! vectors R', R'', n
    real(chm_real) qq(3), qcosp(2), qsinp(2)
    real(chm_real), parameter :: pi3 = pi/three ! 2*pi/6
    real(chm_real), parameter :: one_sq3 = one/sqrt(three), one_sq6 = one/sqrt(six)
    real(chm_real) sina(6), cosa(6)
    integer i

    ! Re-center coordinates such that sum(R) = 0
    xcm = sum(xin)/six
    ycm = sum(yin)/six
    zcm = sum(zin)/six
    x = xin - xcm
    y = yin - ycm
    z = zin - zcm

    if (present(x0)) x0(1:6) = x(1:6)
    if (present(y0)) y0(1:6) = y(1:6)
    if (present(z0)) z0(1:6) = z(1:6)

    do i=1,6
       sina(i) = sin(pi3*(i-1))
       cosa(i) = cos(pi3*(i-1))
    enddo

    Rd = zero
    Rdd = zero
    do i=1,6
       Rd(1) = Rd(1) + x(i)*sina(i)
       Rd(2) = Rd(2) + y(i)*sina(i)
       Rd(3) = Rd(3) + z(i)*sina(i)
       Rdd(1) = Rdd(1) + x(i)*cosa(i)
       Rdd(2) = Rdd(2) + y(i)*cosa(i)
       Rdd(3) = Rdd(3) + z(i)*cosa(i)
    enddo

    if (present(ux)) ux = Rd(1)
    if (present(uy)) uy = Rd(2)
    if (present(uz)) uz = Rd(3)

    if (present(vx)) vx = Rdd(1)
    if (present(vy)) vy = Rdd(2)
    if (present(vz)) vz = Rdd(3)

    call cross3(Rd,Rdd,n)
    if (present(wx)) wx = n(1)
    if (present(wy)) wy = n(2)
    if (present(wz)) wz = n(3)
    call normall(n,3)

    ! Calculate projections on n
    do i=1,6
       call dotpr((/ x(i), y(i), z(i) /), n, 3, zp(i))
    enddo

    if (present(zpo)) zpo(1:6) = zp(1:6)

    qq(3) = zero
    qcosp = zero
    qsinp = zero
    Q = zero
    do i=1,6
       qcosp(1) = qcosp(1) + zp(i)*cosa(i)
       qcosp(2) = qcosp(2) + zp(i)*cos(pi3*2*(i-1))

       qsinp(1) = qsinp(1) + zp(i)*sina(i)
       qsinp(2) = qsinp(2) + zp(i)*sin(pi3*2*(i-1))

       if (mod(i-1,2) == 0) then
          ! (-1)^(i-1) == 1
          qq(3) = qq(3) + zp(i)
       else
          ! (-1)^(i-1) == -1
          qq(3) = qq(3) - zp(i)
       endif

       Q = Q + zp(i)**2
    enddo
    qq(3) = qq(3)*one_sq6
    qcosp = qcosp*one_sq3
    qsinp = -qsinp*one_sq3
    qq(1) = sqrt(qcosp(1)**2 + qsinp(1)**2)
    qq(2) = sqrt(qcosp(2)**2 + qsinp(2)**2)
    Q = sqrt(Q)

    if (present(q2)) q2 = qq(2)
    if (present(q3)) q3 = qq(3)

    if (present(f)) f = qcosp(2)
    if (present(g)) g = qsinp(2)

    if (qcosp(2) > zero) then
       if (qsinp(2) > zero) then
          phi = atan(qsinp(2)/qcosp(2))
       else
          phi = twopi - abs(atan(qsinp(2)/qcosp(2)))
       endif
    else
       if (qsinp(2) > zero) then
          phi = pi - abs(atan(qsinp(2)/qcosp(2)))
       else
          phi = pi + abs(atan(qsinp(2)/qcosp(2)))
       endif
    endif

    if (qq(3) > zero) then
       theta = atan(qq(2)/qq(3))
    else
       theta = pi - abs(atan(qq(2)/qq(3)))
    endif

    if (theta < zero .or. theta > pi) call wrndie(-1,'<PUCKER>','INVALID THETA VALUE')
    if (phi < zero .or. phi > twopi) call wrndie(-1,'<PUCKER>','INVALID PHI VALUE')

    return
  end subroutine pucker6cp

  ! *
  ! * Calculates the restraint potential energy and force from a 6 member pucker ring restraint
  ! *
  ! * Input:
  ! * x, y, z = Original cartesian coordinates
  ! * ind, nind = index and number of index of pucker restraints
  ! * kQ, ktheta, kphi = "spring constant"
  ! * Q0, theta0, phi0 = energy zero values of Q, theta, phi
  ! * expQ, exptheta, expphi = exponents of the potential energy function
  ! *
  ! * Output:
  ! * ener                   = EQ + Etheta, Ephi
  ! * forcex, forcey, forcez = forces for the 6 atoms
  ! *
  ! * The form of the restaining potential energy function is given by:
  ! * EQ = kQ*|Q-hQ0|**expQ
  ! * ... and similarly for Etheta, and Ephi
  ! *
  ! * By: Antti-Pekka Hynninen, January 2012
  ! *
  subroutine epucker(ener, forcex, forcey, forcez, x, y, z)
    use chm_kinds
    use number
    use consta
    use vector
#if KEY_PARALLEL==1
    use parallel,only:mynodp,numnod  
#endif
    implicit none
    ! Input / Output
    real(chm_real), intent(out) :: ener
    real(chm_real) forcex(*), forcey(*), forcez(*)
    real(chm_real), intent(in) :: x(*), y(*), z(*)
    ! Variables
    integer i, j, k, k_start, k_step
    real(chm_real), parameter :: pi3 = pi/three, fourpi6 = four*pi/six
    real(chm_real), parameter :: one_sq3 = one/sqrt(three), one_sq6 = one/sqrt(six)
    real(chm_real) xt(6), yt(6), zt(6)
    real(chm_real) x0(6), y0(6), z0(6), zp(6), nx, ny, nz
    real(chm_real) dEdQ, dEdtheta, dEdphi
    real(chm_real) dQdx, dthetadx, dphidx
    real(chm_real) dQdy, dthetady, dphidy
    real(chm_real) dQdz, dthetadz, dphidz
    real(chm_real) dzpdx(6), dzpdy(6), dzpdz(6)
    real(chm_real) dnxdx, dnydx, dnzdx
    real(chm_real) dnxdy, dnydy, dnzdy
    real(chm_real) dnxdz, dnydz, dnzdz
    real(chm_real) wx, wy, wz, inv_uv, inv_uv3, wfact
    real(chm_real) dwydx, dwzdx
    real(chm_real) dwxdy, dwzdy
    real(chm_real) dwxdz, dwydz
    real(chm_real) ux, uy, uz, vx, vy, vz
    real(chm_real) dvd(6), dud(6)
    real(chm_real) q2, q3
    real(chm_real) dq2dx, dq2dy, dq2dz
    real(chm_real) dq3dx, dq3dy, dq3dz
    real(chm_real) f, g, signval
    real(chm_real) dfdx, dfdy, dfdz
    real(chm_real) dgdx, dgdy, dgdz
    real(chm_real) cosa(6), sina(6), cosb(6), sinb(6), cosc(6)
    real(chm_real) fact1, fact2
    real(chm_real) q,theta,phi

    ! Precalculate cosine and sine tables
    do j=1,6
       cosa(j) = cos(pi3*(j-1))
       sina(j) = sin(pi3*(j-1))
       cosb(j) = cos(fourpi6*(j-1))
       sinb(j) = sin(fourpi6*(j-1))
       cosc(j) = cos(pi*(j-1))
    enddo

    do i=1,6
       ! dvd = dvx/dx = dvy/dy = dvz/dz
       ! dud = dux/dx = duy/dy = duz/dz
       dvd(i) = zero
       dud(i) = zero
       do j=1,6
          dvd(i) = dvd(i) + (deltaij(i,j) - sixth)*cosa(j)
          dud(i) = dud(i) + (deltaij(i,j) - sixth)*sina(j)
       enddo
    enddo

#if KEY_PARALLEL==1
    k_start = mynodp
    k_step = numnod
#else /**/
    k_start = 1
    k_step = 1
#endif 
    q_puck = zero
    theta_puck = zero
    phi_puck = zero
    do k=k_start,ncspuck,k_step

       xt(1:6) = x(indcspuck(1:6,k))
       yt(1:6) = y(indcspuck(1:6,k))
       zt(1:6) = z(indcspuck(1:6,k))

       ! Calculate puckering coordinates
       call pucker6cp(xt, yt, zt, Q, theta, phi, &
            x0, y0, z0, zp, wx, wy, wz, q2, q3, f, g, &
            ux, uy, uz, vx, vy, vz)
       q_puck(k)     = q
       theta_puck(k) = theta
       phi_puck(k)   = phi
       puck_eq = kq_tab(k)*abs(Q-Q0_tab(k))**expQ_tab(k)
       puck_etheta = ktheta_tab(k)*abs(theta-theta0_tab(k))**exptheta_tab(k)
       puck_ephi   = kphi_tab(k)*abs(phi-phi0_tab(k))**expphi_tab(k)
       ener = ener +  puck_eq + puck_etheta + puck_ephi

       inv_uv = one/sqrt(wx**2 + wy**2 + wz**2)
       inv_uv3 = inv_uv**3
       nx = wx*inv_uv
       ny = wy*inv_uv
       nz = wz*inv_uv

       dEdQ     = expQ_tab(k)*kq_tab(k)*(Q-Q0_tab(k))*abs(Q-Q0_tab(k))**(expQ_tab(k)-two)
       dEdtheta = exptheta_tab(k)*ktheta_tab(k)*(theta-theta0_tab(k))*&
            abs(theta-theta0_tab(k))**(exptheta_tab(k)-two)
       dEdphi   = expphi_tab(k)*kphi_tab(k)*(phi-phi0_tab(k))*abs(phi-phi0_tab(k))**(expphi_tab(k)-two)

       do i=1,6
          dwydx = uz*dvd(i) - vz*dud(i)
          dwzdx = vy*dud(i) - uy*dvd(i)
          dwxdy = vz*dud(i) - uz*dvd(i)
          dwzdy = ux*dvd(i) - vx*dud(i)
          dwxdz = uy*dvd(i) - vy*dud(i)
          dwydz = vx*dud(i) - ux*dvd(i)

          wfact = (wy*dwydx + wz*dwzdx)*inv_uv3
          dnxdx =              - wx*wfact
          dnydx = dwydx*inv_uv - wy*wfact
          dnzdx = dwzdx*inv_uv - wz*wfact

          wfact = (wx*dwxdy + wz*dwzdy)*inv_uv3
          dnxdy = dwxdy*inv_uv - wx*wfact
          dnydy =              - wy*wfact
          dnzdy = dwzdy*inv_uv - wz*wfact
          
          wfact = (wx*dwxdz + wy*dwydz)*inv_uv3
          dnxdz = dwxdz*inv_uv - wx*wfact
          dnydz = dwydz*inv_uv - wy*wfact
          dnzdz =              - wz*wfact

          do j=1,6
             dzpdx(j) = (deltaij(i,j) - sixth)*nx + x0(j)*dnxdx + y0(j)*dnydx + z0(j)*dnzdx
             dzpdy(j) = x0(j)*dnxdy + (deltaij(i,j) - sixth)*ny + y0(j)*dnydy + z0(j)*dnzdy
             dzpdz(j) = x0(j)*dnxdz + y0(j)*dnydz + (deltaij(i,j) - sixth)*nz + z0(j)*dnzdz
          enddo

          call dotpr(zp, dzpdx, 6, dQdx)
          call dotpr(zp, dzpdy, 6, dQdy)
          call dotpr(zp, dzpdz, 6, dQdz)
          dQdx = dQdx/Q
          dQdy = dQdy/Q
          dQdz = dQdz/Q

          dfdx = zero
          dfdy = zero
          dfdz = zero
          dgdx = zero
          dgdy = zero
          dgdz = zero
          dq3dx = zero
          dq3dy = zero
          dq3dz = zero
          do j=1,6
             dfdx = dfdx + dzpdx(j)*cosb(j)
             dfdy = dfdy + dzpdy(j)*cosb(j)
             dfdz = dfdz + dzpdz(j)*cosb(j)
             dgdx = dgdx + dzpdx(j)*sinb(j)
             dgdy = dgdy + dzpdy(j)*sinb(j)
             dgdz = dgdz + dzpdz(j)*sinb(j)
             dq3dx = dq3dx + dzpdx(j)*cosc(j)
             dq3dy = dq3dy + dzpdy(j)*cosc(j)
             dq3dz = dq3dz + dzpdz(j)*cosc(j)
          enddo
          dfdx = dfdx*one_sq3
          dfdy = dfdy*one_sq3
          dfdz = dfdz*one_sq3
          dgdx = -dgdx*one_sq3
          dgdy = -dgdy*one_sq3
          dgdz = -dgdz*one_sq3
          dq3dx = dq3dx*one_sq6
          dq3dy = dq3dy*one_sq6
          dq3dz = dq3dz*one_sq6

          dq2dx = (f*dfdx + g*dgdx)/q2
          dq2dy = (f*dfdy + g*dgdy)/q2
          dq2dz = (f*dfdz + g*dgdz)/q2

          fact1 = q2/q3**2
          fact2 = one/(one + (q2/q3)**2)
          dthetadx = (dq2dx/q3 - fact1*dq3dx)*fact2
          dthetady = (dq2dy/q3 - fact1*dq3dy)*fact2
          dthetadz = (dq2dz/q3 - fact1*dq3dz)*fact2
          if (q3 < zero) then
             signval = -sign(one, atan(q2/q3))
             dthetadx = signval*dthetadx
             dthetady = signval*dthetady
             dthetadz = signval*dthetadz
          endif

          fact1 = g/f**2
          fact2 = one/(one + (g/f)**2)
          dphidx = (dgdx/f - dfdx*fact1)*fact2
          dphidy = (dgdy/f - dfdy*fact1)*fact2
          dphidz = (dgdz/f - dfdz*fact1)*fact2
          if (f > zero) then
             if (g > zero) then
                signval = one
             else
                signval = -sign(one, atan(g/f))
             endif
          else
             if (g > zero) then
                signval = -sign(one, atan(g/f))
             else
                signval = sign(one, atan(g/f))
             endif
          endif
          dphidx = signval*dphidx
          dphidy = signval*dphidy
          dphidz = signval*dphidz

          j = indcspuck(i,k)
          forcex(j) = forcex(j) + dEdQ*dQdx + dEdtheta*dthetadx + dEdphi*dphidx
          forcey(j) = forcey(j) + dEdQ*dQdy + dEdtheta*dthetady + dEdphi*dphidy
          forcez(j) = forcez(j) + dEdQ*dQdz + dEdtheta*dthetadz + dEdphi*dphidz
       enddo

    enddo

    return

  contains
    ! * Kronecker delta function
    real(chm_real) function deltaij(i,j)
      integer i, j

      if (i == j) then
         deltaij = one
      else
         deltaij = zero
      endif

      return
    end function deltaij

  end subroutine epucker

  subroutine pucker_constraint_output(istep)
#if KEY_PARALLEL==1
    use mpi                         
#endif
#if KEY_PARALLEL==1
    use parallel                    
#endif
    use stream,only:iolev
    integer,intent(in) :: istep
    integer :: i,j,ierr
    character(len=80) :: fmt
    if(pucker_unit <= 0)return
    if(mod(istep,pkfrq) /= 0) return
    call chmalloc('pucker.src','pucker_constraint_output','qtmp',3*ncspuck,crl=qtmp)
#if KEY_PARALLEL==1
    call chmalloc('pucker.src','pucker_constraint_output','qtmp2',3*ncspuck,crl=qtmp2) 
#endif
    i=1
    do j=1,ncspuck
       qtmp(i) = q_puck(j)
       qtmp(i+1) = theta_puck(j)
       qtmp(i+2) = phi_puck(j)
       i=i+3
    enddo
#if KEY_PARALLEL==1
    qtmp2=qtmp      
#endif
#if KEY_PARALLEL==1
    call mpi_reduce(qtmp2,qtmp,ncspuck*3,MPI_DOUBLE_PRECISION,MPI_SUM,0,comm_charmm,ierr)   
#endif
    if(iolev > 0 ) then
       write(fmt,'("(i10,",i3,"f12.6)")')ncspuck*3
       write(pucker_unit,'(i10,20f15.6)')istep,qtmp(1:3*ncspuck)
    end if
    call chmdealloc('pucker.src','pucker_constraint_output','qtmp',3*ncspuck,crl=qtmp)
#if KEY_PARALLEL==1
    call chmdealloc('pucker.src','pucker_constraint_output','qtmp2',3*ncspuck,crl=qtmp2) 
#endif
    return
  end subroutine pucker_constraint_output

  subroutine setup_pucker_constraint(comlyn,comlen)
    use stream
    use psf
    use number,only: zero
    use string,only: gtrmf,nextf,gtrmi
    use select,only: nxtatm
    use chutil,only: atomid
    use param_store, only: set_param

    implicit none

    integer comlen
    character(len=comlen) :: comlyn
    logical pucker_ok
    integer i,j,qat(6),nqat,pucker_unit_new, pkfrq_new
    character(len=8), dimension(6) :: SID, RID, REN, AC
    integer,allocatable,dimension(:) :: islct


    call chmalloc('pucker.src','setup_pucker_constraint','islct',NATOM,intg=islct)
    
    ! Add puckering restraint

    pucker_ok = .false.

    ncspuck = ncspuck + 1
    call cspuck_ensure_capacity(ncspuck)
    ktheta_tab(ncspuck)   = zero
    kphi_tab(ncspuck)     = zero
    theta0_tab(ncspuck)   = zero
    phi0_tab(ncspuck)     = zero
    exptheta_tab(ncspuck) = zero
    expphi_tab(ncspuck)   = zero

    ! Get atoms
    call nxtatm(qat,nqat,6,comlyn,comlen,islct, &
         segid,resid,atype,ibase,nictot,nseg,res,natom)
    if(nqat /= 6) goto 801

    ! Get restraint constants, values, and exponents
    kQ_tab(ncspuck) = gtrmf(comlyn,comlen,'KCON',zero)
    ktheta_tab(ncspuck) = nextf(comlyn,comlen)
    kphi_tab(ncspuck) = nextf(comlyn,comlen)

    Q0_tab(ncspuck) = gtrmf(comlyn,comlen,'VALU',zero)
    theta0_tab(ncspuck) = nextf(comlyn,comlen)
    phi0_tab(ncspuck) = nextf(comlyn,comlen)

    expQ_tab(ncspuck) = gtrmf(comlyn,comlen,'EXPO',zero)
    exptheta_tab(ncspuck) = nextf(comlyn,comlen)
    expphi_tab(ncspuck) = nextf(comlyn,comlen)

    pucker_unit_new = gtrmi(comlyn,comlen,'OUTU',-1)
    pkfrq_new = gtrmi(comlyn,comlen,'PKFRQ',10)

    if(ncspuck == 1) then 
       pucker_unit = pucker_unit_new
       pkfrq = pkfrq_new
    endif
    if(ncspuck > 1 ) call check_outunit_pucker   !contained subroutine

    indcspuck(1:6,ncspuck) = qat(1:6)
    do i=1,6
       j = indcspuck(i,ncspuck)
       if (j <= 0 .or. j > natomt) goto 801
       call atomid(j,sid(i),rid(i),ren(i),ac(i))
    enddo
    pucker_ok = .true.

801 if (pucker_ok) then
       ! Pucker restraint OK, keep it
       if (prnlev >= 2) write (outu,'(a,i5,a,/,6(1x,a,1x,a,1x,a,/),3(a,3f9.2,/))') &
            ' NEW PUCKERING RESTRAINT ADDED',ncspuck,':', &
            rid(1)(1:idleng), sid(1)(1:idleng), ac(1)(1:idleng), &
            rid(2)(1:idleng), sid(2)(1:idleng), ac(2)(1:idleng), &
            rid(3)(1:idleng), sid(3)(1:idleng), ac(3)(1:idleng), &
            rid(4)(1:idleng), sid(4)(1:idleng), ac(4)(1:idleng), &
            rid(5)(1:idleng), sid(5)(1:idleng), ac(5)(1:idleng), &
            rid(6)(1:idleng), sid(6)(1:idleng), ac(6)(1:idleng), &
            ' CONS=     ',kQ_tab(ncspuck), ktheta_tab(ncspuck), kphi_tab(ncspuck), &
            ' VALUes=   ',Q0_tab(ncspuck), theta0_tab(ncspuck), phi0_tab(ncspuck), &
            ' EXPOnents=',expQ_tab(ncspuck), exptheta_tab(ncspuck), expphi_tab(ncspuck)
    else
       ! Pucker restraint faulty, remove it
       ncspuck = ncspuck - 1
       if (wrnlev >= 2) call wrndie(0,'<cstran>',&
            'Some component of the puckering restraint was invalid, this restraint ignored')
    endif

    call set_param('NCSU',ncspuck)

    call chmdealloc('pucker.src','setup_pucker_constraint','islct',NATOM,intg=islct)

    return

  contains
    subroutine check_outunit_pucker   !contained subroutine
       if(pucker_unit_new /= -1)then
          if(pucker_unit == -1) then
             pucker_unit = pucker_unit_new
             pkfrq       = pkfrq_new
          elseif(pucker_unit /= pucker_unit_new ) then
             if(iolev > 0) then
                write(outu,'(a)'   ) "***** ERROR PUCKER CONSTRAINT>> Output units do not match "
                write(outu,'(a)'   ) "***** ERROR PUCKER CONSTRAINT>>   for multiple puck constraints"
                write(outu,'(a,i4)') "***** ERROR PUCKER CONSTRAINT>>   original unit:  ", pucker_unit
                write(outu,'(a,i4)') "***** ERROR PUCKER CONSTRAINT>>   last unit spec: ", pucker_unit_new 
                write(outu,'(a)'   ) "***** ERROR PUCKER CONSTRAINT>> Only first unit and frequency will be used "
                write(outu,'(a)'   ) "***** ERROR PUCKER CONSTRAINT>>      if this warning is bypassed."
             endif
             call wrndie(-1, "<pucker.src>setup_pucker_constraint ","multiple output units specified") 
          endif
       endif
       return
     end subroutine check_outunit_pucker
    
  end subroutine setup_pucker_constraint

  subroutine cspuck_ensure_capacity(wantcap)
    use cnst_fcm
    integer, intent(in) :: wantcap
    integer, parameter :: mincap = 20
    real, parameter :: pad = 1.25
    integer :: newcap

    if (allocated(kQ_tab)) then
       if (wantcap > size(kQ_tab)) then
          newcap = ceiling(pad * wantcap)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','indcspuck',6,newcap, &
               intg=indcspuck)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','q_puck',newcap,crl=q_puck)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','theta_puck',newcap,crl=theta_puck)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','phi_puck',newcap,crl=phi_puck)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','kQ_tab',newcap,crl=kQ_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','ktheta_tab',newcap, &
               crl=ktheta_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','kphi_tab',newcap,crl=kphi_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','Q0_tab',newcap,crl=Q0_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','theta0_tab',newcap, &
               crl=theta0_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','phi0_tab',newcap,crl=phi0_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','expQ_tab',newcap,crl=expQ_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','exptheta_tab',newcap, &
               crl=exptheta_tab)
          call chmrealloc('cstran.src','cspuck_ensure_capacity','expphi_tab',newcap, &
               crl=expphi_tab)
       endif
    else
       newcap = max(mincap, ceiling(pad * wantcap))
       call chmalloc('cstran.src','cspuck_ensure_capacity','indcspuck',6,newcap,intg=indcspuck)
       call chmalloc('cstran.src','cspuck_ensure_capacity','q_puck',newcap,crl=q_puck)
       call chmalloc('cstran.src','cspuck_ensure_capacity','theta_puck',newcap,crl=theta_puck)
       call chmalloc('cstran.src','cspuck_ensure_capacity','phi_puck',newcap,crl=phi_puck)
       call chmalloc('cstran.src','cspuck_ensure_capacity','kQ_tab',newcap,crl=kQ_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','ktheta_tab',newcap,crl=ktheta_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','kphi_tab',newcap,crl=kphi_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','Q0_tab',newcap,crl=Q0_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','theta0_tab',newcap,crl=theta0_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','phi0_tab',newcap,crl=phi0_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','expQ_tab',newcap,crl=expQ_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','exptheta_tab',newcap, &
            crl=exptheta_tab)
       call chmalloc('cstran.src','cspuck_ensure_capacity','expphi_tab',newcap,crl=expphi_tab)
    endif
  end subroutine cspuck_ensure_capacity

  subroutine cspuck_clear()
    use cnst_fcm
    integer oldcap
    if (allocated(indcspuck)) then
       oldcap = size(kQ_tab)
       call chmdealloc('cstran.src','cspuck_clear','indcspuck',6,oldcap,intg=indcspuck)
       call chmrealloc('cstran.src','cspuck_ensure_capacity','q_puck',oldcap,crl=q_puck)
       call chmdealloc('cstran.src','cspuck_ensure_capacity','theta_puck',oldcap,crl=theta_puck)
       call chmdealloc('cstran.src','cspuck_ensure_capacity','phi_puck',oldcap,crl=phi_puck)
       call chmdealloc('cstran.src','cspuck_clear','kQ_tab',oldcap,crl=kQ_tab)
       call chmdealloc('cstran.src','cspuck_clear','ktheta_tab',oldcap,crl=ktheta_tab)
       call chmdealloc('cstran.src','cspuck_clear','kphi_tab',oldcap,crl=kphi_tab)
       call chmdealloc('cstran.src','cspuck_clear','Q0_tab',oldcap,crl=Q0_tab)
       call chmdealloc('cstran.src','cspuck_clear','theta0_tab',oldcap,crl=theta0_tab)
       call chmdealloc('cstran.src','cspuck_clear','phi0_tab',oldcap,crl=phi0_tab)
       call chmdealloc('cstran.src','cspuck_clear','expQ_tab',oldcap,crl=expQ_tab)
       call chmdealloc('cstran.src','cspuck_clear','exptheta_tab',oldcap,crl=exptheta_tab)
       call chmdealloc('cstran.src','cspuck_clear','expphi_tab',oldcap,crl=expphi_tab)
    endif
  end subroutine cspuck_clear

  subroutine print_pucker_constraints(unit)
    use stream,only:idleng
    use chutil,only: atomid
    integer unit

    integer i,j
    logical foundx
    character(len=8), dimension(6) :: SID,RID,REN,AC
    


    foundx=.true.
    write(unit,'(/,6x,a,/)') 'FORCE CONSTANTS, ZERO VALUES AND EXPONENTS FOR PUCKERING CONSTRAINTS'
    do j=1,ncspuck
       do i=1,6
          call atomid(indcspuck(i,j),sid(i),rid(i),ren(i),ac(i))
       enddo
       write(unit,426) j, &
            sid(1)(1:idleng),rid(1)(1:idleng),ac(1)(1:idleng), &
            sid(2)(1:idleng),rid(2)(1:idleng),ac(2)(1:idleng), &
            sid(3)(1:idleng),rid(3)(1:idleng),ac(3)(1:idleng), &
            sid(4)(1:idleng),rid(4)(1:idleng),ac(4)(1:idleng), &
            sid(5)(1:idleng),rid(5)(1:idleng),ac(5)(1:idleng), &
            sid(6)(1:idleng),rid(6)(1:idleng),ac(6)(1:idleng), &
            kQ_tab(j), ktheta_tab(j), kphi_tab(j), &
            Q0_tab(j), theta0_tab(j), phi0_tab(j), &
            expQ_tab(j), exptheta_tab(j), expphi_tab(j)
426    format(i4,':',3x,4(a,' ',a,' ',a,'/ '),/,8x,2(a,' ',a,' ',a,'/ '),/,8x,&
            'CONS=     ',3f9.2,/,8x,'VALUes=   ',3f9.2,/,8x,'EXPOnents=',3f9.2)
    enddo
    return
  end subroutine print_pucker_constraints


end module pucker_mod

