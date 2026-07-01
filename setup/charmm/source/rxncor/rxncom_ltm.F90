!Puja's zeta put in c43 by Tanmoy 10-16-2017 [QC: 11/17]
module rxncom
  use chm_kinds
  use chm_types
  use dimens_fcm
#if KEY_RXNCOR==1 /*rxncor_fcm*/
  !     rxncom.fcm
  !
  !     ---- data structures for defining reaction coordinate and for
  !     ---- calculating energy due to umbrella potential and the 
  !     ---- corresponding forces.
  !
  !     ----                  J. Kottalam,     June 1989
  !
  integer,allocatable,dimension(:),save :: nmlstt, nrxstt
  real(chm_real),allocatable,dimension(:),save :: lodel, hidel, deldel
  integer,allocatable,dimension(:),save :: delstp, tpopun
  real(chm_real),allocatable,dimension(:),save :: basalo,basahi,basblo,basbhi
  integer,allocatable,dimension(:),save :: treehi,treelo
  real(chm_real),allocatable,dimension(:),save :: kumbpr
  integer,allocatable,dimension(:),save :: umbfrm,nbiasp,qflip
  real(chm_real),allocatable,dimension(:),save :: pumbpr,dl0ptr,smddel
  type(chm_array),allocatable,dimension(:),save :: rbiasp, cbiasp

#if KEY_DOMDEC==1
  ! For DOMDEC, .true. for node on where the computation was performed
  logical :: q_comp_nod = .false.
#endif

  integer,parameter :: maxnod=2000
  !
  character(len=4) nodnam(maxnod)

  type(chm_array),dimension(1:maxnod),save :: wtaa
  type(chm_iarray),dimension(1:maxnod),save :: iataa

  integer :: ngraph,           NRXNCR, &
       rxnind,           nrxn, &
       deftyp(maxnod),   refnod(3,maxnod), &
       rxntrc(maxnod),   rxntrn, &
       trcfrq(maxnod),   trunit(maxnod), &
       rxncnt,           sttstp, &
       UMBMDSTEP,        old_MDSTEP
  real(chm_real) :: delval(5,maxnod), delder(3,maxnod)

   ! zeta coordinate - Puja [QC: 11/17]
   integer,allocatable,dimension(:),save :: iat11(:),iat12(:)
   real(chm_real),allocatable,dimension(:),save :: wt1(:)
   integer,allocatable,dimension(:),save :: iat21(:),iat22(:)
   real(chm_real),allocatable,dimension(:),save :: wt2(:)

  !
  !     ---- We wish to define a graph and a tree with the following
  !     ---- node-types: { ratio, combination, distance, angle,
  !     ---- plane, line, direction, point }
  !
  !     ---- connectivity: node-i connects to node-j iff node-j
  !     ---- is used for defining node-i.
  !
  !     ---- Possible connectivities are listed below.  Each is identified
  !     ---- with a type code.
  !
  integer,parameter :: &
       POINT_ITEM=8, &
       LINE_ITEM=6, &
       PLANE_ITEM=5, &
       DIR_DIR_DIR=777, &
       PLN_PLN_DIR=557, &
       LIN_LIN_DIR=667, &
       LIN_PLN_DIR=657, &
       PLN_LIN_DIR=567, &
#if KEY_ROLLRXNCOR==1
       DIR_DIR_COM=778, &  !these two added by jms 2/2011
       COM_COM_COM=888, &  
#endif 
       RATION_TYPE=10, &
       RATIO_TYPE=10, &
       COMBI_TYPE=20, &
       DISTPTPT_TYPE=31, &
       DISTPTLN_TYPE=32, &
       DISTPTPL_TYPE=33, &
       ANGLDRDR_TYPE=40, &
       ANGLLNLN_TYPE=41, &
       ANGLLNPL_TYPE=42, &
       ANGLPLPL_TYPE=43, &
       PLANPTDR_TYPE=50, &
       PLAN3PT_TYPE=51, &
       PLANPTLN_TYPE=52, &
       PLANPTNM_TYPE=53, &
       PLANPTLL_TYPE=54, &
       PLANPTPL_TYPE=55, &
       LINEPTDR_TYPE=60, &
       LINEPTPT_TYPE=61, &
       LINEPTLP_TYPE=62, &
       LINEPTPP_TYPE=63, &
       LINEPTPL_TYPE=64, &
       DIREPTPT_TYPE=71, &
       DIREDRDR_TYPE=72, &
       CECM_TYPE=85, &  !Puja [QC: 11/17]
       CEC2_TYPE=86, &  !Puja [QC: 11/17]
       VSUM_TYPE=87, &  !Puja [QC: 11/17]
       POINT_TYPE=80, &
       LAST_TYPE=80

  !               element being 
  !      deftyp      defined      in terms of
  !      -----     -----------    -----------
  !
  !                           < distance | angle >
  !                          /
  !        10           ratio     
  !                          \                    !
  !                           < distance | angle >
  !
  !                           < distance | angle | combination >
  !                          /
  !        20     combination
  !                          \                    !
  !                           < distance | angle | combination >
  !
  !                           point
  !                          /
  !        31        distance
  !                          \                    !
  !                           point
  !
  !                           point                             point
  !                          /                                 /
  !        32        distance      or, equivalently,   distance - point (line
  !                          \                                 \                    !
  !                           line                              direction (line)
  !
  !                           point                           point
  !                          /                               /
  !        33        distance      or, equivalently, distance - point (plane)
  !                          \                               \                    !
  !                           plane                           direction (plane)
  !
  !                           direction
  !                          /
  !        40           angle
  !                          \                    !
  !                           direction
  !
  !                           line
  !                          /
  !        41           angle
  !                          \                    !
  !                           line
  !
  !                           line
  !                          /
  !        42           angle
  !                          \                    !
  !                           plane
  !
  !                           plane
  !                          /
  !        43           angle
  !                          \                    !
  !                           plane
  !
  !                           point
  !                          /
  !        50           plane
  !                          \                    !
  !                           direction (of normal)
  !
  !                           point
  !                          /
  !        51           plane - point
  !                          \                    !
  !                           point
  !
  !                           point
  !                          /
  !        52           plane
  !                          \                    !
  !                           line (in plane)
  !
  !                           point
  !                          /
  !        53           plane
  !                          \                    !
  !                           line (normal)
  !
  !                           point
  !                          /
  !        54           plane - line (parallel)
  !                          \                    !
  !                           line (parallel)
  !
  !                           point
  !                          /
  !        55           plane
  !                          \                    !
  !                           plane (parallel)
  !
  !                           point
  !                          /
  !        60            line
  !                          \                     !
  !                           direction
  !
  !                           point
  !                          /
  !        61            line
  !                          \                    !
  !                           point
  ! 
  !                           point
  !                          /
  !        62            line
  !                          \                    !
  !                           line (parallel)
  !
  !                            point
  !                           /
  !        63             line - line (perpendicular)
  !                           \                    !
  !                            line (perpendicular)
  !
  !                            point
  !                           /
  !        64             line
  !                           \                    !
  !                            plane (perpendicular)
  !
  !                            point
  !                           /
  !        71        direction
  !                           \                    !
  !                            point
  !
  !                            direction (perpendicular)
  !                           /
  !        72        direction
  !                           \                    !
  !                            direction (perpendicular)
  !
  !        80        point                       (termianal node)
  !
  !     ---- As the user defines geometrical elements a graph structure
  !     ---- is constructed.
  !
  !     ---- When the reaction coordinate has been defined, the graph is
  !     ---- modified to eliminate all line and plane references by making
  !     ---- reference to points and directions instead.
  ! 
  !     ---- Then a tree is constructed so that a node has at most one
  !     ---- "up" connection and one "down" connection.
  !
  !     ---- The contents of the common block can now be described.
  !
  !     ---- maxnod      maximum number of nodes for which there is space
  !     ---- ngraph      number of nodes in the graph
  !     ---- nrxn        total number of nodes, error when exceeds maxnod
  !     ---- rxnind      the node-index (on the graph) corresponding to
  !     ----                  the reaction coordinate
  !     ---- deftyp      code indicating the node-type and the connectivity-type
  !     ----                (listed above)
  !     ---- refnod(1,i)
  !     ---- refnod(2,i)
  !     ---- refnod(3,i)   When nonzero, indicate other nodes to which
  !     ----                  node-i connects to. In other words, node-i
  !     ----                  is defined in terms of these nodes.
  !     ---- delval(.,i)  value computed for node-i
  !                       e.g   x,y,z   if node-i is a point
  !     ----                    a,b,c   if node-i is a direction
  !     ----                    d,0,0   if node-i is a distance
  !     ----                    not used if node-i is on the graph
  !     ---- delder(.,i)   derivatives of umbrella potential w.r.t 
  !     ----                    node-i
  !     ----                    not used if node-i is on the graph
  !     ---- nodnam(i)     name given by user to node-i on the graph
  !     ---- UMBFRM        indicates the form of umbrella potential
  !     ----                    umbfrm(i) = 1 for harmonic
  !     ----                    umbfrm(i) = 5 for harmonic + bias
  !     ---- DL0PTR        pointer to array of delta0s
  !     ---- delta0        a reference point to impose umbrella potential
  !     ---- KUMBPR        pointer to array of kumbs
  !     ---- kumb          the strength of U. Pot.
  !     ---- eumb          energy due to U. Pot.
  !     ---- lowdel        statistics is kept at delta values
  !     ---- hidel         starting from lowdel upto hidel at
  !     ---- deldel        interval deldel
  !
  !      SPLINE BIASING FUNCTION FOR REACTION COORDINATE
  !      JG 12/96
  !  NPOINTS          MAXIMUM CUBIC SPLINE POINTS ALLOWED
  !  NBIASP           NUMBER OF BIASING POTENTIAL ENERGY POINTS READ
  !                   NBIASP =< NPOINTS
  !  RBIAS            REACTION COORDINATE
  !  CBIAS(4,NBIASP)  CUBIC SPLINE COEFFICIENTS
  !
  !     Pointer to integer
  !  INTEGER NBIASP
  !     Pointer to pointer to real
  !  INTEGER RBIASP, CBIASP

#if KEY_TPS==1
  !     ARD and M. Hagan for TPS, May 2003
  !     Pointers to store basin limits
  !  INTEGER TPOPUN
  !
#endif 
  !                                                            
#endif /* (rxncor_fcm)*/
  !
  ! put the routines needed by everybody here:

contains

  subroutine rxncor_iniall()
    umbmdstep=0
    return
  end subroutine rxncor_iniall
  ! from rxnene.src : for dependency problem... /MH09/

  FUNCTION DDPERI(DD,PUMB)
    !
    !       Adjust deviations when order parameters are periodic
    !       ARD and Jie Hu 06-06-30
    !
    use chm_kinds
    implicit none
    real(chm_real) DD, PUMB, ddperi
    DDPERI = DD
    IF (ABS(PUMB-DD)  <  ABS(DD)) THEN
       DDPERI = DD - PUMB
    ELSE IF (ABS(PUMB+DD)  <  ABS(DD)) THEN
       DDPERI = DD + PUMB
    ENDIF
    RETURN
  END FUNCTION DDPERI



end module rxncom

