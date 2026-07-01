module pbound
  use chm_kinds
  use dimens_fcm
  implicit none
!  This fcm file contains the data structures for hardwired
!  periodic boundry conditions which should be fast.
!========================================================================
!...Control flags
!    qBoun   IMPLEMENTS THE PERIODIC BOUNDARY CONDITIONS FOR MOLECULE 
!  qTOBoun   IN A TRUNCATED OCTAHEDRAL BOX     
!  qRDBoun   IN A RHOMBIC DODECAHEDRAL BOX                          
!  qRHBoun   IN A TWO DIMENSIONAL RHOMBOIDAL BOX
!  qCUBoun   IN A CUBIC BOX
!========================================================================
!
#if KEY_PBOUND==1 /*pboundfcm*/
!
      LOGICAL qBoun, qTOBoun, qRDBoun, qRHBoun, qCUBoun

      real(chm_real) RCutsqB, RCutB
      real(chm_real) BOXINV, BOYINV, BOZINV, XSIZE, YSIZE, ZSIZE

      Integer OutUnit
!========================================================================
!    *******************************************************************
!    ** TRUNCATED OCTAHEDRON                                          **
!    ** THE BOX IS CENTRED AT THE ORIGIN. THE AXES PASS THROUGH THE   **
!    ** CENTRES OF THE SIX SQUARE FACES OF THE TRUNCATED OCTAHEDRON   **
!    ** THE CONTAINING CUBE IS OF UNIT LENGTH                         **
!    *******************************************************************
!    *******************************************************************
!    ** RHOMBIC DODECAHEDRON                                          **
!    ** THE BOX IS CENTRED AT THE ORIGIN. THE X AND Y AXES JOIN THE   **
!    ** CENTRES OF OPPOSITE FACES OF THE DODECAHEDRON. THE Z AXIS     **
!    ** JOINS OPPOSITE VERTICES OF THE RHOMBIC DODECAHEDRON THE       **
!    ** DIAGONAL OF THE RHOMBIC FACE IS OF UNIT LENGTH                **
!    ** AND THE SIDE OF THE CONTAINING CUBE IS SQRT(2.0).             **
!    ** NOTE THAT THE X AND Y AXES PASS THROUGH THE CUBE EDGES, WHILE **
!    ** THE Z AXIS PASSES THROUGH THE CUBE FACES.                     **
!    ** PRINCIPAL VARIABLES:                                          **
!    ** REAL    RT2                SQRT(2.0) TO MACHINE ACCURACY      **
!    ** REAL    RRT2               1.0/SQRT(2.0)                      **
!    *******************************************************************
!    ** RHOMBIC BOX                                                   **
!    ** PERIODIC CORRECTIONS ARE APPLIED IN TWO DIMENSIONS X, Y.      **
!    ** IN MOST APPLICATIONS THE MOLECULES WILL BE CONFINED IN THE    **
!    ** Z DIRECTION BY REAL WALLS RATHER THAN BY PERIODIC BOUNDARIES. **
!    ** THE BOX IS CENTRED AT THE ORIGIN. THE X AXIS LIES ALONG THE   **
!    ** SIDE OF THE RHOMBUS, WHICH IS OF UNIT LENGTH.                 **
!    ** PRINCIPAL VARIABLES:                                          **
!    ** REAL    RT3                SQRT(3.0) TO MACHINE ACCURACY      **
!    ** REAL    RT32               SQRT(3.0)/2.0                      **
!    ** REAL    RRT3               1.0/SQRT(3.0)                      **
!    ** REAL    RRT32              2.0/SQRT(3.0)                      **
!    *******************************************************************
!    ** CUBIC BOX                                                     **
!    ** PERIODIC CORRECTIONS ARE APPLIED IN THREE IMENSIONS X, Y, Z.  **
!    ** IN MOST APPLICATIONS THE MOLECULES WILL BE CONFINED IN THE    **
!    ** THE BOX IS CENTRED AT THE ORIGIN. THE X, Y, Z AXIS ARE OF     **
!    ** UNIT LENGTH.                                                  **
!    ** PRINCIPAL VARIABLES:                                          **
!    ** REAL    RT3                SQRT(3.0) TO MACHINE ACCURACY      **
!    ** REAL    RT32               SQRT(3.0)/2.0                      **
!    ** REAL    RRT3               1.0/SQRT(3.0)                      **
!    ** REAL    RRT32              2.0/SQRT(3.0)                      **
!    *******************************************************************
!========================================================================

!    For TOBound:       
!    *************
        real(chm_real)    R75
        PARAMETER ( R75 = 4.0 / 3.0 )

!    For RDBound:       
!    *************
        real(chm_real)    RT2, RRT2
        PARAMETER ( RT2 = 1.4142136, RRT2 = 1.0 / RT2 )

!    For RHBound:       
!    *************
        real(chm_real)    RT3, RRT3, RT32, RRT32
        PARAMETER ( RT3 = 1.7320508, RRT3 = 1.0 / RT3 )
        PARAMETER ( RT32 = RT3 / 2.0, RRT32 = 1.0 / RT32 )
!

contains

  subroutine pbound_init()
    use number,only:zero
    qBoun   = .FALSE.
    qTOBoun = .FALSE.
    qRDBoun = .FALSE.
    qRHBoun = .FALSE.
    qCUBoun = .FALSE.
    xsize = zero
    ysize = zero
    zsize = zero
    return
  end subroutine pbound_init

  subroutine pbound_getvol(vol)
    implicit none
    real(chm_real),intent(out) :: vol
    
    vol = xsize*ysize*zsize
    
    return
  end subroutine pbound_getvol

#else /* (pboundfcm)*/
  logical, parameter :: qBoun = .false.
#endif /* (pboundfcm)*/
!
end module pbound

