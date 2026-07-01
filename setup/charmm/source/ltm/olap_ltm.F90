module olap
  use chm_kinds
  use dimens_fcm

  implicit none
  character(len=*),private,parameter :: file_name   ="olap_ltm.src"

!
#if KEY_OVERLAP==1 /*overlap_fcm*/
!     This is the overlap module data block
!
!     NSYST      Total number of systems
!     QOLAP      Flag: Are we using this method?
!     NOLAP      Array of pointers into IOLAP(heap) array
!     IOLAP      Pointer into the HEAP array for IOLAP
!     SUPERW     Is the total weight put on the OLAP "energy" term
!     MAXS       Maximum number of subsystems
!     OLAPG      GAMMA value for electrostatic field
!     WOLAPG     Weighting factor for electrostatic field formula
!     VOLWF      Weighting factor for the volume overlap term
!     CHAWF      Weighting factor for the charge overlap term
!     ESPWF      Weighting factor for the el.stat. potential overlap term
!
      INTEGER NSYST,LOLAP
      INTEGER,allocatable,dimension(:) :: NOLAP
      INTEGER MAXS,MAXINDS
      PARAMETER(MAXS=7,MAXINDS=MAXS*(MAXS+1)/2)
!
      LOGICAL QOLAP,QDOLAP
!
      real(chm_real),allocatable,dimension(:) :: CESP
      real(chm_real) SUPERW,S(MAXINDS),SYSW(MAXS),OLAPG,WOLAPG,VOLWF,CHAWF,ESPWF
!
!
contains

  subroutine olap_init()
    qolap=.false.
    qdolap=.false.
    nsyst=0
    return
  end subroutine olap_init

  subroutine allocate_olap()
    use memory
    use psf
    character(len=*),parameter :: routine_name="allocate_olap"
    call chmalloc(file_name,routine_name,'cesp  ',natom,crl=cesp)
    call chmalloc(file_name,routine_name,'nolap ',natom+1,intg=nolap)
    return
  end subroutine allocate_olap

#endif /* (overlap_fcm)*/

end module olap

