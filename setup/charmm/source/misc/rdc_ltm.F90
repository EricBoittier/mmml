module rdc
   use chm_kinds
   use dimens_fcm
   use chm_types

   implicit none

! data structure (COMMON block) for RDC constraints
!
! NRDC1          : actual number of RDC constraints
! NSET           : nmber of RDC sets
! RDCALIS,RDCBLIS: atom lists for RDC indicies
! IGAMMA,JGAMMA  : gyromagnetic ratio for RDC calculation
! KHAR           : RDC harmonic force constatn
! KASY           : RDC asymtotic force constant
! REXP           : experimental RDC values
! DELL,DELU      : Lower and upper bound for experimental RDC
! LHAR           : a length of harmonic potential
! EXPO           : a soft-exponent for asymptotic potential

#if KEY_NOMISC==0 && KEY_RDC==1
   integer, parameter :: default_maxrdc = 1000

      INTEGER MAXRDC, MAXSET, nset, rres

      integer,allocatable,dimension(:) :: NRDC1,NRDCmp1
      integer,allocatable,dimension(:) :: RDCaLIS,RDCbLIS,MPRDC,EXPO
      real(chm_real),allocatable,dimension(:) :: IGAMMA,JGAMMA,REXP,SRDC
      real(chm_real),allocatable,dimension(:) :: LHAR,KHAR,KASY
      real(chm_real),allocatable,dimension(:) :: DELL,DELU

      INTEGER IRDCall

      REAL*8  IGAMMA1, JGAMMA1
! Molecular Frame
      real(chm_real),allocatable,dimension(:) :: MF,MFT,EV

! R matrix
      real(chm_real),allocatable,dimension(:) :: RDCRMAT,RDCRrMAT
      real(chm_real),allocatable,dimension(:) :: SMAT,CT,dred,DCON
      real(chm_real),allocatable,dimension(:) :: RDCVECX,RDCVECY,RDCVECZ,RDCDIST
      real(chm_real),allocatable,dimension(:) :: RDCRMATmp,RDCVECmpX,RDCVECmpY,RDCVECmpZ,RDCDISTmp

! Alignment tensor
      real(chm_real),allocatable,dimension(:) :: SI,AN,WI,W55,UT,V,VWI,UW,WIUT
      LOGICAL QRDC,QRMF,QFIXA,QFIXB,QSLOW,QSRDC
      LOGICAL QKEY

contains

   subroutine rdcset
      use dimens_fcm
      use psf
      use comand
      use string
      use memory
      implicit none

      call rdcset1

      return
   end subroutine rdcset

   SUBROUTINE RDC1(NATOM,X,Y,Z,AMASS,ERDC,DX,DY,DZ)

      INTEGER NATOM
      REAL*8  X(*),Y(*),Z(*),AMASS(*)
      REAL*8  DX(*),DY(*),DZ(*)
      REAL*8  ERDC

      call rdc2(NATOM,X,Y,Z,AMASS,ERDC,DX,DY,DZ)
      return
   end subroutine

   subroutine rdc_init()
     use memory, only: chmalloc
     implicit none

     irdcall = 0

     CALL chmalloc('rdc.src','RDC','RDCaLST',NSET*MAXRDC, intg=RDCaLIS)
     CALL chmalloc('rdc.src','RDC','RDCbLST',NSET*MAXRDC, intg=RDCbLIS)
     CALL chmalloc('rdc.src','RDC','MPRDC'  ,NSET*MAXRDC, intg=MPRDC  )
     CALL chmalloc('rdc.src','RDC','IGAMMA' ,NSET*MAXRDC, crl=IGAMMA  )
     CALL chmalloc('rdc.src','RDC','JGAMMA' ,NSET*MAXRDC, crl=JGAMMA  )
     CALL chmalloc('rdc.src','RDC','SRDC'   ,NSET*MAXRDC, crl=SRDC    )
     CALL chmalloc('rdc.src','RDC','REXP',NSET*MAXRDC, crl=REXP)
     CALL chmalloc('rdc.src','RDC','DELL',NSET*MAXRDC, crl=DELL)
     CALL chmalloc('rdc.src','RDC','DELU',NSET*MAXRDC, crl=DELU)
     CALL chmalloc('rdc.src','RDC','LHAR',NSET*MAXRDC, crl=LHAR)
     CALL chmalloc('rdc.src','RDC','KHAR',NSET*MAXRDC, crl=KHAR)
     CALL chmalloc('rdc.src','RDC','KASY',NSET*MAXRDC, crl=KASY)
     CALL chmalloc('rdc.src','RDC','EXPO',NSET*MAXRDC, intg=EXPO)
     CALL chmalloc('rdc.src','RDC','NRDC1'  ,NSET, intg=NRDC1)
     CALL chmalloc('rdc.src','RDC','NRDCmp1',NSET, intg=NRDCmp1)
   end subroutine rdc_init

   subroutine rdc_add_storage()
     use memory, only: chmrealloc
     implicit none

     maxrdc = 2 * maxrdc

     CALL chmrealloc('rdc.src','RDC','RDCaLST',NSET*MAXRDC, intg=RDCaLIS)
     CALL chmrealloc('rdc.src','RDC','RDCbLST',NSET*MAXRDC, intg=RDCbLIS)
     CALL chmrealloc('rdc.src','RDC','MPRDC'  ,NSET*MAXRDC, intg=MPRDC  )
     CALL chmrealloc('rdc.src','RDC','IGAMMA' ,NSET*MAXRDC, crl=IGAMMA  )
     CALL chmrealloc('rdc.src','RDC','JGAMMA' ,NSET*MAXRDC, crl=JGAMMA  )
     CALL chmrealloc('rdc.src','RDC','SRDC'   ,NSET*MAXRDC, crl=SRDC    )
     CALL chmrealloc('rdc.src','RDC','REXP',NSET*MAXRDC, crl=REXP)
     CALL chmrealloc('rdc.src','RDC','DELL',NSET*MAXRDC, crl=DELL)
     CALL chmrealloc('rdc.src','RDC','DELU',NSET*MAXRDC, crl=DELU)
     CALL chmrealloc('rdc.src','RDC','LHAR',NSET*MAXRDC, crl=LHAR)
     CALL chmrealloc('rdc.src','RDC','KHAR',NSET*MAXRDC, crl=KHAR)
     CALL chmrealloc('rdc.src','RDC','KASY',NSET*MAXRDC, crl=KASY)
     CALL chmrealloc('rdc.src','RDC','EXPO',NSET*MAXRDC, intg=EXPO)
     CALL chmrealloc('rdc.src','RDC','NRDC1'  ,NSET, intg=NRDC1)
     CALL chmrealloc('rdc.src','RDC','NRDCmp1',NSET, intg=NRDCmp1)
   end subroutine rdc_add_storage

   subroutine rdc_uninit(nrdc, nrdcmp)
     use memory, only: chmdealloc
     implicit none

     integer, intent(inout) :: nrdc, nrdcmp

     CALL chmdealloc('rdc.src','RDC','RDCaLST',NSET*MAXRDC, intg=RDCaLIS)
     CALL chmdealloc('rdc.src','RDC','RDCbLST',NSET*MAXRDC, intg=RDCbLIS)
     CALL chmdealloc('rdc.src','RDC','MPRDC'  ,NSET*MAXRDC, intg=MPRDC  )
     CALL chmdealloc('rdc.src','RDC','IGAMMA' ,NSET*MAXRDC, crl=IGAMMA  )
     CALL chmdealloc('rdc.src','RDC','JGAMMA' ,NSET*MAXRDC, crl=JGAMMA  )
     CALL chmdealloc('rdc.src','RDC','SRDC'   ,NSET*MAXRDC, crl=SRDC    )
     CALL chmdealloc('rdc.src','RDC','REXP',NSET*MAXRDC, crl=REXP)
     CALL chmdealloc('rdc.src','RDC','DELL',NSET*MAXRDC, crl=DELL)
     CALL chmdealloc('rdc.src','RDC','DELU',NSET*MAXRDC, crl=DELU)
     CALL chmdealloc('rdc.src','RDC','LHAR',NSET*MAXRDC, crl=LHAR)
     CALL chmdealloc('rdc.src','RDC','KHAR',NSET*MAXRDC, crl=KHAR)
     CALL chmdealloc('rdc.src','RDC','KASY',NSET*MAXRDC, crl=KASY)
     CALL chmdealloc('rdc.src','RDC','EXPO',NSET*MAXRDC, intg=EXPO)
     CALL chmdealloc('rdc.src','RDC','NRDC1'  ,NSET, intg=NRDC1)
     CALL chmdealloc('rdc.src','RDC','NRDCmp1',NSET, intg=NRDCmp1)

     CALL chmdealloc('rdc.src','RDC','MF' ,3*3, crl=MF)
     CALL chmdealloc('rdc.src','RDC','MFT',3*3, crl=MFT)
     CALL chmdealloc('rdc.src','RDC','EV' ,3  , crl=EV)

     CALL chmdealloc('rdc.src','RDC','RDCVECX', NRDC,crl=RDCVECX )
     CALL chmdealloc('rdc.src','RDC','RDCVECY', NRDC,crl=RDCVECY )
     CALL chmdealloc('rdc.src','RDC','RDCVECZ', NRDC,crl=RDCVECZ )
     CALL chmdealloc('rdc.src','RDC','RDCDIST', NRDC,crl=RDCDIST )
     CALL chmdealloc('rdc.src','RDC','RDCRMAT', 6*NRDC,crl=RDCRMAT )
     CALL chmdealloc('rdc.src','RDC','RDCRrMAT',6*NRDC,crl=RDCRrMAT)
     CALL chmdealloc('rdc.src','RDC','SMAT',    5*NRDC,crl=SMAT    )
     CALL chmdealloc('rdc.src','RDC','CT',      6*NRDC,crl=CT      )
     CALL chmdealloc('rdc.src','RDC','DRED',    NRDC,crl=DRED    )
     CALL chmdealloc('rdc.src','RDC','DCON',    NRDC,crl=DCON    )

     IF (allocated(RDCRMATmp)) THEN ! these are always allocated together
        CALL chmdealloc('rdc.src','RDC','RDCRMATmp', 6*NRDCmp,crl=RDCRMATmp)
        CALL chmdealloc('rdc.src','RDC','RDCVECmpX', NRDCmp,crl=RDCVECmpX)
        CALL chmdealloc('rdc.src','RDC','RDCVECmpY', NRDCmp,crl=RDCVECmpY)
        CALL chmdealloc('rdc.src','RDC','RDCVECmpZ', NRDCmp,crl=RDCVECmpZ)
        CALL chmdealloc('rdc.src','RDC','RDCDISTmp', NRDCmp,crl=RDCDISTmp)
     ENDIF

     CALL chmdealloc('rdc.src','RDC','AN',  3*3,crl=AN  )
     CALL chmdealloc('rdc.src','RDC','WI',  5*5,crl=WI  )
     CALL chmdealloc('rdc.src','RDC','W55', 5*5,crl=W55 )
     CALL chmdealloc('rdc.src','RDC','V',   5*5,crl=V   )
     CALL chmdealloc('rdc.src','RDC','VWI', 5*5,crl=VWI )
     CALL chmdealloc('rdc.src','RDC','UW',  5*NRDC,crl=UW  )
     CALL chmdealloc('rdc.src','RDC','WIUT',5*NRDC,crl=WIUT)

     qrdc = .false.
     nset = 0
     nrdc = 0
     nrdcmp = 0
     maxrdc = default_maxrdc

   end subroutine rdc_uninit

#else /* KEY_NOMISC, KEY_RDC */

contains

   subroutine rdcset
      call WRNDIE(-1,'<CHARMM>','RDC code is not compiled.')
      return
   end subroutine rdcset

#endif /* KEY_NOMISC, KEY_RDC */

end module rdc
