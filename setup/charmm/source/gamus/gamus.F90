module gamusmodule
  use chm_kinds
  use dimens_fcm
  use gmmutil
  implicit none

#if KEY_GAMUS==1 /*gamus*/
! Data for GAMUS -- Maragakis, van der Vaart and Karplus, JPC B xxxx, xxxx (2009)
! DOGAMUS -- Is GAMUS active?
! The following data is part of the GAMUS potential file.  MGAMUS and IDGAMUS are read by GAMUSALLOC, the rest by read_gmm_parameter in gmmutil.src
! MGAMUS -- Number of Gaussians used to fit the probability distribution
! IDGAMUS -- Number of reaction coordinates in use (dimensionality of the free energy surface)
! params%gamma -- gamma likelihood value
! pp%weight -- heap pointer to the weights on each Gaussian
! params%mean -- heap pointer to the mean value of each Gaussian
! gamus%sinv -- heap pointer to the variance-covariance matrix of each Gaussian.
! params%lndet -- normalization factor (related to the determinant of params%sinv

! the following are command line options for GAMUS INIT or GAMUS BIAS
! IGAMUSW -- unit for writing energies and reaction coordinates (specify with WUNI)
! IGAMUSF -- frequency for writing energies and reaction coordinates (IFRQ)
! IGAMUSU1 and IGAMUSW1 -- unit for reading potential and writing energies and reaction coordinates when using GAMUS BIAS (RUNI and WUNI)
! IGAMUSQ1 -- unit for reading energies and reaction coordinates with GAMUS BIAS (QUNI)
! GAMUSKT -- specified temperature (TEMP) stored as kt

! these are from GAMUS DIHE
! GAMUSDI -- number of dihedral coordinates in use.
! IGAMUS, JGAMUS, KGAMUS, LGAMUS -- heap pointers for atom numbers of dihedral (index from 1 to GAMUSDI)
! IPHIGAMUS -- index of the dihedral within the reaction coordinates (for now IPHIGAMUS(I)=I always)

! GAMUSEN calls UM1PHI to calculate the dihedral angles for all coordinates, and calls UM2PHI after the energy has been calculated.
! UM2PHI converts the dV/dqi for each reaction coordinate to cartesian forces and adds them to DX, DY, DZ.

! GAMUSQ -- current rxn coordinate values
! GAMUSDQ -- dV/dqi for each reaction coordinate
! GAMUSW1, GAMUSW2, GAMUSW3 -- work arrays for storing partial results for GAMUSEN.

      logical :: dogamus
      integer :: mgamus,idgamus, gamusdi
      type(gmm_parameter) :: params
!      real(kind=chm_real),allocatable,dimension(:) :: params%weight, params%lndet, params%mean, params%sinv
!      real(kind=chm_real),allocatable,dimension(:,:) :: params%mean
!      real(kind=chm_real),allocatable,dimension(:,:,:) :: params%sinv
      real(kind=chm_real) :: gamuskt
      integer :: igamusw, igamusf, igamusu1, igamusw1, igamusq1
      integer,allocatable,dimension(:) :: igamus,jgamus,kgamus,lgamus,iphigamus
      real(kind=chm_real),allocatable,dimension(:) :: gamusq, gamusdq, work1, work2, dnom
      
!      LOGICAL DOGAMUS
!      integer MGAMUS,IDGAMUS
!      integer params%weight,params%mean,params%sinv,params%lndet,IGAMUSW,IGAMUSF,IPHIGAMUS,
!     $     IGAMUSQ1,IGAMUSW1,IGAMUSU1
!      REAL*8  GAMUSKT,params%gamma
!      integer GAMUSDI
!      integer IGAMUS,JGAMUS,KGAMUS,LGAMUS,GAMUSQ,GAMUSDQ
!      integer GAMUSW1,GAMUSW2,GAMUSW3
!      COMMON /GAMUSSTUF0/GAMUSKT,params%gamma
!      COMMON /GAMUSSTUF1/MGAMUS,IDGAMUS,params%weight,params%mean,params%sinv,params%lndet,IGAMUSW,
!     $     IGAMUSF,IPHIGAMUS,GAMUSQ,GAMUSDQ,GAMUSW1,GAMUSW2,GAMUSW3,IGAMUSQ1,
!     $     IGAMUSW1,IGAMUSU1
!      COMMON /GAMUSSTUF2/DOGAMUS
!      COMMON /GAMUSDIHE/GAMUSDI,IGAMUS,JGAMUS,KGAMUS,LGAMUS

#endif /* (gamus)*/

contains

#if KEY_GAMUS==1 /*gamus*/

      SUBROUTINE GAMUSINIT(COMLYN,COMLEN)
      use string
      use consta
      use number
      use stream
#if KEY_PARALLEL==1
      use parallel 
#endif
#if KEY_ENSEMBLE==1
      use ensemble 
#endif
      implicit none
!
      character(len=*) :: comlyn
      integer :: comlen
      character(len=4) :: wrd
      integer :: igamusu
!
! --> unit number for read in GAMUS description:
! --> temperature
! O-> unit number for writ energy/q during simulation
! O-> output update frequency
! O-> clear 
!
!     THE FIRST WORD IN THE COMMAND LINE DETERMINES WHAT IS BEING DONE.
!
      WRD=NEXTA4(COMLYN,COMLEN)
!
!     BRANCH TO THE APPROPRIATE SECTION.
      IF (WRD.EQ.'INIT') THEN
         IF (DOGAMUS) THEN
            CALL WRNDIE(-10,'<GAMUS>','double initialization')
         ENDIF
         DOGAMUS = .TRUE.
         GAMUSDI = 0
!
         GAMUSKT = -ONE
         GAMUSKT = GTRMF(COMLYN,COMLEN,'TEMP',GAMUSKT)
         IF (GAMUSKT.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid temperature')
         ENDIF
         GAMUSKT = KBOLTZ * GAMUSKT 
!
         IGAMUSU = -1
         IGAMUSU = GTRMI(COMLYN,COMLEN,'RUNI',IGAMUSU)
         IF (IGAMUSU.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid read unit')
         ENDIF
!
         IGAMUSW = -1
         IGAMUSW = GTRMI(COMLYN,COMLEN,'WUNI',IGAMUSW)
         IF (IGAMUSW.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid write unit')
         ENDIF
!
         IGAMUSF = 1
         IGAMUSF = GTRMI(COMLYN,COMLEN,'IFRQ',IGAMUSF)
!     .  allocate all the arrays
         CALL GAMUSALLOC(IGAMUSU)
#if KEY_ENSEMBLE==0
         IF((iolev.gt.0)) THEN 
#endif
            IF ((IGAMUSW.GT.0))THEN
            WRITE(IGAMUSW,'("# KT = ",F15.8," TEMP = ",F15.8," NEXT: BIASING ENERGY - Q(1) .. Q(",I4,")")') & 
              GAMUSKT,GAMUSKT/KBOLTZ,IDGAMUS
            ENDIF
#if KEY_ENSEMBLE==0
         ENDIF 
#endif
!     
      ELSEIF (WRD.EQ.'BIAS') THEN
#if KEY_ENSEMBLE==1
         if (whoiam.ne.0) return 
#endif
         IF (DOGAMUS) THEN
            CALL CLEARGAMUS
         ENDIF
!
         IGAMUSU1 = -1
         IGAMUSU1 = GTRMI(COMLYN,COMLEN,'RUNI',IGAMUSU1)
         IF (IGAMUSU1.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid read unit')
         ENDIF
!
         IGAMUSW1 = -1
         IGAMUSW1 = GTRMI(COMLYN,COMLEN,'WUNI',IGAMUSW1)
         IF (IGAMUSW1.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid write unit')
         ENDIF
!     
         IGAMUSQ1 = -1
         IGAMUSQ1 = GTRMI(COMLYN,COMLEN,'QUNI',IGAMUSQ1)
         IF (IGAMUSQ1.LE.ZERO) THEN
            CALL WRNDIE(-10,'<GAMUS>','Invalid read unit')
         ENDIF
!
#if KEY_ENSEMBLE==1
         !if (whoiam.eq.0) then 
#endif
#if KEY_PARALLEL==1
         if (mynod.eq.0) then 
#endif
         CALL GAMUSALLOC(IGAMUSU1)
         CALL GAMUSBIAS
#if KEY_ENSEMBLE==1
         !endif 
#endif
#if KEY_PARALLEL==1
         endif 
#endif
         CALL CLEARGAMUS
         DOGAMUS=.FALSE.

      ELSEIF (WRD.EQ.'CLEA') THEN
         IF (DOGAMUS) THEN
            CALL CLEARGAMUS
         ENDIF
         DOGAMUS = .FALSE.
!
      ELSEIF (WRD.EQ.'DIHE') THEN
         IF (.NOT.DOGAMUS) THEN
            CALL WRNDIE(-10,'<GAMUS>','GAMUS not initialized')
         ENDIF
         GAMUSDI = GAMUSDI + 1
         IF(GAMUSDI.GT.IDGAMUS)THEN
            CALL WRNDIE(-10,'<GAMUS>','Too many dihedrals')
         ENDIF
         call gamusini1(comlyn,comlen)
!         CALL GAMUSINI1(GAMUSDI,HEAP(IGAMUS),HEAP(JGAMUS),HEAP(KGAMUS),
!     $        HEAP(LGAMUS),HEAP(IPHIGAMUS),COMLYN,COMLEN)
      ELSEIF (WRD.EQ.'CHEC') THEN
         IF (GAMUSDI.NE.IDGAMUS) THEN
            CALL WRNDIE(-10,'<GAMUS>','incorrect number of dihedrals')
         ENDIF
      elseif (wrd.eq.'WRIT') then
         igamusu=gtrmi(comlyn,comlen,'UNIT',-1)
         call write_gmm_parameter(params,igamusu)
      elseif (wrd.eq.'FIT') then
         call gamusfit(comlyn,comlen)
      !elseif (wrd.eq.'MARE') then
         !call maretest
      elseif (wrd.eq.'REWE') then
         call gamusreweight(comlyn,comlen)
      elseif (wrd.eq.'INFO') then
         call gamusinfo(comlyn,comlen)
      ENDIF
!
      RETURN
      END SUBROUTINE GAMUSINIT
!
!----------------------------------------------------------
!
      SUBROUTINE GAMUSALLOC(IUNIT)
!
      use chm_kinds
      use memory
      use stream
#if KEY_ENSEMBLE==1
      use ensemble 
#endif
      use param_store, only: set_param
      use parallel,only:psnd4,psnd8

      implicit none
      integer :: iunit
!
      WRITE(OUTU,'(/," ALLOCATING GAMUS MEMORY")')
!
      mgamus=0
      idgamus=0
#if KEY_ENSEMBLE==0
      if (iolev.gt.0) then 
#endif
          rewind(iunit)
          READ(IUNIT,*) MGAMUS, IDGAMUS
#if KEY_ENSEMBLE==0
      endif 
#endif
#if KEY_ENSEMBLE==1
!      write(outu,*) "gamusalloc: ",whoiam,mgamus,idgamus 
#endif
      call flush(outu)
#if KEY_PARALLEL==1
      call psnd4(mgamus,1) 
      call psnd4(idgamus,1)
#endif 
      call alloc_gmm_parameter(mgamus,idgamus,params)
      call read_gmm_parameter(params,iunit)
#if KEY_PARALLEL==1
      call broadcast_gmm_parameter(params) 
#endif
      call set_param("NGAUSS",mgamus)
      call set_param("NDIM",idgamus)

      call chmalloc('gamus.src','gamusalloc','igamus',idgamus,intg=igamus)
      call chmalloc('gamus.src','gamusalloc','jgamus',idgamus,intg=jgamus)
      call chmalloc('gamus.src','gamusalloc','kgamus',idgamus,intg=kgamus)
      call chmalloc('gamus.src','gamusalloc','lgamus',idgamus,intg=lgamus)
      call chmalloc('gamus.src','gamusalloc','iphigamus',idgamus,intg=iphigamus)
      call chmalloc('gamus.src','gamusalloc','gamusq',idgamus,crl=gamusq)
      call chmalloc('gamus.src','gamusalloc','gamusdq',idgamus,crl=gamusdq)
      call chmalloc('gamus.src','gamusalloc','work1',idgamus,crl=work1)
      call chmalloc('gamus.src','gamusalloc','work2',idgamus,crl=work2)
      call chmalloc('gamus.src','gamusalloc','dnom',idgamus,crl=dnom)

!      IGAMUS = ALLHP(INTEG4(IDGAMUS),'gamus.src','GAMUSINIT','IGAMUS')
!      JGAMUS = ALLHP(INTEG4(IDGAMUS),'gamus.src','GAMUSINIT','JGAMUS')
!      KGAMUS = ALLHP(INTEG4(IDGAMUS),'gamus.src','GAMUSINIT','KGAMUS')
!      LGAMUS = ALLHP(INTEG4(IDGAMUS),'gamus.src','GAMUSINIT','LGAMUS')
!      IPHIGAMUS=ALLHP(INTEG4(IDGAMUS),'gamus.src','GAMUSINIT','IPHIGAMUS')
!      GAMUSQ = ALLHP(IREAL8(IDGAMUS),'gamus.src','GAMUSINIT','GAMUSQ')
!     energy-related arrays
!      GAMUSDQ= ALLHP(IREAL8(IDGAMUS),'gamus.src','GAMUSINIT','GAMUSDQ')
!      GAMUSW1= ALLHP(IREAL8(IDGAMUS),'gamus.src','GAMUSINIT','GAMUSW1')
!      GAMUSW2= ALLHP(IREAL8(IDGAMUS),'gamus.src','GAMUSINIT','GAMUSW2')
!      GAMUSW3= ALLHP(IREAL8(IDGAMUS),'gamus.src','GAMUSINIT','GAMUSW3')
      END SUBROUTINE GAMUSALLOC

!----------------------------------------------------------
!
      SUBROUTINE CLEARGAMUS
      use chm_kinds
      use memory
      use stream
      implicit none
!
      WRITE(OUTU,'(/," CLEARING GAMUS MEMORY")')
      !call chmdealloc('gamus.src','gamusalloc','params%weight',mgamus,crl=params%weight)
      !call chmdealloc('gamus.src','gamusalloc','params%mean',mgamus*idgamus,crl=params%mean)
      !call chmdealloc('gamus.src','gamusalloc','params%sinv',mgamus*idgamus*idgamus,crl=params%sinv)
      !call chmdealloc('gamus.src','gamusalloc','params%lndet',mgamus,crl=params%lndet)
      call dealloc_gmm_parameter(params)
      call chmdealloc('gamus.src','gamusalloc','igamus',idgamus,intg=igamus)
      call chmdealloc('gamus.src','gamusalloc','jgamus',idgamus,intg=jgamus)
      call chmdealloc('gamus.src','gamusalloc','kgamus',idgamus,intg=kgamus)
      call chmdealloc('gamus.src','gamusalloc','lgamus',idgamus,intg=lgamus)
      call chmdealloc('gamus.src','gamusalloc','iphigamus',idgamus,intg=iphigamus)
      call chmdealloc('gamus.src','gamusalloc','gamusq',idgamus,crl=gamusq)
      call chmdealloc('gamus.src','gamusalloc','gamusdq',idgamus,crl=gamusdq)
      call chmdealloc('gamus.src','gamusalloc','work1',idgamus,crl=work1)
      call chmdealloc('gamus.src','gamusalloc','work2',idgamus,crl=work2)
      call chmdealloc('gamus.src','gamusalloc','dnom',idgamus,crl=dnom)

!      CALL FREHP(params%weight,IREAL8(MGAMUS),'gamus.src','CLEARGAMUS',
!     $     'params%weight')
!      CALL FREHP(params%mean ,IREAL8(MGAMUS*IDGAMUS),'gamus.src',
!     $     'CLEARGAMUS','params%mean')
!      CALL FREHP(params%sinv ,IREAL8(MGAMUS*IDGAMUS*IDGAMUS),'gamus.src',
!     $     'CLEARGAMUS','params%sinv')
!      CALL FREHP(params%lndet,IREAL8(MGAMUS),'gamus.src','CLEARGAMUS',
!     $     'params%lndet')
!      CALL FREHP(IGAMUS,INTEG4(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'IGAMUS')
!      CALL FREHP(JGAMUS,INTEG4(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'JGAMUS')
!      CALL FREHP(KGAMUS,INTEG4(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'KGAMUS')
!      CALL FREHP(LGAMUS,INTEG4(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'LGAMUS')
!      CALL FREHP(IPHIGAMUS,INTEG4(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'IPHIGAMUS')
!      CALL FREHP(GAMUSQ,IREAL8(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'GAMUSQ')
!      CALL FREHP(GAMUSDQ,IREAL8(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'GAMUSDQ')
!      CALL FREHP(GAMUSW1,IREAL8(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'GAMUSW1')
!      CALL FREHP(GAMUSW2,IREAL8(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'GAMUSW2')
!      CALL FREHP(GAMUSW3,IREAL8(IDGAMUS),'gamus.src','CLEARGAMUS',
!     $     'GAMUSW3')
!
      END SUBROUTINE CLEARGAMUS
!----------------------------------------------------------
!
      SUBROUTINE GAMUSINI1(comlyn,comlen) !(GAMUSDI,IGAMUS,JGAMUS,KGAMUS,LGAMUS,IPHIGAMUS,
!     $     COMLYN,COMLEN)
      use chm_kinds
      use psf
      use string
      use stream
      use memory
      use chutil, only: atomid
      use select, only: nxtatm
      implicit none
      character(len=*) :: comlyn
      integer :: comlen
      integer,dimension(:),allocatable :: scratch
      integer :: qat(4),nqat,i
!
!      integer GAMUSDI
!      integer IGAMUS(*),JGAMUS(*),KGAMUS(*),LGAMUS(*),IPHIGAMUS(*)
!      CHARACTER COMLYN*(*)
!      integer   COMLEN
!     LOCALS:
!      integer   SCRATCH
!      integer   QAT(4),NQAT,I
      character(len=idleng) SIDI,RIDI,RENI,ACI,SIDJ,RIDJ,RENJ,ACJ
       character(len=idleng) SIDK,RIDK,RENK,ACK,SIDL,RIDL,RENL,ACL
!      CHARACTER*8 SIDI,RIDI,RENI,ACI,SIDJ,RIDJ,RENJ,ACJ
!      CHARACTER*8 SIDK,RIDK,RENK,ACK,SIDL,RIDL,RENL,ACL
!      
      IPHIGAMUS(GAMUSDI)=GAMUSDI
      call chmalloc('gamus.src','gamusini1','scratch',natom,intg=scratch)
!      SCRATCH = ALLHP(INTEG4(NATOM),'gamus.src','GAMUSINIT','SCRATCH')
      CALL NXTATM(QAT,NQAT,4,COMLYN,COMLEN,scratch,SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
      call chmdealloc('gamus.src','gamusini1','scratch',natom,intg=scratch)
!      CALL FREHP(SCRATCH,INTEG4(NATOM),'gamus.src','GAMUSINIT',
!     $     'SCRATCH')
      IF(NQAT.NE.4) GOTO 800
      IGAMUS(GAMUSDI)=QAT(1)
      I=IGAMUS(GAMUSDI)
      IF(I.LE.0.OR.I.GT.NATOMT) GOTO 800
      CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
      JGAMUS(GAMUSDI)=QAT(2)
      I=JGAMUS(GAMUSDI)
      IF(I.LE.0.OR.I.GT.NATOMT) GOTO 800
      CALL ATOMID(I,SIDJ,RIDJ,RENJ,ACJ)
      KGAMUS(GAMUSDI)=QAT(3)
      I=KGAMUS(GAMUSDI)
      IF(I.LE.0.OR.I.GT.NATOMT) GOTO 800
      CALL ATOMID(I,SIDK,RIDK,RENK,ACK)
      LGAMUS(GAMUSDI)=QAT(4)
      I=LGAMUS(GAMUSDI)
      IF(I.LE.0.OR.I.GT.NATOMT) GOTO 800
      CALL ATOMID(I,SIDL,RIDL,RENL,ACL)
!
      IF(PRNLEV.GE.2) WRITE(OUTU,490) GAMUSDI,RIDI(1:idleng),SIDI(1:idleng),ACI(1:idleng),&
           RIDJ(1:idleng),SIDJ(1:idleng),ACJ(1:idleng),RIDK(1:idleng),SIDK(1:idleng),ACK(1:idleng),&
           RIDL(1:idleng),SIDL(1:idleng),ACL(1:idleng)
 490     FORMAT(' NEW GAMUS DIHEDRAL ADDED'/I5,':',4X,3(A,1X,A,1X,A,'/  '),(A,1X,A,1X,A))
!
!      
      RETURN
 800  CALL WRNDIE(-10,'<GAMUS>','Wrong dihedral')
      END SUBROUTINE GAMUSINI1
!
!----------------------------------------------------------
!
!
!----------------------------------------------------------
!
      SUBROUTINE GAMUSEN(natom,X,Y,Z,DX,DY,DZ,EBIAS)
      use chm_kinds
      implicit none
!
      integer :: natom
      real(kind=chm_real), dimension(natom) :: x,y,z,dx,dy,dz
!      REAL*8 X(*),Y(*),Z(*),DX(*),DY(*),DZ(*),ETOT(*)
      real(kind=chm_real) :: ebias
      real(kind=chm_real) :: dummy1
      logical :: dummy2

!      REAL*8 DUMMY1
!      LOGICAL DUMMY2
!      integer I,J
!      REAL*8 Q1,Q2
!
      call um1phi(igamus,jgamus,kgamus,lgamus,iphigamus,gamusq,idgamus,x,y,z)
      call gamusen1(gamusq,gamusdq,ebias)

!      CALL UM1PHI(HEAP(IGAMUS),HEAP(JGAMUS),HEAP(KGAMUS),HEAP(LGAMUS),
!     $     HEAP(IPHIGAMUS),HEAP(GAMUSQ),IDGAMUS,X,Y,Z)
!      CALL GAMUSEN1(HEAP(GAMUSQ),HEAP(GAMUSDQ),EBIAS,
!     $     params%gamma,HEAP(params%weight),HEAP(params%mean),HEAP(params%sinv),HEAP(params%lndet),
!     $     MGAMUS,IDGAMUS,GAMUSKT,HEAP(GAMUSW1),HEAP(GAMUSW2),HEAP(GAMUSW3))
!
      DUMMY2=.FALSE.
      call um2phi(dummy1,igamus,jgamus,kgamus,lgamus,iphigamus,idgamus,&
          gamusdq,dx,dy,dz,x,y,z,dummy2,dummy1)
!      CALL UM2PHI(DUMMY1,HEAP(IGAMUS),HEAP(JGAMUS),HEAP(KGAMUS),
!     $     HEAP(LGAMUS),HEAP(IPHIGAMUS),IDGAMUS,HEAP(GAMUSDQ),
!     $     DX,DY,DZ,X,Y,Z,DUMMY2,DUMMY1)!

      END SUBROUTINE GAMUSEN
!
!----------------------------------------------------------
!
!
!----------------------------------------------------------
!
      subroutine gamusen1(q, dq, en)
!      SUBROUTINE GAMUSEN1(Q,DQ,EN,params%gamma,params%weight,params%mean,params%sinv,params%lndet,
!     $     M, ID, GAMUSKT, WORK1, WORK2, dnom)
      use chm_kinds
      use number
      implicit none
      real(kind=chm_real),dimension(idgamus) :: q,dq
      real(kind=chm_real) :: en

!      REAL*8 Q(*), DQ(*), EN
!      REAL*8 params%weight(*),params%mean(*),params%sinv(*),params%lndet(*),params%gamma
!     k * temperature in kcal/mol
!      REAL*8 GAMUSKT
!      integer M, ID
!     working arrays of length D
!      REAL*8 WORK1(*), WORK2(*),dnom(*)
!
      integer I, IM, IOFFM, J, IOFFSI
      REAL(kind=chm_real) SUM, DLNGAUSS, DDEN
!
      !REAL(chm_real),parameter :: DLN2PI = 1.837877066409345483560659472811235279722794d0
!
!     Transform Q from [0,1) to [-180,180)
      DO I=1,idgamus !maybe do this in gamusen when branching to the type of reaction coordinate
         Q(I) = THR6TY * Q(I) - ONE8TY
      END DO
!     Initialize numerator 
      DO I=1,idgamus
         dnom(I)=ZERO
      ENDDO
!     Initialize denominator (log-probability) to its offset
      DDEN = params%gamma

!     Loop IM over M, and accumulate terms of the numerator
!     and denominator of the force.

      DO IM=1,mgamus
!     store the q - m vector in work1
!     put work1 into its minimal image
         work1=q(1:idgamus)-params%mean(1:idgamus,im)
         call min_image(params,work1)
!     store y_i = S_i^(-1) . (q-m_i) in work2
         work2=matmul(params%sinv(:,:,im),work1)
 
!     Calculate part the exponent of the Gaussian, (q-m_i) . y_i
         sum=dot_product(work1,work2)
  
!     Add the log of the normalization and rescale to
!     obtain the logarithm of the normalized gaussian.
!     and finally add the log of the weight of the gaussian.
         DLNGAUSS = params%weight(im) - HALF * (SUM + IDGAMUS * DLN2PI + params%lndet(im))

!     accumulate numerator and denominator
!     change DEXP to EXP for CHARMM
         DO I=1,idgamus
            dnom(I) = dnom(I) + EXP(DLNGAUSS)*WORK2(I)
         END DO
         DDEN = SUMLOG(DDEN, DLNGAUSS)
      END DO
!     DQ, returns minus the biasing force, with correct sign for CHARMM
      DO I=1,idgamus
         DQ(I) = GAMUSKT * dnom(I) / EXP(DDEN)
      ENDDO
!     EN returns the biasing potential, the negative of the current FE estimate
      EN = GAMUSKT * DDEN
!     Transform DQ from [-180,180) to [0,1)
      DO I=1,idgamus !see previous comment
         DQ(I) = DQ(I)*THR6TY
      END DO
!      write(*,*) "gamusen1:",(dq(i),i=1,idgamus)

      END subroutine gamusen1
!----------------------------------------------------------

      SUBROUTINE GAMUSBIAS
      use chm_kinds
      use stream
      implicit none

      LOGICAL OK
      REAL(chm_real) EBIAS
      CHARACTER(len=66) STR

#if KEY_ENSEMBLE==0
      if (iolev.gt.0) then  /*justin, for parallel*/
#endif
! read the header
      READ(IGAMUSQ1,'(A66)',END=100,ERR=100) STR
      WRITE(IGAMUSW1,'(A66)') STR
! loop to read the next coordinate, then print biasing potential
 10   CALL GAMUSRDQ(IDGAMUS,gamusq,OK)
      IF(OK)THEN
         call gamusen1(gamusq,gamusdq,ebias)
!         CALL GAMUSEN1(HEAP(GAMUSQ),HEAP(GAMUSDQ),EBIAS,
!     $     params%gamma,HEAP(params%weight),HEAP(params%mean),HEAP(params%sinv),HEAP(params%lndet),
!     $     MGAMUS,IDGAMUS,GAMUSKT,HEAP(GAMUSW1),HEAP(GAMUSW2),HEAP(GAMUSW3))
         WRITE(IGAMUSW1,'(F15.8)') EBIAS
         GOTO 10
      ENDIF
#if KEY_ENSEMBLE==0
      endif  /*iolev*/
#endif
 100  RETURN
!
      END subroutine gamusbias
!
!----------------------------------------------------------
!
      SUBROUTINE GAMUSRDQ(NQ,Q,OK)
      use number
      implicit none

      integer NQ
      LOGICAL OK
      REAL(chm_real) Q(*),E

      integer K

      OK=.TRUE.
      READ(IGAMUSQ1,'(10(F15.8,1X))',END=100,ERR=100) E,(Q(K),K=1,NQ)
!     Transform Q to [0,1)
      DO K=1,NQ
         Q(K) = ( ONE8TY + Q(K) ) / THR6TY
      END DO
      RETURN
 100  OK=.FALSE.
      RETURN
      END subroutine gamusrdq

!----------------------------------------------------------

      SUBROUTINE GAMUSW(Q,DQ)
      use chm_kinds
      use stream
      use energym

      implicit none

      REAL(chm_real) Q(*),X,DQ(*)
      integer K
!justin adds the "iolev.gt.0" for parallel
#if KEY_ENSEMBLE==0
      if (iolev.gt.0) then 
#endif
          WRITE(IGAMUSW,'(10(F15.8,1X))') ETERM(GAMUS),(Q(K),K=1,IDGAMUS)
#if KEY_ENSEMBLE==0
      endif 
#endif
      END subroutine gamusw

!----------------------------------------------------------

!Clear away any existing potential and allocate enough space for nguass gaussians
      subroutine gamusrealloc(ngauss)
            use chm_kinds
            use memory
            use stream
            use param_store, only: set_param
            implicit none

            integer :: ngauss

            if (allocated(params%weight)) call dealloc_gmm_parameter(params)

            mgamus=ngauss
            call set_param("NGAUSS",ngauss)
            call alloc_gmm_parameter(ngauss,idgamus,params)
      end subroutine gamusrealloc
      
      
      
!GAMUS FIT DATA unit NGAUSS int NDIM int NREF int NITR int SIGMA real GAMMA real 
! unit for data, number of refinements, number of iterations, minimum size of a gaussian, gamma cutoff, output unit for new potential
      SUBROUTINE GAMUSFIT(COMLYN,COMLEN)
            use chm_kinds
            use memory
            use string
            use consta
            use number
            use stream
            !use reawri, only: iseed
            use gmmfit, only: gmm_fit
#if KEY_PARALLEL==1
            use parallel 
#endif
#if KEY_ENSEMBLE==1
            use ensemble 
#endif
            use clcg_mod
            use parallel,only:psnd4,psnd8

            implicit none
!
            character(len=*) :: comlyn
            integer :: comlen
!
            integer :: iundata, ndata, ngauss, nrefine, niter, i, j, iostat,seed,ndim
            real(kind=chm_real) :: minsigma, maxsigma,gammacutoff,wt
            !q stores reaction coordinates, 1st index is the coordinate, 2nd the data point
            real(kind=chm_real), allocatable :: q(:,:),weight(:),qq(:)
            !integer :: myseed 
      
            ndim=idgamus
            iundata=gtrmi(comlyn,comlen,'DATA',-1)
            ngauss=gtrmi(comlyn,comlen,'NGAU',-1)
            nrefine=gtrmi(comlyn,comlen,'NREF',20)
            niter=gtrmi(comlyn,comlen,'NITR',200)
            minsigma=gtrmf(comlyn,comlen,'SIGM',five)
            maxsigma=gtrmf(comlyn,comlen,'SMAX',ninety)
            gammacutoff=gtrmf(comlyn,comlen,'GAMM',-thosnd)
            !iunoutput=gtrmi(comlyn,comlen,'OUTP',-1)
      
            !Sanity checks
            if (iundata<=0) call wrndie(-2,'<GAMUSFIT>',&
                'Invalid unit from which to read weights and reaction coordinates')
            if (ngauss<=0) call wrndie(-2,'<GAMUSFIT>','Invalid number of Gaussians')
            if ((nrefine<=0).or.(niter<=0)) call wrndie(-2,'<GAMUSFIT>','Invalid number of refinements or iterations')
            if (minsigma<=zero) call wrndie(-2,'<GAMUSFIT>','Invalid minimum size of Gaussian')
            if ((idgamus>2).and.(gammacutoff<=-thosnd)) then
                call wrndie(0,'<GAMUSFIT>','You may want to set a cutoff for the Bayesian prior (GAMMa cutoff) &
                 to prevent deep artificial minima. See JPC B 113, 4664 (2009)')
            endif
      
            !Read the data.  First pass -- count the number of data points to determine how much space to allocate
            if (iolev>0) then
                ndata=0
                call chmalloc('gamus2.src','gamusfit','qq',idgamus,crl=qq)
                do while (.true.)
                    read(iundata,*,iostat=iostat) wt,(qq(i),i=1,idgamus)
                    if (iostat<0) exit
                    ndata=ndata+1
                end do
                if (ndata<=0) call wrndie(-2,'<GAMUSFIT>','No data in file')
                !Rewind the unit and make a second pass to actually read the data
                 rewind(unit=iundata)
            endif
#if KEY_PARALLEL==1
            call psnd4(ndata,1) 
#endif
            call chmalloc('gamus2.src','gamusfit','q',idgamus,ndata,crl=q)
            call chmalloc('gamus2.src','gamusfit','weight',ndata,crl=weight)
            if (iolev>0) then
                do i=1,ndata
                   read(iundata,*) weight(i),(q(j,i),j=1,idgamus)
                end do
            endif
#if KEY_PARALLEL==1
            call psnd8m(q,idgamus*ndata)
            call psnd8(weight,ndata)
#endif /*   */
            !output first 5 data points
            !do i=1,5
            !    write(outu,*) "fortran: ",w(i), (q((i-1)*idgamus+j),j=1,idgamus)
            !end do
      
            !Now call GMM_FIT
            call gamusrealloc(ngauss)
                   
            if (prnlev>=2) write(outu,*) 'GAMUSFIT> Calling GMM to fit ',ndata,&
              ' data points to ',ngauss,' Gaussians.'
            !call the GMM program
#if KEY_PARALLEL==1
            call psync 
#endif
            call gmm_fit(ngauss, idgamus, nrefine, niter, gammacutoff, minsigma, maxsigma, ndata, q, weight, params)
#if KEY_PARALLEL==1
            call psync 
#endif
            if (params%gamma<gammacutoff) params%gamma=gammacutoff !impose cutoff again, make sure that it is imposed on all processors
      
            call chmdealloc('gamus2.src','gamusfit','q',idgamus,ndata,crl=q)
            call chmdealloc('gamus2.src','gamusfit','weight',ndata,crl=weight)
           
      END SUBROUTINE GAMUSFIT
      
      
      
!GAMUS REWEight NSIM int QUNI int PUNI int NGAUSS int NREF int NITR int SIGMA real GAMMA real [WEIGHT unit]
!Does everything needed for reweighting starting from a set of energy-reaction coordinate files (from dynamics) and a set of GAMUS potentials.
!These must be opened as a sequence of units.
      subroutine gamusreweight(comlyn,comlen)
            use chm_kinds
            use memory
            use string
            use consta
            use number
            use stream
            !use reawri, only: iseed
            use gmmfit, only: gmm_fit
            use mare, only: do_mare
#if KEY_PARALLEL==1
            use parallel 
#endif
#if KEY_ENSEMBLE==1
            use ensemble 
#endif
            use clcg_mod
            implicit none
!        
            character(len=*) :: comlyn
            integer :: comlen
!
            integer :: iunpotstart, iunqstart,ngauss, nrefine, niter, iunweight, iunwork
            integer :: i, j, k, l
            integer :: iostat, nsim, totaldata, maxdata, start,end, junk, ngauss2
            real(kind=chm_real) :: minsigma, maxsigma, gammacutoff,wt,eiqj,en
            !q stores reaction coordinates, 1st index is the coordinate, 2nd the data point
            real(kind=chm_real), allocatable :: q(:,:),weight(:),qq(:),ejqj(:), work(:,:,:), newlnz(:),dq(:)
            integer, allocatable :: ndata(:),offset(:),nwork(:,:)
            !real(chm_real) d
            CHARACTER(len=66) STR
            logical overlap
      
            !Parse command line
            nsim=gtrmi(comlyn,comlen,'NSIM',-1)
            iunpotstart=gtrmi(comlyn,comlen,'PUNI',-1)
            iunqstart=gtrmi(comlyn,comlen,'QUNI',-1)
            ngauss=gtrmi(comlyn,comlen,'NGAU',-1)
            nrefine=gtrmi(comlyn,comlen,'NREF',20)
            niter=gtrmi(comlyn,comlen,'NITR',200)
            minsigma=gtrmf(comlyn,comlen,'SIGM',five)
            maxsigma=gtrmf(comlyn,comlen,'SMAX',ninety)
            gammacutoff=gtrmf(comlyn,comlen,'GAMM',-thosnd)
            iunwork=gtrmi(comlyn,comlen,'WORK',-1)
            iunweight=gtrmi(comlyn,comlen,'WEIG',-1)
      
            !Sanity checks
            if ((nsim<=0).or.(iunpotstart<=0).or.(iunqstart<=0)) call wrndie(-2, '<GAMUSREWEIGHT>',&
               'Invalid options for GAMUS data to be processed')
            overlap=(iunqstart>=iunpotstart).and.(iunqstart<=(iunpotstart+nsim-1))
            overlap=overlap.or.((iunpotstart>=iunqstart).and.(iunpotstart<=(iunqstart+nsim-1)))
            if (overlap) call wrndie(-2,'<GAMUSREWEIGHT>',&
               'Unit sequences for GAMUS potentials and reaction coordinates overlap')
            if (ngauss<=0) call wrndie(-2,'<GAMUSREWEIGHT>','Invalid number of Gaussians')
            if ((nrefine<=0).or.(niter<=0)) &
               call wrndie(-2,'<GAMUSREWEIGHT>','Invalid number of refinements or iterations')
            if (minsigma<=zero) call wrndie(-2,'<GAMUSREWEIGHT>','Invalid minimum size of Gaussian')
            if ((idgamus>2).and.(gammacutoff<=-thosnd))&
                call wrndie(0,'<GAMUSREWEIGHT>','You may want to set a cutoff for the Bayesian prior (GAMMa cutoff) &
                 to prevent deep artificial minima. See JPC B 113, 4664 (2009)')
        
      
            !Allocate arrays based on how many simulations
            call chmalloc('gamus2.src','gamusreweight','ndata',nsim,intg=ndata)
            call chmalloc('gamus2.src','gamusreweight','offset',nsim,intg=offset)
            call chmalloc('gamus2.src','gamusreweight','newlnz',nsim,crl=newlnz)
            call chmalloc('gamus2.src','gamusreweight','nwork',nsim,nsim,intg=nwork)
            call chmalloc('gamus2.src','gamusreweight','qq',idgamus,crl=qq)
            call chmalloc('gamus2.src','gamusreweight','dq',idgamus,crl=dq)
      
            !Open each e-q file and count the number of data points
            if (iolev>0) then
              do i=1,nsim
                igamusq1=iunqstart + i - 1
                read(igamusq1,*) str
                ndata(i)=0
                do while (.true.)
                   read(igamusq1,*,iostat=iostat) en,(qq(k),k=1,idgamus)
                   if (iostat<0) exit
                   ndata(i)=ndata(i)+1
                end do
                rewind(unit=igamusq1)
                !write(outu,*) "Reading ",ndata(i)," data points from unit ",igamusq1
              end do
            endif
#if KEY_PARALLEL==1
            call psnd4(ndata,nsim) 
#endif
            !write(outu,*) "node ",mynod," ndata ",ndata
      
            !Compute offsets into the w() and q() arrays for the start of each simulation  
            offset(1)=0
            do i=2,nsim
                offset(i)=offset(i-1)+ndata(i-1)
            end do
            totaldata=offset(nsim)+ndata(nsim)
            maxdata=maxval(ndata)
            if (prnlev>=2) write(outu,*) "GAMUSREWEIGHT> ",totaldata," data points in total will be read."
      
            !Now that we know how much data we have, it's time to allocate more memory
            call chmalloc('gamus2.src','gamusreweight','q',idgamus,totaldata,crl=q)
            call chmalloc('gamus2.src','gamusreweight','weight',totaldata,crl=weight)
            call chmalloc('gamus2.src','gamusreweight','ejqj',totaldata,crl=ejqj)
            call chmalloc('gamus2.src','gamusreweight','work',maxdata,nsim,nsim,crl=work)
      
            do i=1,nsim
               newlnz(i)=zero !temporary, not the behavior of the scripts for constructing initial guess for ln(z)
               !this seems to produce the same results though
            end do
      
            if (iolev>0) then 
              do i=1,nsim
                igamusq1=iunqstart + i - 1
                read(igamusq1,*) str
                if (prnlev>=2) write(outu,*) "GAMUSREWEIGHT> Reading ",ndata(i)," data points from unit ",igamusq1
                do j=1,ndata(i)
                   read(igamusq1,*) ejqj(offset(i)+j),(q(k,offset(i)+j),k=1,idgamus)
                end do   
              end do
            endif
#if KEY_PARALLEL==1
            call psnd8(ejqj,totaldata)
            call psnd8m(q,idgamus*totaldata)
#endif 
            !write(outu,*) "node ",mynod," weights1 ",(q(1,i),i=1,5)
            !test
            !do i=1,totaldata
                !start=(i-1)*idgamus
                !write(7,'(4F12.6)') (q(start+j), j=1,idgamus)
            !end do
      
            if ((iolev>0).and.(nsim>1)) then !only one processor needs to do MARE
              nwork = zero
              work = zero
              !compute biased energies and work values
              do i=1,nsim
                 !read potential "i"
                 igamusu1=iunpotstart + i - 1
                   read(igamusu1,*) ngauss2,junk
                call gamusrealloc(ngauss2)
                call read_gmm_parameter(params,igamusu1)
                !W_ji,k = ( E_i(q_j,k) - E_j(q_j,k))
                do j=1,nsim
                   if (i/=j) then
                      nwork(j,i) = ndata(j)
                      do k=1,ndata(j)
                         do l=1,idgamus !transform to [0,1]
                            qq(l)=(one8ty+q(l,offset(j)+k))/thr6ty
                         end do
                         call gamusen1(qq,dq,eiqj) !E_i(q_j,k)
                         en=ejqj(offset(j)+k) !E_j(q_j,k)
                         work(k,j,i) = (eiqj - en) / gamuskt
                      end do
                   endif
                end do
              end do

              if (iunwork>0) then
                write(iunwork,*) nsim
                do i=1,nsim
                   do j=1,nsim
                      if (i/=j) then
                         write(iunwork,*) j-1,i-1,0,nwork(j,i)
                         do k=1,nwork(j,i)
                            write(iunwork,*) work(k,j,i)
                         end do
                      endif
                   end do
                end do
              end if
      
            !Now we're ready to do MARE -- only one processor, to get only one set of output.
      
               call do_mare(nsim, maxdata, nwork, work, newlnz)
            endif  !iolev>0 .and. nsim>1
#if KEY_PARALLEL==1
            !write(outu,*) "Node ",mynod," marker 1"
            call psync 
            call psnd8(newlnz,nsim)
            !write(outu,*) "Node ",mynod," marker 1b"
#endif 
            !if (prnlev>0) write(outu,*) "GAMUSREWEIGHT> ln(Z_i/Z_0): ",(newlnz(i),i=1,nsim)
            if (prnlev>=2) then
               write(outu,'(" Simulation  ln(Z_i/Z_0)")')
               write(outu,'(" ----------  -----------")')
               do i=1,nsim
                  write(outu,'(2X,I9,2X,F11.4)') i,newlnz(i)
               end do
            endif
            !write(outu,*) "node ",mynod," weights2 ",(q(1,i),i=1,5)
            !Now compute the weights: ln w_i = ln(Z_i/Z_0) + E_j(q_j)/kT
            do i=1,nsim
               do k=1,ndata(i)
                  weight(offset(i)+k)=newlnz(i)+ejqj(offset(i)+k)/gamuskt
               end do
            end do
            
            !write out weights and data -- need to work on format
            if ((iolev>0).and.(iunweight>0)) then
              do i=1,totaldata
                write(iunweight,'(10(F15.8,1X))') weight(i),(q(j,i), j=1,idgamus)
              end do
            endif
            
            !Now call GMM_FIT
            call gamusrealloc(ngauss)
      
            if (prnlev>=2) write(outu,*) 'GAMUSREWEIGHT> Fitting ',totaldata,&
              ' data points to ',ngauss,' Gaussians.'
            !call the GMM program
#if KEY_PARALLEL==1
            call psync 
#endif
            !write(outu,*) "node ",mynod," weights3 ",(q(1,i),i=1,5)
            !write(outu,*) "node ",mynod," totaldata ",totaldata,offset
            call gmm_fit(ngauss, idgamus, nrefine, niter, gammacutoff, minsigma, maxsigma, totaldata, q, weight, params)
#if KEY_PARALLEL==1
            call psync 
#endif
            if (params%gamma<gammacutoff) params%gamma=gammacutoff !impose cutoff again, make sure that it is imposed on all processors
            !if (params%ndim==2) then
            !   do i=1,params%ngauss
            !      d=params%sinv(1,1,i)*params%sinv(2,2,i)-params%sinv(1,2,i)*params%sinv(2,1,i)
            !      if (d<=zero) then
            !         write(outu,*) "matrix not positive definite ",i
            !         write(outu,*) "r(1,1), r(1,2), r(2,2) = ",params%r(1,1,i),params%r(1,2,i),params%r(2,2,i)
            !         write(outu,*) "s(1,1), s(1,2), s(2,1), s(2,2) = ",params%sinv(1,1,i),params%sinv(1,2,i),&
            !            params%sinv(2,1,i),params%sinv(2,2,i)
            !         call wrndie(-10,'<gamusreweight>','problem with matrix')
            !      endif
            !   end do
            !endif
            call chmdealloc('gamus2.src','gamusreweight','q',idgamus,totaldata,crl=q)
            call chmdealloc('gamus2.src','gamusreweight','weight',totaldata,crl=weight)
            call chmdealloc('gamus2.src','gamusreweight','ejqj',totaldata,crl=ejqj)
            call chmdealloc('gamus2.src','gamusreweight','work',maxdata,nsim,nsim,crl=work)
            call chmdealloc('gamus2.src','gamusreweight','ndata',nsim,intg=ndata)
            call chmdealloc('gamus2.src','gamusreweight','offset',nsim,intg=offset)
            call chmdealloc('gamus2.src','gamusreweight','newlnz',nsim,crl=newlnz)
            call chmdealloc('gamus2.src','gamusreweight','nwork',nsim,nsim,intg=nwork)
            call chmdealloc('gamus2.src','gamusreweight','qq',idgamus,crl=qq)
            call chmdealloc('gamus2.src','gamusreweight','dq',idgamus,crl=dq)
      
      
      end subroutine gamusreweight

     !GAMUS INFO -- print weights, means, and standard deviations/corresponding eigenvectors of variance-covariance matrices
     subroutine gamusinfo(comlyn,comlen)
      use memory
      use string
      use stream
      implicit none
      character(len=*) :: comlyn
      integer :: comlen
     integer i,j,k,lwork,info
     logical verbose
     real(chm_real),allocatable :: si(:,:),work(:),ev(:)
     verbose=(indxa(comlyn,comlen,'VERB')>0)
     lwork=3*idgamus-1
     call chmalloc('gamus.src','gamusinfo','si',idgamus,idgamus,crl=si)
     call chmalloc('gamus.src','gamusinfo','work',lwork,crl=work)
     call chmalloc('gamus.src','gamusinfo','ev',idgamus,crl=ev)
     if (prnlev>=2) then
        do i=1,params%ngauss
           write(outu,'("Gaussian number ",I4," has weight ",E12.4," and mean ",10F12.4)') &
              i,exp(params%weight(i)),(params%mean(j,i),j=1,idgamus)
           if (verbose) then
              si(:,:)=params%sinv(:,:,i)
              call DSYEV('V','U',idgamus,si,idgamus,ev,work,lwork,info)
              if (info<0) then
                  write(outu,*) "Lapack error: ",info
                  call wrndie(-2,'<GAMUSINFO>','Lapack error')
              end if
              do j=1,idgamus
                  write(outu,'("Dimension ",I4," width ",F12.4," degrees, eigenvector ",10F12.4)') &
                     j,1/sqrt(ev(j)),(si(k,j),k=1,idgamus)
              end do
           end if
        end do
     end if
     call chmdealloc('gamus.src','gamusinfo','si',idgamus,idgamus,crl=si)
     call chmdealloc('gamus.src','gamusinfo','work',lwork,crl=work)
     call chmdealloc('gamus.src','gamusinfo','ev',idgamus,crl=ev)

     end subroutine gamusinfo

#else /* (gamus)*/

     SUBROUTINE GAMUSINIT(comlyn,comlen) 
      implicit none
      character(len=*) :: comlyn
      integer :: comlen
     call wrndie(-5,'<GAMUSINI1>','GAMUS code not compiled.')
     end subroutine gamusinit

#endif /* (gamus)*/

end module gamusmodule     

