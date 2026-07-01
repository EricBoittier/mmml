#if KEY_RDC==1
SUBROUTINE RDCSET1
!------------------------------------------------------------------------
!     RDC Restrain Potential (2006)
!
!     Author: Thenmalarchelvi Rathinavelan (thenmcr@ku.edu) and
!             Wonpil Im (wonpil@ku.edu)
!
!     @ Center for Bioinformatics in Univ. of Kansas
!
!     Reference: 1. ???
!                2. ???

   use dimens_fcm
   use psf
   use consta
   use comand
   use stream
   use string
   use select
   use coord
   use number
   use chutil,only:atomid
   use rdc
   use exfunc
   use memory,only:chmalloc,chmdealloc
   use chm_types
   implicit none

! local
   integer,allocatable,dimension(:) :: islct, jslct
!
      CHARACTER*4 WD
      CHARACTER*8 IATM, JATM
      INTEGER     URDC
      INTEGER     NRDC,nrdcs,nrdcmp,nrdcmps,nrdce,nrdcmpe
      INTEGER     EXPO0,EXPO1
      INTEGER     ITERA,K

      real(chm_real)      KHAR0,KASY0,LHAR0
      real(chm_real)      GYRO

      LOGICAL     EOF,DONE,OLD
      LOGICAL     QBMRB,QXPLOR
!
!
      ITERA=0
!
!
      IF(INDXA(COMLYN,COMLEN,'RESE').GT.0) THEN
         IF(.not.QRDC) THEN
            CALL WRNDIE(1,'<RDC>','Nothing is setup !')
            RETURN
         ENDIF

! memory cleanup
         !call chmdealloc('rdc.src','rdc','islct',natom,intg=islct)
         !call chmdealloc('rdc.src','rdc','jslct',natom,intg=jslct)

         call rdc_uninit(nset, nrdcmp)

         RETURN
      ENDIF

 10      FORMAT(6x,a,i4)
      IF(INDXA(COMLYN,COMLEN,'ANAL').GT.0) THEN
         QKEY=.TRUE.
         URDC=GTRMI(COMLYN,COMLEN,'URDC',OUTU)
            IF(.not.QRDC) THEN
               CALL WRNDIE(1,'<RDC>','Nothing is setup !')
               RETURN
            ENDIF
         DO K=1,NSET
            WRITE(URDC,10) 'MEDIUM ',K
            CALL RDC_MALLOC(K,NRDC,NRDCMP,NRDC1,NRDCMP1, &
                 QKEY,NRDCS,NrdcmpS,NRDCE,NrdcmpE)
            NRDC=NRDCE-NRDCS+1
            NRDCmp=NRDCmpE-NRDCmpS+1
! Molecular Frame
            CALL RDC_MF(NATOM,X,Y,Z,AMASS,MF,MFT,EV, &
                 NRDCS,NRDCE,RDCaLIS,RDCbLIS,QRMF)
! R matrix
            CALL RDC_RMAT(NATOM,X,Y,Z,NRDCS,NRDCE,REXP, &
                 RDCaLIS,RDCbLIS,IGAMMA,JGAMMA, &
                 MF,MFT,RDCDIST,RDCDISTmp, &
                 DCON,DRED,RDCVECX,RDCVECY, &
                 RDCVECZ,RDCVECmpX,RDCVECmpY, &
                 RDCVECmpZ,RDCRrMAT,RDCRMAT, &
                 RDCRMATmp,CT,SMAT,MPRDC)
! Alignment Tensor
         IF (.not.QFIXA) CALL RDC_AT(NRDC,DRED,SMAT, &
                 WI,W55,V,VWI,UW, &
                 WIUT,AN)
! Analysis
            CALL RDC_ANAL(NRDCS,NRDCE,RDCaLIS,RDCbLIS, &
                 DCON,REXP,CT,AN,URDC,K)
            if(K.eq.NSET) RETURN
            write(*,*) K
         ENDDO
         write(*,*) "xhere"
      ENDIF
      QKEY=.FALSE.

      IF(INDXA(COMLYN,COMLEN,'BCAL').GT.0) THEN
         QKEY=.TRUE.
         URDC=GTRMI(COMLYN,COMLEN,'URDC',OUTU)
         IATM=GTRMA(COMLYN,COMLEN,'IATM')
         JATM=GTRMA(COMLYN,COMLEN,'JATM')
         RRES=GTRMI(COMLYN,COMLEN,'RRES',0) ! RDC residues

         IGAMMA1=GYRO(IATM)
         JGAMMA1=GYRO(JATM)

         IF(.not.QRDC) THEN
            CALL WRNDIE(1,'<RDC>','Nothing is setup !')
            RETURN
         ENDIF
         DO K=1,NSET
            WRITE(URDC,10) 'MEDIUM ',K
            CALL RDC_MALLOC(K,NRDC,NRDCMP,NRDC1,NRDCMP1, &
                 QKEY,NRDCS,NrdcmpS,NRDCE,NrdcmpE)
            NRDC=NRDCE-NRDCS+1
            NRDCmp=NRDCmpE-NRDCmpS+1
! Molecular Frame
            CALL RDC_MF(NATOM,X,Y,Z,AMASS,MF,MFT,EV, &
                 NRDCS,NRDCE,RDCaLIS,RDCbLIS,QRMF)
! R matrix
            CALL RDC_RMAT(NATOM,X,Y,Z,NRDCS,NRDCE,REXP, &
                 RDCaLIS,RDCbLIS,IGAMMA,JGAMMA, &
                 MF,MFT,RDCDIST,RDCDISTmp, &
                 DCON,DRED,RDCVECX,RDCVECY, &
                 RDCVECZ,RDCVECmpX,RDCVECmpY, &
                 RDCVECmpZ,RDCRrMAT,RDCRMAT, &
                 RDCRMATmp,CT,SMAT,MPRDC)
! Alignment Tensor
         IF (.not.QFIXA) CALL RDC_AT(NRDC,DRED,SMAT, &
                 WI,W55,V,VWI,UW, &
                 WIUT,AN)
! Back calculation
         CALL RDC_BCAL(NATOM,X,Y,Z,IATM,JATM,AN,MF, &
              MFT,IGAMMA1,JGAMMA1,URDC,NRES,RRES,ibase,atype)
            IF(K.eq.NSET) RETURN
         ENDDO
      ENDIF
      QKEY=.FALSE.
!
!
      IF(.not.QRDC) THEN
         MAXRDC=GTRMI(COMLYN,COMLEN,'MAXR', default_maxrdc)  ! maximum number of RDCs
         NSET  =GTRMI(COMLYN,COMLEN,'NSET',1)    ! Number of RDC SETs
         LHAR0 =GTRMF(COMLYN,COMLEN,'LHAR',ONE)   ! length of harmonic function
         KHAR0 =GTRMF(COMLYN,COMLEN,'KHAR',ONE)   ! harmonic force constant
         KASY0 =GTRMF(COMLYN,COMLEN,'KASY',ONE)   ! asymtotic force constant
         EXPO0 =GTRMI(COMLYN,COMLEN,'EXPO',1)     ! exponential func. for asymtotic function
         URDC  =GTRMI(COMLYN,COMLEN,'URDC',-1) ! reading unit for RDC assignments
         QBMRB =INDXA(COMLYN,COMLEN,'BMRB').GT.0  ! BMRB format?
         QXPLOR=INDXA(COMLYN,COMLEN,'XPLO').GT.0  ! XPLOR format?
         QRMF  =INDXA(COMLYN,COMLEN,'QRMF').GT.0  ! use only RDC atoms for molecular fram?
         QFIXA =INDXA(COMLYN,COMLEN,'FIXA').GT.0  ! fix alignment tensor?
         QFIXB =INDXA(COMLYN,COMLEN,'FIXB').GT.0  ! fix all RDC bonds?
         QSLOW =INDXA(COMLYN,COMLEN,'SLOW').GT.0  ! RDC Full Force Calculation
         QSRDC =INDXA(COMLYN,COMLEN,'SRDC').GT.0  ! SCALING RDCs w.r.t N-H RDC

         call chmalloc('rdc.src','rdc','islct',natom,intg=islct)
         call chmalloc('rdc.src','rdc','jslct',natom,intg=jslct)

         nrdc = 0
         nrdcmp = 0
         call rdc_init() ! alloc storage depending on nset and maxrdc
      ENDIF
!
!
!
      DO K=1,NSET
         DONE=.FALSE.
         EOF=.FALSE.
         IF(URDC.GT.1) THEN
! BMRB format
            IF(QBMRB) THEN
               DO WHILE(.NOT.DONE)
                  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,URDC,EOF,.TRUE.,.FALSE.,'RDC> ')
                  IF(EOF) GOTO 100
                  WD=NEXTA4(COMLYN,COMLEN)
                  if (WD.EQ.'_RDC') DONE=.TRUE.
               enddo
               READ(URDC,11)
 11            format(/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/,/)
               DONE=.FALSE.
               DO WHILE(.NOT.DONE)
                  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,URDC,EOF,.TRUE.,.FALSE.,'RDC> ')
                  IF(EOF) GOTO 100
                  IF(INDX(COMLYN,COMLEN,'STOP',4).GT.0)GOTO 100

                  nrdc = nrdc + 1
                  if (nrdc > maxrdc) call rdc_add_storage()

                  CALL RDC_BMRB(COMLYN,COMLEN,NRDC,NSET,Nrdcmp, &
                       RDCaLIS,RDCbLIS,REXP, &
                       DELL,DELU,ITERA,MPRDC)
                  CALL RDCREADER(ISLCT,JSLCT,NATOM, &
                       NRDC,NSET,Nrdcmp,KHAR0,KASY0,LHAR0,EXPO0, &
                       RDCaLIS,RDCbLIS,REXP, &
                       DELL,DELU,KHAR,LHAR, &
                       KASY,EXPO,IGAMMA,JGAMMA, &
                       SRDC,QBMRB,QXPLOR,QSRDC,MPRDC)
               ENDDO
! XPLOR format
            ELSEIF(QXPLOR) THEN
               DO WHILE(.NOT.DONE)
                  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,URDC,EOF,.TRUE.,.FALSE.,'RDC> ')
                  IF(EOF) GOTO 100
                  WD=NEXTA4(COMLYN,COMLEN)
                  if (WD.EQ.'ASSI') then
                     nrdc = nrdc + 1
                     if (nrdc > maxrdc) call rdc_add_storage()

                     CALL RDC_XPLOR(URDC,NRDC,NSET,Nrdcmp, &
                          RDCaLIS,RDCbLIS,REXP, &
                          DELL,DELU,ITERA,MPRDC)
                     CALL RDCREADER(ISLCT,JSLCT,NATOM, &
                          NRDC,NSET,Nrdcmp,KHAR0,KASY0,LHAR0,EXPO0, &
                          RDCaLIS,RDCbLIS,REXP, &
                          DELL,DELU,KHAR,LHAR, &
                          KASY,EXPO,IGAMMA, &
                          JGAMMA,SRDC,QBMRB,QXPLOR,QSRDC, &
                          MPRDC)
                  endif
               enddo
! CHARMM format
            ELSE
               DO WHILE(.NOT.DONE)
                  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,URDC,EOF,.TRUE.,.FALSE.,'RDC> ')
                  IF(EOF) GOTO 100
                  WD=NEXTA4(COMLYN,COMLEN)
                  IF (WD.eq. 'ASSI') THEN
                     nrdc = nrdc + 1
                     if (nrdc > maxrdc) call rdc_add_storage()

                     CALL RDCREADER(ISLCT,JSLCT,NATOM, &
                          NRDC,NSET,Nrdcmp,KHAR0,KASY0,LHAR0,EXPO0, &
                          RDCaLIS,RDCbLIS,REXP, &
                          DELL,DELU,KHAR,LHAR, &
                          KASY,EXPO,IGAMMA, &
                          JGAMMA,SRDC,QBMRB,QXPLOR,QSRDC, &
                          MPRDC)
                  endif
               enddo
            ENDIF
 100        WD=NEXTA4(COMLYN,COMLEN)
            CALL RDC_MALLOC(K,nrdc,nrdcmp,nrdc1,nrdcmp1, &
                 QKEY,NRDCS,NrdcmpS,NRDCE,NrdcmpE)
!
! OLD main drive
!
         ELSE
            DO WHILE(.NOT.DONE)
               CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'RDC> ')
               WD=NEXTA4(COMLYN,COMLEN)
               IF(WD.EQ.' ') THEN
                  CONTINUE
               ELSE IF(WD.EQ.'RESE') THEN ! to reset
                  NRDC=0
               ELSEIF (WD.eq.'NSET') THEN
                  NSET=NEXTI(COMLYN,COMLEN)
                  IF(NSET.gt.1)CALL RDC_MALLOC(K,nrdc,nrdcmp, &
                       nrdc1,nrdcmp1,QKEY,NRDCS,NrdcmpS, &
                       NRDCE,NrdcmpE)
               ELSE IF(WD.EQ.'ASSI') THEN
                  NRDC=NRDC+1
                  IF(NRDC.GE.MAXRDC) &
                       CALL WRNDIE(-5,'<RDC>','Max number of RDC &
                       restraints exceeded')
                  CALL RDCREADER(ISLCT,JSLCT,NATOM, &
                       NRDC,NSET,Nrdcmp,KHAR0,KASY0,LHAR0,EXPO0, &
                       RDCaLIS,RDCbLIS,REXP, &
                       DELL,DELU,KHAR,LHAR, &
                       KASY,EXPO,IGAMMA,JGAMMA, &
                       SRDC,QBMRB,QXPLOR,QSRDC,MPRDC)
               ELSE IF(WD.EQ.'END ') THEN
                  CALL RDC_MALLOC(K,nrdc,nrdcmp,nrdc1, &
                       nrdcmp1,QKEY,NRDCS,NrdcmpS,NRDCE,NrdcmpE)
                  GOTO 220
               ELSE
                  CALL WRNDIE(-5,'<RDC>','UNKNOWN OPTION')
               ENDIF
            ENDDO
         ENDIF
         URDC=URDC+1
      ENDDO
 220  CONTINUE

!
      call chmdealloc('rdc.src','rdc','islct',natom,intg=islct)
      call chmdealloc('rdc.src','rdc','jslct',natom,intg=jslct)

!
!
! Memory Allocate for Energy and Force Calculations
!
      IF(.not.QRDC) THEN
! Molecular frame
         CALL chmalloc('rdc.src','RDC','MF' ,3*3, crl=MF)
         CALL chmalloc('rdc.src','RDC','MFT',3*3, crl=MFT)
         CALL chmalloc('rdc.src','RDC','EV' ,3  , crl=EV)

! R matrix
         CALL chmalloc('rdc.src','RDC','RDCVECX', NRDC,crl=RDCVECX )
         CALL chmalloc('rdc.src','RDC','RDCVECY', NRDC,crl=RDCVECY )
         CALL chmalloc('rdc.src','RDC','RDCVECZ', NRDC,crl=RDCVECZ )
         CALL chmalloc('rdc.src','RDC','RDCDIST', NRDC,crl=RDCDIST )
         CALL chmalloc('rdc.src','RDC','RDCRMAT', 6*NRDC,crl=RDCRMAT )
         CALL chmalloc('rdc.src','RDC','RDCRrMAT',6*NRDC,crl=RDCRrMAT)
         CALL chmalloc('rdc.src','RDC','SMAT',    5*NRDC,crl=SMAT    )
         CALL chmalloc('rdc.src','RDC','CT',      6*NRDC,crl=CT      )
         CALL chmalloc('rdc.src','RDC','DRED',    NRDC,crl=DRED    )
         CALL chmalloc('rdc.src','RDC','DCON',    NRDC,crl=DCON    )

         IF(NRDCmp.gt.0) THEN
            CALL chmalloc('rdc.src','RDC','RDCRMATmp', 6*NRDCmp,crl=RDCRMATmp)
            CALL chmalloc('rdc.src','RDC','RDCVECmpX', NRDCmp,crl=RDCVECmpX)
            CALL chmalloc('rdc.src','RDC','RDCVECmpY', NRDCmp,crl=RDCVECmpY)
            CALL chmalloc('rdc.src','RDC','RDCVECmpZ', NRDCmp,crl=RDCVECmpZ)
            CALL chmalloc('rdc.src','RDC','RDCDISTmp', NRDCmp,crl=RDCDISTmp)
         ENDIF

! Alignment Tensor
         CALL chmalloc('rdc.src','RDC','AN',  3*3,crl=AN  )
         CALL chmalloc('rdc.src','RDC','WI',  5*5,crl=WI  )
         CALL chmalloc('rdc.src','RDC','W55', 5*5,crl=W55 )
         CALL chmalloc('rdc.src','RDC','V',   5*5,crl=V   )
         CALL chmalloc('rdc.src','RDC','VWI', 5*5,crl=VWI )
         CALL chmalloc('rdc.src','RDC','UW',  5*NRDC,crl=UW  )
         CALL chmalloc('rdc.src','RDC','WIUT',5*NRDC,crl=WIUT)
!
      ENDIF
      QRDC=.TRUE.
      RETURN
    END SUBROUTINE RDCSET1

SUBROUTINE RDCREADER(ISLCT,JSLCT,NATOM,NRDC,NSET,Nrdcmp,KHAR0, &
                 KASY0,LHAR0,EXPO0,RDCaLIS,RDCbLIS,REXP,DELL,DELU,KHAR, &
                 LHAR,KASY,EXPO,IGAMMA,JGAMMA,SRDC,QBMRB,QXPLOR,QSRDC, &
                 MPRDC)
!------------------------------------------------------------------------
!     THIS ROUTINE READS AN RDC FROM CHARMM FILE & ASSIGNS THE FORCE
!     CONST. ETC.

   use dimens_fcm
   use exfunc
   use number
   use comand
   use stream
   use string
   use coord
   use select
   use chutil,only:atomid
   use chm_types
   implicit none

      INTEGER     NSET,NATOM,NRDC,Nrdcmp
      INTEGER     ISLCT(*),JSLCT(*)
      INTEGER     rdcalis(*),rdcblis(*),EXPO(*)
      INTEGER     EXPO0
      real(chm_real)      KHAR0,KASY0,LHAR0
      real(chm_real)      KHAR(*),DELL(*),DELU(*)
      real(chm_real)      REXP(*),LHAR(*),KASY(*)
      real(chm_real)      IGAMMA(*),JGAMMA(*),SRDC(*)
! local
      INTEGER     I,J,IA
      real(chm_real)      GYRO,NG1,HG1
      CHARACTER*8 SIDI,RIDI,RENI,ACI,NG
      CHARACTER*8 SIDJ,RIDJ,RENJ,ACJ,HG
      LOGICAL     QERR,QBMRB,QXPLOR,QSRDC
      INTEGER     IITERA,JITERA,MPRDC(*)
      LOGICAL     IMULT,JMULT

!
      IA=(NSET-1)+NRDC
      IMULT=.FALSE.
      JMULT=.FALSE.
      IITERA=0
      JITERA=0

      IF ((.not.QBMRB) .and. (.not.QXPLOR) ) then
         prnlev = 1
         CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,QERR)
         prnlev = 5
         IF(QERR) CALL WRNDIE(-5,'<RDC>','SELECTION ERROR')

         DO I=1,NATOM
            IF(ISLCT(I).EQ.1) then
               IITERA=IITERA+1
               if(iitera.eq.1) RDCaLIS(NRDC)=I
               if(iitera.eq.2) IMULT=.TRUE.
            endif
         ENDDO

         DO J=1,NATOM
            IF(JSLCT(J).EQ.1) then
               JITERA=JITERA+1
               if(jitera.eq.1) RDCbLIS(NRDC)=J
               if(jitera.eq.2) JMULT=.TRUE.
            endif
         ENDDO

         IF(RDCaLIS(NRDC).EQ.0 .OR. RDCbLIS(NRDC).EQ.0) THEN
            CALL WRNDIE(-5,'<RDC>','Zero atom selected for this restraint.')
         ENDIF
         REXP(NRDC)=GTRMF(COMLYN,COMLEN,'REXP',ZERO) !experimental value
         DELL(NRDC)=GTRMF(COMLYN,COMLEN,'DELL',ZERO) !experimental value lower bound
         DELU(NRDC)=GTRMF(COMLYN,COMLEN,'DELU',ZERO) !experimental value upper bound

         if (DELL(NRDC).eq.0) DELL(NRDC)=REXP(NRDC) !-0.2
         if (DELU(NRDC).eq.0) DELU(NRDC)=REXP(NRDC) !+0.2

      endif

      IF(REXP(NRDC).EQ.ZERO) THEN
         CALL WRNDIE(1,'<RDC>','Warning: RDC value is zero')
      ENDIF


      KHAR(NRDC)=GTRMF(COMLYN,COMLEN,'KHAR',KHAR0) !force const. harmonic
      LHAR(NRDC)=GTRMF(COMLYN,COMLEN,'LHAR',LHAR0) !length of harmonic function
      KASY(NRDC)=GTRMF(COMLYN,COMLEN,'KASY',KASY0) !force const. for switching function
      EXPO(NRDC)=GTRMI(COMLYN,COMLEN,'EXPO',EXPO0) !exponential func. for switching

      I=RDCaLIS(NRDC)
      CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
      IGAMMA(NRDC)=GYRO(ACI)
      J=RDCbLIS(NRDC)
      CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
      JGAMMA(NRDC)=GYRO(ACJ)

      IF(QSRDC)THEN
         NG='N'
         HG='H'
         NG1=GYRO(NG)
         HG1=GYRO(HG)
         SRDC(NRDC)=(NG1*HG1)/(IGAMMA(NRDC)*JGAMMA(NRDC))
      endif

      call RDCWRITER(NRDC,RDCaLIS,RDCbLIS,REXP,DELL,DELU,KHAR,LHAR,KASY,EXPO)

      if(IMULT) then
         MPRDC(NRDC)=iitera
         Nrdcmp=Nrdcmp+1
      elseif(JMULT) then
         MPRDC(nrdc)=jitera
         Nrdcmp=Nrdcmp+1
      elseif((.not.IMULT.and..not.JMULT).and.(.not.QBMRB) .and.(.not.QXPLOR) ) then
         MPRDC(nrdc)=iitera  !or jitera
      endif
!      print*,'itera',nrdc, mprdc(nrdc),rdcalis(nrdc),rdcblis(nrdc)

      RETURN
      END

SUBROUTINE RDC_BMRB(COMLYN,COMLEN,NRDC,NSET,Nrdcmp,RDCaLIS, &
                    RDCbLIS,REXP,DELL,DELU,ITERA,MPRDC)
!------------------------------------------------------------------------
!     THIS ROUTINE READS AN RDC FROM BMRB FILE
!

   use dimens_fcm
   use exfunc
   use psf
   use number
   use chm_types
   use string
   use chutil
   implicit none

      INTEGER     NSET,NRDC,Nrdcmp,COMLEN
      INTEGER     rdcalis(*),rdcblis(*)
      real(chm_real)      REXP(*),DELL(*),DELU(*)
      CHARACTER*(*) COMLYN
! local
      INTEGER     I,IA
      CHARACTER*8 SIDI,RIDI,RENI,ACI
      character*6 iatm, jatm, jtmp, segi,segj
      integer     ires, jres
      integer     resn
      character*4 wd
      integer     segilen,segjlen,ilen,jlen,jtemlen
      INTEGER     ITERA,MPRDC(*)


      ITERA=0
      IA=(NSET-1)+NRDC

      DO I=1,12
         WD=NEXTA4(COMLYN,COMLEN)
      enddo

      WD=NEXTA4(COMLYN,COMLEN)
!      print*,WD !, COMLYN
      if (WD .ne. '.  ') segi = WD
!      print*,comlyn,wd
      IRES=NEXTI(COMLYN,COMLEN)
!      print*,'ires',ires,comlyn
      WD=NEXTA6(COMLYN,COMLEN)
!      print*,comlyn
      IATM=NEXTA6(COMLYN,COMLEN)
!      print*,comlyn
      WD=NEXTA6(COMLYN,COMLEN)
!      print*,comlyn
      if (WD .ne. '.  ') segj = WD
      JRES=NEXTI(COMLYN,COMLEN)
!      print*,segi,segj

      segilen=strlng(segi)
      segjlen=strlng(segj)
!      print*, segi, segj,'seg len', segilen, segjlen


!      print*,'jres',jres,comlyn
      WD=NEXTA6(COMLYN,COMLEN)
      JATM=NEXTA6(COMLYN,COMLEN)
!      print*,'jatm',jatm
!      print*,nset,NRDC
      REXP(NRDC)=NEXTF(COMLYN,COMLEN)
!      print*,'exp...',rexp(NRDC)
!      if (QDEL) then
      DELL(NRDC)=NEXTF(COMLYN,COMLEN)
      DELU(NRDC)=NEXTF(COMLYN,COMLEN)
!      endif

      if (DELL(NRDC).eq.0) DELL(NRDC)=REXP(NRDC)
      if (DELU(NRDC).eq.0) DELU(NRDC)=REXP(NRDC)

      ilen=strlng(iatm)
      jlen=strlng(jatm)

!      print*,'len...', ilen,jlen

      if(iatm(1:1) .eq. '"' .and. iatm(ilen:ilen) .eq. '"')iatm =iatm(2:ilen-1)
      if(jatm(1:1) .eq. '"' .and. jatm(jlen:jlen) .eq. '"')jatm =jatm(2:jlen-1)

      ilen=strlng(iatm)
      jlen=strlng(jatm)

      if(iatm(ilen:ilen) .eq. "#") then   ! These is to always treat
         jtmp=iatm                        ! the jatm for the methylene
         iatm=jatm                        ! spot
         jatm=jtmp

         jtemlen=ilen
         ilen=jlen
         jlen=jtemlen
      endif

      if(jatm(1:jlen).eq.'C') jlen=jlen+1   !These is to avoid from
      if(jatm(1:jlen).eq.'N') jlen=jlen+1   !picking other Cs&Ns

      if(iatm(1:jlen).eq.'C') ilen=ilen+1
      if(iatm(1:jlen).eq.'N') ilen=ilen+1

!      print*,jatm(1:jlen-1),'xxxx ',jatm,' ',ilen,' ',jlen
!      print*,jatm,' ',iatm, ires, jres, segi, segj

      DO I=1,natom
         RESN=GETRES(I,IBASE,NRES)
         call ATOMID(I,SIDI,RIDI,RENI,ACI)
!         print*,RESN,ACI,iatm,ires
!          print*,sidi,ACI

         if (RENI.eq."ILE" .and. ACI(1:3) .eq. 'CD ') ACI(1:3)="CD1"

         if (segilen .eq. 0 .and. segjlen .eq. 0) then
            if(ires.eq.RESN.and.iatm(1:ilen).eq.ACI(1:ilen)) RDCaLIS(NRDC)=I

            if(jatm(jlen:jlen) .eq. "#") then
               if(jres.eq.RESN.and.jatm(1:jlen-1).eq.ACI(1:jlen-1)) then
                  if (itera .eq. 0) RDCbLIS(NRDC)=I
                  ITERA=ITERA+1
               endif
            else
               if(jres.eq.RESN.and.jatm(1:jlen).eq.ACI(1:jlen)) then
                  RDCbLIS(NRDC)=I
                  ITERA=ITERA+1
               endif
            endif
         else
!            print*, segi,segj
            if(SIDI.eq.segi.and.ires.eq.RESN.and.iatm(1:ilen).eq.ACI(1:ilen)) RDCaLIS(NRDC)=I

            if(jatm(jlen:jlen) .eq. "#") then
               if(SIDI.eq.segj.and.jres.eq.RESN.and.jatm(1:jlen-1).eq.ACI(1:jlen-1)) then
                  if (itera .eq. 0) RDCbLIS(NRDC)=I
                  ITERA=ITERA+1
               endif
            else
               if(SIDI.eq.segj.and.jres.eq.RESN.and.jatm(1:jlen).eq.ACI(1:jlen)) then
                  RDCbLIS(NRDC)=I
                  ITERA=ITERA+1
               endif
            endif
         endif
      enddo

      IF(RDCaLIS(NRDC).EQ.0 .OR. RDCbLIS(NRDC).EQ.0) CALL WRNDIE(-5, &
           '<RDC>','Zero atom selected for this restraint.')



      IF(REXP(NRDC).EQ.ZERO) THEN
         CALL WRNDIE(1,'<RDC>','Warning: RDC value is zero')
      ENDIF

      MPRDC(NRDC)=itera
      if(itera.eq.2) nrdcmp=nrdcmp+1

!      print*,'itera',itera,NRDC,nrdcmp,mprdc(NRDC)
!     &     ,rdcalis(NRDC),rdcblis(NRDC)
      return
      end


SUBROUTINE RDC_XPLOR(URDC,NRDC,NSET,Nrdcmp,RDCaLIS,RDCbLIS,REXP, &
                     DELL,DELU,ITERA,MPRDC)
!------------------------------------------------------------------------
!     THIS ROUTINE READS AN RDC FROM XPLOR FILE
!

   use dimens_fcm
   use exfunc
   use psf
   use stream
   use comand
   use chutil
   use string
   use number
   use chm_types
   implicit none

      INTEGER       NRDC,NSET,Nrdcmp,URDC
      INTEGER       rdcalis(*),rdcblis(*)
      real(chm_real)        REXP(*),DELL(*),DELU(*)
! local
      INTEGER       I
      integer       ires, jres
      integer       resn
      integer       Length
      integer       itera,MPRDC(*)
      integer       ilen,jlen,jtemlen,segilen,segjlen,temlen
      character*80  WD,WD1
      CHARACTER*8   SIDI,RIDI,RENI,ACI
      character*5   iatm, jatm, jtmp, segi, segj
      logical       eof,WD2
      real(chm_real)        temp

      eof=.false.

      ITERA=0
      wd1=''
      segi=''
      segj=''
      segilen=0
      segjlen=0

 2    FORMAT(/,/)
 3    FORMAT(80a)
      READ(URDC,2)
      CALL RDCMND(COMLYN,MXCMSZ,COMLEN,urdc,EOF,.TRUE.,.FALSE.,'RDC> ')
      ires=GTRMI(COMLYN,COMLEN,'RESI',0)
      call GTRMWD(COMLYN,COMLEN,'NAME',4,WD,80,LENGTH)
!      print*, wd, COMLYN
      if(WD(length:length) .eq. ')') then
         iatm=WD(1:length-1)
      else
         iatm=WD
      endif

!      print*,comlyn
      call GTRMWD(COMLYN,COMLEN,'SEGID',5,WD1,80,LENGTH)
      temlen=strlng(wd1)
!      print*,wd1,temlen,comlyn

      if(temlen .ne. 0) then
         if(WD1(length:length) .eq. ')') then
            segi=WD1(1:length-1)
         else
            segi=WD1
         endif
         segilen=strlng(segi)
      endif

      CALL RDCMND(COMLYN,MXCMSZ,COMLEN,urdc,EOF,.TRUE.,.FALSE.,'RDC> ')

      jres=GTRMI(COMLYN,COMLEN,'RESI',0)
!       CALL GETBETW(comlyn,'name',')',jatm)
      call GTRMWD(COMLYN,COMLEN,'NAME',4,WD,80,LENGTH)
      if(WD(length:length) .eq. ')') then
         jatm=WD(1:length-1)
         call GTRMWD(COMLYN,COMLEN,'AND',3,WD,80,LENGTH)
      else
         jatm=WD
         WD2=CHECQUE(COMLYN,')')
         call GTRMWD(COMLYN,COMLEN,'AND',3,WD,80,LENGTH)
      endif
!      print*,comlyn, wd

      wd1=''
      call GTRMWD(COMLYN,COMLEN,'SEGID',5,WD1,80,LENGTH)
      temlen=strlng(wd1)
!      print*,wd1,temlen,comlyn

      if(temlen .ne. 0) then
         if(WD1(length:length) .eq. ')') then
            segj=WD1(1:length-1)
         else
            segj=WD1
         endif
         segjlen=strlng(segj)
         call GTRMWD(COMLYN,COMLEN,'(',1,WD,80,LENGTH)
!      print*,wd,wd1,comlyn
         rexp(nrdc)=DECODf(WD,80)
!         print*,wd,rexp(nrdc)
      else
         rexp(nrdc)=DECODf(WD,80)
         WD=NEXTA6(COMLYN,COMLEN)
      endif

      DELL(NRDC)=NEXTF(COMLYN,COMLEN)
      DELU(NRDC)=NEXTF(COMLYN,COMLEN)

!      print*, iatm,segi
!      print*, jatm,segj
!      print*,rexp(nrdc),dell(nrdc),delu(nrdc)

      if (dell(NRDC).ne.zero .and. delu(NRDC).eq.zero) then
         temp = DELL(NRDC)
         DELL(NRDC)=rexp(nrdc)-temp
         DELU(NRDC)=rexp(nrdc)+temp
      elseif (dell(NRDC).eq.zero .and. delu(NRDC).ne.zero) then
         temp = DELU(NRDC)
         DELL(NRDC)=rexp(nrdc)-temp
         DELU(NRDC)=rexp(nrdc)+temp
      else
         DELL(NRDC)=rexp(nrdc)-DELL(NRDC)
         DELU(NRDC)=rexp(nrdc)+DELU(NRDC)
      endif
!     print*, dell(nrdc), delu(nrdc)
!     print*,ires,iatm,jres,jatm,rexp(nrdc)
      ilen=strlng(iatm)
      jlen=strlng(jatm)


      if(iatm(ilen:ilen) .eq. "#") then   ! These is to always treat
         jtmp=iatm                        ! the jatm for the methylene
         iatm=jatm                        ! spot
         jatm=jtmp

         jtemlen=ilen
         ilen=jlen
         jlen=jtemlen
      endif

      if(jatm(1:jlen).eq.'C') jlen=jlen+1   !These is to avoid from
      if(jatm(1:jlen).eq.'N') jlen=jlen+1   !picking other Cs&Ns

      if(iatm(1:jlen).eq.'C') ilen=ilen+1
      if(iatm(1:jlen).eq.'N') ilen=ilen+1

      DO I=1,natom
         RESN=GETRES(I,IBASE,NRES)
         call ATOMID(I,SIDI,RIDI,RENI,ACI)
!         print*,RENI,ACI
         if (RENI.eq."ILE" .and. ACI(1:3) .eq. 'CD ') ACI(1:3)="CD1"

         if (segilen .eq. 0 .and. segjlen .eq. 0) then
            if(ires.eq.RESN.and.iatm(1:ilen).eq.ACI(1:ilen)) RDCaLIS(nrdc)=I
            if(jatm(jlen:jlen) .eq. "#") then
               if(jres.eq.RESN.and.jatm(1:jlen-1).eq.ACI(1:jlen-1)) then
                  if (itera .eq. 0) RDCbLIS(nrdc)=I
                  ITERA=ITERA+1
               endif
            else
               if(jres.eq.RESN.and.jatm(1:jlen).eq.ACI(1:jlen)) then
                  itera=itera+1
                  RDCbLIS(nrdc)=I
               endif
            endif
         else
!            print*,segi,segj
            if(SIDI.eq.segi.and.ires.eq.RESN.and.iatm(1:ilen).eq.ACI(1:ilen)) &
                RDCaLIS(nrdc)=I
            if(jatm(jlen:jlen) .eq. "#") then
               if(SIDI.eq.segj.and.jres.eq.RESN.and.jatm(1:jlen-1).eq.ACI(1:jlen-1)) then
                  if (itera .eq. 0) RDCbLIS(nrdc)=I
                  ITERA=ITERA+1
               endif
            else
               if(SIDI.eq.segj.and.jres.eq.RESN.and.jatm(1:jlen).eq.ACI(1:jlen)) then
                  itera=itera+1
                  RDCbLIS(nrdc)=I
               endif
            endif
         endif
      enddo


      IF(RDCaLIS(NRDC).EQ.0 .OR. RDCbLIS(NRDC).EQ.0) THEN
         CALL WRNDIE(-5,'<RDC>','Zero atom selected for this restraint.')
      ENDIF

      IF(REXP(NRDC).EQ.ZERO) THEN
         CALL WRNDIE(1,'<RDC>','Warning: RDC value is zero')
      ENDIF

      MPRDC(nrdc)=itera
      if(itera.eq.2) nrdcmp=nrdcmp+1
      return
      end

      SUBROUTINE RDC_MALLOC(NSET,NRDC,Nrdcmp,NRDC1,NRDCmp1,QKEY, &
                 NRDCS,NrdcmpS,NRDCE,NrdcmpE)
!------------------------------------------------------------------------
! SUBROUTINE TO SAVE & RETRIVE NUMBER OF RDCs IN MULTIPLE MEDIUM: Array handling


   use dimens_fcm
   use exfunc
   use stream
   use comand
   use number
   use coord
   use chm_types
   implicit none

      INTEGER     NRDC,NSET,Nrdcmp,NRDC1(*),NRDCmp1(*)
      INTEGER     NRDCS,NrdcmpS,NRDCE,NrdcmpE
      LOGICAL     QKEY
! local
      INTEGER     K

      IF (QKEY) THEN
         if(nset.eq.1) then
            nrdcS=1
            nrdcE=nrdc1(nset)
            nrdcmpS=1
            nrdcmpE=nrdcmp1(nset)
         else
            nrdcS=0
            nrdcE=0
            nrdcmpS=0
            nrdcmpE=0
            DO K=1,nset-1
               nrdcS=nrdc1(K)+nrdcS
               nrdcE=nrdc1(K)+nrdcE

               nrdcmpS=nrdcmp1(K)+nrdcmpS
               nrdcmpE=nrdcmp1(K)+nrdcmpE
            enddo
            nrdcS=nrdcS+1
            nrdcE=nrdcE+nrdc1(nset)

            nrdcmpS=nrdcmpS+1
            nrdcmpE=nrdcmpE+nrdcmp1(nset)
         endif
!         print*, 'check...1',nrdcS,nrdcE
      else
         if (nset.eq.1) then
            NRDC1(nset)=NRDC
            NRDCmp1(nset)=nrdcmp
         else
            DO K=1,nset-1
               if(k.gt.1)then
                  NRDC1(nset)=nrdc1(nset)-NRDC1(K)
                  NRDCmp1(nset)=nrdcmp1(nset)-nrdcmp1(K)
               else
                  NRDC1(nset)=nrdc-NRDC1(K)
                  NRDCmp1(nset)=nrdcmp-nrdcmp1(K)
               endif
            enddo
         endif
         IF(PRNLEV.GE.2) WRITE(OUTU,240) NRDC1(NSET)
 240     FORMAT(' RDC:  CURRENT NUMBER OF CONSTRAINTS=',I4)
         IF(PRNLEV.GE.2) WRITE(OUTU,241) NRDC
 241     FORMAT(' RDC:  TOTAL NUMBER OF CONSTRAINTS=',I4)
      ENDIF
      RETURN
      END

      SUBROUTINE RDCWRITER(NRDC,RDCaLIS,RDCbLIS,REXP,DELL,DELU, &
                 KHAR,LHAR,KASY,EXPO)
!------------------------------------------------------------------------
   use dimens_fcm
   use exfunc
   use stream
   use comand
   use number
   use coord
   use chm_types
   use chutil,only:atomid
   implicit none

      INTEGER     NRDC,rdcalis(*),rdcblis(*),EXPO(*)
      real(chm_real)      KHAR(*),DELL(*),DELU(*),KASY(*),REXP(*),LHAR(*)

! local
      INTEGER     I,J
      CHARACTER*8 SIDI,RIDI,RENI,ACI
      CHARACTER*8 SIDJ,RIDJ,RENJ,ACJ

      I=RDCaLIS(NRDC)
      CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
      J=RDCbLIS(NRDC)
      CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)

      IF (NRDC.eq.1) THEN
         write(OUTU,*)
         write(OUTU,500)
      ENDIF
 500  FORMAT(6x,'RDC RESTRAINTS',40x,'EXP',7X,'DELL',4X,'DELU', &
             4X,'LHAR',3X,'KHAR',3X,'KASY',1X,'EXPO')

      IF(PRNLEV.GE.0) THEN
!         WRITE(OUTU,516) NRDC,
!     &        I,SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng),
!     &        J,SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng),
!     &        REXP(NRDC),DELL(NRDC),DELU(NRDC),LHAR(NRDC),
!     &        KHAR(NRDC),KASY(NRDC),EXPO(NRDC)
! 516     FORMAT(5x,I4,2(1X,I4,1X,A4,1X,A3,1X,A4),3(1X,F8.3),3(1X,F6.3),
!     &        1X,I2)

         WRITE(OUTU,516) NRDC, &
              I,SIDI(1:idleng),RENI(1:idleng),RIDI(1:idleng), &
              ACI(1:idleng),J,SIDJ(1:idleng),RENJ(1:idleng), &
              RIDJ(1:idleng),ACJ(1:idleng),REXP(NRDC),DELL(NRDC), &
              DELU(NRDC),LHAR(NRDC),KHAR(NRDC),KASY(NRDC),EXPO(NRDC)
 516     FORMAT(5x,I4,2(1X,I4,1X,A4,1X,A4,1X,A3,1X,A4),3(1X,F8.3), &
              3(1X,F6.3),1X,I2)

      ENDIF
      RETURN
      END


      real(chm_real) FUNCTION GYRO(TYPE)
! taken from nmr.src and modfied for 'N' and added for 'S'
!-----------------------------------------------------------------------
!     Returns the gyromagnetic ratio of a nucleus (identified from
!     the TYPE array).  Constants taken from table 2.1 in
!     "NMR of Proteins and Nucleic Acid" by K. Wuthrich.
!     Units are in SI [RADIAN/(TESLA*SEC)]
!

   use rdc
   use stream
   use number
   use chm_types
   implicit none

      CHARACTER*(*) TYPE
!
      IF(TYPE(1:1).EQ.'H')THEN
         GYRO=26.7510D07
      ELSEIF(TYPE(1:1).EQ.'C')THEN
         GYRO=6.73D07
      ELSEIF(TYPE(1:1).EQ.'N')THEN
         GYRO=2.71160D07              !NOTE:POSITIVE
      ELSEIF(TYPE(1:1).EQ.'P')THEN
         GYRO=10.83D07
      ELSEIF(TYPE(1:1).EQ.'S')THEN
         GYRO=2.0534D07
      ELSE
         IF(PRNLEV.GT.3) WRITE(OUTU,'(A)')'UNKNOWN GYROMAGNETIC RATIO FOR ',TYPE
         CALL WRNDIE(-5,'<RDC>','UNKNOWN GYROMAGNETIC RATIO')
      ENDIF

!      print*,'H-C', 26.7510D07*6.73D07
!      print*,'H-H', 26.7510D07*26.7510D07
!      print*,'H-N', 26.7510D07*2.71160D07
!      print*,'C-N', 6.73D07*2.71160D07
!      print*,'C-C', 6.73D07*6.73D07
!      print*,'S-C', 2.0534D07*6.73D07

!      write(*,*)
!      write(*,*)
!      print*,'H-C', (26.7510D07*2.71160D07)/(26.7510D07*6.73D07)
!      print*,'H-H', (26.7510D07*2.71160D07)/(26.7510D07*26.7510D07)
!      print*,'C-N', (26.7510D07*2.71160D07)/(6.73D07*2.71160D07)
!      print*,'C-C', (26.7510D07*2.71160D07)/(6.73D07*6.73D07)
!      print*,'S-C', (26.7510D07*2.71160D07)/(2.0534D07*6.73D07)
!      write(*,*)
!      write(*,*)
      RETURN
      END


      SUBROUTINE RDC2(NATOM,X,Y,Z,AMASS,ERDC,DX,DY,DZ)
!------------------------------------------------------------------------
   use dimens_fcm
   use rdc
   use exfunc
   use stream
   use stream
   use number
   use chm_types
   use memory
   implicit none

!
      INTEGER NATOM
      real(chm_real)  X(*),Y(*),Z(*),AMASS(*)
      real(chm_real)  DX(*),DY(*),DZ(*)
      real(chm_real)  ERDC
! local
      INTEGER I,nrdcs,nrdcmps,nrdce,nrdcmpe,nrdc,nrdcmp
!
      ERDC=ZERO
      IRDCall=IRDCall+1
      QKEY=.TRUE.

      DO I=1,NSET
         CALL RDC_MALLOC(I,NRDC,NRDCMP,NRDC1,NRDCMP1, &
              QKEY,NRDCS,NrdcmpS,NRDCE,NrdcmpE)
         NRDC=NRDCE-NRDCS+1
         NRDCmp=NRDCmpE-NRDCmpS+1
!         print*,'hi...',NRDC,nrdce,nrdcs
! Molecular Frame
         CALL RDC_MF(NATOM,X,Y,Z,AMASS,MF,MFT,EV, &
              NRDCS,NRDCE,RDCaLIS,RDCbLIS,QRMF)
! R matrix
         CALL RDC_RMAT(NATOM,X,Y,Z,NRDCS,NRDCE,REXP, &
              RDCaLIS,RDCbLIS,IGAMMA,JGAMMA, &
              MF,MFT,RDCDIST,RDCDISTmp, &
              DCON,DRED,RDCVECX,RDCVECY, &
              RDCVECZ,RDCVECmpX,RDCVECmpY, &
              RDCVECmpZ,RDCRrMAT,RDCRMAT, &
              RDCRMATmp,CT,SMAT,MPRDC)
! Alignment Tensor
         IF (.not.QFIXA .OR. IRDCall.eq.1) THEN
            CALL RDC_AT(NRDC,DRED,SMAT,WI,W55, &
                 V,VWI,UW,WIUT,AN)
         endif

! Analytical force
         IF(QSLOW) THEN
            CALL RDC_FORCE_ALL(NATOM,X,Y,Z,AMASS,NRDCS,NRDCE, &
                 RDCaLIS,RDCbLIS,RDCRrMAT, &
                 RDCRMAT,RDCRMATmp,RDCVECX, &
                 RDCVECY,RDCVECZ,RDCVECmpX, &
                 RDCVECmpY,RDCVECmpZ,RDCDIST, &
                 RDCDISTmp,DCON,DRED,REXP, &
                 SRDC,DELL,DELU,LHAR,KHAR, &
                 KASY,EXPO,MF,MFT,EV, &
                 AN,CT,WI,W55,SMAT, &
                 V,VWI,UW,WIUT,QFIXA,QFIXB,QRMF, &
                 QSRDC,DX,DY,DZ,ERDC,MPRDC)
         ELSE
            CALL RDC_FORCE(NATOM,X,Y,Z,AMASS,NRDCS,NRDCE, &
                 RDCaLIS,RDCbLIS,RDCRrMAT, &
                 RDCRMAT,RDCRMATmp,RDCVECX, &
                 RDCVECY,RDCVECZ,RDCVECmpX, &
                 RDCVECmpY,RDCVECmpZ,RDCDIST, &
                 RDCDISTmp,DCON,DRED,REXP, &
                 SRDC,DELL,DELU,LHAR,KHAR, &
                 KASY,EXPO,MF,MFT,EV, &
                 AN,CT,WI,W55, &
                 SMAT,V,VWI,UW,WIUT,QFIXA, &
                 QFIXB,QSRDC,DX,DY,DZ,ERDC,MPRDC)
         ENDIF
      ENDDO
      QKEY=.FALSE.
!
      RETURN
      END

      SUBROUTINE RDC_MF(NATOM,X,Y,Z,AMASS,MF,MFT,EV,NRDCS,NRDCE,RDCaLIS, &
                 RDCbLIS,QRMF)
!---------------------------------------------------------------
   use number
   use consta
   use chm_types
   implicit none

      integer  natom
      INTEGER  NRDCS,NRDCE,RDCaLIS(*),RDCbLIS(*)
      real(chm_real)   x(*),y(*),z(*)
      real(chm_real)   amass(*)
      real(chm_real)   MF(3,3),MFT(3,3)
      LOGICAL  QRMF

! local
      integer  i,j,k,kk
      real(chm_real)   xi,yi,zi
      real(chm_real)   xcm,ycm,zcm,ams,tms
      real(chm_real)   xx,yy,zz,xy,yz,xz
      real(chm_real)   IT(6)
      real(chm_real)   SCR(21),EV(3)

      CHARACTER*8   SIDI,RIDI,RENI,ACI

!
!     Interta tensor components
!


      XCM = ZERO
      YCM = ZERO
      ZCM = ZERO
      TMS = ZERO

      IF(.not.QRMF) THEN
         DO I=1,NATOM
!ccccccccccccccccccccccccccccccccccccccccccccccccc
!            CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
!            if (ridi .eq. '1' .or. ridi .eq. '2' .or. ridi .eq. '3'
!     &           .or. ridi .eq. '4'  .or. ridi .eq. '5'
!     &           .or. ridi .eq. '6'  .or. ridi .eq. '7'
!     &           .or. ridi .eq. '8'  .or. ridi .eq. '9'
!     &           .or. ridi .eq. '10') then
!ccccccccccccccccccccccccccccccccccccccccccccccccc
               AMS = AMASS(I)
               TMS = TMS + AMS
               XCM = XCM + X(I)*AMS
               YCM = YCM + Y(I)*AMS
               ZCM = ZCM + Z(I)*AMS
!            endif
         ENDDO
      ELSE
         DO K=NRDCS,NRDCE
            I=RDCaLIS(K)
            AMS = AMASS(I)
            TMS = TMS + AMS
            XCM = XCM + X(I)*AMS
            YCM = YCM + Y(I)*AMS
            ZCM = ZCM + Z(I)*AMS
            I=RDCbLIS(K)
            AMS = AMASS(I)
            TMS = TMS + AMS
            XCM = XCM + X(I)*AMS
            YCM = YCM + Y(I)*AMS
            ZCM = ZCM + Z(I)*AMS
         ENDDO
      ENDIF

      IF (TMS .GT. ZERO) THEN
         XCM = XCM / TMS
         YCM = YCM / TMS
         ZCM = ZCM / TMS
      ENDIF

      XX = ZERO
      XY = ZERO
      XZ = ZERO
      YY = ZERO
      YZ = ZERO
      ZZ = ZERO

      IF(.not.QRMF) THEN
         DO I=1,NATOM
            AMS = AMASS(I)
            XI=X(I)-XCM
            YI=Y(I)-YCM
            ZI=Z(I)-ZCM
            XX = XX + AMS*XI*XI
            XY = XY + AMS*XI*YI
            XZ = XZ + AMS*XI*ZI
            YY = YY + AMS*YI*YI
            YZ = YZ + AMS*YI*ZI
            ZZ = ZZ + AMS*ZI*ZI
         ENDDO
      ELSE
         DO K=NRDCS,NRDCE
            I=RDCaLIS(K)
            XI=X(I)-XCM
            YI=Y(I)-YCM
            ZI=Z(I)-ZCM
            XX = XX + AMS*XI*XI
            XY = XY + AMS*XI*YI
            XZ = XZ + AMS*XI*ZI
            YY = YY + AMS*YI*YI
            YZ = YZ + AMS*YI*ZI
            ZZ = ZZ + AMS*ZI*ZI
            I=RDCbLIS(K)
            XI=X(I)-XCM
            YI=Y(I)-YCM
            ZI=Z(I)-ZCM
            XX = XX + AMS*XI*XI
            XY = XY + AMS*XI*YI
            XZ = XZ + AMS*XI*ZI
            YY = YY + AMS*YI*YI
            YZ = YZ + AMS*YI*ZI
            ZZ = ZZ + AMS*ZI*ZI
         ENDDO
      ENDIF

      IT(1) =  YY + ZZ
      IT(2) = -XY
      IT(3) = -XZ
      IT(4) =  XX + ZZ
      IT(5) = -YZ
      IT(6) =  XX + YY

!
!     Molecular frame (diagonalisation of intertia tensor)
!

      CALL DIAGQ(3,3,IT,MF,SCR(4),SCR(7),SCR(10),SCR(13),EV, &
           SCR(16),SCR(19),SCR(1),0)

!      WRITE(6,'(/A)') ' Principal Moments of Inertia, amu*A^2'
!      WRITE (6,105) (EV(I),I=1,3)
! 105  FORMAT(' Sorted Eigenvalues: ',3G16.7)
!      WRITE (6,106) (MF(I,1),I=1,3)
! 106  FORMAT(' Principal axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      WRITE (6,107) (MF(I,2),I=1,3)
! 107  FORMAT(' Secondary axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      WRITE (6,108) (MF(I,3),I=1,3)
! 108  FORMAT(' Tertiary axis,  X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      write (6,*)

!     transpose of molecular frame
      call TRANSPS(MFT,MF,3,3)
!
      RETURN
      END

      SUBROUTINE RDC_RMAT(NATOM,X,Y,Z,NRDCS,NRDCE,REXP,rdcalis, &
                 rdcblis,IGAMMA,JGAMMA,MF,MFT,RDCDIST,RDCDISTmp, &
                 DCON,DRED,RDCVECX,RDCVECY,RDCVECZ,RDCVECmpX, &
                 RDCVECmpY,RDCVECmpZ,RDCRrMAT,RDCRMAT,RDCRMATmp, &
                 CT,SMAT,MPRDC)
!--------------------------------------------------------------------
   use number
   use consta
   use chm_types
   implicit none

      integer     natom,NRDCS,NRDCE
      integer     rdcalis(*),rdcblis(*),MPRDC(*)
      real(chm_real)      X(*),Y(*),Z(*),REXP(*)
      real(chm_real)      MF(3,3),MFT(3,3)
      real(chm_real)      igamma(*),jgamma(*)

      real(chm_real)      rdcvecx(*),rdcvecy(*),rdcvecz(*)
      real(chm_real)      DRED(NRDCE-NRDCS+1),DCON(*),rdcdist(*)
      real(chm_real)      RDCRrMAT(*),CT(*)
      real(chm_real)      RDCRMAT(*),RDCRMATmp(*)

      real(chm_real)      rdcdistmp(*)
      real(chm_real)      rdcvecmpx(*),rdcvecmpy(*),rdcvecmpz(*)

      real(chm_real)      SMAT((NRDCE-NRDCS+1)*5)

!     local
      integer     K,KK,I,J,II,JJ,M,IN,IN1,IN2
      real(chm_real)      RT(3,3),RTmp(3,3),C(3,3),MR(3,3)
!
!
!

      M=1
      K=1
      IN2=0
      do KK=nrdcs,nrdce
         IN=(K-1)*6
         II=RDCaLIS(KK)
         JJ=RDCbLIS(KK)
         !write(*,*) "ATOM SET1", II
         !write(*,*) "ATOM SET2", JJ
         !write(*,'(3f19.9)') IGAMMA(k),JGAMMA(k),REXP(k)
         !write(*,*)
         !print*,K,mprdc(k)
         RDCVECX(K) = X(II)-X(JJ)
         RDCVECY(K) = Y(II)-Y(JJ)
         RDCVECZ(K) = Z(II)-Z(JJ)

         RT(1,1)=RDCVECX(K)*RDCVECX(K)
         RT(1,2)=RDCVECX(K)*RDCVECY(K)
         RT(1,3)=RDCVECX(K)*RDCVECZ(K)
         RT(2,1)=RDCVECX(K)*RDCVECY(K)
         RT(2,2)=RDCVECY(K)*RDCVECY(K)
         RT(2,3)=RDCVECY(K)*RDCVECZ(K)
         RT(3,1)=RDCVECX(K)*RDCVECZ(K)
         RT(3,2)=RDCVECY(K)*RDCVECZ(K)
         RT(3,3)=RDCVECZ(K)*RDCVECZ(K)

         RDCRMAT(IN+1)=RT(1,1)
         RDCRMAT(IN+2)=RT(1,2)
         RDCRMAT(IN+3)=RT(1,3)
         RDCRMAT(IN+4)=RT(2,2)
         RDCRMAT(IN+5)=RT(2,3)
         RDCRMAT(IN+6)=RT(3,3)

         RDCDIST(K)=(sqrt(RDCVECX(K)*RDCVECX(K)+ &
                    RDCVECY(K)*RDCVECY(K)+ &
                    RDCVECZ(K)*RDCVECZ(K)))

         RDCDIST(K)=RDCDIST(K)**(-1*FIVE)

         DCON(K)=(-igamma(KK)*jgamma(KK)*(6.626D-11))/(TWO*PI*PI)
         DRED(K)=REXP(KK)/DCON(K)
!         print*,igamma(K),jgamma(K),dcon(k)

!         print*, 'ch..', rexp(k)
!        write(*,'(6f19.9)') RDCDIST(K),DRED(K,1),DCON(k)
!     &              ,IGAMMA(II),JGAMMA(JJ),REXP(k)
!        write(*,*)

         DO I=1,3
            DO J=1,3
               RT(I,J)=RT(I,J)*RDCDIST(K)
            enddo
         enddo
!         DO I=1,3
!            print*,(RT(I,J),J=1,3)
!         enddo
!         write(*,*)
         IF(MPRDC(KK).eq.2) then
!: for methylene group
            !print*,'hi...',MPRDC(K),X(JJ),JJ,JJ+1
            IN1=(M-1)*6
            RDCVECmpX(m) = X(II)-X(JJ+1)
            RDCVECmpY(m) = Y(II)-Y(JJ+1)
            RDCVECmpZ(m) = Z(II)-Z(JJ+1)

            RTmp(1,1)=RDCVECmpX(m)*RDCVECmpX(m)
            RTmp(1,2)=RDCVECmpX(m)*RDCVECmpY(m)
            RTmp(1,3)=RDCVECmpX(m)*RDCVECmpZ(m)
            RTmp(2,1)=RDCVECmpX(m)*RDCVECmpY(m)
            RTmp(2,2)=RDCVECmpY(m)*RDCVECmpY(m)
            RTmp(2,3)=RDCVECmpY(m)*RDCVECmpZ(m)
            RTmp(3,1)=RDCVECmpX(m)*RDCVECmpZ(m)
            RTmp(3,2)=RDCVECmpY(m)*RDCVECmpZ(m)
            RTmp(3,3)=RDCVECmpZ(m)*RDCVECmpZ(m)


!            do I=1,3
!               write(*,*) (RTmp(I,J),J=1,3)
!            enddo

            RDCRMATmp(IN1+1)=RTmp(1,1)
            RDCRMATmp(IN1+2)=RTmp(1,2)
            RDCRMATmp(IN1+3)=RTmp(1,3)
            RDCRMATmp(IN1+4)=RTmp(2,2)
            RDCRMATmp(IN1+5)=RTmp(2,3)
            RDCRMATmp(IN1+6)=RTmp(3,3)

            RDCDISTmp(m)=(sqrt(RDCVECmpX(m)*RDCVECmpX(m)+ &
                         RDCVECmpY(m)*RDCVECmpY(m)+ &
                         RDCVECmpZ(m)*RDCVECmpZ(m)))


            RDCDISTmp(m)=RDCDISTmp(m)**(-1*FIVE)
!            print*,'hi..1',M,rdcdistmp(m)

            DO I=1,3
               DO J=1,3
                  RTmp(I,J)=RTmp(I,J)*RDCDISTmp(m)
                  RT(I,J)  =RTmp(I,J)+RT(I,J)
               enddo
            enddo
!            DO I=1,3
!               print*,(RTmp(I,J),J=1,3)
!            enddo
!            write(*,*)
!            print*,nrdcmp,M
            M=M+1
         endif


!         DO I=1,3
!            print*,(RT(I,J),J=1,3)
!         enddo
         CALL MULNXN(MR,MFT,RT,3) !MFT.RT -asymmetry
         CALL MULNXN(C,MR,MF,3)   !MFT.RT.MF -symmetry

         SMAT(IN2+1) = (C(1,1)-C(3,3))    !/(RDCDIST(K)**FIVE)
         SMAT(IN2+2) = (2*C(1,2))         !/(RDCDIST(K)**FIVE)
         SMAT(IN2+3) = (2*C(1,3))         !/(RDCDIST(K)**FIVE)
         SMAT(IN2+4) = (C(2,2)-C(3,3))    !/(RDCDIST(K)**FIVE)
         SMAT(IN2+5) = (2*C(2,3))         !/(RDCDIST(K)**FIVE)

!         print*,KK,IN2,smat(in2+1),smat(in2+2),smat(in2+3),smat(in2+4),
!     &        smat(in2+5)
!         write(*,'(i4,x,3f16.9)') K,RDCVECX(K),RDCVECY(K),RDCVECZ(K)
!
         CT(IN+1)=C(1,1)              !now OT*R*O/r**5
         CT(IN+2)=C(1,2)
         CT(IN+3)=C(1,3)
         CT(IN+4)=C(2,2)
         CT(IN+5)=C(2,3)
         CT(IN+6)=C(3,3)

         RDCRrMAT(IN+1)=RT(1,1)
         RDCRrMAT(IN+2)=RT(1,2)
         RDCRrMAT(IN+3)=RT(1,3)
         RDCRrMAT(IN+4)=RT(2,2)
         RDCRrMAT(IN+5)=RT(2,3)
         RDCRrMAT(IN+6)=RT(3,3)

!         print*,'dist....',K,rdcdist(k)
!         write(93,*)RDCDIST(K)
         K=K+1
         IN2=IN2+5
      ENDDO


!      print*, (nrdce-nrdcs)*5+1
!      print*,N-1,M-1

!      do K=1,nrdc
!         write(94,*)RDCDIST(K)
!         write(*,*) dcon(k)
!      enddo
!      write(*,*)
!      do K=1,nrdcmp
!         print*,k,RDCDISTmp(K)
!      enddo

!      stop
      RETURN
      END

!-----------------------------------------------------------------------

      SUBROUTINE RDC_AT(NRDC,DRED,SMAT1,WI,W55,V,VWI,UW, &
                       WIUT,AN)

   use number
   use consta
   use fitchg,only:SVDCMP2
   use chm_types
   implicit none

      integer  NRDC
      real(chm_real)   DRED(NRDC),SMAT1(NRDC*5)
      real(chm_real)   AN(3,3)
      real(chm_real)   VWI(5,5),UW(NRDC*5),WIUT(5*NRDC)
! local
      integer  I,J,K
      real(chm_real)   V(5,5),W(5),W55(5,5),WI(5,5),SI(5),smat(nrdc,5)
      real(chm_real)   temp

      DO I=1,5
         DO J=1,5
            WI(I,J)=ZERO
            W55(I,J)=ZERO
            if(I.le.3.and.J.le.3) then
               AN(I,J)=zero
            endif
         enddo
      enddo

      K=1
      DO I=1,NRDC
         DO J=1,5
            SMAT(I,J)=ZERO
            SMAT(I,J)=SMAT1(K)
            K=K+1
         enddo
      enddo



      CALL SVDCMP2(SMAT,NRDC,5,NRDC,5,W,V)

!      DO I=1,NRDC
!         DO J=1,5
!            print*,SMAT(I,J)
!         enddo
!      enddo

!      stop
!      write (*,*) "W= ", (W(I),I=1,5)

      do i=1,5
         if (W(i) .eq. ZERO) then
            WI(i,i)=ZERO
         else
            WI(i,i)=ONE/W(I)
         endif
            W55(i,i)=W(I)
      enddo

! V*WI
      CALL MULNXN(VWI,V,WI,5)      !passing variable

!      DO I=1,3
!         WRITE (*,'(3f20.15)')  (VWI(I,J),J=1,3)
!      enddo

!      WRITE (*,*) "W inverse", ((W(I,J),J=1,5),I=1,5)

!     Inverse of S=V*W*UT (Note: S=SMAT is replaced by U at the end of SVD)
!     Transpose of U

! U*W
      K=1
      DO  I = 1,nrdc
         DO  J = 1,5
            UW(K)=zero
            WIUT(K)=zero

            if (J.eq.1) then
               SI(1)=zero
               SI(2)=zero
               SI(3)=zero
               SI(4)=zero
               SI(5)=zero
            endif

            temp=smat(I,J)
            SMAT1(K)=temp
!            print*,smat(I,J)

            UW(K)=temp*W55(J,1)+temp*W55(J,2)+temp*W55(J,3)+ &
                    temp*W55(J,4)+temp*W55(J,5)

            WIUT(K)=TEMP*WI(1,J)+WIUT(K)
            WIUT(K)=TEMP*WI(2,J)+WIUT(K)
            WIUT(K)=TEMP*WI(3,J)+WIUT(K)
            WIUT(K)=TEMP*WI(4,J)+WIUT(K)
            WIUT(K)=TEMP*WI(5,J)+WIUT(K)

            SI(1)=TEMP*VWI(1,J)+SI(1)
            SI(2)=TEMP*VWI(2,J)+SI(2)
            SI(3)=TEMP*VWI(3,J)+SI(3)
            SI(4)=TEMP*VWI(4,J)+SI(4)
            SI(5)=TEMP*VWI(5,J)+SI(5)

            if(J.eq.5) then
               AN(1,1)=SI(1)*DRED(I)+AN(1,1)
               AN(1,2)=SI(2)*DRED(I)+AN(1,2)
               AN(1,3)=SI(3)*DRED(I)+AN(1,3)
               AN(2,2)=SI(4)*DRED(I)+AN(2,2)
               AN(2,3)=SI(5)*DRED(I)+AN(2,3)
            endif
            K=K+1
         ENDDO
      ENDDO

!      K=1
!      DO  I = 1,nrdc
!         DO  J = 1,5
!            print*,smat1(K)
!            K=K+1
!         enddo
!      enddo
!      stop


      AN(2,1)=AN(1,2)
      AN(3,1)=AN(1,3)
      AN(3,2)=AN(2,3)
      AN(3,3)=(-AN(1,1)-AN(2,2))


!      DO I=1,3
!         WRITE (*,'(3f20.15)')  (AN(I,J),J=1,3)
!      enddo

      RETURN
      END


      SUBROUTINE RDC_FORCE_ALL(NATOM,X,Y,Z,AMASS,NRDCS,NRDCE,rdcalis, &
                 rdcblis,RDCRrMAT,RDCRMAT,RDCRMATmp,RDCVECX,RDCVECY, &
                 RDCVECZ,RDCVECmpX,RDCVECmpY,RDCVECmpZ,RDCDIST, &
                 RDCDISTmp,DCON,DRED,REXP,SRDC,DELL,DELU,LHAR,KHAR,KASY, &
                 EXPO,MF,MFT,EV,AN,CT,WI,W55,U,V,VWI,UW,WIUT,QFIXA, &
                 QFIXB,QRMF,QSRDC,DX,DY,DZ,ERDC,MPRDC)
!------------------------------------------------------------------------
   use number
   use consta
   use chm_types
   use stream
   implicit none

      integer  natom
      real(chm_real)   X(*),Y(*),Z(*),AMASS(*),EV(3)
      real(chm_real)   DX(*),DY(*),DZ(*),ERDC

      integer  NRDCS,NRDCE,EXPO(*)
      integer  rdcalis(*),rdcblis(*),mprdc(*)
      real(chm_real)   MF(3,3),MFT(3,3)
      real(chm_real)   RDCVECX(*),RDCVECY(*),RDCVECZ(*),RDCDIST(*)
      real(chm_real)   RDCVECmpX(*),RDCVECmpY(*),RDCVECmpZ(*),RDCDISTmp(*)
      real(chm_real)   DCON(*),DRED(NRDCE-NRDCS+1)
      real(chm_real)   KHAR(*),KASY(*),DELL(*),DELU(*),LHAR(*),REXP(*),SRDC(*)
      real(chm_real)   RDCRrMAT(*),CT(*)
      real(chm_real)   RDCRMAT(*),RDCRMATmp(*)
      real(chm_real)   WI(5,5),W55(5,5),U(NRDCE-NRDCS+1,5),V(5,5),AN(3,3)
      real(chm_real)   VWI(5,5),UW((NRDCE-NRDCS+1)*5),WIUT(5*(NRDCE-NRDCS+1))
      logical  QFIXA,QFIXB,QRMF,QSRDC
! local
      integer  I,J,K,KK,L,M,MM,N,NN,IN,IN1
! energy
      real(chm_real)   rdcrmin,rdcrmax,rdchmin,rdchmax
      real(chm_real)   kharm,kasym,lharm
      integer  NRDC,expon
      real(chm_real)   C(3,3),RrT(3,3),AC(3,3)
      real(chm_real)   RT(3,3),RTmp(3,3)
      real(chm_real)   softA,softBL(NRDCE-NRDCS+1),softBU(NRDCE-NRDCS+1)
! force
      real(chm_real)   DCAL(NRDCE-NRDCS+1)
      real(chm_real)   XCM,YCM,ZCM,AMS,TMS
      real(chm_real)   DXMF(3,3),DYMF(3,3),DZMF(3,3)
      real(chm_real)   DXMFT(3,3),DYMFT(3,3),DZMFT(3,3)
      real(chm_real)   RX(3,3),RY(3,3),RZ(3,3)
      real(chm_real)   RXmp(3,3),RYmp(3,3),RZmp(3,3)
      real(chm_real)   RDCDX,RDCDY,RDCDZ
      real(chm_real)   numer,numer1,FTM1,FTM2,FTM3,FTMA
      real(chm_real)   SX(NRDCE-NRDCS+1,5),SY(NRDCE-NRDCS+1,5),SZ(NRDCE-NRDCS+1,5)
      real(chm_real)   FTMP1(3,3),FTMP2(3,3)
      real(chm_real)   F2X(3,3),F3X(3,3),F6(3,3),F7(3,3),F9X(3,3),F10(3,3),F11(3,3),F12(3,3)
      real(chm_real)   F2Y(3,3),F3Y(3,3),F9Y(3,3)
      real(chm_real)   F2Z(3,3),F3Z(3,3),F9Z(3,3)
      real(chm_real)   AX(3,3),AY(3,3),AZ(3,3)
      integer  mprdctmp

      real(chm_real)   KRDCVECX,KRDCVECY,KRDCVECZ,KRDCDIST
      real(chm_real)   KRDCVECmpX,KRDCVECmpY,KRDCVECmpZ
      real(chm_real)   KRDCDISTmp,KDCAL,KDCON

!
!
!

! to test energy
!      DCAL(1)=-250.0D0 !asymetric
!      DCAL(2)=-30.0D0 *SRDC(K)

!      DO I=1,450
!         ERDC=0.0D0
!         K=1
!         write(*,*) 'khar', KHAR(K),'kasy',KASY(K),'dell',DELL(K),
!     &              'delu', DELU(K),'rexp', REXP(K),'expo',EXPO(K),
!     &              'lhar',LHAR(K),'dcal',DCAL(K)

!         kharm=KHAR(k)
!         kasym=KASY(k)
!         lharm= 5.0D0
!         expon=1
!         dell(k)=0 !symetric
!         delu(k)=0 !symetric

!         if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
!            rdcrmin=REXP(k)*SRDC(K)
!            rdcrmax=REXP(k)*SRDC(K)
!         else
!            rdcrmin=DELL(k)*SRDC(K)
!            rdcrmax=DELU(k)*SRDC(K)
!         endif

!         rdchmin=rdcrmin-lharm
!         rdchmax=rdcrmax+lharm

! Equilibrium
!         IF (Dcal(K).ge.rdcrmin.and.Dcal(K).le.rdcrmax) then
!            ERDC=ERDC+zero
!            print*,'EQUI'
! Harmonic (lower bound)
!         ELSEIF (Dcal(K).lt.rdcrmin.and.Dcal(K).ge.rdchmin) then
!            ERDC=ERDC+(kharm*(dcal(k)-rdcrmin)**TWO)
!            print*,'HARL'
! Harmonic (upper bound)
!         ELSEIF (Dcal(K).gt.rdcrmax.and.Dcal(K).lt.rdchmax) then
!            print*,'HARU'
!            ERDC=ERDC+(kharm*(dcal(k)-rdcrmax)**TWO)
! soft asympotote (lower bound)
!         ELSEIf (Dcal(K).lt.rdchmin) THEN
!            softBL(K)=(TWO*kharm*lharm-kasym)/
!     &           (expon*(-lharm)**(-expon-1))
!            softA=kharm*lharm*lharm-
!     &           softBL(K)*(-lharm)**(-expon)-
!     &           kasym*lharm
!            ERDC=ERDC+
!     &           SOFTA+
!     &           SOFTBL(K)*(Dcal(K)-rdcrmin)**(-expon)-
!     &           kasym*(DCAL(K)-rdcrmin)
!            print*,'softL'
! soft asympotote (upper bound)
!         ELSEIF (Dcal(K).ge.rdchmax) then
!            softBU(K)=(-TWO*kharm*lharm+kasym)/
!     &           (expon*lharm**(-expon-1))
!            softA=kharm*lharm*lharm-
!     &           softBU(K)*lharm**(-expon)-
!     &           kasym*lharm
!            ERDC=ERDC+
!     &           SOFTA+
!     &           SOFTBU(K)*(DCAL(K)-rdcrmax)**(-expon)+
!     &           kasym*(DCAL(K)-rdcrmax)
!
!         ENDIF
!         write(95,*) DCAL(K),ERDC
!         dcal(k)=dcal(k)+1.0D0
!         dcal(k)=dcal(k)-1.0D0
!      ENDDO
!      stop



! start
!     ERDC=0.0D0
      KK=1
      NRDC=NRDCE-NRDCS+1
      DO K=NRDCS,NRDCE
!         write(*,*) 'khar', KHAR(K),'kasy',KASY(K),'kdel',DELT(K),
!     &              'rexp', REXP(K),'expo',EXPO(K),'lhar',LHAR(K)

         IN=(KK-1)*6

         C(1,1)=CT(IN+1)         !MFT.R.r**-5.MF
         C(1,2)=CT(IN+2)
         C(1,3)=CT(IN+3)
         C(2,1)=CT(IN+2)
         C(2,2)=CT(IN+4)
         C(2,3)=CT(IN+5)
         C(3,1)=CT(IN+3)
         C(3,2)=CT(IN+5)
         C(3,3)=CT(IN+6)

         CALL MULNXN(AC,AN,C,3) !AN.MFT.R.r**-5.MF

         Dcal(KK)=DCON(KK)*(AC(1,1)+AC(2,2)+AC(3,3))

!         write(90,'(a4,2i4,3f12.6,f13.3)'),'DCAL',K,MPRDC(k),REXP(K),
!     &        dcal(k),REXP(K)-dcal(k),dcon(k)

         kharm=KHAR(k)
         kasym=KASY(k)
         lharm=LHAR(k)
         expon=EXPO(k)

         if (QSRDC) then
            kdcal=dcal(kk)*SRDC(K)
            if((rexp(k).eq.DELL(k)).and.(rexp(k).eq.DELU(k))) then
               rdcrmin=REXP(k)*SRDC(k)
               rdcrmax=REXP(k)*SRDC(K)
            else
               rdcrmin=DELL(k)*SRDC(K)
               rdcrmax=DELU(k)*SRDC(K)
            endif
         else
            kdcal=dcal(kk)
            if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(k))) then
               rdcrmin=REXP(k)
               rdcrmax=REXP(k)
            else
               rdcrmin=DELL(k)
               rdcrmax=DELU(k)
            endif
         endif

         rdchmin=rdcrmin-lharm
         rdchmax=rdcrmax+lharm
! Equilibrium
         IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
            ERDC=ERDC+zero
!            print*,'EQUI'
! Harmonic (lower bound)
         ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin) then
            ERDC=ERDC+(kharm*(kdcal-rdcrmin)**TWO)
!            print*,'HARL'
! Harmonic (upper bound)
         ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.lt.rdchmax) then
!            print*,'HARU'
            ERDC=ERDC+(kharm*(kdcal-rdcrmax)**TWO)
!.......................old method-to be removed
!            ELSEIf (Kdcal.lt.rdchmin) THEN
!                  softBL(K)=(TWO*kharm*lharm+kasym)/
!     &                 (expon*lharm**(-expon-1))
!                  softA=kharm*lharm*lharm+
!     &                 softBL(K)*lharm**(-expon)-
!     &                 kasym*lharm
!                  erdc=erdc+
!     &                 SOFTA+
!     &                 SOFTBL(K)*(Kdcal-rdcrmin)**(-expon)-
!     &                 kasym*(KDCAL-rdcrmin)
!.......................................

! soft asympotote (lower bound)
         ELSEIf (Kdcal.lt.rdchmin) THEN
            softBL(KK)=(TWO*kharm*lharm-kasym)/(expon*(-lharm)**(-expon-1))
            softA=kharm*lharm*lharm- &
                  softBL(KK)*(-lharm)**(-expon)- &
                  kasym*lharm
            ERDC=ERDC+ &
                 SOFTA+ &
                 SOFTBL(KK)*(Kdcal-rdcrmin)**(-expon)- &
                 kasym*(KDCAL-rdcrmin)
!            print*,'softL'
! soft asympotote (upper bound)
         ELSEIF (Kdcal.ge.rdchmax) then
            softBU(KK)=(-TWO*kharm*lharm+kasym)/(expon*lharm**(-expon-1))
            softA=kharm*lharm*lharm- &
                  softBU(KK)*lharm**(-expon)- &
                  kasym*lharm
            ERDC=ERDC+ &
                 SOFTA+ &
                 SOFTBU(KK)*(KDCAL-rdcrmax)**(-expon)+ &
                 kasym*(KDCAL-rdcrmax)
!            print*,'softU'
         ENDIF
         KK=KK+1
      ENDDO
!      print*,'egy=  ', ERDC


      XCM = ZERO
      YCM = ZERO
      ZCM = ZERO
      TMS = ZERO
      AMS = ZERO

      IF(.not.QRMF) THEN
         DO I=1,NATOM
            AMS = AMASS(I)
            TMS = TMS + AMS
            XCM = XCM + X(I)*AMS
            YCM = YCM + Y(I)*AMS
            ZCM = ZCM + Z(I)*AMS
         ENDDO
      ELSE
         DO K=NRDCS,NRDCE
            I=RDCaLIS(K)
            AMS = AMASS(I)
            TMS = TMS + AMS
            XCM = XCM + X(I)*AMS
            YCM = YCM + Y(I)*AMS
            ZCM = ZCM + Z(I)*AMS
            I=RDCbLIS(K)
            AMS = AMASS(I)
            TMS = TMS + AMS
            XCM = XCM + X(I)*AMS
            YCM = YCM + Y(I)*AMS
            ZCM = ZCM + Z(I)*AMS
         ENDDO
      ENDIF

      IF (TMS.GT.ZERO) THEN
         XCM = XCM / TMS
         YCM = YCM / TMS
         ZCM = ZCM / TMS
      ENDIF

!      write(*,*)
!      write(*,*)
!      write(*,*)'force'

      N=1
      MM=1
      DO M=NRDCS,NRDCE
         IF (mpRDC(m).EQ.1) MPRDCtmp=mpRDC(m)
         IF (mpRDC(m).EQ.2) MPRDCtmp=3
         DO NN=1,MPRDCtmp
            IF(NN.eq.1) I=RDCaLIS(M)
            IF(NN.eq.2) I=RDCbLIS(M)
            IF(NN.eq.3) I=RDCbLIS(M)+1
!               print*,'LIS..',I
               call RDC_DRMF(X,Y,Z,AMASS,TMS,EV,I,XCM,YCM,ZCM,MF,DXMF,DYMF,DZMF)

            RDCDX=ZERO
            RDCDY=ZERO
            RDCDZ=ZERO
            KK=1
            DO K=NRDCS,NRDCE
               IN=(KK-1)*6
               RrT(1,1)=RDCRrMAT(IN+1)
               RrT(1,2)=RDCRrMAT(IN+2)
               RrT(1,3)=RDCRrMAT(IN+3)
               RrT(2,1)=RDCRrMAT(IN+2)
               RrT(2,2)=RDCRrMAT(IN+4)
               RrT(2,3)=RDCRrMAT(IN+5)
               RrT(3,1)=RDCRrMAT(IN+3)
               RrT(3,2)=RDCRrMAT(IN+5)
               RrT(3,3)=RDCRrMAT(IN+6)

               RT(1,1)=RDCRMAT(IN+1)
               RT(1,2)=RDCRMAT(IN+2)
               RT(1,3)=RDCRMAT(IN+3)
               RT(2,1)=RDCRMAT(IN+2)
               RT(2,2)=RDCRMAT(IN+4)
               RT(2,3)=RDCRMAT(IN+5)
               RT(3,1)=RDCRMAT(IN+3)
               RT(3,2)=RDCRMAT(IN+5)
               RT(3,3)=RDCRMAT(IN+6)

               numer=RDCDIST(KK)**(TWO/FIVE)

! Derivative of Rmatrix
               IF (KK.eq.MM) THEN

                  krdcvecx=rdcvecx(KK)
                  krdcvecy=rdcvecy(KK)
                  krdcvecz=rdcvecz(KK)
                  krdcdist=rdcdist(KK)

                  RX(1,1)=TWO*KRDCVECX
                  RX(1,2)=KRDCVECY
                  RX(1,3)=KRDCVECZ
                  RX(2,1)=KRDCVECY
                  RX(2,2)=ZERO
                  RX(2,3)=ZERO
                  RX(3,1)=KRDCVECZ
                  RX(3,2)=ZERO
                  RX(3,3)=ZERO

                  RY(1,1)=ZERO
                  RY(1,2)=KRDCVECX
                  RY(1,3)=ZERO
                  RY(2,1)=KRDCVECX
                  RY(2,2)=TWO*KRDCVECY
                  RY(2,3)=KRDCVECZ
                  RY(3,1)=ZERO
                  RY(3,2)=KRDCVECZ
                  RY(3,3)=ZERO

                  RZ(1,1)=ZERO
                  RZ(1,2)=ZERO
                  RZ(1,3)=KRDCVECX
                  RZ(2,1)=ZERO
                  RZ(2,2)=ZERO
                  RZ(2,3)=KRDCVECY
                  RZ(3,1)=KRDCVECX
                  RZ(3,2)=KRDCVECY
                  RZ(3,3)=TWO*KRDCVECZ

!               print*,"start",mprdctmp

                  ftm1=FIVE*KRDCVECX*KRDCDIST*numer
                  ftm2=FIVE*KRDCVECY*KRDCDIST*numer
                  ftm3=FIVE*KRDCVECZ*KRDCDIST*numer

                  DO L=1,3
                     DO J=1,3
                        RX(L,J)=RX(L,J)*KRDCDIST-RT(L,J)*FTM1
                        RY(L,J)=RY(L,J)*KRDCDIST-RT(L,J)*FTM2
                        RZ(L,J)=RZ(L,J)*KRDCDIST-RT(L,J)*FTM3
                     enddo
                  enddo

                  if(NN.eq.2) then
!: only for methylene group-1st hydrogen
!                     print*,"2nd.."
                     DO L=1,3
                        DO J=1,3
                           RX(L,J)=-RX(L,J)
                           RY(L,J)=-RY(L,J)
                           RZ(L,J)=-RZ(L,J)
                        enddo
                     enddo
                  endif

                  IF(MPRDCtmp.eq.3.and.(NN.eq.1.or.NN.eq.3))then

                     IN1=(N-1)*6

                     krdcvecmpx=rdcvecmpx(N)
                     krdcvecmpy=rdcvecmpy(N)
                     krdcvecmpz=rdcvecmpz(N)
                     krdcdistmp=rdcdistmp(N)

!: only for methylene groups
!                  print*,'hit',mprdctmp,k,n
                     RTmp(1,1)=RDCRMATmp(IN1+1)
                     RTmp(1,2)=RDCRMATmp(IN1+2)
                     RTmp(1,3)=RDCRMATmp(IN1+3)
                     RTmp(2,1)=RDCRMATmp(IN1+2)
                     RTmp(2,2)=RDCRMATmp(IN1+4)
                     RTmp(2,3)=RDCRMATmp(IN1+5)
                     RTmp(3,1)=RDCRMATmp(IN1+3)
                     RTmp(3,2)=RDCRMATmp(IN1+5)
                     RTmp(3,3)=RDCRMATmp(IN1+6)

                     numer1=RDCDISTmp(N)**(TWO/FIVE)

                     RXmp(1,1)=TWO*KRDCVECmpX
                     RXmp(1,2)=KRDCVECmpY
                     RXmp(1,3)=KRDCVECmpZ
                     RXmp(2,1)=KRDCVECmpY
                     RXmp(2,2)=ZERO
                     RXmp(2,3)=ZERO
                     RXmp(3,1)=KRDCVECmpZ
                     RXmp(3,2)=ZERO
                     RXmp(3,3)=ZERO

                     RYmp(1,1)=ZERO
                     RYmp(1,2)=KRDCVECmpX
                     RYmp(1,3)=ZERO
                     RYmp(2,1)=KRDCVECmpX
                     RYmp(2,2)=TWO*KRDCVECmpY
                     RYmp(2,3)=KRDCVECmpZ
                     RYmp(3,1)=ZERO
                     RYmp(3,2)=KRDCVECmpZ
                     RYmp(3,3)=ZERO

                     RZmp(1,1)=ZERO
                     RZmp(1,2)=ZERO
                     RZmp(1,3)=KRDCVECmpX
                     RZmp(2,1)=ZERO
                     RZmp(2,2)=ZERO
                     RZmp(2,3)=KRDCVECmpY
                     RZmp(3,1)=KRDCVECmpX
                     RZmp(3,2)=KRDCVECmpY
                     RZmp(3,3)=TWO*KRDCVECmpZ


                     ftm1=FIVE*KRDCVECmpX*KRDCDISTmp*numer1
                     ftm2=FIVE*KRDCVECmpY*KRDCDISTmp*numer1
                     ftm3=FIVE*KRDCVECmpZ*KRDCDISTmp*numer1

                     DO L=1,3
                        DO J=1,3
                           RXmp(L,J)=RXmp(L,J)*KRDCDISTmp-FTM1*RTmp(L,J)
                           RYmp(L,J)=RYmp(L,J)*KRDCDISTmp-FTM2*RTmp(L,J)
                           RZmp(L,J)=RZmp(L,J)*KRDCDISTmp-FTM3*RTmp(L,J)

                           if (NN.eq.3) then
!: only for methylene group-2nd hydrogen
                              RX(L,J)=-RXmp(L,J)
                              RY(L,J)=-RYmp(L,J)
                              RZ(L,J)=-RZmp(L,J)
                           else
                              RX(L,J)=RX(L,J)+RXmp(L,J)
                              RY(L,J)=RY(L,J)+RYmp(L,J)
                              RZ(L,J)=RZ(L,J)+RZmp(L,J)
                           endif
                        enddo
                     enddo
!                  DO I=1,3
!                     print*,(RXmp(I,J),J=1,3)
!                  enddo
!                  write(*,*)

!                  write(*,*)
!                  print*,'here'
!                  do L=1,3
!                     write(*,*) (Rxmp1(L,J),J=1,3)
!                  enddo
!                  write(*,*)
!                  print*,'mp...',N,rdcdistmp(n)
                     if (NN.eq.3)  N=N+1
                  endif
               ELSE
                  DO L=1,3
                     DO J=1,3
                        RX(L,J)=ZERO
                        RY(L,J)=ZERO
                        RZ(L,J)=ZERO
                     enddo
                  enddo
               ENDIF

!            DO I=1,3
!               print*,(RXmp1(I,J),J=1,3)
!            enddo
!            write(*,*)

               if(KK.eq.MM) then
! Derivative w.r.t X
                  CALL MULNXN(FTMP1,MFT,RX,3)
                  CALL MULNXN(F2X,FTMP1,MF,3)

                  CALL MULNXN(FTMP2,MFT,RrT,3)
                  CALL MULNXN(F3X,FTMP2,DXMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F6,AN,F2X,3)
                  CALL MULNXN(F7,AN,F3X,3)

                  FTM1=F6(1,1)+F6(2,2)+F6(3,3)+ &
                       (2*(F7(1,1)+F7(2,2)+F7(3,3)))

! Derivative w.r.t Y
                  CALL MULNXN(FTMP1,MFT,RY,3)
                  CALL MULNXN(F2Y,FTMP1,MF,3)

                  CALL MULNXN(F3Y,FTMP2,DYMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F6,AN,F2Y,3)
                  CALL MULNXN(F7,AN,F3Y,3)

                  FTM2=F6(1,1)+F6(2,2)+F6(3,3)+ &
                       (2*(F7(1,1)+F7(2,2)+F7(3,3)))

! Derivative w.r.t Z
                  CALL MULNXN(FTMP1,MFT,RZ,3)
                  CALL MULNXN(F2Z,FTMP1,MF,3)

                  CALL MULNXN(F3Z,FTMP2,DZMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F6,AN,F2Z,3)
                  CALL MULNXN(F7,AN,F3Z,3)

                  FTM3=F6(1,1)+F6(2,2)+F6(3,3)+ &
                       (2*(F7(1,1)+F7(2,2)+F7(3,3)))
               else
! Derivative w.r.t X
                  CALL MULNXN(FTMP2,MFT,RrT,3)
                  CALL MULNXN(F3X,FTMP2,DXMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F7,AN,F3X,3)
                  FTM1=2*(F7(1,1)+F7(2,2)+F7(3,3))

! Derivative w.r.t Y
                  CALL MULNXN(F3Y,FTMP2,DYMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F7,AN,F3Y,3)
                  FTM2=2*(F7(1,1)+F7(2,2)+F7(3,3))

! Derivative w.r.t Z
                  CALL MULNXN(F3Z,FTMP2,DZMF,3)

! Force from OT,R,O,r5
                  CALL MULNXN(F7,AN,F3Z,3)
                  FTM3=2*(F7(1,1)+F7(2,2)+F7(3,3))
               endif

               kharm=KHAR(k)
               kasym=KASY(k)
               lharm=LHAR(k)
               expon=EXPO(k)

               if (QSRDC) then
                  kdcal=dcal(kk)*SRDC(K)
                  kdcon=dcon(kk)*SRDC(k)
                  if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
                     rdcrmin=REXP(k)*SRDC(K)
                     rdcrmax=REXP(k)*SRDC(K)
                  else
                     rdcrmin=DELL(k)*SRDC(K)
                     rdcrmax=DELU(k)*SRDC(K)
                  endif
               else
                  kdcal=dcal(kk)
                  kdcon=dcon(kk)
                  if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
                     rdcrmin=REXP(k)
                     rdcrmax=REXP(k)
                  else
                     rdcrmin=DELL(k)
                     rdcrmax=DELU(k)
                  endif
               endif

               rdchmin=rdcrmin-lharm
               rdchmax=rdcrmax+lharm

! Equilibrium
               IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
                  RDCDX=RDCDX
                  RDCDY=RDCDY
                  RDCDZ=RDCDZ
!               print*,'EQUI'
! Harmonic (lower bound)
               ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin) then
                  FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMIN)
                  RDCDX=RDCDX+(FTMA*FTM1)
                  RDCDY=RDCDY+(FTMA*FTM2)
                  RDCDZ=RDCDZ+(FTMA*FTM3)
!     print*,'HARL'
! Harmonic (upper bound)
               ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.lt.rdchmax) then
!               print*,'HARU'
                  FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMAX)
                  RDCDX=RDCDX+(FTMA*FTM1)
                  RDCDY=RDCDY+(FTMA*FTM2)
                  RDCDZ=RDCDZ+(FTMA*FTM3)
! soft asympotote (lower bound)
               ELSEIf (Kdcal.lt.rdchmin) THEN
                  FTMA=-softBL(KK)*expon*(KDCAL-RDCRMIN)**(-expon-1) &
                       *FTM1*KDCON-KASYM*FTM1*KDCON
                  RDCDX=RDCDX+FTMA
                  FTMA=-softBL(KK)*expon*(KDCAL-RDCRMIN)**(-expon-1) &
                       *FTM2*KDCON-KASYM*FTM2*KDCON
                  RDCDY=RDCDY+FTMA
                  FTMA=-softBL(KK)*expon*(KDCAL-RDCRMIN)**(-expon-1) &
                       *FTM3*KDCON-KASYM*FTM3*KDCON
                  RDCDZ=RDCDZ+FTMA
!               print*,'softL'
! soft asympotote (upper bound)
               ELSEIF (Kdcal.ge.rdchmax) then
                  FTMA=-softBU(KK)*expon*(KDCAL-rdcrmax)**(-expon-1) &
                       *FTM1*KDCON+kasym*FTM1*KDCON
                  RDCDX=RDCDX+FTMA
                  FTMA=-softBU(KK)*expon*(KDCAL-rdcrmax)**(-expon-1) &
                       *FTM2*KDCON+kasym*FTM2*KDCON
                  RDCDY=RDCDY+FTMA
                  FTMA=-softBU(KK)*expon*(KDCAL-rdcrmax)**(-expon-1) &
                       *FTM3*KDCON+kasym*FTM3*KDCON
                  RDCDZ=RDCDZ+FTMA
               ENDIF
!            print*,'part1'
!            print*,rdcdx,rdcdy,rdcdz

! Derivative of S
               IF (.not. QFIXA) then
                  if (KK.eq.MM) then
                     DO L=1,3
                        DO J=1,3
                           F9X(L,J)=F2X(L,J)+F3X(J,L)+F3X(L,J)
                           F9Y(L,J)=F2Y(L,J)+F3Y(J,L)+F3Y(L,J)
                           F9Z(L,J)=F2Z(L,J)+F3Z(J,L)+F3Z(L,J)
                        enddo
                     enddo
                  else
                     DO L=1,3
                        DO J=1,3
                           F9X(L,J)=F3X(J,L)+F3X(L,J)
                           F9Y(L,J)=F3Y(J,L)+F3Y(L,J)
                           F9Z(L,J)=F3Z(J,L)+F3Z(L,J)
                        enddo
                     enddo
                  endif
                  SX(KK,1) = F9X(1,1)-F9X(3,3)
                  SX(KK,2) = TWO*F9X(1,2)
                  SX(KK,3) = TWO*F9X(1,3)
                  SX(KK,4) = F9X(2,2)-F9X(3,3)
                  SX(KK,5) = TWO*F9X(2,3)

                  SY(KK,1) = F9Y(1,1)-F9Y(3,3)
                  SY(KK,2) = TWO*F9Y(1,2)
                  SY(KK,3) = TWO*F9Y(1,3)
                  SY(KK,4) = F9Y(2,2)-F9Y(3,3)
                  SY(KK,5) = TWO*F9Y(2,3)

                  SZ(KK,1) = F9Z(1,1)-F9Z(3,3)
                  SZ(KK,2) = TWO*F9Z(1,2)
                  SZ(KK,3) = TWO*F9Z(1,3)
                  SZ(KK,4) = F9Z(2,2)-F9Z(3,3)
                  SZ(KK,5) = TWO*F9Z(2,3)
!            print*,'rdc..',K,mprdctmp
               endif
               KK=KK+1
            enddo

!         if(mprdctmp(m).eq.2) then
!         print*,'smat'
!            do L=1,nrdc
!               write(*,'(2i4,5f12.8)') mprdctmp,L, (SX(L,J),J=1,5)
!            enddo

!            write(*,*)
!            write(*,*)

!         endif
            IF (.not.QFIXA) then
               CALL RDC_DERAT_ALL(NRDC,WI,W55,U,V,VWI,UW,WIUT,AN,DRED,SX,SY,SZ,AX,AY,AZ)
               KK=1
               DO K=NRDCS,NRDCE
                  IN=(KK-1)*6
                  C(1,1)=CT(IN+1) !MFT.R.r**-5..MF
                  C(1,2)=CT(IN+2)
                  C(1,3)=CT(IN+3)
                  C(2,1)=CT(IN+2)
                  C(2,2)=CT(IN+4)
                  C(2,3)=CT(IN+5)
                  C(3,1)=CT(IN+3)
                  C(3,2)=CT(IN+5)
                  C(3,3)=CT(IN+6)

                  CALL MULNXN(F10,AX,C,3)
                  CALL MULNXN(F11,AY,C,3)
                  CALL MULNXN(F12,AZ,C,3)

                  FTM1=F10(1,1)+F10(2,2)+F10(3,3)
                  FTM2=F11(1,1)+F11(2,2)+F11(3,3)
                  FTM3=F12(1,1)+F12(2,2)+F12(3,3)

                  kharm=KHAR(k)
                  kasym=KASY(k)
                  lharm=LHAR(k)
                  expon=EXPO(k)

                  if (QSRDC) then
                     kdcal=dcal(Kk)*SRDC(K)
                     kdcon=dcon(kK)*SRDC(k)
                     if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K)))then
                        rdcrmin=REXP(k)*SRDC(K)
                        rdcrmax=REXP(k)*SRDC(K)
                     else
                        rdcrmin=DELL(k)*SRDC(K)
                        rdcrmax=DELU(k)*SRDC(K)
                     endif
                  else
                     kdcal=dcal(kK)
                     kdcon=dcon(kK)
                     if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
                        rdcrmin=REXP(k)
                        rdcrmax=REXP(k)
                     else
                        rdcrmin=DELL(k)
                        rdcrmax=DELU(k)
                     endif
                  endif

                  rdchmin=rdcrmin-lharm
                  rdchmax=rdcrmax+lharm
! Equilibrium
                  IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
                     RDCDX=RDCDX
                     RDCDY=RDCDY
                     RDCDZ=RDCDZ
!                 print*,'EQUI'
! Harmonic (lower bound)
                  ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin)then
                     FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMIN)
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
!                 print*,'HARL'
! Harmonic (upper bound)
                  ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.lt.rdchmax)then
!                 print*,'HARU'
                     FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMAX)
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
! soft asympotote (lower bound)
                  ELSEIf (Kdcal.lt.rdchmin) THEN
                     FTMA=-softBL(KK)*expon* &
                          (KDCAL-RDCRMIN)**(-expon-1)* &
                          FTM1*KDCON-KASYM*FTM1*KDCON
                     RDCDX=RDCDX+FTMA
                     FTMA=-softBL(KK)*expon* &
                          (KDCAL-RDCRMIN)**(-expon-1)* &
                          FTM2*KDCON-KASYM*FTM2*KDCON
                     RDCDY=RDCDY+FTMA
                     FTMA=-softBL(KK)*expon* &
                          (KDCAL-RDCRMIN)**(-expon-1)* &
                          FTM3*KDCON-KASYM*FTM3*KDCON
                     RDCDZ=RDCDZ+FTMA
!                 print*,'softL'
! soft asympotote (upper bound)
                  ELSEIF (Kdcal.ge.rdchmax) then
                     FTMA=-softBU(KK)*expon* &
                          (KDCAL-rdcrmax)**(-expon-1)* &
                          FTM1*KDCON+kasym*FTM1*KDCON
                     RDCDX=RDCDX+FTMA
                     FTMA=-softBU(KK)*expon* &
                          (KDCAL-rdcrmax)**(-expon-1)* &
                          FTM2*KDCON+kasym*FTM2*KDCON
                     RDCDY=RDCDY+FTMA
                     FTMA=-softBU(KK)*expon* &
                          (KDCAL-rdcrmax)**(-expon-1)* &
                          FTM3*KDCON+kasym*FTM3*KDCON
                     RDCDZ=RDCDZ+FTMA
                  endif
                  KK=KK+1
               enddo
            ENDIF

            DX(I)=DX(I)+RDCDX
            DY(I)=DY(I)+RDCDY
            DZ(I)=DZ(I)+RDCDZ
         enddo

         if (mprdc(M).eq.1) then
            J=RDCbLIS(M)
            DX(J)=DX(J)-RDCDX
            DY(J)=DY(J)-RDCDY
            DZ(J)=DZ(J)-RDCDZ
         endif
         MM=MM+1
      enddo

      RETURN
      END


      SUBROUTINE RDC_FORCE(NATOM,X,Y,Z,AMASS,NRDCS,NRDCE,rdcalis, &
                 rdcblis,RDCRrMAT,RDCRMAT,RDCRMATmp,RDCVECX,RDCVECY, &
                 RDCVECZ,RDCVECmpX,RDCVECmpY,RDCVECmpZ,RDCDIST, &
                 RDCDISTmp,DCON,DRED,REXP,SRDC,DELL,DELU,LHAR,KHAR, &
                 KASY,EXPO,MF,MFT,EV,AN,CT,WI,W55,SMAT,V,VWI,UW,WIUT, &
                 QFIXA,QFIXB,QSRDC,DX,DY,DZ,ERDC,MPRDC)

!------------------------------------------------------------------------
   use number
   use consta
   use stream
   use chm_types
#if KEY_ENSEMBLE==1
   use ensemble
   use mpi
   use memory
#endif
#if KEY_PARALLEL==1
   use parallel
#endif
   implicit none

      integer  natom
      real(chm_real)   X(*),Y(*),Z(*),AMASS(*),EV(3)
      real(chm_real)   DX(*),DY(*),DZ(*),ERDC
      integer  NRDCS,NRDCE,EXPO(*)
      integer  rdcalis(*),rdcblis(*),mprdc(*)
      real(chm_real)   RDCVECX(*),RDCVECY(*),RDCVECZ(*),RDCDIST(*)
      real(chm_real)   MF(3,3),MFT(3,3)
      real(chm_real)   RDCVECmpX(*),RDCVECmpY(*),RDCVECmpZ(*),RDCDISTmp(*)
      real(chm_real)   DCON(*),DRED(NRDCE-NRDCS+1)
      real(chm_real)   KHAR(*),KASY(*),DELL(*),DELU(*),LHAR(*),REXP(*),SRDC(*)
      real(chm_real)   RDCRrMAT(*),CT(*)
      real(chm_real)   RDCRMAT(*),RDCRMATmp(*)
      real(chm_real)   WI(5,5),W55(5,5),SMAT((NRDCE-NRDCS+1)*5),V(5,5),AN(3,3)
      real(chm_real)   VWI(5,5),UW((NRDCE-NRDCS+1)*5),WIUT(5*(NRDCE-NRDCS+1))
      logical  QFIXA,QFIXB,QSRDC
! local
      integer  I,J,K,KK,L,N,NN,M,MM,IN,IN1
! energy
      real(chm_real)   rdcrmin,rdcrmax,rdchmin,rdchmax
      real(chm_real)   kharm,kasym,lharm
      integer  NRDC,expon
      real(chm_real)   C(3,3),RrT(3,3),AC(3,3)
      real(chm_real)   RT(3,3),RTmp(3,3)
      real(chm_real)   softA,softBL(NRDCE-NRDCS+1),softBU(NRDCE-NRDCS+1)
! force
      real(chm_real)   DCAL(NRDCE-NRDCS+1)
      real(chm_real)   XCM,YCM,ZCM,AMS,TMS
      real(chm_real)   DXMF(3,3),DYMF(3,3),DZMF(3,3)
      real(chm_real)   DXMFT(3,3),DYMFT(3,3),DZMFT(3,3)
      real(chm_real)   RX(3,3),RY(3,3),RZ(3,3)
      real(chm_real)   RXmp(3,3),RYmp(3,3),RZmp(3,3)
      real(chm_real)   RDCDX,RDCDY,RDCDZ
      real(chm_real)   numer,numer1,FTM1,FTM2,FTM3,FTMA
      real(chm_real)   SX(1,5),SY(1,5),SZ(1,5)
      real(chm_real)   FTMP1(3,3),FTMP2(3,3)
      real(chm_real)   F2X(3,3),F6(3,3),F9X(3,3),F10(3,3)
      real(chm_real)   F2Y(3,3),F9Y(3,3),F2Z(3,3),F9Z(3,3)
      real(chm_real)   AX(3,3),AY(3,3),AZ(3,3)
      integer  mprdctmp

      real(chm_real)   KRDCVECX,KRDCVECY,KRDCVECZ,KRDCDIST
      real(chm_real)   KRDCVECmpX,KRDCVECmpY,KRDCVECmpZ
      real(chm_real)   KRDCDISTmp,KDCAL,KDCON
#if KEY_ENSEMBLE==1
      real(chm_real),allocatable,dimension(:) :: dcalbuf
      !real(chm_real)   dcalbuf((nrdce-nrdcs+1)*nensem)
      real(chm_real) ave
      integer ierror
#endif

!      ERDC=0.0D0
      KK=1
      NRDC=NRDCE-NRDCS+1

      DO K=NRDCS,NRDCE
         IN=(KK-1)*6
         C(1,1)=CT(IN+1)         !MFT.R.r**-5.MF
         C(1,2)=CT(IN+2)
         C(1,3)=CT(IN+3)
         C(2,1)=CT(IN+2)
         C(2,2)=CT(IN+4)
         C(2,3)=CT(IN+5)
         C(3,1)=CT(IN+3)
         C(3,2)=CT(IN+5)
         C(3,3)=CT(IN+6)

         CALL MULNXN(AC,AN,C,3) !AN.MFT.R.r**-5.MF

         Dcal(KK)=DCON(KK)*(AC(1,1)+AC(2,2)+AC(3,3))

#if KEY_ENSEMBLE==1
         KK=KK+1
      ENDDO

      if (mynod == 0) then
         call chmalloc('rdc.src','DCALBUF','DCALBUF',(nrdce-nrdcs+1)*nensem,crl=dcalbuf)
         call mpi_barrier(comm_master,ierror)
         call mpi_allgather( &
            dcal, nrdc, mpi_double_precision, &
            dcalbuf, nrdc, mpi_double_precision, &
            comm_master,ierror)

         do n=1,nrdc
            ave = 0.0
            do m=1,nensem
               ave = ave + dcalbuf((m-1)*nrdc+n)
            enddo
            dcal(n) = ave/real(nensem)
         enddo
         call chmdealloc('rdc.src','DCALBUF','DCALBUF',(nrdce-nrdcs+1)*nensem,crl=dcalbuf)
      endif
      !CALL ENSAVE(DCAL,DCALBUF,NRDC)

      KK=1
      NRDC=NRDCE-NRDCS+1
      DO K=NRDCS,NRDCE
#endif


!         write(90,'(a4,2i4,3f12.6,f13.3)'),'DCAL',K,MPRDC(k),REXP(K),
!     &        dcal(k),REXP(K)-dcal(k),dcon(k)

         kharm=KHAR(k)
         kasym=KASY(k)
         lharm=LHAR(k)
         expon=EXPO(k)

         if (QSRDC) then
            kdcal=dcal(Kk)*SRDC(K)
            if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K)))then
               rdcrmin=REXP(k)*SRDC(K)
               rdcrmax=REXP(k)*SRDC(K)
            else
               rdcrmin=DELL(k)*SRDC(K)
               rdcrmax=DELU(k)*SRDC(K)
            endif
         else
            kdcal=dcal(kK)
            if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K)))then
               rdcrmin=REXP(k)
               rdcrmax=REXP(k)
            else
               rdcrmin=DELL(k)
               rdcrmax=DELU(k)
            endif
         endif

!         print*,'scale', SRDC(K)
         rdchmin=rdcrmin-lharm
         rdchmax=rdcrmax+lharm
!         print*,'check',rdcrmin,rexp(k),rdcrmax

! Equilibrium
         IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
            ERDC=ERDC+zero
!            print*,'EQUI'
! Harmonic (lower bound)
         ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin) then
            ERDC=ERDC+(kharm*(kdcal-rdcrmin)**TWO)
!            print*,'HARL'
! Harmonic (upper bound)
         ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.le.rdchmax) then
!            print*,'HARU'
               ERDC=ERDC+(kharm*(kdcal-rdcrmax)**TWO)
! soft asympotote (lower bound)
         ELSEIf (Kdcal.lt.rdchmin) THEN
            softBL(KK)=(TWO*kharm*lharm-kasym)/ &
                      (expon*(-lharm)**(-expon-1))
            softA=kharm*lharm*lharm- &
                  softBL(KK)*(-lharm)**(-expon)- &
                  kasym*lharm
            ERDC=ERDC+ &
                 SOFTA+ &
                 SOFTBL(KK)*(Kdcal-rdcrmin)**(-expon)- &
                 kasym*(KDCAL-rdcrmin)
!            print*,'softL'
! soft asympotote (upper bound)
         ELSEIF (Kdcal.gt.rdchmax) then
            softBU(KK)=(-TWO*kharm*lharm+kasym)/ &
                      (expon*lharm**(-expon-1))
            softA=kharm*lharm*lharm- &
                  softBU(KK)*lharm**(-expon)- &
                  kasym*lharm
            ERDC=ERDC+ &
                 SOFTA+ &
                 SOFTBU(KK)*(KDCAL-rdcrmax)**(-expon)+ &
                 kasym*(KDCAL-rdcrmax)
!            print*,'softU'
         ENDIF
         KK=KK+1
      ENDDO
!      print*,'egy=  ', ERDC


      N=1
      KK=1
      DO K=NRDCS,NRDCE
         IF (mpRDC(K).EQ.1) MPRDCtmp=mpRDC(K)
         IF (mpRDC(K).EQ.2) MPRDCtmp=3
         DO NN=1,MPRDCtmp
            IN=(KK-1)*6
            IF(NN.eq.1) I=RDCaLIS(K)
            IF(NN.eq.2) I=RDCbLIS(K)
            IF(NN.eq.3) I=RDCbLIS(K)+1

            RDCDX=ZERO
            RDCDY=ZERO
            RDCDZ=ZERO

            krdcvecx=rdcvecx(KK)
            krdcvecy=rdcvecy(KK)
            krdcvecz=rdcvecz(KK)
            krdcdist=rdcdist(KK)

            RrT(1,1)=RDCRrMAT(IN+1)
            RrT(1,2)=RDCRrMAT(IN+2)
            RrT(1,3)=RDCRrMAT(IN+3)
            RrT(2,1)=RDCRrMAT(IN+2)
            RrT(2,2)=RDCRrMAT(IN+4)
            RrT(2,3)=RDCRrMAT(IN+5)
            RrT(3,1)=RDCRrMAT(IN+3)
            RrT(3,2)=RDCRrMAT(IN+5)
            RrT(3,3)=RDCRrMAT(IN+6)

            RT(1,1)=RDCRMAT(IN+1)
            RT(1,2)=RDCRMAT(IN+2)
            RT(1,3)=RDCRMAT(IN+3)
            RT(2,1)=RDCRMAT(IN+2)
            RT(2,2)=RDCRMAT(IN+4)
            RT(2,3)=RDCRMAT(IN+5)
            RT(3,1)=RDCRMAT(IN+3)
            RT(3,2)=RDCRMAT(IN+5)
            RT(3,3)=RDCRMAT(IN+6)

            numer=KRDCDIST**(TWO/FIVE)

! Derivative of Rmatrix

            RX(1,1)=TWO*KRDCVECX
            RX(1,2)=KRDCVECY
            RX(1,3)=KRDCVECZ
            RX(2,1)=KRDCVECY
            RX(2,2)=ZERO
            RX(2,3)=ZERO
            RX(3,1)=KRDCVECZ
            RX(3,2)=ZERO
            RX(3,3)=ZERO

            RY(1,1)=ZERO
            RY(1,2)=KRDCVECX
            RY(1,3)=ZERO
            RY(2,1)=KRDCVECX
            RY(2,2)=TWO*KRDCVECY
            RY(2,3)=KRDCVECZ
            RY(3,1)=ZERO
            RY(3,2)=KRDCVECZ
            RY(3,3)=ZERO

            RZ(1,1)=ZERO
            RZ(1,2)=ZERO
            RZ(1,3)=KRDCVECX
            RZ(2,1)=ZERO
            RZ(2,2)=ZERO
            RZ(2,3)=KRDCVECY
            RZ(3,1)=KRDCVECX
            RZ(3,2)=KRDCVECY
            RZ(3,3)=TWO*KRDCVECZ

            ftm1=FIVE*KRDCVECX*KRDCDIST*numer
            ftm2=FIVE*KRDCVECY*KRDCDIST*numer
            ftm3=FIVE*KRDCVECZ*KRDCDIST*numer

            DO L=1,3
               DO J=1,3
                  RX(L,J)=RX(L,J)*KRDCDIST-RT(L,J)*FTM1
                  RY(L,J)=RY(L,J)*KRDCDIST-RT(L,J)*FTM2
                  RZ(L,J)=RZ(L,J)*KRDCDIST-RT(L,J)*FTM3
               enddo
            enddo

            if(NN.eq.2) then
!: only for methylene group-1st hydrogen
               DO L=1,3
                  DO J=1,3
                     RX(L,J)=-RX(L,J)
                     RY(L,J)=-RY(L,J)
                     RZ(L,J)=-RZ(L,J)
                  enddo
               enddo
            endif

            IF(MPRDCtmp.eq.3.and.(NN.eq.1.or.NN.eq.3))then

               IN1=(N-1)*6
               krdcvecmpx=rdcvecmpx(N)
               krdcvecmpy=rdcvecmpy(N)
               krdcvecmpz=rdcvecmpz(N)
               krdcdistmp=rdcdistmp(N)

!: only for methylene groups
               RTmp(1,1)=RDCRMATmp(IN1+1)
               RTmp(1,2)=RDCRMATmp(IN1+2)
               RTmp(1,3)=RDCRMATmp(IN1+3)
               RTmp(2,1)=RDCRMATmp(IN1+2)
               RTmp(2,2)=RDCRMATmp(IN1+4)
               RTmp(2,3)=RDCRMATmp(IN1+5)
               RTmp(3,1)=RDCRMATmp(IN1+3)
               RTmp(3,2)=RDCRMATmp(IN1+5)
               RTmp(3,3)=RDCRMATmp(IN1+6)

               numer1=KRDCDISTmp**(TWO/FIVE)

               RXmp(1,1)=TWO*KRDCVECmpX
               RXmp(1,2)=KRDCVECmpY
               RXmp(1,3)=KRDCVECmpZ
               RXmp(2,1)=KRDCVECmpY
               RXmp(2,2)=ZERO
               RXmp(2,3)=ZERO
               RXmp(3,1)=KRDCVECmpZ
               RXmp(3,2)=ZERO
               RXmp(3,3)=ZERO

               RYmp(1,1)=ZERO
               RYmp(1,2)=KRDCVECmpX
               RYmp(1,3)=ZERO
               RYmp(2,1)=KRDCVECmpX
               RYmp(2,2)=TWO*KRDCVECmpY
               RYmp(2,3)=KRDCVECmpZ
               RYmp(3,1)=ZERO
               RYmp(3,2)=KRDCVECmpZ
               RYmp(3,3)=ZERO

               RZmp(1,1)=ZERO
               RZmp(1,2)=ZERO
               RZmp(1,3)=KRDCVECmpX
               RZmp(2,1)=ZERO
               RZmp(2,2)=ZERO
               RZmp(2,3)=KRDCVECmpY
               RZmp(3,1)=KRDCVECmpX
               RZmp(3,2)=KRDCVECmpY
               RZmp(3,3)=TWO*KRDCVECmpZ

               ftm1=FIVE*KRDCVECmpX*KRDCDISTmp*numer1
               ftm2=FIVE*KRDCVECmpY*KRDCDISTmp*numer1
               ftm3=FIVE*KRDCVECmpZ*KRDCDISTmp*numer1

               DO L=1,3
                  DO J=1,3
                     RXmp(L,J)=RXmp(L,J)*KRDCDISTmp-FTM1*RTmp(L,J)
                     RYmp(L,J)=RYmp(L,J)*KRDCDISTmp-FTM2*RTmp(L,J)
                     RZmp(L,J)=RZmp(L,J)*KRDCDISTmp-FTM3*RTmp(L,J)

                     if (NN.eq.3) then
!: only for methylene group-2nd hydrogen
                        RX(L,J)=-RXmp(L,J)
                        RY(L,J)=-RYmp(L,J)
                        RZ(L,J)=-RZmp(L,J)
                     else
                        RX(L,J)=RX(L,J)+RXmp(L,J)
                        RY(L,J)=RY(L,J)+RYmp(L,J)
                        RZ(L,J)=RZ(L,J)+RZmp(L,J)
                     endif
                  enddo
               enddo
               if (NN.eq.3)  N=N+1
            ENDIF

! Derivative w.r.t X
            CALL MULNXN(FTMP1,MFT,RX,3)
            CALL MULNXN(F2X,FTMP1,MF,3)
            CALL MULNXN(F6,AN,F2X,3)
            FTM1=F6(1,1)+F6(2,2)+F6(3,3)

! Derivative w.r.t Y
            CALL MULNXN(FTMP1,MFT,RY,3)
            CALL MULNXN(F2Y,FTMP1,MF,3)
            CALL MULNXN(F6,AN,F2Y,3)
            FTM2=F6(1,1)+F6(2,2)+F6(3,3)

! Derivative w.r.t Z
            CALL MULNXN(FTMP1,MFT,RZ,3)
            CALL MULNXN(F2Z,FTMP1,MF,3)
            CALL MULNXN(F6,AN,F2Z,3)
            FTM3=F6(1,1)+F6(2,2)+F6(3,3)


            kharm=KHAR(K)
            kasym=KASY(K)
            lharm=LHAR(K)
            expon=EXPO(K)
            if (QSRDC) then
               kdcal=dcal(kK)*SRDC(K)
               kdcon=dcon(kK)*SRDC(k)
               if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
                  rdcrmin=REXP(k)*SRDC(K)
                  rdcrmax=REXP(k)*SRDC(K)
               else
                  rdcrmin=DELL(k)*SRDC(K)
                  rdcrmax=DELU(k)*SRDC(K)
               endif
            else
               kdcal=dcal(kK)
               kdcon=dcon(kK)
               if((rexp(k).eq.DELL(K)).and.(rexp(k).eq.DELU(K))) then
                  rdcrmin=REXP(k)
                  rdcrmax=REXP(k)
               else
                  rdcrmin=DELL(k)
                  rdcrmax=DELU(k)
               endif
            endif
            rdchmin=rdcrmin-lharm
            rdchmax=rdcrmax+lharm
! Equilibrium
            IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
               RDCDX=RDCDX
               RDCDY=RDCDY
               RDCDZ=RDCDZ
!               print*,'EQUI'
! Harmonic (lower bound)
            ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin) then
               FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMIN)
               RDCDX=RDCDX+(FTMA*FTM1)
               RDCDY=RDCDY+(FTMA*FTM2)
               RDCDZ=RDCDZ+(FTMA*FTM3)
!     print*,'HARL'
! Harmonic (upper bound)
            ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.le.rdchmax) then
!               print*,'HARU'
               FTMA=TWO*KHARM*kdcon*(KDCAL-RDCRMAX)
               RDCDX=RDCDX+(FTMA*FTM1)
               RDCDY=RDCDY+(FTMA*FTM2)
               RDCDZ=RDCDZ+(FTMA*FTM3)
! soft asympotote (lower bound)
            ELSEIf (Kdcal.lt.rdchmin) THEN
               FTMA=-softBL(KK)*expon*(KDCAL-RDCRMIN)**(-expon-1) &
                    *KDCON-KASYM*KDCON
               RDCDX=RDCDX+(FTMA*FTM1)
               RDCDY=RDCDY+(FTMA*FTM2)
               RDCDZ=RDCDZ+(FTMA*FTM3)
!               print*,'softL'
! soft asympotote (upper bound)
            ELSEIF (Kdcal.gt.rdchmax) then
               FTMA=-softBU(KK)*expon*(KDCAL-rdcrmax)**(-expon-1) &
                    *KDCON+kasym*KDCON
               RDCDX=RDCDX+(FTMA*FTM1)
               RDCDY=RDCDY+(FTMA*FTM2)
               RDCDZ=RDCDZ+(FTMA*FTM3)
            ENDIF

! Derivative of S
            IF (.not. QFIXA) then
               DO L=1,3
                  DO J=1,3
                     F9X(L,J)=F2X(L,J)
                     F9Y(L,J)=F2Y(L,J)
                     F9Z(L,J)=F2Z(L,J)
                  enddo
               enddo
               SX(1,1) = F9X(1,1)-F9X(3,3)
               SX(1,2) = TWO*F9X(1,2)
               SX(1,3) = TWO*F9X(1,3)
               SX(1,4) = F9X(2,2)-F9X(3,3)
               SX(1,5) = TWO*F9X(2,3)

               SY(1,1) = F9Y(1,1)-F9Y(3,3)
               SY(1,2) = TWO*F9Y(1,2)
               SY(1,3) = TWO*F9Y(1,3)
               SY(1,4) = F9Y(2,2)-F9Y(3,3)
               SY(1,5) = TWO*F9Y(2,3)

               SZ(1,1) = F9Z(1,1)-F9Z(3,3)
               SZ(1,2) = TWO*F9Z(1,2)
               SZ(1,3) = TWO*F9Z(1,3)
               SZ(1,4) = F9Z(2,2)-F9Z(3,3)
               SZ(1,5) = TWO*F9Z(2,3)

!            do L=1,nrdc
!               write(*,'(2i4,5f12.8)') mprdctmp,L, (SX(L,J),J=1,5)
!            enddo

               CALL RDC_DERAT(NRDC,WI,W55,SMAT,V,VWI,UW,WIUT,AN,DRED,SX,SY,SZ,KK,AX,AY,AZ)
               MM=1
               DO M=NRDCS,NRDCE
                  IN=(MM-1)*6
                  C(1,1)=CT(IN+1) !MFT.R.r**-5..MF
                  C(1,2)=CT(IN+2)
                  C(1,3)=CT(IN+3)
                  C(2,1)=CT(IN+2)
                  C(2,2)=CT(IN+4)
                  C(2,3)=CT(IN+5)
                  C(3,1)=CT(IN+3)
                  C(3,2)=CT(IN+5)
                  C(3,3)=CT(IN+6)

                  CALL MULNXN(F10,AX,C,3)
                  FTM1=F10(1,1)+F10(2,2)+F10(3,3)
                  CALL MULNXN(F10,AY,C,3)
                  FTM2=F10(1,1)+F10(2,2)+F10(3,3)
                  CALL MULNXN(F10,AZ,C,3)
                  FTM3=F10(1,1)+F10(2,2)+F10(3,3)

                  kharm=KHAR(M)
                  kasym=KASY(M)
                  lharm=LHAR(M)
                  expon=EXPO(M)

                  if (QSRDC) then
                     kdcal=dcal(Mm)*SRDC(M)
                     kdcon=dcon(Mm)*SRDC(M)
                     if((rexp(M).eq.DELL(M)).and.(rexp(M).eq.DELU(M)))then
                        rdcrmin=REXP(M)*SRDC(M)
                        rdcrmax=REXP(M)*SRDC(M)
                     else
                        rdcrmin=DELL(M)*SRDC(M)
                        rdcrmax=DELU(M)*SRDC(M)
                     endif
                  else
                     kdcal=dcal(MM)
                     kdcon=dcon(MM)
                     if((rexp(M).eq.DELL(M)).and.(rexp(M).eq.DELU(M)))then
                        rdcrmin=REXP(M)
                        rdcrmax=REXP(M)
                     else
                        rdcrmin=DELL(M)
                        rdcrmax=DELU(M)
                     endif
                  endif

                  rdchmin=rdcrmin-lharm
                  rdchmax=rdcrmax+lharm

! Equilibrium
                  IF (Kdcal.ge.rdcrmin.and.Kdcal.le.rdcrmax) then
                     RDCDX=RDCDX
                     RDCDY=RDCDY
                     RDCDZ=RDCDZ
!                 print*,'EQUI'
! Harmonic (lower bound)
                  ELSEIF (Kdcal.lt.rdcrmin.and.Kdcal.ge.rdchmin)then
                     FTMA=TWO*KHARM*KDCON*(KDCAL-RDCRMIN)
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
!                 print*,'HARL'
! Harmonic (upper bound)
                  ELSEIF (Kdcal.gt.rdcrmax.and.Kdcal.le.rdchmax)then
!                 print*,'HARU'
                     FTMA=TWO*KHARM*KDCON*(KDCAL-RDCRMAX)
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
! soft asympotote (lower bound)
                  ELSEIf (Kdcal.lt.rdchmin) THEN
                     FTMA=-softBL(MM)*expon*(KDCAL-RDCRMIN)**(-expon-1)* &
                          KDCON-KASYM*KDCON
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
!                 print*,'softL'
! soft asympotote (upper bound)
                  ELSEIF (Kdcal.gt.rdchmax) then
                     FTMA=-softBU(MM)*expon*(KDCAL-rdcrmax)**(-expon-1)* &
                          KDCON+kasym*KDCON
                     RDCDX=RDCDX+(FTMA*FTM1)
                     RDCDY=RDCDY+(FTMA*FTM2)
                     RDCDZ=RDCDZ+(FTMA*FTM3)
                  endif
                  MM=MM+1
               enddo
            endif

            DX(I)=DX(I)+RDCDX
            DY(I)=DY(I)+RDCDY
            DZ(I)=DZ(I)+RDCDZ

         enddo

         if (mprdc(K).eq.1) then
            J=RDCbLIS(K)
            DX(J)=DX(J)-RDCDX
            DY(J)=DY(J)-RDCDY
            DZ(J)=DZ(J)-RDCDZ
         endif
         KK=KK+1
      enddo

      RETURN
      END



      SUBROUTINE RDC_ANAL(NRDCS,NRDCE,RDCALIS,RDCBLIS,DCON,REXP,CT,AN, &
                          URDC,MED)
!------------------------------------------------------------------------
   use number
   use consta
   use stream
   use chm_types
   use chutil,only:atomid
#if KEY_ENSEMBLE==1
   use ensemble
   use mpi
   use memory
#endif
#if KEY_PARALLEL==1
   use parallel
#endif
   implicit none

      INTEGER  NRDCS,NRDCE,MED
      INTEGER  rdcalis(*),rdcblis(*)
      INTEGER  URDC

      real(chm_real)   DCON(*), REXP(*)
      real(chm_real)   CT(*)
      real(chm_real)   AN(3,3)
! local
      integer  NRDC,K,KK,I,J
! energy
      real(chm_real)   C(3,3),AC(3,3),DCAL(NRDCE-NRDCS+1),AT(6),ADI(3,3)
      real(chm_real)   softA,softBL(NRDCE-NRDCS+1),softBU(NRDCE-NRDCS+1)
      real(chm_real)   xa(3), ya(3), za(3)
      real(chm_real)   dpxa, dpya, dpza
      real(chm_real)   xtheta, ytheta, ztheta

      real(chm_real)   SCR(21),EV(3)

      real(chm_real)   kdcal, diff
      real(chm_real)   sdcal,ssdcal
      integer  NP000,NP010,NP025,NP050,NP100,NP150
      integer  NP200,NP250,NP300,NP350,NP400
      integer  IN
!
      CHARACTER*8   SIDI,RIDI,RENI,ACI
      CHARACTER*8   SIDJ,RIDJ,RENJ,ACJ
#if KEY_ENSEMBLE==1
      real(chm_real),allocatable,dimension(:) :: dcalbuf
      !real(chm_real)   dcalbuf((nrdce-nrdcs+1)*nensem)
      real(chm_real) ave
      integer ierror,m,n
#endif
!
      write(URDC,'(a9,i5,5f20.15)') "A.Tensor", MED, AN(1,1), AN(1,2), AN(1,3), AN(2,2),  AN(2,3)

      AT(1) = AN(1,1)
      AT(2) = AN(1,2)
      AT(3) = AN(1,3)
      AT(4) = AN(2,2)
      AT(5) = AN(2,3)
      AT(6) = -AN(1,1)-AN(2,2)
!
!     Diagonalisation of intertia tensor
!

      CALL DIAGQ(3,3,AT,ADI,SCR(4),SCR(7),SCR(10),SCR(13),EV,SCR(16),SCR(19),SCR(1),0)


!      WRITE(*,*) 'Diagonalized Alignment tensor'
!      WRITE (6,106) (ADI(I,1),I=1,3)
! 106  FORMAT(' Principal axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      WRITE (6,107) (ADI(I,2),I=1,3)
! 107  FORMAT(' Secondary axis, X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      WRITE (6,108) (ADI(I,3),I=1,3)
! 108  FORMAT(' Tertiary axis,  X= ',F10.6,3X,'Y= ',F10.6,3X,'Z= ',F10.6)
!      write (6,*)

!      WRITE(*,*) 'Normalized Alignment tensor'
!      CALL NORMALL(ADI(1,3),3)
!      CALL NORMALL(ADI(1,3),3)
!      CALL NORMALL(ADI(1,3),3)
!      WRITE (6,106) (ADI(I,1),I=1,3)
!      WRITE (6,106) (ADI(I,2),I=1,3)
!      WRITE (6,106) (ADI(I,3),I=1,3)

!      xa(1)=1
!      xa(2)=0
!      xa(3)=0


!      ya(1)=0
!      ya(2)=1
!      ya(3)=0


!      za(1)=0
!      za(2)=0
!      za(3)=1

!      call dotpr(xa,xa,3,dpxa)
!      print*,dpxa

      dpxa = ADI(1,1)
      dpya = ADI(2,2)
      dpza = ADI(3,3)

      xtheta= acos(dpxa)*180/pi
      ytheta= acos(dpya)*180/pi
      ztheta= acos(dpza)*180/pi
!      write(*,*) "Angle bet. X-axis and principal axis of
!     &     A.Tensor" ,Xtheta
!      write(*,*) "Angle bet. Y-axis and principal axis of
!     &     A.Tensor" ,Ytheta
!      write(*,*) "Angle bet. Z-axis and principal axis of
!     &     A.Tensor" ,Ztheta


      SDCAL=ZERO
      SSDCAL=ZERO
      NP000=0
      NP010=0
      NP025=0
      NP050=0
      NP100=0
      NP150=0
      NP200=0
      NP250=0
      NP300=0
      NP350=0
      NP400=0
      KK=1
      NRDC=NRDCE-NRDCS+1
!      print*,nrdc,nrdcs,nrdce

      DO K=NRDCS,NRDCE
         IN=(KK-1)*6

         C(1,1)=CT(IN+1)         !MFT.R.r**-5.MF
         C(1,2)=CT(IN+2)
         C(1,3)=CT(IN+3)
         C(2,1)=CT(IN+2)
         C(2,2)=CT(IN+4)
         C(2,3)=CT(IN+5)
         C(3,1)=CT(IN+3)
         C(3,2)=CT(IN+5)
         C(3,3)=CT(IN+6)

         CALL MULNXN(AC,AN,C,3) !AN.MFT.R.MF

         Dcal(KK)=DCON(KK)*(AC(1,1)+AC(2,2)+AC(3,3))  !/((RDCDIST(K)**5))

#if KEY_ENSEMBLE==1
         KK=KK+1
      ENDDO

      if (mynod == 0) then
         call chmalloc('rdc.src','DCALBUF','DCALBUF',(nrdce-nrdcs+1)*nensem,crl=dcalbuf)
         call mpi_barrier(comm_master,ierror)
         call mpi_allgather( &
            dcal, nrdc, mpi_double_precision, &
            dcalbuf, nrdc, mpi_double_precision, &
            comm_master,ierror)

         do n=1,NRDC
            ave = 0.0
            do m=1,nensem
               ave = ave + dcalbuf((m-1)*nrdc+n)
            enddo
            dcal(n) = ave/real(nensem)
         enddo
         call chmdealloc('rdc.src','DCALBUF','DCALBUF',(nrdce-nrdcs+1)*nensem,crl=dcalbuf)
      endif
      !CALL ENSAVE(DCAL,DCALBUF,NRDC)

      KK=1
      NRDC=NRDCE-NRDCS+1
      DO K=NRDCS,NRDCE
#endif
         kdcal=dcal(KK)

         DIFF=abs(KDCAL-REXP(K))
         SDCAL=SDCAL+DIFF
         SSDCAL=SSDCAL+DIFF*DIFF

         IF(DIFF.LE.0.10) NP000=NP000+1
         IF(DIFF.GT.0.10.AND.DIFF.LE.0.25) NP010=NP010+1
         IF(DIFF.GT.0.25.AND.DIFF.LE.0.50) NP025=NP025+1
         IF(DIFF.GT.0.50.AND.DIFF.LE.1.00) NP050=NP050+1
         IF(DIFF.GT.1.00.AND.DIFF.LE.1.50) NP100=NP100+1
         IF(DIFF.GT.1.50.AND.DIFF.LE.2.00) NP150=NP150+1
         IF(DIFF.GT.2.00.AND.DIFF.LE.2.50) NP200=NP200+1
         IF(DIFF.GT.2.50.AND.DIFF.LE.3.00) NP250=NP250+1
         IF(DIFF.GT.3.00.AND.DIFF.LE.3.50) NP300=NP300+1
         IF(DIFF.GT.3.50.AND.DIFF.LE.4.00) NP350=NP350+1
         IF(DIFF.GT.4.00) NP400=NP400+1
         KK=KK+1
      ENDDO


!
! statistics
!
 101  FORMAT(6x,a,f11.4)
 102  FORMAT(6x,a,i5)

      WRITE(URDC, "(a)") ""
      WRITE(URDC, "(6x,a)") "Statistics of calculated RDCs"
      WRITE(URDC, "(6x, a)") "-----------------------------"
      WRITE(URDC,101) "Mean deviation from experimental values       =",SDCAL/NRDC
      WRITE(URDC,101) "RMS fluctation from the mean                  =",SSDCAL/NRDC-(SDCAL/NRDC)*(SDCAL/NRDC)
      WRITE(URDC, "(a)") ""
      WRITE(URDC,102) "Number of RDCs with deviation of 0.10 and low  =",NP000
      WRITE(URDC,102) "Number of RDCs with deviation bt 0.10 and 0.25 =",NP010
      WRITE(URDC,102) "Number of RDCs with deviation bt 0.25 and 0.50 =",NP025
      WRITE(URDC,102) "Number of RDCs with deviation bt 0.50 and 1.00 =",NP050
      WRITE(URDC,102) "Number of RDCs with deviation bt 1.00 and 1.50 =",NP100
      WRITE(URDC,102) "Number of RDCs with deviation bt 1.50 and 2.00 =",NP150
      WRITE(URDC,102) "Number of RDCs with deviation bt 2.00 and 2.50 =",NP200
      WRITE(URDC,102) "Number of RDCs with deviation bt 2.50 and 3.00 =",NP250
      WRITE(URDC,102) "Number of RDCs with deviation bt 3.00 and 3.50 =",NP300
      WRITE(URDC,102) "Number of RDCs with deviation bt 3.50 and 4.00 =",NP350
      WRITE(URDC,102) "Number of RDCs with deviation of 4.00 and high =",NP400
      WRITE(URDC, "(a)") ""
!
! individual deviation
!
 103  FORMAT(6X,I4,3X,6(1X,A),3F9.3)
      KK=1
      WRITE(URDC,'(47x,a)') "Exp      Calc     Diff"
      DO K=NRDCS,NRDCE
         I=RDCaLIS(K)
         J=RDCbLIS(K)
         CALL ATOMID(I,SIDI,RIDI,RENI,ACI)
         CALL ATOMID(J,SIDJ,RIDJ,RENJ,ACJ)
         WRITE(URDC,103) KK, &
              SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
              SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
              REXP(K),DCAL(KK),abs(REXP(K)-DCAL(KK))
         KK=KK+1
      ENDDO
!
      RETURN
      END


SUBROUTINE RDC_BCAL(NATOM,X,Y,Z,IATM,JATM,AN,MF,MFT,IGAMMA,JGAMMA, &
           URDC,NRES,RRES,IBASE,TYPE)
!------------------------------------------------------------------------
! works for all kinds of RDCs except methylene groups
   use number
   use consta
   use stream
   use exfunc
   use string
   use chutil
   use chm_types
   implicit none

      INTEGER       NATOM,NRES,RRES
      INTEGER       URDC,IBASE(*)
      CHARACTER*(*) TYPE(*)
      real(chm_real)        X(*),Y(*),Z(*)
      real(chm_real)        AN(3,3),MF(3,3),MFT(3,3)

! Local
      real(chm_real)        igamma,jgamma
      CHARACTER*8   SIDI,RIDI,RENI,ACI,IATM
      CHARACTER*8   SIDJ,RIDJ,RENJ,ACJ,JATM,JATM1
      real(chm_real)        RT(3,3),RTmp(3,3),MR(3,3)
      real(chm_real)        C(3,3),AC(3,3),DCAL(natom),DCON
      integer       RESN,rdcalis(natom),rdcblis(natom)
      real(chm_real)        rdcvecx,rdcvecy,rdcvecz,rdcdist
      real(chm_real)        rdcvecmpx,rdcvecmpy,rdcvecmpz,rdcdistmp
      integer       K,KK,II,JJ,RDCT,I,J
      integer       ATNUM,MPRDC(natom)

      K=1
      do KK=1,natom
         CALL ATOMID(KK,SIDI,RIDI,RENI,ACI)
         if (RENI(1:3) .eq. "GLY" .and. IATM .eq. "CA" .and. JATM .eq. "CB" ) GOTO 500
         if (ACI.eq.IATM) then
            RDCaLIS(K)=KK
            RESN=DECODI(RIDI,8)+RRES
!            PRINT*,IBASE(resn)+1,ibase(resn+1)
            if (RENI(1:3) .eq. "GLY" .and. IATM .eq. "CA" .and. JATM .eq. "HA" ) then
               JATM1="HA1 "
               mprdc(k)=2
            else
               JATM1=JATM
               mprdc(k)=1
            endif
            if (resn.le.nres) ATNUM=MATOM(RESN,JATM1,type,IBASE,1,NRES,.TRUE.)
!            print*,RENI(1:3),"  ", JATM
            RDCbLIS(K)=ATNUM
!            print*, rdcalis(k),rdcblis(k)
            K=K+1
         endif
!       print*, 'here'
 500     continue
      enddo
      rdct=K-1
!      print*,K,rdct


      do K=1,rdct
         II=RDCaLIS(K)
         JJ=RDCbLIS(K)
!         print*,II,JJ,K
         RDCVECX = X(II)-X(JJ)
         RDCVECY = Y(II)-Y(JJ)
         RDCVECZ = Z(II)-Z(JJ)

         RT(1,1)=RDCVECX*RDCVECX
         RT(1,2)=RDCVECX*RDCVECY
         RT(1,3)=RDCVECX*RDCVECZ
         RT(2,1)=RDCVECX*RDCVECY
         RT(2,2)=RDCVECY*RDCVECY
         RT(2,3)=RDCVECY*RDCVECZ
         RT(3,1)=RDCVECX*RDCVECZ
         RT(3,2)=RDCVECY*RDCVECZ
         RT(3,3)=RDCVECZ*RDCVECZ

         RDCDIST=(sqrt(RDCVECX*RDCVECX+RDCVECY*RDCVECY+RDCVECZ*RDCVECZ))

         RDCDIST=RDCDIST**(-1*FIVE)

         DCON=(-igamma*jgamma*(6.626D-11))/(TWO*PI*PI)

         DO I=1,3
            DO J=1,3
               RT(I,J)=RT(I,J)*RDCDIST
            enddo
         enddo

         IF(MPRDC(K).eq.2) then
!: for GLY CA-HA, can be used for Methylene
!            print*,'hi...',MPRDC(K),X(JJ),JJ,JJ+1
            RDCVECmpX = X(II)-X(JJ+1)
            RDCVECmpY = Y(II)-Y(JJ+1)
            RDCVECmpZ = Z(II)-Z(JJ+1)

            RTmp(1,1)=RDCVECmpX*RDCVECmpX
            RTmp(1,2)=RDCVECmpX*RDCVECmpY
            RTmp(1,3)=RDCVECmpX*RDCVECmpZ
            RTmp(2,1)=RDCVECmpX*RDCVECmpY
            RTmp(2,2)=RDCVECmpY*RDCVECmpY
            RTmp(2,3)=RDCVECmpY*RDCVECmpZ
            RTmp(3,1)=RDCVECmpX*RDCVECmpZ
            RTmp(3,2)=RDCVECmpY*RDCVECmpZ
            RTmp(3,3)=RDCVECmpZ*RDCVECmpZ

            RDCDISTmp=(sqrt(RDCVECmpX*RDCVECmpX+ &
                         RDCVECmpY*RDCVECmpY+ &
                         RDCVECmpZ*RDCVECmpZ))

            RDCDISTmp=RDCDISTmp**(-1*FIVE)
!            print*,'hi..1',M,rdcdistmp(m)

            DO I=1,3
               DO J=1,3
                  RTmp(I,J)=RTmp(I,J)*RDCDISTmp
                  RT(I,J)  =RTmp(I,J)+RT(I,J)
               enddo
            enddo
!            DO I=1,3
!               print*,(RTmp(I,J),J=1,3)
!            enddo
!            write(*,*)
!            print*,nrdcmp,M
         endif

         CALL MULNXN(MR,MFT,RT,3) !MFT.RT -asymmetry
         CALL MULNXN(C,MR,MF,3) !MFT.RT.MF -symmetry
         CALL MULNXN(AC,AN,C,3) !AN.MFT.R.MF

         Dcal(K)=DCON*(AC(1,1)+AC(2,2)+AC(3,3)) !/((RDCDIST(K)**5))
      enddo

 101  FORMAT(6x,a)
      WRITE(URDC,101)
      WRITE(URDC,101) "BACK CALCULATION OF RDCs"
      WRITE(URDC,101) "------------------------"
 103  FORMAT(6X,I4,3X,6(1X,A),F9.3)

      WRITE(URDC,'(41x,a)') "      Calc     "

      DO KK=1,RDCT
         II=RDCaLIS(KK)
         JJ=RDCbLIS(KK)

         CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
         CALL ATOMID(JJ,SIDJ,RIDJ,RENJ,ACJ)
         WRITE(URDC,103) KK, &
              SIDI(1:idleng),RIDI(1:idleng),ACI(1:idleng), &
              SIDJ(1:idleng),RIDJ(1:idleng),ACJ(1:idleng), &
              DCAL(KK)
      ENDDO
      WRITE(URDC,101)
      RETURN
      END



      SUBROUTINE RDC_DRMF(X,Y,Z,AMASS,TMS,GBEV,L,XCM,YCM,ZCM,O,DXO,DYO,DZO)
!------------------------------------------------------------------------

!  force contributions due to rotation of integration points
!

   use number
   use consta
   use chm_types
   implicit none

      real(chm_real)  X(*),Y(*),Z(*)
      real(chm_real)  O(3,3),GBEV(3),AMASS(*)
      real(chm_real)  XCM,YCM,ZCM,TMS,DCM,AMS

! local
      integer I,L,J,K
      real(chm_real)  dxM(6),dyM(6),dzM(6),drP(3,3)
      real(chm_real)  OT(3,3),OMO(3,3),dxO(3,3),dyO(3,3),dzO(3,3),OM(3,3)
      real(chm_real)  alpha,beta,gamma
      real(chm_real)  dxXX,dxXY,dxXZ,dxYY,dxYZ,dxZZ
      real(chm_real)  dyXX,dyXY,dyXZ,dyYY,dyYZ,dyZZ
      real(chm_real)  dzXX,dzXY,dzXZ,dzYY,dzYZ,dzZZ
      real(chm_real)  xa,ya,za




! tranpose of principal axes matrix
      call TRANSPS(OT,O,3,3)

      xa=x(L)
      ya=y(L)
      za=z(L)

!      write(*,*) xcm,ycm,zcm
!      print*, xa,ya,za

! derivatives of inertia tensor M
!         XX  = AMS*(Xa-XCM)*(Xa-XCM)
!         XY  = AMS*(Xa-XCM)*(Ya-YCM)
!         XZ  = AMS*(Xa-XCM)*(Za-ZCM)
!         YY  = AMS*(Ya-YCM)*(Ya-YCM)
!         YZ  = AMS*(Ya-YCM)*(Za-ZCM)
!         ZZ  = AMS*(Za-ZCM)*(Za-ZCM)

      AMS = AMASS(L)
      dCM = AMS/TMS

!      print*,"COM", AMS,TMS
      dxXX = AMS*(ONE-dCM)*(Xa-XCM)*TWO
      dxXY = AMS*(ONE-dCM)*(Ya-YCM)
      dxXZ = AMS*(ONE-dCM)*(Za-ZCM)
      dxYY = ZERO
      dxYZ = ZERO
      dxZZ = ZERO

      dyXX = ZERO
      dyXY = AMS*(ONE-dCM)*(Xa-XCM)
      dyXZ = ZERO
      dyYY = AMS*(ONE-dCM)*(Ya-YCM)*TWO
      dyYZ = AMS*(ONE-dCM)*(Za-ZCM)
      dyZZ = ZERO

      dzXX = ZERO
      dzXY = ZERO
      dzXZ = AMS*(ONE-dCM)*(Xa-XCM)
      dzYY = ZERO
      dzYZ = AMS*(ONE-dCM)*(Ya-YCM)
      dzZZ = AMS*(ONE-dCM)*(Za-ZCM)*TWO

      dxM(1) =  dxYY + dxZZ
      dxM(2) = -dxXY
      dxM(3) = -dxXZ
      dxM(4) =  dxXX + dxZZ
      dxM(5) = -dxYZ
      dxM(6) =  dxXX + dxYY

      dyM(1) =  dyYY + dyZZ
      dyM(2) = -dyXY
      dyM(3) = -dyXZ
      dyM(4) =  dyXX + dyZZ
      dyM(5) = -dyYZ
      dyM(6) =  dyXX + dyYY

      dzM(1) =  dzYY + dzZZ
      dzM(2) = -dzXY
      dzM(3) = -dzXZ
      dzM(4) =  dzXX + dzZZ
      dzM(5) = -dzYZ
      dzM(6) =  dzXX + dzYY

! calculate d(xyz)O from O(T) x d(xyz)M x O through Eqs. (16), (18), and (19)
      call MULNXNFU(OM,OT,dxM,3)
      call MULNXN(OMO,OM,O,3)
      alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
      beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
      gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
      drP(1,1)= zero
      drP(1,2)= gamma
      drP(1,3)=-beta
      drP(2,1)=-gamma
      drP(2,2)= zero
      drP(2,3)= alpha
      drP(3,1)= beta
      drP(3,2)=-alpha
      drP(3,3)= zero
      call MULNXN(dxO,O,drP,3)

!         call MULNXN(OM,O,drP,3)
!         call MULNXN(dxO,OM,OT,3)

      call MULNXNFU(OM,OT,dyM,3)
      call MULNXN(OMO,OM,O,3)
      alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
      beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
      gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
      drP(1,1)= zero
      drP(1,2)= gamma
      drP(1,3)=-beta
      drP(2,1)=-gamma
      drP(2,2)= zero
      drP(2,3)= alpha
      drP(3,1)= beta
      drP(3,2)=-alpha
      drP(3,3)= zero
      call MULNXN(dyO,O,drP,3)

!         call MULNXN(OM,O,drP,3)
!         call MULNXN(dyO,OM,OT,3)

      call MULNXNFU(OM,OT,dzM,3)
      call MULNXN(OMO,OM,O,3)
      alpha=-OMO(2,3)/(GBEV(2)-GBEV(3))
      beta =-OMO(1,3)/(GBEV(3)-GBEV(1))
      gamma=-OMO(1,2)/(GBEV(1)-GBEV(2))
      drP(1,1)= zero
      drP(1,2)= gamma
      drP(1,3)=-beta
      drP(2,1)=-gamma
      drP(2,2)= zero
      drP(2,3)= alpha
      drP(3,1)= beta
      drP(3,2)=-alpha
      drP(3,3)= zero

      call MULNXN(dzO,O,drP,3)

!         call MULNXN(OM,O,drP,3)
!         call MULNXN(dzO,OM,OT,3)

      return
      end


!-------------------------------------------------------------------------
      SUBROUTINE RDC_DERAT_ALL(NRDC,WI,W55,U1,V,VWI,UW1,WIUT1,AN,DRED,SX,SY,SZ,AX,AY,AZ)

   use number
   use consta
   use chm_types
   implicit none

      integer  NRDC
      real(chm_real)   SX(NRDC,5),SY(NRDC,5),SZ(NRDC,5)
      real(chm_real)   WI(5,5),W55(5,5),U1(NRDC*5),V(5,5),AN(3,3),DRED(NRDC)   !U1 is just smat
      real(chm_real)   VWI(5,5),UW1(NRDC*5),WIUT1(5*NRDC)
      real(chm_real)   AX(3,3),AY(3,3),AZ(3,3)
! local
      integer  L,J,K
! a.tensor
      real(chm_real)   UW(NRDC,5),WIUT(5,NRDC)     !to convert into to 2Dimen. array
      real(chm_real)   SFTM1(5,5),SFTM1X(5,5),SFTM1Y(5,5),SFTM1Z(5,5)
      real(chm_real)   SFTM2X(5,5),SFTM2Y(5,5),SFTM2Z(5,5)
      real(chm_real)   denom1
      real(chm_real)   U(NRDC,5),VT(5,5)  !convert smat into 2Dim. array
      real(chm_real)   UT(5,nrdc)   ! transpose of UT
      real(chm_real)   WX(5,5),WY(5,5),WZ(5,5),VTX(5,5),VTY(5,5),VTZ(5,5)
      real(chm_real)   UX(NRDC,5),UY(NRDC,5),UZ(NRDC,5)
      real(chm_real)   SXVWI(NRDC,5),SYVWI(NRDC,5),SZVWI(NRDC,5)
      real(chm_real)   UWXWI(NRDC,5),UWYWI(NRDC,5),UWZWI(NRDC,5)
      real(chm_real)   UWVTXVWI(NRDC,5),UWVTYVWI(NRDC,5),UWVTZVWI(NRDC,5)
      real(chm_real)   VX(5,5),VY(5,5),VZ(5,5),UTX(5,NRDC),UTY(5,NRDC)
      real(chm_real)   UTZ(5,NRDC),WIX(5,5),WIY(5,5),WIZ(5,5)
      real(chm_real)   VXWIUT(5,NRDC),VYWIUT(5,NRDC),VZWIUT(5,NRDC)
      real(chm_real)   VWIXUT(5,NRDC),VWIYUT(5,NRDC),VWIZUT(5,NRDC)
      real(chm_real)   VWIUTX(5,NRDC),VWIUTY(5,NRDC),VWIUTZ(5,NRDC)
      real(chm_real)   SIDX(5,NRDC),SIDY(5,NRDC),SIDZ(5,NRDC)
      real(chm_real)   AD(5,1)


!     do L=1,NRDC
!     write(*,'(a5,5f15.10))')"SX-A ",(SX(L,J),J=1,5)
!     enddo

!     write(*,*)
!     write(*,*)

!     do L=1,5
!     write(*,'(10f15.9)')(SI(L,J),J=1,NRDC)
!     enddo

! DERIVATIVE OF W

      DO L=1,5
         DO J=1,5
            WX(L,J)=ZERO        !Derivative of W
            WY(L,J)=ZERO        !Derivative of W
            WZ(L,J)=ZERO        !Derivative of W
         enddo
      enddo

! Transpose of U & converting UW1 and WIUT1 into 2 Dimension array
      K=1
      DO L=1,NRDC
         DO J=1,5
            U(L,J)=U1(K)
            UT(J,L)=U(L,J)
            UW(L,J)=UW1(K)
            WIUT(J,L)=WIUT1(K)
!            print*,ut(L,J)
            K=K+1
         enddo
      enddo

!      stop
      CALL MULMXN(SFTM1,UT,5,NRDC,SX,NRDC,5)
      CALL MULNXN(SFTM2X,SFTM1,V,5)

      CALL MULMXN(SFTM1,UT,5,NRDC,SY,NRDC,5)
      CALL MULNXN(SFTM2Y,SFTM1,V,5)

      CALL MULMXN(SFTM1,UT,5,NRDC,SZ,NRDC,5)
      CALL MULNXN(SFTM2Z,SFTM1,V,5)


      DO L=1,5
         WX(L,L)=SFTM2X(L,L)
         WY(L,L)=SFTM2Y(L,L)
         WZ(L,L)=SFTM2Z(L,L)
      enddo

!      DO L=1,5
!      write(*,'(a3,5f15.12))')"WX ",(WX(L,J),J=1,5)
!      write(*,'(a3,5f15.12))')"WI ",(WI(L,J),J=1,5)
!      write(*,'(a3,5f15.12))')"W ",(W55(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)

! DERIVATIVE OF VT

      DO L=1,5
         DO J=1,5
            IF (L .lt. J) then
               denom1=ONE/((W55(L,L)**2)-(W55(J,J)**2))
               SFTM1X(L,J)=denom1*( (W55(L,L)*SFTM2X(L,J)) +(W55(J,J)*SFTM2X(J,L)) )
               SFTM1Y(L,J)=denom1*( (W55(L,L)*SFTM2Y(L,J)) +(W55(J,J)*SFTM2Y(J,L)) )
               SFTM1Z(L,J)=denom1*( (W55(L,L)*SFTM2Z(L,J)) +(W55(J,J)*SFTM2Z(J,L)) )
            elseif (L .gt. J) then
               denom1=-ONE/((W55(J,J)**2)-(W55(L,L)**2))
               SFTM1X(L,J)=denom1*( (W55(J,J)*SFTM2X(J,L)) +(W55(L,L)*SFTM2X(L,J)) )
               SFTM1Y(L,J)=denom1*( (W55(J,J)*SFTM2Y(J,L)) +(W55(L,L)*SFTM2Y(L,J)) )
               SFTM1Z(L,J)=denom1*( (W55(J,J)*SFTM2Z(J,L)) +(W55(L,L)*SFTM2Z(L,J)) )
            else
               SFTM1X(L,J)=ZERO
               SFTM1Y(L,J)=ZERO
               SFTM1Z(L,J)=ZERO
            endif
         enddo
      enddo


      DO L=1,5
         DO J=1,5
            VT(L,J)=V(J,L)
         enddo
      enddo

      call mulnxn(VTX,SFTM1X,VT,5) !note: this is the derivative of VT and not V
      call mulnxn(VTY,SFTM1Y,VT,5) !note: this is the derivative of VT and not V
      call mulnxn(VTZ,SFTM1Z,VT,5) !note: this is the derivative of VT and not V

!      DO L=1,5
!      write(*,'(a4,5f15.10))')"VTD ",(VTX(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)


! DERIVATIVE OF U

      call mulmxn(SXVWI,SX,NRDC,5,VWI,5,5)

      call mulnxn(SFTM2X,WX,WI,5)
      call mulmxn(UWXWI,U,NRDC,5,SFTM2X,5,5)

      call mulnxn(SFTM1,SFTM1X,WI,5)
      call mulmxn(UWVTXVWI,UW,NRDC,5,SFTM1,5,5)



      call mulmxn(SYVWI,SY,NRDC,5,VWI,5,5)

      call mulnxn(SFTM2Y,WY,WI,5)
      call mulmxn(UWYWI,U,NRDC,5,SFTM2Y,5,5)

      call mulnxn(SFTM1,SFTM1Y,WI,5)
      call mulmxn(UWVTYVWI,UW,NRDC,5,SFTM1,5,5)



      call mulmxn(SZVWI,SZ,NRDC,5,VWI,5,5)

      call mulnxn(SFTM2Z,WZ,WI,5)
      call mulmxn(UWZWI,U,NRDC,5,SFTM2Z,5,5)

      call mulnxn(SFTM1,SFTM1Z,WI,5)
      call mulmxn(UWVTZVWI,UW,NRDC,5,SFTM1,5,5)


      DO L=1,NRDC
         DO J=1,5
            UX(L,J)=SXVWI(L,J)-UWXWI(L,J)-UWVTXVWI(L,J)
            UY(L,J)=SYVWI(L,J)-UWYWI(L,J)-UWVTYVWI(L,J)
            UZ(L,J)=SZVWI(L,J)-UWZWI(L,J)-UWVTZVWI(L,J)
         enddo
      enddo



!      DO L=1,NRDC
!         write(*,'(a3,5f15.10))')"UD ",(UX(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)


! Derivative of SI
!
! derivative of V
      DO L=1,5
         DO J=1,5
            VX(L,J)=VTX(J,L)
            VY(L,J)=VTY(J,L)
            VZ(L,J)=VTZ(J,L)
         enddo
      enddo

! derivative of UT
      DO L=1,5
         DO J=1,NRDC
            UTX(L,J)=UX(J,L)
            UTY(L,J)=UY(J,L)
            UTZ(L,J)=UZ(J,L)
         enddo
      enddo

! derivative of W inverse
      call mulnxn(WIX,WI,SFTM2X,5)

      call mulnxn(WIY,WI,SFTM2Y,5)

      call mulnxn(WIZ,WI,SFTM2Z,5)


      DO L=1,5
         DO J=1,5
            WIX(L,J)=-WIX(L,J)
            WIY(L,J)=-WIY(L,J)
            WIZ(L,J)=-WIZ(L,J)
         enddo
      enddo

      call mulmxn(VXWIUT,VX,5,5,WIUT,5,NRDC)

      call mulnxn(SFTM1,V,WIX,5)
      call mulmxn(VWIXUT,SFTM1,5,5,UT,5,NRDC)

      call mulmxn(VWIUTX,VWI,5,5,UTX,5,NRDC)


      call mulmxn(VYWIUT,VY,5,5,WIUT,5,NRDC)

      call mulnxn(SFTM1,V,WIY,5)
      call mulmxn(VWIYUT,SFTM1,5,5,UT,5,NRDC)

      call mulmxn(VWIUTY,VWI,5,5,UTY,5,NRDC)


      call mulmxn(VZWIUT,VZ,5,5,WIUT,5,NRDC)

      call mulnxn(SFTM1,V,WIZ,5)
      call mulmxn(VWIZUT,SFTM1,5,5,UT,5,NRDC)

      call mulmxn(VWIUTZ,VWI,5,5,UTZ,5,NRDC)


      DO L=1,5
         DO J=1,NRDC
            SIDX(L,J)=VXWIUT(L,J)+VWIXUT(L,J)+VWIUTX(L,J)
            SIDY(L,J)=VYWIUT(L,J)+VWIYUT(L,J)+VWIUTY(L,J)
            SIDZ(L,J)=VZWIUT(L,J)+VWIZUT(L,J)+VWIUTZ(L,J)
         enddo
      enddo

!      write(*,*)
!      DO L=1,5
!      write(*,'(a5,75(x,f13.8))')"SIDX ",
!     &        (SIDX(L,J),J=1,NRDC)
!      enddo


!     Derivative of Alignment tensor
      call mulmxn(AD,SIDX,5,NRDC,dred,NRDC,1)
      AX(1,1)=AD(1,1)
      AX(1,2)=AD(2,1)
      AX(1,3)=AD(3,1)
      AX(2,1)=AD(2,1)
      AX(2,2)=AD(4,1)
      AX(2,3)=AD(5,1)
      AX(3,1)=AD(3,1)
      AX(3,2)=AD(5,1)
      AX(3,3)=(-AD(1,1)-AD(4,1))


!      do L=1,3
!         write(*,*)  (AX(L,J),J=1,3)
!      enddo

!     Derivative of Alignment tensor
      call mulmxn(AD,SIDY,5,NRDC,dred,NRDC,1)
      AY(1,1)=AD(1,1)
      AY(1,2)=AD(2,1)
      AY(1,3)=AD(3,1)
      AY(2,1)=AD(2,1)
      AY(2,2)=AD(4,1)
      AY(2,3)=AD(5,1)
      AY(3,1)=AD(3,1)
      AY(3,2)=AD(5,1)
      AY(3,3)=(-AD(1,1)-AD(4,1))

!      do L=1,3
!      write(*,*)  (AY(L,J),J=1,3)
!      enddo


!     Derivative of Alignment tensor
      call mulmxn(AD,SIDZ,5,NRDC,dred,NRDC,1)
      AZ(1,1)=AD(1,1)
      AZ(1,2)=AD(2,1)
      AZ(1,3)=AD(3,1)
      AZ(2,1)=AD(2,1)
      AZ(2,2)=AD(4,1)
      AZ(2,3)=AD(5,1)
      AZ(3,1)=AD(3,1)
      AZ(3,2)=AD(5,1)
      AZ(3,3)=(-AD(1,1)-AD(4,1))

!     do L=1,3
!     write(*,*)  (AZ(L,J),J=1,3)
!     enddo
      RETURN
      END



      SUBROUTINE RDC_DERAT(NRDC,WI,W55,SMAT1,V,VWI,UW,WIUT,AN,DRED,SX, &
                 SY,SZ,K,AX,AY,AZ)
!-------------------------------------------------------------------------
   use number
   use consta
   use chm_types
   implicit none

      integer  NRDC,K
      real(chm_real)   SX(1,5),SY(1,5),SZ(1,5)
      real(chm_real)   WI(5,5),W55(5,5),SMAT1(NRDC*5),V(5,5),AN(3,3),DRED(NRDC)
      real(chm_real)   VWI(5,5),UW(NRDC*5),WIUT(5*NRDC)
      real(chm_real)   AX(3,3),AY(3,3),AZ(3,3)
! local
      integer  I,J,L
      integer  MM,N,NN
! a.tensor
      real(chm_real)   SFTMX(1,5),SFTMY(1,5),SFTMZ(1,5),SFTM1(5,5),SFTM2(5,5)
      real(chm_real)   SFTM3(5,5),SFTM1X(5,5),SFTM1Y(5,5),SFTM1Z(5,5)
      real(chm_real)   SFTM2X(5,5),SFTM2Y(5,5),SFTM2Z(5,5)
      real(chm_real)   denom1
      real(chm_real)   VT(5,5),WX(5,5),WY(5,5),WZ(5,5),VTX(5,5),VTY(5,5),VTZ(5,5)
      real(chm_real)   SXVWI(1,5),SYVWI(1,5),SZVWI(1,5),UWXWI,UWYWI,UWZWI
      real(chm_real)   UWVTXVWI,UWVTYVWI,UWVTZVWI
      real(chm_real)   VX(5,5),VY(5,5),VZ(5,5),UTX(5),UTY(5),UTZ(5),WIX(5,5),WIY(5,5),WIZ(5,5)
      real(chm_real)   VXWIUT,VYWIUT,VZWIUT,VWIXUT,VWIYUT,VWIZUT,VWIUTX,VWIUTY,VWIUTZ
      real(chm_real)   SID,SUMX,SUMY,SUMZ
      real(chm_real)   ADX(5),ADY(5),ADZ(5)
      real(chm_real)   temp(5),temp1(5),temp2(5),temp3

!     do L=1,NRDC
!     write(*,'(a5,5f15.10))')"SX-A ",(SX(L,J),J=1,5)
!     enddo

!     write(*,*)
!     write(*,*)

!     do L=1,5
!     write(*,'(10f15.9)')(SI(L,J),J=1,NRDC)
!     enddo

! DERIVATIVE OF W


      DO  I = 1,5
         DO  J = 1,1
            sftmx(J,I) = ZERO
            sftmy(J,I) = ZERO
            sftmz(J,I) = ZERO

            sxvwi(J,I) = ZERO
            syvwi(J,I) = ZERO
            szvwi(J,I) = ZERO

            DO L = 1,5
               sftmx(J,I) = sftmx(J,I) + SX(J,L) * V(L,I)
               sftmy(J,I) = sftmy(J,I) + Sy(J,L) * V(L,I)
               sftmz(J,I) = sftmz(J,I) + Sz(J,L) * V(L,I)

               sxvwi(J,I) = sxvwi(J,I) + SX(J,L) * Vwi(L,I)
               syvwi(J,I) = syvwi(J,I) + Sy(J,L) * Vwi(L,I)
               szvwi(J,I) = szvwi(J,I) + Sz(J,L) * Vwi(L,I)
            ENDDO
         ENDDO
      ENDDO
!      print*,'hi1'
!      print*, "    "

      MM=0
      DO L=1,5
         ADX(L)=ZERO
         ADY(L)=ZERO
         ADZ(L)=ZERO

         MM=(K+L-1)+((K-1)*4)
         temp3=smat1(MM)
!         print*,MM,temp3
         DO J=1,5
            WX(L,J)=ZERO        !Derivative of W
            WY(L,J)=ZERO        !Derivative of W
            WZ(L,J)=ZERO        !Derivative of W
            VT(L,J)=V(J,L)

            SFTM2X(L,J)=temp3*SFTMX(1,J)
            SFTM2Y(L,J)=temp3*SFTMY(1,J)
            SFTM2Z(L,J)=temp3*SFTMZ(1,J)

            if(L.eq.J) then
               WX(L,J)=SFTM2X(L,J)
               WY(L,J)=SFTM2Y(L,J)
               WZ(L,J)=SFTM2Z(L,J)
            endif
         enddo
      enddo

!      print*, '   '

!      DO J=1,5
!      print*, (SMAT(K,J),J=1,5)
!      enddo
!      if(K.eq.nrdc*5)  stop
!      print*, K
!      stop
!      DO L=1,5
!         write(*,'(a3,5f15.12))')"WX ",(WX(L,J),J=1,5)
!      write(*,'(a3,5f15.12))')"WI ",(WI(L,J),J=1,5)
!      write(*,'(a3,5f15.12))')"W ",(W55(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)

!      stop
! DERIVATIVE OF VT

      DO L=1,5
         DO J=1,5
            IF (L .lt. J) then
               denom1=ONE/((W55(L,L)**2)-(W55(J,J)**2))
               SFTM1X(L,J)=denom1*( (W55(L,L)*SFTM2X(L,J)) +(W55(J,J)*SFTM2X(J,L)) )
               SFTM1Y(L,J)=denom1*( (W55(L,L)*SFTM2Y(L,J)) +(W55(J,J)*SFTM2Y(J,L)) )
               SFTM1Z(L,J)=denom1*( (W55(L,L)*SFTM2Z(L,J)) +(W55(J,J)*SFTM2Z(J,L)) )
            elseif (L .gt. J) then
               denom1=-ONE/((W55(J,J)**2)-(W55(L,L)**2))
               SFTM1X(L,J)=denom1*( (W55(J,J)*SFTM2X(J,L)) +(W55(L,L)*SFTM2X(L,J)) )
               SFTM1Y(L,J)=denom1*( (W55(J,J)*SFTM2Y(J,L)) +(W55(L,L)*SFTM2Y(L,J)) )
               SFTM1Z(L,J)=denom1*( (W55(J,J)*SFTM2Z(J,L)) +(W55(L,L)*SFTM2Z(L,J)) )
            else
               SFTM1X(L,J)=ZERO
               SFTM1Y(L,J)=ZERO
               SFTM1Z(L,J)=ZERO
            endif
         enddo
      enddo

      DO  I = 1,5
         DO  J = 1,5
            vtx(J,I) = ZERO
            vty(J,I) = ZERO
            vtz(J,I) = ZERO

            sftm2x(J,I) = ZERO
            sftm2y(J,I) = ZERO
            sftm2z(J,I) = ZERO

            sftm1(J,I) = ZERO
            sftm2(J,I) = ZERO
            sftm3(J,I) = ZERO

            DO L = 1,5
               vtx(J,I) = vtx(J,I) + SFTM1X(J,L) * VT(L,I)
               vty(J,I) = vty(J,I) + SFTM1Y(J,L) * VT(L,I)
               vtz(J,I) = vtz(J,I) + SFTM1Z(J,L) * VT(L,I)

               sftm2x(J,I) = sftm2x(J,I) + WX(J,L) * wi(L,I)
               sftm2y(J,I) = sftm2y(J,I) + Wy(J,L) * wi(L,I)
               sftm2z(J,I) = sftm2z(J,I) + Wz(J,L) * wi(L,I)

               sftm1(J,I) = sftm1(J,I) + sftm1x(J,L) * wi(L,I)
               sftm2(J,I) = sftm2(J,I) + sftm1y(J,L) * wi(L,I)
               sftm3(J,I) = sftm3(J,I) + sftm1z(J,L) * wi(L,I)
            ENDDO
! derivative of V
            VX(I,J)=VTX(J,I)
            VY(I,J)=VTY(J,I)
            VZ(I,J)=VTZ(J,I)
         ENDDO
      ENDDO


!      DO L=1,5
!      write(*,'(a4,5f15.10))')"VTD ",(VTX(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)


! DERIVATIVE OF U

! derivative of W inverse
      DO  I = 1,5
         DO  J = 1,5
            wix(J,I) = ZERO
            wiy(J,I) = ZERO
            wiz(J,I) = ZERO

            DO L = 1,5
               wix(J,I) = wix(J,I) + WI(J,L) * SFTM2X(L,I)
               wiy(J,I) = wiy(J,I) + WI(J,L) * SFTM2Y(L,I)
               wiz(J,I) = wiz(J,I) + Wi(J,L) * SFTM2Z(L,I)
            ENDDO
            WIX(J,I)=-WIX(J,I)
            WIY(J,I)=-WIY(J,I)
            WIZ(J,I)=-WIZ(J,I)

!            sftm1x(J,I) = ZERO
!            sftm1y(J,I) = ZERO
!            sftm1z(J,I) = ZERO

!            DO L = 1,5
!               sftm1x(J,I) = sftm1x(J,I) + v(J,L) * wiX(L,I)
!               sftm1y(J,I) = sftm1y(J,I) + v(J,L) * wiY(L,I)
!               sftm1z(J,I) = sftm1z(J,I) + v(J,L) * wiZ(L,I)
!            ENDDO
         ENDDO
      ENDDO


!      DO L=1,5
!         DO J=1,5
!            WIX(L,J)=-WIX(L,J)
!            WIY(L,J)=-WIY(L,J)
!            WIZ(L,J)=-WIZ(L,J)
!         enddo
!      enddo

      DO  I = 1,5
         DO  J = 1,5
            sftm1x(J,I) = ZERO
            sftm1y(J,I) = ZERO
            sftm1z(J,I) = ZERO
            DO L = 1,5
               sftm1x(J,I) = sftm1x(J,I) + v(J,L) * wiX(L,I)
               sftm1y(J,I) = sftm1y(J,I) + v(J,L) * wiY(L,I)
               sftm1z(J,I) = sftm1z(J,I) + v(J,L) * wiZ(L,I)
            ENDDO
         ENDDO
      ENDDO

      NN=1
      N=1

      DO I=1,NRDC
         Do MM=1,5
            temp(MM)=smat1(NN)  !UT(MM,I) to get U
            temp1(MM)=UW(NN)
            temp2(MM)=WIUT(NN)
            NN=NN+1
         enddo
         DO J=1,5
            UWXWI=temp(1)*SFTM2X(1,J)+temp(2)*SFTM2X(2,J)+ &
                 temp(3)*SFTM2X(3,J)+temp(4)*SFTM2X(4,J)+ &
                 temp(5)*SFTM2X(5,J)
            UWVTXVWI=temp1(1)*SFTM1(1,J)+temp1(2)*SFTM1(2,J)+ &
                 temp1(3)*SFTM1(3,J)+temp1(4)*SFTM1(4,J)+ &
                 temp1(5)*SFTM1(5,J)

            UWYWI=temp(1)*SFTM2Y(1,J)+temp(2)*SFTM2Y(2,J)+ &
                 temp(3)*SFTM2Y(3,J)+temp(4)*SFTM2Y(4,J)+ &
                 temp(5)*SFTM2Y(5,J)
            UWVTYVWI=temp1(1)*SFTM2(1,J)+temp1(2)*SFTM2(2,J)+ &
                 temp1(3)*SFTM2(3,J)+temp1(4)*SFTM2(4,J)+ &
                 temp1(5)*SFTM2(5,J)

            UWZWI=temp(1)*SFTM2Z(1,J)+temp(2)*SFTM2Z(2,J)+ &
                 temp(3)*SFTM2Z(3,J)+temp(4)*SFTM2Z(4,J)+ &
                 temp(5)*SFTM2Z(5,J)
            UWVTZVWI=temp1(1)*SFTM3(1,J)+temp1(2)*SFTM3(2,J)+ &
                 temp1(3)*SFTM3(3,J)+temp1(4)*SFTM3(4,J)+ &
                 temp1(5)*SFTM3(5,J)

            SUMX=UWXWI+UWVTXVWI
            SUMY=UWYWI+UWVTYVWI
            SUMZ=UWZWI+UWVTZVWI

! derivative of UT
            if (I.eq.K) then
               UTX(J)=SXVWI(1,J)-SUMX
               UTY(J)=SYVWI(1,J)-SUMY
               UTZ(J)=SZVWI(1,J)-SUMZ
            else
               UTX(J)=-SUMX
               UTY(J)=-SUMY
               UTZ(J)=-SUMZ
            endif
            N=N+1
         enddo

         DO J=1,5
            VXWIUT=temp2(1)*VX(J,1)+temp2(2)*VX(J,2) &
                        +temp2(3)*VX(J,3)+ &
                        temp2(4)*VX(J,4)+temp2(5)*VX(J,5)

            VWIXUT=temp(1)*SFTM1X(J,1)+temp(2)*SFTM1X(J,2)+ &
                   temp(3)*SFTM1X(J,3)+temp(4)*SFTM1X(J,4)+ &
                   temp(5)*SFTM1X(J,5)

            VWIUTX=UTX(1)*VWI(J,1)+UTX(2)*VWI(J,2)+ &
                   UTX(3)*VWI(J,3)+UTX(4)*VWI(J,4)+UTX(5)*VWI(J,5)


            VYWIUT=temp2(1)*VY(J,1)+temp2(2)*VY(J,2) &
                        +temp2(3)*VY(J,3)+ &
                        temp2(4)*VY(J,4)+temp2(5)*VY(J,5)

            VWIYUT=temp(1)*SFTM1Y(J,1)+temp(2)*SFTM1Y(J,2)+ &
                   temp(3)*SFTM1Y(J,3)+temp(4)*SFTM1Y(J,4)+ &
                   temp(5)*SFTM1Y(J,5)

            VWIUTY=UTY(1)*VWI(J,1)+UTY(2)*VWI(J,2)+ &
                   UTY(3)*VWI(J,3)+UTY(4)*VWI(J,4)+UTY(5)*VWI(J,5)


            VZWIUT=temp2(1)*VZ(J,1)+temp2(2)*VZ(J,2) &
                        +temp2(3)*VZ(J,3)+ &
                        temp2(4)*VZ(J,4)+temp2(5)*VZ(J,5)

            VWIZUT=temp(1)*SFTM1Z(J,1)+temp(2)*SFTM1Z(J,2)+ &
                   temp(3)*SFTM1Z(J,3)+temp(4)*SFTM1Z(J,4)+ &
                   temp(5)*SFTM1Z(J,5)

            VWIUTZ=UTZ(1)*VWI(J,1)+UTZ(2)*VWI(J,2)+ &
                   UTZ(3)*VWI(J,3)+UTZ(4)*VWI(J,4)+UTZ(5)*VWI(J,5)

            SID=VXWIUT+VWIXUT+VWIUTX
            ADX(J)=(SID*dred(I))+ADX(J)

            SID=VYWIUT+VWIYUT+VWIUTY
            ADY(J)=(SID*dred(I))+ADY(J)

            SID=VZWIUT+VWIZUT+VWIUTZ
            ADZ(J)=(SID*dred(I))+ADZ(J)
         enddo
      enddo

!      DO L=1,NRDC
!         write(*,'(a3,5f15.10))')"UTD ",(UTX(L,J),J=1,5)
!      enddo
!      write(*,*)
!      write(*,*)

!     Derivative of Alignment tensor

      AX(1,1)=ADX(1)
      AX(1,2)=ADX(2)
      AX(1,3)=ADX(3)
      AX(2,1)=ADX(2)
      AX(2,2)=ADX(4)
      AX(2,3)=ADX(5)
      AX(3,1)=ADX(3)
      AX(3,2)=ADX(5)
      AX(3,3)=(-ADX(1)-ADX(4))

!      do L=1,3
!         write(*,*)  (AX(L,J),J=1,3)
!      enddo


      AY(1,1)=ADY(1)
      AY(1,2)=ADY(2)
      AY(1,3)=ADY(3)
      AY(2,1)=ADY(2)
      AY(2,2)=ADY(4)
      AY(2,3)=ADY(5)
      AY(3,1)=ADY(3)
      AY(3,2)=ADY(5)
      AY(3,3)=(-ADY(1)-ADY(4))

!      do L=1,3
!      write(*,*)  (AY(L,J),J=1,3)
!      enddo


      AZ(1,1)=ADZ(1)
      AZ(1,2)=ADZ(2)
      AZ(1,3)=ADZ(3)
      AZ(2,1)=ADZ(2)
      AZ(2,2)=ADZ(4)
      AZ(2,3)=ADZ(5)
      AZ(3,1)=ADZ(3)
      AZ(3,2)=ADZ(5)
      AZ(3,3)=(-ADZ(1)-ADZ(4))


!     do L=1,3
!     write(*,*)  (AZ(L,J),J=1,3)
!     enddo
      RETURN
      END
#endif
