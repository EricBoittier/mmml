#if KEY_RISM==1 /*rism_main*/
SUBROUTINE STATE(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     This subroutine sets up the thermodynamic state of the system
  !     for the temperature and density.
  !     It calculates the dielectric constant and re-adjusts
  !     the asymptotic form of the direct correlation function c(r)
  !     to produce a chosen dielectric constant.  Should not be used
  !     for salt solution.
  !     reference:  Chandler, J. Chem. Phys. 67, 1113 (1977).

  use stream
  use struc
  use distri
  use rism_control
  use consta
  use string
  use number
  use rism
  use chm_kinds
  implicit none
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !
  !     local variables
  real(chm_real) QTOT,DIPOLE,RHOI,CDIE2,YDIE
  CHARACTER(len=6) KEEP
  INTEGER IOFF,I,J,LENC


  !     Make the thermodynamic state (temperature and density)
  !     ------------------------------------------------------
  TEMP=GTRMF(COMLYN,COMLEN,'TEMP',TEMP)
  KBT=KBOLTZ*TEMP

  IF(CHECQUE(COMLYN,'DENSITY'))THEN
     IOFF=0
     DO I=1,NSEGV
        KEEP=ASEGM(I)
        LENC=LEN(KEEP)
        CALL TRIMA(KEEP,LENC)
        RHOI=GTRMF(COMLYN,COMLEN,KEEP(1:LENC),ZERO)
        DO J=1,ISEGM(I)
           RHO(J+IOFF)=RHOI
        ENDDO
        IOFF=IOFF+ISEGM(I)
     ENDDO
  ENDIF

  !     Calculate the dipole moment the natural dielectric constant
  !     -----------------------------------------------------------
  ADIE=ONE
  CDIE=MINONE
  CDIE2=GTRMF(COMLYN,COMLEN,'CDIE',ZERO)

  IF(CDIE2.NE.ZERO)THEN
     IOFF=1
     YDIE=ZERO
     DO I=1,NSEGV
        DIPOLE=ZERO
        CALL MKDIPO(X(IOFF),Y(IOFF),Z(IOFF),CHARG, &
             ICHEM(IOFF),ASEGM(I),ISEGM(I),DIPOLE,QTOT)
        YDIE=YDIE+FOUR*PI*RHO(IOFF)*COEFF*DIPOLE**2/(KBT*NINE)
        IOFF=IOFF+ISEGM(I)
     ENDDO
     CDIE=ONE+THREE*YDIE
     IF(PRNLEV.GT.2) WRITE(OUTU,100) CDIE
100  FORMAT(/,7X,'The natural dielectric constant is ',F10.5)

     IF(CDIE2.GT.ZERO)THEN
        IF(ABS((THREE*YDIE*(CDIE2-ONE))).GT.RSMALL) THEN
           ADIE=(ONE+CDIE2*(THREE*YDIE-ONE))/(THREE*YDIE*(CDIE2-ONE))
        ELSE
           ADIE=ONE
        ENDIF
        IF(PRNLEV.GT.2) WRITE(OUTU,101) CDIE2
101     FORMAT(7X, &
             'The dielectric constant of the fluid has been changed to', &
             F10.5,//,7X,'The effective partial charges will be:')
        DO I=1,NSITV
           IF(PRNLEV.GT.2) WRITE(OUTU,102) I,RESNAM(I),ATNAM(I), &
                SEGMID(I),SQRT(ADIE)*CHARG(ICHEM(I))
102        FORMAT(7X,I4,3(1X,A4),1X,F10.5)
        ENDDO
        CDIE=CDIE2
     ENDIF

  ENDIF

  RETURN
END SUBROUTINE STATE

SUBROUTINE MKDIPO(X,Y,Z,CHARG,ICHEM,ASEGM,NSITE,DIPOLE,QTOT)
  !-----------------------------------------------------------------------
  !     This subroutine calculates the dipole moment
  !     of the segment asegm.
  use number
  use rism
  use stream
  use chm_kinds
  implicit none
  real(chm_real) X(*),Y(*),Z(*),CHARG(*),DIPOLE,QTOT
  INTEGER ICHEM(*),NSITE
  CHARACTER(len=*) ASEGM

  real(chm_real) DIJ2
  INTEGER I,J

  DIPOLE=ZERO
  QTOT=ZERO
  DO I=1,NSITE
     QTOT=QTOT+CHARG(ICHEM(I))
     DO J=1,NSITE
        DIJ2=(X(I)-X(J))**2 + (Y(I)-Y(J))**2 + (Z(I)-Z(J))**2
        !         calculate the -2.0*dipole**2
        DIPOLE=DIPOLE+CHARG(ICHEM(I))*CHARG(ICHEM(J))*DIJ2  
     ENDDO
  ENDDO
  ! the dipole modulus expressed in [unit charge]/angstrom] 
  !
  DIPOLE=SQRT(-HALF*DIPOLE) 
  IF(ABS(QTOT).LE.RSMALL)THEN
     IF(PRNLEV.GT.2) WRITE(OUTU,100) ASEGM,DIPOLE,4.8*DIPOLE
100  FORMAT(7X,'segment ',A,' dipole moment ',F10.5,' = ',F10.5,' D')
  ELSE
     DIPOLE=ZERO
     IF(PRNLEV.GT.2) WRITE(OUTU,101) ASEGM,QTOT
101  FORMAT(7X,'segment ',A,' total charge  ',F10.5)
  ENDIF

  RETURN
END SUBROUTINE MKDIPO

SUBROUTINE ANALYS(COMLYN,COMLEN,RHOGR)
  !-----------------------------------------------------------------------
  !     This subroutine parses the control command for the analysis

  use dimens_fcm
  use string
  use stream
  use struc
  use distri
  use rism_control
  use fft
  use rism
  use chm_kinds
  implicit none
  !     command parser
  CHARACTER(len=*) COMLYN
  INTEGER   COMLEN
  !     Density around a site
  real(chm_real) RHOGR(DVECT,*)
  !
  !
  !     Local variables
  !     ---------------

  !     Cycle iteration control variables
  real(chm_real) TOL,RMIX
  LOGICAL QINIT,CONVRG,QPRINT
  INTEGER IUNIT,IPAIR,I,J,IR

  !     CONTROL SWITCH VARIABLES
  real(chm_real)  SWI(4),SWF(4),DSW(4)
  INTEGER NSW(4)

  !     Solute label
  INTEGER UA,UB,IUA,IUB,UUAB,IOFF

  !     Local variables:
  INTEGER IRMIN(DPAIR,4),IRMAX(DPAIR,4)
  real(chm_real)  SUM(DPAIR,3)

  !     Local Heap management
  !!      INTEGER IOFF1,IOFF2
  !
  IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
  IPAIR=GTRMI(COMLYN,COMLEN,'PAIR',-1)
  QPRINT=(IOLEV.GT.0)
  IF(IUNIT.EQ.OUTU) QPRINT=(PRNLEV.GT.2)

  IF(CHECQUE(COMLYN,'UV'))THEN
     SOL='UV'
     IUA=GTRMI(COMLYN,COMLEN,'SOLU',1)
     IOFF=PUV(IUA)
     DO I=1,NSITU(IUA)
        DO J=1,NSITV
           DO IR=NFIRST,NPOINT
              !! IOFF2 = DVECT*(IOFF-1+INTUV(I,J,IUA)-1) + IR - 1 !! APH: potential bug here
              RHOGR(IR,INTUV(I,J,IUA)) = IPGR(IR,IOFF-1+INTUV(I,J,IUA)) * RHO(J)
           ENDDO
        ENDDO
     ENDDO

     CALL ANALY2(IPGR(1,IOFF),RHOGR,NPUV(IUA),PRNAM(IOFF), &
          IUNIT,QPRINT,IPAIR,IRMIN,IRMAX,SUM)

  ELSE
     IF(PRNLEV.GT.2 .AND. .NOT.CHECQUE(COMLYN,'VV')) &
          WRITE(IUNIT,'(A)') &
          ' RISM:ANALYS> Solvent-Solvent (default) analysis performed.'
     SOL='VV'
     DO I=1,NSITV
        DO J=1,NSITV
           DO IR=NFIRST,NPOINT
              RHOGR(IR,INTVV(I,J)) = IPGR(IR,INTVV(I,J)) * RHO(J)
           ENDDO
        ENDDO
     ENDDO

     CALL ANALY2(IPGR,RHOGR,NPVV,PRNAM, &
          IUNIT,QPRINT,IPAIR,IRMIN,IRMAX,SUM)

  ENDIF

  RETURN
END SUBROUTINE ANALYS

SUBROUTINE ANALY2(GR,RHOGR,NPAIR,PRNAM, &
     IUNIT,QPRINT,IPAIR,IRMIN,IRMAX,SUM)
  !-----------------------------------------------------------------------
  !     This subroutine finds  the 3 first solvation shells for the
  !     desired atom pairs
  !
  use fft
  use rism
  use chm_kinds
  implicit none
  INTEGER NPAIR,IUNIT,IPAIR
  real(chm_real) GR(DVECT,*),RHOGR(DVECT,*)
  CHARACTER(len=13) PRNAM(*)
  LOGICAL QPRINT
  INTEGER IRMIN(DPAIR,4),IRMAX(DPAIR,4)
  real(chm_real)  SUM(DPAIR,3)
  real(chm_real) INTG3D,INTG1D
  EXTERNAL INTG3D,INTG1D

  !     Local variables:
  INTEGER PFIRST,PLAST
  INTEGER IP,I

  IF(IPAIR.GT.0)THEN
     PFIRST=IPAIR
     PLAST=IPAIR
  ELSE
     PFIRST=1
     PLAST=NPAIR
  ENDIF

  DO IP=PFIRST,PLAST
     IRMIN(IP,1)=NFIRST+1
     DO I=1,3

        !         First, find maxima location
        IRMAX(IP,I)=IRMIN(IP,I)
        CALL MAXIMA(GR(1,IP),IRMAX(IP,I))

        !         Second, find the minima location after the i-th maxima
        IRMIN(IP,I+1)=IRMAX(IP,I)
        CALL MINIMA(GR(1,IP),IRMIN(IP,I+1))

        !         Then, calculate the solvation number in those sub-regions
        SUM(IP,I)=INTG3D(RHOGR(1,IP),IRMIN(IP,I),IRMIN(IP,I+1))

     ENDDO
  ENDDO

  !     Finally,write the output
  IF(.NOT.QPRINT) RETURN
  !
  WRITE(IUNIT,100)
100 FORMAT(/,' Analysis of the solvation shells:',/)
  WRITE(IUNIT,101)
101 FORMAT(1X,'pair',4X,'Atoms',10X,'rmax ',5X,'Gmax ', &
       5X,'rmin ',5X,'Gmin ',5X,'shell',/)

  DO IP=PFIRST,PLAST
     WRITE(IUNIT,102) IP,PRNAM(IP), &
          R(IRMAX(IP,1)),GR(IRMAX(IP,1),IP), &
          R(IRMIN(IP,2)),GR(IRMIN(IP,2),IP), &
          SUM(IP,1), &
          R(IRMAX(IP,2)),GR(IRMAX(IP,2),IP), &
          R(IRMIN(IP,3)),GR(IRMIN(IP,3),IP), &
          SUM(IP,2), &
          R(IRMAX(IP,3)),GR(IRMAX(IP,3),IP), &
          R(IRMIN(IP,4)),GR(IRMIN(IP,4),IP), &
          SUM(IP,3)
102  FORMAT(I4,1X,A13,1X,5F10.3,/,2(19X,5F10.3,/))
  ENDDO

  RETURN
END SUBROUTINE ANALY2

SUBROUTINE MAXIMA(GR,IRMAX)
  !-----------------------------------------------------------------------
  !     This subroutine finds the next maxima
  use fft
  use rism
  use chm_kinds
  implicit none
  real(chm_real) GR(DVECT)
  INTEGER IRMAX

  LOGICAL LMAX
  INTEGER IR

  DO IR=IRMAX,NPOINT-1
     LMAX=((GR(IR).GT.GR(IR-1)).AND. &
          (GR(IR).GT.GR(IR+1)))
     IF(LMAX)THEN
        IRMAX=IR
        RETURN
     ENDIF
  ENDDO
END SUBROUTINE MAXIMA

SUBROUTINE MINIMA(GR,IRMIN)
  !-----------------------------------------------------------------------
  !     This subroutine finds the next minima
  use rism
  use fft
  use chm_kinds
  implicit none
  real(chm_real) GR(DVECT)
  INTEGER IRMIN

  LOGICAL LMIN
  INTEGER IR

  DO IR=IRMIN,NPOINT
     LMIN=((GR(IR).LT.GR(IR-1)).AND. &
          (GR(IR).LT.GR(IR+1)))
     IF(LMIN)THEN
        IRMIN=IR
        RETURN
     ENDIF
  ENDDO
END SUBROUTINE MINIMA

SUBROUTINE INVRS(A,AINV,N)
  !-----------------------------------------------------------------------
  !     this subroutine is used to call the IMSL
  !     matrix inverter.

  use rism
  use stream
  use chm_kinds
  implicit none
  INTEGER N
  INTEGER, PARAMETER :: IA=DSITE
  real(chm_real) A(IA,IA),AINV(IA,IA)

  INTEGER IER
  real(chm_real)  WKAREA(IA)

  CALL LINV1F(A,N,IA,AINV,0,WKAREA,IER)
  IF(IER.NE.0)THEN
     IF(PRNLEV.GT.2) WRITE(OUTU,100) IER
100  FORMAT(' *INVRS ERROR* ','Error index was: IER = ',I4)
  ENDIF
  RETURN
END SUBROUTINE INVRS

!========================================================================

FUNCTION INTG3D(FR,N1,N2)
  !     This function integrates the f(r) in 3D space without
  !     assuming equally spaced points.
  use rism
  use number
  use fft
  use consta
  use chm_kinds
  implicit none
  INTEGER N1,N2
  real(chm_real) FR(*),intg3d
  real(chm_real) INTG1D
  EXTERNAL INTG1D
  !     Local variables
  real(chm_real) FR3D(DVECT)
  INTEGER IR

  DO IR=N1,N2
     FR3D(IR)=FOUR*PI*R(IR)**2*FR(IR)
  ENDDO
  INTG3D=INTG1D(FR3D,R,N1,N2)
  RETURN
END FUNCTION INTG3D

!==============================================================================

FUNCTION INTG1D(FX,X,N1,N2)
  !     This function integrates the f(x) in 1D space without 
  !     assuming equally spaced points.  
  use number
  use chm_kinds
  implicit none
  real(chm_real) FX(*),X(*),intg1d
  INTEGER N1,N2

  INTEGER IX
  INTG1D=ZERO
  DO IX=N1,N2-1
     INTG1D=INTG1D+ &
          HALF*(FX(IX)+FX(IX+1))*(X(IX+1)-X(IX))
  ENDDO
  RETURN
end FUNCTION INTG1D

#else /* (rism_main)*/

SUBROUTINE NULL_STATE
  RETURN
end SUBROUTINE NULL_STATE

#endif /* (rism_main)*/

