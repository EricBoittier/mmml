module rism_control
  use chm_kinds
  use dimens_fcm
  use rism
  implicit none
#if KEY_RISM==1 /*rism_glob*/
  !------------------------------------------------------------------
  !    Control variables for the iteration and the thermodynamic 
  !           integration and physical control parameters
  !------------------------------------------------------------------
  !   VARIABLE      PURPOSE
  !   --------      -------
  !     TEMP        Temperature used in the Boltzmann probability
  !     KBT         Stores the factor Kboltzmann*temperature
  !     RHO(DSITV)  Density of the solvent species (up to DSITV sites)
  !     SW(4)       Array for the various switching variables
  !                 SW(1)       Used for scaling the total energy
  !                 SW(2)       scaling the sigma of the vdw energy
  !                 SW(3)       scaling the coulombic charges
  !                 SW(4)       scaling the bridge function
  !     CDIE        Dielectric constant
  !     ADIE        Used to readjust the coulombic charges, so that
  !                 the macroscopic dielectric constant is reproduced
  !     SOL         Used to determine if a 'VV' (solvent-solvent),
  !                 'UV' (solute-solvent) or 'UU' (solute-solute)
  !                 calculation is done
  !     CLOS        Stores the type of closure used
  !
  real(chm_real) TEMP,KBT,RHO(DSITV),SW(4),CDIE,ADIE
  CHARACTER(len=2) SOL
  CHARACTER(len=3) CLOS
#endif /* (rism_glob)*/

end module rism_control


#if KEY_RISM==0 /*rism_main*/

SUBROUTINE RISMCMD
  CALL WRNDIE(-1,'<RISM>','RISM CODE NOT COMPILED')
  RETURN
end SUBROUTINE RISMCMD

#else /* (rism_main)*/

SUBROUTINE RISMCMD
  !-----------------------------------------------------------------------
  !      Author:  Benoit Roux
  !      This program has been written at Harvard University in 1988,
  !      The help and advice of Hsiang Ai Yu is greatfully acknowledged.
  !
  !      Routines were added and the  program was adapted
  !      for CHARMM by Georgios Archontis, in 1992
  !
  !  Useful references are:
  !  S.J. Singer, D. Chandler, Mol. Phys. 55, 621-625 (1985).
  !  D. Chandler, J. Chem. Phys. 67 1113-1124 (1977).
  !  F. Hirata, P.J. Rossky, Chem. Phys. Lett. 83 329-334 (1981).
  !  D. A. Zichi, P.J. Rossky, J. Chem. Phys. 84 1712-1723 (1986).
  !  M. B. Pettitt, M. Karplus, P. J. Rossky,
  !  J. Phys. Chem. 90 6335-6345 (1986).
  !  H. A. Yu, M. Karplus, J. Chem. Phys. 89 2366 (1988).
  !  H. A. Yu, B. Roux, M. Karplus, J. Chem. Phys. 92 5020-5033 (1990).
  !
  !========================================================================
  !
  !
  use dimens_fcm
  use cmdpar
  use comand
  use string
  use memory
  use stream
  !
  use rism
  use rism_control
  use struc
  use distri
  use fft
  use chm_kinds

  implicit none
  !
  !     local
  real(chm_real),allocatable,dimension(:,:) :: IPRHOGR
  CHARACTER(len=4) WRD4
  LOGICAL     OK,EOF,LUSED
  !
  !     Initialize local logicals to .FALSE.
  OK = .FALSE.
  EOF = .FALSE.
  LUSED = .FALSE.
  !
  !     Allocate space for RISM arrays on Heap
  CALL INITRISM(COMLYN,COMLEN)

  !     Set the default values for the fast fourier transform
  !     evaluation; remember that LOGFFT is the default
  !     Initialize the FFT with the defaults
  !     In the COMMON of fft.f90 and control.fcm
  DR = 0.02
  RMIN = -5.12
  NPOINT = 512
  QLOG = .TRUE.
  TEMP = 300.0
  CLOS = 'HNC'
  CALL INITFFT(COMLYN,COMLEN)
  WRITE(OUTU,'(/,7X,A,//)') 'RISM> ENTERING THE RISM MODULE '
  !
  ! Get the command line; as in the main CHARMM routines,
  ! the line is converted into capital letters
  !
1000 CONTINUE
  CALL XTRANE(COMLYN,COMLEN,'RISM')
  CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE., &
       'RISM> ')
  CALL MISCOM(COMLYN,MXCMSZ,COMLEN,LUSED)
  IF (.NOT.(.NOT.LUSED.OR.EOF)) GOTO 1000
  !
  IF (EOF) THEN
     !
     !     If we run out of stuff on a particular stream, pop to the
     !     previous stream. quit when there are no more streams.
     !
     CALL PPSTRM(OK)
     IF (.NOT.(OK)) THEN
        CALL FREERISM
        CALL STOPCH('RISM: END OF FILE')
     ENDIF
     EOF=.FALSE.
     GOTO 1000
  ENDIF
  !
  WRD4=NEXTA4(COMLYN,COMLEN)
  !
  IF (WRD4.EQ.'    ') THEN
     GOTO 1000

  ELSEIF(WRD4.EQ.'READ')THEN
     CALL IREAD

  ELSEIF(WRD4.EQ.'WRIT')THEN
     CALL IWRIT

  ELSEIF(WRD4.EQ.'SETU')THEN
     CALL INITFFT(COMLYN,COMLEN)

  ELSEIF(WRD4.EQ.'EDTZ')THEN
     CALL EDTZM

  ELSEIF(WRD4.EQ.'STAT')THEN
     CALL STATE(COMLYN,COMLEN)

  ELSEIF(WRD4.EQ.'ITER')THEN
     CALL CYCLES(COMLYN,COMLEN)

  ELSEIF(WRD4.EQ.'DERI')THEN
     CALL DERIV0(COMLYN,COMLEN)

  ELSEIF(WRD4.EQ.'SOLV')THEN
     CALL SOLVTION(COMLYN,COMLEN)

  ELSEIF(WRD4.EQ.'ANAL')THEN
     call chmalloc('rism.src','RISM','IPRHOGR',DVECT,NPRVV+NDU*NPRUV+NDUU*NPRUU,crl=IPRHOGR)
     CALL ANALYS(COMLYN,COMLEN,IPRHOGR)
     call chmdealloc('rism.src','RISM','IPRHOGR',DVECT,NPRVV+NDU*NPRUV+NDUU*NPRUU,crl=IPRHOGR)
  ELSEIF(WRD4.EQ.'STOP')THEN
     GOTO 2000

  ELSE
     WRITE(OUTU,'(7X,2A)') &
          'RISM> *ERROR* Unrecognized Command: ',WRD4

  ENDIF

  GOTO 1000

2000 CONTINUE
  CALL FREERISM
  RETURN
END SUBROUTINE RISMCMD

SUBROUTINE INITRISM(COMLYN,COMLEN)
  !-----------------------------------------------------------------------
  !     Initialize (Allocate) Arrays needed for RISM calculations
  !
  use dimens_fcm
  use string
  use stream
  use rism
  use distri
  use chm_kinds
  use memory
  use number
  implicit none
  !
  CHARACTER(len=*) COMLYN
  INTEGER       COMLEN
  !
  INTEGER       NPAIR
  !
  NSOLV=GTRMI(COMLYN,COMLEN,'NSTV',DSITV)
  NSOLU=GTRMI(COMLYN,COMLEN,'NSTU',DSITU)
  NDU  =GTRMI(COMLYN,COMLEN,'NSOL',DU)
  !
  !     Bound Cheking
  IF(NSOLV.GT.DSITV) THEN
     WRITE(OUTU,'(A,/,A,I5)') &
          ' RISM: INIT> Maximum Number of Solvent Sites exceeded.', &
          '             Adjusted to ',DSITV
     NSOLV=DSITV
  ENDIF
  !
  IF(NSOLU.GT.DSITU) THEN
     WRITE(OUTU,'(A,/,A,I5)') &
          ' RISM: INIT> Maximum Number of Solute Sites exceeded.', &
          '             Adjusted to ',DSITU
     NSOLU=DSITU
  ENDIF
  !
  IF(NDU.GT.DU) THEN
     WRITE(OUTU,'(A,/,A,I5)') &
          ' RISM: INIT> Maximum Number of Solutes exceeded.', &
          '             Adjusted to ',DU
     NDU=DU
  ENDIF
  !
  NDUU=NDU*(NDU+1)/2
  NPRVV=NSOLV*(NSOLV+1)/2
  NPRUU=NSOLU*(NSOLU+1)/2
  NPRUV=NSOLU*NSOLV
  NPAIR=NPRVV+NDU*NPRUV+NDUU*NPRUU
  !
  !     USR    Auxiliary array during the iteration cycle
  call chmalloc('rism.src','INITRISM','IPUSR',DVECT,NPAIR,crl=ipusr)
  ipusr = zero
  !
  !     PHIK   Fourier transform of the Coulomb potential (-beta*phik)
  call chmalloc('rism.src','INITRISM','IPPHIK',DVECT,NPAIR,crl=IPPHIK)
  ipphik = zero
  !
  !     GR     radial distribution function
  call chmalloc('rism.src','INITRISM','IPGR',DVECT,NPAIR,crl=IPGR)
  ipgr = zero
  !
  !     XVVK   Fourier transform of the solvent-solvent susceptibility
  call chmalloc('rism.src','INITRISM','IPXVVK',DVECT,NPRVV,crl=IPXVVK)
  ipxvvk = zero
  !
  !     CSR    short range direct correlation function
  call chmalloc('rism.src','INITRISM','IPCSR',DVECT,NPAIR,crl=IPCSR)
  ipcsr = zero
  !
  !     CSK    Fourier transform of the short range direct correl function
  call chmalloc('rism.src','INITRISM','IPCSK',DVECT,NPAIR,crl=IPCSK)
  ipcsk = zero
  !
  !     DGR   derivative of solvent g(r) with respect to solute density
  call chmalloc('rism.src','INITRISM','IPDGR',DVECT,NPRVV*NDU,crl=IPDGR)
  ipdgr = zero
  !
  !     DCSR  derivative of solvent cs(r) with respect to solute density
  call chmalloc('rism.src','INITRISM','IPDCSR',DVECT,NPRVV*NDU,crl=IPDCSR)
  ipdcsr = zero
  !
  RETURN
END SUBROUTINE INITRISM

SUBROUTINE FREERISM
  !-----------------------------------------------------------------------
  !     Free allocated arrays used for RISM calculations
  !
  use rism
  use distri
  use chm_kinds
  use memory
  implicit none
  integer NPAIR

  NPAIR = size(ipusr,2)
  call chmdealloc('rism.src','INITRISM','IPUSR',DVECT,NPAIR,crl=ipusr)
  call chmdealloc('rism.src','INITRISM','IPPHIK',DVECT,NPAIR,crl=IPPHIK)
  call chmdealloc('rism.src','INITRISM','IPGR',DVECT,NPAIR,crl=IPGR)
  call chmdealloc('rism.src','INITRISM','IPXVVK',DVECT,NPRVV,crl=IPXVVK)
  call chmdealloc('rism.src','INITRISM','IPCSR',DVECT,NPAIR,crl=IPCSR)
  call chmdealloc('rism.src','INITRISM','IPCSK',DVECT,NPAIR,crl=IPCSK)
  call chmdealloc('rism.src','INITRISM','IPDGR',DVECT,NPRVV*NDU,crl=IPDGR)
  call chmdealloc('rism.src','INITRISM','IPDCSR',DVECT,NPRVV*NDU,crl=IPDCSR)

  RETURN
END SUBROUTINE FREERISM

#endif /* (rism_main)*/

