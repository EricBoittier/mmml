SUBROUTINE PSFSUM(UNIT)
  !-----------------------------------------------------------------------
  !     Print out the structure file counters.

  use aniso_fcm
  use bases_fcm
  use chm_kinds
  use chm_types
  use code
  use comand
  use dimens_fcm
  use exfunc
  use lonepr
  use number
#if KEY_OPENMM==1
  use omm_ctrl, only: omm_system_changed
#endif
  use param_store, only: set_param
  use psf
  use stream
  use string
#if KEY_TORQUE==1
  use torque,only:qtorque,torque_clear   
#endif
  use image_routines_module, only: climag
  implicit none
  ! . Passed variables.
  INTEGER UNIT
  !
  CHARACTER(len=4) WINIT
  INTEGER I,J
  real(chm_real) QTOT,MTOT,QTOL
  !
  ! Set flags and reset data structures upon changing the PSF.
  ! . Reset the hydrogen-bond, non-bond and image data structures.
  WINIT='RESE'
  J=4
  CALL GTNBCT(WINIT,J,BNBND)
  CALL CLIMAG(BIMAG)
#if KEY_TORQUE==1
  if(qtorque) call TORQUE_CLEAR 
#endif
#if KEY_OPENMM==1
  call omm_system_changed()  
#endif
  !
  QTOT=0.0
  MTOT=0.0
  DO I=1,NATOM
     QTOT=QTOT+CG(I)
     MTOT=MTOT+AMASS(I)
  ENDDO
#if KEY_SINGLE==1
  QTOL=MAX(PT001,ABS(QTOT*PT001))
#else /**/
  QTOL=PT0001
#endif 
  IF(ABS(ANINT(QTOT)-QTOT).GT.QTOL) THEN
     IF(WRNLEV.GE.2) WRITE(OUTU,433) QTOT
433  FORMAT(/,' Warning from PSFSUM: The sum of charges (', &
          F12.6,') is not an integer',/)
     IF(INDXA(COMLYN, COMLEN, 'IGNO') .EQ. 0) &
          CALL WRNDIE(0,'<PSFSUM>','Total charge not an integer')
  ENDIF
  CGTOT=QTOT
  !
  MUSTUP=.TRUE.
#if KEY_ACE==1
  ACEUP=.TRUE.  
#endif
  !
  ! Print out new psf statistics.
  IF(PRNLEV.GE.2) THEN
     WRITE (UNIT, '(1X,A)') &
          'PSFSUM> PSF modified: NONBOND lists and IMAGE atoms cleared.'
     WRITE (UNIT, '(A)') &
          ' PSFSUM> Summary of the structure file counters :'
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of segments      = ', NSEG, &
          '   Number of residues   = ', NRES
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of atoms         = ', NATOM, &
          '   Number of groups     = ', NGRP
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of bonds         = ', NBOND, &
          '   Number of angles     = ', NTHETA
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of dihedrals     = ', NPHI, &
          '   Number of impropers  = ', NIMPHI
     I=0
#if KEY_CMAP==1
     I=NCRTERM
#endif 
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of cross-terms   = ', I, &
          '   Number of autogens   = ', NPAUTO
     WRITE (UNIT, '(A,I8,A,I8)') &
          '         Number of HB acceptors  = ', NACC, &
          '   Number of HB donors  = ', NDON
     WRITE (UNIT, '(A,I8,A,F10.5)') &
          '         Number of NB exclusions = ', NNB, &
          '   Total charge = ', CGTOT
     IF(QDRUDE)THEN
        WRITE (UNIT, '(A,I8,A,I8)') &
             '         Number of Drudes        = ', NDRUDE
        WRITE (UNIT, '(A,I8,A,I8)') &
             '         Number of true-bonds    = ', NBDRUDE, &
             '   Number of zero-bonds = ', NBOND-NBDRUDE
#if KEY_LONEPAIR==1
        WRITE (UNIT, '(A,I8,A,I8)') &
             '         Number of aniso. terms  = ', NANISO, &
             '   Number of lone-pairs = ', NUMLP
#endif 
     ENDIF
  ENDIF
  ! . Update the segment indexing array.
  NICTOT(NSEG+1) = NRES
  !
  !     determine if extended format i/o needed
  qextfmt=qxform()

  !
  ! set command parameters
  !
  CALL set_param('NSEG',NSEG)
  CALL set_param('NRES',NRES)
  CALL set_param('NATO',NATOM)
  CALL set_param('NATOM',NATOM)
  CALL set_param('NGRP',NGRP)
  CALL set_param('NBON',NBOND)
  CALL set_param('NBOND',NBOND)
  CALL set_param('NTHE',NTHETA)
  CALL set_param('NTHETA',NTHETA)
  CALL set_param('NPHI',NPHI)
  CALL set_param('NIMP',NIMPHI)
  CALL set_param('NIMPHI',NIMPHI)
  CALL set_param('NDRUDE',NDRUDE)
  CALL set_param('NBDRUDE',NBDRUDE)
#if KEY_CMAP==1
  CALL set_param('NCRT',NCRTERM)
#endif 
  CALL set_param('NNB',NNB)
  CALL set_param('NACC',NACC)
  CALL set_param('NDON',NDON)
  call set_param('CGTOT',CGTOT)
  call set_param('MASST',MTOT)
#if KEY_LONEPAIR==1
  CALL set_param('NUMLP',NUMLP)             
#endif
  !
  RETURN
END SUBROUTINE PSFSUM

