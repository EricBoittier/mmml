module prssre

  ! flag for dynamic adjustment of IMXCEN based on XTLA for P21
  LOGICAL, PUBLIC :: QP21XCEN

  private

  public getprs, getvol, prsext, prsint, virtot, virshk, viral

contains

  SUBROUTINE GETPRS( COMLYN, COMLEN )
    !-----------------------------------------------------------------------
    !     Process the pressure commands for the system. There are three
    !     modes :
    !
    !     Mode 1 : Initialise all pressure arrays.
    !
    !      Syntax:
    !
    !      PRESsure INITialise
    !
    !     Mode 2 : Calculate and print the instantaneous pressures for
    !              a system.
    !
    !      Syntax:
    !
    !      PRESsure INSTantaneous TEMPerature <Real> VOLUme <Real> -
    !      NDEGf <Integer> NOPRint
    !
    !     The external isotropic pressure and tensor are calculated
    !     if a volume is present. The isotropic internal pressure is
    !     calculated if a volume is present and a temperature has been
    !     given. If no degrees of freedom are specified then a value
    !     of 3*NATOM is taken be default. The virials are always printed.
    !     NOPRint will suppress all printing.
    !
    !     Note: a previous call to energy is required so that the
    !           virials (and volume if not specified) are available
    !           in ENERGY.FCM. The command also accumulates the
    !           average and the square of the pressure variables.
    !
    !     Mode 3 : Print the averages and fluctuations.
    !
    !      PRESsure STATistics
    !
    use chm_kinds
    use consta
    use dimens_fcm
    use energym
    use averfluc
    use number
    use psf
    use stream
    use string
    implicit none
    !     . Passed variables.
    CHARACTER(len=*) COMLYN
    INTEGER   COMLEN
    !     . Local variables.
    CHARACTER(len=4) COMAND
    INTEGER   NDEGF,I
    LOGICAL   QPRINT
    real(chm_real)    TEMP, VOLU
    !     . Local saved variables.
    INTEGER   NFRAME
    SAVE      NFRAME
    !     . Get the command.
    COMAND = NEXTA4 ( COMLYN, COMLEN )
    !=======================================================================
    !     . Initialise the pressure variables.
    !=======================================================================
    IF ( COMAND  ==  'INIT' ) THEN
       !     . The number of frames.
       NFRAME = 0
       !     . Instantaneous values.
       EPROP(PRESSE)  = ZERO
       EPROP(PRESSI)  = ZERO
       EPROP(VIRE)    = ZERO
       EPROP(VIRI)    = ZERO
       EPROP(VIRKE)   = ZERO
       EPROP(VOLUME)  = ZERO
       !     . Average values.
       EPRPA(PRESSE)  = ZERO
       EPRPA(PRESSI)  = ZERO
       EPRPA(VIRE)    = ZERO
       EPRPA(VIRI)    = ZERO
       EPRPA(VIRKE)   = ZERO
       EPRPA(VOLUME)  = ZERO
       !     . Fluctuation values.
       EPRP2A(PRESSE) = ZERO
       EPRP2A(PRESSI) = ZERO
       EPRP2A(VIRE)   = ZERO
       EPRP2A(VIRI)   = ZERO
       EPRP2A(VIRKE)  = ZERO
       EPRP2A(VOLUME) = ZERO
       DO I = 1,LENENV
          EPRESS(I) = ZERO
          EPRSA(I)  = ZERO
          EPRS2A(I) = ZERO
       ENDDO
       !=======================================================================
       !     . Print the pressure statistics.
       !=======================================================================
    ELSE IF ( COMAND  ==  'STAT' ) THEN
       IF ( NFRAME  ==  0 ) THEN
          CALL WRNDIE ( -5, '<GETPRS>', 'Number of frames zero.' )
       ENDIF
       !     . Calculate the pressure/virial averages and fluctations.
       EPRPA(PRESSE)  = EPRPA(PRESSE) / NFRAME
       EPRPA(PRESSI)  = EPRPA(PRESSI) / NFRAME
       EPRPA(VIRE)    = EPRPA(VIRE)   / NFRAME
       EPRPA(VIRI)    = EPRPA(VIRI)   / NFRAME
       EPRPA(VIRKE)   = EPRPA(VIRKE) / NFRAME
       EPRPA(VOLUME)  = EPRPA(VOLUME) / NFRAME
       EPRP2A(PRESSE) = SQRT(EPRP2A(PRESSE)/NFRAME-EPRPA(PRESSE)**2)
       EPRP2A(PRESSI) = SQRT(EPRP2A(PRESSI)/NFRAME-EPRPA(PRESSI)**2)
       EPRP2A(VIRE)   = SQRT(EPRP2A(VIRE)/NFRAME-EPRPA(VIRE)**2)
       EPRP2A(VIRI)   = SQRT(EPRP2A(VIRI)/NFRAME-EPRPA(VIRI)**2)
       EPRP2A(VIRKE)  = SQRT(EPRP2A(VIRKE)/NFRAME-EPRPA(VIRKE)**2)
       EPRP2A(VOLUME) = SQRT(EPRP2A(VOLUME)/NFRAME-EPRPA(VOLUME)**2)
       DO I = 1,LENENV
          EPRSA(I)  = EPRSA(I) / NFRAME
          EPRS2A(I) = SQRT(EPRS2A(I)/NFRAME-EPRSA(I)**2)
       ENDDO
       !     . Get the average volume.
       VOLU = EPRPA(VOLUME)
       !     . Print the averages.
       IF(PRNLEV >= 2) THEN
          WRITE ( OUTU, '(1X,79(''-''))' )
          WRITE ( OUTU, '(36X,A)' ) 'Averages'
          WRITE ( OUTU, '(1X,79(''-''))' )
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'Virial KE       = ', EPRPA(VIRKE), &
               ' Volume          = ', EPRPA(VOLUME)
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'External Virial = ', EPRPA(VIRE), &
               ' Internal Virial = ', EPRPA(VIRI)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'External Virial Tensor  : xx ', EPRSA(VEXX), &
               ' xy ', EPRSA(VEXY), &
               ' xz ', EPRSA(VEXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRSA(VEYX), &
               ' yy ', EPRSA(VEYY), &
               ' yz ', EPRSA(VEYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRSA(VEZX), &
               ' zy ', EPRSA(VEZY), &
               ' zz ', EPRSA(VEZZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'Internal Virial Tensor  : xx ', EPRSA(VIXX), &
               ' xy ', EPRSA(VIXY), &
               ' xz ', EPRSA(VIXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRSA(VIYX), &
               ' yy ', EPRSA(VIYY), &
               ' yz ', EPRSA(VIYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRSA(VIZX), &
               ' zy ', EPRSA(VIZY), &
               ' zz ', EPRSA(VIZZ)
          !     . Check the volume.
          IF ( VOLU  >  ZERO ) THEN
             !     . Print both pressures.
             IF ( EPRPA(PRESSI)  /=  ZERO ) THEN
                WRITE ( OUTU, '(1X,A,F14.5,10X,A,F14.5)' ) &
                     'External Pressure = ', EPRPA(PRESSE), &
                     ' Internal Pressure = ', EPRPA(PRESSI)
                !     . Print the external pressure.
             ELSE
                WRITE ( OUTU, '(1X,A,F14.5)' ) &
                     'External Pressure = ', EPRPA(PRESSE)
             ENDIF
             !     . Print the external pressure tensor.
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  'External Pressure Tensor: xx ', EPRSA(PEXX), &
                  ' xy ', EPRSA(PEXY), &
                  ' xz ', EPRSA(PEXZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          yx ', EPRSA(PEYX), &
                  ' yy ', EPRSA(PEYY), &
                  ' yz ', EPRSA(PEYZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          zx ', EPRSA(PEZX), &
                  ' zy ', EPRSA(PEZY), &
                  ' zz ', EPRSA(PEZZ)
          ENDIF
          !     . Print the fluctuations.
          WRITE ( OUTU, '(1X,79(''-''))' )
          WRITE ( OUTU, '(36X,A)' ) 'Fluctuations'
          WRITE ( OUTU, '(1X,79(''-''))' )
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'Virial KE       = ', EPRP2A(VIRKE), &
               ' Volume          = ', EPRP2A(VOLUME)
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'External Virial = ', EPRP2A(VIRE), &
               ' Internal Virial = ', EPRP2A(VIRI)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'External Virial Tensor  : xx ', EPRS2A(VEXX), &
               ' xy ', EPRS2A(VEXY), &
               ' xz ', EPRS2A(VEXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRS2A(VEYX), &
               ' yy ', EPRS2A(VEYY), &
               ' yz ', EPRS2A(VEYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRS2A(VEZX), &
               ' zy ', EPRS2A(VEZY), &
               ' zz ', EPRS2A(VEZZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'Internal Virial Tensor  : xx ', EPRS2A(VIXX), &
               ' xy ', EPRS2A(VIXY), &
               ' xz ', EPRS2A(VIXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRS2A(VIYX), &
               ' yy ', EPRS2A(VIYY), &
               ' yz ', EPRS2A(VIYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRS2A(VIZX), &
               ' zy ', EPRS2A(VIZY), &
               ' zz ', EPRS2A(VIZZ)
          !     . Check the volume.
          IF ( VOLU  >  ZERO ) THEN
             !     . Print both pressures.
             IF ( EPRPA(PRESSI)  /=  ZERO ) THEN
                WRITE ( OUTU, '(1X,A,F14.5,10X,A,F14.5)' ) &
                     'External Pressure = ', EPRP2A(PRESSE), &
                     ' Internal Pressure = ', EPRP2A(PRESSI)
                !     . Print the external pressure.
             ELSE
                WRITE ( OUTU, '(1X,A,F14.5)' ) &
                     'External Pressure = ', EPRP2A(PRESSE)
             ENDIF
             !     . Print the external pressure tensor.
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  'External Pressure Tensor: xx ', EPRS2A(PEXX), &
                  ' xy ', EPRS2A(PEXY), &
                  ' xz ', EPRS2A(PEXZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          yx ', EPRS2A(PEYX), &
                  ' yy ', EPRS2A(PEYY), &
                  ' yz ', EPRS2A(PEYZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          zx ', EPRS2A(PEZX), &
                  ' zy ', EPRS2A(PEZY), &
                  ' zz ', EPRS2A(PEZZ)
          ENDIF
          WRITE ( OUTU, '(1X,79(''-''))' )
       ENDIF
       !=======================================================================
       !     . Initialise the pressure variables.
       !=======================================================================
    ELSE IF ( COMAND  ==  'INST' ) THEN
       !     . Parse the command line.
       NDEGF  = GTRMI ( COMLYN, COMLEN, 'NDEG', 3*NATOM )
       QPRINT = INDXA ( COMLYN, COMLEN, 'NOPR' )  ==  0
       TEMP   = GTRMF ( COMLYN, COMLEN, 'TEMP', MINONE )
       VOLU   = GTRMF ( COMLYN, COMLEN, 'VOLU',  EPROP(VOLUME) )
       EPROP(VOLUME) = VOLU
       !     . Increment the NFRAME counter.
       NFRAME = NFRAME + 1
       !     . Calculate the virial temperature.
       EPROP(VIRKE) = - THREE * ( EPROP(VIRE) + EPROP(VIRI) ) / TWO
       !     . Print out the virials.
       IF(PRNLEV < 2) QPRINT=.FALSE.
       IF ( QPRINT ) THEN
          WRITE ( OUTU, '(1X,79(''-''))' )
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'Virial KE       = ', EPROP(VIRKE), &
               ' Volume          = ', EPROP(VOLUME)
          WRITE ( OUTU, '(1X,A,F14.5,14X,A,F14.5)' ) &
               'External Virial = ', EPROP(VIRE), &
               ' Internal Virial = ', EPROP(VIRI)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'External Virial Tensor  : xx ', EPRESS(VEXX), &
               ' xy ', EPRESS(VEXY), &
               ' xz ', EPRESS(VEXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRESS(VEYX), &
               ' yy ', EPRESS(VEYY), &
               ' yz ', EPRESS(VEYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRESS(VEZX), &
               ' zy ', EPRESS(VEZY), &
               ' zz ', EPRESS(VEZZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               'Internal Virial Tensor  : xx ', EPRESS(VIXX), &
               ' xy ', EPRESS(VIXY), &
               ' xz ', EPRESS(VIXZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          yx ', EPRESS(VIYX), &
               ' yy ', EPRESS(VIYY), &
               ' yz ', EPRESS(VIYZ)
          WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
               '                          zx ', EPRESS(VIZX), &
               ' zy ', EPRESS(VIZY), &
               ' zz ', EPRESS(VIZZ)
       ENDIF
       !     . Check the volume.
       IF ( VOLU  >  ZERO ) THEN
          !     . Calculate the external pressure.
          CALL PRSEXT
          !     . Calculate the internal pressure (given a temperature).
          IF ( TEMP  >=  ZERO ) THEN
             EPROP(PRESSI) = PATMOS / ( THREE * EPROP(VOLUME) ) * &
                  ( KBOLTZ * NDEGF * TEMP + &
                  EPRESS(VIXX) + EPRESS(VIYY) + EPRESS(VIZZ) )
          ELSE
             EPROP(PRESSI) = PATMOS / ( THREE * EPROP(VOLUME) ) * &
                  ( EPRESS(VIXX) + EPRESS(VIYY) + EPRESS(VIZZ) )
          ENDIF
          !     . Print both pressures.
          IF ( QPRINT ) THEN
             WRITE ( OUTU, '(1X,A,F14.5,10X,A,F14.5)' ) &
                  'External Pressure = ', EPROP(PRESSE), &
                  ' Internal Pressure = ', EPROP(PRESSI)
             !     . Print the external pressure tensor.
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  'External Pressure Tensor: xx ', EPRESS(PEXX), &
                  ' xy ', EPRESS(PEXY), &
                  ' xz ', EPRESS(PEXZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          yx ', EPRESS(PEYX), &
                  ' yy ', EPRESS(PEYY), &
                  ' yz ', EPRESS(PEYZ)
             WRITE ( OUTU, '(1X,3(A,F14.5))' ) &
                  '                          zx ', EPRESS(PEZX), &
                  ' zy ', EPRESS(PEZY), &
                  ' zz ', EPRESS(PEZZ)
          ENDIF
       ENDIF
       !     . Finish.
       IF ( QPRINT ) WRITE ( OUTU, '(1X,79(''-''))' )
       !     . Accumulate the pressure/virial statistics.
       EPRPA(PRESSE)  = EPRPA(PRESSE)  + EPROP(PRESSE)
       EPRPA(PRESSI)  = EPRPA(PRESSI)  + EPROP(PRESSI)
       EPRPA(VIRE)    = EPRPA(VIRE)    + EPROP(VIRE)
       EPRPA(VIRI)    = EPRPA(VIRI)    + EPROP(VIRI)
       EPRPA(VIRKE)   = EPRPA(VIRKE)   + EPROP(VIRKE)
       EPRPA(VOLUME)  = EPRPA(VOLUME)  + EPROP(VOLUME)
       EPRP2A(PRESSE) = EPRP2A(PRESSE) + EPROP(PRESSE)**2
       EPRP2A(PRESSI) = EPRP2A(PRESSI) + EPROP(PRESSI)**2
       EPRP2A(VIRE)   = EPRP2A(VIRE)   + EPROP(VIRE)**2
       EPRP2A(VIRI)   = EPRP2A(VIRI)   + EPROP(VIRI)**2
       EPRP2A(VIRKE)  = EPRP2A(VIRKE)  + EPROP(VIRKE)**2
       EPRP2A(VOLUME) = EPRP2A(VOLUME) + EPROP(VOLUME)**2
       DO I = 1,LENENV
          EPRSA(I)  = EPRSA(I)  + EPRESS(I)
          EPRS2A(I) = EPRS2A(I) + EPRESS(I)**2
       ENDDO
    ENDIF
    !
    RETURN
  END SUBROUTINE GETPRS

  SUBROUTINE GETVOL(VOLUME)
    !-----------------------------------------------------------------------
    !     Calculate the volume of the system which is necessary for
    !     calculation of the pressure. The volume units are cubic
    !     Angstroms.
    !
    use chm_kinds
    use dimens_fcm
    use image
    use number
    implicit none
    !     . Passed variables.
    real(chm_real)   VOLUME
    !     . Local variables.
    !     . Initialisation.
    !     don't change value if not a crystal calculation
    !CCC  VOLUME = ZERO
    !     . A crystal has been defined.
    IF ( XNSYMM  <=  0 ) RETURN
    !     . Calculate the updated lattice vectors.
    !RCZ 92/01/26 - XTLABC will be moved to image.f90
    !     . Calculate the volume ( A.(BXC) / XNSYMM ).
    VOLUME = &
         (XTLABC(1) * ( XTLABC(3)*XTLABC(6) - XTLABC(5)*XTLABC(5) ) + &
         XTLABC(2) * ( XTLABC(5)*XTLABC(4) - XTLABC(2)*XTLABC(6) ) + &
         XTLABC(4) * ( XTLABC(2)*XTLABC(5) - XTLABC(3)*XTLABC(4) ) ) &
         /  XNSYMM
    RETURN
  END SUBROUTINE GETVOL

  SUBROUTINE PRSEXT
    !-----------------------------------------------------------------------
    !     Calculate the pressure for a system using the external
    !     virial formula. The external virial should already be
    !     known from an energy calculation and the volume of the
    !     system must be available. The total (isotropic) pressure
    !     as well as the components of the pressure tensor are
    !     determined.
    !
    !     If the volume is zero or negative the pressure isn't
    !     calculated (i.e. the pressure is taken to be zero).
    !
    use chm_kinds
    use consta
    use dimens_fcm
    use energym
    use number
    implicit none
    !     . Local variables.
    real(chm_real)   VCELL
    !     . Check the volume.
    IF ( EPROP(VOLUME)  >  ZERO ) THEN
       VCELL = PATMOS/EPROP(VOLUME)
       !     . The pressure tensor.
       EPRESS(PEXX) = - EPRESS(VEXX) * VCELL
       EPRESS(PEXY) = - EPRESS(VEXY) * VCELL
       EPRESS(PEXZ) = - EPRESS(VEXZ) * VCELL
       EPRESS(PEYX) = - EPRESS(VEYX) * VCELL
       EPRESS(PEYY) = - EPRESS(VEYY) * VCELL
       EPRESS(PEYZ) = - EPRESS(VEYZ) * VCELL
       EPRESS(PEZX) = - EPRESS(VEZX) * VCELL
       EPRESS(PEZY) = - EPRESS(VEZY) * VCELL
       EPRESS(PEZZ) = - EPRESS(VEZZ) * VCELL
       !     . The total pressure.
       EPROP(PRESSE) = ( EPRESS(PEXX) + EPRESS(PEYY) + EPRESS(PEZZ) ) &
            / THREE
    ENDIF
    !
    RETURN
  END SUBROUTINE PRSEXT

  SUBROUTINE PRSINT(NATOM,AMASS,VX,VY,VZ,VSC,WX,WY,WZ,WSC,vpart_inout,second_call)
    !-----------------------------------------------------------------------
    !     Calculate the pressure for a system using the internal
    !     virial formula. The internal virial should already be
    !     known from an energy calculation and the volume of the
    !     system must be available. The atomic velocities are
    !     passed in. The total (isotropic) pressure as well as
    !     the components of the pressure tensor are determined.
    !
    !     If the volume is zero or negative the pressure isn't
    !     calculated (i.e. the pressure is taken to be zero).
    !
    !     APH: Changes made to reduce the number of gcomb calls in
    !          dynamc()
    !     optional input/output variables added:
    !     vpart_inout(*) is the output/input vpart 
    !     if second_call == .true., only does the last part of the routine
    !
    use chm_kinds
    use consta
    use dimens_fcm
    use number
    !
    use energym
    use reawri
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:natoml,atoml,q_domdec 
#endif
    implicit none
    !     . Passed variables.
    INTEGER  NATOM
    real(chm_real) AMASS(NATOM), VX(NATOM), VY(NATOM), VZ(NATOM), VSC
    real(chm_real)               WX(NATOM), WY(NATOM), WZ(NATOM), WSC
    real(chm_real), optional :: vpart_inout(*)
    logical, optional :: second_call
    !     . Local variables.
    INTEGER  I
    real(chm_real) VCELL, VPART(6), SCR, SCDEL
#if KEY_DOMDEC==1
    real(chm_real) vpart_tmp(6) 
#endif
    integer i00, i01, ia

    !     . Check the volume.
    IF ( EPROP(VOLUME)  <=  ZERO ) RETURN

    VCELL = PATMOS/EPROP(VOLUME)

    if (present(second_call)) then
       if (second_call) then
          vpart(1:6) = vpart_inout(1:6)
          goto 121
       endif
    endif

    !     . Initialisation.
    DO I=1,6
       VPART(I)=0
    ENDDO
    !     . Accumulation.
    IF(VSC > ZERO) THEN
       SCDEL=ONE/VSC**2

#if KEY_DOMDEC==1
       if (q_domdec) then
          i00 = 1
          i01 = natoml
       else
#endif 
          i00=1
          i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
          I00=1+IPARPT(MYNOD)
          i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
       endif  
#endif

#if KEY_DOMDEC==1
       if (q_domdec) then
!$omp parallel do schedule(static) private(ia, i, scr) reduction(+:vpart)
          do ia=i00,i01
             i = atoml(ia)
             SCR = AMASS(I)*SCDEL
             VPART(RPXX) = VPART(RPXX) + SCR * VX(I) * VX(I)
             VPART(RPXY) = VPART(RPXY) + SCR * VX(I) * VY(I)
             VPART(RPXZ) = VPART(RPXZ) + SCR * VX(I) * VZ(I)
             VPART(RPYY) = VPART(RPYY) + SCR * VY(I) * VY(I)
             VPART(RPYZ) = VPART(RPYZ) + SCR * VY(I) * VZ(I)
             VPART(RPZZ) = VPART(RPZZ) + SCR * VZ(I) * VZ(I)
          ENDDO
!$omp end parallel do
       else
#endif 
          do i=i00,i01
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN  
#endif
                SCR = AMASS(I)*SCDEL
                VPART(RPXX) = VPART(RPXX) + SCR * VX(I) * VX(I)
                VPART(RPXY) = VPART(RPXY) + SCR * VX(I) * VY(I)
                VPART(RPXZ) = VPART(RPXZ) + SCR * VX(I) * VZ(I)
                VPART(RPYY) = VPART(RPYY) + SCR * VY(I) * VY(I)
                VPART(RPYZ) = VPART(RPYZ) + SCR * VY(I) * VZ(I)
                VPART(RPZZ) = VPART(RPZZ) + SCR * VZ(I) * VZ(I)
#if KEY_PARASCAL==1
             ENDIF  
#endif
          enddo
#if KEY_DOMDEC==1
       endif  
#endif
!!$       do ia=i00,i01
!!$##IF DOMDEC
!!$          if (q_domdec) then
!!$             i = atoml(ia)
!!$          else
!!$##ENDIF
!!$             i = ia
#if KEY_DOMDEC==1
!!$          endif  
#endif
#if KEY_PARASCAL==1
!!$          IF(JPBLOCK(I) == MYNOD) THEN  
#endif
!!$             SCR = AMASS(I)*SCDEL
!!$             VPART(RPXX) = VPART(RPXX) + SCR * VX(I) * VX(I)
!!$             VPART(RPXY) = VPART(RPXY) + SCR * VX(I) * VY(I)
!!$             VPART(RPXZ) = VPART(RPXZ) + SCR * VX(I) * VZ(I)
!!$             VPART(RPYY) = VPART(RPYY) + SCR * VY(I) * VY(I)
!!$             VPART(RPYZ) = VPART(RPYZ) + SCR * VY(I) * VZ(I)
!!$             VPART(RPZZ) = VPART(RPZZ) + SCR * VZ(I) * VZ(I)
#if KEY_PARASCAL==1
!!$          ENDIF  
#endif
!!$       ENDDO
    ENDIF
    !
    IF(WSC > ZERO) THEN
       SCDEL=ONE/WSC**2

#if KEY_DOMDEC==1
       if (q_domdec) then
          i00 = 1
          i01 = natoml
       else
#endif 
          i00=1
          i01=natom
#if KEY_PARALLEL==1
#if KEY_PARAFULL==1
          I00=1+IPARPT(MYNOD)
          i01=IPARPT(MYNODP)
#endif 
#endif 
#if KEY_DOMDEC==1
       endif  
#endif

#if KEY_DOMDEC==1
       if (q_domdec) then
          ! Create temporary variable for OpenMP reduction to work correctly
          vpart_tmp = zero
!$omp parallel do schedule(static) private(ia, i, scr) reduction(+:vpart_tmp)
          do ia=i00,i01
             i = atoml(ia)
             scr = amass(i)*scdel
             vpart_tmp(rpxx) = vpart_tmp(rpxx) + scr * wx(i) * wx(i)
             vpart_tmp(rpxy) = vpart_tmp(rpxy) + scr * wx(i) * wy(i)
             vpart_tmp(rpxz) = vpart_tmp(rpxz) + scr * wx(i) * wz(i)
             vpart_tmp(rpyy) = vpart_tmp(rpyy) + scr * wy(i) * wy(i)
             vpart_tmp(rpyz) = vpart_tmp(rpyz) + scr * wy(i) * wz(i)
             vpart_tmp(rpzz) = vpart_tmp(rpzz) + scr * wz(i) * wz(i)
          enddo
!$omp end parallel do
          vpart = vpart + vpart_tmp
       else
#endif 
          do i=i00,i01
#if KEY_PARASCAL==1
             IF(JPBLOCK(I) == MYNOD) THEN  
#endif
                SCR = AMASS(I)*SCDEL
                VPART(RPXX) = VPART(RPXX) + SCR * WX(I) * WX(I)
                VPART(RPXY) = VPART(RPXY) + SCR * WX(I) * WY(I)
                VPART(RPXZ) = VPART(RPXZ) + SCR * WX(I) * WZ(I)
                VPART(RPYY) = VPART(RPYY) + SCR * WY(I) * WY(I)
                VPART(RPYZ) = VPART(RPYZ) + SCR * WY(I) * WZ(I)
                VPART(RPZZ) = VPART(RPZZ) + SCR * WZ(I) * WZ(I)
#if KEY_PARASCAL==1
             ENDIF  
#endif
          enddo
#if KEY_DOMDEC==1
       endif  
#endif
!!$       do ia=i00,i01
!!$##IF DOMDEC
!!$          if (q_domdec) then
!!$             i = atoml(ia)
!!$          else
!!$##ENDIF
!!$             i = ia
#if KEY_DOMDEC==1
!!$          endif  
#endif
#if KEY_PARASCAL==1
!!$          IF(JPBLOCK(I) == MYNOD) THEN  
#endif
!!$             SCR = AMASS(I)*SCDEL
!!$             VPART(RPXX) = VPART(RPXX) + SCR * WX(I) * WX(I)
!!$             VPART(RPXY) = VPART(RPXY) + SCR * WX(I) * WY(I)
!!$             VPART(RPXZ) = VPART(RPXZ) + SCR * WX(I) * WZ(I)
!!$             VPART(RPYY) = VPART(RPYY) + SCR * WY(I) * WY(I)
!!$             VPART(RPYZ) = VPART(RPYZ) + SCR * WY(I) * WZ(I)
!!$             VPART(RPZZ) = VPART(RPZZ) + SCR * WZ(I) * WZ(I)
#if KEY_PARASCAL==1
!!$          ENDIF  
#endif
!!$       enddo

       DO I=1,6
          VPART(I)=VPART(I)*HALF
       ENDDO
    ENDIF

#if KEY_PARALLEL==1
#if KEY_DOMDEC==1
  if (q_domdec .and. present(vpart_inout)) then
     vpart_inout(1:6) = vpart(1:6)
     return
  else
#endif 
    CALL GCOMB(VPART,6)
#if KEY_DOMDEC==1
  endif  
#endif
#endif 
    !
    !     . The pressure tensor.
121 continue
    EPRESS(PIXX) = ( VPART(RPXX) + EPRESS(VIXX) ) * VCELL
    EPRESS(PIXY) = ( VPART(RPXY) + EPRESS(VIXY) ) * VCELL
    EPRESS(PIXZ) = ( VPART(RPXZ) + EPRESS(VIXZ) ) * VCELL
    EPRESS(PIYX) = ( VPART(RPXY) + EPRESS(VIYX) ) * VCELL
    EPRESS(PIYY) = ( VPART(RPYY) + EPRESS(VIYY) ) * VCELL
    EPRESS(PIYZ) = ( VPART(RPYZ) + EPRESS(VIYZ) ) * VCELL
    EPRESS(PIZX) = ( VPART(RPXZ) + EPRESS(VIZX) ) * VCELL
    EPRESS(PIZY) = ( VPART(RPYZ) + EPRESS(VIZY) ) * VCELL
    EPRESS(PIZZ) = ( VPART(RPZZ) + EPRESS(VIZZ) ) * VCELL
    !     . The total pressure.
    EPROP(PRESSI) = ( EPRESS(PIXX) + EPRESS(PIYY) + EPRESS(PIZZ) ) &
         / THREE
    !
    RETURN
  END SUBROUTINE PRSINT

  SUBROUTINE VIRTOT(VPRSSI,VPRSSE,NATOM, X, Y, Z, DX, DY, DZ)
    !-----------------------------------------------------------------------
    !     Calculate the external virial for a system. It is defined
    !     simply as the dot product of the external forces with the
    !     atomic positions. Here the calculation is more complicated
    !     as each tensor component of the virial is determined. The
    !     input derivative arrays contain the total derivatives
    !     and so the external virial is calculated by subtracting
    !     the internal virial from the results of the outer product.
    !
    use chm_kinds
    use dimens_fcm
    use number
    use energym
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,groupl,zonelist  
#endif
    implicit none
    !     . Passed variables.
    real(chm_real)   VPRSSI(9),VPRSSE(9)
    INTEGER  NATOM
    real(chm_real)   X(NATOM),   Y(NATOM),  Z(NATOM), &
         DX(NATOM), DY(NATOM), DZ(NATOM)
#if KEY_PARALLEL==1
    INTEGER IS,IQ
#endif 
    !      . Local variables.
    INTEGER  I
    real(chm_real)   VXX, VXY, VXZ, VYX, VYY, VYZ, VZX, VZY, VZZ
    real(chm_real) VPROP,VPRESS(9),WPROP,WPRESS(9)
    !
    !      compute external virial
#if KEY_DOMDEC==1
    if (q_domdec) then
       ! NOTE: external pressure calculation is supressed for now
       eprop(vire) = zero
       eprop(virke) = zero
       vprsse = zero
    else
#endif 
#if KEY_PARALLEL==1
       IS=IPARPT(MYNOD)+1
       IQ=IPARPT(MYNODP)
       CALL VIRAL(VPROP,VPRESS,IS,IQ,X,Y,Z,DX,DY,DZ)
#else /**/
       CALL VIRAL(VPROP,VPRESS,1,NATOM,X,Y,Z,DX,DY,DZ)
#endif 
       DO I=1,9
          VPRSSE(I)=VPRESS(I)-VPRSSI(I)
       ENDDO
       EPROP(VIRE)=VPROP-EPROP(VIRI)
       EPROP(VIRKE)=-THREE*VPROP/TWO
#if KEY_DOMDEC==1
    endif  
#endif
    !
    RETURN
  END SUBROUTINE VIRTOT

  SUBROUTINE VIRSHK(VPRSSI,NATOM, X, Y, Z, DX, DY, DZ, vpress_out, q_calc_virial)
    !-----------------------------------------------------------------------
    !     Add in the shake contribution to the internal virial
    !
    use chm_kinds
    use dimens_fcm
    use number
    use energym
    use parallel
#if KEY_DOMDEC==1
    use domdec_common,only:q_domdec,groupl,zonelist
    use enbxfast,only:calc_virial
#endif 
    implicit none
    !     . Passed variables.
    real(chm_real)   VPRSSI(9),VPRSSE(9)
    INTEGER  NATOM
    real(chm_real)   X(*),   Y(*),  Z(*), &
         DX(*),  DY(*), DZ(*)
    real(chm_real), optional :: vpress_out(*)
    logical, optional :: q_calc_virial
#if KEY_PARALLEL==1
    INTEGER IS,IQ
#endif 
    !     . Local variables.
    INTEGER  I
    real(chm_real)   VXX, VXY, VXZ, VYX, VYY, VYZ, VZX, VZY, VZZ
    real(chm_real) VPROP,VPRESS(9)
    logical do_calc_virial
    !
    !     compute SHAKE virial

#if KEY_PARALLEL==1 /*parallel*/
#if KEY_DOMDEC==1 /*domdec*/
    if (q_domdec) then
       ! If one of the optional parameters is present, the other one must be also!
       if ((present(vpress_out) .and. .not.present(q_calc_virial)) .or. &
            (.not.present(vpress_out) .and. present(q_calc_virial))) then
          call wrndie(-5,'<prssre>','Invalid use of virshk in DOMDEC (1)')
       endif
       do_calc_virial = .true.
       if (present(q_calc_virial)) then
          do_calc_virial = q_calc_virial
       endif
       if (do_calc_virial) then
          call calc_virial(vprop, vpress, x, y, z, dx, dy, dz, groupl, zonelist(1), .false.)
          if (present(vpress_out)) then
             vpress_out(1:9) = vpress(1:9)
             return
          endif
          CALL GCOMB(VPRESS,9)
       else
          ! This means that virial has already been calculated and combined outside
          ! this routine => vpress_out must be present!
          if (.not.present(vpress_out)) then
             call wrndie(-5,'<prssre>','Invalid use of virshk in DOMDEC (2)')
          endif
          vpress(1:9) = vpress_out(1:9)
       endif
       VPROP = (VPRESS(1)+VPRESS(5)+VPRESS(9))/THREE
    else
#endif /* (domdec)*/
       IS=IPARPT(MYNOD)+1
       IQ=IPARPT(MYNODP)
       CALL VIRAL(VPROP,VPRESS,IS,IQ,X,Y,Z,DX,DY,DZ)
       CALL GCOMB(VPRESS,9)
       VPROP = (VPRESS(1)+VPRESS(5)+VPRESS(9))/THREE
#if KEY_DOMDEC==1
    endif  
#endif
#else /* (parallel)*/
    CALL VIRAL(VPROP,VPRESS,1,NATOM,X,Y,Z,DX,DY,DZ)
#endif /* (parallel)*/
    DO I=1,9
       VPRSSI(I)=VPRESS(I)+VPRSSI(I)
    ENDDO
    EPROP(VIRI)=VPROP+EPROP(VIRI)
    !
#if KEY_DOMDEC==1
    if (q_domdec) then
       eprop(virke) = zero
    else
#endif 
       EPROP(VIRKE)=-THREE*(EPROP(VIRI)+EPROP(VIRE))/TWO
#if KEY_DOMDEC==1
    endif  
#endif
    !
    RETURN
  END SUBROUTINE VIRSHK

  SUBROUTINE VIRAL(VPROP,VPRESS,IS,IQ,X,Y,Z,DX,DY,DZ)
    !-----------------------------------------------------------------------
    !     Calculate the external virial for a system. It is defined
    !     simply as the dot product of the external forces with the
    !     atomic positions. Here the calculation is more complicated
    !     as each tensor component of the virial is determined. The
    !     input derivative arrays contain the total derivatives
    !     and so the external virial is calculated by subtracting
    !     the internal virial from the results of the outer product.
    !
    use chm_kinds
    use dimens_fcm
    use number
    use exfunc
    use psf
    implicit none
    !     . Passed variables.
    real(chm_real)   VPROP, VPRESS(9)
    INTEGER  IS,IQ
    real(chm_real)   X(*),  Y(*),  Z(*), &
         DX(*), DY(*), DZ(*)
    !     . Local variables.
    INTEGER  I, NC
    real(chm_real)   VXX, VXY, VXZ, VYX, VYY, VYZ, VZX, VZY, VZZ
    !
    !     . Initialisation.
    VXX = ZERO
    VXY = ZERO
    VXZ = ZERO
    VYX = ZERO
    VYY = ZERO
    VYZ = ZERO
    VZX = ZERO
    VZY = ZERO
    VZZ = ZERO
    !     . Accumulation.
    DO I=IS,IQ
       VXX = VXX + X(I) * DX(I)
       VXY = VXY + X(I) * DY(I)
       VXZ = VXZ + X(I) * DZ(I)
       VYX = VYX + Y(I) * DX(I)
       VYY = VYY + Y(I) * DY(I)
       VYZ = VYZ + Y(I) * DZ(I)
       VZX = VZX + Z(I) * DX(I)
       VZY = VZY + Z(I) * DY(I)
       VZZ = VZZ + Z(I) * DZ(I)
    ENDDO
    !     . Assignment.
    VPRESS(1) = - VXX
    VPRESS(2) = - VXY
    VPRESS(3) = - VXZ
    VPRESS(4) = - VYX
    VPRESS(5) = - VYY
    VPRESS(6) = - VYZ
    VPRESS(7) = - VZX
    VPRESS(8) = - VZY
    VPRESS(9) = - VZZ
    !     . The average virial.
    VPROP = -(VXX+VYY+VZZ)/THREE
    !
    RETURN
  END SUBROUTINE VIRAL

end module prssre

