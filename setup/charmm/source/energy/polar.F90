!CHARMM Element source/energy/polar.src $Revision: 1.2 $
module polarm
  use chm_kinds
  use dimens_fcm
  implicit none

! QPOLAR              logical flag to setup polarization model
! N_O, N_H, N_R       total number of specific atom types
! N_POL               number of atoms involved in the polarization
! I_O()               oxygen list
! I_H()               hydrogen list
! I_R()               rest (other atoms)
!
! R_EQ                equilibrium oxygen-hydrogen distance in H2O
! ALPHA               polarizability of oxygen
! COEFF
! E_OO                oxygen-oxygen interaction function
! E_OH                oxygen-hydrogen interaction function
! E_HH                hydrogen-hydrogen interaction function
! E_OR                oxygen-others interaction
! E_HR                hydrogen-others interaction
! EPOL                polarization energy
! MU()                induced dipole components
! NU()                induced dipole-like Lagrange multipliers
!
! K_KIN               path integral kinetic spring constant
! E_KIN               path integral kinetic spring energy
! NBEAD               number of hydrogen path integral beads
!
! NCYCLE              maximum number of iterations
! TOLRPOL             tolerance on minimum change of epol for iterations

#if KEY_POLAR==1
   LOGICAL QPOLAR

   INTEGER,PARAMETER :: MAXPOL=2000,MAXPO=100,MAXPH=4000
   INTEGER,PARAMETER :: DM=3*MAXPO
   INTEGER N_POL, N_O, N_H, N_R
   INTEGER I_O(MAXPO),I_H(MAXPH),I_R(MAXPOL)
   INTEGER NCYCLE
   INTEGER NBEAD
   INTEGER ISKELE

   real(chm_real) COEFFF, R_EQ, ALPHA
   real(chm_real) E_OO, E_OH, E_HH, E_OR, E_HR, EPOL
   real(chm_real) K_KIN, E_KIN
   real(chm_real) MU(DM), NU(DM)
   real(chm_real) TOLRPOL
#endif 

contains

#if KEY_POLAR==1 /*polar_subs*/

  subroutine polar_iniall()
    qpolar = .false.
    return
  end subroutine polar_iniall

  SUBROUTINE POLAR0
    !-----------------------------------------------------------------------
    ! Setup the Stillinger-David-Weber Polarization Model version PM6 (or PM1)
    ! (also support Feynman path integral simulations for hydrogens)
    !
    use number
    use comand
    use consta
    use coord
    use exclm
    use exfunc
    use param
    use psf
    use stream
    use string
    use chutil,only:atomid

    implicit none

    ! Local variables
    INTEGER I, IBEAD, IATOM, ISEG, IRES
    real(chm_real) TEMP, KBT, MASS, LAMBDA
    CHARACTER(len=8)  SGID,RID,REN,AC
    LOGICAL INIT,SCALE
    LOGICAL VERB
    !     real(chm_real) Q
    !     real(chm_real) E1,DE1,DDE1,E2,DE2,DDE2,VOFF

    ! Initialization of constants
    QPOLAR = .FALSE.
    N_POL  = 0
    N_O    = 0
    N_H    = 0
    N_R    = 0
    COEFFF = 332.1669D0  !must use PM6 conversion constant
    R_EQ   = 0.9584D0
    ALPHA  = 1.444D0

    if (prnlev >= 2) then
       WRITE(OUTU,*)
       WRITE(OUTU,*) '============================'
       WRITE(OUTU,*) 'SETUP THE POLARIZATION MODEL'
       WRITE(OUTU,*) '============================'
       WRITE(OUTU,*)
    endif
    !
    ! Initialization of constants
    NCYCLE  = GTRMI(COMLYN,COMLEN,'NCYC',100)
    TOLRPOL = 1.0D-16
    TOLRPOL = GTRMF(COMLYN,COMLEN,'TOLR',TOLRPOL)
    if (prnlev >= 2) then    
       WRITE(OUTU,*) 'NCYCLE  = ',NCYCLE
       WRITE(OUTU,*) 'TOLRPOL = ',TOLRPOL
    endif
    VERB    = INDXA(COMLYN,COMLEN,'VERB') .GT. 0

    ! Skip PM6-REST electrostatics interactions from atom 1 to SKIPE
    ISKELE   = GTRMI(COMLYN,COMLEN,'SKIPE',0)

    ! Path integral constants
    TEMP    = GTRMF(COMLYN,COMLEN,'TEMP',THRHUN)
    MASS    = GTRMF(COMLYN,COMLEN,'MASS',ONE)
    NBEAD   = GTRMI(COMLYN,COMLEN,'NBEA',1)
    INIT    = INDXA(COMLYN,COMLEN,'INIT') .GT. 0
    SCALE   = INDXA(COMLYN,COMLEN,'SCAL') .GT. 0

    IF(NBEAD.GT.1)THEN
       LAMBDA = (HBAR/SQRT(JKBOLTZ*TEMP*MASS*AMU))/ANGSTROM
       KBT    = (JKBOLTZ*TEMP)/KCALMOL
       K_KIN  = NBEAD*KBT/LAMBDA**2
       if (prnlev >= 2) then
          WRITE(OUTU,*) 'NBEAD   = ',NBEAD
          WRITE(OUTU,*) 'TEMP    = ',TEMP
          WRITE(OUTU,*) 'MASS    = ',MASS
          WRITE(OUTU,*) 'Lambda at ',TEMP,'K  = ',LAMBDA,' [ang]'
          WRITE(OUTU,*) 'Spring constant = ',K_KIN,' [kcal/mol/angs**2]'
       endif
    ENDIF

    !  Exclude nonbond interactions among the selected segments
    !  by removing them from the nonbonded list
    LEXS=.TRUE.
    IF(.NOT.ALLOCATED(SLIST)) CALL ALLOCATE_EXCL_LTM
    CALL RLIST(COMLYN,COMLEN,NEXS,SLIST,MAXSEG)
    if (prnlev >= 2) then
       WRITE(OUTU,104) 'Nonbonded interactions involving',NEXS, &
            ' segments will be excluded (see EXSG in energy.doc)'
104    FORMAT(1X,A,I3,A)
       WRITE(OUTU,105) 'Selected segments:',(SLIST(I),I=1,NEXS)
105    FORMAT(10(1X,A))
       WRITE(OUTU,*)
    endif

    if (prnlev >= 2) WRITE(OUTU,*) 'Making the lists'
    loop10: DO ISEG=1,NSEG
       if (prnlev >= 2) then
          IF(VERB) WRITE(OUTU,'(1X,2A)') '--SEGMENT-- ',SEGID(ISEG)
       endif

       IF(SEGID(ISEG).EQ.SLIST(1))THEN
          if (prnlev >= 2) then
             IF(VERB) WRITE(OUTU,*) 'POLAR:'
          endif
          loop11: DO IRES=NICTOT(ISEG)+1,NICTOT(ISEG+1)

             IF(RES(IRES)(1:1).EQ.'O')THEN
                N_O=N_O+1
                IF(N_O.GT.MAXPO)THEN
                   CALL WRNDIE(-4,'<POLAR>','N_O EXCEEDING MAXPO')
                ENDIF

                if (prnlev >= 2) then
                   IF(VERB) WRITE(OUTU,*) 'OPOL:',IBASE(IRES)+1,IBASE(IRES+1)
                endif
                DO I=IBASE(IRES)+1,IBASE(IRES+1)
                   CALL ATOMID(I,SGID,RID,REN,AC)
                   if (prnlev >= 2) then
                      IF(VERB) WRITE(OUTU,*) N_O,I,  ' IAC  =',ATC(IAC(I)), &
                           ' TYPE =',AC(1:idleng),IRES,' RES  =',RES(IRES)(1:idleng)
                   endif
                   I_O(N_O)=I
                ENDDO
             ELSEIF(RES(IRES)(1:1).EQ.'H')THEN
                N_H=N_H+1
                IF(NBEAD*N_H.GT.MAXPH)THEN
                   CALL WRNDIE(-4,'<POLAR>','N_H EXCEEDING MAXPH')
                ENDIF
                IBEAD=0
                if (prnlev >= 2) then
                   IF(VERB) WRITE(OUTU,*) 'HPOL:',IBASE(IRES)+1,IBASE(IRES+1)
                endif
                IF(NBEAD.NE.(IBASE(IRES+1)-IBASE(IRES)))THEN
                   WRITE(OUTU,*) 'IBASE:',IBASE(IRES+1)-IBASE(IRES)
                   WRITE(OUTU,*) 'NBEAD:',NBEAD
                   CALL WRNDIE(-4,'<POLAR>','INCONSISTENT NBEAD')
                ENDIF
                DO I=IBASE(IRES)+1,IBASE(IRES+1)
                   IBEAD=IBEAD+1
                   CALL ATOMID(I,SGID,RID,REN,AC)
                   if (prnlev >= 2) then
                      IF(VERB) WRITE(OUTU,*) N_H,IBEAD,I, &
                           ' IAC  =',ATC(IAC(I)),' TYPE =',ATYPE(I)(1:idleng), &
                           IRES,' RES  =',RES(IRES)(1:idleng)
                   endif
                   I_H(NBEAD*(N_H-1)+IBEAD)=I
                ENDDO
             ELSE
                CALL WRNDIE(-1,'<POLAR>','UNKNOWN POLAR ATOM')
             ENDIF
          ENDDO loop11

       ELSE
          if (prnlev >= 2) then
             IF(VERB) WRITE(OUTU,*) 'OTHERS:'
          endif
          DO IRES=NICTOT(ISEG)+1,NICTOT(ISEG+1)
             DO I=IBASE(IRES)+1,IBASE(IRES+1)
                N_R=N_R+1
                IF((N_R).GT.MAXPOL)THEN
                   CALL WRNDIE(-4,'<POLAR>','N_R EXCEEDING MAXPOL')
                ENDIF
                I_R(N_R)=I
                CALL ATOMID(I,SGID,RID,REN,AC)
                if (prnlev >= 2) then
                   IF(VERB) WRITE(OUTU,*) N_R,I,' IAC  =',ATC(IAC(I)), &
                        ' TYPE =',ATYPE(I)(1:idleng),IRES, &
                        ' RES  =',RES(IRES)(1:idleng)
                endif
             ENDDO
          ENDDO
       ENDIF
    ENDDO loop10

    N_POL=N_O+N_H
    QPOLAR = N_POL .NE. 0

    if (prnlev >= 2) then
       WRITE(OUTU,'(4(A,I0))') 'N_POL=',N_POL,' N_O=',N_O,' N_H=',N_H,' N_R=',N_R
    endif
    IF((N_O+N_H+N_R).GT.MAXPOL)THEN
       CALL WRNDIE(-4,'<POLAR>','N_O+N_H+N_R EXCEEDING MAXPOL')
    ENDIF

    DO I=1,N_O
       IATOM=I_O(I)
       if (prnlev >= 2) then
          IF(VERB) WRITE(OUTU,*) 'O: ',ATYPE(IATOM)(1:idleng), &
               ATC(IAC(IATOM)),I,1, IATOM,AMASS(IATOM)
       endif
       MU(I)=0.0D0
       NU(I)=0.0D0
    ENDDO

    IF(INIT.AND.IBEAD.GT.1)THEN
       if (prnlev >= 2) then
          WRITE(OUTU,*) 'Coordinates of the PI beads initialized'
       endif
    ENDIF
    IF(SCALE)THEN
       if (prnlev >= 2) then
          WRITE(OUTU,*) 'bead mass scaled'
       endif
    ENDIF

    DO I=1,N_H
       DO IBEAD=1,NBEAD
          IATOM=I_H(NBEAD*(I-1)+IBEAD)
          IF(SCALE) AMASS(IATOM)=AMASS(IATOM)*NBEAD
          if (prnlev >= 2) then
             IF(VERB) WRITE(OUTU,*) 'H: ',ATYPE(IATOM)(1:idleng), &
                  ATC(IAC(IATOM)),I ,IBEAD, IATOM,AMASS(IATOM)
          endif
          IF(INIT.AND.IBEAD.GT.1)THEN
             X(IATOM)=X(I_H(NBEAD*(I-1)+1))
             Y(IATOM)=Y(I_H(NBEAD*(I-1)+1))
             Z(IATOM)=Z(I_H(NBEAD*(I-1)+1))
          ENDIF
       ENDDO
    ENDDO

    DO I=1,N_R
       IATOM=I_R(I)
       if (prnlev >= 2) then
          IF(VERB) WRITE(OUTU,*) 'R: ',ATYPE(IATOM)(1:idleng), &
               ATC(IAC(IATOM)),I,1, IATOM,AMASS(IATOM)
       endif
    ENDDO

    RETURN
  END SUBROUTINE POLAR0

  SUBROUTINE POLAR1(EPOLAR,NATOM,X,Y,Z,FX,FY,FZ,    &
       NATC,IAC,ITC,CNBA,CNBB,CG)
    !
    !     Main subroutine to compute the energy for Stillinger-David PM6 waters
    !     (also support path integral simulations)
    !
    ! Passing variables:
    ! epolar              energy of the polarization model
    ! natom               total number of atom in the system
    ! x,y,z               coordinates of the atoms
    ! fx,fy,fz            cartesian gradiant derivatives (=-forces)
    ! natc                number of atom type (see param.f90)
    ! iac()               pointer for atom types
    ! itc()               pointer for atom types counter
    ! cnba()              VDW Rmin{i,j}**2
    ! cnbb()              VDW well depth
    ! cg()                atomic partial charges
    !
    use stream
    implicit none
    real(chm_real) EPOLAR
    INTEGER NATOM
    real(chm_real) X(*),Y(*),Z(*)
    real(chm_real) FX(*),FY(*),FZ(*)
    INTEGER NATC
    INTEGER IAC(*),ITC(*)
    real(chm_real)  CNBA(*),CNBB(*)
    real(chm_real)  CG(*)
    ! --------------------------------------------------------------
    ! Local variables
    ! ipoint(,)           pointer for cnba() and cnbb()
    ! G1, G2              modified fields
    ! Tdd                 dipole-dipole matrix
    ! kkk                 damping function  K
    ! lll                 damping function  L
    ! dkkk                derivative of damping function  K
    ! dlll                derivative of damping function  L
    ! dx,dy,dz,dr         distances and unit vectors
    ! G1_h, G2_h          modified fields from each bead of the hydrogens
    !
    INTEGER IPOINT(NATC,NATC)
    real(chm_real) G1(DM), G2(DM), TDD(DM,DM)
    real(chm_real) KKK(MAXPO,MAXPOL),DKKK(MAXPO,MAXPOL)
    real(chm_real) LLL(MAXPO,MAXPOL),DLLL(MAXPO,MAXPOL)
    real(chm_real) DX(MAXPO,MAXPOL),DY(MAXPO,MAXPOL),DZ(MAXPO,MAXPOL)
    real(chm_real) DR(MAXPO,MAXPOL)
    real(chm_real) G1_H(DM), G2_H(DM)
    real(chm_real) GTMP,G1TEMP(3),G2TEMP(3)
    real(chm_real) FIJ,EIJ,FTMP(3),AKKK,ALLL
    real(chm_real) MUI_DR,MUJ_DR,NUI_DR,NUJ_DR,MUI_NUJ,MUJ_NUI
    real(chm_real) DX12,DY12,DZ12,DR12,DR2,DR3,DR4
    real(chm_real) E_OO12, E_OH12, E_HH12, E_OR12, E_HR12, EPOL2, DIFF
    real(chm_real) DE_OO12, DE_OH12, DE_HH12, DE_OR12, DE_HR12
    real(chm_real) SIG2,SIG6,SIG12
    real(chm_real) D1,D2
    INTEGER I,J,I1,J1,I2,J2,I3,J3,L,K,K1,K2,IP,JP,IC,ICYCLE,IBEAD
    !
    ! in the matrices like KKK(I,J1) the variables are stored as
    !     I  = 1 to N_O and referes only to oxygens OPOL
    !     J1 = 1 to N_O, N_O+1 to N_O+N_H, N_O+N_H+1 to N_O+N_H+N_R and
    !          referes to all atom
    ! ----------------------------------------------------------------------
    !
    ! This subroutine sets up and solve the system
    !     MU() = alpha*G1() - alpha*TDD(,)*MU()
    !     EPOLAR = <E_OO + E_OR + E_OH + E_HH E_HR - 0.5 MU()*G2()>
    !     where <> means a path integral average = 1/nbead * Sum_ibead ...
    EPOLAR=0.0D0
    E_OO=0.0D0
    E_OR=0.0D0

    !     WRITE(OUTU,*) 'SUBROUTINE POLAR1'
    !     WRITE(OUTU,*) 'NATOM=',NATOM,' N_O=',N_O,' N_H=',N_H
    !     DO 120 I=1,N_O
    !     WRITE(OUTU,*) 'O ', I_O(I), CG(I_O(I))
    ! 120 CONTINUE
    !     DO 121 I=1,N_R
    !     WRITE(OUTU,*) 'R ', I_R(I), CG(I_R(I))
    ! 121 CONTINUE
    !     DO 122 I=1,N_H
    !     DO 122 IBEAD=1,NBEAD
    !     I2=I_H(NBEAD*(I-1)+IBEAD)
    !     WRITE(OUTU,*) 'H ',I,IBEAD,I2,CG(I2)
    ! 122 CONTINUE
    !     IF(.TRUE.) STOP

    !     Initialize the code look up pointer
    !
    DO I=1,NATC
       DO J=1,I
          IPOINT(I,J)=J+((I-1)*I)/2
          IPOINT(J,I)=IPOINT(I,J)
       ENDDO
    ENDDO

    ! Initialization of fields and dipole matrix
    DO I=1,3*N_O
       G1(I)=0.0D0
       G2(I)=0.0D0
       DO J=1,3*N_O
          TDD(I,J)=0.0D0
       ENDDO
    ENDDO

    ! OXYGEN-OXYGEN PAIRS
    loop20: DO I=1,N_O
       I2=I_O(I)
       IP=3*(I-1)
       I3=ITC(IAC(I2))

       loop21: DO J=1,I-1
          J2=I_O(J)
          JP=3*(J-1)
          J3=ITC(IAC(J2))

          ! Distance and damping function
          CALL DIST_IJ(X(I2),Y(I2),Z(I2), X(J2),Y(J2),Z(J2), &
               DX(I,J),DY(I,J),DZ(I,J),DR(I,J),DR2,DR3,DR4)
          CALL DAMPING(KKK(I,J),LLL(I,J),DKKK(I,J),DLLL(I,J), &
               DR(I,J),DR2,DR3,DR4)
          IF(PRNLEV.GE.6) THEN
             WRITE(OUTU,*) 'OXYGEN-OXYGEN '
             WRITE(OUTU,*) 'KKK,LLL ',I2,J2,DR(I,J),KKK(I,J),LLL(I,J)
             WRITE(OUTU,*) 'DKKK,DLLL ',DKKK(I,J),DLLL(I,J)
          ENDIF
          ! Dipole-dipole matrix
          AKKK=ALPHA*KKK(I,J)
          TDD(IP+1,JP+1)=-AKKK*(1.0D0-3.0D0*DX(I,J)*DX(I,J))/DR3 !XX
          TDD(IP+2,JP+2)=-AKKK*(1.0D0-3.0D0*DY(I,J)*DY(I,J))/DR3 !YY
          TDD(IP+3,JP+3)=-AKKK*(1.0D0-3.0D0*DZ(I,J)*DZ(I,J))/DR3 !ZZ
          TDD(IP+1,JP+2)=-AKKK*(-3.0D0*DX(I,J)*DY(I,J))/DR3      !XY
          TDD(IP+1,JP+3)=-AKKK*(-3.0D0*DX(I,J)*DZ(I,J))/DR3      !XZ
          TDD(IP+2,JP+3)=-AKKK*(-3.0D0*DY(I,J)*DZ(I,J))/DR3      !YZ

          ! symmetrize the ip,jp bloc matrix
          TDD(IP+2,JP+1)=TDD(IP+1,JP+2)     !YX
          TDD(IP+3,JP+1)=TDD(IP+1,JP+3)     !ZX
          TDD(IP+3,JP+2)=TDD(IP+2,JP+3)     !ZY

          !  symmetrize the matrix
          DO L=1,3
             DO K=1,3
                TDD(JP+L,IP+K)=TDD(IP+K,JP+L)
             ENDDO
          ENDDO

          ! G1 and G2 fields on the oxygens
          GTMP=ALPHA*KKK(I,J)/DR2
          G1TEMP(1)=GTMP*DX(I,J)
          G1TEMP(2)=GTMP*DY(I,J)
          G1TEMP(3)=GTMP*DZ(I,J)
          GTMP=ALPHA*LLL(I,J)/DR2
          G2TEMP(1)=GTMP*DX(I,J)
          G2TEMP(2)=GTMP*DY(I,J)
          G2TEMP(3)=GTMP*DZ(I,J)
          ! G1 field for the dipoles
          G1(IP+1)=G1(IP+1)+G1TEMP(1)*CG(J2)
          G1(IP+2)=G1(IP+2)+G1TEMP(2)*CG(J2)
          G1(IP+3)=G1(IP+3)+G1TEMP(3)*CG(J2)
          G1(JP+1)=G1(JP+1)-G1TEMP(1)*CG(I2)
          G1(JP+2)=G1(JP+2)-G1TEMP(2)*CG(I2)
          G1(JP+3)=G1(JP+3)-G1TEMP(3)*CG(I2)
          !     WRITE(OUTU,*) I,' G1TEMP ',G1TEMP(1),G1TEMP(2),G1TEMP(3)
          ! G2 field for the energy and force
          G2(IP+1)=G2(IP+1)+G2TEMP(1)*CG(J2)
          G2(IP+2)=G2(IP+2)+G2TEMP(2)*CG(J2)
          G2(IP+3)=G2(IP+3)+G2TEMP(3)*CG(J2)
          G2(JP+1)=G2(JP+1)-G2TEMP(1)*CG(I2)
          G2(JP+2)=G2(JP+2)-G2TEMP(2)*CG(I2)
          G2(JP+3)=G2(JP+3)-G2TEMP(3)*CG(I2)
          !     WRITE(OUTU,*) I,' G2TEMP ',G2TEMP(1),G2TEMP(2),G2TEMP(3)

          ! Interaction
          CALL PHI_OO(E_OO12,DE_OO12,DR(I,J),DR2)
          ! add the coulomb interaction
          E_OO12=E_OO12+COEFFF*CG(I2)*CG(J2)/DR(I,J)
          DE_OO12=DE_OO12-COEFFF*CG(I2)*CG(J2)/DR2
          ! Should we add 1/r**6 vdw ?  So far, no.
          !     IC=IPOINT(I3,J3)
          !     DR12=DR(I,J)
          !     SIG2=CNBA(IC)/DR2
          !     SIG6=SIG2*SIG2*SIG2
          !     SIG12=SIG6*SIG6
          !     E_OO12=E_OO12-CNBB(IC)*(SIG6+SIG6)
          !     DE_OO12=DE_OO12+CNBB(IC)*12*SIG6/DR12
          !     WRITE(6,*) 'VDW',SQRT(CNBA(IC)),CNBB(IC),DR12,
          !    $             -CNBB(IC)*(SIG6+SIG6)
          E_OO=E_OO+E_OO12
          FX(I2)=FX(I2)+DE_OO12*DX(I,J)
          FY(I2)=FY(I2)+DE_OO12*DY(I,J)
          FZ(I2)=FZ(I2)+DE_OO12*DZ(I,J)
          FX(J2)=FX(J2)-DE_OO12*DX(I,J)
          FY(J2)=FY(J2)-DE_OO12*DY(I,J)
          FZ(J2)=FZ(J2)-DE_OO12*DZ(I,J)
          !     WRITE(6,*) 'E_OO=',E_OO,' E_OO12=',E_OO12,I2,J2
       ENDDO loop21

       ! OXYGEN-OTHERS PAIRS
       loop23: DO J=1,N_R
          J1=N_O+N_H+J
          J2=I_R(J)

          !     WRITE(6,*) 'OXYGEN-OTHERS'
          !     WRITE(6,*) 'I,I2,J1,J2',I,I2,J1,J2
          ! Distance and damping function
          CALL DIST_IJ(X(I2),Y(I2),Z(I2), X(J2),Y(J2),Z(J2), &
               DX(I,J1),DY(I,J1),DZ(I,J1),DR(I,J1),DR2,DR3,DR4)
          IF(J2.GT.ISKELE) THEN
             CALL DAMPING(KKK(I,J1),LLL(I,J1),DKKK(I,J1),DLLL(I,J1), &
                  DR(I,J1),DR2,DR3,DR4)
             !     WRITE(OUTU,*) 'KKK,LLL ',I2,J2,DR(I,J1),KKK(I,J1),LLL(I,J1)

             ! G1 and G2 fields on the oxygens
             GTMP=ALPHA*KKK(I,J1)/DR2
             G1TEMP(1)=GTMP*DX(I,J1)
             G1TEMP(2)=GTMP*DY(I,J1)
             G1TEMP(3)=GTMP*DZ(I,J1)
             GTMP=ALPHA*LLL(I,J1)/DR2
             G2TEMP(1)=GTMP*DX(I,J1)
             G2TEMP(2)=GTMP*DY(I,J1)
             G2TEMP(3)=GTMP*DZ(I,J1)
             ! G1 field for the dipoles
             G1(IP+1)=G1(IP+1)+G1TEMP(1)*CG(J2)
             G1(IP+2)=G1(IP+2)+G1TEMP(2)*CG(J2)
             G1(IP+3)=G1(IP+3)+G1TEMP(3)*CG(J2)
             !     WRITE(6,*) I,' G1TEMP ',G1TEMP(1),G1TEMP(2),G1TEMP(3)
             ! G2 field for the energy and force
             G2(IP+1)=G2(IP+1)+G2TEMP(1)*CG(J2)
             G2(IP+2)=G2(IP+2)+G2TEMP(2)*CG(J2)
             G2(IP+3)=G2(IP+3)+G2TEMP(3)*CG(J2)
             !     WRITE(6,*) I,' G2TEMP ',G2TEMP(1),G2TEMP(2),G2TEMP(3)
          ENDIF

          ! VDW and ELEC interactions with others here
          J3=ITC(IAC(J2))
          IC=IPOINT(I3,J3)
          !     WRITE(6,101) I2,J2,' CNBA',SQRT(CNBA(IC)),' CNBB=',CNBB(IC)
          ! 101 FORMAT(1X,2I4,A,F12.4,A,F12.4)
          DR12=DR(I,J1)
          ! vdw interactions
          SIG2=CNBA(IC)/DR2
          SIG6=SIG2*SIG2*SIG2
          SIG12=SIG6*SIG6
          E_OR12=CNBB(IC)*(SIG12-SIG6-SIG6)
          DE_OR12=CNBB(IC)*12*(SIG6-SIG12)/DR12
          ! electrostatics interactions (skip elec up to ISKELE)
          IF(J2.GT.ISKELE) THEN
             E_OR12=E_OR12+COEFFF*CG(I2)*CG(J2)/DR12
             DE_OR12=DE_OR12-COEFFF*CG(I2)*CG(J2)/DR2
          ENDIF
          E_OR=E_OR+E_OR12
          FX(I2)=FX(I2)+DE_OR12*DX(I,J1)
          FY(I2)=FY(I2)+DE_OR12*DY(I,J1)
          FZ(I2)=FZ(I2)+DE_OR12*DZ(I,J1)
          FX(J2)=FX(J2)-DE_OR12*DX(I,J1)
          FY(J2)=FY(J2)-DE_OR12*DY(I,J1)
          FZ(J2)=FZ(J2)-DE_OR12*DZ(I,J1)
          !     WRITE(6,*) 'E_OR=',E_OR,' E_OR12=',E_OR12
          !     WRITE(6,*) 'DE_OR12=',DE_OR12
       ENDDO loop23
    ENDDO loop20

    ! LOOP OVER HYDROGENS BEADS
    loop40: DO IBEAD=1,NBEAD
       !     WRITE(6,*) 'IBEAD=',IBEAD
       E_OH=0.0D0
       E_HH=0.0D0
       E_HR=0.0D0
       EPOL=0.0D0
       EPOL2=0.0D0

       DO I=1,3*N_O
          G1_H(I)=0.0D0
          G2_H(I)=0.0D0
       ENDDO

       ! Oxygen-hydrogens pairs
       DO I=1,N_O
          I2=I_O(I)
          IP=3*(I-1)

          DO J=1,N_H
             J1=N_O+J                  !POINTER IN DR(,)
             J2=I_H(NBEAD*(J-1)+IBEAD) !POINTER IN X(),Y(),Z()
             !           FX(), FY(), FZ()
             ! distances
             CALL DIST_IJ(X(I2),Y(I2),Z(I2), X(J2),Y(J2),Z(J2), &
                  DX(I,J1),DY(I,J1),DZ(I,J1),DR(I,J1),DR2,DR3,DR4)
             CALL DAMPING(KKK(I,J1),LLL(I,J1),DKKK(I,J1),DLLL(I,J1), &
                  DR(I,J1),DR2,DR3,DR4)

             IF(PRNLEV.GE.6)THEN
                WRITE(OUTU,*) 'OXYGEN-HYDROGEN'
                WRITE(OUTU,*) 'KKK,LLL ',I2,J1,DR(I,J1),KKK(I,J1),LLL(I,J1)
                WRITE(OUTU,*) 'DKKK,DLLL ',DKKK(I,J1),DLLL(I,J1)
             ENDIF
             !     WRITE(6,*) 'DX=',DX(I,J1)
             !     WRITE(6,*) 'DY=',DY(I,J1)
             !     WRITE(6,*) 'DZ=',DZ(I,J1)
             ! G1 field for the induced dipoles
             GTMP=ALPHA*KKK(I,J1)*CG(J2)/DR2
             G1_H(IP+1)=G1_H(IP+1)+GTMP*DX(I,J1)
             G1_H(IP+2)=G1_H(IP+2)+GTMP*DY(I,J1)
             G1_H(IP+3)=G1_H(IP+3)+GTMP*DZ(I,J1)
             !     WRITE(6,*) I,' G1 ',G1_H(IP+1),G1_H(IP+2),G1_H(IP+3)
             ! G2 field for the energy and force
             GTMP=ALPHA*LLL(I,J1)*CG(J2)/DR2
             G2_H(IP+1)=G2_H(IP+1)+GTMP*DX(I,J1)
             G2_H(IP+2)=G2_H(IP+2)+GTMP*DY(I,J1)
             G2_H(IP+3)=G2_H(IP+3)+GTMP*DZ(I,J1)
             !     WRITE(6,*) I,' G2 ',G2_H(IP+1),G2_H(IP+2),G2_H(IP+3)

             ! add the phi-oh and Coulomb interaction
             CALL PHI_OH(E_OH12, DE_OH12, DR(I,J1), DR2)
             E_OH12=E_OH12+COEFFF*CG(I2)*CG(J2)/DR(I,J1)
             DE_OH12=DE_OH12-COEFFF*CG(I2)*CG(J2)/DR2
             E_OH=E_OH+E_OH12
             DE_OH12=DE_OH12/NBEAD
             !     WRITE(6,*) 'E_OH=',E_OH,' E_OH12=',E_OH12,I2,J1
             FX(I2)=FX(I2)+DE_OH12*DX(I,J1)
             FY(I2)=FY(I2)+DE_OH12*DY(I,J1)
             FZ(I2)=FZ(I2)+DE_OH12*DZ(I,J1)
             FX(J2)=FX(J2)-DE_OH12*DX(I,J1)
             FY(J2)=FY(J2)-DE_OH12*DY(I,J1)
             FZ(J2)=FZ(J2)-DE_OH12*DZ(I,J1)
          ENDDO
       ENDDO

       ! hydrogen-hydrogen interaction
       DO I=1,N_H
          I2=I_H(NBEAD*(I-1)+IBEAD)
          DO J=1,I-1
             J2=I_H(NBEAD*(J-1)+IBEAD)
             DX12=X(I2)-X(J2)
             DY12=Y(I2)-Y(J2)
             DZ12=Z(I2)-Z(J2)
             DR2=DX12**2+DY12**2+DZ12**2
             DR12=DSQRT(DR2)
             E_HH12=COEFFF*CG(I2)*CG(J2)/DR12
             DE_HH12=-E_HH12/DR2
             DE_HH12=DE_HH12/NBEAD
             E_HH=E_HH+E_HH12
             !     WRITE(6,*) 'E_HH=',E_HH,' E_HH12=',E_HH12,I2,J2
             !     WRITE(6,*) 'DE_HH12=',DE_HH12,DR12
             FX(I2)=FX(I2)+DE_HH12*DX12
             FY(I2)=FY(I2)+DE_HH12*DY12
             FZ(I2)=FZ(I2)+DE_HH12*DZ12
             FX(J2)=FX(J2)-DE_HH12*DX12
             FY(J2)=FY(J2)-DE_HH12*DY12
             FZ(J2)=FZ(J2)-DE_HH12*DZ12
          ENDDO
       ENDDO

       ! hydrogen-others interaction
       DO I=1,N_H
          I2=I_H(NBEAD*(I-1)+IBEAD)
          I3=ITC(IAC(I2))
          DO J=1,N_R
             J2=I_R(J)
             J3=ITC(IAC(J2))
             IC=IPOINT(I3,J3)
             !     WRITE(6,101) I2,J2,' CNBA',SQRT(CNBA(IC)),' CNBB=',CNBB(IC)
             DX12=X(I2)-X(J2)
             DY12=Y(I2)-Y(J2)
             DZ12=Z(I2)-Z(J2)
             DR2=DX12**2+DY12**2+DZ12**2
             DR12=DSQRT(DR2)
             DX12=DX12/DR12
             DY12=DY12/DR12
             DZ12=DZ12/DR12
             ! vdw interactions
             SIG2=CNBA(IC)/DR2
             SIG6=SIG2*SIG2*SIG2
             SIG12=SIG6*SIG6
             E_HR12=CNBB(IC)*(SIG12-SIG6-SIG6)
             DE_HR12=CNBB(IC)*12*(SIG6-SIG12)/DR12
             ! electrostatics interactions (skip elec up to ISKELE)
             IF(J2.GT.ISKELE) THEN
                E_HR12=E_HR12+COEFFF*CG(I2)*CG(J2)/DR12
                DE_HR12=DE_HR12-COEFFF*CG(I2)*CG(J2)/DR2
             ENDIF
             DE_HR12=DE_HR12/NBEAD
             E_HR=E_HR+E_HR12
             !     WRITE(6,*) 'E_HR=',E_HR,' E_HR12=',E_HR12
             !     WRITE(6,*) 'DE_HR12=',DE_HR12,DR12
             FX(I2)=FX(I2)+DE_HR12*DX12
             FY(I2)=FY(I2)+DE_HR12*DY12
             FZ(I2)=FZ(I2)+DE_HR12*DZ12
             FX(J2)=FX(J2)-DE_HR12*DX12
             FY(J2)=FY(J2)-DE_HR12*DY12
             FZ(J2)=FZ(J2)-DE_HR12*DZ12
          ENDDO
       ENDDO

       !     WRITE(6,*) 'FULL TDD MATRIX'
       !     DO 172 I=1,N_O
       !     DO 172 J=1,N_O
       !     IP=3*(I-1)
       !     JP=3*(J-1)
       !     WRITE(6,*) 'MATRIX TDD',I,J
       !     WRITE(6,*) TDD(IP+1,JP+1),TDD(IP+1,JP+2),TDD(IP+1,JP+3)
       !     WRITE(6,*) TDD(IP+2,JP+1),TDD(IP+2,JP+2),TDD(IP+2,JP+3)
       !     WRITE(6,*) TDD(IP+3,JP+1),TDD(IP+3,JP+2),TDD(IP+3,JP+3)
       ! 172 CONTINUE

       !     WRITE(6,*) 'FULL G1, G2 VECTORS'
       !     DO 173 I=1,3*N_O
       !     WRITE(6,*) I,' G1 =',G1(I),'  G2=',G2(I)
       ! 173 CONTINUE

       ! Self-consistent iteration loops
       !     WRITE(6,*) 'NCYCLE  ',NCYCLE,' TOLRPOL ',TOLRPOL
       loop80: DO ICYCLE=1,NCYCLE
          !     WRITE(6,*) 'CYCLE ',ICYCLE
          DO I=1,3*N_O
             MU(I)=G1(I)+G1_H(I)
             NU(I)=G2(I)+G2_H(I)
             DO J=1,3*N_O
                MU(I)=MU(I)+TDD(I,J)*MU(J)
                NU(I)=NU(I)+TDD(I,J)*NU(J)
             ENDDO
             !     WRITE(6,*) I,'MU(I)=',MU(I),' NU(I)=', NU(I)
          ENDDO

          ! Compute the polarization energy
          EPOL=0.0D0
          DO I=1,3*N_O
             EPOL=EPOL-0.5*COEFFF*MU(I)*(G2(I)+G2_H(I))/ALPHA
          ENDDO
          !     WRITE(6,*) 'EPOL=',EPOL
          DIFF=ABS(EPOL-EPOL2)
          !     WRITE(6,*) ICYCLE,'EPOLAR=',EPOLAR,DIFF
          IF(DIFF.LE.TOLRPOL) THEN
             EPOLAR=EPOLAR+(E_OO+E_OR+EPOL+E_OH+E_HH+E_HR)/NBEAD
             IF(PRNLEV.GT.5)THEN
                WRITE(OUTU,*) 'sub total ',EPOL+E_OO+E_OR+E_OH+E_HH+E_HR
                WRITE(OUTU,*) 'EPOLAR    ',EPOLAR
                WRITE(OUTU,*) 'E_OO      ',E_OO
                WRITE(OUTU,*) 'E_OR      ',E_OR
                WRITE(OUTU,*) 'E_OH      ',E_OH
                WRITE(OUTU,*) 'E_HH      ',E_HH
                WRITE(OUTU,*) 'E_HR      ',E_HR
                WRITE(OUTU,*) 'EPOL      ',EPOL
                IF(PRNLEV.GT.6)THEN
                   DO I=1,3*N_O
                      WRITE(OUTU,'(I5,4F12.5)') I,MU(I), &
                           (G2(I)+G2_H(I)),-0.5*COEFFF*MU(I)*(G2(I)+G2_H(I))/ALPHA
                   ENDDO
                ENDIF
             ENDIF
             GOTO 1000
          ENDIF
          EPOL2=EPOL
       ENDDO loop80
       EPOLAR=EPOLAR+(E_OO+E_OR+EPOL+E_OH+E_HH+E_HR)/NBEAD
       WRITE(OUTU,*) '* WARNING, CONVERGENCE NOT REACHED *'
       WRITE(OUTU,*) 'IBEAD  =',IBEAD
       WRITE(OUTU,*) 'EPOL   =',EPOL,' DIFF=',DIFF
       WRITE(OUTU,*) 'EPOLAR = ',EPOL+E_OO+E_OR+E_OH+E_HH+E_HR

       ! OXYGEN FORCE CALCULATION
       !1000 WRITE(6,*) 'FORCE CALCULATION'
1000   CONTINUE
       loop90: DO I=1,N_O
          I2=I_O(I)
          IP=3*(I-1)
          loop91: DO J=1,I-1
             J2=I_O(J)
             JP=3*(J-1)
             MUI_DR=MU(IP+1)*DX(I,J)+MU(IP+2)*DY(I,J)+MU(IP+3)*DZ(I,J)
             MUJ_DR=MU(JP+1)*DX(I,J)+MU(JP+2)*DY(I,J)+MU(JP+3)*DZ(I,J)
             NUI_DR=NU(IP+1)*DX(I,J)+NU(IP+2)*DY(I,J)+NU(IP+3)*DZ(I,J)
             NUJ_DR=NU(JP+1)*DX(I,J)+NU(JP+2)*DY(I,J)+NU(JP+3)*DZ(I,J)
             MUI_NUJ=MU(IP+1)*NU(JP+1)+MU(IP+2)*NU(JP+2)+MU(IP+3)*NU(JP+3)
             MUJ_NUI=MU(JP+1)*NU(IP+1)+MU(JP+2)*NU(IP+2)+MU(JP+3)*NU(IP+3)
             DR2=DR(I,J)**2
             DR3=DR(I,J)*DR2
             DR4=DR2*DR2
             !     WRITE(6,*) 'I,J,I2,J2',I,J,I2,J2
             !     WRITE(6,*) 'MUI_DR',MUI_DR
             !     WRITE(6,*) 'MUJ_DR',MUJ_DR
             !     WRITE(6,*) 'NUI_DR',NUI_DR
             !     WRITE(6,*) 'NUJ_DR',NUJ_DR

             ! Oxygen-Oxygen dipole-charge forces
             FIJ=-0.5*COEFFF*MUI_DR*CG(J2)*(-3*LLL(I,J)/DR3+DLLL(I,J)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*LLL(I,J)/DR3
             FTMP(1)=FIJ*DX(I,J)+MU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J)+MU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J)+MU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 1 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=-0.5*COEFFF*MUJ_DR*CG(I2)*(-3*LLL(I,J)/DR3+DLLL(I,J)/DR2)
             EIJ=-0.5*COEFFF*CG(I2)*LLL(I,J)/DR3
             FTMP(1)=-FIJ*DX(I,J)-MU(JP+1)*EIJ
             FTMP(2)=-FIJ*DY(I,J)-MU(JP+2)*EIJ
             FTMP(3)=-FIJ*DZ(I,J)-MU(JP+3)*EIJ
             !     WRITE(6,*) 'FTMP 2 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=-0.5*COEFFF*NUI_DR*CG(J2)*(-3*KKK(I,J)/DR3+DKKK(I,J)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*KKK(I,J)/DR3
             FTMP(1)=FIJ*DX(I,J)+NU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J)+NU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J)+NU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 3 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=-0.5*COEFFF*NUJ_DR*CG(I2)*(-3*KKK(I,J)/DR3+DKKK(I,J)/DR2)
             EIJ=-0.5*COEFFF*CG(I2)*KKK(I,J)/DR3
             FTMP(1)=-FIJ*DX(I,J)-NU(JP+1)*EIJ
             FTMP(2)=-FIJ*DY(I,J)-NU(JP+2)*EIJ
             FTMP(3)=-FIJ*DZ(I,J)-NU(JP+3)*EIJ
             !     WRITE(6,*) 'FTMP 4 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             ! Oxygen-oxygen dipole-dipole forces
             FIJ=0.5*COEFFF*(MUI_NUJ*(-3*KKK(I,J)/DR4+DKKK(I,J)/DR3) &
                  -3*MUI_DR*NUJ_DR*(-5*KKK(I,J)/DR4+DKKK(I,J)/DR3))
             EIJ=-0.5*COEFFF*3*KKK(I,J)/DR4
             !     WRITE(6,*) 'FIJ=',FIJ
             !     WRITE(6,*) 'EIJ=',EIJ
             FTMP(1)=FIJ*DX(I,J)+(MU(IP+1)*NUJ_DR+NU(JP+1)*MUI_DR)*EIJ
             FTMP(2)=FIJ*DY(I,J)+(MU(IP+2)*NUJ_DR+NU(JP+2)*MUI_DR)*EIJ
             FTMP(3)=FIJ*DZ(I,J)+(MU(IP+3)*NUJ_DR+NU(JP+3)*MUI_DR)*EIJ
             !     WRITE(6,*) 'FTMP 5 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=0.5*COEFFF*(MUJ_NUI*(-3*KKK(I,J)/DR4+DKKK(I,J)/DR3) &
                  -3*NUI_DR*MUJ_DR*(-5*KKK(I,J)/DR4+DKKK(I,J)/DR3))
             !     WRITE(6,*) 'FIJ=',FIJ
             !     eij=-0.5*COEFFF*3*kkk(i,j1)/dr4
             !     WRITE(6,*) 'EIJ=',EIJ
             FTMP(1)=FIJ*DX(I,J)+(NU(IP+1)*MUJ_DR+MU(JP+1)*NUI_DR)*EIJ
             FTMP(2)=FIJ*DY(I,J)+(NU(IP+2)*MUJ_DR+MU(JP+2)*NUI_DR)*EIJ
             FTMP(3)=FIJ*DZ(I,J)+(NU(IP+3)*MUJ_DR+MU(JP+3)*NUI_DR)*EIJ
             !     WRITE(6,*) 'FTMP 6 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)
             !     endif
          ENDDO loop91

          ! Oxygen-hydrogen dipole-charge forces
          DO J=1,N_H
             J1=N_O+J
             J2=I_H(NBEAD*(J-1)+IBEAD)
             MUI_DR=MU(IP+1)*DX(I,J1)+MU(IP+2)*DY(I,J1)+MU(IP+3)*DZ(I,J1)
             NUI_DR=NU(IP+1)*DX(I,J1)+NU(IP+2)*DY(I,J1)+NU(IP+3)*DZ(I,J1)
             !     WRITE(6,*) 'I,J,I2,J2',I,J,I2,J2
             !     WRITE(6,*) 'MUI_DR',MUI_DR
             !     WRITE(6,*) 'NUI_DR',NUI_DR

             DR2=DR(I,J1)**2
             DR3=DR(I,J1)*DR2
             FIJ=-0.5*COEFFF*MUI_DR*CG(J2)*(-3*LLL(I,J1)/DR3+DLLL(I,J1)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*LLL(I,J1)/DR3
             FTMP(1)=FIJ*DX(I,J1)+MU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J1)+MU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J1)+MU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 1 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=-0.5*COEFFF*NUI_DR*CG(J2)*(-3*KKK(I,J1)/DR3+DKKK(I,J1)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*KKK(I,J1)/DR3
             FTMP(1)=FIJ*DX(I,J1)+NU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J1)+NU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J1)+NU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 3 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)
          ENDDO

          ! Oxygen-other dipole-charge forces
          DO J=1,N_R
             J1=N_O+N_H+J
             J2=I_R(J)
             MUI_DR=MU(IP+1)*DX(I,J1)+MU(IP+2)*DY(I,J1)+MU(IP+3)*DZ(I,J1)
             NUI_DR=NU(IP+1)*DX(I,J1)+NU(IP+2)*DY(I,J1)+NU(IP+3)*DZ(I,J1)
             !     WRITE(6,*) 'MUI_DR',MUI_DR
             !     WRITE(6,*) 'NUI_DR',NUI_DR

             DR2=DR(I,J1)**2
             DR3=DR(I,J1)*DR2
             FIJ=-0.5*COEFFF*MUI_DR*CG(J2)*(-3*LLL(I,J1)/DR3+DLLL(I,J1)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*LLL(I,J1)/DR3
             FTMP(1)=FIJ*DX(I,J1)+MU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J1)+MU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J1)+MU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 1 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)

             FIJ=-0.5*COEFFF*NUI_DR*CG(J2)*(-3*KKK(I,J1)/DR3+DKKK(I,J1)/DR2)
             EIJ=-0.5*COEFFF*CG(J2)*KKK(I,J1)/DR3
             FTMP(1)=FIJ*DX(I,J1)+NU(IP+1)*EIJ
             FTMP(2)=FIJ*DY(I,J1)+NU(IP+2)*EIJ
             FTMP(3)=FIJ*DZ(I,J1)+NU(IP+3)*EIJ
             !     WRITE(6,*) 'FTMP 3 =',FTMP(1),FTMP(2),FTMP(3)
             FTMP(1)=FTMP(1)/NBEAD
             FTMP(2)=FTMP(2)/NBEAD
             FTMP(3)=FTMP(3)/NBEAD
             FX(I2)=FX(I2)+FTMP(1)
             FY(I2)=FY(I2)+FTMP(2)
             FZ(I2)=FZ(I2)+FTMP(3)
             FX(J2)=FX(J2)-FTMP(1)
             FY(J2)=FY(J2)-FTMP(2)
             FZ(J2)=FZ(J2)-FTMP(3)
          ENDDO
       ENDDO loop90
    ENDDO loop40

    ! Feynman kinetic energy spring
    E_KIN=0.0D0
    !     WRITE(6,*) 'K_KIN=',K_KIN
    IF(NBEAD.GT.1)THEN
       DO I=1,N_H
          DO IBEAD=1,NBEAD
             I1=I_H(NBEAD*(I-1)+IBEAD)
             I2=I_H(NBEAD*(I-1)+MOD(IBEAD,NBEAD)+1)
             DX12=X(I1)-X(I2)
             DY12=Y(I1)-Y(I2)
             DZ12=Z(I1)-Z(I2)
             DR2=DX12**2+DY12**2+DZ12**2
             E_KIN=E_KIN+0.5*K_KIN*DR2
             !     WRITE(6,*) I,'E_KIN',I1,I2,SQRT(DR2)
             FX(I1)=FX(I1)+K_KIN*DX12
             FY(I1)=FY(I1)+K_KIN*DY12
             FZ(I1)=FZ(I1)+K_KIN*DZ12
             FX(I2)=FX(I2)-K_KIN*DX12
             FY(I2)=FY(I2)-K_KIN*DY12
             FZ(I2)=FZ(I2)-K_KIN*DZ12
          ENDDO
       ENDDO
       EPOLAR=EPOLAR+E_KIN
       IF(PRNLEV.GE.6)THEN
          WRITE(OUTU,*) 'E_KIN=',E_KIN
          WRITE(OUTU,*) 'FINAL EPOLAR ',EPOLAR
       ENDIF
    ENDIF

    RETURN
  END SUBROUTINE POLAR1

  SUBROUTINE DIST_IJ(XI,YI,ZI,XJ,YJ,ZJ,DX,DY,DZ,DR,DR2,DR3,DR4)
    IMPLICIT NONE
    real(chm_real) XI,YI,ZI,XJ,YJ,ZJ,DX,DY,DZ,DR,DR2,DR3,DR4

    DX=XI-XJ
    DY=YI-YJ
    DZ=ZI-ZJ
    DR2=DX**2+DY**2+DZ**2
    DR=DSQRT(DR2)
    DX=DX/DR
    DY=DY/DR
    DZ=DZ/DR
    DR3=DR2*DR
    DR4=DR2*DR2

    RETURN
  END SUBROUTINE DIST_IJ

#if KEY_PM1==1 /*PM1orPM6*/
  ! ===============================================================
  !  Damping functions in the original polarization model version PM1
  !  ref. J. Chem. Phys. 69, 1473 (1978).
  ! ===============================================================

  SUBROUTINE DAMPING(KKK,LLL,DKKK,DLLL,DR,DR2,DR3,DR4)
    IMPLICIT NONE
    real(chm_real) KKK,LLL,DR,DR2,DR3,DR4
    real(chm_real) DKKK,DLLL
    real(chm_real) E1R,E2R,D1,D2,KR,DKR
    real(chm_real) R_EQ
    DATA R_EQ/0.9584D0/

    D1=DR-R_EQ
    D2=D1*D1
    E1R=DEXP(-8.0*D2)
    E2R=DEXP(-2.702563425*DR)
    KR= 1.855785223*D2*E1R+16.95145727*E2R
    DKR=1.855785223*D1*(2.0D0-D2*16.0)*E1R &
         -16.95145727*2.702563425*E2R
    KKK=DR3/(DR3+KR)
    DKKK=3*DR2/(DR3+KR)-(3*DR2+DKR)*DR3/(DR3+KR)**2

    E1R=DEXP(-3.169888166*DR)
    LLL=1.0D0-E1R*(1.0D0 &
         +3.169888166*DR &
         +5.024095492*DR2 &
         -17.99599078*DR3 &
         +23.92285000*DR4)
    DLLL=-(3.169888166*(LLL-1.0D0)+ &
         E1R*(3.169888166 &
         +5.024095492*2*DR &
         -17.99599078*3*DR2 &
         +23.92285000*4*DR3))
    RETURN
  END SUBROUTINE DAMPING

  SUBROUTINE PHI_OO(E_OO,DE_OO,DR,DR2)
    IMPLICIT NONE
    real(chm_real) E_OO,DE_OO, DR,DR2
    real(chm_real) E1R,E2R,E3R,D1,D2,D3
    real(chm_real) COEFF
    DATA COEFF/332.1669D0/
    D1=DR-2.90D0
    D2=DR-2.45D0
    D3=DR-2.70D0
    E1R=DEXP(2.5D0*D1)
    E2R=DEXP(8.0D0*D2)
    E3R=DEXP(-6.0D0*D3)
    E_OO= &
         +24.D0/(1.0D0+E1R)+90.D0/(1.0D0+E2R)+E3R

    DE_OO= &
         -24.0D0*2.5D0*E1R/(1.0D0+E1R)**2 &
         -90.0D0*8.0D0*E2R/(1.0D0+E2R)**2 &
         -6.0D0*E3R

    !     E_OO=E_OO+4*COEFF/DR
    !     DE_OO=DE_OO-4*COEFF/DR2

    RETURN
  END SUBROUTINE PHI_OO

  SUBROUTINE PHI_OH(E_OH,DE_OH,DR,DR2)
    IMPLICIT NONE
    real(chm_real) E_OH,DR,DR2,D1,D2
    real(chm_real) DE_OH
    real(chm_real) E1R,E2R
    real(chm_real) R_EQ,COEFF
    DATA R_EQ,COEFF/0.9584D0,332.1669D0/

    D1=DR-R_EQ
    D2=D1*D1
    E1R=DEXP(-3.699392820*DR)
    E2R=DEXP(-8.0D0*D2)
    E_OH=(COEFF/DR)*(10.D0*E1R) &
         +(-184.6966743*D1+123.9762188*D2)*E2R

    DE_OH=-(COEFF/DR2)*(10.0D0*E1R) &
         +(COEFF/DR)*(-10.0D0*3.699392820*E1R) &
         +(-184.6966743+123.9762188*2*D1)*E2R &
         +(184.6966743*D1-123.9762188*D2)*8.0D0*2*D1*E2R

    !     E_OH=E_OH-2*COEFF/DR
    !     DE_OH=DE_OH+2*COEFF/DR2

    RETURN
  END SUBROUTINE PHI_OH

#else /* (PM1orPM6)*/

  ! ===============================================================
  !  Damping functions necessary in the polarization
  !  Polarization model version PM6  in J. Phys. Chem. 86, 1314 (1982).
  ! ===============================================================
  SUBROUTINE DAMPING(KKK,LLL,DKKK,DLLL,DR,DR2,DR3,DR4)
    IMPLICIT NONE
    real(chm_real) KKK,LLL,DR,DR2,DR3,DR4
    real(chm_real) DKKK,DLLL
    real(chm_real) D1,D2,KR,DKR
    real(chm_real) L1R,L2R,DL1R,DL2R
    real(chm_real) R_EQ
    DATA  R_EQ/0.9584D0/

    ! K function
    D1=DR-R_EQ
    D2=D1*D1
    KR= 2.116045232*D2*DEXP(-8.0*D2) &
         +20.33584298*DEXP(-2.8924958*DR)
    KKK=DR3/(DR3+KR)
    DKR= 2.116045232*(2*D1-16*D1*D2)*DEXP(-8.0*D2) &
         -20.33584298*2.8924958*DEXP(-2.8924958*DR)
    DKKK=3*DR2/(DR3+KR)-(3*DR2+DKR)*DR3/(DR3+KR)**2

    ! L function
    D1=DR-0.9678133088
    D2=D1*D1
    L1R=0.30D0/(DR3+0.30D0) &
         +0.4189697616*DR3*DEXP(-5.685253959*D2)
    DL1R=-0.30D0*3*DR2/(DR3+0.30D0)**2 &
         +0.4189697616*(3*DR2-DR3*5.685253959*2*D1) &
         *DEXP(-5.685253959*D2)

    L2R=DEXP(-3.160792364*DR)*(1.0D0 &
         +3.160792364*DR &
         +4.995304184*DR2 &
         -24.59792968*DR3 &
         +30.71934979*DR4)

    DL2R=-3.160792364*L2R+ &
         DEXP(-3.160792364*DR)* &
         (3.160792364 &
         +4.995304184*2*DR &
         -24.59792968*3*DR2 &
         +30.71934979*4*DR3)

    LLL=1.0D0-0.5*(L1R+L2R)
    DLLL=-0.5*(DL1R+DL2R)

    RETURN
  END SUBROUTINE DAMPING

  SUBROUTINE PHI_OO(E_OO,DE_OO,DR,DR2)
    ! PM6 OO potential (Coulomb OO interaction is included above)
    IMPLICIT NONE
    real(chm_real) E_OO,DR,DR2
    real(chm_real) F1,F2,F3
    real(chm_real) DF1,DF2,DF3
    real(chm_real) DE_OO
    real(chm_real) COEFF
    DATA COEFF/332.1669D0/

    F1=DEXP(-5.113D0*(DR-2.45D0))
    F2=DEXP(11.739D0*(DR-2.49D0))
    F3=DEXP(3.975D0*(DR-3.77D0))
    E_OO=+24.779D0*F1 &
         +33.445D0/(1.0D0+F2) &
         +3.660D0/(1.0D0+F3)

    DF1=-5.113D0*F1
    DF2=11.739D0*F2
    DF3=3.975D0*F3
    DE_OO=+24.779D0*DF1 &
         -33.445D0*DF2/(1.0D0+F2)**2 &
         -3.660D0*DF3/(1.0D0+F3)**2

    ! Coulomb interactions added above in main subroutine
    !     E_OO=E_OO+4*COEFF/DR
    !     DE_OO=DE_OO-4*COEFF/DR2

    RETURN
  END SUBROUTINE PHI_OO

  SUBROUTINE PHI_OH(E_OH,DE_OH,DR,DR2)
    ! PM6 version (Coulomb OH interaction is added above)
    IMPLICIT NONE
    real(chm_real) E_OH,DR,DR2,D1,D2
    real(chm_real) DE_OH
    real(chm_real) E1R,E2R,E3R
    real(chm_real) R_EQ,COEFF
    DATA R_EQ,COEFF/0.9584D0,332.1669D0/

    D1=DR-R_EQ
    D2=D1*D1
    E1R=DEXP(-4.050595693*DR)
    E2R=DEXP(-3.0D0*(DR-1.6D0)**2)
    E3R=DEXP(-16.0D0*D2)
    E_OH=(COEFF/DR)*( &
         +13.59449911*DEXP(-4.050595693*DR)) &
         +10.0D0*DEXP(-3.0D0*(DR-1.6D0)**2) &
         -198.0722802*D1*DEXP(-16.0D0*D2)

    DE_OH= &
         -COEFF*13.59449911*(1.0D0/DR2+4.050595693/DR)*E1R &
         -10*3*2*(DR-1.6D0)*E2R &
         -198.0722802*(1.0D0-2*16*D2)*E3R

    ! Coulomb added above in main subroutine
    !     E_OH=E_OH-2*COEFF/DR
    !     DE_OH=DE_OH+2*COEFF/DR2
    RETURN
  END SUBROUTINE PHI_OH
  ! ==========================================================
#endif /* (PM1orPM6)*/

#else /* (polar_subs)*/
  SUBROUTINE POLAR0
    RETURN
  END SUBROUTINE POLAR0
#endif /* (polar_subs)*/

end module polarm

