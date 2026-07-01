module mltcanon
  use chm_kinds
  use dimens_fcm
  implicit none
#if KEY_MULTCAN==1 /*multcan_fcm*/
  !
  !  This fcm file contains a common block for the data structures
  !  associated with running multicanonical MD.  The variables are:
  !
  !  QMltcan      =     Logical flag for presence of umbrella potential
  !  Ntrms_Max    =     Maximum number of terms for polynomial fit of
  !                     umbrella potential
  !  Ntrms        =     Number of terms in polynomial fit
  !  CMltcan      =     Real array of polynomial coefficients
  !
  Integer Ntrms
  integer,Parameter :: Ntrms_Max = 10 
  real(chm_real) CMltcan(Ntrms_Max)
  Logical QMltcan

#endif /* (multcan_fcm)*/
  !
contains


#if KEY_MULTCAN==0 /*mltcan_main*/
  Subroutine Mltcanon_prs(comlyn, comlen)
    integer comlen
    character(len=*) :: comlyn
    CALL WRNDIE(-1,'<MLTCAN>','MultiCanonical code not compiled.')
    return
  end Subroutine Mltcanon_prs
#else /* (mltcan_main)*/
  subroutine mltcan_init
    qmltcan=.false.
    ntrms=0
    return
  end subroutine mltcan_init

  Subroutine Mltcanon_prs(comlyn, comlen)
    !-----------------------------------------------------------------------
    !  This source file provides the routines necessary to do
    !  Multicanonical sampling MD simulations.  There are two
    !  routines, the first parses and reads in the coefficients 
    !  from fitting the function U+k_B*T*log(P(U,T_0)) to a 
    !  polynomial in the total potential energy, U, and the second 
    !  scales the forces during the course of energy minimization
    !  or dynamics by the derivitive of the function above wrt U.
    !
    use chm_kinds
    use exfunc
    use stream
    use string
    use number
    implicit none

    Character(len=*) Comlyn
    Integer Comlen

    Character(len=2) :: C(10)=(/'C0','C1','C2','C3','C4','C5','C6','C7','C8','C9'/)
    Integer i

    ! First read index for number of terms in polynomial.

    QMltCan = .false.
    If (IndxA(Comlyn, Comlen,'CLEA').gt.0) Return

    Ntrms = GtrmI(Comlyn, Comlen, 'NCOEF', 0)

    If ( Ntrms .le. Ntrms_Max .and. Ntrms .gt. 0 ) then
       QMltCan = .true.
       Do i = 1, Ntrms
          CMltcan(i) = GtrmF(Comlyn, Comlen, C(i),ZERO)
       Enddo
       If ( Prnlev .ge. 2 ) then
          Write(Outu,'(A)') ' Multicanonical sampling will be used'
          Write(Outu,'(A,I4,A)')  &
               ' Distribution function fit to ', Ntrms, ' nterms'
          Write(Outu,'(5(A,A,f12.5,2x))') &
               (C(i),' =',CMltcan(i),i=1,Ntrms)
       Endif
    Else
       Call Wrndie(-3,'<MLTCAN>', &
            ' Too many coefficients for polynomial fit, max=10')
    Endif

    Return
  End Subroutine Mltcanon_prs
  !
  !
  Subroutine GenEnsemble( E , Eumb, Natom, Dx, Dy, Dz )

    use chm_kinds
    !--mfc   use mltcanon
    implicit none

    Integer Natom
    real(chm_real) E, Eumb, Dx(*), Dy(*), Dz(*)

    Integer i
    real(chm_real) Df, Etmp, Epwr

    !  Compute the value of the Umbrella potential and its derivative


    If ( Ntrms .le. 0 ) Return
    Df = 0.0
    Etmp = -E
    Epwr = 1.0
    Do i=1, Ntrms

       Etmp = Etmp + CMltcan(i)*Epwr
       Df   = Df   + float(i-1)*CMltcan(i)*Epwr

       Epwr = Epwr*E

    Enddo

    If ( E .gt. 0.0 ) Df = Df / E

    Do i=1, Natom

       Dx(i) = Df*Dx(i)
       Dy(i) = Df*Dy(i)
       Dz(i) = Df*Dz(i)
    Enddo
    E = E + Etmp
    Eumb = Eumb + Etmp
    Return
  End Subroutine GenEnsemble
#endif /* (mltcan_main)*/
end module mltcanon

