module rush_mod
  use chm_kinds
  use dimens_fcm
  implicit none
!     Data structure for the RUSH force field
! ---------------------------------------------------------

   real(chm_real) rushKPHOB, rushKHBND, rushDRMX, &
         rushKBDON, rushKBACC, rushKARO, rushCARO
   logical QRUSH

contains

  subroutine rush_init()
    qrush=.false.
    return
  end subroutine rush_init

!     code-cleanup October-November 2005
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush(comlyn, comlen)

!     Parses the user input for the RUSH command
!
!     Author: Olgun Guvench, 6 November 2002
!

  use energym
!     real(chm_real)  ETERM(:)
!     integer rushRepu, rushPhob, rushHbnd,
!    &        rushBdon, rushBacc, rushArom

  use string
!     FUNCTION GTRMF

  use number
!  ZERO

  use stream
!  PRNLEV
!  OUTU

!     *** dummy variables
      implicit none
      character(len=*) comlyn
      integer comlen
      if ( checque( comlyn, 'OFF' ) ) then
         QRUSH = .false.
         ETERM( rushRepu ) = ZERO
         ETERM( rushPhob ) = ZERO
         ETERM( rushHbnd ) = ZERO
         ETERM( rushBdon ) = ZERO
         ETERM( rushBacc ) = ZERO
         ETERM( rushArom ) = ZERO
         call rush_off()
      else if ( checque( comlyn, 'HLST' ) ) then
         call rush_hlist( comlyn, comlen )
      else
         QRUSH = .true.
         rushKPHOB =  0.00072
         rushKHBND = -5.1
         rushKBDON =  0.04
         rushKBACC =  0.04
         rushKARO =   0.00
         rushCARO =   0.00
         rushDRMX =   0.50
         rushKPHOB = gtrmf(comlyn, comlen, 'PHOB', rushKPHOB)
         rushKHBND = gtrmf(comlyn, comlen, 'HBND', rushKHBND)
         rushKBDON &
               = gtrmf(comlyn, comlen, 'BDON', rushKBDON)
         rushKBACC  &
               = gtrmf(comlyn, comlen, 'BACC', rushKBACC)
         rushKARO = gtrmf(comlyn, comlen, 'KARO', rushKARO)
         rushCARO &
               = gtrmf(comlyn, comlen, 'CARO', rushCARO)
         rushDRMX = gtrmf(comlyn, comlen, 'DRMX', rushDRMX)

         if( PRNLEV .ge. 4 ) then
            write(OUTU, '(a,f12.6)') "rush: rushKPHOB = ", &
                  rushKPHOB
            write(OUTU, '(a,f10.4)') "rush: rushKHBND = ", &
                  rushKHBND
            write(OUTU, '(a,f10.4)') "rush: rushKBDON = ", &
                  rushKBDON
            write(OUTU, '(a,f10.4)') "rush: rushKBACC = ", &
                  rushKBACC
            write(OUTU, '(a,f10.4)') "rush: rushKARO =  ", &
                  rushKARO
            write(OUTU, '(a,f10.4)') "rush: rushCARO =  ", &
                  rushCARO
            write(OUTU, '(a,f10.4)') "rush: rushDRMX =  ", &
                  rushDRMX
         endif
      endif

      call xtrane( comlyn, comlen, 'RUSH' )

      return
   end subroutine rush

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_off()
!     Call all the rush_ subroutines that allocate memory and have
!     them release any they've used
!
!     Author: Olgun Guvench, 24 October 2005

      implicit none

!     local variables
      logical,parameter :: off = .true.

!     for rush_energy
      real(chm_real)  ERushREPU, ERushPHOB
      real(chm_real)  ERushHBND, ERushBDON
      real(chm_real)  ERushBACC, ERushAROM
      logical useRushREPU, useRushPHOB
      logical useRushBACC, useRushAROM
      logical useRushHBND, useRushBDON
      real(chm_real)   xLocal(1),  yLocal(1),  zLocal(1)
      real(chm_real)  dxLocal(1), dyLocal(1), dzLocal(1)
      integer nAtomLocal

!     for rush_phob_genlist
      integer neighborList(2,1)
      integer nNeighborMax
      real(chm_real)  radius(1)
      integer iacLocal(1)

!     for rush_hbond_buried_genlist
      integer nDonLocal, nDonorNeighborMax
      real(chm_real)  rHXMaxH, rHXMaxNotH
      integer iDonLocal(1)
      integer iHd1Local(1)
      integer donorNeighborList(2,1)
      integer nAccLocal, nAcceptorNeighborMax
      real(chm_real)  rOXMaxH, rOXMaxNotH
      integer iAccLocal(1)
      integer iAc1Local(1)
      integer acceptorNeighborList(2,1)
      character(len=4) typeLocal(1)
      real(chm_real)  deltaR

!     for rush_aromatic
      real(chm_real)  rushKARO
      real(chm_real)  rushCARO
      real(chm_real)  rushDRMX

      ERushREPU = 0.0
      ERushPHOB = 0.0
      ERushHBND = 0.0
      ERushBDON = 0.0
      ERushBACC = 0.0
      ERushAROM = 0.0
      useRushREPU = .false.
      useRushPHOB = .false.
      useRushBACC = .false.
      useRushAROM = .false.
      useRushHBND = .false.
      useRushBDON = .false.
      xLocal(1) = 0.0
      yLocal(1) = 0.0
      zLocal(1) = 0.0
      dxLocal(1) = 0.0
      dyLocal(1) = 0.0
      dzLocal(1) = 0.0
      nAtomLocal = 1

      neighborList(1,1) = 0
      neighborList(2,1) = 0
      nNeighborMax = 1
      radius(1) = 0.0
      iacLocal(1) = 0
      deltaR = 0.0

      nDonLocal =1
      nDonorNeighborMax = 1
      rHXMaxH = 0.0
      rHXMaxNotH = 0.0
      iDonLocal(1) = 0
      iHd1Local(1) = 0
      donorNeighborList(1,1) = 0
      donorNeighborList(2,1) = 0
      nAccLocal = 0
      nAcceptorNeighborMax = 0
      rOXMaxH = 0.0
      rOXMaxNotH = 0.0
      iAccLocal(1) = 0
      iAc1Local(1) = 0
      acceptorNeighborList(1,1) = 0
      acceptorNeighborList(2,1) = 0
      typeLocal(1) = "XXXX"

      ERushAROM = 0.0
      rushKARO = 0.0
      rushCARO = 0.0
      rushDRMX = 0.0

      call rush_energy(  &
            ERushREPU,   ERushPHOB, &
            ERushHBND,   ERushBDON, &
            ERushBACC,   ERushAROM, &
            useRushREPU, useRushPHOB, &
            useRushHBND, useRushBDON, &
            useRushBACC, useRushAROM, &
            xLocal,  yLocal,  zLocal, &
            dxLocal, dyLocal, dzLocal, &
            nAtomLocal, off )

      call rush_phob_genlist( &
            xLocal,yLocal,zLocal, &
            neighborList, nAtomLocal, nNeighborMax, radius, &
            iacLocal, deltaR, off)

      call rush_hbond_buried_genlist( &
            nAtomLocal, &
            xLocal, yLocal, zLocal, &
            nDonLocal, nDonorNeighborMax,  &
            rHXMaxH, rHXMaxNotH, &
            iDonLocal, &
            iHd1Local, &
            donorNeighborList,  &
            nAccLocal, nAcceptorNeighborMax,  &
            rOXMaxH, rOXMaxNotH, &
            iAccLocal, &
            iAc1Local, &
            acceptorNeighborList,  &
            iacLocal, typeLocal, deltaR, &
            off)

      call rush_aromatic( &
            ERushAROM, &
            rushKARO, &
            rushCARO, &
            nAtomLocal,  &
            xLocal, yLocal, zLocal,  &
            dxLocal, dyLocal, dzLocal, &
            rushDRMX, off)

   end subroutine rush_off

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hlist(comlyn, comlen)

!     Wrapper for rush_hlist2
!
!     Author: Olgun Guvench, 23 April 2003
!

  use coord
!     X, Y, Z
  use psf
!     NATOM
!     ATYPE:       IUPAC Name for each atom
!     NACC, IACC, IAC1,
!     NDON, IDON, IHD1,
  use string
!     FUNCTION GTRMI
      implicit none

!     *** dummy variables
      character(len=*) comlyn
      integer comlen

!     *** local variables
      integer output_unit, default_unit

      default_unit = 6
      output_unit = gtrmi(comlyn, comlen, 'UNIT', default_unit)

      call rush_hlist2( &
            NATOM, &
            X, Y, Z,  &
            NACC, IACC, IAC1, &
            NDON, IDON, IHD1, &
            ATYPE, output_unit)

      return
   end subroutine rush_hlist

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hlist2( &
         nAtomLocal,  &
         xLocal, yLocal, zLocal,  &
         nAccLocal, iAccLocal, iAc1Local, &
         nDonLocal, iDonLocal, iHd1Local, &
         typeLocal, output_unit)

!     Writes out the list of hbond donor-acceptor pairs,
!
!     Author: Olgun Guvench, 23 April 2003
!

  use consta
!     parameter TWOPI = 2.0*3.14...
      implicit none

!     *** dummy parameters
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal),  &
            yLocal(nAtomLocal),  &
            zLocal(nAtomLocal)
      integer nAccLocal, nDonLocal
      integer iAccLocal(nAccLocal), iAc1Local(nAccLocal)
      integer iDonLocal(nDonLocal), iHd1Local(nDonLocal)
      character(len=8) typeLocal(nAtomLocal)
      integer output_unit

!     *** local variables
      integer i, iDonor, iAcceptor, C, O, H, N
      real(chm_real) xC(3), xO(3), xH(3), xN(3)
      real(chm_real) rMinCH, rMinCH2, rMaxCH, rMaxCH2
      real(chm_real) rMinNH, rMinNH2, rMaxNH, rMaxNH2
      real(chm_real) rMin, rMin2, rMax, rMax2
      real(chm_real) xOH, yOH, zOH, rOH2
      real(chm_real) fR, fRdxO(3), fRdxH(3)
      real(chm_real) thetaMinCOH, thetaMaxCOH
      real(chm_real) fThetaCOH
      real(chm_real) fThetaCOHdxC(3), fThetaCOHdxO(3), &
            fThetaCOHdxH(3)
      real(chm_real) thetaMinOHN, thetaMaxOHN
      real(chm_real) fThetaOHN
      real(chm_real) fThetaOHNdxO(3), &
            fThetaOHNdxH(3), fThetaOHNdxN(3)
      real(chm_real) percentage

!     *** devel/033 v3pp107-108
      real(chm_real),parameter :: thetaMinOHN_CH = 100.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxOHN_CH = 120.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMinCOH_CH =  90.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxCOH_CH = 110.0*TWOPI/360.0

!     *** devel/013 v3p99
      real(chm_real),parameter :: thetaMinOHN_NH = 120.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxOHN_NH = 140.0*TWOPI/360.0
!     *** devel/015 v3p109
      real(chm_real),parameter :: thetaMinCOH_NH = 100.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxCOH_NH = 120.0*TWOPI/360.0

      rMinCH = 2.3
      rMinCH2 = rMinCH*rMinCH
      rMaxCH = rMinCH + 1.0
      rMaxCH2 = rMaxCH*rMaxCH

      rMinNH = 1.9
      rMinNH2 = rMinNH*rMinNH
      rMaxNH = rMinNH + 1.0
      rMaxNH2 = rMaxNH*rMaxNH

!     *** C is the acceptor antecedent, O is the acceptor,
!     *** H is the donor proton, N is the donor heavy atom

      do iDonor = 1, nDonLocal

         H = iHd1Local(iDonor)
         N = iDonLocal(iDonor)

         if( typeLocal(N)(1:1) .eq. "C" )then
!         *** the donor is an H-C pair
            rMin = rMinCH
            rMin2 = rMinCH2
            rMax = rMaxCH
            rMax2 = rMaxCH2
            thetaMinOHN = thetaMinOHN_CH
            thetaMaxOHN = thetaMaxOHN_CH
            thetaMinCOH = thetaMinCOH_CH
            thetaMaxCOH = thetaMaxCOH_CH
         else
!         *** the donor is an H-N or H-O pair
            rMin = rMinNH
            rMin2 = rMinNH2
            rMax = rMaxNH
            rMax2 = rMaxNH2
            thetaMinOHN = thetaMinOHN_NH
            thetaMaxOHN = thetaMaxOHN_NH
            thetaMinCOH = thetaMinCOH_NH
            thetaMaxCOH = thetaMaxCOH_NH
         endif

         do iAcceptor = 1, nAccLocal

            C = iAc1Local(iAcceptor)
            O = iAccLocal(iAcceptor)

!         *** skip instances where the acceptor atom O
!         *** is the same atom as the donor heavy atom N
            if (O == N) cycle

            percentage = 0.0

            xOH = xLocal(H) - xLocal(O)
            yOH = yLocal(H) - yLocal(O)
            zOH = zLocal(H) - zLocal(O)

            rOH2 = xOH*xOH + yOH*yOH + zOH*zOH

            if( rOH2 .le. rMax2 )then

               xC(1) = xLocal(C)
               xC(2) = yLocal(C)
               xC(3) = zLocal(C)

               xO(1) = xLocal(O)
               xO(2) = yLocal(O)
               xO(3) = zLocal(O)

               xH(1) = xLocal(H)
               xH(2) = yLocal(H)
               xH(3) = zLocal(H)

               xN(1) = xLocal(N)
               xN(2) = yLocal(N)
               xN(3) = zLocal(N)

               call rush_f_r(rMin, rMax, xO, xH, &
                     fR, fRdxO, fRdxH)

               call rush_f_theta(thetaMinCOH, thetaMaxCOH, &
                     xC, xO, xH, &
                     fThetaCOH, fThetaCOHdxC, fThetaCOHdxO, fThetaCOHdxH)

               call rush_f_theta(thetaMinOHN, thetaMaxOHN, &
                     xO, xH, xN, &
                     fThetaOHN, fThetaOHNdxO, fThetaOHNdxH, fThetaOHNdxN)

               percentage = fR*fThetaCOH*fThetaOHN
               if( percentage .gt. 0.0 )then
                  write( output_unit, '(i8,i8,f8.4)' ) &
                        H, O, percentage
               endif

            endif
         enddo

      enddo

      return
   end subroutine rush_hlist2

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_energy(  &
         ERushREPU,   ERushPHOB, &
         ERushHBND,   ERushBDON, &
         ERushBACC,   ERushAROM, &
         useRushREPU, useRushPHOB, &
         useRushHBND, useRushBDON, &
         useRushBACC, useRushAROM, &
         xLocal,  yLocal,  zLocal, &
         dxLocal, dyLocal, dzLocal, &
         nAtomLocal, off )

!     This is the RUSH control subroutine.
!     It calls all the appropriate subroutines
!     and returns the energies and accumulated gradients.
!
!     Author: Olgun Guvench, 3 December 2001
!             revised October 2005
!

  use memory

  use hbondm
!     IHB        Heavy atom donor
!     JHB        Heavy atom acceptor
!     KHB        Hydrogen atom in H bond (on heavy atom donor)
!     NHB        Number of hydrogen bonds

  use psf
!     ATYPE:       IUPAC Name for each atom
!     NDON, NACC, IHD1, IACC
!     IAC       Atom     Parameter type code

  use coord
!     WMAIN     Weight or temp factor specification (also stores surface area)

  use param
!     ITC       Atom   As each atom type is read in, a counter is kept
!                      of the atoms read in. ITC is the value of the
!                      counter when the atom type was read in residue
!                      topology file.
!     NATC             Number of atom types
!     CNBA      NBond  Value of SIGMA**2 for van der Waal terms
!     CNBB      NBond  Value of epsilon for van der Waal terms
!     MAXCN     NBond  Maximum number of non-bonded interactions (based on atom
!                      type)
!                      PARAMETER (MAXCN = MAXITC*(MAXITC+1)/2)

  use bases_fcm
!  BNBND,  LNBND  - Base arrays for nonbond interactions.

  use inbnd
!  INBLO     INT   (NATOM)  - Number of entries below of JNB entries
!  JNB       INT   (NNNB)   - The second atom of nonbond atom pair

  use number
!  ZERO

      implicit none

!     *** dummy variables
      real(chm_real) ERushREPU, ERushPHOB
      real(chm_real) ERushHBND, ERushBDON
      real(chm_real) ERushBACC, ERushAROM
      logical useRushREPU, useRushPHOB
      logical useRushHBND, useRushBDON
      logical useRushBACC, useRushAROM
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal),  &
            dzLocal(nAtomLocal)
      logical off

!     *** local variables
      integer i
      logical,allocatable,dimension(:) :: hBondDon, hBondAcc, hBondExist
      integer,allocatable,dimension(:) :: iOff
      real(chm_real),allocatable,dimension(:) :: radius
      integer repulse_type
      real(chm_real),allocatable,dimension(:) :: sasa
      real(chm_real)  probe
      real(chm_real),allocatable,dimension(:) :: gamma, gammaPhobic, gammaH
      integer :: nNeighborMax
      integer,allocatable,dimension(:),save :: neighborList
      integer,save :: neighborListSize = -1
      integer :: nDonorNeighborMax
      integer,allocatable,dimension(:),save :: donorNeighborList
      integer,save :: donorNeighborListSize = -1
      integer :: nAcceptorNeighborMax
      integer,allocatable,dimension(:),save :: acceptorNeighborList
      integer,save :: acceptorNeighborListSize = -1
      real(chm_real)  unused_real_1, unused_real_2
      real(chm_real)  rushScaleRep

      integer,parameter :: iprfrq=1000

!     write(*, '(a)') "in rush_energy"

      if( off )then  ! release used heap and exit

         if (allocated(neighborList)) then
            call chmdealloc('rush.src','rush_energy','neighborList', &
                  neighborListSize,intg=neighborList)
         endif
         neighborListSize = -1

         if (allocated(donorNeighborList)) then
            call chmdealloc('rush.src','rush_energy','donorNeighborList', &
                  donorNeighborListSize,intg=donorNeighborList)
         endif
         donorNeighborListSize = -1

         if (allocated(acceptorNeighborList)) then
            call chmdealloc('rush.src','rush_energy','acceptorNeighborList', &
                  acceptorNeighborListSize,intg=acceptorNeighborList)
         endif
         acceptorNeighborListSize = -1

      else if( QRUSH )then  ! calculate energy

         call chmalloc('rush.src','rush_energy','iOff',NATC,intg=iOff)

         call chmalloc('rush.src','rush_energy','radius',nAtomLocal,crl=radius)
         call chmalloc('rush.src','rush_energy','gamma',nAtomLocal,crl=gamma)
         call chmalloc('rush.src','rush_energy','gammaPhobic',nAtomLocal,crl=gammaPhobic)
         call chmalloc('rush.src','rush_energy','gammaH',nAtomLocal,crl=gammaH)
         call chmalloc('rush.src','rush_energy','hBondDon',nAtomLocal,log=hBondDon)
         call chmalloc('rush.src','rush_energy','hBondAcc',nAtomLocal,log=hBondAcc)
         call chmalloc('rush.src','rush_energy','hBondExist',nAtomLocal,log=hBondExist)

         call chmalloc('rush.src','rush_energy','sasa',nAtomLocal,crl=sasa)

         if( useRushREPU )then
!         *** calculate the energy penalty for bad vdW overlaps

            rushScaleRep = 1.0
            call rush_vdwrepulse_wca( &
                  ERushRepu, rushScaleRep, &
                  xLocal, yLocal, zLocal, &
                  dxLocal, dyLocal, dzLocal, nAtomLocal, &
                  iOff, &
                  MAXITC, ITC, NATC, &
                  CNBA, CNBB, RSCLF, MAXCN, IAC, &
                  BNBND%INBLO, BNBND%JNB )

         endif

         if( useRushPHOB )then
!         *** calculate the hydrophobic energy
            if( rushKPHOB .eq. ZERO)then
               ERushPhob = ZERO
            else
               probe = 1.2

               call rush_radius(nAtomLocal,ATYPE,radius,probe)

               call rush_gamma(nAtomLocal, &
                     gammaPhobic, gammaH, &
                     MAXHB, MAXAIM, &
                     IHB, JHB, KHB, NHB, &
                     ATYPE, NDON, NACC, IHD1, IACC, &
                     hBondDon, hBondAcc, hBondExist, &
                     rushKPHOB, ZERO, ZERO)

               nNeighborMax = 400
               if( neighborListSize .ne. nAtomLocal*(nNeighborMax+1) )then
                  neighborListSize = nAtomLocal*(nNeighborMax+1)
                  if (allocated(neighborList)) then
                     call chmrealloc('rush.src','rush_energy','neighborList', &
                           neighborListSize,intg=neighborList)
                  else
                     call chmalloc('rush.src','rush_energy','neighborList', &
                           neighborListSize,intg=neighborList)
                  endif
               endif

!           *** assumes radius() has been augmented by probe radius
               call rush_phob( &
                     ERushPhob, unused_real_1, unused_real_2, &
                     xLocal, yLocal, zLocal, &
                     dxLocal, dyLocal, dzLocal, &
                     neighborList, nAtomLocal, nNeighborMax, &
                     sasa, radius, &
                     gammaPhobic, gammaH, &
                     ATYPE, IAC, rushDRMX )
            endif
         endif  ! if( useRushPHOB )then

         if( useRushHBND )then
            if( rushKHBND .eq. ZERO)then
               ERushHBND = ZERO
            else
               call rush_hbond( &
                     ERushHBND, &
                     unused_real_1,   & ! ETERM(energyHbondDipoleAngularCH),
                     rushKHBND, &
                     ZERO,   & ! rushEhBondCH,
                     nAtomLocal, &
                     xLocal, yLocal, zLocal, &
                     dxLocal, dyLocal, dzLocal, &
                     NACC, IACC, IAC1, &
                     NDON, IDON, IHD1, &
                     ATYPE )
            endif
         endif  ! if( useRushHBND )then

         if( useRushBDON .or. useRushBACC )then

            if( ( rushKBDON .eq. ZERO )  &
                  .and. ( rushKBACC .eq. ZERO ) )then

               ERushBDON = ZERO
               ERushBACC = ZERO

            else

               nDonorNeighborMax = 400
               if( donorNeighborListSize  &
                     .ne. NDON*(nDonorNeighborMax+1) )then
                  donorNeighborListSize = NDON*(nDonorNeighborMax+1)
                  if (allocated(donorNeighborList)) then
                     call chmrealloc('rush.src','rush_energy','donorNeighborList', &
                           donorNeighborListSize,intg=donorNeighborList)
                  else
                     call chmalloc('rush.src','rush_energy','donorNeighborList', &
                           donorNeighborListSize,intg=donorNeighborList)
                  endif
               endif

               nAcceptorNeighborMax = 400
               if( acceptorNeighborListSize  &
                     .ne. NACC*(nAcceptorNeighborMax+1) )then
                  acceptorNeighborListSize &
                        = NACC*(nAcceptorNeighborMax+1)
                  if (allocated(acceptorNeighborList)) then
                     call chmrealloc('rush.src','rush_energy','acceptorNeighborList', &
                           acceptorNeighborListSize,intg=acceptorNeighborList)
                  else
                     call chmalloc('rush.src','rush_energy','acceptorNeighborList', &
                           acceptorNeighborListSize,intg=acceptorNeighborList)
                  endif
               endif

               call rush_hbond_buried( &
                     ERushBDON, ERushBACC, &
                     rushKBDON, rushKBACC, &
                     useRushBDON, useRushBACC, &
                     nAtomLocal,  &
                     xLocal, yLocal, zLocal,  &
                     dxLocal, dyLocal, dzLocal, &
                     NDON, IDON, IHD1, &
                     nDonorNeighborMax, donorNeighborList, &
                     NACC, IACC, IAC1, &
                     nAcceptorNeighborMax, acceptorNeighborList, &
                     ATYPE, IAC, rushDRMX)

            endif

         endif  ! if( useRushBDON .or. useRushBACC )then

         if( useRushAROM )then
            if( rushKARO .eq. ZERO )then
               ERushArom = ZERO
            else
               call rush_aromatic( &
                     ErushArom, &
                     rushKARO, &
                     rushCARO, &
                     nAtomLocal,  &
                     xLocal, yLocal, zLocal,  &
                     dxLocal, dyLocal, dzLocal, &
                     rushDRMX, off)
            endif
         endif

         call chmdealloc('rush.src','rush_energy','sasa',nAtomLocal,crl=sasa)
         call chmdealloc('rush.src','rush_energy','hBondExist',nAtomLocal,log=hBondExist)
         call chmdealloc('rush.src','rush_energy','hBondAcc',nAtomLocal,log=hBondAcc)
         call chmdealloc('rush.src','rush_energy','hBondDon',nAtomLocal,log=hBondDon)
         call chmdealloc('rush.src','rush_energy','gammaH',nAtomLocal,crl=gammaH)
         call chmdealloc('rush.src','rush_energy','gammaPhobic',nAtomLocal,crl=gammaPhobic)
         call chmdealloc('rush.src','rush_energy','gamma',nAtomLocal,crl=gamma)
         call chmdealloc('rush.src','rush_energy','radius',nAtomLocal,crl=radius)
         call chmdealloc('rush.src','rush_energy','iOff',NATC,intg=iOff)

      endif    ! if( off )

      return
   end subroutine rush_energy

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                  BEGIN volume exclusion subroutines                CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_vdwrepulse_wca &
         (energyRepulse, scaleRep, &
         xLocal, yLocal, zLocal, &
         dxLocal, dyLocal, dzLocal, nAtomLocal, &
         iOff, &
         maxITCLocal, iTCLocal, nATCLocal, &
         cNBALocal, cNBBLocal, rsclfLocal, maxCNLocal, iACLocal, &
         iNBLOLocal, jNBLocal )

!     Returns a repulsive energy for atom pairs with bad vdW overlap
!     Adds the appropriate gradient components to the d?Local(:)s
!     Uses CHARMM parameters and repulsive part of 6-12 potential
!     ( a la Weeks, Chandler, and Andersen )
!
!     *** n.b. *** does NOT initialize the d?Local(:)s
!                  DOES initialize energyRepulse to 0.0
!
!     Author: Olgun Guvench, 24 May 2004
!
      use param_store, only: set_param
      implicit none

!     *** dummy variables
      real(chm_real)  energyRepulse
      real(chm_real)  scaleRep
      integer nAtomLocal
      real(chm_real)  xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real)  dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
            dzLocal(nAtomLocal)
      integer maxITCLocal, nATCLocal
      integer iTCLocal(*)
      real(chm_real)  cNBALocal(*), cNBBLocal(*), rsclfLocal(*)
      integer maxCNLocal
      integer iACLocal(*)
      integer iNBLOLocal(*), jNBLocal(*)
      integer iOff(*)

!     *** local variables
      integer itemp, nb, i, j, jpr, npr, i1, j1
      integer iaci, ic
      real(chm_real)  crxi, cryi, crzi
      real(chm_real)  dxit, dyit, dzit
      real(chm_real)  df, gradx, grady, gradz
      real(chm_real)  delx, dely, delz
      real(chm_real)  r, r2, req, keq
      real(chm_real)  sig2, sig6, sig12
      integer ntypei, ntypej
      logical is14
      real(chm_real)  sigma2
      real(chm_real)  xj(3), xi(3), dxi(3), dxj(3)
      real(chm_real)  fr

      itemp = 0
      nb = 0
      energyRepulse = 0.0

!     *** initialize the code look up offsets
      j = 0
      do i = 1, nATCLocal
         iOff(i) = j
         j = j+i
      enddo

      do i = 1, nAtomLocal
         npr = iNBLOLocal(i)-itemp
         itemp = iNBLOLocal(i)

         i1 = iTCLocal(iACLocal(i))
         iaci = iOff(i1)

         crxi = xLocal(i)
         cryi = yLocal(i)
         crzi = zLocal(i)

         dxit = dxLocal(i)
         dyit = dyLocal(i)
         dzit = dzLocal(i)

         do jpr = 1, npr
            nb = nb + 1
            if( jNBLocal(nb) .lt. 0 )then
!         *** -js appear in 1-4 pairs
               j = -jNBLocal(nb)
               j1 = iTCLocal(iACLocal(j))
               if( i1 .lt. j1 )then
                  ic = iOff(j1) + i1 + maxCNLocal
               else
                  ic = iaci + j1 + maxCNLocal
               endif
               is14 = .true.
            else
               j = jNBLocal(nb)
               j1 = iTCLocal(iACLocal(j))
               if( i1 .lt. j1 )then
                  ic = iOff(j1) + i1
               else
                  ic = iaci + j1
               endif
               is14 = .false.
            endif

            delx = xLocal(j) - crxi
            dely = yLocal(j) - cryi
            delz = zLocal(j) - crzi
            r2 = delx*delx + dely*dely + delz*delz

            sigma2 = cNBALocal(ic)
            if( .not. is14 )then
!           *** prevent polar hydrogens from getting too close
!           *** to each other
               if( sigma2 .lt. ( 2.0 * 0.225 )**2 )then
                  sigma2 = ( 2.0 * 1.25 )**2
               endif
            endif

!         *** if r**2 < rmin**2, pay a repulsive penalty
            if( r2 .lt. sigma2 )then

!     write(*,'(a,2i8,2f10.5,l10)') "i, j, cnba, cnbb",
!    & i, j, cnbalocal(ic), cnbblocal(ic), is14

               r2 = 1.0/r2

               sig2 = rsclfLocal(i)*rsclfLocal(j)*sigma2*r2
               sig6 = sig2*sig2*sig2
               sig12 = sig6*sig6

               energyRepulse = energyRepulse  &
                     + scaleRep * (cNBBLocal(ic)*(sig12-sig6-sig6 + 1.0))
               df = scaleRep * cNBBLocal(ic)*r2*12.0*(sig6-sig12)

!           *** calculate gradient on atom j due to its interaction with i
               gradx = df * delx
               grady = df * dely
               gradz = df * delz

!           *** accumulate gradients on atom j
               dxLocal(j) = dxLocal(j) + gradx
               dyLocal(j) = dyLocal(j) + grady
               dzLocal(j) = dzLocal(j) + gradz
!           *** accumulate gradients on atom i
               dxit = dxit - gradx
               dyit = dyit - grady
               dzit = dzit - gradz
            endif
         enddo
         dxLocal(i) = dxit
         dyLocal(i) = dyit
         dzLocal(i) = dzit
      enddo

      call set_param("RREP", energyRepulse)

      return
   end subroutine rush_vdwrepulse_wca
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                   END volume exclusion subroutines                 CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                    BEGIN surface area subroutines                  CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_radius(nAtomLocal,typeLocal,radius,rProbe)
!     fills in radius with radii appropriate for SASA calculation

!     author: Olgun Guvench, 15 January 2002

      implicit none

!     *** dummy parameters
      integer nAtomLocal
      character(len=8) typeLocal(nAtomLocal)
      real(chm_real) radius(nAtomLocal)
      real(chm_real) rProbe

!     *** local variables
      integer i

!     *** fill in the radii
      do i = 1, nAtomLocal
         if( typeLocal(i)(1:1) .eq. "N" )then
            radius(i) = 1.55 + rProbe
         else if( typeLocal(i)(1:1) .eq. "O" )then
            radius(i) = 1.52 + rProbe
         else if( typeLocal(i)(1:1) .eq. "C" )then
            radius(i) = 1.70 + rProbe
         else if( typeLocal(i)(1:1) .eq. "H" )then
            radius(i) = 1.20 + rProbe
         else if( typeLocal(i)(1:1) .eq. "S" )then
            radius(i) = 1.80 + rProbe
         else
            call wrndie( -4, '<rush_radius>', 'Unknown element' )
         endif
!og     write(*,'(a,i8,f10.2)') "atom, radius ", i, radius(i)
      enddo

      return
   end subroutine rush_radius

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_gamma(nAtomLocal, gammaPhobic, gammaH, &
         maxHBLocal, maxAIMLocal, &
         iHBLocal, jHBLocal, kHBLocal, nHBLocal, &
         typeLocal, nDonLocal, nAccLocal, iHD1Local, iAccLocal, &
         hBondDon, hBondAcc, hBondExist, &
         scalephobic, scaleHHeavy, scaleHLight)

!     fills in gammaPhobic(i) and gammaH(i) values for SASA based on whether
!     an atom is a hBond donor or acceptor and the type of atom
!
!     *** Author: Olgun Guvench, 1/24/02
!

      implicit none

!     *** dummy variables
      integer nAtomLocal
      real(chm_real)  gammaPhobic(nAtomLocal)
      real(chm_real)  gammaH(nAtomLocal)
      integer maxHBLocal, maxAIMLocal, nHBLocal
      integer iHBLocal(nHBLocal), jHBLocal(nHBLocal), &
            kHBLocal(nHBLocal)
      character(len=8) typeLocal(nAtomLocal)
      integer nDonLocal, nAccLocal
      integer iHD1Local(nDonLocal), iAccLocal(nAccLocal)
      logical hBondDon(nAtomLocal), hBondAcc(nAtomLocal)
      logical hBondExist(nAtomLocal)
      real(chm_real)  scalephobic, scaleHHeavy, scaleHLight

!     *** local variables
      integer i
      real(chm_real) gammaPhobicNp  ! hydrophobic gamma for non-polar atoms
      real(chm_real) gammaPhobicPolar  ! hydrophobic gamma for polar atoms

!     write(*, '(a,f10.4)')
!    &  "rush_gamma: scalephobic = ",
!    &  scalephobic
!     write(*, '(a,f10.4)')
!    &  "rush_gamma: scaleH = ",
!    &  scaleH

      do i = 1, nAtomLocal
         hBondDon(i) = .false.
         hBondAcc(i) = .false.
      enddo

!     *** hBondDon(i) is true if i is the hydrogen atom of a donor moiety
!     *** If a donor moiety has multiple polar hydrogens which it can
!     *** donate, there will be multiple iHD1Locals; i.e., there is
!     *** no such thing as IHD2 in psf.f90, rather for an NH3+ moiety
!     *** there would be three different IDON(i) IHD1(i) pairs with a
!     *** different value of i for each polar hydrogen.  Each IDON(i)
!     *** would reference the same nitrogen atom while each IHD1(i)
!     *** would reference one of its 3 attached polar hydrogens.
      do i = 1, nDonLocal
         hBondDon(iHD1Local(i)) = .true.
      enddo

!     *** hBondAcc(i) is true if i is the heavy atom of an acceptor moiety
      do i = 1, nAccLocal
         hBondAcc(iAccLocal(i)) = .true.
      enddo

!     gammaPhobic = 0.025  ! run5/6
!     gammaPhobic = 0.050  ! run7/8; run11/12
!     gammaPhobic = 0.075  ! run9/10; run13/14
!     gammaPhobic = 0.100  ! run15/16
!     gammaPhobic = 0.037  ! run17/18
!     gammaPhobic = 0.050  ! run19/20
!     gammaPhobic = 0.063  ! run21/22
!     gammaPhobicNp = scalephobic*0.060  ! run23/24/ 29-34
!     gammaPhobicPolar = scalephobic*0.060  ! devel 019 run >= 7
      gammaPhobicNp = scalephobic     ! devel 024
      gammaPhobicPolar = scalephobic  ! devel 024
!     gammaPhobic = 0.070  ! run25/26
!     gammaPhobicPolar = 0.090
!     gammaPhobic = 0.080  ! run27/28
!     gammaPhobicPolar = 0.095

!     *** hydrophobic penalty is greater for polar atoms
!     *** because they confine water molecules to a single locus
!     *** by hydrogen bonding vs. apolar atoms which define
!     *** a 2D surface with no strong energetic interactions with water
!     gammaPhobicPolar = 2.0 * gammaPhobic

      do i = 1, nAtomLocal
         if( hBondDon(i) )then
!         *** assume maximum exposed hydrogen SASA of 25 A**2 (see 1st
!         *** page of Devel 006 in notebook)
!         *** -2.5/25 = -0.100 since it can form 1 H bond worth -2.5
!         *** kcal/mol; we only count 1/2 of the -5 kcal/mol of the
!         *** hbond since the water, when not interacting with the protein
!         *** interacts with water and thus the -2.5 kcal/mol on the
!         *** water side is always the same
            gammaPhobic(i) = gammaPhobicPolar
!         *** treat HA atoms differently since they are at best
!         *** very weak H bonders  3/25/03
            if( typeLocal(i)(1:2)  .eq. "HA" )then
               gammaH(i) =  0.0
!           write(*, '(i6, a)') i, " is a HA atom"
            else
               gammaH(i) =  scaleHLight
!           write(*, '(i6, a)') i, " is a polar hydrogen"
            endif
         else if( hBondAcc(i) )then
            if( typeLocal(i)(1:1) .eq. "O" )then
!           *** assume max exposed SASA of 50 A**2
!           *** -5.0/50 = -0.100 since it can form 2 H bonds worth
!           *** -2.5 kcal/mol each
!           gamma(i) = gamma(i) + (-0.100)
!           gamma(i) = gammaPhobicPolar + (-0.100)  ! runs < 29
!           gamma(i) = gammaPhobicPolar + (-0.085)  ! run 29/30//35/36//41/42
!           gamma(i) = gammaPhobicPolar + (-0.090)  ! run 31/32//37/38//43/44
!           gamma(i) = gammaPhobicPolar + (-0.095)  ! run 33/34//39/40//45/46//47/48
               gammaPhobic(i) = gammaPhobicPolar
               gammaH(i) =  scaleHHeavy
            else if( typeLocal(i)(1:1) .eq. "N" )then
!           *** purely a guess...
!           gamma(i) = gammaPhobicPolar + (-0.095)  !
               gammaPhobic(i) = gammaPhobicPolar
               gammaH(i) =  scaleHHeavy
            else
               call wrndie( -4, '<rush_gamma>',  &
                     'Non-O, non-N h-bond acceptors not allowed' )
            endif
         else
            gammaPhobic(i) = gammaPhobicNp
            gammaH(i) = 0.0
         endif
!       *** apply the scaleH factor
!       gammaH(i) = scaleH * gammaH(i)
      enddo

      return
   end subroutine rush_gamma

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_phob_genlist( &
         xLocal,yLocal,zLocal, &
         neighborList, nAtomLocal, nNeighborMax, radius, &
         iacLocal, deltaR, off)

!    Generates the sasa neighbor list
!
!     Author: Olgun Guvench, 3 January 2002

  use memory
  use stream
!  PRNLEV
!  OUTU
      implicit none

!    *** dummy variables
      integer nAtomLocal, nNeighborMax
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      integer neighborList(nNeighborMax+1, nAtomLocal)
      real(chm_real) radius(nAtomLocal)
      integer iacLocal(nAtomLocal)
      real(chm_real) deltaR
      logical off

!     *** local variables
      integer i, j, nNeighbor, counter
      real(chm_real) dx, dy, dz, dij2, rij2
      real(chm_real) twoDeltaR
      real(chm_real),allocatable,dimension(:),save :: xOld, yOld, zOld
      integer,save :: oldSize = -1

      if( off ) then  ! release used heap and exit

         if (allocated(xOld)) then
            call chmdealloc('rush.src','rush_phob_genlist','xOld',oldSize,crl=xOld)
            call chmdealloc('rush.src','rush_phob_genlist','yOld',oldSize,crl=yOld)
            call chmdealloc('rush.src','rush_phob_genlist','zOld',oldSize,crl=zOld)
         endif

         oldSize = -1

      else  ! generate the list

         twoDeltaR = DeltaR + DeltaR

         if (oldSize /= nAtomLocal) then
            oldSize = nAtomLocal
            if (allocated(xOld)) then
               call chmrealloc('rush.src','rush_phob_genlist','xOld',oldSize,crl=xOld)
               call chmrealloc('rush.src','rush_phob_genlist','yOld',oldSize,crl=yOld)
               call chmrealloc('rush.src','rush_phob_genlist','zOld',oldSize,crl=zOld)
            else
               call chmalloc('rush.src','rush_phob_genlist','xOld',oldSize,crl=xOld)
               call chmalloc('rush.src','rush_phob_genlist','yOld',oldSize,crl=yOld)
               call chmalloc('rush.src','rush_phob_genlist','zOld',oldSize,crl=zOld)
            endif
            call rush_old_init(nAtomLocal,xLocal,yLocal, &
                  zLocal, xOld, yOld, zOld, deltaR)
         endif

         if( rush_update_neighborlist( &
               nAtomLocal, xLocal, yLocal, zLocal, &
               xOld, yOld, zOld, deltaR) )then
!         *** generate the neighbor list based on radii

            if( PRNLEV .GE. 4 ) &
                  write(OUTU, '(a)') &
                  " rush_phob_genlist: neighborlist update"

!         *** initialize the neighbor list matrix
            do i = 1, nAtomLocal
               do counter = 1, nNeighborMax+1
                  neighborList(counter,i) = 0
               enddo
            enddo

            do i = 1, nAtomLocal-1
               if (iacLocal(i) == 15) cycle
               do j = i+1, nAtomLocal
                  if (iacLocal(j) == 15) cycle
                  dx = xLocal(j) - xLocal(i)
                  dy = yLocal(j) - yLocal(i)
                  dz = zLocal(j) - zLocal(i)
                  dij2 = dx*dx + dy*dy + dz*dz
                  rij2 = radius(i) + radius(j) + twoDeltaR
                  rij2 = rij2*rij2
                  if( dij2 .lt. rij2 )then
                     nNeighbor = neighborList(nNeighborMax+1,i) + 1
!               write(*,'(a,2i7,2f10.2,i7)') "i,j,dij2,rij2,nNeighbor=",
!      &         i, j, dij2, rij2, nNeighbor
                     if( nNeighbor .gt. nNeighborMax )goto 9990
                     neighborList(nNeighbor,i) = j
                     neighborList(nNeighborMax+1,i) = nNeighbor

                     nNeighbor = neighborList(nNeighborMax+1,j) + 1
                     if( nNeighbor .gt. nNeighborMax )goto 9990
                     neighborList(nNeighbor,j) = i
                     neighborList(nNeighborMax+1,j) = nNeighbor
                  endif
               enddo
            enddo
         endif

      endif  ! if( off )

      return

9990  call wrndie( -4, '<rush_phob_genlist>',  &
            'nNeighbor exceeds nNeighborMax' )
!     write(*,'(a,i6,a,i6)')
!    & "nNeighbor: ", nNeighbor, "nNeighborMax", nNeighborMax

      return
   end subroutine rush_phob_genlist

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_phob( &
         energyLcpoPhobic, energyLcpoPolar, energyLcpoPolarH, &
         xLocal, yLocal, zLocal,  &
         dxLocal, dyLocal, dzLocal, &
         neighborList, nAtomLocal, nNeighborMax, &
         sasa, radius, gammaPhobic, gammaH,  &
         typeLocal, iacLocal,deltaR)

!
!     Author: Olgun Guvench, 11 Sept 2002
!

     use memory
     use number,only:zero
     use param_store, only: set_param

     implicit none
     
     !    *** dummy variables
     integer nAtomLocal, nNeighborMax
     real(chm_real) energyLcpoPhobic, energyLcpoPolar, energyLcpoPolarH
     real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
          zLocal(nAtomLocal)
     real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
          dzLocal(nAtomLocal)
     integer neighborList(nNeighborMax+1, nAtomLocal)
      real(chm_real) sasa(nAtomLocal)
      real(chm_real) radius(nAtomLocal)
      real(chm_real) gammaPhobic(nAtomLocal)
      real(chm_real) gammaH(nAtomLocal)
      character(len=8) typeLocal(nAtomLocal)
      integer iacLocal(nAtomLocal)
      real(chm_real) deltaR

!     *** local variables
      integer i
      real(chm_real),allocatable,dimension(:) :: drijdx, drijdy, drijdz
      integer,allocatable,dimension(:) :: sasa_type

       sasa(1:nAtomLocal)=zero

!     *** generate the pairlist
      call rush_phob_genlist( &
            xLocal,yLocal,zLocal, &
            neighborList, nAtomLocal, nNeighborMax, radius, &
            iacLocal, deltaR, .false.)

!     *** calculate the areas
      call chmalloc('rush.src','rush_phob','drijdx',nNeighborMax,crl=drijdx)
      call chmalloc('rush.src','rush_phob','drijdy',nNeighborMax,crl=drijdy)
      call chmalloc('rush.src','rush_phob','drijdz',nNeighborMax,crl=drijdz)
      call chmalloc('rush.src','rush_phob','sasa_type',nAtomLocal,intg=sasa_type)

      call rush_phob_area_045( &
            energyLcpoPhobic, energyLcpoPolar, energyLcpoPolarH, &
            xLocal, yLocal, zLocal, &
            dxLocal, dyLocal, dzLocal, &
            neighborList, nAtomLocal, nNeighborMax, &
            sasa, radius, typeLocal, iacLocal, &
            gammaPhobic, gammaH, &
            drijdx, drijdy, drijdz, &
            sasa_type)

      call set_param("RPHO", energyLcpoPhobic)
!     call set_param("POLR", energyLcpoPolar)
!     call set_param("POLH", energyLcpoPolarH)

      call chmdealloc('rush.src','rush_phob','sasa_type',nAtomLocal,intg=sasa_type)
      call chmdealloc('rush.src','rush_phob','drijdz',nNeighborMax,crl=drijdz)
      call chmdealloc('rush.src','rush_phob','drijdy',nNeighborMax,crl=drijdy)
      call chmdealloc('rush.src','rush_phob','drijdx',nNeighborMax,crl=drijdx)

      return
   end subroutine rush_phob

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_phob_area_045( &
         energyPhobic, energyPolar, energyPolarH, &
         xLocal, yLocal, zLocal, &
         dxLocal, dyLocal, dzLocal, &
         neighborList, nAtomLocal, nNeighborMax, &
         sasa, radius, typeLocal, iacLocal, &
         gammaPhobic, gammaH, &
         drijdx, drijdy, drijdz, &
         sasa_type)

!
!     Author: Olgun Guvench, 11 September 2002
!      updated 26 May 2003
!

  use memory
  use number,only:zero
      implicit none

!     *** dummy varadiables
      real(chm_real) energyPhobic, energyPolar, energyPolarH
      integer nAtomLocal, nNeighborMax
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
            dzLocal(nAtomLocal)
      integer neighborList(nNeighborMax+1, nAtomLocal)
      real(chm_real) sasa(nAtomLocal)
      real(chm_real) radius(nAtomLocal)
      character(len=8) typeLocal(nAtomLocal)
      integer iacLocal(nAtomLocal)
      real(chm_real) gammaPhobic(nAtomLocal)
      real(chm_real) gammaH(nAtomLocal)
      real(chm_real) drijdx(nNeighborMax)
      real(chm_real) drijdy(nNeighborMax)
      real(chm_real) drijdz(nNeighborMax)
      integer sasa_type(nAtomLocal)

!     *** local varadiables
      real(chm_real) xij, yij, zij
      real(chm_real) radi, radj, rho, roff2, rij, rij2
      real(chm_real) resid1, resid2, resid3, resid4
      real(chm_real) inv, dx, dy, dz
      real(chm_real) f1, f2, f3
      integer i, j
      integer counter1, counter2
      integer nNeighborI
      integer stype
      real(chm_real) gammaI, sasaI, grad
      real(chm_real) kH, kPh
      real(chm_real) sasatotal
      integer,allocatable,dimension(:) :: nneighbor, nheavyneighbor

      real(chm_real) c(81,0:4)

      data ( (c(i,j), j = 0, 4), i = 1,81 ) &
          / 1.50560e+00, 3.54962e+00, 5.28557e+00,-2.27121e+00, 2.14964e-01, &
            7.33899e-01, 1.78002e+00, 2.71137e+00,-1.21426e+00, 1.19684e-01, &
            1.87140e-02, 4.67551e-02, 7.34916e-02,-3.29046e-02, 3.25404e-03, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.82190e+00, 6.63271e+00, 9.81673e+00,-4.35707e+00, 4.24724e-01, &
            1.30070e+00, 3.11712e+00, 4.71127e+00,-2.08567e+00, 2.03245e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            3.25389e-01, 7.15833e-01, 9.83223e-01,-4.65351e-01, 4.76251e-02, &
            1.83387e+00, 4.06680e+00, 5.56469e+00,-3.02004e+00, 3.47372e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.85848e+00, 6.29102e+00, 8.66271e+00,-4.03443e+00, 4.07583e-01, &
            1.37496e+00, 3.30755e+00, 4.99537e+00,-2.33974e+00, 2.39599e-01, &
            1.79600e-01, 4.43213e-01, 6.86796e-01,-3.05357e-01, 2.99491e-02, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.67219e+00, 6.31232e+00, 9.33049e+00,-4.38940e+00, 4.50965e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            7.12274e+00, 1.54351e+01, 2.08783e+01,-9.99860e+00, 1.03131e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.41632e+00, 1.92135e+01, 2.75980e+01,-1.23846e+01, 1.21422e+00, &
           -4.69063e+01,-2.61250e+01, 7.21916e+01,-2.32843e+01, 2.05781e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            1.30939e+02, 3.16236e+02, 4.77617e+02,-2.30704e+02, 2.43045e+01, &
            1.03827e+01, 2.39752e+01, 3.47876e+01,-1.56128e+01, 1.53440e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.58540e+00, 1.77302e+01, 2.26286e+01,-1.20042e+01, 1.34118e+00, &
            9.51520e+00, 2.02891e+01, 2.67453e+01,-1.40592e+01, 1.56595e+00, &
            9.08015e+00, 2.00195e+01, 2.72941e+01,-1.42318e+01, 1.58131e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.60798e+00, 1.74980e+01, 2.19133e+01,-1.18297e+01, 1.33809e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            1.25415e+01, 2.52664e+01, 3.12382e+01,-1.72838e+01, 1.98952e+00, &
            2.11065e+01, 4.32917e+01, 5.42887e+01,-3.07714e+01, 3.63268e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
           -6.72389e+01,-3.58965e+01, 8.76469e+01,-2.98494e+01, 2.86449e+00, &
            7.21066e+00, 1.56165e+01, 2.09877e+01,-1.08747e+01, 1.19924e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
           -1.22442e+02,-7.77012e+01, 1.27604e+02,-3.86716e+01, 3.45020e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.12634e+01, 4.43065e+01, 5.63105e+01,-3.32906e+01, 4.07863e+00/

      call chmalloc('rush.src','rush_phob_area_045','nneighbor', &
            nAtomLocal,intg=nneighbor)
      call chmalloc('rush.src','rush_phob_area_045','nheavyneighbor', &
            nAtomLocal,intg=nheavyneighbor)

      energyPhobic = 0.0
      energyPolar = 0.0
      energyPolarH = 0.0

!     write(*, '(a)') "in rush_phob_area_045"

      sasatotal = 0.0

      do i = 1, nAtomLocal
         sasa(i) = 0.0
      enddo

      call rush_sasa_type &
            (sasa_type, nneighbor, nheavyneighbor, &
            nAtomLocal)

!     write(*,'(a10,a8,5a20)') "i", "type", "resid**0", "**1/4",
!    &   "**2/4", "**3/4", "**4/4"
      do i = 1, nAtomLocal
         if (iacLocal(i) == 15) cycle

         radi = radius(i)

         nNeighborI = neighborList(nNeighborMax+1, i)

         do counter1 = 1, nNeighborI
            drijdx(counter1) = 0.0
            drijdy(counter1) = 0.0
            drijdz(counter1) = 0.0
         enddo

         resid1 = 0.0

         kPh = gammaPhobic(i)
         kH = gammaH(i)

!       write(*, '(a,i10)') "rush_phob_area_045: nNeighborI =",
!    &   nNeighborI
         do counter1 = 1, nNeighborI

            j = neighborList(counter1,i)
            radj = radius(j)
            rho = radi+radj
            xij = xLocal(j)-xLocal(i)
            yij = yLocal(j)-yLocal(i)
            zij = zLocal(j)-zLocal(i)
            rij2 = xij*xij + yij*yij + zij*zij

            roff2 = rho*rho

            if( rij2 .le. roff2 )then

               resid1 = resid1 - rij2 + roff2

!           *** multiplied into the gradient later
!           grad = -2.0

               drijdx(counter1) = xij
               drijdy(counter1) = yij
               drijdz(counter1) = zij

            endif

         enddo

         resid1 = sqrt(sqrt(resid1))
         resid2 = resid1*resid1
         resid3 = resid2*resid1
         resid4 = resid3*resid1

         stype = sasa_type(i)

!       *** energy
         sasaI = c(stype, 0) &
               + c(stype, 1)*resid1 &
               + c(stype, 2)*resid2 &
               + c(stype, 3)*resid3 &
               + c(stype, 4)*resid4
         sasa(i) = (kPh*resid3 + kH)*sasaI
         energyPhobic = energyPhobic &
               + (kPh*resid3)*sasaI
         if (stype > 50) then
!         *** This is a hydrogen atom
            energyPolarH = energyPolarH &
                  + kH*sasaI
         else
!         *** This is a heavy atom
            energyPolar = energyPolar &
                  + kH*sasaI
         endif

!     write(*, '(a,f10.4)') "rush_phob_area_045: energyPhobic =",
!    & energyPhobic
!     write(*, '(a,f10.4)') "rush_phob_area_045: kPh =",
!    & kPh
!     write(*, '(a,f10.4)') "rush_phob_area_045: resid3 =",
!    & resid3
!     write(*, '(a,f10.4)') "rush_phob_area_045: sasaI =",
!    & sasaI

!       *** gradient
         grad = (                  kH*c(stype, 1)) &
              + (                  kH*c(stype, 2)) * 2.0 * resid1 &
              + (kPh*c(stype, 0) + kH*c(stype, 3)) * 3.0 * resid2 &
              + (kPh*c(stype, 1) + kH*c(stype, 4)) * 4.0 * resid3 &
              + (kPh*c(stype, 2)                 ) * 5.0 * resid4 &
              + (kPh*c(stype, 3)                 ) * 6.0 * resid4*resid1 &
              + (kPh*c(stype, 4)                 ) * 7.0 * resid4*resid2
!       *** -0.5 = -2.0 from above (drijd?) * 1/4 from here
         grad = -0.5 * 1.0/resid3 * grad

         do counter1 = 1, nNeighborI
            j = neighborList(counter1,i)
            dxLocal(i) = dxLocal(i) &
                  - grad*drijdx(counter1)
            dyLocal(i) = dyLocal(i) &
                  - grad*drijdy(counter1)
            dzLocal(i) = dzLocal(i) &
                  - grad*drijdz(counter1)
            dxLocal(j) = dxLocal(j) &
                  + grad*drijdx(counter1)
            dyLocal(j) = dyLocal(j) &
                  + grad*drijdy(counter1)
            dzLocal(j) = dzLocal(j) &
                  + grad*drijdz(counter1)
         enddo

!       write(*,'(a2,2i8,5e20.13)') "sp", i, sasa_type(i), 1.0,
!    &   resid1, resid2, resid3, resid4
!       write(*,'(a2,2i8,f10.4)') "sp", i, sasa_type(i),
!    &   sasaI
         sasatotal = sasatotal + sasaI

      enddo
!     write(*,'(a,f10.4)') "sasa_total ",
!    & sasatotal

      call chmdealloc('rush.src','rush_phob_area_045','nheavyneighbor', &
            nAtomLocal,intg=nheavyneighbor)
      call chmdealloc('rush.src','rush_phob_area_045','nneighbor', &
            nAtomLocal,intg=nneighbor)

      return
   end subroutine rush_phob_area_045

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_phob_area( &
         energyPhobic, energyPolar, energyPolarH, &
         xLocal, yLocal, zLocal, &
         dxLocal, dyLocal, dzLocal, &
         neighborList, nAtomLocal, nNeighborMax, &
         sasa, radius, typeLocal, iacLocal, &
         gammaPhobic, gammaH, &
         drijdx, drijdy, drijdz, &
         sasa_type)

!
!     Author: Olgun Guvench, 11 September 2002
!

  use memory
      implicit none

!     *** dummy varadiables
      real(chm_real) energyPhobic, energyPolar, energyPolarH
      integer nAtomLocal, nNeighborMax
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
            dzLocal(nAtomLocal)
      integer neighborList(nNeighborMax+1, nAtomLocal)
      real(chm_real) sasa(nAtomLocal)
      real(chm_real) radius(nAtomLocal)
      character(len=8) typeLocal(nAtomLocal)
      integer iacLocal(nAtomLocal)
      real(chm_real) gammaPhobic(nAtomLocal)
      real(chm_real) gammaH(nAtomLocal)
      real(chm_real) drijdx(nNeighborMax)
      real(chm_real) drijdy(nNeighborMax)
      real(chm_real) drijdz(nNeighborMax)
      integer sasa_type(nAtomLocal)

!     *** local varadiables
      real(chm_real) xij, yij, zij
      real(chm_real) radi, radj, rho, roff2, rij, rij2
      real(chm_real) resid1, resid2, resid3, resid4
      real(chm_real) inv, dx, dy, dz
      real(chm_real) f1, f2, f3
      integer i, j
      integer counter1, counter2
      integer nNeighborI
      integer stype
      real(chm_real) gammaI, sasaI, grad
      real(chm_real) sasatotal
      integer,allocatable,dimension(:) :: nneighbor, nheavyneighbor

      real(chm_real) c(81,0:4)

      data ( (c(i,j), j = 0, 4), i = 1,81 ) &
          / 1.50560e+00, 3.54962e+00, 5.28557e+00,-2.27121e+00, 2.14964e-01, &
            7.33899e-01, 1.78002e+00, 2.71137e+00,-1.21426e+00, 1.19684e-01, &
            1.87140e-02, 4.67551e-02, 7.34916e-02,-3.29046e-02, 3.25404e-03, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.82190e+00, 6.63271e+00, 9.81673e+00,-4.35707e+00, 4.24724e-01, &
            1.30070e+00, 3.11712e+00, 4.71127e+00,-2.08567e+00, 2.03245e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            3.25389e-01, 7.15833e-01, 9.83223e-01,-4.65351e-01, 4.76251e-02, &
            1.83387e+00, 4.06680e+00, 5.56469e+00,-3.02004e+00, 3.47372e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.85848e+00, 6.29102e+00, 8.66271e+00,-4.03443e+00, 4.07583e-01, &
            1.37496e+00, 3.30755e+00, 4.99537e+00,-2.33974e+00, 2.39599e-01, &
            1.79600e-01, 4.43213e-01, 6.86796e-01,-3.05357e-01, 2.99491e-02, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.67219e+00, 6.31232e+00, 9.33049e+00,-4.38940e+00, 4.50965e-01, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            7.12274e+00, 1.54351e+01, 2.08783e+01,-9.99860e+00, 1.03131e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.41632e+00, 1.92135e+01, 2.75980e+01,-1.23846e+01, 1.21422e+00, &
           -4.69063e+01,-2.61250e+01, 7.21916e+01,-2.32843e+01, 2.05781e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            1.30939e+02, 3.16236e+02, 4.77617e+02,-2.30704e+02, 2.43045e+01, &
            1.03827e+01, 2.39752e+01, 3.47876e+01,-1.56128e+01, 1.53440e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.58540e+00, 1.77302e+01, 2.26286e+01,-1.20042e+01, 1.34118e+00, &
            9.51520e+00, 2.02891e+01, 2.67453e+01,-1.40592e+01, 1.56595e+00, &
            9.08015e+00, 2.00195e+01, 2.72941e+01,-1.42318e+01, 1.58131e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            8.60798e+00, 1.74980e+01, 2.19133e+01,-1.18297e+01, 1.33809e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            1.25415e+01, 2.52664e+01, 3.12382e+01,-1.72838e+01, 1.98952e+00, &
            2.11065e+01, 4.32917e+01, 5.42887e+01,-3.07714e+01, 3.63268e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
           -6.72389e+01,-3.58965e+01, 8.76469e+01,-2.98494e+01, 2.86449e+00, &
            7.21066e+00, 1.56165e+01, 2.09877e+01,-1.08747e+01, 1.19924e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
           -1.22442e+02,-7.77012e+01, 1.27604e+02,-3.86716e+01, 3.45020e+00, &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            0.0        , 0.0        , 0.0        , 0.0        , 0.0        , &
            2.12634e+01, 4.43065e+01, 5.63105e+01,-3.32906e+01, 4.07863e+00/

      call chmalloc('rush.src','rush_phob_area','nneighbor', &
            nAtomLocal,intg=nneighbor)
      call chmalloc('rush.src','rush_phob_area','nheavyneighbor', &
            nAtomLocal,intg=nheavyneighbor)

      energyPhobic = 0.0
      energyPolar = 0.0
      energyPolarH = 0.0

      sasatotal = 0.0

      do i = 1, nAtomLocal
         sasa(i) = 0.0
      enddo

      call rush_sasa_type &
            (sasa_type, nneighbor, nheavyneighbor, &
            nAtomLocal)

!     write(*,'(a10,a8,5a20)') "i", "type", "resid**0", "**1/4",
!    &   "**2/4", "**3/4", "**4/4"
      do i = 1, nAtomLocal
         if (iacLocal(i) == 15) cycle

         radi = radius(i)

         nNeighborI = neighborList(nNeighborMax+1, i)

         do counter1 = 1, nNeighborI
            drijdx(counter1) = 0.0
            drijdy(counter1) = 0.0
            drijdz(counter1) = 0.0
         enddo

         resid1 = 0.0

         gammaI = gammaPhobic(i) + gammaH(i)

         do counter1 = 1, nNeighborI

            j = neighborList(counter1,i)
            radj = radius(j)
            rho = radi+radj
            xij = xLocal(j)-xLocal(i)
            yij = yLocal(j)-yLocal(i)
            zij = zLocal(j)-zLocal(i)
            rij2 = xij*xij + yij*yij + zij*zij

            roff2 = rho*rho

            if( rij2 .le. roff2 )then

               resid1 = resid1 - rij2 + roff2

!           *** multiplied into the gradient later
!           grad = -2.0

               drijdx(counter1) = xij
               drijdy(counter1) = yij
               drijdz(counter1) = zij

            endif

         enddo

         resid1 = sqrt(sqrt(resid1))
         resid2 = resid1*resid1
         resid3 = resid2*resid1
         resid4 = resid3*resid1

         stype = sasa_type(i)

!       *** energy
         sasaI = c(stype, 0) &
               + c(stype, 1)*resid1 &
               + c(stype, 2)*resid2 &
               + c(stype, 3)*resid3 &
               + c(stype, 4)*resid4
         sasa(i) = gammaI*sasaI
         energyPhobic = energyPhobic &
               + gammaPhobic(i)*sasaI
         if (stype > 50) then
!         *** This is a hydrogen atom
            energyPolarH = energyPolarH &
                  + gammaH(i)*sasaI
         else
!         *** This is a heavy atom
            energyPolar = energyPolar &
                  + gammaH(i)*sasaI
         endif

!       *** gradient
         grad = c(stype, 1)*(1.0/resid3) &
               + 2.0*c(stype, 2)*(1.0/resid2) &
               + 3.0*c(stype, 3)*(1.0/resid1) &
               + 4.0*c(stype, 4)
!       *** -2.0 comes from up above
         grad = -0.5 * gammaI * grad

         do counter1 = 1, nNeighborI
            j = neighborList(counter1,i)
            dxLocal(i) = dxLocal(i) &
                  - grad*drijdx(counter1)
            dyLocal(i) = dyLocal(i) &
                  - grad*drijdy(counter1)
            dzLocal(i) = dzLocal(i) &
                  - grad*drijdz(counter1)
            dxLocal(j) = dxLocal(j) &
                  + grad*drijdx(counter1)
            dyLocal(j) = dyLocal(j) &
                  + grad*drijdy(counter1)
            dzLocal(j) = dzLocal(j) &
                  + grad*drijdz(counter1)
         enddo

!       write(*,'(a2,2i8,5e20.13)') "sp", i, sasa_type(i), 1.0,
!    &   resid1, resid2, resid3, resid4
!       write(*,'(a2,2i8,f10.4)') "sp", i, sasa_type(i),
!    &   sasaI
         sasatotal = sasatotal + sasaI

      enddo
!     write(*,'(a,f10.4)') "sasa_total ",
!    & sasatotal

      call chmdealloc('rush.src','rush_phob_area','nheavyneighbor', &
            nAtomLocal,intg=nheavyneighbor)
      call chmdealloc('rush.src','rush_phob_area','nneighbor', &
            nAtomLocal,intg=nneighbor)

      return
   end subroutine rush_phob_area

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_sasa_type &
         (sasa_type, nneighbor, nheavyneighbor, natomlocal)

!
!     Author: Olgun Guvench, 4 October 2002
!

  use psf
      implicit none

!     *** dummy variables
      integer natomlocal
      integer sasa_type(natomlocal)
      integer nneighbor(natomlocal), nheavyneighbor(natomlocal)
      logical found

!     *** local variables
      integer i, j, bond

      do i = 1, natomlocal
         nneighbor(i) = 0
         nheavyneighbor(i) = 0
         sasa_type(i) = 0
      enddo

      do bond = 1, NBOND
         i = IB(bond)
         j = JB(bond)
         if (ATYPE(j)(1:1) /= "H") then
            nheavyneighbor(i) = nheavyneighbor(i) + 1
         endif
         if( IAC(j) .ne. 15 )then
            nneighbor(i) = nneighbor(i) + 1
         endif
         if (ATYPE(i)(1:1) /= "H") then
            nheavyneighbor(j) = nheavyneighbor(j) + 1
         endif
         if( IAC(i) .ne. 15 )then
            nneighbor(j) = nneighbor(j) + 1
         endif
      enddo

      do i = 1, NATOM
         if (ATYPE(i)(1:1) == "C") then
            if( nneighbor(i) .eq. 4 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 1
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 2
               elseif( nheavyneighbor(i) .eq. 3 )then
                  sasa_type(i) = 3
               elseif( nheavyneighbor(i) .eq. 4 )then
                  sasa_type(i) = 4
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 4' )
               else
                  goto 9991
               endif
            elseif( nneighbor(i) .eq. 3 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 5
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 5' )
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 6
               elseif( nheavyneighbor(i) .eq. 3 )then
                  sasa_type(i) = 7
               else
                  goto 9991
               endif
            else
               goto 9990
            endif
         elseif (ATYPE(i)(1:1) == "N") then
            if( nneighbor(i) .eq. 4 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 11
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 12
               elseif( nheavyneighbor(i) .eq. 3 )then
                  sasa_type(i) = 13
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 13' )
               elseif( nheavyneighbor(i) .eq. 4 )then
                  sasa_type(i) = 14
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 14' )
               else
                  goto 9991
               endif
            elseif( nneighbor(i) .eq. 3 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 15
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 16
               elseif( nheavyneighbor(i) .eq. 3 )then
                  sasa_type(i) = 17
               else
                  goto 9991
               endif
            elseif( nneighbor(i) .eq. 2 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 18
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 18' )
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 19
               else
                  goto 9991
               endif
            else
               goto 9990
            endif
         elseif (ATYPE(i)(1:1) == "O") then
            if( nneighbor(i) .eq. 2 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 21
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 22
                  call wrndie( -4, '<rush_sasa_type>', &
                        'No parameters for sasa_type 22' )
               else
                  goto 9991
               endif
            elseif( nneighbor(i) .eq. 1 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  if( IAC(i) .eq. 72 )then
!               *** carboxylate oxygen
                     sasa_type(i) = 24
                  else
                     sasa_type(i) = 23
                  endif
               else
                  goto 9991
               endif
            else
               goto 9990
            endif
         elseif (ATYPE(i)(1:1) == "S") then
            if( nneighbor(i) .eq. 2 )then
               if( nheavyneighbor(i) .eq. 1 )then
                  sasa_type(i) = 31
               elseif( nheavyneighbor(i) .eq. 2 )then
                  sasa_type(i) = 32
               else
                  goto 9991
               endif
            else
               goto 9990
            endif
         endif
      enddo

      do bond = 1, NBOND
         i = IB(bond)
         j = JB(bond)
         if (ATYPE(i)(1:1) == "H" .and. IAC(i) /= 15) then
            sasa_type(i) = sasa_type(j) + 50
         elseif (ATYPE(j)(1:1) == "H" .and. IAC(j) /= 15) then
            sasa_type(j) = sasa_type(i) + 50
         endif
      enddo

      return

9990  call wrndie(-4, '<rush_sasa_type>', 'Unknown nneighbor')

9991  call wrndie( -4, '<rush_sasa_type>', 'Unknown nheavyneighbor' )

   end subroutine rush_sasa_type

!     found = .false.
!     if( atom_type .eq. 1 )then
!       i = 1
!       do while( .not. found )
!         if( i .gt. NBOND )then
!           write( *, '(a)') "rush_sasa_type: can't find H neigbor"
!           call die
!         endif
!         if( IB(i) .eq. atom )then
!           neighbor = JB(i)
!           found = .true.
!         elseif( JB(i) .eq. atom )then
!           neighbor = IB(i)
!           found = .true.
!         endif
!         i = i+1
!       enddo
!       neighbor_type = IAC(neighbor)
!       if( neighbor_type .eq. 54 )then
!         atom_type = 101
!       elseif( neighbor_type .eq. 55 )then
!         atom_type = 102
!       elseif( neighbor_type .eq. 51 )then
!         atom_type = 103
!       elseif( neighbor_type .eq. 53 )then
!         atom_type = 104
!       elseif( neighbor_type .eq. 58 )then
!         atom_type = 105
!       elseif( neighbor_type .eq. 73 )then
!         atom_type = 106
!       else
!         write( *, '(a)') "rush_sasa_type: can't match H neigbor"
!         call die
!       endif
!     endif

!     return
!     end

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                     END surface area subroutines                   CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                 BEGIN intra-molecular h-bond subroutines           CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hbond( &
         energyHbondDipoleAngularNH, &
         energyHbondDipoleAngularCH, &
         eHbondNH, &
         eHbondCH, &
         nAtomLocal,  &
         xLocal, yLocal, zLocal,  &
         dxLocal, dyLocal, dzLocal, &
         nAccLocal, iAccLocal, iAc1Local, &
         nDonLocal, iDonLocal, iHd1Local, &
         typeLocal)

!     Treats h-bonding as a
!     cos(A)**2 * cos(B)**2 * cos(C)**2
!     where A = f(O-H distance)
!     where B = f(C-O-H angle)
!     where C = f(O-H-N angle)
!
!     *** Author: Olgun Guvench, 6/21/02
!         modified Jan 24, 2003 to include C-H donor pairs
!

  use consta
!     parameter TWOPI = 2.0*3.14...
     use param_store, only: set_param

      implicit none

!     *** dummy parameters
      real(chm_real) energyHbondDipoleAngularCH
      real(chm_real) energyHbondDipoleAngularNH
      real(chm_real) eHbondNH
      real(chm_real) eHbondCH
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal),  &
            yLocal(nAtomLocal),  &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal),  &
            dyLocal(nAtomLocal),  &
            dzLocal(nAtomLocal)
      integer nAccLocal, nDonLocal
      integer iAccLocal(nAccLocal), iAc1Local(nAccLocal)
      integer iDonLocal(nDonLocal), iHd1Local(nDonLocal)
      character(len=8) typeLocal(nAtomLocal)

!     *** local variables
      integer i, iDonor, iAcceptor, C, O, H, N
      real(chm_real) xC(3), xO(3), xH(3), xN(3)
      real(chm_real) rMinCH, rMinCH2, rMaxCH, rMaxCH2
      real(chm_real) rMinNH, rMinNH2, rMaxNH, rMaxNH2
      real(chm_real) rMin, rMin2, rMax, rMax2
      real(chm_real) xOH, yOH, zOH, rOH2
      real(chm_real) fR, fRdxC(3), fRdxO(3), fRdxH(3), fRdxN(3)
      real(chm_real) thetaMinCOH, thetaMaxCOH
      real(chm_real) fThetaCOH
      real(chm_real) fThetaCOHdxC(3), fThetaCOHdxO(3), &
            fThetaCOHdxH(3), fThetaCOHdxN(3)
      real(chm_real) thetaMinOHN, thetaMaxOHN
      real(chm_real) fThetaOHN
      real(chm_real) fThetaOHNdxC(3), fThetaOHNdxO(3), &
            fThetaOHNdxH(3), fThetaOHNdxN(3)
      real(chm_real) eHbond, energyTmp

!     *** devel/033 v3pp107-108
      real(chm_real),parameter :: thetaMinOHN_CH = 100.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxOHN_CH = 120.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMinCOH_CH =  90.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxCOH_CH = 110.0*TWOPI/360.0

!     *** devel/013 v3p99
      real(chm_real),parameter :: thetaMinOHN_NH = 120.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxOHN_NH = 140.0*TWOPI/360.0
!     *** devel/015 v3p109
      real(chm_real),parameter :: thetaMinCOH_NH = 100.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxCOH_NH = 120.0*TWOPI/360.0

!     *** the only reason for the existence of these variables
!     *** is to preserve to symmetric appearance of the
!     *** "d?Local(i) =" equations
      do i = 1, 3
         fRdxC(i) = 0.0
         fRdxN(i) = 0.0
         fThetaCOHdxN(i)=0.0
         fThetaOHNdxC(i)=0.0
      enddo

      energyHbondDipoleAngularCH = 0.0
      rMinCH = 2.3
      rMinCH2 = rMinCH*rMinCH
      rMaxCH = rMinCH + 1.0
      rMaxCH2 = rMaxCH*rMaxCH

      energyHbondDipoleAngularNH = 0.0
      rMinNH = 1.9
      rMinNH2 = rMinNH*rMinNH
      rMaxNH = rMinNH + 1.0
      rMaxNH2 = rMaxNH*rMaxNH

!     *** C is the acceptor antecedent, O is the acceptor,
!     *** H is the donor proton, N is the donor heavy atom

      do iDonor = 1, nDonLocal

         H = iHd1Local(iDonor)
         N = iDonLocal(iDonor)

         if( typeLocal(N)(1:1) .eq. "C" )then
!         *** the donor is an H-C pair
            rMin = rMinCH
            rMin2 = rMinCH2
            rMax = rMaxCH
            rMax2 = rMaxCH2
            eHbond = eHbondCH
            thetaMinOHN = thetaMinOHN_CH
            thetaMaxOHN = thetaMaxOHN_CH
            thetaMinCOH = thetaMinCOH_CH
            thetaMaxCOH = thetaMaxCOH_CH
         else
!         *** the donor is an H-N or H-O pair
            rMin = rMinNH
            rMin2 = rMinNH2
            rMax = rMaxNH
            rMax2 = rMaxNH2
            eHbond = eHbondNH
            thetaMinOHN = thetaMinOHN_NH
            thetaMaxOHN = thetaMaxOHN_NH
            thetaMinCOH = thetaMinCOH_NH
            thetaMaxCOH = thetaMaxCOH_NH
         endif

         energyTmp = 0.0
         do iAcceptor = 1, nAccLocal

            C = iAc1Local(iAcceptor)
            O = iAccLocal(iAcceptor)

!         *** skip instances where the acceptor atom O
!         *** is the same atom as the donor heavy atom N
            if (O == N) cycle

            xOH = xLocal(H) - xLocal(O)
            yOH = yLocal(H) - yLocal(O)
            zOH = zLocal(H) - zLocal(O)

            rOH2 = xOH*xOH + yOH*yOH + zOH*zOH

            if( rOH2 .le. rMax2 )then

               xC(1) = xLocal(C)
               xC(2) = yLocal(C)
               xC(3) = zLocal(C)

               xO(1) = xLocal(O)
               xO(2) = yLocal(O)
               xO(3) = zLocal(O)

               xH(1) = xLocal(H)
               xH(2) = yLocal(H)
               xH(3) = zLocal(H)

               xN(1) = xLocal(N)
               xN(2) = yLocal(N)
               xN(3) = zLocal(N)

               call rush_f_r(rMin, rMax, xO, xH, &
                     fR, fRdxO, fRdxH)

               call rush_f_theta(thetaMinCOH, thetaMaxCOH, &
                     xC, xO, xH, &
                     fThetaCOH, fThetaCOHdxC, fThetaCOHdxO, fThetaCOHdxH)

               call rush_f_theta(thetaMinOHN, thetaMaxOHN, &
                     xO, xH, xN, &
                     fThetaOHN, fThetaOHNdxO, fThetaOHNdxH, fThetaOHNdxN)

               energyTmp =  energyTmp &
                     + eHbond*fR*fThetaCOH*fThetaOHN

               dxLocal(C) = dxLocal(C) &
                     + eHbond * (fRdxC(1)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxC(1)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxC(1))
               dyLocal(C) = dyLocal(C) &
                     + eHbond * (fRdxC(2)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxC(2)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxC(2))
               dzLocal(C) = dzLocal(C) &
                     + eHbond * (fRdxC(3)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxC(3)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxC(3))

               dxLocal(O) = dxLocal(O) &
                     + eHbond * (fRdxO(1)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxO(1)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxO(1))
               dyLocal(O) = dyLocal(O) &
                     + eHbond * (fRdxO(2)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxO(2)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxO(2))
               dzLocal(O) = dzLocal(O) &
                     + eHbond * (fRdxO(3)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxO(3)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxO(3))

               dxLocal(H) = dxLocal(H) &
                     + eHbond * (fRdxH(1)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxH(1)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxH(1))
               dyLocal(H) = dyLocal(H) &
                     + eHbond * (fRdxH(2)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxH(2)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxH(2))
               dzLocal(H) = dzLocal(H) &
                     + eHbond * (fRdxH(3)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxH(3)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxH(3))

               dxLocal(N) = dxLocal(N) &
                     + eHbond * (fRdxN(1)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxN(1)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxN(1))
               dyLocal(N) = dyLocal(N) &
                     + eHbond * (fRdxN(2)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxN(2)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxN(2))
               dzLocal(N) = dzLocal(N) &
                     + eHbond * (fRdxN(3)*fThetaCOH*fThetaOHN &
                     + fR*fThetaCOHdxN(3)*fThetaOHN &
                     + fR*fThetaCOH*fThetaOHNdxN(3))

            endif
         enddo

         if( typeLocal(N)(1:1) .eq. "C" )then
            energyHbondDipoleAngularCH = &
                  energyHbondDipoleAngularCH + energyTmp
         else
            energyHbondDipoleAngularNH = &
                  energyHbondDipoleAngularNH + energyTmp
         endif
      enddo

      call set_param("RHBN", energyHbondDipoleAngularNH)
!     call set_param("HBNC", energyHbondDipoleAngularCH)

      return
   end subroutine rush_hbond
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                  END intra-molecular h-bond subroutines            CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                 BEGIN protein-solvent h-bond subroutines           CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hbond_buried( &
         energyHbondBuriedDonor, energyHbondBuriedAcceptor, &
         eHbondDonor, eHbondAcceptor, &
         useRushBDON, useRushBACC, &
         nAtomLocal,  &
         xLocal, yLocal, zLocal,  &
         dxLocal, dyLocal, dzLocal, &
         nDonLocal, iDonLocal, iHd1Local, &
         nDonorNeighborMax, donorNeighborList, &
         nAccLocal, iAccLocal, iAc1Local, &
         nAcceptorNeighborMax, acceptorNeighborList, &
         typeLocal, iacLocal, deltaR)

!     Treats buried h-bond penalty as
!     cos(A)**2 * cos(B)**2
!     where A = f(O:X distance) or f(H:X distance)
!     where B = f(C=O:X angle) or f(N-H:X angle)
!
!     *** Author: Olgun Guvench, 4/29/04
!

  use consta
!     parameter TWOPI = 2.0*3.14...
     use param_store, only: set_param
      implicit none

!     *** dummy parameters
      real(chm_real) energyHbondBuriedDonor
      real(chm_real) energyHbondBuriedAcceptor
      real(chm_real) eHbondDonor
      real(chm_real) eHbondAcceptor
      logical useRushBDON, useRushBACC
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal),  &
            yLocal(nAtomLocal),  &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal),  &
            dyLocal(nAtomLocal),  &
            dzLocal(nAtomLocal)
      integer nDonLocal
      integer iDonLocal(nDonLocal), iHd1Local(nDonLocal)
      integer nDonorNeighborMax
      integer donorNeighborList(nDonorNeighborMax+1, nDonLocal)
      integer nAccLocal
      integer iAccLocal(nAccLocal), iAc1Local(nAccLocal)
      integer nAcceptorNeighborMax
      integer acceptorNeighborList(nAcceptorNeighborMax+1, nAccLocal)
      character(len=4) typeLocal(nAtomLocal)
      integer iacLocal(nAtomLocal)
      real(chm_real) deltaR

!     *** local variables
      real(chm_real) rHXMin, rHXMaxH, rHXMaxNotH
      real(chm_real) rOXMin, rOXMaxH, rOXMaxNotH
      logical off

      off = .false.

      rHXMin = 1.9
!     *** rHXMaxH based on g(OH) and rHXMaxNotH on g(OO)
!     *** see Sorenson, Hura, Glaeser, Head-Gordon,
!     *** J Chem Phys 113, 9149-9161.
!     *** http://www.lsbu.ac.uk/water/evidnc.html
      rHXMaxH = rHXMin + 1.9
      rHXMaxNotH = rHXMin + 2.8

      rOXMin = 1.9
!     *** rOXMaxH based on g(OH) and rOXMaxNotH on g(OO)
!     *** see Sorenson, Hura, Glaeser, Head-Gordon,
!     *** J Chem Phys 113, 9149-9161.
!     *** http://www.lsbu.ac.uk/water/evidnc.html
      rOXMaxH = rOXMin + 1.9
      rOXMaxNotH = rOXMin + 2.8

      call rush_hbond_buried_genlist( &
            nAtomLocal, &
            xLocal, yLocal, zLocal, &
            nDonLocal, nDonorNeighborMax,  &
            rHXMaxH, rHXMaxNotH, &
            iDonLocal, &
            iHd1Local, &
            donorNeighborList,  &
            nAccLocal, nAcceptorNeighborMax,  &
            rOXMaxH, rOXMaxNotH, &
            iAccLocal, &
            iAc1Local, &
            acceptorNeighborList,  &
            iacLocal, typeLocal, deltaR, &
            off)

      call rush_hbond_buried_energy( &
            energyHbondBuriedDonor, &
            energyHbondBuriedAcceptor, &
            eHbondDonor, &
            eHbondAcceptor, &
            useRushBDON, useRushBACC, &
            nAtomLocal,  &
            xLocal, yLocal, zLocal,  &
            dxLocal, dyLocal, dzLocal, &
            nDonLocal, nDonorNeighborMax, &
            rHXMin, rHXMaxH, rHXMaxNotH, &
            iDonLocal, iHd1Local, &
            donorNeighborList, &
            nAccLocal, nAcceptorNeighborMax, &
            rOXMin, rOXMaxH, rOXMaxNotH, &
            iAccLocal, iAc1Local, &
            acceptorNeighborList, &
            typeLocal)

      call set_param("RBDO", energyHbondBuriedDonor)
      call set_param("RBAC", energyHbondBuriedAcceptor)

   end subroutine rush_hbond_buried

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hbond_buried_genlist( &
         nAtomLocal, &
         xLocal, yLocal, zLocal, &
         nDonLocal, nDonorNeighborMax,  &
         rHXMaxH, rHXMaxNotH, &
         iDonLocal, &
         iHd1Local, &
         donorNeighborList,  &
         nAccLocal, nAcceptorNeighborMax,  &
         rOXMaxH, rOXMaxNotH, &
         iAccLocal, &
         iAc1Local, &
         acceptorNeighborList,  &
         iacLocal, typeLocal, deltaR, &
         off)

!    Generates the hbond_buried neighbor lists
!
!     Author: Olgun Guvench, 30 April 2004
!

  use memory
  use stream
!  PRNLEV
!  OUTU
      implicit none

!    *** dummy variables
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)

      integer nDonLocal, nDonorNeighborMax
      real(chm_real)  rHXMaxH, rHXMaxNotH
      integer iDonLocal(nDonLocal)
      integer iHd1Local(nDonLocal)
      integer donorNeighborList( &
            nDonorNeighborMax+1, nDonLocal)

      integer nAccLocal, nAcceptorNeighborMax
      real(chm_real)  rOXMaxH, rOXMaxNotH
      integer iAccLocal(nAccLocal)
      integer iAc1Local(nAccLocal)
      integer acceptorNeighborList( &
            nAcceptorNeighborMax+1, nAccLocal)

      integer iacLocal(nAtomLocal)
      character(len=4) typeLocal(nAtomLocal)
      real(chm_real) deltaR

      logical off

!     *** local variables
      integer i, j, nDonorNeighbor, nAcceptorNeighbor, counter
      real(chm_real) dx, dy, dz, dij2, rij2
      real(chm_real) twoDeltaR, rMax
      real(chm_real),allocatable,dimension(:),save :: xOld, yOld, zOld
      integer,save :: oldSize = -1

      if( off )then  ! release usd heap and exit

         if (allocated(xOld)) then
            call chmdealloc('rush.src','rush_hbond_buried_genlist','xOld', &
                  oldSize,crl=xOld)
            call chmdealloc('rush.src','rush_hbond_buried_genlist','yOld', &
                  oldSize,crl=yOld)
            call chmdealloc('rush.src','rush_hbond_buried_genlist','zOld', &
                  oldSize,crl=zOld)
         endif
         oldSize = -1

      else  ! generate the list

         twoDeltaR = DeltaR + DeltaR

         if (oldSize /= nAtomLocal) then
            oldSize = nAtomLocal
            if (allocated(xOld)) then
               call chmrealloc('rush.src','rush_hbond_buried_genlist','xOld', &
                     oldSize,crl=xOld)
               call chmrealloc('rush.src','rush_hbond_buried_genlist','yOld', &
                     oldSize,crl=yOld)
               call chmrealloc('rush.src','rush_hbond_buried_genlist','zOld', &
                     oldSize,crl=zOld)
            else
               call chmalloc('rush.src','rush_hbond_buried_genlist','xOld', &
                     oldSize,crl=xOld)
               call chmalloc('rush.src','rush_hbond_buried_genlist','yOld', &
                     oldSize,crl=yOld)
               call chmalloc('rush.src','rush_hbond_buried_genlist','zOld', &
                     oldSize,crl=zOld)
            endif
            call rush_old_init(nAtomLocal,xLocal,yLocal, &
                  zLocal, xOld, yOld, zOld, deltaR)
         endif

         if( rush_update_neighborlist( &
               nAtomLocal, xLocal, yLocal, zLocal, &
               xOld, yOld, zOld, deltaR) )then

            if( PRNLEV .GE. 4 ) &
                  write(OUTU, '(a)') &
                  " rush_hbond_buried_genlist: neighborlist update"

!         *** initialize the donor neighbor list matrix
            do i = 1, nDonLocal
               do counter = 1, nDonorNeighborMax+1
                  donorNeighborList(counter,i) = 0
               enddo
            enddo

!         *** fill the donor neighbor list matrix
            do i = 1, nDonLocal
               do j = 1, nAtomLocal
!             *** skip "invisible" hydrogens
!             *** .and. self
!             *** .and. proton antecedent
!             *** the latter is necessary to prevent trouble caused
!             *** by acos of i - j - i angles in subroutine rush_f_theta
                  if( ( iacLocal(j) .ne. 15 )  &
                        .and. ( iHD1Local(i) .ne. j ) &
                        .and. ( iDonLocal(i) .ne. j ) )then
                     dx = xLocal(j) - xLocal(iHd1Local(i))
                     dy = yLocal(j) - yLocal(iHd1Local(i))
                     dz = zLocal(j) - zLocal(iHd1Local(i))
                     dij2 = dx*dx + dy*dy + dz*dz
                     if( typeLocal(j)(1:1) .eq. "H" )then
                        rMax = rHXMaxH
                     else
                        rMax = rHXMaxNotH
                     endif
                     rij2 = rMax + twoDeltaR
                     rij2 = rij2*rij2
                     if( dij2 .lt. rij2 )then
                        nDonorNeighbor  &
                              = donorNeighborList(nDonorNeighborMax+1,i) &
                              + 1
                        if( nDonorNeighbor .gt. nDonorNeighborMax ) &
                              goto 9990
                        donorNeighborList(nDonorNeighbor,i) = j
                        donorNeighborList(nDonorNeighborMax+1,i) &
                              = nDonorNeighbor
                     endif
                  endif
               enddo
            enddo

!         *** initialize the acceptor neighbor list matrix
            do i = 1, nAccLocal
               do counter = 1, nAcceptorNeighborMax+1
                  acceptorNeighborList(counter,i) = 0
               enddo
            enddo

!         *** fill the acceptor neighbor list matrix
            do i = 1, nAccLocal
               do j = 1, nAtomLocal
!             *** skip "invisible" hydrogens
!             *** .and. self
!             *** .and. acceptor antecedent
!             *** the latter is necessary to prevent trouble caused
!             *** by acos of i - j - i angles in subroutine rush_f_theta
                  if( ( iacLocal(j) .ne. 15 )  &
                        .and. ( iAccLocal(i) .ne. j ) &
                        .and. ( iAc1Local(i) .ne. j ) )then
                     dx = xLocal(j) - xLocal(iAccLocal(i))
                     dy = yLocal(j) - yLocal(iAccLocal(i))
                     dz = zLocal(j) - zLocal(iAccLocal(i))
                     dij2 = dx*dx + dy*dy + dz*dz
                     if( typeLocal(j)(1:1) .eq. "H" )then
                        rMax = rHXMaxH
                     else
                        rMax = rHXMaxNotH
                     endif
                     rij2 = rMax + twoDeltaR
                     rij2 = rij2*rij2
                     if( dij2 .lt. rij2 )then
                        nAcceptorNeighbor  &
                              = acceptorNeighborList(nAcceptorNeighborMax+1,i) &
                              + 1
                        if( nAcceptorNeighbor .gt. nAcceptorNeighborMax ) &
                              goto 9995
                        acceptorNeighborList(nAcceptorNeighbor,i) = j
                        acceptorNeighborList(nAcceptorNeighborMax+1,i) &
                              = nAcceptorNeighbor
                     endif
                  endif
               enddo
            enddo
         endif

      endif  ! if( off )

      return

9990  call wrndie( -4, '<rush_hbond_buried_genlist>', &
            'nDonorNeighbor exceeds nDonorNeighborMax' )

9995  call wrndie( -4, '<rush_hbond_buried_genlist>', &
            'nAcceptorNeighbor exceeds nAcceptorNeighborMax' )

      return
   end subroutine rush_hbond_buried_genlist

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_hbond_buried_energy( &
         energyHbondBuriedDonor, &
         energyHbondBuriedAcceptor, &
         eHbondDonor, &
         eHbondAcceptor, &
         useRushBDON, useRushBACC, &
         nAtomLocal,  &
         xLocal, yLocal, zLocal,  &
         dxLocal, dyLocal, dzLocal, &
         nDonLocal, nDonorNeighborMax, &
         rHXMin, rHXMaxH, rHXMaxNotH, &
         iDonLocal, iHd1Local, &
         donorNeighborList, &
         nAccLocal, nAcceptorNeighborMax, &
         rOXMin, rOXMaxH, rOXMaxNotH, &
         iAccLocal, iAc1Local, &
         acceptorNeighborList, &
         typeLocal)

!     *** Author: Olgun Guvench, 4/29/04
!

  use consta
!     parameter TWOPI = 2.0*3.14...
      implicit none

!     *** dummy parameters
      real(chm_real) energyHbondBuriedDonor
      real(chm_real) energyHbondBuriedAcceptor
      real(chm_real) eHbondDonor
      real(chm_real) eHbondAcceptor

      logical useRushBDON, useRushBACC

      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal),  &
            yLocal(nAtomLocal),  &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal),  &
            dyLocal(nAtomLocal),  &
            dzLocal(nAtomLocal)

      integer nDonLocal, nDonorNeighborMax
      real(chm_real) rHXMin, rHXMaxH, rHXMaxNotH
      integer iDonLocal(nDonLocal), iHd1Local(nDonLocal)
      integer donorNeighborList( &
            nDonorNeighborMax+1, nDonLocal)

      integer nAccLocal, nAcceptorNeighborMax
      real(chm_real) rOXMin, rOXMaxH, rOXMaxNotH
      integer iAccLocal(nAccLocal), iAc1Local(nAccLocal)
      integer acceptorNeighborList( &
            nAcceptorNeighborMax+1, nAccLocal)

      character(len=4) typeLocal(nAtomLocal)

!     *** local variables
      integer i, iDonor, iAcceptor, N, H, C, O, X, XMax
      integer iNeighbor, nNeighbor
      real(chm_real) xN(3), xH(3), xC(3), xO(3), xX(3)
      real(chm_real) rMin, r2Min, rMax, r2Max
      real(chm_real) xHX, yHX, zHX, r2HX
      real(chm_real) xOX, yOX, zOX, r2OX
      real(chm_real) fR
      real(chm_real) fRdxN(3), fRdxH(3), fRdxC(3), fRdxO(3), fRdxX(3)
      real(chm_real) fThetaNHX
      real(chm_real) fThetaNHXdxN(3), fThetaNHXdxH(3), &
            fThetaNHXdxX(3)
      real(chm_real) fThetaCOX
      real(chm_real) fThetaCOXdxC(3), fThetaCOXdxO(3), &
            fThetaCOXdxX(3)
      real(chm_real) e, eMin, eTot

!     *** devel/013 v3p99
!     parameter(thetaMinNHX = 120.0*TWOPI/360.0,
!    & thetaMaxNHX = 140.0*TWOPI/360.0)
!     *** devel/015 v3p109
!     parameter(thetaMinCOX = 100.0*TWOPI/360.0,
!    & thetaMaxCOX = 120.0*TWOPI/360.0)
!     *** devel/055
      real(chm_real),parameter :: thetaMinNHX =  80.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxNHX = 100.0*TWOPI/360.0
!     *** devel/055
      real(chm_real),parameter :: thetaMinCOX =  80.0*TWOPI/360.0
      real(chm_real),parameter :: thetaMaxCOX = 100.0*TWOPI/360.0

!     *** the only reason for the existence of this variable
!     *** is to preserve the symmetric appearance of the
!     *** "d?Local(i) =" equations
      do i = 1, 3
         fRdxN(i) = 0.0
         fRdxC(i) = 0.0
      enddo

!     *** deal with donor ( typically N-H : X ) interactions

      energyHbondBuriedDonor = 0.0

      if( useRushBDON )then
         rMin = rHXMin

         eTot = 0.0

         do iDonor = 1, nDonLocal
            nNeighbor =  &
                  donorNeighborlist(nDonorNeighborMax+1,iDonor)
            N = iDonLocal(iDonor)
            H = iHd1Local(iDonor)
!         *** skip C-H hbond donors
            if( typeLocal(N)(1:1) .ne. "C" )then
               XMax = 0
               eMin = 0.0
               do iNeighbor = 1, nNeighbor
                  X = donorNeighborlist(iNeighbor, iDonor)

                  xHX = xLocal(H) - xLocal(X)
                  yHX = yLocal(H) - yLocal(X)
                  zHX = zLocal(H) - zLocal(X)

                  if( typeLocal(X)(1:1) .eq. "H" )then
                     rMax = rHXMaxH
                  else
                     rMax = rHXMaxNotH
                  endif
                  r2Max = rMax*rMax

                  r2HX = xHX*xHX + yHX*yHX + zHX*zHX

                  if( r2HX .le. r2Max )then

                     xN(1) = xLocal(N)
                     xN(2) = yLocal(N)
                     xN(3) = zLocal(N)

                     xH(1) = xLocal(H)
                     xH(2) = yLocal(H)
                     xH(3) = zLocal(H)

                     xX(1) = xLocal(X)
                     xX(2) = yLocal(X)
                     xX(3) = zLocal(X)

                     call rush_f_theta(thetaMinNHX, thetaMaxNHX, &
                           xN, xH, xX, &
                           fThetaNHX, fThetaNHXdxN, fThetaNHXdxH, fThetaNHXdxX)

                     if( fThetaNHX .gt. 0.0 )then

                        call rush_f_r(rMin, rMax, xH, xX, &
                              fR, fRdxH, fRdxX)

                        e = eHbondDonor*( fR*fThetaNHX )

                        dxLocal(N) = dxLocal(N) &
                              + eHbondDonor * (fRdxN(1)*fThetaNHX &
                              + fR*fThetaNHXdxN(1))
                        dyLocal(N) = dyLocal(N) &
                              + eHbondDonor * (fRdxN(2)*fThetaNHX &
                              + fR*fThetaNHXdxN(2))
                        dzLocal(N) = dzLocal(N) &
                              + eHbondDonor * (fRdxN(3)*fThetaNHX &
                              + fR*fThetaNHXdxN(3))

                        dxLocal(H) = dxLocal(H) &
                              + eHbondDonor * (fRdxH(1)*fThetaNHX &
                              + fR*fThetaNHXdxH(1))
                        dyLocal(H) = dyLocal(H) &
                              + eHbondDonor * (fRdxH(2)*fThetaNHX &
                              + fR*fThetaNHXdxH(2))
                        dzLocal(H) = dzLocal(H) &
                              + eHbondDonor * (fRdxH(3)*fThetaNHX &
                              + fR*fThetaNHXdxH(3))

                        dxLocal(X) = dxLocal(X) &
                              + eHbondDonor * (fRdxX(1)*fThetaNHX &
                              + fR*fThetaNHXdxX(1))
                        dyLocal(X) = dyLocal(X) &
                              + eHbondDonor * (fRdxX(2)*fThetaNHX &
                              + fR*fThetaNHXdxX(2))
                        dzLocal(X) = dzLocal(X) &
                              + eHbondDonor * (fRdxX(3)*fThetaNHX &
                              + fR*fThetaNHXdxX(3))

                        eTot = eTot + e

!                 if( e .lt. eMin )then
!                   eMin = e
!                   XMax = X
!                 endif

                     endif
                  endif
               enddo
!           if( XMax .ne. 0 )then
               if( 0 .ne. 0 )then
!             *** calculate energy and forces for hbond burial

                  xX(1) = xLocal(XMax)
                  xX(2) = yLocal(XMax)
                  xX(3) = zLocal(XMax)

                  if( typeLocal(XMax)(1:1) .eq. "H" )then
                     rMax = rHXMaxH
                  else
                     rMax = rHXMaxNotH
                  endif

                  call rush_f_r(rMin, rMax, xH, xX, &
                        fR, fRdxH, fRdxX)

                  call rush_f_theta(thetaMinNHX, thetaMaxNHX, &
                        xN, xH, xX, &
                        fThetaNHX, fThetaNHXdxN, fThetaNHXdxH, fThetaNHXdxX)

                  dxLocal(N) = dxLocal(N) &
                        + eHbondDonor * (fRdxN(1)*fThetaNHX &
                        + fR*fThetaNHXdxN(1))
                  dyLocal(N) = dyLocal(N) &
                        + eHbondDonor * (fRdxN(2)*fThetaNHX &
                        + fR*fThetaNHXdxN(2))
                  dzLocal(N) = dzLocal(N) &
                        + eHbondDonor * (fRdxN(3)*fThetaNHX &
                        + fR*fThetaNHXdxN(3))

                  dxLocal(H) = dxLocal(H) &
                        + eHbondDonor * (fRdxH(1)*fThetaNHX &
                        + fR*fThetaNHXdxH(1))
                  dyLocal(H) = dyLocal(H) &
                        + eHbondDonor * (fRdxH(2)*fThetaNHX &
                        + fR*fThetaNHXdxH(2))
                  dzLocal(H) = dzLocal(H) &
                        + eHbondDonor * (fRdxH(3)*fThetaNHX &
                        + fR*fThetaNHXdxH(3))

                  dxLocal(XMax) = dxLocal(XMax) &
                        + eHbondDonor * (fRdxX(1)*fThetaNHX &
                        + fR*fThetaNHXdxX(1))
                  dyLocal(XMax) = dyLocal(XMax) &
                        + eHbondDonor * (fRdxX(2)*fThetaNHX &
                        + fR*fThetaNHXdxX(2))
                  dzLocal(XMax) = dzLocal(XMax) &
                        + eHbondDonor * (fRdxX(3)*fThetaNHX &
                        + fR*fThetaNHXdxX(3))

                  eTot = eTot + eMin
               endif
            endif

         enddo

         energyHbondBuriedDonor = eTot

      endif  ! if( useRushBDON )

!     *** deal with C=O : X  interactions

      energyHbondBuriedAcceptor = 0.0

      if( useRushBACC )then
         rMin = rOXMin

         eTot = 0.0

         do iAcceptor = 1, nAccLocal
            nNeighbor =  &
                  acceptorNeighborlist(nAcceptorNeighborMax+1,iAcceptor)
            O = iAccLocal(iAcceptor)
            C = iAc1Local(iAcceptor)
!         *** skip C-H hbond acceptors
!         if( typeLocal(N)(1:1) .ne. "C" )then
            XMax = 0
            eMin = 0.0
            do iNeighbor = 1, nNeighbor
               X = acceptorNeighborlist(iNeighbor, iAcceptor)

               xOX = xLocal(O) - xLocal(X)
               yOX = yLocal(O) - yLocal(X)
               zOX = zLocal(O) - zLocal(X)

               if( typeLocal(X)(1:1) .eq. "H" )then
                  rMax = rOXMaxH
               else
                  rMax = rOXMaxNotH
               endif
               r2Max = rMax*rMax

               r2OX = xOX*xOX + yOX*yOX + zOX*zOX

               if( r2OX .le. r2Max )then

                  xC(1) = xLocal(C)
                  xC(2) = yLocal(C)
                  xC(3) = zLocal(C)

                  xO(1) = xLocal(O)
                  xO(2) = yLocal(O)
                  xO(3) = zLocal(O)

                  xX(1) = xLocal(X)
                  xX(2) = yLocal(X)
                  xX(3) = zLocal(X)

                  call rush_f_theta(thetaMinCOX, thetaMaxCOX, &
                        xC, xO, xX, &
                        fThetaCOX, fThetaCOXdxC, fThetaCOXdxO, fThetaCOXdxX)

                  if( fThetaCOX .gt. 0.0 )then

                     call rush_f_r(rMin, rMax, xO, xX, &
                           fR, fRdxO, fRdxX)

                     e = eHbondAcceptor*( fR*fThetaCOX )

!                 if( e .lt. eMin )then
!                   eMin = e
!                   XMax = X
!                 endif

                     dxLocal(C) = dxLocal(C) &
                           + eHbondAcceptor * (fRdxC(1)*fThetaCOX &
                           + fR*fThetaCOXdxC(1))
                     dyLocal(C) = dyLocal(C) &
                           + eHbondAcceptor * (fRdxC(2)*fThetaCOX &
                           + fR*fThetaCOXdxC(2))
                     dzLocal(C) = dzLocal(C) &
                           + eHbondAcceptor * (fRdxC(3)*fThetaCOX &
                           + fR*fThetaCOXdxC(3))

                     dxLocal(O) = dxLocal(O) &
                           + eHbondAcceptor * (fRdxO(1)*fThetaCOX &
                           + fR*fThetaCOXdxO(1))
                     dyLocal(O) = dyLocal(O) &
                           + eHbondAcceptor * (fRdxO(2)*fThetaCOX &
                           + fR*fThetaCOXdxO(2))
                     dzLocal(O) = dzLocal(O) &
                           + eHbondAcceptor * (fRdxO(3)*fThetaCOX &
                           + fR*fThetaCOXdxO(3))

                     dxLocal(X) = dxLocal(X) &
                           + eHbondAcceptor * (fRdxX(1)*fThetaCOX &
                           + fR*fThetaCOXdxX(1))
                     dyLocal(X) = dyLocal(X) &
                           + eHbondAcceptor * (fRdxX(2)*fThetaCOX &
                           + fR*fThetaCOXdxX(2))
                     dzLocal(X) = dzLocal(X) &
                           + eHbondAcceptor * (fRdxX(3)*fThetaCOX &
                           + fR*fThetaCOXdxX(3))

                     eTot = eTot + e
                  endif
               endif
            enddo
!           if( XMax .ne. 0 )then
            if( 0 .ne. 0 )then
!             *** calculate energy and forces for hbond burial

               xX(1) = xLocal(XMax)
               xX(2) = yLocal(XMax)
               xX(3) = zLocal(XMax)

               if( typeLocal(XMax)(1:1) .eq. "H" )then
                  rMax = rOXMaxH
               else
                  rMax = rOXMaxNotH
               endif

               call rush_f_r(rMin, rMax, xO, xX, &
                     fR, fRdxO, fRdxX)

               call rush_f_theta(thetaMinCOX, thetaMaxCOX, &
                     xC, xO, xX, &
                     fThetaCOX, fThetaCOXdxC, fThetaCOXdxO, fThetaCOXdxX)

!                write(*,'(a,3i6)')
!      &          "COX = ", C, O, XMax
!                write(*,'(a,f10.5)')
!      &          "fThetaCOX = ", fThetaCOX
!                write(*,'(a,3f10.5)')
!      &          "fThetaCOXdxC = ",
!      &          fThetaCOXdxC(1),
!      &          fThetaCOXdxC(2),
!      &          fThetaCOXdxC(3)
!                write(*,'(a,3f10.5)')
!      &          "d?local(C) = ",
!      &          dxLocal(C),
!      &          dyLocal(C),
!      &          dzLocal(C)

               dxLocal(C) = dxLocal(C) &
                     + eHbondAcceptor * (fRdxC(1)*fThetaCOX &
                     + fR*fThetaCOXdxC(1))
               dyLocal(C) = dyLocal(C) &
                     + eHbondAcceptor * (fRdxC(2)*fThetaCOX &
                     + fR*fThetaCOXdxC(2))
               dzLocal(C) = dzLocal(C) &
                     + eHbondAcceptor * (fRdxC(3)*fThetaCOX &
                     + fR*fThetaCOXdxC(3))

               dxLocal(O) = dxLocal(O) &
                     + eHbondAcceptor * (fRdxO(1)*fThetaCOX &
                     + fR*fThetaCOXdxO(1))
               dyLocal(O) = dyLocal(O) &
                     + eHbondAcceptor * (fRdxO(2)*fThetaCOX &
                     + fR*fThetaCOXdxO(2))
               dzLocal(O) = dzLocal(O) &
                     + eHbondAcceptor * (fRdxO(3)*fThetaCOX &
                     + fR*fThetaCOXdxO(3))

               dxLocal(XMax) = dxLocal(XMax) &
                     + eHbondAcceptor * (fRdxX(1)*fThetaCOX &
                     + fR*fThetaCOXdxX(1))
               dyLocal(XMax) = dyLocal(XMax) &
                     + eHbondAcceptor * (fRdxX(2)*fThetaCOX &
                     + fR*fThetaCOXdxX(2))
               dzLocal(XMax) = dzLocal(XMax) &
                     + eHbondAcceptor * (fRdxX(3)*fThetaCOX &
                     + fR*fThetaCOXdxX(3))

!                write(*,'(a,3f10.5)')
!      &          "d?local(C) = ",
!      &          dxLocal(C),
!      &          dyLocal(C),
!      &          dzLocal(C)
!                write(*,'(a,3f10.5)')
!      &          "d?local(O) = ",
!      &          dxLocal(O),
!      &          dyLocal(O),
!      &          dzLocal(O)
!                write(*,'(a,3f10.5)')
!      &          "d?local(X) = ",
!      &          dxLocal(XMax),
!      &          dyLocal(XMax),
!      &          dzLocal(XMax)

               eTot = eTot + eMin
            endif
!         endif

         enddo

         energyHbondBuriedAcceptor = eTot
      endif  ! if( useRushBACC )

!     call set_param("HBND", energyHbondBuriedDonor)
!     call set_param("HBNC", energyHbondDipoleAngularAcceptor)

      return
   end subroutine rush_hbond_buried_energy
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                  END protein-solvent h-bond subroutines            CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                BEGIN aromatic interaction subroutines              CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_aromatic( &
         energyAromatic, &
         rushKARO, &
         rushCutoffAromatic, &
         nAtomLocal,  &
         xLocal, yLocal, zLocal,  &
         dxLocal, dyLocal, dzLocal, &
         deltaR, off)

  use memory
  use stream
!  PRNLEV
!  OUTU
      implicit none

!     *** dummy parameters
      real(chm_real)  energyAromatic, rushKARO, rushCutoffAromatic
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
            dzLocal(nAtomLocal)
      real(chm_real)  deltaR
      logical off

!     *** local variables
      integer,save :: oldSize = -1
      real(chm_real),allocatable,dimension(:),save :: xOld, yOld, zOld
      integer :: nPairMax = 100000, nPair = 0
      integer,allocatable,dimension(:),save :: pairList
      integer iatom, jatom
      real(chm_real)  r2, r2Cut
      character(len=8) resName
      logical :: fillList = .true.

      if( off )then

         if (allocated(xOld)) then
            call chmdealloc('rush.src','rush_aromatic','xOld',oldSize,crl=xOld)
            call chmdealloc('rush.src','rush_aromatic','yOld',oldSize,crl=yOld)
            call chmdealloc('rush.src','rush_aromatic','zOld',oldSize,crl=zOld)
         endif
         oldSize = -1

         if (allocated(pairList)) then
            call chmdealloc('rush.src','rush_aromatic','pairList', &
                  nPairMax,intg=pairList)
         endif
         fillList = .true.
         nPairMax = 100000
         nPair = 0

      else

         if (oldSize /= nAtomLocal) then
            oldSize = nAtomLocal
            if (allocated(xOld)) then
               call chmrealloc('rush.src','rush_aromatic','xOld',oldSize,crl=xOld)
               call chmrealloc('rush.src','rush_aromatic','yOld',oldSize,crl=yOld)
               call chmrealloc('rush.src','rush_aromatic','zOld',oldSize,crl=zOld)
            else
               call chmalloc('rush.src','rush_aromatic','xOld',oldSize,crl=xOld)
               call chmalloc('rush.src','rush_aromatic','yOld',oldSize,crl=yOld)
               call chmalloc('rush.src','rush_aromatic','zOld',oldSize,crl=zOld)
            endif
            call rush_old_init(nAtomLocal,xLocal,yLocal, &
                  zLocal, xOld, yOld, zOld, deltaR)
         endif

         if( rush_update_neighborlist( &
               nAtomLocal, xLocal, yLocal, zLocal, &
               xOld, yOld, zOld, deltaR) )then

            if( PRNLEV .GE. 4 ) &
                  write(OUTU, '(a)') &
                  " rush_aromatic: neighborlist update"

            if (.not. allocated(pairList)) then
               call chmalloc('rush.src','rush_aromatic','pairList', &
                     nPairMax,intg=pairList)
            endif

            fillList = .true.

            do while( fillList )
!           *** fill the pairlist
               call rush_aromatic_genlist( &
                     nAtomLocal, xLocal, yLocal, zLocal, &
                     nPairMax, nPair, pairList,  &
                     rushCutoffAromatic, deltaR, fillList )

!           *** if the there wasnt enough memory, grow the pairlist
!           *** and fill it again
               if( fillList )then
                  nPairMax = 2 * nPairMax
                  call chmrealloc('rush.src','rush_aromatic','pairList', &
                        nPairMax,intg=pairList)
               endif
            enddo
         endif

         call rush_aromatic_energy( &
               energyAromatic, &
               rushKARO, &
               nPairMax, nPair, pairList, &
               nAtomLocal, xLocal, yLocal, zLocal, &
               dxLocal, dyLocal, dzLocal, &
               rushCutoffAromatic)

      endif  ! if( off )

   end subroutine rush_aromatic

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_aromatic_genlist( &
         nAtomLocal, xLocal, yLocal, zLocal, &
         nPairMax, nPair, pairList,  &
         cutOff, deltaR, fillList )

  use psf
!     *** dummy parameters
      implicit none
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      integer nPairMax, nPair
      integer pairList(2, nPairMax)
      real(chm_real)  cutOff, deltaR
      logical fillList

!     *** local variables
      integer ires, jres
      integer iatom, jatom, &
            iatomfirst, iatomlast, &
            jatomfirst, jatomlast
      real(chm_real)  r2, r2Cut
      character(len=8) resName
      logical iAromatic, jAromatic

      nPair = 0
      fillList = .false.

      r2Cut = ( cutOff + 2.0 * deltaR )**2

      do ires = 1, NRES - 1
         resName = RES( ires )
         iAromatic = ( resName .eq. 'PHE' ) &
               .or. ( resName .eq. 'TRP' ) &
               .or. ( resName .eq. 'TYR' )
         if( iAromatic )then
            iatomfirst = IBASE( ires ) + 1
            iatomlast = IBASE( ires + 1 )
            do jres = ires+1, NRES
               resName = RES( jres )
               jAromatic = ( resName .eq. 'PHE' ) &
                     .or. ( resName .eq. 'TRP' ) &
                     .or. ( resName .eq. 'TYR' )
               if( jAromatic )then
                  jatomfirst = IBASE( jres ) + 1
                  jatomlast = IBASE( jres + 1 )
                  do iatom = iatomfirst, iatomlast
                     do jatom = jatomfirst, jatomlast
                        r2 = ( xLocal(jatom) - xLocal(iatom) )**2 &
                              + ( yLocal(jatom) - yLocal(iatom) )**2 &
                              + ( zLocal(jatom) - zLocal(iatom) )**2
                        if( r2 .lt. r2Cut )then
                           nPair = nPair + 1
                           pairlist(1, nPair) = iatom
                           pairlist(2, nPair) = jatom
                           if( nPair .eq. nPairMax )then
!                     *** we need more memory
                              fillList = .true.
                              return
                           endif
                        endif
                     enddo
                  enddo
               endif
            enddo
         endif
      enddo

   end subroutine rush_aromatic_genlist

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_aromatic_energy( &
         energyAromatic, &
         rushKARO, &
         nPairMax, nPair, pairList,  &
         nAtomLocal, xLocal, yLocal, zLocal, &
         dxLocal, dyLocal, dzLocal, &
         cutOff)

  use consta
!     parameter CCELEC
  use psf
     use param_store, only: set_param

!     *** dummy parameters
      implicit none
      real(chm_real) energyAromatic, rushKARO
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) dxLocal(nAtomLocal), dyLocal(nAtomLocal), &
            dzLocal(nAtomLocal)
      integer nPairMax, nPair
      integer pairList(2, nPairMax)
      real(chm_real)  cutOff

!     *** local variables
      integer ipair
      integer iatom, jatom
      real(chm_real)  r, r2, r2Cut
      real(chm_real)  delX, delY, delZ
      real(chm_real)  ePair, dEPair, qiqj

      energyAromatic = 0.0
      r2Cut = cutOff**2

      do ipair = 1, nPair
         iatom = pairList(1, ipair)
         jatom = pairList(2, ipair)
         delX = xLocal(jatom) - xLocal(iatom)
         delY = yLocal(jatom) - yLocal(iatom)
         delZ = zLocal(jatom) - zLocal(iatom)
         r2 = delX**2 + delY**2 + delZ**2
         if( r2 .lt. r2Cut )then
            r = sqrt( r2 )
            qiqj = rushKARO * CCELEC * CG(iatom) * CG(jatom)
            ePair = &
                  qiqj * ( 1.0 / r  + r / r2Cut - 2.0 / cutOff )
            dEPair = ( -qiqj / r2 + qiqj / r2Cut ) * 1.0 / r
            dxLocal(jatom) = dxLocal(jatom) + dEPair * delX
            dyLocal(jatom) = dyLocal(jatom) + dEPair * delY
            dzLocal(jatom) = dzLocal(jatom) + dEPair * delZ
            dxLocal(iatom) = dxLocal(iatom) - dEPair * delX
            dyLocal(iatom) = dyLocal(iatom) - dEPair * delY
            dzLocal(iatom) = dzLocal(iatom) - dEPair * delZ
            energyAromatic = energyAromatic + ePair
         endif
      enddo

      call set_param("RARO", energyAromatic)

   end subroutine rush_aromatic_energy
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                 END aromatic interaction subroutines               CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                       BEGIN utility subroutines                    CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_old_init(nAtomLocal,xLocal,yLocal, &
         zLocal, xOld, yOld, zOld, deltaR)

!     *** initializes the ?Old arrays with the ?Local arrays
!     *** + deltaR, so all the coordinates are translated
!     *** by sqrt(3*deltaR**2) = sqrt(3)*deltaR > deltaR
!     Author: Olgun Guvench, 2001-2002

      implicit none

!     *** dummy variables
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal)
      real(chm_real) yLocal(nAtomLocal)
      real(chm_real) zLocal(nAtomLocal)
      real(chm_real) xOld(nAtomLocal)
      real(chm_real) yOld(nAtomLocal)
      real(chm_real) zOld(nAtomLocal)
      real(chm_real) deltaR

!     *** local variables
      integer i

      do i = 1, nAtomLocal
         xOld(i) = xLocal(i) + deltaR
         yOld(i) = yLocal(i) + deltaR
         zOld(i) = zLocal(i) + deltaR
      enddo

      return
   end subroutine rush_old_init

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   logical function rush_update_neighborlist(nAtomLocal, &
         xLocal, yLocal, zLocal, xOld, yOld, zOld, deltaR)

!     Returns true if any atoms have moved more than deltaR

!     Olgun Guvench, 5/2/02

      implicit none

!     *** dummy variables
      integer nAtomLocal
      real(chm_real) xLocal(nAtomLocal), yLocal(nAtomLocal), &
            zLocal(nAtomLocal)
      real(chm_real) xOld(nAtomLocal), yOld(nAtomLocal), &
            zOld(nAtomLocal)
      real(chm_real) deltaR

!     *** local variables
      integer i, j
      real(chm_real) deltaR2, dr2

      rush_update_neighborlist = .false.

      deltaR2 = deltaR*deltaR
      do i = 1, nAtomLocal
         dr2 = (xLocal(i)-xOld(i))**2 &
               + (yLocal(i)-yOld(i))**2 &
               + (zLocal(i)-zOld(i))**2
         if (dr2 >= deltaR2) then
            do j = 1, nAtomLocal
               xOld(j) = xLocal(j)
               yOld(j) = yLocal(j)
               zOld(j) = zLocal(j)
            enddo
            rush_update_neighborlist = .true.
            exit
         endif
      enddo

      return
   end function rush_update_neighborlist

!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_f_r(rMin, rMax, &
         xO, xH, fr, dxO, dxH)

!     *** see v2p103
!     *** rMin and rMax must have units of angstroms
!     Author: Olgun Guvench, 2001-2002

  use consta
!     parameter TWOPI = 2.0*3.14...
      implicit none

!     *** dummy variables
      real(chm_real) rMin, rMax
      real(chm_real) xO(3), xH(3)
      real(chm_real) fr
      real(chm_real) dxO(3), dxH(3)

!     *** local variables
      integer i
      real(chm_real) rMod, cosRMod, dfr
      real(chm_real) xHO(3)
      real(chm_real) r, r2, rMin2, rMax2

!     *** v2p103
      r2 = 0.0
      do i = 1, 3
         xHO(i) = xO(i) - xH(i)
         r2 = r2 + xHO(i)*xHO(i)
      enddo

      rMin2 = rMin * rMin
      rMax2 = rMax * rMax

      if( r2 .le. rMin2 )then

         fr = 1.0
         do i = 1, 3
            dxO(i) = 0.0
            dxH(i) = 0.0
         enddo

      elseif( r2 .le. rMax2 )then

         r = sqrt(r2)
         rMod = (TWOPI/(4.0*(rMax-rMin))) * (r-rMin)
         cosRMod = cos(rMod)
         fr = cosRMod*cosRMod
!       *** v2p103
         dfr = -2.0 * cosRMod * sin(rMod) &
               * (TWOPI/(4.0*r*(rMax-rMin)))

         do i = 1, 3
            dxO(i) = dfr*xHO(i)
            dxH(i) = -dxO(i)
         enddo

      else

         fr = 0.0
         do i = 1, 3
            dxO(i) = 0.0
            dxH(i) = 0.0
         enddo

      endif

      return
   end subroutine rush_f_r
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
   subroutine rush_f_theta(thetaMin, thetaMax, &
         xk, xl, xi, ftheta, dxk, dxl, dxi)

!     *** theta = angle formed by k - l - i
!     *** thetaMin and thetaMax must have units of radians
!     Author: Olgun Guvench, 2001-2002

  use consta
!     parameter TWOPI = 2.0*3.14...
      implicit none

!     *** dummy variables
      real(chm_real) thetaMin, thetaMax
      real(chm_real) xk(3), xl(3), xi(3)
      real(chm_real) ftheta
      real(chm_real) dxk(3), dxl(3), dxi(3)

!     *** local variables
      integer i
      real(chm_real) nu, phi
      real(chm_real) xli(3), xlk(3)
      real(chm_real) rli, rli2, rlk, rlk2
      real(chm_real) li_dot_lk
      real(chm_real) theta, cosTheta, thetaMod, cosThetaMod, sinThetaMod
      real(chm_real) termA, termB1, termB2, dfTheta

!     *** v2 p100,111
      nu = TWOPI/(4.0*(thetaMax-thetaMin))
      phi = 3.0*thetaMax - 4.0*thetaMin

      do i = 1, 3
         xli(i) = xi(i) - xl(i)
      enddo
      rli2 = xli(1)*xli(1) + xli(2)*xli(2) + xli(3)*xli(3)
      rli = sqrt(rli2)
      rli = sqrt(rli2)

      do i = 1, 3
         xlk(i) = xk(i) - xl(i)
      enddo
      rlk2 = xlk(1)*xlk(1) + xlk(2)*xlk(2) + xlk(3)*xlk(3)
      rlk = sqrt(rlk2)

      li_dot_lk = xli(1)*xlk(1) &
            + xli(2)*xlk(2) &
            + xli(3)*xlk(3)

      theta = acos(li_dot_lk/(rli*rlk))

      if( theta .le. thetaMin )then

         fTheta = 0.0
         do i = 1, 3
            dxk(i) = 0.0
            dxl(i) = 0.0
            dxi(i) = 0.0
         enddo

      elseif( theta .le. thetaMax )then

         thetaMod = nu*(theta+phi)

         cosTheta = cos(theta)

         cosThetaMod = cos(thetaMod)
         sinThetaMod = sin(thetaMod)

         termA = 1.0/(rlk*rli)
         termB1 = -li_dot_lk/(rlk*rli*rli*rli)
         termB2 = -li_dot_lk/(rli*rlk*rlk*rlk)

         fTheta = cosThetaMod**2
         dfTheta = 2.0 * nu &
               * cosThetaMod*sinThetaMod &
               * (1.0/sqrt(1.0-cosTheta**2))

         do i = 1, 3
            dxi(i) = dfTheta*(termA*xlk(i)+termB1*xli(i))
            dxk(i) = dfTheta*(termA*xli(i)+termB2*xlk(i))
            dxl(i) = -(dxi(i) + dxk(i))
         enddo

      else

         fTheta = 1.0
         do i = 1, 3
            dxk(i) = 0.0
            dxl(i) = 0.0
            dxi(i) = 0.0
         enddo

      endif

      return
   end subroutine rush_f_theta
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
!C                        END utility subroutines                     CC
!CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
end module rush_mod
