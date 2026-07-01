module mtpl_fcm
#if KEY_MTPL==1 /*(mtpl_main)*/
  ! Christian Kramer, January 2011
  !
  ! This is a module containing a routine that calculates Electrostatic
  ! Interaction Energies based on multipoles up to quadrupoles.
  !
  ! The code is written inspired by the multipole implementation
  ! by Nuria Plattner and Myung-Won Lee
  !
  ! This multipole routine is based on local reference axis systems for each 
  ! multipole. The technical definition of the multipoles is thus completely
  ! different from the previous format and the routines change significantly.
  ! This implementation should allow to calculate multipoles for any system
  ! independent of size and symmetries.
  !
  ! The two major connections to ChARMM are the MTP and the MTPX routine
  ! MTP initializes the multipoles on each atom. MTPX calculates
  ! the electrostatic interaction energies between all charged sites.
  ! In the first implementation the multipole energy is printed to stdout.
  !
  ! -----------------------------------------
  !
  ! Tristan Bereau, July 2012
  !
  ! Implement image atom lists.
  !
  !------------------------------------------
  !
  ! Christian Kramer, December 2012
  !
  ! Speed up energy calculations when using local frames
  ! Implement Forces and Torques
  ! Implement Cutoffs
  !
  ! -----------------------------------------
  !
  ! Tristan Bereau, February 2013
  !
  ! Debugged forces and torques. Corrected conversion of torques into forces. 
  ! Fixed image calculations.
  ! Fixed memory leak due to call to 0th element of INBL.
  !
  ! -----------------------------------------
  !
  ! Tristan Bereau, July 2013
  !
  ! Get rid of IMTPREALLOC. All arrays are large enough from the beginning. 
  ! -> massive performance improvements.
  ! Introduce switching function for simulations with finite cutoff.
  ! Switching function cutoffs now depend on the order of the interaction.
  ! Added PREF keyword to scale all energies and forces by PREF (range: 0.0-1.0).
  !
  !------------------------------------------
  !
  ! Additionally
  !
  !
  ! ChARMM provides:                                                   from module
  !        outu              output unit (stdout?)                               ?
  !        prnlev            amount of details printed                      stream
  !        maxaim            Maximum number of atoms (set at complilation)       ?
  !        NATOM             actual number of atoms                              ?                 
  !        X                 x coordinate                                    coord
  !        Y                 y coordinate                                    coord
  !        Z                 z coordinate                                    coord
  !        JNBL              array that contains atomindices for all             ?
  !                           nonbonded interaction partners to each atom
  !        INBL              array that indexes the end of partners              ?
  !                           within JNBL for each atom
  !                          -> used as list for all nonbonded interactions
  !                           to be calculated
  !        CTONNB            Switching function turn on distance             inbnd
  !                           for nonbonded interactions
  !        CUTNB             Cutoff for electrostatics                       inbnd
  !        CTOFNB            switching function and shifting turn off        inbnd
  !        BOHRR             Bohr radius                                consta_ltm
  !        NATIM             ?                                                   ?
  !        IMELEC            Name for energy term in ETERM
  !        CG                Point charges                                 psf_ltm
  !        DX                Gradient in X-Direction
  !        DY                Gradient in Y-Direction
  !        DZ                Gradient in Z-Direction
  !        MOLN              maximal number of atomic multipole molecules
  !        NMOL              number of primary molecules/fragments
  !
  !
  ! Parameters provided in this module:
  !
  !        QMTPL                           flag for module use
  !        mtp_site(natoms)                flag indicating multipoles on this atom
  !        Dloc(3,natoms)                  array containing local dipole parameters 
  !                                         (D1x,D1y,D1z)
  !        Qloc(5,natoms)                  array containing local Qpole parameters
  !                                         (Q20,Q21c,Q21s,Q22c,Q22s)
  !        Q00_x(natoms)                   flag that indicates whether there is a charge on the atom
  !        Dloc_x(3,natoms)                flag that indicates whether the dipole 
  !                                        coefficients are different from 0.0 (0.00001)  
  !        Qloc_x(5,natoms)                flag that indicates whether the Qpole
  !                                        coefficients are different from 0.0 (0.00001) 
  !        Torque(3,natoms)                array to save the intermediate torques arising on atoms
  !        rank(natoms)                    vector containing the multiple expansion rank [0,1,2]
  !        refkind(natoms)                 vector containing definition of LRA system type
  !        refatms(4,natoms)               list of the reference atoms for each local system 
  !                                         (2 - 4 reference atoms per LRA)
  !        TM(3,3,natoms)                  List of the transformation matrices from local to global frames
  !        
  ! ---------------------------------------------------------
  use chm_kinds
  use dimens_fcm
  implicit none
  logical, save :: QMTPL
  logical, save, dimension(:), allocatable :: mtp_site, Q00_x
  logical, save, dimension(:,:), allocatable :: Dloc_x, Qloc_x
  real(chm_real), save, dimension(:,:), allocatable :: Dloc, Qloc, Torque
  real(chm_real), save, dimension(:,:,:), allocatable :: TM
  integer, dimension(:), save, allocatable :: rank
  integer, dimension(:,:), save, allocatable :: refatms
  character(len=4), dimension(:), save, allocatable :: refkind
  INTEGER, SAVE :: MOLN, MOLSL,NMOL,IMOLL, NUMDA, IMTPIMG, ROFFLGST
  ! Define variables for interaction-dependent switched cutoffs.
  ! Switching function takes two variables: RON and ROFF.
  ! 4 different levels of R^{-N} interactions:
  ! ------------------------------------------------------
  ! |  N    |    Interactions         RON       ROFF     |
  ! ======================================================
  ! |  2    |    1-0                  RON2      ROFF2    |
  ! |  3    |    2-0, 1-1             RON3      ROFF3    |
  ! |  4    |    2-1                  RON4      ROFF4    |
  ! |  5    |    2-2                  RON5      ROFF5    |
  ! ------------------------------------------------------
  real(chm_real), dimension(4), save :: MTPLRON, MTPLROFF
  real(chm_real), save :: PREF

contains

  ! ################

  subroutine MTPL
    ! ---------------------------------------------------------
    ! This subroutine is called from the main charmm program to initialize the
    ! mtp parameters once. It only calls the subroutine that really reads the
    ! parameters and sets the multipole flag on.
    ! ---------------------------------------------------------
    use chm_kinds
    use stream
    use dimens_fcm
    use ensemble
    use parallel
    use psf
#if KEY_MPI==1
    use mpi
#if KEY_PARALLEL==1
#else
    integer COMM_CHARMM 
    COMM_CHARMM = MPI_COMM_WORLD 
#endif
#endif
    implicit none
    integer ierror

    QMTPL = .TRUE.
    IF(PRNLEV.GT.2 .and. mynod.eq.0) &
         WRITE(OUTU,'(A)')' MTPL> MTPL routine activated'
    ! Parse data
    call mtplinit()
#if KEY_MPI==1
    call mpi_barrier(COMM_CHARMM,ierror)
    ! Broadcast data to other nodes
    call mpi_bcast(CG, MAXAIM, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Dloc, 3*MAXAIM, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Qloc, 5*MAXAIM, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Q00_x, MAXAIM, MPI_LOGICAL, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Dloc_x, 3*MAXAIM, MPI_LOGICAL, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Qloc_x, 5*MAXAIM, MPI_LOGICAL, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Torque, 3*MAXAIM, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Rank, MAXAIM, MPI_INTEGER, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Refatms, 4*MAXAIM, MPI_INTEGER, 0, COMM_CHARMM, ierror)
    call mpi_bcast(Refkind, 4*MAXAIM, MPI_CHARACTER, 0, COMM_CHARMM, ierror)
    call mpi_bcast(mtp_site, MAXAIM, MPI_LOGICAL, 0, COMM_CHARMM, ierror)
    call mpi_bcast(MTPLRON, 4, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(MTPLROFF, 4, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(ROFFLGST, 1, MPI_REAL8, 0, COMM_CHARMM, ierror)
    call mpi_bcast(PREF, 1, MPI_REAL8, 0, COMM_CHARMM, ierror)
#endif

    return
  end subroutine MTPL

  ! ##################

  subroutine mtplinit
    ! ---------------------------------------------------------
    ! This subroutine now really does the initialization of the multipole
    ! parameters. The format of the parameter file is discussed in the
    ! documentation. Briefly the order of the atoms must be the same as in the
    ! other input files. Each atom is assigned an atom type, an index, a
    ! multipole rank, the type of the reference axis system, the reference
    !  atoms, the charge, the dipole, the quadrupole.
    !
    ! Dipole parameters are read in spherical harmonic coordinates and stored in
    ! cartesian coordinates, since all equations are expressed in cartesian
    ! (D1x,D1y,D1z) coordinates within their local frame.  Quadrupole parameters
    ! are read and stored in the local frame with their spherical harmonic
    ! coefficients.  
    ! ----------------------------------------------------------
    use chm_kinds
    use stream
    use exfunc
    use dimens_fcm
    use comand
    use psf
!    use param
    use consta
    use coord
    use string
    use memory
    use parallel
    use inbnd
    implicit none

    integer :: inuni                ! The input unit
    integer :: i,j,k,l,m,n,io,a_idx,nrefA
    real(chm_real) :: o,p,q,r,tolparse
    character(len=50) :: wrd1,wrd2
    character(len=230) :: wrd3
    logical :: nonmnp

#if KEY_MTPL_DEBUG==1
    ! For testing time
    real :: t1,t2
    call cpu_time(t1)
#endif

    IMTPIMG = 0 ! becomes 1 if images are used

    ! Allocate the arrays where the multipole information is stored ### This
    ! could be improved by only allocating the arrays for the atoms that carry
    ! multipoles

    call chmalloc('mtpl.src','MTPLINIT','Dloc',3,MAXAIM,crl=Dloc)
    call chmalloc('mtpl.src','MTPLINIT','Qloc',5,MAXAIM,crl=Qloc)
    call chmalloc('mtpl.src','MTPLINIT','Q00_x',MAXAIM,log=Q00_x)
    call chmalloc('mtpl.src','MTPLINIT','Dloc_x',3,MAXAIM,log=Dloc_x)
    call chmalloc('mtpl.src','MTPLINIT','Qloc_x',5,MAXAIM,log=Qloc_x)
    call chmalloc('mtpl.src','MTPLINIT','Torque',3,MAXAIM,crl=Torque)
    call chmalloc('mtpl.src','MTPLINIT','rank',MAXAIM,intg=rank)
    call chmalloc('mtpl.src','MTPLINIT','refatms',4,MAXAIM,intg=refatms)
    call chmalloc('mtpl.src','MTPLINIT','refkind',MAXAIM,ch4=refkind)
    call chmalloc('mtpl.src','MTPLINIT','mtp_site',MAXAIM,log=mtp_site)
    call chmalloc('mtpl.src','MTPLINIT','TM',3,3,MAXAIM,crl=TM)

    ! Keep parsing only on the master node.
    if (mynod.ne.0) return

    ! Initialize the arrays 

    Dloc = 0.0
    Qloc = 0.0
    Q00_x = .false.
    Dloc_x = .false.
    Qloc_x = .false.
    TM = 0.0
    rank = 0
    refatms = 0
    refkind = ''
    mtp_site = .false.
    ! Non-zero thershold for parsing
    tolparse = 0.01

    ! Read in the multipole parameters and fill the arrays

    IF(PRNLEV.GT.2 .and. mynod.eq.0) WRITE(OUTU,'(A)') ' MTPL> MTPLINIT'
    INUNI = GTRMI(COMLYN,COMLEN,'MTPUNIT',-1)

    IF (INUNI.EQ.-1) THEN
       call WRNDIE(-3,'<mtpl.src> mtplinit', &
            'No input file for MTP parameters')
    ELSE
       IF(PRNLEV.GT.2 .and.mynod.eq.0) WRITE(OUTU,'(A)') ' MTPL> Reading input MTP parameters'
    ENDIF

    if(IOLEV.GT.0) then
       do i=1,3
          READ(INUNI,'(A)') wrd3
       enddo
    endif

    ! The error capturing of this routine is not complete yet. The input file
    ! better be absolutely correct.
    if(iolev.gt.0)then
       do i=1,natom
          read(inuni,*,iostat=io)a_idx,wrd1,o,p,q,wrd2,j
          if (io /= 0) then
             write(outu,'(A,I5,A)') &
                  " MTPL> Parsing the first line of atom ",a_idx," failed."
             write(outu,'(A)') " MTPL> Program continues..."
             ! Don't send an error. Assume there are fewer MTP atoms than total
             ! number of atoms in the system.
             exit
          endif
          rank(a_idx) = 0
          mtp_site(a_idx) = .true.
          nonmnp = .false.
          read(inuni,*,iostat=io)wrd1,refkind(a_idx),refatms(:,a_idx)
          if(io /= 0) then
             write(outu,'(A,I5,A)') " MTPL> The LRA definition for atom ",a_idx,&
                  " cannot be read properly."
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'LRA definition cannot be read--check the MTP definition file')
          endif
          read(inuni,*,iostat=io) CG(a_idx)
          if(io /= 0) then
             write(outu,'(A,I5,A)') " MTPL> charge of atom ",a_idx,&
                  " cannot be read properly."
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'Charge of atom cannot be read--check the MTP definition file')
          endif
          read(inuni,*,iostat=io) o,p,q
          if(io /= 0) then
             write(outu,'(A,I5,A)') " MTPL> Q1x of atom ",a_idx,&
                  " cannot be read properly."
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'Dipole term cannot be read--check the MTP definition file')
          endif
          if (abs(p) > tolparse) then
             Dloc(1,a_idx) = p
             Dloc_x(1,a_idx) = .true.
             nonmnp = .true.
             rank(a_idx) = 1
          endif
          if (abs(q) > tolparse) then
             Dloc(2,a_idx) = q
             Dloc_x(2,a_idx) = .true.
             nonmnp = .true.
             rank(a_idx) = 1          
          endif
          if (abs(o) > tolparse) then
             Dloc(3,a_idx) = o
             Dloc_x(3,a_idx) = .true.
             nonmnp = .true.
             rank(a_idx) = 1             
          endif
          read(inuni,*,iostat=io)Qloc(:,a_idx)
          if(io /= 0) then
             write(outu,'(A,I5,A)') " MTPL> Q2x of atom ",a_idx,&
                  " cannot be read properly."
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'Quadrupole term cannot be read--check the MTP definition file')
          endif
          do j=1,5
             if (abs(Qloc(j,a_idx)) > tolparse) then
                Qloc_x(j,a_idx) = .true.
                nonmnp = .true.
                rank(a_idx) = 2           
             else
                Qloc(j,a_idx) = 0.0d0
             endif
          enddo
          read(inuni,'(A)',iostat=io)wrd1
          if(io /= 0 .or. trim(wrd1) /= '') &
               call WRNDIE(-3,'<mtpl.src> mtplinit', &
               'Expecting empty line--check the MTP definition file')
          if (nonmnp .eqv. .false.) mtp_site(a_idx) = .false.
          if (PRNLEV.GT.2) then  
             write(outu,'(A,I5,A,f9.4,A,I1)')" MTPL> Parsed atom: ",a_idx,&
                  "; charge: ",CG(a_idx),&
                  "; rank: ",rank(a_idx)
             write(outu,'(A,f7.3,A,f7.3,A,f7.3)')" MTPL>  dipole: ", &
                  Dloc(1,a_idx),", ",Dloc(2,a_idx),", ",Dloc(3,a_idx)
             write(outu,'(A,f7.3,A,f7.3,A,f7.3,A,f7.3,A,f7.3)') &
                  " MTPL>  quadp.: ", Qloc(1,a_idx),", ",&
                  Qloc(2,a_idx),", ",Qloc(3,a_idx),", ",&
                  Qloc(4,a_idx),", ",Qloc(5,a_idx)
          endif
       enddo
    endif

    DO I=1,NATOM
       if (abs(CG(I)) > tolparse) Q00_x(I) = .true.
       IF (MTP_SITE(I)) THEN
          ! Make sure every MTP site has the right number of neighbors
          nrefA = 4
          do k = 1,4
             if (refatms(k,i) == 0) nrefA = nrefA - 1
          enddo          
          if (nrefA < 1) then
             write(outu,'(A,I5)') &
                  " MTPL> Number of reference atoms is too small for atom ",i
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'Number of reference atoms too small for MTP')
          endif
          ! Sanity check: Compare kind of reference axis system with number of
          ! neighbors. 
          if (trim(refkind(i)) == 'c3v') then
             if (nrefA < 3 .or. nrefA > 4) then
                write(outu,'(A,I5)') &
                     ' MTPL> c3v ref axis system: wrong number of atoms for ',i
                call WRNDIE(-3,'<mtpl.src> mtplinit', &
                     'Problem with c3v reference axis system for MTP')
             endif
          elseif (trim(refkind(i)) == 'ter') then
             if (nrefA < 2 .or. nrefA > 4)then
                write(outu,'(A,I5)') &
                     ' MTPL> ter ref axis system: wrong number of atoms for ',i
                call WRNDIE(-3,'<mtpl.src> mtplinit', &
                     'Problem with ter reference axis system for MTP')
             endif
          elseif (trim(refkind(i)) == 'int') then
             if (nrefA < 2 .or. nrefA > 4)then
                write(outu,'(A,I5)') &
                     ' MTPL> int ref axis system: wrong number of atoms for ',i
                call WRNDIE(-3,'<mtpl.src> mtplinit', &
                     'Problem with ref reference axis system for MTP')
             endif
          elseif (trim(refkind(i)) == 'lin') then
             if (nrefA < 1 .or. nrefA > 2) then
                write(outu,'(A,I5)') &
                     ' MTPL> lin ref axis system: wrong number of atoms for ',i
                call WRNDIE(-3,'<mtpl.src> mtplinit', &
                     'Problem with ref reference axis system for MTP')
             endif
          else
             write(outu,'(A,I5)') &
                  " MTPL> Reference axis system not properly defined for atom ",i
             call WRNDIE(-3,'<mtpl.src> mtplinit', &
                  'Problem with reference axis system for MTP')
          endif
       ENDIF
    ENDDO

    ! Initialize image counting parameters
    NMOL = NATOM
    IMOLL = NATOM
    MOLN = NATOM

#if KEY_MTPL_DEBUG==1
    if (prnlev.gt.5) then
       ! Timing
       call cpu_time(t2)
       write(outu,'(A,f9.4,A)') &
            " MTPL> Time taken to read the input file and allocate the arrays: ",&
            t2-t1, " seconds."
       call flush(outu)
    endif
#endif

    ! Set cutoffs for switching functions
    ! By default, RON{*} = CTONNB, ROFF{*} = CTOFNB
    ! Convert to BOHR.  Store quantity _squared_.
    MTPLRON(1)  = GTRMF(COMLYN,COMLEN,'RON2', CTONNB)
    MTPLRON(2)  = GTRMF(COMLYN,COMLEN,'RON3', CTONNB)
    MTPLRON(3)  = GTRMF(COMLYN,COMLEN,'RON4', CTONNB)
    MTPLRON(4)  = GTRMF(COMLYN,COMLEN,'RON5', CTONNB)
    MTPLROFF(1) = GTRMF(COMLYN,COMLEN,'ROFF2',CTOFNB)
    MTPLROFF(2) = GTRMF(COMLYN,COMLEN,'ROFF3',CTOFNB)
    MTPLROFF(3) = GTRMF(COMLYN,COMLEN,'ROFF4',CTOFNB)
    MTPLROFF(4) = GTRMF(COMLYN,COMLEN,'ROFF5',CTOFNB)
    ROFFLGST    = 0.D0
    PREF        = GTRMF(COMLYN,COMLEN,'PREF',1.D0)
    do I=1,4
       if (prnlev.gt.4) then
          write(outu,'(A,I1,A,f9.4,A)') &
               " MTPL> Switching coefficient R_on  for 1/R^{",I+1,"} interactions: ",&
               MTPLRON(I)," Angstroems."
          write(outu,'(A,I1,A,f9.4,A)') &
               " MTPL> Switching coefficient R_off for 1/R^{",I+1,"} interactions: ",&
               MTPLROFF(I)," Angstroems."
       endif

       if (MTPLROFF(I) .le. MTPLRON(I)) call WRNDIE(-3,&
            '<mtpl.src> mtplinit',"switching parameters are inconsistent.")
       if (MTPLROFF(I) .gt. CTOFNB) call WRNDIE(-3,&
            '<mtpl.src> mtplinit',"MTP cutoff is longer than CTOFNB.")
       if (MTPLROFF(I) .gt. ROFFLGST) ROFFLGST = MTPLROFF(I)
    enddo
    if (prnlev.gt.4) then
       write(outu,'(A,f9.4)') &
            " MTPL> Apply prefactor ", PREF
       if (PREF .lt. 0.D0 .or. PREF .gt. 1.D0) call WRNDIE(-3,&
            '<mtpl.src> mtplinit',"MTP PREF is out of range (0.0-1.0).")
    endif


    return
  end subroutine mtplinit

  ! ##################

  subroutine MTPLX(NATOMX,JNBL,INBL)
    ! ---------------------------------------------------------
    ! This is the main subroutine for calculating the energies
    ! It is called from the energy evaluation function of ChARMM 
    !
    ! The parameters that go in here are
    ! NATOMX              :        number of atoms
    ! JNBL                :        neighbours for E-static calculation
    ! INBL                :        index to JNBL
    !
    ! In the present version the program writes the multipole energy calculated
    ! directly to stdout.
    ! ---------------------------------------------------------
    use chm_kinds
    use stream
    use dimens_fcm
    use consta
    use coord
    use energym
    use deriv
    use psf
    use image
    use number
    use inbnd
    use vector
    use parallel
#if KEY_MPI==1
    use mpi
#if KEY_PARALLEL==1
#else
    integer COMM_CHARMM 
    COMM_CHARMM = MPI_COMM_WORLD 
#endif
#endif
    implicit none
    integer, intent(in) :: NATOMX
    integer, intent(in), dimension(:) :: JNBL,INBL

    ! local variables
    real(chm_real), dimension(3) :: norm_il, ra, rb, cr_a, cr_b, &
         cr_ab, cr_ba
    real(chm_real), dimension(3,3) :: cab, ra2, rb2, rab, TM_i, TM_l
    real(chm_real), dimension(3,3,3) :: ra2b, rb2a
    real(chm_real), dimension(3,3,3,3) :: ra2b2
    real(chm_real), dimension(3,NATOMX) :: XYZb
    real(chm_real) :: o,p,q,r,a2b,b2a,EMTP,R1,R1B,R2B,CTOFNB2,CTONNB2,R2,&
         Rm1,Rm2,Rm3,Rm4,Rm5,Rm6,EMTPa,EMTPtot
    real(chm_real), dimension(4) :: SWITC,DSWITC
    integer :: i,j,k,l,m,n,ii,ij,ik,il,inblim1,ierror

    ! For calculating derivatives
    real(chm_real), dimension(3) :: dRdT1, dRdT2
    real(chm_real), dimension(3,3) :: dRadT1, dRadT2, dRadR1
    real(chm_real), dimension(3,3) :: dRbdT1, dRbdT2, dRbdR2
    real(chm_real), dimension(3,3,3) :: dCabdR1, dCabdR2, dCbadR2, dCbadR1

    ! Storing Derivatives
    real(chm_real), dimension(3) :: F1, F2, tau1, tau2

    ! For timing
    real :: t1,t2

    Torque = 0.0
    EMTPtot=0.d0

#if KEY_MTPL_DEBUG==1
    if (prnlev.gt.3 .and. mynod.eq.0) &
         write(outu,'(A)') " MTPL> Entering MTPLX subroutine"
#endif

    IF (NATOMX .EQ. NATIM) THEN
       NMOL = IMOLL
    ELSE
       NMOL = NATOMX
    ENDIF

    a2b = 1.889726d0
    b2a = 0.52917720859d0

    ! Convert all units to Bohr

    XYZb(1,:) = X(1:NATOMX)*a2b
    XYZb(2,:) = Y(1:NATOMX)*a2b
    XYZb(3,:) = Z(1:NATOMX)*a2b

    ! !!!!!!!!!!!!!!!!!!!!!
    ! Calculate interaction energies
    ! !!!!!!!!!!!!!!!!!!!!!
    ! Charge-charge interactions for non-multipole sites are calculated within
    ! the standard charmm module. Charges are taken from the CG vector if
    ! mtp_site is false, else charges read in in mtp_init are used.
    !
    ! The interaction terms are taken from A. Stone's Book "The theory of
    ! intermolecular forces", Appendix F. 
    !
    !        JNBL                array that contains atomindices for all
    !                        nonbonded interaction partners to each atom
    !        INBL                array that indexes the end of partners 
    !                        within JNBL for each atom
    !                -> used as list for all nonbonded interactions to
    !                   be calculated

#if KEY_MTPL_DEBUG==1
    if (prnlev.gt.8 .and. mynod.eq.0) then
       write(outu,'(A)') " MTPL> Nonbonded Lists:"
       write(outu,'(A)') " MTPL> INBL: "
       do i=1,size(inbl)
          write(outu,'(I5)') inbl(i)
       enddo
       write(outu,'(A)') " MTPL> JNBL: "
       do i=1,size(jnbl)
          write(outu,'(I5)') jnbl(j)
       enddo
       write(outu,'(A)') ""
    endif
#endif

#if KEY_MTPL_DEBUG==1
    ! Timing    
    call cpu_time(t1)
#endif

    ! Calculate Transformation Matrices for all atoms with multipoles
    do i=1,NATOMX
       if (mtp_site(i)) then
#if KEY_MTPL_DEBUG==1
          if (prnlev.gt.5 .and. mynod.eq.0) then
             write(outu,'(A,I5)') &
                  " MTPL> Calculating local axis system for atom",i
          endif
#endif          
          call get_local_xyz(i,TM_i)
          TM(:,:,i) = TM_i
       endif
    enddo

#if KEY_MTPL_DEBUG==1
    ! Timing    
    if (prnlev.gt.5 .and. mynod.eq.0) then
       call cpu_time(t2)
       write(outu,'(A,f9.4,A)') &
            " MTPL> Time taken to calculate the local axis systems ",&
            t2-t1,' seconds.'
       call flush(outu)
    endif
#endif

    ! Iterate over all pairs of atoms indexed in the INBL and JNBL list
    EMTP = 0.0
    do i=1,NATOMX
       n = INBL(i)
       if (i > 1) then
          n = INBL(i)-INBL(i-1)
       endif
       if (n > 0) then
#if KEY_MTPL_DEBUG==1
          if (prnlev.gt.3) write(outu,'(A,I5)') &
               " MTPL> Calculating MTP interactions for Atom ",i
#endif          

          do j=1,n
             ! INBL(i-1), avoid mem leak at i-1=0
             inblim1 = 0
             if (i > 1) then
                inblim1 = INBL(i-1)
             endif
             l = abs(JNBL(inblim1+j))
#if KEY_MTPL_DEBUG==1
             if (prnlev.gt.4) then
                write(outu,'(A,I5)') " MTPL>   with Atom ",l
                write(outu,'(A,I1,A,I1)')" MTPL>  ranks: ",rank(i),&
                     " and ",rank(l)
             endif
#endif
             if (mtp_site(i) .OR. mtp_site(l)) then  
                ! We must go inside the loop if either one atom has MTPs.
                ra = 0.0
                rb = 0.0
                ra2 = 0.0
                rb2 = 0.0
                rab = 0.0
                ra2b = 0.0
                rb2a = 0.0
                ra2b2 = 0.0

                F1 = 0.0
                F2 = 0.0
                tau1 = 0.0
                tau2 = 0.0

                EMTPa = 0.0

                ! Calculate distance between i and l      
                R2  = dot_product(XYZb(:,l)-XYZb(:,i),XYZb(:,l)-XYZb(:,i))
                R1  = sqrt(R2)
                R1B = R1*BOHRR
                R2B = R1B**2
                ! the atom pairs in the list are
                ! not necessarily within the cutoffs 
                if (R1B .GT. ROFFLGST) cycle               

                norm_il = (XYZb(:,l)-XYZb(:,i))/R1
#if KEY_MTPL_DEBUG==1
                if (R1 < 0.1) then
                   write(outu,'(A,I5,A,I5,A)')" MTPL> Atoms ",i,&
                        " and ",l," are closer than 0.1 Bohr."
                   call WRNDIE(-3,'<mtpl.src> mtplx', &
                        'Two atoms are too close during MTP evalution.')
                endif
#endif

                ! Populate SWITC and DSWITC arrays. Computes switching term and
                ! derivative.  Compute as few necessary terms as there are MTP
                ! moments. 
                do II=1,rank(I)+rank(L)
                   call switch(R1B,R2B,MTPLROFF(II)**2,MTPLRON(II)**2,&
                        SWITC(II),DSWITC(II))  
                enddo

                ! Precalculate derivatives for elemental terms
                do ii = 1,3
                   dRdT1(ii) = (-1)*norm_il(ii)
                   dRdT2(ii) = norm_il(ii)
                enddo

                ! Calculate unit vector between i and l, all direction cosines and
                ! their squares and products.
                Rm1 = 1.0/R1
                Rm2 = Rm1 * Rm1
                Rm3 = Rm1 * Rm2
                Rm4 = Rm2 * Rm2
                Rm5 = Rm1 * Rm4
                Rm6 = Rm2 * Rm4
                TM_i = TM(:,:,i)
                TM_l = TM(:,:,l)

                ra = 0.0
                rb = 0.0
                do ii=1,3
                   ! ra = w_a.R; rb = - w_b.R (see Stone p.179)
                   ra(ii) =       dot_product(TM_i(:,ii),norm_il)
                   rb(ii) = -1. * dot_product(TM_l(:,ii),norm_il)
                   do ij=1,ii
                      ra2(ii,ij) = ra(ii)*ra(ij)
                      rb2(ii,ij) = rb(ii)*rb(ij)
                   enddo
                enddo
                do ii=1,3
                   do ij=1,ii
                      do ik=1,3
                         do il=1,ik
                            ra2b2(ii,ij,ik,il) = ra2(ii,ij) * rb2(ik,il)
                         enddo
                      enddo
                      ra2b(ii,ij,:) = ra2(ii,ij) * rb(:)
                      rb2a(ii,ij,:) = rb2(ii,ij) * ra(:)
                   enddo
                enddo
                cab = 0.0
                do ii=1,3
                   do ij=1,3
                      cab(ii,ij) = dot_product(TM_i(:,ii),TM_l(:,ij)) 
                   enddo
                   rab(ii,:) = ra(ii) * rb(:)
                enddo


                ! Calculate derivates of cosines: 
                !   The derivatives needed are derivatives of R, ra, rb, cab with
                !   respect to all translation (Tr) and rotation (Rot) degrees of
                !   freedom of both atoms interacting. The derivatives are stored
                !    as d{R,ra,rb,cab}d{Tr,Rot}{Atom-Index:1,2}.
                !
                !   The first indices are the same as for the original notation of
                !   ra, rb and cab, the last index stands for the
                !   translation/rotation axis, e.g. dRax_dY (Atom 1) =
                !   dRadT1(ii,ij) = dRadT1(1,2). 
                !
                do ii=1,3
                   dRadT1(ii,:) = (1/R1)*((-1)*TM_i(:,ii)-ra(ii)*(dRdT1(:)))
                   dRadT2(ii,:) = (1/R1)*(     TM_i(:,ii)-ra(ii)*(dRdT2(:)))
                   ! Minus sign inverted compared to Stone's book because we
                   ! define rb=-wb.R
                   dRbdT1(ii,:) = (1/R1)*(     TM_l(:,ii)-rb(ii)*(dRdT1(:)))
                   dRbdT2(ii,:) = (1/R1)*((-1)*TM_l(:,ii)-rb(ii)*(dRdT2(:)))                
                   call cross3(norm_il,TM_i(:,ii),cr_a)
                   dRadR1(ii,:) = -cr_a
                   call cross3(norm_il,TM_l(:,ii),cr_b)
                   ! Plus sign because we take the derivative of -wb.R
                   dRbdR2(ii,:) = +cr_b

                enddo
                do ii = 1,3
                   do ij = 1,3
                      cr_a = TM_i(:,ii)
                      cr_b = TM_l(:,ij)
                      call cross3(cr_a,cr_b,cr_ab)
                      dCabdR1(ii,ij,:) =        cr_ab
                      dCabdR2(ii,ij,:) = (-1) * cr_ab
                      call cross3(cr_b,cr_a,cr_ba)
                      dCbadR2(ij,ii,:) =        cr_ba
                      dCbadR1(ij,ii,:) = (-1) * cr_ba
                   enddo
                enddo

                if (rank(i) > 0) then
                   ! Calculate Dipole-Charge Interaction energy
                   if (Q00_x(l) .and. R1B .lt. MTPLROFF(1)) then
                      call EDC(Dloc(:,i),CG(l),Rm2,ra,EMTPa,Dloc_x(:,i),switc(1))
                      call FDC(Dloc(:,i),CG(l),Rm2,Rm3,ra,dRdT1,dRdT2,&
                           dRadT1,dRadT2,dRadR1,F1,F2,tau1,Dloc_x(:,i),&
                           switc(1),Dswitc(1))
                   endif
                   ! Calculate Qpole-Charge Interaction energy
                   if (rank(i) > 1 .and. Q00_x(l) .and. R1B .lt. MTPLROFF(2)) then
                      call EQC(Qloc(:,i),CG(l),Rm3,ra2,EMTPa,Qloc_x(:,i),switc(2))
                      call FQC(Qloc(:,i),CG(l),Rm3,Rm4,ra,ra2,dRdT1,&
                           dRdT2,dRadT1,dRadT2,dRadR1,F1,F2,tau1,Qloc_x(:,i),&
                           switc(2),Dswitc(2))
                   endif
                   ! Calculate Dipole-Dipole Interaction energy
                   if (rank(l) > 0 .and. R1B .lt. MTPLROFF(2)) then
                      call EDD(Dloc(:,i),Dloc(:,l),Rm3,rab,cab,EMTPa,&
                           Dloc_x(:,i),Dloc_x(:,l),switc(2))
                      call FDD(Dloc(:,i),Dloc(:,l),Rm3,Rm4,ra,rb,rab,&
                           cab,dRdT1,dRdT2,dRadT1,dRadT2,dRbdT1,dRbdT2,&
                           dRadR1,dRbdR2,dCabdR1,dCabdR2,F1,F2,tau1,tau2,&
                           Dloc_x(:,i),Dloc_x(:,l),switc(2),Dswitc(2))
                      ! Calculate Qpole-Dipole Interaction energy                   
                      if (rank(i) > 1) then
                         call EQD(Qloc(:,i),Dloc(:,l),Rm4,ra,rb,ra2b,&
                              cab,EMTPa,Qloc_x(:,i),Dloc_x(:,l),switc(3))
                         call FQD(Qloc(:,i),Dloc(:,l),Rm4,Rm5,ra,rb,ra2,&
                              rab,ra2b,cab,dRdT1,dRdT2,dRadT1,dRadT2,&
                              dRbdT1,dRbdT2,dRadR1,dRbdR2,dCabdR1,dCabdR2,&
                              F1,F2,tau1,tau2,Qloc_x(:,i),Dloc_x(:,l),&
                              switc(3),Dswitc(3))
                         ! Calculate Qpole-Qpole Interaction energy
                         if (rank(l) > 1) then
                            call EQQ(Qloc(:,i),Qloc(:,l),Rm5,ra2,rb2,&
                                 rab,ra2b2,cab,EMTPa,Qloc_x(:,i),&
                                 Qloc_x(:,l),switc(4))
                            call FQQ(Qloc(:,i),Qloc(:,l),Rm5,Rm6,ra,rb,&
                                 ra2,rb2,rab,ra2b,rb2a,ra2b2,cab,dRdT1,dRdT2,&
                                 dRadT1,dRadT2,dRbdT1,dRbdT2,dRadR1,dRbdR2,&
                                 dCabdR1,dCabdR2,F1,F2,tau1,tau2,&
                                 Qloc_x(:,i),Qloc_x(:,l),switc(4),Dswitc(4))
                         endif
                      endif
                      ! Calculate Dipole-Qpole Interaction energy
                      if (rank(l) > 1) then
                         call EQD(Qloc(:,l),Dloc(:,i),Rm4,rb,ra,rb2a,&
                              transpose(cab),EMTPa,Qloc_x(:,l),Dloc_x(:,i),&
                              switc(3))
                         call FQD(Qloc(:,l),Dloc(:,i),Rm4,Rm5,rb,ra,rb2,&
                              transpose(rab),rb2a,transpose(cab),dRdT1,dRdT2,&
                              dRbdT1,dRbdT2,dRadT1,dRadT2,dRbdR2,dRadR1,dCbadR2,&
                              dCbadR1,F1,F2,tau2,tau1,Qloc_x(:,l),Dloc_x(:,i),&
                              switc(3),Dswitc(3))
                      endif
                   endif
                endif
                if (rank(l) > 0) then
                   ! Calculate Charge-Dipole Interaction energy
                   if (Q00_x(i)) then
                      call EDC(Dloc(:,l),CG(i),Rm2,rb,EMTPa,Dloc_x(:,l),&
                           switc(1))
                      call FDC(Dloc(:,l),CG(i),Rm2,Rm3,rb,dRdT1,dRdT2,&
                           dRbdT1,dRbdT2,dRbdR2,F1,F2,tau2,Dloc_x(:,l),&
                           switc(1),Dswitc(1))
                   endif
                   ! Calculate Charge-Qpole Interaction energy
                   if (rank(l) > 1 .and. Q00_x(i)) then
                      call EQC(Qloc(:,l),CG(i),Rm3,rb2,EMTPa,Qloc_x(:,l),&
                           switc(2))
                      call FQC(Qloc(:,l),CG(i),Rm3,Rm4,rb,rb2,dRdT2,dRdT1,&
                           dRbdT2,dRbdT1,dRbdR2,F2,F1,tau2,Qloc_x(:,l),&
                           switc(2),Dswitc(2))
                   endif
                endif
                ! The energy and the torques simply get scaled by switc.
                EMTP = EMTP + EMTPa
                tau1 = tau1
                tau2 = tau2
                call save_gradients(F1, F2, tau1, tau2, i, l)
             endif
          enddo
       endif
    enddo

    ! Propagate Torques
    do i=1,NATOMX
       call propagate_torque(i,Torque(:,i))
    enddo
#if KEY_MTPL_DEBUG==1
    ! Timing    
    if (prnlev.gt.5 .and. mynod.eq.0) then
       call cpu_time(t1)
       write(outu,'(A,f9.4,A)') &
            " MTPL> Time taken to calculate MTP interaction energies ",&
            t1-t2," seconds."
       call flush(outu)
    endif
#endif

    ! Write overall interaction energy to stdout
    IF(PRNLEV.GT.4) then
       write(outu,'(A,I2,A,f12.6,A)') &
            " MTPL> Total MTP energy on node ",mynod,": ",&
            EMTP*TOKCAL*PREF," kcal/mol"
    endif

    ! MPI except for image calculation
    if (NATOMX .eq. NATIM .and. NATOM .ne. NATIM) then
       ETERM(IMELEC) = ETERM(IMELEC) + EMTP*TOKCAL*PREF
    else
#if KEY_MPI==1
       call mpi_barrier(COMM_CHARMM,ierror)
       ! Gather results from nodes
       call mpi_reduce(EMTP,EMTPtot,1,MPI_REAL8,MPI_SUM,&
            0,COMM_CHARMM,ierror)
#else
       EMTPtot = EMTP
#endif
       ETERM(ELEC)   = ETERM(ELEC)   + EMTPtot*TOKCAL*PREF
    endif
    
    return
  end subroutine mtplx

  ! ##################

  subroutine get_local_xyz(aidx,TM)
    !  '''Returns the unit vectors of the local coordinate system in terms of
    !  the global unit vectors''' 
    !
    !    local_XYZ assignment principle depends on the kind of the reference
    !    sphere and the number of reference atoms: 
    !
    !    internal: if two or three neighbouring atom types are the same,
    !              their indices are the first, second (and third)
    !
    !              if there are two pairs of neighbouring atoms, the pair with
    !              higher priority according is first
    !
    !              the priority of all other atoms is aused to sort the atoms
    !
    !    c3v     : Special kind of internal refaxis system, where 3 (out of 3 or
    !    4) neighbour atoms have the same kind.
    !
    !    terminal: First atom is always the nearest neighbour.
    !              After that the rules are the same as in the internal case.
    !
    !    Priorities: 
    !       I > Br > Cl > S4 > S3 > S2 > S1 > P4 > P3 > P2 > P1 > F > O2 > O1 >
    !       Nam > Nar > N4 > N3 > N2 > N1 > Car > C4 > C3 > C2 > H 
    !
    !    The numbers indicate the number of nearest neighbours. 
    !    Especially for N and O this is more stable than assigning sp2 or sp3
    !    hybridization.
    !
    !    Charges are additionally distributed along conjugated systems 
    !    (always along N and O, not along C and only along S & P if they have
    !    less than 4 neighbours.)
    !    
    !    Charges add to the priorities: ... -- > - > 0 > + > ++ ...
    !
    !        refkind(natoms)                 vector containing definition of LRA
    !                                        system type 
    !        refatms(natoms,4)               list of the reference atoms for
    !                                        each local system 
    !                                        (2 - 4 reference atoms per LRA)
    use chm_kinds
    use stream
    use dimens_fcm
    use consta
    use coord
    use energym
    use deriv
    use psf
    use image
    use number
    use inbnd
    use vector
    implicit none
    integer, intent(in) :: aidx
    real(chm_real), dimension(3,3), intent(out) :: TM

    ! local variables
    real(chm_real), dimension(3) :: Xloc, Yloc, Zloc, xyz0, xyz1, xyz2, xyz3, &
         xyz4, a,b,c,d,e,f,g 
    real(chm_real) :: o,p,q,r
    integer :: i,j,k,l, nrefA

    ! Collect reference atoms
    nrefA = 4
    do i = 1,4
       if (refatms(i,aidx) == 0) nrefA = nrefA - 1
    enddo
    xyz0 = (/ &
         X(aidx), &
         Y(aidx), &
         Z(aidx) /)
    xyz1 = (/ &
         X(refatms(1,aidx)), &
         Y(refatms(1,aidx)), &
         Z(refatms(1,aidx)) /)
    if (nrefA > 1) xyz2 = (/ &
         X(refatms(2,aidx)), &
         Y(refatms(2,aidx)), &
         Z(refatms(2,aidx)) /)
    if (nrefA > 2) xyz3 = (/ &
         X(refatms(3,aidx)), &
         Y(refatms(3,aidx)), &
         Z(refatms(3,aidx)) /) 
    if (nrefA == 4) xyz4 = (/ &
         X(refatms(4,aidx)), &
         Y(refatms(4,aidx)), &
         Z(refatms(4,aidx)) /)
    ! Differentiate between different kinds of reference axis systems
    if (trim(refkind(aidx)) == 'c3v') then
       ! Z for both nrefA 3&4 points outwards or towards RA4.
       ! Y is perpendicular to the plane of RA3&RA0&RA4 and to the plane of
       ! ((RA1-RA0)+(RA2-RA0))&Z.
       if (nrefA == 4) then
          ! Calc Zloc      
          a = xyz4-xyz0
          b = xyz0-xyz3
          c = xyz0-xyz2
          d = xyz0-xyz1
          e = b+c+d
          Zloc = a+3*e
          call normall(Zloc,3)
          ! Calc Yloc  the exact direction of Y does not really matter, because
          ! its coefficients should be 0 anyway.
          call cross3(-c,Zloc,f)
          call cross3(Zloc,f,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          call cross3(Yloc,Zloc,Xloc)

       else
          ! Calc Zloc
          a = xyz0-xyz1
          b = xyz0-xyz2
          c = xyz0-xyz3
          ! This is for tetragonal centers
          d = a+b+c   
          call cross3(c-a,c-b,Zloc)
          call normall(Zloc,3)
          ! Invert Z if it points to the other direction
          if (dot_product(d,d) > 0 .and. dot_product(d,Zloc) < 0) &
               Zloc = (-1)*Zloc      
          ! Calc Yloc
          call cross3(d,Zloc,e)
          call cross3(Zloc,e,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          call cross3(Yloc,Zloc,Xloc)
       endif

    elseif (trim(refkind(aidx)) == 'ter') then
       ! Z points from RC[0] to AC
       ! nrefA = 4: Y points along RC[3]
       ! nrefA = 3: Y points along missing RC[3] or is perpendicular to the
       ! plane spanned by RC[0], RC[1] and AC.
       ! nrefA = 2: Y is perpendicular to the plane spanned by RC[0], RC[1] and
       ! AC.
       ! Calc Zloc
       Zloc = xyz0-xyz1
       call normall(Zloc,3)

       if (nrefA == 4) then
          ! Calc Yloc
          a = xyz4-xyz1
          b = xyz1-xyz2
          c = xyz1-xyz3
          call cross3(a+b+c,Zloc,d)
          call cross3(Zloc,d,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          call box(-b,-c,a,o)
          ! This is the enantiomer check
          if (o > 0) then                            
             call cross3(Yloc,Zloc,Xloc)
          else
             call cross3(Zloc,Yloc,Xloc)
          endif

       elseif (nrefA == 3) then
          ! Calc Yloc
          a = xyz1-xyz0
          b = xyz1-xyz2
          c = xyz1-xyz3
          ! This is for tetragonal centers
          d = a+b+c                                  
          call cross3(-a,b-a,e)
          call cross3(e,Zloc,f)
          call cross3(Zloc,f,Yloc)
          call normall(Yloc,3)
          ! Invert Y if it points to the other direction
          if (dot_product(d,d) > 0 .and. dot_product(d,Zloc) < 0) &
               Yloc = (-1)*Yloc     
          call cross3(Yloc,Zloc,Xloc)
          ! Make X point towards second Reference Atom
          if (dot_product(Xloc,a-b) < 0) Xloc = (-1)*Xloc        

       else
          ! Calc Yloc
          call cross3(Zloc,xyz2-xyz1,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          call cross3(Yloc,Zloc,Xloc)
       endif

    elseif (trim(refkind(aidx)) == 'int') then
       ! Z for nrefA 4: bisects both angles between 1&2 and 3&4
       ! Y for nrefA 4: is perpendicular to the plane of 3&4&C and in the plane
       ! of 1&2&C.
       if (nrefA == 4) then 
          ! Calc Zloc
          a = xyz0-xyz1
          b = xyz0-xyz2
          c = xyz3-xyz0
          d = xyz4-xyz0
          Zloc = a+b+c+d
          call normall(Zloc,3)
          ! Calc Yloc
          call cross3(c-d,Zloc,e)
          call cross3(Zloc,e,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          ! This is the enantiomer check
          call box(xyz1-xyz4,xyz2-xyz4,xyz3-xyz4,o)                   
          if (o > 0) then
             call cross3(Yloc,Zloc,Xloc)
          else
             call cross3(Zloc,Yloc,Xloc) 
          endif

          ! Z for nrefA 3: is perpendicular to the plane of RC[:]
          ! Y for nrefA 3: points towards RC[2]    
       elseif (nrefA == 3) then
          a = xyz0-xyz1
          b = xyz0-xyz2
          c = xyz0-xyz3
          ! This is needed in case the center is tetragonal
          d = a+b+c
          ! Calc Zloc
          e = xyz1-xyz3
          f = xyz2-xyz3
          call cross3(e,f,g)
          Zloc = g
          call normall(Zloc,3)
          if (dot_product(d,d) > 0 .and. dot_product(d,Zloc) < 0) &
               Zloc = (-1)*Zloc
          ! Calc Yloc
          call cross3(a+b-c,Zloc,d)
          call cross3(Zloc,d,e)
          Yloc = e
          call normall(Yloc,3)
          ! Calc Xloc
          call cross3(Yloc,Zloc,Xloc)
          ! Make X point towards second Reference Atom
          if (dot_product(Xloc,b) > 0) Xloc = (-1)*Xloc          
          ! Z for nrefA 2: is perpendicular to the plane of AC, RC[0] and RC[1]
          ! Y for nrefA 2: bisects the angle RC[0],AC,RC[1]   
       else
          a = xyz0-xyz1
          b = xyz0-xyz2
          ! Calc Zloc   
          call cross3(-a,-b,Zloc)
          call normall(Zloc,3)
          ! Calc Yloc
          call cross3(a+b,Zloc,c)
          call cross3(Zloc,c,Yloc)
          call normall(Yloc,3)
          ! Calc Xloc
          call cross3(Yloc,Zloc,Xloc)
       endif

    elseif (trim(refkind(aidx)) == 'lin') then
       if (nrefA == 1) then
          a = xyz1-xyz0
       else
          ! nrefA == 2
          a = xyz1-xyz2
       endif
       b = (/ 1., 1., 1. /)
       call normall(a,3)
       Zloc = a
       call cross3(b,Zloc,Xloc)
       call cross3(Xloc,Zloc,Yloc)
    else
       write(outu,'(A,A)') &
            ' MTPL> Unknown ref axis system: ',trim(refkind(aidx))
       call WRNDIE(-3,'<mtpl.src> get_local_xyz', &
            'Unknown axis system kind.')
    endif
    
    TM(:,1) = Xloc
    TM(:,2) = Yloc
    TM(:,3) = Zloc

    return
  end subroutine get_local_xyz

  ! ##################

  subroutine D_SpH2C(Dsph,Dcart)
    use chm_kinds
    implicit none
    real(chm_real), dimension(3), intent(in) :: Dsph
    real(chm_real), dimension(3), intent(out) :: Dcart

    Dcart(1) = Dsph(2)
    Dcart(2) = Dsph(3)
    Dcart(3) = Dsph(1)

    return
  end subroutine D_SpH2C

  ! ##################

  subroutine D_C2SpH(Dcart,Dsph)
    use chm_kinds
    implicit none
    real(chm_real), dimension(3), intent(in) :: Dcart
    real(chm_real), dimension(3), intent(out) :: Dsph

    Dsph(1) = Dcart(3)
    Dsph(2) = Dcart(1)
    Dsph(3) = Dcart(2)

    return
  end subroutine D_C2SpH

  ! ##################

  subroutine Q_SpH2C(Qsph,Qcart)
    use chm_kinds
    implicit none
    real(chm_real), dimension(5), intent(in) :: Qsph
    real(chm_real), dimension(3,3), intent(out) :: Qcart
    ! local variables
    real(chm_real) :: Qxx,Qyy,Qzz,Qxy,Qxz,Qyz,sqrt_3
    sqrt_3 = 1.7320508075688772

    Qxx = -0.5*Qsph(1)+0.5*sqrt_3*Qsph(4)
    Qyy = -0.5*Qsph(1)-0.5*sqrt_3*Qsph(4)
    Qzz = Qsph(1)
    Qxy = 0.5*sqrt_3*Qsph(5)
    Qxz = 0.5*sqrt_3*Qsph(2)
    Qyz = 0.5*sqrt_3*Qsph(3)

    Qcart = reshape((/Qxx,Qxy,Qxz,Qxy,Qyy,Qyz,Qxz,Qyz,Qzz/),(/3,3/))

    return
  end subroutine Q_SpH2C

  ! ##################

  subroutine Q_C2SpH(Qcart,Qsph)
    use chm_kinds
    implicit none
    real(chm_real), dimension(3,3), intent(in) :: Qcart
    real(chm_real), dimension(5), intent(out) :: Qsph
    ! local variables
    real(chm_real) :: Q20,Q21c,Q21s,Q22c,Q22s,sqrt_3
    sqrt_3 = 1.7320508075688772
    Qsph(1) = Qcart(3,3)
    Qsph(2) = (2./sqrt_3) * Qcart(1,3)
    Qsph(3) = (2./sqrt_3) * Qcart(2,3)
    Qsph(4) = (1./sqrt_3) * (Qcart(1,1)-Qcart(2,2))
    Qsph(5) = (2./sqrt_3) * Qcart(1,2)

    return
  end subroutine Q_C2SpH

  ! ##################

  subroutine ECC(Q00t,Q00u,Rm1,EMTP)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Q00t,Q00u,Rm1
    real(chm_real), intent(inout) :: EMTP
    real(chm_real) :: EQ0Q0

    EQ0Q0 = (Q00t*Q00u) * Rm1
    EMTP  = EMTP + EQ0Q0

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)')" MTPL>  Energy (Q0-Q0): ",EQ0Q0*TOKCAL*PREF
#endif

    return
  end subroutine ECC

  ! ##################

  subroutine EDC(Dloc_t,Q00u,Rm2,ra,EMTP,Dloc_t_x,switc)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Q00u,Rm2,switc
    real(chm_real), dimension(3), intent(in) :: Dloc_t,ra
    real(chm_real), intent(inout) :: EMTP
    logical, dimension(3), intent(in) :: Dloc_t_x
    real(chm_real) :: EQ1Q0
    integer :: K

    EQ1Q0 = 0.0
    do K=1,3
       if (Dloc_t_x(K)) EQ1Q0 = EQ1Q0 + Dloc_t(K) * ra(K)
    enddo
    EQ1Q0 = Rm2 * Q00u * EQ1Q0 * switc

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)') " MTPL>  Energy (Q1-Q0): ",EQ1Q0*TOKCAL*PREF
#endif

    EMTP  = EMTP + EQ1Q0

    return
  end subroutine EDC

  ! ##################

  subroutine EQC(Qloc_t,Q00u,Rm3,ra2,EMTP,Qloc_t_x,switc)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Q00u,Rm3,switc
    real(chm_real), dimension(3,3), intent(in) :: ra2
    real(chm_real), dimension(5), intent(in) :: Qloc_t
    real(chm_real), intent(inout) :: EMTP
    logical, dimension(5), intent(in) :: Qloc_t_x
    real(chm_real) :: sqrt_3, E_QC, EQ2Q0

    sqrt_3 = 1.7320508075688772

    E_QC = 0.0

    if (Qloc_t_x(1)) E_QC = Qloc_t(1) * 0.5 * (3 * ra2(3,3) - 1)
    if (Qloc_t_x(2)) E_QC = E_QC + Qloc_t(2) * sqrt_3 * ra2(3,1)
    if (Qloc_t_x(3)) E_QC = E_QC + Qloc_t(3) * sqrt_3 * ra2(3,2)
    if (Qloc_t_x(4)) E_QC = E_QC + &
         Qloc_t(4) * sqrt_3 * 0.5 * (ra2(1,1) - ra2(2,2))    
    if (Qloc_t_x(5)) E_QC = E_QC + Qloc_t(5) * sqrt_3 * ra2(2,1)
    EQ2Q0 = Rm3 * E_QC * Q00u * switc

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)') " MTPL>  Energy (Q2-Q0): ",EQ2Q0*TOKCAL*PREF
#endif

    EMTP = EMTP + EQ2Q0

    return
  end subroutine EQC

  ! ##################

  subroutine EDD(Dloc_t,Dloc_u,Rm3,rab,cab,EMTP,Dloc_t_x,Dloc_u_x,&
       switc)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Rm3,switc
    real(chm_real), dimension(3), intent(in) :: Dloc_t, Dloc_u
    real(chm_real), dimension(3,3), intent(in) :: rab, cab
    real(chm_real), intent(inout) :: EMTP
    logical, dimension(3), intent(in) :: Dloc_t_x, Dloc_u_x
    real(chm_real) :: E_DD,EQ1Q1
    integer :: K,L

    E_DD = 0.0
    DO K=1,3
       DO L=1,3
          if (Dloc_t_x(K) .and. Dloc_u_x(L)) then
             E_DD = E_DD + Dloc_t(K) * Dloc_u(L) * (3 * rab(K,L) + cab(K,L))
          endif
       ENDDO
    ENDDO
    EQ1Q1 = Rm3 * E_DD * switc

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)') " MTPL>  Energy (Q1-Q1): ",EQ1Q1*TOKCAL*PREF
#endif

    EMTP = EMTP + EQ1Q1

    return
  end subroutine EDD

  ! ##################

  subroutine EQD(QLOC_T,DLOC_U,Rm4,ra,rb,ra2b,cab,EMTP,Qloc_t_x,Dloc_u_x,&
       switc)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Rm4, switc
    real(chm_real), dimension(5), intent(in) :: QLOC_T
    real(chm_real), dimension(3), intent(in) :: DLOC_U, ra, rb
    real(chm_real), dimension(3,3), intent(in) :: cab
    real(chm_real), dimension(3,3,3), intent(in) :: ra2b
    logical, dimension(3), intent(in) :: Dloc_u_x
    logical, dimension(5), intent(in) :: Qloc_t_x
    real(chm_real), intent(inout) :: EMTP
    real(chm_real) :: SQRT_3, E_QD,EQ2Q1
    integer :: K

    sqrt_3 = 1.7320508075688772

    E_QD = 0.0
    DO K=1,3
       if (.not. Dloc_u_x(K)) cycle
       ! Term 20  - 1b
       if (Qloc_t_x(1)) then
          E_QD = E_QD + QLOC_T(1) * DLOC_U(K) * 0.5 * &
               (15 * ra2b(3,3,K) + 6 * ra(3) * cab(3,K) - 3 * rb(K))
       endif
       ! Term 21c - 1b
       if (Qloc_t_x(2)) then
          E_QD = E_QD + QLOC_T(2) * DLOC_U(K) * SQRT_3 * &
               (ra(1) * cab(3,K) + cab(1,K) * ra(3) + 5 * ra2b(3,1,K))
       endif
       ! Term 21s - 1b
       if (Qloc_t_x(3)) then
          E_QD = E_QD + QLOC_T(3) * DLOC_U(K) * SQRT_3 * &
               (ra(2) * cab(3,K) + cab(2,K) * ra(3) + 5 * ra2b(3,2,K))
       endif
       ! Term 22c - 1b
       if (Qloc_t_x(4)) then
          E_QD = E_QD + QLOC_T(4) * DLOC_U(K) * 0.5 * SQRT_3 * &
               (5 * (ra2b(1,1,K) - ra2b(2,2,K)) + 2 * ra(1) * cab(1,K) &
               - 2 * ra(2) * cab(2,K))
       endif
       ! Term 22s - 1b
       if (Qloc_t_x(5)) then
          E_QD = E_QD + QLOC_T(5) * DLOC_U(K) * SQRT_3 * &
               (ra(1) * cab(2,K) + cab(1,K) * ra(2) + 5 * ra2b(2,1,K))
       endif
    ENDDO
    EQ2Q1 = Rm4 * E_QD * switc

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)') " MTPL>  Energy (Q2-Q1): ",EQ2Q1*TOKCAL*PREF
#endif

    EMTP = EMTP + EQ2Q1

    return
  end subroutine EQD

  ! ##################

  subroutine EQQ(QLOC_T,QLOC_U,Rm5,ra2,rb2,rab,ra2b2,cab,EMTP,&
       Qloc_t_x,Qloc_u_x,switc)
    use chm_kinds
    use stream
    use consta
    implicit none
    real(chm_real), intent(in) :: Rm5, switc
    real(chm_real), dimension(5), intent(in) :: QLOC_T, QLOC_U
    real(chm_real), dimension(3,3), intent(in) :: cab,ra2,rb2,rab
    real(chm_real), dimension(3,3,3,3), intent(in) :: ra2b2
    real(chm_real), intent(inout) :: EMTP
    logical, dimension(5), intent(in) :: Qloc_t_x,Qloc_u_x

    real(chm_real) :: SQRT_3, E_QQ,EQ2Q2,SQRT_3_05,SQRT_3_025

    SQRT_3 = 1.7320508075688772
    SQRT_3_05 = 0.8660254037844386
    SQRT_3_025 = 0.4330127018922193

    E_QQ = 0.0
    ! 20  - 20 
    if (Qloc_t_x(1) .and. Qloc_u_x(1)) then
       E_QQ = E_QQ + QLOC_T(1)*QLOC_U(1)*0.75* &
            (35*ra2b2(3,3,3,3) &
            -5*(ra2(3,3) +rb2(3,3)) +20*rab(3,3)*cab(3,3) +2*cab(3,3)**2 +1)
    endif
    ! 20  - 21c
    if (Qloc_t_x(1) .and. Qloc_u_x(2)) then
       E_QQ = E_QQ + QLOC_T(1)*QLOC_U(2)*SQRT_3_05* &
            (35*ra2b2(3,3,3,1) &
            -5*rb2(3,1) +10*(rab(3,1)*cab(3,3) +rab(3,3)*cab(3,1)) &
            +2*cab(3,1)*cab(3,3))
    endif
    ! 21c - 20 
    if (Qloc_t_x(2) .and. Qloc_u_x(1)) then
       E_QQ = E_QQ + QLOC_T(2)*QLOC_U(1)*SQRT_3_05* &
            (35*ra2b2(3,1,3,3) &
            -5*ra2(3,1) +10*(rab(1,3)*cab(3,3) +rab(3,3)*cab(1,3)) &
            +2*cab(1,3)*cab(3,3))
    endif
    ! 20  - 21s
    if (Qloc_t_x(1) .and. Qloc_u_x(3)) then
       E_QQ = E_QQ + QLOC_T(1)*QLOC_U(3)*SQRT_3_05* &
            (35*ra2b2(3,3,3,2) &
            -5*rb2(3,2) +10*(rab(3,2)*cab(3,3) +rab(3,3)*cab(3,2)) &
            +2*cab(3,2)*cab(3,3))
    endif
    ! 21s - 20 
    if (Qloc_t_x(3) .and. Qloc_u_x(1)) then
       E_QQ = E_QQ + QLOC_T(3)*QLOC_U(1)*SQRT_3_05* &
            (35*ra2b2(3,2,3,3) &
            -5*ra2(3,2) +10*(rab(2,3)*cab(3,3) +rab(3,3)*cab(2,3)) &
            +2*cab(2,3)*cab(3,3))
    endif
    ! 20  - 22c
    if (Qloc_t_x(1) .and. Qloc_u_x(4)) then
       E_QQ = E_QQ + QLOC_T(1)*QLOC_U(4)*SQRT_3_025* &
            (35*(ra2b2(3,3,1,1) -ra2b2(3,3,2,2)) &
            +5*(-rb2(1,1) +rb2(2,2)) &
            +20*(rab(3,1)*cab(3,1) -rab(3,2)*cab(3,2)) &
            +2*(cab(3,1)**2 - cab(3,2)**2))
    endif
    ! 22c - 20 
    if (Qloc_t_x(4) .and. Qloc_u_x(1)) then
       E_QQ = E_QQ + QLOC_T(4)*QLOC_U(1)*SQRT_3_025* &
            (35*(ra2b2(1,1,3,3)  - ra2b2(2,2,3,3)) &
            +5*(-ra2(1,1) +ra2(2,2)) &
            +20*(rab(1,3)*cab(1,3) -rab(2,3)*cab(2,3)) &
            +2*(cab(1,3)**2 - cab(2,3)**2))
    endif
    ! 20  - 22s
    if (Qloc_t_x(1) .and. Qloc_u_x(5)) then
       E_QQ = E_QQ + QLOC_T(1)*QLOC_U(5)*SQRT_3_05* &
            (35*ra2b2(3,3,2,1) &
            -5*rb2(2,1) +10*(rab(3,1)*cab(3,2) +rab(3,2)*cab(3,1)) &
            +2*cab(3,1)*cab(3,2))
    endif
    ! 22s  - 20
    if (Qloc_t_x(5) .and. Qloc_u_x(1)) then
       E_QQ = E_QQ + QLOC_T(5)*QLOC_U(1)*SQRT_3_05* &
            (35*ra2b2(2,1,3,3) &
            -5*ra2(2,1) +10*(rab(1,3)*cab(2,3) +rab(2,3)*cab(1,3)) &
            +2*cab(1,3)*cab(2,3))
    endif
    ! 21c - 21c
    if (Qloc_t_x(2) .and. Qloc_u_x(2)) then
       E_QQ = E_QQ + QLOC_T(2)*QLOC_U(2)* &
            (35*ra2b2(3,1,3,1) &
            +5*(rab(1,1)*cab(3,3) +rab(1,3)*cab(3,1) +rab(3,1)*cab(1,3) &
            +rab(3,3)*cab(1,1)) +cab(1,1)*cab(3,3) + cab(1,3)*cab(3,1))
    endif
    ! 21c - 21s
    if (Qloc_t_x(2) .and. Qloc_u_x(3)) then
       E_QQ = E_QQ + QLOC_T(2)*QLOC_U(3)* &
            (35*ra2b2(3,1,3,2) &
            +5*(rab(1,2)*cab(3,3) +rab(1,3)*cab(3,2) +rab(3,2)*cab(1,3) &
            +rab(3,3)*cab(1,2)) +cab(1,2)*cab(3,3) +cab(1,3)*cab(3,2))
    endif
    ! 21s - 21c
    if (Qloc_t_x(3) .and. Qloc_u_x(2)) then
       E_QQ = E_QQ + QLOC_T(3)*QLOC_U(2)* &
            (35*ra2b2(3,2,3,1) &
            +5*(rab(2,1)*cab(3,3) +rab(3,1)*cab(2,3) +rab(2,3)*cab(3,1) &
            +rab(3,3)*cab(2,1)) +cab(2,1)*cab(3,3) +cab(3,1)*cab(2,3))
    endif
    ! 21c - 22c
    if (Qloc_t_x(2) .and. Qloc_u_x(4)) then
       E_QQ = E_QQ + QLOC_T(2)*QLOC_U(4)*0.5* &
            (35*(ra2b2(3,1,1,1) -ra2b2(3,1,2,2)) &
            +10*(rab(1,1)*cab(3,1) -rab(1,2)*cab(3,2) +rab(3,1)*cab(1,1) &
            -rab(3,2)*cab(1,2)) +2*(cab(1,1)*cab(3,1) -cab(1,2)*cab(3,2)))
    endif
    ! 22c - 21c
    if (Qloc_t_x(4) .and. Qloc_u_x(2)) then
       E_QQ = E_QQ + QLOC_T(4)*QLOC_U(2)*0.5* &
            (35*(ra2b2(1,1,3,1) -ra2b2(2,2,3,1)) &
            +10*(rab(1,1)*cab(1,3) -rab(2,1)*cab(2,3) +rab(1,3)*cab(1,1) &
            -rab(2,3)*cab(2,1)) +2*(cab(1,1)*cab(1,3) -cab(2,1)*cab(2,3)))
    endif
    ! 21c - 22s
    if (Qloc_t_x(2) .and. Qloc_u_x(5)) then
       E_QQ = E_QQ + QLOC_T(2)*QLOC_U(5)* &
            (35*ra2b2(3,1,2,1) &
            +5*(rab(1,1)*cab(3,2) +rab(1,2)*cab(3,1) +rab(3,1)*cab(1,2) &
            +rab(3,2)*cab(1,1)) +cab(1,1)*cab(3,2) +cab(1,2)*cab(3,1))
    endif
    ! 22s - 21c
    if (Qloc_t_x(5) .and. Qloc_u_x(2)) then
       E_QQ = E_QQ + QLOC_T(5)*QLOC_U(2)* &
            (35*ra2b2(2,1,3,1) &
            +5*(rab(1,1)*cab(2,3) +rab(2,1)*cab(1,3) +rab(1,3)*cab(2,1) &
            +rab(2,3)*cab(1,1)) +cab(1,1)*cab(2,3) +cab(2,1)*cab(1,3))
    endif
    ! 21s - 21s
    if (Qloc_t_x(3) .and. Qloc_u_x(3)) then
       E_QQ = E_QQ + QLOC_T(3)*QLOC_U(3)* &
            (35*ra2b2(3,2,3,2) &
            +5*(rab(2,2)*cab(3,3) +rab(2,3)*cab(3,2) +rab(3,2)*cab(2,3) &
            +rab(3,3)*cab(2,2)) +cab(2,2)*cab(3,3) +cab(2,3)*cab(3,2))
    endif
    ! 21s - 22c
    if (Qloc_t_x(3) .and. Qloc_u_x(4)) then
       E_QQ = E_QQ + QLOC_T(3)*QLOC_U(4)*0.5* &
            (35*(ra2b2(3,2,1,1) -ra2b2(3,2,2,2)) &
            +10*(rab(2,1)*cab(3,1) -rab(2,2)*cab(3,2) +rab(3,1)*cab(2,1) &
            -rab(3,2)*cab(2,2)) +2*(cab(2,1)*cab(3,1) -cab(2,2)*cab(3,2)))
    endif
    ! 22c - 21s
    if (Qloc_t_x(4) .and. Qloc_u_x(3)) then
       E_QQ = E_QQ + QLOC_T(4)*QLOC_U(3)*0.5* &
            (35*(ra2b2(1,1,3,2) -ra2b2(2,2,3,2)) &
            +10*(rab(1,2)*cab(1,3) -rab(2,2)*cab(2,3) +rab(1,3)*cab(1,2) &
            -rab(2,3)*cab(2,2)) +2*(cab(1,2)*cab(1,3) -cab(2,2)*cab(2,3)))
    endif
    ! 21s - 22s
    if (Qloc_t_x(3) .and. Qloc_u_x(5)) then
       E_QQ = E_QQ + QLOC_T(3)*QLOC_U(5)* &
            (35*ra2b2(3,2,2,1) &
            +5*(rab(2,1)*cab(3,2) +rab(2,2)*cab(3,1) +rab(3,1)*cab(2,2) &
            +rab(3,2)*cab(2,1)) +cab(2,1)*cab(3,2) +cab(2,2)*cab(3,1))
    endif
    ! 22s - 21s
    if (Qloc_t_x(5) .and. Qloc_u_x(3)) then
       E_QQ = E_QQ + QLOC_T(5)*QLOC_U(3)* &
            (35*ra2b2(2,1,3,2) &
            +5*(rab(1,2)*cab(2,3) +rab(2,2)*cab(1,3) +rab(1,3)*cab(2,2) &
            +rab(2,3)*cab(1,2)) &
            +cab(1,2)*cab(2,3) +cab(2,2)*cab(1,3))
    endif
    ! 22c - 22c
    if (Qloc_t_x(4) .and. Qloc_u_x(4)) then
       E_QQ = E_QQ + QLOC_T(4)*QLOC_U(4)*0.25* &
            (35*(ra2b2(1,1,1,1) -ra2b2(1,1,2,2) -ra2b2(2,2,1,1) &
            +ra2b2(2,2,2,2)) +20*(rab(1,1)*cab(1,1) -rab(1,2)*cab(1,2) &
            -rab(2,1)*cab(2,1) +rab(2,2)*cab(2,2)) &
            +2*(cab(1,1)**2 -cab(1,2)**2 -cab(2,1)**2 +cab(2,2)**2))
    endif
    ! 22c - 22s
    if (Qloc_t_x(4) .and. Qloc_u_x(5)) then
       E_QQ = E_QQ + QLOC_T(4)*QLOC_U(5)*0.5* &
            (35*(ra2b2(1,1,2,1) -ra2b2(2,2,2,1)) &
            +10*(rab(1,1)*cab(1,2) +rab(1,2)*cab(1,1) -rab(2,1)*cab(2,2) &
            -rab(2,2)*cab(2,1)) +2*(cab(1,1)*cab(1,2) -cab(2,1)*cab(2,2)))
    endif
    ! 22s - 22c
    if (Qloc_t_x(5) .and. Qloc_u_x(4)) then
       E_QQ = E_QQ + QLOC_T(5)*QLOC_U(4)*0.5* &
            (35*(ra2b2(2,1,1,1) -ra2b2(2,1,2,2)) &
            +10*(rab(1,1)*cab(2,1) +rab(2,1)*cab(1,1) -rab(1,2)*cab(2,2) &
            -rab(2,2)*cab(1,2)) +2*(cab(1,1)*cab(2,1) -cab(1,2)*cab(2,2)))
    endif
    ! 22s - 22s
    if (Qloc_t_x(5) .and. Qloc_u_x(5)) then
       E_QQ = E_QQ + QLOC_T(5)*QLOC_U(5)* &
            (35*ra2b2(2,1,2,1) &
            +5*(rab(1,1)*cab(2,2) +rab(1,2)*cab(2,1) +rab(2,1)*cab(1,2) &
            +rab(2,2)*cab(1,1)) +cab(1,1)*cab(2,2) +cab(1,2)*cab(2,1))
    endif
    EQ2Q2 = Rm5 * E_QQ * switc

#if KEY_MTPL_DEBUG==1
    write(outu,'(A,f9.4)') " MTPL>  Energy (Q2-Q2): ",EQ2Q2*TOKCAL*PREF
#endif

    EMTP = EMTP + EQ2Q2

    return
  end subroutine EQQ

  ! ##################

  subroutine IMTPL(NATM,IS,IQ)
    ! This is an image updating routine.
    ! For more information on this routine see the original MTP code by Plattner and Won
    use chm_kinds
    use dimens_fcm
    use image
    use number
    use consta
    use memory
    use psf
    use stream
    use parallel
    implicit none
    integer, intent(in) :: NATM
    INTEGER IS, IQ, K, I, L, SCNT

    IF (IMOLL .EQ. 0) THEN
       IMOLL = NMOL
    ENDIF

    SCNT = 0
    K = NATM
    DO I=IS, IQ
       K = K + 1
       IF (K .GT. NATOM) THEN
          IMOLL = IMOLL + 1

          ! Update values
          CG(K)        = CG(I)
          RANK(K)      = RANK(I)
          Q00_X(K)     = Q00_X(I)
          DLOC(:,K)    = DLOC(:,I)
          QLOC(:,K)    = QLOC(:,I)
          DLOC_X(:,K)  = DLOC_X(:,I)
          QLOC_X(:,K)  = QLOC_X(:,I)
          DO L = 1, 4
             ! Add NATM because the local axis system should rely on
             ! neighboring image atoms.
             IF (REFATMS(L,I) /= 0) THEN
                REFATMS(L,K) = K + (REFATMS(L,I)-I)
             ELSE                                                                                                   
                REFATMS(L,K) = 0
             ENDIF
          ENDDO
          REFKIND(K)   = REFKIND(I)
          MTP_SITE(K)  = MTP_SITE(I)
       ENDIF
    ENDDO


    return
  end subroutine IMTPL

  ! ##################

  subroutine box(a, b, c, d)
    ! This calculates the box product between a, b and c and returns it on d.
    ! The box product is defined as a * (b x c).
    use chm_kinds
    use vector
    implicit none
    real(chm_real), dimension(3), intent (in) :: a,b,c
    real(chm_real), intent(out) :: d
    ! local variable
    real(chm_real), dimension(3) :: e

    call cross3(b,c,e)
    d = dot_product(a,e)

    return
  end subroutine box

  ! ##################

  subroutine save_gradients(F1, F2, tau1, tau2, i, l)
    use chm_kinds
    use stream
    use dimens_fcm
    use consta
    use coord
    use energym
    use deriv
    use psf
    use image
    use number
    use inbnd  
    implicit none
    real(chm_real), dimension(3), intent(in) :: F1, F2, tau1, tau2
    real(chm_real) :: b2a
    integer, intent(in) :: i,l

    ! Bohr -> Angstrom
    b2a = 0.529177249

    DX(i) = DX(i) + F1(1) * TOKCAL*PREF / b2a
    DY(i) = DY(i) + F1(2) * TOKCAL*PREF / b2a
    DZ(i) = DZ(i) + F1(3) * TOKCAL*PREF / b2a

    DX(l) = DX(l) + F2(1) * TOKCAL*PREF / b2a
    DY(l) = DY(l) + F2(2) * TOKCAL*PREF / b2a
    DZ(l) = DZ(l) + F2(3) * TOKCAL*PREF / b2a

    Torque(1,i) = Torque(1,i) + tau1(1) * TOKCAL*PREF
    Torque(2,i) = Torque(2,i) + tau1(2) * TOKCAL*PREF
    Torque(3,i) = Torque(3,i) + tau1(3) * TOKCAL*PREF

    Torque(1,l) = Torque(1,l) + tau2(1) * TOKCAL*PREF
    Torque(2,l) = Torque(2,l) + tau2(2) * TOKCAL*PREF
    Torque(3,l) = Torque(3,l) + tau2(3) * TOKCAL*PREF

    return
  end subroutine save_gradients

  ! ##################

  subroutine FCC(Q00t,Q00u,Rm2,dRdT1,dRdT2,F1,F2)
    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv

    implicit none
    real(chm_real), intent(in) :: Q00t,Q00u,Rm2
    real(chm_real), dimension(3), intent(in) :: dRdT1,dRdT2
    real(chm_real), dimension(3), intent(inout) :: F1,F2

    real(chm_real) :: pref
    integer :: ii

    pref = Q00t*Q00u * (-Rm2)

    do ii=1,3
       F1(ii) = F1(ii) + pref * dRdT1(ii)
       F2(ii) = F2(ii) + pref * dRdT2(ii)
    enddo

    return
  end subroutine FCC

  ! ##################

  subroutine FDC(Dloc_t,Q00u,Rm2,Rm3,ra,dRdT1,dRdT2,dRadT1,dRadT2,dRadR1,&
       F1,F2,tau1,Dloc_t_x,switc,Dswitc)
    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv

    implicit none
    real(chm_real), intent(in) :: Q00u,Rm2,Rm3,switc,Dswitc
    real(chm_real), dimension(3), intent(in) :: Dloc_t,ra,dRdT1,dRdT2
    real(chm_real), dimension(3,3), intent(in) :: dRadT1,dRadT2,dRadR1
    real(chm_real), dimension(3), intent(inout) :: F1,F2,tau1
    logical, dimension(3), intent(in) :: Dloc_t_x

    ! local variables
    real(chm_real) :: pref, Rm3_2
    integer :: ii,ij

    Rm3_2 = (-2)*Rm3

    do ii = 1,3
       if (.not. Dloc_t_x(ii)) cycle
       pref = Dloc_t(ii) * Q00u
       do ij = 1,3
          F1(ij) = F1(ij) + pref * (ra(ii) * dRdT1(ij) * Rm3_2 * switc + &
               Rm2 * dRadT1(ii,ij) * switc + Rm2 * ra(ii) * Dswitc * dRdT1(ij))
          F2(ij) = F2(ij) + pref * (ra(ii) * dRdT2(ij) * Rm3_2 * switc + &
               Rm2 * dRadT2(ii,ij) * switc + Rm2 * ra(ii) * Dswitc * dRdT2(ij))
          tau1(ij) = tau1(ij) + pref * Rm2 * dRadR1(ii,ij) * switc
       enddo
    enddo

    return
  end subroutine FDC

  ! #################### 

  subroutine FQC(Qloc_t,Q00u,Rm3,Rm4,ra,ra2,dRdT1,dRdT2,dRadT1,dRadT2,dRadR1,&
       F1,F2,tau1,Qloc_t_x,switc,Dswitc)
    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv
    implicit none
    real(chm_real), intent(in) :: Q00u,Rm3,Rm4,switc,Dswitc
    real(chm_real), dimension(3), intent(in) :: ra,dRdT1,dRdT2
    real(chm_real), dimension(5), intent(in) :: Qloc_t 
    real(chm_real), dimension(3,3), intent(in) :: ra2,dRadT1,dRadT2,dRadR1
    real(chm_real), dimension(3), intent(inout) :: F1,F2,tau1
    logical, dimension(5), intent(in) :: Qloc_t_x

    ! local variables
    real(chm_real) :: sqrt_3, pref, Rm4_3
    integer :: ii

    sqrt_3 = 1.7320508075688772
    Rm4_3 = (-3) * Rm4

    if (Qloc_t_x(1)) then
       pref = 0.5 * Qloc_t(1) * Q00u
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((3 * ra2(3,3) - 1) &
               * dRdT1(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (3 * 2 * ra(3) * dRadT1(3,ii)))
          F2(ii) = F2(ii) + pref * ((3 * ra2(3,3) - 1) &
               * dRdT2(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (3 * 2 * ra(3) * dRadT2(3,ii)))
          tau1(ii) = tau1(ii) + pref * Rm3 * 3 * 2 * ra(3) &
               * dRadR1(3,ii) * switc
       enddo
    endif
    if (Qloc_t_x(2)) then
       pref = Qloc_t(2) * Q00u * sqrt_3
       do ii = 1,3
          F1(ii) = F1(ii) + pref &
               * (ra2(3,1)     * dRdT1(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(1) * dRadT1(3,ii) + &
               ra(3) * dRadT1(1,ii)))
          F2(ii) = F2(ii) + pref &
               * (ra2(3,1)     * dRdT2(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(1) * dRadT2(3,ii) + &
               ra(3) * dRadT2(1,ii)))
          tau1(ii) = tau1(ii) + pref &
               * Rm3 * (ra(1) * dRadR1(3,ii) + &
               ra(3) * dRadR1(1,ii)) * switc
       enddo
    endif
    if (Qloc_t_x(3)) then
       pref = Qloc_t(3) * Q00u * sqrt_3
       do ii=1,3
          F1(ii) = F1(ii) + pref &
               * (ra2(3,2)     * dRdT1(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(2) * dRadT1(3,ii) + &
               ra(3) * dRadT1(2,ii)))
          F2(ii) = F2(ii) + pref &
               * (ra2(3,2)     * dRdT2(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(2) * dRadT2(3,ii) + &
               ra(3) * dRadT2(2,ii)))
          tau1(ii) = tau1(ii) + pref &
               * Rm3 * (ra(2) * dRadR1(3,ii) + &
               ra(3) * dRadR1(2,ii)) * switc
       enddo
    endif
    if (Qloc_t_x(4)) then
       pref = Qloc_t(4) * Q00u * 0.5 * sqrt_3
       do ii = 1,3
          F1(ii) = F1(ii) + pref &
               * ((ra2(1,1) - ra2(2,2)) * dRdT1(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (2 * ra(1) * dRadT1(1,ii) - &
               2 * ra(2) * dRadT1(2,ii)))
          F2(ii) = F2(ii) + pref &
               * ((ra2(1,1) - ra2(2,2)) * dRdT2(ii) * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (2 * ra(1) * dRadT2(1,ii) - &
               2 * ra(2) * dRadT2(2,ii)))
          tau1(ii) = tau1(ii) + pref &
               * Rm3   * (2 * ra(1) * dRadR1(1,ii) - &
               2 * ra(2) * dRadR1(2,ii)) * switc
       enddo
    endif
    if (Qloc_t_x(5)) then
       pref = Qloc_t(5) * Q00u * sqrt_3
       do ii = 1,3
          F1(ii) = F1(ii) + pref * (ra2(2,1) * dRdT1(ii) &
               * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(1) * dRadT1(2,ii) + &
               ra(2) * dRadT1(1,ii)))
          F2(ii) = F2(ii) + pref * (ra2(2,1) * dRdT2(ii) &
               * (Rm4_3 * switc + Rm3 * Dswitc) + &
               Rm3 * switc * (ra(1) * dRadT2(2,ii) + &
               ra(2) * dRadT2(1,ii)))
          tau1(ii) = tau1(ii) + pref * Rm3 * (ra(1) * dRadR1(2,ii) + &
               ra(2) * dRadR1(1,ii)) * switc
       enddo
    endif

    return
  end subroutine FQC

  ! ##################

  subroutine FDD(Dloc_t,Dloc_u,Rm3,Rm4,ra,rb,rab,cab,dRdT1,dRdT2,dRadT1,&
       dRadT2,dRbdT1,dRbdT2,dRadR1,dRbdR2,dCabdR1,dCabdR2,F1,F2,&
       tau1,tau2,Dloc_t_x,Dloc_u_x,switc,Dswitc)
    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv
    implicit none
    real(chm_real), intent(in) :: Rm3,Rm4,switc,Dswitc
    real(chm_real), dimension(3), intent(in) :: Dloc_t,Dloc_u,ra,rb,dRdT1,dRdT2
    real(chm_real), dimension(3,3), intent(in) :: rab,cab,dRadT1,dRadT2,&
         dRbdT1,dRbdT2,dRadR1,dRbdR2
    real(chm_real), dimension(3,3,3), intent(in) :: dCabdR1,dCabdR2
    real(chm_real), dimension(3), intent(inout) :: F1,F2,tau1,tau2
    logical, dimension(3), intent(in) :: Dloc_t_x,Dloc_u_x

    ! local variables
    real(chm_real) :: pref, Rm4_3
    integer :: ii,ij,ik

    Rm4_3 = (-3) * Rm4

    DO ii=1,3
       DO ij=1,3
          if (Dloc_t_x(ii) .and. Dloc_u_x(ij)) then
             pref = Dloc_t(ii) * Dloc_u(ij)
             do ik = 1,3
                F1(ik) = F1(ik) + pref * ((3 * rab(ii,ij) &
                     + cab(ii,ij)) * dRdT1(ik) * Rm4_3 * switc + &
                     Rm3 * (3 * ra(ii) * dRbdT1(ij,ik) + &
                     3 * rb(ij) * dRadT1(ii,ik) + &
                     (3 * rab(ii,ij) + cab(ii,ij)) * Dswitc * dRdT1(ik)))
                F2(ik) = F2(ik) + pref * ((3 * rab(ii,ij) &
                     + cab(ii,ij)) * dRdT2(ik) * Rm4_3 * switc + &
                     Rm3 * (3 * ra(ii) * dRbdT2(ij,ik) + &
                     3 * rb(ij) * dRadT2(ii,ik) + &
                     (3 * rab(ii,ij) + cab(ii,ij)) * Dswitc * dRdT2(ik)))
                tau1(ik) = tau1(ik) + pref * (   3*Rm3 * rb(ij) * dRadR1(ii,ik) + &
                     Rm3 * dCabdR1(ii,ij,ik)) * switc 
                tau2(ik) = tau2(ik) + pref * (   3*Rm3 * ra(ii) * dRbdR2(ij,ik) + &
                     Rm3 * dCabdR2(ii,ij,ik)) * switc
             enddo
          endif
       ENDDO
    ENDDO

    return
  end subroutine FDD

  ! ##################

  subroutine FQD(Qloc_t,Dloc_u,Rm4,Rm5,ra,rb,ra2,rab,ra2b,cab,dRdT1,dRdT2,&
       dRadT1,dRadT2,dRbdT1,dRbdT2,dRadR1,dRbdR2,dCabdR1,dCabdR2,F1,F2,&
       tau1,tau2,Qloc_t_x,Dloc_u_x,switc,Dswitc)
    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv
    implicit none
    real(chm_real), intent(in) :: Rm4,Rm5,switc,Dswitc
    real(chm_real), dimension(3), intent(in) :: Dloc_u,ra,rb,dRdT1,dRdT2
    real(chm_real), dimension(5), intent(in) :: Qloc_t
    real(chm_real), dimension(3,3), intent(in) :: ra2,rab,cab,dRadT1,&
         dRadT2,dRbdT1,dRbdT2,dRadR1,dRbdR2
    real(chm_real), dimension(3,3,3), intent(in) :: ra2b,dCabdR1,dCabdR2
    real(chm_real), dimension(3), intent(inout) :: F1,F2,tau1,tau2
    logical, dimension(3), intent(in) :: Dloc_u_x
    logical, dimension(5), intent(in) :: Qloc_t_x

    ! local variables
    real(chm_real) :: sqrt_3, pref, Rm5_4
    integer :: k,ii

    Rm5_4 = (-4) * Rm5
    sqrt_3 = 1.7320508075688772

    DO k=1,3
       if (.not. Dloc_u_x(k)) cycle
       ! Term 20  - 1b
       if (Qloc_t_x(1)) then
          pref = Qloc_t(1) * Dloc_u(k) * 0.5
          do ii = 1,3
             F1(ii) = F1(ii) + pref * ((15 * ra2b(3,3,k) &
                  + 6 * ra(3) * cab(3,k) - 3 * rb(k)) * dRdT1(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (15 * (ra2(3,3) * dRbdT1(k,ii) + 2* rab(3,k) * dRadT1(3,ii)) &
                  + 6 *  cab(3,k) * dRadT1(3,ii) &
                  - 3 *             dRbdT1(k,ii)))
             F2(ii) = F2(ii) + pref * ((15 * ra2b(3,3,k) &
                  + 6 * ra(3) * cab(3,k) - 3 * rb(k)) * dRdT2(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (15 * (ra2(3,3) * dRbdT2(k,ii) + 2* rab(3,k) * dRadT2(3,ii)) &
                  + 6 *  cab(3,k) * dRadT2(3,ii) &
                  - 3 *             dRbdT2(k,ii)))
             tau1(ii) = tau1(ii) + pref * Rm4 * (15 * 2* rab(3,k) * dRadR1(3,ii) + &
                  6 * (cab(3,k) * dRadR1(3,ii) + &
                  ra(3) * dCabdR1(3,k,ii))) * switc
             tau2(ii) = tau2(ii) + pref * Rm4 * (15 *    ra2(3,3) * dRbdR2(k,ii) + &
                  6 * ra(3) * dCabdR2(3,k,ii) - &
                  3 *         dRbdR2(k,ii)) * switc
          enddo
       endif
       ! Term 21c - 1b
       if (Qloc_t_x(2)) then
          pref = Qloc_t(2) * Dloc_u(k) * sqrt_3
          do ii = 1,3
             F1(ii) = F1(ii) + pref * ((ra(1) * cab(3,K) &
                  + cab(1,K) * ra(3) + 5 * ra2b(3,1,K)) * dRdT1(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(3,k) * dRadT1(1,ii) + &
                  cab(1,k) * dRadT1(3,ii) + &
                  5 * (ra2(3,1) * dRbdT1(k,ii) + &
                  rab(3,k) * dRadT1(1,ii) + &
                  rab(1,k) * dRadT1(3,ii))))
             F2(ii) = F2(ii) + pref * ((ra(1) * cab(3,K) &
                  + cab(1,K) * ra(3) + 5 * ra2b(3,1,K)) * dRdT2(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(3,k) * dRadT2(1,ii) + &
                  cab(1,k) * dRadT2(3,ii) + &
                  5 * (ra2(3,1) * dRbdT2(k,ii) + &
                  rab(3,k) * dRadT2(1,ii) + &
                  rab(1,k) * dRadT2(3,ii))))
             tau1(ii) = tau1(ii) + pref * Rm4 &
                  * (cab(3,k) * dRadR1(1,ii) + ra(1) * dCabdR1(3,k,ii) + &
                  cab(1,k) * dRadR1(3,ii) + ra(3) * dCabdR1(1,k,ii) + &
                  5 * (rab(3,k) * dRadR1(1,ii) + &
                  rab(1,k) * dRadR1(3,ii))) * switc
             tau2(ii) = tau2(ii) + pref * Rm4 * &
                  (ra(1) * dCabdR2(3,k,ii) &
                  + ra(3) * dCabdR2(1,k,ii) &
                  + 5 * ra2(3,1) * dRbdR2(k,ii)) * switc
          enddo
       endif
       ! Term 21s - 1b
       if (Qloc_t_x(3)) then
          pref = Qloc_t(3) * Dloc_u(k) * sqrt_3
          do ii = 1,3
             F1(ii) = F1(ii) + pref * ((ra(2) * cab(3,K) &
                  + cab(2,K) * ra(3) + 5 * ra2b(3,2,K)) * dRdT1(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(3,k) * dRadT1(2,ii) + &
                  cab(2,k) * dRadT1(3,ii) + &
                  5 * (ra2(3,2) * dRbdT1(k,ii) + &
                  rab(3,k) * dRadT1(2,ii) + &
                  rab(2,k) * dRadT1(3,ii))))
             F2(ii) = F2(ii) + pref * ((ra(2) * cab(3,K) + cab(2,K) * ra(3) &
                  + 5 * ra2b(3,2,K)) * dRdT2(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(3,k) * dRadT2(2,ii) + &
                  cab(2,k) * dRadT2(3,ii) + &
                  5 * (ra2(3,2) * dRbdT2(k,ii) + &
                  rab(3,k) * dRadT2(2,ii) + &
                  rab(2,k) * dRadT2(3,ii))))
             tau1(ii) = tau1(ii) + pref * Rm4 * &
                  (cab(3,k) * dRadR1(2,ii) + ra(2) * dCabdR1(3,k,ii) + &
                  cab(2,k) * dRadR1(3,ii) + ra(3) * dCabdR1(2,k,ii) + &
                  5 * (rab(3,k) * dRadR1(2,ii) + &
                  rab(2,k) * dRadR1(3,ii))) * switc
             tau2(ii) = tau2(ii) + pref * Rm4 * (ra(2) * dCabdR2(3,k,ii) + &
                  ra(3) * dCabdR2(2,k,ii) + &
                  5 * ra2(3,2) * dRbdR2(k,ii)) * switc
          enddo
       endif
       ! Term 22c - 1b
       if (Qloc_t_x(4)) then
          pref = Qloc_t(4) * Dloc_u(k) * 0.5 * sqrt_3
          do ii = 1,3
             F1(ii) = F1(ii) + pref * ((5 * (ra2b(1,1,k) - ra2b(2,2,k)) &
                  + 2 * ra(1) * cab(1,K) - 2 * ra(2) * cab(2,K))&
                  * dRdT1(ii) * (Rm5_4 * switc + Rm4 * Dswitc) +&
                  Rm4 * switc * (5 * (rb(k) * 2 * (ra(1) * dRadT1(1,ii) - ra(2) * dRadT1(2,ii)) + &
                  (ra2(1,1) - ra2(2,2)) * dRbdT1(k,ii)) + &
                  2 * cab(1,k) * dRadT1(1,ii) - &
                  2 * cab(2,k) * dRadT1(2,ii))) 
             F2(ii) = F2(ii) + pref * ((5 * (ra2b(1,1,k) - ra2b(2,2,k)) &
                  + 2 * ra(1) * cab(1,K) - 2 * ra(2) * cab(2,K))&
                  * dRdT2(ii) * (Rm5_4 * switc + Rm4 * Dswitc) +&
                  Rm4 * switc * (5 * (rb(k) * 2 * (ra(1) * dRadT2(1,ii) - ra(2) * dRadT2(2,ii)) + &
                  (ra2(1,1) - ra2(2,2)) * dRbdT2(k,ii)) + &
                  2 * (cab(1,k) * dRadT2(1,ii)) - &
                  2 * (cab(2,k) * dRadT2(2,ii)))) 
             tau1(ii) = tau1(ii) + pref * Rm4 * &
                  (5 * rb(k) * 2 * (ra(1) * dRadR1(1,ii) - ra(2) * dRadR1(2,ii)) + &
                  2 * (ra(1) * dCabdR1(1,k,ii) + cab(1,k) * dRadR1(1,ii)) - &
                  2 * (ra(2) * dCabdR1(2,k,ii) + cab(2,k) * dRadR1(2,ii))) * switc
             tau2(ii) = tau2(ii) + pref * Rm4 * &
                  (5 * (ra2(1,1) - ra2(2,2)) * dRbdR2(k,ii) + &
                  2 * ra(1) * dCabdR2(1,k,ii) - &
                  2 * ra(2) * dCabdR2(2,k,ii)) * switc 
          enddo
       endif
       ! Term 22s - 1b
       if (Qloc_t_x(5)) then
          pref = Qloc_t(5) * Dloc_u(k) * sqrt_3
          do ii = 1,3
             F1(ii) = F1(ii) + pref * ((ra(1) * cab(2,K) + cab(1,K) * ra(2) &
                  + 5 * ra2b(2,1,K)) * dRdT1(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(2,k) * dRadT1(1,ii) + cab(1,k) * dRadT1(2,ii) + &
                  5 * (ra2(2,1) * dRbdT1(k,ii) + rab(2,k) * dRadT1(1,ii) + &
                  rab(1,k) * dRadT1(2,ii))))
             F2(ii) = F2(ii) + pref * ((ra(1) * cab(2,K) + cab(1,K) * ra(2) &
                  + 5 * ra2b(2,1,K)) * dRdT2(ii) &
                  * (Rm5_4 * switc + Rm4 * Dswitc) + &
                  Rm4 * switc * (cab(2,k) * dRadT2(1,ii) + cab(1,k) * dRadT2(2,ii) + &
                  5 * (ra2(2,1) * dRbdT2(k,ii) + rab(2,k) * dRadT2(1,ii) + &
                  rab(1,k) * dRadT2(2,ii))))
             tau1(ii) = tau1(ii) + pref * Rm4 * (cab(2,k) * dRadR1(1,ii) &
                  + ra(1) * dCabdR1(2,k,ii) + &
                  cab(1,k) * dRadR1(2,ii) + ra(2) * dCabdR1(1,k,ii) + &
                  5 * (rab(2,k) * dRadR1(1,ii) + rab(1,k) * dRadR1(2,ii))) * switc
             tau2(ii) = tau2(ii) + pref * Rm4 * (ra(1) * dCabdR2(2,k,ii) + &
                  ra(2) * dCabdR2(1,k,ii) + &
                  5 * ra2(2,1) * dRbdR2(k,ii))  * switc
          enddo
       endif
    ENDDO

    return
  end subroutine FQD

  ! ##################

  subroutine FQQ(Qloc_t,Qloc_u,Rm5,Rm6,ra,rb,ra2,rb2,rab,ra2b,rb2a,&
       ra2b2,cab,dRdT1,dRdT2,dRadT1,dRadT2,dRbdT1,&
       dRbdT2,dRadR1,dRbdR2,dCabdR1,dCabdR2,F1,F2,tau1,tau2,Qloc_t_x,Qloc_u_x,&
       switc,Dswitc)

    use chm_kinds
    use stream
    use dimens_fcm
    use psf
    use deriv
    implicit none
    real(chm_real), intent(in) :: Rm5,Rm6,switc,Dswitc
    real(chm_real), dimension(3), intent(in) :: ra,rb,dRdT1,dRdT2
    real(chm_real), dimension(5), intent(in) :: Qloc_t,Qloc_u
    real(chm_real), dimension(3,3), intent(in) :: ra2,rb2,rab,cab,dRadT1,&
         dRadT2,dRbdT1,dRbdT2,dRadR1,dRbdR2
    real(chm_real), dimension(3,3,3), intent(in) :: ra2b,rb2a,dCabdR1,dCabdR2
    real(chm_real), dimension(3,3,3,3), intent(in) :: ra2b2
    real(chm_real), dimension(3), intent(inout) :: F1,F2,tau1,tau2
    logical, dimension(5), intent(in) :: Qloc_t_x, Qloc_u_x

    ! local variables
    real(chm_real) :: sqrt_3, sqrt_3_05, sqrt_3_025, pref, Rm6_5
    integer :: k,ii

    SQRT_3 = 1.7320508075688772
    SQRT_3_05 = 0.8660254037844386
    SQRT_3_025 = 0.4330127018922193
    Rm6_5 = (-5) * Rm6

    ! 20  - 20 
    if (Qloc_t_x(1) .and. Qloc_u_x(1)) then
       pref = Qloc_t(1)*Qloc_u(1)*0.75
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,3,3,3) -5*(ra2(3,3) +rb2(3,3)) &
               + 20*rab(3,3)*cab(3,3) +2*cab(3,3)**2 +1)* dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) &
               + Rm5 * switc * (35 * (2 * rb2a(3,3,3) * dRadT1(3,ii) + 2 * ra2b(3,3,3) * dRbdT1(3,ii)) - &
               5 * (2 * ra(3) * dRadT1(3,ii) + 2 * rb(3) * dRbdT1(3,ii)) + &
               20 * (rb(3) * cab(3,3) * dRadT1(3,ii) + ra(3) * cab(3,3) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,3,3,3) -5*(ra2(3,3) +rb2(3,3)) &
               +20*rab(3,3)*cab(3,3) +2*cab(3,3)**2 +1)* dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(3,3,3) * dRadT2(3,ii) + 2 * ra2b(3,3,3) * dRbdT2(3,ii)) - &
               5 * (2 * ra(3) * dRadT2(3,ii) + 2 * rb(3) * dRbdT2(3,ii)) + &
               20 * (rb(3) * cab(3,3) * dRadT2(3,ii) + ra(3) * cab(3,3) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * 2 * rb2a(3,3,3) * dRadR1(3,ii) - &
               5 * 2 * ra(3) * dRadR1(3,ii) + &
               20 * (rb(3) * cab(3,3) * dRadR1(3,ii) + rab(3,3) * dCabdR1(3,3,ii)) +&
               2 * 2 *cab(3,3) * dCabdR1(3,3,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * 2 * ra2b(3,3,3) * dRbdR2(3,ii) - &
               5 * 2 * rb(3) * dRbdR2(3,ii) + &
               20 * (ra(3) * cab(3,3) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(3,3,ii)) +&
               2 * 2 *cab(3,3) * dCabdR2(3,3,ii)) * switc
       enddo
    endif
    ! 20  - 21c
    if (Qloc_t_x(1) .and. Qloc_u_x(2)) then
       pref = QLOC_T(1)*QLOC_U(2)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,3,3,1) -5*rb2(3,1) &
               +10*(rab(3,1)*cab(3,3) +rab(3,3)*cab(3,1)) &
               +2*cab(3,1)*cab(3,3))* dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(3,1,3) * dRadT1(3,ii) + ra2b(3,3,3) * dRbdT1(1,ii) + &
               ra2b(3,3,1) * dRbdT1(3,ii)) - &
               5 * (rb(3) * dRbdT1(1,ii) + rb(1) * dRbdT1(3,ii)) + &
               10 * (rb(1) * cab(3,3) * dRadT1(3,ii) + ra(3) * cab(3,3) * dRbdT1(1,ii) + &
               rb(3) * cab(3,1) * dRadT1(3,ii) + ra(3) * cab(3,1) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,3,3,1) -5*rb2(3,1) &
               +10*(rab(3,1)*cab(3,3) +rab(3,3)*cab(3,1)) &
               +2*cab(3,1)*cab(3,3))* dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(3,1,3) * dRadT2(3,ii) + ra2b(3,3,3) * dRbdT2(1,ii) + &
               ra2b(3,3,1) * dRbdT2(3,ii)) - &
               5 * (rb(3) * dRbdT2(1,ii) + rb(1) * dRbdT2(3,ii)) + &
               10 * (rb(1) * cab(3,3) * dRadT2(3,ii) + ra(3) * cab(3,3) * dRbdT2(1,ii) + &
               rb(3) * cab(3,1) * dRadT2(3,ii) + ra(3) * cab(3,1) * dRbdT2(3,ii)))) 
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * 2 * rb2a(3,1,3) * dRadR1(3,ii) + &
               10 * (rb(1) * cab(3,3) * dRadR1(3,ii) + rab(3,1) * dCabdR1(3,3,ii) + &
               rb(3) * cab(3,1) * dRadR1(3,ii) + rab(3,3) * dCabdR1(3,1,ii)) + &
               2 * (cab(3,1) * dCabdR1(3,3,ii) + cab(3,3) * dCabdR1(3,1,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(3,3,3) * dRbdR2(1,ii) &
               + ra2b(3,3,1) * dRbdR2(3,ii)) - &
               5 * (rb(3) * dRbdR2(1,ii) + rb(1) * dRbdR2(3,ii)) + &
               10 * (ra(3) * cab(3,3) * dRbdR2(1,ii) + rab(3,1) * dCabdR2(3,3,ii) + &
               ra(3) * cab(3,1) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(3,1,ii)) + &
               2 * (cab(3,1) * dCabdR2(3,3,ii) + cab(3,3) * dCabdR2(3,1,ii))) * switc
       enddo
    endif
    ! 21c - 20 
    if (Qloc_t_x(2) .and. Qloc_u_x(1)) then
       pref = QLOC_T(2)*QLOC_U(1)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,1,3,3) -5*ra2(3,1) &
               +10*(rab(1,3)*cab(3,3) +rab(3,3)*cab(1,3)) &
               +2*cab(1,3)*cab(3,3))* dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * ra2b(3,1,3) * dRbdT1(3,ii) + rb2a(3,3,3) * dRadT1(1,ii) + &
               rb2a(3,3,1) * dRadT1(3,ii)) - &
               5 * (ra(3) * dRadT1(1,ii) + ra(1) * dRadT1(3,ii)) + &
               10 * (ra(1) * cab(3,3) * dRbdT1(3,ii) + rb(3) * cab(3,3) * dRadT1(1,ii) + &
               ra(3) * cab(1,3) * dRbdT1(3,ii) + rb(3) * cab(1,3) * dRadT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,1,3,3) -5*ra2(3,1) +10*(rab(1,3)*cab(3,3) &
               +rab(3,3)*cab(1,3)) +2*cab(1,3)*cab(3,3))* dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * ra2b(3,1,3) * dRbdT2(3,ii) + rb2a(3,3,3) * dRadT2(1,ii) + &
               rb2a(3,3,1) * dRadT2(3,ii)) - &
               5 * (ra(3) * dRadT2(1,ii) + ra(1) * dRadT2(3,ii)) + &
               10 * (ra(1) * cab(3,3) * dRbdT2(3,ii) + rb(3) * cab(3,3) * dRadT2(1,ii) + &
               ra(3) * cab(1,3) * dRbdT2(3,ii) + rb(3) * cab(1,3) * dRadT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(3,3,3) * dRadR1(1,ii) &
               + rb2a(3,3,1) * dRadR1(3,ii)) - &
               5 * (ra(3) * dRadR1(1,ii) + ra(1) * dRadR1(3,ii)) + &
               10 * (rb(3) * cab(3,3) * dRadR1(1,ii) + rab(1,3) * dCabdR1(3,3,ii) + &
               rb(3) * cab(1,3) * dRadR1(3,ii) + rab(3,3) * dCabdR1(1,3,ii)) + &
               2 * (cab(1,3) * dCabdR1(3,3,ii) + cab(3,3) * dCabdR1(1,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (2 * ra2b(3,1,3) * dRbdR2(3,ii)) + &
               10 * (ra(1) * cab(3,3) * dRbdR2(3,ii) + rab(1,3) * dCabdR2(3,3,ii) + &
               ra(3) * cab(1,3) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(1,3,ii)) + &
               2 * (cab(1,3) * dCabdR2(3,3,ii) + cab(3,3) * dCabdR2(1,3,ii))) * switc
       enddo
    endif
    ! 20  - 21s
    if (Qloc_t_x(1) .and. Qloc_u_x(3)) then
       pref = QLOC_T(1)*QLOC_U(3)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,3,3,2) -5*rb2(3,2) +10*(rab(3,2)*cab(3,3)&
               +rab(3,3)*cab(3,2)) +2*cab(3,2)*cab(3,3))* dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(3,2,3) * dRadT1(3,ii) + ra2b(3,3,3) * dRbdT1(2,ii) &
               + ra2b(3,3,2) * dRbdT1(3,ii)) &
               -5 * (rb(3) * dRbdT1(2,ii) + rb(2) * dRbdT1(3,ii)) &
               + 10 * (rb(2) * cab(3,3) * dRadT1(3,ii) + ra(3) * cab(3,3) * dRbdT1(2,ii) &
               + rb(3) * cab(3,2) * dRadT1(3,ii) + ra(3) * cab(3,2) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,3,3,2) -5*rb2(3,2) +10*(rab(3,2)*cab(3,3)&
               +rab(3,3)*cab(3,2)) +2*cab(3,2)*cab(3,3))* dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(3,2,3) * dRadT2(3,ii) + ra2b(3,3,3) * dRbdT2(2,ii) &
               + ra2b(3,3,2) * dRbdT2(3,ii)) &
               -5 * (rb(3) * dRbdT2(2,ii) + rb(2) * dRbdT2(3,ii)) &
               + 10 * (rb(2) * cab(3,3) * dRadT2(3,ii) + ra(3) * cab(3,3) * dRbdT2(2,ii) &
               + rb(3) * cab(3,2) * dRadT2(3,ii) + ra(3) * cab(3,2) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (2 * rb2a(3,2,3) * dRadR1(3,ii)) &
               + 10 * (rb(2) * cab(3,3) * dRadR1(3,ii) + rab(3,2) * dCabdR1(3,3,ii) &
               + rb(3) * cab(3,2) * dRadR1(3,ii) + rab(3,3) * dCabdR1(3,2,ii)) &
               + 2 * (cab(3,2) * dCabdR1(3,3,ii) + cab(3,3) * dCabdR1(3,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(3,3,3) * dRbdR2(2,ii) &
               + ra2b(3,3,2) * dRbdR2(3,ii)) &
               -5 * (rb(3) * dRbdR2(2,ii) + rb(2) * dRbdR2(3,ii)) &
               + 10 * (ra(3) * cab(3,3) * dRbdR2(2,ii) + rab(3,2) * dCabdR2(3,3,ii) &
               + ra(3) * cab(3,2) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(3,2,ii)) &
               +  2 * (cab(3,2) * dCabdR2(3,3,ii) + cab(3,3) * dCabdR2(3,2,ii))) * switc
       enddo
    endif
    ! 21s - 20 
    if (Qloc_t_x(3) .and. Qloc_u_x(1)) then
       pref = QLOC_T(3)*QLOC_U(1)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,2,3,3) -5*ra2(3,2) &
               +10*(rab(2,3)*cab(3,3) +rab(3,3)*cab(2,3)) &
               +2*cab(2,3)*cab(3,3))* dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) &
               +Rm5 * switc * (35 * (2 * ra2b(3,2,3) * dRbdT1(3,ii) + rb2a(3,3,3) * dRadT1(2,ii) &
               + rb2a(3,3,2) * dRadT1(3,ii)) &
               -5 * (ra(3) * dRadT1(2,ii) + ra(2) * dRadT1(3,ii)) &
               + 10 * (ra(2) * cab(3,3) * dRbdT1(3,ii) + rb(3) * cab(3,3) * dRadT1(2,ii) &
               + ra(3) * cab(2,3) * dRbdT1(3,ii) + rb(3) * cab(2,3) * dRadT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,2,3,3) -5*ra2(3,2) &
               +10*(rab(2,3)*cab(3,3) +rab(3,3)*cab(2,3)) &
               +2*cab(2,3)*cab(3,3))* dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * ra2b(3,2,3) * dRbdT2(3,ii) + rb2a(3,3,3) * dRadT2(2,ii) &
               + rb2a(3,3,2) * dRadT2(3,ii)) &
               -5 * (ra(3) * dRadT2(2,ii) + ra(2) * dRadT2(3,ii)) &
               + 10 * (ra(2) * cab(3,3) * dRbdT2(3,ii) + rb(3) * cab(3,3) * dRadT2(2,ii) &
               + ra(3) * cab(2,3) * dRbdT2(3,ii) + rb(3) * cab(2,3) * dRadT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(3,3,3) * dRadR1(2,ii) &
               + rb2a(3,3,2) * dRadR1(3,ii)) &
               -5 * (ra(3) * dRadR1(2,ii) + ra(2) * dRadR1(3,ii)) &
               + 10 * (rb(3) * cab(3,3) * dRadR1(2,ii) + rab(2,3) * dCabdR1(3,3,ii) &
               + rb(3) * cab(2,3) * dRadR1(3,ii) + rab(3,3) * dCabdR1(2,3,ii)) &
               + 2 * (cab(2,3) * dCabdR1(3,3,ii) + cab(3,3) * dCabdR1(2,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * 2 * ra2b(3,2,3) * dRbdR2(3,ii) &
               + 10 * (ra(2) * cab(3,3) * dRbdR2(3,ii) + rab(2,3) * dCabdR2(3,3,ii) &
               + ra(3) * cab(2,3) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(2,3,ii)) &
               + 2 * (cab(2,3) * dCabdR2(3,3,ii) + cab(3,3) * dCabdR2(2,3,ii))) * switc
       enddo
    endif
    ! 20  - 22c
    if (Qloc_t_x(1) .and. Qloc_u_x(4)) then
       pref = QLOC_T(1)*QLOC_U(4)*SQRT_3_025
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(3,3,1,1) -ra2b2(3,3,2,2)) &
               +5*(-rb2(1,1) +rb2(2,2)) +20*(rab(3,1)*cab(3,1) &
               -rab(3,2)*cab(3,2)) +2*(cab(3,1)**2 - cab(3,2)**2)) * dRdT1(ii) &
               * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(1,1,3) * dRadT1(3,ii) + 2 * ra2b(3,3,1) * dRbdT1(1,ii) &
               -2 * rb2a(2,2,3) * dRadT1(3,ii) - 2 * ra2b(3,3,2) * dRbdT1(2,ii)) &
               +5 * (-2 * rb(1) * dRbdT1(1,ii) + 2 * rb(2) * dRbdT1(2,ii)) &
               +20 * (rb(1) * cab(3,1) * dradT1(3,ii) + ra(3) * cab(3,1) * dRbdT1(1,ii) &
               -rb(2) * cab(3,2) * dradT1(3,ii) - ra(3) * cab(3,2) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(3,3,1,1) -ra2b2(3,3,2,2)) &
               +5*(-rb2(1,1) +rb2(2,2)) +20*(rab(3,1)*cab(3,1) &
               -rab(3,2)*cab(3,2)) +2*(cab(3,1)**2 - cab(3,2)**2)) * dRdT2(ii) &
               * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * rb2a(1,1,3) * dRadT2(3,ii) + 2 * ra2b(3,3,1) * dRbdT2(1,ii) &
               -2 * rb2a(2,2,3) * dRadT2(3,ii) - 2 * ra2b(3,3,2) * dRbdT2(2,ii)) &
               +5 * (-2 * rb(1) * dRbdT2(1,ii) + 2 * rb(2) * dRbdT2(2,ii)) &
               +20 * (rb(1) * cab(3,1) * dradT2(3,ii) + ra(3) * cab(3,1) * dRbdT2(1,ii) &
               -rb(2) * cab(3,2) * dradT2(3,ii) - ra(3) * cab(3,2) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (2 * rb2a(1,1,3)* dRadR1(3,ii) &
               -2 * rb2a(2,2,3) * dRadR1(3,ii)) &
               +20 * (rb(1) * cab(3,1) * dradR1(3,ii) + rab(3,1) * dCabdR1(3,1,ii) &
               -rb(2) * cab(3,2) * dradR1(3,ii) - rab(3,2) * dCabdR1(3,2,ii)) &
               +2 * (2 * cab(3,1) * dCabdR1(3,1,ii) - 2 * cab(3,2) * dcabdR1(3,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (2 * ra2b(3,3,1)* dRbdR2(1,ii) &
               - 2 * ra2b(3,3,2) * dRbdR2(2,ii)) &
               +5 * (-2 * rb(1) * dRbdR2(1,ii) + 2 * rb(2) * dRbdR2(2,ii)) &
               +20 * (ra(3) * cab(3,1) * dRbdR2(1,ii) + rab(3,1) * dCabdR2(3,1,ii) &
               -ra(3) * cab(3,2) * dRbdR2(2,ii) - rab(3,2) * dCabdR2(3,2,ii)) &
               +2 * (2 * cab(3,1) * dCabdR2(3,1,ii) - 2 * cab(3,2) * dcabdR2(3,2,ii))) * switc
       enddo
    endif
    ! 22c - 20 
    if (Qloc_t_x(4) .and. Qloc_u_x(1)) then
       pref = QLOC_T(4)*QLOC_U(1)*SQRT_3_025
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(1,1,3,3) -ra2b2(2,2,3,3)) &
               +5*(-ra2(1,1) +ra2(2,2)) +20*(rab(1,3)*cab(1,3) &
               -rab(2,3)*cab(2,3)) +2*(cab(1,3)**2 - cab(2,3)**2)) * dRdT1(ii) &
               * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * ra2b(1,1,3) * dRbdT1(3,ii) + 2 * rb2a(3,3,1) * dRadT1(1,ii) &
               -2 * ra2b(2,2,3) * dRbdT1(3,ii) - 2 * rb2a(3,3,2) * dRadT1(2,ii)) &
               +5 * (-2 * ra(1) * dRadT1(1,ii) + 2 * ra(2) * dRadT1(2,ii)) &
               +20 * (ra(1) * cab(1,3) * drbdT1(3,ii) + rb(3) * cab(1,3) * dRadT1(1,ii) &
               -ra(2) * cab(2,3) * drbdT1(3,ii) - rb(3) * cab(2,3) * dRadT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(1,1,3,3) -ra2b2(2,2,3,3)) &
               +5*(-ra2(1,1) +ra2(2,2)) +20*(rab(1,3)*cab(1,3) &
               -rab(2,3)*cab(2,3)) +2*(cab(1,3)**2 - cab(2,3)**2)) * dRdT2(ii) &
               * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (2 * ra2b(1,1,3) * dRbdT2(3,ii) + 2 * rb2a(3,3,1) * dRadT2(1,ii) &
               -2 * ra2b(2,2,3) * dRbdT2(3,ii) - 2 * rb2a(3,3,2) * dRadT2(2,ii)) &
               +5 * (-2 * ra(1) * dRadT2(1,ii) + 2 * ra(2) * dRadT2(2,ii)) &
               +20 * (ra(1) * cab(1,3) * drbdT2(3,ii) + rb(3) * cab(1,3) * dRadT2(1,ii) &
               -ra(2) * cab(2,3) * drbdT2(3,ii) - rb(3) * cab(2,3) * dRadT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (2 * rb2a(3,3,1)* dRadR1(1,ii) &
               - 2 * rb2a(3,3,2) * dRadR1(2,ii)) &
               +5 * (-2 * ra(1) * dRadR1(1,ii) + 2 * ra(2) * dRadR1(2,ii)) &
               +20 * (rb(3) * cab(1,3) * dRadR1(1,ii) + rab(1,3) * dCabdR1(1,3,ii) &
               -rb(3) * cab(2,3) * dRadR1(2,ii) - rab(2,3) * dCabdR1(2,3,ii)) &
               +2 * (2 * cab(1,3) * dCabdR1(1,3,ii) - 2 * cab(2,3) * dcabdR1(2,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (2 * ra2b(1,1,3)* dRbdR2(3,ii) &
               - 2 * ra2b(2,2,3) * dRbdR2(3,ii)) &
               +20 * (ra(1) * cab(1,3) * drbdR2(3,ii) + rab(1,3) * dCabdR2(1,3,ii) &
               -ra(2) * cab(2,3) * drbdR2(3,ii) - rab(2,3) * dCabdR2(2,3,ii)) &
               +2 * (2 * cab(1,3) * dCabdR2(1,3,ii) - 2 * cab(2,3) * dcabdR2(2,3,ii))) * switc
       enddo
    endif
    ! 20  - 22s
    if (Qloc_t_x(1) .and. Qloc_u_x(5)) then
       pref = QLOC_T(1)*QLOC_U(5)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,3,2,1) -5*rb2(2,1) &
               +10*(rab(3,1)*cab(3,2) +rab(3,2)*cab(3,1)) &
               +2*cab(3,1)*cab(3,2)) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(2 * rb2a(2,1,3) * dRadT1(3,ii) + ra2b(3,3,2) * dRbdT1(1,ii) &
               + ra2b(3,3,1) * dRbdT1(2,ii)) &
               -5 *(rb(2) * dRbdT1(1,ii) + rb(1) * dRbdT1(2,ii)) &
               +10 *(rb(1) * cab(3,2) * dradT1(3,ii) + ra(3) * cab(3,2) * dRbdT1(1,ii) &
               +rb(2) * cab(3,1) * dradT1(3,ii) + ra(3) * cab(3,1) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,3,2,1) -5*rb2(2,1) &
               +10*(rab(3,1)*cab(3,2) +rab(3,2)*cab(3,1)) &
               +2*cab(3,1)*cab(3,2)) * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(2 * rb2a(2,1,3) * dRadT2(3,ii) + ra2b(3,3,2) * dRbdT2(1,ii) &
               + ra2b(3,3,1) * dRbdT2(2,ii)) &
               -5 *(rb(2) * dRbdT2(1,ii) + rb(1) * dRbdT2(2,ii)) &
               +10 *(rb(1) * cab(3,2) * dradT2(3,ii) + ra(3) * cab(3,2) * dRbdT2(1,ii) &
               +rb(2) * cab(3,1) * dradT2(3,ii) + ra(3) * cab(3,1) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *2 * rb2a(2,1,3) * dRadR1(3,ii) &
               +10 *(rb(1) * cab(3,2) * dradR1(3,ii) + rab(3,1) * dCabdR1(3,2,ii) &
               +rb(2) * cab(3,1) * dradR1(3,ii) + rab(3,2) * dCabdR1(3,1,ii)) &
               + 2 *(cab(3,2) * dCabdR1(3,1,ii) + cab(3,1) * dCabdR1(3,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(3,3,2) * dRbdR2(1,ii) &
               + ra2b(3,3,1) * dRbdR2(2,ii)) &
               -5 *(rb(2) * dRbdR2(1,ii) + rb(1) * dRbdR2(2,ii)) &
               +10 *(ra(3) * cab(3,2) * dRbdR2(1,ii) + rab(3,1) * dCabdR2(3,2,ii) &
               + ra(3) * cab(3,1) * dRbdR2(2,ii) + rab(3,2) * dCabdR2(3,1,ii)) &
               + 2 *(cab(3,2) * dCabdR2(3,1,ii) + cab(3,1) * dCabdR2(3,2,ii))) * switc
       enddo
    endif
    ! 22s  - 20
    if (Qloc_t_x(5) .and. Qloc_u_x(1)) then
       pref = QLOC_T(5)*QLOC_U(1)*SQRT_3_05
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(2,1,3,3) -5*ra2(2,1) &
               +10*(rab(1,3)*cab(2,3) +rab(2,3)*cab(1,3)) &
               +2*cab(1,3)*cab(2,3)) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(2 * ra2b(2,1,3) * dRbdT1(3,ii) + rb2a(3,3,2) * dRadT1(1,ii) &
               + rb2a(3,3,1) * dRadT1(2,ii)) &
               -5 *(ra(2) * dRadT1(1,ii) + ra(1) * dRadT1(2,ii)) &
               +10 *(ra(1) * cab(2,3) * drbdT1(3,ii) + rb(3) * cab(2,3) * dRadT1(1,ii) &
               +ra(2) * cab(1,3) * drbdT1(3,ii) + rb(3) * cab(1,3) * dRadT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(2,1,3,3) -5*ra2(2,1) &
               +10*(rab(1,3)*cab(2,3) +rab(2,3)*cab(1,3)) &
               +2*cab(1,3)*cab(2,3)) * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(2 * ra2b(2,1,3) * dRbdT2(3,ii) + rb2a(3,3,2) * dRadT2(1,ii) &
               + rb2a(3,3,1) * dRadT2(2,ii)) &
               -5 *(ra(2) * dRadT2(1,ii) + ra(1) * dRadT2(2,ii)) &
               +10 *(ra(1) * cab(2,3) * drbdT2(3,ii) + rb(3) * cab(2,3) * dRadT2(1,ii) &
               +ra(2) * cab(1,3) * drbdT2(3,ii) + rb(3) * cab(1,3) * dRadT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(3,3,2) * dRadR1(1,ii) &
               + rb2a(3,3,1) * dRadR1(2,ii)) &
               -5 *(ra(2) * dRadR1(1,ii) + ra(1) * dRadR1(2,ii)) &
               +10 *(rb(3) * cab(2,3) * dRadR1(1,ii) + rab(1,3) * dCabdR1(2,3,ii) &
               +rb(3) * cab(1,3) * dRadR1(2,ii) + rab(2,3) * dCabdR1(1,3,ii)) &
               + 2 *(cab(2,3) * dCabdR1(1,3,ii) + cab(1,3) * dCabdR1(2,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *2 * ra2b(2,1,3) * dRbdR2(3,ii) &
               +10 *(ra(1) * cab(2,3) * drbdR2(3,ii) + rab(1,3) * dCabdR2(2,3,ii) &
               +ra(2) * cab(1,3) * drbdR2(3,ii) + rab(2,3) * dCabdR2(1,3,ii)) &
               + 2 *(cab(2,3) * dCabdR2(1,3,ii) + cab(1,3) * dCabdR2(2,3,ii))) * switc
       enddo
    endif
    ! 21c - 21c
    if (Qloc_t_x(2) .and. Qloc_u_x(2)) then
       pref = QLOC_T(2)*QLOC_U(2)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,1,3,1) +5*(rab(1,1)*cab(3,3) &
               +rab(1,3)*cab(3,1) +rab(3,1)*cab(1,3) &
               +rab(3,3)*cab(1,1)) +cab(1,1)*cab(3,3) + cab(1,3)*cab(3,1)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(3,1,3) * dRadT1(1,ii) + rb2a(3,1,1) * dRadT1(3,ii) &
               +ra2b(3,1,3) * dRbdT1(1,ii) + ra2b(3,1,1) * dRbdT1(3,ii)) &
               +5 * (rb(1) * cab(3,3) * dRadT1(1,ii) + ra(1) * cab(3,3) * dRbdT1(1,ii) &
               +rb(3) * cab(3,1) * dRadT1(1,ii) + ra(1) * cab(3,1) * dRbdT1(3,ii) &
               +rb(1) * cab(1,3) * dRadT1(3,ii) + ra(3) * cab(1,3) * dRbdT1(1,ii) &
               +rb(3) * cab(1,1) * dRadT1(3,ii) + ra(3) * cab(1,1) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,1,3,1) +5*(rab(1,1)*cab(3,3) &
               +rab(1,3)*cab(3,1) +rab(3,1)*cab(1,3) &
               +rab(3,3)*cab(1,1)) +cab(1,1)*cab(3,3) + cab(1,3)*cab(3,1)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(3,1,3) * dRadT2(1,ii) + rb2a(3,1,1) * dRadT2(3,ii) &
               +ra2b(3,1,3) * dRbdT2(1,ii) + ra2b(3,1,1) * dRbdT2(3,ii)) &
               +5 * (rb(1) * cab(3,3) * dRadT2(1,ii) + ra(1) * cab(3,3) * dRbdT2(1,ii) &
               +rb(3) * cab(3,1) * dRadT2(1,ii) + ra(1) * cab(3,1) * dRbdT2(3,ii) &
               +rb(1) * cab(1,3) * dRadT2(3,ii) + ra(3) * cab(1,3) * dRbdT2(1,ii) &
               +rb(3) * cab(1,1) * dRadT2(3,ii) + ra(3) * cab(1,1) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(3,1,3) * dRadR1(1,ii) &
               + rb2a(3,1,1) * dRadR1(3,ii)) &
               +5 * (rb(1) * cab(3,3) * dRadR1(1,ii) + rab(1,1) * dCabdR1(3,3,ii) &
               +rb(3) * cab(3,1) * dRadR1(1,ii) + rab(1,3) * dCabdR1(3,1,ii) &
               +rb(1) * cab(1,3) * dRadR1(3,ii) + rab(3,1) * dCabdR1(1,3,ii) &
               +rb(3) * cab(1,1) * dRadR1(3,ii) + rab(3,3) * dCabdR1(1,1,ii)) &
               + cab(3,3) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(3,3,ii) &
               + cab(3,1) * dCabdR1(1,3,ii) + cab(1,3) * dCabdR1(3,1,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(3,1,3) * dRbdR2(1,ii) &
               + ra2b(3,1,1) * dRbdR2(3,ii)) &
               +5 * (ra(1) * cab(3,3) * dRbdR2(1,ii) + rab(1,1) * dCabdR2(3,3,ii) &
               +ra(1) * cab(3,1) * dRbdR2(3,ii) + rab(1,3) * dCabdR2(3,1,ii) &
               +ra(3) * cab(1,3) * dRbdR2(1,ii) + rab(3,1) * dCabdR2(1,3,ii) &
               +ra(3) * cab(1,1) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(1,1,ii)) &
               + cab(3,3) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(3,3,ii) &
               + cab(3,1) * dCabdR2(1,3,ii) + cab(1,3) * dCabdR2(3,1,ii)) * switc
       enddo
    endif
    ! 21c - 21s
    if (Qloc_t_x(2) .and. Qloc_u_x(3)) then
       pref = QLOC_T(2)*QLOC_U(3)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,1,3,2) +5*(rab(1,2)*cab(3,3) &
               +rab(1,3)*cab(3,2) +rab(3,2)*cab(1,3) &
               +rab(3,3)*cab(1,2)) +cab(1,2)*cab(3,3) +cab(1,3)*cab(3,2)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(3,2,3) * dradT1(1,ii) + rb2a(3,2,1) * dradT1(3,ii) &
               +ra2b(3,1,2) * dRbdT1(3,ii) + ra2b(3,1,3) * dRbdT1(2,ii)) &
               + 5 *(rb(2) * cab(3,3) * dRadT1(1,ii) + ra(1) * cab(3,3) * dRbdT1(2,ii) &
               +rb(3) * cab(3,2) * dRadT1(1,ii) + ra(1) * cab(3,2) * dRbdT1(3,ii) &
               +rb(2) * cab(1,3) * dRadT1(3,ii) + ra(3) * cab(1,3) * dRbdT1(2,ii) &
               +rb(3) * cab(1,2) * dRadT1(3,ii) + ra(3) * cab(1,2) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,1,3,2) +5*(rab(1,2)*cab(3,3) &
               +rab(1,3)*cab(3,2) +rab(3,2)*cab(1,3) &
               +rab(3,3)*cab(1,2)) +cab(1,2)*cab(3,3) +cab(1,3)*cab(3,2)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(3,2,3) * dradT2(1,ii) + rb2a(3,2,1) * dradT2(3,ii) &
               +ra2b(3,1,2) * dRbdT2(3,ii) + ra2b(3,1,3) * dRbdT2(2,ii)) &
               + 5 *(rb(2) * cab(3,3) * dRadT2(1,ii) + ra(1) * cab(3,3) * dRbdT2(2,ii) &
               +rb(3) * cab(3,2) * dRadT2(1,ii) + ra(1) * cab(3,2) * dRbdT2(3,ii) &
               +rb(2) * cab(1,3) * dRadT2(3,ii) + ra(3) * cab(1,3) * dRbdT2(2,ii) &
               +rb(3) * cab(1,2) * dRadT2(3,ii) + ra(3) * cab(1,2) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(3,2,3) * dRadR1(1,ii) &
               + rb2a(3,2,1) * dRadR1(3,ii)) &
               + 5 *(rb(2) * cab(3,3) * dRadR1(1,ii) + rab(1,2) * dCabdR1(3,3,ii) &
               +rb(3) * cab(3,2) * dRadR1(1,ii) + rab(1,3) * dCabdR1(3,2,ii) &
               +rb(2) * cab(1,3) * dRadR1(3,ii) + rab(3,2) * dCabdR1(1,3,ii) &
               +rb(3) * cab(1,2) * dRadR1(3,ii) + rab(3,3) * dCabdR1(1,2,ii)) &
               + cab(3,3) * dCabdR1(1,2,ii) + cab(1,2) * dCabdR1(3,3,ii) &
               + cab(3,2) * dCabdR1(1,3,ii) + cab(1,3) * dCabdR1(3,2,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(3,1,2) * dRbdR2(3,ii) &
               + ra2b(3,1,3) * dRbdR2(2,ii)) &
               + 5 *(ra(1) * cab(3,3) * dRbdR2(2,ii) + rab(1,2) * dCabdR2(3,3,ii) &
               +ra(1) * cab(3,2) * dRbdR2(3,ii) + rab(1,3) * dCabdR2(3,2,ii) &
               +ra(3) * cab(1,3) * dRbdR2(2,ii) + rab(3,2) * dCabdR2(1,3,ii) &
               +ra(3) * cab(1,2) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(1,2,ii)) &
               + cab(3,3) * dCabdR2(1,2,ii) + cab(1,2) * dCabdR2(3,3,ii) &
               + cab(3,2) * dCabdR2(1,3,ii) + cab(1,3) * dCabdR2(3,2,ii)) * switc
       enddo
    endif
    ! 21s - 21c
    if (Qloc_t_x(3) .and. Qloc_u_x(2)) then
       pref = QLOC_T(3)*QLOC_U(2)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,2,3,1) +5*(rab(2,1)*cab(3,3) &
               +rab(3,1)*cab(2,3) +rab(2,3)*cab(3,1) &
               +rab(3,3)*cab(2,1)) +cab(2,1)*cab(3,3) +cab(3,1)*cab(2,3)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(3,1,3) * dradT1(2,ii) + rb2a(3,1,2) * dradT1(3,ii) &
               +ra2b(3,2,1) * dRbdT1(3,ii) + ra2b(3,2,3) * dRbdT1(1,ii)) &
               + 5 *(rb(1) * cab(3,3) * dRadT1(2,ii) + ra(2) * cab(3,3) * dRbdT1(1,ii) &
               +rb(1) * cab(2,3) * dRadT1(3,ii) + ra(3) * cab(2,3) * dRbdT1(1,ii) &
               +rb(3) * cab(3,1) * dRadT1(2,ii) + ra(2) * cab(3,1) * dRbdT1(3,ii) &
               +rb(3) * cab(2,1) * dRadT1(3,ii) + ra(3) * cab(2,1) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,2,3,1) +5*(rab(2,1)*cab(3,3) &
               +rab(3,1)*cab(2,3) +rab(2,3)*cab(3,1) &
               +rab(3,3)*cab(2,1)) +cab(2,1)*cab(3,3) +cab(3,1)*cab(2,3)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(3,1,3) * dradT2(2,ii) + rb2a(3,1,2) * dradT2(3,ii) &
               +ra2b(3,2,1) * dRbdT2(3,ii) + ra2b(3,2,3) * dRbdT2(1,ii)) &
               + 5 *(rb(1) * cab(3,3) * dRadT2(2,ii) + ra(2) * cab(3,3) * dRbdT2(1,ii) &
               +rb(1) * cab(2,3) * dRadT2(3,ii) + ra(3) * cab(2,3) * dRbdT2(1,ii) &
               +rb(3) * cab(3,1) * dRadT2(2,ii) + ra(2) * cab(3,1) * dRbdT2(3,ii) &
               +rb(3) * cab(2,1) * dRadT2(3,ii) + ra(3) * cab(2,1) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(3,1,3) * dradR1(2,ii) &
               + rb2a(3,1,2) * dradR1(3,ii)) &
               + 5 *(rb(1) * cab(3,3) * dRadR1(2,ii) + rab(2,1) * dCabdR1(3,3,ii) &
               +rb(1) * cab(2,3) * dRadR1(3,ii) + rab(3,1) * dCabdR1(2,3,ii) &
               +rb(3) * cab(3,1) * dRadR1(2,ii) + rab(2,3) * dCabdR1(3,1,ii) &
               +rb(3) * cab(2,1) * dRadR1(3,ii) + rab(3,3) * dCabdR1(2,1,ii)) &
               + cab(3,3) * dCabdR1(2,1,ii) + cab(2,1) * dCabdR1(3,3,ii) &
               + cab(2,3) * dCabdR1(3,1,ii) + cab(3,1) * dCabdR1(2,3,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(3,2,1) * dRbdR2(3,ii) &
               + ra2b(3,2,3) * dRbdR2(1,ii)) &
               + 5 *(ra(2) * cab(3,3) * dRbdR2(1,ii) + rab(2,1) * dCabdR2(3,3,ii) &
               +ra(3) * cab(2,3) * dRbdR2(1,ii) + rab(3,1) * dCabdR2(2,3,ii) &
               +ra(2) * cab(3,1) * dRbdR2(3,ii) + rab(2,3) * dCabdR2(3,1,ii) &
               +ra(3) * cab(2,1) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(2,1,ii)) &
               + cab(3,3) * dCabdR2(2,1,ii) + cab(2,1) * dCabdR2(3,3,ii) &
               + cab(2,3) * dCabdR2(3,1,ii) + cab(3,1) * dCabdR2(2,3,ii)) * switc
       enddo
    endif
    ! 21c - 22c
    if (Qloc_t_x(2) .and. Qloc_u_x(4)) then
       pref = QLOC_T(2)*QLOC_U(4)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(3,1,1,1) -ra2b2(3,1,2,2)) &
               +10*(rab(1,1)*cab(3,1) -rab(1,2)*cab(3,2) &
               +rab(3,1)*cab(1,1) -rab(3,2)*cab(1,2)) +2*(cab(1,1)*cab(3,1) -cab(1,2)*cab(3,2))) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(1,1,1) * dRadT1(3,ii) + rb2a(1,1,3) * dRadT1(1,ii) &
               +2 * ra2b(3,1,1) * dRbdT1(1,ii) &
               -rb2a(2,2,1) * dRadT1(3,ii) - rb2a(2,2,3) * dRadT1(1,ii) &
               -2 * ra2b(3,1,2) * dRbdT1(2,ii)) &
               +10 * (rb(1) * cab(3,1) * dRadT1(1,ii) + ra(1) * cab(3,1) * dRbdT1(1,ii) &
               -rb(2) * cab(3,2) * dRadT1(1,ii) - ra(1) * cab(3,2) * dRbdT1(2,ii) &
               +rb(1) * cab(1,1) * dRadT1(3,ii) + ra(3) * cab(1,1) * dRbdT1(1,ii) &
               -rb(2) * cab(1,2) * dRadT1(3,ii) - ra(3) * cab(1,2) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(3,1,1,1) -ra2b2(3,1,2,2)) &
               +10*(rab(1,1)*cab(3,1) -rab(1,2)*cab(3,2) &
               +rab(3,1)*cab(1,1) -rab(3,2)*cab(1,2)) +2*(cab(1,1)*cab(3,1) -cab(1,2)*cab(3,2))) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(1,1,1) * dRadT2(3,ii) + rb2a(1,1,3) * dRadT2(1,ii) &
               +2 * ra2b(3,1,1) * dRbdT2(1,ii) &
               -rb2a(2,2,1) * dRadT2(3,ii) - rb2a(2,2,3) * dRadT2(1,ii) &
               -2 * ra2b(3,1,2) * dRbdT2(2,ii)) &
               +10 * (rb(1) * cab(3,1) * dRadT2(1,ii) + ra(1) * cab(3,1) * dRbdT2(1,ii) &
               -rb(2) * cab(3,2) * dRadT2(1,ii) - ra(1) * cab(3,2) * dRbdT2(2,ii) &
               +rb(1) * cab(1,1) * dRadT2(3,ii) + ra(3) * cab(1,1) * dRbdT2(1,ii) &
               -rb(2) * cab(1,2) * dRadT2(3,ii) - ra(3) * cab(1,2) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(1,1,1) * dRadR1(3,ii) &
               + rb2a(1,1,3) * dRadR1(1,ii) &
               -rb2a(2,2,1) * dRadR1(3,ii) - rb2a(2,2,3) * dRadR1(1,ii)) &
               +10 * (rb(1) * cab(3,1) * dRadR1(1,ii) +rab(1,1) * dCabdR1(3,1,ii) &
               -rb(2) * cab(3,2) * dRadR1(1,ii) -rab(1,2) * dCabdR1(3,2,ii) &
               +rb(1) * cab(1,1) * dRadR1(3,ii) +rab(3,1) * dCabdR1(1,1,ii) &
               -rb(2) * cab(1,2) * dRadR1(3,ii) -rab(3,2) * dCabdR1(1,2,ii)) &
               + 2 * (cab(3,1) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(3,1,ii) &
               -cab(3,2) * dCabdR1(1,2,ii) - cab(1,2) * dCabdR1(3,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (2 * ra2b(3,1,1) *dRbdR2(1,ii) &
               -2 * ra2b(3,1,2) * dRbdR2(2,ii)) &
               +10 * (ra(1) * cab(3,1) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(3,1,ii) &
               -ra(1) * cab(3,2) * dRbdR2(2,ii) -rab(1,2) * dCabdR2(3,2,ii) &
               +ra(3) * cab(1,1) * dRbdR2(1,ii) +rab(3,1) * dCabdR2(1,1,ii) &
               -ra(3) * cab(1,2) * dRbdR2(2,ii) -rab(3,2) * dCabdR2(1,2,ii)) &
               + 2 * (cab(3,1) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(3,1,ii) &
               -cab(3,2) * dCabdR2(1,2,ii) - cab(1,2) * dCabdR2(3,2,ii))) * switc
       enddo
    endif
    ! 22c - 21c
    if (Qloc_t_x(4) .and. Qloc_u_x(2)) then
       pref = QLOC_T(4)*QLOC_U(2)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(1,1,3,1) -ra2b2(2,2,3,1)) &
               +10*(rab(1,1)*cab(1,3) -rab(2,1)*cab(2,3) &
               +rab(1,3)*cab(1,1) -rab(2,3)*cab(2,1)) +2*(cab(1,1)*cab(1,3) &
               -cab(2,1)*cab(2,3))) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (ra2b(1,1,1) * dRbdT1(3,ii) + ra2b(1,1,3) * dRbdT1(1,ii) &
               +2 * rb2a(3,1,1) * dRadT1(1,ii) &
               -ra2b(2,2,1) * dRbdT1(3,ii) - ra2b(2,2,3) * dRbdT1(1,ii) &
               -2 * rb2a(3,1,2) * dRadT1(2,ii)) &
               +10 * (rb(1) * cab(1,3) * dRadT1(1,ii) + ra(1) * cab(1,3) * dRbdT1(1,ii) &
               -rb(1) * cab(2,3) * dRadT1(2,ii) - ra(2) * cab(2,3) * dRbdT1(1,ii) &
               +rb(3) * cab(1,1) * dRadT1(1,ii) + ra(1) * cab(1,1) * dRbdT1(3,ii) &
               -rb(3) * cab(2,1) * dRadT1(2,ii) - ra(2) * cab(2,1) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(1,1,3,1) -ra2b2(2,2,3,1)) &
               +10*(rab(1,1)*cab(1,3) -rab(2,1)*cab(2,3) &
               +rab(1,3)*cab(1,1) -rab(2,3)*cab(2,1)) +2*(cab(1,1)*cab(1,3) &
               -cab(2,1)*cab(2,3))) * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (ra2b(1,1,1) * dRbdT2(3,ii) + ra2b(1,1,3) * dRbdT2(1,ii) &
               +2 * rb2a(3,1,1) * dRadT2(1,ii) &
               -ra2b(2,2,1) * dRbdT2(3,ii) - ra2b(2,2,3) * dRbdT2(1,ii) &
               -2 * rb2a(3,1,2) * dRadT2(2,ii)) &
               +10 * (rb(1) * cab(1,3) * dRadT2(1,ii) + ra(1) * cab(1,3) * dRbdT2(1,ii) &
               -rb(1) * cab(2,3) * dRadT2(2,ii) - ra(2) * cab(2,3) * dRbdT2(1,ii) &
               +rb(3) * cab(1,1) * dRadT2(1,ii) + ra(1) * cab(1,1) * dRbdT2(3,ii) &
               -rb(3) * cab(2,1) * dRadT2(2,ii) - ra(2) * cab(2,1) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (2* rb2a(3,1,1)* dRadR1(1,ii) &
               -2 * rb2a(3,1,2) * dRadR1(2,ii)) &
               +10 * (rb(1) * cab(1,3) * dRadR1(1,ii) +rab(1,1) * dCabdR1(1,3,ii) &
               -rb(1) * cab(2,3) * dRadR1(2,ii) -rab(2,1) * dCabdR1(2,3,ii) &
               +rb(3) * cab(1,1) * dRadR1(1,ii) +rab(1,3) * dCabdR1(1,1,ii) &
               -rb(3) * cab(2,1) * dRadR1(2,ii) -rab(2,3) * dCabdR1(2,1,ii)) &
               + 2 * (cab(1,3) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(1,3,ii) &
               -cab(2,3) * dCabdR1(2,1,ii) - cab(2,1) * dCabdR1(2,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(1,1,1) * dRbdR2(3,ii) &
               + ra2b(1,1,3) * dRbdR2(1,ii) &
               -ra2b(2,2,1) * dRbdR2(3,ii) - ra2b(2,2,3) * dRbdR2(1,ii)) &
               +10 * (ra(1) * cab(1,3) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(1,3,ii) &
               -ra(2) * cab(2,3) * dRbdR2(1,ii) -rab(2,1) * dCabdR2(2,3,ii) &
               +ra(1) * cab(1,1) * dRbdR2(3,ii) +rab(1,3) * dCabdR2(1,1,ii) &
               -ra(2) * cab(2,1) * dRbdR2(3,ii) -rab(2,3) * dCabdR2(2,1,ii)) &
               + 2 * (cab(1,3) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(1,3,ii) &
               -cab(2,3) * dCabdR2(2,1,ii) - cab(2,1) * dCabdR2(2,3,ii))) * switc
       enddo
    endif
    ! 21c - 22s
    if (Qloc_t_x(2) .and. Qloc_u_x(5)) then
       pref = QLOC_T(2)*QLOC_U(5)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,1,2,1) +5*(rab(1,1)*cab(3,2) &
               +rab(1,2)*cab(3,1) +rab(3,1)*cab(1,2) +rab(3,2)*cab(1,1)) &
               +cab(1,1)*cab(3,2) +cab(1,2)*cab(3,1)) * dRdT1(ii) &
               * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(2,1,1) * dRadT1(3,ii) + rb2a(2,1,3) * dRadT1(1,ii) &
               +ra2b(3,1,1) * dRbdT1(2,ii) + ra2b(3,1,2) * dRbdT1(1,ii)) &
               + 5 *(rb(1) * cab(3,2) * dradT1(1,ii) + ra(1) * cab(3,2) * dRbdT1(1,ii) &
               +rb(2) * cab(3,1) * dradT1(1,ii) + ra(1) * cab(3,1) * dRbdT1(2,ii) &
               +rb(1) * cab(1,2) * dradT1(3,ii) + ra(3) * cab(1,2) * dRbdT1(1,ii) &
               +rb(2) * cab(1,1) * dradT1(3,ii) + ra(3) * cab(1,1) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,1,2,1) +5*(rab(1,1)*cab(3,2) &
               +rab(1,2)*cab(3,1) +rab(3,1)*cab(1,2) &
               +rab(3,2)*cab(1,1)) +cab(1,1)*cab(3,2) +cab(1,2)*cab(3,1)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(2,1,1) * dRadT2(3,ii) + rb2a(2,1,3) * dRadT2(1,ii) &
               +ra2b(3,1,1) * dRbdT2(2,ii) + ra2b(3,1,2) * dRbdT2(1,ii)) &
               + 5 *(rb(1) * cab(3,2) * dradT2(1,ii) + ra(1) * cab(3,2) * dRbdT2(1,ii) &
               +rb(2) * cab(3,1) * dradT2(1,ii) + ra(1) * cab(3,1) * dRbdT2(2,ii) &
               +rb(1) * cab(1,2) * dradT2(3,ii) + ra(3) * cab(1,2) * dRbdT2(1,ii) &
               +rb(2) * cab(1,1) * dradT2(3,ii) + ra(3) * cab(1,1) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(2,1,1) * dRadR1(3,ii) &
               + rb2a(2,1,3) * dRadR1(1,ii)) &
               + 5 *(rb(1) * cab(3,2) * dradR1(1,ii) +rab(1,1) * dCabdR1(3,2,ii) &
               +rb(2) * cab(3,1) * dradR1(1,ii) +rab(1,2) * dCabdR1(3,1,ii) &
               +rb(1) * cab(1,2) * dradR1(3,ii) +rab(3,1) * dCabdR1(1,2,ii) &
               +rb(2) * cab(1,1) * dradR1(3,ii) +rab(3,2) * dCabdR1(1,1,ii)) &
               + cab(3,2) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(3,2,ii) &
               + cab(3,1) * dCabdR1(1,2,ii) + cab(1,2) * dCabdR1(3,1,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(3,1,1) * dRbdR2(2,ii) &
               + ra2b(3,1,2) * dRbdR2(1,ii)) &
               + 5 *(ra(1) * cab(3,2) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(3,2,ii) &
               +ra(1) * cab(3,1) * dRbdR2(2,ii) +rab(1,2) * dCabdR2(3,1,ii) &
               +ra(3) * cab(1,2) * dRbdR2(1,ii) +rab(3,1) * dCabdR2(1,2,ii) &
               +ra(3) * cab(1,1) * dRbdR2(2,ii) +rab(3,2) * dCabdR2(1,1,ii)) &
               + cab(3,2) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(3,2,ii) &
               + cab(3,1) * dCabdR2(1,2,ii) + cab(1,2) * dCabdR2(3,1,ii)) * switc
       enddo
    endif
    ! 22s - 21c
    if (Qloc_t_x(5) .and. Qloc_u_x(2)) then
       pref = QLOC_T(5)*QLOC_U(2)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(2,1,3,1) +5*(rab(1,1)*cab(2,3) &
               +rab(2,1)*cab(1,3) +rab(1,3)*cab(2,1) &
               +rab(2,3)*cab(1,1)) +cab(1,1)*cab(2,3) +cab(2,1)*cab(1,3)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(ra2b(2,1,1) * dRbdT1(3,ii) + ra2b(2,1,3) * dRbdT1(1,ii) &
               +rb2a(3,1,1) * dRadT1(2,ii) + rb2a(3,1,2) * dRadT1(1,ii)) &
               + 5 *(rb(1) * cab(2,3) * dradT1(1,ii) + ra(1) * cab(2,3) * dRbdT1(1,ii) &
               +rb(1) * cab(1,3) * dradT1(2,ii) + ra(2) * cab(1,3) * dRbdT1(1,ii) &
               +rb(3) * cab(2,1) * dradT1(1,ii) + ra(1) * cab(2,1) * dRbdT1(3,ii) &
               +rb(3) * cab(1,1) * dradT1(2,ii) + ra(2) * cab(1,1) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(2,1,3,1) +5*(rab(1,1)*cab(2,3) &
               +rab(2,1)*cab(1,3) +rab(1,3)*cab(2,1) &
               +rab(2,3)*cab(1,1)) +cab(1,1)*cab(2,3) +cab(2,1)*cab(1,3)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(ra2b(2,1,1) * dRbdT2(3,ii) + ra2b(2,1,3) * dRbdT2(1,ii) &
               +rb2a(3,1,1) * dRadT2(2,ii) + rb2a(3,1,2) * dRadT2(1,ii)) &
               + 5 *(rb(1) * cab(2,3) * dradT2(1,ii) + ra(1) * cab(2,3) * dRbdT2(1,ii) &
               +rb(1) * cab(1,3) * dradT2(2,ii) + ra(2) * cab(1,3) * dRbdT2(1,ii) &
               +rb(3) * cab(2,1) * dradT2(1,ii) + ra(1) * cab(2,1) * dRbdT2(3,ii) &
               +rb(3) * cab(1,1) * dradT2(2,ii) + ra(2) * cab(1,1) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(3,1,1) * dRadR1(2,ii) &
               + rb2a(3,1,2) * dRadR1(1,ii)) &
               + 5 *(rb(1) * cab(2,3) * dradR1(1,ii) +rab(1,1) * dCabdR1(2,3,ii) &
               +rb(1) * cab(1,3) * dradR1(2,ii) +rab(2,1) * dCabdR1(1,3,ii) &
               +rb(3) * cab(2,1) * dradR1(1,ii) +rab(1,3) * dCabdR1(2,1,ii) &
               +rb(3) * cab(1,1) * dradR1(2,ii) +rab(2,3) * dCabdR1(1,1,ii)) &
               + cab(2,3) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(2,3,ii) &
               + cab(1,3) * dCabdR1(2,1,ii) + cab(2,1) * dCabdR1(1,3,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(2,1,1) * dRbdR2(3,ii) &
               + ra2b(2,1,3) * dRbdR2(1,ii)) &
               + 5 *(ra(1) * cab(2,3) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(2,3,ii) &
               +ra(2) * cab(1,3) * dRbdR2(1,ii) +rab(2,1) * dCabdR2(1,3,ii) &
               +ra(1) * cab(2,1) * dRbdR2(3,ii) +rab(1,3) * dCabdR2(2,1,ii) &
               +ra(2) * cab(1,1) * dRbdR2(3,ii) +rab(2,3) * dCabdR2(1,1,ii)) &
               + cab(2,3) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(2,3,ii) &
               + cab(1,3) * dCabdR2(2,1,ii) + cab(2,1) * dCabdR2(1,3,ii)) * switc
       enddo
    endif
    ! 21s - 21s
    if (Qloc_t_x(3) .and. Qloc_u_x(3)) then
       pref = QLOC_T(3)*QLOC_U(3)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,2,3,2) +5*(rab(2,2)*cab(3,3) &
               +rab(2,3)*cab(3,2) +rab(3,2)*cab(2,3) &
               +rab(3,3)*cab(2,2)) +cab(2,2)*cab(3,3) +cab(2,3)*cab(3,2)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(3,2,2) * dRadT1(3,ii) + rb2a(3,2,3) * dRadT1(2,ii) &
               +ra2b(3,2,2) * dRbdT1(3,ii) + ra2b(3,2,3) * dRbdT1(2,ii)) &
               +5 * (rb(2) * cab(3,3) * dRadT1(2,ii) + ra(2) * cab(3,3) * dRbdT1(2,ii) &
               +rb(3) * cab(3,2) * dRadT1(2,ii) + ra(2) * cab(3,2) * dRbdT1(3,ii) &
               +rb(2) * cab(2,3) * dRadT1(3,ii) + ra(3) * cab(2,3) * dRbdT1(2,ii) &
               +rb(3) * cab(2,2) * dRadT1(3,ii) + ra(3) * cab(2,2) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,2,3,2) +5*(rab(2,2)*cab(3,3) &
               +rab(2,3)*cab(3,2) +rab(3,2)*cab(2,3) &
               +rab(3,3)*cab(2,2)) +cab(2,2)*cab(3,3) +cab(2,3)*cab(3,2)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 * (rb2a(3,2,2) * dRadT2(3,ii) + rb2a(3,2,3) * dRadT2(2,ii) &
               +ra2b(3,2,2) * dRbdT2(3,ii) + ra2b(3,2,3) * dRbdT2(2,ii)) &
               +5 * (rb(2) * cab(3,3) * dRadT2(2,ii) + ra(2) * cab(3,3) * dRbdT2(2,ii) &
               +rb(3) * cab(3,2) * dRadT2(2,ii) + ra(2) * cab(3,2) * dRbdT2(3,ii) &
               +rb(2) * cab(2,3) * dRadT2(3,ii) + ra(3) * cab(2,3) * dRbdT2(2,ii) &
               +rb(3) * cab(2,2) * dRadT2(3,ii) + ra(3) * cab(2,2) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(3,2,2) * dRadR1(3,ii) &
               + rb2a(3,2,3) * dRadR1(2,ii)) &
               +5 * (rb(2) * cab(3,3) * dRadR1(2,ii) + rab(2,2) * dCabdR1(3,3,ii) &
               +rb(3) * cab(3,2) * dRadR1(2,ii) + rab(2,3) * dCabdR1(3,2,ii) &
               +rb(2) * cab(2,3) * dRadR1(3,ii) + rab(3,2) * dCabdR1(2,3,ii) &
               +rb(3) * cab(2,2) * dRadR1(3,ii) + rab(3,3) * dCabdR1(2,2,ii)) &
               + cab(3,3) * dCabdR1(2,2,ii) + cab(2,2) * dCabdR1(3,3,ii) &
               + cab(3,2) * dCabdR1(2,3,ii) + cab(2,3) * dCabdR1(3,2,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(3,2,2) * dRbdR2(3,ii) &
               + ra2b(3,2,3) * dRbdR2(2,ii)) &
               +5 * (ra(2) * cab(3,3) * dRbdR2(2,ii) + rab(2,2) * dCabdR2(3,3,ii) &
               +ra(2) * cab(3,2) * dRbdR2(3,ii) + rab(2,3) * dCabdR2(3,2,ii) &
               +ra(3) * cab(2,3) * dRbdR2(2,ii) + rab(3,2) * dCabdR2(2,3,ii) &
               +ra(3) * cab(2,2) * dRbdR2(3,ii) + rab(3,3) * dCabdR2(2,2,ii)) &
               + cab(3,3) * dCabdR2(2,2,ii) + cab(2,2) * dCabdR2(3,3,ii) &
               + cab(3,2) * dCabdR2(2,3,ii) + cab(2,3) * dCabdR2(3,2,ii)) * switc
       enddo
    endif
    ! 21s - 22c
    if (Qloc_t_x(3) .and. Qloc_u_x(4)) then
       pref = QLOC_T(3)*QLOC_U(4)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(3,2,1,1) -ra2b2(3,2,2,2)) &
               +10*(rab(2,1)*cab(3,1) -rab(2,2)*cab(3,2) &
               +rab(3,1)*cab(2,1) -rab(3,2)*cab(2,2)) +2*(cab(2,1)*cab(3,1) &
               -cab(2,2)*cab(3,2))) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(1,1,2) * dRadT1(3,ii) + rb2a(1,1,3) * dRadT1(2,ii) &
               + 2 * ra2b(3,2,1) * dRbdT1(1,ii) &
               -rb2a(2,2,2) * dRadT1(3,ii) - rb2a(2,2,3) * dRadT1(2,ii) &
               - 2 * ra2b(3,2,2) * dRbdT1(2,ii)) &
               +10 *(rb(1) * cab(3,1) * dRadT1(2,ii) + ra(2) * cab(3,1) * dRbdT1(1,ii) &
               -rb(2) * cab(3,2) * dRadT1(2,ii) - ra(2) * cab(3,2) * dRbdT1(2,ii) &
               +rb(1) * cab(2,1) * dRadT1(3,ii) + ra(3) * cab(2,1) * dRbdT1(1,ii) &
               -rb(2) * cab(2,2) * dRadT1(3,ii) - ra(3) * cab(2,2) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(3,2,1,1) -ra2b2(3,2,2,2)) &
               +10*(rab(2,1)*cab(3,1) -rab(2,2)*cab(3,2) &
               +rab(3,1)*cab(2,1) -rab(3,2)*cab(2,2)) +2*(cab(2,1)*cab(3,1) -cab(2,2)*cab(3,2))) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(rb2a(1,1,2) * dRadT2(3,ii) + rb2a(1,1,3) * dRadT2(2,ii) &
               + 2 * ra2b(3,2,1) * dRbdT2(1,ii) &
               -rb2a(2,2,2) * dRadT2(3,ii) - rb2a(2,2,3) * dRadT2(2,ii) &
               - 2 * ra2b(3,2,2) * dRbdT2(2,ii)) &
               +10 *(rb(1) * cab(3,1) * dRadT2(2,ii) + ra(2) * cab(3,1) * dRbdT2(1,ii) &
               -rb(2) * cab(3,2) * dRadT2(2,ii) - ra(2) * cab(3,2) * dRbdT2(2,ii) &
               +rb(1) * cab(2,1) * dRadT2(3,ii) + ra(3) * cab(2,1) * dRbdT2(1,ii) &
               -rb(2) * cab(2,2) * dRadT2(3,ii) - ra(3) * cab(2,2) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(1,1,2) * dRadR1(3,ii) &
               + rb2a(1,1,3) * dRadR1(2,ii) &
               -rb2a(2,2,2) * dRadR1(3,ii) - rb2a(2,2,3) * dRadR1(2,ii)) &
               +10 *(rb(1) * cab(3,1) * dRadR1(2,ii) +rab(2,1) * dCabdR1(3,1,ii) &
               -rb(2) * cab(3,2) * dRadR1(2,ii) -rab(2,2) * dCabdR1(3,2,ii) &
               +rb(1) * cab(2,1) * dRadR1(3,ii) +rab(3,1) * dCabdR1(2,1,ii) &
               -rb(2) * cab(2,2) * dRadR1(3,ii) -rab(3,2) * dCabdR1(2,2,ii)) &
               + 2 *(cab(3,1) * dCabdR1(2,1,ii) + cab(2,1) * dCabdR1(3,1,ii) &
               -cab(3,2) * dCabdR1(2,2,ii) - cab(2,2) * dCabdR1(3,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(2 * ra2b(3,2,1)* dRbdR2(1,ii) &
               - 2 * ra2b(3,2,2) * dRbdR2(2,ii)) &
               +10 *(ra(2) * cab(3,1) * dRbdR2(1,ii) +rab(2,1) * dCabdR2(3,1,ii) &
               -ra(2) * cab(3,2) * dRbdR2(2,ii) -rab(2,2) * dCabdR2(3,2,ii) &
               +ra(3) * cab(2,1) * dRbdR2(1,ii) +rab(3,1) * dCabdR2(2,1,ii) &
               -ra(3) * cab(2,2) * dRbdR2(2,ii) -rab(3,2) * dCabdR2(2,2,ii)) &
               + 2 *(cab(3,1) * dCabdR2(2,1,ii) + cab(2,1) * dCabdR2(3,1,ii) &
               -cab(3,2) * dCabdR2(2,2,ii) - cab(2,2) * dCabdR2(3,2,ii))) * switc
       enddo
    endif
    ! 22c - 21s
    if (Qloc_t_x(4) .and. Qloc_u_x(3)) then
       pref = QLOC_T(4)*QLOC_U(3)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(1,1,3,2) -ra2b2(2,2,3,2)) &
               +10*(rab(1,2)*cab(1,3) -rab(2,2)*cab(2,3) &
               +rab(1,3)*cab(1,2) -rab(2,3)*cab(2,2)) &
               +2*(cab(1,2)*cab(1,3) -cab(2,2)*cab(2,3))) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(ra2b(1,1,2) * dRbdT1(3,ii) + ra2b(1,1,3) * dRbdT1(2,ii) &
               + 2 * rb2a(3,2,1) * dRadT1(1,ii) &
               -ra2b(2,2,2) * dRbdT1(3,ii) - ra2b(2,2,3) * dRbdT1(2,ii) &
               - 2 * rb2a(3,2,2) * dRadT1(2,ii)) &
               +10 *(rb(2) * cab(1,3) * dRadT1(1,ii) + ra(1) * cab(1,3) * dRbdT1(2,ii) &
               -rb(2) * cab(2,3) * dRadT1(2,ii) - ra(2) * cab(2,3) * dRbdT1(2,ii) &
               +rb(3) * cab(1,2) * dRadT1(1,ii) + ra(1) * cab(1,2) * dRbdT1(3,ii) &
               -rb(3) * cab(2,2) * dRadT1(2,ii) - ra(2) * cab(2,2) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(1,1,3,2) -ra2b2(2,2,3,2)) &
               +10*(rab(1,2)*cab(1,3) -rab(2,2)*cab(2,3) &
               +rab(1,3)*cab(1,2) -rab(2,3)*cab(2,2)) &
               +2*(cab(1,2)*cab(1,3) -cab(2,2)*cab(2,3))) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) + &
               Rm5 * switc * (35 *(ra2b(1,1,2) * dRbdT2(3,ii) + ra2b(1,1,3) * dRbdT2(2,ii) &
               + 2 * rb2a(3,2,1) * dRadT2(1,ii) &
               -ra2b(2,2,2) * dRbdT2(3,ii) - ra2b(2,2,3) * dRbdT2(2,ii) &
               - 2 * rb2a(3,2,2) * dRadT2(2,ii)) &
               +10 *(rb(2) * cab(1,3) * dRadT2(1,ii) + ra(1) * cab(1,3) * dRbdT2(2,ii) &
               -rb(2) * cab(2,3) * dRadT2(2,ii) - ra(2) * cab(2,3) * dRbdT2(2,ii) &
               +rb(3) * cab(1,2) * dRadT2(1,ii) + ra(1) * cab(1,2) * dRbdT2(3,ii) &
               -rb(3) * cab(2,2) * dRadT2(2,ii) - ra(2) * cab(2,2) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(2* rb2a(3,2,1)* dRadR1(1,ii) &
               - 2 * rb2a(3,2,2) * dRadR1(2,ii)) &
               +10 *(rb(2) * cab(1,3) * dRadR1(1,ii) +rab(1,2) * dCabdR1(1,3,ii) &
               -rb(2) * cab(2,3) * dRadR1(2,ii) -rab(2,2) * dCabdR1(2,3,ii) &
               +rb(3) * cab(1,2) * dRadR1(1,ii) +rab(1,3) * dCabdR1(1,2,ii) &
               -rb(3) * cab(2,2) * dRadR1(2,ii) -rab(2,3) * dCabdR1(2,2,ii)) &
               + 2 *(cab(1,3) * dCabdR1(1,2,ii) + cab(1,2) * dCabdR1(1,3,ii) &
               -cab(2,3) * dCabdR1(2,2,ii) - cab(2,2) * dCabdR1(2,3,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(1,1,2) * dRbdR2(3,ii) &
               + ra2b(1,1,3) * dRbdR2(2,ii) &
               -ra2b(2,2,2) * dRbdR2(3,ii) - ra2b(2,2,3) * dRbdR2(2,ii)) &
               +10 *(ra(1) * cab(1,3) * dRbdR2(2,ii) +rab(1,2) * dCabdR2(1,3,ii) &
               -ra(2) * cab(2,3) * dRbdR2(2,ii) -rab(2,2) * dCabdR2(2,3,ii) &
               +ra(1) * cab(1,2) * dRbdR2(3,ii) +rab(1,3) * dCabdR2(1,2,ii) &
               -ra(2) * cab(2,2) * dRbdR2(3,ii) -rab(2,3) * dCabdR2(2,2,ii)) &
               + 2 *(cab(1,3) * dCabdR2(1,2,ii) + cab(1,2) * dCabdR2(1,3,ii) &
               -cab(2,3) * dCabdR2(2,2,ii) - cab(2,2) * dCabdR2(2,3,ii))) * switc
       enddo
    endif
    ! 21s - 22s
    if (Qloc_t_x(3) .and. Qloc_u_x(5)) then
       pref = QLOC_T(3)*QLOC_U(5)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(3,2,2,1) +5*(rab(2,1)*cab(3,2) &
               +rab(2,2)*cab(3,1) +rab(3,1)*cab(2,2) &
               +rab(3,2)*cab(2,1)) +cab(2,1)*cab(3,2) &
               +cab(2,2)*cab(3,1)) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (rb2a(2,1,2) * dRadT1(3,ii) + rb2a(2,1,3) * dRadT1(2,ii) &
               +ra2b(3,2,1) * dRbdT1(2,ii) + ra2b(3,2,2) * dRbdT1(1,ii)) &
               +5 * (rb(1) * cab(3,2) * dradT1(2,ii) + ra(2) * cab(3,2) * dRbdT1(1,ii) &
               +rb(2) * cab(3,1) * dradT1(2,ii) + ra(2) * cab(3,1) * dRbdT1(2,ii) &
               +rb(1) * cab(2,2) * dradT1(3,ii) + ra(3) * cab(2,2) * dRbdT1(1,ii) &
               +rb(2) * cab(2,1) * dradT1(3,ii) + ra(3) * cab(2,1) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(3,2,2,1) +5*(rab(2,1)*cab(3,2) &
               +rab(2,2)*cab(3,1) +rab(3,1)*cab(2,2) &
               +rab(3,2)*cab(2,1)) +cab(2,1)*cab(3,2) &
               +cab(2,2)*cab(3,1)) * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (rb2a(2,1,2) * dRadT2(3,ii) + rb2a(2,1,3) * dRadT2(2,ii) &
               +ra2b(3,2,1) * dRbdT2(2,ii) + ra2b(3,2,2) * dRbdT2(1,ii)) &
               +5 * (rb(1) * cab(3,2) * dradT2(2,ii) + ra(2) * cab(3,2) * dRbdT2(1,ii) &
               +rb(2) * cab(3,1) * dradT2(2,ii) + ra(2) * cab(3,1) * dRbdT2(2,ii) &
               +rb(1) * cab(2,2) * dradT2(3,ii) + ra(3) * cab(2,2) * dRbdT2(1,ii) &
               +rb(2) * cab(2,1) * dradT2(3,ii) + ra(3) * cab(2,1) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(2,1,2) * dRadR1(3,ii) &
               + rb2a(2,1,3) * dRadR1(2,ii)) &
               +5 * (rb(1) * cab(3,2) * dradR1(2,ii) +rab(2,1) * dCabdR1(3,2,ii) &
               +rb(2) * cab(3,1) * dradR1(2,ii) +rab(2,2) * dCabdR1(3,1,ii) &
               +rb(1) * cab(2,2) * dradR1(3,ii) +rab(3,1) * dCabdR1(2,2,ii) &
               +rb(2) * cab(2,1) * dradR1(3,ii) +rab(3,2) * dCabdR1(2,1,ii)) &
               + cab(3,2) * dCabdR1(2,1,ii) + cab(2,1) * dCabdR1(3,2,ii) &
               + cab(3,1) * dCabdR1(2,2,ii) + cab(2,2) * dCabdR1(3,1,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(3,2,1) * dRbdR2(2,ii) &
               + ra2b(3,2,2) * dRbdR2(1,ii)) &
               +5 * (ra(2) * cab(3,2) * dRbdR2(1,ii) +rab(2,1) * dCabdR2(3,2,ii) &
               +ra(2) * cab(3,1) * dRbdR2(2,ii) +rab(2,2) * dCabdR2(3,1,ii) &
               +ra(3) * cab(2,2) * dRbdR2(1,ii) +rab(3,1) * dCabdR2(2,2,ii) &
               +ra(3) * cab(2,1) * dRbdR2(2,ii) +rab(3,2) * dCabdR2(2,1,ii)) &
               + cab(3,2) * dCabdR2(2,1,ii) + cab(2,1) * dCabdR2(3,2,ii) &
               + cab(3,1) * dCabdR2(2,2,ii) + cab(2,2) * dCabdR2(3,1,ii)) * switc
       enddo
    endif
    ! 22s - 21s
    if (Qloc_t_x(5) .and. Qloc_u_x(3)) then
       pref = QLOC_T(5)*QLOC_U(3)
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(2,1,3,2) +5*(rab(1,2)*cab(2,3) &
               +rab(2,2)*cab(1,3) +rab(1,3)*cab(2,2) &
               +rab(2,3)*cab(1,2)) +cab(1,2)*cab(2,3) +cab(2,2)*cab(1,3)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (ra2b(2,1,2) * dRbdT1(3,ii) + ra2b(2,1,3) * dRbdT1(2,ii) &
               +rb2a(3,2,1) * dRadT1(2,ii) + rb2a(3,2,2) * dRadT1(1,ii)) &
               +5 * (rb(2) * cab(2,3) * dradT1(1,ii) + ra(1) * cab(2,3) * dRbdT1(2,ii) &
               +rb(2) * cab(1,3) * dradT1(2,ii) + ra(2) * cab(1,3) * dRbdT1(2,ii) &
               +rb(3) * cab(2,2) * dradT1(1,ii) + ra(1) * cab(2,2) * dRbdT1(3,ii) &
               +rb(3) * cab(1,2) * dradT1(2,ii) + ra(2) * cab(1,2) * dRbdT1(3,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(2,1,3,2) +5*(rab(1,2)*cab(2,3) &
               +rab(2,2)*cab(1,3) +rab(1,3)*cab(2,2) &
               +rab(2,3)*cab(1,2)) +cab(1,2)*cab(2,3) +cab(2,2)*cab(1,3)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (ra2b(2,1,2) * dRbdT2(3,ii) + ra2b(2,1,3) * dRbdT2(2,ii) &
               +rb2a(3,2,1) * dRadT2(2,ii) + rb2a(3,2,2) * dRadT2(1,ii)) &
               +5 * (rb(2) * cab(2,3) * dradT2(1,ii) + ra(1) * cab(2,3) * dRbdT2(2,ii) &
               +rb(2) * cab(1,3) * dradT2(2,ii) + ra(2) * cab(1,3) * dRbdT2(2,ii) &
               +rb(3) * cab(2,2) * dradT2(1,ii) + ra(1) * cab(2,2) * dRbdT2(3,ii) &
               +rb(3) * cab(1,2) * dradT2(2,ii) + ra(2) * cab(1,2) * dRbdT2(3,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(3,2,1) * dRadR1(2,ii) &
               + rb2a(3,2,2) * dRadR1(1,ii)) &
               +5 * (rb(2) * cab(2,3) * dradR1(1,ii) +rab(1,2) * dCabdR1(2,3,ii) &
               +rb(2) * cab(1,3) * dradR1(2,ii) +rab(2,2) * dCabdR1(1,3,ii) &
               +rb(3) * cab(2,2) * dradR1(1,ii) +rab(1,3) * dCabdR1(2,2,ii) &
               +rb(3) * cab(1,2) * dradR1(2,ii) +rab(2,3) * dCabdR1(1,2,ii)) &
               + cab(2,3) * dCabdR1(1,2,ii) + cab(1,2) * dCabdR1(2,3,ii) &
               + cab(1,3) * dCabdR1(2,2,ii) + cab(2,2) * dCabdR1(1,3,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(2,1,2) * dRbdR2(3,ii) &
               + ra2b(2,1,3) * dRbdR2(2,ii)) &
               +5 * (ra(1) * cab(2,3) * dRbdR2(2,ii) +rab(1,2) * dCabdR2(2,3,ii) &
               +ra(2) * cab(1,3) * dRbdR2(2,ii) +rab(2,2) * dCabdR2(1,3,ii) &
               +ra(1) * cab(2,2) * dRbdR2(3,ii) +rab(1,3) * dCabdR2(2,2,ii) &
               +ra(2) * cab(1,2) * dRbdR2(3,ii) +rab(2,3) * dCabdR2(1,2,ii)) &
               + cab(2,3) * dCabdR2(1,2,ii) + cab(1,2) * dCabdR2(2,3,ii) &
               + cab(1,3) * dCabdR2(2,2,ii) + cab(2,2) * dCabdR2(1,3,ii)) * switc
       enddo
    endif
    ! 22c - 22c
    if (Qloc_t_x(4) .and. Qloc_u_x(4)) then
       pref = QLOC_T(4)*QLOC_U(4)*0.25
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(1,1,1,1) -ra2b2(1,1,2,2) &
               -ra2b2(2,2,1,1) +ra2b2(2,2,2,2)) &
               +20*(rab(1,1)*cab(1,1) -rab(1,2)*cab(1,2) -rab(2,1)*cab(2,1) +rab(2,2)*cab(2,2)) &
               +2*(cab(1,1)**2 -cab(1,2)**2 -cab(2,1)**2 +cab(2,2)**2)) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * rb2a(1,1,1) * dRadT1(1,ii) + 2 * ra2b(1,1,1) * dRbdT1(1,ii) &
               -2 * rb2a(2,2,1) * dRadT1(1,ii) - 2 * ra2b(1,1,2) * dRbdT1(2,ii) &
               -2 * rb2a(1,1,2) * dRadT1(2,ii) - 2 * ra2b(2,2,1) * dRbdT1(1,ii) &
               +2 * rb2a(2,2,2) * dRadT1(2,ii) + 2 * ra2b(2,2,2) * dRbdT1(2,ii)) &
               +20 *(rb(1) * cab(1,1) *dRadT1(1,ii) + ra(1) * cab(1,1) * dRbdT1(1,ii) &
               -rb(2) * cab(1,2) *dRadT1(1,ii) - ra(1) * cab(1,2) * dRbdT1(2,ii) &
               -rb(1) * cab(2,1) *dRadT1(2,ii) - ra(2) * cab(2,1) * dRbdT1(1,ii) &
               +rb(2) * cab(2,2) *dRadT1(2,ii) + ra(2) * cab(2,2) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(1,1,1,1) -ra2b2(1,1,2,2) &
               -ra2b2(2,2,1,1) +ra2b2(2,2,2,2)) &
               +20*(rab(1,1)*cab(1,1) -rab(1,2)*cab(1,2) -rab(2,1)*cab(2,1) +rab(2,2)*cab(2,2)) &
               +2*(cab(1,1)**2 -cab(1,2)**2 -cab(2,1)**2 +cab(2,2)**2)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * rb2a(1,1,1) * dRadT2(1,ii) + 2 * ra2b(1,1,1) * dRbdT2(1,ii) &
               -2 * rb2a(2,2,1) * dRadT2(1,ii) - 2 * ra2b(1,1,2) * dRbdT2(2,ii) &
               -2 * rb2a(1,1,2) * dRadT2(2,ii) - 2 * ra2b(2,2,1) * dRbdT2(1,ii) &
               +2 * rb2a(2,2,2) * dRadT2(2,ii) + 2 * ra2b(2,2,2) * dRbdT2(2,ii)) &
               +20 *(rb(1) * cab(1,1) *dRadT2(1,ii) + ra(1) * cab(1,1) * dRbdT2(1,ii) &
               -rb(2) * cab(1,2) *dRadT2(1,ii) - ra(1) * cab(1,2) * dRbdT2(2,ii) &
               -rb(1) * cab(2,1) *dRadT2(2,ii) - ra(2) * cab(2,1) * dRbdT2(1,ii) &
               +rb(2) * cab(2,2) *dRadT2(2,ii) + ra(2) * cab(2,2) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(2 * rb2a(1,1,1)* dRadR1(1,ii) &
               -2 * rb2a(2,2,1) * dRadR1(1,ii) &
               -2 * rb2a(1,1,2) * dRadR1(2,ii)  +2 * rb2a(2,2,2) * dRadR1(2,ii)) &
               +20 *(rb(1) * cab(1,1) *dRadR1(1,ii) +rab(1,1) * dCabdR1(1,1,ii) &
               -rb(2) * cab(1,2) *dRadR1(1,ii) -rab(1,2) * dCabdR1(1,2,ii) &
               -rb(1) * cab(2,1) *dRadR1(2,ii) -rab(2,1) * dCabdR1(2,1,ii) &
               +rb(2) * cab(2,2) *dRadR1(2,ii) +rab(2,2) * dCabdR1(2,2,ii)) &
               + 2 *(2 * cab(1,1) * dCabdR1(1,1,ii) - 2 * cab(1,2) * dCabdR1(1,2,ii) &
               -2 * cab(2,1) * dCabdR1(2,1,ii) + 2 * cab(2,2) * dCabdR1(2,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(2 * ra2b(1,1,1)* dRbdR2(1,ii) &
               - 2 * ra2b(1,1,2) * dRbdR2(2,ii) &
               -2 * ra2b(2,2,1) * dRbdR2(1,ii) + 2 * ra2b(2,2,2) * dRbdR2(2,ii)) &
               +20 *(ra(1) * cab(1,1) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(1,1,ii) &
               -ra(1) * cab(1,2) * dRbdR2(2,ii) -rab(1,2) * dCabdR2(1,2,ii) &
               -ra(2) * cab(2,1) * dRbdR2(1,ii) -rab(2,1) * dCabdR2(2,1,ii) &
               +ra(2) * cab(2,2) * dRbdR2(2,ii) +rab(2,2) * dCabdR2(2,2,ii)) &
               + 2 *(2 * cab(1,1) * dCabdR2(1,1,ii) - 2 * cab(1,2) * dCabdR2(1,2,ii) &
               -2 * cab(2,1) * dCabdR2(2,1,ii) + 2 * cab(2,2) * dCabdR2(2,2,ii))) * switc
       enddo
    endif
    ! 22c - 22s
    if (Qloc_t_x(4) .and. Qloc_u_x(5)) then
       pref = QLOC_T(4)*QLOC_U(5)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(1,1,2,1) -ra2b2(2,2,2,1)) &
               +10*(rab(1,1)*cab(1,2) +rab(1,2)*cab(1,1) &
               -rab(2,1)*cab(2,2) -rab(2,2)*cab(2,1)) &
               +2*(cab(1,1)*cab(1,2) -cab(2,1)*cab(2,2))) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * rb2a(2,1,1) * dRadT1(1,ii) + ra2b(1,1,1) * dRbdT1(2,ii) &
               +ra2b(1,1,2) * dRbdT1(1,ii) &
               -2 * rb2a(2,1,2) * dRadT1(2,ii) - ra2b(2,2,1) * dRbdT1(2,ii) &
               -ra2b(2,2,2) * dRbdT1(1,ii)) &
               +10 *(rb(1) * cab(1,2) * dRadT1(1,ii) + ra(1) * cab(1,2) * dRbdT1(1,ii) &
               +rb(2) * cab(1,1) * dRadT1(1,ii) + ra(1) * cab(1,1) * dRbdT1(2,ii) &
               -rb(1) * cab(2,2) * dRadT1(2,ii) - ra(2) * cab(2,2) * dRbdT1(1,ii) &
               -rb(2) * cab(2,1) * dRadT1(2,ii) - ra(2) * cab(2,1) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(1,1,2,1) -ra2b2(2,2,2,1)) &
               +10*(rab(1,1)*cab(1,2) +rab(1,2)*cab(1,1) &
               -rab(2,1)*cab(2,2) -rab(2,2)*cab(2,1)) +2*(cab(1,1)*cab(1,2) -cab(2,1)*cab(2,2))) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * rb2a(2,1,1) * dRadT2(1,ii) + ra2b(1,1,1) * dRbdT2(2,ii) &
               +ra2b(1,1,2) * dRbdT2(1,ii) &
               -2 * rb2a(2,1,2) * dRadT2(2,ii) - ra2b(2,2,1) * dRbdT2(2,ii) &
               -ra2b(2,2,2) * dRbdT2(1,ii)) &
               +10 *(rb(1) * cab(1,2) * dRadT2(1,ii) + ra(1) * cab(1,2) * dRbdT2(1,ii) &
               +rb(2) * cab(1,1) * dRadT2(1,ii) + ra(1) * cab(1,1) * dRbdT2(2,ii) &
               -rb(1) * cab(2,2) * dRadT2(2,ii) - ra(2) * cab(2,2) * dRbdT2(1,ii) &
               -rb(2) * cab(2,1) * dRadT2(2,ii) - ra(2) * cab(2,1) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(2 * rb2a(2,1,1)* dRadR1(1,ii) &
               -2 * rb2a(2,1,2) * dRadR1(2,ii)) &
               +10 *(rb(1) * cab(1,2) * dRadR1(1,ii) +rab(1,1) * dCabdR1(1,2,ii) &
               +rb(2) * cab(1,1) * dRadR1(1,ii) +rab(1,2) * dCabdR1(1,1,ii) &
               -rb(1) * cab(2,2) * dRadR1(2,ii) -rab(2,1) * dCabdR1(2,2,ii) &
               -rb(2) * cab(2,1) * dRadR1(2,ii) -rab(2,2) * dCabdR1(2,1,ii)) &
               + 2 *(cab(1,2) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(1,2,ii) &
               -cab(2,2) * dCabdR1(2,1,ii) - cab(2,1) * dCabdR1(2,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(ra2b(1,1,1) * dRbdR2(2,ii) &
               +ra2b(1,1,2) * dRbdR2(1,ii) &
               - ra2b(2,2,1) * dRbdR2(2,ii) -ra2b(2,2,2) * dRbdR2(1,ii)) &
               +10 *(ra(1) * cab(1,2) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(1,2,ii) &
               +ra(1) * cab(1,1) * dRbdR2(2,ii) +rab(1,2) * dCabdR2(1,1,ii) &
               -ra(2) * cab(2,2) * dRbdR2(1,ii) -rab(2,1) * dCabdR2(2,2,ii) &
               -ra(2) * cab(2,1) * dRbdR2(2,ii) -rab(2,2) * dCabdR2(2,1,ii)) &
               + 2 *(cab(1,2) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(1,2,ii) &
               -cab(2,2) * dCabdR2(2,1,ii) - cab(2,1) * dCabdR2(2,2,ii))) * switc
       enddo
    endif
    ! 22s - 22c
    if (Qloc_t_x(5) .and. Qloc_u_x(4)) then
       pref = QLOC_T(5)*QLOC_U(4)*0.5
       do ii = 1,3
          F1(ii) = F1(ii) + pref * ((35*(ra2b2(2,1,1,1) -ra2b2(2,1,2,2)) &
               +10*(rab(1,1)*cab(2,1) +rab(2,1)*cab(1,1) &
               -rab(1,2)*cab(2,2) -rab(2,2)*cab(1,2)) &
               +2*(cab(1,1)*cab(2,1) -cab(1,2)*cab(2,2))) &
               * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * ra2b(2,1,1) * dRbdT1(1,ii) + rb2a(1,1,1) * dRadT1(2,ii) &
               +rb2a(1,1,2) * dRadT1(1,ii) &
               -2 * ra2b(2,1,2) * dRbdT1(2,ii) - rb2a(2,2,1) * dRadT1(2,ii) &
               -rb2a(2,2,2) * dRadT1(1,ii)) &
               +10 *(rb(1) * cab(2,1) * dRadT1(1,ii) &
               + ra(1) * cab(2,1) * dRbdT1(1,ii) &
               +rb(1) * cab(1,1) * dRadT1(2,ii) + ra(2) * cab(1,1) * dRbdT1(1,ii) &
               -rb(2) * cab(2,2) * dRadT1(1,ii) - ra(1) * cab(2,2) * dRbdT1(2,ii) &
               -rb(2) * cab(1,2) * dRadT1(2,ii) - ra(2) * cab(1,2) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*(ra2b2(2,1,1,1) -ra2b2(2,1,2,2)) &
               +10*(rab(1,1)*cab(2,1) +rab(2,1)*cab(1,1) &
               -rab(1,2)*cab(2,2) -rab(2,2)*cab(1,2)) &
               +2*(cab(1,1)*cab(2,1) -cab(1,2)*cab(2,2))) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 *(2 * ra2b(2,1,1) * dRbdT2(1,ii) + rb2a(1,1,1) * dRadT2(2,ii) &
               +rb2a(1,1,2) * dRadT2(1,ii) &
               -2 * ra2b(2,1,2) * dRbdT2(2,ii) - rb2a(2,2,1) * dRadT2(2,ii) &
               -rb2a(2,2,2) * dRadT2(1,ii)) &
               +10 *(rb(1) * cab(2,1) * dRadT2(1,ii) + ra(1) * cab(2,1) * dRbdT2(1,ii) &
               +rb(1) * cab(1,1) * dRadT2(2,ii) + ra(2) * cab(1,1) * dRbdT2(1,ii) &
               -rb(2) * cab(2,2) * dRadT2(1,ii) - ra(1) * cab(2,2) * dRbdT2(2,ii) &
               -rb(2) * cab(1,2) * dRadT2(2,ii) - ra(2) * cab(1,2) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 *(rb2a(1,1,1) * dRadR1(2,ii) &
               +rb2a(1,1,2) * dRadR1(1,ii) &
               - rb2a(2,2,1) * dRadR1(2,ii) -rb2a(2,2,2) * dRadR1(1,ii)) &
               +10 *(rb(1) * cab(2,1) * dRadR1(1,ii) +rab(1,1) * dCabdR1(2,1,ii) &
               +rb(1) * cab(1,1) * dRadR1(2,ii) +rab(2,1) * dCabdR1(1,1,ii) &
               -rb(2) * cab(2,2) * dRadR1(1,ii) -rab(1,2) * dCabdR1(2,2,ii) &
               -rb(2) * cab(1,2) * dRadR1(2,ii) -rab(2,2) * dCabdR1(1,2,ii)) &
               + 2 *(cab(2,1) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(2,1,ii) &
               -cab(2,2) * dCabdR1(1,2,ii) - cab(1,2) * dCabdR1(2,2,ii))) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 *(2 * ra2b(2,1,1)* dRbdR2(1,ii) &
               -2 * ra2b(2,1,2) * dRbdR2(2,ii)) &
               +10 *(ra(1) * cab(2,1) * dRbdR2(1,ii) +rab(1,1) * dCabdR2(2,1,ii) &
               +ra(2) * cab(1,1) * dRbdR2(1,ii) +rab(2,1) * dCabdR2(1,1,ii) &
               -ra(1) * cab(2,2) * dRbdR2(2,ii) -rab(1,2) * dCabdR2(2,2,ii) &
               -ra(2) * cab(1,2) * dRbdR2(2,ii) -rab(2,2) * dCabdR2(1,2,ii)) &
               + 2 *(cab(2,1) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(2,1,ii) &
               -cab(2,2) * dCabdR2(1,2,ii) - cab(1,2) * dCabdR2(2,2,ii))) * switc
       enddo
    endif
    ! 22s - 22s
    if (Qloc_t_x(5) .and. Qloc_u_x(5)) then
       pref = QLOC_T(5)*QLOC_U(5)
       do ii= 1,3
          F1(ii) = F1(ii) + pref * ((35*ra2b2(2,1,2,1) +5*(rab(1,1)*cab(2,2) &
               +rab(1,2)*cab(2,1) +rab(2,1)*cab(1,2) &
               +rab(2,2)*cab(1,1)) +cab(1,1)*cab(2,2) &
               +cab(1,2)*cab(2,1)) * dRdT1(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (rb2a(2,1,1) * dRadT1(2,ii) + rb2a(2,1,2) * dRadT1(1,ii) &
               +ra2b(2,1,1) * dRbdT1(2,ii) + ra2b(2,1,2) * dRbdT1(1,ii)) &
               + 5 * (rb(1) * cab(2,2) * dRadT1(1,ii) + ra(1) * cab(2,2) * dRbdT1(1,ii) &
               +rb(2) * cab(2,1) * dRadT1(1,ii) + ra(1) * cab(2,1) * dRbdT1(2,ii) &
               +rb(1) * cab(1,2) * dRadT1(2,ii) + ra(2) * cab(1,2) * dRbdT1(1,ii) &
               +rb(2) * cab(1,1) * dRadT1(2,ii) + ra(2) * cab(1,1) * dRbdT1(2,ii))))
          F2(ii) = F2(ii) + pref * ((35*ra2b2(2,1,2,1) +5*(rab(1,1)*cab(2,2) &
               +rab(1,2)*cab(2,1) +rab(2,1)*cab(1,2) &
               +rab(2,2)*cab(1,1)) +cab(1,1)*cab(2,2) +cab(1,2)*cab(2,1)) &
               * dRdT2(ii) * (Rm6_5 * switc + Rm5 * Dswitc) +&
               Rm5 * switc * (35 * (rb2a(2,1,1) * dRadT2(2,ii) + rb2a(2,1,2) * dRadT2(1,ii) &
               +ra2b(2,1,1) * dRbdT2(2,ii) + ra2b(2,1,2) * dRbdT2(1,ii)) &
               + 5 * (rb(1) * cab(2,2) * dRadT2(1,ii) + ra(1) * cab(2,2) * dRbdT2(1,ii) &
               +rb(2) * cab(2,1) * dRadT2(1,ii) + ra(1) * cab(2,1) * dRbdT2(2,ii) &
               +rb(1) * cab(1,2) * dRadT2(2,ii) + ra(2) * cab(1,2) * dRbdT2(1,ii) &
               +rb(2) * cab(1,1) * dRadT2(2,ii) + ra(2) * cab(1,1) * dRbdT2(2,ii))))
          tau1(ii) = tau1(ii) + pref * Rm5 * (35 * (rb2a(2,1,1) * dRadR1(2,ii) &
               + rb2a(2,1,2) * dRadR1(1,ii)) &
               + 5 * (rb(1) * cab(2,2) * dRadR1(1,ii) + rab(1,1) * dCabdR1(2,2,ii) &
               +rb(2) * cab(2,1) * dRadR1(1,ii) + rab(1,2) * dCabdR1(2,1,ii) &
               +rb(1) * cab(1,2) * dRadR1(2,ii) + rab(2,1) * dCabdR1(1,2,ii) &
               +rb(2) * cab(1,1) * dRadR1(2,ii) + rab(2,2) * dCabdR1(1,1,ii)) &
               + cab(2,2) * dCabdR1(1,1,ii) + cab(1,1) * dCabdR1(2,2,ii) &
               + cab(2,1) * dCabdR1(1,2,ii) + cab(1,2) * dCabdR1(2,1,ii)) * switc
          tau2(ii) = tau2(ii) + pref * Rm5 * (35 * (ra2b(2,1,1) * dRbdR2(2,ii) &
               + ra2b(2,1,2) * dRbdR2(1,ii)) &
               + 5 * (ra(1) * cab(2,2) * dRbdR2(1,ii) + rab(1,1) * dCabdR2(2,2,ii) &
               +ra(1) * cab(2,1) * dRbdR2(2,ii) + rab(1,2) * dCabdR2(2,1,ii) &
               +ra(2) * cab(1,2) * dRbdR2(1,ii) + rab(2,1) * dCabdR2(1,2,ii) &
               +ra(2) * cab(1,1) * dRbdR2(2,ii) + rab(2,2) * dCabdR2(1,1,ii)) &
               + cab(2,2) * dCabdR2(1,1,ii) + cab(1,1) * dCabdR2(2,2,ii) &
               + cab(2,1) * dCabdR2(1,2,ii) + cab(1,2) * dCabdR2(2,1,ii)) * switc
       enddo
    endif

    return
  end subroutine FQQ

  ! #################

  subroutine propagate_torque(aidx,torq)
    !  This subroutine translates the torques on the atoms into linear forces on
    !  the reference atoms using classical mechanics. The Torque t (input) is
    !  calculated during the force calculation. We project the torque along the
    !  eigenvectors of the inertia tensor (calculated below):
    !  
    !  It(a,b) = sum(i=1,nrefA) m(i) *(Ri^2*d_ab - x(a) _cross_ x(b))
    !  
    !  a and b are indices for the three cartesian axis, Ri is the distance
    !  between the central atom and the current reference atom and d_ab is
    !  Kronecker's delta.  Using the torque and the moment of inertia, the
    !  angular acceleration can be calculated according to
    !
    !  Alpha_rad = tm/Im
    !  
    !  where Im is an eigenvalue of It and tm is the projection of t along the
    !  eigenvector. Using the angular acceleration, the tangential linear
    !  acceleration (as first order approximation or exact equation for an
    !  infinitely small time step) can be calculated as:
    !
    !  a_lin = Alpha_rad * Ri
    !
    !  , where Ri is the distance between the MTP site and the atom over which
    !  we apply the torque.  Putting everything together, the force on a
    !  neighboring atom becomes:
    !
    !  F = m * (tm x Ri) / Im
    !
    ! where "x" refers to a cross product. Finally, the net force applied on all
    ! the neighboring atoms is counterbalanced on the MTP site.
    !
    use energym
    use deriv
    use psf
    use image
    use number
    use inbnd  
    use chm_kinds
    use stream
    use exfunc
    use dimens_fcm
    use comand
    use psf
!    use param
    use consta
    use coord
    use string
    use memory
    use vector
    implicit none

    integer, intent(in) :: aidx
    real(chm_real), dimension(3), intent(in) :: torq

    ! local variables
    real(chm_real) :: torq_abs, Inert_M, b2a, eigenval_t
    real(chm_real), dimension(3) :: xyz0, xyz1, xyz2, xyz3, xyz4, torq_n, rij, &
         tauxr, netforce, eigenvals, partforce
    real(chm_real), dimension(3,3) :: Inert_T, eigenvecs

    integer :: i,j,k,nrefA,t,ier

    b2a = 0.529177249

    ! Count number of reference atoms
    nrefA = 4
    do i = 1,4
       if (refatms(i,aidx) == 0) nrefA = nrefA - 1
    enddo
    
    ! Enter if statement if the MTP atom holds a torque
    if (sum(abs(torq)) > 1.0d-18) then
       torq_abs = sqrt(dot_product(torq,torq))
       torq_n = torq/torq_abs      

       ! Calculate inertia tensor    
       Inert_T = 0.0
       do i = 1,nrefA
          j = refatms(i,aidx)
          ! The tensor is symmetric, only 6 independent coefficients.
          Inert_T(1,1) = Inert_T(1,1) + AMASS(j) * ((Y(j)-Y(aidx))**2 + (Z(j)-Z(aidx))**2) 
          Inert_T(2,2) = Inert_T(2,2) + AMASS(j) * ((X(j)-X(aidx))**2 + (Z(j)-Z(aidx))**2) 
          Inert_T(3,3) = Inert_T(3,3) + AMASS(j) * ((X(j)-X(aidx))**2 + (Y(j)-Y(aidx))**2)      
          Inert_T(1,2) = Inert_T(1,2) - AMASS(j) *  (X(j)-X(aidx))    * (Y(j)-Y(aidx))
          Inert_T(1,3) = Inert_T(1,3) - AMASS(j) *  (X(j)-X(aidx))    * (Z(j)-Z(aidx))
          Inert_T(2,3) = Inert_T(2,3) - AMASS(j) *  (Y(j)-Y(aidx))    * (Z(j)-Z(aidx))
       enddo
       Inert_T(2,1) = Inert_T(1,2)
       Inert_T(3,1) = Inert_T(1,3)
       Inert_T(3,2) = Inert_T(2,3)

       ! Diagonalize inertia tensor
       call EIGRS(Inert_T,3,11,eigenvals,eigenvecs,3,ier)
       IF (IER  >  128) THEN
          IER = IER - 128
          IF(WRNLEV >= 2) WRITE (OUTU,'(A,I6,A)') & 
               ' MTPL> Failed to converge on root number ', IER, '.' 
       ENDIF

       ! Calculate Forces  
       ! We project the input torque along the inertia tensor's eigenvectors.
       netforce = 0.0d0
       do i = 1,nrefA
          j = refatms(i,aidx)
          rij(1) = X(j)-X(aidx)
          rij(2) = Y(j)-Y(aidx)
          rij(3) = Z(j)-Z(aidx)
          do t = 1,3
             call cross3( -dot_product(torq, &
                  eigenvecs(:,t)) * eigenvecs(:,t) , rij, tauxr)
             ! Avoid divide-by-zero issues
             eigenval_t = max(eigenvals(t),1.0d-12)
             partforce = AMASS(j) * tauxr / eigenval_t
             DX(j) = DX(j) - partforce(1)
             DY(j) = DY(j) - partforce(2)
             DZ(j) = DZ(j) - partforce(3)
             netforce = netforce + partforce
          enddo
       enddo
       ! Propagate net force to MTP atom (there's a translational component
       ! since we're away from the center of mass). 
       DX(aidx) = DX(aidx) + netforce(1)
       DY(aidx) = DY(aidx) + netforce(2)
       DZ(aidx) = DZ(aidx) + netforce(3)
    endif

    return
  end subroutine propagate_torque

  ! ################

  subroutine switch(r1,r2,roff2,ron2,switc,dswitc)
    use chm_kinds
    use stream
    use consta

    real(chm_real), intent(in)    :: r1,r2,roff2,ron2
    real(chm_real), intent(inout) :: switc, dswitc
    ! Calculation in Angstrom; Result in BOHR.
    if (r2 .gt. roff2) then
       switc  = 0.
       dswitc = 0.
    elseif (r2 .lt. ron2) then
       switc  = 1.
       dswitc = 0.
    else
       switc  = ((ROFF2 - R2)**2*(ROFF2+2*R2-3*RON2) &
            /(ROFF2 - RON2)**3)
       dswitc = (12.*R1*(R2-ROFF2)*(R2-RON2)) / &
            (ROFF2-RON2)**3 *BOHRR
    endif
    return
  end subroutine switch
  
#endif /*(mtpl_main)*/
end module mtpl_fcm






