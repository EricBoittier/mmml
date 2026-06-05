!> some derived types to cut down on the number of procedure arguments
module api_types
  use, intrinsic :: iso_c_binding, only: &
       c_int, c_char, c_double
  use chm_kinds, only: chm_real

  implicit none

  !> @struct min_settings
  !> @brief some options for any minimization
  !
  ! see doc/minimiz.doc for full descriptions of each member
  !
  !> @var nstep integer number of cycles of minimization
  !> @var inbfrq integer frequency of regenerating the non-bonded list
  !> @var ihbfrq integer frequency of regenerating the hydrogen bond list
  !> @var nprint integer step freq for printing
  !> @var gradient integer minimize magnitude of gradient of energy instead of energy
  !> @var numerical integer forces will be determined by finite differences
  !> @var iuncrd integer unit to write out a trajectory file for the minimization
  !> @var nsavc integer frequency for writing out frames (only with iuncrd)
  !> @var iunxyz integer unit to write out ... (?)
  !> @var nsavx integer frequency for writing out frames (only with iunxyz)
  !> @var mxyz integer (only with iunxyz)
  !> @var debug integer extra print for debug purposes
  !> @var step real initial step size for the minimization algorithm
  !> @var tolenr real if change in total energy <= tolenr, exit
  !> @var tolgrd real if ave gradient <= tolgrd, exit
  !> @var tolstp real if ave step size <= tolstp, exit
  type, bind(c) :: min_settings
     integer(c_int) :: &
          nstep = 100, &
          inbfrq = 50, &
          ihbfrq = 50, &
          nprint = 10, &
          gradient = 0, &
          numerical = 0, &
          iuncrd = -1, &
          nsavc = 1, &
          iunxyz = -1, &
          nsavx = 1, &
          mxyz = 1, &
          debug = 0
     real(c_double) :: &
          step = 0.02, &
          tolenr = 0.0, tolgrd = 0.0, tolstp = 0.0
  end type min_settings

  !> @struct min_sd_settings
  !> @brief some options for steepest descent minimization
  !
  ! see doc/minimiz.doc for full descriptions of each member
  !
  !> @var noenergy integer number of cycles of minimization
  !> @var lattice integer with CRYSTAL, also optimize unit cell box size and/or shape
  !> @var nocoords integer with CRYSTAL, only optimize unit cell
  type, bind(c) :: min_sd_settings
     integer(c_int) :: &
          noenergy = 0, &
          lattice = 0, &
          nocoords = 0
  end type min_sd_settings

  !> @struct minimize_settings
  !> @brief some options for abner minimization
  !
  ! see doc/minimiz.doc for full descriptions of each member
  !
  !> @var mindim integer dimension of the basis set stored
  !> @var nprint integer step freq for printing
  !> @var nstep integer number of cycles of minimization
  !> @var tolitr integer max num of energy evals allowed for a step
  !> @var eigrng real smallest eigenval considered nonsingular
  !> @var fmem real memory factor to compute average gradient, step size
  !> @var stplim real maximum Newton Raphson step allowed
  !> @var strict real strictness of descent
  type, bind(c) :: min_abnr_settings
     integer(c_int) :: &
          mindim = 5, tolitr = 100
     real(c_double) :: &
          eigrng = 0.0005, fmem = 0.0, &
          stplim = 1.0, strict = 0.1
  end type min_abnr_settings

  !> @struct dynamics_settings
  !> @brief some options local to the dynamc subroutine
  !
  ! these are read from the command line in dynamc and not dynopt
  ! so give them as parameters to the library version of dynamc
  ! see doc/dynamc.doc for full descriptions of each member
  !
  !> @var ieqfrq integer step freq for scaling velocities to FINALT during equilibration
  !> @var ntrfrq integer step freq for stopping rot and trans of molecule after heating
  !> @var ichecw integer check FINALT + TWINDH < ave sys temp < FINALT + TWINDL every IEQFRQ?
  !              0 is false and nonzero is true
  type, bind(c) :: dynamics_settings
     integer(c_int) :: &
          ieqfrq = 0, &
          ntrfrq = 0, &
          ichecw = 1
     real(c_double) :: tbath = 298.0
     integer(c_int) :: &
          iasors = 0, &
          iasvel = 1, &
          iscale = 0, &
          iscvel = 0, &
          isvfrq = 100, &
          iprfrq = 100, &
          ihtfrq = 0

  end type dynamics_settings

  !> @struct generate_settings
  !> @brief some options for the genpsf subroutine if the caller needs C abi
  !
  ! these are read from the command line in genpsf
  ! so give them as parameters to the library version
  ! see doc/struct.doc for full descriptions
  !
  !> @var new_segid string name of the new segment
  !> @var dup_seg string DUPL name of segment to clone
  !> @var patf string FIRS DEFA patch residue name for terminating residue
  !> @var patl string LAST DEFA patch residue name for terminating residue
  !> @var ldrude integer DRUD 0 create drude particles?
  !> @var lsetic integer SETU 0 append ic table from topo file to main ic table?
  !> @var lwarn integer WARN 0 list elts deleted due to nonexistant atoms?
  !> @var lshow integer SHOW 0
  !> @var langle integer ANGL/NOAN autot overide autogeneration option from topo file?
  !> @var lphi integer DIHE/NODI autod overide autogeneration option from topo file?
  !> @var dmass real DMAS 0.0
  type, bind(c) :: c_generate_settings
     character(kind=c_char, len=1), dimension(1:9) :: &
          new_seg, dup_seg, &
          patf, patl

     integer(c_int) :: &
          ldrude = 0, lsetic = 0, &
          lwarn = 0, lshow = 0, &
          langle = 0, lphi = 0

     real(c_double) :: dmass = 0.0
  end type c_generate_settings

  !> @struct generate_settings
  !> @brief some options local to the genpsf subroutine
  !
  ! these are read from the command line in genpsf
  ! so give them as parameters to the library version
  ! see doc/struct.doc for full descriptions
  !
  !> @var new_segid string name of the new segment
  !> @var dup_seg string DUPL name of segment to clone
  !> @var patf string FIRS DEFA patch residue name for terminating residue
  !> @var patl string LAST DEFA patch residue name for terminating residue
  !> @var ldrude logical DRUD .false. create drude particles?
  !> @var lsetic logical SETU .false. append ic table from topo file to main ic table?
  !> @var lwarn logical WARN .false. list elts deleted due to nonexistant atoms?
  !> @var lshow logical SHOW .false.
  !> @var langle logical ANGL/NOAN autot overide autogeneration option from topo file?
  !> @var lphi logical DIHE/NODI autod overide autogeneration option from topo file?
  !> @var dmass real DMAS 0.0
  type :: generate_settings
     character(len=8) :: &
          new_seg, dup_seg, &
          patf, patl

     logical :: &
          ldrude = .false., lsetic = .false., &
          lwarn = .false., lshow = .false., &
          langle = .false., lphi = .false.

     real(chm_real) :: dmass = 0.0
  end type generate_settings

  enum, bind(c)
     enumerator :: found_type = 0
     enumerator :: found_int = 1
     enumerator :: found_real = 2
     enumerator :: found_bool = 3
  end enum

  enum, bind(c)
     enumerator :: bool = 0
     enumerator :: bool_false = 0
     enumerator :: bool_true = 1
  end enum
  
  !> @struct found_value
  !> @brief charmm stores values of a variety of types by name
  !
  !> @var is_found integer 0 if not found, 1 = integer, 2 = real, ... (see FoundTypes enum)
  !> @var int_val integer the value of the found integer 
  !> @var real_val real the value of the found real
  !> @var bool_val integer the value of the found logical 0 = .false. and 1 = .true.
  type, bind(c) :: found_value
     integer(kind(found_type)) :: is_found = bool_false
     integer(c_int) :: int_val = 0, bool_val = 0
     real(c_double) :: real_val = 0
  end type Found_Value

contains

  !> @brief c_generate_settings -> generate_settings
  !
  !> @param[in] c_opts a set of options for the generate command
  !> @return f_opts version of c_opts with strings and logicals
  function c2f_generate_settings(c_opts) result(f_opts)
    use api_util, only: c2f_string

    implicit none

    type(c_generate_settings) :: c_opts
    type(generate_settings) :: f_opts

    f_opts%new_seg = c2f_string(c_opts%new_seg, 8)
    f_opts%dup_seg = c2f_string(c_opts%dup_seg, 8)
    f_opts%patf = c2f_string(c_opts%patf, 8)
    f_opts%patl = c2f_string(c_opts%patl, 8)

    f_opts%ldrude = c_opts%ldrude .ne. 0
    f_opts%lsetic = c_opts%lsetic .ne. 0
    f_opts%lwarn = c_opts%lwarn .ne. 0
    f_opts%lshow = c_opts%lshow .ne. 0
    f_opts%langle = c_opts%langle .ne. 0
    f_opts%lphi = c_opts%lphi .ne. 0

    f_opts%dmass = c_opts%dmass
  end function c2f_generate_settings

end module api_types
