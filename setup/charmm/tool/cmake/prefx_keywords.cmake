list(APPEND keywords
  UNIX
  GNU
  EXPAND
  PUTFCM
  NOGRAPHICS
  CONFIGURE)

if(APPLE)  # __APPLE__ is sometimes not defined for gfortran
  list(APPEND keywords APPLE)
endif()

if(static)
  list(APPEND keywords STATIC)
endif()

if(NOT lite)
  list(APPEND keywords
    ACE
    ADUMB
    # Link ADUMB ↔ RXNCOR (umbrella rxncor). Pref key is ADUMBRXNCOR;
    # CHARMM ``?`` substitutions truncate to 8 chars → ``?ADUMBRXN``.
    ADUMBRXNCOR
    AFM
    ASPENER
    ASPMEMB
    AXD
    BLOCK
    CFF
    CGENFF
    CHEMPERT
    CHEQ
    CMAP
    COMP2
    CONSHELIX
    CPATH
    DENBIAS
    DHDGB
    DIMB
    DMCONS
    DOCK
    DYNVV2
    EMAP
    EPMF
    ESTATS
    FACTS
    FASTEW
    FITCHG
    FLEXPARM
    FLUCQ
    FMA
    FOURD
    FSSHK
    GBFIXAT
    GBINLINE
    GCMC
    GENETIC
    GNN
    GRID
    GSBP
    HDGBVDW
    HFB
    HMCOM
    HQBM
    IMCUBES
    LARMORD
    LONEPAIR
    LOOKUP
    LRVDW
    MC
    MEHMC
    MMFF
    MMPT
    MOLVIB
    MRMD
    MTPL
    MSMMPT
    MULTCAN
    NBIPS
    OLDDYN
    OPLS
    OVERLAP
    PATHINT
    PBEQ
    PBOUND
    PERT
    PHMD
    PM1
    PMEPLSMA
    PNOE
    PRIMO
    PRIMSH
    PROTO
    RDC
    RDFSOL
    REPLICA
    RGYCONS
    RMD
    RPATH
    RXNCONS
    RXNCOR
    SASAE
    SCPISM
    SGLD
    SHAPES
    SHELL
    SMBP
    SMD
    SOFTVDW
    SSNMR
    TAMD
    TNPACK
    TPS
    TRAVEL
    TSALLIS
    TSM
    VALBOND
    WCA)
endif()

if(MKL_FOUND)
  list(APPEND keywords MKL)
elseif(FFTW_FOUND)
  list(APPEND keywords FFTW)
endif()

if(colfft AND (MKL_FOUND OR FFTW_FOUND))
  list(APPEND keywords COLFFT)
endif()

if(colfft AND FFTW_FOUND AND (NOT FFTWF_FOUND))
  list(APPEND keywords COLFFT_NOSP)
endif()

if(MPI_Fortran_FOUND)
  list(REMOVE_ITEM keywords TAMD)
  list(APPEND keywords
    MPI
    PARALLEL
    PARAFULL)
endif()

if(ljpme)
  list(APPEND keywords LJPME)
endif()

if(domdec)
  list(APPEND keywords DOMDEC)
endif()

if(domdec_gpu)
  list(APPEND keywords DOMDEC_GPU)
endif()

if(blade)
  list(APPEND keywords BLADE)
endif()

if(eabf)
  list(APPEND keywords EABF)
endif()

if(ensemble OR abpo)
  list(APPEND keywords ENSEMBLE)
endif()

# ABPO (adaptively biased path optimization) is gated on KEY_ABPO in the
# source (source/ensemble/abpo.F90, abpo_ltm.F90, collvar.F90, ensemble.F90,
# source/charmm/miscom.F90, source/energy/energy.F90, source/dynamc/dcntrl.F90).
# It builds on the ENSEMBLE module (abpo.F90 does `use ensemble`), so abpo
# implies ENSEMBLE (handled above) and additionally defines ABPO itself.
if(abpo)
  list(APPEND keywords ABPO)
endif()

if(nih)
  list(APPEND keywords
    LONGLINE
    NIH
    SAVEFCM
    SHAPES
    SGLD)
endif()

if(tsri)
  list(APPEND keywords
    PMEPLSMA
    IMCUBES
    GBINLINE
    DMCONS
    RGYCONS)
endif()

if(fftdock)
  list(APPEND keywords FFTDOCK)
endif()

if(gamus)
  list(APPEND keywords GAMUS)
endif()

if(pipf)
  list(APPEND keywords PIPF)
endif()

if(repdstr)
  list(APPEND keywords
    REPDSTR
    GENCOMM)
endif()

if(stringm)
  list(APPEND keywords
    STRINGM
    MULTICOM
    NEWBESTFIT)
endif()

if(gamess)
  list(REMOVE_ITEM keywords QUANTUM)
  list(APPEND keywords GAMESS)
endif()

if(nwchem)
  list(REMOVE_ITEM keywords QUANTUM MOLVIB MMFF)
  list(APPEND keywords NWCHEM)
endif()

if(OPENMM_FOUND)
    list(APPEND keywords OPENMM)
endif()

if(OpenMMTorch_FOUND)
  list(APPEND keywords OMMTORCH)
endif()

#eemlp-begin
if(mlmm)
  list(APPEND keywords MLMM)
endif()

if(torch)
  list(APPEND keywords MLPTORCH)
endif()
# eemlp-end


if(EXAFMM_FOUND)
    list(APPEND keywords GRAPE LIBGRAPE)
endif()

if(cuda)
  list(APPEND keywords CUDA)
endif()

if(metal)
  list(APPEND keywords METAL)
endif()

# Gate on both `opencl` option and OpenCL_FOUND: the metal block in
# CMakeLists.txt force-OFFs the `opencl` option (mutually exclusive)
# but leaves OpenCL_FOUND set from the earlier find_package().  Using
# OpenCL_FOUND alone would re-enable the OPENCL keyword on metal builds
# and pull the OpenCL Fortran branch back in (link errors).
if(opencl AND OpenCL_FOUND)
  list(APPEND keywords OPENCL)
endif()

if(X11_FOUND)
  list(REMOVE_ITEM keywords NODISPLAY NOGRAPHICS)
  list(APPEND keywords XDISPLAY)
endif()

# QM/MM options begin: QUANTUM and QCHEM are default ON

if(quantum)
  list(APPEND keywords QUANTUM)
endif()

if(qchem)
  list(APPEND keywords QCHEM)
endif()

if(squantm)
  list(APPEND keywords SQUANTM)
endif()

if(sccdftb)
  list(APPEND keywords SCCDFTB)
  if(MKL_FOUND)
    list(APPEND keywords DFTBMKL)
  endif()
endif()

if(g09)
  list(APPEND keywords G09)
endif()

if(qturbo)
  list(APPEND keywords QTURBO)
endif()

if(mndo97)
  list(APPEND keywords MNDO97)
endif()

if(qmmmsemi)
  list(APPEND keywords QMMMSEMI)
endif()

# QM/MM options end

if(add_keywords)
    foreach(keyword ${add_keywords})
        string(TOUPPER ${keyword} upper_keyword)
        list(APPEND keywords ${upper_keyword})
    endforeach()
    list(REMOVE_DUPLICATES keywords)
endif()

if(remove_keywords)
    foreach(keyword ${remove_keywords})
        string(TOUPPER ${keyword} upper_keyword)
        list(REMOVE_ITEM keywords ${upper_keyword})
    endforeach()
endif()

set(prefx_keywords)
foreach(keyword ${keywords})
  set(prefx_keywords "${prefx_keywords}\n${keyword}")
endforeach()
