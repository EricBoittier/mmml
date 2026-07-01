list(APPEND keywords
  UNIX
  GNU
  EXPAND
  PUTFCM
  NOGRAPHICS)

if(static)
  list(APPEND keywords STATIC)
endif(static)

if(as_library)
  list(APPEND keywords LIBRARY)
endif(as_library)

if(NOT lite)
  list(APPEND keywords
    ACE
    ADUMB
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

if(MPI_FOUND)
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

if(abpo)
  list(APPEND keywords ABPO)
endif()

if(ensemble OR abpo)
  list(APPEND keywords ENSEMBLE)
endif()

if(nih)
  list(APPEND keywords
    LONGLINE
    NIH
    SAVEFCM
    SHAPES
    SGLD)
endif(nih)

if(tsri)
  list(APPEND keywords
    PMEPLSMA
    IMCUBES
    GBINLINE
    DMCONS
    RGYCONS)
endif(tsri)

if(fftdock)
  list(APPEND keywords FFTDOCK)
endif()

if(gamus)
  list(APPEND keywords GAMUS)
endif()

if(pipf)
  list(APPEND keywords PIPF)
endif(pipf)

if(repdstr)
  list(APPEND keywords
    REPDSTR
    GENCOMM)
endif(repdstr)

if(stringm)
  list(APPEND keywords
    STRINGM
    MULTICOM
    NEWBESTFIT)
endif(stringm)

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

if(EXAFMM_FOUND)
    list(APPEND keywords GRAPE LIBGRAPE)
endif()

if(CUDA_FOUND)
  list(APPEND keywords CUDA)
endif()

if(OpenCL_FOUND)
  list(APPEND keywords OPENCL)
endif()

if(X11_FOUND)
  list(REMOVE_ITEM keywords NODISPLAY NOGRAPHICS)
  list(APPEND keywords XDISPLAY)
endif()

# QM/MM options begin: QUANTUM and QCHEM are default ON

if(quantum)
  list(APPEND keywords QUANTUM)
endif(quantum)

if(qchem)
  list(APPEND keywords QCHEM) 
endif(qchem)

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
