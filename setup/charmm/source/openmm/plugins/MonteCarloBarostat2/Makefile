.PHONY: all clean

CXX = $(omm_cc)

base_dir = $(chmsrc)/openmm/plugins/MonteCarloBarostat2

CHARMM_OPENMM_LIB = $(LIB)/openmm_plugins

REF_LIB = $(CHARMM_OPENMM_LIB)/libOpenMMCharmm.so
OPENCL_LIB = $(CHARMM_OPENMM_LIB)/libOpenMMCharmmOpenCL.so

# test CUDATK is a valid directory
# if not, do not build charmm's openmm cuda platform plugin
CUDATK_valid = $(shell ls $(CUDATK))
ifneq ($(CUDATK_valid),)
  CUDA_LIB = $(CHARMM_OPENMM_LIB)/libOpenMMCharmmCUDA.so
else
  CUDA_LIB = 
endif

API_OBJS = $(chmbuild)/MonteCarloBarostat2.o $(chmbuild)/MonteCarloBarostatImpl2.o
REF_OBJS = $(chmbuild)/ReferenceCharmmKernelFactory.o $(chmbuild)/ReferenceCharmmKernels.o $(chmbuild)/ReferenceMonteCarloBarostat2.o
CUDA_OBJS = $(chmbuild)/CudaCharmmKernelFactory.o $(chmbuild)/CudaCharmmKernels.o $(chmbuild)/CudaCharmmKernelSources.o
OPENCL_OBJS = $(chmbuild)/OpenCLCharmmKernelFactory.o $(chmbuild)/OpenCLCharmmKernels.o $(chmbuild)/OpenCLCharmmKernelSources.o
WRAPPER_OBJS = $(chmbuild)/CharmmOpenMMCWrapper.o $(chmbuild)/CharmmOpenMMFortranWrapper.o

REF_INCLUDES = -I"platforms/reference/include" -I"omm_headers"
CUDA_INCLUDES = -I"platforms/cuda/include" -I"omm_headers" -I"$(CUDATK)/include" -I"kernel_src"
OPENCL_INCLUDES = -I"platforms/opencl/include" -I"omm_headers" -I"$(CUDATK)/include" -I"$(CUDATK)/include/CL" -I"kernel_src"

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CXXFLAGS = -std=c++11 -fPIC -g -O2 -I"$(OPENMM_DIR)/include" -I"openmmapi/include" -I"omm_headers" -I"wrappers"
    link_opts = -dynamiclib
    cuda_lib = -L/usr/local/cuda/lib -lcuda
    GPU_LIBS = $(CUDA_LIB)
    objs = $(API_OBJS) $(REF_OBJS) $(WRAPPER_OBJS) $(CUDA_OBJS)
else
    CXXFLAGS = -std=c++11 -fPIC -g -O2 -I"$(OPENMM_DIR)/include" -I"openmmapi/include" -I"omm_headers" -I"wrappers"
    link_opts = -shared
    cuda_lib =
    GPU_LIBS = $(CUDA_LIB) $(OPENCL_LIB)
    objs = $(API_OBJS) $(REF_OBJS) $(WRAPPER_OBJS) $(CUDA_OBJS) $(OPENCL_OBJS)
endif

XML_LIB = $(CHARMM_OPENMM_LIB)/libOpenMMCharmmSerialization.so
XML_OBJS = $(chmbuild)/MonteCarloBarostat2Proxy.o $(chmbuild)/MonteCarloBarostat2SerializationProxyRegistration.o

all: $(REF_LIB) $(XML_LIB) $(GPU_LIBS)

clean:
	-rm $(objs) $(XML_OBJS)
	-rm $(REF_LIB) $(XML_LIB) $(GPU_LIBS) $(WRAPPER_OBJS)

$(REF_LIB): $(API_OBJS) $(REF_OBJS) $(WRAPPER_OBJS)
	$(CXX) $(link_opts) -o $@ $^ -L $(CHARMM_OPENMM_LIB) -L $(OPENMM_DIR)/lib -lOpenMM$(OMMD)

$(XML_LIB): $(REF_LIB) $(XML_OBJS)
	$(CXX) $(link_opts) -L"$(OPENMM_DIR)/lib" -lOpenMM$(OMMD) -L"$(CHARMM_OPENMM_LIB)" -lOpenMMCharmm -o $@ $(XML_OBJS)

$(chmbuild)/MonteCarloBarostat2Proxy.o: serialization/MonteCarloBarostat2Proxy.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<
$(chmbuild)/MonteCarloBarostat2SerializationProxyRegistration.o: serialization/MonteCarloBarostat2SerializationProxyRegistration.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

ifeq ($(UNAME_S),Darwin)
$(CUDA_LIB): $(REF_LIB) $(CUDA_OBJS)
	$(CXX) $(link_opts) -o $@ $^ -framework CUDA -L $(OPENMM_DIR)/lib -L $(OPENMM_DIR)/lib/plugins -lOpenMM$(OMMD) -lOpenMMCUDA$(OMMD) -L $(CHARMM_OPENMM_LIB) -lOpenMMCharmm

$(OPENCL_LIB): $(REF_LIB) $(OPENCL_OBJS)
	$(CXX) $(link_opts) -o $@ $^ -framework OpenCL -L $(OPENMM_DIR)/lib -L $(OPENMM_DIR)/lib/plugins -lOpenMM$(OMMD) -lOpenMMOpenCL$(OMMD) -L $(CHARMM_OPENMM_LIB) -lOpenMMCharmm
else
$(CUDA_LIB): $(REF_LIB) $(CUDA_OBJS)
	$(CXX) $(link_opts) -o $@ $(CUDA_OBJS) \
		-L"$(OPENMM_DIR)/lib" -lOpenMM$(OMMD) \
		-Wl,-rpath,"$(OPENMM_DIR)/lib" \
		-L"$(OPENMM_DIR)/lib/plugins" -lOpenMMCUDA$(OMMD) \
		-Wl,-rpath,"$(OPENMM_DIR)/lib/plugins" \
		-L"$(CHARMM_OPENMM_LIB)" -lOpenMMCharmm \
		-Wl,-rpath,"$(CHARMM_OPENMM_LIB)" \
		-L"$(CUDATK)/lib" $(cuda_lib) \
		-Wl,-rpath,"$(CUDATK)/lib"

$(OPENCL_LIB): $(REF_LIB) $(OPENCL_OBJS)
	$(CXX) $(link_opts) -o $@ $(OPENCL_OBJS) -L $(OPENMM_DIR)/lib -L $(OPENMM_DIR)/lib/plugins -lOpenMM$(OMMD) -lOpenMMOpenCL$(OMMD) -L $(CHARMM_OPENMM_LIB) -lOpenMMCharmm
endif

$(chmbuild)/MonteCarloBarostat2.o: openmmapi/src/MonteCarloBarostat2.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

$(chmbuild)/MonteCarloBarostatImpl2.o: openmmapi/src/MonteCarloBarostatImpl2.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

$(chmbuild)/ReferenceCharmmKernelFactory.o: platforms/reference/src/ReferenceCharmmKernelFactory.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

$(chmbuild)/ReferenceCharmmKernels.o: platforms/reference/src/ReferenceCharmmKernels.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

$(chmbuild)/ReferenceMonteCarloBarostat2.o: platforms/reference/src/ReferenceMonteCarloBarostat2.cpp
	$(CXX) $(CXXFLAGS) $(REF_INCLUDES) -c -o $@ $<

$(chmbuild)/CudaCharmmKernelFactory.o: platforms/cuda/src/CudaCharmmKernelFactory.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -c -o $@ $<

$(chmbuild)/CudaCharmmKernels.o: platforms/cuda/src/CudaCharmmKernels.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -c -o $@ $<

$(chmbuild)/CudaCharmmKernelSources.o: kernel_src/CudaCharmmKernelSources.cpp
	$(CXX) $(CXXFLAGS) $(CUDA_INCLUDES) -c -o $@ $<

$(chmbuild)/OpenCLCharmmKernelFactory.o: platforms/opencl/src/OpenCLCharmmKernelFactory.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDES) -c -o $@ $<

$(chmbuild)/OpenCLCharmmKernels.o: platforms/opencl/src/OpenCLCharmmKernels.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDES) -c -o $@ $<

$(chmbuild)/OpenCLCharmmKernelSources.o: kernel_src/OpenCLCharmmKernelSources.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INCLUDES) -c -o $@ $<

$(chmbuild)/CharmmOpenMMCWrapper.o: wrappers/CharmmOpenMMCWrapper.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(chmbuild)/CharmmOpenMMFortranWrapper.o: wrappers/CharmmOpenMMFortranWrapper.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<
