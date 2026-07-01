.PHONY: all clean

all: $(chmlib)/openmm_plugins/libOpenMMGBSWCPU.so

SRC = $(chmsrc)/openmm/plugins/gbsw

CXX = $(omm_cc)
CXXFLAGS = -std=c++11 -g -O2 -fPIC -msse4.1 \
					 -I"$(OPENMM_DIR)/include" \
					 -I"$(OPENMM_DIR)/include/openmm/reference" \
					 -I"$(SRC)/openmmapi"
LDFLAGS = -L"$(chmlib)/openmm_plugins" -lOpenMMGBSW \
          -Wl,-rpath,"$(chmlib)/openmm_plugins" \
					-L"$(OPENMM_DIR)/lib" -lOpenMM \
          -Wl,-rpath,"$(OPENMM_DIR)/lib" \
					-L"$(OPENMM_DIR)/lib/plugins" -lOpenMMCPU \
          -Wl,-rpath,"$(OPENMM_DIR)/lib/plugins" \
					-lpthread

objects = $(chmbuild)/CpuGBSWForce.o \
					$(chmbuild)/CpuGBSWKernelFactory.o \
					$(chmbuild)/CpuGBSWKernels.o

$(chmlib)/openmm_plugins/libOpenMMGBSWCPU.so : $(objects) $(chmlib)/openmm_plugins/libOpenMMGBSW.so
	$(CXX) -shared $(LDFLAGS) $(objects) -o $@

$(chmbuild)/CpuGBSWForce.o : CpuGBSWForce.cpp CpuGBSWForce.h gmx_atomic.h
	$(CXX) -c $(CXXFLAGS) CpuGBSWForce.cpp -o $@

$(chmbuild)/CpuGBSWKernelFactory.o : CpuGBSWKernelFactory.cpp CpuGBSWKernelFactory.h CpuGBSWKernels.h
	$(CXX) -c $(CXXFLAGS) CpuGBSWKernelFactory.cpp -o $@

$(chmbuild)/CpuGBSWKernels.o : CpuGBSWKernels.cpp CpuGBSWKernels.h CpuGBSWForce.h
	$(CXX) -c $(CXXFLAGS) CpuGBSWKernels.cpp -o $@

