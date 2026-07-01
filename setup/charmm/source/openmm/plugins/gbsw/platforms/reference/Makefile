.PHONY: all clean

all: $(chmlib)/openmm_plugins/libOpenMMGBSWReference.so

SRC = $(chmsrc)/openmm/plugins/gbsw

CXX = $(omm_cc)
CXXFLAGS = -g -O2 -fPIC -I"$(OPENMM_DIR)/include" -I'$(SRC)/openmmapi'
LDFLAGS = -L"$(chmlib)/openmm_plugins" -lOpenMMGBSW \
          -Wl,-rpath,"$(chmlib)/openmm_plugins" \
					-L"$(OPENMM_DIR)/lib" -lOpenMM \
          -Wl,-rpath,"$(OPENMM_DIR)/lib"

objects = $(chmbuild)/GBSWParameters.o \
          $(chmbuild)/ReferenceGBSW.o \
          $(chmbuild)/ReferenceGBSWKernelFactory.o \
          $(chmbuild)/ReferenceGBSWKernels.o

$(chmlib)/openmm_plugins/libOpenMMGBSWReference.so : $(objects) $(chmlib)/openmm_plugins/libOpenMMGBSW.so
	$(CXX) -shared $(LDFLAGS) $(objects) -o $@

$(chmbuild)/GBSWParameters.o : GBSWParameters.cpp GBSWParameters.h
	$(CXX) -c $(CXXFLAGS) GBSWParameters.cpp -o $@

$(chmbuild)/ReferenceGBSW.o : ReferenceGBSW.cpp ReferenceGBSW.h GBSWParameters.h
	$(CXX) -c $(CXXFLAGS) ReferenceGBSW.cpp -o $@

$(chmbuild)/ReferenceGBSWKernelFactory.o : ReferenceGBSWKernelFactory.cpp ReferenceGBSWKernelFactory.h ReferenceGBSWKernels.h
	$(CXX) -c $(CXXFLAGS) ReferenceGBSWKernelFactory.cpp -o $@

$(chmbuild)/ReferenceGBSWKernels.o : ReferenceGBSWKernels.cpp ReferenceGBSWKernels.h ReferenceGBSW.h
	$(CXX) -c $(CXXFLAGS) ReferenceGBSWKernels.cpp -o $@
