.PHONY: all clean

SRC = $(chmsrc)/openmm/plugins/gbsw

CXX = $(omm_cc)
CXXFLAGS = -g -O2 -fPIC -I"$(OPENMM_DIR)/include" -I"$(SRC)/openmmapi" -I"$(SRC)/wrappers"
LDFLAGS = -L"$(OPENMM_DIR)/lib" -lOpenMM \
          -Wl,-rpath,"$(OPENMM_DIR)/lib"

objects = $(chmbuild)/GBSWForceProxy.o \
	$(chmbuild)/GBSWForceImpl.o \
	$(chmbuild)/GBSWForce.o \
	$(chmbuild)/OpenMMGBSWCWrapper.o \
	$(chmbuild)/OpenMMGBSWFortranWrapper.o

all: $(chmlib)/openmm_plugins/libOpenMMGBSW.so $(chmbuild)/OpenMMGBSWFortranModule.f90

$(chmlib)/openmm_plugins/libOpenMMGBSW.so : $(objects)
	$(CXX) -shared $(LDFLAGS) $^ -o $@

$(chmbuild)/GBSWForceProxy.o : GBSWForceProxy.h GBSWForce.h $(SRC)/openmmapi/GBSWForceProxy.cpp
	$(CXX) -c $(CXXFLAGS) GBSWForceProxy.cpp -o $@

$(chmbuild)/GBSWForceImpl.o : GBSWForceImpl.h GBSWKernels.h GBSWForce.h $(SRC)/openmmapi/GBSWForceImpl.cpp
	$(CXX) -c $(CXXFLAGS) GBSWForceImpl.cpp -o $@

$(chmbuild)/GBSWForce.o : GBSWForce.h GBSWForceImpl.h $(SRC)/openmmapi/GBSWForce.cpp
	$(CXX) -c $(CXXFLAGS) GBSWForce.cpp -o $@

$(chmbuild)/OpenMMGBSWCWrapper.o: $(SRC)/wrappers/OpenMMGBSWCWrapper.h $(SRC)/wrappers/OpenMMGBSWCWrapper.cpp
	$(CXX) -c $(CXXFLAGS) $(SRC)/wrappers/OpenMMGBSWCWrapper.cpp -o $@

$(chmbuild)/OpenMMGBSWFortranWrapper.o: $(SRC)/wrappers/OpenMMGBSWCWrapper.h $(SRC)/wrappers/OpenMMGBSWFortranWrapper.cpp
	$(CXX) -c $(CXXFLAGS) $(SRC)/wrappers/OpenMMGBSWFortranWrapper.cpp -o $@

$(chmbuild)/OpenMMGBSWFortranModule.f90: $(SRC)/wrappers/OpenMMGBSWFortranModule.f90 $(SRC)/wrappers/OpenMMGBSWCWrapper.h
	cp $(SRC)/wrappers/OpenMMGBSWFortranModule.f90 "$@"
