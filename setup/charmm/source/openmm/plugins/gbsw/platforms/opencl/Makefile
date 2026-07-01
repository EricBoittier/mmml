.PHONY: all clean

all: $(chmlib)/openmm_plugins/libOpenMMGBSWOpenCL.so

SRC = $(chmsrc)/openmm/plugins/gbsw

CXX = $(omm_cc)
CXXFLAGS = -std=c++11 -g -O2 -fPIC -I"$(OPENMM_DIR)/include" -I"$(CUDATK)/include" \
           -I"$(SRC)/openmmapi" -I"$(chmbuild)" -I"$(SRC)/platforms/opencl"
LDFLAGS = -L"$(chmlib)/openmm_plugins" -lOpenMMGBSW \
          -Wl,-rpath,"$(chmlib)/openmm_plugins" \
					-L"$(OPENMM_DIR)/lib" -lOpenMM \
          -Wl,-rpath,"$(OPENMM_DIR)/lib" \
					-L"$(OPENMM_DIR)/lib/plugins" -lOpenMMOpenCL \
          -Wl,-rpath,"$(OPENMM_DIR)/lib/plugins" \
					$(MAC_OPENCL)

objects = $(chmbuild)/OpenCLGBSWKernelFactory.o \
          $(chmbuild)/OpenCLGBSWKernels.o \
          $(chmbuild)/OpenCLGBSWKernelSources.o

$(chmlib)/openmm_plugins/libOpenMMGBSWOpenCL.so : $(objects) $(chmlib)/openmm_plugins/libOpenMMGBSW.so
	$(CXX) -shared $(LDFLAGS) $(objects) -o $@

$(chmbuild)/OpenCLGBSWKernelFactory.o : OpenCLGBSWKernelFactory.cpp OpenCLGBSWKernelFactory.h OpenCLGBSWKernels.h
	$(CXX) -c $(CXXFLAGS) OpenCLGBSWKernelFactory.cpp -o $@

$(chmbuild)/OpenCLGBSWKernels.o : OpenCLGBSWKernels.cpp OpenCLGBSWKernels.h $(chmbuild)/OpenCLGBSWKernelSources.h
	$(CXX) -c $(CXXFLAGS) OpenCLGBSWKernels.cpp -o $@

$(chmbuild)/OpenCLGBSWKernelSources.o : $(chmbuild)/OpenCLGBSWKernelSources.cpp $(chmbuild)/OpenCLGBSWKernelSources.h
	$(CXX) -c $(CXXFLAGS) $(chmbuild)/OpenCLGBSWKernelSources.cpp -o $@

$(chmbuild)/OpenCLGBSWKernelSources.h : OpenCLGBSWKernelSources.h.in kernels/gbsw.cl kernels/gbsw2.cl kernels/gbswReductions.cl kernels/gbsw_cpu.cl
	cp OpenCLGBSWKernelSources.h.in "$@.in"
	head -19 "$@.in" > "$@.tmp"
	ls kernels/*.cl | sed -e 's/kernels\/\(.*\)\.cl/\1/' \
            | awk '{print "\tstatic const std::string "$$0";"}' \
            >> "$@.tmp"
	tail -5 "$@.in" >> "$@.tmp"
	mv "$@.tmp" "$@"

$(chmbuild)/OpenCLGBSWKernelSources.cpp : OpenCLGBSWKernelSources.cpp.in $(chmbuild)/OpenCLGBSWKernelSources.h kernels/gbsw.cl kernels/gbsw2.cl kernels/gbswReductions.cl kernels/gbsw_cpu.cl
	cp OpenCLGBSWKernelSources.cpp.in "$@.in"
	cat "$@.in" > "$@.tmp"
	for filename in $$(ls kernels/*.cl); do \
            fn=$$(basename "$$filename" | cut -f1 -d.); \
            echo "const string OpenCLGBSWKernelSources::$$fn = "; \
            sed -e 's/"/\\"/g;s/\\n/\\\\n/g' $$filename | \
                awk '{print "\42"$$0"\\n\42"}'; \
            echo ';'; \
        done >> "$@.tmp"
	mv "$@.tmp" "$@"
