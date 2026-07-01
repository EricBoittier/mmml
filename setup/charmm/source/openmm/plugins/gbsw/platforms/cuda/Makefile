.PHONY: all clean

all: $(chmlib)/openmm_plugins/libOpenMMGBSWCUDA.so

SRC = $(chmsrc)/openmm/plugins/gbsw

CXX = $(omm_cc)
CXXFLAGS = -std=c++11 -g -O2 -fPIC -I"$(OPENMM_DIR)/include" -I"$(CUDATK)/include" \
           -I"$(SRC)/openmmapi" -I"$(chmbuild)"
LDFLAGS = -L"$(chmlib)/openmm_plugins" -lOpenMMGBSW \
          -Wl,-rpath,"$(chmlib)/openmm_plugins" \
					-L"$(OPENMM_DIR)/lib" -lOpenMM \
          -Wl,-rpath,"$(OPENMM_DIR)/lib" \
					-L"$(OPENMM_DIR)/lib/plugins" -lOpenMMCUDA \
          -Wl,-rpath,"$(OPENMM_DIR)/lib/plugins" \
					$(MAC_CUDA)

objects = $(chmbuild)/CudaGBSWKernelFactory.o \
          $(chmbuild)/CudaGBSWKernels.o \
          $(chmbuild)/CudaGBSWKernelSources.o

$(chmlib)/openmm_plugins/libOpenMMGBSWCUDA.so : $(objects) $(chmlib)/openmm_plugins/libOpenMMGBSW.so
	$(CXX) -shared $(LDFLAGS) $(objects) -o $@

$(chmbuild)/CudaGBSWKernelFactory.o : CudaGBSWKernelFactory.cpp CudaGBSWKernelFactory.h CudaGBSWKernels.h
	$(CXX) -c $(CXXFLAGS) CudaGBSWKernelFactory.cpp -o $@

$(chmbuild)/CudaGBSWKernels.o : CudaGBSWKernels.cpp CudaGBSWKernels.h $(chmbuild)/CudaGBSWKernelSources.h
	$(CXX) -c $(CXXFLAGS) CudaGBSWKernels.cpp -o $@

$(chmbuild)/CudaGBSWKernelSources.o : $(chmbuild)/CudaGBSWKernelSources.cpp $(chmbuild)/CudaGBSWKernelSources.h
	$(CXX) -c $(CXXFLAGS) $(chmbuild)/CudaGBSWKernelSources.cpp -o $@

$(chmbuild)/CudaGBSWKernelSources.h : CudaGBSWKernelSources.h.in kernels/gbsw1.cu kernels/vectorOps.cu
	cp CudaGBSWKernelSources.h.in "$@.in"
	head -19 "$@.in" > "$@.tmp"
	ls kernels/*.cu | sed -e 's/kernels\/\(.*\)\.cu/\1/' \
            | awk '{print "\tstatic const std::string "$$0";"}' \
            >> "$@.tmp"
	tail -5 "$@.in" >> "$@.tmp"
	mv "$@.tmp" "$@"

$(chmbuild)/CudaGBSWKernelSources.cpp : CudaGBSWKernelSources.cpp.in $(chmbuild)/CudaGBSWKernelSources.h kernels/gbsw1.cu kernels/vectorOps.cu
	cp CudaGBSWKernelSources.cpp.in "$@.in"
	cat "$@.in" > "$@.tmp"
	for filename in $$(ls kernels/*.cu); do \
            fn=$$(basename "$$filename" | cut -f1 -d.); \
            echo "const string CudaGBSWKernelSources::$$fn = "; \
            sed -e 's/"/\\"/g;s/\\n/\\\\n/g' $$filename | \
                awk '{print "\42"$$0"\\n\42"}'; \
            echo ';'; \
        done >> "$@.tmp"
	mv "$@.tmp" "$@"
