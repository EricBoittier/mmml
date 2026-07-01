.SUFFIXES: .cu .c .o

SRCFC = forttest.f90
VGG001 = gpuvdwdirect
VGG002 = gpuvdwdirectp
VGG003 = gpuvdwcutoff
VGG004 = gpuvdwcutoffp
VGG005 = gpuvdwcell
VGG006 = gpuvdwcellp
VGG007 = gpuewredirect
VGG008 = gpuewredirectp
VGG009 = gpuewrecutoff
VGG010 = gpuewrecutoffp
VGG011 = gpuewrecell
VGG012 = gpuewrecellp
VGG013 = gpufft3d
VGG014 = gpuvdwdirectdble
VGG015 = gpuvdwcutoffdble
VGG016 = gpuvdwcutoffex
VGG017 = gpuvdwcutoffpex
VGG018 = gpupme
#VGG019 = gpu

SRK001 = $(VGG001:%=%_kernel.cu)
SRM001 = $(VGG001:%=%_mother.cu)
SRC001 = $(VGG001:%=%_mother.c)
OBJ001 = $(VGG001:%=%_mother.o)
SRK002 = $(VGG002:%=%_kernel.cu)
SRM002 = $(VGG002:%=%_mother.cu)
SRC002 = $(VGG002:%=%_mother.c)
OBJ002 = $(VGG002:%=%_mother.o)
SRK003 = $(VGG003:%=%_kernel.cu)
SRM003 = $(VGG003:%=%_mother.cu)
SRC003 = $(VGG003:%=%_mother.c)
OBJ003 = $(VGG003:%=%_mother.o)
SRK004 = $(VGG004:%=%_kernel.cu)
SRM004 = $(VGG004:%=%_mother.cu)
SRC004 = $(VGG004:%=%_mother.c)
OBJ004 = $(VGG004:%=%_mother.o)
SRK005 = $(VGG005:%=%_kernel.cu)
SRM005 = $(VGG005:%=%_mother.cu)
SRC005 = $(VGG005:%=%_mother.c)
OBJ005 = $(VGG005:%=%_mother.o)
SRK006 = $(VGG006:%=%_kernel.cu)
SRM006 = $(VGG006:%=%_mother.cu)
SRC006 = $(VGG006:%=%_mother.c)
OBJ006 = $(VGG006:%=%_mother.o)
SRK007 = $(VGG007:%=%_kernel.cu)
SRM007 = $(VGG007:%=%_mother.cu)
SRC007 = $(VGG007:%=%_mother.c)
OBJ007 = $(VGG007:%=%_mother.o)
SRK008 = $(VGG008:%=%_kernel.cu)
SRM008 = $(VGG008:%=%_mother.cu)
SRC008 = $(VGG008:%=%_mother.c)
OBJ008 = $(VGG008:%=%_mother.o)
SRK009 = $(VGG009:%=%_kernel.cu)
SRM009 = $(VGG009:%=%_mother.cu)
SRC009 = $(VGG009:%=%_mother.c)
OBJ009 = $(VGG009:%=%_mother.o)
SRK010 = $(VGG010:%=%_kernel.cu)
SRM010 = $(VGG010:%=%_mother.cu)
SRC010 = $(VGG010:%=%_mother.c)
OBJ010 = $(VGG010:%=%_mother.o)
SRK011 = $(VGG011:%=%_kernel.cu)
SRM011 = $(VGG011:%=%_mother.cu)
SRC011 = $(VGG011:%=%_mother.c)
OBJ011 = $(VGG011:%=%_mother.o)
SRK012 = $(VGG012:%=%_kernel.cu)
SRM012 = $(VGG012:%=%_mother.cu)
SRC012 = $(VGG012:%=%_mother.c)
OBJ012 = $(VGG012:%=%_mother.o)
SRK013 = $(VGG013:%=%_kernel.cu)
SRM013 = $(VGG013:%=%_mother.cu)
SRC013 = $(VGG013:%=%_mother.c)
OBJ013 = $(VGG013:%=%_mother.o)
SRK014 = $(VGG014:%=%_kernel.cu)
SRM014 = $(VGG014:%=%_mother.cu)
SRC014 = $(VGG014:%=%_mother.c)
OBJ014 = $(VGG014:%=%_mother.o)
SRK015 = $(VGG015:%=%_kernel.cu)
SRM015 = $(VGG015:%=%_mother.cu)
SRC015 = $(VGG015:%=%_mother.c)
OBJ015 = $(VGG015:%=%_mother.o)
SRK016 = $(VGG016:%=%_kernel.cu)
SRM016 = $(VGG016:%=%_mother.cu)
SRC016 = $(VGG016:%=%_mother.c)
OBJ016 = $(VGG016:%=%_mother.o)
SRK017 = $(VGG017:%=%_kernel.cu)
SRM017 = $(VGG017:%=%_mother.cu)
SRC017 = $(VGG017:%=%_mother.c)
OBJ017 = $(VGG017:%=%_mother.o)
SRK018 = $(VGG018:%=%_kernel.cu)
SRM018 = $(VGG018:%=%_mother.cu)
SRC018 = $(VGG018:%=%_mother.c)
OBJ018 = $(VGG018:%=%_mother.o)
SRK019 = $(VGG019:%=%_kernel.cu)
SRM019 = $(VGG019:%=%_mother.cu)
SRC019 = $(VGG019:%=%_mother.c)
OBJ019 = $(VGG019:%=%_mother.o)

#CC     = gcc -g
CC     = g++
#CC     = icpc -xW 
#CC     = gcc -DGPU
FC     = gfortran
#FC     = ifort
#ICC    = icpc -openmp 
OPTFLAGS = -O
#OPTFLAGS = -g
CFLAGS = $(OPTFLAGS)

TARGET             = gputest
HOMEDIR            = /home/sakamaki
LIBMD3WITHG80      = libmd3withg80s.a
#CUDA_INSTALL_PATH  =/nfs/home/narumi/VirtualGRAPE/GPU/cuda_2.0b/
CUDA_INSTALL_PATH  =$(CUDATK)/
#CUDA_SDK_PATH      =$(CUDA_INSTALL_PATH)/sdk/C
#CUDA_INSTALL_PATH  =/opt/cuda/
#CUDA_SDK_PATH      =$(CUDA_INSTALL_PATH)/sdk/C
#CUDA_SDK_PATH      =/usr/local/cuda/NVIDIA_CUDA_SDK
#CUDA_SDK_PATH      =$(HOMEDIR)/NVIDIA_CUDA_SDK_1.1
LIBGPU             = $(LIBMD3WITHG80) \
	            -L$(CUDA_INSTALL_PATH)/lib64 -lcuda -lcudart -lGL -lGLU -lm -lcufft
#NVCC               = $(CUDA_INSTALL_PATH)bin/nvcc -O
NVCC               = $(CUDA_INSTALL_PATH)bin/nvcc $(OPTFLAGS)
# matrixMul and THD=100 added by Milan ???
# for debug
#CUDAINCLUDES       = -I. -I$(CUDA_INSTALL_PATH)include -I$(CUDA_SDK_PATH)/common/inc -I$(CUDA_SDK_PATH)/src/matrixMul -DTHD=100
CUDAINCLUDES       = -I. -I$(CUDA_INSTALL_PATH)include -DTHD=100
LIB_COPY_DIR       = $(HOMEDIR)/lib/

#.cu.o:
.cu.c:
#	$(NVCC) -cubin $(CUDAINCLUDES) $< -o $@
#	$(NVCC) -ptx $(CUDAINCLUDES) $< -o $@
#	$(NVCC) -cuda -arch sm_11 $(CUDAINCLUDES) $< -o $@
#	$(NVCC) --compile -arch sm_11 $(CUDAINCLUDES) $<
#	$(NVCC) --compile -arch sm_12 -deviceemu $(CUDAINCLUDES) $<
#	$(NVCC) -cuda -deviceemu $(CUDAINCLUDES) $< -o $@
#	$(NVCC) --compile -arch sm_12 --ptxas-options=-v $(CUDAINCLUDES) $<
	$(NVCC) --host-compilation C -cuda -arch sm_12 --ptxas-options=-v $(CUDAINCLUDES) -o $@ $<
#	$(NVCC) --host-compilation C -cuda -arch sm_20 --ptxas-options=-v $(CUDAINCLUDES) -o $@ $<
.c.o:
	$(CC) -c $(CFLAGS)  $<
#	$(CC) -c -g $(CFLAGS) $<

$(TARGET): $(SRCFC) $(LIBMD3WITHG80)
	$(FC) $(FFLAGS) $< $(LIBGPU) -o $@

$(LIBMD3WITHG80) : \
$(OBJ001) $(OBJ002) $(OBJ003) $(OBJ004) $(OBJ005) $(OBJ006) $(OBJ007) $(OBJ008) $(OBJ009) $(OBJ010) \
$(OBJ011) $(OBJ012) $(OBJ013) $(OBJ014) $(OBJ015) $(OBJ016) $(OBJ017) $(OBJ018) $(OBJ019) $(OBJ020) \
$(OBJ021) $(OBJ022) $(OBJ023) $(OBJ024) $(OBJ025) $(OBJ026) $(OBJ027) $(OBJ028) $(OBJ029) $(OBJ030) 
	ar ruv $@ $?
	ranlib $@
#	cp $@ $(LIB_COPY_DIR)
$(OBJ001) : $(SRM001) $(SRK001)
$(OBJ002) : $(SRM002) $(SRK002)
$(OBJ003) : $(SRM003) $(SRK003)
$(OBJ004) : $(SRM004) $(SRK004)
$(OBJ005) : $(SRM005) $(SRK005)
$(OBJ006) : $(SRM006) $(SRK006)
$(OBJ007) : $(SRM007) $(SRK007)
$(OBJ008) : $(SRM008) $(SRK008)
$(OBJ009) : $(SRM009) $(SRK009)
$(OBJ010) : $(SRM010) $(SRK010)
$(OBJ011) : $(SRM011) $(SRK011)
$(OBJ012) : $(SRM012) $(SRK012)
$(OBJ013) : $(SRM013) $(SRK013)
$(OBJ014) : $(SRM014) $(SRK014)
$(OBJ015) : $(SRM015) $(SRK015)
$(OBJ016) : $(SRM016) $(SRK016)
$(OBJ017) : $(SRM017) $(SRK017)
$(OBJ018) : $(SRM018) $(SRK018)
$(OBJ019) : $(SRM019) $(SRK019)

all      : $(TARGET)

dist :
	tar cvfz vggpu.tar.gz Makefile Makefile_intel *.[ch] *.cu *.f90

clean    :
	-rm -f $(TARGET)
	-rm -f *.o $(LIBMD3WITHG80) *.c
