.PHONY: all clean

all:
	$(MAKE_COMMAND) -C openmmapi all
	$(MAKE_COMMAND) -C platforms/cpu all
	$(MAKE_COMMAND) -C platforms/cuda all
	$(MAKE_COMMAND) -C platforms/opencl all
	$(MAKE_COMMAND) -C platforms/reference all

clean: 
	$(MAKE_COMMAND) -C openmmapi clean
	$(MAKE_COMMAND) -C platforms/cpu clean
	$(MAKE_COMMAND) -C platforms/cuda clean
	$(MAKE_COMMAND) -C platforms/opencl clean
	$(MAKE_COMMAND) -C platforms/reference clean
