#include <iostream>
#include <string>
#include <OpenMM.h>

int main(int argc, char ** argv) {
	std::cout << OpenMM::Platform::getOpenMMVersion()
		  << std::endl;
        return 0;
}
