#!/bin/sh

cat <<EOF

Please use the configure script and then make or ninja to configure and compile
CHARMM.

For example,
$ ./configure
$ make -C build/cmake install

See './configure --help', doc/cmake.info, or the relevant '.info' file
in the doc directory for more details.

To build a configuration currently not supported by the configure script,
copy the old install.com from the tool directory into the charmm source root
directory and execute it with the necessary arguemnts.

As of August 2021, only CHARMMRATE and GAMESS-UK are known to be unsupported
by the configure script.

See
doc/charmmrate.info
or
doc/gamess-uk.info

EOF
