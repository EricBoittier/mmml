#!/bin/bash -x
 
#
# Mark Williamson Mar 2010
#
# A script to carry out a clean serial build of charmm c36a1x with a specific prefs.dat file.
# This list was decided upon at the 11th March 2010 meeting at Ann Arbor and aims to
# form the foundation for serial profiling.
#

# Move upwards
cd ../..

#Clean up any old builds of charmm
rm -Rf ./{build,lib}/em64t

# Start the initial build phase
./install.com em64t huge ifort 2 nolog

# Create an official Mike C blessed pref.dat
cp ./tool/profiling/pref.dat ./build/em64t/pref.dat

# Now build fully
./install.com em64t huge ifort nolog keepf

## Debug option ensures a quicker build
#./install.com em64t huge ifort nolog keepf debug
