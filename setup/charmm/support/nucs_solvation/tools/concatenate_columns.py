#!/usr/bin/python

## concatenates columns from charge reparametrisation pairwise interaction
## energy calculations

import string

print 'file with E^solv_IJ in 7th column'
Esolvfile = raw_input(":>")

print 'file with E^vac_IJ in 6th column and E^shield_IJ in 7th column'
Eclbfile = raw_input(":>")

print 'output file'
outfile = raw_input(":>")

print Esolvfile, Eclbfile, outfile

file1 = open(Esolvfile, 'r')
file2 = open(Eclbfile, 'r')
file3 = open(outfile, 'w')

line1 = file1.readline()
line1 = file1.readline()

line2 = file2.readline()
line2 = file2.readline()

file3.write('# COUNTER GROUP1    GROUP2        E^solv          E^vac        E^shield\n')

ii = 0

while 1:
    ii = ii+1
    line1 = file1.readline()
    line2 = file2.readline()
    if not line1:
        break
    if not line2:
        break
    Esolv = string.split(line1)
    Eclb = string.split(line2)
    file3.write('%6d%5s%5s%5s%5s%15s%15s%15s\n' %(ii, Esolv[1], Esolv[2], Esolv[4], Esolv[5], Esolv[6], Eclb[5], Eclb[6]))
file1.close()
file2.close()
file3.close()

