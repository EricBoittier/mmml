#!/usr/bin/python

## concatenates columns of scaling factors calculated with different
## energy functions

## the scaling factors can be taken from CHARMM coordinate files.
## There, they must be given in the WMAIN column.
## The factors can also be taken from the output of the verification
## calculation. There, they are expected to be in the 7th column.

import string

print 'crd coordinate file with lambda in WMAIN column (1) or output from verification calculation (2)?'
filetype = raw_input(":>")

while filetype != '1' and filetype != '2':
    print 'Please type 1 or 2!'
    filetype = raw_input(":>")

print 'file lambda calculated using CDIEL SWITCH'
cdiel_switch = raw_input(":>")
print 'file lambda calculated using CDIEL SHIFT'
cdiel_shift = raw_input(":>")
print 'file lambda calculated using RDIEL SWITCH'
rdiel_switch = raw_input(":>")
print 'file lambda calculated using RDIEL SHIFT'
rdiel_shift = raw_input(":>")

print 'output file'
outfile = raw_input(":>")


file1 = open(cdiel_switch, 'r')
file2 = open(cdiel_shift, 'r')
file3 = open(rdiel_switch, 'r')
file4 = open(rdiel_shift, 'r')
file5 = open(outfile, 'w')

if filetype == '1':
    file5.write('# ATNUM,RESNUM,RESNAME,ATNAME cdiel_switch   cdiel_shift    rdiel_switch   rdiel_shift\n')
    while 1:
        line1 = file1.readline()
        if line1[0] != '*':
            break
    while 1:
        line2 = file2.readline()
        if line2[0] != '*':
            break
    while 1:
        line3 = file3.readline()
        if line3[0] != '*':
            break
    while 1:
        line4 = file4.readline()
        if line4[0] != '*':
            break
    while 1:
        line1 = file1.readline()
        line2 = file2.readline()
        line3 = file3.readline()
        line4 = file4.readline()
        if not line1 or not line2 or not line3 or not line4:
            break
        atomspec = line1[0:20]
        lambda1 = line1[60:70]
        lambda2 = line2[60:70]
        lambda3 = line3[60:70]
        lambda4 = line4[60:70]
        file5.write('%20s  %15s%15s%15s%15s\n' %(atomspec, lambda1, lambda2, lambda3, lambda4))

elif filetype == '2':
    file5.write('# GROUP              cdiel_switch   cdiel_shift    rdiel_switch   rdiel_shift\n')
    while 1:
        line1 = file1.readline()
        line2 = file2.readline()
        line3 = file3.readline()
        line4 = file4.readline()
        if not line1 or not line2 or not line3 or not line4:
            break
        if line1[1] == 'O':
            line1 = file1.readline()
        if line2[1] == 'O':
            line2 = file2.readline()
        if line3[1] == 'O':
            line3 = file3.readline()
        if line4[1] == 'O':
            line4 = file4.readline()
        lambda1 = string.split(line1)
        lambda2 = string.split(line2)
        lambda3 = string.split(line3)
        lambda4 = string.split(line4)
        file5.write('%5s%5s%5s%15s%15s%15s%15s\n' %(lambda1[0], lambda1[1], lambda1[2], lambda1[6], lambda2[6], lambda3[6], lambda4[6]))



file1.close()
file2.close()
file3.close()
file4.close()
file5.close()

        



