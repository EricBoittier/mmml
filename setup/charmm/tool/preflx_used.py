#!/usr/bin/env python

import os, sys, glob, re, string

debug = 0
if len(sys.argv) > 1 and sys.argv[1] == '-d':
   debug = 1

try:
   os.chdir('source')
except:
   sys.stderr.write("PREFLX_USED> Could not change to source subdirectory!\n")
   sys.stderr.write("PREFLX_USED> Make sure you run this from the top level CHARMM directory.\n")
   sys.exit(-1)

# make a list of all the files to read
file_list = glob.glob("*/*.src") + glob.glob("*/*.fcm")

# go through and get any line with a ## anywhere on it excepting certain keywords.
keylines = []
blist = re.compile("##(INCLUDE|ELSE|ERROR|ENDIF|EXFIN|USE)", re.IGNORECASE)
jpar = re.compile("\(.*?\)")
if debug:
   dfp = open("../TIM-dbg.dat2", "w+")
for f in file_list:
   try:
      fp = open(f, 'r')
   except:
      sys.stderr.write("PREFLX_USED> WARNING, could not open file %s.\n" % f)
      continue
   for line in fp:
      line = line.strip()

      # strip out comments because I hate them 
      if line.startswith("c") or line.startswith("C") or line.startswith("!"):
         continue

      # check and see if there is a ## or !##
      if line.startswith('##'):
         if not blist.search(line):
            # parens seem to screw things up .. let's remove them
            line = jpar.sub("", line)
            keylines.append(line)
            if debug:
               dfp.write("%s\n" % line)
      elif '!##' in line:
         x = line.split('!##')

         # convert each keyword to if
         for kw in x[1:]:
            kw = kw.strip()
            pppart = "##IF " + kw
            keylines.append(pppart)
            if debug:
               dfp.write("%s\n" % pppart)

   fp.close()

if debug:
   dfp.close() 

# If the line has anything before the ##, remove it. Then remove non-keyword comments
# and afterwards all exclamation points. Finally kill all the spaces and make new lines
# out of them, putting them into the tlines list.
prem = re.compile("^[^#]*")
comm = re.compile("(.*?)![^#]*$")
excl = re.compile("(.*?)!(.*?)")
tlines = []
if debug:
   dfp = open("../TIM-dbg.dat3", "w+")
for i in range(len(keylines)):
   oline = keylines[i]
   keylines[i] = prem.sub("", keylines[i])
   keylines[i] = comm.sub(r'\1', keylines[i])
   keylines[i] = excl.sub(r'\1 \2', keylines[i])
   if keylines[i].startswith("##KEYWORDS"):
      keylines[i] = ""
   for l in string.splitfields(keylines[i]):
      tlines.append(l)
      if debug:
         dfp.write("%s\n" % l)
      if l == "IF":
         sys.stderr.write("WARNING got IF from oline '%s' mod to '%s'\n" % (oline,keylines[i]))
if debug:
   dfp.close()

# Squash out the various keywords that may be left and any line remaining that
# is just parentheses or is just a single character. .
# Hmmm .. is the elseif valid????
squashes = []
for wrd in ['EXPAND','EXEND','ENDEX','SET','ELIF','ELSEIF','ELSE','IF','IFN','PASS']:
   squashes.append(re.compile("##%s" % wrd))
flines = []
if debug:
   dfp = open("../TIM-dbg.dat4", "w+")
for i in range(len(tlines)):
   oline = tlines[i]
   for reg in squashes:
      tlines[i] = reg.sub("", tlines[i])

   # see what remains
   if tlines[i]:
      flines.append(tlines[i])
      if debug:
         dfp.write("%s\n" % tlines[i])
if debug:
   dfp.close()

# now squash out .not., .when., and the ## itself, NB the ## regexp must be
# first in the list or the other two will not work. 
squashes = []
for wrd in ['##','#','.not.','.when.']:
   squashes.append(re.compile("^%s" % wrd))
if debug:
   dfp = open("../TIM-dbg.dat5", "w+")
for i in range(len(flines)):
   for reg in squashes:
      flines[i] = reg.sub("", flines[i])
   if debug:
      dfp.write("%s\n" % flines[i])
if debug:
   dfp.close()

flines.sort()
last = flines[-1]
for i in range(len(flines)-2, -1, -1):
   if last == flines[i]:
      del flines[i]
   else:
      last = flines[i]
for w in flines: 
   if len(w) > 1: print w


