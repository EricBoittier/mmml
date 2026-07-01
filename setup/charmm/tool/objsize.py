#!/usr/bin/python
# CHARMM output file size analyzer, by Tim Miller
# USAGE:
# tool/objsize.py [-b|-d|-t] [-v] arch
# -b: sorts by bss
# -d: sorts by data
# -t: sorts by text

import sys, commands, getopt, os, glob
import tempfile

def neatify(fname):
    return fname.split('/')[-1]

# returns a list of tuples, tuples are of the form
# (lib,object,text,data,bss)
def getSizes(libdir,vmode = False):
    startdir = os.getcwd()
    try:
        tempdir = tempfile.mkdtemp()
        os.chdir(tempdir)
    except:
        sys.stderr.write("Cannnot create temp directory!\n")
        sys.exit(-1)
    allos = glob.glob(libdir + "/*.o")
    allas = glob.glob(libdir + "/*.a")

    rarr = []
    for fname in allos:
        # get only the second line (w/ the values)
        status, out = commands.getstatusoutput("size %s | grep -v text" % fname)
        if status != 0:
            sys.stderr.write("WARNING> cannot get sizes for %s.\n" % fname)
        outa = out.split()
        neatfname = neatify(fname)
        if vmode:
            sys.stderr.write("DEBUG> %s text = %s, data %s, bss = %s\n" % (neatfname,outa[0],outa[1],outa[2]))
        rarr.append(("",neatfname,int(outa[0]),int(outa[1]),int(outa[2])))
    for fname in allas:
        status, out = commands.getstatusoutput("size %s | grep -v text" % fname)
        if status != 0:
            sys.stderr.write("WARNING> cannot get sizes for %s.\n" % fname)
        files = out.split('\n')
        neatlname = neatify(fname)
        for f in files:
            if f:
                # print "LINE: " + f
                outa = f.split()
                rarr.append((neatlname,outa[5],int(outa[0]),int(outa[1]),int(outa[2])))        

    os.chdir(startdir)
    os.system("rm -rf %s" % tempdir)
    return rarr

# main subroutine
if __name__ == '__main__':
    vmode = False
    bsort = False
    dsort = False
    tsort = False

    # command line processing
    opts, args = getopt.getopt(sys.argv[1:], 'bdtv')
    for o, a in opts:
        if o == '-v':
            vmode = True
        if o == '-b':
            bsort = True
        if o == '-d':
            dsort = True
        if o == '-t':
            tsort = True
    if len(args) != 1:
        sys.stderr.write("USAGE> tool/objsize.py [-b|-d|-t] [-v] arch\n")
        sys.exit(-1)
    tdir = os.getcwd() + "/lib/" + args[0]
    try:
        os.stat(tdir)
    except:
        sys.stderr.write("ERROR> directory %s does not exist.\n" % tdir)
        sys.exit(-1)

    if vmode:
        sys.stderr.write("DEBUG> using directory %s.\n" % tdir)        
    results = getSizes(tdir,vmode)
    if bsort:
        results.sort(lambda x,y:cmp(x[4],y[4]))
    elif dsort:
        results.sort(lambda x,y:cmp(x[3],y[3]))
    elif tsort:
        results.sort(lambda x,y:cmp(x[2],y[2]))

    print "Library              Object                Text            Data             BSS"
    print "-------              ------                ----            ----             ---"
    for r in results:
        print "%-12s\t%-15s\t%10s\t%10s\t%10s" % r
