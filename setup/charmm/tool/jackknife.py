#!/usr/bin/env python

"""
CHARMM jack-knife testing script
by Tim Miller May, 2011
thanks to Bernard Brooks and Mike Garrahan for implementation suggestions

Input script example:
--- begin ---
basefile /v/bigbox12/home/tim/charmm/c36a6-pref-full
testone -ACE +AMBER
testtwo --TSALLIS +AMBER
testthr -SGLD -ADUMB -PERT -BLOCK
--- end ---

Comments on the input script:

- The script starts with a "basefile" or "baseline" command, which specifies
  a file with the initial set of words to use (or the words themselves).

- Comments can be specified with a line starting with "!" or "#".

- Each non-blank, non-comment line that does not start with "basefile" or
  "baseline" specifies a test. The first word is the name of the test, followed
  by a specification of which key words are to be added to the base set (+WORD)
  or deleted from it (-WORD).

- A word that starts with a double plus "++" is added to the given test and
  ALL FUTURE TESTS until the next "basefile" or "baseline" command. Likewise,
  words starting with a double minus "--" are deleted from the base set until
  the next "basefile" or "baseline" command.

- Names of tests must be unique within the input script.
"""

import os, sys, shutil, stat
import copy, subprocess, time

class ParseError(Exception):
    def __str__(self):
        return "Line %d: %s" % (self.lineno,self.problem)

    def __init__(self,lineno,problem):
        self.lineno  = lineno
        self.problem = problem

class ExecError(Exception):
    pass

class commandFile:
    """An iterator that processes the command file and returns the
       next set of key-words
    """
    lineno = 0

    def check_words(self):
        for w in self.basewords:
            if w.upper() == 'END':
                return False
        return True 
    
    def next(self):
        # encase in a big while loop for cycling convenience
        while True:
            line = self.fp.readline()
            if not line:
                raise StopIteration
            self.lineno += 1
            line = line.strip()

            # if comments, get the next line
            if not line:
                continue
            if line.startswith('#') or line.startswith('!'):
                continue

            # process control key-words
            if line.startswith('baseline'):
                self.basewords = line.split()[1:]
                if not self.check_words():
                    raise ParseError(self.lineno,"Bad word in baseline statement")
            elif line.startswith('basefile'):
                fp = open(line.split()[1], 'r')
                self.basewords = []
                for l2 in fp:
                    l2 = l2.strip()
                    if l2:
                        self.basewords.append(l2)
                fp.close()
                if not self.check_words():
                    raise ParseError(self.lineno,"Bad word in basefile %s" % line.split()[1])

            # not a control keyword so this is the name of
            # a test
            else:
                larr = line.split()
                name = larr[0]
                wordset = copy.deepcopy(self.basewords)
                for word in larr[1:]:
                    if word.startswith('++'):
                        # add me into the baswords
                        word = word.replace('++','')
                        if not word in wordset:
                            wordset.append(word)
                            self.basewords.append(word)
                    elif word.startswith('--'):
                        word = word.replace('--','')
                        if word in wordset:
                            wordset.remove(word)
                            self.basewords.remove(word)
                    elif word.startswith('+'):
                        word = word.replace('+','')
                        if not word in wordset:
                            wordset.append(word)
                    elif word.startswith('-'):
                        word = word.replace('-','')
                        if word in wordset:
                            wordset.remove(word)
                    else:
                        sys.stderr.write('Line %d, test %s: skipping word %s, does not start with + or -!\n' % (self.lineno,name,word))

                ret = [name]
                ret.extend(wordset)
                return ret
                    

    def __iter__(self):
        return self

    def __init__(self,fp):
        self.fp = fp
        self.basewords = []

class keywordTest:
    """A class that encapsulates a key word test set"""

    def start_compile(self):
        startdir = os.getcwd()

        os.chdir('%s/%s' % (self.workdir,self.name))
        self.myout=open("jackknife.out", "w")
        self.myerr=open("jackknife.err", "w")

        args = ['./install.com', 'gnu']
        if self.parallel:
            args.extend(['M'])
        self.proc = subprocess.Popen(args, stdout=self.myout, stderr=self.myerr)

        os.chdir(startdir)

    def check_compile(self):
        self.proc.poll()
        r = self.proc.returncode
        if r is None:
            return 0 # the compile has not yet finished
        elif r == 0:
            self.myout.flush()
            return 1 # success
        else:
            self.myout.flush()
            return 2 # failure

    def build_pref(self):
        startdir = os.getcwd()
        os.chdir('%s/%s' % (self.workdir,self.name))

        nullfp = open("/dev/null", "w")

        # create the build/<arch> directory using install.com, sending
        # stdout and stderr to the bit-bucket.
        if self.parallel:
            args = ['./install.com', 'gnu', 'M', '2']
        else:
            args = ['./install.com', 'gnu', '2']
        self.proc = subprocess.Popen(args, stdout=nullfp, stderr=nullfp)
        self.proc.wait()
        if self.proc.returncode != 2: # for some asinine reason install.com uses 2 as the return code here
            raise ExecError()
        del self.proc
        nullfp.close()

        # open a new pref.dat
        fp = open('build/gnu/pref.dat', 'w')
        for word in self.words:
            fp.write(word + '\n')
        fp.write('END\n')
        fp.close()

        os.chdir(startdir)

    # oooh-weee a destructor :-)
    def __del__(self):
        if not self.myout is None:
            self.myout.close()
        if not self.myerr is None:
            self.myerr.close()

    def __init__(self,name,workdir,charmmdir,parallel=False):
        self.name = name
        self.workdir = workdir
        self.words = []
        self.proc = None
        self.myout = None
        self.myerr = None
        self.parallel = parallel

        # check and see if the directory already exists,
        # and if so bail
        try:
            sr = os.stat("%s/%s" % (self.workdir, self.name))
        except OSError, e:
            # probably the directory does not exist, this is in fact OK and what we want
            pass
        else:
            os.rename("%s/%s" % (self.workdir, self.name), "%s/%s.bak.%s" % (self.workdir, self.name, time.ctime().replace(' ','_') ))

        try:
            os.mkdir("%s/%s" % (self.workdir, self.name))
        except:
            sys.stderr.write("Directory %s/%s cannot be created!\n" % (self.workdir, self.name))
            return False

        # copy the build directory and symlink the goodies required to build CHARMM
        os.mkdir("%s/%s/build" % (self.workdir, self.name))
        shutil.copy("%s/install.com" % charmmdir, "%s/%s/install.com" % (self.workdir,self.name))
        shutil.copytree("%s/build/UNX" % charmmdir, "%s/%s/build/UNX" % (self.workdir,self.name))
        os.symlink("%s/source" % charmmdir, "%s/%s/source" % (self.workdir,self.name))
        os.symlink("%s/tool" % charmmdir, "%s/%s/tool" % (self.workdir,self.name))


import time
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option('-v', '--verbose', action='store_true', dest='verbose', \
                      default=False, help='Print debugging information')
    parser.add_option('-c', '--charmmdir', dest='charmmdir', default='.', \
                      metavar='DIRECTORY', help='Root directory of the CHARMM tree')
    parser.add_option('-w', '--workdir', dest='workdir', default='/tmp/buildcharmm', \
                      metavar='DIRECTORY', help='Directory of where to do the compiles')
    parser.add_option('-i', '--input', default='pptest.inp', metavar='FILE', \
                      help='Input control file for the run')
    parser.add_option('-l', '--log', default='stdout', metavar='FILE', \
                      help='Where to log results (stdout for standard output)')
    parser.add_option('-n', '--number', default='1', metavar='INTEGER', \
                      help='Number of test compiles to run simultaneously.')
    parser.add_option('-p', '--parallel', action='store_true', dest='doparallel', \
                      default=False, help='Test on parallel versions of CHARMM')
    (options, args) = parser.parse_args()

    if len(args) > 0:
        sys.stderr.write('WARNING: ignoring un-parsed command line arguments!\n')

    if options.log.lower() != 'stdout':
        logfp = open(options.log, 'w')
    else:
        logfp = sys.stdout

    fp = open(options.input, 'r')
    if not fp:
        sys.stderr.write('ERROR: cannot open input file %s!\n' % input)
        sys.exit(-1)

    try:
        sr = os.stat(options.workdir)
    except OSError, e1:
        sys.stderr.write("WARNING: Cannot access workdir %s ... attempting to create it.\n" % options.workdir)
        try:
            os.mkdir(options.workdir)
        except OSError, e2:
            sys.stderr.write("ERROR: Cannot create working directory %s.\n" % options.workdir)
            sys.exit(-2)
    else:
        if not stat.S_ISDIR(sr.st_mode):
            sys.stderr.write("ERROR: Given working path (%s) is NOT a directory.\n" % options.workdir)
 
    tstsets = set()
    cf = commandFile(fp)
    for wordset in cf:
        if options.verbose:
            logfp.write('Setting up the test named %s.\n' % wordset[0])
        testSet = keywordTest(wordset[0], options.workdir, options.charmmdir, options.doparallel)
        if not testSet:
            logfp.write('Failed to initialize ... skipping.\n')
            continue
        testSet.words.extend(wordset[1:])
        if options.verbose:
            logfp.write('Building pref.dat\n')
        testSet.build_pref()
        tstsets.add(testSet)
    fp.close()

    # actually run the compiles and enumerate results, this operates on
    # a basic queue principle except a set is not a queue :-).
    runsets  = set()
    maxrun   = int(options.number)
    resultDict = {}
    print "Start, len of runsets = %d" % len(runsets)
    while len(runsets) > 0 or len(tstsets) > 0:
        # check for any results that might have happend, note that
        # the copy is to prevent python from bitching about changing
        # the dictionary mid-iteration
        jobstorm = []
        for job in runsets:
            result = job.check_compile()
            if result == 1:
                if options.verbose:
                    logfp.write('Test %s completes OK!\n' % job.name)
                resultDict[job.name] = 'OK'
                jobstorm.append(job)                           
            elif result == 2:
                if options.verbose:
                    logfp.write('Test %s fails!\n' % job.name)
                resultDict[job.name] = 'FAIL'
                jobstorm.append(job)
        for job in jobstorm:
            runsets.remove(job)
        del jobstorm

        # grab the next test to run if one exists
        if len(tstsets) == 0:
            time.sleep(5) # impose a small delay to avoid constant polling
            continue
        job_to_run = tstsets.pop()
        
        # check that we're allowed to run it
        if len(runsets) >= maxrun:
            if options.verbose:
                logfp.write('Runset is full, sleeping for 30 seconds...\n')
            tstsets.add(job_to_run) # return to the list of jobs to run
            time.sleep(30)
            continue

        # it's ok, dequeue it, add it to runsets, and start the compilation
        runsets.add(job_to_run)
        job_to_run.start_compile()
        if options.verbose:
            logfp.write('Started test compile of %s\n' % job_to_run.name)

    for elt in resultDict.keys():
        logfp.write('Test %s: result %s\n' % (elt,resultDict[elt]))
