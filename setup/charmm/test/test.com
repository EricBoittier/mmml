#!/bin/csh
# set echo
# test.com : CHARMM Test Script
#-----------------------------------------------------------------------
# Change Log:
#
# 15-Aug-02  feature checking added with "Test NOT performed." messages
# 15-Aug-97  compare only if benchmark exists
# 31-Jul-95  error output redirected to testcase .out files, for tracebacks
# 09-Feb-94  CHARMM 24 (24a1) and higher version testing script
#            $argv[4] specifies version specific tests, e.g., c22test
# 21-Feb-93  CHARMM 23 (c23f) and later test script
# 01-Jan-92  CHARMM22 release version testing
#            organized into groups and new testcases added
# 05-May-91  CHARMM 22.0.b testing
#            sed filtering added
# 07-Feb-91  CHARMM22 beta testing version
#
# Inquiries to chmgr@tammy.harvard.edu
#-----------------------------------------------------------------------
#
# Arguments
# chm_host = $argv[1], the host machine type
# outdir   = $argv[2], the test output directory name [output ]
# bendir   = $argv[3], the benchmark directory name   [bench  ]
# chm_test = $argv[4], the target test suite          [all    ]
#
#-----------------------------------------------------------------------

#
date
#
phase1:
# command line argument parsing
# set up testing environments
#
set testing = 'all'
#
if ( $#argv == 0 ) then
  echo " "
  echo " Usage: test.com [ M num ] [ X num ][ keeps ] host_machine_type"
  echo "            [ output_dir benchmark_dir testcases ]."
  echo " "
  echo "        M specifies that this is a parallel run requiring mpirun"
  echo "          num  specifies the numnber of processors for parallel run"

  echo "        X specifies that this is a parallel run requiring mpirun/openm"
  echo "          num  specifies the numnber of threads for parallel run"
  echo " "
  echo "        E specifies that this is a test of the ensemble code (only)"
  echo "          four processes will be run"
  echo " "
  echo "        keeps specifies that contents of the scratch directory"
  echo "          are not deleted"
  echo " "
  echo "        host-machine-type is one of the follwoing."
  echo "               altix    for SGI Itanium Altix series (64 bit)"
  echo "               cmake    for installs using cmake instead of install.com"
  echo "               em64t    for Intel compilers on x86_64"
  echo "               gnu      for GNU compilers"
  echo "               osx      for Apple Macintosh computers running OS X"
  echo " "
  echo "        testcases can be one of the following set"
  echo "               nn       for cnntest"
  echo "                        {nn=20,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49}"
  echo "               all      for mmff and all 20-49 tests"
  echo "               mmff     for cmmfftest"
  echo "               nbond    for cnbondtest"
  echo " "
  echo "        The defaults are"
  echo "               output_dir        = output"
  echo "               benchmark_dir     = bench"
  echo "               testcases         = all"
  exit
#
endif
set run_mpi = 0
set run_omp = 0
set ensemble = 0
set keepscr = 0
if ( $1 == "M" ) then
   set run_mpi = 1
   set numpes=$2
   shift
   shift
endif
if ( $1 == "X" ) then
   set run_omp = 1
   set numthreads=$2
    if( $run_mpi == 0 ) then
	set run_mpi = 1
	set numpes = 1
    endif
   shift
   shift
endif
if ( $1 == "E" ) then
   set ensemble = 1
   set numpes=4
   shift
endif

if ( $1 == "keeps" || $1 == "KEEPS" ) then
   set keepscr = 1
   shift
endif

if ( $#argv == 1 ) then
  set chm_host = $argv[1]
#
else if ( $#argv == 2 ) then
  set chm_host = $argv[1]
  set outdir   = $argv[2]
#
else if ( $#argv == 3 ) then
  set chm_host = $argv[1]
  set outdir   = $argv[2]
  set bendir   = $argv[3]
#
else if ( $#argv == 4 ) then
  set chm_host = $argv[1]
  set outdir   = $argv[2]
  set bendir   = $argv[3]
  set testing  = $argv[4]
endif
#
if ( $run_mpi == 1 && $chm_host != 'cmake' ) then
  setenv chmhost ${chm_host}_M
else
  setenv chmhost $chm_host
endif
if ( $?chmroot == 0 ) setenv chmroot `cd ..;pwd`

if ( $?chmexec == 0 && $chmhost == 'cmake' ) then
  setenv chmexec "$chmroot/bin"
else if ( $?chmexec == 0 ) then
  setenv chmexec $chmroot/exec/$chmhost
endif

if ( $?chmtool == 0 ) setenv chmtool $chmroot/tool
if ( $?chmtest == 0 ) setenv chmtest `pwd`
#
if ( $?outdir == 0 ) set outdir = output
if (! -e $outdir   ) mkdir $outdir
#
set cmpr = 1
if ( $?bendir == 0 && -e bench_$chmhost ) set bendir = bench_$chmhost
if ( $?bendir == 0 && -e bench ) set bendir = bench
endif
if ( $?bendir != 0 ) then
  if ( $bendir == 0 ) then
    echo " "
    echo " test.com> No testing against benchmarks."
    set cmpr = 0
  else if ( ! -e $bendir ) then
    echo " "
    echo " test.com> Directory $bendir does not exist."
    echo "           No testing against benchmarks."
    set cmpr = 0
  endif
endif
if ( $?bendir == 0 ) set cmpr = 0
#
if ( $?chm_exec == 0 ) set chm_exec = $chmexec/charmm
if (! -e $chm_exec   ) then
  echo " "
  echo " test.com> the testing target $chm_exec does not exist."
  exit 1
endif
setenv chm_run ""
if ( $chm_host == "ibmaixmp" ) then
#   set chm_exec="poe $chm_exec -procs 4 -rmpool 1 "
   setenv MP_PROCS 4
   setenv MP_HOSTFILE host.file
   setenv MP_SHARED_MEMORY yes
   setenv MP_WAIT_MODE poll
endif
if ( $chm_host == "ibmsp3" ) then
   set chm_exec="poe $chm_exec -procs 4 -rmpool 1 "
   setenv MP_PROCS 4
   setenv MP_HOSTFILE host.file
   setenv MP_SHARED_MEMORY yes
   setenv MP_WAIT_MODE poll
endif

set mpi_out_sep = 0
if ( $run_mpi == 1 || $ensemble == 1 ) then
	echo $run_mpi
        set machinefile
        if ( $?MPI_MACHINEFILE ) then
	   set machinefile = "-machinefile $MPI_MACHINEFILE"
        endif
	if ( $run_omp == 1 ) then
	    set chm_run = "mpirun $machinefile -np $numpes -x OMP_NUM_THREADS=$numthreads --bind-to none"
	else
	    set chm_run = "mpirun $machinefile -np $numpes"
	endif
        if ( $chm_host == alpha ) set chm_run = "d$chm_run"

        # Can we separate output by process rank? (OpenMPI >= 1.3)
        rm xyzzy* >& /dev/null
        mpirun --output-filename xyzzy true >& /dev/null
        if ($status == 0 && (-e xyzzy.0 || -e xyzzy.1.0)) then
           set mpi_out_sep = 1
           rm xyzzy*
        endif
endif

#
if (! -e scratch ) mkdir scratch
set scrdir = $chmtest/scratch
#
if (! -e $chmtest/data ) then
  echo " "
  echo " test.com> Directory $chmtest/data does not exist."
  exit 1
endif
if (! -e $chmtest/datadir.def ) then
  echo " "
  echo " test.com> $chmtest/datadir.def does not exist."
  exit 1
endif
if (! -e $chmtest/seddir ) then
  echo " "
  echo " test.com> $chmtest/seddir does not exist."
  exit 1
endif
#
#-----------------------------------------------------------------------
#
phase2:
#
# run testcase
if ( $cmpr != 0 ) then
  touch        $outdir.rpt
  /bin/date >> $outdir.rpt
  echo "test output $outdir against $bendir" >> $outdir.rpt
endif
#
if ( $testing == all ) set testing = 'mmff 20 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49'
#
limit cputime 5m
limit filesize 1024m
foreach suite ( $testing )
  if ( $ensemble != 1 ) then
  	set testlist = `/bin/ls $chmtest/c${suite}test | sed -e /CVS/d | sed -e  "/\(.*ens.inp\)/d"`
  else
        set testlist = `/bin/ls $chmtest/c${suite}test | sed -e /CVS/d | awk '/ens.inp$/ {print $0}'`
  endif
  echo $testlist
  if ( $chm_host == "t3e" ) then
     if ( -e $chmtest/c${suite}test/.t3e_list) then
       set testlist = `/bin/cat $chmtest/c${suite}test/.t3e_list`
     endif
  endif

  foreach testcase ( $testlist )
    set no_cmpr=0
    set name = $testcase:r
    set name = $name:t
    set ext  = $testcase:e
    set outname = $outdir/$name.out
    if ( $ext == "inp" && ! -e $outname ) then
      set inname = $chmtest/c${suite}test/$testcase
      set corename = $outdir/$name.core
      touch $scrdir/stamp
      if ( $keepscr == 0 ) then
         /bin/rm -f $scrdir/*
      endif
      if (-e $corename) rm $corename
      if ( $mpi_out_sep == 1 ) then
         set errname = $outdir/$name.err
         echo $chm_run --output-filename $outname $chm_exec -input $inname -prevclcg '>&' $errname
         $chm_run --output-filename $outname $chm_exec -input $inname -prevclcg >& $errname
         mv $outname.*0 $outname
         rm $outname.*
         if (-z $errname) rm $errname
      else
         # error output redirected to capture tracebacks
         echo $chm_run $chm_exec -input $inname -prevclcg '>&' $outname
         $chm_run $chm_exec -input $inname -prevclcg >& $outname
      endif
      if (-e core) mv core $corename
    endif
    if ( $cmpr != 0 ) then
      echo " " >> $outdir.rpt
      echo "<** c${suite}test : $name **>" `date` >> $outdir.rpt
      grep " TERMINATION" $outname >& /dev/null
      if ( $status) echo "***** NO TERMINATION  *****" | tee -a $outdir.rpt
      grep "ABNORMAL TERMINATION" $outname >& /dev/null
      if ( ! $status) echo "***** ABNORMAL TERMINATION *****" >> $outdir.rpt
      grep -i "Test NOT performed" $outname |grep -v "!" >& /dev/null
      if ( ! $status) set no_cmpr=1
      if ( $no_cmpr == 0) then
        if ( -e $bendir/$name.out ) then
          sed -f seddir $bendir/$name.out >! BenchMark1
          sed -f seddir $outdir/$name.out >! TestResult
          diff BenchMark1 TestResult >> $outdir.rpt
          /bin/rm -f BenchMark1 TestResult
        else
          echo "  $bendir/$name.out NOT found." >> $outdir.rpt
          echo " " >> $outdir.rpt
        endif
      else
        echo " Test NOT performed." >> $outdir.rpt
        echo " " >> $outdir.rpt
      endif
    endif
  end
end
#
if ( $keepscr == 0 ) then
  /bin/rm -f $scrdir/*
endif
echo " " >> $outdir.rpt
date     >> $outdir.rpt
# summary report
echo Summary of charmm testresults.  `date`
echo chmost $chmhost using $chmexec
echo outputs in directory $outdir
echo =============================================================
echo Testcases that fail:
grep "TESTCASE RESULT: FAIL" $outdir/*.out
echo =============================================================
echo Testcases that do not finish:
grep -L TERMINATION $outdir/*.out
echo =============================================================
echo Testcases that finish abnormally:
grep -l "ABNORMAL TERMINATION" $outdir/*.out
echo =============================================================
echo Number of testcases that have not been run:
egrep -il "TESTCASE RESULT: SKIP|test not performed" $outdir/*.out | wc -l
echo =============================================================
