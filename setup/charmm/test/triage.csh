#! /bin/csh
#
# Quick triage scan of the testcase output for failures
# multiple subdir names are expected as arguments
# get the total number of files, and nominal successful completions
# contributed by Rick Venable, NIH/NHLBI, Sept. 2014

if ( $1 == '' ) then
 echo 'no subdir arg supplied, exiting'
 exit
endif

foreach d ( $* )

echo "  #############  $d"

# total
set tot = `ls $d/*.out | wc -l`
# completed okay
set ok = `grep -l ' NORMAL TERMINATION ' $d/*.out | tee $d/completed.ok | wc -l`

# check for skipped tests
set not = `grep -l -i -e 'Test NOT performed' $d/*.out | tee $d/performed.not | wc -l`

# print numbers
echo "Total files $tot,  Successful completions $ok,  Not performed $not"

# check for errors trapped by CHARMM
set ntrap = `grep -l -e 'Execution terminated ' -e 'ABNORMAL TERMINATION' $d/*.out | tee $d/outtrap.err | wc -l`
cat $d/outtrap.err | sed -e "s;$d;;" -e 's/\.out/.inp/' >! $d/chrmtrap.err 
echo "	=====< $ntrap errors trapped by CHARMM:"
ls -1 c*test/*.inp | grep -f $d/chrmtrap.err | sort -t/ -k 2 | paste - $d/outtrap.err

echo ''
# check for fatal errors
cat $d/completed.ok $d/outtrap.err >! $d/outexit.ok
cat $d/outexit.ok | sed -e "s;$d/;;" -e 's/\.out/.inp/' >! $d/chrmexit.ok

set fatal = `ls $d/*.out | grep -w -v -f $d/outexit.ok | tee $d/outfatal.err | wc -l`
cat $d/outfatal.err | sed -e "s;$d/;;" -e 's/\.out/.inp/' >! $d/fatal.err
echo "	=====< $fatal untrapped fatal errors:"
ls -1 c*test/*.inp | grep -w -f $d/fatal.err | sort -t/ -k 2 | paste - $d/outfatal.err

echo ''
end

