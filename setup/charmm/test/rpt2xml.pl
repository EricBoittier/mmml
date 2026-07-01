#!/usr/bin/perl -w

#
# Reads tests from an output.rpt
# Runs compare.awk to determine pass/fail
# Writes an output.xml like the JUnit Ant task
# Mike Garrahan, 2009
#

use strict;

if (@ARGV != 1) {
   die "Usage: $0 outdir\n";
}
my $outdir = $ARGV[0];
my $rptfile = "$outdir.rpt";
my $xmlfile = "$outdir.xml";
my $xfailfile = "$outdir.xfail";
my $target = $outdir;
$target =~ s/^[^_]+_([^_]*).*$/$1/;

# Returns a copy of its argument modified for safe inclusion in XML.
sub xmlEscape($) {
   my $rv = shift;
   $rv =~ s/&/&amp;/g; # must do first
   $rv =~ s/</&lt;/g;
   $rv =~ s/>/&gt;/g;
   $rv =~ s/"/&quot;/g;
   $rv =~ s/'/&apos;/g;
   $rv =~ s/[^[:cntrl:][:print:]]/*/g;
   return $rv;
}

my %suites = ();
my %result = ();
my $test;
my $suite;
open RPT, "< $rptfile" or die "$rptfile: $!";
while (<RPT>) {
   if (/^<\*\* (c.+test) : (\S+) \*\*>/) {
      $suite = $1;
      $test = $2;
      if (not exists $suites{$suite}) {
         $suites{$suite} = [];
      }
      push @{$suites{$suite}}, $test;
      next;
   }
   next if not defined($test);
   if (/^ Test NOT performed/) {
      $result{$test} = "ignored";
   }
   elsif (/^\*{5} NO TERMINATION/) {
      $result{$test} = "crashed";
   }
}
close RPT;

open AWK, "awk -f compare.awk $rptfile |" or die;
while (<AWK>) {
   if (/TEST (\S+) (\w+)/) {
      $test = $1;
      next if exists $result{$test};
      $result{$test} = lc $2;
   }
}
close AWK;

my %xfail = ();
my $maxfail = 10;
if (-e $xfailfile) {
   open XFAIL, "< $xfailfile" or die;
   foreach $test (<XFAIL>) {
      chomp $test;
      $xfail{$test} = 1;
   }
   close XFAIL;
   $xfail{"placeholder"} = 1;
}

open XML, "> $xmlfile" or die "$xmlfile: $!";
binmode XML, ":utf8";
print XML "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
print XML "<testsuites>\n";
my @failures = ();
my $tot_testct = 0;
foreach $suite (sort keys %suites) {
   my $testct = 0;
   my $failct = 0;
   my $errct = 0;
   foreach $test (@{$suites{$suite}}) {
      if ($result{$test} ne "ignored") {
         ++$testct;
      }
      if ($result{$test} eq "failed") {
         push @failures, "$suite/$test";
         ++$failct;
      }
      elsif ($result{$test} eq "crashed") {
         push @failures, "$suite/$test";
         ++$errct;
      }
   }

   # need counts before we can write this
   print XML "  <testsuite name=\"$suite\" tests=\"$testct\" failures=\"$failct\" errors=\"$errct\">\n";
   foreach $test (@{$suites{$suite}}) {
      if ($result{$test} eq "ignored") {
         next;
      }
      elsif ($result{$test} eq "passed") {
         print XML "    <testcase name=\"$test\" />\n";
      }
      else {
         print XML "    <testcase name=\"$test\">\n";
         if ($result{$test} eq "failed") {
            print XML "      <failure>";
            if (%xfail and not exists $xfail{"$suite/$test"}) {
               print XML "NEW FAILURE\n";
               print XML xmlEscape(`./charmmdif -w ref/$outdir/$test.out $outdir/$test.out | head -n20`);
            }
            print XML "</failure>\n";
         }
         elsif ($result{$test} eq "crashed") {
            print XML "      <error>\n";
            if (%xfail and not exists $xfail{"$suite/$test"}) {
               print XML "NEW CRASH\n";
            }
            if (-e "$outdir/$test.err") {
               print XML xmlEscape(`cat $outdir/$test.err`);
            }
            if (-e "$outdir/$test.core") {
               print XML xmlEscape(`TERM=tty gdb --batch -ex bt ../exec/$target/charmm $outdir/$test.core 2>/dev/null`);
            }
            print XML "      </error>\n";
         }
         print XML "    </testcase>\n";
      }
   }
   print XML "  </testsuite>\n";
   $tot_testct += $testct;
}
print XML "</testsuites>\n";
close XML;

if ($tot_testct == 0) {
   print STDERR "NO TESTS RUN\n";
   exit 2;
}

if (%xfail) {
   my @newfails = ();
   foreach $test (@failures) {
      if (not exists $xfail{$test}) {
         push @newfails, $test;
      }
   }
   if (@newfails) {
      print STDERR "NEW TEST FAILURES:\n";
      foreach $test (@newfails) {
         print STDERR "$test\n";
      }
      exit 1;
   }
   else {
      exit 0;
   }
}
elsif (@failures > $maxfail) {
   print STDERR scalar @failures, " TEST FAILURES\n";
   exit 1;
}
else {
   open XFAIL, "> $xfailfile" or die;
   foreach $test (@failures) {
      print XFAIL "$test\n";
   }
   close XFAIL;
   exit 0;
}
