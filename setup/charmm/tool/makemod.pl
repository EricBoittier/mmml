#!/usr/bin/perl
#
# makemod.pl: make makefile for a package
# Based on prior scripts makemod, mainmake, and finduse
# Mike Garrahan, 2009
#

use strict;

my $prog = 'makemod.pl';
my $rc = 0;
my %definer = ();
my %optlevel = ();
my %pkglevel = ();
my %suppress_missing_warning = ();
my $verbose = 0;
my $recurse = 1;
my @args = ();
foreach my $arg (@ARGV) {
   if ($arg eq '-v') {
      $verbose = 1;
   } elsif ($arg eq '-n') {
      $recurse = 0;
   } else {
      push @args, $arg;
   }
}
if (@args < 4) {
   die <<EOT;
usage: $prog [-v] [-n] package_name actual_path sourcetree_path
                            makefile_name [definitionfile_name]
EOT
}

foreach my $modname ('ieee_arithmetic', 'ieee_exceptions', 'ieee_features',
      'iso_c_binding', 'iso_fortran_env', 'ifport', 'mpi', 'mkl_dfti') {
   $definer{$modname} = 'intrinsic';
}
# custom optimization for individual files
foreach my $filename ('dynamc', 'ecmap', 'grid', 'eef1', 'gbmvmodule') {
   $optlevel{$filename} = 1;
}
$optlevel{'fctall'} = 3;
# custom optimization for entire packages (directories)
foreach my $package ('stringm') {
   $pkglevel{$package} = 3;
}
# suppress warnings when certain files/directories are missing; for example, the string method source
# code is created by preprocessor only when it is requested; the missing file warnings are harmless
# but annoying
foreach my $modname ('multicom','multicom_aux','chirality','confcons','sm_config',    # stringm files
 'sm_var','ftsm_var','ftsm_voronoi','sm0k','ftsm','smcv_master','cv_common','ifstack','stringm') # stringm files
{
 $suppress_missing_warning{$modname}=1;
}


# Fix for automount sites; replace pwd with ampwd fxn
sub ampwd($) {
   my $path = shift;
   $path =~ s#/tmp_mnt##;
   return $path;
}

# Reads modules.txt into a map of module => source.
sub load_modmap() {
   my $mapfile = "$ENV{chmtool}/modules.txt";
   open MODMAP, "< $mapfile"
         or die "$prog: cannot read $mapfile: $!\n";
   while (<MODMAP>) {
      my ($junk, $modname, $defsrc) = split;
      $modname = lc $modname;
      $defsrc =~ s/\.F90$//;
      if (exists $definer{$modname}) {
         die "$prog: duplicate definition of $modname ($definer{$modname}, $defsrc)\n";
      } else {
         $definer{$modname} = $defsrc;
      }
   }
   close MODMAP;
}

# Returns a list of the files which the given source file USEs.
sub objs_used_by($) {
   my $src = shift;
   my %modules = ();
   open SRC, "< $src"
         or die "$prog: cannot read $src: $!\n";
   while (<SRC>) {
      if (/^##USE +(\w+)/ or /^ *use +(\w+)/i) {
         my $modname = lc $1;
         if (!exists($definer{$modname})) {
          if (!exists($suppress_missing_warning{$modname})) {
            warn "$prog: $src($.): module $modname undefined\n";
            $rc = 1;
          }
         } elsif ($definer{$modname} ne $src
               and $definer{$modname} ne 'intrinsic') {
            $modules{$modname} = 1;
         }
      }
   }
   close SRC;
   return map { $definer{$_} } sort keys %modules;
}

sub add_breaks(\@) {
   my $arr = shift;
   my $n = scalar @$arr;
   my $m = $n % 8;
   if ($m == 0) {
      $m = 8;
   }
   my $i = $n - $m;
   while ($i > 0) {
      splice @$arr, $i, 0, "\\\n   ";
      $i -= 8;
   }
}

#
# Main program
#

my $curdir = ampwd(`pwd`);
chomp $curdir;
my @objs = ();
my @rules = ();
my @deps = ();
my @pairs = ();
my $package = $args[0];
my $actpath = ampwd($args[1]);
my $sourcepath = ampwd($args[2]);
my $makefilename = $args[3];
my @macros = ();

# see if there is a definition file to insert
if (@args == 5) {
   my $deffile=$args[4];
   if (-r $deffile) {
      if ($verbose) {
         print "Macro definition file: $deffile\n";
      }
      open DEFS, "< $deffile";
      @macros = <DEFS>;
      close DEFS;
   } else {
      die "$prog: cannot read $deffile: $!\n";
   }
} else {
   @macros = ();
}

load_modmap();

# check to see that actpath is a child of source tree path
my $relpath;
if ($actpath !~ /^$sourcepath/) {
   die "$prog: actual path $actpath is not a sub-directory of source tree path $sourcepath\n";
} else {
   if ($actpath eq $sourcepath) {
      $relpath = '.';
   } else {
      $relpath = $actpath;
      $relpath =~ s#^$sourcepath/##;
   }
}
chdir $actpath;

#
# Determine makefile contents
#
if ($verbose) {
   print "making $makefilename\n";
}
my $pkgsrc = "\$(SRC)/$relpath";
opendir DIR, $curdir
      or die "$prog: cannot list $curdir: $!\n";
my @files = grep { /^[^.].*\.(cu|c|cpp|f|f90|F90|src)$/ } readdir DIR;
closedir DIR;
if (! @files) {
   if ($suppress_missing_warning{$relpath}) {
#    exit 0
   } else {
    die "$prog: no source files in $relpath\n";
   }
}

my $fcdefault;
if (exists $pkglevel{$package}) {
 $fcdefault = "FC$pkglevel{$package}";
} else {
 $fcdefault = "FC2";
}

foreach my $file (sort @files) {
   my ($filename, $ext) = split /\./, $file;
   my $objfile = "$filename.o";
   push @objs, $objfile;
   push @rules, "$objfile : $pkgsrc/$file";
   if ($ext eq 'F90') {
      my $fc;
      if (exists $optlevel{$filename}) {
         $fc = "FC$optlevel{$filename}";
      } else {
         $fc = $fcdefault
      }
      if ($package eq 'charmm') {
         # push @rules, "\t\$(FLX) $pkgsrc/$file";
	 # push @rules, "\tcp $pkgsrc/$file $filename.F90";
	 # push @rules, "\t\$($fc) $filename.F90";
         push @rules, "\t\$($fc) $pkgsrc/$file";
	 # push @rules, "\t\$(REMOVE_F) $filename.F90";
         push @rules, "";
         push @rules, "\$(LIB)/$filename.o : $filename.o";
         push @rules, "\tcp $filename.o \$(LIB)";
      } else {
         if ($relpath eq "gamint/gamess") {
            push @rules, "\t\$(GMS) $pkgsrc/$filename";
         } else {
            # push @rules, "\t\$(FLX) $pkgsrc/$file";
	    # push @rules, "\tcp $pkgsrc/$file $filename.F90";
	    # push @rules, "\t\$($fc) $filename.F90";
            push @rules, "\t\$($fc) $pkgsrc/$file";
         }
         if ($relpath eq "gamint/gamess") {
            push @rules, "\t\$(REMOVE_F) $filename.f";
         } else {
            push @rules, "\t\$(REMOVE_F) $filename.F90";
         }
      }

      my @uses = ();
      foreach my $modfile (objs_used_by($file)) {
         my ($archive, $obj) = split '/', $modfile;
         my $prereq = "$obj.o";
         next if $objfile eq $prereq;
         push @uses, $prereq;
      }
      if (@uses) {
         foreach my $used (@uses) {
            push @pairs, "$objfile $used";
         }
         add_breaks(@uses);
         push @deps, "";
         push @deps, "$objfile : @uses";
      }
      # findinc has nothing to find anymore
   } else {
      if ($ext eq 'cu') {
         push @rules, "\t\$(CUDAC) -c $pkgsrc/$file";
      } elsif ($ext eq 'c') {
         push @rules, "\t\$(CC) \$(FCDEFINES) -c $pkgsrc/$file";
      } elsif ($ext eq 'cpp') {
         push @rules, "\t\$(CCPP) \$(FCDEFINES) \$(INCLUDE) -c $pkgsrc/$file";
      } elsif ($ext eq 'F90') {
         push @rules, "\t\$(FC2) $pkgsrc/$file";
      } elsif ($ext eq 'f') {
         push @rules, "\t\$(F77) $pkgsrc/$file";
      }
   }
   push @rules, "";
}

#
# Write makefile
#
open MAKEFILE, "> $makefilename"
      or die "$prog: cannot write $makefilename: $!\n";
print MAKEFILE "# $package makefile generated by tool/setmk.com\n";
if (@macros) {
   $" = ''; # list separator
   print MAKEFILE @macros;
   print MAKEFILE "#\n";
}

add_breaks(@objs);

print MAKEFILE <<EOT;
# $package library rules
OBJS_$package = @objs

EOT

if ($package eq "charmm") {
   print MAKEFILE "\$(LIB)/stamp : \$(OBJS_$package)\n";
   print MAKEFILE "\ttouch \$(LIB)/stamp\n";
} else {
   print MAKEFILE "\$(LIB)/$package.a : \$(OBJS_$package)\n";
   print MAKEFILE "\t\$(AR_COMMAND) \$(LIB)/$package.a \$(OBJS_$package)\n";
   print MAKEFILE "\t\$(RANLIB) \$(LIB)/$package.a\n";
}

$" = "\n"; # list separator
print MAKEFILE <<EOT;
\t\@echo $package COMPLETED

# $package source file rules
@rules

# $package dependency file
@deps
EOT
close MAKEFILE;

my $tsortfile = "$ENV{chmtool}/tsort.inp";
open TSORT, ">> $tsortfile"
      or die "$prog: cannot write $tsortfile: $!\n";
$" = "\n"; # list separator
print TSORT "@pairs\n";
close TSORT;

exit $rc;
