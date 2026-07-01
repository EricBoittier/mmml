#!/usr/bin/perl

use POSIX;

############################   Set Variables   ################################
@iseq=(7,5);
@atomin=(12,4);
@rid=(23,3);
$syscommand[0]="clear";
#divisor is a testing and debug option
$divisor=10;

# the formats are defined here since they work only with global variables

format IC=
@<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<@###.####@####.##@####.##@####.##@###.####
$atoms,$bonda,$anglea,$phi,$angleb,$bondb
.


#set help string with the BLARG syntax = multi line formatted string assignment
$help=<<'BLARG';

###############################  vector.pl ####################################
# Calculates the shortest angle-change (="vector") between two torsion angles #
# in two CHARMM IC-tables, for ex. -120 --> +120 is -120, not +240 degrees.   #
#                                                                             #
# The output file is an IC table that can be read into CHARMM.                #
#                                                                             #
# This script is used to generate initial paths interpolated in IC for CPR.   #
#                                                                             #
# With the -ta option, the vector will be written only if the |vector| is     #
# larger than a user supplied threshold value (default is 60 degrees).        #
#                                                                             #
# By Fabian Ille & Stefan Fischer, last modified Jun.29-2006                  #
###############################################################################

USAGE:
[1mvector.pl[0m <param1> <param2> <param3> [<param4>] [<param5>]


1) For this help

    <param1> =[1m -h[0m


2) For calculating all vectors:

    <param1> = <reactant IC file>

    <param2> = <product IC file> 

  [<param3>] = Optional <output file> (<first IC File>.ic)


3) For calculating only vectors larger than the threshold value:

    <param1> = [1m-ta[0m <threshold-value>

               The thershold-angle is a numerical value from 1 to 180 degrees.
               The default is 60 degrees.

    <param2> = <first IC file>

    <param3> = <second IC file>

  [<param4>] = Optional <output file> (<first IC File>.ic)
  
BLARG

################################################################################
#Parse input
#Show help or just open files
#The command line entry is set to variable $ARGV which is an array !!!
################################################################################

# # Clearing the screen :
# system($syscommand[0]);

$threshcheck=0; #used to test wether a threshold is used or not in order to change the title of the output
$argcnt=1;
if ($ARGV[0] !~ m/\-ta/)
  {
   foreach $argument (@ARGV)
    {
     $ARGW[$argcnt]=$argument;
     $argcnt++; 
    }
    print "vector.pl is computing vector IC-table\n--------------------------------------------\n";
  }
  elsif ($ARGV[1] =~ m/\D/ && $ARGV[0] =~ m/\-ta/)
  {
    $threshcheck=1;
    $threshold=60;
    foreach $argument (@ARGV)
    { 
     if ($argcnt >= 2)
       {
       $ARGW[$argcnt-1]=$argument;
       }
     $argcnt++; 
    }
    print "vector.pl is computing vector IC-table above $threshold° value;\n";
  }
  else 
  { 
   $threshcheck=1;  
   if ($ARGV[1] > 0 && $ARGV[1] < 181)
    {
     $threshold=$ARGV[1];
    }
    print "vector.pl is computing vector IC-table above $threshold° value;\n";
    foreach $argument (@ARGV)
    {
     if ($argcnt >= 3)
       {
       $ARGW[$argcnt-2]=$argument;
       }
     $argcnt++; 
    }
   }
   

#foreach $argument (@ARGV)
#    {
#     print "DEBUG original input: ".$argument."\n";
#    }
#    print "#####################\n";
#foreach $argument (@ARGW)
#    {
#     print "DEBUG parsed input: ".$argument."\n";
#    }
#   print "DEBUG logical check for threshold setting: ".$threshcheck."\n";
if ($ARGW[1] ne '' and $ARGW[2] ne '')
        {
                if ($ARGW[3] eq '')
                {
                $ARGW[3]=$ARGW[1].".ic";
                };
                open (INA, "$ARGW[1]")|| die "use option -h for help! Couldn't open first input ic table!";
                open (INB, "$ARGW[2]")|| die "use option -h for help! Couldn't open second input ic table!";
		open (IC, "> $ARGW[3]") || die "use option -h for help! Couldn't open output file!";
                $call=ICCalc($ARGW[0]); #Call Subroutine to fix for CHARMM file for MOLOC
        }
else
        {
                system ("echo '".$help."'|less -r");
                die "progam regularly terminated";
        }


sub ICCalc{
# this routine is used to read out two files and calculate the angles 
my $outputcheck=0;
my $arg=$_[0];
my $atome="";
#write a comment into the header of the file
print IC "* Difference of two IC tables.  Calculated with vector.pl\n";
print IC "* First  IC file: $ARGW[1]\n";
print IC "* Second IC file: $ARGW[2]\n";
if ($threshcheck != 1)
     {
      print IC "* No threshold used : showing all IC entries.\n";
     }
   else
     {
      $threshold=60 if ($threshold eq "");
      print "--------------------------------------------\n";
      print IC "* Only showing |torsion differences| larger than $threshold\n";
     }

#read in first file into hashes and store keys in @atoms_a
print "vector.pl is reading first input file:$ARGW[1]\n";
 while ($linea=<INA>)
    {
     if (substr($linea,42,1) eq "." and substr($linea,52,1) eq "." and substr($linea,60,1) eq "." and substr($linea,68,1) eq "." and substr($linea,75,1) eq ".")
       {
       $atome=substr($linea,0,38);
       push(@atoms_a,$atome);
       $bonda_a{$atome}=substr($linea,38,9);
       $anglea_a{$atome}=substr($linea,47,8);
       $phi_a{$atome}=substr($linea,55,8);
       $angleb_a{$atome}=substr($linea,63,8);
       $bondb_a{$atome}=substr($linea,71,9);
       
       }
       else
       {
        print IC $linea;
       }      
    }
#read in second file into hashes and store keys in @atoms_b
print "vector.pl is reading second input file:$ARGW[2]\n";
 while ($lineb=<INB>)
   {
     if (substr($lineb,42,1) eq "." and substr($lineb,52,1) eq "." and substr($lineb,60,1) eq "." and substr($lineb,68,1) eq "." and substr($lineb,75,1) eq ".")
       {
       $atome=substr($lineb,0,38);
       push(@atoms_b,$atome);
       $bonda_b{$atome}=substr($lineb,38,9);
       $anglea_b{$atome}=substr($lineb,47,8);
       $phi_b{$atome}=substr($lineb,55,8);
       $angleb_b{$atome}=substr($lineb,63,8);
       $bondb_b{$atome}=substr($lineb,71,9);
       
       }
    }       
# check for some errors that might have occured;  
$key1=@atoms_a;
$key2=@atoms_b;   

if ($key1 == 0)
   {die "Error: File 1 is not recognized as IC Table from CHARMM\n";}
if ($key2 == 0)
   {die "Error: File 2 is not recognized as IC Table from CHARMM\n";}  
if ($key1 != $key2)
   {
    print "Entries in ".$ARGW[1].":".$key1."\nEntries in ".$ARGW[2].":".$key2."\n"; 
    print "The files do not seem to have the same number of entries!\n\nATTENTION:Some lines might not be processed!\n\n";
    }

#start with the calculation
foreach $key (@atoms_a)
   {

#calculate vector
     $atoms=$key;
     $bonda=$bonda_b{$key}-$bonda_a{$key};
     $bonda=nearest(.0001,$bonda);
     $anglea=$anglea_b{$key}-$anglea_a{$key};
     $anglea=nearest (.01,$anglea);
     $angleb=$angleb_b{$key}-$angleb_a{$key};
     $angleb=nearest (.01,$angleb);
     $bondb=$bondb_b{$key}-$bondb_a{$key};
     $bondb=nearest (.0001,$bondb);
     $phi=angle($phi_a{$key},$phi_b{$key},"vec");
#    $phi=$phi/$divisor;
     $phi=nearest(.01,$phi);
if ($threshcheck==0)
  {
     write IC;
  }
elsif ($threshcheck==1)
  {
   if ( abs($phi) > $threshold)
       {
        write IC;
       }
  }
  }
###########################    Close Filehandle    ############################
close (IC);
################################################################################
print "vector.pl is writing output to file: $ARGW[3]\n";
$dummy=USformat($ARGW[3]);
print "--------------------------------------------\nvector.pl is done\n\n";
}


################################################################################################
# Functions                                                                                    #
# Do not write any subroutines below this point... this is thought only for functions          #
################################################################################################
sub USformat {
#convert to US number mode
my @value=@_;

#open filehandle
open (USIN, $value[0]);

#load the file
while ($line=<USIN>)
{
  push(@array,$line);
}

# close input handle
close (USIN);

# open output file
open (USOUT,"> $value[0]");

#replace all kommas through points
foreach $key (@array)
   {
   $key=~ s/\,/\./g;
   print (USOUT $key);
   }
   
#close filehandle
close (USOUT);
}


sub angle {
#angle(reactant,product,mode)
my @value=@_;
my $result=0;
if ($value[2] eq "vec")
  {
    $result=$value[1]-$value[0];
    if ($result > 180)
       {
        $result=$result-360;
        }
    elsif ($result < -180)
       {
        $result=$result+360;
        }
  }
if ($value[2] eq "dif")
   {
   if (($value[0]*$value[1])>0)
       {
         $result=abs($value[1]-$value[0]);
       }
   else
       {
         if ((abs($value[0])+abs($value[1]))>180)
           {
            $result=360-(abs($value[0])+abs($value[1]));
            }
         else
           {
            $result=(abs($value[0])+abs($value[1]));
            }
       }
  }
 
return ($result);
}
 
sub nearest {
# subroutine taken from CPAN Math::Round
my ($targ, @inputs) = @_;
 my @res = ();
 my $x;

 $targ = abs($targ) if $targ < 0;
 foreach $x (@inputs) {
   if ($x >= 0) {
      push @res, $targ * int(($x + 0.5 * $targ) / $targ);
   } else {
      push @res, $targ * POSIX::ceil(($x - 0.5 * $targ) / $targ);
   }
 }
 if (wantarray) { return @res; }
 else           { return $res[0]; }
}
