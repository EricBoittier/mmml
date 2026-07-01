#!/bin/csh -f
# SCRIPT TO CONVERT DOC FILES TO HTML, AND "ACTIVATE" THE DOC LINKS
# This script has been modified by C.L. Brooks, III to accomodate
# many versions of charmm documentation and customize for TSRI

if ( $#argv == 0 ) then
  echo " "
  echo " Usage: html2doc.com path-to-charmm-root charmm-version version-number "
  echo " "
  echo "  E.g. html2doc.com /mydisk/c25a2 c25a2 25 "
  echo " "
  exit
#
else
  set root = $argv[1]
  set version = $argv[2]
  set vnum = $argv[3]
endif
setenv chmroot $root
setenv docs $chmroot/doc
setenv html $chmroot/html
echo " Building html documentation files for CHARMM version "$version
echo " Using root CHARMM directory as "$chmroot
echo " Taking doc files from "$docs
echo " Placing new html files in "$html

if(! -e $html ) mkdir $html

#goto last
# DEFINE THE COMMON TAIL; CUSTOMIZE THE TEXT BETWEEN cat AND 'FIN'
cat > fin.t << FIN
</PRE>
<HR>
<H4><A HREF="Charmm$vnum.Html">CHARMM .doc</A> Homepage</H4>
<HR>
<CITE>Information and HTML Formatting Courtesy of:</CITE><P>
NIH/DCRT/Laboratory for Structural Biology<BR>
FDA/CBER/OVRR Biophysics Laboratory<BR>
Modified, updated and generalized by C.L. Brooks, III<BR>
The Scripps Research Institute<BR>
<HR>
FIN

# Construct the generic sed and awk scripts

####  First fixdoc.sed
cat > fixdoc.sed << FIN
/^File:/s/:  /: /g
/^File:/s/:	/: /g
/^File:/s/,  /, /g
/^File:/s/ UP/ Up/
/^File:/s/ TOP,/ Top,/
/^File:/s/NEXT/Next/
/^File:/s/PREVIOUS/Previous/
/^File:/s/ up/ Up/
/^File:/s/ top,/ Top,/
/^File:/s/next/Next/
/^File:/s/previous/Previous/
/^File:/s/Prev:/Previous:/
s/&/\&amp;/g
s/</\&lt;/g
s/>/\&gt;/g
s/"/\&quot;/g
FIN

#### Next "build" info2html.awk
echo '{' > info2html.awk
echo 'if ( $1 ~ /^File:/ ) {' >> info2html.awk
echo ' nt = split($0,tags,",")' >> info2html.awk
echo ' for ( i=1; i <= nt; i++) {' >> info2html.awk
echo '  split(tags[i],tval,":")' >> info2html.awk
echo '  chm = substr(tval[2],1,8)' >> info2html.awk
echo '  if ( tval[1] == "File" ) {' >> info2html.awk
echo '   sub("^","<I>",tags[i])' >> info2html.awk
echo '   sub(":",":</I>",tags[i])' >> info2html.awk
echo '   }' >> info2html.awk
echo '  if ( tval[1] == " Node" ) {' >> info2html.awk
echo '   sub("^","<I>",tags[i])' >> info2html.awk
echo '   sub(":",":</I><A NAME=\"",tags[i])' >> info2html.awk
echo '   sub("$","\">",tags[i])' >> info2html.awk
echo '   sub("$",tval[2],tags[i])' >> info2html.awk
echo '   sub("$","</A>",tags[i])' >> info2html.awk
echo '   }' >> info2html.awk
echo '  if ( tval[1] == " Up" ) {' >> info2html.awk
echo '   sub("^","<I>",tags[i])' >> info2html.awk
echo '   sub(":","</I>:",tags[i])' >> info2html.awk
echo '   if ( chm != " (chmdoc" ) {' >> info2html.awk
echo '    sub(":",":<A HREF=\"#",tags[i])' >> info2html.awk
echo '    sub("$","\">",tags[i])' >> info2html.awk
echo '    sub("$",tval[2],tags[i])' >> info2html.awk
echo '    sub("$","</A>",tags[i])' >> info2html.awk
echo '    }' >> info2html.awk
echo '   }' >> info2html.awk
echo '  if ( tval[1] == " Next" ) {' >> info2html.awk
echo '   sub("^","<I>",tags[i])' >> info2html.awk
echo '   sub(":","</I>:",tags[i])' >> info2html.awk
echo '   if ( chm != " (chmdoc" ) {' >> info2html.awk
echo '    sub(":",":<A HREF=\"#",tags[i])' >> info2html.awk
echo '    sub("$","\">",tags[i])' >> info2html.awk
echo '    sub("$",tval[2],tags[i])' >> info2html.awk
echo '    sub("$","</A>",tags[i])' >> info2html.awk
echo '    }' >> info2html.awk
echo '   }' >> info2html.awk
echo '  if ( tval[1] == " Previous" ) {' >> info2html.awk
echo '   sub("^","<I>",tags[i])' >> info2html.awk
echo '   sub(":","</I>:",tags[i])' >> info2html.awk
echo '   if ( chm != " (chmdoc" ) {' >> info2html.awk
echo '    sub(":",":<A HREF=\"#",tags[i])' >> info2html.awk
echo '    sub("$","\">",tags[i])' >> info2html.awk
echo '    sub("$",tval[2],tags[i])' >> info2html.awk
echo '    sub("$","</A>",tags[i])' >> info2html.awk
echo '    }' >> info2html.awk
echo '   }' >> info2html.awk
echo '  }' >> info2html.awk
echo '  printf "%s  ]-[  %s<BR> ", tags[1], tags[2]' >> info2html.awk
echo '  for ( i=3; i < nt; i++)' >> info2html.awk
echo '   printf "%s -=- ", tags[i]' >> info2html.awk
echo '  printf "%s\\n", tags[nt]' >> info2html.awk
echo ' }' >> info2html.awk
echo 'else {' >> info2html.awk
echo ' if ( $0 ~ /^\* / ) ' >> info2html.awk
echo '  if ( $0 ~ /::/ ) {' >> info2html.awk
echo '  split($0,menu,":")' >> info2html.awk
echo '  token = substr(menu[1],2)' >> info2html.awk
echo '  sub(" ","<A HREF=\"# ")' >> info2html.awk
echo '  sub("::","\">::")' >> info2html.awk
echo '  token = token "</A>::"' >> info2html.awk
echo '  sub("::",token)' >> info2html.awk
echo '  }' >> info2html.awk
echo ' print $0' >> info2html.awk
echo ' }' >> info2html.awk
echo '}' >> info2html.awk

# Construct a list of the current doc files referred to in
# commands.doc to make sure all are referenced in html docs
# put results in sed script file for final conversion

# script file header
cat > emacs2href.sed << FIN
/^<I>File:/s/\$/<\/H4><PRE>/
/^<I>File:/s/^/<\/PRE><HR><H4>/
s///
/CHARMM Element/d
FIN

# get names of each doc file referenced in other doc files
#foreach name ( `awk '$1 == "*" { split($0,tags,"("); split(tags[2],tval,")") ; split(tval[1],fval,"/");split(fval[2],nval,".");print nval[1]  }' $docs/commands.doc `)

foreach name ( `ls $docs/*.doc | awk '{n=split($0,t,"/");print t[n]}'` )
if ($name == 'commands.doc' ) then
  echo 's;chmdoc/'${name:r}'.doc;<A HREF="Commands.Html">'${name:r}'.doc</A>;g' >> emacs2href.sed
else if ($name == 'charmm.doc') then
  echo 's;chmdoc/'${name:r}'.doc;<A HREF="Charmm'$vnum'.Html">'${name:r}'.doc</A>;g' >> emacs2href.sed
else
  echo 's;chmdoc/'${name:r}'.doc;<A HREF="'${name:r}'.html">'${name:r}'.doc</A>;g' >> emacs2href.sed
endif
end


# LOOP OVER FILE NAMES IN doc file, use only those with extension .doc
foreach f ( `ls $docs/*.doc | awk '{n=split($0,t,"/");print t[n]}'` )

echo "Working on file "$f
# SETUP VARIABLE HEADER
cat > hdr.t << FIN
<TITLE>$version $f</TITLE>
<H1>CHARMM $version $f</H1>
<PRE>
FIN

########## SED & AWK SCRIPTS DO THE DIRTY WORK HERE, IN A PIPELINE ######
# FIRST, CLEANUP INFO HEADERS, CONVERT [<>&"] CHARS TO HTML ESCAPES (FIXDOC)
# THE AWK SCRIPT ADDS THE HTML FORMATTING TO THE INFO HEADERS (INFO2HTML)
# MAKE INFO HEADERS INTO HTML HEADERS, ADD HOT LINKS TO FILES (EMACS2HREF)
# FINALLY, ADD HEADER AND TAIL WITH CAT AT THE END OF THE PIPE

#  Get machine type to decide whether to use nawk or awk (nawk on SGIs

set my_arch = unknown
if (-e /bin/arch)  set my_arch = `/bin/arch`

echo "******** my_arch is $my_arch *********"

if ( $my_arch == sgi4D) then
  sed -f fixdoc.sed $docs/$f \
    | nawk -f info2html.awk \
    | sed -f emacs2href.sed \
    | cat hdr.t - fin.t > $html/${f:r}.html
else
  sed -f fixdoc.sed $docs/$f \
    | awk -f info2html.awk \
    | sed -f emacs2href.sed \
    | cat hdr.t - fin.t > $html/${f:r}.html
endif
# FINISH LOOP OVER FILE NAMES
end

# Add enhanced files
last:
#  First Charmm$vnum.Html (from Charmm_template.Html)

cat > Charmm$vnum.sed << FIN
s/XXXVERSIONXXX/$version/g
s/XXXVNUMXXX/$vnum/g
/<pre>/a\
FIN

foreach name ( `ls $html | fgrep '.html'` )
if ($name == 'commands.html' ) then
  echo '<A HREF="Commands.Html">command.doc</A>\' >> Charmm$vnum.sed
else if ($name == 'charmm.html') then
  echo '<A HREF="Charmm'$vnum'.Html">charmm.doc</A>\' >> Charmm$vnum.sed
else if ($name == 'block.html' ) then
  echo '<A HREF="'${name:r}'.html">'${name:r}'.doc</A>\' >> Charmm$vnum.sed
  echo '<A HREF="Category.Html">Category.Html</A>\' >> Charmm$vnum.sed
else if ($name == 'nose.html' ) then
  echo '<A HREF="'${name:r}'.html">'${name:r}'.doc</A>\' >> Charmm$vnum.sed
  echo '<A HREF="Overview.Html">Overview.Html</A>\' >> Charmm$vnum.sed
else if ($name == 'mmff_params.html' ) then

else
  echo '<A HREF="'${name:r}'.html">'${name:r}'.doc</A>\' >> Charmm$vnum.sed
endif
end
#  add termination for append
echo ' ' >> Charmm$vnum.sed

sed -f Charmm$vnum.sed Charmm_template.Html > Charmm$vnum.Html

#  Now do Overview.Html

#echo 'at overview'
sed -e 's/XXXVERSIONXXX/'$version'/g' Overview_generic.Html | sed -e 's/XXXVNUMXXX/'$vnum'/g' > Overview.Html


#  Now Commands.Html
#echo 'at commands.sed'

cat > Commands.sed << FIN
s/XXXVERSIONXXX/$version/g
s/XXXVNUMXXX/$vnum/g
s/XXXc$vnum//
/XXXc25/d
FIN

#echo 'at commands'
sed -f Commands.sed Commands_template.Html > Commands.Html

#  Now do Category.Html
sed -f Commands.sed Category_generic.Html > Category.Html

#  Move files to final directory

cp p_A-animate.gif $html/
mv Charmm$vnum.Html Overview.Html Commands.Html Category.Html $html/

echo "You will want to customize all *.Html files for your site"
#exit
# CLEANUP
rm hdr.t fin.t 
rm *.sed *.awk

