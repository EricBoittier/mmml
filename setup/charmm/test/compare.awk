# This awk script is designed to process the "report" output that comes from
# running the CHARMM test.com test script. It produces condensed reports on
# the success/failure of test cases (with key verbose=0 - default) or a 
# report that filters trivial differences, as well as those where the
# relative deviation between two numbers is less than the specified tolerance
# (tol - default = 1.0e-2). This filter is constructed for fields a and b 
# from the test.com report output (x.rpt) as:
# a = abs( lowercase( a ) + 0 ); b = abs( lowercase( b ) + 0 ); 
# diff = abs( a - b ); if( max(a, b) >= tol*tol) diff = diff / max(a,b);
# if( diff > tol && max(a,b) > tol ) linedifferent = .true.;
#
# Usage:
# awk -f compare.awk -v verbose=0 -v tol=0.01 diff_20.rpt
# Produces a passed/failed listing of all tests compared in diff_20.rpt
# awk -f comare.awk -v verbose=1 -v tol=0.01 diff_20.rpt
# Produces a more extensive listing of the "filtered" differences.
#
# Written by C.L. Brooks III, The Scripps Research Institute, December, 2003.
#
BEGIN{
  passed=1;
  debug=0;
  if(ARGC <= 1){
    print "###############################################################################";
    print "# This awk script is designed to process the report output that comes from    #";
    print "# running the CHARMM test.com test script. It produces condensed reports on   #";
    print "# the success/failure of test cases (with key verbose=0 - default) or a       #";
    print "# report that filters trivial differences, as well as those where the         #";
    print "# relative deviation between two numbers is less than the specified tolerance #";
    print "# (tol - default = 1.0e-2). This filter is constructed for fields a and b     #";
    print "# from the test.com report output (x.rpt) as:                                 #";
    print "# a = abs( lowercase( a ) + 0 ); b = abs( lowercase( b ) + 0 );               #";
    print "# diff = abs( a - b ); if( max(a, b) >= tol*tol) diff = diff / max(a,b);      #";
    print "# if( diff > tol && max(a,b) > tol ) linedifferent = .true.;                  #";
    print "# Usage:                                                                      #";
    print "# awk -f compare.awk -v verbose=0 -v tol=0.01 diff_20.rpt                    #";
    print "# Produces a passed/failed listing of all tests compared in diff_20.rpt       #";
    print "# awk -f comare.awk -v verbose=1 -v tol=0.01 diff_20.rpt                     #";
    print "# Produces a more extensive listing of the filtered differences.              #";
    print "# Written by C.L. Brooks III, The Scripps Research Institute, December, 2003. #";
    print "###############################################################################";
    print "Usage: awk -f compare.awk -v verbose=0/1 -v tol=<tolerance> file.rpt";
    exit;
  }
}
{
# Set-up variables, keys and defaults
  if(scott==1){
    key="diffing:"; #for Scott's verify file
    place=1;   # for Scott's verify file
  }
  else{
    key="<**"; #for test.com ".rpt" file
    place=1;   # for test.com ".rpt" file
    if(debug) print "Using test.com .rpt file";
  }
  if(verbose==0){verbose=0;}
  if(tol==0){tol=1.0e-2;}
  while ($place!=key) {
    if($1=="<"||$1==">"||$1=="---"){

      nbad = 0;
      while($1=="<") {
	if ((match($0,"Parallel load") == 0) \
            && (match($0,"New timer") == 0)) {
	  nf++;first[nf] = "";first[nf] = $0;
	}
	status=getline; if(status<1){exit;}
      }
      if($1=="---"){status=getline; if(status<1){exit;}}
      while($1==">") {
	if ((match($0,"Parallel load") == 0) \
            && (match($0,"New timer") == 0)) {
	  ns++;second[ns] = "";second[ns] = $0;
	}
	status=getline; if(status<1){exit;}
      }
    }

    if($place!=key&&!($1=="<"||$1==">"||$1=="---")){

#######################################################################
      if(nf>0||ns>0) {
	if(debug)print lines;
	nc = 0;
	nc = nf;
	if (ns<nc){nc=ns;}
	if(debug)print "nf,ns",nf,ns;
	for(i=1;i<=nc;i++){
	  nfc = split(first[i],fc);
	  nsc = split(second[i],fs);
	  
	  bad=0;
          expected = (fc[2]=="MAXIMUM"&&fc[3]=="SPACE") ;
	  expected = expected || (fs[2]=="MAXIMUM"&&fs[3]=="SPACE") ;
	  expected = expected || (fs[3] == "ADDRESS:") || (fc[3] == "ADDRESS:") ;
	  expected = expected || (fc[2]=="CONTROL"&&fc[3]=="ARRAY") ; 
          expected = expected || (fs[2]=="CONTROL"&&fs[3]=="ARRAY") ;
	  expected = expected || (fc[2]==">>TESTENDIAN") || (fs[2]==">>TESTENDIAN");
          expected = expected || (fc[2]=="Number") || (fs[2]=="Number") ;
	  expected = expected || (fc[2]=="FORCE=") || (fs[2]=="FORCE=");
	  expected = expected || (fc[2]=="VCLOSE:") || (fs[2]=="VCLOSE:");
	  expected = expected || (fc[7]=="99") || (fs[7]=="99");
	  expected = expected || (fc[6]=="99") || (fs[6]=="99");
	  expected = expected || (fc[8]=="99") || (fs[8]=="99");
	  expected = expected || (fc[7]=="5") || (fs[7]=="5");
	  expected = expected || (fc[8]=="5") || (fs[8]=="5");
	  expected = expected || (fc[6]=="5") || (fs[6]=="5");
	  expected = expected || (fc[2]=="CORR."&&fc[3]=="COEFFICIENT") ; 
          expected = expected || (fs[2]=="CORR."&&fs[3]=="COEFFICIENT") ;
	  expected = expected || (fc[3]=="GROUP"&&fc[4]=="PAIRS") ; 
          expected = expected || (fs[3]=="GROUP"&&fs[4]=="PAIRS") ;
	  expected = expected || (fc[3]=="PAIRS"&&fc[4]=="USED") ; 
          expected = expected || (fs[3]=="PAIRS"&&fs[4]=="USED") ;
	  expected = expected || (fc[3]=="ATOM"&&fc[4]=="PAIRS") ; 
          expected = expected || (fs[3]=="ATOM"&&fs[4]=="PAIRS") ;
	  expected = expected || (fc[5]=="ATOM"&&fc[6]=="PAIRS") ; 
          expected = expected || (fs[5]=="ATOM"&&fs[6]=="PAIRS") ;
	  expected = expected || (fc[5]=="atom"&&fc[6]=="pairs") ; 
          expected = expected || (fs[5]=="atom"&&fs[6]=="pairs") ;
	  expected = expected || (fc[5]=="group"&&fc[6]=="pairs") ; 
          expected = expected || (fs[5]=="group"&&fs[6]=="pairs") ;
          # c36a1x: ignore last field of MKIMAT vs. MKIMAT2
          expected = expected || (fc[4]=="has" && fs[4]=="has" \
              && fc[2]==fs[2] && fc[3]==fs[3] \
              && fc[5]==fs[5] && fc[6]==fs[6] && fc[7]==fs[7]) ;
	  if(!expected){
	    j=2;
	    while(j<=nfc&&j<=nsc&&bad!=1){
              fcs0 = sub("D-","E-",fc[j]);
              fss0 = sub("D-","E-",fs[j]);
              fcs = sub("D+","E+",fc[j]);
              fss = sub("D+","E+",fs[j]);
              a=abs(tolower(fc[j])+0);
              b=abs(tolower(fs[j])+0);
              diff = a - b;
	      diff = abs(diff);
	      denom = max(a,b); 
	      if(denom > tol*tol) {diff = diff * denom^(-1);}
	      if( diff > tol && denom > tol ){ 
                  bad=1; nbad++; nbadp[nbad]=i; 
		  if(debug)print "fc[j],fs[j]",fc[j],fs[j],"diff: "diff;
                  if(debug){ print "diff="diff;}
              }
	      j++;
	    }
	  }
	}
	if(nbad > 0||nf>ns||ns>nf){ 
	  if(debug)print "Setting passed to fail for", filename,nbad,nf,ns;
	  passed=0;
	  if(verbose){
	    print lines;
	    for(i=1;(i<=nbad)&&(nbadp[i]<=nf);i++){
	      print first[nbadp[i]];}
	    if(debug)print "nbad,nbadp[nbad],nf",nbad,nbadp[nbad],nf;
	    if(nbadp[nbad]<nf){for(i=nbadp[nbad]+1;i<=nf;i++){print first[i];}}
	    print "---";
	    for(i=1;(i<=nbad)&&(nbadp[i]<=nf);i++){
	      print second[nbadp[i]];}
	    if(debug)print "nbad,nbadp[nbad],ns",nbad,nbadp[nbad],ns;
	    if(nbadp[nbad]<ns){for(i=nbadp[nbad]+1;i<=ns;i++){print second[i];}}
	    nbad = 0;
	  }
	}
      }
      #######################################################################
      
    }
    nf = 0;
    ns = 0;
    lines = $0;if(debug)print lines;
    if($place==key) {
      if(verbose) {print "***Diffing: "$0;}
      else {
	if(passed){print "****TEST "filename" PASSED****";passed=1;}
	else{print ">>>>>TEST "filename" FAILED<<<<<";passed=1;}
      }
      if(key=="<**")filename=$4;
      else{filename=$2;}
    }

    status=getline; if(status<1){break;}
  }
  if(!(verbose)) {
    if(filename!=""){
      if(passed){print "****TEST "filename" PASSED****";passed=1;}
      else{print ">>>>>TEST "filename" FAILED<<<<<";passed=1;}
    }
    else{
      if(verbose)print "No filename";
      passed=1;
    }
  }
  if(key=="<**")filename=$4;
  else{filename=$2;}
  if(verbose){print "***Diffing: "$0;}
}
function abs(y) {
  x = sqrt(y*y);return x}
function max(x,y) {
  z=x;if(y>z){z=y;}return z}
