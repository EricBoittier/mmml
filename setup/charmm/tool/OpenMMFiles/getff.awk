# This file reads the amber*.xml and amber*_obc.xml files from OpenMM
# to extract the residue and atom names with the Amber GBSA OBC
# radii and scale factors.
#
# usage: 
#  cat amber99sb.xml amber99_obc.xml | awk -v tp=[aa|na|ot] -v ff-[amber|charmm] -f getff.awk -
#
# aa = amino acids, na = nucleic acids, ot = other

# This produces the stream files to set the GBSA OBC radii and scale
# factors for calculations using the CHARMM/OpenMM Interface implementation
# of the OPENMM GBSA OBC model.
BEGIN {
    residues = 0; atomtypes = 0; gbsa = 0;ntypes=0; nres=0;
}
{
    if(/<Residues/){residues = 1}
    if(/<\/Residues/){residues = 0}
    if(/<AtomTypes/){atomtypes = 1}
    if(/<\/AtomTypes/){atomtypes = 0}
    if(/<GBSAOBCForce/){gbsa = 1}
    if(/<\/GBSAOBCForce/){gbsa = 0}
    if(atomtypes) {
	if(/<Type/) {
	    split($0,line); ntypes++;
	    itype = getvar(line[2]); atype[itype+1] = itype;
	    aname[itype+1]=getvar(line[3]);
	    radius[itype+1]=0;scale[itype+1]=0;
	    
	}
    }
    if(residues) {
	if(/<Residue /) {nres++;nat=0;
	    split($0,resi); rname[nres]=getvar(substr(resi[2],1,length(resi[2])-1));
#	    print "Processing ",rname[nres];
	}
	if(/<Atom/) {
	    nat++;split($0,resatm)
	    anm[nres,nat]=getvar(resatm[2]);
	    artyp[nres,nat]=getvar(substr(resatm[3],1,length(resatm[3])-2));
#	    print artyp[nres,nat],resatm[3];
	    natres[nres]=nat;
	}
    }
    if(gbsa){
	if(/<Atom/) {
#	    print $0;
	    split($0,gbsaline);
	    itype=getvar(gbsaline[2]); 
	    radius[itype+1] = getvar(gbsaline[4])*10;
	    scale[itype+1] = getvar(substr(gbsaline[5],1,length(gbsaline[5])-2));
	}
    }
}

function getvar(word) {
    x=substr(word,match(word,/=/)+2,length(word)-(match(word,/=/)+2));
    return x }
function protname(name) {
    x = 0;
    if (ff == "amber" &&			\
	(					\
	    name== "ACE"  ||			\
	    name== "ASH"  ||			\
	    name== "CALA"  ||			\
	    name== "CARG"  ||			\
	    name== "CASN"  ||			\
	    name== "CASP"  ||			\
	    name== "CCYS"  ||			\
	    name== "CCYX"  ||			\
	    name== "CGLN"  ||			\
	    name== "CGLU"  ||			\
	    name== "CGLY"  ||			\
	    name== "CHID"  ||			\
	    name== "CHIE"  ||			\
	    name== "CHIP"  ||			\
	    name== "CILE"  ||			\
	    name== "CLEU"  ||			\
	    name== "CLYS"  ||			\
	    name== "CMET"  ||			\
	    name== "CPHE"  ||			\
	    name== "CPRO"  ||			\
	    name== "CSER"  ||			\
	    name== "CTHR"  ||			\
	    name== "CTRP"  ||			\
	    name== "CTYR"  ||			\
	    name== "CVAL"  ||			\
	    name== "CYM"  ||			\
	    name== "CYX"  ||			\
	    name== "GLH"  ||			\
	    name== "HYP"  ||			\
	    name== "LYN"  ||			\
	    name== "NALA"  ||			\
	    name== "NARG"  ||			\
	    name== "NASN"  ||			\
	    name== "NASP"  ||			\
	    name== "NCYS"  ||			\
	    name== "NCYX"  ||			\
	    name== "NGLN"  ||			\
	    name== "NGLU"  ||			\
	    name== "NGLY"  ||			\
	    name== "NHE"  ||			\
	    name== "NHID"  ||			\
	    name== "NHIE"  ||			\
	    name== "NHIP"  ||			\
	    name== "NILE"  ||			\
	    name== "NLEU"  ||			\
	    name== "NLYS"  ||			\
	    name== "NME"  ||			\
	    name== "NMET"  ||			\
	    name== "NPHE"  ||			\
	    name== "NPRO"  ||			\
	    name== "NSER"  ||			\
	    name== "NTHR"  ||			\
	    name== "NTRP"  ||			\
	    name== "NTYR"  ||			\
	    name== "NVAL"  ||			\
	    name== "OHE"			\
	    )					\
	){x = 1;};
    if( ( ff == "amber" || ff == "charmm" ) &&		\
	(						\
	    name== "ALA"  ||				\
	    name== "ARG"  ||				\
	    name== "ASN"  ||				\
	    name== "ASP"  ||				\
	    name== "CYS"  ||				\
	    name== "GLN"  ||				\
	    name== "GLU"  ||				\
	    name== "GLY"  ||				\
	    name== "HID"  ||				\
	    name== "HIE"  ||				\
	    name== "HIP"  ||				\
	    name== "ILE"  ||				\
	    name== "LEU"  ||				\
	    name== "LYS"  ||				\
	    name== "MET"  ||				\
	    name== "PHE"  ||				\
	    name== "PRO"  ||				\
	    name== "SER"  ||				\
	    name== "THR"  ||				\
	    name== "TRP"  ||				\
	    name== "TYR"  ||				\
	    name== "VAL"				\
	    )						\
	){x = 1;};
    return x }
function nuclname(name) {
    x = 0;
    if(	ff == "amber" &&			\
	(					\
	    name== "A"  ||			\
	    name== "A3"  ||			\
	    name== "A5"  ||			\
	    name== "AN"  ||			\
	    name== "C"  ||			\
	    name== "C3"  ||			\
	    name== "C5"  ||			\
	    name== "CN"  ||			\
	    name== "DA"  ||			\
	    name== "DA3"  ||			\
	    name== "DA5"  ||			\
	    name== "DAN"  ||			\
	    name== "DC"  ||			\
	    name== "DC3"  ||			\
	    name== "DC5"  ||			\
	    name== "DCN"  ||			\
	    name== "DG"  ||			\
	    name== "DG3"  ||			\
	    name== "DG5"  ||			\
	    name== "DGN"  ||			\
	    name== "DT"  ||			\
	    name== "DT3"  ||			\
	    name== "DT5"  ||			\
	    name== "DTN"  ||			\
	    name== "G"  ||			\
	    name== "G3"  ||			\
	    name== "G5"  ||			\
	    name== "GN"  ||			\
	    name== "U"  ||			\
	    name== "U3"  ||			\
	    name== "U5"  ||			\
	    name== "UN"				\
	    )					\
	){x = 1;}
    return x }
function ionname(name) {
    x = 0;
    if(	ff == "amber" &&			\
	(					\
	    name== "Br-"  ||			\
	    name== "Cl-"  ||			\
	    name== "Cs+"  ||			\
	    name== "F-"  ||			\
	    name== "I-"  ||			\
	    name== "K+"  ||			\
	    name== "Mg+"  ||			\
	    name== "Na+"			\
	    )					\
	){x = 1;}
    return x }
function getname(name) {
    x = 0;
    if(tp == "aa") {x = protname(name);}
    if(tp == "na") {x = nuclname(name);}
    if(tp == "ot") {x = ionname(name);}
    return x }
function fixrname(name) {
    x = name;
    if(ff == "charmm" ) {
	if(name == "HID"){x = "HSD"}
	if(name == "HIE"){x = "HSE"}
	if(name == "HIP"){x = "HSP"}
    }
    return x }

function prologue() {
    print "* Stream file for Amber GBSA OBC radii and scale factors";
    print "*";
    print " ";
    print "! Preset all to vdW radii and scale -1";
    print "scalar wmain = radius select all end";
    print "scalar wmain mult -1 select all end";
    print "scalar wcomp set -1 select all end";
    print " ";
}
function postlogue() {
    print "! #### Test whether any atoms have not had radii or scale assigned ####";
    print "define missed select property wmain .le. 0 end";
    print "set numns = ?nsel";
    print "define missed select missed .or. property wcomp .le. 0 end";
    print "calc numns = @numns + ?nsel";
    print "prnlev 5";
    print "if @numns .gt. 0 then";
    print "      echo gbsa_obc radii or scale not assigned";
    print "      scalar wmain show select property wmain .le. 0 end";
    print "      scalar wcomp show select property wcomp .le. 0 end";
    print "!      now bomb";
    print "endif";
}
function doace() {
    print " ";
    print "!*** pres ACE/ACED/ACP/ACPD ***";
    print "scalar wmain set 1.25 select  type HY* end";
    print "scalar wcomp set 0.85 select  type HY* end";
    print "scalar wmain set 1.9 select  type CAY end";
    print "scalar wcomp set 0.72 select  type CAY end";
    print "scalar wmain set 1.875 select  type CY end";
    print "scalar wcomp set 0.72 select  type CY end";
    print "scalar wmain set 1.48 select  type OY end";
    print "scalar wcomp set 0.85 select  type OY end";
}
function doprop() {
    print " ";
    print "!*** PROP ***";
    print "define prop select type  hn2 .or. type hn1 end    ! look for prop";
    print "define prop select prop .and. type hn1 end                 ! Number of N-termini";
    print "set nNP = ?nsel";
    print "set ncnt = 0";
    print "label doNP";
    print "      set thisNP = ?selires";
    print "      scalar wmain set 1.625 select ires @thisNP .and. type n end";
    print "      scalar wcomp set 0.79 select ires @thisNP .and. type n end";
    print "      scalar wmain set 1.15 select ires @thisNP .and. type hn* end";
    print "      scalar wcomp set 0.85 select ires @thisNP .and. type hn* end";
    print "      define prop select prop .and. .not. ires @thisNP end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nNP goto doNP";
}
function donter() {
    print " ";
    print "!*** pres NTER/NNEU/GLYP ***";
    print "define nter select (type  ht*) .and. .not. -! look for N-termini";
    print "            (type ct) end                                  ! ct1";
    print "define nter select nter .and. type ht1 end                 ! Number of N-termini";
    print "set nNT = ?nsel";
    print "set ncnt = 0";
    print "label doNT";
    print "      set thisNT = ?selires";
    print "      scalar wmain set 1.625 select ires @thisNT .and. type n end";
    print "      scalar wcomp set 0.79 select ires @thisNT .and. type n end";
    print "      scalar wmain set 1.15 select ires @thisNT .and. type ht* end";
    print "      scalar wcomp set 0.85 select ires @thisNT .and. type ht* end";
    print "      define nter select nter .and. .not. ires @thisNT end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nNT goto doNT";
}
function docter() {
    print " ";
    print "!*** pres CTER ***";
    print "define cter select (type  ot1 .or. type ot2) .and. .not. - ! look for C-termini";
    print "            (type ht2 .or. -                               ! nneu";
    print "            (type ct .or. type ht*) ) end                 ! ct1";
    print "define cter select cter .and. type ot1 end                 ! Number of C-termini";
    print "set nCT = ?nsel";
    print "set ncnt = 0";
    print "label doCT";
    print "      set thisCT = ?selires";
    print "      scalar wmain set 1.48 select ires @thisCT .and. type ot* end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type ot* end";
    print "      define cter select cter .and. .not. ires @thisCT end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nCT goto doCT";
}
function docneu() {
    print " ";
    print "!*** pres CNEU ***";
    print "define cneu select .byres. ( (type  ot1 .or. type ot2) .and. .not. - ! look for C-termini";
    print "            (type ct) )show end                                  ! ct1";
    print "define cneu select type ht2 .and. cneu end                 ! Number of CNEU-termini";
    print "set nCN = ?nsel";
    print "set ncnt = 0";
    print "label doCN";
    print "      set thisCN = ?selires";
    print "      scalar wmain set 1.535 select ires @thisCN .and. type ot2 end";
    print "      scalar wmain set 1.25 select ires @thisCN .and. type ht2 end";
    print "      scalar wcomp set 0.85 select ires @thisCN .and. type ht2 end";
    print "      define cneu select cneu .and. .not. ires @thisCN end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nCN goto doCN";
}
function doct1() {
    print " ";
    print "!*** pres CT1 ***";
    print "define ct1 select .byres. ((type  ot1 .or. type ot2) .or.       - ! look for CT1";
    print "            (type ct .or. type ht*) ) end                   ! ct1";
    print "define ct1 select ct1 .and. type ct end                 ! Number of C-termini";
    print "set nCT = ?nsel";
    print "set ncnt = 0";
    print "label doCT1";
    print "      set thisCT = ?selires";
    print "      scalar wmain set 1.48 select ires @thisCT .and. type ot1 end";
    print "      scalar wmain set 1.535 select ires @thisCT .and. type ot2 end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type ot* end";
    print "      scalar wmain set 1.9 select ires @thisCT .and. type ct end";
    print "      scalar wcomp set 0.72 select ires @thisCT .and. type ct end";
    print "      scalar wmain set 1.25 select ires @thisCT .and. type ht* end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type ht* end";
    print "      define ct1 select ct1 .and. .not. ires @thisCT end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nCT goto doCT1";
}
function doct2() {
    print " ";
    print "!*** pres CT2 ***";
    print "define ct2 select .byres. ( (type  nt .or. type ht*)             - ! look for CT2";
    print "        .and. .not. type cat ) end                          ! ct2";
    print "define ct2 select ct2 .and. type nt end                   ! Number of CT2";
    print "set nCT = ?nsel";
    print "set ncnt = 0";
    print "label doCT2";
    print "      set thisCT = ?selires";
    print "      scalar wmain set 1.625 select ires @thisCT .and. type nt end";
    print "      scalar wcomp set 0.79 select ires @thisCT .and. type nt end";
    print "      scalar wmain set 1.15 select ires @thisCT .and. type ht* end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type ht* end";
    print "      define ct2 select ct2 .and. .not. ires @thisCT end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nCT goto doCT2";
}
function doct3() {
    print " ";
    print "!*** pres CT3 ***";
    print "define ct3 select .byres. ( (type  nt .or. type ht*)             - ! look for CT3";
    print "        .or. type cat ) end                                ! ct3";
    print "define ct3 select ct3 .and. type nt end                   ! Number of CT3";
    print "set nCT = ?nsel";
    print "set ncnt = 0";
    print "label doCT3";
    print "      set thisCT = ?selires";
    print "      scalar wmain set 1.625 select ires @thisCT .and. type nt end";
    print "      scalar wcomp set 0.79 select ires @thisCT .and. type nt end";
    print "      scalar wmain set 1.15 select ires @thisCT .and. type hnt* end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type hnt* end";
    print "      scalar wmain set 1.9 select ires @thisCT .and. type cat end";
    print "      scalar wcomp set 0.72 select ires @thisCT .and. type cat end";
    print "      scalar wmain set 1.25 select ires @thisCT .and. type ht* end";
    print "      scalar wcomp set 0.85 select ires @thisCT .and. type ht* end";
    print "      define ct3 select ct3 .and. .not. ires @thisCT end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nCT goto doCT3";
}
function doaspp() {
    print " ";
    print "!*** Look for ASPP patches ***";
    print "define aspP select resname asp .and. type hd2 end";
    print "set nDH = ?nsel";
    print "set ncnt = 0";
    print "define aspP select none end";
    print "label doDH";
    print "      define thisOne select resname asp .and. type hd2 .and. .not. aspP end";
    print "      set thisDH = ?selires";
    print "      scalar wmain set 1.535 select ires @thisDH .and. type OD2 end";
    print "      scalar wcomp set 0.85 select ires @thisDH .and. type OD2 end";
    print "      scalar wmain set 1.05 select ires @thisDH .and. type HD2 end";
    print "      scalar wcomp set 0.85 select ires @thisDH .and. type HD2 end";
    print "      define aspP select aspP .or. ires @thisDH end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nDH goto doDH";
}
function doglup() {
    print " ";
    print "!*** Look for GLUP patches ***";
    print "define gluP select resname glu .and. type he2 end";
    print "set nEH = ?nsel";
    print "set ncnt = 0";
    print "define gluP select none end";
    print "label doEH";
    print "      define thisOne select resname glu .and. type he2 .and. .not. gluP end";
    print "      set thisEH = ?selires";
    print "      scalar wmain set 1.535 select ires @thisEH .and. type OE2 end";
    print "      scalar wcomp set 0.85 select ires @thisEH .and. type OE2 end";
    print "      scalar wmain set 1.05 select ires @thisEH .and. type HE2 end";
    print "      scalar wcomp set 0.85 select ires @thisEH .and. type HE2 end";
    print "      define gluP select gluP .or. ires @thisEH end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nEH goto doEH";
}
function dolsn() {
    print " ";
    print "!*** pres LSN ***";
    print "define lsn select (type nz .or. type  hz*) .and. .not. -! look for neutral K";
    print "            (type hz3) end                               ! K";
    print "define lsn select lsn .and. type nz end                 ! Number of neutral K";
    print "set nNK = ?nsel";
    print "set ncnt = 0";
    print "label doNK";
    print "      set thisNK = ?selires";
    print "      scalar wmain set 1.625 select ires @thisNK .and. type nz end";
    print "      scalar wcomp set 0.79 select ires @thisNK .and. type nz end";
    print "      scalar wmain set 1.15 select ires @thisNK .and. type hz* end";
    print "      scalar wcomp set 0.85 select ires @thisNK .and. type hz* end";
    print "      define lsn select lsn .and. .not. ires @thisNK end";
    print "      incr ncnt by 1";
    print "if @ncnt .le. @nNK goto doNK";
}
function fixanm(an,rn) {
    if( ff == "charmm" ) {
	if(an == "H"){an="HN"}
	if(rn=="GLY"){if(an=="HA3"){an="HA1"}}
	if(rn=="ILE"){
	    if(an=="CD1"){an="CD"}
	    if(an=="HG13"){an="HG11"}
	    if(an=="HD11"){an="HD1"}
	    if(an=="HD12"){an="HD2"}
	    if(an=="HD13"){an="HD3"}
	}
	if(rn=="LYS"){if(an=="HE3"){an="HE1"}}
	if(rn=="ARG" || rn=="LYS" || rn=="PRO"){
	    if(an=="HD3"){an="HD1"}
	}
	if(rn=="ARG" || rn=="GLN" || rn=="GLU" ||		\
	   rn=="LYS" ||						\
	   rn=="MET" || rn=="PRO"){
	    if(an=="HG3"){an="HG1"}
	}
	if(rn=="SER" || rn=="CYS"){if(an=="HG"){an="HG1"}}
	if(rn=="ARG" || rn=="ASN" || rn=="ASP" || rn=="CYS" || rn=="GLN" || rn=="GLU" || \
	   rn=="HSD" || rn=="HSE" || rn=="HSP" || rn=="LEU" || rn=="LYS" || \
	   rn=="MET" || rn=="PHE" || rn=="PRO" || rn=="SER" || rn=="TRP" || rn=="TYR")
	{
	    if(an=="HB3"){an="HB1";}
	}
    }
    return an
}				     
function residue_RadiusScale(i) {
    if(getname(rname[i])){
	rname[i] = fixrname(rname[i]);
	print "!##### Values For Residue " rname[i] " #####";
	for(j=1;j<=natres[i];j++) {
	    itype = artyp[i,j];
	    anm[i,j] = fixanm(anm[i,j],rname[i]);
	    print anm[i,j],radius[itype+1],scale[itype+1] >"look.dat";
	    if(radius[itype+1] > 0) {
		print "scalar wmain set " radius[itype+1]		\
		    " select resname " rname[i]				\
		    " .and. type " anm[i,j] " end";
	    }
# Do scale factor
	    if(scale[itype+1] > 0) {
		print "scalar wcomp set " scale[itype+1]		\
		    " select resname " rname[i]				\
		    " .and. type " anm[i,j] " end";
	    }
	}
    }
    return
}
END {
# Write the stream file prologue
    prologue();
# Get Residues
    for(i=1;i<=nres;i++) {
	residue_RadiusScale(i);
    }
# Write special cases for all patches/exceptions know of
    if( ff == "charmm" ) {
	doace();
	doprop();
	donter();
	docter();
	docneu();
	doct1();
	doct2();
	doct3();
	doaspp();
	doglup();
	dolsn();
    }
# Now do the postlog
    postlogue();
}
