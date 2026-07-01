BEGIN {nlines = 10; iline = 0;}
{
    line[iline] = $0; 
    iline++;
# Process the current stack of nline lines
    if(iline >= nlines) {
	iline--;
	for(i=0;i<=nlines;i++) {

# Are there any characters in the first five lines? If not, this could be a continuation.
	    if(!match(substr(line[i],1,5),"[^ ]")||match(substr(line[i],1,5),"##USE")) {

		if( match(substr(line[i],6,1),"[^ ]") ) { # This is a continuation line

# Remove the symbol in column 6 from this line
		    a = substr(line[i],6,1);
		    line[i] = substr(line[i],1,5)" "substr(line[i],7,length(line[i]));

# Find out which of the previous lines to add the "&" to
		    for(j=i-1;j>=0;j--){

# Is this a non-format/non-control line? If yes, add &
			if( !match(substr(line[j],1,5),"[^ ]") || match(substr(line[j],1,5),"##USE") || match(substr(line[j],1,5),"[0-9]") ) {
# Does the current line contain a !## construct?
			    if(match(line[j],"!##")){
				sub("!##"," \\& !##",line[j]); break;
			    } else {
				if(match(line[j],"!")) {sub("!"," \\& !",line[j]); break;} 
				else {if(length(line[j])){
					    line[j] = line[j]" &"; break;}}
			    }
			}
		    }
		}
	    }
	}
# write the current stack to the out file
	print line[0];
# roll the stack up one slot
	for(i=1;i<=nlines;i++) {line[i-1] = line[i];}
    }
}
# write out current stack at end
END {for(i=0;i<=nlines;i++) {if(length(line[i])) {print line[i];}}}