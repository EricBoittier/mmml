{
    if ( index( $0, "impnon" ) > 0 )
    {print "##USE chm_kinds";print "      implicit none"}
    else {print $0;}
}

