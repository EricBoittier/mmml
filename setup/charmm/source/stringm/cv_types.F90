! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
! CV_TYPES.MOD
!
! LISTING OF COLLECTIVE VARIABLE TYPES
!
! naming convention: type name is the same as the corresponding module name
! without the `cv_` prefix
module cv_types
#if (KEY_STRINGM==1) /*  automatically protect all code */
      implicit none
       integer, parameter :: &
! posi_x=1, posi_y=2, posi_z=3, ! position (obsolete)
     & posi_com_x=4, posi_com_y=5, posi_com_z=6, & ! com-postion
     & dist_com=7, & ! distance
     & angle_com=8, & ! angle
     & dihe_com=9, & ! dihedral angle
     & anglvec=10, & ! angle between two vectors (possibly in different frames)
     & qcomp_1=11, qcomp_2=12, qcomp_3=13, qcomp_4=14, & ! components of orientation quaternion
     & rmsd=15, & ! RMS distance between a set of atoms
     & drmsd=16, & ! difference in the RMSDs from two structures
     & proj=17, & ! projection onto the vector connecting two reference structures; simulation structure used for orientation;
     & cvrms=99 ! rtmd-like variable (currently compatibility limited to steered dynamics): z=sqrt( 1/N sum^N_i (z_i - z^0_i)^2 )
#endif /* automatically protect all code */
end module cv_types
!
