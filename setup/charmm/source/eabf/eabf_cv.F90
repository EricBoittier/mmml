!-----------------------------------------------------------
 subroutine EABF_CV_PARAMETERS(COMLYN,COMLEN)
!----------------------------------------------------------
 use eabfsrc
 use dimens_fcm
 use exfunc
 use number
 use stream
 use psf
 use coord
 use coordc
 use deriv
 use reawri
 use consta
 use contrl
 use chm_kinds
 use memory
 use string
 use image

 implicit none
    CHARACTER(len=*), intent(in) :: comlyn
    integer :: i,ii
    INTEGER, intent(in) :: comlen
    CHARACTER(LEN=4) WRD

    WRD=NEXTA4(COMLYN,COMLEN)

    IF(WRD == 'DIST') THEN
       write(*,*)"eABF Distance CV Parameters Follow"
       write(*,*)"Two selections should be made to define the atoms/group of atoms to measure a distance between. Center of mass distance is used."
       write(*,*)"Current Number of CVs Employed",eabf_cv_count+1
       !Set Current CV as Distance
       cv_array(eabf_cv_count)=1 !1 is specific to distance
       cv_num_lists(eabf_cv_count)=2 !Distance Requires 2 selections

       lambd_temp=GTRMF(COMLYN,COMLEN,'TLAM',lambd_temp)
       lambda_temps(eabf_cv_count)=lambd_temp

       lambd_mass=GTRMF(COMLYN,COMLEN,'MLAM',lambd_mass)
       lambda_mass(eabf_cv_count)=lambd_mass

       lambd_k=GTRMF(COMLYN,COMLEN,'KLAM',lambd_k)
       lambda_k(eabf_cv_count)=lambd_k
       
       lambd_frict=GTRMF(COMLYN,COMLEN,'LFRC',lambd_frict)
       lambda_frict(eabf_cv_count)=lambd_frict

       lambd_min=GTRMF(COMLYN,COMLEN,'LMIN',lambd_min)
       lambda_min(eabf_cv_count)=lambd_min

       lambd_max=GTRMF(COMLYN,COMLEN,'LMAX',lambd_max)
       lambda_max(eabf_cv_count)=lambd_max

       lambd_bin=GTRMI(COMLYN,COMLEN,'LBIN',lambd_bin)
       lambda_bin_num(eabf_cv_count)=lambd_bin

       lambda_bin_width(eabf_cv_count)=1.0/(lambd_bin+0.0)

       lambd_bnd=GTRMF(COMLYN,COMLEN,'LBND',lambd_bnd)
       lambda_boundary(eabf_cv_count)=lambd_bnd

       !alph=GTRMF(COMLYN,COMLEN,'ALPH',alph)
       !betaa=GTRMF(COMLYN,COMLEN,'BETA',betaa)
       f_alpha(eabf_cv_count)=1.0
       !f_beta(eabf_cv_count)=betaa


       memz=GTRMI(COMLYN,COMLEN,'MEMZ',memz)

       ListA(:)=0.0
       ListB(:)=0.0
       ListC(:)=0.0
       !Read in Selections
       call getatomselected2(COMLYN,COMLEN,NatomA,ListA)
       call getatomselected2(COMLYN,COMLEN,NatomB,ListB)
       call getatomselected2(COMLYN,COMLEN,NatomC,ListC)
       !Transfer Selection to CV Lists
       do i=1,NatomA
          cv_selection_lists(eabf_cv_count,0,i)=ListA(i)
       enddo
       cv_natom_lists(eabf_cv_count,0)=NatomA
       do i=1,NatomB
          cv_selection_lists(eabf_cv_count,1,i)=ListB(i)
       enddo
       cv_natom_lists(eabf_cv_count,1)=NatomB

       !Increase the CV Count
       eabf_cv_count=eabf_cv_count+1
    ENDIF

    IF(WRD == 'RMSD') THEN
       write(*,*)"eABF RMSD CV Parameters Follow"
       write(*,*)"One selection should be made that matches (atom-for-atom) the selection made with the SREF command above to measure RMSD"
       write(*,*)"Current Number of CVs Employed",eabf_cv_count+1
       !Set Current CV as RMSD
       cv_array(eabf_cv_count)=2 !2 is specific to RMSD
       cv_num_lists(eabf_cv_count)=1 !RMSD Requires 1 selection

       lambd_temp=GTRMF(COMLYN,COMLEN,'TLAM',lambd_temp)
       lambda_temps(eabf_cv_count)=lambd_temp

       lambd_mass=GTRMF(COMLYN,COMLEN,'MLAM',lambd_mass)
       lambda_mass(eabf_cv_count)=lambd_mass

       lambd_k=GTRMF(COMLYN,COMLEN,'KLAM',lambd_k)
       lambda_k(eabf_cv_count)=lambd_k

       lambd_frict=GTRMF(COMLYN,COMLEN,'LFRC',lambd_frict)
       lambda_frict(eabf_cv_count)=lambd_frict

       lambd_min=GTRMF(COMLYN,COMLEN,'LMIN',lambd_min)
       lambda_min(eabf_cv_count)=lambd_min

       lambd_max=GTRMF(COMLYN,COMLEN,'LMAX',lambd_max)
       lambda_max(eabf_cv_count)=lambd_max

       lambd_bin=GTRMI(COMLYN,COMLEN,'LBIN',lambd_bin)
       lambda_bin_num(eabf_cv_count)=lambd_bin

       lambda_bin_width(eabf_cv_count)=1.0/(lambd_bin+0.0)

       lambd_bnd=GTRMF(COMLYN,COMLEN,'LBND',lambd_bnd)
       lambda_boundary(eabf_cv_count)=lambd_bnd

       !alph=GTRMF(COMLYN,COMLEN,'ALPH',alph)
       !betaa=GTRMF(COMLYN,COMLEN,'BETA',betaa)
       f_alpha(eabf_cv_count)=1.0
       !f_beta(eabf_cv_count)=betaa

       !ALLOCATE (ListA(1:natom))
       ListA(:)=0.0
       !Read in Selections
       call getatomselected2(COMLYN,COMLEN,NatomA,ListA)
       !Transfer Selection to CV Lists
       do i=1,NatomA
          cv_selection_lists(eabf_cv_count,0,i)=ListA(i)
       enddo

       cv_natom_lists(eabf_cv_count,0)=NatomA
       eabf_cv_count=eabf_cv_count+1
    ENDIF

    IF(WRD == 'DISD') THEN
       write(*,*)"eABF Distance Difference CV Parameters Follow"
       write(*,*)"Four selections should be made to define the atoms/group of atoms to measure the distance between. Center of mass distance is used."
       write(*,*)"The distance difference is distance_1-distance_2, therefore a negative distance difference can occur. Select the LMAX and LMIN accordingly."
       write(*,*)"Current Number of CVs Employed",eabf_cv_count+1
       !Set Current CV as Distance
       cv_array(eabf_cv_count)=3 !3 is specific to distance difference
       cv_num_lists(eabf_cv_count)=4 !Distance difference Requires 4 selections

       lambd_temp=GTRMF(COMLYN,COMLEN,'TLAM',lambd_temp)
       lambda_temps(eabf_cv_count)=lambd_temp

       lambd_mass=GTRMF(COMLYN,COMLEN,'MLAM',lambd_mass)
       lambda_mass(eabf_cv_count)=lambd_mass

       lambd_k=GTRMF(COMLYN,COMLEN,'KLAM',lambd_k)
       lambda_k(eabf_cv_count)=lambd_k

       lambd_frict=GTRMF(COMLYN,COMLEN,'LFRC',lambd_frict)
       lambda_frict(eabf_cv_count)=lambd_frict

       lambd_min=GTRMF(COMLYN,COMLEN,'LMIN',lambd_min)
       lambda_min(eabf_cv_count)=lambd_min

       lambd_max=GTRMF(COMLYN,COMLEN,'LMAX',lambd_max)
       lambda_max(eabf_cv_count)=lambd_max

       lambd_bin=GTRMI(COMLYN,COMLEN,'LBIN',lambd_bin)
       lambda_bin_num(eabf_cv_count)=lambd_bin

       lambda_bin_width(eabf_cv_count)=1.0/(lambd_bin+0.0)

       lambd_bnd=GTRMF(COMLYN,COMLEN,'LBND',lambd_bnd)
       lambda_boundary(eabf_cv_count)=lambd_bnd

       !alph=GTRMF(COMLYN,COMLEN,'ALPH',alph)
       !betaa=GTRMF(COMLYN,COMLEN,'BETA',betaa)
       f_alpha(eabf_cv_count)=1.0
       !f_beta(eabf_cv_count)=betaa

       ListA(:)=0.0
       ListB(:)=0.0
       ListC(:)=0.0
       ListD(:)=0.0
       !Read in Selections
       call getatomselected2(COMLYN,COMLEN,NatomA,ListA)
       call getatomselected2(COMLYN,COMLEN,NatomB,ListB)
       call getatomselected2(COMLYN,COMLEN,NatomC,ListC)
       call getatomselected2(COMLYN,COMLEN,NatomD,ListD)
       !Transfer Selection to CV Lists
       do i=1,NatomA
          cv_selection_lists(eabf_cv_count,0,i)=ListA(i)
       enddo
       cv_natom_lists(eabf_cv_count,0)=NatomA
       do i=1,NatomB
          cv_selection_lists(eabf_cv_count,1,i)=ListB(i)
       enddo
       cv_natom_lists(eabf_cv_count,1)=NatomB
       do i=1,NatomC
          cv_selection_lists(eabf_cv_count,2,i)=ListC(i)
       enddo
       cv_natom_lists(eabf_cv_count,2)=NatomC
       do i=1,NatomD
          cv_selection_lists(eabf_cv_count,3,i)=ListD(i)
       enddo
       cv_natom_lists(eabf_cv_count,3)=NatomD

       !Increase the CV Count
       eabf_cv_count=eabf_cv_count+1
    ENDIF

    IF(WRD == 'RADG') THEN
       write(*,*)"eABF Radius of Gyration CV Parameters Follow"
       write(*,*)"A single selection should be made to define the atoms to measure the Radius of Gyration. Center of mass distance is used."
       write(*,*)"Current Number of CVs Employed",eabf_cv_count+1
       !Set Current CV as Distance
       cv_array(eabf_cv_count)=4 !5 is specific to radius of gyration
       cv_num_lists(eabf_cv_count)=1 !Radius of gyration requires 1 selection

       lambd_temp=GTRMF(COMLYN,COMLEN,'TLAM',lambd_temp)
       lambda_temps(eabf_cv_count)=lambd_temp

       lambd_mass=GTRMF(COMLYN,COMLEN,'MLAM',lambd_mass)
       lambda_mass(eabf_cv_count)=lambd_mass

       lambd_k=GTRMF(COMLYN,COMLEN,'KLAM',lambd_k)
       lambda_k(eabf_cv_count)=lambd_k

       lambd_frict=GTRMF(COMLYN,COMLEN,'LFRC',lambd_frict)
       lambda_frict(eabf_cv_count)=lambd_frict

       lambd_min=GTRMF(COMLYN,COMLEN,'LMIN',lambd_min)
       lambda_min(eabf_cv_count)=lambd_min

       lambd_max=GTRMF(COMLYN,COMLEN,'LMAX',lambd_max)
       lambda_max(eabf_cv_count)=lambd_max

       lambd_bin=GTRMI(COMLYN,COMLEN,'LBIN',lambd_bin)
       lambda_bin_num(eabf_cv_count)=lambd_bin

       lambda_bin_width(eabf_cv_count)=1.0/(lambd_bin+0.0)

       lambd_bnd=GTRMF(COMLYN,COMLEN,'LBND',lambd_bnd)
       lambda_boundary(eabf_cv_count)=lambd_bnd

       !alph=GTRMF(COMLYN,COMLEN,'ALPH',alph)
       !betaa=GTRMF(COMLYN,COMLEN,'BETA',betaa)
       f_alpha(eabf_cv_count)=1.0
       !f_beta(eabf_cv_count)=betaa

       ListA(:)=0.0
       !Read in Selections
       call getatomselected2(COMLYN,COMLEN,NatomA,ListA)
       !Transfer Selection to CV Lists
       do i=1,NatomA
          cv_selection_lists(eabf_cv_count,0,i)=ListA(i)
       enddo
       cv_natom_lists(eabf_cv_count,0)=NatomA

       !Increase the CV Count
       eabf_cv_count=eabf_cv_count+1
    ENDIF

END SUBROUTINE

!-----------------------------------------------------------
SUBROUTINE CALC_XI(curr_cv)
!----------------------------------------------------------
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
use energym
implicit none

integer :: curr_cv,i,ii

real(chm_real) ::a_1_ab_vec_x,a_2_ab_vec_y,a_3_ab_vec_z,b_1_ac_vec_x,b_2_ac_vec_y,b_3_ac_vec_z,a_1_dc_vec_x,a_2_dc_vec_y,a_3_dc_vec_z,&
                   b_1_db_vec_x,b_2_db_vec_y,b_3_db_vec_z,plane_1_norm_x,plane_1_norm_y,plane_1_norm_z,plane_2_norm_x,plane_2_norm_y,&
                   plane_2_norm_z,dihed_angle,vec_x_23,vec_y_23,vec_z_23,norm_cross_x,norm_cross_y,norm_cross_z,&
                   dihed_angle_sign,dist_sum

real(chm_real) FX,FY,FZ,GX,GY,GZ,HX,HY,HZ
real(chm_real) AX,AY,AZ,BX,BY,BZ,RA2,RB2,RA2R,RB2R,RG2,RG,RGR,RGR2
real(chm_real) RABR,CP,AP,SP,E,DF,DDF,CA,SA,ARG,APR

!Calc Distance Based XI_CV
if(cv_array(curr_cv) .EQ. 1)then
   !Transfer Current Selections to ListA & ListB for Distance Calculation
   NatomA=cv_natom_lists(curr_cv,0)
   do i=1,cv_natom_lists(curr_cv,0)
      ListA(i)=cv_selection_lists(curr_cv,0,i)
   enddo
   NatomB=cv_natom_lists(curr_cv,1)
   do i=1,cv_natom_lists(curr_cv,1)
      ListB(i)=cv_selection_lists(curr_cv,1,i)
   enddo

   call calc_com(NatomA,ListA,mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp)
   call calc_com(NatomB,ListB,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp)
   call find_dist(mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp,mcen_dist_tmp)

   mcen_XA(curr_cv)=mcen_XA_tmp
   mcen_YA(curr_cv)=mcen_YA_tmp
   mcen_ZA(curr_cv)=mcen_ZA_tmp
   mcen_XB(curr_cv)=mcen_XB_tmp
   mcen_YB(curr_cv)=mcen_YB_tmp
   mcen_ZB(curr_cv)=mcen_ZB_tmp
   mcen_dist(curr_cv)=mcen_dist_tmp
   xi_cv(curr_cv)=(mcen_dist(curr_cv)-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))

   if(memz .EQ. 1)then
      xi_cv(curr_cv)=(ABS(mcen_ZA_tmp)-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))
   endif

   if(memz .EQ. 2)then
      xi_cv(curr_cv)=(mcen_ZA_tmp-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))
   endif

endif

!Calc RMSD Based XI_CV
if(cv_array(curr_cv) .EQ. 2)then
   !Call On-The-Fly eABF RMSD Calculation
   call EABFRMSDOTF(rmsd_ref_count,curr_cv)
   xi_cv(curr_cv)=(eabf_rms(curr_cv)-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))
   rmsd_ref_count=rmsd_ref_count+1
endif

!Calc Distance Difference Based XI_CV
if(cv_array(curr_cv) .EQ. 3)then
   !Transfer Current Selections to ListA, ListB, ListC, & ListD for Distance Calculation
   NatomA=cv_natom_lists(curr_cv,0)
   do i=1,cv_natom_lists(curr_cv,0)
      ListA(i)=cv_selection_lists(curr_cv,0,i)
   enddo
   NatomB=cv_natom_lists(curr_cv,1)
   do i=1,cv_natom_lists(curr_cv,1)
      ListB(i)=cv_selection_lists(curr_cv,1,i)
   enddo
   NatomC=cv_natom_lists(curr_cv,2)
   do i=1,cv_natom_lists(curr_cv,2)
      ListC(i)=cv_selection_lists(curr_cv,2,i)
   enddo
   NatomD=cv_natom_lists(curr_cv,3)
   do i=1,cv_natom_lists(curr_cv,3)
      ListD(i)=cv_selection_lists(curr_cv,3,i)
   enddo

   !Calculate distance 1
   call calc_com(NatomA,ListA,mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp)
   call calc_com(NatomB,ListB,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp)
   call find_dist(mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp,mcen_dist_tmp)

   mcen_XA(curr_cv)=mcen_XA_tmp
   mcen_YA(curr_cv)=mcen_YA_tmp
   mcen_ZA(curr_cv)=mcen_ZA_tmp
   mcen_XB(curr_cv)=mcen_XB_tmp
   mcen_YB(curr_cv)=mcen_YB_tmp
   mcen_ZB(curr_cv)=mcen_ZB_tmp
   mcen_dist(curr_cv)=mcen_dist_tmp

   !Calculate distance 2
   call calc_com(NatomC,ListC,mcen_XC_tmp,mcen_YC_tmp,mcen_ZC_tmp)
   call calc_com(NatomD,ListD,mcen_XD_tmp,mcen_YD_tmp,mcen_ZD_tmp)
   call find_dist(mcen_XC_tmp,mcen_YC_tmp,mcen_ZC_tmp,mcen_XD_tmp,mcen_YD_tmp,mcen_ZD_tmp,mcen_dist_tmp_2)

   mcen_XC(curr_cv)=mcen_XC_tmp
   mcen_YC(curr_cv)=mcen_YC_tmp
   mcen_ZC(curr_cv)=mcen_ZC_tmp
   mcen_XD(curr_cv)=mcen_XD_tmp
   mcen_YD(curr_cv)=mcen_YD_tmp
   mcen_ZD(curr_cv)=mcen_ZD_tmp
   mcen_dist_2(curr_cv)=mcen_dist_tmp_2

   xi_cv(curr_cv)=((mcen_dist(curr_cv)-mcen_dist_2(curr_cv))-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))

endif

!Calc Radius of Gyration Based XI_CV
if(cv_array(curr_cv) .EQ. 4)then
   !Transfer Current Selections to ListA & ListB for Distance Calculation
   NatomA=cv_natom_lists(curr_cv,0)
   do i=1,cv_natom_lists(curr_cv,0)
      ListA(i)=cv_selection_lists(curr_cv,0,i)
   enddo

   call calc_com(NatomA,ListA,mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp)
   rad_g=0.0
   do i=1,cv_natom_lists(curr_cv,0)
      mcen_XB_tmp=X(ListA(i))
      mcen_YB_tmp=Y(ListA(i))
      mcen_ZB_tmp=Z(ListA(i))
      call find_dist(mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp,mcen_dist_tmp)
      rad_g=rad_g+(mcen_dist_tmp*mcen_dist_tmp)
   enddo
   rad_g=rad_g/cv_natom_lists(curr_cv,0)
   rad_g=SQRT(rad_g)

   mcen_XA(curr_cv)=mcen_XA_tmp
   mcen_YA(curr_cv)=mcen_YA_tmp
   mcen_ZA(curr_cv)=mcen_ZA_tmp
   xi_cv(curr_cv)=(rad_g-lambda_min(curr_cv))/(lambda_max(curr_cv)-lambda_min(curr_cv))

endif


END SUBROUTINE

!-----------------------------------------------------------
SUBROUTINE CALC_COM(NumAtom,ListTmp,COM_X,COM_Y,COM_Z)
!----------------------------------------------------------
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none
real(chm_real) :: tx,ty,tz,tm,COM_X,COM_Y,COM_Z
integer :: NumAtom,i,ii
integer,dimension(NATOM) :: ListTmp

tx=0
ty=0
tz=0
tm=0
COM_X=0
COM_Y=0
COM_Z=0

!----Calculate Mass-weighted Coords & Total Mass----!
do i=1,NumAtom
   ii=ListTmp(i)
   tx=tx+(X(ii)*amass(ii))
   ty=ty+(Y(ii)*amass(ii))
   tz=tz+(Z(ii)*amass(ii))
   tm=tm+amass(ii)
enddo

!----Calculate Center of Mass----!
COM_X=(tx/tm)
COM_Y=(ty/tm)
COM_Z=(tz/tm)

END SUBROUTINE

!-----------------------------------------------------------
SUBROUTINE FIND_DIST(COM_XA,COM_YA,COM_ZA,COM_XB,COM_YB,COM_ZB,COM_Dist)
!----------------------------------------------------------
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none
real(chm_real) :: COM_XA,COM_XB,COM_YA,COM_YB,COM_ZA,COM_ZB,COM_Dist

COM_Dist=0.0
COM_Dist=SQRT(((COM_XA-COM_XB)**2)+((COM_YA-COM_YB)**2)+((COM_ZA-COM_ZB)**2))

if(memz .EQ. 1)then
   COM_DIST=SQRT(((COM_ZA-COM_ZB)**2))
endif
if(memz .EQ. 2)then
   COM_DIST=COM_ZA-COM_ZB
endif

END SUBROUTINE

!------------------------------------------------!
 subroutine DISTRIBUTE_FORCE(curr_cv)
!------------------------------------------------!
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none

real(chm_real) :: tma,tmb,duodXA,duodYA,duodZA,duodXB,duodYB,duodZB,&
                  duoldxa,duoldya,duoldza,duoldxb,duoldyb,duoldzb
integer :: a,aa,b,bb,i,ii,iii,iiii,valid_counter,curr_cv

tma=0
tmb=0

!Calculate Total Mass
do a=1,NatomA
   aa=ListA(a)
   tma=tma+amass(aa)
enddo

!Apply Force
do a=1,NatomA
   aa=ListA(a)


!Order Parameter Dependant Term
duoldxa=(1.0)*((mcen_XA(curr_cv)-mcen_XB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldya=(1.0)*((mcen_YA(curr_cv)-mcen_YB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldza=(1.0)*((mcen_ZA(curr_cv)-mcen_ZB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXA=0.0
duodYA=0.0
duodZA=0.0

duodXA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxa)
duodYA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldya)
duodZA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldza)

DX(aa)=DX(aa)+duodXA
DY(aa)=DY(aa)+duodYA
DZ(aa)=DZ(aa)+duodZA

if(memz .EQ. 1)then
   DX(aa)=DX(aa)-duodXA
   DY(aa)=DY(aa)-duodYA
endif
if(memz .EQ. 2)then
   DX(aa)=DX(aa)-duodXA
   DY(aa)=DY(aa)-duodYA
endif

enddo

!Calculate Total Mass
do b=1,NatomB
   bb=ListB(b)
   tmb=tmb+amass(bb)
enddo

!Apply Force
do b=1,NatomB
   bb=ListB(b)

!Order Parameter Dependant Term
duoldxb=(-1.0)*((mcen_XA(curr_cv)-mcen_XB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldyb=(-1.0)*((mcen_YA(curr_cv)-mcen_YB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldzb=(-1.0)*((mcen_ZA(curr_cv)-mcen_ZB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXB=0.0
duodYB=0.0
duodZB=0.0

duodXB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxb)
duodYB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldyb)
duodZB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldzb)

DX(bb)=DX(bb)+duodXB
DY(bb)=DY(bb)+duodYB
DZ(bb)=DZ(bb)+duodZB

if(memz .EQ. 1)then
   DX(bb)=DX(bb)-duodXB
   DY(bb)=DY(bb)-duodYB
endif
if(memz .EQ. 2)then
   DX(bb)=DX(bb)-duodXB
   DY(bb)=DY(bb)-duodYB
endif


enddo
END SUBROUTINE

!------------------------------------------------!
 subroutine DISTRIBUTE_FORCE_2(curr_rmsd_ref,curr_cv)
!------------------------------------------------!
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none

real(chm_real) :: tma,tmb,duodXA,duodYA,duodZA,duodXB,duodYB,duodZB,&
                  duoldxa,duoldya,duoldza,duoldxb,duoldyb,duoldzb
integer :: a,aa,b,bb,i,ii,iii,iiii,valid_counter,curr_cv,curr_rmsd_ref

!Apply Force
do a=1,NatomA
   aa=ListA(a)

!Order Parameter Dependant Term
duoldxa=(X(aa)-rmsd_refs_x(curr_rmsd_ref,a))/(eabf_rms(curr_cv)*NAtomA*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldya=(Y(aa)-rmsd_refs_y(curr_rmsd_ref,a))/(eabf_rms(curr_cv)*NAtomA*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldza=(Z(aa)-rmsd_refs_z(curr_rmsd_ref,a))/(eabf_rms(curr_cv)*NAtomA*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXA=0.0
duodYA=0.0
duodZA=0.0

duodXA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxa)
duodYA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldya)
duodZA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldza)

DX(aa)=DX(aa)+duodXA
DY(aa)=DY(aa)+duodYA
DZ(aa)=DZ(aa)+duodZA

enddo

END SUBROUTINE

!------------------------------------------------!
 subroutine DISTRIBUTE_FORCE_3(curr_cv)
!------------------------------------------------!
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none

real(chm_real) :: tma,tmb,duodXA,duodYA,duodZA,duodXB,duodYB,duodZB,&
                  duoldxa,duoldya,duoldza,duoldxb,duoldyb,duoldzb,&
                  tmc,tmd,duodXC,duodYC,duodZC,duodXD,duodYD,duodZD,&
                  duoldxc,duoldyc,duoldzc,duoldxd,duoldyd,duoldzd
integer :: a,aa,b,bb,c,cc,d,dd,curr_cv

tma=0
tmb=0
tmc=0
tmd=0

!Distribute Force Over Distance 1

!Calculate Total Mass
do a=1,NatomA
   aa=ListA(a)
   tma=tma+amass(aa)
enddo

!Apply Force
do a=1,NatomA
   aa=ListA(a)


!Order Parameter Dependant Term
duoldxa=(1.0)*((mcen_XA(curr_cv)-mcen_XB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldya=(1.0)*((mcen_YA(curr_cv)-mcen_YB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldza=(1.0)*((mcen_ZA(curr_cv)-mcen_ZB(curr_cv))*(amass(aa)/tma))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXA=0.0
duodYA=0.0
duodZA=0.0

duodXA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxa)
duodYA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldya)
duodZA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldza)

DX(aa)=DX(aa)+duodXA
DY(aa)=DY(aa)+duodYA
DZ(aa)=DZ(aa)+duodZA

enddo

!Calculate Total Mass
do b=1,NatomB
   bb=ListB(b)
   tmb=tmb+amass(bb)
enddo

!Apply Force
do b=1,NatomB
   bb=ListB(b)

!Order Parameter Dependant Term
duoldxb=(-1.0)*((mcen_XA(curr_cv)-mcen_XB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldyb=(-1.0)*((mcen_YA(curr_cv)-mcen_YB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldzb=(-1.0)*((mcen_ZA(curr_cv)-mcen_ZB(curr_cv))*(amass(bb)/tmb))/(mcen_dist(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXB=0.0
duodYB=0.0
duodZB=0.0

duodXB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxb)
duodYB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldyb)
duodZB=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldzb)

DX(bb)=DX(bb)+duodXB
DY(bb)=DY(bb)+duodYB
DZ(bb)=DZ(bb)+duodZB

enddo


!Distribute Force Over Distance 2
!Calculate Total Mass
do c=1,NatomC
   cc=ListC(c)
   tmc=tmc+amass(cc)
enddo

!Apply Force
do c=1,NatomC
   cc=ListC(c)

!Order Parameter Dependant Term
duoldxc=(-1.0)*((mcen_XC(curr_cv)-mcen_XD(curr_cv))*(amass(cc)/tmc))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldyc=(-1.0)*((mcen_YC(curr_cv)-mcen_YD(curr_cv))*(amass(cc)/tmc))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldzc=(-1.0)*((mcen_ZC(curr_cv)-mcen_ZD(curr_cv))*(amass(cc)/tmc))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXC=0.0
duodYC=0.0
duodZC=0.0

duodXC=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxc)
duodYC=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldyc)
duodZC=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldzc)

DX(cc)=DX(cc)+duodXC
DY(cc)=DY(cc)+duodYC
DZ(cc)=DZ(cc)+duodZC

enddo

do d=1,NatomD
   dd=ListD(d)
   tmd=tmd+amass(dd)
enddo

!Apply Force
do d=1,NatomD
   dd=ListD(d)

!Order Parameter Dependant Term
duoldxd=(1.0)*((mcen_XC(curr_cv)-mcen_XD(curr_cv))*(amass(dd)/tmd))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldyd=(1.0)*((mcen_YC(curr_cv)-mcen_YD(curr_cv))*(amass(dd)/tmd))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldzd=(1.0)*((mcen_ZC(curr_cv)-mcen_ZD(curr_cv))*(amass(dd)/tmd))/(mcen_dist_2(curr_cv)*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXD=0.0
duodYD=0.0
duodZD=0.0

duodXD=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxd)
duodYD=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldyd)
duodZD=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldzd)

DX(dd)=DX(dd)+duodXD
DY(dd)=DY(dd)+duodYD
DZ(dd)=DZ(dd)+duodZD

enddo

END SUBROUTINE

!------------------------------------------------!
 subroutine DISTRIBUTE_FORCE_4(curr_cv)
!------------------------------------------------!
use eabfsrc
use dimens_fcm
use exfunc
use number
use stream
use psf
use coord
use coordc
use deriv
use reawri
use consta
use contrl
use chm_kinds
use memory
use string
use parallel
implicit none

real(chm_real) :: duodXA,duodYA,duodZA,duodXB,duodYB,duodZB,&
                  duoldxa,duoldya,duoldza,duoldxb,duoldyb,duoldzb
integer :: a,aa,curr_cv

!Apply Force
do a=1,NatomA
   aa=ListA(a)

!Assign Current Atom to Mass Center B
mcen_XB(curr_cv)=X(aa)
mcen_YB(curr_cv)=Y(aa)
mcen_ZB(curr_cv)=Z(aa)

!Order Parameter Dependant Term
duoldxa=(-1.0)*((mcen_XA(curr_cv)-mcen_XB(curr_cv))/NatomA)/(rad_g*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldya=(-1.0)*((mcen_YA(curr_cv)-mcen_YB(curr_cv))/NatomA)/(rad_g*(lambda_max(curr_cv)-lambda_min(curr_cv)))
duoldza=(-1.0)*((mcen_ZA(curr_cv)-mcen_ZB(curr_cv))/NatomA)/(rad_g*(lambda_max(curr_cv)-lambda_min(curr_cv)))

duodXA=0.0
duodYA=0.0
duodZA=0.0

duodXA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldxa)
duodYA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldya)
duodZA=((-1.0)*lambda_k(curr_cv)*(lambda_particle(curr_cv)-xi_cv(curr_cv))*duoldza)

DX(aa)=DX(aa)+duodXA
DY(aa)=DY(aa)+duodYA
DZ(aa)=DZ(aa)+duodZA

enddo

END SUBROUTINE
