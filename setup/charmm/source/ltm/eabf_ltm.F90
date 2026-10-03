module eabfsrc
   use dimens_fcm
   use chm_kinds

   integer,save::eabf_control,eabf_cv_count,osop,lambd_bin,bin_num,bin_num_2,sorder,recursion_frq,print_frq,restart_frq,restart_step,i_start_lambd,i_end_lambd,spline_center_flag,eabf_rst,symm,sp_gap,sample_cut,var1,var2,var3,var4,recursion_counter,scen,memz,num_particles,min_bin,NatomA,NatomB,NatomC,NatomD,NatomE,NatomF,NatomG,NatomH,NatomI,NatomJ,NatomK,NatomLL,NatomM,NatomN,NatomO,reso,write_flag,rmsd_num_sel,rmsd_ref_count,reporter_only

   integer,allocatable,dimension(:),save::ListA,ListB,ListC,ListD,ListE,ListF,ListG,ListH,ListI,ListJ,ListK,ListL,ListM,ListN,ListO,rot_selct,lambda_bin_num,cv_array,cv_num_lists,rmsd_list_tmp

   integer,allocatable,dimension(:,:),save :: pair_array,cv_natom_lists
   
   integer,allocatable,dimension(:,:,:),save :: cv_selection_lists

   real(chm_real),save::alph,betaa,fe_min,lambd_temp,lambd_mass,lambd_k,lambd_frict,lambd_min,lambd_max,lambd_bnd,test_value,duodub,poslambda,b_w,pos_particle,f_lambd,u_lambd,n_lambd,com_x_ref,com_y_ref,com_z_ref,mcen_XA_tmp,mcen_YA_tmp,mcen_ZA_tmp,mcen_XB_tmp,mcen_YB_tmp,mcen_ZB_tmp,mcen_dist_tmp,eabf_rms_tmp,mcen_XC_tmp,mcen_YC_tmp,mcen_ZC_tmp,mcen_XD_tmp,mcen_YD_tmp,mcen_ZD_tmp,mcen_dist_tmp_2,rad_g,lambda_weight_sum,eabf_eterm

   real(chm_real),allocatable,dimension(:),save::lambd_weight,d_lambd_weight,X_REF,Y_REF,Z_REF,CURR_X_REF,CURR_Y_REF,CURR_Z_REF,gyr_dist,arnc,ax_0,arf_0,av_half_c,ax_c,arnr,af_c,arf_c,av_half_f,ax_f,particle_dudl,particle_fes_min,particle_force,random_num,arnc_p,ax_0_p,arf_0_p,av_half_c_p,ax_c_p,arnr_p,af_c_p,arf_c_p,av_half_f_p,ax_f_p,particle_force_p,lambda_temps,lambda_particle,op_particle,particle_a,particle_b,particle_c,lambda_k,U_copy,lambda_frict,lambda_mass,xi_cv,lambda_min,lambda_max,lambda_bin_width,lambda_boundary,alpha_c,alpha_r,beta_r,dfdlx,dfdl,mcen_XA,mcen_YA,mcen_ZA,mcen_XB,mcen_YB,mcen_ZB,mcen_dist,eabf_rms,f_alpha,f_beta,mcen_XC,mcen_YC,mcen_ZC,mcen_XD,mcen_YD,mcen_ZD,mcen_dist_2,alph_array,lambda_entropy_weight_sum

   real(chm_real),allocatable,dimension(:,:),save::particle_bin_sum,particle_bin_count,particle_free_energy,entropy_particle_free_energy,particle_bin_avg,bin_sum,bin_avg,bin_count,free_energy,rmsd_refs_x,rmsd_refs_y,rmsd_refs_z,initial_rmsd_refs_x,initial_rmsd_refs_y,initial_rmsd_refs_z,lambda_sum,lambda_count,lambda_avg,czar_free_energy,czar_bin_avg,czar_probability,comp_free_energy,force_entropy_x,force_entropy_y,force_entropy_z,entropy_weight_sum

   real(chm_real),allocatable,dimension(:,:,:),save::spline_weight,force_entropy_count_x,force_entropy_count_y,force_entropy_count_z,force_entropy_cross_xx,force_entropy_cross_xy,force_entropy_cross_xz,force_entropy_cross_yx,force_entropy_cross_yy,force_entropy_cross_yz,force_entropy_cross_zx,force_entropy_cross_zy,force_entropy_cross_zz,entropy_weight_sum_2

   real(chm_real),allocatable,dimension(:,:,:,:,:),save::force_entropy_cross_count_xx,force_entropy_cross_count_xy,force_entropy_cross_count_xz,force_entropy_cross_count_yx,force_entropy_cross_count_yy,force_entropy_cross_count_yz,force_entropy_cross_count_zx,force_entropy_cross_count_zy,force_entropy_cross_count_zz
 
end module

