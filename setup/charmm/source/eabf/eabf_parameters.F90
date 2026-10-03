!------------------------------------------------!
 SUBROUTINE READ_EABF_PARAMETERS(COMLYN,COMLEN)
!------------------------------------------------!
 use string
 use eabfsrc
 use psf
 implicit none
 character(len=*), intent(in) :: COMLYN
 integer, intent(in) :: COMLEN


write(*,*)"Reading eABF Parameters"
eabf_control=1
eabf_cv_count=0

write_flag=GTRMI(COMLYN,COMLEN,'WRIT',write_flag)
SCEN=GTRMI(COMLYN,COMLEN,'SCEN',scen) 
sorder=GTRMI(COMLYN,COMLEN,'SORD',sorder)
recursion_frq=GTRMI(COMLYN,COMLEN,'RQFR',recursion_frq)
alph=GTRMF(COMLYN,COMLEN,'ALPH',alph)
betaa=GTRMF(COMLYN,COMLEN,'BETA',betaa)
sample_cut=GTRMI(COMLYN,COMLEN,'SCUT',sample_cut)
OSOP=GTRMI(COMLYN,COMLEN,'OSOP',osop)
print_frq=GTRMI(COMLYN,COMLEN,'PRNT',print_frq)
eabf_rst=GTRMI(COMLYN,COMLEN,'REST',eabf_rst)
restart_frq=GTRMI(COMLYN,COMLEN,'REFR',restart_frq)
num_particles=GTRMI(COMLYN,COMLEN,'NUMP',num_particles)
lambd_bin=GTRMI(COMLYN,COMLEN,'LBIN',lambd_bin)
reporter_only=GTRMI(COMLYN,COMLEN,'PFRP',reporter_only)

recursion_frq=1

write(*,*)"Allocating Memory"
ALLOCATE (spline_weight(0-sorder-5:sorder+5,0:sorder,0:sorder))
ALLOCATE (lambd_weight(0-sorder:lambd_bin+sorder))
ALLOCATE (d_lambd_weight(0-sorder:lambd_bin+sorder))
ALLOCATE (U_copy(0:9))
ALLOCATE (bin_avg(0:num_particles,0:lambd_bin))
ALLOCATE (bin_sum(0:num_particles,0:lambd_bin))
ALLOCATE (bin_count(0:num_particles,0:lambd_bin))
ALLOCATE (free_energy(0:num_particles,0-sorder:lambd_bin+sorder))
ALLOCATE (comp_free_energy(0:num_particles,0-sorder:lambd_bin+sorder))
ALLOCATE (czar_free_energy(0:num_particles,0:lambd_bin))
ALLOCATE (czar_bin_avg(0:num_particles,0:lambd_bin))
ALLOCATE (czar_probability(0:num_particles,0:lambd_bin))
ALLOCATE(arnc(0:num_particles))
ALLOCATE(ax_0(0:num_particles))
ALLOCATE(arf_0(0:num_particles))
ALLOCATE(av_half_c(0:num_particles))
ALLOCATE(ax_c(0:num_particles))
ALLOCATE(arnr(0:num_particles))
ALLOCATE(af_c(0:num_particles))
ALLOCATE(arf_c(0:num_particles))
ALLOCATE(av_half_f(0:num_particles))
ALLOCATE(ax_f(0:num_particles))
ALLOCATE(alpha_c(0:num_particles))
ALLOCATE(alpha_r(0:num_particles))
ALLOCATE(beta_r(0:num_particles))
ALLOCATE(random_num(0:num_particles))
ALLOCATE(f_alpha(0:num_particles))
ALLOCATE(f_beta(0:num_particles))
ALLOCATE(lambda_particle(0:num_particles))
ALLOCATE(lambda_temps(0:num_particles))
ALLOCATE(lambda_k(0:num_particles))
ALLOCATE(lambda_frict(0:num_particles))
ALLOCATE(lambda_mass(0:num_particles))
ALLOCATE(xi_cv(0:num_particles))
ALLOCATE(lambda_min(0:num_particles))
ALLOCATE(lambda_max(0:num_particles))
ALLOCATE(lambda_bin_num(0:num_particles))
ALLOCATE(lambda_bin_width(0:num_particles))
ALLOCATE(lambda_boundary(0:num_particles))
ALLOCATE(particle_dudl(0:num_particles))
ALLOCATE(particle_bin_sum(0:num_particles,0:lambd_bin))
ALLOCATE(particle_bin_count(0:num_particles,0:lambd_bin))
ALLOCATE(particle_bin_avg(0:num_particles,0:lambd_bin))
ALLOCATE(particle_free_energy(0:num_particles,0:lambd_bin))
ALLOCATE(entropy_particle_free_energy(0:num_particles,0:lambd_bin))
ALLOCATE(particle_FES_min(0:num_particles))
ALLOCATE(particle_force(0:num_particles))
ALLOCATE(lambda_sum(0:num_particles,0:lambd_bin))
ALLOCATE(lambda_count(0:num_particles,0:lambd_bin))
ALLOCATE(lambda_avg(0:num_particles,0:lambd_bin))
ALLOCATE(particle_a(0:num_particles))
ALLOCATE(particle_b(0:num_particles))
ALLOCATE(particle_c(0:num_particles))
ALLOCATE(cv_array(0:num_particles))
ALLOCATE(cv_num_lists(0:num_particles))
ALLOCATE(cv_selection_lists(0:num_particles,0:3,0:natom))
ALLOCATE(cv_natom_lists(num_particles,0:3))
ALLOCATE(dfdlx(0:num_particles))
ALLOCATE(dfdl(0:num_particles))
ALLOCATE(ListA(1:natom))
ALLOCATE(ListB(1:natom))
ALLOCATE(ListC(1:natom))
ALLOCATE(ListD(1:natom))
ALLOCATE(mcen_XA(0:num_particles))
ALLOCATE(mcen_YA(0:num_particles))
ALLOCATE(mcen_ZA(0:num_particles))
ALLOCATE(mcen_XB(0:num_particles))
ALLOCATE(mcen_YB(0:num_particles))
ALLOCATE(mcen_ZB(0:num_particles))
ALLOCATE(mcen_dist(0:num_particles))
ALLOCATE(mcen_XC(0:num_particles))
ALLOCATE(mcen_YC(0:num_particles))
ALLOCATE(mcen_ZC(0:num_particles))
ALLOCATE(mcen_XD(0:num_particles))
ALLOCATE(mcen_YD(0:num_particles))
ALLOCATE(mcen_ZD(0:num_particles))
ALLOCATE(mcen_dist_2(0:num_particles))
ALLOCATE(eabf_rms(0:num_particles))
ALLOCATE(alph_array(0:lambd_bin))
lambd_weight(:)=0.0
d_lambd_weight(:)=0.0
U_copy(:)=0
bin_avg(:,:)=0.0
bin_sum(:,:)=0.0
bin_count(:,:)=0.0
free_energy(:,:)=0.0
comp_free_energy(:,:)=0.0
czar_free_energy(:,:)=0.0
czar_bin_avg(:,:)=0.0
czar_probability(:,:)=0.0
alph_array(:)=0.0
arnc(:)=0.0
ax_0(:)=0.0
arf_0(:)=0.0
av_half_c(:)=0.0
ax_c(:)=0.0
arnr(:)=0.0
af_c(:)=0.0
arf_c(:)=0.0
av_half_f(:)=0.0
ax_f(:)=0.0
alpha_c(:)=0.0
alpha_r(:)=0.0
beta_r(:)=0.0
random_num(:)=0.0
f_alpha(:)=0.0
f_beta(:)=0.0
lambda_particle(:)=0.0
lambda_temps(:)=0.0
lambda_k(:)=0.0
lambda_frict(:)=0.0
lambda_mass(:)=0.0
xi_cv(:)=0.0
lambda_min(:)=0.0
lambda_max(:)=0.0
lambda_bin_num(:)=0
lambda_bin_width(:)=0.0
lambda_boundary(:)=0.0
lambda_k(:)=0.0
particle_dudl(:)=0.0
particle_force(:)=0.0
particle_bin_sum(:,:)=0.0
particle_bin_count(:,:)=0.0
particle_bin_avg(:,:)=0.0
particle_free_energy(:,:)=0.0
particle_fes_min(:)=0.0
lambda_sum(:,:)=0.0
lambda_count(:,:)=0.0
lambda_avg(:,:)=0.0
particle_a(:)=0.0
particle_b(:)=0.0
particle_c(:)=0.0
cv_array(:)=0
cv_num_lists(:)=0
cv_selection_lists(:,:,:)=0
cv_natom_lists(:,:)=0
ListA(:)=0
ListB(:)=0
ListC(:)=0
ListD(:)=0
dfdlx(:)=0.0
dfdl(:)=0.0
mcen_XA(:)=0.0
mcen_YA(:)=0.0
mcen_ZA(:)=0.0
mcen_XB(:)=0.0
mcen_YB(:)=0.0
mcen_ZB(:)=0.0
mcen_dist(:)=0.0
mcen_XC(:)=0.0
mcen_YC(:)=0.0
mcen_ZC(:)=0.0
mcen_XD(:)=0.0
mcen_YD(:)=0.0
mcen_ZD(:)=0.0
mcen_dist_2(:)=0.0
eabf_rms(:)=0.0
write(*,*)"Done.."
END SUBROUTINE

