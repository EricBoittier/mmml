SUBROUTINE INIALL
  !
  !     This routine initializes all CHARMM data.
  !

  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use bases_fcm

#if KEY_ACE==1
  use ace_module,only:ace_init          
#endif
  use actclus_mod
#if KEY_AFM==1
  use afm_module,only: afm_init              
#endif
  use block_fcm,only:block_init 
#if KEY_CADPAC==1
  use cadpac_mod,only:cadpac_init  
#endif
  use clcg_mod,only:clcg_iniall
  use rndnum
#if KEY_CHEQ==1
  use cheq,only: cheq_iniall,allocate_cheq             
#endif
#if KEY_HMCOM==1
  use cstran_mod,only:hmcm_init 
#endif
  use code,only:code_init
  use conshelix_m
  use consph,only:consph_init    
  use contrl,only:contrl_init
  use corman_mod,only: corman_init
#if KEY_RMD==1
  use cross,only: cross_iniall_init  
#endif
  use ctitla,only:ctitla_init
#if KEY_CVELOCI==1
  use cveloci_mod,only:cveloci_init  
#endif
  use defltsm,only:deflts_iniall
#if KEY_DIMB==1
  use dimb,only:dimb_init            
#endif
#if KEY_DIMS==1
  use dims,only:dims_init            
#endif
#if KEY_DMCONS==1
  use dmcons,only:dmcons_init        
#endif
#if KEY_EDS==1
  use edsmod,only:eds_init           
#endif
#if KEY_EMAP==1
  use emapmod,only:emap_iniall          
#endif
#if KEY_ESTATS==1
  use estats_mod                     
#endif
  use eutil,only:enerin,skipe
  use etablem,only:etable_init
#if KEY_TNPACK==1
  use euler,only:euler_init          
#endif
  use ewald_1m,only:ewald_init
  use exelecm,only:exelec_init
  use fast,only:faster_init
  use ffieldm,only:ffield_init
#if KEY_MMFF==1 || KEY_CFF==1 /*ff_fcm*/
  use rtf,only:ucase    
  use mmffm
#endif /* (ff_fcm)*/
#if KEY_FLUCQ==1
  use flucq,only:flucq_init          
#endif
#if KEY_FMA==1
  use fmam,only:fma_init             
#endif
#if KEY_FOURD==1
  use fourdm,only:fourd_init         
#endif
#if KEY_FACTS==1
  use facts_module,only:facts_init   
#endif
#if KEY_GENETIC==1
  use galgor,only:galgor_init        
#endif
  use gamess_fcm
#if KEY_GAMUS==1
  use gamusmodule, only: dogamus, mgamus, idgamus, gamusdi, params 
#endif
#if KEY_GCMC==1
  use gcmc,only:gcmc_init            
#endif
  use genborn,only: genborn_init     
  use gbim,only: gbim_init           
#if KEY_GRAPE==1
  use grape,only:grape_ltm_init      
#endif
#if KEY_NOGRAPHICS==0
  use graph,only:graph_ltm_init      
#endif
#if KEY_GRID==1
  use grid_dock,only:grid_init            
#endif
  use hbanal_mod,only:hbanal_init
  use hbondm,only:hbond_init
#if KEY_HQBM==1
  use hqbmm, only: hqinit!, allocate_hqbm  
#endif
#if KEY_TSM==1
  use icpert,only:icpert_init        
#endif
  use image,only:image_init
  use image_routines_module, only: inimag
  use inbnd,only:inbnd_init
#if KEY_MMFF==1
  use io,only:io_init                
#endif
#if KEY_BLOCK==1
  use lambdam,only:ldm_init          
#endif
#if KEY_LONEPAIR==1
  use lonepr,only:lonepair_init      
#endif
#if KEY_RXNCOR==1
  use lupcom,only:lupcom_init        
#endif
  use machutil,only:machdep_init
#if KEY_MC==1
  use mc,only:mc_init                
#endif
#if KEY_MULTCAN==1
  use mltcanon,only:mltcan_init      
#endif
#if KEY_MNDO97==1
  use mndo97,only:mndo97_iniall      
#endif
#if KEY_MSCALE==1
  use mscalemod,only:mscale_iniall   
#endif
#if KEY_NOMISC==0
  use noem,only:noe_init             
#endif
  use nose_mod,only:nose_init
#if KEY_OVERLAP==1
  use olap,only:olap_init            
#endif
  use parallel,only:parallel_iniall,mynod
  use param,only:param_iniall
#if KEY_RPATH==1 && KEY_REPLICA==1
  use pathm,only:path_iniall         
#endif
#if KEY_PBEQ==1
  use pbeq,only:pbeq_iniall          
#endif
#if KEY_PBOUND==1
  use pbound,only:pbound_init        
#endif
#if KEY_PERT==1
  use pert,only:pert_iniall          
#endif
#if KEY_PHMD==1
  use phmd,only:phmd_iniall          
#endif
#if KEY_PIPF==1
  use pipfm, only: pipf_iniall       
#endif
#if KEY_POLAR==1
  use polarm, only: polar_iniall     
#endif
#if KEY_PRIMSH==1
  use primsh, only:primsh_iniall     
#endif
#if KEY_PROTO==1
  use proto_mod,only:proto_init      
#endif
  use psf,only:psf_iniall
  use pull_mod,only:epull_init
#if KEY_QUANTUM==1
  use quantm,only:quantum_init       
#endif
#if KEY_REPDSTR==1
  use repdstr,only:repdstr_iniall 
  use repdstrmod                     
#endif
  use reawri,only:reawri_init,iseed
  use replica_mod,only:replica_iniall
  use resdist_ltm,only:resdist_iniall
#if KEY_RGYCONS==1
  use rgym,only:rgy_iniall           
#endif
  use rtf,only:rtf_iniall
#if KEY_RXNCOR==1
  use rxncom,only:rxncor_iniall      
#endif
#if KEY_RXNCONS==1
  use rxncons,only:rxncons_iniall    
#endif
  use rush_mod,only:rush_init        
#if KEY_SASAE==1
  use sasa,only:sasa_init            
#endif
#if KEY_NOMISC==0
  use sbound,only:sbound_init        
#endif
  use storage
#if KEY_SCCDFTB==1
  use sccdftb,only:sccdftb_init      
#endif
  use selctam,only:select_init
  use shake,only:shake_iniall
#if KEY_SHAPES==1
  use shapes,only:shapes_init        
#endif
#if KEY_SHELL==1
  use shell,only:shell_init          
#endif
#if KEY_SPACDEC==1
  use spacdec,only:spacdec_iniall    
#endif
  use squantm
  use ssbpm,only: ssbpm_init
  use stream,only:stream_iniall
  use string,only:string_iniall
#if KEY_ASPENER==1
  use surface,only:surface_iniall    
#endif
#if KEY_ASPMEMB==1
  use surfmemb,only:surfmemb_iniall  
#endif
#if KEY_MTS==1
  use tbmts,only:tbmts_iniall        
#endif
#if KEY_TORQUE==1
  use torque,only:torque_iniall      
#endif
#if KEY_TRAVEL==1
  use travel,only:travel_iniall      
#endif
#if KEY_TMD==1
  use tmd,only:tmd_iniall            
#endif
#if KEY_RISM==1 /*rism_fcm*/
  use rism
  use struc,only:struc_iniall
  use distri
#endif /* (rism_fcm)*/
#if KEY_ADUMB==1
  use umb                            
#endif
  use umbcor 
  use univ,only: runivf
  use varcutm
#if KEY_ZEROM==1
  use zdata_mod,only: zerom_iniall
#endif
  use timerm,only:timer_iniall
  use valbond, only: vbresd,vbinid,vbdond
#if KEY_CONSHELIX==1
  use conshelix_fcm                  
#endif
#if KEY_SSNMR==1
  use ssnmr,only:ccs_iniall
#endif
  use gopair, only : GoPair_initialize
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
  use derivdhdgb
#endif
#ifdef KEY_RESIZE
  use resize,only: resize_init
#endif
  ! local variables
  implicit none
  INTEGER I,J
  character(len=4) WINIT
#if KEY_MMFF==0 && KEY_CFF==0
  logical ucase
#endif 
  !=======================================================================
  ! VALBOND.FCM
#if KEY_VALBOND==1
  VBRESD = .FALSE.    
#endif
#if KEY_VALBOND==1
  VBINID = .FALSE.    
#endif
#if KEY_VALBOND==1
  VBDOND = .FALSE.    
#endif
  !
  !=======================================================================
 
#if KEY_ACE==1
  call ace_init  
#endif
  call actclus_init
#if KEY_AFM==1
  call afm_init       
#endif
#if KEY_BLOCK==1
  call block_init     
#endif
#if KEY_CADPACK==1
  call cadpac_init    
#endif
#if KEY_HMCOM==1
  call hmcm_init      
#endif
  call code_init
  call consph_init    
#if KEY_CONSHELIX==1
  call conshelix_iniall 
#endif
  call contrl_init
  call corman_init
#if KEY_RMD==1
  call cross_iniall_init 
#endif
  call ctitla_init
#if KEY_CVELOCI==1
  call cveloci_init    
#endif
#if KEY_DIMB==1
  call dimb_init       
#endif
#if KEY_DIMS==1
  call dims_init       
#endif
#if KEY_DMCONS==1
  call dmcons_init     
#endif
#if KEY_EDS==1
  call eds_init        
#endif
#if KEY_EMAP==1
  call emap_iniall     
#endif
  call enerin
  call epull_init
  call etable_init
#if KEY_TNPACK==1
  call euler_init      
#endif
  call ewald_init
  call exelec_init
  call faster_init
  call ffield_init(ucase) 
#if KEY_FLUCQ==1
  call flucq_init      
#endif
#if KEY_FMA==1
  call fma_init        
#endif
#if KEY_FOURD==1
  call fourd_init      
#endif
#if KEY_FACTS==1
  call facts_init      
#endif
#if KEY_GENETIC==1
  call galgor_init     
#endif
#if KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_QCHEM==1 || KEY_SQUANTM==1 || KEY_MNDO97==1 || KEY_QTURBO==1 || KEY_G09==1
  call gamess_ltm_init
#endif 
#if KEY_GCMC==1
  call gcmc_init       
#endif
  call genborn_init    
  call gbim_init       
  call GoPair_initialize()
#if KEY_NOGRAPHICS==0
  call graph_ltm_init  
#endif
#if KEY_GRAPE==1
  call grape_ltm_init  
#endif
#if KEY_GRID==1
  call grid_init       
#endif
  call hbanal_init
  call hbond_init
#if KEY_TSM==1
  call icpert_init     
#endif
  call image_init
  call inbnd_init
#if KEY_MMFF==1
  call io_init         
#endif
#if KEY_NBIPS==1
  call ipsinit         
#endif
#if KEY_BLOCK==1
  call ldm_init        
#endif
#if KEY_LONEPAIR==1
  call lonepair_init   
#endif
#if KEY_RXNCOR==1
  call lupcom_init     
#endif
  call machdep_init
#if KEY_MC==1
  call mc_init         
#endif
#if KEY_MULTCAN==1
  call mltcan_init     
#endif
#if KEY_MNDO97==1
  call mndo97_iniall   
#endif
#if KEY_MSCALE==1
  call mscale_iniall   
#endif
#if KEY_NOMISC==0
  call noe_init        
#endif
  call nose_init
#if KEY_OVERLAP==1
  call olap_init       
#endif
  call parallel_iniall 
  call param_iniall
#if KEY_REPLICA==1 && KEY_RPATH==1
  call path_iniall     
#endif
#if KEY_PBEQ==1
  call pbeq_iniall     
#endif
#if KEY_PBOUND==1
  call pbound_init     
#endif
#if KEY_PERT==1
  call pert_iniall     
#endif
#if KEY_PHMD==1
  call phmd_iniall     
#endif
#if KEY_PIPF==1
  call pipf_iniall     
#endif
#if KEY_POLAR==1
  call polar_iniall    
#endif
#if KEY_PRIMSH==1
  call primsh_iniall   
#endif
#if KEY_PROTO==1
  call proto_init      
#endif
  call psf_iniall      
#ifdef KEY_RESIZE
  call resize_init
#endif  
#if KEY_QUANTUM==1
  call quantum_init    
#endif
  call reawri_init
#if KEY_REPDSTR==1
  call repdstr_iniall  
#endif
  call replica_iniall
  call resdist_iniall
#if KEY_RGYCONS==1
  call rgy_iniall      
#endif
  call rtf_iniall
#if KEY_RXNCOR==1
  call rxncor_iniall   
#endif
#if KEY_RXNCONS==1
  call rxncons_iniall  
#endif
  call rush_init       
#if KEY_SASAE==1
  call sasa_init       
#endif
#if KEY_NOMISC==0
  call sbound_init     
#endif
#if KEY_SCCDFTB==1
  call sccdftb_init    
#endif
  call ssbpm_init
  call storage_allocate()
  call select_init
  call shake_iniall
#if KEY_SHAPES==1
  call shapes_init     
#endif
#if KEY_SPACDEC==1
  call spacdec_iniall  
#endif
#if KEY_SHELL==1
  call shell_init      
#endif
  call clcg_iniall(iseed,mynod)
#if KEY_SQUANTM==1
  call squantm_iniall  
#endif
  call stream_iniall
  call string_iniall
#if KEY_ASPENER==1
  call surface_iniall  
#endif
#if KEY_ASPMEMB==1
  call surfmemb_iniall 
#endif
#if KEY_MTS==1
  call tbmts_iniall     
#endif
  call timer_iniall
#if KEY_TMD==1
  call tmd_iniall         
#endif
#if KEY_TRAVEL==1
  call travel_iniall      
#endif
#if KEY_TSM==1
  call tsmclear           
#endif
#if KEY_TORQUE==1
  call torque_iniall      
#endif
  ! UMB.FCM
#if KEY_ADUMB==1 /*adumb_init*/
  NUMBR=0
  STONUM=.TRUE.
  ! for correlations
  NDISTA=0
  LCORRV=.FALSE.
  LPRCRU=.FALSE.
  LPRRMU=.FALSE.
  RMSMMS=-1
  RMSTRS=-1
  RMACNT=0
  ORATCT=0
  NRMSDI=0
  LCCOR1=.FALSE.
  LCCOR2=.FALSE.
  MXRMSA=0
  ADFREQ=1
  PCORMD=1
#endif /* (adumb_init)*/
#if KEY_GAMUS==1
  dogamus=.false.
  mgamus=0
  idgamus=0
  gamusdi=0
  params%ngauss=0
  params%ndim=0
#endif 
  call deflts_iniall

#if KEY_ESTATS==1
  call energy_anal_iniall     
#endif
#if KEY_ZEROM==1
  call zerom_iniall           
#endif
#if KEY_ACTBOND==1
  call actbond_iniall         
#endif
#if KEY_RISM==1
  call struc_iniall           
#endif
  call runivf(-2)

  !=======================================================================
  ! initiate (some primitive rite?) data structure variables
  !
  winit='INIT'
  j=4
  call skipe(winit,j)
  call incrys
  call inimag(bimag,.true.)
  j=4
  winit='INIT'
  call gthbct(winit,j)
#if KEY_CHEQ==1
  call cheq_iniall()        
#endif
  !=======================================================================
  ! initialize hqbm
#if KEY_HQBM==1
  call hqinit                                                
#endif

  return
end SUBROUTINE INIALL

SUBROUTINE HEADER
  !
  ! This routine prints out the startup header
  !
  use cstuff, only: getpid
  use dimens_fcm
  use machutil, only: daytim
  use param
  use param_store, only: set_param
  use startup
  use stream
  use version

  implicit none
  ! BIOVIA Code Start : Fix for length of USERNM
  character(len=24) USERNM
  ! BIOVIA Code End
  CHARACTER(len=80) OSNAME
  character(len=60) PRODUCT
#if KEY_INSIGHT==1
  character(len=24) fdate
  external     fdate
#endif 

  integer :: month, day, year, hour, minute, second, &
    oslen, mypid

  ! install.com writes REVID here
  include 'revid.h'
  !
  !     Moved here for parallel stuff
#if KEY_XT4==1
  !     CRAY XT4 does not support system calls.
  OSNAME='CRAY XT4/XT5'
#else /**/
  CALL SYSID(OSNAME)
#endif 
  !
  IF(PRNLEV <= 0) RETURN
  !
#if KEY_STRINGM==0 /*  VO system calls cause intermittent problems on NERSC supercomputers */
  CALL DAYTIM(month, day, year, hour, minute, second)

  usernm = ' '
  CALL GETNAM(USERNM)
#endif
  OSLEN = LEN_trim(OSNAME)

  ! -- mikem -- 07/27/92
  PRODUCT='(CHARMM) - Developmental Version '//VERNMC

  WRITE (OUTU, 50) center_pad(PRODUCT, 80)
50 FORMAT('1',/17x,'Chemistry at HARvard Macromolecular Mechanics',/,A)

  IF (len_trim(REVID) > 0) THEN
    WRITE (OUTU, '(A)') center_pad(REVID, 80)
  ENDIF

  WRITE (OUTU, 51)
51 FORMAT(7x,'Copyright(c) 1984-2023  ', &
       'President and Fellows of Harvard College', &
       /,30x,'All Rights Reserved')

  WRITE (OUTU, '(A)') center_pad('Current operating system: ' // OSNAME, 80)

! BIOVIA Code Start : Print out compiler version
#ifdef __INTEL_COMPILER
52 FORMAT(25x,'Compiler version: Intel ',i4,'.',i0)
  WRITE (OUTU, 52) __INTEL_COMPILER, __INTEL_COMPILER_UPDATE
#endif
! BIOVIA Code End

#if KEY_STRINGM==0 /*  VO system calls cause intermittent problems on NERSC supercomputers */
  WRITE(OUTU,54) month, day, year, hour, minute, second, usernm
#endif
54 FORMAT(17x,'Created on ',i2,'/',i0,'/',i2.2, &
       ' at ',i2,':',i2.2,':',i2.2,' by user: ',a12,/)
  ! <- mikem --

  WRITE(OUTU,43) MAXA,MAXRES
43 FORMAT(12X,'Maximum number of ATOMS: ',I9, &
       ', and RESidues:   ',I9)
#ifdef KEY_RESIZE
  WRITE(OUTU,'(12X,A,/,12X,A,/)') &
       'RESIZE: Structure-related arrays are automatically resized.',&
       'Maximum numbers above have limited applicability.'
#endif
  !
#if KEY_NIH==1
  mypid = getpid()
  WRITE(OUTU,44) MYPID
  call set_param('PID',MYPID)
#endif 
44 FORMAT(12X,'PID of the current process is: ',I6)
  RETURN
contains
  function center_pad(text, width)
    character(len=*), intent(in) :: text
    integer, intent(in) :: width
    character(len=width) :: center_pad
    integer :: pad_ct

    pad_ct = (width - len_trim(text)) / 2
    ! BIOVIA Code Start : Bug fix
    if (pad_ct > 0) then
       center_pad = repeat(' ', pad_ct) // text
    else
       center_pad = text
    endif
    ! BIOVIA Code End
  end function center_pad
END SUBROUTINE HEADER

SUBROUTINE STOPCH(DIENAM)
  !
  ! THIS IS THE NORMAL TERMINATION ROUTINE FOR CHARMM.
  !
  use ewald_1m,only:maxkv,pkvec,pkxv,pkyv,pkzv
  use erfcd_mod
  use pme_module,only:pmesh_clear,qpme
  use zdata_mod,only: QZMOD
  use new_timer,only:timer_stop,T_total,finish_timers,write_timers 
#if KEY_CHEQ==1
  use cheq,only:cheqstop              
#endif
#if KEY_REPDSTR==1
  use repdstrmod                      
#endif
#if KEY_GRAPE==1
  use grapemod,only: grapefin         
#endif
  use allocdat
  use deallocdat
  use memory
  !
  use chm_kinds
  use chm_types
  use dimens_fcm
  use bases_fcm
#if KEY_HQBM==1
  use hqbmm, only: hqfin              
#endif
  use inbnd
  use image
#if KEY_LOOKUP==1
  use lookup,only:wwsetup              
#endif
  use selctam
  use scalar_module
  use storage
  use stream
  !MEK.. 94/07/12 add the following three lines
  use psf
  use shake
  use shapes
#if KEY_PARALLEL==1
  use parallel
  use repdstr
  use number
#if KEY_MSCALE==1
  use mscalemod,only:qmscale,qmscmain,mscalefin  
#endif
#endif 

  use lambdam !Css for SS
#if KEY_FSSHK==1
  use fstshk,only:fsrscshk  
#endif
  use datstr,only:freedt_nbond,freedt_image
  use machutil,only:die
  use vangle_mm, only: ptrfin
  use consph,only: consph_cleanup 
#if KEY_EDS==1
  use edsmod,only: eds_cleanup 
#endif
#if KEY_DOMDEC==1
  use domdec_common,only: q_domdec, q_split, ndirect
  use domdec_dr_common, only : nrecip, q_direct_node, q_recip_node, &
                               comm_direct, comm_recip, mynod_split 
#endif
  use nbexcl,only:makitc_clr
#if KEY_ENSEMBLE==1
  use evb_mod, only: qevb, evb_deallocate
#endif
  implicit none
  character(len=10) :: file = "iniall.src"
  character(len=9) :: routine = "stopch"
  !
  character(len=*) DIENAM
  !
  !
  INTEGER I
  LOGICAL QOPEN,QFORM,ERROR,QWRITE
  INTEGER TLEN
  integer,PARAMETER :: TMAX=80
  character(len=TMAX) T

  T = ' '

#if KEY_PARALLEL==1
  IF(QSPLIT)THEN
     numnod=numnodg
     mynod=mynodg
     mynodp=numnod+1
     noddim=nptwo()
     do i=0,numnod
        ippmap(i)=i
     enddo
     do i=1,maxnode
        inode(i)=mod(i,numnod)
     enddo
  ENDIF
  ! Grape/Gpu comes before parallel!
#if KEY_GRAPE==1
  CALL GRAPEFIN   
#endif
  ! Finish MSCALE
#if KEY_MSCALE==1
  IF (QMSCALE.AND.QMSCMAIN) CALL MSCALEFIN    
#endif
#if KEY_REPDSTR==1
  !     We are at the end, so globalize and normalize I/O
  IF(QREPDSTR)THEN
     call repdstr_deallocate
     CALL PSETGLOB
     CALL DREPRESIO(IOLEV,PRNLEV,WRNLEV)
  ENDIF
#endif 
  !     Measuring load balance
  !
!  CALL PSYNC()
!  !
!#if KEY_DOMDEC==1
!  if (.not.q_domdec) then  
!#endif
!#if KEY_ZEROM==0
!     CALL BALPRN()
!#endif
!#if KEY_DOMDEC==1
!  endif  
!#endif
  !
#endif 
  !
  ! close all files
  DO I=1,99
     IF(I /= OUTU) THEN
        CALL VINQRE('UNIT',T,TMAX,TLEN,QOPEN,QFORM,QWRITE,I)
        IF (QOPEN) CALL VCLOSE(I,'KEEP',ERROR)
     ENDIF
  ENDDO
  !
  ! free allocated space
  CALL FREEDT_nbond(BNBND)
  CALL FREEDT_image(BIMAG)
  !
  CALL FREEDT_nbond(BNBNDC)
  CALL FREEDT_nbond(BNBNDR)
  CALL FREEDT_nbond(BNBNDP)
  CALL FREEDT_image(BIMAGC)
  CALL FREEDT_image(BIMAGR)
  CALL FREEDT_image(BIMAGP)

#if KEY_BLOCK==1
  IF(QMCFR) THEN
     call chmdealloc('iniall.src','STOPCH','MCPRO',MCBOXES,crl=MCPRO)
     call chmdealloc('iniall.src','STOPCH','MCCOUNT',MCBOXES,intg=MCCOUNT)
     call chmdealloc('iniall.src','STOPCH','MCLAMD',MCBOXES,crl=MCLAMD)
  ENDIF
#endif 

  call storage_deallocate()

  DO I=1,NUMSKY
     call chmdealloc('iniall.src','STOPCH','PTRSKY(I)%a',LENSKY(I),intgp=PTRSKY(I)%a)
  ENDDO
  CALL PTRFIN

#if KEY_FSSHK==1
  IF (QSHAKE .AND. QFSHAKE) CALL FSRSCSHK     
#endif

  ! Release space held by various parts of the ewald code.
  if(allocated(ewldt))call deallocate_ewldt()
  IF(MAXKV > 0) THEN
     call chmdealloc(file,routine,"pkvec",maxkv,crl=pkvec)
     call chmdealloc(file,routine,"pkxv", maxkv,intg=pkxv)
     call chmdealloc(file,routine,"pkyv", maxkv,intg=pkyv)
     call chmdealloc(file,routine,"pkzv", maxkv,intg=pkzv)
  ENDIF
  IF(QPME) CALL PMESH_CLEAR
  call makitc_clr
  !
  ! Deallocate memory used by memory module
  if(allocated(filnamar)) deallocate(filnamar,arnamar,artypar, &
       procnamar,arrankar,arsizear,arkindar)

  if(allocated(ffilnamar)) deallocate(ffilnamar,farnamar,fartypar, &
       fprocnamar,farrankar,farsizear,farkindar)

  if(allocated(dfilnamar)) deallocate(dfilnamar,darnamar, &
       dartypar,dprocnamar,darrankar,darsizear,diarsizear, &
       darkindar)

  if(allocated(fdfilnamar)) deallocate(fdfilnamar,fdarnamar, &
       fdartypar,fdprocnamar,fdarrankar,fdarsizear,fdiarsizear, &
       fdarkindar)
  !
#if KEY_CHEQ==1
  CALL CHEQSTOP
#endif 
  !
#if KEY_LOOKUP==1
  ! Maybe not necessary?
  T='RELEASE'
  TLEN=7
  CALL WWSETUP(T,TLEN)
#endif 
  !
#if KEY_SHAPES==1
  IF(ORDSHP > 0) CALL FREESHP(.TRUE.)
#endif 
  CALL CONSPH_CLEANUP 
#if KEY_EDS==1
  CALL EDS_CLEANUP 
#endif

  call timer_stop(T_total)
  call finish_timers()
# if KEY_DOMDEC==1
  call write_timers(q_split, ndirect, nrecip, q_direct_node, q_recip_node, &
                    comm_direct, comm_recip, (mynod_split == 0))
# else
  call write_timers()
# endif
  !
  !
  CALL ENSFIN
#if KEY_PARALLEL==1
  CALL PARFIN     
#endif
  !
#if KEY_HQBM==1
  CALL HQFIN      
#endif
  ! 
#if KEY_QCHEM==1
  call unlink('charges.dat')
#endif

#if KEY_ENSEMBLE==1
  if (qevb) call evb_deallocate
#endif
  !
  ! print out termination status
  IF (PRNLEV > 0) CALL PRTSTATS(DIENAM)
  
! BIOVIA Code Start
#if KEY_LICENSE==1
  ! release charmm license
  call release_licenses()
#endif
! BIOVIA Code End
  ! dont go back!
  STOP
  ! Make sure we don't go back...
  CALL DIE
END SUBROUTINE STOPCH

SUBROUTINE PRTSTATS(DIENAM)
  use chm_kinds
  use dimens_fcm
  use bases_fcm
  use inbnd
  use image
  use selctam
  use scalar_module
  use stream
  use timerm
  use machutil,only:jobdat

  implicit none
  !
  character(len=*) DIENAM
  !
  INTEGER ISTOPB,ISTOPD,ISTOPP
  real(chm_real) STOPTC,STOPTE
  character(len=8) SCPUTI,SELAPS,SHOUR,SMINUT,SSECND
  DATA SSECND,SMINUT,SHOUR/'SECONDS ','MINUTES ','HOURS   '/
  !
  IF(DIENAM == 'DIE') THEN
     WRITE(OUTU,1321)
  ELSE
     WRITE(OUTU,1322) DIENAM
  ENDIF
  IF(BOMMIN < 100) THEN
     WRITE(OUTU,1324) BOMMIN
  ELSE
     WRITE(OUTU,1325)
  ENDIF
1321 FORMAT(/20X,'ABNORMAL TERMINATION')
1322 FORMAT(/20X,'NORMAL TERMINATION BY ',A)
1324 FORMAT(20X,'MOST SEVERE WARNING WAS AT LEVEL',I3)
1325 FORMAT(20X,'NO WARNINGS WERE ISSUED')
  !
  ! Print out some run data
  CALL JOBDAT(STOPTE,STOPTC,ISTOPB,ISTOPD,ISTOPP)
  DO WHILE(STOPTE < 0.0)
     STOPTE=STOPTE+86400.0
  ENDDO
  SELAPS=SSECND
  IF (STOPTE >= 60.0) THEN
     STOPTE=STOPTE/60.0
     SELAPS=SMINUT
     IF (STOPTE >= 60.0) THEN
        STOPTE=STOPTE/60.0
        SELAPS=SHOUR
     ENDIF
  ENDIF
  SCPUTI=SSECND
  IF (STOPTC >= 60.0) THEN
     STOPTC=STOPTC/60.0
     SCPUTI=SMINUT
     IF (STOPTC >= 60.0) THEN
        STOPTC=STOPTC/60.0
        SCPUTI=SHOUR
     ENDIF
  ENDIF
  WRITE(OUTU,1350) STOPTE,SELAPS,STOPTC,SCPUTI
1350 FORMAT(/20X,'$$$$$ JOB ACCOUNTING INFORMATION $$$$$'/ &
       20X,' ELAPSED TIME: ',F8.2,2X,A8/ &
       20X,'     CPU TIME: ',F8.2,2X,A8)
#if KEY_PATHSCALE==1
  call & 
#endif
  flush (OUTU)
  RETURN
END SUBROUTINE PRTSTATS

SUBROUTINE GETPREF()
  !
  !  Set substitution paramters based on all compile keywords
  !                               - BRB, July 30, 2003
  !
  use chm_kinds
  use machutil, only: die
  use keywords, only: num_pref_keys, pref_keys
  use param_store, only: set_param

  implicit none

  integer, parameter :: MAXKEYS = 300     ! should match limits in tool/prefx.f
  integer :: i

  IF(num_pref_keys > MAXKEYS) THEN
     CALL WRNDIE(-5,'<GETPREF>', &
          'Too many compile keys. Memory overwrite.')
     CALL DIE ! if you are not dead yet.
  END IF

  do i = 1, num_pref_keys
     if (pref_keys(i) == 'TSM') then
        call set_param('QTSM', 1)
        !sb   CMAP also needs special treatment
     elseif (pref_keys(i) == 'CMAP') then
        call set_param('CMAPSET', 1)
     elseif (pref_keys(i) == 'EDS') then
        call set_param('QEDS', 1)
     else
        call set_param(pref_keys(i), 1)
     end if
  end do

  ! Add key for default size (at the moment) 
  call set_param('XXLARGE', 1)
#if KEY_PARALLEL==0
  call set_param('MYNODE', 0)   
  call set_param('NUMNODE', 1)  
#endif
  return
END SUBROUTINE GETPREF

subroutine parse_size(comlyn,comlen,qrdcmd)
  use dimens_fcm
  use string
  use stream

  implicit none
  character(len=*),intent(inout) :: comlyn
  integer,intent(inout) ::   comlen
  logical,intent(out) ::  qrdcmd
  integer :: size_arg, i
  character(len=6) :: word
  character(len=4) :: wrd,wrd2

  ! If DIMEnsion is present we will process the line and exit as if we
  ! haven't already read any lines yet, otherwise, we have
  ! already read comlyn and need to process it
  call cnvtuc(comlyn,comlen)

  wrd=curra4(comlyn,comlen)

  select case(wrd)
  case("DIME","RESI")
     qrdcmd=.true.
     i=indxa(comlyn,comlen,"DIME")
     i=indxa(comlyn,comlen,"RESI")
  case default
     qrdcmd=.false.
  end select

  if(.not. qrdcmd) return

  if(comlen == 0)then
     call print_charmm_sizes
     stop
  endif

  do while(comlen>0)
     word=nexta6(comlyn,comlen)

     select case(word)
     case('CHSIZE')
        size_arg = nexti(comlyn, comlen)
        call set_chsize(size_arg)
     case('MAXA  ')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxa, size_arg)
     case('MAXB  ')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxb, size_arg)
     case('MAXT  ')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxt, size_arg)
     case('MAXP  ')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxp, size_arg)
     case('MAXIMP')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maximp, size_arg)
     case('MAXNB ')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxnb, size_arg)
     case('MAXPAD')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxpad, size_arg)
     case('MAXRES')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxres, size_arg)
     case('MAXSEG')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxseg, size_arg)
#if KEY_CMAP==1
     case('MAXCRT')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxcrt, size_arg)
#endif 
     case('MAXSHK')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxshk, size_arg)
     case('MAXAIM')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxaim, size_arg)
     case('MAXGRP')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxgrp, size_arg)
     case('MAXNBF')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxnbf, size_arg)
     case('MAXITC')
        size_arg = nexti(comlyn, comlen)
        call set_dimen(new_chsize%maxitc, size_arg)
        call set_dimen(new_chsize%maxcn, size_arg*(size_arg+1)/2)
     case default
        write(outu,'("    DIMENSION> Unknown redimension size: ",a6)') word
        write(outu,'("    DIMENSION> Skipping: ",a6,/)')word
     end select
  enddo
  ! any other command reads go above here
  call xtrane(comlyn, comlen, "parse_size")

  call print_charmm_sizes
  return
end subroutine parse_size

subroutine allocate_all
  use dimens_fcm

!  use actclus_mod  ! Bob should get this allocation to be as needed.
  use cheq
  use cnst_fcm
  use code
  use coord
  use coordc
  use deriv
  use fast
  use fourdm
  use gamess_fcm
#if KEY_MCMA==1
  use mcmamod        
#endif
  use mtp_fcm
  use mtpl_fcm
  use olap
  use param
  use pathm
  use psf
  use quantm
  use rxncons
  use squantm
  use tbmts
#if KEY_DHDGB==1
!AP/MF
  use dhdgb
  use derivdhdgb
  use dhdgbc
#endif
  implicit none

  call set_dimens()
#ifndef KEY_RESIZE
  call freeze_dimens()
#endif
  call store_common_params()

!  call allocate_actclus
#if KEY_CHEQ==1
  call allocate_cheqinit 
#endif
!  call allocate_cnst
  call allocate_code
  call allocate_coord_ltm
  call allocate_coordc
  call allocate_deriv
  call allocate_fast_ltm
#if KEY_MCMA==1
  call allocate_mcma  
#endif
  call allocate_psf_ltm
  call allocate_param_ltm
#if KEY_MTS==1
  call allocate_tbmts  
#endif

#if KEY_DHDGB==1
!AP/MF
  call allocate_dhdgb
  call allocate_derivdhdgb
  call allocate_dhdgbc
#endif
  return
end subroutine allocate_all

subroutine store_common_params()
  use param_store, only: set_param
  use number
  use consta
  use dimens_fcm
  implicit none

  call set_param('PI  ',PI)
  call set_param('KBLZ',KBOLTZ)
  call set_param('TIMFAC',TIMFAC)
  call set_param('CCELEC',CCELEC)
  call set_param('CNVFRQ',CNVFRQ)
  call set_param('SPEEDL',SPEEDL)
  call set_param('MINGRMS',ZERO)
  call set_param('XTLA',ZERO)
  call set_param('XTLB',ZERO)
  call set_param('XTLC',ZERO)
  call set_param('XTLALPHA',ZERO)
  call set_param('XTLBETA',ZERO)
  call set_param('XTLGAMMA',ZERO)

  call set_param('MINECALLS',0)
  call set_param('MINSTEPS',0)
  call set_param('MINCONVRG',0)
  call set_param('XTLXDIM',0)
  call set_param('MAXSEG',MAXSEG)
  call set_param('MAXA',MAXA)
  call set_param('MAXB',MAXB)
  call set_param('MAXT',MAXT)
  call set_param('MAXP',MAXP)
  call set_param('MAXIMP',MAXIMP)
  call set_param('MAXNB',MAXNB)
  call set_param('MAXPAD',MAXPAD)
  call set_param('MAXRES',MAXRES)
  call set_param('MAXATC',MAXATC)
  call set_param('MAXCB',MAXCB)
  call set_param('MAXCT',MAXCT)
  call set_param('MAXCP',MAXCP)
  call set_param('MAXCI',MAXCI)
  call set_param('MAXCH',MAXCH)
  call set_param('MAXCN',MAXCN)
end subroutine store_common_params

subroutine print_charmm_sizes()
  use dimens_fcm
  use stream
  if (wrnlev >= 2) then
     write (outu, 10) 'Size  ', 'Original', 'New'
     write (outu, 20) 'MAXA  ', MAXA, get_dimen(new_chsize%MAXA, MAXA)
     write (outu, 20) 'MAXB  ', MAXB, get_dimen(new_chsize%MAXB, MAXB)
     write (outu, 20) 'MAXT  ', MAXT, get_dimen(new_chsize%MAXT, MAXT)
     write (outu, 20) 'MAXP  ', MAXP, get_dimen(new_chsize%MAXP, MAXP)
     write (outu, 20) 'MAXIMP', MAXIMP, get_dimen(new_chsize%MAXIMP, MAXIMP)
     write (outu, 20) 'MAXNB ', MAXNB,  get_dimen(new_chsize%MAXNB,  MAXNB)
     write (outu, 20) 'MAXPAD', MAXPAD, get_dimen(new_chsize%MAXPAD, MAXPAD)
     write (outu, 20) 'MAXRES', MAXRES, get_dimen(new_chsize%MAXRES, MAXRES)
     write (outu, 20) 'MAXSEG', MAXSEG, get_dimen(new_chsize%MAXSEG, MAXSEG)
     write (outu, 20) 'MAXATC', MAXATC
     write (outu, 20) 'MAXCB ', MAXCB
     write (outu, 20) 'MAXCT ', MAXCT
     write (outu, 20) 'MAXCP ', MAXCP
     write (outu, 20) 'MAXCI ', MAXCI
     write (outu, 20) 'MAXCH ', MAXCH
     write (outu, 20) 'MAXCN ', MAXCN
#if KEY_CMAP==1
     write (outu, 20) 'MAXCRT', MAXCRT, get_dimen(new_chsize%MAXCRT, MAXCRT) 
#endif
10   format(a6,2a12)
20   format(a6,2i12)
  endif

  return
end subroutine print_charmm_sizes

