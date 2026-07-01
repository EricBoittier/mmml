recursive subroutine MAINCOMX(COMLYN,COMLEN,LUSED)
  !
  !  This is CHARMM's main command parser
  !
  !  Note: Miscellaneous commands (READ, WRITe,...) are parsed in MISCOM
  !        It should be called prior to calling this routine (if desired).
  !        Miscellaneous commands are available in subcommand parsers.
  !        The commands parsed here are not.    - BRB
  !
  !  Note: All commands in CHARMM are abbreviated to first 4 characters
  !        (except for graphics subcommands which are 3 characters).
  !
#if KEY_BLADE == 1
  use blade_ctrl_module, only : blade_command
#endif
  use chm_kinds
  use chm_types
  use dimens_fcm
  use number
  use bases_fcm
  use contrl
  use coord
  use coordc
  use ctitla
  use hbondm
  use image
  use image_routines_module, only: imspec, impatc
  use psf
  use minmiz_module, only: minmiz
  use modpsf
  use genpsf_m
  use stream
  use string
  use timerm
  use dcntrl_mod, only: dynopt
  use estats_mod,only: anal
#if KEY_ESTATS==1
  use estats_mod,only: estats        
#endif
  use cveloci_mod,only: cveloci
  use mltcanon,only:mltcanon_prs
  use mmfp,only:mmfp0
  use cadpac_mod,only:cadini
  use cstran_mod,only:cstran
#if KEY_QMMMSEMI==1
  use qmmmsemi,only: qmmm_startup         
#endif
#if KEY_EPMF==1
  use epmf, only: epmf_set                
#endif
#if KEY_PRIMO==1
  use primomodule, only: primo_set        
#endif
  use genborn,only: genborn_set
  use gbim, only: SetGBIM
  use gbmv, only:gbmv_set                 
  use omm_mses, only:mses_set                 
  use gbsw, only: gbsw_set                


#if KEY_DENBIAS==1
  use denbias, only: denbias_set
#endif

  use phmd
  use gnn, only: gnncall
  use tamdmodule, only: tamd
#if KEY_MSCALE==1
  use mscalemod, only: mscale             
#endif
#if KEY_CSA==1 || KEY_DISTENE==1
  use csacommmod                          
#endif
  use repdstrmod, only: repdstrmain
#if KEY_OVERLAP==1
  use olapmod,only:olapcmd                
#endif
#if KEY_RPATH==1
  use EPATHMOD, ONLY: PATHPS              
#endif
#if KEY_FLUCQ==1
  use flucqm, only: fqinit                
#endif
#if KEY_TSALLIS==1
  use tsallis_module,only: iascale,qttsall 
#endif
#if KEY_FACTS==1
  use facts_module,only:fctini            
#endif
#if KEY_AFM==1
  use afm_module,only: afmini              
#endif
#if KEY_AXD==1
  use axd_module,only: axdini              
#endif
  use gukini_mod,only: gukini
  use gamess_fcm,only: qmused_qchem,qmused_g09,qmused_turbo,qmused_gamess
  use gamess_fcm,only: qmused_squantm,qmused_sccdftb,qmused_mndo97
  use gamess_fcm,only: qmused_nwchem
  use traj_mod,only:reatrj,wrttrj,trajio
  use block_fcm, only : block
#if KEY_SCPISM==1
  use scpismm,only:scpparse                
#endif
  use corman_mod,only:corcom
#if KEY_DMCONS==1
  use dmcons,only:dmcset                   
#endif
#if KEY_HQBM==1
  use hqbmm, only: hqbmini                 
#endif
  use pull_mod, only: pull_setup
  use rmsdyn_mod,only:rmsdyn
#if KEY_GENETIC==1
  use galgor,only:genetic_alg             
#endif
#if KEY_PATHINT==1
  use mpathint, only: pint_init            
#endif
#if KEY_POLAR==1
  use polarm, only: polar0                 
#endif
#if KEY_RGYCONS==1
  use rgym, only: rgyset                   
#endif
  use rush_mod, only: rush                 
#if KEY_RXNCOR==1
  use lup, only: lupopt                    
#endif
  use grid_dock, only: gridset
#if KEY_FFTDOCK == 1
  use fftdock, only: fft_dock_set
#if KEY_OPENMM == 1
  use openmm_dock, only: openmm_dock_set
#endif /* OPENMM */
#endif /* FFTDOCK */
#if KEY_ASPENER==1
  use eef1_mod, only: eef1                 
#endif
! #if KEY_MOBHY==1
  use mobhy_mod, only: mobhy
! #endif
#if KEY_PBEQ==1
  use pbeq, only: pbeq0                    
#endif
#if KEY_TORQUE==1
  use torque, only: torque_parse           
#endif
  use consph, only: getphresidues          
#if KEY_EDS==1
  use edsmod, only: process_eds            
#endif
#if KEY_CHEQ==1
  use cheq,only: cheqprep 
#endif
  use scalar_module, only: scalar
#if KEY_PROTO==1
  use proto_mod,only:proto     
#endif
  use intcor_module,only: intcor
  use rdfsol_mod,only:rdfsol
  use rxcons,only:rxconsps
#if KEY_CORSOL==1
  use corsol_mod,only:corsol     
#endif
  use shell,only:shlini
  use correl_mod,only: correl
  use nmrm,only:nmr
  use tmd,only: tmdinit
  use dims,only: dimsinit
  use travelmain,only: trek
  use nbndcc_util
#if KEY_RXNCOR==1
  use rxpath,only:pathc            
#endif
#if KEY_RXNCOR==1
  use rxdefs,only:rxpars           
#endif
  use replica_mod,only:replica
#if KEY_DYNVV2==1
  use tpvv2,only: tpcontrol         
#endif
  use resdist,only:redset
  use machutil,only:jobdat
#if KEY_CHARMMRATE==1
  use charmmrate_mod,only:charmmrate   
#endif
#if KEY_MC==1
  use mcc, only: mccall               
#endif
#if KEY_MC==1
  use mcmoveio, only: moverd, movewr  
#endif
#if KEY_MC==1
  use mcmoveln, only: moveln          
#endif
#if KEY_MC==1
  use mcmvad                          
#endif
# if KEY_ACTBOND==1
  use eintern_fast, only: bactiv
#endif
  use lonepr
  use mtp_fcm, only: mtp
  use mtpl_fcm
  use ediff, only: ediffop
  use enbond_mod
  use eutil
  use fitchg
#if KEY_MCMA==1
  use mcmamod                         
#endif
#if KEY_OPENMM==1
  use omm_ctrl, only : omm_command, omm_system_changed
#endif
  use molvco_m
  use pert_mod
  use qub_m
  use quantm,only:qmused_quantum
  use testch_m
  use select
  use shapes
  use tbmts
  use varcutm
  use vibran_m
#if KEY_DOMDEC==1
  use domdec,only:domdec_com          
#endif
  use linkatom
  use drude
#if KEY_PARALLEL==1
  use parallel,only:mynod,mynodg
  use paral4,only: setcpustruc, test_po
#endif
#if KEY_ENSEMBLE==1
  use ensemble,only:nensem,old_mynod,whoiam,ensprint     
#endif
#if KEY_ABPO==1
  use abpo_ltm  
#endif
#if KEY_ABPO==1
  use abpo,only:abpo_cntrl 
#endif
  use prssre
#if KEY_GAMUS==1
  use gamusmodule,only:gamusinit 
#endif
#if KEY_MULTICOM==1 /*  VO : stringm v */
  use multicom, only: multicom_main
  use ifstack                          ! VO:  parsing conditionals in parallel
#endif
  use zmodule
#if KEY_PNM==1 /* VO: Plasic Network Model */
  use pnm, only : pnm_main
#endif
#if KEY_LARMORD==1
use larmord, only : setup_larmord
#endif
#if KEY_SSNMR==1
  use ssnmr
#endif
#if KEY_RDC==1
  use rdc, only : rdcset
#endif
use gopair, only : GoPair_setup
use freeene_calc, only: frencalc, qmfix, mkdummy
use keywords, only: print_keys
use cstuff, only: unbuffer_stdout
use extbond ! Pezzella 15.05.2019
use triakern ! Pezzella 10.12.2019
use fullkern ! Pezzella 11.07.2020
use opencl_parse_mod, only: ocl_parse
#ifdef KEY_RESIZE
use deriv
use resize,only: resize_array,set_rszf
#endif
  implicit none
  character(len=*) comlyn
  integer comlen
  logical lused

  !     The following are local variables.

#if KEY_TSALLIS==1
  integer   err  
#endif
  integer   freeat
  integer  :: i,ierror
  integer   idm1,idm2,idm3
  integer   istart,itemp
  integer   natiml
  integer   nsgiml

  real(chm_real)    xtmp

  logical    lcomp
  logical   qform,qopen,qwrite

  character(len=4)  wrd
  character(len=6)  access
  character(len=12) form

  integer  tlen
  integer,parameter :: tmax=50
  character(len=tmax) tname

  integer :: icycle=0
  real(chm_real),dimension(1) :: ddv_dummy

#ifdef KEY_RESIZE
  real(chm_real):: cutim0
#endif  

  lused=.true.

#if KEY_MULTICOM==1 /*  VO stringm : conditional evaluation in parallel */
  if (.not.peek_if()) then
   comlen=0
   return ! this if_then_else_endif block disabled on this node
  endif
#endif /* VO */

  !     Shorten word to four characters or less and pad with blanks
  !     main conditional for processing commands

  wrd=nexta4(comlyn,comlen)

  cmds: select case(wrd)
  case('    ') cmds
  case('ANAL') cmds
     call anal(comlyn,comlen)
  case('ATLI') cmds
     call trime(comlyn,comlen)
     call copyst(altcom,mxalsz,altlen,comlyn,comlen)
     comlen=0
  case('AUTO') cmds
     call autogen(comlyn,comlen)
#if KEY_BLADE == 1
  case('BLAD') cmds
     call blade_command(comlyn, comlen)
#endif 
  case('BLOC') cmds
     call block(comlyn,comlen)
#if KEY_TSALLIS==1
  case('TSAL') cmds
     wrd = nexta4(comlyn, comlen)
     if (wrd  ==  'TORS') THEN
        qttsall = .true.
        allocate(iascale(natom), stat=err)
        if (err /= 0) then
           call wrndie(-4,'<charmm_main>', &
                'Abort: unable to allocate IASCALE')
        endif
        wrd = nexta4(comlyn,comlen)
        if (wrd  ==  'SELE') THEN
           ! Assign selection
           call joinwd(comlyn,mxcmsz,comlen,wrd,4)
           call selcta(comlyn,comlen,iascale, &
                x,y,z,wmain,.true.)
        endif
     endif
#endif 
  case('NOSE') cmds
     call nosect(comlyn,comlen)
  case('MTS ') cmds
     call mts(comlyn,comlen)
  case('BOUN') cmds
     call bound(comlyn,comlen)
  case('BUIL') cmds
     call intcor(comlyn,comlen)
  case('CADP') cmds
     call cadini(comlyn,comlen)
  case('CLEA') cmds
     call clear_something
  case('CONS') cmds
     call cstran(comlyn,comlen)
  case('COOR') cmds
     call corcom(comlyn,comlen)
  case('CORR') cmds
     call correl(0,ddv_dummy)
  case('CRYS') cmds
     call crystl
  case('DIME','RESI' ) cmds
     call wrndie(-1,'<CHARMM>', &
          'Use DIMEnsion change only as first CHARMM command')
#if KEY_DIMS==1
  case('DIMS') cmds
     call dimsinit(comlyn,comlen)
#endif 
  case('ENSE') cmds    
     call enscmd(comlyn,comlen)

#if KEY_CHARMMRATE==1
  case('POLY','RATE') cmds
     call charmmrate(comlyn,comlen)
#endif 
  case('NBAC') cmds
     call nbactv(comlyn,comlen)
#if KEY_ACTBOND==1
  case('BACT') cmds
     call bactiv(comlyn,comlen)
#endif 
  case('MKCL') cmds
     call mkclust(comlyn,comlen)
#if KEY_ESTATS==1
  case('ESTA') cmds
     call estats(comlyn,comlen)
#endif /* */
     !**********************************************************
  case('MKPR') cmds
     call mkpres0
  case('CVEL') cmds
     if(prnlev >= 2) write(outu, &
          '(/,1X,''CVELOCI>  SETING UP VECTOR DIRECTIVES'',/)')
     call cveloci(comlyn,comlen)
  case('DEAD') cmds
     !---- Procedure PROCESS-DEADLINE-COMMAND
     cpulim=gtrmf(comlyn,comlen,'CPU',zero)*60.d0
     if(cpulim > 0.0) call jobdat(xtmp,cpuini,idm1,idm2,idm3)
     xtmp=gtrmf(comlyn,comlen,'CLOC',minone)
     if (xtmp >= 0.0) then
        deadhr=xtmp
        deadmn=100.*(xtmp-deadhr)
        pasmid=0
        call chklim
        if(atlim) pasmid=pasmid+1
     else
        deadhr=-1
     endif
     !---- End Procedure PROCESS-DEADLINE-COMMAND
  case('DELE') cmds
     !     call from Subroutine mmff_io.
     call deltic(comlyn,comlen)
  case('DONO','ACCE') cmds
     call edhbatoms(comlyn,comlen,wrd)
#if KEY_TMD==1
  case('TMDI') cmds
     call tmdinit(comlyn,comlen)
#endif 
  case('DYNA') cmds
#if KEY_ABPO==1
     if (q_abpo) then
        call abpo_cntrl(comlyn,comlen)
     else
        call dynopt(comlyn,comlen)
     endif
#else /**/
     call dynopt(comlyn,comlen)
#endif /* ABPO*/
  case('EDIF') cmds
     call ediffop(comlyn,comlen,icycle)
  case('ENER') cmds
#ifdef KEY_RESIZE
     cutim0=cutim ! keep previous cutim
     cutim=GTRMF(COMLYN,COMLEN,'CUTI',CUTIM) ! update cutim
     if (cutim .gt. 0.0) then
        if (abs(cutim0-cutim) .gt. HALF ) then ! new image setting
           call set_rszf('miscom.F90','MAINCOMX',.true.)
        else ! use existing image info
           call set_rszf('miscom.F90','MAINCOMX',.false.)
        endif
     endif
#endif          
     call gete0('ENER', comlyn, comlen)
#ifdef KEY_RESIZE
     call resize_array('miscom.F90','MAINCOMX')
#endif     
#if KEY_CHEQ==1
  case('FQBA') cmds
     call nosefq(comlyn,comlen)
  case('CHEQ') cmds
     call cheqprep(comlyn,comlen)
#endif 
  case('FLUC') cmds
#if KEY_FLUCQ==1
     call fqinit(comlyn,comlen)
#else /**/
     CALL WRNDIE(-1,'<CHARMM>','FLUCQ code is not compiled.')
#endif 
  case('FOUR') cmds
     call parse4d(comlyn, comlen)
#if KEY_FACTS==1
  case('FACT') cmds
     call fctini(comlyn,comlen)
#endif 
  case('GBOR') cmds
     call genborn_set(comlyn, comlen)
  case('GBIM') cmds
     call setgbim(comlyn, comlen)
  case('GBMV') cmds
     call gbmv_set(comlyn, comlen)
  case('MSES') cmds
     call mses_set(comlyn, comlen)
  case('GBSW') cmds
     call gbsw_set(comlyn, comlen)
  case('GOPA') cmds
     call GoPair_setup(comlyn, comlen)

#if KEY_DENBIAS==1
  case('DBIA') cmds
     call denbias_set(comlyn, comlen)
#endif

#if KEY_EPMF==1
  case('EPMF') cmds
     call epmf_set(comlyn,comlen)
#endif 
#if KEY_PRIMO==1
   case('PRIM') cmds
     call primo_set(comlyn,comlen)
#endif 
  case('GENE') cmds
     call handle_generate_command(comlyn, comlen, istart)
  case('GETE') cmds
     call gete0('GETE', comlyn, comlen)
  case('GAME') cmds
     qmused_gamess=.true.
     call gukini(comlyn,comlen)
  case('NWCH') cmds
     qmused_nwchem=.true.
     call gukini(comlyn,comlen)
  case('QCHE') cmds
     qmused_qchem=.true.
     call gukini(comlyn,comlen)
! Guanhua_QC_UW1111 / JZ_UW12
  case('GAUS') cmds
     qmused_g09=.true.
     call gukini(comlyn,comlen)
  case('QTUR') cmds
     qmused_turbo=.true.
     call gukini(comlyn,comlen)

#if KEY_NOGRAPHICS==0
  case('GRAP') cmds
     call graphx
#endif 
  case('HBON') cmds
     if(ihbfrq == 0) ihbfrq=999
     call update(comlyn,comlen,x,y,z,wmain,.true., &
          .false.,.false.,.true.,.false., &
          0,[zero],[zero],[zero],[zero],[zero],[zero])
     if(ihbfrq == 999) ihbfrq=0
#ifdef KEY_RESIZE
     call resize_array('miscom.F90','MAINCOMX')
#endif     
  case('HBTR') cmds
     call hbtrim
  case('HBUI') cmds
     call hbuild(comlyn,comlen)
  case('IC  ') cmds
     call intcor(comlyn,comlen)
  case('INQU') cmds
     !---- Procedure PROCESS-INQUIRE-FILE-COMMAND
     !     inquire all open files
     if(prnlev >= 2) write(outu,'(a)') &
          ' CHARMM: list of open files:'
     do i=1,99
        call vinqre('UNIT',tname,tmax,tlen,qopen,qform,qwrite,i)
        if (qopen) then
           if (qform) then
              form=' formatted'
           else
              form=' unformatted'
           endif
           if (qwrite) then
              access=' write'
           else
              access=' read'
           endif
           if(prnlev >= 2) write(outu,'(9x,i3,1x,3a)') &
                i,tname(1:tlen),access,form
        endif
     enddo
     !---- End Procedure PROCESS-INQUIRE-FILE-COMMAND
  case('INTE') cmds
     call inteop(comlyn,comlen,icycle)
  case('JOIN') cmds
     call joinsg(comlyn,comlen)
  case('LONE') cmds
     call loneprs(comlyn,comlen)
#if KEY_RXNCONS==1
  case('RCON') cmds
     call rxconsps(comlyn, comlen)
#endif 
#if KEY_RXNCOR==1
  case('LUPO') cmds
     call lupopt(comlyn,comlen)
#endif 
#if KEY_CSA==1 || KEY_DISTENE==1
  case('MAST') cmds
     call masterdstr(comlyn,comlen)
  case('ETRA') cmds
     call etraj(comlyn,comlen)
  case('RECE') cmds
     call calcrece(comlyn,comlen)
  case('TRAN') cmds
     call calctran(comlyn,comlen)
#if KEY_CSA==1
  case('CSA ') cmds
     call csacntrl(comlyn,comlen)
#endif 
#endif 
  case('MC  ') cmds
#if KEY_MC==1 /*mc1*/
     call mccall(comlyn,comlen)
#else /*  (mc1)*/
     CALL WRNDIE(-1,'<CHARMM>','MC code is not compiled.')
#endif /* (mc1)*/
  case('GNN ') cmds
     call gnncall(comlyn, comlen)
  case('MERG') cmds
     CALL TMERGE(COMLYN,COMLEN,TITLEB,NTITLB)
  case('MINI') cmds
     CALL MINMIZ(COMLYN,COMLEN)
  case('MLTE') cmds           !bt_080627
     CALL MULTE0(COMLYN,COMLEN)
  case('MMFP') cmds
#if KEY_OPENMM==1
     call omm_system_changed()
#endif
     CALL MMFP0
     !----namkh 02/03/03
  case('MNDO') cmds
#if KEY_MNDO97==1
     qmused_mndo97=.true.
     CALL MNDINI(COMLYN,COMLEN)
#endif 
  case('MOLV') cmds
     CALL MOLVCO(COMLYN,COMLEN)
  case('MONI') cmds
     !---- Procedure PROCESS-MONITOR-COMMANDS
     WRD=NEXTA4(COMLYN,COMLEN)
     IF (WRD == 'DIHE') THEN
        CALL MONDIH(COMLYN,COMLEN)
     ELSE
        CALL WRNDIE(0,'<CHARMM>','UNRECOGNIZED MONITOR OPTION')
     ENDIF
     !---- End Procedure PROCESS-MONITOR-COMMANDS
  case('MOVE') cmds
#if KEY_MC==1 /*mc2*/
     ! ----- Procedure PROCESS-MOVE-COMMANDS
     wrd=nexta4(comlyn,comlen)
     movecmds: select case(wrd)
     case('ADD ') movecmds
        call movead(comlyn,comlen)
     case('LINK') movecmds
        call moveln(comlyn,comlen)
     case('DELE') movecmds
        call movedl(comlyn,comlen)
     case('EDIT') movecmds
        call moveed(comlyn,comlen)
     case('READ') movecmds
        call moverd(comlyn,comlen)
     case('WRIT')  movecmds
        call movewr(comlyn,comlen)
     case default movecmds
        CALL WRNDIE(0,'<CHARMM>','UNRECOGNIZED MOVE OPTION')
     end select movecmds
     ! ----- End Procedure PROCESS-MOVE-COMMANDS
#else /*   (mc2)*/
     CALL WRNDIE(-1,'<CHARMM>','MC code is not compiled.')
#endif /*  (mc2)*/
  case('MLTC') cmds
     call mltcanon_prs(comlyn,comlen)
#if KEY_MSCALE==1
  case('MSCA') cmds
     call mscale(0,comlyn,comlen)
  case('SERV') cmds
     call mscale(1,comlyn,comlen)
#endif 
  case('NBON') cmds
     if(inbfrq == 0) inbfrq=999
#ifdef KEY_RESIZE
     cutim0=cutim ! keep previous cutim
     cutim=GTRMF(COMLYN,COMLEN,'CUTI',CUTIM) ! update cutim
     if (cutim .gt. 0.0) then
        if (abs(cutim0-cutim) .gt. HALF ) then ! new image setting
           call set_rszf('miscom.F90','MAINCOMX',.true.)
        else ! use existing image info
           call set_rszf('miscom.F90','MAINCOMX',.false.)
        endif
     endif
#endif          
     call update(comlyn,comlen,x,y,z,wmain,.true., &
          .false.,.true.,.false.,.true.,0,[zero],[zero],[zero],[zero],[zero],[zero])
     if(inbfrq == 999) inbfrq=0
#ifdef KEY_RESIZE
     call resize_array('miscom.F90','MAINCOMX')
#endif     
  case('NOE ') cmds
     call noeset
  case('RESD') cmds
     call redset
  case('NMR ') cmds
     call nmr
  case('TAMD') cmds
     call tamd
  case('ZMAT') cmds
     call zmtrx
#if KEY_MCMA==1 /*mcma*/
  case('MCMA') cmds
     call mcma
#endif /* (mcma)*/
  case('MMQM') cmds
#if KEY_NOMISC==1 /*mmqm*/
     CALL WRNDIE(-1,'<CHARMM>','MMQM code is not compiled.')
#else /* (mmqm)*/
     call g94ini
#endif /* (mmqm)*/
  case('MTP ') cmds
     !  atomic multipole moments mtp module
     call mtp
  case('MTPL') cmds
     !  atomic multipole moments mtpl module--uses local axis systems for
     !  arbitrarily large molecules.
#if KEY_MTPL==1
     call mtpl
#else
     CALL WRNDIE(-1,'<CHARMM>','MTPL code is not compiled.')
#endif /* mtpl */
  case('OCL ') cmds  ! parse opencl commands
     call ocl_parse(comlyn, comlen)
#if KEY_OVERLAP==1
  case('OLAP') cmds
     call olapcmd(comlyn,comlen)
#endif
#if KEY_OPENMM==1
  case('OMM ') cmds
     call omm_command(comlyn, comlen)
#endif 
#if KEY_PARCMD==1
  case('PARA') cmds
     call parcmd(comlyn,comlen)
#endif 
  case('PATC') cmds
     call patch(comlyn,comlen)

! phmd dependent on gbsw or gbsv
#if KEY_PHMD==1
  case('PHMD') cmds
     call startphmd(comlyn,comlen)
  case('PHTE') cmds
     call dophmdtest(comlyn, comlen)
#endif 
#if KEY_PBEQ==1
  case('PBEQ') cmds
     call pbeq0
#endif 
#if KEY_GRID==1
  case('GRID') cmds
     call gridset(comlyn, comlen)
#endif
#if KEY_FFTDOCK == 1     
  case('FFTG') cmds
     call fft_dock_set(comlyn, comlen)
#if KEY_OPENMM == 1
  case('OMMD') cmds
     call openmm_dock_set(comlyn, comlen)
#endif /* OPENMM */
#endif /* FFTDOCK */
  case('PERT') cmds
     call perts(comlyn,comlen)
#if KEY_POLAR==1
  case('POLA') cmds
     call polar0
#endif 
#if KEY_PATHINT==1
  case('PINT') cmds
     call pint_init(comlyn,comlen)
#endif 
#if KEY_PNM==1
  case('PNM') cmds
     call pnm_main(comlyn,comlen)
#endif 
  case('PRES') cmds
     call getprs(comlyn,comlen)
  case('PRIN') cmds
     call mainio(wrd)
  case('PULL') cmds
     call pull_setup(comlyn,comlen)
  case('QUAN') cmds
! Because of this we cannot compile SQUANTM & QUANTUM at the same time
! fixing this later ...
#if KEY_SQUANTM==1
     qmused_squantm=.true.
     call sqmini(comlyn,comlen)
#else /**/
     qmused_quantum=.true.
     call qmdefn(comlyn,comlen)
#endif 
#if KEY_QMMMSEMI==1
  case('IFQN') cmds
     call qmmm_startup(comlyn,comlen,natom,x,y,z)
#endif 
  case('READ') cmds
     call mainio(wrd)
#if KEY_GENETIC==1
     !---- 12-Jan-98, CLBIII
  case('GALG') cmds
     IF (IndxA(comLyn,comLen,'SETU') > 0) then
        CALL Genetic_Alg(.true., .false.)
     ELSEIF (IndxA(comLyn,comLen,'EVOL') > 0) then
        CALL Genetic_Alg(.false.,.true.)
     endif
#endif 
  case('RENA') cmds
     !---- Procedure PROCESS-RENAME-COMMAND
     call crename(comlyn,comlen)
  case('REPL') cmds
     call replica
  case('REPD') cmds
     call repdstrmain
  case('RESE') cmds
     !---- Procedure PROCESS-RESET-COMMAND
     call iniall
     if(prnlev >= 2) write(outu, &
          '(/,1X,''**** CHARMM DATA STRUCTURES RESET ****'',/)')
     !---- End Procedure PROCESS-RESET-COMMAND
     !-----cb3 added prefx control for compilation
#if KEY_HQBM==1
  case('HQBM') cmds  ! 02-Jul-1997
     call hqbmini
#endif 
#if KEY_AFM==1
  case('AFM') cmds   ! 20-Jun-2003
     call afmini
#endif 
#if KEY_AXD==1
  case('AXD') cmds
     call axdini(natom)
#endif 
  case('RISM') cmds
     call rismcmd
#if KEY_RPATH==1
  case('RPAT') cmds
     call pathps(comlyn,comlen)
#endif 
#if KEY_RXNCOR==1
  case('PATH') cmds
     call pathc(x,y,z,xcomp,ycomp,zcomp,wmain,comlyn,comlen)
  case('RXNC') cmds
     call rxpars
#endif 
  case('SBOU') cmds
     !---- Procedure PROCESS-SOLVENT-BOUNDARY-COMMANDS
#if KEY_NOMISC==1
     CALL WRNDIE(-1,'<CHARMM>','SBOUND code is not compiled.')
#else /**/
     wrd=nexta4(comlyn,comlen)
     solvboun: select case(wrd)
     case('POTE') solvboun
        call sbintg
     case('SET ') solvboun
        call bndset
     case('READ') solvboun
        call sbread
     end select solvboun
#endif 
     !---- End Procedure PROCESS-SOLVENT-BOUNDARY-COMMANDS
  case('SCAL') cmds
     call scalar
#if KEY_SCCDFTB==1
     !---- SCCDFTB CODE (QCME)
  case('SCCD') cmds
     qmused_sccdftb=.true.
     call scctbini(comlyn,comlen)
#endif 
  case('SHAK') cmds
     call shkcom(comlyn, comlen)
#if KEY_SHAPES==1 /*shpcom*/
  case('SHAP') cmds
     call shpcom(comlyn, comlen)
#endif /* (shpcom)*/
  case('SHOW') cmds
     comlen=0
  case('TEST') cmds
     call testch(comlyn,comlen)
#if KEY_PARALLEL==1
  case('TSPO') cmds
     call test_po
  case('CPUS') cmds
     call setcpustruc
#endif

  case('TRAJ') cmds
     !---- Procedure PROCESS-TRAJECTORY-COMMAND
     if (indxa(comlyn,comlen,'READ') > 0) THEN
        if (indxa(comlyn,comlen,'COMP') <= 0) THEN
           call reatrj(x,y,z)
        else
           call reatrj(xcomp,ycomp,zcomp)
        endif
     else if (indxa(comlyn,comlen,'WRIT') > 0) THEN
        if (indxa(comlyn,comlen,'COMP') <= 0) THEN
           call wrttrj(x,y,z)
        else
           call wrttrj(xcomp,ycomp,zcomp)
        endif
     else
        call trajio
     endif
     !---- End Procedure PROCESS-TRAJECTORY-COMMAND
#if KEY_ADUMB==1
  case('UMBR') cmds
     call umban(comlyn,comlen)
#endif 
#if KEY_GAMUS==1
  case('GAMU','GUMB')
     CALL GAMUSINIT(COMLYN,COMLEN)
#endif 
     !mf switch stdout to unbuffered I/O
  case('UNBU') cmds
#if KEY_UNIX==1
     ierror = unbuffer_stdout()
     if (ierror .ne. 0) then
        call wrndie(0, '<CHARMM>', 'UNBU command failed')
     end if
#endif
  case('TREK','TRAV') cmds
     call trek(comlyn,comlen)
  case('TSM ') cmds
     call tsms
  case('WRIT') cmds
     call mainio(wrd)
  case('UPDA') cmds
     !---- Procedure PROCESS-UPDATE-COMMAND
     !
     !     UPDATE NBOND, HBOND, AND CODES LISTS
     !
#ifdef KEY_RESIZE
     cutim0=cutim ! keep previous cutim
     cutim=GTRMF(COMLYN,COMLEN,'CUTI',CUTIM) ! update cutim
     if (cutim .gt. 0.0) then
        if (abs(cutim0-cutim) .gt. HALF ) then ! new image setting
           call set_rszf('miscom.F90','MAINCOMX',.true.)
        else ! use existing image info
           call set_rszf('miscom.F90','MAINCOMX',.false.)
        endif
     endif
#endif     
     lcomp=(indxa(comlyn,comlen,'COMP') > 0)
     if (lcomp) then
        call update(comlyn,comlen,xcomp,ycomp,zcomp,wcomp,.true., &
             .true.,.true.,.true.,.true.,0,[zero],[zero],[zero],[zero],[zero],[zero])
     else
        call update(comlyn,comlen,x,y,z,wmain,.true., &
             .true.,.true.,.true.,.true.,0,[zero],[zero],[zero],[zero],[zero],[zero])
     endif
#ifdef KEY_RESIZE
     call resize_array('miscom.F90','MAINCOMX')
#endif     
     !---- End Procedure PROCESS-UPDATE-COMMAND
  case('VARC') cmds
     call setup_varcut(comlyn,comlen,natom)
  case('VIBR') cmds
     call vibopt(comlyn,comlen)
  case('WHAM') cmds
     call wham0
  case('XTBD') cmds ! Pezzella 15.04.2019
     call ext_bond_set(comlyn,comlen)
  case('TKRN') cmds ! Pezzella 10.12.2019
     call tria_kern_set(comlyn,comlen)
 case('FLMK') cmds  ! Pezzela 11.07.2020
     call fullkern_set(comlyn,comlen)
  case('IMAG') cmds
     !---- Procedure PROCESS-IMAGE-SPECIFY-COMMAND
     !
     !     SPECIFY IMAGE CENTERING OPTIONS
     if(ntrans <= 0) then
        CALL WRNDIE(-2,'<CHARMM>','IMAGES NEED TO BE PRESENT')
        return
     endif
     call imspec(comlyn,comlen,bimag%imcenf, &
          limcen,imxcen,imycen,imzcen,natom,x,y,z,wmain)
     !---- End Procedure PROCESS-IMAGE-SPECIFY-COMMAND
  case('IMPA') cmds
     ! hwm: check for resize (todo)
     call impatc(comlyn,comlen)
#if KEY_NOGRAPHICS==0 && KEY_NODISPLAY==0
  case('DRAW') cmds
     !       For XDISPLAY
     call drawit(-1,x,y,z,xcomp,ycomp,zcomp)
#endif 
#if KEY_NOMISC==0
  case('BARI') cmds
     !---- Procedure PROCESS-BARRIER-COMMAND
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          ' CHARMM> BARRier command is no longer supported.'
     IF(WRNLEV >= 2) WRITE(OUTU,'(A)') &
          '         Use the GRADient option of MINImize commands.'
     CALL WRNDIE(-5,'<CHARMM>','Unsupported command: '//WRD)
     !---- End Procedure PROCESS-BARRIER-COMMAND
  case('RMSD') cmds
     !---- Procedure PROCESS-RMSDYNAMICS-COMMAND
     natiml=natom
     nsgiml=nseg
     if(indxa(comlyn,comlen,'IMAG') > 0) THEN
        natiml=natomt
        nsgiml=nsegt
     endif

     call rmsdyn('main',x,y,z,wmain,xcomp,ycomp,zcomp,wcomp,natiml,amass)

     !---- End Procedure PROCESS-RMSDYNAMICS-COMMAND
#endif 
#if KEY_QUANTUM==1 || KEY_GAMESS==1 || KEY_GAMESSUK==1 || KEY_CADPAC==1 || \
    KEY_SCCDFTB==1 || KEY_QCHEM==1 || KEY_MNDO97==1 || KEY_SQUANTM==1 || \
    KEY_QTURBO==1 || KEY_G09==1
  case('ADDL') cmds
     call addlnat(outu)
  case('RELL') cmds
     call rellnat(outu)
#endif 
#if KEY_QUANTUM==1
  case('MULL') cmds
     call mullik(comlyn,comlen)
#endif 
#if KEY_QUANTUM==1 || KEY_SCCDFTB==1 || KEY_SQUANTM==1 || KEY_QCHEM==1 || KEY_GAMESSUK==1 || KEY_QTURBO==1
  case('QUB ') cmds
     call qub(comlyn,comlen)
#endif 
#if KEY_SQUANTM==1
  case('SQUA') cmds
     qmused_squantm=.true.
     call sqmini(comlyn,comlen)
#endif 
#if KEY_DMCONS==1
  case('DMCO') cmds
     call dmcset
#endif 
#if KEY_RGYCONS==1
  case('RGYR') cmds
     call rgyset
#endif 
  case('ETEN') cmds
     call etenset
  case('ETSR') cmds
     call etsrset
#if KEY_ASPENER==1
  case('EEF1') cmds
     call eef1
#endif 
#if KEY_SCPISM==1
  case('SCPI') cmds
     call scpparse
#endif 
#if KEY_SASAE==1
  case('SASA') cmds
     call sasini(comlyn,comlen)
#endif 
  case('PREF') cmds
     call print_keys()
#if KEY_EMAP==1
  case('EMAP') cmds
     call emapopt(comlyn, comlen)
#endif 
     ! BEGIN DRUDE (B. Roux and G. Lamoureux)
  case('DRUD') cmds
     IF (Indx(comlyn,comlen,'L_WALL',6) > 0) then
        ! Lei Huang, impose hard wall constraint on drude bond length
        L_WALL = GTRMF(COMLYN,COMLEN,'L_WALL',0.2)
     
        IF(L_WALL .gt. 0.0) THEN
          QHARDWALL = .true.
#if KEY_PARALLEL==1
          IF(mynod .eq. 0) then
#endif 
            WRITE(outu,*)   &
            'Hard wall constraint on drude bond length is turned ON. L_WALL = ',L_WALL
#if KEY_PARALLEL==1
          endif
#endif
        ELSE
          QHARDWALL = .false.
#if KEY_PARALLEL==1
          IF(mynod .eq. 0) then
#endif
            WRITE(outu,*)   &
            'Hard wall constraint on drude bond length is turned OFF.'
#if KEY_PARALLEL==1
          endif
#endif
        ENDIF
     ELSE
        call drude0(comlyn,comlen)
     ENDIF

  case('TPCO','NOS2') cmds
#if KEY_DYNVV2==1
     call tpcontrol(comlyn,comlen)                     
#endif
#if KEY_DYNVV2==0
     CALL WRNDIE(0,'<CHARMM>','DYNA VV2 not compiled') 
#endif

  case('FITC') cmds
     call fitcharge

  case('FITP') cmds
     call fitparam

#if KEY_RDFSOL==1
  case('RDFS') cmds
     !       new solvent radial distribution function module
     call rdfsol
#endif 
#if KEY_SHELL==1
  case('SHEL') cmds
     !       shell decomposition module
     call shlini
#endif 

  case('ZMOD') cmds
     call zerom

#if KEY_PROTO==1 /*proto*/
  case('PROT') cmds
     !       prototype functions
     call proto
#endif /*   (proto)*/

  case('PIPF','PFBA') cmds
#if KEY_PIPF==1
     if (wrd == 'PIPF') then
        call pfprep(comlyn,comlen)
     else if (wrd == 'PFBA') then
        call nosepf(comlyn,comlen)
     end if
#else /**/
     CALL WRNDIE(-1,'<CHARMM>','PIPF code is not compiled.')
#endif 
  case('RUSH') cmds
     call rush(comlyn,comlen)
#if KEY_CORSOL==1 /*corsol*/
  case('CORS') cmds
     !        solvent correlation functions
     call corsol
#endif /*     (corsol)*/
#if KEY_TORQUE==1 /*torque*/
  case('TORQ') cmds
     !       process TORQue command
     call torque_parse
#endif /* (torque)*/
  case('CNSP') cmds
     ! process CNSPh command
     call getphresidues(comlyn,comlen)
#if KEY_LARMORD==1
  case('LARM') cmds
     ! set-up calculation of chemical shifts with LarmorD
     call setup_larmord()
#endif
#if KEY_EDS==1
  case('EDS') cmds
     ! process EDS command
     call process_eds(comlyn,comlen)
#endif 
#if KEY_DOMDEC==1
  case('DOMD') cmds
     call domdec_com(comlyn, comlen)
#endif
#if KEY_STRINGM==1 /*  VO stringm v */
  case('STRI') cmds
    CALL SM_MAIN(COMLYN,COMLEN) ! string method parser
#endif
#if KEY_MULTICOM==1 /*  VO stringm communication v */
  case('MCOM') cmds
    CALL MULTICOM_MAIN(COMLYN,COMLEN) ! manipulate communicators for stringm
#endif /* VO ^ */
  case('FREN') cmds
    CALL FRENCALC(COMLYN,COMLEN)
  case('QMFI') cmds
    CALL QMFIX(COMLYN,COMLEN)
  case('MKDU') cmds
    CALL MKDUMMY(COMLYN,COMLEN) ! Write a new parameter file that turns selected atoms to dummy atoms

#if KEY_RDC==1
   case('RDC ') cmds
      call rdcset
#endif
#if KEY_SSNMR==1
  case('CCS ') cmds
    CALL CSSET
#endif
! #if KEY_MOBHY==1
  case('MOBH') cmds
    CALL MOBHY
! #endif
     !
  case default cmds
     lused=.false.
#if KEY_ENSEMBLE==1
     if(nensem > 1 ) write(outu,'(3(a,i3),2a)') &     
#endif
#if KEY_ENSEMBLE==1
          ">>> Ensemble ",whoiam," Node ",mynod," Worldnod ",old_mynod, &    
#endif
#if KEY_ENSEMBLE==1
          " ---- cmd problem = ",comlyn(1:comlen)       
#endif
     call wrndie(0,'<CHARMM>','Unrecognized command: '//WRD)
  end select cmds

  return

end subroutine maincomx

SUBROUTINE MISCOM(COMLYN,MXCMS2,COMLEN,LUSED)
  !-----------------------------------------------------------------------
  !     This routine processes many of the miscellaneous commands
  !     it may be called from any routine that processes commands.
  !
  !      Note: length of all character strings used is 4.
  !
  !     Control flow commands by David States
  !     Overhauled by Bernard R. Brooks and Axel Brunger, 1-MAR-83
  !     Further modifications by Leo Caves, 17-JAN-94 (parameter table)
  !


  ! 12/12/07 SL: use module cross to allocate memory for RMD arrays
#if KEY_RMD==1
  use cross, only: CROSSINIT, QCROS     
#endif
#if KEY_REPDSTR==1
  use repdstrmod
  use mpi
#endif
  use chm_kinds
  use dimens_fcm
  use number
  use cmdpar
  use ctitla
  use eutil
#if KEY_BLOCK==1
  use lambdam       
#endif
  use fast
#if KEY_LOOKUP==1
  use lookup,only:wwsetup        
#endif
  use machdep
  use param_store, only: write_real_params, write_int_params, write_str_params
  use select
  use stream
  use string
  use timerm
  use parallel
  use usermod,only: usersb
  use quick,only:quicka
  use aidxmod,only:aidx
  use repdstr
  use calc,only:calculator
  use clcg_mod,only: randspec, irandom
  use machutil,only:timre,timrb,daytim,wcpu,csystem
#if KEY_VALBOND==1
  use valbond, only: vbcomm   
#endif
  use mtp_fcm
  use mtpl_fcm

  use dcm_fcm,only: dcm

#if KEY_MMPT==1
  use mmpt_fcm,only:mmptinit                              
#endif
#if KEY_MSMMPT==1
  use msmmpt_fcm,only:msmmptinit
#endif
#if KEY_MRMD==1
  use mrmd_fcm,only:mrmd_init
#endif
#if KEY_MULTICOM==1
  use ifstack    ! VO: if/else/endif conditionals in srtingm communication
  use multicom_aux, only: MPI_COMM_PARSER, ME_PARSER, SIZE_PARSER
#endif

#if KEY_UNIX==1
  use cstuff, only: setenv
  use, intrinsic :: iso_c_binding, only: C_NULL_CHAR
#endif /* UNIX */
  
  implicit none

  character(len=*) comlyn
  integer   mxcms2,comlen
  logical   lused

  ! following temp. string sizes taken from cmdpar.f90
  character(len=mxtlen) toktmp
  character(len=mxvlen) valtmp
  ! a variable name LENVAL is declared downstream!
  integer lentok
  logical   eof, done, ok
#if KEY_MULTICOM==1 /*   VO : conditionals  */
  logical :: ok2                
#endif
  character(len=4)   wrd,junk
  integer lenvar,lenval,ipt
  integer,parameter :: maxenv=120
  character(len=maxenv) envval, envvar
  integer wd2len,wd3len
  integer,parameter :: wdmax=50
  character(len=wdmax) wrd2,wrd3
  integer   i, j, ipar, iunit, idummy, irwait
  integer   k,l,m,ii,jj,kk
  real(chm_real)    value, temp, increm
  ! for numerical parameter modifying operations
  integer   parop
  integer,parameter :: incr=1,decr=2,mult=3,divi=4,expo=5

  integer   lablen
  integer,parameter :: labmax=80
  character(len=labmax) lablyn
  integer ilevel

  character(len=1) :: sdblq='"'
  integer idum(1)
# if KEY_REPDSTR==1
  logical qreadstream
  integer ierr
  logical, dimension(:), allocatable :: if_results ! For checking that replicas stay in sync
# endif

  lused=.true.
  done=.false.
  eof=.false.
  ilevel=0

  loop101: do while(.not. done)
     done=.true.
     parOp = -1 
     wrd=nexta4(comlyn,comlen)

     !     main conditional for processing commands
     !
#    if KEY_REPDSTR==1
     if (nrepdstr.gt.1.and.repd_inside_if_block.gt.0) then
       ! We are inside an 'if' block. Check for further control statements.
       if (repd_if_block_stat(repd_inside_if_block).lt.1 .and. wrd.eq.'IF  ') then
         ! IF statement inside non-active 'if' block. Check if this is a new 'if' block.
         i = index(comlyn, 'THEN')
         if (i .gt. 0) then
           !write (outu,'(2(a,i6))') 'DBG: THEN idx= ', i, '  len comlyn= ', len_trim(comlyn)
           if ((len_trim(comlyn) - i) .lt. 4) then
             ! Nothing after the 'THEN'.
             ! We have encountered a nested 'if' block that should be skipped entirely
             repd_inside_if_block = repd_inside_if_block + 1
             repd_if_block_stat(repd_inside_if_block) = 0
           endif
         endif
       else if (wrd.eq.'ELSE') then
         ! Flip the stat for this block.
         !write(outu,'(a,i6)') 'DBG: Else encountered for if block ', repd_inside_if_block
         repd_if_block_stat(repd_inside_if_block) = repd_if_block_stat(repd_inside_if_block) * (-1)
         return
       else if (wrd.eq.'ENDI') then
         ! End this block
         !write(outu,'(a,i6)') 'DBG: End if block ', repd_inside_if_block
         repd_inside_if_block = repd_inside_if_block - 1
         return
       endif
       if (repd_if_block_stat(repd_inside_if_block).lt.1) then
         !write(outu,'(a)') 'DBG: This is a skipped command for me.'
         wrd = '    '
         comlen = 0
       !else
       !  write(outu,'(a)') 'DBG: This is an executed command for me.'
       endif
     endif
#    endif /* KEY_REPDSTR */
#if KEY_MULTICOM==1 /* (mcom)  VO : conditionals for stringm */
     !     if/else/endif followed by 'ELSE IF (peek_if)' MUST be processed first; 
     !     this is to permit the evaluation of multi-line nested conditionals in parallel
     !======================================================================================
      IF (WRD.EQ.'ELSE') THEN
        OK=peek_if() ! 'peek_if()' flags whether current if block (or main) is executed on this node
        OK2=pop_if() ! pop the 'if' stack
        if (.not.OK2) then ! errors popping stack?
          call wrndie(-2,' MISCOM>','UNEXPECTED ELSE STATEMENT.')
        else
        OK2=peek_if()
         IF(PRNLEV.GE.2.and.(OK.and.OK2)) & ! if previous statements executed, then the following cannot!
&             WRITE(OUTU,'(2A)') ' Skip to ELSE or ENDIF'
         call push_if(.not.OK.and.OK2) ! the opposite of the previous condition is true
        endif
     !
      ELSE IF (WRD.EQ.'ENDI') THEN
        OK=pop_if() ! pop the 'if' stack
        if (.not.OK) then 
          call wrndie(0,' MISCOM>','UNEXPECTED ENDIF STATEMENT.')
        endif
        COMLEN=0
     !
      ELSE IF (WRD.EQ.'IF  ') THEN
       OK2=peek_if()          ! execution flag at current level of nesting
       if (OK2) then          ! evaluate condition only if loop active, oherwise just push a 'false' (below)
!---- PROCESS-IF-COMMAND
        OK=.FALSE.
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK)
!     Do text replacement for a single character first operand.
!# <caves>-Aug-11-1993 (Leo Caves) if string (any size) is a symbol,
!                                  then substitute
! don't double substitute
        IF((FFOUR.NE. 'IF @').AND.(FFOUR.NE.'IF ?')) THEN
          IPAR=PARNUM(TOKTMP,LENTOK)
          IF(IPAR.GT.0) THEN
            IF(VALLEN(IPAR).GT.0) &
            CALL COPYST(TOKTMP,MXTLEN,LENTOK,VALNAM(IPAR),VALLEN(IPAR))
          ENDIF
        ENDIF
     !
        WRD=NEXTA4(COMLYN,COMLEN)
        CALL NEXTWD(COMLYN,COMLEN,VALTMP,MXVLEN,LENVAL)
     !
        IF(PRNLEV.GE.2) WRITE(OUTU,92) TOKTMP(1:LENTOK),&
                                       VALTMP(1:LENVAL)
  92    FORMAT(' Comparing "',A,'" and "',A,'".')
     !
        IF (WRD.EQ.'EQ  ' .OR. WRD.EQ.'.EQ.') THEN
          OK=EQST(TOKTMP,LENTOK,VALTMP,LENVAL)
        ELSE IF (WRD.EQ.'NE  ' .OR. WRD.EQ.'.NE.') THEN
          OK=.NOT.EQST(TOKTMP,LENTOK,VALTMP,LENVAL)
        ELSE
          VALUE=DECODF(TOKTMP,LENTOK)
          TEMP=DECODF(VALTMP,LENVAL)
          IF (WRD.EQ.'GT  ' .OR. WRD.EQ.'.GT.') THEN
            OK=VALUE.GT.TEMP
          ELSE IF (WRD.EQ.'GE  ' .OR. WRD.EQ.'.GE.') THEN
            OK=VALUE.GE.TEMP
          ELSE IF (WRD.EQ.'LT  ' .OR. WRD.EQ.'.LT.') THEN
            OK=VALUE.LT.TEMP
          ELSE IF (WRD.EQ.'LE  ' .OR. WRD.EQ.'.LE.') THEN
            OK=VALUE.LE.TEMP
          ELSE IF (WRD.EQ.'AE  ' .OR. WRD.EQ.'.AE.') THEN
            OK=ABS(VALUE-TEMP).LT.PT0001
          ELSE
! Better to trap this, an error could make affect a simulation
! without being noticed. L Nilsson
            call wrndie(-2,'<MISCOM>','Illegal IF test')
          ENDIF
        ENDIF
       else
     !       remove conditional without processing 
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK) ! remove token
        WRD=NEXTA4(COMLYN,COMLEN)                       ! remove operand
        CALL NEXTWD(COMLYN,COMLEN,VALTMP,MXVLEN,LENVAL) ! remove value
       endif
     ! for parallel : all nodes (not just root) scan the body of the conditional; the condition will in
     ! general, be false on some nodes (and true on others).
     !
       OK=OK.and.OK2           ! i.e. conditions at both levels need to be valid
       call push_if(OK) ! add level to the `if' stack 
     !
       IF (OK) THEN ! execute commands
          IF(PRNLEV.GE.2) WRITE(OUTU,94)
  94      FORMAT(' IF test evaluated as true.  Performing command')
          DONE=.false. ! tells miscom to re-start (see end of routine)
          CALL TRIMA(COMLYN,COMLEN)
          IF(COMLYN(1:4).EQ.'THEN') then 
            WRD=NEXTA4(COMLYN,COMLEN) ! i.e. remove 'THEN' from the line; this is a multi-line conditional terminated by ENDIF
          ELSE
           OK=pop_if() ! this is a one-line conditional without ELSE/ENDIF, so pop the stack (reuse 'OK' logical)
          ENDIF 
       ELSE ! I.E. (.NOT.OK) THEN ! will skip loop body
          WRD=NEXTA4(COMLYN,COMLEN)
          JUNK=NEXTA4(COMLYN,COMLEN)
     !
          IF((WRD.EQ.'THEN').AND.(JUNK.eq.'    '))THEN
              IF(PRNLEV.GE.2.and.OK2) WRITE(OUTU,'(A)') & ! write only if prev. level execution valid 
&               ' IF test evaluated as false.  Skip to ELSE or ENDIF'
          ELSE
             IF(PRNLEV.GE.2.and.OK2) WRITE(OUTU,'(A)') &
&               ' IF test evaluated as false.  Skipping command'
             COMLEN=0
             OK=pop_if() ! this is a one-line conditional without ELSE/ENDIF, so pop the stack
          ENDIF
       ENDIF ! OK/NOT OK
     !
      ELSE IF (.not.peek_if()) THEN
         COMLEN=0                ! ignore command line since execution is disabled on this node at this level of nesting
     !============================= DONE WITH CONDITIONALS ===================================================
      else & ! goes with "if" below
#endif /*(mcom) */
     if (wrd == 'BANN') then
        call header
        !
     else if (wrd == 'BOMB' .or. wrd.eq.'BOML') then
        kk=nexti(comlyn,comlen)
        if(kk <  -2)then
#if KEY_MULTICOM==1 /*  VO */
         if (mynod.eq.0) then                   
#endif
           if (prnlev >= 2) write(outu,'(a)') &
                ' MISCOM> Setting BOMLev < -2 is NOT a good idea.'
#if KEY_MULTICOM==1
         endif                                  
#endif
        endif
        bomlev=kk
        call set_param('BOMLEV',bomlev)
        !
     else if (wrd == 'CLOS') then
        call clolgu(comlyn,comlen,idummy)
        !
     ELSE IF (WRD == 'RXMD') THEN
#if KEY_RMD==1
        qcros = .true.
        call crossinit
#else /**/
        call wrndie(-1,'<MISCOM>', &
             'CROSS NOT AVAILABLE - NOT COMPILED WITH RMD FLAG')
#endif 
     ELSE IF (WRD == 'MRMD') THEN
#if KEY_MRMD==1
#if KEY_PARALLEL==1
        IF(MYNOD.EQ.0) &
#endif
        call mrmd_init
#else
        call wrndie(-1,'<MISCOM>', &
             'MRMD NOT AVAILABLE - NOT COMPILED WITH MRMD FLAG')
#endif
   ELSE IF (WRD.EQ.'VALB') THEN
#if KEY_VALBOND==1
         CALL VBCOMM()
#else /**/
         CALL WRNDIE(-1,'<MISCOM>', &
                'VALB NOT AVAILABLE - NOT COMPILED WITH VALBOND FLAG')
#endif 
     else if (wrd == 'MMPT') then
#if KEY_MMPT==1
        CALL MMPTINIT
#else /**/
        CALL wrndie(-1,'<MISCOM>', &
             'MMPT NOT AVAILABLE - NOT COMPILED WITH MMPT FLAG')
#endif  
     else if (wrd == 'MSPT') then
#if KEY_MSMMPT==1
        CALL MSMMPTINIT
#else /**/
        CALL wrndie(-1,'<MISCOM>', &
             'MSMMPT NOT AVAILABLE - NOT COMPILED WITH MSMMPT FLAG')
#endif 
     else if (wrd == 'DATE') then
        if(prnlev >= 2) then
           call daytim(k,l,m,ii,jj,kk)
           write(outu,54) k,l,m,ii,jj,kk
54         FORMAT(17x,'Current date: ',i2,'/',i2,'/',i2,' at ', &
                i2,':',i2,':',i2)
        endif

     else if (wrd == 'DECR') then
        !---- PROCESS-DECREMENT-COMMAND
        parop = decr

     else if (wrd == 'DEFI') then
        !---- process-define-command
        call filsky(comlyn,comlen,lused,.false.,idum)
        if (.not.(lused)) then
           call joinwd(comlyn,mxcms2,comlen,wrd,4)
           done=.true.
        endif
        !
     else if (wrd == 'DIVI') then
        !---- process-divide-command
        parop  = divi
        !
     else if (wrd == 'ECHU') then
        ! get output unit for echo command
        iecho=nexti(comlyn,comlen)
        if(iecho == 0) iecho=outu
     else if (wrd == 'ECHO') then
        ! just echo command line, without concern for its length, and skip first blank
        if(prnlev >= 2)then 
           if(comlen >= 2)then 
              write(iecho,'(a)') comlyn(2:comlen)
           else
              write(iecho,'(1x)')
           endif
        endif
        comlen=0

#if KEY_UNIX==1 /*unix*/
     else if (wrd == 'ENVI') then
        !---- process-environment-command
        call nextwd(comlyn, comlen, envvar, maxenv - 1, lenvar)
        call nextwd(comlyn, comlen, envval, maxenv - 1, lenval)

        ipt = 0
        do i = 1, lenvar
           if (envvar(i:i) /= '"') then  ! clean double quote
              ipt = ipt + 1
              envvar(ipt:ipt) = envvar(i:i)
           end if
        end do
        lenvar = ipt
        
        ipt = 0
        do i = 1, lenval
           if (envval(i:i) /= '"') then  ! clean double quote
              ipt = ipt + 1
              envval(ipt:ipt) = envval(i:i)
           end if
        end do
        lenval = ipt

        envvar = envvar(1:lenvar) // C_NULL_CHAR
        envval = envval(1:lenval) // C_NULL_CHAR
        i = setenv(envvar, envval, 1)

        if (i .ne. 0) then
           call wrndie(0, '<MISCOM>', &
                'failed to change environment variable')
        end if
#endif /* (unix)*/

     else if (wrd == 'EXPO') then
        !---- process-exponent-command
        parop  = expo

        !
     else if (wrd == 'FAST') then
        !---- process-fast-command
        faster=0
        wrd=nexta4(comlyn,comlen)
        fasteroption: select case(wrd)
        case('OFF ' ) fasteroption
           faster=-1
        case('GENE'  ) fasteroption
           faster=1
        case('ON  '  ) fasteroption
           faster=2
        case('DEFA') fasteroption
           faster=0
        case('EXPA') fasteroption
           faster=2
        case('SCAL') fasteroption
           CALL WRNDIE(0,'<MISCOM>','FAST option SCALar not available.')
        CASE('VECT') fasteroption
           CALL WRNDIE(0,'<MISCOM>','FAST option VECTor not available.')
        CASE('PARV') fasteroption
           CALL WRNDIE(0,'<MISCOM>','FAST option PARVec not available.')
        CASE('VPAR') fasteroption
           CALL WRNDIE(0,'<MISCOM>','FAST option VPAR not available.')
        CASE('CRAY') fasteroption
           CALL WRNDIE(0,'<MISCOM>','FAST option CRAY not available.')
#if KEY_PARALLEL==1
        case('DPCO') fasteroption
           dpcomm=1
        case('SPCO') fasteroption
           dpcomm=-1
#endif 
        case('BUCK') fasteroption
           IF(WRNLEV >= 2) WRITE (OUTU,'(A)') &
                ' MISCOM> Bucket fast routine is removed.'
        case default
           i=4
           faster=nexti(wrd,i)
           if(faster == 5) then
              if(wrnlev >= 2) write (outu,'(a)') &
                   ' MISCOM> Bucket fast routine is removed.'
           endif
           if(faster < -1 .or. faster > 3) then
              if(wrnlev >= 2) write (outu,'(a,i3)') &
                   ' MISCOM> Unknown FAST Level ',FASTER
              faster=0
              if(wrnlev >= 2) write(outu,'(a,i3)') &
                   '         FASTER will be set to ',FASTER
           endif
        end select fasteroption
        !
        !     set up LFAST value and inform the user the FASTER level set
        IF (FASTER < 0) THEN
           FASTER=-1
           LFAST =-1
#if KEY_PBOUND==1
           !     also turn of the hardwired periodic boundary in this case
           WRD='OFF'
           I=3
           CALL BOUND(WRD,I)
#endif 
        ENDIF
        LFAST=FASTER
        !
        IF (FASTER == -1) THEN
           IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
                ' MISCOM> FAST option: OFF (full feature routines)'
        ELSE IF (FASTER == 0) THEN
           IF(PRNLEV >= 2) WRITE (OUTU,'(A)')  &
                ' MISCOM> FAST option: DEFAULT (use best available)'
        ELSE IF (FASTER == 1) THEN
           IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
                ' MISCOM> FAST option: ON (generic fast routines)'
        ELSE IF (FASTER == 2) THEN
           IF(PRNLEV >= 2) WRITE (OUTU,'(A)') &
                ' MISCOM> FAST option: EXPANDED (limited fast routines)'
        ELSE
           IF(PRNLEV >= 2) WRITE (OUTU,'(A)')  &
                ' MISCOM> Bad FAST option: (coding error)'
        ENDIF
        CALL set_param('LFAST',LFAST)
        CALL set_param('FASTER',FASTER)
        !
     ELSE IF (WRD == 'FORM') THEN
        !---- PROCESS-FORM-COMMAND
        CALL NEXTWD(COMLYN,COMLEN,FMTWD,FMTMAX,FMTLEN)
        !
        !yw++
     ELSE IF (WRD == 'IOFO') THEN
        !---- process IOFOrmat command
        !     IOFOrmat [EXTEnded  ]
        !              [NOEXtended]
        qoldfmt=.false.
        qoldfmt=indxa(comlyn,comlen,'NOEX') > 0
        if (qoldfmt) then
           qextfmt=.false.
           idleng=4
           if (prnlev >= 2) write(outu,'(a)') &
                ' MISCOM> Expanded I/O format is NOT used.'
        endif
        qnewfmt=.false.
        qnewfmt=indxa(comlyn,comlen,'EXTE') > 0
        if (qnewfmt) then
           qextfmt=.true.
           idleng=8
           if (prnlev >= 2) write(outu,'(a)') &
                ' MISCOM> Expanded I/O format is used.'
        endif
        !yw--
     ELSE IF (WRD == 'GET ') THEN
        !---- GET-COMMAND-PARAMETER
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK)
        IF (IUNIT  >  0) THEN
           CALL RDCMND(COMLYN,MXCMS2,COMLEN,IUNIT,EOF, &
#if KEY_MULTICOM==0 /*  VO broadcast value of get to all local nodes */
                .FALSE.,&
#else
                .TRUE., &
#endif
                .FALSE.,' ')
#if KEY_PARALLEL==1 || KEY_ENSEMBLE==1
           !     Do not use parallel code in RDCMND since we should not compress
           !     command line parameters.
#if KEY_MULTICOM==0 /*  VO skip broadcast; it must be done within RDCMND for some communicators */
           CALL PSND4(COMLEN,1)
           CALL PSND4(EOF,1)
#endif
           IF(COMLEN > 0) CALL PSNDC(COMLYN(1:COMLEN),1)
#endif 
           IF(EOF) THEN
              COMLYN='END-OF-FILE'
              COMLEN=11
              IF(WRNLEV >= 2) WRITE(OUTU,3320) IUNIT
3320          FORMAT(' MISCOM WARNING:', &
                   ' EOF when getting parameter from unit',I4)
           ENDIF
           CALL TRIMA(COMLYN,COMLEN)
           IPAR = PARINS(TOKTMP,LENTOK,COMLYN,COMLEN)
           IF(IPAR < 0) THEN
              CALL WRNDIE(0,'<MISCOM>','Failed to install parameter.')
           ENDIF
        ELSE
           CALL WRNDIE(0,'<MISCOM>','Illegal or missing UNIT for GET')
        ENDIF
        COMLEN=0
        !
        !
     ELSE IF (WRD == 'GOTO') THEN
#       if KEY_REPDSTR==1
        if (nrepdstr.gt.1) then
          if (repd_inside_if_block.gt.0) then
            ! GOTO inside 'if' block not supported yet
            ! To do this properly we would have to keep track of the
            ! positions of all 'if'/'then'/'else's, gotos, and labels.
            ! repd_if_block_enabled =.false.
            repd_inside_if_block = 0
            if (prnlev.gt.1) then
              write(outu,'(a)') "Warning: 'GOTO' inside 'IF' block not yet supported for repd."
              write(outu,'(a)') "Warning: Asynchronous 'IF' disabled."
              write(outu,'(a)') "Warning: When reading from main input file, input must be"
              write(outu,'(a)') "Warning: written so processes stay in sync."
            endif
            CALL WRNDIE(-3,'<MISCOM>','GOTO not yet supported in IF block') ! added this error and commented out repd_if_block_enabled =.false. so error can be bypassed if necessary
          endif
        endif
#       endif /* KEY_REPDSTR */
        !---- PROCESS-GOTO-COMMAND
        CALL NEXTWD(COMLYN,COMLEN,WRD2,WDMAX,WD2LEN)
#if KEY_MULTICOM==0 /*  VO there could be several parser communicators (even with iolev < 0) */
        IF(IOLEV > 0) THEN
#else
        if (ME_PARSER.eq.0) then
#endif
           REWIND ISTRM
           OK=.FALSE.
           EOF=.FALSE.
111        CONTINUE
           READ(ISTRM,'(A)',ERR=945) COMLYN(1:80)
           COMLEN=80
           CALL CNVTUC(COMLYN,COMLEN)
           WRD=NEXTA4(COMLYN,COMLEN)
           IF (WRD == 'LABE') THEN
              CALL NEXTWD(COMLYN,COMLEN,WRD3,WDMAX,WD3LEN)
              OK=EQST(WRD2,WD2LEN,WRD3,WD3LEN)
           ENDIF
           IF (.NOT.(OK.OR.EOF)) GOTO 111
           !
           IF (.NOT.EOF) GOTO 845
945        CALL WRNDIE(-2,'<MISCOM>','Unable to find GOTO label')
845        CONTINUE
        ENDIF
        !
#if KEY_MULTICOM==0 /* (mcom)  VO : stringm conditionals already processed above */
     ELSE IF (WRD == 'IF  ') THEN
        !---- PROCESS-IF-COMMAND
        OK=.FALSE.
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK)
        !     Do text replacement for a single character first operand.
        !# <caves>-Aug-11-1993 (Leo Caves) if string (any size) is a symbol,
        !                                  then substitute
        ! don't double substitute
        IF((FFOUR /=  'IF @').AND.(FFOUR.NE.'IF ?')) THEN
           IPAR=PARNUM(TOKTMP,LENTOK)
           IF(IPAR > 0) THEN
              IF(VALLEN(IPAR) > 0) &
                   CALL COPYST(TOKTMP,MXTLEN,LENTOK,VALNAM(IPAR),VALLEN(IPAR))
           ENDIF
        ENDIF
        !
        WRD=NEXTA4(COMLYN,COMLEN)
        CALL NEXTWD(COMLYN,COMLEN,VALTMP,MXVLEN,LENVAL)
        !
        IF(PRNLEV >= 2) WRITE(OUTU,92) TOKTMP(1:LENTOK), &
             VALTMP(1:LENVAL)
92      FORMAT(' Comparing "',A,'" and "',A,'".')
        !
        IF (WRD == 'EQ  ' .OR. WRD.EQ.'.EQ.') THEN
           OK=EQST(TOKTMP,LENTOK,VALTMP,LENVAL)
        ELSE IF (WRD == 'NE  ' .OR. WRD.EQ.'.NE.') THEN
           OK=.NOT.EQST(TOKTMP,LENTOK,VALTMP,LENVAL)
        ELSE
           VALUE=DECODF(TOKTMP,LENTOK)
           TEMP=DECODF(VALTMP,LENVAL)
           IF (WRD == 'GT  ' .OR. WRD.EQ.'.GT.') THEN
              OK=VALUE > TEMP
           ELSE IF (WRD == 'GE  ' .OR. WRD.EQ.'.GE.') THEN
              OK=VALUE >= TEMP
           ELSE IF (WRD == 'LT  ' .OR. WRD.EQ.'.LT.') THEN
              OK=VALUE < TEMP
           ELSE IF (WRD == 'LE  ' .OR. WRD.EQ.'.LE.') THEN
              OK=VALUE <= TEMP
           ELSE IF (WRD == 'AE  ' .OR. WRD.EQ.'.AE.') THEN
              OK=ABS(VALUE-TEMP) < PT0001
           ELSE
! Better to trap this, an error could make affect a simulation
! without being noticed. L Nilsson
            call wrndie(-2,'<MISCOM>','Illegal IF test')
           ENDIF
        ENDIF
        !
#       if KEY_REPDSTR==1
        if (repd_if_block_enabled.and.nrepdstr.gt.1) then ! Asynchronous IF
          ! Asynchronous 'if' block support is active.
          ! When split into replicas, different replicas may want to execute
          ! different commands. However, the parser currently requires that
          ! replicas stay in sync. If an 'if then' block is encountered, each
          ! replica needs to keep track of whether the test evaluated as OK
          ! or not. If so, commands can be executed, otherwise commands are
          ! read but skipped.
          done = .false.
          call trima(comlyn, comlen)
          if (comlyn(1:4) == 'THEN') then
            ! This 'if' statement contains a 'THEN'. Check if this is
            ! a 1 line 'if <X> then <command>' statement or if this the
            ! start of an 'if' block.
            wrd = nexta4(comlyn, comlen) ! parse out the 'THEN'
            call trima(comlyn, comlen)
            !write(outu,'(a,a,a)') 'DBG: wrd after then= "', trim(comlyn), '"'
            if (comlyn(1:4) == '    ') then
              ! No command after 'THEN'; start of 'if <X> then' block
              repd_inside_if_block = repd_inside_if_block + 1
              ! Sanity check
              if (repd_inside_if_block .gt. max_repd_if_block) &
                call wrndie(-5,'<MISCOM>','Maximum number of nested IF blocks with REPD reached.')
              if (OK) then
                repd_if_block_stat(repd_inside_if_block) = 1
              else
                repd_if_block_stat(repd_inside_if_block) = -1
              endif
              if (prnlev.ge.6) write(outu,'(a,i6,l2)') ' Inside REPD IF block ', repd_inside_if_block, OK
              done = .true.
            endif
          endif
          if (.not.OK) then
            comlen = 0
            if (prnlev.ge.2) write(outu,'(a)') ' IF test evaluated as false.'
          else
            if (prnlev.ge.2) write(outu,'(a)') ' IF test evaluated as true.'
          endif
        else ! Asynchronous IF 
          if (qrepdstr.and.(.not.qrdqtt)) then
            ! Asynchronous 'if' block support is not active and reading from a
            ! single script. All replicas must stay in sync.
            allocate( if_results( nrepdstr ) )
            if (qrepmaster) &
              call mpi_allgather(ok, 1, MPI_LOGICAL, if_results, 1, MPI_LOGICAL, comm_rep_master, ierr)
            call mpi_bcast(if_results, nrepdstr, MPI_LOGICAL, 0, comm_rpg, ierr)
            do i = 2, nrepdstr
              if (if_results(i) .neqv. if_results(1)) &
                call wrndie(-5,'<MISCOM>','IF result differs between replicas and asynchronous IF disabled.')
            enddo
            deallocate( if_results )
          endif
#       endif /* KEY_REPDSTR */
        IF (OK) THEN
           IF(PRNLEV >= 2) WRITE(OUTU,94)
94         FORMAT(' IF test evaluated as true.  Performing command')
           DONE=.FALSE.
           CALL TRIMA(COMLYN,COMLEN)
           IF(COMLYN(1:4) == 'THEN') WRD=NEXTA4(COMLYN,COMLEN)
        ENDIF
        IF (.NOT.OK) THEN
           WRD=NEXTA4(COMLYN,COMLEN)
           JUNK=NEXTA4(COMLYN,COMLEN)
           IF((WRD == 'THEN').AND.(JUNK.eq.'    '))THEN
              IF(PRNLEV >= 2) WRITE(OUTU,'(A)')  &
                   ' IF test evaluated as false.  Skip to ELSE or ENDIF'
#             if KEY_REPDSTR==1
              qreadstream = .false.
              if (qrepdstr) then
                if (qrdqtt .or. istrm.ne.5) then
                  ! REPD and reading a stream - all replica masters do it
                  if (mynod.eq.0) qreadstream = .true.
                else if (mynodg.eq.0) then
                  ! REPD and not reading a stream - only global master does it.
                  qreadstream = .true.
                endif
              else
                ! Not repd - use iolev
                if (iolev > 0) qreadstream = .true.
              endif
              if ( qreadstream ) then
#             else /* KEY_REPDSTR */
              IF(IOLEV > 0) THEN
#             endif /* KEY_REPDSTR */
                 ILEVEL=1
                 OK=.FALSE.
112              CONTINUE
                 READ(ISTRM,'(A)',END=946,ERR=946) COMLYN(1:80)
                 COMLEN=80
                 CALL CNVTUC(COMLYN,COMLEN)
                 WRD=NEXTA4(COMLYN,COMLEN)
                 IF (WRD == 'IF  '.and. INDEX(COMLYN,'THEN') /= 0)THEN
                    ILEVEL=ILEVEL+1
                 ENDIF
                 IF (WRD == 'ENDI')THEN
                    ILEVEL=ILEVEL-1
                 ENDIF
                 OK=(ILEVEL == 0)
                 IF ((WRD == 'ELSE').AND.(ILEVEL <= 1)) OK=.TRUE.
                 IF (.NOT.OK) GOTO 112
946              IF (.NOT.OK) THEN    
                    CALL WRNDIE(-2,'<MISCOM>', &
                         'Unable to find ELSE or ENDIF')
                 ENDIF
              ENDIF
           ELSE
              IF(PRNLEV >= 2) WRITE(OUTU,'(A)')  &
                   ' IF test evaluated as false.  Skipping command'
              COMLEN=0
           ENDIF
        ENDIF
#       if KEY_REPDSTR==1
        endif ! Asynchronous IF 
#       endif
        !
#endif /*(mcom) */
     ELSE IF (WRD == 'INCR') THEN
        !---- PROCESS-INCREMENT-COMMAND
        parOp = INCR
        !
#if KEY_MULTICOM==0 /* (mcom)  VO stringm conditionals already processed above */
     ELSE IF (WRD == 'ENDI') THEN
        COMLEN=0
        !
#endif /*(mcom) */
     ELSE IF (WRD == 'LABE') THEN
        COMLEN=0
!        FLUSH (OUTU)  ! for MMTSB
        !
#if KEY_MULTICOM==0 /* (mcom)  VO stringm conditionals already processed */
     ELSE IF (WRD == 'ELSE') THEN
        IF(PRNLEV >= 2)THEN
           WRITE(OUTU,'(2A)')  &
                ' Skip commands until next ENDIF'
        ENDIF
        IF(IOLEV > 0) THEN
           OK=.FALSE.
           ILEVEL=1
113        CONTINUE
           READ(ISTRM,'(A)',END=947,ERR=947) COMLYN(1:80)
           COMLEN=80
           CALL CNVTUC(COMLYN,COMLEN)
           WRD=NEXTA4(COMLYN,COMLEN)
           IF (WRD == 'IF  '.and. INDEX(COMLYN,'THEN') /= 0)THEN
              ILEVEL=ILEVEL+1
           ENDIF
           IF (WRD == 'ENDI')THEN
              ILEVEL=ILEVEL-1
           ENDIF
           OK=(ILEVEL == 0)
           IF (.NOT.OK) GOTO 113
947        IF (.NOT.OK) THEN
              CALL WRNDIE(-2,'<MISCOM>','Unable to find ENDIF')
           ENDIF
        ENDIF
        !
#endif /*(mcom) */
     ELSE IF (WRD == 'CALC') THEN
        ! Dec-01-1994 (Benoit Roux ) arithmetic parameter manipulation
        ! get the token
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK)
        CALL TRIMA(COMLYN,COMLEN)
        ! grab the parameter which is to be modified
        IPAR = PARNUM(TOKTMP,LENTOK)
        IF(IPAR <= 0) THEN
           ! install
           !         CALL WrnDie(0,'<MISCOM>',
           !    ,              'Non-existent parameter. '//swdtch(1:swdlen))
           IF(NUMPAR+1 > MAXPAR)THEN
              CALL WrnDie(0,'<MISCOM>','Parameter table full. Ignored.')
           ENDIF
           NUMPAR = NUMPAR + 1
           IPAR = NUMPAR
           TOKNAM(IPAR) = TOKTMP(1:LENTOK)
           TOKLEN(IPAR) = LENTOK
        ENDIF
        !
        ! dispense with an assignment operator here.
        IF (COMLYN(1:2) == '= ') THEN
           COMLYN(1:2) = '  '
           CALL TRIMB(COMLYN,COMLEN)
        ENDIF
        ! evaluate the arithmetic expression
        call calculator(VALUE,COMLYN)
        ! Set the new value to the parameter
        CALL ENCODF(value,VALNAM(IPAR),MXVLEN,VALLEN(IPAR))
        IF(PRNLEV >= 2) CALL PARWRI(OUTU,IPAR,1)
        COMLEN = 0
        !
     ELSE IF (WRD == 'LET ') THEN
        !# <caves>-Aug-13-1993 (Leo Caves) Limited parameter manipulation
        IPAR =  PARMAN(COMLYN,COMLEN)
        IF(IPAR < 0) THEN
           CALL WrnDie(0,'<MISCOM>','Error in parameter modification') 
           COMLEN = 0
        ENDIF
        !
     ELSE IF (WRD == 'LOWE') THEN
        LOWER=.TRUE.
        !
     ELSE IF (WRD == 'MULT') THEN
        !---- PROCESS-MULTIPLY-COMMAND
        parOp = MULT
        !
     ELSE IF (WRD.EQ.'MTP') THEN
        !  atomic multipole moments MTP module
        CALL MTP
        !

     ELSE IF (WRD.EQ.'DCM') THEN              
        !  atomic multipole moments DCM module 
        CALL DCM 

     ELSE IF (WRD.EQ.'MTPL') THEN
        !  atomic multipole moments MTPL module--uses local axis systems for
        !  arbitrarily large molecules
#if KEY_MTPL==1
        CALL MTPL
#else
        CALL WrnDie(0,'<MISCOM>','MTPL module not compiled.')
#endif /* mtpl */
        !
     ELSE IF (WRD == 'NOBO') THEN
        BOMLEV=-1
        CALL set_param('BOMLEV',BOMLEV)
     ELSE IF (WRD == 'OPEN') THEN
        CALL OPNLGU(COMLYN,COMLEN,IDUMMY)
     ELSE IF (WRD == 'OUTU') THEN
        OUTU  =NEXTI(COMLYN,COMLEN)
        CALL set_param('OUTU',OUTU)
#if KEY_REPDSTR==1
        QWRQTT=.TRUE.                                  
        CALL DREPSETIO(IOLEV,PRNLEV,WRNLEV)                   
#endif
     ELSE IF (WRD == 'PRLE' .OR. WRD.EQ.'PRNL') THEN
        I=NEXTI(COMLYN,COMLEN)
#if KEY_PARALLEL==1
        J=GTRMI(COMLYN,COMLEN,'NODE',-100)
        IF(J == -100) THEN
           PRNLEV=I
        ELSE IF(J < 0 .OR. J > NUMNOD) THEN
           CALL WRNDIE(0,'<MISCOM>','Bad node number - ignored')
        ELSE
           IF(MYNOD == J) PRNLEV=I
        ENDIF
        plnod0=prnlev
        IF(MYNOD /= 0) plnod0=0
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
        IF (MYNOD /= 0) THEN
           prnlev = -100
           wrnlev = -100
        ENDIF

#endif
! BIOVIA Code End
        call gbor(plnod0,1)

#else /**/
        PRNLEV=I
#endif 
        CALL set_param('PRNLEV',PRNLEV)
#if KEY_NOMISC==0
     ELSE IF (WRD == 'QUIC' .OR. WRD.EQ.'Q') THEN
        CALL QUICKA()
     ELSE IF (WRD == 'AIDX') THEN
        CALL AIDX()
#endif 
     ELSE IF (WRD == 'RAND') THEN
        CALL RANDSPEC()
        !
     ELSE IF (WRD == 'IRAN') THEN
        CALL IRANDOM()
        !
     ELSE IF (WRD == 'RETU') THEN
        CALL PPSTRM(OK)
        !.....mf050629
     ELSE IF (WRD == 'REWF') THEN
        reallow=.false.
     ELSE IF (WRD == 'REWT') THEN
        reallow=.true.
        !
     ELSE IF (WRD == 'REWI') THEN
        !
        ! Rewind file (useful for repetitive stream changing)
        !
        IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',-1)
        IF (IUNIT > 0) THEN
           IF(IOLEV > 0) REWIND IUNIT
           IF(PRNLEV >= 2) WRITE(OUTU,1220) IUNIT
1220       FORMAT(20X,'REWINDING UNIT ',I5)
        ELSE
           CALL WRNDIE(0,'<MISCOM>','Logical unit number incorrect.')
        ENDIF
        !
     ELSE IF (WRD == 'SET ') THEN
        !---- SET-COMMAND-PARAMETER
        !# <caves>-Aug-10-1993 (Leo Caves) New parameter table
        ! get the token
        CALL NEXTWD(COMLYN,COMLEN,TOKTMP,MXTLEN,LENTOK)
        CALL TRIMA(COMLYN,COMLEN)
        !
        ! dispense with an assignment operator here.
        IF (COMLYN(1:2) == '= ') THEN
           COMLYN(1:2) = '  '
           CALL TRIMB(COMLYN,COMLEN)
        ENDIF
        ! install token
        IPAR = PARINS(TOKTMP,LENTOK,COMLYN,COMLEN)
        IF(IPAR < 0) THEN
           CALL WrnDie(0,'<MISCOM>','Failed to install parameter.')
        ENDIF
        COMLEN = 0
        !
        !
     ELSE IF (WRD == 'SHOW') THEN
        !---- process-show-command
        WRD = NEXTA4(COMLYN,COMLEN)
        ! this command not relevant if I/O level not appropriate
        IF((WRD == '   ').OR.(WRD.EQ.'BUIL'))THEN
           IF(PRNLEV >= 2) THEN
              WRITE(OUTU,143)
              call write_real_params(outu)
              call write_int_params(outu)
              call write_str_params(outu)
           ENDIF
        ELSE IF (WRD(1:3) == 'PAR') THEN
           ! dump parameter table
           IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
           IF(PRNLEV >= 2) THEN
              WRITE(OUTU,'(A,2(I5,A))') &
                   ' MISCOM: No. of parameters= ',NUMPAR,'  (max.=',MAXPAR,')'
              ! verbose gets you string limits
              IF(INDXA(COMLYN,COMLEN,'VERB') > 0)  &
                   WRITE(OUTU,'(2(A,I5))')  &
                   ' MISCOM: Maximum string sizes: tokens= ',MXTLEN, &
                   ' values= ',MXVLEN
              IF(NUMPAR >= 1) THEN
                 ! here's the table
                 DO IPAR = 1, NUMPAR
                    CALL PARWRI(IUNIT,IPAR,-1)
                 ENDDO
              ENDIF
           ENDIF
        ELSE
           ! if the show command keyword was unrecognized,
           ! perhaps this is the CORREL SHOW command.  Reset the command line.
           CALL JOINWD(COMLYN,MXCMS2,COMLEN,WRD,4)
           WRD2='SHOW '
           CALL JOINWD(COMLYN,MXCMS2,COMLEN,WRD2,5)
           LUSED=.FALSE.
           DONE=.TRUE.
        ENDIF

143     FORMAT(' MISCOM: List of command substitution parameter values')

     ELSE IF (WRD == 'STRE') THEN
        !
#if KEY_REPDSTR==1 
        IF(QREPDSTR) QRDQTT=.TRUE.
#endif 
        CALL PUSTRM(COMLYN,MXCMS2,COMLEN)
        !
        ! Process automatic set options: 
        !  STREam filename   parm1 parm2 parm3 ...
        !     this becomes:   IN1   IN2   IN3  ... 
        IF(COMLEN > 0) THEN
           I=0
           DO WHILE(COMLEN > 0 .AND. I < 9)
              I=I+1
              WRITE(TOKTMP,166) I
166           FORMAT('IN',I1)
              LENTOK=3
              CALL TRIMA(COMLYN,COMLEN)
              CALL NEXTWD(COMLYN,COMLEN,WRD2,WDMAX,WD2LEN)
              ! install token
              IPAR = PARINS(TOKTMP,LENTOK,WRD2,WD2LEN)
              IF(IPAR < 0) THEN
                 CALL WrnDie(0,'<MISCOM>','Failed to install parameter.')
              ENDIF
           ENDDO
        ENDIF
        !
     ELSE IF (WRD == 'SKIP') THEN
        CALL SKIPE(COMLYN,COMLEN)
        !
     ELSE IF (WRD == 'TIME') THEN
        !---- PROCESS-TIME-COMMAND
        IF (INDXA(COMLYN,COMLEN,'NOW')  >  0) THEN
           CALL TIMRE
        ELSE IF (INDXA(COMLYN,COMLEN,'DIFF')  >  0) THEN
           CALL TIMRE
           CALL TIMRB
        ELSE
           TIMER=NEXTI(COMLYN,COMLEN)
           CALL set_param('TIMER',TIMER)
        ENDIF
        !
#if KEY_LOOKUP==1
     ELSE IF (WRD == 'NBSO'.OR.WRD.EQ.'LOOK')THEN
        CALL WWSETUP(COMLYN,COMLEN) 
#endif 
        !
     ELSE IF (WRD == 'TITL') THEN
        !---- PROCESS-TITLE-COMMAND
        IF (INDXA(COMLYN,COMLEN,'COPY') > 0) THEN
           NTITLA=NTITLB
           DO I=1,NTITLB
              TITLEA(I)=TITLEB(I)
           ENDDO
        ENDIF
        CALL RDTITL(TITLEA,NTITLA,ISTRM,0)
        !
#if KEY_BLOCK==1
     ELSE IF (WRD == 'LDTI') THEN
        !---- PROCESS-TITLE-COMMAND FOR Lambda Dynamics
        CALL RDTITL(TITLEL,NTITLL,ISTRM,0)
#endif 
        !
     ELSE IF (WRD == 'TRIM') THEN
        !---- PROCESS-TRIM-COMMAND
        CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
        IPAR=PARNUM(SWDTCH,SWDLEN)
        IF (IPAR > 0) THEN
           I=VALLEN(IPAR)
           DO WHILE(I > 1 .AND. VALNAM(IPAR)(I:I) == ' ')
              I=I-1
           ENDDO
           J=1
           DO WHILE(J < I .AND. VALNAM(IPAR)(J:J) == ' ')
              J=J+1
           ENDDO
           I=GTRMI(COMLYN,COMLEN,'TO',I)
           J=GTRMI(COMLYN,COMLEN,'FROM',J)
           IF(I > MXVLEN)I=MXVLEN
           IF(I > VALLEN(IPAR)) &
                CALL FILSPC(VALNAM(IPAR),I,VALLEN(IPAR))
           CALL COPSUB(VALNAM(IPAR),MXVLEN, &
                VALLEN(IPAR),VALNAM(IPAR),J,I)
           IF(PRNLEV >= 2) CALL PARWRI(OUTU,IPAR,1) 
        ELSE
           CALL WrnDie(0,'<MISCOM>', &
                ' Parameter not found. Nothing to trim.')
        ENDIF
        !
     ELSE IF (WRD == 'UPPE') THEN
        LOWER=.FALSE.
     ELSE IF (WRD == 'USE ') THEN
        CALL SETUPFF(COMLYN,COMLEN)
        !
     ELSE IF (WRD == 'USER') THEN
        CALL USERSB
     ELSE IF (WRD == 'WRNL') THEN
        I=NEXTI(COMLYN,COMLEN)
#if KEY_PARALLEL==1
        J=GTRMI(COMLYN,COMLEN,'NODE',-100)
        IF(J == -100) THEN
           WRNLEV=I
#if KEY_MULTICOM==0 /*  VO stringm v */
        ELSE IF(J < 0 .OR. J >= NUMNOD) THEN
#else
        ELSE IF(J < 0 .OR. J >= SIZE_PARSER) THEN
#endif
           CALL WRNDIE(0,'<MISCOM>','Bad node number - ignored')
        ELSE
#if KEY_MULTICOM==0
           IF(MYNOD == J) WRNLEV=I
#else
           IF(ME_PARSER == J) WRNLEV=I
#endif /* VO stringm v */
        ENDIF
! BIOVIA Code Start : Fix for Windows
#if KEY_WIN32==1 || KEY_WIN64==1
        IF(MYNOD.NE.0) WRNLEV=-1
#endif
! BIOVIA Code End
#else /**/
        WRNLEV=I
#endif 
        CALL set_param('WRNLEV',WRNLEV)
     ELSE IF (WRD == 'IOLE') THEN
        I=NEXTI(COMLYN,COMLEN)
#if KEY_PARALLEL==1
        J=GTRMI(COMLYN,COMLEN,'NODE',-100)
        IF(J == -100) THEN
           IOLEV=I
#if KEY_MULTICOM==0 /*  VO stringm v */
        ELSE IF(J < 0 .OR. J >= NUMNOD) THEN
#else
        ELSE IF(J < 0 .OR. J >= SIZE_PARSER) THEN
#endif
           CALL WRNDIE(0,'<MISCOM>','Bad node number - ignored')
        ELSE
#if KEY_MULTICOM==0
           IF(MYNOD == J) IOLEV=I
#else
           IF(ME_PARSER == J) IOLEV=I
#endif /* VO stringm v */
        ENDIF
#else /**/
        IOLEV=I
#endif 
        CALL set_param('IOLEV',IOLEV)
     ELSE IF (WRD == 'LONG') THEN
        QLONGL=.TRUE.
     ELSE IF (WRD == 'SHOR') THEN
        QLONGL=.FALSE.
        !
     ELSE IF (WRD == 'SYST') THEN
        CALL CSYSTEM(COMLYN,COMLEN)

     ELSE IF (WRD == 'FRCD') THEN
        iunit = GTRMI(COMLYN,COMLEN,'UNIT',outu)
        call DumpEnFrc(iunit)

     ELSE IF (WRD == 'STOP') THEN
        CALL STOPCH('NORMAL STOP')
        !
     ELSE
        CALL JOINWD(COMLYN,MXCMS2,COMLEN,WRD,4)
        LUSED=.FALSE.
        DONE=.TRUE.
     ENDIF
     !
     !# <caves>-Aug-13-1993 (Leo Caves)
     ! support existing parameter operations. prefer that any further operations
     ! be placed in a separate evaluation function. 
     ! eg LET command: SUBROUTINE PARMAN
     IF (parOp > 0) THEN
        INCREM=GTRMF(COMLYN,COMLEN,'BY',ONE)
        CALL NEXTWD(COMLYN,COMLEN,SWDTCH,SWDMAX,SWDLEN)
        ! lookup
        IPAR = PARNUM(SWDTCH,SWDLEN)
        IF (IPAR >= 0) THEN
           VALUE=DECODF(VALNAM(IPAR),VALLEN(IPAR))
           IF      (parOp == INCR) THEN
              VALUE = VALUE + INCREM
           ELSE IF (parOp == DECR) THEN
              VALUE = VALUE - INCREM
           ELSE IF (parOp == MULT) THEN
              VALUE = VALUE * INCREM
           ELSE IF (parOp == DIVI) THEN
              IF (INCREM /= ZERO) THEN
                 VALUE = VALUE / INCREM
              ELSE
                 CALL WRNDIE(0,'<MISCOM>','Divide by zero. IGNORED.')
              ENDIF
           ELSE IF (parOp == EXPO) THEN
              VALUE = EXP(VALUE)
           ENDIF
           !
           CALL ENCODF(VALUE,VALNAM(IPAR),MXVLEN,VALLEN(IPAR))
           IF (PRNLEV >= 2) CALL PARWRI(OUTU,IPAR,1)
        ELSE
           CALL WrnDie(0,'<MISCOM>',  &
                'Parameter not found. Nothing to modify.')
        ENDIF ! IPAR
     ENDIF ! parOp
     !
  enddo loop101
  !
  if (lused) call xtrane(comlyn,comlen,'miscom')
  return
end subroutine miscom

!---------------------------------------------------------------------
!          SETUPFF
!---------------------------------------------------------------------
subroutine setupff(comlyn,comlen)
  !----------------------------------------------------------------------
  use chm_kinds
  use dimens_fcm
#if KEY_DOMDEC==1
  use domdec_common, only: q_domdec
#endif
  use consta
  use energym
  use ffieldm
  use inbnd
  use machdep
#if KEY_PERT==1
  use pert, only: qpert
#endif
  use psf
  use param_store, only: set_param
#if KEY_CFF==1
  use cff_fcm  
#endif
#if KEY_MMFF==1
  use mmffm  
#endif
  use stream
  use string

  implicit none

  character(len=*) comlyn
  integer   comlen
  integer   i

  if(indxa(comlyn,comlen,'AMBER') > 0) then
     ffield=amberffn
     ccelec=ccelec_amber
     call set_param('CCELEC',ccelec)
     if(prnlev >= 2) write(outu,'(" AMBER Force Field will be used")')
  endif
     
  if(indxa(comlyn,comlen,'CHARMM') > 0) then
     ffield=charmm
     ccelec=ccelec_charmm
     call set_param('CCELEC',ccelec)
     if(prnlev >= 2) write(outu,'(" CHARMM Force Field will be used")')
  endif
     
#if KEY_MMFF==1 || KEY_CFF==1 /*mmff_cff*/
#if KEY_CFF==1
  if(indxa(comlyn,comlen,'CFF') > 0) then
     ffield=cff
     ! Allocate data structures for CFF force fields. cb3
     if(.not. allocated(itflg)) call allocate_cff
  endif
#endif 
#if KEY_MMFF==1
  if(indxa(comlyn,comlen,'MMFF') > 0) then

! BIOVIA Code Start : Bug fix
#if KEY_LICENSE==1
     CALL checkout_license_mmff()
#endif
! BIOVIA Code End

#if KEY_DOMDEC==1
     if (q_domdec) then
        call wrndie(-2,'<SETUPFF>','MMFF is not compatible with DOMDEC')
     end if
#endif /* domdec */

#if KEY_PERT==1
     if (qpert) then
        call wrndie(-2,'<SETUPFF>','MMFF is not compatible with PERT')
     end if     
#endif /* pert */
     
     e14fac=0.75
     v14fac=1.
     ffield=mmff
     ! Allocate data structures for CFF force fields. cb3
     if(.not. allocated(bondtype)) then
        call allocate_mmff
        call mmff_init
     endif
     if(indxa(comlyn,comlen,'TYPE') > 0) then
        call mmff_setup
        i=indxa(comlyn,comlen,'ATOM')
     endif
  endif
#endif 

  if(indxa(comlyn,comlen,'CHARMM') > 0) then
     ffield=charmm
     if(prnlev >= 2) write(outu,'(a)') &
          ' CHARMM Force Field will be used'
  endif
#else /* (mmff_cff)*/
  if(indxa(comlyn,comlen,'CFF') > 0) then
     call wrndie(-2,'<SETUPFF>','CFF code not compiled')
  endif
  if(indxa(comlyn,comlen,'MMFF') > 0) then
     call wrndie(-2,'<SETUPFF>','MMFF code not compiled')
  endif
  if(indxa(comlyn,comlen,'CHARMM') > 0) then
     if(prnlev >= 2) write(outu,'(a)') &
          ' CHARMM Force Field will be used'
  endif
#endif /* (mmff_cff)*/
  i=indxa(comlyn,comlen,'FORCE')
  i=indxa(comlyn,comlen,'FIELD')
  return
end subroutine setupff
