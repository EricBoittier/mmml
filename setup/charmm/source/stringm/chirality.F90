! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!



module chirality
#if (KEY_STRINGM==1) /*  automatically protect all code */
 use chm_kinds
 implicit none
 private
!
 !
 ! in this module :
 ! chirality is determined using a dihedral angle
 ! rule specified in a `rules` table (below) or in an
 ! externally provided file (for added flexibility:
 ! it is clear that different force fields and/or different
 ! molecules will have specific corresponding rules);
 ! the rules table lists the residue name,
 ! the corresponding dihedral, the threshold in degrees for
 ! passing the test, the direction in which
 ! the threshold can be exceeded, the atom name whose position
 ! can be reflected around a symmetry plane (normally, a proton,
 ! because nothing is attached to it except for the parent heavy atom)
 ! and the atom which lies in the symmetry plane whose distance
 ! vector to the previous atom is normal to the symmetry plane
 ! (e.g. the parent atom of the hydrogen)
!
!
  type dihedral_rule
   character(len=8) :: resname='', aname(6)=''
   real(chm_real) :: threshold
   integer :: sgn
  end type dihedral_rule
!
  interface operator(.eq.)
    module procedure compare_rules
  end interface operator(.eq.)
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
  type(dihedral_rule), pointer :: dihedral_rules(:), dihedral_rules_bkp(:)
  integer, parameter :: rules_expand = 10 ! reallocation size increment
  integer :: num_rules=0
!
  logical :: chirality_initialized=.false., qprint=.true., qflip=.false.
!
  character(len=80), pointer :: dihedral_rules_data(:)
  character(len=80), target :: dihedral_rules_charmm22(25)=&

#if (KEY_PATHSCALE==0)
& [character(len=80) :: &
& 'ALA N C CB HA 0. -1 HA CA',& ! the dihedral specified should be less than 0
& 'ARG N C CB HA 0. -1 HA CA',&
& 'ASN N C CB HA 0. -1 HA CA',&
& 'ASP N C CB HA 0. -1 HA CA',&
& 'CYS N C CB HA 0. -1 HA CA',&
& 'GLN N C CB HA 0. -1 HA CA',&
& 'GLU N C CB HA 0. -1 HA CA',&
& 'HIS N C CB HA 0. -1 HA CA',&
& 'HSC N C CB HA 0. -1 HA CA',&
& 'HSD N C CB HA 0. -1 HA CA',&
& 'HSE N C CB HA 0. -1 HA CA',&
& 'HSP N C CB HA 0. -1 HA CA',&
& 'ILE N C CB HA 0. -1 HA CA',&
& 'ILE CA CG1 CG2 HB 0. -1 HB CB',&
& 'LEU N C CB HA 0. -1 HA CA',&
& 'LYS N C CB HA 0. -1 HA CA',&
& 'MET N C CB HA 0. -1 HA CA',&
& 'PHE N C CB HA 0. -1 HA CA',&
& 'PRO N C CB HA 0. -1 HA CA',&
& 'SER N C CB HA 0. -1 HA CA',&
& 'THR N C CB HA 0. -1 HA CA',&
& 'THR OG1 CA CG2 HB 0. 1 HB CB',& ! T has a chiral center at the CB atom
& 'TRP N C CB HA 0. -1 HA CA',&
& 'TYR N C CB HA 0. -1 HA CA',&
& 'VAL N C CB HA 0. -1 HA CA'&
&]
#else
&(/'ALA N C CB HA 0. -1 HA CA',&
& 'ARG N C CB HA 0. -1 HA CA',&
& 'ASN N C CB HA 0. -1 HA CA',&
& 'ASP N C CB HA 0. -1 HA CA',&
& 'CYS N C CB HA 0. -1 HA CA',&
& 'GLN N C CB HA 0. -1 HA CA',&
& 'GLU N C CB HA 0. -1 HA CA',&
& 'HIS N C CB HA 0. -1 HA CA',&
& 'HSC N C CB HA 0. -1 HA CA',&
& 'HSD N C CB HA 0. -1 HA CA',&
& 'HSE N C CB HA 0. -1 HA CA',&
& 'HSP N C CB HA 0. -1 HA CA',&
& 'ILE N C CB HA 0. -1 HA CA',&
& 'ILE CA CG1 CG2 HB 0. -1 HB CB',&
& 'LEU N C CB HA 0. -1 HA CA',&
& 'LYS N C CB HA 0. -1 HA CA',&
& 'MET N C CB HA 0. -1 HA CA',&
& 'PHE N C CB HA 0. -1 HA CA',&
& 'PRO N C CB HA 0. -1 HA CA',&
& 'SER N C CB HA 0. -1 HA CA',&
& 'THR N C CB HA 0. -1 HA CA',&
& 'THR OG1 CA CG2 HB 0. 1 HB CB',&
& 'TRP N C CB HA 0. -1 HA CA',&
& 'TYR N C CB HA 0. -1 HA CA',&
& 'VAL N C CB HA 0. -1 HA CA'/)
#endif
!
  character(len=80), target :: pseudo_dihedral_rules_charmm22(42)=& ! these rules enforce pseudo-chirality, which concerns equivalent atoms
#if (KEY_PATHSCALE==0)
& [character(len=80) :: &
& 'ANY CA HT3 HT2 HT1 0. 2 HT1 HT2',& ! for N-terminal residues except PROP
& 'ALA CA HB3 HB2 HB1 0. 2 HB1 HB2',& ! proton ordering within methyl groups
& 'ASN CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ASP CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ARG CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ARG CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'ARG CG NE HD2 HD1 0. 2 HD1 HD2',&
& 'CYS CA SG HB2 HB1 0. 2 HB1 HB2',&
& 'GLN CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'GLN CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'GLU CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'GLU CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'GLY N  C  HA2 HA1 0. 2 HA1 HA2',&
& 'HIS CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'HSD CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'HSE CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ILE CB HG23 HG22 HG21 0. 2 HG21 HG22',&
& 'ILE CG1 HD3 HD2 HD1 0. 2 HD1 HD2',&
& 'ILE CB CD HG12 HG11 0. 2 HG11 HG12',&
& 'LEU CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'LEU CB CD2 CD1 HG 0. -1 HG CG',&
& 'LEU CG HD13 HD12 HD11 0. 2 HD11 HD12',&
& 'LEU CG HD23 HD22 HD21 0. 2 HD21 HD22',&
& 'LYS CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'LYS CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'LYS CG CE HD2 HD1 0. 2 HD1 HD2',&
& 'LYS CE HZ3 HZ2 HZ1 0. 2 HZ1 HZ2',&
& 'LYS CD NZ HE2 HE1 0. 2 HE1 HE2',&
& 'MET CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'MET CB SD HG2 HG1 0. 2 HG1 HG2',&
& 'MET SD HE3 HE2 HE1 0. 2 HE1 HE2',&
& 'PHE CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'PRO CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'PRO CB CD HG2 HG1 0. -2 HG1 HG2',&
& 'PRO N  CG HD2 HD1 0. 2 HD1 HD2',&
& 'SER CA OG HB2 HB1 0. 2 HB1 HB2',&
& 'THR CB HG23 HG22 HG21 0. 2 HG21 HG22',&
& 'TRP CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'TYR CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'VAL CA CG2 CG1 HB 0. -1 HB CB',&
& 'VAL CB HG13 HG12 HG11 0. 2 HG11 HG12',&
& 'VAL CB HG23 HG22 HG21 0. 2 HG21 HG22'&
&]
#else
&(/'ANY CA HT3 HT2 HT1 0. 2 HT1 HT2',&
& 'ALA CA HB3 HB2 HB1 0. 2 HB1 HB2',&
& 'ASN CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ASP CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ARG CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ARG CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'ARG CG NE HD2 HD1 0. 2 HD1 HD2',&
& 'CYS CA SG HB2 HB1 0. 2 HB1 HB2',&
& 'GLN CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'GLN CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'GLU CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'GLU CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'GLY N  C  HA2 HA1 0. 2 HA1 HA2',&
& 'HIS CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'HSD CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'HSE CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'ILE CB HG23 HG22 HG21 0. 2 HG21 HG22',&
& 'ILE CG1 HD3 HD2 HD1 0. 2 HD1 HD2',&
& 'ILE CB CD HG12 HG11 0. 2 HG11 HG12',&
& 'LEU CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'LEU CB CD2 CD1 HG 0. -1 HG CG',&
& 'LEU CG HD13 HD12 HD11 0. 2 HD11 HD12',&
& 'LEU CG HD23 HD22 HD21 0. 2 HD21 HD22',&
& 'LYS CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'LYS CB CD HG2 HG1 0. 2 HG1 HG2',&
& 'LYS CG CE HD2 HD1 0. 2 HD1 HD2',&
& 'LYS CE HZ3 HZ2 HZ1 0. 2 HZ1 HZ2',&
& 'LYS CD NZ HE2 HE1 0. 2 HE1 HE2',&
& 'MET CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'MET CB SD HG2 HG1 0. 2 HG1 HG2',&
& 'MET SD HE3 HE2 HE1 0. 2 HE1 HE2',&
& 'PHE CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'PRO CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'PRO CB CD HG2 HG1 0. -2 HG1 HG2',&
& 'PRO N  CG HD2 HD1 0. 2 HD1 HD2',&
& 'SER CA OG HB2 HB1 0. 2 HB1 HB2',&
& 'THR CB HG23 HG22 HG21 0. 2 HG21 HG22',&
& 'TRP CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'TYR CA CG HB2 HB1 0. 2 HB1 HB2',&
& 'VAL CA CG2 CG1 HB 0. -1 HB CB',&
& 'VAL CB HG13 HG12 HG11 0. 2 HG11 HG12',&
& 'VAL CB HG23 HG22 HG21 0. 2 HG21 HG22'/)
#endif
!
  character(len=80), target :: pseudo_dihedral_rules_charmm22_prop(1)=& ! for the proline patch
#if (KEY_PATHSCALE==0)
& [character(len=80) :: &
& 'PRO CA CD HN2 HN1 0. 2 HN1 HN2'&
&]
#else
&(/'PRO CA CD HN2 HN1 0. 2 HN1 HN2'/)
#endif
!
  logical, parameter :: qdebug=.false.
! subroutines listing
  public chirality_main
  private compute_dihedral
  private chirality_init
  private chirality_finalize
  public chirality_check ! I am making this public so that checks can be called from within CHARMM
                         ! it is obviously the programmer`s responsibility to make sure everything is set up for this call
  private chirality_read_rules
  private compare_rules
!
 contains
!
!============================================
 function compare_rules(r1,r2)
 type(dihedral_rule), intent(in) :: r1, r2
 logical :: compare_rules
 compare_rules=(r1%resname.eq.r2%resname).and.all(r1%aname(1:4).eq.r2%aname(1:4)) ! stop here VO .and.abs(r1%threshold-r2%threshold).lt.1d-6.and.r1%sgn.eq.r2%sgn
 end function compare_rules
!============================================
 subroutine chirality_main(comlyn,comlen,islct)
 use string
 use stream
#if (KEY_MULTICOM==1)
 use multicom_aux; 
#endif
!
 character(len=*) :: comlyn
 integer :: comlen, l, j
 integer, intent(in) :: islct(:)
!
 type(dihedral_rule), pointer :: r
 integer :: errnum, ifile
 logical :: qcheck
 logical :: qadd ! whether to overwite rules upon reading, or to add rules
 character(len=len("CHIRALITY_MAIN>") ),parameter::whoami="CHIRALITY_MAIN>";!macro
!
 qcheck=.false. ! whether to run checker
 qadd=.false.
!
 l=comlen
 999 continue
 if (l.le.1 .or. indxa(comlyn, comlen, 'HELP').gt.0) then
  if &
#if (KEY_MULTICOM==0)
& (iolev.gt.0)& 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0)& 
#endif
& then
  write(info,'(2A)') whoami, ' CHIRALITY CHECKER ( VO / KARPLUS GROUP / HARVARD U. 2012 )'&
& ,whoami, ' _______________________________________________________________________________'&
& ,whoami, ' DESCRIPTION: FIND AND CORRECT CHIRALITY ERRORS ACCORDING'&
& ,whoami, ' DESCRIPTION: TO PRESCRIBED RULES, APPLIED TO SELECTED RESIDUES'&
& ,whoami, ' SYNTAX : coor chirality <COMMANDS> <ATOM SELECTION>'&
& ,whoami, ' COMMANDS CAN BE ONE OR MORE OF THE FOLLOWING:'&
& ,whoami, '   INIT  - initialize/reinitialize'&
& ,whoami, '   FINA  - deallocate all arrays'&
& ,whoami, '   CHECK - check structure'&
& ,whoami, '   FIX   - attempt to correct errors during checking'&
& ,whoami, '   NOFX  - do not attempt to correct errors'&
& ,whoami, '   NOPR  - disable output'&
& ,whoami, '   PRIN  - enable output'&
& ,whoami, '   HELP  - print this screen'&
& ,whoami, '   RULE  - print rules that are currently defined'&
& ,whoami, '   PSEU  - also check for pseudo-chirality errors (i.e. atom ordering)'&
& ,whoami, '   PROP  - also check for pseudo-chirality in the Proline patch (c22)'&
& ,whoami, '   READ <UNIT> <ADD> - read rules from unit (or from input stream if unit is omitted)'&
& ,whoami, '    "ADD" will append rules to the existing rules'&
& ,whoami, ' ATOM SELECTION IS OPTIONAL (ALL ATOMS WILL BE SELECTED BY DEFAULT)'&
& ,whoami, ' _______________________________________________________________________________'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif ! me==0
  return
 endif
!
 if (indxa(comlyn, comlen, 'FINA').gt.0) then ; call chirality_finalize() ; return ; endif
 if ((indxa(comlyn, comlen, 'INIT').gt.0).or.(.not. chirality_initialized)) then ;
  call chirality_init() ;
  qprint=prnlev.ge.3 .and. &
#if (KEY_ENSEMBLE==0) && (KEY_STRINGM==0)
& iolev.gt.0 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0) 
#endif
!
  call chirality_read_rules()
 endif
!
 if (indxa(comlyn, comlen, 'NOPR').gt.0) then
  qprint=.false.
 elseif (indxa(comlyn, comlen, 'PRIN').gt.0) then
  qprint=prnlev.ge.3 .and. &
#if (KEY_ENSEMBLE==0) && (KEY_STRINGM==0)
& iolev.gt.0 
#endif
#if (KEY_ENSEMBLE==1) || (KEY_STRINGM==1)
& (ME_LOCAL.eq.0) 
#endif
!
 endif
!
 if (indxa(comlyn, comlen, 'PSEU').gt.0) then
  dihedral_rules_data=>pseudo_dihedral_rules_charmm22
  qadd=indxa(comlyn, comlen, 'ADD').gt.0
  if (.not.qadd) num_rules=0 ! overwrite rules
  call chirality_read_rules()
  qadd=.true. ! so that additional adds on the same line add to the rule set
 endif
!
 if (indxa(comlyn, comlen, 'PROP').gt.0) then
  dihedral_rules_data=>pseudo_dihedral_rules_charmm22_prop
  if (.not.qadd) qadd=indxa(comlyn, comlen, 'ADD').gt.0
  if (.not.qadd) num_rules=0 ! overwrite rules
  call chirality_read_rules()
  qadd=.true. ! so that additional adds on the same line add to the rule set
 endif
!
 if (indxa(comlyn, comlen, 'READ').gt.0) then
  if (indxa(comlyn, comlen, 'ADD').le.0) num_rules=0 ! overwrite rules
  j=-1 ; ifile=gtrmi(comlyn, comlen, 'UNIT', j); ! support i8 compilation
  if (ifile .eq. -1) then
   ifile=istrm
   call chirality_read_rules(ifile)
  else
   call chirality_read_rules(ifile)
  endif
 endif ! read
!
 if (indxa(comlyn, comlen, 'RULE').gt.0) then
!
  if &
#if (KEY_MULTICOM==0)
& (iolev.gt.0)& 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0)& 
#endif
& then
!
  write(info,'(2A)') whoami, ' __________________________________________________________________________';
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  write(info,'(2A)') whoami, ' THE FOLLOWING RULES ARE CURRENTLY DEFINED:';
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  write(info,'(2A)') whoami, ' __________________________________________________________________________';
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  do j=1,num_rules
   r=>dihedral_rules(j)
   write(info,'(A," ",I3," ",5A,G12.5,I3," ",2A)') whoami, j, r%resname, r%aname(1:4), r%threshold, r%sgn, r%aname(5:6)
   ;write(OUTU,'(A)') pack(info,info.ne.'');info='';
  enddo
  write(info,'(2A)') whoami, ' __________________________________________________________________________';
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif ! me==0
 endif ! RULES
!
 if (indxa(comlyn, comlen, 'NOFX').gt.0) then
  qflip=.false.
 elseif (indxa(comlyn, comlen, 'FIX').gt.0) then ! whether to flip the atom specified in the rule to change chirality
  qflip=.true.
 endif
!
 if ((indxa(comlyn, comlen, 'CHCK').gt.0) .or. &
 & (indxa(comlyn, comlen, 'CHECK').gt.0).or. &
 & (indxa(comlyn, comlen, 'CHEC').gt.0) .or. &
 & (indxa(comlyn, comlen, 'CHK').gt.0) .or. &
 & qcheck) &
 & errnum=chirality_check(islct)
!
 if (comlen.eq.l) then
  comlyn='HELP '//comlyn
  call trima(comlyn, comlen)
  goto 999
 endif
 end subroutine chirality_main
!============================================
 subroutine chirality_init()
 if (chirality_initialized) then
   call chirality_finalize()
 else
   nullify(dihedral_rules, dihedral_rules_bkp)
 endif
 dihedral_rules_data=>dihedral_rules_charmm22
 allocate(dihedral_rules(rules_expand)) ; num_rules=0
 chirality_initialized=.true.
 end subroutine chirality_init
!============================================
 subroutine chirality_finalize()
 if (chirality_initialized) then
   if (associated(dihedral_rules)) deallocate(dihedral_rules)
   num_rules=0
   if (associated(dihedral_rules_bkp)) deallocate(dihedral_rules_bkp)
 endif
 chirality_initialized=.false.
 end subroutine chirality_finalize
!============================================
 subroutine chirality_read_rules(ifile)
 use string
 use stream
 integer, optional :: ifile
 logical :: qread
!
 character(len=len("CHIRALITY_READ_RULES>") ),parameter::whoami="CHIRALITY_READ_RULES>";!macro
 integer, parameter :: maxrulelength=200
 character(len=maxrulelength) :: line
 integer :: i, j, k, l
 integer :: ioerr
!
 type(dihedral_rule), pointer :: r
!
 character, parameter :: comment(2) = (/'*','!'/)
!
 if (present(ifile)) then ; qread=ifile.gt.0 ; else ; qread=.false. ; endif
!
 if (qread) then
  if (qprint) then
   write(info,'(2A,I4)') whoami, ' READING DIHEDRAL RULES FROM UNIT ',ifile;
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  do while (.true.)
! skip comments
   read(ifile,'(A)',IOSTAT=ioerr) line
   if (ioerr.eq.0) then
    l=len(line)
    call trima(line, l)
    if (any(comment.eq.line(1:1))) cycle ! skip comments
    if (l.eq.0) exit
    if (line(1:3).eq.'END') exit
! read rule
! reallocate rules array, if too small
    if (num_rules.eq.size(dihedral_rules)) then ! reallocate
     if (associated(dihedral_rules_bkp)) deallocate(dihedral_rules_bkp)
     allocate(dihedral_rules_bkp(num_rules+rules_expand))
     dihedral_rules_bkp(1:num_rules)=dihedral_rules
     deallocate(dihedral_rules) ; dihedral_rules=>dihedral_rules_bkp ; nullify(dihedral_rules_bkp)
    endif
    r=>dihedral_rules(num_rules+1)
    read(line,*,IOSTAT=ioerr) r%resname, r%aname(1:4), r%threshold, r%sgn, r%aname(5:6)
    if (ioerr.eq.0) then
     num_rules=num_rules+1 ! successful read
! check for duplicate rules
     do j=1,num_rules-1
      if (r .eq. dihedral_rules(j)) then
! if (any(r .eq. dihedral_rules(1:num_rules))) then
       write(info,'(A)') 'IGNORING DUPLICATE OR CONFLICTING RULE: '//line(1:l); 
       call wrndie(0,whoami,trim(info(1)))
       num_rules=num_rules-1
       exit
      endif
     enddo
    else
     call wrndie(0,whoami,trim('DETECTED ERRORS IN RULES FILE.'))
     write(info, '(A)') 'SKIPPING LINE:'//line(1:l); 
     call wrndie(0,whoami,trim(info(1)))
     continue
    endif ! read error
   else ! end of file error
     exit
   endif
  enddo ! over all lines in the file
!
  if (qprint) then
   write(info,'(A,I4,A,I4)') whoami, num_rules, 'DIHEDRAL RULES READ FROM UNIT',ifile ;
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
 else ! qread
! read internal file
! reallocate rules array, if too small
  l=size(dihedral_rules_data)+num_rules
  if (l.gt.size(dihedral_rules)) then ! reallocate
   if (associated(dihedral_rules_bkp)) deallocate(dihedral_rules_bkp)
   allocate(dihedral_rules_bkp(l))
   dihedral_rules_bkp(1:num_rules)=dihedral_rules
   deallocate(dihedral_rules) ; dihedral_rules=>dihedral_rules_bkp ; nullify(dihedral_rules_bkp)
  endif
  j=num_rules
  do i=1,size(dihedral_rules_data)
   j=j+1
   r=>dihedral_rules(j)
   read(dihedral_rules_data(i),*) &
& r%resname, &
& r%aname(1:4), &
& r%threshold, &
& r%sgn,&
& r%aname(5:6)
! the user can load the same internal file more than once (e.g. PSEU)
! therefore we should chek for duplicated here also
   do k=1,j-1
    if (r .eq. dihedral_rules(k)) then
       write(info,'(A)') 'IGNORING DUPLICATE OR CONFLICTING RULE: '//dihedral_rules_data(i)
       call wrndie(0,whoami,trim(info(1)))
     j=j-1
     exit
    endif
   enddo
!
  enddo ! internal file
  num_rules=j
 endif
end subroutine chirality_read_rules
!============================================
 function chirality_check(islct_, r__) result(errnum)
 use string
 use stream
 use coord; use coordc
 use psf
 use consta, only : pi
#if (KEY_MULTICOM==1)
 use multicom_aux; 
#endif
 use confcons, only : ifindc
 use param_store, only: set_param

 implicit none

 integer, optional, intent(in) :: islct_(:)
 integer, pointer :: islct(:)
 real(chm_real), optional :: r__(:,:)
 real(chm_real), pointer :: r_(:,:)
!
 character, parameter :: op(-1:1) = (/'>',' ','<'/) ! trick to write the correct inequality
 type(dihedral_rule), pointer :: rr
 integer :: i,j,ind(6), errnum, ires, iseg
 real(chm_real) :: phi, d
 logical :: flagged
 integer :: ierror
!
!
#if (KEY_PARALLEL==1)
#if (KEY_SINGLE==1)
 integer :: mpifloat=MPI_REAL 
#endif
#if (KEY_SINGLE==0)
 integer :: mpifloat=MPI_REAL8 
#endif
#if (KEY_INTEGER8==0)
 integer :: mpiint=MPI_INTEGER 
#endif
#if (KEY_INTEGER8==1)
 integer :: mpiint=MPI_INTEGER8 
#endif
 integer :: mpichar=MPI_CHARACTER
 integer :: mpibool=MPI_LOGICAL
#endif
!
 character(len=len("CHIRALITY_CHECK>") ),parameter::whoami="CHIRALITY_CHECK>";!macro
!
 if (present(islct_)) then
  allocate(islct(size(islct_))); islct=islct_
 else
  allocate(islct(natom)) ; islct=1
 endif
!
 if (present(r__)) then
   allocate(r_(size(r__,1),size(r__,2))); r_=r__
 else
   allocate(r_(size(x),3)) ; r_(:,1)=x ; r_(:,2)=y ; r_(:,3)=z
 endif
!
!
#if (KEY_MULTICOM==1)
 if (ME_LOCAL.eq.0) then
#elif (KEY_PARALLEL==1)
 if (mynod.eq.0) then
#endif
!
 errnum=0 ! number of errors
!
! charmm-specific code
!
 do iseg=1,nseg ! loop over all segments
  do ires=nictot(iseg)+1,nictot(iseg+1) ! loop over all residues in the segment
! determine residue name if the residue is flagged
   j=ibase(ires)
   flagged=.false.
   do while (.not.flagged.and.j.lt.ibase(ires+1))
    j=j+1
     if (islct(j).eq.1) flagged=.true.
   enddo
   if (flagged) then
! loop over all rules
 rules : do j=1,num_rules
     rr=>dihedral_rules(j)
     if ( res(ires)(1:len_trim(res(ires))).eq.rr%resname(1:len_trim(rr%resname)) .or. &
& rr%resname(1:3).eq.'ANY') &
& then
! find atom indices corresponding to the rule
! use an aux. routine from confcons
      do i=1,6
       ind(i)=ifindc(rr%aname(i),atype,ibase(ires)+1,ibase(ires+1))
       if (ind(i).le.0) then
        if (rr%resname(1:3).ne.'ANY') then ! silent skip on 'ANY' mismatch
          write(info,'(A)') 'SKIPPING RULE '//itoa(j)//': ATOM "'//rr%aname(i)(1:len_trim(rr%aname(i)))//&
& '" NOT FOUND IN RESIDUE "'//res(ires)(1:len_trim(res(ires)))//'"'
          call wrndie(0,whoami,trim(info(1)))
        endif ! rule mismatch
        cycle rules
       endif
      enddo
!
      phi=180d0/pi*compute_dihedral( r_(ind(1),1),r_(ind(1),2),r_(ind(1),3),&
& r_(ind(2),1),r_(ind(2),2),r_(ind(2),3),&
& r_(ind(3),1),r_(ind(3),2),r_(ind(3),3),&
& r_(ind(4),1),r_(ind(4),2),r_(ind(4),3) )
!aaa
! if (qdebug) write(0,*) res(ires), ind, phi, rr%sgn*phi, rr%threshold*phi, qprint
!aaa
      if ( rr%sgn * phi .lt. rr%sgn*rr%threshold) then ! violation
       errnum=errnum+1
       if (qprint) then
        write(info,'(9A)') whoami,' ________________________________________________________________________________';
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
        write(info,'(16A,G11.5,A1,G11.5,A)') whoami,' DIHEDRAL ANGLE ',&
& rr%aname(1)(1:len_trim(rr%aname(1))),'-',&
& rr%aname(2)(1:len_trim(rr%aname(2))),'-',&
& rr%aname(3)(1:len_trim(rr%aname(3))),'-',&
& rr%aname(4)(1:len_trim(rr%aname(4))),&
& ' IN RESIDUE ',rr%resname(1:len_trim(rr%resname)),' ',resid(ires)(1:len_trim(resid(ires))),', SEGID: ',segid(iseg)(1:len_trim(segid(iseg))),&
& ' IS OUT OF RANGE (',phi,op(rr%sgn/abs(rr%sgn)),rr%threshold,')' ;
           write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif ! qprint
       if (qflip) then
         if (abs(rr%sgn).eq.1) then ! invert an atom about a symmetry plane
          r_(ind(5),1)=r_(ind(6),1)-(r_(ind(5),1)-r_(ind(6),1))
          r_(ind(5),2)=r_(ind(6),2)-(r_(ind(5),2)-r_(ind(6),2))
          r_(ind(5),3)=r_(ind(6),3)-(r_(ind(5),3)-r_(ind(6),3))
          if (qprint) then
 write(info,'(5A)') whoami, ' REFLECTED ATOM ',rr%aname(5)(1:len_trim(rr%aname(5))),' ABOUT ATOM ',rr%aname(6)(1:len_trim(rr%aname(6))) ;
 write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
         elseif (abs(rr%sgn).eq.2) then ! swap coordinates of two atoms
          d=r_(ind(5),1) ; r_(ind(5),1)=r_(ind(6),1) ; r_(ind(6),1)=d
          d=r_(ind(5),2) ; r_(ind(5),2)=r_(ind(6),2) ; r_(ind(6),2)=d
          d=r_(ind(5),3) ; r_(ind(5),3)=r_(ind(6),3) ; r_(ind(6),3)=d
          if (qprint) then
 write(info,'(5A)') whoami, ' SWAPPED COORDINATES OF ATOMS ',rr%aname(5)(1:len_trim(rr%aname(5))),&
                             &' AND ',rr%aname(6)(1:len_trim(rr%aname(6))) ;
                             write(OUTU,'(A)') pack(info,info.ne.'');info='';
          endif
         else
          write(info,'(A)') ' UNKNOWN DIRECTIVE CODE ('//itoa(abs(rr%sgn))//') IN RULE '//itoa(j)
          call wrndie(0,whoami,trim(info(1)))
         endif
! recompute dihedral
         phi=180d0/pi*compute_dihedral( r_(ind(1),1),r_(ind(1),2),r_(ind(1),3),&
& r_(ind(2),1),r_(ind(2),2),r_(ind(2),3),&
& r_(ind(3),1),r_(ind(3),2),r_(ind(3),3),&
& r_(ind(4),1),r_(ind(4),2),r_(ind(4),3) )
         if (qprint) then
          write(info,'(2A,G11.5)') whoami,' NEW DIHEDRAL VALUE IS ',phi ;
          write(OUTU,'(A)') pack(info,info.ne.'');info='';
         endif ! qprint
       endif ! qflip
      endif ! violation
     endif ! rule match
    enddo rules ! over all rules
   endif ! residue is selected (flagged)
  enddo ! over all residues
 enddo ! over all segments
!
!============== BROADCAST TO OTHER NODES ====
#if (KEY_PARALLEL==1) || (KEY_MULTICOM==1)
 endif ! mynod == 0
 call mpi_bcast(r_,3*natom,mpifloat,0,MPI_COMM_LOCAL,ierror)
#endif
! populate main coordinates, if needed
 if (present(r__)) then
  r__=r_
 else
  x=r_(:,1) ; y=r_(:,2) ; z=r_(:,3)
 endif
!
! set error count
 call set_param('CHIERR',errnum)
!
! print summary
!
 if (qprint) then
  write(info,'(9A)') whoami,' ________________________________________________________________________________' ;
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  if (qflip.and.errnum.gt.0) then
   write(info,'(A,I5,A)') whoami, errnum, ' VIOLATIONS WERE FOUND AND CORRECTED. SYSTEM MAY NEED TO BE MINIMIZED.'
  else
   write(info,'(A,I5,A)') whoami, errnum, ' VIOLATIONS WERE FOUND.'
  endif
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
 endif
!
 if (associated(r_)) deallocate(r_)
 if (associated(islct)) deallocate(islct)
!
end function chirality_check
!============================================================================
 function compute_dihedral(x1,y1,z1,x2,y2,z2,x3,y3,z3,x4,y4,z4) result(theta)
 real(chm_real) :: theta, costh, sinth,&
& x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4,&
& dx12, dy12, dz12, dx23, dy23, dz23, dx34, dy34, dz34,&
& vx, vy, vz, vn, ux, uy, uz, un, wx, wy, wz, wn
 dx12=x2-x1; dy12=y2-y1; dz12=z2-z1;
 dx23=x3-x2; dy23=y3-y2; dz23=z3-z2;
 dx34=x4-x3; dy34=y4-y3; dz34=z4-z3;
! note:
! costh = [ (d34 x d32) . (d23 x d21) / |(d34 x d23)| |(d23 x d12)| ] =
! = cos [ (u . v) / |u| |v| ] (definition of u and v)
 ux=dy23*dz34-dy34*dz23;
 uy=dz23*dx34-dz34*dx23;
 uz=dx23*dy34-dx34*dy23;
 un=sqrt(ux*ux+uy*uy+uz*uz);
!
 vx=dy12*dz23-dy23*dz12;
 vy=dz12*dx23-dz23*dx12;
 vz=dx12*dy23-dx23*dy12;
 vn=sqrt(vx*vx+vy*vy+vz*vz);
!
 wx=dy23*vz-vy*dz23;
 wy=dz23*vx-vz*dx23;
 wz=dx23*vy-vx*dy23;
 wn=sqrt(wx*wx+wy*wy+wz*wz);
!
 if (un.eq.0d0) un=1d0
 if (vn.eq.0d0) vn=1d0
 if (wn.eq.0d0) wn=1d0
 costh=(ux*vx+uy*vy+uz*vz)/(un*vn)
 sinth=(wx*ux+wy*uy+wz*uz)/(wn*un)
 theta=atan2(sinth, costh)
!
 end function compute_dihedral
!
#endif /* automatically protect all code */
end module chirality
