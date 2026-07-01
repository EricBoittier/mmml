! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!



module confcons
#if (KEY_STRINGM==1) /*  automatically protect all code */
 use chm_kinds
 implicit none
 private
!
 !
 ! in this module :
 ! the CONFormational CONSistency module
 ! is applicable to groups of atoms that have
 ! rotational symmetry because some of the constituent
 ! atoms are indistinguishable, e.g. H1, H2, and H3 in
 ! a methyl group. In specifying the coordinates of a molecule
 ! with such a group, the atoms can be arranged in three completely
 ! equivalent ways, relative to a reference conformation.
 ! These are H1/H2/H3, H2/H3/H1, H3/H1/H2. They are related by
 ! rotations around the bond that involves the parent atom
 ! (e.g. methyl carbon). Note that the conformation H1/H3/H2
 ! is not allowed because it corresponds to a change of (pseudo)
 ! chirality, which is assumed to be fixed (see chirality module)
 ! The purpose of this module is to identify all such symmetry groups
 ! in a molecular coordinate set and choose the rotation that
 ! brings the coordinates into the closest proximity (in the sense
 ! of a certain RMSD ) to the coordinates in a reference set.
 ! This is done according to a set of rules, which can be
 ! specified by the user; a default set for the charmm22 protein topology
 ! is defined in this module
!
!
  type confcons_rule
   character(len=8) :: resname
   character(len=8), pointer :: test_atoms(:)
   character(len=8), pointer :: orient_atoms(:)
   character(len=8), pointer :: orient_atom_permutations(:,:)
   character(len=8), pointer :: other_atom_permutations(:,:)
! contains ! some compilers lack this feature
! final :: confcons_rule_finalize
  end type confcons_rule
!
  interface operator(.eq.)
   module procedure compare_rules
  end interface operator(.eq.)
!
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
!
  type(confcons_rule), pointer :: confcons_rules(:), confcons_rules_bkp(:)
  integer, parameter :: rules_expand = 10 ! reallocation size increment
  integer :: num_rules=0
!
  logical :: confcons_initialized=.false., qprint=.true., qfix=.false.
!
  character(len=80), pointer :: confcons_rules_data(:)
  character(len=80), target :: confcons_rules_charmm22(5)=&

#if (KEY_PATHSCALE==0)
& [character(len=80) :: &
& 'ARG 1 CD 1 CZ    2 1 NH1 NH2         2 HH11 HH21 HH12 HH22', &
& 'ASP 1 CA 1 CG    2 1 OD1 OD2         0', &
& 'GLU 1 CB 1 CD    2 1 OE1 OE2         0', &
& 'PHE 1 CA 0       2 2 CD1 CD2 CE2 CE1 2 HD1 HD2 HE1 HE2', &
& 'TYR 1 CA 0       2 2 CD1 CD2 CE2 CE1 2 HD1 HD2 HE1 HE2' &
&]
#else
& (/'ARG 1 CD 1 CZ    2 1 NH1 NH2         2 HH11 HH21 HH12 HH22', &
& 'ASP 1 CA 1 CG    2 1 OD1 OD2         0', &
& 'GLU 1 CB 1 CD    2 1 OE1 OE2         0', &
& 'PHE 1 CA 0       2 2 CD1 CD2 CE2 CE1 2 HD1 HD2 HE1 HE2', &
& 'TYR 1 CA 0       2 2 CD1 CD2 CE2 CE1 2 HD1 HD2 HE1 HE2'/)
#endif
! other rules
  character(len=80), target :: hydrogen_confcons_rules_charmm22(17)=&
#if (KEY_PATHSCALE==0)
& [character(len=80) :: &
& 'ALA 1 HA 1 CB   3 1 HB1  HB2  HB3  0', &
!
& 'ARG 1 NE 1 NH1  2 1 HH11 HH12      0', &
& 'ARG 1 NE 1 NH2  2 1 HH21 HH22      0', &
!
& 'ASN 1 CB 1 ND2  2 1 HD21 HD22      0', &
& 'GLN 1 CG 1 NE2  2 1 HE21 HE22      0', &
!
& 'ILE 1 CB 1 CD   3 1 HD1  HD2  HD3  0', &
& 'ILE 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
!
& 'LEU 1 CB 1 CD1  3 1 HD11 HD12 HD13 0', &
& 'LEU 1 CB 1 CD2  3 1 HD21 HD22 HD23 0', &
!
& 'LYS 1 CD 1 NZ   3 1 HZ1  HZ2  HZ3  0', &
& 'MET 1 CG 1 CE   3 1 HE1  HE2  HE3  0', &
& 'THR 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
!
& 'VAL 1 CA 1 CG1  3 1 HG11 HG12 HG13 0', &
& 'VAL 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
!
& 'ALAD 1 OL 1 CL 3 1 HL1 HL2 HL3 0', &
& 'ALAD 1 HR 1 CR 3 1 HR1 HR2 HR3 0', &
& 'ALAD 1 HA 1 CB 3 1 HB1 HB2 HB3 0' &
&]
#else
& (/'ALA 1 HA 1 CB   3 1 HB1  HB2  HB3  0', &
& 'ARG 1 NE 1 NH1  2 1 HH11 HH12      0', &
& 'ARG 1 NE 1 NH2  2 1 HH21 HH22      0', &
& 'ASN 1 CB 1 ND2  2 1 HD21 HD22      0', &
& 'GLN 1 CG 1 NE2  2 1 HE21 HE22      0', &
& 'ILE 1 CB 1 CD   3 1 HD1  HD2  HD3  0', &
& 'ILE 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
& 'LEU 1 CB 1 CD1  3 1 HD11 HD12 HD13 0', &
& 'LEU 1 CB 1 CD2  3 1 HD21 HD22 HD23 0', &
& 'LYS 1 CD 1 NZ   3 1 HZ1  HZ2  HZ3  0', &
& 'MET 1 CG 1 CE   3 1 HE1  HE2  HE3  0', &
& 'THR 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
& 'VAL 1 CA 1 CG1  3 1 HG11 HG12 HG13 0', &
& 'VAL 1 CA 1 CG2  3 1 HG21 HG22 HG23 0', &
& 'ALAD 1 OL 1 CL 3 1 HL1 HL2 HL3 0', &
& 'ALAD 1 HR 1 CR 3 1 HR1 HR2 HR3 0', &
& 'ALAD 1 HA 1 CB 3 1 HB1 HB2 HB3 0' /)
#endif
!
! subroutines listing
  public confcons_main
  private confcons_init
  private confcons_finalize
  public confcons_check
  private confcons_read_rules
  private compare_rules
  private confcons_rule_finalize
  private confcons_rules_finalize
  private confcons_rules_initialize
 public ifindc
!
 contains
!!============================================
 function compare_rules(r1,r2) result(q)
 type(confcons_rule), intent(in) :: r1, r2
 logical :: q
 q=.false.
 if (r1%resname.ne.r2%resname) return
 if (.not.(associated(r1%test_atoms).eqv.associated(r2%test_atoms))) return
 if (.not.(associated(r1%orient_atoms).eqv.associated(r2%orient_atoms))) return
 if (.not.(associated(r1%orient_atom_permutations).eqv.associated(r2%orient_atom_permutations))) return
 if (.not.(associated(r1%other_atom_permutations).eqv.associated(r2%other_atom_permutations))) return
!
 if (size(r1%test_atoms).ne.size(r2%test_atoms)) return
 if (size(r1%orient_atoms).ne.size(r2%orient_atoms)) return
 if (size(r1%orient_atom_permutations).ne.size(r2%orient_atom_permutations)) return
 if (size(r1%other_atom_permutations).ne.size(r2%other_atom_permutations)) return
!
 if (any(r1%test_atoms.ne.r2%test_atoms)) return
 if (any(r1%orient_atoms.ne.r2%orient_atoms)) return
 if (any(r1%orient_atom_permutations.ne.r2%orient_atom_permutations)) return
 if (any(r1%other_atom_permutations.ne.r2%other_atom_permutations)) return
!
 q=.true. ! if we are here, then all tests passed
 end function compare_rules
!============================================
 subroutine confcons_rule_finalize(r)
 type(confcons_rule) :: r
 if(associated(r%test_atoms))deallocate(r%test_atoms)
 if(associated(r%orient_atoms))deallocate(r%orient_atoms)
 if(associated(r%orient_atom_permutations))deallocate(r%orient_atom_permutations)
 if(associated(r%other_atom_permutations))deallocate(r%other_atom_permutations)
 end subroutine confcons_rule_finalize
!============================================
 subroutine confcons_rules_finalize(rules)
 type(confcons_rule), pointer :: rules(:), r
 integer :: i
 do i=1,size(rules)
  r=>rules(i)
  call confcons_rule_finalize(r)
 enddo
 end subroutine confcons_rules_finalize
!============================================
 subroutine confcons_rules_initialize(rules)
 type(confcons_rule), pointer :: rules(:), r
 integer :: i
 do i=1,size(rules)
  r=>rules(i)
  nullify(r%test_atoms,r%orient_atoms,r%orient_atom_permutations,r%other_atom_permutations)
 enddo
 end subroutine confcons_rules_initialize
!============================================
 subroutine confcons_init()
 if (confcons_initialized) then
   call confcons_finalize()
 else
   nullify(confcons_rules, confcons_rules_bkp)
 endif
 confcons_rules_data=>confcons_rules_charmm22
 allocate(confcons_rules(rules_expand)) ; num_rules=0
 call confcons_rules_initialize(confcons_rules)
 confcons_initialized=.true.
 end subroutine confcons_init
!============================================
 subroutine confcons_finalize()
 if (confcons_initialized) then
   if (associated(confcons_rules)) then
    call confcons_rules_finalize(confcons_rules)
    deallocate(confcons_rules)
    num_rules=0
   endif
   if (associated(confcons_rules_bkp)) then
    call confcons_rules_finalize(confcons_rules_bkp)
    deallocate(confcons_rules_bkp)
   endif
 endif
 confcons_initialized=.false.
 end subroutine confcons_finalize
!============================================
 subroutine confcons_read_rules(ifile)
 use string
 use stream
!
 integer, optional :: ifile
 logical :: qread
!
 character(len=len("CONFCONS_READ_RULES>") ),parameter::whoami="CONFCONS_READ_RULES>";!macro
 integer, parameter :: maxrulelength=150
 character(len=maxrulelength) :: line, line_bkp
 integer :: i, j, k, l, m, n, ii, l_bkp
 integer :: ioerr
!
 type(confcons_rule), pointer :: r
!
 character, parameter :: comment(2) = (/'*','!'/)
!
 if (present(ifile)) then ; qread=ifile.gt.0 ; else ; qread=.false. ; endif
!
 if (qread) then
  if (qprint) then
   write(info,'(2A,I4)') whoami, ' READING RULES FROM UNIT ',ifile
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  do while (.true.)
! skip comments
   read(ifile,'(A)',IOSTAT=ioerr) line
!

!
   if (ioerr.eq.0) then
    l=len(line)
    call trima(line, l)
    if (any(comment.eq.line(1:1))) cycle ! skip comments
    if (l.eq.0) exit
    if (line(1:3).eq.'END') exit
! read rule
! reallocate rules array, if too small
    if (num_rules.eq.size(confcons_rules)) then ! reallocate
     if (associated(confcons_rules_bkp)) then
      call confcons_rules_finalize(confcons_rules_bkp)
      deallocate(confcons_rules_bkp)
     endif
     allocate(confcons_rules_bkp(num_rules+rules_expand))
     call confcons_rules_initialize(confcons_rules_bkp) ! nullify backup pointers
     confcons_rules_bkp(1:num_rules)=confcons_rules ! copy rules to new location (pointer assignment; data stays)
! cannot "finalize" confcons because that will destroy the data, not just the pointers
! call confcons_rules_finalize(confcons_rules) ! destroy original rules array
     deallocate(confcons_rules) ; confcons_rules=>confcons_rules_bkp ! point to new rules array
! cannot initialize below because it will nullify _the same_ pointers that confcons_rules points to
! call confcons_rules_initialize(confcons_rules_bkp) ;
     nullify(confcons_rules_bkp) ! nullify backup pointers
    endif ! end of reallocation
    r=>confcons_rules(num_rules+1)
! read rule using string processing routines
    line_bkp=line ; l_bkp=l ! save line because it is destroyed by string routines
! residue name
    r%resname=nexta8(line,l)
! test atoms
    j=nexti(line, l)
    if (j.ge.0) then ; allocate(r%test_atoms(j)) ; do i=1,j ; r%test_atoms(i)=nexta8(line,l) ; enddo
    else ; ioerr = 1 ; endif
! orientation atoms
    j=nexti(line, l)
    if (j.ge.0) then ; allocate(r%orient_atoms(j)) ; do i=1,j ; r%orient_atoms(i)=nexta8(line,l) ; enddo
    else ; ioerr = 1 ; endif
! permutation pairs (part of orientation set)
    k=nexti(line, l); ! number of atoms inside group (permutations)
    if (k.gt.0) then
     j=nexti(line, l) ! number of orientation groups
     if (j.gt.0) then
      allocate(r%orient_atom_permutations(k,j))
      do i=1,j ; do m=1,k ; r%orient_atom_permutations(m,i)=nexta8(line,l) ; enddo ; enddo
     else ; ioerr = 1 ; endif ! j>0
!
     j=nexti(line, l) ! number of other groups that are to be permuted (not part of orientation)
     if (j.ge.0) then ! this can be zero if there are no other atoms
      allocate(r%other_atom_permutations(k,j))
      do i=1,j ; do m=1,k ; r%other_atom_permutations(m,i)=nexta8(line,l) ; enddo ; enddo
     else ; ioerr = 1 ; endif ! j>0 ?
    else ; ioerr = 1 ; endif ! k>0 ?
!
    if (ioerr.eq.0) then
     num_rules=num_rules+1 ! successful read
! check for duplicate rules
     do j=1,num_rules-1
      if (r .eq. confcons_rules(j)) then
! if (any(r .eq. confcons_rules(1:num_rules))) then
       write(info,'(A)') 'IGNORING DUPLICATE OR CONFLICTING RULE: '//line_bkp(1:l_bkp)
       call wrndie(0,whoami,trim(info(1)))
       num_rules=num_rules-1
       exit
      endif
     enddo
    else
     call wrndie(0,whoami,trim('DETECTED ERRORS IN RULES FILE.'))
     write(info, '(A)') 'SKIPPING LINE:'//line_bkp(1:l_bkp)
     call wrndie(0,whoami,trim(info(1)))
     continue
    endif ! read error
   else ! end of file error
     exit
   endif
  enddo ! over all lines in the file
!
  if (qprint) then
   write(info,'(A,I4,A,I4)') whoami, num_rules, ' RULES READ FROM UNIT',ifile
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
 else ! qread
! read internal file
! reallocate rules array, if too small
  ii=size(confcons_rules_data)+num_rules ! total number of rules
  if (ii.gt.size(confcons_rules)) then ! reallocate
   if (associated(confcons_rules_bkp)) then
    call confcons_rules_finalize(confcons_rules_bkp)
    deallocate(confcons_rules_bkp)
   endif
   allocate(confcons_rules_bkp(ii)) ! new size
   call confcons_rules_initialize(confcons_rules_bkp) ! nullify pointers inside bkp_rules
   confcons_rules_bkp(1:num_rules)=confcons_rules ! copy rules from rule array (pointer assignment; data in place)
! cannot "finalize" confcons because that will destroy the data, not just the pointers
! call confcons_rules_finalize(confcons_rules) ! destroy original rules array
   deallocate(confcons_rules) ; confcons_rules=>confcons_rules_bkp ! point to new rules array
! cannot initialize below because it will nullify _the same_ pointers that confcons_rules points to
! call confcons_rules_initialize(confcons_rules_bkp) ;
   nullify(confcons_rules_bkp) ! nullify backup rules array
  endif ! end of reallocation
!
  n=num_rules
  do ii=1,size(confcons_rules_data)
   read(confcons_rules_data(ii),'(A)') line
   l=len(line)
   call trima(line, l)
   n=n+1
   r=>confcons_rules(n)
! read rule using string processing routines
   r%resname=nexta8(line,l)
   j=nexti(line, l) ; allocate(r%test_atoms(j)) ; do i=1,j ; r%test_atoms(i)=nexta8(line,l) ; enddo
   j=nexti(line, l) ; allocate(r%orient_atoms(j)) ; do i=1,j ; r%orient_atoms(i)=nexta8(line,l) ; enddo
!
   k=nexti(line, l) ! number of atoms inside group (permutations)
   j=nexti(line, l) ! number of orientation groups
   allocate(r%orient_atom_permutations(k,j))
   do i=1,j ; do m=1,k ; r%orient_atom_permutations(m,i)=nexta8(line,l) ; enddo ; enddo
!
   j=nexti(line, l) ! number of other groups that are to be permuted (not part of orientation)
   allocate(r%other_atom_permutations(k,j))
   do i=1,j ; do m=1,k ; r%other_atom_permutations(m,i)=nexta8(line,l) ; enddo ; enddo
!
! the user can load the same internal file more than once (e.g. HYDR)
! therefore we should chek for duplicated here also
   do j=1,n-1
    if (r .eq. confcons_rules(j)) then
! if (any(r .eq. confcons_rules(1:num_rules))) then
     write(info, '(A)') 'IGNORING DUPLICATE OR CONFLICTING RULE: '//confcons_rules_data(ii)
     call wrndie(0,whoami,trim(info(1)))
     n=n-1
     exit
    endif
   enddo
!






  enddo ! internal file
  num_rules=n
 endif
end subroutine confcons_read_rules
!============================================
subroutine confcons_main(comlyn,comlen,islct)
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
 type(confcons_rule), pointer :: r
 integer :: errnum, ifile
 integer :: ngroups, npermute
 logical :: qcheck
 character(len=len("CONFCONS_MAIN>") ),parameter::whoami="CONFCONS_MAIN>";!macro
!
 qcheck=.false. ! whether to run checker
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
  write(info,'(2A)') whoami, ' CONFORMATION CONSISTENCY CHECKER ( VO / KARPLUS GROUP / HARVARD U. 2008-12 )',&
  & whoami, ' ____________________________________________________________________________________',&
  & whoami, ' DESCRIPTION: FIND AND ELIMINATE APPARENT INCONSISTENCY IN THE LABELING OF',&
  & whoami, ' DESCRIPTION: ATOMS IN RESIDUES WITH SYMMETRY ACCORDING TO PRESCRIBED RULES',&
  & whoami, ' SYNTAX : coor confcons <COMMANDS> <ATOM SELECTION>',&
  & whoami, ' COMMANDS CAN BE ONE OR MORE OF THE FOLLOWING:',&
  & whoami, '   INIT  - initialize/reinitialize',&
  & whoami, '   FINA  - deallocate all arrays',&
  & whoami, '   CHECK - check structure',&
  & whoami, '   FIX   - attempt to correct errors during checking',&
  & whoami, '   NOFX  - do not attempt to correct errors',&
  & whoami, '   NOPR  - disable output',&
  & whoami, '   PRIN  - enable output',&
  & whoami, '   HELP  - print this screen',&
  & whoami, '   RULE  - print rules that are currently defined',&
  & whoami, '   HYDR <ADD> - check rules involving hydrogen atom groups (excluded by default)',&
  & whoami, '   READ <UNIT> <ADD> - read rules from unit (or from input stream if unit is omitted)',&
  & whoami, '    "ADD" will append rules to the existing rules',&
  & whoami, ' ATOM SELECTION IS OPTIONAL (ALL ATOMS WILL BE SELECTED BY DEFAULT)',&
  & whoami, ' NOTE: DEFAULT RULES ARE BASED ON THE CHARMM 22 PROTEIN RESIDUE TOPOLOGY',&
  & whoami, ' ____________________________________________________________________________________'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  return
 endif
!
 if (indxa(comlyn, comlen, 'FINA').gt.0) then ; call confcons_finalize() ; return ; endif
 if ((indxa(comlyn, comlen, 'INIT').gt.0).or.(.not. confcons_initialized)) then ;
  call confcons_init() ;
  qprint=prnlev.ge.3 .and. &
#if (KEY_MULTICOM==0)
& iolev.gt.0 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0) 
#endif
!
  call confcons_read_rules()
 endif
!
 if (indxa(comlyn, comlen, 'NOPR').gt.0) then
  if (qprint) then
   write(info,'(2A)') whoami,' WILL PRINT ESSENTIAL OUTPUT'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  qprint=.false.
 elseif (indxa(comlyn, comlen, 'PRIN').gt.0) then
  qprint=prnlev.ge.3 .and. &
#if (KEY_MULTICOM==0)
& iolev.gt.0 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0) 
#endif
  if (qprint) then
   write(info,'(2A)') whoami,' WILL PRINT COMPLETE OUTPUT'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
 endif
!
 if (indxa(comlyn, comlen, 'HYDR').gt.0) then
  confcons_rules_data=>hydrogen_confcons_rules_charmm22
  if (indxa(comlyn, comlen, 'ADD').le.0) num_rules=0 ! overwrite rules
  call confcons_read_rules()
 endif
!
 if (indxa(comlyn, comlen, 'READ').gt.0) then
  if (indxa(comlyn, comlen, 'ADD').le.0) num_rules=0 ! overwrite rules
  j=-1; ifile=gtrmi(comlyn, comlen, 'UNIT', j); ! support i8 compilation
  if (ifile .eq. -1) then
   ifile=istrm
   call confcons_read_rules(ifile)
  else
   call confcons_read_rules(ifile)
  endif
 endif ! read
!
 if (indxa(comlyn, comlen, 'RULE').gt.0) then
  if &
#if (KEY_MULTICOM==0)
& (iolev.gt.0)& 
#endif
#if (KEY_MULTICOM==1)
& (ME_LOCAL.eq.0)& 
#endif
& then
  write(info,'(2A)') whoami, ' __________________________________________________________________________'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  write(info,'(2A)') whoami, ' THE FOLLOWING RULES ARE CURRENTLY DEFINED:'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  write(info,'(2A)') whoami, ' __________________________________________________________________________'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  do j=1,num_rules
   r=>confcons_rules(j)
   write(info,'(A," ",I3," : ",A)') whoami, j, r%resname
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
   ngroups=size(r%orient_atom_permutations,2)
   npermute=size(r%orient_atom_permutations,1)
   if (ngroups.eq.1) then
    write(info,'(A,'//itoa(npermute)//'A)') '    ORIENTATION GROUPS      : ',  r%orient_atom_permutations
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   elseif (ngroups.gt.1) then
    write(info,'(A,'//itoa(npermute)//'A,'//itoa(ngroups-1)//'(" / ",'//itoa(npermute)//'A))') '    ORIENTATION GROUPS      : ',  r%orient_atom_permutations
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
!
   if (size(r%orient_atoms).gt.0) then
    write(info,'(A,'//itoa(size(r%orient_atoms))//'A)' )             '    OTHER ORIENTATION ATOMS : ',  r%orient_atoms
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
   if (size(r%orient_atoms).gt.0) then
    write(info,'(A,'//itoa(size(r%test_atoms))//'A)' )               '    TEST ATOMS              : ',  r%test_atoms
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
!
   ngroups=size(r%other_atom_permutations,2)
   if (ngroups.eq.1) then
    write(info,'(A,'//itoa(npermute)//'A)') '    OTHER PERMUTATION GROUPS: ',  r%orient_atom_permutations
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
   elseif (ngroups.gt.1) then
    write(info,'(A,'//itoa(npermute)//'A,'//itoa(ngroups-1)//'(" / ",'//itoa(npermute)//'A))') '    OTHER PERMUTATION GROUPS: ',  r%other_atom_permutations
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
  enddo
  write(info,'(2A)') whoami, ' __________________________________________________________________________'
  write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif ! me==0
 endif ! RULES
!
 if (indxa(comlyn, comlen, 'NOFX').gt.0) then
  if (qprint) then
   write(info,'(2A)') whoami,' WILL NOT ATTEMPT TO FIX INCONSISTENCIES'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  qfix=.false.
 elseif (indxa(comlyn, comlen, 'FIX').gt.0) then ! whether to move atoms to fix the inconsistency
  if (qprint) then
   write(info,'(2A)') whoami,' WILL ATTEMPT TO FIX INCONSISTENCIES'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
  qfix=.true.
 endif
!
 if ((indxa(comlyn, comlen, 'CHCK').gt.0) .or. &
 & (indxa(comlyn, comlen, 'CHECK').gt.0) .or.&
 & (indxa(comlyn, comlen, 'CHEC').gt.0) .or.&
 & (indxa(comlyn, comlen, 'CHK').gt.0) .or. &
 & qcheck) &
 & errnum=confcons_check(islct)
!
 if (comlen.eq.l) then ! no valid optional were found
  comlyn='HELP '//comlyn
  call trima(comlyn, comlen)
  call wrndie(0,whoami,trim('NO VALID COMMAND(S) SPECIFIED'))
  goto 999
 endif
!
end subroutine confcons_main
!============================================
function confcons_check(islct_, r__, rref__) result(errnum)
 use string
 use stream
 use multicom_aux;
 use coord; use coordc
 use psf
 use number
 use bestfit, only : eig3s, RMSBestFit, rmsd, norm3, veccross3
!
 use corsubs,only:rotls1
 use param_store, only: set_param

  implicit none

#if (KEY_NEWBESTFIT==1) /*  VO 1.12 */
! use cnst_fcm, only: qnewbestfit
! avoid modifying cnst_fcm
   logical, parameter :: qnewbestfit=.true.
#endif
!
 integer, optional, intent(in) :: islct_(:)
 integer, pointer :: islct(:)
 real(chm_real), optional :: r__(:,:), rref__(:,:) ! perform checking using optional coordinate arrays
 real(chm_real), pointer, dimension(:,:) :: r_, rref_
!
 type(confcons_rule), pointer :: rr
 integer :: i, j, k, l, m, errnum, ires, iseg
 integer :: itest, iorient, iorient_permute, iother_permute, itotal, ipermute, npermute, ngroups
 integer :: istart, iend
 integer :: i0, i1
 integer, allocatable :: ind(:)
 real(chm_real), allocatable :: r1(:,:), r2(:,:), ow(:)
 real(chm_real) :: u(3,3), rcom1(3), rcom2(3) ! rotation matrix, center-of-mass coordinates
 real(chm_real) :: msd0, msd1, msd2
 logical :: flagged
 character(len=len("CONFCONS_CHECK>") ),parameter::whoami="CONFCONS_CHECK>";!macro
 integer :: ierror
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
 if (present(rref__)) then
   allocate(rref_(size(rref__,1),size(rref__,2))); rref_=rref__
 else
   allocate(rref_(size(x),3)) ; rref_(:,1)=xcomp ; rref_(:,2)=ycomp ; rref_(:,3)=zcomp
 endif
!
#if (KEY_MULTICOM==1)
 if (ME_LOCAL.eq.0) then
#elif (KEY_PARALLEL==1)
 if (mynod.eq.0) then
#endif
!
 errnum=0 ! number of errors
!
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
     rr=>confcons_rules(j)
!
     if (res(ires)(1:len_trim(res(ires))).eq.rr%resname(1:len_trim(rr%resname))) then ! rule match :
!=======================================================================================
! find atom indices corresponding to the rule
! offsets for a single index array when all of the atoms are concatenated:
     itest =0
     iorient =size(rr%test_atoms)+itest
     iorient_permute=size(rr%orient_atoms)+iorient
     iother_permute =size(rr%orient_atom_permutations)+iorient_permute
     itotal =size(rr%other_atom_permutations)+iother_permute
!
     allocate(ind(itotal),r1(itotal,3),r2(itotal,3),ow(itotal))
!
     istart=ibase(ires)+1 ; iend=ibase(ires+1)
     do i=itest+1, iorient ;
      ind(i)=ifindc(rr%test_atoms (i-itest),atype,istart,iend);
! test atoms
      if (ind(i).le.0) then
        write(info, '(A)') 'SKIPPING RULE '//itoa(j)//': ATOM NOT FOUND IN RESIDUE.'
        call wrndie(0,whoami,trim(info(1)))
        deallocate(ind, r1, r2, ow) ; cycle rules
      endif
!
     enddo
! orientation atoms
     do i=iorient+1, iorient_permute ;
      ind(i)=ifindc(rr%orient_atoms(i-iorient),atype,istart,iend);
!
      if (ind(i).le.0) then
        write(info, '(A)') 'SKIPPING RULE '//itoa(j)//': ATOM NOT FOUND IN RESIDUE.'
        call wrndie(0,whoami,trim(info(1)))
        deallocate(ind, r1, r2, ow) ; cycle rules
      endif
!
     enddo
! permutation atoms (part of orientation set)
     i=iorient_permute+1
     do l=1,size(rr%orient_atom_permutations,2) ! over atom groups
      do k=1,size(rr%orient_atom_permutations,1) ! over permutations
       ind(i)=ifindc(rr%orient_atom_permutations(k,l),atype,istart,iend)
!
       if (ind(i).le.0) then
        write(info, '(A)') 'SKIPPING RULE '//itoa(j)//': ATOM NOT FOUND IN RESIDUE.' 
        call wrndie(0,whoami,trim(info(1)))
        deallocate(ind, r1, r2, ow) ; cycle rules
       endif
!
       i=i+1
      enddo
     enddo
! other permutation atoms (if any)
     i=iother_permute+1
     do l=1,size(rr%other_atom_permutations,2) ! over atom groups
      do k=1,size(rr%other_atom_permutations,1) ! over permutations
       ind(i)=ifindc(rr%other_atom_permutations(k,l),atype,istart,iend)
!
       if (ind(i).le.0) then
        write(info,'(A)') 'SKIPPING RULE '//itoa(j)//': ATOM NOT FOUND IN RESIDUE.'
        call wrndie(0,whoami,trim(info(1)))
        deallocate(ind, r1, r2, ow) ; cycle rules
       endif
!
       i=i+1
      enddo
     enddo
!



! set orientation weights
     ow=0d0
     ow(iorient+1:iother_permute) = 1d0 / ( iother_permute - iorient ) ! does not vary with permutations
!
! compute initial (zeroth permutation)
!
! load test/orientation coordinates
     r1=0d0; r2=0d0;
     do i=itest+1,iother_permute
      r1(i,:)=r_(ind(i),:)
      r2(i,:)=rref_(ind(i),:)
     enddo
! check for undefined coords
     if ( any ( (r1.eq.anum) .or. (r2.eq.anum) ) ) then
!
      if (qprint) then
       write(info,'(6A)') whoami, 'UNDEFINED TEST/ORIENTATION COORDINATES IN RESIDUE ',&
& resid(ires)(1:len_trim(resid(ires))),rr%resname(1:len_trim(rr%resname)),&
& ' ',segid(iseg)(1:len_trim(segid(iseg))), '. SKIPPING RULE.'
       write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
      deallocate(ind, r1, r2, ow) ; cycle rules
     endif
!



! remove COM before orientation
     rcom1=matmul(transpose(r1),ow); r1(:,1)=r1(:,1)-rcom1(1); r1(:,2)=r1(:,2)-rcom1(2); r1(:,3)=r1(:,3)-rcom1(3);
     rcom2=matmul(transpose(r2),ow); r2(:,1)=r2(:,1)-rcom2(1); r2(:,2)=r2(:,2)-rcom2(2); r2(:,3)=r2(:,3)-rcom2(3);


! compute best-fit orientation
#if (KEY_NEWBESTFIT==1)
     if (qnewbestfit) then ! new routine
      call RMSBestFit(r1,r2,ow,u) ! align r1 onto r2, BUT : transpose of u (below) aligns r2 onto r1
      r2=matmul(r2, u) ! apply rotation operation
     else
#endif
! standard CHARMM rotation routine
      call rotls1(r1(:,1),r1(:,2),r1(:,3),r2(:,1),r2(:,2),r2(:,3),&
& itotal, &
& RESHAPE( (/( i/2, i=2,2*itotal+1 )/) , (/2,itotal/) ), & ! index correspondence array : (1,1 ; 2,2 ; 3,3 ; ... ; itotal,itotal)
       itotal, ow,.false.,.false.)
!
#if (KEY_NEWBESTFIT==1)
     endif
#endif
!
     ipermute=0 ! permutation (offset) that gives the best alignment
     msd0=sum( (r1(itest+1:iorient,:)-r2(itest+1:iorient,:))**2 ) / (iorient - itest) ! initial msd
     msd1=msd0
!
! now cycle through all permutations to determine the optimum
!
     npermute=size(rr%orient_atom_permutations,1)
     ngroups =size(rr%orient_atom_permutations,2)
!
     do k=1, npermute-1 ! over possible permutations
      i=iorient_permute ! last index of orient group
      do l=1, ngroups ! over groups (counter not used)
       i=i+1 ! point to first index of permutation group
       rcom1=r1(i,:) ! save coordinates of first atom in group (reuse rcom triplet)
       do m=1, npermute-1 ! over atoms within groups (iterator not used)
        r1(i,:)=r1(i + 1,:) ! execute permutation
        i=i+1
       enddo ! atoms within group
       r1(i,:)=rcom1 ! complete cyclic permutation
!

      enddo ! groups
! compute new best fit; note that the COM does not change due to permutation
!


!
#if (KEY_NEWBESTFIT==1)
      if (qnewbestfit) then ! new routine
        call RMSBestFit(r1,r2,ow,u) ! align r1 onto r2, BUT : transpose of u (below) aligns r2 onto r1
        r2=matmul(r2, u) ! apply rotation operation
      else
#endif
! standard CHARMM rotation routine
       call rotls1(r1(:,1),r1(:,2),r1(:,3),r2(:,1),r2(:,2),r2(:,3),&
& itotal, &
& RESHAPE( (/( i/2, i=2,2*itotal+1 )/) , (/2,itotal/) ), & ! index correspondence array : (1,1 ; 2,2 ; 3,3 ; ... ; itotal,itotal)
         itotal, ow, .false., .false. )
!
#if (KEY_NEWBESTFIT==1)
      endif
#endif
!
      msd2=sum( (r1(itest+1:iorient,:)-r2(itest+1:iorient,:))**2 ) / ( iorient - itest )
      if (msd2.lt.msd1) then ; msd1=msd2 ; ipermute=k ; endif
!

     enddo ! permutations (k)
!
! decide whether to modify the coordinates
     if (ipermute.gt.0) then
      errnum = errnum + 1
!
      if (qfix) then
!
! loop over orientation groups
      i=iorient_permute+1
      do l=1, ngroups ! over groups (counter not used)
!
! copy to temp. array
       do m=0, npermute-1 ! over atoms within groups (counter not used)
        i0=i+m ; i1=i+mod( m + ipermute, npermute )
        r1(i0,:)=r_(ind(i1),:)
       enddo ! atoms within group
! copy temp to coordinate array
       do i0=i, i+npermute-1
        r_(ind(i0),:)=r1(i0,:)
       enddo ! atoms within group
! move on to the next group
       i=i+npermute
      enddo ! groups
!
! loop over other groups
!
      i=iother_permute+1
      do l=1, size(rr%other_atom_permutations,2) ! number of 'other' groups
!
! copy to temp. array
       do m=0, npermute-1 ! over atoms within groups (counter not used)
        i0=i+m ; i1=i+mod( m + ipermute, npermute )
        r1(i0,:)=r_(ind(i1),:)
       enddo ! atoms within group
! copy temp to coordinate array
       do i0=i, i+npermute-1
        r_(ind(i0),:)=r1(i0,:)
       enddo ! atoms within group
! move on to the next group
       i=i+npermute
      enddo ! groups
!
      endif ! qfix
     endif ! ipermute>0
!
     if (qprint) then
      if (ipermute.gt.0) then
       if (qfix) then
        write(info,669) whoami, ' RESIDUE: ',resid(ires)(1:len_trim(resid(ires))),' ',rr%resname(1:len_trim(rr%resname)),&
& ' ',segid(iseg)(1:len_trim(segid(iseg))),&
& sqrt(msd0), sqrt(msd1)
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       else
        write(info,670) whoami, ' RESIDUE: ',resid(ires)(1:len_trim(resid(ires))),' ',rr%resname(1:len_trim(rr%resname)),&
& ' ',segid(iseg)(1:len_trim(segid(iseg))),&
& sqrt(msd0), sqrt(msd1)
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif ! qfix
!
       if (ngroups.eq.1) then
        write(info,'(A,'//itoa(npermute)//'A)') '                  PERMUTATION GROUPS : ',  rr%orient_atom_permutations
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       elseif (ngroups.gt.1) then
        write(info,'(A,'//itoa(npermute)//'A,'//itoa(ngroups-1)//'(" / ",'//itoa(npermute)//'A))') '                  PERMUTATION GROUPS : ',  rr%orient_atom_permutations
        write(OUTU,'(A)') pack(info,info.ne.'');info='';
       endif
      else
! do not print this non-essential information
! write(info,670) whoami, ' RESIDUE: ',resid(ires)(1:len_trim(resid(ires))),' ',rr%resname(1:len_trim(rr%resname)),&
!& ' ',segid(iseg)(1:len_trim(segid(iseg))),&
!& sqrt(msd0), sqrt(msd1)
! write(OUTU,'(A)') pack(info,info.ne.'');info='';
      endif
     endif ! qprint
!
669 format(7A,' (initial RMSD) ', F8.5,'  (final) ',F8.5,' MODIFIED')
670 format(7A,' (initial RMSD) ', F8.5,'  (final) ',F8.5)
!
     deallocate(ind, r1, r2, ow)
!
     endif ! rule match
    enddo rules ! over rules
   endif ! flagged
  enddo ! over all residues
 enddo ! over all segments
!
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
 call set_param('CONFERR',errnum)
!
! print summary
!
 if (qprint) then
!
  if (errnum.gt.0) then
   write(info,'(2A)') whoami,' ________________________________________________________________________________'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
  endif
!
  if (qfix) then
   if (errnum.eq.1) then
    write(info,'(A,I5,A)') whoami, errnum, ' INCONSISTENCY WAS FOUND AND CORRECTED.'
   write(OUTU,'(A)') pack(info,info.ne.'');info='';
   elseif (errnum.gt.1) then
    write(info,'(A,I5,A)') whoami, errnum, ' INCONSISTENCIES WERE FOUND AND CORRECTED.'
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
  else
   if (errnum.eq.1) then
    write(info,'(A,I5,A)') whoami, errnum, ' INCONSISTENCY WAS FOUND.'
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   elseif (errnum.gt.1) then
    write(info,'(A,I5,A)') whoami, errnum, ' INCONSISTENCIES WERE FOUND.'
    write(OUTU,'(A)') pack(info,info.ne.'');info='';
   endif
  endif
 endif
!
 if(associated(r_))deallocate(r_)
 if(associated(rref_))deallocate(rref_)
 if(associated(islct))deallocate(islct)
!
end function confcons_check
!
!==============================================================
!
function ifindc(string1, atype, istart, iend)
  ! Victor Ovchinnikov 2008, MIT/Harvard
  ! find the atom index that corresponds to atom name; residue limits are passed in
  ! used by conformational_consistency subroutine
  use chm_kinds
!
  character(len=*), intent(in) :: string1, atype(*)
  integer, intent(in) :: istart, iend
  ! local
  integer :: l,ifindc
  logical :: found
  ! begin
  l=len_trim(string1)
  ifindc=istart-1
  found=.false.
  do while ( .not.found.and.(ifindc.lt.iend))
     ifindc=ifindc+1
     found=string1(1:l).eq.atype(ifindc)(1:len_trim(atype(ifindc)))
  enddo
  if (.not.found) ifindc=0
end function ifindc
!
!==============================================================
!
#endif /* automatically protect all code */
end module confcons
!
