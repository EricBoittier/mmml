module mmcifio_mod
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use chm_kinds
  use stream
  use string, only: encodi, trima
  use chutil, only: initia, matom, atomid
  use number
#ifdef KEY_RESIZE
  use resize, only: resize_psf, drsz0
#endif
  implicit none

  private
  public :: mmcif_read_sequence, mmcif_read_coor, mmcif_write_coor

  integer, parameter :: CIFTOK=256
  integer, parameter :: CIFLINE=4096

  type cif_stream
     integer :: unit = -1
     character(len=CIFLINE) :: line = ' '
     integer :: pos = 1
     integer :: llen = 0
     logical :: has_push = .false.
     character(len=CIFTOK) :: push = ' '
  end type cif_stream

  type mmcif_atom_row
     character(len=8) :: atom = ' '
     character(len=8) :: resn = ' '
     character(len=8) :: sid = ' '
     character(len=8) :: rid = ' '
     character(len=8) :: chain = ' '
     integer :: iseq = 0
     integer :: model = 1
     real(chm_real) :: x = 0.0_chm_real
     real(chm_real) :: y = 0.0_chm_real
     real(chm_real) :: z = 0.0_chm_real
     real(chm_real) :: w = 0.0_chm_real
     logical :: is_atom = .false.
     logical :: is_hetatm = .false.
     logical :: has_coordinates = .false.
  end type mmcif_atom_row

  type mmcif_seq_row
     character(len=8) :: resn = ' '
     character(len=8) :: sid = ' '
     character(len=8) :: rid = ' '
     character(len=8) :: chain = ' '
  end type mmcif_seq_row

contains

#ifdef KEY_RESIZE
  subroutine mmcif_read_sequence(iunit,istart,chain,segidx, &
#else
  subroutine mmcif_read_sequence(iunit,res,nres,resid,istart,chain,segidx, &
#endif
       nchain,nskip,skip,nali,alias,latom,lhetatm,lseqres,ifirst,nr0,sqsegid, &
       use_label)
#ifdef KEY_RESIZE
    use psf, only: nres, res, resid
#endif
    integer, intent(in) :: iunit,istart,nchain,nskip,nali,ifirst
#ifdef KEY_RESIZE
    integer, intent(inout) :: nr0
#else
    integer, intent(inout) :: nres,nr0
    character(len=*), intent(inout) :: res(*),resid(*)
#endif
    character(len=*), intent(in) :: chain,segidx,skip(*),alias(2,*)
    logical, intent(in) :: latom,lhetatm,lseqres,use_label
    character(len=*), intent(inout) :: sqsegid

    type(mmcif_atom_row), allocatable :: atoms(:)
    type(mmcif_seq_row), allocatable :: seqs(:)
    integer :: natom_rows,nseq_rows,i,dummy
    character(len=8) :: target_sid,last_sid,last_rid,last_resn,rid

    target_sid=' '
    last_sid=' '
    last_rid=' '
    last_resn=' '

    if (lseqres) then
       call read_scheme_rows(iunit,use_label,seqs,nseq_rows)
       if (nseq_rows > 0) then
          call select_target_seq(seqs,nseq_rows,chain,segidx,nchain,target_sid)
          do i=1,nseq_rows
             if (.not. sequence_filter(seqs(i)%sid,seqs(i)%chain,chain,segidx, &
                  nchain,target_sid)) cycle
             rid=seqs(i)%rid
             if (rid == ' ') call encodi(ifirst+nres-istart+1,rid,8,dummy)
#ifdef KEY_RESIZE
             call add_sequence_residue(seqs(i)%resn,rid,nr0)
#else
             call add_sequence_residue(res,nres,resid,seqs(i)%resn,rid,nr0)
#endif
             if (sqsegid == ' ') sqsegid=seqs(i)%sid
          enddo
          return
       endif
    endif

    call read_atom_rows(iunit,use_label,1,atoms,natom_rows)
    call select_target_atom(atoms,natom_rows,chain,segidx,nchain,target_sid)
    do i=1,natom_rows
       if (atoms(i)%is_atom .and. .not.latom) cycle
       if (atoms(i)%is_hetatm .and. .not.lhetatm) cycle
       if (.not.atoms(i)%is_atom .and. .not.atoms(i)%is_hetatm) cycle
       if (.not.sequence_filter(atoms(i)%sid,atoms(i)%chain,chain,segidx, &
            nchain,target_sid)) cycle
       if (atoms(i)%sid == last_sid .and. atoms(i)%rid == last_rid .and. &
            atoms(i)%resn == last_resn) cycle
#ifdef KEY_RESIZE
       call add_sequence_residue(atoms(i)%resn,atoms(i)%rid,nr0)
#else
       call add_sequence_residue(res,nres,resid,atoms(i)%resn,atoms(i)%rid,nr0)
#endif
       last_sid=atoms(i)%sid
       last_rid=atoms(i)%rid
       last_resn=atoms(i)%resn
       if (sqsegid == ' ') sqsegid=atoms(i)%sid
    enddo
  end subroutine mmcif_read_sequence

  subroutine mmcif_read_coor(iunit,x,y,z,wmain,natom,islct,ioffs,res,nres, &
       atype,ibase,segid,resid,nictot,nseg,lrsid,model,use_label)
    integer, intent(in) :: iunit,natom,ioffs,nres,nseg,model
    integer, intent(in) :: islct(*),ibase(*),nictot(*)
    real(chm_real), intent(inout) :: x(*),y(*),z(*),wmain(*)
    character(len=*), intent(in) :: res(*),atype(*),segid(*),resid(*)
    logical, intent(in) :: lrsid,use_label

    type(cif_stream) :: ts
    character(len=CIFTOK) :: tok
    logical :: eof
    integer :: i,iseg,ires,istp,ipoint,isres
    integer :: nmiss,nmult,nrng,nseqm,errcnt,ndesl
    character(len=8) :: sid,rid,resin,atomin

    isres=ioffs+1
    nmiss=0
    nmult=0
    nrng=0
    nseqm=0
    errcnt=0
    ndesl=0

    call init_stream(ts,iunit)
    do
       call next_token(ts,tok,eof)
       if (eof) exit
       if (upper_token(tok) == 'LOOP_') call read_atom_loop_coor(ts,use_label,max(1,model))
    enddo

    do i=1,natom
       if (islct(i) == 1) then
          if (.not.initia(i,x,y,z)) then
             nmiss=nmiss+1
             if (nmiss <= 10 .and. wrnlev >= 2) write(outu,65) i,atype(i)
65           format(' ** WARNING ** After reading mmCIF, there are no', &
                  ' coordinates for selected atom:',I8,1X,A)
          endif
       endif
    enddo

    if (nmiss > 0 .and. wrnlev >= 2) write(outu,74) nmiss
74  format(/' ** A total of',I6,' selected atoms have no coordinates')
    if (errcnt > 5 .and. wrnlev >= 2) write(outu,75) errcnt
75  format(/' ** A total of',I5,' warnings were encountered during', &
         ' mmCIF coordinate reading **')
    if (nmult > 0 .and. wrnlev >= 2) write(outu,55) nmult
55  format(/' ** WARNING ** Coordinates were overwritten for',I6,' atoms.')
    if (ndesl > 0 .and. wrnlev >= 2) write(outu,77) ndesl
77  format(/' ** MESSAGE **',I6,' atoms in mmCIF file were ignored', &
         ' because of the specified atom selection.')
    if (nrng > 0 .and. wrnlev >= 2) write(outu,78) nrng
78  format(/' ** MESSAGE **',I6,' atoms in mmCIF file were outside', &
         ' the specified sequence range.')
    if (nmiss+errcnt+nmult > 0) call diewrn(2)
    if (nseqm > 0 .and. wrnlev >= 2) then
       write(outu,79) nseqm
79     format(/' ** WARNING **',I6,' atoms in mmCIF file had a', &
            ' sequence mismatch.')
       call diewrn(0)
    endif

  contains

    subroutine read_atom_loop_coor(ts,use_label,wanted_model)
      type(cif_stream), intent(inout) :: ts
      logical, intent(in) :: use_label
      integer, intent(in) :: wanted_model
      character(len=CIFTOK), allocatable :: heads(:), vals(:)
      character(len=CIFTOK) :: tok
      logical :: eof,target,done,keep
      integer :: nhead,idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid
      integer :: idx_latom,idx_lresn,idx_lsid,idx_lrid,idx_x,idx_y,idx_z
      integer :: idx_b,idx_model,idx_ins
      type(mmcif_atom_row) :: row

      call read_loop_headers(ts,heads,nhead,tok,eof)
      if (eof .or. nhead == 0) return
      target=index(lower_token(heads(1)),'_atom_site.') == 1
      allocate(vals(nhead))
      if (.not.target) then
         call skip_loop(ts,tok,nhead)
         return
      endif

      idx_group=find_header(heads,nhead,'_atom_site.group_pdb')
      idx_id=find_header(heads,nhead,'_atom_site.id')
      idx_latom=find_header(heads,nhead,'_atom_site.label_atom_id')
      idx_lresn=find_header(heads,nhead,'_atom_site.label_comp_id')
      idx_lsid=find_header(heads,nhead,'_atom_site.label_asym_id')
      idx_lrid=find_header(heads,nhead,'_atom_site.label_seq_id')
      idx_atom=find_header(heads,nhead,'_atom_site.auth_atom_id')
      idx_resn=find_header(heads,nhead,'_atom_site.auth_comp_id')
      idx_sid=find_header(heads,nhead,'_atom_site.auth_asym_id')
      idx_rid=find_header(heads,nhead,'_atom_site.auth_seq_id')
      idx_x=find_header(heads,nhead,'_atom_site.cartn_x')
      idx_y=find_header(heads,nhead,'_atom_site.cartn_y')
      idx_z=find_header(heads,nhead,'_atom_site.cartn_z')
      idx_b=find_header(heads,nhead,'_atom_site.b_iso_or_equiv')
      idx_model=find_header(heads,nhead,'_atom_site.pdbx_pdb_model_num')
      idx_ins=find_header(heads,nhead,'_atom_site.pdbx_pdb_ins_code')
      if (use_label) then
         idx_atom=idx_latom
         idx_resn=idx_lresn
         idx_sid=idx_lsid
         idx_rid=idx_lrid
         idx_ins=0
      endif

      done=.false.
      do while (.not.done)
         call read_loop_row(ts,tok,nhead,vals,done,eof)
         if (done .or. eof) exit
         call make_atom_row(vals,idx_group,idx_id,idx_atom,idx_resn,idx_sid, &
              idx_rid,idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,wanted_model, &
              0,row,keep)
         if (keep) call handle_atom_row(row)
      enddo
    end subroutine read_atom_loop_coor

    subroutine handle_atom_row(row)
      type(mmcif_atom_row), intent(in) :: row

      if (.not.row%is_atom .and. .not.row%is_hetatm) return
      if (.not.row%has_coordinates) return
      sid=row%sid
      rid=row%rid
      resin=row%resn
      atomin=row%atom
       if (lrsid) then
          iseg=1
          do while (iseg <= nseg)
             if (segid(iseg) == sid) exit
             iseg=iseg+1
          enddo
          if (iseg > nseg) then
             ires=-99999999
          else
             ires=nictot(iseg)+1
             istp=nictot(iseg+1)
             do while (ires <= istp)
                if (resid(ires) == rid) exit
                ires=ires+1
             enddo
             if (ires > istp) ires=-99999999
          endif
       else
          ires=-99999999
          read(rid,*,err=10) ires
10        continue
          ires=ires+isres-1
       endif

       if (ires < 1 .or. ires > nres) then
          nrng=nrng+1
          if (nrng <= 5 .and. wrnlev >= 2) write(outu,82) sid,rid,resin,atomin
82        format(/' ** WARNING ** For atom in mmCIF file, could not find', &
               ' residue in PSF, and is thus ignored:',/ &
               /'  SEGID=',A,' RESID=',A,' RESNAME= ',A,' TYPE= ',A)
          if (nrng <= 5) call diewrn(1)
          return
       endif

       if (res(ires) /= resin) then
          nseqm=nseqm+1
          if (nseqm <= 5 .and. wrnlev >= 2) write(outu,85) ires,res(ires),resin
85        format(/' ** WARNING ** For atom in mmCIF file, the residue type', &
               ' does not match that (RESN) in the PSF:',I5,' PSF= ',A, &
               ' INPUT= ',A)
       endif

       ipoint=matom(ires,atomin,atype,ibase,ires,ires,.false.)
       if (ipoint < 0) then
          errcnt=errcnt+1
          if (errcnt < 20 .and. wrnlev >= 2) write(outu,45) row%iseq, &
               ires,resid(ires),resin,atomin
45        format(/' ** WARNING ** For atom in mmCIF file, the corresponding', &
               ' residue in the PSF lacks that atom:',/ &
               ' INDEX=',I8,' IRES=',I6,' RESID=',A,' RES=',A,' ATOM=',A)
       else if (ipoint < 1 .or. ipoint > natom) then
          nrng=nrng+1
       else if (islct(ipoint) == 1) then
          if (initia(ipoint,x,y,z)) nmult=nmult+1
          x(ipoint)=row%x
          y(ipoint)=row%y
          z(ipoint)=row%z
          wmain(ipoint)=row%w
       else
          ndesl=ndesl+1
       endif
    end subroutine handle_atom_row
  end subroutine mmcif_read_coor

  subroutine mmcif_write_coor(iunit,title,ntitl,x,y,z,wmain,res,atype,ibase, &
       nres,natom,islct,model)
    integer, intent(in) :: iunit,ntitl,nres,natom,model
    character(len=*), intent(in) :: title(*),res(*),atype(*)
    integer, intent(in) :: ibase(*),islct(*)
    real(chm_real), intent(in) :: x(*),y(*),z(*),wmain(*)
    integer :: ires,i,iseq,mdl
    character(len=8) :: sid,rid,ren,ac

    mdl=model
    if (mdl == 0) mdl=1
    write(iunit,'(A)') 'data_charmm'
    write(iunit,'(A)') '#'
    if (ntitl > 0) then
       write(iunit,'(A)') '_struct.title'
       write(iunit,'(A)') ';'
       do i=1,ntitl
          write(iunit,'(A)') trim(title(i))
       enddo
       write(iunit,'(A)') ';'
       write(iunit,'(A)') '#'
    endif
    write(iunit,'(A)') 'loop_'
    write(iunit,'(A)') '_atom_site.group_PDB'
    write(iunit,'(A)') '_atom_site.id'
    write(iunit,'(A)') '_atom_site.type_symbol'
    write(iunit,'(A)') '_atom_site.label_atom_id'
    write(iunit,'(A)') '_atom_site.label_alt_id'
    write(iunit,'(A)') '_atom_site.label_comp_id'
    write(iunit,'(A)') '_atom_site.label_asym_id'
    write(iunit,'(A)') '_atom_site.label_entity_id'
    write(iunit,'(A)') '_atom_site.label_seq_id'
    write(iunit,'(A)') '_atom_site.pdbx_PDB_ins_code'
    write(iunit,'(A)') '_atom_site.Cartn_x'
    write(iunit,'(A)') '_atom_site.Cartn_y'
    write(iunit,'(A)') '_atom_site.Cartn_z'
    write(iunit,'(A)') '_atom_site.occupancy'
    write(iunit,'(A)') '_atom_site.B_iso_or_equiv'
    write(iunit,'(A)') '_atom_site.pdbx_formal_charge'
    write(iunit,'(A)') '_atom_site.auth_seq_id'
    write(iunit,'(A)') '_atom_site.auth_comp_id'
    write(iunit,'(A)') '_atom_site.auth_asym_id'
    write(iunit,'(A)') '_atom_site.auth_atom_id'
    write(iunit,'(A)') '_atom_site.pdbx_PDB_model_num'
    iseq=0
    do ires=1,nres
       do i=ibase(ires)+1,ibase(ires+1)
          if (islct(i) /= 1) cycle
          iseq=iseq+1
          call atomid(i,sid,rid,ren,ac)
          write(iunit,100) 'ATOM',iseq,'?',cif_word(ac),'.',cif_word(ren), &
               cif_word(sid),'?',ires,'.',x(i),y(i),z(i),1.0_chm_real, &
               wmain(i),'?',cif_word(rid),cif_word(ren),cif_word(sid), &
               cif_word(ac),mdl
       enddo
    enddo
    write(iunit,'(A)') '#'
100 format(A,1X,I10,1X,A,1X,A,1X,A,1X,A,1X,A,1X,A,1X,I10,1X,A, &
         3(1X,F20.10),1X,F6.2,1X,F10.4,1X,A,1X,A,1X,A,1X,A,1X,A,1X,I10)
  end subroutine mmcif_write_coor

  subroutine read_atom_rows(iunit,use_label,model,rows,nrows)
    integer, intent(in) :: iunit,model
    logical, intent(in) :: use_label
    type(mmcif_atom_row), allocatable, intent(out) :: rows(:)
    integer, intent(out) :: nrows
    type(cif_stream) :: ts
    character(len=CIFTOK) :: tok
    logical :: eof

    allocate(rows(0))
    nrows=0
    call init_stream(ts,iunit)
    do
       call next_token(ts,tok,eof)
       if (eof) exit
       if (upper_token(tok) == 'LOOP_') call read_atom_loop(ts,use_label,model,rows,nrows)
    enddo
  end subroutine read_atom_rows

  subroutine read_scheme_rows(iunit,use_label,rows,nrows)
    integer, intent(in) :: iunit
    logical, intent(in) :: use_label
    type(mmcif_seq_row), allocatable, intent(out) :: rows(:)
    integer, intent(out) :: nrows
    type(cif_stream) :: ts
    character(len=CIFTOK) :: tok
    logical :: eof

    allocate(rows(0))
    nrows=0
    call init_stream(ts,iunit)
    do
       call next_token(ts,tok,eof)
       if (eof) exit
       if (upper_token(tok) == 'LOOP_') call read_scheme_loop(ts,use_label,rows,nrows)
    enddo
  end subroutine read_scheme_rows

  subroutine read_atom_loop(ts,use_label,model,rows,nrows)
    type(cif_stream), intent(inout) :: ts
    logical, intent(in) :: use_label
    integer, intent(in) :: model
    type(mmcif_atom_row), allocatable, intent(inout) :: rows(:)
    integer, intent(inout) :: nrows
    character(len=CIFTOK), allocatable :: heads(:), vals(:)
    character(len=CIFTOK) :: tok
    logical :: eof,target,done
    integer :: nhead,idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid
    integer :: idx_latom,idx_lresn,idx_lsid,idx_lrid,idx_x,idx_y,idx_z
    integer :: idx_b,idx_model,idx_ins

    call read_loop_headers(ts,heads,nhead,tok,eof)
    if (eof .or. nhead == 0) return
    target=index(lower_token(heads(1)),'_atom_site.') == 1
    allocate(vals(nhead))
    if (.not.target) then
       call skip_loop(ts,tok,nhead)
       return
    endif

    idx_group=find_header(heads,nhead,'_atom_site.group_pdb')
    idx_id=find_header(heads,nhead,'_atom_site.id')
    idx_latom=find_header(heads,nhead,'_atom_site.label_atom_id')
    idx_lresn=find_header(heads,nhead,'_atom_site.label_comp_id')
    idx_lsid=find_header(heads,nhead,'_atom_site.label_asym_id')
    idx_lrid=find_header(heads,nhead,'_atom_site.label_seq_id')
    idx_atom=find_header(heads,nhead,'_atom_site.auth_atom_id')
    idx_resn=find_header(heads,nhead,'_atom_site.auth_comp_id')
    idx_sid=find_header(heads,nhead,'_atom_site.auth_asym_id')
    idx_rid=find_header(heads,nhead,'_atom_site.auth_seq_id')
    idx_x=find_header(heads,nhead,'_atom_site.cartn_x')
    idx_y=find_header(heads,nhead,'_atom_site.cartn_y')
    idx_z=find_header(heads,nhead,'_atom_site.cartn_z')
    idx_b=find_header(heads,nhead,'_atom_site.b_iso_or_equiv')
    idx_model=find_header(heads,nhead,'_atom_site.pdbx_pdb_model_num')
    idx_ins=find_header(heads,nhead,'_atom_site.pdbx_pdb_ins_code')
    if (use_label) then
       idx_atom=idx_latom
       idx_resn=idx_lresn
       idx_sid=idx_lsid
       idx_rid=idx_lrid
       idx_ins=0
    endif

    done=.false.
    do while (.not.done)
       call read_loop_row(ts,tok,nhead,vals,done,eof)
       if (done .or. eof) exit
       call append_atom_row(rows,nrows,vals,idx_group,idx_id,idx_atom,idx_resn, &
            idx_sid,idx_rid,idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,model)
    enddo
  end subroutine read_atom_loop

  subroutine read_scheme_loop(ts,use_label,rows,nrows)
    type(cif_stream), intent(inout) :: ts
    logical, intent(in) :: use_label
    type(mmcif_seq_row), allocatable, intent(inout) :: rows(:)
    integer, intent(inout) :: nrows
    character(len=CIFTOK), allocatable :: heads(:), vals(:)
    character(len=CIFTOK) :: tok
    logical :: eof,target,done
    integer :: nhead,idx_sid,idx_chain,idx_resn,idx_rid,idx_ins

    call read_loop_headers(ts,heads,nhead,tok,eof)
    if (eof .or. nhead == 0) return
    target=index(lower_token(heads(1)),'_pdbx_poly_seq_scheme.') == 1
    allocate(vals(nhead))
    if (.not.target) then
       call skip_loop(ts,tok,nhead)
       return
    endif

    idx_chain=find_header(heads,nhead,'_pdbx_poly_seq_scheme.pdb_strand_id')
    idx_sid=find_header(heads,nhead,'_pdbx_poly_seq_scheme.asym_id')
    idx_ins=find_header(heads,nhead,'_pdbx_poly_seq_scheme.pdb_ins_code')
    if (use_label) then
       idx_resn=find_header(heads,nhead,'_pdbx_poly_seq_scheme.mon_id')
       idx_rid=find_header(heads,nhead,'_pdbx_poly_seq_scheme.seq_id')
       idx_ins=0
    else
       if (idx_chain /= 0) idx_sid=idx_chain
       idx_resn=find_header(heads,nhead,'_pdbx_poly_seq_scheme.auth_mon_id')
       if (idx_resn == 0) idx_resn=find_header(heads,nhead,'_pdbx_poly_seq_scheme.mon_id')
       idx_rid=find_header(heads,nhead,'_pdbx_poly_seq_scheme.auth_seq_num')
       if (idx_rid == 0) idx_rid=find_header(heads,nhead,'_pdbx_poly_seq_scheme.pdb_seq_num')
    endif

    done=.false.
    do while (.not.done)
       call read_loop_row(ts,tok,nhead,vals,done,eof)
       if (done .or. eof) exit
       call append_seq_row(rows,nrows,vals,idx_sid,idx_chain,idx_resn,idx_rid, &
            idx_ins)
    enddo
  end subroutine read_scheme_loop

  subroutine append_atom_row(rows,nrows,vals,idx_group,idx_id,idx_atom,idx_resn, &
       idx_sid,idx_rid,idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,wanted_model)
    type(mmcif_atom_row), allocatable, intent(inout) :: rows(:)
    integer, intent(inout) :: nrows
    character(len=*), intent(in) :: vals(:)
    integer, intent(in) :: idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid
    integer, intent(in) :: idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,wanted_model
    type(mmcif_atom_row) :: row
    logical :: keep

    call make_atom_row(vals,idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid, &
         idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,wanted_model,nrows+1,row,keep)
    if (.not.keep) return
    call grow_atom_rows(rows,nrows)
    rows(nrows)=row
  end subroutine append_atom_row

  subroutine make_atom_row(vals,idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid, &
       idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins,wanted_model,default_id,row,keep)
    character(len=*), intent(in) :: vals(:)
    integer, intent(in) :: idx_group,idx_id,idx_atom,idx_resn,idx_sid,idx_rid
    integer, intent(in) :: idx_x,idx_y,idx_z,idx_b,idx_model,idx_ins
    integer, intent(in) :: wanted_model,default_id
    type(mmcif_atom_row), intent(out) :: row
    logical, intent(out) :: keep
    character(len=8) :: ins
    character(len=CIFTOK) :: group

    keep=.false.
    row%atom=' '
    row%resn=' '
    row%sid=' '
    row%rid=' '
    row%chain=' '
    row%iseq=0
    row%model=1
    row%x=0.0_chm_real
    row%y=0.0_chm_real
    row%z=0.0_chm_real
    row%w=0.0_chm_real
    row%is_atom=.false.
    row%is_hetatm=.false.
    row%has_coordinates=.false.
    if (idx_atom == 0 .or. idx_resn == 0 .or. idx_sid == 0 .or. idx_rid == 0) return

    group=upper_token(value_at(vals,idx_group))
    row%is_atom=trim(group) == 'ATOM'
    row%is_hetatm=trim(group) == 'HETATM'
    if (.not.row%is_atom .and. .not.row%is_hetatm) return
    row%iseq=to_int(value_at(vals,idx_id),default_id)
    row%atom=charmm_word(value_at(vals,idx_atom))
    row%resn=charmm_word(value_at(vals,idx_resn))
    row%sid=charmm_word(value_at(vals,idx_sid))
    row%chain=row%sid
    row%rid=charmm_word(value_at(vals,idx_rid))
    ins=charmm_word(value_at(vals,idx_ins))
    if (ins /= ' ') row%rid=trim(row%rid)//trim(ins)
    row%has_coordinates=parse_real(value_at(vals,idx_x),row%x)
    if (row%has_coordinates) &
         row%has_coordinates=parse_real(value_at(vals,idx_y),row%y)
    if (row%has_coordinates) &
         row%has_coordinates=parse_real(value_at(vals,idx_z),row%z)
    row%w=to_real(value_at(vals,idx_b),0.0_chm_real)
    row%model=to_int(value_at(vals,idx_model),1)
    if (wanted_model > 0 .and. row%model /= wanted_model) return
    keep=.true.
  end subroutine make_atom_row

  subroutine append_seq_row(rows,nrows,vals,idx_sid,idx_chain,idx_resn,idx_rid, &
       idx_ins)
    type(mmcif_seq_row), allocatable, intent(inout) :: rows(:)
    integer, intent(inout) :: nrows
    character(len=*), intent(in) :: vals(:)
    integer, intent(in) :: idx_sid,idx_chain,idx_resn,idx_rid,idx_ins
    type(mmcif_seq_row) :: row
    character(len=8) :: ins

    if (idx_sid == 0 .or. idx_resn == 0 .or. idx_rid == 0) return
    row%sid=charmm_word(value_at(vals,idx_sid))
    row%chain=charmm_word(value_at(vals,idx_chain))
    row%resn=charmm_word(value_at(vals,idx_resn))
    row%rid=charmm_word(value_at(vals,idx_rid))
    ins=charmm_word(value_at(vals,idx_ins))
    if (ins /= ' ') row%rid=trim(row%rid)//trim(ins)
    call grow_seq_rows(rows,nrows)
    rows(nrows)=row
  end subroutine append_seq_row

  subroutine read_loop_headers(ts,heads,nhead,first,eof)
    type(cif_stream), intent(inout) :: ts
    character(len=CIFTOK), allocatable, intent(out) :: heads(:)
    integer, intent(out) :: nhead
    character(len=CIFTOK), intent(out) :: first
    logical, intent(out) :: eof
    character(len=CIFTOK) :: tok

    allocate(heads(0))
    nhead=0
    first=' '
    do
       call next_token(ts,tok,eof)
       if (eof) return
       if (tok == '#') cycle
       if (tok(1:1) == '_') then
          call grow_headers(heads,nhead)
          heads(nhead)=tok
       else
          first=tok
          return
       endif
    enddo
  end subroutine read_loop_headers

  subroutine read_loop_row(ts,first,nhead,vals,done,eof)
    type(cif_stream), intent(inout) :: ts
    character(len=CIFTOK), intent(inout) :: first
    integer, intent(in) :: nhead
    character(len=CIFTOK), intent(out) :: vals(nhead)
    logical, intent(out) :: done,eof
    integer :: i
    character(len=CIFTOK) :: tok,utok

    done=.false.
    eof=.false.
    tok=first
    if (tok == ' ') call next_token(ts,tok,eof)
    if (eof) then
       done=.true.
       return
    endif
    utok=upper_token(tok)
    if (tok == '#' .or. utok == 'LOOP_' .or. index(utok,'DATA_') == 1 .or. &
         tok(1:1) == '_') then
       if (tok /= '#') call push_token(ts,tok)
       done=.true.
       return
    endif
    vals(1)=tok
    do i=2,nhead
       call next_token(ts,tok,eof)
       if (eof .or. tok == '#') then
          done=.true.
          return
       endif
       vals(i)=tok
    enddo
    first=' '
  end subroutine read_loop_row

  subroutine skip_loop(ts,first,nhead)
    type(cif_stream), intent(inout) :: ts
    character(len=CIFTOK), intent(inout) :: first
    integer, intent(in) :: nhead
    character(len=CIFTOK), allocatable :: vals(:)
    logical :: done,eof
    allocate(vals(nhead))
    done=.false.
    do while (.not.done)
       call read_loop_row(ts,first,nhead,vals,done,eof)
       if (eof) exit
    enddo
  end subroutine skip_loop

  subroutine init_stream(ts,iunit)
    type(cif_stream), intent(out) :: ts
    integer, intent(in) :: iunit
    ts%unit=iunit
    ts%line=' '
    ts%pos=1
    ts%llen=0
    ts%has_push=.false.
    rewind(iunit)
  end subroutine init_stream

  subroutine next_token(ts,tok,eof)
    type(cif_stream), intent(inout) :: ts
    character(len=CIFTOK), intent(out) :: tok
    logical, intent(out) :: eof
    integer :: start
    character(len=1) :: quote

    tok=' '
    eof=.false.
    if (ts%has_push) then
       tok=ts%push
       ts%push=' '
       ts%has_push=.false.
       return
    endif

    do
       if (ts%pos > ts%llen) then
          read(ts%unit,'(A)',end=900,err=900) ts%line
          ts%llen=len_trim(ts%line)
          ts%pos=1
          if (ts%llen == 0) cycle
       endif
       do while (ts%pos <= ts%llen .and. ts%line(ts%pos:ts%pos) <= ' ')
          ts%pos=ts%pos+1
       enddo
       if (ts%pos > ts%llen) cycle
       if (ts%line(ts%pos:ts%pos) == '#') then
          ts%pos=ts%llen+1
          tok='#'
          return
       endif
       if (ts%pos == 1 .and. ts%line(1:1) == ';') then
          call read_multiline(ts,tok,eof)
          return
       endif
       quote=ts%line(ts%pos:ts%pos)
       if (quote == '"' .or. quote == "'") then
          ts%pos=ts%pos+1
          start=ts%pos
          do while (ts%pos <= ts%llen .and. ts%line(ts%pos:ts%pos) /= quote)
             ts%pos=ts%pos+1
          enddo
          tok=ts%line(start:min(ts%pos-1,start+CIFTOK-2))
          if (ts%pos <= ts%llen) ts%pos=ts%pos+1
          return
       endif
       start=ts%pos
       do while (ts%pos <= ts%llen .and. ts%line(ts%pos:ts%pos) > ' ' .and. &
            ts%line(ts%pos:ts%pos) /= '#')
          ts%pos=ts%pos+1
       enddo
       tok=ts%line(start:min(ts%pos-1,start+CIFTOK-2))
       return
    enddo
900 eof=.true.
  end subroutine next_token

  subroutine read_multiline(ts,tok,eof)
    type(cif_stream), intent(inout) :: ts
    character(len=CIFTOK), intent(out) :: tok
    logical, intent(out) :: eof
    tok=' '
    eof=.false.
    do
       read(ts%unit,'(A)',end=900,err=900) ts%line
       if (len_trim(ts%line) > 0 .and. ts%line(1:1) == ';') exit
    enddo
    ts%pos=1
    ts%llen=0
    return
900 eof=.true.
  end subroutine read_multiline

  subroutine push_token(ts,tok)
    type(cif_stream), intent(inout) :: ts
    character(len=*), intent(in) :: tok
    ts%push=tok
    ts%has_push=.true.
  end subroutine push_token

  function find_header(heads,nhead,name) result(idx)
    character(len=*), intent(in) :: heads(:),name
    integer, intent(in) :: nhead
    integer :: idx,i
    idx=0
    do i=1,nhead
       if (lower_token(heads(i)) == lower_token(name)) then
          idx=i
          return
       endif
    enddo
  end function find_header

  function value_at(vals,idx) result(val)
    character(len=*), intent(in) :: vals(:)
    integer, intent(in) :: idx
    character(len=CIFTOK) :: val
    if (idx <= 0 .or. idx > size(vals)) then
       val='?'
    else
       val=vals(idx)
    endif
  end function value_at

#ifdef KEY_RESIZE
  subroutine add_sequence_residue(resn,rid,nr0)
    use psf, only: nres, res, resid
#else
  subroutine add_sequence_residue(res,nres,resid,resn,rid,nr0)
    character(len=*), intent(inout) :: res(*),resid(*)
#endif
    character(len=*), intent(in) :: resn,rid
#ifdef KEY_RESIZE
    integer, intent(inout) :: nr0
#else
    integer, intent(inout) :: nres,nr0
#endif
    nres=nres+1
#ifdef KEY_RESIZE
    if (nres > nr0) then
       nr0=nres+drsz0
       call resize_psf('mmcifio.F90','MMCIF_SEQUENCE','NRES',nr0,.true.)
    endif
#endif
    res(nres)=resn
    resid(nres)=rid
  end subroutine add_sequence_residue

  subroutine select_target_atom(rows,nrows,chain,segidx,nchain,target_sid)
    type(mmcif_atom_row), intent(in) :: rows(:)
    integer, intent(in) :: nrows,nchain
    character(len=*), intent(in) :: chain,segidx
    character(len=8), intent(out) :: target_sid
    integer :: i,count
    target_sid=' '
    if (segidx /= ' ') then
       target_sid=charmm_word(segidx)
       return
    endif
    if (chain /= ' ') then
       target_sid=charmm_word(chain)
       return
    endif
    if (nchain <= 0) return
    count=0
    do i=1,nrows
       if (rows(i)%sid == ' ') cycle
       if (i == 1 .or. rows(i)%sid /= rows(max(1,i-1))%sid) then
          count=count+1
          if (count == nchain) then
             target_sid=rows(i)%sid
             return
          endif
       endif
    enddo
  end subroutine select_target_atom

  subroutine select_target_seq(rows,nrows,chain,segidx,nchain,target_sid)
    type(mmcif_seq_row), intent(in) :: rows(:)
    integer, intent(in) :: nrows,nchain
    character(len=*), intent(in) :: chain,segidx
    character(len=8), intent(out) :: target_sid
    integer :: i,count
    target_sid=' '
    if (segidx /= ' ') then
       target_sid=charmm_word(segidx)
       return
    endif
    if (chain /= ' ') then
       target_sid=charmm_word(chain)
       return
    endif
    if (nchain <= 0) return
    count=0
    do i=1,nrows
       if (rows(i)%sid == ' ') cycle
       if (i == 1 .or. rows(i)%sid /= rows(max(1,i-1))%sid) then
          count=count+1
          if (count == nchain) then
             target_sid=rows(i)%sid
             return
          endif
       endif
    enddo
  end subroutine select_target_seq

  logical function sequence_filter(sid,row_chain,chain,segidx,nchain,target_sid)
    character(len=*), intent(in) :: sid,row_chain,chain,segidx,target_sid
    integer, intent(in) :: nchain
    sequence_filter=.true.
    if (segidx /= ' ') sequence_filter=charmm_word(segidx) == sid
    if (chain /= ' ') sequence_filter=charmm_word(chain) == row_chain .or. &
         charmm_word(chain) == sid
    if (nchain > 0) sequence_filter=target_sid == sid
  end function sequence_filter

  subroutine grow_atom_rows(rows,nrows)
    type(mmcif_atom_row), allocatable, intent(inout) :: rows(:)
    type(mmcif_atom_row), allocatable :: tmp(:)
    integer, intent(inout) :: nrows
    integer :: old
    old=size(rows)
    if (nrows >= old) then
       allocate(tmp(max(16,old*2+1)))
       if (old > 0) tmp(1:old)=rows
       call move_alloc(tmp,rows)
    endif
    nrows=nrows+1
  end subroutine grow_atom_rows

  subroutine grow_seq_rows(rows,nrows)
    type(mmcif_seq_row), allocatable, intent(inout) :: rows(:)
    type(mmcif_seq_row), allocatable :: tmp(:)
    integer, intent(inout) :: nrows
    integer :: old
    old=size(rows)
    if (nrows >= old) then
       allocate(tmp(max(16,old*2+1)))
       if (old > 0) tmp(1:old)=rows
       call move_alloc(tmp,rows)
    endif
    nrows=nrows+1
  end subroutine grow_seq_rows

  subroutine grow_headers(heads,nhead)
    character(len=CIFTOK), allocatable, intent(inout) :: heads(:)
    character(len=CIFTOK), allocatable :: tmp(:)
    integer, intent(inout) :: nhead
    integer :: old
    old=size(heads)
    if (nhead >= old) then
       allocate(tmp(max(16,old*2+1)))
       if (old > 0) tmp(1:old)=heads
       call move_alloc(tmp,heads)
    endif
    nhead=nhead+1
  end subroutine grow_headers

  function charmm_word(tok) result(out)
    character(len=*), intent(in) :: tok
    character(len=8) :: out
    character(len=CIFTOK) :: tmp
    integer :: l
    tmp=adjustl(tok)
    if (tmp == '?' .or. tmp == '.') then
       out=' '
       return
    endif
    tmp=upper_token(tmp)
    out=' '
    l=min(len_trim(tmp),8)
    if (l > 0) out(1:l)=tmp(1:l)
  end function charmm_word

  function cif_word(tok) result(out)
    character(len=*), intent(in) :: tok
    character(len=32) :: out
    character(len=32) :: tmp
    tmp=adjustl(tok)
    if (len_trim(tmp) == 0) then
       out='?'
    else
       out=trim(tmp)
    endif
  end function cif_word

  function upper_token(tok) result(out)
    character(len=*), intent(in) :: tok
    character(len=len(tok)) :: out
    integer :: i,ia
    out=tok
    do i=1,len_trim(out)
       ia=iachar(out(i:i))
       if (ia >= iachar('a') .and. ia <= iachar('z')) out(i:i)=achar(ia-32)
    enddo
  end function upper_token

  function lower_token(tok) result(out)
    character(len=*), intent(in) :: tok
    character(len=len(tok)) :: out
    integer :: i,ia
    out=tok
    do i=1,len_trim(out)
       ia=iachar(out(i:i))
       if (ia >= iachar('A') .and. ia <= iachar('Z')) out(i:i)=achar(ia+32)
    enddo
  end function lower_token

  integer function to_int(tok,default)
    character(len=*), intent(in) :: tok
    integer, intent(in) :: default
    to_int=default
    if (tok == '?' .or. tok == '.') return
    read(tok,*,err=10) to_int
10  continue
  end function to_int

  real(chm_real) function to_real(tok,default)
    character(len=*), intent(in) :: tok
    real(chm_real), intent(in) :: default
    to_real=default
    if (tok == '?' .or. tok == '.') return
    read(tok,*,err=10) to_real
10  continue
  end function to_real

  logical function parse_real(tok,value)
    character(len=*), intent(in) :: tok
    real(chm_real), intent(out) :: value
    integer :: io_status
    value=0.0_chm_real
    parse_real=.false.
    if (tok == '?' .or. tok == '.') return
    read(tok,*,iostat=io_status) value
    parse_real=io_status == 0 .and. ieee_is_finite(value)
  end function parse_real

end module mmcifio_mod
