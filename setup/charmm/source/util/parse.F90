!> @file parse.F90
!> @brief CHARMM command-line tokenizer and parameter substitution.
!>
!> Provides the routines that turn one logical CHARMM command (read
!> from a stream, possibly spanning multiple lines via hyphen
!> continuation) into a single uppercased command line, with @-style
!> parameters and ?-style energy/value tokens substituted in.
!>
!> Public routines:
!>   - rdcmnd   read and assemble one command
!>   - subenr   substitute one ?name energy/value token
!>   - xtrane   warn about extraneous characters in a residual string
!>
!> Private helper:
!>   - eof_cleanup   shared EOF-handling tail used by rdcmnd

!> @brief Read one logical CHARMM command from a unit.
!>
!> Reads input lines from @c unit and assembles them into a single
!> command stored in @c comlyn / @c comlen. Multiple input lines are
!> joined into one command when the trailing non-blank character is a
!> hyphen (continuation). Comments (everything from a `!` to end-of-
!> line) are stripped. Lowercase letters are converted to uppercase.
!>
!> Output also has @-style parameters expanded (via @c parse1) and
!> ?-style energy/value tokens substituted (via @c subenr).
!>
!> @c qprint controls whether read records are echoed to @c outu, with
!> each echoed line prefixed by @c echost so it is easy to locate when
!> scanning the output.
!>
!> @c qcmprs controls whitespace compression (CHARMM-standard parsing).
!> On parallel builds it also gates broadcasting of the assembled
!> command to peer ranks.
!>
!> Special cases:
!>   - If @c eof is .true. on entry, returns immediately.
!>   - If @c unit is negative, returns immediately with @c eof set.
!>   - For G94INP> input, the hyphen-continuation mechanism is
!>     disabled so Gaussian input files pass through unchanged.
!>
!> @param[inout] comlyn  Assembled command line (uppercased, trimmed)
!> @param[in]    mxcms2  Maximum length of comlyn buffer
!> @param[out]   comlen  Length of the assembled command (in chars)
!> @param[in]    unit    Fortran unit number to read from
!> @param[inout] eof     End-of-file flag; set to .true. when the
!>                       input stream is exhausted
!> @param[in]    qcmprs  If .true., compress whitespace and broadcast
!> @param[in]    qprint  If .true., echo each read record to outu
!> @param[in]    echost  Echo prefix string for the output log
!>
!> @author Robert Bruccoleri, David States, Bernard Brooks, Youngdo Won
!> @author Victor Anisimov  (2004) — disable HYPHEN for G94INP input
!> @author Rick Lapp        (1999) — disable case conversion for CFF
subroutine rdcmnd(comlyn,mxcms2,comlen,unit,eof, &
     qcmprs,qprint,echost)
#if KEY_REPDSTR==1
  use repdstrmod
#endif
  use chm_kinds
  use dimens_fcm
  use exfunc
  use cmdpar,only:parse1
  use string
  use stream
  use parallel
  use repdstr
#if KEY_PARALLEL==1
  use mpi_f08
#endif
  ! VO begin
#if KEY_MULTICOM==1
  use mpi_f08
  use multicom_aux
  use ifstack
#endif
  ! VO end
#if KEY_ENSEMBLE==1
  use ensemble,only:ensprint
#endif
#if KEY_CFF==1
  use rtf,only:ucase
#endif
  implicit none
  !
  integer mxcms2,comlen,unit
  character(len=*) comlyn
  !
  !
  integer enrlen
  integer,parameter :: enrmax=20, mxcard=200
  character(len=enrmax) enrst
  integer cardln
  character(len=mxcard) card
  integer wdlen,iend,ipar,j
  logical eof,qprint,qcmprs,usehyp
  character(len=*) echost
  character(len=80) chint
  integer reason       ! iostat from card-reading reads (used by both
                       ! REPDSTR and non-REPDSTR branches)
#if KEY_REPDSTR==1
  logical repd_print
#endif
  logical qmonoscript
  character(len=1) :: hyphen='-', exclmk='!', atmark='@', sdblq='"', &
       sques='?'
  !
#if KEY_ENSEMBLE==1
  write(chint,*)qcmprs,iolev
#endif
#if KEY_ENSEMBLE==1
  call ensprint("RDCMND>>> qcmprs,iolev",chint)
#endif
#if KEY_PARALLEL==1
  if (.not.qrepdstr) then
     if(.not.qcmprs .and. iolev < 0) then
        comlen=0
        return
     endif
  endif
#endif
  if(eof) return
  if(index(echost,'G94INP>') > 0) then
     ! G94 input file case: do not use HYPHEN mechanism
     usehyp=.false.
  else
     ! Standard case: use HYPEN mechanism
     usehyp=.true.
  endif
# if KEY_REPDSTR==1
  if (repd_inside_if_block > 0 .and. repd_if_block_stat(repd_inside_if_block) < 1) then
    ! Inside if block but not executing. Be silent
    repd_print = .false.
  else
    repd_print = qprint
  endif
# endif
  !--- Clean up the previous command line in case of something left
  comlyn=' '
  comlen=0
  if (unit < 0) then
     call eof_cleanup(eof, card, cardln, qcmprs)
     return
  end if
# if KEY_REPDSTR==1
  if (repd_print .and. prnlev >= 3) write(outu, '(2x)')
# else
  if (qprint .and. prnlev >= 3) write(outu, '(2x)')
# endif

read_card: do
  !
  card(1:mxcard)=' '
  reason=0
  !     In case of distributed I/O we read from "every" node
#if KEY_REPDSTR==1
  ! Possible cases for reading input in parallel:
  !   1. If no repdstr, only threads with iolev > 0 read.
  !   2. If using repdstr and we are reading main input,
  !      only the world master reads.
  !   3. If using repdstr and we are not reading from stdin
  !      (i.e. a stream or another file) all comm_charmm masters need to read.
  ! qmonoscript will be true if only the world master needs to read
  ! and broadcast.
  qmonoscript = .false.
  if (qrepdstr) then
    qmonoscript = .true.
    if (qrdqtt .or. unit /= 5) qmonoscript = .false.
  endif

  !! The next two lines completely break subcommand parsing. I have
  !! no idea why Milan did this -- Tim Miller

  qmonoscript=qmonoscript.and.(echost == 'CHARMM>' .or. echost == 'MSCALE> ' &
              .or. echost == 'BLOCK> ')
  if (qmonoscript) then
     ! one input CHARMM script only for all independent groups:
     ! each mynod=0 has iolev>0, so we need the test on mynodg here!
     if(mynodg == 0) read(unit,'(a)',iostat=reason) card
  else
     ! we are in the streams or no QREPD: each mynod=0 has to do it:
     if(iolev > 0) read(unit,'(a)',iostat=reason) card
  endif
  ! If anyone encountered an error bring the whole thing down TODO enable this?
  !if (reason.gt.0) then
  !  call wrndie(-5,'<RDCMND>','Error encountered during input read.')
  !endif
#else /**/
#if KEY_MULTICOM==1 /*  VO string v */
  if (mpi_comm_parser == mpi_comm_null) return ! should not be necessary, but added for clarity
  if (me_parser == 0 .or. iolev > 0) then      ! can do without this by setting iolev>=0 where MPI_PARSER=0, but kept for clarity
#else /*  VO string end ^ */
  if(iolev > 0) then
#endif
   ! Use iostat-based EOF detection (matching the REPDSTR branch above).
   ! With `end=9`, gfortran refuses subsequent reads on a unit already
   ! past EOF and aborts with "Sequential READ or WRITE not allowed
   ! after EOF marker". This bites pycharmm's eval_charmm_script,
   ! which can call rdcmnd repeatedly on a SCRATCH unit after the
   ! script's last line has been consumed.
   read(unit,'(a)',iostat=reason) card
   if (reason /= 0) then
      call eof_cleanup(eof, card, cardln, qcmprs)
      return
   end if
  endif
#endif /* KEY_REPDSTR */
  cardln=len_trim(card)

# if KEY_REPDSTR==1
  ! NOTE: If not qmonoscript, the broadcast to comm_charmm is currently
  !       handled by the call to psndc below.
  if (qmonoscript) then
    ! Only the world master read input. Broadcast io result.
    if (qrepmaster) then
      call mpi_bcast(reason, 1,         mpi_integer, 0, comm_rep_master, j)
      call mpi_bcast(card,   len(card), mpi_char,    0, comm_rep_master, j)
    endif
  endif
# endif

#if KEY_PARALLEL==1
  if(qcmprs) then
#if KEY_ENSEMBLE==1
     call ensprint("RDCMND broadcast"," ")
#endif
#    if KEY_REPDSTR==1
     ! Send the read result to the rest of comm_charmm to see if EOF was reached
     call mpi_bcast(reason, 1, mpi_integer, 0, comm_charmm, j)
     if (reason /= 0) then
       ! EOF encountered during read.
       card='END-OF-FILE'
       cardln = len_trim(card)
       eof=.true.
       return
     endif
#    endif
     ! VO string v
#if KEY_MULTICOM==1
#    if KEY_REPDSTR==1
     ! Under repdstr, distribute the assembled command within the local
     ! replica group (comm_charmm) rather than over the global stringm
     ! parser communicator (mpi_comm_parser).
     !
     ! mpi_comm_parser spans every replica, so broadcasting the command on
     ! it couples all replicas into one collective for each read.  That is
     ! fatal to a per-replica stream excursion such as
     !     if ?myrep .eq. 0 stream <file>
     ! where only some replicas descend into a nested stream: the replica(s)
     ! inside the stream issue their excursion reads on mpi_comm_parser while
     ! the replicas that stayed on the main input issue their next-command
     ! reads on the same communicator.  The mismatched broadcasts overwrite
     ! the command buffer on the replicas that stayed behind (observed as a
     ! garbled command like "C" -> "Unrecognized command").
     !
     ! Keeping the distribution replica-local matches the non-stringm
     ! (KEY_MULTICOM==0) build, which handles this case correctly, and is
     ! the natural choice for repdstr where each replica is an independent
     ! parser group.  Cross-replica delivery of shared main-input commands
     ! is already handled by the comm_rep_master broadcast above (the
     ! qmonoscript path).
     if (qrepdstr) then
        call mpi_bcast(card,mxcard,mpi_byte,0,comm_charmm,j)
     else
        call mpi_bcast(card,mxcard,mpi_byte,0,mpi_comm_parser,j)
     endif
#    else /* KEY_REPDSTR */
     call mpi_bcast(card,mxcard,mpi_byte,0,mpi_comm_parser,j)
#    endif /* KEY_REPDSTR */
#else
     call psndc(card,1)
#endif
     ! VO string ^
#if KEY_ENSEMBLE==1
     call ensprint("RDCMND command",card)
#endif
     if(card == 'END-OF-FILE') then
        eof=.true.
        return
     endif
# if KEY_REPDSTR
  else if (.not. qrepdstr .and. reason /= 0) then ! qcmprs
    ! No REPD and EOF/error - bail out.
    call eof_cleanup(eof, card, cardln, qcmprs)
    return
# endif /* KEY_REPDSTR */
  endif ! qcmprs
#endif

  cardln=mxcard
  call trime(card,cardln)
  if(cardln == 0) cardln=1
  if(qprint.and.prnlev >= 3 &
#if KEY_MULTICOM==1 /*  VO stringm : conditional evaluation in parallel */
 &          .and. peek_if() &
#endif
#if KEY_REPDSTR==1
 &          .and. repd_print &
#endif
 &                          ) write(outu, '(1x,a8,3x,a)') echost, card(1:cardln)

  iend = index(card(1:cardln), exclmk(1:1))
  if (iend == 1) cycle read_card        ! whole line is a comment
  if (iend /= 0) then
     cardln=iend-1
     call trime(card,cardln)
  endif

#if KEY_CFF==1
  if (ucase) &
#endif
       call cnvtuc(card,cardln)
  if(qcmprs) call cmprst(card,cardln)
  if (cardln == 0) exit read_card       ! empty after compress; ADDST below
  if(card(cardln:cardln) == hyphen.and.usehyp) then
     if (cardln == 1) cycle read_card   ! only a hyphen; get next line
     if(comlen+cardln-1 > mxcms2) then
        call wrndie(-1,'<RDCMND>','Command line too long: truncated.')
     endif
     call addst(comlyn,mxcms2,comlen,card,cardln-1)
     cycle read_card                    ! continuation; get next line
  endif
  exit read_card                        ! complete command, no continuation
end do read_card
  !
  if(comlen+cardln > mxcms2) then
     call wrndie(-1,'<RDCMND>','Command line too long: truncated.')
  endif
  call addst(comlyn,mxcms2,comlen,card,cardln)
  !
  !     Before returning the string make any parameter
  !     substitutions that may be required.
  !
  if(qcmprs) ffour=comlyn(1:4)
# if KEY_REPDSTR==1
  if (repd_inside_if_block > 0 .and. &
      repd_if_block_stat(repd_inside_if_block) < 1) then
     ! Inside a REPD if-block with this branch disabled — skip parsing
     ! and substitutions, go straight to cleanup.
     call trima(comlyn, comlen)
     return
  endif
# endif /* KEY_REPDSTR */

#if KEY_MULTICOM==1 /* VO stringm conditional execution in parallel */
  if (peek_if()) &
#endif
  call parse1(comlyn,mxcms2,comlen,qprint)
  if(comlen == 0)return
  !
  !     Before returning the string make any energy
  !     substitutions that may be required.
  !
  ipar = index(comlyn(1:(comlen - 1)), sques(1:1))
#if KEY_MULTICOM==1 /*  VO stringm : conditional evaluation in parallel */
  ! Skip the substitution loop when this rank is sitting out the conditional.
  if (.not. peek_if()) ipar = 0
#endif
  do while (ipar > 0)
     call copsub(enrst,enrmax,enrlen,comlyn,ipar+1, &
          min(comlen,enrmax+ipar))
     call subenr(wdlen,enrlen,enrst,enrmax)
     if (enrlen > 0) then
        if (qprint .and. prnlev >= 3) then
           write(outu, "(' RDCMND substituted energy or value ""',80A1)") &
                (comlyn(j:j),j=ipar,ipar+wdlen),sdblq, &
                ' ','t','o',' ',sdblq,(enrst(j:j),j=1,enrlen),sdblq
        endif
        call copsub(scrtch,scrmax,scrlen,comlyn,ipar+wdlen+1,comlen)
        comlen=ipar-1
        call addst(comlyn,mxcms2,comlen,enrst,enrlen)
        call addst(comlyn,mxcms2,comlen,scrtch,scrlen)
        ipar = index(comlyn(1:(comlen - 1)), sques(1:1))
     else
        !  WE WANT THE WHOLE PARAMETER NAME
        if(wrnlev >= 2) write(outu, "(' RDCMND: can not substitute energy ""',80A1)") &
             (comlyn(j:j),j=ipar,ipar+wdlen),sdblq
        ipar=0
     endif
  end do
  call trima(comlyn,comlen)
  return
end subroutine rdcmnd

!> @brief Shared EOF-handling tail for rdcmnd.
!>
!> Called from each of the three places in @c rdcmnd that detect
!> end-of-input (a negative unit, a non-zero iostat from the card read,
!> or the REPDSTR no-broadcast bail-out). On parallel/qcmprs builds the
!> string @c 'END-OF-FILE' is broadcast to peer ranks via @c psndc or
!> @c mpi_bcast so they can also exit the read loop.
!>
!> @param[inout] eof     Set to .true. on return
!> @param[inout] card    If broadcasting, set to 'END-OF-FILE' (for the
!>                       transmitted message). Otherwise left as-is.
!> @param[out]   cardln  Trimmed length of @c card (only meaningful if
!>                       qcmprs and KEY_PARALLEL=1).
!> @param[in]    qcmprs  Input flag from rdcmnd; gates broadcasting
subroutine eof_cleanup(eof, card, cardln, qcmprs)
#if KEY_PARALLEL==1
#if KEY_REPDSTR==1
  use repdstr,    only: qrepdstr, qrdqtt, psetglob, psetloc
#endif
#if KEY_MULTICOM==1
  use mpi_f08,         only: mpi_bcast, mpi_byte
  use multicom_aux,    only: mpi_comm_parser
#endif
#if KEY_ENSEMBLE==1
  use ensemble,   only: ensprint
#endif
#endif
  implicit none
  logical,            intent(inout) :: eof
  character(len=*),   intent(inout) :: card
  integer,            intent(out)   :: cardln
  logical,            intent(in)    :: qcmprs
#if KEY_PARALLEL==1 && KEY_MULTICOM==1
  integer :: ierr
#endif

  eof = .true.
#if KEY_PARALLEL==1
  if (qcmprs) then
     card = 'END-OF-FILE'
#if KEY_REPDSTR==1
     if (qrepdstr .and. .not. qrdqtt) call psetglob
#endif
     cardln = len_trim(card)
#if KEY_ENSEMBLE==1
     call ensprint("RDCMND broadcast", " ")
#endif
#if KEY_MULTICOM==0
     call psndc(card, 1)
#endif
#if KEY_MULTICOM==1
     call mpi_bcast(card, len(card), mpi_byte, 0, mpi_comm_parser, ierr)
#endif
#if KEY_ENSEMBLE==1
     call ensprint("RDCMND command", card)
#endif
#if KEY_REPDSTR==1
     if (qrepdstr .and. .not. qrdqtt) call psetloc
#endif
  end if
#endif
end subroutine eof_cleanup

!> @brief Substitute one ?-style energy/value token in a CHARMM command.
!>
!> Looks at the first whitespace-delimited word in @c enrst (parsed via
!> @c nexta8) and produces a textual replacement when the word matches:
!>
!>   - an energy property  (CEPROP[i]  → EPROP[i])
!>   - an energy term      (CETERM[i]  → ETERM[i])
!>   - a virial component  (CEPRSS[i]  → EPRESS[i])
!>   - 'RAND' / 'RANDOM'   → next call to @c ranumb()
!>   - 'ISEE' / 'ISEED'    → current @c irndsd integer
!>   - any miscellaneous parameter registered via @c find_param,
!>     attempted in the order: real, integer, character.
!>
!> If no match is found, @c enrlen is left at 0 to signal failure to
!> the caller (rdcmnd's substitution loop, which then prints a warning).
!>
!> Intended to remain internal to the parser; not part of the public
!> CHARMM API.
!>
!> @param[out]   wdlen   Length of the recognized substitution token in
!>                       @c enrst (input form), used by the caller to
!>                       know how many characters to overwrite.
!> @param[inout] enrlen  On entry: length of the input candidate string.
!>                       On exit: length of the substituted text, or 0
!>                       if no substitution was produced.
!> @param[inout] enrst   On entry: candidate token. On exit: substituted
!>                       text (when a match is found).
!> @param[in]    enrmax  Maximum length of @c enrst.
!>
!> @author Bernard R. Brooks (1983)
subroutine subenr(wdlen,enrlen,enrst,enrmax)
  use chm_kinds
  use exfunc
  use energym
  use param_store
  use rndnum
  use clcg_mod,only:ranumb
  use string
  !
  implicit none
  !
  integer wdlen,enrlen,enrmax
  character(len=*) enrst
  character(len=8) wrd
  integer i
  !
  real(chm_real) r
  integer ival
  logical :: found
  found = .false.
  !
  wrd=nexta8(enrst,enrlen)
  if(wrd == ' ') return
  wdlen=8
  call trime(wrd,wdlen)
  !
  !     Do energy value substitutions
  enrlen=0
  do i=1,lenenp
     if(wrd == ceprop(i)) then
        r=eprop(i)
        call encodf(r,enrst,enrmax,enrlen)
        return
     endif
  enddo
  do i=1,lenent
     if(wrd == ceterm(i)) then
        r=eterm(i)
        call encodf(r,enrst,enrmax,enrlen)
        return
     endif
  enddo
  do i=1,lenenv
     if(wrd == ceprss(i)) then
        r=epress(i)
        call encodf(r,enrst,enrmax,enrlen)
        return
     endif
  enddo

  !     Do random number substitution
  if(wrd == 'RAND' .or. wrd == 'RANDOM') then
     r=ranumb()
     call encodf(r,enrst,enrmax,enrlen)
     return
  endif

  !     Do random number substitution
  if(wrd == 'ISEE' .or. wrd == 'ISEED') then
     ival = irndsd
     call encodi(ival,enrst,enrmax,enrlen)
     return
  endif

  !     Do miscellaneous real substitutions
  call find_param(wrd, r, found)
  if (found) then
    call encodf(r,enrst,enrmax,enrlen)
    return
  end if

  !     Do miscellaneous integer substitutions
  call find_param(wrd, ival, found)
  if (found) then
    call encodi(ival, enrst, enrmax, enrlen)
    return
  end if

  !     Do miscellaneous character substitutions
  call find_param(wrd, enrst, found)
  if (found) then
    enrlen = 8
    call trime(enrst, enrlen)
    return
  end if

  return
end subroutine subenr

!> @brief Warn if a residual command string contains extraneous text.
!>
!> After a CHARMM command parser has consumed the tokens it understood,
!> any remaining non-blank content in @c st is an indication of a typo
!> or unrecognized option. This routine trims @c st, and if anything
!> non-blank remains it prints a warning identifying which command
!> the leftover came from (via @c idst). On exit @c stlen is reset to
!> 0 so the same residue is not re-reported by a later caller.
!>
!> The warning is gated on @c wrnlev >= 2 and @c prnlev >= 2.
!>
!> @param[inout] st     Command-line residue to inspect; reset to "" on
!>                      exit if anything was reported.
!> @param[inout] stlen  Length of @c st on entry; 0 on exit.
!> @param[in]    idst   Identifier for the calling command, used in the
!>                      warning message.
!>
!> @author Robert Bruccoleri
subroutine xtrane(st,stlen,idst)
  use chm_kinds
  use stream
  use string
  implicit none
  character(len=*) st
  integer stlen
  character(len=*) idst
  !
  call trima(st,stlen)
  if(stlen > 0) then
     if(wrnlev >= 2) then
        if (prnlev >= 2) write(outu, &
             '(" **** Warning ****  The following extraneous characters",&
             &/," were found while command processing in ",A)') idst
        if (prnlev >= 2) call prntst(outu,st,stlen,1,80)
     endif
     stlen=0
  endif
  !
  return
end subroutine xtrane
