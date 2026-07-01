module calc
  use chm_kinds
  implicit none

contains
  SUBROUTINE CALCULATOR(value,exprss)
    !-----------------------------------------------------------------------
    !     Intepretor of arithmetic expression
    !     Author: Benoit Roux.
    !     Evaluate the algebraic numerical expression "exprss(1:last)"
    !     This function supports all mathematical numerical expression
    !     with nesting of parentheses, e.g., exp(1.0-cos(2*(log(2*pi))**2)/.5) 
    !     The parsing is actually very crude since the expression "exprss" is
    !     translated back and forth between character string and a real variable.
    !     But it works!
    ! Changed internal variables to real(chm_real), to allow for large frame counts
    ! November 2007, L. Nilsson
    !
  use stream

    ! Passed variables
    real(chm_real) value
    character(len=*) exprss
    ! Local variables
    character(len=80)  exprss2
    real(chm_real) tempo
    logical done
    integer i,j,last,i1,i2,last2

    j=1
    do while(j > 0)
       i=index(exprss,' ')
       call trim2(exprss(i:),j)
    enddo

    last=index(exprss,' ')-1
    IF(PRNLEV > 3) write(outu,'(2a)') 'Evaluating: ', exprss(1:last)

    ! Check for simple expression to evaluate between parenthesis
    done=.false.
    do while (.not. done)
       i1=1
       done=.true.

       loop1: do i=i1,last
          if(exprss(i:i) == '(') i1=i
          if(exprss(i:i) == ')')then
             done=.false.
             i2=i
             tempo=simple(exprss,i1+1,i2-1)
             !
             ! Check to see if the parenthesis belong to a known function
             call chkfnct(exprss,i1,tempo)

             ! Finally, insert the result inside the string and clean it
             write(exprss2,'(e25.15)') tempo
             call trim2(exprss2,last2)
             exprss2(last2+1:)=exprss(i2+1:)
             exprss(i1:)=exprss2(1:)
             call trim2(exprss,last)
             exit loop1
          endif
       enddo loop1
    enddo

    value=simple(exprss,1,last)
    exprss=' '

    return
  end SUBROUTINE CALCULATOR

  subroutine trim2(c1,last)
    !-----------------------------------------------------------------------
    ! This routine compresses a string by removing all blanks. 

    character(len=*) c1
    integer i,last,lenc1
    !
    lenc1=len(c1)
    do i=1,lenc1
       if(c1(i:i) /= ' ')then
          c1(1:lenc1)=c1(i:lenc1)
          exit
       endif
    end do
    last=lenc1
    do i=lenc1,1,-1
       if(c1(i:i) == ' ') last=i-1
    enddo
    return
  end subroutine trim2

  subroutine chkfnct(exprss,i1,tempo)
    !-----------------------------------------------------------------------
    ! Check to see if the parenthesis belong to a known function

    character(len=*) exprss
    integer i1
    real(chm_real) tempo
    logical still_checking

    ! Five letters functions
    still_checking=.true.
    select case (i1)
    case (6:)
       if( exprss(i1-5:i1) == 'LOG10(' )then
          tempo=log10(tempo)
          i1=i1-5
          still_checking=.false.
       else
          if(still_checking)call check4
          if(still_checking)call check3
          if(still_checking)call check2
       endif

    case (5)
       if(still_checking)call check4
       if(still_checking)call check3
       if(still_checking)call check2

    case (4)
       if(still_checking)call check3
       if(still_checking)call check2

    case (3)
       if(still_checking)call check2

    end select
    return

    contains
      subroutine check4
        ! Four letters functions
        if( exprss(i1-4:i1) == 'ASIN(')then
           tempo=asin(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'ACOS(' )then
           tempo=acos(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'ATAN(' )then
           tempo=atan(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'SQRT(' )then
           if(tempo < 0.0)then
              CALL WRNDIE(0,'<chkfnct>','sqrt() of negative number')
              
           endif
           tempo=sqrt(abs(tempo))
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'TANH(' )then
           tempo=tanh(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'COSH(' )then
           tempo=cosh(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'SINH(' )then
           tempo=sinh(tempo)
           i1=i1-4
          still_checking=.false.
        elseif( exprss(i1-4:i1) == 'FACT(' )then
           tempo=factorl(tempo)
           i1=i1-4
          still_checking=.false.
        endif
      end subroutine check4

      subroutine check3
           ! Three letters functions
        if( exprss(i1-3:i1) == 'SIN(' )then
           tempo=sin(tempo)
           i1=i1-3
          still_checking=.false.
        elseif( exprss(i1-3:i1) == 'COS(' )then
           tempo=cos(tempo)
           i1=i1-3
          still_checking=.false.
        elseif( exprss(i1-3:i1) == 'TAN(' )then
           tempo=tan(tempo)
           i1=i1-3
          still_checking=.false.
        elseif( exprss(i1-3:i1) == 'EXP(' )then
           tempo=exp(tempo)
           i1=i1-3
          still_checking=.false.
        elseif( exprss(i1-3:i1) == 'INT(' )then
           tempo=int(tempo)
           i1=i1-3
          still_checking=.false.
        elseif( exprss(i1-3:i1) == 'ABS(' )then
           tempo=abs(tempo)
           i1=i1-3
          still_checking=.false.
           
       endif
       return
     end subroutine check3

      subroutine check2
           ! Two letters functions
        if( exprss(i1-2:i1) == 'LN(' )then
           if(tempo <= 0.0) then
              CALL WRNDIE(0,'<chkfnct>','ln() of negative number')
           endif
           tempo=log(abs(tempo))
           i1=i1-2
          still_checking=.false.
        endif
        return
      end subroutine check2
        
  end subroutine chkfnct

  function simple(exprss,first,last) result(simple_rtn)
    !-----------------------------------------------------------------------

    character(len=*) exprss
    integer first,last,i
    logical add,subt
    logical mult, divi, modop
    real(chm_real) tempo,factor,power,term,simple_rtn

    ! Initialize the term and the factor to evaluate a sum of product of factors
    term=0.0
    factor=1.0
    add=.false.
    subt=.false.
    mult=.true.
    divi=.false.
    modop = .false.

    i=first
    do while(i <= last)
       tempo=gtnumb(exprss,i,last)

       if(exprss(i:i+1) == '**')then
          i=i+2
          power=gtnumb(exprss,i,last)
          if(tempo < 0.0)then
             CALL WRNDIE(0,'<simple>','use of ** with negative number')
          endif
          tempo=abs(tempo)**power
       endif

       if (divi) then
          factor = factor / tempo
       else if (modop) then
          factor = mod(factor, tempo)
       else
          factor = factor * tempo
       endif

       mult = exprss(i:i) == '*'
       divi = exprss(i:i) == '/'
       add = exprss(i:i) == '+'
       subt = exprss(i:i) == '-'
       modop = exprss(i:i) == '%'
       if (mult .or. divi .or. modop) i = i + 1
       if(add.or.subt.or.(i == last+1))then
          term=term+factor
          factor=1.0
       endif
    end do

    simple_rtn=term

    return
  end function simple

  function gtnumb(exprss,first,last) result(gtnumb_rtn)
    !-----------------------------------------------------------------------
    ! This function gets the sign and the value of a number
    implicit none
    character(len=*) exprss
    integer first,last,i,ivar
    real(chm_real) signe,gtnumb_rtn
    character(len=10),save :: varnam(25)
    integer,save :: varlen(25),nvar
    real(chm_real),save :: valvar(25)
    logical tmpflag1,tmpflag2

    signe=1.0
    do i=first,last
       if(exprss(i:i) == '+')then
          signe=signe
       elseif(exprss(i:i) == '-')then
          signe=-signe
       else
          exit
       endif
    enddo

    first=i
    do i=first,last+1
       tmpflag1=.false.
       tmpflag2=.false.
       if(i > 1) then
          tmpflag1 = (exprss(i:i) == '+').and.(exprss(i-1:i) /= 'E+')
          tmpflag2 = (exprss(i:i) == '-').and.(exprss(i-1:i) /= 'E-')
       endif
       if (exprss(i:i) == '*' .or. &
            exprss(i:i) == '/' .or. &
            exprss(i:i) == '%' .or. &
            tmpflag1 .or. tmpflag2 .or. &
            i == last + 1) then
          call fndvar(varnam,varlen,nvar,ivar,exprss(first:i-1),i-first)
          if(ivar > 0)then
             gtnumb_rtn=valvar(ivar)
          else
             gtnumb_rtn=fcnvrt(exprss(first:i-1))
             !     read(exprss(first:i-1),*,err=999) gtnumb
          endif
          gtnumb_rtn=signe*gtnumb_rtn
          first=i
          return
       endif
    enddo

    first=i
    gtnumb_rtn=0.0
    return

  end function gtnumb

  function factorl(x) result(factor)
    !-----------------------------------------------------------------------
    real(chm_real) x,factor
    !     factor(x) = x! = x*(x-1)* ...1
    factor=1.0
    do while(x > 0.0)
       factor=factor*x
       x=x-1.0
    enddo
    return
  end function factorl

  subroutine fndvar(varnam,varlen,nvar,ivar,exprss,iexprss)
    !-----------------------------------------------------------------------
    character(len=*) exprss
    character(len=10) varnam(*)
    integer varlen(*),ivar,nvar,iexprss,i

    ivar=0
    do i=1,nvar
       if(varnam(i)(1:varlen(i)) == exprss(1:iexprss))then
          ivar=i
       endif
    enddo
    return
  end subroutine fndvar

  function fcnvrt(c1) result(fcnvrt_rtn)
    !-----------------------------------------------------------------------
    real(chm_real) fcnvrt_rtn
    character(len=*) c1
    character(len=7) fmt
    character(len=80) scratwd
    integer lenc1
    !
    call trim2(c1,lenc1)
    if(lenc1 == 0)then
       CALL WRNDIE(0,'<FCNVRT>','zero length string')
       fcnvrt_rtn=0.0
       return
    endif
    write(fmt,100) '(f',lenc1,'.0)'
100 format(a,i2,a)
    scratwd=c1(1:lenc1)
    read(scratwd,fmt,err=1000) fcnvrt_rtn
    return
1000 CONTINUE
    scratwd='Error in conversion: '//c1(1:lenc1)
    CALL WRNDIE(0,'<FCNVRT>',scratwd(1:lenc1+21))
    fcnvrt_rtn=0.0
    return
  end function fcnvrt
end module calc

