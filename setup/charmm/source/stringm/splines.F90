#if (KEY_STRINGM==1) /*  automatically protect all code */
! ********************************************************************************!
! Preprocessed by GNU CPP !
! ********************************************************************************!
!
#if (KEY_STRINGM==1)
!
      subroutine spline_b_val ( ndata, tdata, ydata, tval, yval )
!! SPLINE_B_VAL evaluates a cubic B spline approximant.
! Discussion:
! The cubic B spline will approximate the data, but is not
! designed to interpolate it.
! In effect, two "phantom" data values are appended to the data,
! so that the spline will interpolate the first and last data values.
! Modified:
! 07 April 1999
! Author:
! John Burkardt
! Parameters:
! Input, integer NDATA, the number of data values.
! Input, real TDATA(NDATA), the abscissas of the data.
! Input, real(chm_real) YDATA(NDATA), the data values.
! Input, real(chm_real) TVAL, a point at which the spline is to be evaluated.
! Output, real(chm_real) YVAL, the value of the function at TVAL.
!
! Ported to f77 and incorporated into CHARMM by Victor
! 5.08
!
      use chm_kinds
      implicit none
      integer ndata
!
      real(chm_real) bval
      integer left
      integer right
      real(chm_real) tdata(ndata)
      real(chm_real) tval
      real(chm_real) u
      real(chm_real) ydata(ndata)
      real(chm_real) yval
!
! Find the nearest interval [ TDATA(LEFT), TDATA(RIGHT) ] to TVAL.
!
      call rvec_bracket ( ndata, tdata, tval, left, right )
!
! Evaluate the 5 nonzero B spline basis functions in the interval,
! weighted by their corresponding data values.
!
      u = ( tval - tdata(left) ) / ( tdata(right) - tdata(left) )
      yval = 0.0E+00
!
! B function associated with node LEFT - 1, (or "phantom node"),
! evaluated in its 4th interval.
!
      bval = ( 1.0E+00 - 3.0E+00 * u + 3.0E+00 * u**2 - u**3 ) / 6.0E+00
      if ( left-1 > 0 ) then
       yval = yval + ydata(left-1) * bval
      else
       yval = yval + ( 2.0E+00 * ydata(1) - ydata(2) ) * bval
      end if
!
! B function associated with node LEFT,
! evaluated in its third interval.
!
      bval = ( 4.0E+00 - 6.0E+00 * u**2 + 3.0E+00 * u**3 ) / 6.0E+00
      yval = yval + ydata(left) * bval
!
! B function associated with node RIGHT,
! evaluated in its second interval.
!
      bval = ( 1.0E+00 + 3.0E+00 * u + 3.0E+00 * u**2 - 3.0E+00 * u**3 )&
     & / 6.0E+00
      yval = yval + ydata(right) * bval
!
! B function associated with node RIGHT+1, (or "phantom node"),
! evaluated in its first interval.
!
      bval = u**3 / 6.0E+00
      if ( right+1 <= ndata ) then
       yval = yval + ydata(right+1) * bval
      else
       yval = yval + ( 2.0E+00 * ydata(ndata) - ydata(ndata-1) ) * bval
      end if
      return
      end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      subroutine rvec_bracket ( n, x, xval, left, right )
!
!*******************************************************************************
!
!! RVEC_BRACKET searches a sorted array for successive brackets of a value.
! Discussion:
!
! If the values in the vector are thought of as defining intervals
! on the real line, then this routine searches for the interval
! nearest to or containing the given value.
!
! Modified:
!
! 06 April 1999
!
! Author:
!
! John Burkardt
!
! Parameters:
!
! Input, integer N, length of input array.
!
! Input, real X(N), an array sorted into ascending order.
!
! Input, real XVAL, a value to be bracketed.
!
! Output, integer LEFT, RIGHT, the results of the search.
! Either:
! XVAL < X(1), when LEFT = 1, RIGHT = 2;
! XVAL > X(N), when LEFT = N-1, RIGHT = N;
! or
! X(LEFT) <= XVAL <= X(RIGHT).
!
       use chm_kinds
       implicit none
!
       integer n
!
       integer i
       integer left
       integer right
       real(chm_real) x(n)
       real(chm_real) xval
!
       do i = 2, n - 1
         if ( xval .lt. x(i) ) then
           left = i - 1
           right = i
           return
         end if
        end do

       left = n - 1
       right = n

       return
      end
      subroutine rvec_bracket3 ( n, t, tval, left )
!
!*******************************************************************************
!
!! RVEC_BRACKET3 finds the interval containing or nearest a given value.
!
!
! Discussion:
!
! The routine always returns the index LEFT of the sorted array
! T with the property that either
! * T is contained in the interval [ T(LEFT), T(LEFT+1) ], or
! * T < T(LEFT) = T(1), or
! * T > T(LEFT+1) = T(N).
!
! The routine is useful for interpolation problems, where
! the abscissa must be located within an interval of data
! abscissas for interpolation, or the "nearest" interval
! to the (extreme) abscissa must be found so that extrapolation
! can be carried out.
!
! Modified:
!
! 05 April 1999
!
! Author:
!
! John Burkardt
!
! Parameters:
!
! Input, integer N, length of the input array.
!
! Input, real T(N), an array sorted into ascending order.
!
! Input, real TVAL, a value to be bracketed by entries of T.
!
! Input/output, integer LEFT.
!
! On input, if 1 <= LEFT <= N-1, LEFT is taken as a suggestion for the
! interval [ T(LEFT), T(LEFT+1) ] in which TVAL lies. This interval
! is searched first, followed by the appropriate interval to the left
! or right. After that, a binary search is used.
!
! On output, LEFT is set so that the interval [ T(LEFT), T(LEFT+1) ]
! is the closest to TVAL; it either contains TVAL, or else TVAL
! lies outside the interval [ T(1), T(N) ].
!
       use chm_kinds
       implicit none
!
       integer n
!
       integer high
       integer left
       integer low
       integer mid
       real(chm_real) t(n)
       real(chm_real) tval
!
! Check the input data.
!
       if ( n .lt. 2 ) then
call wrndie(0,'RVEC_BRACKET3',trim('N must be at least 2. Dying.'))
         return
       end if
!
! If LEFT is not between 1 and N-1, set it to the middle value.
!
       if ( left .lt. 1 .or. left .gt. n - 1 ) then
         left = ( n + 1 ) / 2
       end if
!
! CASE 1: TVAL < T(LEFT):
! Search for TVAL in [T(I), T(I+1)] for intervals I = 1 to LEFT-1.
!
       if ( tval .lt. t(left) ) then

         if ( left .eq. 1 ) then
           return
         else if ( left .eq. 2 ) then
           left = 1
           return
         else if ( tval .ge. t(left-1) ) then
           left = left - 1
           return
         else if ( tval .le. t(2) ) then
           left = 1
           return
         end if
!
! ...Binary search for TVAL in [T(I), T(I+1)] for intervals I = 2 to LEFT-2.
!
         low = 2
         high = left - 2

         do

           if ( low .eq. high ) then
             left = low
             return
           end if

           mid = ( low + high + 1 ) / 2

           if ( tval .ge. t(mid) ) then
             low = mid
           else
             high = mid - 1
           end if

         end do
!
! CASE2: T(LEFT+1) < TVAL:
! Search for TVAL in {T(I),T(I+1)] for intervals I = LEFT+1 to N-1.
!
       else if ( tval .gt. t(left+1) ) then

         if ( left .eq. n - 1 ) then
           return
         else if ( left .eq. n - 2 ) then
           left = left + 1
           return
         else if ( tval .le. t(left+2) ) then
           left = left + 1
           return
         else if ( tval .ge. t(n-1) ) then
           left = n - 1
           return
         end if
!
! ...Binary search for TVAL in [T(I), T(I+1)] for intervals I = LEFT+2 to N-2.
!
         low = left + 2
         high = n - 2

         do

           if ( low .eq. high ) then
             left = low
             return
           end if

           mid = ( low + high + 1 ) / 2

           if ( tval .ge. t(mid) ) then
             low = mid
           else
             high = mid - 1
           end if

         end do
!
! CASE3: T(LEFT) <= TVAL <= T(LEFT+1):
! T is in [T(LEFT), T(LEFT+1)], as the user said it might be.
!
       else

       end if

       return
      end
!
! adopted for F77 by Victor Ovchinnikov; added boundary condition from Ren et al 2007
!
      subroutine spline_cubic_set ( n, t, y, ibcbeg, ybcbeg, &
     & ibcend, ybcend, ypp )
!
!*******************************************************************************
!
!! SPLINE_CUBIC_SET computes the second derivatives of a cubic spline.
!
!
! Discussion:
!
! For data interpolation, the user must call SPLINE_CUBIC_SET to
! determine the second derivative data, passing in the data to be
! interpolated, and the desired boundary conditions.
!
! The data to be interpolated, plus the SPLINE_CUBIC_SET output,
! defines the spline. The user may then call SPLINE_CUBIC_VAL to
! evaluate the spline at any point.
!
! The cubic spline is a piecewise cubic polynomial. The intervals
! are determined by the "knots" or abscissas of the data to be
! interpolated. The cubic spline has continous first and second
! derivatives over the entire interval of interpolation.
!
! For any point T in the interval T(IVAL), T(IVAL+1), the form of
! the spline is
!
! SPL(T) = A(IVAL)
! + B(IVAL) * ( T - T(IVAL) )
! + C(IVAL) * ( T - T(IVAL) )**2
! + D(IVAL) * ( T - T(IVAL) )**3
!
! If we assume that we know the values Y(*) and YPP(*), which represent
! the values and second derivatives of the spline at each knot, then
! the coefficients can be computed as:
!
! A(IVAL) = Y(IVAL)
! B(IVAL) = ( Y(IVAL+1) - Y(IVAL) ) / ( T(IVAL+1) - T(IVAL) )
! - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * ( T(IVAL+1) - T(IVAL) ) / 6
! C(IVAL) = YPP(IVAL) / 2
! D(IVAL) = ( YPP(IVAL+1) - YPP(IVAL) ) / ( 6 * ( T(IVAL+1) - T(IVAL) ) )
!
! Since the first derivative of the spline is
!
! SPL`(T) = B(IVAL)
! + 2 * C(IVAL) * ( T - T(IVAL) )
! + 3 * D(IVAL) * ( T - T(IVAL) )**2,
!
! the requirement that the first derivative be continuous at interior
! knot I results in a total of N-2 equations, of the form:
!
! B(IVAL-1) + 2 C(IVAL-1) * (T(IVAL)-T(IVAL-1))
! + 3 * D(IVAL-1) * (T(IVAL) - T(IVAL-1))**2 = B(IVAL)
!
! or, setting H(IVAL) = T(IVAL+1) - T(IVAL)
!
! ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)
! - ( YPP(IVAL) + 2 * YPP(IVAL-1) ) * H(IVAL-1) / 6
! + YPP(IVAL-1) * H(IVAL-1)
! + ( YPP(IVAL) - YPP(IVAL-1) ) * H(IVAL-1) / 2
! =
! ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
! - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * H(IVAL) / 6
!
! or
!
! YPP(IVAL-1) * H(IVAL-1) + 2 * YPP(IVAL) * ( H(IVAL-1) + H(IVAL) )
! + YPP(IVAL) * H(IVAL)
! =
! 6 * ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
! - 6 * ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)
!
! Boundary conditions must be applied at the first and last knots.
! The resulting tridiagonal system can be solved for the YPP values.
!
! Modified:
!
! 20 November 2000
!
! Author:
!
! John Burkardt
!
! Modified for F77 by Victor Ovchinnikov; 5/3/07
!
!
! Parameters:
!
! Input, integer N, the number of data points; N must be at least 2.
!
! Input, real T(N), the points where data is specified.
! The values should be distinct, and increasing.
!
! Input, real Y(N), the data values to be interpolated.
!
! Input, integer IBCBEG, the left boundary condition flag:
!
! 0: the spline should be a quadratic over the first interval;
! 1: the first derivative at the left endpoint should be YBCBEG;
! 2: the second derivative at the left endpoint should be YBCBEG.
! 3: ADDED BY Victor Ovchinnikov:
! the second derivative at the right endpoint is determined by matching
! the THIRD derivative of the Lagrangian interpolant of the last FOUR points
!
! Input, real YBCBEG, the left boundary value, if needed.
!
! Input, integer IBCEND, the right boundary condition flag:
!
! 0: the spline should be a quadratic over the last interval;
! 1: the first derivative at the right endpoint should be YBCEND;
! 2: the second derivative at the right endpoint should be YBCEND.
! 3: ADDED BY Victor Ovchinnikov:
! the second derivative at the right endpoint is determined by matching
! the THIRD derivative of the Lagrangian interpolant of the last FOUR points
!
!
! Input, real YBCEND, the right boundary value, if needed.
!
! Output, real YPP(N), the second derivatives of the cubic spline.
!
       use chm_kinds
       implicit none
!
       integer n
!
       real(chm_real) diag(n)
       integer i
       integer ibcbeg
       integer ibcend
       real(chm_real) sub(2:n)
       real(chm_real) sup(1:n-1)
       real(chm_real) t(n)
       real(chm_real) y(n)
       real(chm_real) ybcbeg
       real(chm_real) ybcend
       real(chm_real) ypp(n)
!
! Check.
!
       if ( n .le. 1 ) then
call wrndie(0,'SPLINE_CUBIC_SET',trim('N must be > 1. Dying'))
         return
       end if

       do i = 1, n-1
         if ( t(i) .ge. t(i+1) ) then
call wrndie(0,'SPLINE_CUBIC_SET',trim('Knots must be strictly increasing. Dying.'))
         return
         end if
       end do
!
! Set the first equation.
!
       if ( ibcbeg .eq. 0 ) then
         ypp(1) = 0.0E+00
         diag(1) = 1.0E+00
         sup(1) = -1.0E+00
       else if ( ibcbeg .eq. 1 ) then
         ypp(1) = ( y(2) - y(1) ) / ( t(2) - t(1) ) - ybcbeg
         diag(1) = ( t(2) - t(1) ) / 3.0E+00
         sup(1) = ( t(2) - t(1) ) / 6.0E+00
       else if ( ibcbeg .eq. 2 ) then
         ypp(1) = ybcbeg
         diag(1) = 1.0E+00
         sup(1) = 0.0E+00
! VO
       else if ( ibcbeg .eq. 3 ) then
        if (n.gt.3) then
         diag(1) = -1.0
         sup(1) = 1.0
         ypp(1) =6.0*(y(1)/((t(1)-t(2))*(t(1)-t(3))*(t(1)-t(4))) + &
     & y(2)/((t(2)-t(1))*(t(2)-t(3))*(t(2)-t(4))) + &
     & y(3)/((t(3)-t(1))*(t(3)-t(2))*(t(3)-t(4))) + &
     & y(4)/((t(4)-t(1))*(t(4)-t(2))*(t(4)-t(3))))
        else
call wrndie(0,'SPLINE_CUBIC_SET',trim('ibcbeg=3 requires N>3. Dying.'))
         return
        endif

       else
call wrndie(0,'SPLINE_CUBIC_SET',trim('Invalid boundary flag IBCBEG. Dying.'))
         return
       end if
!
! Set the intermediate equations.
!
       do i = 2, n-1
         ypp(i) = ( y(i+1) - y(i) ) / ( t(i+1) - t(i) ) &
     & - ( y(i) - y(i-1) ) / ( t(i) - t(i-1) )
         sub(i) = ( t(i) - t(i-1) ) / 6.0E+00
         diag(i) = ( t(i+1) - t(i-1) ) / 3.0E+00
         sup(i) = ( t(i+1) - t(i) ) / 6.0E+00
       end do
!
! Set the last equation.
!
       if ( ibcend .eq. 0 ) then
         ypp(n) = 0.0E+00
         sub(n) = -1.0E+00
         diag(n) = 1.0E+00
       else if ( ibcend .eq. 1 ) then
         ypp(n) = ybcend - ( y(n) - y(n-1) ) / ( t(n) - t(n-1) )
         sub(n) = ( t(n) - t(n-1) ) / 6.0E+00
         diag(n) = ( t(n) - t(n-1) ) / 3.0E+00
       else if ( ibcend .eq. 2 ) then
         ypp(n) = ybcend
         sub(n) = 0.0E+00
         diag(n) = 1.0E+00
! VO
       else if ( ibcend .eq. 3 ) then
        if (n.gt.3) then
         diag(n) = 1.0
         sub(n) = -1.0
         ypp(n) =6.0* &
     & (y(n-3)/((t(n-3)-t(n-2))*(t(n-3)-t(n-1))*(t(n-3)-t(n)))+ &
     & y(n-2)/((t(n-2)-t(n-3))*(t(n-2)-t(n-1))*(t(n-2)-t(n)))+&
     & y(n-1)/((t(n-1)-t(n-3))*(t(n-1)-t(n-2))*(t(n-1)-t(n)))+&
     & y(n)/((t(n)-t(n-3))*(t(n)-t(n-2))*(t(n)-t(n-1))))
        else
call wrndie(0,'SPLINE_CUBIC_SET',trim('ibcend=3 requires N>3. Dying.'))
         return
        endif

       else
call wrndie(0,'SPLINE_CUBIC_SET',trim('Invalid boundary flag IBCEND. Dying.'))
         return
       end if
!
! Special case:
! N = 2, IBCBEG = IBCEND = 0.
!
       if ( n .eq. 2 .and. ibcbeg .eq. 0 .and. ibcend .eq. 0 ) then

         ypp(1) = 0.0E+00
         ypp(2) = 0.0E+00
!
! Solve the linear system.
!
       else

         call s3_fs ( sub, diag, sup, n, ypp, ypp )

       end if

       return
      end
!-----------------------------------------------------------
!-----------------------------------------------------------
      subroutine spline_cubic_val ( n, t, y, ypp, tval, yval, &
     & ypval, yppval )
!
!*******************************************************************************
!
!! SPLINE_CUBIC_VAL evaluates a cubic spline at a specific point.
!
!
! Discussion:
!
! SPLINE_CUBIC_SET must have already been called to define the
! values of YPP.
!
! For any point T in the interval T(IVAL), T(IVAL+1), the form of
! the spline is
!
! SPL(T) = A
! + B * ( T - T(IVAL) )
! + C * ( T - T(IVAL) )**2
! + D * ( T - T(IVAL) )**3
!
! Here:
! A = Y(IVAL)
! B = ( Y(IVAL+1) - Y(IVAL) ) / ( T(IVAL+1) - T(IVAL) )
! - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * ( T(IVAL+1) - T(IVAL) ) / 6
! C = YPP(IVAL) / 2
! D = ( YPP(IVAL+1) - YPP(IVAL) ) / ( 6 * ( T(IVAL+1) - T(IVAL) ) )
!
! Modified:
!
! 20 November 2000
!
! Author:
!
! John Burkardt
!
! Parameters:
!
! Input, integer N, the number of data values.
!
! Input, real T(N), the knot values.
!
! Input, real Y(N), the data values at the knots.
!
! Input, real YPP(N), the second derivatives of the spline at the knots.
!
! Input, real TVAL, a point, typically between T(1) and T(N), at
! which the spline is to be evalulated. If TVAL lies outside
! this range, extrapolation is used.
!
! Output, real YVAL, YPVAL, YPPVAL, the value of the spline, and
! its first two derivatives at TVAL.
!
       use chm_kinds
       implicit none
       integer n
!
       real(chm_real) dt
       real(chm_real) h
       integer left
       integer right
       real(chm_real) t(n)
       real(chm_real) tval
       real(chm_real) y(n)
       real(chm_real) ypp(n)
       real(chm_real) yppval
       real(chm_real) ypval
       real(chm_real) yval
!
! Determine the interval [T(LEFT), T(RIGHT)] that contains TVAL.
! Values below T(1) or above T(N) use extrapolation.
!
       call rvec_bracket ( n, t, tval, left, right )
!
! Evaluate the polynomial.
!
       dt = tval - t(left)
       h = t(right) - t(left)

       yval = y(left) &
     & + dt * ( ( y(right) - y(left) ) / h &
     & - ( ypp(right) / 6.0E+00 + ypp(left) / 3.0E+00 ) * h&
     & + dt * ( 0.5E+00 * ypp(left) &
     & + dt * ( ( ypp(right) - ypp(left) ) / ( 6.0E+00 * h ) ) ) )

       ypval = ( y(right) - y(left) ) / h &
     & - ( ypp(right) / 6.0E+00 + ypp(left) / 3.0E+00 ) * h &
     & + dt * ( ypp(left) &
     & + dt * ( 0.5E+00 * ( ypp(right) - ypp(left) ) / h ) )

       yppval = ypp(left) + dt * ( ypp(right) - ypp(left) ) / h

       return
      end
!
      subroutine spline_cubic_val2 ( n, t, y, ypp, left, &
     & tval, yval, ypval, yppval )
!
!*******************************************************************************
!
!! SPLINE_CUBIC_VAL2 evaluates a cubic spline at a specific point.
!
!
! Discussion:
!
! This routine is a modification of SPLINE_CUBIC_VAL; it allows the
! user to speed up the code by suggesting the appropriate T interval
! to search first.
!
! SPLINE_CUBIC_SET must have already been called to define the
! values of YPP.
!
! In the LEFT interval, let RIGHT = LEFT+1. The form of the spline is
!
! SPL(T) =
! A
! + B * ( T - T(LEFT) )
! + C * ( T - T(LEFT) )**2
! + D * ( T - T(LEFT) )**3
!
! Here:
! A = Y(LEFT)
! B = ( Y(RIGHT) - Y(LEFT) ) / ( T(RIGHT) - T(LEFT) )
! - ( YPP(RIGHT) + 2 * YPP(LEFT) ) * ( T(RIGHT) - T(LEFT) ) / 6
! C = YPP(LEFT) / 2
! D = ( YPP(RIGHT) - YPP(LEFT) ) / ( 6 * ( T(RIGHT) - T(LEFT) ) )
!
! Modified:
!
! 20 November 2000
!
! Author:
!
! John Burkardt
!
! Parameters:
!
! Input, integer N, the number of knots.
!
! Input, real T(N), the knot values.
!
! Input, real Y(N), the data values at the knots.
!
! Input, real YPP(N), the second derivatives of the spline at
! the knots.
!
! Input/output, integer LEFT, the suggested T interval to search.
! LEFT should be between 1 and N-1. If LEFT is not in this range,
! then its value will be ignored. On output, LEFT is set to the
! actual interval in which TVAL lies.
!
! Input, real TVAL, a point, typically between T(1) and T(N), at
! which the spline is to be evalulated. If TVAL lies outside
! this range, extrapolation is used.
!
! Output, real YVAL, YPVAL, YPPVAL, the value of the spline, and
! its first two derivatives at TVAL.
!
!
       use chm_kinds
       implicit none
       integer n
!
       real(chm_real) dt
       real(chm_real) h
       integer left
       integer right
       real(chm_real) t(n)
       real(chm_real) tval
       real(chm_real) y(n)
       real(chm_real) ypp(n)
       real(chm_real) yppval
       real(chm_real) ypval
       real(chm_real) yval
!
! Determine the interval [T(LEFT), T(RIGHT)] that contains TVAL.
!
! What you want from RVEC_BRACKET3 is that TVAL is to be computed
! by the data in interval {T(LEFT), T(RIGHT)].
!
       left = 0
       call rvec_bracket3 ( n, t, tval, left )
       right = left + 1
!
! In the interval LEFT, the polynomial is in terms of a normalized
! coordinate ( DT / H ) between 0 and 1.
!
       dt = tval - t(left)
       h = t(right) - t(left)

       yval = y(left) + dt * ( ( y(right) - y(left) ) / h &
     & - ( ypp(right) / 6.0E+00 + ypp(left) / 3.0E+00 ) * h&
     & + dt * ( 0.5E+00 * ypp(left) &
     & + dt * ( ( ypp(right) - ypp(left) ) / ( 6.0E+00 * h ) ) ) )

       ypval = ( y(right) - y(left) ) / h &
     & - ( ypp(right) / 6.0E+00 + ypp(left) / 3.0E+00 ) * h &
     & + dt * ( ypp(left) &
     & + dt * ( 0.5E+00 * ( ypp(right) - ypp(left) ) / h ) )

       yppval = ypp(left) + dt * ( ypp(right) - ypp(left) ) / h

       return
      end
!*************************************************************************
      subroutine s3_fs ( a1, a2, a3, n, b, x )
!
!*******************************************************************************
!
!! S3_FS factors and solves a tridiagonal linear system.
!
!
! Note:
!
! This algorithm requires that each diagonal entry be nonzero.
!
! Modified:
!
! 05 December 1998
!
! Author:
!
! John Burkardt
!
! Parameters:
!
! Input/output, real A1(2:N), A2(1:N), A3(1:N-1).
! On input, the nonzero diagonals of the linear system.
! On output, the data in these vectors has been overwritten
! by factorization information.
!
! Input, integer N, the order of the linear system.
!
! Input/output, real B(N).
! On input, B contains the right hand side of the linear system.
! On output, B has been overwritten by factorization information.
!
! Output, real X(N), the solution of the linear system.
!
       use chm_kinds
       implicit none
!
       integer n
!
       real(chm_real) a1(2:n)
       real(chm_real) a2(1:n)
       real(chm_real) a3(1:n-1)
       real(chm_real) b(n)
       integer i
       real(chm_real) x(n)
       real(chm_real) xmult
!
! The diagonal entries cannot be zero.
!
       do i = 1, n
         if ( a2(i) .eq. 0.0E+00 ) then
call wrndie(0,'S3_FS',trim('Zero diagonal value. Dying'))
           return
         end if
       end do

       do i = 2, n-1

         xmult = a1(i) / a2(i-1)
         a2(i) = a2(i) - xmult * a3(i-1)

         b(i) = b(i) - xmult * b(i-1)

       end do

       xmult = a1(n) / a2(n-1)
       a2(n) = a2(n) - xmult * a3(n-1)

       x(n) = ( b(n) - xmult * b(n-1) ) / a2(n)
       do i = n-1, 1, -1
         x(i) = ( b(i) - a3(i) * x(i+1) ) / a2(i)
       end do

       return
      end
!-----------------------------------------------------------------------
      subroutine linear_interp(xin, yin, nin, xout, yout, nout, dydxout)
      use chm_kinds
      implicit none
      integer :: nin, nout
      real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
      real(chm_real), optional :: dydxout(nout) ! tangent computation
      real(chm_real) :: dydx(nout)
!
      integer i,j
! very little error checking
!
! begin
      if (nin.eq.1) then
       yout=yin(1)
       if (present(dydxout)) dydxout=0d0
      return
      elseif (nin.eq.0) then
       yout=0d0
       if (present(dydxout)) dydxout=0d0
call wrndie(0,'LINEAR_INTERP',trim('SOURCE ARRAY LENGTH IS ZERO. WILL RETURN ZEROS.'))
       return
      endif
!
      i=1
      j=2
      do while (i.le.nout)
       do while ( xin(j).lt.xout(i).and.j.lt.nin )
        j=j+1
       enddo
       dydx(i)=(yin(j)-yin(j-1))/(xin(j)-xin(j-1)) ! slope
       yout(i)=yin(j-1)+(xout(i)-xin(j-1))*dydx(i)
       i=i+1
      enddo
      if (present(dydxout)) dydxout=dydx
      return
      end
!---------------------------------------------------------
      function smooth2(t,x,delta)
! smoothing function; the endpoints are modified!
! uses the linear interpolation subroutine
      use chm_kinds
      implicit none
 character(len=132)::info(21)=(/'','','','','','','','','','','','','','','','','','','','',''/);! output buffer
      integer :: delta
      real(chm_real) :: t(:), x(:)
! locals
      integer :: nt, nx, i
      real(chm_real), dimension(size(t)) :: xtemp, xnew, smooth2
!
! pathscale compiler needs an explicit interface
!
      interface
      subroutine linear_interp(xin, yin, nin, xout, yout, nout, dydxout)
      use chm_kinds
      implicit none
      integer :: nin, nout
      real(chm_real) :: xin(nin), yin(nin), xout(nout), yout(nout)
      real(chm_real), optional :: dydxout(nout) ! tangent computation
      end subroutine linear_interp
      end interface
!
! do work
      nx=size(x)
      nt=size(t)
      if (nx.ne.nt) then
    call wrndie(0,'SMOOTH2',trim('ARRAYS MUST BE OF THE SAME LENGTH'))
       return
      endif
!
      xnew=0d0

      if (delta.gt.nx) then
       delta=max(1,floor(1.0d0*nx/3.0d0))
write(info(1),*)'FILTER WIDTH TOO LARGE. SETTING TO ',delta,'.';call wrndie(0,'SMOOTH2',trim(info(1)))
!
      elseif (delta.le.0) then
       delta=max(1,floor(1.0d0*nx/3.0d0))
write(info(1),*)'FILTER WIDTH ZERO. SETTING TO ',delta,'.';call wrndie(0,'SMOOTH2',trim(info(1)))
      endif
!
      do i=1, delta
       call linear_interp(t(i:nx:delta), x(i:nx:delta), &
     & (nx-i)/delta+1 ,t, xtemp, nx)
       xnew=xnew+xtemp
      enddo
      xnew=xnew/delta
      smooth2=xnew
      end function smooth2
!----------------------------------------------------------------------
#endif
#endif /* automatically protect all code */
