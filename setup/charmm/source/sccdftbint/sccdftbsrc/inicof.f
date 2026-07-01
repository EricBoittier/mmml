        Subroutine inicof(cextr,coefk,IOrder)
        Implicit Real*8(A-H,O-Z)
        Dimension cextr(*)
        Dimension C(0:9,3:9)
        Real*8 Kappa(3:9), alpha(3:9)
        Data Kappa/1.692D0,1.75D0,1.804D0,1.838D0,1.86D0,1.88D0,1.89D0/
        Data alpha/150.0D-3, 57.0D-3, 18.0D-3, 5.5D-3, 1.6D-3, 0.44D-3,
     $             0.12D-3/
        Data Zero/0.0D0/, One/1.0D0/, Two/2.0d0/

        If(IOrder.gt.9) then
           Write(*,*) 'inicof: the allowed highest extrapolation order',
     $         ' is',9
           STOP
        endif
        Do 100 I = 3, 9
        Do 100 J = 0, 9
          C(J,I) = Zero
 100    Continue
C       order 3 -- 105 (1st order in 2011 JCP paper)
        C(0,3) =   -2.0d0
        C(1,3) =    3.0d0
        C(2,3) =    0.0d0
        C(3,3) =   -1.0d0
C       order 4 -- 107 (3rd order in 2011 JCP paper)
        C(0,4) =   -3.0d0
        C(1,4) =    6.0d0
        C(2,4) =   -2.0d0
        C(3,4) =   -2.0d0
        C(4,4) =    1.0d0
C       order 5 -- 109 (5th order in 2011 JCP paper)
        C(0,5) =   -6.0d0
        C(1,5) =   14.0d0
        C(2,5) =   -8.0d0
        C(3,5) =   -3.0d0
        C(4,5) =    4.0d0
        C(5,5) =   -1.0d0
C       order 6 -- 111 (7th order in 2011 JCP paper)
        C(0,6) =  -14.0d0
        C(1,6) =   36.0d0
        C(2,6) =  -27.0d0
        C(3,6) =   -2.0d0
        C(4,6) =   12.0d0
        C(5,6) =   -6.0d0
        C(6,6) =    1.0d0
C       order 7 -- 113 (9th order in 2011 JCP paper)
        C(0,7) =  -36.0d0
        C(1,7) =   99.0d0
        C(2,7) =  -88.0d0
        C(3,7) =   11.0d0
        C(4,7) =   32.0d0
        C(5,7) =  -25.0d0
        C(6,7) =    8.0d0
        C(7,7) =   -1.0d0
C       order 8 -- 115 (11th order in 2011 JCP paper)
        C(0,8) =  -99.0d0
        C(1,8) =  286.0d0
        C(2,8) = -286.0d0
        C(3,8) =   78.0d0
        C(4,8) =   78.0d0
        C(5,8) =  -90.0d0
        C(6,8) =   42.0d0
        C(7,8) =  -10.0d0
        C(8,8) =    1.0d0
C       order 9 -- 117 (13th order in 2011 JCP paper)
        C(0,9) = -286.0d0
        C(1,9) =  858.0d0
        C(2,9) = -936.0d0
        C(3,9) =  364.0d0
        C(4,9) =  168.0d0
        C(5,9) = -300.0d0
        C(6,9) =  184.0d0
        C(7,9) =  -63.0d0
        C(8,9) =   12.0d0
        C(9,9) =   -1.0d0
C       final coefficients
        Do 300 I = 0, IOrder
           cextr(I) = C(I,IOrder)*alpha(IOrder)
 300    Continue
        Coefk = Kappa(IOrder)
        cextr(0) = 2.0d0 - Coefk + cextr(0)
        cextr(1) = cextr(1) - 1.0d0
        WRITE(*,'(A,F12.6,X,A,20(F12.6,X))')
     $       'INICOF> Kappa=',CoefK,'Coef:',cextr(0:Iorder)
        Return
        End

