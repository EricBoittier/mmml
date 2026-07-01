module rxenemod
  use chm_types
  use dimens_fcm
  implicit none

  private

  ! .true. if vpress is present in call to rxnene
  logical q_vpress
  ! local contribution to virial pressure, added to vpress_inout
  real(chm_real) vpress(9)

  ! Public subroutines
#if KEY_RXNCOR==1
  public rxnene, ascend  
#endif
  ! JMS 9/2012 -- Adaptive umbrella sampling needs DECEND as well 
#if KEY_ADUMBRXNCOR==1
  public decend 
#endif
contains

#if KEY_RXNCOR==1 /*rxene_main*/
  subroutine RXNENE(EUMB, vpress_inout)
    !
    !       ---- to calculate energy due to umbrella potential
    !
    !       ---- Kottalam, June 89
    !
    use number
    use rxncom
    use rxdefs,only: umbpot

    real(chm_real)  EUMB
    real(chm_real), optional :: vpress_inout(9)
    !
    INTEGER IR, INDEX
    real(chm_real)  E
    !
    !       ---- Recall that routine RXNDEF made a tree data structure.
    !       ---- First, ascend to the root to calculate reaction coordinate
    !       ---- (Root above, branches below ...     Bhagavad Gita 15:1)
    !
    call ASCEND
    !
    !       ---- Now we have the reaction coordinate. Calculate the umbrella
    !       ---- potential and derivatives w.r.t. reaction coordinate
    !       ---- ARD and MFH 03-05-13 ---- Added loop
    !
    eumb = zero
    q_vpress = .false.
    if (present(vpress_inout)) then
       q_vpress = .true.
       vpress(1:9) = zero
    endif
    do ir = 1, nrxncr
       index = treelo(ir)
       !          write(*,*)'rxnene>index,ir,nrxncr=',index,ir,nrxncr
       !          write(*,*)'rxnene>delder(1,index)=',delder(1,index)
       !          write(*,*)'rxnene>delval(1,index)=',delval(1,index)
       !          write(*,*)'rxnene>allocated(basahi)',allocated(basahi)
       call umbpot(e,delder(1,index),delval(1,index), &
            ir,umbfrm,kumbpr,dl0ptr, &
            nbiasp,cbiasp,rbiasp, &
            deftyp(index),pumbpr &
#if KEY_SMD==1
            ,smddel,basalo,basahi,  &      
#endif
#if KEY_SMD==1
            basblo,basbhi,qflip    &       
#endif
            )
       eumb = eumb + e
    enddo
    !
    !       ---- Descend into the branches to calculate derivatives
    !       ---- w.r.t. cartesians
    !       ---- ARD and MFH 03-05-13 ---- Added loop
    !
    do ir = 1, nrxncr
       call decend(ir,umbfrm,kumbpr,dl0ptr)
    enddo
    !
    !       ---- By the way, let us keep track of the evolution of
    !       ---- the reaction coordinate
    !
    ! yousung
    !     rxncnt = rxncnt + 1 is processed whenever SUB rxnene is called,
    !     which is then called at every IPRF, ISVF, and NTRF call,
    !     besides at normal MD steps. Consequence is that one has more
    !     'hits' in .stt file than the number of MD steps actually taken.
    !     So, instead of rxncnt, use MDSTEP2 in contrl.f90 as a control
    !     variable for updating statististics.
    !     Make a minor change in dynamc/dynamc.src (MDSTEP2=ISTEP has to be
    !     moved up).
    !     old_MDSTEP is in COMMON block in rxncom.f90
    !yw   MDSTEP2 is renamed UMBMDSTEP and reside in rxncom.f90
    if ((umbmdstep  /=  old_mdstep) .or. (umbmdstep == 0)) then
       old_mdstep =  umbmdstep

       call rxnstt(delval,rxncnt,sttstp,delstp,nmlstt, &
            lodel,hidel,deldel,treelo, &
            nrxncr,rxntrn,trcfrq,rxntrc,trunit,deftyp)

    end if

    if (q_vpress) then
       vpress_inout(1:9) = vpress_inout(1:9) + vpress(1:9)
    endif

    return
  end subroutine RXNENE

  !==================================================================
  !        ASCEND
  !==================================================================
  subroutine ASCEND
    !
    !       ---- to ascend to the root to calculate reaction coordinate
    !       ---- Kottalam, June 89
    !
    use number
    use exfunc
    !
    use rxncom
    use stream

    !       ---- declare local variables
    !
    integer i, n, iat, point1, point2, dir, dir1, dir2
    integer lpnt1,lpnt2,pointa
    ! Puja [QC: 11/17]
    integer n1, n2, j

    real(chm_real) :: delinv,dist_pt_ln,lsqterm,lsqx,lsqy,lsqz,adel,bdel,cdel
    integer delta1, delta2, dir3, wt, k
    real(chm_real) delx, dely, delz, ct, dist, a, b, c, dot
    real(chm_real) epsx, epsy, epsz, theta, disc, a1, a2, a3, a4, &
         b1, b2, b3, b4, c1, c2, c3, c4, din
    real(chm_real) :: xa,ya,za,x1,y1,z1,x2,y2,z2,dt, &
         dd21inv,delx21,dely21,delz21,delxa1,delya1,delza1,invdist,dist_pt_ln_sq
    character(len=6) from
    !
    !       ---- Loop in reverse so that the referenced elements are
    !       ---- calculated before the referencing element
    !
    loop100: do i = nrxn, ngraph+1, -1
       !
       select case (deftyp(i))
       case(POINT_TYPE)
          !           ---- point in terms of atomic coordinates
          !
          ! write(*,*)'ascend>i,refnod=',i,(refnod(k,i),k=1,3)

          n = refnod(1,i)
          iat = refnod(2,i)
          wt = refnod(3,i)
          from = 'ASCEND'
          ! write(*,*)'ascend>n,iat=',n,iat
          ! write(*,*)'ascend>i,iat,iataa(i)%a(1)=',i,iat,iataa(iat)%a(1)
          !          call print_struc(i,n,iataa)
          call GETPNT (n, iataa(iat)%a, wtaa(iat)%a, from, delval(1,i), &
               delder(1,i))
          !
       case(DIREPTPT_TYPE) 
          !           ---- direction from a point to another point
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          delx = delval(1,point1)  - delval(1,point2)
          dely = delval(2,point1)  - delval(2,point2)
          delz = delval(3,point1)  - delval(3,point2)
          din = sqrt (delx*delx + dely*dely + delz*delz)
          if (din == zero) then
             IF(WRNLEV >= 2) write (outu,*) &
                  ' RXNCOR attempted to find', &
                  ' direction from a point to itself'
             call wrndie (-3, '<ASCEND>',  &
                  ' Wrong or bad RXNCOR definition')
          end if
          din = ONE / din
          delval(1,i) = din * delx
          delval(2,i) = din * dely
          delval(3,i) = din * delz
          delder(1,point1) = din
          !
       case(DIREDRDR_TYPE)
          !           ---- direction as a cross product of two other directions
          !
          dir1 = refnod(1,i)
          dir2 = refnod(2,i)
          delx = delval(2,dir1) * delval(3,dir2) &
               - delval(2,dir2) * delval(3,dir1)
          dely = delval(1,dir2) * delval(3,dir1) &
               - delval(1,dir1) * delval(3,dir2)
          delz = delval(1,dir1) * delval(2,dir2) &
               - delval(1,dir2) * delval(2,dir1)
          din = sqrt (delx*delx + dely*dely + delz*delz)
          if (din == zero) then
             IF(WRNLEV >= 2) write (outu,*) &
                  ' RXNCOR attempted to find', &
                  ' cross product between parallel vectors'
             call wrndie (-3, '<ASCEND>',  &
                  ' Wrong or bad RXNCOR definition')
          end if
          din = ONE / din
          delval(1,i) = din * delx
          delval(2,i) = din * dely
          delval(3,i) = din * delz
          delder(1,dir1) = din
          !
       case(ANGLDRDR_TYPE)
          !           ---- angle between two directions
          !
          dir1 = refnod(1,i)
          dir2 = refnod(2,i)
          dir3 = refnod(3,i)
          a1 = delval(1,dir1)
          b1 = delval(2,dir1)
          c1 = delval(3,dir1)
          a2 = delval(1,dir2)
          b2 = delval(2,dir2)
          c2 = delval(3,dir2)
          a3 = delval(1,dir3)
          b3 = delval(2,dir3)
          c3 = delval(3,dir3)
          a4 = b1*c2 - c1*b2
          b4 = c1*a2 - a1*c2
          c4 = a1*b2 - b1*a2
          ct   = a1*a2 + b1*b2 + c1*c2
          disc = a4*a3 + b4*b3 + c4*c3
          if (ct > one)  ct = ONE
          if (ct < -one) ct = -ONE
          theta = acos(ct)
          if (disc < zero) theta = - theta
          delval(1,i) = theta
          !
       case(DISTPTPT_TYPE)
          !           ---- distance between two points
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          delx = delval(1,point1) - delval(1,point2)
          dely = delval(2,point1) - delval(2,point2)
          delz = delval(3,point1) - delval(3,point2)
          dist = sqrt (delx*delx + dely*dely + delz*delz)
          delval(1,i) = dist
          delder(1,point1) = delx
          delder(2,point1) = dely
          delder(3,point1) = delz
          !
       case(DISTPTLN_TYPE)
          !           ---- shortest distance between a point and a line
          !           ---- (The line is represented in the "point-direction" form)
          !
          !--- a is the point off the line
          !--- 1 is the first point of the line
          !--- 2 is the secnd point of the line
          pointa = refnod(1,i)
          lpnt1 = refnod(2,i)
          lpnt2 = refnod(3,i)
          xa = delval(1,pointa)
          ya = delval(2,pointa)
          za = delval(3,pointa)
          x1 = delval(1,lpnt1)
          y1 = delval(2,lpnt1)
          z1 = delval(3,lpnt1)
          x2 = delval(1,lpnt2)
          y2 = delval(2,lpnt2)
          z2 = delval(3,lpnt2)
          
          delxa1 = xa - x1
          delya1 = ya - y1
          delza1 = za - z1
          delx21 = x2 - x1
          dely21 = y2 - y1
          delz21 = z2 - z1
          dd21inv=one/(delx21*delx21+dely21*dely21+delz21*delz21)
          lsqterm=dd21inv*( delxa1*delx21 + delya1*dely21 + delza1*delz21 )
          lsqx = delxa1 - delx21*lsqterm
          lsqy = delya1 - dely21*lsqterm
          lsqz = delza1 - delz21*lsqterm
          dist_pt_ln_sq = (lsqx*lsqx + lsqy*lsqy + lsqz*lsqz)
          if(dist_pt_ln_sq <= tiny(one)) then
             invdist = -one
             dist_pt_ln = zero
          else
             invdist = one/sqrt(dist_pt_ln_sq)
             dist_pt_ln = dist_pt_ln_sq*invdist
          endif
          delval(1,i) = dist_pt_ln
          delval(2,i) = invdist
          !---  Derivs are missing invdist factor in case dist is zero,
          !---     to be dealt with in subroutine descend

          !---  derivs wrt lpnt1 --------------
          delder(1,pointa) = (lsqx - &
               (lsqx*delx21 + lsqy*dely21 + lsqz*delz21)*dd21inv &
               *delx21)  
          delder(2,pointa) = (lsqy - &
               (lsqx*delx21 + lsqy*dely21 + lsqz*delz21)*dd21inv &
               *dely21)
          delder(3,pointa) = (lsqz - &
               (lsqx*delx21 + lsqy*dely21 + lsqz*delz21)*dd21inv &
               *delz21)

          !---  derivs wrt lpnt1 --------------

          dt = (two*x1-x2-xa + two*lsqterm*delx21)*dd21inv
          delder(1,lpnt1) = ( lsqx*(-one+lsqterm - delx21*dt) - &
               lsqy*dely21*dt - lsqz*delz21*dt)

          dt = (two*y1-y2-ya + two*lsqterm*dely21)*dd21inv
          delder(2,lpnt1) = ( - lsqx*delx21*dt + &
               lsqy*(-one + lsqterm - dely21*dt) - lsqz*delz21*dt)

          dt = (two*z1-z2-za + two*lsqterm*delz21)*dd21inv
          delder(3,lpnt1) = ( - lsqx*delx21*dt - &
               lsqy*dely21*dt + lsqz*(-one + lsqterm - delz21*dt) )
          
          !---  derivs wrt lpnt2 --------------

          dt = (xa-x1 - two*lsqterm*delx21)*dd21inv
          delder(1,lpnt2) = ( lsqx*(-lsqterm - delx21*dt) &
               - lsqy*dely21*dt - lsqz*delz21*dt)

          dt = (ya-y1 - two*lsqterm*dely21)*dd21inv
          delder(2,lpnt2) = ( - lsqx*delx21*dt + &
               lsqy*(-lsqterm - dely21*dt) - lsqz*delz21*dt)

          dt = (za-z1 + two*lsqterm*delz21)*dd21inv
          delder(3,lpnt2) = ( - lsqx*delx21*dt - &
               lsqy*dely21*dt + lsqz*(-lsqterm - delz21*dt) )

          !
       case(DISTPTPL_TYPE)
          !           ---- shortest distance between a point and a plane
          !           ---- (The plane is represented in the point-direction form
          !           ---- i.e., direction of the normal)
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          dir = refnod(3,i)
          delx = delval(1,point1) - delval(1,point2)
          dely = delval(2,point1) - delval(2,point2)
          delz = delval(3,point1) - delval(3,point2)
          a = delval(1,dir)
          b = delval(2,dir)
          c = delval(3,dir)
          dist = a*delx + b*dely + c*delz
          delval(1,i) = dist
          delder(1,point1) = delx
          delder(2,point1) = dely
          delder(3,point1) = delz
          !
       case(RATIO_TYPE)
          !
          !           ---- ratio of two distances
          !
          delta1 = refnod(1,i)
          delta2 = refnod(2,i)
          delval(1,i) = delval(1,delta1) / delval(1,delta2)
          !
       case(COMBI_TYPE, 81)
          !           ---- a linear combination of two scalars or vectors
          !
          delta1 = refnod(1,i)
          delta2 = refnod(2,i)
          delval(1,i) = delval(4,i)*delval(1,delta1) &
               + delval(5,i)*delval(1,delta2)
          if (deftyp(i) == 81) then
             delval(2,i) = delval(4,i)*delval(2,delta1) &
                  + delval(5,i)*delval(2,delta2)
             delval(3,i) = delval(4,i)*delval(3,delta1) &
                  + delval(5,i)*delval(3,delta2)
          endif

       !Puja [QC: 11/17]
       case(CECM_TYPE)
          ! Proton transfer channel
          ! modified center of excess charge.
          ! Peter H. Koenig, Nilanjan Ghosh, August 2004
          !  iat1 = refnod(1,i)
          !  iat2 = refnod(2,i)
            n1 = iat11(1)
            n2 = iat12(1)
            from = 'ASCEND'
            call PROTONMCEC(iat11, iat12, wt1, from, delval(1,i), delder(1,i))
       case(CEC2_TYPE)
          !  iat1 = refnod(1,i)
          !  iat2 = refnod(2,i)
          !  wt   = refnod(3,i)
            from = 'ASCEND'
            call PROTONMCECDA(iat21, iat22, wt2,from, delval(1,i), delder(1,i))
       case(VSUM_TYPE)
            delta1 = refnod(1,i)
            delta2 = refnod(2,i)
            delval(1,i) = delval(1,delta1) + delval(1,delta2)
            delval(2,i) = delval(2,delta1) + delval(2,delta2)
            delval(3,i) = delval(3,delta1) + delval(3,delta2)
       !Puja [QC: 11/17]

       case default
          !          else
          !
          !           ---- what else can it be ?
          !
          call wrndie (-5, '<RXNENE>',  &
               ' CHARMM internal error. wrong type code')
          !
       end select
       !
    enddo loop100
    !
    return
    !
  end subroutine ascend


  !===================================================================
  !          RXNSTT
  !===================================================================
  SUBROUTINE RXNSTT(DELVAL,RXNCNT,STTSTP,DELSTT,NMLSTT,LODEL,HIDEL, &
       DELDEL,TREELO,NRXNCR,RXNTRN,TRCFRQ,RXNTRC, &
       TRUNIT,DEFTYP)
    !
    !         Collect reaction coordinate statistics and write traces.
    !         This routine assumes a scalar reaction coordinate.
    !
    use number
    use exfunc
    use stream

    INTEGER RXNCNT, STTSTP, NRXNCR, NMLSTT(:), TREELO(:)
    INTEGER TRUNIT(:), TRCFRQ(:), RXNTRC(:), RXNTRN, DELSTT(:)
    INTEGER DEFTYP(:)
    real(chm_real)  DELVAL(5,*)
    real(chm_real)  LODEL(:), HIDEL(:), DELDEL(:)

    !
    integer i, trc, u, IR, J
    real(chm_real) delta
    LOGICAL INCLUD
    !
    !     ---- keep a count of the number of calls
    !
    rxncnt = rxncnt + 1
    !
    !     ---- collect statistics
    !

    !     Loop over reaction coordinates.  
    IF (RXNCNT  >  STTSTP) THEN
    ! DTM debug - was incorrect for higher dimension reaction coordinates
    !   J  = 0
       J = 1
       INCLUD = .TRUE.
       DO IR = 1, NRXNCR
          DELTA = DELVAL(1,TREELO(IR))
          IF (DELDEL(IR)  /=  zero .AND.  &
               DELTA  >=  LODEL(IR) .AND. DELTA  <  HIDEL(IR)) THEN
    ! DTM debug
    !         I = INT((DELTA-LODEL(IR))/DELDEL(IR)) + 1
             I = INT((DELTA-LODEL(IR))/DELDEL(IR))
             J = J + NMLSTT(IR)*I
          ELSE
             INCLUD = .FALSE.
          ENDIF
       ENDDO
       IF (INCLUD) DELSTT(J) = DELSTT(J) + 1
    ENDIF
    !
    !     ---- print nodes with the trace request
    !
    ! yj070912.bfx: current MD step = MDSTEP2 = rxncnt-1
    loop10: do i = 1, rxntrn
       ! yj-        if (mod(rxncnt,trcfrq(i)) == 0) then
       if (mod(rxncnt-1,trcfrq(i)) == 0) then
          trc = rxntrc(i)
          u = trunit(i)
          IF (IOLEV > 0) THEN
#if KEY_ROLLRXNCOR==1
             if(deftyp(trc)  ==  71 .or. deftyp(trc)  ==  72 .or. deftyp(trc) == 81) then
#else /**/
             if(deftyp(trc)  ==  71 .or. deftyp(trc)  ==  72) then
#endif 
                ! yj-         write (u,2) real(rxncnt), delval(1,trc), delval(2,trc),
                write (u,2) real(rxncnt-1), delval(1,trc), delval(2,trc), &
                     delval(3,trc)
             else
                ! yj-              write (u,1) real(rxncnt), delval(1,trc)
                write (u,1) real(rxncnt-1), delval(1,trc)
             endif
          ENDIF
1         format (f15.4,5x,f10.4)
2         format (f15.4,5x,f10.4,f10.4,f10.4)
       end if
    enddo loop10
    !
    return
  end subroutine rxnstt

  !===================================================================
  !          DESCEND
  !===================================================================
  SUBROUTINE DECEND(IR,UMFORM,KUMB,DELTA0)
    !
    !       ---- Descend into the branches to calculate derivatives
    !       ---- w.r.t. cartesians        ... Kottalam, June 89
    !
    use number
    use exfunc
    !
    use rxncom
    use consta

    !       ---- passed variables
    INTEGER ir
    real(chm_real)  KUMB(:),DELTA0(:)
    integer UMFORM(:)  ! BUG - was real??? /MH09/
    !
    !       ---- local variables
    !
    integer dstype, n, iat, point1, point2, dir, dir1, dir2,pointa,lpnt1,lpnt2
    integer delta1, delta2, dir3 , i, wt
    real(chm_real) a, b, c, din, da, db, dc, aa, bb, cc, ab, ac, bc
    real(chm_real) dx2, dy2, dz2, st, dist, invdist, delx, dely, delz,  &
         epsx, epsy, dumbrella
    real(chm_real) epsz, dot, dddd, ddist, theta, dabyst, dt
    real(chm_real) a1, b1, c1, a2, b2, c2, val2
    logical delang
    character(len=6) from
    !
    loop100: do i = TREELO(IR), TREEHI(IR)
       !
       dstype = deftyp(i)
       !
       select case (dstype)
       case(POINT_TYPE)
          !
          n = refnod(1,i)
          iat = refnod(2,i)
          wt = refnod(3,i)
          from = 'DECEND'
          call GETPNT (n, iataa(iat)%a, wtaa(iat)%a, from, delval(1,i), &
               delder(1,i))
          !
       case(DIREPTPT_TYPE)
          !
          !           ---- direction from a point to another point
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          a = delval(1,i)
          b = delval(2,i)
          c = delval(3,i)
          din = delder(1,point1)
          da = delder(1,i)
          db = delder(2,i)
          dc = delder(3,i)
          aa = a*a - one
          bb = b*b - one
          cc = c*c - one
          ab = a*b
          ac = a*c
          bc = b*c
          dx2 = din * (aa*da + ab*db + ac*dc)
          dy2 = din * (ab*da + bb*db + bc*dc)
          dz2 = din * (ac*da + bc*db + cc*dc)
          delder(1,point1) = - dx2
          delder(2,point1) = - dy2
          delder(3,point1) = - dz2
          delder(1,point2) = dx2
          delder(2,point2) = dy2
          delder(3,point2) = dz2
          !
       case(DIREDRDR_TYPE)
          !
          !           ---- direction as a cross product of two other directions
          !
          dir1 = refnod(1,i)
          dir2 = refnod(2,i)
          a = delval(1,i)
          b = delval(2,i)
          c = delval(3,i)
          a1 = delval(1,dir1)
          b1 = delval(2,dir1)
          c1 = delval(3,dir1)
          a2 = delval(1,dir2)
          b2 = delval(2,dir2)
          c2 = delval(3,dir2)
          din = delder(1,dir1)
          da = delder(1,i)
          db = delder(2,i)
          dc = delder(3,i)
          aa = a*a - one
          bb = b*b - one
          cc = c*c - one
          ab = a*b
          ac = a*c
          bc = b*c
          !           ARD 03-05-13 Changed sign of the derivative
          delder(1,dir1) = - din * ( da*(ac*b2-ab*c2) &
               + db*(bc*b2-bb*c2) + dc*(cc*b2-bc*c2) )
          delder(2,dir1) = - din * ( da*(aa*c2-ac*a2) &
               + db*(ab*c2-bc*a2) + dc*(ac*c2-cc*a2) )
          delder(3,dir1) = - din * ( da*(ab*a2-aa*b2) &
               + db*(bb*a2-ab*b2) + dc*(bc*a2-ac*b2) )
          delder(1,dir2) =   din * ( da*(ac*b1-ab*c1) &
               + db*(bc*b1-bb*c1) + dc*(cc*b1-bc*c1) )
          delder(2,dir2) =   din * ( da*(aa*c1-ac*a1) &
               + db*(ab*c1-bc*a1) + dc*(ac*c1-cc*a1) )
          delder(3,dir2) =   din * ( da*(ab*a1-aa*b1) &
               + db*(bb*a1-ab*b1) + dc*(bc*a1-ac*b1) )
          !
       case(ANGLDRDR_TYPE)
          !
          !           ---- angle between two directions
          !
          dir1  = refnod(1,i)
          dir2  = refnod(2,i)
          dir3  = refnod(3,i)
          theta = delval(1,i)
          da    = delder(1,i)
          st    = sin (theta)
          if (abs(st) < tenm5) then
             delang = i == ngraph+1 .and. UMFORM(IR) == 1
             dt =  delval(1,ngraph+1) - DELTA0(IR)
             if (delang .and. DELTA0(IR) == zero) then
                dabyst = two * KUMB(IR) * (one+dt*dt/six)
             else if (delang .and. DELTA0(IR) == pi) then
                dabyst = -two * KUMB(IR) * (one+dt*dt/six)
             else if (st > zero) then
                dabyst = da / tenm5
             else
                dabyst = - da / tenm5
             end if
          else
             dabyst = da / st
          end if
          if (theta > pi) dabyst = - dabyst
          delder(1,dir1) = - delval(1,dir2) * dabyst
          delder(2,dir1) = - delval(2,dir2) * dabyst
          delder(3,dir1) = - delval(3,dir2) * dabyst
          delder(1,dir2) = - delval(1,dir1) * dabyst
          delder(2,dir2) = - delval(2,dir1) * dabyst
          delder(3,dir2) = - delval(3,dir1) * dabyst
          delder(1,dir3) = zero
          delder(2,dir3) = zero
          delder(3,dir3) = zero
          !
       case(DISTPTPT_TYPE)
          !
          !           ---- distance between two points
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          dist   = delval(1,i)
          !           delder(*,point1) currently contains the vector difference
          !           between positions 1 and 2 (r1-r2) calculated in ASCEND
          delx   = delder(1,point1)
          dely   = delder(2,point1)
          delz   = delder(3,point1)
          if (dist > tenm5) then
             delder(1,point1) = delder(1,i) * delx / dist
             delder(2,point1) = delder(1,i) * dely / dist
             delder(3,point1) = delder(1,i) * delz / dist
          else
             delder(1,point1) = delder(1,i)
             delder(2,point1) = delder(1,i)
             delder(3,point1) = delder(1,i)
          end if
          delder(1,point2) = - delder(1,point1)
          delder(2,point2) = - delder(2,point1)
          delder(3,point2) = - delder(3,point1)
          !
       case(DISTPTLN_TYPE)
          !           ---- shortest distance between a point and a line
          !           ---- (The line is represented in the "point-direction" form)
          !
          !--- a is the point off the line
          !--- 1 is the first point of the line
          !--- 2 is the secnd point of the line
          pointa = refnod(1,i)
          lpnt1 = refnod(2,i)
          lpnt2 = refnod(3,i)
          dist = delval(1,i)
          invdist = delval(2,i)
          dumbrella = delder(1,i)
          if(invdist < zero) then
             ! When the point is on the line, there is a singularity in the
             ! derivative of the distance, the umbrella will have a singularity
             ! in its derivatives. We set to zero for this low liklihood event and 
             ! let the MD move us off the point. NB that when the umbrella is 
             ! centered on the line (d0 = 0) or the umbrella constant is zero,
             ! this derivative is correct. MFC and Gregg Beckham Dec09.
             delder(1,pointa) = zero
             delder(2,pointa) = zero
             delder(3,pointa) = zero
             delder(1,lpnt1)  = zero
             delder(2,lpnt1)  = zero
             delder(3,lpnt1)  = zero
             delder(1,lpnt2)  = zero
             delder(2,lpnt2)  = zero
             delder(3,lpnt2)  = zero
          else
             delder(1,pointa) = dumbrella*invdist*delder(1,pointa)
             delder(2,pointa) = dumbrella*invdist*delder(2,pointa)
             delder(3,pointa) = dumbrella*invdist*delder(3,pointa)
             delder(1,lpnt1)  = dumbrella*invdist*delder(1,lpnt1)
             delder(2,lpnt1)  = dumbrella*invdist*delder(2,lpnt1)
             delder(3,lpnt1)  = dumbrella*invdist*delder(3,lpnt1)
             delder(1,lpnt2)  = dumbrella*invdist*delder(1,lpnt2)
             delder(2,lpnt2)  = dumbrella*invdist*delder(2,lpnt2)
             delder(3,lpnt2)  = dumbrella*invdist*delder(3,lpnt2)
          endif

       case(DISTPTPL_TYPE)
          !
          !           ---- shortest distance between a point and a plane
          !
          point1 = refnod(1,i)
          point2 = refnod(2,i)
          dir = refnod(3,i)
          a = delval(1,dir)
          b = delval(2,dir)
          c = delval(3,dir)
          delx = delder(1,point1)
          dely = delder(2,point1)
          delz = delder(3,point1)
          ddist = delder(1,i)
          delder(1,point1) = a * ddist
          delder(2,point1) = b * ddist
          delder(3,point1) = c * ddist
          delder(1,point2) = - delder(1,point1)
          delder(2,point2) = - delder(2,point1)
          delder(3,point2) = - delder(3,point1)
          delder(1,dir) = delx * ddist
          delder(2,dir) = dely * ddist
          delder(3,dir) = delz * ddist
          !
       case(RATIO_TYPE)
          !
          !           ---- ratio of two distances
          !
          delta1 = refnod(1,i)
          delta2 = refnod(2,i)
          val2 = delval(1,delta2)
          if (val2 < tenm5 .and. val2 > zero) then
             val2 = tenm5
          else if (val2 > -tenm5 .and. val2 < tenm5) then
             val2 = -tenm5
          end if
          delder(1,delta1) = delder(1,i) / val2
          delder(1,delta2) = - delder(1,delta1) * delval(1,i)
          !
       case(COMBI_TYPE, 81)
          !
          !           ---- a linear combination of two scalars or vectors
          !
          delta1 = refnod(1,i)
          delta2 = refnod(2,i)
          delder(1,delta1) = delval(4,i) * delder(1,i)
          delder(1,delta2) = delval(5,i) * delder(1,i)
          if (dstype == 81) then
             delder(2,delta1) = delval(4,i) * delder(2,i)
             delder(2,delta2) = delval(5,i) * delder(2,i)
             delder(3,delta1) = delval(4,i) * delder(3,i)
             delder(3,delta2) = delval(5,i) * delder(3,i)
          endif

       !Puja [QC: 11/17]
       case(CECM_TYPE)
       !
       !     ---- Proton transfer channel
       !     modified center of excess charge.
       !     Peter H. Koenig, Nilanjan Ghosh, August 2004

          ! iat1 = refnod(1,i)
          ! iat2 = refnod(2,i)
          ! wt = refnod(3,i)
          from = 'DECEND'
          call PROTONMCEC(iat11,iat12,wt1,from,delval(1,i),delder(1,i))

       case(CEC2_TYPE)
       !
       !     ---- Proton transfer channel
       !     modified center of excess charge.
       !     additional term for coupled donor-acceptor
       !     Peter H. Koenig, August 2004
       !
          !  iat1 = refnod(1,i)
          !  iat2 = refnod(2,i)
          !  wt = refnod(3,i)
          from = 'DECEND'
          call PROTONMCECDA(iat21,iat22,wt2,from,delval(1,i),delder(1,i))

      case(VSUM_TYPE)
      !     ---- vector sum
      !     Peter H. Koenig
          delta1 = refnod(1,i)
          delta2 = refnod(2,i)
          delder(1,delta1) = delder(1,i)
          delder(2,delta1) = delder(2,i)
          delder(3,delta1) = delder(3,i)
          delder(1,delta2) = delder(1,i)
          delder(2,delta2) = delder(2,i)
          delder(3,delta2) = delder(3,i)

       !Puja [QC: 11/17]

          !
       end select
       !
    enddo loop100
    !
    return
    !
  end subroutine decend
  !
  subroutine GETPNT (n, iat, wt, from, xp, der)
    !       ---- calculates a point for use in rxncor
    use coord
    use deriv
    use number

    integer n, iat(1:n), i
    real(chm_real) wt(n)
    real(chm_real) xp(3), der(3)
    character(len=6) from
    real(chm_real) wtder(3)
    !
    if (from == 'ASCEND') then
       !
       xp(1) = zero
       xp(2) = zero
       xp(3) = zero
       do i = 1, n
          xp(1) = xp(1) + wt(i) * x(iat(i))
          xp(2) = xp(2) + wt(i) * y(iat(i))
          xp(3) = xp(3) + wt(i) * z(iat(i))
       enddo
       !
    else
       !
       if (q_vpress) then
          do i = 1, n
             wtder(1) = wt(i) * der(1)
             wtder(2) = wt(i) * der(2)
             wtder(3) = wt(i) * der(3)
             dx(iat(i)) = dx(iat(i)) + wtder(1)
             dy(iat(i)) = dy(iat(i)) + wtder(2)
             dz(iat(i)) = dz(iat(i)) + wtder(3)
        
             vpress(1) = vpress(1) - x(iat(i))*wtder(1)
             vpress(2) = vpress(2) - x(iat(i))*wtder(2)
             vpress(3) = vpress(3) - x(iat(i))*wtder(3)

             vpress(4) = vpress(4) - y(iat(i))*wtder(1)
             vpress(5) = vpress(5) - y(iat(i))*wtder(2)
             vpress(6) = vpress(6) - y(iat(i))*wtder(3)

             vpress(7) = vpress(7) - z(iat(i))*wtder(1)
             vpress(8) = vpress(8) - z(iat(i))*wtder(2)
             vpress(9) = vpress(9) - z(iat(i))*wtder(3)
          enddo
       else
          do i = 1, n
             dx(iat(i)) = dx(iat(i)) + wt(i) * der(1)
             dy(iat(i)) = dy(iat(i)) + wt(i) * der(2)
             dz(iat(i)) = dz(iat(i)) + wt(i) * der(3)
          enddo
       endif
       !
    end if
    !
    return
    !
  END subroutine getpnt


  subroutine print_struc(i,n,iataa)

    integer n,k,i
    type(chm_iarray) iataa(:)
    write(*,*)'iataa>i,n=',i,n
    write(*,*)(iataa(i)%a(k), k=1,n)
    return
  end subroutine print_struc


!--------Puja------- [QC: 11/17]

       SUBROUTINE PROTONMCEC(iat1, iat2, weight, from, coor, dcoor)

       use exfunc
       use coord
       use deriv

       ! atom numbers for heavy atoms and hydrogen atoms
       integer:: iat1(*), iat2(*)
       ! weights for heavy atoms
       real(chm_real):: weight(*)
       character(len=6) from
       real(chm_real):: coor(3), dcoor(3)
       integer:: ox,oxa,hy,hya
       integer:: no,nh

       !real(chm_real):: fsw, dfsw
       real(chm_real):: f_sw, df_sw
       real(chm_real):: dist
       real(chm_real):: rsw, dsw
       real(chm_real):: dd, delx, dely, delz

       rsw = weight(1)
       dsw = weight(2)


       ! number of heavy atoms and hydrogen atoms is
       ! contained in the first element of the atomlists
       no = iat1(1)
       nh = iat2(1)

       if (from=='ASCEND') then

       ! first compute the CEC
          coor(1)=0.0d0
          coor(2)=0.0d0
          coor(3)=0.0d0
          do ox=2,no+1
             oxa = iat1(ox)
             coor(1) = coor(1) - weight(ox+1) * x(oxa)
             coor(2) = coor(2) - weight(ox+1) * y(oxa)
             coor(3) = coor(3) - weight(ox+1) * z(oxa)
          enddo
          do hy=2,nh+1
             hya = iat2(hy)
             coor(1) = coor(1) + x(hya)
             coor(2) = coor(2) + y(hya)
             coor(3) = coor(3) + z(hya)
          enddo
          ! now compute corrections using
          ! bond vectors
          do ox=2,no+1
             oxa = iat1(ox)
             do hy=2,nh+1
                hya = iat2(hy)

                delx = x(hya)-x(oxa)
                dely = y(hya)-y(oxa)
                delz = z(hya)-z(oxa)

                dist       = delx**2+dely**2+delz**2
                dist = sqrt(dist)
                f_sw = fsw(dist,rsw,dsw)

                coor(1) = coor(1) - f_sw*delx
                coor(2) = coor(2) - f_sw*dely
                coor(3) = coor(3) - f_sw*delz
             enddo
          enddo
       else
 !
 !     GRADIENTS
 !
          do ox=2,no+1
             oxa     = iat1(ox)
             dx(oxa) = dx(oxa) - weight(ox+1) * dcoor(1)
             dy(oxa) = dy(oxa) - weight(ox+1) * dcoor(2)
             dz(oxa) = dz(oxa) - weight(ox+1) * dcoor(3)
          enddo
          do hy=2,nh+1
             hya = iat2(hy)
             dx(hya) = dx(hya) + dcoor(1)
             dy(hya) = dy(hya) + dcoor(2)
             dz(hya) = dz(hya) + dcoor(3)
          enddo
         ! Do correction using projection of
         ! bond vectors
          do ox=2,no+1
             oxa = iat1(ox)
             do hy=2,nh+1
                hya = iat2(hy)

                delx = x(hya)-x(oxa)
                dely = y(hya)-y(oxa)
                delz = z(hya)-z(oxa)

                dist       = delx**2+dely**2+delz**2

                dist  = sqrt(dist)
                f_sw  = fsw(dist,rsw,dsw)
                df_sw = dfsw(dist,rsw,dsw)

                dd = df_sw / dist

                dx(hya) = dx(hya)-dcoor(2)*delx*dd*dely-dcoor(3)*delx*dd*delz   &
                        - dcoor(1)*delx**2*dd-dcoor(1)*f_sw

                dy(hya) = dy(hya)-dcoor(1)*dely*dd*delx-dcoor(3)*dely*dd*delz   &
                        - dcoor(2)*dely**2*dd-dcoor(2)*f_sw

                dz(hya) = dz(hya)-dcoor(1)*delz*dd*delx-dcoor(2)*delz*dd*dely   &
                        - dcoor(3)*delz**2*dd-dcoor(3)*f_sw

                dx(oxa) = dx(oxa)+dcoor(2)*delx*dd*dely+dcoor(3)*delx*dd*delz   &
                        + dcoor(1)*delx**2*dd+dcoor(1)*f_sw

                dy(oxa) = dy(oxa)+dcoor(1)*dely*dd*delx+dcoor(3)*dely*dd*delz   &
                        + dcoor(2)*dely**2*dd+dcoor(2)*f_sw

                dz(oxa) = dz(oxa)+dcoor(1)*delz*dd*delx+dcoor(2)*delz*dd*dely   &
                        + dcoor(3)*delz**2*dd+dcoor(3)*f_sw

             enddo
          enddo
       endif

       RETURN
       END  SUBROUTINE PROTONMCEC



       SUBROUTINE PROTONMCECDA(iat1, iat2, weight, from, coor, dcoor)

       use exfunc
       use coord
       use deriv
       use stream

       ! atom numbers for heavy atoms and hydrogen atoms
       integer:: iat1(*), iat2(*)
       ! weights for heavy atoms
       real(chm_real)::  weight(*),coor(3), dcoor(3)
       character(len=6) from
       integer:: ox,oxa,hy,hya

       ! total number of heavy atoms and hydrogen atoms in the mCEC
       integer:: no,nh


      !external fsw, dfsw
      !real(chm_real):: fsw, dfsw
      real(chm_real):: f_sw, df_sw
      real(chm_real):: dist
      real(chm_real):: rsw, dsw
      real(chm_real):: mp(3), v(3)
      integer:: k, a2, at2
      real(chm_real):: wn, wd, fact, w1, w2, f1
      real(chm_real):: vsmall
      real(chm_real):: d1,d2,d3, g, g1, g2, g3
      vsmall = 1.0d-40
      rsw = weight(1)
      dsw = weight(2)
      no = iat1(1)
      nh = iat2(1)
      k = 15
      if (from=='ASCEND') then
         ! midpoint of two acceptor atoms
         mp(1) = 0.0d0
         mp(2) = 0.0d0
         mp(3) = 0.0d0
         ! first entry is the total number of heavy atoms
         do ox=2,no+1
            oxa = iat1(ox)
            mp(1) = mp(1) + 0.5 * x(oxa)
            mp(2) = mp(2) + 0.5 * y(oxa)
            mp(3) = mp(3) + 0.5 * z(oxa)
         enddo
!
!     Calculate reaction coordinate
!
         coor(1)= 0.0d0
         coor(2)= 0.0d0
         coor(3)= 0.0d0
         do ox=2,no+1
            oxa = iat1(ox)
            v(1) = mp(1) - x(oxa)
            v(2) = mp(2) - y(oxa)
            v(3) = mp(3) - z(oxa)
            wn = 0.0d0
            wd = 0.0d0
            do hy=2,nh+1
               hya = iat2(hy)
               d1 = x(hya)-x(oxa)
               d2 = y(hya)-y(oxa)
               d3 = z(hya)-z(oxa)
               dist = d1**2 + d2**2 + d3**2
               dist = sqrt(dist)
               f_sw = fsw(dist,rsw,dsw)
               wn   = wn + f_sw ** (k+1)
               wd   = wd + f_sw **  k
            enddo
            fact = 0.0d0
            if (wd>=vsmall) then
               fact = wn / wd
            endif
            coor(1) = coor(1) + fact * v(1)
            coor(2) = coor(2) + fact * v(2)
            coor(3) = coor(3) + fact * v(3)
         enddo

           return
        else
        ! calculate gradients
           mp(1) = 0.0d0
           mp(2) = 0.0d0
           mp(3) = 0.0d0
 
           do ox=2,no+1
              oxa = iat1(ox)
              mp(1) = mp(1) + 0.5d0 * x(oxa)
              mp(2) = mp(2) + 0.5d0 * y(oxa)
              mp(3) = mp(3) + 0.5d0 * z(oxa)
           enddo
           do ox=2,no+1
              oxa = iat1(ox)
              v(1) = mp(1) - x(oxa)
              v(2) = mp(2) - y(oxa)
              v(3) = mp(3) - z(oxa)
 
              wn = 0.0d0
              wd = 0.0d0
              do hy=2,nh+1
                 hya = iat2(hy)
 
                 d1 = x(hya)-x(oxa)
                 d2 = y(hya)-y(oxa)
                 d3 = z(hya)-z(oxa)
                 dist = d1**2 + d2**2 + d3**2
 
                 dist = sqrt(dist)
                 f_sw = fsw(dist,rsw,dsw)
                 w1   = f_sw **  k
                 w2   = f_sw *   w1
                 wn   = wn + w2
                 wd   = wd + w1
              enddo
              fact=0.0
              if (wd>=vsmall) then
                    fact = wn / wd
              endif
              dx(oxa) = dx(oxa) - fact * dcoor(1)
              dy(oxa) = dy(oxa) - fact * dcoor(2)
              dz(oxa) = dz(oxa) - fact * dcoor(3)
 
              do a2 = 2,no+1
                 at2 = iat1(a2)
                 dx(at2) = dx(at2) + 0.5d0 * fact * dcoor(1)
                 dy(at2) = dy(at2) + 0.5d0 * fact * dcoor(2)
                 dz(at2) = dz(at2) + 0.5d0 * fact * dcoor(3)
              enddo
              do hy=2,nh+1
                 hya = iat2(hy)
 
                 d1 = x(hya)-x(oxa)
                 d2 = y(hya)-y(oxa)
                 d3 = z(hya)-z(oxa)
                 dist = d1**2 + d2**2 + d3**2
 
                 dist = sqrt(dist)
                 f_sw = fsw(dist,rsw,dsw)
                 df_sw= dfsw(dist,rsw,dsw)
 
                 w1   = f_sw ** (k-1)
                 w2   = f_sw * w1
 
                 f1   = 0.0d0
                 if (wd.ge. vsmall) then
                    f1   = (wd*(k+1)*w2 - wn*k*w1 ) / (wd*wd)
                    f1   = f1 * df_sw
                    f1   = f1 / dist
                 endif
 
                 g1 = dcoor(1)*v(1)*f1
                 g2 = dcoor(2)*v(2)*f1
                 g3 = dcoor(3)*v(3)*f1
                 g  = g1 + g2 + g3
 
                 dx(hya) = dx(hya) + g * d1
                 dy(hya) = dy(hya) + g * d2
                 dz(hya) = dz(hya) + g * d3
 
                 dx(oxa) = dx(oxa) - g * d1
                 dy(oxa) = dy(oxa) - g * d2
                 dz(oxa) = dz(oxa) - g * d3
 
              enddo
           enddo
        endif
        RETURN
        END subroutine PROTONMCECDA
 
  !##############stuff for zeta coordinate##############
 
        real(chm_real) FUNCTION FSW(r, rsw, dsw)
          implicit none
          real(chm_real):: r,rsw,dsw
 
          fsw = 1.0d0 / ( 1.0d0 + exp ((r-rsw)/dsw))
 
          return
        END function FSW
 
        real(chm_real) FUNCTION DFSW(r, rsw, dsw)
          implicit none
          real(chm_real) :: r,rsw,dsw,e
 
          e = (exp ((r-rsw)/dsw))
          if (e .lt.1.0d99) then
             dfsw = -1.0d0 / ((1.0d0 + e )*(1.0d0 + e )) * e / dsw
          else
             dfsw = 0.0d0
          endif
 
          return
        END function  DFSW
 
 
 !--------Puja------- [QC: 11/17]

  
#endif /* (rxene_main)*/
  
end module rxenemod

