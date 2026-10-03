#if KEY_MODELLER==1
SUBROUTINE MDSET2

   use dimens_fcm
   use psf
   use comand
   use stream
   use string
   use select
   use coord
   use number
   use chutil,only:atomid
   use modeller
   use memory

   implicit none

   INTEGER           I,J,N,II,k
   INTEGER           IUNIT
   CHARACTER*8       SIDI,RIDI,RENI,ACI
   CHARACTER*8       SIDJ,RIDJ,RENJ,ACJ

   LOGICAL           DONE,EOF,OK,QANAL,QERR
   CHARACTER(len=4)  WRD
   integer,allocatable,dimension(:) :: islct,jslct
   INTEGER qat(8), nqat,qqat(8)
   REAL*8 tmpx(maxmin),tmpy(maxmin),sdy(maxmin)
   REAL*8 ytmp(maxmin),x1(maxmin),x2(maxmin),y2tmp(maxmin)
   real*8 x1t(0:maxmin),x2t(0:maxmin)
   real*8 yt(0:maxmin,0:maxmin),y1a(maxmin,maxmin),y2a(maxmin,maxmin),y12a(maxmin,maxmin)
   real*8 tmy(4),tmy1(4),tmy2(4),tmy12(4)
   integer v1,v2,v3,v4,jj
   real*8 c(4,4)

!021722
   logical lreset
   lreset = .FALSE.

   DONE = .FALSE.
   EOF  = .FALSE.
   OK   = .TRUE.


   call chmalloc('modeller.src','modeller','islct',natom,intg=islct)
   call chmalloc('modeller.src','modeller','jslct',natom,intg=jslct)

      DO WHILE(.NOT.DONE)
         CALL RDCMND(COMLYN,MXCMSZ,COMLEN,ISTRM,EOF,.TRUE.,.TRUE.,'MOD> ')

!
! This should not happen. for now there is no stream handling inside
! the mod module.
      IF(EOF)THEN
         CALL PPSTRM(OK)
         IF(.NOT.OK)  RETURN
         ENDIF
        
         WRD='    '
         WRD=NEXTA4(COMLYN,COMLEN)
         IF(WRD.EQ.'    ') THEN
            CONTINUE


         ELSE IF(WRD.EQ.'RESE') THEN

!021722
            lreset = .TRUE.

            QMODEL = .FALSE.
            call mod_uniniall()
!c 012110 initialize
            do i=1,numcatmd
              lperiod(i)=.false.
              ldspline(i)=.false.
              laspline(i)=.false.
              lrotspline(i)=.false.
            enddo
            
            numcatmd=0
!c            lper=.false.
!c            NOESCA=1.0
         else if(wrd.eq.'MGAU') then
         if (.not. md_initialized) call mod_iniall()

              numcatmd=numcatmd+1
              stoagmd(numcatmd,1)=mdnum+1

              lperiod(numcatmd)=.false.

         else if(wrd.eq.'MPER') then
         if (.not. md_initialized) call mod_iniall()

! Multiple periodic distribution
              numcatmd=numcatmd+1
              stoagmd(numcatmd,1)=mdnum+1
              lperiod(numcatmd)=.true.
         else if(wrd.eq.'DSPL') then
         if (.not. md_initialized) call mod_iniall()

! distance cubic spline
              numcatmd=numcatmd+1
              stoagmd(numcatmd,1)=mdnum+1
              ldspline(numcatmd)=.true.

         else if(wrd.eq.'ASPL') then
         if (.not. md_initialized) call mod_iniall()

! dihedral angle cubic spline
              numcatmd=numcatmd+1
              stoagmd(numcatmd,1)=mdnum+1
              laspline(numcatmd)=.true.

         else if(wrd.eq.'ROTS') then
         if (.not. md_initialized) call mod_iniall()

! rotamer potential (phi vs. xi1)/ (xi1 vs. xi2)
              numcatmd=numcatmd+1
              stoagmd(numcatmd,1)=mdnum+1
              lrotspline(numcatmd)=.true.

         ELSE IF(WRD.EQ.'ASSI') THEN
         if (.not. md_initialized) call mod_iniall()
         QMODEL = .TRUE.
              if(lperiod(numcatmd).or.laspline(numcatmd)) then

! select four atoms in a row 
               do i=1,4
                 CALL NXTATM(QAT,NQAT,1,COMLYN,COMLEN,ISLCT, SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
! atom number
                  qqat(i)=qat(1)

                   if(nqat.eq.0) then
                     write(6,*) 'No atom selected'
                     stop
                   endif

               enddo


              elseif(lrotspline(numcatmd)) then
! select eight atoms in a row 
               do i=1,8
                 CALL NXTATM(QAT,NQAT,1,COMLYN,COMLEN,ISLCT, SEGID,RESID,ATYPE,IBASE,NICTOT,NSEG,RES,NATOM)
! atom number
                  qqat(i)=qat(1)

                   if(nqat.eq.0) then
                    write(6,*) 'No atom selected'
                    stop
                   endif 

               enddo

              else
!! multiple gaussian etc. and distance spline
            CALL SELCTD(COMLYN,COMLEN,ISLCT,JSLCT,X,Y,Z,WMAIN,.TRUE.,  QERR)
              endif


!c number of assign 
            MDNUM=MDNUM+1

              FORmd(mdnum)=GTRMF(COMLYN,COMLEN,'FORC',10.d0)
              tnmin(mdnum)=GTRMI(COMLYN,COMLEN,'NMIN',1)
              tnpos(mdnum)=GTRMI(COMLYN,COMLEN,'NPOS',1)
              lder(mdnum)=GTRMF(COMLYN,COMLEN,'LDER',0.d0)
              hder(mdnum)=GTRMF(COMLYN,COMLEN,'HDER',0.d0)
              lval(mdnum)=GTRMF(COMLYN,COMLEN,'LVAL',5.d0)
              hval(mdnum)=GTRMF(COMLYN,COMLEN,'HVAL',10.d0)
              inte(mdnum)=GTRMF(COMLYN,COMLEN,'INTE',1.0d0)

!c parameters for rots
              sxmin(mdnum)=GTRMF(COMLYN,COMLEN,'XMIN',-180.d0)
              sxmax(mdnum)=GTRMF(COMLYN,COMLEN,'XMAX',180.d0)
              symin(mdnum)=GTRMF(COMLYN,COMLEN,'YMIN',-180.d0)
              symax(mdnum)=GTRMF(COMLYN,COMLEN,'YMAX',180.d0)
              sxint(mdnum)=GTRMF(COMLYN,COMLEN,'XINT',15.d0)
              syint(mdnum)=GTRMF(COMLYN,COMLEN,'YINT',15.d0)

!c a bug  ulsig was defined as integer
!c
              ulsig(mdnum)=GTRMF(COMLYN,COMLEN,'ULSIG',1.d0)
              lmhar(mdnum)=(INDXA(COMLYN, COMLEN,'MHAR').GT.0)
              lgmha(mdnum)=(INDXA(COMLYN, COMLEN,'GMHA').GT.0)
              loqua(mdnum)=(INDXA(COMLYN, COMLEN,'OQUA').GT.0)
              lsgar(mdnum)=(INDXA(COMLYN, COMLEN,'SGAR').GT.0)
!c single gaussian with soft asymtote
              lsgaa(mdnum)=(INDXA(COMLYN, COMLEN,'SGAA').GT.0)
!c dspline with soft asymtote
              ldspa(mdnum)=(INDXA(COMLYN, COMLEN,'DSPA').GT.0)
              lsper(mdnum)=(INDXA(COMLYN, COMLEN,'SPER').GT.0)
!c flat-bottom single periodic
              lfbsp(mdnum)=(INDXA(COMLYN, COMLEN,'FBSP').GT.0)
!c bicubic interpolation
              lcubi(mdnum)=(INDXA(COMLYN, COMLEN,'CUBI').GT.0)

!c for soft asymptote
              knem(mdnum)=GTRMF(COMLYN,COMLEN,'FMAX',1.d0)
              rsm(mdnum)=GTRMF(COMLYN,COMLEN,'RSWI',1.d0)
              expm(mdnum)=GTRMF(COMLYN,COMLEN,'EXPT',1.d0)
              pucm(mdnum)=GTRMF(COMLYN,COMLEN,'RMAX',1.d0)


!c read all of minimum targets
              if( (INDXA(COMLYN, COMLEN,'MIN').gt.0) ) then
              do i=1,tnmin(mdnum)
               mdmin(mdnum,i)=nextf(comlyn,comlen) 
              enddo
              endif
              if( (INDXA(COMLYN, COMLEN,'POS').gt.0) ) then

              if(lrotspline(numcatmd)) then

              do i=1,tnpos(mdnum)*tnpos(mdnum)
               mdpos(mdnum,i)=nextf(comlyn,comlen) 
              enddo

              else

              do i=1,tnpos(mdnum)
               mdpos(mdnum,i)=nextf(comlyn,comlen) 
              enddo

              endif 

              endif

              if((INDXA(COMLYN, COMLEN,'SIGM').gt.0).or. (INDXA(COMLYN, COMLEN,'WEIG').gt.0)) then
              do i=1,tnmin(mdnum)
               mdsig(mdnum,i)=nextf(comlyn,comlen) 
              enddo
              endif

            IF (mdNUM.GE.mdMAX) THEN
               CALL WRNDIE(0,'<MDSET>', 'Max number of Modeller energy restraints exceeded')
               stop
               GOTO 120
            ENDIF

!C i 
            if(lperiod(numcatmd).or.laspline(numcatmd)) then
            MDIPT(MDNUM)=mdNM2+1
            mdINM(mdNUM)=0
            N=mdNM2
             do i=1,4
                  IF(MDNM2.GE.mdMX2) THEN
                     CALL WRNDIE(0,'<MDSET>', 'Max number of MD atoms exceeded1')
                     mdNUM=mdNUM-1
                     mdNM2=N
                     stop
                     GOTO 120
                  ENDIF
                  mdNM2=mdNM2+1
                  mdLIS(mdNM2)=qqat(i)
                  mdINM(mdNUM)=mdINM(mdNUM)+1
             enddo

            elseif(lrotspline(numcatmd)) then
            MDIPT(MDNUM)=mdNM2+1
            mdINM(mdNUM)=0
            N=mdNM2
             do i=1,8
                  IF(MDNM2.GE.mdMX2) THEN
                     CALL WRNDIE(0,'<MDSET>', 'Max number of MD atoms exceeded2')
                     mdNUM=mdNUM-1
                     mdNM2=N
                     stop
                     GOTO 120
                  ENDIF
                  mdNM2=mdNM2+1
                  mdLIS(mdNM2)=qqat(i)
                  mdINM(mdNUM)=mdINM(mdNUM)+1
             enddo

!c j
            else

            MDIPT(MDNUM)=mdNM2+1
            mdINM(mdNUM)=0
            N=mdNM2
            DO I=1,NATOM
               IF(ISLCT(I).EQ.1) THEN
                  IF(MDNM2.GE.mdMX2) THEN
                     CALL WRNDIE(0,'<MDSET>',   'Max number of MD atoms exceeded3')
                     mdNUM=mdNUM-1
                     mdNM2=N
                     stop
                     GOTO 120
                  ENDIF
                  mdNM2=mdNM2+1
                  mdLIS(mdNM2)=I
                  mdINM(mdNUM)=mdINM(mdNUM)+1
               ENDIF
            ENDDO

            mdJPT(mdNUM)=mdNM2+1
            mdJNM(mdNUM)=0
            DO J=1,NATOM
               IF(JSLCT(J).EQ.1) THEN
                  IF(mdNM2.GE.mdMX2) THEN
                     CALL WRNDIE(0,'<MDSET>','Max number of MD atoms exceeded4')
                     mdNUM=mdNUM-1
                     mdNM2=N
                     stop
                     GOTO 120
                  ENDIF
                  mdNM2=mdNM2+1
                  mdLIS(mdNM2)=J
                  mdJNM(mdNUM)=mdJNM(mdNUM)+1
               ENDIF
            ENDDO

            endif

            if(lperiod(numcatmd).or.laspline(numcatmd).or. lrotspline(numcatmd)) then

! if periodic distribution, it has single selection
            IF(mdINM(mdNUM).EQ.0) THEN
               CALL WRNDIE(0,'<MDSET>',  'Zero atom selected for this restraint. Ignored.')
               mdNUM=mdNUM-1
               mdNM2=N
               stop
               GOTO 120
            ENDIF

            else

!! if multiple gaussian, it has double selections
            IF(mdINM(mdNUM).EQ.0 .or. mdjnm(mdnum).eq.0) THEN
               CALL WRNDIE(0,'<MDSET>', 'Zero atom selected for this restraint. Ignored.')
               mdNUM=mdNUM-1
               mdNM2=N
               stop
               GOTO 120
            ENDIF

            endif

            if(ldspline(numcatmd).or.laspline(numcatmd)) then

!! distance and dihedral angle cubic spline
!! get the second derivatives for cubic spline interpolation
             do i=1,tnpos(mdnum)
               tmpx(i)=lval(mdnum)+(i-1)*inte(mdnum)
               tmpy(i)=mdpos(mdnum,i)
             enddo
             call mspline(tmpx,tmpy,tnpos(mdnum),lder(mdnum), hder(mdnum),sdy)
             do i=1,tnpos(mdnum)
               smdpos(mdnum,i)=sdy(i)
             enddo

            endif


            if(lrotspline(numcatmd)) then
 
!c new method bicubic interpolation. it's faster  
             if(.not.lcubi(mdnum)) then

              cdnum=cdnum+1  
              md2cd(mdnum)=cdnum
              if(cdnum.gt.mdmaxt) then
          write(OUTU,*) 'cdnum ',cdnum,' is larger than mdmaxt',mdmaxt
                 stop
              endif

              do i=0,tnpos(mdnum)+1
               x1t(i)=sxmin(mdnum)+(i-1)*sxint(mdnum) 
              enddo
              do j=0,tnpos(mdnum)+1
               x2t(j)=symin(mdnum)+(j-1)*syint(mdnum)
              enddo

              do i=1,tnpos(mdnum)
               do j=1,tnpos(mdnum)
                ii=(i-1)*tnpos(mdnum)+j  
                yt(i,j)=mdpos(mdnum,ii)  
               enddo
              enddo  

! expansion
              do i=0,0
               do j=1,tnpos(mdnum)
                yt(i,j)=yt(tnpos(mdnum),j)
               enddo
              enddo
              do i=tnpos(mdnum)+1,tnpos(mdnum)+1
               do j=1,tnpos(mdnum)
                yt(i,j)=yt(1,j)
               enddo
              enddo
              do j=0,0
               do i=1,tnpos(mdnum)
                yt(i,j)=yt(i,tnpos(mdnum))
               enddo
              enddo
              do j=tnpos(mdnum)+1,tnpos(mdnum)+1
               do i=1,tnpos(mdnum)
                yt(i,j)=yt(i,1)
               enddo
              enddo
              yt(0,0)=yt(tnpos(mdnum),tnpos(mdnum))
              yt(tnpos(mdnum)+1,tnpos(mdnum)+1)=yt(0,0)
    
              do j=1,tnpos(mdnum)
               do k=1,tnpos(mdnum)
                y1a(j,k)=(yt(j+1,k)-yt(j-1,k))/(x1t(j+1)-x1t(j-1))
                y2a(j,k)=(yt(j,k+1)-yt(j,k-1))/(x2t(k+1)-x2t(k-1))
                y12a(j,k)=(yt(j+1,k+1)-yt(j+1,k-1)-yt(j-1,k+1)+yt(j-1,k-1))/((x1t(j+1)-x1t(j-1))*(x2t(k+1)-x2t(k-1)))
               enddo
              enddo

!c get coefficients
      do i=1,tnpos(mdnum)
         do j=1,tnpos(mdnum)
            v1=i
            v2=i+1
            v3=j
            v4=j+1
            tmy(1)   = yt(v1,v3)
            tmy(2)   = yt(v2,v3)
            tmy(3)   = yt(v2,v4)
            tmy(4)   = yt(v1,v4)
            tmy1(1)  = y1a(v1,v3)
            tmy1(2)  = y1a(v2,v3)
            tmy1(3)  = y1a(v2,v4)
            tmy1(4)  = y1a(v1,v4)
            tmy2(1)  = y2a(v1,v3)
            tmy2(2)  = y2a(v2,v3)
            tmy2(3)  = y2a(v2,v4)
            tmy2(4)  = y2a(v1,v4)
            tmy12(1) = y12a(v1,v3)
            tmy12(2) = y12a(v2,v3)
            tmy12(3) = y12a(v2,v4)
            tmy12(4) = y12a(v1,v4)

            call bcucof(tmy,tmy1,tmy2,tmy12,x1t(v2)-x1t(v1), x2t(v4)-x2t(v3),c)

            do ii=1,4
               do jj=1,4
                  ccl(cdnum,i,j,ii,jj)=c(ii,jj)
               enddo
            enddo

         enddo
      enddo

             else

!c get the second derivatives for bicubic spline interpolation
!c (old method; bicubic spline interpolation)

            do i=1,tnpos(mdnum)
              x1(i)=sxmin(mdnum)+(i-1)*sxint(mdnum) 
            enddo
            do j=1,tnpos(mdnum)
              x2(j)=symin(mdnum)+(j-1)*syint(mdnum)
            enddo

!! phi
            do i=1,tnpos(mdnum)
             do j=1,tnpos(mdnum)
               ii=(i-1)*tnpos(mdnum)+j 
               ytmp(j)=mdpos(mdnum,ii)
             enddo
             call mspline(x2,ytmp,tnpos(mdnum),1.d30,1.d30,y2tmp)
             do j=1,tnpos(mdnum)
               ii=(i-1)*tnpos(mdnum)+j 
               ytwo(mdnum,ii)=y2tmp(j)
             enddo 
            enddo

!! chi1
            do j=1,tnpos(mdnum)
             do i=1,tnpos(mdnum)
               ii=(i-1)*tnpos(mdnum)+j 
               ytmp(i)=mdpos(mdnum,ii)
             enddo
             call mspline(x1,ytmp,tnpos(mdnum),1.d30,1.d30,y2tmp)
             do i=1,tnpos(mdnum)
               ii=(i-1)*tnpos(mdnum)+j 
               ytwo2(mdnum,ii)=y2tmp(i)
             enddo 
            enddo

            endif

            endif


            IF(PRNLEV.GE.2) THEN
                if(lperiod(numcatmd).or.laspline(numcatmd) .or.lrotspline(numcatmd)) then

                WRITE(OUTU,'(2(A,1X,I5))')'  MD: ADDING RESTRAINT #',MDNUM,', # atoms 1st set',MDINM(mdNUM)

                else
!! multiple gaussian and distance spline
                WRITE(OUTU,'(3(A,1X,I5))')'  MD: ADDING RESTRAINT #',MDNUM, ', # atoms 1st set',MDINM(mdNUM), ', # atoms 2nd set',MDjNM(mdNUM)

                endif

 511                FORMAT('        FIRST SET ATOM:',I5,3(1X,A))
                if(lperiod(numcatmd).or.laspline(numcatmd).or. lrotspline(numcatmd)) then
!C i
                 DO I=1,mdINM(mdNUM)
                    II=mdLIS(mdIPT(mdNUM)+I-1)
                    CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
                    WRITE(OUTU,511) II,SIDI(1:idleng), RIDI(1:idleng),ACI(1:idleng)
                 ENDDO

                else
!C i
                 DO I=1,mdINM(mdNUM)
                    II=mdLIS(mdIPT(mdNUM)+I-1)
                    CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
                    WRITE(OUTU,511) II,SIDI(1:idleng),  RIDI(1:idleng),ACI(1:idleng)
                 ENDDO
!C j
                 DO I=1,mdJNM(mdNUM)
                    II=mdLIS(mdJPT(mdNUM)+I-1)
                    CALL ATOMID(II,SIDI,RIDI,RENI,ACI)
                    WRITE(OUTU,512) II,SIDI(1:idleng),  RIDI(1:idleng),ACI(1:idleng)
 512                FORMAT('       SECOND SET ATOM:',I5,3(1X,A))
                 ENDDO

                endif 

                 if (lperiod(numcatmd)) then
!!multiple period 

!!single gaussian (sper)
                 WRITE(OUTU,515) FORmd(mdNUM),tnmin(mdNUM)

                 else if (ldspline(numcatmd) .or.laspline(numcatmd).or.lrotspline(numcatmd)) then

                 WRITE(OUTU,516) FORmd(mdNUM),tnpos(mdNUM)
 516             FORMAT('   FORC=',F10.3,'  NPOS=',i4)

                 else 
!multiple gaussian
                 WRITE(OUTU,515) FORmd(mdNUM),tnmin(mdNUM)
 515             FORMAT('   FORC=',F10.3,'  NMD=',i4)
                 endif

            ENDIF
 120        CONTINUE
         ELSE IF(WRD.EQ.'PRIN') THEN
            QANAL=(INDXA(COMLYN,COMLEN,'ANAL').GT.0)
            lanamd=qanal
            iUNIjmd=GTRMI(COMLYN,COMLEN,'UNIT',OUTU)
 
            if(qanal) then
            if(prnlev.ge.5) then
            write(6,*) 'CAT  '
            do i=1,numcatmd
             write(6,*) 'hiru',stoagmd(i,1),stoagmd(i,2)
            enddo
            write(6,*) 'mdnum',mdnum
            do i=1,mdnum
             write(6,*) 'MD : Tn sel.atoms : init_index', i,mdinm(i),mdipt(i),mdjnm(i),mdipt(i)
            do j=1,mdinm(i)
             write(6,*) 'selected atom number',mdlis(mdipt(i)+j-1)
            enddo
            do j=1,mdjnm(i)
             write(6,*) 'selected atom number',mdlis(mdjpt(i)+j-1)
            enddo
             write(6,*) 'Force=',formd(i),'Number of min= ',tnmin(i)
            do j=1,tnmin(i)
             write(6,*) 'min=', mdmin(i,j),'sigma=',mdsig(i,j)
            enddo
            enddo
            endif
            endif
!C
         ELSE IF(WRD.EQ.'READ') THEN
!c fixing
            IUNIT=GTRMI(COMLYN,COMLEN,'UNIT',ISTRM)
            CALL NOEREA(IUNIT)
         ELSE IF(WRD.EQ.'END ') THEN
!021722 debug
            !stoagmd(numcatmd,2)=mdnum
            if( .not. lreset ) stoagmd(numcatmd,2)=mdnum

            DONE=.TRUE.
            
         ELSE
            CALL WRNDIE(0,'<MDSET>','UNKNOWN OPTION')
         ENDIF

      ENDDO
!C
 220  CONTINUE
      IF(PRNLEV.GE.2) WRITE(OUTU,230) MDNUM
 230  FORMAT(' MD:  CURRENT NUMBER OF CONSTRAINTS=',I4)

!c use csnum to define dynamics allocation of the following array
!c ssnmr.fcm (csmax and csmx2 term)

 
       write(6,*) 'CAT  '
       do i=1,numcatmd
       enddo

       if(prnlev.ge.6) then
         write(6,*) 'mdnum',mdnum
         do i=1,mdnum
           write(6,*) 'MD : Tn sel.atoms : init_index', i,mdinm(i),mdipt(i),mdjnm(i),mdjpt(i)
           do j=1,mdinm(i)
           write(6,*) 'selected atom number',mdlis(mdipt(i)+j-1)
           enddo
           do j=1,mdjnm(i)
           write(6,*) 'selected atom number',mdlis(mdjpt(i)+j-1)
           enddo
          enddo
        endif
   IF (DONE) call chmdealloc('modeller.src','modeller','islct',natom,intg=islct)
!c
RETURN
END SUBROUTINE MDSET2


      SUBROUTINE MDCNS(EN,DX,DY,DZ,X,Y,Z, FORmd,mdnum,mdipt,mdjpt,&
                       mdinm,mdjnm,mdlis,&
                       lanamd,stoagmd,numcatmd,&
                       iunijmd,&
                       tnmin,mdmin,mdsig,&
                       lperiod,ulsig,lmhar,&
                       lgmha,loqua,&
                       lsgar,lsper,lsgaa,ldspa,&
                       rsm,knem,expm,pucm,&
                       ldspline,laspline,&
                       lrotspline,&
                       mdpos,tnpos,smdpos,&
                       lder,hder,lval,hval,inte,&
                       sxmin,sxmax,symin,symax,&
                       sxint,syint,ytwo,ytwo2,&
                       lfbsp, &
                       ccl,lcubi,md2cd )
!C
!C author: Jinhyuk Lee
!C
   use number
   use dimens_fcm
   use stream
#if KEY_PARALLEL==1
   use parallel
#endif
   use consta
   use memory
   use vector
   use chutil, only: atomid
   use param_store, only: set_param

   implicit none
   real(chm_real) en
   real(chm_real) X(*),Y(*),Z(*)
   real(chm_real) DX(*),DY(*),DZ(*)

   integer mdnum,mdnm2,mdipt(*),mdinm(*),mdlis(*),mdjpt(*),mdjnm(*),numcat

! must be in agreement with values in modeller_ltm.F90
   integer mdmax, mdmx2, maxncmd, maxmin, mdmaxt, maxmint
   parameter(mdmax = 50000, mdmx2 = 150000, maxncmd = 200, maxmin = 1089, mdmaxt = 20000, maxmint = 34)
   real*8 ccl(mdmaxt,maxmint,maxmint,4,4)

   logical lcubi(mdmax) ! or lcubi(*)
   integer md2cd(mdmax),cdnum

   real*8 formd(*),mdmin(mdmax,maxmin),mdsig(mdmax,maxmin)
   real*8 ulsig(*)
   logical lanamd
   integer i,ii,jj,j,k,fatom,satom
   integer nnum,hnum,onum
   integer stoagmd(maxncmd,2),numcatmd,tnmin(*)
   integer std

   logical lperiod(*),lmhar(*),lgmha(*),loqua(*)
   logical lsgar(*),lsper(*),lsgaa(*),ldspa(*),lfbsp(*)

   ! for soft asymptote
   real*8 rsm(*),knem(*),expm(*),pucm(*)
   real*8 asymal,asymbl,asymau,asymbu,prek
   logical ldspline(*),laspline(*),lrotspline(*)
   integer tnpos(*)
   real*8 mdpos(mdmax,maxmin),lder(mdmax),hder(mdmax), lval(mdmax),hval(mdmax),inte(mdmax),&
         smdpos(mdmax,maxmin)
   real*8 sxmin(mdmax),sxmax(mdmax),symin(mdmax), &
            symax(mdmax),sxint(mdmax),syint(mdmax),&
            ytwo(mdmax,maxmin),ytwo2(mdmax,maxmin)
   real*8 ytmp(maxmin),x1(maxmin),x2(maxmin),y2tmp(maxmin)
   integer m,n,o,p,kk,iii,ii2,jj2
   integer am,an,ao,ap
   real*8 rm(3),rn(3),ro(3),rp(3)
   real*8 rmn(3),ron(3),rop(3),rmo(3),rnp(3)
   real*8 rmnon(3),ronop(3),dott,dott1,dott2
   real*8 xi1,xi2,dott3
   real*8 rety2(maxmin),retdy2(maxmin)
   real*8 retyx1,retdyx1,retyx2,retdyx2,invn
   real*8 sx,sy
   integer fat(8),tmpa,ai,aj,ak,al
   real*8 ri(3),rj(3),rk(3),rl(3)
   real*8 rij(3),rkj(3),rkl(3),rik(3),rjl(3)
   real*8 rijkj(3),rkjkl(3),dot,dot1,dot2
   real*8 aa(3),bb(3)
   real*8 rkjaa(3),rikaa(3),rklbb(3),rjlbb(3)
   real*8 rijaa(3),rijbb(3)
   real*8 mrkj,drijkj,drklkj
   real*8 xi,r3(3),dot3,signxi,oxi
   real*8 tdi(3),tdj(3),tdk(3),tdl(3)

   CHARACTER*8 SIDI,RIDI,RENI,ACI
   character*4 atna

   real*8 enemd,dist,sumexp,tmpe,mdsigtmp,sumexpm,pre
   real*8 jix,jiy,jiz
   real*8 dxi,dyi,dzi,dxj,dyj,dzj
   real*8 sdipxi,sdipyi,sdipzi,sdipxj,sdipyj,sdipzj
   real*8 dpxi,dpyi,dpzi,dpxj,dpyj,dpzj
   real*8 front,after1,after2
   real*8 tmpsum

! Print analysis
   integer iunijmd,ll,mm
! Sort array
   integer io,itag(maxmin),optmindex
   real*8 tmpsort(maxmin),avetmp
   real*8 tmpx(maxmin),tmpy(maxmin),tmpy2(maxmin),sdy(maxmin)
   real*8 rety,retdy
   real*8 rtemp,tmpp
   real*8 bbb,mmm,aaa,ppp,dcl,dcu,lff,uff,dlff,duff

   integer tmpi
   integer buf
   real*8  xin2,yin2
   integer v1,v2,v3,v4
   real*8 t,u,ry,dy1,dy2

   real*8 ddd1,ddd2,ddd3
! Initialize
      enemd=0.d0

!      write(6,*) 'hi'

!c number of divided md
      do k=1,numcatmd

!cdebug
       if(prnlev.ge.6) then
!c        write(6,*) 'Parameters for CS and DC'
!c        write(6,*) 'cat S11 S22 S33 phics nudc numres '
!c        write(6,'(i2,5f7.3,i3)') k,ms11,ms22,ms33,phics,nudc,
!c     &                           stoag(k,2)-stoag(k,1)+1
        write(6,*) 'Num of Cat',stoagmd(k,2)-stoagmd(k,1)+1
       endif

!c     number of MD
!c        write(6,*) 'stoagmd',stoagmd(k,1),
!c     &     stoagmd(k,2),lperiod(k),laspline(k),ldspline(k)

!! big big if
!! 011810
       if(lperiod(k)) then
!! periodic distribution
       do i=stoagmd(k,1),stoagmd(k,2)
!c mdinm(i) must be same to mdjnm(i)
        if(prnlev.ge.6) write(6,*) 'Multiple Periodic Distribution'
        do j=1,mdinm(i)
          fat(j)=mdlis(mdipt(i)+j-1)
          call atomid(fat(j),sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) j,fat(j),sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
        enddo

!! change i and j each other to become agreement with dihedral angles in Modeller
!! atomnumber
!! i
!!       ai=fat(1)
!!       ri(1)=x(ai)
!!       ri(2)=y(ai)
!!       ri(3)=z(ai)
       ai=fat(4)
       ri(1)=x(ai)
       ri(2)=y(ai)
       ri(3)=z(ai)
!! j
       aj=fat(2)
       rj(1)=x(aj)
       rj(2)=y(aj)
       rj(3)=z(aj)
!! k
       ak=fat(3)
       rk(1)=x(ak)
       rk(2)=y(ak)
       rk(3)=z(ak)
!! l
!!       al=fat(4)
!!       rl(1)=x(al)
!!       rl(2)=y(al)
!!       rl(3)=z(al)
       al=fat(1)
       rl(1)=x(al)
       rl(2)=y(al)
       rl(3)=z(al)

!!       write(6,*) rl(1),rl(2),rl(3)
!!dihedral angle
!! xi=sign(xi)acos((rijXrkj)d(rkjXrkl)/|rijXrkj||rkjXrkl|)
!! xi=sign(xi)acos((rijkj)d(rkjkl)/|rijkj||rkjkl|)
!! xi=sign(xi)acos(dot/dot1/dot2)
       do j=1,3        
        rij(j)=rj(j)-ri(j)
        rkj(j)=rj(j)-rk(j)
        rkl(j)=rl(j)-rk(j)
        rik(j)=rk(j)-ri(j)
        rjl(j)=rl(j)-rj(j)
       enddo
       call cross3(rij,rkj,rijkj)
       call cross3(rkj,rkl,rkjkl)
       call dotpr(rijkj,rkjkl,3,dot)
       call dotpr(rijkj,rijkj,3,dot1)
       dot1=sqrt(dot1)
       call dotpr(rkjkl,rkjkl,3,dot2)
       dot2=sqrt(dot2)

       xi=acos(dot/dot1/dot2)

!! 012110
       if((dot/dot1/dot2).le.-1.d0) then
        xi=acos(-1.d0)
       elseif((dot/dot1/dot2).ge.1.d0) then
        xi=acos(1.d0)
       endif
!!       write(6,*) 'hi',xi,acos(dot/dot1/dot2)
       oxi=xi

!! for agreement of dihedral sign with modeller definition
!!       call cross3(rkjkl,rijkj,r3)
       call cross3(rijkj,rkjkl,r3)
       call dotpr(rkj,r3,3,dot3)

       if (dot3.ge.0.d0) then
         signxi=1.d0
       else
         signxi=-1.d0
       endif

       xi=signxi*xi
!c       if(xi.le.0.d0) then
!c        xi=xi+pi
!c       else
!c        xi=xi-pi
!c       endif 

       if(prnlev.ge.6) write(6,*) 'xi',signxi,xi*180.d0/pi,dot/dot1/dot2,acos(dot/dot1/dot2),acos(-1.d0)

!! single gaussian (sper) dihedral angle potential
        if(lsper(i)) then
         if(prnlev.ge.6) write(6,*) 'Single Gaussian Dihedral Angle Restraint Potential'

         if(tnmin(i).ge.2) then
          write(6,*) 'The number of min and sigma must be one in sgar'
          stop
         endif
!c        write(6,*) i,tnmin(i),one,two,pi
         do ii=1,tnmin(i)
!c see modeller manual (A.62) with removing constant term
!c kbT is necessary kb: boltzman , T: room temp 297.15K in modeller manual 9v7
          rtemp=297.15d0
          tmpe=KBOLTZ*rtemp*one/two*((xi-mdmin(i,ii))/mdsig(i,ii))**two
!cc     &        -log(one/mdmin(i,ii)/sqrt(two*pi))
         enddo
!! forces starting
         ii=1
         tmpp=KBOLTZ*rtemp*(xi-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii)
!!herehere
!! method (A.55)~(A.60)
!!forces 
!! d xi / dxi
       call dotpr(rkj,rkj,3,mrkj)
       mrkj=sqrt(mrkj) 
       tdi(1)=mrkj/dot1**2.d0*rijkj(1)
       tdi(2)=mrkj/dot1**2.d0*rijkj(2)
       tdi(3)=mrkj/dot1**2.d0*rijkj(3)
!! d xi / dxl
       tdl(1)=-1.d0*mrkj/dot2**2.d0*rkjkl(1)
       tdl(2)=-1.d0*mrkj/dot2**2.d0*rkjkl(2)
       tdl(3)=-1.d0*mrkj/dot2**2.d0*rkjkl(3)
!! d xi / dxj
       call dotpr(rij,rkj,3,drijkj)
       call dotpr(rkl,rkj,3,drklkj)
       tdj(1)=(drijkj/mrkj**2.d0-1.d0)*tdi(1)-drklkj/mrkj**2.d0*tdl(1)
       tdj(2)=(drijkj/mrkj**2.d0-1.d0)*tdi(2)-drklkj/mrkj**2.d0*tdl(2)
       tdj(3)=(drijkj/mrkj**2.d0-1.d0)*tdi(3)-drklkj/mrkj**2.d0*tdl(3)
!! d xi / dxk
       tdk(1)=(drklkj/mrkj**2.d0-1.d0)*tdl(1)-drijkj/mrkj**2.d0*tdi(1)
       tdk(2)=(drklkj/mrkj**2.d0-1.d0)*tdl(2)-drijkj/mrkj**2.d0*tdi(2)
       tdk(3)=(drklkj/mrkj**2.d0-1.d0)*tdl(3)-drijkj/mrkj**2.d0*tdi(3)

!! add the calculated forces to the originals
       dx(ai)=dx(ai)-tmpp*tdi(1)
       dy(ai)=dy(ai)-tmpp*tdi(2)
       dz(ai)=dz(ai)-tmpp*tdi(3)
       dx(aj)=dx(aj)-tmpp*tdj(1)
       dy(aj)=dy(aj)-tmpp*tdj(2)
       dz(aj)=dz(aj)-tmpp*tdj(3)
       dx(ak)=dx(ak)-tmpp*tdk(1)
       dy(ak)=dy(ak)-tmpp*tdk(2)
       dz(ak)=dz(ak)-tmpp*tdk(3)
       dx(al)=dx(al)-tmpp*tdl(1)
       dy(al)=dy(al)-tmpp*tdl(2)
       dz(al)=dz(al)-tmpp*tdl(3)

!!        if(lsper(i)) then
!! 051211
!! fbsp - flat-bottom single dihedral potential
        else if (lfbsp(i)) then
         if(prnlev.ge.6) then 
          write(6,*) 'Flat bottom periodic potential'
          write(6,*) 'b=',formd(i),'m=',mdmin(i,1),'a=',mdsig(i,1),'v=',xi
          ddd1=abs(mdmin(i,1)-xi)
          ddd2=abs(mdmin(i,1)-(xi-2.d0*pi))
          ddd3=abs(mdmin(i,1)-(xi+2.d0*pi))
          if(ddd1.le.ddd2) then
           if(ddd1.le.ddd3) then
             if(ddd1.le.mdsig(i,1)) then
       write(6,*) 'diffa',0.0d0
             else
       write(6,*) 'diffa',(ddd1-mdsig(i,1))*180.d0/pi            
             endif
           else
             if(ddd3.le.mdsig(i,1)) then
       write(6,*) 'diffa',0.0d0
             else
       write(6,*) 'diffa',(ddd3-mdsig(i,1))*180.d0/pi            
             endif
           endif
          else
           if(ddd2.le.ddd3) then
             if(ddd2.le.mdsig(i,1)) then
       write(6,*) 'diffa',0.0d0
             else
       write(6,*) 'diffa',(ddd2-mdsig(i,1))*180.d0/pi            
             endif
           else
             if(ddd3.le.mdsig(i,1)) then
       write(6,*) 'diffa',0.0d0
             else
       write(6,*) 'diffa',(ddd3-mdsig(i,1))*180.d0/pi            
             endif
           endif
          endif
         endif

         bbb=formd(i)
         mmm=mdmin(i,1)
         aaa=mdsig(i,1)
         ppp=2.d0*(pi-aaa)
         dcl=mmm+aaa-pi
         dcu=aaa-pi-mmm
         lff=bbb*cos(pi/(pi-aaa)*(xi-mmm+pi))+bbb
         uff=bbb*cos(pi/(pi-aaa)*(xi-mmm-pi))+bbb
!! force values
         dlff=bbb*sin(pi/(pi-aaa)*(xi-mmm+pi))*pi/(pi-aaa)
         duff=bbb*sin(pi/(pi-aaa)*(xi-mmm-pi))*pi/(pi-aaa)

!! energy and force
         if(dcu.ge.0.d0) then
           if(xi.le.mmm-aaa) then
            tmpe=lff
            tmpp=dlff
           else 
            if(xi.ge.mmm+aaa.and.xi.lt.mmm+aaa+ppp) then
              tmpe=uff
              tmpp=duff
            else
              if(xi.ge.mmm+aaa+ppp+2.d0*aaa) then
                tmpe=lff
                tmpp=dlff
              else 
               tmpe=0.d0
               tmpp=0.d0
              endif
            endif
           endif
         else 
           if(dcl.ge.0.d0) then
             if(xi.le.mmm-aaa-ppp-2.d0*aaa) then
               tmpe=uff
               tmpp=duff
             else 
               if(xi.ge.mmm-aaa-ppp.and.xi.lt.mmm-aaa) then
                 tmpe=lff
                 tmpp=dlff
               else
                 if(xi.ge.mmm+aaa) then
                   tmpe=uff
                   tmpp=duff
                 else 
                   tmpe=0.d0
                   tmpp=0.d0
                 endif
               endif
             endif
           else
             if(xi.le.mmm-aaa) then
               tmpe=lff
               tmpp=dlff
             else 
               if(xi.ge.mmm+aaa) then
                 tmpe=uff 
                 tmpp=duff 
               else
                 tmpe=0.d0
                 tmpp=0.d0
               endif
             endif
           endif
          endif

!! force components
!! method (A.55)~(A.60)
!!forces 
!! d xi / dxi
       call dotpr(rkj,rkj,3,mrkj)
       mrkj=sqrt(mrkj) 
       tdi(1)=mrkj/dot1**2.d0*rijkj(1)
       tdi(2)=mrkj/dot1**2.d0*rijkj(2)
       tdi(3)=mrkj/dot1**2.d0*rijkj(3)
!! d xi / dxl
       tdl(1)=-1.d0*mrkj/dot2**2.d0*rkjkl(1)
       tdl(2)=-1.d0*mrkj/dot2**2.d0*rkjkl(2)
       tdl(3)=-1.d0*mrkj/dot2**2.d0*rkjkl(3)
!! d xi / dxj
       call dotpr(rij,rkj,3,drijkj)
       call dotpr(rkl,rkj,3,drklkj)
       tdj(1)=(drijkj/mrkj**2.d0-1.d0)*tdi(1)-drklkj/mrkj**2.d0*tdl(1)
       tdj(2)=(drijkj/mrkj**2.d0-1.d0)*tdi(2)-drklkj/mrkj**2.d0*tdl(2)
       tdj(3)=(drijkj/mrkj**2.d0-1.d0)*tdi(3)-drklkj/mrkj**2.d0*tdl(3)
!! d xi / dxk
       tdk(1)=(drklkj/mrkj**2.d0-1.d0)*tdl(1)-drijkj/mrkj**2.d0*tdi(1)
       tdk(2)=(drklkj/mrkj**2.d0-1.d0)*tdl(2)-drijkj/mrkj**2.d0*tdi(2)
       tdk(3)=(drklkj/mrkj**2.d0-1.d0)*tdl(3)-drijkj/mrkj**2.d0*tdi(3)

!! add the calculated forces to the originals
       dx(ai)=dx(ai)+tmpp*tdi(1)
       dy(ai)=dy(ai)+tmpp*tdi(2)
       dz(ai)=dz(ai)+tmpp*tdi(3)
       dx(aj)=dx(aj)+tmpp*tdj(1)
       dy(aj)=dy(aj)+tmpp*tdj(2)
       dz(aj)=dz(aj)+tmpp*tdj(3)
       dx(ak)=dx(ak)+tmpp*tdk(1)
       dy(ak)=dy(ak)+tmpp*tdk(2)
       dz(ak)=dz(ak)+tmpp*tdk(3)
       dx(al)=dx(al)+tmpp*tdl(1)
       dy(al)=dy(al)+tmpp*tdl(2)
       dz(al)=dz(al)+tmpp*tdl(3)
!!e051211

!!        if(lsper(i)) then

        else
! multiple periodic function (mper) function still working 
! energy and forces terms are necessary



!!        if(lsper(i)) then
        endif

       enemd=enemd+tmpe
       if(prnlev.ge.6) write(6,*) 'energy of mod',i,tmpe,enemd


!c number of MD
       enddo

!! 011810
       else if(laspline(k)) then

!c 012210
!!       std=stoagmd(k,2)

!!debug
!!       write(6,*) 'mdnum',mdnum,std

!! dihedral angle spline 
       do i=stoagmd(k,1),stoagmd(k,2)
!c mdinm(i) must be same to mdjnm(i)
        if(prnlev.ge.6) write(6,*) 'Dihedral angle Cubic Spline'
        do j=1,mdinm(i)
          fat(j)=mdlis(mdipt(i)+j-1)
          call atomid(fat(j),sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) j,fat(j),sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
        enddo

!! change i and j each other to become agreement with dihedral angles in Modeller
!!atomnumber
!! i
!!       ai=fat(1)
!!       ri(1)=x(ai)
!!       ri(2)=y(ai)
!!       ri(3)=z(ai)
       ai=fat(4)
       ri(1)=x(ai)
       ri(2)=y(ai)
       ri(3)=z(ai)
!! j
       aj=fat(2)
       rj(1)=x(aj)
       rj(2)=y(aj)
       rj(3)=z(aj)
!! k
       ak=fat(3)
       rk(1)=x(ak)
       rk(2)=y(ak)
       rk(3)=z(ak)
!! l
!!       al=fat(4)
!!       rl(1)=x(al)
!!       rl(2)=y(al)
!!       rl(3)=z(al)
       al=fat(1)
       rl(1)=x(al)
       rl(2)=y(al)
       rl(3)=z(al)

!!       write(6,*) rl(1),rl(2),rl(3)
!!dihedral angle
!! xi=sign(xi)acos((rijXrkj)d(rkjXrkl)/|rijXrkj||rkjXrkl|)
!! xi=sign(xi)acos((rijkj)d(rkjkl)/|rijkj||rkjkl|)
!! xi=sign(xi)acos(dot/dot1/dot2)
       do j=1,3        
        rij(j)=rj(j)-ri(j)
        rkj(j)=rj(j)-rk(j)
        rkl(j)=rl(j)-rk(j)
        rik(j)=rk(j)-ri(j)
        rjl(j)=rl(j)-rj(j)
       enddo
       call cross3(rij,rkj,rijkj)
       call cross3(rkj,rkl,rkjkl)
       call dotpr(rijkj,rkjkl,3,dot)
       call dotpr(rijkj,rijkj,3,dot1)
       dot1=sqrt(dot1)
       call dotpr(rkjkl,rkjkl,3,dot2)
       dot2=sqrt(dot2)

       xi=acos(dot/dot1/dot2)

!! 012110
       if((dot/dot1/dot2).le.-1.d0) then
        xi=acos(-1.d0)
       elseif((dot/dot1/dot2).ge.1.d0) then
        xi=acos(1.d0)
       endif
!!       write(6,*) 'hi',xi,acos(dot/dot1/dot2)
       oxi=xi

!! for agreement of dihedral sign with modeller definition
!!       call cross3(rkjkl,rijkj,r3)
       call cross3(rijkj,rkjkl,r3)
       call dotpr(rkj,r3,3,dot3)

       if (dot3.ge.0.d0) then
         signxi=1.d0
       else
         signxi=-1.d0
       endif

       xi=signxi*xi

!c       if(xi.le.0.d0) then
!c        xi=xi+pi
!c       else
!c        xi=xi-pi
!c       endif 

!c       write(6,*) 'xi',signxi,xi,dot/dot1/dot2,
!c     & acos(dot/dot1/dot2),acos(-1.d0)

!! spline subroutine
!! energy 
       do ii=1,tnpos(i)
        tmpx(ii)=lval(i)+(ii-1)*inte(i)
        tmpy(ii)=mdpos(i,ii)
        tmpy2(ii)=smdpos(i,ii)
       enddo

       tmpi=tnpos(i)
!c       write(6,*) 'hiru',i,tnpos(i),tmpx(1),tmpx(tmpi)

!! 012110 030810
       call msplint(tmpx,tmpy,tmpy2,tnpos(i),xi,rety,retdy,lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))


       if(prnlev.ge.7) write(6,*) 'hiru',i,xi*180.d0/pi,rety,retdy

       enemd=enemd+rety

!! method (A.49)~(A.54)
!! equations seem to be something wrong
!!        aa(1)=1.d0/dot1*(rkjkl(1)/dot2-cos(xi)*rijkj(1)/dot1)
!!        aa(2)=1.d0/dot1*(rkjkl(2)/dot2-cos(xi)*rijkj(2)/dot1)
!!        aa(3)=1.d0/dot1*(rkjkl(3)/dot2-cos(xi)*rijkj(3)/dot1)
!!        bb(1)=1.d0/dot2*(rijkj(1)/dot1-cos(xi)*rkjkl(1)/dot2)
!!        bb(2)=1.d0/dot2*(rijkj(2)/dot1-cos(xi)*rkjkl(2)/dot2)
!!        bb(3)=1.d0/dot2*(rijkj(3)/dot1-cos(xi)*rkjkl(3)/dot2)
!!
!!        call cross3(rkj,aa,rkjaa)
!!        call cross3(rik,aa,rikaa)
!!        call cross3(rkl,bb,rklbb)
!!        call cross3(rjl,bb,rjlbb)
!!        call cross3(rij,aa,rijaa)
!!        call cross3(rij,bb,rijbb)
!!
!!! d cos xi / dxi
!!        tdi(1)=rkjaa(1)
!!        tdi(2)=rkjaa(2)
!!        tdi(3)=rkjaa(3)
!!
!!! d cos xi / dxj
!!        tdj(1)=rikaa(1)-rklbb(1)
!!        tdj(2)=rikaa(2)-rklbb(2)
!!        tdj(3)=rikaa(3)-rklbb(3)
!!
!!! d cos xi / dxk
!!        tdk(1)=rjlbb(1)-rijaa(1)
!!        tdk(2)=rjlbb(2)-rijaa(2)
!!        tdk(3)=rjlbb(3)-rijaa(3)
!!
!!! d cos xi / dxl
!!        tdl(1)=rijbb(1)
!!        tdl(2)=rijbb(2)
!!        tdl(3)=rijbb(3)
!!
!!! add the calculated forces to the originals
!!       dx(ai)=dx(ai)+retdy/sin(xi)*tdi(1)
!!       dy(ai)=dy(ai)+retdy/sin(xi)*tdi(2)
!!       dz(ai)=dz(ai)+retdy/sin(xi)*tdi(3)
!!       dx(aj)=dx(aj)+retdy/sin(xi)*tdj(1)
!!       dy(aj)=dy(aj)+retdy/sin(xi)*tdj(2)
!!       dz(aj)=dz(aj)+retdy/sin(xi)*tdj(3)
!!       dx(ak)=dx(ak)+retdy/sin(xi)*tdk(1)
!!       dy(ak)=dy(ak)+retdy/sin(xi)*tdk(2)
!!       dz(ak)=dz(ak)+retdy/sin(xi)*tdk(3)
!!       dx(al)=dx(al)+retdy/sin(xi)*tdl(1)
!!       dy(al)=dy(al)+retdy/sin(xi)*tdl(2)
!!       dz(al)=dz(al)+retdy/sin(xi)*tdl(3)


!! method (A.55)~(A.60)
!!forces 
!! d xi / dxi
       call dotpr(rkj,rkj,3,mrkj)
       mrkj=sqrt(mrkj) 
       tdi(1)=mrkj/dot1**2.d0*rijkj(1)
       tdi(2)=mrkj/dot1**2.d0*rijkj(2)
       tdi(3)=mrkj/dot1**2.d0*rijkj(3)
!! d xi / dxl
       tdl(1)=-1.d0*mrkj/dot2**2.d0*rkjkl(1)
       tdl(2)=-1.d0*mrkj/dot2**2.d0*rkjkl(2)
       tdl(3)=-1.d0*mrkj/dot2**2.d0*rkjkl(3)
!! d xi / dxj
       call dotpr(rij,rkj,3,drijkj)
       call dotpr(rkl,rkj,3,drklkj)
       tdj(1)=(drijkj/mrkj**2.d0-1.d0)*tdi(1)-drklkj/mrkj**2.d0*tdl(1)
       tdj(2)=(drijkj/mrkj**2.d0-1.d0)*tdi(2)-drklkj/mrkj**2.d0*tdl(2)
       tdj(3)=(drijkj/mrkj**2.d0-1.d0)*tdi(3)-drklkj/mrkj**2.d0*tdl(3)
!! d xi / dxk
       tdk(1)=(drklkj/mrkj**2.d0-1.d0)*tdl(1)-drijkj/mrkj**2.d0*tdi(1)
       tdk(2)=(drklkj/mrkj**2.d0-1.d0)*tdl(2)-drijkj/mrkj**2.d0*tdi(2)
       tdk(3)=(drklkj/mrkj**2.d0-1.d0)*tdl(3)-drijkj/mrkj**2.d0*tdi(3)

!! add the calculated forces to the originals
       dx(ai)=dx(ai)-retdy*tdi(1)
       dy(ai)=dy(ai)-retdy*tdi(2)
       dz(ai)=dz(ai)-retdy*tdi(3)
       dx(aj)=dx(aj)-retdy*tdj(1)
       dy(aj)=dy(aj)-retdy*tdj(2)
       dz(aj)=dz(aj)-retdy*tdj(3)
       dx(ak)=dx(ak)-retdy*tdk(1)
       dy(ak)=dy(ak)-retdy*tdk(2)
       dz(ak)=dz(ak)-retdy*tdk(3)
       dx(al)=dx(al)-retdy*tdl(1)
       dy(al)=dy(al)-retdy*tdl(2)
       dz(al)=dz(al)-retdy*tdl(3)

!c number of MD
       enddo

!c 100610
!c bicubic spline interpolation for phi/chi1, chi1/chi2, phi/psi
       elseif(lrotspline(k)) then

       do i=stoagmd(k,1),stoagmd(k,2)

!c mdinm(i) must be same to mdjnm(i)
        if(prnlev.ge.6) write(6,*) 'Bicubic Spline'
        do j=1,mdinm(i)
          fat(j)=mdlis(mdipt(i)+j-1)
          call atomid(fat(j),sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) j,fat(j),sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
        enddo

!! 1st dihedral(ijkl)  phi   xi1    phi
!! 2nd dihedral(mnop)  xi1   xi2    psi
!!atomnumber
!! change 1<-->4 and 5<-->8 in order to match dihedral angles in CHARMM
!! i
       ai=fat(4)
       ri(1)=x(ai)
       ri(2)=y(ai)
       ri(3)=z(ai)
!! j
       aj=fat(2)
       rj(1)=x(aj)
       rj(2)=y(aj)
       rj(3)=z(aj)
!! k
       ak=fat(3)
       rk(1)=x(ak)
       rk(2)=y(ak)
       rk(3)=z(ak)
!! l
       al=fat(1)
       rl(1)=x(al)
       rl(2)=y(al)
       rl(3)=z(al)
!! m
       am=fat(8)
       rm(1)=x(am)
       rm(2)=y(am)
       rm(3)=z(am)
!! n
       an=fat(6)
       rn(1)=x(an)
       rn(2)=y(an)
       rn(3)=z(an)
!! o
       ao=fat(7)
       ro(1)=x(ao)
       ro(2)=y(ao)
       ro(3)=z(ao)
!! p
       ap=fat(5)
       rp(1)=x(ap)
       rp(2)=y(ap)
       rp(3)=z(ap)

!!       write(6,*) 'here',ai,aj,ak,al,am,an,ao,ap

!! dihe(1st)
       do j=1,3        
        rij(j)=rj(j)-ri(j)
        rkj(j)=rj(j)-rk(j)
        rkl(j)=rl(j)-rk(j)
        rik(j)=rk(j)-ri(j)
        rjl(j)=rl(j)-rj(j)
       enddo
       call cross3(rij,rkj,rijkj)
       call cross3(rkj,rkl,rkjkl)
       call dotpr(rijkj,rkjkl,3,dot)
       call dotpr(rijkj,rijkj,3,dot1)
       dot1=sqrt(dot1)
       call dotpr(rkjkl,rkjkl,3,dot2)
       dot2=sqrt(dot2)

       xi=acos(dot/dot1/dot2)

       if((dot/dot1/dot2).le.-1.d0) then
        xi=acos(-1.d0)
       elseif((dot/dot1/dot2).ge.1.d0) then
        xi=acos(1.d0)
       endif
!!       write(6,*) 'hi',xi,acos(dot/dot1/dot2)
       oxi=xi

!! for agreement of dihedral sign with modeller definition
!!       call cross3(rkjkl,rijkj,r3)
       call cross3(rijkj,rkjkl,r3)
       call dotpr(rkj,r3,3,dot3)

       if (dot3.ge.0.d0) then
         signxi=1.d0
       else
         signxi=-1.d0
       endif

       xi1=signxi*xi*180.d0/pi
!!       write(6,*) '1st dihe',xi1

!! dihe(2nd) 
!! ijkl
!! mnop
       do j=1,3        
        rmn(j)=rn(j)-rm(j)
        ron(j)=rn(j)-ro(j)
        rop(j)=rp(j)-ro(j)
        rmo(j)=ro(j)-rm(j)
        rnp(j)=rp(j)-rn(j)
       enddo
       call cross3(rmn,ron,rmnon)
       call cross3(ron,rop,ronop)
       call dotpr(rmnon,ronop,3,dott)
       call dotpr(rmnon,rmnon,3,dott1)
       dott1=sqrt(dott1)
       call dotpr(ronop,ronop,3,dott2)
       dott2=sqrt(dott2)

       xi=acos(dott/dott1/dott2)

       if((dott/dott1/dott2).le.-1.d0) then
        xi=acos(-1.d0)
       elseif((dott/dott1/dott2).ge.1.d0) then
        xi=acos(1.d0)
       endif
!!       write(6,*) 'hi',xi,acos(dott/dott1/dott2)
       oxi=xi

!! for agreement of dihedral sign with modeller definition
!!       call cross3(ronop,rmnon,r3)
       call cross3(rmnon,ronop,r3)
       call dotpr(ron,r3,3,dott3)

       if (dott3.ge.0.d0) then
         signxi=1.d0
       else
         signxi=-1.d0
       endif

       xi2=signxi*xi*180.d0/pi
!!       write(6,*) '2nd dihe',xi2


!! bicubic spline subroutine

!! time consuming step ?
       tmpi=tnpos(i)
       do ii=1,tnpos(i)
         x1(ii)=sxmin(i)+(ii-1)*sxint(i)
       enddo
       do j=1,tnpos(i)
         x2(j)=symin(i)+(j-1)*syint(i)
       enddo


!!checking routine
!!----------------------------------------------------------------------
!!        iii=10
!!        kk=tnpos(i)*iii
!!        invn=1.d0/iii
!!
!!        do ii2=1,kk-(iii-1)
!!           sx=sxmin(i)+(ii2-1)*sxint(i)*invn
!!          do jj2=1,kk-(iii-1)
!!           sy=symin(i)+(jj2-1)*syint(i)*invn
!!
!!       do ii=1,tnpos(i)
!!         do j=1,tnpos(i)
!!           jj=(ii-1)*tnpos(i)+j 
!!           ytmp(j)=mdpos(i,jj)
!!           y2tmp(j)=ytwo(i,jj)
!!         enddo
!!         call msplint(x2,ytmp,y2tmp,tnpos(i),sy,rety2(ii),
!!     &                retdy2(ii),
!!     &                lder(i),hder(i),ldspline(k),ldspa(i),
!!     &                rsm(i),knem(i),expm(i)
!!     &               )
!!       enddo 
!!
!!       call mspline(x1,rety2,tnpos(i),retdy2(1),retdy2(tmpi),y2tmp)
!!       call msplint(x1,rety2,y2tmp,tmpi,sx,retyx1,retdyx1,
!!     &              lder(i),hder(i),ldspline(k),ldspa(i),
!!     &              rsm(i),knem(i),expm(i)
!!     &             )
!!       
!!       write(6,*) 'xi1',sx,sy,retyx1,retdyx1
!!
!!       do j=1,tnpos(i)
!!         do ii=1,tnpos(i)
!!           jj=(ii-1)*tnpos(i)+j 
!!           ytmp(ii)=mdpos(i,jj)
!!           y2tmp(ii)=ytwo2(i,jj)
!!         enddo
!!         call msplint(x1,ytmp,y2tmp,tnpos(i),sx,rety2(j),
!!     &                retdy2(j),
!!     &                lder(i),hder(i),ldspline(k),ldspa(i),
!!     &                rsm(i),knem(i),expm(i)
!!     &               )
!!       enddo 
!!
!!       call mspline(x2,rety2,tnpos(i),retdy2(1),retdy2(tmpi),y2tmp)
!!       call msplint(x2,rety2,y2tmp,tmpi,sy,retyx2,retdyx2,
!!     &              lder(i),hder(i),ldspline(k),ldspa(i),
!!     &              rsm(i),knem(i),expm(i)
!!     &             )
!!       
!!       write(6,*) 'xi2',sx,sy,retyx2,retdyx2
!!         enddo
!!        enddo
!!
!!        write(6,*) 'stop: checking routine rotspline'
!!        stop
!!----------------------------------------------------------------------
         
!! 101012
       if (.not.lcubi(i)) then

!c 101212
          cdnum=md2cd(i)

!! 2nd bicubic interpolation
!c find i,j where sx and sy are included in the grid
            do ii=1,tnpos(i)-1
               if(xi1.ge.x1(ii).and.xi1.lt.x1(ii+1)) then
                  v1=ii
                  v2=ii+1
                  goto 900
               endif
            enddo
 900                continue
            do j=1,tnpos(i)-1
               if(xi2.ge.x2(j).and.xi2.lt.x2(j+1)) then
                  v3=j
                  v4=j+1
                  goto 901
               endif
            enddo
 901                continue

            t=(xi1-x1(v1))/(x1(v2)-x1(v1))
            u=(xi2-x2(v3))/(x2(v4)-x2(v3))

            ry=0.
            dy2=0.
            dy1=0.
            do ii=4,1,-1
               ry=t*ry+((ccl(cdnum,v1,v3,ii,4)*u+ccl(cdnum,v1,v3,ii,3))*u+ccl(cdnum,v1,v3,ii,2))*u+ccl(cdnum,v1,v3,ii,1)
               dy2=t*dy2+(3.*ccl(cdnum,v1,v3,ii,4)*u+2.*ccl(cdnum,v1,v3,ii,3))*u+ccl(cdnum,v1,v3,ii,2)
               dy1=u*dy1+(3.*ccl(cdnum,v1,v3,4,ii)*t+2.*ccl(cdnum,v1,v3,3,ii))*t+ccl(cdnum,v1,v3,2,ii)
            enddo
            dy1=dy1/(x1(v2)-x1(v1))
            dy2=dy2/(x2(v4)-x2(v3))
!!            write(11,*) sx,sy,ry,dy1,dy2

            retyx1=ry
            retyx2=ry
            retdyx1=dy1
            retdyx2=dy2

!!       if (.not.lcubi(i)) then
       else

!! 1st method
!! d y/ d x1 (phi) in the state of fixing xi1
       do ii=1,tnpos(i)
         do j=1,tnpos(i)
           jj=(ii-1)*tnpos(i)+j 
           ytmp(j)=mdpos(i,jj)
           y2tmp(j)=ytwo(i,jj)
         enddo
         call msplint(x2,ytmp,y2tmp,tnpos(i),xi2,rety2(ii),retdy2(ii),lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))
       enddo 

!!!       call mspline(x1,rety2,tnpos(i),retdy2(1),retdy2(tmpi),y2tmp)
       call mspline(x1,rety2,tnpos(i),1.d30,1.d30,y2tmp)
       call msplint(x1,rety2,y2tmp,tmpi,xi1,retyx1,retdyx1,lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))

!!       write(6,*) 'xi1',xi1,xi2,retyx1,retdyx1

!! d y/ d x2 (xi2) in the state of fixing phi
       do j=1,tnpos(i)
         do ii=1,tnpos(i)
           jj=(ii-1)*tnpos(i)+j 
           ytmp(ii)=mdpos(i,jj)
           y2tmp(ii)=ytwo2(i,jj)
         enddo
         call msplint(x1,ytmp,y2tmp,tnpos(i),xi1,rety2(j),retdy2(j),lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))
       enddo 

!!!       call mspline(x2,rety2,tnpos(i),retdy2(1),retdy2(tmpi),y2tmp)
       call mspline(x2,rety2,tnpos(i),1.d30,1.d30,y2tmp)
       call msplint(x2,rety2,y2tmp,tmpi,xi2,retyx2,retdyx2,lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))

!!       write(6,*) 'xi2',xi1,xi2,retyx2,retdyx2


!!       if (.not.lcubi(i)) then
       endif

!! add energy
!! retyx1 == retyx2 
!!       write(6,*) 'here',i,retyx1,retyx2
!!       enemd=enemd+(retyx1+retyx2)/2.d0
!! 111210
!!       enemd=enemd+retyx1
       enemd=enemd+retyx1*formd(i)


!! method (A.55)~(A.60)
!! forces  phi
!! d xi / dxi
       call dotpr(rkj,rkj,3,mrkj)
       mrkj=sqrt(mrkj) 
       tdi(1)=mrkj/dot1**2.d0*rijkj(1)
       tdi(2)=mrkj/dot1**2.d0*rijkj(2)
       tdi(3)=mrkj/dot1**2.d0*rijkj(3)
!! d xi / dxl
       tdl(1)=-1.d0*mrkj/dot2**2.d0*rkjkl(1)
       tdl(2)=-1.d0*mrkj/dot2**2.d0*rkjkl(2)
       tdl(3)=-1.d0*mrkj/dot2**2.d0*rkjkl(3)
!! d xi / dxj
       call dotpr(rij,rkj,3,drijkj)
       call dotpr(rkl,rkj,3,drklkj)
       tdj(1)=(drijkj/mrkj**2.d0-1.d0)*tdi(1)-drklkj/mrkj**2.d0*tdl(1)
       tdj(2)=(drijkj/mrkj**2.d0-1.d0)*tdi(2)-drklkj/mrkj**2.d0*tdl(2)
       tdj(3)=(drijkj/mrkj**2.d0-1.d0)*tdi(3)-drklkj/mrkj**2.d0*tdl(3)
!! d xi / dxk
       tdk(1)=(drklkj/mrkj**2.d0-1.d0)*tdl(1)-drijkj/mrkj**2.d0*tdi(1)
       tdk(2)=(drklkj/mrkj**2.d0-1.d0)*tdl(2)-drijkj/mrkj**2.d0*tdi(2)
       tdk(3)=(drklkj/mrkj**2.d0-1.d0)*tdl(3)-drijkj/mrkj**2.d0*tdi(3)

!! bicubic space is drawn by degs 
!! force (kcal/mol deg) -> (kcal /mol rad) * 180/pi
!! add the calculated forces to the originals
!! 111210
       dx(ai)=dx(ai)-retdyx1*tdi(1)*180.d0/pi*formd(i)
       dy(ai)=dy(ai)-retdyx1*tdi(2)*180.d0/pi*formd(i)
       dz(ai)=dz(ai)-retdyx1*tdi(3)*180.d0/pi*formd(i)
       dx(aj)=dx(aj)-retdyx1*tdj(1)*180.d0/pi*formd(i)
       dy(aj)=dy(aj)-retdyx1*tdj(2)*180.d0/pi*formd(i)
       dz(aj)=dz(aj)-retdyx1*tdj(3)*180.d0/pi*formd(i)
       dx(ak)=dx(ak)-retdyx1*tdk(1)*180.d0/pi*formd(i)
       dy(ak)=dy(ak)-retdyx1*tdk(2)*180.d0/pi*formd(i)
       dz(ak)=dz(ak)-retdyx1*tdk(3)*180.d0/pi*formd(i)
       dx(al)=dx(al)-retdyx1*tdl(1)*180.d0/pi*formd(i)
       dy(al)=dy(al)-retdyx1*tdl(2)*180.d0/pi*formd(i)
       dz(al)=dz(al)-retdyx1*tdl(3)*180.d0/pi*formd(i)

!! forces  xi1
!! d xi / dxi
       call dotpr(ron,ron,3,mrkj)
       mrkj=sqrt(mrkj) 
       tdi(1)=mrkj/dott1**2.d0*rmnon(1)
       tdi(2)=mrkj/dott1**2.d0*rmnon(2)
       tdi(3)=mrkj/dott1**2.d0*rmnon(3)
!! d xi / dxl
       tdl(1)=-1.d0*mrkj/dott2**2.d0*ronop(1)
       tdl(2)=-1.d0*mrkj/dott2**2.d0*ronop(2)
       tdl(3)=-1.d0*mrkj/dott2**2.d0*ronop(3)
!! d xi / dxj
       call dotpr(rmn,ron,3,drijkj)
       call dotpr(rop,ron,3,drklkj)
       tdj(1)=(drijkj/mrkj**2.d0-1.d0)*tdi(1)-drklkj/mrkj**2.d0*tdl(1)
       tdj(2)=(drijkj/mrkj**2.d0-1.d0)*tdi(2)-drklkj/mrkj**2.d0*tdl(2)
       tdj(3)=(drijkj/mrkj**2.d0-1.d0)*tdi(3)-drklkj/mrkj**2.d0*tdl(3)
!! d xi / dxk
       tdk(1)=(drklkj/mrkj**2.d0-1.d0)*tdl(1)-drijkj/mrkj**2.d0*tdi(1)
       tdk(2)=(drklkj/mrkj**2.d0-1.d0)*tdl(2)-drijkj/mrkj**2.d0*tdi(2)
       tdk(3)=(drklkj/mrkj**2.d0-1.d0)*tdl(3)-drijkj/mrkj**2.d0*tdi(3)

!! add the calculated forces to the originals
!! 111210
       dx(am)=dx(am)-retdyx2*tdi(1)*180.d0/pi*formd(i)
       dy(am)=dy(am)-retdyx2*tdi(2)*180.d0/pi*formd(i)
       dz(am)=dz(am)-retdyx2*tdi(3)*180.d0/pi*formd(i)
       dx(an)=dx(an)-retdyx2*tdj(1)*180.d0/pi*formd(i)
       dy(an)=dy(an)-retdyx2*tdj(2)*180.d0/pi*formd(i)
       dz(an)=dz(an)-retdyx2*tdj(3)*180.d0/pi*formd(i)
       dx(ao)=dx(ao)-retdyx2*tdk(1)*180.d0/pi*formd(i)
       dy(ao)=dy(ao)-retdyx2*tdk(2)*180.d0/pi*formd(i)
       dz(ao)=dz(ao)-retdyx2*tdk(3)*180.d0/pi*formd(i)
       dx(ap)=dx(ap)-retdyx2*tdl(1)*180.d0/pi*formd(i)
       dy(ap)=dy(ap)-retdyx2*tdl(2)*180.d0/pi*formd(i)
       dz(ap)=dz(ap)-retdyx2*tdl(3)*180.d0/pi*formd(i)

!c number of MD
       enddo


!! 011310
       else if (ldspline(k)) then
!! distance cubic spline

!c 012210
!!       std=stoagmd(k,2)

!! debug
!!       write(6,*) 'mdnum',mdnum,std
       
       do i=stoagmd(k,1),stoagmd(k,2)
!c mdinm(i) must be same to mdjnm(i)
        if(prnlev.ge.6) write(6,*) 'Distance Cubic Spline'
        do j=1,mdinm(i)
          fatom=mdlis(mdipt(i)+j-1)
          call atomid(fatom,sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) '1st',fatom,sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
          satom=mdlis(mdjpt(i)+j-1)
          call atomid(satom,sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) '2nd',satom,sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
        enddo

!c distance
       jix=x(satom)-x(fatom)
       jiy=y(satom)-y(fatom)
       jiz=z(satom)-z(fatom)
       dist=sqrt(jix*jix+jiy*jiy+jiz*jiz)

!! spline subroutine
!! energy
       do ii=1,tnpos(i)
        tmpx(ii)=lval(i)+(ii-1)*inte(i)
        tmpy(ii)=mdpos(i,ii)
        tmpy2(ii)=smdpos(i,ii)
       enddo

!! 012110 030810
       call msplint(tmpx,tmpy,tmpy2,tnpos(i),dist,rety,retdy,lder(i),hder(i),ldspline(k),ldspa(i),rsm(i),knem(i),expm(i))

       if(prnlev.ge.7) write(6,*) dist,rety,retdy
       enemd=enemd+rety

       dx(fatom)=dx(fatom)-retdy/dist*jix
       dy(fatom)=dy(fatom)-retdy/dist*jiy
       dz(fatom)=dz(fatom)-retdy/dist*jiz

       dx(satom)=dx(satom)+retdy/dist*jix
       dy(satom)=dy(satom)+retdy/dist*jiy
       dz(satom)=dz(satom)+retdy/dist*jiz

!!here


!c number of MD
       enddo

!!       if(lperiod(k)) then
!!       else if (ldspline(k)) then
       else

!c     number of MD
!c       write(6,*) 'stoagmd',stoagmd(k,1),stoagmd(k,2)

! multiple gaussian distribution or multiple harmonics
       do i=stoagmd(k,1),stoagmd(k,2)
!c mdinm(i) must be same to mdjnm(i)
        if(prnlev.ge.6) write(6,*) 'Multiple Gaussian Distribution'
        do j=1,mdinm(i)
          fatom=mdlis(mdipt(i)+j-1)
          call atomid(fatom,sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) '1st',fatom,sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
          satom=mdlis(mdjpt(i)+j-1)
          call atomid(satom,sidi,ridi,reni,aci)
          if(prnlev.ge.6) then
          write(6,*) '2nd',satom,sidi(1:idleng),ridi(1:idleng),aci(1:idleng)
          endif
        enddo

!c distance
       jix=x(satom)-x(fatom)
       jiy=y(satom)-y(fatom)
       jiz=z(satom)-z(fatom)
       dist=sqrt(jix*jix+jiy*jiy+jiz*jiz)
!c       write(6,*) 'dist',dist
       sumexp=0.d0
       sumexpm=1.d0
       if(prnlev.ge.6) then
        write(6,*) 'tnmin',tnmin(i)
       endif


! c081509
!! movable one quadratic potential
       if(loqua(i)) then

       do ii=1,tnmin(i) 
        tmpsort(ii)=mdmin(i,ii)
       enddo
       io=tnmin(i)

!! sort 
       call sortag(tmpsort,io,itag)

!c       do ii=1,tnmin(i)
!c        write(6,*) ii,itag(ii),mdmin(i,ii),ii,tmpsort(ii)
!c       enddo

!c find optminindex
       if(dist.le.tmpsort(1)) then
         optmindex=1
       endif
       do ii=1,tnmin(i)-1
         if((tmpsort(ii).lt.dist).and.(tmpsort(ii+1).ge.dist)) then
          avetmp=(tmpsort(ii)+tmpsort(ii+1))/2.d0
!!          write(6,*) 'avetmp',avetmp
          if(dist.le.avetmp) then
           optmindex=ii
          else
           optmindex=ii+1
          endif
         continue
         endif
       enddo
       if(dist.gt.tmpsort(io)) then
         optmindex=io
       endif

       if(prnlev.gt.6) write(6,*) 'optminex',i,dist,optmindex

       tmpe=formd(i)*(dist-tmpsort(optmindex))*(dist-tmpsort(optmindex))

!!       if(loqua(i)) then
       elseif(lsgar(i)) then
!c012610
!! single gaussian restraint potential
        if(prnlev.ge.6) write(6,*) 'Single Gaussian Restraint Potential'

        if(tnmin(i).ge.2) then
         write(6,*) 'The number of min and sigma must be one in sgar'
         stop
        endif
!c        write(6,*) i,tnmin(i),one,two,pi
        do ii=1,tnmin(i)
!c see modeller manual (A.62) with removing constant term
!c kbT is necessary kb: boltzman , T: room temp 297.15K in modeller manual 9v7
         rtemp=297.15d0
         tmpe=KBOLTZ*rtemp*one/two*((dist-mdmin(i,ii))/mdsig(i,ii))**two
!cc     &        -log(one/mdmin(i,ii)/sqrt(two*pi))
        enddo

!c030810
!!single gaussian restraint potential with soft asymtote
!!       if(loqua(i)) then
       elseif(lsgaa(i)) then
        if(prnlev.ge.6) write(6,*) 'Single Gaussian Potential with SoftAsymtote'

        write(6,*) 'here',i,rsm(i),knem(i),expm(i),pucm(i),dist

        rtemp=297.15d0

!c ii = 1 
        do ii=1,tnmin(i)
         prek=KBOLTZ*rtemp*one/two/mdsig(i,ii)**two
        enddo

        asymbu=(-two*prek*rsm(i)+knem(i))/(expm(i)*rsm(i)**(-expm(i)-one))
        asymau=prek*rsm(i)**two-knem(i)*rsm(i)-asymbu/rsm(i)**expm(i)

        asymbl=(-two*prek*rsm(i)+knem(i))/expm(i)/(-rsm(i))**(-expm(i)-one)
        asymal=prek*rsm(i)**two-knem(i)*rsm(i)+asymbl/(-rsm(i))**expm(i)

        if(dist.le.(pucm(i)-rsm(i)))then
         tmpe=asymal-asymbl/(dist-pucm(i))**expm(i)-knem(i)*(dist-pucm(i))
        elseif((dist.gt.(pucm(i)-rsm(i))).and.(dist.le.pucm(i)))then
         tmpe=prek*(dist-pucm(i))**two
        elseif((dist.gt.pucm(i)).and.(dist.le.(pucm(i)+rsm(i))))then
         tmpe=prek*(dist-pucm(i))**two
        else
         tmpe=asymau+asymbu/(dist-pucm(i))**expm(i)+knem(i)*(dist-pucm(i))
        endif


!!       if(loqua(i)) then
       else

       do ii=1,tnmin(i)
        if(prnlev.ge.6) then
!c062509
         if(lmhar(i).or.lgmha(i))then
!!multiple harmonics
        write(6,'(a12,i4,5f7.3)')'mdmin,mdsig',i,mdmin(i,ii),mdsig(i,ii),formd(i),dist,(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)

         else
!!multiple gaussian
        write(6,'(a12,i4,5f7.3)')'mdmin,mdsig',i,mdmin(i,ii),mdsig(i,ii),formd(i),dist,exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii))

         endif
        endif

!c062509
        if(lmhar(i).or.lgmha(i))then

        sumexpm=sumexpm*(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)

        else

!c 061909 bug fix
!c Bugs in multiple gaussian potential 
!c when the number of minimums is one, it must be the upper and lower
!c bound potential (no normal potential)
        if(tnmin(i).eq.1) then
        mdsigtmp=ulsig(i)
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)
!c        write(6,*) 'Hello', mdsigtmp,dist,mdmin(i,ii),sumexp
        else

!c 060809 bug fix
!c change the potential in upper and lower bound
        if(ii.eq.1) then
!c lower bound
        if(dist.ge.mdmin(i,ii)) then
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii))
        else
        mdsigtmp=ulsig(i)
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)
        endif

        elseif(ii.eq.tnmin(i)) then
!c upper bound
        if(dist.le.mdmin(i,ii)) then
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii))
        else
        mdsigtmp=ulsig(i)
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)
        endif

        else
!c normal
        sumexp=sumexp+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii))
        endif

!!        if(tnmin(i).eq.1) then
        endif

!!        if(lmhar(i).or.lgmha(i))then
        endif

       enddo

       if(lmhar(i)) then
       tmpe=formd(i)*sumexpm
!c062509
       elseif(lgmha(i)) then
       pre=1.d0/(1.d0+sumexpm)
       tmpe=formd(i)*dlog(1.d0+sumexpm)
       else
       tmpe=-formd(i)*dlog(sumexp)
       endif

!!       if(loqua(i)) then
       endif

!c energy
       enemd=enemd+tmpe

       if(prnlev.ge.6) write(6,*) 'energy of mod',i,tmpe,enemd


!c start derivatives
!c (dist)' by x (i)
       dxi = 1.d0/dist*((x(fatom)-x(satom)))
       dyi = 1.d0/dist*((y(fatom)-y(satom)))
       dzi = 1.d0/dist*((z(fatom)-z(satom)))
!c (dist)' by x (j)
       dxj = 1.d0/dist*((-x(fatom)+x(satom)))
       dyj = 1.d0/dist*((-y(fatom)+y(satom)))
       dzj = 1.d0/dist*((-z(fatom)+z(satom)))

       sdipxi=0.d0
       sdipyi=0.d0
       sdipzi=0.d0
       sdipxj=0.d0
       sdipyj=0.d0
       sdipzj=0.d0


!c081509
!!movable one quadratic potential
       if(loqua(i)) then
!! i
       dpxi=TWO*formd(i)*dxi*(dist-tmpsort(optmindex))
       dpyi=TWO*formd(i)*dyi*(dist-tmpsort(optmindex))
       dpzi=TWO*formd(i)*dzi*(dist-tmpsort(optmindex))
!! j
       dpxj=TWO*formd(i)*dxj*(dist-tmpsort(optmindex))
       dpyj=TWO*formd(i)*dyj*(dist-tmpsort(optmindex))
       dpzj=TWO*formd(i)*dzj*(dist-tmpsort(optmindex))


!!       if(loqua(i)) then
!c012610
       elseif(lsgar(i)) then
        ii=1
        tmpp=KBOLTZ*rtemp*(dist-mdmin(i,ii))/mdsig(i,ii)/mdsig(i,ii)
        dpxi=tmpp*dxi
        dpyi=tmpp*dyi
        dpzi=tmpp*dzi
        dpxj=-dpxi
        dpyj=-dpyi
        dpzj=-dpzi


!!       if(loqua(i)) then
!! 030810
       elseif(lsgaa(i)) then
        ii=1

        if(dist.le.(pucm(i)-rsm(i)))then
         tmpp=asymbl*expm(i)*(dist-pucm(i))**(-expm(i)-one)-knem(i)
        elseif((dist.gt.(pucm(i)-rsm(i))).and.(dist.le.pucm(i)))then
         tmpp=two*prek*(dist-pucm(i))
        elseif((dist.gt.pucm(i)).and.(dist.le.(pucm(i)+rsm(i))))then
         tmpp=two*prek*(dist-pucm(i))
        else
         tmpp=-asymbu*expm(i)*(dist-pucm(i))**(-expm(i)-one)+knem(i)
        endif

!C        write(6,*) 'hiru',tmpp,dxi,dyi,dzi

        dpxi=tmpp*dxi
        dpyi=tmpp*dyi
        dpzi=tmpp*dzi
        dpxj=-dpxi
        dpyj=-dpyi
        dpzj=-dpzi


!!       if(loqua(i)) then
       else

       do ii=1,tnmin(i)

!! 060809 multiple harmics
!! 062509 gentle multiple harmonics
       if(lmhar(i).or.lgmha(i)) then

        if(ii.eq.1) then
        after1=0.d0
        front=1.d0
        do ll=1,tnmin(i)
        after2=1.d0
        front=front*(dist-mdmin(i,ll))/mdsig(i,ll)
!! inner loop
         do mm=1,tnmin(i)
          if(ll.ne.mm) after2=after2*(dist-mdmin(i,mm))/mdsig(i,mm)
         enddo
        after1=after1+after2
        enddo 

!!        if(ii.eq.1) then
        endif

       else


!c 061909 bug fix
!c Bugs in multiple gaussian potential 
!c when the number of minimums is one, it must be the upper and lower
!c bound potential (no normal potential)
        if(tnmin(i).eq.1) then
        mdsigtmp=ulsig(i)

        else

!c bug fix 060809
!c (ip=E)' by x (i)

       if(ii.eq.1) then
!! lower bound
        if(dist.ge.mdmin(i,ii)) then
        mdsigtmp=mdsig(i,ii)
        else
        mdsigtmp=ulsig(i)
        endif

       elseif(ii.eq.tnmin(i)) then
!c upper bound
        if(dist.le.mdmin(i,ii)) then
        mdsigtmp=mdsig(i,ii)
        else
        mdsigtmp=ulsig(i)
        endif
       else
        mdsigtmp=mdsig(i,ii)
       endif

!!        if(tnmin(i).eq.1) then
       endif

!c i
       sdipxi=sdipxi+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dxi
       sdipyi=sdipyi+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dyi
       sdipzi=sdipzi+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dzi
!c j
       sdipxj=sdipxj+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dxj
       sdipyj=sdipyj+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dyj
       sdipzj=sdipzj+exp(-(dist-mdmin(i,ii))*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*(-TWO*(dist-mdmin(i,ii))/mdsigtmp/mdsigtmp)*dzj

!!       if(lmhar(i))then
       endif

!       do ii=1,tnmin(i)
       enddo

!c062509 071409
       if(lmhar(i)) then
!! i
       dpxi=TWO*formd(i)*dxi*front*after1
       dpyi=TWO*formd(i)*dyi*front*after1
       dpzi=TWO*formd(i)*dzi*front*after1
!! j
       dpxj=TWO*formd(i)*dxj*front*after1
       dpyj=TWO*formd(i)*dyj*front*after1
       dpzj=TWO*formd(i)*dzj*front*after1

!c 062509 gentle multiple gaussian
!c 071409
       elseif(lgmha(i)) then
!! i
       dpxi=pre*formd(i)*TWO*dxi*front*after1
       dpyi=pre*formd(i)*TWO*dyi*front*after1
       dpzi=pre*formd(i)*TWO*dzi*front*after1
!! j
       dpxj=pre*formd(i)*TWO*dxj*front*after1
       dpyj=pre*formd(i)*TWO*dyj*front*after1
       dpzj=pre*formd(i)*TWO*dzj*front*after1

       else
!c i
       dpxi=-formd(i)*sdipxi/sumexp
       dpyi=-formd(i)*sdipyi/sumexp
       dpzi=-formd(i)*sdipzi/sumexp
!c j
       dpxj=-formd(i)*sdipxj/sumexp
       dpyj=-formd(i)*sdipyj/sumexp
       dpzj=-formd(i)*sdipzj/sumexp

!!       if(lmhar(i)) then
       endif

!!       if(loqua(i)) then
       endif 

!c total i
       dx(fatom)=dx(fatom)+dpxi
       dy(fatom)=dy(fatom)+dpyi
       dz(fatom)=dz(fatom)+dpzi
!c j
       dx(satom)=dx(satom)+dpxj
       dy(satom)=dy(satom)+dpyj
       dz(satom)=dz(satom)+dpzj

!c number of MD
       enddo

!! big big if lperiod(k)
      endif

!c number of divided md
      enddo

!cdebug

      en=enemd

      if(lanamd) then
!! 061806
!! analysis routine
!!      write(6,*) 'analysis'

      tmpsum=0.d0
      do k=1,numcatmd
       do i=stoagmd(k,1),stoagmd(k,2)
         write(6,*) numcatmd,stoagmd(k,1),stoagmd(k,2),lperiod(k),laspline(k),ldspline(k)
         write(6,*) 'energy=',en

!c        do j=1,mdinm(i)
!c          fatom=mdlis(mdipt(i)+j-1)
!c          call atomid(fatom,sidi,ridi,reni,aci)
!c          satom=mdlis(mdjpt(i)+j-1)
!c          call atomid(satom,sidi,ridi,reni,aci)
!c        enddo
!c        jix=x(satom)-x(fatom)
!c        jiy=y(satom)-y(fatom)
!c        jiz=z(satom)-z(fatom)
!c        dist=sqrt(jix*jix+jiy*jiy+jiz*jiz)
!c        do ii=1,tnmin(i)
!c         tmpsum=tmpsum+(dist-mdmin(i,ii))*(dist-mdmin(i,ii))
!c         write(6,*) 'hiru', dist,mdmin(i,ii)
!c        enddo
!c
       enddo
      enddo

!c It is correct, when single template (single harmonics)
!c      write(6,*) 'dist RMSD',sqrt(tmpsum/stoagmd(1,2))
!c      call setmsr('RMSD',sqrt(tmpsum/stoagmd(1,2))) 
      endif

!      write(6,*) 'hi2'

      return
      end

!! 012110 030810
      subroutine msplint(xa,ya,y2a,n,x,y,dy,yp1,ypn,ldspline,ldspa,rsm,knem,expm)

!c obtain from numerical recipes in fortran
      integer n
      real*8 x,y,xa(n),y2a(n),ya(n),dy,yp1,ypn
      integer k,khi,klo
      real*8 a,b,h,rsm,knem,expm,au,bu,al,bl,cu,cl
      logical ldspline,ldspa

!!      write(6,*) 'hiru',ldspline

      klo=1
      khi=n
 1    if (khi-klo.gt.1.d0) then
         k=(khi+klo)/2.d0
         if(xa(k).gt.x) then
            khi=k
         else
            klo=k
         endif
         goto 1
      endif
      h=xa(khi)-xa(klo)
      if(h.eq.0.d0) then
         write(6,*) 'bad xa input in splint'
         stop
      endif

      a=(xa(khi)-x)/h
      b=(x-xa(klo))/h

!! distance spline
      if(ldspline) then

!! 030810 distance spline with soft-asymtote
      if(ldspa) then
!c       write(6,*) 'hiru',ldspa,rsm,knem,expm,x
!c xa(n)=ru knem=Fm expm=SE rsw=rsm
!c xa(1)=rl ku=ypn kl=yp1
!c Cu=ya(n)-ypn*xa(n)
!c Cl=ya(1)-yp1*xa(1)
!c Au=ku*(ru+rsw)+Cu-Bu/rsm**SE-Fm*rsw
!c Bu=(Fm-ku)/SE/rsw**(-SE-1)
!c Al=kl*(rl-rsw)+Cl+Bl/(-rsm)**SE-Fm*rsw
!c Bl=(Fm+kl)/SE/(-rsw)**(-SE-1)
!c 
        cu=ya(n)-ypn*xa(n)
        cl=ya(1)-yp1*xa(1)
        bu=(knem-ypn)/expm/rsm**(-expm-1.d0)
        au=ypn*(xa(n)+rsm)+cu-bu/rsm**expm-knem*rsm
        bl=(knem+yp1)/expm/(-rsm)**(-expm-1.d0)
        al=yp1*(xa(1)-rsm)+cl+bl/(-rsm)**expm-knem*rsm
        if(x.le.(xa(1)-rsm)) then
          y=al-bl/(x-xa(1))**expm-knem*(x-xa(1))
          dy=bl*expm*(x-xa(1))**(-expm-1.d0)-knem
        elseif((x.gt.(xa(1)-rsm)).and.(x.le.xa(1))) then
          y=yp1*x+cl
          dy=yp1
        elseif((x.gt.xa(1)).and.(x.le.xa(n))) then
!! spline
        y=a*ya(klo)+b*ya(khi)+((a**3.d0-a)*y2a(klo)+(b**3.d0-b)*y2a(khi))*(h**2.d0)/6.d0
        dy=(ya(khi)-ya(klo))/(xa(khi)-xa(klo))-(3.d0*a**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(klo)+(3.d0*b**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(khi)

        elseif((x.gt.xa(n)).and.(x.le.(xa(n)+rsm)))then
          y=ypn*x+cu
          dy=ypn
        else
          y=au+bu/(x-xa(n))**expm+knem*(x-xa(n))
          dy=-bu*expm*(x-xa(n))**(-expm-1.d0)+knem
!c          write(6,*) 'hiru2',y,au,bu,x,xa(n),expm,knem,cu
        endif


!!      if(ldspa) then
      else

!! 012109
!! linear interpolation out of range in case of "distances".
      if(x.lt.xa(1)) then
!! out of range
!c      y=ya(1)
!c      dy=0.
      y=yp1*x+ya(1)-yp1*xa(1)
      dy=yp1

      else if(x.gt.xa(n)) then
!! out of range
!c      y=ya(n)
!c      dy=0.
      y=ypn*x+ya(n)-ypn*xa(n)
      dy=ypn

      else

      y=a*ya(klo)+b*ya(khi)+((a**3.d0-a)*y2a(klo)+(b**3.d0-b)*y2a(khi))*(h**2.d0)/6.d0
      dy=(ya(khi)-ya(klo))/(xa(khi)-xa(klo))-(3.d0*a**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(klo)+(3.d0*b**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(khi)

      endif

!!      if(ldspa) then
      endif

!!      if(ldspline) then
      else
!! aspline

      y=a*ya(klo)+b*ya(khi)+((a**3.d0-a)*y2a(klo)+(b**3.d0-b)*y2a(khi))*(h**2.d0)/6.d0
      dy=(ya(khi)-ya(klo))/(xa(khi)-xa(klo))-(3.d0*a**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(klo)+(3.d0*b**2.d0-1.d0)/6.d0*(xa(khi)-xa(klo))*y2a(khi)

      endif

      return
      end

      subroutine mspline(x,y,n,yp1,ypn,y2)
!c obtain from numerical recipes in fortran
!c maxmin should be same value to one in modeller.fcm
      integer n,maxmin
      integer i,k
      real*8 yp1,ypn,x(*),y(*)
      parameter(maxmin=1089)
      real*8 p,qn,sig,un,u(maxmin),y2(maxmin)

      if(yp1.gt.0.99d30) then
         y2(1)=0.d0
         u(1)=0.d0
      else
         y2(1)=-0.5d0
         u(1)=(3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
      endif
      do i=2,n-1
         sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
         p=sig*y2(i-1)+2.d0
         y2(i)=(sig-1.d0)/p
         u(i)=(6.d0*((y(i+1)-y(i))/(x(i+1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p
      enddo
      if(ypn.gt.0.99d30) then
         qn=0.d0
         un=0.d0
      else
         qn=0.5d0
         un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
      endif
      y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)
      do k=n-1,1,-1
         y2(k)=y2(k)*y2(k+1)+u(k)
      enddo

      return
      end


!c 101012
      subroutine bcucof(y,y1,y2,y12,d1,d2,c)
      real*8 d1,d2,c(4,4),y(4),y1(4),y12(4),y2(4)
      integer i,j,k,l
      real*8 d1d2,xx,cl(16),wt(16,16),x(16)
      save wt
      data wt/1,0,-3,2,4*0,-3,0,9,-6,2,0,-6,4,8*0,3,0,-9,6,-2,0,6,-4,10*0,9,-6,2*0,-6,4,2*0,3,-2,6*0,-9,6,2*0,6,-4,4*0,1,0,-3,2,-2,0,6,-4,1,0,-3,2,8*0,-1,0,3,-2,1,0,-3,2,10*0,-3,2,2*0,3,-2,6*0,3,-2,2*0,-6,4,2*0,3,-2,0,1,-2,1,5*0,-3,6,-3,0,2,-4,2,9*0,3,-6,3,0,-2,4,-2,10*0,-3,3,2*0,2,-2,2*0,-1,1,6*0,3,-3,2*0,-2,2,5*0,1,-2,1,0,-2,4,-2,0,1,-2,1,9*0,-1,2,-1,0,1,-2,1,10*0,1,-1,2*0,-1,1,6*0,-1,1,2*0,2,-2,2*0,-1,1/
      d1d2=d1*d2
      do i=1,4
         x(i)=y(i)
         x(i+4)=y1(i)*d1
         x(i+8)=y2(i)*d2
         x(i+12)=y12(i)*d1d2
      enddo
      do i=1,16
         xx=0.
         do k=1,16
            xx=xx+wt(i,k)*x(k)
         enddo
         cl(i)=xx
      enddo
      l=0
      do i=1,4
         do j=1,4
            l=l+1
            c(i,j)=cl(l)
         enddo
      enddo

!      subroutine modellerdummy

return
end

#endif
