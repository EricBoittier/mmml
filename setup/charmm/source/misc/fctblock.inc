      do i=1,reanat
         do k=1,2
         fcttp1   = sqrt(fct01xs(k,i)*fct01xs(k,i) &
                        +fct01ys(k,i)*fct01ys(k,i) &
                        +fct01zs(k,i)*fct01zs(k,i))
         fcttp2   = fct02ss(k,i)

         fctis1(k,i)= one/fcttp1
         fctis2(k,i)= one/fcttp2

         fctmov(k,i)= fct01ss(k,i)
         fctmos(k,i)= fcttp1*fctis2(k,i)

         fcttp1   =                   fctmov(k,i)                    &
                   +fctcb1(fctidx(i))            *fctmos(k,i)        &
                   +fctcb2(fctidx(i))*fctmov(k,i)*fctmos(k,i)        &
                   +fctqos
         fcttp2   = one                                              &
                   +fctcb2(fctidx(i))            *fctmos(k,i)
         fcttp3   = fctcb1(fctidx(i))                                &
                   +fctcb2(fctidx(i))*fctmov(k,i)
         fcttp4   = fctca2(fctidx(i))*(fcttp1-fctca3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctca0(fctidx(i))                                &
                   +fctca1(fctidx(i))*fcttp6
         fcttp8   = fctca1(fctidx(i))*fctca2(fctidx(i))              &
                   /(one/fcttp5+two+fcttp5)
         fctqsg(k,i)= fcttp7
         fctqdv(k,i)= fcttp2*fcttp8
         fctqds(k,i)= fcttp3*fcttp8
         if (fctqsg(k,i) > -0.001d0) then
            call wrndie(-4,'<FCTENE>','Solvation energy too high.')
         endif
         fctqsg(k,i)= fctqsg(k,i)*fctqun(fctidx(i))
         fctqdv(k,i)= fctqdv(k,i)*fctqun(fctidx(i))
         fctqds(k,i)= fctqds(k,i)*fctqun(fctidx(i))

         fcttp1   =                   fctmov(k,i)                    &
                   +fctcd1(fctidx(i))            *fctmos(k,i)        &
                   +fctcd2(fctidx(i))*fctmov(k,i)*fctmos(k,i)        &
                   +fctuos
         fcttp2   = one                                              &
                   +fctcd2(fctidx(i))          *fctmos(k,i)
         fcttp3   = fctcd1(fctidx(i))                                &
                   +fctcd2(fctidx(i))*fctmov(k,i)
         fcttp4   = fctcc2(fctidx(i))*(fcttp1-fctcc3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcc0(fctidx(i))                                &
                   +fctcc1(fctidx(i))*fcttp6
         fcttp8   = fctcc1(fctidx(i))*fctcc2(fctidx(i))              &
                   /(one/fcttp5+two+fcttp5)
         fctusg(k,i)= fcttp7
         fctudv(k,i)= fcttp2*fcttp8
         fctuds(k,i)= fcttp3*fcttp8
         if (fctusg(k,i) < 0.001d0) then
            call wrndie(-4,'<FCTENE>','Surface too low.')
         endif

         fcttp1   =                   fctmov(k,i)                    &
                   +fctcf1(fctidx(i))            *fctmos(k,i)        &
                   +fctcf2(fctidx(i))*fctmov(k,i)*fctmos(k,i)        &
                   +fctvos
         fcttp2   = one                                              &
                     +fctcf2(fctidx(i))          *fctmos(k,i)
         fcttp3   = fctcf1(fctidx(i))                                &
                     +fctcf2(fctidx(i))*fctmov(k,i)
         fcttp4   = fctce2(fctidx(i))*(fcttp1-fctce3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctce0(fctidx(i))                                &
                     +fctce1(fctidx(i))*fcttp6
         fcttp8   = fctce1(fctidx(i))*fctce2(fctidx(i))              &
                     /(one/fcttp5+two+fcttp5)
         fctvsg(k,i)= fcttp7
         fctvdv(k,i)= fcttp2*fcttp8
         fctvds(k,i)= fcttp3*fcttp8

         fcttp1   =                   fctmov(k,i)                    &
                   +fctch1(fctidx(i))            *fctmos(k,i)        &
                   +fctch2(fctidx(i))*fctmov(k,i)*fctmos(k,i)        &
                   +fctwos
         fcttp2   = one                                              &
                   +fctch2(fctidx(i))            *fctmos(k,i)
         fcttp3   = fctch1(fctidx(i))                                &
                   +fctch2(fctidx(i))*fctmov(k,i)
         fcttp4   = fctcg2(fctidx(i))*(fcttp1-fctcg3(fctidx(i)))
         fcttp5   = dexp(-fcttp4)
         fcttp6   = one/(one+fcttp5)
         fcttp7   = fctcg0(fctidx(i))                                &
                   +fctcg1(fctidx(i))*fcttp6
         fcttp8   = fctcg1(fctidx(i))*fctcg2(fctidx(i))              &
                   /(one/fcttp5+two+fcttp5)
         fctwsg(k,i)= fcttp7
         fctwdv(k,i)= fcttp2*fcttp8
         fctwds(k,i)= fcttp3*fcttp8

         fctisg(k,i)= one/fctqsg(k,i)
         ! VdW TERM ===================================================
         fctvdw01      = fcttau*fctisg(k,i)/two
         fctvdw02      = one/(-fctvdw01+fctvdwwrd)
         fctvdwen(k,i) = fctvdwfac(i)*fctvdw02**3
         fctvdwdf(k,i) =-three*fctvdwen(k,i)*fctvdw02                &
                        *fctvdw01*fctisg(k,i)
         ! Salt TERM ==================================================
         ! GBSW like
         ! ---------
         fctslt01    = half*fctscreen*fcttau*fctisg(k,i)
         fctslt02    = fcties*dexp(fctslt01)
         fctslt03    = (fctiem-fctslt02)*fcttai

         fctqsgslt   = fctslt03*fctqsg(k,i)
         fctcgslt(k,i) = (fctiem+(fctslt01-one)*fctslt02)*fcttai*fctcgs(i)
         ! ============================================================
         fct04xs  =(fct03ss(k,i)*fct01xs(k,i)            &
                   +fct01xs(k,i)*fct01xx(k,i)            &
                   +fct01ys(k,i)*fct01xy(k,i)            &
                   +fct01zs(k,i)*fct01xz(k,i))           &
                   *fctis1(k,i) *fctis2(k,i)             &
                   -fct03xs(k,i)*fctis2(k,i)*fctmos(k,i)
         fct04ys  =(fct03ss(k,i)*fct01ys(k,i)            &
                   +fct01xs(k,i)*fct01xy(k,i)            &
                   +fct01ys(k,i)*fct01yy(k,i)            &
                   +fct01zs(k,i)*fct01yz(k,i))           &
                   *fctis1(k,i) *fctis2(k,i)             &
                   -fct03ys(k,i)*fctis2(k,i)*fctmos(k,i)
         fct04zs  =(fct03ss(k,i)*fct01zs(k,i)            &
                   +fct01xs(k,i)*fct01xz(k,i)            &
                   +fct01ys(k,i)*fct01yz(k,i)            &
                   +fct01zs(k,i)*fct01zz(k,i))           &
                   *fctis1(k,i) *fctis2(k,i)             &
                   -fct03zs(k,i)*fctis2(k,i)*fctmos(k,i)

         fctqsx(k,i)=-fct02xs(k,i)*fctqdv(k,i)-fct04xs*fctqds(k,i)
         fctqsy(k,i)=-fct02ys(k,i)*fctqdv(k,i)-fct04ys*fctqds(k,i)
         fctqsz(k,i)=-fct02zs(k,i)*fctqdv(k,i)-fct04zs*fctqds(k,i)
         fctusx(k,i)=-fct02xs(k,i)*fctudv(k,i)-fct04xs*fctuds(k,i)
         fctusy(k,i)=-fct02ys(k,i)*fctudv(k,i)-fct04ys*fctuds(k,i)
         fctusz(k,i)=-fct02zs(k,i)*fctudv(k,i)-fct04zs*fctuds(k,i)
         fctvsx(k,i)=-fct02xs(k,i)*fctvdv(k,i)-fct04xs*fctvds(k,i)
         fctvsy(k,i)=-fct02ys(k,i)*fctvdv(k,i)-fct04ys*fctvds(k,i)
         fctvsz(k,i)=-fct02zs(k,i)*fctvdv(k,i)-fct04zs*fctvds(k,i)
         fctwsx(k,i)=-fct02xs(k,i)*fctwdv(k,i)-fct04xs*fctwds(k,i)
         fctwsy(k,i)=-fct02ys(k,i)*fctwdv(k,i)-fct04ys*fctwds(k,i)
         fctwsz(k,i)=-fct02zs(k,i)*fctwdv(k,i)-fct04zs*fctwds(k,i)

! ==========================================
#if KEY_PARALLEL==1
         if(mynod==0)then
#endif 
! ==========================================
         tmppl1(k,i) = fctcgs(i)*fctqsgslt
         tmpnp1(k,i) = fctnpc(i)*fctqsg(k,i)    &
                      +fctusg(k,i)*fctvsg(k,i)  &
                      +fctwsg(k,i)+fctvdwen(k,i)

         tmpplssx(k,i)=(fctvdwdf(k,i)+fctcgslt(k,i)+fctnpc(i))*fctqsx(k,i)
         tmpnpssx(k,i)= fctvsg(k,i)*fctusx(k,i) &
                       +fctusg(k,i)*fctvsx(k,i) &
                       +fctwsx(k,i)
         tmpplssy(k,i)=(fctvdwdf(k,i)+fctcgslt(k,i)+fctnpc(i))*fctqsy(k,i)
         tmpnpssy(k,i)= fctvsg(k,i)*fctusy(k,i) &
                       +fctusg(k,i)*fctvsy(k,i) &
                       +fctwsy(k,i)
         tmpplssz(k,i)=(fctvdwdf(k,i)+fctcgslt(k,i)+fctnpc(i))*fctqsz(k,i)
         tmpnpssz(k,i)= fctvsg(k,i)*fctusz(k,i) &
                       +fctusg(k,i)*fctvsz(k,i) &
                       +fctwsz(k,i)
! ==========================================
#if KEY_PARALLEL==1
         endif
#endif 
! ==========================================
         enddo
      enddo

! ==========================================
#if KEY_PARALLEL==1
   if(mynod==0)then
#endif 
! ==========================================
      do i=1,reanat
#ifdef FACTS_IMOVE
         if(imove(i)==0) then                                          
#endif
            Ubl= iblock(i)
            kk=Ubl*(Ubl-1)/2+Ubl

            blkpl = tmppl1(1,i) + tmppl1(2,i)*(blcoe(kk)-one)
            ! Should it not be blcov(kk) ??
            blknp = tmpnp1(1,i) + tmpnp1(2,i)*(blcoe(kk)-one)

            blkplssx= tmpplssx(1,i) + tmpplssx(2,i)*(blcoe(kk)-one) &
                     +tmpnpssx(1,i) + tmpnpssx(2,i)*(blcov(kk)-one)
            blkplssy= tmpplssy(1,i) + tmpplssy(2,i)*(blcoe(kk)-one) &
                     +tmpnpssy(1,i) + tmpnpssy(2,i)*(blcov(kk)-one)
            blkplssz= tmpplssz(1,i) + tmpplssz(2,i)*(blcoe(kk)-one) &
                     +tmpnpssz(1,i) + tmpnpssz(2,i)*(blcov(kk)-one)

            fctpl1   = fctpl1 + blkpl
            fctnp1   = fctnp1 + blknp

            fctssx(i) = fctssx(i) + blkplssx
            fctssy(i) = fctssy(i) + blkplssy
            fctssz(i) = fctssz(i) + blkplssz
#ifdef FACTS_IMOVE
         endif                                                         
#endif
      enddo
! ==========================================
#if KEY_PARALLEL==1
   endif
#endif 
! ==========================================
      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct2ll(i-1)+1
         endif
         b=fct2ll(i)
         do j=a,b
            u=i
            v=fct2lb(j)
            vux=x(v)-x(u)
            vuy=y(v)-y(u)
            vuz=z(v)-z(u)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            if (fctdvu < fct2cn) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct2ci
               if(Ubl==Vbl) then
                  ! Salt
                  fct03vu1 = fctpco*fctqsg(1,V)*fctdvu*fctqsg(1,U)
                  fct04vu1 = dexp(-fctikp*fct03vu1)
                  fct05vu1 = fct04vu1/fct03vu1
                  fct06vu1 = fct05vu1+fctikp*fct04vu1
                  fct07vu1 = one/(one+fct05vu1)

                  ! 1/f_GB
                  fct08vu1 = sqrt(fct01vu*fct07vu1)
                  fct09vu1 = -cg(V)*cg(U)*fct02vu*fct08vu1

                  ! fcties = 1. / 78.5
                  fct10vu1 = dexp(-fctscreen/fct08vu1)*fcties
                  fct11vu1 = (fctiem-fct10vu1)*fct09vu1*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu1 = fct09vu1*(fct08vu1*(fctikp*fct04vu1-one)        &
                            *fct02vu                                         &
                            *(fct08vu1*(fctiem-fct10vu1)-fctscreen*fct10vu1) &
                            +(fctiem-fct10vu1)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu1 = half*fct09vu1*fct02vu*fct08vu1*fctdvu          &
                            *fct06vu1*(fct08vu1*(fctiem-fct10vu1)           &
                            -fctscreen*fct10vu1)

                  fctggg(1,U)= fctggg(1,U)-fct13vu1
                  fctggg(1,V)= fctggg(1,V)-fct13vu1
                  ! ------------------------------------
                  fct03vu2 = fctpco*fctqsg(2,V)*fctdvu*fctqsg(2,U)
                  fct04vu2 = dexp(-fctikp*fct03vu2)
                  fct05vu2 = fct04vu2/fct03vu2
                  fct06vu2 = fct05vu2+fctikp*fct04vu2
                  fct07vu2 = one/(one+fct05vu2)

                  ! 1/f_GB
                  fct08vu2 = sqrt(fct01vu*fct07vu2)
                  fct09vu2 = -cg(V)*cg(U)*fct02vu*fct08vu2

                  ! fcties = 1. / 78.5
                  fct10vu2 = dexp(-fctscreen/fct08vu2)*fcties
                  fct11vu2 = (fctiem-fct10vu2)*fct09vu2*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu2 = fct09vu2*(fct08vu2*(fctikp*fct04vu2-one)        &
                            *fct02vu                                         &
                            *(fct08vu2*(fctiem-fct10vu2)-fctscreen*fct10vu2) &
                            +(fctiem-fct10vu2)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu2 = half*fct09vu2*fct02vu*fct08vu2*fctdvu          &
                            *fct06vu2*(fct08vu2*(fctiem-fct10vu2)           &
                            -fctscreen*fct10vu2)

                  fctggg(2,U)= fctggg(2,U)-fct13vu2
                  fctggg(2,V)= fctggg(2,V)-fct13vu2
                  ! ------------------------------------
                  blkpl   = (fct11vu2*(blcoe(kk)-one)+fct11vu1) ! *fct02vu
                  fctpl2  = fctpl2 + blkpl

                  fctiix(U) = fctiix(U)                           &
                             +(-fct12vu2*vux+fct13vu2*fctisg(2,U) &
                             *fctqsx(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vux+fct13vu1*fctisg(1,U) &
                             *fctqsx(1,U))

                  fctiiy(U) = fctiiy(U)                           &
                             +(-fct12vu2*vuy+fct13vu2*fctisg(2,U) &
                             *fctqsy(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vuy+fct13vu1*fctisg(1,U) &
                             *fctqsy(1,U))

                  fctiiz(U) = fctiiz(U)                           &
                             +(-fct12vu2*vuz+fct13vu2*fctisg(2,U) &
                             *fctqsz(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vuz+fct13vu1*fctisg(1,U) &
                             *fctqsz(1,U))

                  fctiix(V) = fctiix(V)                           &
                             +(fct12vu2*vux+fct13vu2*fctisg(2,V)  &
                             *fctqsx(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vux+fct13vu1*fctisg(1,V)  &
                             *fctqsx(1,V))

                  fctiiy(V) = fctiiy(V)                           &
                             +(fct12vu2*vuy+fct13vu2*fctisg(2,V)  &
                             *fctqsy(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vuy+fct13vu1*fctisg(1,V)  &
                             *fctqsy(1,V))

                  fctiiz(V) = fctiiz(V)                           &
                             +(fct12vu2*vuz+fct13vu2*fctisg(2,V)  &
                             *fctqsz(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vuz+fct13vu1*fctisg(1,V)  &
                             *fctqsz(1,V))
               else
                  ! Salt
                  fct03vu1 = fctpco*fctqsg(1,V)*fctdvu*fctqsg(1,U)
                  fct04vu1 = dexp(-fctikp*fct03vu1)
                  fct05vu1 = fct04vu1/fct03vu1
                  fct06vu1 = fct05vu1+fctikp*fct04vu1
                  fct07vu1 = one/(one+fct05vu1)

                  ! 1/f_GB
                  fct08vu1 = sqrt(fct01vu*fct07vu1)
                  fct09vu1 = -cg(V)*cg(U)*fct02vu*fct08vu1

                  ! fcties = 1. / 78.5
                  fct10vu1 = dexp(-fctscreen/fct08vu1)*fcties
                  fct11vu1 = (fctiem-fct10vu1)*fct09vu1*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu1 = fct09vu1*(fct08vu1*(fctikp*fct04vu1-one)        &
                            *fct02vu                                         &
                            *(fct08vu1*(fctiem-fct10vu1)-fctscreen*fct10vu1) &
                            +(fctiem-fct10vu1)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu1 = half*fct09vu1*fct02vu*fct08vu1*fctdvu          &
                            *fct06vu1*(fct08vu1*(fctiem-fct10vu1)           &
                            -fctscreen*fct10vu1)
                  ! ------------------------------------
                  fctggg(1,U)= fctggg(1,U)-fct13vu1 * blcoe(kk)
                  fctggg(1,V)= fctggg(1,V)-fct13vu1 * blcoe(kk)
                  ! ------------------------------------
                  ! blkpl   = fct02vu*fct11vu1*blcoe(kk)
                  blkpl   = fct11vu1*blcoe(kk)
                  fctpl2  = fctpl2 + blkpl

                  fctiix(U)= fctiix(U) + blcoe(kk)*(-fct12vu1*vux &
                            +fct13vu1*fctisg(1,U)*fctqsx(1,U))
                  fctiiy(U)= fctiiy(U) + blcoe(kk)*(-fct12vu1*vuy &
                            +fct13vu1*fctisg(1,U)*fctqsy(1,U))
                  fctiiz(U)= fctiiz(U) + blcoe(kk)*(-fct12vu1*vuz &
                            +fct13vu1*fctisg(1,U)*fctqsz(1,U))
                  fctiix(V)= fctiix(V) + blcoe(kk)*(fct12vu1*vux  &
                            +fct13vu1*fctisg(1,V)*fctqsx(1,V))
                  fctiiy(V)= fctiiy(V) + blcoe(kk)*(fct12vu1*vuy  &
                            +fct13vu1*fctisg(1,V)*fctqsy(1,V))
                  fctiiz(V)= fctiiz(V) + blcoe(kk)*(fct12vu1*vuz  &
                            +fct13vu1*fctisg(1,V)*fctqsz(1,V))
               endif
            endif
         enddo
      enddo

      ! Image Atoms
      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct4ll(i-1)+1
         endif
         b=fct4ll(i)
         do j=a,b
            u=i
            v=fct4lb(j)
            vux=x(V)-x(U)
            vuy=y(V)-y(U)
            vuz=z(V)-z(U)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            u=imattr(i)
            if (fctdvu < fct2cn) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fct01vu  = one/fctdvu
               fct02vu  = one-fctdvu*fct2ci
               if(Ubl==Vbl) then
                  ! Salt
                  fct03vu1 = fctpco*fctqsg(1,V)*fctdvu*fctqsg(1,U)
                  fct04vu1 = dexp(-fctikp*fct03vu1)
                  fct05vu1 = fct04vu1/fct03vu1
                  fct06vu1 = fct05vu1+fctikp*fct04vu1
                  fct07vu1 = one/(one+fct05vu1)

                  ! 1/f_GB
                  fct08vu1 = sqrt(fct01vu*fct07vu1)
                  fct09vu1 = -cg(V)*cg(U)*fct02vu*fct08vu1

                  ! fcties = 1. / 78.5
                  fct10vu1 = dexp(-fctscreen/fct08vu1)*fcties
                  fct11vu1 = (fctiem-fct10vu1)*fct09vu1*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu1 = fct09vu1*(fct08vu1*(fctikp*fct04vu1-one)        &
                            *fct02vu                                         &
                            *(fct08vu1*(fctiem-fct10vu1)-fctscreen*fct10vu1) &
                            +(fctiem-fct10vu1)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu1 = half*fct09vu1*fct02vu*fct08vu1*fctdvu          &
                            *fct06vu1*(fct08vu1*(fctiem-fct10vu1)           &
                            -fctscreen*fct10vu1)

                  fctggg(1,U)= fctggg(1,U)-fct13vu1
                  fctggg(1,V)= fctggg(1,V)-fct13vu1
                  ! ------------------------------------
                  fct03vu2 = fctpco*fctqsg(2,V)*fctdvu*fctqsg(2,U)
                  fct04vu2 = dexp(-fctikp*fct03vu2)
                  fct05vu2 = fct04vu2/fct03vu2
                  fct06vu2 = fct05vu2+fctikp*fct04vu2
                  fct07vu2 = one/(one+fct05vu2)

                  ! 1/f_GB
                  fct08vu2 = sqrt(fct01vu*fct07vu2)
                  fct09vu2 = -cg(V)*cg(U)*fct02vu*fct08vu2

                  ! fcties = 1. / 78.5
                  fct10vu2 = dexp(-fctscreen/fct08vu2)*fcties
                  fct11vu2 = (fctiem-fct10vu2)*fct09vu2*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu2 = fct09vu2*(fct08vu2*(fctikp*fct04vu2-one)        &
                            *fct02vu                                         &
                            *(fct08vu2*(fctiem-fct10vu2)-fctscreen*fct10vu2) &
                            +(fctiem-fct10vu2)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu2 = half*fct09vu2*fct02vu*fct08vu2*fctdvu          &
                            *fct06vu2*(fct08vu2*(fctiem-fct10vu2)           &
                            -fctscreen*fct10vu2)

                  fctggg(2,U)= fctggg(2,U)-fct13vu2
                  fctggg(2,V)= fctggg(2,V)-fct13vu2
                  ! ------------------------------------
                  blkpl   = (fct11vu2*(blcoe(kk)-one)+fct11vu1) ! *fct02vu
                  fctpl2  = fctpl2 + blkpl

                  fctiix(U) = fctiix(U)                           &
                             +(-fct12vu2*vux+fct13vu2*fctisg(2,U) &
                             *fctqsx(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vux+fct13vu1*fctisg(1,U) &
                             *fctqsx(1,U))

                  fctiiy(U) = fctiiy(U)                           &
                             +(-fct12vu2*vuy+fct13vu2*fctisg(2,U) &
                             *fctqsy(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vuy+fct13vu1*fctisg(1,U) &
                             *fctqsy(1,U))

                  fctiiz(U) = fctiiz(U)                           &
                             +(-fct12vu2*vuz+fct13vu2*fctisg(2,U) &
                             *fctqsz(2,U))*(blcoe(kk)-one)        &
                             +(-fct12vu1*vuz+fct13vu1*fctisg(1,U) &
                             *fctqsz(1,U))

                  fctiix(V) = fctiix(V)                           &
                             +(fct12vu2*vux+fct13vu2*fctisg(2,V)  &
                             *fctqsx(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vux+fct13vu1*fctisg(1,V)  &
                             *fctqsx(1,V))

                  fctiiy(V) = fctiiy(V)                           &
                             +(fct12vu2*vuy+fct13vu2*fctisg(2,V)  &
                             *fctqsy(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vuy+fct13vu1*fctisg(1,V)  &
                             *fctqsy(1,V))

                  fctiiz(V) = fctiiz(V)                           &
                             +(fct12vu2*vuz+fct13vu2*fctisg(2,V)  &
                             *fctqsz(2,V))*(blcoe(kk)-one)        &
                             +(fct12vu1*vuz+fct13vu1*fctisg(1,V)  &
                             *fctqsz(1,V))
               else
                  ! Salt
                  fct03vu1 = fctpco*fctqsg(1,V)*fctdvu*fctqsg(1,U)
                  fct04vu1 = dexp(-fctikp*fct03vu1)
                  fct05vu1 = fct04vu1/fct03vu1
                  fct06vu1 = fct05vu1+fctikp*fct04vu1
                  fct07vu1 = one/(one+fct05vu1)

                  ! 1/f_GB
                  fct08vu1 = sqrt(fct01vu*fct07vu1)
                  fct09vu1 = -cg(V)*cg(U)*fct02vu*fct08vu1

                  ! fcties = 1. / 78.5
                  fct10vu1 = dexp(-fctscreen/fct08vu1)*fcties
                  fct11vu1 = (fctiem-fct10vu1)*fct09vu1*fct02vu

                  ! fct12vu: (xi-xj) components
                  fct12vu1 = fct09vu1*(fct08vu1*(fctikp*fct04vu1-one)        &
                            *fct02vu                                         &
                            *(fct08vu1*(fctiem-fct10vu1)-fctscreen*fct10vu1) &
                            +(fctiem-fct10vu1)*fct2fn)

                  ! fct13vu: (QSXi) components
                  fct13vu1 = half*fct09vu1*fct02vu*fct08vu1*fctdvu          &
                            *fct06vu1*(fct08vu1*(fctiem-fct10vu1)           &
                            -fctscreen*fct10vu1)
                  ! ------------------------------------
                  fctggg(1,U)= fctggg(1,U)-fct13vu1 * blcoe(kk)
                  fctggg(1,V)= fctggg(1,V)-fct13vu1 * blcoe(kk)
                  ! ------------------------------------
                  ! blkpl   = fct02vu*fct11vu1*blcoe(kk)
                  blkpl   = fct11vu1*blcoe(kk)
                  fctpl2  = fctpl2 + blkpl

                  fctiix(U)= fctiix(U) + blcoe(kk)*(-fct12vu1*vux &
                            +fct13vu1*fctisg(1,U)*fctqsx(1,U))
                  fctiiy(U)= fctiiy(U) + blcoe(kk)*(-fct12vu1*vuy &
                            +fct13vu1*fctisg(1,U)*fctqsy(1,U))
                  fctiiz(U)= fctiiz(U) + blcoe(kk)*(-fct12vu1*vuz &
                            +fct13vu1*fctisg(1,U)*fctqsz(1,U))
                  fctiix(V)= fctiix(V) + blcoe(kk)*(fct12vu1*vux  &
                            +fct13vu1*fctisg(1,V)*fctqsx(1,V))
                  fctiiy(V)= fctiiy(V) + blcoe(kk)*(fct12vu1*vuy  &
                            +fct13vu1*fctisg(1,V)*fctqsy(1,V))
                  fctiiz(V)= fctiiz(V) + blcoe(kk)*(fct12vu1*vuz  &
                            +fct13vu1*fctisg(1,V)*fctqsz(1,V))
               endif
            endif
         enddo
      enddo

! ==========================================
#if KEY_PARALLEL==1
      call gcombr(fctggg,reanat2) ! MH2023: extend the gcomb interface for fix
#endif 
! ==========================================

      do k=1,2
         do i=1,reanat
            fctkkk(k,i)= -fctisg(k,i)*fctggg(k,i)
            ! fcthhh(k,i)=  fctcgs(i)+fctnpc(i)+fctvdwdf(k,i)
            fcthhh(k,i)=  fctcgslt(k,i)+fctnpc(i)+fctvdwdf(k,i)
         enddo
      enddo

      do i=1,reanat
         if (i == 1) then
            a=1
         else
            a=fct1ll(i-1)+1
         endif
         b=fct1ll(i)
         do j=a,b
            u=i
            v=fct1lb(j)
            vux=x(V)-x(U)
            vuy=y(V)-y(U)
            vuz=z(V)-z(U)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            if ((fctdvu < fct1cn(fctidx(U)))  .or. &
                (fctdvu < fct1cn(fctidx(V)))) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(U))) then
#ifdef FACTS_IMOVE
               if(imove(V)==0) then                                    
#endif
                  if(Ubl==Vbl) then
                     fct03vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctivu
                     fct04vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctsvu*fctivu
                     fct05vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))
                     fct06vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctsvu
                     fct07vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctivu
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuz*vuz

                     do k=1,2
                        fct04vuxs =(fct03vuss *fct01xs(k,U)  &
                                   +fct01xs(k,U)*fct01vuxx   &
                                   +fct01ys(k,U)*fct01vuxy   &
                                   +fct01zs(k,U)*fct01vuxz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuxs *fctis2(k,U)   &
                                   *fctmos(k,U)
                        fct04vuys =(fct03vuss *fct01ys(k,U)  &
                                   +fct01xs(k,U)*fct01vuxy   &
                                   +fct01ys(k,U)*fct01vuyy   &
                                   +fct01zs(k,U)*fct01vuyz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuys *fctis2(k,U)   &
                                   *fctmos(k,U)
                        fct04vuzs =(fct03vuss *fct01zs(k,U)  &
                                   +fct01xs(k,U)*fct01vuxz   &
                                   +fct01ys(k,U)*fct01vuyz   &
                                   +fct01zs(k,U)*fct01vuzz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuzs *fctis2(k,U)   &
                                   *fctmos(k,U)

                        fctqsvux  = fct02vuxs*fctqdv(k,U) &
                                   +fct04vuxs*fctqds(k,U)
                        fctqsvuy  = fct02vuys*fctqdv(k,U) &
                                   +fct04vuys*fctqds(k,U)
                        fctqsvuz  = fct02vuzs*fctqdv(k,U) &
                                   +fct04vuzs*fctqds(k,U)
                        fctusvux  = fct02vuxs*fctudv(k,U) &
                                   +fct04vuxs*fctuds(k,U)
                        fctusvuy  = fct02vuys*fctudv(k,U) &
                                   +fct04vuys*fctuds(k,U)
                        fctusvuz  = fct02vuzs*fctudv(k,U) &
                                   +fct04vuzs*fctuds(k,U)
                        fctvsvux  = fct02vuxs*fctvdv(k,U) &
                                   +fct04vuxs*fctvds(k,U)
                        fctvsvuy  = fct02vuys*fctvdv(k,U) &
                                   +fct04vuys*fctvds(k,U)
                        fctvsvuz  = fct02vuzs*fctvdv(k,U) &
                                   +fct04vuzs*fctvds(k,U)
                        fctwsvux  = fct02vuxs*fctwdv(k,U) &
                                   +fct04vuxs*fctwds(k,U)
                        fctwsvuy  = fct02vuys*fctwdv(k,U) &
                                   +fct04vuys*fctwds(k,U)
                        fctwsvuz  = fct02vuzs*fctwdv(k,U) &
                                   +fct04vuzs*fctwds(k,U)

                        tmpsix(k) = fctkkk(k,U)*fctqsvux
                        tmpsiy(k) = fctkkk(k,U)*fctqsvuy
                        tmpsiz(k) = fctkkk(k,U)*fctqsvuz

#ifdef FACTS_IMOVE
                        if(imove(U)==0) then                           
#endif
                           tmpsix(k) = tmpsix(k)            &
                                      +fcthhh(k,U)*fctqsvux &
                                      +fctvsg(k,U)*fctusvux &
                                      +fctusg(k,U)*fctvsvux &
                                      +fctwsvux
                           tmpsiy(k) = tmpsiy(k)            &
                                      +fcthhh(k,U)*fctqsvuy &
                                      +fctvsg(k,U)*fctusvuy &
                                      +fctusg(k,U)*fctvsvuy &
                                      +fctwsvuy
                           tmpsiz(k) = tmpsiz(k)            &
                                      +fcthhh(k,U)*fctqsvuz &
                                      +fctvsg(k,U)*fctusvuz &
                                      +fctusg(k,U)*fctvsvuz &
                                      +fctwsvuz
#ifdef FACTS_IMOVE
                        endif                                          
#endif
                     enddo

                     fctsix(V) = fctsix(V)                 &
                                +tmpsix(2)*(blcoe(kk)-one) &
                                +tmpsix(1)
                     fctsiy(V) = fctsiy(V)                 &
                                +tmpsiy(2)*(blcoe(kk)-one) &
                                +tmpsiy(1)
                     fctsiz(V) = fctsiz(V)                 &
                                +tmpsiz(2)*(blcoe(kk)-one) &
                                +tmpsiz(1)
                  else
                     fct03vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctivu*blcoe(kk)
                     fct04vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctsvu*fctivu*blcoe(kk)
                     fct05vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))*blcoe(kk)
                     fct06vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctsvu*blcoe(kk)
                     fct07vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctivu*blcoe(kk)
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuz*vuz

                     fct04vuxs =(fct03vuss *fct01xs(1,U)  &
                                +fct01xs(1,U)*fct01vuxx   &
                                +fct01ys(1,U)*fct01vuxy   &
                                +fct01zs(1,U)*fct01vuxz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuxs *fctis2(1,U)   &
                                *fctmos(1,U)
                     fct04vuys =(fct03vuss *fct01ys(1,U)  &
                                +fct01xs(1,U)*fct01vuxy   &
                                +fct01ys(1,U)*fct01vuyy   &
                                +fct01zs(1,U)*fct01vuyz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuys *fctis2(1,U)   &
                                *fctmos(1,U)
                     fct04vuzs =(fct03vuss *fct01zs(1,U)  &
                                +fct01xs(1,U)*fct01vuxz   &
                                +fct01ys(1,U)*fct01vuyz   &
                                +fct01zs(1,U)*fct01vuzz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuzs *fctis2(1,U)   &
                                *fctmos(1,U)

                     fctqsvux  = fct02vuxs*fctqdv(1,U) &
                                +fct04vuxs*fctqds(1,U)
                     fctqsvuy  = fct02vuys*fctqdv(1,U) &
                                +fct04vuys*fctqds(1,U)
                     fctqsvuz  = fct02vuzs*fctqdv(1,U) &
                                +fct04vuzs*fctqds(1,U)
                     fctusvux  = fct02vuxs*fctudv(1,U) &
                                +fct04vuxs*fctuds(1,U)
                     fctusvuy  = fct02vuys*fctudv(1,U) &
                                +fct04vuys*fctuds(1,U)
                     fctusvuz  = fct02vuzs*fctudv(1,U) &
                                +fct04vuzs*fctuds(1,U)
                     fctvsvux  = fct02vuxs*fctvdv(1,U) &
                                +fct04vuxs*fctvds(1,U)
                     fctvsvuy  = fct02vuys*fctvdv(1,U) &
                                +fct04vuys*fctvds(1,U)
                     fctvsvuz  = fct02vuzs*fctvdv(1,U) &
                                +fct04vuzs*fctvds(1,U)
                     fctwsvux  = fct02vuxs*fctwdv(1,U) &
                                +fct04vuxs*fctwds(1,U)
                     fctwsvuy  = fct02vuys*fctwdv(1,U) &
                                +fct04vuys*fctwds(1,U)
                     fctwsvuz  = fct02vuzs*fctwdv(1,U) &
                                +fct04vuzs*fctwds(1,U)

                     fctsix(V) = fctsix(V)            &
                                +fctkkk(1,U)*fctqsvux
                     fctsiy(V) = fctsiy(V)            &
                                +fctkkk(1,U)*fctqsvuy
                     fctsiz(V) = fctsiz(V)            &
                                +fctkkk(1,U)*fctqsvuz

#ifdef FACTS_IMOVE
                     if(imove(U)==0) then                              
#endif
                        fctsix(V) = fctsix(V)            &
                                   +fcthhh(1,U)*fctqsvux &
                                   +fctvsg(1,U)*fctusvux &
                                   +fctusg(1,U)*fctvsvux &
                                   +fctwsvux
                        fctsiy(V) = fctsiy(V)            &
                                   +fcthhh(1,U)*fctqsvuy &
                                   +fctvsg(1,U)*fctusvuy &
                                   +fctusg(1,U)*fctvsvuy &
                                   +fctwsvuy
                        fctsiz(V) = fctsiz(V)            &
                                   +fcthhh(1,U)*fctqsvuz &
                                   +fctvsg(1,U)*fctusvuz &
                                   +fctusg(1,U)*fctvsvuz &
                                   +fctwsvuz
#ifdef FACTS_IMOVE
                     endif                                             
#endif
                  endif
#ifdef FACTS_IMOVE
               endif                                                   
#endif
            endif
            if (fctdvu < fct1cn(fctidx(V))) then
#ifdef FACTS_IMOVE
               if(imove(u)==0) then                                    
#endif
                  if(Ubl==Vbl) then
                     fct03uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctivu
                     fct04uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctsvu*fctivu
                     fct05uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))
                     fct06uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctsvu
                     fct07uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctivu
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuz*vuz

                     do k=1,2
                        fct04uvxs =(fct03uvss *fct01xs(k,V)  &
                                   +fct01xs(k,V)*fct01uvxx   &
                                   +fct01ys(k,V)*fct01uvxy   &
                                   +fct01zs(k,V)*fct01uvxz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvxs *fctis2(k,V)   &
                                   *fctmos(k,V)
                        fct04uvys =(fct03uvss *fct01ys(k,V)  &
                                   +fct01xs(k,V)*fct01uvxy   &
                                   +fct01ys(k,V)*fct01uvyy   &
                                   +fct01zs(k,V)*fct01uvyz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvys *fctis2(k,V)   &
                                   *fctmos(k,V)
                        fct04uvzs =(fct03uvss *fct01zs(k,V)  &
                                   +fct01xs(k,V)*fct01uvxz   &
                                   +fct01ys(k,V)*fct01uvyz   &
                                   +fct01zs(k,V)*fct01uvzz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvzs *fctis2(k,V)   &
                                   *fctmos(k,V)

                        fctqsuvx  = fct02uvxs*fctqdv(k,V) &
                                   +fct04uvxs*fctqds(k,V)
                        fctqsuvy  = fct02uvys*fctqdv(k,V) &
                                   +fct04uvys*fctqds(k,V)
                        fctqsuvz  = fct02uvzs*fctqdv(k,V) &
                                   +fct04uvzs*fctqds(k,V)
                        fctusuvx  = fct02uvxs*fctudv(k,V) &
                                   +fct04uvxs*fctuds(k,V)
                        fctusuvy  = fct02uvys*fctudv(k,V) &
                                   +fct04uvys*fctuds(k,V)
                        fctusuvz  = fct02uvzs*fctudv(k,V) &
                                   +fct04uvzs*fctuds(k,V)
                        fctvsuvx  = fct02uvxs*fctvdv(k,V) &
                                   +fct04uvxs*fctvds(k,V)
                        fctvsuvy  = fct02uvys*fctvdv(k,V) &
                                   +fct04uvys*fctvds(k,V)
                        fctvsuvz  = fct02uvzs*fctvdv(k,V) &
                                   +fct04uvzs*fctvds(k,V)
                        fctwsuvx  = fct02uvxs*fctwdv(k,V) &
                                   +fct04uvxs*fctwds(k,V)
                        fctwsuvy  = fct02uvys*fctwdv(k,V) &
                                   +fct04uvys*fctwds(k,V)
                        fctwsuvz  = fct02uvzs*fctwdv(k,V) &
                                   +fct04uvzs*fctwds(k,V)

                        tmpsix(k) = fctkkk(k,V)*fctqsuvx
                        tmpsiy(k) = fctkkk(k,V)*fctqsuvy
                        tmpsiz(k) = fctkkk(k,V)*fctqsuvz

#ifdef FACTS_IMOVE
                        if(imove(V)==0) then                           
#endif
                           tmpsix(k) = tmpsix(k)            &
                                      +fcthhh(k,V)*fctqsuvx &
                                      +fctvsg(k,V)*fctusuvx &
                                      +fctusg(k,V)*fctvsuvx &
                                      +fctwsuvx
                           tmpsiy(k) = tmpsiy(k)            &
                                      +fcthhh(k,V)*fctqsuvy &
                                      +fctvsg(k,V)*fctusuvy &
                                      +fctusg(k,V)*fctvsuvy &
                                      +fctwsuvy
                           tmpsiz(k) = tmpsiz(k)            &
                                      +fcthhh(k,V)*fctqsuvz &
                                      +fctvsg(k,V)*fctusuvz &
                                      +fctusg(k,V)*fctvsuvz &
                                      +fctwsuvz
#ifdef FACTS_IMOVE
                        endif                                          
#endif
                     enddo
                     fctsix(U) = fctsix(U)                 &
                                +tmpsix(2)*(blcoe(kk)-one) &
                                +tmpsix(1)
                     fctsiy(U) = fctsiy(U)                 &
                                +tmpsiy(2)*(blcoe(kk)-one) &
                                +tmpsiy(1)
                     fctsiz(U) = fctsiz(U)                 &
                                +tmpsiz(2)*(blcoe(kk)-one) &
                                +tmpsiz(1)
                  else
                     fct03uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctivu*blcoe(kk)
                     fct04uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctsvu*fctivu*blcoe(kk)
                     fct05uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))*blcoe(kk)
                     fct06uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctsvu*blcoe(kk)
                     fct07uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctivu*blcoe(kk)
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuz*vuz

                     fct04uvxs =(fct03uvss *fct01xs(1,V)  &
                                +fct01xs(1,V)*fct01uvxx   &
                                +fct01ys(1,V)*fct01uvxy   &
                                +fct01zs(1,V)*fct01uvxz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvxs *fctis2(1,V)   &
                                *fctmos(1,V)
                     fct04uvys =(fct03uvss *fct01ys(1,V)  &
                                +fct01xs(1,V)*fct01uvxy   &
                                +fct01ys(1,V)*fct01uvyy   &
                                +fct01zs(1,V)*fct01uvyz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvys *fctis2(1,V)   &
                                *fctmos(1,V)
                     fct04uvzs =(fct03uvss *fct01zs(1,V)  &
                                +fct01xs(1,V)*fct01uvxz   &
                                +fct01ys(1,V)*fct01uvyz   &
                                +fct01zs(1,V)*fct01uvzz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvzs *fctis2(1,V)   &
                                *fctmos(1,V)

                     fctqsuvx  = fct02uvxs*fctqdv(1,V) &
                                +fct04uvxs*fctqds(1,V)
                     fctqsuvy  = fct02uvys*fctqdv(1,V) &
                                +fct04uvys*fctqds(1,V)
                     fctqsuvz  = fct02uvzs*fctqdv(1,V) &
                                +fct04uvzs*fctqds(1,V)
                     fctusuvx  = fct02uvxs*fctudv(1,V) &
                                +fct04uvxs*fctuds(1,V)
                     fctusuvy  = fct02uvys*fctudv(1,V) &
                                +fct04uvys*fctuds(1,V)
                     fctusuvz  = fct02uvzs*fctudv(1,V) &
                                +fct04uvzs*fctuds(1,V)
                     fctvsuvx  = fct02uvxs*fctvdv(1,V) &
                                +fct04uvxs*fctvds(1,V)
                     fctvsuvy  = fct02uvys*fctvdv(1,V) &
                                +fct04uvys*fctvds(1,V)
                     fctvsuvz  = fct02uvzs*fctvdv(1,V) &
                                +fct04uvzs*fctvds(1,V)
                     fctwsuvx  = fct02uvxs*fctwdv(1,V) &
                                +fct04uvxs*fctwds(1,V)
                     fctwsuvy  = fct02uvys*fctwdv(1,V) &
                                +fct04uvys*fctwds(1,V)
                     fctwsuvz  = fct02uvzs*fctwdv(1,V) &
                                +fct04uvzs*fctwds(1,V)

                     fctsix(U) = fctsix(U)            &
                                +fctkkk(1,V)*fctqsuvx
                     fctsiy(U) = fctsiy(U)            &
                                +fctkkk(1,V)*fctqsuvy
                     fctsiz(U) = fctsiz(U)            &
                                +fctkkk(1,V)*fctqsuvz

#ifdef FACTS_IMOVE
                     if(imove(V)==0) then                              
#endif
                        fctsix(U) = fctsix(U)            &
                                   +fcthhh(1,V)*fctqsuvx &
                                   +fctvsg(1,V)*fctusuvx &
                                   +fctusg(1,V)*fctvsuvx &
                                   +fctwsuvx
                        fctsiy(U) = fctsiy(U)            &
                                   +fcthhh(1,V)*fctqsuvy &
                                   +fctvsg(1,V)*fctusuvy &
                                   +fctusg(1,V)*fctvsuvy &
                                   +fctwsuvy
                        fctsiz(U) = fctsiz(U)            &
                                   +fcthhh(1,V)*fctqsuvz &
                                   +fctvsg(1,V)*fctusuvz &
                                   +fctusg(1,V)*fctvsuvz &
                                   +fctwsuvz
#ifdef FACTS_IMOVE
                     endif                                             
#endif
                  endif
#ifdef FACTS_IMOVE
               endif                                                   
#endif
            endif
         enddo
      enddo

      do i=reanat+1,imanat
         if (i == reanat+1) then
            a=1
         else
            a=fct3ll(i-1)+1
         endif
         b=fct3ll(i)
         do j=a,b
            u=i
            v=fct3lb(j)
            vux=x(V)-x(U)
            vuy=y(V)-y(U)
            vuz=z(V)-z(U)
            ! fctdvu=vux*vux+vuy*vuy+vuz*vuz
            fctdvu=max(RSMALL,vux*vux+vuy*vuy+vuz*vuz)
            u=imattr(i)
            if ((fctdvu < fct1cn(fctidx(U)))  .or. &
                  (fctdvu < fct1cn(fctidx(V)))) then
               ! -------------------------------
               Ubl = iblock(U)
               Vbl = iblock(V)
               kk=max(Ubl,Vbl)
               kk=kk*(kk-1)/2+min(Ubl,Vbl)
               ! -------------------------------
               fctivu=one/fctdvu
               fctsvu=sqrt(fctivu)
            endif
            if (fctdvu < fct1cn(fctidx(U))) then
#ifdef FACTS_IMOVE
               if(imove(V)==0) then                                    
#endif
                  if(Ubl==Vbl) then
                     fct03vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctivu
                     fct04vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctsvu*fctivu
                     fct05vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))
                     fct06vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctsvu
                     fct07vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctivu
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuz*vuz

                     do k=1,2
                        fct04vuxs =(fct03vuss *fct01xs(k,U)  &
                                   +fct01xs(k,U)*fct01vuxx   &
                                   +fct01ys(k,U)*fct01vuxy   &
                                   +fct01zs(k,U)*fct01vuxz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuxs *fctis2(k,U)   &
                                   *fctmos(k,U)
                        fct04vuys =(fct03vuss *fct01ys(k,U)  &
                                   +fct01xs(k,U)*fct01vuxy   &
                                   +fct01ys(k,U)*fct01vuyy   &
                                   +fct01zs(k,U)*fct01vuyz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuys *fctis2(k,U)   &
                                   *fctmos(k,U)
                        fct04vuzs =(fct03vuss *fct01zs(k,U)  &
                                   +fct01xs(k,U)*fct01vuxz   &
                                   +fct01ys(k,U)*fct01vuyz   &
                                   +fct01zs(k,U)*fct01vuzz)  &
                                   *fctis1(k,U) *fctis2(k,U) &
                                   -fct03vuzs *fctis2(k,U)   &
                                   *fctmos(k,U)

                        fctqsvux  = fct02vuxs*fctqdv(k,U) &
                                   +fct04vuxs*fctqds(k,U)
                        fctqsvuy  = fct02vuys*fctqdv(k,U) &
                                   +fct04vuys*fctqds(k,U)
                        fctqsvuz  = fct02vuzs*fctqdv(k,U) &
                                   +fct04vuzs*fctqds(k,U)
                        fctusvux  = fct02vuxs*fctudv(k,U) &
                                   +fct04vuxs*fctuds(k,U)
                        fctusvuy  = fct02vuys*fctudv(k,U) &
                                   +fct04vuys*fctuds(k,U)
                        fctusvuz  = fct02vuzs*fctudv(k,U) &
                                   +fct04vuzs*fctuds(k,U)
                        fctvsvux  = fct02vuxs*fctvdv(k,U) &
                                   +fct04vuxs*fctvds(k,U)
                        fctvsvuy  = fct02vuys*fctvdv(k,U) &
                                   +fct04vuys*fctvds(k,U)
                        fctvsvuz  = fct02vuzs*fctvdv(k,U) &
                                   +fct04vuzs*fctvds(k,U)
                        fctwsvux  = fct02vuxs*fctwdv(k,U) &
                                   +fct04vuxs*fctwds(k,U)
                        fctwsvuy  = fct02vuys*fctwdv(k,U) &
                                   +fct04vuys*fctwds(k,U)
                        fctwsvuz  = fct02vuzs*fctwdv(k,U) &
                                   +fct04vuzs*fctwds(k,U)

                        tmpsix(k) = fctkkk(k,U)*fctqsvux
                        tmpsiy(k) = fctkkk(k,U)*fctqsvuy
                        tmpsiz(k) = fctkkk(k,U)*fctqsvuz

#ifdef FACTS_IMOVE
                        if(imove(U)==0) then                           
#endif
                           tmpsix(k) = tmpsix(k)            &
                                      +fcthhh(k,U)*fctqsvux &
                                      +fctvsg(k,U)*fctusvux &
                                      +fctusg(k,U)*fctvsvux &
                                      +fctwsvux
                           tmpsiy(k) = tmpsiy(k)            &
                                      +fcthhh(k,U)*fctqsvuy &
                                      +fctvsg(k,U)*fctusvuy &
                                      +fctusg(k,U)*fctvsvuy &
                                      +fctwsvuy
                           tmpsiz(k) = tmpsiz(k)            &
                                      +fcthhh(k,U)*fctqsvuz &
                                      +fctvsg(k,U)*fctusvuz &
                                      +fctusg(k,U)*fctvsvuz &
                                      +fctwsvuz
#ifdef FACTS_IMOVE
                        endif                                          
#endif
                     enddo
                     fctsix(V) = fctsix(V)                 &
                                +tmpsix(2)*(blcoe(kk)-one) &
                                +tmpsix(1)
                     fctsiy(V) = fctsiy(V)                 &
                                +tmpsiy(2)*(blcoe(kk)-one) &
                                +tmpsiy(1)
                     fctsiz(V) = fctsiz(V)                 &
                                +tmpsiz(2)*(blcoe(kk)-one) &
                                +tmpsiz(1)
                  else
                     fct03vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctivu*blcoe(kk)
                     fct04vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fctsvu*fctivu*blcoe(kk)
                     fct05vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))*blcoe(kk)
                     fct06vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctsvu*blcoe(kk)
                     fct07vuss = fctvol(fctidx(V))              &
                                *(one-fctdvu*fct1ci(fctidx(U))) &
                                *fct1fn(fctidx(U))              &
                                *fctivu*blcoe(kk)
                     fct02vuxs = fct05vuss*vux
                     fct02vuys = fct05vuss*vuy
                     fct02vuzs = fct05vuss*vuz
                     fct03vuxs = (-fct04vuss+fct06vuss)*vux
                     fct03vuys = (-fct04vuss+fct06vuss)*vuy
                     fct03vuzs = (-fct04vuss+fct06vuss)*vuz
                     fct01vuxx = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vux
                     fct01vuxy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuy
                     fct01vuxz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vux*vuz
                     fct01vuyy = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuy
                     fct01vuyz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuy*vuz
                     fct01vuzz = half*fct07vuss*(one+fct1cn(fctidx(U)) &
                                *fctivu)*vuz*vuz

                     fct04vuxs =(fct03vuss *fct01xs(1,U)  &
                                +fct01xs(1,U)*fct01vuxx   &
                                +fct01ys(1,U)*fct01vuxy   &
                                +fct01zs(1,U)*fct01vuxz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuxs *fctis2(1,U)   &
                                *fctmos(1,U)
                     fct04vuys =(fct03vuss *fct01ys(1,U)  &
                                +fct01xs(1,U)*fct01vuxy   &
                                +fct01ys(1,U)*fct01vuyy   &
                                +fct01zs(1,U)*fct01vuyz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuys *fctis2(1,U)   &
                                *fctmos(1,U)
                     fct04vuzs =(fct03vuss *fct01zs(1,U)  &
                                +fct01xs(1,U)*fct01vuxz   &
                                +fct01ys(1,U)*fct01vuyz   &
                                +fct01zs(1,U)*fct01vuzz)  &
                                *fctis1(1,U) *fctis2(1,U) &
                                -fct03vuzs *fctis2(1,U)   &
                                *fctmos(1,U)

                     fctqsvux  = fct02vuxs*fctqdv(1,U) &
                                +fct04vuxs*fctqds(1,U)
                     fctqsvuy  = fct02vuys*fctqdv(1,U) &
                                +fct04vuys*fctqds(1,U)
                     fctqsvuz  = fct02vuzs*fctqdv(1,U) &
                                +fct04vuzs*fctqds(1,U)
                     fctusvux  = fct02vuxs*fctudv(1,U) &
                                +fct04vuxs*fctuds(1,U)
                     fctusvuy  = fct02vuys*fctudv(1,U) &
                                +fct04vuys*fctuds(1,U)
                     fctusvuz  = fct02vuzs*fctudv(1,U) &
                                +fct04vuzs*fctuds(1,U)
                     fctvsvux  = fct02vuxs*fctvdv(1,U) &
                                +fct04vuxs*fctvds(1,U)
                     fctvsvuy  = fct02vuys*fctvdv(1,U) &
                                +fct04vuys*fctvds(1,U)
                     fctvsvuz  = fct02vuzs*fctvdv(1,U) &
                                +fct04vuzs*fctvds(1,U)
                     fctwsvux  = fct02vuxs*fctwdv(1,U) &
                                +fct04vuxs*fctwds(1,U)
                     fctwsvuy  = fct02vuys*fctwdv(1,U) &
                                +fct04vuys*fctwds(1,U)
                     fctwsvuz  = fct02vuzs*fctwdv(1,U) &
                                +fct04vuzs*fctwds(1,U)

                     fctsix(V) = fctsix(V)            &
                                +fctkkk(1,U)*fctqsvux
                     fctsiy(V) = fctsiy(V)            &
                                +fctkkk(1,U)*fctqsvuy
                     fctsiz(V) = fctsiz(V)            &
                                +fctkkk(1,U)*fctqsvuz

#ifdef FACTS_IMOVE
                     if(imove(U)==0) then                              
#endif
                        fctsix(V) = fctsix(V)            &
                                   +fcthhh(1,U)*fctqsvux &
                                   +fctvsg(1,U)*fctusvux &
                                   +fctusg(1,U)*fctvsvux &
                                   +fctwsvux
                        fctsiy(V) = fctsiy(V)            &
                                   +fcthhh(1,U)*fctqsvuy &
                                   +fctvsg(1,U)*fctusvuy &
                                   +fctusg(1,U)*fctvsvuy &
                                   +fctwsvuy
                        fctsiz(V) = fctsiz(V)            &
                                   +fcthhh(1,U)*fctqsvuz &
                                   +fctvsg(1,U)*fctusvuz &
                                   +fctusg(1,U)*fctvsvuz &
                                   +fctwsvuz
#ifdef FACTS_IMOVE
                     endif                                             
#endif
                  endif
#ifdef FACTS_IMOVE
               endif                                                   
#endif
            endif
            if (fctdvu < fct1cn(fctidx(V))) then
#ifdef FACTS_IMOVE
               if(imove(U)==0) then                                    
#endif
                  if(Ubl==Vbl) then
                     fct03uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctivu
                     fct04uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctsvu*fctivu
                     fct05uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))
                     fct06uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctsvu
                     fct07uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctivu
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuz*vuz

                     do k=1,2
                        fct04uvxs =(fct03uvss *fct01xs(k,V)  &
                                   +fct01xs(k,V)*fct01uvxx   &
                                   +fct01ys(k,V)*fct01uvxy   &
                                   +fct01zs(k,V)*fct01uvxz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvxs *fctis2(k,V)   &
                                   *fctmos(k,V)
                        fct04uvys =(fct03uvss *fct01ys(k,V)  &
                                   +fct01xs(k,V)*fct01uvxy   &
                                   +fct01ys(k,V)*fct01uvyy   &
                                   +fct01zs(k,V)*fct01uvyz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvys *fctis2(k,V)   &
                                   *fctmos(k,V)
                        fct04uvzs =(fct03uvss *fct01zs(k,V)  &
                                   +fct01xs(k,V)*fct01uvxz   &
                                   +fct01ys(k,V)*fct01uvyz   &
                                   +fct01zs(k,V)*fct01uvzz)  &
                                   *fctis1(k,V) *fctis2(k,V) &
                                   -fct03uvzs *fctis2(k,V)   &
                                   *fctmos(k,V)

                        fctqsuvx  = fct02uvxs*fctqdv(k,V) &
                                   +fct04uvxs*fctqds(k,V)
                        fctqsuvy  = fct02uvys*fctqdv(k,V) &
                                   +fct04uvys*fctqds(k,V)
                        fctqsuvz  = fct02uvzs*fctqdv(k,V) &
                                   +fct04uvzs*fctqds(k,V)
                        fctusuvx  = fct02uvxs*fctudv(k,V) &
                                   +fct04uvxs*fctuds(k,V)
                        fctusuvy  = fct02uvys*fctudv(k,V) &
                                   +fct04uvys*fctuds(k,V)
                        fctusuvz  = fct02uvzs*fctudv(k,V) &
                                   +fct04uvzs*fctuds(k,V)
                        fctvsuvx  = fct02uvxs*fctvdv(k,V) &
                                   +fct04uvxs*fctvds(k,V)
                        fctvsuvy  = fct02uvys*fctvdv(k,V) &
                                   +fct04uvys*fctvds(k,V)
                        fctvsuvz  = fct02uvzs*fctvdv(k,V) &
                                   +fct04uvzs*fctvds(k,V)
                        fctwsuvx  = fct02uvxs*fctwdv(k,V) &
                                   +fct04uvxs*fctwds(k,V)
                        fctwsuvy  = fct02uvys*fctwdv(k,V) &
                                   +fct04uvys*fctwds(k,V)
                        fctwsuvz  = fct02uvzs*fctwdv(k,V) &
                                   +fct04uvzs*fctwds(k,V)

                        tmpsix(k) = fctkkk(k,V)*fctqsuvx
                        tmpsiy(k) = fctkkk(k,V)*fctqsuvy
                        tmpsiz(k) = fctkkk(k,V)*fctqsuvz

#ifdef FACTS_IMOVE
                        if(imove(V)==0) then                           
#endif
                           tmpsix(k) = tmpsix(k)            &
                                      +fcthhh(k,V)*fctqsuvx &
                                      +fctvsg(k,V)*fctusuvx &
                                      +fctusg(k,V)*fctvsuvx &
                                      +fctwsuvx
                           tmpsiy(k) = tmpsiy(k)            &
                                      +fcthhh(k,V)*fctqsuvy &
                                      +fctvsg(k,V)*fctusuvy &
                                      +fctusg(k,V)*fctvsuvy &
                                      +fctwsuvy
                           tmpsiz(k) = tmpsiz(k)            &
                                      +fcthhh(k,V)*fctqsuvz &
                                      +fctvsg(k,V)*fctusuvz &
                                      +fctusg(k,V)*fctvsuvz &
                                      +fctwsuvz
#ifdef FACTS_IMOVE
                        endif                                          
#endif
                     enddo
                     fctsix(U) = fctsix(U)                 &
                                +tmpsix(2)*(blcoe(kk)-one) &
                                +tmpsix(1)
                     fctsiy(U) = fctsiy(U)                 &
                                +tmpsiy(2)*(blcoe(kk)-one) &
                                +tmpsiy(1)
                     fctsiz(U) = fctsiz(U)                 &
                                +tmpsiz(2)*(blcoe(kk)-one) &
                                +tmpsiz(1)
                  else
                     fct03uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctivu*blcoe(kk)
                     fct04uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fctsvu*fctivu*blcoe(kk)
                     fct05uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))*blcoe(kk)
                     fct06uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctsvu*blcoe(kk)
                     fct07uvss = fctvol(fctidx(U))              &
                                *(one-fctdvu*fct1ci(fctidx(V))) &
                                *fct1fn(fctidx(V))              &
                                *fctivu*blcoe(kk)
                     fct02uvxs =-fct05uvss*vux
                     fct02uvys =-fct05uvss*vuy
                     fct02uvzs =-fct05uvss*vuz
                     fct03uvxs =-(-fct04uvss+fct06uvss)*vux
                     fct03uvys =-(-fct04uvss+fct06uvss)*vuy
                     fct03uvzs =-(-fct04uvss+fct06uvss)*vuz
                     fct01uvxx = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vux
                     fct01uvxy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuy
                     fct01uvxz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vux*vuz
                     fct01uvyy = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuy
                     fct01uvyz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuy*vuz
                     fct01uvzz = half*fct07uvss*(one+fct1cn(fctidx(V)) &
                                *fctivu)*vuz*vuz

                     fct04uvxs =(fct03uvss *fct01xs(1,V)  &
                                +fct01xs(1,V)*fct01uvxx   &
                                +fct01ys(1,V)*fct01uvxy   &
                                +fct01zs(1,V)*fct01uvxz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvxs *fctis2(1,V)   &
                                *fctmos(1,V)
                     fct04uvys =(fct03uvss *fct01ys(1,V)  &
                                +fct01xs(1,V)*fct01uvxy   &
                                +fct01ys(1,V)*fct01uvyy   &
                                +fct01zs(1,V)*fct01uvyz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvys *fctis2(1,V)   &
                                *fctmos(1,V)
                     fct04uvzs =(fct03uvss *fct01zs(1,V)  &
                                +fct01xs(1,V)*fct01uvxz   &
                                +fct01ys(1,V)*fct01uvyz   &
                                +fct01zs(1,V)*fct01uvzz)  &
                                *fctis1(1,V) *fctis2(1,V) &
                                -fct03uvzs *fctis2(1,V)   &
                                *fctmos(1,V)

                     fctqsuvx  = fct02uvxs*fctqdv(1,V) &
                                +fct04uvxs*fctqds(1,V)
                     fctqsuvy  = fct02uvys*fctqdv(1,V) &
                                +fct04uvys*fctqds(1,V)
                     fctqsuvz  = fct02uvzs*fctqdv(1,V) &
                                +fct04uvzs*fctqds(1,V)
                     fctusuvx  = fct02uvxs*fctudv(1,V) &
                                +fct04uvxs*fctuds(1,V)
                     fctusuvy  = fct02uvys*fctudv(1,V) &
                                +fct04uvys*fctuds(1,V)
                     fctusuvz  = fct02uvzs*fctudv(1,V) &
                                +fct04uvzs*fctuds(1,V)
                     fctvsuvx  = fct02uvxs*fctvdv(1,V) &
                                +fct04uvxs*fctvds(1,V)
                     fctvsuvy  = fct02uvys*fctvdv(1,V) &
                                +fct04uvys*fctvds(1,V)
                     fctvsuvz  = fct02uvzs*fctvdv(1,V) &
                                +fct04uvzs*fctvds(1,V)
                     fctwsuvx  = fct02uvxs*fctwdv(1,V) &
                                +fct04uvxs*fctwds(1,V)
                     fctwsuvy  = fct02uvys*fctwdv(1,V) &
                                +fct04uvys*fctwds(1,V)
                     fctwsuvz  = fct02uvzs*fctwdv(1,V) &
                                +fct04uvzs*fctwds(1,V)

                     fctsix(U) = fctsix(U)            &
                                +fctkkk(1,V)*fctqsuvx
                     fctsiy(U) = fctsiy(U)            &
                                +fctkkk(1,V)*fctqsuvy
                     fctsiz(U) = fctsiz(U)            &
                                +fctkkk(1,V)*fctqsuvz

#ifdef FACTS_IMOVE
                     if(imove(V)==0) then                              
#endif
                        fctsix(U) = fctsix(U)            &
                                   +fcthhh(1,V)*fctqsuvx &
                                   +fctvsg(1,V)*fctusuvx &
                                   +fctusg(1,V)*fctvsuvx &
                                   +fctwsuvx
                        fctsiy(U) = fctsiy(U)            &
                                   +fcthhh(1,V)*fctqsuvy &
                                   +fctvsg(1,V)*fctusuvy &
                                   +fctusg(1,V)*fctvsuvy &
                                   +fctwsuvy
                        fctsiz(U) = fctsiz(U)            &
                                   +fcthhh(1,V)*fctqsuvz &
                                   +fctvsg(1,V)*fctusuvz &
                                   +fctusg(1,V)*fctvsuvz &
                                   +fctwsuvz
#ifdef FACTS_IMOVE
                     endif                                             
#endif
                  endif
#ifdef FACTS_IMOVE
               endif                                                   
#endif
            endif
         enddo
      enddo
