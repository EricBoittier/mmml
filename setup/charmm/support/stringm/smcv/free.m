% plot FE computed in CHARMM by smcv
%
close all;
styles={'r-','g-','b-','m-','c-','k-','r--','g--','b--','m--','c--','k--'};
leg={};

fnames={'fe.dat'...%, 'fe2.dat', 'fe3.dat', ...% 'fe4.dat' ...
         };
%
clear data;
for i=1:length(fnames)
 fname=char(fnames(i));
 if (i==1)
  data=load(fname);
 else
  data=[data; load(fname)];
 end 
end
%

[niter, nrep]=size(data);

% compute block averages;this is faster and more meaningful if force averages are short
bsize=2500;

ib=1;
%ib=2001;
ie=niter;
%ie=100;
tfac=.001; %nanoseconds per iteration
nsample=floor((ie-ib+1)/bsize)+sign(mod(ie-ib+1,bsize));
fe=zeros(nsample,nrep-1);
j=1;
for i=ib:bsize:ie
 fe(j,:)=mean(data( i : i+min(bsize-1,ie-i) , 2:end),1);
 j=j+1;
% leg=[leg {['iteration ',num2str(i-1)]}];
 leg=[leg {['avg: ',num2str(tfac*(i-1)),' -- ',num2str(tfac*(i+min(bsize-1,ie-i))),' (ns)']}];
end 

figure; hold on;box on;

for i=1:nsample
 plot(fe(i,1:end),[char(styles(mod(i-1,length(styles))+1)),'*'], 'linewidth', 2)
% leg=[leg {['iteration ',num2str(i)]}];
end 

fave=mean(fe,1);
fstd=std(fe,1);

%mean
%plot([1:nrep-1],fave,'k-*','linewidth',3);
%std
%plot([1:nrep-1],fave+fstd,'k:','linewidth',3);
%plot([1:nrep-1],fave-fstd,'k:','linewidth',3);
%leg=[leg {['Average']}];


legend(leg,1);
box on;
ylabel('\it Free Energy (kcal/mol)', 'fontsize',14);
xlabel('\it Replica', 'fontsize',14);
set(gcf, 'paperpositionmode', 'auto');
print(gcf, '-dpsc', 'fe.eps');
print(gcf, '-djpeg100', 'fe.jpg');

