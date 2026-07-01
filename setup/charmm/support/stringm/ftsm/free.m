% the force[yy].dat files

if ~exist('nofig')
 nofig=0;
end

if ~exist('styles')
 styles={'r-','g-','b-','m-','c-','k-','r--','g--','b--','m--','c--','k--'};
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%% load multiple force files listed in quotes %%%%%%%%%%%%
fnames={'./force.dat'}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (~exist('read'))
 read=1;
end
read=1
if (read==1)
%
clear data;
for i=1:length(fnames)
 fname=char(fnames(i));
 if (i==1)
  fc=load(fname);
 else
  fc=[fc; load(fname)];
 end 
end
%
[niter,m]=size(fc);
ncv=3;
niter=niter/ncv;

cv=[]; for i=1:ncv ; cv=[cv ; 1:m]; end ;

[ncv,m]=size(cv);

r=zeros(ncv,m,niter);
f=zeros(ncv,m,niter);

row=1;
for i=1:niter
 for j=1:ncv
  r(j,:,i)=cv(j,:);
  f(j,:,i)=fc(row,:); row=row+1;
 end
end
read=0;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ind=[1:2];  %include planar force and curvature at index 2
%ind=[1:1]; %planar force only

dr=r(ind,2:end,:)-r(ind,1:end-1,:); 
fc=0.5*(f(ind,1:end-1,:)+f(ind,2:end,:)); 

dw=-squeeze(sum(dr.*fc,1));
dw=reshape(dw,m-1,niter) ; % fix for single sample point to get dimensions correct

work=zeros(m,niter);
for i=2:m
 work(i,:)=work(i-1,:)+dw(i-1,:);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% bin and plot average work curves %%%%%%%%%
%
ie   =niter;
ib   =1;
ib=round(niter * 0.5);
nbox =2;
bsize=ceil( (ie-ib+1)/nbox);

lw=1.;

%
nsample=floor((ie-ib+1)/bsize)+sign(mod(ie-ib+1,bsize));
fe=zeros(nsample,m);
j=1;
for i=ib:bsize:ie
 fe(j,:)=mean(work(:, i : i+min(bsize-1,ie-i) ),2);
 j=j+1;
% leg=[leg {['iteration ',num2str(i-1)]}];
end 

if ~nofig
 close all;
 figure('position',[200,200,450,350]); hold on; box on;
 leg={};
end
%
alpha=[0:m-1]; alpha=alpha/alpha(end);
%
for i=1:nsample
 plot(alpha,fe(i,1:end),[char(styles(mod(i-1,length(styles))+1)),'x'], 'linewidth', lw)
end 

fave=mean(fe,1);
fstd=std(fe,1);

%mean
%plot(alpha,fave,'rx-','linewidth',lw);
%std
%plot(alpha,fave+fstd,'k--','linewidth',1);
%plot(alpha,fave-fstd,'k--','linewidth',1);
%leg=[leg {['Average']}];

%legend(leg,2);
box on;
ylabel('\it F(\alpha) (kcal/mol)', 'fontsize',14);
xlabel('\it \alpha', 'fontsize',14);
%
xlim([0 1]);
%ylim([0 9]);
set(gcf, 'paperpositionmode', 'auto');
%print(gcf, '-dpsc', 'fe_contrib.eps');
print(gcf, '-dpsc', 'fe.eps');
print('-dtiff', '-r300','fe.tif');
%
%
return
% plot evolution of FE drop
iw=10;
dfe=zeros(1,nsample);
for i=1:nsample
 dfe(i)=min(fe(i,end-iw+1:end) - fe(i,1:iw));
end
figure(3);
plot(dfe,'k-x');
ylabel('\it Endpoint Free Energy Difference(kcal/mol)', 'fontsize',14);
xlabel('\it Iteration \# ', 'fontsize',14);

