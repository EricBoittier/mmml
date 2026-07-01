% VO 12.2013
% earlier versions were called vlog[yy].m
% compared output with earliest versions of vlog3 from 2011 -- results are the same
% Old description:
% read and process complete log from V tesselation calculations
% This is tricky because on a 64 bit machine the record markers are 
% written as 4-byte integers; and they refer to the number of intervening
% 4-byte words
%
% new version supports multiple files (concatenated into one log)
% this version processes new logs (code version 1.22.10)
%
%
% user options :
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
temperature=300;%temperature used to convert probability to free energy (K)
maxdist=1;      %allow collisions between replicas separated by at most d-1 cells ; use < 1 to include _all_ fluxes
fixfluxval=0.1; %if the logs do not have collisions for some adjacent cells, use this value (zero does nothing; try 1 if unsure)
fixrate=1;      %set to 1 if want to approximate missing rates between adjacent cells using interpolation
fixrateval=1e10;%if >0 set a spurious transition rate (per second) between adjacent replicas into the q-matrix (generally >> 1)
                %fixrateval will be ignored if fixrate=1, _unless_ rate interpolation fails, in which case fixrateval will be used
maxdistrate=1;  %if set to 1, maxdist above is applied to the rate calculation (no effect if maxdist<0)
timestep=1;     %simulation step in _femtoseconds_ needed to compute the rate (in seconds)
targetmstone=...  % milestones to which the MFPT will be computed (from all other milestones): use {'all' 'first' 'last'}, or { i1 i2 ... }
               { 'last' }; 
fefile='fe.dat';%ASCII file that contains the computed free energy profile
mfptfile='mfpt.dat';% ASCII file that contains the computed hitting probabilities
dfilter=1;      %to smoothen free energy profiles set dfilter > 1
mfptffe=1;      %set to 1 to use smoothed free energy profile in hitting time (MFPT) computation
timefromzero=0; %set to 1 if the time step was reset to zero in sequential voronoi logs
guessnrep=1 ;   %set to 1 if want to guess number of replicas from log files, otherwise set below
numrep=32;      %number of replicas; ignored if guessnrep = 1
guessfilesize=0;%use system calls to determine log file size for fast reading (linux only); otherwise set below
datasize=10000000;% size of log data array (for faster reading) ; ignored if guessfilesize=1
% Voronoi log files below : use lognames = { 'name1' 'name2' ... etc }
lognames={...
...%        './ftsm/voro0.log'...
        './smcv/voro0.log'...
};
% modifications below this line should not normally be needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% might need to change values below if integer sizes change
intsize=4 ;     %bytes per int
intfmt='int32'; %integer format (essentially, 32/64 bit)
kboltz=1.987191d-3;        % boltzmann constant
hline=' ================================================================================================\n';
fprintf(hline);
fprintf(' Will compute free energy profile and mean first passage time from Voronoi tessellation logs.\n');
fprintf(hline);
if ~(exist('read'));read=1;end
if (read)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nfile=length(lognames);
data=zeros(0,1);
% loop over files
% keep track of time in case the calculation was restarted from zero each time (relevant for timefromzero=1))
toffset = 0;
ifile=1;
%nfile=1;
for j=ifile:nfile
 fname=char(lognames(j));
 fprintf([' Reading Voronoi log file ',fname,'\n'])
%get file size (linux only)
 if (guessfilesize)
  attrib=['ls -s ',fname];
  [i,attrib]=system(attrib);
% s=strread(attrib, '%s','delimiter',fname);
% s=str2num(char(s(1)))*1024/4; % file size in Kbytes * 1024/4 int per kB (since all entries are integer); OS/hardware - specific
  [s,junk]=strread(attrib, '%d%s');
  datasize=s*1024/4; % file size in Kbytes * 1024/4 int per kB (since all entries are integer); OS/hardware - specific
 end
 fid=fopen(fname,'r');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% `allocate' data: this is essential for fast speed
%data=zeros(filesize,1); % use
 d=zeros(datasize,1);
 n=fread(fid,1,'int32')/intsize; % this record delimiter (record size in Fortran binary)
 d(1:n)=fread(fid,n,intfmt);i=n+1;
 n=fread(fid,1,'int32')/intsize; % this record delimiter
 n=fread(fid,1,'int32')/intsize; % next record delimiter

 while (length(n)>0)
  d(i:i+n-1)=fread(fid,n,intfmt);i=i+n;
  n=fread(fid,1,'int32')/intsize; % this record delimiter
  n=fread(fid,1,'int32')/intsize; % next record delimiter
 end
 % trim data
 d=d(1:i-1);
% ad hoc : fix time : here assume that the data is consecutive, just the timestep numbering is offset
 if (timefromzero)
  times=d(5:5:end) + toffset;      % extract time and correct
  d(5:5:end)=times ;               % replace time
  toffset = max(times) ;           % compute new offset
 end
%
 data=[data;d];
end % loop over files
%
data=reshape(data,5,[]);
ncross=length(data)/5;
% determine number of replicas (optional):
if (guessnrep) ; nrep=max(data(1,:)); else ; nrep=numrep ; end
d=data; %want to modify (truncate) data below, but keep the original record (in d)
read=0; %do not reread this data if script rerun
else
 fprintf(' Using previouly read logs. Type "read=1" or "clear" to force re-read.\n');
end % read
%%%%%%%%%%%%%%%% to consider a subset of the trajectory %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time=d(5,:);
tmin=min(time); tmax=max(time);
%
%plot(time) ; % check to make sure time increasing correctly
%return
%select a window of time
tbeg=tmin+round ( 0 * (tmax-tmin) );
tend=round(tmax);
ind=intersect ( find(time>=tbeg), find(time<=tend) );
data=d(:,ind);
ncross=length(ind);
%%%%%%%%%%%%% 1: compute free energy %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  reconstruct his matrix
n=nrep;
his=zeros(n);
occupancy=zeros(n,1);
whereami =zeros(n,1);
when     =tbeg*ones(n,1);
%
for l=1:ncross
 id=data(1,l); 
 i=data(2,l); j=data(3,l); k=data(4,l); t=data(5,l)-1;
 his(i,j)=his(i,j)+1;
% determine occupancy
 whereami(id)=k;
 occupancy(i)=occupancy(i)+(t-when(id));
 when(id)=t;
end
% approximately, add the remaining steps
for id=1:n
 occupancy(whereami(id))=occupancy(whereami(id))+tmax-when(id);
end
%
%=============== write his matrix to file (debugging)==================
%fname=['his',num2str(ifile),'.dat'];
%fid=fopen(fname,'wt');
%for i=1:n
% for j=1:n
%  fprintf(fid,'%7d ', his(i,j));
% end
% fprintf(fid,'\n');
%end
%fclose(fid);
%============================================================
fprintf(hline);
fprintf(' Checking fluxes:\n');
%check fluxes
du=diag(his,1);  % upper diagonal (forward flux)
dl=diag(his,-1); % lower diagonal (backward flux)
%check for nonadjacent fluxes and warn
his2=diag(du,1)+diag(dl,-1) ; % matrix with adjacent fluxes only
his2=his-his2 ; % matrix with diagonal fluxes removed; only nonadjacent ones are left
[ii,jj]=find(his2);
nonadj=length(ii);
if (nonadj>0) ; 
 fprintf([' =========> ! WARNING : found ', num2str(nonadj),' fluxes between nonadjacent cells:\n']);
 for i=1:nonadj
  fdist=abs(ii(i)-jj(i));
  fprintf(['     ', num2str(ii(i)),' ===> ', num2str(jj(i)),' (',num2str(fdist),' cells)']);
  if ((maxdist>0) && (maxdist< fdist)) ; fprintf([' (Will be ignored because you set maxdist = ',num2str(maxdist),')\n']);
  else;fprintf('\n');end
 end
end
%
%check adjacent fluxes (his +1/-1 diagonals)
%reuse and overwrite diagonals computed above
du=min(1,du); du=1-du;
dl=min(1,dl); dl=1-dl;
nupper=length(find(du));
nlower=length(find(dl));
if (nupper>0) ; fprintf([' =========> ! WARNING : found ',num2str(nupper),' zero adjacent forward flux(es).\n']); end;
if (nlower>0) ; fprintf([' =========> ! WARNING : found ',num2str(nlower),' zero adjacent backward flux(es).\n']); end;
if (fixfluxval>0) && ((nlower+nupper) >0)
 fprintf([' Setting missing flux(es) to ',num2str(fixfluxval),'.\n']);
% fix diagonals if requested
 his2=diag(du,1)+diag(dl,-1) ;
% add matrix with diagonal fix
 his=his + fixfluxval * his2;
end
%
fprintf(hline);
fprintf(' Computing reaction free energy:\n');
%%%%%%%%%%%%% construct rate matrix
fprintf(' Constructing transition rate matrix...\n');
occupancy=occupancy/mean(occupancy) ; % for better numerical stability
r=zeros(n);
%
if (maxdist<1) ; maxdistfe = n ; else ; maxdistfe=maxdist ; end
for i=1:n
% compute indices
  ibeg = max(1,i-maxdistfe) ; iend=min(i+maxdistfe,n) ; inds=[ibeg:iend]; 
  r(i,inds)= his(inds,i)./occupancy(inds);
  r(i,i)= - sum(his(i,inds))/occupancy(i);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%% original code which includes all crossing events:
%for i=1:n
%  r(i,:)= his(:,i)./occupancy;
%  r(i,i)= - sum(his(i,:))/occupancy(i);
%end
%%%%%%%%%%%%%% set c(1)=1
c=zeros(n,1);
c(1)=1;
f=zeros(n,1);
f(:)=f(:)-r(:,1)*c(1);
fprintf(' Solving for free energy...\n');
c(2:end)=r(1:end-1,2:end)\f(1:end-1);
c=abs(c); % badly scaled matrices can erroneously produce small negative probabilities
%%%%%%%%%%%%%%%% solve for FE
beta=1.0/(temperature * kboltz);
%
f=-log(c)/beta;

alpha=[0:length(f)-1]; alpha=alpha/alpha(end);
if (dfilter>1) ; fs=smooth2(alpha,f,dfilter); else ; fs=f ; end
if (exist('figfe')); figure(figfe)
else ; figfe=figure;
end ; set(figfe,'position',[70,100,450,350]);
lw=1.5;
ms=12;
plot(alpha,fs,'r.-', 'linewidth',lw,'markersize',ms); % plot smoothed free energy
xlim([0 1]);
%plot(fs,'r.-', 'linewidth',lw,'markersize',ms);
%ylim([-15 22]);
box on;
title('\it Free energy of reaction','fontsize',13);
ylabel('\it F(\alpha) (kcal/mol)', 'fontsize',14);
xlabel('\it \alpha', 'fontsize',14);
%
set(gcf, 'paperpositionmode', 'auto');
%print(gcf, '-dpsc', 'vfe.eps');
%print(gcf, '-dtiff', '-r300', 'vfe.tif');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% compute MFPT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf(hline);
fprintf(' Constructing transition Q-matrix...\n');
% compute rate matrix
% calculate the number of hops between milestones (N), and the time spent at a milestone (R)

if (maxdistrate<1) ; maxdistht=n; 
else 
 maxdistht=maxdistfe ; 
 if (maxdist>0);
  if (nonadj>0); fprintf([' ==> Will ignore transitions between replicas separated by more than ', num2str(maxdistht),' cell(s) ...\n']); end
 end
end %set maxdist hitting time

whereami =zeros(n,1);
when     =tbeg*ones(n,1);
N=zeros(n,n,n);
R=zeros(n,n);
mstone=zeros(1,n)-1; % initialize to -1
for l=1:ncross
 id=data(1,l); i=data(2,l); j=data(3,l); k=data(4,l); t=data(5,l)-1; % note: k is either i or j 
 whereami(id)=k;
 m=mstone(id); %(i,m) is the milestone at which replica id is currently sitting
%
 if (abs(i-j)<=maxdistht)     % crossed to a different milestone that is within the maximum allowed distance 
%                             % note that the above d-restriction will be very problematic when crossings are actually allowed !)
  if m<0 % initial condition
   m=j;
  else
   if j~=m  %crossed to a different milestone
    N(i,m,j)=N(i,m,j)+1;  % number of cell crossings from [i,m] to [i,j] (obviously, through cell i); i - cell[replica]; m - old neighbor; j - new
   end
  end
 else   %not within allowed distance:
  j=m ; %( i-j > maxdistht ); ignore the cross attempt and keep the current milestone
 end
 if (m>=0)
  R(i,m)=R(i,m)+(t-when(id)); %update residence time at milestone m
 end
%
 when(id)=t; % update current time for replica id
 if (k==i);  % stayed in the same cell
  mstone(id)=j; % new milestone is i,j
 else ;      % crossed : swap i,j; this is the same milestone but indexed differently for the other replica
  mstone(id)=i;
 end %record the current milestone;
end
% approximately, add the remaining steps
for id=1:n
 R(whereami(id),mstone(id))=R(whereami(id),mstone(id))+tmax-when(id);
end
%
% `correct' N, R by equilibrium distribution (c)
% in the calculation below, the milestones are indexed : ij as the boundary between cells i and j ; obviously, ji is the same milestone,
% which is why the (equilibrium) residence times matrix Req(i,j) should be symmetric
% Note that this explanation does not apply to N(i,j,k), which counts transitions from milestone ij to ik; for example N(j,i,k) counts
% transitions from milestone ij (eqv. ji) to ik (not the same as jk)
%
Req=zeros(n,n);   % equilibrium residence time at milestone ij
Neq=zeros(n,n,n);
% compute transition rates
% decide whether to use smoothed FE
%if (mfptffe); pdf=exp( -beta * fs ); else ; pdf=c; end ; pdf=pdf/sum(pdf) ;
if (mfptffe); ferate=fs; else ; ferate=f; end
% weight transition rates by replica free energy
%pdf(:)=1;
for i=1:n
 for j=1:n
% could do vectorially, but clearer this way
%  Req(i,j)=pdf(i)*R(i,j) + pdf(j)*R(j,i) ;% Average time spent at milestone [i,j] ; note two equivalent indexings; Req is symmetric
%  Neq(i,j,:)=pdf(i)*N(i,j,:) ;
%
% equivalent way below because the rate matrix q is computed as Neq/Req so pdf(i) cancels
%
%  w=pdf(j)/pdf(i);
  w=exp(-beta*(ferate(j)-ferate(i)));
  Req(i,j)=R(i,j) + w * R(j,i) ;
  Neq(i,j,:)=N(i,j,:) ;
 end
end
%
%return
% change units of R to s (need simulation ts, e.g. 1 fs)
Req = Req*timestep*1e-15;
% remove 0s from Req (it's OK because in the quotient below, N will be 0 for these entries)
zind = find(Req>0);
iflag=min(min( Req(zind) )); % make negative entries of the same magnitude as the positive ones
Req(find(Req==0)) = -iflag;
% compute Q (instantaneous transition rate matrix)
q=zeros(n,n,n);
for k=1:n
 q(:,:,k)=squeeze(Neq(:,:,k))./Req ; % rate of hopping from i,j to i,k (normalized by the residence time)
end
q(find(q<0)) = 0 ; % just in case a jump occured in the last entry, in which case N>0 but corresponding R<0 (as initialized) 
%
%put in spurious transition rates into q-matrix to avoid singularity (if requested)
if ((nupper+nlower)>0)
 if (fixrate)
  fprintf([' ==> Estimating missing rates between adjacent milestones from nearby milestones.\n']);
  for i=2:n-1
% forward rate:
   if (q(i,i-1,i+1)==0) ;
    qrate=0; qnorm=0;
    if ( (i>2  ) && q(i-1,i-2,i)>0 ) ; qrate=qrate+q(i-1,i-2,i) ; qnorm=qnorm+1; end
    if ( (i<n-1) && q(i+1,i,i+2)>0 ) ; qrate=qrate+q(i+1,i,i+2) ; qnorm=qnorm+1; end
    if (qnorm==0)
     fprintf([' ====> WARNING ! : Could not estimate the rate ',num2str(i),'/',num2str(i-1),'==>', num2str(i),'/',num2str(i+1),'\n']);
     fprintf([' Setting to ',num2str(fixrate),'\n']);
     q(i,i-1,i+1)=fixrate
    else
     fprintf([' Interpolating the rate ',num2str(i),'/',num2str(i-1),'==>', num2str(i),'/',num2str(i+1),' to ',num2str(qrate/qnorm),'\n']);
     q(i,i-1,i+1)=qrate/qnorm;
    end
   end % q==0
% backward rate:
   if (q(i,i+1,i-1)==0) ;
    qrate=0; qnorm=0;
    if ( (i>2  ) && q(i-1,i,i-2)>0 ) ; qrate=qrate+q(i-1,i,i-2) ; qnorm=qnorm+1; end
    if ( (i<n-1) && q(i+1,i+2,i)>0 ) ; qrate=qrate+q(i+1,i+2,i) ; qnorm=qnorm+1; end
    if (qnorm==0)
     fprintf([' ====> WARNING ! : Could not estimate the rate ',num2str(i),'/',num2str(i-1),'==>', num2str(i),'/',num2str(i+1),'\n']);
     fprintf([' Setting to ',num2str(fixrate),'\n']);
     q(i,i+1,i-1)=fixrate
    else
     fprintf([' Interpolating the rate ',num2str(i),'/',num2str(i-1),'==>', num2str(i),'/',num2str(i+1),' to ',num2str(qrate/qnorm),'\n']);
     q(i,i+1,i-1)=qrate/qnorm;
    end
   end % q==0
  end % over all main milestones
 elseif ( fixrate>0)
  fprintf([' ==> Setting missing rates between adjacent milestones to ',num2str(fixrate),'.\n']);
  for i=2:n-1
   if (q(i,i-1,i+1)==0) ; q(i,i-1,i+1)=fixrate; end
   if (q(i,i+1,i-1)==0) ; q(i,i+1,i-1)=fixrate; end
  end
 end
end
%
% compute mean time of escape from ms i,j
tau=sum(q,3); % some of these may be zero, so exclude them from below:
ind=(find(tau>0));
tau(ind) = 1. / tau(ind);
p=zeros(n,n,n);
for k=1:n
 p(:,:,k)=q(:,:,k).*tau(:,:); % probability to hit ms i,k after hitting i,j 
end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% 3: compute mean first passage time to last milestone
% this requires reindexing q into a MxM 2D matrix
% create map:
% number of milestones visited:
%
%[mi mj]=find(Req>0); % milestones visited (based on nonzero residence times)
% find unique
%x=mi>mj; % true if i>j
% exclude milestone that corresponds to the target milestone, i.e. to which the mean first passage time corresponds ; also exclude [0 0]
%mstones2=setdiff([mi.*x mj.*x], [n n-1; 0 0], 'rows'); % double index
%mi=mstones2(:,1); mj=mstones2(:,2) ; % redefine double indices
%milenum=length(mi)  % number of milestones
% reverse lookup
%mstones1=zeros(n,n); for i=1:milenum; mstones1(mi(i), mj(i))=i; end; % single index (so that now :   mstones2(mstones1(mi(i),mj(i)),:)=[mi(i) mj(i)];  )
%
% find relevant entries in q
qq=zeros(size(q));
%for i=1:n; qq(i,:,:)=squeeze(q(i,:,:))+squeeze(q(i,:,:))'; end % symmetrize q so we count edge pairs for which at least one rate is positive  
%for i=1:n; qq(i,:,:)=min(squeeze(q(i,:,:)),squeeze(q(i,:,:))'); end % antisymmetrize q so we count edge pairs for which BOTH rates are positive  
% NOTE that antisymmetrization does not make a difference here for cases in which matrices are nearly singular (or, perhaps, at all)
ind=find(q>0)-1; % nonzero indices of q; subtract 1 to start at 0
% recreate index triplets
kk=floor(ind/n/n);
jj=floor((ind-kk*n*n)/n);
ii=ind-kk*n*n-jj*n;
% indices should start from 1:
ii=ii+1; jj=jj+1; kk=kk+1;
ind3=[ii jj kk]; % these are the triplets for which q is nonzero
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find number of relevant milestones 
mstones2=unique([ii jj ; ii kk; jj ii; kk ii],'rows');
mi=mstones2(:,1); mj=mstones2(:,2) ; % define indices
x=mi>mj; % true if i>j
% throw out redundancy; exclude [0 0]
mstones2=setdiff([mi.*x mj.*x], [0 0], 'rows'); % double index
mi=mstones2(:,1); mj=mstones2(:,2) ; % redefine double indices
milenum=length(mi);  % number of milestones
% reverse lookup: single index (so that:   mstones2(mstones1(mi(i),mj(i)),:)=mstones2(i,:) ;  )
mstones1=zeros(n,n); for i=1:milenum; mstones1(mi(i), mj(i))=i; mstones1(mj(i), mi(i))=i; end; % note that mstones1 is symmetric
% determine whether the main milestones are sampled
du=diag(mstones1,1);
[imissing]=find(du==0);
li=length(imissing);
if (li>0)
 fprintf([' =========> ! WARNING : the following ',num2str(li),' (main) milestones are not sampled :\n']);
 for i=1:li
  fprintf(['    ',num2str(i),'/',num2str(i+1),'\n']);
 end
end
%
% populate 2D rate Q-matrix: loop over the relevant entries in Q;
%
qq=zeros(milenum,milenum);
for l=1:length(ind)
 i=ii(l); j=jj(l); k=kk(l);
% if ( mstones1(i,j)>0 & mstones1(i,k)>0 ); %this will exclude the target milestone
  qq(mstones1(i,j),mstones1(i,k))=q(i,j,k);
% end
end
% populate diagonal to create q-matrix
qq=qq-diag(sum(qq,2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loop over target milestones:
lms=length(targetmstone);
targetm=zeros(0,2);
for i=1:lms
 mst=cell2mat(targetmstone(i));
 if (strcmp(mst,'all')); targetm=[[1:n-1]' [2:n]'] ; break
 elseif (strcmp(mst,'first')); targetm=[targetm ; [1 2]] ;
 elseif (strcmp(mst,'last' )); targetm=[targetm ; [n-1 n]] ;
 elseif (mst>0 && mst<n)     ; targetm=[targetm ; [mst mst+1]];
 end
end
% print requested target milestones
targetm=unique(targetm,'rows');  %remove duplicate entries
lms=size(targetm,1);
if (lms>0)
 fprintf(hline);
 fprintf(' Will compute mean first passage times to the following milestones:\n');
 for i=1:lms; fprintf(['   ',num2str(targetm(i,1)),'/',num2str(targetm(i,2)),'\n']); end
end
%
% now compute all MFPTs
fprintf(hline);
mfpts=zeros(lms,n-1);
for i=1:lms;
 ind=mstones1(targetm(i,1), targetm(i,2));
%
 fprintf(['  Computing MFPTs to milestone ',num2str(targetm(i,1)),'/',num2str(targetm(i,2)),'\n']);
% construct q-matrix without rows and columns corresponding to excluded milestone
 if (0&&(maxdist==1) && (maxdistrate)) % special purpose code in which the q matrix can be split in two:
% note that I did not find a significant difference in the rates, so I may disable this code in the future
% compute MFPT from milestones to the left and from those to the right separately
  qqq1=qq(1:ind-1,1:ind-1);
  qqq2=qq(ind+1:end,ind+1:end);
% invert each matrix
  t1=-qqq1\ones(ind-1,1);
  t2=-qqq2\ones(milenum-ind,1);
  t=abs([t1;0;t2]);
 else % general case
% exclude corresponding row + column
  qqq=[qq(1:ind-1,1:ind-1) qq(1:ind-1,ind+1:end) ; qq(ind+1:end,1:ind-1) qq(ind+1:end,ind+1:end) ];
% invert matrix
  t=-qqq\ones(milenum-1,1);
  t=abs(t); % sometimes ill-conditioned matrices produce negative values
% rehash t because we `deleted' a milestone above
  t=[t(1:ind-1);0;t(ind:end)];
 end
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save mean first passage times from main milestones
 tn=zeros(n-1,1);
 for j=1:n-1
% it is technically possible that not all of the main milestones are visited; therefore:
  m=mstones1(j, j+1);
  if(m>0); tn(j)=t(m); end
 end
 mfpts(i,:)=tn(:);
end
%
%save data
fprintf(hline);
fprintf([' Saving free energy profile to file ',fefile,'\n']);
eval(['save ',fefile,' -ascii -double fs']);
fprintf([' Saving mean first passage times to file ',mfptfile,'\n']);
eval(['save ',mfptfile,' -ascii -double mfpts']);
%
% plot MFPT to last requested milestone

if (exist('fight')); figure(fight);
else ; fight=figure;
end ; set(fight,'position',[523,100,450,350]);
%
semilogy(alpha(1:end-1), tn,'k.-', 'linewidth', lw, 'markersize', ms);
box on;
title(['\it Mean first passage time to milestone ',num2str(targetm(end,1)),'/',num2str(targetm(end,2))] ,'fontsize',13);
ylabel('\it MFPT(\alpha) (s)', 'fontsize',14);
xlabel('\it \alpha', 'fontsize',14);
%
fprintf(hline);
%
print(gcf, '-dpsc', 'mfpt.eps');
print(gcf, '-dtiff', '-r300', 'mfpt.tif');
