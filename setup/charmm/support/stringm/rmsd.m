%
close all;

if (~exist('read'))
 read=1;
end

read=1

if (read==1)
%%%%%%%%%% can load multiple files %%%%%%%%%%%%
fnames={'./sm0k/rmsd.dat'};
fnames={'./smcv/rmsd0.dat'}; % from smcv
%
clear rmsa;
for i=1:length(fnames)
 fname=char(fnames(i));
 if (i==1)
  rmsa=load(fname);
 else
  rmsa=[rmsa; load(fname)];
 end 
end
read=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[niter, nrep]=size(rmsa);

figure; hold on;box on;

shift=0.2; % how curves that correspond to successive replicas will be shifted
subplot(1,2,1);hold on

for i=2:nrep
 plot(rmsa(:,i)+i*shift,'-');
end
box on;

title('RMSDs from initial string');
ylabel('RMSD(Ang)');
xlabel('iteration');
%%%%%%%%%%%%%%%%%%%%%%% average
subplot(1,2,2);hold on
plot( sqrt(mean(rmsa(:,2:end).^2,2))/(nrep-1) );
box on;
ylabel('Replica index');
xlabel('iteration');
title('Combined RMSD from initial string');
set(gcf, 'paperpositionmode', 'auto');

print(gcf, '-depsc2', 'rmsd.eps');
print(gcf, '-dtiff','-r300', 'rmsd.tif');
