% string length
%
close all;

if (~exist('read'))
 read=1;
end

if (read==1)
%%%%%%%%%% load multiple files %%%%%%%%%%%%
fnames={'./sm0k/arc.dat'}; % from sm0k
%
clear s;
for i=1:length(fnames)
 fname=char(fnames(i));
 if (i==1)
  s=load(fname);
 else
  s=[s; load(fname)];
 end 
end
read=1;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot(s(:,1), sum(s(:,2:end),2));

box on;
ylabel('String length(Ang)');
xlabel('iteration');

set(gcf, 'paperpositionmode', 'auto');

print(gcf, '-depsc2', 'arclen.eps');
print(gcf, '-djpeg100', 'arclen.jpg');
