% plot energies along the string

styles={'r','g','b','m','c','k','y'};

ind=[2, 10:10:60, 126];

figure; hold on; box on; grid on;

leg={};
for i=1:length(ind)
 e=load(['./string',num2str(ind(i)),'.dat']);
 rep=e(:,1); % replica number
 tote=e(:,2); % total energy; see file for other columns
 plot(rep,tote, char(styles(mod(i-1,length(styles))+1)),'linewidth',1.5);
 leg=[leg, {['iteration ', num2str(ind(i))] }];
end 
 
legend(leg,-1);
 
xlabel('\it Replica', 'fontsize',14);
ylabel('\it Potential energy (kcal/mol)','fontsize',14);

set(gcf,'Paperpositionmode','auto');
print(gcf, '-depsc2','zts.eps');
print(gcf, '-dtiff','-r300','zts.tif');
