%% Grid analysis
load('grid_results')
p1= p1_laptop_norm2grid;
allp= p1;
allp(:,:,2)= p2; allp(:,:,3)= p4; allp(:,:,4)= p8;

figure(1);
plot([21:25],[p1(:,5),p2(:,5),p4(:,5),p8(:,5)]);
legend('P=1','P=2','P=4','P=8');
title('Search time per problem size for var. numbers of processes');
xlabel('N'); ylabel('time (s)');
grid minor;

figure(2);
plot([21:25],[p1(:,3),p2(:,3),p4(:,3),p8(:,3)]);
legend('P=1','P=2','P=4','P=8');
title('Points communication time per problem size for var. numbers of processes');
xlabel('N'); ylabel('time (s)');
grid minor;

figure(3);
plot([1,2,4,8], [reshape(allp(1,5,:),[],1), reshape(allp(2,5,:),[],1), ...
								 reshape(allp(3,5,:),[],1), reshape(allp(4,5,:),[],1), ...
								 reshape(allp(5,5,:),[],1)]);
legend('N=21','N=22','N=23','N=24','N=25');
title('Search time per number of processes for var. problem sizes');
xlabel('P'); ylabel('time (s)');
grid minor;

figure(4);
plot([1,2,4,8], [reshape(allp(1,3,:),[],1), reshape(allp(2,3,:),[],1), ...
								 reshape(allp(3,3,:),[],1), reshape(allp(4,3,:),[],1), ...
								 reshape(allp(5,3,:),[],1)]);
legend('N=21','N=22','N=23','N=24','N=25');
title('Points communication time per number of processes for var. problem sizes');
xlabel('P'); ylabel('time (s)');
grid minor;


