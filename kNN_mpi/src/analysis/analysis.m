%% kNN result analysis
close all;
clear all;
load parallel1times
tmp= [rank,k,N,nmk,ptTrans,totComm,search];
tmp= sortrows(tmp);
%{
for rank_i= 0 : length(unique(tmp(:,1)))-1
	split(rank_i+1)= max(find(tmp(:,1)==rank_i));
end
%}
split0= tmp(1:max(find(tmp(:,1)==0)),:);
split1= tmp(max(find(tmp(:,1)==0))+1:max(find(tmp(:,1)==1)),:);
split2= tmp(max(find(tmp(:,1)==1))+1:max(find(tmp(:,1)==2)),:);
split3= tmp(max(find(tmp(:,1)==2))+1:max(find(tmp(:,1)==3)),:);
split(:,:,1)= split0;
split(:,:,2)= split1;
split(:,:,3)= split2;
split(:,:,4)= split3;
parallel= median(split,3);
parallel= parallel(:,3:end);
%insert 25/12 row
parallel= [parallel(1:20,:);parallel(21,:);parallel(21:end,:)];
parallel(21,2)= 12; parallel(21,5)= 7200;
clear('split*');

load serial1times
serial= [rank,k,N,nmk,ptTrans,totComm,search];
serial= serial(:,3:end);
%insert 24/12 row
serial= [serial;serial(19,:)];
serial(20,2)= 12; serial(20,5)= 2000;

%% t-mesh
figure(1);
tmp1= parallel(parallel(:,1)==21,3);
tmp2= parallel(parallel(:,1)==22,3);
tmp3= parallel(parallel(:,1)==23,3);
tmp4= parallel(parallel(:,1)==24,3);
tmp5= parallel(parallel(:,1)==25,3);
plot([12:16],[tmp1,tmp2,tmp3,tmp4,tmp5]);
legend('N=21','N=22','N=23','N=24', 'N=25');
title('Parallel (P=4) points communication time (shared memory)');
xlabel('mesh'); ylabel('time (s)');
grid minor;

figure(2);
tmp1= parallel(parallel(:,1)==21,5);
tmp2= parallel(parallel(:,1)==22,5);
tmp3= parallel(parallel(:,1)==23,5);
tmp4= parallel(parallel(:,1)==24,5);
tmp5= parallel(parallel(:,1)==25,5);
plot([12:16],[tmp1,tmp2,tmp3,tmp4,tmp5]);
legend('N=21','N=22','N=23','N=24', 'N=25');
title('Parallel (P=4) search time (shared memory)');
xlabel('mesh'); ylabel('time (s)');
grid minor;

figure(3);
tmp1= serial(serial(:,1)==21,5);
tmp2= serial(serial(:,1)==22,5);
tmp3= serial(serial(:,1)==23,5);
tmp4= serial(serial(:,1)==24,5);
plot([16:-1:12],[tmp1,tmp2,tmp3,tmp4]);
legend('N=21','N=22','N=23','N=24');
title('Serial search time');
xlabel('mesh'); ylabel('time (s)');
grid minor;

%% t-N
figure(4);
tmp1= parallel(parallel(:,2)==12,3);
tmp2= parallel(parallel(:,2)==13,3);
tmp3= parallel(parallel(:,2)==14,3);
tmp4= parallel(parallel(:,2)==15,3);
tmp5= parallel(parallel(:,2)==16,3);
plot([21:25],[tmp1,tmp2,tmp3,tmp4,tmp5]);
legend('mesh=12','mesh=13','mesh=14','mesh=15', 'mesh=16');
title('Parallel (P=4) points communication time (shared memory)');
xlabel('N'); ylabel('time (s)');
grid minor;

figure(5);
tmp1= parallel(parallel(:,2)==12,5);
tmp2= parallel(parallel(:,2)==13,5);
tmp3= parallel(parallel(:,2)==14,5);
tmp4= parallel(parallel(:,2)==15,5);
tmp5= parallel(parallel(:,2)==16,5);
plot([21:25],[tmp1,tmp2,tmp3,tmp4,tmp5]);
legend('mesh=12','mesh=13','mesh=14','mesh=15', 'mesh=16');
title('Parallel (P=4) search time (shared memory)');
xlabel('N'); ylabel('time (s)');
grid minor;

figure(6);
tmp1= serial(serial(:,2)==12,5);
tmp2= serial(serial(:,2)==13,5);
tmp3= serial(serial(:,2)==14,5);
tmp4= serial(serial(:,2)==15,5);
tmp5= serial(serial(:,2)==16,5);
plot([21:24],[tmp1,tmp2,tmp3,tmp4,tmp5]);
legend('N=21','N=22','N=23','N=24');
legend('mesh=12','mesh=13','mesh=14','mesh=15', 'mesh=16');
title('Serial search time');
xlabel('N'); ylabel('time (s)');
grid minor;
