%% load imported data
clear all;
load importedData.mat;
%% erase wrong data
dupl=findDupl(omp)
if ~isempty(dupl)
	omp=[omp(1:dupl,:);omp(dupl+2:end,:)];
	pthread=[pthread(1:dupl,:);pthread(dupl+2:end,:)];
end
%dupl=findDupl(pthread)
%dupl=findDupl(samplebitonic)
omp=[omp(1:32,:);omp(36:end-1,:)];
pthread=[pthread(1:32,:);pthread(36:end-1,:)];
qsort=[qsort(1:32,:);qsort(36:end-1,:)];
samplebitonic= samplebitonic(1:end-3,:);
%% prepare plot data
%sizes
sz= omp(1:10,1)*ones([1,8]);
%actual data
ompCtr= reshape(omp(:,3),10,[]);
ompS= reshape(omp(:,4),10,[]);
pthrCtr= reshape(pthread(:,3),10,[]);
pthrS= reshape(pthread(:,4),10,[]);
qsort= reshape(qsort(:,4),10,[]);
sampleRecbit= reshape(samplebitonic(:,3),10,[]);

clear('omp','pthread','samplebitonic','dupl');
%% plot
figure(1);
semilogy(sz,pthrS);
grid on;
title('pthread (std::thread) execution time -- log-log');
legend('2^1 thread', '2^2 threads', '2^3 threads', '2^4 threads', '2^5 threads',...
			 '2^6 threads', '2^7 threads', '2^8 threads','Location','NorthWest');
xlabel('log problem size');
ylabel('log sort time (ms)');

figure(2);
semilogy(sz,ompS);
grid on;
title('OpenMP execution time -- log-log');
legend('2^1 thread', '2^2 threads', '2^3 threads', '2^4 threads', '2^5 threads',...
			 '2^6 threads', '2^7 threads', '2^8 threads','Location','NorthWest');
xlabel('log problem size');
ylabel('log sort time (ms)');

figure(3);
semilogy(sz(:,1),mean(qsort,2));
grid on;
title('qsort execution time -- log-log');
xlabel('log problem size');
ylabel('log sort time (ms)');

figure(4);
semilogy(sz(:,1),mean(sampleRecbit,2));
grid on;
title('serial bitonic execution time -- log-log');
xlabel('log problem size');
ylabel('log sort time (ms)');

%% Compare all versions at optimal thread numbers
figure(5);
semilogy(sz(:,1:4),[ompS(:,3),pthrS(:,3),qsort(:,3),sampleRecbit(:,3)]);
grid on;
title('Comparing the 4 different implementations');
legend('OpenMP@2^3','pthread@2^3','qsort','serial bitonic','Location','NorthWest');
xlabel('log problem size');
ylabel('log sort time (ms)');
