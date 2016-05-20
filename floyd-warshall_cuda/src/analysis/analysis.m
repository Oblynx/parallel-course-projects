clear all;
close all;
load('logs.mat');

%% 1: Psize
p1= logs1(logs1(:,2)==0.33,:);
p2= logs1(logs1(:,2)==0.45,:);
p3= logs1(logs1(:,2)==0.66,:);
figure(1);
plot(log2(p1(:,1)),p1(:,3:end)); hold on;
plot(log2(p2(:,1)),p2(:,3:end)); hold on;
plot(log2(p3(:,1)),p3(:,3:end)); hold on;
hold off;
legend('cpu','simple gpu','block gpu','multi_x_y','multi_y','Location','northwest');
ylabel('time(s)');
xlabel('size(logN)');
grid minor;
title('Run1: Execution time vs problem size');
axis([7,12,0,90]);

%% 1: p
p1= logs1(logs1(:,1)==128,:);
p2= logs1(logs1(:,1)==256,:);
p3= logs1(logs1(:,1)==512,:);
p4= logs1(logs1(:,1)==1024,:);
p5= logs1(logs1(:,1)==2048,:);
p6= logs1(logs1(:,1)==4096,:);
%mean([std(p1)./mean(p1);std(p2)./mean(p2);std(p3)./mean(p3);std(p4)./mean(p4);std(p5)./mean(p5)]);
stddev1= mean([std(p1);std(p2);std(p3);std(p4);std(p5);std(p6)]);
stddev1= stddev1(:,3:end)

%% 2: Psize
p0= logs2(logs2(:,2)==0.01,:);
p1= logs2(logs2(:,2)==0.33,:);
p2= logs2(logs2(:,2)==0.45,:);
p3= logs2(logs2(:,2)==0.66,:);
p4= logs2(logs2(:,2)==0.99,:);
figure(2);
%{
plot(log2(p0(1:end-1,1)),p0(1:end-1,3:end)); hold on;
plot(log2(p1(1:end-1,1)),p1(1:end-1,3:end)); hold on;
plot(log2(p2(1:end-1,1)),p2(1:end-1,3:end)); hold on;
plot(log2(p3(1:end-1,1)),p3(1:end-1,3:end)); hold on;
plot(log2(p4(1:end-1,1)),p4(1:end-1,3:end)); hold on;
%}
plot(log2(p0(:,1)),p0(:,3:end)); hold on;
plot(log2(p1(:,1)),p1(:,3:end)); hold on;
plot(log2(p2(:,1)),p2(:,3:end)); hold on;
plot(log2(p3(:,1)),p3(:,3:end)); hold on;
plot(log2(p4(:,1)),p4(:,3:end)); hold on;
hold off;
legend('cpu','simple gpu','block gpu','multi_x_y','multi_y','Location','northwest');
ylabel('time(s)');
xlabel('size(logN)');
grid minor;
axis([7,13,0,350]);
title('Run2: Execution time vs problem size');

%% 2: p
p1= logs2(logs2(:,1)==128,:);
p2= logs2(logs2(:,1)==256,:);
p3= logs2(logs2(:,1)==512,:);
p4= logs2(logs2(:,1)==1024,:);
p5= logs2(logs2(:,1)==2048,:);
p6= logs2(logs2(:,1)==4096,:);
p7= logs2(logs2(:,1)==8192,:);
%mean([std(p1)./mean(p1);std(p2)./mean(p2);std(p3)./mean(p3);std(p4)./mean(p4);std(p5)./mean(p5)]);
stddev2= mean([std(p1);std(p2);std(p3);std(p4);std(p5);std(p6);std(p7)]);
stddev2= stddev2(:,3:end)

clear('p*');
