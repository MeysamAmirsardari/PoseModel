%
% Meysam Amirsardari
% PoseModel
% Winter 2022
% IPM-SCS
%==========================================================================
clc; clear; close all;

T = readtable('C:\Users\EMINENT\Desktop\filtered1.csv','NumHeaderLines',1);
%%
clear; clc;
predict = csvread('C:\Users\EMINENT\Desktop\filtered1.csv',3,0);


%%
clc;
AB = readmatrix('C:\Users\EMINENT\Desktop\filtered1.csv');

%%
clc; close all;

test = aa(:,2);
test(10000:18000) = NaN(1,8001);

subplot(2,1,1)
hold on
%plot(aa(:,2))
plot(test)
hold off
grid on; grid minor;

subplot(2,1,2)
plot(aa(:,4))
grid on; grid minor;

%%
clc;

len = length(predict(:,2));
CL_thresh = 0.18;
dist_thresh = 7;
std_thresh = 3;
step1 = NaN(len,1);
bad_med = zeros(len,1);
bad_CL = zeros(len,1);


medFiltered = medfilt1(predict(:,2),4,'omitnan','truncate');

for idx=1:len
    if abs(medFiltered(idx)-predict(idx,2)) > dist_thresh
        bad_med(idx) = 1;
    end
    
    if predict(idx,4) < CL_thresh 
        bad_CL(idx) = 1;
    end
end

stdSerie = movstd(predict(:,2),5,'omitnan')/5;
len_std = length(stdSerie);
std_t = std(predict(:,2))/len;
std_win_len = 3;

bad_std = zeros(len_std,1);

for idx=std_win_len+1:len_std-std_win_len
    if abs(stdSerie(idx)-std_t) > std_thresh
        bad_std(idx-std_win_len:idx+std_win_len) = ones(1,2*std_win_len+1);
    end
end

%%

e1 = sum(bad_med)
e2 = sum(bad_CL)
e3 = sum(bad_std)

s = bad_med+bad_CL+bad_std;

e_t = sum(s>0)

eee = sum(bad_med.*bad_CL.*bad_std)

%%
clc;

























