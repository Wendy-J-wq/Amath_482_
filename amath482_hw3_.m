clear all; close all; clc;

%% Test1: ideal case
% input
load('cam1_1.mat'); load('cam2_1.mat');load('cam3_1.mat');
numFrames1a = size(vidFrames1_1,4);
numFrames2a = size(vidFrames2_1,4);
numFrames3a = size(vidFrames3_1,4);
[height1 width1 rgb1 num_frames1] = size(vidFrames1_1);
[height2 width2 rgb2 num_frames2] = size(vidFrames2_1);
[height3 width3 rgb3 num_frames3] = size(vidFrames3_1);
%%
% filter
width = 50;
filter = zeros(480,640);
filter(300-2.6*width:1:300+2.6*width, 350-width:1:350+width) = 1;

data1 = []; % inital blank matrix setting
for j = 1:numFrames1a % run each frame
    X = vidFrames1_1(:,:,:,j);
    Xabw = rgb2gray(X); % convert colorful to gray
    Xabw2 = double(Xabw); % convert unit to double
    %imshow(X);drawnow
    Xf = Xabw2.*filter; % use filter to data
    thresh = Xf > 250; % select the bright spot
    indeces = find(thresh);
    [Y,X] = ind2sub(size(thresh),indeces); %locate
    data1 = [data1; mean(X), mean(Y)]; % get one set coordinates
end

%%

width = 50;
filter = zeros(480,640);
filter(250-3*width:1:250+3*width, 290-1.3*width:1:290+1.3*width) = 1;
data2 = [];
for j = 1:numFrames2a
    X = vidFrames2_1(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf> 250;
    indeces = find(thresh);
    [Y,X] = ind2sub(size(thresh),indeces);
    data2 = [data2; mean(X), mean(Y)];
end
%%

width = 50;
filter = zeros(480,640);
filter(250-width:1:250+2*width, 360-2.5*width:1:360+2.5*width) = 1;
data3 = [];
for j = 1:numFrames3a
    X = vidFrames3_1(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    thresh = Xf > 247;
    indeces = find(thresh);
    [Y,X] = ind2sub(size(thresh),indeces);
    data3 = [data3; mean(X), mean(Y)];
end
%% let all video data corresponds
[M,I] = min(data1(1:20,2));
data1 = data1(I:end,:);
%
[M,I] = min(data2(1:20,2));
data2 = data2(I:end,:);
%
[M,I] = min(data3(1:20,2));
data3 = data3(I:end,:);
%% convert to the same length
data3 = data3(1:length(data1),:);
data2 = data2(1:length(data1),:);

%% combine the data to one matrix and use svd
all = [data1 data2 data3]';
[m,n] = size(all);
mn = mean(all,2); % find mean value
all = all - repmat(mn,1,n); % minus mean value
[U,S,V] = svd(all/sqrt(n-1)); % svd
lambda = diag(S).^2; % get lambda
Y = U'*all; % the variance
sig = diag(S);

%%
figure(1)
plot(1:6,lambda/sum(lambda),'ko','linewidth',2);
title('Ideal case: Energy of each diagonal variance');
xlabel('Diagonal Variances'); ylabel('Energy Captured');


%%
figure(2)
subplot(2,1,1)
plot(1:218,all(2,:),1:218,all(1,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Ideal case: Displacement across z axis and xy plane (camera1)');
legend('z','xy');

subplot(2,1,2)
plot(1:218,Y(1,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Ideal case: Displacement across principal component directions');
legend('Principle component');



%%
clear all; close all; clc;
%% Test2: noisy case
load('cam1_2.mat'); load('cam2_2.mat');load('cam3_2.mat');
numFrames1b = size(vidFrames1_2,4);
numFrames2b = size(vidFrames2_2,4);
numFrames3b = size(vidFrames3_2,4);

width = 50;
filter = zeros(480,640);
filter(300 - 2.6*width:1:300+2.6*width, 350-width:1:350+2*width) = 1;

data1 = [];
for j = 1:numFrames1b
    X = vidFrames1_2(:,:,:,j);
    %imshow(X);drawnow
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind = find(Xf > 250);
    [Y,X] = ind2sub(size(Xf > 250),ind);
    data1 = [data1; mean(X),mean(Y)];
end
%%
width = 50;
filter = zeros(480,640);
filter(250-3*width:1:250+3*width, 290-1.3*width:1:290+1.3*width) = 1;
data2 = [];
for j = 1:numFrames2b
    X = vidFrames2_2(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind2 = find(Xf > 237);
    [Y,X] = ind2sub(size(Xf > 237),ind2);
    data2 = [data2; mean(X), mean(Y)];
end

%%
width = 50;
filter = zeros(480,640);
filter(250-width:1:250+2*width, 360-2.5*width:1:360+2.5*width) = 1;
data3 = [];
for j = 1:numFrames3b
    X = vidFrames3_2(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind3 = find(Xf > 245);
    [Y,X] = ind2sub(size(Xf > 245),ind3);
    data3 = [data3; mean(X), mean(Y)];
end
%%
[M,I] = min(data1(1:20,2));
data1 = data1(I:end,:);

[M,I] = min(data2(1:20,2));
data2 = data2(I:end,:);


[M,I] = min(data3(1:20,2));
data3 = data3(I:end,:);

%%
 data2 = data2(1:length(data1),:);

 data3 = data3(1:length(data1),:);
% 
alldata = [data1 data2 data3]';
%%
[m2,n2] = size(alldata);
mn2 = mean(alldata,2);
alldata = alldata - repmat(mn2,1,n2);
[U,S,V] = svd(alldata/sqrt(n2-1));
lambda2 = diag(S).^2;
Y = U'*alldata;

%%
figure(3)
plot(1:6,lambda2/sum(lambda2),'ko','linewidth',2);
title('Noisy case: Energy of each diagonal variance');
xlabel('Diagonal Variances'); ylabel('Energy Captured');
%%
figure(4)
subplot(2,1,1)
plot(1:287, alldata(2,:),1:287,alldata(1,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Noisy case : Displacement across z axis and xy plane(camera 1)');
legend('z','xy');

% subplot(3,1,2)
% plot(1:314, alldata(4,:),1:314,alldata(3,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Noisy case : camera 2');
% legend('z','xy');
% 
% subplot(3,1,3)
% plot(1:314, alldata(6,:),1:314,alldata(5,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Noisy case : camera 3');
% legend('z','xy');

subplot(2,1,2)
plot(1:287,Y(1,:),1:287,Y(2,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Noisy case: Displacement across principal component directions');
legend('Principle component1','Principle component2');


%%
clear all; close all; clc;
%% Test3: Horizontal Displacement
load('cam1_3.mat'); load('cam2_3.mat');load('cam3_3.mat');
numFrames1b = size(vidFrames1_3,4);
numFrames2b = size(vidFrames2_3,4);
numFrames3b = size(vidFrames3_3,4);

width = 50;
filter = zeros(480,640);
filter(300 - 2.6*width:1:300+2.6*width, 350-width:1:350+2*width) = 1;

data1 = [];
for j = 1:numFrames1b
    X = vidFrames1_3(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind = find(Xf > 250);
    [Y,X] = ind2sub(size(Xf > 250),ind);
    data1 = [data1; mean(X),mean(Y)];
end
%%
width = 50;
filter = zeros(480,640);
filter(250-3*width:1:250+3*width, 290-1.3*width:1:290+1.3*width) = 1;
data2 = [];
for j = 1:numFrames2b
    X = vidFrames2_3(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind2 = find(Xf > 250);
    [Y,X] = ind2sub(size(Xf > 250),ind2);
    data2 = [data2; mean(X), mean(Y)];
end

%%
width = 50;
filter = zeros(480,640);
filter(250-width:1:250+2*width, 360-2.5*width:1:360+2.5*width) = 1;
data3 = [];
for j = 1:numFrames3b
    X = vidFrames3_3(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind3 = find(Xf > 247);
    [Y,X] = ind2sub(size(Xf > 247),ind3);
    data3 = [data3; mean(X), mean(Y)];
end

%%
[M,I] = min(data1(1:20,2));
data1 = data1(I:end,:);

%%
[M,I] = min(data2(1:20,2));
data2 = data2(I:end,:);

%% 
[M,I] = min(data3(1:20,2));
data3 = data3(I:end,:);
%%
data1 = data1(1:length(data3),:);
data2 = data2(1:length(data3),:);
alldata = [data1 data2 data3]';
%%
[m3,n3] = size(alldata);
mn3 = mean(alldata,2);
alldata = alldata - repmat(mn3,1,n3);

[U,S,V] = svd(alldata/sqrt(n3-1));
lambda3 = diag(S).^2;
Y = U'*alldata;


%%
figure(5)
plot(1:6,lambda3/sum(lambda3),'ko','linewidth',2);
title('Horizontal displacement: Energy of each diagonal variance');
xlabel('Diagonal Variances'); ylabel('Energy Captured');

%%
figure(5)
subplot(2,1,1)
plot(1:237, alldata(2,:),1:237,alldata(1,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Horizontal Displacement : Displacement across z axis and xy plane(camera 1)');
legend('z','xy');

% subplot(3,1,2)
% plot(1:237, alldata(4,:),1:237,alldata(3,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Horizontal Displacement : camera 2');
% legend('z','xy');
% 
% subplot(3,1,3)
% plot(1:237, alldata(6,:),1:237,alldata(5,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Horizontal displacement : camera 3');
% legend('z','xy');
subplot(2,1,2)
plot(1:237,Y(1,:),1:237,Y(2,:),1:237,Y(3,:),1:237,Y(4,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Horizontal Displacement: Displacement across principal component directions');
legend('Principle component1','Principle component2','Principle component3','Principle component4');


%%
clear all; close all; clc;
%% Test4: Horizontal Displacement and rotation
load('cam1_4.mat'); load('cam2_4.mat');load('cam3_4.mat');
numFrames1b = size(vidFrames1_4,4);
numFrames2b = size(vidFrames2_4,4);
numFrames3b = size(vidFrames3_4,4);

width = 50;
filter = zeros(480,640);
filter(300 - 1.5*width:1:300+3*width, 350-1.5*width:1:350+2*width) = 1;

data1 = [];
for j = 1:numFrames1b
    X = vidFrames1_4(:,:,:,j);%imshow(X);drawnow
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind = find(Xf > 243);
    [Y,X] = ind2sub(size(Xf > 243),ind);
    data1 = [data1; mean(X),mean(Y)];
end
%%
width = 50;
filter = zeros(480,640);
filter(250-3*width:1:250+3.5*width, 290-2.5*width:1:290+2.7*width) = 1;
data2 = [];
for j = 1:numFrames2b
    X = vidFrames2_4(:,:,:,j);%imshow(X);drawnow
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind2 = find(Xf > 245);
    [Y,X] = ind2sub(size(Xf > 245),ind2);
    data2 = [data2; mean(X), mean(Y)];
end

%%
width = 50;
filter = zeros(480,640);
filter(250-1.8*width:1:250+2.5*width, 360-2.5*width:1:360+2.8*width) = 1;
data3 = [];
for j = 1:numFrames3b
    X = vidFrames3_4(:,:,:,j);
    Xabw = rgb2gray(X);
    Xabw2 = double(Xabw);
    Xf = Xabw2.*filter;
    ind3 = find(Xf > 234);
    [Y,X] = ind2sub(size(Xf > 234),ind3);
    data3 = [data3; mean(X), mean(Y)];
end

%%
[M,I] = min(data1(1:20,2));
data1 = data1(I:end,:);

[M,I] = min(data2(1:20,2));
data2 = data2(I:end,:);

[M,I] = min(data3(1:20,2));
data3 = data3(I:end,:);

%%

data1 = data1(1:length(data3),:);
data2 = data2(1:length(data3),:);


alldata = [data1 data2 data3]';
%%
[m4,n4] = size(alldata);
mn4 = mean(alldata,2);
alldata = alldata - repmat(mn4,1,n4);
[U,S,V] = svd(alldata/sqrt(n4-1));
lambda4 = diag(S).^2;
Y = U'*alldata;


%%
figure(7)
plot(1:6,lambda4/sum(lambda4),'ko','linewidth',2);
title('Horizontal displacement and rotation: Energy of each diagonal variance');
xlabel('Diagonal Variances'); ylabel('Energy Captured');

%%
figure(8)
subplot(2,1,1)
plot(1:375, alldata(2,:),1:375,alldata(1,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Horizontal Displacement and rotation: Displacement across z axis and xy plane(camera 1)');
legend('z','xy');

% subplot(3,1,2)
% plot(1:232, alldata(4,:),1:232,alldata(3,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Horizontal Displacement : camera 2');
% legend('z','xy');
% 
% subplot(3,1,3)
% plot(1:232, alldata(6,:),1:232,alldata(5,:),'linewidth',2);
% xlabel('Time (frames)'); ylabel('Displacement (pixels)');
% title('Horizontal displacement : camera 3');
% legend('z','xy');
subplot(2,1,2)
plot(1:375,Y(1,:),1:375,Y(2,:),1:375,Y(3,:),'linewidth',2);
xlabel('Time (frames)'); ylabel('Displacement (pixels)');
title('Horizontal Displacement and Rotation: Displacement across principal component directions');
legend('Principle component1','Principle component2','Principle component3');
