clear all; close all; clc

%%
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels-idx1-ubyte');
[images2, labels2] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

%%
% images1 = double(reshape(images, size(images,1)*size(images,2),[]));
% [U,S,V] = svd(images1,'econ');
% 
% plot(diag(S),'ko','linewidth',2);
% set(gca,'Fontsize',16,'xlim',[0,200]);
[m2,~,image_num2] = size(images2);
reshape_img2 = zeros(m2^2, image_num2);
for i = 1: image_num2
    reshape_img2(:,i) = im2double(reshape(images2(:,:,i),m2^2,1));
end

%%
[m,~,image_num] = size(images);
reshape_img = zeros(m^2, image_num);
for i = 1: image_num
    reshape_img(:,i) = im2double(reshape(images(:,:,i),m^2,1));
end

%singular value spectrum
[U,S,V] = svd(reshape_img,'econ');
% figure(1)
% plot(diag(S),'ko','linewidth',2);
% set(gca,'Fontsize',16,'xlim',[0,200]);

%% 
digit_0 = images(labels == 0);
digit_1 = images(labels == 1);
digit_2 = images(labels == 2);
digit_3 = images(labels == 3);
digit_4 = images(labels == 4);
digit_5 = images(labels == 5);
digit_6 = images(labels == 6);
digit_7 = images(labels == 7);
digit_8 = images(labels == 8);
digit_9 = images(labels == 9);

%%
proj = U(:,[1,2,3])'* reshape_img;

proj_0 = proj(:,labels == 0);
proj_1 = proj(:,labels == 1);
proj_2 = proj(:,labels == 2);
proj_3 = proj(:,labels == 3);
proj_4 = proj(:,labels == 4);
proj_5 = proj(:,labels == 5);
proj_6 = proj(:,labels == 6);
proj_7 = proj(:,labels == 7);
proj_8 = proj(:,labels == 8);
proj_9 = proj(:,labels == 9);

figure(2)
plot3(proj_0(1,:),proj_0(2,:),proj_0(3,:),"o");hold on
plot3(proj_1(1,:),proj_1(2,:),proj_1(3,:),"o");hold on
plot3(proj_2(1,:),proj_2(2,:),proj_2(3,:),"o");hold on
plot3(proj_3(1,:),proj_3(2,:),proj_3(3,:),"o");hold on
plot3(proj_4(1,:),proj_4(2,:),proj_4(3,:),"o");hold on
plot3(proj_5(1,:),proj_5(2,:),proj_5(3,:),"o");hold on
plot3(proj_6(1,:),proj_6(2,:),proj_6(3,:),"o");hold on
plot3(proj_7(1,:),proj_7(2,:),proj_7(3,:),"o");hold on
plot3(proj_8(1,:),proj_8(2,:),proj_8(3,:),"o");hold on
plot3(proj_9(1,:),proj_9(2,:),proj_9(3,:),"o");


title('3D project V-modes of digits')
legend('0','1','2','3','4','5','6','7','8','9');










%%
% digit_3 = images(labels == 3);
% digit_4 = images(labels == 4); 
% 
% feature = 20;
% n_3 = 6131;
% n_4 = 5842;
% [U,S,V] = svd([digit_0 digit_1],'econ');
% P = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
% dogs = P(1:feature,1:n_3);
% cats = P(1:feature,n_4+1:n_4+n_3);
%%
digit_0 = reshape_img(labels == 0);
digit_7 = reshape_img(labels == 7); 
%%
[u1, s1, v1] = svd([digit_0 digit_7],'econ');
%%
feature = 20;
zeros = digit_0(1:feature,labels == 0);

P = S*V'; % projection onto principal components: X = USV' --> U'X = SV'

ones = P(1:feature,labels == 8);
twos = P(1:feature,labels == 9);

%% Calculate scatter matrices
n_3 = 5851; n_4 = 5949;
m_1 = mean(ones,2);
m_2 = mean(twos,2);

Sw = 0; % within class variances
for k = 1:n_3
    Sw = Sw + (ones(:,k) - m_1)*(ones(:,k) - m_1)';
end
for k = 1:n_4
   Sw =  Sw + (twos(:,k) - m_2)*(twos(:,k) - m_2)';
end

Sb = (m_1-m_2)*(m_1-m_2)'; % between class

%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

v_1 = w'*ones;
v_2 = w'*twos;

%% Make dogs below the threshold

if mean(v_1) > mean(v_2)
    w = -w;
    v_1 = -v_1;
    v_2 = -v_2;
end

%% Plot two digits projections (not for function)

figure(4)
plot(v_1,zeros(6742),'ob','Linewidth',2)
hold on
plot(v_2,ones(5958),'dr','Linewidth',2)
title("two digits: 1 and 2");
legend();

%% Find the threshold value

sort_ones = sort(v_1);
sort_sevens = sort(v_2);

t1 = length(sort_ones);
t2 = 1;
while sort_ones(t1) > sort_sevens(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort_ones(t1) + sort_sevens(t2))/2;


%%
figure(5)
subplot(1,2,1)
histogram(sort_ones,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit_1')
subplot(1,2,2)
histogram(sort_sevens,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit_2');


%% Find the threshold value

sort_ones = sort(v_1);
sort_twos = sort(v_2);

t1 = length(sort_ones);
t2 = 1;
while sort_ones(t1) > sort_twos(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort_ones(t1) + sort_twos(t2))/2;


%%
figure(5)
subplot(1,2,1)
histogram(sort_ones,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit:1')
subplot(1,2,2)
histogram(sort_twos,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit:2')








%% three digits
feature = 80;
% Number of observations of each class
n_3 = 5923;%0
n_4 = 6265;%7
n_5 = 5421;
N = n_3 + n_4 + n_5;
%Mean of each class

P = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
threes = P(1:feature,labels == 0);
fours = P(1:feature,labels == 7);
fives = P(1:feature, labels == 5);

m_3 = mean(threes,2); 
m_4 = mean(fours,2);
m_5 = mean(fives,2);

%%
Sw = 0; % within class variances
for k = 1:n_3
    Sw = Sw + (ones(:,k) - m_3)*(ones(:,k) - m_3)';
end
for k = 1:n_4
   Sw =  Sw + (sevens(:,k) - m_4)*(sevens(:,k) - m_4)';
end
for k = 1:n_5
   Sw =  Sw + (sevens(:,k) - m_5)*(sevens(:,k) - m_5)';
end
%
me = [m_3,m_4,m_5];
overall_m = (m_3+m_4+m_5)/3;
Sb = 0;
for j = 1:3
    Sb = Sb + (me(:,j)-overall_m)*(me(:,j)-overall_m)';
end
%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

v_3 = w'*threes;
v_4 = w'*fours;
v_5 = w'*fives;

%% Make dogs below the threshold

% if mean(v_1) > mean(v_7)
%     w = -w;
%     v_1 = -v_1;
%     v_7 = -v_7;
% end

%% Plot dog/cat projections (not for function)

figure(4)
plot(v_3,threes(6131),'ob','Linewidth',2)
hold on
plot(v_4,fours(5842),'or','Linewidth',2)
hold on
plot(v_5,fives(5421),'og','Linewidth',2)
%% Find the threshold value

sort_threes = sort(v_3);
sort_fours = sort(v_4);
sort_fives = sort(v_5);

t3 = length(sort_threes);
t4 = 1;
while sort_threes(t3) > sort_fours(t4)
    t3 = t3 - 1;
    t4 = t4 + 1;
end
threshold = (sort_threes(t3) + sort_fours(t4))/2;


%%
figure(5)
subplot(1,2,1)
histogram(sort_threes,40); hold on, plot([threshold threshold], [0 500],'r')
set(gca,'Fontsize',14)
title('digit3')
subplot(1,2,2)
histogram(sort_fours,40); hold on, plot([threshold threshold], [0 500],'r')
set(gca,'Fontsize',14)
title('digit4')

%%
t3 = length(sort_threes);
t5 = 1;
while sort_threes(t4) > sort_fives(t5)
    t3 = t3 - 1;
    t5 = t5 + 1;
end
threshold = (sort_threes(t3) + sort_fives(t5))/2;


figure(5)
subplot(1,2,1)
histogram(sort_threes,40); hold on, plot([threshold threshold], [0 500],'r')
set(gca,'Fontsize',14)
title('digit:3')
subplot(1,2,2)
histogram(sort_fives,40); hold on, plot([threshold threshold], [0 500],'r')
set(gca,'Fontsize',14)
title('digit:5')










%%  most easy
%% pick two digits

feature = 20;

n_1 = 6742;
n_7 = 6265;
P = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
ones = P(1:feature,labels == 1);
sevens = P(1:feature,labels == 7);

%% Calculate scatter matrices

m_1 = mean(ones,2);
m_7 = mean(sevens,2);

Sw = 0; % within class variances
for k = 1:n_1
    Sw = Sw + (ones(:,k) - m_1)*(ones(:,k) - m_1)';
end
for k = 1:n_7
   Sw =  Sw + (sevens(:,k) - m_7)*(sevens(:,k) - m_7)';
end

Sb = (m_1-m_7)*(m_1-m_7)'; % between class

%% Find the best projection line

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);

%% Project onto w

v_1 = w'*ones;
v_7 = w'*sevens;

%% Make dogs below the threshold

if mean(v_1) > mean(v_7)
    w = -w;
    v_1 = -v_1;
    v_7 = -v_7;
end

%% Plot dog/cat projections (not for function)

figure(4)
plot(v_1,zeros(6742),'ob','Linewidth',2)
hold on
plot(v_7,ones(6265),'dr','Linewidth',2)


%% Find the threshold value

sort_ones = sort(v_1);
sort_sevens = sort(v_7);

t1 = length(sort_ones);
t2 = 1;
while sort_ones(t1) > sort_sevens(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort_ones(t1) + sort_sevens(t2))/2;

%%
figure(5)
subplot(1,2,1)
histogram(sort_ones,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit:1')
subplot(1,2,2)
histogram(sort_sevens,60); hold on, plot([threshold threshold], [0 600],'r')
set(gca,'Fontsize',14)
title('digit:7');
%%
a = size(find((sort_ones >= threshold)));
accuracy = 1- a/size(sort_ones);






%% test
[images_t, labels_t] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

[m_t,~,image_num_t] = size(images_t);
reshape_img_t = zeros(m_t^2, image_num_t);
for i = 1: image_num_t
    reshape_img_t(:,i) = im2double(reshape(images_t(:,:,i),m_t^2,1));
end
%%
TestMat = U(:,1:feature)'*reshape_img;
pvals = w'*TestMat;
zeros = find(labels == 3);
ones = find(labels == 4);
nCorrect = 0;
index = 0;

for i = 1:length(ones)
    index = index + 1;
    if pvals(ones(index)) < threshold;
        nCorrect = nCorrect + 1;
    end
end
index = 0;
for i = 1:length(zeros)
    index = index + 1;
    if pvals(zeros(index)) < threshold;
        nCorrect = nCorrect + 1;
    end
end
nCorrect / (length(zeros)+length(ones))








%%
% classification tree on fisheriris data
digit_0 = images(labels == 0);
digit_1 = images(labels == 1); 
digit_1 = digit_1(1:length(digit_0),:);
%%
meas = meas([double(digit_0) double(digit_1)]);
%%
load fisheriris;
tree=fitctree(meas,species,'MaxNumSplits',3,'CrossVal','on');
view(tree.Trained{1},'Mode','graph');
classError = kfoldLoss(tree)
%%
% SVM classifier with training data, labels and test set
xtrain = reshape_img;
label = labels;
test = reshape_img_t;
%%
Mdl = fitcsvm(reshape_img(1,:),label');
labels = predict(Mdl,test);
%%
% function [U,S,V,threshold,w,digit1,digit2] = dc_trainer(digit_x,digit_y,feature)
%     nd = size(digit_x);
%     nc = size(digit_y);
%     [U,S,V] = svd([digit_x digit_y],'econ'); 
%     P = S*V';
%     U = U(:,1:feature); % Add this in
%     digit_x_1 = P(1:feature,1:nd);
%     digit_y_1 = P(1:feature,nd+1:nd+nc);
%     m_x = mean(digit_x_1,2);
%     m_y = mean(digit_y_1,2);
% 
%     Sw = 0;
%     for k=1:nd
%         Sw = Sw + (digit_x_1(:,k)-m_x)*(digit_x_1(:,k)-m_x)';
%     end
%     for k=1:nc
%         Sw = Sw + (digit_y_1(:,k)-mc)*(digit_y_1(:,k)-mc)';
%     end
%     Sb = (md-mc)*(md-mc)';
%     
%     [V2,D] = eig(Sb,Sw);
%     [lambda,ind] = max(abs(diag(D)));
%     w = V2(:,ind);
%     w = w/norm(w,2);
%     vdog = w'*dogs;
%     vcat = w'*cats;
%     
%     if mean(vdog)>mean(vcat)
%         w = -w;
%         vdog = -vdog;
%         vcat = -vcat;
%     end
%     
%     % Don't need plotting here
%     sortdog = sort(vdog);
%     sortcat = sort(vcat);
%     t1 = length(sortdog);
%     t2 = 1;
%     while sortdog(t1)>sortcat(t2)
%     t1 = t1-1;
%     t2 = t2+1;
%     end
%     threshold = (sortdog(t1)+sortcat(t2))/2;
% 
%     % We don't need to plot results
% end
% 
% [U,S,V] = svd(reshape_img,'econ');
% ones = P(1:feature,labels == 1);
% twos = P(1:feature,labels == 2);
% [U,S,V,threshold,w,sortdog,sortcat] = dc_trainer(ones,twos,20);
