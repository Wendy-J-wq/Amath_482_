clear all; close all; clc;
%%
% load movie
load('cam1_1.mat'); load('cam2_1.mat'); load('cam3_1.mat');
% implay(vidFrames1_1);
numFrames1 = size(vidFrames3_1,4);
[height1 width1 rgb1 num_frames1] = size(vidFrames1_1);

%%
% watch movie
x1 = []; y1 = [];
for j = 1:numFrames1
    X1_1 = vidFrames1_1(:,:,:,j);
    X2_1 = vidFrames2_1(:,:,:,j);
     X3_1 = vidFrames3_1(:,:,:,j);
%     I = rgb2gray(X1_1);
%     [y,x] = find(I==255) ;
%     n=length(x);  xx=x.*x; yy=y.*y; xy=x.*y;
%     A=[sum(x) sum(y) n;sum(xy) sum(yy) sum(y);sum(xx) sum(xy) sum(x)];
%     B=[-sum(xx+yy) ; -sum(xx.*y+yy.*y) ; -sum(xx.*x+xy.*y)];
%     a=A\B;
%     xc = -.5*a(1)
%     yc = -.5*a(1)
%     I = im2double(I);
    imshow(X3_1); drawnow
    %imshow(I);drawnow
    S = sum(I,3);
    
%     [~,idx] = max(S(:));
%     [row,col] = ind2sub(size(S),idx);
%     x1 = [x1 x];
%     y1 = [y1 y];
end
% plot(xc,yc);
%%
% for k = 1:numFrames1-1
% d(:, :, k) = imabsdiff((g(:, :, k), g(:, :, k+1));
% end
% imview(d(:, :, 1), [])
%%
I = X1_1;
S = sum(I,3); %
[~,idx] = max(S(:));
[row,col] = ind2sub(size(S),idx);
%%
hsv = rgb2hsv(X1_1);
v = hsv(:,:,3);
max_v = max(max(v));
[r, c] = find(v == max_v);
%%
plot(r,c)