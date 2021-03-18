clear all; close all; clc
%% load video

video1 = VideoReader('ski_drop_low.mp4');
video2 = VideoReader('monte_carlo_low.mp4');

dt1 = 1/video1.Framerate;
t1 = 0:dt1:video1.Duration;
vidFrames = read(video1);
numFrames = get(video1,'numberOfFrames');
%%
frame = im2double(vidFrames(:,:,:,1));
X = zeros(size(frame,1)*size(frame,2),numFrames);
for j = 1:numFrames
    frame = vidFrames(:,:,:,j);
    frame = im2double(rgb2gray(frame));
    X(:,j) = reshape(frame,540*960,[]);
    % imshow(frame);drawnow
end
%%
X1 = X(:,1:end-1);
X2 = X(:,2:end);
[U,S,V] = svd(X1,'econ');

figure(1)
plot(diag(S)/sum(diag(S)),'ko','Linewidth',2);
set(gca,'FontSize',16,'Xlim',[0,60]);
ylabel('Singular Value');
xlabel('Energy Captured');
title('Energy of Singular Value');
%%
% r = 2;
% U_r = U(:,1:r);
% S_r = S(1:r,1:r);
% V_r = V(:,1:r);
% low_rank_dynamic = U_r' * X2 * V_r / S_r; % low rank dynamic
% [W_r, D] = eig(low_rank_dynamic);
% phi = X2 * V_r / S_r * W_r; % DMD modes
% lambda = diag(D); % eigenvalues of discrete time
% omega = log(lambda)/dt1; % eigenvalue of continious time
% 
% figure(2)
% bar(abs(omega),'b');
% title('Absolute Omega Values','Fontsize',16);
% set(gca,'Xtick',[]);
% xlabel('omegas');
% ylabel('Absolute Value');
%% compute DMD mode amplitudes
%x1 = X(:,1);
%%
%b = phi\x1; % DMD mode amplitudes

%% reconstruction
% mm1 = size(X1,2);
% X_modes = zeros(2,mm1);
%%
r = 2;
U = U(:,1:r);
S = S(1:r,1:r);
V = V(:,1:r);

S = U' * X2 * V * diag(1./diag(S));
[eV,D] = eig(S);
mu = diag(D);
omega = log(mu) / dt1;
phi = U * eV;
y0 = phi \ X1(:,1); % pseufoinverse to get initial conditions
u_modes = zeros(length(y0),numFrames);
for iter = 1:numFrames
    u_modes(:,iter) = y0 .* exp(omega*iter);
end
X_dmd = phi * u_modes;%low rank 
%%
Xsparse = X1 - abs(X_dmd(1:453)); % high rank
R = Xsparse.*(Xsparse < 0);
%%
X_bg = R + abs(X_dmd(:,end-1)); % background
X_fg = Xsparse - R; % foreground
X_recons = X_bg + X_fg;
%%
data_bg = reshape(X_bg,[size(frame,1),size(frame,2),(length(t1)-1)]);
%imshow(uint8(data(:,:,453)))
for i = 1:453
    imshow(im2uint8(data_bg(:,:,i)))
end
%%
data_fg = reshape(X_fg,[size(frame,1),size(frame,2),(length(t1)-1)]);

for i = 1:453
    imshow(im2uint8(data_bg(:,:,i)))
end


