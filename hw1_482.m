clear all; close all; clc
load subdata.mat 

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); % domain discretization
x = x2(1:n); % only the first n points (periodicity)
y = x; % set y
z = x; % set z (since the 3-D dimension)

k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);
[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

% Reshape the data
% for j=1:49
%     Un(:,:,:)=reshape(subdata(:,j),n,n,n);
%     M = max(abs(Un),[],'all');
%     close all, 
%     isosurface(X,Y,Z,abs(Un)/M,0.7)
%     axis([-20 20 -20 20 -20 20]), grid on, drawnow
%     pause(1)
% end

% averaging of the spectrum
Un = zeros(n,n,n);
Utnave = zeros(n,n,n);
for j = 1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Utn = fftshift(fftn(reshape(subdata(:,j),n,n,n)));
    Utnave = Utnave + Utn;
end
Utave = Utnave/49;

% determine the frequency signature (center frequency)
[m,ind] = max(Utave(:));
[ind_x,ind_y,ind_z] = ind2sub([n,n,n],ind)
center_Kx = Kx(ind_x,ind_y,ind_z);
center_Ky = Ky(ind_x,ind_y,ind_z);
center_Kz = Kz(ind_x,ind_y,ind_z);

% filter the signal
tau = 0.2; % center of the window
filter = exp(-tau*((Kx - center_Kx).^2 +(Ky - center_Ky).^2 +(Kz - center_Kz).^2)); % define the filter
% set original x,y,and z
x = zeros(49,1);
y = zeros(49,1);
z = zeros(49,1);
% use the signal of subdata and rearrange the data
for j = 1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    Utn = fftshift(fftn(reshape(subdata(:,j),n,n,n)));
    unf = filter.*Utn; % build the function with filter
    unfshift = ifftshift(unf);
    un = ifftn(unfshift); % rearrange its order and get into the time domain
    [m,ind] = max(un(:)); % find the max of the signals as the target
    [ind_x,ind_y,ind_z] = ind2sub([n,n,n],ind); % get index in 3 dimension to use target ind
    x(j,1) = X(ind_x,ind_y,ind_z);
    y(j,1) = Y(ind_x,ind_y,ind_z);
    z(j,1) = Z(ind_x,ind_y,ind_z);
end
figure(3);
% plot the path of the submarine
plot3(x,y,z,'o-'); grid on, drawnow;
xlabel('x');
ylabel('y');
zlabel('height');
title('the path of the submarine');
set(gca,'Fontsize',16);

% record the data of x and y in a table to locate the submarine
T = table(x,y);
writetable(T,'location_table.xls');