clear all; close all; clc
%% get the signal plot of GNR
figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
plot((1:length(y))/Fs,y);
xlabel('Time [sec]'); ylabel('Amplitude');
title("'Sweet Child O' Mine");
p8 = audioplayer(y,Fs);playblocking(p8);
%% Build Gabor filter of GNR
Signal = y';
L = tr_gnr;
n = length(Signal);
k = (1/L)*[0:n/2-1 -n/2:-1]; %set the frequency domain k
ks = fftshift(k); % fourier transform of k
t2 = linspace(0,L,n+1); %discretization
t = t2(1:n);  % only the first n points (periodicity)

a = 1000; % set window scale
tau = 0:0.1:L; % set window center
for j = 1:length(tau)
    g = exp(-a*(t-tau(j)).^2); % window function
    yg = g.*Signal; % apply gabor filter
    ygt = fft(yg); 
    ygt_spec(:,j) = fftshift(abs(ygt)); %reorder
end

pcolor(tau, ks, ygt_spec)
shading interp
colormap(hot)
set(gca,'ylim',[0 1000],'Fontsize',16)
xlabel('time(t)'),ylabel('frequency(k)')
title('a = 1000','Fontsize',16)

%% get the signal plot of floyd
figure(2)
[y2, Fs2] = audioread('Floyd.m4a');
tr_gnr2 = length(y2)/Fs2; % record time in seconds
% plot((1:length(y2))/Fs2,y2);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title('Comfortably Numb');
% p = audioplayer(y2,Fs2); playblocking(p);
%% Build Gabor filter of floyd
s = y2';
L2 = tr_gnr2;
n2 = length(s);
k = (1/L2)*[0:n2/2-1 -n2/2:-1];
ks = fftshift(k);
t2 = linspace(0,L2,n2+1);
t = t2(1:n2);

a2 = 6000;
tau = 0:1:L2;
for j = 1:length(tau)
    g2 = exp(-a2*(t-tau(j)).^2); % window function
    yg2 = g2.*s;
    ygt2 = fft(yg2);
    ygt2_spec(:,j) = fftshift(abs(ygt2));
end

% pcolor(tau, ks, ygt2_spec(1:end-1,:));
% shading interp
% colormap(hot);
% set(gca,'ylim',[0,1000],'Fontsize',16)
% xlabel('time(t)'),ylabel('frequency(k)')
% title('a2 = 6000','Fontsize',16)

%% filter bass
% Floyd
s2_fft = fft(s); %fourier transform
stg_spec = zeros(length(tau),n2);
notes = zeros(length(tau),n2); % original set

s2_filter = s2_fft.*fftshift(50 < abs((1:n2)/Fs2) < 250); % filter bass
s2_inverse = ifft(s2_filter); %inverse filter function into time domain
t = (1:n2)/Fs2;
for i =1:length(tau)
   g = exp(-10*(t-tau(i)).^2);
   Sg = g.*s2_inverse;
   Sgt = fft(Sg);
   
   Sgt = Sgt(1:n-1);
   ind = max(abs(Sgt));
    filter = exp(-0.01 * (k - k(ind)).^2);
    Sgtf = Sgt .* filter; % filter overtone with a Gaussian filter
    bass_notes = Sgtf;
    bass_notes(k > 250) = 0; % filter out any frequency higher than 250
    guitar_notes = Sgtf - bass_notes;
   Sgt_spec(i,:) = abs(fftshift(Sgt));
end 

%% plot the spectrogram of bass in floyd
figure(3)
pcolor(tau, ks, Sgt_spec(:,1:end-1)'); % note: the dimention should be agree
shading interp
title('Bass Spectuogram of "Comfortably Numb"')
set(gca,'ylim',[50 250],'Fontsize',16)
xlabel('Time(sec)');
ylabel('Frequency(Hertz)');
colormap(hot);
%% plot the spectrogram of guitar in floyd
a = 1500;
tau = 0:01:L2;
for j = 1:length(tau)
    g2 = exp(-a*(t-tau(j)).^2); % window function
    yg2 = g2.*s;
    ygt2 = fft(yg2);
    ygt2_spec(:,j) = fftshift(abs(ygt2));
end
s2_fft = fft(s); %fourier transform
stg_spec = zeros(length(tau),n2);
notes = zeros(length(tau),n2); % original set

s2_filter = s2_fft.*fftshift(50 < abs((1:n2)/Fs2) < 250); % filter bass
s2_inverse = ifft(s2_filter); %inverse filter function into time domain
t = (1:n2)/Fs2;
for i =1:length(tau)
   g = exp(-10*(t-tau(i)).^2);
   Sg = g.*s2_inverse;
   Sgt = fft(Sg);
   Sgt_spec(i,:) = abs(fftshift(Sgt));
end 

figure(4)
pcolor(tau, ks, Sgt_spec(:,1:end-1)'); % note: the dimention should be agree
shading interp
title('Guitar Spectuogram of "Comfortably Numb"')
set(gca,'ylim',[200 500],"xlim",[0 15],'Fontsize',16)
xlabel('Time(sec)');
ylabel('Frequency(Hertz)');
colormap(hot);