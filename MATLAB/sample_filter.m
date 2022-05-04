clear
close all

dreamer_dataset = load('EEG-Emotion-Recognition\MATLAB\DREAMER.mat');
dreamer_sig1 = dreamer_dataset.DREAMER.Data{1,22}.EEG.stimuli{17,1};
dreamer_sig1_e1 = dreamer_sig1(:,13);

% FFT
Fs = 128;
N_fft = 128;

% Sine Test
test_cosine = cos(2*pi*15*(0:(1/Fs):1/15));
input_sig = dreamer_sig1_e1 - mean(dreamer_sig1_e1);


fft_sig1 = fft(input_sig,N_fft);
fft_sig1 = fft_sig1/max(fft_sig1);

figure(1)
plot((0:length(input_sig)-1)*1/Fs,input_sig);
xlabel("Time in (s)")

figure(2)
plot((-N_fft/2:N_fft/2-1)*Fs/N_fft,fftshift(abs(fft_sig1)));
title("FFT")

figure(3)
plot((-N_fft/2:N_fft/2-1)*Fs/N_fft,20*log(fftshift(abs(fft_sig1))));
ylabel('Power (db)')
title("FFT")

L = length(input_sig);
Y = fft(input_sig);

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = Fs*(0:(L/2))/L;

figure(4)
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

%%
    
% h1=firpm(17,[0 4 7 8 64]/(Fs/2), [0 1 1 0 0]);
% 
% f = [0 2 4 7 8 64]/(Fs/2);
% a = [0 0 1 1 0 0];
% h1 = firpm(17,f,a);
% 
% figure(5)
% plot((-0.5:1/4096:0.5-1/4096)*Fs,20*log10(abs(fftshift(fft(h1,4096)))))
% % axis([-0.5 0.5 -100 5])
% title('Equal Ripple Remez Filter Design Example')

% [n,fo,ao,w] = firpmord([1500 2000],[1 0],[0.001 0.01],8000);
% b = firpm(n,fo,ao,w);
% fvtool(b,1)

rp = 3;           % Passband ripple in dB 
rs = 40;          % Stopband ripple in dB
fs = 128;        % Sampling frequency
dev = [10^(-rs/20) (10^(rp/20)-1)/(10^(rp/20)+1) 10^(-rs/20)]; 

f = [3 4 7 8];    % Cutoff frequencies
a = [0 1 0];        % Desired amplitudes

[n,fo,ao,w] = firpmord(f,a,dev,fs);
bp1 = firpm(n,fo,ao,w);

figure(5)
freqz(bp1,1,1024,fs)
title('4-7Hz BP Filter Designed to Specifications')

f = [7 8 12 13];    % Cutoff frequencies
a = [0 1 0];        % Desired amplitudes

[n,fo,ao,w] = firpmord(f,a,dev,fs);
bp2 = firpm(n,fo,ao,w);

figure(6)
freqz(bp2,1,1024,fs)
title('8-12Hz BP Filter Designed to Specifications')

f = [11 12 30 31];    % Cutoff frequencies
a = [0 1 0];        % Desired amplitudes

[n,fo,ao,w] = firpmord(f,a,dev,fs);
bp3 = firpm(n,fo,ao,w);

figure(6)
freqz(bp3,1,1024,fs)
title('12-30Hz BP Filter Designed to Specifications')
