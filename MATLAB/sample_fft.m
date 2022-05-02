dreamer_mat_file = [pwd '\DREAMER.mat'];
file_data = load(dreamer_mat_file);

random_signal = file_data.DREAMER.Data{1,22}.EEG.stimuli{17,1}(:,13);
mean_ary(1:length(random_signal), 1) = mean(random_signal);
new_stimuli_signal = random_signal - mean_ary;

Fs = 128;

Y = fft(new_stimuli_signal);
L = length(new_stimuli_signal);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
figure;
f = Fs*(0:(L/2))/L;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
