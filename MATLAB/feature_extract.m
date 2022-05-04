clear 
close all
clc

dreamer_mat_file = [pwd '\DREAMER.mat'];
file_data = load(dreamer_mat_file);
all_features = [];

% filter_def provides the Band Pass filters
% bp1 - Theta 4-7 Hz
% bp2 - Alpha 8-12 Hz
% bp3 - beta   12 - 30 Hz
[bp1,bp2,bp3,~] =  filter_def(0,0);

for i = 1:23
    per_emotion = [];
    for j = 1:18

        stimuli_signal = file_data.DREAMER.Data{1,i}.EEG.stimuli{j,1};
        features = [];

        for k = 1:14
            el_signal = stimuli_signal(:,k);
            mean_ary(1:length(el_signal), 1) = mean(el_signal);
            new_stimuli_signal = el_signal - mean_ary;
            
            % Apply Bandpass filtering to get the theta, alpha and beta
            new_stimuli_signal_theta = filter(bp1,1,new_stimuli_signal);
            new_stimuli_signal_alpha = filter(bp2,1,new_stimuli_signal);
            new_stimuli_signal_beta = filter(bp3,1,new_stimuli_signal);
            

            theta_4_7 = bandpower(new_stimuli_signal_theta,128,[4 7]);
            alpha_8_12 = bandpower(new_stimuli_signal_alpha,128,[8 12]);
            beta_13_30 = bandpower(new_stimuli_signal_beta,128,[13 30]);
            features = [features ; [theta_4_7 alpha_8_12 beta_13_30]];
        end
        features = reshape(features, 1, []);
        el_signal = [];
        stimuli_signal = [];
        mean_ary = [];
    per_emotion = [per_emotion; features];
    end
    all_features = [all_features; per_emotion];
end
a = 1;

save('all_features.mat', 'all_features');
