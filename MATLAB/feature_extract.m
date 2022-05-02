dreamer_mat_file = [pwd '\DREAMER.mat'];
file_data = load(dreamer_mat_file);
all_features = [];
for i = 1:23
    per_emotion = [];
    for j = 1:18

        stimuli_signal = file_data.DREAMER.Data{1,i}.EEG.stimuli{j,1};
        features = [];

        for k = 1:14
            el_signal = stimuli_signal(:,k);
            mean_ary(1:length(el_signal), 1) = mean(el_signal);
            new_stimuli_signal = el_signal - mean_ary;
            theta_4_7 = bandpower(new_stimuli_signal,128,[4 7]);
            alpha_8_12 = bandpower(new_stimuli_signal,128,[8 12]);
            beta_13_30 = bandpower(new_stimuli_signal,128,[13 30]);
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
