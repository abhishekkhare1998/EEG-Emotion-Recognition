dreamer_mat_file = [pwd '\DREAMER.mat'];
file_data = load(dreamer_mat_file);

for i = 1:23
    valence_scores = file_data.DREAMER.Data{1, i}.ScoreValence;
    arousal_scores = file_data.DREAMER.Data{1, i}.ScoreArousal;
    dominance_scores = file_data.DREAMER.Data{1, i}.ScoreDominance;
    electrodes = [1:18]';
    metadata_matrix = [electrodes valence_scores arousal_scores dominance_scores];
    for j = 1:18
        name_base = [pwd '\..\Python\Data\Baseline' '\Dreamer_' num2str(i) '_' num2str(j) '.csv'];
        name_stimuli = [pwd '\..\Python\Data\Stimuli' '\Dreamer_' num2str(i) '_' num2str(j) '.csv'];
        name_meta = [pwd '\..\Python\Data\Metadata' '\Dreamer_VAD_' num2str(i) '.csv'];
        csvwrite(name_base, file_data.DREAMER.Data{1,i}.EEG.baseline{j,1});
        csvwrite(name_stimuli, file_data.DREAMER.Data{1,i}.EEG.stimuli{j,1});
        csvwrite(name_meta, metadata_matrix);
    end
end
