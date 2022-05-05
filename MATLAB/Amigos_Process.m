per_user = [];
per_user_label = [];

for i=1:40

    per_emotion = [];
    per_emotion_label = [];

    filepath = "Data_Preprocessed_P"+ num2str(i,'%02d') + "/Data_Preprocessed_P" + num2str(i,'%02d') + ".mat";

    data = load(filepath);

    EEG = data.joined_data;

    for j = 1:20
        
        EEG_Emotion = EEG{j};

        label_Emotion = data.labels_selfassessment{j};

        per_emotion_electrode = [];

        if((sum(isnan(EEG_Emotion),'all') > 0) || (size(EEG_Emotion,1) == 0) || (size(label_Emotion,1) == 0))
            per_emotion_electrode = zeros(14,3);
        else
            for k = 1:14    

                EEG_Emotion_Electrode = EEG_Emotion(:,k);
                theta = bandpower(EEG_Emotion_Electrode,128,[4 7]);
                alpha = bandpower(EEG_Emotion_Electrode,128,[8 12]);
                beta  = bandpower(EEG_Emotion_Electrode,128,[13 30]);
                per_emotion_electrode = [per_emotion_electrode; theta alpha beta];
            end

            per_emotion_electrode = reshape(per_emotion_electrode,1,[]);
            per_emotion = [per_emotion; per_emotion_electrode];
            per_emotion_label = [per_emotion_label; label_Emotion];

        end
    end

    per_user = [per_user; per_emotion];
    per_user_label = [per_user_label; per_emotion_label];

end

save('all_features_amigos.mat',"per_user");
save('labels_amigos.mat',"per_user_label");