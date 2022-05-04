per_user = [];

for i=1:40

    per_emotion = [];

    filepath = "Data_Original_P"+ num2str(i,'%02d') + "/Data_Original_P" + num2str(i,'%02d') + ".mat";

    data = load(filepath);

    EEG = data.EEG_DATA;

    for j = 1:20
        
        EEG_Emotion = EEG{j};

        per_emotion_electrode = [];

        if(size(EEG_Emotion,1) == 0)
            per_emotion_electrode = zeros(14,3);
        else
            for k = 4:17    

                EEG_Emotion_Electrode = EEG_Emotion(:,k);
                theta = bandpower(EEG_Emotion_Electrode,128,[4 7]);
                alpha = bandpower(EEG_Emotion_Electrode,128,[8 12]);
                beta  = bandpower(EEG_Emotion_Electrode,128,[13 30]);
                per_emotion_electrode = [per_emotion_electrode; theta alpha beta];
            end
        end
        
        per_emotion_electrode = reshape(per_emotion_electrode,1,[]);

        per_emotion = [per_emotion; per_emotion_electrode];

    end

    per_user = [per_user; per_emotion];

end

save('all_features_amigos.mat',"per_user");