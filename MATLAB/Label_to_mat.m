img_labels = load('labels_amigos.mat');

label = img_labels.per_user_label;

threshold1 = 3.5;
threshold2 = 6.4;

k = zeros(size(label,1),2);

for i=1:size(label,1)
    %Valence
    if(label(i,1) > threshold2)
        k(i,1) = 2;
    elseif(label(i,1) > threshold1)
        k(i,1) = 1;
    else
        k(i,1) = 0;
    end
    %Arousal
    if(label(i,2) > threshold2)
        k(i,2) = 2;
    elseif(label(i,2) > threshold1)
        k(i,2) = 1;
    else
        k(i,2) = 0;
    end
end


save('Images_Class.mat','k','-v7.3');