img_labels = load('labels_amigos.mat');

label = img_labels.per_user_label;

threshold = 4.9;

k = zeros(size(label,1),2);

for i=1:size(label,1)
    %Valence
    if(label(i,1) > threshold)
        k(i,1) = 1;
    else
        k(i,1) = 0;
    end
    %Arousal
    if(label(i,2) > threshold)
        k(i,2) = 1;
    else
        k(i,2) = 0;
    end
end

name = 'Images_Class.csv';
csvwrite(name,k);