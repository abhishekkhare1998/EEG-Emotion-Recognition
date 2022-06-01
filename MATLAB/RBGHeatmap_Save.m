data = load("all_features_amigos.mat");

electrode_coord = [4, 8;
                   1, 7;
                   3, 7;
                   2, 6;
                   1, 5;
                   1, 3;
                   4, 1;
                   6, 1;
                   9, 3;
                   9, 5;
                   8, 6;
                   7, 7;
                   9, 7;
                   6, 8];

electrode_coord = electrode_coord + 1;

features = data.per_user;

Img_Cell = {};

for j = 1:size(features,1)

    user = features(j,:);
    red = user(1:14);
    blue = user(15:28);
    green = user(29:42);

    imred = zeros(11);
    imblue = zeros(11);
    imgreen = zeros(11);

    for i=1:14
        imred(electrode_coord(i,1),electrode_coord(i,2)) = red11(i);
        imblue(electrode_coord(i,1),electrode_coord(i,2)) = blue11(i);
        imgreen(electrode_coord(i,1),electrode_coord(i,2)) = green11(i);
    end


    imrednew = imresize(imred,[512,512]);
    imbluenew = imresize(imblue,[512,512]);
    imgreennew = imresize(imgreen,[512,512]);

    imtot = cat(3,imrednew,imbluenew,imgreennew);

    Img_Cell{end+1} = imtot;

end

save('all_img_amigos.mat','Img_Cell','-v7.3');
