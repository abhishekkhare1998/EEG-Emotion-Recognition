img_mat_file = 'all_img_amigos.mat';
img_data = load(img_mat_file);

%csvwrite('Img_Data.csv',img_data.Img_Cell);

for i=1:numel(img_data.Img_Cell)
    A_temp = img_data.Img_Cell{i};
    name_base = ['Images_Data' '/Img_Data_' num2str(i) '.csv'];
    csvwrite(name_base,A_temp);
end