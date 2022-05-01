dreamer_mat_file = [pwd '\DREAMER.mat'];
file_data = load(dreamer_mat_file);
for i = 1:23
    for j = 1:18
        name = [pwd '\..\Python\Data' '\Dreamer_' num2str(i) '_' num2str(j) '.csv'];
        csvwrite(name, file_data.DREAMER.Data{1,i}.EEG.baseline{j,1});
    end
end
