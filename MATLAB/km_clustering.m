db_name = 'AMIGOS';   % AMIGOS, DREAMER
k_clusters = 4;
use_PCA = true;
PCA_dimensions = 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(db_name,'AMIGOS')
    load_file_name = "all_features_amigos.mat";
    data = load(load_file_name);
    input = data.per_user;
end
if strcmp(db_name,'DREAMER')
    load_file_name = "all_features.mat";
    data = load(load_file_name);
    input = data.all_features;
end



feat_vec_length = length(input(1,:));
if (use_PCA)
[wcoeff,~,latent,~,explained] = pca(input','VariableWeights','variance');
feat_vec_length = PCA_dimensions;
final_inp = wcoeff(:,1:PCA_dimensions);
else
    final_inp = input;
end

avg_vector = zeros(1, feat_vec_length);

for i = 1:length(input(:,1))
    avg_vector = avg_vector + input(i,1:feat_vec_length);
end
avg_vector = avg_vector/length(input(:,1));

random_init_images = zeros(1, feat_vec_length, k_clusters);
for i = 1:k_clusters
    random_init_images(:, :, i) = randomise_image(avg_vector);
end

[class_averages_rand, clustered_data_rand] = perform_km_clustering(input(:,1:feat_vec_length), random_init_images);
clustered_data_rand = clustered_data_rand + 1;
C = hist(clustered_data_rand,1:max(clustered_data_rand));
histogram('Categories',string((1:k_clusters)),'BinCounts',C)
a = 1;

function [class_averages, clustered_data] = perform_km_clustering(training_images, averages)
classified_data = perform_knn(averages, training_images);

num_of_changes = 11;
old_classified = classified_data;

while num_of_changes >= 4
    new_averages = calculate_new_averages(classified_data, training_images, 8);
    classified_data_new = perform_knn(new_averages, training_images);
    num_of_changes = check_classification_changes(old_classified, classified_data_new);
    old_classified = classified_data_new;
end
clustered_data = classified_data_new;
class_averages = new_averages;


end

function num_of_changes = check_classification_changes(old_classified, classified_data_new)
diff = old_classified - classified_data_new;
indices = find(diff ~= 0);
num_of_changes = length(indices);
end

function new_averages = calculate_new_averages(classified_data, images, num_of_clusters)
cluster_matrix(1:num_of_clusters,1:length(classified_data)) = -1;
temp_ary = zeros(num_of_clusters, 1);
new_averages = zeros(1, length(images(1,:)), num_of_clusters);
    for i = 1:length(classified_data)
        cluster_matrix(classified_data(i)+1, temp_ary(classified_data(i)+1)+1) = i;
        temp_ary(classified_data(i)+1) = temp_ary(classified_data(i)+1) + 1;
    end

for i = 1:num_of_clusters
    sum = zeros(1, length(images(1,:)));
    for j=1:temp_ary(i)
        sum = sum + images(cluster_matrix(i, j), :);
    end
    new_averages(:, :, i) = sum/temp_ary(i);
    
end

end

function [classified_data, error_rate_matrix] = perform_knn(averages, test_images)

test_images_len = length(test_images(:, 1));
num_of_clusters = length(averages(1, 1, :));

classified_data = zeros(test_images_len, 1);

for i = 1:test_images_len
    min_dist = 10000000000000000;
    image_vec =  test_images(i,:);
    
    for j = 1:num_of_clusters
        mean_vec = averages(:,:,j);
        dist = dot(image_vec-mean_vec, image_vec-mean_vec);
        %dist = norm(image-class_avg_list(:,:,j), 2);
        if min_dist >= dist
            min_dist = dist;
            computed_class = j-1;
        end
    end
    classified_data(i) = computed_class;

end

end

%%%%

function random_image = randomise_image(serialsed_avg)
    for i = 1:length(serialsed_avg)
        %var_1 = max(1, serialsed_avg(i)/2);
        var_1 = serialsed_avg(i);
        serialsed_avg(i) = max(0, serialsed_avg(i) + (-1*var_1) + 2*var_1*rand(1,1));
    end
    random_image = serialsed_avg;
end

function serial_vec = serialise_vector(vector)
    init_vec_len = zeros(1, length(vector(1,:))*length(vector(:, 1)));
    pointer = 1;
    for k_clusters = 1:length(vector(:,1))
        init_vec_len(1, pointer:pointer+length(vector(1,:))-1) = vector(k_clusters, :);
        pointer = pointer + length(vector(1,:));
    end
    serial_vec = init_vec_len;
end

function resulting_vec = deserialise_vector(vector, dim1, dim2)
    resulting_vec = zeros(dim1, dim2);
    vec_ptr = 0;
    for i = 1:dim1
        for j = 1:dim2
            vec_ptr = vec_ptr + 1;
            resulting_vec(i, j) = vector(vec_ptr);
        end
    end
end

