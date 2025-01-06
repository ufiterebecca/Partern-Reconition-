function rds = reduce(ds, parts)
% Function reducing samples count of individual classes in ds
% ds - data set to be reduced (sample = row; in the first column labels)
% parts - row vector of reduction coefficients for individual classes
%	(1 means no reduction; 0 means no samples of given class to be left)
% rds - reduced data set

    labels = unique(ds(:,1));
    if rows(labels) ~= columns(parts)
        error("Class number does not agree with the coefficients number.");
    end

    if max(parts) > 1 || min(parts) < 0
        error("Invalid reduction coefficients.");
    end

    rds = [];
    % for each class
    for i = 1:length(labels)
        % select only one class samples from ds
        class_mask = ds(:,1) == labels(i);
        class_samples = ds(class_mask, :);
        
        % shuffle samples of this class with randperm
        n_samples = size(class_samples, 1);
        shuffled_indices = randperm(n_samples);
        shuffled_samples = class_samples(shuffled_indices, :);
        
        % select proper part of shuffled class and append it to rds
        samples_to_keep = round(n_samples * parts(i));
        if samples_to_keep > 0
            selected_samples = shuffled_samples(1:samples_to_keep, :);
            rds = [rds; selected_samples];
        end
    end
end