% tiny data file to verify pdf functions
load pdf_test.txt
size(pdf_test)

% how many classes are there?
labels = unique(pdf_test(:,1))

% how many samples are in each class?
[labels'; sum(pdf_test(:,1) == labels')]
		  % ^^^ how does this expression work?

% what's the layout of the samples?
% will it work?
plot2features(pdf_test, 2, 3)

% check if statistics package is present
normpdf(0, 0, 1)
% it can work directly - nothing to be done 
% it can be installed but not loaded - pkg load statistics
% it can be not installed at all - use __normpdf function provided instead


pdfindep_para = para_indep(pdf_test)
% para_indep indep is already implemented; it should give:

% pdfindep_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%       0.21772   0.19172
%       0.19087   0.27179

% now you have to implement pdf_indep and then verify it

pi_pdf = pdf_indep(pdf_test([2 7 12 17],2:end), pdfindep_para)

%pi_pdf =
%  1.4700e+000  4.5476e-007
%  3.4621e+000  4.9711e-005
%  6.7800e-011  2.7920e-001
%  5.6610e-008  1.8097e+000

% multivariate normal distribution - parameters ...

pdfmulti_para = para_multi(pdf_test)

%pdfmulti_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    mu =
%       0.7970000   0.8200000
%      -0.0090000   0.0270000
%    sig =
%    ans(:,:,1) =
%       0.047401   0.018222
%       0.018222   0.036756
%    ans(:,:,2) =
%       0.036432  -0.033186
%      -0.033186   0.073868  

% ... and probability density function (use mvnpdf in pdf_multi)

pm_pdf = pdf_multi(pdf_test([2 7 12 17],2:end), pdfmulti_para)

%pm_pdf =w
%  7.9450e-001  6.5308e-017
%  3.9535e+000  3.8239e-013
%  1.6357e-009  8.6220e-001
%  4.5833e-006  2.8928e+000

% parameters for Parzen window approximation 
pdfparzen_para = para_parzen(pdf_test, 0.5)
									 % ^^^ window width

%pdfparzen_para =
%  scalar structure containing the fields:
%    labels =
%       1
%       2
%    samples =
%    {
%      [1,1] =
%         1.10000   0.95000
%         0.98000   0.61000
% .....
%         0.69000   0.93000
%         0.79000   1.01000
%      [2,1] =
%        -0.010000   0.380000
%         0.250000  -0.440000
% .....
%        -0.110000   0.030000
%         0.120000  -0.090000
%    }
%    parzenw =  0.50000

% now you have to implement pdf_parzen and then verify it

pp_pdf = pdf_parzen(pdf_test([2 7 12 17],2:end), pdfparzen_para)

%pp_pdf =
%  9.7779e-001  6.1499e-008
%  2.1351e+000  4.2542e-006
%  9.4059e-010  9.8823e-001
%  2.0439e-006  1.9815e+000


% now you can start work with cards!
[train test] = load_cardsuits_data();

% Our first look at the data
size(train)
size(test)
labels = unique(train(:,1))
unique(test(:,1))
[labels'; sum(train(:,1) == labels')]

% the first task after loading the data is checking
% training set for outliers; to this end we usually compute 
% simple statistics: mean, median, std, 
% and/or plot histogram of individual feature: hist
% and/or plot two features at a time: plot2features

[mean(train); median(train)]
hist(train(:,1))
plot2features(train, 4, 6)
					%^^^^ just an example

% to identify outliers you can use two output argument versions 
% of min and max functions


% because the minimum or maximum values can be determined always,
% it's worth to look at neighbors of the suspected sample in the training set

% let's assume that sample 41 is suspected
% it seems that these three rows are very similar to each other...
% that's because 41 is evidently not an outlier index

% Plot histograms for each feature
for feat = 2:size(train, 2)
    % Class-specific histograms (separate plots to avoid alpha issues)
    classes = unique(train(:,1));
    colors = 'rgbcmykw';
    
    figure;
    for c = 1:length(classes)
        subplot(2,4,c);  % 2x4 grid of subplots for 8 classes
        class_data = train(train(:,1) == classes(c), feat);
        hist(class_data, 20, colors(c));
        title(sprintf('Class %d', classes(c)));
        xlabel(sprintf('Feature %d', feat)); 
        ylabel('Frequency');
        grid on;
    end
end

%find outliers in the training set
function cleaned_train = find_class_outliers(train)
    classes = unique(train(:,1));
    MIN_THRESHOLD = 1e-6;
    NEIGHBORS_TO_SHOW = 2;
    Z_SCORE_THRESHOLD = 3.5;
    MEAN_MEDIAN_THRESHOLD = 1.5;
    
    % Keep track of rows to remove
    rows_to_remove = [];
    
    for c = classes'
        class_mask = train(:,1) == c;
        class_samples = train(class_mask, :);
        class_indices = find(class_mask);
        
        % For each feature
        for f = 2:size(train, 2)
            feature_values = class_samples(:, f);
            
            if all(abs(feature_values) < MIN_THRESHOLD)
                continue;
            end
            
            % Calculate statistics
            mean_val = mean(feature_values);
            median_val = median(feature_values);
            std_val = std(feature_values);
            
            % Print class and feature statistics
            fprintf('\nClass %d, Feature %d Statistics:\n', c, f);
            fprintf('Mean: %.6f\n', mean_val);
            fprintf('Median: %.6f\n', median_val);
            fprintf('Std: %.6f\n', std_val);
            
            % Check mean-median difference
            mean_median_diff = abs(mean_val - median_val) / std_val;
            fprintf('Mean-Median difference in std units: %.2f\n', mean_median_diff);
            
            if mean_median_diff > MEAN_MEDIAN_THRESHOLD
                fprintf('Large mean-median difference detected\n');
                
                % Find values that are pulling mean away from median
                for i = 1:length(feature_values)
                    val = feature_values(i);
                    if abs(val - median_val) > MEAN_MEDIAN_THRESHOLD * std_val
                        fprintf('Potential outlier at index %d: %.6f (%.2f std from median)\n', ...
                                class_indices(i), val, abs(val - median_val)/std_val);
                        rows_to_remove = [rows_to_remove class_indices(i)];
                    end
                end
            end
            
            % Check z-scores
            z_scores = abs(feature_values - mean_val) / std_val;
            outlier_indices = find(z_scores > Z_SCORE_THRESHOLD);
            
            if ~isempty(outlier_indices)
                fprintf('Z-score outliers found:\n');
                for idx = outlier_indices'
                    fprintf('Index %d: value %.6f, z-score %.2f\n', ...
                           class_indices(idx), feature_values(idx), z_scores(idx));
                    rows_to_remove = [rows_to_remove class_indices(idx)];
                end
            end
            
            % Print quartile information
            q1 = quantile(feature_values, 0.25);
            q3 = quantile(feature_values, 0.75);
            iqr = q3 - q1;
            fprintf('Q1: %.6f, Q3: %.6f, IQR: %.6f\n', q1, q3, iqr);
        end
    end
    
    % Remove outliers
    if ~isempty(rows_to_remove)
        rows_to_remove = unique(rows_to_remove); % Remove duplicates
        fprintf('\nRemoving %d outliers from rows: ', length(rows_to_remove));
        fprintf('%d ', rows_to_remove);
        fprintf('\n');
        fprintf('Dataset size before: %d\n', size(train, 1));
        cleaned_train = train;
        cleaned_train(rows_to_remove, :) = [];
        fprintf('Dataset size after: %d\n', size(cleaned_train, 1));
        
        % Print impact on class distribution
        fprintf('\nClass distribution after removal:\n');
        remaining_classes = unique(cleaned_train(:,1));
        for c = remaining_classes'
            fprintf('Class %d: %d samples\n', c, sum(cleaned_train(:,1) == c));
        end
    else
        fprintf('\nNo outliers found.\n');
        cleaned_train = train;
    end
end 

% Use the function
train = find_class_outliers(train);
           

% the procedure of searching for and removing outliers must be repeated 
% until no outliers exist in the training set

% after removing outliers, you can deal with the selection of TWO features for classification
% in this case, it is enough to look at the graphs of two features and choose the ones that
% give relatively well separated classes

function plot_feature_combinations(train)
    % Plot scatter plots for all feature combinations, colored by class
    %
    % Parameters:
    %   train - matrix where first column is class labels and rest are features
    
    features = 2:size(train, 2);
    classes = unique(train(:,1));
    colors = 'rgbcmykw';  % 8 different colors for 8 classes
    markers = 'ox+*sdv^';  % 8 different markers for 8 classes
    
    for i = features
        for j = i+1:size(train, 2)
            figure;
            hold on;
            
            % Plot each class with different color and marker
            for c = 1:length(classes)
                class_mask = train(:,1) == classes(c);
                scatter(train(class_mask,i), train(class_mask,j), 50, ...
                        [colors(c), markers(c)], 'filled');
            end
            
            xlabel(['Feature ', num2str(i)]);
            ylabel(['Feature ', num2str(j)]);
            title(sprintf('Features %d vs %d', i, j));
            legend(cellstr(num2str(classes)), 'Location', 'best');
            grid on;
            hold off;
        end
    end
end

plot_feature_combinations(train);

% % after selecting features reduce both sets:
% train = train(:, [1 4 6]);
% test = test(:, [1 4 6]);
% 				% ^^^ please, don't use these features!
	
% Select features 2 and 4 for classification
train = train(:, [1 2 4]);  % [class_label feature2 feature4]
test = test(:, [1 2 4]);

% Now proceed with classification using these features
% % POINT 2

pdfindep_para = para_indep(train);
pdfmulti_para = para_multi(train);
pdfparzen_para = para_parzen(train, 0.0005); 
% this window width should be included in your report!

priors = ones(1,8) * 0.125;  % Equal priors for all 8 classes

% Point 2 results with equal priors
base_ercf = zeros(1,3);
base_ercf(1) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para, priors) != test(:,1));
base_ercf(2) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para, priors) != test(:,1));
base_ercf(3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para, priors) != test(:,1));
base_ercf

% % before moving to point 3 it would be wise to
% % implement and test reduce function
% % let's start with small test set - just 2 classes

rdlab = unique(pdf_test(:,1));
reduced = reduce(pdf_test, [0.8 0.4]);
[rdlab'; sum(reduced(:,1) == rdlab')]

% % ans =
% %     1    2
% %     8    4


% % POINT 3

% % In the next point, the reduce function will be useful, which reduces the number of samples 
% % in the individual classes (in this case, the reduction will be the same in all classes - 
% % OF THE TRAINING SET)
% % Because reduce has to draw samples randomly, the experiment should be repeated 5 times
% % In the report, please provide only the mean value and the standard deviation 
% % of the error coefficient

parts = [0.1 0.25 0.5];
rep_cnt = 5; % at least
class_count = length(unique(train(:,1)));
results = zeros(length(parts), rep_cnt, 3); % [parts, repetitions, classifiers]

% For each reduction part
for p = 1:length(parts)
    % For each repetition
    for r = 1:rep_cnt
        % Reduce training set with same coefficient for all classes
        reduced_train = reduce(train, parts(p) * ones(1, class_count));
        
        % Train classifiers on reduced set
        pdfindep_para = para_indep(reduced_train);
        pdfmulti_para = para_multi(reduced_train);
        pdfparzen_para = para_parzen(reduced_train, 0.001);
        
        % Test classifiers
        results(p,r,1) = mean(bayescls(test(:,2:end), @pdf_indep, pdfindep_para) != test(:,1));
        results(p,r,2) = mean(bayescls(test(:,2:end), @pdf_multi, pdfmulti_para) != test(:,1));
        results(p,r,3) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
    end
end

% Calculate mean and std for each combination
mean_results = squeeze(mean(results, 2));  % Average across repetitions
std_results = squeeze(std(results, 0, 2));  % Std across repetitions

% Display results
fprintf('\nPoint 3 Results:\n');
fprintf('Reduction\tIndependent\tMultivariate\tParzen\n');
for p = 1:length(parts)
    fprintf('%.2f\t\t%.4f±%.4f\t%.4f±%.4f\t%.4f±%.4f\n', ...
        parts(p), ...
        mean_results(p,1), std_results(p,1), ...
        mean_results(p,2), std_results(p,2), ...
        mean_results(p,3), std_results(p,3));
end
% Point 3 Results:
% Reduction	Independent	Multivariate	Parzen
% 0.10		0.0360±0.0095	0.0104±0.0020	0.0971±0.0149
% 0.25		0.0310±0.0078	0.0079±0.0028	0.0533±0.0109
% 0.50		0.0338±0.0064	0.0059±0.0011	0.0394±0.0037


% % note that for given experiment you should reduce all classes in the training
% % set with the same reduction coefficient; assuming that class_count is the 
% % number of different classes in the training set you can take 3/4 random samples
% % of each class with:
% % 	reduced_train = reduce(train, 0.75 * ones(1, class_count))
% % 

% % POINT 4
% % Point 4 concerns only Parzen window classifier (on the full training set)

parzen_widths = [0.0001, 0.0005, 0.001, 0.005, 0.01];
parzen_res = zeros(1, columns(parzen_widths));

% Test each width
for w = 1:length(parzen_widths)
    pdfparzen_para = para_parzen(train, parzen_widths(w));
    parzen_res(w) = mean(bayescls(test(:,2:end), @pdf_parzen, pdfparzen_para) != test(:,1));
end

% Plot results
figure;
semilogx(parzen_widths, parzen_res, 'b-o');
xlabel('Window Width');
ylabel('Error Rate');
title('Parzen Window Classification Error vs Window Width');
grid on;

% % POINT 5
% % In point 5 you should reduce TEST SET
% %
 


% New reduction pattern based on error rates and feature characteristics
targeted_parts = zeros(1,8);

% Classes with errors get more samples
targeted_parts(1) = 1.0;  % Class 1: 0.44% error
targeted_parts(6) = 1.0;  % Class 6: 1.75% error

% Classes with perfect classification but high feature variance
targeted_parts(3) = 0.75;  % High variance in Feature 2
targeted_parts(5) = 0.75;  % Moderate variance

% Classes with perfect classification and stable features
targeted_parts([2 4 7 8]) = 0.5;  % Most stable classes

% Calculate corresponding prior probabilities
total_parts = sum(targeted_parts);
targeted_apriori = targeted_parts / total_parts;

fprintf('\nComparing All Approaches:\n');

% Original settings
parts = [1.0 0.5 0.5 1.0 1.0 0.5 0.5 1.0];
original_apriori = [0.165 0.085 0.085 0.165 0.165 0.085 0.085 0.165];

% Test all three approaches
% 1. Without priors
reduced_test_original = reduce(test, parts);
no_prior_results = zeros(1, 3);
no_prior_results(1) = mean(bayescls(reduced_test_original(:,2:end), @pdf_indep, pdfindep_para) != reduced_test_original(:,1));
no_prior_results(2) = mean(bayescls(reduced_test_original(:,2:end), @pdf_multi, pdfmulti_para) != reduced_test_original(:,1));
no_prior_results(3) = mean(bayescls(reduced_test_original(:,2:end), @pdf_parzen, pdfparzen_para) != reduced_test_original(:,1));

% 2. Original reduction with original priors
original_results = zeros(1, 3);
original_results(1) = mean(bayescls(reduced_test_original(:,2:end), @pdf_indep, pdfindep_para, original_apriori) != reduced_test_original(:,1));
original_results(2) = mean(bayescls(reduced_test_original(:,2:end), @pdf_multi, pdfmulti_para, original_apriori) != reduced_test_original(:,1));
original_results(3) = mean(bayescls(reduced_test_original(:,2:end), @pdf_parzen, pdfparzen_para, original_apriori) != reduced_test_original(:,1));

% 3. Targeted reduction with calculated priors
reduced_test_targeted = reduce(test, targeted_parts);
targeted_results = zeros(1, 3);
targeted_results(1) = mean(bayescls(reduced_test_targeted(:,2:end), @pdf_indep, pdfindep_para, targeted_apriori) != reduced_test_targeted(:,1));
targeted_results(2) = mean(bayescls(reduced_test_targeted(:,2:end), @pdf_multi, pdfmulti_para, targeted_apriori) != reduced_test_targeted(:,1));
targeted_results(3) = mean(bayescls(reduced_test_targeted(:,2:end), @pdf_parzen, pdfparzen_para, targeted_apriori) != reduced_test_targeted(:,1));

% Display comprehensive results
fprintf('\n1. Without Prior Probabilities:\n');
fprintf('Parts: '); fprintf('%.1f ', parts); fprintf('\n');
fprintf('Results:\n');
fprintf('Independent: %.4f\n', no_prior_results(1));
fprintf('Multivariate: %.4f\n', no_prior_results(2));
fprintf('Parzen: %.4f\n', no_prior_results(3));

fprintf('\n2. Original Approach (with original priors):\n');
fprintf('Parts: '); fprintf('%.1f ', parts); fprintf('\n');
fprintf('Priors: '); fprintf('%.3f ', original_apriori); fprintf('\n');
fprintf('Results:\n');
fprintf('Independent: %.4f\n', original_results(1));
fprintf('Multivariate: %.4f\n', original_results(2));
fprintf('Parzen: %.4f\n', original_results(3));

fprintf('\n3. Targeted Approach:\n');
fprintf('Parts: '); fprintf('%.1f ', targeted_parts); fprintf('\n');
fprintf('Priors: '); fprintf('%.3f ', targeted_apriori); fprintf('\n');
fprintf('Results:\n');
fprintf('Independent: %.4f\n', targeted_results(1));
fprintf('Multivariate: %.4f\n', targeted_results(2));
fprintf('Parzen: %.4f\n', targeted_results(3));

% Per-class analysis for all approaches
fprintf('\nPer-class error rates comparison:\n');
fprintf('Class\tNo Prior\tOriginal\tTargeted\n');
for c = 1:8
    % No prior
    class_mask_orig = reduced_test_original(:,1) == c;
    error_no_prior = mean(bayescls(reduced_test_original(class_mask_orig,2:end), @pdf_multi, pdfmulti_para) != reduced_test_original(class_mask_orig,1));
    
    % Original prior
    error_original = mean(bayescls(reduced_test_original(class_mask_orig,2:end), @pdf_multi, pdfmulti_para, original_apriori) != reduced_test_original(class_mask_orig,1));
    
    % Targeted prior
    class_mask_target = reduced_test_targeted(:,1) == c;
    error_targeted = mean(bayescls(reduced_test_targeted(class_mask_target,2:end), @pdf_multi, pdfmulti_para, targeted_apriori) != reduced_test_targeted(class_mask_target,1));
    
    fprintf('%d\t%.4f\t%.4f\t%.4f\n', c, error_no_prior, error_original, error_targeted);
end
% % POINT 6
% % In point 6 we should consider data normalization

% std(train(:,2:end))

% % Should we normalize?
% % If YES remember to normalize BOTH training and testing sets

% % YOUR CODE GOES HERE 
% %
feature_std = std(train(:,2:end));
fprintf('\nFeature standard deviations: %.4f %.4f\n', feature_std);

% If standard deviations are very different, normalize
if max(feature_std)/min(feature_std) > 10
    fprintf('Features have very different scales. Normalizing...\n');
    
    % Calculate normalization parameters from training set
    train_mean = mean(train(:,2:end));
    train_std = std(train(:,2:end));
    
    % Normalize training and test sets
    normalized_train = train;
    normalized_test = test;
    normalized_train(:,2:end) = (train(:,2:end) - train_mean) ./ train_std;
    normalized_test(:,2:end) = (test(:,2:end) - train_mean) ./ train_std;
    
    % Test with normalized data
    pdfindep_para = para_indep(normalized_train);
    pdfmulti_para = para_multi(normalized_train);
    pdfparzen_para = para_parzen(normalized_train, 0.0005);
    
    norm_results = zeros(1, 3);
    norm_results(1) = mean(bayescls(normalized_test(:,2:end), @pdf_indep, pdfindep_para) != normalized_test(:,1));
    norm_results(2) = mean(bayescls(normalized_test(:,2:end), @pdf_multi, pdfmulti_para) != normalized_test(:,1));
    norm_results(3) = mean(bayescls(normalized_test(:,2:end), @pdf_parzen, pdfparzen_para) != normalized_test(:,1));
    
    fprintf('\nResults before normalization: %.4f %.4f %.4f\n', base_ercf);
    fprintf('Results after normalization: %.4f %.4f %.4f\n', norm_results);
else
    fprintf('Feature scales are similar. Normalization not needed.\n');
end

% Test 1-NN classifier
% Note: cls1nn processes one test sample at a time, so we need to loop
predictions = zeros(size(test, 1), 1);
for i = 1:size(test, 1)
    predictions(i) = cls1nn(test(i,2:end), train);
end

nn_error = mean(predictions != test(:,1));
fprintf('1-NN Classifier Error: %.4f\n', nn_error);









