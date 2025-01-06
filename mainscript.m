% mainscript is rather short this time

% primary component count
comp_count = 40; 

[tvec tlab tstv tstl] = readSets(); 

% shift labels by one to use labels directly as indices
tlab += 1;
tstl += 1;

% look at the 100 digits in the training set
imsize = 28;
fim = zeros((imsize + 2) * 10 + 2);

for clid = 1:10
  rowid = (clid - 1) * (imsize + 2) + 3;
  clsamples = find(tlab == clid)(1:10);
  for spid = 1:10
	colid = (spid - 1) * (imsize + 2) + 3;
	im = 1-reshape(tvec(clsamples(spid),:), imsize, imsize)';
	fim(rowid:rowid+imsize-1, colid:colid+imsize-1) = im;
  end
end
imshow(fim)

% check number of samples in each class
labels = unique(tlab)';
[labels; sum(tlab == labels); sum(tstl == labels)]

% compute and perform PCA transformation
[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);


% To successfully prepare ensemble you have to implement perceptron function
% I would use 10 first zeros and 10 fisrt ones 
% and only 2 first primary components
% It'll allow printing of intermediate results in perceptron function

tenzeros = tvec(tlab == 1, 1:2)(1:10, :);
tenones = tvec(tlab == 2, 1:2)(1:10, :);
pclass = tenzeros;
nclass = tenones;
plot(tenzeros(:,1), tenzeros(:,2), "r*", tenones(:,1), tenones(:,2), "bs")
%
% YOUR CODE GOES HERE - testing of the perceptron function

% 1. Preparation of the function to compute separation plane parameters given a training set
% containing just two classes - perceptron. The easiest way to accomplish it is to use
% two-dimensional data sets , which can be visualized together with the separating plane.

% Use perceptron function to get separating plane
[sepplane, mispos, misneg] = perceptron(pclass, nclass);

% Plot the separating line
plot(tenzeros(:,1), tenzeros(:,2), "r*", tenones(:,1), tenones(:,2), "bs")
hold on
x_vals = [-1, 1];
y_vals = (-sepplane(2) * x_vals - sepplane(1)) / sepplane(3);
plot(x_vals, y_vals, 'k-', 'LineWidth', 2)

% Now experiment with the margin 
% It make sense to use "easy" (0 vs. 1) and "difficult" (4 vs. 9) cases.
%
%
% YOUR CODE GOES HERE - experiments with marging in the perceptron function

% 2. Checking the algorithm for multidimensional digits data. You should store individual one vs.
% one classifiers quality to put them in your report. Because you’ll have to check expansion of
% the features reduce dimensionality of data with PCA (40-60 primary components).

repeat = 100;

[tvec tlab tstv tstl] = readSets();

[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

% Easy case
pclass = tvec(tlab == 0, :);
nclass = tvec(tlab == 1, :);
total_pos_acc = 0;
total_neg_acc = 0;
for i = 1:repeat
    [sepplane, mispos, misneg] = perceptron(pclass, nclass);
    pos_acc = 1 - mispos / size(pclass, 1);
    total_pos_acc += pos_acc;
    neg_acc = 1 - misneg / size(nclass, 1);
    total_neg_acc += neg_acc;
end
disp("EASY CASE")
disp("Positive acc: "), disp(total_pos_acc / repeat)
disp("Negative acc: "), disp(total_neg_acc / repeat)


% Difficult case
pclass = tvec(tlab == 4, :);
nclass = tvec(tlab == 9, :);
total_pos_acc = 0;
total_neg_acc = 0;
for i = 1:repeat
    [sepplane, mispos, misneg] = perceptron(pclass, nclass);
    pos_acc = 1 - mispos / size(pclass, 1);
    total_pos_acc += pos_acc;
    neg_acc = 1 - misneg / size(nclass, 1);
    total_neg_acc += neg_acc;
end
disp("DIFFICULT CASE")
disp("Positive acc: "), disp(total_pos_acc / repeat)
disp("Negative acc: "), disp(total_neg_acc / repeat)

% 3. Canonical solution is 45 voting classifiers - one for each pair of digits - and making the final
% decision with unanimity voting (only digits with 9 votes are classified; if the number of
% votes is smaller classifier produces reject decision). You should report individual classifier
% error rates as well as quality of the ensemble. Quite interesting insight into ensemble
% operation can give you a confusion matrix.

%Shift labels by 1
tlab += 1;
tstl += 1;

disp("OVO OVO OVO OVO OVO OVO")
% training of the whole ensemble
[ovo, err] = trainOVOensemble(tvec, tlab, @perceptron);
err

% check your ensemble on train set
clab = unamvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = unamvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)


%4. Preparing one-versus-rest ensemble of classifiers. Here you’ll have to think about
%organizing the training (a function similar to trainOVOensemble). Even before, you
%should decide about representation of classifiers in the matrix; with code reuse in mind it
%would be wise to put in the negative class label in the second column. Such a value that will
%allow you – with minimal changes in code – to use voting and unamvoting functions.
%Alternative is of course to write separate functions for OVR ensemble.
%
% YOUR CODE GOES HERE

% Train and test the OVR ensemble

disp("OVR OVR OVR OVR OVR OVR")
[ovo, err] = trainOVRensemble(tvec, tlab, @perceptron);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

%Reduced Dataset OVR
disp("REDUCE 20 OVR OVR OVR OVR OVR OVR")
[ovo, err] = reduce_trainOVRensemble(tvec, tlab, @perceptron, 0.2);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

disp("REDUCE 80 OVR OVR OVR OVR OVR OVR")
[ovo, err] = reduce_trainOVRensemble(tvec, tlab, @perceptron, 0.8);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)



% expand features
% trainExp = expandFeatures(tvec);
% testExp = expandFeatures(tstv);











% Train and test the OVO ensemble on the expanded features
disp("EXPAND OVO OVO OVO OVO OVO OVO")

[tvec tlab tstv tstl] = readSets();

[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

% Expand features
tvec = expandFeatures(tvec);
tstv = expandFeatures(tstv);

%Shift labels by 1
tlab += 1;
tstl += 1;

% training whole ensemble
[ovo, err] = trainOVOensemble(tvec, tlab, @perceptron);
err

% ensemble check train set
clab = unamvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = unamvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)






% Train and test the OVR ensemble on the expanded features
disp("EXPAND OVR OVR OVR OVR OVR OVR")
% training whole ensemble
[ovo, err] = trainOVRensemble(tvec, tlab, @perceptron);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)



%Reduced Dataset OVR
disp("EXPAND REDUCE 20 OVR OVR OVR OVR OVR OVR")
[ovo, err] = reduce_trainOVRensemble(tvec, tlab, @perceptron, 0.2);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

disp("EXPAND REDUCE 80 OVR OVR OVR OVR OVR OVR")
[ovo, err] = reduce_trainOVRensemble(tvec, tlab, @perceptron, 0.8);
err

% ensemble check train set
clab = groupvoting(tvec, ovo);
cfmx = confMx(tlab, clab)
compErrors(cfmx)

% repeat on test set
clab = groupvoting(tstv, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)







% Think about improving your classifier further :)

disp("NEWVOTE")
[tvec tlab tstv tstl] = readSets();

[mu trmx] = prepTransform(tvec, comp_count);
tvec = pcaTransform(tvec, mu, trmx);
tstv = pcaTransform(tstv, mu, trmx);

% shift labels by one to use labels directly as indices
tlab += 1;
tstl += 1;

[ovo err] = trainOVOensemble(tvec, tlab, @perceptron);
[ovr err] = trainOVRensemble(tvec, tlab, @perceptron);

clab = customvoting(tstv, ovr, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

% % % Expand features
tvec = expandFeatures(tvec);
tstv = expandFeatures(tstv);

[ovo err] = trainOVOensemble(tvec, tlab, @perceptron);
[ovr err] = trainOVRensemble(tvec, tlab, @perceptron);

clab = customvoting(tstv, ovr, ovo);
cfmx = confMx(tstl, clab)
compErrors(cfmx)

% - 1 means 1 minus rest
