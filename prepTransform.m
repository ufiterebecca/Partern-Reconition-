function [mu trmx] = prepTransform(tvec, comp_count)
% Computes transformation matrix to PCA space
% tvec - training set (one row represents one sample)
% comp_count - count of principal components in the final space
% mu - mean value of the training set
% trmx - transformation matrix to comp_count-dimensional PCA space

	mu = mean(tvec);
	cmx = cov(tvec); 

	[evec eval] = eig(cmx);
	eval = diag(eval);

	[eval evid] = sort(eval, "descend");
	evec = evec(:, evid(1:end));

	trmx = evec(:, 1:comp_count);
end
