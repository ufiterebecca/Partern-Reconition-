function pdf = pdf_parzen(pts, para)
% Approximates probability density function with Parzen window
% pts  - contains points for which pdf is computed (sample = row)
% para - structure containing parameters:
%	para.labels - class labels
%	para.samples - cell array containing class samples
%	para.parzenw - Parzen window width
% pdf - probability density matrix
%	row count = number of samples in pts
%	column count = number of classes

	% final result matrix
	pdf = rand(rows(pts), rows(para.samples));

	% YOUR CODE GOES HERE
		for clid=1:rows(para.labels)
		% you know number of samples in this class so you can allocate 
		% intermediate matrix (it contains columns f1 f2 ... fn from diagram in instruction)
			cltr = para.samples{clid};
			onedpdfs = zeros(rows(cltr), columns(cltr));
			% don't forget to adjust Parzen window width
			hn = para.parzenw / sqrt(rows(cltr));
			
			% for each sample in pts
			for ptid = 1:rows(pts)
				% for each feature
				for ftid = 1:columns(cltr)
					% fill proper column in onedpdfs with call to normpdf
					onedpdfs(:,ftid) = normpdf(pts(ptid, ftid), cltr(:, ftid), hn);
				end
				% aggregate onedpdfs into a scalar value
				% and store it in proper element of pdf
				pdf(ptid, clid) = mean(prod(onedpdfs,2));
			end
		end
		
end
