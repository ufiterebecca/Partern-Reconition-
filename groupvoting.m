function clab = groupvoting(tset, clsmx)
% Simple unanimity voting function
% 	tset - matrix containing test data; one row represents one sample
% 	clsmx - voting committee matrix
% Output:
%	clab - classification result

    % class processing
	labels = unique(clsmx(:, 1));
	reject = max(labels) + 1;

    % cast votes of classifiers
    votes = voting2(tset, clsmx);

    % count votes
    validvotes = sum(votes, 2) == 1;
    decisions = votes .*validvotes;

    % find the most voted class
    [mv clab] = max(decisions, [], 2);

    % if there is no unanimity, reject
    clab(mv~=1) = reject;
end
