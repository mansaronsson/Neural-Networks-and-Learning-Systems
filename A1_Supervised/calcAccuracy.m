function [ acc ] = calcAccuracy( cM )
% CALCACCURACY Takes a confusion matrix amd calculates the accuracy

% **************** Our code ************************

acc = trace(cM) / sum(cM, 'all');

end

