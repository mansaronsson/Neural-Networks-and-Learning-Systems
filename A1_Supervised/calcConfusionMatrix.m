function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

cM = zeros(NClasses);

% ****************** Our code ***********************
for i=1:NClasses
    
    for j=1:NClasses
        
        if(i == j)
            % Creates the diagonal with the correct classifications
            cM(i,j) = sum(LPred(LPred == LTrue) == i);
        else
            
            cM(i,j) = sum(LPred(LTrue == i) == j);
            
        end
        
    end
    
end
