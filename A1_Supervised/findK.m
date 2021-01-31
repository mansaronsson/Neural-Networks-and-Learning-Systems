function k = findK(XTrain, LTrain, kMax)
%FINDK Finds the optimal k for the kNN algorithm
%   Detailed explanation goes here

prevAcc = 0;
for i=1:kMax
    
    % Classify training data
    LPredTrain = kNN(XTrain, i, XTrain, LTrain);
    
    % The confucionMatrix
    cM = calcConfusionMatrix(LPredTrain, LTrain);

    % The accuracy
    acc = calcAccuracy(cM);
    
    if(acc < prevAcc)
       k = i-1;
       return
    end
    
    prevAcc = acc;
end

end

