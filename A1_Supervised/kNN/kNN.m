function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% ******************** Our code *****************************
NSamples = size(X,1);

% Create a square matrix where distances between samples are stored in the
% corresponding row/column index
M = hypot(X(:,1)-XTrain(:,1).', X(:,2)-XTrain(:,2).');

DistMat = zeros(k, NSamples);               % The rows contain the distances of the closest samples to that sample
IndMat = zeros(k, NSamples);                % The corresponding indices of DistMat

% Calculate the matrices
for i=1:NSamples
    
    [dist, ind] = sort(M(:,i));
    DistMat(:,i) = dist(2:k+1); % 2:k+1 because the smallest distance is the distance to itself
    IndMat(:,i) = ind(2:k+1);
    
end

% Classification
ClassMat = zeros(k, NSamples);  % The corresponding classes of DistMat
for i=1:k
    ClassMat(i,:) = LTrain(IndMat(i,:),:);
end

ClassFreqMat = zeros(NClasses, NSamples);   % The frequency of each class of ClassMat
for i=1:NClasses
    ClassFreqMat(i,:) = sum(ClassMat == i);
end

% Correct classification and creation of LPred for all classes where one
% class dominates the other in frequenzy of near neighbours. We save this
% maximum frequency in MaxFreq to see if there are multiple classes that
% has the same number of close neighbours.
[MaxFreq, LPred] = max(ClassFreqMat,[],1);

% Specifies the number of "maximum classes". NMaxFreq contains 1 for all
% classes that are dominating.
NMaxFreq = zeros(1, NSamples);
for i=1:NClasses
    NMaxFreq = NMaxFreq + (ClassFreqMat(i,:) == MaxFreq);
end

% Create a vector with all the indices where there is a conflict in class
% frequency
SpecCase = find(NMaxFreq > 1);
for i=1:length(SpecCase)
    
    % Create a vector with class indices that has the same frequency
    SameFreqClass = find(ClassFreqMat(:,SpecCase(i)) == MaxFreq(SpecCase(i)));
    
    minDist = inf;
    minInd = 0;
    % Loops through the number of classes of the same frequency
    for j=1:length(SameFreqClass)

        % Calculate the mean of the samples of each class
        m = mean(DistMat(ClassMat(:, SpecCase(i)) == SameFreqClass(j), SpecCase(i)));
        
        % TODO: There is a weakness in the logic here. If the mean distances
        % between class samples are the same, it will be classified as the
        % "lowest" class index. Need to add more logic...
        if(m < minDist)
            minDist = m;
            minInd = SameFreqClass(j);
        end
        
    end
    
    % Changes the special cases of LPred
    LPred(SpecCase(i)) = minInd;
end

