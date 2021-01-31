function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Classifies all samples of X after the k-closest nearest neighbours.
% If two clases has an equal amout of near neighbours, the mean distance of
% each class samples decides. If the mean distance is the same, the closest
% single sample dicedes.
%
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
% M = hypot(X(:,1)-XTrain(:,1).', X(:,2)-XTrain(:,2).');
M = pdist2(X, XTrain);

DistMat = zeros(NSamples, k);               % The rows contain the distances of the closest samples to that sample
IndMat = zeros(NSamples, k);                % The corresponding indices of DistMat

% Calculate the matrices
for i=1:NSamples
    
    [dist, ind] = sort(M(i,:));
    DistMat(i,:) = dist(1:k);
    IndMat(i,:) = ind(1:k);
    
end

% Classification
ClassMat = zeros(NSamples, k);  % The corresponding classes of DistMat
for i=1:k
    
    ClassMat(:,i) = LTrain(IndMat(:,i));
    
end

ClassFreqMat = zeros(NSamples, NClasses);   % The frequency of each class of ClassMat
for i=1:NClasses
    ClassFreqMat(:,i) = sum(ClassMat == i, 2);
end

% Correct classification and creation of LPred for all classes where one
% class dominates the other in frequenzy of near neighbours. We save this
% maximum frequency in MaxFreq to see if there are multiple classes that
% has the same number of close neighbours.
[MaxFreq, LPred] = max(ClassFreqMat,[],2);

% Specifies the number of "maximum classes". NMaxFreq contains 1 for all
% classes that are dominating.
NMaxFreq = zeros(NSamples, 1);
for i=1:NClasses
    NMaxFreq = NMaxFreq + (ClassFreqMat(:,i) == MaxFreq);
end

% Create a vector with all the indices where there is a conflict in class
% frequency
SpecCase = find(NMaxFreq > 1);
if (~isempty(SpecCase))
    for i=1:length(SpecCase)
    
    % Create a vector with class indices that has the same frequency
    SameFreqClass = find(ClassFreqMat(SpecCase(i),:) == MaxFreq(SpecCase(i)));
    
    minDist = inf;
    minInd = 0;
        % Loops through the number of classes of the same frequency
        for j=1:length(SameFreqClass)

            % Calculate the mean of the samples of each class
            classRowInd = ClassMat(SpecCase(i), :) == SameFreqClass(j);
            d = DistMat(SpecCase(i), classRowInd);
            m = mean(d);

            % Classifies as the class with lowest mean distance from target. If
            % the distance is the same, it classifies as the class with the
            % closest sample of those classes.
            if(m < minDist)
                minDist = m;
                minInd = SameFreqClass(j);

            elseif(m == minDist)

                classRowIndOld = ClassMat(:, SpecCase(i)) == minInd;
                dOld = DistMat(classRowIndOld, SpecCase(i));

                if(min(d) < min(dOld))
                    minInd = SameFreqClass(j);
                end

            end

        end

        % Changes the special cases of LPred
        LPred(SpecCase(i)) = minInd;
    end
end


