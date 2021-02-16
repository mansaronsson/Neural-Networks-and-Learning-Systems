%This script will help you test out your kNN code
%Select which data to use:
% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

% You can plot and study dataset 1 to 3 by running:
plotCase(X,D)

% Select a subset of the training samples
% The algorithm will test numBins-numTestBins different values of k
% starting at k=1
numBins = 20;                   % Number of bins you want to devide your data into
numTestBins = 5;                % Number of bins that should be used as test data
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

% Note: XBins, DBins, LBins will be cell arrays, to extract a single bin from them use e.g.
% XBin1 = XBins{1};
%
% Or use the combineBins helper function to combine several bins into one matrix (good for cross validataion)
% XBinComb = combineBins(XBins, [1,2,3]);

% Add your own code to setup data for training and test here

XTest  = combineBins(XBins, numBins-numTestBins+1:numBins);
LTest  = combineBins(LBins, numBins-numTestBins+1:numBins);

acc = zeros(numBins-numTestBins, 1);

for i=1:numBins-numTestBins % Starts at 2 because index NTestBins is used as test data
    trainingRange = 1:numBins-numTestBins;
    trainingRange(i) = [];
    
    XTrain = combineBins(XBins, trainingRange);
    LTrain = combineBins(LBins, trainingRange);
    XVal = XBins{i};
    LVal = LBins{i};

    % Classify training data
    LPredTrain = kNN(XTrain, i, XTrain, LTrain);
    % Classify test data
    LPredVal  = kNN(XVal , i, XTrain, LTrain);

    % The confucionMatrix
    cM = calcConfusionMatrix(LPredVal, LVal);

    % The accuracy
    accuracy = calcAccuracy(cM);
    
    % Saves the training data the yielded the best result
    if(accuracy > max(acc))
        bestXTrain = XTrain;
        bestLTrain = LTrain;
        bestLPredTrain = LPredTrain;
    end
    
    acc(i) = accuracy;

end

x = 1:numBins-numTestBins;
figure, plot(x, acc)
title('Accuracy during training');
xlabel('k');
ylabel('Accuracy');
[~,k] = max(acc)

% Reassign to the training data the yielded the best tesult
XTrain = bestXTrain;
LTrain = bestLTrain;
LPredTrain = bestLPredTrain;

% Accuracy on test data
LPredTest  = kNN(XTest , k, XTrain, LTrain);
cM = calcConfusionMatrix(LPredTest, LTest);
accuracy = calcAccuracy(cM)

% Plot classifications
% Note: You should not have to modify this code
if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredTrain, XTest, LTest, LPredTest, 'kNN', [], k);
else
    plotResultsOCR(XTest, LTest, LPredTest)
end
