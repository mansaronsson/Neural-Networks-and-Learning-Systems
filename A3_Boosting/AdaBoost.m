%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 100; 

% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 2000;

% Number of weak classifiers
nbrWeakClassifiers = 100; 

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

%     figure(1);
%     colormap gray;
%     for k=1:25
%         subplot(5,5,k), imagesc(faces(:,:,10*k));
%         axis image;
%         axis off;
%     end
% 
%     figure(2);
%     colormap gray;
%     for k=1:25
%         subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
%         axis image;
%         axis off;
%     end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

%     figure(3);
%     colormap gray;
%     for k = 1:nbrHaarFeatures
%         subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
%         axis image;
%         axis off;
%     end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

D = ones(nbrTrainImages, 1) * 1/nbrTrainImages;  % weights
P = ones(nbrWeakClassifiers, 1);                 % polarities
T = zeros(nbrWeakClassifiers, 1);                % thresholds
F = zeros(nbrWeakClassifiers, 1);                % best features
C = zeros(nbrWeakClassifiers, nbrTrainImages);   % classifications
alpha = zeros(nbrWeakClassifiers, 1);

C_test = zeros(nbrWeakClassifiers, nbrTestImages);  % classifications of test data

for j=1:nbrWeakClassifiers 

    minE = inf;
    P_weak = 1; 
    
    for k=1:nbrHaarFeatures 
        for i=1:nbrTrainImages 

            T_weak = xTrain(k,i);
            C_weak = WeakClassifier(T_weak, P_weak, xTrain(k,:));
            E_weak = WeakClassifierError(C_weak, D, yTrain);

            % Change min error and change polarities
            if(E_weak > 0.5)
                E_weak = 1 - E_weak;
                P_weak = P_weak * (-1);
                C_weak = C_weak * (-1);
            end

            % Saves information of best feature
            if(E_weak < minE)
                minE = E_weak;
                C(j,:) = C_weak;
                P(j) = P_weak;
                T(j) = T_weak;
                F(j) = k;
            end
        end
    end

    % Update and renormalize weights
    eps = 1e-5;     % Small value used to stablize the alpha calculation
    alpha(j) = (1/2) * log((1-minE)/(minE+eps));
    D = D.*(exp(-alpha(j)*yTrain.*C(j,:)))';
    D = D./sum(D);
    
    % Weak classifier for test data
    C_test(j,:) = WeakClassifier(T(j), P(j), xTest(F(j),:));
end

%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

acc_train = zeros(nbrWeakClassifiers, 1);
acc_test = zeros(nbrWeakClassifiers, 1);

C_strong_train = zeros(nbrWeakClassifiers, nbrTrainImages);
C_strong_test = zeros(nbrWeakClassifiers, nbrTestImages);

for i=1:nbrWeakClassifiers
    
    C_strong_train(i,:) = sign(sum(alpha(1:i).*C(1:i,:)));
    acc_train(i) = sum(C_strong_train(i,:) == yTrain) / nbrTrainImages;
    
    C_strong_test(i,:) = sign(sum(alpha(1:i).*C_test(1:i,:)));
    acc_test(i) = sum(C_strong_test(i,:) == yTest) / nbrTestImages;
end

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

[val, ind] = max(acc_test);
figure, plot(linspace(1, nbrWeakClassifiers, nbrWeakClassifiers), acc_train)
hold on
plot(linspace(1, nbrWeakClassifiers, nbrWeakClassifiers), acc_test)
xlabel("Number of weak classifiers")
ylabel("Accuracy")
title(sprintf("Accuracy of strong classifier. %d training data (%d faces). %d test data (%d faces). %d Haar-features.",...
    nbrTrainImages, nbrTrainImages/2, nbrTestImages, 4916-(nbrTrainImages/2), nbrHaarFeatures))
grid on
plot(ind, val, 'ob')
legend("Train", "Test", sprintf("Acc: %.1f%%, nbrWeak: %d", val*100, ind), 'Location', 'best')
hold off


%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

misC = C_strong_test(ind, :) ~= yTest;
misCImg = testImages(:,:,misC);

figure
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(misCImg(:,:,10*k));
    axis image;
    axis off;
end
sgtitle("Misclassified faces and non-faces");

%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

% Picks the Haar-feature used most frequently by weak calssifiers
[freqHist, indHist] = histcounts(F, unique(F));
[~, idx] = sort(freqHist, 'descend');
bestHaar = haarFeatureMasks(:,:,indHist(idx(1:25)));

figure
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(bestHaar(:,:,k),[-1 2]);
    axis image;
    axis off;
end
sgtitle("Most frequently used Haar-features");