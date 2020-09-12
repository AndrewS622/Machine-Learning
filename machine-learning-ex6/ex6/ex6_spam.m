%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% ==================== Part 1: Email Preprocessing ====================
%  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
%  to convert each email into a vector of features. In this part, you will
%  implement the preprocessing steps for each email. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given email.

fprintf('\nPreprocessing sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);

% Print Stats
fprintf('Word Indices: \n');
fprintf(' %d', word_indices);
fprintf('\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Feature Extraction ====================
%  Now, you will convert each email into a vector of features in R^n. 
%  You should complete the code in emailFeatures.m to produce a feature
%  vector for a given email.

fprintf('\nExtracting features from sample email (emailSample1.txt)\n');

% Extract Features
file_contents = readFile('emailSample1.txt');
word_indices  = processEmail(file_contents);
features      = emailFeatures(word_indices);

% Print Stats
fprintf('Length of feature vector: %d\n', length(features));
fprintf('Number of non-zero entries: %d\n', sum(features > 0));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.

% Load the Spam Email dataset
% You will have X, y in your environment
load('spamTrain.mat');

fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take 1 to 2 minutes) ...\n')

C = 0.1;
model = svmTrain(X, y, C, @linearKernel);

p = svmPredict(model, X);

fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);

%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat

% Load the test dataset
% You will have Xtest, ytest in your environment
load('spamTest.mat');

fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')

p = svmPredict(model, Xtest);

fprintf('Test Accuracy: %f\n', mean(double(p == ytest)) * 100);
pause;


%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
%

% Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!
filenames = {'spamSample1.txt', 'emailSample1.txt', 'emailSample2.txt', ... 
    'spamSample2.txt'};

for i = 1:length(filenames)
    filename = filenames{i};
    
    % Read and predict
    file_contents = readFile(filename);
    word_indices  = processEmail(file_contents);
    x             = emailFeatures(word_indices);
    p = svmPredict(model, x);

    fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
    fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
end

%% =================== Part 7: Building Your Own Dataset =====================
% The files for this analysis can be downloaded from the SpamAssassin
% Public Corpus and are separated into 5 groups of files: two spam file
% sets, two easy file sets, and one hard file set. The files are .gzip.bz2
% files and must be extracted in the \Examples folder prior to analysis.
% This example will go through the entirety of SVM classification analysis,
% from importing the initial data and extracting a vocabulary list, to
% constructing feature vectors, creating training, validation, and test
% sets, optimizing hyperparameters and model formulations, and final test
% set accuracy quantification, as well as error analysis

% load filenames for all downloaded files
easy_ham_path = '\Examples\easy_ham';
cd([pwd, easy_ham_path]);
easy_ham = ls;
cd ../..

easy_ham_2_path = '\Examples\easy_ham_2';
cd([pwd, easy_ham_2_path]);
easy_ham_2 = ls;
cd ../..

hard_ham_path = '\Examples\hard_ham';
cd([pwd, hard_ham_path]);
hard_ham = ls;
cd ../..

spam_path = '\Examples\spam';
cd([pwd, spam_path]);
spam = ls;
cd ../..

spam_2_path = '\Examples\spam_2';
cd([pwd, spam_2_path]);
spam_2 = ls;
cd ../..


% use processEmails to extract all words from all emails to find the most
% common to be used as features. Sorting occurs after each extraction since
% this will place the most common words at the top of the cell array, which
% should decrease the time spent searching for each new word
words = {};
words = processEmails(easy_ham_path, easy_ham, words);
words = sortrows(words, 2, 'descend');
words = processEmails(easy_ham_2_path, easy_ham_2, words);
words = sortrows(words, 2, 'descend');
words = processEmails(hard_ham_path, hard_ham, words);
words = sortrows(words, 2, 'descend');
words = processEmails(spam_path, spam, words);
words = sortrows(words, 2, 'descend');
words = processEmails(spam_2_path, spam_2, words);
words = sortrows(words, 2, 'descend');

% select words with highest frequencies of occurrence across all training
% sets as feature list
word_freq = cell2mat(words(:,2));
num_words = sum(word_freq > 100);
word_cell = {};
for i = 1:num_words
    word_cell{i,1} = i;
    word_cell{i,2} = words{i,1};
end

% save processed and filtered feature list as vocab list txt file
fid = fopen('vocab2.txt','w');
fprintf(fid, '%s\n', word_cell{:,2});
fclose(fid);

% use processEmail function to index list for each email
% note: update getVocabList() function to read in vocab2.txt, update n to 
% 2594 and comment out the first fscanf for each line since this file does
% not have integers at the start of the line. Also change n in
% emailFeatures to convert each index list into a feature vector
X = [];
for i = 3:(length(easy_ham)-1)
    filename = [pwd, easy_ham_path, '\', easy_ham(i,:)];
    contents = readFile(filename);
    word_indices = processEmail(contents);
    X(end + 1,:) = emailFeatures(word_indices);
end

% repeat for all email folders
for i = 3:(length(easy_ham_2)-1)
    filename = [pwd, easy_ham_2_path, '\', easy_ham_2(i,:)];
    contents = readFile(filename);
    word_indices = processEmail(contents);
    X(end + 1,:) = emailFeatures(word_indices);
end
for i = 3:(length(hard_ham)-1)
    filename = [pwd, hard_ham_path, '\', hard_ham(i,:)];
    contents = readFile(filename);
    word_indices = processEmail(contents);
    X(end + 1,:) = emailFeatures(word_indices);
end
for i = 4:(length(spam)-1)
    filename = [pwd, spam_path, '\', spam(i,:)];
    contents = readFile(filename);
    word_indices = processEmail(contents);
    X(end + 1,:) = emailFeatures(word_indices);
end
for i = 3:(length(spam_2)-1)
    filename = [pwd, spam_2_path, '\', spam_2(i,:)];
    contents = readFile(filename);
    word_indices = processEmail(contents);
    X(end + 1,:) = emailFeatures(word_indices);
end

% construct output vector; note: -9 in y1 construction is used to account
% for the . and .. entries at the start of each folder filenames and the
% cmds at the end; an extra -1 is included in the -7 for y2 since the first
% true file in the spam folder does not contain an email
y1 = zeros(length(easy_ham) + length(easy_ham_2) + length(hard_ham) - 9, 1);
y2 = ones(length(spam) + length(spam_2) - 7, 1);
y = [y1;y2];

% construct training, validation, and test sets
p = randperm(length(y));
train = ceil(0.6 * length(y));
val = ceil(0.5*(length(y) - train));
test = length(y) - train - val;
Xtrain = X(p(1:train),:);
ytrain = y(p(1:train));
Xval = X(p(train + (1:val)),:);
yval = y(p(train + (1:val)));
Xtest = X(p(train + val + (1:test)),:);
ytest = y(p(train + val + (1:test)));

% look at distributions of sets
fprintf('Fraction of spam samples in total dataset.\n');
disp(mean(y));
fprintf('Fraction of spam samples in training dataset.\n');
disp(mean(ytrain));
fprintf('Fraction of spam samples in validation dataset.\n');
disp(mean(yval));
fprintf('Fraction of spam samples in test dataset.\n');
disp(mean(ytest));

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('Distribution of features in total dataset.\n');
disp(mean(X(:,1:10)));
fprintf('Distribution of features in training dataset.\n');
disp(mean(Xtrain(:,1:10)));
fprintf('Distribution of features in validation dataset.\n');
disp(mean(Xval(:,1:10)));
fprintf('Distribution of features in test dataset.\n');
disp(mean(Xtest(:,1:10)));

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% train models with linear kernels for different values of C
fprintf('\nTraining Linear SVM (Spam Classification)\n')
fprintf('(this may take several minutes) ...\n')

Cs = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
accuracy = zeros(length(Cs), 1);
accuracy_RBF = zeros(length(Cs));
for i = 1:length(Cs)
    C = Cs(i);
    model = svmTrain(Xtrain, ytrain, C, @linearKernel);

    pred = svmPredict(model, Xval);
    accuracy(i) = mean(double(pred == yval)) * 100;
end
plot(Cs, accuracy);
set(gca, 'XScale', 'log')
xlabel('Value of parameter C');
ylabel('Classification Accuracy');
title('Linear Kernel SVM');

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

fprintf('\nTraining Gaussian SVM (Spam Classification)\n')
fprintf('(this may take several minutes) ...\n')
for i = 1:length(Cs)
    for j = 1:length(Cs)
        C = Cs(i);
        sigma = Cs(j);
        model = svmTrain(Xtrain, ytrain, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        
        pred = svmPredict(model, Xval);
        accuracy_RBF(i, j) = mean(double(pred == yval)) * 100;
    end
end
surf(Cs, Cs, accuracy_RBF);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
xlabel('Value of parameter C');
ylabel('Value of parameter sigma');
zlabel('Classification Accuracy');
title('Gaussian Kernel SVM');

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% select best model
fprintf('Maximum linear classifier validation accuracy\n');
[maxLin, maxLinInd] = max(accuracy);
disp(maxLin);
fprintf('Parameter C which yielded maximum validation accuracy\n');
disp(Cs(maxLinInd));
fprintf('Maximum Gaussian classifier validation accuracy\n');
disp(max(max(accuracy_RBF)));
fprintf('Parameters which yielded maximum validation accuracy (C, sigma)\n');
[~, maxGauInd1] = max(accuracy_RBF);
[~, maxGauInd2] = max(max(accuracy_RBF));
disp(Cs(maxGauInd1(maxGauInd2)));
disp(Cs(maxGauInd2));

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

% train model using parameters with minimal error and use test set
C = Cs(maxLinInd);
model = svmTrain(Xtrain, ytrain, C, @linearKernel);

pred = svmPredict(model, Xtest);
fprintf('Test Accuracy: %f\n', mean(double(pred == ytest)) * 100);

% where did the model perform best and worst on the test set?
p_test = p(train + val + (1:test));
samples = repmat('e', length(easy_ham) + length(easy_ham_2) - 6,1);
samples = [samples; repmat('h', length(hard_ham) - 3, 1)];
samples = [samples; repmat('s', length(spam) + length(spam_2) - 7,1)];
samples_test = samples(p_test);

% which did the model get right?
fprintf('The model got the following proportions correct\n');
right = samples_test(pred == ytest);
easy = 0;
hard = 0;
spam_num = 0;
for i = 1:length(right)
    if right(i) == 'e'
        easy = easy + 1;
    elseif right(i) == 'h'
        hard = hard + 1;
    else
        spam_num = spam_num + 1;
    end
end
fprintf('Easy:\t');
disp(easy);
fprintf('Hard:\t');
disp(hard);
fprintf('Spam:\t');
disp(spam_num);

% Which did the model get wrong?
fprinft('The model got the following wrong:\n');
wrong = samples_test(pred ~= ytest);
easy_wrong = 0;
hard_wrong = 0;
spam_wrong = 0;
for i = 1:length(wrong)
    if wrong(i) == 'e'
        easy_wrong = easy_wrong + 1;
    elseif wrong(i) == 'h'
        hard_wrong = hard_wrong + 1;
    else
        spam_wrong = spam_wrong + 1;
    end
end
fprintf('Easy:\t');
disp(easy_wrong);
fprintf('Hard:\t');
disp(hard_wrong);
fprintf('Spam:\t');
disp(spam_wrong);

fprintf('The fraction of each category incorrect is then:\n');
fprintf('Easy:\t');
disp(easy_wrong/easy);
fprintf('Hard:\t');
disp(hard_wrong/hard);
fprintf('Spam:\t');
disp(spam_wrong/spam_num);

fprintf('\n\n');
fprintf('Overall, the best model easily classifies the easy emails as non-spam.\n');
fprintf('However, it is less successful at the harder emails.\n');
fprintf('Even more importantly, it is least successful at identifying spam emails.\n');
fprintf('This suggests a necessity to use a precision-recall evaluation scheme rather than just accuracy.\n');