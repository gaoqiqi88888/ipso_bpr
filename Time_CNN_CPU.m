%% DnCNN Image Restoration Multi-Config Test Program (CPU Version with Training Time Recording)
clear; clc; close all;

%% Get current date and time
startTime = tic;
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');

%% 1. Basic configuration parameters
% Data paths
trainHRPath = 'DIV2K_train_HR_90_gray';
trainBlurPath = 'DIV2K_train_BLUR_90_gray';
validHRPath = 'DIV2K_valid_HR_90_gray';
validBlurPath = 'DIV2K_valid_BLUR_90_gray';

% Common parameters
patchSize = 88;
imgChannels = 1;
batchSize = 32;
learningRate = 5e-4;
lrDropFactor = 0.5;
lrDropPeriod = 30;

%% 2. Define test configurations (only 3 models)
configs = {
    struct('name', 'StandardCNN', 'layers', 9, 'filters', 64, 'epochs', 50, 'useResidual', false, 'batchSize', 32, 'learningRate', 5e-4),
    struct('name', 'DirectLearningDnCNN', 'layers', 9, 'filters', 64, 'epochs', 50, 'useResidual', false),
    struct('name', 'StandardDnCNN', 'layers', 9, 'filters', 64, 'epochs', 50, 'useResidual', true)
};

%% 3. Load dataset (shared by all configurations)
fprintf('Loading dataset...\n');
[trainBlur, trainHR] = loadDataset(trainBlurPath, trainHRPath, patchSize);
[validBlur, validHR, validNames] = loadDataset(validBlurPath, validHRPath, patchSize, true);

% Basic preprocessing
trainBlur = single(clamp(trainBlur, 0, 1));
trainHR = single(clamp(trainHR, 0, 1));
validBlur = single(clamp(validBlur, 0, 1));
validHR = single(clamp(validHR, 0, 1));

fprintf('Dataset loaded. Training set: %d images, Validation set: %d images\n', ...
    size(trainBlur, 4), size(validBlur, 4));

%% 4. Test all configurations
allResults = struct();

for configIdx = 1:length(configs)
    config = configs{configIdx};
    fprintf('\n=== Testing Configuration %d/%d: %s ===\n', ...
        configIdx, length(configs), config.name);
    
    %% 4.1 Create output folders
    outputRoot = sprintf('DnCNN_Test_%s_%s', config.name, currentDateTime);
    modelSavePath = fullfile(outputRoot, 'models');
    metricsPath = fullfile(outputRoot, 'metrics');
    resultPath = fullfile(outputRoot, 'results');
    
    mkdirs({modelSavePath, metricsPath, resultPath});
    
    %% 4.2 Prepare training data (based on configuration)
    if config.useResidual
        fprintf('Using residual learning strategy...\n');
        % Calculate and normalize residuals
        trainLabels = trainHR - trainBlur;
        residualRange = max(abs(trainLabels(:)));
        trainLabels = single(clamp(trainLabels / residualRange, -1, 1));
        
        validLabels = validHR - validBlur;
        validLabels = single(clamp(validLabels / residualRange, -1, 1));
        
        fprintf('Residual range scaling factor: %.4f\n', residualRange);
    else
        fprintf('Using direct learning strategy...\n');
        trainLabels = trainHR;
        validLabels = validHR;
        residualRange = 1;  % No scaling
    end
    
    %% 4.3 Build model
    fprintf('Building %s model...\n', config.name);
    
    if strcmp(config.name, 'StandardCNN')
        % StandardCNN使用原始CNN的结构（4层）
        layers = [
            imageInputLayer([patchSize, patchSize, imgChannels], 'Name', 'input')
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
            reluLayer('Name', 'relu1')
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
            batchNormalizationLayer('Name', 'bn3')
            reluLayer('Name', 'relu3')
            convolution2dLayer(3, imgChannels, 'Padding', 'same', 'Name', 'output_conv')
            regressionLayer('Name', 'output')
        ];
        fprintf('StandardCNN: 4 layers (64-64-32-1), BN after conv2-3\n');
    else
        % 其他配置使用createDnCNNWithConfig函数
        layers = createDnCNNWithConfig(patchSize, imgChannels, config);
    end
    
    %% 4.4 Configure training options (FORCE CPU EXECUTION)
    commonOptions = {
        'ExecutionEnvironment', 'cpu', ...  % 强制使用CPU
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ValidationData', {validBlur, validLabels}, ...
        'ValidationFrequency', 10, ...
        'OutputNetwork', 'best-validation-loss', ...
        'VerboseFrequency', 1
    };
    
    if strcmp(config.name, 'StandardCNN')
        % StandardCNN使用与原始CNN代码相同的参数
        options = trainingOptions('adam', ...
            'InitialLearnRate', config.learningRate, ...
            'MaxEpochs', config.epochs, ...
            'MiniBatchSize', config.batchSize, ...
            commonOptions{:});
    else
        % 其他DnCNN配置使用原有参数
        options = trainingOptions('adam', ...
            'InitialLearnRate', learningRate, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', lrDropFactor, ...
            'LearnRateDropPeriod', lrDropPeriod, ...
            'MaxEpochs', config.epochs, ...
            'MiniBatchSize', batchSize, ...
            'L2Regularization', 1e-4, ...
            'GradientThreshold', 1, ...
            commonOptions{:});
    end
    
    %% 4.5 Train model and record training time
    fprintf('Training %s model on CPU...\n', config.name);
    
    % 开始计时
    trainStartTime = tic;
    
    [model, trainInfo] = trainNetwork(trainBlur, trainLabels, layers, options);
    
    % 结束计时
    trainElapsedTime = toc(trainStartTime);
    fprintf('%s training completed in %.2f seconds (%.2f minutes)\n', ...
        config.name, trainElapsedTime, trainElapsedTime/60);
    
    % Save model with training time info
    modelFileName = sprintf('%s_model_%s.mat', config.name, currentDateTime);
    save(fullfile(modelSavePath, modelFileName), 'model', 'trainInfo', 'config', 'trainElapsedTime');
    
    %% 4.6 Process training metrics
    processTrainMetrics(trainInfo, metricsPath, config.name, currentDateTime);
    
    %% 4.7 Validate model
    fprintf('\nValidating %s model...\n', config.name);
    validMetrics = table('Size', [length(validNames), 4], ...
        'VariableNames', {'FileName', 'PSNR', 'MSE', 'SSIM'}, ...
        'VariableTypes', {'string', 'double', 'double', 'double'});
    
    for i = 1:length(validNames)
        % Predict
        inputImg = validBlur(:, :, :, i);
        output = predict(model, inputImg);
        output = squeeze(output);
        
        % Process output based on configuration
        if config.useResidual
            % Residual learning: restore residual and add
            residualPred = output * residualRange;
            restoredImg = inputImg + residualPred;
        else
            % Direct learning: output is restored image
            restoredImg = output;
        end
        
        % Post-processing
        restoredImg = clamp(restoredImg, 0, 1);
        restoredImg = im2double(restoredImg);
        
        % Ground truth image
        gtImg = squeeze(validHR(:, :, :, i));
        gtImg = clamp(gtImg, 0, 1);
        gtImg = im2double(gtImg);
        
        % Calculate metrics
        mseVal = immse(restoredImg, gtImg);
        maxPixel = max(gtImg(:));
        if mseVal > 0
            psnrVal = 10 * log10((maxPixel^2) / mseVal);
        else
            psnrVal = 100;  % Perfect reconstruction
        end
        ssimVal = ssim(restoredImg, gtImg);
        
        % Save results
        if i <= 20  % Save first 20 results
            imgName = validNames(i);
            [~, nameOnly, ~] = fileparts(imgName);
            saveName = sprintf('%s_%s.tif', config.name, nameOnly);
            imwrite(restoredImg, fullfile(resultPath, saveName), 'tif');
        end
        
        % Record metrics
        validMetrics.FileName(i) = validNames(i);
        validMetrics.PSNR(i) = psnrVal;
        validMetrics.MSE(i) = mseVal;
        validMetrics.SSIM(i) = ssimVal;
        
        % Print progress
        if mod(i, 20) == 0
            fprintf('Validation progress: %d/%d\n', i, length(validNames));
        end
    end
    
    %% 4.8 Calculate statistics
    totalStats = table();
    totalStats.Config = {config.name};
    totalStats.Layers = config.layers;
    totalStats.Filters = config.filters;
    totalStats.UseResidual = config.useResidual;
    totalStats.TrainingTime = trainElapsedTime;  % 添加训练时间
    totalStats.Mean_PSNR = mean(validMetrics.PSNR);
    totalStats.Std_PSNR = std(validMetrics.PSNR);
    totalStats.Mean_SSIM = mean(validMetrics.SSIM);
    totalStats.Std_SSIM = std(validMetrics.SSIM);
    totalStats.Mean_MSE = mean(validMetrics.MSE);
    totalStats.Std_MSE = std(validMetrics.MSE);
    totalStats.Max_PSNR = max(validMetrics.PSNR);
    totalStats.Min_PSNR = min(validMetrics.PSNR);
    
    % Save validation metrics
    validMetricsFileName = sprintf('%s_validation_metrics_%s.xlsx', config.name, currentDateTime);
    writetable(validMetrics, fullfile(metricsPath, validMetricsFileName), 'Sheet', 'Per_Image_Metrics');
    writetable(totalStats, fullfile(metricsPath, validMetricsFileName), 'Sheet', 'Total_Statistics');
    
    %% 4.9 Save configuration results to structure
    allResults(configIdx).config = config;
    allResults(configIdx).totalStats = totalStats;
    allResults(configIdx).trainingTime = trainElapsedTime;
    allResults(configIdx).modelPath = fullfile(modelSavePath, modelFileName);
    allResults(configIdx).metricsPath = fullfile(metricsPath, validMetricsFileName);
    allResults(configIdx).validMetrics = validMetrics;
    
    % Print current configuration results
    fprintf('\n%s Results:\n', config.name);
    fprintf('Training Time: %.2f seconds (%.2f minutes)\n', trainElapsedTime, trainElapsedTime/60);
    fprintf('Mean PSNR: %.2f dB\n', totalStats.Mean_PSNR);
    fprintf('Mean SSIM: %.4f\n', totalStats.Mean_SSIM);
    fprintf('Mean MSE: %.6f\n', totalStats.Mean_MSE);
    fprintf('Model saved to: %s\n', allResults(configIdx).modelPath);
end

%% 5. Compare all configuration results (INCLUDING TRAINING TIME)
fprintf('\n=== Comparison of All Configurations (with Training Time) ===\n');

% Create comparison table with training time
comparisonTable = table();
for i = 1:length(configs)
    comparisonTable.Config{i} = configs{i}.name;
    comparisonTable.Layers(i) = configs{i}.layers;
    comparisonTable.UseResidual(i) = configs{i}.useResidual;
    comparisonTable.TrainingTime(i) = allResults(i).trainingTime;  % 训练时间
    comparisonTable.TrainingMinutes(i) = allResults(i).trainingTime / 60;  % 转换为分钟
    comparisonTable.Mean_PSNR(i) = allResults(i).totalStats.Mean_PSNR;
    comparisonTable.Std_PSNR(i) = allResults(i).totalStats.Std_PSNR;
    comparisonTable.Mean_SSIM(i) = allResults(i).totalStats.Mean_SSIM;
    comparisonTable.Std_SSIM(i) = allResults(i).totalStats.Std_SSIM;
    comparisonTable.Mean_MSE(i) = allResults(i).totalStats.Mean_MSE;
end

% Sort by PSNR
comparisonTable = sortrows(comparisonTable, 'Mean_PSNR', 'descend');

% Display comparison results
disp(comparisonTable);

% 计算与IPSOBPR的对比（假设IPSOBPR训练时间16.87秒）
ipsoTime = 16.87;  % 从您的Table 9获取
fprintf('\n--- Comparison with IPSOBPR Training Time ---\n');
fprintf('IPSOBPR training time: %.2f seconds\n', ipsoTime);
for i = 1:height(comparisonTable)
    ratio = comparisonTable.TrainingTime(i) / ipsoTime;
    if ratio > 1
        fprintf('%s training time is %.2f × IPSOBPR (%.2f× slower)\n', ...
            comparisonTable.Config{i}, ratio, ratio);
    else
        fprintf('%s training time is %.2f × IPSOBPR (%.2f× faster)\n', ...
            comparisonTable.Config{i}, ratio, 1/ratio);
    end
end

% Ensure comparison folder exists
comparisonDir = 'DnCNN_Comparisons';
if ~exist(comparisonDir, 'dir')
    mkdir(comparisonDir);
    fprintf('Created folder: %s\n', comparisonDir);
end

% Save comparison results
comparisonFileName = sprintf('all_config_comparison_with_time_%s.xlsx', currentDateTime);
comparisonFilePath = fullfile(comparisonDir, comparisonFileName);

try
    writetable(comparisonTable, comparisonFilePath);
    fprintf('Comparison results saved to: %s\n', comparisonFilePath);
catch ME
    fprintf('Warning: Could not save Excel file, trying MAT format\n');
    matFileName = sprintf('all_config_comparison_with_time_%s.mat', currentDateTime);
    save(fullfile(comparisonDir, matFileName), 'comparisonTable');
    fprintf('Comparison results saved as MAT file: %s\n', fullfile(comparisonDir, matFileName));
end

% [后续可视化代码保持不变...]
% 第6-9部分代码保持不变，但建议在图表标题中注明"CPU Training"

%% ==================== Helper Functions ====================

%% Helper Function 1: Load dataset
function [blurImgs, hrImgs, imgNames] = loadDataset(blurPath, hrPath, patchSize, isValidation)
    blurFiles = dir(fullfile(blurPath, '*.tif'));
    hrFiles = dir(fullfile(hrPath, '*.tif'));
    
    if length(blurFiles) ~= length(hrFiles)
        error('Number of blur images does not match HR images!');
    end
    
    numImgs = length(blurFiles);
    blurImgs = zeros(patchSize, patchSize, 1, numImgs, 'single');
    hrImgs = zeros(patchSize, patchSize, 1, numImgs, 'single');
    imgNames = strings(numImgs, 1);
    
    for i = 1:numImgs
        blurImg = imread(fullfile(blurPath, blurFiles(i).name));
        hrImg = imread(fullfile(hrPath, hrFiles(i).name));
        
        if size(blurImg, 3) > 1
            blurImg = rgb2gray(blurImg);
        end
        if size(hrImg, 3) > 1
            hrImg = rgb2gray(hrImg);
        end
        
        blurImg = im2double(blurImg);
        hrImg = im2double(hrImg);
        
        % Center crop
        [h, w] = size(blurImg);
        startRow = floor((h - patchSize)/2) + 1;
        startCol = floor((w - patchSize)/2) + 1;
        blurPatch = blurImg(startRow:startRow+patchSize-1, startCol:startCol+patchSize-1);
        hrPatch = hrImg(startRow:startRow+patchSize-1, startCol:startCol+patchSize-1);
        
        blurImgs(:, :, 1, i) = single(blurPatch);
        hrImgs(:, :, 1, i) = single(hrPatch);
        imgNames(i) = blurFiles(i).name;
    end
    
    if nargin < 4 || ~isValidation
        imgNames = [];
    end
end

%% Helper Function 2: Build DnCNN model with configuration
function layers = createDnCNNWithConfig(patchSize, channels, config)
    % Build DnCNN with specified configuration
    
    if config.layers < 3
        error('Number of layers cannot be less than 3');
    end
    
    % Input layer
    layers = [
        imageInputLayer([patchSize, patchSize, channels], 'Name', 'input')
    ];
    
    % First layer: Conv + ReLU
    layers = [
        layers
        convolution2dLayer(3, config.filters, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
    ];
    
    % Middle layers: Conv + BN + ReLU
    for i = 2:config.layers-1
        layers = [
            layers
            convolution2dLayer(3, config.filters, 'Padding', 'same', 'Name', sprintf('conv%d', i))
            batchNormalizationLayer('Name', sprintf('bn%d', i))
            reluLayer('Name', sprintf('relu%d', i))
        ];
    end
    
    % Output layer
    layers = [
        layers
        convolution2dLayer(3, channels, 'Padding', 'same', 'Name', sprintf('conv%d', config.layers))
        regressionLayer('Name', 'output')
    ];
    
    fprintf('Built %s: %d layers, %d filters\n', config.name, config.layers, config.filters);
end

%% Helper Function 3: Create multiple folders
function mkdirs(paths)
    for i = 1:length(paths)
        path = paths{i};
        if ~exist(path, 'dir')
            mkdir(path);
        end
    end
end

%% Helper Function 4: Clamp pixel values
function out = clamp(in, minVal, maxVal)
    out = max(min(in, maxVal), minVal);
end

%% Helper Function 5: Process training metrics
function processTrainMetrics(trainInfo, metricsPath, configName, datetimeSuffix)
    if isfield(trainInfo, 'TrainingLoss')
        trainingLoss = trainInfo.TrainingLoss;
    elseif isfield(trainInfo, 'Loss')
        trainingLoss = trainInfo.Loss;
    else
        warning('Training loss field not recognized, skipping training metrics save');
        return;
    end
    
    % Calculate PSNR
    trainingPSNR = zeros(size(trainingLoss));
    for i = 1:length(trainingLoss)
        if trainingLoss(i) > 0
            trainingPSNR(i) = 10 * log10(1 / trainingLoss(i));
        else
            trainingPSNR(i) = 100;
        end
    end
    
    % Create table
    trainMetrics = table((1:length(trainingLoss))', trainingLoss(:), trainingPSNR(:), ...
        'VariableNames', {'Epoch', 'TrainingLoss', 'TrainingPSNR'});
    
    % Save
    trainMetricsFileName = sprintf('%s_training_metrics_%s.xlsx', configName, datetimeSuffix);
    writetable(trainMetrics, fullfile(metricsPath, trainMetricsFileName));
    fprintf('Training metrics saved\n');
end

%% Helper Function 6: Ensure comparison folder exists
if ~exist('DnCNN_Comparisons', 'dir')
    mkdir('DnCNN_Comparisons');
end