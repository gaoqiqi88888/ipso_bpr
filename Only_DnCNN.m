% DnCNN图像复原多配置测试程序
clear; clc; close all;

%% 获取当前日期时间
startTime = tic;
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');

%% 1. 基础配置参数
% 数据路径
trainHRPath = 'DIV2K_train_HR_90_gray';
trainBlurPath = 'DIV2K_train_BLUR_90_gray';
validHRPath = 'DIV2K_valid_HR_90_gray';
validBlurPath = 'DIV2K_valid_BLUR_90_gray';

% 通用参数
patchSize = 50;  % 统一使用50x50的patch
imgChannels = 1;
batchSize = 32;
learningRate = 5e-4;
lrDropFactor = 0.5;
lrDropPeriod = 30;

%% 2. 定义测试配置
%% 优化配置 - 添加到configs数组中测试
configs = {
    struct('name', 'ShallowDnCNN', 'layers', 9, 'filters', 64, 'epochs', 80, 'useResidual', true),
    struct('name', 'MediumDnCNN', 'layers', 13, 'filters', 64, 'epochs', 80, 'useResidual', true),
    struct('name', 'DirectLearning', 'layers', 9, 'filters', 64, 'epochs', 80, 'useResidual', false),
    
    % 新增优化配置
    struct('name', 'OptimalDnCNN_11L', 'layers', 11, 'filters', 64, 'epochs', 100, 'useResidual', true),
    struct('name', 'WideDnCNN_9L', 'layers', 9, 'filters', 128, 'epochs', 80, 'useResidual', true),
    struct('name', 'DeepDnCNN_17L', 'layers', 17, 'filters', 64, 'epochs', 120, 'useResidual', true)
};

%% 3. 加载数据集（所有配置共享）
fprintf('正在加载数据集...\n');
[trainBlur, trainHR] = loadDataset(trainBlurPath, trainHRPath, patchSize);
[validBlur, validHR, validNames] = loadDataset(validBlurPath, validHRPath, patchSize, true);

% 基础预处理
trainBlur = single(clamp(trainBlur, 0, 1));
trainHR = single(clamp(trainHR, 0, 1));
validBlur = single(clamp(validBlur, 0, 1));
validHR = single(clamp(validHR, 0, 1));

fprintf('数据集加载完成。训练集: %d张，验证集: %d张\n', ...
    size(trainBlur, 4), size(validBlur, 4));

%% 4. 遍历所有配置进行测试
allResults = struct();

for configIdx = 1:length(configs)
    config = configs{configIdx};
    fprintf('\n=== 开始测试配置 %d/%d: %s ===\n', ...
        configIdx, length(configs), config.name);
    
    %% 4.1 创建输出文件夹
    outputRoot = sprintf('DnCNN_Test_%s_%s', config.name, currentDateTime);
    modelSavePath = fullfile(outputRoot, 'models');
    metricsPath = fullfile(outputRoot, 'metrics');
    resultPath = fullfile(outputRoot, 'results');
    
    mkdirs({modelSavePath, metricsPath, resultPath});
    
    %% 4.2 准备训练数据（根据配置）
    if config.useResidual
        fprintf('使用残差学习策略...\n');
        % 计算残差并归一化
        trainLabels = trainHR - trainBlur;
        residualRange = max(abs(trainLabels(:)));
        trainLabels = single(clamp(trainLabels / residualRange, -1, 1));
        
        validLabels = validHR - validBlur;
        validLabels = single(clamp(validLabels / residualRange, -1, 1));
        
        fprintf('残差范围缩放因子: %.4f\n', residualRange);
    else
        fprintf('使用直接学习策略...\n');
        trainLabels = trainHR;
        validLabels = validHR;
        residualRange = 1;  % 无缩放
    end
    
    %% 4.3 构建模型
    fprintf('构建%s模型...\n', config.name);
    layers = createDnCNNWithConfig(patchSize, imgChannels, config);
    
    %% 4.4 配置训练选项
    options = trainingOptions('adam', ...
        'InitialLearnRate', learningRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', lrDropFactor, ...
        'LearnRateDropPeriod', lrDropPeriod, ...
        'MaxEpochs', config.epochs, ...
        'MiniBatchSize', batchSize, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'Plots', 'training-progress', ...
        'ValidationData', {validBlur, validLabels}, ...
        'ValidationFrequency', 10, ...
        'OutputNetwork', 'best-validation-loss', ...
        'VerboseFrequency', 1, ...
        'L2Regularization', 1e-4, ...
        'GradientThreshold', 1);
    
    %% 4.5 训练模型
    fprintf('开始训练%s模型...\n', config.name);
    [model, trainInfo] = trainNetwork(trainBlur, trainLabels, layers, options);
    
    % 保存模型
    modelFileName = sprintf('%s_model_%s.mat', config.name, currentDateTime);
    save(fullfile(modelSavePath, modelFileName), 'model', 'trainInfo', 'config');
    
    %% 4.6 处理训练指标
    processTrainMetrics(trainInfo, metricsPath, config.name, currentDateTime);
    
    %% 4.7 验证模型
    fprintf('\n开始验证%s模型...\n', config.name);
    validMetrics = table('Size', [length(validNames), 4], ...
        'VariableNames', {'FileName', 'PSNR', 'MSE', 'SSIM'}, ...
        'VariableTypes', {'string', 'double', 'double', 'double'});
    
    for i = 1:length(validNames)
        % 预测
        inputImg = validBlur(:, :, :, i);
        output = predict(model, inputImg);
        output = squeeze(output);
        
        % 根据配置处理输出
        if config.useResidual
            % 残差学习：恢复残差并相加
            residualPred = output * residualRange;
            restoredImg = inputImg + residualPred;
        else
            % 直接学习：输出即为复原图像
            restoredImg = output;
        end
        
        % 后处理
        restoredImg = clamp(restoredImg, 0, 1);
        restoredImg = im2double(restoredImg);
        
        % 真实图像
        gtImg = squeeze(validHR(:, :, :, i));
        gtImg = clamp(gtImg, 0, 1);
        gtImg = im2double(gtImg);
        
        % 计算指标
        mseVal = immse(restoredImg, gtImg);
        maxPixel = max(gtImg(:));
        psnrVal = 10 * log10((maxPixel^2) / max(mseVal, 1e-8));
        ssimVal = ssim(restoredImg, gtImg);
        
        % 保存结果
        if i <= 20  % 保存前20个结果
            imgName = validNames(i);
            [~, nameOnly, ~] = fileparts(imgName);
            saveName = sprintf('%s_%s.tif', config.name, nameOnly);
            imwrite(restoredImg, fullfile(resultPath, saveName), 'tif');
        end
        
        % 记录指标
        validMetrics.FileName(i) = validNames(i);
        validMetrics.PSNR(i) = psnrVal;
        validMetrics.MSE(i) = mseVal;
        validMetrics.SSIM(i) = ssimVal;
        
        % 打印进度
        if mod(i, 20) == 0
            fprintf('验证进度：%d/%d\n', i, length(validNames));
        end
    end
    
    %% 4.8 计算统计指标
    totalStats = table();
    totalStats.Config = {config.name};
    totalStats.Layers = config.layers;
    totalStats.Filters = config.filters;
    totalStats.UseResidual = config.useResidual;
    totalStats.Mean_PSNR = mean(validMetrics.PSNR);
    totalStats.Std_PSNR = std(validMetrics.PSNR);
    totalStats.Mean_SSIM = mean(validMetrics.SSIM);
    totalStats.Std_SSIM = std(validMetrics.SSIM);
    totalStats.Mean_MSE = mean(validMetrics.MSE);
    totalStats.Std_MSE = std(validMetrics.MSE);
    totalStats.Max_PSNR = max(validMetrics.PSNR);
    totalStats.Min_PSNR = min(validMetrics.PSNR);
    
    % 保存验证指标
    validMetricsFileName = sprintf('%s_验证指标_%s.xlsx', config.name, currentDateTime);
    writetable(validMetrics, fullfile(metricsPath, validMetricsFileName), 'Sheet', '单张图像指标');
    writetable(totalStats, fullfile(metricsPath, validMetricsFileName), 'Sheet', '总统计指标');
    
    %% 4.9 保存配置结果到结构体
    allResults(configIdx).config = config;
    allResults(configIdx).totalStats = totalStats;
    allResults(configIdx).modelPath = fullfile(modelSavePath, modelFileName);
    allResults(configIdx).metricsPath = fullfile(metricsPath, validMetricsFileName);
    
    % 打印当前配置结果
    fprintf('\n%s 配置结果:\n', config.name);
    fprintf('平均PSNR: %.2f dB\n', totalStats.Mean_PSNR);
    fprintf('平均SSIM: %.4f\n', totalStats.Mean_SSIM);
    fprintf('平均MSE: %.6f\n', totalStats.Mean_MSE);
    fprintf('模型已保存至: %s\n', allResults(configIdx).modelPath);
end

%% 5. 比较所有配置结果
%% 5. 比较所有配置结果 - 修改保存路径部分
fprintf('\n=== 所有配置结果比较 ===\n');

% 创建比较表格（保持不变）
comparisonTable = table();
for i = 1:length(configs)
    comparisonTable.Config{i} = configs{i}.name;
    comparisonTable.Layers(i) = configs{i}.layers;
    comparisonTable.UseResidual(i) = configs{i}.useResidual;
    comparisonTable.Mean_PSNR(i) = allResults(i).totalStats.Mean_PSNR;
    comparisonTable.Std_PSNR(i) = allResults(i).totalStats.Std_PSNR;
    comparisonTable.Mean_SSIM(i) = allResults(i).totalStats.Mean_SSIM;
    comparisonTable.Std_SSIM(i) = allResults(i).totalStats.Std_SSIM;
    comparisonTable.Mean_MSE(i) = allResults(i).totalStats.Mean_MSE;
end

% 按PSNR排序
comparisonTable = sortrows(comparisonTable, 'Mean_PSNR', 'descend');

% 显示比较结果
disp(comparisonTable);

% 确保比较文件夹存在 - 修复这部分
comparisonDir = 'DnCNN_Comparisons';
if ~exist(comparisonDir, 'dir')
    mkdir(comparisonDir);
    fprintf('创建文件夹: %s\n', comparisonDir);
end

% 保存比较结果（添加异常处理）
comparisonFileName = sprintf('所有配置比较结果_%s.xlsx', currentDateTime);
comparisonFilePath = fullfile(comparisonDir, comparisonFileName);

try
    writetable(comparisonTable, comparisonFilePath);
    fprintf('比较结果已保存至: %s\n', comparisonFilePath);
catch ME
    fprintf('警告: 无法保存Excel文件，尝试保存为MAT格式\n');
    % 尝试保存为MAT文件
    matFileName = sprintf('所有配置比较结果_%s.mat', currentDateTime);
    save(fullfile(comparisonDir, matFileName), 'comparisonTable');
    fprintf('比较结果已保存为MAT文件: %s\n', fullfile(comparisonDir, matFileName));
    
    % 尝试保存为CSV
    try
        csvFileName = sprintf('所有配置比较结果_%s.csv', currentDateTime);
        writetable(comparisonTable, fullfile(comparisonDir, csvFileName));
        fprintf('比较结果已保存为CSV文件: %s\n', fullfile(comparisonDir, csvFileName));
    catch
        % 如果CSV也失败，至少保存了MAT文件
    end
end

%% 6. 可视化比较结果
figure('Position', [100, 100, 1200, 800]);

% 子图1: PSNR比较
subplot(2, 2, 1);
bar(comparisonTable.Mean_PSNR);
set(gca, 'XTickLabel', comparisonTable.Config);
ylabel('PSNR (dB)');
title('各配置平均PSNR比较');
grid on;

% 添加数值标签
for i = 1:height(comparisonTable)
    text(i, comparisonTable.Mean_PSNR(i), ...
        sprintf('%.2f', comparisonTable.Mean_PSNR(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom');
end

% 子图2: SSIM比较
subplot(2, 2, 2);
bar(comparisonTable.Mean_SSIM);
set(gca, 'XTickLabel', comparisonTable.Config);
ylabel('SSIM');
title('各配置平均SSIM比较');
grid on;

% 子图3: MSE比较
subplot(2, 2, 3);
bar(comparisonTable.Mean_MSE);
set(gca, 'XTickLabel', comparisonTable.Config);
ylabel('MSE');
title('各配置平均MSE比较');
grid on;

% 子图4: 配置参数
subplot(2, 2, 4);
text(0.1, 0.9, sprintf('测试时间: %s', currentDateTime), 'FontSize', 10);
text(0.1, 0.8, sprintf('总图像数: %d', length(validNames)), 'FontSize', 10);
text(0.1, 0.7, sprintf('最佳配置: %s', comparisonTable.Config{1}), 'FontSize', 10);
text(0.1, 0.6, sprintf('最佳PSNR: %.2f dB', comparisonTable.Mean_PSNR(1)), 'FontSize', 10);
text(0.1, 0.5, sprintf('最佳SSIM: %.4f', comparisonTable.Mean_SSIM(1)), 'FontSize', 10);
axis off;
title('测试摘要');

% 保存可视化结果
saveas(gcf, fullfile('DnCNN_Comparisons', sprintf('配置比较图_%s.png', currentDateTime)));
fprintf('可视化结果已保存\n');

%% 7. 保存所有结果
finalSavePath = sprintf('DnCNN_多配置测试_完整结果_%s.mat', currentDateTime);
save(fullfile('DnCNN_Comparisons', finalSavePath), ...
    'allResults', 'comparisonTable', 'configs', 'currentDateTime');
fprintf('所有结果已保存至: %s\n', fullfile('DnCNN_Comparisons', finalSavePath));

%% 8. 计算总运行时间
endTime = toc(startTime);
hours = floor(endTime / 3600);
minutes = floor((endTime - hours*3600) / 60);
seconds = endTime - hours*3600 - minutes*60;

fprintf('\n=== 所有测试完成 ===\n');
fprintf('总运行时间: %d小时%d分钟%.2f秒\n', hours, minutes, seconds);
fprintf('最佳配置: %s (PSNR: %.2f dB)\n', ...
    comparisonTable.Config{1}, comparisonTable.Mean_PSNR(1));

%% ==================== 辅助函数 ====================

%% 辅助函数1：加载数据集
function [blurImgs, hrImgs, imgNames] = loadDataset(blurPath, hrPath, patchSize, isValidation)
    blurFiles = dir(fullfile(blurPath, '*.tif'));
    hrFiles = dir(fullfile(hrPath, '*.tif'));
    
    if length(blurFiles) ~= length(hrFiles)
        error('模糊图像与清晰图像数量不匹配！');
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
        
        % 中心裁剪
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

%% 辅助函数2：根据配置构建DnCNN模型
function layers = createDnCNNWithConfig(patchSize, channels, config)
    % 构建指定配置的DnCNN
    
    if config.layers < 3
        error('层数不能少于3层');
    end
    
    % 输入层
    layers = [
        imageInputLayer([patchSize, patchSize, channels], 'Name', 'input')
    ];
    
    % 第一层：Conv + ReLU
    layers = [
        layers
        convolution2dLayer(3, config.filters, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
    ];
    
    % 中间层：Conv + BN + ReLU
    for i = 2:config.layers-1
        layers = [
            layers
            convolution2dLayer(3, config.filters, 'Padding', 'same', 'Name', sprintf('conv%d', i))
            batchNormalizationLayer('Name', sprintf('bn%d', i))
            reluLayer('Name', sprintf('relu%d', i))
        ];
    end
    
    % 输出层
    layers = [
        layers
        convolution2dLayer(3, channels, 'Padding', 'same', 'Name', sprintf('conv%d', config.layers))
        regressionLayer('Name', 'output')
    ];
    
    fprintf('构建%s: %d层，%d个滤波器\n', config.name, config.layers, config.filters);
end

%% 辅助函数3：创建多级文件夹
function mkdirs(paths)
    for i = 1:length(paths)
        path = paths{i};
        if ~exist(path, 'dir')
            mkdir(path);
        end
    end
end

%% 辅助函数4：像素值截断
function out = clamp(in, minVal, maxVal)
    out = max(min(in, maxVal), minVal);
end

%% 辅助函数5：处理训练指标
function processTrainMetrics(trainInfo, metricsPath, configName, datetimeSuffix)
    if isfield(trainInfo, 'TrainingLoss')
        trainingLoss = trainInfo.TrainingLoss;
    elseif isfield(trainInfo, 'Loss')
        trainingLoss = trainInfo.Loss;
    else
        warning('未识别训练损失字段，跳过训练指标保存');
        return;
    end
    
    % 计算PSNR
    trainingPSNR = zeros(size(trainingLoss));
    for i = 1:length(trainingLoss)
        trainingPSNR(i) = 10 * log10(1 / max(trainingLoss(i), 1e-8));
    end
    
    % 创建表格
    trainMetrics = table((1:length(trainingLoss))', trainingLoss(:), trainingPSNR(:), ...
        'VariableNames', {'Epoch', 'TrainingLoss', 'TrainingPSNR'});
    
    % 保存
    trainMetricsFileName = sprintf('%s_训练指标_%s.xlsx', configName, datetimeSuffix);
    writetable(trainMetrics, fullfile(metricsPath, trainMetricsFileName));
    fprintf('训练指标已保存\n');
end

%% 辅助函数6：确保比较文件夹存在
if ~exist('DnCNN_Comparisons', 'dir')
    mkdir('DnCNN_Comparisons');
end