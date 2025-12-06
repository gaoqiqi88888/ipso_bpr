% CNN高斯模糊图像复原主程序（完整修正版）-模型 A 是任务专用型
% 所有输出添加日期时间，避免覆盖
% ICNN 是为 "高斯模糊复原" 任务专门优化的 "专用型" 网络。
% 它的设计没有盲从通用的最佳实践，而是针对 "从模糊图像中精确恢复细节" 这一特定目标进行了简化和强化。
% 强化：通过堆叠更多相同的卷积层（5 层），它提供了更大的模型容量和更深的特征提取能力，这对于学习逆模糊过程这种复杂的映射关系至关重要。
% 简化：它刻意去掉了批量归一化（Batch Normalization）。正如我们之前讨论的，BN 层可能会破坏图像的精细结构信息，这对于复原任务是不利的。因此，去掉 BN 是一种有针对性的优化，而不是设计上的落后。
clear; clc; close all;

%% 新增：记录程序开始时间 + 获取当前日期时间（统一后缀）
startTime = tic;
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');  % 格式：YYYYMMDD_HHMMSS

%% 1. 配置参数（输出路径添加日期时间）
% 数据路径
trainHRPath = 'DIV2K_train_HR_90_gray';
trainBlurPath = 'DIV2K_train_BLUR_90_gray';  % 高斯模糊数据集
validHRPath = 'DIV2K_valid_HR_90_gray';
validBlurPath = 'DIV2K_valid_BLUR_90_gray';

% 输出根路径（添加日期时间）
outputRoot = sprintf('GaussianDeblur_Results_%s', currentDateTime);
trainResultPath = fullfile(outputRoot, sprintf('train_results_%s', currentDateTime));
validResultPath = fullfile(outputRoot, sprintf('valid_results_%s', currentDateTime));
modelSavePath = fullfile(outputRoot, 'models');
metricsPath = fullfile(outputRoot, 'metrics');

% 创建输出文件夹
mkdirs({trainResultPath, validResultPath, modelSavePath, metricsPath});

% 核心参数（针对高斯模糊优化）
patchSize = 88;
imgChannels = 1;
epochs = 50;
batchSize = 16;
learningRate = 1e-4;

%% 2. 数据加载与预处理
[trainBlur, trainHR] = loadDataset(trainBlurPath, trainHRPath, patchSize);
[validBlur, validHR, validNames] = loadDataset(validBlurPath, validHRPath, patchSize, true);

% 格式与范围校正
trainBlur = single(clamp(trainBlur, 0, 1));
trainHR = single(clamp(trainHR, 0, 1));
validBlur = single(clamp(validBlur, 0, 1));
validHR = single(clamp(validHR, 0, 1));

% 检查数据
fprintf('训练模糊图维度: %s, 范围: [%.4f, %.4f]\n', ...
    mat2str(size(trainBlur)), min(trainBlur(:)), max(trainBlur(:)));
fprintf('训练清晰图维度: %s, 范围: [%.4f, %.4f]\n', ...
    mat2str(size(trainHR)), min(trainHR(:)), max(trainHR(:)));

%% 3. 构建简单的CNN模型（先确保能运行）
fprintf('构建CNN模型...\n');
layers = createSimpleCNN(patchSize, imgChannels);

% 显示网络结构
analyzeNetwork(layers);

%% 4. 配置训练选项
options = trainingOptions('adam', ...
    'InitialLearnRate', learningRate, ...
    'MaxEpochs', epochs, ...
    'MiniBatchSize', batchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', {validBlur, validHR}, ...
    'ValidationFrequency', 10, ...
    'OutputNetwork', 'best-validation-loss', ...
    'VerboseFrequency', 1);

%% 5. 训练模型
fprintf('开始训练CNN模型...\n');
[model, trainInfo] = trainNetwork(trainBlur, trainHR, layers, options);

% 保存模型
modelFileName = sprintf('final_model_%s.mat', currentDateTime);
save(fullfile(modelSavePath, modelFileName), 'model', 'trainInfo');
bestModelFileName = sprintf('best_validation_model_%s.mat', currentDateTime);
save(fullfile(modelSavePath, bestModelFileName), 'model');

fprintf('模型训练完成！\n');

%% 6. 处理训练指标
processTrainMetrics(trainInfo, metricsPath, currentDateTime);

%% 7. 验证模型并保存结果
fprintf('开始验证模型...\n');
if ~isempty(validNames)
    validMetrics = table('Size', [length(validNames), 4], ...
        'VariableNames', {'FileName', 'PSNR', 'MSE', 'SSIM'}, ...
        'VariableTypes', {'string', 'double', 'double', 'double'});

    for i = 1:length(validNames)
        % 预测并处理复原图像
        inputImg = validBlur(:, :, :, i);
        restoredImg = predict(model, inputImg);
        restoredImg = squeeze(restoredImg);
        restoredImg = clamp(restoredImg, 0, 1);
        restoredImg = im2double(restoredImg);
        
        % 真实清晰图像
        gtImg = squeeze(validHR(:, :, :, i));
        gtImg = clamp(gtImg, 0, 1);
        gtImg = im2double(gtImg);
        
        % 计算指标
        mseVal = immse(restoredImg, gtImg);
        maxPixel = max(gtImg(:));
        psnrVal = 10 * log10((maxPixel^2) / max(mseVal, 1e-8));
        ssimVal = ssim(restoredImg, gtImg);
        
        % 保存图像
        imgName = validNames(i);
        imwrite(restoredImg, fullfile(validResultPath, imgName), 'tif');
        validMetrics.FileName(i) = imgName;
        validMetrics.PSNR(i) = psnrVal;
        validMetrics.MSE(i) = mseVal;
        validMetrics.SSIM(i) = ssimVal;
        
        % 打印进度
        if mod(i, 10) == 0
            fprintf('验证进度：%d/%d | 图像：%s | PSNR：%.2f dB\n', ...
                i, length(validNames), imgName, psnrVal);
        end
    end

    % 计算验证集总指标
    totalStats = table();
    totalStats.Mean_PSNR = mean(validMetrics.PSNR);
    totalStats.Std_PSNR = std(validMetrics.PSNR);
    totalStats.Max_PSNR = max(validMetrics.PSNR);
    totalStats.Min_PSNR = min(validMetrics.PSNR);
    totalStats.Mean_MSE = mean(validMetrics.MSE);
    totalStats.Std_MSE = std(validMetrics.MSE);
    totalStats.Max_MSE = max(validMetrics.MSE);
    totalStats.Min_MSE = min(validMetrics.MSE);
    totalStats.Mean_SSIM = mean(validMetrics.SSIM);
    totalStats.Std_SSIM = std(validMetrics.SSIM);
    totalStats.Max_SSIM = max(validMetrics.SSIM);
    totalStats.Min_SSIM = min(validMetrics.SSIM);

    % 保存验证指标
    validMetricsFileName = sprintf('验证指标_%s.xlsx', currentDateTime);
    writetable(validMetrics, fullfile(metricsPath, validMetricsFileName), 'Sheet', '单张图像指标');
    writetable(totalStats, fullfile(metricsPath, validMetricsFileName), 'Sheet', '总统计指标');
    fprintf('\n验证集总指标统计：\n');
    disp(totalStats);
else
    fprintf('无验证数据，跳过验证步骤\n');
    validMetrics = [];
    totalStats = [];
end

%% 8. 保存训练集样本结果
saveTrainSamples(model, trainBlur, trainHR, trainResultPath, 20);

%% 9. 保存所有关键变量
allVarsFileName = sprintf('gaussian_deblur_all_vars_%s.mat', currentDateTime);
savePath = fullfile(outputRoot, allVarsFileName);
save(savePath, ...
    'model', 'trainInfo', ...
    'trainBlur', 'trainHR', 'validBlur', 'validHR', 'validNames', ...
    'validMetrics', 'totalStats', ...
    'epochs', 'batchSize', 'learningRate', 'patchSize');

%% 计算并显示总运行时间
endTime = toc(startTime);
hours = floor(endTime / 3600);
minutes = floor((endTime - hours*3600) / 60);
seconds = endTime - hours*3600 - minutes*60;

fprintf('\n所有变量已保存至：%s\n', savePath);
fprintf('程序总运行时间：%d小时%d分钟%.2f秒\n', hours, minutes, seconds);
fprintf('所有流程完成！\n');
fprintf('结果保存路径：%s\n', outputRoot);

%% 辅助函数定义

function [blurImgs, hrImgs, imgNames] = loadDataset(blurPath, hrPath, patchSize, isValidation)
    if nargin < 4
        isValidation = false;
    end
    
    blurFiles = dir(fullfile(blurPath, '*.tif'));
    hrFiles = dir(fullfile(hrPath, '*.tif'));
    
    if length(blurFiles) ~= length(hrFiles)
        error('模糊图像与清晰图像数量不匹配！当前模糊图：%d张，清晰图：%d张', ...
            length(blurFiles), length(hrFiles));
    end
    
    numImgs = length(blurFiles);
    blurImgs = zeros(patchSize, patchSize, 1, numImgs, 'single');
    hrImgs = zeros(patchSize, patchSize, 1, numImgs, 'single');
    
    if isValidation
        imgNames = strings(numImgs, 1);
    else
        imgNames = [];
    end
    
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
        [h, w] = size(blurImg);
        
        % 中心裁剪
        startRow = floor((h - patchSize)/2) + 1;
        startCol = floor((w - patchSize)/2) + 1;
        blurPatch = blurImg(startRow:startRow+patchSize-1, startCol:startCol+patchSize-1);
        hrPatch = hrImg(startRow:startRow+patchSize-1, startCol:startCol+patchSize-1);
        
        blurImgs(:, :, 1, i) = single(blurPatch);
        hrImgs(:, :, 1, i) = single(hrPatch);
        
        if isValidation
            imgNames(i) = blurFiles(i).name;
        end
    end
end

function layers = createSimpleCNN(patchSize, channels)
    % 简单的CNN模型，确保能正常运行
    layers = [
        imageInputLayer([patchSize, patchSize, channels], 'Name', 'input')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
        reluLayer('Name', 'relu3')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv4')
        reluLayer('Name', 'relu4')
        
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv5')
        reluLayer('Name', 'relu5')
        
        convolution2dLayer(3, channels, 'Padding', 'same', 'Name', 'output_conv')
        
        regressionLayer('Name', 'output')
    ];
    
    fprintf('简单CNN模型构建完成，共%d层\n', numel(layers));
end

function mkdirs(paths)
    for i = 1:length(paths)
        path = paths{i};
        if ~exist(path, 'dir')
            mkdir(path);
            fprintf('创建文件夹：%s\n', path);
        end
    end
end

function out = clamp(in, minVal, maxVal)
    out = max(min(in, maxVal), minVal);
end

function processTrainMetrics(trainInfo, metricsPath, currentDateTime)
    if isempty(trainInfo)
        fprintf('训练信息为空，跳过指标处理\n');
        return;
    end
    
    % 保存训练历史
    trainHistory = struct();
    if isfield(trainInfo, 'TrainingLoss')
        trainHistory.TrainingLoss = trainInfo.TrainingLoss;
    end
    if isfield(trainInfo, 'ValidationLoss')
        trainHistory.ValidationLoss = trainInfo.ValidationLoss;
    end
    if isfield(trainInfo, 'BaseLearnRate')
        trainHistory.LearningRate = trainInfo.BaseLearnRate;
    end
    
    metricsFileName = sprintf('训练历史_%s.mat', currentDateTime);
    save(fullfile(metricsPath, metricsFileName), 'trainHistory');
    
    % 绘制训练曲线
    if isfield(trainInfo, 'TrainingLoss') && ~isempty(trainInfo.TrainingLoss)
        figure('Visible', 'off');
        plot(trainInfo.TrainingLoss, 'b-', 'LineWidth', 2);
        xlabel('迭代次数');
        ylabel('训练损失');
        title('训练损失曲线');
        grid on;
        saveas(gcf, fullfile(metricsPath, sprintf('训练损失曲线_%s.png', currentDateTime)));
        close gcf;
    end
    
    fprintf('训练历史已保存\n');
end

function saveTrainSamples(model, trainBlur, trainHR, trainResultPath, numSamples)
    if numSamples > size(trainBlur, 4)
        numSamples = size(trainBlur, 4);
    end
    
    fprintf('保存%d个训练样本结果...\n', numSamples);
    
    for i = 1:numSamples
        try
            % 预测复原图像
            inputImg = trainBlur(:, :, :, i);
            restoredImg = predict(model, inputImg);
            restoredImg = squeeze(restoredImg);
            restoredImg = clamp(restoredImg, 0, 1);
            restoredImg = im2double(restoredImg);
            
            % 保存复原图像
            imgName = sprintf('train_sample_%03d_restored.tif', i);
            imwrite(restoredImg, fullfile(trainResultPath, imgName), 'tif');
            
            % 保存对应的模糊图像
            blurImg = squeeze(inputImg);
            blurImg = clamp(blurImg, 0, 1);
            blurImg = im2double(blurImg);
            blurName = sprintf('train_sample_%03d_blur.tif', i);
            imwrite(blurImg, fullfile(trainResultPath, blurName), 'tif');
            
            % 保存对应的清晰图像
            hrImg = squeeze(trainHR(:, :, :, i));
            hrImg = clamp(hrImg, 0, 1);
            hrImg = im2double(hrImg);
            hrName = sprintf('train_sample_%03d_hr.tif', i);
            imwrite(hrImg, fullfile(trainResultPath, hrName), 'tif');
            
        catch ME
            fprintf('保存训练样本%d时出错: %s\n', i, ME.message);
        end
    end
    
    fprintf('训练样本结果保存完成\n');
end