% CNN 版本1 无时间戳，不建议使用
% CNN图像复原複主程序（增加总指标统计和变量保存）
% 模型 B 更像一个 "标准型" 或 "通用型" 的图像复原网络。
% 它采用了现代深度学习中常见的一些 "最佳实践"，比如批量归一化（Batch Normalization）和逐步减少通道数的设计。这些技术在图像分类、目标检测等许多计算机视觉任务中被证明是有效的，可以加速训练并在一定程度上防止过拟合。因此，它更像是一个通用的图像修复网络模板。
clear; clc; close all;

%% 新增：记录程序开始时间
startTime = tic;  % tic函数记录起始时间点

%% 1. 配置参数
% 数据路径
trainHRPath = 'DIV2K_train_HR_90_gray';
trainBlurPath = 'DIV2K_train_BLUR_90_gray';
validHRPath = 'DIV2K_valid_HR_90_gray';
validBlurPath = 'DIV2K_valid_BLUR_90_gray';

% 输出路径
outputRoot = 'CNN_Image_Restoration_Results';
trainResultPath = fullfile(outputRoot, 'train_results');
validResultPath = fullfile(outputRoot, 'valid_results');
modelSavePath = fullfile(outputRoot, 'models');
metricsPath = fullfile(outputRoot, 'metrics');

% 创建输出文件夹
mkdirs({trainResultPath, validResultPath, modelSavePath, metricsPath});

% 核心参数
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

%% 3. 构建CNN模型
layers = createRestorationCNN(patchSize, imgChannels);

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
save(fullfile(modelSavePath, 'final_model.mat'), 'model', 'trainInfo');

%% 6. 处理训练指标
processTrainMetrics(trainInfo, metricsPath);

%% 7. 验证模型并保存结果（含总指标统计）
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
    
    % 保存图像与指标
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

% 新增：计算验证集总指标（平均值、标准差、最大值、最小值）
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

% 保存总指标到Excel（单独 sheet 或文件）
writetable(validMetrics, fullfile(metricsPath, '验证指标.xlsx'), 'Sheet', '单张图像指标');
writetable(totalStats, fullfile(metricsPath, '验证指标.xlsx'), 'Sheet', '总统计指标');
fprintf('\n验证集总指标统计：\n');
disp(totalStats);

%% 8. 保存训练集样本结果
saveTrainSamples(model, trainBlur, trainResultPath, 20);


%% 9. 新增：保存所有关键变量到带日期时间的MAT文件
% 获取当前日期时间（格式：YYYYMMDD_HHMMSS，精确到秒，避免覆盖）
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');
saveFileName = sprintf('cnnr%s.mat', currentDateTime);
savePath = fullfile(outputRoot, saveFileName);

% 保存关键变量（根据需要调整保存的变量列表）
save(savePath);  % 超参数

% fprintf('\n所有变量已保存至：%s\n', savePath);
% fprintf('所有流程完成！\n');
% fprintf('结果保存路径：%s\n', outputRoot);


%% 新增：计算并显示总运行时间
endTime = toc(startTime);  % toc函数计算与startTime的差值（单位：秒）
% 转换为 时:分:秒 格式
hours = floor(endTime / 3600);
minutes = floor((endTime - hours*3600) / 60);
seconds = endTime - hours*3600 - minutes*60;

fprintf('\n程序总运行时间：%d小时%d分钟%.2f秒\n', hours, minutes, seconds);
fprintf('所有流程完成！\n');
fprintf('结果保存路径：%s\n', outputRoot);



%% 辅助函数1：加载数据集
function [blurImgs, hrImgs, imgNames] = loadDataset(blurPath, hrPath, patchSize, isValidation)
    blurFiles = dir(fullfile(blurPath, '*.tif'));
    hrFiles = dir(fullfile(hrPath, '*.tif'));
    
    if length(blurFiles) ~= length(hrFiles)
        error('模糊图像与清晰图像数量不匹配！当前模糊图：%d张，清晰图：%d张', ...
            length(blurFiles), length(hrFiles));
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

%% 辅助函数2：构建CNN模型
function layers = createRestorationCNN(patchSize, channels)
    layers = [
        imageInputLayer([patchSize, patchSize, channels], 'Name', 'input')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        convolution2dLayer(3, channels, 'Padding', 'same', 'Name', 'output_conv')
        regressionLayer('Name', 'output')
    ];
    
    fprintf('CNN网络构建完成，共%d层\n', numel(layers));
end

%% 辅助函数3：创建多级文件夹
function mkdirs(paths)
    for i = 1:length(paths)
        path = paths{i};
        if ~exist(path, 'dir')
            mkdir(path);
            fprintf('创建文件夹：%s\n', path);
        end
    end
end

%% 辅助函数4：像素值截断
function out = clamp(in, minVal, maxVal)
    out = max(min(in, maxVal), minVal);
end

%% 辅助函数5：处理训练指标
function processTrainMetrics(trainInfo, metricsPath)
    if isfield(trainInfo, 'TrainingLoss')
        trainingLoss = trainInfo.TrainingLoss;
    elseif isfield(trainInfo, 'Loss')
        trainingLoss = trainInfo.Loss;
    else
        warning('未识别训练损失字段，跳过训练指标保存');
        return;
    end
    
    trainingPSNR = 10 * log10(1 ./ max(trainingLoss, 1e-8));
    epochs = length(trainingLoss);
    
    trainMetrics = table((1:epochs)', trainingLoss', trainingPSNR', ...
        'VariableNames', {'Epoch', 'TrainingLoss', 'TrainingPSNR'});
    writetable(trainMetrics, fullfile(metricsPath, '训练指标.xlsx'));
    fprintf('训练指标已保存\n');
end

%% 辅助函数6：保存训练集样本结果
function saveTrainSamples(model, trainBlur, savePath, sampleNum)
    sampleNum = min(sampleNum, size(trainBlur, 4));
    if sampleNum <= 0
        warning('训练集无样本，跳过保存');
        return;
    end
    
    trainIndices = randperm(size(trainBlur, 4), sampleNum);
    for i = 1:sampleNum
        idx = trainIndices(i);
        inputImg = trainBlur(:, :, :, idx);
        restoredImg = predict(model, inputImg);
        restoredImg = squeeze(restoredImg);
        restoredImg = clamp(restoredImg, 0, 1);
        restoredImg = im2double(restoredImg);
        
        imgName = sprintf('train_%04d.tif', idx);
        imwrite(restoredImg, fullfile(savePath, imgName), 'tif');
    end
    fprintf('训练集%d个复原样本已保存\n', sampleNum);
end