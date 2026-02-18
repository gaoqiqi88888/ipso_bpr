%% Time_StandardCNN.m
% 计算 StandardCNN 模型在验证集上的平均推理时间

clear; clc; close all;

%% ====== 1. 基础参数配置 ======
picsize = [90, 90]; % 图像尺寸
patchSize = 88;     % 模型输入尺寸
maxit = 100;        % 每张图测试次数

%% ====== 2. 加载训练好的 StandardCNN 模型 ======
fprintf('正在加载 StandardCNN 模型...\n');
load('StandardCNN_model_20260210_204956.mat');  % 加载模型
fprintf('模型加载完成\n');

% 检查模型变量名
if exist('model', 'var')
    cnn_model = model;
elseif exist('net', 'var')
    cnn_model = net;
else
    error('找不到模型变量，请检查 .mat 文件中的变量名');
end

%% ====== 3. 获取验证集图片列表 ======
tifDir = 'ipso\valid';
tifFiles = dir(fullfile(tifDir, '*.tif'));
num_tifs = length(tifFiles);
fprintf('找到 %d 张验证图片\n', num_tifs);

if num_tifs == 0
    error('未找到验证图片，请检查路径: %s', tifDir);
end

%% ====== 4. 加载验证集图片 ======
fprintf('正在加载验证图片...\n');
validBlur = zeros(patchSize, patchSize, 1, num_tifs, 'single');

for i = 1:num_tifs
    img_path = fullfile(tifDir, tifFiles(i).name);
    img = imread(img_path);
    
    % 转换为灰度
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    
    % 转换为double并归一化
    img = im2double(img);
    
    % 中心裁剪为 patchSize × patchSize
    [h, w] = size(img);
    startRow = floor((h - patchSize)/2) + 1;
    startCol = floor((w - patchSize)/2) + 1;
    img_cropped = img(startRow:startRow+patchSize-1, startCol:startCol+patchSize-1);
    
    validBlur(:, :, 1, i) = single(img_cropped);
    
    if mod(i, 20) == 0
        fprintf('  已加载 %d/%d 张图片\n', i, num_tifs);
    end
end
fprintf('验证图片加载完成\n');

%% ====== 5. 测试推理时间 ======
fprintf('\n========== 开始测试 StandardCNN 推理时间 ==========\n');

% 预热
fprintf('系统预热中...\n');
for warm = 1:20
    output = predict(cnn_model, validBlur(:, :, :, 1));
end
fprintf('预热完成\n');

% 正式测试
all_times = [];

for pic_idx = 1:num_tifs
    fprintf('\n正在测试第 %d/%d 张: %s\n', pic_idx, num_tifs, tifFiles(pic_idx).name);
    
    pic_times = zeros(maxit, 1);
    
    for iter = 1:maxit
        tic;
        output = predict(cnn_model, validBlur(:, :, :, pic_idx));
        pic_times(iter) = toc;
    end
    
    % 去除异常值（去除最高和最低的5%）
    pic_times = sort(pic_times);
    trim_count = round(maxit * 0.05);
    if trim_count > 0
        pic_times = pic_times(trim_count+1 : end-trim_count);
    end
    
    mean_pic_time = mean(pic_times);
    all_times = [all_times; pic_times];
    
    fprintf('  平均推理时间: %.6f 秒 (%.3f 毫秒)\n', ...
        mean_pic_time, mean_pic_time*1000);
end

%% ====== 6. 统计分析 ======
fprintf('\n========== StandardCNN 推理时间统计 ==========\n');
fprintf('测试图片数量: %d 张\n', num_tifs);
fprintf('每张图片有效测试次数: %d 次\n', length(pic_times));

mean_time = mean(all_times);
std_time = std(all_times);
min_time = min(all_times);
max_time = max(all_times);
median_time = median(all_times);

fprintf('\n【整体统计】\n');
fprintf('平均推理时间: %.6f 秒 (%.3f 毫秒)\n', mean_time, mean_time*1000);
fprintf('标准差:       %.6f 秒\n', std_time);
fprintf('中位数:       %.6f 秒 (%.3f 毫秒)\n', median_time, median_time*1000);
fprintf('最长时间:     %.6f 秒 (%.3f 毫秒)\n', max_time, max_time*1000);
fprintf('最短时间:     %.6f 秒 (%.3f 毫秒)\n', min_time, min_time*1000);

%% ====== 7. 保存结果 ======
save('StandardCNN_inference_times.mat', 'all_times', ...
     'mean_time', 'std_time', 'min_time', 'max_time', 'median_time', 'num_tifs');
fprintf('\n结果已保存到 StandardCNN_inference_times.mat\n');