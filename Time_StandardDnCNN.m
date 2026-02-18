% StandardDnCNN_Time.m
% 计算StandardDnCNN模型在验证集上的平均推理时间

clear; clc; close all;

%% 1. 加载训练好的模型
fprintf('正在加载模型...\n');
load('StandardDnCNN_model_20260210_204956.mat');  % 加载后应有变量名为 model
fprintf('模型加载完成\n');

%% 2. 加载验证集数据
% 请根据您的实际数据路径修改
validBlurPath = 'DIV2K_valid_BLUR_90_gray';  % 验证集模糊图像路径

% 获取所有验证图片文件
validFiles = dir(fullfile(validBlurPath, '*.tif'));
num_valid = length(validFiles);  % 定义num_valid变量
fprintf('找到 %d 张验证图片\n', num_valid);

if num_valid == 0
    error('未找到验证图片，请检查路径: %s', validBlurPath);
end

% 预分配存储数组
% 假设图片尺寸为90×90（请根据实际尺寸修改）
imgSize = 88;  % 请根据您的实际图片尺寸修改
validBlur = zeros(imgSize, imgSize, 1, num_valid, 'single');

%% 3. 加载所有验证图片
fprintf('正在加载验证图片...\n');
for i = 1:num_valid
    imgPath = fullfile(validBlurPath, validFiles(i).name);
    img = imread(imgPath);
    
    % 转换为灰度（如果是彩色图）
    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    
    % 转换为double并归一化到[0,1]
    img = im2double(img);


    % 中心裁剪为 88×88
    [h, w] = size(img);
    startRow = floor((h - imgSize)/2) + 1;
    startCol = floor((w - imgSize)/2) + 1;
    img_cropped = img(startRow:startRow+imgSize-1, startCol:startCol+imgSize-1);
    
    % 存入数组
    validBlur(:, :, 1, i) = single(img_cropped);
    
   
    if mod(i, 20) == 0
        fprintf('  已加载 %d/%d 张图片\n', i, num_valid);
    end
end
fprintf('验证图片加载完成\n');



%% 6. 显示结果
fprintf('\n========== StandardDnCNN 推理时间统计 ==========\n');
fprintf('测试图片数量: %d 张\n', num_valid);
fprintf('平均推理时间: %.6f 秒 (%.2f 毫秒)\n', mean_cnn_time, mean_cnn_time*1000);
fprintf('标准差:      %.6f 秒\n', std_cnn_time);
fprintf('最长时间:    %.6f 秒\n', max_cnn_time);
fprintf('最短时间:    %.6f 秒\n', min_cnn_time);
fprintf('==============================================\n');

%% 7. （可选）保存结果
save('StandardDnCNN_inference_times.mat', 'cnn_times', 'mean_cnn_time', ...
     'std_cnn_time', 'min_cnn_time', 'max_cnn_time');
fprintf('结果已保存到 StandardDnCNN_inference_times.mat\n');