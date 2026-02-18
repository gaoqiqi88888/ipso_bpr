%% Time_IPSOBPR_final.m
% 完全独立的 IPSOBPR 推理时间测试脚本
% 不使用 Sim_Pic_PSO，避免网络对象兼容性问题

clear; clc; close all;

%% ====== 1. 基础参数配置 ======
inputnum = 9;      % 输入节点数（3×3滑动窗口）
hiddennum = 9;     % 隐藏层节点数
outputnum = 1;     % 输出节点数
picsize = [90, 90]; % 图像尺寸
maxit = 100;       % 每张图测试次数

%% ====== 2. 加载预训练的 IPSO 最优参数 ======
fprintf('正在加载 IPSO 优化结果...\n');
mat_name = 'D:\matlab dev\ipso_bpr_v3\ipso\dataset\best90_0801_pop50_gen50_20260211';
load(mat_name);

% 提取 IPSO 最优参数
[~, IPSO_best_idx] = min(IPSO_bestfitness);
ipso_params = IPSO_bestchrom(IPSO_best_idx, :);
fprintf('IPSO 参数维度: %d\n', length(ipso_params));

% 计算BPNN核心参数维度
core_params = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;
fprintf('BPNN核心参数维度: %d\n', core_params);
fprintf('额外参数维度: %d\n', length(ipso_params) - core_params);

%% ====== 3. 获取验证集图片列表 ======
tifDir = 'ipso\valid';
tifFiles = dir(fullfile(tifDir, '*.tif'));
num_tifs = length(tifFiles);
fprintf('找到 %d 张验证图片\n', num_tifs);

if num_tifs == 0
    error('未找到验证图片，请检查路径: %s', tifDir);
end

%% ====== 4. 预分配存储数组 ======
inference_times = zeros(num_tifs, 1);
all_times = [];

%% ====== 5. 主测试循环 ======
fprintf('\n========== 开始测试 IPSOBPR 推理时间 ==========\n');

for pic_idx = 1:num_tifs
    img_path = fullfile(tifDir, tifFiles(pic_idx).name);
    fprintf('\n正在测试第 %d/%d 张: %s\n', pic_idx, num_tifs, tifFiles(pic_idx).name);
    
    % 读取图片并获取输入数据
    [P_Matrix, ~, ~, ~] = Read_Pic_PSO(img_path, picsize);
    % P_Matrix 是 9 × N 的矩阵，每列是一个3×3窗口的像素值
    
    % 预热：第一张图片的前几次不计时
    if pic_idx == 1
        fprintf('  系统预热中...\n');
        for warm = 1:20
            output = predict_bpnn(ipso_params, P_Matrix, inputnum, hiddennum, outputnum);
        end
        fprintf('  预热完成\n');
    end
    
    % 正式测试
    pic_times = zeros(maxit, 1);
    
    for iter = 1:maxit
        tic;
        output = predict_bpnn(ipso_params, P_Matrix, inputnum, hiddennum, outputnum);
        pic_times(iter) = toc;
    end
    
    % ========== 在这里添加异常值去除代码 ==========
    % 对当前图片的100次时间进行排序
    pic_times = sort(pic_times);
    
    % 去掉最高和最低的5% (100次就去掉各5次)
    trim_count = round(maxit * 0.05);
    if trim_count > 0
        pic_times = pic_times(trim_count+1 : end-trim_count);
    end
    % ==============================================
    
    % 计算当前图片的平均推理时间（已去除异常值）
    inference_times(pic_idx) = mean(pic_times);
    all_times = [all_times; pic_times];
    
    fprintf('  平均推理时间: %.6f 秒 (%.3f 毫秒)\n', ...
        inference_times(pic_idx), inference_times(pic_idx)*1000);
end

%% ====== 6. 统计分析 ======
fprintf('\n========== IPSOBPR 推理时间统计 ==========\n');
fprintf('测试图片数量: %d 张\n', num_tifs);
fprintf('每张图片有效测试次数: %d 次 (原%d次, 去除%d个极值)\n', ...
    length(pic_times), maxit, trim_count*2);

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

%% ====== 7. 与 StandardDnCNN 对比 ======
fprintf('\n========== 与 StandardDnCNN 推理时间对比 ==========\n');

if exist('StandardDnCNN_inference_times.mat', 'file')
    data = load('StandardDnCNN_inference_times.mat');
    if isfield(data, 'mean_cnn_time')
        cnn_time = data.mean_cnn_time;
    elseif isfield(data, 'mean_time')
        cnn_time = data.mean_time;
    else
        cnn_time = 0.004044;  % 您之前测得的默认值
    end
    
    fprintf('StandardDnCNN 平均推理时间: %.6f 秒 (%.3f 毫秒)\n', ...
        cnn_time, cnn_time*1000);
    fprintf('IPSOBPR 平均推理时间:        %.6f 秒 (%.3f 毫秒)\n', ...
        mean_time, mean_time*1000);
    
    if mean_time > 0
        speed_ratio = cnn_time / mean_time;
        fprintf('速度倍数: IPSOBPR 是 StandardDnCNN 的 %.2f 倍\n', speed_ratio);
    end
else
    fprintf('未找到 StandardDnCNN_inference_times.mat\n');
    fprintf('使用默认对比值 4.044 毫秒\n');
    
    cnn_time = 0.004044;
    fprintf('StandardDnCNN 平均推理时间: %.6f 秒 (4.044 毫秒)\n', cnn_time);
    fprintf('IPSOBPR 平均推理时间:        %.6f 秒 (%.3f 毫秒)\n', ...
        mean_time, mean_time*1000);
    
    if mean_time > 0
        speed_ratio = cnn_time / mean_time;
        fprintf('速度倍数: IPSOBPR 是 StandardDnCNN 的 %.2f 倍\n', speed_ratio);
    end
end

%% ====== 8. 保存结果 ======
save('IPSOBPR_inference_times.mat', 'inference_times', 'all_times', ...
     'mean_time', 'std_time', 'min_time', 'max_time', 'median_time');
fprintf('\n结果已保存到 IPSOBPR_inference_times.mat\n');
fprintf('\n========== 测试完成 ==========\n');