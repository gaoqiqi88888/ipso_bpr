%% Compare_All_Inference_Times.m
% 对比所有模型的推理时间

clear; clc;

%% ====== 加载所有模型的推理时间 ======
fprintf('加载 IPSOBPR 推理时间...\n');
if exist('IPSOBPR_inference_times.mat', 'file')
    load('IPSOBPR_inference_times.mat', 'mean_time');
    ipso_time = mean_time;
else
    ipso_time = 0.001947;  % 默认值
end

fprintf('加载 StandardDnCNN 推理时间...\n');
if exist('StandardDnCNN_inference_times.mat', 'file')
    data = load('StandardDnCNN_inference_times.mat');
    if isfield(data, 'mean_cnn_time')
        dncnn_time = data.mean_cnn_time;
    elseif isfield(data, 'mean_time')
        dncnn_time = data.mean_time;
    else
        dncnn_time = 0.004044;
    end
else
    dncnn_time = 0.004044;
end

fprintf('加载 StandardCNN 推理时间...\n');
if exist('StandardCNN_inference_times.mat', 'file')
    load('StandardCNN_inference_times.mat', 'mean_time');
    standard_cnn_time = mean_time;
else
    standard_cnn_time = 0;  % 未找到
end

fprintf('加载 DirectLearningDnCNN 推理时间...\n');
if exist('DirectLearningDnCNN_inference_times.mat', 'file')
    load('DirectLearningDnCNN_inference_times.mat', 'mean_time');
    direct_time = mean_time;
else
    direct_time = 0;  % 未找到
end

%% ====== 创建对比表格 ======
methods = {'IPSOBPR'; 'StandardDnCNN'; 'StandardCNN'; 'DirectLearningDnCNN'};
times_ms = [ipso_time*1000; dncnn_time*1000; standard_cnn_time*1000; direct_time*1000];

% 计算相对速度（以 IPSOBPR 为基准）
speed_ratios = times_ms / times_ms(1);

% 创建表格
T = table(methods, times_ms, speed_ratios, ...
    'VariableNames', {'Method', 'InferenceTime_ms', 'RelativeSpeed'});

% 按时间排序
T = sortrows(T, 'InferenceTime_ms');

%% ====== 显示结果 ======
fprintf('\n========== 所有模型推理时间对比 ==========\n');
disp(T);

fprintf('\n【与 IPSOBPR 对比】\n');
for i = 2:height(T)
    fprintf('%s 是 IPSOBPR 的 %.2f 倍 (%.2f ms vs %.2f ms)\n', ...
        T.Method{i}, T.RelativeSpeed(i), T.InferenceTime_ms(i), T.InferenceTime_ms(1));
end

%% ====== 保存结果 ======
save('All_Inference_Times_Comparison.mat', 'T');
fprintf('\n结果已保存到 All_Inference_Times_Comparison.mat\n');