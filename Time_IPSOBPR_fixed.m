%% Time_IPSOBPR_fixed.m
% 独立脚本：测量 IPSOBPR 推理时间（只取前100个参数）

clear; clc; close all;

%% ====== 1. 基础参数配置 ======
inputnum = 9;      % 输入节点数
hiddennum = 9;     % 隐藏层节点数
outputnum = 1;     % 输出节点数
picsize = [90, 90]; % 图像尺寸
maxit = 100;       % 每张图测试次数

% 计算BPNN参数总维度
numsum = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;  % 9*9+9+9*1+1 = 100
fprintf('BPNN 参数维度: %d\n', numsum);

%% ====== 2. 加载预训练的 IPSO 最优参数 ======
fprintf('正在加载 IPSO 优化结果...\n');
mat_name = 'D:\matlab dev\ipso_bpr_v3\ipso\dataset\best90_0801_pop50_gen50_20260211';
load(mat_name);

% 查看 .mat 文件中包含哪些变量
fprintf('\n.mat 文件中的变量:\n');
whos

% 提取 IPSO 最优参数
[~, IPSO_best_idx] = min(IPSO_bestfitness);
ipso_params_full = IPSO_bestchrom(IPSO_best_idx, :);
fprintf('IPSO 完整参数维度: %d\n', length(ipso_params_full));

% 只取前 numsum 个参数（BPNN 权重和阈值）
if length(ipso_params_full) >= numsum
    ipso_params = ipso_params_full(1:numsum);
    fprintf('提取前 %d 个参数作为 BPNN 参数\n', numsum);
else
    error('参数维度不足，无法提取');
end

%% ====== 3. 验证参数维度 ======
fprintf('\n参数维度验证:\n');
fprintf('  输入->隐藏权重: %d\n', inputnum*hiddennum);
fprintf('  隐藏层阈值:    %d\n', hiddennum);
fprintf('  隐藏->输出权重: %d\n', hiddennum*outputnum);
fprintf('  输出层阈值:    %d\n', outputnum);
fprintf('  总维度:        %d\n', numsum);