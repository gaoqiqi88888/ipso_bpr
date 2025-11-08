%% 新版论文需要。重要文件，2025年国庆后新版 PSO和IPSO对比,对比PSO_improved_exp算法——幂次递减惯性权重
% PSO与IPSO对比代码（适配Figure 6绘制需求，格式优化版）
%% Paper Dynamic Demonstration -- Comparison of PSO and IPSO Algorithms
%% This code compares PSO and IPSO algorithms for multiple rounds of testing. Maxit specifies the number of comparisons (100 in the paper)
%% The image size is 90×90. The two optimization algorithms are run 100 times respectively, and the optimal fitness values are used for comparison
%% Result data (e.g., best90_r100_lenna.mat) is saved in the "result" folder
%% Basic data for Table 2, Figure 6 (Adaptation Curves Comparison) are generated from this file
%% Figure 6 uses the running results of Lenna.tif for visualization

% 清空环境变量
clear
close all
clc

t1 = clock;  % 记录总运行时间起点
%% 1. 
% Maxit = 100;                % 算法对比轮次（论文要求100次以消除随机误差）
Maxit = 100;

% 获取当前日期（格式：2025-10-06）
current_date = datestr(now(), 'yyyy-mm-dd');  % '-' 分隔，符合日常阅读习惯
% 拼接文件名（如 'ipso/best90_r100_10_2025-10-06.mat'）
mat_name = sprintf('ipso/best90_r100_%d_%s.mat', Maxit, current_date);
% 验证结果
fprintf('带日期的MAT文件名：%s\n', mat_name);


% 1.1 图像与数据预处理参数
picname = 'img/Lenna.tif';  % 测试图像（论文指定Lenna.tif）

picsize = [90, 90];         % 图像尺寸（论文设定90×90以加速收敛）
gauss_kernel_size = 9;      % 高斯模糊核大小
gauss_sigma = 1;            % 高斯模糊标准差
salt_pepper_density = 0.02; % 椒盐噪声密度（混合退化场景）

% 1.2 BPNN网络结构（与原代码一致，保证对比公平性）
inputnum = 9;               % 输入节点数（3×3滑动窗口）
hiddennum = 9;              % 隐藏层节点数
outputnum = 1;              % 输出节点数（单像素预测）
numsum = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;  % PSO/IPSO优化维度（BPNN权值+阈值总数）

% 1.3 PSO与IPSO算法参数（关键参数与原GA/IGA匹配，确保可比性）
sizepop = 100;              % 种群/粒子群规模（原GA sizepop=10，此处按论文需求调整为100）
maxgen = 100;               % 最大迭代次数（原GA maxgen=20，论文设定100次以显化收敛差异）
% maxgen = 50;

% 1.4 PSO核心控制参数
c1 = 1.5;                   % 个体学习因子
c2 = 1.5;                   % 全局学习因子
w_init = 0.9;               % IPSO初始惯性权重
w_final = 0.3;              % IPSO最终惯性权重
v_max = 0.5;                % 粒子最大速度（防止位置更新过度）
v_min = -0.5;               % 粒子最小速度
pos_max = 1;                % 粒子位置上限（BPNN参数范围）
pos_min = -1;               % 粒子位置下限
perturb_trigger_ratio = 0.7;% IPSO高斯扰动触发阈值（迭代次数70%后）
perturb_std = 0.1;          % 高斯扰动标准差

% 1.5 结果存储数组初始化
% PSO结果存储
PSO_bestchrom = zeros(Maxit, numsum);       % 每轮最优粒子（BPNN参数）
PSO_bestfitness = zeros(Maxit, 1);          % 每轮最优适应度（BPNN预测误差）
PSO_trace_bestfitness = zeros(maxgen, Maxit);% 每轮迭代的最优适应度曲线
PSO_time = zeros(Maxit, 1);                 % 每轮运行时间（分钟）

% IPSO结果存储（结构与PSO一致）
IPSO_bestchrom = zeros(Maxit, numsum);
IPSO_bestfitness = zeros(Maxit, 1);
IPSO_trace_bestfitness = zeros(maxgen, Maxit);
IPSO_time = zeros(Maxit, 1);

%% 2. 图像退化处理（混合退化：高斯模糊+椒盐噪声，匹配IPSOBPR算法场景）
% % 2.1 读取并标准化原始图像
% image_orgin = imread(picname);
% if size(image_orgin, 3) == 3
%     image_orgin = rgb2gray(image_orgin);  % 转为灰度图
% end
% image_resized = imresize(image_orgin, picsize);  % 调整为90×90
% image_resized = double(image_resized) / 256;     % 归一化到[0,1]
% 
% % 2.2 添加高斯模糊
% w_gauss = fspecial('gaussian', gauss_kernel_size, gauss_sigma);
% image_blurred = imfilter(image_resized, w_gauss, 'replicate');

[P_Matrix, T_Matrix, image_resized, image_blurred] = Read_Pic_Add_Blurr1(picname, picsize, inputnum);

[P_Matrix, T_Matrix] = generate_training_data(image_blurred, image_resized, inputnum);

% 2.3 添加椒盐噪声（混合退化）
% image_degraded = imnoise(image_blurred, 'salt & pepper', salt_pepper_density);

% 2.4 生成BPNN训练数据（3×3滑动窗口提取特征）
% [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_resized, inputnum);

% 2.5 初始化BPNN网络（与原代码一致）
net.trainParam.epochs = 1000;    % 训练次数（与原代码一致）
net.trainParam.lr = 0.1;         % 学习率
net.trainParam.goal = 1e-5;      % 训练目标误差
net.trainParam.showWindow = false;  % 关闭训练窗口（批量实验用）
net.trainParam.showCommandLine = false;  % 关闭命令行输出
net = newff(P_Matrix, T_Matrix, hiddennum);
% 定义适应度函数（BPNN预测误差，minimization目标）
fobj = @(x) cal_fitness(x, inputnum, hiddennum, outputnum, net, P_Matrix, T_Matrix);

%% 3. PSO与IPSO多轮对比实验
for index = 1:Maxit
    fprintf('==================== 第%d轮对比实验 ====================\n', index);
    
    % 3.1 PSO算法运行（对比基准）
    tt1 = clock;
    [bestchrom_PSO, bestfitness_PSO, trace_best_PSO] = PSO_standard(sizepop, maxgen, numsum, fobj, c1, c2, w_init, v_max, v_min, pos_max, pos_min);
    % 存储PSO结果
    PSO_bestchrom(index, :) = bestchrom_PSO;
    PSO_bestfitness(index) = bestfitness_PSO;
    PSO_trace_bestfitness(:, index) = trace_best_PSO;
    % 计算PSO运行时间
    tt2 = clock;
    PSO_time(index) = etime(tt2, tt1) / 60;  % 转换为分钟
    fprintf('PSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_PSO, PSO_time(index));
    
    % 3.2 IPSO算法运行（改进算法，含自适应惯性权重+高斯扰动）
    tt3 = clock;
    [bestchrom_IPSO, bestfitness_IPSO, trace_best_IPSO] = PSO_improved_exp(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std);
    % 存储IPSO结果
    IPSO_bestchrom(index, :) = bestchrom_IPSO;
    IPSO_bestfitness(index) = bestfitness_IPSO;
    IPSO_trace_bestfitness(:, index) = trace_best_IPSO;
    % 计算IPSO运行时间
    tt4 = clock;
    IPSO_time(index) = etime(tt4, tt3) / 60;  % 转换为分钟
    fprintf('IPSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_IPSO, IPSO_time(index));
end

%% 4. 结果统计与Figure 6绘制（适配论文Figure 6：适应度曲线对比）
% 4.1 统计100轮中最优的一次结果（消除随机误差，论文常用策略）
[PSO_best_overall, PSO_best_idx] = min(PSO_bestfitness);
[IPSO_best_overall, IPSO_best_idx] = min(IPSO_bestfitness);

% % 4.1.2 统计100轮中位数结果（替代最优一次，降低随机离群影响）
% PSO_median_fit  = median(PSO_bestfitness);                % 中位适应度
% [~, PSO_median_idx] = min(abs(PSO_bestfitness - PSO_median_fit)); % 最近轮次
% 
% IPSO_median_fit  = median(IPSO_bestfitness);
% [~, IPSO_median_idx] = min(abs(IPSO_bestfitness - IPSO_median_fit));

% 4.2 提取最优轮次的适应度曲线（用于Figure 6）
PSO_best_curve = PSO_trace_bestfitness(:, PSO_best_idx);
IPSO_best_curve = IPSO_trace_bestfitness(:, IPSO_best_idx);

% 4.3 绘制Figure 6（Adaptation Curves Comparison）
figure('Position', [100, 100, 800, 500]);  % 设置图像大小
plot(1:maxgen, PSO_best_curve, 'b--', 'LineWidth', 2);  % PSO曲线：蓝色虚线
hold on
plot(1:maxgen, IPSO_best_curve, 'r-', 'LineWidth', 2);   % IPSO曲线：红色实线
hold off

% 图表美化（符合论文规范）
title('Comparison of Adaptation Curves', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Iterations', 'FontSize', 12);
ylabel('Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
legend('PSO', 'IPSO', 'Location', 'best', 'FontSize', 11);
grid on;  % 添加网格线（增强可读性）
set(gca, 'FontSize', 10);  % 设置坐标轴字体大小

% 4.4 保存Figure 6（高清格式，适合论文插入）
print('Figure_6_Adaptation_Curves.eps', '-depsc', '-r600');  % EPS格式，600dpi
print('Figure_6_Adaptation_Curves.png', '-dpng', '-r600');  % PNG格式备份
print('Figure_6_Adaptation_Curves.tiff', '-dtiff', '-r600');  % TIFF格式备份


%% 4.2new  改为"中位数轮次"收敛曲线（替代最优轮次）
% PSO 中位数
PSO_median_fit  = median(PSO_bestfitness);                           % 中位数值
[~, PSO_median_idx] = min(abs(PSO_bestfitness - PSO_median_fit));   % 最近轮次索引

% IPSO 中位数
IPSO_median_fit  = median(IPSO_bestfitness);
[~, IPSO_median_idx] = min(abs(IPSO_bestfitness - IPSO_median_fit));

% 提取对应收敛曲线
PSO_median_curve  = PSO_trace_bestfitness(:, PSO_median_idx);
IPSO_median_curve = IPSO_trace_bestfitness(:, IPSO_median_idx);

%% 4.3new  绘制 Figure 6（中位数收敛曲线对比）
figure('Position', [100, 100, 800, 500]);
plot(1:maxgen, PSO_median_curve, 'b--', 'LineWidth', 2); hold on
plot(1:maxgen, IPSO_median_curve, 'r-', 'LineWidth', 2); hold off

title('Comparison of Median Adaptation Curves', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Iterations', 'FontSize', 12);
ylabel('Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
legend('PSO (median run)', 'IPSO (median run)', 'Location', 'best', 'FontSize', 11);
grid on; set(gca, 'FontSize', 10);

%% 4.4new  保存高清图
print('Figure_6_Median_Adaptation_Curves.eps', '-depsc', '-r600');
print('Figure_6_Median_Adaptation_Curves.png', '-dpng', '-r600');
print('Figure_6_Median_Adaptation_Curves.tiff', '-dtiff', '-r600');

%% 5. 结果输出与数据保存
% 5.1 打印最优秀统计结果
fprintf('\n==================== 100轮对比实验统计结果 ====================\n');
fprintf('PSO - 最优适应度: %.6f, 平均适应度: %.6f, 平均运行时间: %.2f分钟\n', ...
    PSO_best_overall, mean(PSO_bestfitness), mean(PSO_time));
fprintf('IPSO - 最优适应度: %.6f, 平均适应度: %.6f, 平均运行时间: %.2f分钟\n', ...
    IPSO_best_overall, mean(IPSO_bestfitness), mean(IPSO_time));
fprintf('IPSO相对PSO适应度提升: %.2f%%\n', (PSO_best_overall - IPSO_best_overall)/PSO_best_overall * 100);


%% 5.1 打印中位数统计结果
fprintf('\n==================== 100轮对比实验统计结果（中位数） ====================\n');

fprintf('PSO - 中位适应度: %.6f, 平均适应度: %.6f, 平均运行时间: %.2f分钟\n', ...
        PSO_median_fit, mean(PSO_bestfitness), mean(PSO_time));

fprintf('IPSO - 中位适应度: %.6f, 平均适应度: %.6f, 平均运行时间: %.2f分钟\n', ...
        IPSO_median_fit, mean(IPSO_bestfitness), mean(IPSO_time));

fprintf('IPSO相对PSO适应度提升（中位数）: %.2f%%\n', ...
        (PSO_median_fit - IPSO_median_fit)/PSO_median_fit * 100);



% 5.2 总运行时间统计
t2 = clock;
total_time = etime(t2, t1) / 60;
fprintf('\n总运行时间: %.2f分钟\n', total_time);

% 保存结果
save(mat_name);
% 5.2 保存结果数据（便于后续Table 2生成）

% save(mat_name, ...
%     'PSO_bestchrom', 'PSO_bestfitness', 'PSO_trace_bestfitness', 'PSO_time', ...
%     'IPSO_bestchrom', 'IPSO_bestfitness', 'IPSO_trace_bestfitness', 'IPSO_time', ...
%     'PSO_best_curve', 'IPSO_best_curve');



%% 5. 绘制 Figure 7. Comparison of Best Fitness Values
% 5.1 准备数据：提取100轮实验中每轮的PSO与IPSO最优适应度
% PSO每轮最优适应度（已在多轮实验中存储于PSO_bestfitness数组）
PSO_per_round_best = PSO_bestfitness;  
% IPSO每轮最优适应度（对应IPSO_bestfitness数组）
IPSO_per_round_best = IPSO_bestfitness;  
% 生成轮次序号（1~100）
rounds = 1:Maxit;  

% 5.2 创建Figure 7图像窗口（设置尺寸与位置，与Figure 6风格统一）
figure('Position', [200, 200, 800, 500]);  % 窗口位置(x,y)与大小(width,height)

% 5.5 图表美化 + 均值参考线（Best 版）
PSO_mean_best  = mean(PSO_per_round_best);
IPSO_mean_best = mean(IPSO_per_round_best);

figure('Position', [200, 200, 800, 500]);
% 散点
plot(rounds, PSO_per_round_best,  'b--o', ...
     'LineWidth', 1.5, 'MarkerSize', 4, ...
     'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none'); hold on
plot(rounds, IPSO_per_round_best, 'r-^', ...
     'LineWidth', 1.5, 'MarkerSize', 4, ...
     'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');

%% 均值参考线

% 1. 计算均值
PSO_mean_best  = mean(PSO_per_round_best);
IPSO_mean_best = mean(IPSO_per_round_best);

% 2. 水平线
plot(rounds, ones(1,numel(rounds))*PSO_mean_best,  'b:', 'LineWidth', 2);
plot(rounds, ones(1,numel(rounds))*IPSO_mean_best, 'r:', 'LineWidth', 2);

% 3. 文字标签 —— 横向移出密集区 + 白底边框
xText = max(rounds) + 3;          % 放在最右端之外，避免重叠
yOff  = 0.15;                     % 上下微偏移，防止重叠线

text(xText, PSO_mean_best + yOff, ...
     sprintf('PSO Mean\n%.4f', PSO_mean_best), ...
     'Color', 'b', ...
     'FontSize', 9, ...
     'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', ...
     'BackgroundColor', 'white', ...
     'Margin', 1);                 % 白底边框

text(xText, IPSO_mean_best - yOff, ...
     sprintf('IPSO Mean\n%.4f', IPSO_mean_best), ...
     'Color', 'r', ...
     'FontSize', 9, ...
     'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', ...
     'BackgroundColor', 'white', ...
     'Margin', 1);

% 4. 把 x 轴范围稍微右扩，让文字落在图内
xlim([1, max(rounds) + 8]);

hold off
title('Comparison of Best Fitness Values', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Experiments (Rounds)', 'FontSize', 12);
ylabel('Best Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
legend('PSO - Per Round Best', 'IPSO - Per Round Best', 'Location', 'best', 'FontSize', 11);
grid on; set(gca, 'FontSize', 10);

% 保存
print(fullfile('ipso','Figure_7_Best_Fitness_Comparison_withMean.eps'), '-depsc', '-r600');
print(fullfile('ipso','Figure_7_Best_Fitness_Comparison_withMean.png'), '-dpng', '-r600');
print(fullfile('ipso','Figure_7_Best_Fitness_Comparison_withMean.tiff'), '-dtiff', '-r600');



%% 5. 绘制 Figure 7. Comparison of Median Fitness Values
% 5.1 准备数据：100 轮中每轮的中位适应度
PSO_per_round_median  = median(PSO_trace_bestfitness, 1);   % 每列中位数 → 1×100
IPSO_per_round_median = median(IPSO_trace_bestfitness, 1);
rounds = 1:numel(PSO_per_round_median);                    % 1~100

% 5.5 图表美化 + 均值参考线（Median 版）
%% 5.5 图表美化 + 均值参考线（Median 版）
PSO_mean_median  = mean(PSO_per_round_median);
IPSO_mean_median = mean(IPSO_per_round_median);

figure('Position', [200, 200, 800, 500]);
plot(rounds, PSO_per_round_median,  'b--o', ...
     'LineWidth', 1.5, 'MarkerSize', 4, ...
     'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none'); hold on
plot(rounds, IPSO_per_round_median, 'r-^', ...
     'LineWidth', 1.5, 'MarkerSize', 4, ...
     'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');

% 均值水平线
plot(rounds, ones(1,numel(rounds))*PSO_mean_median,  'b:', 'LineWidth', 2);
plot(rounds, ones(1,numel(rounds))*IPSO_mean_median, 'r:', 'LineWidth', 2);

% 文字标签 —— 横向移出 + 白底边框
xText = max(rounds) + 3;   % 右移
yOff  = 0.15;              % 上下微偏移

text(xText, PSO_mean_median + yOff, ...
     sprintf('PSO Mean\n%.4f', PSO_mean_median), ...
     'Color', 'b', 'FontSize', 9, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', ...
     'BackgroundColor', 'white', 'Margin', 1);

text(xText, IPSO_mean_median - yOff, ...
     sprintf('IPSO Mean\n%.4f', IPSO_mean_median), ...
     'Color', 'r', 'FontSize', 9, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', ...
     'BackgroundColor', 'white', 'Margin', 1);

hold off
title('Comparison of Median Fitness Values', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Number of Experiments (Rounds)', 'FontSize', 12);
ylabel('Median Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
legend('PSO - Per Round Median', 'IPSO - Per Round Median', 'Location', 'best', 'FontSize', 11);
grid on; set(gca, 'FontSize', 10);

% 把 x 轴右扩，确保文字落在图内
xlim([1, max(rounds) + 8]);

%% 5.6 保存高清图（含均值标注）
print(fullfile('ipso','Figure_7_Median_Fitness_Comparison_withMean.eps'), '-depsc', '-r600');
print(fullfile('ipso','Figure_7_Median_Fitness_Comparison_withMean.png'), '-dpng', '-r600');
print(fullfile('ipso','Figure_7_Median_Fitness_Comparison_withMean.tiff'), '-dtiff', '-r600');


% %% 附录：自定义函数（需与主代码放在同一目录下）
% % 1. generate_training_data.m：生成BPNN训练数据（3×3滑动窗口）
% function [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_clear, inputnum)
%     [h, w] = size(image_degraded);
%     data_len = (h-2)*(w-2);  % 滑动窗口数量（边缘像素不参与，避免边界效应）
%     P_Matrix = zeros(inputnum, data_len);  % 输入矩阵（9×N）
%     T_Matrix = zeros(1, data_len);         % 目标矩阵（1×N，清晰图像中心像素）
%     t = 1;
% 
%     % 遍历图像，提取3×3窗口
%     for i = 2:h-1
%         for j = 2:w-1
%             % 提取退化图像的3×3窗口作为输入
%             P_Matrix(1,t) = image_degraded(i-1,j-1);
%             P_Matrix(2,t) = image_degraded(i-1,j);
%             P_Matrix(3,t) = image_degraded(i-1,j+1);
%             P_Matrix(4,t) = image_degraded(i,j-1);
%             P_Matrix(5,t) = image_degraded(i,j);
%             P_Matrix(6,t) = image_degraded(i,j+1);
%             P_Matrix(7,t) = image_degraded(i+1,j-1);
%             P_Matrix(8,t) = image_degraded(i+1,j);
%             P_Matrix(9,t) = image_degraded(i+1,j+1);
%             % 提取清晰图像的中心像素作为目标
%             T_Matrix(1,t) = image_clear(i,j);
%             t = t + 1;
%         end
%     end
% end





