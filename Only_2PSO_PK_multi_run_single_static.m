%% 新版论文需要。重要文件，2025年国庆后修改，PSO和IPSO对比，直接读取结果，对比PSO_improved_exp算法——幂次递减惯性权重
% PSO与IPSO对比代码（适配Figure 6绘制需求，格式优化版）
%% Paper Static Demonstration -- Comparison of PSO and IPSO Algorithms
%% This code compares PSO and IPSO algorithms for multiple rounds of testing. Maxit specifies the number of comparisons (100 in the paper)
%% The image size is 90×90. The two optimization algorithms are run 100 times respectively, and the optimal fitness values are used for comparison
%% Result data (e.g., best90_r100_lenna.mat) is saved in the "result" folder
%% Basic data for Table 2, Figure 6 (Adaptation Curves Comparison) are generated from this file
%% Figure 6 uses the running results of Lenna.tif for visualization

% 清空环境变量
clear
close all
clc

% 保存结果
mat_name = "ipso\ipso_pk_20251016v3ok\best90_r100_100_2025-10-16.mat";
load(mat_name);

%% 4. 结果统计与Figure 6绘制（适配论文Figure 6：适应度曲线对比）
% 4.1 统计100轮中最优的一次结果（消除随机误差，论文常用策略）
[PSO_best_overall, PSO_best_idx] = min(PSO_bestfitness);
[IPSO_best_overall, IPSO_best_idx] = min(IPSO_bestfitness);

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


%% 附录：自定义函数（需与主代码放在同一目录下）
% 1. generate_training_data.m：生成BPNN训练数据（3×3滑动窗口）
function [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_clear, inputnum)
    [h, w] = size(image_degraded);
    data_len = (h-2)*(w-2);  % 滑动窗口数量（边缘像素不参与，避免边界效应）
    P_Matrix = zeros(inputnum, data_len);  % 输入矩阵（9×N）
    T_Matrix = zeros(1, data_len);         % 目标矩阵（1×N，清晰图像中心像素）
    t = 1;
    
    % 遍历图像，提取3×3窗口
    for i = 2:h-1
        for j = 2:w-1
            % 提取退化图像的3×3窗口作为输入
            P_Matrix(1,t) = image_degraded(i-1,j-1);
            P_Matrix(2,t) = image_degraded(i-1,j);
            P_Matrix(3,t) = image_degraded(i-1,j+1);
            P_Matrix(4,t) = image_degraded(i,j-1);
            P_Matrix(5,t) = image_degraded(i,j);
            P_Matrix(6,t) = image_degraded(i,j+1);
            P_Matrix(7,t) = image_degraded(i+1,j-1);
            P_Matrix(8,t) = image_degraded(i+1,j);
            P_Matrix(9,t) = image_degraded(i+1,j+1);
            % 提取清晰图像的中心像素作为目标
            T_Matrix(1,t) = image_clear(i,j);
            t = t + 1;
        end
    end
end





