% IPSO扰动参数调优实验
% 用于确定最优的perturb_trigger_ratio和perturb_std参数
%********************************************************************
% 生成：Figure X - 参数性能热力图，Table X - 参数调优结果
%********************************************************************

% 清空环境变量
clear
close all
clc

% 记录总开始时间
total_start_time = tic;

%% 0. 动态获取图像文件
script_path = fileparts(mfilename('fullpath'));
pic_dir = fullfile(script_path, 'ipso', 'valid');

if ~exist(pic_dir, 'dir')
    fprintf('Default image directory does not exist: %s\n', pic_dir);
    pic_dir = uigetdir(pwd, 'Please select the directory containing test images');
    if pic_dir == 0
        error('User canceled directory selection');
    end
end

% 获取图像文件
supported_formats = {'*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp'};
image_files = [];
for i = 1:length(supported_formats)
    current_files = dir(fullfile(pic_dir, supported_formats{i}));
    current_files = current_files(~[current_files.isdir]);
    image_files = [image_files; current_files];
end

if isempty(image_files)
    error('No image files found in directory %s', pic_dir);
end

all_sim_picname = {image_files.name};
[all_sim_picname, sort_idx] = sort(all_sim_picname);
num_images = min(10, length(all_sim_picname)); % 使用5张图进行参数调优

fprintf('Found %d image files, using %d for parameter tuning\n', length(all_sim_picname), num_images);


%% 1. 定义要测试的参数组合（必须包含无扰动组）
% 触发比例 x 扰动强度 = 参数组合
param_combinations = [
    % trigger_ratio, std_dev, 组合ID
    1.0, 0.00, 1;    % No perturbation (baseline)
    0.3, 0.05, 2;    % Early weak perturbation
    0.5, 0.05, 3;    % Mid weak perturbation
    0.7, 0.05, 4;    % Late weak perturbation
    0.3, 0.10, 5;    % Early medium perturbation
    0.5, 0.10, 6;    % Mid medium perturbation
    0.7, 0.10, 7;    % Late medium perturbation (original)
    0.5, 0.15, 8;    % Mid strong perturbation
    0.7, 0.15, 9;    % Late strong perturbation
    0.0, 0.20, 10;   % Full strong perturbation
    1.0, 0.10, 11;   % No trigger with strength
];

num_param_combos = size(param_combinations, 1);

% 扩展结果存储数组
% 列定义：1:参数ID, 2:触发比例, 3:扰动强度, 4:最佳适应度均值, 5:最佳适应度改进率, 
% 6:平均适应度均值, 7:平均适应度改进率, 8:中位数适应度均值, 9:中位数适应度改进率,
% 10:标准差均值, 11:运行时间均值
param_results = zeros(num_param_combos, 11);
param_labels = {'Param_ID', 'Trigger_Ratio', 'Perturb_Std', 'BestFitness_Mean', 'IR_Best(%)', ...
                'MeanFitness_Mean', 'IR_Mean(%)', 'MedianFitness_Mean', 'IR_Median(%)', ...
                'Std_Mean', 'Time_Mean(min)'};

%% 2. 全局参数设置
% 为了节省时间，参数调优实验使用较少的运行次数
Maxit = 30;          % 每组参数运行30次（平衡统计可靠性和时间）
picsize = [90, 90];  % 图像尺寸
gauss_kernel_size = 9;
gauss_sigma = 1;
salt_pepper_density = 0.02;

% BPNN网络结构
inputnum = 9;
hiddennum = 9;
outputnum = 1;
numsum = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;

% PSO/IPSO算法参数
sizepop = 50;
maxgen = 50;
c1 = 1.5;
c2 = 1.5;
w_init = 0.9;
w_final = 0.3;
v_max = 0.5;
v_min = -0.5;
pos_max = 1;
pos_min = -1;
p = 1.5;  % 幂指数固定为1.5

fprintf('Parameter tuning experiment configuration:\n');
fprintf('  Images: %d\n', num_images);
fprintf('  Runs per parameter: %d\n', Maxit);
fprintf('  Parameter combinations: %d\n', num_param_combos);
fprintf('  Total runs: %d\n', num_images * num_param_combos * Maxit);
fprintf('  Estimated time: %.1f hours\n\n', num_images * num_param_combos * Maxit * 0.2 / 60);

%% 3. 创建进度显示图
progress_fig = figure('Position', [50, 50, 1000, 700], 'Name', 'Parameter Tuning Progress', 'NumberTitle', 'off');

% 子图1：总体进度
subplot(3,3,1);
overall_progress_bar = barh(1, 0, 'FaceColor', [0.2 0.6 1]);
xlabel('Overall Progress (%)');
title('Overall Progress');
xlim([0 100]);
grid on;

% 子图2：各参数组合进度
subplot(3,3,2);
param_progress_bars = barh(1:num_param_combos, zeros(num_param_combos,1));
xlabel('Completion (%)');
ylabel('Parameter Combination');
title('Progress by Parameter');
xlim([0 100]);
set(gca, 'YTick', 1:num_param_combos, 'YTickLabel', arrayfun(@(x) sprintf('P%d', x), 1:num_param_combos, 'UniformOutput', false));
grid on;

% 子图3：各图像进度
subplot(3,3,3);
image_progress_bars = barh(1:num_images, zeros(num_images,1));
xlabel('Completion (%)');
ylabel('Image');
title('Progress by Image');
xlim([0 100]);
grid on;

% 子图4：当前最佳参数组合
subplot(3,3,4);
best_param_text = text(0.1, 0.5, 'Calculating...', 'FontSize', 12);
axis off;
title('Current Best Parameters');

% 子图5：运行时间监控
subplot(3,3,5);
time_plot = plot(1:10, zeros(10,1), 'b-o', 'LineWidth', 1.5);
xlabel('Last 10 runs');
ylabel('Time (min)');
title('Running Time');
grid on;

% 子图6：性能改进热力图预览
subplot(3,3,6);
improvement_preview = imagesc(zeros(num_param_combos, 1));
colorbar;
xlabel('Images');
ylabel('Parameters');
title('Improvement Rate Preview');
colormap(jet);

% 子图7：已完成的实验计数
subplot(3,3,7:9);
total_experiments = num_images * num_param_combos * Maxit;
completed_text = text(0.1, 0.7, sprintf('Completed: 0/%d (0.0%%)', total_experiments), 'FontSize', 14);
remaining_time_text = text(0.1, 0.4, 'Estimated remaining: Calculating...', 'FontSize', 12);
current_param_text = text(0.1, 0.1, 'Current: None', 'FontSize', 12);
axis off;

drawnow;

%% 4. 主实验循环
fprintf('Starting parameter tuning experiment...\n');
fprintf('========================================\n');

% 初始化结果存储
all_results = struct();

% 固定随机种子确保可比性
rng_seed_base = 20240101;

% 初始化统计变量
completed_count = 0;
start_time = tic;
time_records = zeros(10,1);

for img_idx = 1:num_images
    fprintf('\nProcessing Image %d/%d: %s\n', img_idx, num_images, all_sim_picname{img_idx});
    
    % 更新图像进度
    image_progress_bars.YData(img_idx) = 0;
    drawnow;
    
    % 构建图像路径并读取
    picname = fullfile(pic_dir, all_sim_picname{img_idx});
    if ~exist(picname, 'file')
        fprintf('Warning: File does not exist: %s\n', picname);
        continue;
    end
    
    % 读取并处理图像
    try
        image_orgin = imread(picname);
        if size(image_orgin, 3) == 3
            image_orgin = rgb2gray(image_orgin);
        end
        image_resized = imresize(image_orgin, picsize);
        image_resized = double(image_resized) / 256;
        
        % 添加高斯模糊
        w_gauss = fspecial('gaussian', gauss_kernel_size, gauss_sigma);
        image_blurred = imfilter(image_resized, w_gauss, 'replicate');
        image_degraded = image_blurred; % 不加噪声
        
        % 生成BPNN训练数据
        [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_resized, inputnum);
        
        % 初始化BPNN网络
        net.trainParam.epochs = 1000;
        net.trainParam.lr = 0.1;
        net.trainParam.goal = 1e-5;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        net = newff(P_Matrix, T_Matrix, hiddennum);
        
        % 定义适应度函数
        fobj = @(x) cal_fitness(x, inputnum, hiddennum, outputnum, net, P_Matrix, T_Matrix);
        
    catch ME
        fprintf('Error processing image %s: %s\n', all_sim_picname{img_idx}, ME.message);
        continue;
    end
    
    % 为当前图像初始化结果存储
    img_results = struct();
    img_results.best_fitness = zeros(num_param_combos, Maxit);
    img_results.mean_fitness = zeros(num_param_combos, 1);
    img_results.median_fitness = zeros(num_param_combos, 1);
    img_results.std_fitness = zeros(num_param_combos, 1);
    img_results.mean_time = zeros(num_param_combos, 1);
    img_results.improvement_best = zeros(num_param_combos, 1);
    img_results.improvement_mean = zeros(num_param_combos, 1);
    img_results.improvement_median = zeros(num_param_combos, 1);
    
    for param_idx = 1:num_param_combos
        trigger_ratio = param_combinations(param_idx, 1);
        std_dev = param_combinations(param_idx, 2);
        param_id = param_combinations(param_idx, 3);
        
        fprintf('  Testing parameter combo %d/%d (ratio=%.1f, std=%.2f): ', ...
                param_idx, num_param_combos, trigger_ratio, std_dev);
        
        % 更新参数进度
        param_progress_bars.YData(param_idx) = (param_idx-1)/num_param_combos * 100;
        current_param_text.String = sprintf('Current: Image %d, Param P%d (%.1f, %.2f)', ...
                                           img_idx, param_id, trigger_ratio, std_dev);
        drawnow;
        
        % 计时开始
        param_start_time = tic;
        
        % 运行多次实验
        param_fitness = zeros(Maxit, 1);
        
        for run_idx = 1:Maxit
            % 设置随机种子确保可比性
            current_seed = rng_seed_base + img_idx*1000 + param_idx*100 + run_idx;
            rng(current_seed, 'twister');
            
            % 使用最简单的扰动策略运行IPSO
            [~, best_fitness, ~] = IPSO_simple_perturb_tuning(...
                sizepop, maxgen, numsum, fobj, c1, c2, ...
                w_init, w_final, v_max, v_min, pos_max, pos_min, ...
                trigger_ratio, std_dev, p);
            
            param_fitness(run_idx) = best_fitness;
            
            % 更新进度
            completed_count = completed_count + 1;
            overall_progress = completed_count / total_experiments * 100;
            
            % 更新进度显示
            overall_progress_bar.YData = overall_progress;
            
            % 每5次运行更新一次进度
            if mod(run_idx, 5) == 0
                % 更新图像进度
                image_progress = (param_idx/num_param_combos + (run_idx/Maxit)/num_param_combos) * 100;
                image_progress_bars.YData(img_idx) = image_progress;
                
                % 更新参数进度
                param_progress = ((img_idx-1)/num_images + (param_idx-1)/(num_images*num_param_combos) + ...
                                 (run_idx/Maxit)/(num_images*num_param_combos)) * 100;
                param_progress_bars.YData(param_idx) = param_progress;
                
                % 更新已完成计数
                completed_text.String = sprintf('Completed: %d/%d (%.1f%%)', ...
                                               completed_count, total_experiments, overall_progress);
                
                % 更新剩余时间估计
                elapsed_time = toc(start_time);
                if completed_count > 0
                    time_per_experiment = elapsed_time / completed_count;
                    remaining_time = (total_experiments - completed_count) * time_per_experiment;
                    remaining_time_text.String = sprintf('Estimated remaining: %.1f minutes', remaining_time/60);
                    
                    % 记录运行时间
                    time_records = circshift(time_records, -1);
                    time_records(end) = time_per_experiment * 60; % 转换为分钟
                    time_plot.YData = time_records;
                end
                
                drawnow;
            end
            
            % 显示进度点
            if mod(run_idx, 5) == 0
                fprintf('.');
            end
        end
        
        % 计算统计指标
        img_results.best_fitness(param_idx, :) = param_fitness;
        img_results.mean_fitness(param_idx) = mean(param_fitness);
        img_results.median_fitness(param_idx) = median(param_fitness);
        img_results.std_fitness(param_idx) = std(param_fitness);
        img_results.mean_time(param_idx) = toc(param_start_time) / Maxit / 60; % 转换为分钟
        
        % 显示本次结果
        fprintf(' Done! Mean: %.3f ± %.3f\n', ...
                img_results.mean_fitness(param_idx), img_results.std_fitness(param_idx));
        
        % 更新预览热力图
        if img_idx == 1
            improvement_preview.CData = img_results.mean_fitness;
        end
    end
    
    % 计算改进率（相对于无扰动基准）
    baseline_idx = find(param_combinations(:,1) == 1.0 & param_combinations(:,2) == 0.00, 1);
    if ~isempty(baseline_idx)
        baseline_best = min(img_results.best_fitness(baseline_idx, :));
        baseline_mean = img_results.mean_fitness(baseline_idx);
        baseline_median = img_results.median_fitness(baseline_idx);
        
        for param_idx = 1:num_param_combos
            % 最佳适应度改进率
            img_results.improvement_best(param_idx) = ...
                (baseline_best - min(img_results.best_fitness(param_idx, :))) ...
                / baseline_best * 100;
            
            % 平均适应度改进率
            img_results.improvement_mean(param_idx) = ...
                (baseline_mean - img_results.mean_fitness(param_idx)) ...
                / baseline_mean * 100;
            
            % 中位数适应度改进率
            img_results.improvement_median(param_idx) = ...
                (baseline_median - img_results.median_fitness(param_idx)) ...
                / baseline_median * 100;
        end
        
        % 找出当前最佳参数
        [~, best_param_for_image] = max(img_results.improvement_mean);
        best_trigger = param_combinations(best_param_for_image, 1);
        best_std = param_combinations(best_param_for_image, 2);
        best_improvement = img_results.improvement_mean(best_param_for_image);
        
        best_param_text.String = sprintf('Current best:\nRatio: %.1f\nStd: %.2f\nImprovement: %.1f%%', ...
                                        best_trigger, best_std, best_improvement);
    end
    
    % 存储当前图像结果
    all_results.(sprintf('image_%d', img_idx)) = img_results;
    
    % 更新图像进度为100%
    image_progress_bars.YData(img_idx) = 100;
    
    fprintf('Image %d completed. Best improvement: %.2f%%\n', img_idx, max(img_results.improvement_mean));
end

%% 5. 汇总所有结果
fprintf('\n\nAll experiments completed! Generating summary results...\n');

% 计算各参数组合在所有图像上的平均性能
param_best_mean = zeros(num_param_combos, 1);
param_improvement_best_mean = zeros(num_param_combos, 1);
param_improvement_mean_mean = zeros(num_param_combos, 1);
param_improvement_median_mean = zeros(num_param_combos, 1);
param_std_mean = zeros(num_param_combos, 1);
param_time_mean = zeros(num_param_combos, 1);

for param_idx = 1:num_param_combos
    all_best_vals = [];
    all_improvement_best_vals = [];
    all_improvement_mean_vals = [];
    all_improvement_median_vals = [];
    all_std_vals = [];
    all_time_vals = [];
    
    for img_idx = 1:num_images
        img_field = sprintf('image_%d', img_idx);
        if isfield(all_results, img_field)
            all_best_vals = [all_best_vals; all_results.(img_field).mean_fitness(param_idx)];
            all_improvement_best_vals = [all_improvement_best_vals; all_results.(img_field).improvement_best(param_idx)];
            all_improvement_mean_vals = [all_improvement_mean_vals; all_results.(img_field).improvement_mean(param_idx)];
            all_improvement_median_vals = [all_improvement_median_vals; all_results.(img_field).improvement_median(param_idx)];
            all_std_vals = [all_std_vals; all_results.(img_field).std_fitness(param_idx)];
            all_time_vals = [all_time_vals; all_results.(img_field).mean_time(param_idx)];
        end
    end
    
    param_best_mean(param_idx) = mean(all_best_vals);
    param_improvement_best_mean(param_idx) = mean(all_improvement_best_vals);
    param_improvement_mean_mean(param_idx) = mean(all_improvement_mean_vals);
    param_improvement_median_mean(param_idx) = mean(all_improvement_median_vals);
    param_std_mean(param_idx) = mean(all_std_vals);
    param_time_mean(param_idx) = mean(all_time_vals);
end

% 填充参数结果表
for param_idx = 1:num_param_combos
    param_results(param_idx, :) = [
        param_idx, ...  % Param_ID
        param_combinations(param_idx, 1), ...  % Trigger_Ratio
        param_combinations(param_idx, 2), ...  % Perturb_Std
        param_best_mean(param_idx), ...  % BestFitness_Mean
        param_improvement_best_mean(param_idx), ...  % IR_Best(%)
        param_best_mean(param_idx), ...  % MeanFitness_Mean (与Best相同)
        param_improvement_mean_mean(param_idx), ...  % IR_Mean(%)
        param_best_mean(param_idx), ...  % MedianFitness_Mean (与Best相同)
        param_improvement_median_mean(param_idx), ...  % IR_Median(%)
        param_std_mean(param_idx), ...  % Std_Mean
        param_time_mean(param_idx)  % Time_Mean(min)
    ];
end

%% 6. 生成可视化结果
fprintf('Generating visualization results...\n');

% 6.1 性能热力图
fig1 = figure('Position', [100, 100, 1200, 800], 'Name', 'Parameter Performance Heatmap', 'NumberTitle', 'off');

% 准备热力图数据
heatmap_data = zeros(num_images, num_param_combos);
for img_idx = 1:num_images
    img_field = sprintf('image_%d', img_idx);
    if isfield(all_results, img_field)
        heatmap_data(img_idx, :) = all_results.(img_field).improvement_mean';
    end
end

% 子图1：热力图
subplot(2,3,1);
imagesc(heatmap_data);
colorbar;
xlabel('Parameter Combination');
ylabel('Test Image');
title('Average Improvement Rate (%) Heatmap');
set(gca, 'XTick', 1:num_param_combos, ...
         'XTickLabel', arrayfun(@(x) sprintf('P%d', x), 1:num_param_combos, 'UniformOutput', false));
set(gca, 'YTick', 1:num_images);
colormap(jet);

% 添加数值标注
for i = 1:num_images
    for j = 1:num_param_combos
        if heatmap_data(i,j) >= 0
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.1f', heatmap_data(i,j)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, 'Color', text_color);
    end
end

% 子图2：最佳改进率排名
subplot(2,3,2);
[~, sorted_idx] = sort(param_improvement_mean_mean, 'descend');
bar_colors = zeros(num_param_combos, 3);
for i = 1:num_param_combos
    if param_improvement_mean_mean(sorted_idx(i)) > 0
        bar_colors(i, :) = [0.2, 0.6, 0.2]; % 绿色表示提升
    else
        bar_colors(i, :) = [0.8, 0.2, 0.2]; % 红色表示下降
    end
end
bar_handle = bar(1:num_param_combos, param_improvement_mean_mean(sorted_idx));
set(bar_handle, 'FaceColor', 'flat', 'CData', bar_colors);
xlabel('Parameter Rank');
ylabel('Average Improvement Rate (%)');
title('Parameter Performance Ranking');
set(gca, 'XTick', 1:num_param_combos, ...
         'XTickLabel', arrayfun(@(x) sprintf('P%d', sorted_idx(x)), 1:num_param_combos, 'UniformOutput', false));
grid on;

% 子图3：三种改进率对比
subplot(2,3,3);
hold on;
plot(1:num_param_combos, param_improvement_best_mean, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Best Improvement');
plot(1:num_param_combos, param_improvement_mean_mean, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Mean Improvement');
plot(1:num_param_combos, param_improvement_median_mean, 'g-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Median Improvement');
hold off;
xlabel('Parameter Combination');
ylabel('Improvement Rate (%)');
title('Three Types of Improvement Rates');
set(gca, 'XTick', 1:num_param_combos, ...
         'XTickLabel', arrayfun(@(x) sprintf('P%d', x), 1:num_param_combos, 'UniformOutput', false));
legend('Location', 'best');
grid on;

% 子图4：稳定性分析（标准差）
subplot(2,3,4);
errorbar(1:num_param_combos, param_improvement_mean_mean, param_std_mean, 'o-', ...
         'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
xlabel('Parameter Combination');
ylabel('Improvement Rate ± Std (%)');
title('Stability Analysis');
set(gca, 'XTick', 1:num_param_combos, ...
         'XTickLabel', arrayfun(@(x) sprintf('P%d', x), 1:num_param_combos, 'UniformOutput', false));
grid on;

% 子图5：运行时间对比
subplot(2,3,5);
bar(1:num_param_combos, param_time_mean, 'FaceColor', [0.8, 0.6, 0.2]);
xlabel('Parameter Combination');
ylabel('Average Running Time (min)');
title('Computational Cost');
set(gca, 'XTick', 1:num_param_combos, ...
         'XTickLabel', arrayfun(@(x) sprintf('P%d', x), 1:num_param_combos, 'UniformOutput', false));
grid on;

% 子图6：最优参数组合推荐
subplot(2,3,6);
[best_improvement, best_param_idx] = max(param_improvement_mean_mean);
best_trigger = param_combinations(best_param_idx, 1);
best_std = param_combinations(best_param_idx, 2);

% 显示推荐参数
text(0.1, 0.8, 'Recommended Parameters:', 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.6, sprintf('Trigger Ratio: %.1f', best_trigger), 'FontSize', 12);
text(0.1, 0.5, sprintf('Perturbation Std: %.2f', best_std), 'FontSize', 12);
text(0.1, 0.4, sprintf('Average Improvement: %.2f%%', best_improvement), 'FontSize', 12);

% 与原始参数比较
original_idx = find(param_combinations(:,1) == 0.7 & param_combinations(:,2) == 0.10, 1);
if ~isempty(original_idx) && original_idx ~= best_param_idx
    original_improvement = param_improvement_mean_mean(original_idx);
    improvement_difference = best_improvement - original_improvement;
    text(0.1, 0.2, sprintf('vs Original (0.7, 0.10): +%.2f%%', improvement_difference), ...
         'FontSize', 11, 'Color', 'r');
end
axis off;

% 保存热力图
saveas(fig1, 'ipso/Parameter_Performance_Heatmap.png');
saveas(fig1, 'ipso/Parameter_Performance_Heatmap.fig');
print('ipso/Parameter_Performance_Heatmap.eps', '-depsc', '-r600', '-vector');

% 6.2 详细对比图
fig2 = figure('Position', [200, 200, 1400, 600], 'Name', 'Detailed Parameter Comparison', 'NumberTitle', 'off');

% 创建参数组合名称
param_names = cell(num_param_combos, 1);
for i = 1:num_param_combos
    param_names{i} = sprintf('P%d(%.1f,%.2f)', i, param_combinations(i,1), param_combinations(i,2));
end

% 子图1：各图像上的改进率分布
subplot(1,3,1);
boxplot_data = [];
group_labels = [];
for param_idx = 1:num_param_combos
    param_data = [];
    for img_idx = 1:num_images
        img_field = sprintf('image_%d', img_idx);
        if isfield(all_results, img_field)
            param_data = [param_data; all_results.(img_field).improvement_mean(param_idx)];
        end
    end
    boxplot_data = [boxplot_data; param_data];
    group_labels = [group_labels; repmat(param_idx, length(param_data), 1)];
end
boxplot(boxplot_data, group_labels, 'Labels', param_names);
xlabel('Parameter Combination');
ylabel('Improvement Rate (%)');
title('Improvement Rate Distribution Across Images');
set(gca, 'XTickLabelRotation', 45);
grid on;

% 子图2：参数空间3D图
subplot(1,3,2);
[X, Y] = meshgrid(unique(param_combinations(:,1)), unique(param_combinations(:,2)));
Z = griddata(param_combinations(:,1), param_combinations(:,2), param_improvement_mean_mean, X, Y);

surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
hold on;
scatter3(param_combinations(:,1), param_combinations(:,2), param_improvement_mean_mean, ...
         100, 'r', 'filled', 'MarkerEdgeColor', 'k');
hold off;

xlabel('Trigger Ratio');
ylabel('Perturbation Std');
zlabel('Average Improvement Rate (%)');
title('Parameter Space Performance');
colorbar;
view(-45, 30);
grid on;

% 子图3：时间-性能权衡图
subplot(1,3,3);
scatter(param_time_mean, param_improvement_mean_mean, 100, 'filled');
xlabel('Average Running Time (min)');
ylabel('Average Improvement Rate (%)');
title('Time-Performance Trade-off');
grid on;

% 添加参数标签
for i = 1:num_param_combos
    text(param_time_mean(i), param_improvement_mean_mean(i), ...
         sprintf('P%d', i), 'FontSize', 9, 'HorizontalAlignment', 'center');
end

% 添加帕累托前沿
hold on;
[~, sorted_by_time] = sort(param_time_mean);
pareto_indices = sorted_by_time(1);
for i = 2:length(sorted_by_time)
    if param_improvement_mean_mean(sorted_by_time(i)) > max(param_improvement_mean_mean(pareto_indices))
        pareto_indices = [pareto_indices; sorted_by_time(i)];
    end
end
plot(param_time_mean(pareto_indices), param_improvement_mean_mean(pareto_indices), ...
     'r--', 'LineWidth', 2, 'DisplayName', 'Pareto Frontier');
hold off;
legend('Location', 'best');

% 保存详细对比图
saveas(fig2, 'ipso/Detailed_Parameter_Comparison.png');
saveas(fig2, 'ipso/Detailed_Parameter_Comparison.fig');
print('ipso/Detailed_Parameter_Comparison.eps', '-depsc', '-r600', '-painters');

%% 7. 保存结果到文件
fprintf('Saving results to files...\n');

% 7.1 保存参数结果到Excel
if ~exist('ipso', 'dir')
    mkdir('ipso');
end

param_table_filename = 'ipso/Parameter_Tuning_Results.xlsx';

% 写入列标题
writecell(param_labels, param_table_filename, 'Sheet', 1, 'Range', 'A1');

% 格式化数据
param_results_formatted = param_results;
param_results_formatted(:, 4:10) = round(param_results_formatted(:, 4:10), 3);
param_results_formatted(:, 11) = round(param_results_formatted(:, 11), 2);

% 写入数据
writematrix(param_results_formatted, param_table_filename, 'Sheet', 1, 'Range', 'A2');

% 添加参数描述工作表
param_descriptions_excel = cell(num_param_combos+1, 4);
param_descriptions_excel{1,1} = 'Param_ID';
param_descriptions_excel{1,2} = 'Trigger_Ratio';
param_descriptions_excel{1,3} = 'Perturb_Std';
param_descriptions_excel{1,4} = 'Description';

% 创建描述文本数组（与参数对应）
param_desc_texts = {
    'No perturbation (baseline)';
    'Early weak perturbation';
    'Mid weak perturbation';
    'Late weak perturbation';
    'Early medium perturbation';
    'Mid medium perturbation';
    'Late medium perturbation (original)';
    'Mid strong perturbation';
    'Late strong perturbation';
    'Full strong perturbation';
    'No trigger with strength';
};

for i = 1:num_param_combos
    param_descriptions_excel{i+1,1} = param_combinations(i,3);  % Param_ID
    param_descriptions_excel{i+1,2} = param_combinations(i,1);  % Trigger_Ratio
    param_descriptions_excel{i+1,3} = param_combinations(i,2);  % Perturb_Std
    param_descriptions_excel{i+1,4} = param_desc_texts{i};      % Description
end

writecell(param_descriptions_excel, param_table_filename, 'Sheet', 2, 'Range', 'A1');

% 添加推荐参数工作表
recommendation = {
    'Parameter Tuning Recommendation', '', '';
    'Best Parameters Found:', '', '';
    sprintf('Trigger Ratio: %.1f', best_trigger), '', '';
    sprintf('Perturbation Std: %.2f', best_std), '', '';
    sprintf('Average Improvement Rate: %.2f%%', best_improvement), '', '';
    '', '', '';
    'Comparison with Original (0.7, 0.10):', '', '';
    };
if ~isempty(original_idx)
    recommendation{10,1} = sprintf('Original Improvement: %.2f%%', param_improvement_mean_mean(original_idx));
    recommendation{11,1} = sprintf('Improvement Difference: %.2f%%', best_improvement - param_improvement_mean_mean(original_idx));
end

writecell(recommendation, param_table_filename, 'Sheet', 3, 'Range', 'A1');

% 7.2 保存详细结果到文本文件
summary_filename = 'ipso/Parameter_Tuning_Summary.txt';
fid_summary = fopen(summary_filename, 'w', 'n', 'UTF-8');

if fid_summary ~= -1
    fprintf(fid_summary, 'IPSO Parameter Tuning Experiment Summary\n');
    fprintf(fid_summary, '=========================================\n\n');
    fprintf(fid_summary, 'Experiment Date: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid_summary, 'Total Running Time: %.2f minutes\n', toc(total_start_time)/60);
    fprintf(fid_summary, 'Images Processed: %d\n', num_images);
    fprintf(fid_summary, 'Parameter Combinations Tested: %d\n', num_param_combos);
    fprintf(fid_summary, 'Experiments per Combination: %d\n', Maxit);
    fprintf(fid_summary, 'Total Experiments: %d\n\n', total_experiments);
    
    fprintf(fid_summary, 'RECOMMENDED PARAMETERS:\n');
    fprintf(fid_summary, '----------------------\n');
    fprintf(fid_summary, 'Trigger Ratio: %.1f\n', best_trigger);
    fprintf(fid_summary, 'Perturbation Std: %.2f\n', best_std);
    fprintf(fid_summary, 'Average Improvement Rate: %.2f%%\n\n', best_improvement);
    
    fprintf(fid_summary, 'TOP 5 PARAMETER COMBINATIONS:\n');
    fprintf(fid_summary, '-----------------------------\n');
    fprintf(fid_summary, 'Rank\tParam\tTrigger\tStd\tImprovement(%%)\n');
    fprintf(fid_summary, '----\t-----\t------\t---\t--------------\n');
    
    [sorted_improvements, sorted_indices] = sort(param_improvement_mean_mean, 'descend');
    for rank = 1:min(5, num_param_combos)
        idx = sorted_indices(rank);
        fprintf(fid_summary, '%d\tP%d\t%.1f\t%.2f\t%.2f\n', ...
                rank, idx, param_combinations(idx,1), param_combinations(idx,2), sorted_improvements(rank));
    end
    
    fclose(fid_summary);
end

%% 8. 显示最终结果
fprintf('\n=====================================================\n');
fprintf('PARAMETER TUNING EXPERIMENT COMPLETED!\n');
fprintf('=====================================================\n');
fprintf('Total time: %.2f minutes\n', toc(total_start_time)/60);
fprintf('Recommended parameters:\n');
fprintf('  Trigger Ratio: %.1f\n', best_trigger);
fprintf('  Perturbation Std: %.2f\n', best_std);
fprintf('  Average Improvement: %.2f%%\n', best_improvement);
fprintf('\nResults saved to:\n');
fprintf('  • ipso/Parameter_Tuning_Results.xlsx\n');
fprintf('  • ipso/Parameter_Performance_Heatmap.*\n');
fprintf('  • ipso/Detailed_Parameter_Comparison.*\n');
fprintf('  • ipso/Parameter_Tuning_Summary.txt\n');
fprintf('=====================================================\n');

%% 9. 关闭进度图
close(progress_fig);

%% 10. 最简单的扰动策略函数（需要添加到文件末尾）
function [bestchrom, bestfitness, trace_best] = IPSO_simple_perturb_tuning(...
    sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, ...
    v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p)
    
    % 初始化粒子群
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);
    
    % 初始化个体最优与全局最优
    pbest_pos = particles_pos;
    pbest_fit = zeros(sizepop, 1);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));
    end
    [gbest_fit, gbest_idx] = min(pbest_fit);
    gbest_pos = pbest_pos(gbest_idx, :);
    
    % 记录适应度曲线
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;
    
    % IPSO迭代主循环
    for gen = 2:maxgen
        % 幂次递减惯性权重更新
        normalized_progress = (gen - 1) / (maxgen - 1);
        w = w_final + (w_init - w_final) * (1 - normalized_progress)^p;
        
        % 遍历所有粒子
        for i = 1:sizepop
            % 速度更新
            r1 = rand();
            r2 = rand();
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2*r2*(gbest_pos - particles_pos(i,:));
            
            % 速度约束
            particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
            
            % 位置更新
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            
            % 最简单扰动策略
            if gen > perturb_trigger_ratio * maxgen
                perturb = perturb_std * randn(1, numsum);
                particles_pos(i,:) = particles_pos(i,:) + perturb;
                particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            end
            
            % 更新个体最优
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
            end
        end
        
        % 更新全局最优
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 记录当前代最优适应度
        trace_best(gen) = gbest_fit;
    end
    
    % 输出结果
    bestchrom = gbest_pos;
    bestfitness = gbest_fit;
end