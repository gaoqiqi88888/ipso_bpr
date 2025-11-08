%********************************************************************
%F6论文适应度曲线，F7最佳适应度100次对比，T3每张图的适应度最优值和均值对比
%********************************************************************
%文件名可以修改，不限于10次

% 清空环境变量
clear
close all
clc

t1_total = clock;  % 记录10张图总运行时间起点
%% 0. 多图测试初始化（10张图列表+Table 3数据存储）
% 10张测试图像列表（按需求修改路径）
all_sim_picname = ["i01.tif", "i02.tif", "i03.tif", "i04.tif", "i05.tif", ...
                   "i06.tif", "i07.tif", "i08.tif", "i09.tif", "i10.tif"];

% all_sim_picname = ["i01.tif", "i02.tif"];

pic_path_prefix = 'img/';  % 图像文件夹路径（根据实际路径调整）
num_images = length(all_sim_picname);  % 图像数量（10张）

% Table 3数据存储数组（列：图像序号, PSO最优适应度, PSO平均适应度, PSO平均时间, 
%                          IPSO最优适应度, IPSO平均适应度, IPSO平均时间, 提升率(%)）
Table3_data = zeros(num_images, 8);  
col_labels = {'Image_ID', 'PSO_Best_Fitness', 'PSO_Mean_Fitness', 'PSO_Mean_Time(min)', ...
              'IPSO_Best_Fitness', 'IPSO_Mean_Fitness', 'IPSO_Mean_Time(min)', 'Improvement_Rate(%)'};

%% 1. 全局参数设置（与原单图代码一致，保证对比公平性）
Maxit = 2;                % 算法对比轮次（论文要求100次消除随机误差）
% 1.1 图像与数据预处理参数
picsize = [90, 90];         % 图像尺寸（论文设定90×90加速收敛）
gauss_kernel_size = 9;      % 高斯模糊核大小
gauss_sigma = 1;            % 高斯模糊标准差
salt_pepper_density = 0.02; % 椒盐噪声密度（混合退化场景）

% 1.2 BPNN网络结构
inputnum = 9;               % 输入节点数（3×3滑动窗口）
hiddennum = 9;              % 隐藏层节点数
outputnum = 1;              % 输出节点数（单像素预测）
numsum = inputnum*hiddennum + hiddennum + hiddennum*outputnum + outputnum;  % PSO/IPSO优化维度

% 1.3 PSO与IPSO算法参数
sizepop = 100;              % 种群/粒子群规模
maxgen = 100;               % 最大迭代次数

% 1.4 PSO核心控制参数
c1 = 1.5;                   % 个体学习因子
c2 = 1.5;                   % 全局学习因子
w_init = 0.9;               % IPSO初始惯性权重
w_final = 0.3;              % IPSO最终惯性权重
v_max = 0.5;                % 粒子最大速度
v_min = -0.5;               % 粒子最小速度
pos_max = 1;                % 粒子位置上限
pos_min = -1;               % 粒子位置下限
perturb_trigger_ratio = 0.7;% IPSO高斯扰动触发阈值
perturb_std = 0.1;          % 高斯扰动标准差

%% 2. 10张图循环测试（核心新增逻辑）
for img_idx = 1:num_images
    fprintf('=====================================================\n');
    fprintf('==================== 第%d张图测试（%s） ====================\n', img_idx, all_sim_picname{img_idx});
    fprintf('=====================================================\n');
    
    t1_single = clock;  % 记录单张图运行时间起点
    %% 2.1 读取当前测试图像（适配多图路径）
    picname = fullfile(pic_path_prefix, all_sim_picname{img_idx});  % 拼接图像完整路径
    
    %% 2.2 结果存储数组初始化（每张图独立存储，避免数据覆盖）
    % PSO结果存储
    PSO_bestchrom = zeros(Maxit, numsum);       % 每轮最优粒子（BPNN参数）
    PSO_bestfitness = zeros(Maxit, 1);          % 每轮最优适应度
    PSO_trace_bestfitness = zeros(maxgen, Maxit);% 每轮迭代的最优适应度曲线
    PSO_time = zeros(Maxit, 1);                 % 每轮运行时间（分钟）

    % IPSO结果存储（结构与PSO一致）
    IPSO_bestchrom = zeros(Maxit, numsum);
    IPSO_bestfitness = zeros(Maxit, 1);
    IPSO_trace_bestfitness = zeros(maxgen, Maxit);
    IPSO_time = zeros(Maxit, 1);

    %% 2.3 图像退化处理（与原代码一致）
    % 2.3.1 读取并标准化原始图像
    image_orgin = imread(picname);
    if size(image_orgin, 3) == 3
        image_orgin = rgb2gray(image_orgin);  % 转为灰度图
    end
    image_resized = imresize(image_orgin, picsize);  % 调整为90×90
    image_resized = double(image_resized) / 256;     % 归一化到[0,1]

    % 2.3.2 添加高斯模糊
    w_gauss = fspecial('gaussian', gauss_kernel_size, gauss_sigma);
    image_blurred = imfilter(image_resized, w_gauss, 'replicate');

    % 2.3.3 添加椒盐噪声（混合退化）
    image_degraded = imnoise(image_blurred, 'salt & pepper', salt_pepper_density);

    % 2.3.4 生成BPNN训练数据（3×3滑动窗口提取特征）
    [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_resized, inputnum);

    % 2.3.5 初始化BPNN网络（与原代码一致）
    net.trainParam.epochs = 1000;    % 训练次数
    net.trainParam.lr = 0.1;         % 学习率
    net.trainParam.goal = 1e-5;      % 训练目标误差
    net.trainParam.showWindow = false;  % 关闭训练窗口
    net.trainParam.showCommandLine = false;  % 关闭命令行输出
    net = newff(P_Matrix, T_Matrix, hiddennum);
    % 定义适应度函数（BPNN预测误差，minimization目标）
    fobj = @(x) cal_fitness(x, inputnum, hiddennum, outputnum, net, P_Matrix, T_Matrix);

    %% 2.4 PSO与IPSO多轮对比实验（与原代码一致）
    for index = 1:Maxit
        fprintf('第%d张图 - 第%d轮对比实验\n', img_idx, index);
        
        % 2.4.1 PSO算法运行（对比基准）
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
        
        % 2.4.2 IPSO算法运行（改进算法）
        tt3 = clock;
        [bestchrom_IPSO, bestfitness_IPSO, trace_best_IPSO] = PSO_improved(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std);
        % 存储IPSO结果
        IPSO_bestchrom(index, :) = bestchrom_IPSO;
        IPSO_bestfitness(index) = bestfitness_IPSO;
        IPSO_trace_bestfitness(:, index) = trace_best_IPSO;
        % 计算IPSO运行时间
        tt4 = clock;
        IPSO_time(index) = etime(tt4, tt3) / 60;  % 转换为分钟
        fprintf('IPSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_IPSO, IPSO_time(index));
    end

    %% 2.5 单张图结果统计与Table 3数据填充（新增）
    % % 2.5.1 统计100轮结果
    % [PSO_best_overall, ~] = min(PSO_bestfitness);
    % PSO_mean_fitness = mean(PSO_bestfitness);
    % PSO_mean_time = mean(PSO_time);
    % 
    % [IPSO_best_overall, ~] = min(IPSO_bestfitness);
    % IPSO_mean_fitness = mean(IPSO_bestfitness);
    % IPSO_mean_time = mean(IPSO_time);
    % 
    % fprintf('PSO - 最优适应度平均值: %.6f',  PSO_mean_time);
    % fprintf('IPSO - 最优适应度平均值: %.6f',  IPSO_mean_time);

    % 2.5.1 统计100轮结果
    [PSO_best_overall, ~] = min(PSO_bestfitness);
    PSO_mean_fitness = mean(PSO_bestfitness);
    PSO_median_fitness = median(PSO_bestfitness);      % <-- 新增中位数
    PSO_mean_time = mean(PSO_time);
    PSO_median_time = median(PSO_time);                % <-- 新增中位数

    [IPSO_best_overall, ~] = min(IPSO_bestfitness);
    IPSO_mean_fitness = mean(IPSO_bestfitness);
    IPSO_median_fitness = median(IPSO_bestfitness);    % <-- 新增中位数
    IPSO_mean_time = mean(IPSO_time);
    IPSO_median_time = median(IPSO_time);              % <-- 新增中位数

    % 打印结果（保留原有风格，仅补充中位数）
    fprintf('PSO  - 最优适应度平均值: %.6f，中位数: %.6f；平均耗时: %.6f s，耗时中位数: %.6f s\n', ...
            PSO_mean_fitness, PSO_median_fitness, PSO_mean_time, PSO_median_time);
    fprintf('IPSO - 最优适应度平均值: %.6f，中位数: %.6f；平均耗时: %.6f s，耗时中位数: %.6f s\n', ...
            IPSO_mean_fitness, IPSO_median_fitness, IPSO_mean_time, IPSO_median_time);

    
    % 计算IPSO相对PSO的适应度提升率
    improvement_rate = (PSO_best_overall - IPSO_best_overall) / PSO_best_overall * 100;
    
    % 填充Table 3数据（第img_idx行）
    Table3_data(img_idx, :) = [img_idx, ...
                               PSO_best_overall, PSO_mean_fitness, PSO_mean_time, ...
                               IPSO_best_overall, IPSO_mean_fitness, IPSO_mean_time, ...
                               improvement_rate];

    %% 2.6 绘制Figure 6（每张图1个Fig6，区分图像序号）
    % 2.6.1 提取最优轮次的适应度曲线
    [~, PSO_best_idx] = min(PSO_bestfitness);
    [~, IPSO_best_idx] = min(IPSO_bestfitness);
    PSO_best_curve = PSO_trace_bestfitness(:, PSO_best_idx);
    IPSO_best_curve = IPSO_trace_bestfitness(:, IPSO_best_idx);

    % 2.6.2 绘制Figure 6（Adaptation Curves Comparison）
    fig6_name = sprintf('Figure_6_Adaptation_Curves_%s', all_sim_picname{img_idx}(1:3));  % 命名：Fig6_xxx_i01
    figure('Name', fig6_name, 'Position', [100+img_idx*50, 100, 800, 500]);
    plot(1:maxgen, PSO_best_curve, 'b--', 'LineWidth', 2);  % PSO曲线：蓝色虚线
    hold on
    plot(1:maxgen, IPSO_best_curve, 'r-', 'LineWidth', 2);   % IPSO曲线：红色实线
    hold off

    % 图表美化（符合论文规范）
    title(sprintf('Figure 6. Adaptation Curves (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Iterations', 'FontSize', 12);
    ylabel('Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO', 'IPSO', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10);

    % 2.6.3 保存Figure 6（高清格式，区分图像）
    print(fullfile('ipso', sprintf('%s.eps', fig6_name)), '-depsc', '-r600');
    print(fullfile('ipso', sprintf('%s.png', fig6_name)), '-dpng', '-r600');
    print(fullfile('ipso', sprintf('%s.tiff', fig6_name)), '-dtiff', '-r600');

    %% 2.7 绘制Figure 7（每张图1个Fig7，区分图像序号）
    % 2.7.1 准备数据
    PSO_per_round_best = PSO_bestfitness;  
    IPSO_per_round_best = IPSO_bestfitness;  
    rounds = 1:Maxit;  

    % 2.7.2 绘制Figure 7
    fig7_name = sprintf('Figure_7_Best_Fitness_%s', all_sim_picname{img_idx}(1:3));  % 命名：Fig7_xxx_i01
    figure('Name', fig7_name, 'Position', [200+img_idx*50, 200, 800, 500]);
    % PSO曲线：蓝色虚线+圆点
    plot(rounds, PSO_per_round_best, 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');  
    hold on
    % IPSO曲线：红色实线+三角
    plot(rounds, IPSO_per_round_best, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');  
    hold off

    % 图表美化
    title(sprintf('Figure 7. Best Fitness Values (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Experiments (Rounds)', 'FontSize', 12);
    ylabel('Best Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO - Per Round Best', 'IPSO - Per Round Best', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10);

    % 2.7.3 保存Figure 7
    print(fullfile('ipso', sprintf('%s.eps', fig7_name)), '-depsc', '-r600');
    print(fullfile('ipso', sprintf('%s.png', fig7_name)), '-dpng', '-r600');
    print(fullfile('ipso', sprintf('%s.tiff', fig7_name)), '-dtiff', '-r600');

    %% 2.8 保存单张图结果MAT文件（10个文件，区分图像）
    mat_name = sprintf('ipso/best90_r100_%s', all_sim_picname{img_idx}(1:3));  % 命名：best90_r100_i01.mat
    % save(mat_name, ...
    %     'PSO_bestchrom', 'PSO_bestfitness', 'PSO_trace_bestfitness', 'PSO_time', ...
    %     'IPSO_bestchrom', 'IPSO_bestfitness', 'IPSO_trace_bestfitness', 'IPSO_time', ...
    %     'PSO_best_curve', 'IPSO_best_curve', 'all_sim_picname', 'img_idx');
    save(mat_name);

    %% 2.9 单张图运行时间统计
    t2_single = clock;
    single_time = etime(t2_single, t1_single) / 60;
    fprintf('第%d张图（%s）测试完成，耗时: %.2f分钟\n\n', img_idx, all_sim_picname{img_idx}, single_time);
end

% %% 3. 生成Table 3（Excel文件，论文用）
% % 3.1 创建Excel文件（存储在ipso目录下）
% table3_filename = fullfile('ipso', 'Table3_Comparison_GABP_IGABP_Fitness.xlsx');
% % 3.2 写入列标题
% writecell(col_labels, table3_filename, 'Sheet', 1, 'Range', 'A1:H1');
% % 3.3 写入数据（保留6位小数，时间保留2位，提升率保留2位）
% % 格式化数据（确保表格美观）
% Table3_data_formatted = Table3_data;
% Table3_data_formatted(:, [2,3,5,6]) = round(Table3_data_formatted(:, [2,3,5,6]), 6);  % 适应度保留6位
% Table3_data_formatted(:, [4,7]) = round(Table3_data_formatted(:, [4,7]), 2);          % 时间保留2位
% Table3_data_formatted(:, 8) = round(Table3_data_formatted(:, 8), 2);                  % 提升


%% 3. 生成Table 3（Excel文件，论文用）
% 3.1 确保ipso目录存在（避免因目录不存在导致写入失败）
if ~exist('ipso', 'dir')
    mkdir('ipso');  % 若目录不存在则创建
end

% 3.2 创建Excel文件（存储在ipso目录下）
table3_filename = fullfile('ipso', 'Table3_Comparison_GABP_IGABP_Fitness.xlsx');

% 3.3 写入列标题（覆盖原有内容，确保表头正确）
writecell(col_labels, table3_filename, 'Sheet', 1, 'Range', 'A1:H1');

% 3.4 格式化数据（保留指定小数位数，确保表格美观）
Table3_data_formatted = Table3_data;
Table3_data_formatted(:, [2,3,5,6]) = round(Table3_data_formatted(:, [2,3,5,6]), 6);  % 适应度保留6位小数
Table3_data_formatted(:, [4,7]) = round(Table3_data_formatted(:, [4,7]), 2);          % 时间保留2位小数
Table3_data_formatted(:, 8) = round(Table3_data_formatted(:, 8), 2);                  % 提升率保留2位小数

% 3.5 核心修正：将格式化后的数据写入Excel（从A2单元格开始，与表头对齐）
% 注意：使用writematrix函数，确保数据正确填充到表头下方
writematrix(Table3_data_formatted, table3_filename, 'Sheet', 1, 'Range', 'A2:H11');  % 10行数据（A2到H11）

% 3.6 提示信息（确认表格生成成功）
fprintf('Table 3已成功生成，保存路径：%s\n', table3_filename);