% ！！！非常重要
%********************************************************************
%F6论文适应度曲线，F7最佳适应度100次对比，T3每张图的适应度最优值和均值对比
%********************************************************************
%文件名可以修改，不限于10次

% 清空环境变量
clear
close all
clc

t1_total = clock;  % 记录10张图总运行时间起点

%% 0. 动态获取图像文件 - 完整版本
% 获取当前脚本所在目录
script_path = fileparts(mfilename('fullpath'));

% 构建图像目录路径（根据您的实际目录结构调整）
pic_dir = fullfile(script_path, 'ipso', 'valid');

% 如果默认目录不存在，让用户选择目录
if ~exist(pic_dir, 'dir')
    fprintf('默认图像目录不存在: %s\n', pic_dir);
    pic_dir = uigetdir(pwd, '请选择包含测试图像的目录');
    if pic_dir == 0
        error('用户取消了目录选择');
    end
end

% 获取目录下所有支持的图像文件（不包含子目录）
supported_formats = {'*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.png', '*.bmp'};
image_files = [];
for i = 1:length(supported_formats)
    current_files = dir(fullfile(pic_dir, supported_formats{i}));
    % 过滤掉子目录，只保留文件
    current_files = current_files(~[current_files.isdir]);
    image_files = [image_files; current_files];
end

if isempty(image_files)
    error('在目录 %s 中未找到任何图像文件', pic_dir);
end

% 提取文件名并排序（确保顺序一致）
all_sim_picname = {image_files.name};
[all_sim_picname, sort_idx] = sort(all_sim_picname);
num_images = length(all_sim_picname);

fprintf('找到 %d 个图像文件:\n', num_images);
for i = 1:num_images
    fprintf('  %d. %s\n', i, all_sim_picname{i});
end

% 扩展表格数据存储数组（更多列用于统计分析）
% 列定义：
% 1:Image_ID, 2:PSO_Best_Fitness, 3:IPSO_Best_Fitness, 4:IR_Best(%), 
% 5:PSO_Mean_Fitness, 6:IPSO_Mean_Fitness, 7:IR_Mean(%), 
% 8:PSO_median_fitness, 9:IPSO_median_fitness, 10:IR_median(%), 
% 11:PSO_std, 12:IPSO_std, 13:T-test_p-value, 14:Significance, 
% 15:PSO_Mean_Time(min), 16:IPSO_Mean_Time(min)
Table3_data = zeros(num_images, 16);  
col_labels = {'Image_ID', 'PSO_Best_Fitness', 'IPSO_Best_Fitness', 'IR_Best(%)', ...
              'PSO_Mean_Fitness', 'IPSO_Mean_Fitness', 'IR_Mean(%)', ...
              'PSO_median_fitness', 'IPSO_median_fitness', 'IR_median(%)', ...
              'PSO_std', 'IPSO_std', 'T-test_p-value', 'Significance', ...
              'PSO_Mean_Time(min)', 'IPSO_Mean_Time(min)'};

%% 1. 全局参数设置（与原单图代码一致，保证对比公平性）
% Maxit = 100;                % 算法对比轮次（论文要求100次消除随机误差）

% 折中方案：根据图像数量智能调整
if num_images <= 5
    Maxit = 100;    % 图像少，用完整次数
elseif num_images <= 15  
    Maxit = 50;     % 中等数量，平衡模式
else
    Maxit = 30;     % 图像多，保证覆盖性
end
Maxit = 100; 

fprintf('选择实验次数: %d次/图像 (共%d张图像)\n', Maxit, num_images);
fprintf('预计总时间: %.1f小时\n', num_images * Maxit * 0.00817); % 假设每次6分钟

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
sizepop = 50;              % 种群/粒子群规模
maxgen = 50;               % 最大迭代次数
fprintf('种群/粒子群规模: %d\n', sizepop);
fprintf('最大迭代次数: %d\n', maxgen);

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

%% 2. 多张图循环测试（核心新增逻辑）
for img_idx = 1:num_images
    fprintf('=====================================================\n');
    fprintf('==================== 第%d张图测试（%s） ====================\n', img_idx, all_sim_picname{img_idx});
    fprintf('=====================================================\n');
    
    t1_single = clock;  % 记录单张图运行时间起点
    
    % 创建文本文件用于保存输出结果
    [~, name_only, ~] = fileparts(all_sim_picname{img_idx});
    txt_filename = sprintf('ipso/best90_r100_%s.txt', name_only);
    
    % 确保ipso目录存在
    if ~exist('ipso', 'dir')
        mkdir('ipso');
    end
    
    % 打开文本文件
    fid = fopen(txt_filename, 'w', 'n', 'UTF-8');
    if fid == -1
        fprintf('警告: 无法创建文本文件 %s，结果将仅输出到命令行\n', txt_filename);
        fid = 1; % 使用标准输出
    else
        fprintf('结果将保存到: %s\n', txt_filename);
    end
    
    % 写入文件头
    fprintf(fid, '=====================================================\n');
    fprintf(fid, '==================== 第%d张图测试（%s） ====================\n', img_idx, all_sim_picname{img_idx});
    fprintf(fid, '=====================================================\n\n');
    
    %% 2.1 动态构建完整图像路径并读取图像
    picname = fullfile(pic_dir, all_sim_picname{img_idx});
    
    % 检查文件是否存在
    if ~exist(picname, 'file')
        fprintf('警告: 文件不存在: %s\n', picname);
        fprintf(fid, '警告: 文件不存在: %s\n', picname);
        fprintf('跳过该图像，继续处理下一张...\n');
        fprintf(fid, '跳过该图像，继续处理下一张...\n');
        fclose(fid);
        continue;  % 跳过不存在的文件
    end
    
    %% 2.2 读取当前测试图像
    try
        image_orgin = imread(picname);
        fprintf('成功读取图像: %s\n', all_sim_picname{img_idx});
        fprintf(fid, '成功读取图像: %s\n\n', all_sim_picname{img_idx});
    catch ME
        fprintf('错误: 无法读取图像 %s\n', picname);
        fprintf(fid, '错误: 无法读取图像 %s\n', picname);
        fprintf('错误信息: %s\n', ME.message);
        fprintf(fid, '错误信息: %s\n', ME.message);
        fprintf('跳过该图像，继续处理下一张...\n');
        fprintf(fid, '跳过该图像，继续处理下一张...\n');
        fclose(fid);
        continue;
    end
    
    %% 2.3 结果存储数组初始化（每张图独立存储，避免数据覆盖）
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

    %% 2.4 图像退化处理（与原代码一致）
    % 2.4.1 读取并标准化原始图像
    if size(image_orgin, 3) == 3
        image_orgin = rgb2gray(image_orgin);  % 转为灰度图
    end
    image_resized = imresize(image_orgin, picsize);  % 调整为90×90
    image_resized = double(image_resized) / 256;     % 归一化到[0,1]

    % 2.4.2 添加高斯模糊
    w_gauss = fspecial('gaussian', gauss_kernel_size, gauss_sigma);
    image_blurred = imfilter(image_resized, w_gauss, 'replicate');

    % 2.4.3 添加椒盐噪声（混合退化）
    % image_degraded = imnoise(image_blurred, 'salt & pepper', salt_pepper_density);
    image_degraded = image_blurred;     %不加噪声

    % 2.4.4 生成BPNN训练数据（3×3滑动窗口提取特征）
    [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_resized, inputnum);

    % 2.4.5 初始化BPNN网络（与原代码一致）
    net.trainParam.epochs = 1000;    % 训练次数
    net.trainParam.lr = 0.1;         % 学习率
    net.trainParam.goal = 1e-5;      % 训练目标误差
    net.trainParam.showWindow = false;  % 关闭训练窗口
    net.trainParam.showCommandLine = false;  % 关闭命令行输出
    net = newff(P_Matrix, T_Matrix, hiddennum);
    % 定义适应度函数（BPNN预测误差，minimization目标）
    fobj = @(x) cal_fitness(x, inputnum, hiddennum, outputnum, net, P_Matrix, T_Matrix);

    %% 2.5 PSO与IPSO多轮对比实验（与原代码一致）
    for index = 1:Maxit
        fprintf('第%d张图 - 第%d轮对比实验\n', img_idx, index);
        fprintf(fid, '第%d张图 - 第%d轮对比实验\n', img_idx, index);
        
        % 2.5.1 PSO算法运行（对比基准）
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
        fprintf(fid, 'PSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_PSO, PSO_time(index));
        
        % 2.5.2 IPSO算法运行（改进算法）
        tt3 = clock;
        p = 1.5;  % 在参数设置部分定义幂指数
        % p=0.5：前期快速下降，后期缓慢 → 侧重早期全局搜索
        % p=1.0：线性递减 → 均衡搜索
        % p=1.5：前期缓慢，后期快速下降 → 侧重后期局部精细搜索
        % p=2.0：更极端的后期快速下降

        [bestchrom_IPSO, bestfitness_IPSO, trace_best_IPSO] = PSO_improved_p(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std,p);
        % 存储IPSO结果
        IPSO_bestchrom(index, :) = bestchrom_IPSO;
        IPSO_bestfitness(index) = bestfitness_IPSO;
        IPSO_trace_bestfitness(:, index) = trace_best_IPSO;
        % 计算IPSO运行时间
        tt4 = clock;
        IPSO_time(index) = etime(tt4, tt3) / 60;  % 转换为分钟
        fprintf('IPSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_IPSO, IPSO_time(index));
        fprintf(fid, 'IPSO - 最优适应度: %.6f, 运行时间: %.2f分钟\n', bestfitness_IPSO, IPSO_time(index));
        
        fprintf('\n');
        fprintf(fid, '\n');
    end

    %% 2.6 单张图结果统计与Table 3数据填充
    % 2.6.1 统计100轮结果
    [PSO_best_overall, PSO_best_idx] = min(PSO_bestfitness);
    PSO_mean_fitness = mean(PSO_bestfitness);
    PSO_median_fitness = median(PSO_bestfitness);
    PSO_std_fitness = std(PSO_bestfitness);
    PSO_mean_time = mean(PSO_time);
    PSO_median_time = median(PSO_time);

    [IPSO_best_overall, IPSO_best_idx] = min(IPSO_bestfitness);
    IPSO_mean_fitness = mean(IPSO_bestfitness);
    IPSO_median_fitness = median(IPSO_bestfitness);
    IPSO_std_fitness = std(IPSO_bestfitness);
    IPSO_mean_time = mean(IPSO_time);
    IPSO_median_time = median(IPSO_time);

    % 2.6.2 计算各种提升率
    IR_best = (PSO_best_overall - IPSO_best_overall) / PSO_best_overall * 100;
    IR_mean = (PSO_mean_fitness - IPSO_mean_fitness) / PSO_mean_fitness * 100;
    IR_median = (PSO_median_fitness - IPSO_median_fitness) / PSO_median_fitness * 100;

    % 2.6.3 进行t检验（本地实现版本）
    [p_value, t_statistic] = manual_ttest2(PSO_bestfitness, IPSO_bestfitness);
    
    % 判断显著性水平
    if p_value < 0.001
        significance = '***';
    elseif p_value < 0.01
        significance = '**';
    elseif p_value < 0.05
        significance = '*';
    else
        significance = 'ns';
    end

    % 打印结果
    fprintf('\n=============== 第%d张图统计结果 ===============\n', img_idx);
    fprintf('PSO  - 最优适应度: %.6f, 平均值: %.6f, 中位数: %.6f, 标准差: %.6f\n', ...
            PSO_best_overall, PSO_mean_fitness, PSO_median_fitness, PSO_std_fitness);
    fprintf('IPSO - 最优适应度: %.6f, 平均值: %.6f, 中位数: %.6f, 标准差: %.6f\n', ...
            IPSO_best_overall, IPSO_mean_fitness, IPSO_median_fitness, IPSO_std_fitness);
    fprintf('t检验统计量: %.4f, p值: %.6f, 显著性: %s\n', t_statistic, p_value, significance);
    fprintf('提升率 - 最优: %.2f%%, 均值: %.2f%%, 中位数: %.2f%%\n', IR_best, IR_mean, IR_median);
    
    % 同时输出到文件
    fprintf(fid, '\n=============== 第%d张图统计结果 ===============\n', img_idx);
    fprintf(fid, 'PSO  - 最优适应度: %.6f, 平均值: %.6f, 中位数: %.6f, 标准差: %.6f\n', ...
            PSO_best_overall, PSO_mean_fitness, PSO_median_fitness, PSO_std_fitness);
    fprintf(fid, 'IPSO - 最优适应度: %.6f, 平均值: %.6f, 中位数: %.6f, 标准差: %.6f\n', ...
            IPSO_best_overall, IPSO_mean_fitness, IPSO_median_fitness, IPSO_std_fitness);
    fprintf(fid, 't检验统计量: %.4f, p值: %.6f, 显著性: %s\n', t_statistic, p_value, significance);
    fprintf(fid, '提升率 - 最优: %.2f%%, 均值: %.2f%%, 中位数: %.2f%%\n', IR_best, IR_mean, IR_median);
    
    % 填充Table 3数据（第img_idx行）
    Table3_data(img_idx, :) = [img_idx, ...
                               PSO_best_overall, IPSO_best_overall, IR_best, ...
                               PSO_mean_fitness, IPSO_mean_fitness, IR_mean, ...
                               PSO_median_fitness, IPSO_median_fitness, IR_median, ...
                               PSO_std_fitness, IPSO_std_fitness, p_value, 0, ... % 显著性用0占位，后面处理
                               PSO_mean_time, IPSO_mean_time];

    % 处理显著性标记（转换为数字便于Excel存储）
    switch significance
        case '***'
            sig_value = 3;
        case '**'
            sig_value = 2;
        case '*'
            sig_value = 1;
        otherwise
            sig_value = 0;
    end
    Table3_data(img_idx, 14) = sig_value;

    %% 2.7 绘制Figure 6（每张图1个Fig6，区分图像序号）
    % 2.7.1 提取最优轮次的适应度曲线
    PSO_best_curve = PSO_trace_bestfitness(:, PSO_best_idx);
    IPSO_best_curve = IPSO_trace_bestfitness(:, IPSO_best_idx);

    % 2.7.2 绘制Figure 6（Adaptation Curves Comparison）
    fig6_name = sprintf('Adaptation_Curves_%s', name_only);
    figure('Name', fig6_name, 'Position', [100+img_idx*50, 100, 800, 500],'Visible', 'off');
    plot(1:maxgen, PSO_best_curve, 'b--', 'LineWidth', 2);  % PSO曲线：蓝色虚线
    hold on
    plot(1:maxgen, IPSO_best_curve, 'r-', 'LineWidth', 2);   % IPSO曲线：红色实线
    hold off

    % 图表美化（符合论文规范）
    title(sprintf('Adaptation Curves (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Iterations', 'FontSize', 12);
    ylabel('Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO', 'IPSO', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10);

    % 2.7.3 保存Figure 6（高清格式，区分图像）
    print(fullfile('ipso', sprintf('%s.eps', fig6_name)), '-depsc', '-r600');
    print(fullfile('ipso', sprintf('%s.png', fig6_name)), '-dpng', '-r600');
    print(fullfile('ipso', sprintf('%s.tiff', fig6_name)), '-dtiff', '-r600');

    %% 2.8 绘制Figure 7（每张图1个Fig7，区分图像序号）
    % 2.8.1 准备数据
    PSO_per_round_best = PSO_bestfitness;  
    IPSO_per_round_best = IPSO_bestfitness;  
    rounds = 1:Maxit;  

    % 2.8.2 绘制Figure 7
    fig7_name = sprintf('Best_Fitness_%s', name_only);
    figure('Name', fig7_name, 'Position', [200+img_idx*50, 200, 800, 500],'Visible', 'off');
    % PSO曲线：蓝色虚线+圆点
    plot(rounds, PSO_per_round_best, 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');  
    hold on
    % IPSO曲线：红色实线+三角
    plot(rounds, IPSO_per_round_best, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');  
    hold off

    % 图表美化
    title(sprintf('Best Fitness Values (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Experiments (Rounds)', 'FontSize', 12);
    ylabel('Best Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO - Per Round Best', 'IPSO - Per Round Best', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10);

    % 2.8.3 保存Figure 7
    print(fullfile('ipso', sprintf('%s.eps', fig7_name)), '-depsc', '-r600');
    print(fullfile('ipso', sprintf('%s.png', fig7_name)), '-dpng', '-r600');
    print(fullfile('ipso', sprintf('%s.tiff', fig7_name)), '-dtiff', '-r600');

    %% 2.9 保存单张图结果MAT文件
    mat_name = sprintf('ipso/best90_%s_pop%d_gen%d_%s.mat', ...
                    name_only, sizepop, maxgen, datestr(now, 'yyyymmdd'));
    save(mat_name);

    %% 2.10 单张图运行时间统计
    t2_single = clock;
    single_time = etime(t2_single, t1_single) / 60;
    fprintf('第%d张图（%s）测试完成，耗时: %.2f分钟\n\n', img_idx, all_sim_picname{img_idx}, single_time);
    fprintf(fid, '第%d张图（%s）测试完成，耗时: %.2f分钟\n\n', img_idx, all_sim_picname{img_idx}, single_time);
    
    % 关闭文本文件（只关闭真正打开的文件句柄）
    if fid > 2          % 1=stdout，2=stderr，>2 才是我方才 fopen 的文件
      fclose(fid);
     fprintf('结果已保存到: %s\n', txt_filename);
    end
end

%% 3. 生成Table 3（Excel文件，论文用）
% 3.1 确保ipso目录存在
if ~exist('ipso', 'dir')
    mkdir('ipso');
end

% 3.2 创建Excel文件
table3_filename = fullfile('ipso', 'Table3_Comprehensive_Comparison_PSOBP_IPSOBP.xlsx');

% 3.3 写入列标题
writecell(col_labels, table3_filename, 'Sheet', 1, 'Range', 'A1:P1');

% 3.4 格式化数据
Table3_data_formatted = Table3_data;
Table3_data_formatted(:, [2,3,5,6,8,9,11,12]) = round(Table3_data_formatted(:, [2,3,5,6,8,9,11,12]), 6);
Table3_data_formatted(:, [4,7,10]) = round(Table3_data_formatted(:, [4,7,10]), 2);  % 提升率保留2位小数
% Table3_data_formatted(:, 13) = round(Table3_data_formatted(:, 13), 4);  % p值保留4位小数

% 对p值进行智能格式化：小于0.0001的显示为科学计数法
p_values = Table3_data(:, 13);
for i = 1:length(p_values)
    if p_values(i) < 0.0001
        Table3_data_formatted(i, 13) = p_values(i);  % 保持原值，不四舍五入
    else
        Table3_data_formatted(i, 13) = round(p_values(i), 4);  % 普通值四舍五入到4位小数
    end
end

Table3_data_formatted(:, [15,16]) = round(Table3_data_formatted(:, [15,16]), 2);  % 时间保留2位小数

% 3.5 写入数据
writematrix(Table3_data_formatted, table3_filename, 'Sheet', 1, 'Range', 'A2');

% 3.6 添加显著性标记说明的工作表

% 修改显著性标记说明工作表部分（原代码第427行）
significance_notes = {
    '显著性标记说明:', '', '', '';  % 确保每行都有4个元素
    '***', 'p < 0.001 (极显著)', '', '';
    '**',  'p < 0.01 (高度显著)', '', '';
    '*',   'p < 0.05 (显著)', '', '';
    'ns',  'p ≥ 0.05 (不显著)', '', '';
    '', '', '', '';
    '注: 提升率计算方式: (PSO - IPSO) / PSO × 100%', '', '', '';
    '正值表示IPSO性能优于PSO', '', '', '';
    't检验使用Welch''s t-test，适用于方差不等的情况', '', '', '';
};

writecell(significance_notes, table3_filename, 'Sheet', 2, 'Range', 'A1');

% 3.7 添加总体统计信息的工作表
% 3.7 添加总体统计信息的工作表
overall_stats = {
    '总体统计信息', '';
    '处理图像总数:', num2str(num_images);
    '每图实验轮次:', num2str(Maxit);
    '总运行时间(分钟):', num2str(round(etime(clock, t1_total)/60, 2));
    '', '';
    'PSO平均最优适应度:', num2str(round(mean(Table3_data(:, 2)), 6));
    'IPSO平均最优适应度:', num2str(round(mean(Table3_data(:, 3)), 6));
    '平均提升率(%):', num2str(round(mean(Table3_data(:, 4)), 2));
    '', '';
    '显著性测试结果:', '';
    '显著改善图像数量:', num2str(sum(Table3_data(:, 13) < 0.05));
    '极显著改善图像数量:', num2str(sum(Table3_data(:, 13) < 0.001));
    };
writecell(overall_stats, table3_filename, 'Sheet', 3, 'Range', 'A1');

%% 4. 生成汇总图表
% 4.1 绘制所有图像提升率对比图
figure('Name', 'Overall_Improvement_Rate', 'Position', [300, 300, 1000, 600]);
subplot(2,2,1);
bar(Table3_data(:, 4));
title('各图像最优适应度提升率');
xlabel('图像序号');
ylabel('提升率 (%)');
grid on;

subplot(2,2,2);
bar(Table3_data(:, 7));
title('各图像平均适应度提升率');
xlabel('图像序号');
ylabel('提升率 (%)');
grid on;

subplot(2,2,3);
histogram(Table3_data(:, 4), 10);
title('最优适应度提升率分布');
xlabel('提升率 (%)');
ylabel('图像数量');
grid on;

subplot(2,2,4);
scatter(Table3_data(:, 2), Table3_data(:, 3), 50, 'filled');
hold on;
plot([min(Table3_data(:, 2)), max(Table3_data(:, 2))], [min(Table3_data(:, 2)), max(Table3_data(:, 2))], 'r--', 'LineWidth', 2);
title('PSO vs IPSO 最优适应度散点图');
xlabel('PSO最优适应度');
ylabel('IPSO最优适应度');
legend('数据点', 'y=x参考线', 'Location', 'best');
grid on;

print(fullfile('ipso', 'Overall_Improvement_Analysis.eps'), '-depsc', '-r600');
print(fullfile('ipso', 'Overall_Improvement_Analysis.png'), '-dpng', '-r600');

%% 5. 总运行时间统计和总结
t2_total = clock;
total_time = etime(t2_total, t1_total) / 60;
fprintf('=====================================================\n');
fprintf('所有图像处理完成！总运行时间: %.2f分钟\n', total_time);
fprintf('Table 3已保存至: %s\n', table3_filename);
fprintf('处理了 %d 张图像\n', num_images);
fprintf('平均每张图耗时: %.2f分钟\n', total_time/num_images);
fprintf('显著性改善图像数量: %d/%d\n', sum(Table3_data(:, 13) < 0.05), num_images);
fprintf('=====================================================\n');

% 创建总结果摘要文本文件
summary_filename = 'ipso/Experiment_Comprehensive_Summary.txt';
fid_summary = fopen(summary_filename, 'w', 'n', 'UTF-8');
if fid_summary ~= -1
    fprintf(fid_summary, 'IPSOBPR实验综合总结报告\n');
    fprintf(fid_summary, '========================\n\n');
    fprintf(fid_summary, '总运行时间: %.2f分钟\n', total_time);
    fprintf(fid_summary, '处理图像数量: %d张\n', num_images);
    fprintf(fid_summary, '每张图像实验轮次: %d次\n', Maxit);
    fprintf(fid_summary, 'Table 3文件: %s\n', table3_filename);
    fprintf(fid_summary, '平均最优适应度提升率: %.2f%%\n', mean(Table3_data(:, 4)));
    fprintf(fid_summary, '显著性改善图像比例: %.1f%%\n', sum(Table3_data(:, 13) < 0.05)/num_images*100);
    fprintf(fid_summary, '生成时间: %s\n', datestr(now));
    fclose(fid_summary);
    fprintf('实验总结已保存到: %s\n', summary_filename);
end

%% 6. 本地t检验函数定义
function [p_value, t_statistic] = manual_ttest2(sample1, sample2)
    % 手动实现Welch's t检验（适用于方差不等的情况）
    % 输入：两个独立样本
    % 输出：p值和t统计量
    
    n1 = length(sample1);
    n2 = length(sample2);
    
    % 计算均值和方差
    mean1 = mean(sample1);
    mean2 = mean(sample2);
    var1 = var(sample1);
    var2 = var(sample2);
    
    % 计算t统计量（Welch's t-test）
    t_statistic = (mean1 - mean2) / sqrt(var1/n1 + var2/n2);
    
    % 计算自由度（Welch-Satterthwaite方程）
    df_numerator = (var1/n1 + var2/n2)^2;
    df_denominator = (var1/n1)^2/(n1-1) + (var2/n2)^2/(n2-1);
    df = df_numerator / df_denominator;
    
    % 计算双边检验的p值
    % 使用t分布的累积分布函数（CDF）
    p_value = 2 * (1 - tcdf(abs(t_statistic), df));
    
    % 如果p值大于1（由于数值误差），调整为1
    if p_value > 1
        p_value = 1;
    end
    
    fprintf('手动t检验结果: t(%.2f) = %.4f, p = %.6f\n', df, t_statistic, p_value);
end

function p = tcdf(t, df)
    % 手动实现t分布的累积分布函数（CDF）
    % 使用近似方法计算t分布的CDF
    
    if df <= 0
        error('自由度必须为正数');
    end
    
    % 对于大自由度，近似为正态分布
    if df > 100
        p = 0.5 * (1 + erf(t / sqrt(2)));
    else
        % 使用Beta函数近似计算t分布CDF
        x = (t + sqrt(t^2 + df)) / (2 * sqrt(t^2 + df));
        p = betainc(x, df/2, df/2);
    end
end

function y = erf(x)
    % 误差函数近似计算
    % 使用Abramowitz and Stegun近似公式
    
    a1 =  0.254829592;
    a2 = -0.284496736;
    a3 =  1.421413741;
    a4 = -1.453152027;
    a5 =  1.061405429;
    p  =  0.3275911;
    
    t = 1.0 ./ (1.0 + p*abs(x));
    y = 1.0 - (a1*t + a2*t.^2 + a3*t.^3 + a4*t.^4 + a5*t.^5) .* exp(-x.^2);
    y(x < 0) = -y(x < 0);
end