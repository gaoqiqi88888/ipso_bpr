% ！！！当前测试方案2扰动
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
    fprintf('Default image directory does not exist: %s\n', pic_dir);
    pic_dir = uigetdir(pwd, 'Please select the directory containing test images');
    if pic_dir == 0
        error('User canceled directory selection');
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
    error('No image files found in directory %s', pic_dir);
end

% 提取文件名并排序（确保顺序一致）
all_sim_picname = {image_files.name};
[all_sim_picname, sort_idx] = sort(all_sim_picname);
num_images = length(all_sim_picname);

fprintf('Found %d image files:\n', num_images);
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
% Maxit = 10; 
Maxit = 100; 

fprintf('Selected experiment count: %d times/image (Total %d images)\n', Maxit, num_images);
fprintf('Estimated total time: %.1f hours\n', num_images * Maxit * 0.00817); % 假设每次6分钟

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
fprintf('Population size: %d\n', sizepop);
fprintf('Maximum iterations: %d\n', maxgen);

% 1.4 PSO核心控制参数 
c1 = 1.5;                   % 个体学习因子
c2 = 1.5;                   % 全局学习因子
w_init = 0.9;               % IPSO初始惯性权重
w_final = 0.3;              % IPSO最终惯性权重
v_max = 0.5;                % 粒子最大速度
v_min = -0.5;               % 粒子最小速度
pos_max = 1;                % 粒子位置上限
pos_min = -1;               % 粒子位置下限
% perturb_trigger_ratio = 0.7;% IPSO高斯扰动触发阈值
perturb_trigger_ratio = 0.7;% IPSO高斯扰动触发阈值
perturb_std = 0.1;          % 高斯扰动标准差

%% 2. 多张图循环测试（核心新增逻辑）
for img_idx = 1:num_images
    fprintf('=====================================================\n');
    fprintf('==================== Image %d Testing (%s) ====================\n', img_idx, all_sim_picname{img_idx});
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
        fprintf('Warning: Cannot create text file %s, results will be output to command line only\n', txt_filename);
        fid = 1; % 使用标准输出
    else
        fprintf('Results will be saved to: %s\n', txt_filename);
    end
    
    % 写入文件头
    fprintf(fid, '=====================================================\n');
    fprintf(fid, '==================== Image %d Testing (%s) ====================\n', img_idx, all_sim_picname{img_idx});
    fprintf(fid, '=====================================================\n\n');
    
    %% 2.1 动态构建完整图像路径并读取图像
    picname = fullfile(pic_dir, all_sim_picname{img_idx});
    
    % 检查文件是否存在
    if ~exist(picname, 'file')
        fprintf('Warning: File does not exist: %s\n', picname);
        fprintf(fid, 'Warning: File does not exist: %s\n', picname);
        fprintf('Skipping this image, continuing to next one...\n');
        fprintf(fid, 'Skipping this image, continuing to next one...\n');
        fclose(fid);
        continue;  % 跳过不存在的文件
    end
    
    %% 2.2 读取当前测试图像
    try
        image_orgin = imread(picname);
        fprintf('Successfully read image: %s\n', all_sim_picname{img_idx});
        fprintf(fid, 'Successfully read image: %s\n\n', all_sim_picname{img_idx});
    catch ME
        fprintf('Error: Cannot read image %s\n', picname);
        fprintf(fid, 'Error: Cannot read image %s\n', picname);
        fprintf('Error message: %s\n', ME.message);
        fprintf(fid, 'Error message: %s\n', ME.message);
        fprintf('Skipping this image, continuing to next one...\n');
        fprintf(fid, 'Skipping this image, continuing to next one...\n');
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
     p = 1.5;  % 在参数设置部分定义幂指数

    for index = 1:Maxit
        fprintf('Image %d - Round %d comparison experiment\n', img_idx, index);
        fprintf(fid, 'Image %d - Round %d comparison experiment\n', img_idx, index);
        
        % 2.5.1 PSO算法运行（对比基准）
        tt1 = clock;
        % [bestchrom_PSO, bestfitness_PSO, trace_best_PSO] = PSO_standard(sizepop, maxgen, numsum, fobj, c1, c2, w_init, v_max, v_min, pos_max, pos_min);
        [bestchrom_PSO, bestfitness_PSO, trace_best_PSO] = PSO_improved_p1(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std,p);
        % 存储PSO结果
        PSO_bestchrom(index, :) = bestchrom_PSO;
        PSO_bestfitness(index) = bestfitness_PSO;
        PSO_trace_bestfitness(:, index) = trace_best_PSO;
        % 计算PSO运行时间
        tt2 = clock;
        PSO_time(index) = etime(tt2, tt1) / 60;  % 转换为分钟
        fprintf('PSO - Best fitness: %.6f, Running time: %.2f minutes\n', bestfitness_PSO, PSO_time(index));
        fprintf(fid, 'PSO - Best fitness: %.6f, Running time: %.2f minutes\n', bestfitness_PSO, PSO_time(index));
        
        % 2.5.2 IPSO算法运行（改进算法）
        tt3 = clock;
        p = 1.5;  % 在参数设置部分定义幂指数
        % p=0.5：前期快速下降，后期缓慢 → 侧重早期全局搜索
        % p=1.0：线性递减 → 均衡搜索
        % p=1.5：前期缓慢，后期快速下降 → 侧重后期局部精细搜索
        % p=2.0：更极端的后期快速下降

        % [bestchrom_IPSO, bestfitness_IPSO, trace_best_IPSO] = PSO_standard_p2(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std,p);
        [bestchrom_IPSO, bestfitness_IPSO, trace_best_IPSO] = PSO_improved_p2(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std,p);


        % 存储IPSO结果
        IPSO_bestchrom(index, :) = bestchrom_IPSO;
        IPSO_bestfitness(index) = bestfitness_IPSO;
        IPSO_trace_bestfitness(:, index) = trace_best_IPSO;
        % 计算IPSO运行时间
        tt4 = clock;
        IPSO_time(index) = etime(tt4, tt3) / 60;  % 转换为分钟
        fprintf('IPSO - Best fitness: %.6f, Running time: %.2f minutes\n', bestfitness_IPSO, IPSO_time(index));
        fprintf(fid, 'IPSO - Best fitness: %.6f, Running time: %.2f minutes\n', bestfitness_IPSO, IPSO_time(index));
        
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
    fprintf('\n=============== Image %d Statistical Results ===============\n', img_idx);
    fprintf('PSO  - Best fitness: %.6f, Mean: %.6f, Median: %.6f, Std: %.6f\n', ...
            PSO_best_overall, PSO_mean_fitness, PSO_median_fitness, PSO_std_fitness);
    fprintf('IPSO - Best fitness: %.6f, Mean: %.6f, Median: %.6f, Std: %.6f\n', ...
            IPSO_best_overall, IPSO_mean_fitness, IPSO_median_fitness, IPSO_std_fitness);
    fprintf('t-test statistic: %.4f, p-value: %.6f, Significance: %s\n', t_statistic, p_value, significance);
    fprintf('Improvement rate - Best: %.2f%%, Mean: %.2f%%, Median: %.2f%%\n', IR_best, IR_mean, IR_median);
    
    % 同时输出到文件
    fprintf(fid, '\n=============== Image %d Statistical Results ===============\n', img_idx);
    fprintf(fid, 'PSO  - Best fitness: %.6f, Mean: %.6f, Median: %.6f, Std: %.6f\n', ...
            PSO_best_overall, PSO_mean_fitness, PSO_median_fitness, PSO_std_fitness);
    fprintf(fid, 'IPSO - Best fitness: %.6f, Mean: %.6f, Median: %.6f, Std: %.6f\n', ...
            IPSO_best_overall, IPSO_mean_fitness, IPSO_median_fitness, IPSO_std_fitness);
    fprintf(fid, 't-test statistic: %.4f, p-value: %.6f, Significance: %s\n', t_statistic, p_value, significance);
    fprintf(fid, 'Improvement rate - Best: %.2f%%, Mean: %.2f%%, Median: %.2f%%\n', IR_best, IR_mean, IR_median);
    
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
    figure('Name', fig6_name, 'Position', [100+img_idx*50, 100, 800, 500], 'Visible', 'off', ...
           'PaperUnits', 'inches', 'PaperPositionMode', 'auto');

    plot(1:maxgen, PSO_best_curve, 'b--', 'LineWidth', 2);  % PSO曲线：蓝色虚线
    hold on
    plot(1:maxgen, IPSO_best_curve, 'r-', 'LineWidth', 2);   % IPSO曲线：红色实线
    hold off

    % 图表美化（英文版，符合SCI论文规范）
    title(sprintf('Comparison of Adaptation Curves (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Iterations', 'FontSize', 12);
    ylabel('Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO', 'IPSO', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10, 'LineWidth', 1);

    % 2.7.3 保存Figure 6（三种格式，高DPI）
    fig6_path = fullfile('ipso', fig6_name);

    % EPS格式（矢量图，适合LaTeX）
    print([fig6_path '.eps'], '-depsc', '-r600', '-painters');
    fprintf('Saved EPS: %s.eps\n', fig6_path);

    % PDF格式（矢量图，适合打印）
    print([fig6_path '.pdf'], '-dpdf', '-r600', '-painters');
    fprintf('Saved PDF: %s.pdf\n', fig6_path);

    % TIFF格式（位图，1200 dpi）
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 8 5];  % 8×5英寸
    print([fig6_path '.tiff'], '-dtiff', '-r1200');
    fprintf('Saved TIFF: %s.tiff (1200 dpi)\n', fig6_path);

    % 关闭图形，释放内存
    close(gcf);

    %% 2.8 绘制Figure 7（每张图1个Fig7，区分图像序号）
    % 2.8.1 准备数据
    PSO_per_round_best = PSO_bestfitness;  
    IPSO_per_round_best = IPSO_bestfitness;  
    rounds = 1:Maxit;  

    % 2.8.2 绘制Figure 7
    fig7_name = sprintf('Best_Fitness_%s', name_only);
    figure('Name', fig7_name, 'Position', [200+img_idx*50, 200, 800, 500], 'Visible', 'off', ...
           'PaperUnits', 'inches', 'PaperPositionMode', 'auto');

    % PSO曲线：蓝色虚线+圆点
    plot(rounds, PSO_per_round_best, 'b--o', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'b', 'MarkerFaceColor', 'none');  
    hold on
    % IPSO曲线：红色实线+三角
    plot(rounds, IPSO_per_round_best, 'r-^', 'LineWidth', 1.5, 'MarkerSize', 4, 'MarkerEdgeColor', 'r', 'MarkerFaceColor', 'none');  
    hold off

    % 图表美化（英文版）
    title(sprintf('Best Fitness Values in 100 Runs (%s)', all_sim_picname{img_idx}), 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Number of Experiments (Rounds)', 'FontSize', 12);
    ylabel('Best Fitness Value (BPNN Prediction Error)', 'FontSize', 12);
    legend('PSO - Per Round Best', 'IPSO - Per Round Best', 'Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 10, 'LineWidth', 1);

    % 2.8.3 保存Figure 7（三种格式，高DPI）
    fig7_path = fullfile('ipso', fig7_name);

    % EPS格式
    print([fig7_path '.eps'], '-depsc', '-r600', '-painters');
    fprintf('Saved EPS: %s.eps\n', fig7_path);

    % PDF格式
    print([fig7_path '.pdf'], '-dpdf', '-r600', '-painters');
    fprintf('Saved PDF: %s.pdf\n', fig7_path);

    % TIFF格式（位图，1200 dpi）
    fig = gcf;
    fig.PaperUnits = 'inches';
    fig.PaperPosition = [0 0 8 5];  % 8×5英寸
    print([fig7_path '.tiff'], '-dtiff', '-r1200');
    fprintf('Saved TIFF: %s.tiff (1200 dpi)\n', fig7_path);

    % 关闭图形，释放内存
    close(gcf);

    %% 2.9 保存单张图结果MAT文件
    mat_name = sprintf('ipso/best90_%s_pop%d_gen%d_%s.mat', ...
                    name_only, sizepop, maxgen, datestr(now, 'yyyymmdd'));
    save(mat_name);

    %% 2.10 单张图运行时间统计
    t2_single = clock;
    single_time = etime(t2_single, t1_single) / 60;
    fprintf('Image %d (%s) testing completed, time taken: %.2f minutes\n\n', img_idx, all_sim_picname{img_idx}, single_time);
    fprintf(fid, 'Image %d (%s) testing completed, time taken: %.2f minutes\n\n', img_idx, all_sim_picname{img_idx}, single_time);
    
    % 关闭文本文件（只关闭真正打开的文件句柄）
    if fid > 2          % 1=stdout，2=stderr，>2 才是我方才 fopen 的文件
      fclose(fid);
     fprintf('Results saved to: %s\n', txt_filename);
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

% 3.6 添加显著性标记说明的工作表（英文版）
significance_notes = {
    'Significance Marking Instructions:', '', '', '';
    '***', 'p < 0.001 (Extremely Significant)', '', '';
    '**',  'p < 0.01 (Highly Significant)', '', '';
    '*',   'p < 0.05 (Significant)', '', '';
    'ns',  'p ≥ 0.05 (Not Significant)', '', '';
    '', '', '', '';
    'Note: Improvement Rate Calculation: (PSO - IPSO) / PSO × 100%', '', '', '';
    'Positive values indicate IPSO outperforms PSO', '', '', '';
    't-test uses Welch''s t-test, suitable for unequal variances', '', '', '';
};

writecell(significance_notes, table3_filename, 'Sheet', 2, 'Range', 'A1');

% 3.7 添加总体统计信息的工作表（英文版）
overall_stats = {
    'Overall Statistical Information', '';
    'Total Images Processed:', num2str(num_images);
    'Experiments per Image:', num2str(Maxit);
    'Total Running Time (minutes):', num2str(round(etime(clock, t1_total)/60, 2));
    '', '';
    'Average PSO Best Fitness:', num2str(round(mean(Table3_data(:, 2)), 6));
    'Average IPSO Best Fitness:', num2str(round(mean(Table3_data(:, 3)), 6));
    'Average Improvement Rate (%):', num2str(round(mean(Table3_data(:, 4)), 2));
    '', '';
    'Significance Test Results:', '';
    'Number of Images with Significant Improvement:', num2str(sum(Table3_data(:, 13) < 0.05));
    'Number of Images with Extremely Significant Improvement:', num2str(sum(Table3_data(:, 13) < 0.001));
    };
writecell(overall_stats, table3_filename, 'Sheet', 3, 'Range', 'A1');

%% 4. 生成汇总图表（英文版，四种独立图表）
% 4.1 设置图形参数
figure_width = 8;  % 英寸
figure_height = 6; % 英寸
line_width = 1.5;
marker_size = 6;
font_size_title = 14;
font_size_axes = 12;
font_size_labels = 10;

% 4.2 各图像最优适应度提升率柱状图（英文版）
figure('Name', 'Best_Fitness_Improvement_Per_Image', 'Visible', 'off', ...
       'PaperUnits', 'inches', 'PaperPosition', [0 0 figure_width figure_height], ...
       'PaperSize', [figure_width figure_height], 'Color', 'w');

bar_colors = zeros(num_images, 3);
for i = 1:num_images
    if Table3_data(i, 4) > 0
        bar_colors(i, :) = [0.2, 0.4, 0.8]; % 蓝色表示提升
    else
        bar_colors(i, :) = [0.8, 0.2, 0.2]; % 红色表示下降
    end
end

bar_handles = bar(Table3_data(:, 4));
set(bar_handles, 'FaceColor', 'flat', 'CData', bar_colors, 'EdgeColor', 'k', 'LineWidth', 0.5);

hold on;
plot([0, num_images+1], [0, 0], 'k-', 'LineWidth', 1); % 零线
hold off;

title('Best Fitness Improvement Rate per Image', 'FontSize', font_size_title, 'FontWeight', 'bold');
xlabel('Image Index', 'FontSize', font_size_axes);
ylabel('Improvement Rate (%)', 'FontSize', font_size_axes);
grid on;
set(gca, 'FontSize', font_size_labels, 'LineWidth', 1, 'Box', 'on');

% 保存三种格式
bar_path = fullfile('ipso', 'Best_Fitness_Improvement_Per_Image');
print([bar_path '.eps'], '-depsc', '-r600', '-painters');
print([bar_path '.pdf'], '-dpdf', '-r600', '-painters');
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 figure_width figure_height];
print([bar_path '.tiff'], '-dtiff', '-r1200');
fprintf('Saved bar chart: %s (EPS/PDF/TIFF)\n', bar_path);
close(gcf);

% 4.3 各图像平均适应度提升率柱状图（英文版）
figure('Name', 'Average_Fitness_Improvement_Per_Image', 'Visible', 'off', ...
       'PaperUnits', 'inches', 'PaperPosition', [0 0 figure_width figure_height], ...
       'PaperSize', [figure_width figure_height], 'Color', 'w');

bar(Table3_data(:, 7), 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);

title('Average Fitness Improvement Rate per Image', 'FontSize', font_size_title, 'FontWeight', 'bold');
xlabel('Image Index', 'FontSize', font_size_axes);
ylabel('Improvement Rate (%)', 'FontSize', font_size_axes);
grid on;
set(gca, 'FontSize', font_size_labels, 'LineWidth', 1, 'Box', 'on');

% 保存三种格式
avg_path = fullfile('ipso', 'Average_Fitness_Improvement_Per_Image');
print([avg_path '.eps'], '-depsc', '-r600', '-painters');
print([avg_path '.pdf'], '-dpdf', '-r600', '-painters');
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 figure_width figure_height];
print([avg_path '.tiff'], '-dtiff', '-r1200');
fprintf('Saved average chart: %s (EPS/PDF/TIFF)\n', avg_path);
close(gcf);

% 4.4 最优适应度提升率分布直方图（英文版）
figure('Name', 'Distribution_of_Best_Fitness_Improvement_Rates', 'Visible', 'off', ...
       'PaperUnits', 'inches', 'PaperPosition', [0 0 figure_width figure_height], ...
       'PaperSize', [figure_width figure_height], 'Color', 'w');

% 计算最优分布
hist_edges = -10:2:20; % 从-10%到20%，步长2%
hist_counts = histcounts(Table3_data(:, 4), hist_edges);
hist_centers = (hist_edges(1:end-1) + hist_edges(2:end)) / 2;

bar(hist_centers, hist_counts, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
hold on;
% 添加正态分布拟合
pd = fitdist(Table3_data(:, 4), 'Normal');
x_fit = linspace(min(Table3_data(:, 4)), max(Table3_data(:, 4)), 100);
y_fit = pdf(pd, x_fit) * Maxit * (hist_edges(2) - hist_edges(1));
plot(x_fit, y_fit, 'r-', 'LineWidth', 2);
hold off;

title('Distribution of Best Fitness Improvement Rates', 'FontSize', font_size_title, 'FontWeight', 'bold');
xlabel('Improvement Rate (%)', 'FontSize', font_size_axes);
ylabel('Number of Images', 'FontSize', font_size_axes);
legend('Histogram', 'Normal Fit', 'Location', 'best', 'FontSize', font_size_labels);
grid on;
set(gca, 'FontSize', font_size_labels, 'LineWidth', 1, 'Box', 'on');

% 保存三种格式
hist_path = fullfile('ipso', 'Distribution_of_Best_Fitness_Improvement_Rates');
print([hist_path '.eps'], '-depsc', '-r600', '-painters');
print([hist_path '.pdf'], '-dpdf', '-r600', '-painters');
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 figure_width figure_height];
print([hist_path '.tiff'], '-dtiff', '-r1200');
fprintf('Saved histogram: %s (EPS/PDF/TIFF)\n', hist_path);
close(gcf);

% 4.5 PSO vs IPSO最优适应度散点图（英文版）
figure('Name', 'Scatter_Plot_PSO_vs_IPSO', 'Visible', 'off', ...
       'PaperUnits', 'inches', 'PaperPosition', [0 0 figure_width figure_height+1], ...
       'PaperSize', [figure_width figure_height+1], 'Color', 'w');

% 提取数据
pso_best = Table3_data(:, 2);
ipso_best = Table3_data(:, 3);

% 绘制散点
scatter(pso_best, ipso_best, marker_size*10, 'filled', ...
        'MarkerFaceColor', [0.2, 0.4, 0.8], 'MarkerEdgeColor', 'k');

hold on;

% y=x参考线
min_val = min([pso_best; ipso_best]);
max_val = max([pso_best; ipso_best]);
margin = (max_val - min_val) * 0.1;
x_range = [min_val-margin, max_val+margin];
plot(x_range, x_range, 'r--', 'LineWidth', 2, 'DisplayName', 'y = x Reference Line');

% 线性拟合
p = polyfit(pso_best, ipso_best, 1);
y_fit = polyval(p, x_range);
plot(x_range, y_fit, 'g-', 'LineWidth', 2, 'DisplayName', 'Linear Fit');

hold off;

title('Scatter Plot of Best Fitness: PSO vs IPSO', 'FontSize', font_size_title, 'FontWeight', 'bold');
xlabel('Best Fitness (PSO)', 'FontSize', font_size_axes);
ylabel('Best Fitness (IPSO)', 'FontSize', font_size_axes);
legend('Location', 'best', 'FontSize', font_size_labels);
grid on;
set(gca, 'FontSize', font_size_labels, 'LineWidth', 1, 'Box', 'on');

% 添加拟合方程文本
fit_equation = sprintf('y = %.4fx + %.4f', p(1), p(2));
R2 = corr(pso_best, ipso_best)^2;
text(0.05, 0.95, {fit_equation, sprintf('R² = %.4f', R2)}, ...
     'Units', 'normalized', 'FontSize', 10, 'VerticalAlignment', 'top', ...
     'BackgroundColor', [1 1 1 0.7]);

% 保存三种格式
scatter_path = fullfile('ipso', 'Scatter_Plot_PSO_vs_IPSO');
print([scatter_path '.eps'], '-depsc', '-r600', '-painters');
print([scatter_path '.pdf'], '-dpdf', '-r600', '-painters');
fig = gcf;
fig.PaperUnits = 'inches';
fig.PaperPosition = [0 0 figure_width figure_height+1];
print([scatter_path '.tiff'], '-dtiff', '-r1200');
fprintf('Saved scatter plot: %s (EPS/PDF/TIFF)\n', scatter_path);
close(gcf);

%% 5. 总运行时间统计和总结（英文版）
t2_total = clock;
total_time = etime(t2_total, t1_total) / 60;
fprintf('=====================================================\n');
fprintf('All image processing completed! Total running time: %.2f minutes\n', total_time);
fprintf('Table 3 saved to: %s\n', table3_filename);
fprintf('Processed %d images\n', num_images);
fprintf('Average time per image: %.2f minutes\n', total_time/num_images);
fprintf('Number of significantly improved images: %d/%d\n', sum(Table3_data(:, 13) < 0.05), num_images);
fprintf('=====================================================\n');

% 创建总结果摘要文本文件（英文版）
summary_filename = 'ipso/Experiment_Comprehensive_Summary.txt';
fid_summary = fopen(summary_filename, 'w', 'n', 'UTF-8');
if fid_summary ~= -1
    fprintf(fid_summary, 'IPSOBPR Experiment Comprehensive Summary Report\n');
    fprintf(fid_summary, '================================================\n\n');
    fprintf(fid_summary, 'Total Running Time: %.2f minutes\n', total_time);
    fprintf(fid_summary, 'Number of Processed Images: %d\n', num_images);
    fprintf(fid_summary, 'Experiments per Image: %d times\n', Maxit);
    fprintf(fid_summary, 'Table 3 File: %s\n', table3_filename);
    fprintf(fid_summary, 'Average Best Fitness Improvement Rate: %.2f%%\n', mean(Table3_data(:, 4)));
    fprintf(fid_summary, 'Proportion of Significantly Improved Images: %.1f%%\n', sum(Table3_data(:, 13) < 0.05)/num_images*100);
    
    % 计算总体统计
    fprintf(fid_summary, '\nOverall Statistics:\n');
    fprintf(fid_summary, '  PSO Average Best Fitness: %.6f ± %.6f\n', mean(Table3_data(:, 2)), std(Table3_data(:, 2)));
    fprintf(fid_summary, '  IPSO Average Best Fitness: %.6f ± %.6f\n', mean(Table3_data(:, 3)), std(Table3_data(:, 3)));
    fprintf(fid_summary, '  Best Improvement Range: %.2f%% to %.2f%%\n', min(Table3_data(:, 4)), max(Table3_data(:, 4)));
    fprintf(fid_summary, '  Median Improvement Rate: %.2f%%\n', median(Table3_data(:, 4)));
    
    fprintf(fid_summary, '\nGenerated on: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fclose(fid_summary);
    fprintf('Experiment summary saved to: %s\n', summary_filename);
end

%% 6. 本地t检验函数定义（保持不变）
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
    
    fprintf('Manual t-test result: t(%.2f) = %.4f, p = %.6f\n', df, t_statistic, p_value);
end

function p = tcdf(t, df)
    % 手动实现t分布的累积分布函数（CDF）
    % 使用近似方法计算t分布的CDF
    
    if df <= 0
        error('Degrees of freedom must be positive');
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

%% 7. 打印完成信息
fprintf('\n=====================================================\n');
fprintf('Code execution completed successfully!\n');
fprintf('All charts have been saved in three formats:\n');
fprintf('  1. EPS (Vector format for LaTeX)\n');
fprintf('  2. PDF (Vector format for printing)\n');
fprintf('  3. TIFF (1200 dpi bitmap for journal submission)\n');
fprintf('=====================================================\n');

% 列出生成的主要文件
fprintf('\nGenerated files in "ipso" folder:\n');
fprintf('  • Table3_Comprehensive_Comparison_PSOBP_IPSOBP.xlsx\n');
fprintf('  • Best_Fitness_Improvement_Per_Image.[eps/pdf/tiff]\n');
fprintf('  • Average_Fitness_Improvement_Per_Image.[eps/pdf/tiff]\n');
fprintf('  • Distribution_of_Best_Fitness_Improvement_Rates.[eps/pdf/tiff]\n');
fprintf('  • Scatter_Plot_PSO_vs_IPSO.[eps/pdf/tiff]\n');
fprintf('  • Experiment_Comprehensive_Summary.txt\n');
fprintf('  • For each image: Adaptation_Curves_* and Best_Fitness_* files\n');