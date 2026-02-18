% 元启发算法对比，重要，运行时间见带_Time版
%********************************************************************
% IPSO与多种先进优化算法对比实验
% 对比算法：GA(遗传算法), WOA(鲸鱼优化), HBA(2022前沿), RIME(2023前沿)等算法对比
% 对比指标：最优值、均值、标准差、收敛速度、显著性检验
%********************************************************************

%% 对比实验主程序 - IPSO_vs_StateOfArt.m
function IPSO_vs_StateOfArt()
%% 0. 清空环境变量
clear; close all; clc;
t_total = clock;
fprintf('========== IPSO vs State-of-the-art Algorithms Comparison ==========\n');

% 创建结果目录 - 只需添加这一行！
if ~exist('comparison_results', 'dir')
    mkdir('comparison_results');
end

%% 1. 全局参数设置
% 1.1 实验配置
Maxit = 10;                 % 每张图运行次数（平衡时间与统计显著性）
num_images_limited = 10;    % 对比实验选取前5张图（节省时间）
save_results = true;        % 是否保存结果

% 1.2 算法统一参数
sizepop = 50;              % 种群/粒子群规模
maxgen = 50;               % 最大迭代次数
num_runs = Maxit;          % 独立运行次数

% 1.3 BPNN结构参数
inputnum = 9;
hidden_layers = [9];
outputnum = 1;

% 智能判断隐层结构并自动计算numsum
if length(hidden_layers) == 1
    % 单隐层结构
    numsum = inputnum*hidden_layers(1) + ...   % 输入层→隐层权重
             hidden_layers(1) + ...            % 隐层偏置
             hidden_layers(1)*outputnum + ...  % 隐层→输出层权重
             outputnum;                       % 输出层偏置
    fprintf('单隐层结构: %d 个节点, 参数总数: %d\n', hidden_layers(1), numsum);
    
elseif length(hidden_layers) == 2
    % 双隐层结构
    numsum = inputnum*hidden_layers(1) + ...      % 输入层→隐层1
             hidden_layers(1) + ...               % 隐层1偏置
             hidden_layers(1)*hidden_layers(2) + ... % 隐层1→隐层2
             hidden_layers(2) + ...               % 隐层2偏置
             hidden_layers(2)*outputnum + ...     % 隐层2→输出层
             outputnum;                          % 输出层偏置
    fprintf('双隐层结构: [%d, %d], 参数总数: %d\n', hidden_layers(1), hidden_layers(2), numsum);
    
else
    error('仅支持单隐层或双隐层结构，当前隐层数: %d', length(hidden_layers));
end


% 1.4 图像处理参数
picsize = [90, 90];
gauss_kernel_size = 9;
gauss_sigma = 1;


% 1.5 PSO/IPSO专用参数
c1 = 1.5; c2 = 1.5;
w_init = 0.9; w_final = 0.3;
v_max = 0.5; v_min = -0.5;
pos_max = 1; pos_min = -1;
perturb_trigger_ratio = 0.7;
perturb_std = 0.1;
p = 1.5;  % 惯性权重幂指数

% 1.6 GA参数
ga_params = struct();
ga_params.pc = 0.8;        % 交叉概率
ga_params.pm = 0.05;       % 变异概率
ga_params.select_ratio = 0.5;  % 选择比例

% 1.7 GWO参数
gwo_params = struct();
gwo_params.a_init = 2;     % 收敛因子初始值
gwo_params.a_final = 0;    % 收敛因子最终值


% 1.8 WOA参数
woa_params = struct();
woa_params.a_init = 2;
woa_params.a_final = 0;
woa_params.b = 1;          % 螺旋形状常数

% 1.9 RIME参数
rime_params = struct();
rime_params.R = 5;         % 软霜冰参数
rime_params.K = 0.1;       % 附着参数
rime_params.E_init = 1;    % 环境因子初始
rime_params.E_final = 0;   % 环境因子最终


%% 1.10 CPO参数 (2024最新算法)
cpo_params = struct();
cpo_params.Tf = 0.8;        % 跟踪因子
cpo_params.N_min = 5;       % 最小种群规模
cpo_params.alpha = 0.2;     % 防御角度参数
cpo_params.beta = 1.5;      % Levy飞行参数

%% 1.11 HBA参数 (2022顶刊)
hba_params = struct();
hba_params.beta = 6;          % 嗅觉因子
hba_params.C = 2;             % 常数
hba_params.alpha_init = 0.98; % 密度因子初始
hba_params.alpha_final = 0.1; % 密度因子最终

%% 2. 获取测试图像
script_path = fileparts(mfilename('fullpath'));
pic_dir = fullfile(script_path, 'ipso', 'valid');

% 如果默认目录不存在，使用当前目录
if ~exist(pic_dir, 'dir')
    pic_dir = pwd;
end

% 获取图像文件
image_files = dir(fullfile(pic_dir, '*.tif'));
if isempty(image_files)
    image_files = dir(fullfile(pic_dir, '*.png'));
end
if isempty(image_files)
    image_files = dir(fullfile(pic_dir, '*.jpg'));
end

if isempty(image_files)
    error('No test images found');
end

% 取前N张图进行对比
all_sim_picname = {image_files(1:min(num_images_limited, length(image_files))).name};
num_images = length(all_sim_picname);
fprintf('Selected %d images for comparison test\n', num_images);

%% 3. 初始化结果存储结构
% 算法列表
algorithms = {'PSO', 'IPSO', 'GA', 'HBA', 'WOA', 'RIME'};
num_algorithms = length(algorithms);

% 每张图的结果
Results = cell(num_images, 1);
for img = 1:num_images
    Results{img}.name = all_sim_picname{img};
    Results{img}.best_fitness = zeros(num_algorithms, num_runs);
    Results{img}.trace = cell(num_algorithms, 1);
    Results{img}.time = zeros(num_algorithms, num_runs);
    Results{img}.best_chrom = cell(num_algorithms, 1);
end

% 总体统计表
Comparison_Table = cell(num_images * num_algorithms + 1, 12);
Comparison_Table(1, :) = {'Image', 'Algorithm', 'Best', 'Mean', 'Median', 'Std', ...
                          'Best_Rank', 'Mean_Rank', 'Time(min)', 'Converge_Gen', ...
                          'Improvement_vs_PSO(%)', 'Significance'};

%% 4. 主循环 - 每张图像
for img_idx = 1:num_images
    fprintf('\n%s\n', repmat('=', 1, 60));
    fprintf('Processing Image %d/%d: %s\n', img_idx, num_images, all_sim_picname{img_idx});
    fprintf('%s\n', repmat('=', 1, 60));
    
    %% 4.1 读取并预处理图像
    picname = fullfile(pic_dir, all_sim_picname{img_idx});
    image_orgin = imread(picname);
    
    % 灰度化、归一化
    if size(image_orgin, 3) == 3
        image_orgin = rgb2gray(image_orgin);
    end
    image_resized = imresize(image_orgin, picsize);
    image_resized = double(image_resized) / 256;
    
    % 退化处理
    w_gauss = fspecial('gaussian', gauss_kernel_size, gauss_sigma);
    image_blurred = imfilter(image_resized, w_gauss, 'replicate');
    image_degraded = image_blurred;  % 不加噪声，保持一致性
    
    % 生成训练数据
    [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_resized, inputnum);
    
    % 初始化BP网络
    net = newff(P_Matrix, T_Matrix, hidden_layers);
    net.trainParam.epochs = 1000;
    net.trainParam.lr = 0.1;
    net.trainParam.goal = 1e-5;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % 适应度函数
    % fobj = @(x) cal_fitness_bpnn(x, inputnum, hidden_layers, outputnum, net, P_Matrix, T_Matrix);
    fobj = @(x) cal_fitness1(x, inputnum, hidden_layers, outputnum, net, P_Matrix, T_Matrix);
    % fobj = @(x) cal_fitness(x, inputnum, hiddennum, outputnum, net, P_Matrix, T_Matrix);
    
    %% 4.2 各算法独立运行num_runs次
    for run = 1:num_runs
        if mod(run, 5) == 0
            fprintf('  Run %d/%d\n', run, num_runs);
        end
        
        % 统一随机种子，保证公平对比
        rng(run * img_idx, 'twister');
        
        %% PSO
        t_start = tic;
        fprintf('PSO');
        [bestchrom, bestfitness, trace] = PSO_standard(...
            sizepop, maxgen, numsum, fobj, c1, c2, w_init, ...
            v_max, v_min, pos_max, pos_min);
        Results{img_idx}.time(1, run) = toc(t_start);
        Results{img_idx}.best_fitness(1, run) = bestfitness;
        Results{img_idx}.trace{1}(:, run) = trace;
        Results{img_idx}.best_chrom{1}{run} = bestchrom;
        
        %% IPSO
        t_start = tic;
        fprintf('IPSO');
        [bestchrom, bestfitness, trace] = PSO_improved(...
            sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, ...
            v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p);
        Results{img_idx}.time(2, run) = toc(t_start);
        Results{img_idx}.best_fitness(2, run) = bestfitness;
        Results{img_idx}.trace{2}(:, run) = trace;
        Results{img_idx}.best_chrom{2}{run} = bestchrom;
        
        %% GA
        t_start = tic;
        fprintf('GA');
        [bestchrom, bestfitness, trace] = GA_optimizer(...
            sizepop, maxgen, numsum, fobj, pos_min, pos_max, ga_params);
        Results{img_idx}.time(3, run) = toc(t_start);
        Results{img_idx}.best_fitness(3, run) = bestfitness;
        Results{img_idx}.trace{3}(:, run) = trace;
        Results{img_idx}.best_chrom{3}{run} = bestchrom;
        
        %% GWO
        % t_start = tic;
        % fprintf('GWO');
        % [bestchrom, bestfitness, trace] = GWO_optimizer(...
        %     sizepop, maxgen, numsum, fobj, pos_min, pos_max, gwo_params);
        % Results{img_idx}.time(4, run) = toc(t_start);
        % Results{img_idx}.best_fitness(4, run) = bestfitness;
        % Results{img_idx}.trace{4}(:, run) = trace;
        % Results{img_idx}.best_chrom{4}{run} = bestchrom;

        %% CPO (2024) - 替换原来的GWO
        % t_start = tic;
        % fprintf('GWO');
        % [bestchrom, bestfitness, trace] = CPO_optimizer(...
        %     sizepop, maxgen, numsum, fobj, pos_min, pos_max, cpo_params);
        % Results{img_idx}.time(4, run) = toc(t_start);
        % Results{img_idx}.best_fitness(4, run) = bestfitness;
        % Results{img_idx}.trace{4}(:, run) = trace;
        % Results{img_idx}.best_chrom{4}{run} = bestchrom;

        %% HBA
        t_start = tic;
        [bestchrom, bestfitness, trace] = HBA_optimizer(...
            sizepop, maxgen, numsum, fobj, pos_min, pos_max, hba_params);
        Results{img_idx}.time(4, run) = toc(t_start);
        Results{img_idx}.best_fitness(4, run) = bestfitness;
        Results{img_idx}.trace{4}(:, run) = trace;
        Results{img_idx}.best_chrom{4}{run} = bestchrom;
        
        %% WOA
        t_start = tic;
        fprintf('WOA');
        [bestchrom, bestfitness, trace] = WOA_optimizer(...
            sizepop, maxgen, numsum, fobj, pos_min, pos_max, woa_params);
        Results{img_idx}.time(5, run) = toc(t_start);
        Results{img_idx}.best_fitness(5, run) = bestfitness;
        Results{img_idx}.trace{5}(:, run) = trace;
        Results{img_idx}.best_chrom{5}{run} = bestchrom;
        
        %% RIME (2023)
        t_start = tic;
        fprintf('RIME');
        [bestchrom, bestfitness, trace] = RIME_optimizer(...
            sizepop, maxgen, numsum, fobj, pos_min, pos_max, rime_params);
        Results{img_idx}.time(6, run) = toc(t_start);
        Results{img_idx}.best_fitness(6, run) = bestfitness;
        Results{img_idx}.trace{6}(:, run) = trace;
        Results{img_idx}.best_chrom{6}{run} = bestchrom;
    end
    
    %% 4.3 统计当前图像结果
    fprintf('\n--- Image %d Statistics ---\n', img_idx);
    for a = 1:num_algorithms
        % 基本统计量
        best_val = min(Results{img_idx}.best_fitness(a, :));
        mean_val = mean(Results{img_idx}.best_fitness(a, :));
        median_val = median(Results{img_idx}.best_fitness(a, :));
        std_val = std(Results{img_idx}.best_fitness(a, :));
        time_val = mean(Results{img_idx}.time(a, :));
        
        % 收敛代数估计（达到最佳值95%所需的迭代次数）
        converge_gen = estimate_convergence_gen(Results{img_idx}.trace{a}, maxgen);
        
        % 相对于PSO的提升率
        pso_best = min(Results{img_idx}.best_fitness(1, :));
        if a == 1
            impr_vs_pso = 0;
            sig_str = '-';
        else
            impr_vs_pso = (pso_best - best_val) / pso_best * 100;
            % 显著性检验
            [h, p] = ttest2(Results{img_idx}.best_fitness(1, :), ...
                           Results{img_idx}.best_fitness(a, :));
            if p < 0.001
                sig_str = '***';
            elseif p < 0.01
                sig_str = '**';
            elseif p < 0.05
                sig_str = '*';
            else
                sig_str = 'ns';
            end
        end
        
        % 打印结果
        fprintf('  %-6s: Best=%.4e, Mean=%.4e, Std=%.4e, Time=%.2fs, Conv=%d, IR=%.2f%% %s\n', ...
                algorithms{a}, best_val, mean_val, std_val, time_val, ...
                converge_gen, impr_vs_pso, sig_str);
        
        % 填充对比表
        row_idx = (img_idx-1)*num_algorithms + a + 1;
        Comparison_Table(row_idx, :) = {
            all_sim_picname{img_idx}, ...  % Image
            algorithms{a}, ...             % Algorithm
            best_val, ...                 % Best
            mean_val, ...                % Mean
            median_val, ...              % Median
            std_val, ...                % Std
            0, ...                      % Best_Rank (待填充)
            0, ...                      % Mean_Rank (待填充)
            time_val, ...               % Time
            converge_gen, ...           % Converge_Gen
            impr_vs_pso, ...            % Improvement vs PSO
            sig_str                     % Significance
        };
    end
    
    % 计算排名（基于均值和最优值）
    best_vals = zeros(num_algorithms, 1);
    mean_vals = zeros(num_algorithms, 1);
    for a = 1:num_algorithms
        best_vals(a) = min(Results{img_idx}.best_fitness(a, :));
        mean_vals(a) = mean(Results{img_idx}.best_fitness(a, :));
    end
    
    [~, best_rank] = sort(best_vals);
    [~, mean_rank] = sort(mean_vals);
    
    for a = 1:num_algorithms
        row_idx = (img_idx-1)*num_algorithms + a + 1;
        Comparison_Table{row_idx, 7} = find(best_rank == a);
        Comparison_Table{row_idx, 8} = find(mean_rank == a);
    end
end



%% 6. 保存结果
if save_results
    % 确保目录存在
    if ~exist('comparison_results', 'dir')
        mkdir('comparison_results');
    end
    
    % 6.1 保存详细结果
    matname = sprintf('IPSO_Comparison_Full_Results_%s.mat', ...
                          datestr(now, 'yyyymmdd'));
    save(matname, ...
         'Results', 'Comparison_Table', 'algorithms', 'num_images');

        
    % 6.2 导出对比表到Excel
    try
        filename = sprintf('comparison_results/IPSO_Comparison_Table_%s.xlsx', ...
                          datestr(now, 'yyyymmdd'));
        writecell(Comparison_Table, filename);
        fprintf('Comparison table saved to: %s\n', filename);
    catch ME
        fprintf('Warning: Could not save Excel file: %s\n', ME.message);
    end
    
    % 6.3 生成LaTeX表格
    generate_latex_table(Comparison_Table, algorithms, num_images);
end

%% 5. 绘制对比图表
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('Generating comparison charts...\n');
fprintf('%s\n', repmat('=', 1, 60));

% 5.1 收敛曲线对比图（每张图）
for img_idx = 1:num_images
    plot_convergence_comparison(Results{img_idx}, algorithms, maxgen, img_idx);
end

% 5.2 算法性能箱线图
plot_performance_boxplot(Results, algorithms, num_images);

% 5.3 算法排名雷达图
plot_algorithm_ranking(Comparison_Table, algorithms, num_images);

% 5.4 收敛速度对比
plot_convergence_speed(Results, algorithms, maxgen, num_images);

% 5.5 Friedman检验排名图
plot_friedman_test(Comparison_Table, algorithms, num_images, num_algorithms);

%% 7. 输出总结
t_total_end = clock;
total_time = etime(t_total_end, t_total) / 60;
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('COMPARISON EXPERIMENT COMPLETED\n');
fprintf('Total time: %.2f minutes\n', total_time);
fprintf('Algorithms compared: %s\n', strjoin(algorithms, ', '));
fprintf('Images processed: %d\n', num_images);
fprintf('Runs per algorithm per image: %d\n', num_runs);
fprintf('%s\n', repmat('=', 1, 60));
end