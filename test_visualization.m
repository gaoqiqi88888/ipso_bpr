%% ==================== 测试代码 - 只测试显示部分 ====================
function test_visualization()
%% 0. 清空环境
clear; close all; clc;
fprintf('========== 测试可视化模块 ==========\n');

%% 1. 创建测试目录
if ~exist('comparison_results', 'dir')
    mkdir('comparison_results');
end

%% 2. 生成模拟数据
fprintf('生成模拟实验数据...\n');

% 算法设置
algorithms = {'PSO', 'IPSO', 'GA', 'GWO', 'WOA', 'RIME'};
num_algorithms = length(algorithms);
num_images = 3;  % 测试3张图
num_runs = 10;   % 每张图运行10次
maxgen = 50;     % 迭代次数

% 创建模拟Results结构
Results = cell(num_images, 1);
for img = 1:num_images
    Results{img}.name = sprintf('test_image_%d.tif', img);
    Results{img}.best_fitness = zeros(num_algorithms, num_runs);
    Results{img}.trace = cell(num_algorithms, 1);
    Results{img}.time = zeros(num_algorithms, num_runs);
    
    % 为每个算法生成模拟数据
    for a = 1:num_algorithms
        % 生成不同性能水平的模拟数据
        base_value = 30 + randn * 5;
        switch a
            case 1  % PSO
                results = base_value + randn(num_runs, 1) * 3;
            case 2  % IPSO
                results = base_value - 5 + randn(num_runs, 1) * 2;
            case 3  % GA
                results = base_value + 15 + randn(num_runs, 1) * 8;
            case 4  % GWO
                results = base_value - 8 + randn(num_runs, 1) * 1.5;
            case 5  % WOA
                results = base_value + 8 + randn(num_runs, 1) * 4;
            case 6  % RIME
                results = base_value + 12 + randn(num_runs, 1) * 5;
        end
        Results{img}.best_fitness(a, :) = results';
        
        % 生成收敛曲线 - 修正版
        trace_data = zeros(maxgen, num_runs);
        for r = 1:num_runs
            % trace是50×1向量
            trace = linspace(results(r), results(r)*0.6, maxgen)' + randn(maxgen,1)*0.5;
            trace_data(:, r) = trace;  % 现在维度匹配：50×1 = 50×1
        end
        Results{img}.trace{a} = trace_data;
        Results{img}.time(a, :) = 10 + randn(num_runs, 1) * 2;
    end
end

% 创建模拟Comparison_Table
Comparison_Table = cell(num_images * num_algorithms + 1, 13);
Comparison_Table(1, :) = {'Image', 'Algorithm', 'Best', 'Mean', 'Median', 'Std', ...
                          'Best_Rank', 'Mean_Rank', 'Time(s)', 'Converge_Gen', ...
                          'Improvement_vs_PSO(%)', 'p_value', 'Significance'};

for img = 1:num_images
    for a = 1:num_algorithms
        row_idx = (img-1)*num_algorithms + a + 1;
        data = Results{img}.best_fitness(a, :);
        pso_data = Results{img}.best_fitness(1, :);
        
        [h, p] = ttest2(pso_data, data);
        if p < 0.001
            sig = '***';
        elseif p < 0.01
            sig = '**';
        elseif p < 0.05
            sig = '*';
        else
            sig = 'ns';
        end
        
        Comparison_Table(row_idx, :) = {
            sprintf('test_image_%d.tif', img), ...
            algorithms{a}, ...
            min(data), ...
            mean(data), ...
            median(data), ...
            std(data), ...
            randi(num_algorithms), ...
            randi(num_algorithms), ...
            mean(Results{img}.time(a, :)), ...
            randi([20, 40]), ...
            (mean(pso_data) - mean(data)) / mean(pso_data) * 100, ...
            p, ...
            sig
        };
    end
end

fprintf('模拟数据生成完成！\n\n');

%% 3. 测试各个绘图函数
fprintf('========== 开始测试绘图函数 ==========\n');

%% 3.1 测试收敛曲线图
fprintf('\n📈 测试1: 收敛曲线对比图...\n');
try
    plot_convergence_comparison(Results{1}, algorithms, maxgen, 1);
    fprintf('  ✅ 收敛曲线图绘制成功\n');
catch ME
    fprintf('  ❌ 收敛曲线图失败: %s\n', ME.message);
end

%% 3.2 测试性能箱线图
fprintf('\n📊 测试2: 性能箱线图...\n');
try
    plot_performance_boxplot(Results, algorithms, num_images);
    fprintf('  ✅ 箱线图绘制成功\n');
catch ME
    fprintf('  ❌ 箱线图失败: %s\n', ME.message);
end

%% 3.3 测试算法排名柱状图
fprintf('\n📊 测试3: 算法排名柱状图...\n');
try
    plot_algorithm_ranking(Comparison_Table, algorithms, num_images, num_algorithms);
    fprintf('  ✅ 排名柱状图绘制成功\n');
catch ME
    fprintf('  ❌ 排名柱状图失败: %s\n', ME.message);
end

%% 3.4 测试Friedman检验排名图
fprintf('\n📊 测试4: Friedman检验排名图...\n');
try
    plot_friedman_test(Comparison_Table, algorithms, num_images, num_algorithms);
    fprintf('  ✅ Friedman检验图绘制成功\n');
catch ME
    fprintf('  ❌ Friedman检验图失败: %s\n', ME.message);
end

%% 3.5 测试表格导出
fprintf('\n📋 测试5: 表格导出功能...\n');
try
    write_table_to_excel(Comparison_Table, 'comparison_results/test_comparison_table.xlsx');
    fprintf('  ✅ Excel表格导出成功\n');
catch ME
    fprintf('  ❌ Excel表格导出失败: %s\n', ME.message);
end

try
    generate_text_table(Comparison_Table, algorithms, num_images);
    fprintf('  ✅ 文本表格生成成功\n');
catch ME
    fprintf('  ❌ 文本表格生成失败: %s\n', ME.message);
end

%% 4. 测试总结
fprintf('\n%s\n', repmat('=', 1, 60));
fprintf('测试完成！\n');
fprintf('结果保存在: comparison_results/\n');
fprintf('%s\n', repmat('=', 1, 60));

%% 列出生成的文件
fprintf('\n生成的文件列表:\n');
files = dir('comparison_results/*.png');
for i = 1:length(files)
    fprintf('  📄 %s\n', files(i).name);
end
files = dir('comparison_results/*.fig');
for i = 1:length(files)
    fprintf('  📄 %s\n', files(i).name);
end
files = dir('comparison_results/*.xlsx');
for i = 1:length(files)
    fprintf('  📄 %s\n', files(i).name);
end
files = dir('comparison_results/*.txt');
for i = 1:length(files)
    fprintf('  📄 %s\n', files(i).name);
end

fprintf('\n✅ 所有测试完成！\n');