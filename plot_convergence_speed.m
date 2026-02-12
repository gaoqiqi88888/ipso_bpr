%% 绘制收敛速度对比图
function plot_convergence_speed(Results, algorithms, maxgen, num_images)
    % 确保目录存在
    if ~exist('comparison_results', 'dir')
        mkdir('comparison_results');
    end
    
    num_algorithms = length(algorithms);
    
    % 计算每个算法的平均收敛代数
    mean_converge = zeros(num_algorithms, 1);
    std_converge = zeros(num_algorithms, 1);
    
    for a = 1:num_algorithms
        converge_gens = [];
        for img = 1:num_images
            % 获取当前算法在当前图像的收敛曲线
            trace_data = Results{img}.trace{a};
            if isempty(trace_data)
                continue;
            end
            
            % 对每次运行计算收敛代数
            for run = 1:size(trace_data, 2)
                trace = trace_data(:, run);
                final_value = trace(end);
                threshold = final_value * 1.05;  % 5%阈值
                
                converge_gen = find(trace <= threshold, 1);
                if isempty(converge_gen)
                    converge_gen = maxgen;
                end
                converge_gens = [converge_gens; converge_gen];
            end
        end
        mean_converge(a) = mean(converge_gens);
        std_converge(a) = std(converge_gens);
    end
    
    % 绘制收敛速度对比图
    figure('Position', [100, 100, 1000, 600]);
    
    % 绘制柱状图
    b = bar(mean_converge, 'FaceColor', [0.2, 0.6, 0.5], 'EdgeColor', 'k', 'LineWidth', 1);
    
    % 添加误差棒
    hold on;
    x_pos = 1:num_algorithms;
    errorbar(x_pos, mean_converge, std_converge, 'k', 'LineStyle', 'none', 'LineWidth', 1);
    hold off;
    
    % 设置坐标轴
    xlabel('Algorithm', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Convergence Generation', 'FontSize', 12, 'FontWeight', 'bold');
    title('Convergence Speed Comparison (Lower is Better)', ...
          'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:num_algorithms, 'XTickLabel', algorithms, 'FontSize', 11);
    grid on;
    
    % 添加数值标签
    for i = 1:num_algorithms
        text(i, mean_converge(i) + 1.5, sprintf('%.1f', mean_converge(i)), ...
             'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
    end
    
    % 保存
    saveas(gcf, 'comparison_results/convergence_speed.png');
    saveas(gcf, 'comparison_results/convergence_speed.fig');
    close(gcf);
    
    % 打印统计信息
    fprintf('\n📈 收敛速度统计（平均收敛代数）:\n');
    for a = 1:num_algorithms
        fprintf('  %-6s: %.1f ± %.1f\n', algorithms{a}, mean_converge(a), std_converge(a));
    end
end