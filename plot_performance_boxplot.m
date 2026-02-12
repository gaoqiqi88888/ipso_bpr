%% 绘制算法性能箱线图 - 最终修正版
function plot_performance_boxplot(Results, algorithms, num_images)
    % 确保目录存在
    if ~exist('comparison_results', 'dir')
        mkdir('comparison_results');
    end
    
    % 检查数据量
    first_img = Results{1};
    num_runs = size(first_img.best_fitness, 2);
    
    if num_runs < 2
        fprintf('⚠️ 箱线图需要至少2次运行，当前=%d次，跳过\n', num_runs);
        return;
    end
    
    figure('Position', [100, 100, 1200, 600]);
    
    % 收集所有图像的所有运行结果
    all_data = [];
    group_labels = {};
    
    for img = 1:num_images
        for a = 1:length(algorithms)
            % 获取当前算法在当前图像上的所有运行结果
            current_data = Results{img}.best_fitness(a, :)';
            current_length = length(current_data);
            
            % 添加到总数据
            all_data = [all_data; current_data];
            
            % 创建对应的标签 - 确保长度一致！
            current_labels = repmat({sprintf('%s_Img%d', algorithms{a}, img)}, current_length, 1);
            group_labels = [group_labels; current_labels];
        end
    end
    
    % 验证数据长度一致
    fprintf('📊 箱线图数据点数量: %d, 标签数量: %d\n', length(all_data), length(group_labels));
    
    % 创建分组箱线图
    boxplot(all_data, group_labels);
    
    ylabel('Best Fitness Value', 'FontSize', 12, 'FontWeight', 'bold');
    title('Performance Distribution Across All Images', ...
          'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 10, 'XTickLabelRotation', 45);
    
    % 保存
    saveas(gcf, 'comparison_results/boxplot_performance.png');
    saveas(gcf, 'comparison_results/boxplot_performance.fig');
    close(gcf);
    
    fprintf('✅ 箱线图绘制完成\n');
end