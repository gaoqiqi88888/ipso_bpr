%% 算法排名柱状图 - 修正版
function plot_algorithm_ranking(Comparison_Table, algorithms, num_images)
    % 确保目录存在
    if ~exist('comparison_results', 'dir')
        mkdir('comparison_results');
    end
    
    num_algorithms = length(algorithms);
    
    % 计算平均排名
    mean_ranks = zeros(num_algorithms, 1);
    std_ranks = zeros(num_algorithms, 1);
    
    for a = 1:num_algorithms
        ranks = [];
        for img = 1:num_images
            row_idx = (img-1)*num_algorithms + a + 1;
            % 确保行索引不超出范围
            if size(Comparison_Table, 1) >= row_idx
                % 获取Mean_Rank（第8列）
                rank_value = Comparison_Table{row_idx, 8};
                if ~isempty(rank_value) && isnumeric(rank_value)
                    ranks = [ranks; rank_value];
                end
            end
        end
        mean_ranks(a) = mean(ranks);
        std_ranks(a) = std(ranks);
    end
    
    % 绘制柱状图
    figure('Position', [100, 100, 900, 600]);
    
    % 绘制柱状图
    b = bar(mean_ranks, 'FaceColor', [0.3, 0.6, 0.8], 'EdgeColor', 'k', 'LineWidth', 1);
    
    % 添加误差棒
    hold on;
    x_pos = 1:num_algorithms;
    errorbar(x_pos, mean_ranks, std_ranks, 'k', 'LineStyle', 'none', 'LineWidth', 1);
    hold off;
    
    % 设置坐标轴
    xlabel('Algorithm', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Average Rank (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Algorithm Ranking Comparison Across All Images', ...
          'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:num_algorithms, 'XTickLabel', algorithms, 'FontSize', 11);
    grid on;
    
    % 添加数值标签
    for i = 1:num_algorithms
        text(i, mean_ranks(i) + 0.1, sprintf('%.2f', mean_ranks(i)), ...
             'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
    end
    
    % 保存
    saveas(gcf, 'comparison_results/algorithm_ranking.png');
    saveas(gcf, 'comparison_results/algorithm_ranking.fig');
    close(gcf);
    
    fprintf('  ✅ 算法排名柱状图绘制成功\n');
end