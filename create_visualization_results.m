function create_visualization_results(results, param_combinations, total_time)
    % 创建汇总分析图
    
    num_images = length(fieldnames(results));
    num_combinations = size(param_combinations, 1);
    
    % 提取组合名称
    combo_names = cell(num_combinations, 1);
    for i = 1:num_combinations
        combo_names{i} = sprintf('R%.1f-S%.2f', ...
            param_combinations(i,1), param_combinations(i,2));
    end
    
    % 1. 热力图：不同参数组合的平均性能
    fig1 = figure('Position', [100, 100, 1200, 800], 'Name', '参数性能热力图', 'NumberTitle', 'off');
    
    % 准备数据
    mean_improvement_matrix = zeros(num_images, num_combinations);
    best_improvement_matrix = zeros(num_images, num_combinations);
    median_improvement_matrix = zeros(num_images, num_combinations);
    
    for img_idx = 1:num_images
        img_field = sprintf('image_%d', img_idx);
        if isfield(results, img_field)
            mean_improvement_matrix(img_idx, :) = results.(img_field).improvement_mean';
            best_improvement_matrix(img_idx, :) = results.(img_field).improvement_best';
            median_improvement_matrix(img_idx, :) = results.(img_field).improvement_median';
        end
    end
    
    % 绘制平均改进率热力图
    subplot(2,2,1);
    imagesc(mean_improvement_matrix);
    colorbar;
    xlabel('参数组合');
    ylabel('测试图像');
    title('平均适应度改进率 (%) 热力图');
    set(gca, 'XTick', 1:num_combinations, 'XTickLabel', combo_names, 'XTickLabelRotation', 45);
    ylabel('图像编号');
    colormap(jet);
    
    % 添加数值标注
    for i = 1:num_images
        for j = 1:num_combinations
            text(j, i, sprintf('%.1f', mean_improvement_matrix(i,j)), ...
                'HorizontalAlignment', 'center', 'FontSize', 8, ...
                'Color', mean_improvement_matrix(i,j) >= 0 ? 'w' : 'k');
        end
    end
    
    % 2. 最佳改进率柱状图
    subplot(2,2,2);
    avg_best_improvement = mean(best_improvement_matrix, 1);
    bar(1:num_combinations, avg_best_improvement, 'FaceColor', [0.2, 0.4, 0.8]);
    hold on;
    % 标出正值（提升）和负值（下降）
    positive_idx = avg_best_improvement > 0;
    negative_idx = avg_best_improvement < 0;
    bar(find(positive_idx), avg_best_improvement(positive_idx), 'FaceColor', 'g');
    bar(find(negative_idx), avg_best_improvement(negative_idx), 'FaceColor', 'r');
    hold off;
    
    xlabel('参数组合');
    ylabel('最佳适应度改进率 (%)');
    title('各参数组合平均最佳改进率');
    set(gca, 'XTick', 1:num_combinations, 'XTickLabel', combo_names, 'XTickLabelRotation', 45);
    grid on;
    
    % 3. 中位数改进率对比
    subplot(2,2,3);
    avg_median_improvement = mean(median_improvement_matrix, 1);
    error_data = std(median_improvement_matrix, 0, 1);
    errorbar(1:num_combinations, avg_median_improvement, error_data, 'o-', ...
        'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    
    xlabel('参数组合');
    ylabel('中位数适应度改进率 (%)');
    title('各参数组合中位数改进率（含标准差）');
    set(gca, 'XTick', 1:num_combinations, 'XTickLabel', combo_names, 'XTickLabelRotation', 45);
    grid on;
    
    % 4. 3D曲面图：触发比例 vs 扰动强度 vs 平均改进率
    subplot(2,2,4);
    trigger_ratios = param_combinations(:,1);
    std_devs = param_combinations(:,2);
    avg_mean_improvement = mean(mean_improvement_matrix, 1)';
    
    % 为曲面插值准备数据
    [X, Y] = meshgrid(unique(trigger_ratios), unique(std_devs));
    Z = griddata(trigger_ratios, std_devs, avg_mean_improvement, X, Y);
    
    surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    hold on;
    scatter3(trigger_ratios, std_devs, avg_mean_improvement, 50, 'r', 'filled');
    hold off;
    
    xlabel('触发比例');
    ylabel('扰动强度');
    zlabel('平均改进率 (%)');
    title('参数空间性能3D图');
    colorbar;
    view(-45, 30);
    
    % 保存图形
    saveas(fig1, 'ipso/parameter_performance_heatmap.png');
    saveas(fig1, 'ipso/parameter_performance_heatmap.fig');
    
    % 5. 详细性能对比图（折线图）
    fig2 = figure('Position', [200, 200, 1400, 500], 'Name', '详细性能对比', 'NumberTitle', 'off');
    
    % 子图1：三种改进率对比
    subplot(1,3,1);
    hold on;
    plot(1:num_combinations, avg_best_improvement, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '最佳改进率');
    plot(1:num_combinations, avg_mean_improvement, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '平均改进率');
    plot(1:num_combinations, avg_median_improvement, 'g-^', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', '中位数改进率');
    hold off;
    
    xlabel('参数组合');
    ylabel('改进率 (%)');
    title('三种改进率对比');
    set(gca, 'XTick', 1:num_combinations, 'XTickLabel', combo_names, 'XTickLabelRotation', 45);
    legend('Location', 'best');
    grid on;
    
    % 子图2：稳定性分析（标准差）
    subplot(1,3,2);
    std_of_improvement = std(mean_improvement_matrix, 0, 1);
    bar(1:num_combinations, std_of_improvement, 'FaceColor', [0.8, 0.4, 0.2]);
    
    xlabel('参数组合');
    ylabel('改进率标准差');
    title('各参数组合稳定性分析');
    set(gca, 'XTick', 1:num_combinations, 'XTickLabel', combo_names, 'XTickLabelRotation', 45);
    grid on;
    
    % 找出性能最好的3个组合
    [~, sorted_idx] = sort(avg_mean_improvement, 'descend');
    top_combinations = sorted_idx(1:min(3, num_combinations));
    
    % 子图3：最优参数组合性能展示
    subplot(1,3,3);
    colors = {'r', 'g', 'b'};
    hold on;
    for k = 1:length(top_combinations)
        idx = top_combinations(k);
        plot(1:num_images, mean_improvement_matrix(:, idx), ...
            [colors{k} '-o'], 'LineWidth', 2, 'MarkerSize', 6, ...
            'DisplayName', sprintf('%s (%.1f%%)', combo_names{idx}, avg_mean_improvement(idx)));
    end
    hold off;
    
    xlabel('测试图像');
    ylabel('改进率 (%)');
    title(sprintf('最优参数组合在各图像的改进率'));
    legend('Location', 'best');
    grid on;
    
    % 保存图形
    saveas(fig2, 'ipso/detailed_performance_comparison.png');
    saveas(fig2, 'ipso/detailed_performance_comparison.fig');
    
    fprintf('可视化结果已生成并保存！\n');
end