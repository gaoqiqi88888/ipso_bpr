%********************************************************************
% 绘制Friedman检验排名图
%********************************************************************
function plot_friedman_test(Comparison_Table, algorithms, num_images, num_algorithms)
% 提取排名矩阵
ranks = zeros(num_images, num_algorithms);
for img = 1:num_images
    for a = 1:num_algorithms
        row_idx = (img-1)*num_algorithms + a + 1;
        ranks(img, a) = Comparison_Table{row_idx, 8};  % Mean rank
    end
end

% 计算平均排名
mean_ranks = mean(ranks, 1);

% 排序
[sorted_ranks, sorted_idx] = sort(mean_ranks);
sorted_algorithms = algorithms(sorted_idx);

% 绘制
figure('Position', [100, 100, 900, 500]);
barh(sorted_ranks, 'FaceColor', [0.3, 0.6, 0.8]);
xlabel('Average Rank', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Algorithm', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Friedman Test: Algorithm Ranking (Lower is Better)'), ...
      'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'YTickLabel', sorted_algorithms, 'FontSize', 11);
grid on;

% 添加数值标签
for i = 1:length(sorted_ranks)
    text(sorted_ranks(i) + 0.1, i, sprintf('%.2f', sorted_ranks(i)), ...
         'FontSize', 10, 'FontWeight', 'bold');
end

saveas(gcf, 'comparison_results/friedman_ranking.png');
saveas(gcf, 'comparison_results/friedman_ranking.fig');
close(gcf);
end