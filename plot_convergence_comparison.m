%********************************************************************
% 绘制算法收敛曲线对比图
%********************************************************************
function plot_convergence_comparison(Results, algorithms, maxgen, img_idx)
figure('Position', [100, 100, 1000, 600], 'Visible', 'off');
colors = lines(length(algorithms));
line_styles = {'-', '--', '-.', ':', '-', '--'};

for a = 1:length(algorithms)
    % 计算平均收敛曲线
    trace_data = Results.trace{a};
    mean_trace = mean(trace_data, 2);
    
    % 绘制主曲线
    semilogy(1:maxgen, mean_trace, ...
             'Color', colors(a, :), ...
             'LineStyle', line_styles{a}, ...
             'LineWidth', 2);
    hold on;
end

hold off;
grid on;
xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Fitness Value (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Convergence Comparison - %s', Results.name), ...
      'FontSize', 14, 'FontWeight', 'bold');
legend(algorithms, 'Location', 'northeast', 'FontSize', 10);
set(gca, 'FontSize', 11, 'LineWidth', 1.2);

% 保存
saveas(gcf, sprintf('comparison_results/convergence_%s.png', ...
       strrep(Results.name, '.', '_')));
saveas(gcf, sprintf('comparison_results/convergence_%s.fig', ...
       strrep(Results.name, '.', '_')));
close(gcf);
end