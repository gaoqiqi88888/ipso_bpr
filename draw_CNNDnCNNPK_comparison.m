% 绘制IPSO-BPR论文对比图 - 分开保存版本
clear; clc; close all;

%% 1. 设置路径和参数
% 数据文件路径
dataFile = 'C:\ipso_bpr_v3\DnCNN_Comparisons\all_config_comparison_20260210_113930ipso paper2 user.xlsx';

% 输出目录
outputDir = 'C:\ipso_bpr_v3\paper';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
    fprintf('创建输出目录: %s\n', outputDir);
end

% 获取当前日期时间
currentDateTime = datestr(now, 'yyyymmdd_HHMMSS');
fprintf('当前时间戳: %s\n', currentDateTime);

% 分辨率设置
dpi = 1200;

%% 2. 读取数据
fprintf('正在读取数据文件: %s\n', dataFile);
data = readtable(dataFile);

% 按PSNR降序排序
data = sortrows(data, 'Mean_PSNR', 'descend');

% 显示数据
disp('读取的数据:');
disp(data);

%% 3. 创建PSNR对比图（单独图形）
fprintf('\n=== 创建PSNR对比图 ===\n');
figPSNR = figure('Position', [100, 100, 800, 600], 'Color', 'white', 'Name', 'PSNR Comparison');

barData = data.Mean_PSNR;
hBar = bar(barData, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.7);

% 设置X轴标签
set(gca, 'XTick', 1:length(data.Config), ...
    'XTickLabel', data.Config, ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'XTickLabelRotation', 20, ...
    'LineWidth', 1.5, ...
    'Box', 'on');

ylabel('PSNR (dB)', 'FontSize', 18, 'FontWeight', 'bold');
title('Average PSNR Comparison', 'FontSize', 20, 'FontWeight', 'bold');
grid on;

% 设置y轴范围（基于数据动态调整）
psnrMin = min(barData);
psnrMax = max(barData);
psnrRange = psnrMax - psnrMin;
ylim([psnrMin - 0.1*psnrRange, psnrMax + 0.15*psnrRange]);

% 在柱子上添加数值标签
for i = 1:length(barData)
    text(i, barData(i), ...
        sprintf('%.2f', barData(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'Color', 'k', ...
        'Margin', 2);
end

% 添加网格线
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);

% 添加副标题
% annotation('textbox', [0.15, 0.02, 0.7, 0.05], ...
%     'String', sprintf('Generated on: %s', currentDateTime), ...
%     'FontSize', 10, 'FontWeight', 'normal', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'middle', ...
%     'EdgeColor', 'none', ...
%     'BackgroundColor', 'none');

drawnow;
pause(0.5);

%% 4. 保存PSNR图
psnrBaseName = sprintf('PSNR_comparison_%s', currentDateTime);

% 保存为FIG格式
psnrFigFile = fullfile(outputDir, [psnrBaseName, '.fig']);
saveas(figPSNR, psnrFigFile);
fprintf('PSNR FIG文件已保存: %s\n', psnrFigFile);

% 保存为TIFF格式
psnrTiffFile = fullfile(outputDir, [psnrBaseName, '.tiff']);
print(figPSNR, psnrTiffFile, '-dtiff', sprintf('-r%d', dpi));
fprintf('PSNR TIFF文件已保存: %s\n', psnrTiffFile);

% 保存为EPS格式
psnrEpsFile = fullfile(outputDir, [psnrBaseName, '.eps']);
print(figPSNR, psnrEpsFile, '-depsc', '-r1200', '-painters');
fprintf('PSNR EPS文件已保存: %s\n', psnrEpsFile);

% 保存为PDF格式
psnrPdfFile = fullfile(outputDir, [psnrBaseName, '.pdf']);
print(figPSNR, psnrPdfFile, '-dpdf', '-r1200', '-bestfit', '-painters');
fprintf('PSNR PDF文件已保存: %s\n', psnrPdfFile);

%% 5. 创建SSIM对比图（单独图形）
fprintf('\n=== 创建SSIM对比图 ===\n');
figSSIM = figure('Position', [200, 100, 800, 600], 'Color', 'white', 'Name', 'SSIM Comparison');

barDataSSIM = data.Mean_SSIM;
hBarSSIM = bar(barDataSSIM, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.7);

% 设置X轴标签
set(gca, 'XTick', 1:length(data.Config), ...
    'XTickLabel', data.Config, ...
    'FontSize', 14, 'FontWeight', 'bold', ...
    'XTickLabelRotation', 20, ...
    'LineWidth', 1.5, ...
    'Box', 'on');

ylabel('SSIM', 'FontSize', 18, 'FontWeight', 'bold');
title('Average SSIM Comparison', 'FontSize', 20, 'FontWeight', 'bold');
grid on;

% 设置y轴范围（基于数据动态调整）
ssimMin = min(barDataSSIM);
ssimMax = max(barDataSSIM);
ssimRange = ssimMax - ssimMin;
ylim([ssimMin - 0.1*ssimRange, ssimMax + 0.15*ssimRange]);

% 在柱子上添加数值标签
for i = 1:length(barDataSSIM)
    text(i, barDataSSIM(i), ...
        sprintf('%.4f', barDataSSIM(i)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 14, 'FontWeight', 'bold', ...
        'Color', 'k', ...
        'Margin', 2);
end

% 添加网格线
set(gca, 'GridLineStyle', '--', 'GridAlpha', 0.3);

% 添加副标题
% annotation('textbox', [0.15, 0.02, 0.7, 0.05], ...
%     'String', sprintf('Generated on: %s', currentDateTime), ...
%     'FontSize', 10, 'FontWeight', 'normal', ...
%     'HorizontalAlignment', 'center', ...
%     'VerticalAlignment', 'middle', ...
%     'EdgeColor', 'none', ...
%     'BackgroundColor', 'none');

drawnow;
pause(0.5);

%% 6. 保存SSIM图
ssimBaseName = sprintf('SSIM_comparison_%s', currentDateTime);

% 保存为FIG格式
ssimFigFile = fullfile(outputDir, [ssimBaseName, '.fig']);
saveas(figSSIM, ssimFigFile);
fprintf('SSIM FIG文件已保存: %s\n', ssimFigFile);

% 保存为TIFF格式
ssimTiffFile = fullfile(outputDir, [ssimBaseName, '.tiff']);
print(figSSIM, ssimTiffFile, '-dtiff', sprintf('-r%d', dpi));
fprintf('SSIM TIFF文件已保存: %s\n', ssimTiffFile);

% 保存为EPS格式
ssimEpsFile = fullfile(outputDir, [ssimBaseName, '.eps']);
print(figSSIM, ssimEpsFile, '-depsc', '-r1200', '-painters');
fprintf('SSIM EPS文件已保存: %s\n', ssimEpsFile);

% 保存为PDF格式
ssimPdfFile = fullfile(outputDir, [ssimBaseName, '.pdf']);
print(figSSIM, ssimPdfFile, '-dpdf', '-r1200', '-bestfit', '-painters');
fprintf('SSIM PDF文件已保存: %s\n', ssimPdfFile);

%% 7. 可选：创建组合图（但不保存，只用于显示）
fprintf('\n=== 创建组合图（仅显示） ===\n');
figCombined = figure('Position', [300, 100, 1400, 600], 'Color', 'white', 'Name', 'Combined PSNR and SSIM Comparison');

% 左侧：PSNR
subplot(1, 2, 1);
bar(data.Mean_PSNR, 'FaceColor', [0.2, 0.4, 0.8], 'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.7);
set(gca, 'XTick', 1:length(data.Config), 'XTickLabel', data.Config, ...
    'FontSize', 12, 'FontWeight', 'bold', 'XTickLabelRotation', 15, 'LineWidth', 1.5);
ylabel('PSNR (dB)', 'FontSize', 16, 'FontWeight', 'bold');
title('Average PSNR', 'FontSize', 18, 'FontWeight', 'bold');
grid on;

% 右侧：SSIM
subplot(1, 2, 2);
bar(data.Mean_SSIM, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'k', 'LineWidth', 1.5, 'BarWidth', 0.7);
set(gca, 'XTick', 1:length(data.Config), 'XTickLabel', data.Config, ...
    'FontSize', 12, 'FontWeight', 'bold', 'XTickLabelRotation', 15, 'LineWidth', 1.5);
ylabel('SSIM', 'FontSize', 16, 'FontWeight', 'bold');
title('Average SSIM', 'FontSize', 18, 'FontWeight', 'bold');
grid on;

% 添加总标题
annotation('textbox', [0.3, 0.96, 0.4, 0.04], ...
    'String', 'IPSO-BPR vs DnCNN Performance Comparison', ...
    'FontSize', 20, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'BackgroundColor', 'none');

annotation('textbox', [0.3, 0.02, 0.4, 0.04], ...
    'String', sprintf('Generated on: %s', currentDateTime), ...
    'FontSize', 10, 'FontWeight', 'normal', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', ...
    'EdgeColor', 'none', ...
    'BackgroundColor', 'none');

%% 8. 生成数据汇总表并保存
summaryTable = data(:, {'Config', 'Mean_PSNR', 'Mean_SSIM'});
summaryFileName = sprintf('performance_summary_%s.xlsx', currentDateTime);
summaryFile = fullfile(outputDir, summaryFileName);
writetable(summaryTable, summaryFile);
fprintf('\n数据汇总表已保存: %s\n', summaryFile);

%% 9. 显示最佳模型
[bestPSNR, bestIdx] = max(data.Mean_PSNR);
[bestSSIM, bestSSIMIdx] = max(data.Mean_SSIM);

fprintf('\n=== 性能分析 ===\n');
fprintf('最佳PSNR模型: %s (%.2f dB)\n', data.Config{bestIdx}, bestPSNR);
fprintf('最佳SSIM模型: %s (%.4f)\n', data.Config{bestSSIMIdx}, bestSSIM);
fprintf('IPSO-BPR相对于StandardCNN的PSNR提升: %.2f dB\n', ...
    data.Mean_PSNR(strcmp(data.Config, 'IPSOBPR')) - data.Mean_PSNR(strcmp(data.Config, 'StandardCNN')));
fprintf('IPSO-BPR相对于StandardCNN的SSIM提升: %.4f\n', ...
    data.Mean_SSIM(strcmp(data.Config, 'IPSOBPR')) - data.Mean_SSIM(strcmp(data.Config, 'StandardCNN')));

%% 10. 显示保存的文件信息
fprintf('\n=== 文件保存完成 ===\n');
fprintf('输出目录: %s\n', outputDir);
fprintf('\nPSNR相关文件:\n');
fprintf('  1. %s\n', psnrFigFile);
fprintf('  2. %s\n', psnrTiffFile);
fprintf('  3. %s\n', psnrEpsFile);
fprintf('  4. %s\n', psnrPdfFile);

fprintf('\nSSIM相关文件:\n');
fprintf('  1. %s\n', ssimFigFile);
fprintf('  2. %s\n', ssimTiffFile);
fprintf('  3. %s\n', ssimEpsFile);
fprintf('  4. %s\n', ssimPdfFile);

fprintf('\n数据文件:\n');
fprintf('  1. %s\n', summaryFile);

fprintf('\n=== 所有文件已保存，带有时间戳: %s ===\n', currentDateTime);

%% 11. 提示用户
fprintf('\n=== 提示 ===\n');
fprintf('1. PSNR和SSIM图已分别保存为独立文件\n');
fprintf('2. 所有文件已添加时间戳，避免覆盖\n');
fprintf('3. 包含4种格式: .fig, .tiff, .eps, .pdf\n');
fprintf('4. 组合图仅用于屏幕显示，未保存\n');
fprintf('5. 如需保存组合图，请手动使用图形窗口的保存功能\n');