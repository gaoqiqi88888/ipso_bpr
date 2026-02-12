% 打开.fig文件
figFile = 'D:\matlab dev\ipso_bpr_v3\comparison_results\convergence_0801_tif.fig';
hFig = openfig(figFile, 'new', 'invisible');  % 'invisible'防止图形窗口弹出

% 或者使用openfig的可见模式
% hFig = openfig(figFile);

% 获取当前图形句柄（如果openfig未返回）
if ~exist('hFig', 'var')
    hFig = gcf;
end

% 确保图形窗口激活
figure(hFig);

% 定义输出路径（与原文件同目录）
[filepath, name, ~] = fileparts(figFile);
outputBase = fullfile(filepath, name);

%% 1. 保存为EPS格式
% 使用print函数保存为EPS
print(hFig, [outputBase '.eps'], '-depsc2', '-r1200');  % 彩色EPS，1200dpi
% 或者使用saveas（分辨率较低）
% saveas(hFig, [outputBase '.eps'], 'epsc');

%% 2. 保存为PDF格式
% 方法1：使用print（推荐，矢量格式）
print(hFig, [outputBase '.pdf'], '-dpdf', '-r1200');  % 1200dpi

% 方法2：使用exportgraphics（MATLAB R2020a+）
% exportgraphics(hFig, [outputBase '.pdf'], 'ContentType', 'vector', 'Resolution', 1200);

%% 3. 保存为1200dpi的TIFF格式
% 方法1：使用print
% print(hFig, [outputBase '_1200dpi.tif'], '-dtiff', '-r1200');

% 方法2：使用exportgraphics（MATLAB R2020a+，推荐）
exportgraphics(hFig, [outputBase '_1200dpi.tif'], 'Resolution', 1200);

% 方法3：使用saveas（分辨率可能受限）
% saveas(hFig, [outputBase '.tif'], 'tiff');

%% 关闭图形（可选）
close(hFig);

disp('所有格式保存完成！');
disp(['EPS: ' outputBase '.eps']);
disp(['PDF: ' outputBase '.pdf']);
disp(['TIFF (1200dpi): ' outputBase '_1200dpi.tif']);