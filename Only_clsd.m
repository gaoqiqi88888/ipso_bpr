% 定义文件夹路径
blurFolder = 'C:\matlab dev\datasets\DIV2K_valid_BLUR_90_gray';
sharpFolder = 'C:\matlab dev\datasets\DIV2K_valid_HR_90_gray';
outputFolder = 'C:\matlab dev\datasets\DIV2K_valid_CLSD_90_gray';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);  % 创建输出文件夹
end

% 获取模糊图像文件列表
blurFiles = dir(fullfile(blurFolder, '*.tif'));
if isempty(blurFiles)
    error('未找到模糊图像文件！');
end

% 定义高斯模糊核并强制转为双精度
w = double(fspecial('gaussian', [9, 9], 1));
if numel(w) < 2
    error('高斯核尺寸错误，必须至少有2个元素！');
end

% 初始化指标存储变量
metrics = table('Size', [length(blurFiles), 4], ...
    'VariableNames', {'FileName', 'PSNR', 'MSE', 'SSIM'}, ...
    'VariableTypes', {'string', 'double', 'double', 'double'});

% 定义约束最小二乘滤波的参数
lambda = 0.01;  % 正则化参数
% 创建拉普拉斯算子并转为双精度
laplacian = double(fspecial('laplacian', 0));

% 批量处理图像
for i = 1:length(blurFiles)
    fileName = blurFiles(i).name;
    blurPath = fullfile(blurFolder, fileName);
    sharpPath = fullfile(sharpFolder, fileName);
    
    % 读取图像并转换为双精度
    blurImg = im2double(imread(blurPath));
    sharpImg = im2double(imread(sharpPath));
    
    % 提取中间88×88区域
    [h, w_img] = size(blurImg);
    if h < 88 || w_img < 88
        error(['图像 ', fileName, ' 尺寸小于88×88，无法提取区域！']);
    end
    startRow = floor((h - 88)/2) + 1;
    startCol = floor((w_img - 88)/2) + 1;
    blurPatch = blurImg(startRow:startRow+87, startCol:startCol+87);
    sharpPatch = sharpImg(startRow:startRow+87, startCol:startCol+87);
    
    % --------------------------
    % 约束最小二乘滤波复原（使用位置参数）
    % --------------------------
    % 校验参数类型
    fprintf('参数类型校验 - 图像：%s，模糊核：%s，约束算子：%s\n',...
        class(blurPatch), class(w), class(laplacian));
    
    % 使用正确的位置参数调用 deconvreg
    % 语法: J = deconvreg(I, PSF, NOISEPOWER, LRANGE, REGOP)
    restoredPatch = deconvreg(blurPatch, w, lambda, [], laplacian);
    
    % 限制像素值范围
    restoredPatch = max(min(restoredPatch, 1), 0);
    
    % 计算评价指标
    mseVal = immse(restoredPatch, sharpPatch);
    psnrVal = psnr(restoredPatch, sharpPatch);
    ssimVal = ssim(restoredPatch, sharpPatch);
    
    % 保存复原结果
    outputPath = fullfile(outputFolder, fileName);
    imwrite(restoredPatch, outputPath, 'tif');
    
    % 存储指标
    metrics.FileName(i) = fileName;
    metrics.PSNR(i) = psnrVal;
    metrics.MSE(i) = mseVal;
    metrics.SSIM(i) = ssimVal;
    
    % 控制台显示信息
    fprintf('-----------------------------------------------------\n');
    fprintf('处理进度：%d/%d\n', i, length(blurFiles));
    fprintf('图像名称：%s\n', fileName);
    fprintf('复原区域尺寸：%dx%d\n', size(restoredPatch,1), size(restoredPatch,2));
    fprintf('MSE：%.6f\n', mseVal);
    fprintf('PSNR：%.2f dB\n', psnrVal);
    fprintf('SSIM：%.4f\n', ssimVal);
end

% 保存指标到Excel
excelPath = fullfile(outputFolder, '约束最小二乘滤波复原图像指标.xlsx');
writetable(metrics, excelPath);
fprintf('\n-----------------------------------------------------\n');
fprintf('指标已保存至：%s\n', excelPath);

% 保存工作区变量
currentDateTime = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
varMatPath = fullfile(outputFolder, ['约束最小二乘滤波复原变量存档_', currentDateTime, '.mat']);
save(varMatPath);

fprintf('工作区所有变量已保存至：%s\n', varMatPath);
fprintf('所有处理完成！\n');