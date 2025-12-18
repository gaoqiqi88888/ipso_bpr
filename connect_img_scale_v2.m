% % 这是一个用于自动排版和可视化IPSO（改进粒子群优化）算法图像处理结果的MATLAB工具。该脚本能够智能地整理、处理和组合多个实验图像，生成专业美观的对比图。
% % 
% % ### 主要功能：
% % 
% % 1. **批量图像处理**：自动读取指定目录下的TIFF图像文件
% % 2. **智能分组**：按实验编号和图像类型自动分组排序
% % 3. **尺寸统一化**：将所有图像调整为统一尺寸（默认88×88像素）
% % 4. **局部放大效果**：为每幅图像添加局部放大展示，突出细节对比
% % 5. **自定义布局**：支持灵活配置图像类型显示顺序
% % 6. **自动化标签**：从文件名提取日期代码作为行标签，自动添加列标签
% % 7. **高质量输出**：支持1200 DPI高分辨率TIFF格式输出
% % 8. **时间戳管理**：输出文件自动添加时间戳，避免文件覆盖
% % 
% % ### 技术特点：
% % 
% % - 使用中心裁剪或调整大小算法处理不同尺寸的原始图像
% % - 为ORG图像显示选择框和放大区域，其他图像只显示放大区域
% % - 添加细白色边框增强视觉效果
% % - 支持灰度图和RGB图混合处理
% % - 提供主图和详细子图两种输出格式

% 自动排版IPSO结果图 - 自定义顺序带局部放大效果（带日期时间标记）
clear; clc; close all;

%% 1. 设置路径和参数
imageDir = 'ipso\connect'; % 图像目录
outputDir = 'ipso'; % 输出目录
targetSize = [88, 88]; % 统一的目标尺寸 [高度, 宽度]

% 生成带日期时间的输出文件名
currentTime = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
outputName = sprintf('combined_IPSO_results_%s.tif', string(currentTime));

% 自定义图像类型顺序（可轻松修改）
% imageTypes = {'ORG', 'BLU', 'BPR', 'PSOBPR', 'IPSOBPR'};
imageTypes = {'ORG', 'BLU', 'WFR', 'CLSR', 'BPR', 'PSOBPR', 'IPSOBPR'};
% imageTypes = {'ORG', 'BLU', 'WFR', 'CLSR', 'BPR', 'PSOBPR', 'IPSOBPR', 'CNNR'};
numTypes = length(imageTypes);

% 局部放大参数
zoomFactor = 2.0;        % 放大倍数（不显示在图上）
zoomBoxRatio = 0.2;      % 放大区域占原图的比例
borderWidth = 1;         % 图像间白色边框宽度（更细）

% 创建输出目录（如果不存在）
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% 获取目录下所有tif文件
imageFiles = dir(fullfile(imageDir, '*.tif'));
if isempty(imageFiles)
    error('在目录 %s 中未找到tif文件', imageDir);
end

% 提取文件名
filenames = {imageFiles.name};
numImages = length(filenames);

fprintf('Found %d image files:\n', numImages);

%% 2. 按实验编号分组（1-10号实验）并提取日期代码
% 提取实验编号（文件名开头的数字）和日期代码
expNumbers = zeros(1, numImages);
dateCodes = cell(1, numImages); % 存储日期代码

for i = 1:numImages
    filename = filenames{i};
    
    % 提取实验编号（第一个数字）
    numStr = regexp(filename, '^(\d+)', 'match', 'once');
    if ~isempty(numStr)
        expNumbers(i) = str2double(numStr);
    else
        expNumbers(i) = inf;
    end
    
    % 提取日期代码（假设日期代码是4位数字，位于下划线之间）
    % 例如：1_ORG_0801.tif -> 提取0801
    datePattern = '_\d{4}\.';
    dateMatch = regexp(filename, datePattern, 'match');
    if ~isempty(dateMatch)
        % 提取4位数字，去掉下划线和点号
        dateStr = regexp(dateMatch{1}, '\d{4}', 'match', 'once');
        dateCodes{i} = dateStr;
    else
        % 如果没有找到日期代码，尝试其他模式
        % 查找文件名中的所有4位数字组合
        allNumbers = regexp(filename, '\d{4}', 'match');
        if ~isempty(allNumbers)
            % 取最后一个4位数字作为日期代码
            dateCodes{i} = allNumbers{end};
        else
            dateCodes{i} = '0000';
        end
    end
end

% 按实验编号排序
[sortedNumbers, sortIdx] = sort(expNumbers);
sortedFiles = filenames(sortIdx);
sortedDateCodes = dateCodes(sortIdx);

%% 3. 按实验编号分组
expGroups = unique(sortedNumbers);
numExps = length(expGroups);

fprintf('\nExperiment grouping:\n');
for i = 1:numExps
    expNum = expGroups(i);
    expIndices = find(sortedNumbers == expNum);
    fprintf('Exp %d: ', expNum);
    for j = 1:length(expIndices)
        fprintf('%s ', sortedFiles{expIndices(j)});
    end
    fprintf('\n');
end

%% 4. 统一图像尺寸并计算排版布局
% 读取第一张图像获取类型
firstImage = imread(fullfile(imageDir, sortedFiles{1}));
[~, ~, channels] = size(firstImage);

% 使用统一的目标尺寸
height = targetSize(1);
width = targetSize(2);

% 计算带边框的图像尺寸
imageWithBorderHeight = height + 2 * borderWidth;
imageWithBorderWidth = width + 2 * borderWidth;

% 边距设置
leftMargin = 80;   % 左侧边距用于标签
topMargin = 50;    % 顶部边距用于标签（减小顶部边距，因为不显示主标题了）
rightMargin = 20;  % 右侧边距
bottomMargin = 30; % 底部边距

% 计算大图尺寸（包含边距和边框）
totalWidth = imageWithBorderWidth * numTypes + leftMargin + rightMargin;
totalHeight = imageWithBorderHeight * numExps + topMargin + bottomMargin;

% 根据图像类型创建空白大图
if channels == 1
    % 灰度图
    combinedImage = uint8(255 * ones(totalHeight, totalWidth));
else
    % RGB图
    combinedImage = uint8(255 * ones(totalHeight, totalWidth, 3));
end

%% 5. 处理所有图像
fprintf('\nProcessing images with custom order and zoom effects...\n');

% 预加载所有图像并统一尺寸
allImages = cell(1, numImages);

% 存储每个实验组的放大区域坐标
zoomRegions = cell(numExps, 1);

for i = 1:numImages
    filename = sortedFiles{i};
    img = imread(fullfile(imageDir, filename));
    [h, w, c] = size(img);
    
    % 统一图像尺寸
    if h ~= targetSize(1) || w ~= targetSize(2)
        fprintf('Resizing: %s (%dx%d -> %dx%d)\n', filename, h, w, targetSize(1), targetSize(2));
        
        % 方法1: 中心裁剪
        if h > targetSize(1) || w > targetSize(2)
            % 从中心裁剪
            startH = floor((h - targetSize(1)) / 2) + 1;
            startW = floor((w - targetSize(2)) / 2) + 1;
            endH = startH + targetSize(1) - 1;
            endW = startW + targetSize(2) - 1;
            img = img(startH:endH, startW:endW, :);
        else
            % 方法2: 调整大小（如果原图较小）
            img = imresize(img, targetSize);
        end
    end
    
    % 确保图像是目标尺寸
    [newH, newW, ~] = size(img);
    if newH ~= targetSize(1) || newW ~= targetSize(2)
        img = imresize(img, targetSize);
    end
    
    allImages{i} = img;
end

fprintf('\nAll images processed\n');

%% 6. 按自定义顺序排列图像并添加放大效果
fprintf('\nArranging images with custom order and adding zoom effects...\n');

% 首先处理所有实验组，确定每个实验组的放大区域
for expIdx = 1:numExps
    expNum = expGroups(expIdx);
    expIndices = find(sortedNumbers == expNum);
    expFiles = sortedFiles(expIndices);
    
    % 找到该实验组的ORG图像
    orgIndex = 0;
    for i = 1:length(expIndices)
        if contains(expFiles{i}, 'ORG')
            orgIndex = expIndices(i);
            break;
        end
    end
    
    if orgIndex > 0
        % 从ORG图像计算放大区域
        orgImage = allImages{orgIndex};
        [h, w, ~] = size(orgImage);
        
        % 计算放大框的尺寸
        boxSize = round(min(h, w) * zoomBoxRatio);
        
        % 选择放大区域 - 左上角四分之一图的中心右下角位置
        quarterH = floor(h / 2);
        quarterW = floor(w / 2);
        
        % 左上角四分之一区域的右下角位置
        centerRow = floor(quarterH * 0.75);  % 中心偏右下
        centerCol = floor(quarterW * 0.75);  % 中心偏右下
        
        % 计算放大框的范围
        halfBox = floor(boxSize / 2);
        rowStart = max(1, centerRow - halfBox);
        rowEnd = min(h, centerRow + halfBox);
        colStart = max(1, centerCol - halfBox);
        colEnd = min(w, centerCol + halfBox);
        
        % 存储该实验组的放大区域坐标
        zoomRegions{expIdx} = [rowStart, rowEnd, colStart, colEnd];
        fprintf('Experiment %d zoom region: [%d:%d, %d:%d]\n', expNum, rowStart, rowEnd, colStart, colEnd);
    else
        warning('Experiment %d: No ORG image found, using default zoom region', expNum);
        % 使用默认区域
        [h, w, ~] = size(allImages{expIndices(1)});
        boxSize = round(min(h, w) * zoomBoxRatio);
        rowStart = floor(h/4);
        rowEnd = rowStart + boxSize;
        colStart = floor(w/4);
        colEnd = colStart + boxSize;
        zoomRegions{expIdx} = [rowStart, rowEnd, colStart, colEnd];
    end
end

% 然后按自定义顺序排列图像并应用放大效果
for expIdx = 1:numExps
    expNum = expGroups(expIdx);
    expIndices = find(sortedNumbers == expNum);
    expFiles = sortedFiles(expIndices);
    
    % 按自定义顺序获取图像索引
    sortedExpIndices = zeros(1, numTypes);
    
    for typeIdx = 1:numTypes
        type = imageTypes{typeIdx};
        for fileIdx = 1:length(expFiles)
            if contains(expFiles{fileIdx}, type)
                sortedExpIndices(typeIdx) = expIndices(fileIdx);
                break;
            end
        end
    end
    
    % 获取该实验组的放大区域
    zoomRegion = zoomRegions{expIdx};
    
    % 将排序后的图像放入大图（带白色边框）
    for typeIdx = 1:numTypes
        if sortedExpIndices(typeIdx) > 0
            fileIdx = sortedExpIndices(typeIdx);
            filename = sortedFiles{fileIdx};
            img = allImages{fileIdx};
            
            % 为图像添加放大效果
            if contains(filename, 'ORG')
                % ORG图像：显示选择框和正确的放大内容
                img = addZoomEffectWithBox(img, zoomRegion, zoomFactor);
            else
                % 其他图像：不显示选择框，但显示正确的放大内容
                img = addZoomEffectWithoutBox(img, zoomRegion, zoomFactor);
            end
            
            % 计算在大图中的位置（考虑边距和边框）
            rowStart = (expIdx - 1) * imageWithBorderHeight + topMargin + borderWidth + 1;
            rowEnd = rowStart + height - 1;
            colStart = (typeIdx - 1) * imageWithBorderWidth + leftMargin + borderWidth + 1;
            colEnd = colStart + width - 1;
            
            % 将图像放入大图
            if size(combinedImage, 3) == 1 && size(img, 3) == 1
                combinedImage(rowStart:rowEnd, colStart:colEnd) = img;
            elseif size(combinedImage, 3) == 3 && size(img, 3) == 1
                % 灰度图放入RGB大图
                combinedImage(rowStart:rowEnd, colStart:colEnd, 1) = img;
                combinedImage(rowStart:rowEnd, colStart:colEnd, 2) = img;
                combinedImage(rowStart:rowEnd, colStart:colEnd, 3) = img;
            else
                combinedImage(rowStart:rowEnd, colStart:colEnd, :) = img;
            end
            
            fprintf('Exp %d - %s: %s\n', expNum, imageTypes{typeIdx}, filename);
        end
    end
end

%% 7. 添加标签（在边距区域，不会与图像重叠）
% 创建标签图像
labelImage = combinedImage;

% 不添加主标题（根据要求3）

% 添加左侧行标签（使用提取的日期代码）
for expIdx = 1:numExps
    expNum = expGroups(expIdx);
    expIndices = find(sortedNumbers == expNum);
    
    % 从该实验的第一个文件中提取日期代码
    if ~isempty(expIndices)
        firstFileIdx = expIndices(1);
        dateCode = sortedDateCodes{firstFileIdx};
        
        % 如果没有提取到日期代码，使用实验编号作为后备
        if isempty(dateCode) || strcmp(dateCode, '0000')
            dateCode = sprintf('Exp%d', expNum);
        end
        
        labelPosY = (expIdx - 0.5) * imageWithBorderHeight + topMargin;
        labelPosX = leftMargin / 3;
        
        % 在边距区域添加文本
        labelImage = insertText(labelImage, [labelPosX, labelPosY], dateCode, ...
            'FontSize', 16, 'TextColor', 'red', 'BoxColor', 'white', ...
            'BoxOpacity', 0.8, 'AnchorPoint', 'Center', 'Font', 'Arial Bold');
    end
end

% 添加上部列标签
for typeIdx = 1:numTypes
    labelPosX = (typeIdx - 0.5) * imageWithBorderWidth + leftMargin;
    labelPosY = topMargin / 2; % 调整位置，因为不显示主标题了
    
    labelImage = insertText(labelImage, [labelPosX, labelPosY], imageTypes{typeIdx}, ...
        'FontSize', 14, 'TextColor', 'blue', 'BoxColor', 'white', ...
        'BoxOpacity', 0.8, 'AnchorPoint', 'Center', 'Font', 'Arial Bold');
end

%% 8. 显示和保存结果
figure('Position', [100, 100, min(1400, totalWidth), min(900, totalHeight)], ...
       'Name', 'IPSO Algorithm Results - Custom Order', 'NumberTitle', 'off');
   
if size(labelImage, 3) == 1
    imshow(labelImage, 'Border', 'tight');
else
    imshow(labelImage, 'Border', 'tight');
end

% 保存大图到ipso目录（要求2：1200 DPI）
outputPath = fullfile(outputDir, outputName);

% 根据图像类型使用不同的保存方式
if size(labelImage, 3) == 1
    % 灰度图
    imwrite(labelImage, outputPath, 'Resolution', 1200);
else
    % RGB图
    imwrite(labelImage, outputPath, 'Resolution', 1200);
end

fprintf('\nCombined image with custom order saved: %s\n', outputPath);
fprintf('Image dimensions: %d x %d (including margins and borders)\n', totalWidth, totalHeight);
fprintf('Layout: %d figures x %d methods\n', numExps, numTypes);
fprintf('Image order: %s\n', strjoin(imageTypes, ', '));
fprintf('Resolution: 1200 DPI\n');

%% 9. 创建详细的子图显示（可选）
figure('Position', [200, 200, 1400, 900], ...
       'Name', 'IPSO Algorithm Detailed Results', 'NumberTitle', 'off');

for expIdx = 1:numExps
    expNum = expGroups(expIdx);
    expIndices = find(sortedNumbers == expNum);
    expFiles = sortedFiles(expIndices);
    
    % 按类型排序
    sortedExpIndices = zeros(1, numTypes);
    
    for typeIdx = 1:numTypes
        type = imageTypes{typeIdx};
        for fileIdx = 1:length(expFiles)
            if contains(expFiles{fileIdx}, type)
                sortedExpIndices(typeIdx) = expIndices(fileIdx);
                break;
            end
        end
    end
    
    % 显示子图
    for typeIdx = 1:numTypes
        if sortedExpIndices(typeIdx) > 0
            subplot(numExps, numTypes, (expIdx-1)*numTypes + typeIdx);
            img = allImages{sortedExpIndices(typeIdx)};
            imshow(img);
            
            if expIdx == 1
                title(imageTypes{typeIdx}, 'FontSize', 10, 'FontWeight', 'bold');
            end
            
            if typeIdx == 1
                % 使用提取的日期代码作为ylabel
                firstFileIdx = expIndices(1);
                dateCode = sortedDateCodes{firstFileIdx};
                if isempty(dateCode) || strcmp(dateCode, '0000')
                    dateCode = sprintf('Exp%d', expNum);
                end
                ylabel(dateCode, 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
    end
end

% 保存子图版本（1200 DPI）
subplotOutputPath = fullfile(outputDir, sprintf('IPSO_detailed_results_%s.tif', string(currentTime)));
print(gcf, subplotOutputPath, '-dtiff', '-r1200'); % 使用print命令指定1200 DPI
fprintf('Detailed subplot version saved: %s (1200 DPI)\n', subplotOutputPath);

%% 10. 带选择框的放大效果函数（用于ORG图像）
function outputImage = addZoomEffectWithBox(originalImage, zoomRegion, zoomFactor)
    % 获取图像尺寸
    [h, w, c] = size(originalImage);
    
    % 提取放大区域坐标
    rowStart = zoomRegion(1);
    rowEnd = zoomRegion(2);
    colStart = zoomRegion(3);
    colEnd = zoomRegion(4);
    boxSize = rowEnd - rowStart + 1;
    
    % 提取要放大的区域
    zoomRegionImage = originalImage(rowStart:rowEnd, colStart:colEnd, :);
    
    % 放大区域
    zoomedSize = round(boxSize * zoomFactor);
    zoomedRegion = imresize(zoomRegionImage, [zoomedSize, zoomedSize], 'nearest');
    
    % 创建输出图像（与原图相同尺寸）
    outputImage = originalImage;
    
    % 计算放大图在右下角的位置
    zoomRowStart = h - zoomedSize + 1;
    zoomRowEnd = h;
    zoomColStart = w - zoomedSize + 1;
    zoomColEnd = w;
    
    % 将放大图放在右下角
    if c == 1
        outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColEnd) = zoomedRegion;
    else
        outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColEnd, :) = zoomedRegion;
    end
    
    % 在原图上绘制白色放大框（左上角选择框）
    lineWidth = 1;
    
    % 左上角放大框（白色）
    outputImage(rowStart:rowStart+lineWidth-1, colStart:colEnd) = 255; % 上边
    outputImage(rowEnd-lineWidth+1:rowEnd, colStart:colEnd) = 255;     % 下边
    outputImage(rowStart:rowEnd, colStart:colStart+lineWidth-1) = 255; % 左边
    outputImage(rowStart:rowEnd, colEnd-lineWidth+1:colEnd) = 255;     % 右边
    
    % 右下角放大图边框（白色）
    outputImage(zoomRowStart:zoomRowStart+lineWidth-1, zoomColStart:zoomColEnd) = 255; % 上边
    outputImage(zoomRowEnd-lineWidth+1:zoomRowEnd, zoomColStart:zoomColEnd) = 255;     % 下边
    outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColStart+lineWidth-1) = 255; % 左边
    outputImage(zoomRowStart:zoomRowEnd, zoomColEnd-lineWidth+1:zoomColEnd) = 255;     % 右边
end

%% 11. 不带选择框的放大效果函数（用于非ORG图像）
function outputImage = addZoomEffectWithoutBox(originalImage, zoomRegion, zoomFactor)
    % 获取图像尺寸
    [h, w, c] = size(originalImage);
    
    % 提取放大区域坐标
    rowStart = zoomRegion(1);
    rowEnd = zoomRegion(2);
    colStart = zoomRegion(3);
    colEnd = zoomRegion(4);
    boxSize = rowEnd - rowStart + 1;
    
    % 提取要放大的区域
    zoomRegionImage = originalImage(rowStart:rowEnd, colStart:colEnd, :);
    
    % 放大区域
    zoomedSize = round(boxSize * zoomFactor);
    zoomedRegion = imresize(zoomRegionImage, [zoomedSize, zoomedSize], 'nearest');
    
    % 创建输出图像（与原图相同尺寸）
    outputImage = originalImage;
    
    % 计算放大图在右下角的位置
    zoomRowStart = h - zoomedSize + 1;
    zoomRowEnd = h;
    zoomColStart = w - zoomedSize + 1;
    zoomColEnd = w;
    
    % 将放大图放在右下角
    if c == 1
        outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColEnd) = zoomedRegion;
    else
        outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColEnd, :) = zoomedRegion;
    end
    
    % 只在右下角绘制白色放大框（不绘制左上角选择框）
    lineWidth = 1;
    
    % 右下角放大图边框（白色）
    outputImage(zoomRowStart:zoomRowStart+lineWidth-1, zoomColStart:zoomColEnd) = 255; % 上边
    outputImage(zoomRowEnd-lineWidth+1:zoomRowEnd, zoomColStart:zoomColEnd) = 255;     % 下边
    outputImage(zoomRowStart:zoomRowEnd, zoomColStart:zoomColStart+lineWidth-1) = 255; % 左边
    outputImage(zoomRowStart:zoomRowEnd, zoomColEnd-lineWidth+1:zoomColEnd) = 255;     % 右边
end

fprintf('\nImage combination with custom order completed!\n');
fprintf('Output files are saved with timestamp: %s\n', string(currentTime));