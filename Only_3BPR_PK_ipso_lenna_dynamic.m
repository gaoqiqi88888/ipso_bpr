%% 重要代码，PSO和IPSO单一图像的一次对比

% Paper_single_1000_artifical_dynamic.m（适配PSO/IPSO对比与artificial图分析，可直接复制）
%% 清空环境变量
clear
close all
clc

% 核心配置：单图（artificial.tif）对比分析，数据来源于PSO/IPSO优化结果
myt1 = clock;
%% 1. 基础参数与路径配置
% 1.1 数据加载路径（PSO/IPSO优化结果文件，替换原GA/IGA数据）
mat_name = 'ipso/best90_r100_Lenna.mat';  % 指定PSO/IPSO结果数据文件
load(mat_name);  % 加载数据：含PSO_bestchrom、PSO_bestfitness、IPSO_bestchrom、IPSO_bestfitness等

% 1.2 实验控制参数（按老论文单图对比逻辑）
maxit = 3;      % 图像还原对比次数（老论文常用1000次以统计稳定性）
inputnum = 9;      % 输入节点数（3×3滑动窗口，与原网络一致）
hiddennum = 9;     % 隐藏层节点数
outputnum = 1;     % 输出节点数（单像素预测）
picsize = [90, 90]; % 图像尺寸（统一为90×90，与优化阶段一致）

% 1.3 图像路径配置（仅针对artificial.tif，按老论文路径格式）
%all_sim_picname = ["artificial.tif"];  % 待分析图像（单图）
all_sim_picname = ["lenna.tif"];  % 待分析图像（单图）
% 构建图像完整路径（img2/为图像存储目录，与原代码路径逻辑一致）
%all_sim_dir_picname = strcat("img2/", all_sim_picname);
all_sim_dir_picname = strcat("img2/", all_sim_picname);
n = length(all_sim_dir_picname);  % 图像数量（此处n=1）

%% 2. PSO/IPSO最优参数统计（替换原GA/IGA逻辑）
% 2.1 统计PSO最优参数（适应度最小对应最优BPNN参数）
[PSO_best, PSO_best_idx] = min(PSO_bestfitness);  % PSO最优适应度与对应轮次
PSO_mean = mean(PSO_bestfitness);                 % PSO平均适应度
fprintf('PSO - 最优适应度: %.6f, 平均适应度: %.6f\n', PSO_best, PSO_mean);

% 2.2 统计IPSO最优参数（同理）
[IPSO_best, IPSO_best_idx] = min(IPSO_bestfitness);  % IPSO最优适应度与对应轮次
IPSO_mean = mean(IPSO_bestfitness);                 % IPSO平均适应度
fprintf('IPSO - 最优适应度: %.6f, 平均适应度: %.6f\n', IPSO_best, IPSO_mean);

% 2.3 绘制PSO/IPSO适应度曲线（按老论文图表风格，替换原GA/IGA曲线）
figure('Position', [100, 100, 600, 400]);
plot(PSO_trace_bestfitness(:, PSO_best_idx), 'b--', 'LineWidth', 1.5);  % PSO：蓝色虚线
hold on
plot(IPSO_trace_bestfitness(:, IPSO_best_idx), 'r-', 'LineWidth', 1.5);   % IPSO：红色实线
hold off
% 图表标注（符合老论文规范：英文标签、清晰图例）
title('Fitness Curve (PSO vs IPSO)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Iteration Number', 'FontName', 'Times New Roman', 'FontSize', 10);
ylabel('Fitness Value (BPNN Prediction Error)', 'FontName', 'Times New Roman', 'FontSize', 10);
legend('PSO', 'IPSO', 'Location', 'best', 'FontName', 'Times New Roman', 'FontSize', 9);
grid on;
% 保存曲线（老论文常用TIFF格式，600dpi高清）
f = getframe(gcf);
imwrite(f.cdata, ['ipso/PSO_IPSO_Fitness_Curve.tif'], 'tiff', 'Resolution', 600); 

%% 3. 初始化结果存储数组（替换原GA/IGA相关数组，新增SSIM存储）
% 3.1 评价指标存储（PSNR/MSE/SSIM，均为1000次对比的结果）
% 无优化BP（BPR）
BPR_psnr = zeros(1, maxit);  
BPR_mse = zeros(1, maxit);  
BPR_ssim = zeros(1, maxit);  
% PSO优化BP（PSOBPR）
PSOBPR_psnr = zeros(1, maxit);  
PSOBPR_mse = zeros(1, maxit);  
PSOBPR_ssim = zeros(1, maxit);  
% IPSO优化BP（IPSOBPR）
IPSOBPR_psnr = zeros(1, maxit);  
IPSOBPR_mse = zeros(1, maxit);  
IPSOBPR_ssim = zeros(1, maxit);  

% 3.2 复原图像存储（保存1000次对比中的最优复原图）
BPR_image_restored_noedge = zeros(picsize(1)-2, picsize(2)-2, maxit);  
PSOBPR_image_restored_noedge = zeros(picsize(1)-2, picsize(2)-2, maxit);  
IPSOBPR_image_restored_noedge = zeros(picsize(1)-2, picsize(2)-2, maxit);  

%% 4. 循环测试artificial.tif（单图对比，按老论文逐次复原逻辑）
for pic_index = 1:n
    sim_pic_dirname = all_sim_dir_picname(pic_index);  % 当前图像路径（artificial.tif）
    fprintf('\n********** 正在分析图像：%s **********\n', sim_pic_dirname);

    % 4.1 读取并预处理图像（调用原Read_Pic函数，确保退化逻辑与优化阶段一致）
    [P_Matrix, T_Matrix, image_resized, image_blurred] = Read_Pic_PSO(sim_pic_dirname, picsize);
    inputn = P_Matrix;  % BPNN输入数据（退化图像窗口）
    outputn = T_Matrix; % BPNN目标数据（清晰图像中心像素）

    % 4.2 初始化BPNN网络（与优化阶段结构完全一致）
    net = newff(inputn, outputn, hiddennum);

    % 4.3 1000次复原对比循环（核心：逐次测试三种算法）
    for index = 1:maxit
        fprintf('第%d次复原对比（共%d次）\n', index, maxit);

        % -------------------------- 1. 无优化BP（BPR，对比基准）--------------------------
        x = [];  % 无优化参数
        [BPR_psnr(1,index), BPR_mse(1,index), ~, BPR_ssim(1,index), ...
         image_resized, image_blurred, BPR_restored] = Sim_Pic_PSO(0, net, inputn, outputn, x, ...
         inputnum, hiddennum, outputnum, sim_pic_dirname, picsize);
        % 存储当前次复原图像
        BPR_image_restored_noedge(:,:,index) = BPR_restored;
        % 打印当前次结果（按老论文格式输出）
        fprintf('-------------------------------------------\n');
        fprintf('BPR  - 复原PSNR: %.2f dB, MSE: %.2f, SSIM: %.4f\n', ...
            BPR_psnr(1,index), BPR_mse(1,index), BPR_ssim(1,index));

        % -------------------------- 2. PSO优化BP（PSOBPR）--------------------------
        x = PSO_bestchrom(PSO_best_idx, :);  % 加载PSO最优参数（100轮中最优轮次）
        [PSOBPR_psnr(1,index), PSOBPR_mse(1,index), ~, PSOBPR_ssim(1,index), ...
         image_resized, image_blurred, PSO_restored] = Sim_Pic_PSO(1, net, inputn, outputn, x, ...
         inputnum, hiddennum, outputnum, sim_pic_dirname, picsize);
        % 存储当前次复原图像
        PSOBPR_image_restored_noedge(:,:,index) = PSO_restored;
        % 打印当前次结果
        fprintf('-------------------------------------------\n');
        fprintf('PSOBPR - 复原PSNR: %.2f dB, MSE: %.2f, SSIM: %.4f\n', ...
            PSOBPR_psnr(1,index), PSOBPR_mse(1,index), PSOBPR_ssim(1,index));

        % -------------------------- 3. IPSO优化BP（IPSOBPR）--------------------------
        x = IPSO_bestchrom(IPSO_best_idx, :);  % 加载IPSO最优参数（100轮中最优轮次）
        [IPSOBPR_psnr(1,index), IPSOBPR_mse(1,index), ~, IPSOBPR_ssim(1,index), ...
         image_resized, image_blurred, IPSO_restored] = Sim_Pic_PSO(1, net, inputn, outputn, x, ...
         inputnum, hiddennum, outputnum, sim_pic_dirname, picsize);
        % 存储当前次复原图像
        IPSOBPR_image_restored_noedge(:,:,index) = IPSO_restored;
        % 打印当前次结果
        fprintf('-------------------------------------------\n');
        fprintf('IPSOBPR - 复原PSNR: %.2f dB, MSE: %.2f, SSIM: %.4f\n', ...
            IPSOBPR_psnr(1,index), IPSOBPR_mse(1,index), IPSOBPR_ssim(1,index));
    end

    %% 5. 统计结果计算（按老论文逻辑：最优值、平均值，用于后续图表）
    fprintf('\n-------------------------- %s 统计结果 --------------------------\n', all_sim_picname{pic_index});
    % 5.1 PSNR统计（PSNR越大越好）
    fprintf('【PSNR统计结果】\n');
    % BPR
    [BPR_psnr_best, BPR_psnr_idx] = max(BPR_psnr);
    BPR_psnr_mean = mean(BPR_psnr);
    fprintf('BPR  - 最优PSNR: %.2f dB (第%d次), 平均PSNR: %.2f dB\n', ...
        BPR_psnr_best, BPR_psnr_idx, BPR_psnr_mean);
    % PSOBPR
    [PSOBPR_psnr_best, PSOBPR_psnr_idx] = max(PSOBPR_psnr);
    PSOBPR_psnr_mean = mean(PSOBPR_psnr);
    fprintf('PSOBPR - 最优PSNR: %.2f dB (第%d次), 平均PSNR: %.2f dB\n', ...
        PSOBPR_psnr_best, PSOBPR_psnr_idx, PSOBPR_psnr_mean);
    % IPSOBPR
    [IPSOBPR_psnr_best, IPSOBPR_psnr_idx] = max(IPSOBPR_psnr);
    IPSOBPR_psnr_mean = mean(IPSOBPR_psnr);
    fprintf('IPSOBPR - 最优PSNR: %.2f dB (第%d次), 平均PSNR: %.2f dB\n', ...
        IPSOBPR_psnr_best, IPSOBPR_psnr_idx, IPSOBPR_psnr_mean);

    % 5.2 MSE统计（MSE越小越好）
    fprintf('\n【MSE统计结果】\n');
    % BPR
    [BPR_mse_best, BPR_mse_idx] = min(BPR_mse);
    BPR_mse_mean = mean(BPR_mse);
    fprintf('BPR  - 最优MSE: %.2f (第%d次), 平均MSE: %.2f\n', ...
        BPR_mse_best, BPR_mse_idx, BPR_mse_mean);
    % PSOBPR
    [PSOBPR_mse_best, PSOBPR_mse_idx] = min(PSOBPR_mse);
    PSOBPR_mse_mean = mean(PSOBPR_mse);
    fprintf('PSOBPR - 最优MSE: %.2f (第%d次), 平均MSE: %.2f\n', ...
        PSOBPR_mse_best, PSOBPR_mse_idx, PSOBPR_mse_mean);
    % IPSOBPR
    [IPSOBPR_mse_best, IPSOBPR_mse_idx] = min(IPSOBPR_mse);
    IPSOBPR_mse_mean = mean(IPSOBPR_mse);
    fprintf('IPSOBPR - 最优MSE: %.2f (第%d次), 平均MSE: %.2f\n', ...
        IPSOBPR_mse_best, IPSOBPR_mse_idx, IPSOBPR_mse_mean);

    % 5.3 SSIM统计（SSIM越接近1越好）
    fprintf('\n【SSIM统计结果】\n');
    % BPR
    [BPR_ssim_best, BPR_ssim_idx] = max(BPR_ssim);
    BPR_ssim_mean = mean(BPR_ssim);
    fprintf('BPR  - 最优SSIM: %.4f (第%d次), 平均SSIM: %.4f\n', ...
        BPR_ssim_best, BPR_ssim_idx, BPR_ssim_mean);
    % PSOBPR
    [PSOBPR_ssim_best, PSOBPR_ssim_idx] = max(PSOBPR_ssim);
    PSOBPR_ssim_mean = mean(PSOBPR_ssim);
    fprintf('PSOBPR - 最优SSIM: %.4f (第%d次), 平均SSIM: %.4f\n', ...
        PSOBPR_ssim_best, PSOBPR_ssim_idx, PSOBPR_ssim_mean);
    % IPSOBPR
    [IPSOBPR_ssim_best, IPSOBPR_ssim_idx] = max(IPSOBPR_ssim);
    IPSOBPR_ssim_mean = mean(IPSOBPR_ssim);
    fprintf('IPSOBPR - 最优SSIM: %.4f (第%d次), 平均SSIM: %.4f\n', ...
        IPSOBPR_ssim_best, IPSOBPR_ssim_idx, IPSOBPR_ssim_mean);

    %% 6. 提取最优复原图像（用于老论文风格的图像展示）
    % 6.1 按PSNR最优提取（老论文常用PSNR作为最优图像筛选指标）
    BPR_best_restored = BPR_image_restored_noedge(:,:,BPR_psnr_idx);
    PSOBPR_best_restored = PSOBPR_image_restored_noedge(:,:,PSOBPR_psnr_idx);
    IPSOBPR_best_restored = IPSOBPR_image_restored_noedge(:,:,IPSOBPR_psnr_idx);

    % 6.2 保存图像（按老论文命名格式：结果目录+图像类型+算法，TIFF格式600dpi）
    % 原始清晰图
    imwrite(image_resized, strcat("ipso/", num2str(pic_index), "_ORG_artificial.tif"), ...
        'tiff', 'Resolution', 600);
    % 退化模糊图
    imwrite(image_blurred, strcat("ipso/", num2str(pic_index), "_BLU_artificial.tif"), ...
        'tiff', 'Resolution', 600);
    % 各算法最优复原图
    imwrite(BPR_best_restored, strcat("ipso/", num2str(pic_index), "_BPR_artificial.tif"), ...
        'tiff', 'Resolution', 600);
    imwrite(PSOBPR_best_restored, strcat("ipso/", num2str(pic_index), "_PSOBPR_artificial.tif"), ...
        'tiff', 'Resolution', 600);
    imwrite(IPSOBPR_best_restored, strcat("ipso/", num2str(pic_index), "_IPSOBPR_artificial.tif"), ...
        'tiff', 'Resolution', 600);

    %% 7. 绘制老论文风格对比图表（PSNR/MSE/SSIM柱状图，带数值标注）
    % 7.1 数据整理（平均+最优值，用于柱状图）
    % PSNR数据
    psnr_data = [BPR_psnr_mean, PSOBPR_psnr_mean, IPSOBPR_psnr_mean;
                 BPR_psnr_best, PSOBPR_psnr_best, IPSOBPR_psnr_best];
    % MSE数据
    mse_data = [BPR_mse_mean, PSOBPR_mse_mean, IPSOBPR_mse_mean;
                BPR_mse_best, PSOBPR_mse_best, IPSOBPR_mse_best];
    % SSIM数据
    ssim_data = [BPR_ssim_mean, PSOBPR_ssim_mean, IPSOBPR_ssim_mean;
                 BPR_ssim_best, PSOBPR_ssim_best, IPSOBPR_ssim_best];
    % 算法标签（老论文常用英文标签）
    algo_labels = {'BPR', 'PSOBPR', 'IPSOBPR'};


end

%% 8. 保存实验数据（按老论文格式，便于后续补充分析）
save_data_name = 'ipso/Paper_single_1000_artificial_PSO_IPSO.mat';
save(save_data_name, ...
    'BPR_psnr', 'BPR_mse', 'BPR_ssim', ...
    'PSOBPR_psnr', 'PSOBPR_mse', 'PSOBPR_ssim', ...
    'IPSOBPR_psnr', 'IPSOBPR_mse', 'IPSOBPR_ssim', ...
    'BPR_best_restored', 'PSOBPR_best_restored', 'IPSOBPR_best_restored');

% 9. 总运行时间统计（老论文实验报告必备）
myt2 = clock;
total_time = etime(myt2, myt1) / 60;  % 转换为分钟
fprintf('\n-------------------------- 实验完成 --------------------------\n');
fprintf('总运行时间: %.2f 分钟\n', total_time);
fprintf('所有结果（图像+数据）已保存至 ipso/ 目录\n');



% 算法标签-------------------------------------------------------------
% %% 7. 绘制老论文风格对比图表（PSNR/MSE/SSIM柱状图，带数值标注）
% % 7.1 数据整理（平均+最优值，用于柱状图）
% % 算法标签（老论文常用英文标签）
% algo_labels = {'BPR', 'PSOBPR', 'IPSOBPR'};
% 
% % 7.2 PSNR对比图（调整Y轴范围放大差异）
% figure('Position', [200, 200, 600, 400]);
% X = 1:3;
% h = bar(X, psnr_data, 0.8);  % 柱状图宽度0.8，避免重叠
% 
% % 设置更窄的Y轴范围来放大PSNR差异
% psnr_min = min(psnr_data(:)) - 0.05;
% psnr_max = max(psnr_data(:)) + 0.1;
% ylim([psnr_min, psnr_max]);
% 
% % 图表格式（匹配老论文：Times New Roman字体、清晰刻度）
% set(gca, 'XTickLabel', algo_labels, 'FontName', 'Times New Roman', 'FontSize', 11);
% ylabel('PSNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 12);
% xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
% title('PSNR Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
% % 图例（平均/最优，水平排列在顶部，老论文常用位置）
% hl = legend({'Mean PSNR', 'Best PSNR'}, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
% set(hl, 'Orientation', 'horizontal');
% % 数值标注（调整位置避免重叠，加粗显示）
% for i = 1:size(psnr_data,1)
%     for j = 1:size(psnr_data,2)
%         text(j, psnr_data(i,j)+0.01, sprintf('%.2f', psnr_data(i,j)), ...
%              'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'FontSize', 9, ...
%              'FontWeight', 'bold');
%     end
% end
% grid on;
% % 保存图表
% print('ipso/PSNR_Comparison_lenna.tif', '-dtiff', '-r600');
% 
% % 7.3 MSE对比图（调整Y轴范围放大差异）
% figure('Position', [300, 200, 600, 400]);
% h = bar(X, mse_data, 0.8);
% 
% % 设置更窄的Y轴范围来放大MSE差异
% mse_min = min(mse_data(:)) - 0.5;
% mse_max = max(mse_data(:)) + 0.5;
% ylim([mse_min, mse_max]);
% 
% set(gca, 'XTickLabel', algo_labels, 'FontName', 'Times New Roman', 'FontSize', 11);
% ylabel('MSE', 'FontName', 'Times New Roman', 'FontSize', 12);
% xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
% title('MSE Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
% hl = legend({'Mean MSE', 'Best MSE'}, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
% set(hl, 'Orientation', 'horizontal');
% % 数值标注（调整位置，加粗显示）
% for i = 1:size(mse_data,1)
%     for j = 1:size(mse_data,2)
%         text(j, mse_data(i,j)+0.1, sprintf('%.2f', mse_data(i,j)), ...
%              'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'FontSize', 9, ...
%              'FontWeight', 'bold');
%     end
% end
% grid on;
% print('ipso/MSE_Comparison_lenna.tif', '-dtiff', '-r600');
% 
% % 7.4 SSIM对比图（调整Y轴范围放大差异）
% figure('Position', [400, 200, 600, 400]);
% h = bar(X, ssim_data, 0.8);
% 
% % 设置更窄的Y轴范围来放大SSIM差异
% ssim_min = min(ssim_data(:)) - 0.001;
% ssim_max = max(ssim_data(:)) + 0.002;
% ylim([ssim_min, ssim_max]);
% 
% set(gca, 'XTickLabel', algo_labels, 'FontName', 'Times New Roman', 'FontSize', 11);
% ylabel('SSIM', 'FontName', 'Times New Roman', 'FontSize', 12);
% xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
% title('SSIM Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
% hl = legend({'Mean SSIM', 'Best SSIM'}, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
% set(hl, 'Orientation', 'horizontal');
% % 数值标注（保留4位小数，调整位置，加粗显示）
% for i = 1:size(ssim_data,1)
%     for j = 1:size(ssim_data,2)
%         text(j, ssim_data(i,j)+0.0005, sprintf('%.4f', ssim_data(i,j)), ...
%              'HorizontalAlignment', 'center', 'FontName', 'Times New Roman', 'FontSize', 9, ...
%              'FontWeight', 'bold');
%     end
% end
% grid on;
% print('ipso/SSIM_Comparison_lenna.tif', '-dtiff', '-r600');



% 算法标签-------------------------------------------------------------
% 算法标签-------------------------------------------------------------
%% 7. 绘制老论文风格对比图表（数值左对齐版）
% 7.1 数据整理与参数设置
algo_labels = {'BPR', 'PSOBPR', 'IPSOBPR'};
data_rows = {'Mean', 'Best'};
colors = [0 0.447 0.741; 0.85 0.325 0.098];  % 蓝、红
bar_width = 0.8;  % 总宽度（每组柱子的整体宽度）
single_bar_width = bar_width / length(data_rows);  % 单个柱子宽度（平分总宽度）

% 7.2 PSNR对比图（数值左对齐）
figure('Position', [200, 200, 600, 400]);
X = 1:3;
h = bar(X, psnr_data, bar_width);  % 绘制分组柱状图
set(h(1), 'FaceColor', colors(1,:));  % Mean（蓝）
set(h(2), 'FaceColor', colors(2,:));  % Best（红）

% Y轴范围调整
psnr_min = min(psnr_data(:)) - 0.05;
psnr_max = max(psnr_data(:)) + 0.1;
ylim([psnr_min, psnr_max]);

% 基础格式
set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
    'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('PSNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 12);
xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
title('PSNR Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
hl = legend(data_rows, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
set(hl, 'Orientation', 'horizontal');

% -------------------------- 核心修正：正确计算每个柱子的左侧位置 --------------------------
for i = 1:length(h)  % i=1（蓝柱，Mean），i=2（红柱，Best）
    % 获取每个柱子的实际X坐标（考虑分组偏移）
    x_endpoints = h(i).XEndPoints;  % 每个柱子的右侧端点坐标
    
    for j = 1:length(x_endpoints)
        val = psnr_data(i, j);
        % 计算柱子左侧X坐标：右侧端点 - 单个柱子宽度
        left_x = x_endpoints(j) - single_bar_width;
        % 标注：左对齐，从柱子左侧开始
        text(left_x + single_bar_width - 0.05, val + 0.01, sprintf('%.2f', val), ...
             'HorizontalAlignment', 'left', ...  % 左对齐
             'VerticalAlignment', 'bottom', ...
             'FontName', 'Times New Roman', 'FontSize', 9, ...
             'FontWeight', 'bold', ...
             'Color', colors(i,:));  % 数值颜色与柱子一致
    end
end

grid on;
print('ipso/PSNR_Comparison_lenna.tif', '-dtiff', '-r600');

% 7.3 MSE对比图（同逻辑）
figure('Position', [300, 200, 600, 400]);
h = bar(X, mse_data, bar_width);
set(h(1), 'FaceColor', colors(1,:));
set(h(2), 'FaceColor', colors(2,:));

mse_min = min(mse_data(:)) - 0.5;
mse_max = max(mse_data(:)) + 0.5;
ylim([mse_min, mse_max]);

set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
    'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('MSE', 'FontName', 'Times New Roman', 'FontSize', 12);
xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
title('MSE Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
hl = legend(data_rows, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
set(hl, 'Orientation', 'horizontal');

% 数值左对齐标注
for i = 1:length(h)
    x_endpoints = h(i).XEndPoints;
    for j = 1:length(x_endpoints)
        val = mse_data(i, j);
        left_x = x_endpoints(j) - single_bar_width;
        text(left_x + single_bar_width - 0.05, val + 0.1, sprintf('%.2f', val), ...
             'HorizontalAlignment', 'left', ...
             'VerticalAlignment', 'bottom', ...
             'FontName', 'Times New Roman', 'FontSize', 9, ...
             'FontWeight', 'bold', ...
             'Color', colors(i,:));
    end
end

grid on;
print('ipso/MSE_Comparison_lenna.tif', '-dtiff', '-r600');

% 7.4 SSIM对比图（同逻辑）
figure('Position', [400, 200, 600, 400]);
h = bar(X, ssim_data, bar_width);
set(h(1), 'FaceColor', colors(1,:));
set(h(2), 'FaceColor', colors(2,:));

ssim_min = min(ssim_data(:)) - 0.001;
ssim_max = max(ssim_data(:)) + 0.002;
ylim([ssim_min, ssim_max]);

set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
    'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('SSIM', 'FontName', 'Times New Roman', 'FontSize', 12);
xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
title('SSIM Comparison (lenna.tif)', 'FontSize', 13, 'FontWeight', 'bold');
hl = legend(data_rows, 'Location', 'North', 'FontName', 'Times New Roman', 'FontSize', 10);
set(hl, 'Orientation', 'horizontal');

% 数值左对齐标注
for i = 1:length(h)
    x_endpoints = h(i).XEndPoints;
    for j = 1:length(x_endpoints)
        val = ssim_data(i, j);
        left_x = x_endpoints(j) - single_bar_width;
        text(left_x + single_bar_width - 0.05, val + 0.0005, sprintf('%.4f', val), ...
             'HorizontalAlignment', 'left', ...
             'VerticalAlignment', 'bottom', ...
             'FontName', 'Times New Roman', 'FontSize', 9, ...
             'FontWeight', 'bold', ...
             'Color', colors(i,:));
    end
end

grid on;
print('ipso/SSIM_Comparison_lenna.tif', '-dtiff', '-r600');
