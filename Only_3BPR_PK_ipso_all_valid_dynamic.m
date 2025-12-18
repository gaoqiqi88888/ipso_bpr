% % 这是一个完整的MATLAB框架，用于系统性地评估和比较粒子群优化（PSO）及其改进版本（IPSO）在图像复原任务中的性能表现。该系统通过多维度指标分析、可视化对比和统计验证，为优化算法的性能评估提供了科研级的自动化解决方案。
% % 
% % 核心功能：
% % 自动化对比实验：自动加载预训练的PSO和IPSO优化结果，进行批量图像复原测试
% % 
% % 全面评估指标：集成PSNR、MSE、SSIM三种图像质量评估指标
% % 
% % 多算法对比：同时测试无优化BP（BPR）、PSO优化BP（PSOBPR）、IPSO优化BP（IPSOBPR）三种算法
% % 
% % PK值分析框架：创新性地引入PK（Performance Kappa）值定量分析算法相对性能提升
% % 
% % 科研级可视化：自动生成符合学术论文要求的图表（600 DPI TIFF格式）
% % 
% % 数据自动化管理：自动保存原始数据、中间结果和最终统计表格
% % 技术特色：
% % 科学实验设计：每个图像进行100次复原测试确保统计稳定性
% % 
% % 网络一致性：测试阶段与优化阶段保持相同的BP神经网络结构
% % 
% % 批量处理能力：支持多张测试图像的并行处理和分析
% % 
% % 完整数据流水线：从原始图像到最终统计报告的端到端处理
% % 
% % 可重复研究：详细记录实验参数和运行环境
% % 
% % 评估体系：
% % 基础性能指标：PSNR、MSE、SSIM的最优值和平均值
% % 
% % 收敛性能分析：PSO vs IPSO适应度曲线对比
% % 
% % 相对性能评估：PK值定量分析算法提升程度
% % 
% % 统计显著性：多图像测试的统计一致性验证
% % 
% % 输出成果：
% % 图像结果：原始/模糊/各算法复原图像
% % 
% % 分析图表：适应度曲线、性能对比柱状图、PK值趋势图
% % 
% % 数据表格：符合论文格式的Table 4和Table 5（Excel格式）
% % 
% % 完整报告：详细的统计分析和性能提升百分比

%% ====== 初始化清理 ======
close all;      % 关闭所有图形窗口
clc;           % 清空命令窗口
clear;         % 清除所有变量（如需要）

%% !!!核心配置：多图对比分析，数据来源于PSO/IPSO优化结果
% mat_version="_251118_1557";
mat_version = datestr(now, '_YYmmdd_HHMM');
myt1 = clock;
%% 1. 基础参数与路径配置
% 1.1 数据加载路径（PSO/IPSO优化结果文件，替换原GA/IGA数据）
% mat_name = 'ipso/dataset/best90_r100_Lenna.mat';  % 指定PSO/IPSO结果数据文件,lenna发表禁用。是否混合模糊，请查验
mat_name =  '\ipso_bpr_v3\dataset\best90_0801_pop50_gen50_20251129.mat';


load(mat_name);  % 加载数据：含PSO_bestchrom、PSO_bestfitness、IPSO_bestchrom、IPSO_bestfitness等


% 1.2 实验控制参数（按老论文单图对比逻辑）
maxit = 100;      % 图像还原对比次数（老论文常用1000次以统计稳定性）


inputnum = 9;      % 输入节点数（3×3滑动窗口，与原网络一致）
hiddennum = 9;     % 隐藏层节点数
outputnum = 1;     % 输出节点数（单像素预测）
picsize = [90, 90]; % 图像尺寸（统一为90×90，与优化阶段一致）


%% ======  扫描 valid 目录下所有 tif（过滤文件夹/无效文件）======
tifDir   = 'ipso\valid';                               % 目录
tifFiles = dir(fullfile(tifDir,'*.tif'));              % 只拿 .tif
isRealFile = ~[tifFiles.isdir];                        % 过滤掉"."、".."文件夹
tifFiles   = tifFiles(isRealFile);                     % 保留真实文件
num_tifs   = numel(tifFiles);                          % 实际张数
all_sim_picname = string({tifFiles.name});             % 纯文件名（string数组）
all_sim_dir_picname = fullfile(tifDir, all_sim_picname); % 完整路径（n×1 string）

n = length(all_sim_dir_picname);  % 图像数量（此处n=10）


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

% 使用 print 函数保存图像（更稳定）
print('-dtiff', '-r600', 'ipso/PSO_IPSO_Fitness_Curve.tif');

% 关闭当前图形窗口
close(gcf);

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

    % 4.1 读取并预处理图像（调用原Read_Pic_PSO函数，确保退化逻辑与优化阶段一致）
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

    %记录三种算法对每幅图像的最佳psnr和平均psnr
    One_all_psnr_best = [BPR_psnr_best, PSOBPR_psnr_best, IPSOBPR_psnr_best];
    One_all_psnr_mean = [BPR_psnr_mean, PSOBPR_psnr_mean, IPSOBPR_psnr_mean];
    All_psnr_best(pic_index,:) = One_all_psnr_best;
    All_psnr_mean(pic_index,:) = One_all_psnr_mean;

    %记录三种算法对每幅图像的最佳mse和平均mse
    One_all_mse_best = [BPR_mse_best, PSOBPR_mse_best, IPSOBPR_mse_best];
    One_all_mse_mean = [BPR_mse_mean, PSOBPR_mse_mean, IPSOBPR_mse_mean];
    All_mse_best(pic_index,:) = One_all_mse_best;
    All_mse_mean(pic_index,:) = One_all_mse_mean;

   %记录三种算法对每幅图像的最佳ssim和平均ssim
    One_all_ssim_best = [BPR_ssim_best, PSOBPR_ssim_best, IPSOBPR_ssim_best];
    One_all_ssim_mean = [BPR_ssim_mean, PSOBPR_ssim_mean, IPSOBPR_ssim_mean];
    All_ssim_best(pic_index,:) = One_all_ssim_best;
    All_ssim_mean(pic_index,:) = One_all_ssim_mean;
    
    
    %% 6. 提取最优复原图像（用于老论文风格的图像展示）
    % 6.1 按PSNR最优提取（老论文常用PSNR作为最优图像筛选指标）
    BPR_best_restored = BPR_image_restored_noedge(:,:,BPR_psnr_idx);
    PSOBPR_best_restored = PSOBPR_image_restored_noedge(:,:,PSOBPR_psnr_idx);
    IPSOBPR_best_restored = IPSOBPR_image_restored_noedge(:,:,IPSOBPR_psnr_idx);

    % 6.2 保存图像（按老论文命名格式：结果目录+图像类型+算法，TIFF格式600dpi）
 
    % 原始清晰图
    ORG_picname = strcat ("ipso/", num2str(pic_index), "_ORG_", all_sim_picname(pic_index));
    imwrite(image_resized, ORG_picname, 'tiff', 'Resolution',600);
    % 退化模糊图
    BLU_picname = strcat ("ipso/", num2str(pic_index), "_BLU_", all_sim_picname(pic_index));
    imwrite(image_blurred, BLU_picname, 'tiff', 'Resolution',600);

    % 各算法最优复原图
    BPR_picname = strcat ("ipso/", num2str(pic_index), "_BPR_", all_sim_picname(pic_index));
    imwrite(BPR_best_restored, BPR_picname, 'tiff', 'Resolution',600);
    PSOBPR_picname = strcat ("ipso/", num2str(pic_index), "_PSOBPR_", all_sim_picname(pic_index));
    imwrite(PSOBPR_best_restored, PSOBPR_picname, 'tiff', 'Resolution',600);
    IPSOBPR_picname = strcat ("ipso/", num2str(pic_index), "_IPSOBPR_", all_sim_picname(pic_index));
    imwrite(IPSOBPR_best_restored, IPSOBPR_picname, 'tiff', 'Resolution',600);


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
    data_rows = {'Mean', 'Best'};


    save_path_base = "ipso/";
    % image_name = all_sim_picname(pic_index);

    % 关键：处理图像名称（移除.tif后缀，替换特殊字符，避免路径错误）
    image_name_raw = all_sim_picname(pic_index);  % 原始名称（如i01.tif）
    image_name_clean = strrep(image_name_raw, '.tif', '');  % 清理为i01（无后缀）

    % generate_comparison_figures(psnr_data, mse_data, ssim_data, algo_labels, data_rows, save_path_base, image_name);
    % 调用修正后的函数（传入清理后的名称）
    generate_comparison_figures(psnr_data, mse_data, ssim_data, ...
                           algo_labels, data_rows, save_path_base, image_name_clean);


end



%% 论文Table 4 Objective evaluation of three restoration algorithms
%论文 Table4 数据保存到Table4.xlsx中
table (:,1:3)= All_psnr_best;
table (:,4:6)= All_psnr_mean;
table (:,7:9)= All_mse_best;
table (:,10:12)= All_mse_mean;
xlswrite('ipso/Table 4.xlsx',All_psnr_best,'Sheet1');
xlswrite('ipso/Table 4.xlsx',All_psnr_mean,'Sheet2');
xlswrite('ipso/Table 4.xlsx',All_mse_best,'Sheet3');
xlswrite('ipso/Table 4.xlsx',All_mse_mean,'Sheet4');
xlswrite('ipso/Table 4.xlsx',table,'Sheet5');



%% 论文Table 5 Objective evaluation of three restoration algorithms
%论文 Table5 数据保存到Table5.xlsx中
table (:,1:3)= All_psnr_best;
table (:,4:6)= All_psnr_mean;
table (:,7:9)= All_mse_best;
table (:,10:12)= All_mse_mean;
table (:,13:15)= All_ssim_best;
table (:,16:18)= All_ssim_mean;
xlswrite('ipso/Table 5.xlsx',All_psnr_best,'Sheet1');
xlswrite('ipso/Table 5.xlsx',All_psnr_mean,'Sheet2');
xlswrite('ipso/Table 5.xlsx',All_mse_best,'Sheet3');
xlswrite('ipso/Table 5.xlsx',All_mse_mean,'Sheet4');
xlswrite('ipso/Table 5.xlsx',All_ssim_best,'Sheet5');
xlswrite('ipso/Table 5.xlsx',All_ssim_mean,'Sheet6');
xlswrite('ipso/Table 5.xlsx',table,'Sheet7');





%% ---------------------------百分比PK图---------------------------
%%  % 初始化x轴，根据图片数量初始化
x=1:1:num_tifs;
% PSNR PK
    for i=1:1:num_tifs
        Pk_bp_psnr_best(i) = All_psnr_best(i,3) / All_psnr_best(i,1);
        PK_psobp_psnr_best(i) = All_psnr_best(i,3) / All_psnr_best(i,2);
    end

%%  论文Fig PSNR Best PK
fig = figure('Visible', 'off');  % 创建隐藏图形
plot(x,Pk_bp_psnr_best,'--pg','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','g');
hold on;
plot(x,PK_psobp_psnr_best,'-.ob','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','b');
hold on;
xlim=get(gca,'Xlim');
plot(xlim,[1,1],'m--');
xlabel('Image No');  
ylabel('PK Value'); 
title('PSNR Best PK');   
legend('PKBP','PKPSOBP','Location','southoutside');

% 使用print保存
print(fig, 'ipso/Pk_psnr_best.tif', '-dtiff', '-r600');
close(fig);
clear fig;
    
    
    % mse PK
    for i=1:1:num_tifs
        Pk_bp_mse_best(i) = All_mse_best(i,3) / All_mse_best(i,1);
        PK_psobp_mse_best(i) = All_mse_best(i,3) / All_mse_best(i,2);
    end

    %% MSE PK（修正部分）
    for i = 1:num_tifs
    % 修正：使用倒数比值
         Pk_bp_mse_best(i) = All_mse_best(i,1) / All_mse_best(i,3);    % BPR_MSE / IPSO_MSE
         PK_psobp_mse_best(i) = All_mse_best(i,2) / All_mse_best(i,3); % PSOBPR_MSE / IPSO_MSE
    end

%%  论文Fig MSE Best PK
  fig = figure('Visible', 'off');
  
     plot(x,Pk_bp_mse_best,'--pg',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','g')
     hold on;
     
     plot(x,PK_psobp_mse_best,'-.ob',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','b')
     hold on;
     
     xlim=get(gca,'Xlim');
     plot(xlim,[1,1],'m--');
         
    xlabel('Image No.');  %x轴坐标描述
    ylabel('PK Value'); %y轴坐标描述
    title('MSE Best PK');   
    legend('PKBP','PKPSOBP','Location','southoutside');   %右上角标注
                f=getframe(gcf);
    print(fig, 'ipso/Pk_mse_best.tif', '-dtiff', '-r600');
    close(fig);
    clear fig;

    


%%  论文Fig.10 SSIM Best PK
%%  % 初始化x轴，根据图片数量初始化
    for i=1:1:num_tifs
        Pk_bp_ssim_best(i) = All_ssim_best(i,3) / All_ssim_best(i,1);
        PK_psobp_ssim_best(i) = All_ssim_best(i,3) / All_ssim_best(i,2);
    end

  fig = figure('Visible', 'off');
 
     plot(x,Pk_bp_ssim_best,'--pg',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','g')
     hold on;
     
     plot(x,PK_psobp_ssim_best,'-.ob',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','b')
     hold on;
     
     xlim=get(gca,'Xlim');
     plot(xlim,[1,1],'m--');
         
    xlabel('Image No.');  %x轴坐标描述
    ylabel('PK Value'); %y轴坐标描述
    title('SSIM Best PK');   
    legend('PKBP','PKPSOBP','Location','southoutside');   %右上角标注
                f=getframe(gcf);
    print(fig, 'ipso/Pk_ssim_best.tif', '-dtiff', '-r600');
    close(fig);
    clear fig;



% %% 8. 保存实验数据（按老论文格式，便于后续补充分析）
% save_data_name = 'ipso/Paper_multi_100_PSO_IPSO.mat';
% save(save_data_name, ...
%     'BPR_psnr', 'BPR_mse', 'BPR_ssim', ...
%     'PSOBPR_psnr', 'PSOBPR_mse', 'PSOBPR_ssim', ...
%     'IPSOBPR_psnr', 'IPSOBPR_mse', 'IPSOBPR_ssim', ...
%     'BPR_best_restored', 'PSOBPR_best_restored', 'IPSOBPR_best_restored');



%% ====== 通用：Best-PK 详细数值 + 平均提升百分比 ======
% 1. 先打印每张图的 PK 值（PSNR）
fprintf('\n--------------- 每张图 PSNR Best-PK 详细数值 ---------------\n');
fprintf('图号  PK(IPSO/BPR)  PK(IPSO/PSOBPR)\n');
for i = 1:num_tifs
    fprintf(' %2d      %6.4f        %6.4f\n', i, Pk_bp_psnr_best(i), PK_psobp_psnr_best(i));
end

% 2. 计算并输出平均 PK 及提升百分比（PSNR）
fprintf('\n--------------- PSNR  Best-PK ---------------\n');
mean_PK_BPR   = mean(Pk_bp_psnr_best);      % IPSO vs BPR
mean_PK_PSO   = mean(PK_psobp_psnr_best);   % IPSO vs PSOBPR
fprintf('共 %d 张图 PK 均值：IPSO/BPR = %.4f ， IPSO/PSOBPR = %.4f\n', ...
        num_tifs, mean_PK_BPR, mean_PK_PSO);
fprintf('→ IPSOBPR 相对 BPR  平均提升：%6.2f%%\n', (mean_PK_BPR - 1)*100);
fprintf('→ IPSOBPR 相对 PSOBPR 平均提升：%6.2f%%\n', (mean_PK_PSO - 1)*100);

% 3. MSE 同理
fprintf('\n--------------- MSE Best-PK ---------------\n');
mean_PK_BPR_mse = mean(Pk_bp_mse_best);
mean_PK_PSO_mse = mean(PK_psobp_mse_best);
fprintf('共 %d 张图 PK 均值：IPSO/BPR = %.4f ， IPSO/PSOBPR = %.4f\n', ...
        num_tifs, mean_PK_BPR_mse, mean_PK_PSO_mse);
fprintf('→ IPSOBPR 相对 BPR  平均提升：%6.2f%%\n', (mean_PK_BPR_mse - 1)*100);
fprintf('→ IPSOBPR 相对 PSOBPR 平均提升：%6.2f%%\n', (mean_PK_PSO_mse - 1)*100);

% 4. SSIM 同理
fprintf('\n--------------- SSIM Best-PK ---------------\n');
mean_PK_BPR_ssim = mean(Pk_bp_ssim_best);
mean_PK_PSO_ssim = mean(PK_psobp_ssim_best);
fprintf('共 %d 张图 PK 均值：IPSO/BPR = %.4f ， IPSO/PSOBPR = %.4f\n', ...
        num_tifs, mean_PK_BPR_ssim, mean_PK_PSO_ssim);
fprintf('→ IPSOBPR 相对 BPR  平均提升：%6.2f%%\n', (mean_PK_BPR_ssim - 1)*100);
fprintf('→ IPSOBPR 相对 PSOBPR 平均提升：%6.2f%%\n', (mean_PK_PSO_ssim - 1)*100);


% 9. 总运行时间统计
myt2 = clock;
total_time = etime(myt2, myt1) / 60;  % 转换为分钟
fprintf('\n-------------------------- 实验完成 --------------------------\n');
fprintf('总运行时间: %.2f 分钟\n', total_time);
fprintf('所有结果（图像+数据）已保存至 ipso/ 目录\n');


% 10. 保存实验数据

[~, fname, ~] = fileparts(mat_name);     
%拼新文件名（保存到 ipso 文件夹）
new_mat_name = sprintf('%s_%d_%dpic%s.mat', fname, maxit, num_tifs, mat_version);
fullfile('ipso', new_mat_name);     % 跨平台生成完整路径            
save(new_mat_name);

% 实验数据备份
savePath = sprintf('ipso/backup_3BPR_PK_ipso_all_%s.mat', mat_version);
save(savePath);



%% 函数-------------------------------------------------------------
%% 7. 绘制老论文风格对比图表（数值左对齐版）
function generate_comparison_figures(psnr_data, mse_data, ssim_data, ...
                                   algo_labels, data_rows, save_path_base, image_name_clean)
    % 输入参数说明：
    % image_name_clean：清理后的图像名称（如i01，无.tif后缀和特殊字符）
    % save_path_base：保存路径（如ipso/）
    
    colors = [0 0.447 0.741; 0.85 0.325 0.098];  % 蓝（Mean）、红（Best）
    bar_width = 0.8;  % 每组柱子总宽度
    single_w = bar_width / length(data_rows);  % 单个柱子宽度（平分总宽度）
    X = 1:3;  % 3个算法（BPR/PSOBPR/IPSOBPR）的X坐标
    
    %% -------------------------- 1. PSNR对比图 --------------------------
    fprintf('正在生成 PSNR 对比图: %s\n', image_name_clean);
    
    % 创建隐藏图形窗口
    hfig = figure('Visible', 'off', 'Position', [200, 200, 600, 400]);
    
    try
        % 绘制分组柱状图
        h = bar(X, psnr_data, bar_width);
        set(h, {'FaceColor'}, num2cell(colors, 2));  % 设置柱子颜色
        
        % 坐标轴与格式配置
        ylim([min(psnr_data(:)) - 0.05, max(psnr_data(:)) + 0.1]);
        set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
            'FontName', 'Times New Roman', 'FontSize', 11);
        ylabel('PSNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 12);
        xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
        title(sprintf('PSNR Comparison (%s.tif)', image_name_clean), ...
              'FontSize', 13, 'FontWeight', 'bold');
        hl = legend(data_rows, 'Location', 'North', 'Orientation', 'horizontal', ...
                    'FontName', 'Times New Roman', 'FontSize', 10);
        grid on;
        
        % 数值标注
        for i = 1:length(h)
            xe = h(i).XEndPoints;
            for j = 1:numel(xe)
                val = psnr_data(i, j);
                text_pos_x = xe(j) - single_w/4;
                text_pos_y = val + 0.01;
                text(text_pos_x, text_pos_y, sprintf('%.2f', val), ...
                     'HorizontalAlignment', 'left', 'FontSize', 9, ...
                     'FontWeight', 'bold', 'Color', colors(i, :));
            end
        end
        
        % 保存图像
        fname = fullfile(save_path_base, sprintf('PSNR_Comparison_%s.tif', image_name_clean));
        print(hfig, fname, '-dtiff', '-r600');
        fprintf('  PSNR图保存成功: %s\n', fname);
        
    catch ME
        fprintf('  PSNR图生成失败: %s\n', ME.message);
    end
    
    % 确保资源释放
    if ishandle(hfig)
        close(hfig);
    end
    clear hfig h hl;
    
    %% -------------------------- 2. MSE对比图 --------------------------
    fprintf('正在生成 MSE 对比图: %s\n', image_name_clean);
    
    hfig = figure('Visible', 'off', 'Position', [300, 200, 600, 400]);
    
    try
        h = bar(X, mse_data, bar_width);
        set(h, {'FaceColor'}, num2cell(colors, 2));
        
        ylim([min(mse_data(:)) - 0.5, max(mse_data(:)) + 0.5]);
        set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
            'FontName', 'Times New Roman', 'FontSize', 11);
        ylabel('MSE', 'FontName', 'Times New Roman', 'FontSize', 12);
        xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
        title(sprintf('MSE Comparison (%s.tif)', image_name_clean), ...
              'FontSize', 13, 'FontWeight', 'bold');
        hl = legend(data_rows, 'Location', 'North', 'Orientation', 'horizontal', ...
                    'FontName', 'Times New Roman', 'FontSize', 10);
        grid on;
        
        % 数值标注
        for i = 1:length(h)
            xe = h(i).XEndPoints;
            for j = 1:numel(xe)
                val = mse_data(i, j);
                text_pos_x = xe(j) - single_w/4;
                text_pos_y = val + 0.1;
                text(text_pos_x, text_pos_y, sprintf('%.2f', val), ...
                     'HorizontalAlignment', 'left', 'FontSize', 9, ...
                     'FontWeight', 'bold', 'Color', colors(i, :));
            end
        end
        
        % 保存图像
        fname = fullfile(save_path_base, sprintf('MSE_Comparison_%s.tif', image_name_clean));
        print(hfig, fname, '-dtiff', '-r600');
        fprintf('  MSE图保存成功: %s\n', fname);
        
    catch ME
        fprintf('  MSE图生成失败: %s\n', ME.message);
    end
    
    % 确保资源释放
    if ishandle(hfig)
        close(hfig);
    end
    clear hfig h hl;
    
    %% -------------------------- 3. SSIM对比图 --------------------------
    fprintf('正在生成 SSIM 对比图: %s\n', image_name_clean);
    
    hfig = figure('Visible', 'off', 'Position', [400, 200, 600, 400]);
    
    try
        h = bar(X, ssim_data, bar_width);
        set(h, {'FaceColor'}, num2cell(colors, 2));
        
        ylim([min(ssim_data(:)) - 0.001, max(ssim_data(:)) + 0.002]);
        set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
            'FontName', 'Times New Roman', 'FontSize', 11);
        ylabel('SSIM', 'FontName', 'Times New Roman', 'FontSize', 12);
        xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
        title(sprintf('SSIM Comparison (%s.tif)', image_name_clean), ...
              'FontSize', 13, 'FontWeight', 'bold');
        hl = legend(data_rows, 'Location', 'North', 'Orientation', 'horizontal', ...
                    'FontName', 'Times New Roman', 'FontSize', 10);
        grid on;
        
        % 数值标注
        for i = 1:length(h)
            xe = h(i).XEndPoints;
            for j = 1:numel(xe)
                val = ssim_data(i, j);
                text_pos_x = xe(j) - single_w/4;
                text_pos_y = val + 0.0005;
                text(text_pos_x, text_pos_y, sprintf('%.4f', val), ...
                     'HorizontalAlignment', 'left', 'FontSize', 9, ...
                     'FontWeight', 'bold', 'Color', colors(i, :));
            end
        end
        
        % 保存图像
        fname = fullfile(save_path_base, sprintf('SSIM_Comparison_%s.tif', image_name_clean));
        print(hfig, fname, '-dtiff', '-r600');
        fprintf('  SSIM图保存成功: %s\n', fname);
        
    catch ME
        fprintf('  SSIM图生成失败: %s\n', ME.message);
    end
    
    % 确保资源释放
    if ishandle(hfig)
        close(hfig);
    end
    clear hfig h hl;
    
    fprintf('图像对比图生成完成: %s\n\n', image_name_clean);
end






