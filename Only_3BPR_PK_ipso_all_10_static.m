%% 重要代码，PSO和IPSO单一图像的一次对比

% Paper_single_1000_artifical_dynamic.m（适配PSO/IPSO对比与artificial图分析，可直接复制）
%% 清空环境变量
clear
close all
clc

    load("ipso/valid_dataset/3rpk_test101801.mat");

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

%%  论文Fig.8 PSNR Best PK
  figure;
 
     plot(x,Pk_bp_psnr_best,'--pg',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','g')
     hold on;
     
     plot(x,PK_psobp_psnr_best,'-.ob',...
        'LineWidth',2,...
        'MarkerSize',10,...
        'MarkerEdgeColor','b')
     hold on;
     
     xlim=get(gca,'Xlim');
     plot(xlim,[1,1],'m--');
         
    xlabel('Image No');  %x轴坐标描述
    ylabel('PK Value'); %y轴坐标描述
    title('PSNR Best PK');   
    legend('PKBP','PKPSOBP','Location','southoutside');   %右上角标注
                f=getframe(gcf);
    imwrite(f.cdata,['ipso/','Pk_psnr_best.tif'], 'tiff', 'Resolution',600); 
    print(['ipso/','Pk_pnsr_best.eps'], '-depsc');

    
    
    % mse PK
    for i=1:1:num_tifs
        Pk_bp_mse_best(i) = All_mse_best(i,3) / All_mse_best(i,1);
        PK_psobp_mse_best(i) = All_mse_best(i,3) / All_mse_best(i,2);
    end


%%  论文Fig.9 MSE Best PK
  figure; 
  
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
    imwrite(f.cdata,['ipso/','Pk_mse_best.tif'], 'tiff', 'Resolution',600); 
    print(['ipso/','Pk_mse_best.eps'], '-depsc');


    


%%  论文Fig.10 SSIM Best PK
%%  % 初始化x轴，根据图片数量初始化
    for i=1:1:num_tifs
        Pk_bp_ssim_best(i) = All_ssim_best(i,3) / All_ssim_best(i,1);
        PK_psobp_ssim_best(i) = All_ssim_best(i,3) / All_ssim_best(i,2);
    end

  figure;
 
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
    imwrite(f.cdata,['ipso/','Pk_ssim_best.tif'], 'tiff', 'Resolution',600); 
    print(['ipso/','Pk_ssim_best.eps'], '-depsc');



% %% 8. 保存实验数据（按老论文格式，便于后续补充分析）
% save_data_name = 'ipso/Paper_multi_100_PSO_IPSO.mat';
% save(save_data_name, ...
%     'BPR_psnr', 'BPR_mse', 'BPR_ssim', ...
%     'PSOBPR_psnr', 'PSOBPR_mse', 'PSOBPR_ssim', ...
%     'IPSOBPR_psnr', 'IPSOBPR_mse', 'IPSOBPR_ssim', ...
%     'BPR_best_restored', 'PSOBPR_best_restored', 'IPSOBPR_best_restored');

% 9. 总运行时间统计（老论文实验报告必备）
myt2 = clock;
total_time = etime(myt2, myt1) / 60;  % 转换为分钟
fprintf('\n-------------------------- 实验完成 --------------------------\n');
fprintf('总运行时间: %.2f 分钟\n', total_time);
fprintf('所有结果（图像+数据）已保存至 ipso/ 目录\n');




%% 新增
%% ---------------------------三种算法最佳值对比图---------------------------
% fprintf('=== 生成三种算法最佳值对比图 ===\n');
% 
% % 准备图像名称（清理后缀）
% image_names_clean = cell(1, num_tifs);
% for i = 1:num_tifs
%     image_names_clean{i} = strrep(all_sim_picname(i), '.tif', '');
% end
% 
% % 调用绘图函数
% plot_best_comparison(All_psnr_best, All_mse_best, All_ssim_best, ...
%                     {'BPR', 'PSOBPR', 'IPSOBPR'}, image_names_clean, 'ipso/');

% fprintf('最佳值对比图已保存至 ipso/ 目录\n');
%% ---------------------------算法均值统计打印---------------------------
fprintf('\n\n=== 详细算法均值统计 ===\n');

% 打印Best值的均值统计
fprintf('\n=== 算法Best值均值统计 ===\n');
fprintf('PSNR Best均值 - BPR: %.4f dB, PSOBPR: %.4f dB, IPSOBPR: %.4f dB\n', ...
        mean(All_psnr_best(:,1)), mean(All_psnr_best(:,2)), mean(All_psnr_best(:,3)));
fprintf('MSE Best均值  - BPR: %.4f, PSOBPR: %.4f, IPSOBPR: %.4f\n', ...
        mean(All_mse_best(:,1)), mean(All_mse_best(:,2)), mean(All_mse_best(:,3)));
fprintf('SSIM Best均值 - BPR: %.6f, PSOBPR: %.6f, IPSOBPR: %.6f\n', ...
        mean(All_ssim_best(:,1)), mean(All_ssim_best(:,2)), mean(All_ssim_best(:,3)));

% 打印Mean值的均值统计
fprintf('\n=== 算法Mean值均值统计 ===\n');
fprintf('PSNR Mean均值 - BPR: %.4f dB, PSOBPR: %.4f dB, IPSOBPR: %.4f dB\n', ...
        mean(All_psnr_mean(:,1)), mean(All_psnr_mean(:,2)), mean(All_psnr_mean(:,3)));
fprintf('MSE Mean均值  - BPR: %.4f, PSOBPR: %.4f, IPSOBPR: %.4f\n', ...
        mean(All_mse_mean(:,1)), mean(All_mse_mean(:,2)), mean(All_mse_mean(:,3)));
fprintf('SSIM Mean均值 - BPR: %.6f, PSOBPR: %.6f, IPSOBPR: %.6f\n', ...
        mean(All_ssim_mean(:,1)), mean(All_ssim_mean(:,2)), mean(All_ssim_mean(:,3)));

% 性能提升统计
fprintf('\n=== 性能提升统计 (相对于BPR) ===\n');
psnr_improvement_pso = (mean(All_psnr_best(:,2)) - mean(All_psnr_best(:,1))) / mean(All_psnr_best(:,1)) * 100;
psnr_improvement_ipso = (mean(All_psnr_best(:,3)) - mean(All_psnr_best(:,1))) / mean(All_psnr_best(:,1)) * 100;
fprintf('PSNR提升 - PSOBPR: %.2f%%, IPSOBPR: %.2f%%\n', psnr_improvement_pso, psnr_improvement_ipso);

mse_improvement_pso = (mean(All_mse_best(:,1)) - mean(All_mse_best(:,2))) / mean(All_mse_best(:,1)) * 100;
mse_improvement_ipso = (mean(All_mse_best(:,1)) - mean(All_mse_best(:,3))) / mean(All_mse_best(:,1)) * 100;
fprintf('MSE降低  - PSOBPR: %.2f%%, IPSOBPR: %.2f%%\n', mse_improvement_pso, mse_improvement_ipso);

ssim_improvement_pso = (mean(All_ssim_best(:,2)) - mean(All_ssim_best(:,1))) / mean(All_ssim_best(:,1)) * 100;
ssim_improvement_ipso = (mean(All_ssim_best(:,3)) - mean(All_ssim_best(:,1))) / mean(All_ssim_best(:,1)) * 100;
fprintf('SSIM提升 - PSOBPR: %.2f%%, IPSOBPR: %.2f%%\n', ssim_improvement_pso, ssim_improvement_ipso);




% 10. 保存实验数据

save_mat_name = erase(string(mat_name), '.mat');
new_mat_name=strcat(save_mat_name,"_1000_10pic_",mat_version,".mat");
save(new_mat_name);


save("ipso/3rpk_test101801");


function generate_comparison_figures(psnr_data, mse_data, ssim_data, ...
                                   algo_labels, data_rows, save_path_base, image_name_clean)
    % 输入参数说明：
    % image_name_clean：清理后的图像名称（如i01，无.tif后缀和特殊字符）
    % save_path_base：保存路径（如ipso/）
    
    colors = [0 0.447 0.741; 0.85 0.325 0.098];  % 蓝（Mean）、红（Best）
    bar_width = 0.8;  % 每组柱子总宽度
    single_w = bar_width / length(data_rows);  % 单个柱子宽度（平分总宽度）
    X = 1:3;  % 3个算法（BPR/PSOBPR/IPSOBPR）的X坐标
    
    % 确保保存路径存在
    if ~exist(save_path_base, 'dir')
        mkdir(save_path_base);
    end
    
    %% -------------------------- 1. PSNR对比图 --------------------------
    hfig1 = figure('Visible','off', 'Position',[200 200 750 550]);  % 增加宽度和高度
    h = bar(X, psnr_data, bar_width);
    set(h, {'FaceColor'}, num2cell(colors, 2));
    
    % 动态调整Y轴范围
    y_min = min(psnr_data(:));
    y_max = max(psnr_data(:));
    y_range = y_max - y_min;
    y_margin = y_range * 0.3;  % 更大的边距
    ylim([y_min - 0.05, y_max + y_margin]);
    
    set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel('PSNR (dB)', 'FontName', 'Times New Roman', 'FontSize', 12);
    xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
    title(sprintf('PSNR Comparison (%s.tif)', image_name_clean), ...
          'FontSize', 13, 'FontWeight', 'bold');
    
    % 图例放在顶部外部（最安全）
    hl = legend(data_rows, 'Location', 'northoutside', 'Orientation', 'horizontal', ...
                'FontName', 'Times New Roman', 'FontSize', 10);
    
    grid on;
    
    % 数值标注 - 统一放在柱子内部顶部
    for i = 1:length(h)
        xe = h(i).XEndPoints;
        for j = 1:numel(xe)
            val = psnr_data(i, j);
            text_pos_x = xe(j);
            
            % 统一放在柱子内部，避免所有遮挡问题
            text_pos_y = val - y_range * 0.03;
            
            text(text_pos_x, text_pos_y, sprintf('%.2f', val), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                 'FontSize', 9, 'FontWeight', 'bold', 'Color', colors(i, :), ...
                 'BackgroundColor', 'white', 'EdgeColor', colors(i, :), ...
                 'LineWidth', 0.5, 'Margin', 2);
        end
    end
    
    % 保存PSNR图像
    fname = fullfile(save_path_base, sprintf('PSNR_Comparison_%s.tif', image_name_clean));
    try
        print(hfig1, fname, '-dtiff', '-r600');
        fprintf('PSNR图像已保存: %s\n', fname);
    catch ME
        fprintf('保存PSNR图像时出错: %s\n', ME.message);
    end
    
    % 安全关闭图形
    if ishandle(hfig1)
        close(hfig1);
    end
    clear hfig1;
    
    %% -------------------------- 2. MSE对比图 --------------------------
    hfig2 = figure('Visible','off', 'Position',[300, 200, 750, 550]);
    h = bar(X, mse_data, bar_width);
    set(h, {'FaceColor'}, num2cell(colors, 2));
    
    % 动态调整Y轴范围
    y_min = min(mse_data(:));
    y_max = max(mse_data(:));
    y_range = y_max - y_min;
    y_margin = y_range * 0.3;
    ylim([max(0, y_min - 5), y_max + y_margin]);  % MSE最小值不能为负
    
    set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel('MSE', 'FontName', 'Times New Roman', 'FontSize', 12);
    xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
    title(sprintf('MSE Comparison (%s.tif)', image_name_clean), ...
          'FontSize', 13, 'FontWeight', 'bold');
    
    % 图例放在顶部外部
    hl = legend(data_rows, 'Location', 'northoutside', 'Orientation', 'horizontal', ...
                'FontName', 'Times New Roman', 'FontSize', 10);
    grid on;
    
    % 数值标注 - 统一放在柱子内部顶部
    for i = 1:length(h)
        xe = h(i).XEndPoints;
        for j = 1:numel(xe)
            val = mse_data(i, j);
            text_pos_x = xe(j);
            
            % 统一放在柱子内部
            text_pos_y = val - y_range * 0.03;
            
            text(text_pos_x, text_pos_y, sprintf('%.2f', val), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                 'FontSize', 9, 'FontWeight', 'bold', 'Color', colors(i, :), ...
                 'BackgroundColor', 'white', 'EdgeColor', colors(i, :), ...
                 'LineWidth', 0.5, 'Margin', 2);
        end
    end
    
    % 保存MSE图像
    fname = fullfile(save_path_base, sprintf('MSE_Comparison_%s.tif', image_name_clean));
    try
        print(hfig2, fname, '-dtiff', '-r600');
        fprintf('MSE图像已保存: %s\n', fname);
    catch ME
        fprintf('保存MSE图像时出错: %s\n', ME.message);
    end
    
    % 安全关闭图形
    if ishandle(hfig2)
        close(hfig2);
    end
    clear hfig2;
    
    %% -------------------------- 3. SSIM对比图 --------------------------
    hfig3 = figure('Visible','off', 'Position',[400, 200, 750, 550]);
    h = bar(X, ssim_data, bar_width);
    set(h, {'FaceColor'}, num2cell(colors, 2));
    
    % 动态调整Y轴范围
    y_min = min(ssim_data(:));
    y_max = max(ssim_data(:));
    y_range = y_max - y_min;
    y_margin = y_range * 0.3;
    ylim([max(0, y_min - 0.005), min(1, y_max + y_margin)]);  % SSIM范围[0,1]
    
    set(gca, 'XTick', X, 'XTickLabel', algo_labels, ...
        'FontName', 'Times New Roman', 'FontSize', 11);
    ylabel('SSIM', 'FontName', 'Times New Roman', 'FontSize', 12);
    xlabel('Algorithm', 'FontName', 'Times New Roman', 'FontSize', 12);
    title(sprintf('SSIM Comparison (%s.tif)', image_name_clean), ...
          'FontSize', 13, 'FontWeight', 'bold');
    
    % 图例放在顶部外部
    hl = legend(data_rows, 'Location', 'northoutside', 'Orientation', 'horizontal', ...
                'FontName', 'Times New Roman', 'FontSize', 10);
    grid on;
    
    % 数值标注 - 统一放在柱子内部顶部
    for i = 1:length(h)
        xe = h(i).XEndPoints;
        for j = 1:numel(xe)
            val = ssim_data(i, j);
            text_pos_x = xe(j);
            
            % 统一放在柱子内部
            text_pos_y = val - y_range * 0.03;
            
            text(text_pos_x, text_pos_y, sprintf('%.4f', val), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
                 'FontSize', 9, 'FontWeight', 'bold', 'Color', colors(i, :), ...
                 'BackgroundColor', 'white', 'EdgeColor', colors(i, :), ...
                 'LineWidth', 0.5, 'Margin', 2);
        end
    end
    
    % 保存SSIM图像
    fname = fullfile(save_path_base, sprintf('SSIM_Comparison_%s.tif', image_name_clean));
    try
        print(hfig3, fname, '-dtiff', '-r600');
        fprintf('SSIM图像已保存: %s\n', fname);
    catch ME
        fprintf('保存SSIM图像时出错: %s\n', ME.message);
    end
    
    % 安全关闭图形
    if ishandle(hfig3)
        close(hfig3);
    end
    clear hfig3;
    
    fprintf('所有对比图生成完成！\n');
end



%% 绘制三种算法最佳值对比图（均值垂直错开显示）
function plot_best_comparison(All_psnr_best, All_mse_best, All_ssim_best, algo_labels, image_names, save_path)
    % 输入参数:
    % All_psnr_best - [n×3] 每张图像三种算法的最优PSNR
    % All_mse_best - [n×3] 每张图像三种算法的最优MSE  
    % All_ssim_best - [n×3] 每张图像三种算法的最优SSIM
    % algo_labels - 算法标签 {'BPR', 'PSOBPR', 'IPSOBPR'}
    % image_names - 图像名称数组
    % save_path - 保存路径

    num_images = size(All_psnr_best, 1);
    x = 1:num_images;
    colors = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.4 0.8 0.4]; % 蓝、橙、绿
    line_styles = {'-', '--', '-.'}; % 实线、虚线、点划线
    markers = {'o', 's', '^'}; % 圆形、方形、三角形
    marker_size = 8;
    
    % 计算平均值
    psnr_means = mean(All_psnr_best, 1);
    mse_means = mean(All_mse_best, 1);
    ssim_means = mean(All_ssim_best, 1);
    
    %% 1. PSNR最佳值对比图
    hfig1 = figure('Position', [100, 300, 1200, 500], 'Visible', 'off');
    hold on;
    
    % 绘制每个算法的数据点和连线
    for algo = 1:3
        plot(x, All_psnr_best(:, algo), line_styles{algo}, ...
             'Color', colors(algo, :), 'LineWidth', 1.5, ...
             'Marker', markers{algo}, 'MarkerSize', marker_size, ...
             'MarkerFaceColor', colors(algo, :), 'MarkerEdgeColor', colors(algo, :));
    end
    
    % 设置坐标轴
    xlim([0, num_images + 3]); % 增加更多右侧空间
    y_range = max(All_psnr_best(:)) - min(All_psnr_best(:));
    y_margin = y_range * 0.1;
    ylim([min(All_psnr_best(:)) - y_margin, max(All_psnr_best(:)) + y_margin]);
    
    xlabel('Image Number', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('PSNR (dB)', 'FontSize', 12, 'FontName', 'Times New Roman');
    title('Best PSNR Comparison of Three Algorithms', 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    
    % 设置x轴刻度
    xticks(1:num_images);
    if num_images <= 15
        xticklabels(image_names);
    else
        xticklabels(1:num_images);
    end
    xtickangle(45);
    
    % 添加网格
    grid on;
    grid minor;
    
    % 添加图例（放在图形外部）
    legend(algo_labels, 'Location', 'northeastoutside', 'FontSize', 11, 'FontName', 'Times New Roman');
    
    % 方案1: 垂直错开显示均值
    mean_x_start = num_images + 0.8;
    mean_x_end = num_images + 2.2;
    text_x_pos = num_images + 2.3;
    
    % 计算垂直偏移量
    vertical_spacing = y_range * 0.08;
    
    for algo = 1:3
        % 计算垂直位置（错开显示）
        if algo == 1
            text_y_pos = psnr_means(algo) - vertical_spacing;
        elseif algo == 2
            text_y_pos = psnr_means(algo);
        else
            text_y_pos = psnr_means(algo) + vertical_spacing;
        end
        
        % 绘制从实际均值到显示位置的斜线
        plot([mean_x_start, mean_x_end], [psnr_means(algo), text_y_pos], ...
             ':', 'Color', colors(algo, :), 'LineWidth', 1.5);
        
        % 在错开位置显示文本
        text(text_x_pos, text_y_pos, sprintf('Mean: %.2f', psnr_means(algo)), ...
             'Color', colors(algo, :), 'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'FontName', 'Times New Roman', ...
             'BackgroundColor', 'white', 'EdgeColor', colors(algo, :), 'LineWidth', 1);
    end
    
    hold off;
    
    % 保存PSNR图
    fname_psnr_tif = fullfile(save_path, 'PSNR_Best_Comparison.tif');
    print(hfig1, fname_psnr_tif, '-dtiff', '-r600');
    fname_psnr_eps = fullfile(save_path, 'PSNR_Best_Comparison.eps');
    print(hfig1, fname_psnr_eps, '-depsc');
    
    fprintf('PSNR最佳值对比图已保存 (TIFF 600dpi + EPS)\n');
    if ishandle(hfig1), close(hfig1); end
    clear hfig1;
    
    %% 2. MSE最佳值对比图（类似PSNR的修改）
    hfig2 = figure('Position', [200, 200, 1200, 500], 'Visible', 'off');
    hold on;
    
    for algo = 1:3
        plot(x, All_mse_best(:, algo), line_styles{algo}, ...
             'Color', colors(algo, :), 'LineWidth', 1.5, ...
             'Marker', markers{algo}, 'MarkerSize', marker_size, ...
             'MarkerFaceColor', colors(algo, :), 'MarkerEdgeColor', colors(algo, :));
    end
    
    xlim([0, num_images + 3]);
    y_range_mse = max(All_mse_best(:)) - min(All_mse_best(:));
    y_margin_mse = y_range_mse * 0.1;
    ylim([max(0, min(All_mse_best(:)) - y_margin_mse), max(All_mse_best(:)) + y_margin_mse]);
    
    xlabel('Image Number', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('MSE', 'FontSize', 12, 'FontName', 'Times New Roman');
    title('Best MSE Comparison of Three Algorithms', 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    
    xticks(1:num_images);
    if num_images <= 15
        xticklabels(image_names);
    else
        xticklabels(1:num_images);
    end
    xtickangle(45);
    
    grid on; grid minor;
    legend(algo_labels, 'Location', 'northeastoutside', 'FontSize', 11, 'FontName', 'Times New Roman');
    
    % MSE图的垂直错开显示
    vertical_spacing_mse = y_range_mse * 0.08;
    
    for algo = 1:3
        if algo == 1
            text_y_pos = mse_means(algo) - vertical_spacing_mse;
        elseif algo == 2
            text_y_pos = mse_means(algo);
        else
            text_y_pos = mse_means(algo) + vertical_spacing_mse;
        end
        
        plot([mean_x_start, mean_x_end], [mse_means(algo), text_y_pos], ...
             ':', 'Color', colors(algo, :), 'LineWidth', 1.5);
        
        text(text_x_pos, text_y_pos, sprintf('Mean: %.2f', mse_means(algo)), ...
             'Color', colors(algo, :), 'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'FontName', 'Times New Roman', ...
             'BackgroundColor', 'white', 'EdgeColor', colors(algo, :), 'LineWidth', 1);
    end
    
    hold off;
    
    fname_mse_tif = fullfile(save_path, 'MSE_Best_Comparison.tif');
    print(hfig2, fname_mse_tif, '-dtiff', '-r600');
    fname_mse_eps = fullfile(save_path, 'MSE_Best_Comparison.eps');
    print(hfig2, fname_mse_eps, '-depsc');
    
    fprintf('MSE最佳值对比图已保存 (TIFF 600dpi + EPS)\n');
    if ishandle(hfig2), close(hfig2); end
    clear hfig2;
    
    %% 3. SSIM最佳值对比图（类似修改）
    hfig3 = figure('Position', [300, 100, 1200, 500], 'Visible', 'off');
    hold on;
    
    for algo = 1:3
        plot(x, All_ssim_best(:, algo), line_styles{algo}, ...
             'Color', colors(algo, :), 'LineWidth', 1.5, ...
             'Marker', markers{algo}, 'MarkerSize', marker_size, ...
             'MarkerFaceColor', colors(algo, :), 'MarkerEdgeColor', colors(algo, :));
    end
    
    xlim([0, num_images + 3]);
    y_range_ssim = max(All_ssim_best(:)) - min(All_ssim_best(:));
    y_margin_ssim = y_range_ssim * 0.1;
    ylim([max(0, min(All_ssim_best(:)) - y_margin_ssim), min(1, max(All_ssim_best(:)) + y_margin_ssim)]);
    
    xlabel('Image Number', 'FontSize', 12, 'FontName', 'Times New Roman');
    ylabel('SSIM', 'FontSize', 12, 'FontName', 'Times New Roman');
    title('Best SSIM Comparison of Three Algorithms', 'FontSize', 14, 'FontName', 'Times New Roman', 'FontWeight', 'bold');
    
    xticks(1:num_images);
    if num_images <= 15
        xticklabels(image_names);
    else
        xticklabels(1:num_images);
    end
    xtickangle(45);
    
    grid on; grid minor;
    legend(algo_labels, 'Location', 'northeastoutside', 'FontSize', 11, 'FontName', 'Times New Roman');
    
    % SSIM图的垂直错开显示
    vertical_spacing_ssim = y_range_ssim * 0.08;
    
    for algo = 1:3
        if algo == 1
            text_y_pos = ssim_means(algo) - vertical_spacing_ssim;
        elseif algo == 2
            text_y_pos = ssim_means(algo);
        else
            text_y_pos = ssim_means(algo) + vertical_spacing_ssim;
        end
        
        plot([mean_x_start, mean_x_end], [ssim_means(algo), text_y_pos], ...
             ':', 'Color', colors(algo, :), 'LineWidth', 1.5);
        
        text(text_x_pos, text_y_pos, sprintf('Mean: %.4f', ssim_means(algo)), ...
             'Color', colors(algo, :), 'FontSize', 10, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'left', 'FontName', 'Times New Roman', ...
             'BackgroundColor', 'white', 'EdgeColor', colors(algo, :), 'LineWidth', 1);
    end
    
    hold off;
    
    fname_ssim_tif = fullfile(save_path, 'SSIM_Best_Comparison.tif');
    print(hfig3, fname_ssim_tif, '-dtiff', '-r600');
    fname_ssim_eps = fullfile(save_path, 'SSIM_Best_Comparison.eps');
    print(hfig3, fname_ssim_eps, '-depsc');
    
    fprintf('SSIM最佳值对比图已保存 (TIFF 600dpi + EPS)\n');
    if ishandle(hfig3), close(hfig3); end
    clear hfig3;
    
    fprintf('所有最佳值对比图生成完成！\n');
    
    % 打印均值信息
    fprintf('\n=== 算法均值统计 ===\n');
    fprintf('PSNR均值 - BPR: %.2f, PSOBPR: %.2f, IPSOBPR: %.2f dB\n', psnr_means(1), psnr_means(2), psnr_means(3));
    fprintf('MSE均值  - BPR: %.2f, PSOBPR: %.2f, IPSOBPR: %.2f\n', mse_means(1), mse_means(2), mse_means(3));
    fprintf('SSIM均值 - BPR: %.4f, PSOBPR: %.4f, IPSOBPR: %.4f\n', ssim_means(1), ssim_means(2), ssim_means(3));
end