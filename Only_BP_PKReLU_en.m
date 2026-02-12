%% 激活函数对比实验版本 - 使用Satlin近似Leaky ReLU

%% 清空环境变量
clear
close all
clc

%% 1. 实验设置
inputnum = 9;      % 输入节点
hiddennum = 9;     % 隐藏层节点
outputnum = 1;     % 输出节点

% 定义要对比的激活函数组合
activation_configs = {
    {'tansig', 'purelin'},     % 原论文配置（对照）
    {'logsig', 'purelin'},     % 替代sigmoid函数
    {'poslin', 'purelin'},     % ReLU (Matlab中叫poslin)
    {'satlin', 'purelin'},     % 饱和线性函数（近似Leaky ReLU）
    {'tansig', 'tansig'}       % 输出层也用tansig（非常规）
};

config_names = {
    'Tansig-Purelin (原论文)',
    'Logsig-Purelin',
    'ReLU-Purelin',
    'Satlin-Purelin (近似Leaky)',
    'Tansig-Tansig'
};

% 实验重复次数（减少随机性影响）
num_repeats = 3;
num_configs = length(activation_configs);

%% 2. 图像读取与预处理
image_origin = imread('img/Lenna.tif'); 
if size(image_origin,3) == 3
    image_origin = rgb2gray(image_origin);
end

image_resized = imresize(double(image_origin),[90,90]); 
image_resized = image_resized./256; 

% 高斯模糊退化处理
w_gauss = fspecial('gaussian', [9, 9], 1);
image_blurred = imfilter(image_resized, w_gauss, 'replicate');

%% 3. 构建训练数据
[H, W] = size(image_resized);
sample_num = (H-2)*(W-2);

P_Matrix = zeros(inputnum, sample_num, 'double');
T_Matrix = zeros(outputnum, sample_num, 'double');

t = 1;
for i = 2:H-1
    for j = 2:W-1
        P_Matrix(1,t) = image_blurred(i-1,j-1);
        P_Matrix(2,t) = image_blurred(i-1,j);
        P_Matrix(3,t) = image_blurred(i-1,j+1);
        P_Matrix(4,t) = image_blurred(i,j-1);
        P_Matrix(5,t) = image_blurred(i,j);
        P_Matrix(6,t) = image_blurred(i,j+1);
        P_Matrix(7,t) = image_blurred(i+1,j-1);
        P_Matrix(8,t) = image_blurred(i+1,j);
        P_Matrix(9,t) = image_blurred(i+1,j+1);
        
        T_Matrix(1,t) = image_resized(i,j);
        t = t + 1;
    end
end

inputn = P_Matrix;
outputn = T_Matrix;

%% 4. 不同激活函数网络训练与评估
% 存储结果
results = struct();
results.config_names = config_names;
results.psnr_blurred = zeros(num_configs, num_repeats);
results.psnr_bpnn = zeros(num_configs, num_repeats);
results.mse_bpnn = zeros(num_configs, num_repeats);
results.ssim_bpnn = zeros(num_configs, num_repeats);
results.training_time = zeros(num_configs, num_repeats);
results.convergence_epochs = zeros(num_configs, num_repeats);

% 为每种配置准备图像复原结果
restored_images = cell(num_configs, 1);

for config_idx = 1:num_configs
    fprintf('\n========================================================\n');
    fprintf('实验配置 %d/%d: %s\n', config_idx, num_configs, config_names{config_idx});
    fprintf('========================================================\n');
    
    % 准备复原图像存储
    restored_images{config_idx} = zeros(H-2, W-2, num_repeats, 'double');
    
    for repeat_idx = 1:num_repeats
        fprintf('\n--- 重复实验 %d/%d ---\n', repeat_idx, num_repeats);
        
        % 4.1 创建网络（根据配置选择激活函数）
        act_funcs = activation_configs{config_idx};
        
        % 使用newff创建标准网络
        net = newff(inputn, outputn, hiddennum, act_funcs, 'trainlm');
        
        % 4.2 设置训练参数（统一设置）
        net.trainParam.epochs = 100;
        net.trainParam.lr = 0.1;
        net.trainParam.goal = 1e-5;
        net.trainParam.showWindow = 0;  % 不显示训练窗口，加快速度
        net.trainParam.show = NaN;      % 不显示中间结果
        
        % 4.3 训练网络并计时
        fprintf('开始训练... ');
        tic;
        net_trained = train(net, inputn, outputn);
        training_time = toc;
        
        results.training_time(config_idx, repeat_idx) = training_time;
        
        % 记录收敛情况
        if isfield(net_trained, 'trainRecord')
            tr = net_trained.trainRecord;
            results.convergence_epochs(config_idx, repeat_idx) = length(tr.perf);
        end
        
        fprintf('训练完成! 用时: %.2f秒\n', training_time);
        
        % 4.4 图像复原
        Y = sim(net_trained, P_Matrix);
        
        % 重组复原图像
        image_restored_temp = zeros(H-2, W-2, 'double');
        t = 1;
        for i = 1:H-2
            for j = 1:W-2
                pred_value = Y(1,t);
                image_restored_temp(i,j) = max(min(pred_value, 1), 0);
                t = t + 1;
            end
        end
        
        restored_images{config_idx}(:,:,repeat_idx) = image_restored_temp;
        
        % 4.5 计算评价指标
        image_resized_noedge = image_resized(2:end-1, 2:end-1);
        image_resized_noedge_255 = image_resized_noedge * 255;
        image_restored_noedge_255 = image_restored_temp * 255;
        
        % MSE
        mse_temp = mean((image_resized_noedge_255(:) - image_restored_noedge_255(:)).^2);
        results.mse_bpnn(config_idx, repeat_idx) = mse_temp;
        
        % PSNR
        if mse_temp < eps, mse_temp = eps; end
        psnr_temp = 10 * log10(255^2 / mse_temp);
        results.psnr_bpnn(config_idx, repeat_idx) = psnr_temp;
        
        % SSIM
        ssim_temp = ssim(uint8(image_restored_noedge_255), uint8(image_resized_noedge_255));
        results.ssim_bpnn(config_idx, repeat_idx) = max(0, min(1, ssim_temp));
        
        fprintf('结果: PSNR=%.2f dB, MSE=%.2f, SSIM=%.4f\n', ...
            psnr_temp, mse_temp, ssim_temp);
    end
    
    % 4.6 计算该配置的平均结果
    results.psnr_mean(config_idx) = mean(results.psnr_bpnn(config_idx, :));
    results.psnr_std(config_idx) = std(results.psnr_bpnn(config_idx, :));
    results.mse_mean(config_idx) = mean(results.mse_bpnn(config_idx, :));
    results.mse_std(config_idx) = std(results.mse_bpnn(config_idx, :));
    results.ssim_mean(config_idx) = mean(results.ssim_bpnn(config_idx, :));
    results.ssim_std(config_idx) = std(results.ssim_bpnn(config_idx, :));
    results.time_mean(config_idx) = mean(results.training_time(config_idx, :));
    results.time_std(config_idx) = std(results.training_time(config_idx, :));
    
    fprintf('\n%s 平均结果:\n', config_names{config_idx});
    fprintf('  PSNR: %.2f ± %.2f dB\n', results.psnr_mean(config_idx), results.psnr_std(config_idx));
    fprintf('  MSE:  %.2f ± %.2f\n', results.mse_mean(config_idx), results.mse_std(config_idx));
    fprintf('  SSIM: %.4f ± %.4f\n', results.ssim_mean(config_idx), results.ssim_std(config_idx));
    fprintf('  训练时间: %.2f ± %.2f 秒\n', results.time_mean(config_idx), results.time_std(config_idx));
end

%% 5. 计算退化图像的基准指标
image_blurred_noedge = image_blurred(2:end-1, 2:end-1);
image_blurred_noedge_255 = image_blurred_noedge * 255;
image_resized_noedge_255 = image_resized(2:end-1, 2:end-1) * 255;

MSE_blurred = mean((image_resized_noedge_255(:) - image_blurred_noedge_255(:)).^2);
PSNR_blurred = 10 * log10(255^2 / MSE_blurred);
SSIM_blurred = ssim(uint8(image_blurred_noedge_255), uint8(image_resized_noedge_255));

fprintf('\n========================================================\n');
fprintf('基准指标（高斯模糊图像）:\n');
fprintf('  PSNR: %.2f dB\n', PSNR_blurred);
fprintf('  MSE:  %.2f\n', MSE_blurred);
fprintf('  SSIM: %.4f\n', SSIM_blurred);
fprintf('========================================================\n');

%% 修改后的结果可视化部分（英文版）

% 6. 结果可视化与比较
figure('Name', 'Performance Comparison of Different Activation Functions', ...
    'Position', [100, 100, 1600, 900]);

% 子图1: PSNR对比（带误差棒）
subplot(2, 3, 1);
errorbar(1:num_configs, results.psnr_mean, results.psnr_std, 'o-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b', 'CapSize', 10);
hold on;
yline(PSNR_blurred, 'r--', 'LineWidth', 2, 'Label', 'Blurred Image PSNR');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels({'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'});
xtickangle(30);
ylabel('PSNR (dB)', 'FontSize', 12, 'FontWeight', 'bold');
title('PSNR Comparison (Higher is Better)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
hold off;

% 子图2: MSE对比
subplot(2, 3, 2);
errorbar(1:num_configs, results.mse_mean, results.mse_std, 's-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'CapSize', 10);
hold on;
yline(MSE_blurred, 'r--', 'LineWidth', 2, 'Label', 'Blurred Image MSE');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels({'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'});
xtickangle(30);
ylabel('MSE (Lower is Better)', 'FontSize', 12, 'FontWeight', 'bold');
title('MSE Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
hold off;

% 子图3: SSIM对比 - 调整y轴范围从0.85开始
subplot(2, 3, 3);
errorbar(1:num_configs, results.ssim_mean, results.ssim_std, 'd-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g', 'CapSize', 10);
hold on;
yline(SSIM_blurred, 'r--', 'LineWidth', 2, 'Label', 'Blurred Image SSIM');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels({'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'});
xtickangle(30);
ylabel('SSIM (Higher is Better)', 'FontSize', 12, 'FontWeight', 'bold');
title('SSIM Comparison', 'FontSize', 14, 'FontWeight', 'bold');
% 设置y轴范围从0.85开始
ylim([0.85, 1.0]);
grid on;
box on;
hold off;

% 子图4: 训练时间对比
subplot(2, 3, 4);
bar_colors = [0.2 0.6 0.8; 0.8 0.2 0.2; 0.2 0.8 0.2; 0.8 0.6 0.2; 0.6 0.2 0.8];
bars = bar(1:num_configs, results.time_mean, 'FaceColor', 'flat');
for i = 1:num_configs
    bars.CData(i,:) = bar_colors(i,:);
end
hold on;
errorbar(1:num_configs, results.time_mean, results.time_std, 'k.', 'LineWidth', 1.5);
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels({'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'});
xtickangle(30);
ylabel('Training Time (seconds)', 'FontSize', 12, 'FontWeight', 'bold');
title('Training Time Comparison', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
hold off;

% 子图5: 改为显示训练效率（时间/质量比）
subplot(2, 3, 5);

% 计算效率指标：PSNR per second
efficiency = results.psnr_mean ./ results.time_mean;
efficiency_std = results.psnr_std ./ results.time_mean; % 近似

bars2 = bar(1:num_configs, efficiency, 'FaceColor', 'flat');
for i = 1:num_configs
    bars2.CData(i,:) = bar_colors(i,:);
end
hold on;
errorbar(1:num_configs, efficiency, efficiency_std, 'k.', 'LineWidth', 1.5);
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels({'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'});
xtickangle(30);
ylabel('PSNR per Second (dB/s)', 'FontSize', 12, 'FontWeight', 'bold');
title('Training Efficiency (Higher is Better)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
hold off;

% 子图6: 改进的雷达图
subplot(2, 3, 6);

% 计算效率指标（与子图5保持一致）
efficiency = results.psnr_mean ./ results.time_mean;

% 归一化各项指标
psnr_norm = (results.psnr_mean - min(results.psnr_mean)) / ...
    (max(results.psnr_mean) - min(results.psnr_mean));
mse_norm = 1 - (results.mse_mean - min(results.mse_mean)) / ...
    (max(results.mse_mean) - min(results.mse_mean));  % MSE越低越好
ssim_norm = (results.ssim_mean - min(results.ssim_mean)) / ...
    (max(results.ssim_mean) - min(results.ssim_mean));
time_norm = 1 - (results.time_mean - min(results.time_mean)) / ...
    (max(results.time_mean) - min(results.time_mean));  % 时间越短越好

% 效率归一化（值越高越好）
efficiency_norm = (efficiency - min(efficiency)) / ...
    (max(efficiency) - min(efficiency));

% 创建雷达图
theta = linspace(0, 2*pi, 6);
theta = theta(1:end-1); % 5个顶点
theta = [theta, theta(1)];  % 闭合图形

% 准备数据 - 将Convergence改为Efficiency
radar_labels = {'PSNR', 'MSE', 'SSIM', 'Time', 'Efficiency'};
legend_labels = {'Tansig-Purelin', 'Logsig-Purelin', 'ReLU-Purelin', 'Satlin-Purelin', 'Tansig-Tansig'};

% 创建极坐标图
ax = polaraxes;
% 立即调整位置
ax.Position = [0.68, 0.1, 0.22, 0.35];
ax.ThetaZeroLocation = 'top';
ax.ThetaDir = 'counterclockwise';
hold on;

for i = 1:num_configs
    % 使用efficiency_norm替换convergence_norm
    radar_data = [psnr_norm(i), mse_norm(i), ssim_norm(i), time_norm(i), efficiency_norm(i)];
    radar_data = [radar_data, radar_data(1)];  % 闭合
    
    polarplot(theta, radar_data, ...
        'Color', bar_colors(i,:), ...
        'LineWidth', 2.5, ...
        'Marker', 'o', ...
        'MarkerSize', 6, ...
        'MarkerFaceColor', bar_colors(i,:), ...
        'DisplayName', legend_labels{i});
end

% 设置标签
ax.ThetaTick = linspace(0, 360, 6);
ax.ThetaTickLabel = radar_labels;
ax.RLim = [0, 1.1];
ax.RTick = 0:0.2:1;
ax.FontSize = 10;
ax.GridAlpha = 0.3;
ax.LineWidth = 1;

legend('Location', 'southoutside', 'NumColumns', 2, 'FontSize', 10, 'Box', 'off');
title('Radar Chart of Comprehensive Performance', 'FontSize', 14, 'FontWeight', 'bold');
hold off;

sgtitle('Performance Comparison of BPNN with Different Activation Functions', ...
    'FontSize', 16, 'FontWeight', 'bold');

%% 7. 最佳配置的详细结果展示（英文版）
[best_psnr, best_psnr_idx] = max(results.psnr_mean);
best_config_name = config_names{best_psnr_idx};

% 英文输出
fprintf('\n========================================================\n');
fprintf('Best Configuration Analysis: %s\n', best_config_name);
fprintf('========================================================\n');
fprintf('Average PSNR: %.2f ± %.2f dB\n', results.psnr_mean(best_psnr_idx), results.psnr_std(best_psnr_idx));
fprintf('Average MSE:  %.2f ± %.2f\n', results.mse_mean(best_psnr_idx), results.mse_std(best_psnr_idx));
fprintf('Average SSIM: %.4f ± %.4f\n', results.ssim_mean(best_psnr_idx), results.ssim_std(best_psnr_idx));
fprintf('Average Training Time: %.2f ± %.2f seconds\n', results.time_mean(best_psnr_idx), results.time_std(best_psnr_idx));
fprintf('Improvement compared to original configuration (tansig-purelin):\n');

baseline_idx = 1;
psnr_improvement = results.psnr_mean(best_psnr_idx) - results.psnr_mean(baseline_idx);
mse_improvement = (results.mse_mean(baseline_idx) - results.mse_mean(best_psnr_idx)) / results.mse_mean(baseline_idx) * 100;
time_ratio = results.time_mean(best_psnr_idx) / results.time_mean(baseline_idx);

fprintf('  PSNR Improvement: %.2f dB\n', psnr_improvement);
fprintf('  MSE Reduction: %.1f%%\n', mse_improvement);
fprintf('  Training Time Ratio: %.2fx\n', time_ratio);

% 英文图像显示
figure('Name', sprintf('Restoration Results: %s', best_config_name), ...
    'Position', [100, 100, 1400, 300]);

best_restored = restored_images{best_psnr_idx}(:,:,1);

subplot(1,4,1);
imshow(uint8(image_resized*255));
title('Original Image', 'FontSize', 12, 'FontWeight', 'bold');

subplot(1,4,2);
imshow(uint8(image_blurred*255));
title('Gaussian Blurred Image', 'FontSize', 12, 'FontWeight', 'bold');

subplot(1,4,3);
imshow(uint8(best_restored*255));
title(sprintf('%s Restoration', best_config_name), 'FontSize', 12, 'FontWeight', 'bold');

% 误差图（英文）
subplot(1,4,4);
error_map = abs(double(image_resized(2:end-1, 2:end-1)*255) - double(best_restored*255));
imagesc(error_map);
colorbar;
colormap('hot');
title('Restoration Error Map', 'FontSize', 12, 'FontWeight', 'bold');
xlabel(sprintf('Mean Error: %.2f', mean(error_map(:))), 'FontSize', 10);

%% 8. 统计显著性检验（英文输出）
fprintf('\n========================================================\n');
fprintf('Statistical Significance Test (t-test, α=0.05)\n');
fprintf('========================================================\n');

for i = 2:num_configs
    [h_psnr, p_psnr] = ttest2(results.psnr_bpnn(1,:), results.psnr_bpnn(i,:));
    [h_mse, p_mse] = ttest2(results.mse_bpnn(1,:), results.mse_bpnn(i,:));
    
    significance_psnr = '';
    if h_psnr == 1
        if results.psnr_mean(i) > results.psnr_mean(1)
            significance_psnr = 'Significant Improvement';
        else
            significance_psnr = 'Significant Decline';
        end
    else
        significance_psnr = 'No Significant Difference';
    end
    
    fprintf('\n%s vs %s:\n', config_names{1}, config_names{i});
    fprintf('  PSNR: p=%.4f, %s\n', p_psnr, significance_psnr);
    fprintf('  MSE:  p=%.4f\n', p_mse);
end