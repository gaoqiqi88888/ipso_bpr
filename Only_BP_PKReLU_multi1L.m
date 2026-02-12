%% 激活函数对比实验版本 - 使用Satlin近似Leaky ReLU，未在论文全面展开

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
num_repeats = 10;
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

%% 6. 结果可视化与比较
% 6.1 性能指标对比图
figure('Name', '不同激活函数性能对比', 'Position', [100, 100, 1400, 800]);

% 子图1: PSNR对比（带误差棒）
subplot(2, 3, 1);
errorbar(1:num_configs, results.psnr_mean, results.psnr_std, 'o-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
yline(PSNR_blurred, 'r--', 'LineWidth', 2, 'Label', '模糊图像PSNR');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels(config_names);
xtickangle(45);
ylabel('PSNR (dB)');
title('PSNR对比 (越高越好)');
grid on;
hold off;

% 子图2: MSE对比
subplot(2, 3, 2);
errorbar(1:num_configs, results.mse_mean, results.mse_std, 's-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
hold on;
yline(MSE_blurred, 'r--', 'LineWidth', 2, 'Label', '模糊图像MSE');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels(config_names);
xtickangle(45);
ylabel('MSE (越低越好)');
title('MSE对比');
grid on;
hold off;

% 子图3: SSIM对比
subplot(2, 3, 3);
errorbar(1:num_configs, results.ssim_mean, results.ssim_std, 'd-', ...
    'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'g');
hold on;
yline(SSIM_blurred, 'r--', 'LineWidth', 2, 'Label', '模糊图像SSIM');
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels(config_names);
xtickangle(45);
ylabel('SSIM (越高越好)');
ylim([0, 1]);
title('SSIM对比');
grid on;
hold off;

% 子图4: 训练时间对比
subplot(2, 3, 4);
bar(1:num_configs, results.time_mean, 'FaceColor', [0.8, 0.6, 0.2]);
hold on;
errorbar(1:num_configs, results.time_mean, results.time_std, 'k.', 'LineWidth', 1.5);
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels(config_names);
xtickangle(45);
ylabel('训练时间 (秒)');
title('训练时间对比');
grid on;
hold off;

% 子图5: 收敛速度对比（收敛所需epochs）
subplot(2, 3, 5);
convergence_mean = mean(results.convergence_epochs, 2);
convergence_std = std(results.convergence_epochs, 0, 2);
bar(1:num_configs, convergence_mean, 'FaceColor', [0.2, 0.6, 0.8]);
hold on;
errorbar(1:num_configs, convergence_mean, convergence_std, 'k.', 'LineWidth', 1.5);
xlim([0.5, num_configs+0.5]);
xticks(1:num_configs);
xticklabels(config_names);
xtickangle(45);
ylabel('收敛所需迭代次数');
title('收敛速度对比 (越低越快)');
grid on;
hold off;

% 子图6: 综合评分雷达图
subplot(2, 3, 6);
% 归一化各项指标
psnr_norm = (results.psnr_mean - min(results.psnr_mean)) / ...
    (max(results.psnr_mean) - min(results.psnr_mean));
mse_norm = 1 - (results.mse_mean - min(results.mse_mean)) / ...
    (max(results.mse_mean) - min(results.mse_mean));  % MSE越低越好
ssim_norm = (results.ssim_mean - min(results.ssim_mean)) / ...
    (max(results.ssim_mean) - min(results.ssim_mean));
time_norm = 1 - (results.time_mean - min(results.time_mean)) / ...
    (max(results.time_mean) - min(results.time_mean));  % 时间越短越好

% 创建雷达图
theta = linspace(0, 2*pi, 5);
theta = [theta, theta(1)];  % 闭合图形

for i = 1:num_configs
    radar_data = [psnr_norm(i), mse_norm(i), ssim_norm(i), time_norm(i), convergence_mean(i)/100];
    radar_data = [radar_data, radar_data(1)];  % 闭合
    polarplot(theta, radar_data, 'LineWidth', 2);
    hold on;
end

legend(config_names, 'Location', 'southoutside', 'NumColumns', 2);
title('综合性能雷达图 (归一化)');
rlim([0, 1]);
hold off;

sgtitle('不同激活函数BPNN性能对比实验', 'FontSize', 16, 'FontWeight', 'bold');

%% 7. 最佳配置的详细结果展示
% 找出最佳PSNR配置
[best_psnr, best_psnr_idx] = max(results.psnr_mean);
best_config_name = config_names{best_psnr_idx};

fprintf('\n========================================================\n');
fprintf('最佳配置分析: %s\n', best_config_name);
fprintf('========================================================\n');
fprintf('平均PSNR: %.2f ± %.2f dB\n', results.psnr_mean(best_psnr_idx), results.psnr_std(best_psnr_idx));
fprintf('平均MSE:  %.2f ± %.2f\n', results.mse_mean(best_psnr_idx), results.mse_std(best_psnr_idx));
fprintf('平均SSIM: %.4f ± %.4f\n', results.ssim_mean(best_psnr_idx), results.ssim_std(best_psnr_idx));
fprintf('平均训练时间: %.2f ± %.2f 秒\n', results.time_mean(best_psnr_idx), results.time_std(best_psnr_idx));
fprintf('相对于原论文配置(tansig-purelin)的提升:\n');

% 与原论文配置对比
baseline_idx = 1;  % 原论文配置
psnr_improvement = results.psnr_mean(best_psnr_idx) - results.psnr_mean(baseline_idx);
mse_improvement = (results.mse_mean(baseline_idx) - results.mse_mean(best_psnr_idx)) / results.mse_mean(baseline_idx) * 100;
time_ratio = results.time_mean(best_psnr_idx) / results.time_mean(baseline_idx);

fprintf('  PSNR提升: %.2f dB\n', psnr_improvement);
fprintf('  MSE降低: %.1f%%\n', mse_improvement);
fprintf('  训练时间比率: %.2fx\n', time_ratio);

% 显示最佳配置的复原图像（使用第一次重复的结果）
figure('Name', sprintf('最佳配置复原结果: %s', best_config_name), ...
    'Position', [100, 100, 1200, 300]);

best_restored = restored_images{best_psnr_idx}(:,:,1);

subplot(1,4,1);
imshow(uint8(image_resized*255));
title('原始清晰图像');

subplot(1,4,2);
imshow(uint8(image_blurred*255));
title('高斯模糊图像');

subplot(1,4,3);
imshow(uint8(best_restored*255));
title(sprintf('%s复原结果', best_config_name));

% 误差图
subplot(1,4,4);
error_map = abs(double(image_resized(2:end-1, 2:end-1)*255) - double(best_restored*255));
imagesc(error_map);
colorbar;
colormap('hot');
title('复原误差图');
xlabel(sprintf('平均误差: %.2f', mean(error_map(:))));

%% 8. 统计显著性检验
fprintf('\n========================================================\n');
fprintf('统计显著性检验 (t检验, α=0.05)\n');
fprintf('========================================================\n');

for i = 2:num_configs  % 与原论文配置对比
    % PSNR显著性检验
    [h_psnr, p_psnr] = ttest2(results.psnr_bpnn(1,:), results.psnr_bpnn(i,:));
    
    % MSE显著性检验  
    [h_mse, p_mse] = ttest2(results.mse_bpnn(1,:), results.mse_bpnn(i,:));
    
    significance_psnr = '';
    if h_psnr == 1
        if results.psnr_mean(i) > results.psnr_mean(1)
            significance_psnr = '显著提升';
        else
            significance_psnr = '显著下降';
        end
    else
        significance_psnr = '无显著差异';
    end
    
    fprintf('\n%s vs %s:\n', config_names{1}, config_names{i});
    fprintf('  PSNR: p=%.4f, %s\n', p_psnr, significance_psnr);
    fprintf('  MSE:  p=%.4f\n', p_mse);
end

%% 9. 保存所有结果
save('activation_comparison_results.mat', 'results', 'restored_images', ...
    'config_names', 'PSNR_blurred', 'MSE_blurred', 'SSIM_blurred', ...
    'image_resized', 'image_blurred');

% 保存最佳配置的网络
if exist('net_trained', 'var')
    save('best_network.mat', 'net_trained');
end

fprintf('\n========================================================\n');
fprintf('实验完成！\n');
fprintf('所有结果已保存到: activation_comparison_results.mat\n');
fprintf('========================================================\n');