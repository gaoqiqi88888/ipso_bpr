% -----------------------------------------------------------------------------------
function [PSNR_BP, MSE_BP, NMSE_BP, SSIM_BP, image_resized, image_blurred, image_restored_noedge] = Sim_Pic_PSO_L2(init, net, inputn, outputn, x, inputnum, hidden_layers, outputnum, picname, picsize)
%% 功能：双层BPNN图像复原与评价（适配PSO/IPSO参数，支持18-9网络结构）
% 输入：init=1表示加载优化参数（PSO/IPSO），init=0表示无优化
%       hidden_layers = [18, 9] 双层隐藏层节点数
% 输出：PSNR/MSE/NMSE/SSIM（均调用已实现的函数）、复原图像等

%% 1. 验证网络层数
if length(hidden_layers) ~= 2
    error('网络结构应为双层隐藏层，当前设置: %s', mat2str(hidden_layers));
end

% 计算各层参数维度
w1_size = inputnum * hidden_layers(1);           % 9×18 = 162
b1_size = hidden_layers(1);                      % 18
w2_size = hidden_layers(1) * hidden_layers(2);   % 18×9 = 162
b2_size = hidden_layers(2);                      % 9
w3_size = hidden_layers(2) * outputnum;          % 9×1 = 9
b3_size = outputnum;                             % 1
total_params = w1_size + b1_size + w2_size + b2_size + w3_size + b3_size;  % 361

%% 2. 加载优化参数（PSO/IPSO参数解码，适配双层网络）
if (init == 1)
    % 验证参数维度
    if length(x) ~= total_params
        error('参数维度不匹配: 期望%d个参数，实际有%d个参数', total_params, length(x));
    end
    
    % 解码粒子为BPNN权值与阈值（双层网络需要三组参数）
    start_idx = 1;
    
    % 2.1 解码输入层->隐藏层1的权重和偏置
    w1 = x(start_idx:start_idx + w1_size - 1);
    start_idx = start_idx + w1_size;
    B1 = x(start_idx:start_idx + b1_size - 1);
    start_idx = start_idx + b1_size;
    
    % 2.2 解码隐藏层1->隐藏层2的权重和偏置
    w2 = x(start_idx:start_idx + w2_size - 1);
    start_idx = start_idx + w2_size;
    B2 = x(start_idx:start_idx + b2_size - 1);
    start_idx = start_idx + b2_size;
    
    % 2.3 解码隐藏层2->输出层的权重和偏置
    w3 = x(start_idx:start_idx + w3_size - 1);
    start_idx = start_idx + w3_size;
    B3 = x(start_idx:start_idx + b3_size - 1);
    
    % 2.4 赋值到BPNN（双层网络结构）
    % 判断网络类型并相应赋值
    try
        % 尝试使用newff的赋值方式
        net.iw{1,1} = reshape(w1, hidden_layers(1), inputnum);       % 18×9矩阵
        net.lw{2,1} = reshape(w2, hidden_layers(2), hidden_layers(1)); % 9×18矩阵
        net.lw{3,2} = reshape(w3, outputnum, hidden_layers(2));      % 1×9矩阵
        net.b{1} = reshape(B1, hidden_layers(1), 1);                 % 18×1
        net.b{2} = reshape(B2, hidden_layers(2), 1);                 % 9×1
        net.b{3} = B3;                                               % 1×1
    catch
        % 如果失败，尝试feedforwardnet的赋值方式
        try
            net.IW{1} = reshape(w1, hidden_layers(1), inputnum);
            net.LW{2,1} = reshape(w2, hidden_layers(2), hidden_layers(1));
            net.LW{3,2} = reshape(w3, outputnum, hidden_layers(2));
            net.b{1} = reshape(B1, hidden_layers(1), 1);
            net.b{2} = reshape(B2, hidden_layers(2), 1);
            net.b{3} = B3;
        catch ME
            error('网络权重赋值失败: %s\n请确保使用正确的网络创建方式', ME.message);
        end
    end
    
    fprintf('双层网络参数加载成功: w1(%d×%d), w2(%d×%d), w3(%d×%d)\n', ...
            hidden_layers(1), inputnum, ...
            hidden_layers(2), hidden_layers(1), ...
            outputnum, hidden_layers(2));
end

%% 3. BPNN训练（按老论文参数）
if init == 0
    % BPR算法：无优化参数，需要训练网络
    net.trainParam.epochs = 1000;    % 训练次数
    net.trainParam.lr = 0.1;         % 学习率
    net.trainParam.goal = 1e-5;      % 训练目标误差
    net.trainParam.showWindow = false;  % 关闭训练窗口
    net.trainParam.showCommandLine = false;
    
    fprintf('BPR算法: 开始训练双层BPNN网络...\n');
    net = train(net, inputn, outputn);  % 训练网络
    fprintf('BPR算法: 网络训练完成\n');
else
    % PSO/IPSO算法：已加载优化参数，不需要训练
    % 可以设置小规模训练或不训练
    net.trainParam.epochs = 1;       % 最小训练次数
    net.trainParam.lr = 0.01;        % 小学习率
    net.trainParam.goal = 1e-3;      % 宽松目标
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
    % 可选：微调网络
    % net = train(net, inputn, outputn);
end

%% 4. 图像复原（调用Read_Pic_PSO获取当前图像数据）
[P_Matrix, ~, image_resized, image_blurred] = Read_Pic_PSO(picname, picsize);
Y = sim(net, P_Matrix);  % 网络预测：退化图像→清晰图像像素

%% 5. 复原图像重构（去除边缘）
[H, W] = size(image_resized);
image_restored_noedge = zeros(H-2, W-2);
t = 1;
for i = 1:H-2
    for j = 1:W-2
        % 约束像素值在[0,1]范围内
        pixel_value = Y(1,t);
        image_restored_noedge(i,j) = max(min(pixel_value, 1), 0);
        t = t + 1;
    end
end

%% 6. 评价指标计算
% 6.1 预处理：裁剪原始图/退化图边缘
image_resized_noedge = image_resized(2:end-1, 2:end-1);  % 原始图去除边缘
image_blurred_noedge = image_blurred(2:end-1, 2:end-1);  % 退化图去除边缘

% 6.2 转换为8位灰度值（0-255范围）
img_org = uint8(image_resized_noedge * 255);
img_rec = uint8(image_restored_noedge * 255);
img_blur = uint8(image_blurred_noedge * 255);

% 6.3 计算指标
try
    PSNR_BP = Cal_PSNR(img_org, img_rec);      % PSNR函数
    MSE_BP = Cal_MSE(img_org, img_rec);        % MSE函数
    NMSE_BP = nmse(img_org, img_blur, img_rec); % NMSE函数
    SSIM_BP = Cal_SSIM(img_org, img_rec);      % SSIM函数
    
    % 输出调试信息
    fprintf('图像复原指标: PSNR=%.2f dB, MSE=%.2f, SSIM=%.4f\n', ...
            PSNR_BP, MSE_BP, SSIM_BP);
catch ME
    fprintf('指标计算错误: %s\n', ME.message);
    PSNR_BP = 0;
    MSE_BP = 0;
    NMSE_BP = 0;
    SSIM_BP = 0;
end
end