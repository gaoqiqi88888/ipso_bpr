function restored_img = IPSOBPR_predict(x, inputn, inputnum, hiddennum, outputnum)
% IPSOBPR_PREDICT 使用优化参数进行图像复原推理
%   输入:
%       x: 优化后的参数向量（前 numsum 个是网络参数）
%       inputn: 输入数据（3×3窗口）
%       inputnum: 输入节点数
%       hiddennum: 隐藏层节点数
%       outputnum: 输出节点数
%   输出:
%       restored_img: 复原图像

    %% 1. 从参数向量中提取权重和阈值
    % 输入层到隐藏层权重 (hiddennum × inputnum)
    w1_size = inputnum * hiddennum;
    w1 = x(1 : w1_size);
    w1 = reshape(w1, hiddennum, inputnum);
    
    % 隐藏层阈值 (hiddennum × 1)
    b1 = x(w1_size + 1 : w1_size + hiddennum);
    b1 = reshape(b1, hiddennum, 1);
    
    % 隐藏层到输出层权重 (outputnum × hiddennum)
    w2_size = hiddennum * outputnum;
    w2 = x(w1_size + hiddennum + 1 : w1_size + hiddennum + w2_size);
    w2 = reshape(w2, outputnum, hiddennum);
    
    % 输出层阈值 (outputnum × 1)
    b2 = x(w1_size + hiddennum + w2_size + 1 : end);
    b2 = reshape(b2, outputnum, 1);
    
    %% 2. 前向传播计算
    % 隐藏层输入
    hidden_input = w1 * inputn' + b1;
    
    % 隐藏层输出（使用 tansig 激活函数）
    hidden_output = tansig(hidden_input);
    
    % 输出层输入
    output_input = w2 * hidden_output + b2;
    
    % 输出层输出（使用 purelin 线性激活函数）
    output = purelin(output_input);
    
    %% 3. 将输出向量重构成图像
    img_size = sqrt(length(output));
    if abs(round(img_size) - img_size) < 1e-10
        img_size = round(img_size);
        restored_img = reshape(output, img_size, img_size)';
    else
        restored_img = reshape(output, 88, 88)';
        fprintf('  警告: 输出维度 %d 不是完美平方\n', length(output));
    end
    
end