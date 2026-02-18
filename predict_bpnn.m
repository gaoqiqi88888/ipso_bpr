function output = predict_bpnn(x, input_data, inputnum, hiddennum, outputnum)
% PREDICT_BPNN 使用优化参数进行BPNN前向传播
%   输入:
%       x: 优化后的参数向量（包含所有权重和阈值）
%       input_data: 输入数据（每个样本为一列）
%       inputnum: 输入节点数
%       hiddennum: 隐藏层节点数
%       outputnum: 输出节点数
%   输出:
%       output: 网络输出

    %% 1. 从参数向量中解析权重和阈值
    % 输入层到隐藏层权重 (hiddennum × inputnum)
    w1_size = inputnum * hiddennum;
    w1 = x(1 : w1_size);
    w1 = reshape(w1, hiddennum, inputnum);
    
    % 隐藏层阈值 (hiddennum × 1)
    b1 = x(w1_size + 1 : w1_size + hiddennum);
    b1 = b1(:);  % 确保是列向量
    
    % 隐藏层到输出层权重 (outputnum × hiddennum)
    w2_size = hiddennum * outputnum;
    w2 = x(w1_size + hiddennum + 1 : w1_size + hiddennum + w2_size);
    w2 = reshape(w2, outputnum, hiddennum);
    
    % 输出层阈值 (outputnum × 1)
    b2 = x(w1_size + hiddennum + w2_size + 1 : end);
    b2 = b2(:);  % 确保是列向量
    
    %% 2. 前向传播计算
    % 隐藏层输入
    hidden_input = w1 * input_data + b1;
    
    % 隐藏层输出（使用 tansig 激活函数）
    hidden_output = tansig(hidden_input);
    
    % 输出层输入
    output_input = w2 * hidden_output + b2;
    
    % 输出层输出（使用 purelin 线性激活函数）
    output = purelin(output_input);
    
end