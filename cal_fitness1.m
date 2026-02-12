function fitness = cal_fitness2_single(x, inputnum, hidden_layers, outputnum, net, inputn, outputn)
    % 单层9节点网络适应度函数
    % 网络结构：输入(9) → 隐藏层(9) → 输出(1)
    % 参数顺序：w1, b1, w2, b2
    
    % 检查是否为单隐层
    if length(hidden_layers) ~= 1
        warning('输入hidden_layers=%d层，强制按单隐层处理', length(hidden_layers));
    end
    hiddennum = hidden_layers(1);  % 取第一个值作为隐层节点数
    
    % 计算各层参数维度
    % 输入层 -> 隐藏层
    w1_size = inputnum * hiddennum;     % 9×9 = 81
    b1_size = hiddennum;               % 9
    
    % 隐藏层 -> 输出层
    w2_size = hiddennum * outputnum;   % 9×1 = 9
    b2_size = outputnum;              % 1
    
    % 计算各参数在向量x中的起始位置
    start_idx = 1;
    
    % 1. 解码输入层->隐藏层的权重和偏置
    w1 = x(start_idx:start_idx + w1_size - 1);
    start_idx = start_idx + w1_size;
    b1 = x(start_idx:start_idx + b1_size - 1);
    start_idx = start_idx + b1_size;
    
    % 2. 解码隐藏层->输出层的权重和偏置
    w2 = x(start_idx:start_idx + w2_size - 1);
    start_idx = start_idx + w2_size;
    b2 = x(start_idx:start_idx + b2_size - 1);
    
    % 验证参数总数
    expected_total = w1_size + b1_size + w2_size + b2_size;  % 81+9+9+1=100
    actual_total = length(x);
    if actual_total ~= expected_total
        error('参数维度不匹配: 期望%d个参数，实际有%d个参数', expected_total, actual_total);
    end
    
    % 赋值权值与阈值到BPNN（单隐层网络）
    % net.IW{1,1}：输入层到隐藏层权重
    % net.LW{2,1}：隐藏层到输出层权重
    % net.b{1}：隐藏层偏置
    % net.b{2}：输出层偏置
    
    % 确保网络是单隐层结构
    if length(net.layers) ~= 2
        error('网络层数不匹配: 期望2层（1个隐藏层+1个输出层），实际有%d层', length(net.layers));
    end
    
    % 设置第一层权重（输入->隐藏层）
    net.IW{1,1} = reshape(w1, hiddennum, inputnum);  % 9×9矩阵
    
    % 设置第二层权重（隐藏层->输出）
    net.LW{2,1} = reshape(w2, outputnum, hiddennum);  % 1×9矩阵
    
    % 设置偏置
    net.b{1} = reshape(b1, hiddennum, 1);  % 9×1
    net.b{2} = b2(:);                      % 1×1
    
    % 计算BPNN预测误差
    pred_out = sim(net, inputn);
    fitness = 0.1 * sum(sum(abs(pred_out - outputn)));
end