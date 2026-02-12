function fitness = cal_fitness2(x, inputnum, hidden_layers, outputnum, net, inputn, outputn)
    % 双层18-9网络适应度函数
    % 网络结构：输入(9) → 隐藏层1(18) → 隐藏层2(9) → 输出(1)
    % 参数顺序：w1, b1, w2, b2, w3, b3
    
    % 计算各层参数维度
    % 输入层 -> 隐藏层1
    w1_size = inputnum * hidden_layers(1);           % 9×18 = 162
    b1_size = hidden_layers(1);                      % 18
    
    % 隐藏层1 -> 隐藏层2  
    w2_size = hidden_layers(1) * hidden_layers(2);   % 18×9 = 162
    b2_size = hidden_layers(2);                      % 9
    
    % 隐藏层2 -> 输出层
    w3_size = hidden_layers(2) * outputnum;          % 9×1 = 9
    b3_size = outputnum;                             % 1
    
    % 计算各参数在向量x中的起始位置
    start_idx = 1;
    
    % 1. 解码输入层->隐藏层1的权重和偏置
    w1 = x(start_idx:start_idx + w1_size - 1);
    start_idx = start_idx + w1_size;
    b1 = x(start_idx:start_idx + b1_size - 1);
    start_idx = start_idx + b1_size;
    
    % 2. 解码隐藏层1->隐藏层2的权重和偏置
    w2 = x(start_idx:start_idx + w2_size - 1);
    start_idx = start_idx + w2_size;
    b2 = x(start_idx:start_idx + b2_size - 1);
    start_idx = start_idx + b2_size;
    
    % 3. 解码隐藏层2->输出层的权重和偏置
    w3 = x(start_idx:start_idx + w3_size - 1);
    start_idx = start_idx + w3_size;
    b3 = x(start_idx:start_idx + b3_size - 1);
    
    % 验证参数总数
    expected_total = w1_size + b1_size + w2_size + b2_size + w3_size + b3_size;  % 162+18+162+9+9+1=361
    actual_total = length(x);
    if actual_total ~= expected_total
        error('参数维度不匹配: 期望%d个参数，实际有%d个参数', expected_total, actual_total);
    end
    
    % 赋值权值与阈值到BPNN（双层网络需要设置3个权重矩阵）
    % 注意：feedforwardnet的权重存储方式：
    % net.IW{1,1}：输入层到第1隐藏层权重
    % net.LW{2,1}：第1隐藏层到第2隐藏层权重  
    % net.LW{3,2}：第2隐藏层到输出层权重
    % net.b{1}：第1隐藏层偏置
    % net.b{2}：第2隐藏层偏置
    % net.b{3}：输出层偏置
    
    % 确保网络有足够的层
    if length(net.layers) ~= 3  % 2个隐藏层 + 1个输出层
        error('网络层数不匹配: 期望3层（2个隐藏层+1个输出层），实际有%d层', length(net.layers));
    end
    
    % 设置第一层权重（输入->隐藏层1）
    net.IW{1,1} = reshape(w1, hidden_layers(1), inputnum);  % 18×9矩阵
    
    % 设置第二层权重（隐藏层1->隐藏层2）
    net.LW{2,1} = reshape(w2, hidden_layers(2), hidden_layers(1));  % 9×18矩阵
    
    % 设置第三层权重（隐藏层2->输出）
    net.LW{3,2} = reshape(w3, outputnum, hidden_layers(2));  % 1×9矩阵
    
    % 设置偏置
    net.b{1} = reshape(b1, hidden_layers(1), 1);  % 18×1
    net.b{2} = reshape(b2, hidden_layers(2), 1);  % 9×1
    net.b{3} = b3;                                % 1×1
    
    % 计算BPNN预测误差（适应度=归一化总误差，越小越好）
    % 与原代码误差计算方式一致，保证对比公平性
    pred_out = sim(net, inputn);
    fitness = 0.1 * sum(sum(abs(pred_out - outputn)));
    
    % 可选：添加调试输出（训练时可关闭）
    % fprintf('适应度计算: 总参数=%d, 适应度=%.6f\n', actual_total, fitness);
end