%********************************************************************
% 估计收敛代数
%********************************************************************
function converge_gen = estimate_convergence_gen(trace_matrix, maxgen)
    if isempty(trace_matrix)
        converge_gen = maxgen;
        return;
    end
    
    mean_trace = mean(trace_matrix, 2);
    final_value = mean_trace(end);
    threshold = final_value * 1.05;  % 5%阈值
    
    converge_gen = find(mean_trace <= threshold, 1);
    if isempty(converge_gen)
        converge_gen = maxgen;
    end
end

%********************************************************************
% BP神经网络适应度函数
%********************************************************************
function fitness = cal_fitness_bpnn(x, inputnum, hidden_layers, outputnum, ...
                                   net, P_train, T_train)
    % 权重数量计算
    w1_num = inputnum * hidden_layers(1);
    w2_num = hidden_layers(1) * hidden_layers(2);
    w3_num = hidden_layers(2) * outputnum;
    b1_num = hidden_layers(1);
    b2_num = hidden_layers(2);
    b3_num = outputnum;
    
    % 解析参数
    idx = 1;
    net.IW{1,1} = reshape(x(idx:idx+w1_num-1), hidden_layers(1), inputnum);
    idx = idx + w1_num;
    
    net.LW{2,1} = reshape(x(idx:idx+w2_num-1), hidden_layers(2), hidden_layers(1));
    idx = idx + w2_num;
    
    net.LW{3,2} = reshape(x(idx:idx+w3_num-1), outputnum, hidden_layers(2));
    idx = idx + w3_num;
    
    net.b{1} = x(idx:idx+b1_num-1)';
    idx = idx + b1_num;
    
    net.b{2} = x(idx:idx+b2_num-1)';
    idx = idx + b2_num;
    
    net.b{3} = x(idx:idx+b3_num-1)';
    
    % 预测
    y = sim(net, P_train);
    fitness = mse(T_train - y);
end

%********************************************************************
% 生成训练数据
%********************************************************************
function [P, T] = generate_training_data(degraded_img, clean_img, window_size)
    [h, w] = size(degraded_img);
    P = zeros((h-2)*(w-2), 9);
    T = zeros((h-2)*(w-2), 1);
    
    idx = 0;
    for i = 2:h-1
        for j = 2:w-1
            idx = idx + 1;
            window = degraded_img(i-1:i+1, j-1:j+1);
            P(idx, :) = window(:)';
            T(idx) = clean_img(i, j);
        end
    end
end