% 2. cal_fitness.m：适应度函数（BPNN预测误差计算）
function fitness = cal_fitness(x, inputnum, hiddennum, outputnum, net, inputn, outputn)
    % 解码粒子位置为BPNN权值与阈值
    w1 = x(1:inputnum*hiddennum);                  % 输入-隐藏层权值
    b1 = x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);  % 隐藏层阈值
    w2 = x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);  % 隐藏-输出层权值
    b2 = x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:end);  % 输出层阈值
    
    % 赋值权值与阈值到BPNN
    net.iw{1,1} = reshape(w1, hiddennum, inputnum);
    net.lw{2,1} = reshape(w2, outputnum, hiddennum);
    net.b{1} = reshape(b1, hiddennum, 1);
    net.b{2} = b2;
    
    % 计算BPNN预测误差（适应度=归一化总误差，越小越好）
    pred_out = sim(net, inputn);
    fitness = 0.1 * sum(sum(abs(pred_out - outputn)));  % 与原代码误差计算方式一致，保证对比公平性
end