function a = leaky_relu(n, ~)
    % Leaky ReLU 传递函数，负半轴斜率 = 0.01
    alpha = 0.01;
    a = max(n, alpha * n);
end