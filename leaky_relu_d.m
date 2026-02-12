function da = leaky_relu_d(n, ~, ~)
    % Leaky ReLU 导数
    alpha = 0.01;
    da = ones(size(n));
    da(n < 0) = alpha;
end