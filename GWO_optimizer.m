%********************************************************************
% 灰狼优化算法(GWO)优化器
% 参考文献：Mirjalili et al., 2014
%********************************************************************
function [alpha_pos, alpha_score, trace] = GWO_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 初始化灰狼种群
positions = rand(popsize, dim) .* (ub - lb) + lb;
fitness = zeros(popsize, 1);

for i = 1:popsize
    fitness(i) = fobj(positions(i, :));
end

% 初始化alpha, beta, delta
[alpha_score, alpha_idx] = min(fitness);
alpha_pos = positions(alpha_idx, :);

[beta_score, beta_idx] = min(fitness(fitness ~= alpha_score));
if isempty(beta_idx)
    beta_idx = randi(popsize);
end
beta_pos = positions(beta_idx, :);

[delta_score, delta_idx] = min(fitness(fitness ~= alpha_score & fitness ~= beta_score));
if isempty(delta_idx)
    delta_idx = randi(popsize);
end
delta_pos = positions(delta_idx, :);

trace = zeros(maxgen, 1);
trace(1) = alpha_score;

%% 主循环
for gen = 1:maxgen
    % 线性衰减的收敛因子
    a = params.a_init - (params.a_init - params.a_final) * (gen / maxgen);
    
    for i = 1:popsize
        % 更新位置（alpha, beta, delta引导）
        for j = 1:dim
            % Alpha
            r1 = rand; r2 = rand;
            A1 = 2 * a * r1 - a;
            C1 = 2 * r2;
            D_alpha = abs(C1 * alpha_pos(j) - positions(i, j));
            X1 = alpha_pos(j) - A1 * D_alpha;
            
            % Beta
            r1 = rand; r2 = rand;
            A2 = 2 * a * r1 - a;
            C2 = 2 * r2;
            D_beta = abs(C2 * beta_pos(j) - positions(i, j));
            X2 = beta_pos(j) - A2 * D_beta;
            
            % Delta
            r1 = rand; r2 = rand;
            A3 = 2 * a * r1 - a;
            C3 = 2 * r2;
            D_delta = abs(C3 * delta_pos(j) - positions(i, j));
            X3 = delta_pos(j) - A3 * D_delta;
            
            % 加权平均
            positions(i, j) = (X1 + X2 + X3) / 3;
        end
        
        % 边界处理
        positions(i, :) = max(min(positions(i, :), ub), lb);
    end
    
    % 评估新位置
    for i = 1:popsize
        fitness(i) = fobj(positions(i, :));
        
        % 更新alpha, beta, delta
        if fitness(i) < alpha_score
            alpha_score = fitness(i);
            alpha_pos = positions(i, :);
        elseif fitness(i) < beta_score
            beta_score = fitness(i);
            beta_pos = positions(i, :);
        elseif fitness(i) < delta_score
            delta_score = fitness(i);
            delta_pos = positions(i, :);
        end
    end
    
    trace(gen) = alpha_score;
end
end