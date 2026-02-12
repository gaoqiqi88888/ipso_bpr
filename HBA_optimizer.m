%% ==================== HBA优化算法 (2022) ====================
%  Honey Badger Algorithm (HBA)
%  参考文献: Hashim, F. A., et al. (2022). 
%  "Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems"
%  Mathematics and Computers in Simulation, 192, 84-110.
%==================================================================
function [best_pos, best_fitness, trace] = HBA_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 1. 参数设置
beta = params.beta;      % 嗅觉因子 (默认6)
C = params.C;            % 常数 (默认2)
alpha_init = params.alpha_init;  % 密度因子初始 (默认0.98)
alpha_final = params.alpha_final; % 密度因子最终 (默认0.1)

%% 2. 初始化种群
X = rand(popsize, dim) .* (ub - lb) + lb;
fitness = zeros(popsize, 1);
for i = 1:popsize
    fitness(i) = fobj(X(i, :));
end

[best_fitness, best_idx] = min(fitness);
best_pos = X(best_idx, :);
trace = zeros(maxgen, 1);
trace(1) = best_fitness;

%% 3. 主循环
for t = 1:maxgen
    % 密度因子 - 非线性衰减
    alpha = alpha_final + (alpha_init - alpha_final) * (1 - t/maxgen);
    
    for i = 1:popsize
        % 随机选择个体
        r1 = rand;
        r2 = rand;
        r3 = rand;
        r4 = rand;
        r5 = rand;
        
        % 随机选择索引
        j = randi(popsize);
        while j == i
            j = randi(popsize);
        end
        
        %% 第一阶段：挖掘模式（局部搜索）
        if r1 < 0.5
            F = 1;
        else
            F = -1;
        end
        
        % 嗅觉强度
        I = r2 * (best_fitness / (fitness(i) + eps))^2;
        
        % 更新位置 - 挖掘模式
        X_new = best_pos + F * beta * I * best_pos + ...
                F * alpha * (X(j, :) - X(i, :)) * abs(cos(2*pi*r3)) * (1 - exp(-r4));
        
        %% 第二阶段：采蜜模式（全局探索）
        if r5 < 0.5
            X_new = X_new + F * r4 * alpha * (X(j, :) - X(i, :));
        else
            X_new = best_pos + F * r4 * alpha * (X(j, :) - X(i, :));
        end
        
        % 边界处理
        X_new = max(min(X_new, ub), lb);
        
        % 评估
        new_fitness = fobj(X_new);
        if new_fitness < fitness(i)
            X(i, :) = X_new;
            fitness(i) = new_fitness;
            if new_fitness < best_fitness
                best_fitness = new_fitness;
                best_pos = X_new;
            end
        end
    end
    
    trace(t) = best_fitness;
end
end