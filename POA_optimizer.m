%% ==================== POA优化算法 (2023) ====================
%  Pelican Optimization Algorithm (POA)
%  参考文献: Trojovský, P., & Dehghani, M. (2023). 
%  "Pelican Optimization Algorithm: A novel nature-inspired algorithm"
%  Heliyon, 9(3), e14887.
%==================================================================
function [best_pos, best_fitness, trace] = POA_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 1. 参数设置
R = params.R;           % 搜索半径 (默认0.2)
C = params.C;           % 收敛因子 (默认1.5)

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
    % 动态参数 - 搜索半径随迭代衰减
    r = R * (1 - t/maxgen);
    
    for i = 1:popsize
        %% 第一阶段：全局勘探（模拟塘鹅俯冲捕鱼）
        % 随机选择猎物位置
        prey_idx = randi(popsize);
        while prey_idx == i
            prey_idx = randi(popsize);
        end
        
        if fitness(prey_idx) < fitness(i)
            % 向更优个体移动（向优质猎物俯冲）
            X_new = X(i, :) + rand(1, dim) .* (X(prey_idx, :) - C * X(i, :));
        else
            % 向随机方向探索（寻找新猎物）
            X_new = X(i, :) + rand(1, dim) .* (X(i, :) - X(prey_idx, :));
        end
        
        % 边界处理
        X_new = max(min(X_new, ub), lb);
        
        % 评估新位置
        new_fitness = fobj(X_new);
        if new_fitness < fitness(i)
            X(i, :) = X_new;
            fitness(i) = new_fitness;
            if new_fitness < best_fitness
                best_fitness = new_fitness;
                best_pos = X_new;
            end
        end
        
        %% 第二阶段：局部开发（模拟塘鹅水面滑翔）
        % 在当前最优解附近进行局部精细搜索
        X_new = best_pos + r * (2 * rand(1, dim) - 1) .* (ub - lb);
        X_new = max(min(X_new, ub), lb);
        
        % 评估新位置
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
    
    % 显示进度（可选）
    if mod(t, 10) == 0 && t > 1
        fprintf('  POA Gen %d/%d, Best: %.4e\n', t, maxgen, best_fitness);
    end
end
end