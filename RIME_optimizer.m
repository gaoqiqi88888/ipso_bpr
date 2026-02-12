%********************************************************************
% RIME优化算法（2023年最新算法）
% 参考文献：Su et al., 2023, Neurocomputing
% 基于霜冰现象的物理启发优化算法
%********************************************************************
%% RIME优化算法(2023) - 修正版
function [best_pos, best_fitness, trace] = RIME_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 参数设置
R = params.R;           % 软霜冰参数
K = params.K;           % 附着参数
E_init = params.E_init; % 环境因子初始
E_final = params.E_final; % 环境因子最终

%% 初始化种群
X = rand(popsize, dim) .* (ub - lb) + lb;
fitness = zeros(popsize, 1);
for i = 1:popsize
    fitness(i) = fobj(X(i, :));
end

[best_fitness, best_idx] = min(fitness);
best_pos = X(best_idx, :);

% 存储所有粒子的历史最优
Pbest = X;
Pbest_fitness = fitness;

trace = zeros(maxgen, 1);
trace(1) = best_fitness;

%% 主循环
for t = 1:maxgen
    % 环境因子（非线性衰减）
    E = E_init - (E_init - E_final) * (t / maxgen)^2;
    
    % 归一化适应度
    norm_fitness = (fitness - min(fitness)) / (max(fitness) - min(fitness) + eps);
    
    for i = 1:popsize
        %% 1. 软霜冰搜索（Exploitation）
        r1 = rand;
        if r1 < E
            for j = 1:dim
                r2 = rand;
                if r2 < R
                    % 粒子间相互作用
                    r3 = rand;
                    if r3 < K
                        % 改进的附着机制 - 修正：ub和lb是标量
                        X(i, j) = Pbest(randi(popsize), j) + ...
                                 randn * (ub - lb) * exp(-t / maxgen);  % 去掉(j)
                    else
                        % 粒子自我更新 - 修正：ub和lb是标量
                        X(i, j) = best_pos(j) + ...
                                 (ub - lb) * (rand - 0.5) * E;  % 去掉(j)
                    end
                end
            end
        else
            %% 2. 硬霜冰搜索（Exploration）
            for j = 1:dim
                r4 = rand;
                X(i, j) = lb + r4 * (ub - lb);  % 去掉(j)
            end
        end
        
        % 边界处理 - 修正：ub和lb是标量
        X(i, :) = max(min(X(i, :), ub), lb);
        
        % 评估
        new_fitness = fobj(X(i, :));
        
        % 更新个体历史最优
        if new_fitness < Pbest_fitness(i)
            Pbest(i, :) = X(i, :);
            Pbest_fitness(i) = new_fitness;
        end
        
        % 更新全局最优
        if new_fitness < best_fitness
            best_fitness = new_fitness;
            best_pos = X(i, :);
        end
        
        fitness(i) = new_fitness;
    end
    
    %% 3. 正向贪婪选择机制
    for i = 1:popsize
        if Pbest_fitness(i) < fitness(i)
            X(i, :) = Pbest(i, :);
            fitness(i) = Pbest_fitness(i);
        end
    end
    
    trace(t) = best_fitness;
end
end