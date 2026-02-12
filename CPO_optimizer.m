%% ==================== CPO优化算法 (2024) ====================
%  Crested Porcupine Optimizer (CPO)
%  参考文献: Abdel-Basset, M., et al. (2024). 
%  "Crested Porcupine Optimizer: A new metaheuristic for global optimization"
%  Knowledge-Based Systems, 284, 111287.
%==================================================================
function [best_pos, best_fitness, trace] = CPO_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 1. 参数设置
Tf = params.Tf;           % 跟踪因子 (0.7-0.9)
N_min = params.N_min;     % 最小种群规模 (3-5)
alpha = params.alpha;     % 防御角度参数 (0.1-0.3)
beta = params.beta;       % 飞行参数 (1.2-1.8)

%% 2. 初始化种群
X = rand(popsize, dim) .* (ub - lb) + lb;
fitness = zeros(popsize, 1);
for i = 1:popsize
    fitness(i) = fobj(X(i, :));
end

% 全局最优
[best_fitness, best_idx] = min(fitness);
best_pos = X(best_idx, :);

% 循环种群（用于防御机制）
X_cycle = X;
fitness_cycle = fitness;

trace = zeros(maxgen, 1);
trace(1) = best_fitness;

%% 3. 主循环
for t = 1:maxgen
    % 动态参数
    rho = 1 - t / maxgen;           % 防御概率衰减
    T = exp(-t / maxgen);           % 温度参数
    
    for i = 1:popsize
        %% 第一阶段：群体防御机制 (Exploration)
        if rand < rho
            % 视觉防御 - 随机选择个体
            k = randi(popsize);
            while k == i
                k = randi(popsize);
            end
            
            % 声波防御 - Levy飞行
            if rand < 0.5
                % Levy飞行策略
                step = levy_flight(dim, beta);
                X_new = X(i, :) + step .* (X(k, :) - X(i, :)) .* T;
            else
                % 随机扰动
                r1 = rand(1, dim);
                X_new = X(i, :) + r1 .* (X_cycle(randi(popsize), :) - X(i, :));
            end
        
        %% 第二阶段：个体防御机制 (Exploitation)
        else
            % 化学防御 - 局部搜索
            if rand < 0.5
                sigma = 0.2 * (1 - t/maxgen);
                X_new = best_pos + randn(1, dim) .* sigma .* (ub - lb);
            
            % 物理防御 - 自适应步长
            else
                r2 = rand;
                if rand < 0.5
                    X_new = X(i, :) + r2 * alpha * (best_pos - X(i, :));
                else
                    X_new = X(i, :) + r2 * alpha * (X_cycle(randi(popsize), :) - X(i, :));
                end
            end
        end
        
        %% 边界处理
        X_new = max(min(X_new, ub), lb);
        
        %% 评估新位置
        new_fitness = fobj(X_new);
        
        %% 贪婪选择
        if new_fitness < fitness(i)
            X(i, :) = X_new;
            fitness(i) = new_fitness;
            
            if new_fitness < best_fitness
                best_fitness = new_fitness;
                best_pos = X_new;
            end
        end
    end
    
    %% 4. 循环种群更新
    if mod(t, 5) == 0  % 每5代更新一次循环种群
        X_cycle = X;
        fitness_cycle = fitness;
    end
    
    %% 5. 种群规模缩减策略（可选）
    if t > maxgen * 0.7 && popsize > N_min
        new_popsize = max(N_min, round(popsize * 0.95));
        if new_popsize < popsize
            [~, idx] = sort(fitness);
            X = X(idx(1:new_popsize), :);
            fitness = fitness(idx(1:new_popsize));
            popsize = new_popsize;
        end
    end
    
    trace(t) = best_fitness;
    
    % 显示进度
    % if mod(t, 10) == 0
    %     fprintf('CPO Gen %d/%d, Best Fitness: %.4e\n', t, maxgen, best_fitness);
    % end
end
end

%% Levy飞行函数
function L = levy_flight(dim, beta)
    % 计算Levy分布步长
    sigma = (gamma(1+beta) * sin(pi*beta/2) / ...
             (gamma((1+beta)/2) * beta * 2^((beta-1)/2)))^(1/beta);
    
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    step = u ./ (abs(v).^(1/beta));
    
    % 归一化
    L = 0.01 * step;
end