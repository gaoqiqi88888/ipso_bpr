%********************************************************************
% 遗传算法(GA)优化器
% 适用于BPNN参数优化
%********************************************************************
%% 遗传算法(GA) - 最终修正版
function [best_chrom, best_fitness, trace] = GA_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

    % 初始化
    population = rand(popsize, dim) .* (ub - lb) + lb;
    fitness = zeros(popsize, 1);
    for i = 1:popsize
        fitness(i) = fobj(population(i, :));
    end
    
    [best_fitness, best_idx] = min(fitness);
    best_chrom = population(best_idx, :);
    trace = zeros(maxgen, 1);
    trace(1) = best_fitness;
    
    for gen = 2:maxgen
        % 选择
        selected_idx = selection_tournament(fitness, popsize, params.select_ratio);
        new_population = population(selected_idx, :);
        new_fitness = fitness(selected_idx);
        
        % 获取当前种群大小（重要修正！）
        current_popsize = size(new_population, 1);
        
        % 交叉 - 使用current_popsize而不是popsize
        for i = 1:2:current_popsize-1
            if rand < params.pc
                [new_population(i, :), new_population(i+1, :)] = ...
                    crossover_sbx(new_population(i, :), new_population(i+1, :), lb, ub);
            end
        end
        
        % 变异
        for i = 1:current_popsize
            if rand < params.pm
                new_population(i, :) = mutation_polynomial(...
                    new_population(i, :), lb, ub, gen/maxgen);
            end
        end
        
        % 边界处理
        new_population = max(min(new_population, ub), lb);
        
        % 评估新个体
        for i = 1:current_popsize
            new_fitness(i) = fobj(new_population(i, :));
        end
        
        % 精英保留 - 如果种群大小变小了，需要先恢复种群大小
        if current_popsize < popsize
            % 随机补充个体到原大小
            additional_num = popsize - current_popsize;
            additional_idx = randi(current_popsize, additional_num, 1);
            new_population = [new_population; new_population(additional_idx, :)];
            new_fitness = [new_fitness; new_fitness(additional_idx)];
        end
        
        % 精英保留
        [current_best, current_best_idx] = min(new_fitness);
        if current_best < best_fitness
            best_fitness = current_best;
            best_chrom = new_population(current_best_idx, :);
        else
            [~, worst_idx] = max(new_fitness);
            new_population(worst_idx, :) = best_chrom;
            new_fitness(worst_idx) = best_fitness;
        end
        
        population = new_population;
        fitness = new_fitness;
        trace(gen) = best_fitness;
    end
end

%% 锦标赛选择
function selected_idx = selection_tournament(fitness, k, ratio)
popsize = length(fitness);
select_num = round(popsize * ratio);
selected_idx = zeros(select_num, 1);
for i = 1:select_num
    candidates = randperm(popsize, 2);
    [~, winner] = min(fitness(candidates));
    selected_idx(i) = candidates(winner);
end
end

%% 模拟二进制交叉 - 修正版
function [c1, c2] = crossover_sbx(p1, p2, lb, ub)
    dim = length(p1);
    c1 = zeros(1, dim);
    c2 = zeros(1, dim);
    eta_c = 20;
    
    for i = 1:dim
        if abs(p1(i) - p2(i)) > 1e-10
            if p1(i) < p2(i)
                y1 = p1(i); y2 = p2(i);
            else
                y1 = p2(i); y2 = p1(i);
            end
            
            % 修正：lb和ub是标量，不是向量
            yl = lb;  % 原来写的是 lb(i)
            yu = ub;  % 原来写的是 ub(i)
            
            rand_var = rand;
            beta = 1 + 2 * (y1 - yl) / (y2 - y1);
            alpha = 2 - beta^(-(eta_c+1));
            
            if rand_var <= 1/alpha
                beta_q = (rand_var * alpha)^(1/(eta_c+1));
            else
                beta_q = (1/(2 - rand_var * alpha))^(1/(eta_c+1));
            end
            
            c1(i) = 0.5 * ((y1 + y2) - beta_q * (y2 - y1));
            
            beta = 1 + 2 * (yu - y2) / (y2 - y1);
            alpha = 2 - beta^(-(eta_c+1));
            
            if rand_var <= 1/alpha
                beta_q = (rand_var * alpha)^(1/(eta_c+1));
            else
                beta_q = (1/(2 - rand_var * alpha))^(1/(eta_c+1));
            end
            
            c2(i) = 0.5 * ((y1 + y2) + beta_q * (y2 - y1));
            
            c1(i) = max(min(c1(i), yu), yl);
            c2(i) = max(min(c2(i), yu), yl);
        else
            c1(i) = p1(i);
            c2(i) = p2(i);
        end
    end
end

%% 多项式变异 - 修正版
function mutated = mutation_polynomial(ind, lb, ub, ratio)
    dim = length(ind);
    mutated = ind;
    eta_m = 20 + 20 * (1 - ratio);
    
    for i = 1:dim
        if rand < 1/dim
            y = ind(i);
            yl = lb;  % 修正：lb是标量
            yu = ub;  % 修正：ub是标量
            
            delta1 = (y - yl) / (yu - yl);
            delta2 = (yu - y) / (yu - yl);
            rand_var = rand;
            
            if rand_var < 0.5
                delta_q = (2 * rand_var + (1 - 2 * rand_var) * ...
                          (1 - delta1)^(eta_m+1))^(1/(eta_m+1)) - 1;
            else
                delta_q = 1 - (2 * (1 - rand_var) + 2 * (rand_var - 0.5) * ...
                          (1 - delta2)^(eta_m+1))^(1/(eta_m+1));
            end
            
            mutated(i) = y + delta_q * (yu - yl);
            mutated(i) = max(min(mutated(i), yu), yl);
        end
    end
end