%********************************************************************
% 鲸鱼优化算法(WOA)优化器
% 参考文献：Mirjalili & Lewis, 2016
%********************************************************************
function [best_pos, best_score, trace] = WOA_optimizer(...
    popsize, maxgen, dim, fobj, lb, ub, params)

%% 初始化鲸鱼种群
positions = rand(popsize, dim) .* (ub - lb) + lb;
fitness = zeros(popsize, 1);

for i = 1:popsize
    fitness(i) = fobj(positions(i, :));
end

[best_score, best_idx] = min(fitness);
best_pos = positions(best_idx, :);
trace = zeros(maxgen, 1);
trace(1) = best_score;

%% 主循环
for gen = 1:maxgen
    % 线性衰减的收敛因子
    a = params.a_init - (params.a_init - params.a_final) * (gen / maxgen);
    a2 = -1 + gen * ((-1) / maxgen);  % 螺旋参数
    
    for i = 1:popsize
        r1 = rand; r2 = rand;
        A = 2 * a * r1 - a;
        C = 2 * r2;
        b = params.b;
        l = (a2 - 1) * rand + 1;
        p = rand;
        
        for j = 1:dim
            if p < 0.5
                if abs(A) < 1
                    % 包围猎物
                    D = abs(C * best_pos(j) - positions(i, j));
                    positions(i, j) = best_pos(j) - A * D;
                else
                    % 随机搜索
                    rand_idx = randi(popsize);
                    D = abs(C * positions(rand_idx, j) - positions(i, j));
                    positions(i, j) = positions(rand_idx, j) - A * D;
                end
            else
                % 气泡网攻击（螺旋更新）
                D = abs(best_pos(j) - positions(i, j));
                positions(i, j) = D * exp(b * l) * cos(l * 2 * pi) + best_pos(j);
            end
        end
        
        % 边界处理
        positions(i, :) = max(min(positions(i, :), ub), lb);
        
        % 更新适应度
        new_fitness = fobj(positions(i, :));
        if new_fitness < fitness(i)
            fitness(i) = new_fitness;
            if new_fitness < best_score
                best_score = new_fitness;
                best_pos = positions(i, :);
            end
        end
    end
    
    trace(gen) = best_score;
end
end