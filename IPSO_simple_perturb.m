function [bestchrom, bestfitness, trace_best] = IPSO_simple_perturb(...
    sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, ...
    v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p)
    
    % 初始化粒子群
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);
    
    % 初始化个体最优与全局最优
    pbest_pos = particles_pos;
    pbest_fit = zeros(sizepop, 1);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));
    end
    [gbest_fit, gbest_idx] = min(pbest_fit);
    gbest_pos = pbest_pos(gbest_idx, :);
    
    % 记录适应度曲线
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;
    
    % IPSO迭代主循环
    for gen = 2:maxgen
        % 幂次递减惯性权重更新
        normalized_progress = (gen - 1) / (maxgen - 1);
        w = w_final + (w_init - w_final) * (1 - normalized_progress)^p;
        
        % 遍历所有粒子
        for i = 1:sizepop
            % 速度更新
            r1 = rand();
            r2 = rand();
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2*r2*(gbest_pos - particles_pos(i,:));
            
            % 速度约束
            particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
            
            % 位置更新
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            
            % 最简单扰动策略
            if gen > perturb_trigger_ratio * maxgen
                perturb = perturb_std * randn(1, numsum);
                particles_pos(i,:) = particles_pos(i,:) + perturb;
                particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            end
            
            % 更新个体最优
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
            end
        end
        
        % 更新全局最优
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 记录当前代最优适应度
        trace_best(gen) = gbest_fit;
    end
    
    % 输出结果
    bestchrom = gbest_pos;
    bestfitness = gbest_fit;
end