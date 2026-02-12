function [bestchrom, bestfitness, trace_best] = PSO_improved_L2(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p)
    % 针对361维双层网络的改进PSO算法（含幂次递减惯性权重+高斯扰动）
    % numsum = 361 (双层18-9网络的总参数数量)
    % p: 幂次递减参数（p=1.5推荐）
    
    % 1. 初始化粒子群（位置、速度均在合法范围）
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);
    
    % 验证维度
    if size(particles_pos, 2) ~= numsum
        error('粒子位置维度错误: 期望 %d 维, 实际 %d 维', numsum, size(particles_pos, 2));
    end
    
    % 2. 初始化个体最优与全局最优（首次计算适应度）
    pbest_pos = particles_pos;
    pbest_fit = zeros(sizepop, 1);
    
    fprintf('IPSO: 开始初始化 %d 个粒子的适应度 (维度=%d, p=%.1f)...\n', sizepop, numsum, p);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));
        if mod(i, 10) == 0
            fprintf('  已完成 %d/%d 个粒子初始化\n', i, sizepop);
        end
    end
    
    [gbest_fit, gbest_idx] = min(pbest_fit);
    gbest_pos = pbest_pos(gbest_idx, :);
    
    fprintf('IPSO: 初始化完成: 最佳适应度 = %.6f\n', gbest_fit);
    
    % 3. 初始化适应度曲线记录
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;
    
    % 4. IPSO迭代主循环
    fprintf('IPSO: 开始 %d 代迭代 (幂次递减 p=%.1f)...\n', maxgen-1, p);
    
    for gen = 2:maxgen
        % 显示进度
        if mod(gen, 10) == 0
            fprintf('  IPSO第 %d/%d 代, 当前最优适应度: %.6f\n', gen, maxgen, gbest_fit);
        end
        
        % 4.1 幂次递减自适应惯性权重更新
        % 原代码是线性递减: w = w_init - (w_init - w_final) * (gen / maxgen);
        % 现在改为幂次递减:
        ratio = gen / maxgen;  % 迭代进度 (0到1)
        w = w_final + (w_init - w_final) * (1 - ratio)^p;
        
        % 幂次递减说明：
        % p=1.0: 线性递减（与原代码相同）
        % p>1.0: 前期下降慢，后期下降快（侧重后期局部搜索）
        % p<1.0: 前期下降快，后期下降慢（侧重早期全局搜索）
        
        % 4.2 遍历粒子更新速度、位置与扰动
        for i = 1:sizepop
            % （1）速度更新（使用361维独立的随机数）
            r1 = rand(1, numsum);  % 每个维度的独立随机数
            r2 = rand(1, numsum);
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1 .* r1 .* (pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2 .* r2 .* (gbest_pos - particles_pos(i,:));
            
            % （2）速度约束（逐维度约束）
            for d = 1:numsum
                if particles_vel(i,d) > v_max
                    particles_vel(i,d) = v_max;
                elseif particles_vel(i,d) < v_min
                    particles_vel(i,d) = v_min;
                end
            end
            
            % （3）位置更新
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            
            % （4）位置约束（逐维度约束）
            for d = 1:numsum
                if particles_pos(i,d) > pos_max
                    particles_pos(i,d) = pos_max;
                elseif particles_pos(i,d) < pos_min
                    particles_pos(i,d) = pos_min;
                end
            end
            
            % （5）late-iteration高斯扰动（迭代超70%时触发）
            if gen > perturb_trigger_ratio * maxgen
                perturb = perturb_std * randn(1, numsum);  % 361维高斯扰动
                particles_pos(i,:) = particles_pos(i,:) + perturb;
                
                % 重新约束位置（扰动后可能超出边界）
                for d = 1:numsum
                    if particles_pos(i,d) > pos_max
                        particles_pos(i,d) = pos_max;
                    elseif particles_pos(i,d) < pos_min
                        particles_pos(i,d) = pos_min;
                    end
                end
            end
            
            % （6）更新个体最优并检查全局最优
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
                
                % 如果个体最优改进，检查是否更新全局最优
                if current_fit < gbest_fit
                    gbest_fit = current_fit;
                    gbest_pos = particles_pos(i,:);
                end
            end
        end
        
        % 4.3 额外的全局最优检查（遍历所有个体最优）
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 4.4 记录当前代最优适应度
        trace_best(gen) = gbest_fit;
        
        % 4.5 检查收敛条件（可选）
        if gen > 10
            recent_improve = abs(trace_best(gen-9:gen-1) - trace_best(gen-8:gen));
            if max(recent_improve) < 1e-6
                fprintf('  IPSO第 %d 代检测到收敛，提前终止迭代\n', gen);
                % 填充剩余的trace_best为当前最优值
                trace_best(gen+1:end) = gbest_fit;
                break;
            end
        end
    end
    
    % 5. 赋值输出参数
    bestchrom = gbest_pos;
    bestfitness = gbest_fit;
    
    fprintf('IPSO算法完成: 最终适应度 = %.6f, 迭代次数 = %d, 最终权重 = %.3f\n\n', ...
            bestfitness, min(gen, maxgen), w);
end