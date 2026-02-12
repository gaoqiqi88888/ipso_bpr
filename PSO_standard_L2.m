function [bestchrom, bestfitness, trace_best] = PSO_standard_L2(sizepop, maxgen, numsum, fobj, c1, c2, w, v_max, v_min, pos_max, pos_min)
    % 针对361维双层网络的PSO算法
    % numsum = 361 (双层18-9网络的总参数数量)
    
    % 1. 初始化粒子群（确保位置和速度在合法范围内）
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);  % 粒子位置
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);        % 粒子速度
    
    % 验证维度
    if size(particles_pos, 2) ~= numsum
        error('粒子位置维度错误: 期望 %d 维, 实际 %d 维', numsum, size(particles_pos, 2));
    end
    
    % 2. 初始化个体最优与全局最优（首次计算适应度）
    pbest_pos = particles_pos;  % 初始个体最优位置=当前位置
    pbest_fit = zeros(sizepop, 1);
    
    fprintf('开始初始化 %d 个粒子的适应度 (维度=%d)...\n', sizepop, numsum);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));  % 计算每个粒子的初始适应度
        if mod(i, 10) == 0
            fprintf('  已完成 %d/%d 个粒子初始化\n', i, sizepop);
        end
    end
    
    [gbest_fit, gbest_idx] = min(pbest_fit);  % 全局最优适应度（最小误差）
    gbest_pos = pbest_pos(gbest_idx, :);      % 全局最优位置
    
    fprintf('初始化完成: 最佳适应度 = %.6f\n', gbest_fit);
    
    % 3. 初始化适应度曲线记录数组
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;  % 记录第1代（初始）的全局最优适应度
    
    % 4. PSO迭代主循环（从第2代开始更新）
    fprintf('开始 %d 代PSO迭代...\n', maxgen-1);
    
    for gen = 2:maxgen
        % 显示进度
        if mod(gen, 10) == 0
            fprintf('  第 %d/%d 代, 当前最优适应度: %.6f\n', gen, maxgen, gbest_fit);
        end
        
        % 4.1 遍历每个粒子更新速度和位置
        for i = 1:sizepop
            % （1）更新速度（固定惯性权重w，平衡惯性、个体学习、全局学习）
            r1 = rand(1, numsum);  % 为每个维度生成随机数
            r2 = rand(1, numsum);
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1 .* r1 .* (pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2 .* r2 .* (gbest_pos - particles_pos(i,:));
            
            % （2）速度约束（防止速度过大导致位置溢出）
            % 对361维分别进行约束
            for d = 1:numsum
                if particles_vel(i,d) > v_max
                    particles_vel(i,d) = v_max;
                elseif particles_vel(i,d) < v_min
                    particles_vel(i,d) = v_min;
                end
            end
            
            % （3）更新位置
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            
            % （4）位置约束（确保粒子在BPNN参数合法范围内）
            % 对361维分别进行约束
            for d = 1:numsum
                if particles_pos(i,d) > pos_max
                    particles_pos(i,d) = pos_max;
                elseif particles_pos(i,d) < pos_min
                    particles_pos(i,d) = pos_min;
                end
            end
            
            % （5）更新个体最优（若当前位置适应度更优）
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
                
                % （6）如果个体最优改进，检查是否更新全局最优
                if current_fit < gbest_fit
                    gbest_fit = current_fit;
                    gbest_pos = particles_pos(i,:);
                end
            end
        end
        
        % 4.2 额外的全局最优检查（遍历所有个体最优）
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 4.3 记录当前代的全局最优适应度（填充适应度曲线）
        trace_best(gen) = gbest_fit;
        
        % 4.4 检查收敛条件（可选）
        if gen > 10
            recent_improve = abs(trace_best(gen-9:gen-1) - trace_best(gen-8:gen));
            if max(recent_improve) < 1e-6
                fprintf('  第 %d 代检测到收敛，提前终止迭代\n', gen);
                % 填充剩余的trace_best为当前最优值
                trace_best(gen+1:end) = gbest_fit;
                break;
            end
        end
    end
    
    % 5. 函数结束前明确赋值返回参数
    bestchrom = gbest_pos;       % 全局最优位置（对应BPNN最优参数）
    bestfitness = gbest_fit;     % 全局最优适应度（最小BPNN预测误差）
    
    fprintf('PSO算法完成: 最终适应度 = %.6f, 迭代次数 = %d\n\n', bestfitness, gen);
end