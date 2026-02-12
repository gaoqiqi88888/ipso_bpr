% 3. PSO_standard.m：标准PSO算法函数
% 3. PSO_standard.m：标准PSO算法函数（修复返回参数赋值问题）
function [bestchrom, bestfitness, trace_best] = PSO_standard_p2(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p)
    % 1. 初始化粒子群（确保位置和速度在合法范围内）
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);  % 粒子位置
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);        % 粒子速度
    
    % 2. 初始化个体最优与全局最优（首次计算适应度）
    pbest_pos = particles_pos;  % 初始个体最优位置=当前位置
    pbest_fit = zeros(sizepop, 1);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));  % 计算每个粒子的初始适应度
    end
    [gbest_fit, gbest_idx] = min(pbest_fit);  % 全局最优适应度（最小误差）
    gbest_pos = pbest_pos(gbest_idx, :);      % 全局最优位置
    
    % 3. 初始化适应度曲线记录数组
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;  % 记录第1代（初始）的全局最优适应度
    
    w= w_init;

    % 4. PSO迭代主循环（从第2代开始更新）
    for gen = 2:maxgen
        % 4.1 遍历每个粒子更新速度和位置
        for i = 1:sizepop
            % （1）更新速度（固定惯性权重w，平衡惯性、个体学习、全局学习）
            r1 = rand();  % [0,1]随机数，增加搜索随机性
            r2 = rand();
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2*r2*(gbest_pos - particles_pos(i,:));
            
            % （2）速度约束（防止速度过大导致位置溢出）
            particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
            
            % （3）更新位置
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            
            % （4）位置约束（确保粒子在BPNN参数合法范围内）
            particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            
            % （5）更新个体最优（若当前位置适应度更优）
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
            end
        end

        %**************


        %**************
        
        % 4.2 更新全局最优（遍历所有个体最优，取最优值）
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 4.3 记录当前代的全局最优适应度（填充适应度曲线）
        trace_best(gen) = gbest_fit;
    end
    
    % 5. 函数结束前明确赋值返回参数（核心修复：确保每个输出参数都有值）
    bestchrom = gbest_pos;       % 全局最优位置（对应BPNN最优参数）
    bestfitness = gbest_fit;     % 全局最优适应度（最小BPNN预测误差）
    % trace_best已在循环中完整填充，无需额外赋值
end