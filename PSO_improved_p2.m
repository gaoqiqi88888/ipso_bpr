% p1：该版本有问题未修改

% PSO_improved_p.m：改进PSO（IPSO）算法函数（含幂次递减惯性权重+高斯扰动）
% 输入参数（15个）：
% sizepop-粒子群规模, maxgen-最大迭代次数, numsum-优化维度, fobj-适应度函数,
% c1-个体学习因子, c2-全局学习因子, w_init-初始惯性权重, w_final-最终惯性权重,
% v_max-最大速度, v_min-最小速度, pos_max-位置上限, pos_min-位置下限,
% perturb_trigger_ratio-扰动触发比例, perturb_std-扰动标准差, p-幂次递减指数
% 输出参数（3个）：bestchrom-全局最优位置, bestfitness-全局最优适应度, trace_best-适应度曲线
function [bestchrom, bestfitness, trace_best] = PSO_improved_p2(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std, p)
    % 1. 初始化粒子群（位置、速度均在合法范围）
    particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);
    particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);
    
    % 2. 初始化个体最优与全局最优（首次计算适应度）
    pbest_pos = particles_pos;
    pbest_fit = zeros(sizepop, 1);
    for i = 1:sizepop
        pbest_fit(i) = fobj(particles_pos(i,:));  % 计算初始适应度
    end
    [gbest_fit, gbest_idx] = min(pbest_fit);
    gbest_pos = pbest_pos(gbest_idx, :);
    
    % 3. 初始化适应度曲线记录
    trace_best = zeros(maxgen, 1);
    trace_best(1) = gbest_fit;

    % w = w_init;
    
    % 4. IPSO迭代主循环
    for gen = 2:maxgen
        % 4.1 幂次递减惯性权重更新
        % 公式：w = w_final + (w_init - w_final) * (1 - (gen-1)/(maxgen-1))^p
        normalized_progress = (gen - 1) / (maxgen - 1);  % 归一化进度 [0,1]
        w = w_final + (w_init - w_final) * (1 - normalized_progress)^p;
        

        % 4.2 遍历粒子更新速度、位置与扰动
        for i = 1:sizepop
            % （1）速度更新
            r1 = rand();
            r2 = rand();
            particles_vel(i,:) = w * particles_vel(i,:) + ...
                                 c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
                                 c2*r2*(gbest_pos - particles_pos(i,:));
            % （2）速度约束
            particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
            
            % （3）位置更新
            particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
            % （4）位置约束
            particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
            
            % （5） late-iteration高斯扰动（迭代超设定比例时触发）
            x = 0;
            % 方案3：只对部分粒子扰动，保护优秀粒子
            if gen > perturb_trigger_ratio * maxgen
                % 1. 只扰动非精英粒子（排名后10%）
                if i > sizepop*0.9
                    perturb = perturb_std * randn(1, numsum);  %原方案
                    %下面是自适应升级方案
                    % current_std = perturb_std * (0.5 + 0.5*(w/w_init));
                    % perturb = current_std * randn(1, numsum);
                    particles_pos(i,:) = particles_pos(i,:) + perturb;
                    particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);  % 重新约束位置
                end
            end



            % （6）更新个体最优
            current_fit = fobj(particles_pos(i,:));
            if current_fit < pbest_fit(i)
                pbest_fit(i) = current_fit;
                pbest_pos(i,:) = particles_pos(i,:);
            end
        end
        
        % 4.3 更新全局最优
        [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
        if current_gbest_fit < gbest_fit
            gbest_fit = current_gbest_fit;
            gbest_pos = pbest_pos(current_gbest_idx, :);
        end
        
        % 4.4 记录当前代最优适应度
        trace_best(gen) = gbest_fit;
    end

    if x==1
      fprintf('已扰动');
    end
    
    % 5. 赋值输出参数
    bestchrom = gbest_pos;
    bestfitness = gbest_fit;
end


          % 下面是参考方案
          % x = 0;
          %   % 方案1：70%高斯扰动
          %   % if gen > perturb_trigger_ratio * maxgen
          %   %     x = 1;
          %   %     perturb = perturb_std * randn(1, numsum);  % 高斯扰动（均值0，标准差perturb_std）
          %   %     particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %     particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);  % 重新约束位置
          %   % end
          % 
          %   % 方案2：扰动强度随迭代递减（与权重衰减协调）
          %   % if gen > perturb_trigger_ratio * maxgen
          %   %     % 扰动强度随迭代递减
          %   %     % current_perturb_std = perturb_std * (1 - (gen-1)/(maxgen-1));
          %   % 
          %   %     % 或与权重w协调
          %   %     current_perturb_std = perturb_std * w;  % 扰动强度与权重成正比
          %   % 
          %   %     perturb = current_perturb_std * randn(1, numsum);
          %   %     particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   % end
          % 
          % 
          %   % 方案3：只对部分粒子扰动，保护优秀粒子
          %   if gen > perturb_trigger_ratio * maxgen
          %       % 1. 只扰动非精英粒子（排名后10%）
          %       % if i > sizepop*0.9
          %           perturb = perturb_std * randn(1, numsum);  %原方案
          %           %下面是自适应升级方案
          %           % current_std = perturb_std * (0.5 + 0.5*(w/w_init));
          %           % perturb = current_std * randn(1, numsum);
          % 
          %           particles_pos(i,:) = particles_pos(i,:) + perturb;
          %           particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);  % 重新约束位置
          %       % end
          %   % 
          %   %     % % 或 2. 基于粒子质量概率性扰动——验证失败
          %   %     % particle_rank = get_particle_rank(i);  % 粒子排名（1=最好）
          %   %     % particle_rank = get_particle_rank(i, pbest_fit);
          %   %     % perturb_prob = 0.8 * (particle_rank/sizepop);  % 差粒子扰动概率高
          %   %     % 
          %   %     % if rand() < perturb_prob
          %   %     %     perturb = perturb_std * randn(1, numsum);
          %   %     %     particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %     % end
          %   % 
          %   %     x=1;
          % 
          % 
          % 
          %   % 分阶段扰动策略 失败
          %   % if gen < 0.4 * maxgen  % 前期：增强探索
          %   %     % 小概率扰动所有粒子（探索性扰动）
          %   %     if rand() < 0.1  % 10%概率
          %   %         perturb = perturb_std * 0.5 * randn(1, numsum);  % 弱扰动
          %   %         particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %         particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
          %   %         % fprintf('前期扰动');
          %   %     end
          %   % 
          %   % elseif gen > 0.4 * maxgen && gen < 0.8 * maxgen  % 中期：平衡策略
          %   %     % 扰动中等质量粒子
          %   %     current_fit = pbest_fit(i);
          %   %     median_fit = median(pbest_fit);
          %   % 
          %   %     if current_fit > median_fit * 0.8  % 不是最好的，也不是最差的
          %   %         if rand() < 0.2
          %   %             perturb = perturb_std * randn(1, numsum);
          %   %             particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %             particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
          %   %             % fprintf('中期扰动');
          %   %         end
          %   %     end
          %   % 
          %   % else  % 后期：精细扰动
          %   %     % 只扰动表现差的粒子（如您的原方案，但调整参数）
          %   %     if i > sizepop * 0.7  % 扰动后30%粒子
          %   %         % 更小的扰动强度（精细调整）
          %   %         perturb = perturb_std * 0.3 * randn(1, numsum);
          %   %         % perturb = perturb_std  * randn(1, numsum);
          %   %         particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %         particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
          %   %         % fprintf('后期扰动');
          %   %     end
          %   % end
          % 
          %   % 添加停滞检测-失败
          %   % stagnation_window = 10;  % 检测窗口
          %   % improvement_threshold = 1e-4;
          %   % 
          %   % if gen > stagnation_window
          %   %     % 计算最近stagnation_window代的改进
          %   %     recent_improvement = abs(trace_best(gen-1) - trace_best(gen-stagnation_window));
          %   % 
          %   %     if recent_improvement < improvement_threshold
          %   %         % 检测到停滞，执行扰动
          %   %         if rand() < 0.3  % 30%粒子被扰动
          %   %             perturb = perturb_std * randn(1, numsum);            % 添加停滞检测-失败
          %   % stagnation_window = 10;  % 检测窗口
          %   % improvement_threshold = 1e-4;
          %   % 
          %   % if gen > stagnation_window
          %   %     % 计算最近stagnation_window代的改进
          %   %     recent_improvement = abs(trace_best(gen-1) - trace_best(gen-stagnation_window));
          %   % 
          %   %     if recent_improvement < improvement_threshold
          %   %         % 检测到停滞，执行扰动
          %   %         if rand() < 0.3  % 30%粒子被扰动
          %   %             perturb = perturb_std * randn(1, numsum);
          %   %             particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %             particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
          %   % 
          %   %             if i == 1
          %   %                 fprintf('第%03d代: 检测到停滞，执行扰动\n', gen);
          %   %             end
          %   %         end
          %   %     end
          %   %             particles_pos(i,:) = particles_pos(i,:) + perturb;
          %   %             particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
          %   % 
          %   %             if i == 1
          %   %                 fprintf('第%03d代: 检测到停滞，执行扰动\n', gen);
          %   %             end
          %   %         end
          %   %     end
          %   end