%% 附录：自定义函数（需与主代码放在同一目录下）
% 1. generate_training_data.m：生成BPNN训练数据（3×3滑动窗口）
function [P_Matrix, T_Matrix] = generate_training_data(image_degraded, image_clear, inputnum)
    [h, w] = size(image_degraded);
    data_len = (h-2)*(w-2);  % 滑动窗口数量（边缘像素不参与，避免边界效应）
    P_Matrix = zeros(inputnum, data_len);  % 输入矩阵（9×N）
    T_Matrix = zeros(1, data_len);         % 目标矩阵（1×N，清晰图像中心像素）
    t = 1;
    
    % 遍历图像，提取3×3窗口
    for i = 2:h-1
        for j = 2:w-1
            % 提取退化图像的3×3窗口作为输入
            P_Matrix(1,t) = image_degraded(i-1,j-1);
            P_Matrix(2,t) = image_degraded(i-1,j);
            P_Matrix(3,t) = image_degraded(i-1,j+1);
            P_Matrix(4,t) = image_degraded(i,j-1);
            P_Matrix(5,t) = image_degraded(i,j);
            P_Matrix(6,t) = image_degraded(i,j+1);
            P_Matrix(7,t) = image_degraded(i+1,j-1);
            P_Matrix(8,t) = image_degraded(i+1,j);
            P_Matrix(9,t) = image_degraded(i+1,j+1);
            % 提取清晰图像的中心像素作为目标
            T_Matrix(1,t) = image_clear(i,j);
            t = t + 1;
        end
    end
end

% % 2. cal_fitness.m：适应度函数（BPNN预测误差计算）
% function fitness = cal_fitness(x, inputnum, hiddennum, outputnum, net, inputn, outputn)
%     % 解码粒子位置为BPNN权值与阈值
%     w1 = x(1:inputnum*hiddennum);                  % 输入-隐藏层权值
%     b1 = x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);  % 隐藏层阈值
%     w2 = x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);  % 隐藏-输出层权值
%     b2 = x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:end);  % 输出层阈值
% 
%     % 赋值权值与阈值到BPNN
%     net.iw{1,1} = reshape(w1, hiddennum, inputnum);
%     net.lw{2,1} = reshape(w2, outputnum, hiddennum);
%     net.b{1} = reshape(b1, hiddennum, 1);
%     net.b{2} = b2;
% 
%     % 计算BPNN预测误差（适应度=归一化总误差，越小越好）
%     pred_out = sim(net, inputn);
%     fitness = 0.1 * sum(sum(abs(pred_out - outputn)));  % 与原代码误差计算方式一致，保证对比公平性
% end
% 
% % 3. PSO_standard.m：标准PSO算法函数
% % 3. PSO_standard.m：标准PSO算法函数（修复返回参数赋值问题）
% function [bestchrom, bestfitness, trace_best] = PSO_standard(sizepop, maxgen, numsum, fobj, c1, c2, w, v_max, v_min, pos_max, pos_min)
%     % 1. 初始化粒子群（确保位置和速度在合法范围内）
%     particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);  % 粒子位置
%     particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);        % 粒子速度
% 
%     % 2. 初始化个体最优与全局最优（首次计算适应度）
%     pbest_pos = particles_pos;  % 初始个体最优位置=当前位置
%     pbest_fit = zeros(sizepop, 1);
%     for i = 1:sizepop
%         pbest_fit(i) = fobj(particles_pos(i,:));  % 计算每个粒子的初始适应度
%     end
%     [gbest_fit, gbest_idx] = min(pbest_fit);  % 全局最优适应度（最小误差）
%     gbest_pos = pbest_pos(gbest_idx, :);      % 全局最优位置
% 
%     % 3. 初始化适应度曲线记录数组
%     trace_best = zeros(maxgen, 1);
%     trace_best(1) = gbest_fit;  % 记录第1代（初始）的全局最优适应度
% 
%     % 4. PSO迭代主循环（从第2代开始更新）
%     for gen = 2:maxgen
%         % 4.1 遍历每个粒子更新速度和位置
%         for i = 1:sizepop
%             % （1）更新速度（固定惯性权重w，平衡惯性、个体学习、全局学习）
%             r1 = rand();  % [0,1]随机数，增加搜索随机性
%             r2 = rand();
%             particles_vel(i,:) = w * particles_vel(i,:) + ...
%                                  c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
%                                  c2*r2*(gbest_pos - particles_pos(i,:));
% 
%             % （2）速度约束（防止速度过大导致位置溢出）
%             particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
% 
%             % （3）更新位置
%             particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
% 
%             % （4）位置约束（确保粒子在BPNN参数合法范围内）
%             particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
% 
%             % （5）更新个体最优（若当前位置适应度更优）
%             current_fit = fobj(particles_pos(i,:));
%             if current_fit < pbest_fit(i)
%                 pbest_fit(i) = current_fit;
%                 pbest_pos(i,:) = particles_pos(i,:);
%             end
%         end
% 
%         % 4.2 更新全局最优（遍历所有个体最优，取最优值）
%         [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
%         if current_gbest_fit < gbest_fit
%             gbest_fit = current_gbest_fit;
%             gbest_pos = pbest_pos(current_gbest_idx, :);
%         end
% 
%         % 4.3 记录当前代的全局最优适应度（填充适应度曲线）
%         trace_best(gen) = gbest_fit;
%     end
% 
%     % 5. 函数结束前明确赋值返回参数（核心修复：确保每个输出参数都有值）
%     bestchrom = gbest_pos;       % 全局最优位置（对应BPNN最优参数）
%     bestfitness = gbest_fit;     % 全局最优适应度（最小BPNN预测误差）
%     % trace_best已在循环中完整填充，无需额外赋值
% end
% 
% % PSO_improved.m：改进PSO（IPSO）算法函数（含自适应惯性权重+高斯扰动）
% % 输入参数（14个，与主程序第101行调用一致）：
% % sizepop-粒子群规模, maxgen-最大迭代次数, numsum-优化维度, fobj-适应度函数,
% % c1-个体学习因子, c2-全局学习因子, w_init-初始惯性权重, w_final-最终惯性权重,
% % v_max-最大速度, v_min-最小速度, pos_max-位置上限, pos_min-位置下限,
% % perturb_trigger_ratio-扰动触发比例, perturb_std-扰动标准差
% % 输出参数（3个）：bestchrom-全局最优位置, bestfitness-全局最优适应度, trace_best-适应度曲线
% function [bestchrom, bestfitness, trace_best] = PSO_improved(sizepop, maxgen, numsum, fobj, c1, c2, w_init, w_final, v_max, v_min, pos_max, pos_min, perturb_trigger_ratio, perturb_std)
%     % 1. 初始化粒子群（位置、速度均在合法范围）
%     particles_pos = pos_min + rand(sizepop, numsum) * (pos_max - pos_min);
%     particles_vel = v_min + rand(sizepop, numsum) * (v_max - v_min);
% 
%     % 2. 初始化个体最优与全局最优（首次计算适应度）
%     pbest_pos = particles_pos;
%     pbest_fit = zeros(sizepop, 1);
%     for i = 1:sizepop
%         pbest_fit(i) = fobj(particles_pos(i,:));  % 计算初始适应度
%     end
%     [gbest_fit, gbest_idx] = min(pbest_fit);
%     gbest_pos = pbest_pos(gbest_idx, :);
% 
%     % 3. 初始化适应度曲线记录
%     trace_best = zeros(maxgen, 1);
%     trace_best(1) = gbest_fit;
% 
%     % 4. IPSO迭代主循环
%     for gen = 2:maxgen
%         % 4.1 自适应惯性权重更新（线性递减：从w_init到w_final）
%         w = w_init - (w_init - w_final) * (gen / maxgen);
% 
%         % 4.2 遍历粒子更新速度、位置与扰动
%         for i = 1:sizepop
%             % （1）速度更新
%             r1 = rand();
%             r2 = rand();
%             particles_vel(i,:) = w * particles_vel(i,:) + ...
%                                  c1*r1*(pbest_pos(i,:) - particles_pos(i,:)) + ...
%                                  c2*r2*(gbest_pos - particles_pos(i,:));
%             % （2）速度约束
%             particles_vel(i,:) = max(min(particles_vel(i,:), v_max), v_min);
% 
%             % （3）位置更新
%             particles_pos(i,:) = particles_pos(i,:) + particles_vel(i,:);
%             % （4）位置约束
%             particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);
% 
%             % （5） late-iteration高斯扰动（迭代超70%时触发）
%             if gen > perturb_trigger_ratio * maxgen
%                 perturb = perturb_std * randn(1, numsum);  % 高斯扰动（均值0，标准差perturb_std）
%                 particles_pos(i,:) = particles_pos(i,:) + perturb;
%                 particles_pos(i,:) = max(min(particles_pos(i,:), pos_max), pos_min);  % 重新约束位置
%             end
% 
%             % （6）更新个体最优
%             current_fit = fobj(particles_pos(i,:));
%             if current_fit < pbest_fit(i)
%                 pbest_fit(i) = current_fit;
%                 pbest_pos(i,:) = particles_pos(i,:);
%             end
%         end
% 
%         % 4.3 更新全局最优
%         [current_gbest_fit, current_gbest_idx] = min(pbest_fit);
%         if current_gbest_fit < gbest_fit
%             gbest_fit = current_gbest_fit;
%             gbest_pos = pbest_pos(current_gbest_idx, :);
%         end
% 
%         % 4.4 记录当前代最优适应度
%         trace_best(gen) = gbest_fit;
%     end
% 
%     % 5. 赋值输出参数（确保每个输出都有值）
%     bestchrom = gbest_pos;
%     bestfitness = gbest_fit;
% end
