function generate_detailed_report(results, param_combinations)
    % 生成详细的文本报表
    
    report_filename = 'ipso/perturbation_parameter_tuning_report.txt';
    fid = fopen(report_filename, 'w', 'n', 'UTF-8');
    
    if fid == -1
        error('无法创建报表文件');
    end
    
    fprintf(fid, 'IPSO扰动参数调优实验详细报表\n');
    fprintf(fid, '==================================================\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
    num_images = length(fieldnames(results));
    num_combinations = size(param_combinations, 1);
    
    % 1. 参数组合列表
    fprintf(fid, '一、测试的参数组合:\n');
    fprintf(fid, '编号\t触发比例\t扰动强度\t组合描述\n');
    fprintf(fid, '----\t--------\t--------\t----------\n');
    
    for i = 1:num_combinations
        trigger = param_combinations(i,1);
        stddev = param_combinations(i,2);
        
        if trigger == 1.0 && stddev == 0.0
            desc = '无扰动对照（基准）';
        elseif trigger == 1.0 && stddev > 0
            desc = '不触发但有扰动强度';
        elseif trigger == 0.0
            desc = '全程扰动';
        else
            desc = sprintf('%.0f%%迭代后扰动', trigger*100);
        end
        
        fprintf(fid, '%2d\t%.2f\t\t%.3f\t\t%s\n', i, trigger, stddev, desc);
    end
    
    fprintf(fid, '\n\n');
    
    % 2. 总体性能排名
    fprintf(fid, '二、总体性能排名（按平均改进率）:\n');
    fprintf(fid, '排名\t组合\t\t平均改进率\t最佳改进率\t中位数改进率\t稳定性\n');
    fprintf(fid, '----\t----\t\t----------\t----------\t------------\t------\n');
    
    % 计算各组合的平均性能
    avg_mean_improvement = zeros(num_combinations, 1);
    avg_best_improvement = zeros(num_combinations, 1);
    avg_median_improvement = zeros(num_combinations, 1);
    std_of_improvement = zeros(num_combinations, 1);
    
    for i = 1:num_combinations
        all_mean_improv = [];
        all_best_improv = [];
        all_median_improv = [];
        
        for img_idx = 1:num_images
            img_field = sprintf('image_%d', img_idx);
            if isfield(results, img_field)
                all_mean_improv = [all_mean_improv; results.(img_field).improvement_mean(i)];
                all_best_improv = [all_best_improv; results.(img_field).improvement_best(i)];
                all_median_improv = [all_median_improv; results.(img_field).improvement_median(i)];
            end
        end
        
        avg_mean_improvement(i) = mean(all_mean_improv);
        avg_best_improvement(i) = mean(all_best_improv);
        avg_median_improvement(i) = mean(all_median_improv);
        std_of_improvement(i) = std(all_mean_improv);
    end
    
    % 排序
    [sorted_avg, sorted_idx] = sort(avg_mean_improvement, 'descend');
    
    for rank = 1:num_combinations
        idx = sorted_idx(rank);
        combo_str = sprintf('R%.1f-S%.2f', param_combinations(idx,1), param_combinations(idx,2));
        
        fprintf(fid, '%2d\t%s\t%.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.3f\n', ...
                rank, combo_str, ...
                sorted_avg(rank), avg_best_improvement(idx), ...
                avg_median_improvement(idx), std_of_improvement(idx));
    end
    
    fprintf(fid, '\n\n');
    
    % 3. 最优参数推荐
    fprintf(fid, '三、最优参数推荐:\n');
    best_idx = sorted_idx(1);
    best_trigger = param_combinations(best_idx,1);
    best_std = param_combinations(best_idx,2);
    
    fprintf(fid, '推荐组合: 触发比例 = %.1f, 扰动强度 = %.3f\n', best_trigger, best_std);
    fprintf(fid, '平均改进率: %.2f%% (相对于无扰动基准)\n', sorted_avg(1));
    
    if best_trigger == 1.0 && best_std == 0.0
        fprintf(fid, '结论: 无需扰动，标准PSO效果最好\n');
    else
        fprintf(fid, '结论: IPSO扰动策略有效，提升明显\n');
    end
    
    fprintf(fid, '\n\n');
    
    % 4. 各图像详细结果
    fprintf(fid, '四、各图像详细结果:\n\n');
    
    for img_idx = 1:num_images
        img_field = sprintf('image_%d', img_idx);
        if ~isfield(results, img_field)
            continue;
        end
        
        fprintf(fid, '图像 %d:\n', img_idx);
        fprintf(fid, '参数组合\t\t平均适应度\t最佳适应度\t平均改进率\t最佳改进率\n');
        fprintf(fid, '--------\t\t----------\t----------\t----------\t----------\n');
        
        % 找出该图像的最佳组合
        [~, best_for_image] = max(results.(img_field).improvement_mean);
        
        for i = 1:num_combinations
            combo_str = sprintf('R%.1f-S%.2f', param_combinations(i,1), param_combinations(i,2));
            
            if i == best_for_image
                combo_str = ['*' combo_str '*'];
            end
            
            fprintf(fid, '%s\t%.4f\t\t%.4f\t\t%.2f%%\t\t%.2f%%\n', ...
                    combo_str, ...
                    results.(img_field).mean_fitness(i), ...
                    min(results.(img_field).best_fitness(i,:)), ...
                    results.(img_field).improvement_mean(i), ...
                    results.(img_field).improvement_best(i));
        end
        
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    
    fprintf('详细报表已保存至: %s\n', report_filename);
end