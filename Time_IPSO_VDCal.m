% 在 MATLAB 命令窗口运行
load('D:\matlab dev\ipso_bpr_v3\ipso\dataset\best90_0801_pop50_gen50_20260211');
[~, IPSO_best_idx] = min(IPSO_bestfitness);
ipso_params = IPSO_bestchrom(IPSO_best_idx, :);
fprintf('IPSO 参数维度: %d\n', length(ipso_params));