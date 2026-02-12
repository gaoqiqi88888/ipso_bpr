%********************************************************************
% 生成LaTeX格式对比表格
%********************************************************************
function generate_latex_table(Comparison_Table, algorithms, num_images)
filename = 'comparison_results/comparison_table.tex';
fid = fopen(filename, 'w');

% 写入文件头
fprintf(fid, '\\begin{table}[htbp]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Comparison of IPSO with State-of-the-art Algorithms}\n');
fprintf(fid, '\\label{tab:ipso_comparison}\n');
fprintf(fid, '\\begin{tabular}{lccccccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Image & Algorithm & Best & Mean & Std & Time(s) & Rank & IR\\%% \\\\\n');
fprintf(fid, '\\midrule\n');

% 写入数据
for img = 1:num_images
    for a = 1:length(algorithms)
        row_idx = (img-1)*length(algorithms) + a + 1;
        
        % 格式化输出
        if a == 1
            img_name = strrep(Comparison_Table{row_idx, 1}, '_', '\_');
            fprintf(fid, '\\multirow{%d}{*}{%s} ', length(algorithms), img_name);
        else
            fprintf(fid, ' ');
        end
        
        alg = Comparison_Table{row_idx, 2};
        best = Comparison_Table{row_idx, 3};
        meanv = Comparison_Table{row_idx, 4};
        stdv = Comparison_Table{row_idx, 6};
        time = Comparison_Table{row_idx, 9};
        rank = Comparison_Table{row_idx, 8};
        ir = Comparison_Table{row_idx, 11};
        sig = Comparison_Table{row_idx, 12};
        
        fprintf(fid, '& %s & %.2e & %.2e & %.2e & %.2f & %.1f & %.2f%s \\\\\n', ...
                alg, best, meanv, stdv, time, rank, ir, sig);
    end
    
    if img < num_images
        fprintf(fid, '\\midrule\n');
    end
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);
fprintf('LaTeX table saved to: %s\n', filename);
end