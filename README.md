# ipso_bpr
ipso bpr(matlab)

———————————————————————————————————————————

***Paper 5.3.3 Quantitative Performance Evaluation***

1. FIGURE 7.	Comparison of algorithm convergence.

2. FIGURE 8.	Comparison of the best fitness values.

3. TABLE 3 Comparative Performance Metrics for the Baseline and IPSOBPNN

Only_2PSO_PK_multi_run_all_ttest2.m

|--PSO_standard.m

|--PSO_improved_p.m



***Paper 5.4 COMPARISON OF THE EFFECTS OF FIVE IMAGE RESTORATION ALGORITHMS***

1. TABLE 5 Comprehensive PSNR Comparisons of Five Restoration Algorithms (dB)（5.4.1 Peak signal-to-noise ratio (PSNR) comparisons）

Only_vwnr.m
|--Cloumn：WFR		

Only_clsd.m
|--Cloumn：CLSR

Only_3BPR_PK_ipso_all_valid_dynamic.m
|--Cloumn：BPR	PSOBPR	IPSOBPR

The images in the connect directory are experimental results that will later be combined into FIGURE 10. 


2. FIGURE 10.	Comparison of algorithm restoration results.（5.4.2 Visual analysis）

connect_img_scale_v2.m


3. TABLE 6 Objective Evaluation of Different Restoration Algorithms（5.4.3 Quantitative performance analysis）

readme_Only_res_all_printv2

4.Fig 11. Architecture-Specific Average PSNR Comparison
Fig 12. Architecture-Specific Average SSIM Comparison（5.4.4 Comparison with Deep Learning Approaches for Image Restoration）

Only_CNNvsDnCNN.m(Comparison results saved to: DnCNN_Comparisons\all_config_comparison_20260210_204956.xlsx)（No ipsobpr）
draw_CNNDnCNNPK_comparison.m（Input:C:\ipso_bpr_v3\DnCNN_Comparisons\all_config_comparison_20260210_113930ipso paper2 user.xlsx）（add ipsobpr）
Data in all_config_comparison_20260210_113930ipso paper2 user.xlsx：
Config              	UseResidual	Mean_PSNR	Mean_SSIM
StandardDnCNN        	TRUE	  33.06140344	0.962386673
IPSOBPR	              FALSE  	31.1968979	0.953913935
DirectLearningDnCNN	  FALSE  	29.92810544	0.936897033
StandardCNN	          FALSE  	27.51528768	0.903577047

