# #Improved Particle Swarm Optimization with Exponentially Adaptive Inertia Weights and Late-Iteration Perturbation for Backpropagation Neural Network-Based Image Restoration 



———————————————————————————————————————————

***5.3 IPSOBPR algorithm***

**（5.3.3 Quantitative Performance Evaluation）**

1. FIGURE 7.	Comparison of algorithm convergence.

2. FIGURE 8.	Comparison of the best fitness values.

3. TABLE 3 Comparative Performance Metrics for the Baseline and IPSOBPNN

Only_2PSO_PK_multi_run_all_ttest2.m

|--PSO_standard.m

|--PSO_improved_p.m

Table 4. Experimental comparison of different metaheuristic algorithms



**（5.3.6 Comparison with Classical and State-of-the-Art Metaheuristic Algorithms）**

Fig 9. Comparison of the Convergence Curves

IPSO_vs_StateOfArt.m



***Paper 5.4 COMPARISON OF THE EFFECTS OF FIVE IMAGE RESTORATION ALGORITHMS***

**（5.4.1 Peak signal-to-noise ratio (PSNR) comparisons）**

TABLE 5 Comprehensive PSNR Comparisons of Five Restoration Algorithms (dB)

Only_vwnr.m
|--Cloumn：WFR		

Only_clsd.m
|--Cloumn：CLSR

Only_3BPR_PK_ipso_all_valid_dynamic.m
|--Cloumn：BPR	PSOBPR	IPSOBPR

The images in the connect directory are experimental results that will later be combined into FIGURE 10. 

**（5.4.2 Visual analysis）**

FIGURE 10.	Comparison of algorithm restoration results.

connect_img_scale_v2.m

TABLE 6 Objective Evaluation of Different Restoration Algorithms（5.4.3 Quantitative performance analysis）

readme_Only_res_all_printv2

**（5.4.4 Comparison with Deep Learning Approaches for Image Restoration）**

Fig 11. Architecture-Specific Average PSNR Comparison
Fig 12. Architecture-Specific Average SSIM Comparison

Only_CNNvsDnCNN.m(Comparison results saved to: DnCNN_Comparisons\all_config_comparison_20260210_204956.xlsx)（No ipsobpr）
draw_CNNDnCNNPK_comparison.m（Input:all_config_comparison_20260210_113930ipso paper2 user.xlsx）（Add ipsobpr）

