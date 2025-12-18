这是一个完整的MATLAB框架，用于系统性地评估和比较粒子群优化（PSO）及其改进版本（IPSO）在图像复原任务中的性能表现。该系统通过多维度指标分析、可视化对比和统计验证，为优化算法的性能评估提供了科研级的自动化解决方案。

### 核心功能：

1. **自动化对比实验**：自动加载预训练的PSO和IPSO优化结果，进行批量图像复原测试
2. **全面评估指标**：集成PSNR、MSE、SSIM三种图像质量评估指标
3. **多算法对比**：同时测试无优化BP（BPR）、PSO优化BP（PSOBPR）、IPSO优化BP（IPSOBPR）三种算法
4. **PK值分析框架**：创新性地引入PK（Performance Kappa）值定量分析算法相对性能提升
5. **科研级可视化**：自动生成符合学术论文要求的图表（600 DPI TIFF格式）
6. **数据自动化管理**：自动保存原始数据、中间结果和最终统计表格

### 技术特色：

- **科学实验设计**：每个图像进行100次复原测试确保统计稳定性
- **网络一致性**：测试阶段与优化阶段保持相同的BP神经网络结构
- **批量处理能力**：支持多张测试图像的并行处理和分析
- **完整数据流水线**：从原始图像到最终统计报告的端到端处理
- **可重复研究**：详细记录实验参数和运行环境

### 评估体系：

1. **基础性能指标**：PSNR、MSE、SSIM的最优值和平均值
2. **收敛性能分析**：PSO vs IPSO适应度曲线对比
3. **相对性能评估**：PK值定量分析算法提升程度
4. **统计显著性**：多图像测试的统计一致性验证

### 输出成果：

- **图像结果**：原始/模糊/各算法复原图像
- **分析图表**：适应度曲线、性能对比柱状图、PK值趋势图
- **数据表格**：符合论文格式的Table（Excel格式）
- **完整报告**：详细的统计分析和性能提升百分比

------

# English Description

## PSO vs IPSO Algorithm Performance Comparison System for Image Restoration

This is a comprehensive MATLAB framework for systematically evaluating and comparing the performance of Particle Swarm Optimization (PSO) and its Improved version (IPSO) in image restoration tasks. The system provides a research-grade automated solution for optimization algorithm performance assessment through multi-dimensional metric analysis, visual comparison, and statistical validation.

### Core Features:

1. **Automated Comparative Experiments**: Automatically loads pre-trained PSO and IPSO optimization results and conducts batch image restoration tests
2. **Comprehensive Evaluation Metrics**: Integrates three image quality assessment metrics: PSNR, MSE, and SSIM
3. **Multi-Algorithm Comparison**: Simultaneously tests three algorithms: unoptimized BP (BPR), PSO-optimized BP (PSOBPR), and IPSO-optimized BP (IPSOBPR)
4. **PK Value Analysis Framework**: Innovatively introduces PK (Performance Kappa) values for quantitative analysis of relative algorithm performance improvement
5. **Research-Grade Visualization**: Automatically generates publication-ready charts (600 DPI TIFF format)
6. **Automated Data Management**: Automatically saves raw data, intermediate results, and final statistical tables

### Technical Features:

- **Scientific Experimental Design**: 100 restoration tests per image to ensure statistical stability
- **Network Consistency**: Maintains the same BP neural network structure between testing and optimization phases
- **Batch Processing Capability**: Supports parallel processing and analysis of multiple test images
- **Complete Data Pipeline**: End-to-end processing from raw images to final statistical reports
- **Reproducible Research**: Detailed recording of experimental parameters and runtime environment

### Evaluation System:

1. **Basic Performance Metrics**: Best and average values of PSNR, MSE, and SSIM
2. **Convergence Performance Analysis**: PSO vs IPSO fitness curve comparison
3. **Relative Performance Assessment**: Quantitative analysis of algorithm improvement using PK values
4. **Statistical Significance**: Statistical consistency verification across multiple images

### Output Deliverables:

- **Image Results**: Original/blurred/restored images for each algorithm
- **Analysis Charts**: Fitness curves, performance comparison bar charts, PK value trend plots
- **Data Tables**: Table  in publication-ready Excel format
- **Complete Reports**: Detailed statistical analysis and performance improvement percentages

------

# GitHub Repository Summary

markdown

```
# PSO-vs-IPSO-Image-Restoration

A comprehensive MATLAB framework for comparative analysis of PSO and IPSO algorithms in image restoration tasks.

## 🎯 Key Features
- **Automated Testing**: Batch processing of multiple test images
- **Multi-Metric Evaluation**: PSNR, MSE, SSIM with best/mean values
- **Three-Algorithm Comparison**: BPR (baseline), PSOBPR, IPSOBPR
- **PK Analysis**: Quantitative relative performance assessment
- **Publication-Ready Outputs**: 600 DPI charts, Excel tables, formatted reports

## 📊 Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **MSE**: Mean Squared Error (lower is better)  
- **SSIM**: Structural Similarity (closer to 1 is better)
- **PK Values**: Relative performance improvement ratios

## 📁 Output Structure
```



results/
├── images/ # ORG, BLU, BPR, PSOBPR, IPSOBPR images
├── charts/ # Comparison charts (600 DPI TIFF)
├── tables/ # Excel tables
├── data/ # MAT files with raw results
└── reports/ # Statistical analysis reports

text

```
## 🚀 Quick Start
1. Load pre-trained PSO/IPSO results: `best90_0801_pop50_gen50_20251129.mat`
2. Set test parameters in config section
3. Run main script: `ipso_pso_comparison_main.m`
4. Find all results in `ipso/` directory

## 📈 Applications
- Academic research on optimization algorithms
- Performance benchmarking of PSO variants
- Image restoration algorithm development
- Teaching material for optimization courses

## 🔧 Requirements
- MATLAB R2018b+
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
- Neural Network Toolbox
```