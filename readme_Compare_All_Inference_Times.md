# Model Inference Time Comparison

## Overview
This MATLAB script `Compare_All_Inference_Times.m` compares the inference times of multiple image restoration models, including IPSOBPR, StandardDnCNN, StandardCNN, and DirectLearningDnCNN. It helps evaluate and quantify the runtime efficiency of these models for deployment considerations.

## Features
- Loads average inference times from precomputed `.mat` files for each model.
- Supports fallback to default values if timing files or variables are missing.
- Converts inference times to milliseconds for readability.
- Calculates relative speed ratios based on IPSOBPR inference time as the baseline.
- Displays a sorted comparison table of all models' inference times.
- Saves the comparison results to a `.mat` file (`All_Inference_Times_Comparison.mat`) for future use.

## Usage Instructions
1. Ensure inference time `.mat` files for each model are available in the working directory:
   - `IPSOBPR_inference_times.mat`
   - `StandardDnCNN_inference_times.mat`
   - `StandardCNN_inference_times.mat`
   - `DirectLearningDnCNN_inference_times.mat`
   
2. Run the MATLAB script:
    ```matlab
    Compare_All_Inference_Times
    ```

3. View the printed comparison table and relative speed information in the MATLAB command window.

4. The detailed comparison table is saved as `All_Inference_Times_Comparison.mat`.

## Outputs
- **Comparison Table** (`T`):
  - **Method**: Model name.
  - **InferenceTime_ms**: Average inference time in milliseconds.
  - **RelativeSpeed**: Speed factor relative to IPSOBPR (smaller is faster).

- The table is sorted by inference time ascending, making it easy to identify the fastest model.

## Dependencies
- MATLAB with basic file I/O support.
- MATLAB `.mat` files containing inference time variables (`mean_time` or `mean_cnn_time`).

## Notes
- Default inference time fallback values are used if corresponding `.mat` files or variables are missing.
- You can customize or extend inference time measurements by generating `.mat` files with measured timing data.

## Contact
For questions or contributions, please open an issue or submit a pull request.

---

Thank you for using this inference time comparison tool!





## 代码功能详细分析

`Compare_All_Inference_Times.m` 脚本用于对比多个不同模型的推理（Inference）时间，具体功能包括：

1. **加载模型推理时间数据**
   - 依次尝试加载包含各模型平均推理时间的 `.mat` 文件：
     - `IPSOBPR_inference_times.mat`：IPSO-BPR模型推理时间。
     - `StandardDnCNN_inference_times.mat`：标准DnCNN模型推理时间（文件中可能存在两个字段 `mean_cnn_time` 或 `mean_time`）。
     - `StandardCNN_inference_times.mat`：简单标准CNN推理时间。
     - `DirectLearningDnCNN_inference_times.mat`：直接学习DnCNN推理时间。
   - 若对应文件或字段缺失，使用默认推理时间值（或0表示未找到）。
2. **构建对比表格**
   - 将读取或默认的推理时间（单位秒）转换成毫秒（ms）。
     -计算以IPSOBPR模型时间为基准的相对速度（倍数）。
   - 生成包含方法名称、推理时间（ms）、相对速度的表格，并按推理时间升序排序。
3. **结果展示与输出**
   - 在命令窗口打印推理时间对比的完整表格。
   - 按顺序输出各模型相对于IPSOBPR的速度倍数及对应时间。
   - 保存所有对比结果到 `All_Inference_Times_Comparison.mat` 文件，方便后续使用和分析。

本脚本实用且直观，可快速量化各种模型推理效率，为性能优化和实际部署提供参考依据。



