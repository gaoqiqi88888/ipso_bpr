# CNNvsDnCNN Image Restoration - Multi-Config Test

## Project Overview
This project implements a multi-configuration comparative test of convolutional neural networks (CNN) for image denoising based on MATLAB. It includes three types of models: a simple CNN, direct learning DnCNN, and residual learning DnCNN. Using the DIV2K grayscale image dataset, the project trains and validates these models to compare the impact of network architecture and residual learning on denoising performance. Evaluation metrics include PSNR, SSIM, and MSE.

## Key Features
- Denoising training and testing using the DIV2K grayscale image dataset.
- Supports multiple CNN configurations:
  - Standard simple CNN
  - DnCNN with direct learning (non-residual)
  - Standard DnCNN with residual learning
- Automated training, validation, and model saving.
- Calculates and saves PSNR, SSIM, and MSE metrics for each validation image.
- Generates and saves Excel reports comparing model performances.
- Outputs sample denoised images for visual comparisons.

## Dataset
- **Training set**: `DIV2K_train_HR_90_gray` (high-quality images), `DIV2K_train_BLUR_90_gray` (noisy images)
- **Validation set**: `DIV2K_valid_HR_90_gray`, `DIV2K_valid_BLUR_90_gray`

Please prepare the datasets in advance and place them under the specified folder paths (supports `.tif` grayscale images).

## Environment Requirements
- MATLAB R2019b or later (compatible with newer versions)
- Deep Learning Toolbox
- Image Processing Toolbox

## Usage Instructions

1. Prepare the training and validation datasets and set the corresponding paths in the script.
2. Run the `Only_CNNvsDnCNN.m` script.
3. The program will automatically train, validate, compute metrics, and generate visual results and Excel reports.
4. Outputs will be saved in timestamp-named folders, including model weights, metrics, and denoised images.

## Directory Structure
- `Only_CNNvsDnCNN.m`: Core testing script
- `DIV2K_train_HR_90_gray/`: Training high-quality images
- `DIV2K_train_BLUR_90_gray/`: Training noisy images
- `DIV2K_valid_HR_90_gray/`: Validation high-quality images
- `DIV2K_valid_BLUR_90_gray/`: Validation noisy images
- Output directories (auto-generated):
  - `DnCNN_Test_<ConfigName>_<Timestamp>/models/`: Trained models
  - `DnCNN_Test_<ConfigName>_<Timestamp>/metrics/`: Excel metrics
  - `DnCNN_Test_<ConfigName>_<Timestamp>/results/`: Denoised images
  - `DnCNN_Comparisons/`: Summary and comparison charts for all configurations

## Results
- Training loss and PSNR curves for each model.
- Per-image PSNR, SSIM, and MSE on the validation set.
- Overall performance comparison table and ranking.
- Visual examples of denoised images.
- Bar charts comparing PSNR, SSIM, and MSE across models.

## Reference
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. *IEEE Transactions on Image Processing*, 26(7), 3142-3155.

## Contact
For questions or further information, please contact the author.

---

Contributions, forks, and pull requests are welcome!



# DnCNN Image Restoration - Multi-Config Test

## 项目简介
本项目基于 MATLAB 实现了图像去噪领域经典卷积神经网络 DnCNN 的多配置对比测试，包括简单CNN、DnCNN的直接学习及残差学习三种模型。通过在DIV2K灰度图去噪数据集上的训练与验证，比较不同结构及残差策略对去噪性能的影响，指标涵盖PSNR、SSIM和MSE。

## 主要功能
- 使用DIV2K灰度图像数据集进行去噪训练和测试。
- 支持多种CNN配置：
  - 标准简单CNN
  - DnCNN直接学习（非残差）
  - 标准DnCNN残差学习。
- 自动训练、验证流程及模型保存。
- 计算并存储针对验证集的每张图像PSNR、SSIM、MSE指标。
- 生成并保存各模型的性能对比Excel及条形图。
- 输出部分去噪结果图像，支持可视化效果对比。

## 数据集
- **训练集**: `DIV2K_train_HR_90_gray`（高质量图像）、`DIV2K_train_BLUR_90_gray`（含噪图像）
- **验证集**: `DIV2K_valid_HR_90_gray`、`DIV2K_valid_BLUR_90_gray`
  

数据集需提前准备，并放在对应路径下（支持 `.tif` 格式灰度图）。

## 环境依赖
- MATLAB R2019b 及以上版本（兼容较新版本）
- Deep Learning Toolbox
- Image Processing Toolbox

## 使用说明

1. 确保准备好训练与验证数据集，并设置好代码中的路径参数。
2. 运行 `Only_CNNvsDnCNN.m` 脚本。
3. 程序自动完成训练，验证，指标计算，生成可视化结果和Excel报告。
4. 结果保存在以时间戳命名的文件夹中，包含模型权重，指标和去噪图像。

## 目录结构说明
- `Only_CNNvsDnCNN.m`: 核心测试脚本
- `DIV2K_train_HR_90_gray/`: 训练集高质量图像
- `DIV2K_train_BLUR_90_gray/`: 训练集含噪图像
- `DIV2K_valid_HR_90_gray/`: 验证集高质量图像
- `DIV2K_valid_BLUR_90_gray/`: 验证集含噪图像
- 输出目录（自动创建）：
  - `DnCNN_Test_<ConfigName>_<Timestamp>/models/`：训练模型
  - `DnCNN_Test_<ConfigName>_<Timestamp>/metrics/`：指标Excel
  - `DnCNN_Test_<ConfigName>_<Timestamp>/results/`：去噪结果图像
  - `DnCNN_Comparisons/`：所有配置总结及对比图表

## 结果展示
- 各模型的训练损失及PSNR变化曲线。
- 验证集上每图像PSNR、SSIM、MSE。
- 各模型整体指标对比表及排序。
- 去噪图像增强效果示例。
- 性能比较条形图（PSNR/SSIM/MSE三视角）。

## 参考文献
- Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7), 3142-3155.

## 联系方式
如有疑问或希望获取更多信息，请联系作者。

---

欢迎大家fork及贡献代码！