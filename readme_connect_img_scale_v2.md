# 中文描述

## IPSO算法结果自动排版工具

这是一个用于自动排版和可视化IPSO（改进粒子群优化）算法图像处理结果的MATLAB工具。该脚本能够智能地整理、处理和组合多个实验图像，生成专业美观的对比图。

### 主要功能：

1. **批量图像处理**：自动读取指定目录下的TIFF图像文件
2. **智能分组**：按实验编号和图像类型自动分组排序
3. **尺寸统一化**：将所有图像调整为统一尺寸（默认88×88像素）
4. **局部放大效果**：为每幅图像添加局部放大展示，突出细节对比
5. **自定义布局**：支持灵活配置图像类型显示顺序
6. **自动化标签**：从文件名提取日期代码作为行标签，自动添加列标签
7. **高质量输出**：支持1200 DPI高分辨率TIFF格式输出
8. **时间戳管理**：输出文件自动添加时间戳，避免文件覆盖

### 技术特点：

- 使用中心裁剪或调整大小算法处理不同尺寸的原始图像
- 为ORG图像显示选择框和放大区域，其他图像只显示放大区域
- 添加细白色边框增强视觉效果
- 支持灰度图和RGB图混合处理
- 提供主图和详细子图两种输出格式

### 应用场景：

- 学术论文中的算法性能对比展示
- 实验报告的自动化结果整理
- 算法评估和性能分析
- 科研项目的演示材料制作

------

# English Description

## IPSO Algorithm Results Auto-Layout Tool

This is a MATLAB tool for automatically arranging and visualizing IPSO (Improved Particle Swarm Optimization) algorithm image processing results. The script intelligently organizes, processes, and combines multiple experimental images to generate professional and aesthetically pleasing comparison charts.

### Key Features:

1. **Batch Image Processing**: Automatically reads TIFF image files from specified directories
2. **Smart Grouping**: Automatically groups and sorts images by experiment number and image type
3. **Size Standardization**: Resizes all images to uniform dimensions (default 88×88 pixels)
4. **Local Zoom Effects**: Adds local magnification display to each image to highlight detail comparison
5. **Custom Layout**: Supports flexible configuration of image type display order
6. **Auto-labeling**: Extracts date codes from filenames as row labels, automatically adds column labels
7. **High-Quality Output**: Supports 1200 DPI high-resolution TIFF format output
8. **Timestamp Management**: Output files automatically include timestamps to prevent file overwriting

### Technical Features:

- Uses center cropping or resizing algorithms to handle original images of different sizes
- Displays selection boxes and zoom areas for ORG images, only zoom areas for other images
- Adds thin white borders to enhance visual effects
- Supports mixed processing of grayscale and RGB images
- Provides two output formats: main combined image and detailed subplot image

### Application Scenarios:

- Algorithm performance comparison in academic papers
- Automated results organization for experimental reports
- Algorithm evaluation and performance analysis
- Presentation material creation for research projects



# IPSO-Algorithm-Results-Layout

MATLAB tool for automated layout and visualization of IPSO algorithm image processing results.

## Features
- 🔄 **Batch processing** of TIFF images
- 📊 **Smart grouping** by experiment number and image type
- 📐 **Size standardization** to uniform dimensions
- 🔍 **Local zoom effects** for detail highlighting
- 🎨 **Custom layout** with flexible image order
- 🏷️ **Auto-labeling** from filename patterns
- 📈 **High-resolution output** (1200 DPI TIFF)
- ⏰ **Timestamped filenames** to prevent overwrites

## Use Cases
- Academic paper figure generation
- Experimental result visualization
- Algorithm performance comparison
- Research presentation materials

## Requirements
- MATLAB R2018b or later
- Image Processing Toolbox

## Quick Start
1. Place your TIFF images in `ipso\connect` folder
2. Run `ipso_layout_script.m`
3. Get professional comparison charts automatically!

## File Naming Convention
Images should follow: `{experiment_number}_{image_type}_{date_code}.tif`
Example: `1_ORG_0801.tif`, `1_BLU_0801.tif`