import numpy as np
from PIL import Image
# 注意：如果你选择使用 zoom 进行缩放，需要安装 scipy (pip install scipy)
# from scipy.ndimage import zoom 
import os # 用于检查文件是否存在

def extract_spectral_features(image_path, output_dim=None):
    """
    提取图像的频谱特征（对数幅度谱），进行中心化、对数变换、可选的尺寸调整和归一化。

    参数:
        image_path (str): 图像文件路径。
        output_dim (tuple, optional): 输出频谱的期望尺寸 (height, width)。
                                     如果为 None，则不调整尺寸。
                                     如果指定尺寸小于原始尺寸，则进行中心裁剪。
                                     如果指定尺寸大于原始尺寸，则进行缩放（需要 scipy）。
                                     默认为 None。

    返回:
        numpy.ndarray or None: 处理后的频谱幅度图 (归一化到 [0, 1])。
                              如果图像加载失败或路径无效，则返回 None。
    """
    # --- 1. 加载和预处理图像 ---
    if not os.path.exists(image_path) or not os.path.isfile(image_path):
        print(f"Warning: Image path not found or not a file: {image_path}")
        return None
        
    try:
        # 加载图像并转换为灰度图
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # --- 2. 计算傅里叶变换和幅度谱 ---
    # 对图像进行二维快速傅里叶变换 (2D FFT)
    f_transform = np.fft.fft2(image_array)
    
    # 将零频率分量（直流分量）移到频谱中心
    f_shift = np.fft.fftshift(f_transform)
    
    # 计算幅度谱 |F(u,v)|
    magnitude_spectrum = np.abs(f_shift)
    
    # --- 3. 对数变换 ---
    # 使用 log1p (log(1 + x)) 增强低幅度频率成分并压缩动态范围，更稳定
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)

    # --- 4. 尺寸调整 (可选) ---
    processed_spectrum = log_magnitude_spectrum
    if output_dim is not None and isinstance(output_dim, tuple) and len(output_dim) == 2:
        current_rows, current_cols = log_magnitude_spectrum.shape
        target_rows, target_cols = output_dim

        if target_rows == current_rows and target_cols == current_cols:
            # 尺寸相同，无需操作
            pass
        elif target_rows <= current_rows and target_cols <= current_cols:
            # --- 中心裁剪 (如果目标尺寸小于等于当前尺寸) ---
            center_row, center_col = current_rows // 2, current_cols // 2
            start_row = center_row - target_rows // 2
            end_row = start_row + target_rows
            start_col = center_col - target_cols // 2
            end_col = start_col + target_cols
            
            # 确保索引在边界内（虽然理论上中心裁剪不会越界）
            start_row, end_row = max(0, start_row), min(current_rows, end_row)
            start_col, end_col = max(0, start_col), min(current_cols, end_col)
            
            processed_spectrum = log_magnitude_spectrum[start_row:end_row, start_col:end_col]
            
            # 如果由于整除导致尺寸差1像素，可能需要微调或用特定库保证精确尺寸
            # 这里假设整除后的裁剪尺寸足够接近目标
            if processed_spectrum.shape != output_dim:
                 # 可以添加警告或使用更精确的裁剪/填充方法，或者使用插值缩放
                 print(f"Warning: Cropped spectrum shape {processed_spectrum.shape} differs slightly from target {output_dim}. Using interpolation instead.")
                 # 退回到缩放方法（需要scipy）
                 try:
                     from scipy.ndimage import zoom
                     zoom_factors = (target_rows / current_rows, target_cols / current_cols)
                     processed_spectrum = zoom(log_magnitude_spectrum, zoom_factors, order=1) # order=1 for bilinear interpolation
                 except ImportError:
                     print("Error: Scipy is required for resizing when target dimensions require scaling. Please install scipy.")
                     # 或者返回裁剪结果，或者返回None
                     # return processed_spectrum # 返回裁剪结果
                     return None # 返回None，因为无法精确满足要求
        
        else:
            # --- 缩放 (如果目标尺寸大于当前尺寸，或混合情况) ---
            # 注意：这需要安装 scipy (`pip install scipy`)
            try:
                from scipy.ndimage import zoom
                zoom_factors = (target_rows / current_rows, target_cols / current_cols)
                # 使用双线性插值 (order=1) 或其他阶数
                processed_spectrum = zoom(log_magnitude_spectrum, zoom_factors, order=1) 
            except ImportError:
                print("Error: Scipy is required for resizing the spectrum. Please install scipy.")
                # 如果没有 scipy，可以选择返回 None 或原始未调整尺寸的谱图
                return None # 或者 return log_magnitude_spectrum
            except Exception as e:
                 print(f"Error during spectrum zoom: {e}")
                 return None

    # --- 5. 归一化 ---
    # 使用 Min-Max 归一化将值缩放到 [0, 1] 范围
    min_val = np.min(processed_spectrum)
    max_val = np.max(processed_spectrum)
    
    if max_val > min_val:
        normalized_spectrum = (processed_spectrum - min_val) / (max_val - min_val)
    else:
        # 如果所有值都相同（例如，全黑图像的频谱），则设为0或0.5
        normalized_spectrum = np.zeros_like(processed_spectrum) 

    return normalized_spectrum

# --- 示例用法 ---
if __name__ == '__main__':
    # 创建一个虚拟图像文件用于测试
    try:
        dummy_image = Image.new('L', (256, 256), color=128)
        # 添加一些变化
        pix = dummy_image.load()
        for i in range(100, 150):
            for j in range(100, 150):
                pix[i, j] = 200
        dummy_image_path = "dummy_test_image.png"
        dummy_image.save(dummy_image_path)

        # 1. 提取原始尺寸的频谱
        spectrum_original = extract_spectral_features(dummy_image_path)
        if spectrum_original is not None:
            print(f"Original spectrum shape: {spectrum_original.shape}")
            print(f"Original spectrum min/max: {np.min(spectrum_original):.4f} / {np.max(spectrum_original):.4f}")
            # 可选：显示图像 (需要 matplotlib)
            # import matplotlib.pyplot as plt
            # plt.imshow(spectrum_original, cmap='gray')
            # plt.title("Original Log Magnitude Spectrum (Normalized)")
            # plt.show()

        # 2. 提取并调整到指定尺寸 (例如 128x128)
        target_dim = (128, 128)
        spectrum_resized = extract_spectral_features(dummy_image_path, output_dim=target_dim)
        if spectrum_resized is not None:
            print(f"\nResized spectrum shape: {spectrum_resized.shape}")
            print(f"Resized spectrum min/max: {np.min(spectrum_resized):.4f} / {np.max(spectrum_resized):.4f}")
             # 可选：显示图像 (需要 matplotlib)
            # import matplotlib.pyplot as plt
            # plt.imshow(spectrum_resized, cmap='gray')
            # plt.title(f"Resized Log Magnitude Spectrum {target_dim} (Normalized)")
            # plt.show()

        # 3. 测试无效路径
        print("\nTesting invalid path:")
        spectrum_invalid = extract_spectral_features("non_existent_image.png")
        if spectrum_invalid is None:
            print("Correctly handled invalid path.")

        # 清理虚拟文件
        os.remove(dummy_image_path)

    except ImportError as e:
        print(f"Import error for testing (PIL or Scipy): {e}")
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        if os.path.exists("dummy_test_image.png"):
            os.remove("dummy_test_image.png") # Ensure cleanup even if error occurs later