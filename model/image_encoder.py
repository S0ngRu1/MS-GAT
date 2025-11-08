import torch.nn as nn
from transformers import ViTFeatureExtractor, ViTModel

class ImageEncoder(nn.Module):
    def __init__(self, pretrained_dir, image_encoder='base', feature_dim=768):
        """
        image_encoder: base / large
        """
        super(ImageEncoder, self).__init__()

        assert image_encoder in ['vit-base', 'vit-large']

        tokenizer = ViTFeatureExtractor
        model = ViTModel

        if image_encoder == 'vit-base':
            config = f'{pretrained_dir}/vit-base/config.json'
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit-base/')
            self.model = model.from_pretrained(pretrained_dir+'/vit-base/', config=config, add_pooling_layer=False)
            self.model.config.hidden_size = feature_dim  
        else:
            config = f'{pretrained_dir}/vit-large/config.json'
            self.tokenizer = tokenizer.from_pretrained(pretrained_dir+'/vit-large/')
            self.model = model.from_pretrained(pretrained_dir+'/vit-large/', config=config, add_pooling_layer=False)
            self.model.config.hidden_size = feature_dim  

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, pixel_values):
        last_hidden_states = self.model(pixel_values=pixel_values).last_hidden_state
        return last_hidden_states


# import torch.nn as nn
# # 从 transformers 库导入 Swin Transformer 相关组件
# from transformers import AutoFeatureExtractor, SwinModel 
# import logging # 推荐添加日志记录

# logger = logging.getLogger(__name__)

# class ImageEncoder(nn.Module):
#     def __init__(self, image_encoder_id='Pretrained/swin-base-patch4-window7-224/'):
#         """
#         使用 Swin Transformer 作为图像编码器。

#         Args:
#             image_encoder_id (str): Hugging Face Hub 上的 Swin Transformer 模型标识符
#                                      (例如 'microsoft/swin-tiny-patch4-window7-224', 
#                                      'microsoft/swin-base-patch4-window7-224', etc.) 
#                                      或者是一个包含模型文件的本地目录路径。
#                                      默认为 'microsoft/swin-tiny-patch4-window7-224'。
#         """
#         super(ImageEncoder, self).__init__()

#         self.image_encoder_id = image_encoder_id
        
#         try:
#             # 使用 AutoFeatureExtractor 自动加载与模型匹配的特征提取器（预处理器）
#             # 它通常能正确处理 Swin 需要的图像尺寸、归一化等
#             self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.image_encoder_id)
            
#             # 加载 SwinModel 模型
#             # add_pooling_layer=False: 使模型输出最后一层的特征序列 (B, num_patches, hidden_dim)，
#             #                          类似于 ViT 的 last_hidden_state (不含 CLS token)
#             #                          num_patches 是最后一个 stage 的 patch 数量 (e.g., 7x7=49 for 224 input)
#             # add_pooling_layer=True: 会在最后添加一个全局平均池化层，输出 (B, hidden_dim)
#             # 我们设为 False 以便和原 ViT 代码的输出结构更相似（都输出序列特征）
#             self.model = SwinModel.from_pretrained(self.image_encoder_id, add_pooling_layer=False)

#             # 获取模型最后一层的隐藏层维度，用于后续模块参考 (可选)
#             # Swin 的配置中通常用 hidden_sizes 列表存储每个 stage 的维度
#             self.output_feature_dim = self.model.config.embed_dim
#             logger.info(f"Loaded Swin Transformer: {self.image_encoder_id}")
#             logger.info(f"Output feature dimension (last stage): {self.output_feature_dim}")
            
#         except Exception as e:
#             logger.error(f"Error loading Swin Transformer model '{self.image_encoder_id}': {e}")
#             raise e # 重新抛出异常，以便调用者知道加载失败

#     def get_feature_extractor(self):
#         """
#         获取用于预处理图像的特征提取器。
#         (注意：在 Hugging Face 中，处理图像的通常叫 Feature Extractor 而不是 Tokenizer)
#         """
#         return self.feature_extractor

#     def forward(self, pixel_values):
#         """
#         对输入的像素值进行编码。

#         Args:
#             pixel_values (torch.Tensor): 由特征提取器处理后的图像张量，
#                                          通常形状为 (batch_size, num_channels, height, width)。

#         Returns:
#             torch.Tensor: Swin Transformer 最后一层输出的隐藏状态序列。
#                           形状: (batch_size, num_patches_in_final_stage, hidden_size_in_final_stage)。
#                           例如，对于输入 224x224 的 Swin-Tiny，形状为 (batch_size, 49, 768)。
#         """
#         try:
#             outputs = self.model(pixel_values=pixel_values)
#             # 获取最后一层的隐藏状态
#             last_hidden_states = outputs.last_hidden_state
#             return last_hidden_states
#         except Exception as e:
#             logger.error(f"Error during SwinModel forward pass: {e}")
#             # 可以考虑返回一个特定错误值或重新抛出
#             raise e


if __name__ == "__main__":
    vit_normal = ImageEncoder()
