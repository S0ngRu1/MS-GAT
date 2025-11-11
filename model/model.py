import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from model.text_encoder import TextEncoder
from model.image_encoder import ImageEncoder

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x
    
class multimodal_attention(nn.Module):
    """
    dot-product attention mechanism
    """
    def __init__(self, attention_dropout=0.2):
        super(multimodal_attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
       
        attention = torch.matmul(q, k.transpose(-2, -1))
        #print('attention.shape:{}'.format(attention.shape))
        if scale:
            attention = attention * scale

        if attn_mask:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        #print('attention.shftmax:{}'.format(attention))
        attention = self.dropout(attention)
        attention = torch.matmul(attention, v)
        #print('attn_final.shape:{}'.format(attention.shape))

        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=768, num_heads=8, dropout=0.3):
        super(MultiHeadAttention, self).__init__()
        
        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(1, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(1, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = multimodal_attention(dropout)
        self.linear_final = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        residual = query
        query = query.unsqueeze(-1)
        key = key.unsqueeze(-1)
        value = value.unsqueeze(-1)
        #print("query.shape:{}".format(query.shape))
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        #batch_size = key.size(0)
        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)
        #print('key.shape:{}'.format(key.shape))
        # split by heads
        key = key.view(-1, num_heads, self.model_dim, dim_per_head)
        value = value.view(-1, num_heads, self.model_dim, dim_per_head)
        query = query.view(-1, num_heads, self.model_dim, dim_per_head)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads)**-0.5
        attention = self.dot_product_attention(query, key, value, 
                                               scale, attn_mask)
        attention = attention.view(-1, self.model_dim, dim_per_head * num_heads)
        #print('attention_con_shape:{}'.format(attention.shape))
        # final linear projection
        output = self.linear_final(attention).squeeze(-1)
        #print('output.shape:{}'.format(output.shape))
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)
        return output


class PositionalWiseFeedForward(nn.Module):
    """
    Fully-connected network 
    """
    def __init__(self, model_dim=768, ffn_dim=3072, dropout=0.3):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, model_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)
        output = x
        return output


class multimodal_fusion_layer(nn.Module):
    """
    A layer of fusing features 
    """
    def __init__(self, model_dim=768):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim)
        self.attention_2 = MultiHeadAttention(model_dim)
        
        self.feed_forward_1 = PositionalWiseFeedForward(model_dim)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim)
        self.fusion_linear = nn.Linear(model_dim*2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):

        output_1 = self.attention_1(image_output, text_output, text_output,
                                 attn_mask)
        
        output_2 = self.attention_2(text_output, image_output, image_output,
                                 attn_mask)
        output_1 = self.feed_forward_1(output_1)
        output_2 = self.feed_forward_2(output_2)
        
        output = torch.cat([output_1, output_2], dim=1)
        output = self.fusion_linear(output)

        return output



class SimilarityGATFusionNet(nn.Module):
    """
    A GNN model using GAT layers to fuse multimodal entity information
    from a graph with similarity-weighted edges.
    """
    def __init__(self,
                 embed_dim: int,
                 gnn_hidden_dim: int,
                 num_gat_layers: int = 2,
                 gat_heads: int = 4,
                 dropout_rate: float = 0.3,
                 num_classes: int = 2): 
        super().__init__()

        self.embed_dim = embed_dim
        self.gnn_hidden_dim = gnn_hidden_dim
        self.num_gat_layers = num_gat_layers
        self.gat_heads = gat_heads
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.gat_layers = nn.ModuleList()
        current_dim = embed_dim

        # GAT Layers
        for i in range(num_gat_layers):
            # GATConv takes edge_attr if edge_dim is specified.
            # The attention mechanism can implicitly use these edge weights.
            conv = GATConv(current_dim,
                           gnn_hidden_dim,
                           heads=gat_heads,
                           dropout=dropout_rate,
                           edge_dim=1, # Our edge_attr is 1-dimensional (the weight)
                           concat=True) # Concatenate heads' outputs
            self.gat_layers.append(conv)
            current_dim = gnn_hidden_dim * gat_heads # Output dim after concatenation

        # Readout and Classifier
        # Use both global pooling and modality node embeddings
        self.classifier_input_dim = current_dim + 2 * current_dim # GlobalPool + TextModality + ImageModality
        # Adjust classifier input dim if using average instead of concat for last GAT layer
        # For simplicity, assume last layer also concatenates heads
        # Or alternatively, make the last layer average:
        # last_conv = GATConv(..., concat=False) -> current_dim = gnn_hidden_dim
        # self.classifier_input_dim = gnn_hidden_dim + 2*gnn_hidden_dim

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.classifier_input_dim, num_classes)


    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the GAT network.

        Args:
            data: A PyG Data object created by GraphBuilder. Must contain:
                  x, edge_index, edge_attr, text_modality_idx, image_modality_idx

        Returns:
            Logits for classification (shape: [batch_size, num_classes]).
            Note: Assumes batching handled by PyG's Batch object if multiple graphs are processed.
                  This example shows processing a single graph.
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()
        edge_attr = edge_attr.float()
        text_modality_idx = data.text_modality_idx
        image_modality_idx = data.image_modality_idx
        batch = data.batch if hasattr(data, 'batch') else None # Handle batching

        # Pass through GAT layers
        for i, layer in enumerate(self.gat_layers):
            # GATConv expects edge_attr named 'edge_attr'
            x = layer(x, edge_index, edge_attr=edge_attr)
            if i < self.num_gat_layers - 1: # Apply activation/dropout except last layer's output maybe
                 x = F.elu(x) # ELU is common with GAT
                 x = self.dropout(x)


        # --- Readout ---
        # 1. Global mean pooling over all nodes in the graph (per graph in batch)
        global_pool_features = global_mean_pool(x, batch=batch) # Shape: (batch_size, gnn_hidden_dim * heads)

        # 2. Get the final embeddings of the modality nodes
        # Need to handle batching correctly if applicable
        if batch is None: # Single graph case
             text_modality_final_embed = x[text_modality_idx].unsqueeze(0) # Shape: (1, gnn_hidden_dim * heads)
             image_modality_final_embed = x[image_modality_idx].unsqueeze(0) # Shape: (1, gnn_hidden_dim * heads)
        else: # Batched graph case
            # Find the indices of modality nodes within the flattened batch tensor 'x'
            text_mod_indices = [data.num_nodes * i + data.text_modality_idx for i in range(batch.max().item() + 1)] # Approximate; requires careful index calculation based on batch structure
            image_mod_indices = [data.num_nodes * i + data.image_modality_idx for i in range(batch.max().item() + 1)] # This part might need adjustment depending on how PyG batches your specific graph structure.
            # It's often simpler to extract *before* batching or use dedicated batch handling.
            # For simplicity, let's assume single graph for now.
            # TODO: Implement robust batch handling for modality node extraction if needed.
            text_modality_final_embed = x[text_modality_idx].unsqueeze(0) # Placeholder
            image_modality_final_embed = x[image_modality_idx].unsqueeze(0) # Placeholder


        # Concatenate features for classifier
        fused_features = torch.cat([
            global_pool_features,
            text_modality_final_embed.squeeze(0),
            image_modality_final_embed.squeeze(0)
        ], dim=1) # Shape: (batch_size, classifier_input_dim)

        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features) # Shape: (batch_size, num_classes)

        return logits


class BaseModel(nn.Module):
    """基础模型类"""
    def __init__(self, args, model_dim=768, dropout=0.3):
        super().__init__()
        self.args = args
        self.model_dim = model_dim
        self.image_classifier = Classifier(in_dim=self.model_dim, out_dim=2,dropout=dropout)
        self.text_classifier = Classifier(in_dim=self.model_dim, out_dim=2,dropout=dropout)
        self.fft_feature_dim = 128 * 128
        
        # 公共的FFT编码器
        self.fft_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, model_dim)
        )
        
        # 公共的门控MLP
        self.image_gate_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )
        self.text_gate_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )
        
        # 公共损失计算
        self.log_var_a = nn.Parameter(torch.zeros(1))
        self.log_var_b = nn.Parameter(torch.zeros(1))
        self.log_var_c = nn.Parameter(torch.zeros(1))

    def compute_loss(self, loss_a, loss_b, loss_c):
        loss_weighted_a = 0.5 * torch.exp(-self.log_var_a) * loss_a + 0.5 * self.log_var_a
        loss_weighted_b = 0.5 * torch.exp(-self.log_var_b) * loss_b + 0.5 * self.log_var_b
        loss_weighted_c = 0.5 * torch.exp(-self.log_var_c) * loss_c + 0.5 * self.log_var_c
        return loss_weighted_a + loss_weighted_b + loss_weighted_c


class FND2Model(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args,** kwargs)
        self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
        self.self_attention = MultiHeadAttention(self.model_dim, kwargs['num_heads'], kwargs['dropout'])
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(self.model_dim)
            for _ in range(kwargs['num_layers'])
        ])
        self.mm_classifier = Classifier(self.model_dim, 2, kwargs['dropout'])

    def forward(self, text=None, sentiment_text=None, image=None, fft_imge=None, label=None, infer=False, graph_data=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        if text is None:
            raise ValueError("text cannot be None for FND-2 model")
        text_feat = self.text_encoder(text=text)
        bert_cls = text_feat[:, 0, :]
        
        if sentiment_text is None:
            raise ValueError("sentiment_text cannot be None for FND-2 model")
        sentiment_feat = sentiment_text.squeeze(1)  
        text_gate = self.text_gate_mlp(torch.cat((bert_cls, sentiment_feat), dim=1))
        fused_text = text_gate * bert_cls + (1 - text_gate) * sentiment_feat
        
        if image is None:
            raise ValueError("image cannot be None for FND-2 model")
        try:
            image_squeezed = image.squeeze(1) 
            image_feat = self.image_encoder(pixel_values=image_squeezed)
        except Exception as e:
            logger.info(f"Image encoding error: {e}")
            image_feat = torch.randn((text.size(0), 197, self.model_dim), device=text.device)
        image_cls = image_feat[:, 0, :]
        
        if fft_imge is None:
            raise ValueError("fft_imge cannot be None for FND-2 model")
        fft_imge = fft_imge.unsqueeze(1).float()
        fft_feat = self.fft_encoder(fft_imge)
        image_gate = self.image_gate_mlp(torch.cat((image_cls, fft_feat), dim=1))
        combined_image = image_gate * image_cls + (1 - image_gate) * fft_feat
        
        output_text = self.text_classifier(fused_text)
        output_image = self.image_classifier(combined_image)
        
        fusion = None
        for layer in self.fusion_layers:
            fusion = layer(combined_image, fused_text)
        output_mm = self.mm_classifier(fusion)
        if infer:
            return output_mm
        loss_text = torch.mean(criterion(output_text, label))
        loss_image = torch.mean(criterion(output_image, label))
        loss_mm = torch.mean(criterion(output_mm, label))
        total_loss = loss_mm + loss_text + loss_image
        return total_loss, loss_mm, output_mm

class FND2CLIPModel(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args,** kwargs)
        # CLIP不需要独立编码器（直接使用预提取特征）
        self.self_attention = MultiHeadAttention(self.model_dim, kwargs['num_heads'], kwargs['dropout'])
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(self.model_dim)
            for _ in range(kwargs['num_layers'])
        ])
        self.mm_classifier = Classifier(self.model_dim, 2, kwargs['dropout'])

    def forward(self, text=None, sentiment_text=None, image=None, fft_imge=None, label=None, infer=False, graph_data=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # 文本处理（CLIP文本特征直接输入）
        if text is None or sentiment_text is None:
            raise ValueError("text and sentiment_text cannot be None for CLIP model")
        text_squeezed = text.squeeze(1)  # 确保输入非空
        sentiment_feat = sentiment_text.squeeze(1)  # 确保输入非空
        text_gate = self.text_gate_mlp(torch.cat((text_squeezed, sentiment_feat), dim=1))
        fused_text = text_gate * text_squeezed + (1 - text_gate) * sentiment_feat
        
        # 图像处理（CLIP图像特征直接输入）
        if image is None:
            raise ValueError("image cannot be None for CLIP model")
        image_cls = image.squeeze(1)  # 确保输入非空
        
        # FFT特征处理
        if fft_imge is None:
            raise ValueError("fft_imge cannot be None for CLIP model")
        fft_imge = fft_imge.unsqueeze(1).float()
        fft_feat = self.fft_encoder(fft_imge)
        image_gate = self.image_gate_mlp(torch.cat((image_cls, fft_feat), dim=1))
        combined_image = image_gate * image_cls + (1 - image_gate) * fft_feat
        
        # 分类与融合
        output_text = self.text_classifier(fused_text)
        output_image = self.image_classifier(combined_image)
        
        fusion = None
        for layer in self.fusion_layers:
            fusion = layer(combined_image, fused_text)
        output_mm = self.mm_classifier(fusion)
        
        if infer:
            return output_mm
        
        # 损失计算（使用加权损失）
        loss_text = torch.mean(criterion(output_text, label))
        loss_image = torch.mean(criterion(output_image, label))
        loss_mm = torch.mean(criterion(output_mm, label))
        total_loss = self.compute_loss(loss_text, loss_image, loss_mm)
        return total_loss, loss_mm, output_mm
    
    
class FND2SGATModel(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args,** kwargs)
        # SGAT特有的图融合层
        self.sgat_layer = SimilarityGATFusionNet(
            embed_dim=768,
            gnn_hidden_dim=128,
            num_gat_layers=1,
            gat_heads=8,
            dropout_rate=0.3,
            num_classes=2
        )
        # 基础编码器
        self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)

    def forward(self, text=None, sentiment_text=None, image=None, fft_imge=None, label=None, infer=False, graph_data=None):
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        # 文本处理
        if text is None or sentiment_text is None:
            raise ValueError("text and sentiment_text cannot be None for SGAT model")
        text_feat = self.text_encoder(text=text)
        bert_cls = text_feat[:, 0, :]
        sentiment_feat = sentiment_text.squeeze(1)  # 确保输入非空
        text_gate = self.text_gate_mlp(torch.cat((bert_cls, sentiment_feat), dim=1))
        fused_text = text_gate * bert_cls + (1 - text_gate) * sentiment_feat
        
        # 图像处理
        if image is None:
            raise ValueError("image cannot be None for SGAT model")
        try:
            image_squeezed = image.squeeze(1)  # 确保输入非空
            image_feat = self.image_encoder(pixel_values=image_squeezed)
        except Exception as e:
            print(f"Image encoding error: {e}")
            image_feat = torch.randn((text.size(0), 197, self.model_dim), device=text.device)
        image_cls = image_feat[:, 0, :]
        
        # FFT特征处理
        if fft_imge is None:
            raise ValueError("fft_imge cannot be None for SGAT model")
        fft_imge = fft_imge.unsqueeze(1).float()
        fft_feat = self.fft_encoder(fft_imge)
        image_gate = self.image_gate_mlp(torch.cat((image_cls, fft_feat), dim=1))
        combined_image = image_gate * image_cls + (1 - image_gate) * fft_feat
        
        # 分类与图融合
        output_text = self.text_classifier(fused_text)
        output_image = self.image_classifier(combined_image)
        
        # 图融合（必须提供graph_data）
        if graph_data is None:
            output_mm = output_text  # 降级处理
        else:
            output_mm = self.sgat_layer(graph_data)
        
        if infer:
            return output_mm
        
        # 损失计算
        loss_text = torch.mean(criterion(output_text, label))
        loss_image = torch.mean(criterion(output_image, label))
        loss_mm = torch.mean(criterion(output_mm, label))
        total_loss = self.compute_loss(loss_text, loss_image, loss_mm)
        return total_loss, loss_mm, output_mm
    

class MCANModel(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args,** kwargs)
        self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
        self.fusion_layers = nn.ModuleList([
            multimodal_fusion_layer(self.model_dim)
        ])
        self.mm_classifier = Classifier(self.model_dim, 2, kwargs['dropout'])

    def forward(self, text=None, image=None, label=None, infer=False, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if text is None or image is None:
            raise ValueError("text and image cannot be None for MCAN model")
        
        text_feat = self.text_encoder(text=text)
        image_squeezed = image.squeeze(1)  # 确保输入非空
        image_feat = self.image_encoder(pixel_values=image_squeezed)
        
        output_text = self.text_classifier(text_feat[:, 0, :])
        output_image = self.image_classifier(image_feat[:, 0, :])
        
        fusion = None
        for layer in self.fusion_layers:
            fusion = layer(image_feat[:, 0, :], text_feat[:, 0, :])
        output_mm = self.mm_classifier(fusion)
        
        if infer:
            return output_mm
        
        loss_mm = torch.mean(criterion(output_mm, label))
        return loss_mm, loss_mm, output_mm


class BERTModel(BaseModel):
    def __init__(self, args,** kwargs):
        super().__init__(args, **kwargs)
        self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)

    def forward(self, text=None, label=None, infer=False,** kwargs):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if text is None:
            raise ValueError("text cannot be None for BERT model")
        
        text_feat = self.text_encoder(text=text)
        output_text = self.text_classifier(text_feat[:, 0, :])
        
        if infer:
            return output_text
        
        loss_text = torch.mean(criterion(output_text, label))
        return loss_text, loss_text, output_text


class ViTModel(BaseModel):
    def __init__(self, args, **kwargs):
        super().__init__(args,** kwargs)
        self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)

    def forward(self, image=None, label=None, infer=False, **kwargs):
        criterion = nn.CrossEntropyLoss(reduction='none')
        if image is None:
            raise ValueError("image cannot be None for ViT model")
        
        image_squeezed = image.squeeze(1) 
        image_feat = self.image_encoder(pixel_values=image_squeezed)
        output_image = self.image_classifier(image_feat[:, 0, :])
        
        if infer:
            return output_image
        
        loss_image = torch.mean(criterion(output_image, label))
        return loss_image, loss_image, output_image


def create_model(args,** kwargs):
    """根据args.method创建对应的模型实例"""
    method = args.method
    if method == 'FND-2':
        return FND2Model(args, **kwargs)
    elif method == 'FND-2-CLIP':
        return FND2CLIPModel(args,** kwargs)
    elif method == 'FND-2-SGAT':
        return FND2SGATModel(args, **kwargs)
    elif method == 'MCAN':
        return MCANModel(args,** kwargs)
    elif method == 'BERT':
        return BERTModel(args, **kwargs)
    elif method == 'ViT':
        return ViTModel(args,** kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(Classifier, self).__init__()
        self.post_dropout = nn.Dropout(p=dropout)
        self.hidden_layer = LinearLayer(in_dim, 256)   
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)   
        self.classify = LinearLayer(256, out_dim)

    def forward(self, input):
        hidden_output = self.leaky_relu(self.hidden_layer(input))
        hidden_output_drop = self.post_dropout(hidden_output)
        output = self.classify(hidden_output_drop)
        return output






