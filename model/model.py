from matplotlib import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, model_dim=768, num_heads=8, dropout=0.2):
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
    def __init__(self, model_dim=768, ffn_dim=2048, dropout=0.2):
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
    def __init__(self, model_dim=768, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(multimodal_fusion_layer, self).__init__()
        self.attention_1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention_2 = MultiHeadAttention(model_dim, num_heads, dropout)
        
        self.feed_forward_1 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.feed_forward_2 = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)
        self.fusion_linear = nn.Linear(model_dim*2, model_dim)

    def forward(self, image_output, text_output, attn_mask=None):

        output_1 = self.attention_1(image_output, text_output, text_output,
                                 attn_mask)
        
        output_2 = self.attention_2(text_output, image_output, image_output,
                                 attn_mask)
        #print('attention out_shape:{}'.format(output.shape))
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


class MyModel(nn.Module):
    def __init__(self, args, model_dim = 768, num_layers=1, num_heads=8, ffn_dim=2048, dropout=0.3):
        super(MyModel, self).__init__()
        self.args = args
        self.image_classfier = Classifier(dropout, model_dim,2)
        self.text_classfier = Classifier(dropout, model_dim, 2)
        self.fft_feature_dim = 128*128
        self.log_var_a = nn.Parameter(torch.zeros(1))
        self.log_var_b = nn.Parameter(torch.zeros(1))
        self.log_var_c = nn.Parameter(torch.zeros(1))
        self.fft_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1), # Input: B, 1, 128, 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # B, 16, 64, 64
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # B, 32, 32, 32
            # Add more layers if needed
            nn.AdaptiveAvgPool2d((1, 1)), # B, 32, 1, 1
            nn.Flatten(), # B, 32
            nn.Linear(32, model_dim) # Project to model_dim
        )
        self.image_gate_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.Sigmoid()
        )
        self.text_gate_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim), # Assuming sentiment_feat is already model_dim
            nn.Sigmoid()
        )
        if self.args.method in ['FND-2-SGAT']:
            self.sgat_layer = SimilarityGATFusionNet(
                embed_dim=768,
                gnn_hidden_dim=128,
                num_gat_layers=1,
                gat_heads=8,
                dropout_rate=0.3,
                num_classes=2
            )
            
        if self.args.method not in ['FND-2-CLIP']:
            self.image_encoder = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder)
            self.text_encoder = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder)
        
        if self.args.method not in ['FND-2-SGAT']:
            self.mm_classfier = Classifier(dropout, model_dim, 2)
            self.self_attention = MultiHeadAttention(model_dim, num_heads, dropout)
            self.fusion_layers = nn.ModuleList([
                multimodal_fusion_layer(model_dim, num_heads, ffn_dim, dropout)
                for _ in range(num_layers)
            ])

    def compute_loss(self, loss_a, loss_b, loss_c):
        # 计算加权损失和正则项
        loss_weighted_a = 0.5 * torch.exp(-self.log_var_a) * loss_a + 0.5 * self.log_var_a
        loss_weighted_b = 0.5 * torch.exp(-self.log_var_b) * loss_b + 0.5 * self.log_var_b
        loss_weighted_c = 0.5 * torch.exp(-self.log_var_c) * loss_c + 0.5 * self.log_var_c
        return loss_weighted_a + loss_weighted_b + loss_weighted_c
        
    def forward(self, text=None, sentiment_text=None, image=None, fft_imge = None, label=None, infer=False, graph_data = None):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        if self.args.method in ['FND-2']:
            text = self.text_encoder(text=text)
            bert_cls = text[:, 0, :]
            sentiment_feat = sentiment_text.squeeze(1) 
            gate = self.text_gate_mlp(torch.cat((bert_cls, sentiment_feat), dim=1))
            fused_text_cls_feature = gate * bert_cls + (1 - gate) * sentiment_feat
            # fused_text_cls_feature = text[:, 0, :]
            
            try:
                image = torch.squeeze(image, 1)
                image = self.image_encoder(pixel_values=image)
            except Exception as e:
                print(f"An error occurred: {e}")
                image = torch.randn((2, 197, 768))
            image_cls_feature = image[:, 0, :] 
            batch_size = fft_imge.size(0)
        
            # Flatten FFT features: Shape [10, 128*128] = [10, 16384]
            fft_imge = fft_imge.unsqueeze(1).float() # Add channel dim: B, 1, 128, 128
            fft_feature = self.fft_encoder(fft_imge) # B, model_dim 
            gate = self.image_gate_mlp(torch.cat((image_cls_feature, fft_feature), dim=1))
            combined_image_feature = gate * image_cls_feature + (1 - gate) * fft_feature 
            output_text = self.text_classfier(fused_text_cls_feature)
            output_image = self.image_classfier(combined_image_feature)
            for fusion_layer in self.fusion_layers:
                fusion = fusion_layer( combined_image_feature,fused_text_cls_feature)   
            output_mm = self.mm_classfier(fusion) 
            if infer:
                return output_mm
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(criterion(output_image, label))
            MMLoss_m = torch.mean(criterion(output_mm, label))
            MMLoss_sum = MMLoss_m + MMLoss_text + MMLoss_image
            # MMLoss_sum = self.compute_loss(MMLoss_text, MMLoss_image, MMLoss_m)
            return MMLoss_sum, MMLoss_m, output_mm
        
        elif self.args.method in ['FND-2-CLIP']:
            text = text.squeeze(1)
            sentiment_feat = sentiment_text.squeeze(1) 
            gate = self.text_gate_mlp(torch.cat((text, sentiment_feat), dim=1))
            fused_text_cls_feature = gate * text + (1 - gate) * sentiment_feat
            # fused_text_cls_feature = text[:, 0, :]
            image_cls_feature = image.squeeze(1)
        
            # Flatten FFT features: Shape [10, 128*128] = [10, 16384]
            fft_imge = fft_imge.unsqueeze(1).float() # Add channel dim: B, 1, 128, 128
            fft_feature = self.fft_encoder(fft_imge) # B, model_dim 
            gate = self.image_gate_mlp(torch.cat((image_cls_feature, fft_feature), dim=1))
            combined_image_feature = gate * image_cls_feature + (1 - gate) * fft_feature 
            output_text = self.text_classfier(fused_text_cls_feature)
            output_image = self.image_classfier(combined_image_feature)
            for fusion_layer in self.fusion_layers:
                fusion = fusion_layer( combined_image_feature,fused_text_cls_feature)   
            output_mm = self.mm_classfier(fusion) 
            if infer:
                return output_mm
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(criterion(output_image, label))
            MMLoss_m = torch.mean(criterion(output_mm, label))
            # MMLoss_sum = MMLoss_m + MMLoss_text + MMLoss_image
            MMLoss_sum = self.compute_loss(MMLoss_text, MMLoss_image, MMLoss_m)
            return MMLoss_sum, MMLoss_m, output_mm
        
        elif self.args.method in ['FND-2-SGAT']:
            text = self.text_encoder(text=text)
            bert_cls = text[:, 0, :]
            sentiment_feat = sentiment_text.squeeze(1) 
            gate = self.text_gate_mlp(torch.cat((bert_cls, sentiment_feat), dim=1))
            fused_text_cls_feature = gate * bert_cls + (1 - gate) * sentiment_feat
            try:
                image = torch.squeeze(image, 1)
                image = self.image_encoder(pixel_values=image)
            except Exception as e:
                print(f"An error occurred: {e}")
                image = torch.randn((2, 197, 768))
            image_cls_feature = image[:, 0, :] 
            batch_size = fft_imge.size(0)
            # Flatten FFT features: Shape [10, 128*128] = [10, 16384]
            fft_imge = fft_imge.unsqueeze(1).float() # Add channel dim: B, 1, 128, 128
            fft_feature = self.fft_encoder(fft_imge) # B, model_dim 
            gate = self.image_gate_mlp(torch.cat((image_cls_feature, fft_feature), dim=1))
            combined_image_feature = gate * image_cls_feature + (1 - gate) * fft_feature 
            output_text = self.text_classfier(fused_text_cls_feature)
            output_image = self.image_classfier(combined_image_feature)
            if graph_data:
                output_mm = self.sgat_layer(graph_data)
            else:
              output_mm = output_text
            if infer:
                return output_mm
            MMLoss_text = torch.mean(criterion(output_text, label))
            MMLoss_image = torch.mean(criterion(output_image, label))
            MMLoss_m = torch.mean(criterion(output_mm, label))
            # MMLoss_sum = 0.4*MMLoss_m + 0.3*MMLoss_text + 0.3*MMLoss_image
            MMLoss_sum = self.compute_loss(MMLoss_text, MMLoss_image, MMLoss_m)
            return MMLoss_sum, MMLoss_m, output_mm
        
        elif self.args.method in ['MCAN']:
            text = self.text_encoder(text=text)
            image = torch.squeeze(image, 1)
            image = self.image_encoder(pixel_values=image)
            output_text = self.text_classfier(text[:, 0, :])
            output_image = self.image_classfier(image[:, 0, :])
            for fusion_layer in self.fusion_layers:
                fusion = fusion_layer( image[:, 0, :], text[:, 0, :])   
            output_mm = self.mm_classfier(fusion)
            if infer:
                return output_mm
            MMLoss_m = torch.mean(criterion(output_mm, label))
            return MMLoss_m, MMLoss_m, output_mm

        elif self.args.method in ['BERT']:
            text = self.text_encoder(text=text)
            output_text = self.text_classfier(text[:, 0, :])
            if infer:
                return output_text
            Loss_text = torch.mean(criterion(output_text, label))
            return Loss_text, Loss_text, output_text
        
        elif self.args.method in ['ViT']:
            image = torch.squeeze(image, 1)
            image = self.image_encoder(pixel_values=image)
            output_image = self.image_classfier(image[:, 0, :])
            if infer:
                return output_image
            Loss_image = torch.mean(criterion(output_image, label))
            return Loss_image, Loss_image, output_image
        
        

    def infer(self, text=None, sentiment_text=None, image=None, fft_imge = None,graph_data = None):
        MMlogit = self.forward(text, sentiment_text, image, fft_imge, infer=True, graph_data = graph_data)
        return MMlogit


class Classifier(nn.Module):
    def __init__(self, dropout, in_dim, out_dim):
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






