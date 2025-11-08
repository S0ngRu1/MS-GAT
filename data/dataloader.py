import re
import os
from loguru import logger
from PIL import Image
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Tuple
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model.text_encoder import TextEncoder
from model.image_encoder import ImageEncoder
from model.clip_encoder import ClipEncoder
from model.fft import extract_spectral_features
from Sentiment_demo import predict_one as sentiment_predict, Config as sentiment_config, Model as sentiment_model
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import pandas as pd
from PIL import Image
import os
import numpy as np

        
def custom_collate_fn(batch):
    # 过滤掉无效的 Data 对象及其对应的其他数据
    valid_batch = []
    for item in batch:
        if isinstance(item[6], Data):  # 检查图数据是否有效
            valid_batch.append(item)
    
    if not valid_batch:
        logger.warning("No valid data found in the batch. Skipping this batch.")
        # 可以选择返回 None 或者一个默认值
        return None
    
    # 提取图数据
    graph_data = [item[6] for item in valid_batch]
    graph_batch = Batch.from_data_list(graph_data)

    # 提取其他数据
    batch_image = torch.stack([item[0] for item in valid_batch])
    fft_image = torch.stack([item[1] for item in valid_batch])
    sentiment_output = torch.stack([item[2] for item in valid_batch])
    text_input_ids = torch.stack([item[3] for item in valid_batch])
    text_token_type_ids = torch.stack([item[4] for item in valid_batch])
    text_attention_mask = torch.stack([item[5] for item in valid_batch])
    batch_label = torch.tensor([item[7] for item in valid_batch])

    return (batch_image, fft_image, sentiment_output, text_input_ids, text_token_type_ids, text_attention_mask, graph_batch, batch_label)

def MMDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train_set = MMDataset(args, mode='train')
        valid_set = MMDataset(args, mode='val')
        test_set = MMDataset(args, mode='test')
        logger.info(f'Train Dataset: {len(train_set)}')
        logger.info(f'Valid Dataset: {len(valid_set)}')
        logger.info(f'Test Dataset: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        return train_loader, valid_loader, test_loader
    else:
        logger.info('数据集无效')
        return None, None, None



def GraphDataLoader(args):
    if args.dataset in ['Weibo17','Weibo21','CFND_dataset']:
        train_set = GraphDataset(args, mode='train')
        valid_set = GraphDataset(args, mode='val')
        test_set = GraphDataset(args, mode='test')
        logger.info(f'Train Dataset: {len(train_set)}')
        logger.info(f'Valid Dataset: {len(valid_set)}')
        logger.info(f'Test Dataset: {len(test_set)}')
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=False, drop_last=True,collate_fn=custom_collate_fn)
        return train_loader, valid_loader, test_loader
    else:
        logger.info('数据集无效')
        return None, None, None


class GraphDataset(Dataset):
    def __init__(self, args, mode):
        self.args = args
        self.save = []
        self.max_length = 256
        if args.dataset in ['Weibo17','Weibo21']:
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '.csv', encoding='utf-8').dropna()
            self.entity_df = pd.read_json('datasets/'+args.dataset+'/'+'processed_data_'+mode + '.json', encoding='utf-8').dropna()
        elif args.dataset in ['CFND_dataset']:
            self.entity_df = pd.read_json('datasets/'+args.dataset+'/'+'processed_data_'+mode + '.json', encoding='utf-8').dropna()          
            self.df = pd.read_csv('datasets/'+args.dataset+'/'+mode + '_data.csv', encoding='utf-8').dropna()
            

        else:
            logger.info('数据集无效')
            return
        if self.args.method in ['FND-2-CLIP']:
            self.clip_encoder = ClipEncoder(self.args, pretrained_dir=args.pretrained_dir)
        elif self.args.method in ['FND-2-SGAT']:
            self.clip_encoder = ClipEncoder(self.args, pretrained_dir=args.pretrained_dir)
            self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
            self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
            self.graph_builder = GraphBuilder(
                similarity_threshold_text=0.5,
                similarity_threshold_image=0.5,
                similarity_threshold_cross=0.5
            )
        else:
            self.text_tokenizer = TextEncoder(pretrained_dir=args.pretrained_dir, text_encoder=args.text_encoder).get_tokenizer()
            self.image_tokenizer = ImageEncoder(pretrained_dir=args.pretrained_dir, image_encoder=args.image_encoder).get_tokenizer()
        self.img_width = 224
        self.img_height = 224
        self.depth = 3
        self.transforms = get_transforms()
        self.sentiment_config = sentiment_config()
        self.sentiment_config.device = args.device
        self.sentiment_model = sentiment_model(self.sentiment_config).to(self.sentiment_config.device)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = None
        label = None
        img_path = ''
        fft_imge = None
        text_ner = []
        image_ner = []
        if self.args.dataset in ['Weibo17','Weibo21']:
            _, image_name, text, label = self.df.iloc[index].values
            img_path = self.args.data_dir +'/'+ self.args.dataset +'/new_images/' + image_name
            text = preprocess_text(text)
            entity_data = self.entity_df.iloc[index]
            text_ner = entity_data['title_ner']
            plain_text_ner = entity_data.get("plain_text_ner", [])
            obj_list = entity_data.get("obj_list", [])
            if obj_list == '':
                obj_list = []
            image_ner = plain_text_ner + obj_list
        
        elif self.args.dataset in ['CFND_dataset']:
            _, title, image, label = self.df.iloc[index].values
            entity_data = self.entity_df.iloc[index]
            text_ner = entity_data['title_ner']
            plain_text_ner = entity_data.get("plain_text_ner", [])
            obj_list = entity_data.get("obj_list", [])
            if obj_list == '':
                obj_list = []
            image_ner = plain_text_ner + obj_list
            img_path = os.path.join(self.args.data_dir,self.args.dataset, image)
            text = preprocess_text(title)
            
        image_entity_embeds = []
        text_entity_embeds = []
        graph_data = None
        sentiment_output = sentiment_predict(text, self.sentiment_config, self.sentiment_model)
        if self.args.method in ['FND-2-SGAT','Weibo17','Weibo21']:
            text_tokens = self.text_tokenizer(text, max_length=self.max_length, add_special_tokens=True, truncation=True, 
                                        padding='max_length', return_tensors="pt")
            original_shape = (1, 3, 224, 224)
            zero_img_inputs = torch.zeros(original_shape)
            if text_ner:
                text_entity_embeds = self.clip_encoder.get_entity_features(text_ner)
            if image_ner:
                image_entity_embeds = self.clip_encoder.get_entity_features(image_ner)
                
            try:
                if os.path.exists(img_path) and os.path.isfile(img_path):
                    image = Image.open(os.path.join(img_path)).convert("RGB")
                    image = self.transforms(image)
                    fft_imge = extract_spectral_features(img_path,output_dim=(128, 128))
                    fft_imge = torch.tensor(fft_imge).to(self.args.device)
                    img_inputs = self.image_tokenizer(images=image, return_tensors="pt").pixel_values
                    text_feature,img_feature =  self.clip_encoder.get_features(text,img_path)
                    if text_entity_embeds  == []:
                        text_entity_embeds = text_feature
                        
                    else:
                        text_entity_embeds = torch.cat((text_feature, text_entity_embeds), dim=0)
                    if image_entity_embeds == []:
                        image_entity_embeds = img_feature
                    else:
                        image_entity_embeds = torch.cat((img_feature, image_entity_embeds), dim=0)
                    graph_data = self.graph_builder.build_graph(text_entity_embeds,image_entity_embeds)
                else:
                    img_inputs = zero_img_inputs
                    fft_imge = None  
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                img_inputs = zero_img_inputs
                fft_imge = None  
            return img_inputs, fft_imge, sentiment_output, text_tokens['input_ids'], text_tokens['token_type_ids'], text_tokens['attention_mask'], graph_data, label
        



def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculates pairwise cosine similarity between rows of x and rows of y."""
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)
    return torch.mm(x_norm, y_norm.transpose(0, 1))

class GraphBuilder:
    """
    Builds a heterogeneous graph with similarity and modality edges.
    Nodes: [Text Entities, Image Entities, Text Modality Node, Image Modality Node]
    Edges:
        - Intra-modal similarity edges (Text-Text, Image-Image)
        - Inter-modal similarity edges (Text-Image)
        - Modality connection edges (Text Entity - Text Modality, Image Entity - Image Modality)
    """
    def __init__(self,
                 similarity_threshold_text: float = 0.7,
                 similarity_threshold_image: float = 0.7,
                 similarity_threshold_cross: float = 0.7,
                 modality_edge_weight: float = 1):
        """
        Args:
            similarity_threshold_*: Threshold above which similarity edges are created.
            modality_edge_weight: Weight for edges connecting entities to modality nodes.
        """
        self.theta_t = similarity_threshold_text
        self.theta_i = similarity_threshold_image
        self.theta_ti = similarity_threshold_cross
        self.modality_w = modality_edge_weight

    def build_graph(self,
                    text_entity_embeds: torch.Tensor, # Shape: (num_text_entities, embed_dim)
                    image_entity_embeds: torch.Tensor # Shape: (num_image_entities, embed_dim)
                   ) -> Data:
        """
        Constructs the PyG Data object representing the graph.

        Args:
            text_entity_embeds: Embeddings for text entities.
            image_entity_embeds: Embeddings for image entities.

        Returns:
            A PyG Data object containing node features (x), edge indices (edge_index),
            and edge attributes/weights (edge_attr).
        """
        device = text_entity_embeds.device
        num_text = text_entity_embeds.size(0)
        num_image = image_entity_embeds.size(0)
        embed_dim = text_entity_embeds.size(1)

        # --- 1. Node Features (x) ---
        # Initialize modality node embeddings (e.g., average of entity embeddings or learnable)
        # Here, we use the average for simplicity. Consider nn.Parameter for learnable ones.
        text_modality_embed = text_entity_embeds[0].unsqueeze(0)
        image_modality_embed = image_entity_embeds[0].unsqueeze(0)

        x = torch.cat([
            text_entity_embeds,       # Nodes 0 to num_text - 1
            image_entity_embeds,      # Nodes num_text to num_text + num_image - 1
            text_modality_embed,      # Node num_text + num_image
            image_modality_embed      # Node num_text + num_image + 1
        ], dim=0)

        num_total_nodes = x.size(0)
        edge_index_list: List[Tuple[int, int]] = []
        edge_attr_list: List[float] = []

        # --- 2. Edges (edge_index, edge_attr) ---

        # Helper to add edges (ensures undirected graph)
        def add_edge(u, v, weight):
            edge_index_list.append((u, v))
            edge_attr_list.append(weight)
            edge_index_list.append((v, u)) # Add reverse edge for undirected graph
            edge_attr_list.append(weight)

        # a) Intra-modal Similarity Edges (Text-Text)
        if num_text > 1:
            sim_tt = pairwise_cosine_similarity(text_entity_embeds, text_entity_embeds)
            adj_tt = sim_tt > self.theta_t
            # Prevent self-loops from similarity
            adj_tt.fill_diagonal_(False)
            edges_tt = adj_tt.nonzero(as_tuple=False) # Get indices where similarity > threshold
            for i in range(edges_tt.size(0)):
                u, v = edges_tt[i, 0].item(), edges_tt[i, 1].item()
                # Add edge only once for pairs (u, v) where u < v to avoid duplicates with reverse edges
                if u < v:
                    weight = sim_tt[u, v].item()
                    add_edge(u, v, weight)

        # b) Intra-modal Similarity Edges (Image-Image)
        if num_image > 1:
            sim_ii = pairwise_cosine_similarity(image_entity_embeds, image_entity_embeds)
            adj_ii = sim_ii > self.theta_i
            adj_ii.fill_diagonal_(False)
            edges_ii = adj_ii.nonzero(as_tuple=False)
            # Offset indices by num_text
            base_idx_i = num_text
            for i in range(edges_ii.size(0)):
                u, v = edges_ii[i, 0].item(), edges_ii[i, 1].item()
                if u < v:
                    weight = sim_ii[u, v].item()
                    add_edge(base_idx_i + u, base_idx_i + v, weight)


        # c) Inter-modal Similarity Edges (Text-Image)
        if num_text > 0 and num_image > 0:
            sim_ti = pairwise_cosine_similarity(text_entity_embeds, image_entity_embeds)
            adj_ti = sim_ti > self.theta_ti
            edges_ti = adj_ti.nonzero(as_tuple=False)
            # Offset image indices by num_text
            base_idx_i = num_text
            for i in range(edges_ti.size(0)):
                u, v = edges_ti[i, 0].item(), edges_ti[i, 1].item() # u is text idx, v is image idx
                weight = sim_ti[u, v].item()
                add_edge(u, base_idx_i + v, weight) # Connect text node u to image node (base_idx_i + v)


        # d) Modality Connection Edges
        text_modality_idx = num_text + num_image
        image_modality_idx = num_text + num_image + 1

        # Connect text entities to text modality node
        for i in range(num_text):
            add_edge(i, text_modality_idx, self.modality_w)

        # Connect image entities to image modality node
        base_idx_i = num_text
        for i in range(num_image):
             add_edge(base_idx_i + i, image_modality_idx, self.modality_w)


        if not edge_index_list: # Handle cases with no edges
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_attr = torch.empty((0,), dtype=torch.float, device=device)
        else:
             edge_index = torch.tensor(edge_index_list, dtype=torch.long, device=device).t().contiguous()
             edge_attr = torch.tensor(edge_attr_list, dtype=torch.float, device=device)


        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        graph_data.num_text_entities = num_text
        graph_data.num_image_entities = num_image
        graph_data.text_modality_idx = text_modality_idx
        graph_data.image_modality_idx = image_modality_idx

        return graph_data