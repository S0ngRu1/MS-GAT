import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import List, Tuple, Optional

# Helper function to calculate pairwise cosine similarity efficiently
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
                 similarity_threshold_text: float = 0.5,
                 similarity_threshold_image: float = 0.5,
                 similarity_threshold_cross: float = 0.5,
                 modality_edge_weight: float = 1.0):
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
        text_modality_embed = text_entity_embeds.mean(dim=0, keepdim=True) if num_text > 0 else torch.zeros(1, embed_dim, device=device)
        image_modality_embed = image_entity_embeds.mean(dim=0, keepdim=True) if num_image > 0 else torch.zeros(1, embed_dim, device=device)

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
        # Decide on the weight for this hub-to-hub edge.
        # Could be a fixed weight, or based on overall modality similarity if you define one.
        hub_to_hub_weight = 1.0 # Example weight
        add_edge(text_modality_idx, image_modality_idx, hub_to_hub_weight)
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
                 num_classes: int = 2): # Example: 2 for fake/real classification
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
            text_modality_final_embed,
            image_modality_final_embed
        ], dim=1) # Shape: (batch_size, classifier_input_dim)

        fused_features = self.dropout(fused_features)
        logits = self.classifier(fused_features) # Shape: (batch_size, num_classes)

        return logits

# --- Example Usage ---
if __name__ == '__main__':
    # --- Hyperparameters ---
    EMBED_DIM = 768  # Example dimension from Sentence-BERT
    GNN_HIDDEN_DIM = 128
    NUM_GAT_LAYERS = 2
    GAT_HEADS = 4
    DROPOUT = 0.3
    NUM_CLASSES = 2 # Fake/Real
    SIM_THRESHOLD_T = 0.8
    SIM_THRESHOLD_I = 0.7
    SIM_THRESHOLD_TI = 0.6

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Dummy Data ---
    # Simulate having embeddings for entities from one news item
    num_text_entities = 5
    num_image_entities = 8
    text_embeds = torch.randn(num_text_entities, EMBED_DIM, device=DEVICE)
    image_embeds = torch.randn(num_image_entities, EMBED_DIM, device=DEVICE)

    # --- Build Graph ---
    graph_builder = GraphBuilder(
        similarity_threshold_text=SIM_THRESHOLD_T,
        similarity_threshold_image=SIM_THRESHOLD_I,
        similarity_threshold_cross=SIM_THRESHOLD_TI
    )
    graph_data = graph_builder.build_graph(text_embeds, image_embeds)
    print("Graph Data Example:")
    print(graph_data)
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.num_edges}")
    print(f"Edge attributes (weights) sample: {graph_data.edge_attr[:10] if graph_data.num_edges > 0 else 'No edges'}")
    print(f"Text Modality Node Index: {graph_data.text_modality_idx}")
    print(f"Image Modality Node Index: {graph_data.image_modality_idx}")


    # --- Instantiate Model ---
    model = SimilarityGATFusionNet(
        embed_dim=EMBED_DIM,
        gnn_hidden_dim=GNN_HIDDEN_DIM,
        num_gat_layers=NUM_GAT_LAYERS,
        gat_heads=GAT_HEADS,
        dropout_rate=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print("\nModel Architecture:")
    print(model)

    # --- Forward Pass (Single Graph Example) ---
    model.eval() # Set to evaluation mode
    with torch.no_grad():
        # For batching, you would use PyG's DataLoader and Batch object
        output_logits = model(graph_data.to(DEVICE))

    print(f"\nOutput Logits Shape: {output_logits.shape}") # Should be [1, NUM_CLASSES] for single graph
    print(f"Output Logits: {output_logits}")