"""
Enhanced chunk aggregator training:
- Embedding caching
- Hybrid Triplet + BCE loss
- Checkpoint saving
- Configurable n_chunks, batch size, triplets per epoch
"""

import os
import sys
import json
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Transformer embeddings
# ---------------------------
from transformers import pipeline
# Global UnixCoder pipeline
pipe = pipeline("feature-extraction", model="microsoft/unixcoder-base", device=-1)

# ---------------------------
# Fix for project root imports
# ---------------------------
sys.path.append(os.path.abspath(".."))

# ---------------------------
# Dataset and chunk splitter
# ---------------------------
from data.dataset_factory import get_dataset_generator                  # pylint: disable=wrong-import-position
from preprocessing.embedding_chunks import get_ready_to_embed_chunks    # pylint: disable=wrong-import-position


# ---------------------------
# Format code + AST
# ---------------------------
def format_chunks_for_encoder(java_code: str):
    """
    Format parsable chunks for the encoder
    """
    formatted_chunks = []
    chunks = get_ready_to_embed_chunks(java_code)

    for _, code_chunk, ast_chunk in chunks:
        if not code_chunk.strip() and not ast_chunk:
            continue

        # AST dict -> string
        ast_str = json.dumps(ast_chunk) if isinstance(ast_chunk, dict) else str(ast_chunk)
        chunk_text = f"<s> <encoder-only> {code_chunk.strip()} <AST> {ast_str.strip()} </AST> </s>"
        formatted_chunks.append(chunk_text)
    return formatted_chunks


# ---------------------------
# Embed chunks with caching
# ---------------------------
embedding_cache = {}

def embed_java_chunks(java_code: str, n_chunks: int = 5):
    """
    Embed java chunks
    """
    if java_code in embedding_cache:
        return embedding_cache[java_code]

    formatted_chunks = format_chunks_for_encoder(java_code)
    embeddings = []

    for chunk in formatted_chunks:
        emb = pipe(chunk)  # [[hidden_dim]]
        emb_tensor = torch.tensor(emb[0])
        if emb_tensor.ndim == 2:
            emb_tensor = emb_tensor.mean(dim=0)
        embeddings.append(emb_tensor)

    if not embeddings:
        embeddings = [torch.zeros(pipe.model.config.hidden_size)]

    # Pad or truncate
    if len(embeddings) < n_chunks:
        pad = [torch.zeros_like(embeddings[0])] * (n_chunks - len(embeddings))
        embeddings.extend(pad)
    elif len(embeddings) > n_chunks:
        embeddings = embeddings[:n_chunks]

    result = torch.stack(embeddings)
    embedding_cache[java_code] = result
    return result


# ---------------------------
# Chunk aggregator model
# ---------------------------
class ChunkAggregator(nn.Module):
    """
    Chunk aggregator class
    """
    def __init__(self, emb_dims, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dims, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Optional BCE head
        self.bce_head = nn.Linear(emb_dims, 1)

    def forward(self, chunk_embeddings):
        """
        Forward feed for neural network
        """
        # Weighted aggregation
        logits = self.mlp(chunk_embeddings)
        weights = torch.softmax(logits, dim=1)
        agg_emb = (weights * chunk_embeddings).sum(dim=1)
        # BCE prediction
        bce_logits = self.bce_head(agg_emb)
        return agg_emb, bce_logits


# ---------------------------
# Losses
# ---------------------------
def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Define triplet log loss
    """
    d_pos = F.pairwise_distance(anchor, positive) # pylint: disable=not-callable
    d_neg = F.pairwise_distance(anchor, negative) # pylint: disable=not-callable
    loss = F.relu(d_pos - d_neg + margin)
    return loss.mean()


# ---------------------------
# Dataset
# ---------------------------
class FunctionTripletDataset(Dataset):
    """
    Dataset loader
    """
    def __init__(self, n_chunks=5, triplets_per_epoch=100):
        self.n_chunks = n_chunks
        self.triplets_per_epoch = triplets_per_epoch

        self.plag_gen = get_dataset_generator(
            dataset_name='sourcecodeplag', mode='plagiarized'
            )
        self.non_plag_gen = get_dataset_generator(
            dataset_name='sourcecodeplag', mode='non_plagiarized'
            )

    def __len__(self):
        return self.triplets_per_epoch

    def __getitem__(self, idx):
        sample = next(self.plag_gen)
        code_a = sample.code_a
        code_b = sample.code_b
        emb_a = embed_java_chunks(code_a, self.n_chunks)
        emb_b = embed_java_chunks(code_b, self.n_chunks)

        neg_sample = next(self.non_plag_gen)
        code_neg = neg_sample.code_b
        emb_neg = embed_java_chunks(code_neg, self.n_chunks)

        return emb_a, emb_b, emb_neg


# ---------------------------
# Training
# ---------------------------
# pylint: disable=too-many-locals
def train(
    model,      # pylint: disable=redefined-outer-name
    dataloader, # pylint: disable=redefined-outer-name
    optimizer,  # pylint: disable=redefined-outer-name
    device,     # pylint: disable=redefined-outer-name
    epochs=10):
    """
    Entrypoint for training
    """
    margin=1.0
    hybrid=True
    checkpoint_dir="models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            emb_anchor, bce_anchor = model(anchor)
            emb_positive, _ = model(positive)
            emb_negative, _ = model(negative)

            loss = triplet_loss(emb_anchor, emb_positive, emb_negative, margin)

            # Hybrid BCE: anchor vs positive = 1, anchor vs negative = 0
            if hybrid:
                bce_labels = torch.cat(
                    [torch.ones_like(bce_anchor),
                    torch.zeros_like(bce_anchor)],
                    dim=0
                    )
                bce_preds = torch.cat([bce_anchor, bce_anchor], dim=0)  # reuse anchor logits
                bce_loss = F.binary_cross_entropy_with_logits(bce_preds, bce_labels)
                loss = loss + bce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"chunk_aggregator_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


# ---------------------------
# Load model
# ---------------------------
def load_model(model_class, path, *args, **kwargs):
    """
    Load model
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    N_CHUNKS = 150
    HIDDEN_DIM = 64
    BATCH_SIZE = 2
    EPOCHS = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FunctionTripletDataset(n_chunks=N_CHUNKS, triplets_per_epoch=50)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    emb_dim = pipe.model.config.hidden_size
    models = ChunkAggregator(emb_dim, HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(models.parameters(), lr=1e-3)

    train(models, dataloader, optimizer, device, epochs=EPOCHS)
