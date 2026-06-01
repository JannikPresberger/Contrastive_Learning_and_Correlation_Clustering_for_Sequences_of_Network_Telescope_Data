import torch
import torch.nn as nn

import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ProjectionHead(nn.Module):
    def __init__(self, d_model, d_out):
        super(ProjectionHead,self).__init__()

        self.fc1 = nn.Linear(d_model,d_model)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_model,d_out)

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward,self).__init__()

        self.fc1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_seq_len):
        super(PositionalEncoding,self).__init__()

        pe = torch.zeros(max_seq_len,d_model)
        pos = torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2,dtype=float) * -(math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(pos * div_term)
        pe[:,1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):

        return x + self.pe[:,:x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model//num_heads

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)

        self.W_o = nn.Linear(d_model,d_model)

        self.normalization_factor = 1 / math.sqrt(self.d_model)

    def split_attention_heads(self,x):

        batch_size, sequence_length, d_model = x.size()

        #we split our embedded vectors into num_heads many section of length d_k = d_model//num_heads
        #to calculate the attention scores we need calculate for every entry in the sequence the dot product between the d_k dimensional embedding vectors which are the result of the split
        #to do this we need to change the dim of our tensor from (batch, seq_len, num_heads, d_k) to (batch,num_heads, seq_len, d_k) because torch matmul always takes the last two dims as the matrices to multiply and does this for all other dims 

        return x.view(batch_size,sequence_length,self.num_heads,self.d_k).transpose(1,2)

    def join_attention_heads(self,x):

        batch_size, num_heads, sequence_length, d_k = x.shape

        x = x.transpose(1,2).contiguous().view(batch_size,sequence_length,self.d_model)

        return x

    
    def forward(self,x_q,x_k,x_v,mask = None):
        
        Q = self.split_attention_heads(self.W_q(x_q))
        K = self.split_attention_heads(self.W_k(x_k))
        V = self.split_attention_heads(self.W_v(x_v))

        #print(Q.shape,K.shape)

        beta = torch.matmul(Q,torch.transpose(K,-1,-2)) * self.normalization_factor

        if mask is not None:
            beta = beta.masked_fill(mask,float('-inf'))

        alpha = torch.softmax(beta,dim=-1)

        out = self.join_attention_heads(torch.matmul(alpha,V))

        out = self.W_o(out)

        return out, alpha

class MultiHeadEncoder(nn.Module):
    def __init__(self, n_attention_heads, d_model, d_ff,prob_dropout):
        super(MultiHeadEncoder,self).__init__()

        self.attention = MultiHeadAttention(d_model,n_attention_heads)
        self.ff = FeedForward(d_model,d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(prob_dropout)

    def forward(self,x,mask):

        attn_out, attention_scores = self.attention(x,x,x,mask)

        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)

        x = self.norm2(x + self.dropout(ff_out))

        return x, attention_scores

class TransformerBackbone(nn.Module):
    def __init__(self, n_decoder_layers,n_attention_heads,d_model,d_oct_emb,d_port_emb,d_protoc_emb,d_time_emb,d_ff,prob_dropout,max_sequence_len,num_classes=256):
        super(TransformerBackbone,self).__init__()

        self.d_oct_emb = d_oct_emb
        self.d_model = d_model
        self.num_classes = num_classes

        self.octet_emb = nn.ModuleList([nn.Embedding(num_classes,d_oct_emb) for _ in range(0,4)])

        self.port_emb = nn.Embedding(65536,d_port_emb)

        self.protoc_emb = nn.Embedding(256,d_protoc_emb)

        self.time_emb = nn.Linear(1,d_time_emb)

        self.embedding_ll = nn.Linear(4 * d_oct_emb + d_port_emb + d_protoc_emb + d_time_emb, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)

        self.positional_encoding = PositionalEncoding(d_model,max_seq_len=max_sequence_len)

        self.decoder_layers = nn.ModuleList([MultiHeadEncoder(n_attention_heads,d_model,d_ff,prob_dropout) for _ in range(n_decoder_layers)]) 

    def emb(self, x):
        dst_ips = x['ip.dst'].long()              # (B, T, 4)
        ports = x['port'].long()                  # (B, T)
        protos = x['protocol'].long()             # (B, T)
        timestamps = x['timestamp'].float()       # (B, T)

        # ---- IP embedding ----
        octet_embeddings = [
            emb(dst_ips[:, :, i]) 
            for i, emb in enumerate(self.octet_emb)
        ]
        ip_emb = torch.cat(octet_embeddings, dim=-1)

        # ---- Port embedding ----
        port_emb = self.port_emb(ports)

        # ---- Protocol embedding ----
        proto_emb = self.protoc_emb(protos)

        # ---- Time delta encoding ----
        delta_t = torch.zeros_like(timestamps)        # (B, T)
        delta_t[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]

        # log scaling 
        delta_t = torch.log1p(delta_t)

        # mask padded positions
        delta_t = delta_t.masked_fill(x['masks'], 0.0)

        # project to embedding space
        time_emb = self.time_emb(delta_t.unsqueeze(-1))  # (B, T, d_time_emb)

        # ---- Fuse everything ----
        combined = torch.cat([ip_emb, port_emb, proto_emb, time_emb], dim=-1)

        return self.embedding_norm(self.embedding_ll(combined))

    def create_forward_mask(self,x):
        sequence_length = x.size(-2)
        causal_mask = torch.triu(torch.ones(sequence_length,sequence_length),diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return causal_mask.to(device)
  
    def forward(self,x):

        decoder_out = self.emb(x)

        decoder_out = decoder_out.squeeze(2)

        decoder_out = self.positional_encoding(decoder_out)


        full_mask = self.create_forward_mask(decoder_out)

        if "masks" in x:
            full_mask = x['masks'].unsqueeze(1).unsqueeze(2) | full_mask

        for dec_layer in self.decoder_layers:
            decoder_out, attention_scores = dec_layer(decoder_out,full_mask)

        return decoder_out

class IPSeqClassificationTransformer(nn.Module):
    def __init__(self, backbone,projection_head=None):
        super(IPSeqClassificationTransformer,self).__init__()

        self.backbone = backbone

        self.projection_head = projection_head
    
    def forward(self,x,return_backbone=False):

        decoder_out = self.backbone(x)

        valid_token_mask = (~x['masks']).unsqueeze(-1)


        decoder_out = decoder_out * valid_token_mask

        summed_embeddings = decoder_out.sum(dim=1)
        valid_token_counts = valid_token_mask.sum(dim=1)


        pooled = summed_embeddings / valid_token_counts.clamp(min=1)

        if torch.isnan(pooled).any():
            print("NaN detected in pooled, replacing with zeros")
            pooled = torch.nan_to_num(pooled, nan=0.0)

        if return_backbone or self.projection_head is None:
            print("Returning backbone embeddings")
            return nn.functional.normalize(pooled,p=2,dim=-1,eps=1e-8)


        logit = self.projection_head(pooled)

        if torch.isnan(logit).any():
            print("NaN detected in logit, replacing with zeros")
            logit = torch.nan_to_num(logit, nan=0.0)

        logit = nn.functional.normalize(logit,p=2,dim=-1,eps=1e-8)
    
        return logit
