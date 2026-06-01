import sys
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn

from transformer import IPSeqClassificationTransformer
from data import DataManager

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_contrastive_sequence_comparison(train_datamanager:DataManager,model:IPSeqClassificationTransformer,checkpoint_path,n_steps=100,lr=1e-3):

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    model.train()

    print("Starting contrastive training...")

    progress = tqdm(
        total=n_steps,
        colour='green',
        file=sys.stdout,
        desc=f'Training '
    )

    for step_idx, (seq_1,seq_2) in enumerate(train_datamanager):

        optimizer.zero_grad()

        for key, val in seq_1.items():
            seq_1[key] = val.to(device)
            
        for key, val in seq_2.items():
            seq_2[key] = val.to(device)


        inputs_seq_1 = {}
        inputs_seq_2 = {}


        inputs_seq_1 = seq_1
        inputs_seq_2 = seq_2

        embeddings_seq_1 = model(inputs_seq_1)
        embeddings_seq_2 = model(inputs_seq_2)

        effective_batch_size = embeddings_seq_1.size(0)

        E = torch.cat((embeddings_seq_1,embeddings_seq_2),dim=0)

        similarity_matrix = torch.matmul(E,E.T)

        self_similarity_mask = torch.eye(similarity_matrix.size(0),dtype=torch.bool).to(device)
        similarity_matrix = similarity_matrix.masked_fill(self_similarity_mask,float('-inf'))

        positive_indices = torch.arange(effective_batch_size, device=E.device)
        positive_indices = torch.cat((positive_indices + effective_batch_size, positive_indices), dim=0)

        temperature = 0.07
        logits = similarity_matrix / temperature

        loss = nn.functional.cross_entropy(logits, positive_indices)

        loss.backward()
        optimizer.step()

        progress.update(1)
        progress.set_postfix({"Loss": loss.item()})

        if step_idx % 10_000 == 0:
            torch.save(model.state_dict(),checkpoint_path + 'transformer_contrast_comp_weights' + str(step_idx) + '.pth') 

        if step_idx >= n_steps - 1:
            break

    torch.save(model.state_dict(),checkpoint_path + 'transformer_contrast_comp_weights.pth') 
