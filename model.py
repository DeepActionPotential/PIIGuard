

import torch.nn as nn
from torchcrf import CRF

class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels, pad_idx=0, pad_label_id=-100):
        super().__init__()
        self.pad_label_id = pad_label_id
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layer for projecting to label space
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, tags=None, mask=None):
        embeds = self.embedding(input_ids)            # [B, L, E]
        lstm_out, _ = self.lstm(embeds)               # [B, L, 2*H]
        emissions = self.hidden2tag(lstm_out)         # [B, L, num_labels]
        
        if tags is not None:
            # Convert ignored labels to 0 for CRF
            crf_tags = tags.clone()
            crf_tags[crf_tags == self.pad_label_id] = 0
            
            # Negative log likelihood
            loss = -self.crf(emissions, crf_tags, mask=mask, reduction='mean')
            return loss
        else:
            # Decode (Viterbi) paths
            return self.crf.decode(emissions, mask=mask)