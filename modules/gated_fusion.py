import easydict

import torch
import torch.nn as nn
import torch.nn.functional as F

class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.img_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_img_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_img_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_img_feat = torch.tanh(self.img_linear(att_img_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

        gate_img = self.gate_linear(torch.cat([new_img_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_img = torch.sigmoid(gate_img)  # [batch_size, max_seq_len, 1]
        gate_img = gate_img.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_img, new_img_feat) + torch.mul(1 - gate_img, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features
    
class GatedFusion(nn.Module):
    
    def __init__(self, hdim=768):
        
        super(GatedFusion, self).__init__()
        
        self.gate_1 = nn.Linear(hdim * 2, hdim)
        self.gate_2 = nn.Linear(hdim * 2, hdim)
        
        self.layer_norm = nn.LayerNorm(hdim)
   
    def forward(self, ftrs1, ftrs2):
        
        ftrs1_weight = F.sigmoid(self.gate_1(torch.cat((ftrs1, ftrs2), dim=1)))
        ftrs2_weight = F.sigmoid(self.gate_2(torch.cat((ftrs1, ftrs2), dim=1)))
        
        return self.layer_norm(
            ftrs1 * ftrs1_weight + ftrs2 * ftrs2_weight
        )
    
        