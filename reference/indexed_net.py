import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 1. CNN Backbone
# =========================
class Backbone(nn.Module):
    def __init__(self, in_channels=32, hidden=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: [B, C, 17, 17]
        return self.conv(x)  # [B, H, 17, 17]


# =========================
# 2. Encode 150 legal positions
# =========================
class PositionIndexer:
    """
    负责：
    17x17 -> 150 indexing
    """

    def __init__(self, legal_mask_17x17):
        """
        legal_mask_17x17: [17,17] bool tensor
        """
        self.legal_pos = legal_mask_17x17.nonzero(as_tuple=False)  # [150,2]

    def gather(self, feat):
        """
        feat: [B, C, 17,17]
        return: [B, 150, C]
        """
        B, C, _, _ = feat.shape
        idx = self.legal_pos.to(feat.device)  # [150,2]

        out = []
        for b in range(B):
            f = feat[b]  # [C,17,17]
            vals = f[:, idx[:, 0], idx[:, 1]]  # [C,150]
            out.append(vals.transpose(0, 1))   # [150,C]

        return torch.stack(out, dim=0)  # [B,150,C]


# =========================
# 3. Policy Head (bilinear graph)
# =========================
class PolicyHead(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()

        self.project = nn.Linear(hidden, hidden, bias=False)

    def forward(self, node_feat):
        """
        node_feat: [B,150,C]
        return: [B,150,150]
        """

        B, N, C = node_feat.shape

        # (1) project features
        h = self.project(node_feat)  # [B,150,C]

        # (2) score matrix = dot product
        # score(i,j) = h_i · node_j
        logits = torch.matmul(h, node_feat.transpose(1, 2))  # [B,150,150]

        return logits


# =========================
# 4. Full Model
# =========================
class XiangqiMCTSModel(nn.Module):
    def __init__(self, legal_mask):
        super().__init__()

        self.backbone = Backbone()
        self.indexer = PositionIndexer(legal_mask)
        self.policy = PolicyHead()

        self.value_head = nn.Sequential(
            nn.Linear(128 * 17 * 17, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        x: [B, C, 17,17]
        """

        feat = self.backbone(x)  # [B,128,17,17]

        # ---- policy ----
        nodes = self.indexer.gather(feat)  # [B,150,128]
        policy_logits = self.policy(nodes)  # [B,150,150]

        # ---- value ----
        v = feat.flatten(1)
        value = self.value_head(v)

        return policy_logits, value