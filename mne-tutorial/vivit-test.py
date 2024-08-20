import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from module import Attention, PreNorm, FeedForward
import numpy as np

# from https://github.com/rishikksh20/ViViT-pytorch 

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth): # depth에 따라 Attention, FFN 몇 번 진행할지
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)), #  Normalization 먼저 하고, Attention 진행
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)) #  Normalization 먼저 하고, MLP 진행
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x # Norm -> Att -> Residual Conn
            x = ff(x) + x # Norm -> FFN -> Residual Conn
        return self.norm(x) # 마지막으로 Norm 


  
class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, num_frames, dim = 192, depth = 4, heads = 3, pool = 'cls', in_channels = 3, dim_head = 64, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'


        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2 # image도 정사각형, Patch도 정사각형
        patch_dim = in_channels * patch_size ** 2 # patch가 채널 개수만큼 있겠지
        self.to_patch_embedding = nn.Sequential( 
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size), # 어떤 수 5개가 곱해진 dimension을 이런 식으로 변형한다. 결국에는 어떤 수 4개가 곱해지도록
            # b : 비디오 수 / t : 프레임 수 / c : 채널 수 (RGB : 3개) 
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim)) # 각 프레임에서의 패치 수. +1은 For CLS token
        # nn.Parameter 로 감싸진 텐서는 모델의 학습 과정에서 자동으로 최적화
        self.space_token = nn.Parameter(torch.randn(1, 1, dim)) 
        # 공간적 처리를 위한 CLS 토큰. dim = 토큰 임베딩의 차원, 트랜스포머 입력 차원과 일치
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        # 시간적 처리를 위한 CLS 토큰. 
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape 
        # 배치 개수 / 프레임 개수 / 패치 개수 / 패치 길이 * 패치 길이 * 채널 수 (디멘션)

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        # cls_space_tokens의 shape는 (batch_size, time_frames, 1, patch_embedding_dim)
        x = torch.cat((cls_space_tokens, x), dim=2)
        # x의 shape는 (batch_size, time_frames, num_patches + 1, patch_embedding_dim)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)
        

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
    
    
    

if __name__ == "__main__":
    
    img = torch.ones([1, 16, 3, 224, 224]).cuda()
    
    model = ViViT(224, 16, 100, 16).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    
    out = model(img)
    
    print("Shape of out :", out.shape)      # [B, num_classes]