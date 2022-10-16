import torch.nn as nn
import torch

class SKFusion(nn.Module):
	def __init__(self, dim, n_inputs=2, reduction=8):
		super(SKFusion, self).__init__()
		
		self.n_inputs = n_inputs
		d = max(int(2*dim/reduction), 4)
		
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.mlp = nn.Sequential(
			nn.Conv2d(2*dim, d, 1, bias=False), 
			nn.ReLU(),
			nn.Conv2d(d, dim*2, 1, bias=False)
		)
		
		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats): # 2, B, C, H, W -> B, C ,H ,W
		B, C, H, W = in_feats[0].shape
		
		in_feats = torch.cat(in_feats, dim=1)
		# in_feats = in_feats.view(B, self.height, C, H, W) 
		# print(in_feats.shape)
		# feats_sum = torch.sum(in_feats, dim=1)
		# print(feats_sum.shape)
		attn = self.mlp(self.avg_pool(in_feats))
		
        
		attn = self.softmax(attn.view(B, self.n_inputs, C, 1, 1))
		
		in_feats = in_feats.view(B, self.n_inputs, C, H, W) 
		out = torch.sum(in_feats*attn, dim=1)
		
		return out      	

model = SKFusion(dim=64)  
x1 = torch.rand(2,64,32,32)
x2 = torch.rand(2,64,32,32)
out = model([x1, x2])
# print(out.shape)