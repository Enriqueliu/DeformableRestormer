import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from timm.models.layers import to_2tuple, trunc_normal_
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from torch.cuda.amp import autocast, GradScaler


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class LayerNormProxy(nn.Module):
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return rearrange(x, 'b h w c -> b c h w')

##########################################################################
## FFN , same as ViT
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, activate):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        if activate == "gelu":
            self.activate = nn.GELU()
        elif activate == "leakyrelu":
            self.activate = nn.LeakyReLU(0.1)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.activate(x)
        x = self.project_out(x)
        return x





##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################

def block_images_einops(x, patch_size, phase):
    """Image to patches."""
    b, c, height, width = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    if phase == "local":
        x = rearrange(
            x, "n c (gh fh) (gw fw) -> (n gh gw) c fh fw",
            n=b, c=c, gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    elif phase == "global":
        x = rearrange(
            x, "n c (gh fh) (gw fw) -> (n fh fw) c gh gw",
            n=b, c=c, gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])

    return x

def unblock_images_einops(x, grid_size, patch_size, phase):
    """patches to images."""
    if phase == "local":
        x = rearrange(
            x, "(n gh gw) c fh fw -> n c (gh fh) (gw fw)",
            gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    elif phase == "global":
        x = rearrange(
            x, "(n fh fw) c gh gw -> n c (gh fh) (gw fw)",
            gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x

## Deformable attention
class DeformableAttention(nn.Module):
    def __init__(self, n_heads, n_channel, n_groups, offset_range_factor=1, use_pe=True, dwc_pe=False, fixed_pe=False, q_size=[16, 16], kv_size=[16, 16], activate = "gelu"):
        super(DeformableAttention, self).__init__()
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_channel
        self.n_head_channels = self.nc // n_heads
        self.n_heads = n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.scale = self.n_head_channels ** -0.5
        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.fixed_pe = fixed_pe
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups

        if activate == "gelu":
            self.conv_offset = nn.Sequential(
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, 5, 4, 2, groups=self.n_group_channels, bias=False),
                    LayerNorm(self.n_group_channels, 'BiasFree'),
                    nn.GELU(),
                    nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
            )
        elif activate == "leakyrelu":
            self.conv_offset = nn.Sequential(
                    nn.Conv2d(self.n_group_channels, self.n_group_channels, 5, 4, 2, groups=self.n_group_channels, bias=False),
                    LayerNorm(self.n_group_channels, 'BiasFree'),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False)
            )
        

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0,bias=False
        )

        # TODO: position info
        if self.use_pe:
            if self.dwc_pe:
                self.rpe_table = nn.Conv2d(self.nc, self.nc, 
                                           kernel_size=3, stride=1, padding=1, groups=self.nc,bias=False)
            elif self.fixed_pe:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.q_h * self.q_w, self.kv_h * self.kv_w)
                )
                trunc_normal_(self.rpe_table, std=0.01)
            else:
                self.rpe_table = nn.Parameter(
                    torch.zeros(self.n_heads, self.kv_h * 2 - 1, self.kv_w * 2 - 1)
                )
                trunc_normal_(self.rpe_table, std=0.01)
        else:
            self.rpe_table = None
         
    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device), 
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device)
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1) # B * g H W 2
        
        return ref
    
    def forward(self, x):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)

        q_off = rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels) # 将q分为多个group

        offset = self.conv_offset(q_off) # B * g 2 Hg Wg
        
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk # n_sample个reference points

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor) # delta{p} = s * tanh(delta{p})

        offset = rearrange(offset, 'b p h w -> b h w p') # B * g 2 Hg Wg -> B*g Hg Wg 2
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).tanh()
        
        x_sampled = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W), 
            grid=pos[..., (1, 0)], # y, x -> x, y
            mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        # print(q.shape)
        # print(B, self.n_heads, self.n_head_channels, H*W)
        q = q.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, N_sample
        attn = attn.mul(self.scale)
        
        #TODO: position info
        if self.use_pe:
            if self.dwc_pe:
                residual_lepe = self.rpe_table(q.reshape(B, C, H, W)).reshape(B * self.n_heads, self.n_head_channels, H * W)
            elif self.fixed_pe:
                rpe_table = self.rpe_table
                attn_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                attn = attn + attn_bias.reshape(B * self.n_heads, H * W, self.n_sample)
            else:
                rpe_table = self.rpe_table
                rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
                
                q_grid = self._get_ref_points(H, W, B, dtype, device)
                
                displacement = (q_grid.reshape(B * self.n_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.n_groups, n_sample, 2).unsqueeze(1)).mul(0.5)
                
                attn_bias = F.grid_sample(
                    input=rpe_bias.reshape(B * self.n_groups, self.n_group_heads, 2 * H - 1, 2 * W - 1),
                    grid=displacement[..., (1, 0)],
                    mode='bilinear', align_corners=True
                ) # B * g, h_g, HW, Ns
                
                attn_bias = attn_bias.reshape(B * self.n_heads, H * W, n_sample)
                
                attn = attn + attn_bias

        attn = F.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        if self.use_pe and self.dwc_pe:
            out = out + residual_lepe
        out = out.reshape(B, C, H, W)
        #TODO: Dropout
        out = self.proj_out(out)
        return out

##########################################################################
## SK Fusion Layer
class SKFusion(nn.Module):
	def __init__(self, dim, n_inputs=2, reduction=8, activate="gelu"):
		super(SKFusion, self).__init__()
		
		self.n_inputs = n_inputs
		d = max(int(2*dim/reduction), 4)
		
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		
		if activate == "gelu":
			self.mlp = nn.Sequential(
			nn.Conv2d(2*dim, d, 1, bias=False), 
			nn.GELU(),
			nn.Conv2d(d, dim*2, 1, bias=False)
			)	
		elif activate == "leakyrelu":
			self.mlp = nn.Sequential(
				nn.Conv2d(2*dim, d, 1, bias=False), 
				nn.LeakyReLU(0.1),
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

class MAB_Local(nn.Module):
    def __init__(self, n_channel, patch_size, n_heads, n_groups, activate):
        super(MAB_Local, self).__init__()
        self.patch_size = patch_size
        # self.gmlp = Local_gMLP(num_channel, patch_size[0]*patch_size[1])
        self.deformable_attn = DeformableAttention(n_heads=n_heads, n_channel=n_channel, n_groups=n_groups, q_size=patch_size, kv_size=patch_size, activate=activate)
    def forward(self, x):
        # n, h, w, c = x.shape
        n, c, h, w = x.shape
        patch_h, patch_w = self.patch_size
        grid_h, grid_w = h // patch_h, w // patch_w
        x = block_images_einops(x, patch_size=(patch_h, patch_w), phase="local")
        x = self.deformable_attn(x)
        x = unblock_images_einops(x, grid_size=(grid_h, grid_w), patch_size=(patch_h, patch_w), phase="local")
        return x

class MAB_Global(nn.Module):
    def __init__(self, n_channel, grid_size, n_heads, n_groups, activate):
        super(MAB_Global, self).__init__()
        self.grid_size = grid_size
        # self.gmlp = Global_gMLP(num_channel, grid_size[0]*grid_size[1])
        self.deformable_attn = DeformableAttention(n_heads=n_heads, n_channel=n_channel, n_groups=n_groups, q_size=grid_size, kv_size=grid_size, activate=activate)

    def forward(self, x): # n, c, h, w -> n, c, h, w 
        n, c, h, w = x.shape
        grid_h, grid_w = self.grid_size
        patch_h, patch_w = h // grid_h, w // grid_h
        x = block_images_einops(x, patch_size=(patch_h, patch_w), phase="global")
        x = self.deformable_attn(x)
        x = unblock_images_einops(x, grid_size=(grid_h, grid_w), patch_size=(patch_h, patch_w), phase="global")
        return x

class MAB(nn.Module):
    def __init__(self, num_channel, n_heads, n_groups, patch_size=[16,16], grid_size=[16,16], activate="gelu"):
        super(MAB, self).__init__()
        self.norm = LayerNorm(num_channel, 'BiasFree')
        self.conv1 = nn.Conv2d(in_channels=num_channel, out_channels=2*num_channel,kernel_size=1,stride=1,padding=0,bias=False)
        # self.simple_gate = SimpleGate()
        if activate == "gelu":
            self.activate = nn.GELU()
        elif activate == "leakyrelu":
            self.activate = nn.LeakyReLU(0.1)

        self.local_branch = MAB_Local(n_channel=num_channel, patch_size=patch_size, n_heads=n_heads, n_groups=n_groups, activate=activate)
        self.global_branch = MAB_Global(n_channel=num_channel, grid_size=grid_size, n_heads=n_heads, n_groups=n_groups, activate=activate)
        # self.conv2 = nn.Conv2d(in_channels=2*num_channel, out_channels=num_channel,kernel_size=1,stride=1,padding=0,bias=False)
        self.sk_block = SKFusion(dim=num_channel,activate=activate)
    def forward(self, x): # n, c, h, w
        short_cut = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.activate(x)

        x_local, x_global = x.chunk(2, dim=1)
        x_local = self.local_branch(x_local)
        x_global = self.global_branch(x_global)

        x = self.sk_block([x_local, x_global])
        # x = torch.cat([x_local, x_global], dim=1)
        # x = self.conv2(x)
        # x = self.ca_block(x)
        return x + short_cut

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_groups, ffn_expansion_factor, bias, LayerNorm_type, patch_size, grid_size, activate):
        super(TransformerBlock, self).__init__()
        self.deformable_MAB = MAB(dim, num_heads, num_groups, patch_size, grid_size, activate)
        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias, activate)

    def forward(self, x):
        x = self.deformable_MAB(x)
        # x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class deformableSK(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 32,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [2,4,8,16],
        groups = [1,2,4,8],
        ffn_expansion_factor = 4,
        bias = False,
        LayerNorm_type = 'BiasFree',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        activate = 'leakyrelu'               ## Other option 'leakyrelu'
    ):

        super(deformableSK, self).__init__()
        # def __init__(self, dim, num_heads, num_groups, ffn_expansion_factor, bias, LayerNorm_type, patch_size, grid_size)
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], num_groups=groups[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[16,16],grid_size=[16,16], activate=activate) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], num_groups=groups[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[16,16],grid_size=[16,16], activate=activate) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], num_groups=groups[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[8,8],grid_size=[8,8], activate=activate) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], num_groups=groups[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[8,8],grid_size=[8,8], activate=activate) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], num_groups=groups[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[8,8],grid_size=[8,8], activate=activate) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], num_groups=groups[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[16,16],grid_size=[16,16], activate=activate) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], num_groups=groups[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[16,16],grid_size=[16,16], activate=activate) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], num_groups=groups[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, patch_size=[16,16],grid_size=[16,16], activate=activate) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

# model = deformableSK()
# # x = torch.rand(1, 3, 256, 256)
# # print(model(x).shape)

# tensor = (torch.rand(1, 3, 256, 256))
# flops = FlopCountAnalysis(model, tensor)
# print("FLOPs: ", flops.total())
# print(parameter_count_table(model))