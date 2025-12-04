import torch
import torch.nn as nn
import numpy as np
import math

class lqformerattention(nn.Module):
    # 多头注意力机制，支持下投影和上投影
    # embed_dim: 输入的embedding维度，也就是down_dim
    # up_dim=2560
    def __init__(self, embed_dim, num_heads, down_dim, up_dim):
        super().__init__()
        self.num_heads = num_heads
        self.down_dim = down_dim
        self.embed_dim = embed_dim
        # 每个头负责自己的那一些维度
        self.down_head_dim = down_dim // num_heads
        self.head_dim = embed_dim // num_heads
        self.up_dim = up_dim
        self.q_proj = nn.Linear(self.down_dim, self.down_dim, bias=True)
        self.k_proj = nn.Linear(self.down_dim, self.down_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        

    def forward(self, query, key, value, attention_mask=None):
        bsz, q_len, _ = query.size()
        k_len = key.size(1)
        v_len = value.size(1)

        # 将最后一个维度拆成了两个：即每个头负责的那部分维度
        # 然后进行转置，使得多头注意力的维度排列为 (batch_size, num_heads, seq_len, head_dim)
        # 即最后两个维度就是对于所有token，每个头需要负责处理的内容
        query = self.q_proj(query).view(bsz, q_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        key = self.k_proj(key).view(bsz, k_len, self.num_heads, self.down_head_dim).transpose(1, 2)
        value = self.v_proj(value).view(bsz, v_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        # 交叉注意力公式，softmax里面的部分
        # 除的是每个头的down_head_dim
        attn_weights = torch.matmul(
            query.to(torch.float32), key.to(torch.float32).transpose(2, 3)
        ) / math.sqrt(self.down_head_dim)

        if attention_mask is not None:
            # 多余的全赋成一个很小的值，softmax后接近0
            # 和计算出的注意力权重加起来（形状一样，因为repeat过，看LQFormerLayer代码）
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -1e4)
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行softmax归一化
        # 最后和v乘起来
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value.dtype)
        attn_output = torch.matmul(attn_weights, value)

        # 重新排列回原来的形状
        # 一直在使用transpose变换形状，这里使用contiguous保持内存连续性
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return attn_output, attn_weights
    
class LQFormerLayer(nn.Module):
    # up_dim是固定值2560
    def __init__(self, d_model, mm_model, n_heads, down_dim, up_dim):
        super(LQFormerLayer, self).__init__()
        self.t2q_attn = lqformerattention(embed_dim=down_dim, num_heads=n_heads, down_dim=down_dim, up_dim=up_dim)
        self.i2q_attn = lqformerattention(embed_dim=d_model, num_heads=n_heads, down_dim=down_dim, up_dim=up_dim)
        # 将指定维度的token进行LayerNorm归一化
        self.ln_text = nn.LayerNorm(down_dim)
        self.ln_q = nn.LayerNorm(down_dim)
        self.ln_kv = nn.LayerNorm(down_dim)
        self.n_heads = n_heads


    # 传入的token(除了image_tokens)均为下投影后的token，最后维度都是down_dim
    def forward(self, learnable_tokens, image_tokens, image_tokens_down, text_tokens, text_mask=None):
        # Down-project learnable tokens and text tokens
        
        # Residual connection for learnable tokens before self-attention
        residual_learnable = learnable_tokens
        
        # Layer norm
        # 对最后一维度进行归一化
        learnable_tokens = self.ln_q(learnable_tokens)
        text_tokens = self.ln_text(text_tokens)
        batch_size = learnable_tokens.size(0)   
        if text_mask is not None:
            attention_mask = text_mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            # 将mask扩展到多头注意力的维度，一共n_heads个头，每个头都看这learnable_tokens.size(1)个query token
            attention_mask = attention_mask.repeat(1, self.n_heads, learnable_tokens.size(1), 1)
        else:
            attention_mask = None
        # 出来的形状和query tokens一样
        attn_output, _ = self.t2q_attn(query=learnable_tokens, key=text_tokens, value=text_tokens, attention_mask=attention_mask)
        
        # Cross-attention: learnable tokens query image tokens
        # 第二次交叉注意力，注意v成了原始的image_tokens，且不再用注意力mask
        # v用原始的是为了信息保真
        image_tokens_down = self.ln_kv(image_tokens_down)
        # 此时最后一个维度已经成了d_model=1152，倒数第二维度还是learnable tokens的个数
        attn_output, attention_map = self.i2q_attn(query=attn_output, key=image_tokens_down, value=image_tokens, attention_mask=None)
        
        # attention_map中每个头还没合并，这里对每个头的那一维度进行平均，得到最终的attention map
        # 头部以下合并，最后也是和attn_output一样的形状了
        attention_map = torch.mean(attention_map, dim=1)
        return attn_output, attention_map

class LQFormer(nn.Module):
    def __init__(self, config, num_layers=1):
        super(LQFormer, self).__init__()
        self.mm_model = config.hidden_size
        self.d_model = 1152
        self.down_dim = 576
        # 第一次交叉注意力（看名字）的下投影，从llm自己的hidden size投到lqformer的down_dim
        self.down_projector_learnable_text = nn.Linear(self.mm_model, self.down_dim, bias=True)
        # 第二次交叉注意力（看名字）的下投影，从d_model投到down_dim
        self.down_projector_image = nn.Linear(self.d_model, self.down_dim, bias=True)
        # 初始化LQFormerLayer层
        self.layers = nn.ModuleList([LQFormerLayer(mm_model=self.mm_model, d_model = 1152, n_heads=config.num_attention_heads, down_dim = 576, up_dim = 2560) for _ in range(num_layers)])
        # 上投影，把lqformer的输出d_model投回llm的hidden size
        self.up_projector = nn.Linear(self.d_model, self.mm_model)

    # 这里的token都是llm的hidden size
    # 文字描述定长，mask中实际有的位置为1，多余的是0
    # 注意原始图像特征的hidden_size不等于llm的，他的是d_model=1152
    def forward(self, learnable_tokens, image_tokens, text_tokens, text_mask=None):
        # query tokens下投影：llm_hidden_size -> down_dim
        learnable_tokens_down = self.down_projector_learnable_text(learnable_tokens)
        # 同理，text tokens下投影：llm_hidden_size -> down_dim
        text_tokens_down = self.down_projector_learnable_text(text_tokens)
        # image tokens下投影：d_model -> down_dim
        image_tokens_down = self.down_projector_image(image_tokens)
        # Pass through the layers
        for layer in self.layers:
            residual = learnable_tokens
            learnable_tokens, attention_map = layer(learnable_tokens_down, image_tokens, image_tokens_down, text_tokens_down, text_mask)
            # 上投影：d_model -> llm_hidden_size
            learnable_tokens = self.up_projector(learnable_tokens)
            # 最后和原来的learnable tokens做残差连接（也就是加起来了）
            learnable_tokens = residual + learnable_tokens
        return learnable_tokens
