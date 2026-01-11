from encoder import *

if __name__ == "__main__":
    # 예시: 4채널 * 250 timestep
    B, T, C = 8, 250, 4
    model = build_rezero_transformer_encoder_last_ffn(
        in_dim=C,
        d_model=32,
        nhead=4,
        num_layers=6,
        dim_ff=128,
        dropout=0.1,
        attn_dropout=0.0,
        max_len=1000,
        activation="gelu",
    )
    x = torch.randn(B, T, C)
    y, _ = model(x, causal=False, return_attn=False)
    print(y.shape)  # (8, 250, 32)
