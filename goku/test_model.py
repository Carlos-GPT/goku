import torch
from model import AdaLayerNorm, FeedForward, RMSNorm, Attention, Block

def test_model():
    print("Starting test_model...")

    # Teste AdaLayerNorm
    print("Testing AdaLayerNorm...")
    ada_layer_norm = AdaLayerNorm(embedding_dim=128)
    x = torch.randn(2, 10, 128)
    timestep = torch.randn(2, 128)
    output = ada_layer_norm(x, timestep)
    print("AdaLayerNorm output:", output)

    # Teste FeedForward
    print("Testing FeedForward...")
    feed_forward = FeedForward(dim=128)
    hidden_states = torch.randn(2, 10, 128)
    output = feed_forward(hidden_states)
    print("FeedForward output:", output)

    # Teste RMSNorm
    print("Testing RMSNorm...")
    rms_norm = RMSNorm(dim=128)
    x = torch.randn(2, 10, 128)
    output = rms_norm(x)
    print("RMSNorm output:", output)

    # Teste Attention
    print("Testing Attention...")
    attention = Attention(q_dim=128)
    inputs_q = torch.randn(2, 10, 128)
    inputs_kv = torch.randn(2, 10, 128)
    output = attention(inputs_q, inputs_kv)
    print("Attention output:", output)

    # Teste Block
    print("Testing Block...")
    block = Block(dim=128, num_attention_heads=8, attention_head_dim=16)
    hidden_states = torch.randn(2, 10, 128)
    timestep = torch.randn(2, 128)
    output = block(hidden_states, timestep=timestep)
    print("Block output:", output)

if __name__ == "__main__":
    print("Running test_model.py...")
    test_model()
    print("Finished test_model.")