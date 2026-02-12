import torch
from config import GPTConfig, TrainConfig
from model import GPTLanguageModel

def test_training_loop():
    # Setup minimal config for testing
    gpt_config = GPTConfig(
        vocab_size=100,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=32
    )
    
    train_config = TrainConfig(
        batch_size=4,
        max_iters=5,
        eval_interval=2,
        eval_iters=1,
        learning_rate=1e-3,
        device='cpu' # Use CPU for simple test
    )
    
    print("Initializing model...")
    model = GPTLanguageModel(gpt_config)
    model.to(train_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    
    print("Generating dummy data...")
    # Create dummy data
    data = torch.randint(0, gpt_config.vocab_size, (100,), dtype=torch.long)
    train_data = data
    val_data = data

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - gpt_config.block_size, (train_config.batch_size,))
        x = torch.stack([data[i:i+gpt_config.block_size] for i in ix])
        y = torch.stack([data[i+1:i+gpt_config.block_size+1] for i in ix])
        return x.to(train_config.device), y.to(train_config.device)

    @torch.no_grad()
    def estimate_loss(model):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(train_config.eval_iters)
            for k in range(train_config.eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    print("Starting training loop...")
    initial_loss = estimate_loss(model)['train']
    print(f"Initial loss: {initial_loss:.4f}")

    for iter in range(train_config.max_iters):
        if iter % train_config.eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
    final_loss = estimate_loss(model)['train']
    print(f"Final loss: {final_loss:.4f}")
    
    print("Training loop verified successfully.")

if __name__ == "__main__":
    test_training_loop()
