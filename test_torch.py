try:
    print("Testing torch import...")
    import torch
    print(f"torch: {torch.__version__}")
    
    print("Testing torch tensor operation...")
    x = torch.randn(5, 5)
    y = x @ x.T
    print("torch operation: OK")
except Exception as e:
    print(f"FAILED: {e}")
except BaseException as e:
    print(f"CRITICAL FAILED: {type(e)}")
