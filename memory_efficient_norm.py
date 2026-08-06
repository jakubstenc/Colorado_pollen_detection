import numpy as np

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Memory efficient per-channel 0.5-99.5 percentile contrast stretch -> uint8 RGB."""
    out = np.zeros_like(arr, dtype=np.uint8)
    
    # Subsample for percentile computation if image is huge
    stride = max(1, min(arr.shape[0], arr.shape[1]) // 1000)
    
    for c in range(arr.shape[-1]):
        # Subsample to estimate percentiles
        sample_ch = arr[::stride, ::stride, c].astype(np.float32)
        valid_sample = sample_ch[sample_ch > 0]
        
        if valid_sample.size == 0:
            lo, hi = 0, 255
        else:
            lo, hi = np.percentile(valid_sample, [0.5, 99.5])
            
        print(f"Channel {c} lo: {lo}, hi: {hi}")
        
        # Process in chunks to save memory
        chunk_size = 2000
        for y in range(0, arr.shape[0], chunk_size):
            y_end = min(y + chunk_size, arr.shape[0])
            for x in range(0, arr.shape[1], chunk_size):
                x_end = min(x + chunk_size, arr.shape[1])
                
                chunk = arr[y:y_end, x:x_end, c].astype(np.float32)
                chunk = np.clip((chunk - lo) / (hi - lo + 1e-8), 0, 1) * 255.0
                out[y:y_end, x:x_end, c] = chunk.astype(np.uint8)
                
    return out

if __name__ == "__main__":
    arr = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
    out = normalize_to_uint8(arr)
    print(out.shape)
