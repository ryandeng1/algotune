import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, np.ndarray]:
        signal_x = np.array(problem["signal_x"])
        signal_y = np.array(problem["signal_y"])
        mode = problem.get("mode", "full")
        
        m, n = len(signal_x), len(signal_y)
        
        if m == 0 or n == 0:
            return {"convolution": np.array([])}
        
        linear_conv_len = m + n - 1
        
        # Find smallest power of two >= linear_conv_len
        N = 1
        while N < linear_conv_len:
            N *= 2
        
        # Pad signals to length N
        x_padded = np.pad(signal_x, (0, N - m), 'constant')
        y_padded = np.pad(signal_y, (0, N - n), 'constant')
        
        # Compute FFT and multiply
        X = np.fft.fft(x_padded)
        Y = np.fft.fft(y_padded)
        Z = X * Y
        
        # Inverse FFT
        z = np.fft.ifft(Z)
        
        # Extract linear convolution (full)
        linear_conv = z.real[:linear_conv_len]
        
        # Apply mode
        if mode == "full":
            result = linear_conv
        elif mode == "same":
            start = (linear_conv_len - max(m, n)) // 2
            result = linear_conv[start:start + max(m, n)]
        elif mode == "valid":
            if m >= n:
                start = m - n
                result = linear_conv[start:start + (m - n + 1)]
            else:
                start = n - m
                result = linear_conv[start:start + (n - m + 1)]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return {"convolution": result}
