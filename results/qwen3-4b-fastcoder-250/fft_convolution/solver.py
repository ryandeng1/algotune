import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, list]:
        signal_x = np.array(problem["signal_x"])
        signal_y = np.array(problem["signal_y"])
        mode = problem.get("mode", "full")
        
        len_x = len(signal_x)
        len_y = len(signal_y)
        
        if len_x == 0 or len_y == 0:
            return {"convolution": []}
        
        N = 1
        while N < len_x + len_y - 1:
            N *= 2
        
        x_padded = np.pad(signal_x, (0, N - len_x), 'constant')
        y_padded = np.pad(signal_y, (0, N - len_y), 'constant')
        
        X = np.fft.fft(x_padded)
        Y = np.fft.fft(y_padded)
        Z = X * Y
        z = np.fft.ifft(Z)
        
        if mode == "full":
            result = z.real[:len_x + len_y - 1]
        elif mode == "same":
            output_len = max(len_x, len_y)
            start = (N - output_len) // 2
            result = z.real[start:start + output_len]
        elif mode == "valid":
            if len_x >= len_y:
                output_len = max(0, len_x - len_y + 1)
            else:
                output_len = max(0, len_y - len_x + 1)
            start = (N - output_len) // 2
            result = z.real[start:start + output_len]
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return {"convolution": result.tolist()}
