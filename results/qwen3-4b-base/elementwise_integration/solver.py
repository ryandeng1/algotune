import numpy as np
from scipy.special import gamma as scipy_gamma

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        a = np.array(problem["a"])
        b = np.array(problem["b"])
        lower = np.array(problem["lower"])
        upper = np.array(problem["upper"])
        n = len(a)
        results = np.zeros(n)
        tolerance = 1e-10
        
        for i in range(n):
            a_i = a[i]
            b_i = b[i]
            low_i = lower[i]
            up_i = upper[i]
            
            total = 0.0
            k = 0
            while True:
                num = up_i**(k+1) - low_i**(k+1)
                fact = np.factorial(k+1)
                gamma_val = scipy_gamma(a_i * k + b_i)
                denom = fact * gamma_val
                term = num / denom
                
                total += term
                
                if abs(term) < tolerance:
                    break
                
                k += 1
            
            results[i] = total
        
        return {"result": results.tolist()}
