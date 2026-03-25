import numpy as np

class Solver:
    def solve(self, problem: dict[str, Any]) -> dict[str, Any]:
        try:
            X = np.array(problem["data_points"])
            X_q = np.array(problem["query_points"])
            kernel = problem["kernel"]
            bandwidth = problem["bandwidth"]
            num_points = problem["num_points"]
            num_query_points = problem["num_query_points"]
            dims = problem["dims"]
            
            if X.ndim != 2 or X_q.ndim != 2:
                return {"error": "Data points or query points are not 2D arrays."}
                
            if X.shape[0] == 0:
                return {"error": "No data points provided."}
                
            if X_q.shape[0] == 0:
                return {"log_density": []}
                
            if X.shape[1] != X_q.shape[1]:
                return {"error": "Data points and query points have different dimensions."}
                
            if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
                return {"error": "Bandwidth must be positive."}
                
            bandwidth = float(bandwidth)
            
            def gaussian_log_density(X, X_q, h, d):
                diff = X_q[:, None, :] - X[None, :, :]
                squared_dist = np.sum(diff**2, axis=2)
                a = -squared_dist / (2 * h**2)
                a_max = np.max(a, axis=1)
                b = a - a_max[:, None]
                exp_b = np.exp(b)
                sum_exp = np.sum(exp_b, axis=1)
                log_sum = a_max + np.log(sum_exp)
                log_term = -np.log(X.shape[0]) - (d / 2) * np.log(2 * np.pi * h**2)
                return log_term + log_sum
            
            def tophat_log_density(X, X_q, h, d):
                diff = X_q[:, None, :] - X[None, :, :]
                squared_dist = np.sum(diff**2, axis=2)
                mask = squared_dist <= h**2
                count = np.sum(mask, axis=1)
                return np.where(count > 0, -np.log(X.shape[0]) + np.log(count) - d * np.log(h), -np.inf)
            
            def epanechnikov_log_density(X, X_q, h, d):
                diff = X_q[:, None, :] - X[None, :, :]
                squared_dist = np.sum(diff**2, axis=2)
                mask = squared_dist <= h**2
                sum_val = np.sum((1 - squared_dist / (h**2)) * mask, axis=1)
                return np.where(sum_val > 0, -np.log(X.shape[0]) + np.log(sum_val) - np.log(4 * h**2), -np.inf)
            
            kernel_func = {
                'gaussian': gaussian_log_density,
                'tophat': tophat_log_density,
                'epanechnikov': epanechnikov_log_density
            }.get(kernel)
            
            if kernel_func is None:
                return {"error": f"Unknown kernel: {kernel}"}
            
            log_density = kernel_func(X, X_q, bandwidth, dims)
            
            return {"log_density": log_density.tolist()}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}
