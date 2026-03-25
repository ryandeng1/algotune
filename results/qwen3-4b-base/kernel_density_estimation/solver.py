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
                raise ValueError("Data points or query points are not 2D arrays.")
            if X.shape[0] == 0:
                raise ValueError("No data points provided.")
            if X_q.shape[0] == 0:
                return {"log_density": []}
            if X.shape[1] != X_q.shape[1]:
                raise ValueError("Data points and query points have different dimensions.")
            
            if not isinstance(bandwidth, (int, float)) or bandwidth <= 0:
                raise ValueError("Bandwidth must be positive.")
            bandwidth = float(bandwidth)
            
            n, d = X.shape
            m = X_q.shape[0]
            diff = X[:, None, :] - X_q[None, :, :]
            squared_dist = np.sum(diff**2, axis=2)
            
            if kernel == "gaussian":
                exp_term = np.exp(-squared_dist / (2 * bandwidth**2))
                sum_exp = np.sum(exp_term, axis=0)
                log_density = (-d * np.log(bandwidth * np.sqrt(2 * np.pi)) 
                               - np.log(n) 
                               + np.log(sum_exp))
            elif kernel == "tophat":
                mask = squared_dist <= bandwidth**2
                sum_tophat = np.sum(mask, axis=0)
                log_density = np.log(sum_tophat) - np.log(n) - d * np.log(bandwidth)
            elif kernel == "epanechnikov":
                mask = squared_dist <= bandwidth**2
                term = np.where(mask, 1 - squared_dist / bandwidth**2, 0)
                sum_ep = np.sum(term, axis=0)
                log_density = np.log(3 * sum_ep / (4 * bandwidth**2)) - np.log(n)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            return {"log_density": log_density.tolist()}
        except KeyError as e:
            return {"error": f"Missing key: {e}"}
        except (ValueError, TypeError) as e:
            return {"error": f"Computation error: {e}"}
