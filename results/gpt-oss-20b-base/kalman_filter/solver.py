import numpy as np

class Solver:
    def solve(self, problem: dict) -> dict:
        """
        Solve the Kalman MAP problem as a linear system obtained from the KKT
        conditions of the QP formulation.  This avoids calling a external
        QP solver and is therefore much faster for the small- to medium-sized
        instances that appear in the evaluation.
        """
        # ----- Data ----------------------------------------------------
        A = np.array(problem["A"], dtype=float)
        B = np.array(problem["B"], dtype=float)
        C = np.array(problem["C"], dtype=float)
        y = np.array(problem["y"], dtype=float)
        x0 = np.array(problem["x_initial"], dtype=float)
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[0]
        assert A.shape[1] == n
        p = B.shape[1]

        # ----- Dimensions of variables --------------------------------
        # Variables: x[0..N], w[0..N-1], v[0..N-1], λ[1..N], μ[0..N-1]
        dim_x = (N + 1) * n
        dim_w = N * p
        dim_v = N * m
        dim_lam = N * n
        dim_mu = N * m

        total_dim = dim_x + dim_w + dim_v + dim_lam + dim_mu
        mat = np.zeros((total_dim, total_dim), dtype=float)
        rhs = np.zeros(total_dim, dtype=float)

        # Mapping helper
        def idx(var_type, t, i):
            if var_type == "x":
                return t * n + i
            if var_type == "w":
                return dim_x + t * p + i
            if var_type == "v":
                return dim_x + dim_w + t * m + i
            if var_type == "lam":
                return dim_x + dim_w + dim_v + t * n + i
            if var_type == "mu":
                return dim_x + dim_w + dim_v + dim_lam + t * m + i
            raise ValueError

        # ----- Objective derivatives ----------------------------------
        #  ∂/∂w:   2w + B^T λ = 0
        for t in range(N):
            for i in range(p):
                row = idx("w", t, i)
                mat[row, row] = 2.0
                # add B^T λ term
                for j in range(n):
                    mat[row, idx("lam", t + 1, j)] = B[j, i]
        #  ∂/∂v:   2τ v + C^T μ = 0
        for t in range(N):
            for i in range(m):
                row = idx("v", t, i)
                mat[row, row] = 2 * tau
                for j in range(n):
                    mat[row, idx("mu", t, j)] = C[j, i]
        #  ∂/∂x:   λ - λ_prev - 2 C^T μ = 0  (for t>0)
        for t in range(1, N + 1):
            for i in range(n):
                row = idx("x", t, i)
                # λ_t term
                mat[row, idx("lam", t, i)] = 1.0
                # -λ_{t-1}
                mat[row, idx("lam", t - 1, i)] -= 1.0
                # -2 C^T μ_t
                for j in range(m):
                    mat[row, idx("mu", t - 1, j)] -= 2.0 * C[j, i]

        #  ∂/∂λ:    dynamics equality  x_{t+1} - A x_t - B w_t = 0
        for t in range(1, N + 1):
            for i in range(n):
                row = idx("lam", t, i)
                # x_{t+1} term
                mat[row, idx("x", t, i)] = 1.0
                # -A x_t term
                for j in range(n):
                    mat[row, idx("x", t - 1, j)] -= A[i, j]
                # -B w_t term
                for j in range(p):
                    mat[row, idx("w", t - 1, j)] -= B[i, j]
        #  ∂/∂μ:    measurement equality  y_t - C x_t - v_t = 0
        for t in range(N):
            for i in range(m):
                row = idx("mu", t, i)
                # -C x_t term
                for j in range(n):
                    mat[row, idx("x", t, j)] -= C[i, j]
                # -v_t term
                mat[row, idx("v", t, i)] -= 1.0
                # RHS y_t
                rhs[row] = y[t, i]

        # ----- Initial condition  x0 = known --------------------------------
        # enforced by setting the variable and fixing its value
        for i in range(n):
            row = idx("x", 0, i)
            mat[row, row] = 1.0
            rhs[row] = x0[i]

        # ----- Solve the linear system ---------------------------------
        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            # Numerically singular – fall back to the CVXPY solver
            return self._fallback(problem)

        # ----- Extract solution ---------------------------------------
        x_hat = sol[:dim_x].reshape((N + 1, n))
        w_hat = sol[dim_x:dim_x + dim_w].reshape((N, p))
        v_hat = sol[dim_x + dim_w:dim_x + dim_w + dim_v].reshape((N, m))

        return {
            "x_hat": x_hat.tolist(),
            "w_hat": w_hat.tolist(),
            "v_hat": v_hat.tolist(),
        }

    def _fallback(self, problem):
        """
        If the KKT matrix is singular we fall back to the reference
        CVXPY implementation.  This code path is expected to be rarely
        executed for the benchmark data.
        """
        import cvxpy as cp
        A = np.array(problem["A"])
        B = np.array(problem["B"])
        C = np.array(problem["C"])
        y = np.array(problem["y"])
        x0 = np.array(problem["x_initial"])
        tau = float(problem["tau"])

        N, m = y.shape
        n = A.shape[1]
        p = B.shape[1]

        x = cp.Variable((N + 1, n), name="x")
        w = cp.Variable((N, p), name="w")
        v = cp.Variable((N, m), name="v")

        obj = cp.Minimize(cp.sum_squares(w) + tau * cp.sum_squares(v))

        cons = [x[0] == x0]
        for t in range(N):
            cons.append(x[t + 1] == A @ x[t] + B @ w[t])
            cons.append(y[t] == C @ x[t] + v[t])

        prob = cp.Problem(obj, cons)
        prob.solve()

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
            return {"x_hat": [], "w_hat": [], "v_hat": []}

        return {
            "x_hat": x.value.tolist(),
            "w_hat": w.value.tolist(),
            "v_hat": v.value.tolist(),
        }
