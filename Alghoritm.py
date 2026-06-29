import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Alghoritm:
    def __init__(self, Q, p, x0, fun, x_star):
        self.Q = Q
        self.p = p
        self.x0 = x0
        self.fun = fun
        raw = x_star if isinstance(x_star, list) else [x_star]
        self.x_stars = [np.array([float(xi) for xi in x], dtype=float) for x in raw]
        self.f_star = float(min(self.f(x) for x in self.x_stars))

    def f(self, x):
        match(self.fun):
            case "quadratic":
                return 1/2 * x.T @ self.Q @ x - self.p.T @ x
            case "non-linear":
                return (x[0]**4 - x[0]**2) + (x[1]**4 - x[1]**2)

    def grad_f(self, x):
        match(self.fun):
            case "quadratic":
                return self.Q @ x - self.p
            case "non-linear":
                return np.array([4*x[0]**3 - 2*x[0], 4*x[1]**3 - 2*x[1]])

    def heavy_ball_continuous(self, alpha=3.0, T=20.0, n_points=1000, v0=None):
        x0 = np.asarray(self.x0, dtype=float)
        n = len(x0)

        if v0 is None:
            v0 = np.zeros_like(x0)
        else:
            v0 = np.asarray(v0, dtype=float)

        y0 = np.concatenate([x0, v0])

        def system(t, y):
            x = y[:n]
            v = y[n:]

            return np.concatenate([v, -alpha * v - self.grad_f(x)])

        t_eval = np.linspace(0, T, n_points)

        sol = solve_ivp(
            system,
            t_span=(0, T),
            y0=y0,
            t_eval=t_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05
        )

        xs = sol.y[:n].T
        vs = sol.y[n:].T

        return sol.t, xs, vs, sol

    def heavy_ball_discrete(self, alpha=0.01, beta=0.8, max_iter=1000, tol=1e-8):
        x = np.asarray(self.x0, dtype=float)
        v = np.zeros_like(x)

        xs = [x.copy()]
        vs = [v.copy()]

        for k in range(max_iter):
            grad = self.grad_f(x)

            if np.abs(self.f(x) - self.f_star) < tol:
                break

            if alpha is None and self.fun == 'non-linear':
                step = self.armijo(x, grad)
            elif alpha is None:
                step = np.linalg.norm(grad) ** 2 / (grad @ self.Q @ grad)
            else:
                step = alpha

            v = beta * v - step * grad
            x = x + v

            xs.append(x.copy())

            xs.append(x.copy())
            vs.append(v.copy())

        return np.array(xs), np.array(vs), k

    def nesterov_continuous(self, alpha, T=1000.0, n_points=1000):
        t_span = (0.01, T)

        v0 = np.zeros_like(self.x0)
        y0 = np.concatenate((self.x0, v0))
        n = len(self.x0)

        def system(t, y):
            x = y[:n]
            v = y[n:]

            dx = v
            dv = -(alpha / t) * v - self.grad_f(x)

            return np.concatenate((dx, dv))

        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        sol = solve_ivp(
            system,
            t_span,
            y0,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=0.05
        )

        xs = sol.y[:n].T
        vs = sol.y[n:].T

        return sol.t, xs, vs, sol

    def nesterov_discrete(self, alpha=0.01, max_iter=500, mode='wypukly', kappa=None,
                          restart='none', restart_interval=50, tol=1e-8):
        x = self.x0.copy()
        y = self.x0.copy()
        t = 1.0

        xs = [x.copy()]
        ys = [y.copy()]
        fs = [self.f(x)]
        restart_iters = []

        for k in range(max_iter):
            grad = self.grad_f(x)
            x_old = x.copy()
            f_old = fs[-1]

            if alpha is None and self.fun == 'non-linear':
                step = self.armijo(x, grad)
            elif alpha is None:
                step = np.linalg.norm(grad) ** 2 / (grad @ self.Q @ grad)
            else:
                step = alpha

            x_new = y - step * self.grad_f(y)
            f_new = self.f(x_new)

            if f_new - self.f_star < tol:
                break

            if mode == 'silnie_wypukly':
                beta = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)

            else:
                do_restart = (
                        (restart == 'adaptive' and f_new > f_old) or
                        (restart == 'fixed' and (k + 1) % restart_interval == 0)
                )

                if do_restart:
                    x = x_old.copy()
                    y = x_old.copy()
                    t = 1.0
                    restart_iters.append(k)
                    xs.append(x.copy())
                    ys.append(y.copy())
                    fs.append(f_old)
                    continue

                t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
                beta = (t - 1.0) / t_new
                t = t_new

            y = x_new + beta * (x_new - x_old)
            x = x_new

            xs.append(x.copy())
            ys.append(y.copy())
            fs.append(f_new)

        return np.array(xs), np.array(ys), np.array(fs), restart_iters, k


    def armijo(self, x, grad, alpha0=1.0, beta=0.5, c=1e-4):
        alpha = alpha0
        f0 = self.f(x)
        slope = - np.dot(grad, grad)

        while self.f(x - alpha * grad) > f0 + c * alpha * slope:
            alpha *= beta

        return alpha

    def gradient_descent(self, alpha=None, tol=1e-8, max_iter=500):
        x = np.asarray(self.x0, dtype=float)

        xs = [x.copy()]

        for k in range(max_iter):
            grad = self.grad_f(x)
            if self.f(x) - self.f_star < tol:
                break

            if alpha is None:
                if self.fun == 'quadratic':
                    step = np.linalg.norm(grad)**2 / (grad.T @ self.Q @ grad)
                else:
                    step = self.armijo(x, grad)
            else:
                step = alpha

            x = x - step * grad
            xs.append(x.copy())

        return np.array(xs), k

    def plot_trajectory(self, trajectories, labels=None, title="Trajektoria algorytmu",
                        levels=40, ax=None):
        if isinstance(trajectories, dict):
            labels = list(trajectories.keys())
            traj_list = [np.asarray(v) for v in trajectories.values()]
        else:
            traj_list = [np.asarray(t) for t in trajectories]
            if labels is None:
                labels = [f"metoda {i + 1}" for i in range(len(traj_list))]
            if len(labels) != len(traj_list):
                raise ValueError("Liczba etykiet musi być równa liczbie trajektorii.")

        all_pts = np.vstack(traj_list)
        x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
        y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()

        x_stars_arr = np.asarray([np.asarray(s, dtype=float) for s in self.x_stars])
        for xs_pt in x_stars_arr:
            x_min = min(x_min, xs_pt[0])
            x_max = max(x_max, xs_pt[0])
            y_min = min(y_min, xs_pt[1])
            y_max = max(y_max, xs_pt[1])

        margin_x = 0.2 * (x_max - x_min + 1e-8)
        margin_y = 0.2 * (y_max - y_min + 1e-8)
        x_grid = np.linspace(x_min - margin_x, x_max + margin_x, 300)
        y_grid = np.linspace(y_min - margin_y, y_max + margin_y, 300)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.f(np.array([X[i, j], Y[i, j]]))

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.contour(X, Y, Z, levels=levels)

        markers = ["o", "x", ".", "^", "s", "D", "v", "P"]
        for i, (xs, label) in enumerate(zip(traj_list, labels)):
            marker = markers[i % len(markers)]
            line, = ax.plot(xs[:, 0], xs[:, 1], marker=marker, markersize=3, label=label)
            ax.scatter(xs[-1, 0], xs[-1, 1], s=80, color=line.get_color(),
                       zorder=5, label=f"koniec: {label}")

        ax.scatter(traj_list[0][0, 0], traj_list[0][0, 1], s=80, color="white",
                   zorder=6, label="start")

        for i, xs_pt in enumerate(x_stars_arr):
            ax.scatter(xs_pt[0], xs_pt[1], s=100, marker="*", color="purple",
                       label="minimum dokładne" if i == 0 else None)

        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True)

        if standalone:
            plt.show()

    def plot_function_gap(self, trajectories, labels=None, title="Wykres zbieżności",
                          log_scale=True, eps=1e-16, split_plots=True, smooth_window=51):
        from scipy.signal import savgol_filter

        f_star = self.f_star

        if isinstance(trajectories, dict):
            items = list(trajectories.items())
        else:
            if labels is None:
                labels = [f"metoda {i + 1}" for i in range(len(trajectories))]
            if len(labels) != len(trajectories):
                raise ValueError("Liczba etykiet musi być taka sama jak liczba trajektorii.")
            items = list(zip(labels, trajectories))

        gaps_by_label = {}
        for label, xs in items:
            xs = np.asarray(xs, dtype=float)
            if xs.ndim != 2:
                raise ValueError("Każda trajektoria musi mieć shape (liczba_iteracji, wymiar).")
            values = np.array([self.f(x) for x in xs], dtype=float)
            gaps_by_label[label] = values - f_star

        if not split_plots:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes_map = {label: ax for label in gaps_by_label}
        else:
            discrete_labels = [l for l in gaps_by_label if "continuous" not in l]
            continuous_labels = [l for l in gaps_by_label if "continuous" in l]
            has_both = len(discrete_labels) > 0 and len(continuous_labels) > 0

            if has_both:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                axes_map = {l: ax1 for l in discrete_labels}
                axes_map.update({l: ax2 for l in continuous_labels})
                ax1.set_title("Metody dyskretne")
                ax2.set_title("Metody ciągłe")
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                axes_map = {label: ax for label in gaps_by_label}

        for label, gaps in gaps_by_label.items():
            ax = axes_map[label]
            gaps_to_plot = np.maximum(gaps, eps) if log_scale else gaps
            k = np.arange(len(gaps))
            is_continuous = "continuous" in label

            if is_continuous and split_plots and len(gaps) > smooth_window:
                line, = ax.plot(k, gaps_to_plot, alpha=0.2, markersize=0)
                color = line.get_color()
                log_gaps = np.log10(gaps_to_plot)
                smoothed = savgol_filter(log_gaps, window_length=smooth_window, polyorder=3)
                ax.plot(k, 10 ** smoothed, color=color, linewidth=2, label=label)
            else:
                ax.plot(k, gaps_to_plot, marker="o", markersize=3, label=label)

        seen_axes = set()
        for label, ax in axes_map.items():
            if id(ax) in seen_axes:
                continue
            seen_axes.add(id(ax))
            if log_scale:
                ax.set_yscale("log")
                ax.set_ylabel("$f(x_k) - f^*$, skala log")
            else:
                ax.set_ylabel("$f(x_k) - f^*$")
            is_cont_ax = any("continuous" in l for l, a in axes_map.items() if a is ax)
            ax.set_xlabel("$k$ / " + ("indeks punktu ODE" if is_cont_ax else "iteracja"))
            ax.legend()
            ax.grid(True)

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def plot_alpha_sensitivity(self, alphas, T=20.0, n_points=1000, log_scale=True,
                               eps=1e-16, show_trajectories=True, mode='discrete'):
        f_star = self.f_star
        x_star = self.x_stars

        results = {}
        for alpha in alphas:
            if mode == 'continuous':
                t_arr, xs, vs, sol = self.heavy_ball_continuous(alpha=alpha, T=T, n_points=n_points)
                results[alpha] = (t_arr, xs)
            else:
                xs, vs, filler = self.heavy_ball_discrete(alpha=alpha)
                t_arr = np.arange(len(xs))
                results[alpha] = (t_arr, xs)

        fig, ax = plt.subplots(figsize=(9, 5))
        for alpha, (t_arr, xs) in results.items():
            values = np.array([self.f(x) for x in xs], dtype=float)
            gaps = np.maximum(values - f_star, eps)
            ax.plot(t_arr, gaps, label=f"α = {alpha}")

        if log_scale:
            ax.set_yscale("log")
            ax.set_ylabel("$f(x(t)) - f^*$, skala log")
        else:
            ax.set_ylabel("$f(x(t)) - f^*$")


        if mode == 'discrete':
            ax.set_xlabel("$k$")
            ax.set_title("Wpływ parametru $\\alpha$ na zbieżność Heavy Ball (dyskretna)")
        elif mode == 'continuous':
            ax.set_xlabel("$t$")
            ax.set_title("Wpływ parametru $\\alpha$ na zbieżność Heavy Ball (ciągła)")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

        if show_trajectories and np.asarray(self.x0).shape[0] == 2:
            fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=True)
            if len(alphas) == 1:
                axes = [axes]

            all_pts = []
            for _, (_, xs) in results.items():
                xs_finite = xs[np.all(np.isfinite(xs), axis=1)]
                if len(xs_finite) > 0:
                    all_pts.append(xs_finite)
            all_pts = np.vstack(all_pts)
            x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
            y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
            mx = 0.2 * (x_max - x_min + 1e-8)
            my = 0.2 * (y_max - y_min + 1e-8)
            xg = np.linspace(x_min - mx, x_max + mx, 250)
            yg = np.linspace(y_min - my, y_max + my, 250)
            X, Y = np.meshgrid(xg, yg)
            Z = np.array([[self.f(np.array([X[i, j], Y[i, j]])) for j in range(X.shape[1])]
                          for i in range(X.shape[0])])

            for ax, alpha in zip(axes, alphas):
                _, xs = results[alpha]
                ax.contour(X, Y, Z, levels=30)
                ax.plot(xs[:, 0], xs[:, 1], lw=1.2, label=f"α = {alpha}")
                ax.scatter(*xs[0], s=60, zorder=5, label="start")
                ax.scatter(*xs[-1], s=60, marker="x", zorder=5, label="koniec")
                ax.set_title(f"α = {alpha}")
                ax.set_xlabel("$x_1$")
                ax.grid(True)

            axes[0].set_ylabel("$x_2$")
            plt.suptitle("Trajektorie Heavy Ball dla różnych $\\alpha$", y=1.02)
            plt.tight_layout()
            plt.show()

    def plot_restart_comparison(self, alpha=0.01, max_iter=500, restart_interval=50, log_scale=True, eps=1e-16):
        f_star = self.f_star

        strategies = [
            ('none', 'Nesterov bez restartu', 'blue'),
            ('adaptive', 'Nesterov restart adaptacyjny', 'green'),
            ('fixed', f'Nesterov restart co {restart_interval} iter.', 'orange'),
        ]

        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, label, color in strategies:
            xs, ys, values, restart_iters, filler = self.nesterov_discrete(
                alpha=alpha,
                max_iter=max_iter,
                restart=strategy,
                restart_interval=restart_interval
            )

            gaps = np.maximum(values - f_star, eps)
            k = np.arange(len(gaps))

            ax.plot(k, gaps, label=label, color=color, lw=1.8)

            for r in restart_iters:
                ax.axvline(x=r, color=color, alpha=0.3, linestyle='--', lw=0.9)

        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('$f(x_k) - f^*$, skala log')
        else:
            ax.set_ylabel('$f(x_k) - f^*$')

        ax.set_xlabel('iteracja $k$')
        ax.set_title('Wpływ restartu na zbieżność metody Nesterova')
        ax.legend()
        ax.grid(True, which='both', alpha=0.4)
        plt.tight_layout()
        plt.show()