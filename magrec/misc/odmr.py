# ODMR line-shape tools for NV contrast spectra (fluorescence dips vs. microwave frequency).
#
# Workflow matches what is common in NV lab code and frameworks like QUDI: smooth the
# noisy photon contrast with Savitzky–Golay (preserves line shape better than a plain box
# average), then fit a small polynomial baseline plus a sum of absorptive Lorentzian dips.

from __future__ import annotations

import asyncio
import os
import sys
import time
import warnings
from concurrent.futures import as_completed
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks, savgol_filter
class _ProgressTracker:
    """Lightweight progress display for console and Jupyter."""

    def __init__(self, total: int, enabled: bool, desc: str = "ODMR fit") -> None:
        self.total = max(int(total), 1)
        self.enabled = bool(enabled)
        self.desc = desc
        self.n = 0
        self.t0 = time.perf_counter()
        self._last_render_t = 0.0
        self._is_notebook = "ipykernel" in sys.modules
        self._display_handle = None
        if self.enabled and self._is_notebook:
            try:
                from IPython.display import display

                self._display_handle = display("", display_id=True)
            except Exception:
                self._display_handle = None

    def _status(self) -> str:
        elapsed = max(time.perf_counter() - self.t0, 1e-12)
        avg = elapsed / max(self.n, 1)
        rem = max(self.total - self.n, 0)
        eta = rem * avg
        return (
            f"{self.desc}: {self.n}/{self.total} traces "
            f"| avg {avg:.4f} s/trace | ETA {eta:.1f} s"
        )

    def _render(self, final: bool = False) -> None:
        if not self.enabled:
            return
        text = self._status()
        if self._display_handle is not None:
            try:
                self._display_handle.update(text)
                return
            except Exception:
                self._display_handle = None
        if final:
            print(f"\r{text}")
        else:
            print(f"\r{text}", end="", flush=True)

    def update(self, inc: int = 1) -> None:
        if not self.enabled:
            return
        self.n = min(self.n + int(inc), self.total)
        now = time.perf_counter()
        # Throttle render to avoid flooding output while keeping responsiveness.
        if (now - self._last_render_t) >= 0.2 or self.n >= self.total:
            self._render(final=self.n >= self.total)
            self._last_render_t = now

    def close(self) -> None:
        if not self.enabled:
            return
        self.n = min(self.n, self.total)
        self._render(final=True)



def _odd_window(n: int, max_frac: float = 0.15) -> int:
    n = int(n)
    w = min(n - (1 - n % 2), int(max_frac * n) | 1)
    w = max(w, 5)
    if w % 2 == 0:
        w -= 1
    return max(w, 5)


def _lorentzian_dip(x: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    w = max(float(width), 1e-12)
    return -float(amplitude) * (w**2) / ((x - float(center)) ** 2 + w**2)


def _two_dip_model(
    x: np.ndarray,
    b0: float,
    b1: float,
    c1: float,
    w1: float,
    a1: float,
    c2: float,
    w2: float,
    a2: float,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    base = b0 + b1 * x
    return base + _lorentzian_dip(x, c1, w1, a1) + _lorentzian_dip(x, c2, w2, a2)


def _guess_two_centers(y_smooth: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_smooth, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    inverted = float(np.nanmax(y)) - y
    d = max(n // 12, 8)
    noise = float(np.median(np.abs(np.diff(y)))) + 1e-12
    prom = max(3.0 * noise, 0.05 * float(np.max(inverted)))

    peaks, props = find_peaks(inverted, prominence=prom, distance=d)
    if len(peaks) >= 2:
        order = np.argsort(props["prominences"])[::-1]
        idx = sorted(peaks[order[:2]])
        return float(x[idx[0]]), float(x[idx[1]])

    if len(peaks) == 1:
        i0 = int(peaks[0])
        left = slice(0, max(i0 - d, 0))
        right = slice(min(i0 + d, n), n)
        c1 = float(x[i0])
        rest = list(range(left.start, left.stop)) + list(range(right.start, right.stop))
        if not rest:
            rest = [0, n - 1]
        i1 = rest[int(np.argmin(y[rest]))]
        c2 = float(x[i1])
        return (c1, c2) if c1 <= c2 else (c2, c1)

    half = n // 2
    i1 = int(np.argmin(y[:half]))
    i2 = half + int(np.argmin(y[half:]))
    c1, c2 = float(x[i1]), float(x[i2])
    return (c1, c2) if c1 <= c2 else (c2, c1)


def _smooth_trace(
    y: np.ndarray,
    savgol_window: Optional[int],
    savgol_polyorder: int,
) -> Tuple[np.ndarray, int, int]:
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if savgol_window is None:
        win = _odd_window(n)
    else:
        win = int(savgol_window)
        if win % 2 == 0:
            win += 1
        win = max(5, min(win, n - (1 - n % 2)))
    poly = min(int(savgol_polyorder), win - 1)
    if poly < 1:
        poly = 1

    pad = win // 2
    y_pad = np.pad(y, pad, mode="reflect")
    y_s_pad = savgol_filter(y_pad, win, poly, mode="interp")
    y_smooth = y_s_pad[pad : pad + n]
    return y_smooth, win, poly


def _sanitize_initial_p0(p0: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    p = np.asarray(p0, dtype=np.float64).reshape(8).copy()
    p[2] = float(np.clip(p[2], xmin, xmax))
    p[5] = float(np.clip(p[5], xmin, xmax))
    p[3] = max(p[3], 1e-9)
    p[6] = max(p[6], 1e-9)
    p[4] = max(p[4], 0.0)
    p[7] = max(p[7], 0.0)
    return p


@dataclass
class ODMRTwoDipFitResult:
    x: np.ndarray
    signal_raw: np.ndarray
    signal_smoothed: np.ndarray
    fit_curve: np.ndarray
    dip_centers: np.ndarray
    dip_centers_stderr: np.ndarray
    popt: np.ndarray
    pcov: np.ndarray
    savgol_window: int
    savgol_polyorder: int


@dataclass
class ODMRTwoDipStackFitResult:
    frequency_axis: np.ndarray
    dip_centers: np.ndarray          # (n_x, n_y, 2)
    dip_centers_stderr: np.ndarray   # (n_x, n_y, 2)
    dip_splitting: np.ndarray        # (n_x, n_y)
    popt: np.ndarray                 # (n_x, n_y, 8)
    success: np.ndarray              # (n_x, n_y)
    rmse: np.ndarray                 # (n_x, n_y)
    savgol_window: int
    savgol_polyorder: int


def _failed_fit_payload(
    x: np.ndarray,
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    win: int,
    poly: int,
    full_output: bool,
) -> Any:
    nans2 = np.full(2, np.nan)
    nan8 = np.full(8, np.nan)
    if full_output:
        return ODMRTwoDipFitResult(
            x=x,
            signal_raw=y_raw,
            signal_smoothed=y_smooth,
            fit_curve=np.full_like(y_smooth, np.nan),
            dip_centers=nans2.copy(),
            dip_centers_stderr=nans2.copy(),
            popt=nan8.copy(),
            pcov=np.full((8, 8), np.nan),
            savgol_window=win,
            savgol_polyorder=poly,
        )
    return (nan8, nans2.copy(), nans2.copy(), np.nan, win, poly)


def _two_dip_fit_impl(
    signal: np.ndarray,
    x: np.ndarray,
    *,
    savgol_window: Optional[int],
    savgol_polyorder: int,
    max_iter: int,
    initial_popt: Optional[np.ndarray],
    full_output: bool,
) -> Tuple[Any, bool]:
    y = np.asarray(signal, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()
    n = y.size
    if n < 8:
        raise ValueError("signal must have length at least 8 for two-dip ODMR fit")
    if x.shape[0] != n:
        raise ValueError("freqs must match signal length")

    y_smooth, win, poly = _smooth_trace(y, savgol_window, savgol_polyorder)

    ymin = float(np.min(y_smooth))
    ymax = float(np.max(y_smooth))
    b0_g = float(np.median(y_smooth))
    amp_g = max(ymax - ymin, 1e-6)
    span = float(np.ptp(x)) + 1e-12
    w_g = max(span / 40.0, 1e-6)
    xmin, xmax = float(np.min(x)), float(np.max(x))

    bounds_lo = [-np.inf, -np.inf, xmin, 1e-9, 0.0, xmin, 1e-9, 0.0]
    bounds_hi = [np.inf, np.inf, xmax, span, amp_g * 20.0, xmax, span, amp_g * 20.0]

    if initial_popt is not None and np.all(np.isfinite(initial_popt)):
        p0 = _sanitize_initial_p0(initial_popt, xmin, xmax)
    else:
        c1, c2 = _guess_two_centers(y_smooth, x)
        p0 = np.array([b0_g, 0.0, c1, w_g, amp_g * 0.5, c2, w_g, amp_g * 0.5], dtype=np.float64)

    def _try_fit(p0_try: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            return curve_fit(
                _two_dip_model,
                x,
                y_smooth,
                p0=p0_try,
                bounds=(bounds_lo, bounds_hi),
                maxfev=max_iter,
            )

    try:
        popt, pcov = _try_fit(p0)
    except (RuntimeError, ValueError):
        if initial_popt is not None:
            try:
                c1, c2 = _guess_two_centers(y_smooth, x)
                p0_fb = np.array([b0_g, 0.0, c1, w_g, amp_g * 0.5, c2, w_g, amp_g * 0.5], dtype=np.float64)
                popt, pcov = _try_fit(p0_fb)
            except (RuntimeError, ValueError):
                return _failed_fit_payload(x, y, y_smooth, win, poly, full_output), False
        else:
            return _failed_fit_payload(x, y, y_smooth, win, poly, full_output), False

    y_hat = _two_dip_model(x, *popt)
    c_a, c_b = float(popt[2]), float(popt[5])
    try:
        perr = np.sqrt(np.diag(pcov))
        s_a, s_b = float(perr[2]), float(perr[5])
    except ValueError:
        s_a, s_b = np.nan, np.nan

    if c_a <= c_b:
        centers = np.array([c_a, c_b], dtype=np.float64)
        stderr = np.array([s_a, s_b], dtype=np.float64)
    else:
        centers = np.array([c_b, c_a], dtype=np.float64)
        stderr = np.array([s_b, s_a], dtype=np.float64)

    if full_output:
        return (
            ODMRTwoDipFitResult(
                x=x,
                signal_raw=y,
                signal_smoothed=y_smooth,
                fit_curve=y_hat,
                dip_centers=centers,
                dip_centers_stderr=stderr,
                popt=popt,
                pcov=pcov,
                savgol_window=win,
                savgol_polyorder=poly,
            ),
            True,
        )

    rmse = float(np.sqrt(np.mean((y_smooth - y_hat) ** 2)))
    return (popt, centers, stderr, rmse, win, poly), True


def fit_odmr_two_lorentzian_dips(
    signal: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    *,
    savgol_window: Optional[int] = None,
    savgol_polyorder: int = 3,
    max_iter: int = 2000,
    initial_popt: Optional[np.ndarray] = None,
) -> ODMRTwoDipFitResult:
    y = np.asarray(signal, dtype=np.float64).ravel()
    if freqs is None:
        x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.asarray(freqs, dtype=np.float64).ravel()
    res, ok = _two_dip_fit_impl(
        y,
        x,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        max_iter=max_iter,
        initial_popt=initial_popt,
        full_output=True,
    )
    if not ok:
        raise RuntimeError("ODMR two-dip fit did not converge")
    return res


def _fit_odmr_row_worker(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    x, row_data, row_j, savgol_window, savgol_polyorder, max_iter, first_col_seed = args
    x = np.asarray(x, dtype=np.float64)
    row_data = np.asarray(row_data, dtype=np.float64)
    nx = row_data.shape[1]

    popt_out = np.full((nx, 8), np.nan)
    centers_out = np.full((nx, 2), np.nan)
    stderr_out = np.full((nx, 2), np.nan)
    success_out = np.zeros(nx, dtype=bool)
    rmse_out = np.full(nx, np.nan)
    win_g, poly_g = 5, 1

    seed = None if first_col_seed is None else np.asarray(first_col_seed, dtype=np.float64)
    for i in range(nx):
        initial = seed if i == 0 else popt_out[i - 1]
        if initial is not None and not np.all(np.isfinite(initial)):
            initial = None

        out, ok = _two_dip_fit_impl(
            row_data[:, i],
            x,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            max_iter=max_iter,
            initial_popt=initial,
            full_output=False,
        )
        if ok:
            popt, centers, stderr, rmse, win_g, poly_g = out
            popt_out[i] = popt
            centers_out[i] = centers
            stderr_out[i] = stderr
            rmse_out[i] = rmse
            success_out[i] = True

    return row_j, popt_out, centers_out, stderr_out, success_out, rmse_out, win_g, poly_g


def fit_odmr_two_dip_stack(
    data: np.ndarray,
    freqs: np.ndarray,
    *,
    savgol_window: Optional[int] = None,
    savgol_polyorder: int = 3,
    max_iter: int = 2000,
    warm_start_order: Literal["row_parallel", "raster"] = "row_parallel",
    max_workers: Optional[int] = None,
    executor: Optional[Executor] = None,
    vertical_seed: bool = True,
    show_progress: bool = True,
    progress_desc: str = "ODMR fit",
) -> ODMRTwoDipStackFitResult:
    data = np.asarray(data, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    if data.ndim != 3:
        raise ValueError("data must have shape (n_freq, n_x, n_y)")

    nf, nx, ny = data.shape
    if freqs.shape[0] != nf:
        raise ValueError("len(freqs) must match data.shape[0]")
    progress = _ProgressTracker(total=nx * ny, enabled=show_progress, desc=progress_desc)

    _, win_ref, poly_ref = _smooth_trace(data[:, 0, 0], savgol_window, savgol_polyorder)

    popt_arr = np.full((nx, ny, 8), np.nan)
    centers_arr = np.full((nx, ny, 2), np.nan)
    stderr_arr = np.full((nx, ny, 2), np.nan)
    success_arr = np.zeros((nx, ny), dtype=bool)
    rmse_arr = np.full((nx, ny), np.nan)

    if warm_start_order == "raster":
        for j in range(ny):
            for i in range(nx):
                if i > 0:
                    seed = popt_arr[i - 1, j]
                elif j > 0:
                    seed = popt_arr[nx - 1, j - 1]
                else:
                    seed = None
                if seed is not None and not np.all(np.isfinite(seed)):
                    seed = None

                out, ok = _two_dip_fit_impl(
                    data[:, i, j],
                    freqs,
                    savgol_window=savgol_window,
                    savgol_polyorder=savgol_polyorder,
                    max_iter=max_iter,
                    initial_popt=seed,
                    full_output=False,
                )
                if ok:
                    p, c, s, r, win_ref, poly_ref = out
                    popt_arr[i, j] = p
                    centers_arr[i, j] = c
                    stderr_arr[i, j] = s
                    success_arr[i, j] = True
                    rmse_arr[i, j] = r
                progress.update(1)

    elif warm_start_order == "row_parallel":
        if vertical_seed:
            col0_seed = None
            for j in range(ny):
                args = (
                    freqs,
                    np.ascontiguousarray(data[:, :, j]),
                    j,
                    savgol_window,
                    savgol_polyorder,
                    max_iter,
                    col0_seed,
                )
                _, po, ce, se, su, rm, win_ref, poly_ref = _fit_odmr_row_worker(args)
                popt_arr[:, j, :] = po
                centers_arr[:, j, :] = ce
                stderr_arr[:, j, :] = se
                success_arr[:, j] = su
                rmse_arr[:, j] = rm
                if su[-1] and np.all(np.isfinite(po[-1])):
                    col0_seed = po[-1].copy()
                else:
                    col0_seed = None
                progress.update(nx)
        else:
            tasks = [
                (
                    freqs,
                    np.ascontiguousarray(data[:, :, j]),
                    j,
                    savgol_window,
                    savgol_polyorder,
                    max_iter,
                    None,
                )
                for j in range(ny)
            ]
            if executor is not None:
                futures = [executor.submit(_fit_odmr_row_worker, t) for t in tasks]
                for fut in as_completed(futures):
                    row_j, po, ce, se, su, rm, win_ref, poly_ref = fut.result()
                    popt_arr[:, row_j, :] = po
                    centers_arr[:, row_j, :] = ce
                    stderr_arr[:, row_j, :] = se
                    success_arr[:, row_j] = su
                    rmse_arr[:, row_j] = rm
                    progress.update(nx)
            else:
                workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_fit_odmr_row_worker, t) for t in tasks]
                    for fut in as_completed(futures):
                        row_j, po, ce, se, su, rm, win_ref, poly_ref = fut.result()
                        popt_arr[:, row_j, :] = po
                        centers_arr[:, row_j, :] = ce
                        stderr_arr[:, row_j, :] = se
                        success_arr[:, row_j] = su
                        rmse_arr[:, row_j] = rm
                        progress.update(nx)
    else:
        raise ValueError("warm_start_order must be 'row_parallel' or 'raster'")
    progress.close()

    dip_splitting = centers_arr[:, :, 1] - centers_arr[:, :, 0]
    return ODMRTwoDipStackFitResult(
        frequency_axis=freqs.copy(),
        dip_centers=centers_arr,
        dip_centers_stderr=stderr_arr,
        dip_splitting=dip_splitting,
        popt=popt_arr,
        success=success_arr,
        rmse=rmse_arr,
        savgol_window=win_ref,
        savgol_polyorder=poly_ref,
    )


class ODMR:
    """Uniform frequency axis (Hz) plus two-dip ODMR fitting for single traces and stacks."""

    GAMMA_NV_ELECTRON_HZ_PER_T: float = 28.0249492e9

    def __init__(
        self,
        *,
        freq_start_hz: float = 0.0,
        df_per_index_hz: float = 1.0,
        gamma_hz_per_tesla: Optional[float] = None,
    ) -> None:
        self.freq_start_hz = float(freq_start_hz)
        self.df_per_index_hz = float(df_per_index_hz)
        self.gamma_hz_per_tesla = (
            float(gamma_hz_per_tesla)
            if gamma_hz_per_tesla is not None
            else float(self.GAMMA_NV_ELECTRON_HZ_PER_T)
        )

    def frequency_axis(self, n: int) -> np.ndarray:
        n = int(n)
        if n < 0:
            raise ValueError("n must be non-negative")
        return self.freq_start_hz + np.arange(n, dtype=np.float64) * self.df_per_index_hz

    def index_from_frequency_hz(self, f_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return (np.asarray(f_hz, dtype=np.float64) - self.freq_start_hz) / self.df_per_index_hz

    def zeeman_branch_separation_hz(self, B_parallel_tesla: float) -> float:
        return 2.0 * self.gamma_hz_per_tesla * float(B_parallel_tesla)

    def B_parallel_from_branch_separation(self, delta_f_hz: float) -> float:
        g = self.gamma_hz_per_tesla
        if g == 0:
            raise ValueError("gamma_hz_per_tesla must be non-zero")
        return float(delta_f_hz) / (2.0 * g)

    def spectral_bins_for_branch_separation(self, B_parallel_tesla: float) -> float:
        if self.df_per_index_hz == 0:
            raise ValueError("df_per_index_hz must be non-zero")
        return self.zeeman_branch_separation_hz(B_parallel_tesla) / self.df_per_index_hz

    def fit_two_dips(
        self,
        signal: np.ndarray,
        *,
        savgol_window: Optional[int] = None,
        savgol_polyorder: int = 3,
        max_iter: int = 2000,
        initial_popt: Optional[np.ndarray] = None,
    ) -> ODMRTwoDipFitResult:
        y = np.asarray(signal, dtype=np.float64).ravel()
        return fit_odmr_two_lorentzian_dips(
            y,
            self.frequency_axis(y.size),
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            max_iter=max_iter,
            initial_popt=initial_popt,
        )

    def fit_two_dips_stack(
        self,
        data: np.ndarray,
        *,
        savgol_window: Optional[int] = None,
        savgol_polyorder: int = 3,
        max_iter: int = 2000,
        warm_start_order: Literal["row_parallel", "raster"] = "row_parallel",
        max_workers: Optional[int] = None,
        executor: Optional[Executor] = None,
        vertical_seed: bool = True,
        show_progress: bool = True,
        progress_desc: str = "ODMR fit",
    ) -> ODMRTwoDipStackFitResult:
        data = np.asarray(data, dtype=np.float64)
        freqs = self.frequency_axis(data.shape[0])
        return fit_odmr_two_dip_stack(
            data,
            freqs,
            savgol_window=savgol_window,
            savgol_polyorder=savgol_polyorder,
            max_iter=max_iter,
            warm_start_order=warm_start_order,
            max_workers=max_workers,
            executor=executor,
            vertical_seed=vertical_seed,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    async def afit_two_dips_stack(
        self,
        data: np.ndarray,
        *,
        executor: Optional[Executor] = None,
        **kwargs: Any,
    ) -> ODMRTwoDipStackFitResult:
        """
        Async wrapper around `fit_two_dips_stack`.

        Supports the same progress arguments (`show_progress`, `progress_desc`).
        """
        loop = asyncio.get_running_loop()

        def _run() -> ODMRTwoDipStackFitResult:
            return self.fit_two_dips_stack(data, executor=executor, **kwargs)

        return await loop.run_in_executor(None, _run)

    def plot(
        self,
        result: ODMRTwoDipFitResult,
        *,
        ax: Optional[Any] = None,
        show_raw: bool = True,
        show_smoothed: bool = True,
        show_fit: bool = True,
        show_dip_markers: bool = True,
        show_residuals: bool = False,
        frequency_unit: str = "Hz",
        xlabel: Optional[str] = None,
        ylabel: str = "contrast",
        title: Optional[str] = None,
        legend: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        raw_kwargs: Optional[Dict[str, Any]] = None,
        smoothed_kwargs: Optional[Dict[str, Any]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        dip_kwargs: Optional[Dict[str, Any]] = None,
        residual_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Any]:
        import matplotlib.pyplot as plt

        unit_div: Dict[str, float] = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
        if frequency_unit not in unit_div:
            raise ValueError(f"frequency_unit must be one of {list(unit_div)}, got {frequency_unit!r}")
        div = unit_div[frequency_unit]
        x = np.asarray(result.x, dtype=np.float64) / div

        raw_kw: Dict[str, Any] = {"alpha": 0.35, "lw": 0.9, "label": "raw"}
        raw_kw.update(raw_kwargs or {})
        sm_kw: Dict[str, Any] = {"lw": 1.2, "label": f"SG smoothed (w={result.savgol_window})"}
        sm_kw.update(smoothed_kwargs or {})
        ft_kw: Dict[str, Any] = {"lw": 1.5, "label": "two Lorentzian dips + linear baseline"}
        ft_kw.update(fit_kwargs or {})
        dp_kw: Dict[str, Any] = {"color": "C3", "zorder": 5, "s": 36, "label": "dip centers"}
        dp_kw.update(dip_kwargs or {})
        res_kw: Dict[str, Any] = {"color": "0.35", "lw": 0.9, "label": "residual (smooth − fit)"}
        res_kw.update(residual_kwargs or {})

        xlab = xlabel if xlabel is not None else f"frequency ({frequency_unit})"

        if show_residuals and ax is not None:
            raise ValueError("show_residuals=True requires ax=None (or plot residuals yourself).")

        if ax is None:
            if figsize is None:
                figsize = (8.0, 5.5) if show_residuals else (8.0, 4.0)
            if show_residuals:
                fig, (ax_main, ax_res) = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    figsize=figsize,
                    gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
                )
            else:
                fig, ax_main = plt.subplots(figsize=figsize)
                ax_res = None
        else:
            fig = ax.figure
            ax_main = ax
            ax_res = None

        if show_raw:
            ax_main.plot(x, result.signal_raw, **raw_kw)
        if show_smoothed:
            ax_main.plot(x, result.signal_smoothed, **sm_kw)
        if show_fit:
            ax_main.plot(x, result.fit_curve, **ft_kw)
        if show_dip_markers:
            y_at = np.interp(result.dip_centers, result.x, result.fit_curve)
            ax_main.scatter(result.dip_centers / div, y_at, **dp_kw)

        ax_main.set_ylabel(ylabel)
        if title is not None:
            ax_main.set_title(title)
        if legend:
            ax_main.legend(loc="best", fontsize=8)

        if show_residuals and ax_res is not None:
            resid = result.signal_smoothed - result.fit_curve
            ax_res.axhline(0.0, color="0.75", lw=0.8, ls="--")
            ax_res.plot(x, resid, **res_kw)
            ax_res.set_ylabel("Δ")
            ax_res.set_xlabel(xlab)
        else:
            ax_main.set_xlabel(xlab)

        fig.tight_layout()
        out_axes: Any = (ax_main, ax_res) if ax_res is not None else ax_main
        return fig, out_axes
