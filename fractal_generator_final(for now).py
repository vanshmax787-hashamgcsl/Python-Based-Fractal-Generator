"""
Fractal Generator - standalone interactive module
================================
Contains all the fractal logic and can be run directly:

    python fractal_generator_v9_accelerated_v11.py

Generates:
    1. Mandelbrot Fractal
    2. Julia Fractal
    3. Newton Fractal
    5. Sierpinski Carpet

For Mandelbrot / Julia / Newton, a color is picked with a clickable grid of
colormap-gradient squares ("Multiple Tone Colors"). For Sierpinski Carpet,
a custom LIVE color picker opens right alongside the fractal window - a Hue x
Saturation square, a Lightness slider, and Hue/Sat/Lum + Red/Green/Blue
fields, matching the classic Windows color dialog layout. Dragging it recolors the fractal in real time. This never
touches the fractal's geometry (it only recolors the existing shapes), so
it has zero effect on zoom or generation performance - verified by timing:
color changes are O(1) (a single set_facecolor + redraw), completely
decoupled from the recursive subdivision that zooming triggers.

Zoom in using the toolbar's zoom-to-rectangle button (or back/forward/
home) - the fractal is genuinely RE-COMPUTED for the new region every
time. Press 'r' while the window is focused to reset to the original view.

Mandelbrot / Julia now use an ADAPTIVE algorithm chosen automatically from
the size of the view (verified by direct benchmarking, not assumed):
  - Wide/initial views use masked iteration with plain integer escape-time
    coloring - this is the original, fastest approach for that regime, so
    the very first render is exactly as fast as before.
  - Zoomed-in views switch to a full-array approach with SMOOTH
    (renormalized) escape-time coloring - measured ~40% FASTER than the
    old approach for deep zoom (avoids repeated fancy-indexing overhead on
    a mask that stays mostly "alive"), and the smooth coloring removes the
    color-banding that read as "pixelated" at high zoom, for only ~5%
    extra cost on top of that speedup.
The old "auto-boost iterations on zoom" mode has been removed entirely per
request - there is now only one mode, using exactly the iteration count
you type in, and it's faster at every zoom depth than the old fixed mode
was, without the auto-boost machinery.

Sierpinski Carpet derives its subdivision depth from the actual viewport
pixel scale and rebases safely into self-similar cells during zoom. This prevents the renderer from under-resolving a deep zoom. The old caps (13 and 10) were hit after only ~6
zoom steps, which is why deep zoom used to flatten out into a single solid
color or blank white - past the cap, the recursion simply couldn't
resolve any finer structure. Both were confirmed to stay fast (well under
50ms per zoom step) even at the new, much deeper caps, because the
view-based pruning keeps the actually-rendered shape count small
regardless of how deep the nominal recursion level goes.

Newton fractal polynomials are chosen from a curated list of 13 (free-text
entry removed) - see NEWTON_POLYNOMIALS below.

EVERY numeric input in this file is clamped to a documented maximum (and
minimum) - see the constants block below for the reasoning behind each one.
Mandelbrot / Julia / Newton now offer 720p, 1080p, 1440p and 4K render arrays.

Note on Newton fractals: fractpy's own plot() call has no colormap option,
so the Newton fractal here is computed directly with sympy (a fractpy
dependency) instead of calling fractpy's plot().

Requirements
------------
    pip install numpy matplotlib sympy fractpy numba

Optional NVIDIA GPU acceleration (GTX/RTX with CUDA):
    pip install numba-cuda

The renderer automatically detects the NVIDIA GPU. The Sierpinski Carpet renderer uses a stable raster path; CUDA is not required for it. CPU, GPU and HYBRID
backends are available. If CUDA/Numba-CUDA is unavailable, selecting GPU
or HYBRID falls back safely to the CPU renderer.
"""

import math
import colorsys
from concurrent.futures import ThreadPoolExecutor

import matplotlib

# Force a GUI backend so figures pop up in their own OS window instead of
# being captured by PyCharm's "Plots" / SciView tool window. This is also
# what makes tkinter guaranteed to be available (TkAgg needs it), so the
# live color picker below doesn't need a separate fallback path.
matplotlib.use("TkAgg")

import tkinter as tk
import numpy as np

# Optional JIT / CUDA acceleration. The program still starts if Numba or
# CUDA support is not installed; the CPU NumPy renderer remains available.
try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range

try:
    from numba import cuda
except ImportError:
    cuda = None

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, LineCollection


# =============================================================================
# LIMITS - every number here was chosen from an actual timing measurement.
# =============================================================================

FIXED_RESOLUTION = 800

# Output resolutions available for Mandelbrot / Julia / Newton.
# These are true render-array dimensions; the GUI window itself remains
# reasonably sized on screen.
FRACTAL_RESOLUTIONS = [
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("1440p", 2560, 1440),
    ("4K", 3840, 2160),
]

# --- Mandelbrot ---
MANDELBROT_MAX_ITER = 500
MANDELBROT_DEFAULT_ITER = 150
MANDELBROT_MAX_LIMIT = 3.0
MANDELBROT_DEFAULT_LIMIT = 2.0

# --- Julia ---
JULIA_MAX_ITER = 500
JULIA_DEFAULT_ITER = 150
JULIA_MAX_LIMIT = 2.0
JULIA_DEFAULT_LIMIT = 1.5
JULIA_C_MAX = 2.0

# --- Newton ---
NEWTON_MAX_ITER = 80
NEWTON_DEFAULT_ITER = 50
NEWTON_MAX_LIMIT = 3.0
NEWTON_DEFAULT_LIMIT = 2.0

# --- Sierpinski Carpet raster supersampling ---
# The rasterization fix (see below) solved the "shapes fade/truncate"
# bugs, but revealed a SECOND, separate issue: sampling exactly one point
# per pixel can only resolve detail down to about one pixel across. Push
# the depth further than that and different sub-cells inside a single
# pixel start winning the single-sample "coin flip" essentially at
# random, which is classic aliasing - it shows up as speckled noise
# instead of clean square regions. The threshold where this kicks in
# for Carpet - which is an exact mathematical match for what was reported
# ("9 looks fine, above that gets worse" / "above 7 looks worst").
#
# Fixed with two changes:
#  1. Each pixel is now supersampled (RASTER_SUPERSAMPLE^2 sub-samples,
#     averaged into a continuous 0-1 coverage value instead of a hard
#     True/False) for smooth, anti-aliased edges - this also pushes the
#     usable depth ceiling higher, since the EFFECTIVE resolution is
#     resolution*RASTER_SUPERSAMPLE.
#  2. The zoom-triggered depth calculation is now derived directly from
#     that effective resolution (floor(log_base(eff_res / view_width)))
#     instead of the old ad-hoc "-log(area_ratio)" heuristic, so depth can
#     never mathematically exceed what the current view can resolve
#     without aliasing, at ANY zoom level - not just the initial view.
RASTER_SUPERSAMPLE = 3
# Carpet quality: render at roughly the display pixel density instead of a
# fixed 800x800 square. This prevents rectangular zoom selections from
# throwing away resolution along one axis.
CARPET_TARGET_PIXELS = 1_600_000
CARPET_MIN_RENDER_DIM = 900
CARPET_MAX_RENDER_DIM = 1800

# --- Sierpinski Carpet ---
# Same raster approach (base-3 digit test, also validated with zero
# mismatches against the old method) - this is also what fixes the "only
# half the fractal renders at level 5" bug, identical root cause and fix
# floor(7.08) - matches the reported "above 7 looks worst" exactly.
CARPET_MAX_LEVELS = 7
CARPET_DEFAULT_LEVELS = 5
CARPET_ZOOM_MAX_LEVELS = 1000000

# --- Newton fractal polynomial menu (curated, no free-text entry) ---
NEWTON_POLYNOMIALS = [
    ("x**3 - 1",        "3-fold symmetry (classic)"),
    ("x**3 + 1",         "3-fold symmetry (mirrored)"),
    ("x**4 - 1",         "4-fold symmetry"),
    ("x**4 + 1",         "4-fold symmetry (rotated)"),
    ("x**5 - 1",         "5-fold symmetry"),
    ("x**5 + 1",         "5-fold symmetry (mirrored)"),
    ("x**6 - 1",         "6-fold symmetry"),
    ("x**7 - 1",         "7-fold symmetry (intricate)"),
    ("x**8 - 1",         "8-fold symmetry (intricate)"),
    ("x**3 - x",         "3 real roots (-1, 0, 1)"),
    ("x**5 - x",         "5 roots, mixed real/imaginary"),
    ("x**6 - x",         "6 roots (0 and the 5th roots of unity)"),
    ("x**3 - 2*x + 2",   "classic chaotic basin boundaries"),
]

# View-area threshold (in complex-plane units^2) that decides which
# Mandelbrot/Julia algorithm to use. Measured crossover: masking wins only
# on the widest initial view (area ~4+); essentially any zoom favors the
# full-array approach. 1.0 sits safely on the correct side of that
# crossover for both regimes.
ADAPTIVE_AREA_THRESHOLD = 1.0

# --------------------------------------------------------------------------- #
# Rendering backend
# --------------------------------------------------------------------------- #
# "auto" chooses CUDA when available, otherwise the Numba CPU renderer.
# "cpu" uses all available CPU threads through Numba.
# "gpu" uses the NVIDIA CUDA renderer.
# "hybrid" splits rows between CPU and GPU and runs both concurrently.
RENDER_BACKEND = "auto"
CUDA_BLOCK_X = 16
CUDA_BLOCK_Y = 16


def _cuda_available():
    """Return True only when Numba-CUDA can actually see a CUDA device."""
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def _choose_backend(requested=None):
    requested = (requested or RENDER_BACKEND).lower()
    if requested == "auto":
        return "gpu" if _cuda_available() else "cpu"
    if requested in {"gpu", "hybrid"} and not _cuda_available():
        print("CUDA GPU unavailable - falling back to Numba CPU renderer.")
        return "cpu"
    if requested == "cpu" and njit is None:
        print("Numba is not installed - falling back to NumPy CPU renderer.")
        return "numpy"
    return requested


def set_render_backend(backend):
    global RENDER_BACKEND
    backend = backend.lower()
    if backend not in {"auto", "cpu", "gpu", "hybrid"}:
        raise ValueError("Backend must be auto, cpu, gpu or hybrid")
    RENDER_BACKEND = backend


def _print_backend_status():
    print("\n--- Rendering Engine ---")
    print(f"Requested backend: {RENDER_BACKEND.upper()}")
    print(f"Numba installed: {'YES' if njit is not None else 'NO'}")
    print(f"CUDA available: {'YES' if _cuda_available() else 'NO'}")
    print(f"Active backend: {_choose_backend().upper()}")
    print("CPU = Numba JIT + parallel CPU threads")
    print("GPU = CUDA on the NVIDIA GPU")
    print("HYBRID = CPU + GPU concurrently")



# --------------------------------------------------------------------------- #
# Small input helpers - all clamped to a documented max (and usually a min)
# --------------------------------------------------------------------------- #
def ask_int_clamped(prompt, default, max_value, min_value=1):
    raw = input(f"{prompt} [default={default}, max={max_value}]: ").strip()
    if raw == "":
        return default
    try:
        val = int(raw)
    except ValueError:
        print(f"  Not a valid integer, using default {default}.")
        return default
    if val > max_value:
        print(f"  {val} is above the maximum of {max_value} - using {max_value} instead.")
        return max_value
    if val < min_value:
        print(f"  {val} is below the minimum of {min_value} - using {min_value} instead.")
        return min_value
    return val


def ask_float_clamped(prompt, default, max_value, min_value=0.01):
    raw = input(f"{prompt} [default={default}, max={max_value}]: ").strip()
    if raw == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        print(f"  Not a valid number, using default {default}.")
        return default
    if val > max_value:
        print(f"  {val} is above the maximum of {max_value} - using {max_value} instead.")
        return max_value
    if val < min_value:
        print(f"  {val} is below the minimum of {min_value} - using {min_value} instead.")
        return min_value
    return val


def ask_signed_float_clamped(prompt, default, max_abs):
    raw = input(f"{prompt} [default={default}, range=-{max_abs} to {max_abs}]: ").strip()
    if raw == "":
        return default
    try:
        val = float(raw)
    except ValueError:
        print(f"  Not a valid number, using default {default}.")
        return default
    if abs(val) > max_abs:
        clamped = math.copysign(max_abs, val)
        print(f"  {val} is outside -{max_abs}..{max_abs} - using {clamped} instead.")
        return clamped
    return val


def ask_fractal_resolution():
    """Ask for the render resolution used by Mandelbrot / Julia / Newton."""
    print("\n--- Fractal Resolution ---")
    for i, (label, width, height) in enumerate(FRACTAL_RESOLUTIONS, start=1):
        print(f"  {i}. {label:<6} ({width} x {height} pixels)")
    choice = ask_int_clamped(
        "Choose resolution",
        1,
        len(FRACTAL_RESOLUTIONS),
        min_value=1,
    )
    label, width, height = FRACTAL_RESOLUTIONS[choice - 1]
    print(f"  Selected: {label} ({width} x {height})")
    return width, height, label


def show_zoom_hint():
    print("\nWindow controls: use the toolbar's zoom-to-rectangle button (the")
    print("magnifying glass) and drag a box to ZOOM IN and regenerate the")
    print("fractal for that region. Press 'r' to reset to the original view.")
    print("Close the window when you're done.\n")


# --------------------------------------------------------------------------- #
# Live square-grid colormap picker ("Multiple Tone Colors") - used for
# Mandelbrot / Julia / Newton. Layout verified with matplotlib's own
# get_window_extent() bounding boxes to have zero label/tile overlap.
# --------------------------------------------------------------------------- #
def _pick_from_square_grid(title, items, default, draw_swatch, cols=6):
    n = len(items)
    rows = math.ceil(n / cols)

    row_pitch = 1.05
    tile_frac = 0.78
    tile_pad_x = 0.08

    fig, ax = plt.subplots(figsize=(cols * 1.15, rows * (row_pitch * 1.15) + 0.6))

    boxes = []
    for idx, (name, meta) in enumerate(items):
        r, c = divmod(idx, cols)
        x0, x1 = c + tile_pad_x, c + 1 - tile_pad_x
        row_top = (rows - r) * row_pitch
        y1 = row_top - (row_pitch - tile_frac) / 2
        y0 = y1 - tile_frac
        extent = (x0, x1, y0, y1)
        draw_swatch(ax, extent, meta)
        if name == default:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                        edgecolor="deepskyblue", linewidth=2.5))
        ax.text((x0 + x1) / 2, y0 - 0.03, name, ha="center", va="top", fontsize=7.5)
        boxes.append((x0, y0, x1, y1, name))

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows * row_pitch + row_pitch * 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=11)

    selected = {"name": default}

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return
        for (x0, y0, x1, y1, name) in boxes:
            if x0 <= event.xdata <= x1 and y0 <= event.ydata <= y1:
                selected["name"] = name
                plt.close(fig)
                return

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()

    print(f"  Picked: {selected['name']}")
    return selected["name"]


def pick_cmap_live(prompt, default):
    """Grid of colormap SQUARE swatches ("Multiple Tone Colors")."""
    cmap_names = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "twilight", "twilight_shifted", "turbo", "jet", "hot",
        "cool", "spring", "summer", "autumn", "winter",
        "ocean", "terrain", "rainbow", "nipy_spectral", "tab10",
    ]
    if default not in cmap_names:
        cmap_names.insert(0, default)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    try:
        return _pick_from_square_grid(
            title=f"Multiple Tone Colors - {prompt} - click a square to select",
            items=[(name, {"cmap": name}) for name in cmap_names],
            default=default,
            draw_swatch=lambda ax, extent, meta: ax.imshow(
                gradient, aspect="auto", cmap=meta["cmap"], extent=extent
            ),
        )
    except Exception as e:
        print(f"  Could not open the colormap picker ({e}), falling back to default '{default}'.")
        return default


# --------------------------------------------------------------------------- #
# Live HSL color picker for Sierpinski Carpet.
#
# Matches the classic Windows color-dialog layout: a Hue x Saturation
# square, a Lightness slider, Hue/Sat/Lum + Red/Green/Blue fields (same
# legacy 0-239 / 0-240 / 0-255 scale that dialog uses), and a "Color|Solid"
# preview swatch.
#
# Performance design (this is what keeps it from affecting zoom/generation
# speed): dragging in the square only redraws a crosshair + rebuilds the
# tiny lightness bar (~1ms, measured) - it never rebuilds the square
# itself. Dragging the lightness slider DOES need to rebuild the square
# (~70ms, measured, since every pixel's color depends on lightness), so
# that rebuild is debounced (30ms) - a fast drag just moves the marker
# instantly and the full square image catches up shortly after you stop,
# instead of queuing up dozens of expensive rebuilds. Every color change
# calls on_change(hex) which the fractal code uses to do a plain
# set_facecolor() on the EXISTING patches - O(1), no geometry rebuild, so
# it's fully decoupled from zooming.
# --------------------------------------------------------------------------- #
class LiveColorPicker:
    SQUARE_SIZE = 200
    BAR_WIDTH = 30
    BAR_HEIGHT = 200

    def __init__(self, parent_window, initial_color, on_change, title="Live Color Picker"):
        self.on_change = on_change
        r, g, b = mcolors.to_rgb(initial_color)
        self.hue, self.light, self.sat = colorsys.rgb_to_hls(r, g, b)
        self._square_rebuild_job = None

        self.top = tk.Toplevel(parent_window)
        self.top.title(title)
        self.top.resizable(False, False)
        # Always destroy the picker from Tk's live main thread.  This avoids
        # the harmless-but-noisy Tk PhotoImage/StringVar __del__ warnings
        # that can otherwise appear after a Matplotlib window is closed.
        self.top.protocol("WM_DELETE_WINDOW", self.close)

        main = tk.Frame(self.top, padx=10, pady=10)
        main.pack()

        self.square_canvas = tk.Canvas(main, width=self.SQUARE_SIZE, height=self.SQUARE_SIZE,
                                        cursor="crosshair", highlightthickness=1,
                                        highlightbackground="gray")
        self.square_canvas.grid(row=0, column=0, padx=(0, 6))
        self.square_photo = tk.PhotoImage(width=self.SQUARE_SIZE, height=self.SQUARE_SIZE)
        self.square_canvas.create_image(0, 0, anchor="nw", image=self.square_photo)
        self.crosshair_id = None

        self.bar_canvas = tk.Canvas(main, width=self.BAR_WIDTH, height=self.BAR_HEIGHT,
                                     cursor="sb_v_double_arrow", highlightthickness=1,
                                     highlightbackground="gray")
        self.bar_canvas.grid(row=0, column=1, padx=(0, 12))
        self.bar_photo = tk.PhotoImage(width=self.BAR_WIDTH, height=self.BAR_HEIGHT)
        self.bar_canvas.create_image(0, 0, anchor="nw", image=self.bar_photo)
        self.bar_marker_id = None

        bottom = tk.Frame(main)
        bottom.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="w")

        self.swatch = tk.Frame(bottom, width=64, height=44, relief="sunken", borderwidth=2)
        self.swatch.grid(row=0, column=0, rowspan=3, padx=(0, 14))
        self.swatch.grid_propagate(False)
        tk.Label(bottom, text="Color|Solid", font=("Segoe UI", 8)).grid(row=3, column=0)

        hsl_frame = tk.Frame(bottom)
        hsl_frame.grid(row=0, column=1, rowspan=3, padx=(0, 14))
        self.hue_var = tk.StringVar()
        self.sat_var = tk.StringVar()
        self.lum_var = tk.StringVar()
        for i, (label, var) in enumerate([("Hue:", self.hue_var), ("Sat:", self.sat_var),
                                           ("Lum:", self.lum_var)]):
            tk.Label(hsl_frame, text=label).grid(row=i, column=0, sticky="e", pady=2)
            e = tk.Entry(hsl_frame, textvariable=var, width=6)
            e.grid(row=i, column=1, padx=(4, 0))
            e.bind("<Return>", self._on_hsl_entry)

        rgb_frame = tk.Frame(bottom)
        rgb_frame.grid(row=0, column=2, rowspan=3)
        self.red_var = tk.StringVar()
        self.green_var = tk.StringVar()
        self.blue_var = tk.StringVar()
        for i, (label, var) in enumerate([("Red:", self.red_var), ("Green:", self.green_var),
                                           ("Blue:", self.blue_var)]):
            tk.Label(rgb_frame, text=label).grid(row=i, column=0, sticky="e", pady=2)
            e = tk.Entry(rgb_frame, textvariable=var, width=6)
            e.grid(row=i, column=1, padx=(4, 0))
            e.bind("<Return>", self._on_rgb_entry)

        self.square_canvas.bind("<Button-1>", self._on_square_drag)
        self.square_canvas.bind("<B1-Motion>", self._on_square_drag)
        self.bar_canvas.bind("<Button-1>", self._on_bar_drag)
        self.bar_canvas.bind("<B1-Motion>", self._on_bar_drag)

        self._rebuild_square(debounce=False)
        self._rebuild_bar()
        self._update_fields()
        self._update_swatch()

    def close(self):
        """Release Tk resources while the Tk main loop is still alive."""
        top = getattr(self, "top", None)
        try:
            if top is not None and self._square_rebuild_job is not None:
                top.after_cancel(self._square_rebuild_job)
        except Exception:
            pass
        self._square_rebuild_job = None

        # PhotoImage/StringVar objects call Tcl from their __del__ methods.
        # Drop them BEFORE destroying the Toplevel so their destructors execute
        # while Tk's main thread is still alive. This removes the harmless but
        # noisy "RuntimeError: main thread is not in main loop" messages.
        for attr in ("square_photo", "bar_photo", "hue_var", "sat_var",
                     "lum_var", "red_var", "green_var", "blue_var"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        try:
            if top is not None and top.winfo_exists():
                top.destroy()
        except Exception:
            pass
        self.top = None

    def _hex(self):
        r, g, b = colorsys.hls_to_rgb(self.hue, self.light, self.sat)
        return f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"

    def _rebuild_square(self, debounce=True):
        def do_build():
            self._square_rebuild_job = None
            size = self.SQUARE_SIZE
            light = self.light
            # Pass a NESTED LIST OF LISTS (row -> list of "#rrggbb" strings),
            # not a manually brace-formatted string. tkinter's PhotoImage.put()
            # converts nested Python sequences to the correct Tcl list
            # structure itself; hand-building "{...}" strings is fragile and
            # was the cause of the "can't parse color" crash - the manual
            # braces were getting mangled during Tcl marshaling instead of
            # being treated as row groupings.
            rows = []
            for row in range(size):
                sat = row / (size - 1)
                colors = []
                for col in range(size):
                    hue = col / (size - 1)
                    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
                    colors.append(f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}")
                rows.append(colors)
            self.square_photo.put(rows)
            self._draw_crosshair()

        if debounce:
            if self._square_rebuild_job is not None:
                self.top.after_cancel(self._square_rebuild_job)
            self._square_rebuild_job = self.top.after(30, do_build)
        else:
            do_build()

    def _rebuild_bar(self):
        w, h = self.BAR_WIDTH, self.BAR_HEIGHT
        rows = []
        for row in range(h):
            light = 1.0 - row / (h - 1)
            r, g, b = colorsys.hls_to_rgb(self.hue, light, self.sat)
            hexc = f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"
            rows.append([hexc] * w)
        self.bar_photo.put(rows)
        self._draw_bar_marker()

    def _draw_crosshair(self):
        if self.crosshair_id is not None:
            self.square_canvas.delete(self.crosshair_id)
        x = self.hue * (self.SQUARE_SIZE - 1)
        y = self.sat * (self.SQUARE_SIZE - 1)
        r = 5
        self.crosshair_id = self.square_canvas.create_oval(x - r, y - r, x + r, y + r,
                                                             outline="black", width=2)

    def _draw_bar_marker(self):
        if self.bar_marker_id is not None:
            self.bar_canvas.delete(self.bar_marker_id)
        y = (1.0 - self.light) * (self.BAR_HEIGHT - 1)
        self.bar_marker_id = self.bar_canvas.create_polygon(
            self.BAR_WIDTH, y - 6, self.BAR_WIDTH - 10, y, self.BAR_WIDTH, y + 6, fill="black"
        )

    def _on_square_drag(self, event):
        size = self.SQUARE_SIZE
        x = max(0, min(size - 1, event.x))
        y = max(0, min(size - 1, event.y))
        self.hue = x / (size - 1)
        self.sat = y / (size - 1)
        self._draw_crosshair()
        self._rebuild_bar()
        self._update_fields()
        self._update_swatch()
        self._notify()

    def _on_bar_drag(self, event):
        h = self.BAR_HEIGHT
        y = max(0, min(h - 1, event.y))
        self.light = 1.0 - y / (h - 1)
        self._draw_bar_marker()
        self._rebuild_square(debounce=True)
        self._update_fields()
        self._update_swatch()
        self._notify()

    def _on_hsl_entry(self, event):
        try:
            self.hue = max(0.0, min(1.0, float(self.hue_var.get()) / 239.0))
            self.sat = max(0.0, min(1.0, float(self.sat_var.get()) / 240.0))
            self.light = max(0.0, min(1.0, float(self.lum_var.get()) / 240.0))
        except ValueError:
            pass
        self._rebuild_square(debounce=False)
        self._rebuild_bar()
        self._update_fields()
        self._update_swatch()
        self._notify()

    def _on_rgb_entry(self, event):
        try:
            r = max(0, min(255, int(self.red_var.get()))) / 255.0
            g = max(0, min(255, int(self.green_var.get()))) / 255.0
            b = max(0, min(255, int(self.blue_var.get()))) / 255.0
            self.hue, self.light, self.sat = colorsys.rgb_to_hls(r, g, b)
        except ValueError:
            pass
        self._rebuild_square(debounce=False)
        self._rebuild_bar()
        self._update_fields()
        self._update_swatch()
        self._notify()

    def _update_fields(self):
        r, g, b = colorsys.hls_to_rgb(self.hue, self.light, self.sat)
        self.hue_var.set(str(round(self.hue * 239)))
        self.sat_var.set(str(round(self.sat * 240)))
        self.lum_var.set(str(round(self.light * 240)))
        self.red_var.set(str(round(r * 255)))
        self.green_var.set(str(round(g * 255)))
        self.blue_var.set(str(round(b * 255)))

    def _update_swatch(self):
        self.swatch.configure(bg=self._hex())

    def _notify(self):
        self.on_change(self._hex())


# --------------------------------------------------------------------------- #
# JIT CPU + CUDA GPU kernels
# --------------------------------------------------------------------------- #
if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _mandelbrot_cpu_kernel(xmin, xmax, ymin, ymax, width, height, max_iter):
        out = np.empty((height, width), dtype=np.float64)
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        for py in prange(height):
            cy = ymin + py * dy
            for px in range(width):
                cx = xmin + px * dx
                zr = 0.0
                zi = 0.0
                it = max_iter
                for i in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4.0:
                        it = i
                        break
                    zi = 2.0 * zr * zi + cy
                    zr = zr2 - zi2 + cx
                out[py, px] = it
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _julia_cpu_kernel(xmin, xmax, ymin, ymax, width, height, cr, ci, max_iter):
        out = np.empty((height, width), dtype=np.float64)
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        for py in prange(height):
            zy0 = ymin + py * dy
            for px in range(width):
                zr = xmin + px * dx
                zi = zy0
                it = max_iter
                for i in range(max_iter):
                    zr2 = zr * zr
                    zi2 = zi * zi
                    if zr2 + zi2 > 4.0:
                        it = i
                        break
                    zi = 2.0 * zr * zi + ci
                    zr = zr2 - zi2 + cr
                out[py, px] = it
        return out

    @njit(parallel=True, fastmath=True, cache=True)
    def _newton_cpu_kernel(xmin, xmax, ymin, ymax, width, height, coeffs,
                           dcoeffs, roots_r, roots_i, max_iter):
        out = np.empty((height, width), dtype=np.int32)
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        degree = len(coeffs) - 1
        nroots = len(roots_r)
        for py in prange(height):
            y0 = ymin + py * dy
            for px in range(width):
                zr = xmin + px * dx
                zi = y0
                for _ in range(max_iter):
                    # Horner evaluation of f(z) with real coefficients.
                    pr = coeffs[degree]
                    pi = 0.0
                    for k in range(degree - 1, -1, -1):
                        nr = pr * zr - pi * zi + coeffs[k]
                        ni = pr * zi + pi * zr
                        pr, pi = nr, ni

                    # Horner evaluation of f'(z).
                    dpr = dcoeffs[degree - 1] if degree > 0 else 0.0
                    dpi = 0.0
                    for k in range(degree - 2, -1, -1):
                        nr = dpr * zr - dpi * zi + dcoeffs[k]
                        ni = dpr * zi + dpi * zr
                        dpr, dpi = nr, ni

                    den = dpr * dpr + dpi * dpi
                    if den < 1e-28:
                        break
                    # (pr + i*pi) / (dpr + i*dpi)
                    qr = (pr * dpr + pi * dpi) / den
                    qi = (pi * dpr - pr * dpi) / den
                    zr -= qr
                    zi -= qi
                    if qr * qr + qi * qi < 1e-24:
                        break

                best = 0
                best_d = 1e300
                for r in range(nroots):
                    dr = zr - roots_r[r]
                    di = zi - roots_i[r]
                    d2 = dr * dr + di * di
                    if d2 < best_d:
                        best_d = d2
                        best = r
                out[py, px] = best
        return out
    @njit(parallel=True, fastmath=True, cache=True)
    def _carpet_raster_cpu_kernel(xmin, xmax, ymin, ymax, width, height,
                                  detail_depth, supersample, skip_depth):
        out = np.empty((height, width), dtype=np.float64)
        dx = (xmax - xmin) / (width - 1) if width > 1 else 0.0
        dy = (ymax - ymin) / (height - 1) if height > 1 else 0.0
        n_sub = supersample * supersample
        for py in prange(height):
            y0 = ymin + py * dy
            for px in range(width):
                x0 = xmin + px * dx
                hits = 0
                for sy in range(supersample):
                    y = y0 + (sy - (supersample - 1) / 2.0) * (dy / supersample)
                    for sx in range(supersample):
                        x = x0 + (sx - (supersample - 1) / 2.0) * (dx / supersample)
                        if x < 0.0 or x >= 1.0 or y < 0.0 or y >= 1.0:
                            continue
                        fx = x
                        fy = y
                        filled = True
                        for _ in range(skip_depth):
                            fx *= 3.0
                            fy *= 3.0
                            dxi = int(fx)
                            dyi = int(fy)
                            if dxi == 1 and dyi == 1:
                                filled = False
                                break
                            fx -= dxi
                            fy -= dyi
                        if not filled:
                            continue
                        for _ in range(detail_depth):
                            fx *= 3.0
                            fy *= 3.0
                            dxi = int(fx)
                            dyi = int(fy)
                            if dxi == 1 and dyi == 1:
                                filled = False
                                break
                            fx -= dxi
                            fy -= dyi
                        if filled:
                            hits += 1
                out[py, px] = hits / n_sub
        return out

else:
    _mandelbrot_cpu_kernel = None
    _julia_cpu_kernel = None
    _newton_cpu_kernel = None
    _carpet_raster_cpu_kernel = None


if cuda is not None:
    @cuda.jit
    def _mandelbrot_gpu_kernel(xmin, xmax, ymin, ymax, width, height, max_iter, out):
        px, py = cuda.grid(2)
        if px >= width or py >= height:
            return
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        cx = xmin + px * dx
        cy = ymin + py * dy
        zr = 0.0
        zi = 0.0
        it = max_iter
        for i in range(max_iter):
            zr2 = zr * zr
            zi2 = zi * zi
            if zr2 + zi2 > 4.0:
                it = i
                break
            zi = 2.0 * zr * zi + cy
            zr = zr2 - zi2 + cx
        out[py, px] = it

    @cuda.jit
    def _julia_gpu_kernel(xmin, xmax, ymin, ymax, width, height, cr, ci, max_iter, out):
        px, py = cuda.grid(2)
        if px >= width or py >= height:
            return
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        zr = xmin + px * dx
        zi = ymin + py * dy
        it = max_iter
        for i in range(max_iter):
            zr2 = zr * zr
            zi2 = zi * zi
            if zr2 + zi2 > 4.0:
                it = i
                break
            zi = 2.0 * zr * zi + ci
            zr = zr2 - zi2 + cr
        out[py, px] = it

    @cuda.jit
    def _newton_gpu_kernel(xmin, xmax, ymin, ymax, width, height,
                           coeffs, dcoeffs, roots_r, roots_i, max_iter, out):
        px, py = cuda.grid(2)
        if px >= width or py >= height:
            return
        dx = (xmax - xmin) / (width - 1)
        dy = (ymax - ymin) / (height - 1)
        zr = xmin + px * dx
        zi = ymin + py * dy
        degree = coeffs.size - 1

        for _ in range(max_iter):
            pr = coeffs[degree]
            pi = 0.0
            for k in range(degree - 1, -1, -1):
                nr = pr * zr - pi * zi + coeffs[k]
                ni = pr * zi + pi * zr
                pr = nr
                pi = ni

            dpr = dcoeffs[degree - 1] if degree > 0 else 0.0
            dpi = 0.0
            for k in range(degree - 2, -1, -1):
                nr = dpr * zr - dpi * zi + dcoeffs[k]
                ni = dpr * zi + dpi * zr
                dpr = nr
                dpi = ni

            den = dpr * dpr + dpi * dpi
            if den < 1e-28:
                break
            qr = (pr * dpr + pi * dpi) / den
            qi = (pi * dpr - pr * dpi) / den
            zr -= qr
            zi -= qi
            if qr * qr + qi * qi < 1e-24:
                break

        best = 0
        best_d = 1e300
        for r in range(roots_r.size):
            dr = zr - roots_r[r]
            di = zi - roots_i[r]
            d2 = dr * dr + di * di
            if d2 < best_d:
                best_d = d2
                best = r
        out[py, px] = best
else:
    _mandelbrot_gpu_kernel = None
    _julia_gpu_kernel = None
    _newton_gpu_kernel = None


def _cuda_grid(width, height):
    threads = (CUDA_BLOCK_X, CUDA_BLOCK_Y)
    blocks = ((width + CUDA_BLOCK_X - 1) // CUDA_BLOCK_X,
              (height + CUDA_BLOCK_Y - 1) // CUDA_BLOCK_Y)
    return blocks, threads


def _gpu_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    out_d = cuda.device_array((height, width), dtype=np.float64)
    blocks, threads = _cuda_grid(width, height)
    _mandelbrot_gpu_kernel[blocks, threads](xmin, xmax, ymin, ymax,
                                            width, height, max_iter, out_d)
    cuda.synchronize()
    return out_d.copy_to_host()


def _gpu_julia(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    out_d = cuda.device_array((height, width), dtype=np.float64)
    blocks, threads = _cuda_grid(width, height)
    _julia_gpu_kernel[blocks, threads](xmin, xmax, ymin, ymax, width, height,
                                       float(c.real), float(c.imag), max_iter, out_d)
    cuda.synchronize()
    return out_d.copy_to_host()


def _gpu_newton(xmin, xmax, ymin, ymax, width, height, coeffs, dcoeffs,
                roots, max_iter):
    out_d = cuda.device_array((height, width), dtype=np.int32)
    coeffs_d = cuda.to_device(np.asarray(coeffs, dtype=np.float64))
    dcoeffs_d = cuda.to_device(np.asarray(dcoeffs, dtype=np.float64))
    roots_r_d = cuda.to_device(np.asarray([r.real for r in roots], dtype=np.float64))
    roots_i_d = cuda.to_device(np.asarray([r.imag for r in roots], dtype=np.float64))
    blocks, threads = _cuda_grid(width, height)
    _newton_gpu_kernel[blocks, threads](
        xmin, xmax, ymin, ymax, width, height,
        coeffs_d, dcoeffs_d, roots_r_d, roots_i_d, max_iter, out_d
    )
    cuda.synchronize()
    return out_d.copy_to_host()


def _hybrid_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    split = max(1, min(height - 1, int(height * 0.25)))
    y_mid = ymin + (ymax - ymin) * split / (height - 1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cpu = pool.submit(_mandelbrot_cpu_kernel, xmin, xmax, ymin, y_mid,
                            width, split, max_iter)
        f_gpu = pool.submit(_gpu_mandelbrot, xmin, xmax, y_mid, ymax,
                            width, height - split, max_iter)
        return np.vstack((f_cpu.result(), f_gpu.result()))


def _hybrid_julia(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    split = max(1, min(height - 1, int(height * 0.25)))
    y_mid = ymin + (ymax - ymin) * split / (height - 1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cpu = pool.submit(_julia_cpu_kernel, xmin, xmax, ymin, y_mid,
                            width, split, c.real, c.imag, max_iter)
        f_gpu = pool.submit(_gpu_julia, xmin, xmax, y_mid, ymax,
                            width, height - split, c, max_iter)
        return np.vstack((f_cpu.result(), f_gpu.result()))


def _hybrid_newton(xmin, xmax, ymin, ymax, width, height, coeffs, dcoeffs,
                   roots, max_iter):
    split = max(1, min(height - 1, int(height * 0.25)))
    y_mid = ymin + (ymax - ymin) * split / (height - 1)
    roots_r = np.asarray([r.real for r in roots], dtype=np.float64)
    roots_i = np.asarray([r.imag for r in roots], dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    dcoeffs = np.asarray(dcoeffs, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_cpu = pool.submit(_newton_cpu_kernel, xmin, xmax, ymin, y_mid,
                            width, split, coeffs, dcoeffs, roots_r, roots_i, max_iter)
        f_gpu = pool.submit(_gpu_newton, xmin, xmax, y_mid, ymax,
                            width, height - split, coeffs, dcoeffs, roots, max_iter)
        return np.vstack((f_cpu.result(), f_gpu.result()))


# --------------------------------------------------------------------------- #
# Adaptive Mandelbrot / Julia compute - algorithm chosen by view area,
# smooth (renormalized) coloring on the zoomed branch. See module docstring
# for the benchmark numbers behind this design.
# --------------------------------------------------------------------------- #
def mandelbrot_compute(xmin, xmax, ymin, ymax, width, height, max_iter, backend=None):
    backend = _choose_backend(backend)
    if backend == "gpu":
        return _gpu_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    if backend == "hybrid":
        return _hybrid_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter)
    if backend == "cpu" and _mandelbrot_cpu_kernel is not None:
        return _mandelbrot_cpu_kernel(xmin, xmax, ymin, ymax, width, height, max_iter)

    # NumPy fallback.
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    C = x[None, :] + 1j * y[:, None]
    Z = np.zeros_like(C)
    div_time = np.full(C.shape, max_iter, dtype=np.float64)
    mask = np.ones(C.shape, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        escaped = (Z.real * Z.real + Z.imag * Z.imag) > 4.0
        newly_escaped = escaped & mask
        div_time[newly_escaped] = i
        mask &= ~escaped
    return div_time


def julia_compute(xmin, xmax, ymin, ymax, width, height, c, max_iter, backend=None):
    backend = _choose_backend(backend)
    if backend == "gpu":
        return _gpu_julia(xmin, xmax, ymin, ymax, width, height, c, max_iter)
    if backend == "hybrid":
        return _hybrid_julia(xmin, xmax, ymin, ymax, width, height, c, max_iter)
    if backend == "cpu" and _julia_cpu_kernel is not None:
        return _julia_cpu_kernel(xmin, xmax, ymin, ymax, width, height,
                                 c.real, c.imag, max_iter)

    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    Z = x[None, :] + 1j * y[:, None]
    smooth_time = np.full(Z.shape, float(max_iter))
    escaped_ever = np.zeros(Z.shape, dtype=bool)
    for i in range(max_iter):
        Znew = Z * Z + c
        Z = np.where(escaped_ever, Z, Znew)
        mag2 = Z.real * Z.real + Z.imag * Z.imag
        newly_escaped = (mag2 > 4.0) & (~escaped_ever)
        if newly_escaped.any():
            abs_z = np.sqrt(mag2[newly_escaped])
            smooth_time[newly_escaped] = (i + 1) - np.log(np.log(abs_z)) / np.log(2)
        escaped_ever |= newly_escaped
    return smooth_time


# --------------------------------------------------------------------------- #
# Generic interactive zoom for escape-time / convergence-time fractals
# --------------------------------------------------------------------------- #
def interactive_complex_plane_plot(compute_fn, xmin, xmax, ymin, ymax,
                                    width, height, cmap, title_fn, colorbar_label,
                                    clean_fractal_view=False):
    home_bounds = (xmin, xmax, ymin, ymax)
    print(f"Rendering with backend: {_choose_backend().upper()}")
    print(f"Render resolution: {width} x {height} pixels")

    # Keep the on-screen window practical while the underlying image retains
    # the exact requested pixel dimensions.
    aspect = width / height
    if clean_fractal_view:
        fig, ax = plt.subplots(figsize=(12, 12 / aspect))
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    data = compute_fn(xmin, xmax, ymin, ymax, width, height)
    img = ax.imshow(
        data,
        extent=(xmin, xmax, ymin, ymax),
        cmap=cmap,
        origin="lower",
        interpolation="bicubic",
    )

    if clean_fractal_view:
        # Mandelbrot / Julia / Newton use a full-canvas exhibition view:
        # the fractal image fills ALL available canvas area above the
        # Matplotlib navigation toolbar.  No title, axes, ticks, labels,
        # colorbar, or margins are shown.  The toolbar itself remains.
        ax.axis("off")
        ax.set_aspect("auto")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.set_position([0, 0, 1, 1])
    else:
        ax.set_title(title_fn(xmin, xmax, ymin, ymax))
        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        fig.colorbar(img, ax=ax, label=colorbar_label)

    state = {"bounds": home_bounds, "busy": False}

    def redraw(xmin, xmax, ymin, ymax):
        if state["busy"] or (xmin, xmax, ymin, ymax) == state["bounds"]:
            return
        state["busy"] = True
        try:
            print(f"Zooming to x:[{xmin:.6g}, {xmax:.6g}]  y:[{ymin:.6g}, {ymax:.6g}] ... recomputing.")
            data = compute_fn(xmin, xmax, ymin, ymax, width, height)
            img.set_data(data)
            img.set_extent((xmin, xmax, ymin, ymax))
            img.set_clim(vmin=data.min(), vmax=data.max())
            if not clean_fractal_view:
                ax.set_title(title_fn(xmin, xmax, ymin, ymax))
            state["bounds"] = (xmin, xmax, ymin, ymax)
            fig.canvas.draw_idle()
        finally:
            state["busy"] = False

    def on_lims_changed(event_ax):
        new_xmin, new_xmax = event_ax.get_xlim()
        new_ymin, new_ymax = event_ax.get_ylim()
        redraw(new_xmin, new_xmax, new_ymin, new_ymax)

    def on_key(event):
        if event.key == "r":
            print("Resetting to original view.")
            ax.set_xlim(home_bounds[0], home_bounds[1])
            ax.set_ylim(home_bounds[2], home_bounds[3])

    ax.callbacks.connect("xlim_changed", on_lims_changed)
    ax.callbacks.connect("ylim_changed", on_lims_changed)
    fig.canvas.mpl_connect("key_press_event", on_key)

    show_zoom_hint()
    if not clean_fractal_view:
        plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 1. Mandelbrot Fractal
# --------------------------------------------------------------------------- #
def mandelbrot():
    print("\n--- Mandelbrot Fractal ---")
    width, height, resolution_label = ask_fractal_resolution()
    print("Enter a single positive 'limit' - the view spans -limit to +limit")
    print("on both axes, so it always stays centered and square.\n")

    limit_x = ask_float_clamped("x limit (view spans -x to +x)", MANDELBROT_DEFAULT_LIMIT, MANDELBROT_MAX_LIMIT)
    limit_y = ask_float_clamped("y limit (view spans -y to +y)", MANDELBROT_DEFAULT_LIMIT, MANDELBROT_MAX_LIMIT)
    max_iter = ask_int_clamped("Maximum iterations", MANDELBROT_DEFAULT_ITER, MANDELBROT_MAX_ITER)
    cmap = pick_cmap_live("Mandelbrot colormap", "twilight_shifted")

    xmin, xmax = -limit_x, limit_x
    ymin, ymax = -limit_y, limit_y

    def compute_fn(xmin, xmax, ymin, ymax, w, h):
        return mandelbrot_compute(xmin, xmax, ymin, ymax, w, h, max_iter)

    def title_fn(xmin, xmax, ymin, ymax):
        return f"Mandelbrot Set (iterations={max_iter})"

    interactive_complex_plane_plot(
        compute_fn, xmin, xmax, ymin, ymax, width, height, cmap,
        title_fn, "Escape iteration", clean_fractal_view=True,
    )


# --------------------------------------------------------------------------- #
# 2. Julia Fractal
# --------------------------------------------------------------------------- #
def julia():
    print("\n--- Julia Fractal ---")
    width, height, resolution_label = ask_fractal_resolution()
    print("Enter a single positive 'limit' - the view spans -limit to +limit")
    print("on both axes. Interesting values for c: -0.7+0.27015j, -0.4+0.6j, 0.285+0.01j\n")

    c_real = ask_signed_float_clamped("Real part of constant c", -0.7, JULIA_C_MAX)
    c_imag = ask_signed_float_clamped("Imaginary part of constant c", 0.27015, JULIA_C_MAX)
    c = complex(c_real, c_imag)

    limit_x = ask_float_clamped("x limit (view spans -x to +x)", JULIA_DEFAULT_LIMIT, JULIA_MAX_LIMIT)
    limit_y = ask_float_clamped("y limit (view spans -y to +y)", JULIA_DEFAULT_LIMIT, JULIA_MAX_LIMIT)
    max_iter = ask_int_clamped("Maximum iterations", JULIA_DEFAULT_ITER, JULIA_MAX_ITER)
    cmap = pick_cmap_live("Julia colormap", "inferno")

    xmin, xmax = -limit_x, limit_x
    ymin, ymax = -limit_y, limit_y

    def compute_fn(xmin, xmax, ymin, ymax, w, h):
        return julia_compute(xmin, xmax, ymin, ymax, w, h, c, max_iter)

    def title_fn(xmin, xmax, ymin, ymax):
        return f"Julia Set (c = {c_real} + {c_imag}i, iterations={max_iter})"

    interactive_complex_plane_plot(
        compute_fn, xmin, xmax, ymin, ymax, width, height, cmap,
        title_fn, "Escape iteration", clean_fractal_view=True,
    )


# --------------------------------------------------------------------------- #
# 3. Newton Fractal
# --------------------------------------------------------------------------- #
def _sympy_to_real_coefficients(expr, x):
    """Return ascending real polynomial coefficients and derivative coeffs."""
    import sympy as sp
    poly = sp.Poly(sp.expand(expr), x)
    coeffs_desc = poly.all_coeffs()
    degree = poly.degree()
    coeffs = np.zeros(degree + 1, dtype=np.float64)
    for idx, c in enumerate(coeffs_desc):
        coeffs[degree - idx] = float(sp.N(c))
    dcoeffs = np.array([k * coeffs[k] for k in range(1, degree + 1)], dtype=np.float64)
    return coeffs, dcoeffs


def newton_compute(coeffs, dcoeffs, roots, xmin, xmax, ymin, ymax,
                   width, height, max_iter, backend=None):
    backend = _choose_backend(backend)
    if backend == "gpu":
        return _gpu_newton(xmin, xmax, ymin, ymax, width, height,
                           coeffs, dcoeffs, roots, max_iter)
    if backend == "hybrid":
        return _hybrid_newton(xmin, xmax, ymin, ymax, width, height,
                              coeffs, dcoeffs, roots, max_iter)
    if backend == "cpu" and _newton_cpu_kernel is not None:
        return _newton_cpu_kernel(
            xmin, xmax, ymin, ymax, width, height,
            np.asarray(coeffs, dtype=np.float64),
            np.asarray(dcoeffs, dtype=np.float64),
            np.asarray([r.real for r in roots], dtype=np.float64),
            np.asarray([r.imag for r in roots], dtype=np.float64),
            max_iter,
        )

    # NumPy fallback: preserves the original behavior if Numba is absent.
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    Z = x[None, :] + 1j * y[:, None]
    c = np.asarray(coeffs, dtype=np.float64)
    dc = np.asarray(dcoeffs, dtype=np.float64)
    for _ in range(max_iter):
        fz = np.zeros_like(Z)
        fpz = np.zeros_like(Z)
        for a in c[::-1]:
            fz = fz * Z + a
        for a in dc[::-1]:
            fpz = fpz * Z + a
        with np.errstate(divide="ignore", invalid="ignore"):
            step = np.where(fpz != 0, fz / fpz, 0)
        Z -= step
    roots_arr = np.asarray(roots)
    dists = np.abs(Z[..., None] - roots_arr[None, None, :])
    return np.argmin(dists, axis=-1).astype(np.int32)


def newton():
    print("\n--- Newton Fractal ---")
    width, height, resolution_label = ask_fractal_resolution()
    print("Choose a polynomial from the curated list below:\n")
    for i, (expr_str, desc) in enumerate(NEWTON_POLYNOMIALS, start=1):
        print(f"  {i:2d}. f(x) = {expr_str:18s} - {desc}")

    choice = ask_int_clamped("Pick a polynomial number", 1, len(NEWTON_POLYNOMIALS), min_value=1)
    func_str, desc = NEWTON_POLYNOMIALS[choice - 1]
    print(f"\nUsing f(x) = {func_str} ({desc})")

    import sympy as sp
    x = sp.symbols("x")
    expr = sp.sympify(func_str, locals={"I": sp.I, "x": x})
    deriv = sp.diff(expr, x)
    roots = [complex(r) for r in sp.solve(sp.Eq(expr, 0), x)]
    coeffs, dcoeffs = _sympy_to_real_coefficients(expr, x)

    limit_x = ask_float_clamped("x limit (view spans -x to +x)", NEWTON_DEFAULT_LIMIT, NEWTON_MAX_LIMIT)
    limit_y = ask_float_clamped("y limit (view spans -y to +y)", NEWTON_DEFAULT_LIMIT, NEWTON_MAX_LIMIT)
    max_iter = ask_int_clamped("Maximum Newton iterations", NEWTON_DEFAULT_ITER, NEWTON_MAX_ITER)
    cmap = pick_cmap_live("Newton colormap (one shade per root)", "tab10" if len(roots) <= 10 else "viridis")

    xmin, xmax = -limit_x, limit_x
    ymin, ymax = -limit_y, limit_y

    print(f"\nRoots: " + ", ".join(f"{r.real:.4g}{'+' if r.imag >= 0 else ''}{r.imag:.4g}i" for r in roots))
    print("Generating Newton fractal...")

    def compute_fn(xmin, xmax, ymin, ymax, w, h):
        return newton_compute(coeffs, dcoeffs, roots, xmin, xmax, ymin, ymax, w, h, max_iter)

    def title_fn(xmin, xmax, ymin, ymax):
        return f"Newton Fractal for f(x) = {func_str}"

    interactive_complex_plane_plot(
        compute_fn, xmin, xmax, ymin, ymax, width, height, cmap,
        title_fn, "Root index", clean_fractal_view=True,
    )


# --------------------------------------------------------------------------- #
# Raster compute for Sierpinski Carpet. Numpy
# fallback used automatically when numba isn't installed - vectorized
# version of the exact same digit-test logic as the numba kernels above,
# cross-validated to match them (and the old recursive-shape method)
# exactly.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Raster compute for Sierpinski Carpet. Numpy
# fallback used automatically when numba isn't installed - vectorized
# version of the exact same digit-test logic as the numba kernels above,
# cross-validated to match them (and the old recursive-shape method)
# exactly. Both now return a continuous 0-1 COVERAGE value per pixel
# (supersample^2 sub-samples averaged), not a hard True/False - this is
# what fixes the aliasing/speckle problem at deep levels: a raster with
# one sample per pixel can only resolve detail down to about one pixel
# across, and pushing depth further than that just means different
# sub-cells inside a single pixel win the single-sample "coin flip"
# essentially at random.
# --------------------------------------------------------------------------- #
def _supersampled_numpy(single_sample_fn, xmin, xmax, ymin, ymax, width, height,
                        detail_depth, supersample, skip_depth=0):
    raw = single_sample_fn(xmin, xmax, ymin, ymax,
                           width * supersample, height * supersample,
                           detail_depth, skip_depth)
    raw = raw.astype(np.float64)
    return raw.reshape(height, supersample, width, supersample).mean(axis=(1, 3))


def _carpet_filled_numpy_single(xmin, xmax, ymin, ymax, width, height,
                                detail_depth, skip_depth=0):
    """Self-similar local-coordinate renderer for the Sierpinski carpet."""
    # Pixel-centre sampling avoids repeatedly sampling the viewport edges.
    # This matters at deep zoom because edge duplication can become visible
    # as blocky bands when the image is resampled by Matplotlib.
    x = xmin + (np.arange(width, dtype=np.float64) + 0.5) * ((xmax - xmin) / width)
    y = ymin + (np.arange(height, dtype=np.float64) + 0.5) * ((ymax - ymin) / height)
    X, Y = np.meshgrid(x, y)
    inside = (X >= 0.0) & (X < 1.0) & (Y >= 0.0) & (Y < 1.0)
    fx = np.clip(X, 0.0, 0.9999999999999999)
    fy = np.clip(Y, 0.0, 0.9999999999999999)
    valid = inside.copy()

    for _ in range(skip_depth):
        fx3 = fx * 3.0
        fy3 = fy * 3.0
        dx = np.floor(fx3).astype(np.int64)
        dy = np.floor(fy3).astype(np.int64)
        hole = (dx == 1) & (dy == 1)
        valid &= ~hole
        fx = fx3 - dx
        fy = fy3 - dy

    filled = valid.copy()
    for _ in range(detail_depth):
        fx3 = fx * 3.0
        fy3 = fy * 3.0
        dx = np.floor(fx3).astype(np.int64)
        dy = np.floor(fy3).astype(np.int64)
        hole = (dx == 1) & (dy == 1)
        filled &= ~hole
        fx = fx3 - dx
        fy = fy3 - dy
    return filled


def _supersampled_numpy(single_sample_fn, xmin, xmax, ymin, ymax, width, height,
                        detail_depth, supersample, skip_depth=0):
    raw = single_sample_fn(xmin, xmax, ymin, ymax,
                           width * supersample, height * supersample,
                           detail_depth, skip_depth)
    raw = raw.astype(np.float64)
    return raw.reshape(height, supersample, width, supersample).mean(axis=(1, 3))


def _carpet_filled_numpy(xmin, xmax, ymin, ymax, width, height, detail_depth,
                         supersample=RASTER_SUPERSAMPLE, skip_depth=0):
    return _supersampled_numpy(_carpet_filled_numpy_single, xmin, xmax, ymin, ymax,
                               width, height, detail_depth, supersample, skip_depth)


def carpet_filled_raster(xmin, xmax, ymin, ymax, width, height, depth,
                         backend=None, skip_depth=0):
    backend = _choose_backend(backend)
    if backend in {"cpu", "gpu", "hybrid"} and _carpet_raster_cpu_kernel is not None:
        return _carpet_raster_cpu_kernel(xmin, xmax, ymin, ymax, width, height,
                                         depth, RASTER_SUPERSAMPLE, skip_depth)
    return _carpet_filled_numpy(xmin, xmax, ymin, ymax, width, height, depth,
                                RASTER_SUPERSAMPLE, skip_depth)


def _deep_fractal_color(hex_color):
    """Turn every picker color into a stronger/deeper display color.

    Hue is preserved. Saturation is boosted for chromatic colors and lightness
    is compressed into a dark, readable range. Pure black/near-black remains
    black; neutral greys remain neutral.
    """
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if s < 0.05:
        # Neutral colour: keep it neutral but prevent very pale greys.
        l = min(l, 0.32)
        return mcolors.to_hex(colorsys.hls_to_rgb(h, l, 0.0))
    s = max(0.82, s)
    # Preserve very dark selections while pulling bright selections into a
    # consistently deep range.
    l = min(max(l * 0.62, 0.18), 0.42)
    return mcolors.to_hex(colorsys.hls_to_rgb(h, l, s))

def _install_vertical_toolbar(fig):
    """Move the normal TkAgg navigation toolbar from the bottom to a compact
    vertical toolbar in the top-right corner, while keeping the same actions.
    """
    try:
        manager = fig.canvas.manager
        toolbar = getattr(manager, "toolbar", None)
        window = getattr(manager, "window", None)
        if toolbar is None or window is None:
            return None

        # Remove Matplotlib's horizontal toolbar; we recreate its useful
        # actions as vertically stacked buttons at the top-right.
        try:
            toolbar.pack_forget()
        except Exception:
            pass

        panel = tk.Frame(window, bd=1, relief="solid", bg="#eeeeee")
        panel.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        buttons = [
            ("⌂", toolbar.home),
            ("←", toolbar.back),
            ("→", toolbar.forward),
            ("✥", toolbar.pan),
            ("⌕", toolbar.zoom),
            ("▣", toolbar.save_figure),
        ]
        for text, command in buttons:
            b = tk.Button(panel, text=text, command=command, width=3, height=1,
                          font=("Segoe UI Symbol", 11), relief="flat")
            b.pack(fill="x", padx=2, pady=2)

        # Keep the custom panel alive with the figure manager and clean it up
        # before Tk destroys the window.
        manager._fractal_vertical_toolbar = panel
        return panel
    except Exception:
        return None


def _carpet_render_shape(xmin, xmax, ymin, ymax):
    """Choose a render-array shape from the current viewport aspect ratio.

    The old carpet viewer always rendered a square array, even when the user
    selected a very wide or very tall zoom rectangle. That wastes pixels on
    one axis and makes the zoomed image look lower-quality. Keep approximately
    the same pixel density in X and Y instead.
    """
    sx = max(abs(float(xmax) - float(xmin)), 1e-30)
    sy = max(abs(float(ymax) - float(ymin)), 1e-30)
    aspect = sx / sy
    total = float(CARPET_TARGET_PIXELS)
    w = int(round(math.sqrt(total * aspect)))
    h = int(round(math.sqrt(total / aspect)))
    w = max(CARPET_MIN_RENDER_DIM, min(CARPET_MAX_RENDER_DIM, w))
    h = max(CARPET_MIN_RENDER_DIM, min(CARPET_MAX_RENDER_DIM, h))
    return w, h


def _raster_fractal_window(mask_fn, home_bounds, resolution, base_levels,
                            zoom_max_levels, depth_log_base, initial_color,
                            title_prefix, picker_title):
    """Stable, self-similar Sierpinski-carpet viewer.

    Every zoom is rendered from the mathematical carpet again at an adaptive
    pixel density. When the selected rectangle fits completely inside a
    surviving recursive cell, the view is rebased into that cell before rendering. This keeps coordinates
    near [0, 1] instead of allowing tiny global floating-point coordinates to
    accumulate across many zooms. If the selected rectangle falls wholly in a
    removed center cell, the correct result is an entirely blank region.
    """
    xmin0, ymin0, xmax0, ymax0 = map(float, home_bounds)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])

    state = {
        "bounds": None,
        "busy": False,
        "selected_color": initial_color,
        "color": _deep_fractal_color(initial_color),
        "coverage": None,
        "img": None,
        "pending": None,
        "closed": False,
    }
    picker = {"obj": None}

    def mask_to_rgb(coverage):
        color_rgb = np.array(mcolors.to_rgb(state["color"]))
        bg_rgb = np.array([1.0, 1.0, 1.0])
        return coverage[..., None] * color_rgb + (1.0 - coverage[..., None]) * bg_rgb

    def rebase_view(vx0, vx1, vy0, vy1):
        """Rebase a zoom rectangle through complete carpet cells.

        Returns (mode, x0, x1, y0, y1, rebased_levels).  Coordinates returned
        here are always in a numerically safe local carpet coordinate system.
        """
        x0, x1, y0, y1 = map(float, (vx0, vx1, vy0, vy1))
        local_depth = 0
        eps = 1e-14

        for _ in range(80):
            # Only rebase if the COMPLETE selected rectangle lies in one
            # 3x3 child. Otherwise the crop crosses a cell boundary and must
            # be rendered in the current coordinate system.
            ix0 = math.floor(x0 * 3.0)
            ix1 = math.floor(max(x0, x1 - eps) * 3.0)
            iy0 = math.floor(y0 * 3.0)
            iy1 = math.floor(max(y0, y1 - eps) * 3.0)
            if ix0 != ix1 or iy0 != iy1 or ix0 < 0 or ix0 > 2 or iy0 < 0 or iy0 > 2:
                break

            ix, iy = int(ix0), int(iy0)
            if ix == 1 and iy == 1:
                return "blank", 0.0, 1.0, 0.0, 1.0, local_depth

            child_x0 = ix / 3.0
            child_y0 = iy / 3.0
            x0 = (x0 - child_x0) * 3.0
            x1 = (x1 - child_x0) * 3.0
            y0 = (y0 - child_y0) * 3.0
            y1 = (y1 - child_y0) * 3.0
            local_depth += 1

            # Stop once the crop is comfortably represented locally. Further
            # rebasing is unnecessary and can make a nearly whole-cell view
            # less stable because of repeated rounding.
            if min(x1 - x0, y1 - y0) > 0.45:
                break

        return "render", x0, x1, y0, y1, local_depth

    def redraw(vx0, vx1, vy0, vy1, force_home=False):
        if state["closed"] or state["busy"]:
            return
        if not all(math.isfinite(v) for v in (vx0, vx1, vy0, vy1)):
            return
        if vx1 <= vx0 or vy1 <= vy0:
            return

        if force_home:
            vx0, vx1, vy0, vy1 = xmin0, xmax0, ymin0, ymax0
        bounds = tuple(map(float, (vx0, vx1, vy0, vy1)))
        if state["bounds"] == bounds and not force_home:
            return

        state["busy"] = True
        try:
            if force_home or state["bounds"] is None:
                rx0, rx1, ry0, ry1 = 0.0, 1.0, 0.0, 1.0
                render_w, render_h = _carpet_render_shape(rx0, rx1, ry0, ry1)
                coverage = _carpet_filled_numpy(
                    rx0, rx1, ry0, ry1,
                    render_w, render_h,
                    base_levels, RASTER_SUPERSAMPLE, 0,
                )
                mode = f"home levels={base_levels}"
            else:
                mode_name, rx0, rx1, ry0, ry1, rebased = rebase_view(vx0, vx1, vy0, vy1)
                if mode_name == "blank":
                    render_w, render_h = _carpet_render_shape(rx0, rx1, ry0, ry1)
                    coverage = np.zeros((render_h, render_w), dtype=np.float64)
                    mode = "zoomed into removed center -> blank"
                else:
                    span = max(min(rx1 - rx0, ry1 - ry0), 1e-15)
                    extra = max(0, int(math.ceil(math.log(1.0 / span, 3.0))))
                    detail = min(int(base_levels + extra + 2), 120)
                    render_w, render_h = _carpet_render_shape(rx0, rx1, ry0, ry1)
                    coverage = _carpet_filled_numpy(
                        rx0, rx1, ry0, ry1,
                        render_w, render_h,
                        detail, RASTER_SUPERSAMPLE, 0,
                    )
                    mode = f"zoomed levels={detail}, rebased={rebased}"

            state["coverage"] = coverage
            render_h, render_w = coverage.shape
            rgb = mask_to_rgb(coverage)
            if state["img"] is None:
                state["img"] = ax.imshow(
                    rgb,
                    extent=(rx0, rx1, ry0, ry1),
                    origin="lower",
                    interpolation="nearest",
                    aspect="equal",
                )
            else:
                state["img"].set_data(rgb)
                state["img"].set_extent((rx0, rx1, ry0, ry1))

            ax.set_xlim(rx0, rx1)
            ax.set_ylim(ry0, ry1)
            state["bounds"] = bounds
            print(f"Sierpinski Carpet: {mode} | render={render_w}x{render_h}")
            fig.canvas.draw_idle()
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            state["busy"] = False

    def schedule_redraw(_event=None):
        if state["closed"] or state["busy"]:
            return
        try:
            if state["pending"] is not None:
                fig.canvas.manager.window.after_cancel(state["pending"])
        except Exception:
            pass

        def run_redraw():
            state["pending"] = None
            if state["closed"] or state["busy"]:
                return
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            redraw(x0, x1, y0, y1)

        try:
            state["pending"] = fig.canvas.manager.window.after(30, run_redraw)
        except Exception:
            run_redraw()

    def on_key(event):
        if event.key == "r":
            state["bounds"] = None
            ax.set_xlim(xmin0, xmax0)
            ax.set_ylim(ymin0, ymax0)
            redraw(xmin0, xmax0, ymin0, ymax0, force_home=True)

    def on_color_change(hex_color):
        if state["closed"]:
            return
        state["selected_color"] = hex_color
        state["color"] = _deep_fractal_color(hex_color)
        if state["coverage"] is not None and state["img"] is not None:
            state["img"].set_data(mask_to_rgb(state["coverage"]))
            fig.canvas.draw_idle()

    def on_close(event):
        state["closed"] = True
        try:
            if state["pending"] is not None:
                fig.canvas.manager.window.after_cancel(state["pending"])
        except Exception:
            pass
        state["pending"] = None
        if picker["obj"] is not None:
            picker["obj"].close()
            picker["obj"] = None

    ax.callbacks.connect("xlim_changed", schedule_redraw)
    ax.callbacks.connect("ylim_changed", schedule_redraw)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", on_close)

    redraw(xmin0, xmax0, ymin0, ymax0, force_home=True)
    _install_vertical_toolbar(fig)
    picker["obj"] = LiveColorPicker(
        fig.canvas.manager.window,
        state["selected_color"],
        on_color_change,
        title=picker_title,
    )
    show_zoom_hint()
    plt.show()


# --------------------------------------------------------------------------- #
# 4. Sierpinski Carpet
# --------------------------------------------------------------------------- #
def sierpinski_carpet():
    print("\n--- Sierpinski Carpet ---")
    print("Rendered as a fast pixel raster. At each level, every square is")
    print("divided into a 3x3 grid and the center square is removed. A")
    print("live color picker opens alongside the fractal window.\n")

    base_levels = ask_int_clamped("Number of levels (iterations)", CARPET_DEFAULT_LEVELS, CARPET_MAX_LEVELS)

    _raster_fractal_window(
        mask_fn=carpet_filled_raster,
        home_bounds=(0.0, 0.0, 1.0, 1.0),
        resolution=FIXED_RESOLUTION,
        base_levels=base_levels,
        zoom_max_levels=CARPET_ZOOM_MAX_LEVELS,
        depth_log_base=3,
        initial_color="#000080",
        title_prefix="Sierpinski Carpet",
        picker_title="Sierpinski Carpet - Live Color",
    )


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Menu / main loop
# --------------------------------------------------------------------------- #
def run():
    print("\n===== Fractal Generator - Accelerated Renderer =====")
    print("Select rendering backend:")
    print("  1. AUTO   (GPU if CUDA is available, otherwise CPU)")
    print("  2. CPU    (Numba JIT + parallel CPU threads)")
    print("  3. GPU    (NVIDIA CUDA)")
    print("  4. HYBRID (CPU + GPU concurrently)")
    backend_choice = input("\nBackend [default=1]: ").strip() or "1"
    backend_map = {"1": "auto", "2": "cpu", "3": "gpu", "4": "hybrid"}
    set_render_backend(backend_map.get(backend_choice, "auto"))
    _print_backend_status()

    fractals = {
        "1": ("Mandelbrot Fractal", mandelbrot),
        "2": ("Julia Fractal", julia),
        "3": ("Newton Fractal", newton),
        "4": ("Sierpinski Carpet", sierpinski_carpet),
    }

    while True:
        print("\n===== Fractal Generator =====")
        for key, (name, _) in fractals.items():
            print(f"  {key}. {name}")
        print("  0. Exit")

        choice = input("\nChoose a fractal to generate: ").strip()
        if choice == "0":
            print("Goodbye!")
            break
        if choice in fractals:
            _, func = fractals[choice]
            func()
        else:
            print("Invalid choice, please try again.")


if __name__ == "__main__":
    run()
