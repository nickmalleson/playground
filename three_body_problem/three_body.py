"""Three-body problem playground.

Three stars orbiting each other under Newtonian gravity, integrated with RK4.
Sliders let you fiddle with masses, simulation speed, trail length and zoom,
and a preset menu offers some classic (and some chaotic) starting conditions.

Run with:  python three_body.py
"""

import math
import random
import tkinter as tk
from tkinter import ttk
from collections import deque

G = 1.0          # gravitational constant (simulation units)
SOFTENING = 0.01  # softening length to avoid singularities at close encounters
DT = 0.001        # base integration timestep

CANVAS_SIZE = 760
STAR_COLOURS = ["#ffd54f", "#4fc3f7", "#ef5350"]   # gold, blue, red
TRAIL_COLOURS = ["#8d6e63", "#37474f", "#6a1b4d"]  # dimmer trail shades
BG_COLOUR = "#0b0e1a"


class Body:
    def __init__(self, mass, x, y, vx, vy):
        self.mass = mass
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy


def accelerations(state, masses):
    """state = [x1,y1,x2,y2,x3,y3, vx1,vy1,...]; returns derivative of state."""
    n = 3
    pos = state[: 2 * n]
    vel = state[2 * n:]
    acc = [0.0] * (2 * n)
    for i in range(n):
        xi, yi = pos[2 * i], pos[2 * i + 1]
        for j in range(n):
            if i == j:
                continue
            dx = pos[2 * j] - xi
            dy = pos[2 * j + 1] - yi
            r2 = dx * dx + dy * dy + SOFTENING * SOFTENING
            inv_r3 = 1.0 / (r2 * math.sqrt(r2))
            f = G * masses[j] * inv_r3
            acc[2 * i] += f * dx
            acc[2 * i + 1] += f * dy
    return vel + acc


def rk4_step(state, masses, dt):
    k1 = accelerations(state, masses)
    s2 = [s + 0.5 * dt * k for s, k in zip(state, k1)]
    k2 = accelerations(s2, masses)
    s3 = [s + 0.5 * dt * k for s, k in zip(state, k2)]
    k3 = accelerations(s3, masses)
    s4 = [s + dt * k for s, k in zip(state, k3)]
    k4 = accelerations(s4, masses)
    return [
        s + dt / 6.0 * (a + 2 * b + 2 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    ]


# ---------------------------------------------------------------- presets ---

def preset_figure_eight():
    """The famous Chenciner–Montgomery figure-eight choreography."""
    x, y = 0.97000436, -0.24308753
    vx, vy = -0.93240737, -0.86473146
    return [
        Body(1.0, x, y, -vx / 2, -vy / 2),
        Body(1.0, -x, -y, -vx / 2, -vy / 2),
        Body(1.0, 0.0, 0.0, vx, vy),
    ]


def preset_lagrange_triangle():
    """Equal masses on an equilateral triangle, all circling the centre."""
    r = 1.0
    m = 1.0
    # circular velocity for the equilateral three-body relative equilibrium
    v = math.sqrt(G * m / (r * math.sqrt(3)))
    bodies = []
    for k in range(3):
        ang = 2 * math.pi * k / 3 + math.pi / 2
        bodies.append(
            Body(m, r * math.cos(ang), r * math.sin(ang),
                 -v * math.sin(ang), v * math.cos(ang))
        )
    return bodies


def preset_sun_and_planets():
    """A heavy star with two lighter companions - decays into chaos nicely."""
    return [
        Body(3.0, 0.0, 0.0, 0.0, -0.12),
        Body(0.4, 1.6, 0.0, 0.0, 1.35),
        Body(0.4, -2.4, 0.0, 0.0, -1.05),
    ]


def preset_random():
    """Random positions and gentle random velocities, net momentum removed."""
    bodies = []
    for _ in range(3):
        ang = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.6, 1.6)
        bodies.append(
            Body(
                random.uniform(0.5, 2.0),
                r * math.cos(ang), r * math.sin(ang),
                random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5),
            )
        )
    # remove net momentum so the system doesn't drift off screen
    total_m = sum(b.mass for b in bodies)
    px = sum(b.mass * b.vx for b in bodies) / total_m
    py = sum(b.mass * b.vy for b in bodies) / total_m
    for b in bodies:
        b.vx -= px
        b.vy -= py
    return bodies


PRESETS = {
    "Figure eight": preset_figure_eight,
    "Lagrange triangle": preset_lagrange_triangle,
    "Sun + companions": preset_sun_and_planets,
    "Random chaos": preset_random,
}


# -------------------------------------------------------------------- app ---

class ThreeBodyApp:
    def __init__(self, root):
        self.root = root
        root.title("Three-Body Problem Playground")
        root.configure(bg="#1a1d2e")

        self.canvas = tk.Canvas(
            root, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg=BG_COLOUR, highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, padx=8, pady=8, rowspan=2)

        panel = ttk.Frame(root, padding=10)
        panel.grid(row=0, column=1, sticky="new")
        self._build_controls(panel)

        self.running = True
        self.trails = [deque() for _ in range(3)]
        self.time_elapsed = 0.0
        self.load_preset("Figure eight")
        self._tick()

    # ------------------------------------------------------------ controls

    def _build_controls(self, panel):
        row = 0
        ttk.Label(panel, text="Preset").grid(row=row, column=0, sticky="w")
        self.preset_var = tk.StringVar(value="Figure eight")
        combo = ttk.Combobox(
            panel, textvariable=self.preset_var, state="readonly",
            values=list(PRESETS.keys()), width=18,
        )
        combo.grid(row=row, column=1, sticky="ew", pady=2)
        combo.bind("<<ComboboxSelected>>",
                   lambda e: self.load_preset(self.preset_var.get()))
        row += 1

        self.mass_vars = []
        for i in range(3):
            ttk.Label(panel, text=f"Mass of star {i + 1}").grid(
                row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
            row += 1
            var = tk.DoubleVar(value=1.0)
            self.mass_vars.append(var)
            scale = ttk.Scale(panel, from_=0.1, to=4.0, variable=var,
                              command=lambda v, idx=i: self._set_mass(idx))
            scale.grid(row=row, column=0, columnspan=2, sticky="ew")
            row += 1

        ttk.Label(panel, text="Simulation speed").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
        row += 1
        self.speed_var = tk.DoubleVar(value=8.0)
        ttk.Scale(panel, from_=1.0, to=40.0, variable=self.speed_var).grid(
            row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Label(panel, text="Trail length").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
        row += 1
        self.trail_var = tk.IntVar(value=900)
        ttk.Scale(panel, from_=0, to=3000, variable=self.trail_var).grid(
            row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        ttk.Label(panel, text="Zoom").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
        row += 1
        self.zoom_var = tk.DoubleVar(value=140.0)
        ttk.Scale(panel, from_=20.0, to=350.0, variable=self.zoom_var).grid(
            row=row, column=0, columnspan=2, sticky="ew")
        row += 1

        self.follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel, text="Follow centre of mass", variable=self.follow_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(12, 0))
        row += 1

        btns = ttk.Frame(panel)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        self.pause_btn = ttk.Button(btns, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        ttk.Button(btns, text="Restart", command=self.restart).pack(
            side="left", expand=True, fill="x", padx=(4, 0))
        row += 1

        ttk.Button(panel, text="Randomise!", command=self.randomise).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        row += 1

        self.status = ttk.Label(panel, text="", wraplength=180,
                                foreground="#666")
        self.status.grid(row=row, column=0, columnspan=2,
                         sticky="w", pady=(16, 0))
        panel.columnconfigure(1, weight=1)

    # ------------------------------------------------------------- actions

    def load_preset(self, name):
        self.bodies = PRESETS[name]()
        for var, body in zip(self.mass_vars, self.bodies):
            var.set(body.mass)
        self._clear_trails()
        self.time_elapsed = 0.0

    def restart(self):
        self.load_preset(self.preset_var.get())

    def randomise(self):
        self.preset_var.set("Random chaos")
        self.load_preset("Random chaos")

    def toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    def _set_mass(self, idx):
        self.bodies[idx].mass = self.mass_vars[idx].get()

    def _clear_trails(self):
        for t in self.trails:
            t.clear()

    # ----------------------------------------------------------- main loop

    def _tick(self):
        if self.running:
            self._step_physics()
        self._draw()
        self.root.after(16, self._tick)  # ~60 fps

    def _step_physics(self):
        masses = [b.mass for b in self.bodies]
        state = []
        for b in self.bodies:
            state += [b.x, b.y]
        for b in self.bodies:
            state += [b.vx, b.vy]

        steps = int(self.speed_var.get())
        for _ in range(steps):
            state = rk4_step(state, masses, DT)
        self.time_elapsed += steps * DT

        for i, b in enumerate(self.bodies):
            b.x, b.y = state[2 * i], state[2 * i + 1]
            b.vx, b.vy = state[6 + 2 * i], state[6 + 2 * i + 1]
            self.trails[i].append((b.x, b.y))

        max_trail = int(self.trail_var.get())
        for t in self.trails:
            while len(t) > max_trail:
                t.popleft()

    def _world_to_screen(self, x, y, cx, cy, scale):
        sx = CANVAS_SIZE / 2 + (x - cx) * scale
        sy = CANVAS_SIZE / 2 - (y - cy) * scale
        return sx, sy

    def _centre(self):
        if self.follow_var.get():
            total_m = sum(b.mass for b in self.bodies)
            cx = sum(b.mass * b.x for b in self.bodies) / total_m
            cy = sum(b.mass * b.y for b in self.bodies) / total_m
            return cx, cy
        return 0.0, 0.0

    def _draw(self):
        c = self.canvas
        c.delete("all")
        scale = self.zoom_var.get()
        cx, cy = self._centre()

        # trails
        for i, trail in enumerate(self.trails):
            if len(trail) < 2:
                continue
            pts = []
            step = max(1, len(trail) // 600)  # cap points drawn per trail
            for x, y in list(trail)[::step]:
                pts += self._world_to_screen(x, y, cx, cy, scale)
            if len(pts) >= 4:
                c.create_line(*pts, fill=TRAIL_COLOURS[i], width=1)

        # stars (radius grows gently with mass)
        for i, b in enumerate(self.bodies):
            sx, sy = self._world_to_screen(b.x, b.y, cx, cy, scale)
            r = 4 + 4 * math.sqrt(b.mass)
            # soft glow
            c.create_oval(sx - r * 2, sy - r * 2, sx + r * 2, sy + r * 2,
                          fill="", outline=STAR_COLOURS[i], width=1)
            c.create_oval(sx - r, sy - r, sx + r, sy + r,
                          fill=STAR_COLOURS[i], outline="")

        self.status.config(
            text=f"t = {self.time_elapsed:7.2f}\n"
                 "Tip: nudge a mass slider mid-orbit to watch a stable "
                 "system descend into chaos.")


def main():
    root = tk.Tk()
    ThreeBodyApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
