"""Draw the design diagrams (black & white, print friendly) into docs/.

Run:  python docs/make_diagrams.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle

OUT = Path(__file__).parent
FONT = dict(family="DejaVu Sans", size=9, color="black")


def canvas(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fill="white", lw=1.5, style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill,
                                ec="black", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", **FONT)


def arrow(ax, x1, y1, x2, y2, text="", dashed=False, tx=0, ty=0.08):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", lw=1.3, color="black",
                                linestyle="--" if dashed else "-"))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center",
                fontsize=8)


def save(fig, name):
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def architecture():
    fig, ax = canvas(10, 6.2)
    box(ax, 0.3, 4.9, 2.2, 0.8, "Input\nVideo file (.mp4/.avi)\nor synthetic demo", "#e8e8e8")
    box(ax, 3.2, 4.9, 3.6, 0.8, "CLI  (main.py)\nargparse: generate | run | demo", "#e8e8e8")
    box(ax, 7.5, 4.9, 2.2, 0.8, "Config\n(validated parameters)", "#e8e8e8")
    box(ax, 0.3, 3.5, 9.4, 0.8, "Pipeline  (pipeline.py)  - frame loop & orchestration", "#cfcfcf")
    box(ax, 0.3, 1.9, 2.2, 1.0, "Module 1\nMotionDetector\nMOG2 background\nsubtraction", "white")
    box(ax, 2.85, 1.9, 2.2, 1.0, "Module 2\nFlowEstimator\nFarneback\noptical flow", "white")
    box(ax, 5.4, 1.9, 2.2, 1.0, "Module 3\nCentroidTracker\nID assignment\nover frames", "white")
    box(ax, 7.95, 1.9, 1.8, 1.0, "Module 4\nAnalytics\nheatmap, CSV,\nJSON, charts", "white")
    box(ax, 0.3, 0.2, 3.0, 0.8, "Support: video_io, logger,\nsynthetic (demo generator)", "#e8e8e8")
    box(ax, 4.0, 0.2, 5.7, 0.8, "Outputs: annotated.mp4, optical_flow.mp4, tracks.csv,\nsummary.json, heatmap.png, timeline.png, motionscope.log", "#e8e8e8")
    arrow(ax, 2.5, 5.3, 3.2, 5.3)
    arrow(ax, 7.5, 5.3, 6.8, 5.3)
    arrow(ax, 5.0, 4.9, 5.0, 4.3)
    for x in (1.4, 3.95, 6.5, 8.85):
        arrow(ax, x if x < 8 else 8.85, 3.5, x, 2.9)
    arrow(ax, 6.5, 1.9, 6.5, 1.0, dashed=True)
    arrow(ax, 8.85, 1.9, 8.0, 1.0)
    ax.set_title("Figure 1: System architecture", fontsize=11, weight="bold")
    save(fig, "architecture.png")


def workflow():
    fig, ax = canvas(7, 9)
    steps = [
        (8.2, "Start: user runs CLI command"),
        (7.2, "Validate arguments and Config"),
        (6.2, "Open input video  (error -> log + exit)"),
        (5.2, "Read next frame"),
        (4.2, "Detect motion (MOG2 mask + boxes)\nCompute optical flow (speed, direction)"),
        (3.0, "Update tracker (assign / create / drop IDs)"),
        (2.0, "Draw annotations, write video frames,\nrecord analytics"),
    ]
    for y, t in steps:
        box(ax, 1.2, y, 4.6, 0.7, t, "white" if "Start" not in t else "#d9d9d9")
    for (y1, _), (y2, _) in zip(steps, steps[1:]):
        arrow(ax, 3.5, y1, 3.5, y2 + 0.7)
    # decision
    ax.add_patch(plt.Polygon([[3.5, 1.5], [5.2, 0.9], [3.5, 0.3], [1.8, 0.9]],
                             fc="white", ec="black", lw=1.5))
    ax.text(3.5, 0.9, "More frames?", ha="center", va="center", **FONT)
    arrow(ax, 3.5, 2.0, 3.5, 1.5)
    ax.plot([5.2, 6.4, 6.4], [0.9, 0.9, 5.55], "k", lw=1.3)
    ax.annotate("", xy=(5.8, 5.55), xytext=(6.4, 5.55),
                arrowprops=dict(arrowstyle="->", lw=1.3))
    ax.text(6.55, 3.0, "Yes", fontsize=9)
    box(ax, 0.2, -0.9, 6.6, 0.7, "No: save CSV, JSON, heatmap, timeline; print summary; End", "#d9d9d9")
    arrow(ax, 3.5, 0.3, 3.5, -0.2)
    ax.set_ylim(-1.1, 9.0)
    ax.set_title("Figure 2: Processing workflow", fontsize=11, weight="bold")
    save(fig, "workflow.png")


def usecase():
    fig, ax = canvas(9, 5.6)
    ax.add_patch(Rectangle((2.6, 0.3), 4.6, 5.0, fc="white", ec="black", lw=1.5))
    ax.text(4.9, 5.05, "MotionScope system", ha="center", weight="bold", **{k: v for k, v in FONT.items() if k != "color"})
    cases = [(4.9, 4.2, "Generate demo video"), (4.9, 3.3, "Detect motion in video"),
             (4.9, 2.4, "Estimate optical flow"), (4.9, 1.5, "Track objects (IDs)"),
             (4.9, 0.7, "View reports & heatmap")]
    for x, y, t in cases:
        ax.add_patch(Ellipse((x, y), 3.4, 0.7, fc="white", ec="black", lw=1.3))
        ax.text(x, y, t, ha="center", va="center", **FONT)
        arrow(ax, 1.3, 2.6, x - 1.7, y)
    # actor
    ax.add_patch(plt.Circle((0.8, 3.1), 0.2, fc="white", ec="black", lw=1.5))
    ax.plot([0.8, 0.8], [2.9, 2.2], "k", lw=1.5)
    ax.plot([0.4, 1.2], [2.7, 2.7], "k", lw=1.5)
    ax.plot([0.8, 0.5], [2.2, 1.7], "k", lw=1.5)
    ax.plot([0.8, 1.1], [2.2, 1.7], "k", lw=1.5)
    ax.text(0.8, 1.4, "User\n(student / analyst)", ha="center", **FONT)
    ax.set_title("Figure 3: Use case diagram", fontsize=11, weight="bold")
    save(fig, "usecase.png")


def classes():
    fig, ax = canvas(10, 6.6)

    def cls(x, y, w, name, attrs, methods):
        h = 0.34 * (len(attrs) + len(methods)) + 0.75
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=1.5))
        ax.plot([x, x + w], [y + h - 0.4, y + h - 0.4], "k", lw=1.2)
        ax.text(x + w / 2, y + h - 0.2, name, ha="center", va="center", weight="bold", fontsize=9)
        ya = y + h - 0.65
        for a in attrs:
            ax.text(x + 0.08, ya, a, va="center", fontsize=7.5)
            ya -= 0.34
        ya -= 0.05
        ax.plot([x, x + w], [ya + 0.17, ya + 0.17], "k", lw=0.8)
        ya -= 0.12
        for m in methods:
            ax.text(x + 0.08, ya, m, va="center", fontsize=7.5)
            ya -= 0.34
        return h

    cls(0.2, 4.0, 2.6, "Config", ["history, var_threshold", "min_area, min_hits", "max_distance ..."], ["validate()"])
    cls(3.3, 4.0, 3.0, "MotionDetector", ["min_area", "_subtractor (MOG2)"], ["apply(frame) -> mask, boxes"])
    cls(6.9, 4.0, 3.0, "FlowEstimator", ["scale", "_prev_gray"], ["update(frame) -> flow"])
    cls(0.2, 0.3, 3.3, "CentroidTracker", ["centroids, boxes, hits", "trails, missing"], ["update(boxes) -> tracks", "_register(box)", "_deregister(id)"])
    cls(4.0, 0.3, 2.9, "Analytics", ["heat, rows", "frame_stats"], ["add_frame(...)", "save_all(out_dir)"])
    cls(7.4, 0.3, 2.5, "Pipeline\n(run_pipeline)", [], ["run_pipeline(...)"])
    arrow(ax, 8.65, 1.75, 8.65, 4.0)  # pipeline uses flow
    arrow(ax, 7.4, 1.5, 6.9, 1.5)
    arrow(ax, 7.9, 1.75, 4.9, 4.0, dashed=True)
    arrow(ax, 7.4, 1.0, 3.5, 1.0, dashed=True)
    arrow(ax, 8.2, 1.75, 2.8, 4.7, dashed=True)
    ax.text(5.0, 6.45, "Pipeline creates and uses all components (arrows = uses)", ha="center", fontsize=8)
    ax.set_title("Figure 4: Class / component diagram", fontsize=11, weight="bold", y=1.02)
    save(fig, "class_diagram.png")


def sequence():
    fig, ax = canvas(10, 6.2)
    names = ["User/CLI", "Pipeline", "Detector", "FlowEstimator", "Tracker", "Analytics"]
    xs = [0.9, 2.6, 4.3, 6.0, 7.7, 9.2]
    for x, n in zip(xs, names):
        box(ax, x - 0.7, 5.5, 1.4, 0.45, n)
        ax.plot([x, x], [5.5, 0.3], "k", lw=0.9, linestyle="--")
    msgs = [(0, 1, 5.1, "run(input, config)"), (1, 2, 4.6, "apply(frame)"),
            (2, 1, 4.25, "mask, boxes"), (1, 3, 3.8, "update(frame)"),
            (3, 1, 3.45, "flow"), (1, 4, 3.0, "update(boxes)"),
            (4, 1, 2.65, "tracks"), (1, 5, 2.2, "add_frame(mask, tracks, flow)"),
            (1, 1, 1.7, "write annotated frame (loop)"), (1, 5, 1.2, "save_all()"),
            (1, 0, 0.7, "result summary")]
    for a, b, y, t in msgs:
        if a == b:
            ax.annotate("", xy=(xs[a] + 0.05, y - 0.2), xytext=(xs[a] + 0.05, y),
                        arrowprops=dict(arrowstyle="->", lw=1.2, connectionstyle="arc3,rad=-1.2"))
            ax.text(xs[a] + 0.5, y - 0.1, t, fontsize=8)
        else:
            arrow(ax, xs[a], y, xs[b], y, t, ty=0.06)
    ax.set_title("Figure 5: Sequence diagram (one frame of the loop)", fontsize=11, weight="bold")
    save(fig, "sequence.png")


if __name__ == "__main__":
    for fn in (architecture, workflow, usecase, classes, sequence):
        fn()
        print("wrote", fn.__name__)
