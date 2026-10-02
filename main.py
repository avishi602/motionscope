"""Command-line interface for MotionScope.

Examples:
    python main.py demo
    python main.py generate --out data/demo.mp4
    python main.py run --input data/demo.mp4 --out outputs
"""
import argparse
import sys

from motionscope.config import Config, ConfigError
from motionscope.logger import get_logger
from motionscope.pipeline import run_pipeline
from motionscope.synthetic import generate_demo_video
from motionscope.video_io import VideoError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="motionscope",
        description="Motion detection, optical flow and object tracking.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show debug messages")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="create a synthetic demo video")
    gen.add_argument("--out", default="data/demo.mp4")
    gen.add_argument("--frames", type=int, default=150)

    run = sub.add_parser("run", help="analyse a video file")
    run.add_argument("--input", required=True, help="path to input video")
    run.add_argument("--out", default="outputs", help="output folder")
    run.add_argument("--min-area", type=int, default=500)
    run.add_argument("--max-distance", type=float, default=80.0)
    run.add_argument("--max-disappeared", type=int, default=10)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--max-frames", type=int, default=0)
    run.add_argument("--no-flow-video", action="store_true",
                     help="skip writing the optical-flow video (faster)")

    demo = sub.add_parser("demo", help="generate demo video and analyse it")
    demo.add_argument("--out", default="outputs")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger(log_file="outputs/motionscope.log"
                     if args.command != "generate" else None,
                     verbose=args.verbose)
    try:
        if args.command == "generate":
            path = generate_demo_video(args.out, frames=args.frames)
            log.info("Demo video written to %s", path)
            return 0

        if args.command == "demo":
            video = generate_demo_video("data/demo.mp4")
            cfg = Config()
            input_path, out_dir = video, args.out
        else:  # run
            cfg = Config(min_area=args.min_area,
                         max_distance=args.max_distance,
                         max_disappeared=args.max_disappeared,
                         warmup_frames=args.warmup,
                         max_frames=args.max_frames,
                         save_flow_video=not args.no_flow_video)
            input_path, out_dir = args.input, args.out

        result = run_pipeline(input_path, out_dir, cfg,
                              log_file=f"{out_dir}/motionscope.log",
                              verbose=args.verbose)
        s = result["summary"]
        print("\n=== RESULT ===")
        print(f"Frames processed : {s['frames_processed']}")
        print(f"Unique objects   : {s['unique_objects']}")
        print(f"Peak at once     : {s['peak_simultaneous_objects']}")
        for name, path in result["files"].items():
            print(f"  {name:16s} -> {path}")
        return 0
    except (VideoError, ConfigError) as err:
        log.error("%s", err)
        return 2
    except Exception as err:  # last-resort guard: never show a raw traceback
        log.error("Unexpected failure: %s", err)
        return 1


if __name__ == "__main__":
    sys.exit(main())
