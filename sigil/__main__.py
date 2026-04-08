"""CLI entry-point for Sigil (§5.3–5.5).

Usage:
    sigil run          – start the daemon
    sigil record       – enter recording mode
    sigil train        – retrain models from recordings
    sigil config       – show / edit configuration
    sigil gui          – open the graphical configuration app
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GDK_BACKEND", "x11")
os.environ.setdefault("EGL_DRIVER", "sw")

# Preload to fix EGL conflicts BEFORE any GTK imports
import mediapipe
import tensorflow

import argparse
import logging
import sys

from sigil import __version__


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="sigil",
        description="Sigil – Gesture Control Framework for Hyprland",
    )
    root.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    root.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity (-v info, -vv debug)",
    )
    sub = root.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Start the Sigil daemon")
    run_p.add_argument(
        "--backend",
        choices=["gtk4-layer-shell", "gtk4", "opencv"],
        default=None,
        help="Force a specific overlay backend (default: auto-detect)",
    )

    # ── record ───────────────────────────────────────────────────────────────
    rec = sub.add_parser("record", help="Record gesture samples")
    rec.add_argument("class_name", help="Name of the gesture class")
    rec.add_argument(
        "-m",
        "--mode",
        choices=["instant", "gradual", "sequential"],
        default="instant",
        help="Recording mode (default: instant)",
    )
    rec.add_argument(
        "--hand",
        choices=["left", "right", "both"],
        default="right",
        help="Which hand to record (default: right)",
    )
    rec.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=0,
        help="Auto-stop after N samples (0 = manual stop with ESC)",
    )

    # ── train ────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="(Re)train models from recordings")
    tr.add_argument(
        "--type",
        choices=["instant", "dynamic", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    tr.add_argument("--epochs", type=int, default=30, help="Training epochs")
    tr.add_argument("--batch", type=int, default=32, help="Batch size")

    # ── config ───────────────────────────────────────────────────────────────
    cfg = sub.add_parser("config", help="Show or edit configuration")
    cfg.add_argument("--path", action="store_true", help="Print config file path and exit")
    cfg.add_argument(
        "--edit",
        action="store_true",
        help="Open config in $EDITOR",
    )

    # ── gui ──────────────────────────────────────────────────────────────────
    sub.add_parser("gui", help="Open the graphical configuration app")

    return root


def _setup_logging(verbosity: int) -> None:
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Subcommand handlers ─────────────────────────────────────────────────────


def _cmd_run(args: argparse.Namespace) -> None:
    import logging
    import os

    # Suppress MediaPipe/TensorFlow verbose logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Suppress absl logging if available
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    # Suppress MediaPipe logger specifically
    logging.getLogger("mediapipe").setLevel(logging.ERROR)

    from sigil.daemon import Daemon

    daemon = Daemon(force_backend=getattr(args, "backend", None))
    daemon.run()


def _cmd_record(args: argparse.Namespace) -> None:
    """Record samples interactively using the daemon loop."""
    import time

    from sigil.config import load_config
    from sigil.overlay import Overlay
    from sigil.recorder import Recorder, RecordMode
    from sigil.tracker import Tracker

    mode_map = {
        "instant": RecordMode.INSTANT,
        "gradual": RecordMode.GRADUAL,
        "sequential": RecordMode.SEQUENTIAL,
    }

    cfg = load_config()
    tracker = Tracker(cfg.tracking)
    recorder = Recorder(cfg.recording)
    overlay = Overlay(enabled=True, overlay_cfg=cfg.overlay_settings)
    recorder.set_status_callback(overlay.set_recording_status)

    tracker.open()
    recorder.start(args.class_name, mode_map[args.mode], hand=args.hand)
    print(f"Recording '{args.class_name}' ({args.mode}) — press ESC to stop")

    count = 0
    try:
        while True:
            frame = tracker.read()
            if frame is None:
                time.sleep(0.01)
                continue

            recorder.capture(frame)
            count += 1

            alive = overlay.draw(frame, [], fps=tracker.fps)
            if not alive:
                break

            if args.num_samples > 0 and count >= args.num_samples:
                print(f"Reached {args.num_samples} samples, stopping.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        overlay.close()
        tracker.close()
        print(f"Saved {count} samples for class '{args.class_name}'")


def _cmd_train(args: argparse.Namespace) -> None:
    from sigil.trainer import retrain, train_instant

    try:
        if args.type == "all":
            retrain()
        elif args.type == "instant":
            train_instant()
        elif args.type == "dynamic":
            print("Dynamic model training not yet implemented", file=sys.stderr)
            sys.exit(1)
    except ImportError as exc:
        print(f"Training dependency missing: {exc}", file=sys.stderr)
        print("Install with: pip install sigil[train]", file=sys.stderr)
        sys.exit(1)


def _cmd_config(args: argparse.Namespace) -> None:
    import os
    import subprocess

    from sigil.config import CONFIG_DIR

    config_path = CONFIG_DIR / "config.yaml"

    if args.path:
        print(config_path)
        return

    if args.edit:
        editor = os.environ.get("EDITOR", "nano")
        subprocess.run([editor, str(config_path)], check=False)
        return

    # Default: print current config
    if config_path.exists():
        print(config_path.read_text())
    else:
        print(f"No config found at {config_path}. Run 'sigil run' to generate defaults.")


def _cmd_gui() -> None:
    """Launch the GTK4 configuration GUI."""
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    logging.getLogger("mediapipe").setLevel(logging.ERROR)

    from sigil.ui.app import run_gui

    run_gui()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    handlers = {
        "run": lambda: _cmd_run(args),
        "record": lambda: _cmd_record(args),
        "train": lambda: _cmd_train(args),
        "config": lambda: _cmd_config(args),
        "gui": lambda: _cmd_gui(),
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0)

    handler()


if __name__ == "__main__":
    main()
