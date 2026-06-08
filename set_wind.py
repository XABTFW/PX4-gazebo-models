#!/usr/bin/env python3
"""Set the Gazebo wind vector for a running gz world."""

import argparse
import math
import shlex
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish a gz.msgs.Wind message to /world/<world>/wind/."
    )
    parser.add_argument(
        "vector",
        nargs="*",
        type=float,
        metavar="V",
        help="Wind linear velocity vector in world ENU frame, in m/s.",
    )
    parser.add_argument(
        "-w",
        "--world",
        help="Gazebo world name. Default: auto-detect the running world.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        help="Horizontal wind speed in m/s. Used with --direction-deg.",
    )
    parser.add_argument(
        "--direction-deg",
        type=float,
        default=0.0,
        help="Horizontal direction in degrees from +X toward +Y. Default: 0.",
    )
    parser.add_argument(
        "--vertical",
        type=float,
        default=0.0,
        help="Vertical wind speed in m/s when using --speed. Default: 0.",
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable wind in the running world.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Publish the message this many times. Default: 1.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between repeated publishes. Default: 0.2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gz command without running it.",
    )

    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    if args.interval < 0:
        parser.error("--interval must be nonnegative")

    if args.disable:
        if args.vector or args.speed is not None:
            parser.error("--disable cannot be combined with wind velocity inputs")
        return args

    if args.vector and args.speed is not None:
        parser.error("Use either X Y Z or --speed, not both")

    if args.vector:
        if len(args.vector) != 3:
            parser.error("Provide exactly three vector values: X Y Z")
        return args

    if args.speed is None:
        parser.error("Provide X Y Z, --speed, or --disable")

    return args


def wind_vector(args):
    if args.disable:
        return 0.0, 0.0, 0.0

    if args.vector:
        return tuple(args.vector)

    direction_rad = math.radians(args.direction_deg)
    return (
        clean_zero(args.speed * math.cos(direction_rad)),
        clean_zero(args.speed * math.sin(direction_rad)),
        args.vertical,
    )


def clean_zero(value):
    return 0.0 if abs(value) < 1e-12 else value


def gz_command(world, x, y, z, enable_wind):
    topic = f"/world/{world}/wind/"
    payload = (
        f"linear_velocity: {{x: {x:.6g}, y: {y:.6g}, z: {z:.6g}}}, "
        f"enable_wind: {'true' if enable_wind else 'false'}"
    )
    return [
        "gz",
        "topic",
        "-t",
        topic,
        "-m",
        "gz.msgs.Wind",
        "-p",
        payload,
    ]


def detect_world():
    completed = subprocess.run(
        ["gz", "topic", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return "windy"

    worlds = []

    for line in completed.stdout.splitlines():
        if not line.startswith("/world/") or not line.endswith("/clock"):
            continue

        world = line.removeprefix("/world/").removesuffix("/clock")

        if world and world not in worlds:
            worlds.append(world)

    if len(worlds) == 1:
        return worlds[0]

    if len(worlds) > 1:
        print(
            "Multiple Gazebo worlds are running. Please choose one with --world.",
            file=sys.stderr,
        )
        return None

    return "windy"


def main():
    args = parse_args()
    x, y, z = wind_vector(args)
    world = args.world or detect_world()

    if world is None:
        return 2

    command = gz_command(world, x, y, z, not args.disable)

    if args.dry_run:
        print(shlex.join(command))
        return 0

    for publish_index in range(args.repeat):
        completed = subprocess.run(command, check=False)

        if completed.returncode != 0:
            return completed.returncode

        if publish_index + 1 < args.repeat:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
