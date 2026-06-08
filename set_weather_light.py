#!/usr/bin/env python3
"""Change GZ Sim sunlight presets during a running simulation."""

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LightPreset:
    intensity: float
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float, float]
    direction: tuple[float, float, float]
    shadows: bool = True
    off: bool = False


PRESETS = {
    "sunny": LightPreset(
        intensity=1.25,
        diffuse=(1.0, 0.96, 0.86, 1.0),
        specular=(0.45, 0.42, 0.35, 1.0),
        direction=(0.001, 0.625, -0.78),
    ),
    "cloudy": LightPreset(
        intensity=0.55,
        diffuse=(0.62, 0.66, 0.70, 1.0),
        specular=(0.08, 0.08, 0.08, 1.0),
        direction=(0.001, 0.625, -0.78),
        shadows=False,
    ),
    "overcast": LightPreset(
        intensity=0.30,
        diffuse=(0.42, 0.45, 0.48, 1.0),
        specular=(0.02, 0.02, 0.02, 1.0),
        direction=(0.001, 0.625, -0.78),
        shadows=False,
    ),
    "dusk": LightPreset(
        intensity=0.35,
        diffuse=(0.95, 0.48, 0.24, 1.0),
        specular=(0.18, 0.10, 0.05, 1.0),
        direction=(-0.35, 0.65, -0.20),
    ),
    "night": LightPreset(
        intensity=0.04,
        diffuse=(0.05, 0.06, 0.10, 1.0),
        specular=(0.0, 0.0, 0.0, 1.0),
        direction=(0.001, 0.625, -0.78),
        shadows=False,
    ),
    "off": LightPreset(
        intensity=0.0,
        diffuse=(0.0, 0.0, 0.0, 1.0),
        specular=(0.0, 0.0, 0.0, 1.0),
        direction=(0.001, 0.625, -0.78),
        shadows=False,
        off=True,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Set or cycle GZ Sim directional light presets. This affects "
            "Gazebo camera images observed through ROS bridges."
        )
    )
    parser.add_argument(
        "weather",
        nargs="?",
        choices=sorted(PRESETS),
        help="Weather preset to apply once.",
    )
    parser.add_argument(
        "--cycle",
        nargs="+",
        choices=sorted(PRESETS),
        help="Cycle through presets forever, for example: --cycle sunny cloudy overcast",
    )
    parser.add_argument(
        "-w",
        "--world",
        help="Gazebo world name. Default: auto-detect the running world.",
    )
    parser.add_argument(
        "--light",
        default="sunUTC",
        help="Directional light name in the world. Default: sunUTC.",
    )
    parser.add_argument(
        "--gz-bin",
        default="gz",
        help="Gazebo CLI executable. Default: gz.",
    )
    parser.add_argument(
        "--transition",
        type=float,
        default=0.0,
        help="Seconds used to fade between presets. Default: 0.",
    )
    parser.add_argument(
        "--hold",
        type=float,
        default=10.0,
        help="Seconds to hold each preset when using --cycle. Default: 10.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Number of preset changes when using --cycle; 0 runs forever. Default: 0.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=10.0,
        help="Light update rate during transitions, in Hz. Default: 10.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="Gazebo service timeout in milliseconds. Default: 2000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gz commands without running them.",
    )

    args = parser.parse_args()

    if bool(args.weather) == bool(args.cycle):
        parser.error("Provide either one weather preset or --cycle PRESET ...")

    if args.transition < 0:
        parser.error("--transition must be nonnegative")

    if args.hold < 0:
        parser.error("--hold must be nonnegative")

    if args.cycles < 0:
        parser.error("--cycles must be nonnegative")

    if args.rate <= 0:
        parser.error("--rate must be positive")

    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    return args


def detect_world(gz_bin):
    completed = subprocess.run(
        [gz_bin, "topic", "-l"],
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        return "default"

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
        print("Multiple Gazebo worlds are running. Please choose one with --world.", file=sys.stderr)
        return None

    return "default"


def color_field(name, rgba):
    r, g, b, a = rgba
    return f"{name}: {{r: {r:.6g}, g: {g:.6g}, b: {b:.6g}, a: {a:.6g}}}"


def vector_field(name, xyz):
    x, y, z = xyz
    return f"{name}: {{x: {x:.6g}, y: {y:.6g}, z: {z:.6g}}}"


def light_payload(light_name, preset):
    return (
        f'name: "{light_name}", '
        "type: DIRECTIONAL, "
        f"{color_field('diffuse', preset.diffuse)}, "
        f"{color_field('specular', preset.specular)}, "
        f"{vector_field('direction', preset.direction)}, "
        "range: 2000, "
        "attenuation_constant: 1, "
        "attenuation_linear: 0, "
        "attenuation_quadratic: 0, "
        f"cast_shadows: {'true' if preset.shadows else 'false'}, "
        f"intensity: {preset.intensity:.6g}, "
        f"is_light_off: {'true' if preset.off else 'false'}"
    )


def gz_command(gz_bin, world, light_name, preset, timeout):
    return [
        gz_bin,
        "service",
        "-s",
        f"/world/{world}/light_config",
        "--reqtype",
        "gz.msgs.Light",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        str(timeout),
        "--req",
        light_payload(light_name, preset),
    ]


def run_command(command, dry_run):
    if dry_run:
        print(shlex.join(command), flush=True)
        return 0

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        print(f"Gazebo light update failed: {message}", file=sys.stderr)
        return completed.returncode

    if "data: false" in completed.stdout:
        print("Gazebo rejected the light update. Check --world and --light.", file=sys.stderr)
        return 1

    return 0


def blend(a, b, ratio):
    return tuple((1.0 - ratio) * x + ratio * y for x, y in zip(a, b))


def blend_bool(a, b, ratio):
    return b if ratio >= 1.0 else a


def interpolate(start, end, ratio):
    return LightPreset(
        intensity=(1.0 - ratio) * start.intensity + ratio * end.intensity,
        diffuse=blend(start.diffuse, end.diffuse, ratio),
        specular=blend(start.specular, end.specular, ratio),
        direction=blend(start.direction, end.direction, ratio),
        shadows=blend_bool(start.shadows, end.shadows, ratio),
        off=blend_bool(start.off, end.off, ratio),
    )


def apply_preset(args, world, preset):
    command = gz_command(args.gz_bin, world, args.light, preset, args.timeout)
    return run_command(command, args.dry_run)


def transition(args, world, start, end):
    if args.transition == 0:
        return apply_preset(args, world, end)

    steps = max(1, int(args.transition * args.rate))
    interval = args.transition / steps

    for step in range(1, steps + 1):
        preset = interpolate(start, end, step / steps)
        code = apply_preset(args, world, preset)

        if code != 0:
            return code

        if step < steps:
            time.sleep(interval)

    return 0


def run_cycle(args, world):
    names = args.cycle
    current = PRESETS[names[0]]

    code = apply_preset(args, world, current)
    if code != 0:
        return code

    index = 0
    changes = 0

    while True:
        if args.cycles and changes >= args.cycles:
            return 0

        if args.hold > 0:
            time.sleep(args.hold)

        next_name = names[(index + 1) % len(names)]
        next_preset = PRESETS[next_name]

        print(f"weather -> {next_name}", flush=True)
        code = transition(args, world, current, next_preset)

        if code != 0:
            return code

        current = next_preset
        index = (index + 1) % len(names)
        changes += 1


def main():
    args = parse_args()
    world = args.world or detect_world(args.gz_bin)

    if world is None:
        return 2

    if args.weather:
        return apply_preset(args, world, PRESETS[args.weather])

    return run_cycle(args, world)


if __name__ == "__main__":
    sys.exit(main())
