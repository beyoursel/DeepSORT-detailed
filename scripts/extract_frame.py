#!/usr/bin/env python3
"""
从视频中抽取一帧，保存为图片。

用法：
    python3 scripts/extract_frame.py --input ./test_video.avi --output ./test_frame.jpg --time 1.0
    python3 scripts/extract_frame.py --input ./test_video.avi --output ./test_frame.jpg --percent 0.5

参数：
    --input    输入视频路径
    --output   输出图片路径
    --time     抽取第几秒（浮点数，与 --percent 二选一）
    --percent  按视频总时长百分比抽取（0.0 ~ 1.0，与 --time 二选一）
"""

import argparse
import sys

try:
    import cv2
except ImportError:
    print("Error: opencv-python is not installed. Please run: pip install opencv-python")
    sys.exit(1)


def extract_frame(input_path: str, output_path: str, time_sec: float = None, percent: float = None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open video {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("Error: video has no frames")
        sys.exit(1)

    if percent is not None:
        target_frame = int(total_frames * percent)
    elif time_sec is not None:
        target_frame = int(time_sec * fps)
    else:
        target_frame = 0

    target_frame = max(0, min(target_frame, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"Error: failed to read frame at index {target_frame}")
        sys.exit(1)

    cv2.imwrite(output_path, frame)
    print(f"Saved frame {target_frame} ({target_frame / fps:.2f}s) to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract a frame from a video")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output image path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--time", type=float, help="Timestamp in seconds")
    group.add_argument("--percent", type=float, help="Position as ratio of total duration (0.0 ~ 1.0)")
    args = parser.parse_args()

    extract_frame(args.input, args.output, args.time, args.percent)


if __name__ == "__main__":
    main()
