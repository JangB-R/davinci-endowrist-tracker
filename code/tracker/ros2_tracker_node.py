r"""
ros2_tracker_node.py -- ROS2 publishing wrapper around the tracker. SKELETON.

Cannot run on the Windows dev machine (no ROS2). Finish ON-SITE after
resolving two CLAUDE.md open items:
  1. ROS2 distro (check README/package.xml in
     abanmerali/daVinci-EndoWrist-Instrument-Control-System).
  2. The repo's existing message/topic conventions -- adopt those instead of
     the placeholders below if they differ.

Placeholder interface (frozen field set, per CLAUDE.md section 9 -- angles +
timestamp + per-angle confidence; theta_R present but confidence 0.0 in v1):
  /endowrist/joint_states   sensor_msgs/JointState
      name:     [theta_W, theta_G1, theta_G2, theta_R]
      position: radians (NaN when unmeasured)   header.stamp: capture time
  /endowrist/tracker_state  std_msgs/String (JSON)
      {t, angles_deg, confidence, warnings, fps} -- full diagnostics, and the
      stream the optional scoring system should consume.

Run (on-site, after `pip install ultralytics` in the ROS2 python env):
  ros2 run <pkg> tracker_node  -- or directly:
  python3 ros2_tracker_node.py --model best.pt --source 0 \
      --calib camera_calibration.npz --zero-offsets zero_offsets.json
Accepts the same arguments as run_tracker.py (parser is reused).
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import cv2

from mask_geometry import JOINT_NAMES, compute_frame
from run_tracker import load_undistort, main as _run_tracker_main  # noqa: F401
from run_tracker import masks_from_result, open_source

NAN = float("nan")


class TrackerNode(Node):
    def __init__(self, args):
        super().__init__("endowrist_tracker")
        self.args = args
        self.pub_js = self.create_publisher(JointState, "/endowrist/joint_states", 10)
        self.pub_state = self.create_publisher(String, "/endowrist/tracker_state", 10)

        from ultralytics import YOLO
        self.model = YOLO(args.model)
        self.zero_offsets = {}
        if args.zero_offsets:
            with open(args.zero_offsets) as fh:
                self.zero_offsets = json.load(fh)
        self.maps = None
        self.fps = 0.0
        self.t_prev = time.time()

    def spin_frames(self):
        for _idx, (frame, _live) in enumerate(open_source(self.args.source)):
            if not rclpy.ok():
                break
            if self.args.calib:
                if self.maps is None:
                    self.maps = load_undistort(self.args.calib, frame.shape)
                frame = cv2.remap(frame, self.maps[0], self.maps[1],
                                  cv2.INTER_LINEAR)
            result = self.model(frame, imgsz=self.args.imgsz, verbose=False)[0]
            masks, det_confs = masks_from_result(result, frame.shape,
                                                 self.args.min_conf)
            fr = compute_frame(masks, det_confs=det_confs,
                               flip_sign=self.args.flip_sign,
                               zero_offsets=self.zero_offsets)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self.t_prev, 1e-6)
            self.t_prev = now
            self.publish(fr, now)
            rclpy.spin_once(self, timeout_sec=0.0)

    def publish(self, fr, t):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(JOINT_NAMES)
        js.position = [math.radians(fr.angles[j]) if fr.angles.get(j) is not None
                       else NAN for j in JOINT_NAMES]
        self.pub_js.publish(js)

        st = String()
        st.data = json.dumps({
            "t": round(t, 3),
            "angles_deg": {j: fr.angles.get(j) for j in JOINT_NAMES},
            "confidence": {j: fr.confidence.get(j, 0.0) for j in JOINT_NAMES},
            "warnings": fr.warnings,
            "fps": round(self.fps, 1),
        })
        self.pub_state.publish(st)


def main():
    # Reuse run_tracker's argparse by importing its parser setup would drag in
    # display args; keep it simple and share the flags that matter.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-conf", type=float, default=0.4)
    ap.add_argument("--flip-sign", action="store_true")
    ap.add_argument("--zero-offsets", default=None)
    args = ap.parse_args()

    rclpy.init()
    node = TrackerNode(args)
    try:
        node.spin_frames()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
