#!/usr/bin/env python3
"""ROS 2 waypoint-following behavior for ELEC 555 Week 5.

Run inside the sourced MentorPi ROS 2 container. The robot reads its pose from
odometry and drives through a list of (x, y) waypoints with a bounded go-to-goal
controller: turn toward the next waypoint, drive to it, advance when close, stop
at the end. It optionally logs its trajectory so you can COMPARE the real run to
the virtual-twin simulation (the Week 5 notebook / robot script run the SAME
controller).

SAFETY FIRST (same rules as Weeks 2-4)
--------
* Default ``--mode report``: computes and PRINTS pose, target waypoint, and the
  command it WOULD send, but publishes ZERO velocity. Nothing moves.
* ``--mode drive`` publishes motion, every command clamped to small
  ``--max-speed`` / ``--max-turn`` limits.
* If no odometry arrives for ``--data-timeout`` seconds, the robot STOPS.
* On Ctrl+C / shutdown / error, the control timer is CANCELLED and a zero
  velocity is then published REPEATEDLY for about half a second before the node
  exits. A single last-gasp publish right before ``destroy_node()`` is not
  guaranteed to reach the chassis (the DDS writer may still be queuing it), so
  the robot could keep running its last command. After Ctrl+C, watch the wheels
  for a moment to confirm the stop.
* When the last waypoint is reached the script holds zero for a second, saves the
  trajectory log, and exits on its own -- no Ctrl+C required.

ONE SOURCE OF TRUTH: ``--tolerance`` is the only thing that decides whether a
waypoint counts as reached. If you extend this script (your own PID, your own
"slow down near the goal" radius), read that same value rather than hardcoding a
second number -- a controller that converges on one threshold while a different
threshold decides when to advance can settle just outside the circle and never
finish. Same for ``--max-speed``: it is the single cap, applied once.

Begin in report mode with the wheels off the ground, roll the robot by hand and
confirm the reported pose and target waypoint change sensibly, then drive in a
clear, open area.

WAYPOINTS are relative to WHERE THE ROBOT IS WHEN THE SCRIPT STARTS: the first
odometry message is captured as the origin, so the robot always begins at (0, 0)
facing +x, no matter what the raw odometry says. (Raw ``/odom`` is only near zero
right after the chassis driver boots -- without this, a plan run after any prior
driving would aim at wherever the driver happened to start. ``--absolute`` restores
the raw odom-frame behavior.) Keep plans small for a lab (a ~1 m loop).

Note the logged pose is what odometry BELIEVES, and odometry DRIFTS: the robot can
believe it returned to (0, 0) while its true position is off by much more. That is
why the tutorial has you tape-mark the start and measure the PHYSICAL end offset --
the tape measure, not this log, is the real sim-to-real evidence.

MentorPi defaults (per https://docs.hiwonder.com/projects/MentorPi/en/latest/):
odometry ``/odom`` (nav_msgs/Odometry); chassis ``/controller/cmd_vel`` (Twist;
+z left, -z right; <=0.6 m/s, <=2.0 rad/s). The robot's own SLAM/AMCL stack
(slam_toolbox, ch. 6-7) gives a map-frame pose; this lab uses raw odometry.
"""

import argparse
import math
import sys
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
    from sensor_msgs.msg import LaserScan, Image
except ImportError as exc:
    print("Missing a ROS module. Run inside the sourced MentorPi ROS 2 container.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


try:
    import cv2
    from cv_bridge import CvBridge

except ImportError as exc:
    print("NumPy / SciPy not installed. Install SciPy and NumPy inside the sourced MentorPi ROS2 container", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    import numpy as np
    from scipy.spatial import KDTree
except ImportError as exc:
    print("NumPy / SciPy not installed. Install SciPy and NumPy inside the sourced MentorPi ROS2 container", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

MENTORPI_MAX_SPEED = 0.6   # m/s
MENTORPI_MAX_TURN = 2.0    # rad/s


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def parse_triplet(text):
    """Parse a command-line triplet such as 35,60,40."""

    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use three integers, for example 35,60,40") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("use exactly three values separated by commas")
    return values

def canonical_space(text):
    """Return the OpenCV space name used by this script."""

    key = text.strip().lower()
    if key not in SPACE_ALIASES:
        choices = ", ".join(sorted(SPACE_ALIASES))
        raise argparse.ArgumentTypeError(f"choose one color space: {choices}")
    return SPACE_ALIASES[key]

#Color Spaces info
SPACE_INFO = {
    "hsv": {
        "labels": ("H", "S", "V"),
        "minimum": (0, 0, 0),
        "maximum": (179, 255, 255),
        "code": cv2.COLOR_BGR2HSV,
        "note": "HSV is usually the easiest first choice for colored objects.",
    },
    "rgb": {
        "labels": ("R", "G", "B"),
        "minimum": (0, 0, 0),
        "maximum": (255, 255, 255),
        "code": cv2.COLOR_BGR2RGB,
        "note": "RGB is direct, but brightness changes move all channels.",
    },
    "lab": {
        "labels": ("L", "a", "b"),
        "minimum": (0, 0, 0),
        "maximum": (255, 255, 255),
        "code": cv2.COLOR_BGR2LAB,
        "note": "Lab separates lightness from color-opponent channels.",
    },
    "ycrcb": {
        "labels": ("Y", "Cr", "Cb"),
        "minimum": (0, 0, 0),
        "maximum": (255, 255, 255),
        "code": cv2.COLOR_BGR2YCrCb,
        "note": "YCrCb separates brightness from video-style chroma channels.",
    },
}

SPACE_ALIASES = {
    "hsv": "hsv",
    "rgb": "rgb",
    "lab": "lab",
    "ycrcb": "ycrcb",
    "ycbcr": "ycrcb",
}

#Ideal PID values: kp=3.0 ki=0.0 kd=0.02
class PIDController():
    
    def __init__(self, kp: float, ki: float, kd: float, dt=0.02):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = 0.0

        self.integral = 0.0
        self.izone = 0.0
        self.int_min = -float('inf')
        self.int_max = float('inf')

        self.prev_error = 0.0

        self.tolerance = 0.0

        self.dt = dt

    def set_izone(self, izone: float):
        self.izone = izone

    def set_integrator_range(self, int_min: float, int_max: float):
            if (int_min > int_max):
                int_min, int_max = int_max, int_min

            self.int_max = int_max
            self.int_min = int_min

    def set_setpoint(self, setpoint: float):
        self.setpoint = setpoint

    def set_tolerance(self, tolerance: float):
        self.tolerance = tolerance

    def at_tolerance(self, current_state):
        return abs(current_state - self.setpoint) <= self.tolerance

    def calculate(self, current_state: float):
        error = self.setpoint - current_state
        
        if self.at_tolerance(current_state):
            self.integral = 0.0
            self.prev_error = error
            return 0.0

        if abs(error) <= self.izone:
            self.integral += error * self.dt
            self.integral = clip(self.integral, self.int_min, self.int_max)
        else:
            self.integral = 0.0

        dedt = (error - self.prev_error) / self.dt

        pid = self.kp * error + self.ki * self.integral + self.kd * dedt

        self.prev_error = error

        return pid



class SelfDrive(Node):
    def __init__(self, args):
        super().__init__("week6_self_drive")
        self.args = args
        self.stopping = False       # set by stop_robot(); blocks any further motion
        self.finished = False       # plan complete -> main loop exits cleanly

        self.frame = 0.0
        self.frames_seen = 0.0
        self.mask = 0.0

        self.lane_error_x = 0.0
        self.forward_distance = float("inf") #Forward distance detected by LiDAR

        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan, qos_profile_sensor_data)

        kp, ki, kd = self.args.lin_pid
        self.speed_controller = PIDController(kp, ki, kd, dt=1/self.args.rate)
        self.speed_controller.set_setpoint(0.0)
        self.speed_controller.set_tolerance(self.args.tolerance)

        kp, ki, kd = self.args.ang_pid
        self.turn_controller = PIDController(kp, ki, kd, dt=1/self.args.rate)
        self.turn_controller.set_setpoint(0.0)
        self.turn_controller.set_tolerance(math.radians(3))

        self.get_logger().info(
            f"MODE={args.mode.upper()}  waypoints={self.waypoints}  tol={args.tolerance:.2f} m  "
            f"max_speed={args.max_speed:.2f}  max_turn={args.max_turn:.2f}  -> {args.cmd_vel_topic}"
        )
        if args.mode == "report":
            self.get_logger().info("REPORT mode: computing only, publishing ZERO velocity. Nothing moves.")
        else:
            self.get_logger().warn("DRIVE mode: the robot WILL move. Keep the area clear; hand on Ctrl+C.")

        self.timer = self.create_timer(1.0 / args.rate, self.control_step)

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg,desired_encoding="bgr8")

        self.frames_seen += 1
        converted = cv2.cvtColor(frame, SPACE_INFO[self.args.space]["code"])
        mask = self.make_mask(converted)
        if self.args.kernel_size > 1:
            kernel = np.ones((self.args.kernel_size, self.args.kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.mask = mask
        result, message = self.draw_result(frame, mask)
        
    def make_mask(self, converted):
        mask = np.zeros(converted.shape[:2], dtype=np.uint8)
        for color_range in self.args.ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(converted, color_range.lower, color_range.upper))
        return mask

    def detect_lane(self):
        h, w = self.mask.shape

        # Only consider the bottom portion of the image.
        roi_top = int(h * 0.45)

        roi = np.zeros_like(self.mask)
        roi[roi_top:, :] = self.mask[roi_top:, :]

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Remove tiny noise
        contours = [
            c for c in contours
            if cv2.contourArea(c) > 100
        ]

        if not contours:
            self.lane_error_x = None
            return

        # Assume the largest blue region is the tape
        contour = max(contours, key=cv2.contourArea)

        points = np.squeeze(contour)

        # Fit a line to the tape
        vx, vy, x0, y0 = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx = float(vx)
        vy = float(vy)
        x0 = float(x0)
        y0 = float(y0)

        if abs(vy) < 1e-6:
            self.lane_error_x = 0
            return

        # Look ahead into the image
        lookahead_y = int(h * 0.75)

        # x-coordinate of tape at lookahead_y
        tape_x = x0 + (lookahead_y - y0) * (vx / vy)

        # Normalize to approximately [-1, 1]
        image_center = w / 2.0

        self.lane_error_x = (tape_x - image_center) / image_center
        
    def on_scan(self, msg: LaserScan):
        pass
        

    def control_step(self):

        self.detect_lane()
        turn = clip(self.turn_controller.calculate(-self.lane_error_x), -self.args.max_speed, self.args.max_speed)


        velocity = Twist()
        velocity.angular.z = turn
        self.publish(velocity, "Lane Following")
        pass

    def publish(self, twist: Twist, note: str):
        self.get_logger().info(f"{note} | v={twist.linear.x:+.3f} w={twist.angular.z:+.3f}")
        if self.args.mode == "drive" and not self.stopping:
            self.cmd_pub.publish(twist)
        else:
            self.cmd_pub.publish(Twist())

    
    def stop_robot(self):
        # A single publish() here is not a reliable stop: destroy_node() and
        # rclpy.shutdown() can run before the DDS writer has actually flushed the
        # message, so the chassis never sees it and keeps executing its last
        # command. Publish zero repeatedly, spinning briefly between sends, so
        # the stop has multiple chances to get out before the node is torn down.
        #
        # Cancel the control timer FIRST. Spinning is what gives the zero command
        # time to go out, but spin_once() also runs whatever callback is ready --
        # and a pending control_step() would publish a NON-zero command right
        # after our zero, leaving "keep driving" as the last thing the chassis
        # heard. Stopping means stopping the thing that commands motion, then
        # sending zero, in that order.
        self.stopping = True
        try:
            self.timer.cancel()
        except Exception:
            pass
        stop = Twist()
        for _ in range(10):
            # Guard each attempt separately: spin_once() runs other callbacks,
            # and one of them raising must not cancel the remaining stop
            # publishes -- that would quietly turn this back into a one-shot
            # stop, which is the bug this loop exists to prevent.
            try:
                self.cmd_pub.publish(stop)
                rclpy.spin_once(self, timeout_sec=0.05)
            except Exception:
                pass

def build_parser():
    p = argparse.ArgumentParser(description="Week 6 self drive capstone")
 
    p.add_argument("--image-topic", default="/ascamera/camera_publisher/rgb0/image", help="RGB image topic.")
    p.add_argument("--scan-topic", default="/scan_raw", help="LD19 LaserScan topic (try /scan if /scan_raw is absent; confirm with ros2 topic list).")
    p.add_argument("--cmd-vel-topic", default="/controller/cmd_vel", help="Twist topic the MentorPi chassis listens to (verify with ros2 topic list).")
    p.add_argument("--tolerance", type=float, default=0.12, help="Distance to count a waypoint as reached (m).")
    p.add_argument("--max-speed", type=float, default=0.15, help=f"Bounded forward speed (m/s); hard-capped at {MENTORPI_MAX_SPEED}.")
    p.add_argument("--max-turn", type=float, default=0.8, help=f"Bounded turn rate (rad/s); hard-capped at {MENTORPI_MAX_TURN}.")
    p.add_argument("--ang-pid", type=float, default=[1.0, 0.0, 0.0], nargs=3, help="PID error to turn rate.")
    p.add_argument("--lin-pid", type=float, default=[3.0, 0.0, 0.02], nargs=3, help="PID error to forward speed.")
    p.add_argument("--data-timeout", type=float, default=0.6, help="Stop if no odometry for this many seconds.")
    p.add_argument("--rate", type=float, default=10.0, help="Control loop rate (Hz).")
    p.add_argument("--space", type=canonical_space, default="hsv", help="hsv, rgb, lab, ycrcb, or ycbcr")
    p.add_argument("--color-name", default="red", help="red, green, blue, or a custom label")
    p.add_argument("--lower", type=parse_triplet, help="first lower threshold triplet")
    p.add_argument("--upper", type=parse_triplet, help="first upper threshold triplet")
    p.add_argument("--lower2", type=parse_triplet, help="optional second lower threshold triplet")
    p.add_argument("--upper2", type=parse_triplet, help="optional second upper threshold triplet")
    return p


def main():
    args = build_parser().parse_args()
    if not args.waypoints.strip():
        print("Provide at least one waypoint.", file=sys.stderr)
        raise SystemExit(2)
    if args.max_speed > MENTORPI_MAX_SPEED:
        print(f"--max-speed {args.max_speed} exceeds the MentorPi limit; capping to {MENTORPI_MAX_SPEED} m/s.", file=sys.stderr)
        args.max_speed = MENTORPI_MAX_SPEED
    if args.max_turn > MENTORPI_MAX_TURN:
        print(f"--max-turn {args.max_turn} exceeds the MentorPi limit; capping to {MENTORPI_MAX_TURN} rad/s.", file=sys.stderr)
        args.max_turn = MENTORPI_MAX_TURN

    # The controller only re-checks "am I there yet?" once per control tick, so
    # between checks the robot travels max_speed/rate metres. If the tolerance is
    # not comfortably bigger than that step, the robot can jump straight over the
    # acceptance circle every tick, never register the waypoint as reached, and
    # orbit it forever. Same idea as Week 4's inference latency: the loop rate,
    # not the sensor, sets how finely the robot can act.
    step = args.max_speed / args.rate
    if args.tolerance < 2.0 * step:
        print(f"WARNING: --tolerance {args.tolerance:.3f} m is small next to the "
              f"{step:.3f} m the robot covers per control tick "
              f"({args.max_speed:.2f} m/s / {args.rate:.0f} Hz). The robot may "
              f"circle a waypoint instead of reaching it. Use at least "
              f"{2.0 * step:.2f} m, or lower --max-speed.", file=sys.stderr)

    rclpy.init()
    node = SelfDrive(args)
    try:
        # Not rclpy.spin(): this loop lets the node end the run itself once the
        # plan is complete, so the trajectory log is saved without waiting for a
        # Ctrl+C. Ctrl+C still works at any point.
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.finished:
            node.get_logger().info("plan complete -> stopping")
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.save_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
