#!/usr/bin/env python3
"""ROS 2 self-driving-car capstone scaffold for ELEC 555 Week 6.

Run inside the sourced MentorPi ROS 2 container. This integrates the whole course
into one bounded behavior: FOLLOW a lane line (color, Week 1), STOP for a stop
cue such as a red light / stop sign (Week 3-4), and STOP for obstacles with the
safety bubble (Week 2, the AVOID state) -- arbitrated by a priority state machine.

It is a *scaffold*: it works as-is on a taped lane with a red stop cue and a
LiDAR, and it is meant to be extended (e.g. swap the color stop cue for your Week 4
YOLOv5 sign detections on ``/yolov5_ros2/object_detect``).

SAFETY FIRST (same rules as Weeks 2-5)
--------------------------------------
* Default ``--mode report``: computes and PRINTS the state and the command it WOULD
  send, but publishes ZERO velocity. Nothing moves.
* ``--mode drive`` publishes motion, every command clamped to small
  ``--max-speed`` / ``--max-turn`` limits.
* State priority is AVOID (obstacle) > STOP (rule) > DRIVE (lane). Safety wins.
* If the camera frame or LiDAR scan is missing/stale, or the lane is lost, the car
  STOPS. On Ctrl+C / shutdown / error, the control timer is CANCELLED and a zero
  velocity is then published REPEATEDLY for about half a second before the node
  exits (a single last-gasp publish is not guaranteed to reach the chassis before
  teardown, and without the cancel the spin that flushes it would let the control
  timer publish motion again); watch the wheels to confirm the stop.

Begin in report mode with the wheels off the ground; move the lane line, a red
cue, and your hand (obstacle) in front of the sensors and confirm the STATE
changes correctly. Then drive on a taped course in a clear area.

MentorPi defaults (per https://docs.hiwonder.com/projects/MentorPi/en/latest/):
camera ``/ascamera/camera_publisher/rgb0/image``; LiDAR (LD19) ``/scan_raw``;
chassis ``/controller/cmd_vel`` (Twist; +z left, -z right; <=0.6 m/s, <=2.0 rad/s).
The MentorPi's own autonomous-driving stack uses Lab-color lane keeping + a
YOLOv5 sign model -- see the Autonomous Driving lesson.
"""

import argparse
import math
import sys

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import LaserScan, Image
except ImportError as exc:
    print("Missing a ROS module. Run inside the sourced MentorPi ROS 2 container.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

try:
    import cv2
    from cv_bridge import CvBridge
    import numpy as np
except ImportError as exc:
    print("OpenCV or NumPy not installed inside the sourced MentorPi ROS2 container.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

MENTORPI_MAX_SPEED = 0.6
MENTORPI_MAX_TURN = 2.0


def clip(v, lo, hi):
    return max(lo, min(hi, v))


def parse_triplet(text):
    try:
        values = tuple(int(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use three integers, for example 35,60,40") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError("use exactly three values separated by commas")
    return values


class PIDController:
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
        super().__init__("week6_self_driving")
        self.args = args
        self.stopping = False
        self.finished = False

        self.bridge = CvBridge()

        self.frames_seen = 0
        self.lane_error_x = None
        self.lane_history = []
        self.max_history = 5
        self.consecutive_losses = 0
        self.search_direction = 1
        self.search_start_time = None

        self.last_image = self.get_clock().now()
        self.last_scan = self.get_clock().now()

        self.stop_dist = (None, None)
        self.stop_error = (None, None)
        self.stop_until = None
        self.ignore_cue_until = None

        self.light_dist = (None, None)
        self.light_error = (None, None)
        self.light_color = "UNKNOWN"

        self.forward_distance = float("inf")

        self.stop_sign_w = 0.12
        self.stop_sign_h = 0.12
        self.traffic_light_w = 0.06
        self.traffic_light_h = 0.06

        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.mask_pub = self.create_publisher(Image, "week6_self_driving/total_mask", 10)
        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, args.scan_topic, self.on_scan, qos_profile_sensor_data)

        kp, ki, kd = self.args.lin_pid
        self.speed_controller = PIDController(kp, ki, kd, dt=1.0/self.args.rate)
        self.speed_controller.set_setpoint(0.0)
        self.speed_controller.set_tolerance(self.args.tolerance)

        kp, ki, kd = self.args.ang_pid
        self.turn_controller = PIDController(kp, ki, kd, dt=1.0/self.args.rate)
        self.turn_controller.set_setpoint(0.0)
        self.turn_controller.set_tolerance(math.radians(3))

        self.get_logger().info(
            f"MODE={args.mode.upper()} | "
            f"lane HSV=({args.lower},{args.upper}) | "
            f"max_speed={args.max_speed:.2f} m/s | "
            f"max_turn={args.max_turn:.2f} rad/s"
        )
        if args.mode == "report":
            self.get_logger().info("REPORT mode: computing only, publishing ZERO velocity. Nothing moves.")
        else:
            self.get_logger().warn("DRIVE mode: the robot WILL move. Keep the area clear; hand on Ctrl+C.")

        self.timer = self.create_timer(1.0 / args.rate, self.control_step)

    def fresh(self, stamp):
        if stamp is None:
            return False
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if self.frames_seen < 10:
            return age <= self.args.data_timeout * 2
        return age <= self.args.data_timeout

    def on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frames_seen += 1
        self.lane_error_x, lane_mask = self.lane_error(frame)
        
        stop_data = self.detect_stop_sign(frame)
        if stop_data[0] is not None:
            self.stop_dist = (stop_data[0], stop_data[1])
            self.stop_error = (stop_data[2], stop_data[3])
        else:
            self.stop_dist = (None, None)
            self.stop_error = (None, None)

        light_data = self.detect_traffic_light(frame)
        if light_data[0] is not None:
            self.light_dist = (light_data[0], light_data[1])
            self.light_error = (light_data[2], light_data[3])
            self.light_color = light_data[4]
        else:
            self.light_dist = (None, None)
            self.light_error = (None, None)
            self.light_color = "UNKNOWN"

        stop_mask = stop_data[4] if stop_data[4] is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
        light_mask = light_data[5] if light_data[5] is not None else np.zeros(frame.shape[:2], dtype=np.uint8)
        lane_mask = lane_mask if lane_mask is not None else np.zeros(frame.shape[:2], dtype=np.uint8)

        total_mask = cv2.bitwise_or(stop_mask, light_mask)
        total_mask = cv2.bitwise_or(total_mask, lane_mask)

        final_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR) 

        self.mask_pub.publish(self.bridge.cv2_to_imgmsg(final_mask))

        self.last_image = self.get_clock().now()

    def lane_error(self, img):
        h, w = img.shape[:2]
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # How much the ROI cuts from the image. Used to focus on the track
        roi_start = int(h * (1 - self.args.lane_lookahead))
        roi = hsv[roi_start:, :]
        
        band = cv2.inRange(roi, np.array(self.args.lower), np.array(self.args.upper))

        if self.args.lower2 is not None and self.args.upper2 is not None:
            band2 = cv2.inRange(roi, np.array(self.args.lower2), np.array(self.args.upper2))
            band = cv2.bitwise_or(band, band2)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)

        M = cv2.moments(band, binaryImage=True)
        if M["m00"] < self.args.min_lane_area:
            return None, None
            
        cx = M["m10"] / M["m00"]

        # Full-size image with the band. Only used for publish and debugging
        full_band = np.zeros((h, w), dtype=np.uint8)
        full_band[roi_start:, :] = band
        
        return (cx - w / 2.0) / (w / 2.0), full_band

    def detect_stop_sign(self, img):
        h, w = img.shape[:2]

        lower_red1 = self.args.stop_red_lower1
        upper_red1 = self.args.stop_red_upper1
        lower_red2 = self.args.stop_red_lower2
        upper_red2 = self.args.stop_red_upper2

        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            num_corners = len(approx)

            if 7 <= num_corners <= 9:
                _, _, pixel_w, pixel_h = cv2.boundingRect(contour)
                distance_w = (self.args.fx * self.stop_sign_w) / pixel_w
                distance_h = (self.args.fy * self.stop_sign_h) / pixel_h

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return distance_w, distance_h, (cx - w/2)/(w/2), (cy - h/2)/(h/2), red_mask

        return None, None, None, None, None

    def detect_traffic_light(self, img):
        h, w = img.shape[:2]

        lower_red1 = self.args.light_red_lower1
        upper_red1 = self.args.light_red_upper1
        lower_red2 = self.args.light_red_lower2
        upper_red2 = self.args.light_red_upper2

        lower_green = self.args.light_green_lower
        upper_green = self.args.light_green_upper

        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        full_mask = cv2.bitwise_or(red_mask, green_mask)

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = (4 * np.pi * area) / (perimeter ** 2)

            if 0.85 <= circularity <= 1.2:
                contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                mean_hsv = cv2.mean(hsv, mask=contour_mask)
                mean_hue = mean_hsv[0]

                _, _, pixel_w, pixel_h = cv2.boundingRect(contour)
                distance_w = (self.args.fx * self.traffic_light_w) / pixel_w
                distance_h = (self.args.fy * self.traffic_light_h) / pixel_h

                if mean_hue < 12 or mean_hue > 165:
                    color = "RED"
                elif 35 <= mean_hue <= 85:
                    color = "GREEN"
                else:
                    color = "UNKNOWN"

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return distance_w, distance_h, (cx - w/2)/(w/2), (cy - h/2)/(h/2), color, red_mask

        return None, None, None, None, None, None

    def on_scan(self, msg: LaserScan):
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = ranges.size
        if n == 0:
            return
        angles = msg.angle_min + np.arange(n) * msg.angle_increment
        valid = (
            np.isfinite(ranges)
            & (ranges >= msg.range_min)
            & (ranges <= msg.range_max)
        )
        half = math.radians(self.args.front_fov / 2.0)
        wrapped = np.arctan2(np.sin(angles), np.cos(angles))
        front = valid & (np.abs(wrapped) <= half)

        self.forward_distance = float(np.min(ranges[front])) if np.any(front) else math.inf
        self.last_scan = self.get_clock().now()

    def control_step(self):
        now = self.get_clock().now()
        twist = Twist()

        if not self.fresh(self.last_image) or not self.fresh(self.last_scan):
            return self.publish(twist, "STOP", "no fresh camera/scan")

        if self.forward_distance <= self.args.stop_distance:
            return self.publish(twist, "AVOID", f"obstacle {self.forward_distance:.2f} m")

        holding = self.stop_until is not None and now.nanoseconds < self.stop_until
        ignoring = self.ignore_cue_until is not None and now.nanoseconds < self.ignore_cue_until

        red_light_detected = (
            self.light_color == "RED" 
            and self.light_dist[0] is not None 
            and self.light_dist[0] <= self.args.stop_distance
        )

        stop_sign_detected = (
            self.stop_dist[0] is not None 
            and self.stop_dist[0] <= self.args.stop_distance
        )

        stop_cue_detected = red_light_detected or stop_sign_detected

        if holding:
            return self.publish(twist, "STOP", "holding at stop cue")

        if stop_cue_detected and not ignoring:
            self.stop_until = now.nanoseconds + int(self.args.stop_hold * 1e9)
            self.ignore_cue_until = now.nanoseconds + int((self.args.stop_hold + self.args.stop_ignore) * 1e9)
            return self.publish(twist, "STOP", "stop cue detected")

        if self.lane_error_x is None:
            self.consecutive_losses += 1
            
            if self.consecutive_losses < 15 and len(self.lane_history) >= 3:
                predicted_error = self.lane_history[-1] * 0.6
                turn = clip(
                    self.turn_controller.calculate(predicted_error),
                    -self.args.max_turn * 0.4,
                    self.args.max_turn * 0.4
                )
                speed = self.args.max_speed * 0.15
                twist.linear.x = speed
                twist.angular.z = turn
                self.publish(twist, "RECOVERY", f"loss={self.consecutive_losses}")
                return
            
            return
        
        self.consecutive_losses = 0
        self.lane_history.append(self.lane_error_x)
        if len(self.lane_history) > self.max_history:
            self.lane_history.pop(0)
        self.search_start_time = None

        turn = clip(
            self.turn_controller.calculate(self.lane_error_x),
            -self.args.max_turn,
            self.args.max_turn
        )

        base_speed = clip(
            self.speed_controller.calculate(
                self.args.stop_distance - clip(self.forward_distance, 0, 25)
            ),
            -self.args.max_speed,
            self.args.max_speed
        )
        
        speed = base_speed * (1.0 - min(1.0, abs(self.lane_error_x) * 0.5))
        
        if self.forward_distance <= self.args.slow_distance:
            speed *= 0.4

        speed = clip(speed, 0.02, self.args.max_speed)

        twist.linear.x = speed
        twist.angular.z = turn

        self.publish(twist, "DRIVE", f"lane_err={self.lane_error_x:+.2f}")

    def publish(self, twist: Twist, state: str, why: str):
        lane_log = "none" if self.lane_error_x is None else f"{self.lane_error_x:+.3f}"
        obs = "inf" if not math.isfinite(self.forward_distance) else f"{self.forward_distance:.2f}"
        
        stop_detected = self.stop_dist[0] is not None
        stop_log = f"{self.stop_dist[0]:.2f}m" if stop_detected else "none"

        light_detected = self.light_dist[0] is not None
        light_log = f"{self.light_color} {self.light_dist[0]:.2f}m" if light_detected else "none"

        holding = self.stop_until is not None
        ignoring = self.ignore_cue_until is not None

        self.get_logger().info(
            f"[{state}] {why} | "
            f"lane={lane_log} | "
            f"obs={obs} | "
            f"stop={stop_log} | "
            f"light={light_log} | "
            f"holding={holding} ignoring={ignoring} | "
            f"v={twist.linear.x:+.3f} w={twist.angular.z:+.3f}"
        )

        if self.args.mode == "drive" and not self.stopping:
            self.cmd_pub.publish(twist)
        else:
            self.cmd_pub.publish(Twist())

    def stop_robot(self):
        self.stopping = True
        try:
            self.timer.cancel()
        except Exception:
            pass
        stop = Twist()
        for _ in range(10):
            try:
                self.cmd_pub.publish(stop)
                rclpy.spin_once(self, timeout_sec=0.05)
            except Exception:
                pass


def build_parser():
    p = argparse.ArgumentParser(description="Week 6 self drive")
    
    p.add_argument("--mode", choices=("report", "drive"), default="report", help="report = compute/print only; drive = move robot.")
    p.add_argument("--image-topic", default="/ascamera/camera_publisher/rgb0/image", help="RGB image topic.")
    p.add_argument("--scan-topic", default="/scan_raw", help="LD19 LaserScan topic.")
    p.add_argument("--cmd-vel-topic", default="/controller/cmd_vel", help="Twist command topic.")
    
    p.add_argument("--tolerance", type=float, default=0.12, help="Distance tolerance (m).")
    p.add_argument("--max-speed", type=float, default=0.15, help=f"Bounded forward speed (m/s); capped at {MENTORPI_MAX_SPEED}.")
    p.add_argument("--max-turn", type=float, default=0.8, help=f"Bounded turn rate (rad/s); capped at {MENTORPI_MAX_TURN}.")
    p.add_argument("--ang-pid", type=float, default=[1.0, 0.0, 0.0], nargs=3, help="PID error to turn rate.")
    p.add_argument("--lin-pid", type=float, default=[3.0, 0.0, 0.02], nargs=3, help="PID error to forward speed.")
    p.add_argument("--rate", type=float, default=10.0, help="Control loop rate (Hz).")
    p.add_argument("--stop-hold", type=float, default=3.0, help="Seconds to hold at a stop cue.")
    p.add_argument("--stop-ignore", type=float, default=4.0, help="Seconds to ignore the cue after a stop (so the car can leave).")
    p.add_argument("--data-timeout", type=float, default=2.0, help="Stop if camera/scan is older than this (s).")
    p.add_argument("--front-fov", type=float, default=30.0, help="Front FOV for LiDAR angle filtering (deg).")
    p.add_argument("--stop-distance", type=float, default=0.35, help="Robot stops X meters from obstacles")
    p.add_argument("--slow-distance", type=float, default=0.8, help="Obstacle caution distance (m).")
    p.add_argument("--lane-lookahead", type=float, default=0.45, help="Percentage of the image to cut off when looking for lane. 50% = cut off top half")

    p.add_argument("--fx", type=float, default=590.0, help="Focal length horizontal.")
    p.add_argument("--fy", type=float, default=590.0, help="Focal length vertical.")

    p.add_argument("--lower", type=parse_triplet, default=(15, 70, 70), help="lane lower bound 1")
    p.add_argument("--upper", type=parse_triplet, default=(35, 255, 255), help="lane upper bound 1")
    p.add_argument("--lower2", type=parse_triplet, default=None, help="Optional lane lower bound 2")
    p.add_argument("--upper2", type=parse_triplet, default=None, help="Optional lane upper bound 2")
    p.add_argument("--min-lane-area", type=float, default=200, help="Minimum area seen for lane detection")

    p.add_argument("--stop-red-lower1", type=parse_triplet, default=(0, 70, 80))
    p.add_argument("--stop-red-upper1", type=parse_triplet, default=(10, 255, 255))
    p.add_argument("--stop-red-lower2", type=parse_triplet, default=(170, 70, 80))
    p.add_argument("--stop-red-upper2", type=parse_triplet, default=(179, 255, 255))

    p.add_argument("--light-red-lower1", type=parse_triplet, default=(0, 70, 80))
    p.add_argument("--light-red-upper1", type=parse_triplet, default=(10, 255, 255))
    p.add_argument("--light-red-lower2", type=parse_triplet, default=(170, 70, 80))
    p.add_argument("--light-red-upper2", type=parse_triplet, default=(179, 255, 255))
    p.add_argument("--light-green-lower", type=parse_triplet, default=(45, 70, 70))
    p.add_argument("--light-green-upper", type=parse_triplet, default=(85, 255, 255))

    return p


def main():
    args = build_parser().parse_args()

    if args.max_speed > MENTORPI_MAX_SPEED:
        print(f"--max-speed {args.max_speed} exceeds limit; capping to {MENTORPI_MAX_SPEED} m/s.", file=sys.stderr)
        args.max_speed = MENTORPI_MAX_SPEED
    if args.max_turn > MENTORPI_MAX_TURN:
        print(f"--max-turn {args.max_turn} exceeds limit; capping to {MENTORPI_MAX_TURN} rad/s.", file=sys.stderr)
        args.max_turn = MENTORPI_MAX_TURN

    step = args.max_speed / args.rate
    if args.tolerance < 2.0 * step:
        print(f"WARNING: --tolerance {args.tolerance:.3f} m is small next to the "
              f"{step:.3f} m step covered per control tick. Use at least "
              f"{2.0 * step:.2f} m or lower --max-speed.", file=sys.stderr)

    rclpy.init()
    node = SelfDrive(args)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.finished:
            node.get_logger().info("Plan complete -> stopping")
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()