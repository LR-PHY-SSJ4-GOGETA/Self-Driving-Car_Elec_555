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
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
except ImportError as exc:
    print("Missing a ROS module. Run inside the sourced MentorPi ROS 2 container.", file=sys.stderr)
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


def parse_waypoints(text):
    pts = []
    for pair in text.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x, y = pair.split(",")
        pts.append((float(x), float(y)))
    return pts

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
        self.waypoints = parse_waypoints(args.waypoints)
        self.wi = 0
        self.pose = None            # (x, y, yaw) in the start frame (0,0 facing +x at launch)
        self.odom_twist = None
        self.origin = None          # first odom pose; waypoints are relative to it
        self.last_odom = None

        self.log = [] if args.log_file else None
        self.stopping = False       # set by stop_robot(); blocks any further motion
        self.finished = False       # plan complete -> main loop exits cleanly
        self.done_ticks = 0         # zero-velocity ticks held after the last waypoint

        self.cmd_pub = self.create_publisher(Twist, args.cmd_vel_topic, 10)

        self.create_subscriber()

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
    
    def control_step(self):
        twist = Twist()
        if self.pose is None:
            self.publish(twist, "no odometry yet -> stop")
            return
        age = (self.get_clock().now() - self.last_odom).nanoseconds / 1e9
        if age > self.args.data_timeout:
            self.publish(twist, f"odometry stale ({age:.1f}s) -> stop")
            return

        x, y, yaw = self.fused_pose if self.fused_pose is not None else self.pose
        if self.log is not None:
            time = self.get_clock().now().nanoseconds / 1e9

            odom = self.pose        or (float("nan"), float("nan"), float("nan"))
            icp = self.icp_pose     or (float("nan"), float("nan"), float("nan"))
            fused = self.fused_pose or (float("nan"), float("nan"), float("nan"))

            self.log.append((time,

                odom[0], odom[1], odom[2],
                icp[0], icp[1], icp[2],
                fused[0], fused[1], fused[2]
            ))

        if self.wi >= len(self.waypoints):
            # Hold zero for a second so the chassis definitely has the stop, then
            # let main() fall out of its spin loop and save the log.
            self.done_ticks += 1
            self.publish(twist, "all waypoints reached -> stop")
            if self.done_ticks >= int(self.args.rate):
                self.finished = True
            return

        gx, gy = self.waypoints[self.wi]
        dist = math.hypot(gx - x, gy - y)

        if self.speed_controller.at_tolerance(dist):
            self.get_logger().info(f"reached waypoint {self.wi} ({gx:.2f}, {gy:.2f})")
            self.wi += 1
            self.publish(twist, "waypoint reached")
            return

        herr = wrap(math.atan2(gy - y, gx - x) - yaw)
        turn = -clip(self.turn_controller.calculate(herr), -self.args.max_turn, self.args.max_turn)
        # slow the forward speed when badly mis-aimed (cos gate), like the twin
        speed = clip(-self.speed_controller.calculate(dist), 0.0, self.args.max_speed) * max(0.0, math.cos(herr))

        twist.linear.x = clip(speed, 0.0, self.args.max_speed)
        twist.angular.z = clip(turn, -self.args.max_turn, self.args.max_turn)
        self.publish(twist, f"-> wp {self.wi} ({gx:.2f},{gy:.2f}) dist={dist:.2f} herr={math.degrees(herr):+.0f}deg")

    def publish(self, twist: Twist, note: str):
        pose = self.fused_pose if self.fused_pose is not None else self.pose

        p = "none" if pose is None else f"({pose[0]:+.2f},{pose[1]:+.2f}) yaw={math.degrees(pose[2]):+.0f}"
        self.get_logger().info(f"pose={p} | {note} | v={twist.linear.x:+.3f} w={twist.angular.z:+.3f}")
        if self.args.mode == "drive" and not self.stopping:
            self.cmd_pub.publish(twist)
        else:
            self.cmd_pub.publish(Twist())

    def publish_scan_pose(self, x, y, yaw, timestamp, avg_error):

        msg = PoseWithCovarianceStamped()

        msg.header.stamp = timestamp.to_msg()
        msg.header.frame_id = "odom"

        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0

        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        #Guessed covariance values. Im a bit worried about how it will turn out
        variance_xy = max(0.04, (3.0 * avg_error) ** 2)      # ~20 cm confidence interval
        variance_yaw = max(math.radians(10.0) ** 2, ...)      # ~10 degree confidence interval

        cov = [0.0] * 36

        cov[0] = variance_xy      # x
        cov[7] = variance_xy      # y
        cov[35] = variance_yaw    # yaw

        msg.pose.covariance = cov

        self.scan_pub.publish(msg)


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

    def save_log(self):
        if not self.log:
            return
        try:
            with open(self.args.log_file, "w") as f:
                f.write(
                    "time,"
                    "odom_x,odom_y,odom_yaw,"
                    "icp_x,icp_y,icp_yaw,"
                    "fused_x,fused_y,fused_yaw\n"
                )
                for row in self.log:
                    f.write(",".join(f"{v:.4f}" for v in row) + "\n")
            self.get_logger().info(f"saved {len(self.log)} trajectory points to {self.args.log_file}")
        except Exception as exc:
            self.get_logger().error(f"could not save log: {exc}")


def build_parser():
    p = argparse.ArgumentParser(description="Week 5 bounded waypoint-following behavior.")
    p.add_argument("--mode", choices=("report", "drive"), default="report",
                   help="report = compute + print but DO NOT move (default); drive = move with bounded commands.")
    p.add_argument("--waypoints", default="0.8,0.0; 0.8,0.8; 0.0,0.8; 0.0,0.0",
                   help='Waypoints "x,y; x,y; ..." relative to the robot\'s pose at launch (start = 0,0 facing +x).')
    p.add_argument("--absolute", action="store_true",
                   help="Interpret waypoints in the raw odom frame instead of relative to the start pose.")
    p.add_argument("--odom-topic", default="/odom", help="Odometry topic (nav_msgs/Odometry).")
    p.add_argument("--scan-topic", default="/scan_raw", help="LD19 LaserScan topic (try /scan if /scan_raw is absent; confirm with ros2 topic list).")
    p.add_argument("--cmd-vel-topic", default="/controller/cmd_vel", help="Twist topic the MentorPi chassis listens to (verify with ros2 topic list).")
    p.add_argument("--tolerance", type=float, default=0.12, help="Distance to count a waypoint as reached (m).")
    p.add_argument("--max-speed", type=float, default=0.15, help=f"Bounded forward speed (m/s); hard-capped at {MENTORPI_MAX_SPEED}.")
    p.add_argument("--max-turn", type=float, default=0.8, help=f"Bounded turn rate (rad/s); hard-capped at {MENTORPI_MAX_TURN}.")
    p.add_argument("--ang-pid", type=float, default=[1.0, 0.0, 0.0], nargs=3, help="PID error to turn rate.")
    p.add_argument("--lin-pid", type=float, default=[3.0, 0.0, 0.02], nargs=3, help="PID error to forward speed.")
    p.add_argument("--data-timeout", type=float, default=0.6, help="Stop if no odometry for this many seconds.")
    p.add_argument("--rate", type=float, default=10.0, help="Control loop rate (Hz).")
    p.add_argument("--log-file", default="", help="Optional CSV path to save the (x,y,yaw) trajectory for sim-vs-real comparison.")
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
    node = WaypointFollow(args)
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
