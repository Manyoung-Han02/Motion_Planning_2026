"""Nonholonomic robot model skeletons."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, tan


@dataclass(frozen=True)
class RobotState:
    """Planar pose and circular footprint for a car-like robot."""

    x: float
    y: float
    theta: float
    radius: float


@dataclass(frozen=True)
class RobotControl:
    """Velocity and steering command for a kinematic bicycle model."""

    velocity: float
    steering_angle: float


@dataclass(frozen=True)
class RobotSpec:
    """Physical and control limits for a nonholonomic robot."""

    wheelbase: float
    max_speed: float
    max_steering_angle: float


@dataclass(frozen=True)
class Robot:
    """Robot instance with start and goal poses."""

    id: str
    spec: RobotSpec
    start: RobotState
    goal: RobotState

    @property
    def radius(self) -> float:
        """Return the robot footprint radius used for collision checks."""
        return self.start.radius

    def clamp_control(self, control: RobotControl) -> RobotControl:
        """Clamp a raw control command to the robot limits."""
        velocity = max(-self.spec.max_speed, min(self.spec.max_speed, control.velocity))
        steering = max(
            -self.spec.max_steering_angle,
            min(self.spec.max_steering_angle, control.steering_angle),
        )
        return RobotControl(velocity=velocity, steering_angle=steering)

    def step(self, state: RobotState, control: RobotControl, dt: float) -> RobotState:
        """Propagate one kinematic bicycle step.

        This lightweight implementation is enough for smoke tests and demos.
        More accurate integration can be added once planners need it.
        """
        bounded = self.clamp_control(control)
        x_dot = bounded.velocity * cos(state.theta)
        y_dot = bounded.velocity * sin_theta(state.theta)
        theta_dot = bounded.velocity / self.spec.wheelbase * tan(bounded.steering_angle)
        return RobotState(
            x=state.x + x_dot * dt,
            y=state.y + y_dot * dt,
            theta=state.theta + theta_dot * dt,
            radius=state.radius,
        )


def sin_theta(theta: float) -> float:
    """Small wrapper kept separate to make future vectorization straightforward."""
    from math import sin

    return sin(theta)
