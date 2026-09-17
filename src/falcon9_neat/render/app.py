"""Interactive Panda3D visualization for the landing simulator."""
from __future__ import annotations

import math
from pathlib import Path
import random
import numpy as np

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import AntialiasAttrib, ClockObject, Quat, TextNode, WindowProperties, loadPrcFileData

from ..config import AppConfig
from ..controllers import HeuristicLandingController
from ..environment import FalconLandingEnv
from ..neat.genome import Genome
from ..neat.network import NeatLandingController
from .rocket_model import build_booster, build_full_vehicle
from .world import build_world


class LandingSimApp(ShowBase):
    def __init__(self, config: AppConfig, controller, showroom: bool = False):
        # PRC variables must be set before ShowBase creates the window.
        loadPrcFileData("", f"window-title Falcon 9 NEAT Landing Simulator\nframebuffer-multisample 1\nmultisamples {config.render.multisamples}")
        super().__init__()
        self.config = config
        self.controller = controller
        self.showroom = showroom
        self.disableMouse()
        props = WindowProperties()
        props.setSize(config.render.window_width, config.render.window_height)
        self.win.requestProperties(props)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.setBackgroundColor(0.46, 0.60, 0.73, 1.0)
        self.camLens.setFov(58)
        self.camLens.setNearFar(0.2, 7000)

        build_world(self.render, config.render.shadow_map_size)
        self.env = FalconLandingEnv(config, deterministic=True)
        self.obs = self.env.reset()
        self.rocket, self.exhaust = build_booster(config.rocket.length_m, config.rocket.diameter_m)
        self.rocket.reparentTo(self.render)
        self.full_vehicle = None
        if showroom:
            self.rocket.hide()
            self.full_vehicle = build_full_vehicle()
            self.full_vehicle.reparentTo(self.render)
            self.full_vehicle.setPos(0, 0, 35)

        self.paused = False
        self.last_action = np.zeros(3)
        self.last_reward = 0.0
        self.episode_return = 0.0
        self.camera_mode = 0
        self.orbit_angle = 0.0
        self._reset_countdown = 0.0

        self.hud = OnscreenText(
            text="", pos=(-1.31, 0.92), align=TextNode.ALeft, scale=0.038,
            fg=(0.95,0.97,1.0,1), shadow=(0,0,0,0.75), mayChange=True,
        )
        self.help = OnscreenText(
            text="SPACE pause  |  R reset  |  C camera  |  ESC quit",
            pos=(0, -0.94), align=TextNode.ACenter, scale=0.032,
            fg=(0.92,0.94,0.96,1), shadow=(0,0,0,0.7),
        )
        self.accept("escape", self.userExit)
        self.accept("space", self._toggle_pause)
        self.accept("r", self._reset)
        self.accept("c", self._cycle_camera)
        self.taskMgr.add(self._update, "simulation-update")
        self.taskMgr.add(self._update_camera, "camera-update", sort=20)
        self._sync_visual()

    def _toggle_pause(self):
        self.paused = not self.paused

    def _cycle_camera(self):
        self.camera_mode = (self.camera_mode + 1) % 3

    def _reset(self):
        self.obs = self.env.reset()
        self.episode_return = 0.0
        self.last_reward = 0.0
        self._reset_countdown = 0.0
        self.paused = False
        self._sync_visual()

    def _sync_visual(self):
        if self.showroom or self.env.state is None:
            return
        s = self.env.state
        self.rocket.setPos(*map(float, s.position_m))
        q = s.quaternion
        self.rocket.setQuat(Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        throttle = float(self.last_action[0]) if len(self.last_action) else 0.0
        if throttle > 0.03:
            self.exhaust.show()
            flicker = 0.92 + 0.08 * math.sin(37.0 * s.time_s)
            self.exhaust.setScale(0.65 + 0.42 * throttle, 0.65 + 0.42 * throttle, (0.35 + 0.9 * throttle) * flicker)
            self.exhaust.setColorScale(1.0, 0.82, 0.55, 0.16 + 0.42 * throttle)
        else:
            self.exhaust.hide()

    def _update(self, task):
        dt_render = min(ClockObject.getGlobalClock().getDt(), 0.1)
        if self.showroom:
            if self.full_vehicle is not None:
                self.full_vehicle.setH(self.full_vehicle.getH() + 7.0 * dt_render)
            self.hud.setText("FALCON 9 PROCEDURAL SHOWROOM\n70 m class full-stack display model")
            return task.cont
        if self.paused:
            return task.cont
        if self._reset_countdown > 0:
            self._reset_countdown -= dt_render
            if self._reset_countdown <= 0:
                self._reset()
            return task.cont

        # Fixed physics timestep; catch up at most four steps per render frame.
        steps = max(1, min(4, int(round(dt_render / self.config.environment.dt_s))))
        for _ in range(steps):
            self.last_action = self.controller.act(self.obs)
            result = self.env.step(self.last_action)
            self.obs = result.observation
            self.last_reward = result.reward
            self.episode_return += result.reward
            if result.terminated:
                self._reset_countdown = 2.8
                break
        self._sync_visual()
        info = result.info
        self.hud.setText(
            f"FALCON 9 / NEAT LANDING SIM\n"
            f"t       {info['time_s']:6.2f} s\n"
            f"alt     {info['altitude_m']:7.1f} m\n"
            f"range   {info['lateral_distance_m']:7.1f} m\n"
            f"v-z     {info['vertical_speed_mps']:7.2f} m/s\n"
            f"v-xy    {info['horizontal_speed_mps']:7.2f} m/s\n"
            f"tilt    {info['tilt_deg']:7.2f} deg\n"
            f"fuel    {info['fuel_kg']:7.0f} kg\n"
            f"throttle {self.last_action[0]*100:6.1f}%\n"
            f"return  {self.episode_return:8.1f}\n"
            f"state   {info['outcome'].upper()}"
        )
        return task.cont

    def _update_camera(self, task):
        dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
        self.orbit_angle += 0.08 * dt
        if self.showroom:
            self.camera.setPos(105 * math.cos(self.orbit_angle), 105 * math.sin(self.orbit_angle), 48)
            self.camera.lookAt(0,0,32)
            return task.cont
        if self.env.state is None:
            return task.cont
        p = self.env.state.position_m
        alt = max(0.0, p[2] - self.config.rocket.length_m * 0.5)
        if self.camera_mode == 0:
            # Cinematic orbit whose radius contracts near touchdown.
            dist = max(72.0, min(170.0, 88.0 + alt * 0.045))
            angle = self.orbit_angle + 0.7
            target_z = max(15.0, p[2])
            self.camera.setPos(p[0] + dist*math.cos(angle), p[1] + dist*math.sin(angle), target_z + 28.0)
            self.camera.lookAt(float(p[0]), float(p[1]), float(p[2]))
        elif self.camera_mode == 1:
            self.camera.setPos(float(p[0] + 95), float(p[1] - 95), float(max(38.0, p[2] + 20)))
            self.camera.lookAt(float(p[0]), float(p[1]), float(p[2]))
        else:
            self.camera.setPos(115, -145, 65)
            self.camera.lookAt(float(p[0]), float(p[1]), float(max(18.0, p[2])))
        return task.cont


def run_visualization(config: AppConfig, checkpoint: str | Path | None = None, controller_kind: str = "heuristic", showroom: bool = False) -> None:
    if controller_kind == "neat":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for controller=neat")
        controller = NeatLandingController(Genome.load(checkpoint))
    else:
        controller = HeuristicLandingController()
    app = LandingSimApp(config, controller, showroom=showroom)
    app.run()
