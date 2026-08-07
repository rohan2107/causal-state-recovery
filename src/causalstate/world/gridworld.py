from __future__ import annotations

import numpy as np
from gymnasium.spaces import Box
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.envs import DoorKeyEnv

from causalstate.world.spurious import corrupt

OBS_DIM = 9

VAR_SLICES = {
    "s0": slice(0, 3), # Agent position (x, y) and direction
    "s1": slice(3, 4), # Whether the agent is carrying a key (0 or 1)
    "s2": slice(4, 5), # Whether the door is open (0 or 1)
    "s3": slice(5, 7), # Goal position (x, y)
    "s4": slice(7, 8), # Noise variable
    "s5": slice(8, 9), # Spurious variable
}

OBS_LOW = np.array(
    [1, 1, 0,
     0,
     0,
     1, 1,
     -np.inf,
     0],
    dtype=np.float32,
)

OBS_HIGH = np.array(
    [4, 4, 3,
     1,
     1,
     4, 4,
     np.inf,
     1],
    dtype=np.float32,
)

CAUSAL_ORACLE = frozenset({"s0", "s1", "s2", "s3"})
ALL_VARS = ("s0", "s1", "s2", "s3", "s4", "s5")


def get_var(obs: np.ndarray, var: str) -> np.ndarray:
    return obs[VAR_SLICES[var]]


class FactoredGridWorld(DoorKeyEnv):

    def __init__(
        self,
        noise_std: float = 1.0,
        rho: float | None = None,
        layout_seed: int = 2,
        max_steps: int = 360,
        render_mode: str | None = None,
    ):
        super().__init__(
            size=6,
            max_steps=max_steps,
            render_mode=render_mode,
        )

        self.noise_std = noise_std
        self.rho = rho
        self.layout_seed = layout_seed

        self.observation_space = Box(
            low=OBS_LOW,
            high=OBS_HIGH,
            dtype=np.float32,
        )

        self._door = None
        self._door_pos = None
        self._goal_pos = None

        self._s4 = 0.0
        self._s5 = 0.0

        self._clamps = {}

        self._writers = {
            "s0": self._write_s0,
            "s1": self._write_s1,
            "s2": self._write_s2,
            "s3": self._write_s3,
            "s4": self._write_s4,
            "s5": self._write_s5,
        }

    def _locate_objects(self) -> None:
        door = None
        door_pos = None
        goal_pos = None
        for x in range(self.width):
            for y in range(self.height):
                cell = self.grid.get(x, y)
                if isinstance(cell, Door):
                    if door is None:
                        door = cell
                        door_pos = (x, y)
                    else:
                        raise ValueError("Multiple doors found in the grid.")
                elif isinstance(cell, Goal):
                    if goal_pos is None:
                        goal_pos = (x, y)
                    else:
                        raise ValueError("Multiple goals found in the grid.")

        if door is None:
            raise ValueError("Expected exactly one Door.")

        if goal_pos is None:
            raise ValueError("Expected exactly one Goal.")

        self._door = door
        self._door_pos = door_pos
        self._goal_pos = goal_pos

    def extract_state(self) -> np.ndarray:
        s0 = np.array(
            [
                self.agent_pos[0],
                self.agent_pos[1],
                self.agent_dir,
            ],
            dtype=np.float32,
        )
        s1 = int(isinstance(self.carrying, Key))
        s2 = int(self._door.is_open)
        s3 = self._goal_pos
        s4 = self._s4
        s5 = self._s5

        return np.array(
            [
                *s0,
                s1,
                s2,
                *s3,
                s4,
                s5,
            ],
            dtype=np.float32
        )

    def gen_obs(self) -> np.ndarray:
        return self.extract_state()

    def _gen_grid(self, width: int, height: int) -> None:
        episode_rng = self.np_random
        self.np_random = np.random.default_rng(self.layout_seed)
        try:
            super()._gen_grid(width, height)
        finally:
            self.np_random = episode_rng

        self._locate_objects()
        door_col = self._door_pos[0]
        self.place_agent(
            top=(1, 0),
            size=(door_col - 1, height),
        )
        self.grid.set(
            self._goal_pos[0],
            self._goal_pos[1],
            None,
        )
        goal = Goal()
        goal_pos = self.place_obj(
            goal,
            top=(door_col + 1, 1),
            size=(width - door_col - 2, height - 2),
        )
        self._goal_pos = (
            int(goal_pos[0]),
            int(goal_pos[1])
        )

    def _apply_state_clamps(self) -> None:
        for var in ("s0", "s1", "s2", "s3"):
            if var in self._clamps:
                self._writers[var](self._clamps[var])

    def _resample_nuisance(self) -> None:
        self._s4 = self.np_random.normal(0.0, self.noise_std)
        if self.rho is None:
            self._s5 = 0.0
        else:
            self._s5 = corrupt(
                int(self._door.is_open),
                self.rho,
                self.np_random,
            )

    def _apply_nuisance_clamps(self) -> None:
        for var in ("s4", "s5"):
            if var in self._clamps:
                self._writers[var](self._clamps[var])

    def _settle(self) -> None:
        self._apply_state_clamps()
        self._resample_nuisance()
        self._apply_nuisance_clamps()

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        _, info = super().reset(seed=seed, options=options)
        self._clamps.clear()
        self._settle()
        return self.extract_state(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        _, reward, terminated, truncated, info = super().step(action)
        self._settle()
        info["Y"] = int(
            terminated and tuple(self.agent_pos) == self._goal_pos
        )
        return self.extract_state(), reward, terminated, truncated, info

    def _validate_binary(self, var, value) -> int:
        if value not in (0, 1, False, True):
            raise ValueError(f"Invalid value for {var}: {value}")
        return int(value)

    def _validate_cell(
        self,
        var,
        value,
        *,
        allow_door: bool,
    ) -> tuple[int, int]:
        if not hasattr(value, "__len__") or len(value) != 2:
            raise ValueError(f"Expected an iterable of length 2 for {var}.")

        col, row = value

        if not float(col).is_integer():
            raise ValueError(f"Invalid column for {var}: {col}")

        if not float(row).is_integer():
            raise ValueError(f"Invalid row for {var}: {row}")

        col = int(col)
        row = int(row)

        if not (1 <= col <= self.width - 2):
            raise ValueError(f"Invalid column for {var}: {col}")

        if not (1 <= row <= self.height - 2):
            raise ValueError(f"Invalid row for {var}: {row}")

        if isinstance(self.grid.get(col, row), Wall):
            raise ValueError(f"Position ({col}, {row}) is a wall.")

        if not allow_door and (col, row) == self._door_pos:
            raise ValueError(f"Position ({col}, {row}) is the door position.")

        return (col, row)

    def _validate(self, var, value):
        if var == "s0":
            if not hasattr(value, "__len__") or len(value) != 3:
                raise ValueError(f"Expected an iterable of length 3 for {var}.")

            col, row = self._validate_cell(
                var,
                value[:2],
                allow_door=True,
            )

            direction = value[2]

            if not float(direction).is_integer():
                raise ValueError(f"Invalid direction for {var}: {direction}")

            direction = int(direction)

            if direction not in (0, 1, 2, 3):
                raise ValueError(f"Invalid direction for {var}: {direction}")

            return (col, row, direction)

        elif var in ("s1", "s2", "s5"):
            return self._validate_binary(var, value)

        elif var == "s3":
            return self._validate_cell(
                var,
                value,
                allow_door=False,
            )

        elif var == "s4":
            value = float(value)

            if not np.isfinite(value):
                raise ValueError(f"Invalid value for {var}: {value}")

            return value

        raise ValueError(f"Unknown variable: {var}")

    def _write_s0(self, value):
        col, row, direction = value
        self.agent_pos = (int(col), int(row))
        self.agent_dir = int(direction)

    def _write_s1(self, value):
        if value:
            key = Key(color="yellow")
            key.cur_pos = (-1, -1)
            self.carrying = key
        else:
            self.carrying = None

    def _write_s2(self, value):
        if value:
            self._door.is_locked = False
            self._door.is_open = True
        else:
            self._door.is_open = False

    def _write_s3(self, value):
        col, row = value

        self.grid.set(
            self._goal_pos[0],
            self._goal_pos[1],
            None,
        )

        self.put_obj(
            Goal(),
            int(col),
            int(row),
        )

        self._goal_pos = (int(col), int(row))

    def _write_s4(self, value):
        self._s4 = float(value)

    def _write_s5(self, value):
        self._s5 = int(value)

    def do(self, var, value, hold=False) -> None:
        value = self._validate(var, value)
        self._writers[var](value)
        if hold:
            self._clamps[var] = value

    def release(self, var=None) -> None:
        if var is None:
            self._clamps.clear()
            return

        if var not in self._writers:
            raise ValueError(f"Unknown variable: {var}")
        self._clamps.pop(var, None)

    def intervention_domain(self, var):
        if var == "s0":
            states = []
            for col in range(1, self.width - 1):
                for row in range(1, self.height - 1):
                    if isinstance(self.grid.get(col, row), Wall):
                        continue
                    for direction in range(4):
                        states.append((col, row, direction))
            return tuple(states)
        elif var == "s3":
            states = []
            door_col = self._door_pos[0]
            for col in range(door_col + 1, self.width - 1):
                for row in range(1, self.height - 1):
                    if isinstance(self.grid.get(col, row), Wall):
                        continue
                    states.append((col, row))
            return tuple(states)
        elif var in ("s1", "s2", "s5"):
            return (0, 1)
        elif var == "s4":
            return None

        raise ValueError(f"Unknown variable: {var}")
