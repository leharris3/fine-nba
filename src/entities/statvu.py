import os
import numpy as np
from typing import Dict, List, Optional


class PlayerPosition:

    def __init__(self, player_position: List):
        self.team_id: str = player_position[0]
        self.player_id: str = player_position[1]
        self.x: str = player_position[2]
        self.y: str = player_position[3]
        self.z: str = player_position[4]

    def to_dict(self) -> Dict:
        return {
            "team_id": self.team_id,
            "player_id": self.player_id,
            "x": self.x,
            "y": self.y,
            "z": self.z,
        }


class Moment:

    def __init__(self, moment: List):
        self.period: str = moment[0]
        self.moment_id: str = moment[1]
        self.time_remaining_in_quarter: str = moment[2]
        self.shot_clock: str = moment[3]
        self.player_positions: List[PlayerPosition] = self.get_player_positions(
            moment[5]
        )

    def get_player_positions(
        self, player_positions: List[List]
    ) -> List[PlayerPosition]:
        player_positions_arr = []
        for pp in player_positions:
            player_positions_arr.append(PlayerPosition(pp))
        return player_positions_arr

    def to_dict(self) -> Dict:
        moment_dict = {
            "period": self.period,
            "moment_id": self.moment_id,
            "time_remaining": self.time_remaining_in_quarter,
            "shot_clock": self.shot_clock,
            "player_positions": [pp.to_dict() for pp in self.player_positions],
        }
        return moment_dict


class Event:

    def __init__(self, event: Dict):
        self.event_id: str = event["eventId"]
        self.visitor: str = event["visitor"]
        self.home: str = event["home"]
        self.moments: List[Moment] = self.get_moments(event["moments"])

    def get_moments(self, moments: List[List]) -> List[Moment]:
        moments_arr = []
        for moment in moments:
            moments_arr.append(Moment(moment))
        return moments_arr


class StatVUAnnotation:
    """
    Wrapper and helper functions for sportvu annotations for 15'-16' NBA games.
    Data Source: https://github.com/linouk23/NBA-Player-Movements
    """

    def __init__(self, fp: str):
        assert os.path.isfile(fp), f"Error: invalid path to statvu file: {fp}"
        assert fp.endswith(".json"), f"Error: invalid ext, expect .json for fp: {fp}"
        self.fp: str = fp
        with open(fp, "r") as f:
            self.data: Dict = ujson.load(f)

        self.gameid: str = self.data["gameid"]
        self.gamedata: str = self.data["gamedate"]
        self.events: List[Event] = self.get_events(self.data["events"])
        self.quarter_time_remaining_moment_map = (
            self.get_quarter_time_remaining_moment_map()
        )

    def get_events(self, events: List[Dict]) -> List[Event]:
        events_arr = []
        for event in events:
            events_arr.append(Event(event))
        return events_arr

    def get_quarter_time_remaining_moment_map(self) -> Dict[int, Dict[float, Moment]]:
        """
        {
            quarter: {
                time_remaining: Moment
            }
        }
        """

        quarter_time_remaining_map = {}
        for event in self.events:
            for moment in event.moments:
                period = moment.period
                time_remaining = moment.time_remaining_in_quarter
                if period not in quarter_time_remaining_map:
                    quarter_time_remaining_map[period] = {}
                quarter_time_remaining_map[period][time_remaining] = moment
        return quarter_time_remaining_map

    def find_closest_moment(self, val: float, period: int) -> Moment:
        """
        Return the closest `Moment` to a time remaining `val` in a game `period`.
        """

        tr_moments_map_subset: Dict[Moment] = self.quarter_time_remaining_moment_map[
            period
        ]

        keys: np.array = np.array(list(tr_moments_map_subset.keys())).astype(float)
        closest_idx: int = np.argmin(abs(keys - val))
        return tr_moments_map_subset[keys[closest_idx]]
