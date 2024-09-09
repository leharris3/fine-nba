from enum import Enum

class ActionName(str, Enum):
    MISSED_THREE_POINTER = "3-"
    ASSISTING = "Assisting"
    SCREEN = "Screen"
    REBOUND = "Rebound"
    TURNOVER = "Turnover"
    MADE_SINGLE_FREETHROW = "1+"
    MISSED_SINGLE_FREETHROW = "1-"
    AND_ONE = "2+1"
    MISSED_TWO_POINTER = "2-"
    MADE_TWO_POINTER = "2+"
    FOUL = "Foul"
    PICK_N_ROLL = "Pick'n'Roll"
    POST = "Post"
    STEAL = "Steal"
    TECHNICAL_FOUL = "Technical foul"
    MADE_THREE_POINTER = "3+"
    SECOND_FOUL = "2F"
    THIRD_FOUL = "3F"
    UNSPORTMANLIKE_FOUL = "Unsportmanlike foul"
    THREE_PLUS_ONE = "3+1"
    SECOND_CHANCE = "Second chance"
    MADE_TWO_FREETHROWS = "2FT+"
    MISSED_TWO_FREETHROWS = "2FT-"
    MADE_THREE_FREETHROWS = "3FT+"
    MISSED_THREE_FREETHROWS = "3FT-"

def get_action_description(path):
    try:
        action = path.split("_")[-4]
        for enum_key, enum_value in ActionName.__members__.items():
            if enum_value.value == action:
                template = get_action_template(enum_key)
                return template
        return None
    except IndexError:
        return None

def get_action_template(enum_key):
    templates = {
        'MISSED_THREE_POINTER': "A basketball player missing a three-point shot",
        'ASSISTING': "A basketball player assisting on a play",
        'SCREEN': "A basketball player setting a screen",
        'REBOUND': "A basketball player grabbing a rebound",
        'TURNOVER': "A basketball player committing a turnover",
        'MADE_SINGLE_FREETHROW': "A basketball player making a free throw",
        'MISSED_SINGLE_FREETHROW': "A basketball player missing a free throw",
        'AND_ONE': "A basketball player scoring and being fouled",
        'MISSED_TWO_POINTER': "A basketball player missing a two-point shot",
        'MADE_TWO_POINTER': "A basketball player making a two-point shot",
        'FOUL': "A basketball player committing a foul",
        'PICK_N_ROLL': "A basketball player executing a pick and roll",
        'POST': "A basketball player posting up",
        'STEAL': "A basketball player stealing the ball",
        'TECHNICAL_FOUL': "A basketball player receiving a technical foul",
        'MADE_THREE_POINTER': "A basketball player making a three-point shot",
        'SECOND_FOUL': "A basketball player committing their second foul",
        'THIRD_FOUL': "A basketball player committing their third foul",
        'UNSPORTMANLIKE_FOUL': "A basketball player committing an unsportsmanlike foul",
        'THREE_PLUS_ONE': "A basketball player making a three-pointer and being fouled",
        'SECOND_CHANCE': "A basketball player getting a second chance opportunity",
        'MADE_TWO_FREETHROWS': "A basketball player making two free throws",
        'MADE_THREE_FREETHROWS': "A basketball player making three free throws",
        'MISSED_THREE_FREETHROWS': "A basketball player missing three free throws",
        'DISQUALIFYING_FOUL': "A basketball player committing a disqualifying foul"
    }
    return templates.get(enum_key)