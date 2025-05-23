from random import randint

# Общие настройки
SCALE_FACTOR = 1.25
ORIGINAL_SIZE = (640, 480)
SCALED_SIZE = (int(ORIGINAL_SIZE[0] * SCALE_FACTOR), int(ORIGINAL_SIZE[1] * SCALE_FACTOR))
INFO_PANEL_HEIGHT = 200
TOTAL_HEIGHT = SCALED_SIZE[1] + INFO_PANEL_HEIGHT

# Параметры детекции
PIXELS_PER_METER = 3 * SCALE_FACTOR
FACTOR_DECLINIS = 5
FLOW_RATE_WINDOW = 30  # Окно расчета интенсивности (секунды)

# Настройки ROI
ROI_CONFIG = [
    {
        "name": "first traffic flow",
        "points": [(0, 350), (320, 350), (225, 480), (0, 480)],
        "color": (0, 255, 0),
        "lanes": [
            {"name": "lane 1", "points": [(0, 350), (190, 350), (35, 480), (0, 480)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))},
            {"name": "lane 2", "points": [(191, 350), (255, 350), (105, 480), (36, 480)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))},
            {"name": "lane 3", "points": [(256, 350), (320, 350), (225, 480), (106, 480)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))}
        ]
    },
    {
        "name": "second traffic flow",
        "points": [(320, 350), (225, 480), (640, 480), (640, 350)],
        "color": (0, 0, 255),
        "lanes": [
            {"name": "lane 1", "points": [(320, 350), (225, 480), (350, 480), (405, 350)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))},
            {"name": "lane 2", "points": [(406, 350), (351, 480), (430, 480), (455, 350)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))},
            {"name": "lane 3", "points": [(456, 350), (431, 480), (510, 480), (500, 350)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))},
            {"name": "lane 4", "points": [(501, 350), (511, 480), (640, 480), (640, 350)], "color": (randint(0, 255), randint(0, 255), randint(0, 255))}
        ]
    }
]