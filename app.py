import os
import io
import json
import random
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"

# 단일 사용자용 글로벌 환경 (멀티유저용이라면 별도 세션 관리 필요)
global_env = None

########################
# 엘조윈(가호) 확률
########################
GAHO_TABLE = {
    0: [0.40, 0.30, 0.20, 0.10],
    1: [0.35, 0.25, 0.25, 0.15],
    2: [0.30, 0.20, 0.30, 0.20],
    3: [0.25, 0.15, 0.35, 0.25],
    4: [0.20, 0.10, 0.40, 0.30],
    5: [0.15, 0.05, 0.45, 0.35],
    6: [0.05, 0.05, 0.45, 0.45],
    7: [0.00, 0.00, 0.40, 0.60],
    8: [0.00, 0.00, 0.10, 0.90],
}

########################
# 사전에 정의된 경로 (예: 투구 6단계, 장갑 6단계)
########################
predefined_paths = {
    ("투구", 6): [
        (7,4),(7,5),(7,6),(7,7),
        (6,7),(5,7),(4,7),(3,7),(2,7),(1,7),(0,7),
        (0,6),(0,5),(0,4),(0,3),(0,2),(0,1),(0,0),
        (1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),
        (7,1),(7,2),(7,3)
    ],
    ("장갑", 6): [
        (7,3),(7,2),(7,1),(7,0),
        (6,0),(5,0),(4,0),(3,0),(2,0),(1,0),(0,0),
        (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),
        (1,7),(2,7),(3,7),(4,7),(5,7),(6,7),(7,7),
        (7,6),(7,5),(7,4)
    ]
}

########################
# 기대 비용(계속 vs 복원) 계산
########################
def decide_action(env, p_success):
    if env.current_part == "무기":
        roll_cost = 140
        restore_cost = 2200
    else:
        roll_cost = 110
        restore_cost = 740

    cost_c = roll_cost + (1 - p_success) * restore_cost
    cost_r = restore_cost
    if cost_c < cost_r:
        return f"계속 진행 (굴리기≈{cost_c:.1f}, 복원={cost_r:.1f})"
    else:
        return f"복원 (굴리기≈{cost_c:.1f}, 복원={cost_r:.1f})"

########################
# Environment 클래스
########################
class TranscendenceEnv:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.map_data = {}
        for item in data["maps"]:
            p = item["part"]
            s = item["stage"]
            al = item["action_limit"]
            g = item["grid"]
            self.map_data[(p, s)] = {
                "action_limit": al,
                "grid": g
            }

        self.grid = None
        self.max_actions = 0
        self.remaining_actions = 0

        self.start = None
        self.end = None
        self.agent_pos = None
        self.path = []
        self.path_idx = 0

        self.special_tiles = {}

        self.enhance_count = 0
        self.awaken_next_turn = False
        self.last_move_distance = 0

        self.dice_probs = GAHO_TABLE[0]
        self.current_part = None

        self.simulation_mode = False

    def set_category(self, part, stage):
        self.current_part = part
        info = self.map_data.get((part, stage))
        if not info:
            return
        self.max_actions = info["action_limit"]
        self.remaining_actions = self.max_actions

        self.grid = np.array(info["grid"], dtype=int)

        start_pos = np.where(self.grid == 2)
        end_pos = np.where(self.grid == 3)
        if len(start_pos[0]) > 0:
            self.start = (start_pos[0][0], start_pos[1][0])
        else:
            self.start = None
        if len(end_pos[0]) > 0:
            self.end = (end_pos[0][0], end_pos[1][0])
        else:
            self.end = None

        self.special_tiles.clear()
        self.enhance_count = 0
        self.awaken_next_turn = False
        self.last_move_distance = 0

        self.agent_pos = self.start

        self.path = []
        self.path_idx = 0

        if (part, stage) in predefined_paths:
            self.path = predefined_paths[(part, stage)]
        else:
            if self.start and self.end:
                self.path = self.find_path()

    def set_gaho_level(self, gaho):
        self.dice_probs = GAHO_TABLE.get(gaho, GAHO_TABLE[0])

    def find_path(self):
        if self.grid is None or not self.start or not self.end:
            return []
        visited = set([self.start])
        queue = deque([(self.start, [self.start])])
        while queue:
            (rr, cc), path_so_far = queue.popleft()
            if (rr, cc) == self.end:
                return path_so_far
            for (dr, dc) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = rr + dr, cc + dc
                if (0 <= nr < 8 and 0 <= nc < 8 and self.grid[nr, nc] in [1,2,3]) and ((nr, nc) not in visited):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path_so_far + [(nr, nc)]))
        return []

    def roll_dice(self, user_input=None):
        if user_input:
            try:
                val = int(user_input)
                if 1 <= val <= 4:
                    base_val = val
                else:
                    base_val = random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]
            except:
                base_val = random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]
        else:
            base_val = random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]

        dist = base_val + self.enhance_count
        if self.awaken_next_turn:
            dist *= 3
            self.awaken_next_turn = False
        return dist

    def step(self, user_input=None):
        if self.remaining_actions <= 0 or not self.agent_pos:
            return (self.agent_pos, 0, True, 0)

        self.remaining_actions -= 1
        dist = self.roll_dice(user_input)

        if self.path and self.start and self.end:
            new_idx = min(self.path_idx + dist, len(self.path) - 1)
            moved = new_idx - self.path_idx
            self.path_idx = new_idx
            self.agent_pos = self.path[new_idx]
            self.last_move_distance = moved
        else:
            self.last_move_distance = 0

        self.resolve_tile_effects()

        done = False
        rew = 0
        if self.agent_pos == self.end:
            done = True
            rew = 1
        if self.remaining_actions <= 0:
            done = True

        return (self.agent_pos, rew, done, dist)

    def resolve_tile_effects(self):
        while True:
            if not self.agent_pos:
                break
            tile = self.special_tiles.get(self.agent_pos, None)
            if not tile:
                break
            moved = False

            if tile == "전개":
                if self.path:
                    old_idx = self.path_idx
                    new_idx = min(self.path_idx + 4, len(self.path) - 1)
                    self.path_idx = new_idx
                    self.agent_pos = self.path[new_idx]
                    self.last_move_distance = new_idx - old_idx
                moved = True

            elif tile == "강화":
                self.enhance_count += 1
                break

            elif tile == "각성":
                self.awaken_next_turn = True
                break

            elif tile == "복제":
                if self.last_move_distance > 0 and self.path:
                    old_idx = self.path_idx
                    new_idx = min(self.path_idx + self.last_move_distance, len(self.path) - 1)
                    self.path_idx = new_idx
                    self.agent_pos = self.path[new_idx]
                    self.last_move_distance = new_idx - old_idx
                moved = True

            elif tile == "정화":
                self.purification_move()
                moved = True

            if not moved:
                break

    def purification_move(self):
        dist = self.roll_dice(None)
        if self.path:
            old_idx = self.path_idx
            new_idx = min(self.path_idx + dist, len(self.path) - 1)
            self.path_idx = new_idx
            self.agent_pos = self.path[new_idx]
            self.last_move_distance = new_idx - old_idx

    def place_special_tile(self, r, c, tile_type):
        if self.grid is not None:
            if 0 <= r < 8 and 0 <= c < 8:
                if self.grid[r, c] in [1,2,3]:
                    self.special_tiles[(r, c)] = tile_type

    def place_agent(self, r, c):
        if self.grid is not None:
            if 0 <= r < 8 and 0 <= c < 8:
                if self.grid[r, c] in [1,2,3]:
                    self.agent_pos = (r, c)
                    self.start = (r, c)
                    if (self.current_part, 6) in predefined_paths:
                        self.path = predefined_paths[(self.current_part, 6)]
                    else:
                        self.path = self.find_path()
                    self.path_idx = 0

    def clone_environment(self):
        new_env = TranscendenceEnv.__new__(TranscendenceEnv)
        new_env.map_data = self.map_data
        new_env.grid = np.copy(self.grid)
        new_env.max_actions = self.max_actions
        new_env.remaining_actions = self.remaining_actions
        new_env.start = self.start
        new_env.end = self.end
        new_env.agent_pos = self.agent_pos
        new_env.path = list(self.path)
        new_env.path_idx = self.path_idx
        new_env.special_tiles = dict(self.special_tiles)
        new_env.enhance_count = self.enhance_count
        new_env.awaken_next_turn = self.awaken_next_turn
        new_env.last_move_distance = self.last_move_distance
        new_env.dice_probs = list(self.dice_probs)
        new_env.current_part = self.current_part
        new_env.simulation_mode = True
        return new_env

    def monte_carlo_success_probability(self, num_sim=1000):
        if self.agent_pos == self.end:
            return 1.0
        if self.remaining_actions <= 0:
            return 0.0
        success_count = 0
        for _ in range(num_sim):
            sim_env = self.clone_environment()
            done = False
            rew = 0
            while not done:
                _, reward, done, _ = sim_env.step()
                if done:
                    rew = reward
            if rew == 1:
                success_count += 1
        return success_count / float(num_sim)

    def get_next_dice_info(self):
        lines = []
        for i, p in enumerate(self.dice_probs):
            base_val = i + 1
            dist = base_val + self.enhance_count
            if self.awaken_next_turn:
                dist *= 3
            lines.append(f"{base_val}({p*100:.1f}%) => {dist}")
        return "\n".join(lines)

########################
# 지도 그리기 (Pillow 사용)
########################
TILE_SIZE = 50
MAP_WIDTH = 8 * TILE_SIZE
MAP_HEIGHT = 8 * TILE_SIZE

# 이미지 파일은 static 폴더에서 불러옴
def load_image(filename):
    path = os.path.join("static", filename)
    return Image.open(path).resize((TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS)

tile_img = load_image("tile.png")

special_images = {}
for name in ["전개", "강화", "각성", "복제", "정화"]:
    try:
        special_images[name] = load_image(f"{name}.png")
    except Exception as e:
        print(f"Error loading image for {name}: {e}")
        special_images[name] = None

def draw_map_image(env):
    im = Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    if env.grid is None:
        return im
    for r in range(8):
        for c in range(8):
            x = c * TILE_SIZE
            y = r * TILE_SIZE
            if env.grid[r, c] == 0:
                draw.rectangle([x, y, x+TILE_SIZE, y+TILE_SIZE], fill="black")
            else:
                im.paste(tile_img, (x, y))
            if (r, c) in env.special_tiles:
                tile_type = env.special_tiles[(r, c)]
                if tile_type in special_images and special_images[tile_type]:
                    im.paste(special_images[tile_type], (x, y), special_images[tile_type])
            if env.start == (r, c):
                draw.text((x+TILE_SIZE//2, y+TILE_SIZE//2), "O", fill="blue", font=font, anchor="mm")
            if env.end == (r, c):
                draw.text((x+TILE_SIZE//2, y+TILE_SIZE//2), "X", fill="red", font=font, anchor="mm")
            if env.agent_pos == (r, c):
                draw.text((x+TILE_SIZE//2, y+TILE_SIZE//2), "A", fill="yellow", font=font, anchor="mm")
    return im

########################
# Flask 엔드포인트
########################
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_category", methods=["POST"])
def set_category_endpoint():
    global global_env
    data = request.get_json()
    part = data.get("part")
    stage = int(data.get("stage"))
    gaho = int(data.get("gaho"))
    monte_count = int(data.get("monte_count"))
    global_env = TranscendenceEnv("transcendence_maps.json")
    global_env.set_category(part, stage)
    global_env.set_gaho_level(gaho)
    p = global_env.monte_carlo_success_probability(monte_count)
    info = {
        "actions_text": f"남은 액션: {global_env.remaining_actions}/{global_env.max_actions}",
        "dice_info": global_env.get_next_dice_info(),
        "prob_text": f"성공확률: {p*100:.2f}%",
        "rec_text": decide_action(global_env, p)
    }
    return jsonify(info)

@app.route("/simulate_step", methods=["POST"])
def simulate_step_endpoint():
    global global_env
    data = request.get_json()
    dice_input = data.get("dice_input")
    monte_count = int(data.get("monte_count"))
    if dice_input == "" or dice_input == "0":
        p = global_env.monte_carlo_success_probability(monte_count)
    else:
        global_env.step(dice_input)
        p = global_env.monte_carlo_success_probability(monte_count)
    info = {
        "actions_text": f"남은 액션: {global_env.remaining_actions}/{global_env.max_actions}",
        "dice_info": global_env.get_next_dice_info(),
        "prob_text": f"성공확률: {p*100:.2f}%",
        "rec_text": decide_action(global_env, p)
    }
    return jsonify(info)

@app.route("/place_tile", methods=["POST"])
def place_tile_endpoint():
    global global_env
    data = request.get_json()
    tile_type = data.get("tile_type")
    row = int(data.get("row"))
    col = int(data.get("col"))
    if tile_type == "에이전트":
        global_env.place_agent(row, col)
    else:
        global_env.place_special_tile(row, col, tile_type)
    p = global_env.monte_carlo_success_probability(1000)
    info = {
        "actions_text": f"남은 액션: {global_env.remaining_actions}/{global_env.max_actions}",
        "dice_info": global_env.get_next_dice_info(),
        "prob_text": f"성공확률: {p*100:.2f}%",
        "rec_text": decide_action(global_env, p)
    }
    return jsonify(info)

@app.route("/map_image")
def map_image_endpoint():
    global global_env
    if global_env is None:
        im = Image.new("RGB", (MAP_WIDTH, MAP_HEIGHT), "white")
    else:
        im = draw_map_image(global_env)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render의 PORT 환경 변수 사용
    app.run(host='0.0.0.0', port=port)

