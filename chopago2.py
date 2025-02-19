import json
import tkinter as tk
from tkinter import ttk, simpledialog
import random
import numpy as np
from collections import deque
from PIL import Image, ImageTk
from PIL.Image import Resampling

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
# 사전에 정의된 경로
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
    if env.current_part=="무기":
        roll_cost=140
        restore_cost=2200
    else:
        roll_cost=110
        restore_cost=740

    cost_c= roll_cost + (1 - p_success)*restore_cost
    cost_r= restore_cost
    if cost_c<cost_r:
        return f"계속 진행 (굴리기≈{cost_c:.1f}, 복원={cost_r:.1f})"
    else:
        return f"복원 (굴리기≈{cost_c:.1f}, 복원={cost_r:.1f})"

########################
# Environment
########################
class TranscendenceEnv:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data=json.load(f)

        self.map_data={}
        for item in data["maps"]:
            p=item["part"]
            s=item["stage"]
            al=item["action_limit"]
            g=item["grid"]
            self.map_data[(p,s)] = {
                "action_limit": al,
                "grid": g
            }

        self.grid=None
        self.max_actions=0
        self.remaining_actions=0

        self.start=None
        self.end=None
        self.agent_pos=None
        self.path=[]
        self.path_idx=0

        self.special_tiles={}

        self.enhance_count=0
        self.awaken_next_turn=False
        self.last_move_distance=0

        self.dice_probs = GAHO_TABLE[0]
        self.current_part=None

        self.gaho=0
        self.simulation_mode=False

        # 사전 경로 사용중인지 여부
        self.using_predefined=False

    def set_category(self, part, stage):
        self.current_part=part
        info=self.map_data[(part, stage)]
        self.max_actions=info["action_limit"]
        self.remaining_actions=self.max_actions

        self.grid=np.array(info["grid"], dtype=int)

        start_pos=np.where(self.grid==2)
        end_pos=np.where(self.grid==3)
        if len(start_pos[0])>0:
            self.start=(start_pos[0][0],start_pos[1][0])
        else:
            self.start=None
        if len(end_pos[0])>0:
            self.end=(end_pos[0][0],end_pos[1][0])
        else:
            self.end=None

        self.special_tiles.clear()
        self.enhance_count=0
        self.awaken_next_turn=False
        self.last_move_distance=0

        self.agent_pos=self.start

        self.path=[]
        self.path_idx=0

        # 사전 경로 먼저
        if (part, stage) in predefined_paths:
            self.path= predefined_paths[(part,stage)]
            self.using_predefined=True
        else:
            if self.start and self.end:
                self.path=self.find_path()
                self.using_predefined=False

    def set_gaho_level(self, g):
        self.dice_probs= GAHO_TABLE.get(g, GAHO_TABLE[0])
        self.gaho=g

    def find_path(self):
        if self.grid is None or not self.start or not self.end:
            return []
        from collections import deque

        def can_walk(r,c):
            return (0<=r<8 and 0<=c<8 and self.grid[r,c] in [1,2,3])

        visited=set([self.start])
        queue=deque([(self.start,[self.start])])
        while queue:
            (rr,cc), path_so_far=queue.popleft()
            if (rr,cc)==self.end:
                return path_so_far
            for (dr,dc) in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr,nc= rr+dr, cc+dc
                if can_walk(nr,nc) and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append(((nr,nc), path_so_far+[(nr,nc)]))
        return []

    def roll_dice(self, user_input=None):
        if user_input:
            try:
                val=int(user_input)
                if 1<=val<=4:
                    base_val=val
                else:
                    base_val=random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]
            except:
                base_val=random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]
        else:
            base_val=random.choices([1,2,3,4], weights=self.dice_probs, k=1)[0]

        dist= base_val + self.enhance_count
        if self.awaken_next_turn:
            dist*=3
            self.awaken_next_turn=False
        return dist

    def step(self, user_input=None):
        if self.remaining_actions<=0 or not self.agent_pos:
            return (self.agent_pos, 0, True, 0)

        self.remaining_actions-=1
        dist=self.roll_dice(user_input)

        if self.path and self.start and self.end:
            new_idx=min(self.path_idx+dist, len(self.path)-1)
            moved=new_idx-self.path_idx
            self.path_idx=new_idx
            self.agent_pos=self.path[new_idx]
            self.last_move_distance=moved
        else:
            self.last_move_distance=0

        self.resolve_tile_effects()

        done=False
        rew=0
        if self.agent_pos==self.end:
            done=True
            rew=1
        if self.remaining_actions<=0:
            done=True

        return (self.agent_pos, rew, done, dist)

    def resolve_tile_effects(self):
        while True:
            if not self.agent_pos:
                break
            tile=self.special_tiles.get(self.agent_pos,None)
            if not tile:
                break
            moved=False

            if tile=="전개":
                if self.path:
                    old_idx=self.path_idx
                    new_idx=min(self.path_idx+4, len(self.path)-1)
                    self.path_idx=new_idx
                    self.agent_pos=self.path[new_idx]
                    self.last_move_distance=(new_idx-old_idx)
                moved=True

            elif tile=="강화":
                self.enhance_count+=1
                break

            elif tile=="각성":
                self.awaken_next_turn=True
                break

            elif tile=="복제":
                if self.last_move_distance>0 and self.path:
                    old_idx=self.path_idx
                    new_idx=min(self.path_idx+self.last_move_distance, len(self.path)-1)
                    self.path_idx=new_idx
                    self.agent_pos=self.path[new_idx]
                    self.last_move_distance=(new_idx-old_idx)
                moved=True

            elif tile=="정화":
                self.purification_move()
                moved=True

            if not moved:
                break

    def purification_move(self):
        if self.simulation_mode:
            dist=self.roll_dice(None)
        else:
            dist=tk.simpledialog.askinteger("정화","정화로 몇칸 이동?")
            if not dist or dist<1:
                dist=self.roll_dice(None)

        if self.path:
            old_idx=self.path_idx
            new_idx=min(self.path_idx+dist, len(self.path)-1)
            self.path_idx=new_idx
            self.agent_pos=self.path[new_idx]
            self.last_move_distance=(new_idx-old_idx)

    def place_special_tile(self, r,c,tile_type):
        if self.grid is not None:
            if 0<=r<8 and 0<=c<8:
                if self.grid[r,c] in [1,2,3]:
                    self.special_tiles[(r,c)] = tile_type

    def place_agent(self, r,c):
        if self.grid is not None:
            if 0<=r<8 and 0<=c<8:
                if self.grid[r,c] in [1,2,3]:
                    self.agent_pos=(r,c)
                    self.start=(r,c)
                    if (self.current_part,6) in predefined_paths:
                        self.path= predefined_paths[(self.current_part,6)]
                        self.using_predefined=True
                    else:
                        self.path= self.find_path()
                        self.using_predefined=False
                    self.path_idx=0

    def clone_environment(self):
        new_env=TranscendenceEnv.__new__(TranscendenceEnv)
        new_env.map_data=self.map_data
        new_env.grid=np.copy(self.grid)
        new_env.max_actions=self.max_actions
        new_env.remaining_actions=self.remaining_actions
        new_env.start=self.start
        new_env.end=self.end
        new_env.agent_pos=self.agent_pos
        new_env.path=list(self.path)
        new_env.path_idx=self.path_idx
        new_env.special_tiles=dict(self.special_tiles)

        new_env.enhance_count=self.enhance_count
        new_env.awaken_next_turn=self.awaken_next_turn
        new_env.last_move_distance=self.last_move_distance
        new_env.dice_probs=list(self.dice_probs)
        new_env.current_part=self.current_part
        new_env.gaho=self.gaho
        new_env.using_predefined=self.using_predefined

        new_env.simulation_mode=True
        return new_env

    def monte_carlo_success_probability(self, num_sim=1000):
        if self.agent_pos==self.end:
            return 1.0
        if self.remaining_actions<=0:
            return 0.0
        success_count=0
        for _ in range(num_sim):
            sim_env=self.clone_environment()
            done=False
            rew=0
            while not done:
                _, reward, done, _= sim_env.step()
                if done:
                    rew=reward
            if rew==1:
                success_count+=1
        return success_count/float(num_sim)

    def get_next_dice_info(self):
        lines=[]
        for i,p in enumerate(self.dice_probs):
            base_val=i+1
            dist= base_val+self.enhance_count
            if self.awaken_next_turn:
                dist*=3
            lines.append(f"{base_val}({p*100:.1f}%) => {dist}")
        return "\n".join(lines)

    ###########################
    # State info
    ###########################
    def get_remaining_distance_to_end(self):
        """predefined_paths인 경우 => (len(path)-1) - path_idx
           BFS인 경우 => (len(path)-1) - path_idx
           (둘 다 동일 actually).
        """
        if self.path and self.start and self.end and self.agent_pos:
            return max(0, (len(self.path)-1)- self.path_idx)
        return None

    def has_passed_tile(self, r,c):
        if not self.path:
            return False
        if (r,c) in self.path:
            idx=self.path.index((r,c))
            if idx< self.path_idx:
                return True
        return False

    def get_distance_to_tile(self, tile_r, tile_c):
        """
        if using_predefined:
            - 타일이 self.path에 있으면 => tile_idx - path_idx
            - 없으면 => None
        else:
            => BFS
        """
        if self.using_predefined:
            if not self.path:
                return None
            if (tile_r,tile_c) in self.path:
                tile_idx= self.path.index((tile_r,tile_c))
                if tile_idx< self.path_idx:
                    # 이미 지나침
                    return None
                else:
                    return tile_idx - self.path_idx
            else:
                # path에 없음 => 도달 불가
                return None
        else:
            # BFS
            return self.bfs_distance((tile_r,tile_c))

    def bfs_distance(self, tile_pos):
        if not self.agent_pos or not tile_pos:
            return None
        if self.agent_pos==tile_pos:
            return 0

        from collections import deque
        def can_walk(rr,cc):
            return (0<=rr<8 and 0<=cc<8 and self.grid[rr,cc] in [1,2,3])

        visited=set([self.agent_pos])
        queue=deque([(self.agent_pos,0)])
        while queue:
            (rr,cc), dist= queue.popleft()
            if (rr,cc)==tile_pos:
                return dist
            for (dr,dc) in [(1,0),(-1,0),(0,1),(0,-1)]:
                nr,nc= rr+dr, cc+dc
                if can_walk(nr,nc) and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append(((nr,nc), dist+1))
        return None

    def print_state_info(self):
        print("===== Current State =====")
        # 남은 액션
        print(f"남은 액션 수: {self.remaining_actions}")
        # 엘조윈
        print(f"엘조윈 가호: {self.gaho}")
        # 강화
        print(f"강화 수: {self.enhance_count}")
        # 각성
        print(f"각성 여부: {self.awaken_next_turn}")
        # 남은 거리(끝점)
        dist_end= self.get_remaining_distance_to_end()
        if dist_end is None:
            print("남은 거리(끝점): ???(도달불가)")
        else:
            print(f"남은 거리(끝점): {dist_end}")

        # 특수 타일 (이미 지나친 것은 제외)
        tile_list=[]
        for (rr,cc), tile_type in self.special_tiles.items():
            if self.has_passed_tile(rr,cc):
                continue
            d= self.get_distance_to_tile(rr,cc)
            if d is None:
                dist_str= "도달 불가"
            else:
                dist_str= f"n={d}"
            tile_list.append((tile_type, rr,cc, dist_str))

        if not tile_list:
            print("특수 타일: 없음(이미 지나침/불가)")
        else:
            print("특수 타일:")
            for (tp, rr,cc,dstr) in tile_list:
                print(f" - {tp}@({rr},{cc}), {dstr}")
        print("=========================")

########################
# Main App
########################
class TranscendenceApp:
    def __init__(self, root, json_path):
        self.root=root
        self.root.title("PredefPaths => tile distance by path-index; BFS otherwise")

        self.env=TranscendenceEnv(json_path)

        self.parts=["투구","어깨","상의","하의","장갑","무기"]
        self.stages=list(range(1,8))
        self.gaho_levels=list(range(0,9))

        frame_top=tk.Frame(root)
        frame_top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(frame_top, text="부위:").pack(side=tk.LEFT)
        self.combo_part=ttk.Combobox(frame_top, values=self.parts, width=8)
        self.combo_part.pack(side=tk.LEFT, padx=5)
        self.combo_part.current(0)

        tk.Label(frame_top, text="단계:").pack(side=tk.LEFT)
        self.combo_stage=ttk.Combobox(frame_top, values=self.stages, width=4)
        self.combo_stage.pack(side=tk.LEFT, padx=5)
        self.combo_stage.current(0)

        tk.Label(frame_top, text="엘조윈 가호:").pack(side=tk.LEFT)
        self.combo_gaho=ttk.Combobox(frame_top, values=self.gaho_levels, width=4)
        self.combo_gaho.pack(side=tk.LEFT, padx=5)
        self.combo_gaho.current(0)

        tk.Label(frame_top, text="MC:").pack(side=tk.LEFT)
        self.entry_monte=tk.Entry(frame_top, width=6)
        self.entry_monte.pack(side=tk.LEFT, padx=5)
        self.entry_monte.insert(0,"1000")

        btn_set=tk.Button(frame_top, text="카테고리 확정", command=self.on_set_category)
        btn_set.pack(side=tk.LEFT, padx=10)

        self.label_actions=tk.Label(frame_top, text="남은 액션:0/0")
        self.label_actions.pack(side=tk.LEFT, padx=10)

        frame_mid=tk.Frame(root)
        frame_mid.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(frame_mid, text="주사위(0=확률만,1~4=이동):").pack(side=tk.LEFT)
        self.entry_dice=tk.Entry(frame_mid, width=5)
        self.entry_dice.pack(side=tk.LEFT, padx=5)

        btn_calc=tk.Button(frame_mid, text="계산", command=self.on_calculate)
        btn_calc.pack(side=tk.LEFT, padx=5)

        self.label_dice_info=tk.Label(frame_mid, text="(주사위)")
        self.label_dice_info.pack(side=tk.LEFT, padx=10)

        self.label_probability=tk.Label(frame_mid, text="성공확률:???")
        self.label_probability.pack(side=tk.LEFT, padx=10)

        self.label_recommend=tk.Label(frame_mid, text="(추천)")
        self.label_recommend.pack(side=tk.LEFT, padx=10)

        frame_bottom=tk.Frame(root)
        frame_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.canvas_map=tk.Canvas(frame_bottom, width=400, height=400, bg="white")
        self.canvas_map.pack(side=tk.LEFT, padx=5, pady=5)

        tile_img= Image.open("tile.png").resize((50,50), Resampling.LANCZOS)
        self.img_tile=ImageTk.PhotoImage(tile_img)

        self.special_img={}
        for nm in ["전개","강화","각성","복제","정화"]:
            im=Image.open(f"{nm}.png").resize((50,50),Resampling.LANCZOS)
            self.special_img[nm]=ImageTk.PhotoImage(im)
        self.special_img["에이전트"]=None

        self.frame_tiles=tk.Frame(frame_bottom, width=150, height=400, bg="lightblue")
        self.frame_tiles.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.special_tile_types=["전개","강화","각성","복제","정화","에이전트"]
        self.tile_labels=[]
        y_offset=10
        for stype in self.special_tile_types:
            if stype in self.special_img and self.special_img[stype]:
                lbl=tk.Label(self.frame_tiles, text=stype,
                             image=self.special_img[stype],
                             compound='top',
                             bg="white", width=60, height=60, bd=2, relief="ridge")
            else:
                lbl=tk.Label(self.frame_tiles, text=stype,
                             bg="white", width=8, height=3, bd=2, relief="ridge")
            lbl.place(x=10, y=y_offset)
            y_offset+=70

            lbl.bind("<Button-1>", self.on_tile_drag_start)
            lbl.bind("<B1-Motion>", self.on_tile_drag_move)
            lbl.bind("<ButtonRelease-1>", self.on_tile_drag_release)

            self.tile_labels.append(lbl)

        # drag data
        self.drag_data={
            "tile_type":None,
            "offset_x":0,"offset_y":0,
            "win":None,"lbl":None
        }

        self.draw_map()

    def parse_montecarlo(self):
        t=self.entry_monte.get().strip()
        try:
            v=int(t)
            if v<1:
                v=1000
            return v
        except:
            return 1000

    def on_set_category(self):
        part=self.combo_part.get()
        st=int(self.combo_stage.get())
        gh=int(self.combo_gaho.get())

        self.env.set_category(part, st)
        self.env.set_gaho_level(gh)

        self.draw_map()
        self.update_actions_label()
        self.update_dice_info_display()

        n=self.parse_montecarlo()
        p=self.env.monte_carlo_success_probability(n)
        self.label_probability.config(text=f"성공확률:{p*100:.2f}%")
        r=decide_action(self.env,p)
        self.label_recommend.config(text=r)

    def on_calculate(self):
        self.env.simulation_mode=False
        dice_txt=self.entry_dice.get().strip()

        # 1) 만약 주사위 굴림 => step first
        done=False
        if dice_txt and dice_txt!="0":
            pos,reward,done,dist= self.env.step(dice_txt)

        # 2) 상태 출력(이미 변경된 상태)
        self.env.print_state_info()

        # 3) MC
        n=self.parse_montecarlo()
        p=self.env.monte_carlo_success_probability(n)
        self.label_probability.config(text=f"성공확률:{p*100:.2f}%")
        rec=decide_action(self.env,p)
        self.label_recommend.config(text=rec)

        if done:
            print("게임 종료")

        self.draw_map()
        self.update_actions_label()
        self.update_dice_info_display()

    def draw_map(self):
        self.canvas_map.delete("all")
        if self.env.grid is None or self.env.grid.size==0:
            return
        cs=50
        for r in range(8):
            for c in range(8):
                x=c*cs
                y=r*cs
                val=self.env.grid[r,c]
                if val==0:
                    self.canvas_map.create_rectangle(x,y,x+cs,y+cs, fill="black", outline="white")
                else:
                    self.canvas_map.create_image(x,y, anchor="nw", image=self.img_tile)

                if (r,c) in self.env.special_tiles:
                    stype=self.env.special_tiles[(r,c)]
                    if stype in self.special_img and self.special_img[stype]:
                        self.canvas_map.create_image(x,y, anchor="nw", image=self.special_img[stype])

                if (r,c)==self.env.start:
                    self.canvas_map.create_text(x+25,y+25,text="O", fill="white", font=("Arial",14,"bold"))
                if (r,c)==self.env.end:
                    self.canvas_map.create_text(x+25,y+25,text="X", fill="white", font=("Arial",14,"bold"))
                if self.env.agent_pos==(r,c):
                    self.canvas_map.create_text(x+25,y+25,text="A", fill="yellow", font=("Arial",14,"bold"))

    def update_actions_label(self):
        self.label_actions.config(
            text=f"남은 액션:{self.env.remaining_actions}/{self.env.max_actions}"
        )

    def update_dice_info_display(self):
        info=self.env.get_next_dice_info()
        self.label_dice_info.config(text=f"주사위:\n{info}")

    def on_tile_drag_start(self, event):
        w=event.widget
        tile_type= w.cget("text")

        self.drag_data["tile_type"]=tile_type
        self.drag_data["offset_x"]= event.x
        self.drag_data["offset_y"]= event.y

        win=tk.Toplevel(self.root)
        win.withdraw()
        win.overrideredirect(True)
        win.lift()
        win.attributes("-topmost",True)

        lbl=tk.Label(win,bd=0,bg="white")
        if tile_type in self.env.special_tiles:
            lbl.config(image=self.env.special_tiles[tile_type])
        if tile_type in self.special_img and self.special_img[tile_type]:
            lbl.config(image=self.special_img[tile_type])
        else:
            lbl.config(text=tile_type,width=1,height=1)
        lbl.pack()

        mx= event.x_root
        my= event.y_root
        w_x= mx- self.drag_data["offset_x"]
        w_y= my- self.drag_data["offset_y"]
        win.geometry(f"+{w_x}+{w_y}")
        win.deiconify()

        self.drag_data["win"]=win
        self.drag_data["lbl"]=lbl

    def on_tile_drag_move(self, event):
        if not self.drag_data["win"]:
            return
        dx= event.x_root - (self.drag_data["win"].winfo_x() + self.drag_data["offset_x"])
        dy= event.y_root - (self.drag_data["win"].winfo_y() + self.drag_data["offset_y"])
        w_x= self.drag_data["win"].winfo_x()+dx
        w_y= self.drag_data["win"].winfo_y()+dy
        self.drag_data["win"].geometry(f"+{w_x}+{w_y}")

    def on_tile_drag_release(self, event):
        tile_type= self.drag_data["tile_type"]
        win= self.drag_data["win"]
        if not win:
            return

        c_x= self.canvas_map.winfo_rootx()
        c_y= self.canvas_map.winfo_rooty()
        mx= event.x_root
        my= event.y_root

        cs=50
        row= int((my-c_y)//cs)
        col= int((mx-c_x)//cs)

        if 0<=row<8 and 0<=col<8:
            if tile_type=="에이전트":
                self.env.place_agent(row,col)
            else:
                self.env.place_special_tile(row,col, tile_type)
            self.draw_map()

        win.destroy()
        self.drag_data={
            "tile_type":None,
            "offset_x":0,"offset_y":0,
            "win":None,"lbl":None
        }

if __name__=="__main__":
    JSON_FILE_PATH= "transcendence_maps.json"
    root=tk.Tk()
    app=TranscendenceApp(root, JSON_FILE_PATH)
    root.mainloop()
