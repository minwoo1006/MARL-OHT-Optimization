"""
  조작키:
    SPACE  : 일시정지 / 재개
    →      : 일시정지 중 다음 프레임
    ←      : 일시정지 중 이전 프레임 (스냅샷 기반)
    Mouse  : 드래그로 화면 이동, 휠로 확대/축소
    + / -  : 확대 / 축소
    F      : 전체 맵 맞춤
    R      : 녹화 시작 / 중지
    ESC    : 종료
"""

import pygame
import sys
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field


# 상수 및 색상 정의
GRID_COLS     = 10
GRID_ROWS     = 6
CELL_SIZE     = 90
MARGIN        = 50
SIDEBAR_WIDTH = 280

WIN_W = GRID_COLS * CELL_SIZE + MARGIN * 2 + SIDEBAR_WIDTH
WIN_H = GRID_ROWS * CELL_SIZE + MARGIN * 2 + 80

C_BG         = (18,  18,  30)
C_RAIL_SPINE = (220, 80,  80)
C_RAIL_BAY   = (80, 140, 220)
C_NODE       = (60,  60,  90)
C_PORT       = (255, 215,  0)
C_TEXT       = (220, 220, 220)
C_TEXT_DIM   = (120, 120, 140)
C_SIDEBAR_BG = (28,  28,  45)
C_PANEL_LINE = (50,  50,  75)

OHT_COLORS = [
    (100, 220, 255), (140, 255, 140), (255, 200, 100), (200, 140, 255),
    (255, 130, 170), (100, 255, 200), (255, 255, 100), (160, 200, 255),
    (255, 160, 100), (180, 255, 180),
]

CHART_H        = 80
FLASH_DURATION = 8
RECORD_DIR     = "recordings"

# 헬퍼 함수
def stall_color(stall: int) -> tuple:
    """stall_count 0~15 → 초록→노랑→주황→빨강"""
    if stall == 0:
        return (80, 220, 80)
    ratio = min(stall / 15.0, 1.0)
    if ratio < 0.5:
        r = int(80 + (255 - 80) * (ratio / 0.5))
        g = 220
        b = int(80 * (1 - ratio / 0.5))
    else:
        r = 255
        g = int(220 * (1 - (ratio - 0.5) / 0.5))
        b = 0
    return (r, g, b)


def grid_to_px(x: int, y: int) -> tuple:
    px = MARGIN + x * CELL_SIZE + CELL_SIZE // 2
    py = MARGIN + y * CELL_SIZE + CELL_SIZE // 2
    return px, py


# 스텝 이동용

@dataclass
class FrameSnapshot:
    """매 스텝 환경 상태를 저장하는 스냅샷"""
    step:            int
    agent_positions: dict
    agent_targets:   dict
    agent_states:    dict
    stall_counters:  dict
    loading_timers:  dict
    delivery_count:  int
    collision_count: int
    infos:           dict
    collision_nodes: set = field(default_factory=set)



# 시각화 클래스
class OHTVisualizer:
    """
    OHTFabEnv를 받아 Pygame으로 실시간 렌더링합니다.

    사용 예:
        viz = OHTVisualizer(env)
        viz.init()
        for step in range(max_steps):
            prev_positions = env.agent_positions.copy()
            _, _, _, _, infos = env.step(actions)
            collision_nodes = viz.detect_collisions(prev_positions, env)
            viz.push_snapshot(step, infos, collision_nodes)
            if not viz.render():
                break
        viz.close()
    """

    def __init__(self, env, fps: int = 6):
        self.env  = env
        self.fps  = fps

        self.screen  = None
        self.clock   = None
        self.font_sm = None
        self.font_md = None
        self.font_lg = None

        self._agent_colors: dict = {}

        # [2] 일시정지 / 스텝 조작
        self._paused     = False
        self._snapshots: list = []
        self._snap_idx   = -1

        # [3] 충돌 이펙트: {node: 남은 프레임}
        self._flash_nodes: dict = {}

        # [4] 실시간 차트
        self._throughput_hist: list = []
        self._collision_hist:  list = []

        # [5] 히트맵: {node: 누적 방문 횟수}
        self._visit_counts: dict = defaultdict(int)

        # [6] 녹화
        self._recording  = False
        self._frame_idx  = 0
        self._dragging = False
        self._drag_origin = (0, 0)
        self._pan_origin = (0, 0)

        self._configure_view()

    def _configure_view(self):
        nodes = list(self.env.graph.nodes())
        xs = [node[0] for node in nodes]
        ys = [node[1] for node in nodes]
        self._min_x = min(xs)
        self._min_y = min(ys)
        span_x = max(xs) - self._min_x + 1
        span_y = max(ys) - self._min_y + 1
        draw_w = GRID_COLS * CELL_SIZE
        draw_h = GRID_ROWS * CELL_SIZE
        self._fit_cell_size = max(1.5, min(CELL_SIZE, draw_w / span_x, draw_h / span_y))
        self._cell_size = self._fit_cell_size
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._node_radius = max(2, min(8, int(self._cell_size * 0.35)))
        self._port_size = max(4, min(28, int(self._cell_size * 1.2)))
        self._agent_radius = max(4, min(18, int(self._cell_size * 0.8)))
        self._agent_offset = max(4, min(18, int(self._cell_size * 0.8)))

    def _set_zoom(self, zoom, anchor=None):
        old_cell = self._cell_size
        old_zoom = self._zoom
        self._zoom = max(0.5, min(16.0, zoom))
        self._cell_size = self._fit_cell_size * self._zoom

        if anchor is not None and old_zoom != self._zoom:
            ax, ay = anchor
            map_x = (ax - MARGIN - self._pan_x) / old_cell + self._min_x
            map_y = (ay - MARGIN - self._pan_y) / old_cell + self._min_y
            self._pan_x = ax - MARGIN - (map_x - self._min_x) * self._cell_size
            self._pan_y = ay - MARGIN - (map_y - self._min_y) * self._cell_size

        self._node_radius = max(2, min(8, int(self._cell_size * 0.35)))
        self._port_size = max(4, min(28, int(self._cell_size * 1.2)))
        self._agent_radius = max(4, min(18, int(self._cell_size * 0.8)))
        self._agent_offset = max(4, min(18, int(self._cell_size * 0.8)))

    def _reset_view(self):
        self._pan_x = 0
        self._pan_y = 0
        self._set_zoom(1.0)

    def _grid_to_px(self, x: int, y: int) -> tuple:
        px = MARGIN + self._pan_x + (x - self._min_x) * self._cell_size + self._cell_size / 2
        py = MARGIN + self._pan_y + (y - self._min_y) * self._cell_size + self._cell_size / 2
        return int(px), int(py)


    # 초기화 / 종료
    def init(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(
            "MARL OHT Fab | wheel=zoom drag=pan +/-=zoom F=fit SPACE=pause R=rec ESC=quit"
        )
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        for i, agent_id in enumerate(self.env.possible_agents):
            self._agent_colors[agent_id] = OHT_COLORS[i % len(OHT_COLORS)]

        if not os.path.exists(RECORD_DIR):
            os.makedirs(RECORD_DIR)

    def close(self):
        if self._recording:
            print(f"\n📁 녹화 프레임 {self._frame_idx}장 저장 → '{RECORD_DIR}/'")
            print(f"  ffmpeg -r {self.fps} -i {RECORD_DIR}/frame_%05d.png -vcodec libx264 output.mp4")
        pygame.quit()


    # 외부 호출 메서드─

    def detect_collisions(self, prev_positions: dict, env) -> set:
        """충돌 발생 노드를 반환합니다. env.step() 직후 호출하세요."""
        pos_count = defaultdict(int)
        for agent_id in env.agents:
            pos_count[env.agent_positions[agent_id]] += 1
        return {node for node, cnt in pos_count.items() if cnt > 1}

    def push_snapshot(self, step: int, infos: dict, collision_nodes: set = None):
        """현재 env 상태를 스냅샷으로 저장합니다. render() 전에 호출하세요."""
        env = self.env
        snap = FrameSnapshot(
            step            = step,
            agent_positions = env.agent_positions.copy(),
            agent_targets   = env.agent_targets.copy(),
            agent_states    = env.agent_states.copy(),
            stall_counters  = env.stall_counters.copy(),
            loading_timers  = env.loading_timers.copy(),
            delivery_count  = env.delivery_count,
            collision_count = env.collision_count,
            infos           = {k: dict(v) for k, v in infos.items()},
            collision_nodes = collision_nodes or set(),
        )
        self._snapshots.append(snap)
        self._snap_idx = len(self._snapshots) - 1

        # [3] 충돌 이펙트 등록
        for node in (collision_nodes or set()):
            self._flash_nodes[node] = FLASH_DURATION

        # [4] 차트 데이터 누적
        self._throughput_hist.append(env.delivery_count)
        self._collision_hist.append(env.collision_count)

        # [5] 히트맵 카운트
        for agent_id in env.agents:
            self._visit_counts[env.agent_positions[agent_id]] += 1

    def render(self) -> bool:
        """
        현재 스냅샷 기준으로 한 프레임을 그립니다.
        Returns: False이면 종료 요청
        """
        if self._snap_idx < 0:
            return True

        # 재생 중 이벤트 (SPACE, R, ESC만 — ←→는 일시정지 루프에서 처리)
        for event in pygame.event.get():
            handled = self._handle_common_event(event, allow_pause=True)
            if handled == "quit":
                return False

        # 일시정지 중: 내부에서 키 입력 대기 (바깥 루프 진행 차단)
        while self._paused:
            for event in pygame.event.get():
                handled = self._handle_common_event(event, allow_pause=False)
                if handled == "quit":
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._paused = False
                        print("  ▶ 재개")
                    elif event.key == pygame.K_RIGHT:
                        self._snap_idx = min(self._snap_idx + 1, len(self._snapshots) - 1)
                    elif event.key == pygame.K_LEFT:
                        self._snap_idx = max(self._snap_idx - 1, 0)
            self._draw_frame(paused=True)
            pygame.display.flip()
            self.clock.tick(30)

        self._draw_frame(paused=False)

        # [3] 이펙트 카운터 감소
        for node in list(self._flash_nodes):
            self._flash_nodes[node] -= 1
            if self._flash_nodes[node] <= 0:
                del self._flash_nodes[node]

        # [6] 녹화
        if self._recording:
            path = os.path.join(RECORD_DIR, f"frame_{self._frame_idx:05d}.png")
            pygame.image.save(self.screen, path)
            self._frame_idx += 1

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def wait_until_closed(self):
        """Keep the last rendered frame open for inspection."""
        if self._snap_idx < 0:
            return
        print("  Episode ended. Inspect the final frame, then press ESC or close the window.")
        while True:
            for event in pygame.event.get():
                handled = self._handle_common_event(event, allow_pause=False)
                if handled == "quit":
                    return
            self._draw_frame(paused=False)
            pygame.display.flip()
            self.clock.tick(30)

    def _handle_common_event(self, event, allow_pause=True):
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._dragging = True
            self._drag_origin = event.pos
            self._pan_origin = (self._pan_x, self._pan_y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging = False
        elif event.type == pygame.MOUSEMOTION and self._dragging:
            dx = event.pos[0] - self._drag_origin[0]
            dy = event.pos[1] - self._drag_origin[1]
            self._pan_x = self._pan_origin[0] + dx
            self._pan_y = self._pan_origin[1] + dy
        elif event.type == pygame.MOUSEWHEEL:
            anchor = pygame.mouse.get_pos()
            factor = 1.15 if event.y > 0 else 1 / 1.15
            self._set_zoom(self._zoom * factor, anchor=anchor)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "quit"
            if allow_pause and event.key == pygame.K_SPACE:
                self._paused = True
                print(f"  pause (Step {self._current_snap.step})")
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                self._set_zoom(self._zoom * 1.15, anchor=(WIN_W // 2, WIN_H // 2))
            elif event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                self._set_zoom(self._zoom / 1.15, anchor=(WIN_W // 2, WIN_H // 2))
            elif event.key == pygame.K_f:
                self._reset_view()
            elif event.key == pygame.K_r:
                self._recording = not self._recording
                print(f"  {'REC start' if self._recording else 'REC stop'}")
        return None

    # 내부 드로잉
    @property
    def _current_snap(self) -> FrameSnapshot:
        return self._snapshots[self._snap_idx]

    def _draw_frame(self, paused: bool):
        snap = self._current_snap
        self.screen.fill(C_BG)
        self._draw_heatmap()
        self._draw_rails()
        self._draw_nodes()
        self._draw_collision_flash()
        self._draw_agents(snap)
        self._draw_sidebar(snap)
        self._draw_statusbar(snap, paused)

    def _draw_rails(self):
        for u, v, data in self.env.graph.edges(data=True):
            edge_type = data.get("edge_type", "bay")
            color = C_RAIL_SPINE if edge_type == "spine" else C_RAIL_BAY
            width = max(1, min(4 if edge_type == "spine" else 2, int(self._cell_size / 3)))
            x1, y1 = self._grid_to_px(*u)
            x2, y2 = self._grid_to_px(*v)
            if edge_type == "spine":
                dx, dy = y2-y1, x1-x2
                length = math.hypot(dx, dy) or 1
                ox, oy = int(dx/length*min(6, self._cell_size * 0.25)), int(dy/length*min(6, self._cell_size * 0.25))
                x1, y1, x2, y2 = x1+ox, y1+oy, x2+ox, y2+oy
            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)
            self._draw_arrowhead(x1, y1, x2, y2, color, size=max(3, min(8, int(self._cell_size * 0.45))))

    def _draw_arrowhead(self, x1, y1, x2, y2, color, size=8):
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx/length, dy/length
        tip_x, tip_y = x2 - ux*12, y2 - uy*12
        pts = [
            (tip_x, tip_y),
            (tip_x - ux*size + uy*size*0.5, tip_y - uy*size - ux*size*0.5),
            (tip_x - ux*size - uy*size*0.5, tip_y - uy*size + ux*size*0.5),
        ]
        pygame.draw.polygon(self.screen, color, pts)

    def _draw_nodes(self):
        for node, data in self.env.graph.nodes(data=True):
            px, py = self._grid_to_px(*node)
            if data.get("is_port"):
                half = self._port_size // 2
                rect = pygame.Rect(px-half, py-half, self._port_size, self._port_size)
                pygame.draw.rect(self.screen, C_PORT, rect, border_radius=4)
                pygame.draw.rect(self.screen, (200, 160, 0), rect, 2, border_radius=4)
                if self._cell_size >= 18:
                    lbl = self.font_sm.render(f"{node[0]},{node[1]}", True, (30, 30, 30))
                    self.screen.blit(lbl, (px - lbl.get_width()//2, py - lbl.get_height()//2))
            else:
                pygame.draw.circle(self.screen, C_NODE, (px, py), self._node_radius)
                pygame.draw.circle(self.screen, (90, 90, 120), (px, py), self._node_radius, 1)

    # ── [3] 충돌 이펙트 ──

    def _draw_collision_flash(self):
        for node, remaining in self._flash_nodes.items():
            px, py  = self._grid_to_px(*node)
            ratio   = remaining / FLASH_DURATION
            radius  = int(self._agent_radius + 10 + (1 - ratio) * 10)
            alpha   = int(220 * ratio)
            surf    = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (255, 50, 50, alpha), (radius, radius), radius)
            self.screen.blit(surf, (px - radius, py - radius))
            s = 10
            pygame.draw.line(self.screen, (255, 80, 80), (px-s, py-s), (px+s, py+s), 3)
            pygame.draw.line(self.screen, (255, 80, 80), (px+s, py-s), (px-s, py+s), 3)

    # ── [5] 히트맵 ──

    def _draw_heatmap(self):
        if not self._visit_counts:
            return
        max_v = max(self._visit_counts.values()) or 1
        for node, count in self._visit_counts.items():
            px, py  = self._grid_to_px(*node)
            ratio   = count / max_v
            radius  = int(max(3, self._node_radius) + ratio * max(4, self._agent_radius))
            if ratio < 0.5:
                r = int(40  + ratio * 2 * 180)
                g = int(100 + ratio * 2 * 80)
                b = int(200 - ratio * 2 * 160)
            else:
                r = int(220 + (ratio - 0.5) * 2 * 35)
                g = int(180 - (ratio - 0.5) * 2 * 160)
                b = 40
            alpha = int(40 + ratio * 80)
            surf  = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (r, g, b, alpha), (radius, radius), radius)
            self.screen.blit(surf, (px - radius, py - radius))

    # ── 에이전트 ──

    def _draw_agents(self, snap: FrameSnapshot):
        pos_count = defaultdict(int)
        for aid in self.env.possible_agents:
            if aid in snap.agent_positions:
                pos_count[snap.agent_positions[aid]] += 1

        pos_idx = defaultdict(int)
        for aid in self.env.possible_agents:
            if aid not in snap.agent_positions:
                continue
            pos   = snap.agent_positions[aid]
            idx   = pos_idx[pos]
            pos_idx[pos] += 1

            px, py = self._grid_to_px(*pos)
            if pos_count[pos] > 1:
                angle = (2 * math.pi / pos_count[pos]) * idx
                px   += int(math.cos(angle) * self._agent_offset)
                py   += int(math.sin(angle) * self._agent_offset)

            base  = self._agent_colors[aid]
            stall = snap.infos.get(aid, {}).get("stall_count", 0)
            sc    = stall_color(stall)

            if snap.agent_states.get(aid, 0) == 1:
                base = tuple(int(c * 0.5) for c in base)

            pygame.draw.circle(self.screen, base, (px, py), self._agent_radius)
            pygame.draw.circle(self.screen, sc,   (px, py), self._agent_radius, max(1, min(3, self._agent_radius // 3)))

            if self._agent_radius >= 8:
                lbl = self.font_md.render(aid.split("_")[-1], True, (20, 20, 20))
                self.screen.blit(lbl, (px - lbl.get_width()//2, py - lbl.get_height()//2))

            tx, ty = self._grid_to_px(*snap.agent_targets[aid])
            self._draw_dashed_line(px, py, tx, ty, sc)

    def _draw_dashed_line(self, x1, y1, x2, y2, color, dash=6):
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx/length, dy/length
        for i in range(0, int(length), dash*2):
            sx_ = int(x1 + ux * i)
            sy_ = int(y1 + uy * i)
            ex_ = int(x1 + ux * min(i+dash, length))
            ey_ = int(y1 + uy * min(i+dash, length))
            pygame.draw.line(self.screen, color, (sx_, sy_), (ex_, ey_), 1)

    # ── 사이드바 ──

    def _draw_sidebar(self, snap: FrameSnapshot):
        sx = GRID_COLS * CELL_SIZE + MARGIN * 2
        pygame.draw.rect(self.screen, C_SIDEBAR_BG, (sx, 0, SIDEBAR_WIDTH, WIN_H))
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx, 0), (sx, WIN_H), 1)

        self.screen.blit(self.font_lg.render("OHT Status", True, C_TEXT), (sx+12, 14))

        y = 48
        for label, value in [
            ("Step",      str(snap.step)),
            ("Delivery",  str(snap.delivery_count)),
            ("Collision", str(snap.collision_count)),
        ]:
            self.screen.blit(self.font_sm.render(f"{label}:", True, C_TEXT_DIM), (sx+12, y))
            self.screen.blit(self.font_md.render(value, True, C_TEXT), (sx+105, y))
            y += 20

        y += 6
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx+8, y), (sx+SIDEBAR_WIDTH-8, y), 1)
        y += 10

        for aid in self.env.possible_agents:
            if aid not in snap.agent_positions:
                continue
            color = self._agent_colors[aid]
            stall = snap.infos.get(aid, {}).get("stall_count", 0)
            sc    = stall_color(stall)
            state = snap.agent_states.get(aid, 0)
            pos   = snap.agent_positions.get(aid)
            tgt   = snap.agent_targets.get(aid)
            timer = snap.loading_timers.get(aid, 0)

            pygame.draw.circle(self.screen, color, (sx+20, y+8), 8)
            pygame.draw.circle(self.screen, sc,    (sx+20, y+8), 8, 2)
            self.screen.blit(self.font_md.render(aid, True, C_TEXT), (sx+34, y))

            if state == 1:
                st_str, st_col = f"LOADING ({timer}/5)", (180, 180, 80)
            elif stall >= 10:
                st_str, st_col = f"DEADLOCK! ({stall}/15)", (255, 60, 60)
            elif stall >= 5:
                st_str, st_col = f"STALL ({stall}/15)", (255, 140, 0)
            else:
                st_str, st_col = "MOVING", (100, 220, 100)

            self.screen.blit(self.font_sm.render(st_str, True, st_col), (sx+34, y+18))
            self.screen.blit(self.font_sm.render(f"{pos} -> {tgt}", True, C_TEXT_DIM), (sx+12, y+34))

            bw = SIDEBAR_WIDTH - 24
            pygame.draw.rect(self.screen, (50, 50, 70), (sx+12, y+50, bw, 6), border_radius=3)
            fw = int(bw * stall / 15)
            if fw > 0:
                pygame.draw.rect(self.screen, sc, (sx+12, y+50, fw, 6), border_radius=3)

            y += 72
            if y + 72 > WIN_H - CHART_H - 60:
                break

        # [4] 라인차트
        self._draw_chart(sx, WIN_H - CHART_H - 45)

    # ── [4] 실시간 라인차트 ──

    def _draw_chart(self, sx: int, cy: int):
        cw = SIDEBAR_WIDTH - 20
        pygame.draw.rect(self.screen, (22, 22, 38), (sx+10, cy, cw, CHART_H), border_radius=4)
        pygame.draw.rect(self.screen, C_PANEL_LINE, (sx+10, cy, cw, CHART_H), 1, border_radius=4)

        self.screen.blit(self.font_sm.render("Throughput", True, (100, 220, 255)), (sx+12, cy+2))
        self.screen.blit(self.font_sm.render("Collision",  True, (255, 100, 100)), (sx+110, cy+2))

        def draw_line(history, color):
            if len(history) < 2:
                return
            max_val = max(history) or 1
            n       = min(len(history), cw - 4)
            pts     = history[-n:]
            coords  = [
                (sx + 12 + int(i * (cw-4) / max(len(pts)-1, 1)),
                 cy + CHART_H - 8 - int((v / max_val) * (CHART_H - 20)))
                for i, v in enumerate(pts)
            ]
            if len(coords) >= 2:
                pygame.draw.lines(self.screen, color, False, coords, 2)

        draw_line(self._throughput_hist, (100, 220, 255))
        draw_line(self._collision_hist,  (255, 100, 100))

    # ── 하단 상태바 ──

    def _draw_statusbar(self, snap: FrameSnapshot, paused: bool):
        by = WIN_H - 40
        pygame.draw.line(self.screen, C_PANEL_LINE, (0, by), (WIN_W, by), 1)

        x = 16
        for color, label in [
            (C_PORT,         "Port"),
            (C_RAIL_SPINE,   "Spine"),
            (C_RAIL_BAY,     "Bay"),
            ((80, 220, 80),  "Normal"),
            ((255, 140, 0),  "Stall"),
            ((255, 60, 60),  "Deadlock"),
            ((255, 50, 50),  "Collision"),
        ]:
            pygame.draw.rect(self.screen, color, (x, by+12, 14, 14), border_radius=2)
            surf = self.font_sm.render(label, True, C_TEXT_DIM)
            self.screen.blit(surf, (x+18, by+13))
            x += surf.get_width() + 30

        # [2] 일시정지 표시
        if paused:
            ps = self.font_md.render(
                f"PAUSED  frame {self._snap_idx} / {len(self._snapshots)-1}",
                True, (255, 220, 80)
            )
            self.screen.blit(ps, (WIN_W//2 - ps.get_width()//2, by+12))

        # [6] 녹화 표시
        if self._recording:
            rs = self.font_md.render(f"REC {self._frame_idx}", True, (255, 60, 60))
            self.screen.blit(rs, (WIN_W - rs.get_width() - 10, by+12))


#테스트
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from envs.oht_env import OHTFabEnv
    from agents.dijkstra_baseline import DijkstraBaselineAgent

    NUM_OHTS  = 5
    MAX_STEPS = 300

    env = OHTFabEnv(num_ohts=NUM_OHTS, max_steps=MAX_STEPS)
    env.reset()

    viz      = OHTVisualizer(env, fps=6)
    viz.init()
    baseline = DijkstraBaselineAgent(env.graph)

    for step in range(MAX_STEPS):
        if not env.agents:
            break

        prev_positions = env.agent_positions.copy()
        actions = {aid: baseline.get_action(env, aid) for aid in env.agents}
        _, _, _, _, infos = env.step(actions)

        collision_nodes = viz.detect_collisions(prev_positions, env)
        viz.push_snapshot(step, infos, collision_nodes)

        if not viz.render():
            break

    viz.close()
    print("✅ 시각화 종료")
