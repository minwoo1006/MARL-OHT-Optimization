"""
utils/visualization.py
───────────────────────────
3주차 팀원 3 담당: Pygame 기반 OHT 팹 실시간 시각화 (메가 팹 대응)

기능 목록:
  [기본]
  - stall_count에 따라 OHT 색상 변화 (정상=초록, 정체=노랑→주황→빨강)
  - Spine / Bay / Port / Stocker 노드 구분 렌더링
  - 우측 사이드바에 에이전트 상태 패널 표시

  [2주차 유지]
  - SPACE: 일시정지 / 재개
  - ←→: 일시정지 중 스텝 이동
  - R: 녹화 시작/중지
  - 충돌 발생 노드 빨간 번쩍임 이펙트
  - Throughput / Collision 실시간 라인차트
  - 노드 통행량 히트맵 오버레이

  [3주차 신규]
  - 줌 인/아웃 (+/-키, 마우스휠)
  - 구역 이동 pan (WASD 또는 화살표키 — 재생 중)
  - Hot Lot OHT 빨간 점멸 표시 (is_hot_lot 기반)
  - Stocker 노드 초록 사각형 표시
  - 미니맵: 우측 하단에 전체 맵 축소판 + 현재 뷰포트 위치 표시

  조작키:
    SPACE       : 일시정지 / 재개
    ←→          : 일시정지 중 스텝 이동
    WASD / ↑↓←→ : 맵 이동 (pan)
    +/-          : 줌 인/아웃
    마우스휠     : 줌 인/아웃
    R           : 녹화 시작/중지
    ESC         : 종료
"""

import pygame
import sys
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field

# ──────────────────────────────────────────────
# 상수 및 색상
# ──────────────────────────────────────────────

SIDEBAR_WIDTH = 300
WIN_W = 1400
WIN_H = 860
MAP_W = WIN_W - SIDEBAR_WIDTH   # 맵 영역 너비
MAP_H = WIN_H - 80              # 맵 영역 높이 (하단 상태바 제외)

# 줌 설정
CELL_SIZE_DEFAULT = 20          # 기본 셀 크기 (메가팹 대응)
CELL_SIZE_MIN     = 4
CELL_SIZE_MAX     = 90
PAN_SPEED         = 30          # 한 번 이동 픽셀

# 미니맵
MINIMAP_W = 200
MINIMAP_H = 120
MINIMAP_MARGIN = 10

# Hot Lot 점멸
HOT_LOT_BLINK_PERIOD = 20       # 점멸 주기 (프레임)

C_BG          = (18,  18,  30)
C_RAIL_SPINE  = (220, 80,  80)
C_RAIL_BAY    = (80, 140, 220)
C_NODE        = (60,  60,  90)
C_PORT        = (255, 215,  0)
C_STOCKER     = (100, 220, 100)  # ✅ [신규] Stocker 노드 색
C_HOT_LOT     = (255, 60,  60)  # ✅ [신규] Hot Lot 강조색
C_TEXT        = (220, 220, 220)
C_TEXT_DIM    = (120, 120, 140)
C_SIDEBAR_BG  = (28,  28,  45)
C_PANEL_LINE  = (50,  50,  75)
C_MINIMAP_BG  = (20,  20,  38)
C_VIEWPORT    = (255, 220, 80)

OHT_COLORS = [
    (100, 220, 255), (140, 255, 140), (255, 200, 100), (200, 140, 255),
    (255, 130, 170), (100, 255, 200), (255, 255, 100), (160, 200, 255),
    (255, 160, 100), (180, 255, 180),
]

CHART_H        = 80
FLASH_DURATION = 8
RECORD_DIR     = "recordings"


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

def stall_color(stall: int) -> tuple:
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


# ──────────────────────────────────────────────
# 스냅샷
# ──────────────────────────────────────────────

@dataclass
class FrameSnapshot:
    step:             int
    agent_positions:  dict
    agent_targets:    dict
    agent_states:     dict
    agent_priorities: dict       # ✅ [신규] Hot Lot 우선순위
    stall_counters:   dict
    loading_timers:   dict
    delivery_count:   int
    collision_count:  int
    infos:            dict
    collision_nodes:  set = field(default_factory=set)


# ──────────────────────────────────────────────
# 메인 시각화 클래스
# ──────────────────────────────────────────────

class OHTVisualizer:

    def __init__(self, env, fps: int = 6, cell_size: int = None):
        self.env  = env
        self.fps  = fps

        # 그래프에서 맵 범위 계산
        nodes = list(env.graph.nodes())
        self._map_cols = max(n[0] for n in nodes) + 1
        self._map_rows = max(n[1] for n in nodes) + 1

        # 줌 / 팬
        self._cell_size = cell_size or self._auto_cell_size()
        self._pan_x = 0   # 픽셀 오프셋
        self._pan_y = 0
        self._frame_count = 0  # 점멸 계산용

        self.screen  = None
        self.clock   = None
        self.font_sm = None
        self.font_md = None
        self.font_lg = None
        self._agent_colors: dict = {}

        # 일시정지 / 스냅샷
        self._paused     = False
        self._snapshots: list = []
        self._snap_idx   = -1

        # 충돌 이펙트
        self._flash_nodes: dict = {}

        # 실시간 차트
        self._throughput_hist: list = []
        self._collision_hist:  list = []

        # 히트맵
        self._visit_counts: dict = defaultdict(int)

        # 녹화
        self._recording  = False
        self._frame_idx  = 0

    def _auto_cell_size(self) -> int:
        """맵 크기에 맞게 셀 크기 자동 결정"""
        cs_w = MAP_W // self._map_cols
        cs_h = MAP_H // self._map_rows
        return max(CELL_SIZE_MIN, min(cs_w, cs_h, CELL_SIZE_MAX))

    # ── 좌표 변환 ──

    def _grid_to_px(self, x: int, y: int) -> tuple:
        """그리드 좌표 → 화면 픽셀 (팬/줌 적용)"""
        px = self._pan_x + x * self._cell_size + self._cell_size // 2
        py = self._pan_y + y * self._cell_size + self._cell_size // 2
        return px, py

    def _is_visible(self, px: int, py: int, margin: int = 20) -> bool:
        """픽셀 좌표가 맵 영역 안에 있는지 확인 (렌더링 최적화)"""
        return (-margin <= px <= MAP_W + margin and
                -margin <= py <= MAP_H + margin)

    # ──────────────────────────────────────────
    # 초기화 / 종료
    # ──────────────────────────────────────────

    def init(self):
        pygame.init()
        self.screen  = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption(
            "🏭 MARL OHT Mega-Fab  |  SPACE=일시정지  WASD=이동  +/-=줌  R=녹화  ESC=종료"
        )
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 12)
        self.font_md = pygame.font.SysFont("consolas", 14, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 17, bold=True)

        for i, agent_id in enumerate(self.env.possible_agents):
            self._agent_colors[agent_id] = OHT_COLORS[i % len(OHT_COLORS)]

        if not os.path.exists(RECORD_DIR):
            os.makedirs(RECORD_DIR)

        # 초기 팬: 맵 중앙이 화면 중앙에 오도록
        self._center_map()

    def _center_map(self):
        total_w = self._map_cols * self._cell_size
        total_h = self._map_rows * self._cell_size
        self._pan_x = (MAP_W - total_w) // 2
        self._pan_y = (MAP_H - total_h) // 2

    def close(self):
        if self._recording:
            print(f"\n📁 녹화 프레임 {self._frame_idx}장 → '{RECORD_DIR}/'")
            print(f"  ffmpeg -r {self.fps} -i {RECORD_DIR}/frame_%05d.png -vcodec libx264 output.mp4")
        pygame.quit()

    # ──────────────────────────────────────────
    # 외부 호출 메서드
    # ──────────────────────────────────────────

    def detect_collisions(self, prev_positions: dict, env) -> set:
        pos_count = defaultdict(int)
        for agent_id in env.agents:
            pos_count[env.agent_positions[agent_id]] += 1
        return {node for node, cnt in pos_count.items() if cnt > 1}

    def push_snapshot(self, step: int, infos: dict, collision_nodes: set = None):
        env = self.env
        snap = FrameSnapshot(
            step             = step,
            agent_positions  = env.agent_positions.copy(),
            agent_targets    = env.agent_targets.copy(),
            agent_states     = env.agent_states.copy(),
            agent_priorities = env.agent_priorities.copy(),  # ✅ [신규]
            stall_counters   = env.stall_counters.copy(),
            loading_timers   = env.loading_timers.copy(),
            delivery_count   = env.delivery_count,
            collision_count  = env.collision_count,
            infos            = {k: dict(v) for k, v in infos.items()},
            collision_nodes  = collision_nodes or set(),
        )
        self._snapshots.append(snap)
        self._snap_idx = len(self._snapshots) - 1

        for node in (collision_nodes or set()):
            self._flash_nodes[node] = FLASH_DURATION

        self._throughput_hist.append(env.delivery_count)
        self._collision_hist.append(env.collision_count)

        for agent_id in env.agents:
            self._visit_counts[env.agent_positions[agent_id]] += 1

    def render(self) -> bool:
        if self._snap_idx < 0:
            return True

        self._frame_count += 1

        # 재생 중 이벤트
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self._paused = True
                    print(f"  ⏸ 일시정지 (Step {self._current_snap.step})")
                elif event.key == pygame.K_r:
                    self._recording = not self._recording
                    print(f"  {'🔴 녹화 시작' if self._recording else '⏹ 녹화 중지'}")
                # ✅ [신규] 팬 이동
                elif event.key in (pygame.K_w, pygame.K_UP):
                    self._pan_y += PAN_SPEED
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    self._pan_y -= PAN_SPEED
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    self._pan_x += PAN_SPEED
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    self._pan_x -= PAN_SPEED
                # ✅ [신규] 줌
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    self._zoom(1)
                elif event.key == pygame.K_MINUS:
                    self._zoom(-1)
            # ✅ [신규] 마우스휠 줌
            elif event.type == pygame.MOUSEWHEEL:
                self._zoom(event.y)

        # 일시정지 루프
        while self._paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    elif event.key == pygame.K_SPACE:
                        self._paused = False
                        print("  ▶ 재개")
                    elif event.key == pygame.K_RIGHT:
                        self._snap_idx = min(self._snap_idx + 1, len(self._snapshots) - 1)
                    elif event.key == pygame.K_LEFT:
                        self._snap_idx = max(self._snap_idx - 1, 0)
                    elif event.key == pygame.K_r:
                        self._recording = not self._recording
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self._zoom(1)
                    elif event.key == pygame.K_MINUS:
                        self._zoom(-1)
                elif event.type == pygame.MOUSEWHEEL:
                    self._zoom(event.y)
            self._draw_frame(paused=True)
            pygame.display.flip()
            self.clock.tick(30)

        self._draw_frame(paused=False)

        for node in list(self._flash_nodes):
            self._flash_nodes[node] -= 1
            if self._flash_nodes[node] <= 0:
                del self._flash_nodes[node]

        if self._recording:
            path = os.path.join(RECORD_DIR, f"frame_{self._frame_idx:05d}.png")
            pygame.image.save(self.screen, path)
            self._frame_idx += 1

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def _zoom(self, direction: int):
        """줌 인/아웃. 맵 중심 기준으로 스케일 변경."""
        old_cs = self._cell_size
        new_cs = max(CELL_SIZE_MIN, min(CELL_SIZE_MAX, self._cell_size + direction * 2))
        if new_cs == old_cs:
            return
        # 화면 중심 기준으로 팬 조정
        cx, cy = MAP_W // 2, MAP_H // 2
        scale = new_cs / old_cs
        self._pan_x = int(cx - (cx - self._pan_x) * scale)
        self._pan_y = int(cy - (cy - self._pan_y) * scale)
        self._cell_size = new_cs

    # ──────────────────────────────────────────
    # 내부 드로잉
    # ──────────────────────────────────────────

    @property
    def _current_snap(self) -> FrameSnapshot:
        return self._snapshots[self._snap_idx]

    def _draw_frame(self, paused: bool):
        snap = self._current_snap
        self.screen.fill(C_BG)

        # 맵 클리핑 영역 설정
        map_surface = pygame.Surface((MAP_W, MAP_H))
        map_surface.fill(C_BG)

        self._draw_heatmap(map_surface)
        self._draw_rails(map_surface)
        self._draw_nodes(map_surface)
        self._draw_collision_flash(map_surface)
        self._draw_agents(map_surface, snap)

        self.screen.blit(map_surface, (0, 0))

        self._draw_minimap(snap)       # ✅ [신규] 미니맵
        self._draw_sidebar(snap)
        self._draw_statusbar(snap, paused)

    def _draw_rails(self, surf):
        cs = self._cell_size
        for u, v, data in self.env.graph.edges(data=True):
            x1, y1 = self._grid_to_px(*u)
            x2, y2 = self._grid_to_px(*v)

            # 화면 밖이면 스킵 (성능 최적화)
            if not (self._is_visible(x1, y1) or self._is_visible(x2, y2)):
                continue

            edge_type = data.get("edge_type", "bay")
            color = C_RAIL_SPINE if edge_type == "spine" else C_RAIL_BAY
            width = max(1, min(4, cs // 8)) if edge_type == "spine" else max(1, cs // 12)

            if edge_type == "spine":
                dx, dy = y2-y1, x1-x2
                length = math.hypot(dx, dy) or 1
                off = max(2, cs // 10)
                ox, oy = int(dx/length*off), int(dy/length*off)
                x1, y1, x2, y2 = x1+ox, y1+oy, x2+ox, y2+oy

            pygame.draw.line(surf, color, (x1, y1), (x2, y2), width)
            if cs >= 10:
                self._draw_arrowhead(surf, x1, y1, x2, y2, color, size=max(4, cs//6))

    def _draw_arrowhead(self, surf, x1, y1, x2, y2, color, size=6):
        dx, dy = x2-x1, y2-y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx/length, dy/length
        tip_x, tip_y = x2 - ux*size, y2 - uy*size
        pts = [
            (tip_x, tip_y),
            (tip_x - ux*size + uy*size*0.5, tip_y - uy*size - ux*size*0.5),
            (tip_x - ux*size - uy*size*0.5, tip_y - uy*size + ux*size*0.5),
        ]
        pygame.draw.polygon(surf, color, pts)

    def _draw_nodes(self, surf):
        cs = self._cell_size
        node_r = max(2, cs // 4)

        for node, data in self.env.graph.nodes(data=True):
            px, py = self._grid_to_px(*node)
            if not self._is_visible(px, py):
                continue

            is_port    = data.get("is_port", False)
            is_stocker = data.get("is_stocker", False)  # ✅ [신규]

            if is_port:
                rect = pygame.Rect(px - node_r, py - node_r, node_r*2, node_r*2)
                pygame.draw.rect(surf, C_PORT, rect, border_radius=2)
                if cs >= 20:
                    lbl = self.font_sm.render(f"{node[0]},{node[1]}", True, (30, 30, 30))
                    surf.blit(lbl, (px - lbl.get_width()//2, py - lbl.get_height()//2))
            elif is_stocker:
                # ✅ [신규] Stocker: 초록 사각형
                rect = pygame.Rect(px - node_r, py - node_r, node_r*2, node_r*2)
                pygame.draw.rect(surf, C_STOCKER, rect, border_radius=1)
                pygame.draw.rect(surf, (60, 160, 60), rect, 1, border_radius=1)
            else:
                pygame.draw.circle(surf, C_NODE, (px, py), max(2, node_r - 1))

    def _draw_collision_flash(self, surf):
        cs = self._cell_size
        for node, remaining in self._flash_nodes.items():
            px, py  = self._grid_to_px(*node)
            if not self._is_visible(px, py):
                continue
            ratio  = remaining / FLASH_DURATION
            radius = int(cs//2 + (1 - ratio) * cs//4)
            alpha  = int(200 * ratio)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 50, 50, alpha), (radius, radius), radius)
            surf.blit(s, (px - radius, py - radius))

    def _draw_heatmap(self, surf):
        if not self._visit_counts:
            return
        cs = self._cell_size
        max_v = max(self._visit_counts.values()) or 1
        for node, count in self._visit_counts.items():
            px, py = self._grid_to_px(*node)
            if not self._is_visible(px, py):
                continue
            ratio  = count / max_v
            radius = int(cs//4 + ratio * cs//3)
            if ratio < 0.5:
                r = int(40  + ratio * 2 * 180)
                g = int(100 + ratio * 2 * 80)
                b = int(200 - ratio * 2 * 160)
            else:
                r = int(220 + (ratio - 0.5) * 2 * 35)
                g = int(180 - (ratio - 0.5) * 2 * 160)
                b = 40
            alpha = int(30 + ratio * 70)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (r, g, b, alpha), (radius, radius), radius)
            surf.blit(s, (px - radius, py - radius))

    def _draw_agents(self, surf, snap: FrameSnapshot):
        cs = self._cell_size
        radius = max(4, cs // 2 - 1)

        pos_count = defaultdict(int)
        for aid in self.env.possible_agents:
            if aid in snap.agent_positions:
                pos_count[snap.agent_positions[aid]] += 1

        pos_idx = defaultdict(int)
        for aid in self.env.possible_agents:
            if aid not in snap.agent_positions:
                continue
            pos = snap.agent_positions[aid]
            idx = pos_idx[pos]
            pos_idx[pos] += 1

            px, py = self._grid_to_px(*pos)
            if not self._is_visible(px, py, margin=radius+5):
                continue

            if pos_count[pos] > 1:
                angle = (2 * math.pi / pos_count[pos]) * idx
                px   += int(math.cos(angle) * radius)
                py   += int(math.sin(angle) * radius)

            base   = self._agent_colors[aid]
            stall  = snap.infos.get(aid, {}).get("stall_count", 0)
            sc     = stall_color(stall)
            is_hot = snap.agent_priorities.get(aid, 0) == 1  # ✅ [신규]

            if snap.agent_states.get(aid, 0) == 1:
                base = tuple(int(c * 0.5) for c in base)

            # ✅ [신규] Hot Lot 점멸: 짝수 프레임에 빨간 테두리 강조
            if is_hot:
                blink_on = (self._frame_count % HOT_LOT_BLINK_PERIOD) < (HOT_LOT_BLINK_PERIOD // 2)
                border_color = C_HOT_LOT if blink_on else (200, 200, 200)
                border_w = max(3, cs // 8)
            else:
                border_color = sc
                border_w = 2

            pygame.draw.circle(surf, base, (px, py), radius)
            pygame.draw.circle(surf, border_color, (px, py), radius, border_w)

            # ✅ [신규] Hot Lot이면 불꽃 마커
            if is_hot and cs >= 12:
                flame = self.font_sm.render("🔥", True, C_HOT_LOT)
                surf.blit(flame, (px - flame.get_width()//2, py - radius - flame.get_height()))

            # 에이전트 번호
            if cs >= 10:
                lbl = self.font_md.render(aid.split("_")[-1], True, (20, 20, 20))
                surf.blit(lbl, (px - lbl.get_width()//2, py - lbl.get_height()//2))

            # 목적지 점선
            if aid in snap.agent_targets:
                tx, ty = self._grid_to_px(*snap.agent_targets[aid])
                self._draw_dashed_line(surf, px, py, tx, ty, border_color)

    def _draw_dashed_line(self, surf, x1, y1, x2, y2, color, dash=6):
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
            pygame.draw.line(surf, color, (sx_, sy_), (ex_, ey_), 1)

    # ✅ [신규] 미니맵
    def _draw_minimap(self, snap: FrameSnapshot):
        mx = MAP_W - MINIMAP_W - MINIMAP_MARGIN
        my = MAP_H - MINIMAP_H - MINIMAP_MARGIN

        # 배경
        pygame.draw.rect(self.screen, C_MINIMAP_BG,
                         (mx-2, my-2, MINIMAP_W+4, MINIMAP_H+4), border_radius=4)
        pygame.draw.rect(self.screen, C_PANEL_LINE,
                         (mx-2, my-2, MINIMAP_W+4, MINIMAP_H+4), 1, border_radius=4)

        scale_x = MINIMAP_W / self._map_cols
        scale_y = MINIMAP_H / self._map_rows

        # 레일 (spine만 표시)
        for u, v, data in self.env.graph.edges(data=True):
            if data.get("edge_type") != "spine":
                continue
            x1 = mx + int(u[0] * scale_x)
            y1 = my + int(u[1] * scale_y)
            x2 = mx + int(v[0] * scale_x)
            y2 = my + int(v[1] * scale_y)
            pygame.draw.line(self.screen, (120, 50, 50), (x1, y1), (x2, y2), 1)

        # 에이전트 점
        for aid in self.env.possible_agents:
            if aid not in snap.agent_positions:
                continue
            pos    = snap.agent_positions[aid]
            is_hot = snap.agent_priorities.get(aid, 0) == 1
            color  = C_HOT_LOT if is_hot else self._agent_colors[aid]
            px = mx + int(pos[0] * scale_x)
            py = my + int(pos[1] * scale_y)
            pygame.draw.circle(self.screen, color, (px, py), 2)

        # 현재 뷰포트 표시
        vp_x = mx + int(-self._pan_x * scale_x / self._cell_size)
        vp_y = my + int(-self._pan_y * scale_y / self._cell_size)
        vp_w = int(MAP_W * scale_x / self._cell_size)
        vp_h = int(MAP_H * scale_y / self._cell_size)
        pygame.draw.rect(self.screen, C_VIEWPORT,
                         (vp_x, vp_y, vp_w, vp_h), 1)

        # 레이블
        lbl = self.font_sm.render("MiniMap", True, C_TEXT_DIM)
        self.screen.blit(lbl, (mx, my - 14))

    def _draw_sidebar(self, snap: FrameSnapshot):
        sx = MAP_W
        pygame.draw.rect(self.screen, C_SIDEBAR_BG, (sx, 0, SIDEBAR_WIDTH, WIN_H))
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx, 0), (sx, WIN_H), 1)

        self.screen.blit(self.font_lg.render("OHT Status", True, C_TEXT), (sx+12, 12))

        # ✅ [신규] 줌 레벨 표시
        zoom_str = f"Zoom: {self._cell_size}px"
        self.screen.blit(self.font_sm.render(zoom_str, True, C_TEXT_DIM), (sx+180, 16))

        y = 42
        hot_count = sum(1 for aid in self.env.possible_agents
                        if snap.agent_priorities.get(aid, 0) == 1)
        for label, value in [
            ("Step",      str(snap.step)),
            ("Delivery",  str(snap.delivery_count)),
            ("Collision", str(snap.collision_count)),
            ("🔥 HotLot", str(hot_count)),   # ✅ [신규]
        ]:
            self.screen.blit(self.font_sm.render(f"{label}:", True, C_TEXT_DIM), (sx+12, y))
            self.screen.blit(self.font_md.render(value, True, C_TEXT), (sx+120, y))
            y += 18

        y += 6
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx+8, y), (sx+SIDEBAR_WIDTH-8, y), 1)
        y += 8

        for aid in self.env.possible_agents:
            if aid not in snap.agent_positions:
                continue
            color  = self._agent_colors[aid]
            stall  = snap.infos.get(aid, {}).get("stall_count", 0)
            sc     = stall_color(stall)
            state  = snap.agent_states.get(aid, 0)
            pos    = snap.agent_positions.get(aid)
            tgt    = snap.agent_targets.get(aid)
            timer  = snap.loading_timers.get(aid, 0)
            is_hot = snap.agent_priorities.get(aid, 0) == 1  # ✅ [신규]

            # ✅ [신규] Hot Lot이면 점멸 원
            dot_color = C_HOT_LOT if (is_hot and self._frame_count % HOT_LOT_BLINK_PERIOD < HOT_LOT_BLINK_PERIOD//2) else color
            pygame.draw.circle(self.screen, dot_color, (sx+18, y+7), 7)
            pygame.draw.circle(self.screen, sc,        (sx+18, y+7), 7, 2)

            name_str = f"{'🔥' if is_hot else ''}{aid}"
            self.screen.blit(self.font_md.render(name_str, True, C_TEXT), (sx+30, y))

            if state == 1:
                st_str, st_col = f"LOADING ({timer}/5)", (180, 180, 80)
            elif stall >= 10:
                st_str, st_col = f"DEADLOCK! ({stall}/15)", (255, 60, 60)
            elif stall >= 5:
                st_str, st_col = f"STALL ({stall}/15)", (255, 140, 0)
            elif is_hot:
                st_str, st_col = "HOT LOT 🔥", C_HOT_LOT
            else:
                st_str, st_col = "MOVING", (100, 220, 100)

            self.screen.blit(self.font_sm.render(st_str, True, st_col), (sx+30, y+16))
            self.screen.blit(self.font_sm.render(f"{pos}→{tgt}", True, C_TEXT_DIM), (sx+10, y+30))

            bw = SIDEBAR_WIDTH - 22
            pygame.draw.rect(self.screen, (50, 50, 70), (sx+10, y+44, bw, 5), border_radius=2)
            fw = int(bw * stall / 15)
            if fw > 0:
                pygame.draw.rect(self.screen, sc, (sx+10, y+44, fw, 5), border_radius=2)

            y += 62
            if y + 62 > WIN_H - CHART_H - 55:
                break

        self._draw_chart(sx, WIN_H - CHART_H - 42)

    def _draw_chart(self, sx: int, cy: int):
        cw = SIDEBAR_WIDTH - 20
        pygame.draw.rect(self.screen, (22, 22, 38), (sx+10, cy, cw, CHART_H), border_radius=4)
        pygame.draw.rect(self.screen, C_PANEL_LINE, (sx+10, cy, cw, CHART_H), 1, border_radius=4)
        self.screen.blit(self.font_sm.render("Throughput", True, (100, 220, 255)), (sx+12, cy+2))
        self.screen.blit(self.font_sm.render("Collision",  True, (255, 100, 100)), (sx+120, cy+2))

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

    def _draw_statusbar(self, snap: FrameSnapshot, paused: bool):
        by = WIN_H - 40
        pygame.draw.line(self.screen, C_PANEL_LINE, (0, by), (WIN_W, by), 1)

        x = 10
        for color, label in [
            (C_PORT,         "Port"),
            (C_STOCKER,      "Stocker"),    # ✅ [신규]
            (C_RAIL_SPINE,   "Spine"),
            (C_RAIL_BAY,     "Bay"),
            ((80, 220, 80),  "Normal"),
            ((255, 140, 0),  "Stall"),
            ((255, 60, 60),  "Deadlock"),
            (C_HOT_LOT,      "HotLot🔥"),  # ✅ [신규]
        ]:
            pygame.draw.rect(self.screen, color, (x, by+12, 12, 12), border_radius=2)
            surf = self.font_sm.render(label, True, C_TEXT_DIM)
            self.screen.blit(surf, (x+15, by+13))
            x += surf.get_width() + 28

        if paused:
            ps = self.font_md.render(
                f"⏸ PAUSED  ← {self._snap_idx}/{len(self._snapshots)-1} →",
                True, (255, 220, 80)
            )
            self.screen.blit(ps, (MAP_W//2 - ps.get_width()//2, by+12))

        if self._recording:
            rs = self.font_md.render(f"● REC {self._frame_idx}", True, (255, 60, 60))
            self.screen.blit(rs, (MAP_W - rs.get_width() - 10, by+12))


# ──────────────────────────────────────────────
# 단독 실행 (테스트)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from envs.oht_env import OHTFabEnv
    from agents.dijkstra_baseline import DijkstraBaselineAgent

    # 작은 맵으로 테스트
    env = OHTFabEnv(
        num_ohts=5,
        max_steps=300,
        width=30, height=20, bay_interval=8, bay_depth=4
    )
    env.reset()

    viz      = OHTVisualizer(env, fps=6)
    viz.init()
    baseline = DijkstraBaselineAgent(env.graph)

    for step in range(300):
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