"""
Pygame 기반 OHT 팹 실시간 시각화

  - 맵 크기 10x6 (Spine-and-Bay) 대응
  - stall_count에 따라 OHT 색상 변화 (정상=초록, 정체=노랑→주황→빨강)
  - Spine / Bay / Port 노드 구분 렌더링
  - 우측 사이드바에 에이전트 상태 패널 표시
"""

import pygame
import sys
import math

# 상수 및 색상 정의
# 그리드 설정 (10x6)
GRID_COLS = 10
GRID_ROWS = 6
CELL_SIZE = 90       # 셀 하나의 픽셀 크기
MARGIN     = 50      # 맵 여백

# 사이드바
SIDEBAR_WIDTH = 280

# 전체 윈도우 크기
WIN_W = GRID_COLS * CELL_SIZE + MARGIN * 2 + SIDEBAR_WIDTH
WIN_H = GRID_ROWS * CELL_SIZE + MARGIN * 2 + 80  # 하단 상태바 공간

# 색상
C_BG          = (18,  18,  30)   # 배경 (다크 네이비)
C_RAIL_SPINE  = (220, 80,  80)   # Spine 레일 (RED)
C_RAIL_BAY    = (80, 140, 220)   # Bay 레일 (BLUE)
C_NODE        = (60,  60,  90)   # 일반 노드
C_PORT        = (255, 215,  0)   # 포트 노드 (금색)
C_SPINE_NODE  = (200, 100, 100)  # Spine 노드
C_TEXT        = (220, 220, 220)
C_TEXT_DIM    = (120, 120, 140)
C_SIDEBAR_BG  = (28,  28,  45)
C_PANEL_LINE  = (50,  50,  75)

# OHT 색상 팔레트 (에이전트 ID별)
OHT_COLORS = [
    (100, 220, 255),  # 하늘
    (140, 255, 140),  # 연두
    (255, 200, 100),  # 오렌지
    (200, 140, 255),  # 보라
    (255, 130, 170),  # 핑크
    (100, 255, 200),  # 민트
    (255, 255, 100),  # 노랑
    (160, 200, 255),  # 하늘
    (255, 160, 100),  # 살구
    (180, 255, 180),  # 연녹
]

# stall_count에 따른 경고 색상
def stall_color(stall: int) -> tuple:
    """stall_count 0~15를 초록→노랑→주황→빨강으로 변환"""
    if stall == 0:
        return (80, 220, 80)      # 정상: 초록
    ratio = min(stall / 15.0, 1.0)
    if ratio < 0.5:
        # 초록 → 노랑
        r = int(80  + (255 - 80)  * (ratio / 0.5))
        g = int(220)
        b = int(80  * (1 - ratio / 0.5))
    else:
        # 노랑 → 빨강
        r = 255
        g = int(220 * (1 - (ratio - 0.5) / 0.5))
        b = 0
    return (r, g, b)


# 좌표 변환 
def grid_to_px(x: int, y: int) -> tuple[int, int]:
    """그리드 좌표 (x, y) → 화면 픽셀 중심 좌표"""
    px = MARGIN + x * CELL_SIZE + CELL_SIZE // 2
    py = MARGIN + y * CELL_SIZE + CELL_SIZE // 2
    return px, py


# 시각화 

class OHTVisualizer:   #OHTFabEnv를 받아 Pygame으로 실시간 렌더링

    def __init__(self, env, fps: int = 6):
        self.env = env
        self.fps = fps
        self.screen = None
        self.clock  = None
        self.font_sm = None
        self.font_md = None
        self.font_lg = None
        self._agent_colors: dict[str, tuple] = {}

    def init(self):   # 에피소드 시작 전 호출해서 Pygame 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("🏭 MARL OHT Fab Visualizer — Spine-and-Bay")
        self.clock   = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        # 에이전트별 고정 색상 배정
        for i, agent_id in enumerate(self.env.possible_agents):
            self._agent_colors[agent_id] = OHT_COLORS[i % len(OHT_COLORS)]

    def render(self, step: int, infos: dict = None):
        """
        step:  현재 스텝 번호
        infos: env.step()이 반환한 infos dic.
        """
        self._last_infos = infos or {}
        # 창 닫기
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

        self.screen.fill(C_BG)

        self._draw_rails()
        self._draw_nodes()
        self._draw_agents()
        self._draw_sidebar(step)
        self._draw_statusbar(step)

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):  # Pygame 종료
        pygame.quit()

    
    # 시각화 METHOD
    def _draw_rails(self): #그래프 엣지 화살표
        graph = self.env.graph
        for u, v, data in graph.edges(data=True):
            edge_type = data.get("edge_type", "bay")
            color = C_RAIL_SPINE if edge_type == "spine" else C_RAIL_BAY
            width = 4 if edge_type == "spine" else 2

            x1, y1 = grid_to_px(*u)
            x2, y2 = grid_to_px(*v)

            # 겹치는 선 구분
            if edge_type == "spine":
                offset = 6
                dx = y2 - y1  # 수직 방향으로 오프셋
                dy = x1 - x2
                length = math.hypot(dx, dy) or 1
                ox = int(dx / length * offset)
                oy = int(dy / length * offset)
                x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy

            pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)

            # 화살표 머리
            self._draw_arrowhead(x1, y1, x2, y2, color, size=8)

    def _draw_arrowhead(self, x1, y1, x2, y2, color, size=8): # 라인 끝에 삼각형 화살표를 그립니다.
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length  # 단위 벡터
        tip_x = x2 - ux * 12
        tip_y = y2 - uy * 12
        lx = tip_x - ux * size + uy * size * 0.5
        ly = tip_y - uy * size - ux * size * 0.5
        rx = tip_x - ux * size - uy * size * 0.5
        ry = tip_y - uy * size + ux * size * 0.5
        pygame.draw.polygon(self.screen, color, [(tip_x, tip_y), (lx, ly), (rx, ry)])

    def _draw_nodes(self): #노드
        graph = self.env.graph
        for node, data in graph.nodes(data=True):
            px, py = grid_to_px(*node)
            is_port = data.get("is_port", False)

            if is_port:
                # 포트
                rect = pygame.Rect(px - 14, py - 14, 28, 28)
                pygame.draw.rect(self.screen, C_PORT, rect, border_radius=4)
                pygame.draw.rect(self.screen, (200, 160, 0), rect, 2, border_radius=4)
                label = self.font_sm.render(f"{node[0]},{node[1]}", True, (30, 30, 30))
                self.screen.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))
            else:
                # 일반 노드
                pygame.draw.circle(self.screen, C_NODE, (px, py), 8)
                pygame.draw.circle(self.screen, (90, 90, 120), (px, py), 8, 1)

    def _draw_agents(self): # OHT Agent
        env = self.env
        agent_list = list(env.agents)

        # 같은 노드에 여러 에이전트가 있을 때 겹침 방지
        pos_count: dict = {}
        for agent_id in agent_list:
            pos = env.agent_positions[agent_id]
            pos_count[pos] = pos_count.get(pos, 0) + 1

        pos_idx: dict = {}
        for agent_id in agent_list:
            pos = env.agent_positions[agent_id]
            idx = pos_idx.get(pos, 0)
            pos_idx[pos] = idx + 1

            px, py = grid_to_px(*pos)
            count = pos_count[pos]
            if count > 1:
                angle = (2 * math.pi / count) * idx
                px += int(math.cos(angle) * 18)
                py += int(math.sin(angle) * 18)

            # 색상 결정 (infos에서 stall_count 우선 참조)
            base_color   = self._agent_colors[agent_id]
            stall        = self._last_infos.get(agent_id, {}).get("stall_count", 0)
            border_color = stall_color(stall)

            # LOADING 상태면 반투명 효과 (밝기 낮춤)
            if env.agent_states.get(agent_id, 0) == 1:
                base_color = tuple(int(c * 0.5) for c in base_color)

            # 원 그리기
            radius = 18
            pygame.draw.circle(self.screen, base_color, (px, py), radius)
            pygame.draw.circle(self.screen, border_color, (px, py), radius, 3)

            # 에이전트 번호
            num = agent_id.split("_")[-1]
            label = self.font_md.render(num, True, (20, 20, 20))
            self.screen.blit(label, (px - label.get_width() // 2, py - label.get_height() // 2))

            # 목적지까지 점선
            target = env.agent_targets[agent_id]
            tx, ty = grid_to_px(*target)
            self._draw_dashed_line(px, py, tx, ty, border_color, dash=6)

    def _draw_dashed_line(self, x1, y1, x2, y2, color, dash=8): #점선
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        step = dash * 2
        for i in range(0, int(length), step):
            sx = int(x1 + ux * i)
            sy = int(y1 + uy * i)
            ex = int(x1 + ux * min(i + dash, length))
            ey = int(y1 + uy * min(i + dash, length))
            pygame.draw.line(self.screen, color, (sx, sy), (ex, ey), 1)

    def _draw_sidebar(self, step: int): # 에이전트별 상태 패널.
        env = self.env
        sx = GRID_COLS * CELL_SIZE + MARGIN * 2
        sy = 0

        # 사이드바 배경
        pygame.draw.rect(self.screen, C_SIDEBAR_BG, (sx, sy, SIDEBAR_WIDTH, WIN_H))
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx, 0), (sx, WIN_H), 1)

        # 타이틀
        title = self.font_lg.render("OHT Status", True, C_TEXT)
        self.screen.blit(title, (sx + 12, 14))

        # 전체 지표
        y_cursor = 48
        metrics = [
            ("Step",      f"{step}"),
            ("Delivery",  f"{env.delivery_count}"),
            ("Collision", f"{env.collision_count}"),
        ]
        for label, value in metrics:
            lbl_surf = self.font_sm.render(f"{label}:", True, C_TEXT_DIM)
            val_surf = self.font_md.render(value, True, C_TEXT)
            self.screen.blit(lbl_surf, (sx + 12, y_cursor))
            self.screen.blit(val_surf, (sx + 105, y_cursor))
            y_cursor += 20

        # 구분선
        y_cursor += 6
        pygame.draw.line(self.screen, C_PANEL_LINE, (sx + 8, y_cursor), (sx + SIDEBAR_WIDTH - 8, y_cursor), 1)
        y_cursor += 10

        # 에이전트별 패널
        for agent_id in env.possible_agents:
            if agent_id not in env.agents:
                continue

            color  = self._agent_colors[agent_id]
            stall  = self._last_infos.get(agent_id, {}).get("stall_count", 0)
            sc     = stall_color(stall)
            state  = env.agent_states.get(agent_id, 0)
            pos    = env.agent_positions.get(agent_id, (-1, -1))
            target = env.agent_targets.get(agent_id, (-1, -1))
            timer  = env.loading_timers.get(agent_id, 0)

            # 에이전트 색 동그라미
            pygame.draw.circle(self.screen, color, (sx + 20, y_cursor + 8), 8)
            pygame.draw.circle(self.screen, sc,    (sx + 20, y_cursor + 8), 8, 2)

            # 이름
            name_surf = self.font_md.render(agent_id, True, C_TEXT)
            self.screen.blit(name_surf, (sx + 34, y_cursor))

            # 상태 문자열
            if state == 1:
                state_str = f"LOADING ({timer}/5)"
                state_col = (180, 180, 80)
            elif stall >= 10:
                state_str = f"DEADLOCK! ({stall}/15)"
                state_col = (255, 60, 60)
            elif stall >= 5:
                state_str = f"STALL ({stall}/15)"
                state_col = (255, 140, 0)
            else:
                state_str = "MOVING"
                state_col = (100, 220, 100)

            st_surf = self.font_sm.render(state_str, True, state_col)
            self.screen.blit(st_surf, (sx + 34, y_cursor + 18))

            # 위치 & 목적지
            loc_str = f"{pos} → {target}"
            loc_surf = self.font_sm.render(loc_str, True, C_TEXT_DIM)
            self.screen.blit(loc_surf, (sx + 12, y_cursor + 34))

            # stall 바 (진행바)
            bar_x, bar_y = sx + 12, y_cursor + 50
            bar_w = SIDEBAR_WIDTH - 24
            bar_h = 6
            pygame.draw.rect(self.screen, (50, 50, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
            fill_w = int(bar_w * stall / 15)
            if fill_w > 0:
                pygame.draw.rect(self.screen, sc, (bar_x, bar_y, fill_w, bar_h), border_radius=3)

            y_cursor += 72

            # 오버플로우 방지
            if y_cursor + 72 > WIN_H - 60:
                break

    def _draw_statusbar(self, step: int):  # 상태바
        by = WIN_H - 40
        pygame.draw.line(self.screen, C_PANEL_LINE, (0, by), (WIN_W, by), 1)

        legends = [
            (C_PORT,        "Port"),
            (C_RAIL_SPINE,  "Spine"),
            (C_RAIL_BAY,    "Bay"),
            ((80, 220, 80), "Normal"),
            ((255, 140, 0), "Stall"),
            ((255, 60, 60), "Deadlock"),
        ]
        x = 16
        for color, label in legends:
            pygame.draw.rect(self.screen, color, (x, by + 12, 14, 14), border_radius=2)
            surf = self.font_sm.render(label, True, C_TEXT_DIM)
            self.screen.blit(surf, (x + 18, by + 13))
            x += surf.get_width() + 38


# 테스트

if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from envs.oht_env import OHTFabEnv
    from agents.dijkstra_baseline import DijkstraBaselineAgent

    NUM_OHTS = 5
    MAX_STEPS = 300

    env = OHTFabEnv(num_ohts=NUM_OHTS, max_steps=MAX_STEPS)
    env.reset()

    viz = OHTVisualizer(env, fps=6)
    viz.init()

    baseline = DijkstraBaselineAgent(env.graph)

    for step in range(MAX_STEPS):
        if not env.agents:
            break

        actions = {agent_id: baseline.get_action(env, agent_id) for agent_id in env.agents}
        _, _, _, _, infos = env.step(actions)

        viz.render(step, infos)

    viz.close()
    print("✅ 시각화 종료")
