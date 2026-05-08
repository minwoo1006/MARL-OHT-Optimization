import networkx as nx
import matplotlib.pyplot as plt

def create_fab_graph():
    """
    반도체 팹(Fab)의 OHT 레일 네트워크를 유향 그래프(Directed Graph)로 생성합니다.
    노드 이름은 (x, y) 좌표를 사용합니다.
    """
    G = nx.DiGraph()

    # 1. 메인 루프 (Main Loop) 구성 - 시계 방향 일방통행
    # 상단 (Left to Right)
    main_loop_edges = [
        ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)),
        # 우측 (Top to Bottom)
        ((4, 0), (4, 1)), ((4, 1), (4, 2)), ((4, 2), (4, 3)), ((4, 3), (4, 4)),
        # 하단 (Right to Left)
        ((4, 4), (3, 4)), ((3, 4), (2, 4)), ((2, 4), (1, 4)), ((1, 4), (0, 4)),
        # 좌측 (Bottom to Top)
        ((0, 4), (0, 3)), ((0, 3), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0))
    ]
    G.add_edges_from(main_loop_edges, edge_type="main")

    # 2. 우회로 및 설비 진입로 (Bypass / Bay) - 위에서 아래로 일방통행
    bypass_edges = [
        ((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (2, 3)), ((2, 3), (2, 4))
    ]
    G.add_edges_from(bypass_edges, edge_type="bypass")

    # 3. 설비 포트(Port) 속성 추가
    # 웨이퍼를 싣고 내리는 특정 노드를 '포트'로 지정
    ports = [(1, 0), (3, 0), (2, 2), (1, 4), (3, 4)]
    for node in G.nodes():
        G.nodes[node]['is_port'] = True if node in ports else False

    return G

def draw_fab_graph(G):
    """
    생성된 그래프를 시각화합니다.
    """
    # 노드 이름 자체가 (x, y) 좌표이므로 위치(pos)로 그대로 사용
    pos = {node: (node[0], -node[1]) for node in G.nodes()} # y좌표는 화면 출력을 위해 뒤집음
    
    plt.figure(figsize=(8, 6))
    
    # 노드 색상 분리 (일반 노드: 회색, 포트 노드: 파란색)
    node_colors = ['#1f77b4' if G.nodes[n]['is_port'] else '#cccccc' for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, 
            font_size=10, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=20)
    
    plt.title("OHT Fab Rail Network Layout (NetworkX)", fontsize=16)
    plt.show()

if __name__ == "__main__":
    fab_graph = create_fab_graph()
    print(f"✅ 팹 그래프 생성 완료: 노드 {fab_graph.number_of_nodes()}개, 엣지 {fab_graph.number_of_edges()}개")
    draw_fab_graph(fab_graph)