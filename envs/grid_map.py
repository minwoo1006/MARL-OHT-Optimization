import networkx as nx
import matplotlib.pyplot as plt

def create_fab_graph(layout_type="spine_bay"):
    """
    반도체 팹(Fab)의 OHT 레일 네트워크를 유향 그래프(Directed Graph)로 생성합니다.
    """
    if layout_type == "basic":
        return create_basic_fab_graph()
    else:
        return create_spine_and_bay_graph()

def create_basic_fab_graph():
    G = nx.DiGraph()
    # 기존 5x5 루프 로직 (백업용)
    main_loop_edges = [
        ((0, 0), (1, 0)), ((1, 0), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)),
        ((4, 0), (4, 1)), ((4, 1), (4, 2)), ((4, 2), (4, 3)), ((4, 3), (4, 4)),
        ((4, 4), (3, 4)), ((3, 4), (2, 4)), ((2, 4), (1, 4)), ((1, 4), (0, 4)),
        ((0, 4), (0, 3)), ((0, 3), (0, 2)), ((0, 2), (0, 1)), ((0, 1), (0, 0))
    ]
    G.add_edges_from(main_loop_edges, edge_type="main")
    bypass_edges = [((2, 0), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (2, 3)), ((2, 3), (2, 4))]
    G.add_edges_from(bypass_edges, edge_type="bypass")
    ports = [(1, 0), (3, 0), (2, 2), (1, 4), (3, 4)]
    for node in G.nodes():
        G.nodes[node]['is_port'] = True if node in ports else False
    return G

def create_spine_and_bay_graph():
    """
    현실적인 Spine-and-Bay 구조의 그래프를 생성합니다.
    - Spine: 중앙의 메인 고속 도로 (복선 구조)
    - Bay: Spine에서 분기되어 설비(Port)가 배치된 루프 구역
    """
    G = nx.DiGraph()
    
    # 1. 중앙 Spine (x=0~9, y=2: 우행, y=3: 좌행)
    for x in range(9):
        G.add_edge((x, 2), (x+1, 2), edge_type="spine") # 상단 Spine (우측행)
        G.add_edge((x+1, 3), (x, 3), edge_type="spine") # 하단 Spine (좌측행)
    
    # Spine 양 끝 연결 (U턴 구간)
    G.add_edge((9, 2), (9, 3), edge_type="spine")
    G.add_edge((0, 3), (0, 2), edge_type="spine")

    # 2. Bay 1 (상단 좌측, x=2~4)
    bay1_edges = [
        ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), 
        ((3, 0), (4, 0)), ((4, 0), (4, 1)), ((4, 1), (4, 2))
    ]
    G.add_edges_from(bay1_edges, edge_type="bay")
    
    # 3. Bay 2 (상단 우측, x=6~8)
    bay2_edges = [
        ((6, 2), (6, 1)), ((6, 1), (6, 0)), ((6, 0), (7, 0)), 
        ((7, 0), (8, 0)), ((8, 0), (8, 1)), ((8, 1), (8, 2))
    ]
    G.add_edges_from(bay2_edges, edge_type="bay")

    # 4. Bay 3 (하단 좌측, x=2~4)
    bay3_edges = [
        ((4, 3), (4, 4)), ((4, 4), (4, 5)), ((4, 5), (3, 5)), 
        ((3, 5), (2, 5)), ((2, 5), (2, 4)), ((2, 4), (2, 3))
    ]
    G.add_edges_from(bay3_edges, edge_type="bay")
    
    # 5. Bay 4 (하단 우측, x=6~8)
    bay4_edges = [
        ((8, 3), (8, 4)), ((8, 4), (8, 5)), ((8, 5), (7, 5)), 
        ((7, 5), (6, 5)), ((6, 5), (6, 4)), ((6, 4), (6, 3))
    ]
    G.add_edges_from(bay4_edges, edge_type="bay")

    # 6. 포트(Port) 설정 - 각 Bay의 정중앙 및 주변 노드
    ports = [
        (2, 0), (3, 0), (4, 0), # Bay 1
        (6, 0), (7, 0), (8, 0), # Bay 2
        (2, 5), (3, 5), (4, 5), # Bay 3
        (6, 5), (7, 5), (8, 5)  # Bay 4
    ]
    for node in G.nodes():
        G.nodes[node]['is_port'] = True if node in ports else False
    
    return G

def draw_fab_graph(G):
    pos = {node: (node[0], -node[1]) for node in G.nodes()}
    plt.figure(figsize=(12, 6))
    node_colors = ['#1f77b4' if G.nodes[n].get('is_port') else '#cccccc' for n in G.nodes()]
    edge_colors = ['red' if G[u][v]['edge_type'] == 'spine' else 'blue' for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors,
            node_size=600, font_size=8, font_weight="bold", arrows=True, arrowstyle='-|>', arrowsize=15)
    
    plt.title("OHT Spine-and-Bay Network Layout", fontsize=16)
    plt.show()

if __name__ == "__main__":
    fab_graph = create_spine_and_bay_graph()
    print(f"✅ Spine-and-Bay 그래프 생성 완료: 노드 {fab_graph.number_of_nodes()}개, 엣지 {fab_graph.number_of_edges()}개")
    # draw_fab_graph(fab_graph) # CLI 환경에서는 주석 처리
