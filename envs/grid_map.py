import networkx as nx
import matplotlib.pyplot as plt

def create_fab_graph(layout_type="mega", **kwargs):
    """
    반도체 팹(Fab)의 OHT 레일 네트워크를 유향 그래프(Directed Graph)로 생성합니다.
    """
    if layout_type == "basic":
        return create_basic_fab_graph()
    elif layout_type == "spine_bay":
        return create_spine_and_bay_graph()
    elif layout_type == "mega":
        return create_mega_fab_graph(**kwargs)
    else:
        return create_mega_fab_graph(**kwargs)

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
        G.nodes[node]['is_stocker'] = False
    return G

def create_spine_and_bay_graph():
    """
    Week 2 버전: 10x6 Spine-and-Bay 구조
    """
    G = nx.DiGraph()
    
    # 1. 중앙 Spine (x=0~9, y=2: 우행, y=3: 좌행)
    for x in range(9):
        G.add_edge((x, 2), (x+1, 2), edge_type="spine")
        G.add_edge((x+1, 3), (x, 3), edge_type="spine")
    G.add_edge((9, 2), (9, 3), edge_type="spine")
    G.add_edge((0, 3), (0, 2), edge_type="spine")

    # 2. Bay 1~4
    bay_configs = [
        {"edges": [((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 0), (3, 0)), ((3, 0), (4, 0)), ((4, 0), (4, 1)), ((4, 1), (4, 2))], "ports": [(2, 0), (3, 0), (4, 0)]},
        {"edges": [((6, 2), (6, 1)), ((6, 1), (6, 0)), ((6, 0), (7, 0)), ((7, 0), (8, 0)), ((8, 0), (8, 1)), ((8, 1), (8, 2))], "ports": [(6, 0), (7, 0), (8, 0)]},
        {"edges": [((4, 3), (4, 4)), ((4, 4), (4, 5)), ((4, 5), (3, 5)), ((3, 5), (2, 5)), ((2, 5), (2, 4)), ((2, 4), (2, 3))], "ports": [(2, 5), (3, 5), (4, 5)]},
        {"edges": [((8, 3), (8, 4)), ((8, 4), (8, 5)), ((8, 5), (7, 5)), ((7, 5), (6, 5)), ((6, 5), (6, 4)), ((6, 4), (6, 3))], "ports": [(6, 5), (7, 5), (8, 5)]}
    ]
    
    all_ports = []
    for config in bay_configs:
        G.add_edges_from(config["edges"], edge_type="bay")
        all_ports.extend(config["ports"])

    for node in G.nodes():
        G.nodes[node]['is_port'] = True if node in all_ports else False
        G.nodes[node]['is_stocker'] = False
    
    return G

def create_mega_fab_graph(width=100, height=60, bay_interval=10, bay_depth=5):
    """
    수백 x 수백 스케일을 지원하는 메가 팹 제너레이터.
    - width: 팹의 가로 길이
    - height: 팹의 세로 길이
    - bay_interval: Bay가 배치되는 간격
    - bay_depth: Bay 루프의 깊이 (Spine에서 얼마나 멀어지는지)
    """
    G = nx.DiGraph()
    
    spine_y_up = height // 2 - 1
    spine_y_down = height // 2
    
    # 1. Hierarchical Spine (East-bound & West-bound)
    for x in range(width - 1):
        G.add_edge((x, spine_y_up), (x+1, spine_y_up), edge_type="spine")
        G.add_edge((x+1, spine_y_down), (x, spine_y_down), edge_type="spine")
    
    # U-turns at ends
    G.add_edge((width-1, spine_y_up), (width-1, spine_y_down), edge_type="spine")
    G.add_edge((0, spine_y_down), (0, spine_y_up), edge_type="spine")

    # 2. Parametric Bay Generation
    ports = []
    stockers = []
    
    for x in range(bay_interval, width - bay_interval, bay_interval):
        # Upper Bay
        entry_up = (x, spine_y_up)
        exit_up = (x + 2, spine_y_up)
        
        bay_up_edges = [
            (entry_up, (x, spine_y_up - 1)),
            ((x, spine_y_up - 1), (x, spine_y_up - bay_depth)),
            ((x, spine_y_up - bay_depth), (x + 1, spine_y_up - bay_depth)),
            ((x + 1, spine_y_up - bay_depth), (x + 2, spine_y_up - bay_depth)),
            ((x + 2, spine_y_up - bay_depth), (x + 2, spine_y_up - 1)),
            ((x + 2, spine_y_up - 1), exit_up)
        ]
        G.add_edges_from(bay_up_edges, edge_type="bay")
        ports.append((x + 1, spine_y_up - bay_depth))
        stockers.append((x, spine_y_up - 1)) # 입구 근처에 스토커 배치
        
        # Lower Bay
        entry_down = (x + 2, spine_y_down)
        exit_down = (x, spine_y_down)
        
        bay_down_edges = [
            (entry_down, (x + 2, spine_y_down + 1)),
            ((x + 2, spine_y_down + 1), (x + 2, spine_y_down + bay_depth)),
            ((x + 2, spine_y_down + bay_depth), (x + 1, spine_y_down + bay_depth)),
            ((x + 1, spine_y_down + bay_depth), (x, spine_y_down + bay_depth)),
            ((x, spine_y_down + bay_depth), (x, spine_y_down + 1)),
            ((x, spine_y_down + 1), exit_down)
        ]
        G.add_edges_from(bay_down_edges, edge_type="bay")
        ports.append((x + 1, spine_y_down + bay_depth))
        stockers.append((x + 2, spine_y_down + 1))
        
    for node in G.nodes():
        G.nodes[node]['is_port'] = True if node in ports else False
        G.nodes[node]['is_stocker'] = True if node in stockers else False
        
    return G

def draw_fab_graph(G):
    # 너무 크면 그리지 않음
    if G.number_of_nodes() > 200:
        print(f"⚠️ 그래프가 너무 커서 시각화를 생략합니다. (노드 {G.number_of_nodes()}개)")
        return
        
    pos = {node: (node[0], -node[1]) for node in G.nodes()}
    plt.figure(figsize=(15, 8))
    node_colors = []
    for n in G.nodes():
        if G.nodes[n].get('is_port'): node_colors.append('gold')
        elif G.nodes[n].get('is_stocker'): node_colors.append('lightgreen')
        else: node_colors.append('lightgray')
        
    edge_colors = ['red' if G[u][v]['edge_type'] == 'spine' else 'blue' for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=False, node_color=node_colors, edge_color=edge_colors,
            node_size=100, arrows=True, arrowstyle='-|>', arrowsize=10)
    
    plt.title("OHT Fab Network Layout", fontsize=16)
    plt.show()

if __name__ == "__main__":
    # 메가 팹 테스트 (100x60)
    mega_fab = create_mega_fab_graph(width=100, height=60)
    print(f"✅ Mega Fab 생성 완료: 노드 {mega_fab.number_of_nodes()}개, 엣지 {mega_fab.number_of_edges()}개")
    print(f"   - 포트: {len([n for n, d in mega_fab.nodes(data=True) if d.get('is_port')])}개")
    print(f"   - 스토커: {len([n for n, d in mega_fab.nodes(data=True) if d.get('is_stocker')])}개")
