import collections

class Task:
    def __init__(self, start_node, goal_node, is_hot_lot=False):
        self.start_node = start_node
        self.goal_node = goal_node
        self.is_hot_lot = is_hot_lot

class ScenarioScheduler:
    """
    현실적인 MES를 모사하여 OHT에게 결정론적인 시나리오 기반 작업을 할당합니다.
    """
    def __init__(self, tasks=None):
        self.task_queue = collections.deque(tasks if tasks else [])
        self.active_tasks = {} # agent_id: Task

    def add_task(self, start, goal, is_hot_lot=False):
        self.task_queue.append(Task(start, goal, is_hot_lot))

    def get_next_task(self, agent_id):
        """
        에이전트에게 대기 중인 다음 작업을 할당합니다.
        큐가 비어있으면 None을 반환합니다 (에이전트는 대기해야 함).
        """
        if self.task_queue:
            task = self.task_queue.popleft()
            self.active_tasks[agent_id] = task
            return task
        return None

    def complete_task(self, agent_id):
        return self.active_tasks.pop(agent_id, None)

    def reset(self, initial_tasks=None):
        self.task_queue = collections.deque(initial_tasks if initial_tasks else [])
        self.active_tasks = {}

def create_default_scenario(ports, num_tasks=50):
    """
    테스트를 위한 기본 시나리오 생성 (10% 확률로 Hot Lot 발생)
    """
    import random
    tasks = []
    for _ in range(num_tasks):
        start, goal = random.sample(ports, 2)
        is_hot_lot = random.random() < 0.1
        tasks.append(Task(start, goal, is_hot_lot))
    return tasks
