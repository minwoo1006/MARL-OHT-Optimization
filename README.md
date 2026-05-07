# MARL-OHT-Optimization
# 🏭 MARL 기반 OHT 트래픽 최적화 및 데드락 방지 프로젝트
> **Multi-Agent Reinforcement Learning for OHT Pathfinding and Throughput Optimization**

본 프로젝트는 반도체 팹(Fab)의 핵심 물류 시스템인 **OHT(Overhead Hoist Transport)**의 효율성을 극대화하기 위해 다중 에이전트 강화학습(MARL)을 적용합니다. 특히 고밀도 환경에서의 **교착 상태(Deadlock) 해소**와 **긴급 물량(Hot Lot) 우선 배송** 지능 구현에 초점을 맞춥니다.

---

## 📂 디렉터리 구성 및 스크립트 상세 설명

효율적인 협업과 코드 모듈화를 위해 다음과 같이 구조를 정의합니다.

### 1. Root Directory
- **`main.py`**: 프로젝트의 메인 실행 파일입니다. 환경 생성, 알고리즘 설정(Hyperparameters), Ray RLlib 학습 루프를 총괄합니다.
- **`requirements.txt`**: 프로젝트 실행에 필요한 라이브러리 목록(`gymnasium`, `pettingzoo`, `ray[rllib]`, `torch`, `pygame`, `networkx` 등)입니다.
- **`README.md`**: 프로젝트 개요, 상세 로드맵, 구조 설명서입니다.

### 2. `envs/` (Architect 담당 구역)
- **`__init__.py`**: `envs` 폴더를 파이썬 패키지로 인식하게 하며, 커스텀 환경을 `gym.register`에 등록하여 어디서든 불러올 수 있게 합니다.
- **`oht_env.py`**: **프로젝트의 핵심 환경 클래스**입니다. `PettingZoo.ParallelEnv`를 상속받아 OHT의 물리적 주행 규칙, 충돌 판정, 보상 계산 로직을 구현합니다.
- **`grid_map.py`**: `NetworkX`를 사용하여 팹의 복잡한 선로 구조를 그래프(Nodes & Edges) 데이터로 생성하고 관리합니다.

### 3. `agents/` (Algorithm 담당 구역)
- **`__init__.py`**: 에이전트 모듈 초기화 파일입니다.
- **`mappo.py`**: Multi-Agent PPO 알고리즘의 Actor-Critic 신경망 구조와 에이전트별 정책 업데이트 로직이 포함됩니다.
- **`qmix.py`**: 에이전트들의 개별 Q-value를 통합하여 전체 시스템 가치를 최적화하는 QMIX 알고리즘 및 Mixing Network 구현체입니다.

### 4. `utils/` (Analysis & Visualization 담당 구역)
- **`__init__.py`**: 유틸리티 모듈 초기화 파일입니다.
- **`visualization.py`**: `Pygame`을 이용한 2D 시뮬레이션 뷰어입니다. 학습된 에이전트들의 주행과 정체 상황을 실시간으로 렌더링합니다.
- **`logger.py`**: Weights & Biases(W&B) API와 연동하여 학습 지표를 클라우드에 기록하는 유틸리티입니다.
- **`metrics.py`**: 물동량 처리량(Throughput), 평균 사이클 타임(ACT), 데드락 발생 빈도 등 성능 지표를 계산하고 통계화합니다.

### 5. `tests/`
- **`test_env.py`**: 환경 로직(OHT 이동, 충돌 판정 등)이 정상 작동하는지 확인하는 단위 테스트 스크립트입니다.

---

## 📅 4주 집중 로드맵 (Roadmap)

### **1주차: 설계 및 환경 구축 (Foundation Phase)**
- **목표:** 에이전트가 움직일 수 있는 '규격화된 운동장' 완성
- **팀원 1:** `NetworkX` 기반 레일 그래프 설계 및 `oht_env.py` 기초(step, reset) 구현
- **팀원 2:** OHT의 상태(State) 정보(위치, 목적지, 속도) 및 행동(Action) 정의 확정
- **팀원 3:** 비교군이 될 '최단 경로(Dijkstra) 기반 이동 규칙' 베이스라인 코드 작성
- **공통 과업:** 주말까지 **"OHT 1대가 충돌 없이 목적지에 도착하는 환경"** 검증

### **2주차: 지능 주입 및 초기 학습 (Brain Building Phase)**
- **목표:** 다중 에이전트(MARL) 알고리즘 적용 및 기본 협력 학습
- **팀원 1:** 맵 내 OHT 대수 확장(3~5대) 및 충돌/데드락 감지 로직 고도화
- **팀원 2:** Ray RLlib 세팅 및 MAPPO/QMIX 알고리즘 환경 연결
- **팀원 3:** `visualization.py`를 이용한 실시간 모니터링용 기본 뷰어 개발
- **공통 과업:** 주말까지 **"여러 대의 OHT가 협력하여 이동"**하는 초기 모델 도출

### **3주차: 창의성 구현 및 성능 고도화 (Innovation Phase)**
- **목표:** "Hot Lot(우선순위)" 기능 추가 및 성능 최적화
- **팀원 1:** **Hot Lot(긴급 물량)** 시스템 및 우선순위 기반 차등 보상(Reward) 수식 적용
- **팀원 2:** 하이퍼파라미터(Learning Rate, Batch Size 등) 튜닝 및 학습 안정화
- **팀원 3:** W&B를 통한 학습 곡선 모니터링 및 정체 구간 해소 데이터 수집
- **공통 과업:** 주말까지 **"지능형 양보가 포함된 고도화 모델"** 완성

### **4주차: 검증 및 최종 발표 준비 (Final Polish Phase)**
- **목표:** 압도적인 성과 지표(KPI) 산출 및 발표물 완벽 세팅
- **팀원 1:** 에이전트 대수 확장 테스트(Scalability) 및 모델 견고성 최종 검증
- **팀원 2:** '기존 규칙' vs 'RL 모델' 성과 비교(Throughput, Cycle Time) 그래프 도출
- **팀원 3:** 시뮬레이션 데모 영상 편집 및 최종 발표 자료(PPT) 제작
- **공통 과업:** **목요일 학습 종료**, 금/토요일 발표 리허설 및 스크립트 최종 검토

---

## 🛠️ 기술 스택 (Tech Stack)
- **Language**: Python 3.10+
- **RL Framework**: Ray RLlib, PettingZoo
- **Environment**: Gymnasium, NetworkX
- **Deep Learning**: PyTorch 
- **Tracking**: Weights & Biases (W&B)

# 참고 사이트
- https://flatland-association.github.io/flatland-book/intro.html
- https://github.com/flatland-association/flatland-rl
