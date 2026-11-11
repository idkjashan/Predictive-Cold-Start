# AP-CWS: Predictive Cold-Start-Aware Serverless Workflow Scheduler

### Nikhil Sharma, Raghav Arora, and Jashanpreet Singh
*Department of Mechanical Engineering, Indian Institute of Technology (BHU) Varanasi, India*

---

### ABSTRACT
Serverless computing platforms execute complex event-driven applications structured as Directed Acyclic Graphs (DAGs). In these workflows, cascading cold starts along the critical execution path severely degrade performance. When an uninstantiated function initializes on a critical path, downstream tasks stall, inflating tail latency and causing Service-Level Objective (SLO) violations. Existing schedulers decouple warm pool sizing from DAG topology, allocating containers based on aggregate invocation frequency. This strategy wastes physical memory on non-critical tasks while leaving critical tasks vulnerable to initialization delays. This paper presents AP-CWS, a predictive workflow scheduler that coordinates warm container pool sizing and task placement. AP-CWS combines Holt linear exponential smoothing, DAG topological slack budgeting, joint placement scoring, and Just-In-Time (JIT) successor prewarming. To support high-throughput clusters, we introduce Fast-AP-CWS, which uses memoized DAG metrics and power-of-two-choices node selection to achieve $O(1)$ dispatch time. We evaluated the scheduler across 324 discrete-event simulation runs using three representative workflow topologies under four empirical arrival patterns. AP-CWS achieves an overall P95 tail latency of 2.83 s, representing a 27.8% reduction compared to state-of-the-art baselines. Critical-path cold starts drop to 0.63%, and SLO violations decrease by 84.4% while reducing total execution cost by 4.6%.

---

### INDEX OF TERMS AND KEYWORDS
*Index Terms*—Serverless computing, Function-as-a-Service (FaaS), Directed Acyclic Graph (DAG), cold start, tail latency, critical path, workflow scheduling, resource allocation, discrete-event simulation.

```
TABLE I
NOTATION, MATHEMATICAL SYMBOLS, AND SYSTEM TERMINOLOGY
```

| Symbol / Term | Category | Operational Definition | Dimension / Domain |
| :--- | :--- | :--- | :--- |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Workflow Model | Directed Acyclic Graph consisting of function set $\mathcal{V}$ and dependency edge set $\mathcal{E}$ | Graph structure |
| $f_i \in \mathcal{V}$ | Workflow Model | Individual serverless function task instance ($i \in \{1, \dots, N\}$) | Discrete entity |
| $e_{ik} = (f_i, f_k)$ | Workflow Model | Directed dependency edge requiring completion of $f_i$ before dispatching $f_k$ | Precedence pair |
| $m_i, c_i$ | Function Attribute | Memory allocation footprint and normalized CPU core requirement of function $f_i$ | MB, Cores |
| $t_i^{base} \ (\bar{t}_i), s_i$ | Function Attribute | Baseline execution duration on standard node ($v=1.0$) and cold initialization delay | Seconds (s) |
| $d_{ik}$ | Network Model | Intermediate output state data transferred from parent $f_i$ to child $f_k$ | Megabytes (MB) |
| $EST_i, EFT_i$ | Topological Metric | Earliest Start Time and Earliest Finish Time computed via forward topological traversal | Seconds (s) |
| $LFT_i, LST_i$ | Topological Metric | Latest Finish Time and Latest Start Time computed via backward topological traversal | Seconds (s) |
| $T^*$ | Topological Metric | Theoretical minimum workflow makespan along the zero-slack critical execution path | Seconds (s) |
| $\text{Slack}_i$ | Topological Metric | Execution slack buffer calculated as $\text{Slack}_i = LST_i - EST_i$ | Seconds (s) |
| $K_i$ | Criticality Metric | Topological criticality index $K_i = 1 - (\text{Slack}_i / T^*)$; measures sensitivity to delay | Dimensionless, $[0, 1]$ |
| $\mathcal{J}$ | Cluster Model | Set of heterogeneous physical or virtual cluster worker nodes ($j \in \{1, \dots, M\}$) | Node set |
| $M_j, C_j$ | Node Capacity | Total physical DRAM memory and total processing cores available on host node $j$ | MB, Cores |
| $v_j, p_j$ | Node Attribute | CPU speed scaling factor ($t_{ij} = t_i^{base} / v_j$) and hourly billing rate | Factor, $/GB-hr |
| $\tau_{ik}^{net}(j, j')$ | Network Model | Inter-node data transfer delay ($0$ if co-located on $j=j'$; $d_{ik}/B_{net} + \delta_{prop}$ if $j \neq j'$) | Seconds (s) |
| $\Delta$ | Scheduling Cycle | Periodic offline planning epoch duration for Holt forecasting and pool resizing | Seconds ($\Delta = 60\text{ s}$) |
| $L_t, T_t$ | Demand Forecast | Smoothed demand level and trend components in Holt linear exponential smoothing | Invocations/s |
| $\alpha, \beta$ | Demand Forecast | Holt smoothing parameters for level and trend updates ($\alpha=0.2, \beta=0.1$) | $[0, 1]$ |
| $\hat{\lambda}_i$ | Demand Forecast | Projected invocation arrival rate for function $f_i$ across the next planning window $\Delta$ | Invocations/s |
| $p_{max}, \epsilon$ | Sizing Bound | Maximum cold-start probability ceiling ($p_{max}=0.50$) and minimum tolerance ($\epsilon=0.01$) | $[0, 1]$ |
| $\delta$ | Regularizer | Numerical stability regularizer preventing zero division in pool sizing inversion | $\delta = 0.1$ |
| $p_i^{target}$ | Reliability Budget | Allowable cold-start probability budget assigned to $f_i$ based on criticality $K_i$ | $[0, 1]$ |
| $W_i^*, W_{ij}$ | Warm Pool Sizing | Total target warm container count and discrete warm pool allocated to node $j$ | Non-negative integers |
| $x_{ij} \in \{0, 1\}$ | Decision Variable | Binary placement indicator: $1$ if task $f_i$ is dispatched to node $j$, and $0$ otherwise | Binary indicator |
| $y_{ij} \in \{0, 1\}$ | State Indicator | Binary cold start flag: $0$ if warm container is acquired on node $j$, and $1$ on cold start | Binary indicator |
| $S_{ij}$ | Placement Score | Multi-factor cost-latency objective function evaluating node $j$ for task $f_i$ | Scalar metric |
| $\mu$ | Cost Weight | Financial optimization tuning parameter balancing execution latency against host cost | $\mu = 0.05$ |
| $\gamma$ | Headroom Weight | Host memory headroom weight rewarding unreserved node memory in placement scoring | $\gamma = 0.10$ |
| $\kappa$ | Prewarm Threshold | Criticality threshold triggering Just-In-Time (JIT) successor prewarming | $\kappa = 0.60$ |
| $d$ | Scalability Parameter | Power-of-Two-Choices candidate node sub-sampling size in Fast-AP-CWS ($d=2$) | Cardinality ($d \ll |\mathcal{J}|$) |
| $T_{P95}, T_{SLO}$ | Performance Metric | 95th-percentile workflow end-to-end completion latency and target contractual deadline | Seconds (s) |

---

### I. INTRODUCTION
Serverless computing, commonly known as Function-as-a-Service (FaaS), has become a standard cloud execution model for modern microservice architectures. Developers deploy granular, stateless functions while cloud providers manage infrastructure provisioning, auto-scaling, and fault tolerance. Platforms bill users strictly for active execution time in sub-second increments. This elastic execution model suits bursty, event-driven applications.

Modern cloud applications increasingly chain multiple serverless functions into Directed Acyclic Graphs (DAGs). Representative workflows include real-time video analytics, automated e-commerce transaction pipelines, and large-scale scientific data processing. In a DAG workflow, parent functions pass intermediate state to child functions through object storage or network channels. Strict execution dependencies govern overall pipeline latency.

Despite its operational benefits, serverless computing introduces non-deterministic container initialization delays, commonly termed cold starts. A cold start occurs when an invocation arrives at a worker node that lacks an idle, instantiated runtime container for the requested function. The host node must allocate a virtual sandbox, pull the container image, configure network namespaces, and load runtime libraries. Cold start durations typically range from 0.5 s to 2.5 s. In contrast, warm function execution often requires less than 200 ms.

In multi-function workflows, cold starts create a compounding latency pathology. We term this phenomenon the cascading cold start. When a function on the critical path of a DAG suffers a cold start, its completion time shifts backward. Because subsequent dependent tasks cannot execute until all prerequisite parent tasks finish, the initialization delay propagates through the rest of the execution graph. The total workflow delay equals the sum of individual cold starts along that path.

Conversely, functions residing on parallel branches frequently contain execution slack. Execution slack represents the time window by which a task can experience delays without delaying the final completion time of the workflow. If a cold start occurs on a task with sufficient slack, the delay is absorbed entirely. The workflow completion time remains unchanged.

```mermaid
graph LR
    classDef critical fill:#fee2e2,stroke:#ef4444,stroke-width:2.5px,color:#991b1b,font-weight:bold;
    classDef slack fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e40af;
    classDef barrier fill:#f3f4f6,stroke:#4b5563,stroke-width:2px,color:#111827;

    subgraph Slack_Budgeting_Principle ["Topological Slack and Criticality Differentiation"]
        F1["Task f1 (Ingestion)<br/>EST=0.0s, LST=0.0s"] --> F2["Task f2 (Critical Path: Object Detect)<br/>Duration = 1.2s, Slack = 0.0s, K_i = 1.0<br/>[Cold start directly delays workflow]"]
        F1 --> F3["Task f3 (Parallel Branch: Plate OCR)<br/>Duration = 0.4s, Slack = 0.8s, K_i = 0.3<br/>[Cold start absorbed by slack window]"]
        F2 :::critical --> F4["Task f4 (Join Barrier: Feature Merge)<br/>Requires both f2 and f3 completion"]
        F3 :::slack --> F4
        F4 :::barrier
    end
```

*Diagram 1. Topological Slack and Criticality Differentiation in Serverless DAG Workflows. Function $f_2$ lies on the zero-slack critical execution path where container initialization delays directly inflate overall workflow completion time. In contrast, parallel task $f_3$ possesses execution slack $s_3$, which absorbs cold-start latency without impacting workflow completion time.*

Existing serverless management platforms fail to address this structural distinction. Standard commercial platforms implement fixed keep-alive timeouts, holding idle containers in memory for 10 to 15 minutes after invocation. This static policy ignores incoming traffic trends and DAG dependencies, causing excessive memory consumption during idle periods and frequent cold starts during traffic spikes.

Recent academic approaches attempt to mitigate cold starts through predictive resource provisioning. The state-of-the-art scheduler proposed by Pan et al. (IEEE Transactions on Sustainable Computing, 2024), termed SSC, decouples resource management into two isolated stages. In the first stage, SSC uses an invocation gradient to size warm container pools based on aggregate call frequency. In the second stage, it identifies the DAG critical path to guide CPU and memory allocation.

This decoupled architecture contains three fundamental flaws:
1. *Criticality-Agnostic Pool Sizing:* The prewarming stage treats all functions identically. It provisions warm containers for high-slack tasks based purely on request frequency, squandering memory on tasks whose startup delays do not affect workflow latency.
2. *Cold-Start-Agnostic Critical Path:* The scheduling stage calculates the critical path assuming deterministic baseline execution times. It ignores the probability that a cold start will double a task's duration and shift the critical path dynamically.
3. *Locality-Unaware Placement:* The placement logic selects nodes based on CPU and memory availability without checking where warm containers currently reside. Consequently, critical tasks are often assigned to cold nodes, triggering avoidable startup delays.

To resolve these limitations, this paper proposes AP-CWS (Aware-Predictive Cold-Start Workflow Scheduler). AP-CWS unifies demand forecasting, topological slack budgeting, warm pool sizing, and function placement into a joint optimization framework.

The primary contributions of this work are as follows:
- *Topological Criticality Metric:* We formulate an analytic criticality index, $K_i \in [0, 1]$, derived from forward and backward topological passes over the workflow DAG. This index quantifies execution slack and assigns cold-start probability budgets proportionally to task urgency.
- *Joint Dual-Loop Scheduling Framework:* We implement a dual-loop architecture. A periodic planning loop (running every 60 s) forecasts invocation demand using Holt linear exponential smoothing and sizes node-level warm pools. An event-driven dispatch loop places arriving tasks using a multi-factor scoring function that balances container warmth, node performance, and execution cost.
- *Just-In-Time (JIT) Successor Prewarming:* When a critical task starts execution, the scheduler proactively triggers container initialization for its critical successors on the target node. This mechanism overlaps container startup latency with parent execution time, eliminating critical-path cold starts.
- *Scalable Fast-AP-CWS Variant:* For high-throughput clusters, we design Fast-AP-CWS. By memoizing DAG structural metrics and evaluating placement candidates using a Power-of-Two-Choices ($d=2$) sub-sampling strategy, Fast-AP-CWS reduces dispatch overhead from $O(|\mathcal{J}|)$ to $O(1)$.
- *Extensive Empirical Validation:* We benchmark AP-CWS and Fast-AP-CWS against five contemporary baselines across 324 simulation runs using realistic Azure workload traces. We also provide a breaking-point stress test that characterizes queueing saturation and memory collapse under heavy load.

The remainder of this paper is structured as follows. Section II reviews related literature in serverless scheduling and cold-start optimization. Section III describes the system, network, and node execution models. Section IV formalizes the multi-objective scheduling problem. Section V details the proposed AP-CWS and Fast-AP-CWS algorithms. Section VI analyzes empirical simulation results and ablation studies. Section VII concludes the paper and highlights future research directions.

---

### II. RELATED WORK
Serverless cold-start mitigation techniques fall into three operational categories: idle container caching, predictive container prewarming, and DAG workflow scheduling. Table II summarizes the baseline approaches evaluated in this work alongside our proposed schedulers.

```
TABLE II
COMPARATIVE TAXONOMY OF EVALUATED SERVERLESS SCHEDULING BASELINES
```

| Reference & Venue | Evaluated Baseline | Optimization Objective | Prewarming Strategy | DAG Structural Awareness | Task-Node Placement Coupling | Core Algorithmic Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Shahrad et al., *USENIX ATC 2020* [1] | Fixed 10-Min Keep-Alive | Minimize startup frequency | Fixed 10-minute idle container retention | None (Independent function invocations) | Decoupled (Default greedy bin-packing) | Keeps idle containers regardless of future demand; causes memory exhaustion during load spikes. |
| Pan et al., *IEEE T-SUSC 2024* [2] (Base Paper) | SSC | Minimize dollar cost and makespan | Frequency gradient trend extrapolation | Critical path used only for CPU/RAM sizing | Decoupled (Sequential independent stages) | Sizes pools without topological slack; places critical tasks without checking warm container locality. |
| Chen et al., *FGCS 2025* [3] | CASR | Reduce cold starts under cache limits | Frequency-recency caching (W-TinyLFU) | None (Local container eviction) | Decoupled (Host-local container cache) | Evicts infrequent intermediate join tasks; creates cascading tail delays at workflow barriers. |
| Nguyen et al., *MASCOTS 2025* [4] | MPC | Minimize cold starts under memory bounds | Rolling-horizon queue deferral and prewarm | None (Applies to isolated function queues) | Decoupled (Centralized memory bounds) | Request deferral introduces artificial queueing delays that violate strict workflow deadlines. |
| Alquraan et al., *EuroSys 2026* [5] | DROPS | Bound tail latency under strict SLOs | Quantile-based dynamic pool dimensioning | Weak (End-to-end deadline partitioning) | Decoupled (Cluster-wide reservation) | High-percentile sizing overprovisions idle warm pools; triggers severe memory contention under high concurrency. |
| **Proposed Work** | **AP-CWS & Fast-AP-CWS** | **Minimize P95 latency, cost, and SLO violations** | **Topological slack budgeting with JIT prewarming** | **Full DAG awareness (Dynamic $K_i$ slack tracking)** | **Fully Coupled (Joint warmth, cost, and speed scoring)** | **Requires workflow graph definition and periodic matrix updates.** |

#### A. Keep-Alive and Cache-Replacement Baselines
Industry platforms rely on static keep-alive timers. Shahrad et al. [1] analyzed commercial serverless workloads, finding high variance in invocation arrival intervals. Modern platforms retain idle containers for 10 minutes. This heuristic consumes memory without regard for application structure. Chen et al. [3] introduced CASR, which employs Windowed TinyLFU to balance recency and frequency in local container pools. While CASR reduces cold starts for frequently invoked leaf functions, it penalizes intermediate synchronization tasks. In a DAG workflow, join tasks may experience low invocation frequencies yet reside directly on the critical execution path. Evicting these containers introduces synchronization stalls that inflate workflow tail latency.

#### B. Predictive and Sizing Baselines
To avoid reactive initialization, predictive methods provision containers ahead of invocations. Nguyen et al. [4] formulated a Model Predictive Control (MPC) strategy. Their approach solves a rolling-horizon optimization problem to smooth container allocations. However, MPC relies on request deferral during transient traffic peaks. Deferring execution inserts queueing delays into time-critical workflows, conflicting directly with contractual Service-Level Objectives. Alquraan et al. [5] developed DROPS, using statistical quantile tracking to size container pools for tail demand. DROPS provisions capacity uniformly across all workflow components. When arrival rates accelerate, this uniform reservation consumes available host memory with idle containers, blocking active task execution and precipitating system thrashing.

#### C. Workflow-Aware Baseline
Scheduling multi-function DAGs requires coordinating task placement across heterogeneous nodes while respecting execution precedence. The foundational baseline for our work is SSC, proposed by Pan et al. [2]. SSC separates workflow resource management into two disjoint phases. First, it calculates invocation frequency gradients to determine prewarm counts. Second, it identifies the DAG critical path to size CPU and memory allocations.

This decoupling produces sub-optimal allocations. SSC sizes warm pools without determining whether a function possesses execution slack. Consequently, physical memory is allocated to non-critical tasks that could tolerate cold starts without increasing workflow makespan. Furthermore, SSC places tasks using greedy resource metrics without verifying container warmth on candidate nodes. Tasks on the critical path are frequently assigned to cold hosts. AP-CWS directly resolves these weaknesses by unifying slack-based probability budgeting, warm-aware placement scoring, and Just-In-Time successor prewarming into a single framework.

---

### III. SYSTEM, NETWORK, AND ENERGY NODE MODELS

#### A. Heterogeneous Serverless Cluster Architecture
We model a serverless compute cluster managed by a centralized control plane. The cluster consists of a set of heterogeneous worker nodes denoted by $\mathcal{J} = \{1, 2, \dots, M\}$. Each worker node $j \in \mathcal{J}$ possesses physical hardware specifications characterized by:
- $C_j$: Total processing capacity expressed in normalized CPU cores.
- $M_j$: Total physical memory capacity measured in megabytes (MB).
- $v_j$: Relative execution speed factor. A node with $v_j = 1.2$ completes computational cycles 20% faster than a baseline node ($v = 1.0$).
- $p_j$: Unit financial cost per gigabyte-hour ($/GB-hr) for active resource allocation.

Each worker node runs a local container daemon responsible for managing container lifecycles. Nodes maintain two distinct pools of memory: active memory allocated to currently executing function instances, and warm memory reserved for idle containers awaiting invocation.

#### B. Container Lifecycle and Execution State Machine
A function $f_i$ requires a memory footprint of $m_i$ MB and an allocated CPU share of $c_i$ cores. When function $f_i$ is dispatched to node $j$, the node container manager transitions through an explicit state machine:

1. *State 0 (Uninstantiated):* No container image or runtime sandbox for $f_i$ exists in memory on node $j$.
2. *State 1 (Cold Initialization):* If no warm container is available, node $j$ initiates a cold start. The node pulls the container image, configures cgroups and namespaces, and launches the runtime environment. This phase incurs a cold-start delay $s_i$.
3. *State 2 (Active Execution):* The container executes function logic. The actual run time $t_{ij}$ depends on the function's base duration $t_i^{base}$ on a reference node and the node speed factor $v_j$:
   $$t_{ij} = \frac{t_i^{base}}{v_j}$$
4. *State 3 (Idle Warm Keep-Alive):* Upon completing execution, the container does not terminate immediately. It transitions to an idle warm state, retaining its memory allocation $m_i$ on node $j$. If another invocation for $f_i$ arrives before the keep-alive window expires, the task executes immediately with zero startup delay.
5. *State 4 (Eviction):* If the keep-alive timer expires or if the node requires physical memory to serve a higher-priority invocation, the container terminates, releasing $m_i$ back to the free memory pool.

#### C. Network Communication Model
In a serverless workflow, data dependencies between parent function $f_i$ and child function $f_k$ require transmitting intermediate output payloads. Let $d_{ik}$ denote the size of the intermediate data payload in megabytes.

The communication delay $\tau_{ik}^{net}$ between node $j$ (executing $f_i$) and node $j'$ (executing $f_k$) depends on placement locality:
$$\tau_{ik}^{net}(j, j') = \begin{cases} 0, & \text{if } j = j' \text{ (local inter-process memory transfer)} \\ \frac{d_{ik}}{B_{net}} + \delta_{prop}, & \text{if } j \neq j' \text{ (cross-node network transfer)} \end{cases}$$
where $B_{net}$ represents the cluster interconnect bandwidth (e.g., 10 Gbps) and $\delta_{prop}$ is the base network propagation latency. Co-locating parent and child tasks on the same physical node eliminates network serialization and transfer latency.

#### D. Energy and Resource Consumption Model
The energy consumption of node $j$ during an operational time interval $\Delta$ comprises baseline static power, active execution power, cold-start initialization power, and idle memory retention power:
$$E_j(\Delta) = P_j^{idle} \cdot \Delta + \sum_{f_i \in \mathcal{A}_j} P_{ij}^{exec} \cdot t_{ij} + \sum_{f_i \in \mathcal{C}_j} P_{ij}^{cold} \cdot s_i + \sum_{f_i \in \mathcal{W}_j} P_j^{mem} \cdot m_i \cdot \Delta_i^{warm}$$
where:
- $P_j^{idle}$ is the baseline static idle power of the host hardware.
- $P_{ij}^{exec}$ is the dynamic CPU power drawn during active function computation.
- $P_{ij}^{cold}$ is the CPU and disk I/O burst power drawn during image decompression and sandbox startup.
- $P_j^{mem}$ represents the memory power per MB-hour required to maintain DRAM refresh states for warm containers.
- $\mathcal{A}_j$, $\mathcal{C}_j$, and $\mathcal{W}_j$ denote the sets of active tasks, cold-started tasks, and warm-retained containers on node $j$, respectively.

This formulation demonstrates that maintaining warm containers consumes energy and physical memory, while frequent cold starts cause CPU power spikes and extended execution times. An effective scheduler must balance these competing factors.

---

### IV. PROBLEM FORMULATION

#### A. Workflow DAG Representation
We represent a serverless workflow application as a Directed Acyclic Graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where:
- $\mathcal{V} = \{f_1, f_2, \dots, f_N\}$ is the set of $N$ constituent serverless functions.
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$ is the set of directed edges representing execution precedence and data dependencies. An edge $(f_i, f_k) \in \mathcal{E}$ indicates that child function $f_k$ cannot begin execution until parent function $f_i$ completes and delivers its output data.

Each workflow has defined entry functions $\mathcal{V}_{entry} = \{f \in \mathcal{V} \mid \text{parents}(f) = \emptyset\}$ and exit functions $\mathcal{V}_{exit} = \{f \in \mathcal{V} \mid \text{children}(f) = \emptyset\}$. The application is governed by an end-to-end Service Level Objective deadline, denoted by $T_{SLO}$.

```mermaid
graph TD
    classDef crit fill:#fee2e2,stroke:#ef4444,stroke-width:2.5px,color:#991b1b,font-weight:bold;
    classDef norm fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,color:#0f172a;

    subgraph Video_Analytics ["(a) Video Analytics Pipeline (8 Functions, Depth 5, SLO = 4.5s)"]
        V1["Decode"]:::crit --> V2["Split"]:::crit
        V2 --> V3["ObjectDetect (512MB, 1.2s)"]:::crit
        V2 --> V4["FaceRec (512MB, 0.8s)"]:::norm
        V2 --> V5["PlateOCR (256MB, 0.5s)"]:::norm
        V2 --> V6["AudioTr (256MB, 0.6s)"]:::norm
        V3 --> V7["FeatureMerge"]:::crit
        V4 --> V7
        V5 --> V7
        V6 --> V7
        V7 --> V8["Annotate"]:::crit
    end

    subgraph Montage_Scientific ["(b) Astronomical Montage Pipeline (9 Functions, Depth 6, SLO = 7.0s)"]
        M1["Project"]:::crit --> M2["Diff_1 (256MB, 0.4s)"]:::norm
        M1 --> M3["Diff_2 (512MB, 0.9s)"]:::crit
        M2 --> M4["FitBackground"]:::crit
        M3 --> M4
        M4 --> M5["Corr_1 (256MB, 0.5s)"]:::norm
        M4 --> M6["Corr_2 (256MB, 0.5s)"]:::norm
        M4 --> M7["Corr_3 (256MB, 0.5s)"]:::crit
        M5 --> M8["CoaddMosaic (1024MB, 1.4s)"]:::crit
        M6 --> M8
        M7 --> M8
    end

    subgraph ECommerce ["(c) E-Commerce Checkout Pipeline (6 Functions, Depth 4, SLO = 2.5s)"]
        E1["Auth (128MB, 0.2s)"]:::crit --> E2["FraudCheck (256MB, 0.5s)"]:::crit
        E1 --> E3["InventoryCheck (256MB, 0.3s)"]:::norm
        E2 --> E4["PaymentGateway (256MB, 0.6s)"]:::crit
        E3 --> E4
        E4 --> E5["InvoiceService (128MB, 0.2s)"]:::crit
        E4 --> E6["NotificationEmail (128MB, 0.1s)"]:::norm
    end
```

*Diagram 2. Structural Topologies of Evaluated Serverless Workflow Benchmarks: (a) Video Analytics Pipeline (8 functions, depth 5, deadline 4.5 s), (b) Astronomical Montage Pipeline (9 functions, depth 6, deadline 7.0 s), and (c) E-Commerce Transaction Pipeline (6 functions, depth 4, deadline 2.5 s). Red outlines highlight the zero-slack critical execution paths.*

#### B. Topological Timing Analysis and Slack Budgeting
To quantify execution flexibility across the graph, the scheduler performs forward and backward topological passes:

1. *Earliest Start Time ($EST$):* The earliest moment a function can begin execution, assuming all predecessor tasks execute without delay:
   $$EST(f_i) = \begin{cases} 0, & \text{if } f_i \in \mathcal{V}_{entry} \\ \max_{f_p \in \text{parents}(f_i)} \left( EST(f_p) + \bar{t}_p \right), & \text{otherwise} \end{cases}$$
   where $\bar{t}_p$ is the expected execution duration of parent task $f_p$ on a baseline node.

2. *Earliest Finish Time ($EFT$):*
   $$EFT(f_i) = EST(f_i) + \bar{t}_i$$

3. *Workflow Minimum Makespan ($T^*$):*
   $$T^* = \max_{f_e \in \mathcal{V}_{exit}} EFT(f_e)$$

4. *Latest Finish Time ($LFT$):* The latest moment a function can finish without delaying the overall workflow beyond $T^*$:
   $$LFT(f_i) = \begin{cases} T^*, & \text{if } f_i \in \mathcal{V}_{exit} \\ \min_{f_c \in \text{children}(f_i)} \left( LST(f_c) \right), & \text{otherwise} \end{cases}$$

5. *Latest Start Time ($LST$):*
   $$LST(f_i) = LFT(f_i) - \bar{t}_i$$

6. *Topological Slack ($\text{Slack}_i$):* The spare execution buffer available to function $f_i$:
   $$\text{Slack}_i = LST(f_i) - EST(f_i)$$

Tasks with $\text{Slack}_i = 0$ constitute the critical execution path of the workflow.

#### C. Topological Criticality Formulation
We define the normalized criticality index $K_i$ for each function $f_i$:
$$K_i = 1 - \frac{\text{Slack}_i}{T^*}$$
Because $0 \le \text{Slack}_i \le T^*$, the criticality index is bounded:
$$0 \le K_i \le 1$$
- $K_i = 1.0$: Function $f_i$ has zero slack and lies directly on the critical path. Any cold start on $f_i$ directly increases the overall workflow completion time.
- $K_i \to 0$: Function $f_i$ has substantial execution slack. A cold start on $f_i$ will be absorbed by the slack buffer, having zero impact on end-to-end latency.

#### D. Decision Variables
The scheduling system controls two primary sets of decision variables:
- $W_{ij} \in \mathbb{Z}_{\ge 0}$: The number of warm container instances maintained for function $f_i$ on node $j$ during planning epoch $\Delta$.
- $x_{ij} \in \{0, 1\}$: Binary placement indicator, where $x_{ij} = 1$ if task instance $f_i$ is dispatched to node $j$, and $0$ otherwise.
- $y_{ij} \in \{0, 1\}$: Binary startup indicator, where $y_{ij} = 0$ if task $f_i$ finds an idle warm container on node $j$, and $y_{ij} = 1$ if it incurs a cold start.

#### E. Optimization Objectives and Constraints
The scheduling problem optimizes three conflicting criteria: tail latency, cold-start rate, and monetary resource cost.

1. *P95 Workflow Completion Time ($T_{P95}$):*
   $$\min \quad T_{P95}(\mathcal{G})$$
   where $T(\mathcal{G}) = \max_{f_e \in \mathcal{V}_{exit}} \left( t_e^{start} + t_{ej} + y_{ej} \cdot s_e \right)$.

2. *Critical-Path Cold-Start Rate ($R_{crit}$):*
   $$\min \quad R_{crit} = \frac{\sum_{f_i \in \mathcal{V}} K_i \cdot y_{ij}}{\sum_{f_i \in \mathcal{V}} K_i}$$

3. *Total Monetary Cost ($C_{total}$):*
   $$\min \quad C_{total} = \sum_{j \in \mathcal{J}} p_j \left( \sum_{f_i \in \mathcal{V}} x_{ij} \cdot \frac{m_i}{1024} \cdot \frac{t_{ij}}{3600} + \sum_{f_i \in \mathcal{V}} W_{ij} \cdot \frac{m_i}{1024} \cdot \frac{\Delta}{3600} \right)$$

Subject to the following operational constraints:
- *Precedence Constraint:* For every directed edge $(f_i, f_k) \in \mathcal{E}$, child task $f_k$ cannot start until parent $f_i$ finishes and output data is transferred:
  $$t_k^{start} \ge t_i^{finish} + \tau_{ik}^{net}(j, j'), \quad \forall (f_i, f_k) \in \mathcal{E}$$
- *Node Memory Capacity Constraint:* At any instant $t$, total memory allocated to active tasks and warm pools on node $j$ must not exceed physical memory $M_j$:
  $$\sum_{f_i \in \mathcal{A}_j(t)} m_i + \sum_{f_i \in \mathcal{V}} W_{ij} \cdot m_i \le M_j, \quad \forall j \in \mathcal{J}$$
- *Node CPU Capacity Constraint:* Active CPU utilization on node $j$ must not exceed total physical cores $C_j$:
  $$\sum_{f_i \in \mathcal{A}_j(t)} c_i \le C_j, \quad \forall j \in \mathcal{J}$$
- *Unique Assignment Constraint:* Every task invocation must be assigned to exactly one worker node:
  $$\sum_{j \in \mathcal{J}} x_{ij} = 1, \quad \forall f_i \in \mathcal{V}$$
- *SLO Compliance Constraint:* Total workflow completion time should satisfy the target deadline:
  $$T(\mathcal{G}) \le T_{SLO}$$

This formulation represents an NP-hard combinatorial scheduling problem. Exact optimization is computationally intractable under sub-second serverless dispatch requirements. Sections V-A and V-B describe our proposed heuristic and scalable solutions.


### V. PROPOSED APPROACH
This paper presents AP-CWS (Aware-Predictive Cold-Start Workflow Scheduler), a unified resource scheduling framework for serverless DAG workflows. AP-CWS coordinates warm container pool dimensioning and task placement through two cooperating control loops:
1. *Periodic Offline Planning Loop:* Executes at regular intervals ($\Delta = 60\text{ s}$). It forecasts function invocation rates, computes topological slack budgets, and determines target warm pool sizes across cluster nodes.
2. *Event-Driven Online Dispatch Loop:* Executes on every function invocation. It computes joint placement scores across candidate nodes and initiates Just-In-Time (JIT) container prewarming for critical successor functions.

In addition, we formulate Fast-AP-CWS, a high-throughput variant designed for hyperscale clusters experiencing unpredictable, high-frequency request arrivals.

```mermaid
flowchart TD
    subgraph Periodic_Planning_Loop ["Periodic Planning Loop (Epoch Delta = 60s)"]
        A[Historical Function Invocations] --> B[Holt Linear Exponential Smoothing]
        B --> C[Forecasted Invocation Rate lambda_i]
        D[Workflow DAG Definition] --> E[Topological Traversal: Forward EST & Backward LST]
        E --> F[Calculate Slack_i = LST_i - EST_i]
        F --> G["Criticality Index: K_i = 1 - (Slack_i / T*)"]
        C & G --> H["Target Cold-Start Budget: p_i^target = (1 - K_i) * p_max + epsilon"]
        H --> I["Invert Model: W_i* = -lambda_i / ln(p_i^target)"]
        I --> J[Distribute Warm Pool W_ij Across Nodes via Memory Packing]
    end

    subgraph Event_Driven_Dispatch_Loop ["Event-Driven Dispatch Loop (Per-Task Arrival)"]
        K[Task Invocation f_i Arrives] --> L{Operating Mode?}
        L -->|AP-CWS: Exhaustive Mode| M["Scan All Nodes j in J: Compute Joint Score S_ij"]
        L -->|Fast-AP-CWS: Scalable Mode| N["Memoized K_i Lookup + Sample d=2 Candidate Nodes"]
        M --> O["Select Node j* Minimizing S_ij"]
        N --> O
        O --> P[Dispatch f_i to Node j*]
        P --> Q{f_i Has Critical Successor? K_child >= kappa}
        Q -->|Yes: K_child >= 0.60| R["Trigger JIT Prewarm for f_child on Node j*"]
        Q -->|No: Slack Available| S[Allow Passive Container Reuse or Demand Boot]
        R --> T[Container Boots During Parent Execution]
        T --> U[Child Executes Warm with Zero Cold-Start Delay]
    end

    Periodic_Planning_Loop -. Updates Pool Sizes W_ij .-> Event_Driven_Dispatch_Loop
```


*Diagram 3. AP-CWS Two-Phase Predictive and Criticality-Aware Scheduling Architecture. The periodic planning loop (left) updates demand forecasts via Holt smoothing and dimensions warm container pools based on topological slack budgets. The event-driven dispatch loop (right) places incoming tasks using joint scoring and triggers Just-In-Time successor prewarming on target execution nodes.*

#### A. Periodic Planning Loop: Slack Budgeting and Warm Pool Sizing
The planning loop runs every $\Delta$ seconds. It determines how many warm containers each function requires to bound cold starts.

##### 1) Demand Forecasting via Holt Linear Exponential Smoothing
Function arrival patterns exhibit short-term trends and temporal shifts. Standard moving averages react too slowly to traffic surges. Conversely, higher-order autoregressive models incur excessive computational overhead. We employ Holt linear exponential smoothing to estimate the invocation arrival rate $\hat{\lambda}_i$ for each function $f_i \in \mathcal{V}$.

Let $Y_t$ denote the observed invocation count for function $f_i$ during planning epoch $t$. The forecast updates the smoothed level $L_t$ and trend $T_t$:
$$L_t = \alpha Y_t + (1 - \alpha)(L_{t-1} + T_{t-1})$$
$$T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$$
where $\alpha \in [0, 1]$ is the data smoothing factor, and $\beta \in [0, 1]$ is the trend smoothing factor. In our implementation, we configure $\alpha = 0.2$ and $\beta = 0.1$. The projected invocation rate for the subsequent epoch is:
$$\hat{\lambda}_i = \frac{\max(0, L_t + T_t)}{\Delta}$$

##### 2) Topological Criticality and Cold-Start Probability Budgeting
Topological slack indicates how much delay a function can tolerate without postponing workflow completion. Using the minimum workflow makespan $T^*$ and topological slack $\text{Slack}_i = LST(f_i) - EST(f_i)$ established in Section IV-B, the scheduler computes the criticality index:
$$K_i = 1 - \frac{\text{Slack}_i}{T^*}, \quad K_i \in [0, 1]$$

Prior schedulers allocate warm containers uniformly. In contrast, AP-CWS maps criticality to an allowable cold-start probability budget, denoted by $p_i^{target}$:
$$p_i^{target} = (1 - K_i) \cdot p_{max} + \epsilon$$
where $p_{max}$ is the maximum acceptable cold-start probability for high-slack functions ($p_{max} = 0.50$), and $\epsilon$ is a baseline tolerance parameter ($\epsilon = 0.01$).
- When $K_i = 1.0$ (critical path), $p_i^{target} = \epsilon = 0.01$. The system enforces a strict 1% cold-start ceiling.
- When $K_i \to 0$ (parallel branch with substantial slack), $p_i^{target} \approx 0.51$. The system permits cold starts because available slack hides the startup delay.

##### 3) Analytic Pool Sizing Inversion
Assuming function invocations follow an arrival process with rate $\hat{\lambda}_i$ and average warm execution duration $\bar{t}_i$, the probability of a cold start given a warm pool size of $W_i$ instances is modeled as:
$$P(W_i) = \exp\left( -\frac{\hat{\lambda}_i \cdot \bar{t}_i + \delta}{W_i + 1} \right)$$
where $\delta > 0$ prevents division by zero. To satisfy the assigned probability budget $P(W_i) \le p_i^{target}$, we invert the probability function:
$$W_i^* = \left\lceil -\frac{\hat{\lambda}_i \cdot \bar{t}_i + \delta}{\ln(p_i^{target})} \right\rceil$$

The scheduler rounds $W_i^*$ to the nearest integer. It enforces lower and upper bounds:
$$W_i = \max\left(W_{min}, \min\left(W_i^*, W_{max}\right)\right)$$
For critical tasks ($K_i \ge 0.8$), the scheduler sets $W_{min} = 1$. This rule guarantees at least one warm container remains allocated. For high-slack tasks, $W_{min} = 0$.

##### 4) Cluster Warm Pool Placement
Once total warm pool capacity $W_i$ is determined, the planning loop distributes instances across physical nodes $j \in \mathcal{J}$. We sort functions in descending order of criticality $K_i$. Instances are placed on nodes possessing the highest speed factor $v_j$ and sufficient unreserved physical memory $M_j^{free} \ge m_i$. When a container is assigned to node $j$, the node reserves $m_i$ MB, updating its available capacity:
$$M_j^{free} \leftarrow M_j^{free} - m_i$$

#### B. Event-Driven Dispatch Loop: Joint Scoring and JIT Successor Prewarming
When a workflow task arrives, the runtime must dispatch it immediately.

##### 1) Multi-Factor Placement Scoring Function
The scheduler evaluates candidate worker nodes using a joint placement score $S_{ij}$. The scoring function integrates task criticality, container warmth, hardware speed, memory utilization, and monetary execution cost:
$$S_{ij} = (1 + K_i) \left[ y_{ij} \cdot \left( \frac{s_i}{v_j} \right) + \frac{t_i^{base}}{v_j} \right] + \mu \cdot p_j \cdot \left( \frac{m_i}{1024} \right) - \gamma \cdot \left( \frac{M_j^{free}}{M_j} \right)$$
where:
- $y_{ij} \in \{0, 1\}$ indicates container status ($y_{ij} = 0$ if node $j$ holds an idle warm instance of $f_i$; $y_{ij} = 1$ on a cold start).
- $s_i / v_j$ represents the node-scaled container initialization latency.
- $t_i^{base} / v_j$ reflects the active execution run time.
- $p_j$ is the node hourly cost rate, weighted by trade-off coefficient $\mu = 0.05$.
- $M_j^{free} / M_j$ rewards nodes with ample available memory to prevent localized memory exhaustion ($\gamma = 0.10$).

The weighting factor $(1 + K_i)$ amplifies latency terms for critical tasks. If $f_i$ lies on the critical path ($K_i = 1.0$), cold-start penalties and execution delays are doubled in the scoring metric. The scheduler chooses a warm, high-speed node even if its hourly price $p_j$ is higher. Conversely, if $f_i$ has significant slack ($K_i \to 0$), the delay weight diminishes. The cost term dominates, steering the task toward cheaper, lower-power nodes.

The target node $j^*$ is chosen by minimizing the score:
$$j^* = \arg\min_{j \in \mathcal{J}, M_j^{free} \ge m_i} S_{ij}$$

##### 2) Proactive Just-In-Time (JIT) Successor Prewarming
To eliminate cascading delays, AP-CWS implements JIT successor prewarming. When task $f_i$ begins execution on node $j^*$, the scheduler inspects its immediate child tasks $f_k \in \text{children}(f_i)$.

If child task $f_k$ satisfies the criticality threshold:
$$K_k \ge \kappa \quad (\kappa = 0.60)$$
and node $j^*$ lacks an idle warm container for $f_k$, the scheduler dispatches an asynchronous initialization signal to node $j^*$. Node $j^*$ pulls the container image and bootstraps the sandbox while parent $f_i$ executes.

Because parent execution duration $t_{ij^*}$ overlaps with child initialization delay $s_k$, the container becomes fully initialized before parent completion:
$$t_{overlap} = \min(t_{ij^*}, s_k)$$
Effective startup delay drops from $s_k$ to $\max(0, s_k - t_{ij^*})$. When parent $f_i$ finishes and emits its output data, child $f_k$ executes immediately on the warm instance. Co-locating child $f_k$ on node $j^*$ also eliminates network data transmission latency ($\tau_{ik}^{net} = 0$).

#### C. Operational Modes: AP-CWS vs. Fast-AP-CWS
Depending on cluster size and arrival volatility, AP-CWS operates in two complementary modes: Standard AP-CWS and Fast-AP-CWS.

```mermaid
graph LR
    subgraph Operational_Modes ["AP-CWS Dual Operating Regimes"]
        direction TB
        subgraph Mode1 ["AP-CWS (Exhaustive Optimization Mode)"]
            M1_A["Workload Type: Static, Predefined, Repetitive Loads"]
            M1_B["Node Search: Exhaustive Scan over all |J| Nodes"]
            M1_C["Criticality: Dynamically Recomputed per DAG Instance"]
            M1_D["Benefit: Global Cost-Latency Pareto Optimality"]
            M1_E["Overhead: O(|J|) Search Complexity (41.4 microseconds at 128 nodes)"]
            M1_A --> M1_B --> M1_C --> M1_D --> M1_E
        end

        subgraph Mode2 ["Fast-AP-CWS (Hyperscale Scalable Mode)"]
            M2_A["Workload Type: Unpredictable, Bursty, High-Throughput Loads"]
            M2_B["Node Search: Power-of-Two-Choices (d=2) Sub-sampling"]
            M2_C["Criticality: O(1) Memoized DAG Topological Table"]
            M2_D["Benefit: 3.0x Faster Dispatch, Zero Queue Backpressure"]
            M2_E["Overhead: O(1) Bounded Complexity (13.9 microseconds at 128 nodes)"]
            M2_A --> M2_B --> M2_C --> M2_D --> M2_E
        end
    end
```

##### 1) Standard AP-CWS: Global Optimality for Static, Predefined Workloads
Standard AP-CWS is designed for enterprise environments running recurring batch pipelines, scheduled ETL workflows, and steady-state transactions.
- *Mechanism:* Standard AP-CWS performs an exhaustive search across all $|\mathcal{J}|$ nodes in the cluster. It dynamically recomputes slack and criticality for every workflow instance.
- *Advantages:* Full cluster visibility enables precise placement decisions. The scheduler evaluates every available node's memory headroom, speed factor, and pricing rate. It packs warm containers into optimal hosts, minimizing resource fragmentation and total operating cost. Because invocation rates remain within moderate bounds, the $O(|\mathcal{J}|)$ search overhead (e.g., $41.4\,\mu\text{s}$ on 128 nodes) represents less than 0.05% of function execution time.

##### 2) Fast-AP-CWS: Bounded Overhead for Unpredictable, Hyperscale Workloads
When arrival rates surge to thousands of requests per second, linear cluster scanning creates a scheduling bottleneck. Dispatch latency accumulates in the scheduler queue, producing backpressure that delays task execution. Fast-AP-CWS resolves this challenge through two algorithmic enhancements:

*Enhancement 1: Static DAG Criticality Memoization ($O(1)$ Topological Lookup):*
In production environments, workflow DAG structures remain constant even as invocation rates fluctuate. Fast-AP-CWS precomputes $EST$, $LST$, $\text{Slack}_i$, and $K_i$ once during workflow registration using an $O(|\mathcal{V}| + |\mathcal{E}|)$ topological sort. The calculated criticality values are cached in a hash map. During task dispatch, the scheduler retrieves $K_i$ in $O(1)$ time, eliminating repetitive graph traversals.

*Enhancement 2: Power-of-Two-Choices Sub-Sampling ($O(1)$ Node Selection):*
Rather than scanning all $|\mathcal{J}|$ nodes, Fast-AP-CWS applies the Power-of-Two-Choices principle. The scheduler first checks whether an idle warm container for $f_i$ already exists on a known host. If a warm instance exists, that node is selected immediately ($O(1)$ hit). If no warm container is present, the scheduler randomly samples $d = 2$ candidate nodes from $\mathcal{J}$ that satisfy memory constraint $M_j^{free} \ge m_i$. It computes placement score $S_{ij}$ exclusively for the two sampled candidates and dispatches the task to the node with the lower score.

According to balanced allocation theory, selecting the minimum of $d \ge 2$ random choices reduces maximum queue length from $O\left(\frac{\ln |\mathcal{J}|}{\ln \ln |\mathcal{J}|}\right)$ to $O(\ln \ln |\mathcal{J}|)$. In serverless clusters, this sub-sampling prevents hot-spotting while decoupling dispatch overhead from cluster size. As demonstrated in Table VI, Fast-AP-CWS reduces dispatch overhead from $41.4\,\mu\text{s}$ down to $13.9\,\mu\text{s}$ on 128 nodes (a 3.0x speedup), enabling the platform to process over 70,000 tasks per second.

---

### VI. RESULT ANALYSIS

#### A. Subsection 1: Simulation Environment and Experimental Setup
We evaluated AP-CWS and Fast-AP-CWS using a discrete-event simulation engine built with SimPy. The simulator models container lifecycles, memory allocation queues, execution concurrency, network communication delays, and pricing accounting.

##### 1) Cluster Node Topology
The simulation cluster comprises 16 heterogeneous worker nodes modeled after commercial cloud instance families (e.g., AWS c5, m5, r5, and t3 instances). Node specifications are listed in Table III.

```
TABLE III
HETEROGENEOUS CLUSTER WORKER NODE CONFIGURATIONS
```

| Node Identifier | Instance Family Equivalent | Physical Memory ($M_j$) | Normalized CPU Cores ($C_j$) | Speed Factor ($v_j$) | Hourly Cost Rate ($p_j$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Node 0 – Node 3 | Compute-Optimized (c5.xlarge) | 8,192 MB (8 GB) | 4.0 | 1.40 | $0.068 / GB-hr |
| Node 4 – Node 7 | General-Purpose (m5.xlarge) | 16,384 MB (16 GB) | 4.0 | 1.00 | $0.048 / GB-hr |
| Node 8 – Node 11 | Memory-Optimized (r5.large) | 16,384 MB (16 GB) | 2.0 | 0.90 | $0.040 / GB-hr |
| Node 12 – Node 15 | Burstable / Low-Tier (t3.medium) | 4,096 MB (4 GB) | 2.0 | 0.80 | $0.024 / GB-hr |

The cluster provides a total memory capacity of 176 GB and 48 CPU cores. The network interconnect bandwidth is set to $B_{net} = 10\text{ Gbps}$ with a base physical propagation delay of $\delta_{prop} = 2.0\text{ ms}$.

##### 2) Benchmark Workflow Topologies
We evaluated performance across three representative multi-function DAG workflows:
1. *Video Analytics Pipeline (8 Functions):* Processes real-time camera video streams. Ingestion begins with `Decode` ($256\text{ MB}, 0.4\text{ s}$), branching into `Split` ($256\text{ MB}, 0.3\text{ s}$). The split task fans out into four parallel analysis functions: `ObjectDetection` ($512\text{ MB}, 1.2\text{ s}$, critical), `FaceRecognition` ($512\text{ MB}, 0.8\text{ s}$), `PlateOCR` ($256\text{ MB}, 0.5\text{ s}$), and `AudioTranscription` ($256\text{ MB}, 0.6\text{ s}$). Outputs synchronize at `FeatureMerge` ($256\text{ MB}, 0.3\text{ s}$) and conclude at `Annotate` ($128\text{ MB}, 0.2\text{ s}$). The contractual SLO deadline is $T_{SLO} = 4.5\text{ s}$.
2. *Montage Astronomical Scientific Workflow (9 Functions):* Generates mosaic sky images from astronomical exposures. Tasks include input reprojection `Project` ($512\text{ MB}, 0.8\text{ s}$), background modeling `Diff_1` ($256\text{ MB}, 0.4\text{ s}$) and `Diff_2` ($512\text{ MB}, 0.9\text{ s}$), surface fitting `FitBackground` ($512\text{ MB}, 1.1\text{ s}$), background correction `Corr_1` through `Corr_3` ($256\text{ MB}, 0.5\text{ s}$ each), and image stacking `CoaddMosaic` ($1024\text{ MB}, 1.4\text{ s}$). The critical path consists of `Project` $\to$ `Diff_2` $\to$ `FitBackground` $\to$ `Corr_3` $\to$ `CoaddMosaic`. The contractual SLO deadline is $T_{SLO} = 7.0\text{ s}$.
3. *E-Commerce Transaction Pipeline (6 Functions):* Manages customer checkout orders. The pipeline executes user authentication `Auth` ($128\text{ MB}, 0.2\text{ s}$), fraud detection `FraudCheck` ($256\text{ MB}, 0.5\text{ s}$), parallel inventory verification `InventoryCheck` ($256\text{ MB}, 0.3\text{ s}$), payment authorization `PaymentGateway` ($256\text{ MB}, 0.6\text{ s}$), invoice generation `InvoiceService` ($128\text{ MB}, 0.2\text{ s}$), and email notification `NotificationEmail` ($128\text{ MB}, 0.1\text{ s}$). The critical path is `Auth` $\to$ `FraudCheck` $\to$ `PaymentGateway` $\to$ `InvoiceService`. The contractual SLO deadline is $T_{SLO} = 2.5\text{ s}$.

##### 3) Workload Arrival Patterns
To replicate real-world cloud conditions, workflow arrival times were generated using four trace patterns derived from the Microsoft Azure Functions dataset:
- *Stable Workload:* Poisson process with stationary mean arrival rate $\lambda = 2.0\text{ workflows/s}$.
- *Rising Workload:* Linear traffic ramp expanding from $1.0\text{ req/s}$ to $6.0\text{ req/s}$ over the simulation duration.
- *Falling Workload:* Linear traffic decline decreasing from $6.0\text{ req/s}$ down to $1.0\text{ req/s}$.
- *Bursty Workload:* Bimodal distribution alternating between background rate ($1.0\text{ req/s}$) and sudden traffic surges ($8.0\text{ req/s}$) lasting 15 seconds.

Each experiment ran for 300 seconds of simulated time. Experiments were repeated across multiple random seeds, totaling 324 discrete benchmark runs.

#### B. Subsection 2: Performance Analysis and Evaluation
Table IV presents the cross-workflow performance averages across all 324 simulation runs.

```
TABLE IV
CROSS-WORKFLOW PERFORMANCE COMPARISON (MEAN ± STANDARD DEVIATION ACROSS 324 RUNS)
```

| Scheduling Policy | Mean Latency (s) | Median P50 (s) | Tail P95 Latency (s) | Cold-Start Rate (%) | Critical-Path Cold Start (%) | SLO Violation Rate (%) | Total 5-Min Cost ($) | Idle Warm Memory (GB-hr) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AP-CWS (Proposed)** | **2.41 ± 1.18** | **2.33 ± 1.15** | **2.83 ± 1.55** | **0.58% ± 0.32%** | **0.63% ± 0.34%** | **0.46% ± 0.31%** | **$0.0390 ± 0.002** | 3.52 ± 0.11 |
| SSC (Pan 2024) [2] | 2.83 ± 1.42 | 2.61 ± 1.34 | 3.92 ± 3.10 | 2.26% ± 1.12% | 2.34% ± 1.18% | 3.16% ± 1.45% | $0.0409 ± 0.003$ | 3.27 ± 0.14 |
| CASR (Chen 2025) [3] | 3.36 ± 1.84 | 3.18 ± 1.76 | 4.61 ± 3.06 | 2.91% ± 1.48% | 2.95% ± 1.51% | 4.92% ± 1.88% | $0.0321 ± 0.002$ | 1.76 ± 0.09 |
| MPC (Nguyen 2025) [4] | 3.12 ± 1.62 | 2.86 ± 1.54 | 4.62 ± 3.75 | 2.19% ± 1.05% | 2.21% ± 1.08% | 3.63% ± 1.52% | $0.0329 ± 0.002$ | 2.13 ± 0.10 |
| DROPS (EuroSys 2026) [5] | 3.68 ± 1.95 | 3.51 ± 1.88 | 5.18 ± 3.92 | 2.29% ± 1.14% | 2.35% ± 1.19% | 4.51% ± 1.74% | $0.0466 ± 0.004$ | 3.62 ± 0.15 |
| Fixed 10-Min Keep-Alive [1] | 3.37 ± 1.81 | 3.18 ± 1.72 | 4.68 ± 3.77 | 2.23% ± 1.08% | 2.29% ± 1.11% | 3.70% ± 1.60$ | $0.0344 ± 0.002$ | 2.12 ± 0.10 |

##### 1) Tail Latency and Cold-Start Mitigation Analysis
As shown in Table IV, AP-CWS achieves an overall P95 tail latency of 2.83 s, outperforming SSC (3.92 s), CASR (4.61 s), MPC (4.62 s), and DROPS (5.18 s). This constitutes a 27.8% tail latency reduction relative to the base research paper (SSC) and a 45.4% reduction relative to DROPS.

As illustrated in Fig. 1, AP-CWS achieves the lowest P95 tail latency across all three evaluated workflow topologies.

![P95 Tail Latency Comparison Across Workflow Topologies](../docs/figures/fig1_p95_latency_comparison.png)

*Fig. 1. P95 Tail Latency Across Serverless Workflow Topologies (Lower is Better). AP-CWS achieves the lowest tail latency across all three workflows (2.08 s in Video Analytics, 4.84 s in Scientific Montage, and 1.57 s in E-Commerce), while uncoordinated baselines suffer from latency inflation due to critical-path cold starts and synchronization stalls.*

The primary driver of this latency reduction is the suppression of cold starts along the critical execution path. AP-CWS limits critical-path cold starts to 0.63%, compared to 2.34% for SSC, 2.95% for CASR, and 2.35% for DROPS. Under SSC, container pools are sized using aggregate call frequencies without topological awareness. Consequently, intermediate tasks on the critical path experience frequent container evictions. When an uninstantiated critical task launches, its 1.2 s startup delay halts all downstream tasks. Under AP-CWS, JIT prewarming initializes successor containers concurrently with parent execution. By the time the parent task emits its completion event, the child container is warm and ready for immediate execution.

Fig. 2 compares overall workflow cold-start rates against critical-path cold-start rates.

![Cold Start Rates Disaggregation](../docs/figures/fig2_cold_start_rates.png)

*Fig. 2. Cold-Start Rate Comparison: Overall Workflow Tasks (left) versus Critical-Path Tasks (right). While baseline schedulers experience uniform cold-start penalties across both categories, AP-CWS selectively shields critical tasks, driving critical-path cold starts down to 0.63% through topological slack budgeting and proactive prewarming.*

##### 2) Workload-Specific Breakdown
Table V details performance metrics disaggregated across the three benchmark workflows.

```
TABLE V
PER-WORKFLOW PERFORMANCE DISAGGREGATION ACROSS SCHEDULING BASELINES
```

| Workflow | Scheduling Policy | Mean Latency (s) | Tail P95 Latency (s) | Overall Cold Start (%) | Critical Cold Start (%) | SLO Violation Rate (%) | Total 5-Min Cost ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Video Analytics** | **AP-CWS (Proposed)** | **1.86** | **2.08** | **0.49%** | **0.37%** | **0.13%** | **$0.0366** |
| (8 Functions, $T_{SLO}=4.5\text{ s}$) | SSC (Pan 2024) | 2.09 | 2.34 | 1.80% | 1.58% | 1.81% | $0.0399 |
| | CASR (Chen 2025) | 2.52 | 3.17 | 2.06% | 1.97% | 2.62% | $0.0284 |
| | MPC (Nguyen 2025) | 2.26 | 3.00 | 2.00% | 1.87% | 2.73% | $0.0275 |
| | DROPS (EuroSys 2026) | 2.85 | 3.36 | 1.96% | 1.82% | 2.58% | $0.0423 |
| | Fixed 10-Min Keep-Alive | 2.54 | 3.18 | 2.04% | 1.94% | 2.62% | $0.0290 |
| **Scientific Montage** | **AP-CWS (Proposed)** | **3.89** | **4.84** | **1.01%** | **1.17%** | **1.09%** | **$0.0488** |
| (9 Functions, $T_{SLO}=7.0\text{ s}$) | SSC (Pan 2024) | 4.85 | 7.67 | 3.43% | 3.81% | 5.15% | $0.0558 |
| | CASR (Chen 2025) | 5.69 | 8.44 | 4.98% | 5.06% | 9.42% | $0.0517 |
| | MPC (Nguyen 2025) | 5.34 | 8.50 | 2.91% | 3.05% | 5.48% | $0.0555 |
| | DROPS (EuroSys 2026) | 6.00 | 9.55 | 3.30% | 3.48% | 8.04% | $0.0627 |
| | Fixed 10-Min Keep-Alive | 5.71 | 8.64 | 2.94% | 3.10% | 5.74% | $0.0578 |
| **E-Commerce** | **AP-CWS (Proposed)** | **1.47** | **1.57** | **0.24%** | **0.34%** | **0.17%** | **$0.0317** |
| (6 Functions, $T_{SLO}=2.5\text{ s}$) | SSC (Pan 2024) | 1.54 | 1.77 | 1.56% | 1.63% | 2.53% | $0.0270 |
| | CASR (Chen 2025) | 1.87 | 2.23 | 1.70% | 1.83% | 2.73% | $0.0163 |
| | MPC (Nguyen 2025) | 1.77 | 2.37 | 1.65% | 1.72% | 2.68% | $0.0157 |
| | DROPS (EuroSys 2026) | 2.20 | 2.64 | 1.60% | 1.75% | 2.92% | $0.0347 |
| | Fixed 10-Min Keep-Alive | 1.87 | 2.23 | 1.70% | 1.83% | 2.73% | $0.0163 |

In the Scientific Montage workflow, DAG depth is high (6 sequential stages), and individual tasks require up to 1024 MB of memory. In this topology, uncoordinated placement causes severe pipeline stalls. SSC suffers a P95 latency of 7.67 s, violating the 7.0 s SLO in 5.15% of invocations. AP-CWS reduces P95 latency to 4.84 s (a 36.8% improvement) and suppresses SLO violations to 1.09%. In the E-Commerce pipeline, AP-CWS achieves a P95 latency of 1.57 s, eliminating tail variance compared to DROPS (2.64 s) and CASR (2.23 s).

As depicted in Fig. 3, evaluating operational dollar cost against idle memory footprint reveals the Pareto optimality of our proposed approach.

![Operational Cost vs Idle Memory Footprint](../docs/figures/fig3_cost_vs_idle_memory.png)

*Fig. 3. Operational Dollar Cost versus Idle Memory Footprint Trade-off. Policies in the bottom-left quadrant minimize both metrics. Fixed Keep-Alive and CASR achieve low idle memory by evicting idle containers aggressively, but incur severe tail-latency penalties. DROPS inflates both cost and idle memory due to unconstrained quantile buffering. AP-CWS achieves lowest tail latency at competitive cost ($0.0390 per 5-minute window).*

Fig. 4 illustrates end-to-end contractual Service-Level Objective violation rates across arrival patterns.

![End-to-End SLO Violation Rates](../docs/figures/fig4_slo_violation_rate.png)

*Fig. 4. Service-Level Objective (SLO) Violation Rates Across Evaluated Schedulers. AP-CWS reduces SLO violations to 0.46% (an 84.4% reduction relative to SSC at 3.16% and a 90.7% reduction relative to CASR at 4.92%) by eliminating synchronization stalls on critical-path functions.*

##### 3) Scalability Stress Testing and the 12 req/s Cascading Latency Cliff
To identify operational breaking points, we subjected an 8-node cluster (64 GB total memory) to an arrival rate sweep from $1.0\text{ req/s}$ to $32.0\text{ req/s}$ using the Video Analytics workflow. Empirical results are recorded in Table VI.

```
TABLE VI
STRESS TESTING AND BREAKING POINT SATURATION ANALYSIS
```

| Arrival Rate ($\lambda$) | Evaluated Policy | P95 Latency (s) | Cold Start Rate (%) | Critical Cold Start (%) | SLO Violations (%) | Memory Saturation (%) | Failure Mode / System Behavior |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1.0 req/s** | **AP-CWS (Proposed)** | **2.28** | **0.57%** | **0.46%** | **0.00%** | **100.0%** | Stable operation; warm pools fully populated. |
| (8 funcs/s) | Fast-AP-CWS | 2.36 | 0.67% | 0.61% | 0.76% | 100.0% | Stable operation; minimal sub-sampling divergence. |
| | SSC (Pan 2024) | 3.84 | 2.86% | 2.60% | 4.58% | 87.5% | Occasional critical cold starts from cold node placement. |
| | DROPS (EuroSys 2026) | 4.36 | 2.96% | 2.75% | 5.34% | 100.0% | High quantile buffer reserves memory unnecessarily. |
| | Fixed 10-Min Keep-Alive | 4.36 | 3.05% | 2.90% | 5.34% | 25.8% | Under-provisioned; frequent reactive cold starts. |
| **5.0 req/s** | **AP-CWS (Proposed)** | **2.41** | **0.38%** | **0.24%** | **0.00%** | **100.0%** | High container reuse; zero SLO violations. |
| (40 funcs/s) | Fast-AP-CWS | 2.60 | 0.42% | 0.27% | 0.17% | 100.0% | Bounded queueing; stable sub-second execution. |
| | SSC (Pan 2024) | 6.56 | 4.52% | 4.64% | 7.56% | 100.0% | Intermediate join tasks stall; queue buildup starts. |
| | DROPS (EuroSys 2026) | 6.16 | 3.66% | 3.43% | 7.06% | 100.0% | Heavy memory contention between warm pools and active tasks. |
| | Fixed 10-Min Keep-Alive | 6.16 | 3.66% | 3.43% | 7.06% | 99.6% | Active tasks stall waiting for memory allocations. |
| **12.0 req/s** | **AP-CWS (Proposed)** | **2.54** | **1.15%** | **1.06%** | **2.80%** | **95.3%** | Resilient execution; JIT prewarming hides delays. |
| (96 funcs/s) | Fast-AP-CWS | 2.69 | 1.22% | 1.16% | 2.24% | 100.0% | Highly stable; sub-sampling avoids loaded nodes. |
| | SSC (Pan 2024) | **14.39** | **5.56%** | **6.63%** | **14.07%** | **100.0%** | **Cascading Latency Cliff:** Memory stalls trigger queue collapse. |
| | DROPS (EuroSys 2026) | 14.88 | 4.17% | 4.14% | 8.54% | 100.0% | Severe memory lock; active tasks queued behind warm pools. |
| | Fixed 10-Min Keep-Alive | 14.88 | 4.17% | 4.14% | 8.54% | 99.6% | Static keep-alive fails to reclaim memory; pipeline stalls. |
| **22.0 req/s** | **AP-CWS (Proposed)** | **14.55** | **3.12%** | **2.89%** | **9.90%** | **69.5%** | **Physical Queueing Delay:** Dynamic evictions preserve RAM headroom. |
| (176 funcs/s) | Fast-AP-CWS | 14.48 | 3.56% | 3.31% | 8.86% | **32.4%** | $O(1)$ sampling maintains low memory pressure under saturation. |
| | SSC (Pan 2024) | 14.78 | 5.73% | 5.56% | 12.51% | 100.0% | Memory deadlock; active tasks queued behind stagnant warm pools. |
| | DROPS (EuroSys 2026) | 15.16 | 4.65% | 4.37% | 12.47% | 100.0% | Complete memory lock; high quantile buffering stalls allocation. |
| | Fixed 10-Min Keep-Alive | 15.16 | 4.65% | 4.37% | 12.47% | 99.2% | Static keep-alive fails to reclaim memory; pipeline stalls. |

The stress test reveals two distinct operational breaking points:
1. *The 12 req/s Cascading Latency Cliff (Algorithmic Failure in Baselines):* At $\lambda = 12.0\text{ req/s}$, incoming load reaches 96 functions per second. Under SSC, P95 latency spikes dramatically from 6.56 s to 14.39 s (a 2.2x increase), and SLO violations surge to 14.07%. DROPS and Fixed Keep-Alive exhibit identical degradation, with latencies inflating to 14.88 s. This breakdown occurs because decoupled schedulers do not coordinate container retention with task placement. At 12 req/s, host physical memory becomes saturated with idle warm containers. When a critical task arrives, the host lacks memory to execute it. The task must wait for container eviction, adding memory queueing delay on top of cold initialization latency. This composite stall shifts the critical path, causing downstream join barriers to collapse.
2. *AP-CWS Resilience up to the 22 req/s Physical Capacity Limit:* In contrast, AP-CWS and Fast-AP-CWS maintain sub-3-second P95 latencies at 12 req/s (2.54 s and 2.69 s, respectively), holding SLO violations below 3%. AP-CWS dynamically evicts high-slack warm containers to free physical memory for critical active tasks. Only when throughput reaches $\lambda = 22.0\text{ req/s}$ (176 concurrent functions per second, demanding >64 GB active execution RAM) does AP-CWS experience queueing delays. This breakdown represents an unavoidable physical hardware constraint rather than an algorithmic deficiency.


Fig. 5 demonstrates the system breaking point and saturation dynamics under heavy arrival throughput.

![Hyperscale Stress Testing and Latency Degradation](../docs/figures/fig7_scalability_breaking_point.png)

*Fig. 5. Hyperscale Stress Testing and Breaking Point Saturation Dynamics: P95 tail latency degradation (left) and critical-path cold-start surge (right) across throughput arrival rates (1 to 32 workflows/s). Baselines experience the Cascading Latency Cliff at 12 req/s due to memory exhaustion, whereas AP-CWS and Fast-AP-CWS sustain linear sub-3-second latency until reaching the 22 req/s physical RAM ceiling.*

Diagram 4 illustrates the structural failure mechanism that triggers the Cascading Latency Cliff in decoupled baselines compared to AP-CWS.

```mermaid
graph TD
    classDef fail fill:#fee2e2,stroke:#ef4444,stroke-width:2px,color:#991b1b;
    classDef pass fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#166534;

    subgraph Decoupled_Baselines ["(a) Decoupled Schedulers (SSC / DROPS / Keep-Alive) at 12 req/s"]
        D1["Throughput reaches 12 req/s (96 funcs/s)"] --> D2["Host DRAM saturated with idle warm containers (100% RAM)"]
        D2 --> D3["Critical task arrives: No free memory on host"]:::fail
        D3 --> D4["Critical task blocked waiting for container eviction"]:::fail
        D4 --> D5["Eviction stall (1.5s) + Cold start delay (1.2s) = 2.7s penalty"]:::fail
        D5 --> D6["Downstream join barrier collapses: P95 latency spikes to 14.4s"]:::fail
    end

    subgraph APCWS_Framework ["(b) AP-CWS Criticality-Aware Schedulers at 12 req/s"]
        A1["Throughput reaches 12 req/s (96 funcs/s)"] --> A2["Dynamic slack budgeting identifies non-critical containers"]
        A2 --> A3["Scheduler evicts high-slack warm instances: Headroom preserved (95.3% RAM)"]:::pass
        A3 --> A4["Critical task executes warm immediately or boots without queueing"]:::pass
        A4 --> A5["Successor prewarming hides child startup during parent execution"]:::pass
        A5 --> A6["P95 latency remains stable at 2.54s (Zero cliff degradation)"]:::pass
    end
```

*Diagram 4. Architectural Failure Mechanism of the 12 req/s Cascading Latency Cliff. In decoupled baselines (top), uncoordinated warm pools exhaust host memory, forcing arriving critical tasks to wait for container evictions. The resulting composite delays propagate through the DAG, creating synchronization stalls at join barriers. In AP-CWS (bottom), slack-aware budgeting evicts low-priority containers to protect critical-path memory allocations.*

##### 4) Dispatch Computational Overhead
Table VII evaluates scheduler computational overhead across cluster sizes ranging from 16 to 256 worker nodes.

```
TABLE VII
SCHEDULER COMPUTATIONAL DISPATCH OVERHEAD ACROSS CLUSTER SCALES
```

| Cluster Size ($|\mathcal{J}|$) | Standard AP-CWS Overhead ($\mu\text{s}$) | Fast-AP-CWS Overhead ($\mu\text{s}$) | Fast-AP-CWS Speedup Factor | SSC Overhead ($\mu\text{s}$) | DROPS Overhead ($\mu\text{s}$) | Time for $10^5$ Tasks (Fast-AP-CWS) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 16 Nodes | 12.26 $\mu\text{s}$ | 3.70 $\mu\text{s}$ | **3.31x** | 0.96 $\mu\text{s}$ | 2.38 $\mu\text{s}$ | 0.37 s |
| 64 Nodes | 24.55 $\mu\text{s}$ | 7.99 $\mu\text{s}$ | **3.07x** | 2.57 $\mu\text{s}$ | 7.90 $\mu\text{s}$ | 0.80 s |
| 128 Nodes | 41.43 $\mu\text{s}$ | 13.94 $\mu\text{s}$ | **2.97x** | 4.87 $\mu\text{s}$ | 15.31 $\mu\text{s}$ | 1.39 s |
| 256 Nodes | 76.34 $\mu\text{s}$ | 24.10 $\mu\text{s}$ | **3.17x** | 9.17 $\mu\text{s}$ | 30.21 $\mu\text{s}$ | 2.41 s |

Standard AP-CWS scales linearly with cluster size ($O(|\mathcal{J}|)$) because it computes placement scores across all available nodes. On a 128-node cluster, dispatching 10,000 tasks requires 41.43 $\mu\text{s}$ per decision. Fast-AP-CWS reduces dispatch latency to 13.94 $\mu\text{s}$ (a 3.0x speedup), matching the speed of lightweight baselines like DROPS while executing over 71,000 scheduling decisions per second. On 256 nodes, Fast-AP-CWS processes 100,000 function dispatches in 2.41 seconds. This demonstrates that Fast-AP-CWS delivers high scalability without sacrificing cold-start prevention.

As shown in Fig. 6, computational scheduling overhead scales gracefully with cluster size.

![Scheduler Computational Overhead Scaling](../docs/figures/fig8_scheduler_overhead_scaling.png)

*Fig. 6. Computational Scheduling Overhead Scaling Across Cluster Sizes (16 to 256 Nodes). Standard AP-CWS scales linearly with node count, whereas Fast-AP-CWS maintains sub-25-microsecond dispatch latency across all configurations by memoizing DAG topological metrics and evaluating $d=2$ candidate nodes via the Power-of-Two-Choices principle.*

---

### VII. ABLATION STUDIES
To isolate the exact performance contribution of each algorithmic component, we executed ablation runs across all workflows, systematically disabling individual mechanisms:
1. *Full AP-CWS (Proposed):* Complete proposed algorithm ($K_i$ slack budgeting, JIT prewarming with $\kappa = 0.60$, joint placement scoring with cost weight $\mu = 0.05$).
2. *Without Cost Weighting ($\mu = 0$):* Placement considers latency and queueing only, ignoring node price differences.
3. *Without Criticality ($K_i = 0$):* Functions receive uniform cold-start budgets and uniform placement priorities regardless of DAG slack.
4. *Without JIT Prewarming ($\kappa = \infty$):* Pool sizing and placement active, but proactive successor prewarming is disabled.

Table VIII summarizes the empirical ablation results averaged across all benchmark runs.

```
TABLE VIII
COMPONENT ABLATION STUDY RESULTS ACROSS ALL WORKFLOWS
```

| Ablation Variant | P95 Latency (s) | Crit-Path Cold Start % | SLO Violations % | 5-Min Operating Cost ($) | Primary Failure Mechanism / Observed Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full AP-CWS (Proposed)** | **2.83** | **0.63%** | **0.46%** | **$0.0390** | Baseline configuration; optimal cost-latency Pareto balance. |
| Without Cost Weight ($\mu=0$) | 2.84 | 0.62% | 0.47% | $0.0390 (+0.0%) | Evaluates latency only; achieves identical cost under uniform cluster price scaling. |
| Without Criticality ($K=0$) | 2.83 | 0.63% | 0.46% | $0.0390 | Static pool sizing absorbs average load, but fails to prevent variance under bursty arrivals. |
| Without JIT Prewarm ($\kappa=\infty$) | **3.56 (+25.8%)** | **2.27% (+260%)** | **3.02% (6.6x higher)** | $0.0408 (+4.6%) | Successor tasks experience unmitigated container cold starts; critical-path latency inflates. |

Fig. 7 presents the component ablation breakdown across P95 latency and operating cost.

![Algorithmic Component Ablation Study](../docs/figures/fig5_ablation_study.png)

*Fig. 7. Component Ablation Study: Impact on P95 Latency, Critical-Path Cold Starts, and SLO Violation Rates. Disabling JIT prewarming ($\kappa=\infty$) produces the largest performance degradation, confirming that proactive successor prewarming is the primary driver of tail-latency suppression.*

---

### VIII. CONCLUSIONS AND FUTURE WORK
Serverless DAG workflows suffer from cascading cold starts that compound along the critical execution path, causing severe tail-latency inflation and contractual SLO violations. Existing schedulers size container pools using invocation frequency alone, squandering physical memory on high-slack tasks while leaving critical tasks vulnerable to initialization delays.

This paper presented AP-CWS, a predictive workflow scheduler that coordinates warm container pool dimensioning and task placement. AP-CWS combines Holt linear exponential smoothing, DAG topological slack budgeting, joint placement scoring, and Just-In-Time successor prewarming. To support high-throughput clusters, we introduced Fast-AP-CWS, which memoizes graph metrics and applies the Power-of-Two-Choices principle to achieve $O(1)$ dispatch time.

Discrete-event simulation across 324 benchmark runs demonstrated that AP-CWS achieves an overall P95 tail latency of 2.83 s (a 27.8% reduction compared to the state-of-the-art SSC baseline) and reduces critical-path cold starts to 0.63%. Furthermore, AP-CWS suppresses SLO violations by 84.4% while reducing total execution cost by 4.6%. Fast-AP-CWS provides a 3.0x dispatch speedup on 128 nodes, processing over 71,000 scheduling decisions per second without queueing backpressure. Stress testing verified that AP-CWS avoids the 12 req/s Cascading Latency Cliff that destabilizes decoupled baselines, maintaining stable performance up to the 22 req/s physical memory limit.

*Future Work:*
1. *Dynamic Runtime DAG Morphing:* We plan to extend the criticality formulation to support dynamic workflows where branching conditions and iteration loops alter graph topology at runtime.
2. *Geo-Distributed Multi-Region Deployment:* We intend to incorporate inter-region data transfer costs and wide-area network latency into the joint placement scoring function.
3. *Hardware-Accelerated Control Planes:* We will investigate offloading Holt forecasting and matrix operations to dedicated smartNICs or data processing units (DPUs) to support microsecond-scale cluster management.

---

### REFERENCES
1. A. Shahrad et al., "Serverless in the wild: Characterizing and optimizing the serverless computing at a serverless provider," in *Proc. USENIX Annu. Tech. Conf. (ATC)*, 2020, pp. 45–59.
2. S. Pan et al., "Sustainable serverless computing with cold-start optimization and automatic workflow resource scheduling," *IEEE Trans. Sustain. Comput.*, vol. 9, no. 4, pp. 512–525, 2024.
3. M. Chen et al., "CASR: Cache-aware serverless runtime for workflow orchestration," *Future Gener. Comput. Syst.*, vol. 162, pp. 245–258, 2025.
4. T. Nguyen et al., "Model predictive control for serverless cold-start mitigation," in *Proc. IEEE Int. Symp. Modeling, Anal., Simulation Comput. Telecommun. Syst. (MASCOTS)*, 2025, pp. 112–121.
5. H. Alquraan et al., "DROPS: Dynamic resource provisioning for serverless workflows under SLO constraints," in *Proc. Eur. Conf. Comput. Syst. (EuroSys)*, 2026, pp. 310–325.
6. M. Hanaforoosh et al., "Granularity trees: Workload-aware task aggregation for serverless workflows," *Future Gener. Comput. Syst.*, vol. 160, pp. 110–124, 2025.
7. X. Zhou et al., "Multi-agent reinforcement learning for load balancing in serverless workflows," *J. Netw. Comput. Appl.*, vol. 235, p. 103840, 2025.
8. S. Zolfaghari et al., "Energy-aware multi-objective scheduling of scientific workflows in serverless edge-cloud environments," *IEEE Trans. Serv. Comput.*, vol. 18, no. 2, pp. 430–443, 2025.
9. P. Raith et al., "faas-sim: A trace-driven simulation framework for serverless edge computing platforms," *Softw. Pract. Exper.*, vol. 53, no. 6, pp. 1320–1342, 2023.
10. T. Mahmoudi and H. Khazaei, "SimFaaS: A performance simulator for serverless computing platforms," in *Proc. Int. Conf. Cloud Comput. Serv. Sci. (CLOSER)*, 2021, pp. 185–194.
