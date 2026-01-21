from typing import Dict, List, Tuple, Optional
import math

# -------------------------
# Robust penalty (Huber)
# -------------------------
def huber(x: float, delta: float = 10.0) -> float:
    ax = abs(x)
    if ax <= delta:
        return 0.5 * ax * ax
    return delta * (ax - 0.5 * delta)

def l2(p: List[float], q: List[float]) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))

# -------------------------
# 1) compute template edge lengths mu_{jk}
# template_kp_dict: {idx: {"xy":[x,y], ...}}
# -------------------------
def compute_template_lengths(
    template_kp_dict: Dict[int, Dict],
    edges: List[Tuple[int, int]],
) -> Dict[Tuple[int, int], float]:
    mu = {}
    for j, k in edges:
        if j not in template_kp_dict or k not in template_kp_dict:
            continue
        pj = template_kp_dict[j]["xy"]
        pk = template_kp_dict[k]["xy"]
        mu[(j, k)] = l2(pj, pk)
        mu[(k, j)] = mu[(j, k)]
    return mu

# -------------------------
# 2) build adjacency on available nodes
# candidates: {idx: [[x,y],[x,y],[x,y]]}  (K can vary)
# edges: global kinematic edges
# -------------------------
def build_adjacency(
    candidates: Dict[int, List[List[float]]],
    edges: List[Tuple[int, int]],
) -> Dict[int, List[int]]:
    adj = {u: [] for u in candidates.keys()}
    for a, b in edges:
        if a in candidates and b in candidates:
            adj[a].append(b)
            adj[b].append(a)
    return adj

# -------------------------
# 3) root a forest (because missing nodes can split graph)
# -------------------------
def get_forest_roots(adj: Dict[int, List[int]]) -> List[int]:
    nodes = list(adj.keys())
    visited = set()
    roots = []
    for s in nodes:
        if s in visited:
            continue
        # BFS to mark component
        stack = [s]
        visited.add(s)
        comp = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    stack.append(v)
                    comp.append(v)
        # choose a stable root: prefer hips if exists, else arbitrary
        # you can customize this (e.g., 9 or 10)
        root = None
        for prefer in (9, 10, 5, 6):  # hips first, then shoulders
            if prefer in comp:
                root = prefer
                break
        roots.append(root if root is not None else comp[0])
    return roots

# -------------------------
# 4) Tree DP on an acyclic component
# dp[u][mu] = best cost of subtree rooted at u when u uses candidate mu
# back[u][mu][child] = argmin candidate index for child
# unary optional: {idx: [cost0, cost1, ...]}  length = K_u
# pairwise: huber(|dist - template_length|) on each kinematic edge
# -------------------------
def solve_component_tree_dp(
    root: int,
    adj: Dict[int, List[int]],
    candidates: Dict[int, List[List[float]]],
    mu_len: Dict[Tuple[int, int], float],
    lam: float = 1.0,
    delta: float = 10.0,
    unary: Optional[Dict[int, List[float]]] = None,
) -> Dict[int, int]:
    # build parent/children by DFS (assume graph is tree-like; if small loops exist, this DP is not valid)
    parent = {root: -1}
    order = [root]
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v == parent[u]:
                continue
            if v in parent:
                # loop detected; for simplified version, just skip revisiting
                # (later we can add ICM or break cycles explicitly)
                continue
            parent[v] = u
            stack.append(v)
            order.append(v)

    # children lists
    children = {u: [] for u in order}
    for u in order:
        for v in adj[u]:
            if parent.get(v, None) == u:
                children[u].append(v)

    # init dp/back
    dp = {}
    back = {}  # back[u][i][child] = best child cand idx
    for u in order:
        Ku = len(candidates[u])
        dp[u] = [0.0] * Ku
        back[u] = [dict() for _ in range(Ku)]

    # post-order (reverse DFS order)
    for u in reversed(order):
        Ku = len(candidates[u])
        # unary cost
        unary_u = unary[u] if (unary is not None and u in unary) else [0.0] * Ku

        for iu in range(Ku):
            cost = float(unary_u[iu])

            pu = candidates[u][iu]
            # add best contributions from each child
            for c in children[u]:
                Kc = len(candidates[c])
                best_val = None
                best_ic = 0

                # template reference length for edge (u,c)
                ref = mu_len.get((u, c), None)
                # if template length missing, treat pairwise cost as 0 (safe)
                for ic in range(Kc):
                    pc = candidates[c][ic]
                    pair = 0.0
                    if ref is not None:
                        d = l2(pu, pc)
                        pair = huber(d - ref, delta=delta)

                    val = dp[c][ic] + lam * pair
                    if best_val is None or val < best_val:
                        best_val = val
                        best_ic = ic

                cost += float(best_val) if best_val is not None else 0.0
                back[u][iu][c] = best_ic

            dp[u][iu] = cost

    # choose best root state
    Kr = len(candidates[root])
    best_ir = min(range(Kr), key=lambda i: dp[root][i])

    # backtrack
    chosen = {}
    def dfs_assign(u: int, iu: int):
        chosen[u] = iu
        for c in children[u]:
            ic = back[u][iu][c]
            dfs_assign(c, ic)

    dfs_assign(root, best_ir)
    return chosen

# -------------------------
# 5) Full solver: forest components
# returns: chosen_index {idx: m_idx}, and chosen_xy {idx: [x,y]}
# -------------------------
def select_keypoints_by_kinematic_dp(
    template_kp_dict: Dict[int, Dict],
    candidates: Dict[int, List[List[float]]],
    edges: List[Tuple[int, int]],
    lam: float = 1.0,
    delta: float = 10.0,
    unary: Optional[Dict[int, List[float]]] = None,
) -> Tuple[Dict[int, int], Dict[int, List[float]]]:
    # compute template lengths
    mu_len = compute_template_lengths(template_kp_dict, edges)

    # build adjacency on available nodes only
    adj = build_adjacency(candidates, edges)
    if not adj:
        return {}, {}

    roots = get_forest_roots(adj)

    chosen_index = {}
    for r in roots:
        comp_choice = solve_component_tree_dp(
            root=r, adj=adj, candidates=candidates, mu_len=mu_len,
            lam=lam, delta=delta, unary=unary
        )
        chosen_index.update(comp_choice)

    chosen_xy = {idx: candidates[idx][m] for idx, m in chosen_index.items()}
    return chosen_index, chosen_xy


from typing import Dict, Any, List, Tuple
import numpy as np

KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]

def keep_common_add_id_and_stack(
    kp_dict_list: List[Dict[int, Any]],
    order: List[int] = KEY_BODY,
    xy_key: str = "xy",
    out_key: str = "xy",        # 覆盖 xy -> [x,y,idx]；不想覆盖可改 "xy_id"
    dtype=np.float32,
) -> Tuple[List[Dict[int, Any]], List[int], np.ndarray]:
    """
    输入:
      kp_dict_list: [ {idx: {"xy":[x,y], ...}}, ... ]
    输出:
      filtered_list: list，长度 L；每个 dict 只保留共同 idx，并把 xy 扩成 [x,y,idx]
      common_ids: 共同 idx 的排序列表（先按 order，再按数值）
      arr: np.ndarray, shape (L, N, 3)，每行是 [x, y, idx]
    """
    L = len(kp_dict_list)
    if L == 0:
        return [], [], np.zeros((0, 0, 3), dtype=dtype)

    # 1) 共同 key
    common = set(kp_dict_list[0].keys())
    for d in kp_dict_list[1:]:
        common &= set(d.keys())

    # 2) 排序：KEY_BODY 优先
    common_in_order = [k for k in order if k in common]
    common_extra = sorted([k for k in common if k not in set(order)])
    common_ids = common_in_order + common_extra
    N = len(common_ids)

    # 3) 生成 filtered_list + arr
    filtered_list: List[Dict[int, Any]] = []
    arr = np.full((L, N, 3), np.nan, dtype=dtype)

    for li, d in enumerate(kp_dict_list):
        nd = {}
        for ni, idx in enumerate(common_ids):
            v = d[idx]
            xy = v.get(xy_key, None)
            if xy is None or len(xy) < 2:
                raise ValueError(f"Invalid xy for idx={idx} at list[{li}]")
            x, y = float(xy[0]), float(xy[1])

            vv = dict(v)
            vv[out_key] = [x, y, int(idx)]
            nd[int(idx)] = vv

            arr[li, ni, 0] = x
            arr[li, ni, 1] = y
            arr[li, ni, 2] = float(idx)

        filtered_list.append(nd)

    return filtered_list, common_ids, arr
