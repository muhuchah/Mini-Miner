import csv
import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import math

import matplotlib.pyplot as plt


# =========================================================
# Logging / Debug
# =========================================================
DEBUG = False  # <- keep False for clean output

DBG_BLOCKCHAIN = False
DBG_NETWORK = False
DBG_MINER_LIFECYCLE = False
DBG_MINER_PROGRESS = False
DBG_MINER_SUBMIT = False
DBG_DIFFICULTY = False
DBG_REJECTS = False

MINER_PROGRESS_EVERY_NONCES = 50000
MINER_PROGRESS_EVERY_SEC = 2.0


def dprint(*args):
    if DEBUG:
        print(*args)


def info(msg: str):
    print(msg)


# =========================================================
# Utility / constants
# =========================================================
TARGET_BLOCK_TIME_SEC = 1.0
RETARGET_WINDOW = 3
MAX_HASH = 2**256 - 1


def now_ms() -> int:
    return int(time.time() * 1000)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def block_header_for_hash(block_dict: dict) -> bytes:
    h = block_dict["header"]
    s = f'{h["parent"]}|{h["digest"]}|{h["difficulty"]:.12f}|{h["timestamp"]}|{h["nonce"]}'
    return s.encode("utf-8")


def body_digest(body: str) -> str:
    return sha256_hex(body.encode("utf-8"))


def difficulty_to_target(difficulty: float) -> int:
    target = int(MAX_HASH / max(difficulty, 1e-12))
    return max(1, min(MAX_HASH, target))


def hash_meets_difficulty(block_hash_hex: str, difficulty: float) -> bool:
    target = difficulty_to_target(difficulty)
    return int(block_hash_hex, 16) <= target


def make_results_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path("results") / ts
    p.mkdir(parents=True, exist_ok=True)
    return p


# =========================================================
# Core data structures
# =========================================================
@dataclass
class Block:
    parent: str
    body: str
    difficulty: float
    timestamp: int
    nonce: int
    miner_id: str
    digest: str = field(init=False)
    block_hash: str = field(init=False)

    def __post_init__(self):
        self.digest = body_digest(self.body)
        d = self.to_dict()
        self.block_hash = sha256_hex(block_header_for_hash(d))

    def to_dict(self) -> dict:
        return {
            "header": {
                "parent": self.parent,
                "digest": self.digest,
                "difficulty": float(self.difficulty),
                "timestamp": int(self.timestamp),
                "nonce": int(self.nonce),
            },
            "body": self.body,
        }


class Blockchain:
    def __init__(self, genesis_difficulty: float = 50.0):
        self.lock = threading.RLock()
        self.blocks: Dict[str, Block] = {}
        self.height: Dict[str, int] = {}
        self.children: Dict[str, List[str]] = {}

        g = Block(
            parent="0" * 64,
            body="genesis",
            difficulty=genesis_difficulty,
            timestamp=now_ms(),
            nonce=0,
            miner_id="genesis",
        )
        self.genesis_hash = g.block_hash
        self.blocks[self.genesis_hash] = g
        self.height[self.genesis_hash] = 0
        self.children[self.genesis_hash] = []
        self.tip = self.genesis_hash

    def add_block(self, b: Block) -> bool:
        with self.lock:
            if b.block_hash in self.blocks:
                return False
            if b.parent not in self.blocks:
                return False
            if b.digest != body_digest(b.body):
                return False
            if not hash_meets_difficulty(b.block_hash, b.difficulty):
                return False

            self.blocks[b.block_hash] = b
            h = self.height[b.parent] + 1
            self.height[b.block_hash] = h
            self.children.setdefault(b.parent, []).append(b.block_hash)
            self.children.setdefault(b.block_hash, [])

            tip_h = self.height[self.tip]
            if h > tip_h or (h == tip_h and b.block_hash < self.tip):
                self.tip = b.block_hash
            return True

    def get_tip(self) -> Tuple[str, int]:
        with self.lock:
            return self.tip, self.height[self.tip]

    def get_main_chain_hashes(self) -> List[str]:
        with self.lock:
            chain = []
            cur = self.tip
            while cur in self.blocks:
                chain.append(cur)
                if cur == self.genesis_hash:
                    break
                cur = self.blocks[cur].parent
            chain.reverse()
            return chain

    def get_main_chain_blocks(self) -> List[Block]:
        hs = self.get_main_chain_hashes()
        with self.lock:
            return [self.blocks[h] for h in hs]

    def next_difficulty(self, parent_hash: str) -> float:
        with self.lock:
            parent = self.blocks[parent_hash]
            parent_h = self.height[parent_hash]
            old_diff = parent.difficulty
            next_height = parent_h + 1

            if next_height == 0 or next_height % RETARGET_WINDOW != 0:
                return old_diff

            cur = parent_hash
            endpoint_new = self.blocks[cur]
            for _ in range(RETARGET_WINDOW - 1):
                cur = self.blocks[cur].parent
            endpoint_old = self.blocks[cur]

            actual_timespan = max((endpoint_new.timestamp - endpoint_old.timestamp) / 1000.0, 0.001)
            expected_timespan = RETARGET_WINDOW * TARGET_BLOCK_TIME_SEC
            ratio = expected_timespan / actual_timespan
            ratio = max(0.25, min(4.0, ratio))
            return max(1.0, old_diff * ratio)

    def orphan_stats(self) -> Tuple[int, Dict[str, int]]:
        with self.lock:
            main = set(self.get_main_chain_hashes())
            all_hashes = set(self.blocks.keys())
            orphans = all_hashes - main
            by_miner: Dict[str, int] = {}
            for h in orphans:
                m = self.blocks[h].miner_id
                by_miner[m] = by_miner.get(m, 0) + 1
            return len(orphans), by_miner


class Miner(threading.Thread):
    def __init__(self, miner_id: str, network: "NetworkSimulator", per_hash_delay: float = 0.001, body_prefix: str = "txs"):
        super().__init__(daemon=True)
        self.miner_id = miner_id
        self.network = network
        self.per_hash_delay = per_hash_delay
        self.body_prefix = body_prefix
        self.running = threading.Event()
        self.running.set()
        self.local_tip_hash = None
        self.local_tip_height = 0
        self.rand = random.Random()

    def stop(self):
        self.running.clear()

    def notify_new_tip(self, tip_hash: str, tip_height: int):
        if tip_height >= self.local_tip_height:
            self.local_tip_hash = tip_hash
            self.local_tip_height = tip_height

    def run(self):
        tip, h = self.network.blockchain.get_tip()
        self.local_tip_hash, self.local_tip_height = tip, h
        last_progress_t = time.time()

        while self.running.is_set():
            parent = self.local_tip_hash
            parent_h = self.local_tip_height
            difficulty = self.network.blockchain.next_difficulty(parent)

            body = f"{self.body_prefix}:{self.miner_id}:{now_ms()}:{self.rand.randint(0, 1<<30)}"
            nonce = 0

            while self.running.is_set():
                if self.local_tip_hash != parent or self.local_tip_height != parent_h:
                    break

                b = Block(
                    parent=parent,
                    body=body,
                    difficulty=difficulty,
                    timestamp=now_ms(),
                    nonce=nonce,
                    miner_id=self.miner_id,
                )

                if hash_meets_difficulty(b.block_hash, difficulty):
                    self.network.submit_block(b, from_miner=self.miner_id)
                    break

                nonce += 1

                if DBG_MINER_PROGRESS:
                    now_t = time.time()
                    if nonce % MINER_PROGRESS_EVERY_NONCES == 0 or (now_t - last_progress_t) >= MINER_PROGRESS_EVERY_SEC:
                        dprint(f"[MINER HEARTBEAT] {self.miner_id} h={parent_h} nonce={nonce} diff={difficulty:.2f}")
                        last_progress_t = now_t

                if self.per_hash_delay > 0:
                    time.sleep(self.per_hash_delay)


class NetworkSimulator:
    def __init__(self, genesis_difficulty: float = 50.0):
        self.blockchain = Blockchain(genesis_difficulty=genesis_difficulty)
        self.miners: Dict[str, Miner] = {}
        self.miners_lock = threading.Lock()
        self.delay: Dict[Tuple[str, str], float] = {}
        self.forks_created = 0
        self.forks_lock = threading.Lock()
        self.new_block_event = threading.Event()

    def set_pair_delay(self, src: str, dst: str, d: float):
        self.delay[(src, dst)] = d

    def get_delay(self, src: str, dst: str) -> float:
        return self.delay.get((src, dst), 0.0)

    def add_miner(self, miner_id: str, per_hash_delay: float):
        m = Miner(miner_id, self, per_hash_delay=per_hash_delay)
        with self.miners_lock:
            self.miners[miner_id] = m
        m.start()

    def remove_miner(self, miner_id: str):
        with self.miners_lock:
            m = self.miners.pop(miner_id, None)
        if m:
            m.stop()
            m.join(timeout=2.0)

    def stop_all(self):
        with self.miners_lock:
            ids = list(self.miners.keys())
        for mid in ids:
            self.remove_miner(mid)

    def submit_block(self, b: Block, from_miner: str):
        cur_tip, _ = self.blockchain.get_tip()
        if b.parent != cur_tip:
            with self.forks_lock:
                self.forks_created += 1

        accepted = self.blockchain.add_block(b)
        if not accepted:
            return

        self.new_block_event.set()
        self.new_block_event.clear()

        with self.miners_lock:
            miners_list = list(self.miners.values())

        for m in miners_list:
            d = self.get_delay(from_miner, m.miner_id)
            threading.Thread(target=self._deliver_tip_after_delay, args=(m, d), daemon=True).start()

    def _deliver_tip_after_delay(self, miner: Miner, delay_s: float):
        if delay_s > 0:
            time.sleep(delay_s)
        tip, h = self.blockchain.get_tip()
        miner.notify_new_tip(tip, h)


# =========================================================
# Helpers
# =========================================================
def wait_until_main_height(
    net: NetworkSimulator,
    target_height: int,
    log_prefix: str = "",
    log_every_sec: float = 2.0,
    timeout_sec: Optional[float] = None,
):
    t0 = time.time()
    last_log = t0
    last_h = -1

    while True:
        tip, h = net.blockchain.get_tip()
        now = time.time()
        elapsed = now - t0

        if h != last_h:
            if DEBUG:
                dprint(f"{log_prefix}[HEIGHT] {last_h} -> {h} ({tip[:10]}..)")
            last_h = h

        if now - last_log >= log_every_sec:
            info(f"{log_prefix}elapsed={elapsed:.1f}s | main_height={h}/{target_height}")
            last_log = now

        if h >= target_height:
            return True, elapsed, h

        if timeout_sec is not None and elapsed >= timeout_sec:
            return False, elapsed, h

        time.sleep(0.05)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# =========================================================
# Experiment A
# =========================================================
def experiment_a(results_dir: Path):
    info("\n=== Experiment A: single miner, difficulty adaptation ===")
    net = NetworkSimulator(genesis_difficulty=50000.0)
    net.add_miner("M1", per_hash_delay=0.001)

    reached, elapsed, _ = wait_until_main_height(net, target_height=30, log_prefix="  [A] ", timeout_sec=1800)
    net.stop_all()

    blocks = net.blockchain.get_main_chain_blocks()
    rows = []
    for h, b in enumerate(blocks):
        rows.append({
            "height": h,
            "timestamp_ms": b.timestamp,
            "difficulty": b.difficulty,
            "miner": b.miner_id,
            "hash_prefix": b.block_hash[:16],
        })

    write_csv(results_dir / "experiment_a_blocks.csv", rows, ["height", "timestamp_ms", "difficulty", "miner", "hash_prefix"])
    info(f"[A] done | reached={reached} | elapsed={elapsed:.2f}s | saved experiment_a_blocks.csv")

    return {"reached": reached, "elapsed_sec": elapsed, "final_height": len(blocks) - 1}


# =========================================================
# Experiment B
# =========================================================
def run_b_once(n_miners: int, target_h: int = 25) -> float:
    net = NetworkSimulator(genesis_difficulty=2000.0)
    for i in range(n_miners):
        net.add_miner(f"M{i+1}", per_hash_delay=0.001)
    reached, elapsed, _ = wait_until_main_height(net, target_h, log_prefix=f"  [B {n_miners}m] ", timeout_sec=1800)
    net.stop_all()
    return elapsed if reached else float("nan")


def schedule_for_experiment_b(net: NetworkSimulator, base_delay: float = 0.001):
    """
    Timeline:
      t=0   : already 1 miner running
      t=5   : +1 miner  (total 2)
      t=10  : +1 miner  (total 3)
      t=15  : +1 miner  (total 4)
      t=20  : -1 miner  (total 3)
      t=25  : -1 miner  (total 2)
      t=30  : -1 miner  (total 1)
      t=35  : -1 miner  (total 0) -> finish
    """
    miner_ids = ["B_M1", "B_M2", "B_M3", "B_M4"]

    # start with one miner
    net.add_miner(miner_ids[0], per_hash_delay=base_delay)
    info("[B] t=0s  -> start with 1 miner (B_M1)")

    # additions
    time.sleep(5)
    net.add_miner(miner_ids[1], per_hash_delay=base_delay)
    info("[B] t=5s  -> added B_M2 (total=2)")

    time.sleep(5)
    net.add_miner(miner_ids[2], per_hash_delay=base_delay)
    info("[B] t=10s -> added B_M3 (total=3)")

    time.sleep(5)
    net.add_miner(miner_ids[3], per_hash_delay=base_delay)
    info("[B] t=15s -> added B_M4 (total=4)")

    # removals
    time.sleep(5)
    net.remove_miner(miner_ids[3])
    info("[B] t=20s -> removed B_M4 (total=3)")

    time.sleep(5)
    net.remove_miner(miner_ids[2])
    info("[B] t=25s -> removed B_M3 (total=2)")

    time.sleep(5)
    net.remove_miner(miner_ids[1])
    info("[B] t=30s -> removed B_M2 (total=1)")

    time.sleep(5)
    net.remove_miner(miner_ids[0])
    info("[B] t=35s -> removed B_M1 (total=0)")



def experiment_b(results_dir: Path):
    info("\n=== Experiment B: Dynamic miner count (add/remove every 5s) ===")

    net = NetworkSimulator(genesis_difficulty=2000.0)

    # run schedule in a separate thread so mining/network continues
    sched_thread = threading.Thread(
        target=schedule_for_experiment_b,
        args=(net, 0.001),
        daemon=True
    )
    t0 = time.time()
    sched_thread.start()

    # wait until schedule completes
    sched_thread.join()

    # give delayed tips a moment to settle
    time.sleep(1.0)

    # ensure all miners are stopped
    net.stop_all()

    # collect main chain
    blocks = net.blockchain.get_main_chain_blocks()
    # skip genesis for interval stats
    non_gen = blocks[1:] if len(blocks) > 1 else []

    # build per-block rows
    rows = []
    prev_ts = None
    intervals = []
    for h, b in enumerate(blocks):
        rel_t = (b.timestamp / 1000.0) - t0
        interval = None
        if prev_ts is not None:
            interval = (b.timestamp - prev_ts) / 1000.0
            intervals.append(interval)
        prev_ts = b.timestamp

        rows.append({
            "height": h,
            "time_from_start_sec": rel_t,
            "difficulty": b.difficulty,
            "miner": b.miner_id,
            "block_interval_sec": "" if interval is None else interval,
            "hash_prefix": b.block_hash[:16],
        })

    write_csv(
        results_dir / "experiment_b_dynamic.csv",
        rows,
        ["height", "time_from_start_sec", "difficulty", "miner", "block_interval_sec", "hash_prefix"]
    )

    # ---- Plot 1: difficulty change over time
    if non_gen:
        x_t = [((b.timestamp / 1000.0) - t0) for b in non_gen]
        y_d = [b.difficulty for b in non_gen]

        plt.figure()
        plt.plot(x_t, y_d, marker="o")
        plt.xlabel("Time from start (s)")
        plt.ylabel("Difficulty")
        plt.title("Experiment B: Difficulty vs Time (dynamic miners)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "experiment_b_difficulty.png")
        plt.close()
    else:
        # create empty placeholder plot
        plt.figure()
        plt.title("Experiment B: Difficulty vs Time (no mined blocks)")
        plt.tight_layout()
        plt.savefig(results_dir / "experiment_b_difficulty.png")
        plt.close()

    # ---- Plot 2: average block time in 5-second bins
    # bins: [0,5), [5,10), ... up to schedule end (~35s) + a tail
    if len(blocks) >= 2:
        times = [((b.timestamp / 1000.0) - t0) for b in blocks]  # includes genesis at height0
        block_intervals = [ (blocks[i].timestamp - blocks[i-1].timestamp)/1000.0 for i in range(1, len(blocks)) ]
        # assign each interval to the time of the newer block
        interval_times = times[1:]

        end_t = max(40.0, max(interval_times) + 1.0)
        n_bins = int(math.ceil(end_t / 5.0))
        bin_centers = []
        bin_avgs = []

        for bi in range(n_bins):
            lo = bi * 5.0
            hi = (bi + 1) * 5.0
            vals = [v for tt, v in zip(interval_times, block_intervals) if lo <= tt < hi]
            avg_v = sum(vals) / len(vals) if vals else float("nan")
            bin_centers.append((lo + hi) / 2.0)
            bin_avgs.append(avg_v)

        plt.figure()
        plt.plot(bin_centers, bin_avgs, marker="o")
        plt.xlabel("Time bin center (s)")
        plt.ylabel("Average block interval (s)")
        plt.title("Experiment B: Avg block time per 5s window")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(results_dir / "experiment_b_avg_block_time.png")
        plt.close()

        overall_avg = sum(block_intervals) / len(block_intervals)
    else:
        plt.figure()
        plt.title("Experiment B: Avg block time per 5s window (insufficient blocks)")
        plt.tight_layout()
        plt.savefig(results_dir / "experiment_b_avg_block_time.png")
        plt.close()
        overall_avg = float("nan")

    info(
        f"[B] done | blocks={len(blocks)-1} (excluding genesis) | "
        f"overall_avg_block_time={overall_avg:.4f}s | "
        f"saved experiment_b_dynamic.csv + 2 plots"
    )

    return {
        "blocks_excluding_genesis": max(0, len(blocks)-1),
        "overall_avg_block_time_sec": overall_avg,
        "artifacts": [
            "experiment_b_dynamic.csv",
            "experiment_b_difficulty.png",
            "experiment_b_avg_block_time.png",
        ],
    }


# =========================================================
# Experiment C
# =========================================================
def experiment_c(results_dir: Path):
    info("\n=== Experiment C: miner share (fast vs slow) ===")
    net = NetworkSimulator(genesis_difficulty=50.0)
    net.add_miner("M1_fast", per_hash_delay=0.0008)
    net.add_miner("M2_slow", per_hash_delay=0.0024)

    reached, elapsed, _ = wait_until_main_height(net, target_height=40, log_prefix="  [C] ", timeout_sec=300)
    net.stop_all()

    main_blocks = net.blockchain.get_main_chain_blocks()[1:]  # exclude genesis
    counts = {"M1_fast": 0, "M2_slow": 0}
    for b in main_blocks:
        if b.miner_id in counts:
            counts[b.miner_id] += 1

    total = max(1, len(main_blocks))
    rows = [
        {"miner": "M1_fast", "main_blocks": counts["M1_fast"], "share": counts["M1_fast"] / total},
        {"miner": "M2_slow", "main_blocks": counts["M2_slow"], "share": counts["M2_slow"] / total},
    ]
    write_csv(results_dir / "experiment_c_shares.csv", rows, ["miner", "main_blocks", "share"])

    plt.figure()
    plt.bar([r["miner"] for r in rows], [r["share"] for r in rows])
    plt.ylim(0, 1)
    plt.ylabel("Main chain share")
    plt.title("Experiment C: Main-chain block share")
    plt.tight_layout()
    plt.savefig(results_dir / "experiment_c_shares.png")
    plt.close()

    info(f"[C] done | reached={reached} | elapsed={elapsed:.2f}s | saved CSV+plot")
    return {"reached": reached, "elapsed_sec": elapsed, "counts": counts}


# =========================================================
# Experiment D
# =========================================================
def run_d_with_delta(delta: float):
    info(f"\n[D] START delta={delta}s")
    net = NetworkSimulator(genesis_difficulty=2000.0)

    net.add_miner("M1_fast", per_hash_delay=0.0008)
    net.add_miner("M2_slow", per_hash_delay=0.0024)

    net.set_pair_delay("M1_fast", "M2_slow", delta)
    net.set_pair_delay("M2_slow", "M1_fast", delta)

    reached, elapsed, final_h = wait_until_main_height(
        net, target_height=40, log_prefix=f"  [D {delta}s] ", log_every_sec=2.0, timeout_sec=600
    )

    _, h2 = net.blockchain.get_tip()
    if not reached and h2 >= 40:
        reached = True
        final_h = h2

    net.stop_all()

    forks = net.forks_created
    orphan_total, orphan_by_miner = net.blockchain.orphan_stats()

    info(
        f"[D] END delta={delta}s | reached={reached} | elapsed={elapsed:.2f}s | "
        f"final_h={final_h} | forks={forks} | orphans={orphan_total}"
    )
    return {
        "delta_sec": delta,
        "elapsed_sec": elapsed,
        "forks": forks,
        "orphans_total": orphan_total,
        "reached": reached,
        "final_height": final_h,
        "orphans_by_miner": orphan_by_miner,
    }


def experiment_d(results_dir: Path):
    info("\n=== Experiment D: network delay and forks ===")
    rows = []
    for delta in [0.5, 1.0, 2.0]:
        rows.append(run_d_with_delta(delta))

    write_csv(
        results_dir / "experiment_d_delays.csv",
        rows,
        ["delta_sec", "elapsed_sec", "forks", "orphans_total", "reached", "final_height", "orphans_by_miner"],
    )

    # Plot 1: time vs delay
    plt.figure()
    plt.plot([r["delta_sec"] for r in rows], [r["elapsed_sec"] for r in rows], marker="o")
    plt.xlabel("Network delay delta (s)")
    plt.ylabel("Time to 40 blocks (s)")
    plt.title("Experiment D: Time vs Network Delay")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "experiment_d_time_vs_delay.png")
    plt.close()

    # Plot 2: forks/orphans vs delay
    x = [r["delta_sec"] for r in rows]
    forks = [r["forks"] for r in rows]
    orph = [r["orphans_total"] for r in rows]

    plt.figure()
    plt.plot(x, forks, marker="o", label="Forks")
    plt.plot(x, orph, marker="s", label="Orphans")
    plt.xlabel("Network delay delta (s)")
    plt.ylabel("Count")
    plt.title("Experiment D: Forks / Orphans vs Delay")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "experiment_d_forks_orphans.png")
    plt.close()

    info("[D] done | saved experiment_d_delays.csv + 2 plots")
    return rows


# =========================================================
# Main
# =========================================================
def main():
    results_dir = make_results_dir()
    info(f"[RUN] results_dir={results_dir}")

    summary = {}

    # summary["experiment_a"] = experiment_a(results_dir)
    # summary["experiment_b"] = experiment_b(results_dir)
    # summary["experiment_c"] = experiment_c(results_dir)
    summary["experiment_d"] = experiment_d(results_dir)

    with open(results_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    info(f"\n[RUN] ALL DONE. Outputs saved in: {results_dir}")


if __name__ == "__main__":
    main()

