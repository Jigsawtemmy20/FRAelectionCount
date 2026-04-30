from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional


Mode = Literal["single_seat_rcv", "multi_seat_stv"]


# -----------------------------
# Core data structures
# -----------------------------
@dataclass
class Candidate:
    candidate_id: str
    withdrawn: bool = False


@dataclass
class Ranking:
    rank: int
    candidate_ids: List[str]


@dataclass
class Ballot:
    ballot_id: str
    rankings: List[Ranking] = field(default_factory=list)
    current_transfer_value: float = 1.0

    def is_undervote(self) -> bool:
        return len(self.rankings) == 0

    def sorted_rankings(self) -> List[Ranking]:
        return sorted(self.rankings, key=lambda r: r.rank)

    def highest_ranked_active(
        self,
        active_candidate_ids: set[str],
    ) -> Optional[str]:
        """
        Implements Sec. 322(c) and Section 6 of the spec:

        - Undervotes: no rankings → ballot never counts.
        - Skipped rankings: allowed; we just move to the next rank that has any candidates.
        - Repeated rankings: earliest usable active ranking wins.
        - Same-rank groups: if the first rank at which any active candidate appears
          contains more than one active candidate, the ballot becomes inactive.
        """
        if self.is_undervote():
            return None

        for ranking in self.sorted_rankings():
            active_in_rank = [cid for cid in ranking.candidate_ids if cid in active_candidate_ids]

            if not active_in_rank:
                # All candidates at this rank are inactive; continue to later ranks
                continue

            # This is the first rank where any active candidate appears
            if len(active_in_rank) == 1:
                return active_in_rank[0]
            else:
                # Same-rank group with multiple active candidates: ballot becomes inactive
                return None

        # All ranked candidates inactive
        return None


@dataclass
class Election:
    election_id: str
    seat_count: int
    mode: Mode
    candidates: List[Candidate]
    ballots: List[Ballot]
    tie_break_order: List[str]
    max_ranks_allowed: Optional[int] = None

    # Internal state
    round_number: int = 0
    threshold: Optional[float] = None
    round_log: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        # seat_count ≥ 1
        if self.seat_count < 1:
            raise ValueError("seat_count must be at least 1")

        # mode valid
        if self.mode not in ("single_seat_rcv", "multi_seat_stv"):
            raise ValueError("mode must be 'single_seat_rcv' or 'multi_seat_stv'")

        # Candidate IDs unique
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate IDs must be unique")

        # tie_break_order contains all candidates exactly once
        if set(self.tie_break_order) != set(ids):
            raise ValueError("tie_break_order must contain each candidate exactly once")

        valid_ids = set(ids)

        # Ballot validation
        for ballot in self.ballots:
            for ranking in ballot.rankings:
                if ranking.rank <= 0:
                    raise ValueError(
                        f"Ballot {ballot.ballot_id} has invalid rank {ranking.rank}"
                    )
                for cid in ranking.candidate_ids:
                    if cid not in valid_ids:
                        raise ValueError(
                            f"Ballot {ballot.ballot_id} references unknown candidate '{cid}'"
                        )

        # max_ranks_allowed
        if self.max_ranks_allowed is not None:
            for ballot in self.ballots:
                for ranking in ballot.rankings:
                    if ranking.rank > self.max_ranks_allowed:
                        raise ValueError(
                            f"Ballot {ballot.ballot_id} exceeds max_ranks_allowed={self.max_ranks_allowed}"
                        )


# -----------------------------
# Helper functions
# -----------------------------
def _initial_candidate_status(election: Election) -> Dict[str, str]:
    status: Dict[str, str] = {}
    for c in election.candidates:
        status[c.candidate_id] = "withdrawn" if c.withdrawn else "active"
    return status


def _active_candidates(status: Dict[str, str]) -> List[str]:
    return [cid for cid, s in status.items() if s == "active"]


def _elected_candidates(status: Dict[str, str]) -> List[str]:
    return [cid for cid, s in status.items() if s == "elected"]


def _tie_break(
    tied_ids: List[str],
    tie_break_order: List[str],
) -> str:
    order_index = {cid: i for i, cid in enumerate(tie_break_order)}
    return min(tied_ids, key=lambda cid: order_index[cid])


def _count_votes_single_round(
    election: Election,
    status: Dict[str, str],
    use_transfer_values: bool,
) -> Dict[str, float]:
    """
    Counts votes for the current round.

    - Only active candidates receive votes.
    - Each ballot contributes either 1.0 (RCV) or its current_transfer_value (STV)
      to its highest-ranked active candidate.
    """
    active_set = set(_active_candidates(status))
    totals: Dict[str, float] = {cid: 0.0 for cid in status.keys()}

    for ballot in election.ballots:
        if ballot.is_undervote():
            continue
        cid = ballot.highest_ranked_active(active_set)
        if cid is None:
            continue
        value = ballot.current_transfer_value if use_transfer_values else 1.0
        totals[cid] += value

    return totals


def _compute_threshold(first_round_total: float, seat_count: int) -> float:
    # Sec. 324(2): floor(first_round_active_votes / (seat_count + 1)) + 1
    return math.floor(first_round_total / (seat_count + 1)) + 1


def _truncate_4(x: float) -> float:
    # Sec. 324(7): truncate to 4 decimal places
    return math.floor(x * 10_000) / 10_000.0


def _apply_threshold_to_elected(
    totals: Dict[str, float],
    status: Dict[str, str],
    threshold: Optional[float],
) -> None:
    """
    For logging purposes, in future rounds an elected candidate is deemed
    to have vote total equal to the threshold (Sec. 322(b)(2)(B)).
    We do not count new ballots for them, but we reflect the threshold in totals.
    """
    if threshold is None:
        return
    for cid, s in status.items():
        if s == "elected":
            totals[cid] = threshold


# -----------------------------
# Single-seat RCV (Sec. 322(a))
# -----------------------------
def run_single_seat_rcv(election: Election) -> Dict:
    status = _initial_candidate_status(election)
    rounds: List[Dict] = []

    while True:
        election.round_number += 1
        totals = _count_votes_single_round(election, status, use_transfer_values=False)

        rounds.append(
            {
                "round": election.round_number,
                "vote_totals": {cid: totals.get(cid, 0.0) for cid in status.keys()},
                "status": status.copy(),
                "action": None,
            }
        )

        active = _active_candidates(status)
        if len(active) <= 2:
            # When two or fewer active candidates remain, highest vote total wins (Sec. 322(a)(3))
            max_votes = max(totals.get(cid, 0.0) for cid in active) if active else 0.0
            top = [cid for cid in active if totals.get(cid, 0.0) == max_votes]
            winner = (
                _tie_break(top, election.tie_break_order) if len(top) > 1 else top[0]
            )
            status[winner] = "elected"
            rounds[-1]["action"] = {"type": "elect", "candidate": winner}
            break

        # Eliminate lowest active candidate (Sec. 322(a)(2))
        min_votes = min(totals.get(cid, 0.0) for cid in active)
        lowest = [cid for cid in active if totals.get(cid, 0.0) == min_votes]
        eliminated = (
            _tie_break(lowest, election.tie_break_order)
            if len(lowest) > 1
            else lowest[0]
        )
        status[eliminated] = "eliminated"
        rounds[-1]["action"] = {"type": "eliminate", "candidate": eliminated}

    final_status = status.copy()
    winners = [cid for cid, s in final_status.items() if s == "elected"]

    return {
        "winners": winners,
        "rounds": rounds,
        "final_candidate_status": final_status,
    }


# -----------------------------
# Multi-seat STV (Sec. 322(b))
# -----------------------------
def run_multi_seat_stv(election: Election) -> Dict:
    status = _initial_candidate_status(election)
    rounds: List[Dict] = []

    # First round: count votes and compute threshold
    election.round_number += 1
    totals = _count_votes_single_round(election, status, use_transfer_values=True)
    first_round_total = sum(
        v for cid, v in totals.items() if status[cid] == "active"
    )
    election.threshold = _compute_threshold(first_round_total, election.seat_count)

    rounds.append(
        {
            "round": election.round_number,
            "vote_totals": totals.copy(),
            "status": status.copy(),
            "threshold": election.threshold,
            "action": None,
        }
    )

    while True:
        active = _active_candidates(status)
        elected = _elected_candidates(status)
        seats_filled = len(elected)
        seats_remaining = election.seat_count - seats_filled

        # Completion condition (Sec. 322(b)(4))
        if seats_remaining <= 0:
            break

        if len(active) + seats_filled <= election.seat_count:
        # Elect all remaining active candidates
            for cid in active:
                status[cid] = "elected"

            rounds.append(
            {
            "round": election.round_number,
            "vote_totals": totals.copy(),
            "status": status.copy(),
            "threshold": election.threshold,
            "action": {
                "type": "fill_remaining_seats",
                "candidates": active,
            },
        }
    )
            break

        # Check for candidates meeting or exceeding threshold (Sec. 322(b)(2))
        elected_this_round: List[str] = []
        for cid in active:
            if totals.get(cid, 0.0) >= (election.threshold or 0):
                status[cid] = "elected"
                elected_this_round.append(cid)

        if elected_this_round:
            # Distribute surpluses simultaneously (Sec. 322(b)(2)(C))
            action_detail = []
            for cid in elected_this_round:
                vote_total = totals.get(cid, 0.0)
                if vote_total <= 0:
                    action_detail.append(
                        {"candidate": cid, "surplus_fraction": 0.0}
                    )
                    continue

                surplus_fraction = _truncate_4(
                    (vote_total - (election.threshold or 0)) / vote_total
                )
                if surplus_fraction < 0:
                    surplus_fraction = 0.0

                action_detail.append(
                    {"candidate": cid, "surplus_fraction": surplus_fraction}
                )

                if surplus_fraction <= 0:
                    # Candidate reached threshold exactly; no value transfers, but logic preserved
                    continue

                # For each ballot currently counting for this candidate,
                # compute new_transfer_value and let it move on in future rounds
                active_set_for_this = {cid}
                for ballot in election.ballots:
                    current_cid = ballot.highest_ranked_active(active_set_for_this)
                    if current_cid != cid:
                        continue
                    new_tv = _truncate_4(
                        ballot.current_transfer_value * surplus_fraction
                    )
                    ballot.current_transfer_value = new_tv

            # Next round: recount with updated transfer values
            election.round_number += 1
            totals = _count_votes_single_round(
                election, status, use_transfer_values=True
            )
            _apply_threshold_to_elected(totals, status, election.threshold)
            rounds.append(
                {
                    "round": election.round_number,
                    "vote_totals": totals.copy(),
                    "status": status.copy(),
                    "threshold": election.threshold,
                    "action": {"type": "elect_and_transfer", "details": action_detail},
                }
            )
            continue

        # No one elected: eliminate lowest active candidate (Sec. 322(b)(3))
        if active:
            min_votes = min(totals.get(cid, 0.0) for cid in active)
            lowest = [cid for cid in active if totals.get(cid, 0.0) == min_votes]
            eliminated = (
                _tie_break(lowest, election.tie_break_order)
                if len(lowest) > 1
                else lowest[0]
            )
            status[eliminated] = "eliminated"
            action = {"type": "eliminate", "candidate": eliminated}
        else:
            action = None

        # Recount for next round
        election.round_number += 1
        totals = _count_votes_single_round(election, status, use_transfer_values=True)
        _apply_threshold_to_elected(totals, status, election.threshold)
        rounds.append(
            {
                "round": election.round_number,
                "vote_totals": totals.copy(),
                "status": status.copy(),
                "threshold": election.threshold,
                "action": action,
            }
        )

    final_status = status.copy()
    winners = [cid for cid, s in final_status.items() if s == "elected"]

    return {
        "winners": winners,
        "rounds": rounds,
        "final_candidate_status": final_status,
    }


# -----------------------------
# Public entry point
# -----------------------------
def run_election(election: Election) -> Dict:
    if election.mode == "single_seat_rcv":
        return run_single_seat_rcv(election)
    else:
        return run_multi_seat_stv(election)


# -----------------------------
# JSON loader
# -----------------------------
def load_election_from_json(path: str) -> Election:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # -----------------------------
    # Handle metadata wrapper
    # -----------------------------
    metadata = data.get("metadata", {})

    # Allow fallback to top-level for compatibility
    election_id = metadata.get("election_id", data.get("election_id"))
    seat_count = metadata.get("seat_count", data.get("seat_count"))
    mode = metadata.get("mode", data.get("mode"))
    tie_break_order = metadata.get("tie_break_order", data.get("tie_break_order"))
    max_ranks_allowed = metadata.get("max_ranks_allowed", data.get("max_ranks_allowed"))

    required_fields = {
        "election_id": election_id,
        "seat_count": seat_count,
        "mode": mode,
        "tie_break_order": tie_break_order,
    }

    for k, v in required_fields.items():
        if v is None:
            raise ValueError(f"Missing required field '{k}' (top-level or metadata)")

    # -----------------------------
    # Candidates (ignore extra fields)
    # -----------------------------
    candidates = [
        Candidate(
            candidate_id=c["candidate_id"],
            withdrawn=c.get("withdrawn", False),
        )
        for c in data.get("candidates", [])
    ]

    # -----------------------------
    # Ballots (ignore extra fields)
    # -----------------------------
    ballots: List[Ballot] = []
    for b in data.get("ballots", []):
        rankings = [
            Ranking(
                rank=r["rank"],
                candidate_ids=r["candidate_ids"],
            )
            for r in b.get("rankings", [])
        ]

        ballots.append(
            Ballot(
                ballot_id=b["ballot_id"],
                rankings=rankings,
            )
        )

    # -----------------------------
    # Build election
    # -----------------------------
    election = Election(
        election_id=election_id,
        seat_count=seat_count,
        mode=mode,
        candidates=candidates,
        ballots=ballots,
        tie_break_order=tie_break_order,
        max_ranks_allowed=max_ranks_allowed,
    )

    return election


# -----------------------------
# Interactive CLI
# -----------------------------
if __name__ == "__main__":
    print("Fair Representation Act Counting Engine")
    print("Interactive mode.")
    print("Enter path to election JSON (matching the Simulation Layer Design Specification).")
    path = input("Path: ").strip()

    try:
        election = load_election_from_json(path)
    except Exception as e:
        print("Error: Invalid election JSON.")
        print("Expected format (simplified):")
        print('  {')
        print('    "election_id": "string",')
        print('    "seat_count": integer,')
        print('    "mode": "single_seat_rcv" | "multi_seat_stv",')
        print('    "candidates": [ { "candidate_id": "C1", "withdrawn": false }, ... ],')
        print('    "ballots": [')
        print('      {')
        print('        "ballot_id": "B1",')
        print('        "rankings": [ { "rank": 1, "candidate_ids": ["C1"] }, ... ]')
        print('      },')
        print('      ...')
        print('    ],')
        print('    "tie_break_order": ["C1", "C2", ...]')
        print('  }')
        print(f"Details: {e}")
    else:
        result = run_election(election)
        print("\nWinners:", result["winners"])
        print("\nFinal candidate status:")
        for cid, status in result["final_candidate_status"].items():
            print(f"  {cid}: {status}")
        print("\nRounds:")
        for rnd in result["rounds"]:
            print(f"  Round {rnd['round']}:")
            print(f"    Vote totals: {rnd['vote_totals']}")
            if "threshold" in rnd and rnd["threshold"] is not None:
                print(f"    Threshold: {rnd['threshold']}")
            if rnd.get("action"):
                print(f"    Action: {rnd['action']}")
