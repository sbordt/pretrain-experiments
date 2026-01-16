import random
from typing import Optional, Tuple, Iterable
import hashlib

Interval = Tuple[int, int]

def overlaps_closed(a: Interval, b: Interval) -> bool:
    # [a0,a1] and [b0,b1] overlap if they intersect at all
    return not (a[1] < b[0] or b[1] < a[0])

class _Node:
    __slots__ = ("lo", "hi", "prio", "left", "right", "max_end")
    def __init__(self, lo: int, hi: int):
        self.lo = lo
        self.hi = hi
        self.prio = random.random()
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.max_end = hi

    def recalc(self):
        me = self.hi
        if self.left and self.left.max_end > me: me = self.left.max_end
        if self.right and self.right.max_end > me: me = self.right.max_end
        self.max_end = me

class IntervalSet:
    """
    Disjoint closed intervals with fast overlap test and insert.
      - overlaps((lo,hi)) -> bool     O(log n) expected
      - add((lo,hi))                  O(log n) expected (must not overlap)
    """

    def __init__(self, it: Optional[Iterable[Interval]] = None):
        self._root: Optional[_Node] = None
        if it:
            for lo, hi in it:
                self.add((lo, hi))

    # ----- rotations -----
    def _rot_r(self, y: _Node) -> _Node:
        x = y.left; y.left = x.right; x.right = y
        y.recalc(); x.recalc(); return x

    def _rot_l(self, x: _Node) -> _Node:
        y = x.right; x.right = y.left; y.left = x
        x.recalc(); y.recalc(); return y

    # ----- insert by lo (assumes no overlap) -----
    def _insert(self, t: Optional[_Node], lo: int, hi: int) -> _Node:
        if not t: return _Node(lo, hi)
        if lo < t.lo:
            t.left = self._insert(t.left, lo, hi)
            if t.left.prio < t.prio: t = self._rot_r(t)
        else:
            t.right = self._insert(t.right, lo, hi)
            if t.right.prio < t.prio: t = self._rot_l(t)
        t.recalc()
        return t

    def add(self, iv: Interval) -> None:
        lo, hi = iv
        if hi < lo:
            raise ValueError("Interval must satisfy lo <= hi")
        if self.overlaps(iv):
            raise ValueError("Interval overlaps existing interval")
        self._root = self._insert(self._root, lo, hi)

    def overlaps(self, iv: Interval) -> bool:
        lo, hi = iv
        t = self._root
        while t:
            # If left subtree might contain an overlap, go left
            if t.left and t.left.max_end >= lo:
                t = t.left
                continue
            # Check current node
            if overlaps_closed((t.lo, t.hi), (lo, hi)):
                return True
            # Nothing left can overlap; go right
            t = t.right
        return False

    # Optional: return one overlapping interval (or None)
    def find_overlap(self, iv: Interval) -> Optional[Interval]:
        lo, hi = iv
        t = self._root
        while t:
            if t.left and t.left.max_end >= lo:
                t = t.left
                continue
            if overlaps_closed((t.lo, t.hi), (lo, hi)):
                return (t.lo, t.hi)
            t = t.right
        return None

    def __len__(self) -> int:
        def cnt(n: Optional[_Node]) -> int:
            return 0 if n is None else 1 + cnt(n.left) + cnt(n.right)
        return cnt(self._root)

    def to_list(self) -> list[Interval]:
        """Return all intervals as a sorted list of (lo, hi) tuples."""
        out: list[Interval] = []

        def _inorder(n: Optional[_Node]):
            if not n:
                return
            _inorder(n.left)
            out.append((n.lo, n.hi))
            _inorder(n.right)

        _inorder(self._root)
        return out
    
    def __hash__(self) -> int:
        """
        Compute a stable hash of the current intervals.
        The hash is based on the sorted list of intervals, so it's independent
        of insertion order and tree structure.
        """
        # Get sorted intervals
        intervals = self.to_list()
        
        # Convert to a string representation for hashing
        # Format: "[(lo1,hi1),(lo2,hi2),...]"
        intervals_str = str(intervals)
        
        # Use SHA-256 for a stable hash, then convert to int
        hash_bytes = hashlib.sha256(intervals_str.encode('utf-8')).digest()
        
        # Convert first 8 bytes to int (Python's hash is typically 64-bit)
        return int.from_bytes(hash_bytes[:8], 'big', signed=True)
    
    def hash_fast(self) -> int:
        """
        Alternative faster hash using Python's built-in hash on the tuple of intervals.
        Still stable within a single Python process, but may vary across processes.
        """
        return hash(tuple(self.to_list()))
    
    def __eq__(self, other) -> bool:
        """Check if two IntervalSets contain the same intervals."""
        if not isinstance(other, IntervalSet):
            return False
        return self.to_list() == other.to_list()
    
    def __repr__(self) -> str:
        return f"IntervalSet({self.to_list()})"