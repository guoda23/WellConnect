from math import log2
from scipy.stats import entropy

# All integer partitions of 10 into ≤ 10 parts (order doesn’t matter)
partitions_of_10 = [
    [10],
    [9,1],
    [8,2],
    [8,1,1],
    [7,3],
    [7,2,1],
    [7,1,1,1],
    [6,4],
    [6,3,1],
    [6,2,2],
    [6,2,1,1],
    [6,1,1,1,1],
    [5,5],
    [5,4,1],
    [5,3,2],
    [5,3,1,1],
    [5,2,2,1],
    [5,2,1,1,1],
    [5,1,1,1,1,1],
    [4,4,2],
    [4,4,1,1],
    [4,3,3],
    [4,3,2,1],
    [4,3,1,1,1],
    [4,2,2,2],
    [4,2,2,1,1],
    [4,2,1,1,1,1],
    [4,1,1,1,1,1,1],
    [3,3,3,1],
    [3,3,2,2],
    [3,3,2,1,1],
    [3,3,1,1,1,1],
    [3,2,2,2,1],
    [3,2,2,1,1,1],
    [3,2,1,1,1,1,1],
    [3,1,1,1,1,1,1,1],
    [2,2,2,2,2],
    [2,2,2,2,1,1],
    [2,2,2,1,1,1,1],
    [2,2,1,1,1,1,1,1],
    [2,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1],
]

def entropy_from_partition(part):
    """Compute Shannon entropy (base 2) from integer counts using scipy.stats.entropy."""
    total = sum(part)
    probs = [c/total for c in part if c > 0]
    return entropy(probs, base=2)

# Compute distinct entropies
partition_entropies = [(part, round(entropy_from_partition(part), 4)) for part in partitions_of_10]
unique_entropies = sorted({e for _, e in partition_entropies})

print("All unique entropy values (rounded to 4 decimals):")
print(len(unique_entropies))
print("\nNumber of unique entropy values:", len(unique_entropies))

print("\nPartitions with entropy < 1.0:")
for part, e in partition_entropies:
    if e < 1.0:
        print(f"{part} → {e}")
