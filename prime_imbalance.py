import math

def is_prime(n):
    """Check if a number is prime using trial division."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def next_prime(n):
    """Find the smallest prime greater than n."""
    candidate = n + 1
    while not is_prime(candidate):
        candidate += 1
    return candidate

def find_prime_in_interval(start, end):
    """Find a prime in the interval [start, end]. Returns None if none exists."""
    for candidate in range(max(2, int(start)), int(end) + 1):
        if is_prime(candidate):
            return candidate
    return None

def find_prime_imbalance_pair(t, epsilon=0.01):
    """
    Find prime pair (p, q) such that |(p-q)/(p+q) - t| < epsilon.
    
    Based on the algorithm from "On the Density of Prime Imbalances in the Unit Interval".
    
    Args:
        t (float): Target normalized imbalance in (0, 1)
        epsilon (float): Tolerance for approximation
    
    Returns:
        tuple: (p, q, actual_imbalance, error) where p > q are primes
    
    Raises:
        ValueError: If t is not in (0, 1) or epsilon <= 0
    """
    if not (0 < t < 1):
        raise ValueError("t must be in the interval (0, 1)")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    
    # Step 1: Transform t to r using the bijection from Lemma 1
    r = (1 + t) / (1 - t)
    
    # Step 2: Set delta based on epsilon and r
    delta = (epsilon * (r + 1)**2) / 16
    
    # Step 3: Choose M to satisfy the requirements
    M = max(1600 / (epsilon * (r + 1)**2), 25)
    M = int(math.ceil(M))
    
    print(f"Algorithm parameters:")
    print(f"  Target t = {t}")
    print(f"  Tolerance epsilon = {epsilon}")
    print(f"  Transformed r = {r:.6f}")
    print(f"  Delta delta = {delta:.6f}")
    print(f"  Parameter M = {M}")
    
    # Step 4: Find prime q in the interval (M, 2M) using Bertrand's postulate
    q = next_prime(M)
    if q >= 2 * M:
        # This shouldn't happen for reasonable inputs due to Bertrand's postulate
        raise RuntimeError(f"Could not find prime in interval ({M}, {2*M})")
    
    print(f"  Found q = {q} in interval ({M}, {2*M})")
    
    # Step 5: Construct target interval for p
    target_center = r * q
    interval_half_width = delta * q
    interval_start = target_center - interval_half_width
    interval_end = target_center + interval_half_width
    
    interval_length = 2 * interval_half_width
    print(f"  Target interval for p: [{interval_start:.1f}, {interval_end:.1f}]")
    print(f"  Interval length: {interval_length:.1f}")
    
    # Verify that interval length >= 200 and start >= 25 (requirements for Lemma 3)
    if interval_length < 200:
        print(f"  Warning: Interval length {interval_length:.1f} < 200")
    if interval_start < 25:
        print(f"  Warning: Interval start {interval_start:.1f} < 25")
    
    # Step 6: Find prime p in the target interval
    p = find_prime_in_interval(interval_start, interval_end)
    
    if p is None:
        # Try expanding the search slightly
        expanded_start = interval_start - 100
        expanded_end = interval_end + 100
        p = find_prime_in_interval(expanded_start, expanded_end)
        if p is None:
            raise RuntimeError(f"Could not find prime in interval [{interval_start:.1f}, {interval_end:.1f}]")
        print(f"  Found p = {p} in expanded search region")
    else:
        print(f"  Found p = {p} in target interval")
    
    # Step 7: Calculate the actual normalized imbalance and error
    actual_imbalance = (p - q) / (p + q)
    error = abs(actual_imbalance - t)
    
    print(f"\nResults:")
    print(f"  Prime pair: p = {p}, q = {q}")
    print(f"  Actual imbalance: (p-q)/(p+q) = {actual_imbalance:.8f}")
    print(f"  Target imbalance: t = {t:.8f}")
    print(f"  Error: |actual - target| = {error:.8f}")
    print(f"  Success: error < epsilon? {error < epsilon}")
    
    return p, q, actual_imbalance, error

def verify_construction_bounds(p, q, t, epsilon):
    """
    Verify that the found prime pair satisfies the theoretical bounds from the paper.
    """
    r = (1 + t) / (1 - t)
    delta = (epsilon * (r + 1)**2) / 16
    
    # Check if |p/q - r| < delta
    ratio_error = abs(p/q - r)
    print(f"\nTheoretical verification:")
    print(f"  |p/q - r| = {ratio_error:.8f}")
    print(f"  delta = {delta:.8f}")
    print(f"  Ratio bound satisfied: |p/q - r| < delta? {ratio_error < delta}")
    
    # Check the theoretical error bound from Lemma 2
    theoretical_bound = (2 * delta) / ((r + 1 - delta) * (r + 1))
    actual_error = abs((p - q)/(p + q) - t)
    print(f"  Theoretical error bound: {theoretical_bound:.8f}")
    print(f"  Actual error: {actual_error:.8f}")
    print(f"  Bound satisfied: actual < theoretical? {actual_error < theoretical_bound}")

# Example usage and test cases
if __name__ == "__main__":
    # Test case 1: t = 1/3 (from the paper)
    print("=" * 60)
    print("Test Case 1: t = 1/3, epsilon = 0.01")
    print("=" * 60)
    p1, q1, imbalance1, error1 = find_prime_imbalance_pair(1/3, 0.01)
    verify_construction_bounds(p1, q1, 1/3, 0.01)
    
    print("\n" + "=" * 60)
    print("Test Case 2: t = 1/3, epsilon = 0.1 (larger tolerance)")
    print("=" * 60)
    p2, q2, imbalance2, error2 = find_prime_imbalance_pair(1/3, 0.1)
    verify_construction_bounds(p2, q2, 1/3, 0.1)
    
    print("\n" + "=" * 60)
    print("Test Case 3: t = 0.8, epsilon = 0.05 (high imbalance)")
    print("=" * 60)
    p3, q3, imbalance3, error3 = find_prime_imbalance_pair(0.8, 0.05)
    verify_construction_bounds(p3, q3, 0.8, 0.05)
    
    print("\n" + "=" * 60)
    print("Test Case 4: t = 0.1, epsilon = 0.05 (low imbalance)")
    print("=" * 60)
    p4, q4, imbalance4, error4 = find_prime_imbalance_pair(0.1, 0.05)
    verify_construction_bounds(p4, q4, 0.1, 0.05)