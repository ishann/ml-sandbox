"""
Problem URL: https://leetcode.com/problems/two-sum/

Approach:
    1. Iterate through the array and create hash map for [idx, target-num].
    2. O(1) look-up if target-num in hash map.
Time Complexity:
    One-pass through the array with constant computations.
    Thus, O(n).
Space Complexity:
    Hash-map stores upto one element corresponding to each element in nums.
    Thus, O(n).
"""
tests = [{"inp": {"nums":[2,7,11,15], "target":9},
         "out": [0,1]},
        {"inp": {"nums":[3,2,4], "target":6},
         "out": [1,2]},
        {"inp": {"nums":[3,3], "target":6},
         "out": [0,1]}]

def two_sum(nums, target):
    hash_map = dict()

    for idx, num in enumerate(nums):
        diff = target-num
        if diff in hash_map:
            return [idx, hash_map[diff]]
        else:
            hash_map[num] = idx

def execute(test):

    nums, target = test["inp"]["nums"], test["inp"]["target"]
    expected_out = sorted(test["out"])
    function_out = sorted(two_sum(nums, target)) # type: ignore

    if expected_out==function_out:
        print("Correct.".format(function_out))
        return 1
    else: 
        print("Incorrect.\nExpected {}. Got {}.".format(expected_out, function_out))
        return 0

num_correct = 0
for idx, test in enumerate(tests):
    print("Running test [{}/{}].".format(idx+1, len(tests)))
    num_correct+=execute(test)

if num_correct==len(tests):
    print("\nAll tests pass.")
else:
    print("\nSome tests fail.")
