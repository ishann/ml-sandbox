"""
Problem URL: https://leetcode.com/problems/func-name

Approach:
    1. ...
    2. ...
    3. ...

Time Complexity:

Space Complexity:

"""
tests = [{"inp": None,
          "out": None}]

def func_name(nums, target):
    hash_map = dict()

    for idx, num in enumerate(nums):
        diff = target-num
        if diff in hash_map:
            return [idx, hash_map[diff]]
        else:
            hash_map[num] = idx

def execute(test):

    """Parse the test object."""
    inputs = ...
    expected_out = ...

    """Compute output."""
    function_out = func_name(inputs)

    """Check if output is correct and print/ return appropriately."""
    if expected_out==function_out:
        print("Correct.")
        return 1
    else:
        print("Incorrect. \nExpected {}. Got {}.".format(expected_out,
                                                         function_out))
        return 0

num_correct = 0
for idx, test in enumerate(tests):
    print("Running test [{}/{}].".format(idx+1, len(tests)))
    num_correct+=execute(test)

if num_correct==len(tests):
    print("\nAll tests pass.")
else:
    print("\nSome tests fail.")

