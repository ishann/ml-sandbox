"""
Problem URL: https://leetcode.com/problems/valid-parentheses/

Approach:
    1. If opening bracket, push on stack.
    2. If closing bracket, pop from stack.
    3. Compare.

Time Complexity:

Space Complexity:

"""
tests = [{"inp": {"s": "()"},
          "out": True},
         {"inp": {"s": "()[]{}"},
          "out": True},
         {"inp": {"s": "(]"},
          "out": False}]


def is_valid(s):

    stack = []
    opening = ["(", "{", "["]
    opener_for_closer = {")": "(", "}": "{", "]": "["}

    for c in s:
        if c in opening:
            stack.append(c)
        else:
            if len(stack)==0:
                return False
                
            if stack.pop()!=opener_for_closer[c]:
                return False

    return True if len(stack)==0 else False

def execute(test):

    """Parse the test object."""
    s = test["inp"]["s"]
    expected_out = test["out"]

    """Compute output."""
    function_out = is_valid(s)

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

