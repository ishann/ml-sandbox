"""
Approach:
    This is a well-known problem with a straightforward implementation of stacks.
    If opening bracket encountered, append to stack.

TC:
    Linear parse.
    => O(N).

Space:
    Stack can take all the parentheses.
    => O(N).
"""
class Solution:
    def isValid(self, s: str) -> bool:

        hashmap = {")":"(",
                  "]":"[",
                  "}":"{", }

        stack = []

        for c in s:
            if c in hashmap:
                if stack and stack[-1]==hashmap[c]:
                    stack.pop()
                else:
                    return False
            else:
                stack.append(c)

        return True if len(stack)==0 else False


