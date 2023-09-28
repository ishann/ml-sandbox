"""
Problem URL: https://leetcode.com/problems/backspace-string-compare

Approach:
    Get the cleaned strings.
    Compare them.
TC:
    O(M+N)
Space:
    O(M+N)

NOTE: There may be a possible optimizations if we use two pointers and
      we compare the strings from back to front:
       1. # means skip a character.
       2. if encounter mismatched charaters, return False.
       3. if reach end of string, return True.
"""
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:

        def clean_str(chars):

            result = []
            for char in chars:
                if char=="#":
                    if result:
                        result.pop()
                    else:
                        continue
                else:
                    result.append(char)
            
            return "".join(result)

        clean_s = clean_str(s)
        clean_t = clean_str(t)

        return True if clean_s==clean_t else False

        