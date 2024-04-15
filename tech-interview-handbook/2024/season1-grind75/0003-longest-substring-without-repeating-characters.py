"""
Approach:
    Two pointers: l,r.
    Keep sliding r and checking if current substring is a set.
    Every time a new r is added, we may have to slide l and remove s[l] until s[r] not in set.
    Answer is max of encountered len(s_set)s as we create them.

Time Complexity:
    Linear parse with two pointers => O(N).

Space:
    Two extra integer pointers => O(1).

"""
class Solution:

    def lengthOfLongestSubstring(self, s: str) -> int:

        l = 0
        result = 0

        s_set = set()

        for r in range(len(s)):

            while s[r] in s_set:
                s_set.remove(s[l])
                l+=1

            s_set.add(s[r])

            result = max(result, len(s_set))

        return result


