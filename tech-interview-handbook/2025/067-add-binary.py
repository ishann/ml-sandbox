"""
go from right to left.
total at each step is (carry+a[idx_a]+b[idx_b]) % 2
append str(total) to beginning of list (this might be too slow,
but we can just append to end of list and reverse once in the end).
carry is updated as total//2
make sure that idx_a and idx_b are valid.
return as string.
"""
class Solution:
    def addBinary(self, a: str, b: str) -> str:

        idx_a, idx_b = len(a)-1, len(b)-1
        res = []
        carry = 0

        while idx_a>=0 or idx_b>=0 or carry:

            total = carry

            if idx_a>=0:
                total += int(a[idx_a])
                idx_a -= 1

            if idx_b>=0:
                total += int(b[idx_b])
                idx_b -= 1

            res = [str(total%2)] + res
            carry = total//2

        return "".join(res)


