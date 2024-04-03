"""
Problem URL: https://leetcode.com/problems/string-to-integer-atoi

Method 2:
    Deterministic Finite Automaton.

Method 1 (Brute Force):
    Approach:
        While DFA would be the right way to do this, it did not immediately occur to me.

        So, I worked this out the hard way.

        Passes most sane test cases.
        Does not pass insane test cases such as: "-91283472332" --> "-2147483648".

        WOF:
        1. A split satisfying cond1_num or cond2_num may be ending in non-digits.
            Eg. "123abc", "+123abc", "-123abc".
    TC:
        Linear pass for s.split() => O(N).
        Linear pass with O(1) ops in the main loop => O(N).
        Total: O(N).    
    Space:
        split_s => O(N).
"""
class Solution:

    def myAtoi(self, s):
        """
        Based on DFA suggestion in the Discussion section.
        """
        val, state, pos, sign = 0, 0, 0, 1

        if len(s)==0:
            return 0

        print(pos, type(pos), s, type(s))

        while pos<len(s):

            c = str[pos] # type: ignore
            if state==0:
                if c==" ":
                    state=0
                elif c=="+" or c=="-":
                    state=1
                    sign = 1 if c=="+" else -1
                elif c.isdigit(): # type: ignore
                    state=2
                    val=val*10+int(c) # type: ignore
            elif state==1:
                if c.isdigit(): # type: ignore
                    state=2
                    val=val*10+int(c) # type: ignore
                else:
                    return 0
            elif state==2:
                if c.isdigit(): # type: ignore
                    state=2
                    val=val*10+int(c) # type: ignore
                else:
                    break
            else:
                return 0
            
            pos+=1

        val *= sign

        val = min(val, 2**31-1)
        val = max(-(2**31), val)
    
    def extract_num(self, str_):

        num = 0
        for c in str_:
            # str_ may be ending in non-digits. Eg. "123abc".
            if not c.isdigit():
                break
            num = 10*num+int(c)

        return num

    def myAtoi_nonDFA(self, s):

        if len(s)==0:
            return 0

        sign = 1

        # Exactly one element in this list would be the actual number.
        split_s = s.split(" ")

        for element in split_s:
            
            # If empty.
            if len(element)==0:
                continue

            cond1_number = element[0].isdigit()
            cond2_number = (element[0] in "+-") and (element[1].isdigit)
            
            # Unsigned number.
            if cond1_number:
                return self.extract_num(element)
            # Signed number.
            elif cond2_number:
                if element[0]=="+":
                    return self.extract_num(element[1:])
                else:
                    return -1*self.extract_num(element[1:])
            else:
                # Words before numbers. Eg. "words and 987" => 0 (not 987).
                return 0
                # If we want to permit words before numbers, i.e.,
                # "words and 987" => 987, we can add a "continue" instead of "return 0".

        return 0

