"""
Approach:
    Encode: [str(num_chars) + num_chars delimiter + string] x N.
    Decode: Iterate:
                Find length of each word.
                Extract that word.
                Move forward.

Time Complexity:
    Encoding takes a linear parse => O(N).
    Decodding takes a linear parse => O(N).
    => O(N).

Space:
    Ignoring output space, we only use a few extra string and int variables.
    => O(1).
"""
class Solution:
    def encode(self, strs):

        #num_words = len(strs)
        #num_words_delimiter = "!"
        num_chars_delimiter = "@"

        #encoded = str(num_words) + num_words_delimiter
        encoded = ""

        for str_ in strs:
            encoded += str(len(str_)) + num_chars_delimiter + str_

        return encoded

    def decode(self, str):
        # "4@lint4@code4@love3@you"

        result = []
        i = 0

        # Iterates over each word.
        while i<len(str):
            j=i
            # Iterate and [i:j] will give length of the word.
            while str[j] != "@":
                j+=1
            len_ = int(str[i:j])
            result.append(str[j+1 : j+1+len_])
            i = j+1+len_

        return result


