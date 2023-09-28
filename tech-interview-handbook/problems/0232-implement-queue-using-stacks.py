"""
Problem URL: https://leetcode.com/problems/implement-queue-using-stacks

Ops to focus on = push and pop.
"""
class MyQueue:

    def __init__(self):

        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        
        self.stack1.append(x)

    def pop(self):

        #print("Before 1st loop:")
        #print(f"stack1:{self.stack1}, stack2:{self.stack2}")
        while self.stack1:
            el = self.stack1.pop()
            self.stack2.append(el)
        #print("After 1st loop:")
        #print(f"stack1:{self.stack1}, stack2:{self.stack2}")

        el_to_pop = self.stack2.pop()
        #print(f"el_to_pop: {el_to_pop}")
        while self.stack2:
            el = self.stack2.pop()
            self.stack1.append(el)
        #print("After 2nd loop:")
        #print(f"stack1:{self.stack1}, stack2:{self.stack2}")
        
        return el_to_pop
            
    def peek(self):
        
        while self.stack1:
            el = self.stack1.pop()
            self.stack2.append(el)

        el_to_peek = self.stack2.pop()
        self.stack1.append(el_to_peek)

        while self.stack2:
            el = self.stack2.pop()
            self.stack1.append(el)

        return el_to_peek

    def empty(self):

        return True if not self.stack1 else False
        


# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()