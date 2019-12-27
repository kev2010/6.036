from util import *

class SM:
    start_state = None  # default start state

    def transition_fn(self, s, x):
        '''s:       the current state
           x:       the given input
           returns: the next state'''
        raise NotImplementedError

    def output_fn(self, s):
        '''s:       the current state
           returns: the corresponding output'''
        raise NotImplementedError

    def transduce(self, input_seq):
        '''input_seq: the given list of inputs
           returns:   list of outputs given the inputs'''
        # Your code here
        s = 0
        result = []
        for i in input_seq:
            s = self.transition_fn(s,i)
            result.append(self.output_fn(s))
        return result


class Accumulator(SM):
    start_state = 0

    def transition_fn(self, s, x):
        return s + x

    def output_fn(self, s):
        return s


class Binary_Addition(SM): # Your code here
    start_state = (0,0) # Change

    def transition_fn(self, s, x):
        t = x[0]+x[1]+s[1]
        return (t % 2, t//2)

    def output_fn(self, s):
        return s[0]


class Reverser(SM): # Your code here
    start_state = ([], 0)

    def transition_fn(self, s, x):
        if x == 'end':
            return (s[0], 1)
        elif s[1] == 0:
            return (s[0]+[x], s[1])
        else:
            return (s[0][:-1], s[1])

    def output_fn(self, s):
        if s[1] == 1:
            return s[0][-1] if len(s[0])>0 else None


class RNN(SM): # Your code here
    def __init__(self, Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2):
        self.Wsx = Wsx
        self.Wss = Wss
        self.Wo = Wo
        self.Wss_0 = Wss_0
        self.Wo_0 = Wo_0
        self.f1 = f1
        self.f2 = f2
        self.start_state = np.zeros((Wss_0.shape[0], Wss_0.shape[1]))

    def transition_fn(self, s, x):
        return self.f1(self.Wss @ s + self.Wsx @ x + self.Wss_0)

    def output_fn(self, s):
        return self.f2(self.Wo @ s + self.Wo_0)

    # Back propgation through time
    # xs is matrix of inputs: l by k
    # dLdz2 is matrix of output errors:  1 by k
    # states is matrix of state values: m by k
    def bptt(self, xs, dLtdz2, states):
        dWsx = np.zeros_like(self.Wsx)
        dWss = np.zeros_like(self.Wss)
        dWo = np.zeros_like(self.Wo)
        dWss0 = np.zeros_like(self.Wss0)
        dWo0 = np.zeros_like(self.Wo0)
        # Derivative of future loss (from t+1 forward) wrt state at time t
        # initially 0;  will pass "back" through iterations
        dFtdst = np.zeros((self.hidden_dim, 1))
        k = xs.shape[1]
        # Technically we are considering time steps 1..k, but we need
        # to index into our xs and states with indices 0..k-1
        for t in range(k - 1, -1, -1):
            # Get relevant quantities
            xt = xs[:, t:t + 1]
            st = states[:, t:t + 1]
            stm1 = states[:, t - 1:t] if t - 1 >= 0 else self.init_state
            dLtdz2t = dLtdz2[:, t:t + 1]
            # Compute gradients step by step
            # ==> Use self.df1(st) to get dfdz1;
            # ==> Use self.Wo, self.Wss, etc. for weight matrices
            # derivative of loss at time t wrt state at time t
            dLtdst = np.transpose(self.Wo) @ dLtdz2t  # Your code
            # derivatives of loss from t forward
            dFtm1dst = dLtdst + dFtdst  # Your code
            dFtm1dz1t = self.df1(st) * dFtm1dst  # Your code
            dFtm1dstm1 = np.transpose(self.Wss) @ dFtm1dz1t  # Your code
            # gradients wrt weights
            dLtdWo = dLtdz2t @ np.transpose(st)  # Your code
            dLtdWo0 = dLtdz2t  # Your code
            dFtm1dWss = dFtm1dz1t @ np.transpose(stm1)  # Your code
            dFtm1dWss0 = dFtm1dz1t  # Your code
            dFtm1dWsx = dFtm1dz1t @ np.transpose(xt)  # Your code
            # Accumulate updates to weights
            dWsx += dFtm1dWsx
            dWss += dFtm1dWss
            dWss0 += dFtm1dWss0
            dWo += dLtdWo
            dWo0 += dLtdWo0
            # pass delta "back" to next iteration
            dFtdst = dFtm1dstm1
        return dWsx, dWss, dWo, dWss0, dWo0


# Enter the parameter matrices and vectors for an instance of the RNN class such that
# the output is 1 if the cumulative sum of the inputs is positive, -1 if the cumulative sum is negative
# and 0 if otherwise. Make sure that you scale the outputs so that the output activation values are very close
# to 1, 0 and -1. Note that both the inputs and outputs are 1x1.

Wsx =    np.array([[1]])
Wss =    np.array([[1]])
Wo =     np.array([[1]])
Wss_0 =  np.array([[0]])
Wo_0 =   np.array([[0]])
f1 =     lambda x: x
f2 =     lambda x: np.sign(x)
acc_sign = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)

# Enter the parameter matrices and vectors for an instance of the RNN class
# such that it implements the following autoregressive model:
# y_t = 1y_{t-1} - 2y_{t-2} + 3y_{t-3} where x_t = y_{t-1}

Wsx =    np.array([[1], [0], [0]])
Wss =    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
Wo =     np.array([[1, -2, 3]])
Wss_0 =  np.zeros((3, 1))
Wo_0 =   np.array([[0]])
f1 =     lambda x : x
f2 =     lambda x : x
auto = RNN(Wsx, Wss, Wo, Wss_0, Wo_0, f1, f2)


