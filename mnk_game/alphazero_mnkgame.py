from .mcts_mnkgame import MonteCarloTreeSearchMnkGame
from .alphazero_net import AlphaZeroNet


def bitboard_to_tensor(bb):
    pass



class AlphaZeroMnkGame(MonteCarloTreeSearchMnkGame):
    def __init__(self, max_thinking_time, max_rollout, processes, policy,
                 exploration_const, num_simulations, m, n, k):
        super().__init__(max_thinking_time, max_rollout, processes, policy,
                         exploration_const, num_simulations)
        self.m, self.n, self.k = m, n, k
        self.net = AlphaZeroNet(m, n, k)

    def update_net(self, net):
        pass

    @staticmethod
    def score(node, c):
        # puct
        return node.r + c * node.prior * math.sqrt(node.parent.n) / (1 + node.n)


    def solve(self, board: MnkBoard, turn: int, moves) -> Tuple[int, int]:
        start = time.time()
        if len(moves) < 2 or self.root is None:
            print("Initializing new tree...")
            self.root = MnkState(board, turn, self.policy, None, None)
            policy, _ = self.net(bitboard_to_tensor(self.root.board))
            self.root.prior =
        else:
            if not self.update_tree(moves[-2:]):
                self.root = MnkState(board, turn, self.policy, None, None)
                policy, _ = self.net(bitboard_to_tensor(self.root.board))
                self.root.prior =

        while time.time()-start < self.max_thinking_time and \
                self.total_rollout < self.max_rollout:
            self.loop()
        return self.get_results()

    def selection(self):
        node = self.root
        while not node.is_leaf():
            selected_node = max(
                node.children.values(), key=lambda child: self.score(child, self.c))
            node = selected_node
        return node

    def choosing_policy(self, states):
        return random.choice(states)

    def expansion(self, node):
        possible_pos = node.board.get_possible_pos()
        if node.board.check_endgame() == 0 and possible_pos:
            policy, value = self.net(bitboard_to_tensor(node.board))
            for state in node.next_states():
                i, j = state.last_move
                state.prior = policy[i*self.n + j]
                node.children[state.last_move] = state
            node.r += value
            node.n += 1

    def backpropagation(self, node):
        value = node.r / node.n
        while node is not None:
            node.n += 1
            node.r += value
            node = node.parent

    def loop(self):
        node = self.selection()
        self.expansion(node)
        self.backpropagation(node)

