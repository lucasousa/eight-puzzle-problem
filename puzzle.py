from collections import deque

import matplotlib.pyplot as plt
import networkx as nx


class EightPuzzleProblem:
    def __init__(self, puzzle, parent=None, move=None):
        self.puzzle = puzzle
        self.parent = parent
        self.move = move

        for i in range(3):
            for j in range(3):
                if puzzle[i][j] == 0:
                    self.blank_row = i
                    self.blank_col = j
                    return

    def generate_moves(self):
        moves = []
        directions = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
        ]  # calculate: down, up, right and left
        for d in directions:
            new_row, new_col = self.blank_row + d[0], self.blank_col + d[1]
            if 0 <= new_row < 3 and 0 <= new_col < 3:
                move = (new_row, new_col)
                moves.append(move)
        return moves

    def move_square(self, move):
        new_puzzle = [row[:] for row in self.puzzle]
        new_blank_row, new_blank_col = move
        (
            new_puzzle[self.blank_row][self.blank_col],
            new_puzzle[new_blank_row][new_blank_col],
        ) = (
            new_puzzle[new_blank_row][new_blank_col],
            new_puzzle[self.blank_row][self.blank_col],
        )
        return EightPuzzleProblem(new_puzzle, self, move)

    def is_goal_state(self, goal_state):
        return self.puzzle == goal_state

    def get_path(self):
        path = []
        current = self
        while current:
            path.append(current.move)
            current = current.parent
        return path[::-1]


def bfs(start_state, goal_state):
    queue = deque([start_state])
    visited = set()
    search_tree = nx.DiGraph()

    while queue:
        current_state = queue.popleft()

        if current_state.is_goal_state(goal_state):
            return current_state.get_path(), search_tree

        visited.add(tuple(map(tuple, current_state.puzzle)))

        for move in current_state.generate_moves():
            new_state = current_state.move_square(move)
            if tuple(map(tuple, new_state.puzzle)) not in visited:
                queue.append(new_state)
                visited.add(tuple(map(tuple, new_state.puzzle)))
                search_tree.add_edge(
                    tuple(map(tuple, current_state.puzzle)),
                    tuple(map(tuple, new_state.puzzle)),
                )

    return None, search_tree


def visualize_tree(search_tree, path):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(search_tree)

    # Draw the search tree
    nx.draw(
        search_tree,
        pos,
        with_labels=True,
        node_size=700,
        node_color="lightblue",
        font_size=10,
        arrowsize=10,
    )

    # Color nodes in the path to the goal state
    for node in search_tree.nodes():
        if node in path:
            nx.draw_networkx_nodes(
                search_tree, pos, nodelist=[node], node_color="red", node_size=700
            )

    # Draw edges
    nx.draw_networkx_edges(search_tree, pos)

    plt.title("Search Tree with Path to Goal State")
    plt.show()


if __name__ == "__main__":
    start_state = [[1, 2, 3], [0, 4, 6], [7, 5, 8]]

    goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    start = EightPuzzleProblem(start_state)
    path, search_tree = bfs(start, goal_state)

    if path:
        print("Solution found!")
        print("Path to goal state:")
        for move in path:
            print(move)
        print()

        # Visualize the search tree
        visualize_tree(search_tree, path)
    else:
        print("No solution found!")


"""def bfs(start_state, goal_state):
    queue = deque([start_state])
    visited = set()

    while queue:
        current_state = queue.popleft()

        if current_state.is_goal_state(goal_state):
            return current_state.get_path()

        visited.add(tuple(map(tuple, current_state.puzzle)))

        for move in current_state.generate_moves():
            new_state = current_state.move_square(move)
            if tuple(map(tuple, new_state.puzzle)) not in visited:
                queue.append(new_state)
                visited.add(tuple(map(tuple, new_state.puzzle)))


if __name__ == "__main__":
    start_state = [
        [1, 2, 3],
        [0, 4, 6],
        [7, 5, 8],
    ]
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    start = EightPuzzleProblem(start_state)
    path = bfs(start, goal)

    if path:
        print("Solution found!")
        print("Path to goal state:")
        for move in path:
            print("Move > ", move)
    else:
        print("No solution found!")
"""
