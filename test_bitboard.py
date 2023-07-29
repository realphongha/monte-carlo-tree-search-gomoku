import numpy as np


def board_to_bitboard(b):
    bb = 0
    h, w = b.shape[:2]
    for i in range(h+1):
        i_ = h - 1 - i
        if i_ < 0:
            continue
        for j in range(w):
            if b[i_, j]:
                bb += (1 << (i + j*(h+1)))
    return bb

def bitboard_to_board(bb, h, w):
    b = np.zeros((h, w), dtype=bool)
    for i in range(h+1):
        i_ = h - 1 - i
        if i_ < 0:
            continue
        for j in range(w):
            b[i_, j] = bb & (1 << (i + j*(h+1)))
    return b

def won(bb, m, n, k):
    # vertical
    res = bb
    for i in range(1, k):
        res &= (bb >> i)
    if res:
        return True
    
    # horizontal
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+1)*i))
    if res:
        return True
    
    # diagonal \
    res = bb
    for i in range(1, k):
        res &= (bb >> (n*i))
    if res:
        return True
    
    # diagonal /
    res = bb
    for i in range(1, k):
        res &= (bb >> ((n+2)*i))
    if res:
        return True
    return False


def put(bb, i, j, h, w):
    # i - height
    i_ = h - 1 - i
    bb += (1 << (i_ + j*(h+1)))
    return bb

def get(bb, i, j, h):
    # i - height
    return bb & (1 << (h - 1 - i + j*(h+1))) != 0

if __name__ == "__main__":
    board = [[0, 1, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [1, 1, 0, 1]]
    board = np.array(board)
    h, w = board.shape[:2]
    bb = board_to_bitboard(board)
    b = bitboard_to_board(bb, h, w).astype(np.uint8)
    print(board)
    print(bb)
    print(b)
    print(won(bb, w, h, 3))
    print(get(bb, 1, 0, 4))
    bb = put(bb, 1, 0, 4, 4)
    print(get(bb, 1, 0, 4))
