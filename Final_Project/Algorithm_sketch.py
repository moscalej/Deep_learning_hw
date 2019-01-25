import numpy as np
from keras.models import Sequential, Model


class Puzzle:
    def __init__(self, axis_size: int, first_crop: int):

        self.cyclic_puzzle = np.ones(axis_size, axis_size) * -1
        self.cyclic_puzzle[0, 0] = first_crop
        self.axis_size = axis_size
        self.relative_dims = dict({3: 0, 6: 0, 9: 0, 12: 0})
        self.next_candidates = {first_crop: set(3, 6, 9, 12)}
        self.relative_coo[first_crop] = (0, 0)  # (right, left)
        self.neighbours_def = [(1, 0), (0, -1), (-1, 0), (0, 1)]
        self.directions_def = [3, 6, 9, 12]

    def add_piece(self, target_crop, new_crop: int, orientation: int) -> None:
        """
        adds a piece to puzzle (relative coo and abs coo (cyclic_puzzle))
        updates next_candidates, relative dims
        :param target_crop:
        :type target_crop:
        :param new_crop:
        :type new_crop:
        :param orientation:
        :type orientation:
        :return:
        :rtype: None
        """
        neighbours = self.neighbours_def
        directions = self.directions_def
        (targetX, targetY) = self.relative_coo[target_crop]  # relative coo
        for direct, (dX, dY) in enumerate(zip(directions, neighbours)):
            if direct != orientation:  # execute given orientation
                continue
            self.relative_coo[new_crop] = (targetX + dX, targetY + dY)
            curr_dim = self.relative_dims[direct]
            if max(abs(dX * targetX), abs(dY * targetY)) == curr_dim:  # check if relative dimension expanded
                self.relative_dims[direct] = self.relative_dims[direct] + 1
            (absX, absY) = self.get_abs_coo(self.relative_coo[new_crop])
            self.cyclic_puzzle[absX, absY] = new_crop  # put the piece in the puzzle
            neigh_tups = self.get_neigh(absX, absY)
            directs = {3, 6, 9, 12}
            for (crop_ind, direct) in neigh_tups:
                directs.remove(direct)
                self.next_candidates[crop_ind].remove((6 + direct) % 12)  # remove opposite direction
            self.next_candidates[new_crop] = directs


    def get_neigh(self, absX, absY):
        neighbours = self.neighbours_def
        directions = self.directions_def
        return [(self.cyclic_puzzle[absX + dX, absY + dY], directions[ind]) for ind, (dX, dY) in enumerate(neighbours)
                if self.cyclic_puzzle[absX + dX, absY + dY] != -1]

    def get_abs_coo(self, relX, relY):
        return (relX % self.axis_size, relY % self.axis_size)


def get_prob_dict(crop_list, matcher):
    neighbours_def = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    directions_def = set([3, 6, 9, 12])
    crop_num = len(crop_list)
    prob_dict = dict()
    for crop_ind in range(crop_num):
        prob_dict[crop_ind] = {}
        for crop_cand_ind in range(crop_num):
            prob_dict[crop_ind][crop_cand_ind] = {}
            for direction in directions_def:
                prob_dict[crop_ind][crop_cand_ind][direction] = matcher()




# High Level
def Assemble(crop_list: list, matcher: Model) -> np.array:
    axis_size = int(np.sqrt(len(crop_list)))

    match_prob_dict = get_prob_dict(crop_list, matcher)  # sorted dictionary

    anchor_crop = get_top_match(dict_matches)
    puzzle = Puzzle(axis_size, anchor_crop)

    for _ in range(axis_size**2-1):
        candidates = puzzle.next_candidates
        target, new, orient = choose_next(candidates, match_prob_dict)
        puzzle.add_piece(target, new, orient)
