from numba import jit, njit

from Final_Project.models import Preprocesses
import numpy as np
from keras.models import Sequential, Model
from collections import defaultdict

clock2orient = {12: 0, 6: 1, 9: 2, 3: 3}


class Puzzle:
    def __init__(self, axis_size: int, first_crop: int):

        self.cyclic_puzzle = np.ones(axis_size, axis_size) * -1
        self.cyclic_puzzle[0, 0] = first_crop
        self.axis_size = axis_size
        self.relative_dims = dict({3: 0, 6: 0, 9: 0, 12: 0})
        self.next_candidates = {first_crop: set([3, 6, 9, 12])}  # keys: puzzle_pieces, values: orientation not used
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
            directs = set([3, 6, 9, 12])
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


def get_prob_dict(crop_list: list, matcher: Model) -> np.ndarray:
    """
    Build a probability model for ech picture
    :param crop_list:
    :type crop_list:
    :param matcher:
    :type matcher:
    :return:
    :rtype:
    """
    directions_def = [0, 1, 2, 3]  # [up, down, left, right]
    crop_num = len(crop_list)
    keys = []
    tasks = []
    for crop_ind in range(crop_num):
        for cand_ind in range(crop_num):
            for orient in directions_def:
                keys.append(tuple([crop_ind, cand_ind, orient]))
                tasks.append(Preprocesses.stich(crop_list[crop_ind], crop_list[cand_ind], orient))

    tasks = np.array(tasks)
    results = np.zeros([crop_num, crop_num, 4], dtype=np.float64)
    print(f'Shape of task {tasks.shape}')
    predicted = matcher.predict(tasks)
    return fast_fill_mat(predicted[:, 1], keys, results)


# @njit()
def fast_fill_mat(predic: np.ndarray, keys: list, results: np.ndarray) -> np.ndarray:
    """
    This method will fill a tensor with the probabilities of all the borders
    :param predic:
    :type predic:
    :param keys:
    :type keys:
    :param results:
    :type results:
    :return:
    :rtype:
    """
    for pred, key in zip(predic, keys):
        results[key] = pred
    return results


def choose_next(candidates, match_prob_dict):
    best_candidate = (-1, -1, -1)
    best_candidate_prob = 0
    if candidates == []:  # choose greedy  # todo smart choice
        for candidate_ind in range(match_prob_dict.shape[0]):
            candidate = match_prob_dict[candidate_ind]  # candidate is a matrix num_crops X directions
            max_matches_inds = np.argmax(candidate, axis=0)
            max_matches = candidate[max_matches_inds]  # prob_list
            mean_match_prob = np.mean(max_matches)
            if mean_match_prob > best_candidate_prob:
                best_candidate_prob = mean_match_prob
                best_candidate = (candidate_ind, None, None)

    else:
        for candidate, clocks in candidates.items():
            candidate_mat = match_prob_dict[candidate, :, :]
            for clock in clocks:
                orient = clock2orient[clock]
                options = candidate_mat[:, orient]
                best_crop = np.argmax(options)
                best_prob = options[best_crop]
                if best_prob > best_candidate_prob:
                    best_candidate_prob = best_prob
                    best_candidate = tuple(candidate, best_crop, orient)

    return best_candidate


def matcher_wrap(matcher, crop1, crop2, orient):
    stiched = Preprocesses.stich(crop1, crop2, orient)
    return matcher.predict(stiched)


# High Level
def Assemble(crop_list: list, matcher: Model) -> np.array:
    crop_num = len(crop_list)
    axis_size = int(np.sqrt(crop_num))

    match_prob_dict = get_prob_dict(crop_list, matcher)  # sorted dictionary

    anchor_crop, _, _ = choose_next([], match_prob_dict)
    puzzle = Puzzle(axis_size, anchor_crop)

    for _ in range(axis_size ** 2 - 1):
        candidates = puzzle.next_candidates
        target, new, orient = choose_next(candidates, match_prob_dict)
        puzzle.add_piece(target, new, orient)


def predict(path_list, matcher, shape):
    results = []
    for path in path_list:
        crops = Preprocesses.preprocess_inference(path, shape)  # todo: size handling
        results.append(Assemble(crops, matcher))
    pass
