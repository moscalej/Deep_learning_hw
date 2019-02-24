# %%
from math import sqrt

from numba import jit, njit
import numpy as np
from Final_Project.models import Preprocesses
import numpy as np
from keras.models import Sequential, Model
from collections import defaultdict
import yaml
from Final_Project.models.Preprocesses import reshape_all
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt

clock2orient = {12: 0, 6: 1, 9: 2, 3: 3}
orient2clock = {value: key for key, value in clock2orient.items()}
clock2str = {12: 'above', 6: 'below', 9: 'left', 3: 'right'}
orient2str = {0: 'above', 1: 'below', 2: 'left', 3: 'right'}


class Puzzle:
    def __init__(self, axis_size: int, first_crop: int, num_pieces: int):
        print(f"Puzzle started with {first_crop}")
        self.cyclic_puzzle = np.ones([axis_size, axis_size]) * -1
        self.cyclic_puzzle[0, 0] = first_crop
        self.axis_size = axis_size
        self.relative_dims = dict({3: 0, 6: 0, 9: 0, 12: 0})
        self.next_candidates = {first_crop: set([3, 6, 9, 12])}  # keys: puzzle_pieces, values: orientation not used
        self.relative_coo = dict()
        self.relative_coo[first_crop] = (0, 0)  # (right, left)
        self.relative2ind = dict()
        self.relative2ind[(0, 0)] = first_crop
        self.neighbours_def = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.directions_def = [3, 6, 9, 12]
        self.num_pieces = num_pieces

    def add_piece(self, attach2, _2attach: int, clock: int) -> None:
        """
        adds a piece to puzzle (relative coo and abs coo (cyclic_puzzle))
        updates next_candidates, relative dims
        :param attach2:
        :type attach2:
        :param _2attach:
        :type _2attach:
        :param clock:
        :type clock:
        :return:
        :rtype: None
        """
        print(f"adding puzzle piece: {_2attach} {clock2str[clock]} to {attach2}")
        neighbours = self.neighbours_def
        directions = self.directions_def
        (targetX, targetY) = self.relative_coo[attach2]  # relative coo
        for direct, (dX, dY) in zip(directions, neighbours):
            if direct != clock:  # execute given orientation
                continue
            self.relative_coo[_2attach] = (targetX + dX, targetY + dY)
            self.relative2ind[(targetX + dX, targetY + dY)] = _2attach
            curr_dim = self.relative_dims[direct]
            changed_axis = [abs(dX), abs(dY)].index(1)
            temp = self.relative_coo[_2attach][changed_axis]
            expand_dim = False
            if dX + dY > 0:
                if temp > curr_dim and temp > 0:
                    expand_dim = True
            elif dX + dY < 0:
                if temp < curr_dim and temp < 0:
                    expand_dim = True
            if expand_dim:  # check if relative dimension expanded
                self.relative_dims[direct] = self.relative_dims[direct] + dY + dX
            (absX, absY) = self._get_abs_coo(self.relative_coo[_2attach][0], self.relative_coo[_2attach][1])
            self.cyclic_puzzle[absX, absY] = _2attach  # put the piece in the puzzle
            neigh_tups = self._get_neigh(absX, absY)
            new_directs = set([3, 6, 9, 12])
            for (crop_ind, direct_) in neigh_tups:  # directions relative to new
                new_directs.add(direct_)  # avoid key errors
                new_directs.remove(direct_)
                rem = (6 + direct_) % 12 if direct_ != 6 else 12
                self.next_candidates[crop_ind].add(rem)
                self.next_candidates[crop_ind].remove(rem)  # remove opposite direction
            self.next_candidates[_2attach] = new_directs
            self._remove_out_of_frame()

    def get_puzzle(self, mode='label'):
        sorted = []
        label = [-1] * self.num_pieces
        rel_row = self.relative_dims[12]
        location_count = 0
        for row in range(self.axis_size):
            rel_col = self.relative_dims[9]
            for col in range(self.axis_size):
                sorted.append(int(self.cyclic_puzzle[self._get_abs_coo(rel_row, rel_col)]))
                # label.append(int())
                label[self.relative2ind[(rel_row, rel_col)]] = location_count
                location_count += 1
                rel_col += 1
            rel_row += 1
        if mode == 'label':
            return label
        else:
            return sorted

    def _remove_out_of_frame(self):
        # horizontal frame
        right_edge = []
        left_edge = []
        if self.relative_dims[3] - self.relative_dims[9] == self.axis_size - 1:
            right_edge = [piece for (x, y), piece in self.relative2ind.items() if self.relative_dims[3] == x]
            left_edge = [piece for (x, y), piece in self.relative2ind.items() if self.relative_dims[9] == x]
        if len(right_edge):
            for piece in right_edge:
                self.next_candidates[piece].add(3)
                self.next_candidates[piece].remove(3)
        if len(left_edge):
            for piece in left_edge:
                self.next_candidates[piece].add(9)
                self.next_candidates[piece].remove(9)

        # vertical frame
        bottom_edge = []
        upper_edge = []

        if self.relative_dims[6] - self.relative_dims[12] == self.axis_size - 1:
            bottom_edge = [piece for (x, y), piece in self.relative2ind.items() if self.relative_dims[6] == y]
            upper_edge = [piece for (x, y), piece in self.relative2ind.items() if self.relative_dims[12] == y]
        if len(bottom_edge):
            for piece in bottom_edge:
                self.next_candidates[piece].add(6)
                self.next_candidates[piece].remove(6)

        if len(upper_edge):
            for piece in upper_edge:
                self.next_candidates[piece].add(12)
                self.next_candidates[piece].remove(12)

    def _get_neigh(self, absX, absY):
        neighbours = self.neighbours_def
        directions = self.directions_def
        return [(self.cyclic_puzzle[self._get_abs_coo(absX + dX, absY + dY)], directions[ind]) for ind, (dX, dY) in
                enumerate(neighbours)
                if self.cyclic_puzzle[self._get_abs_coo(absX + dX, absY + dY)] != -1]

    def _get_abs_coo(self, relX, relY):
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
    tasks_0 = []
    tasks_1 = []

    for candidate in range(crop_num):
        for center in range(crop_num):
            keys.append((candidate, center))  # (center of the universe, candidate)
            tasks_0.append(crop_list[center])
            tasks_1.append(crop_list[candidate])

    tasks_0 = np.expand_dims(np.array(tasks_0), 3)
    tasks_1 = np.expand_dims(np.array(tasks_1), 3)
    results = np.zeros([crop_num, crop_num, 5], dtype=np.float64)
    print(f'Shape of task {tasks_0.shape}, {tasks_1.shape}')
    predicted = matcher.predict([tasks_0, tasks_1])
    return fast_fill_mat(predicted, keys, results)


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
    # results /= results.sum(axis=1, keepdims=True) + 1e-20
    return results


def choose_next(candidates, match_prob_dict):
    best_candidate = (-1, -1, -1)
    best_candidate_prob = 0 - 1e-9
    if candidates == []:  # choose greedy  # todo smart choice
        for candidate_ind in range(match_prob_dict.shape[0]):
            attach2 = match_prob_dict[:, candidate_ind, :]  # candidate is a matrix num_crops X directions
            max_matches_inds = np.argmax(attach2, axis=0)
            max_matches = attach2[max_matches_inds]  # prob_list
            mean_match_prob = np.mean(max_matches)
            if mean_match_prob > best_candidate_prob:
                best_candidate_prob = mean_match_prob
                best_candidate = (candidate_ind, None, None)
        match_prob_dict[best_candidate[0], :, :] = -1
    else:
        for attach2, clocks in candidates.items():
            _2attach_mat = match_prob_dict[:, attach2, :]
            for clock in clocks:
                orient = clock2orient[clock]
                options = _2attach_mat[:, orient]
                _2attach = np.argmax(options)
                best_prob = options[_2attach]
                if best_prob > best_candidate_prob:
                    best_candidate_prob = best_prob
                    best_candidate = tuple([_2attach, attach2, orient2clock[orient]])
        match_prob_dict[best_candidate[0], :, :] = -1

    return best_candidate


def matcher_wrap(matcher, crop1, crop2, orient):
    stiched = Preprocesses.stich(crop1, crop2, orient)
    return matcher.predict(stiched)


# High Level


def chooss_ood(prob_tensor, crop_num):
    return prob_tensor[:, :, :4]


def assemble(crop_list: list, matcher: Model) -> np.array:
    crop_num = len(crop_list)
    axis_size = int(np.sqrt(crop_num))
    prob_tensor = get_prob_dict(crop_list, matcher)  # sorted dictionary
    prob_tensor_cut = chooss_ood(prob_tensor, crop_num)
    anchor_crop, _, _ = choose_next([], prob_tensor_cut)
    puzzle = Puzzle(axis_size, anchor_crop, crop_num)

    for ind in range(axis_size ** 2 - 1):
        candidates = puzzle.next_candidates
        _2attach, attach2, orient = choose_next(candidates, prob_tensor)
        puzzle.add_piece(attach2, _2attach, orient)
    print(puzzle.cyclic_puzzle)
    return puzzle.get_puzzle()


def predict_2(images: list, showimage: bool = True):
    with open(
            r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\parameters.YAML')as fd:  # todo
        param = yaml.load(fd)
    model = load_model(param['Discri']['path'])
    crops = reshape_all(images, 64, param['Discri']['x_mean'], param['Discri']['x_std'])
    if showimage:
        plot_crops(crops)
    labels = assemble(crop_list=crops, matcher=model)
    if showimage:
        plot_in_order(labels, images)
    return labels

def predict_3(images: list):
    with open(
            r'C:\Users\amoscoso\Documents\Technion\deeplearning\Deep_learning_hw\Final_Project\parameters.YAML')as fd:  # todo
        param = yaml.load(fd)
    model = load_model(param['Discri']['path'])
    accuary = []
    for img in images:
        crops = reshape_all(img, 64, param['Discri']['x_mean'], param['Discri']['x_std'])
        num_crops = len(crops)
        cuts = int(sqrt(num_crops))
        awnser = [x if x< cuts**2 else -1 for x in range(num_crops) ]
        labels = assemble(crop_list=crops, matcher=model)
        accuary.append(np.mean(labels == awnser))

    return np.mean(accuary)


# test for assembler
# assume puzzle
# 0,1,2
# 3,4,5,
# 6,7,8
# up - 0, down - 1, left - 2, right - 3
def test_anssemble():
    axis_size = 3
    prob_tensor = np.zeros([9, 9, 4])
    prob_tensor[0, [1, 3], [3, 1]] = 1
    prob_tensor[1, [0, 4, 2], [2, 1, 3]] = 1
    prob_tensor[2, [1, 5], [2, 1]] = 1
    prob_tensor[3, [0, 4, 6], [0, 3, 1]] = 1
    prob_tensor[5, [2, 4, 8], [0, 2, 1]] = 1
    prob_tensor[6, [3, 7], [0, 3]] = 1
    prob_tensor[7, [6, 4, 8], [2, 0, 3]] = 1
    prob_tensor[8, [7, 5], [2, 0]] = 1
    prob_tensor *= 0.9
    prob_tensor[4, [3, 1, 5, 7], [2, 0, 3, 1]] = 1
    anchor_crop, _, _ = choose_next([], prob_tensor)
    puzzle = Puzzle(axis_size, anchor_crop)
    for ind in range(axis_size ** 2 - 1):
        candidates = puzzle.next_candidates
        target, new, orient = choose_next(candidates, prob_tensor)
        puzzle.add_piece(target, new, orient)
    puzzle.get_puzzle()
    return puzzle.get_puzzle('a')


def sinkhorn(A, n_iter=4):
    for i in range(n_iter):
        A /= A.sum(axis=1, keepdims=True) + 1e-20
        A /= A.sum(axis=2, keepdims=True) + 1e-20
    return A


def plot_crops(crops):
    crop_num = len(crops)
    axis_size = int(np.sqrt(crop_num))
    fig = plt.figure(figsize=(axis_size, axis_size))
    ax = [plt.subplot(axis_size+1, axis_size, i + 1) for i in range(crop_num)]
    for ind, crop in enumerate(crops):
        ax[ind].imshow(crop)
        ax[ind].set_title(ind)
        # plt.imshow(crop)
        ax[ind].set_aspect('equal')
        ax[ind].set_xticklabels([])
        ax[ind].set_yticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle("Original crops")
    plt.show()


def plot_in_order(labels, crops):
    crop_num = len(labels)
    axis_size = int(np.sqrt(crop_num))
    fig = plt.figure(figsize=(axis_size, axis_size))
    ax = [plt.subplot(axis_size, axis_size, i + 1) for i in range(crop_num)]
    for ind, label in enumerate(labels):
        current_image = crops[label]
        ax[ind].imshow(current_image)
        plt.imshow(current_image)
        ax[ind].set_aspect('equal')
        ax[ind].set_xticklabels([])
        ax[ind].set_yticklabels([])
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()
