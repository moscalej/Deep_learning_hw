import numpy as np
from keras.models import Sequential, Model


class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None


class SLinkedList:
    def __init__(self):
        self.headval = None
        self.above = 0
        self.below = 0

    def listprint(self):
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval

    def AtBegining(self, newdata):
        NewNode = Node(newdata)
        NewNode.nextval = self.headval
        self.headval = NewNode

    def AtEnd(self, newdata):
        NewNode = Node(newdata)
        if self.headval is None:
            self.headval = NewNode
            return
        laste = self.headval
        while (laste.nextval):
            laste = laste.nextval
        laste.nextval = NewNode


# High Level:

class Puzzle:
    def __init__(self, axis_size: int, first_crop: int):

        self.horizontals = dict()
        self.horizontal = SLinkedList()
        self.horizontal.AtBegining(SLinkedList().AtBegining(first_crop))
        # self.verticals = dict()
        # self.verticals['0'] = SLinkedList()
        # self.verticals.AtBegining(first_crop)

        self.axis_size = axis_size
        self.height = 1
        self.width = 1
        self.left = 0
        self.right = 0
        self.next_candidates = {first_crop: [3, 6, 9, 12]}
        self.crops_coo[first_crop] = (0,0)  # (right, left)
    def add_piece(self, target_crop, new_crop: int, orientation: int):
        horizontals_ind = self.crops_coo(target_crop)
        (targetX, targetY) =self.crop_coo[target_crop][0]
        if orientation == 3:  # right
            if  targetX< self.right:  # linked list for vertical exists
                distance = np.abs(self.left)+targetX +1
                vertical_list = n_steps(distance)

                .AtEnd(new_crop)
        if orientation == 9:  # left
            self.horizontals[horizontals_ind].AtBegining(new_crop)
        if orientation == 12:  # above
        if orientation == 6:  # below



class Crop:
    def __init__(self, full_size, img_ind):
        self.full_size = full_size
        self.img_inds = img_ind


def get_tasks(crops_list: list, axis_size: int) -> dict:
    pass


def create_crops(image_list):
    create_crops


def Assemble(image_list: list, matcher: Model) -> np.array:
    candidates = len(image_list)
    axis_size = int(np.sqrt(len(image_list)))
    crops_list = create_crops(image_list)
    while (True):
        task_dict = get_tasks(crops_list, axis_size)
        for target, space in task_dict.items():
            pairs = matcher(target, space)
        crops_list = filter_and_stich(pairs, crops_list)
