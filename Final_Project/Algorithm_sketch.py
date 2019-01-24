High Level:

def Assemble(crops_collection: list, matcher): -> np.array
    num_real = get_num_real(len(crops_collection))
    while(True):
        task_dict = asign_tasks(crops_collection)
        for target, space in task_dict.items():
            pairs = matcher(target, space)
        crops_collection = filter_and_stich(pairs, crops_collection)
