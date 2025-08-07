import numpy as np

# Initial track_id_map: suppose we have tracks labeled 1 to 4
import numpy as np

def single_pass_merge(track_id_map, queue):
    for track_0, track_1 in queue:
        mask = (track_id_map == track_1)
        track_id_map[mask] = track_0
        queue[queue == track_1] = track_0  # update queue on the fly
    return track_id_map

def iterative_merge(track_id_map, queue):
    changed = True
    while changed:
        changed = False
        for track_0, track_1 in queue:
            if track_0 == track_1:
                continue
            mask = (track_id_map == track_1)
            if np.any(mask):
                track_id_map[mask] = track_0
                queue[queue == track_1] = track_0
                changed = True
    return track_id_map


track_id_map = np.array([5, 4, 3, 2, 1])
queue = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5]
])

print("Original track_id_map:", track_id_map)

result_single_pass = single_pass_merge(track_id_map.copy(), queue.copy())
print("Single pass merge result:", result_single_pass)

result_iterative = iterative_merge(track_id_map.copy(), queue.copy())
print("Iterative merge result:", result_iterative)
