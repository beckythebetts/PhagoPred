import numpy as np


class Track:
    def __init__(self, frames, indices):
        self.track_dict = {frame: index for frame, index in zip([frames], [indices])}

    def add_frame(self, frame, index):
        self.track_dict[frame] = index

class NearestNeighbourTracking:
    def __init__(self, frames, indices, centres):
        if len(frames) == len(indices) and len(frames) == len(indices):
            self.frames = np.array(frames)
            self.indices = np.array(indices)
            self.centres = np.array(centres)
        else:
            raise Exception('Input arrays must be same length')

    def track(self):
        unique_frames = np.unique(self.frames)
        for i, frame in enumerate(unique_frames):
            if i != 0:
                old_indices, old_centres = new_indices, new_centres
                current_tracks = [tracks for tracks in self.tracked if list(tracks.track_dict.keys())[-1]==unique_frames[i-1]]
            frame_indices = np.squeeze(np.argwhere(self.frames==frame))
            new_indices, new_centres = np.reshape(self.indices[frame_indices], -1), np.reshape(self.centres[frame_indices], (-1, 2))
            if i == 0:
                self.tracked = [Track(frame, index) for index in new_indices]
                continue
            distances = np.reshape(np.linalg.norm(old_centres[:, np.newaxis] - new_centres[np.newaxis,], axis=2), (len(old_centres), len(new_centres)))
            #print(distances)
            if len(new_indices) >= len(old_indices):
                new_indices_temp = new_indices.copy()
                for j, old_index in enumerate(old_indices):
                    new_index = new_indices_temp[np.argmin(distances[j])]
                    new_indices_temp = np.delete(new_indices_temp, np.argwhere(new_indices_temp==new_index))
                    distances = np.delete(distances, np.argmin(distances[j]), axis=1)
                    #print('new>old', distances)
                    for i, track in enumerate(current_tracks):
                        if list(track.track_dict.values())[-1]==old_index:
                            track.add_frame(frame, new_index)
                            current_tracks = np.delete(current_tracks, i)
                            continue
                for new_index in new_indices_temp:
                    self.tracked = np.append(self.tracked, Track(frame, new_index))
            else:
                old_indices_temp = old_indices.copy()
                for j, new_index in enumerate(new_indices):
                    old_index = old_indices_temp[np.argmin(distances[:, j])]
                    old_indices_temp = np.delete(old_indices_temp, np.argwhere(old_indices_temp==old_index))
                    distances = np.delete(distances, np.argmin(distances[:, j]), axis=0)
                    #print('old>new', distances)
                    for i, track in enumerate(current_tracks):
                        if list(track.track_dict.values())[-1]==old_index:
                            track.add_frame(frame, new_index)
                            current_tracks = np.delete(current_tracks, i)
                            continue

        # for tracked in self.tracked:
        #     print(tracked.track_dict)


def main():
    test = NearestNeighbourTracking(frames=[0, 1, 2, 2, 3, 3, 4, 6], indices=[2, 2, 2, 9, 2, 9, 9, 8], centres=[[0,0], [1,2], [2, 1], [3, 5], [2, 2], [0, 0], [2, 2], [3, 3]])
    test.track()


if __name__ == '__main__':
    main()