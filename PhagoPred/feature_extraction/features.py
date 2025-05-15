class BaseFeature:
    def __init__(self):
        self.name = self.__class__.__name__

    def get_names(self):
        return self.name
    
    def compute(self):
        raise NotImplementedError(f"{self.get_name()} has no compute method implemented")
    

class MorphologyModes(BaseFeature):

    def get_names(self):
        return [f'Mode {i}' for i in range(self.model.num_kmeans_clusters)]
    
    def set_morphology_model(self, model: MorphologyFit):
        self.model = model
    
    def compute(self, expanded_frame_mask, frame_image, num_cells):
        self.model.apply(expanded_frame_mask, num_cells)

