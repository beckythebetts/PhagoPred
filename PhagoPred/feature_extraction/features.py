from PhagoPred.feature_extraction.morphology.fitting import MorphologyFit

class BaseFeature:

    primary_feature = False
    derived_feature = False

    def __init__(self):
        self.name = [self.__class__.__name__]
        self.index_positions = None

    def get_names(self):
        return self.name
    
    def compute(self):
        raise NotImplementedError(f"{self.get_name()} has no compute method implemented")
    
    def set_index_positions(self, start: int, end: int = None) -> None:
        self.index_positions = (start, end)
    
    def get_index_positions(self) -> list[int]:
        return self.index_positions
    

class MorphologyModes(BaseFeature):

    primary_feature = True

    def get_names(self):
        return [f'Mode {i}' for i in range(self.model.num_kmeans_clusters)]
    
    def set_morphology_model(self, model: MorphologyFit):
        self.model = model
    
    def compute(self, expanded_frame_mask, frame_image, num_cells):
        """
        Results are of dims [num_cells, num_clusters]
        """
        expanded_frame_mask = expanded_frame_mask.cpu().numpy()
        results = self.model.apply_expanded_mask(expanded_frame_mask, num_cells)
        return results

