from dataclasses import dataclass

from .transforms import Transform
from .thresholds import Threshold

@dataclass
class Rule:
    """Rule to apply to features / hazard functions."""
    def __init__(self, features: list[str], transforms: list[Transform], thresholds = list[Threshold]):
        
        
    
if __name__ =='__main__':
    rule = Rule(Marker.GRADIENT())