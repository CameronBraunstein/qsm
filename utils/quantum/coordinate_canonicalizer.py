def get_coordinate_canonicalizer(args):
    cc =CoordinateCanonicalizer(args)
    return cc

class CoordinateCanonicalizer:
    def __init__(self,args):
        pass
    def set_bundle(self,bundle):
        self.height_range, self.width_range = bundle
    def canonicalize_coordinates(self,y,x):
        height_offset = self.height_range[0]
        width_offset = self.width_range[0]
        canonical_y = y -height_offset
        canonical_x = x -width_offset
        return canonical_y,canonical_x