# System library imports.
import numpy as np
from enable.colors import ColorTrait
from chaco.api import Base2DPlot
from kiva.constants import FILL


class HintonPlotRenderer(Base2DPlot):

    #### 'PlotComponent' interface ############################################

    bgcolor = 'gray'

    #### 'HintonPlotRenderer' interface #######################################

    on_color = ColorTrait('white')
    off_color = ColorTrait('black')

    ###########################################################################
    # 'Base2DPlot' interface.
    ###########################################################################

    def _render(self, gc):
        """ Draw the plot.
        """
        W = self.value.data
        max_weight = 2 ** np.ceil(np.log2(np.max(np.abs(W))))

        # Compute the (maximum) side length of each 'pixel'.
        mapper = self.index_mapper
        base_side = np.abs((mapper.x_high_pos - mapper.x_low_pos) / 
                           (mapper.range.x_range.high - 
                            (mapper.range.x_range.low)))

        # Translate each index to screen space.
        indices = list(np.ndindex(W.T.shape))
        points = self.map_screen(indices)
        if 'right' in self.origin:
            pounts[:,0] -= base_side
        if 'top' in self.origin:
            points[:,1] -= base_side

        with gc:
            for idx, (x,y) in zip(indices, points):
                w = W.T[idx]
                if w != 0:
                    scale = np.sqrt(min(1.0, np.abs(w) / max_weight))
                    side = base_side * scale
                    offset = (base_side - side) / 2
                    color = self.on_color_ if w > 0 else self.off_color_
                    gc.set_fill_color(color)
                    gc.draw_rect((x+offset, y+offset, side, side), FILL)

    def map_index(self, screen_pt, threshold=0.0, outside_returns_none=True,
                  index_only=False):
        """ Maps a screen space point to an index into the plot's index array.

        Implements the AbstractPlotRenderer interface. Uses 0.0 for *threshold*,
        regardless of the passed value.
        """
        # For image plots, treat hittesting threshold as 0.0, because it's
        # the only thing that really makes sense.
        return super(HintonPlotRenderer, self).map_index(
            screen_pt, 0.0, outside_returns_none, index_only)
