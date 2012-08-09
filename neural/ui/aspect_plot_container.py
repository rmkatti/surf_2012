# System library imports.
import numpy as np
from chaco.api import HPlotContainer, VPlotContainer


class AspectStackedPlotMixin(object):
    """ A mixin for StackedPlotContainers that preserves aspect ratios during
    layout.
    """

    def _do_stack_layout(self, components, align):
        size = list(self.bounds)
        ndx = self.stack_index
        other_ndx = 1 - ndx

        # Allocate space proportionally along the main direction.
        available = size[ndx] - (len(components)-1) * self.spacing
        available -= sum(c.outer_bounds[ndx] - c.bounds[ndx] for c in components)
        current = sum(c.bounds[ndx] for c in components)
        scale = available / current
        for component in components:
            component.bounds = np.array(component.bounds) * scale
            #component.resizable = ''

        # If the resulting bounds are too large in the other direction, scale
        # down appropriately.
        largest = max(components, key = lambda c: c.outer_bounds[other_ndx])
        largest_outer = largest.outer_bounds[other_ndx]
        if largest_outer > size[other_ndx]:
            largest_bounds = largest.bounds[other_ndx]
            largest_padding = largest_outer - largest_bounds
            scale = (size[other_ndx] - largest_padding) / largest_bounds
            for component in components:
                component.bounds = np.array(component.bounds) * scale

        # Finally, assign positions to each component.
        cur_pos = 0
        for component in components:
            position = list(component.outer_position)
            position[ndx] = cur_pos

            bounds = list(component.outer_bounds)
            cur_pos += bounds[ndx] + self.spacing

            position[other_ndx] = 0
            if align == "min":
                pass
            elif align == "max":
                position[other_ndx] = size[other_ndx] - bounds[other_ndx]
            elif align == "center":
                position[other_ndx] = (size[other_ndx] - bounds[other_ndx]) / 2.0

            component.outer_position = position
            component.do_layout()


class HAspectPlotContainer(AspectStackedPlotMixin, HPlotContainer):
    """ An HPlotContainer that preserves the respective aspect ratios of its
    components.
    """
    pass

class VAspectPlotContainer(AspectStackedPlotMixin, VPlotContainer):
    """ A VPlotContainer that preserves the respective aspect ratios of its
    components.
    """
    pass
