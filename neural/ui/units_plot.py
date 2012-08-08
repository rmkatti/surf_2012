# System library imports.
from traits.api import Array, Bool, List, HasTraits, Instance, Range, \
    on_trait_change
from traitsui.api import Item, View
from enable.api import BaseTool, Component, ComponentEditor
from chaco.api import ArrayPlotData, ImagePlot, Plot, VPlotContainer
import chaco.default_colormaps as cm


class UnitsPlot(HasTraits):

    editable = Bool(False)

    layers = List(Array)
    pixel_size = Range(low=1, value=25)

    plot = Instance(Component)
    traits_view = View(Item('plot',
                            editor = ComponentEditor(),
                            show_label = False),
                       resizable = True)

    @on_trait_change('editable, layers, pixel_size')
    def rebuild_plot(self):
        container = VFitPlotContainer(bgcolor = 'lightgray',
                                      halign = 'center',
                                      stack_order = 'top_to_bottom')
        for layer in self.layers:
            data = ArrayPlotData(image = layer)
            plot = Plot(data,
                        width = layer.shape[1] * self.pixel_size,
                        height = layer.shape[0] * self.pixel_size,
                        padding = self.pixel_size,
                        resizable = '')
            plot.x_axis.visible = False
            plot.y_axis.visible = False

            # Specify color range manually to handle case of all 0's or 1's.
            img_plot = plot.img_plot(
                'image', colormap = cm.gray, origin = 'top left')[0]
            img_plot.color_mapper.range.low_setting = 0.0
            img_plot.color_mapper.range.high_setting = 1.0

            if self.editable:
                img_plot.tools.append(UnitToggleTool(img_plot))

            container.add(plot)
        self.plot = container


class UnitToggleTool(BaseTool):
    
    # BaseTool interface
    component = Instance(ImagePlot)

    def normal_left_down(self, event):
        self.handle_mouse_event(event)

    def normal_right_down(self, event):
        self.handle_mouse_event(event)

    def normal_mouse_move(self, event):
        self.handle_mouse_event(event)

    def handle_mouse_event(self, event):
        if event.left_down or event.right_down:
            plot = self.component
            x_idx, y_idx = idx = plot.map_index((event.x, event.y))
            if idx != (None, None):
                image_data = plot.value
                val = 1 if event.left_down else 0
                if image_data.data[y_idx, x_idx] != val:
                    image_data.data[y_idx, x_idx] = val
                    image_data.data_changed = True


class VFitPlotContainer(VPlotContainer):
    """ A VPlotContainer that fits its components but uses extra space if
    available.
    """

    # Container interface
    fit_components = 'hv'

    def get_preferred_size(self, components=None):
        size = super(VPlotContainer, self).get_preferred_size()
        size = (max(self.outer_bounds[0], size[0]), 
                max(self.outer_bounds[1], size[1]))
        self._cached_preferred_size = size
        return size
                

if __name__ == '__main__':
    import numpy as np

    y = np.random.random((1, 10)) < 0.2
    x = np.random.random((2, 20)) < 0.5
    layers = [y, x]

    plot = UnitsPlot(editable=True, layers=layers)
    plot.configure_traits()
