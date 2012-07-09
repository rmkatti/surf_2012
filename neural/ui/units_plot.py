# System library imports.
from traits.api import Array, List, Float, HasTraits, Instance, on_trait_change
from traitsui.api import Item, View
from enable.api import ComponentEditor
from chaco.api import ArrayPlotData, Plot, VPlotContainer
import chaco.default_colormaps as cm


class UnitsPlot(HasTraits):

    layers = List(Array)
    pixel_size = Float(25.0)

    plot = Instance(VPlotContainer)
    traits_view = View(Item('plot',
                            editor = ComponentEditor(),
                            show_label = False),
                       resizable = True)

    @on_trait_change('layers')
    def rebuild_plot(self):
        container = VPlotContainer(bgcolor = 'lightgray',
                                   fit_components = 'hv',
                                   halign = 'center', 
                                   stack_order = 'top_to_bottom')
        for layer in self.layers:
            data = ArrayPlotData(image = layer)
            plot = Plot(data,
                        width = layer.shape[1] * self.pixel_size,
                        height = layer.shape[0] * self.pixel_size,
                        resizable = '')
            plot.x_axis.visible = False
            plot.y_axis.visible = False
            renderer = plot.img_plot(
                'image', colormap = cm.gray, origin = 'top left')[0]
            renderer.color_mapper.range.low_setting = 0.0
            renderer.color_mapper.range.high_setting = 1.0
            container.add(plot)
        self.plot = container


if __name__ == '__main__':
    import numpy as np

    y = np.random.random((1, 10)) < 0.2
    x = np.random.random((2, 20)) < 0.5
    layers = [y, x]

    plot = UnitsPlot(layers=layers)
    plot.configure_traits()
