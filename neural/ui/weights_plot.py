# System library imports.
import numpy as np
from traits.api import Array, HasTraits, Instance, Unicode
from traitsui.api import Item, View
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, ColorBar, Plot, HPlotContainer, \
    LinearMapper
from chaco.tools.image_inspector_tool import ImageInspectorTool, \
    ImageInspectorOverlay
import chaco.default_colormaps as cm


class WeightsPlot(HasTraits):
    """ Visualizes a neural network weight matrix.
    """

    # The title of the plot.
    title = Unicode

    # The weight matrix to visualize.
    weights = Array(dtype=np.double, shape=(None,None))

    container = Instance(Component)
    plot = Instance(Component)
    traits_view = View(Item('container',
                            editor = ComponentEditor(),
                            show_label = False),
                       resizable = True)

    def _plot_default(self):
        height, width = self.weights.shape
        plot_data = ArrayPlotData(image = self.weights)

        plot = Plot(plot_data,
                    aspect_ratio = float(width) / float(height),
                    default_origin = 'top left')
        plot.padding = 25
        plot.title = self.title
        plot.x_axis.visible = False
        plot.y_axis.visible = False

        img_plot = plot.img_plot('image', name='image',
                                 colormap = cm.gray, 
                                 origin = 'top left')[0]
        inspector = WeightInspectorTool(img_plot)
        img_plot.tools.append(inspector)

        return plot
    
    def _container_default(self):
        plot = self.plot
        colormap = plot.color_mapper
        colorbar = ColorBar(index_mapper = LinearMapper(range=colormap.range),
                            color_mapper = colormap,
                            plot = plot,
                            orientation = 'v', resizable = 'v',
                            width = 30, padding = 20)
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        container = HPlotContainer(bgcolor = 'lightgray',
                                   use_backbuffer = True)
        container.add(plot)
        container.add(colorbar)

        inspector = plot.plots['image'][0].tools[0]
        overlay = ImageInspectorOverlay(component = container,
                                        image_inspector = inspector,
                                        bgcolor = 'white',
                                        border_visible = True)
        container.overlays.append(overlay)

        return container

    def _title_changed(self):
        if self.traits_inited():
            self.plot.title = self.title

    def _weights_changed(self):
        if self.traits_inited():
            height, width = self.weights.shape
            plot = self.plot
            plot.aspect_ratio = float(width) / float(height)
            plot.data.set_data('image', self.weights)


class WeightInspectorTool(ImageInspectorTool):

    def normal_mouse_move(self, event):
        plot = self.component
        if plot is not None:
            ndx = plot.map_index((event.x, event.y))
            if ndx == (None, None):
                self.new_value = None
            else:
                x, y = ndx
                self.new_value = dict(indices = (y+1,x+1),
                                      data_value = plot.value.data[y, x])
                self.last_mouse_position = (event.x, event.y)


if __name__ == '__main__':
    weights = np.random.randn(20, 20)
    plot = WeightsPlot(title = '20x20 standard normal weights',
                       weights = weights)
    plot.configure_traits()
