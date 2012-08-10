# System library imports.
import numpy as np
from traits.api import Array, HasTraits, Instance, Unicode
from traitsui.api import Item, View
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, Plot
from chaco.tools.image_inspector_tool import ImageInspectorOverlay

# Local imports.
from hinton_plot_renderer import HintonPlotRenderer
from weights_plot import WeightInspectorTool


class HintonPlot(HasTraits):
    """ Visualizes a neural network weight matrix.

    Uses color to denote sign and area to denote magnitude.
    """

    # The title of the plot.
    title = Unicode

    # The weight matrix to visualize.
    weights = Array(dtype=np.double, shape=(None,None))

    plot = Instance(Component)
    traits_view = View(Item('plot',
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

        plot.renderer_map['cmap_img_plot'] = HintonPlotRenderer
        img_plot = plot.img_plot('image', name='image',
                                 origin = 'top left')[0]
        inspector = WeightInspectorTool(img_plot)
        overlay = ImageInspectorOverlay(component = plot,
                                        image_inspector = inspector,
                                        bgcolor = 'white',
                                        border_visible = True)
        img_plot.tools.append(inspector)
        img_plot.overlays.append(overlay)

        return plot

    def _title_changed(self):
        if self.traits_inited():
            self.plot.title = self.title

    def _weights_changed(self):
        if self.traits_inited():
            height, width = self.weights.shape
            plot = self.plot
            plot.aspect_ratio = float(width) / float(height)
            plot.data.set_data('image', self.weights)


if __name__ == '__main__':
    weights = np.random.randn(20, 20)
    plot = HintonPlot(title = '20x20 standard normal weights',
                      weights = weights)
    plot.configure_traits()
