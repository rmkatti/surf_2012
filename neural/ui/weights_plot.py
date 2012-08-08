# System library imports.
import numpy as np
from traits.api import Array, HasTraits, Instance, Unicode
from traitsui.api import Item, View
from enable.api import Component, ComponentEditor
from chaco.api import ArrayPlotData, ColorBar, ImagePlot, Plot, \
    HPlotContainer, LinearMapper
from chaco.ticks import ShowAllTickGenerator
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
        plot.padding_top = 75
        plot.title = self.title
        plot.x_axis.orientation = 'top'
        plot.x_axis.tick_in = 0
        plot.y_axis.tick_in = 0
        self._update_ticks(plot)

        img_plot = plot.img_plot('image',
                                 colormap = cm.gray, 
                                 origin = 'top left')[0]
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
            self._update_ticks(plot)

    def _update_ticks(self, plot):
        # Align ticks at center, not left/top, of corresponding pixel. Use
        # the convention that the origin is (1,1).
        for i, axis in enumerate((plot.y_axis, plot.x_axis)):
            ticks = np.arange(-0.5, self.weights.shape[i]+1, 5)
            ticks[0] += 1
            axis.tick_generator = ShowAllTickGenerator(positions=ticks)
            axis.tick_label_formatter = lambda tick: str(int(np.ceil(tick)))


if __name__ == '__main__':
    weights = np.random.randn(20, 20)
    plot = WeightsPlot(title = '20x20 standard normal weights',
                       weights = weights)
    plot.configure_traits()
