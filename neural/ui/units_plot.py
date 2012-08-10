# System library imports.
from traits.api import Array, Enum, Event, List, HasTraits, Instance, Int, \
    Tuple, on_trait_change
from traitsui.api import Item, View
from enable.api import BaseTool, Component, ComponentEditor
from chaco.api import ArrayPlotData, ImagePlot, Plot
from chaco.tools.image_inspector_tool import ImageInspectorTool, \
    ImageInspectorOverlay
import chaco.default_colormaps as cm

# Local imports.
from aspect_plot_container import VAspectPlotContainer


class UnitsPlot(HasTraits):
    """ Plots multiple layers of binary-valued neural network units.
    """

    # The list of 2-dimensional binary-valued units matrices to visualize.
    layers = List(Array(shape=(None,None)))

    # Fired when a unit is activated (double-clicked). The event is a tuple of
    # form (layer, row, column), indexed from zero.
    activated = Event(Tuple(Int, Int, Int))

    # The currently active tool.
    tool = Enum('none', 'activate', 'toggle')

    plot = Instance(Component)
    traits_view = View(Item('plot',
                            editor = ComponentEditor(),
                            show_label = False),
                       resizable = True)

    @on_trait_change('layers, tool')
    def rebuild_plot(self):
        container = VAspectPlotContainer(bgcolor = 'lightgray',
                                         halign = 'center',
                                         stack_order = 'top_to_bottom')
        overlay = ImageInspectorOverlay(component = container,
                                        bgcolor = 'white',
                                        border_visible = True)
        container.overlays.append(overlay)

        for layer_index, layer in enumerate(self.layers):
            height, width = layer.shape
            data = ArrayPlotData(image = layer)
            plot = Plot(data,
                        aspect_ratio = float(width) / float(height),
                        width = width, height = height,
                        padding = 15)
            plot.x_axis.visible = False
            plot.y_axis.visible = False

            # Specify color range manually to handle case of all 0's or 1's.
            img_plot = plot.img_plot(
                'image', colormap = cm.gray, origin = 'top left')[0]
            img_plot.color_mapper.range.low_setting = 0.0
            img_plot.color_mapper.range.high_setting = 1.0

            def update_overlay(inspector, name, active):
                overlay.image_inspector = inspector if active else None
                overlay.visible = active
            inspector = UnitInspectorTool(img_plot)
            inspector.on_trait_change(update_overlay, 'active')
            img_plot.tools.append(inspector)

            self._add_tool(layer_index, img_plot)

            container.add(plot)
        
        self.plot = container

    def _add_tool(self, layer_index, plot):
        if self.tool == 'activate':

            def update_activated(ndx):
                self.activated = (layer_index, ndx[1], ndx[0])

            tool = UnitActivateTool(plot)
            tool.on_trait_change(update_activated, 'activated')
            plot.tools.append(tool)

        elif self.tool == 'toggle':
            plot.tools.append(UnitToggleTool(plot))


class UnitActivateTool(BaseTool):

    # Fired when a unit is activated (double-clicked).
    activated = Event

    def normal_left_dclick(self, event):
        plot = self.component
        if plot and isinstance(plot, ImagePlot):
            ndx = plot.map_index((event.x, event.y))
            if ndx != (None, None):
                self.activated = ndx


class UnitInspectorTool(ImageInspectorTool):

    # Fired when the inspector becomes active or inactive.
    active = Event

    def normal_mouse_enter(self, event):
        self.active = True
        super(UnitInspectorTool, self).normal_mouse_enter(event)

    def normal_mouse_leave(self, event):
        self.active = False
        super(UnitInspectorTool, self).normal_mouse_leave(event)

    def normal_mouse_move(self, event):
        plot = self.component
        if plot and isinstance(plot, ImagePlot):
            ndx = plot.map_index((event.x, event.y))
            if ndx == (None, None):
                self.new_value = None
            else:
                x, y = ndx
                self.new_value = dict(indices = (x+1,y+1))
                self.last_mouse_position = (event.x, event.y)


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


if __name__ == '__main__':
    import numpy as np

    y = np.random.random((1, 10)) < 0.2
    x = np.random.random((5, 20)) < 0.5
    layers = [y, x]

    plot = UnitsPlot(layers=layers, tool='toggle')
    plot.configure_traits()
