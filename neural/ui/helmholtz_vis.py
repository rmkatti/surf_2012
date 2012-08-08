# System library imports
import numpy as np
from traits.api import HasTraits, Any, Array, Bool, Button, DelegatesTo, \
    Enum, Instance, on_trait_change
from traitsui.api import View, HGroup, VGroup, Item, Label, EnumEditor, \
    RangeEditor, TabularEditor, spring
from traitsui.tabular_adapter import TabularAdapter

# Local imports
from neural.helmholtz import HelmholtzMachine
from units_plot import UnitsPlot


class HelmholtzVis(HasTraits):

    machine = Any # Instance(HelmholtzMachine)
    layer_shapes = Array(dtype=int, shape=(None,2))
    pixel_size = DelegatesTo('plot')

    model = Enum('generative', 'recognition')
    clamp_top_units = Bool(False)

    plot = Instance(UnitsPlot)
    sample_button = Button(label='Sample!')

    def default_traits_view(self):
        model_editor = EnumEditor(values = {'generative':'Generative model',
                                            'recognition':'Recognition model'})
        shapes_editor = TabularEditor(adapter = LayerShapesAdapter(),
                                      operations = ['edit'])
        pixel_size_editor = RangeEditor(is_float = False,
                                        low = 1, high=128,
                                        mode = 'spinner')
        side_pane = VGroup(
            VGroup(HGroup(Label('Sample from:'),
                          Item('model',
                               editor = model_editor,
                               show_label = False)),
                   HGroup(Label('Clamp top-level units?'),
                          Item('clamp_top_units',
                               show_label = False),
                          enabled_when = "model == 'generative'"),
                   label = 'Sampling configuration',
                   show_border = True),
            VGroup(Label('Layer shapes:'),
                   Item('layer_shapes',
                        editor = shapes_editor,
                        show_label = False),
                   HGroup(Label('Pixel size:'),
                          Item('pixel_size',
                               editor = pixel_size_editor,
                               show_label = False)),
                   label = 'Display configuration',
                   show_border = True),
            spring,
            Item('sample_button'),
            show_labels = False),
        return View(
            HGroup(side_pane,
                   Item('plot',
                        show_label = False,
                        style = 'custom',
                        width = 0.75),
                   layout = 'split'),
            resizable = True,
            title = 'Helmholtz Machine')

    def _machine_changed(self):
        machine = self.machine
        self.plot.layers = []
        if machine:
            layer_topology = np.prod(self.layer_shapes, axis=1)
            if not np.all(layer_topology == machine.topology):
                self.layer_shapes = [ (1,size) for size in machine.topology ]
            self.plot.layers = map(np.zeros, self.layer_shapes)

    def _plot_default(self):
        return UnitsPlot(editable = True)

    @on_trait_change('sample_button')
    def sample(self):
        if self.model == 'generative':
            data = None
            if self.clamp_top_units:
                data = self.plot.layers[0].flatten()
            layers = self.machine.sample_generative_dist(all_layers=True, 
                                                         top_units=data)
        elif self.model == 'recognition':
            data = self.plot.layers[-1].flatten()
            layers = self.machine.sample_recognition_dist(data)
        self.plot.layers = map(np.reshape, layers, self.layer_shapes)

    @on_trait_change('layer_shapes')
    def _update_layer_shapes(self):
        layers = self.plot.layers
        if layers:
            self.plot.layers = map(np.reshape, layers, self.layer_shapes)


class LayerShapesAdapter(TabularAdapter):
    
    columns = ['Rows', 'Columns']
    alignment = 'right'

    def _set_text(self, value):
        value = int(value)
        machine = self.object.machine
        if machine:
            size = machine.topology[self.row]
            if size % value == 0:
                self.item[self.column_id] = value
                self.item[int(not(self.column_id))] = size / value
        else:
            self.item[self.column_id] = value
        self.object._update_layer_shapes()
